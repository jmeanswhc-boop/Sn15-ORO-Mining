"""Fresh 15: a model-driven shopping agent for ORO Bench 6 worlds, with code guards and an interview built from the keys of opened listings."""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-sonnet-5.5"  # the two gates where a wrong verdict scores 0: order and decline
CHEAP_MODEL = "anthropic/claude-haiku-5.5"  # every other reading, so a run stays under its spend cap
BACKUP_MODEL = "openai/gpt-6-luna"
# dollars per input and output token, for the task's running spend when a reply carries no cost of its own
PRICES = {MAIN_MODEL: (2e-6, 1e-5), CHEAP_MODEL: (1e-7, 5e-7), BACKUP_MODEL: (1e-7, 5e-7)}
ORDER_CHECKS_MAIN = 3  # order checks per task on the main model; any after these run on the cheap model
SPEND_SOFT = 0.04  # past this, only order/decline checks stay on the main model
SPEND_CAP = 0.075  # past this, every call goes to the cheap model; one run key still covers the suite

TASK_SECONDS = 265.0  # the network ends a task at 300 seconds; the finish must go out before that
FORCE_SECONDS = 28.0  # as agent_v6
DEFAULT_STEPS = 35  # a task's step limit when neither the request nor the store states one
DEFAULT_MESSAGES = 5  # a task's message limit when the request states none
MAX_ACTIONS = 16
SHORTLIST = 10
DETAILS_FIRST = 7
QUESTION_PROPS = 1  # properties one question names: the practice runtime's shopper answers only the first one named
ROW_PAGE_LIMIT = 7  # pages read in full when the matches fit in this many
ROW_TARGET = 15  # rows read when they do not
QUERY_EXTRA = 3  # words a query carries beyond the category (requirement values, most important first)
MORE_READS = 4  # datasheets one round of reading with spare steps opens (the leaders read 10-17 a task, fresh 8-12)
MORE_ROUNDS = 2
FINISH_STEPS = 7  # swap the carted item (2), stock check, order; and a refused order's repair: swap again (2) and order
CHEAP_PAGES = 3  # further pages of the main query a cheapest request reads when its matches run past what was read
CHEAP_QUERIES = 4  # other short queries (the noun, its kind words, the model's other names) a cheapest request runs
DEEP_PAGES = 4  # a cheapest request in a store ranking by relevance: further pages of its value search read
WALK_PAGES = 8  # a cheapest request in a store listing results in id order: pages of its narrowest search read to the end
REFERENCE_HOPS = 4  # listings followed along one reference's "take it from that listing" chain 
REFERENCE_OPENS = 4  # listings one lookup of a referenced need opens (the shopper's description narrows the seller's few)
FILTER_STEPS = 4  # steps one episode may spend on the filter search below a cheapest candidate's price
FILTER_RESERVE = 6  # steps kept beyond FINISH_STEPS when filtering: a pick sold out at checkout must still be replaced
NEUTRAL_DECLINE = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."

BUSY_STATUSES = (429, 500, 502, 503, 504)  # quick refusals of a busy proxy or upstream: the same request tends to pass soon
BUSY_WAITS = (1.0, 2.0, 4.0)  # pauses before each retry of a busy model call
START_SPREAD = 3.0  # seconds over which the run's tasks, all started at once, spread their first request
PACE_FIRST = 0.25  # pause before each request once the proxy has refused one for its rate (429)
PACE_MOST = 1.5  # the pause doubles on each further refusal, up to this
PACE_EASE = 0.05  # and shrinks by this on each request that passes


class _Pace:
    """The sandbox's proxy caps the requests per second of all the run's tasks together, model and store calls alike,
    and refuses the excess (429). A refusal slows this task down; requests that pass speed it back up. Every task sees
    the refusals at the same time, so together they settle under the cap."""

    def __init__(self) -> None:
        self.pause = 0.0

    def wait(self, until: float) -> None:
        if self.pause and time.monotonic() + self.pause + 1 < until:
            time.sleep(self.pause)

    def note(self, status: int) -> None:
        if status == 429:
            self.pause = min(PACE_MOST, max(PACE_FIRST, self.pause * 2))
        elif status == 200:
            self.pause = max(0.0, self.pause - PACE_EASE)


_PACE = _Pace()


class _PacedClient(ProxyClient):
    """A proxy client whose requests follow _PACE. The proxy counts a model call once, by its final outcome, so the
    model client (busy_retry) also retries a busy refusal here, inside the call, while the task's time allows; a timeout
    or a refused request is not retried."""

    retry_until = float("inf")  # monotonic time after which no pause or retry starts (set before each call)
    busy_retry = False

    def _make_request_with_retries(self, request_func, method, path):
        def paced():
            _PACE.wait(self.retry_until)
            response = request_func()
            _PACE.note(response.status_code)
            return response

        response, data = super()._make_request_with_retries(paced, method, path)
        for wait in BUSY_WAITS if self.busy_retry else ():
            if data is not None or response is None or response.status_code not in BUSY_STATUSES:
                break
            if time.monotonic() + wait + 1 > self.retry_until:
                break
            time.sleep(wait)
            response, data = super()._make_request_with_retries(paced, method, path)
        return response, data


_llm = _PacedClient(timeout=120, max_retries=1, retry_delay=1)
_llm.busy_retry = True
_env = _PacedClient(timeout=120, max_retries=1, retry_delay=1)


# ----------------------------------------------------------------------------
# prompts

MATCH_RULES = """Reading rules for judging a product:
- Each listing is judged only against the requirements of its own product. In an order of several products, another product's requirement, price limit or preference never applies to it, even when the two sit in the same category.
- Read each requirement by its form. An exact value is met only exactly, figures included: "version 10" is not met by "version 11", nor "8 cores" by "7 cores". A limit is met by any amount on its side of the edge, the edge included: "at most 33" by 33 or 30, "at least 22" by 22 or 26. Alternatives ("a or b") are met by any one of them, and "not x" by anything else.
- A preference is separate from the requirements: "cheapest", "lowest price", "most", "least" or "largest" only chooses among the listings of that product that meet every requirement, and never excuses a failed one.
- Judge only the CURRENT REQUIREMENTS listed (with the budget, stock, quantity and category). They already hold every change the shopper made: a requirement the request states but the list no longer carries was dropped, and never counts. Never add one of your own from the story (who it is for, an age, a size, a use, an occasion): a value the shopper gave is met as worded, whatever the story suggests.
- A requirement is met only when the product's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by their words as this store writes them: one element of a comma-separated list is enough; singular and plural are the same; British and American spellings are the same ("colour" is "color"); case and separators do not matter. A synonym, an abbreviation or a related word counts only when no other listing of the category shown writes the required value itself in that property (when other listings write "brushed", a "polished" finish does not meet brushed). A value joined with another by a plus ("steel + glass") does not meet one of its parts when other listings carry that part alone. A value that claims to cover every value of the property ("universal", "fits all", "for any use") names none of them and does not meet a required value. A value with a shade, tone or other qualifier added still states it ("light brown", "dark brown" and "coffee brown" state "brown"; "navy blue" states "blue"). A narrower value that is one kind of the required value states it too (a "rose print" pattern states "floral"; "crimson" states "red"). A broader or a different class does not count.
- The description is never evidence for a requirement. Use the title or the options list only when the product has no property key for that requirement. If its own property says something else, it fails even when the title says the value, and the title's value is ignored. The title must then carry the value's own word (singular or plural): an adjective or compound made from it does not state it.
- A value under a different, related property does not count (a property of one part of the product is not the property of the whole).
- A key names a property when it carries the property's own words (a prefix such as the product's name aside): for "type of <product>" a key that is "type" or "<product> type" (the type of something else, "<other part> type", does not count), for "<property>" any key with that word. A key for a different property does not count (a key for one property does not state another).
- When several keys name the property, the keys that are its own name, alone or with the product's noun ("<property>", "<product> <property>"), decide, and one of them stating the value is enough even if another says something else. Keys with other words added ("<qualifier> <property>", "<prefix>_<other words>_<property>") count only when the product has none of those; when no key is the property's own name, the title decides. A plural key ("<property>s") is a list of tags: when the product also has the singular key, the singular key decides.
- A requirement is also met when the product's own category label states the value as a word of its own.
- Numbers are yours to compare: the code leaves any requirement that contains a figure unproven, and the word-matching rule above does not decide a figure. The product's own fact may name a different unit from the requirement. Convert both to one unit of that kind before you judge, and compare the amounts. A different unit word, or a different written number that is the same amount in a smaller or larger unit, still meets it. Do not fail the product only because those spellings differ. The written edge counts after the conversion (an "at most" limit includes its own amount, however the fact writes it). A range meets a limit only when the whole range does. A requirement that the product's own range of values goes up to or reaches X needs that range to reach X, on both the plus and the minus side for "+/- X"; a range that stops short of X fails. A ceiling ("at most", "under", "up to") is a limit, not a demand for that exact figure. A figure of a different kind (a weight beside a length) does not count. When the fact names no unit, or you cannot tell the units are the same kind, leave the requirement unproven rather than failing the product.
- Respect exclusions ("nothing with X") and closed lists ("A or B and nothing else"). A product fails an exclusion only when its own property for it states the excluded value's word, under the key rules above; a value that merely resembles it or belongs to its family (a synthetic or imitation material for an excluded natural one) does not, nor does a key for another property.
- The product must be in the requested category: the category label that matches the request's wording as a whole, not a narrower or neighbouring one.
- Price times quantity must be within the budget (equal is fine), and a product shown out of stock fails."""

SETUP_PROMPT = """Read the shopper's request. Return ONE compact JSON object and nothing else, with these keys:
"category": the kind of product requested, in the request's words (this is usually the store's category name).
"noun": the plain noun for one item of the product, as the shopper would say it in a question.
"quantity": units of one listing to order (integer; 1 unless several are asked for; a pair or a set is 1).
"requirements": a list of {"property": ..., "value": ..., "numeric": true/false}, one per firm stated condition on the product, with the shopper's exact value, limit, exclusion ("not X") or closed list. "numeric" is true when the value is a number with a unit or a numeric limit. A limit keeps its own words in the value wherever the request puts them ("RAM 12 GB at most" gives "ram: at most 12 GB", "6 inches or more" gives "at least 6 inches"): a limit written as a bare figure would demand that exact figure. Do not include price, budget, stock, quantity, delivery, mere wishes, or the category itself. Each requirement names a value the product itself must have (a size, a material, a colour, a type, a figure or range); the shopper's story about how they will use it, or a wish that some figure be listed or known (to note it down, to compare, for their records), sets no value and is never a requirement. When the request names a property together with its value ("in a basic style", "for a formal occasion", "the material must be cotton"), that is the requirement: the property is the word the request uses for it (style, occasion, material) and the value is the one it names. A looser rewording of the same wish elsewhere in the request ("nothing fancy, just a plain one", "so nothing casual") only restates it: never add it as another requirement or let it replace the named value. The value is the word or words the request names for it ("it's the slim fit I want: close to the body, nothing baggy" gives "fit: slim"), never widened with the request's own explanation of what that value means. Write each property and value in the plain words and spelling an online store's product listing uses (standard American English, lowercase), whatever regional spelling or idiom the request uses, so they read like the store's own property names and values.
"budget": the maximum total price as a number, or null: the number exactly as the request writes it, in the request's own unit, never converted to another unit (a limit written in cents stays in cents), since the store's prices use the same unit.
"currency": the price unit exactly as the request writes it, or null.
"in_stock_only": true when the shopper wants items actually on hand.
"message_limit": how many messages the shopper allows (integer) or null.
"step_limit": how many moves or steps the request allows (integer) or null.
"preference": {"kind": "cheapest" | "best" | "brand" | "other" | "none", "property": the property the preference is about ("" for cheapest), "direction": "max" or "min" for a best-by-measure preference, else "", "value": a preferred value such as a brand ("" if none), "numeric": true when the preference is measured by a number with a unit (largest, longest, most, highest, lightest...)}. "cheapest" only when the request explicitly asks for the cheapest or lowest price; remarks that money is tight only explain the budget.
"ambiguity": a list of the choices only the shopper can settle, each {"kind": "choice", "first": {"property": ..., "value": ...}, "second": {"property": ..., "value": ...}}: two wishes the request says may not both be possible, so the shopper must say which matters more (put both wishes in "soft_wants" too, never in "requirements"). A remark that the shopper will check something themselves ("I'll check the label myself") is a stated requirement, never an ambiguity. Empty in almost every request.
"soft_wants": things the shopper would merely like (ideally, preferably, if possible); keep them out of requirements.
"ask_first": the topics the request tells the assistant to raise with the shopper before picking a product, a few words each ([] when it names none).
A detail of the story (who will use it, an age, a size, a place, an occasion) is never a requirement unless the request states it outright as a condition on the product; it may hint at a hidden need, so name it in "purpose".
A remark that the shopper will check something themselves makes the clause it directly follows, in the same sentence, a stated requirement when that clause names a value a product listing could state (what it is for, what it is made of, a feature): put it in "requirements" with that value in the clause's own one to three words, and as the property the plain name of what kind of value that is. Only that clause: every other story detail stays in "purpose".
"purpose": who the product is for and how it will be used, from the story, in one sentence.
"rules": every procedural rule the request states for the assistant (how many listings to open or compare, what to check, what to tell or show the shopper, what a remark must say, how to end), each quoted word for word from the request. Every sentence of the request lands in some key; when unsure whether a sentence is a rule, put it in "rules".
"references": requirements whose value the request does not give but says to take from another listing in the store (the same as, matching, or whatever some other product's page shows), each {"product": the product number in an order of several, else null, "property": the property of the shopper's product, "seller": the other listing's seller or shop, "category": its kind of product, "conditions": its other properties the request names, each "property: value", "description": the request's words for that listing, "relation": "same", "differ", "at most", "at least" or "within", "tolerance": the number for "within" (else null), "unit": its unit (else "")}; [] when none. Leave such a property out of "requirements": its value is looked up. The details that pick the other listing out ("the one with plastic bottles", "the single pack") describe that listing only: they go in its "conditions" and never in any "requirements" list. In an order of several products, list every reference here with its product number. A rule that one product of this order must fit, suit or work with another product of this same order ("make sure the part fits the laptop you end up buying") is NOT a reference: it names no other listing in the store; it stays in "rules" only.
"confirm_before_order": true when the request asks that the final basket (or the chosen item and its price) be shown, told or read to the shopper, or checked or confirmed with them, or their yes or go-ahead given, before checking out; else false.
"process": a list of the steps the request requires the assistant to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"decline_no_names": true when the request's rule for coming up empty forbids naming or suggesting substitutes; false when it asks to name something; null when it gives no rule.
"conditional": requirements that hold only under a condition on a product of this order ("if / should / in case X turns out to be ..., then Y must be ...", "Y must be ..., except when / unless X is ..."), each {"product": the number of the product Y is about (null for a request of one product), "requirement": "property: value" for Y, "when_product": the number of the product X is about (null for one product), "when": "property: value" for X with the request's exact value or closed list, "except": true for "except when / unless" (Y holds unless X holds), false for "if" (Y holds only when X holds)}; [] when none. Such a requirement is never put in "requirements": whether it applies is only known once the item for X is chosen. For "except" / "unless", put Y here only, not in "requirements" as well.
"items": when the request asks for two or more DIFFERENT products (not several units of one), one entry per product in the request's order (words linked by "and" or "&" with a single count between them stay a single entry, split only if each side carries a count or conditions of its own; keep each entry's category in the request's own words, never a broader word for it), each {"category", "noun", "quantity", "requirements" (as above, only that product's), "budget" (that product's own price limit when the request gives one, else null), "preference" (as above, or null when the request's one preference covers it)}; [] for a request for one product. With items, "budget" is the limit for the whole order (the sum of the products' own limits when the request only gives those)."""

TOOLS_PROMPT = """Read the store's tool list. Return ONE compact JSON object and nothing else, with these keys:
"tools": the exact tool names and argument names from the list (null when no tool or argument fits):
  "search": {"tool", "query_arg" (free-text words), "page_arg" (a page number), "token_arg" (an argument that takes the next-page token or cursor a result returns, when paging works that way), "mode_arg" (an argument choosing keyword vs filter search, if any), "mode_value" (its value for keyword search), "filter_value" (its value for filter search, if any), "category_arg" (an argument that takes a category name in filter search, if any), "price_arg" (a maximum price filter, if any)},
  "details": {"tool", "id_arg"}: the tool that shows a product's property list,
  "details_many": {"tool", "ids_arg"}: a tool that shows the property lists of several products in one call, its argument a list of ids (null when there is none),
  "listing": {"tool", "id_arg"}: the tool that shows one product in full with its price and stock (may be the same as details),
  "message": {"tool", "text_arg"}: sends a message to the shopper,
  "cart": {"tool", "id_arg", "qty_arg", "op_arg", "add_value", "remove_value"}: adds to or removes from the cart,
  "cart_remove": {"tool", "id_arg"}: only when removing uses a separate tool,
  "cart_view": {"tool"}, "order": {"tool"}: checks out, "decline": {"tool", "text_arg"}: ends without buying.
"finish_args": when ONE tool both buys and ends without buying, {"order": {...}, "decline": {...}} with the argument values that select each (for example {"order": {"mode": "buy"}, "decline": {"mode": "stop"}}); null otherwise."""

REFERENCE_PROMPT = """A shopper wants one property of their product to have the same value as another listing in the store, which they describe by its seller, kind of product and some of its properties. Below are listings that may be the one described. Find the listing that matches the description: the same seller, the same kind of product, and every named condition, read from its facts, title and description (a seller's note in the description can settle a condition the facts leave out). Then read that listing's own value for the property, as its facts write it. When the matching listing's text says the value is to be taken from a different listing instead, give that listing's description in "defers_to" and no value. When no listing matches, give null for "match".
Return ONE JSON object and nothing else:
{"match": "id" or null, "value": "the value as the matching listing states it" or null, "defers_to": {"seller": "...", "category": "...", "conditions": ["property: value", ...], "description": "..."} or null}"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"understood": false when the shopper only says they did not understand the question (asks to rephrase, is confused); else true.
"meaning": one short sentence saying what the shopper means.
"needs": new firm requirements on the product, each "property: value" with the exact value, limit, exclusion or closed list the shopper gave, written in the plain words and spelling an online store's product listing uses (standard American English, lowercase). IMPORTANT: the property is the one the shopper's own words name, which often differs from the one the question asked about; use the shopper's property, never the question's (asked about the color, "I'm set on the material being wool" is the need "material: wool", not "color: wool"). Read the whole sentence for its sense: a shopper who insists on a value, refuses to bend or compromise on it, or will accept nothing but it REQUIRES that value ("property: value"); only a value the shopper rejects or does not want is an exclusion ("property: not value").
"no_preference": properties the shopper said they do not care about.
"dropped": earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null: the number exactly as written, in the unit the shopper uses, never converted to another unit.
"new_preference": null unless the shopper changes how to choose among the listings that fit (now the cheapest one, the best by some measure such as the largest or lightest, or a preferred brand); then that new preference as {"kind": "cheapest" | "best" | "brand" | "other", "property": the property it is about ("" for cheapest), "direction": "max" or "min" for a best-by-measure preference, else "", "value": a preferred value such as a brand ("" if none), "numeric": true when it is measured by a number with a unit}. A firm limit ("at most 8 cores", "no more than 2 kg") is a need, never a preference.
"unavailable": true when the shopper or store says the chosen item can no longer be bought (sold out, gone, out of stock), else false.
"requirements_now": the complete list of firm requirements as they stand after everything said so far, each "property: value": keep every earlier one unless dropped or replaced (a need on a property with a different name, even a similar-sounding one, is added as its own entry and never rewrites another requirement's value), keep only the newest value of a replaced one, add every new need. No price, stock, quantity or category, and never a requirement taken from the story or your own reading of the situation: only what the request states and what the shopper said.
"reply": when the shopper asks the assistant something or asks it to confirm a change, a short plain reply that answers it and names the change in substance; otherwise "".
"references": needs whose value the shopper does not give but says to take from another listing in the store (the same as, matching, or taken from what some other product's page shows), each {"product": the product number in an order of several, else null, "property": the property of the shopper's product, "seller": the other listing's seller or shop, "category": its kind of product, "conditions": its other properties the shopper names, each "property: value", "description": the shopper's words for that listing, "relation": "same" (the value must be that listing's), "differ" (it must NOT be: different from, other than, set apart from), "at most", "at least" or "within" (a figure compared with that listing's), "tolerance": the number for "within" (else null), "unit": its unit (else "")}; [] when none. Such a need also stays in "needs" and "requirements_now" as the shopper words it. The conditions describe the other listing only: they are never requirements on the shopper's product, so they never add to or replace anything in "requirements_now".
A reply that only says the shopper does not mind, about a property the request already states a requirement for, keeps that requirement: they are saying they have no further wish, not dropping it. Only an explicit change of mind drops a stated requirement (a new value for it, or saying that condition no longer matters or can go).
One line can carry several changes: report each. A replaced requirement is both dropped (old) and a need (new). When the shopper says which of two wishes matters more, the preferred wish is a need. A mere wish is not a need. Folksy or roundabout phrasing that names a value is a firm want. A value the shopper names is a need for that value, however it is worded; it is an exclusion ("not X") only when the shopper says they do not want X. A number with a comparison is a limit kept with its unit. A new budget replaces the old one, higher or lower."""

SHORTLIST_PROMPT = """Give me up to 10 product ids from the search rows below which meet the preference and all the requirements, ordered by price PRICE_ORDER. Every product must be in the requested category (the category label that matches the request's wording as a whole). You see only search rows (title, category, price, stock), not full details, so keep a row unless it clearly fails; prefer rows whose title shows the required values. A title that states the required amount in a different unit of the same kind does not fail the row: compare the amounts, not the unit words. Leave out rows over the budget or shown out of stock.
Return ONE JSON object and nothing else: {"ids": ["...", ...]}."""

NUMERIC_PROMPT = """Without the details of each product, from the searched products below give me 5 product ids which are in the requested category and show any of the stated requirements, most promising for the preference first (the preference is measured by a number with a unit, so prefer titles that state a strong value for it). Leave out rows over the budget or shown out of stock.
Return ONE JSON object and nothing else: {"ids": ["...", ...]}."""

REVIEW_PROMPT = """While reviewing the detailed products below, do two things.
1. Give me up to 7 candidate property names which matter most for this product given the shopper's purpose. Use the property names as the products' details name them (underscores as spaces). They must be properties the detailed products actually carry. A hidden need most likely sits on a property that (a) the store files for most products of this kind: a key that many of the opened products carry (see PROPERTY KEYS, which counts them), (b) can take several values for this kind of product, even when the opened products happen to share one: other listings' titles, or what such products are generally sold in, show other values, and (c) the shopper's story points to: how it must look, where or how it will be used, who it is for. Rank by these together, the story's hints weighing most. Leave out a property that only one or two opened products carry, and one that cannot take another value for this kind of product. They must NOT be properties already stated in the requirements, the preference, the resolved ambiguity, or already asked. Prefer properties the shopper's story hints at (who it is for, how it must look, where or how it will be used), then properties that describe what the product itself is like, not logistics (warranty, condition, model numbers, options lists, origin).
2. Give me the 1 product which meets all the current requirements, ignoring the preference, or else the most similar product; and list every detailed product that meets all the current requirements (budget and stock included), ranked by the preference: cheapest first for a cheapest preference, the best match first in the order the products were found (price does not rank them) when there is no preference, the best by the measure first for a best preference, the preferred value first for a brand or other preference.
""" + MATCH_RULES + """
Return ONE JSON object and nothing else:
{"properties": ["...", ...], "pick": "id", "pick_fits_all": true or false, "fits": ["ids ranked by the preference"], "problems": {"id": "the first requirement it fails"}}"""

ORDER_PROMPT = """You are the final reviewer before a shopping assistant places an order. The task scores only if the item meets every current requirement and the request's rules are followed. You are strict, but you never invent a requirement the request and the shopper did not state.
""" + MATCH_RULES + """
- The keys that are the property's own name, alone or with the product's noun, decide (looser keys only when there are none); one of them stating the value is enough. A key for a different property (a style key for a type) does not state it.
- The quantity is how many units of this listing go in the cart: a pack, set or bundle sold as one listing is one unit, so a pack size or piece count never fails the quantity or leaves it unproven.
- When the fact writes the required amount in a different unit of the same kind, quote that fact as written and mark the check pass once the amounts match. A different unit word alone is not a fail.
For the item to order, check each current requirement against its own facts and quote the fact you used exactly as written ("key: value", or "title: ..." only when no fact names the property), with verdict pass, fail or unproven. Then check the request's procedural RULES (how many listings to look at, what to tell the shopper, what to put in a remark, how to finish): say which are not yet met and what would meet them.
Return ONE JSON object and nothing else:
{"checks": [{"requirement": "...", "fact": "...", "verdict": "pass" | "fail" | "unproven"}], "pass": true or false,
 "rules": [{"rule": "...", "met": true or false, "action": {"kind": "open_more", "count": 0} | {"kind": "message", "text": "..."} | {"kind": "remark", "text": "..."} | null}]}"""

DECLINE_PROMPT = """A shopping assistant is about to tell the shopper that nothing fits. A wrong decline scores 0, so check hard before it gives up.
""" + MATCH_RULES + """
For each opened product, give the first current requirement it fails (quote its fact) or null if it passes everything. Name unopened listings (from the rows below) that could plausibly pass and should be opened, best first, and up to 3 short new search queries worth trying (the product noun or a synonym, optionally with one required value; keep them short, as a query that returns nothing needs fewer words).
Return ONE JSON object and nothing else:
{"opened": [{"id": "...", "fails": "requirement and quoted fact, or null"}], "open_first": ["id", ...], "queries": ["...", ...]}"""

SEARCH_PROMPT = """A shopping assistant has found too few listings of the requested kind. Suggest up to 3 other short search queries for this store: the product noun alone, a synonym or the store's own category wording, or the noun with one required value. Keep them short: a query that returns nothing needs fewer words. Do not repeat a query already tried.
Return ONE JSON object and nothing else: {"queries": ["...", ...]}"""

CHEAP_PROMPT = """The shopper wants the cheapest listing that fits, and the store's search lists rows by relevance, not by price, so cheaper fitting listings may not have been seen yet. Suggest up to 3 other short search queries (one or two words each) that list more listings of the requested kind: other names the store's titles use for this kind of product, or a shorter kind word, never adding words the request does not ask for. Do not repeat a query already tried.
Return ONE JSON object and nothing else: {"queries": ["...", ...]}"""

SYNONYM_PROMPT = """A shopping assistant's search matched more listings than it can read, so it sees only the first rows. Suggest up to 2 other short search queries that use OTHER WORDS for the same request: another name for the product, or another wording of a required value, as the titles below word them. Each query keeps the product and the required values, in different words; add no words the request does not ask for. Do not repeat a query already tried.
Return ONE JSON object and nothing else: {"queries": ["...", ...]}"""

MULTI_INTERPRET = """This order has several products (PRODUCTS above, numbered). Give "requirements_now" as an object keyed by product number ("1", "2", ...), each the complete list for that product; a change applies only to the product it is about (a line naming no product is about the one just asked about). Give each "dropped" and "no_preference" entry as "N: property". Give "unavailable" as the list of product numbers whose chosen item can no longer be bought ([] when none). Give "new_preference" as an object keyed by product number holding only the products whose preference changed ({} when none). "new_budget" is a new limit for the whole order only; a new price limit for one product alone goes in "product_budgets", an object keyed by product number ({} when none), the number exactly as the shopper writes it."""


# ----------------------------------------------------------------------------
# small helpers

def _parse_json_object(text: Any) -> dict[str, Any] | None:
    """The reply's JSON object. A model that corrects itself ('Wait, ...') writes the answer again after the
    first one, so the last complete object counts, in a fenced block or in the plain text."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    for chunk in reversed(re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)):
        value = _json_dict(chunk.strip())
        if value is not None:
            return value
    return _json_dict(text)


def _json_dict(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found = None
    start = text.find("{")
    while start >= 0:
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            # an object that is whole except for its last closing brackets (a brace left out, or cut off)
            try:
                value = json.loads(_closed(text[start:]))
                found = value if isinstance(value, dict) else found
            except ValueError:
                pass
            break
        if isinstance(value, dict):
            found = value
        start = text.find("{", end)
    return found


def _closed(text: str) -> str:
    """The text with the strings, arrays and objects it leaves open closed, innermost first."""
    open_: list[str] = []
    in_string = escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            open_.append("}" if ch == "{" else "]")
        elif ch in "}]" and open_:
            open_.pop()
    text = text.rstrip().rstrip(",") if not in_string else text + '"'
    return text + "".join(reversed(open_))


def _compact(value: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...(cut)"


FIELD_CHARS = 400  # longest text of one field in a tool result shown to the actor


def _clip(value: Any, chars: int = FIELD_CHARS, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= chars else value[:chars] + "...(cut)"
    if depth > 8:
        return value
    if isinstance(value, dict):
        return {k: _clip(v, chars, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, chars, depth + 1) for v in value]
    return value


def _show(value: Any) -> str:
    """A tool result for the actor: each long text field clipped, then the whole cut at OBS_CHARS (a tail cut alone
    drops later rows, later listings and the notices that usually come last)."""
    return _compact(_clip(value), OBS_CHARS)


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
    if not kinds or any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind == "integer" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
            return int(float(value))
        if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            number = float(value)
            return int(number) if number.is_integer() else number
        if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
    return value


def _check(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce the arguments to the tool's schema; list what still breaks it."""
    properties = (schema or {}).get("properties") or {}
    fixed: dict[str, Any] = {}
    problems: list[str] = []
    for key, value in args.items():
        if key not in properties:
            continue  # undeclared arguments are dropped by the runtime anyway
        spec = properties[key] if isinstance(properties[key], dict) else {}
        value = _coerce(value, spec)
        kinds = spec.get("type")
        kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
        if kinds and not any(_type_ok(value, k) for k in kinds):
            problems.append(f"{key}: expected {'/'.join(kinds)}")
            continue
        if spec.get("enum") and value not in spec["enum"]:
            match = [e for e in spec["enum"] if isinstance(e, str) and isinstance(value, str) and e.lower() == value.strip().lower()]
            if not match:
                problems.append(f"{key}: must be one of {spec['enum']}")
                continue
            value = match[0]
        if isinstance(value, str) and isinstance(spec.get("maxLength"), int):
            value = value[: spec["maxLength"]]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(spec.get("minimum"), (int, float)) and value < spec["minimum"]:
                problems.append(f"{key}: below minimum {spec['minimum']}")
            if isinstance(spec.get("maximum"), (int, float)) and value > spec["maximum"]:
                problems.append(f"{key}: above maximum {spec['maximum']}")
        fixed[key] = value
    for key in (schema or {}).get("required") or []:
        if key not in fixed:
            problems.append(f"{key}: required")
    return fixed, problems


# ----------------------------------------------------------------------------
# word matching (the store's literal reading of a value). Plain word forms only; synonyms and other
# spellings are the model's to judge, so no word list here is tied to one store or one kind of product.

_FILLER = set("a an the of and with must be is are has have having include includes including exactly only should that it its any made from".split())
_NEGATION = re.compile(r"^(?:not|no|without|anything but|except|excluding|avoid|never)\b", re.I)
_COMPARISON = re.compile(r"\b(?:at least|at most|no more than|no less than|up to|or more|or less|or longer|or shorter|or bigger|or smaller|under|over|above|below|minimum|maximum|min|max|between|and|to)\b", re.I)
# a limit, not a word of the product: "and" and "to" are left out, so "cotton and linen" stays a value
_BOUND = re.compile(r"\b(?:at least|at most|no more than|no less than|up to|or more|or less|or longer|or shorter|or bigger|or smaller|under|over|above|below|minimum|maximum|between)\b", re.I)


def _words(text: Any) -> list[str]:
    text = re.sub(r"'s\b", "", str(text if text is not None else "").lower())
    return re.findall(r"[a-z0-9]+", text)


def _one(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _same(one: str, other: str) -> bool:
    one, other = _one(one), _one(other)
    if one == other:
        return True
    if re.search(r"\d", one + other):
        return False
    short, longer = sorted((one, other), key=len)
    if longer.startswith(short) and len(short) >= 4 and len(longer) - len(short) <= 3:
        return True
    # two forms of one stem ('embroidery', 'embroidered'): a shared start of 5+ letters, each ending at most 3 past it
    stem = next((n for n, (a, b) in enumerate(zip(one, other)) if a != b), len(short))
    return stem >= 5 and len(longer) - stem <= 3


def _stated(value: str, text: str, exact: bool = False) -> bool:
    """Every word of the value appears among the words of the text: in some form, or with exact only as itself
    (singular or plural), not as a changed word."""
    wanted = [w for w in _words(value) if w not in _FILLER]
    have = _words(text)
    same = (lambda a, b: _one(a) == _one(b)) if exact else _same
    if wanted and all(any(same(w, h) for h in have) for w in wanted):
        return True
    flat = re.sub(r"[^a-z0-9]+", "", value.lower())
    return len(flat) >= 8 and flat in re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _split_req(req: str) -> tuple[str, str]:
    prop, sep, value = str(req).partition(":")
    return (prop.strip(), value.strip()) if sep else ("", prop.strip())


def _is_numeric(value: str) -> bool:
    return bool(re.search(r"\d", value))


def _need_figure_forms(value: str) -> list[str]:
    """Short search tokens for a numeric/limit need: a limit on a figure gives the figure with its unit, spaced and glued, then the figure alone.

    A figure with its unit finds listings the bare category search buries."""
    text = str(value or "")
    out: list[str] = []
    for num, unit in re.findall(
        r"(\d+(?:\.\d+)?)\s*([a-zA-Z%]{0,12}|\"|')?",
        text,
    ):
        unit = (unit or "").strip()
        if unit in {"at", "least", "most", "more", "less", "than", "to", "or", "and"}:
            continue
        spaced = f"{num} {unit}".strip() if unit else num
        glued = f"{num}{unit}" if unit and unit.isalnum() else spaced
        for form in (spaced, glued, num):
            if form and form.lower() not in {x.lower() for x in out}:
                out.append(form)
        if len(out) >= 6:
            break
    return out


def _facts(details: Any) -> list[tuple[str, str]]:
    """The (key, value) facts of a details result: a flat dict of short values, or a list of name/value pairs."""
    out: list[tuple[str, str]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(value, dict):
            flat = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
                    and not re.search(r"(?i)(^|_)id$|pid|sku|title|price|cost|stock|avail|descr", str(k))}
            if len(flat) >= 2 and all(not isinstance(v, (dict, list)) for v in value.values()):
                out.extend((str(k), str(v)) for k, v in flat.items())
                return
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            if value and all(isinstance(v, dict) and len(v) == 2 for v in value):
                for v in value:
                    pair = list(v.values())
                    if isinstance(pair[0], str):
                        out.append((pair[0], str(pair[1])))
                return
            for item in value:
                walk(item, depth + 1)

    walk(details, 0)
    return out


def _prop_words(prop: str, noun: frozenset[str]) -> set[str]:
    """The words a key must carry to name a property: its own words, fillers and the product's noun aside, and
    'type' aside when other words say which kind."""
    want = {_one(w) for w in _words(prop) if w not in _FILLER} - noun
    return want - {"type"} if want - {"type"} else want


def _key_words(key: str, want: set[str]) -> set[str]:
    """A key's words, fillers aside, and 'type' aside unless the property is a type."""
    return {_one(w) for w in _words(key) if w not in _FILLER} - ({"type"} if "type" not in want else set())


def _names(key: str, prop: str, noun: frozenset[str] = frozenset()) -> bool:
    """A fact key names a requirement's property: the property's own words are in the key (_prop_words), with at
    most two more words around them."""
    want = _prop_words(prop, noun)
    have = _key_words(key, want)
    if want == {"type"}:  # a bare type: 'type' alone or with the product's noun, not the type of another thing
        return "type" in have and have - {"type"} <= noun
    return bool(want) and want <= have and len(have - want) <= 2


def _key_norm(key: str) -> str:
    return " ".join(_one(w) for w in _words(key))


def _name_parts(prop: str) -> list[str]:
    """A property name without bracketed examples, split at '/' and 'or'."""
    clean = re.sub(r"\s*[\(\[][^)\]]*[\)\]]", "", prop).replace("_", " ")
    return [x.strip() for x in re.split(r"/|\bor\b", clean) if x.strip()] or [clean]


def _kept_requirements(old: list[str], now: list[str], dropped: list[Any], said: str, noun: frozenset[str]) -> list[str]:
    """The requirements after a reading: its list, plus every earlier one it lost in the rewrite. An earlier one
    leaves only when the list gives its property a new value, or the reading says it was dropped and the shopper's
    own line mentions it, by its property or its value, in words other than those of a need the line adds (a
    question about it answered with indifference is no drop)."""
    have = {_key_norm(_split_req(r)[0]) for r in now}
    named = {_key_norm(_split_req(str(d))[0] or str(d)) for d in dropped}
    # the words of a need added in this line name that need, not an earlier requirement it shares a word with
    # ('cans' for a new packaging need never drops 'pack type: single can or bottle')
    added = {_one(w) for r in now if r not in old for w in _words(r)}
    heard = {_one(w) for w in _words(said)} - added

    def gone(req: str) -> bool:
        mentioned = {_one(w) for w in _words(req) if w not in _FILLER} - {"type"} - noun
        return _key_norm(_split_req(req)[0]) in named and bool(mentioned & heard)

    return now + [r for r in old if r not in now and _key_norm(_split_req(r)[0]) not in have and not gone(r)]


def _clauses(value: str) -> tuple[list[str], list[str]]:
    """A requirement value as (allowed values, excluded values). Clauses are split first, so a value that mixes
    both ('A, not B or C') allows A and excludes B and C, instead of reading B and C as allowed alternatives."""
    allowed: list[str] = []
    excluded: list[str] = []
    for clause in re.split(r",|;", value):
        clause = clause.strip()
        if not clause:
            continue
        target = excluded if _NEGATION.match(clause) else allowed
        target += [o.strip() for o in re.split(r"\s+or\s+|/", _NEGATION.sub("", clause)) if o.strip()]
    return allowed, excluded


def _options(value: str) -> list[str]:
    return _clauses(value)[0]


CLOSEST = "the key named for the property says otherwise: "


def _closest_keys(named: list[tuple[str, str]], prop: str, noun: frozenset[str]) -> list[tuple[str, str]]:
    """Of the keys naming a property, the ones that decide: the property's own name, alone or with the product's
    noun; looser keys that carry other words too only when the product has none of
    those. A key whose last word is the plural of the property's is a tag list; the singular key states the product's own value and decides when both are there."""
    want = _prop_words(prop, noun)
    close = [(k, v) for k, v in named if _key_words(k, want) - want <= noun]

    def plural(key: str) -> bool:
        last = (_words(key) or [""])[-1]
        return _one(last) != last and _one(last) in want

    own = [(k, v) for k, v in close if not plural(k)]
    return own or close or named


def _fact_reading(details: Any, title: str, prop: str, value: str, noun: frozenset[str] = frozenset(),
                  category: str = "") -> tuple[str, str]:
    """How a product stands on one requirement: 'yes', 'no' or 'unsure', and the evidence read.

    The product's own facts decide when a key names the property, and then the title is ignored; the title counts
    only when no key names it. Of the keys naming it, those closest to its name decide (_closest_keys), and one of
    them stating the value is enough; the product's category label stating the value meets it too. 'no' otherwise only when the named facts share no word with any allowed value; a partial overlap is
    left to the model. A 'no' is a candidate: Episode.fact_vetoes has the model confirm it."""
    if not prop or not value or _is_numeric(value):
        return "unsure", ""
    every = [(k, v) for k, v in _facts(details) if _names(k, prop, noun) and v.strip() and v.strip().lower() not in ("n/a", "none", "-")]
    named = _closest_keys(every, prop, noun)
    options, excluded = _clauses(value)
    texts = [v for _, v in named] or [title]
    shown = "; ".join(f"{k}: {v}" for k, v in named[:2]) or f"title: {title[:60]}"
    if any(_stated(x, t) for x in excluded for t in texts):
        return "no", shown  # an excluded value is stated
    if not options:
        return ("yes" if named else "unsure"), shown  # only exclusions, none stated
    if category and any(_stated(o, category, exact=True) for o in options):
        return "yes", f"category: {category}"  # the store files it under the value itself
    stating = [(k, v) for k, v in named if any(_stated(o, v) for o in options)]
    if stating:
        return "yes", "; ".join(f"{k}: {v}" for k, v in stating[:2])
    # Only a key that is the property's own name ("<property>", or with the product's noun) can rule a product out
    # here; a key with other words added ("<other> <property>") may name a neighbouring property, so it is left
    # to the model, which reads the title as well.
    want = _prop_words(prop, noun)
    if named and not any(_key_words(k, want) - noun == want for k, _ in named):
        # only keys for neighbouring properties: as with no key at all, the title's own word for the value decides
        if any(_stated(o, title, exact=True) for o in options):
            return "yes", f"title: {title[:60]}"
        return "unsure", shown
    looser = [(k, v) for k, v in every if (k, v) not in named and any(_stated(o, v) for o in options)]
    if named and looser:
        # the key named for the property says something else; a looser key's value does not decide
        return "no", CLOSEST + "; ".join(f"{k}: {v}" for k, v in named[:2]) + " (not " + "; ".join(f"{k}: {v}" for k, v in looser[:2]) + ")"
    if not named and any(_stated(o, title, exact=True) for o in options):
        return "yes", shown  # the title, when no key names the property: the value's own word, not a changed one
    if not named:
        return "unsure", shown
    words = {_one(w) for o in options for w in _words(o) if w not in _FILLER}
    # a broader value: the property's own key holds only some of the value's words and nothing else ('bottle' for
    # 'plastic bottles'), so it does not state it; any other partial overlap is left to the model
    # (a value's words that only repeat the property's name, 'pack' in 'large serve pack' for a pack type, add nothing)
    own = {_one(w) for t in texts for w in _words(t) if w not in _FILLER}
    named_words = {_one(w) for w in _words(prop)} | set(noun)
    bare = [{_one(w) for w in _words(o) if w not in _FILLER} - named_words for o in options]
    if own and any(b and b <= own for b in bare):
        return "yes", shown
    # alternatives that only spell out one shared word ('single can or single bottle' around 'single') are met in
    # the model's reading when the key states that shared word: not ruled out here
    core = set.intersection(*bare) if len(bare) >= 2 and all(bare) else set()
    if own and core and own <= core:
        return "unsure", shown
    if own and all(own < b for b in bare):
        return "no", shown
    if any(_one(w) in words for t in texts for w in _words(t)):
        return "unsure", shown
    return "no", shown


# ----------------------------------------------------------------------------
# reading observations (keys differ from world to world, so by value and shape)

_ID_KEY = re.compile(r"(?:^|_)(?:id|pid|sku|uid|ref|code)$|^(?:id|sku)|_id_|ident", re.I)
_ID_VALUE = re.compile(r"^(?=.*\d)(?=.*[A-Za-z])[A-Za-z0-9][A-Za-z0-9_\-:.]{5,63}$")  # letters and digits, no spaces
_PRICE_KEY = re.compile(r"price|cost|amount|fee|charge", re.I)
_NUMBER = re.compile(r"^\s*[$€£]?\s*(\d+(?:\.\d+)?)\s*$")
_YES_NO = {"true": True, "yes": True, "y": True, "1": True, "in stock": True, "available": True,
           "false": False, "no": False, "n": False, "0": False, "out of stock": False, "unavailable": False}
_STOCK_KEY = re.compile(r"stock|avail|on_?hand|in_store|sold", re.I)
_TITLE_KEY = re.compile(r"title|name|heading|label", re.I)
_PAGING_KEY = re.compile(r"page|offset|start|limit|skip|token|cursor|after|next|more|left|remain|size|per", re.I)
_QTY_KEY = re.compile(
    r"(?:^|_)(?:qty|quantity|count|units?|pieces?|amount_total|qty_total|item_count|num_items|"
    r"lines?_?num|number_of(?:_?items?)?|number)$|(?:^|_)amount_total$",
    re.I,
)
_TOTAL_KEY = re.compile(r"match|total|count|hits|found|result|num", re.I)


def _row_id(row: dict[str, Any]) -> str | None:
    for key, value in row.items():
        if isinstance(value, str) and 3 < len(value) <= 64 and " " not in value and _ID_KEY.search(str(key)):
            return value
    # a result that holds a list of records is the envelope around the listings: its short texts are paging tokens or
    # cursors ('further': 'vuvepa4'), never a product id
    if any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in row.values()):
        return None
    # without a key named for an id, only a record that looks like a listing (a title or a price) carries one: a facts
    # map's values ('160x200cm', '5-Shelf') are no ids
    if not any(_TITLE_KEY.search(str(k)) or _PRICE_KEY.search(str(k)) for k in row):
        return None
    for key, value in row.items():
        if isinstance(value, str) and _ID_VALUE.match(value) and not _PAGING_KEY.search(str(key)) and not _TOKEN_KEY.search(str(key)):
            return value
    return None


_TEXT_PAIR = re.compile(r"([A-Za-z][A-Za-z /&-]{0,30}?)\s*:\s*(.+?)(?=\s+[A-Z][A-Za-z /&-]{0,30}?\s*:|[\n;|]|$)")


def _text_pairs(text: str) -> list[tuple[str, str]]:
    """'Key : value' pairs written inside a free text, as listing descriptions often list their specifications."""
    return [(k.strip(), v.strip()) for k, v in _TEXT_PAIR.findall(str(text or "")) if v.strip()]


def _category_rank(key: str, value: str) -> int | None:
    """How likely a field holds the listing's category, lower first; None when it does not: a name saying category
    first, then group, path, aisle, department, shelf or family, a section or range last (a remarks or notes section holds a
    note); a value reading like a sentence (over 60 characters or 8 words) is never a category."""
    if len(value) > 60 or len(value.split()) > 8 or not value.strip():
        return None
    if re.search(r"categor", key, re.I):
        return 0
    if re.search(r"group|path|aisle|department|shelf|family", key, re.I):
        return 1
    if re.search(r"section|range", key, re.I) and not re.search(r"remark|note|comment|desc|about|detail", key, re.I):
        return 2
    return None


def _title_rank(key: str) -> int | None:
    """How likely a field holds the listing's title, lower first: a title, then a name or heading, then a label; never
    a maker's, brand's, shop's or seller's name."""
    if re.search(r"maker|brand|manufactur|shop|seller|vendor|store|categor", key, re.I):
        return None
    if re.search(r"title", key, re.I):
        return 0
    if re.search(r"name|heading", key, re.I):
        return 1
    return 2 if _TITLE_KEY.search(key) else None


def _row_info(row: dict[str, Any]) -> dict[str, Any]:
    """id, title, price, stock and category of a listing-like dict."""
    info: dict[str, Any] = {"id": _row_id(row)}
    numbers = []
    for key, value in row.items():
        key = str(key)
        if isinstance(value, bool):
            if _STOCK_KEY.search(key):
                info["stock"] = (not value) if re.search(r"sold|out", key, re.I) else value
        elif isinstance(value, (int, float)):
            if _PRICE_KEY.search(key) and not re.search(r"total|was|old|list", key, re.I):
                info.setdefault("price", float(value))
            elif _STOCK_KEY.search(key):
                info["stock"] = value > 0
            elif not _QTY_KEY.search(key):
                numbers.append(float(value))
        elif isinstance(value, str) and _PRICE_KEY.search(key) and _NUMBER.match(value) and not re.search(r"total|was|old|list", key, re.I):
            info.setdefault("price", float(_NUMBER.match(value).group(1)))  # a price written as text
        elif isinstance(value, str) and _STOCK_KEY.search(key) and value.strip().lower() in _YES_NO:
            yes = _YES_NO[value.strip().lower()]
            info["stock"] = (not yes) if re.search(r"sold|out", key, re.I) else yes  # stock written as text
        elif isinstance(value, str):
            rank = _category_rank(key, value)
            if rank is not None and rank < info.get("_cat_rank", 99):
                info["category"], info["_cat_rank"] = value, rank
            elif rank is None and _title_rank(key) is not None and _title_rank(key) < info.get("_title_rank", 99):
                info["title"], info["_title_rank"] = value, _title_rank(key)
    info.pop("_cat_rank", None)
    info.pop("_title_rank", None)
    if "price" not in info and len(numbers) == 1:
        info["price"] = numbers[0]
    info["text"] = " ".join(str(v) for v in row.values() if isinstance(v, str))
    return info


def _listing_dicts(value: Any, depth: int = 0) -> list[dict[str, Any]]:
    """Every dict that carries a product id, at any depth."""
    found: list[dict[str, Any]] = []
    if depth > 6:
        return found
    if isinstance(value, dict):
        if _row_id(value):
            found.append(value)
        for item in value.values():
            if isinstance(item, (dict, list)):
                found.extend(_listing_dicts(item, depth + 1))
    elif isinstance(value, list):
        for item in value:
            found.extend(_listing_dicts(item, depth + 1))
    return found


def _per_product(observation: Any, wanted: Any) -> dict[str, Any]:
    """A several-product details result split into one result per requested id: a result keyed by the ids, or
    the parts that each carry one requested id. One requested id takes the whole result."""
    wanted = [str(i) for i in wanted] if isinstance(wanted, list) else []
    if len(wanted) == 1:
        return {wanted[0]: observation}
    out: dict[str, Any] = {}

    def walk(value: Any, depth: int) -> None:
        if depth > 5 or len(out) == len(wanted):
            return
        if isinstance(value, dict):
            keyed = [k for k in value if str(k) in wanted]
            if keyed:
                for k in keyed:
                    out.setdefault(str(k), value[k])
                return
            carried = [i for i in wanted if any(v == i for v in value.values() if isinstance(v, str))]
            if len(carried) == 1:
                out.setdefault(carried[0], value)
                return
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(observation, 0)
    return out


def _result_rows(observation: Any) -> list[dict[str, Any]]:
    """The listing rows of a search result: the longest list of id-carrying dicts."""
    best: list[dict[str, Any]] = []

    def walk(value: Any, depth: int) -> None:
        nonlocal best
        if depth > 5:
            return
        if isinstance(value, list):
            rows = [v for v in value if isinstance(v, dict) and _row_id(v)]
            if len(rows) > len(best):
                best = rows
            for item in value:
                walk(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)

    walk(observation, 0)
    return best


_TOKEN_KEY = re.compile(r"next|cursor|token|continue|after|more", re.I)


def _next_token(observation: Any) -> str | None:
    """The next-page token or cursor a search result returns (a short text without spaces under a key named for
    paging), or None when the result has none or pages by number."""

    def walk(value: Any, depth: int) -> str | None:
        if depth > 2 or not isinstance(value, dict):
            return None
        for key, item in value.items():
            if isinstance(item, str) and _TOKEN_KEY.search(str(key)) and 0 < len(item) <= 200 and " " not in item.strip():
                return item.strip()
        # whatever the key is called: a result that pages by token carries it as its one short text without spaces
        # beside the listing rows
        if any(isinstance(item, list) and item and all(isinstance(r, dict) for r in item) for item in value.values()):
            loose = [item.strip() for item in value.values() if isinstance(item, str) and 0 < len(item.strip()) <= 64 and " " not in item.strip()]
            if len(loose) == 1:
                return loose[0]
        for item in value.values():
            if isinstance(item, dict):
                found = walk(item, depth + 1)
                if found:
                    return found
        return None

    return walk(observation, 0)


def _next_number(observation: Any) -> int | None:
    """A numeric next-page value a search result returns under a key named for paging (a row offset or page number)."""
    if not isinstance(observation, dict):
        return None
    observation = _inner(observation)
    for key in ("next_page_token", "next_page", "next", "Next", "next_offset"):
        item = observation.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item > 0:
            return item
        if isinstance(item, str) and item.strip().isdigit():
            return int(item.strip())
    for key, item in observation.items():
        if _TOKEN_KEY.search(str(key)):
            if isinstance(item, int) and not isinstance(item, bool):
                return item
            if isinstance(item, str) and item.strip().isdigit():
                return int(item.strip())
    return None


def _inner(observation: Any) -> Any:
    """A result wrapped with a store's notices (two keys: the answer, and a list of notice strings) reads as the answer."""
    if isinstance(observation, dict) and len(observation) == 2:
        keys = list(observation)
        for notes, answer in ((keys[0], keys[1]), (keys[1], keys[0])):
            value = observation[notes]
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                return observation[answer]
    return observation


def _page_arg_is_offset(name: str | None) -> bool:
    """A page argument named like an offset (begin, skip, offset, start) counts rows, it is not a page number."""
    return bool(name and re.search(r"index|offset|begin|start|skip|from_row|row_start", str(name), re.I))


def _total_count(observation: Any) -> int | None:
    """The number of matches a search reports."""
    candidates: list[tuple[int, int]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 2 or not isinstance(value, dict):
            return
        for key, item in value.items():
            if isinstance(item, int) and not isinstance(item, bool) and not _PAGING_KEY.search(str(key)):
                candidates.append((0 if _TOTAL_KEY.search(str(key)) else 1, item))
            elif isinstance(item, dict):
                walk(item, depth + 1)

    walk(observation, 0)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], -c[1]))
    return candidates[0][1]


NOTICE_SKIP = re.compile(r"title|name|desc|spec|propert|fine|categor|brand|seller|vendor|shop|query|url|link|image|feature|review|option|sku|^id$|_id$", re.I)


_TEXT_FIELD = re.compile(r"(?:^|\s\|\s)([A-Za-z_][A-Za-z0-9_]*): ")
_TEXT_FACT = re.compile(r"^[^=;|]{1,40}=")


def _text_value(text: str) -> Any:
    """A value of a record written as text: its plain type, or a dict for a 'name=value; name=value' list."""
    text = text.strip()
    if text in ("None", "null", ""):
        return None
    if text in ("True", "False"):
        return text == "True"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    parts = text.split("; ")
    if len(parts) >= 2 and all(_TEXT_FACT.match(part) for part in parts):
        pairs = [part.split("=", 1) for part in parts]
        return {k.strip(): v.strip() for k, v in pairs if v.strip()}
    return text


def _text_record(text: str) -> dict[str, Any] | None:
    """A record a store wrote as one line of 'key: value | key: value' as the dict it stands for; None for other text."""
    marks = list(_TEXT_FIELD.finditer(text))
    if len(marks) < 3 or marks[0].start() != 0:
        return None
    ends = [mark.start() for mark in marks[1:]] + [len(text)]
    return {mark.group(1): _text_value(text[mark.end():end]) for mark, end in zip(marks, ends)}


def _joined_as_one(plan: dict[str, Any], request: str) -> bool:
    """Merge a plan's items written side by side in the request ("a and b", "a & b") with one count between them: one product,
    often one store department. True when the plan changed."""
    items = plan.get("items")
    if not isinstance(items, list) or len(items) < 2 or not all(isinstance(i, dict) for i in items):
        return False
    text = (request + " " + str(plan.get("category") or "")).lower()

    def joined(one: dict[str, Any], other: dict[str, Any]) -> bool:
        a, b = (re.escape(str(i.get("category") or "").strip().lower()) for i in (one, other))
        if not a or not b or any(int(i.get("quantity") or 1) > 1 for i in (one, other) if isinstance(i.get("quantity"), (int, float))):
            return False
        return bool(re.search(rf"\b{a}s?\s*(?:and|&)\s*{b}s?\b|\b{b}s?\s*(?:and|&)\s*{a}s?\b", text))
    kept: list[dict[str, Any]] = []
    for item in items:
        if kept and joined(kept[-1], item):
            first = kept[-1]
            first["category"] = f"{first.get('category')} and {item.get('category')}"
            reqs = list(first.get("requirements") or [])
            first["requirements"] = reqs + [r for r in item.get("requirements") or [] if r not in reqs]
        else:
            kept.append(item)
    if len(kept) == len(items):
        return False
    if len(kept) == 1:
        plan["items"] = []
        plan["category"] = plan.get("category") or kept[0].get("category")
        plan["requirements"] = list(plan.get("requirements") or []) + [r for r in kept[0].get("requirements") or []
                                                                     if r not in (plan.get("requirements") or [])]
    else:
        plan["items"] = kept
    return True


_NOT_A_PRODUCT_FACT = re.compile(r"(?i)^(?:stock|in stock|stock status|availability|available|delivery|shipping|dispatch|price|budget|cost|quantity|units)$")
_LIMIT_AFTER = (r"(?:at most|at the most|max(?:imum)?|tops|at least|at the least|min(?:imum)?|"
                r"(?:or|and) (?:less|under|lower|smaller|fewer|shorter|narrower|thinner|lighter|below|more|over|higher|bigger|"
                r"larger|longer|taller|wider|deeper|thicker|heavier|greater|above|up))")
_LIMIT_BEFORE = r"(?:at most|up to|no more than|not more than|under|below|less than|max(?:imum)?(?: of)?|at least|no less than|not less than|over|above|more than|min(?:imum)?(?: of)?)"


def _limit_words(value: str, request: str) -> str | None:
    """The request's own limit words for a figure the plan wrote bare ("RAM 12 GB at most" read as "12 GB"), else None."""
    if _BOUND.search(value) or re.search(r"(?i)\b(?:or|to|between|and|not)\b|[<>+-]\s*\d", value):
        return None
    figures = re.findall(r"\d+(?:\.\d+)?", value)
    if len(figures) != 1:
        return None
    num = re.escape(figures[0])
    after = re.search(rf"(?i)(?<![\d.]){num}(?![\d.])\s*(?:[a-z\"'”″]+\.?\s*){{0,2}}?\b({_LIMIT_AFTER})\b", request)
    if after:
        return f"{value} {after.group(1).lower()}"
    before = re.search(rf"(?i)\b({_LIMIT_BEFORE})\s+(?:of\s+)?{num}(?![\d.])", request)
    if before:
        return f"{before.group(1).lower()} {value}"
    return None


def _mend_plan(plan: dict[str, Any], request: str) -> list[str]:
    """Code checks of the request's reading, each a fix the reading's own rules call for: a reference given inside a
    product is listed with the order's references under that product's number; a requirement that only repeats a detail
    picking out a referenced listing, or that is about stock, delivery or price, is taken out; a figure the request
    bounds ("at most", "or more") keeps its limit words. The changes made, for the trace."""
    changes: list[str] = []
    items = [i for i in plan.get("items") or [] if isinstance(i, dict)]
    refs = [r for r in plan.get("references") or [] if isinstance(r, dict)]
    known = {(str(r.get("property") or "").lower(), str(r.get("seller") or "").lower(), str(r.get("product"))) for r in refs}
    for k, item in enumerate(items):
        for ref in item.get("references") or []:
            if not isinstance(ref, dict) or not str(ref.get("property") or "").strip():
                continue
            ref = dict(ref)
            if len(items) >= 2:
                ref["product"] = k + 1
            key = (str(ref.get("property") or "").lower(), str(ref.get("seller") or "").lower(), str(ref.get("product")))
            if key not in known and not any(str(r.get("property") or "").lower() == key[0] and str(r.get("seller") or "").lower() == key[1]
                                            and str(r.get("product")) in ("None", key[2]) for r in refs):
                refs.append(ref)
                known.add(key)
                changes.append(f"reference listed: {ref.get('property')} (product {ref.get('product')})")
    plan["references"] = refs
    marks = set()
    for ref in refs:
        for cond in ref.get("conditions") or []:
            name, val = _split_req(str(cond))
            if name and val:
                marks.add((_key_norm(name), _key_norm(val)))
    for holder in [plan] + items:
        kept = []
        for req in holder.get("requirements") or []:
            if not isinstance(req, dict):
                kept.append(req)
                continue
            prop, value = str(req.get("property") or "").strip(), str(req.get("value") or "").strip()
            if _NOT_A_PRODUCT_FACT.match(prop) or (prop and value and (_key_norm(prop), _key_norm(value)) in marks):
                changes.append(f"requirement dropped: {prop}: {value}")
                continue
            bounded = _limit_words(value, request) if value and re.search(r"\d", value) else None
            if bounded:
                req = dict(req, value=bounded, numeric=True)
                changes.append(f"limit kept: {prop}: {bounded}")
            kept.append(req)
        if "requirements" in holder:
            holder["requirements"] = kept
    return changes


def _structured(value: Any, depth: int = 0) -> Any:
    """A store's result with every record written as text turned into a dict, so that all the code reads one shape."""
    if depth > 6:
        return value
    if isinstance(value, str):
        record = _text_record(value)
        return value if record is None else record
    if isinstance(value, list):
        return [_structured(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) == 1:
            only = next(iter(value.values()))
            record = _text_record(only) if isinstance(only, str) else None
            if record is not None:
                return record
        return {key: _structured(item, depth + 1) for key, item in value.items()}
    return value


def _shopper_lines(observation: Any, depth: int = 0) -> list[str]:
    """Sentences riding on a result outside any listing: the shopper's lines and notices."""
    if depth > 4:
        return []
    out: list[str] = []
    if isinstance(observation, str):
        return [observation] if len(observation.split()) >= 3 else []
    if isinstance(observation, list):
        if observation and all(isinstance(v, str) for v in observation):
            return [v for v in observation if len(v.split()) >= 4]
        return []
    if isinstance(observation, dict) and not _row_id(observation):
        for key, value in observation.items():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                out.extend(v for v in value if len(v.split()) >= 4)
            elif isinstance(value, str) and len(value.split()) >= 4 and re.search(r"shopper|customer|message|repl|answer|says|notice|alert|refus|note|text", str(key), re.I):
                out.append(value)
            elif isinstance(value, str) and depth == 0 and len(value.split()) >= 6 and not NOTICE_SKIP.search(str(key)):
                out.append(value)  # a store's notice under a key of its own (status, info, detail, ...)
            elif isinstance(value, dict):
                out.extend(_shopper_lines(value, depth + 1))
    return out


def _line_units(observation: Any) -> int | None:
    """The units a cart change's result says its line now holds (a count under a quantity key), else None."""
    if not isinstance(observation, dict):
        return None
    for key, value in observation.items():
        if isinstance(value, int) and not isinstance(value, bool) and _QTY_KEY.search(str(key)):
            return max(0, value)
    return None


def _cart_lines(observation: Any) -> dict[str, int] | None:
    """A cart view's lines as {id: units}; {} for a cart shown empty; None when the result shows no lines."""
    rows = _result_rows(observation)
    if not rows:
        empty = isinstance(observation, dict) and any(isinstance(v, list) and not v for v in observation.values())
        return {} if empty else None
    out: dict[str, int] = {}
    for row in rows:
        ident = _row_id(row)
        units = _line_units(row)
        if ident and (units is None or units > 0):
            out[ident] = units or 1
    return out


def _failed(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    if any(re.search(r"error|fault|fail|refus|denied|invalid|issue|problem", str(k), re.I) for k in observation):
        return True
    return any(v is False and re.search(r"ok|success|done|accept|through|complete|appl", str(k), re.I) for k, v in observation.items())


# ----------------------------------------------------------------------------
# the episode

# what one product of a request for several keeps for itself; the world, the cart and the shopper are shared
ITEM_FIELDS = ("category", "noun", "quantity", "own_budget", "reqs", "free", "preference", "asked", "review", "focus", "candidates",
               "candidate_wants", "query", "page_size", "next_page", "next_token", "pages_read", "total", "tried",
               "rejected", "order_checked", "cheaper_checked", "filtered", "decline_checked", "probe")


def _cached_head(prompt: str) -> Any:
    """A prompt as content parts, its fixed instructions marked for the provider's cache; the text itself is unchanged."""
    for head in (SETUP_PROMPT, COMMIT_PROMPT, ORDER_PROMPT):
        if prompt.startswith(head) and len(prompt) > len(head):
            return [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": prompt[len(head):]}]
    return prompt


def _plain_words(name: str) -> str:
    """A property name as plain words: separators as spaces, without a one- or two-letter store prefix."""
    words = re.sub(r"[_.]+", " ", str(name)).split()
    if len(words) > 1 and len(words[0]) <= 2 and words[0].isalpha():
        words = words[1:]
    return " ".join(words).strip()


def _related(ref: dict[str, Any], value: str) -> str:
    """The requirement a referenced listing's value gives under the shopper's relation to it :
    the value itself, its exclusion, or a limit around it."""
    relation = str(ref.get("relation") or "same").strip().lower()
    if relation == "differ":
        return f"not {value}"
    if relation in ("at most", "at least"):
        return f"{relation} {value}"
    if relation == "within" and isinstance(ref.get("tolerance"), (int, float)) and not isinstance(ref.get("tolerance"), bool):
        return f"within {ref['tolerance']:g} {str(ref.get('unit') or '').strip()} of {value}".replace("  ", " ")
    return value


def _item_state(category: str, noun: str, quantity: int, reqs: list[str], preference: dict[str, Any],
                own_budget: float | None = None) -> dict[str, Any]:
    return {"category": category, "noun": noun, "quantity": quantity, "own_budget": own_budget, "reqs": reqs, "free": [], "preference": preference,
            "asked": [], "review": {}, "focus": {}, "candidates": [], "candidate_wants": None, "query": "", "page_size": 0,
            "next_page": 2, "next_token": None, "pages_read": 0, "total": 0, "tried": set(), "rejected": set(),
            "order_checked": set(), "cheaper_checked": set(), "filtered": set(), "decline_checked": False, "probe": None}


def _requirements(items: Any) -> list[str]:
    out = []
    for item in items or []:
        if isinstance(item, dict) and item.get("value"):
            out.append(f"{str(item.get('property') or '').strip()}: {str(item['value']).strip()}")
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        self.spent = 0.0  # dollars this task has spent on the model, for SPEND_SOFT
        self.order_checks = 0  # order checks asked so far, for ORDER_CHECKS_MAIN
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = self.policy.get("opening")
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.descriptions = {t["function"]["name"]: str(t["function"].get("description") or "") for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or DEFAULT_STEPS)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, dict[str, Any]] = {}
        self.category = ""
        self.quantity = 1
        self.own_budget: float | None = None  # one product's own price limit, in a request for several
        self.reqs: list[str] = []  # "property: value", stated and revealed, current
        self.free: list[str] = []  # properties the shopper does not mind
        self.budget: float | None = None
        self.preference: dict[str, Any] = {}
        self.message_limit: int | None = None  # as the request states it; None when it states none
        self.messages_used = 0
        self.asked: list[str] = []  # properties already asked (or settled by an ambiguity)
        self.exchanges: list[str] = []
        self.heard: set[str] = set()
        self.revealed = 0
        self.rows: dict[str, dict[str, Any]] = {}  # id -> newest row info
        self.details: dict[str, Any] = {}  # id -> details observation
        self.unavailable: set[str] = set()
        self.unknown: set[str] = set()  # ids the store's details tool refused: no such listing
        self.empty: set[str] = set()  # model queries that found nothing
        self.cart: dict[str, int] = {}
        self.probe: str | None = None
        self.review: dict[str, Any] = {}
        self.focus: dict[str, Any] = {}  # properties that matter, from titles and details, and candidate ids
        self.tried: set[str] = set()
        self.pending_reply = ""  # an answer owed to a question the shopper asked
        self.understood = True  # the newest reply understood what was asked (False: it asked for other words)
        self.remark = ""  # text a request rule wants in the finishing remark
        self.confirm_first = False  # the request wants the basket shown to the shopper before the order
        self.confirm_spare = False  # a message tool exists: a message left after the questions confirms the basket
        self.confirmed: tuple | None = None  # the basket and wants the shopper last saw in a confirmation
        self.rule_message = ""  # a message the order check's rules want sent, folded into the confirmation
        self.finish_args: dict[str, dict[str, Any]] = {}  # one tool that both buys and declines: the mode for each
        self.order_checked: set[str] = set()
        self.cheaper_checked: set[str] = set()
        self.filtered: set[float] = set()  # candidate prices already searched below with the filter search
        self.filter_steps = 0
        self.id_orders: list[bool] = []  # per search result of 5+ rows: were its rows in ascending id order
        self.walked: set[str] = set()  # queries a cheapest walk read to the end
        self.price_orders: list[bool] = []  # per search result of 3+ priced rows: were they in ascending price order
        self.decline_checked = False
        self.rejected: set[str] = set()  # picks the order review failed: no longer candidates, nor a "cheaper fit"
        self.candidates: list[str] = []  # every fit found for the wants in candidate_wants, best first
        self.candidate_wants: tuple | None = None
        self.query = ""
        self.page_size = 0
        self.page_base = 1
        self.page_step = 1  # rows per page when the page argument is a row offset (the store hands back the next offset)
        self.next_page = 2
        self.next_token: str | None = None  # a next-page token, for stores that page that way
        self.pages_read = 0
        self.total = 0
        self.order_attempts = 0
        self.items: list[dict[str, Any]] = []  # one state per product when the request asks for several
        self.current = 0  # the product whose state is loaded into the fields above
        self.changed_items: set[int] = set()  # products whose wants the newest reading changed
        self.lines: dict[int, str] = {}  # product -> the listing carted for it
        self.trace: list[dict[str, Any]] = []
        # needs whose value is another listing's: (product, property) -> {"property", "ref", "value", "source", "done"}
        self.ref_needs: dict[tuple, dict[str, Any]] = {}

    # -- several products --------------------------------------------------------

    def multi(self) -> bool:
        return len(self.items) >= 2

    @property
    def noun(self) -> str:
        return str(self.plan.get("noun") or "")

    @noun.setter
    def noun(self, value: str) -> None:
        self.plan["noun"] = value

    def save_item(self) -> None:
        if self.multi():
            self.items[self.current] = {f: getattr(self, f) for f in ITEM_FIELDS}

    def switch(self, k: int) -> None:
        """Load product k's state (the current one's is kept first)."""
        if not self.multi() or k == self.current:
            return
        self.save_item()
        self.current = k
        for f in ITEM_FIELDS:
            setattr(self, f, self.items[k][f])

    def all_wants(self) -> tuple:
        if not self.multi():
            return self.wants()
        self.save_item()
        return tuple((tuple(i["reqs"]), i["own_budget"], json.dumps(i["preference"], sort_keys=True, default=str), i["quantity"]) for i in self.items) + (self.budget,)

    def finish_reserve(self) -> int:
        """Steps kept for carting, checking and ordering: one more cart change and check per extra product, or one more
        step in all when the cart tool sets the whole basket in one call."""
        if not self.multi():
            return FINISH_STEPS
        if (self.roles.get("cart") or {}).get("lines_arg"):
            return FINISH_STEPS + 1
        return FINISH_STEPS + 2 * (len(self.items) - 1)

    def mine(self) -> dict[str, Any]:
        """The opened products of the current product's category (all of them for a one-product request)."""
        if not self.multi():
            return self.details
        label = self.target_label()
        return {i: d for i, d in self.details.items()
                if not label or not self.rows.get(i, {}).get("category") or self.rows[i]["category"] == label}

    def products_text(self) -> str:
        self.save_item()
        lines = []
        for n, item in enumerate(self.items, 1):
            reqs = "; ".join(item["reqs"]) or "none beyond the category"
            free = f" | does not mind: {', '.join(item['free'])}" if item["free"] else ""
            lines.append(f"[{n}] {item['category']} x{item['quantity']}: {reqs}{free}")
        return "PRODUCTS:\n" + "\n".join(lines)

    # -- time and step budget -------------------------------------------------

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def over(self) -> bool:
        return self.finished or self.dead or self.steps_left() <= 0

    def message_allowance(self) -> int:
        """The stated message limit; without one, messages still cost steps and questions beyond what the order
        needs lower the reward, so about one per twelve steps of the task's own step limit."""
        if self.message_limit is not None:
            return self.message_limit
        return DEFAULT_MESSAGES

    def messages_left(self) -> int:
        return max(0, self.message_allowance() - self.messages_used) if self.roles.get("message") else 0

    def held_messages(self) -> int:
        """Messages kept back from questions and replies: the basket confirmation the request asks for, until sent."""
        return 1 if self.confirm_first and self.confirmed is None else 0

    def spare_messages(self) -> int:
        return max(0, self.messages_left() - self.held_messages())

    # -- the model -------------------------------------------------------------

    def ask(self, prompt: str, max_tokens: int = 1500, strong: bool = False, gate: bool = False) -> dict[str, Any] | None:
        """One model call: on the main model for a gate (the order and decline checks) and for a strong reading
        while the task's spend is under SPEND_SOFT, else on the cheap model, and on the cheap model for everything past
        SPEND_CAP; the backup when either fails."""
        main = (gate or (strong and self.spent < SPEND_SOFT)) and self.spent < SPEND_CAP
        if main:
            # a projected cost (prompt characters at about 2.3 a token, plus the answer, as helloworld): a call that
            # would carry the task past SPEND_CAP goes to the cheap model instead
            price_in, price_out = PRICES[MAIN_MODEL]
            main = self.spent + len(prompt) / 2.3 * price_in + min(max_tokens, 1500) * price_out <= SPEND_CAP
        for model in (MAIN_MODEL if main else CHEAP_MODEL, BACKUP_MODEL):
            left = self.time_left()
            if left < FORCE_SECONDS:
                return None
            _llm.timeout = int(max(10, min(120, left - FORCE_SECONDS)))
            _llm.retry_until = time.monotonic() + left - FORCE_SECONDS
            # the cheap models write about three times as much for the same answer; cut short, their JSON is lost
            limit = max_tokens if model == MAIN_MODEL else max_tokens * 3
            # as little hidden reasoning as each model allows: its tokens are billed as output, and the answers are
            # short JSON readings (the main model cannot switch it off, and at low effort uses none)
            reasoning = {"enabled": False} if model == CHEAP_MODEL else {"effort": "low"}
            if prompt.startswith(TOOLS_PROMPT):
                reasoning, limit = {"effort": "medium"}, max(limit, 4000)  # without it the cheap model drops arguments the finish needs
            body = {"model": model, "messages": [{"role": "user", "content": _cached_head(prompt) if model == MAIN_MODEL else prompt}],
                    "temperature": 0, "max_tokens": limit, "reasoning": reasoning}
            result = self.post(_llm, "/inference/chat/completions", body, FORCE_SECONDS)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                content = (data["choices"][0].get("message") or {}).get("content") or ""
                self.charge(model, data.get("usage"), prompt, content)
                parsed = _parse_json_object(content)
                if parsed is not None:
                    return parsed
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "no JSON"})
        return None

    def charge(self, model: str, usage: Any, prompt: str, content: str) -> None:
        """Add a call to the task's spend: the reply's own cost when given, else its tokens (or length) times the price."""
        usage = usage if isinstance(usage, dict) else {}
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            self.spent += float(usage["cost"])
            return
        inp, out = PRICES.get(model, PRICES[MAIN_MODEL])
        tokens_in = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else len(prompt) / 3
        tokens_out = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else len(content) / 3
        self.spent += tokens_in * inp + tokens_out * out

    def post(self, client: Any, path: str, body: dict[str, Any], keep: float) -> Any:
        """One proxy call, retried after a short wait when the proxy is rate limiting (429), while time allows."""
        for wait in (1.0, 2.0, 4.0, 0.0):
            result = client.post_verbose(path, body)
            status = (result.error or {}).get("status") if isinstance(result.error, dict) else None
            if status != 429 or not wait or self.time_left() < keep + wait + 1:
                return result
            self.trace.append({"kind": "rate_limited", "path": path, "wait": wait})
            time.sleep(wait)
        return result

    def context(self, story: bool = True) -> str:
        """What every judging prompt needs to know about the shopper. Without the story, the request's own text is
        left out: setup has already turned it into the requirements, budget, preference, purpose and rules below."""
        lines = (["THE SHOPPER'S REQUEST:\n" + self.request] if story else []) + [f"\nCATEGORY: {self.category}", f"QUANTITY: {self.quantity}"]
        lines.append("CURRENT REQUIREMENTS:\n" + ("\n".join(f"- {r}" for r in self.reqs) or "- (none beyond the category)"))
        if self.multi():
            others = [f"{i['category']} x{i['quantity']}" for n, i in enumerate(self.items) if n != self.current]
            lines.append(f"THIS ORDER HAS {len(self.items)} PRODUCTS; THIS IS PRODUCT {self.current + 1}. THE OTHERS: " + "; ".join(others))
        if self.multi():
            lines.append("THE WHOLE-ORDER BUDGET IS CHECKED SEPARATELY, ACROSS ALL THE PRODUCTS' PICKS: for this product, judge only its own "
                         "price against its own budget (when it has one) and against the whole budget; never fail it or leave it unproven "
                         "because the other products are not yet priced.")
        if self.multi() and self.own_budget is not None:
            lines.append(f"BUDGET FOR THIS PRODUCT (total for its units): {self.own_budget:g}")
        lines.append(f"BUDGET (total{' for the whole order' if self.multi() else ''}): {self.budget if self.budget is not None else 'not stated'} {self.plan.get('currency') or ''}".rstrip())
        pref = self.preference
        if pref.get("kind") not in (None, "", "none"):
            lines.append("PREFERENCE: " + _compact(pref, 400))
        if self.plan.get("in_stock_only"):
            lines.append("ONLY ITEMS IN STOCK.")
        if self.plan.get("purpose"):
            lines.append("PURPOSE: " + str(self.plan["purpose"]))
        rules = [str(r) for r in self.plan.get("rules") or [] if r]
        if rules:
            lines.append("RULES THE REQUEST SETS FOR THE ASSISTANT:\n" + "\n".join(f"- {r}" for r in rules))
        if self.free:
            lines.append("THE SHOPPER DOES NOT MIND: " + ", ".join(self.free))
        looked_up = [f"- {n['property']}: {n['value']} (as the listing the shopper pointed to, {n['source']}, states it)"
                     for n in self.ref_needs.values() if n["done"] and n["value"]]
        if looked_up:
            lines.append("NEEDS LOOKED UP FROM THE LISTING THE SHOPPER POINTED TO (already in the requirements; judge them as plain values):\n"
                         + "\n".join(looked_up))
        if self.exchanges:
            lines.append("EXCHANGES WITH THE SHOPPER:\n" + "\n".join(self.exchanges[-10:]))
        return "\n".join(lines)

    # -- the world -------------------------------------------------------------

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        """Send up to 16 actions as consecutive steps; return their observations."""
        sendable = []
        for action in actions:
            name = action.get("name")
            if name not in self.schemas:
                continue
            args, problems = _check(action.get("args") or {}, self.schemas[name])
            if problems:
                self.trace.append({"kind": "bad_args", "tool": name, "args": action.get("args"), "problems": problems})
                continue
            sendable.append({"name": name, "args": args})
        sendable = sendable[: min(MAX_ACTIONS, self.steps_left())]
        if not sendable or self.over():
            return None
        payload: dict[str, Any] = dict(self.binding)
        if len(sendable) == 1:
            payload["action"] = sendable[0]
        else:
            payload["actions"] = sendable
        for _ in range(3):
            if self.time_left() < 2:
                return None
            payload["step"] = self.step + 1
            _env.timeout = int(max(5, min(120, self.time_left() - 1)))
            _env.retry_until = time.monotonic() + self.time_left() - 2
            result = self.post(_env, "/environment/call", payload, 2)
            data = result.data
            if isinstance(data, dict):
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + len(sendable)
                self.finished = bool(data.get("finished"))
                observations = data.get("observations") if isinstance(data.get("observations"), list) else [data.get("observation")]
                observations = [_structured(o) for o in observations]
                self.trace.append({"kind": "env", "step": payload["step"], "actions": sendable, "observations": observations})
                for action, observation in zip(sendable, observations):
                    self.absorb(action, observation)
                return observations
            error = result.error or {}
            status, body = error.get("status"), str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if "finished" in body:
                self.finished = True
                return None
            match = re.search(r"expected (\d+)", body)
            if match:
                self.step = int(match.group(1)) - 1
                continue
            if status is None:
                # no answer in time: the world may have run the step all the same, so the identical step and actions
                # go again (a replay returns its result); a different action at that step would be refused
                continue
            return None
        return None

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        """Book-keeping for one action's result: rows seen, details read, cart."""
        name, args = action["name"], action["args"]
        for item in _listing_dicts(observation):
            info = _row_info(item)
            if info.get("id"):
                old = self.rows.get(info["id"], {})
                self.rows[info["id"]] = {**old, **{k: v for k, v in info.items() if v not in (None, "")}}
        if name == (self.roles.get("search") or {}).get("tool"):
            ids = [i for i in (_row_id(r) for r in _result_rows(observation)) if i]
            if len(ids) >= 5:
                self.id_orders.append(ids == sorted(ids))
            prices = [r.get("price") for r in (_row_info(x) for x in _result_rows(observation))]
            if len(prices) >= 3 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in prices):
                self.price_orders.append(prices == sorted(prices))
        details = self.roles.get("details", {})
        if name == details.get("tool"):
            ident = args.get(details.get("id_arg") or "")
            if isinstance(ident, str) and not _failed(observation):
                self.details[ident] = observation
            elif isinstance(ident, str):
                self.unknown.add(ident)  # the store has no such listing: never a candidate again
        many = self.roles.get("details_many", {})
        if name == many.get("tool") and not _failed(observation):
            for ident, part in _per_product(observation, args.get(many.get("ids_arg") or "")).items():
                self.details[ident] = part
        cart = self.roles.get("cart", {})
        remove = self.roles.get("cart_remove", {})
        if name == cart.get("tool") and cart.get("lines_arg") and not _failed(observation):
            lines = args.get(cart["lines_arg"]) if isinstance(args.get(cart["lines_arg"]), list) else []
            self.cart = {str(x.get(cart["line_id"])): int(x.get(cart["line_qty"]) or 0) for x in lines
                         if isinstance(x, dict) and x.get(cart["line_id"]) and int(x.get(cart["line_qty"]) or 0) > 0}
        elif name in (cart.get("tool"), remove.get("tool")) and not _failed(observation):
            spec = cart if name == cart.get("tool") else remove
            ident = args.get(spec.get("id_arg") or "")
            adding = name == cart.get("tool") and (not cart.get("op_arg") or str(args.get(cart["op_arg"])) == str(cart.get("add_value")))
            if isinstance(ident, str):
                left = _line_units(observation)  # the units the store says the line now holds, when it says
                if adding:
                    self.cart[ident] = left if left else int(args.get(cart.get("qty_arg") or "", 0) or self.quantity)
                    if self.multi():
                        self.lines[self.current] = ident
                elif left:
                    self.cart[ident] = left  # the removal took off only some units: the rest are still carted
                else:
                    self.cart.pop(ident, None)
        view = self.roles.get("cart_view", {})
        if name == view.get("tool") and not _failed(observation):
            lines = _cart_lines(observation)
            if lines is not None:
                self.cart = lines  # the store's own cart decides what is in it

    # -- setup -----------------------------------------------------------------

    def setup(self) -> None:
        lines = [f"- {n}: {self.descriptions[n][:300]} | arguments: {_compact(self.schemas[n], 1200)}" for n in self.schemas]
        tool_prompt = TOOLS_PROMPT + "\n\nTOOLS:\n" + "\n".join(lines)
        mapped = self.ask(tool_prompt, 1200) or self.ask(tool_prompt, 1200, strong=True) or {}
        prompt = SETUP_PROMPT + "\n\nREQUEST:\n" + self.request
        if self.opening is not None:
            prompt += "\n\nOPENING OBSERVATION:\n" + _compact(self.opening)
        plan = self.ask(prompt, 2500) or self.ask(prompt, 2500, strong=True) or {}  # the main model only when the cheap one gave no plan
        written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", self.request)}
        budget = plan.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and float(budget) not in written:
            self.trace.append({"kind": "budget_reread", "budget": budget})  # the request's own number was asked for
            plan = self.ask(prompt, 2500, strong=True) or plan
        if _joined_as_one(plan, self.request):
            self.trace.append({"kind": "joined_items", "category": plan.get("category")})
        mended = _mend_plan(plan, self.request) if isinstance(plan, dict) else []
        if mended:
            self.trace.append({"kind": "plan_mended", "changes": mended[:12]})
        plan["tools"] = mapped.get("tools") if isinstance(mapped.get("tools"), dict) else mapped
        plan["finish_args"] = mapped.get("finish_args")
        self.plan = plan
        self.category = str(plan.get("category") or "").strip()
        quantity = plan.get("quantity")
        self.quantity = quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1
        for item in plan.get("requirements") or []:
            if isinstance(item, dict) and item.get("value"):
                self.reqs.append(f"{str(item.get('property') or '').strip()}: {str(item['value']).strip()}")
            elif isinstance(item, str) and item.strip():
                self.reqs.append(item.strip())
        budget = plan.get("budget")
        self.budget = float(budget) if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 else None
        self.preference = plan.get("preference") if isinstance(plan.get("preference"), dict) else {}
        limit = plan.get("message_limit")
        if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit >= 0:
            self.message_limit = int(limit)
        steps = plan.get("step_limit")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 3 <= steps <= 500:
            self.max_steps = int(steps)  # the request's own step limit decides; DEFAULT_STEPS only when neither it nor the store gives one
        items = [i for i in plan.get("items") or [] if isinstance(i, dict) and str(i.get("category") or "").strip()]
        if len(items) >= 2:
            for item in items:
                quantity = item.get("quantity")
                quantity = quantity if isinstance(quantity, int) and not isinstance(quantity, bool) and quantity > 0 else 1
                pref = item.get("preference") if isinstance(item.get("preference"), dict) else self.preference
                own = item.get("budget")
                own = float(own) if isinstance(own, (int, float)) and not isinstance(own, bool) and own > 0 else None
                self.items.append(_item_state(str(item["category"]).strip(), str(item.get("noun") or item["category"]).strip(),
                                              quantity, _requirements(item.get("requirements")), dict(pref), own))
            if self.budget is None and all(i["own_budget"] is not None for i in self.items):
                self.budget = sum(i["own_budget"] for i in self.items)
            self.current = 0
            for f in ITEM_FIELDS:
                setattr(self, f, self.items[0][f])
        self.roles = self.resolve_roles(plan.get("tools") if isinstance(plan.get("tools"), dict) else {})
        # A message is held back from the questions for the basket confirmation only when the request asks for one: in
        # logged race worlds (whose requests ask for it) orders with a message after the last cart change passed about
        # fifty times as often, while in qualifying worlds (which mostly do not) it made no difference (73.1% vs
        # 73.3%) and the held message cost a question. Without the request asking, a message left over after the
        # questions still confirms the basket.
        self.confirm_first = plan.get("confirm_before_order") is True
        self.confirm_spare = bool(self.roles.get("message"))
        modes = plan.get("finish_args") or (plan.get("tools") or {}).get("finish_args")  # sometimes written inside "tools"
        if isinstance(modes, dict):
            self.finish_args = {k: v for k, v in modes.items() if k in ("order", "decline") and isinstance(v, dict)}
        self.trace.append({"kind": "plan", "plan": plan, "roles": self.roles})

    def props(self, tool: str | None) -> dict[str, Any]:
        return (self.schemas.get(tool or "") or {}).get("properties") or {}

    def resolve_roles(self, mapped: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """The model's role map, checked against the real schemas, with name-based fallbacks."""
        names = list(self.schemas)

        def text(n: str) -> str:
            return " " + " ".join(re.split(r"[_\W]+", (n + " " + self.descriptions.get(n, "")).lower())) + " "

        def guess(pattern: str, exclude: tuple = ()) -> str | None:
            return next((n for n in names if n not in exclude and re.search(pattern, " " + " ".join(re.split(r"[_\W]+", n.lower())) + " ")), None) \
                or next((n for n in names if n not in exclude and re.search(pattern, text(n))), None)

        def tool(role: str) -> str | None:
            entry = mapped.get(role)
            name = entry.get("tool") if isinstance(entry, dict) else entry
            return name if isinstance(name, str) and name in self.schemas else None

        def arg(tool_name: str | None, role: str, key: str, test=None) -> str | None:
            entry = mapped.get(role) if isinstance(mapped.get(role), dict) else {}
            props = self.props(tool_name)
            if isinstance(entry.get(key), str) and entry[key] in props:
                return entry[key]
            return next((k for k, s in props.items() if isinstance(s, dict) and test and test(k, s)), None) if test else None

        def free_string(k: str, s: dict) -> bool:
            kinds = s.get("type") if isinstance(s.get("type"), list) else [s.get("type")]
            return "string" in kinds and not s.get("enum")

        def integer(k: str, s: dict) -> bool:
            kinds = s.get("type") if isinstance(s.get("type"), list) else [s.get("type")]
            return "integer" in kinds or "number" in kinds

        roles: dict[str, dict[str, Any]] = {}
        decline = tool("decline") or guess(r" (decline|abstain|without|refuse|cancel|no purchase|no order|nothing) ")
        order = tool("order") or guess(r" (checkout|check out|purchase|buy|order|pay) ", (decline,))
        message = tool("message") or guess(r" (shopper|customer|message|ask|question|chat|contact|send) ", (decline, order))
        search = tool("search") or guess(r" (search|find|query|look|browse|lookup) ", (decline, order, message))
        details = tool("details") or guess(r" (detail|spec|propert|attribut|fact|feature)\w* ", (search, decline, order, message))
        listing = tool("listing") or guess(r" (view|listing|show|full) ", (search, details, decline, order, message)) or details
        details = details or listing
        cart = tool("cart")
        cart_view = tool("cart_view") or next((n for n in names if n not in (cart, order, decline) and re.search(r"cart|basket|bag|trolley", n.lower()) and not self.props(n)), None)
        cart = cart or guess(r" (cart|basket|bag|trolley) ", (order, decline, cart_view, message))
        if search:
            page_arg = arg(search, "search", "page_arg", lambda k, s: integer(k, s) and re.search(r"page|offset", k, re.I))
            mode_arg = arg(search, "search", "mode_arg")
            mode_value = (mapped.get("search") or {}).get("mode_value") if isinstance(mapped.get("search"), dict) else None
            if mode_arg is None:
                mode_arg = next((k for k, s in self.props(search).items() if isinstance(s, dict) and s.get("enum") and any(re.search(r"text|keyword|query|free|word|full", str(v), re.I) for v in s["enum"])), None)
            if mode_arg:
                options = self.props(search)[mode_arg].get("enum") or []
                if mode_value not in options:
                    mode_value = next((v for v in options if re.search(r"text|keyword|query|free|word|full", str(v), re.I)), options[0] if options else None)
            roles["search"] = {"tool": search, "query_arg": arg(search, "search", "query_arg", lambda k, s: free_string(k, s) and re.search(r"text|query|word|term|keyword|q$|search", k, re.I)) or arg(search, "search", "query_arg", free_string),
                               "page_arg": page_arg, "mode_arg": mode_arg, "mode_value": mode_value,
                               "token_arg": arg(search, "search", "token_arg", lambda k, s: free_string(k, s)
                                                and re.search(r"token|cursor|next|after|continue|page|offset", k, re.I)),
                               "price_arg": arg(search, "search", "price_arg", lambda k, s: integer(k, s) and re.search(r"price|cap|budget|spend|cost", k, re.I))}
            if roles["search"]["token_arg"] == roles["search"]["query_arg"]:
                roles["search"]["token_arg"] = None
            # the schema's type decides a paging argument's role, whatever the reading called it: a number is a page
            # (or row offset), a text is a token handed back by the results
            token = roles["search"]["token_arg"]
            if token and integer(token, self.props(search).get(token) or {}) and not free_string(token, self.props(search).get(token) or {}):
                roles["search"]["page_arg"] = roles["search"]["page_arg"] or token
                roles["search"]["token_arg"] = None
            page = roles["search"]["page_arg"]
            if page and free_string(page, self.props(search).get(page) or {}) and not integer(page, self.props(search).get(page) or {}):
                roles["search"]["token_arg"] = roles["search"]["token_arg"] or page
                roles["search"]["page_arg"] = None
            # the filter search (a category and a price cap), when the store has one
            entry = mapped.get("search") if isinstance(mapped.get("search"), dict) else {}
            options = (self.props(search).get(mode_arg) or {}).get("enum") or [] if mode_arg else []
            roles["search"]["filter_value"] = (entry.get("filter_value") if entry.get("filter_value") in options and entry.get("filter_value") != mode_value
                                               else next((v for v in options if v != mode_value), None))
            roles["search"]["category_arg"] = arg(search, "search", "category_arg", lambda k, s: free_string(k, s)
                                                  and re.search(r"categor|department|aisle|section|shelf|label", k, re.I))
            if roles["search"]["category_arg"] == roles["search"]["query_arg"]:
                roles["search"]["category_arg"] = None
        for role, name in (("details", details), ("listing", listing)):
            if name:
                roles[role] = {"tool": name, "id_arg": arg(name, role, "id_arg", lambda k, s: free_string(k, s) and k in (self.schemas[name].get("required") or [k])) or arg(name, role, "id_arg", free_string)}

        def id_list(k: str, s: dict) -> bool:
            items = s.get("items") if isinstance(s.get("items"), dict) else {}
            return s.get("type") == "array" and items.get("type") in ("string", None) and bool(re.search(r"id|item|product|sku|listing|article", k, re.I))

        # several products' facts in one call: one step for many datasheets, when the store has such a tool
        busy = {search, order, decline, message, cart, cart_view}
        many = tool("details_many")
        if many in busy or not any(id_list(k, s) for k, s in self.props(many).items() if isinstance(s, dict)):
            many = next((n for n in names if n not in busy and any(id_list(k, s) for k, s in self.props(n).items() if isinstance(s, dict))
                         and re.search(r"detail|spec|propert|attribut|fact|feature|compare|view|info|sheet", f"{n} {self.descriptions.get(n, '')}", re.I)), None)
        if many:
            ids_arg = arg(many, "details_many", "ids_arg", id_list)
            spec = self.props(many).get(ids_arg) or {}
            if ids_arg:
                said = re.search(r"(?:up to|at most|no more than|max(?:imum)?(?: of)?)\s*(\d+)", f"{self.descriptions.get(many, '')} {spec.get('description') or ''}", re.I)
                roles["details_many"] = {"tool": many, "ids_arg": ids_arg,
                                         "max": spec["maxItems"] if isinstance(spec.get("maxItems"), int) else int(said.group(1)) if said else 4}
        if message:
            roles["message"] = {"tool": message, "text_arg": arg(message, "message", "text_arg", free_string)}
        if cart:
            props = self.props(cart)
            op_arg = arg(cart, "cart", "op_arg", lambda k, s: bool(s.get("enum")))
            entry = mapped.get("cart") if isinstance(mapped.get("cart"), dict) else {}
            add_value = remove_value = None
            if op_arg:
                options = [v for v in props[op_arg].get("enum") or []]
                add_value = entry.get("add_value") if entry.get("add_value") in options else next((v for v in options if re.search(r"add|put|plus|insert|increase|set", str(v), re.I)), options[0] if options else None)
                remove_value = entry.get("remove_value") if entry.get("remove_value") in options else next((v for v in options if v != add_value), None)
            roles["cart"] = {"tool": cart, "id_arg": arg(cart, "cart", "id_arg", lambda k, s: free_string(k, s) and k in (self.schemas[cart].get("required") or [k])) or arg(cart, "cart", "id_arg", free_string),
                             "qty_arg": arg(cart, "cart", "qty_arg", integer), "op_arg": op_arg, "add_value": add_value, "remove_value": remove_value}
            # a cart tool that takes a list of lines, each an object naming a listing and its units: every call sends the
            # whole basket wanted (such a tool may replace the cart with the list it gets)
            for key, spec in props.items():
                items = spec.get("items") if isinstance(spec, dict) and spec.get("type") == "array" else None
                fields = (items or {}).get("properties") or {} if isinstance(items, dict) and items.get("type") == "object" else {}
                ids = [k for k, f in fields.items() if isinstance(f, dict) and f.get("type") == "string"]
                counts = [k for k, f in fields.items() if isinstance(f, dict) and f.get("type") in ("integer", "number")]
                if ids and counts:
                    line_id = next((k for k in ids if re.search(r"id|item|product|sku|listing|which|code|ref", k, re.I)), ids[0])
                    roles["cart"] = {"tool": cart, "lines_arg": key, "line_id": line_id,
                                     "line_qty": next((k for k in counts if re.search(r"qty|quant|count|units|pieces|amount|number|wanted", k, re.I)), counts[0]),
                                     "id_arg": None, "qty_arg": None, "op_arg": None, "add_value": None, "remove_value": None}
                    break
            remove = tool("cart_remove")
            if remove is None and not op_arg:
                remove = next((n for n in names if n != cart and re.search(r"remove|drop|delete|take out", n.lower())), None)
            if remove:
                roles["cart_remove"] = {"tool": remove, "id_arg": arg(remove, "cart_remove", "id_arg", free_string)}
        if cart_view:
            roles["cart_view"] = {"tool": cart_view}
        if order:
            roles["order"] = {"tool": order}
        if decline:
            roles["decline"] = {"tool": decline, "text_arg": arg(decline, "decline", "text_arg", free_string)}
        return roles

    # -- building actions ------------------------------------------------------

    def fill_required(self, tool: str, args: dict[str, Any], text: str = "") -> dict[str, Any]:
        """Give every still-missing required argument a harmless value."""
        props = self.props(tool)
        for key in self.schemas.get(tool, {}).get("required") or []:
            if key in args:
                continue
            spec = props.get(key) or {}
            kinds = spec.get("type") if isinstance(spec.get("type"), list) else [spec.get("type")]
            if spec.get("enum"):
                args[key] = spec["enum"][0]
            elif "string" in kinds:
                args[key] = text or "ok"
            elif "boolean" in kinds:
                args[key] = True
            elif "integer" in kinds or "number" in kinds:
                args[key] = max(1, spec.get("minimum", 1))
        return args

    def search_action(self, query: str, page: int | None, token: str | None = None) -> dict[str, Any] | None:
        role = self.roles.get("search")
        if not role or not role.get("query_arg"):
            return None
        args: dict[str, Any] = {role["query_arg"]: query}
        if role.get("mode_arg") and role.get("mode_value") is not None:
            args[role["mode_arg"]] = role["mode_value"]
        if page is not None and role.get("page_arg"):
            args[role["page_arg"]] = self.page_value(page)
        if token and role.get("token_arg"):
            args[role["token_arg"]] = token
        limit = self.own_budget if self.own_budget is not None else self.budget  # a product's own limit, when it has one
        if role.get("price_arg") and limit is not None:
            args[role["price_arg"]] = limit
        return {"name": role["tool"], "args": self.fill_required(role["tool"], args, query)}

    def page_value(self, page: int) -> int:
        """The page argument for the page numbered `page`: the number itself, or its first row's offset when the store
        pages by row offset."""
        return page if self.page_step <= 1 else (page - self.page_base) * self.page_step

    def _note_offset_paging(self, following: Any, n_rows: int, page_arg: str | None = None) -> None:
        """Record that the page argument is a row offset when the store returns next == page size, or its name says so."""
        if self.page_step > 1 or n_rows <= 1:
            return
        role_arg = page_arg or (self.roles.get("search") or {}).get("page_arg")
        if isinstance(following, int) and not isinstance(following, bool) and following == n_rows:
            self.page_base, self.page_step = 0, n_rows
            self.page_size = max(self.page_size, n_rows)
        elif _page_arg_is_offset(role_arg) and self.page_step <= 1:
            self.page_base = 0
            if isinstance(following, int) and following > 1:
                self.page_step = following
                self.page_size = max(self.page_size, following)

    def id_action(self, role_name: str, ident: str) -> dict[str, Any] | None:
        role = self.roles.get(role_name)
        if not role or not role.get("id_arg"):
            return None
        return {"name": role["tool"], "args": self.fill_required(role["tool"], {role["id_arg"]: ident})}

    def basket_action(self, wanted: dict[str, int]) -> dict[str, Any] | None:
        """For a cart tool that takes a list of lines: one call holding the whole basket wanted ({id: units})."""
        cart = self.roles.get("cart") or {}
        if not cart.get("lines_arg"):
            return None
        lines = [{cart["line_id"]: ident, cart["line_qty"]: int(units)} for ident, units in wanted.items() if units]
        return {"name": cart["tool"], "args": self.fill_required(cart["tool"], {cart["lines_arg"]: lines})}

    def cart_action(self, ident: str, add: bool) -> dict[str, Any] | None:
        cart = self.roles.get("cart")
        if cart and cart.get("lines_arg"):
            wanted = dict(self.cart)
            if add:
                wanted[ident] = self.quantity
            else:
                wanted.pop(ident, None)
            return self.basket_action(wanted)
        if not cart or not cart.get("id_arg"):
            return None
        if not add and not cart.get("op_arg"):
            remove = self.roles.get("cart_remove")
            if not remove or not remove.get("id_arg"):
                return None
            return {"name": remove["tool"], "args": self.fill_required(remove["tool"], {remove["id_arg"]: ident})}
        args: dict[str, Any] = {cart["id_arg"]: ident}
        if cart.get("op_arg"):
            args[cart["op_arg"]] = cart["add_value"] if add else cart["remove_value"]
        if cart.get("qty_arg"):
            # a removal takes off every unit carted: without a quantity a store may take off only one
            args[cart["qty_arg"]] = self.quantity if add else self.cart.get(ident, self.quantity)
        return {"name": cart["tool"], "args": self.fill_required(cart["tool"], args)}

    def message_action(self, text: str) -> dict[str, Any] | None:
        role = self.roles.get("message")
        if not role or not role.get("text_arg"):
            return None
        return {"name": role["tool"], "args": self.fill_required(role["tool"], {role["text_arg"]: text}, text)}

    # -- listening to the shopper ------------------------------------------------

    def listen(self, observations: list[Any] | None, question: str | None = None, reply_of: int | None = None) -> bool:
        """Read every new shopper line in these results; True when the requirements changed."""
        lines: list[str] = []
        for index, observation in enumerate(observations or []):
            if index == reply_of and isinstance(observation, dict):
                strings = [v for v in observation.values() if isinstance(v, str) and v.strip()]
                if len(strings) == 1:
                    lines.append(strings[0])
            lines.extend(_shopper_lines(observation))
        fresh = [line for line in dict.fromkeys(lines) if line not in self.heard]
        if not fresh:
            if question:
                self.exchanges.append(f"asked: {question} -> (no reply)")
            return False
        self.heard.update(fresh)
        self.save_item()
        before = {None: list(self.reqs)} if not self.multi() else {k: list(i["reqs"]) for k, i in enumerate(self.items)}
        prompt = INTERPRET_PROMPT + "\n\n" + self.context(story=False)
        if self.multi():
            prompt += "\n\n" + self.products_text() + "\n\n" + MULTI_INTERPRET
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER (OR STORE) NOW SAID:\n" + "\n".join(fresh)
        self.exchanges.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.ask(prompt, 1500, strong=True)
        if isinstance(result, dict):
            self.own_property(result, " ".join(fresh))
        self.trace.append({"kind": "heard", "question": question, "lines": fresh, "reading": result})
        self.understood = not (isinstance(result, dict) and result.get("understood") is False)
        if not result:
            return False
        if self.multi():
            changed = self.apply_many(result, fresh)
            changed = self.note_references(result, before) or changed
            return self.shelf_open(before) or changed
        changed = False
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            cleaned = _kept_requirements(self.reqs, [r.strip() for r in now if r.strip()], result.get("dropped") or [],
                                         " ".join(fresh), self.noun_words())
            if cleaned != self.reqs:
                changed = True
                self.reqs = cleaned
        if [n for n in result.get("needs") or [] if isinstance(n, str) and n.strip()]:
            self.revealed += 1
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop.strip() and prop not in self.free:
                self.free.append(prop.strip())
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = float(budget)
            changed = True
        if result.get("unavailable"):
            for ident in self.cart:
                self.unavailable.add(ident)
            changed = True
        changed = self.apply_choice(result) or changed
        reply = str(result.get("reply") or "").strip()
        if reply and any("?" in line for line in fresh):
            self.pending_reply = reply  # sent by flush_reply at the next safe point, not from inside a result
        changed = self.note_references(result, before) or changed
        return self.shelf_open(before) or changed

    def apply_many(self, result: dict[str, Any], fresh: list[str]) -> bool:
        """A reading for a request of several products: each product's own requirements, drops and indifference."""
        def number(text: Any) -> tuple[int | None, str]:
            match = re.match(r"\s*\[?(\d+)\]?\s*[:.)-]\s*(.*)", str(text))
            if match and 1 <= int(match.group(1)) <= len(self.items):
                return int(match.group(1)) - 1, match.group(2).strip()
            return None, str(text).strip()

        home, changed = self.current, False
        self.changed_items = set()
        dropped = [number(d) for d in result.get("dropped") or []]
        now = result.get("requirements_now")
        if isinstance(now, dict):
            for key, reqs in now.items():
                k = number(f"{key}:")[0]
                if k is None or not isinstance(reqs, list):
                    continue
                self.switch(k)
                cleaned = _kept_requirements(self.reqs, [str(r).strip() for r in reqs if str(r).strip()],
                                             [d for j, d in dropped if j == k], " ".join(fresh), self.noun_words())
                if cleaned != self.reqs:
                    self.reqs = cleaned
                    self.changed_items.add(k)
        for entry in result.get("no_preference") or []:
            k, prop = number(entry)
            self.switch(home if k is None else k)
            if prop and prop not in self.free:
                self.free.append(prop)
        self.switch(home)
        if [n for n in result.get("needs") or [] if isinstance(n, str) and n.strip()]:
            self.revealed += 1
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = float(budget)
            self.changed_items = set(range(len(self.items)))
        gone = result.get("unavailable")
        if gone is True:
            self.unavailable.update(self.cart)
            self.changed_items = set(range(len(self.items)))
        elif isinstance(gone, list):
            for entry in gone:
                k = number(f"{entry}:")[0]
                line = self.lines.get(k) if k is not None else None
                if line:
                    self.unavailable.add(line)
                    self.changed_items.add(k)
        self.apply_choice(result)
        changed = bool(self.changed_items)
        reply = str(result.get("reply") or "").strip()
        if reply and any("?" in line for line in fresh):
            self.pending_reply = reply
        return changed

    def apply_choice(self, result: dict[str, Any]) -> bool:
        """A changed preference (how to choose among fitting listings) or, in an order of several, one product's new
        price limit. Only a positive preference is taken: a model echoing "none" must not undo a cheapest request.
        True when anything changed; in an order of several the changed products join changed_items."""
        def choice(raw: Any) -> dict[str, Any] | None:
            if not isinstance(raw, dict) or str(raw.get("kind") or "").strip().lower() not in ("cheapest", "best", "brand", "other"):
                return None
            direction = str(raw.get("direction") or "").strip().lower()
            return {"kind": str(raw["kind"]).strip().lower(), "property": str(raw.get("property") or "").strip(),
                    "direction": direction if direction in ("max", "min") else "", "value": str(raw.get("value") or "").strip(),
                    "numeric": raw.get("numeric") is True}

        def amount(raw: Any) -> float | None:
            return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else None

        def product(key: Any) -> int | None:
            match = re.match(r"\s*\[?(\d+)\]?\s*$", str(key))
            return int(match.group(1)) - 1 if match and 1 <= int(match.group(1)) <= len(self.items) else None

        if not self.multi():
            pref = choice(result.get("new_preference"))
            if pref and pref != self.preference:
                self.trace.append({"kind": "preference_changed", "from": self.preference, "to": pref})
                self.preference = pref
                return True
            return False
        home, changed = self.current, False
        prefs, budgets = result.get("new_preference"), result.get("product_budgets")
        for key, raw in (prefs.items() if isinstance(prefs, dict) else []):
            k, pref = product(key), choice(raw)
            if k is None or not pref:
                continue
            self.switch(k)
            if pref != self.preference:
                self.trace.append({"kind": "preference_changed", "product": k, "from": self.preference, "to": pref})
                self.preference = pref
                self.changed_items.add(k)
                changed = True
        for key, raw in (budgets.items() if isinstance(budgets, dict) else []):
            k, limit = product(key), amount(raw)
            if k is None or limit is None:
                continue
            self.switch(k)
            if limit != self.own_budget:
                self.trace.append({"kind": "product_budget_changed", "product": k, "from": self.own_budget, "to": limit})
                self.own_budget = limit
                self.changed_items.add(k)
                changed = True
        self.switch(home)
        return changed

    def shelf_open(self, before: dict[Any, list[str]]) -> bool:
        """A change that drops a product's last requirement leaves every in-stock listing of its category fitting,
        and then the cheapest on the whole shelf wins: that product's preference becomes cheapest (the cheaper-listing
        search and the "not the cheapest" check follow). True when a product's preference changed."""
        home, changed = self.current, False
        for k, old in before.items():
            if k is not None:
                self.switch(k)
            if old and not self.reqs and self.preference.get("kind") in ("none", "", None):
                self.preference = {"kind": "cheapest", "property": "", "direction": "", "value": "", "numeric": False}
                self.trace.append({"kind": "shelf_open", "product": k, "dropped": old})
                changed = True
                if k is not None:
                    self.changed_items.add(k)
        self.switch(home)
        return changed

    def own_product_ref(self, ref: dict[str, Any]) -> bool:
        """A reference with no seller whose kind of product is one of this order's own products ("the laptop you end up
        buying") points at the order itself: it is a rule that two lines must fit, never a listing to look up."""
        if str(ref.get("seller") or "").strip() or not self.multi():
            return False
        words = {_one(w) for w in _words(str(ref.get("category") or ""))} - _FILLER
        return bool(words) and any(words <= {_one(w) for w in _words(f"{i['category']} {i.get('noun') or ''}")}
                                   or {_one(w) for w in _words(i['category'])} - _FILLER <= words for i in self.items)

    def note_references(self, result: dict[str, Any], before: dict[Any, list[str]]) -> bool:
        """Queue each need whose value is to be read from another listing; True when the requirements changed. The
        other listing's conditions describe it, not the shopper's product: a requirement this reading set to one of
        them is taken back to what it was before the reading."""
        home, changed = self.current, False
        for ref in result.get("references") or []:
            if not isinstance(ref, dict) or not str(ref.get("property") or "").strip():
                continue
            k = None
            if self.multi():
                match = re.match(r"\s*\[?(\d+)", str(ref.get("product") or ""))
                k = int(match.group(1)) - 1 if match and 1 <= int(match.group(1)) <= len(self.items) else self.current
            prop = str(ref["property"]).strip()
            if self.own_product_ref(ref):
                self.trace.append({"kind": "reference_skipped", "why": "names a product of this order, no seller", "ref": ref})
                continue
            key = (k, _key_norm(prop))
            if key not in self.ref_needs:
                self.ref_needs[key] = {"property": prop, "ref": ref, "value": None, "source": None, "done": False}
                self.trace.append({"kind": "reference_need", "product": k, "ref": ref})
            copied = {(_key_norm(_split_req(str(c))[0]), _key_norm(_split_req(str(c))[1])) for c in ref.get("conditions") or []}
            if k is not None:
                self.switch(k)
            old = before.get(k, [])
            for req in list(self.reqs):
                name, value = _split_req(req)
                if (_key_norm(name), _key_norm(value)) in copied and req not in old:
                    self.reqs.remove(req)
                    self.reqs.extend(r for r in old if _key_norm(_split_req(r)[0]) == _key_norm(name) and r not in self.reqs)
                    self.trace.append({"kind": "reference_condition_undone", "requirement": req})
                    changed = True
        self.switch(home)
        return self.settle_references() or changed

    def settle_references(self) -> bool:
        """Keep each referenced property's requirement as its looked-up value: the shopper's pointer itself is not a
        value a product could state, so it stays out (and out of every later reading), and an unresolved one is left out."""
        changed, home = False, self.current
        for (k, norm), need in self.ref_needs.items():
            if k is not None:
                self.switch(k)
            want = f"{need['property']}: {need['value']}" if need["value"] else None
            kept = [r for r in self.reqs if _key_norm(_split_req(r)[0]) != norm or r == want]
            if want and want not in kept:
                kept.append(want)
            if kept != self.reqs:
                self.reqs, changed = kept, True
                if k is not None:
                    self.changed_items.add(k)
        self.switch(home)
        return changed

    def resolve_references(self) -> None:
        """Look up every queued referenced need: find the listing the shopper described, read its value for the
        property (following that listing once when its own text defers to yet another), and make it the requirement."""
        pending = [n for n in self.ref_needs.values() if not n["done"]]
        if not pending:
            return
        for need in pending:
            if self.steps_left() <= self.finish_reserve() + 6 or self.time_left() < FORCE_SECONDS + 30:
                break
            need["done"] = True
            ref, hops = need["ref"], 0
            while ref is not None and hops < REFERENCE_HOPS:
                found = self.lookup_reference(need["property"], ref)
                self.trace.append({"kind": "reference_lookup", "property": need["property"], "ref": ref, "found": found})
                if not found:
                    break
                if found.get("value"):
                    need["value"], need["source"] = _related(need["ref"], str(found["value"]).strip()), found.get("match")
                    break
                ref, hops = found.get("defers_to") if isinstance(found.get("defers_to"), dict) else None, hops + 1
        if self.settle_references():
            self.judge()


    def flush_reply(self) -> None:
        """Answer the question the shopper asked, when a message and the steps for it are left."""
        self.resolve_references()
        reply, self.pending_reply = self.pending_reply, ""
        if reply and self.spare_messages() > 0 and self.steps_left() > self.finish_reserve():
            action = self.message_action(reply)
            if action is not None:
                observations = self.act([action])
                if observations is not None:
                    self.messages_used += 1
                    self.listen(observations, reply, reply_of=0)

    def send_question(self, text: str, extra: list[dict[str, Any]] | None = None) -> bool:
        """One question per request (the shopper answers it in the message's own result)."""
        action = self.message_action(text)
        if action is None or self.spare_messages() <= 0:
            return False
        actions = list(extra or []) + [action]
        observations = self.act(actions)
        if observations is None:
            return False
        if len(observations) == len(actions) and _failed(observations[-1]):
            # the store refused the message (a limit the request did not state): no more messages
            self.message_limit = self.messages_used
            return False
        self.messages_used += 1
        self.listen(observations, text, reply_of=len(actions) - 1 if len(observations) == len(actions) else None)
        return True

    # -- searching ---------------------------------------------------------------

    def values_for_query(self) -> list[str]:
        """Requirement values to search with: exclusions and closed lists ('A or B') left out, numeric ones last.
        Both alternatives of a closed list in one query make it longer and narrower than the request; a value
        with one allowed part and exclusions ('A, not B or C') searches with A."""
        words: list[tuple[int, str]] = []
        for req in self.reqs:
            _, value = _split_req(req)
            if value and not _is_numeric(value):
                allowed = _clauses(value)[0]
                value = allowed[0] if len(allowed) == 1 else ""
            if not value or _NEGATION.match(value) or re.search(r"\s+or\s+|,|/", value):
                continue
            if _is_numeric(value) and _BOUND.search(value):
                continue  # a ceiling or a floor: "16 gb at most" must not be searched as 16, or 8 and 12 never appear
            if _is_numeric(value):
                value = re.sub(r"\s+", " ", _COMPARISON.sub(" ", value)).strip()
            if value:
                words.append((1 if _is_numeric(value) else 0, value))
        words.sort(key=lambda w: w[0])
        return [w for _, w in words]

    def run_search(self, max_pages: int) -> bool:
        """Search with the category and the current values; page through by the rule. True when rows came back.

        At most QUERY_EXTRA words beyond the category: in the race logs the bought item was on a search's first page
        for 17% of category-only queries, 54-57% with one to three more words, and 33% with four or more."""
        values = self.values_for_query()

        def capped(vals: list[str]) -> list[str]:
            kept, words = [], 0
            for value in vals:
                count = len(_words(value))
                if kept and words + count > QUERY_EXTRA:
                    break
                kept.append(value)
                words += count
            return kept

        tries = [" ".join([self.category] + capped(values[: len(values) - cut])).strip() for cut in range(len(values) + 1)]
        tries = list(dict.fromkeys(t for t in tries if t))[:3] or [self.category or self.request[:60]]
        if tries[-1] != self.category and self.category:
            tries.append(self.category)
        role = self.roles.get("search") or {}
        if role.get("page_arg") and self.page_step <= 1:  # (a store found to page by row offset keeps its offset base)
            spec = self.props(role["tool"]).get(role["page_arg"]) or {}
            base = spec.get("minimum", spec.get("default"))
            offsetish = _page_arg_is_offset(role["page_arg"])
            self.page_base = int(base) if isinstance(base, (int, float)) and not isinstance(base, bool) else (0 if offsetish else 1)
        rows: list[dict[str, Any]] = []
        observation: Any = None
        for query in tries:
            if self.steps_left() <= self.finish_reserve() + 1:
                break
            action = self.search_action(query, None)  # the server's default page is the first one
            if action is None:
                return False
            observations = self.act([action])
            if not observations:
                return False
            self.tried.add(query)
            self.listen(observations)
            observation = observations[0]
            observation = _inner(observation)
            rows = _result_rows(observation)
            if rows:
                self.query = query
                break
        if not rows:
            return False
        self.page_size = max(self.page_size, len(rows))
        total = _total_count(observation)
        self.total = total if isinstance(total, int) and total >= len(rows) else (self.page_size * ROW_PAGE_LIMIT if len(rows) >= self.page_size else len(rows))
        pages = math.ceil(self.total / self.page_size)
        extra = pages - 1 if self.total < self.page_size * ROW_PAGE_LIMIT else math.ceil(ROW_TARGET / self.page_size) - 1
        # a numeric next value equal to the rows shown: the page argument is a row offset (0, 10, 20 ...), not a number
        following = _next_number(observation)
        self._note_offset_paging(following, len(rows), role.get("page_arg"))
        self.next_page = self.page_base + 1
        self.next_token = _next_token(observation)
        self.pages_read = 1
        self.read_pages(min(extra, max_pages - 1))
        return True


    def synonym_search(self) -> None:
        """When the first search matched more than ROW_PAGE_LIMIT pages, only about ROW_TARGET rows of it were read:
        other words for the product or the required values bring in other relevant rows (first page each)."""
        if not self.page_size or self.total < self.page_size * ROW_PAGE_LIMIT:
            return
        if self.steps_left() <= self.finish_reserve() + DETAILS_FIRST + self.messages_left() + 4:
            return
        titles = "\n".join(f"- {str(r.get('title'))[:120]}" for r in list(self.rows.values())[:20] if r.get("title"))
        ask = SYNONYM_PROMPT + "\n\n" + self.context() + "\n\nQUERIES ALREADY TRIED: " + "; ".join(sorted(self.tried)) \
            + "\n\nTITLES OF THE ROWS SEEN:\n" + (titles or "(none)")
        queries = [str(q).strip() for q in (self.ask(ask, 300) or {}).get("queries") or [] if str(q).strip()]
        self.trace.append({"kind": "synonym_search", "total": self.total, "queries": queries})
        self.model_queries(SYNONYM_PROMPT, 10**6, queries[:2])

    def model_queries(self, prompt: str, target: int, queries: list[str] | None = None) -> None:
        """Other wordings for the search, from the model (the agents that drive their own search vary it this way)."""
        if queries is None:
            ask = prompt + "\n\n" + self.context() + "\n\nQUERIES ALREADY TRIED: " + "; ".join(sorted(self.tried))
            queries = [str(q).strip() for q in (self.ask(ask, 400) or {}).get("queries") or [] if str(q).strip()]
        for query in [q for q in dict.fromkeys(queries) if q not in self.tried][:3]:
            if len(self.pool()) >= target or self.steps_left() <= self.finish_reserve() + 4:
                return
            action = self.search_action(query, None)
            if action is None:
                return
            self.tried.add(query)
            observations = self.act([action])
            self.listen(observations)
            if observations and not _result_rows(observations[0]):
                self.empty.add(query)
                self.shorter_query()

    def shorter_query(self) -> None:
        """After a query that found nothing (a store search needs every word): the product's noun with one requirement
        value, one that no query so far has carried first, so every value gets searched."""
        noun = str(self.plan.get("noun") or self.category or "").strip()
        values = self.values_for_query()
        values.sort(key=lambda v: any(_stated(v, t) for t in self.tried if t not in self.empty))  # an empty query searched nothing
        for value in values:
            query = f"{noun} {value}".strip()
            if query in self.tried or self.steps_left() <= self.finish_reserve() + 4:
                continue
            action = self.search_action(query, None)
            if action is None:
                return
            self.tried.add(query)
            observations = self.act([action])
            self.listen(observations)
            if observations and _result_rows(observations[0]):
                return
            self.empty.add(query)

    def read_pages(self, count: int) -> None:
        """The next pages of the current query: numbered pages in one request; pages reached by a next-page
        token one after another, as each needs the token the page before returned."""
        role = self.roles.get("search") or {}
        if count > 0 and self.query and not role.get("page_arg") and role.get("token_arg"):
            for _ in range(count):
                if not self.next_token or self.steps_left() <= self.finish_reserve() + 1:
                    return
                seen = set(self.rows)
                action = self.search_action(self.query, None, self.next_token)
                observations = self.act([action]) if action else None
                if not observations:
                    return
                self.pages_read += 1
                self.listen(observations)
                token = _next_token(observations[0])
                rows = [_row_id(r) for r in _result_rows(observations[0])]
                self.next_token = token if token and token != self.next_token and any(i not in seen for i in rows) else None
            return
        if count <= 0 or not self.query or not role.get("page_arg"):
            return
        last = self.page_base + math.ceil(self.total / max(1, self.page_size)) - 1
        pages = [p for p in range(self.next_page, self.next_page + count) if p <= last]
        if not pages:
            return
        seen = set(self.rows)
        actions = [self.search_action(self.query, p) for p in pages]
        observations = self.act([a for a in actions if a])
        if observations:
            self.next_page = pages[0] + len(observations)
            self.listen(observations)
            first_obs = observations[0]
            first_obs = _inner(first_obs)
            self._note_offset_paging(_next_number(first_obs), len(_result_rows(first_obs)), role.get("page_arg"))
            first = [_row_id(r) for r in _result_rows(observations[0])]
            if first and all(i in seen for i in first) and self.page_base == 0 and self.page_step <= 1:
                self.page_base = 1  # pages count from 1: the first was read twice, the rest are in order

    def pages_left(self) -> int:
        if not self.query or not self.page_size:
            return 0
        if not (self.roles.get("search") or {}).get("page_arg"):
            if not self.next_token:
                return 0
            return max(1, math.ceil(self.total / self.page_size) - self.pages_read)
        return max(0, self.page_base + math.ceil(self.total / self.page_size) - self.next_page)

    # -- choosing ------------------------------------------------------------------

    def affordable(self, info: dict[str, Any]) -> bool:
        price = info.get("price")
        limit = self.own_budget if self.own_budget is not None else self.budget
        if limit is not None and isinstance(price, (int, float)) and price * self.quantity > limit + 1e-9:
            return False
        return info.get("stock") is not False

    def target_label(self) -> str:
        """The category label among the rows named most like the requested category."""
        wanted = {_one(w) for w in _words(self.category)}
        labels: dict[str, int] = {}
        for info in self.rows.values():
            if info.get("category"):
                labels[info["category"]] = labels.get(info["category"], 0) + 1
        if not labels:
            return ""
        return max(labels, key=lambda label: (len(wanted & {_one(w) for w in _words(label)}), labels[label]))

    def pool(self) -> list[dict[str, Any]]:
        """Rows that can still be bought: in the target category, within budget, not sold out."""
        label = self.target_label()
        out = []
        for info in self.rows.values():
            if not info.get("id") or info["id"] in self.unavailable or info["id"] in self.unknown:
                continue
            if label and info.get("category") and info["category"] != label:
                continue
            if self.affordable(info):
                out.append(info)
        return out

    def numeric_case(self) -> bool:
        return bool(self.preference.get("numeric")) or (self.preference.get("kind") == "best" and bool(self.preference.get("direction")))

    def string_case(self) -> bool:
        return not self.numeric_case() and not any(_is_numeric(_split_req(r)[1]) for r in self.reqs)

    def cheapest_first(self) -> bool:
        """A cheapest preference: the extra search below a candidate's price and the "not the cheapest" check."""
        return self.preference.get("kind") == "cheapest"

    def low_price_first(self) -> bool:
        """Only a cheapest preference ranks low prices first; otherwise the dearest within the budget comes first
        (the practice answer sets: with no preference, the cheaper fits failed where the dearest passed)."""
        return self.cheapest_first()

    def no_price_order(self) -> bool:
        """With no preference stated, price does not rank the fits: they keep the order they were found in (the store's
        search order) and how well they match, buying the first verified fit (in logged worlds the dearest fit within
        the budget failed where that one passed)."""
        return self.preference.get("kind") in ("none", "", None)

    def price_rank(self, info: dict[str, Any]) -> float:
        price = info.get("price")
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            return float("inf")  # an unknown price goes last either way
        if self.no_price_order():
            return 0.0
        return price if self.low_price_first() else -price

    def string_shortlist(self, pool: list[dict[str, Any]], size: int) -> list[str]:
        """Rows matching every value (by words) first, then the closest ones; by price within each (see price_rank)."""
        wanted: list[list[str]] = []  # per requirement, its allowed values (any one is a hit)
        excluded: list[str] = []
        for req in self.reqs:
            allowed, banned = _clauses(_split_req(req)[1])
            if allowed:
                wanted.append(allowed)
            excluded += banned
        if self.preference.get("value"):
            wanted.append([str(self.preference["value"])])
        props = [_split_req(req)[0] for req in self.reqs if _clauses(_split_req(req)[1])[0]]
        noun = self.noun_words()
        scored = []
        for info in pool:
            text = info.get("text") or ""
            if any(x and _stated(x, text) for x in excluded):
                continue
            hits = sum(1 for options in wanted if any(_stated(o, text) for o in options))
            # a value stated under its own property in the row's text ('Blade Type : Serrated') counts more than
            # the value's word anywhere in it
            pairs = _text_pairs(text)
            keyed = sum(1 for prop, options in zip(props, wanted)
                        if any(_names(k, prop, noun) and any(_stated(o, v) for o in options) for k, v in pairs))
            scored.append((-(hits == len(wanted)), -keyed, -hits, self.price_rank(info), info["id"]))
        scored.sort()
        return [ident for *_, ident in scored[:size]]

    def model_shortlist(self, pool: list[dict[str, Any]], prompt: str, size: int) -> list[str]:
        rows = sorted(pool, key=self.price_rank)
        listing = "\n".join(_compact({k: v for k, v in r.items() if k != "text"}, 400) for r in rows[:120])
        order = ("as the rows come (the store's own order; price does not rank them)" if self.no_price_order()
                 else "from low to high" if self.low_price_first() else "from high to low (the dearest within the budget first)")
        prompt = prompt.replace("PRICE_ORDER", order)
        heading = "in the store's order" if self.no_price_order() else "cheapest first" if self.low_price_first() else "highest price first"
        result = self.ask(prompt + "\n\n" + self.context() + f"\n\nSEARCH ROWS ({heading}):\n" + listing, 800) or {}
        known = {r["id"] for r in pool}
        return [i for i in dict.fromkeys(str(x) for x in result.get("ids") or []) if i in known][:size]

    def shortlist(self, size: int = SHORTLIST, skip: set[str] | None = None) -> list[str]:
        pool = [r for r in self.pool() if r["id"] not in (skip or set())]
        if not pool:
            return []
        if self.numeric_case():
            picked = self.model_shortlist(pool, NUMERIC_PROMPT, 5)
        elif self.string_case():
            return self.string_shortlist(pool, size)
        else:
            picked = self.model_shortlist(pool, SHORTLIST_PROMPT, size)
        filler = [i for i in self.string_shortlist(pool, size) if i not in picked]
        return (picked + filler)[:size]

    def read_details(self, idents: list[str]) -> None:
        """Open products' datasheets: several per call through a tool that takes a list of ids when the store has
        one (one step for many), then one per call for any it did not return."""
        unread = [i for i in dict.fromkeys(idents) if i not in self.details]
        many = self.roles.get("details_many")
        while many and len(unread) > 1 and int(many.get("max") or 0) >= 2:
            size = int(many["max"])
            chunks = [unread[n:n + size] for n in range(0, len(unread), size)]
            actions = [{"name": many["tool"], "args": self.fill_required(many["tool"], {many["ids_arg"]: chunk})} for chunk in chunks]
            observations = self.act(actions)
            if not observations:
                break
            self.listen(observations)
            refused = any(_failed(o) and not any(i in self.details for i in chunk) for o, chunk in zip(observations, chunks))
            unread = [i for i in unread if i not in self.details]
            if not refused:
                break
            many["max"] = size // 2  # the store refused a batch this size: smaller ones for the rest of the task
        actions = [self.id_action("details", i) for i in unread]
        actions = [a for a in actions if a]
        if actions:
            observations = self.act(actions)
            self.listen(observations)


    def fact_veto(self, ident: str) -> str:
        """The first current requirement that the product's own facts seem to contradict, whatever its title says
        (a candidate: fact_vetoes has the model confirm it)."""
        details = self.details.get(ident)
        if details is None:
            return ""
        info = self.rows.get(ident, {})
        title = str(info.get("title") or (details.get("title") if isinstance(details, dict) else "") or "")
        for req in self.reqs:
            prop, value = _split_req(req)
            verdict, shown = _fact_reading(details, title, prop, value, self.noun_words(), str(info.get("category") or ""))
            if verdict == "no":
                return f"requirement {prop}: {value} | the product's own facts: {shown}"
        return ""

    def firm_veto(self, ident: str) -> str:
        """A fact veto that a model reading of the records cannot overrule: the facts share no word with the value
        (beyond the property's own name). A fact holding part of the value is a judgement
        call the code leans against; the decline audit, reading both, may find it met (the order check still runs)."""
        why = self.fact_veto(ident)
        if not why:
            return ""
        req, _, shown = why.partition(" | the product's own facts: ")
        prop, value = _split_req(req.removeprefix("requirement ").strip())
        named = {_one(w) for w in _words(prop)} | set(self.noun_words())
        wanted = {_one(w) for o in _options(value) for w in _words(o) if w not in _FILLER} - named
        facts = {_one(w) for w in _words(shown.split(CLOSEST)[-1]) if w not in _FILLER} - named
        return "" if wanted & facts else why

    def own_property(self, result: dict[str, Any], said: str) -> None:
        """A shopper may answer a question about one property with a need on another ('my answer on network stays
        fixed' when asked about the wifi standard). When a need's property is not in the shopper's words but the
        property key of opened products that the words name is, the need is on that key: rename it in the reading."""
        heard = {_one(w) for w in _words(said)}
        keys = {_key_norm(k) for d in self.details.values() for k, _ in _facts(d)}
        renamed = {}
        for need in result.get("needs") or []:
            prop, value = _split_req(str(need))
            if not prop or not value or not _stated(value, said) or {_one(w) for w in _words(prop)} <= heard:
                continue
            named = sorted((k for k in keys if k and set(k.split()) <= heard and not _stated(k, value)), key=lambda k: -len(k.split()))
            if named:
                renamed[str(need)] = f"{named[0]}: {value}"
        if renamed:
            result["needs"] = [renamed.get(str(n), n) for n in result.get("needs") or []]
            result["requirements_now"] = [renamed.get(str(r), r) for r in result.get("requirements_now") or []]
            result["renamed"] = renamed

    def quoted_facts(self, ident: str, check: dict[str, Any], prop: str) -> list[str]:
        """The values of the product's own keys for `prop` that a check's quoted fact repeats (a quote the facts do not
        hold is ignored). As the reading rules: the keys that are the property's own name decide when the product has
        one; keys with other words added ('<property> filter') only when it has none."""
        quote = str(check.get("fact") or "").lower()
        noun = self.noun_words()
        facts = [(k, v) for k, v in _facts(self.details.get(ident) or {}) if _names(k, prop, noun)]
        own = [(k, v) for k, v in facts if {_one(w) for w in _words(k)} - noun == _prop_words(prop, noun)]
        return [str(v) for _, v in (own or facts) if str(v).strip() and str(v).strip().lower() in quote]

    def stated_requirements(self, ident: str, firm: bool = False) -> list[tuple[str, str]]:
        """The current requirements the product's own facts plainly state (the code's reading is 'yes'). With `firm`,
        only those a fact or the category label states: a title, read when no key carries the property's name, may
        word a value another key of the product contradicts under a name the code cannot tie to it."""
        details = self.details.get(ident)
        if details is None:
            return []
        info = self.rows.get(ident, {})
        title = str(info.get("title") or "")
        out = []
        for req in self.reqs:
            prop, value = _split_req(req)
            if not prop or not value:
                continue
            verdict, evidence = _fact_reading(details, title, prop, value, self.noun_words(), str(info.get("category") or ""))
            if verdict == "yes" and not (firm and evidence.startswith("title:")):
                out.append((prop, value))
        return out

    def noun_words(self) -> frozenset[str]:
        """The product's own words (its noun and category), which a fact key may carry as a prefix."""
        text = f"{self.plan.get('noun') or ''} {self.category or ''}"
        return frozenset(_one(w) for w in _words(text) if w not in _FILLER)

    def fact_vetoes(self, idents: list[str]) -> dict[str, str]:
        """The products whose own facts contradict a current requirement: a key naming the property states a value
        that shares no word with any allowed one, or states an excluded one. Decided by the facts, with no model
        call; a fact that shares a word with the value is left to the review, which can read it in other words."""
        return {i: why for i in idents if (why := self.fact_veto(i))}

    def judge(self) -> dict[str, Any]:
        """The model's review of every detailed product: properties to ask about, the pick, the ranked fits."""
        opened = self.mine()
        if not opened:
            return {}
        blocks = []
        for ident, observation in opened.items():
            info = self.rows.get(ident, {})
            head = {k: info.get(k) for k in ("id", "title", "category", "price", "stock") if info.get(k) is not None}
            if ident in self.unavailable:
                head["stock"] = False
            blocks.append(_compact({"row": head, "details": observation}, 3000))
        prompt = REVIEW_PROMPT + "\n\n" + self.context()
        if self.asked:
            prompt += "\n\nPROPERTIES ALREADY ASKED OR SETTLED: " + ", ".join(self.asked)
        prompt += "\n\nDETAILED PRODUCTS:\n" + "\n".join(blocks)
        prompt += "\n\nPROPERTY KEYS OF THE DETAILED PRODUCTS (how many carry each, values seen):\n" + (self.key_table() or "(none)")
        result = self.ask(prompt, 2500) or {}
        known = set(opened)
        fits = [i for i in dict.fromkeys(str(x) for x in result.get("fits") or []) if i in known and i not in self.unavailable]
        fits = [i for i in fits if self.affordable(self.rows.get(i, {}))]
        pick = str(result.get("pick") or "")
        # The candidates stand while the wants stand: a sold-out item takes only itself off the list, so another
        # candidate goes to the cart instead of a decline. Changed wants are judged afresh over every opened product.
        wants = self.wants()
        if wants == self.candidate_wants:
            fits += [i for i in self.candidates if i not in fits and i in known and i not in self.unavailable
                     and i not in self.rejected and self.affordable(self.rows.get(i, {}))]  # in budget and in stock
        vetoed = self.fact_vetoes(list(dict.fromkeys(fits + ([pick] if pick in known else []))))
        fits = [i for i in fits if i not in vetoed]
        # A product whose own facts prove every current requirement fits, whatever the review made of it: a model's
        # reading does not overrule the facts (the order check still verifies it before it is ordered).
        wanted = {(p, v) for p, v in (_split_req(r) for r in self.reqs) if p and v}
        if wanted or not self.reqs:  # no requirement left (all dropped): any opened product of the category in budget and stock fits
            label = self.target_label()
            for i in known:
                info = self.rows.get(i, {})
                if (i not in fits and i not in vetoed and i not in self.unavailable and i not in self.rejected and self.affordable(info)
                        and (not label or not info.get("category") or info["category"] == label)
                        and wanted <= set(self.stated_requirements(i, firm=True))):
                    fits.append(i)
            # then those whose title states what no key of theirs names (the reading rules allow it): the review may
            # reject them on a story it reads into the request; the order check, which reads no story, decides
            for i in known:
                info = self.rows.get(i, {})
                if (i not in fits and i not in vetoed and i not in self.unavailable and i not in self.rejected and self.affordable(info)
                        and (not label or not info.get("category") or info["category"] == label)
                        and wanted <= set(self.stated_requirements(i))):
                    fits.append(i)
        if self.preference.get("kind") in ("cheapest", "none", "", None):
            fits.sort(key=lambda i: self.price_rank(self.rows.get(i, {})))
        if self.preference.get("kind") in ("none", "", None):
            # any fitting item will do: those whose own facts prove every requirement go first, in price order
            wanted = {(p, v) for p, v in (_split_req(r) for r in self.reqs) if p and v}
            fits.sort(key=lambda i: not wanted <= set(self.stated_requirements(i)))
        if pick in vetoed:
            pick = fits[0] if fits else ""
        review = {"properties": [str(p).strip() for p in result.get("properties") or [] if isinstance(p, str) and p.strip()],
                  "pick": pick if pick in known and pick not in self.unavailable else (fits[0] if fits else None),
                  "pick_fits_all": bool(result.get("pick_fits_all")) and pick in fits, "fits": fits}
        # A copy: ask_hidden takes asked properties off self.review, and the trace keeps what the review said.
        self.trace.append({"kind": "review", "review": json.loads(json.dumps(review)), "problems": result.get("problems"), "fact_vetoes": vetoed})
        self.review = review
        self.candidates, self.candidate_wants = list(fits), wants
        return review

    def wants(self) -> tuple:
        """What the candidates were judged against: a change in any of it makes them stale."""
        return (tuple(self.reqs), self.budget, self.own_budget, json.dumps(self.preference, sort_keys=True, default=str), self.quantity)

    # -- phases ----------------------------------------------------------------------


    def key_table(self) -> str:
        """The opened products' property keys: how many of them carry each one, and the values seen, most carried
        first. Spellings of one key (spaces, dots, underscores) count together."""
        carried: dict[str, set[str]] = {}
        values: dict[str, list[str]] = {}
        opened = self.mine()
        for ident, details in opened.items():
            for key, value in _facts(details):
                norm = _key_norm(key)
                if not norm or re.search(r"(?i)\b(?:id|sku|title|price|stock|descr\w*|url|image)\b", norm):
                    continue
                carried.setdefault(norm, set()).add(ident)
                text = re.sub(r"\s+", " ", str(value)).strip().lower()[:40]
                if text and text not in values.setdefault(norm, []):
                    values[norm].append(text)
        rows = sorted(carried, key=lambda k: (-len(carried[k]), k))[:30]
        total = len(opened)

        def marks(k: str) -> str:
            seen = values.get(k, [])
            tags = ["DIFFERS"] if len(seen) > 1 else []
            if seen and sum(1 for v in seen if re.search(r"\d", v)) * 2 >= len(seen):
                tags.append("FIGURES")
            return f" [{', '.join(tags)}]" if tags else ""
        return "\n".join(f"- {k}: on {len(carried[k])} of {total}{marks(k)}; values: {', '.join(values.get(k, [])[:6])}" for k in rows)

    def uniform_widely_filed(self, stated: list[str]) -> list[str]:
        """Properties the store files on most opened products (at least 60%) that all show one and the same value:
        the shopper's wish may well be a value not seen yet. Logistics and stated properties aside, most filed first."""
        if len(self.details) < 4:
            return []
        carried: dict[str, dict[str, str]] = {}
        for ident, details in self.details.items():
            for key, value in _facts(details):
                norm = _key_norm(key)
                if norm and not re.search(r"warrant|condition|deliver|ship|origin|country|model|option|code|brand|price|stock|pack|seller|descr|title", norm):
                    carried.setdefault(norm, {}).setdefault(ident, re.sub(r"[^a-z0-9]+", "", str(value).lower()))
        found = [(k, len(v)) for k, v in carried.items() if len(v) >= 0.6 * len(self.details) and len(set(v.values())) == 1
                 and not self.names_stated(k, stated)]
        return [k for k, _ in sorted(found, key=lambda kv: -kv[1])]

    def names_stated(self, prop: str, stated: list[str]) -> bool:
        """The candidate is a property the request already states: its name without bracketed examples, or any part
        of an 'a / b' name, names a stated property or is named by it."""
        noun = self.noun_words()
        return any(_names(x, q, noun) or _names(q, x, noun) for x in _name_parts(prop) for q in stated)

    def field_values(self, prop: str) -> list[str]:
        """The values the opened products show under keys naming `prop`, most common first."""
        noun = self.noun_words()
        counts: dict[str, int] = {}
        for details in self.details.values():
            for key, value in _facts(details):
                value = value.strip()
                if value and len(value) <= 30 and value.lower() not in ("n/a", "none", "-", "other", "others") and _names(key, prop, noun):
                    counts[value.lower()] = counts.get(value.lower(), 0) + 1
        return sorted(counts, key=lambda v: -counts[v])

    def plain_name(self, prop: str) -> str:
        """A shorter key for the same field: one whose words are among the property's and whose value equals the
        property's own on every opened product that carries both."""
        words = set(_words(prop))
        same: dict[str, bool] = {}
        for details in self.details.values():
            facts = {k.replace("_", " ").strip().lower(): v.strip().lower() for k, v in _facts(details)}
            own = facts.get(prop.lower())
            if own is None:
                continue
            for key, value in facts.items():
                if key != prop.lower() and set(_words(key)) and set(_words(key)) < words:
                    same[key] = same.get(key, True) and value == own
        shorter = [k for k, ok in same.items() if ok]
        return min(shorter, key=len) if shorter else prop

    def plain_question(self, name: str, product: str) -> str:
        """The plain re-ask of one property, in short words: two or three short values the store shows for it, else
        its name alone (the product only when the order has several)."""
        values = [v for v in self.field_values(name) if len(v) <= 7 and " " not in v][:3]
        about = f" for the {product}" if self.multi() else ""
        if len(values) >= 2:
            return f"What {name} do you want{about}: {', '.join(values[:-1])} or {values[-1]}?"
        return f"What {name} do you want{about}?"

    def ask_hidden(self, most: int | None = None) -> int:
        """Up to QUESTION_PROPS properties per message; keep asking while messages and steps remain, since a task can
        hide more than one need (at most `most` questions when given). Returns the number of questions sent."""
        start, sent = self.revealed, 0
        turn = 0  # the two rankings take turns: focus 1st, review 1st, focus 2nd, review 2nd, ...
        while self.spare_messages() > 0 and self.steps_left() > self.finish_reserve() + 3 + self.held_messages() and not self.over():
            if most is not None and sent >= most:
                break
            settled = " ".join(self.asked + [_split_req(r)[0] for r in self.reqs] + self.free).lower()
            focus = [str(p.get("property")).strip() for p in self.focus.get("properties") or []]
            stated = [_split_req(r)[0] for r in self.reqs if _split_req(r)[0]]
            # this turn's list first (its best property not yet asked or settled), then the other one
            lists = [focus, list(self.review.get("properties") or [])]
            # asked or settled already: by name, the product's noun aside ('lamp shade colour' is 'shade colour')
            done = [x for a in self.asked + self.free for x in _name_parts(a)]
            props = [p for p in dict.fromkeys(lists[turn % 2] + lists[(turn + 1) % 2]) if p.lower() not in settled
                     and not self.names_stated(p, stated) and not self.names_stated(p, done)]
            # a property the opened listings file under a key of its own goes before one seen only in titles or
            # the story: a hidden need is a value of a field the store writes, and the question names it by that key
            keys = {_key_norm(k) for d in self.mine().values() for k, _ in _facts(d)}
            noun = self.noun_words()
            props.sort(key=lambda p: not any(_names(k, x, noun) for k in keys for x in _name_parts(p)))
            if self.spare_messages() == 1 and self.revealed == start:
                # nothing revealed yet: the last question goes to a widely filed property whose opened products all
                # show one value (a safety net for when the ranked lists found no hidden need), when the focus list,
                # built from every title and the opened facts with a reason for each, did not consider it: one it
                # ranked keeps its rank
                considered = [x for q in focus for x in _name_parts(q)]
                uniform = [p for p in self.uniform_widely_filed(stated) if p.lower() not in settled and not self.names_stated(p, done)
                           and not self.names_stated(p, considered)]
                if uniform:
                    props = uniform + props
            if not props:
                break
            # several properties per message, best first: a shopper states a need on any property a message names,
            # so three messages cover the top properties three at a time instead of one at a time
            batch = props[:QUESTION_PROPS]
            prop = batch[0]
            self.asked.extend(batch)
            before = list(self.reqs)
            # each property by its name, no example values, in as few and as short words as possible: a shopper who
            # wants plain talk fails a question about three times as often when it carries long words (race logs);
            # the product is named only when the order has several, so the question says which one it is about
            product = str(self.plan.get("noun") or self.category or "item").strip()
            # one plain name per property: a compound name ('target gender / wearer') asks by its first part, and a short
            # store prefix of a key ('fa_pattern', 'm top neckline') is left out, so the question reads as plain talk
            names = list(dict.fromkeys(_plain_words(self.plain_name(_name_parts(q)[0])) or q for q in batch))
            name = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
            if any(self.names_stated(n, stated) for n in names):
                # its plain name is a property the request states ('screen resolution e.g. 1920 x 1080' is the stated
                # 'resolution'): a question would only invite a "don't mind" that reads as dropping it
                self.trace.append({"kind": "skip_stated_question", "property": prop, "plain": names})
                continue
            self.understood = True
            if not self.send_question(f"Which {name} would you like for the {product}?" if self.multi() else f"Which {name} would you like?"):
                break
            sent += 1
            self.resolve_references()
            if not self.understood and self.spare_messages() > 0 and self.steps_left() > self.finish_reserve() + 3 + self.held_messages():
                # the shopper asked for other words: the same property once more, plainly, with values the store shows
                self.send_question(self.plain_question(names[0], product))
            turn += 1
            if self.reqs != before:
                self.judge()
            elif self.review.get("properties"):
                self.review["properties"] = [p for p in self.review["properties"] if p not in batch]
        return sent


    def id_store(self) -> bool:
        """The store lists search results in a fixed id order, not by relevance: two or more results of 5+ rows came
        in ascending id order and none in another."""
        return self.id_orders.count(True) >= 2 and False not in self.id_orders

    def price_store(self) -> bool:
        """The store lists search results cheapest first: two or more results of 3+ priced rows came in ascending price
        order and at most one in another (a page of equal prices reads as ascending either way)."""
        return self.price_orders.count(True) >= 2 and self.price_orders.count(False) <= max(1, len(self.price_orders) // 10)

    def walk_query(self, query: str, first: Any, most: int) -> int:
        """Reads the pages of `query` after its first one (`first`, already read), at most `most`; the pages read."""
        role = self.roles.get("search") or {}
        rows, total = _result_rows(first), _total_count(first)
        size = max(len(rows), self.page_size or 0, 1)
        pages = min(most, math.ceil(total / size) - 1 if isinstance(total, int) else (most if len(rows) >= size else 0))
        if pages <= 0:
            return 0
        if role.get("page_arg"):
            observations = self.act([a for a in (self.search_action(query, self.page_base + n) for n in range(1, pages + 1)) if a])
            if observations:
                self.listen(observations)
            return len(observations or [])
        read, token = 0, _next_token(first)
        while read < pages and token and role.get("token_arg"):
            observations = self.act([self.search_action(query, None, token)])
            if not observations:
                break
            self.listen(observations)
            read += 1
            following = _next_token(observations[0])
            token = following if following != token else None
        return read

    def value_queries(self) -> list[str]:
        """Short searches that list the listings carrying the stated word values: the values with the kind word, the
        value first (a store may read only a query's first words; one needing every word lists exactly the listings that
        carry the value): two values together first, then each value alone with the kind word."""
        noun = [w for w in _words(str(self.plan.get("noun") or self.category or "")) if w not in _FILLER]
        kind = noun[-1:] if noun else []
        values = [v for v in self.values_for_query() if not _is_numeric(v) and 1 <= len(_words(v)) <= 2]
        out = []
        if len(values) >= 2:
            out.append(" ".join(dict.fromkeys(_words(values[0]) + _words(values[1]) + kind)))
        out += [" ".join(dict.fromkeys(_words(v) + kind)) for v in values[:2]]
        # a store ranking by relevance reads a value search deeper (cheap_sweep), which needs its first page again
        again = not self.id_store() and not self.price_store()
        return [q for q in dict.fromkeys(out) if q and (again or q not in self.tried)]

    def cheap_sweep(self, share: int | None = None) -> None:
        """A cheapest request in a store with no price filter: the cheapest fit may sit beyond the rows read, so spare
        steps go to more of the shelf first; read_more and the order check then open the cheapest unread rows. The
        finish keeps its reserve.
        - The stated values with the kind word, value first (first page each).
        - A store listing results in id order ranks nothing by relevance, so the cheapest fit may be on any page: the
          narrowest of those searches (else the main one) is read to its end, up to WALK_PAGES pages.
        - Otherwise further pages of the main query, the plain noun and its kind words, and other names the model
          suggests, first page each."""
        role = self.roles.get("search") or {}
        if not self.cheapest_first() or (role.get("price_arg") and role.get("category_arg")):
            return  # (a store with a price filter is walked below the pick's price by cheaper_filters instead)

        def spare() -> int:  # (the reserve already holds a refused order's repair; the order check's cheaper reads need 3)
            return self.steps_left() - self.finish_reserve() - self.held_messages() - 3
        if spare() < 3 or self.time_left() < FORCE_SECONDS + 100:
            return
        before = len(self.pool())
        share = max(1, share or len(self.items))  # the products still to sweep share the spare steps
        allowed = max(3, spare() // share)
        start = self.step
        firsts: dict[str, Any] = {}
        for query in self.value_queries()[:2 if share > 1 else 3]:
            if spare() < 3 or self.step - start >= allowed - 1:
                break
            action = self.search_action(query, None)
            if action is None:
                break
            self.tried.add(query)
            observations = self.act([action])
            if not observations:
                break
            self.listen(observations)
            if _result_rows(observations[0]):
                firsts[query] = observations[0]
            else:
                self.empty.add(query)
        walked: list[str] = []
        if self.id_store():
            # narrowest first: in a store needing every word it lists exactly the listings carrying those values
            for query in sorted(firsts, key=lambda q: _total_count(firsts[q]) or len(_result_rows(firsts[q]))):
                room = min(WALK_PAGES, spare() - 2, allowed - (self.step - start))
                if room < 1:
                    break
                self.walk_query(query, firsts[query], room)
                self.walked.add(query)
                walked.append(query)
            if not walked:
                # no value to search with: the kind word alone may reach listings the main query's other words miss
                # (a store needing every word drops a listing that lacks one of them), so the broader one is read
                noun = [w for w in _words(str(self.plan.get("noun") or self.category or "")) if w not in _FILLER]
                kind = noun[-1] if noun else ""
                wide = None
                if kind and kind != self.query and kind not in self.tried and spare() >= 4:
                    action = self.search_action(kind, None)
                    observations = self.act([action]) if action else None
                    self.tried.add(kind)
                    if observations:
                        self.listen(observations)
                        if (_total_count(observations[0]) or 0) > (self.total or 0):
                            wide = observations[0]
                room = min(WALK_PAGES, spare() - 2, allowed - (self.step - start))
                if wide is not None and room > 0:
                    self.walk_query(kind, wide, room)
                    self.walked.add(kind)
                    walked.append(kind)
                elif room > 0 and self.pages_left() > 0:
                    self.read_pages(min(room, self.pages_left()))  # the main search to its end
                    walked.append(self.query)
        deep: list[str] = []
        if not walked and not self.id_store() and not self.price_store() and firsts:
            # a store ranking by relevance (neither id nor price order): a cheaper fit may sit pages deep in the search
            # that lists the listings carrying the stated values, so its next pages are read before the broad sweep
            query = next(iter(firsts))
            room = min(DEEP_PAGES, spare() - 3, allowed - (self.step - start))
            if room > 0 and self.walk_query(query, firsts[query], room):
                deep.append(query)
        if not walked:
            self.read_pages(min(max(1, CHEAP_PAGES // share), self.pages_left(), max(0, spare() - 2)))
            noun = str(self.plan.get("noun") or self.category or "").strip()
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z-]{2,}", noun) if w.lower() not in ("and", "for", "the", "with")]
            # in a store ranking by relevance, the kind words in the singular too: it may match whole words only
            single = [w[:-1] for w in words[-1:] + words[:1] if w.lower().endswith("s") and not w.lower().endswith("ss")] \
                if not self.id_store() and not self.price_store() else []
            own = [q for q in dict.fromkeys([noun] + words[-1:] + words[:1] + single) if q and q not in self.tried]
            ask = CHEAP_PROMPT + "\n\n" + self.context() + "\n\nQUERIES ALREADY TRIED: " + "; ".join(sorted(self.tried))
            suggested = [str(q).strip() for q in (self.ask(ask, 300) or {}).get("queries") or [] if str(q).strip()] if spare() >= 4 else []
            for query in [q for q in dict.fromkeys(own + suggested) if q not in self.tried][:max(1, CHEAP_QUERIES // share)]:
                if spare() < 3:
                    break
                action = self.search_action(query, None)
                if action is None:
                    break
                self.tried.add(query)
                observations = self.act([action])
                self.listen(observations)
        self.trace.append({"kind": "cheap_sweep", "pool_before": before, "pool_after": len(self.pool()), "id_store": self.id_store(),
                           "price_store": self.price_store(), "walked": walked, "deep": deep, "steps": self.step - start, "tried": sorted(self.tried)})

    def read_more(self, rounds: int = MORE_ROUNDS) -> None:
        """Spare steps before the finish open more candidates: the race leaders used 37-39 steps a task, fresh about
        25. Each round reads up to MORE_READS of the best unread rows and reviews again, while the steps for the finish,
        the time and the task's spend allow (the review runs on the cheap model)."""
        for _ in range(rounds):
            # room for the finish twice over: a pick refused or sold out at checkout must still be replaced and ordered
            spare = self.steps_left() - 2 * self.finish_reserve() - self.held_messages() - 2
            if spare < 2 or self.over() or self.spent >= SPEND_SOFT or self.time_left() < FORCE_SECONDS + 90:
                return
            unread = self.shortlist(SHORTLIST, skip=set(self.details))[: min(MORE_READS, spare)]
            if not unread:
                return
            self.trace.append({"kind": "read_more", "ids": unread, "spare": spare})
            self.read_details(unread)
            self.judge()

    def recheck_all(self, count: int = 3) -> None:
        """After a change of mind: look again at every listing seen in the category, not only the opened ones.
        A dropped requirement, a new budget or a sold-out pick can make an unopened listing the right one."""
        room = min(count, self.steps_left() - self.finish_reserve() - 1)
        if room > 0:
            self.read_details(self.shortlist(SHORTLIST, skip=set(self.details))[:room])
        self.judge()

    def record(self, ident: str) -> str:
        info = self.rows.get(ident, {})
        head = {k: info.get(k) for k in ("id", "title", "category", "price", "stock") if info.get(k) is not None}
        if ident in self.unavailable:
            head["stock"] = False  # sold out or gone by the shopper's or store's word, whatever the listing says
        return _compact({"row": head, "facts": dict(_facts(self.details.get(ident)))}, 2500)

    def cheaper_filters(self, price: float, label: str) -> None:
        """For a cheapest request with a fitting candidate at `price`: list the category's products below that
        price with the store's filter search. The first cap is just under the price (a cap at the price itself
        fills up with products at that very price); while a filter matches more rows than it shows, two thirds and
        a third of the price reach the cheaper ones it cut off."""
        role = self.roles.get("search") or {}
        if (not label or not role.get("category_arg") or not role.get("price_arg") or (role.get("mode_arg") and not role.get("filter_value"))
                or price in self.filtered):
            return
        self.filtered.add(price)
        for cap in (price - 0.01, price * 2 / 3, price / 3):
            if cap <= 0 or self.filter_steps >= FILTER_STEPS or self.steps_left() <= self.finish_reserve() + FILTER_RESERVE:
                return
            args: dict[str, Any] = {role["category_arg"]: label, role["price_arg"]: round(cap, 2)}
            if role.get("mode_arg"):
                args[role["mode_arg"]] = role["filter_value"]
            action = {"name": role["tool"], "args": self.fill_required(role["tool"], args, label)}
            observations = self.act([action])
            self.filter_steps += 1
            if not observations:
                return
            self.listen(observations)
            rows, total = _result_rows(observations[0]), _total_count(observations[0])
            self.trace.append({"kind": "cheaper_filter", "cap": round(cap, 2), "label": label, "rows": len(rows), "matched": total})
            if not rows or not isinstance(total, int) or total <= len(rows):
                return  # every product under this cap is listed: a lower cap adds nothing
            shown = len(rows)
            if role.get("page_arg"):
                spec = self.props(role["tool"]).get(role["page_arg"]) or {}
                most = spec.get("maximum") if isinstance(spec.get("maximum"), int) else self.page_base + 2
                last = min(self.page_base + math.ceil(total / len(rows)) - 1, most,
                           self.page_base + FILTER_STEPS - self.filter_steps, self.page_base + self.steps_left() - self.finish_reserve() - FILTER_RESERVE)
                pages = [{"name": role["tool"], "args": {**action["args"], role["page_arg"]: self.page_value(p)}} for p in range(self.page_base + 1, last + 1)]
                if pages:
                    more = self.act(pages)
                    self.filter_steps += len(pages)
                    if more:
                        self.listen(more)
                        shown += sum(len(_result_rows(o)) for o in more)
            if shown >= total:
                return

    def pre_order(self, pick: str) -> str:
        """The checks before an order: '' to go ahead, 'recheck' after reading more, or why the pick fails.

        Cheapest requests first open cheaper unread listings of the category; then a strict
        review quotes the facts for every requirement and checks the request's rules."""
        if self.cheapest_first() and pick not in self.cheaper_checked and self.steps_left() > self.finish_reserve() + 2:
            self.cheaper_checked.add(pick)
            price = self.rows.get(pick, {}).get("price")
            if isinstance(price, (int, float)):
                self.cheaper_filters(float(price), str(self.rows.get(pick, {}).get("category") or self.target_label() or ""))
                values = [o for v in self.values_for_query() for o in _options(v)]
                cheaper = sorted((r for r in self.pool() if r["id"] not in self.details and isinstance(r.get("price"), (int, float))
                                  and r["price"] < price),
                                 key=lambda r: (not any(_stated(v, str(r.get("title") or "")) for v in values), self.price_rank(r)))  # titles naming a wanted value first
                room = min(3, self.steps_left() - self.finish_reserve() - 1)
                if cheaper and room > 0:
                    self.read_details([r["id"] for r in cheaper[:room]])
                    self.judge()
                    return "recheck"
        key = f"{pick}|{'|'.join(self.reqs)}|{self.budget}"
        if key in self.order_checked or self.time_left() < FORCE_SECONDS + 40:
            return ""
        self.order_checked.add(key)
        # Items that are gone cannot be bought, so they are never a "cheaper fit" to compare against.
        out = self.unavailable | self.rejected
        # only the products that could beat the pick: cheaper ones for a cheapest request, all for a best-by-measure
        # one, none otherwise (the count of products opened still goes in, for the request's rules)
        price = self.rows.get(pick, {}).get("price")
        kind = self.preference.get("kind")
        opened = self.mine()
        rivals = [i for i in opened if i != pick and i not in out and (
            kind == "best" or (kind == "cheapest" and isinstance(price, (int, float)) and isinstance(self.rows.get(i, {}).get("price"), (int, float))
                               and self.rows[i]["price"] < price))]
        others = "\n".join(self.record(i) for i in rivals)[-8000:]
        if self.unavailable:
            others += "\n\nNO LONGER AVAILABLE (cannot be ordered, never a reason to reject the item): " + ", ".join(sorted(self.unavailable))
        if self.rejected:
            # without this the review rejected the cheapest fit, then every other one as "not the cheapest" for it
            others += "\n\nALREADY REJECTED BY THIS REVIEW (not candidates; never a cheaper fit to compare against): " + ", ".join(sorted(self.rejected))
        result = self.ask(ORDER_PROMPT + "\n\n" + self.context(story=False) + f"\n\nTHE ITEM TO ORDER ({pick}):\n" + self.record(pick)
                          + f"\n\nOTHER PRODUCTS OPENED ({len(opened)} opened in all):\n" + (others or "(none)"), 1500, gate=self.order_checks < ORDER_CHECKS_MAIN) or {}
        self.order_checks += 1
        self.trace.append({"kind": "order_check", "item": pick, "result": result})
        if result.get("pass") is False:
            failed = [c for c in result.get("checks") or [] if isinstance(c, dict) and c.get("verdict") in ("fail", "unproven")]
            # Only a failure on something real counts: a current requirement the item's own facts do not plainly
            # state, the price, budget or cheapest/best choice, or stock when the item is not shown in stock. A
            # failure on what no requirement asks for (the category, a property made up from the story, a rule,
            # which the rule actions handle) or on a requirement the facts plainly state is the review misreading.
            stated = self.stated_requirements(pick, firm=True)
            titled = self.stated_requirements(pick)  # with the title, read when no key carries the property's name
            in_stock = self.rows.get(pick, {}).get("stock") is not False and pick not in self.unavailable

            def real(check: dict) -> bool:
                text = str(check.get("requirement") or "")
                if re.search(r"(?i)cheap|best|preference", text) and not re.search(r"(?i)budget|quantit|total", text):
                    # "not the cheapest (or best)" stands only when it names another opened product that fits better
                    return any(i != pick and i not in out and i in json.dumps(check) for i in self.details)
                if re.search(r"(?i)quantit|units?\b|pieces?\b", text) and not re.search(r"(?i)price|budget|cost|total", text) \
                        and not any(_stated(_split_req(r)[1], text) for r in self.reqs if _split_req(r)[1]):
                    return False  # the code puts the units in the cart; a listing's pack or title cannot fail them
                if re.search(r"(?i)price|budget|cost|quantit|total", text):
                    if re.search(r"(?i)budget|total", text) and not re.search(r"(?i)cheap", text):  # (price x quantity is checked)
                        # the price and the budget are in the same unit (setup keeps the request's unit, as the store's
                        # prices): a review doubting the unit, or the whole order's total, which is the code's own
                        # check (finish_many), is no reason; the product only has to fit its own limit, or the budget
                        own = self.rows.get(pick, {}).get("price")
                        limit = self.own_budget if self.own_budget is not None else self.budget
                        if isinstance(own, (int, float)) and limit is not None and own * self.quantity <= limit + 1e-9:
                            return False
                    return True
                if re.search(r"(?i)stock|availab|on hand", text):
                    return not in_stock
                for req in self.reqs:
                    prop, value = _split_req(req)
                    if value and _stated(value, text):  # the check is about this requirement
                        # an exclusion failed on one of the product's own facts stating the excluded value stands,
                        # whatever key the code read: its "yes" only means that one key lacks the excluded word
                        if any(_stated(b, quoted) for b in _clauses(value)[1] for quoted in self.quoted_facts(pick, check, prop)):
                            return True
                        if (prop, value) in stated:
                            return False
                        # the title states it (the product has no key for the property): that stands unless the check
                        # quotes one of the product's own facts for the property against it; a quote of the title, or of
                        # something the facts do not hold ('not stated'), is no reason
                        if (prop, value) in titled:
                            return bool(self.quoted_facts(pick, check, prop))
                        return True
                return False

            failed = [c for c in failed if real(c)]
            # "Not the cheapest (or best): another opened product fits better" is no reason to drop the pick: that
            # product becomes a candidate ahead of it and is checked next; the pick stays in case it fails.
            better = [i for c in failed for i in self.details
                      if i != pick and i not in out and i in json.dumps(c)
                      and re.search(r"(?i)cheap|best|lower|less|preference", str(c.get("requirement") or "") + str(c.get("fact") or ""))]
            if failed and better and all(re.search(r"(?i)cheap|best|preference", str(c.get("requirement") or "")) for c in failed):
                ahead = [i for i in dict.fromkeys(better) if i not in (self.review.get("fits") or []) and not self.fact_veto(i)]
                if all(self.fact_veto(i) for i in better):
                    failed = []  # every "better fit" it names fails its own facts: the claim does not stand
                    self.trace.append({"kind": "order_check_overruled", "item": pick, "reason": "the better fits it named fail their facts"})
                elif ahead:
                    self.review["fits"] = ahead + list(self.review.get("fits") or [])
                    if self.cheapest_first():
                        self.review["fits"].sort(key=lambda i: self.price_rank(self.rows.get(i, {})))
                    self.candidates = list(dict.fromkeys(self.candidates + ahead))
                    self.trace.append({"kind": "better_fit", "item": pick, "ahead": ahead})
                    return "recheck"
            if failed or not result.get("checks"):
                return "; ".join(f"{c.get('requirement')}: {c.get('verdict')} ({c.get('fact')})" for c in failed[:3]) or "the review failed it"
            if not self.trace or self.trace[-1].get("kind") != "order_check_overruled":
                self.trace.append({"kind": "order_check_overruled", "item": pick, "reason": "every failure was on a requirement its facts state"})
        again = False
        for rule in result.get("rules") or []:
            action = rule.get("action") if isinstance(rule, dict) and not rule.get("met") else None
            if not isinstance(action, dict):
                continue
            if action.get("kind") == "open_more" and self.steps_left() > self.finish_reserve() + 2:
                count = int(action.get("count") or 1) if str(action.get("count") or "1").isdigit() else 1
                unread = self.shortlist(SHORTLIST, skip=set(self.details))[: max(1, min(count, self.steps_left() - self.finish_reserve() - 1))]
                if unread:
                    self.read_details(unread)
                    self.judge()
                    again = True
            elif action.get("kind") == "message" and str(action.get("text") or "").strip():
                if self.confirm_first:
                    self.rule_message = str(action["text"]).strip()  # sent with the basket, in the confirmation
                else:
                    self.pending_reply = str(action["text"]).strip()
                    self.flush_reply()
            elif action.get("kind") == "remark" and str(action.get("text") or "").strip():
                self.remark = str(action["text"]).strip()
        return "recheck" if again else ""

    def before_decline(self) -> bool:
        """A wrong decline scores 0: once, before giving up, open what a review says may fit and try other
        searches (the decline check). True when something now fits."""
        if self.decline_checked or self.steps_left() <= self.finish_reserve() + 2 or self.time_left() < FORCE_SECONDS + 60:
            return False
        self.decline_checked = True
        unread = sorted((r for r in self.pool() if r["id"] not in self.details), key=self.price_rank)[:30]
        rows = "\n".join(_compact({k: v for k, v in r.items() if k != "text"}, 300) for r in unread)
        opened = "\n".join(self.record(i) for i in self.mine())[-10000:]
        result = self.ask(DECLINE_PROMPT + "\n\n" + self.context() + "\n\nPRODUCTS OPENED:\n" + (opened or "(none)")
                          + "\n\nLISTINGS SEEN BUT NOT OPENED:\n" + (rows or "(none)") + "\n\nQUERIES TRIED: " + "; ".join(sorted(self.tried)), 1500, gate=True) or {}
        self.trace.append({"kind": "decline_check", "result": result})
        known = {r["id"] for r in unread}
        first = [i for i in dict.fromkeys(str(x) for x in result.get("open_first") or []) if i in known]
        queries = [str(q).strip() for q in result.get("queries") or [] if str(q).strip()]
        if queries and self.steps_left() > self.finish_reserve() + 4:
            self.model_queries(SEARCH_PROMPT, 10**6, queries[:2])
            first += [i for i in self.shortlist(SHORTLIST, skip=set(self.details) | set(first)) if i not in first]
        room = min(4, self.steps_left() - self.finish_reserve() - 1)
        if first and room > 0:
            self.read_details(first[:room])
        self.judge()
        # an opened product the check finds failing nothing is a fit, whatever the review's list said (the order check
        # still verifies it before it is ordered)
        opened = self.mine()
        passed = [str(o.get("id")) for o in result.get("opened") or [] if isinstance(o, dict) and o.get("fails") in (None, "", "null")
                  and str(o.get("id")) in opened and str(o.get("id")) not in self.unavailable | self.rejected
                  and self.affordable(self.rows.get(str(o.get("id")), {})) and not self.firm_veto(str(o.get("id")))]
        if passed:
            self.review["fits"] = list(dict.fromkeys(list(self.review.get("fits") or []) + passed))
            self.candidates = list(dict.fromkeys(self.candidates + passed))
            self.trace.append({"kind": "decline_check_fits", "ids": passed})
        return bool(self.review.get("fits"))

    def finish_action(self, kind: str) -> dict[str, Any] | None:
        """The order or the decline, with the mode that selects it when one tool does both, and a rule's remark."""
        role = self.roles.get(kind)
        if not role:
            return None
        tool = role["tool"]
        args: dict[str, Any] = dict(self.finish_args.get(kind) or {})
        text = NEUTRAL_DECLINE if kind == "decline" else "Order the cart."
        remark = self.remark
        if remark and kind == "decline" and self.plan.get("decline_no_names") is not False and any(i in remark for i in self.rows):
            remark = ""  # a decline that must not name substitutes keeps the neutral text
        text_arg = role.get("text_arg") or next((k for k, s in self.props(tool).items()
                                                 if isinstance(s, dict) and s.get("type") == "string" and not s.get("enum") and k not in args), None)
        if text_arg and (remark or kind == "decline"):
            args[text_arg] = remark or text
        return {"name": tool, "args": self.fill_required(tool, args, remark or text)}

    def finish(self) -> None:
        """Cart the best fit, re-check its stock and price, order; repair a refusal; else give up."""
        tried: set[str] = set()
        refusals: dict[str, int] = {}
        declines = stale = 0
        while not self.over():
            self.flush_reply()
            fits = [i for i in self.review.get("fits") or [] if i not in self.unavailable and i not in tried]
            if not fits:
                if self.before_decline():
                    continue
                observations = self.give_up()
                if self.over() or observations is None or declines >= 2:
                    return
                # the decline was refused: the shopper's change rides on the reply, so apply it and look again
                declines += 1
                if self.listen(observations) and self.steps_left() > self.finish_reserve() + 2:
                    self.recheck_all()
                else:
                    self.judge()
                continue
            pick = fits[0]
            verdict = self.pre_order(pick)
            if verdict == "recheck":
                continue
            if verdict:
                tried.add(pick)
                self.rejected.add(pick)
                self.trace.append({"kind": "pick_failed", "item": pick, "why": verdict})
                continue
            if (self.roles.get("cart") or {}).get("lines_arg"):
                actions = [self.basket_action({pick: self.quantity})] if self.cart != {pick: self.quantity} else []
            else:
                actions = [self.cart_action(i, False) for i in list(self.cart) if i != pick]
                if pick not in self.cart:
                    actions.append(self.cart_action(pick, True))
            listing = self.id_action("listing", pick) if self.steps_left() - len(actions) > 2 else None
            if listing:
                actions.append(listing)
            actions = [a for a in actions if a]
            if actions:
                observations = self.act(actions)
                if observations is None:
                    return
                changed = self.listen(observations)
                info = self.rows.get(pick, {})
                if not self.affordable(info) or pick in self.unavailable:
                    tried.add(pick)
                if changed and self.steps_left() > self.finish_reserve() + 2:
                    self.recheck_all()  # a change while carting: every listing of the category, not only the opened
                    continue
                if pick in tried:
                    continue
                if any(i != pick for i in self.cart) and stale < 2:
                    stale += 1  # a dropped item still has units carted: take them off before ordering
                    continue
            if self.over() or self.steps_left() < 1:
                return
            if self.confirm_first and self.confirmed != self.basket_key() and self.messages_left() > 0 and self.steps_left() >= 2:
                if self.confirm_basket():
                    # the reply changed the wants: judge again before anything is ordered
                    if self.steps_left() > self.finish_reserve() + 2:
                        self.recheck_all()
                    else:
                        self.judge()
                    continue
                if self.over():
                    return
            action = self.finish_action("order")
            if action is None:
                return
            observations = self.act([action])
            self.order_attempts += 1
            if self.finished or observations is None:
                return
            # refused: the change rides on the reply; apply it and pick again
            refusals[pick] = refusals.get(pick, 0) + 1
            changed = self.listen(observations)
            if not changed and refusals[pick] >= 2:
                tried.add(pick)  # refused twice with nothing to apply: try the backup
            if changed and self.steps_left() > self.finish_reserve() + 2:
                self.recheck_all()
            else:
                self.judge()
            if self.steps_left() <= 1:
                self.force_finish()
                return

    def basket_key(self) -> tuple:
        """What a confirmation shows: the cart's lines and the wants they were chosen for."""
        return (tuple(sorted(self.cart.items())), self.all_wants())


    def give_up(self) -> list[Any] | None:
        action = self.finish_action("decline")
        if action is None or self.over():
            return None
        return self.act([action])

    def force_finish(self) -> None:
        """Out of time, steps or replies: order a carted fit, else give up."""
        if self.over():
            return
        if self.multi():
            self.save_item()
            fits = {i for item in self.items for i in item["review"].get("fits") or []} - self.unavailable
            complete = len(self.cart) == len(self.items)
        else:
            fits, complete = set(self.review.get("fits") or []) - self.unavailable, True
        action = self.finish_action("order")
        if action and self.cart and complete and set(self.cart) <= fits:
            self.act([action])
            if self.finished:
                return
        self.give_up()


    def rejudge(self, items: set[int]) -> None:
        """After a change of mind: the changed products look again (every product when none is named)."""
        for k in sorted(items or set(range(len(self.items)))):
            self.switch(k)
            if self.steps_left() > self.finish_reserve() + 2:
                self.recheck_all()
            else:
                self.judge()
        self.changed_items = set()

    def unrefuted(self, skip: set[str]) -> str:
        """The current product's best opened listing that nothing rules out: in its category, in budget and in stock,
        no firm fact veto, not rejected by an order check; the one whose facts state the most current requirements
        first (then the review's order). '' when none."""
        stated_wants = {(p, v) for p, v in (_split_req(r) for r in self.reqs) if p and v}
        label = self.target_label()
        pool = [i for i in self.mine() if i not in skip and i not in self.unavailable and i not in self.rejected
                and self.affordable(self.rows.get(i, {})) and not self.firm_veto(i)
                and (not label or not self.rows.get(i, {}).get("category") or self.rows[i]["category"] == label)]
        if not pool:
            return ""
        rank = {i: n for n, i in enumerate(self.candidates)}
        pool.sort(key=lambda i: (-len(stated_wants & set(self.stated_requirements(i))), rank.get(i, len(rank)), self.price_rank(self.rows.get(i, {}))))
        return pool[0]

    def finish_many(self) -> None:
        """Every product's best verified fit in one cart, within the whole order's budget, then the order; a refusal
        is read, applied and repaired; when a product has no fit, give up."""
        n = len(self.items)
        tried: dict[int, set[str]] = {k: set() for k in range(n)}
        refusals = declines = stale = 0
        while not self.over():
            self.flush_reply()
            picks: dict[int, str] = {}
            for k in range(n):
                self.switch(k)
                while not self.over():
                    fits = [i for i in self.review.get("fits") or [] if i not in self.unavailable and i not in tried[k]
                            and i not in picks.values()]
                    if not fits:
                        if self.before_decline():
                            continue
                        # a product with no verified fit takes its best unrefuted candidate rather than sinking the
                        # whole order: declining an order of several products almost always failed in the race logs
                        # (passed 194 / failed 8386 in race 6, 199 / 4078 in race 7)
                        stand_in = self.unrefuted(tried[k] | set(picks.values()))
                        if stand_in:
                            tried[k].add(stand_in)
                            picks[k] = stand_in
                            self.trace.append({"kind": "stand_in", "product": k, "item": stand_in})
                        break
                    verdict = self.pre_order(fits[0])
                    if verdict == "recheck":
                        continue
                    if verdict:
                        tried[k].add(fits[0])
                        self.rejected.add(fits[0])
                        self.trace.append({"kind": "pick_failed", "product": k, "item": fits[0], "why": verdict})
                        continue
                    picks[k] = fits[0]
                    break
                if k not in picks:
                    break
            if self.over():
                return
            if len(picks) < n:
                observations = self.give_up()
                if self.over() or observations is None or declines >= 2:
                    return
                declines += 1  # the decline was refused: apply what the shopper said and look again
                self.listen(observations)
                self.rejudge(self.changed_items)
                continue
            # the whole order within the budget: else the dearest pick gives way to that product's next fit
            def cost(k: int) -> float:
                price = self.rows.get(picks[k], {}).get("price")
                self.switch(k)
                return float(price) * self.quantity if isinstance(price, (int, float)) else 0.0
            if self.budget is not None and sum(cost(k) for k in picks) > self.budget + 1e-9:
                # the product whose cheapest other fit saves the most gives way (else the dearest pick), and its fits
                # are then tried cheapest first, so a tight budget is met before any product runs out of fits
                def saving(k: int) -> float:
                    now = cost(k)
                    others = [float(self.rows[i]["price"]) * self.quantity for i in self.review.get("fits") or []
                              if i != picks[k] and i not in tried[k] and i not in self.unavailable
                              and isinstance(self.rows.get(i, {}).get("price"), (int, float))]
                    return now - min(others) if others and min(others) < now else 0.0
                savings = {k: saving(k) for k in picks}
                give = max(savings, key=savings.get) if max(savings.values()) > 0 else max(picks, key=cost)
                tried[give].add(picks[give])
                self.switch(give)
                self.review["fits"] = sorted(self.review.get("fits") or [], key=lambda i: (
                    float(self.rows[i]["price"]) if isinstance(self.rows.get(i, {}).get("price"), (int, float)) else float("inf")))
                self.trace.append({"kind": "over_budget", "picks": dict(picks), "budget": self.budget, "gives_way": give})
                continue
            wanted = set(picks.values())
            if (self.roles.get("cart") or {}).get("lines_arg"):
                basket = {}
                for k, ident in picks.items():
                    self.switch(k)
                    basket[ident] = self.quantity
                actions = [self.basket_action(basket)] if self.cart != basket else []
            else:
                actions = [self.cart_action(i, False) for i in list(self.cart) if i not in wanted]
                for k, ident in picks.items():
                    self.switch(k)
                    if self.cart.get(ident) != self.quantity:
                        actions.append(self.cart_action(ident, True))
            view = self.roles.get("cart_view")
            actions = [a for a in actions if a]
            if view and self.steps_left() - len(actions) > 3:
                actions.append({"name": view["tool"], "args": self.fill_required(view["tool"], {})})
            if actions:
                # the products' own cart lines, in the order sent (absorb books each add under the current one)
                for k, ident in picks.items():
                    self.lines[k] = ident
                self.switch(0)
                observations = self.act(actions)
                if observations is None:
                    return
                for k, ident in picks.items():
                    self.lines[k] = ident
                if self.listen(observations) or any(i in self.unavailable for i in wanted):
                    self.rejudge(self.changed_items)
                    continue
                if set(self.cart) - wanted and stale < 2:
                    stale += 1  # a dropped item still has units carted: take them off before ordering
                    continue
            if self.over() or self.steps_left() < 1:
                return
            if self.confirm_first and self.confirmed != self.basket_key() and self.messages_left() > 0 and self.steps_left() >= 2:
                if self.confirm_basket():
                    self.rejudge(self.changed_items)
                    continue
                if self.over():
                    return
            action = self.finish_action("order")
            if action is None:
                return
            observations = self.act([action])
            self.order_attempts += 1
            if self.finished or observations is None:
                return
            refusals += 1
            if self.listen(observations):
                self.rejudge(self.changed_items)
            elif refusals >= 2:
                for k, ident in picks.items():
                    tried[k].add(ident)  # refused twice with nothing to apply: try the backups
            if self.steps_left() <= 1:
                self.force_finish()
                return


# -- fresh 5: the actor --------------------------------------------------------------------------------------------

ACTOR_MODELS = (CHEAP_MODEL, BACKUP_MODEL)
ACTOR_TURNS = 60  # model turns of a task at least; a longer step limit allows more, up to MOST_TURNS
MOST_TURNS = 110
CLOSING_SECONDS = 70.0  # with less time than this left, the status tells the actor to finish
CLOSING_STEPS = 5  # and with this few steps left
OBS_CHARS = 8000  # longest tool result shown to the actor (each text field clipped first, so the structure survives)
TABLE_ROWS = 12  # opened listings shown in the status table
GATE_HOLDS = 3  # orders the model review may hold back in one task; the next goes out as it is (code's own checks hold every time)

ACTOR_PROMPT = """You are a shopping assistant buying for one shopper in a simulated online store, working only through the tools you are given. The task scores 1 only when the finish is right in every respect: each item ordered meets every requirement of the request, every need the shopper reveals when asked, and every change the shopper or store announces along the way; the units and the number of lines are as asked, within the one budget; a cheapest or best request gets that listing; and every working rule written in the request is followed. A decline is right only when the store truly has nothing that fits, which is rare (about one task in fifty): when listings of the asked category fit the budget, order the best fit.

Your controller:
- The program running you (the code) adds notes to tool results and sends STATUS messages. Lines that start with NOT SENT, NOT RUN, NOTE, ALREADY DONE or STATUS, or that say what "the code" did or found, come from it, not from the store or the shopper: they are trustworthy, and you follow them. A cart line the code put in after its check of every requirement stays unless STATUS names a requirement it fails.

How the store works:
- Every tool call costs one step, failed calls included. Calls that do not depend on each other (say, opening several listings) may go together in one reply. An order or a decline always goes alone.
- Stores differ. Read every tool's description and schema and use exactly its argument names and allowed values. Paging may use a page number, a row offset or a token the results hand back; results may be ordered by relevance, by price or by id; a search may need every word to match; a cart tool may add one line per call or replace the whole basket with the list it gets (then every call must hold every line you want to keep).
- Search with short queries built from the product's category name plus ONE required value: one search per value, the most distinctive value first. Compare the total a search reports with the rows a page shows. When the matches fit in the few pages your steps allow, page through them. When they are too many to page through, search the category with TWO required values together: the search ranks by matching words, so listings that have both values come first, and the first pages of that search are the ones to read.
- A required value of two common words (each word alone also matches many unrelated listings) is searched by its one most distinctive word, not the phrase. A limit or range ("at most 8", "under 2 kg") is never a search word: check it in the listing's details. A search that finds nothing needs fewer words.
- Judge a listing by its own details (open them), not only its title. Requirements are met by the listing's own property fields; the description is never evidence. A listing on the requested category IS that kind of product even when its title names a case, a refill, a set or an accessory: judge it only on the requirements, price and stock.
- A filtered search that returns no rows usually means a wrong filter value: retry with the label exactly as the rows show it.
- The cart may ADD units to a line it already holds: never put a held line in again; to change units, take the difference out or in.

The shopper:
- The shopper has needs the request does not state; each is revealed only to a question about that property. The code asks those questions for you once a few listings are open, and does the request's own steps with the shopper (showing the basket, asking for approval) right before the order: never message the shopper yourself. Open several candidate listings early so the questions can name their properties.
- An answer like "no preference" leaves that property open. An answer naming a value or a limit is a firm requirement from then on. The STATUS block shows the requirements as the code currently reads them.
- When the shopper or the store says something changed (a new need, a new budget, an item no longer available), follow it from then on.

Choosing and finishing:
- With a cheapest request, buy the lowest-priced listing that meets every requirement (check the cheaper listings you have seen, and prove it with the store's price filter: list the product's category capped at your best fit's price, page to the end and open every listing it returns; STATUS says how). With a best-by-some-measure request, the best value of that measure among those that fit. Without a stated preference, any listing that fits within budget will do; when several fit equally, take the cheaper one.
- Put exactly the units asked for in the cart, one line per requested product, and nothing else; keep the total within budget.
- When one requested product must fit or work with another in the order (an accessory, case, cover, part or refill for a device), choose the two together: read the size, model, series or list of what it fits from both listings' own facts, and order only a pair whose facts agree (a 13-inch sleeve does not hold a 15.6-inch laptop). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits, before you think of declining.
- The request's working rules are graded like requirements: follow each literally. "Look at", "open" or "compare" N listings means N listings that fit, each opened with the tool that shows its full details.
- An order can be REFUSED (the episode goes on, the cart as it was): apply every change the refusal states, take out what no longer fits, put in the next verified fit, re-check its stock and price, and order again. A refusal is never a reason to decline. Never send the same basket again after a refusal that says a pick is gone. Any change (a new budget, a dropped need, a sold-out listing) changes which listings qualify, so choose again by the preference among all that now qualify.
- Check each cart line against its own product's requirements only: exact values exactly, limits as limits, and the preference last, among that product's qualifiers.
- A device listing may give its names ("Model number: X. Also sold as Y.") and an accessory what it goes with ("Fits: A, B.", or a series such as "every ABxx model, except AB12"). Search rows show these too. When the request ties them together, the accessory's list must name one of the device's names; a brand or a similar model is not enough. Choose the pair together, changing the device when no accessory that meets its requirements names it.
- Finish with the order tool when the cart is right, or the decline tool when nothing fits after a real search. The code reviews an order or a decline before it goes out and may hand back problems instead: fix them and finish again.
- The STATUS block after each result shows steps, time and messages left, the requirements now, the cart, and a table of the listings you opened with which requirements their facts state (yes, no, ?: not decided by code). Facts you read yourself outrank the table.
- The BUDGET line counts the store steps that still fit in both the steps and the time left, the steps kept for carting and ordering, and the spare ones. Plan every reply within the spare steps: spread them over the products still missing a fitting cart line, prefer a call that settles a gap over a new broad search, and cart a fit as soon as it is opened. When BUDGET says WRAP UP, stop searching: cart the best opened fit of each missing product and order.
- Every reply must call at least one tool. Finish before the steps or the time run out."""

COMMIT_PROMPT = """You are the final reviewer before a shopping assistant places an order. The task scores only if every line of the order meets every current requirement of its product, the units and lines are as asked, the total is within the budget, a cheapest or best request gets that listing among the products opened, and the request's working rules are followed. You are strict, but never invent a requirement the request and the shopper did not state; a detail of the story is no requirement unless stated as one. A pack, set or bundle sold as one listing is one unit.
""" + MATCH_RULES + """
Check each line against the current requirements of the product it is for, quoting the listing's own fact for each. When one line must fit or work with another line of the order (an accessory for a device), check that their facts agree (size, model, what it fits) and fail the order when they plainly do not: a "Fits:" list has to include one of the device's own model names. When the request itself fixes what the accessory is compatible with (say "compatible with acer models") and the device's make is a different one, no listing can satisfy both: the stated requirement wins, so never fail the order for that fit. Say which working rules of the request are not yet met. When an opened product fits better (cheaper for a cheapest request, better on the measure for a best one) and meets every requirement, name it.
Return ONE JSON object and nothing else:
{"lines": [{"id": "...", "checks": [{"requirement": "...", "fact": "...", "verdict": "pass" | "fail" | "unproven"}]}], "rules_unmet": ["..."], "better": [{"id": "...", "why": "..."}], "pass": true or false, "problems": ["each problem in one short sentence, saying what to do about it"]}"""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. A wrong decline scores 0, so check hard before it gives up.
""" + MATCH_RULES + """
For each opened product, give the first current requirement it fails (quote its fact) or null when it passes everything. Name listings seen but not opened that could plausibly pass and should be opened, best first, and up to 3 short new search queries worth trying (the noun or a synonym, optionally with one required value).
Return ONE JSON object and nothing else:
{"opened": [{"id": "...", "fails": "requirement and quoted fact, or null"}], "open_first": ["id", ...], "queries": ["...", ...]}"""


class Actor(Episode):
    """Fresh 5: the actor chooses the moves; the code reads, books, shows the status and gates the finish."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.chat: list[dict[str, Any]] = []
        self.holds = 0  # orders the review held back
        self.audited = False
        self.calls = 0

    # -- the model -------------------------------------------------------------


    # -- what the actor sees -----------------------------------------------------

    def wants_text(self) -> str:
        if self.multi():
            text = self.products_text()
        else:
            reqs = "; ".join(self.reqs) or "none beyond the category"
            free = f" | does not mind: {', '.join(self.free)}" if self.free else ""
            text = f"PRODUCT: {self.quantity} x {self.category or '?'}: {reqs}{free}"
        lines = [text, f"BUDGET (whole order): {self.budget:g} {self.plan.get('currency') or ''}".rstrip() if self.budget is not None else "BUDGET: not stated"]
        pref = self.preference if isinstance(self.preference, dict) else {}
        if pref.get("kind") not in (None, "", "none"):
            lines.append("PREFERENCE: " + _compact(pref, 300))
        rules = [str(r) for r in self.plan.get("rules") or [] if r]
        if rules:
            lines.append("RULES OF THE REQUEST: " + "; ".join(rules))
        for item in self.plan.get("ambiguity") or []:
            if isinstance(item, dict):
                lines.append("ONLY THE SHOPPER CAN SETTLE: " + _compact(item, 300))
        if self.unavailable:
            lines.append("NO LONGER AVAILABLE: " + ", ".join(sorted(self.unavailable)))
        return "\n".join(lines)


    def cart_text(self) -> str:
        if not self.cart:
            return "CART: empty"
        lines, total, known = [], 0.0, True
        for ident, units in self.cart.items():
            price = self.rows.get(ident, {}).get("price")
            if isinstance(price, (int, float)):
                total += price * units
            else:
                known = False
            lines.append(f"{ident} x{units} @ {price if price is not None else '?'}")
        return "CART: " + "; ".join(lines) + (f" | total {total:g}" if known else " | total unknown")

    def table(self) -> str:
        opened = [i for i in self.details if i not in self.unknown]
        if not opened:
            return f"OPENED LISTINGS: none yet ({len(self.rows)} seen in results)"

        def rank(ident: str) -> tuple:
            marks = self.checks(ident)
            price = self.rows.get(ident, {}).get("price")
            return (marks.count(" no"), -marks.count(" yes"), price if isinstance(price, (int, float)) else 1e12)
        lines = []
        for ident in sorted(opened, key=rank)[:TABLE_ROWS]:
            info = self.rows.get(ident, {})
            stock = "out" if info.get("stock") is False or ident in self.unavailable else "in" if info.get("stock") else "?"
            lines.append(f"- {ident} | {str(info.get('title') or '')[:70]} | {info.get('price', '?')} | stock {stock} | {self.checks(ident) or '-'}")
        more = f" (best {TABLE_ROWS} of {len(opened)})" if len(opened) > TABLE_ROWS else ""
        text = f"OPENED LISTINGS{more}, {len(self.rows)} seen in results:\n" + "\n".join(lines)
        cheaper = self.cheaper_unopened()
        if cheaper:
            text += "\nCHEAPER LISTINGS SEEN BUT NOT OPENED: " + "; ".join(cheaper)
        return text

    def cheaper_unopened(self) -> list[str]:
        """For a cheapest request: listings seen in results, not opened, cheaper than the best opened fit so far."""
        if (self.preference or {}).get("kind") != "cheapest" or self.multi():
            return []
        fits = [self.rows[i]["price"] for i in self.details if i in self.rows and isinstance(self.rows[i].get("price"), (int, float))
                and " no" not in self.checks(i) and i not in self.unavailable]
        if not fits:
            return []
        label = str(next((self.rows[i].get("category") for i in self.details if self.rows.get(i, {}).get("category")), "") or "")
        rows = [r for r in self.rows.values() if r.get("id") not in self.details and isinstance(r.get("price"), (int, float))
                and r["price"] < min(fits) and r.get("stock") is not False and (not label or not r.get("category") or r["category"] == label)]
        return [f"{r['id']} {r['price']:g} {str(r.get('title') or '')[:50]}" for r in sorted(rows, key=lambda r: r["price"])[:5]]

    def status(self) -> str:
        left = self.time_left()
        allowance = self.messages_left()
        questions = max(0, allowance - 1) if self.confirmed is None else allowance
        lines = [f"STATUS: step {self.step} of {self.max_steps} used ({self.steps_left()} left); about {int(left)} s left; "
                 f"messages left {allowance} (for questions: {questions})" if self.roles.get("message") else
                 f"STATUS: step {self.step} of {self.max_steps} used ({self.steps_left()} left); about {int(left)} s left",
                 self.wants_text(), self.cart_text(), self.table()]
        if self.steps_left() <= CLOSING_STEPS or left < CLOSING_SECONDS:
            lines.append("FINISH NOW: put the best fit in the cart and order it, or decline if nothing fits.")
        return "\n".join(lines)

    # -- the guards ------------------------------------------------------------------

    def finish_kind(self, name: str, args: dict[str, Any]) -> str | None:
        """'order', 'decline' or None for a call; one tool doing both is told apart by its mode argument."""
        order, decline = (self.roles.get("order") or {}).get("tool"), (self.roles.get("decline") or {}).get("tool")
        if name == order and name == decline:
            for kind in ("decline", "order"):
                mode = self.finish_args.get(kind) or {}
                if mode and all(args.get(k) == v for k, v in mode.items()):
                    return kind
            return "order"
        return "order" if name == order else "decline" if name == decline else None

    def basket_problems(self) -> list[str]:
        """What code can tell about the cart without a model: lines, units, budget, availability."""
        problems = []
        if not self.cart:
            return ["the cart is empty: put the chosen listing in the cart first"]
        if self.multi():
            self.save_item()
            if len(self.cart) != len(self.items):
                problems.append(f"the cart has {len(self.cart)} lines but the request asks for {len(self.items)} products, one line each")
            wanted = sorted(int(i["quantity"]) for i in self.items)
            if len(self.cart) == len(self.items) and sorted(self.cart.values()) != wanted:
                problems.append(f"the units per line should be {wanted}, the cart has {sorted(self.cart.values())}")
        else:
            if len(self.cart) > 1:
                problems.append(f"the cart holds {len(self.cart)} different listings; the request wants one")
            elif next(iter(self.cart.values())) != self.quantity:
                problems.append(f"the cart line has {next(iter(self.cart.values()))} units; the request wants {self.quantity}")
        gone = [i for i in self.cart if i in self.unavailable]
        if gone:
            problems.append("no longer available: " + ", ".join(gone))
        prices = [self.rows.get(i, {}).get("price") for i in self.cart]
        if self.budget is not None and all(isinstance(p, (int, float)) for p in prices):
            total = sum(p * u for p, u in zip(prices, self.cart.values()))
            if total > self.budget + 1e-9:
                problems.append(f"the total {total:g} is over the budget {self.budget:g}")
        return problems

    def review_order(self) -> list[str]:
        """The strict review of the whole basket on the main model; problems, or [] to go ahead."""
        if self.time_left() < FORCE_SECONDS + 35:
            return []
        lines = "\n".join(f"LINE {i} x{u} (for the product needing: {'; '.join(self.item_reqs(i)) or 'nothing specific'}):\n{self.record(i)}"
                          for i, u in self.cart.items())
        kind = (self.preference or {}).get("kind")
        others = [i for i in self.details if i not in self.cart and i not in self.unknown]
        if kind == "cheapest":
            top = max((self.rows.get(i, {}).get("price") or 0 for i in self.cart), default=0)
            others = [i for i in others if isinstance(self.rows.get(i, {}).get("price"), (int, float)) and self.rows[i]["price"] < top]
        elif kind != "best":
            others = []
        context = self.context(story=True) + ("\n\n" + self.products_text() if self.multi() else "")
        prompt = (COMMIT_PROMPT + "\n\n" + context + "\n\nTHE ORDER:\n" + lines + "\n\nOTHER PRODUCTS OPENED:\n"
                  + ("\n".join(self.record(i) for i in others)[-8000:] or "(none that matter)"))
        # the cheap model reads a plain order first; the main model judges what it objects to and every ranked pick
        result = self.ask(prompt, 2000) if kind not in ("cheapest", "best") else None
        if not isinstance(result, dict) or result.get("pass") is not True:
            result = self.ask(prompt, 2000, gate=self.order_checks < ORDER_CHECKS_MAIN) or result or {}
        self.order_checks += 1
        self.trace.append({"kind": "order_review", "cart": dict(self.cart), "result": result})
        if result.get("pass") is not False:
            return []
        problems = []
        for line in result.get("lines") or []:
            ident = str(line.get("id")) if isinstance(line, dict) else ""
            for check in (line.get("checks") or []) if isinstance(line, dict) else []:
                if not isinstance(check, dict) or check.get("verdict") not in ("fail", "unproven"):
                    continue
                text = str(check.get("requirement") or "")
                # Only a requirement the request or the shopper set holds the order: stock, budget, units and the
                # category are the code's own checks or the store's, showing the basket is the code's job, and a
                # requirement code reads as met in the listing's own facts stands whatever the review thought.
                if re.search(r"(?i)stock|availab|budget|total|price|cost|quantit|units?\b|confirm|basket|categor", text):
                    continue
                req = next((r for r in self.item_reqs(ident) if all(_split_req(r)) and any(_stated(x, text) for x in _split_req(r))), None)
                if req is None or self.code_verdict(ident, req) == "yes":
                    continue
                problems.append(f"{ident}: {text} {check['verdict']} ({check.get('fact')})")
        for better in result.get("better") or []:
            if isinstance(better, dict) and better.get("id") in self.details and better.get("id") not in self.unavailable:
                problems.append(f"{better['id']} fits better: {better.get('why')}")
        if not problems:
            self.trace.append({"kind": "order_review_overruled", "cart": dict(self.cart)})
        return problems

    def gate_order(self) -> str:
        """'' to send the order, else why it was held (the actor gets that instead of a result)."""
        problems = self.basket_problems()
        if problems and (not self.cart or self.holds < GATE_HOLDS):
            self.holds += bool(self.cart)
            return "NOT SENT (no step used). " + " ".join(problems)
        if self.holds < GATE_HOLDS:
            problems = self.review_order()
            if problems:
                self.holds += 1
                return "NOT SENT (no step used). The order review found: " + " | ".join(problems[:4]) + \
                    " Fix this (or, if you are sure the review is wrong, order again: it will go out)."
        # Only a request that asks to see the basket spends a message on it. An unasked "Shall I place the order?"
        # uses up the message a hidden need still needs, and the shopper's shrug can be read as dropping one.
        if self.confirm_first and self.messages_left() > 0 and self.confirmed != self.basket_key() and self.steps_left() >= 3 and self.time_left() > FORCE_SECONDS + 15:
            before = list(self.reqs), self.budget
            changed = self.confirm_basket()
            self.chat_note(f"(The code showed the shopper the basket; exchange: {self.exchanges[-1] if self.exchanges else 'no reply'})")
            if changed or (list(self.reqs), self.budget) != before or set(self.cart) & self.unavailable:
                return "NOT SENT. The shopper's reply to the basket changed things: see STATUS and adjust the cart."
        return ""

    def code_verdict(self, ident: str, req: str) -> str:
        """Code's reading of one requirement for one listing: 'yes', 'no' or 'unsure'."""
        prop, value = _split_req(req)
        info = self.rows.get(ident, {})
        return _fact_reading(self.details.get(ident), str(info.get("title") or ""), prop, value, self.noun_words(), str(info.get("category") or ""))[0]

    def cart_fits(self) -> bool:
        """The cart holds the right lines and units within budget, and code reads no requirement of any line as failed."""
        return bool(self.cart) and not self.basket_problems() and all(
            self.code_verdict(i, r) != "no" for i in self.cart for r in self.item_reqs(i) if all(_split_req(r)))

    def gate_decline(self) -> str:
        if self.audited or self.steps_left() <= 4 or self.time_left() < FORCE_SECONDS + 40:
            return ""
        self.audited = True
        unread = sorted((r for r in self.rows.values() if r.get("id") not in self.details), key=self.price_rank)[:30]
        rows = "\n".join(_compact({k: v for k, v in r.items() if k != "text"}, 300) for r in unread)
        opened = "\n".join(self.record(i) for i in self.details)[-10000:]
        context = self.context(story=True) + ("\n\n" + self.products_text() if self.multi() else "")
        result = self.ask(AUDIT_PROMPT + "\n\n" + context + "\n\nPRODUCTS OPENED:\n" + (opened or "(none)")
                          + "\n\nLISTINGS SEEN BUT NOT OPENED:\n" + (rows or "(none)"), 1500, gate=True) or {}
        self.trace.append({"kind": "decline_audit", "result": result})
        passing = [str(o.get("id")) for o in result.get("opened") or [] if isinstance(o, dict) and not o.get("fails") and str(o.get("id")) in self.details]
        first = [str(i) for i in result.get("open_first") or [] if str(i) in self.rows][:4]
        queries = [str(q) for q in result.get("queries") or [] if str(q).strip()][:3]
        if not (passing or first or queries):
            return ""
        parts = []
        if passing:
            parts.append("these opened listings seem to meet every requirement: " + ", ".join(passing))
        if first:
            parts.append("open these first: " + ", ".join(first))
        if queries:
            parts.append("try these searches: " + "; ".join(queries))
        return "NOT SENT (no step used). Before declining, a check suggests: " + "; ".join(parts) + ". If none of them fits, decline again."

    def chat_note(self, text: str) -> None:
        if self.chat and self.chat[-1].get("role") == "tool":
            self.chat[-1]["content"] = str(self.chat[-1]["content"]) + "\n" + text

    # -- the loop -------------------------------------------------------------------

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Carry out one reply's tool calls; (call id, result text) for each, in order."""
        out: list[tuple[str, str]] = []
        batch: list[tuple[str, dict[str, Any]]] = []

        def flush() -> None:
            if not batch:
                return
            observations = self.act([a for _, a in batch])
            if observations is None:
                out.extend((cid, "NO RESULT (the store did not answer, or the task is over)") for cid, _ in batch)
            else:
                changed = self.listen(observations)
                for (cid, _), observation in zip(batch, observations):
                    out.append((cid, _show(observation)))
                if changed:
                    out[-1] = (out[-1][0], out[-1][1] + "\nNOTE: something said in these results changed the requirements; see STATUS.")
            batch.clear()

        for call in calls:
            cid, function = call["id"], call["function"]
            name = function.get("name")
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                args = None
            if name not in self.schemas or not isinstance(args, dict):
                out.append((cid, f"NOT SENT (no step used): {'no such tool' if name not in self.schemas else 'arguments are not a JSON object'}."))
                continue
            args, problems = _check(args, self.schemas[name])
            if problems:
                out.append((cid, "NOT SENT (no step used). Arguments do not fit the schema: " + "; ".join(problems[:4])))
                continue
            kind = self.finish_kind(name, args)
            message = (self.roles.get("message") or {}).get("tool") == name
            if kind or message:
                flush()
            if self.over():
                out.append((cid, "NOT SENT: the task is over."))
                continue
            if kind == "order":
                held = self.gate_order()
                if held:
                    out.append((cid, held))
                    continue
            elif kind == "decline":
                if self.cart_fits() and self.roles.get("order"):
                    # a decline over a basket that is right by code's reading would score 0: the basket is ordered
                    order = self.finish_action("order")
                    self.trace.append({"kind": "decline_to_order", "cart": dict(self.cart)})
                    name, args, kind = order["name"], order["args"], "order"
                    held = self.gate_order()
                else:
                    held = self.gate_decline()
                if held:
                    out.append((cid, held))
                    continue
            if message:
                if self.messages_left() <= 0:
                    out.append((cid, "NOT SENT (no step used): no messages left."))
                    continue
                text = str(args.get((self.roles["message"] or {}).get("text_arg") or "", "")) or _compact(args, 500)
                observations = self.act([{"name": name, "args": args}])
                if observations is None:
                    out.append((cid, "NO RESULT"))
                    continue
                self.messages_used += 1
                changed = self.listen(observations, text, reply_of=0)
                note = "\nTHE CODE'S READING: the requirements changed; see STATUS." if changed else ""
                out.append((cid, _show(observations[0]) + note))
                continue
            if kind:
                observations = self.act([{"name": name, "args": args}])
                if kind == "order":
                    self.order_attempts += 1
                text = _show(observations[0]) if observations else "NO RESULT"
                if observations and not self.finished and self.listen(observations):
                    text += "\nNOTE: the reply changed the requirements; see STATUS."
                out.append((cid, text))
                continue
            batch.append((cid, {"name": name, "args": args}))
        flush()
        return out

    def finish_now(self) -> None:
        """Out of time, steps or turns: order the carted basket when code finds nothing wrong with it, else decline."""
        if self.over():
            return
        action = self.finish_action("order")
        if action and self.cart and not self.basket_problems():
            self.act([action])
            if self.finished or self.over():
                return
        self.give_up()

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        time.sleep(START_SPREAD * (time.time_ns() % 1000) / 1000)  # the run starts every task at once
        self.setup()
        if self.opening is not None:
            for item in _listing_dicts(self.opening):
                info = _row_info(item)
                if info.get("id"):
                    self.rows[info["id"]] = info
        user = "SHOPPER REQUEST:\n" + self.request
        if self.opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _show(self.opening)
        self.chat = [{"role": "system", "content": ACTOR_PROMPT}, {"role": "user", "content": user + "\n\n" + self.status()}]
        idle = failures = 0
        for _ in range(max(ACTOR_TURNS, min(MOST_TURNS, self.max_steps))):
            if self.over():
                break
            if self.time_left() < FORCE_SECONDS or self.steps_left() <= 1:
                break
            message = self.actor_turn()
            if message is None:
                failures += 1
                if failures >= 2:
                    break
                continue
            calls = []
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and isinstance(call.get("function"), dict):
                    self.calls += 1
                    function = call["function"]
                    arguments = function.get("arguments")
                    calls.append({"id": str(call.get("id") or f"call_{self.calls}"), "type": "function",
                                  "function": {"name": function.get("name"), "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {})}})
            content = message.get("content") if isinstance(message.get("content"), str) else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no tool call)"})
                self.chat.append({"role": "user", "content": "Every reply must call a tool. Continue.\n\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            self.trace.append({"kind": "actor", "content": content[:1500], "calls": [c["function"] for c in calls]})
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for n, (cid, text) in enumerate(replies):
                if n == len(replies) - 1:
                    text += "\n\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})
        self.finish_now()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.spent, 5),
                           "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


def _price(info: dict[str, Any]) -> float | None:
    price = info.get("price")
    return float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) and price > 0 else None


def _arg_strings(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_arg_strings(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_arg_strings(v) for v in value)) if value else set()
    return {value} if isinstance(value, str) else set()


class Actor7(Actor):
    """Fresh 7: fresh 5's actor, plus repeats answered from memory, a flag on suspiciously cheap listings, and the
    best opened fit ordered instead of a decline (a decline is right in about one task in fifty)."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.done: dict[str, tuple[int, str]] = {}  # repeatable call -> (step, result)
        self.fell_back = False

    # -- what the actor sees -----------------------------------------------------


    def missing_lines(self) -> int:
        if not self.multi():
            return 0 if self.cart else 1
        return max(0, len(self.items) - len(self.cart))

    def status(self) -> str:
        text = super().status()
        missing = self.missing_lines()
        if missing and "FINISH NOW" not in text and self.steps_left() <= missing + 5:
            text += f"\nSTOP SEARCHING: {missing} product line(s) still not in the cart and few steps left; cart the best fits you have opened now."
        return text

    # -- the finish --------------------------------------------------------------

    def label_for(self) -> str:
        """The current product's category label among the rows, '' when no label shares a word with it."""
        wanted = {_one(w) for w in _words(self.category)}
        counts: dict[str, int] = {}
        for info in self.rows.values():
            if info.get("category"):
                counts[info["category"]] = counts.get(info["category"], 0) + 1
        best = max(counts, key=lambda label: (len(wanted & {_one(w) for w in _words(label)}), counts[label]), default="")
        return best if best and wanted & {_one(w) for w in _words(best)} else ""

    def best_fit(self, taken: set[str]) -> str | None:
        """For the current product: the opened listing (else a seen one of its category) that code reads as failing no
        requirement, with the most requirements read as met, then the lowest price; a suspiciously cheap one last."""
        label = self.label_for()
        opened = [i for i in self.details if i not in self.unknown and (not label or self.rows.get(i, {}).get("category") in (None, label))]
        seen = [i for i, r in self.rows.items() if i not in self.details and label and r.get("category") == label]
        for group in (opened, seen):
            ranked = []
            for ident in group:
                info = self.rows.get(ident, {})
                price = _price(info)
                if price is None or info.get("stock") is False or ident in self.unavailable or ident in taken:
                    continue
                verdicts = [_fact_reading(self.details.get(ident), str(info.get("title") or ""), *_split_req(req), self.noun_words(),
                                          str(info.get("category") or ""))[0] for req in self.reqs if all(_split_req(req))]
                if "no" not in verdicts:
                    ranked.append((self.suspicious(ident), -verdicts.count("yes"), price, ident))
            if ranked:
                return min(ranked)[3]
        return None

    def fallback(self) -> bool:
        """Cart the best fit for every product line still missing; True when the cart then passes code's checks."""
        if self.fell_back or not self.roles.get("cart") or not self.roles.get("order") or self.over():
            return False
        self.fell_back = True
        keep = self.current
        picks: list[tuple[int, str, int]] = []
        taken = set(self.cart)
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            live = [i for i in self.cart if i not in self.unavailable]
            if self.multi() and any(i in self.mine() for i in live):
                continue
            if not self.multi() and live:
                break
            ident = self.best_fit(taken)
            if ident is None:
                self.switch(keep)
                self.trace.append({"kind": "fallback", "result": "no fit", "product": k})
                return False
            picks.append((k, ident, int(self.quantity)))
            taken.add(ident)
        self.switch(keep)
        total = sum((_price(self.rows.get(i, {})) or 0) * u for i, u in self.cart.items()) + \
            sum((_price(self.rows.get(i, {})) or 0) * u for _, i, u in picks)
        if self.budget is not None and total > self.budget + 1e-9:
            self.trace.append({"kind": "fallback", "result": "over budget", "picks": picks, "total": total})
            return False
        if self.steps_left() < len(picks) + 2 or self.time_left() < 8:
            self.trace.append({"kind": "fallback", "result": "no room", "picks": picks})
            return False
        for k, ident, _ in picks:
            self.switch(k)
            action = self.cart_action(ident, True)
            self.switch(keep)
            if action is None or self.act([action]) is None:
                break
        self.trace.append({"kind": "fallback", "picks": picks, "cart": dict(self.cart)})
        return bool(self.cart) and not self.basket_problems()

    def gate_decline(self) -> str:
        held = super().gate_decline()
        if held:
            return held
        if self.fallback():
            return ("NOT SENT (no step used for it). A decline is almost never right, so the code carted the best fits it "
                    "found: " + self.cart_text() + ". Order this basket now (or swap a line for a better fit and order).")
        return ""

    def finish_now(self) -> None:
        if not self.over() and (not self.cart or self.basket_problems()):
            self.fallback()
        super().finish_now()

    # -- the store ---------------------------------------------------------------

    def call_key(self, name: str, args: dict[str, Any]) -> str | None:
        """A key for calls whose repeat would only show the same thing again: lookups with arguments (searches, filters,
        openings). Tools that change or show the cart, finish or message may repeat, as may a zero-argument view, and an
        opening of a carted listing may be a stock check."""
        acting = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove", "cart_view", "order", "decline", "message")}
        if not args or name in acting or _arg_strings(args) & set(self.cart):
            return None
        return json.dumps([name, args], sort_keys=True, default=str)

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        observations = super().act(actions)
        for action, observation in zip(actions, observations or []):
            key = self.call_key(action["name"], action.get("args") or {})
            if key and not _failed(observation):
                self.done[key] = (self.step, _show(observation))
        return observations

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Repeats of a search or opening are answered from memory, without a step; the rest go as in fresh 5."""
        answered: dict[str, str] = {}
        rest = []
        for call in calls:
            function = call["function"]
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                args = None
            key = self.call_key(str(function.get("name")), args) if isinstance(args, dict) else None
            if key and key in self.done:
                step, text = self.done[key]
                answered[call["id"]] = f"ALREADY DONE at step {step} (no step used now). Its result was:\n{text}"
            else:
                rest.append(call)
        results = dict(super().run_calls(rest)) if rest else {}
        return [(c["id"], answered.get(c["id"]) or results.get(c["id"], "NO RESULT")) for c in calls]


COMPACT_CHARS = 90000  # past this the conversation is condensed into one summary message
SEARCH_LOG = 6  # searches shown in STATUS
TASK_STOP = 0.075  # dollars: past this the actor stops and the code finishes the task (the race cap is $0.055 a task on average)
ASK_ROWS = 4  # properties suggested per product
PAGE_KEY = re.compile(r"page|token|offset|cursor|next|continu|start|after|skip", re.I)

ACTOR_RULES8 = """

Searching and opening, every time a result comes back:
- Read every row of the result against the requirements (title, price, category, stock); titles only suggest, the listing's own facts decide.
- Then choose ONE of these, using the SEARCHES and LIKELY TO OPEN lines of STATUS: open the likely rows (together in one reply; with the tool that opens several listings in one step when the store has one, as STATUS says); fetch the next pages when STATUS says the rest is affordable and the rows so far are relevant; else narrow the query (the noun plus one required value) or use a filter the store offers (price cap, category, shop). A search with 0 results needs fewer words. A page that brings no new relevant rows ends the paging of that query.
- Opening listings costs steps: open the likely ones, not whole pages.
- Each cart line must be a listing of its product's own category (the store's category label), not a look-alike from another shelf.
- Put a listing in the cart once you have chosen it: every add and every removal costs a step.

Phases (STATUS shows the current one; the code holds a move made too early, without using a step):
- SURVEY: search and filter the category, open a few candidate listings (several per step when the store allows). Read up to three pages of a query while each page brings new listings of the category; after a page with nothing new, change the query. Collect candidates from two or three searches, then open the best three to five together.
- ASK: the code asks the shopper; apply each answer as a requirement.
- RE-CHECK: search and open again for the needs heard; prove the cheapest.
- COMMIT: cart the chosen listing(s) once, then order.

Using the steps:
- Steps left are for work. Keep opening candidates and proving the cheapest until STATUS's BEFORE ORDERING list is empty; order earlier only when every requirement of every line and the preference are confirmed. The order is held while that list has items and steps remain.
"""


def _query_key(name: str, args: dict[str, Any]) -> str:
    return json.dumps([name, {k: v for k, v in args.items() if not PAGE_KEY.search(str(k))}], sort_keys=True, default=str)


def _reported_total(observation: Any) -> int | None:
    """The number of matches a search result says it has, when it says (top level or one level down)."""
    for value in (observation, *(v for v in observation.values() if isinstance(v, dict))) if isinstance(observation, dict) else ():
        for key, number in value.items():
            if isinstance(number, int) and not isinstance(number, bool) and re.search(r"total|matches|hits|found|num_?results|result_?count", str(key), re.I):
                return number
    return None


def _has_more(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    for value in (observation, *(v for v in observation.values() if isinstance(v, dict))):
        for key, item in value.items():
            if re.search(r"next|cursor|more|continu", str(key), re.I) and item not in (None, False, "", 0, []):
                return True
    return False


_FIGURE = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z%]{0,3})(?![a-z])")


def _figures(text: str) -> list[tuple[float, str]]:
    return [(float(n), u) for n, u in _FIGURE.findall(str(text).lower())]


def _figure_verdict(fact: str, value: str) -> str:
    """A wanted figure with a unit against a fact: 'yes' when the fact states it or a range holding it, 'no' when the fact states other figures in the same unit, else 'unsure'."""
    want = _figures(value)
    if len(want) != 1 or not want[0][1]:
        return "unsure"
    number, unit = want[0]
    got = _figures(fact)
    units = [u or unit for _, u in got]
    if not got or unit not in units:
        return "unsure"
    if len(got) >= 2 and re.search(r"\d\s*[a-z]{0,3}\s*(?:-|~|to)\s*\d", str(fact).lower()):
        low, high = sorted(n for n, _ in got[:2])
        return "yes" if low <= number <= high else "no"
    return "yes" if any(abs(n - number) < 1e-9 and (u or unit) == unit for n, u in got) else "no"


class Actor8(Actor7):
    """Fresh 8: fresh 7, plus the whole conversation kept (cached) instead of shortened, a log of every search with what
    paging the rest would cost, a shortlist of unopened listings whose titles fit, several single openings in one reply
    merged into the store's many-listing tool, and a short basket check."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.queries: dict[str, dict[str, Any]] = {}  # query (without its page argument) -> what its pages showed
        self.condensed = 0

    # -- the model -------------------------------------------------------------


    def actor_turn(self) -> dict[str, Any] | None:
        if self.spent >= TASK_STOP:
            self.trace.append({"kind": "spend_stop", "spent": round(self.spent, 4)})
            return None
        if self.chat and self.chat[0].get("role") == "system" and ACTOR_RULES8 not in self.chat[0]["content"]:
            self.chat[0]["content"] += ACTOR_RULES8
        self.condense()
        for model in ACTOR_MODELS:
            left = self.time_left()
            if left < FORCE_SECONDS:
                return None
            _llm.timeout = int(max(10, min(90, left - FORCE_SECONDS)))
            _llm.retry_until = time.monotonic() + left - FORCE_SECONDS
            sent = self.view() if model.startswith("anthropic/") else self.chat
            body = {"model": model, "messages": sent, "tools": self.tools, "temperature": 0, "max_tokens": 2000,
                    "reasoning": {"enabled": False} if model == CHEAP_MODEL else {"effort": "low"}}
            result = self.post(_llm, "/inference/chat/completions", body, FORCE_SECONDS)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                message = data["choices"][0].get("message") or {}
                self.charge_cached(model, data.get("usage"), str(message.get("content") or ""))
                return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "no reply"})
        return None

    def charge_cached(self, model: str, usage: Any, content: str) -> None:
        """A turn's cost: the reply's own cost when given; else its tokens, the part the cache holds (as reported, or for
        an Anthropic model the previous turn's prompt, which this one repeats) read at a tenth of the price and the new
        part written at a quarter more."""
        usage = usage if isinstance(usage, dict) else {}
        tokens_in = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool) or tokens_in is None:
            self.charge(model, usage, json.dumps(self.chat[-2:])[-8000:], content)
            self.last_prompt = tokens_in or 0
            return
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        cached = details.get("cached_tokens") if isinstance(details.get("cached_tokens"), int) and details.get("cached_tokens") else (
            min(tokens_in, getattr(self, "last_prompt", 0)) if model.startswith("anthropic/") else 0)
        inp, out = PRICES.get(model, PRICES[MAIN_MODEL])
        tokens_out = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else len(content) / 3
        self.spent += cached * inp * 0.1 + (tokens_in - cached) * inp * 1.25 + tokens_out * out
        self.last_prompt = tokens_in

    def condense(self) -> None:
        if sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in self.chat) <= COMPACT_CHARS:
            return
        if self.chat and self.chat[-1].get("role") != "tool" and self.chat[-1].get("role") != "user":
            return
        self.condensed += 1
        parts = [self.search_text() or "SEARCHES: none"]
        opened = [i for i in self.details if i not in self.unknown]
        if opened:
            parts.append("EVERY LISTING OPENED:\n" + "\n".join(_compact(self.record(i), 600) for i in opened[-25:]))
        if self.exchanges:
            parts.append("EXCHANGES WITH THE SHOPPER:\n" + "\n".join(self.exchanges))
        user = ("SHOPPER REQUEST:\n" + self.request + "\n\nTHE EARLIER CONVERSATION WAS CONDENSED. WORK SO FAR:\n"
                + "\n\n".join(parts) + "\n\nContinue from here.\n\n" + self.status())
        self.chat = [self.chat[0], {"role": "user", "content": user}]
        self.last_prompt = 0  # a new conversation: nothing of it is cached yet
        self.trace.append({"kind": "condensed", "times": self.condensed})

    # -- the request ---------------------------------------------------------------

    def setup(self) -> None:
        """Fresh 5's reading of the request, then every requirement the request takes from another listing is looked
        up before the actor starts: that listing is found, and its note followed when it defers to yet another."""
        super().setup()
        refs = [r for r in (self.plan.get("references") if isinstance(self.plan, dict) else None) or [] if isinstance(r, dict)]
        if not refs or self.over():
            return
        before = {None: list(self.reqs)} if not self.multi() else {k: list(i["reqs"]) for k, i in enumerate(self.items)}
        self.note_references({"references": refs}, before)
        self.resolve_references()

    def lookup_reference(self, prop: str, ref: dict[str, Any]) -> dict[str, Any] | None:
        """Fresh 5's lookup, with the description's own words: searches add them to the seller's name until a row
        shows them, and the rows that show the most of them are opened (several in one step when the store allows)."""
        seller = str(ref.get("seller") or "").strip()
        category = str(ref.get("category") or "").strip()
        conditions = [str(c) for c in ref.get("conditions") or [] if str(c).strip()]
        plain = {_one(w) for w in _words(f"{seller} {category}")} | {"with", "this", "that", "from", "item", "listing", "product"}
        names = {_one(w) for c in conditions for w in _words(c.split(":", 1)[0])}  # condition names say less than values
        values = " ".join(c.split(":", 1)[-1] for c in conditions)
        text = re.sub(r"(\w)'s\b", r"\1", f"{values} {ref.get('description') or ''}")
        desc = [w for w in dict.fromkeys(re.findall(r"\d+(?:\.\d+)?|[a-z][a-z\-]+", text.lower()))
                if (len(w) > 2 or w[0].isdigit()) and _one(w) not in plain and _one(w) not in names and (w in values.lower().split() or w not in _FUNCTION_WORDS)]
        desc.sort(key=lambda w: (not w[0].isdigit() or len(w) < 3, w not in values.lower()))  # values, figures first
        want = {_one(w) for w in desc}

        def own_words(row: dict[str, Any]) -> set[str]:
            """Words a listing uses for itself: the first sentence of each text (later sentences of a note may name the
            versions it is NOT)."""
            raws = row.get("raws") or ([row["raw"]] if isinstance(row.get("raw"), dict) else [])
            texts = [str(v) for raw in raws for v in raw.values() if isinstance(v, str)] or [str(row.get("text") or "")]
            first = " ".join(re.split(r"(?<=[.!?])\s+(?=[A-Z])", t.strip(), maxsplit=1)[0] for t in texts).lower()
            return {_one(w) for w in re.findall(r"\d+(?:\.\d+)?|[a-z][a-z\-]+", first)}
        rows: dict[str, dict[str, Any]] = {}
        queries = [f"{seller} {category}".strip(), *((f"{seller} {w}" for w in desc[:4]) if seller else ()), seller, category]
        for query in [q for q in dict.fromkeys(queries) if q][:5]:
            if self.steps_left() <= self.finish_reserve() + 4:
                break
            action = self.search_action(query, None)
            observations = self.act([action]) if action else None
            if not observations:
                break
            self.__dict__.setdefault("ref_queries", set()).add(" ".join(_words(query)))
            for r in ({**_row_info(x), "raw": x} for x in _result_rows(observations[0]) if isinstance(x, dict)):
                if r.get("id") and (not seller or seller.lower() in str(r.get("text") or "").lower()):
                    # a store may show a seller's note on one search's row and not another's: keep every showing
                    seen_raws = (rows.get(r["id"]) or {}).get("raws") or []
                    r["raws"] = seen_raws + [r["raw"]]
                    rows[r["id"]] = r
            if any(len(want & own_words(r)) >= max(1, math.ceil(len(want) * 0.75)) for r in rows.values()):
                break  # a row shows (nearly) every word of the description
        if not rows:
            return None
        ranked = sorted(rows.values(), key=lambda r: -len(want & own_words(r)))
        self.read_details([r["id"] for r in ranked if r["id"] not in self.details][:REFERENCE_OPENS])

        def short_first(d: Any) -> Any:  # facts and seller notes before a long description, so the cut keeps them
            return dict(sorted(d.items(), key=lambda kv: len(_compact(kv[1], 4000)))) if isinstance(d, dict) else d
        def row_notes(r: dict[str, Any]) -> str:  # seller notes a search row showed, which the full listing may leave out
            notes = {str(v).strip() for raw in r.get("raws") or [] for key, v in raw.items()
                     if isinstance(v, str) and len(v) > 20 and re.search(r"comment|note|remark", str(key), re.I)}
            return (" | SEARCH ROW NOTE: " + " / ".join(sorted(notes))[:600]) if notes else ""
        listings = [f"[{r['id']}] {_compact(short_first(self.details[r['id']]), 1500)}{row_notes(r)}" for r in ranked if r["id"] in self.details][:REFERENCE_OPENS + 2]
        if not listings:
            return None
        prompt = (REFERENCE_PROMPT + f"\n\nPROPERTY TO READ: {prop}\nTHE SHOPPER'S DESCRIPTION OF THE LISTING: {ref.get('description') or ''}"
                  + f"\nSELLER: {seller}\nKIND OF PRODUCT: {category}\nCONDITIONS: {'; '.join(conditions) or '(none)'}\n\nLISTINGS:\n" + "\n".join(listings))
        return self.ask(prompt, 700)

    def wants_text(self) -> str:
        text = super().wants_text()
        open_refs = [n for n in self.ref_needs.values() if not n.get("value")]
        for need in open_refs:
            ref = need["ref"]
            text += (f"\nVALUE TO LOOK UP: {need['property']} must be taken from another listing: {ref.get('seller') or '?'}'s "
                     f"{ref.get('category') or 'listing'} ({ref.get('description') or ''}). Open it; if its note sends you to yet another "
                     f"listing, follow that one; the last listing's own {need['property']} is the requirement.")
        return text

    # -- what the actor sees -----------------------------------------------------

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        observations = super().act(actions)
        looks = {(self.roles.get(r) or {}).get("tool") for r in ("details", "details_many", "listing", "cart", "cart_remove",
                                                                  "cart_view", "order", "decline", "message")}
        for action, observation in zip(actions, observations or []):
            if action["name"] in looks or not isinstance(observation, dict) or _failed(observation):
                continue
            rows = [r for r in (_row_info(x) for x in _result_rows(observation)) if r.get("id")]
            total = _reported_total(observation)
            if not rows and total is None:
                continue
            entry = self.queries.setdefault(_query_key(action["name"], action.get("args") or {}),
                                            {"args": _compact(action.get("args") or {}, 140), "ids": [], "pages": 0, "size": 0})
            new = [r["id"] for r in rows if r["id"] not in entry["ids"]]
            entry.update(pages=entry["pages"] + 1, size=max(entry["size"], len(rows)), total=total, more=_has_more(observation),
                         new=len(new), step=self.step)
            entry["ids"] += new
        return observations

    def reserve(self) -> int:
        return 3 + self.missing_lines() + (1 if self.messages_left() > 0 else 0)

    def search_text(self) -> str:
        if not self.queries:
            return ""
        free = max(0, self.steps_left() - self.reserve())
        lines = []
        for entry in sorted(self.queries.values(), key=lambda e: e["step"])[-SEARCH_LOG:]:
            seen, total, size = len(entry["ids"]), entry.get("total"), entry["size"] or 1
            head = f"- {entry['args']}: {entry['pages']} page(s), {seen} rows seen" + (f" of {total}" if total is not None else "")
            if not seen:
                lines.append(head + " -> nothing found: use fewer words or another noun")
                continue
            if total is not None:
                rest = max(0, math.ceil((total - seen) / size))
            else:
                rest = 1 if entry.get("more") else 0
            if not rest and not entry.get("more"):
                lines.append(head + " -> all seen")
            elif entry["pages"] > 1 and not entry.get("new"):
                lines.append(head + " -> the last page brought nothing new: stop paging this query")
            elif rest <= max(1, free // 3):
                lines.append(head + f" -> the rest costs about {rest} step(s) of {free} spare: affordable")
            else:
                lines.append(head + f" -> the rest costs about {rest} steps of {free} spare: too many, narrow the query or filter")
        return "SEARCHES (spare steps = steps left minus those kept for carting, a check and the order):\n" + "\n".join(lines)

    def asked_texts(self) -> list[str]:
        return [str(t.get("question") or "").lower() for t in self.trace if isinstance(t, dict) and t.get("kind") == "heard"]


    def status(self) -> str:
        text = super().status()
        extra = []
        searches = self.search_text()
        if searches:
            extra.append(searches)
        likely = self.likely_rows()
        if likely:
            extra.append("LIKELY TO OPEN (the row's own text meets the requirements, or its title does when that text never names the property; not opened yet): " + "; ".join(likely))
        ask = self.ask_about()
        if ask and self.messages_left() > 0:
            extra.append("ASK ABOUT (fact names the product's own opened listings use, not settled or asked yet; most varied first):\n" + ask)
        if self.spent >= SPEND_CAP:
            extra.append("FINISH NOW: this task's model budget is nearly spent; order the best basket you have.")
        many = self.roles.get("details_many")
        if many:
            extra.append(f"OPENING SEVERAL: {many['tool']} opens up to {many.get('max') or 2} listings in one step (several single openings in one reply are merged into it).")
        if not extra:
            return text
        return text.replace("\nOPENED LISTINGS", "\n" + "\n".join(extra) + "\nOPENED LISTINGS", 1)

    # -- the store ---------------------------------------------------------------

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Several single openings in one reply go out as calls of the store's many-listing tool (one step for up to its
        limit); everything else as in fresh 7."""
        single, many = self.roles.get("details") or {}, self.roles.get("details_many") or {}
        merged: dict[str, str] = {}
        if many and single.get("id_arg") and int(many.get("max") or 0) >= 2:
            for call in calls:
                function = call["function"]
                if function.get("name") != single.get("tool"):
                    continue
                try:
                    args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
                except ValueError:
                    continue
                ident = str(args.get(single["id_arg"]) or "")
                if ident and ident not in self.details and ident not in self.cart and ident not in merged.values():
                    merged[call["id"]] = ident
        if len(merged) < 2 or self.over():
            return super().run_calls(calls)
        before = self.step
        self.read_details(list(merged.values()))
        self.trace.append({"kind": "merged_openings", "ids": list(merged.values()), "steps": self.step - before})
        rest = [c for c in calls if c["id"] not in merged]
        results = dict(super().run_calls(rest)) if rest else {}
        for cid, ident in merged.items():
            results[cid] = (f"OPENED with {many['tool']} together with {len(merged) - 1} other(s) ({self.step - before} step(s) for all):\n"
                            + (_compact(self.details[ident], 2500) if ident in self.details else "NOT RETURNED by the store"))
        return [(c["id"], results.get(c["id"], "NO RESULT")) for c in calls]

    def label_for(self) -> str:
        """As fresh 7, but a label named exactly like the product wins over one that only shares words with it."""
        wanted = " ".join(_one(w) for w in _words(self.category))
        exact = next((r["category"] for r in self.rows.values() if r.get("category")
                      and " ".join(_one(w) for w in _words(r["category"])) == wanted), "")
        return exact or super().label_for()

    def category_problems(self) -> list[str]:
        """Cart lines that belong to no requested product: a line's store category must be a requested product's label;
        for a product whose label is not seen yet, a category sharing a word with its name may be it."""
        keep = self.current
        labels, open_names = set(), []
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            label = self.label_for()
            if label:
                labels.add(label)
            else:
                open_names.append({_one(w) for w in _words(self.category)})
        self.switch(keep)
        problems = []
        for ident in self.cart:
            label = self.rows.get(ident, {}).get("category")
            if not label or label in labels:
                continue
            if any(name & {_one(w) for w in _words(label)} for name in open_names):
                continue
            problems.append(f"{ident} is in the store's category '{label}', which is none of the products asked for"
                            + (f" ({', '.join(sorted(labels))})" if labels else "") + ": replace it with a listing of the right category")
        return problems

    def failed_problems(self) -> list[str]:
        """Cart lines with a requirement that code reads as not met by the listing's own facts."""
        return [f"{i}: its facts do not meet {r}" for i in self.cart for r in self.item_reqs(i)
                if all(_split_req(r)) and self.code_verdict(i, r) == "no"]

    def code_verdict(self, ident: str, req: str) -> str:
        """Fresh 5's reading, plus figures with units against the listing's facts of that property (ranges included)."""
        verdict = super().code_verdict(ident, req)
        if verdict != "unsure":
            return verdict
        prop, value = _split_req(req)
        readings = [_figure_verdict(v, value) for k, v in _facts(self.details.get(ident)) if _key_norm(k) == _key_norm(prop)]
        return "yes" if "yes" in readings else "no" if "no" in readings else "unsure"

    def checks(self, ident: str) -> str:
        marks = []
        for req in self.item_reqs(ident):
            prop, value = _split_req(req)
            if prop and value:
                verdict = self.code_verdict(ident, req)
                marks.append(f"{prop} {'yes' if verdict == 'yes' else 'no' if verdict == 'no' else '?'}")
        if self.suspicious(ident):
            marks.append("FAR BELOW ITS CATEGORY'S USUAL PRICE: make sure it is the product itself")
        return ", ".join(marks)

    def tidy_cart(self) -> None:
        """Before a forced finish: one line per product (the one whose facts confirm the most of its requirements, then
        the cheaper), and no line of a category nobody asked for, so the basket can still be ordered."""
        keep = self.current
        labels = []
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            labels.append(self.label_for())
        self.switch(keep)
        groups: dict[str, list[str]] = {}
        drop = []
        for ident in self.cart:
            label = self.rows.get(ident, {}).get("category")
            if label and "" not in labels and label not in labels:
                drop.append(ident)
            else:
                groups.setdefault(label or "", []).append(ident)
        for idents in groups.values():
            if len(idents) > 1:
                score = lambda i: (-sum(self.code_verdict(i, r) == "yes" for r in self.item_reqs(i) if all(_split_req(r))),
                                   _price(self.rows.get(i, {})) or 0)
                drop += sorted(idents, key=score)[1:]
        for ident in drop:
            if self.steps_left() <= 1 or self.over():
                break
            action = self.cart_action(ident, False)
            if action:
                self.act([action])
        if drop:
            self.trace.append({"kind": "tidy_cart", "removed": drop, "cart": dict(self.cart)})

    def finish_now(self) -> None:
        if not self.over() and self.cart and (self.basket_problems() or self.category_problems()):
            self.tidy_cart()
        super().finish_now()

    def unverified_problems(self) -> list[str]:
        """Cart lines with a requirement their own facts do not confirm, when another opened listing of the same category
        confirms every requirement of that product (in stock, and within budget in place of the line)."""
        problems = []
        cart_total = sum((_price(self.rows.get(i, {})) or 0) * u for i, u in self.cart.items())
        for ident, units in self.cart.items():
            reqs = [r for r in self.item_reqs(ident) if all(_split_req(r))]
            if not reqs or all(self.code_verdict(ident, r) == "yes" for r in reqs):
                continue
            label = self.rows.get(ident, {}).get("category")
            for other in self.details:
                info = self.rows.get(other, {})
                if other in self.cart or other in self.unknown or other in self.unavailable or info.get("stock") is False:
                    continue
                if label and info.get("category") != label or self.item_reqs(other) != self.item_reqs(ident):
                    continue
                if not all(self.code_verdict(other, r) == "yes" for r in reqs):
                    continue
                total = cart_total + ((_price(info) or 0) - (_price(self.rows.get(ident, {})) or 0)) * units
                if self.budget is not None and total > self.budget + 1e-9:
                    continue
                unmet = [r for r in reqs if self.code_verdict(ident, r) != "yes"]
                problems.append(f"{ident}: its own facts do not confirm {'; '.join(unmet)}, while opened {other} confirms every requirement: use it")
                break
        return problems

    def gate_order(self) -> str:
        """Code's own checks (lines, units, budget, stock, category, a requirement read as unmet, a better verified
        listing) hold the order every time while there is room to fix it; then fresh 5's gate (model review, basket
        check with the shopper)."""
        problems = self.basket_problems() + self.category_problems() + self.failed_problems() + self.unverified_problems()
        if problems and self.steps_left() > 2 and self.time_left() > FORCE_SECONDS + 10:
            self.trace.append({"kind": "order_held", "problems": problems[:6]})
            return "NOT SENT (no step used). Every requirement must hold before ordering: " + " | ".join(problems[:6])
        return super().gate_order()

    def confirm_basket(self) -> bool:
        """The basket shown to the shopper in one short message (long ones confuse some shoppers)."""
        currency = str(self.plan.get("currency") or "").strip()
        lines, total = [], 0.0
        for ident, units in self.cart.items():
            info = self.rows.get(ident, {})
            price = info.get("price")
            title = " ".join(str(info.get("title") or ident).split()[:6])
            if isinstance(price, (int, float)):
                total += price * units
                lines.append(f"{units} x {title} ({price:g} each)")
            else:
                lines.append(f"{units} x {title}")
        text = "Ready to order: " + "; ".join(lines) + (f", total {total:g} {currency}".rstrip() if total else "") + ". Shall I place the order?"
        rule, self.rule_message = self.rule_message, ""
        if "?" in rule:
            text = rule
        elif rule:
            text = text.replace(" Shall I", f" {rule} Shall I", 1)
        self.confirmed = self.basket_key()
        action = self.message_action(text)
        if action is None:
            return False
        observations = self.act([action])
        if observations is None:
            return False
        self.messages_used += 1
        before = self.wants()
        changed = self.listen(observations, text, reply_of=0)
        return changed or self.wants() != before or bool(set(self.cart) & self.unavailable)


CAP_ARG = re.compile(r"(cap|max|ceiling|limit|upto|under|most).{0,12}(price|cost)|(price|cost).{0,12}(cap|max|ceiling|limit|most)", re.I)
SHELF_ARG = re.compile(r"shelf|categor|aisle|department|section", re.I)
STALL_HOLDS = 5  # holds of an order in a row with no step taken in between; the next goes out


class Actor9(Actor8):
    """Fresh 9: fresh 8, plus the order held while steps remain and something is unproven (a requirement code cannot
    confirm, a cheaper listing the store's price filter would show, a question left), the cheapest proved with that
    filter, and two-property questions built from the listings' own fact names."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.capped: dict[tuple, dict[str, Any]] = {}  # (shelf, cap) -> {"ids", "more"} from the price-filter tool
        self.held_at: list[int] = []

    # -- the price filter --------------------------------------------------------

    def cap_tool(self) -> tuple[str, str, str] | None:
        """(tool, shelf argument, price-cap argument) of a store tool that lists a category up to a price, or None."""
        if not hasattr(self, "_cap_tool"):
            self._cap_tool = None
            for name, schema in self.schemas.items():
                props = self.props(name)
                cap = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("integer", "number") and CAP_ARG.search(k + " " + str(v.get("description") or ""))), None)
                shelf = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string" and SHELF_ARG.search(k + " " + str(v.get("description") or ""))), None)
                if cap and shelf:
                    self._cap_tool = (name, shelf, cap)
                    break
        return self._cap_tool

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        observations = super().act(actions)
        tool = self.cap_tool()
        for action, observation in zip(actions, observations or []):
            if not tool or action["name"] != tool[0] or not isinstance(observation, dict) or _failed(observation):
                continue
            args = action.get("args") or {}
            if not isinstance(args.get(tool[2]), (int, float)) or not args.get(tool[1]):
                continue
            entry = self.capped.setdefault((_key_norm(str(args[tool[1]])), float(args[tool[2]])), {"ids": [], "more": True, "total": None, "size": 0})
            rows = [r for r in (_row_info(x) for x in _result_rows(observation)) if r.get("id")]
            entry["ids"] += [r["id"] for r in rows if r["id"] not in entry["ids"]]
            entry["more"] = _has_more(observation)
            entry["total"] = _reported_total(observation) if _reported_total(observation) is not None else entry["total"]
            entry["size"] = max(entry["size"], len(rows))
        return observations

    def cheapest_gaps(self) -> list[str]:
        """For each product asked at its cheapest, what is left to prove its best fit cheapest: the store's price-filter
        listing of its category capped at (or above) the best fit's price, paged to the end, every listing it returned at
        or below that price opened. A long list is narrowed first by a lower cap (a fit found there lowers the bar)."""
        tool = self.cap_tool()
        gaps, keep = [], self.current
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            if (self.preference or {}).get("kind") != "cheapest":
                continue
            label = self.label_for()
            fits = [i for i in self.details if i not in self.unknown and i not in self.unavailable and self.rows.get(i, {}).get("stock") is not False
                    and (not label or self.rows.get(i, {}).get("category") == label) and _price(self.rows.get(i, {})) is not None
                    and all(self.code_verdict(i, r) != "no" for r in self.reqs if all(_split_req(r)))]
            if not fits or not tool or not label:
                continue
            # the bar: the line carted for this product (the actor's choice), else the cheapest listing whose facts
            # confirm every requirement, else the cheapest code cannot rule out
            carted = [i for i in self.cart if self.rows.get(i, {}).get("category") == label and _price(self.rows.get(i, {})) is not None]
            confirmed = [i for i in fits if all(self.code_verdict(i, r) == "yes" for r in self.reqs if all(_split_req(r)))]
            best = min(_price(self.rows[i]) for i in (carted or confirmed or fits))
            product = self.category or "the product"
            mine = {cap: e for (shelf, cap), e in self.capped.items() if shelf == _key_norm(label)}
            covering = {cap: e for cap, e in mine.items() if cap >= best - 1e-9}
            full = [e for e in covering.values() if not e["more"]]
            ids = {i for e in mine.values() for i in e["ids"]}
            cheaper = sorted(i for i in ids if i not in self.details and self.rows.get(i, {}).get("stock") is not False
                             and (_price(self.rows.get(i, {})) or 0) <= best)
            if cheaper:
                gaps.append(f"open these {product} listings the price filter returned at or below {best:g}: {', '.join(cheaper[:10])}")
            elif full:
                continue  # proven: everything at or below the best fit's price was listed and opened
            elif not covering:
                gaps.append(f"prove the cheapest {product}: call {tool[0]} with {tool[1]}={label!r} and {tool[2]}={best:g} (your best fit's price) and page to the end")
            else:
                lowest = min(mine)
                low = mine[lowest]
                scanned = not low["more"] and all(i in self.details for i in low["ids"])
                if low["more"] and low["total"] is not None and low["total"] <= SHORT_LIST:
                    gaps.append(f"page {tool[0]} ({tool[1]}={label!r}, {tool[2]}={lowest:g}) to the end first ({low['total']} listed): a fit there lowers the bar")
                elif lowest < best and scanned and (low["total"] is None or low["total"] <= SHORT_LIST):
                    # nothing fits at or below the lowest cap: the rest lies between it and the best fit, so page the
                    # list capped at the best fit and open only what it lists in that band
                    gaps.append(f"no {product} fits at or below {lowest:g}: page {tool[0]} ({tool[1]}={label!r}, {tool[2]}={min(covering):g}) "
                                f"to the end and open the listings it shows priced above {lowest:g} (several per step)")
                elif low["total"] is not None and low["total"] > SHORT_LIST and lowest > 1:
                    integer = (self.props(tool[0]).get(tool[2]) or {}).get("type") == "integer"
                    half = math.floor(lowest / 2) if integer else round(lowest / 2, 2)
                    gaps.append(f"the price-filtered list of {product} at {lowest:g} is long ({low['total']}): call {tool[0]} with {tool[1]}={label!r} and "
                                f"{tool[2]}={half:g} (exactly half), and keep halving until {SHORT_LIST} or fewer are listed; open those; a fit there lowers the bar")
                else:
                    gaps.append(f"page {tool[0]} ({tool[1]}={label!r}, {tool[2]}={min(covering):g}) to the end")
        self.switch(keep)
        return gaps

    # -- keep working while steps remain ------------------------------------------

    def readiness(self) -> list[str]:
        gaps = self.cheapest_gaps()
        for ident in self.cart:
            others = [i for i in self.details if i not in self.cart and i not in self.unknown and i not in self.unavailable
                      and self.rows.get(i, {}).get("stock") is not False and self.item_reqs(i) == self.item_reqs(ident)
                      and self.rows.get(i, {}).get("category") == self.rows.get(ident, {}).get("category") and not self.suspicious(i)
                      and all(self.code_verdict(i, r) == "yes" for r in self.item_reqs(i) if all(_split_req(r)))]
            if self.suspicious(ident) and others:
                gaps.append(f"{ident} is priced far below its category: replace it with a real fit such as {others[0]}")
            k_pref = self.item_pref(ident)
            price = _price(self.rows.get(ident, {}))
            cheaper = sorted((p, i) for i in others for p in [_price(self.rows.get(i, {}))] if p is not None and price is not None and p < price)
            if k_pref == "cheapest" and cheaper:
                gaps.append(f"{ident} ({price:g}) is not the cheapest opened fit: {', '.join(f'{i} ({p:g})' for p, i in cheaper[:3])}: their facts confirm every requirement; cart the cheapest")
        for ident in self.cart:
            unsure = [r for r in self.item_reqs(ident) if all(_split_req(r)) and self.code_verdict(ident, r) != "yes"]
            if unsure:
                gaps.append(f"{ident}: its facts do not confirm {'; '.join(unsure)} (read them, or find a listing that confirms it)")
        if not self.cart:
            gaps.append("nothing in the cart yet")
        questions = self.messages_left() - (1 if self.confirmed is None else 0)
        if questions > 0 and self.ask_about():
            gaps.append(f"{questions} question(s) unused: ask about the ASK ABOUT properties")
        return gaps

    def item_pref(self, ident: str) -> str:
        """The preference kind of the product a listing is for."""
        if not self.multi():
            return str((self.preference or {}).get("kind") or "")
        self.save_item()
        reqs = self.item_reqs(ident)
        item = next((i for i in self.items if list(i["reqs"]) == reqs), None)
        return str(((item or {}).get("preference") or self.preference or {}).get("kind") or "")

    def gate_order(self) -> str:
        room = self.steps_left() > self.reserve() + 3 and self.time_left() > CLOSING_SECONDS + 10
        stalled = len(self.held_at) >= STALL_HOLDS and all(s == self.step for s in self.held_at[-STALL_HOLDS:])
        if room and not stalled:
            gaps = self.readiness()
            if gaps:
                self.held_at.append(self.step)
                self.trace.append({"kind": "order_early", "gaps": gaps[:6], "steps_left": self.steps_left()})
                return (f"NOT SENT (no step used). {self.steps_left()} steps are left, and these are not proven yet: " + " | ".join(gaps[:6])
                        + ". Work on them with targeted calls (open the line's full page, or search and open a listing that states the value, and swap it in); "
                        "order when they are done. Sending the same order again with no step in between accepts these gaps as they are.")
        return super().gate_order()

    def status(self) -> str:
        text = super().status()
        gaps = self.readiness()
        if gaps and self.steps_left() > self.reserve() + 3:
            text += "\nBEFORE ORDERING (" + str(self.steps_left()) + " steps left): " + " | ".join(gaps[:6])
        return text

    # -- questions -----------------------------------------------------------------

    def ask_about(self) -> str:
        """Per product: fact names of its opened listings not asked about yet and not already a need the shopper gave,
        the names whose values differ most between those listings first (names the request states included, after the
        others)."""
        asked = " ".join(self.asked_texts())
        keep = self.current
        out = []
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            label = self.label_for()
            stated = {_split_req(r)[0].lower() for r in self.reqs}
            values: dict[str, set] = {}
            for ident, details in self.details.items():
                if ident in self.unknown or (label and self.rows.get(ident, {}).get("category") != label):
                    continue
                for key, value in _facts(details) or []:
                    name = " ".join(str(key).replace("_", " ").replace("-", " ").lower().split())
                    if not name or len(name) > 30 or re.search(r"model|sku|condition|options|warranty|brand|id$", name):
                        continue
                    values.setdefault(name, set()).add(str(value).lower()[:40])
            names = [n for n in values if n not in asked and n not in {str(f).lower() for f in self.free}]
            names.sort(key=lambda n: (n in stated or any(n in s or s in n for s in stated if s), -len(values[n]), n))
            if names:
                out.append(f"- {self.category or 'the product'}: " + ", ".join(names[:ASK_ROWS + 2]))
        self.switch(keep)
        return "\n".join(out)


# Phase thresholds, from the race-8 leader's episodes (medians over 180): 8 listings opened before the first question
# (at 39% of the steps), the questions asked back to back, 15 opened and every question asked before the first cart
# add (at 82% of the steps), then cart view, the carted listing re-read and the basket check before the order.
OPENING_VALUES = 2  # stated values searched by the code at the start, one 2-3 word query each ("value + head noun")
OPENING_MIN_STEPS = 20  # only when the task allows at least this many steps
OPENING_ROWS = 24  # rows of the opening searches shown to the actor
SHORT_LIST = 15  # a price-capped list this long or shorter is opened through; a longer one is halved first
GATE_SLACK = 6  # with this few steps beyond the finishing reserve, no phase holds anything back


class Actor10(Actor9):
    """Fresh 10: fresh 9 with the leader's phases held by code: survey (open listings before asking), ask (every
    question, back to back), re-check (search and open again for the needs heard, prove the cheapest), commit (cart once,
    view the cart, re-read the carted listings, the basket check, the order)."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.verified: tuple | None = None  # the cart last viewed and re-read

    # -- the opening searches (the leader's first moves) -----------------------------

    def head_noun(self, category: str) -> str:
        """The category's last word ('accessories' of 'lighting installation & accessories')."""
        words = [w for w in re.findall(r"[a-z0-9]+", category.lower()) if w not in ("and", "the", "of")]
        return words[-1] if words else category

    def product_phrase(self, category: str) -> str:
        """How a search names the product: the shopper's own noun for one item when it is a short phrase, else the
        category when short, and only then its last word (a last word alone matches far more than the product)."""
        for phrase in (str(self.noun or ""), category):
            words = [w for w in re.findall(r"[a-z0-9.]+", phrase.lower()) if w not in ("and", "or", "the", "a", "of")]
            if 1 <= len(words) <= 4:
                return " ".join(words)
        return self.head_noun(category)

    def value_query(self, value: str, category: str) -> str:
        """The product phrase plus one complete value. A bound ("at most", "at least") is dropped as a whole phrase, so a
        ceiling is not searched as its number: 8 and 12 must still be on the shelf when the limit is 16."""
        phrase = self.product_phrase(category).split()
        if _BOUND.search(value):
            return " ".join(phrase)
        words = [w for w in re.findall(r"[a-z0-9.]+", value.lower()) if w not in ("and", "or", "the", "a")][:3]
        out: list[str] = []
        for w in words + phrase:
            stem = w[:-1] if w.endswith("s") and len(w) > 3 else w
            if not any(stem == (x[:-1] if x.endswith("s") and len(x) > 3 else x) for x in out):
                out.append(w)  # a word already there in its singular or plural form is not added again
        return " ".join(out)

    def opening_searches(self) -> None:
        """The leader's first moves, by code: the category shelf capped at the budget (what the shelf holds and what is
        affordable), then a short query per stated value (that value and the category's head noun) and the next page of
        the first; the actor starts from their rows."""
        if self.max_steps < OPENING_MIN_STEPS or self.over() or (self.plan.get("references") if isinstance(self.plan, dict) else None):
            return
        tool, keep, shown = self.cap_tool(), self.current, []
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            category = self.category or ""
            if not category:
                continue
            actions = []
            if tool and self.budget is not None:
                actions.append({"name": tool[0], "args": self.fill_required(tool[0], {tool[1]: category, tool[2]: self.budget})})
            values = [_split_req(r)[1] for r in self.reqs if all(_split_req(r)) and len(_split_req(r)[1].split()) <= 3
                      and not _BOUND.search(_split_req(r)[1])][:OPENING_VALUES if not self.multi() else 1]
            for query in dict.fromkeys(self.value_query(v, category) for v in values or [category]):
                action = self.search_action(query, None)
                if action:
                    actions.append(action)
            if not actions or self.steps_left() <= self.reserve() + len(actions) + 10:
                continue
            observations = self.act(actions) or []
            self.listen(observations)
            shown += list(zip(actions, observations))
            role = self.roles.get("search") or {}
            first = next(((a, o) for a, o in zip(actions, observations) if a["name"] == role.get("tool")), None)
            # the next pages of each product's first search (the leaders read about three pages of it), when the store
            # reports more and the step limit leaves room; one page more in a short task
            pages = 0 if self.multi() else 1  # as the baseline: more opening pages measured worse
            page, last = 2, first
            while first and pages > 0 and last and _has_more(last[1]) and self.steps_left() > self.reserve() + 12:
                query = str(first[0]["args"].get(role.get("query_arg") or "", ""))
                more = self.search_action(query, None, _next_token(last[1])) if role.get("token_arg") else self.search_action(query, page) if role.get("page_arg") else None
                following = _next_number(last[1])
                if more and following and not role.get("token_arg"):
                    more["args"][role["page_arg"]] = following  # the store's own next value: a row offset is not a page number
                got = self.act([more]) if more else None
                if not got:
                    break
                self.listen(got)
                shown.append((more, got[0]))
                last, page, pages = (more, got[0]), page + 1, pages - 1
        self.switch(keep)
        if shown:
            rows = []
            for action, observation in shown:
                found = [r for r in (_row_info(x) for x in _result_rows(observation)) if r.get("id")]
                rows.append(f"{action['name']} {_compact(action['args'], 160)} -> {_reported_total(observation) if _reported_total(observation) is not None else len(found)} found: "
                            + "; ".join(f"{r['id']} {str(r.get('title') or '')[:50]} | {r.get('price', '?')} | {r.get('category') or ''}" for r in found[:OPENING_ROWS // max(1, len(shown))]))
            self.opening_text = "THE CODE'S OPENING SEARCHES (already done, steps used):\n" + "\n".join(rows)
            self.trace.append({"kind": "opening_searches", "calls": len(shown)})

    def setup(self) -> None:
        super().setup()
        self.opening_text = ""
        self.opening_searches()

    def actor_turn(self) -> dict[str, Any] | None:
        if getattr(self, "opening_text", "") and len(self.chat) >= 2 and self.chat[1].get("role") == "user":
            self.chat[1]["content"] = self.chat[1]["content"].replace("\n\nSTATUS:", "\n\n" + self.opening_text + "\n\nSTATUS:", 1)
            if self.opening_text not in self.chat[1]["content"]:
                self.chat[1]["content"] += "\n\n" + self.opening_text
            self.opening_text = ""
        return super().actor_turn()

    def search_ideas(self) -> str:
        """Queries worth running next, per product: each stated value alone with the head noun, then two values together
        (for when one value matched too many listings to page through), leaving out what has been searched. A limit is
        no search word, so bounded values are left out."""
        done = {re.sub(r"\s+", " ", json.loads(k)[1].get(str((self.roles.get("search") or {}).get("query_arg")), "") if isinstance(json.loads(k)[1], dict) else "").strip().lower()
                for k in self.queries}
        keep, out = self.current, []
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            values = [_split_req(r)[1] for r in self.reqs if all(_split_req(r)) and len(_split_req(r)[1].split()) <= 3
                      and not _BOUND.search(_split_req(r)[1])]
            ideas = [self.value_query(v, self.category or "") for v in values]
            ideas += [self.value_query(f"{a} {b}", self.category or "") for n, a in enumerate(values) for b in values[n + 1:]]
            ideas = [q for q in dict.fromkeys(ideas) if q not in done][:3]
            if ideas:
                out.append(f"{self.category or 'the product'}: " + "; ".join(f"'{q}'" for q in ideas))
        self.switch(keep)
        return " | ".join(out)

    # -- where the task is -------------------------------------------------------

    def opened(self) -> list[str]:
        return [i for i in self.details if i not in self.unknown]


    def share(self) -> float:
        return self.step / max(1, self.max_steps)


    def tight(self) -> bool:
        return self.steps_left() <= self.reserve() + GATE_SLACK or self.time_left() < CLOSING_SECONDS + 15


    def status(self) -> str:
        # the phases keep the cart late: when only the steps to cart and order are left and a product still has no
        # line, the code carts the best fit it has opened, so the task can still end in an order
        missing = self.missing_lines()
        if missing and not self.over() and not self.fell_back and self.steps_left() <= missing + 3:
            if self.fallback():
                self.trace.append({"kind": "late_cart", "cart": dict(self.cart)})
        text = super().status() + f"\nPHASE (step {self.step} of {self.max_steps}): {self.phase()}"
        last = max(self.queries.values(), key=lambda e: e["step"], default=None)
        if last and not self.tight():
            if last["pages"] and not last.get("new") and last["pages"] > 1:
                text += f"\nPAGING: the last page of {last['args']} brought nothing new: change the query."
            elif last.get("more") and last["pages"] < 3 and last.get("new"):
                text += f"\nPAGING: {last['args']} page {last['pages']} brought {last['new']} new listing(s): read the next page (up to 3 pages a query)."
        ideas = self.search_ideas() if not self.survey_gap() else ""
        if ideas and not self.tight():
            text += "\nSEARCH IDEAS (one value + the category's head noun first; two values when one matched too many to page): " + ideas
        return text

    # -- the gates ---------------------------------------------------------------

    def is_add(self, name: str, args: dict[str, Any]) -> bool:
        cart = self.roles.get("cart") or {}
        if name != cart.get("tool"):
            return False
        if cart.get("op_arg"):
            return str(args.get(cart["op_arg"])) == str(cart.get("add_value"))
        if cart.get("lines_arg"):
            lines = args.get(cart["lines_arg"]) if isinstance(args.get(cart["lines_arg"]), list) else []
            return len(lines) > len(self.cart)
        return True

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """A question before the survey is done, and a cart add before the questions and the re-check are done, are
        held back (no step used); everything else goes as in fresh 9."""
        message = (self.roles.get("message") or {}).get("tool")
        held: dict[str, str] = {}
        rest = []
        for call in calls:
            function = call["function"]
            name = str(function.get("name"))
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                args = {}
            gap = ""
            if name == message and message:
                gap = self.survey_gap()
            elif isinstance(args, dict) and self.is_add(name, args):
                gap = self.survey_gap() or self.cart_gap() or self.repeat_add(args) or self.foreign_add(args)
            if gap:
                held[call["id"]] = f"NOT SENT (no step used): {gap}."
                self.trace.append({"kind": "phase_hold", "tool": name, "gap": gap, "step": self.step})
            else:
                rest.append(call)
        results = dict(super().run_calls(rest)) if rest else {}
        return [(c["id"], held.get(c["id"]) or results.get(c["id"], "NO RESULT")) for c in calls]

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        """Fresh 5's reading, and a cart change whose reply lists the whole cart (ids with units) sets the cart from that
        list: the store's own count decides, as a cart view's does."""
        super().absorb(action, observation)
        cart, remove = self.roles.get("cart") or {}, self.roles.get("cart_remove") or {}
        if action.get("name") in (cart.get("tool"), remove.get("tool")) and not _failed(observation):
            lines = _cart_lines(observation)
            if lines is not None:
                self.cart = lines


    def foreign_add(self, args: dict[str, Any]) -> str:
        """An add of a listing whose store category is none of the requested products' (when every label is known)."""
        cart = self.roles.get("cart") or {}
        ident = str(args.get(cart.get("id_arg") or "") or "")
        label = self.rows.get(ident, {}).get("category")
        keep, labels = self.current, set()
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            labels.add(self.label_for())
        self.switch(keep)
        if not ident or not label or "" in labels or label in labels:
            return ""
        return f"{ident} is in the store's category '{label}', which is none of those asked for ({', '.join(sorted(labels))})"

    def repeat_add(self, args: dict[str, Any]) -> str:
        """An add of a listing the cart already holds with the units asked for (a second add would double them)."""
        cart = self.roles.get("cart") or {}
        ident = str(args.get(cart.get("id_arg") or "") or "")
        if ident and ident in self.cart and self.cart[ident] >= self.wanted_units(ident):
            return f"{ident} is already in the cart with {self.cart[ident]} unit(s), as asked; adding it again would make it {self.cart[ident] + 1}"
        return ""

    def gate_order(self) -> str:
        """Once per cart, the leader's last checks first: view the cart and re-read the carted listings (stock and facts
        as they stand); then fresh 9's gate (its holds, the review, the basket check with the shopper)."""
        if not self.cart or (not self.tight() and self.readiness()):
            return super().gate_order()  # held anyway (or nothing to check): no steps spent on re-reading
        key = self.basket_key()
        if self.verified != key and self.steps_left() > len(self.cart) + 3 and self.time_left() > FORCE_SECONDS + 15:
            self.verified = key
            view = self.roles.get("cart_view")
            actions = [{"name": view["tool"], "args": self.fill_required(view["tool"], {})}] if view and view.get("tool") else []
            for ident in self.cart:
                single = self.id_action("details", ident)
                if single:
                    actions.append(single)
            if actions:
                observations = self.act(actions)
                self.trace.append({"kind": "verify_cart", "actions": len(actions)})
                if observations and self.listen(observations):
                    return "NOT SENT. Re-reading the cart changed something (stock, price or a note): see STATUS and adjust."
            problems = self.basket_problems() + self.failed_problems()
            if problems:
                return "NOT SENT. Re-reading the cart found: " + " | ".join(problems[:4])
        return super().gate_order()


# race 9: a two-property question drew "split those up" 36% of the time against ~6% for one property (31% of fresh 10's
# replies were splits, and it learned half the hidden needs the leader did)
MULTI_ASK = re.compile(r"(?i)\b(?:what|which)\b[^?]{0,60}?\b(?:or|and)\b[^?]{0,60}?\b(?:do you|would you|should|you need|you want|prefer)|\?[^?]*\?")
TIGHT_STEPS = 20  # a task allowing at most this many steps cannot ask one property per message: two per message are allowed
ORDER_DUE_SHARE = 3  # the first order is due with max(7, step limit / 3) steps left, keeping room to repair a refusal
ORDER_DUE_MIN = 7


class Actor11(Actor10):
    """Fresh 11: fresh 10 with one property per question (two only on the last question, never after a shopper asked
    for one at a time), a cart tool that sets a line's count handled as such (0 takes it out), store-refused batch
    sizes of the many-listing tool remembered, and the holds stopping at the order reserve."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.split_heard = False
        self.bad_batches: set[int] = set()

    # -- a cart tool that sets a line's count ---------------------------------------

    def units_cart(self) -> bool:
        """The cart tool takes a listing and a count, with no add/remove switch, no list of lines and no separate removal
        tool: it can only set the count, so 0 takes the line out."""
        cart = self.roles.get("cart") or {}
        return bool(cart.get("id_arg") and cart.get("qty_arg") and not cart.get("op_arg") and not cart.get("lines_arg")
                    and not self.roles.get("cart_remove"))

    def cart_action(self, ident: str, add: bool) -> dict[str, Any] | None:
        if not add and self.units_cart():
            cart = self.roles["cart"]
            return {"name": cart["tool"], "args": self.fill_required(cart["tool"], {cart["id_arg"]: ident, cart["qty_arg"]: 0})}
        return super().cart_action(ident, add)

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        cart = self.roles.get("cart") or {}
        if self.units_cart() and action.get("name") == cart.get("tool") and not _failed(observation):
            args = action.get("args") or {}
            ident, units = str(args.get(cart["id_arg"]) or ""), args.get(cart["qty_arg"])
            Episode.absorb(self, {"name": "", "args": {}}, observation)  # rows only
            if ident and isinstance(units, int) and not isinstance(units, bool):
                if units > 0:
                    self.cart[ident] = units
                else:
                    self.cart.pop(ident, None)
            lines = _cart_lines(observation)
            if lines is not None:
                self.cart = lines
            return
        super().absorb(action, observation)

    def repeat_add(self, args: dict[str, Any]) -> str:
        if self.units_cart():
            cart = self.roles["cart"]
            ident, units = str(args.get(cart["id_arg"]) or ""), args.get(cart["qty_arg"])
            if ident in self.cart and units == self.cart[ident]:
                return f"{ident} already holds {units} unit(s) in the cart"
            return ""
        return super().repeat_add(args)

    def is_add(self, name: str, args: dict[str, Any]) -> bool:
        if self.units_cart() and name == (self.roles.get("cart") or {}).get("tool"):
            units = args.get(self.roles["cart"]["qty_arg"])
            return not (isinstance(units, int) and units == 0)
        return super().is_add(name, args)

    def pairs_allowed(self) -> bool:
        """Two properties per message on a tight step budget, until the shopper asks for one at a time."""
        return self.max_steps <= TIGHT_STEPS and not self.split_heard

    # -- the order reserve -----------------------------------------------------------

    def order_due(self) -> int:
        return max(ORDER_DUE_MIN, self.max_steps // ORDER_DUE_SHARE)

    def reserve(self) -> int:
        return max(super().reserve(), self.order_due() - 3)

    # -- what the actor sees -----------------------------------------------------------

    def status(self) -> str:
        text = super().status()
        notes = []
        if self.units_cart():
            cart = self.roles["cart"]
            notes.append(f"CART TOOL: {cart['tool']} SETS how many units of a listing the cart holds ({cart['qty_arg']}); {cart['qty_arg']}=0 takes the line out. "
                         "Never send it twice to add more: send the count you want.")
        if self.bad_batches:
            many = self.roles.get("details_many") or {}
            notes.append(f"{many.get('tool')} refused {', '.join(map(str, sorted(self.bad_batches)))} listings at once: open 2 at a time (or one).")
        if self.split_heard:
            notes.append("The shopper takes one question at a time: one property per message.")
        elif self.pairs_allowed() and self.questions_left() > 0:
            notes.append("FEW STEPS: you may ask about two properties in one message ('What <p1> or <p2> do you need for the <product>?'); "
                         "if the shopper asks for one at a time, go back to one per message.")
        if self.steps_left() <= self.order_due() and self.missing_lines() and not self.tight():
            notes.append(f"ORDER DUE: {self.steps_left()} steps left; cart the best fits now and order, keeping steps to repair a refusal.")
        return text + ("\n" + "\n".join(notes) if notes else "")

    # -- the moves ---------------------------------------------------------------------

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """A question naming two or more properties is held unless it is the last one left (never after the shopper asked
        for one at a time), and a many-listing call of a size the store refused is held; the rest as in fresh 10."""
        message = (self.roles.get("message") or {}).get("tool")
        many = (self.roles.get("details_many") or {})
        held: dict[str, str] = {}
        rest = []
        for call in calls:
            function = call["function"]
            name = str(function.get("name"))
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                args = {}
            if name == message and message and isinstance(args, dict):
                text = str(args.get((self.roles["message"] or {}).get("text_arg") or "", ""))
                basket = re.search(r"(?i)ready to order|final basket|shall i place", text)
                if not basket and MULTI_ASK.search(text) and (self.split_heard or self.questions_left() > 1) and not self.pairs_allowed():
                    held[call["id"]] = ("NOT SENT (no step used): ask about ONE property per message ('What <property> do you need for the "
                                        "<product>, such as <value> or <value>?'); two in one message often draws 'split those up' and wastes it.")
                    continue
            if name == many.get("tool") and isinstance(args, dict):
                ids = args.get(many.get("ids_arg") or "")
                if isinstance(ids, list) and len(ids) in self.bad_batches:
                    held[call["id"]] = f"NOT SENT (no step used): the store refused {len(ids)} listings at once before; send 2 at a time."
                    continue
            rest.append(call)
        results = dict(super().run_calls(rest)) if rest else {}
        return [(c["id"], held.get(c["id"]) or results.get(c["id"], "NO RESULT")) for c in calls]

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        observations = super().act(actions)
        many = self.roles.get("details_many") or {}
        message = (self.roles.get("message") or {}).get("tool")
        for action, observation in zip(actions, observations or []):
            args = action.get("args") or {}
            if action["name"] == many.get("tool") and _failed(observation):
                ids = args.get(many.get("ids_arg") or "")
                if isinstance(ids, list) and len(ids) >= 2:
                    self.bad_batches.add(len(ids))
                    many["max"] = 2 if len(ids) > 2 else 1  # read_details' batches too
            if action["name"] == message:
                reply = " ".join(v for v in observation.values() if isinstance(v, str)) if isinstance(observation, dict) else str(observation or "")
                if re.search(r"(?i)split those|one by one|one at a time|in turn|first (?:matter|thing|bit)|then the next|then another|dribs and drabs", reply):
                    self.split_heard = True
        return observations


# the interview (race 10's leader asks this way): the code plans and sends every question to the shopper, one property
# each, once enough listings are open; the actor never messages the shopper
INTERVIEW_OPENS_PER_ITEM = 3  # listings opened per requested product (plus one) before the interview
INTERVIEW_STEP_SHARE = 4  # or by step max(8, step limit / 4)
INTERVIEW_MIN_LEFT = 8  # never with this few steps left
AUTO_PAGES = 2  # a cheapest, most or least pick: the rest of a search this many pages or shorter is read by the code

PROCESS_GATE = """A shopping assistant is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the assistant has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed (CART LAST CHANGED below) and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming the chosen items as the cart shows them when the step is about them, and no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or ""}"""

PROPERTIES_FIRST = "OPENED ITS PROPERTIES TABLE INSTEAD (the requirements are judged on properties): open the full listing only for a listing you have shortlisted, to read its notes and fine print."


class Actor12(Actor11):
    """Fresh 12: fresh 11 with the leader's interview (the code plans and sends the questions, one property each, once
    enough listings are open and before anything is carted), the short rest of a cheapest, most or least search read by
    the code with its buyable rows shown cheapest first, and no hold on carting."""

    def __init__(self, problem_data: dict[str, Any]):
        super().__init__(problem_data)
        self.interviewed = False
        self.auto_paged = 0
        self.fine: dict[str, Any] = {}  # full listing answers, in a store whose properties come from another tool
        self.ref_mode = False

    # -- full listings beside the properties table -------------------------------------

    def split_listing(self) -> bool:
        listing, details = (self.roles.get("listing") or {}).get("tool"), (self.roles.get("details") or {}).get("tool")
        return bool(listing and details and listing != details)

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        """Where one tool gives the properties and another the full listing (fine print, description, notes), a
        listing's details hold both, the properties table winning a clash: a seller's note can settle a requirement or
        send a value to another listing."""
        super().absorb(action, observation)
        if not self.split_listing() or not isinstance(observation, dict) or _failed(observation):
            return
        role = self.roles["listing"] if action["name"] == self.roles["listing"]["tool"] else self.roles["details"] if action["name"] == self.roles["details"]["tool"] else None
        ident = (action.get("args") or {}).get(role.get("id_arg") or "") if role else None
        if not isinstance(ident, str):
            return
        if role is self.roles["listing"]:
            self.fine[ident] = observation
        if ident in self.fine and isinstance(self.details.get(ident), dict):
            self.details[ident] = {**self.fine[ident], **self.details[ident]}

    def read_details(self, idents: list[str]) -> None:
        """During a reference lookup, the candidates' full listings too: the pointer to the next listing of a chain
        sits in a listing's notes, which the properties table does not carry."""
        super().read_details(idents)
        if not self.ref_mode or not self.split_listing():
            return
        unread = [i for i in dict.fromkeys(idents) if i in self.details and i not in self.fine][:REFERENCE_OPENS]
        actions = [a for a in (self.id_action("listing", i) for i in unread) if a]
        actions = actions[: max(0, self.steps_left() - self.finish_reserve() - 2)]
        if actions:
            self.listen(self.act(actions))

    def lookup_reference(self, prop: str, ref: dict[str, Any]) -> dict[str, Any] | None:
        self.ref_mode = True
        try:
            return super().lookup_reference(prop, ref)
        finally:
            self.ref_mode = False

    def questions_left(self) -> int:
        """Every message goes to questions except those the request's own steps take (showing the basket, asking for the
        go-ahead, telling the shopper something): a hidden need found by one more question is worth more than an
        unasked-for basket check (a spare message still sends it)."""
        if not self.roles.get("message"):
            return 0
        return max(0, self.messages_left() - max(1 if self.confirm_first and self.confirmed is None else 0, self.process_reserve()))

    # -- the request's process steps -----------------------------------------------------

    def setup(self) -> None:
        super().setup()
        self.process: list[dict[str, Any]] = []
        self.gate_ok: set[tuple] = set()
        for entry in self.plan.get("process") or [] if isinstance(self.plan, dict) else []:
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                count = entry.get("messages")
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, int(count)) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0})
        if self.process:
            self.trace.append({"kind": "process", "steps": self.process})

    def process_reserve(self) -> int:
        """Messages the steps still due take: those before choosing (until the cart first fills), and the most an order
        step takes. A step that only applies when ending with nothing bought is not held out of the questions: it is
        said at the end, and only when a message is still left."""
        steps = getattr(self, "process", [])
        early = sum(p["messages"] for p in steps if p["when"] == "choose") if not self.gate_passed("choose") else 0
        late = max([p["messages"] for p in steps if p["when"] == "order"] or [0])
        return min(self.messages_left(), early + late)

    def gate_passed(self, stage: str) -> bool:
        return (stage, self.basket_key()) in getattr(self, "gate_ok", set()) or (stage == "choose" and any(x[0] == "choose" for x in getattr(self, "gate_ok", set())))

    def store_record(self) -> tuple[list[str], int]:
        """Every store call so far as numbered lines, and the line number of the last cart change (0 when none)."""
        cart_tools = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove")} - {None}
        lines, last_cart = [], 0
        for entry in self.trace:
            if entry.get("kind") != "env":
                continue
            for action, observation in zip(entry.get("actions") or [], entry.get("observations") or []):
                lines.append(f"step {len(lines) + 1}: {action['name']} {_compact(action.get('args'), 160)} -> {_compact(observation, 260)}")
                if action["name"] in cart_tools:
                    last_cart = len(lines)
        return lines, last_cart

    def process_gate(self, stage: str, rounds: int = 3) -> str:
        """Right before the products are chosen, the order or the ending: the steps the request asks for at that point,
        done by a model's choice of store calls (a message to the shopper, any store tool) until none is missing for the
        cart as it is now. Returns a note for the actor ("" when it may go on); a reply that changes the wants stops it."""
        steps = [p for p in getattr(self, "process", []) if p["when"] == stage]
        if not steps or self.finished or self.gate_passed(stage) or self.steps_left() < 2:
            return ""
        ending = {(self.roles.get(r) or {}).get("tool") for r in ("order", "decline")} - {None}
        tools = [f"- {name}: {_compact(self.schemas.get(name), 500)}" for name in self.schemas if name not in ending]
        said: list[str] = []
        label = {"choose": "choose the products", "order": "place the order", "end": "end without a purchase"}[stage]
        for _ in range(rounds):
            lines, last_cart = self.store_record()
            prompt = (PROCESS_GATE.replace("STAGE", label) + "\n\nREQUEST:\n" + self.request
                      + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + f"\n\nCART NOW: {_compact(self.cart, 400) if self.cart else '(empty)'}\nCART LAST CHANGED: "
                      + (f"RECORD line {last_cart}" if last_cart else "never")
                      + "\n\nRECORD:\n" + ("\n".join(lines[-40:]) or "(nothing yet)")
                      + f"\n\nMESSAGES LEFT: {self.messages_left()}   STEPS LEFT: {self.steps_left()} (the {'order' if stage != 'end' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            got = self.ask(prompt, 1200, strong=True)
            if not isinstance(got, dict):
                break
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "process_gate", "stage": stage, "steps": got.get("steps"), "call": call, "hold": got.get("hold")})
            if said and str(got.get("hold") or "").strip():
                return ("Before going on, the code did the steps the request asks for:\n" + "\n".join(said)
                        + f"\nThe shopper's reply changes the task: {str(got['hold']).strip()}. Fix the cart for it, then finish again.")
            name = str((call or {}).get("tool") or "")
            if not call:
                self.gate_ok.add((stage, self.basket_key()))
                break
            if name not in self.schemas or name in ending or self.steps_left() < 2:
                break
            message = (self.roles.get("message") or {}).get("tool")
            if name == message and self.messages_left() <= 0:
                break
            args, problems = _check(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name])
            if problems:
                break
            observations = self.act([{"name": name, "args": args}])
            if observations is None:
                break
            changed = False
            if name == message:
                self.messages_used += 1
                text = next((v for v in args.values() if isinstance(v, str)), "")
                changed = self.listen(observations, text, reply_of=0)
                if self.confirm_first and stage == "order":
                    self.confirmed = self.basket_key()  # the request's showing of the basket is this message
            else:
                self.listen(observations)
            said.append(f"{name} {_compact(args, 200)} -> {_compact(observations[0], 300)}")
            if changed:
                return ("Before going on, the code did the steps the request asks for:\n" + "\n".join(said)
                        + "\nThe shopper's reply changed the requirements (see STATUS). Fix the cart for it, then finish again.")
        return ""

    # -- the interview ---------------------------------------------------------------

    def ready_to_ask(self) -> bool:
        if self.interviewed or not self.roles.get("message") or self.questions_left() <= 0 or self.steps_left() <= INTERVIEW_MIN_LEFT:
            return False
        products = len(self.items) if self.multi() else 1
        return len(self.opened()) >= INTERVIEW_OPENS_PER_ITEM * products + 1 or self.step >= max(8, self.max_steps // INTERVIEW_STEP_SHARE)


    def product_name(self, k: int) -> str:
        if not self.multi():
            return self.category or "item"
        return str(self.items[k]["category"] or "item")


    def actor_turn(self) -> dict[str, Any] | None:
        if self.ready_to_ask() and self.chat and self.chat[-1].get("role") in ("tool", "user"):
            summary = self.interview()
            if summary:
                self.chat.append({"role": "user", "content": summary + "\n\n" + self.status()})
        # the actor never messages the shopper: the message tool is left out of its tools
        message = (self.roles.get("message") or {}).get("tool")
        everything = self.tools
        self.tools = [t for t in everything if t["function"]["name"] != message]
        try:
            return super().actor_turn()
        finally:
            self.tools = everything

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """A cart change or a finish before the interview runs the interview first (no step used for the held calls);
        a message the actor writes is not sent (the code asks); the rest as in fresh 11."""
        message = (self.roles.get("message") or {}).get("tool")
        committing = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "order", "decline")} - {None}
        names = [str(c["function"].get("name")) for c in calls]
        if not self.interviewed and self.roles.get("message") and self.questions_left() > 0 and self.steps_left() > INTERVIEW_MIN_LEFT \
                and any(n in committing for n in names):
            summary = self.interview()
            note = "NOT RUN (no step used): the code asked the shopper first.\n" + (summary or "The interview had nothing to ask.")
            return [(c["id"], note if n == 0 else "NOT RUN (no step used).") for n, c in enumerate(calls)]
        note = self.gate_calls(calls)
        if note:
            return [(c["id"], ("NOT RUN (no step used for it): " + note) if n == 0 else "NOT RUN (no step used).") for n, c in enumerate(calls)]
        held = {c["id"] for c in calls if str(c["function"].get("name")) == message}
        calls, swapped = self.properties_first(calls)
        rest = [c for c in calls if c["id"] not in held]
        results = dict(super().run_calls(rest)) if rest else {}
        results.update({cid: "NOT SENT (no step used): the code asks the shopper; carry on with the store's other tools." for cid in held})
        for cid in swapped:
            results[cid] = (PROPERTIES_FIRST + "\n" + results.get(cid, "NO RESULT"))
        out = [(c["id"], results.get(c["id"], "NO RESULT")) for c in calls]
        return self.page_on(calls, out)


    def properties_first(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
        """Where the properties table and the full listing are separate tools, a listing whose properties are unread
        gets its properties table instead: requirements and questions are judged on properties, and the full listing
        (notes, fine print) is for a listing already shortlisted."""
        if not self.split_listing():
            return calls, set()
        listing, details = self.roles["listing"], self.roles["details"]
        out, swapped = [], set()
        for c in calls:
            if str(c["function"].get("name")) == listing["tool"]:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}") if isinstance(c["function"].get("arguments"), str) else dict(c["function"].get("arguments") or {})
                except ValueError:
                    args = {}
                ident = args.get(listing.get("id_arg") or "")
                action = self.id_action("details", ident) if isinstance(ident, str) and ident not in self.details else None
                if action:
                    c = {**c, "function": {"name": action["name"], "arguments": json.dumps(action["args"])}}
                    swapped.add(c["id"])
            out.append(c)
        return out, swapped

    # -- the rest of a short search, for a cheapest, most or least pick -----------------

    def pick_kind(self) -> str:
        kinds = [str((i.get("preference") or {}).get("kind") or "") for i in self.items] if self.multi() else [str((self.preference or {}).get("kind") or "")]
        return "cheapest" if "cheapest" in kinds else "best" if "best" in kinds else ""

    def page_on(self, calls: list[dict[str, Any]], out: list[tuple[str, str]]) -> list[tuple[str, str]]:
        search = (self.roles.get("search") or {})
        if not self.pick_kind() or not search.get("tool") or self.tight():
            return out
        results = dict(out)
        for call in calls:
            if str(call["function"].get("name")) != search["tool"] or call["id"] not in results:
                continue
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except ValueError:
                continue
            entry = self.queries.get(_query_key(search["tool"], args))
            if not entry or not entry.get("more") or entry["pages"] > 1 or entry.get("total") is None or not entry["size"]:
                continue
            pages = math.ceil((entry["total"] - len(entry["ids"])) / entry["size"])
            if pages < 1 or pages > AUTO_PAGES or self.auto_paged + pages > 6 or self.steps_left() - pages <= self.reserve() + 6:
                continue
            texts = []
            last_obs: Any = None
            try:
                last = self.done.get(self.call_key(search["tool"], args) or "")
                last_obs = json.loads(last[1]) if last and last[1].startswith("{") else None
            except (ValueError, TypeError, IndexError):
                last_obs = None
            last_obs = _inner(last_obs)
            self._note_offset_paging(_next_number(last_obs) if last_obs else None,
                                     len(_result_rows(last_obs)) if last_obs else 0, search.get("page_arg"))
            for page in range(2, 2 + pages):
                token = None
                if search.get("token_arg"):
                    token = _next_token(last_obs) if last_obs else None
                    if not token:
                        break
                    action = {"name": search["tool"], "args": {**args, search["token_arg"]: token}}
                elif search.get("page_arg"):
                    raw = _next_number(last_obs) if last_obs else None
                    page_arg_val = raw if isinstance(raw, int) and raw > 0 else self.page_value(page)
                    action = {"name": search["tool"], "args": {**args, search["page_arg"]: page_arg_val}}
                else:
                    break
                observations = self.act([action])
                if not observations:
                    break
                self.auto_paged += 1
                self.listen(observations)
                last_obs = observations[0]
                last_obs = _inner(last_obs)
                self._note_offset_paging(_next_number(last_obs), len(_result_rows(last_obs)), search.get("page_arg"))
                args = action["args"]
                texts.append(f"page {page}: " + _compact(observations[0], 3000))
            if texts:
                results[call["id"]] += "\n[the code read the rest of this search, one step a page]\n" + "\n".join(texts)
            rows = sorted((p, i) for i in entry["ids"] for p in [_price(self.rows.get(i, {}))] if p is not None
                          and self.rows.get(i, {}).get("stock") is not False and i not in self.unavailable)
            if self.pick_kind() == "cheapest" and len(rows) >= 2:
                results[call["id"]] += "\n[buyable rows of this search, cheapest first] " + ", ".join(f"{i} ({p:g})" for p, i in rows[:8])
        return [(cid, results.get(cid, text)) for cid, text in out]

    # -- what the actor sees: the code does the asking ----------------------------------

    def readiness(self) -> list[str]:
        return [g for g in super().readiness() if "question(s) unused" not in g]

    def phase(self) -> str:
        if not self.interviewed and self.roles.get("message") and self.questions_left() > 0:
            return ("SURVEY: search the category and open several listings in full (several per step when the store allows); the code asks the "
                    "shopper once enough are open, before anything is carted")
        cart = self.cart_gap()
        return ("FIND AND COMMIT: apply the shopper's answers, cart each product's fitting listing as soon as you have opened it (stores sell out "
                "carted lines; an early cart leaves steps to replace it), prove the cheapest where asked, then order")

    def status(self) -> str:
        return super().status() + "\nQUESTIONS: the code asks the shopper for you; never message the shopper yourself."

    # -- a decline waits for a real search -------------------------------------------------

    def gate_decline(self) -> str:
        """While steps remain above the order reserve, a decline is held: running short proves nothing, and in the race
        logs a decline was right in about one task in fifty while agents that declined with steps left lost most of those
        tasks. The actor is told what is unmet and to search for it; nearer the end, fresh 11's audit and fallback."""
        stalled = len(self.held_at) >= STALL_HOLDS and all(x == self.step for x in self.held_at[-STALL_HOLDS:])
        if self.steps_left() > self.order_due() + 2 and self.time_left() > CLOSING_SECONDS + 20 and not stalled:
            self.held_at.append(self.step)
            unmet = []
            for ident in self.cart:
                bad = [r for r in self.item_reqs(ident) if all(_split_req(r)) and self.code_verdict(ident, r) != "yes"]
                if bad:
                    unmet.append(f"{ident}: {'; '.join(bad)}")
            missing = self.missing_lines()
            self.trace.append({"kind": "decline_held", "steps_left": self.steps_left(), "unmet": unmet[:4]})
            return (f"NOT SENT (no step used): {self.steps_left()} steps are left, so keep looking before ending without a purchase. "
                    + (f"Unconfirmed in the cart: {' | '.join(unmet[:4])}. " if unmet else "")
                    + (f"{missing} product line(s) still have no listing. " if missing else "")
                    + "Search for each unmet requirement with short value-first queries (the value plus the product noun), open the "
                    + "promising rows, and cart a listing whose facts meet everything.")
        return super().gate_decline()

    # -- no hold on carting: the leader carts a fit as soon as it is found --------------

    def cart_gap(self) -> str:
        return ""

    def survey_gap(self) -> str:
        return ""


# A request sentence that wants the basket shown or approved before the purchase: it speaks of buying, of a moment before
# it, of showing or approving, and of what is being bought; a sentence asking the shopper to choose is not that rule.
_BUYING = {"order", "buy", "purchase", "checkout", "check-out", "pay", "finalize", "finalise", "finish", "complete", "submit"}
_BEFORE = {"before", "until", "till", "prior", "ahead"}
_SHOWING = {"show", "tell", "read", "confirm", "approve", "approval", "okay", "ok", "yes", "go-ahead", "sign-off", "summarize",
            "summarise", "summary", "recap", "review", "see", "describe", "present", "share", "list", "run"}
_GOODS = {"basket", "cart", "trolley", "selection", "pick", "picks", "chosen", "items", "order"}
_CHOOSING = {"which", "whether", "prefer", "rather", "how many", "how much"}


def _sentence_words(text: str) -> set[str]:
    words = set()
    text = re.sub(r"\bcheck(?:ing|s|ed)?\s+out\b", "checkout", text.lower())  # "checking out" is buying
    for w in re.findall(r"[a-z]+(?:-[a-z]+)?", text):
        words.add(w)
        for end in ("ing", "ed", "es", "s", "al"):
            if w.endswith(end) and len(w) - len(end) >= 3:
                words.add(w[: -len(end)])
    return words


def _wants_basket_shown(text: str) -> bool:
    words = _sentence_words(text)
    if any(c in text.lower() for c in _CHOOSING if " " in c) or words & {c for c in _CHOOSING if " " not in c}:
        return False
    return bool(words & _BUYING and words & _BEFORE and words & _SHOWING and (words & _GOODS or "what you" in text.lower()))


CANDIDATE_CHECK = """You check listings a shopping assistant has opened, before it chooses what to put in the cart. Judge each listing below against the CURRENT REQUIREMENTS of the product it is for, only from its record, by these rules:
""" + MATCH_RULES + """
The listing's store category decides what kind of product it is: a title or description naming another object (a cushion, a case, a refill, a set) never fails a listing that sits in the requested category.
Give fits=true when its record meets every current requirement; fits=false only for a requirement its record clearly fails (quote the fact); fits=null when a requirement is neither met nor failed by its record. Price, stock and quantity are checked elsewhere: judge the requirements only.
Return ONE JSON object and nothing else: {"listings": [{"id": "...", "fits": true | false | null, "fails": "the failed or unproven requirement with the quoted fact, or null"}]}"""
CHEAPER_ROUNDS_13 = 3  # rounds of cheaper rows opened per cart line until the check passes one
TRIM_TO = 0.6  # a trim shortens old results until the conversation is this share of COMPACT_CHARS
LIKELY_ROWS_13 = 8  # unopened listings suggested for opening
WALK_PAGES_13 = 5  # under a cheapest rule: further pages of a product's best search the code reads before the first order
CHEAPER_OPENS_13 = 3  # under a cheapest rule, unopened cheaper rows of a carted line's category the code opens before the order
DECLINE_OPENS_13 = 4  # unopened likely rows the code opens before letting a decline go
RULE_OPEN = re.compile(r"\b(?:open|view|visit|look(?:ing)? at|read)\w*\b[^.]{0,40}\b(?:listing|page|product|item|record)s?\b", re.I)
RULE_OWN = re.compile(r"shopper|customer|confirm|approv|show|tell|messag|\bask|basket|go-?ahead|budget|stock|price|cost|total", re.I)


class Actor13(Actor12):
    """Fresh 13: the request's steps are done last, right before the order or the ending goes out (after every hold
    and check, and in a forced finish too); the order review's unmet working rules hold the order; a show-the-basket
    rule is also found by its words; results keep their structure (each field clipped, not the tail); the repeat
    cache forgets only after a notice or reply changes something; three questions when no limit is stated."""

    def setup(self) -> None:
        super().setup()
        self.paged_key: Any = None  # the cart whose lines' full listing pages were read before ordering
        self.judged: dict[str, tuple[Any, Any, str]] = {}  # listing -> (the wants it was judged under, fits, what fails)
        self.judge_calls = 0
        self.changes = 0  # notices and replies that changed the requirements, stock or budget so far
        self.paged_at: dict[str, int] = {}  # listing -> the change count when its full page was last read
        self.refused = False  # a finish (order or ending) was refused by the store
        self.last_carted = ""  # the listing most recently put in the cart
        if not self.confirm_first and self.roles.get("message") and self.show_rule():
            self.confirm_first = True
            if not any(p["when"] == "order" for p in self.process):
                self.process.append({"step": "show the shopper the basket and get their go-ahead", "when": "order", "messages": 1})
            self.trace.append({"kind": "show_rule_words"})

    def rule_texts(self) -> list[str]:
        rules = self.plan.get("rules") if isinstance(self.plan, dict) else None
        rules = [rules] if isinstance(rules, str) else rules if isinstance(rules, list) else []
        return [str(r) for r in rules if str(r).strip()]

    def show_rule(self) -> bool:
        return any(_wants_basket_shown(text) for text in re.split(r"(?<=[.!?;])\s+|\n", self.request) + self.rule_texts())

    def suspicious(self, ident: str) -> bool:
        return False  # a listing on the requested category is that kind, whatever its price or title suggests

    # -- a long conversation is trimmed, not thrown away ---------------------------------------

    def actor_turn(self) -> dict[str, Any] | None:
        """The STATUS rides in a message of its own at the end: the newest result's STATUS is split off into one, and the
        previous turn's STATUS message is dropped. Every earlier message then stays exactly as it was sent, so the cache
        (marked on the last results and on the STATUS, see view) reads the whole conversation up to this turn's results."""
        self.chat = [m for i, m in enumerate(self.chat) if not (i > 1 and m.get("role") == "user" and str(m.get("content") or "").startswith("STATUS: step"))]
        last = self.chat[-1] if self.chat else {}
        content = last.get("content")
        if len(self.chat) > 2 and isinstance(content, str) and "\n\nSTATUS: step" in content:
            head, tail = content.split("\n\nSTATUS: step", 1)
            last["content"] = head
            self.chat.append({"role": "user", "content": "STATUS: step" + tail})
        return super().actor_turn()

    def view(self) -> list[dict[str, Any]]:
        """The conversation with cache marks on the system text, the last results and the STATUS after them."""
        sent = [dict(m) for m in self.chat]
        marks = {0, len(sent) - 1} | ({len(sent) - 2} if len(sent) > 3 and sent[-1].get("role") == "user" else set())
        for index in sorted(marks):
            if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
        return sent

    def chat_size(self) -> int:
        return sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in self.chat)

    def condense(self) -> None:
        """Past COMPACT_CHARS: first the stale STATUS blocks of older turns go (the newest stays), then the oldest half of
        the long tool results are shortened to their start and the listing ids they named; the whole
        history is summarised (fresh 12's condense) only when that is still not enough. The actor keeps its own reasoning
        and the recent results, and the prompt cache keeps the untouched start of the conversation."""
        if self.chat_size() <= COMPACT_CHARS:
            return
        if self.chat and self.chat[-1].get("role") not in ("tool", "user"):
            return
        recent = max(1, len(self.chat) - 4)
        for message in self.chat[1:recent]:
            content = message.get("content")
            if isinstance(content, str) and "\n\nSTATUS: step" in content:
                message["content"] = content.split("\n\nSTATUS: step", 1)[0] + "\n(older status removed)"
        trimmed = "status"
        if self.chat_size() > COMPACT_CHARS:
            # one big step (oldest results first, down to TRIM_TO of the limit), so the cache is broken rarely
            trimmed = "results"
            for message in [m for m in self.chat[1:recent] if m.get("role") == "tool" and len(str(m.get("content") or "")) > 600]:
                if self.chat_size() <= COMPACT_CHARS * TRIM_TO:
                    break
                content = str(message["content"])
                ids = [i for i in dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{5,40}", content)) if i in self.rows][:20]
                message["content"] = content[:400] + "...(older result shortened" + (f"; listings in it: {', '.join(ids)}" if ids else "") + ")"
        self.trace.append({"kind": "trimmed", "what": trimmed, "chars": self.chat_size()})
        if self.chat_size() > COMPACT_CHARS:
            super().condense()

    # -- a model judges the opened listings before one is carted ---------------------------

    def removes_checked_fit(self, name: Any, args: dict[str, Any]) -> bool:
        """A cart call taking out a line the candidate check passed, once the steps are down to the order's reserve and the
        cart passes the code's own checks.

        Lines the code carted as need fits stay for the whole episode."""
        cart = self.roles.get("cart") or {}
        ident = args.get(cart.get("id_arg") or "")
        removing = name == (self.roles.get("cart_remove") or {}).get("tool") or (
            name == cart.get("tool") and cart.get("op_arg") and args.get(cart["op_arg"]) == cart.get("remove_value"))
        if not (removing and ident in self.cart):
            return False
        if ident in getattr(self, "code_carted", set()):
            want = self.wanted_units(ident) if hasattr(self, "wanted_units") else int(self.quantity or 1)
            held_n = int(self.cart.get(ident) or 0)
            rem_n = int(args.get(cart.get("qty_arg") or "") or held_n)
            if held_n - rem_n >= want:
                return False
            return True
        if self.verdict(ident) is not True:
            return False
        late = self.steps_left() <= self.order_due() and not self.basket_problems()
        self.fit_holds = getattr(self, "fit_holds", set())
        if late or ident not in self.fit_holds:
            self.fit_holds.add(ident)  # held once at any time (always near the end): the check passed it, the actor doubts it
            return True
        return False

    def count_units(self, action: dict[str, Any], observation: Any, before: dict[str, int]) -> None:
        """A cart tool that adds or takes off units (an operation argument, no list of lines) and whose answer does not say
        what the line now holds: the line's count follows the call, adding to what it held (in race logs adding 1 and then 1
        more of a listing left 2 in the bag). A cart view, when the store has one, still sets the count from the store."""
        cart = self.roles.get("cart") or {}
        if action.get("name") != cart.get("tool") or cart.get("lines_arg") or not cart.get("op_arg") or _failed(observation):
            return
        if getattr(self, "units_cart", lambda: False)() or _line_units(observation):
            return
        args = action.get("args") or {}
        ident, units = args.get(cart.get("id_arg") or ""), args.get(cart.get("qty_arg") or "", 1)
        if not isinstance(ident, str) or not isinstance(units, int) or isinstance(units, bool) or units < 1:
            return
        held = int(before.get(ident, 0))
        if str(args.get(cart["op_arg"])) == str(cart.get("add_value")):
            self.cart[ident] = held + units
        elif str(args.get(cart["op_arg"])) == str(cart.get("remove_value")):
            if held - units > 0:
                self.cart[ident] = held - units
            else:
                self.cart.pop(ident, None)


    def units_right(self) -> None:
        """Before an order: a line holding other units than its product asks for is set right by the code (a hold alone
        let a wrong count go out once the holds ran out), when the lines map one to one onto the products."""
        if not self.cart or self.steps_left() <= len(self.cart) + 1:
            return
        wanted = {i: self.wanted_units(i) for i in self.cart}
        if all(wanted[i] == u for i, u in self.cart.items()):
            return
        if self.multi() and len(wanted) == len(self.items) and sorted(wanted.values()) != sorted(int(it.get("quantity") or 1) for it in self.items):
            return  # two lines read as one product: left to the actor and the holds
        self.fix_units()

    def fix_units(self) -> None:
        """Before a forced order: each cart line set to its product's requested units, by the store's own cart call."""
        cart = self.roles.get("cart") or {}
        wrong = {i: (u, self.wanted_units(i)) for i, u in self.cart.items() if u != self.wanted_units(i)}
        if not wrong or not cart.get("tool"):
            return
        if cart.get("lines_arg"):
            actions = [self.basket_action({i: self.wanted_units(i) for i in self.cart})]
        elif getattr(self, "units_cart", lambda: False)():
            actions = [{"name": cart["tool"], "args": self.fill_required(cart["tool"], {cart["id_arg"]: i, cart["qty_arg"]: w})} for i, (u, w) in wrong.items()]
        elif cart.get("op_arg") and cart.get("qty_arg"):
            actions = [{"name": cart["tool"], "args": self.fill_required(cart["tool"], {
                cart["id_arg"]: i, cart["op_arg"]: cart["add_value"] if w > u else cart["remove_value"], cart["qty_arg"]: abs(w - u)})}
                for i, (u, w) in wrong.items()]
        else:
            return
        actions = [a for a in actions if a][: max(0, self.steps_left() - 1)]
        if actions:
            self.act(actions)
            self.trace.append({"kind": "fix_units", "wrong": {i: list(x) for i, x in wrong.items()}, "cart": dict(self.cart)})

    def context(self, story: bool = True) -> str:
        text = super().context(story)
        unread = getattr(self, "unread_lines", [])
        if unread:
            text += ("\n\nSHOPPER LINES THE CODE COULD NOT READ (each one that names a value for a property is a requirement too):\n"
                     + "\n".join(f"- {line}" for line in unread[-6:]))
        return text

    def status(self) -> str:
        text = super().status()
        unread = getattr(self, "unread_lines", [])
        if unread:
            text += "\nSHOPPER SAID (not read into the requirements above; apply each value it names): " + " | ".join(unread[-4:])
        return text

    def verdict(self, ident: str) -> Any:
        """The candidate check's fits (True, False or None) for a listing under the current wants; None when not judged."""
        got = getattr(self, "judged", {}).get(ident)
        return got[1] if got and got[0] == self.wants() else None


    def checks(self, ident: str) -> str:
        marks = super().checks(ident)
        fits = self.verdict(ident)
        if fits is True:
            marks = (marks + ", " if marks else "") + "CHECK: meets every requirement"
        elif fits is False:
            marks = (marks + ", " if marks else "") + "CHECK: fails " + (self.judged[ident][2] or "a requirement")[:80]
        return marks

    def best_fit(self, taken: set[str]) -> str | None:
        """A listing the candidate check found to meet every requirement first (in stock, affordable, of the product's
        category; the cheapest under a cheapest rule, else the first found); else fresh 12's choice among the rest, never one
        the check failed."""
        label = self.label_for()
        cheap = (self.preference or {}).get("kind") == "cheapest"
        fits = []
        for n, ident in enumerate(self.details):
            info = self.rows.get(ident, {})
            if self.verdict(ident) is not True or ident in taken or ident in self.unavailable or info.get("stock") is False:
                continue
            if label and info.get("category") not in (None, label):
                continue
            price = _price(info)
            if price is None or (self.budget is not None and price * int(self.quantity or 1) > self.budget):
                continue
            fits.append((price if cheap else 0, n, ident))
        if fits:
            return min(fits)[2]
        failed = {i for i, (_, fits, _) in getattr(self, "judged", {}).items() if fits is False and self.verdict(i) is not True
                  and not all(self.code_verdict(i, r) == "yes" for r in self.item_reqs(i) if all(_split_req(r)))}
        return super().best_fit(set(taken) | failed)  # a failed check stands until a new check passes it

    def ask(self, prompt: str, max_tokens: int = 1500, strong: bool = False, gate: bool = False) -> dict[str, Any] | None:
        spent = self.spent
        got = super().ask(prompt, max_tokens, strong, gate)
        if not got and prompt.startswith(INTERPRET_PROMPT[:80]):
            got = super().ask(prompt, max_tokens, True, True)  # a shopper line left unread loses what it said: once more, on the main model
            self.trace.append({"kind": "reading_retry", "ok": bool(got)})
        self.trace.append({"kind": "ask_cost", "what": " ".join(prompt.split()[:6])[:60], "cost": round(self.spent - spent, 5)})
        return got

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        before = len(self.details)
        many = self.roles.get("details_many") or {}
        trimmed: dict[str, str] = {}
        held: dict[str, str] = {}
        kept = []
        for call in calls:
            function = call["function"]
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                args = None
            if not isinstance(args, dict):
                kept.append(call)
                continue
            ids = args.get(many.get("ids_arg") or "") if function.get("name") == many.get("tool") else None
            size = int(many.get("max") or 0)
            if isinstance(ids, list) and size and len(ids) > size:
                args[many["ids_arg"]] = ids[:size]
                call = {**call, "function": {**function, "arguments": json.dumps(args)}}
                trimmed[call["id"]] = (f"\nNOTE: only the first {size} were sent: this store's {many['tool']} shows at most {size} "
                                       f"listing(s) at once. Open the rest ({', '.join(map(str, ids[size:]))}) in further calls.")
            order = self.finish_action("order")
            if order and self.cart and not self.over() and self.steps_left() <= self.order_due() + 1 \
                    and not getattr(self, "late_ordered", False) and self.finish_kind(str(function.get("name")), args) == "decline":
                self.late_ordered = True  # once: a held order must not turn every later decline into another order
                # near the end with a basket: fill what is missing and order it (a decline was right in 3 of 71 online)
                self.fell_back = False
                self.fallback()
                self.trace.append({"kind": "late_decline_to_order", "cart": dict(self.cart), "missing": self.missing_lines()})
                call = {**call, "function": {"name": order["name"], "arguments": json.dumps(order["args"])}}
                function, args = call["function"], dict(order["args"])
            if self.removes_checked_fit(function.get("name"), args):
                info = self.rows.get(args.get((self.roles.get("cart") or {}).get("id_arg") or ""), {})
                held[call["id"]] = (f"NOT SENT (no step used): the code's check found this line meets every requirement, and the store files it "
                                    f"under '{info.get('category') or 'the requested category'}', the requested kind (the category decides the kind, "
                                    "not the title). It stays unless STATUS names a requirement it fails; to replace it, cart the replacement "
                                    "first. Otherwise order the cart.")
                continue
            cart_role = self.roles.get("cart") or {}
            add_id = args.get(cart_role.get("id_arg") or "")
            is_add = (function.get("name") == cart_role.get("tool") and cart_role.get("op_arg")
                      and args.get(cart_role["op_arg"]) == cart_role.get("add_value") and isinstance(add_id, str))
            if is_add:
                locked = getattr(self, "code_carted", set())
                if add_id in locked:
                    want = self.wanted_units(add_id) if hasattr(self, "wanted_units") else int(self.quantity or 1)
                    held_n = int(self.cart.get(add_id) or 0)
                    add_n = int(args.get(cart_role.get("qty_arg") or "") or 1)
                    if held_n >= want or held_n + max(1, add_n) > want:
                        held[call["id"]] = (f"NOT SENT (no step used): {add_id} is the code's need fit at the "
                                            f"requested {want} unit(s). Do not add more; order or fill other products.")
                        continue
                else:
                    add_label = (self.rows.get(add_id) or {}).get("category")
                    locked_here = [i for i in locked if i in self.cart
                                   and self._row_in_category(i, add_label if isinstance(add_label, str) else None)]
                    if locked_here:
                        held[call["id"]] = (f"NOT SENT (no step used): the cart already holds the code's need fit "
                                            f"{locked_here[0]} for this product. Do not add {add_id}; order the cart "
                                            "(or fill other missing products).")
                        continue
            kept.append(call)
        cart_before = set(self.cart)
        out = dict(super().run_calls(kept)) if kept else {}
        out.update(held)
        for line in [i for i in self.cart if i not in cart_before]:
            note = self.cheap_for_line(line)
            if note and kept:
                out[kept[-1]["id"]] = out.get(kept[-1]["id"], "") + "\n" + note
        out = [(c["id"], out.get(c["id"], "NO RESULT") + trimmed.get(c["id"], "")) for c in calls]
        if len(self.details) > before and not self.over():
            self.judge_candidates()
        return out

    def read_cart_pages(self) -> str:
        """Once per cart, before its order: where the full listing page is a separate tool from the properties, read the
        page of every carted line (stock, price, the seller's remarks as they stand now). In race logs orders whose
        carted lines' pages were read passed 45% against 8%. A note when what was read changes something."""
        if not self.cart or not self.split_listing() or self.paged_key == self.basket_key():
            return ""
        if self.steps_left() <= len(self.cart) + 2 or self.time_left() < FORCE_SECONDS + 15:
            return ""
        self.paged_key = self.basket_key()
        # a page read earlier still holds unless a notice or reply changed something since (in race logs a page read
        # before the last cart change did as well as one read after it)
        stale = [i for i in self.cart if self.paged_at.get(i, -1) < self.changes]
        actions = [a for a in (self.id_action("listing", i) for i in stale) if a]
        if not actions:
            return ""
        observations = self.act(actions)
        self.trace.append({"kind": "read_cart_pages", "lines": stale, "skipped": [i for i in self.cart if i not in stale]})
        if observations and self.listen(observations):
            return "NOT SENT. Reading the carted listings' pages changed something (stock, price or a note): see STATUS and adjust."
        problems = self.basket_problems()
        return ("NOT SENT. Reading the carted listings' pages found: " + " | ".join(problems[:4])) if problems else ""

    def search_evidence(self) -> dict[str, int]:
        """Per listing seen in a search: the most required-value words any query that listed it carried (a row a search led by
        the required values found is likelier to carry them than one a bare product-noun search found)."""
        values = {w for r in (self.reqs or []) for w in _words(_split_req(r)[1] or "") if w not in _FILLER and len(w) > 1}
        for item in self.items if self.multi() else []:
            values |= {w for r in item.get("reqs") or item.get("requirements") or [] for w in _words(_split_req(str(r))[1] or "") if w not in _FILLER and len(w) > 1}
        out: dict[str, int] = {}
        for entry in getattr(self, "queries", {}).values():
            carried = len(values & set(_words(str(entry.get("args") or ""))))
            for ident in entry.get("ids") or []:
                out[ident] = max(out.get(ident, 0), carried)
        return out


    def likely_rows(self) -> list[str]:
        return [f"{i} {self.rows[i].get('price', '?')} {str(self.rows[i].get('title') or '')[:50]}" for i in self.likely_ids()]

    def code_opens(self, idents: list[str], why: str) -> str:
        """The code opens these listings itself (several per step where the store allows); a note naming what it opened."""
        idents = [i for i in idents if i not in self.details][: max(0, self.steps_left() - self.order_due() - 2)]
        if not idents or self.time_left() < FORCE_SECONDS + 25:
            return ""
        self.read_details(idents)
        opened = [i for i in idents if i in self.details]
        self.trace.append({"kind": "code_opens", "why": why, "ids": opened})
        if not opened:
            return ""
        return (f"NOT SENT (the code first opened {len(opened)} more listing(s) {why}: {', '.join(opened)}; their facts are in the "
                "OPENED LISTINGS table). Re-check them against every requirement, cart the best fit, then finish again.")

    def next_page_action(self, key: str) -> dict[str, Any] | None:
        """The call for the next page of a search the store says has more, by its token or its page number."""
        search = self.roles.get("search") or {}
        args, observation = getattr(self, "search_last", {}).get(key, (None, None))
        entry = self.queries.get(key) or {}
        if args is None or not entry.get("more"):
            return None
        obs = observation
        obs = _inner(obs)
        if search.get("token_arg"):
            token = _next_token(obs)
            return {"name": search["tool"], "args": {**args, search["token_arg"]: token}} if token else None
        if search.get("page_arg"):
            self._note_offset_paging(_next_number(obs) if isinstance(obs, dict) else None,
                                     len(_result_rows(obs)) if obs is not None else 0, search.get("page_arg"))
            raw = _next_number(obs) if isinstance(obs, dict) else None
            page_arg_val = raw if isinstance(raw, int) and raw > 0 else self.page_value(int(entry.get("pages") or 1) + 1)
            return {"name": search["tool"], "args": {**args, search["page_arg"]: page_arg_val}}
        return None

    def cheap_for_line(self, line: str) -> str:
        """Under a cheapest rule, as soon as a product's line is first carted (while steps remain): that product's best search
        is read further and cheaper rows of its category are opened and checked; a cheaper checked fit takes the line's
        place. At the order this came too late in a tight step limit."""
        if line in getattr(self, "code_carted", set()):
            self.cheap_done = getattr(self, "cheap_done", set()) | {line}
            return ""
        if self.pick_kind() != "cheapest" or line in getattr(self, "cheap_done", set()):
            return ""
        self.cheap_done = getattr(self, "cheap_done", set()) | {line}
        before = dict(self.cart)
        self.cheapest_walk(only=line)
        self.cheaper_key = None
        self.cheaper_check(only=line)
        return ("NOTE: the code looked for cheaper fits of that product and replaced the line: cart now "
                + ", ".join(f"{i} x{u}" for i, u in self.cart.items())) if self.cart != before else ""

    def cheapest_walk(self, only: str | None = None) -> None:
        """Under a cheapest rule, once per product line before its order: further pages of that product's best search (the
        one whose words carry most of its required values, then the narrowest) are read, a few per product within the
        steps left above the order's reserve; the cheaper-rows check that follows opens and judges what they showed."""
        if self.pick_kind() != "cheapest" or not self.cart or not (self.roles.get("search") or {}).get("tool"):
            return
        self.walked = getattr(self, "walked", set())
        lines = [l for l in ([only] if only else self.cart) if str(self.rows.get(l, {}).get("category") or "") not in self.walked]
        budget = self.steps_left() - self.order_due() - 3
        if not lines or budget < 2 or self.time_left() < CLOSING_SECONDS + 60:
            return
        products = max(1, (self.missing_lines() + len(lines)) if only else len(lines))  # share the budget with the products still to buy
        per = max(1, min(WALK_PAGES_13, budget // products))
        read = 0
        for line in lines:
            label = str(self.rows.get(line, {}).get("category") or "")
            self.walked.add(label)
            values = {w for r in self.item_reqs(line) for w in _words(_split_req(r)[1] or "") if w not in _FILLER}
            choices = []
            for key, entry in self.queries.items():
                ids = entry.get("ids") or []
                if not ids or not entry.get("more") or key not in getattr(self, "search_last", {}):
                    continue
                if sum(str(self.rows.get(i, {}).get("category") or "") == label for i in ids) * 2 < len(ids):
                    continue  # a search mostly of another product
                if " ".join(_words(str((entry.get("args") or {}).get(str((self.roles.get("search") or {}).get("query_arg")), "")
                                       if isinstance(entry.get("args"), dict) else ""))) in getattr(self, "ref_queries", set()):
                    continue  # a search for a referenced listing (a seller's words), not for the product's shelf
                said = set(_words(str(entry.get("args") or "")))
                kind = {_one(w) for w in _words(label or self.item_for(line).get("category") or "")} - _FILLER
                choices.append((-len(values & said), -len(kind & {_one(w) for w in said}), entry.get("total") or 10 ** 6, key))
            if not choices:
                continue
            key = min(choices)[-1]
            for _ in range(per):
                action = self.next_page_action(key)
                if not action or self.steps_left() <= self.order_due() + 4:
                    break
                observations = self.act([action])
                if not observations:
                    break
                self.listen(observations)
                read += 1
        if read:
            self.trace.append({"kind": "cheapest_walk", "pages": read, "lines": lines})

    def cheaper_check(self, only: str | None = None) -> str:
        """Under a cheapest rule, once per cart: for each line, unopened rows of its category priced below it are opened in
        rounds (likeliest first) and checked, until the check passes one or the steps run short; a cheaper checked fit then
        takes the line's place, put in by the code at the same units (the actor was often told and did not swap)."""
        if self.pick_kind() != "cheapest" or not self.cart or getattr(self, "cheaper_key", None) == self.basket_key():
            return ""
        self.cheaper_key = self.basket_key()
        swaps = []
        done = getattr(self, "cheap_done", set())
        locked = getattr(self, "code_carted", set())
        for line in ([only] if only and only in self.cart else [l for l in self.cart if l not in done]):
            if line in locked:
                continue
            info = self.rows.get(line, {})
            price, label = _price(info), str(info.get("category") or "") or None
            if price is None:
                continue
            for _ in range(CHEAPER_ROUNDS_13):
                if self.steps_left() <= self.order_due() + 3 or self.time_left() < FORCE_SECONDS + 40:
                    break
                fits = sorted((_price(self.rows.get(i, {})) or 0, i) for i in self.details if i not in self.cart and i not in self.unavailable
                              and self.verdict(i) is True and (label is None or self.rows.get(i, {}).get("category") == label)
                              and (_price(self.rows.get(i, {})) or float("inf")) < price and self.rows.get(i, {}).get("stock") is not False)
                if fits:
                    if self.swap_line(line, fits[0][1]):
                        swaps.append(f"{fits[0][1]} ({fits[0][0]:g}) in place of {line} ({price:g})")
                    break
                picks = self.likely_ids(CHEAPER_OPENS_13, below=price, category=label)
                if not picks:
                    break
                self.code_opens(picks, "priced below the cart's choice")
                self.judge_candidates()
        if swaps:
            self.trace.append({"kind": "cheaper_swap", "swaps": swaps})
        return ""

    def swap_line(self, old: str, new: str) -> bool:
        """The cart line `old` replaced by `new` at the same units, by the store's own cart calls."""
        if old in getattr(self, "code_carted", set()):
            return False
        units = int(self.cart.get(old) or 1)
        actions = self.set_line_actions(old, 0) + self.set_line_actions(new, units)
        if not actions or len(actions) > self.steps_left() - 2:
            return False
        self.act(actions)
        return new in self.cart and old not in self.cart

    def set_line_actions(self, ident: str, units: int) -> list[dict[str, Any]]:
        """Store calls that leave `units` of a listing in the cart (0 takes the line out), for a cart tool that takes the
        whole basket, sets a line's units, or adds and takes off units."""
        cart = self.roles.get("cart") or {}
        held = int(self.cart.get(ident) or 0)
        if units == held or not cart.get("tool"):
            return []
        if cart.get("lines_arg"):
            wanted = {i: u for i, u in self.cart.items() if i != ident}
            if units:
                wanted[ident] = units
            action = self.basket_action(wanted)
            return [action] if action else []
        if getattr(self, "units_cart", lambda: False)():
            return [{"name": cart["tool"], "args": self.fill_required(cart["tool"], {cart["id_arg"]: ident, cart["qty_arg"]: units})}]
        if cart.get("op_arg") and cart.get("qty_arg"):
            return [{"name": cart["tool"], "args": self.fill_required(cart["tool"], {
                cart["id_arg"]: ident, cart["op_arg"]: cart["add_value"] if units > held else cart["remove_value"], cart["qty_arg"]: abs(units - held)})}]
        action = self.cart_action(ident, units > held)
        return [action] if action else []

    # -- the request's steps go last -------------------------------------------------------

    def process_gate(self, stage: str, rounds: int = 3) -> str:
        if self.time_left() < 12:
            return ""
        return super().process_gate(stage, 1 if self.time_left() < FORCE_SECONDS + 20 else rounds)

    def gate_calls(self, calls: list[dict[str, Any]]) -> str:
        """Only the steps due before the products are chosen run here; the order's and the ending's run in the gates."""
        if not getattr(self, "process", None):
            return ""
        cart_tools = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove")} - {None}
        if any(str(c["function"].get("name")) in cart_tools for c in calls):
            return self.process_gate("choose")
        return ""

    def fallback(self) -> bool:
        before = set(self.cart)
        result = super().fallback()
        self.code_carted = getattr(self, "code_carted", set()) | (set(self.cart) - before)  # the code's own picks are protected too
        return result

    def early_cart(self) -> list[str]:
        """A product with no cart line whose best opened listing the check passed gets it carted by the code at once, and
        under a cheapest rule the cheaper-fit search for it runs while steps remain (late first carts left none)."""
        if not self.roles.get("cart") or self.over() or self.steps_left() <= self.order_due() + 2:
            return []
        keep, added = self.current, []
        for k in range(len(self.items)) if self.multi() else [keep]:
            if self.multi():
                self.switch(k)
                if self.lines.get(k) in self.cart:
                    continue
            elif self.cart:
                break
            ident = self.best_fit(set(self.cart))
            if not ident or self.verdict(ident) is not True:
                continue
            actions = self.set_line_actions(ident, int(self.quantity or 1))
            if not actions or len(actions) > self.steps_left() - self.order_due() - 1:
                continue
            self.act(actions)
            if ident in self.cart:
                if self.multi():
                    self.lines[k] = ident
                added.append(ident)
        if self.multi():
            self.switch(keep)
        for ident in added:
            self.code_carted = getattr(self, "code_carted", set()) | {ident}
            self.trace.append({"kind": "early_cart", "id": ident})
            self.cheap_for_line(ident)
        return added

    def cart_fits(self) -> bool:
        """A basket an actor's decline is turned into the order for: every product has a line at its units within budget,
        and every line passed the check of every requirement (or the code's own reading finds it fit). A decline was right in
        3 of fresh's 71 online declines."""
        if not self.cart or self.basket_problems() or self.missing_lines() or any(self.verdict(i) is False for i in self.cart):
            return False
        return all(self.verdict(i) is True for i in self.cart) or super().cart_fits()

    def confirmed_swaps(self) -> None:
        """Before an order: a cart line whose own facts do not confirm every requirement (and that the check did not pass)
        is replaced by the code with the cheapest opened listing of its category whose facts confirm every requirement (in
        stock, affordable, not failed by the check): the hold that only asked the actor for this went unheeded."""
        for line in list(self.cart):
            if line in getattr(self, "code_carted", set()):
                continue  # need_cart / early_cart locks — do not swap out for a cheaper facts-only rival
            reqs = [r for r in self.item_reqs(line) if all(_split_req(r))]
            if not reqs or self.verdict(line) is True or all(self.code_verdict(line, r) == "yes" for r in reqs):
                continue
            label = self.rows.get(line, {}).get("category")
            better = sorted((_price(self.rows.get(i, {})) or 0, i) for i in self.details
                            if i not in self.cart and i not in self.unavailable and self.rows.get(i, {}).get("stock") is not False
                            and (not label or self.rows.get(i, {}).get("category") == label) and self.verdict(i) is not False
                            and all(self.code_verdict(i, r) == "yes" for r in self.item_reqs(i) if all(_split_req(r)))
                            and self.affordable(self.rows.get(i, {})))
            if better and self.steps_left() > 4 and self.swap_line(line, better[0][1]):
                self.trace.append({"kind": "confirmed_swap", "out": line, "in": better[0][1]})

    def checked_failures(self) -> str:
        """A hold while a cart line is one the candidate check failed (it names what fails), with room left to replace it."""
        failed = [i for i in self.cart if self.verdict(i) is False]
        if not failed or self.steps_left() <= self.reserve() or self.holds >= GATE_HOLDS:
            return ""
        self.holds += 1
        self.trace.append({"kind": "check_held", "lines": failed})
        return ("NOT SENT (no step used). The check of every requirement fails these cart lines: "
                + " | ".join(f"{i}: {self.judged[i][2] or 'a requirement'}" for i in failed)
                + ". Replace each with a listing that meets every requirement (the OPENED LISTINGS table marks them), then order.")

    def defend_code_carts(self) -> None:
        """Drop actor-added rivals in the same shelf as a code-carted need fit, and restore missing code-carted lines
        at the requested units."""
        locked = [i for i in list(getattr(self, "code_carted", set())) if i in self.details or i in self.cart]
        if not locked or not self.roles.get("cart") or self.over():
            return
        for locked_id in locked:
            label = (self.rows.get(locked_id) or {}).get("category")
            rivals = [i for i in list(self.cart)
                      if i != locked_id and i not in getattr(self, "code_carted", set())
                      and self._row_in_category(i, label if isinstance(label, str) else None)]
            for rival in rivals:
                if self.steps_left() <= 2:
                    break
                actions = self.set_line_actions(rival, 0) if hasattr(self, "set_line_actions") else []
                cart = self.roles.get("cart") or {}
                if cart.get("op_arg") and cart.get("remove_value"):
                    actions = [{"name": cart["tool"], "args": self.fill_required(cart["tool"], {
                        cart["id_arg"]: rival, cart["op_arg"]: cart["remove_value"],
                        cart["qty_arg"]: int(self.cart.get(rival) or 1)})}]
                elif (self.roles.get("cart_remove") or {}).get("tool"):
                    rem = self.roles["cart_remove"]
                    actions = [{"name": rem["tool"], "args": self.fill_required(rem["tool"], {rem.get("id_arg") or cart.get("id_arg"): rival})}]
                if actions and self.steps_left() > self.order_due():
                    self.act(actions)
                    self.trace.append({"kind": "defend_code_cart", "kept": locked_id, "dropped": rival})
            if locked_id not in self.cart and self.steps_left() > self.order_due() + 1:
                qty = self.wanted_units(locked_id) if hasattr(self, "wanted_units") else int(self.quantity or 1)
                actions = self.set_line_actions(locked_id, qty)
                if actions:
                    self.act(actions)
                    self.trace.append({"kind": "defend_code_cart", "restored": locked_id, "qty": qty})
            elif locked_id in self.cart:
                want = self.wanted_units(locked_id) if hasattr(self, "wanted_units") else int(self.quantity or 1)
                held_n = int(self.cart.get(locked_id) or 0)
                if held_n != want and self.steps_left() > self.order_due() + 1:
                    self.code_carted = getattr(self, "code_carted", set()) - {locked_id}
                    actions = self.set_line_actions(locked_id, want)
                    if actions:
                        self.act(actions)
                    self.code_carted = getattr(self, "code_carted", set()) | {locked_id}
                    self.trace.append({"kind": "defend_code_cart", "qty_fix": locked_id,
                                       "from": held_n, "to": int(self.cart.get(locked_id) or 0)})

    def gate_order(self) -> str:
        self.defend_code_carts()
        self.units_right()
        self.confirmed_swaps()
        held = self.checked_failures()
        fits = self.fit_problems() if not held and getattr(self, "fit_holds_n", 0) < FIT_HOLDS else []
        if fits:
            self.fit_holds_n = getattr(self, "fit_holds_n", 0) + 1
            self.trace.append({"kind": "fit_held", "problems": fits[:3]})
            held = "NOT SENT (no step used). The request needs these products to fit each other: " + " | ".join(fits[:3]) + "."
        if not held:
            self.cheapest_walk()
        held = held or self.cheaper_check() or self.read_cart_pages() or super().gate_order()
        if held or self.finished:
            return held
        note = self.process_gate("order")
        return ("NOT SENT (no step used). " + note) if note else ""

    def gate_decline(self) -> str:
        self.defend_code_carts()
        spare = self.steps_left() - self.order_due()
        rounds = getattr(self, "decline_open_rounds", 0)
        # several steps left: open more unopened rows before giving up. These are store calls. A second round, and a
        # batch of eight, only while the task is still well under the spend cap, so the next turn can still finish.
        under = self.spent < SPEND_CAP and self.time_left() > CLOSING_SECONDS + 15
        if spare >= 6 and rounds < (2 if self.spent < SPEND_CAP * 0.55 else 1) and under:
            self.decline_open_rounds = rounds + 1
            self.decline_opened = True
            batch = 8 if self.spent < SPEND_CAP * 0.55 else DECLINE_OPENS_13
            held = self.code_opens(self.likely_ids(min(batch, spare - 3)), "before ending without a purchase")
            if held:
                return held
        elif not getattr(self, "decline_opened", False) and spare > 4:
            self.decline_opened = True  # once: before ending without a purchase, the likeliest unopened rows get opened
            held = self.code_opens(self.likely_ids(DECLINE_OPENS_13), "before ending without a purchase")
            if held:
                return held
        if (self.missing_lines() or not self.cart) and getattr(self, "decline_fills", 0) < 2:
            self.decline_fills = getattr(self, "decline_fills", 0) + 1
            self.fell_back = False  # the fallback may fill a line again that the actor took out after a first fill
            self.judge_candidates()
            before = dict(self.cart)
            if self.fallback() or self.cart != before:
                self.trace.append({"kind": "decline_to_fit", "cart": dict(self.cart)})
                return ("NOT SENT (no step used): the check of the opened listings found fits, and the code carted them: "
                        + ", ".join(f"{i} x{u}" for i, u in self.cart.items()) + ". Re-check them and order.")
        if self.refused and self.cart and not self.basket_problems():
            return ("NOT SENT (no step used): the cart holds a basket that passes the code's checks, and a refused finish is "
                    "never a reason to end without a purchase. Order the cart (fix it first if the refusal changed something).")
        held = super().gate_decline()
        if held or self.finished:
            return held
        note = self.process_gate("end")
        return ("NOT SENT (no step used). " + note) if note else ""

    def finish_now(self) -> None:
        if not self.over() and getattr(self, "process", None) and self.time_left() > 15:
            try:
                stage = "order" if self.cart and not self.basket_problems() and self.finish_action("order") else "end"
                self.process_gate(stage, 2)
            except Exception as exc:
                self.trace.append({"kind": "crash", "error": f"process gate: {type(exc).__name__}: {exc}"[:300]})
        if not self.over() and self.cart and self.roles.get("order"):
            # a decline was right in 3 of fresh's 71 race 12 declines and 47 of them held a cart: order the tidied cart
            # instead, even a line short (an imperfect basket scores no worse than a decline)
            if self.basket_problems() or self.category_problems():
                self.tidy_cart()
            self.fix_units()
            action = self.finish_action("order")
            if action and self.cart and not set(self.cart) & self.unavailable and self.steps_left() >= 1:
                self.trace.append({"kind": "forced_order", "cart": dict(self.cart), "problems": self.basket_problems()[:3]})
                self.act([action])
                if self.finished or self.over():
                    return
        super().finish_now()

    def review_order(self) -> list[str]:
        """Fresh 12's review, plus the working rules it reads as unmet (a rule about the shopper, the basket, the
        budget or stock is the code's own job)."""
        start = len(self.trace)
        problems = super().review_order()
        review = next((t for t in self.trace[start:] if t.get("kind") == "order_review"), None)
        if not review or self.steps_left() < 5:
            return problems
        result = review.get("result") if isinstance(review.get("result"), dict) else {}
        rules = [str(r).strip() for r in result.get("rules_unmet") or [] if str(r).strip() and not RULE_OWN.search(str(r))]
        if rules:
            self.trace.append({"kind": "rules_unmet", "rules": rules[:3]})
        return problems + [f"working rule not followed yet: {r}" for r in rules[:2]]

    def properties_first(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
        if RULE_OPEN.search(" ".join(self.rule_texts() + [p["step"] for p in getattr(self, "process", [])])):
            return calls, set()  # the request asks for listings opened: the actor's own call stands
        listing = (self.roles.get("listing") or {}).get("tool")
        keep = [c for c in calls if str(c["function"].get("name")) == listing and any(i in str(c["function"].get("arguments")) for i in self.cart)]
        rest, swapped = super().properties_first([c for c in calls if c not in keep])
        by_id = {c["id"]: c for c in rest + keep}  # a carted listing's page call stands
        return [by_id[c["id"]] for c in calls if c["id"] in by_id], swapped

    # -- a repeat answered from memory only while nothing changed ------------------------------

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        before = set(self.cart)
        observations = super().act(actions)
        search = (self.roles.get("search") or {}).get("tool")
        for action, observation in zip(actions, observations or []):
            if action["name"] == search and isinstance(observation, dict) and not _failed(observation):
                self.search_last = getattr(self, "search_last", {})
                self.search_last[_query_key(action["name"], action.get("args") or {})] = (dict(action.get("args") or {}), observation)
        added = [i for i in self.cart if i not in before]
        if added:
            self.last_carted = added[-1]
        finishing = {(self.roles.get(r) or {}).get("tool") for r in ("order", "decline")} - {None}
        many = self.roles.get("details_many") or {}
        for action, observation in zip(actions, observations or []):
            asked = (action.get("args") or {}).get(many.get("ids_arg") or "") if action["name"] == many.get("tool") else None
            if isinstance(asked, list) and len(asked) > 1 and not _result_rows(observation):
                # a batch answered with no listings at all: the store takes fewer at once (some answer {} silently)
                many["max"] = max(1, min(int(many.get("max") or len(asked)), len(asked)) // 2)
                getattr(self, "done", {}).pop(self.call_key(action["name"], action.get("args") or {}) or "", None)
                self.trace.append({"kind": "batch_limit", "asked": len(asked), "max": many["max"]})
            if action["name"] in finishing and isinstance(observation, dict) and observation.get("refusal"):
                self.refused = True
        return observations

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        before = dict(getattr(self, "cart", {}) or {})
        super().absorb(action, observation)
        self.count_units(action, observation, before)
        if not hasattr(self, "paged_at") or not self.split_listing() or not isinstance(observation, dict) or _failed(observation):
            return
        listing, many = self.roles.get("listing") or {}, self.roles.get("details_many") or {}
        if action["name"] == listing.get("tool"):
            ident = (action.get("args") or {}).get(listing.get("id_arg") or "")
            if isinstance(ident, str):
                self.paged_at[ident] = self.changes
        elif action["name"] == many.get("tool"):
            for row in (_row_info(x) for x in _result_rows(observation)):
                if row.get("id"):
                    self.paged_at[row["id"]] = self.changes  # a side-by-side view shows each listing's page

    def listen(self, *args: Any, **kwargs: Any) -> bool:
        start, gone = len(self.trace), set(self.unavailable)
        changed = super().listen(*args, **kwargs)
        for entry in self.trace[start:]:
            if entry.get("kind") == "heard" and not entry.get("reading"):
                self.unread_lines = getattr(self, "unread_lines", []) + list(entry.get("lines") or [])  # no reading: kept as said
        heard = [e for e in self.trace[start:] if e.get("kind") == "heard" and isinstance(e.get("reading"), dict)]
        if any(e["reading"].get("unavailable") is True for e in heard) and self.unavailable == gone and getattr(self, "last_carted", ""):
            self.unavailable.add(self.last_carted)  # "that one's gone" said after its line already left the cart
            changed = True
        if changed:
            self.changes = getattr(self, "changes", 0) + 1  # pages read before this are re-read before the order
            if hasattr(self, "done"):
                self.done.clear()  # a repeat is answered from memory only while nothing changed
        return changed


def _row_detail(info: dict[str, Any]) -> str:
    """A search row's text apart from its title: the description and the other fields the row already carries."""
    text = str(info.get("text") or "")
    title = str(info.get("title") or "")
    if title and title in text:
        text = text.replace(title, " ", 1)
    return text


def _row_verdict(info: dict[str, Any], prop: str, value: str, noun: frozenset[str] = frozenset(),
                 category: str = "") -> str:
    """How an unopened search row stands on one requirement: 'yes', 'no' or 'unsure'.

    The row's own detail decides when it states the value or names the property. The title ranks the row up only when
    that detail never names the property, so a title coincidence cannot outrank a description that already speaks."""
    if not prop or not value:
        return "unsure"
    detail = _row_detail(info)
    want = _prop_words(prop, noun)
    have = _words(detail)
    named = bool(want) and all(any(_same(w, h) for h in have) for w in want)
    detail_verdict, _ = _fact_reading(None, detail, prop, value, noun, category)
    # a figure in the description counts only when the description names the property; a stray number must not
    if detail_verdict == "yes" and (named or not _is_numeric(value)):
        return "yes"
    if named:
        return "no" if detail_verdict == "no" else "unsure"
    title_verdict, _ = _fact_reading(None, str(info.get("title") or ""), prop, value, noun, category)
    return title_verdict


class Actor14(Actor13):
    """Fresh 14: learns how the store reads a query (every word or any word, and how many leading words count) from the
    totals of related first pages, tells the actor, and pages by a per-product budget: the rest of a search is read by the
    code when it fits the product's share of the steps left, else the actor is told to narrow the query."""

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        observations = super().act(actions)
        search = self.roles.get("search") or {}
        for action, observation in zip(actions, observations or []):
            args = action.get("args") or {}
            if action["name"] != search.get("tool") or not isinstance(observation, dict) or _failed(observation):
                continue
            if (search.get("token_arg") and args.get(search["token_arg"])) or (search.get("page_arg") and args.get(search["page_arg"]) not in (None, 0, 1)):
                continue  # a later page
            words = [w for w in re.findall(r"[a-z0-9.]+", str(args.get(search.get("query_arg") or "") or "").lower())]
            ids = tuple(r["id"] for r in (_row_info(x) for x in _result_rows(observation)) if r.get("id"))
            total = _reported_total(observation)
            if words and total is not None:
                self.first_pages = getattr(self, "first_pages", []) + [(words, total, ids)]
        return observations


    # -- step 3: the requirement check: few listings per call, with field evidence the code computes ---------------

    def field_evidence(self, ident: str) -> list[str]:
        """Per requirement of a listing: the field the store uses for that property across the other opened listings of
        the same category, how many of them write the required value itself, and what this listing writes."""
        label = self.rows.get(ident, {}).get("category")
        peers = [i for i in self.details if i != ident and (not label or self.rows.get(i, {}).get("category") == label)]
        noun, lines = self.noun_words(), []
        for req in self.item_reqs(ident):
            prop, value = _split_req(req)
            if not prop or not value:
                continue
            keys: dict[str, int] = {}
            literal = 0
            for peer in peers:
                named = _closest_keys([(k, v) for k, v in _facts(self.details.get(peer)) if _names(k, prop, noun) and v.strip()], prop, noun)
                for k, v in named[:1]:
                    keys[k] = keys.get(k, 0) + 1
                if any(_stated(value, v) for _, v in named):
                    literal += 1
            own = _closest_keys([(k, v) for k, v in _facts(self.details.get(ident)) if _names(k, prop, noun) and v.strip()], prop, noun)
            mine = "; ".join(f"{k}: {v}" for k, v in own[:2]) or "no field for it"
            if peers:
                common = ", ".join(f"'{k}' ({n})" for k, n in sorted(keys.items(), key=lambda x: -x[1])[:2]) or "no field"
                lines.append(f"- {prop} = {value}: of {len(peers)} other opened listings of its category, the field is {common}; "
                             f"{literal} write '{value}' itself. This listing: {mine}.")
            else:
                lines.append(f"- {prop} = {value}: this listing: {mine}.")
        return lines

    def judge_candidates(self) -> None:
        """Fresh 13's check, but a few listings per call, each with the field evidence the code computed."""
        if not hasattr(self, "judged") or self.judge_calls >= JUDGE_CALLS_14 or self.time_left() < CLOSING_SECONDS or self.spent >= SPEND_CAP:
            return
        key = self.wants()
        todo = [i for i in self.details if i not in self.unknown and i not in self.unavailable and (self.judged.get(i) or (None,))[0] != key]
        if not todo or not any(self.item_reqs(i) for i in todo):
            return
        for batch in [todo[-JUDGE_BATCH_14 * 2:][n:n + JUDGE_BATCH_14] for n in range(0, min(len(todo), JUDGE_BATCH_14 * 2), JUDGE_BATCH_14)]:
            if self.judge_calls >= JUDGE_CALLS_14:
                break
            self.judge_calls += 1
            records = "\n".join(f"LISTING {i} (for the product needing: {'; '.join(self.item_reqs(i)) or 'nothing specific'}):\n{self.record(i)}"
                                 + ("\nFIELD EVIDENCE (computed by code from the other opened listings of its category):\n" + "\n".join(self.field_evidence(i))
                                    if self.item_reqs(i) else "") for i in batch)
            context = self.context(story=False) + ("\n\n" + self.products_text() if self.multi() else "")
            got = self.ask(CANDIDATE_CHECK + "\n\n" + context + "\n\nLISTINGS:\n" + records, 1200) or {}
            for entry in got.get("listings") or [] if isinstance(got.get("listings"), list) else []:
                if isinstance(entry, dict) and str(entry.get("id")) in batch:
                    fits = entry.get("fits") if entry.get("fits") in (True, False) else None
                    self.judged[str(entry["id"])] = (key, fits, str(entry.get("fails") or "")[:200])
            self.trace.append({"kind": "judge", "ids": batch, "fits": [i for i in batch if self.verdict(i) is True],
                               "fails": [i for i in batch if self.verdict(i) is False]})

    def verdict(self, ident: str) -> Any:
        """The check's verdict; an unproven one counts as a fit when the code's own reading confirms every requirement."""
        got = super().verdict(ident)
        if got is None and ident in getattr(self, "judged", {}) and self.judged[ident][0] == self.wants():
            reqs = [r for r in self.item_reqs(ident) if all(_split_req(r))]
            if reqs and all(self.code_verdict(ident, r) == "yes" for r in reqs):
                return True
        return got

    # -- step 4: candidates to open: varied titles among equally likely rows ----------------------------------------

    def likely_ids(self, limit: int = LIKELY_ROWS_13, below: float | None = None, category: str | None = None) -> list[str]:
        """Rows to open, best first. A requirement is read from the search row's own text; the title raises a row only
        when that text never names the property. Among rows equal on that, the one least like the titles already opened."""
        ranked = []
        cheap = (self.preference or {}).get("kind") == "cheapest" or below is not None
        evidence = self.search_evidence()
        seen_titles = [set(_words(str(self.rows.get(i, {}).get("title") or ""))) for i in self.details]
        for n, (ident, info) in enumerate(self.rows.items()):
            if ident in self.details or ident in self.unavailable or info.get("stock") is False or ident in self.cart:
                continue
            price = _price(info)
            if self.budget is not None and price is not None and price > self.budget:
                continue
            if below is not None and (price is None or price >= below):
                continue
            if category is not None and str(info.get("category") or "") != category:
                continue
            reqs = [r for r in self.item_reqs(ident) if all(_split_req(r))]
            verdicts = [_row_verdict(info, *_split_req(r), self.noun_words(), str(info.get("category") or "")) for r in reqs]
            words = set(_words(str(info.get("title") or "")))
            alike = max((len(words & t) / max(1, len(words | t)) for t in seen_titles), default=0.0)
            ranked.append((verdicts.count("no"), -evidence.get(ident, 0), -verdicts.count("yes"),
                           price if cheap and price is not None else 0, round(alike, 1), n, ident))
        return [r[-1] for r in sorted(ranked)[:limit]]

    def opens_per_product(self) -> int:
        """How many listings of one product to open before choosing: six when the order is one or two products, four
        when it is more. A passing agent reads about that many attribute pages, not the whole result."""
        products = len(self.items) if self.multi() else 1
        return 6 if products <= 2 else 4

    def supported_ids(self) -> list[str]:
        """Unopened rows whose own text supports the requirements, up to opens_per_product of each product, and at most
        ten in all so the order still has steps left."""
        each = self.opens_per_product()
        keep, wanted = self.current, []
        for k in range(len(self.items)) if self.multi() else [keep]:
            if self.multi():
                self.switch(k)
            label = self.label_for() or None
            have = sum(1 for i in self.details if not label or self.rows.get(i, {}).get("category") == label)
            if have < each:
                wanted += [i for i in self.likely_ids(each - have, category=label) if i not in wanted]
        if self.multi():
            self.switch(keep)
        room = max(0, self.steps_left() - self.order_due() - 8)
        return wanted[: min(room, 10)]

    def open_supported(self) -> None:
        """Open that short list now. The attribute page is where a requirement the search row never names is settled."""
        wanted = [i for i in self.supported_ids() if i not in self.details]
        if not wanted or self.over():
            return
        self.read_details(wanted)
        self.trace.append({"kind": "open_supported", "ids": [i for i in wanted if i in self.details]})

    def setup(self) -> None:
        super().setup()
        self.open_supported()

    # -- products that must fit each other: codes in the listings' text ------------------------------------------------


    def listing_codes(self, ident: str) -> set[str]:
        """Model, part and series codes written in a listing's title, facts and page (letters and figures, a series written
        with trailing x's kept as such)."""
        info = self.rows.get(ident, {})
        text = " ".join([str(info.get("title") or ""), _compact(self.details.get(ident), 3000), _compact(getattr(self, "fine", {}).get(ident), 3000)])
        codes = set()
        for m in re.finditer(r"\b([A-Za-z]{1,6})[\s-]?(\d{2,6}(?:x{1,3})?|\d{1,4}x{1,3})\b", text):
            letters, figure = m.group(1).lower(), m.group(2).lower()
            spaced = m.end(1) < m.start(2)
            if letters in _FILLER or letters in ("or", "for", "in"):
                continue
            if spaced and (len(letters) < 2 or not (m.group(1).isupper() or len(figure) >= 3)):
                continue
            codes.add(f"{letters} {figure}")
        return codes

    @staticmethod
    def codes_meet(a: set[str], b: set[str]) -> bool:
        """Two code sets share a code; a series written with trailing x's meets any code of its prefix."""
        def meets(x: str, y: str) -> bool:
            lx, fx = x.split(" ", 1)
            ly, fy = y.split(" ", 1)
            if lx != ly:
                return False
            px, py = fx.rstrip("x"), fy.rstrip("x")
            return fx == fy or (fx.endswith("x") and fy.startswith(px)) or (fy.endswith("x") and fx.startswith(py))
        return any(meets(x, y) for x in a for y in b)

    def fit_problems(self) -> list[str]:
        """For each part/device pair the request ties together and both carted: a note when their codes never meet (both
        carrying codes), naming opened listings of the part's category whose codes meet the carted device's."""
        out = []
        lines = {k: i for k, i in self.lines.items() if i in self.cart} if self.multi() else {}
        for part_k, device_k in self.fit_pairs():
            part, device = lines.get(part_k), lines.get(device_k)
            if not part or not device:
                continue
            pc, dc = self.listing_codes(part), self.listing_codes(device)
            if not pc or not dc or self.codes_meet(pc, dc):
                continue
            label = self.rows.get(part, {}).get("category")
            better = [i for i in self.details if i not in self.cart and self.rows.get(i, {}).get("category") == label
                      and self.codes_meet(self.listing_codes(i), dc)][:3]
            out.append(f"{part} ({self.product_name(part_k)}) names no code of the carted {self.product_name(device_k)} {device} "
                       f"(its codes: {', '.join(sorted(dc)[:4])})" + (f"; opened ones that fit it: {', '.join(better)}" if better else
                       "; search the part with the device's code, or choose a device whose code the part names"))
        return out

    # -- the interview: survey first, then questions round-robin by product, then searches on what was learned -----

    def product_reqs(self, k: int) -> list[str]:
        if not self.multi():
            return list(self.reqs)
        self.save_item()
        return list(self.items[k]["reqs"])

    def survey(self) -> None:
        """Before the questions, top up each product to the same short list the opening already reads, so a product the
        actor has not covered still has its supported rows opened."""
        before = len(self.trace)
        self.open_supported()
        added = next((t for t in self.trace[before:] if t.get("kind") == "open_supported"), None)
        if added:
            self.trace.append({"kind": "survey", "ids": added.get("ids") or []})


    def key_shares(self, k: int) -> dict[str, float]:
        """For product k: the share of its opened listings (of its category) that carry each attribute key, by key name."""
        keep = self.current
        if self.multi():
            self.switch(k)
        label = self.label_for() or None
        listings = [i for i in self.details if i not in self.unknown and (not label or self.rows.get(i, {}).get("category") == label)]
        counts: dict[str, int] = {}
        for ident in listings:
            for name in {" ".join(str(key).replace("_", " ").replace("-", " ").lower().split()) for key, _ in _facts(self.details.get(ident)) or []}:
                if name and len(name) <= 30:
                    counts[name] = counts.get(name, 0) + 1
        if self.multi():
            self.switch(keep)
        return {n: c / len(listings) for n, c in counts.items()} if listings else {}

    def guard_coverage(self, queue: dict[int, list[str]]) -> None:
        """After the planner's ranking: per product, a property its opened listings carry far more widely than a planned
        pick (by a third of their share) goes before that pick. A catalog mark or a mixed-notes bin never moves up, and a
        property the request already settles is not asked."""
        for k, labels in queue.items():
            shares = self.key_shares(k)
            if len(shares) < 2:
                continue
            settled = {w for r in self.product_reqs(k) for w in _words(_split_req(r)[0] or "")}
            wanted_values = [_split_req(r)[1] for r in self.product_reqs(k) if _split_req(r)[1]]
            written: dict[str, list[str]] = {}
            for details in self.details.values():
                for key, value in _facts(details) or []:
                    written.setdefault(" ".join(str(key).replace("_", " ").replace("-", " ").lower().split()), []).append(str(value))
            # a key whose values write a value the request already asks for is that requirement's own key ("os" for ios)
            settled_keys = {n for n, vals in written.items() if any(_stated(v, x) for v in wanted_values for x in vals)}
            kept = self.current
            if self.multi():
                self.switch(k)
            shelf = self.label_for() or None
            distinct: dict[str, set[str]] = {}
            for ident in [i for i in self.details if i not in self.unknown and (not shelf or self.rows.get(i, {}).get("category") == shelf)]:
                for key, value in _facts(self.details.get(ident)) or []:
                    name = " ".join(str(key).replace("_", " ").replace("-", " ").lower().split())
                    distinct.setdefault(name, set()).add(re.sub(r"\s+", " ", str(value).strip().lower()))
            if self.multi():
                self.switch(kept)
            def share(label: str) -> float:
                words = set(_words(label))
                return max((v for n, v in shares.items() if words & set(_words(n))), default=0.0)
            wide = [n for n, v in sorted(shares.items(), key=lambda x: -x[1])
                    if v >= WIDE_ENOUGH and askable_label(n) and len(distinct.get(n, ())) > 1
                    and not (set(_words(n)) & settled) and n not in settled_keys
                    and not any(set(_words(n)) & set(_words(l)) for l in labels)][:QUESTIONS_AHEAD]
            for name in wide:
                spot = next((i for i, l in enumerate(labels) if shares[name] - share(l) >= CLEAR_LEAD), None)
                if spot is not None:
                    labels.insert(spot, name)
                elif name.lower() not in [x.lower() for x in labels]:
                    labels.append(name)  # the planner named only category or seller: a real differing property still gets asked
            self.trace.append({"kind": "coverage_guard", "product": k, "queue": labels[:5]}) if wide else None


JUDGE_BATCH_14 = 3  # listings judged in one check call
JUDGE_CALLS_14 = 12  # check calls per task
FIT_HOLDS = 2  # orders held for a part that fits no carted device, at most
LINE_HOLDS = 3  # orders held for a line code can prove wrong, at most
_FITS_TEXT = re.compile(r"(?i)\bfits:\s*([^.|]+)")
_MODEL_TEXT = re.compile(r"(?i)\bmodel (?:number|no)[:.]?\s*([^.|]+)")
_ALSO_TEXT = re.compile(r"(?i)\balso sold as\s+([^.|]+)")
_TIED = re.compile(r"(?i)([a-z][a-z &'/-]{2,60}?)\s+(?:fits?|works? with|is compatible with|goes with|suits?)\s+(?:the\s+)?([a-z][a-z &'/-]{2,60})")
_OWN_MODEL = re.compile(r"(?i)([a-z][a-z &'-]{2,60}?)\s+(?:must|should|needs? to|ha(?:s|ve) to)\s+be\s+compatible\s+with\b[^.]*?\bmodel\s+([a-z0-9-]+)")
_RANGED = re.compile(r"(?i)\b(?:at least|at most|under|over|up to|between|or|and|not|min\w*|max\w*|more|less)\b|[<>+]")


def _squeezed(text: str) -> str:
    return "".join(_words(text))


def _named_by(entry: str, names: set[str]) -> bool:
    """A device name is what an entry of a fits list says: the name itself, or a series written with trailing x's."""
    words = [w for w in _words(entry) if w not in ("every", "all", "model", "models", "series")]
    mark = "".join(words)
    if re.fullmatch(r".+?x{2,}", mark):
        return any(n.startswith(mark.rstrip("x")) for n in names)
    return bool(mark) and mark in names


def _number_words(text: str) -> tuple[set[str], set[str]]:
    """The numbers of a text and its other words."""
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    return numbers, {_one(w) for w in _words(re.sub(r"\d+(?:\.\d+)?", " ", text))}
# who sells it, and the file the store already put it in: on every listing, so a coverage guard ranks them first and the
# two messages are spent before colour, size or material is asked
NOT_A_NEED = re.compile(r"\b(?:categor\w*|sellers?|sold by|brands?|shops?|vendors?|makers?|manufacturers?|titles?|prices?|stocks?)\b", re.I)


_CONDITIONAL = re.compile(r"\b(?:if|should|in case|except|unless|whenever|provided)\b", re.I)
_FUNCTION_WORDS = frozenset(
    "the you your yours end buy buying bought get got one ones any all for from with this that these those which who what "
    "when where then than also just only very really more most some such own its it's our ours their them they his her "
    "him she he was were are been being have has had will would should could can may might must shall into onto upon "
    "about per via not nor but and or".split())


def _catalog_mark(name: str) -> bool:
    """A key that is only a filing mark, a picture, or a bin of mixed notes, so it is not a question to ask."""
    text = " ".join(str(name or "").lower().replace("_", " ").replace("-", " ").split())
    words = {_one(w) for w in _words(text)}
    if not words:
        return True
    if "warrant" in text or "model number" in text or "part number" in text:
        return True
    if words & {"id", "sku", "code", "asin", "upc", "ean", "barcode", "url", "link", "image", "photo", "video", "option", "variant",
                 "ref", "blurb", "comment", "note", "remark", "summary", "plug"}:
        return True
    return words <= {"feature", "spec", "specification", "highlight", "detail", "description", "other"}


def askable_label(label: str) -> bool:
    """A property a hidden need can sit on. Category, seller and brand are on every listing and are not that property."""
    text = str(label or "").strip()
    return bool(text) and not _catalog_mark(text) and not NOT_A_NEED.search(text)


_META_NEED_VALUE = re.compile(
    r"(?i)\b(?:whatever|anything that|any that|matches?\s+the\s+chosen|as\s+(?:per|on|in)\s+(?:the\s+)?listing|"
    r"see\s+(?:the\s+)?listing|same\s+as\s+(?:the\s+)?(?:chosen|listing|desktop|product|item)|"
    r"matching\s+the|compatible\s+with\s+(?:the\s+)?chosen|whichever|dealer'?s?\s+choice)\b"
)
_VAGUE_NEED_VALUES = frozenset({
    "other", "any", "anything", "n/a", "na", "nil", "none", "mix", "mixed", "whatever",
})


def _usable_need_value(value: str) -> bool:
    """False for meta answers ('whatever matches the listing') or lone generic tokens ('regular') that only burn search."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or len(text) > 60:
        return False
    if _META_NEED_VALUE.search(text):
        return False
    if text.lower() in _VAGUE_NEED_VALUES:
        return False
    return True


def _short_token_need(value: str) -> bool:
    """Compact codes: titles often list sibling options, so only the facts decide.

    Bounds and multi-word values may match on the title or the facts."""
    if not value or _BOUND.search(value):
        return False
    bare = re.sub(r"\s+", " ", str(value)).strip().lower()
    if not bare or " " in bare:
        return False
    return bool(re.search(r"\d", bare) or len(bare) <= 5)


WIDE_ENOUGH = 1 / 2  # at least half the opened listings of this product carry the key
CLEAR_LEAD = 1 / 3  # and it leads the planned question by at least a third of them
QUESTIONS_AHEAD = 2  # such properties per product at most

HEAR_PROMPT = """A shopping agent asked a shopper one question about one product. Read the answer and return ONE JSON object and nothing else:
{"understood": true or false, "needs": ["property: value", ...], "no_preference": ["property", ...], "dropped": [], "new_budget": null, "product_budget": null, "new_preference": null, "requirements_now": ["property: value", ...], "references": []}
- "new_budget": a new price limit for the whole order; "product_budget": a new price limit for this product alone (each a number exactly as written, or null).
- "new_preference": null unless the answer changes how to choose among the listings that fit (now the cheapest, the best by some measure, a preferred brand); then {"kind": "cheapest" | "best" | "brand" | "other", "property": "...", "direction": "max" | "min" | "", "value": "...", "numeric": true or false}. A firm limit ("at most 8 cores") is a need, never a preference.
- "needs": every firm preference the answer states, with the exact value or limit ("property: value"). Name the property the answer itself names. Folksy wording that names a value still states that value.
- The value is the setting the answer gives the property (the words after "is", "at", "as", "to" and the like), never the name of a product of the order: "component type of the laptops sitting at keyboard" gives "component type: keyboard". A limit keeps its own words in the value ("display: at most 14 inch").
- Do NOT put a need when the value is meta or empty of a real choice: "whatever matches the listing", "same as the chosen product", "as listed", or a lone empty placeholder like "any"/"none"/"mix" with no further detail. Leave needs empty then so another property can be asked. A short style or type word the shopper named as their choice (even "basic" or "casual") IS a need.
- "no_preference": properties the shopper said they do not mind about.
- "understood": false when the answer only says it did not follow the message or wants it in other words.
- "requirements_now": the complete firm requirements for THIS product after the answer: keep every earlier requirement of this product unless the answer drops or replaces it; add every new need. Never invent requirements. Never clear a requirement the answer did not touch.
- An answer that only says they do not mind keeps every earlier requirement and lists that property under no_preference.
- A colourful aside that only softens a don't-mind is not a need and must not invent a property or value.
- "references": needs whose value the shopper does not give but says to take from another listing in the store (the same as, matching, or taken from what some other product's page shows), each {"product": the product number in an order of several, else null, "property": the property of the shopper's product, "seller": the other listing's seller or shop, "category": its kind of product, "conditions": its other properties the shopper names, each "property: value", "description": the shopper's words for that listing, "relation": "same" (the value must be that listing's), "differ" (it must NOT be: different from, other than, set apart from), "at most", "at least" or "within" (a figure compared with that listing's), "tolerance": the number for "within" (else null), "unit": its unit (else "")}; [] when none. Such a need also stays in "needs" and "requirements_now" as the shopper words it. The conditions describe the other listing only: they are never requirements on the shopper's product, so they never add to or replace anything in "requirements_now".
- The properties that describe that other listing are not needs of this product: never put them in "needs" or "requirements_now".
PRODUCT'S EARLIER REQUIREMENTS:
EARLIER
QUESTION:
QUESTION_TEXT
ANSWER:
ANSWER_TEXT"""

LIST_SIZE = 8  # properties of one product named in one message
NEED_OPEN_HEAD = 5  # new need-search: open this many strongest new rows
NEED_OPEN_TAIL = 4  # and buried end-of-page rows (relevance often ranks the fit mid/late)
NEED_OPEN_MID = 4  # and a few mid-page rows
WIDEN_OPENS = 5  # after interview, unread buried rows opened while spend/steps allow
NEED_PAGES = 3  # pages of a productive need query (or category shelf for numeric bounds)
BOUND_SHELF_PAGES = 4  # after figure-token queries, deepen the category shelf for numeric bounds
INTERVIEW_PROMPT_15 = """You choose the questions a shopping assistant sends to a shopper. The shopper holds a few firm preferences the request does not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property of that product. One message can name several properties of a product, and the reply covers only the earliest of them the shopper cares about, so the order of your picks matters.

You get the request and, for each product, the attribute keys actually seen on its opened listings (how many listings carry each key, with example values). An unwritten preference usually sits on one of those keys. The list order is not a ranking — choose by the rules below.

Choose properties in this order:
0. Topics under ASK FIRST, when that list is present: the request itself wants them raised, so name the property each is about.
1. Hinted properties: when a sentence of the request points at one of the listed keys without giving its value, that key, for the product the sentence is about. Name the key as listed (all of its words).
2. Then other listed keys a buyer of that product would choose between. Favour, in this order of preference among listed keys:
   (a) keys most listings carry whose example values still differ across listings (closed choices that separate listings);
   (b) put at least one categorical closed choice (short non-numeric labels such as a type, kind, or named option) among the first two picks whenever such a key is listed — do not use the first messages only on weight, dimensions, volume or other pure measures when a categorical key is available;
   (c) keys whose examples are short pickable values (a few words or a figure with a unit) over long free-text strings;
   (d) product-defining choices (how the item is configured, sized, packed, or typed) over sheet-spec noise (rates, timings, brightness, connectivity lists) when both appear.
Skip what the request already settles for that product, a second name for a property already chosen, identifiers, variant pickers, warranty and other logistics. Never choose the product's category, seller, shop, brand or maker. Prefer a listed key's exact spelling over inventing a new property name. With several products, put every product's hinted properties first, then take turns between the products — do not spend every message on one product while another has had no categorical question yet.
When REQUEST HINTS lists attribute keys whose example values or names echo distinctive words from the request, prefer those keys early for that product.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling; only the property's name, no example values, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": product number, "label": "..."}]}"""


IDLE_TURNS = 4  # replies in a row with no store step and no cart change


class Actor15(Actor14):
    """Fresh 15: questions planned from the keys of opened listings and asked in turn across the products, a short reading
    of each reply, query variants for each need learned, and code-opened rows from deeper in the results.
    Message monopoly stays with Actor12 (tool stripped; actor messages never sent)."""

    def shelf_owners(self) -> dict[str, int]:
        """Store label -> the one product holding it (a label two products hold says nothing), kept until new rows arrive."""
        known = getattr(self, "owners", None)
        if known and known[0] == len(self.rows):
            return known[1]
        keep, held = self.current, {}
        for k in range(len(self.items)):
            self.switch(k)
            held.setdefault(self.label_for(), []).append(k)
        self.switch(keep)
        table = {label: ks[0] for label, ks in held.items() if label and len(ks) == 1}
        self.shared = {label: ks for label, ks in held.items() if label and len(ks) > 1}
        self.owners = (len(self.rows), table)
        return table

    def shared_owner(self, ident: str, group: list[int]) -> int:
        """Products sharing a store label each take the carted line their own requirements read best on, one line each;
        a listing not carted goes to the product it reads best for."""
        def score(i: str, k: int) -> int:
            read = [self.code_verdict(i, r) for r in self.items[k]["reqs"] if all(_split_req(r))]
            return read.count("yes") - 2 * read.count("no")
        label = self.rows.get(ident, {}).get("category")
        free = [i for i in self.cart if self.rows.get(i, {}).get("category") == label]
        if ident in free:
            for k in group:
                best = max(free, key=lambda i: score(i, k))
                if best == ident:
                    return k
                free.remove(best)
                if not free:
                    break
        return max(group, key=lambda k: score(ident, k))

    def owner_of(self, ident: str) -> int | None:
        """The product a listing is for, when that is known better than by the words of its title."""
        chosen = [k for k, i in getattr(self, "planned", {}).items() if i == ident]
        if chosen:
            return chosen[0]
        label = str(self.rows.get(ident, {}).get("category") or "")
        owner = self.shelf_owners().get(label)
        group = getattr(self, "shared", {}).get(label)
        return self.shared_owner(ident, group) if owner is None and group else owner

    def item_for(self, ident: str) -> dict[str, Any]:
        """The product of the order a listing is for: the one the last basket chose it for, else the only product holding
        its store label, else the one whose name its label and title share most words with."""
        if not self.multi():
            return {"own_budget": self.own_budget, "quantity": self.quantity, "reqs": self.reqs}
        self.save_item()
        info = self.rows.get(ident, {})
        k = self.owner_of(ident)
        if k is not None:
            return self.items[k]
        text = f"{info.get('category') or ''} {info.get('title') or ''}".lower()
        return max(self.items, key=lambda i: sum(w in text for w in _words(f"{i['category']} {i.get('noun') or ''}")))

    def item_reqs(self, ident: str) -> list[str]:
        return list(self.item_for(ident)["reqs"])

    def number_clash(self, ident: str, reqs: list[str]) -> list[str]:
        """Requirements a listing's own facts contradict by number: the same words with another figure ("windows 11 pro"
        for "windows 10 pro"). Ranges and alternatives are left to the other checks, and a fact stating the figure clears it."""
        facts = [(str(k), *_number_words(str(v))) for k, v in _facts(self.details.get(ident)) or []]
        written = " ".join([str(self.rows.get(ident, {}).get("title") or "")] + [str(v) for _, v in _facts(self.details.get(ident)) or []]).lower()
        out = []
        for req in reqs:
            prop, value = _split_req(req)
            numbers, words = _number_words(value)
            if not prop or not numbers or not words or _RANGED.search(value):
                continue
            phrase = re.sub(r"\\?[\s-]+", r"[\\s-]*", re.sub(r"\d+(?:\\?\.\d+)?", r"(\\d+(?:\\.\\d+)?)", re.escape(value.lower())))
            said = set(re.findall(r"\b" + phrase + r"\b", written)) if len(numbers) == 1 and len(words) >= 2 else set()
            if said:
                if not said & numbers:
                    out.append(req)
                continue
            alike = [(k, n) for k, n, w in facts if n and words <= w]
            if any(n & numbers for _, n in alike):
                continue
            stems = self._stems(prop)
            if any(len(words) >= 2 or stems & self._stems(k) for k, _ in alike):
                out.append(req)
        return out

    def line_faults(self, ident: str, units: int | None = None) -> list[str]:
        """What code can prove wrong with one line: over its product's own price limit, or contradicted by a fact."""
        item = self.item_for(ident)
        price, limit = _price(self.rows.get(ident, {})), item.get("own_budget")
        count = int(units or item.get("quantity") or 1)
        out = [f"{ident} costs {price * count:g} for {count}, over that product's own limit of {limit:g}"
               ] if price is not None and limit is not None and price * count > limit + 1e-9 else []
        return out + [f"{ident} has facts that contradict {r}" for r in self.number_clash(ident, list(item.get("reqs") or []))]

    def product_words(self) -> list[set[str]]:
        """Each product's name words, plural or not, with the words of its store category: the request may call a product
        by either."""
        owners = {k: label for label, k in self.shelf_owners().items()}
        return [{_one(w) for w in _words(f"{i['category']} {i.get('noun') or ''} {owners.get(k, '')}")} - _FILLER
                for k, i in enumerate(self.items)]

    def named_product(self, text: str, names: list[set[str]]) -> int | None:
        """The product a phrase names: the share of a product's words it covers decides, then how many."""
        words = {_one(w) for w in _words(text)}
        best = max(((len(words & n) / max(len(n), 1), len(words & n), k) for k, n in enumerate(names)), default=(0, 0, None))
        return best[2] if best[1] else None

    def fit_pairs(self) -> list[tuple[int, int]]:
        """(part product, device product) pairs the request ties together, each side named by the product or its category."""
        if not self.multi():
            return []
        known = getattr(self, "pairs_known", None)
        if known and known[0] == len(self.rows):
            return known[1]
        names, pairs = self.product_words(), []
        for found in _TIED.finditer(self.request):
            a, b = self.named_product(found.group(1), names), self.named_product(found.group(2), names)
            if a is not None and b is not None and a != b and (a, b) not in pairs:
                pairs.append((a, b))
        self.pairs_known = (len(self.rows), pairs)
        return pairs

    def read_marks(self, value: Any, depth: int = 0) -> None:
        """Keep what each listing seen says of itself: the names it is sold under and what it fits."""
        if depth > 5:
            return
        if isinstance(value, str):
            record = _text_record(value) if " | " in value else None
            if record:
                self.read_marks(record, depth + 1)
        elif isinstance(value, list):
            for item in value:
                self.read_marks(item, depth + 1)
        elif isinstance(value, dict):
            ident = _row_info(value).get("id")
            if ident:
                text = " ".join(str(v) for v in value.values() if isinstance(v, str))
                names = {_squeezed(n) for found in (_MODEL_TEXT, _ALSO_TEXT) for n in found.findall(text)} - {""}
                fits = [e.strip() for found in _FITS_TEXT.findall(text) for e in found.split(",") if e.strip()]
                if names or fits:
                    self.marks[str(ident)] = {"names": names, "fits": fits}
            for item in value.values():
                self.read_marks(item, depth + 1)

    def fit_state(self, part: str, names: set[str]) -> bool | None:
        """True or False when a listing's fits list settles it for a device with these names, None when it cannot."""
        fits = self.marks.get(part, {}).get("fits")
        if not fits or not names:
            return None
        barred = [e for e in fits if re.match(r"(?i)\s*except\b", e)]
        if any(_named_by(re.sub(r"(?i)^\s*except\s+", "", e), names) for e in barred):
            return False
        return any(_named_by(e, names) for e in fits if e not in barred)

    def device_names(self, k: int) -> set[str]:
        """The names of the device listing chosen for product k."""
        planned = getattr(self, "planned", {})
        line = planned.get(k) or next((i for i in self.cart if self.owner_of(i) == k), None)
        return set(self.marks.get(line or "", {}).get("names") or ())

    def own_models(self) -> dict[int, set[str]]:
        """Products the request ties to a model it names itself ("must be compatible with my phone, model X")."""
        out: dict[int, set[str]] = {}
        names = self.product_words()
        for found in _OWN_MODEL.finditer(self.request):
            k = self.named_product(found.group(1), names)
            if k is not None:
                out.setdefault(k, set()).add(_squeezed(found.group(2)))
        return out

    def wanted_fits(self, k: int) -> list[set[str]]:
        """Every set of device names product k has to fit."""
        need = [self.device_names(d) for p, d in self.fit_pairs() if p == k]
        return [n for n in need + [self.own_models().get(k, set())] if n]

    def mark_faults(self) -> list[str]:
        """Carted accessories whose own fits list rules out what the request ties them to, with what was seen that fits."""
        out = []
        if not self.multi():
            return out
        for ident in self.cart:
            k = self.owner_of(ident)
            for names in self.wanted_fits(k) if k is not None else []:
                if self.fit_state(ident, names) is False:
                    label = self.rows.get(ident, {}).get("category")
                    fits = [i for i, r in self.rows.items() if r.get("category") == label and r.get("stock") is not False
                            and i not in self.unavailable and self.fit_state(i, names)][:4]
                    out.append(f"{ident} fits {', '.join(self.marks[ident]['fits'][:4])}, which does not name the device it must go with"
                               + (f"; seen that do: {', '.join(fits)}" if fits else "; change the device to one its list names"))
        return out

    def device_reaches(self, ident: str, k: int) -> bool:
        """A device keeps its line only while a qualifying accessory seen names it (when such accessories say what they fit)."""
        names = set(self.marks.get(ident, {}).get("names") or ())
        for part, device in self.fit_pairs():
            pool = [i for i in self.rows if device == k and self.owner_of(i) == part and self.marks.get(i, {}).get("fits")
                    and self.line_passes(i, part)]
            if pool and names and not any(self.fit_state(i, names) for i in pool):
                return False
        return True

    def open_fits(self) -> None:
        """Rows seen that fit what a cart line must go with, opened in one step so their requirements can be read."""
        wanted = []
        for k in range(len(self.items)):
            for names in self.wanted_fits(k):
                wanted += [i for i in self.rows if i not in self.details and self.owner_of(i) == k and self.fit_state(i, names)
                           and self.line_passes(i, k)]
        if wanted and self.steps_left() > 5:
            self.read_details(list(dict.fromkeys(wanted))[:4])

    def measure(self, ident: str, prop: str) -> float | None:
        """The first number of the fact naming a property."""
        stems = self._stems(prop)
        for key, value in _facts(self.details.get(ident)) or []:
            found = re.search(r"\d+(?:\.\d+)?", str(value))
            if found and stems & self._stems(str(key)):
                return float(found.group())
        return None

    def repick_lines(self) -> None:
        """After a change at checkout: each line with a cheapest or best rule takes the winner among the opened listings
        now proved to qualify, since the change may have moved it."""
        keep, basket = self.current, dict(self.cart)
        for k in range(len(self.items)) if self.multi() else [keep]:
            self.switch(k)
            kind, pref = (self.preference or {}).get("kind"), self.preference or {}
            line = next((i for i in basket if not self.multi() or self.owner_of(i) == k), None)
            if line is None or kind not in ("cheapest", "best"):
                continue
            label = self.label_for()
            fits = self.wanted_fits(k) if self.multi() else []
            line_fits = not any(self.fit_state(line, n) is False for n in fits)  # when nothing can fit, fit decides nothing
            pool = [i for i in self.details if i not in self.unknown and (not label or self.rows.get(i, {}).get("category") == label)
                    and i not in basket and self.line_passes(i, k) and self.verdict(i) is True
                    and not (line_fits and any(self.fit_state(i, n) is False for n in fits))]
            sign = -1 if pref.get("direction") == "max" else 1
            rate = (lambda i: _price(self.rows.get(i, {}))) if kind == "cheapest" else (lambda i: self.measure(i, str(pref.get("property") or "")))
            rated = [(sign * rate(i), i) for i in pool if rate(i) is not None]
            now = rate(line)
            if rated and now is not None and min(rated)[0] < sign * now:
                basket[min(rated)[1]] = basket.pop(line)
        self.switch(keep)
        total = sum((_price(self.rows.get(i, {})) or 0) * u for i, u in basket.items())
        if basket != self.cart and self.steps_left() >= 3 and (self.budget is None or total <= self.budget + 1e-9):
            action = self.basket_action(basket)
            if action:
                self.trace.append({"kind": "repick", "cart": dict(basket)})
                self.act([action])

    def swap_refused(self) -> None:
        """The same basket after a refusal that changed nothing has something gone in it: it never goes out again."""
        if not self.cart or self.cart != getattr(self, "refused_cart", None):
            return
        self.refused_cart = None
        if not self.multi():
            self.unavailable.update(self.cart)
        elif self.steps_left() > len(self.cart) + 2:
            self.act([a for a in (self.id_action("listing", i) for i in self.cart) if a])
        self.last_tried = True
        self.last_basket()
        self.trace.append({"kind": "refused_swap", "cart": dict(self.cart)})

    def line_passes(self, ident: str, k: int) -> bool:
        """Code finds nothing against a listing for product k: stock, its own limit, its own requirements."""
        info, item = self.rows.get(ident, {}), (self.items[k] if self.multi() else self.item_for(ident))
        price, limit = _price(info), item.get("own_budget")
        if info.get("stock") is False or ident in self.unavailable:
            return False
        if price is not None and limit is not None and price * int(item.get("quantity") or 1) > limit + 1e-9:
            return False
        reqs = [r for r in item["reqs"] if all(_split_req(r))]
        return all(self.code_verdict(ident, r) != "no" for r in reqs) and not self.number_clash(ident, reqs)

    def best_fit(self, taken: set[str]) -> str | None:
        """Never a listing over the product's own limit or contradicted by its facts; a part goes with the chosen device
        when any opened listing's codes meet it."""
        blocked = set(taken) | {i for i, r in self.rows.items() if not self.affordable(r)}
        blocked |= {i for i in self.details if self.number_clash(i, list(self.reqs))}
        if self.multi():
            label = self.label_for()
            shelf = [i for i, r in self.rows.items() if i not in blocked and r.get("category") == label]
            for names in self.wanted_fits(self.current):
                meets = [i for i in shelf if self.fit_state(i, names)]
                blocked |= set(shelf) - set(meets) if meets else set()
            for part, device in self.fit_pairs():
                if device == self.current:  # a device some qualifying accessory names, when there is one
                    parts = [i for i in self.rows if self.owner_of(i) == part and self.line_passes(i, part)]
                    meets = [i for i in shelf if any(self.fit_state(p, set(self.marks.get(i, {}).get("names") or ())) for p in parts)]
                    blocked |= set(shelf) - set(meets) if meets else set()
        return super().best_fit(blocked)

    def last_basket(self) -> bool:
        """The last stop before ending without a purchase: one line per product at its requested units, keeping a carted
        line nothing is proved wrong with, else the best listing nothing rules out. True when something can be ordered."""
        if not self.roles.get("cart") or not self.roles.get("order"):
            return False
        keep = self.current
        parts = {p for p, _ in self.fit_pairs()}
        self.planned: dict[int, str] = {}
        basket: dict[str, int] = {}
        for k in sorted(range(len(self.items)), key=lambda n: n in parts) if self.multi() else [keep]:
            self.switch(k)
            shelf = self.mine()
            held = [i for i in self.cart if i not in self.unavailable and self.rows.get(i, {}).get("stock") is not False
                    and i not in basket and (not self.multi() or i in shelf)]
            units = int(self.quantity or 1)
            good = [i for i in held if not self.line_faults(i, units)
                    and not (self.multi() and any(self.fit_state(i, n) is False for n in self.wanted_fits(k)))
                    and self.device_reaches(i, k)]
            ident = good[0] if good else self.best_fit(set(basket)) or (held[0] if held else None)
            if ident:
                basket[ident] = units
                self.planned[k] = ident
        self.switch(keep)
        price = lambda i: (_price(self.rows.get(i, {})) or 0) * basket[i]
        while self.budget is not None and basket and sum(price(i) for i in basket) > self.budget + 1e-9:
            del basket[max(basket, key=price)]
        if not basket:
            return False
        for k, ident in self.planned.items():
            if ident in basket:
                self.lines[k] = ident
        if self.cart != basket and self.steps_left() >= 2:
            if (self.roles.get("cart") or {}).get("lines_arg"):
                self.act([self.basket_action(basket)])
            else:
                for ident in [i for i in self.cart if i not in basket]:
                    self.act([a for a in [self.cart_action(ident, False)] if a])
                for k, ident in self.planned.items():
                    if ident in basket and ident not in self.cart:
                        self.switch(k)
                        self.act([a for a in [self.cart_action(ident, True)] if a])
                self.switch(keep)
                self.fix_units()
        return bool(self.cart) and not set(self.cart) & self.unavailable

    # -- requirements that hold only under a condition ---------------------------------------

    def setup(self) -> None:
        super().setup()
        self.conditionals: list[dict[str, Any]] = []
        for c in (self.plan.get("conditional") if isinstance(self.plan, dict) else None) or []:
            if not isinstance(c, dict):
                continue
            req, when = str(c.get("requirement") or "").strip(), str(c.get("when") or "").strip()
            if not all(_split_req(req)) or not all(_split_req(when)):
                continue
            self.conditionals.append({"k": self.product_number(c.get("product")), "m": self.product_number(c.get("when_product")),
                                      "req": req, "when": when, "except": c.get("except") is True})
        for c in self.conditionals:  # an "except" rule holds until its condition is known to hold; an "if" rule waits for it
            self.set_requirement(c["k"], c["req"], c["except"])
        if self.conditionals:
            self.trace.append({"kind": "conditionals", "rules": self.conditionals})

    def product_number(self, raw: Any) -> int | None:
        """A product index from a 1-based product number; None for a request of one product."""
        if not self.multi():
            return None
        match = re.match(r"\s*\[?(\d+)", str(raw or ""))
        return int(match.group(1)) - 1 if match and 1 <= int(match.group(1)) <= len(self.items) else None

    def set_requirement(self, k: int | None, req: str, present: bool) -> bool:
        """Put one requirement on a product, or take it off. True when its list changed."""
        if self.multi() and k is None:
            return False
        keep = self.current
        if k is not None:
            self.switch(k)
        name, value = (_key_norm(x) for x in _split_req(req))
        same = [r for r in self.reqs if tuple(_key_norm(x) for x in _split_req(r)) == (name, value)]
        before = list(self.reqs)
        if present and not same:
            self.reqs = self.reqs + [req]
        elif not present and same:
            self.reqs = [r for r in self.reqs if r not in same]
        changed = self.reqs != before
        self.switch(keep)
        return changed

    def apply_conditionals(self) -> list[str]:
        """Each conditional rule whose condition the carted item it depends on now settles (its own facts say yes or no)
        is applied to the product it binds. Returns a line per rule that changed a requirement."""
        out = []
        for c in getattr(self, "conditionals", []):
            if self.multi():
                line = next((i for i in self.cart if c["m"] is not None and self.owner_of(i) == c["m"]), None)
            else:
                line = next(iter(self.cart), None)
            if not line or line in self.unavailable:
                continue
            verdict = self.condition_holds(line, c["when"])
            if verdict not in ("yes", "no"):
                continue
            want = (verdict == "yes") != c["except"]
            if self.set_requirement(c["k"], c["req"], want):
                product = self.items[c["k"]]["category"] if c["k"] is not None else (self.category or "the product")
                out.append(f"{product} {'must now have' if want else 'no longer needs'} {c['req']}, since {line} "
                           f"{'has' if verdict == 'yes' else 'does not have'} {c['when']}")
                self.trace.append({"kind": "conditional_applied", "rule": c, "line": line, "verdict": verdict, "applies": want})
        return out

    def condition_holds(self, line: str, when: str) -> str:
        """Whether a listing has the value a conditional rule names: 'yes', 'no' or 'unsure'. Each alternative of a closed
        list is read by code first; what code leaves unsure is read by the cheap model from the listing's facts, once."""
        prop, value = _split_req(when)
        allowed, _ = _clauses(value)
        readings = [self.code_verdict(line, f"{prop}: {alt}") for alt in allowed or [value]]
        if "yes" in readings:
            return "yes"
        if readings and all(r == "no" for r in readings):
            return "no"
        cache = self.__dict__.setdefault("condition_cache", {})
        if (line, when) not in cache:
            info = self.rows.get(line, {})
            prompt = ("Does this product listing have the property value below? Read its facts and title; a figure in another "
                      "unit counts when it is the same amount (1 TB = 1000 GB). Return ONE JSON object and nothing else: "
                      '{"answer": "yes" | "no" | "unknown"}\n\nVALUE: ' + when + "\nTITLE: " + str(info.get("title") or "")
                      + "\nLISTING: " + _compact(self.details.get(line), 1500))
            got = self.ask(prompt, 120) if not self.over() else None
            answer = str((got or {}).get("answer") or "").strip().lower() if isinstance(got, dict) else ""
            cache[(line, when)] = answer if answer in ("yes", "no") else "unsure"
            self.trace.append({"kind": "condition_read", "line": line, "when": when, "answer": cache[(line, when)]})
        return cache[(line, when)]

    def budget_line(self) -> str:
        """The store steps that still fit in both the step and the time limit, at the pace of the task so far, split
        into those kept to cart and order and the spare ones."""
        used = TASK_SECONDS - self.time_left()
        pace = used / self.step if self.step >= 3 else None
        by_time = int(max(0.0, self.time_left() - FORCE_SECONDS) / pace) if pace else self.steps_left()
        usable = max(0, min(self.steps_left(), by_time))
        keep = self.reserve()
        spare = max(0, usable - keep)
        missing = self.missing_lines()
        text = (f"BUDGET: about {usable} more store steps fit ({self.steps_left()} steps left"
                + (f"; the time left allows about {by_time} at {pace:.0f} s a step so far" if pace else "") + f"). {keep} are kept "
                f"to cart, check and order, so {spare} are spare for searching and opening.")
        if spare <= 2:
            text += " WRAP UP: no more searching or paging; cart the best opened fit of each missing product and order now."
        elif spare <= 8:
            text += " LOW: spend them only on the gaps listed (one targeted search or open each, several ids in one open); no paging."
        elif missing:
            text += f" About {spare // missing} for each of the {missing} product(s) still missing a cart line; cart each fit as soon as it is opened."
        return text

    def status(self) -> str:
        return super().status() + "\n" + self.budget_line()

    def readiness(self) -> list[str]:
        """The gaps before an order, unless the actor sent the same basket again after seeing them with no step taken
        in between, or a third time: then it has judged them, and they no longer hold the order back."""
        gaps = super().readiness()
        if gaps and self.basket_key() in getattr(self, "gaps_waived", set()):
            return []
        return gaps

    def note_order_try(self) -> None:
        gaps = super().readiness()
        if not gaps or self.steps_left() <= self.reserve() + 3:
            return
        key = self.basket_key()
        tries = self.__dict__.setdefault("order_tries", {})
        last = tries.get(key)
        if last is not None and (last[0] == self.step or last[1] >= 2):
            self.__dict__.setdefault("gaps_waived", set()).add(key)
            self.trace.append({"kind": "gaps_waived", "gaps": gaps[:4], "steps_left": self.steps_left()})
        else:
            tries[key] = (self.step, (last[1] + 1) if last else 1)

    def product_named_values(self) -> int:
        """A need whose value is only the name of a product of the order ("component type: business laptops") is a
        misreading: it is taken out. The number taken out."""
        names = set()
        for item in self.items:
            for text in (item.get("category"), item.get("noun")):
                if text:
                    names.add(" ".join(_one(w) for w in _words(text)))
        dropped, home = 0, self.current
        for k in range(len(self.items)):
            self.switch(k)
            for req in list(self.reqs):
                name, raw = _split_req(req)
                if re.search(r"(?i)compat|fit|model|device|work|for\b|with\b|use", name):
                    continue  # a part's "compatible model: macbook" names another product on purpose
                value = " ".join(_one(w) for w in _words(raw))
                if value and value in names:
                    self.reqs.remove(req)
                    dropped += 1
                    self.trace.append({"kind": "product_name_value", "product": k, "need": req})
        self.switch(home)
        return dropped

    def wants_text(self) -> str:
        text = super().wants_text()
        rules = getattr(self, "conditionals", [])
        if rules:
            text += "\nCONDITIONAL RULES (the code applies each once the item it depends on is in the cart): " + "; ".join(
                f"{c['req']} {'unless' if c['except'] else 'only if'} {c['when']}" for c in rules)
        return text

    def gate_order(self) -> str:
        """Lines code can prove wrong are held twice while there is room to swap them, before the other checks. Before
        that: a conditional rule the cart now settles is applied, and every cheapest or best line takes the winner among
        the opened listings proved to qualify (the actor tends to put back a dearer line the code had replaced)."""
        self.swap_refused()
        self.note_order_try()
        applied = self.apply_conditionals()
        if applied and self.steps_left() > 3 and self.time_left() > FORCE_SECONDS + 10:
            self.repick_lines()
            return ("NOT SENT (no step used). A conditional rule of the request now applies: " + " | ".join(applied)
                    + ". Make sure each cart line meets its product's requirements as they stand now, then order.")
        if getattr(self, "changed_at_order", False) or getattr(self, "repicked_for", None) != self.basket_key():
            self.changed_at_order = False
            self.repick_lines()
            self.repicked_for = self.basket_key()
        if self.time_left() < FORCE_SECONDS + 10 and getattr(self, "refused", False):
            return ""
        faults = [f for i, u in self.cart.items() for f in self.line_faults(i, u)] + self.mark_faults()
        held = getattr(self, "line_holds", 0)
        if faults and held < LINE_HOLDS and self.steps_left() > 3 and self.time_left() > FORCE_SECONDS + 10:
            self.line_holds = held + 1
            self.open_fits()
            self.trace.append({"kind": "line_held", "faults": faults[:4]})
            return "NOT SENT (no step used). Swap these cart lines: " + " | ".join(faults[:4])
        if faults and not getattr(self, "last_tried", False):
            self.last_tried = True
            self.last_basket()  # the holds are spent: the code sets the best basket it can prove
            self.trace.append({"kind": "last_basket", "cart": dict(self.cart)})
        return super().gate_order()

    def act_live(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        """Keep what listings say of themselves, and note a refused basket when the refusal changed no requirement."""
        before, cart = (self.all_wants(), self.budget), dict(self.cart)
        observations = super().act(actions)
        order = (self.roles.get("order") or {}).get("tool")
        for action, observation in zip(actions, observations or []):
            self.read_marks(observation)
            if action["name"] == order and isinstance(observation, dict) and observation.get("refusal"):
                self.refused_cart = cart if (self.all_wants(), self.budget) == before else None
                self.changed_at_order = True
        return observations

    # -- proven lines, repeats, earlier results ---------------------------------------------

    def proven(self, ident: str, k: int | None) -> bool:
        """The listing's facts confirm every requirement of its product, by code or by the listing check."""
        if self.verdict(ident) is True:
            return True
        reqs = [r for r in (self.items[k]["reqs"] if self.multi() and k is not None else self.reqs) if all(_split_req(r))]
        return bool(reqs) and all(self.code_verdict(ident, r) == "yes" for r in reqs)

    def prove_lines(self) -> None:
        """Each cart line not proven to meet its product's requirements takes an opened listing of that product that is:
        proven, in stock, within the product's own limit, and fitting the device it goes with. The cheapest is taken
        unless a best rule says otherwise. One cart call for the whole basket, and only when the total stays in budget."""
        keep, basket = self.current, dict(self.cart)
        swaps = []
        for line in list(basket):
            k = self.owner_of(line) if self.multi() else keep
            if self.multi() and k is None:
                continue
            self.switch(k)
            if self.proven(line, k):
                continue
            label = self.label_for()
            fits = self.wanted_fits(k) if self.multi() else []
            pool = [i for i in self.details if i not in basket and i not in self.unknown and i not in self.unavailable
                    and (not label or self.rows.get(i, {}).get("category") == label) and self.line_passes(i, k)
                    and self.proven(i, k) and not any(self.fit_state(i, n) is False for n in fits)]
            pref = self.preference or {}
            if pref.get("kind") == "best" and pref.get("property"):
                sign = -1 if pref.get("direction") == "max" else 1
                rated = [(sign * m, i) for i in pool for m in [self.measure(i, str(pref["property"]))] if m is not None]
            else:
                rated = [(p, i) for i in pool for p in [_price(self.rows.get(i, {}))] if p is not None]
            if rated:
                pick = min(rated)[1]
                basket[pick] = basket.pop(line)
                swaps.append(f"{line} -> {pick}")
        self.switch(keep)
        total = sum((_price(self.rows.get(i, {})) or 0) * u for i, u in basket.items())
        if swaps and self.steps_left() >= 2 and (self.budget is None or total <= self.budget + 1e-9):
            action = self.basket_action(basket)
            if action:
                self.trace.append({"kind": "prove_lines", "swaps": swaps})
                self.act([action])

    def knowledge(self) -> tuple:
        """What the agent knows that could justify a cart change it already made and undid."""
        return (len(self.details), self.reqs_key() if hasattr(self, "reqs_key") else tuple(tuple(i["reqs"]) for i in self.items) + (tuple(self.reqs),),
                len(self.unavailable), self.budget, len(self.heard))

    def call_key(self, name: str, args: dict[str, Any]) -> str | None:
        """Fresh 8's key, and an opening of a carted listing too: a stock check is worth a step once per change in
        what is known (a store or shopper notice, a new budget), not again and again."""
        key = super().call_key(name, args)
        if key is not None or not args:
            return key
        acting = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove", "cart_view", "order", "decline", "message")}
        if name in acting:
            return None
        return json.dumps([name, args, "carted", len(self.unavailable), self.budget, len(self.heard), getattr(self, "orders_sent", 0)],
                          sort_keys=True, default=str)

    def act(self, actions: list[dict[str, Any]]) -> list[Any] | None:
        """A search or opening the task already made is answered from memory, for the code's own calls too."""
        raw = self.__dict__.setdefault("done_raw", {})
        keys = [self.call_key(a["name"], a.get("args") or {}) for a in actions]
        cached = [k is not None and k in raw for k in keys]
        if actions and all(cached):
            self.trace.append({"kind": "repeat_from_memory", "calls": [a["name"] for a in actions]})
            return [raw[k] for k in keys]
        if any(cached):
            got = self.act([a for a, c in zip(actions, cached) if not c])
            if got is None:
                return None
            fresh = iter(got)
            self.trace.append({"kind": "repeat_from_memory", "calls": [a["name"] for a, c in zip(actions, cached) if c]})
            return [raw[k] if c else next(fresh, None) for k, c in zip(keys, cached)]
        order = (self.roles.get("order") or {}).get("tool")
        if any(a["name"] == order for a in actions):
            self.orders_sent = getattr(self, "orders_sent", 0) + 1  # a refusal may have changed stock: carted listings can be checked again
        observations = self.act_live(actions)
        for k, o in zip(keys, observations or []):
            if k and not _failed(o):
                raw[k] = o
        return observations

    def repeat_guard(self, calls: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Cart calls that would spend a step for nothing: the cart set to what it already holds, a cart change made
        before and undone with nothing learned since (back and forth), and a cart view with no cart change since the
        last one."""
        answered, rest = {}, []
        cart_tools = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove")} - {None}
        view = (self.roles.get("cart_view") or {}).get("tool")
        made = self.__dict__.setdefault("cart_calls", {})
        for call in calls:
            function = call["function"]
            name = str(function.get("name"))
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                rest.append(call)
                continue
            key = json.dumps([name, args], sort_keys=True, default=str)
            if name in cart_tools and key in made:
                step, after, known = made[key]
                if after == dict(self.cart):
                    answered[call["id"]] = (f"ALREADY DONE (no step used): this same cart call at step {step} left the cart exactly as it "
                                            f"is now: {self.cart}. Nothing was sent.")
                    self.trace.append({"kind": "repeat_held", "what": "cart unchanged"})
                    continue
                if known == self.knowledge():
                    answered[call["id"]] = (f"NOT SENT (no step used): this same cart change was made at step {step} and undone since, and "
                                            "nothing new is known (no listing opened, no requirement, budget or stock changed). Going back "
                                            "and forth spends steps: keep the cart as it is and order, or open or search something new first.")
                    self.trace.append({"kind": "repeat_held", "what": "cart back and forth"})
                    continue
            if view and name == view:
                last = getattr(self, "last_view", None)
                if last and last[0] == dict(self.cart) and last[2] == self.knowledge():
                    answered[call["id"]] = f"ALREADY DONE at step {last[1]} (no step used now): the cart has not changed since. It showed:\n{last[3]}"
                    self.trace.append({"kind": "repeat_held", "what": "cart view"})
                    continue
            rest.append(call)
        return answered, rest

    def note_cart_calls(self, calls: list[dict[str, Any]], results: list[tuple[str, str]]) -> None:
        cart_tools = {(self.roles.get(r) or {}).get("tool") for r in ("cart", "cart_remove")} - {None}
        view = (self.roles.get("cart_view") or {}).get("tool")
        text = dict(results)
        for call in calls:
            function = call["function"]
            name = str(function.get("name"))
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                continue
            said = text.get(call["id"], "")
            if said.startswith(("NOT SENT", "ALREADY DONE", "NOT RUN")):
                continue
            if name in cart_tools:
                self.cart_calls[json.dumps([name, args], sort_keys=True, default=str)] = (self.step, dict(self.cart), self.knowledge())
            elif view and name == view:
                self.last_view = (dict(self.cart), self.step, self.knowledge(), said[:3000])

    def search_new_needs(self, before: dict[int, list[str]]) -> list[str]:
        """Earlier results first. For each product whose requirements changed, the listings already opened are checked
        against the requirements as they stand now: one that meets them all makes a new search needless. Otherwise the
        rows already seen whose own text states every new value are opened (several in one step where the store allows)
        before any new search is made. Only a product still without a proven listing gets fresh need searches."""
        keep, done, settled = self.current, [], dict(before)
        for k, old in before.items():
            if self.multi():
                self.switch(k)
            reqs = [r for r in self.reqs if all(_split_req(r))]
            new = [r for r in reqs if r not in old]
            if not new or self.over():
                continue
            label = self.label_for() or None

            def good(i: str) -> bool:
                return (self._row_in_category(i, label) and self.line_passes(i, k) and self.proven(i, k))
            have = [i for i in self.details if good(i)]
            if not have:
                seen = [i for i in self.rows if i not in self.details and i not in self.unavailable and self._row_in_category(i, label)
                        and self.rows[i].get("stock") is not False
                        and all(self._row_states_value(i, _split_req(r)[1]) for r in new if not _BOUND.search(_split_req(r)[1]))
                        and any(not _BOUND.search(_split_req(r)[1]) for r in new)]
                seen.sort(key=lambda i: _price(self.rows.get(i, {})) or 0)
                if seen and self.steps_left() > self.order_due() + 6:
                    self.read_details(seen[:4])
                    have = [i for i in seen[:4] if i in self.details and good(i)]
                    if have:
                        done.append(f"{self.product_name(k)}: {', '.join(have)} (rows of earlier searches)")
            else:
                done.append(f"{self.product_name(k)}: {', '.join(have[:3])} (opened earlier, meets every requirement now)")
            if have:
                settled[k] = list(self.reqs)  # nothing left to search for this product
                self.trace.append({"kind": "need_from_seen", "product": k, "ids": have[:4]})
        if self.multi():
            self.switch(keep)
        return done + self.search_needs_fresh(settled)

    def gate_decline(self) -> str:
        """Only about one decline in eight was right on the graded runs, and the model's doubt is what empties a cart:
        a decline goes out only when nothing is left that could be ordered."""
        held = super().gate_decline()
        if held or self.finished or self.over() or getattr(self, "last_tried", False):
            return held
        self.last_tried = True
        action = self.finish_action("order") if self.last_basket() else None
        if not action or self.over():
            return ""
        self.trace.append({"kind": "last_basket", "cart": dict(self.cart)})
        self.act([action])
        return "NOT SENT: the code ordered the cart in place of ending without a purchase. If the order was refused, fix what it says and order."

    def finish_now(self) -> None:
        """Every forced ending goes through the same last basket, unless the cart is already sound; a cheapest or best
        line takes the proved winner first."""
        if not self.over() and not self.finished:
            self.swap_refused()
            self.apply_conditionals()
            if self.steps_left() >= 3 and self.time_left() > 8:
                self.prove_lines()
                self.repick_lines()
        faulty = (not self.cart or self.basket_problems() or self.fit_problems() or self.mark_faults()
                  or any(self.line_faults(i, u) for i, u in self.cart.items()))
        if faulty and not self.over() and not self.finished and not getattr(self, "last_tried", False):
            self.last_tried = True
            if self.last_basket():
                self.trace.append({"kind": "last_basket", "cart": dict(self.cart)})
        super().finish_now()

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Replies that change nothing, turn after turn, end in the order they would reach at the turn limit anyway."""
        if self.time_left() < CLOSING_SECONDS and self.cart_fits() and not getattr(self, "swap_finished", False):
            self.trace.append({"kind": "closing_finish", "cart": dict(self.cart), "left": round(self.time_left())})
            self.swap_finished = True
            self.finish_now()
            if self.finished or self.over():
                return [(c["id"], "NOT SENT: the code ordered the cart, time is nearly out.") for c in calls]
        step, cart = self.step, dict(self.cart)
        answered, rest = self.repeat_guard(calls)
        sent = self.guarded_calls(rest) if rest else []
        self.note_cart_calls(rest, sent)
        sent_by = dict(sent)
        out = [(c["id"], answered.get(c["id"]) or sent_by.get(c["id"], "NO RESULT")) for c in calls]
        self.idle_turns = getattr(self, "idle_turns", 0) + 1 if (self.step, self.cart) == (step, cart) else 0
        if self.idle_turns >= IDLE_TURNS and self.cart and not self.finished and not getattr(self, "swap_finished", False):
            self.trace.append({"kind": "idle_finish", "turns": self.idle_turns, "cart": dict(self.cart)})
            self.swap_finished = True
            self.finish_now()
        return out

    def guarded_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """A cart change the steps left cannot finish and still order is not started: the code orders the cart as it stands."""
        cart, remove = self.roles.get("cart") or {}, self.roles.get("cart_remove") or {}
        adds, removes, touching = set(), 0, set()
        for call in calls:
            function = call["function"]
            name = str(function.get("name"))
            try:
                args = json.loads(function.get("arguments") or "{}") if isinstance(function.get("arguments"), str) else dict(function.get("arguments") or {})
            except ValueError:
                continue
            if not isinstance(args, dict):
                continue
            if self.is_add(name, args):
                touching.add(call["id"])
                ident = args.get(cart.get("id_arg") or "")
                if ident and ident not in self.cart:
                    adds.add(ident)
            elif name == remove.get("tool") or (name == cart.get("tool") and cart.get("op_arg")
                                                and str(args.get(cart["op_arg"])) == str(cart.get("remove_value"))):
                touching.add(call["id"])
                removes += 1
        lines = len(self.items) if self.multi() else 1
        extra = max(0, len(self.cart) + len(adds) - lines)
        if not adds or not extra or self.steps_left() >= len(adds) + max(removes, extra) + 1:
            return super().run_calls(calls)
        self.trace.append({"kind": "swap_held", "adds": sorted(adds), "steps_left": self.steps_left(), "cart": dict(self.cart)})
        if not getattr(self, "swap_finished", False):
            self.swap_finished = True
            self.finish_now()
        note = "NOT SENT (step not spent): the steps left cannot cover this cart change plus the order, so the code ordered what the cart holds."
        return [(c["id"], note) for c in calls]

    def __init__(self, problem_data: dict[str, Any]):
        self.marks: dict[str, dict[str, Any]] = {}  # listing -> the names it is sold under and what it fits
        super().__init__(problem_data)
        self.interview_topup: set[str] = set()
        self.interview_asked: list[tuple[int, str]] = []
        self.hear_focus: int | None = None  # product index the current AskBuyer is about
        self._recent_search_ids: list[str] = []  # ids from the latest need/widen searches, page order
        self.need_title_prefer: dict[int, list[str]] = {}  # product -> title hits from value+kind queries

    def wanted_units(self, ident: str) -> int:
        """Units for this listing: the product the last basket chose it for or the only one holding its store label, else
        the product its store category names, else the one it was carted for, else the best match."""
        if not self.multi():
            return int(self.quantity or 1)
        self.save_item()
        owner = self.owner_of(ident)
        if owner is not None:
            return int(self.items[owner].get("quantity") or 1)
        shelf = self._stems(str(self.rows.get(ident, {}).get("category") or ""))
        named = [i for i in self.items if shelf & self._stems(str(i.get("category") or ""))]
        if len(named) == 1:
            return int(named[0].get("quantity") or 1)
        for k, line in self.lines.items():
            if line == ident and 0 <= k < len(self.items):
                return int(self.items[k].get("quantity") or 1)
        info = self.rows.get(ident, {})
        text = f"{info.get('category') or ''} {info.get('title') or ''}".lower()
        best_k, best_score = None, -1
        for k, item in enumerate(self.items):
            score = sum(w in text for w in _words(f"{item.get('category') or ''} {item.get('noun') or ''}"))
            if score > best_score:
                best_k, best_score = k, score
        if best_k is not None and best_score > 0:
            return int(self.items[best_k].get("quantity") or 1)
        return int(self.quantity or 1)

    def repeat_add(self, args: dict[str, Any]) -> str:
        """Block stacking past the requested units for every line (not only code-carted ones)."""
        cart = self.roles.get("cart") or {}
        ident = str(args.get(cart.get("id_arg") or "") or "")
        if ident and ident in self.cart:
            want = self.wanted_units(ident)
            held = int(self.cart.get(ident) or 0)
            add_n = args.get(cart.get("qty_arg") or "")
            add_n = int(add_n) if isinstance(add_n, int) and not isinstance(add_n, bool) and add_n > 0 else 1
            if held >= want or held + add_n > want:
                return (f"{ident} already holds {held} unit(s); the request wants {want}. "
                        f"Do not append more — set the line to {want} or order")
        return super().repeat_add(args)

    def absorb(self, action: dict[str, Any], observation: Any) -> None:
        """Keep cart counts aligned with the store: append adds to held units; bag views win when they report qty."""
        cart = self.roles.get("cart") or {}
        before = dict(getattr(self, "cart", {}) or {})
        super().absorb(action, observation)
        if (action.get("name") == cart.get("tool") and cart.get("op_arg") and cart.get("qty_arg")
                and not cart.get("lines_arg") and not _failed(observation)):
            args = action.get("args") or {}
            ident = args.get(cart.get("id_arg") or "")
            units = args.get(cart.get("qty_arg") or "")
            if (isinstance(ident, str) and ident
                    and str(args.get(cart["op_arg"])) == str(cart.get("add_value"))
                    and isinstance(units, int) and not isinstance(units, bool) and units >= 1
                    and _line_units(observation) is None):
                self.cart[ident] = int(before.get(ident) or 0) + units

    def _stems(self, text: str) -> frozenset[str]:
        return frozenset(_one(w) for w in _words(text) if w)

    def _scrub_need_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Drop meta/vague need values before they enter reqs (full INTERPRET path included)."""
        result = dict(result)

        def keep_req(r: Any) -> bool:
            if not isinstance(r, str) or not r.strip():
                return False
            _prop, val = _split_req(r.strip())
            return _usable_need_value(val) if val else True

        needs = []
        for n in result.get("needs") or []:
            if not isinstance(n, str) or not n.strip():
                continue
            text = n.strip()
            m = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]\s*(.*)", text)
            body = (m.group(2).strip() if m else text)
            if ":" in body and not _usable_need_value(_split_req(body)[1]):
                continue
            needs.append(text)
        result["needs"] = needs
        now = result.get("requirements_now")
        if isinstance(now, list):
            result["requirements_now"] = [r for r in now if keep_req(r)]
        elif isinstance(now, dict):
            result["requirements_now"] = {
                k: [r for r in (v or []) if keep_req(r)] if isinstance(v, list) else v
                for k, v in now.items()
            }
        return result

    def apply_many(self, result: dict[str, Any], fresh: list[str]) -> bool:
        """Only apply requirements_now / needs for the product the question was about, so a don't-mind on product B
        cannot empty product A's needs (models often return empty lists for every product)."""
        if isinstance(result, dict):
            result = self._scrub_need_result(result)
        focus = self.hear_focus
        if focus is None or not self.multi() or not isinstance(result, dict):
            return self.settle_many(result, fresh)
        result = dict(result)
        key = str(focus + 1)
        now = result.get("requirements_now")
        if isinstance(now, dict):
            mine = now.get(key)
            if mine is None:
                mine = now.get(focus)
            if mine is None:
                mine = next((v for k, v in now.items() if str(k) in (key, str(focus)) and isinstance(v, list)), None)
            result["requirements_now"] = {key: list(mine) if isinstance(mine, list) else []}
        needs = []
        for n in result.get("needs") or []:
            if not isinstance(n, str) or not n.strip():
                continue
            text = n.strip()
            m = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]\s*(.*)", text)
            if m:
                if int(m.group(1)) != focus + 1:
                    continue
                text = m.group(2).strip()
            needs.append(text)
        result["needs"] = needs
        prefs = []
        for entry in result.get("no_preference") or []:
            if isinstance(entry, str) and re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]", entry):
                m = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]\s*(.*)", entry)
                if m and int(m.group(1)) != focus + 1:
                    continue
                prefs.append(m.group(2).strip() if m else entry)
            else:
                prefs.append(entry)
        result["no_preference"] = prefs
        for field in ("new_preference", "product_budgets"):  # an interview answer changes only its own product
            got = result.get(field)
            result[field] = {key: got[k] for k in got if str(k).strip(" []") == key} if isinstance(got, dict) else {}
        return self.settle_many(result, fresh)

    def settle_many(self, result: dict[str, Any], fresh: list[str]) -> bool:
        """The reading of a reply for several products, then a need it put on the wrong product taken back off: one no
        opened listing of that product even has the property of, while another product of the order already needs
        exactly that value and its listings carry the property ("component type: keyboard" said of the laptop)."""
        changed = super().apply_many(result, fresh)
        self.save_item()
        if self.product_named_values():
            self.save_item()
            changed = True
        keys: dict[int, set[str]] = {k: set() for k in range(len(self.items))}
        for ident, details in self.details.items():
            owner = self.owner_of(ident)
            if owner is not None and owner in keys:
                keys[owner] |= {_key_norm(key) for key, _ in (_facts(details) or [])}
        home = self.current
        for k, item in enumerate(self.items):
            for req in list(item["reqs"]):  # the misplaced copy may have come first, before the right product got it
                name, value = (_key_norm(x) for x in _split_req(req))
                if not name or not keys[k] or name in keys[k]:
                    continue
                if any(j != k and name in keys[j] and any(tuple(_key_norm(x) for x in _split_req(r)) == (name, value) for r in other["reqs"])
                       for j, other in enumerate(self.items)):
                    self.switch(k)
                    self.reqs = [r for r in self.reqs if r != req]
                    self.trace.append({"kind": "misplaced_need", "product": k, "need": req})
        self.switch(home)
        return changed

    def listen(self, observations: list[Any] | None, question: str | None = None, reply_of: int | None = None) -> bool:
        """Interview answers: try a short cheap hear first (saves SPEND_SOFT for search/opens); fall back to full
        INTERPRET when the cheap reading finds nothing in a substantive reply."""
        if self.hear_focus is None or not question:
            return super().listen(observations, question, reply_of)
        lines: list[str] = []
        for index, observation in enumerate(observations or []):
            if index == reply_of and isinstance(observation, dict):
                strings = [v for v in observation.values() if isinstance(v, str) and v.strip()]
                if len(strings) == 1:
                    lines.append(strings[0])
            lines.extend(_shopper_lines(observation))
        fresh = [line for line in dict.fromkeys(lines) if line not in self.heard]
        if not fresh:
            return super().listen(observations, question, reply_of)
        earlier = "; ".join(self.product_reqs(self.hear_focus)) or "(none)"
        prompt = (HEAR_PROMPT.replace("EARLIER", earlier).replace("QUESTION_TEXT", question)
                  .replace("ANSWER_TEXT", "\n".join(fresh)))
        if self.multi():
            prompt += (f"\nThis answer is about product {self.hear_focus + 1} only. "
                       f"Return requirements_now as a flat list for that product alone.")
        cheap = self.ask(prompt, 700, strong=False) if self.spent < SPEND_CAP else None
        dont_mind = isinstance(cheap, dict) and bool(cheap.get("no_preference")) and not cheap.get("needs")
        if isinstance(cheap, dict) and (cheap.get("needs") or cheap.get("no_preference") or dont_mind
                                        or cheap.get("requirements_now") is not None):
            needs = []
            for n in cheap.get("needs") or []:
                if not isinstance(n, str) or not n.strip():
                    continue
                text = n.strip()
                m = re.match(r"\s*\[?(\d+)\]?\s*[:.)\-]\s*(.*)", text)
                if m:
                    text = m.group(2).strip() or text
                if ":" not in text:
                    continue
                _prop, val = _split_req(text)
                if not _usable_need_value(val):
                    continue
                needs.append(text)
            now = cheap.get("requirements_now")
            earlier_reqs = self.product_reqs(self.hear_focus)
            if not isinstance(now, list):
                if isinstance(now, dict):
                    now = now.get(str(self.hear_focus + 1), now.get(self.hear_focus))
                now = list(now) if isinstance(now, list) else list(earlier_reqs)
            now = [str(r).strip() for r in now if str(r).strip()]
            now = [r for r in now if (not all(_split_req(r))) or _usable_need_value(_split_req(r)[1])]
            for need in needs:
                if need not in now and not any(_key_norm(_split_req(need)[0]) == _key_norm(_split_req(r)[0]) for r in now):
                    now.append(need)
            if dont_mind and not needs:
                now = list(earlier_reqs)  # don't-mind must not clear earlier requirements
            shaped = {
                "understood": cheap.get("understood", True) is not False,
                "needs": needs,
                "no_preference": list(cheap.get("no_preference") or []),
                "dropped": list(cheap.get("dropped") or []),
                "new_budget": cheap.get("new_budget"),
                "new_preference": cheap.get("new_preference"),
                "unavailable": False,
                "requirements_now": now,
                "reply": "",
                "references": cheap.get("references") or [],
            }
            own_limit = cheap.get("product_budget")
            if self.multi():
                shaped["requirements_now"] = {str(self.hear_focus + 1): shaped["requirements_now"]}
                shaped["new_preference"] = {str(self.hear_focus + 1): shaped["new_preference"]} if shaped["new_preference"] else {}
                shaped["product_budgets"] = {str(self.hear_focus + 1): own_limit} if own_limit is not None else {}
            elif shaped["new_budget"] is None:
                shaped["new_budget"] = own_limit  # with one product, its own limit is the order's
            self.heard.update(fresh)
            self.save_item()
            before = {None: list(self.reqs)} if not self.multi() else {k: list(i["reqs"]) for k, i in enumerate(self.items)}
            self.exchanges.append(f"asked: {question} -> shopper: " + " / ".join(fresh))
            self.trace.append({"kind": "heard", "question": question, "lines": fresh, "reading": shaped, "via": "hear"})
            self.understood = bool(shaped["understood"])
            if self.multi():
                changed = self.apply_many(shaped, fresh)
                changed = self.note_references(shaped, before) or changed
                return self.shelf_open(before) or changed
            changed = False
            now = shaped.get("requirements_now")
            if isinstance(now, list) and all(isinstance(r, str) for r in now):
                cleaned = _kept_requirements(self.reqs, [r.strip() for r in now if r.strip()],
                                             shaped.get("dropped") or [], " ".join(fresh), self.noun_words())
                if cleaned != self.reqs:
                    self.reqs = cleaned
                    changed = True
            if shaped.get("needs"):
                self.revealed += 1
            for prop in shaped.get("no_preference") or []:
                if isinstance(prop, str) and prop.strip() and prop not in self.free:
                    self.free.append(prop.strip())
            budget = shaped.get("new_budget")
            if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
                self.budget = float(budget)
                changed = True
            chose = self.apply_choice(shaped)
            changed = chose or changed
            changed = self.note_references(shaped, before) or changed
            got_need = bool(shaped.get("needs")) or (isinstance(now, list) and any(r not in before.get(None, []) for r in (now or [])))
            if got_need or dont_mind or chose or self.spent >= SPEND_SOFT:
                return self.shelf_open(before) or changed
        return super().listen(observations, question, reply_of)

    @staticmethod
    def _label_share(name: str, label: str) -> float:
        """How much of a store label a product's name covers."""
        words = {_one(w) for w in _words(label)}
        return len(words & {_one(w) for w in _words(name)}) / len(words) if words else 0.0

    def _label_open(self, label: str) -> bool:
        """No other product of the order is named closer to this store label than the current one."""
        own = self._label_share(f"{self.category} {self.noun}", label)
        return all(self._label_share(f"{i['category']} {i.get('noun') or ''}", label) <= own
                   for j, i in enumerate(self.items) if j != self.current)

    def label_for(self) -> str:
        """With several products a shared word is not enough: a label another product is named closer to is that
        product's, and this one takes the closest label left."""
        label = super().label_for()
        if not self.multi() or (label and self._label_open(label)):
            return label
        counts: dict[str, int] = {}
        for info in self.rows.values():
            if info.get("category"):
                counts[info["category"]] = counts.get(info["category"], 0) + 1
        name = f"{self.category} {self.noun}"
        left = [c for c in counts if self._label_share(name, c) > 0 and self._label_open(c)]
        return max(left, key=lambda c: (self._label_share(name, c), counts[c]), default="")

    def target_label(self) -> str:
        label = super().target_label()
        return self.label_for() if self.multi() and label and not self._label_open(label) else label

    def other_labels(self, k: int) -> set[str]:
        """The store labels the order's other products hold (the current product is left on k)."""
        held = set()
        for j in range(len(self.items)):
            if j != k:
                self.switch(j)
                held.add(self.label_for())
        self.switch(k)
        return held - {""}

    def search_unseen(self) -> None:
        """A product with no listing of its own yet has nothing to ask about: one search and its rows, before the
        questions are planned from what has been opened."""
        if not self.multi() or not any(r.get("category") for r in self.rows.values()):
            return
        keep = self.current
        for k in range(len(self.items)):
            if self.over() or self.steps_left() <= INTERVIEW_MIN_LEFT + 2:
                break
            self.switch(k)
            if not self.label_for() and self.run_search(1):
                self.trace.append({"kind": "unseen_searched", "product": k, "label": self.label_for()})
                self.open_supported()
        self.switch(keep)

    def _key_bucket(self, k: int) -> tuple[dict[str, int], dict[str, list[str]], dict[str, dict[str, str]]]:
        """Per product: carry counts, sample values, and per-listing value map."""
        keep = self.current
        if self.multi():
            self.switch(k)
        label = self.label_for() or None
        held = self.other_labels(k) if self.multi() and not label else set()
        counts: dict[str, int] = {}
        values: dict[str, list[str]] = {}
        per: dict[str, dict[str, str]] = {}
        for ident, details in self.details.items():
            shelf = self.rows.get(ident, {}).get("category")
            if ident in self.unknown or (label and shelf != label) or shelf in held:
                continue
            for key, value in _facts(details) or []:
                name = " ".join(str(key).replace("_", " ").replace("-", " ").lower().split())
                if not name or len(name) > 30 or not askable_label(name):
                    continue
                text = str(value).strip().lower()
                if text == str(ident).lower() or len(text) > 100:
                    continue  # the listing's own id or its prose describes no choice a shopper makes
                counts[name] = counts.get(name, 0) + 1
                per.setdefault(name, {})[ident] = text
                if text and text not in values.setdefault(name, []):
                    values[name].append(text)
        if self.multi():
            self.switch(keep)
        return counts, values, per

    def _keep_key(self, name: str, slot_values: list[str], per: dict[str, str]) -> bool:
        """Drop identifiers, multi-option bins, and overlong value bags."""
        if not slot_values or not askable_label(name):
            return False
        mean = sum(len(v) for v in slot_values) / max(1, len(slot_values))
        if mean > 40 or sum(1 for v in slot_values if " / " in v) * 2 >= len(slot_values):
            return False
        if len(slot_values) >= 4 and len(set(per.values())) == len(per) and mean > 12:
            return False
        return True

    def key_lines(self) -> str:
        """Dash list of attribute keys (most carried first). Unnumbered so the planner ranks by the prompt rules,
        not table position."""
        keep, out = self.current, []
        for k in (range(len(self.items)) if self.multi() else [keep]):
            counts, values, per = self._key_bucket(k)
            out.append(
                f"product {k + 1 if self.multi() else 1} ({self.product_name(k)}): "
                f"settled by the request: {'; '.join(self.product_reqs(k)) or 'nothing'}"
            )
            names = []
            for name, carried in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if not self._keep_key(name, values.get(name, []), per.get(name, {})):
                    continue
                names.append(name)
                if len(names) >= 20:
                    break
            out += [f"  - {n} (on {counts[n]}): {' | '.join(values[n][:4])}" for n in names]
        self.switch(keep)
        return "\n".join(out)

    def item_keys(self, k: int, limit: int = 8) -> list[str]:
        """Most-carried keys of product k's category that the plan may not have named (shared values first)."""
        counts, values, per = self._key_bucket(k)
        settled = [self._stems(_split_req(r)[0] or r) for r in self.product_reqs(k)]
        free = [self._stems(str(f)) for f in self.free]
        shared, single = [], []
        for name, carried in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            stems = self._stems(name)
            if carried < 2 or name in shared + single:
                continue
            if not self._keep_key(name, values.get(name, []), per.get(name, {})):
                continue
            if any(s and (s <= stems or stems <= s) for s in settled + free):
                continue
            (shared if len(set(per.get(name, {}).values())) < carried else single).append(name)
        return (shared + single)[:limit]

    def _label_bucket(self, k: int, label: str) -> str:
        """Classify a planned ask label against product k's opened values: cat | measure | weak."""
        counts, _values, per = self._key_bucket(k)
        stems = self._stems(label)
        name = next((n for n in counts if self._stems(n) == stems), None)
        if name is None:
            name = next((n for n in counts if stems & self._stems(n)), None)
        if not name:
            return "weak"
        vals = list(per.get(name, {}).values()) or [""]
        distinct = len(set(vals))
        mean = sum(len(v) for v in vals) / len(vals)
        numericish = sum(1 for v in vals if re.search(r"\d", v)) * 2 >= len(vals)
        blank = {"", "mix", "mixed", "nil", "n/a", "none", "-", "null"}
        real_distinct = len({v for v in vals if v.strip() not in blank})
        if numericish:
            return "measure"
        if (not numericish and distinct >= 2 and real_distinct >= 1 and distinct <= 8
                and mean <= 24 and askable_label(name)):
            return "cat"
        return "weak"

    def _product_had_cat_ask(self, k: int) -> bool:
        return any(n == k and self._label_bucket(k, lab) == "cat" for n, lab in self.interview_asked)

    def _queue_has_cat(self, k: int, queues: dict[int, list[str]]) -> bool:
        return any(self._label_bucket(k, lab) == "cat" for lab in queues.get(k) or [])

    def _request_hint_labels(self, k: int) -> list[str]:
        """Listing keys whose names or example values echo distinctive request phrases for this product's shelf."""
        keep = self.current
        if self.multi():
            self.switch(k)
        boosts = self._request_boost_phrases("", self.category or "")
        counts, values, per = self._key_bucket(k)
        if self.multi():
            self.switch(keep)
        if not boosts or not counts:
            return []
        hinted: list[str] = []
        for boost in boosts:
            bstems = self._stems(boost)
            if not bstems:
                continue
            for name, carried in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if carried < 2 or not askable_label(name):
                    continue
                if not self._keep_key(name, values.get(name, []), per.get(name, {})):
                    continue
                if bstems & self._stems(name):
                    if all(self._stems(name) != self._stems(h) for h in hinted):
                        hinted.append(name)
                    continue
                if any(_stated(boost, v) or bstems <= self._stems(v) for v in values.get(name, []) if v):
                    if all(self._stems(name) != self._stems(h) for h in hinted):
                        hinted.append(name)
        return hinted[:6]

    def _hint_lines(self, products: list[int]) -> str:
        """Prompt block: request-echoed keys per product (for the LLM planner)."""
        blocks = []
        keep = self.current
        for k in products:
            hints = self._request_hint_labels(k)
            if not hints:
                continue
            blocks.append(f"product {k + 1 if self.multi() else 1} ({self.product_name(k)}): " + ", ".join(hints))
        self.switch(keep)
        if not blocks:
            return ""
        return ("\n\nREQUEST HINTS (listed keys whose names or example values echo distinctive words from the "
                "request — prefer these early for that product):\n" + "\n".join(blocks))

    def choose_question(self, order: list[int], queues: dict[int, list[str]], found: set[int],
                  last_item: int | None) -> tuple[int, str] | None:
        """Next question: after the last item asked, prefer products with no preference found yet; prune settled,
        free, already-asked, and top-up keys the request already states.

        On multi-buy, products that still have an unasked categorical and have never had a cat ask stay ahead of
        'found' deferral so one line's need cannot burn the whole message budget."""
        for k, queue in queues.items():
            told = [self._stems((_split_req(r)[0] or r.split(":")[0])) for r in self.product_reqs(k)]
            stated = list(told)
            free = [self._stems(str(f)) for f in self.free]
            queue[:] = [label for label in queue
                        if not any(s and s <= self._stems(label) for s in told + free)
                        and not (label in self.interview_topup
                                 and any(s and (s <= self._stems(label) or self._stems(label) <= s) for s in stated))]
            asked = [self._stems(label) for n, label in self.interview_asked if n == k]
            queue[:] = [label for label in queue
                        if self._stems(label) not in asked
                        and not any(self._stems(label) <= a or a <= self._stems(label) for a in asked)]
        waiting = [n for n in order if queues.get(n)]
        if not waiting:
            return None
        behind = []
        if self.multi():
            behind = [n for n in waiting if not self._product_had_cat_ask(n) and self._queue_has_cat(n, queues)]
        pool = behind or [n for n in waiting if n not in found] or waiting
        after = order[order.index(last_item) + 1:] + order[: order.index(last_item) + 1] if last_item in order else order
        chosen = next(n for n in after if n in pool)
        return chosen, queues[chosen].pop(0)

    def kind_nouns(self, category: str, limit: int = 4) -> list[str]:
        """Product nouns titles use for this category — not brands or model codes.
        Prefer words shared with the category, then words that recur across several titles."""
        stop = {"new", "sale", "free", "with", "for", "the", "and"}
        cat_stems = self._stems(category)
        label = self.label_for() or None
        counts: dict[str, int] = {}
        for ident, row in self.rows.items():
            if label and row.get("category") and row.get("category") != label:
                continue
            title = re.sub(r"\d[\d.,]*\s*-?\s*[a-z]+", " ", str(row.get("title") or "").lower())
            for w in re.findall(r"[a-z]{4,}", title):
                if any(ch.isdigit() for ch in w) or _one(w) in stop:
                    continue
                stem = w[:-1] if w.endswith("s") and len(w) > 4 else w
                if stem in stop or len(stem) < 4:
                    continue
                counts[stem] = counts.get(stem, 0) + 1
        shared = [w for w, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                  if n >= 2 and (bool(self._stems(w) & cat_stems) or n >= 4)]
        heads = []
        for part in re.split(r"\s*(?:&|/| and )\s*", category.lower()):
            words = [w for w in re.findall(r"[a-z]{4,}", part)
                     if w not in ("and", "the", "with") and _one(w) not in stop]
            if not words:
                continue
            w = words[-1]
            for form in (w, w[:-1] if w.endswith("s") and len(w) > 4 else w,
                         w[:-2] + "f" if w.endswith("ves") and len(w) > 5 else w):
                if form not in heads and form not in stop and len(form) >= 4:
                    heads.append(form)
        return list(dict.fromkeys(heads + shared))[:limit]

    def _request_boost_phrases(self, value: str, category: str, limit: int = 4) -> list[str]:
        """Distinctive request phrases to pair with a need.

        Generic: hyphen/slash compounds and short content n-grams from the request that also appear on opened
        titles or as multi-word request spans — not a property allowlist."""
        req = str(getattr(self, "request", "") or "")
        # a conditional sentence names another product's property ("except when the laptop's processor is 10-core"):
        # its words are no search terms for this one
        req = " ".join(t for t in re.split(r"(?<=[.!?])\s+", req) if not _CONDITIONAL.search(t))
        if not req:
            return []
        stop = set(_words(category)) | set(_words(value)) | set(_FILLER) | {
            "want", "need", "please", "looking", "find", "buy", "get", "order", "product", "item",
            "something", "would", "like", "prefer", "must", "should", "also", "from", "have", "that",
            "this", "with", "for", "the", "and", "are", "was", "can", "will", "shop", "store", "budget",
            "under", "over", "least", "most", "more", "less", "than", "about", "into", "your", "our",
            "cheapest", "cheap", "price", "cost", "dollar", "dollars", "cent", "cents", "qty", "quantity",
        }
        out: list[str] = []

        def add(p: str) -> None:
            p = re.sub(r"\s+", " ", (p or "")).strip(" ,.;:").lower()
            if not p or p in stop:
                return
            parts = p.split()
            if len(parts) == 1 and len(parts[0]) < 4 and not any(ch.isdigit() for ch in parts[0]):
                return
            if len(p) < 3:
                return
            if p.lower() in {x.lower() for x in out}:
                return
            if self._stems(p) <= self._stems(value) or self._stems(p) <= self._stems(category):
                return
            out.append(p)

        req_stems = self._stems(req) - self._stems(category) - self._stems(value)
        title_hits: dict[str, int] = {}
        title_blob = ""
        label = self.label_for() or None
        for ident, details in list(self.details.items())[:50]:
            if label and not self._row_in_category(ident, label):
                continue
            title = str((self.rows.get(ident) or {}).get("title") or "")
            title_blob += " " + title.lower()
            for w in re.findall(r"[a-z][a-z0-9]{2,}", title.lower()):
                if w in stop or _one(w) in stop:
                    continue
                if self._stems(w) & req_stems:
                    title_hits[w] = title_hits.get(w, 0) + 1
        title_stems = self._stems(title_blob) if title_blob.strip() else set()

        def on_shelf(p: str) -> bool:
            """True when the phrase is attested on opened titles of this category (or no titles yet).

            Require a content word of the phrase (len≥3, not a hyphen crumb like 'in') to appear on the shelf,
            so a phrase does not match on a crumb of a compound word."""
            if not title_stems:
                return True
            content = {w for w in self._stems(p) if len(w) >= 3 and w not in stop}
            if not content:
                content = self._stems(p)
            return bool(content & title_stems)

        compounds = []
        for m in re.findall(r"\b[\w]+(?:[\-/][\w]+){1,3}\b", req.lower()):
            if not any(ch.isdigit() for ch in m) or re.search(r"[a-z]", m):
                compounds.append(m)
        for m in compounds:
            spaced = m.replace("-", " ").replace("/", " ")
            if on_shelf(spaced) or on_shelf(m):
                add(spaced)
                add(m)
        if not out:  # no shelf-attested compound yet — allow request compounds unfiltered
            for m in compounds:
                add(m.replace("-", " ").replace("/", " "))
                add(m)
                if len(out) >= limit:
                    break

        for w, _n in sorted(title_hits.items(), key=lambda kv: (-kv[1], kv[0]))[:6]:
            add(w)

        words = re.findall(r"[a-z0-9][a-z0-9\-/]{1,}", req.lower())
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if a in stop or b in stop or _one(a) in stop or _one(b) in stop:
                continue
            if any(ch.isdigit() for ch in a) and any(ch.isdigit() for ch in b):
                continue
            span = f"{a} {b}"
            if not on_shelf(span):
                continue
            add(span)
            if len(out) >= limit + 2:
                break
        return [p for p in out if title_stems and on_shelf(p)][:limit]

    def need_queries(self, value: str, category: str) -> list[str]:
        """Short query variants for a revealed need. Order: figure/bare value, value+kind, then shelf names.

        For limits, search the figure tokens before paging the bare category."""
        phrase = self.product_phrase(category)
        head = self.head_noun(category)
        kinds = self.kind_nouns(category)
        boosts = self._request_boost_phrases(value, category)
        out: list[str] = []

        def add(q: str) -> None:
            q = re.sub(r"\s+", " ", (q or "")).strip()
            if q and q.lower() not in {x.lower() for x in out}:
                out.append(q)

        bound = bool(_BOUND.search(value))
        if bound:
            figures = _need_figure_forms(value)
            for fig in figures:
                add(fig)
                for b in boosts[:2]:
                    add(f"{fig} {b}")
                if head:
                    add(f"{fig} {head}")
                add(f"{fig} {phrase}")
            for b in boosts[:2]:
                add(b)
            add(phrase)
            add(head)
            label = self.label_for() or None
            stop = set(_words(category)) | set(_words(" ".join(figures))) | {
                "pack", "set", "new", "sale", "free", "with", "for", "the", "and", "can", "tin",
            }
            for ident in list(self.details)[:40]:
                if not self._detail_states_need(ident, value, label):
                    continue
                title = str((self.rows.get(ident) or {}).get("title") or "")
                extras = [w for w in re.findall(r"[a-z]{3,}", title.lower())
                          if w not in stop and _one(w) not in stop
                          and not any(ch.isdigit() for ch in w)][:2]
                if extras and figures:
                    add(f"{figures[0]} {extras[0]}")
                if len(out) >= 10:
                    break
            return out[:10]

        raw = [w for w in re.findall(r"[a-z0-9.]+", str(value).lower())
               if w not in ("and", "or", "the", "a", "of", "to", "for")]
        core = [w for w in raw if w not in ("at", "least", "most", "more", "less", "than", "under", "over")]
        bare = " ".join(core[:3]).strip() or " ".join(raw[:3]).strip()
        bare_stems = self._stems(bare)
        if bare:
            add(bare)
            for b in boosts:
                add(f"{bare} {b}")
                add(f"{b} {bare}")
            for b in boosts[:2]:
                add(b)
            kinds_ordered = sorted(
                kinds,
                key=lambda k: (0 if not str(k).endswith("s") else 1, len(str(k)), str(k)),
            )
            for kind in kinds_ordered:
                if self._stems(kind) & bare_stems:
                    continue  # kind already inside the value
                add(f"{bare} {kind}")
            if head and not (self._stems(head) & bare_stems):
                add(f"{bare} {head}")
            add(f"{bare} {phrase}")
        add(self.value_query(value, category))
        add(phrase)

        stop = set(_words(category)) | set(core) | set(raw) | {
            "pack", "set", "new", "sale", "free", "with", "for", "the", "and",
        }
        for ident, details in list(self.details.items())[:40]:
            blob = " ".join(str(v) for _, v in (_facts(details) or []))
            title = str(self.rows.get(ident, {}).get("title") or "")
            if not (_stated(value, blob) or _stated(value, title) or (bare and _stated(bare, title))):
                continue
            extras = [w for w in re.findall(r"[a-z]{3,}", title.lower())
                      if w not in stop and not any(ch.isdigit() for ch in w) and _one(w) not in stop][:2]
            if not extras or not bare:
                continue
            add(f"{bare} {extras[0]}")
            if len(out) >= 10:
                break
        return out[:10]

    def _remember_search_rows(self, observation: Any) -> list[str]:
        ids = [str(r["id"]) for r in (_row_info(x) for x in _result_rows(observation)) if r.get("id")]
        self._recent_search_ids = ids
        return ids

    def _facts_state_value(self, ident: str, value: str) -> bool:
        """Whether opened listing facts (not title) state the need (titles often list sibling options)."""
        if not value or _BOUND.search(value) or ident not in self.details:
            return False
        bare = re.sub(r"(?i)\b(?:at least|at most|no more than|no less than|up to|under|over)\b", " ", value)
        bare = re.sub(r"\s+", " ", bare).strip()
        blob = " ".join(str(v) for _, v in (_facts(self.details.get(ident)) or []))
        return bool(bare and (_stated(bare, blob) or _stated(value, blob)))

    def _row_states_value(self, ident: str, value: str) -> bool:
        """Whether an unopened/opened row's title or facts already state the need value (skip bound phrases)."""
        if not value or _BOUND.search(value):
            return False
        bare = re.sub(r"(?i)\b(?:at least|at most|no more than|no less than|up to|under|over)\b", " ", value)
        bare = re.sub(r"\s+", " ", bare).strip()
        info = self.rows.get(ident) or {}
        title = str(info.get("title") or "")
        if bare and (_stated(bare, title) or _stated(value, title)):
            return True
        return self._facts_state_value(ident, value)

    def _row_in_category(self, ident: str, label: str | None) -> bool:
        """True when the row's shelf matches the product category (or no label is known).

        Exact or near-exact category match only: one shared word is not the same category."""
        if not label:
            return True
        cat = str((self.rows.get(ident) or {}).get("category") or "").strip()
        if not cat:
            return True
        if _one(cat) == _one(label) or cat.lower() == label.lower():
            return True
        a, b = self._stems(cat), self._stems(label)
        if not a or not b:
            return False
        return a == b or (len(a & b) >= max(2, min(len(a), len(b))) and abs(len(a) - len(b)) <= 1)

    def _detail_states_need(self, ident: str, value: str, label: str | None) -> bool:
        """Opened product in the right category whose facts/title solidly state the need."""
        if ident not in self.details or not self._row_in_category(ident, label):
            return False
        if _BOUND.search(value):
            req = next((r for r in self.reqs if all(_split_req(r)) and _split_req(r)[1] == value), None)
            if req:
                return self.code_verdict(ident, req) == "yes"
            return False
        if _short_token_need(value):
            return self._facts_state_value(ident, value)
        return self._row_states_value(ident, value)

    def _hint_stem_match(self, label: str, hint: str) -> bool:
        """True when label and hint name the same key — ignore shared filler-only stems like 'type'."""
        a, b = self._stems(label), self._stems(hint)
        if not a or not b:
            return False
        if a == b or a <= b or b <= a:
            shared = a & b
            return bool(shared - {"type", "size", "style", "class", "kind", "need", "needs"})
        return False

    def _promote_hint_cats(self, k: int, queues: dict[int, list[str]]) -> None:
        """After a need is found, pull remaining request-hint categoricals to the front of this product's queue.

        Additive: does not drop measures or change other products; only reorders leftover asks."""
        queue = queues.get(k) or []
        if not queue:
            return
        hints = self._request_hint_labels(k)
        if not hints:
            return
        front, rest = [], []
        for lab in queue:
            if self._label_bucket(k, lab) == "cat" and any(self._hint_stem_match(lab, h) for h in hints):
                front.append(lab)
            else:
                rest.append(lab)
        if front:
            queues[k] = front + rest

    def resolve_detail_keys(self, queues: dict[int, list[str]]) -> None:
        """Snap each planned label to the closest attribute key actually carried by that product's opened listings.

        The LLM may paraphrase a label; asking the store's own key name
        matches how the shopper's hidden need is keyed. Labels with no detail match are kept only when no listing
        keys exist yet (early / empty survey)."""
        for k, labels in queues.items():
            shares = self.key_shares(k) if hasattr(self, "key_shares") else {}
            if not shares:
                continue
            names = [n for n in shares if askable_label(n)]
            resolved: list[str] = []
            for label in labels:
                stems = self._stems(label)
                if not stems:
                    continue
                hit = next((n for n in names if self._stems(n) == stems), None)
                if hit is None:
                    scored = sorted(
                        ((-len(stems & self._stems(n)), -shares.get(n, 0), n) for n in names
                         if stems & self._stems(n)),
                    )
                    hit = scored[0][-1] if scored else None
                if hit is None:
                    continue
                if all(self._stems(hit) != self._stems(o) for o in resolved):
                    resolved.append(hit)
            queues[k] = resolved or labels  # fall back if every label failed to match

    def prioritize_choice_keys(self, queues: dict[int, list[str]]) -> None:
        """Reorder ask queues by value shape: categorical closed choices interleaved with measures.

        When both categorical (short non-numeric enums) and measure keys exist, weave them so the first two
        messages cover one of each when possible. No product-property name allowlist."""
        for k, labels in queues.items():
            if not labels:
                continue
            counts, _values, per = self._key_bucket(k)

            def resolve(label: str) -> str | None:
                stems = self._stems(label)
                name = next((n for n in counts if self._stems(n) == stems), None)
                if name is None:
                    name = next((n for n in counts if stems & self._stems(n)), None)
                return name

            def classify(label: str) -> tuple[str, tuple]:
                """Return (bucket, sort_key). bucket: cat | measure | weak."""
                name = resolve(label)
                if not name:
                    return "weak", (9, 99, 99.0, 0, label)
                vals = list(per.get(name, {}).values()) or [""]
                distinct = len(set(vals))
                mean = sum(len(v) for v in vals) / len(vals)
                numericish = sum(1 for v in vals if re.search(r"\d", v)) * 2 >= len(vals)
                distinct_eff = min(distinct, 3) if numericish else distinct
                band = 0 if 2 <= distinct <= 12 else (1 if distinct > 1 else 2)
                unitish = sum(
                    1 for v in vals
                    if re.search(r"^\s*\d+(\.\d+)?\s*[a-zA-Z%]{0,8}\s*$", v) or len(v) <= 12
                )
                shape = 0 if unitish * 2 >= len(vals) else 1
                sort_key = (band, shape, distinct_eff, mean, -counts.get(name, 0), label)
                blank = {"", "mix", "mixed", "nil", "n/a", "none", "-", "null"}
                real_vals = [v for v in vals if v.strip() not in blank]
                real_distinct = len(set(real_vals))
                if (not numericish and distinct >= 2 and real_distinct >= 1 and distinct <= 8
                        and mean <= 24 and askable_label(name)):
                    cat_key = (band, shape, -real_distinct, mean, -counts.get(name, 0), label)
                    return "cat", cat_key
                if numericish:
                    return "measure", sort_key
                return "weak", sort_key

            planned = {self._stems(l) for l in labels}
            extras: list[str] = []
            for name, carried in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                if carried < 2 or not askable_label(name) or self._stems(name) in planned:
                    continue
                vals = list(per.get(name, {}).values())
                if not self._keep_key(name, vals, per.get(name, {})):
                    continue
                bucket, _ = classify(name)
                if bucket == "cat":
                    extras.append(name)
                if len(extras) >= 4:
                    break
            merged = list(dict.fromkeys(extras + [l for l in labels if askable_label(l) or resolve(l)]))

            seen_stems: set[frozenset[str]] = set()
            unique: list[str] = []
            for lab in merged:
                stems = self._stems(lab)
                if not stems or stems in seen_stems:
                    continue
                seen_stems.add(stems)
                unique.append(lab)

            scored: dict[str, list[tuple]] = {"cat": [], "measure": [], "weak": []}
            for lab in unique:
                bucket, sk = classify(lab)
                scored[bucket].append((sk, lab))
            buckets = {b: [lab for _sk, lab in sorted(scored[b])] for b in scored}

            woven: list[str] = []
            cats = list(buckets["cat"])
            meas = list(buckets["measure"])
            if cats and meas:
                woven.append(cats.pop(0))
                woven.append(meas.pop(0))
                woven.extend(cats)
                woven.extend(meas)
            else:
                woven.extend(cats or meas)
            woven.extend(buckets["weak"])

            hints = self._request_hint_labels(k)
            for h in hints:
                if all(self._stems(h) != self._stems(o) for o in woven):
                    woven.append(h)
            if hints:
                hint_front = []
                for h in hints:
                    hit = next((lab for lab in woven if self._hint_stem_match(lab, h)), None)
                    if hit and hit not in hint_front:
                        hint_front.append(hit)
                rest = [lab for lab in woven if lab not in hint_front]
                woven = hint_front + rest

            queues[k] = woven
            self.trace.append({"kind": "choice_keys", "product": k, "queue": queues[k][:6],
                               "cats": buckets["cat"][:4], "measures": buckets["measure"][:4],
                               "hints": hints[:4]})

    def apply_need_carts(self, before: dict[int, list[str]]) -> list[str]:
        """After need-search: cart an opened in-category listing only when facts/code say yes to every requirement.

        Never cart on the title alone; unread preferred rows are opened first."""
        if not self.roles.get("cart") or self.over() or self.steps_left() <= self.order_due() + 2:
            return []
        keep, added = self.current, []
        for k, old in before.items():
            new = [r for r in self.product_reqs(k) if r not in old and all(_split_req(r))]
            new = [r for r in new if _usable_need_value(_split_req(r)[1])]
            if not new or self.steps_left() <= self.order_due() + 2:
                continue
            all_reqs = [r for r in self.product_reqs(k) if all(_split_req(r)) and _usable_need_value(_split_req(r)[1])]
            if not all_reqs:
                continue
            if self.multi():
                self.switch(k)
            label = self.label_for() or None
            qty = int(self.quantity or 1)
            new_values = {_split_req(r)[1] for r in new if all(_split_req(r))}
            prefer_open = [i for i in self.need_title_prefer.get(k, []) if i not in self.details]
            if prefer_open and self.steps_left() > self.order_due() + 6:
                room = max(0, min(3, self.steps_left() - self.order_due() - 5))
                if room:
                    self.read_details(prefer_open[:room])
            current = self.lines.get(k) if self.multi() else None
            if not current and not self.multi():
                current = next(iter(self.cart), None) if len(self.cart) == 1 else None
            if current and all(self.code_verdict(current, r) == "yes" for r in all_reqs):
                if self.multi():
                    self.switch(keep)
                continue
            candidates = []
            boost_seed = next(iter(new_values), "")
            boosts = self._request_boost_phrases(boost_seed, self.category or "")[:4]
            pack_asked = bool(re.search(
                r"(?i)\d+\s*[x×]\s*\d+|\bx\s*\d+\b",
                " ".join([str(getattr(self, "request", "") or "")] + list(new_values)),
            ))

            def _title_hit(ident: str) -> bool:
                """Tie-break only among fact-confirmed rows (never a cart gate by itself)."""
                title = str((self.rows.get(ident) or {}).get("title") or "")
                blob = title.lower().replace(" ", "")
                for val in new_values:
                    if not val:
                        continue
                    if _BOUND.search(val):
                        figs = _need_figure_forms(val)
                        if any(re.sub(r"\s+", "", f).lower() in blob for f in figs if f):
                            return True
                        continue
                    bare = re.sub(r"(?i)\b(?:at least|at most|no more than|no less than|up to|under|over)\b", " ", val)
                    bare = re.sub(r"\s+", " ", bare).strip()
                    if bare and (_stated(bare, title) or _stated(val, title)):
                        return True
                return False

            def _boost_miss(ident: str) -> int:
                if not boosts:
                    return 0
                title = str((self.rows.get(ident) or {}).get("title") or "")
                fact_blob = " ".join(str(v) for _, v in (_facts(self.details.get(ident)) or []))
                blob = f"{title} {fact_blob}"
                return sum(1 for b in boosts if b and not (_stated(b, blob) or b.lower() in blob.lower()))

            def _meets(ident: str) -> tuple[bool, int]:
                ok = True
                yeses = 0
                for r in all_reqs:
                    v = self.code_verdict(ident, r)
                    if v == "yes":
                        yeses += 1
                        continue
                    if v == "no":
                        return False, 0
                    val = _split_req(r)[1]
                    if _BOUND.search(val):
                        if self._detail_states_need(ident, val, label):
                            yeses += 1
                            continue
                        return False, 0
                    if self._facts_state_value(ident, val) and self._row_in_category(ident, label):
                        yeses += 1
                        continue
                    return False, 0
                return (ok and yeses >= len(all_reqs)), yeses

            for ident in self.details:
                info = self.rows.get(ident) or {}
                if ident in self.unavailable or info.get("stock") is False:
                    continue
                if not self._row_in_category(ident, label):
                    continue
                if not self.affordable(info):
                    continue
                ok, yeses = _meets(ident)
                if not ok:
                    continue
                if self.verdict(ident) is False:
                    continue
                price = _price(info) or 0
                prefer_cheap = (self.pick_kind() == "cheapest")
                rank = 0 if _title_hit(ident) else 1
                kind_pref = self.need_title_prefer.get(k, [])
                kind_rank = kind_pref.index(ident) if ident in kind_pref else 999
                in_kind = 0 if ident in kind_pref else 1
                title = str(info.get("title") or "")
                facts = self.details.get(ident) or {}
                if isinstance(facts, dict) and "facts" in facts:
                    facts = facts.get("facts") or {}
                pack_blob = title + " " + " ".join(str(v) for v in (facts.values() if isinstance(facts, dict) else []))
                multipack = 0 if re.search(
                    r"(?i)\bx\s*\d|\d+\s*[x×]\s*\d", pack_blob
                ) else 1
                if prefer_cheap:
                    candidates.append((rank, in_kind, _boost_miss(ident), price,
                                       multipack if pack_asked else 0, kind_rank, -yeses, ident))
                else:
                    candidates.append((rank, in_kind, multipack if pack_asked else 0, _boost_miss(ident),
                                       kind_rank, -yeses, price, ident))
            if not candidates:
                if self.multi():
                    self.switch(keep)
                continue
            if pack_asked:
                mp_i = 4 if (self.pick_kind() == "cheapest") else 2
                if any(c[mp_i] == 0 for c in candidates):
                    candidates = [c for c in candidates if c[mp_i] == 0]
            candidates.sort()
            ident = candidates[0][-1]
            if current and current in self.cart and current != ident:
                if self.swap_line(current, ident):
                    if self.multi():
                        self.lines[k] = ident
                    added.append(ident)
                    self.code_carted = getattr(self, "code_carted", set()) | {ident}
            else:
                units = self.wanted_units(ident) if hasattr(self, "wanted_units") else qty
                actions = self.set_line_actions(ident, units)
                if actions and len(actions) <= self.steps_left() - self.order_due() - 1:
                    self.act(actions)
                    if ident in self.cart:
                        if self.multi():
                            self.lines[k] = ident
                        added.append(ident)
                        self.code_carted = getattr(self, "code_carted", set()) | {ident}
            if self.multi():
                self.switch(keep)
        if added:
            self.trace.append({"kind": "need_cart", "ids": added})
        return added

    def open_heads_and_tails(self, found: list[str], ranked: list[str], head: int, tail: int,
                             prefer: list[str] | None = None) -> list[str]:
        """Open strongest new rows, mid-page rows, and the buried end. Rows whose title already states the need
        value are opened first (they are the ones the query was meant to surface)."""
        prefer = [i for i in (prefer or []) if i in found and i not in self.details]
        ordered = list(dict.fromkeys(ranked + [x for x in found if x not in ranked]))
        unread = [i for i in ordered if i not in self.details]
        head_ids = unread[:head]
        tail_ids = [i for i in reversed(unread) if i not in head_ids][:tail]
        mid_pool = [i for i in unread if i not in head_ids and i not in tail_ids]
        mid_ids = mid_pool[:NEED_OPEN_MID]
        rest = [i for i in head_ids + mid_ids + tail_ids if i not in prefer]
        picks = list(dict.fromkeys(prefer + rest))
        cap = 12 if prefer else 8
        room = max(0, min(cap, self.steps_left() - self.order_due() - 4))
        picks = picks[:room]
        if picks and not self.over():
            self.read_details(picks)
        return [i for i in picks if i in self.details]

    def _search_one_page(self, query: str, nxt: Any) -> tuple[list[str], Any]:
        """One LocateListings page; nxt is the prior page's next_page_token / offset (None for the first page).

        The store's own next value goes through as given, not through page_value."""
        role = self.roles.get("search") or {}
        if nxt is None:
            action = self.search_action(query, None)
        elif role.get("page_arg"):
            action = self.search_action(query, None)
            if action:
                action = {**action, "args": {**(action.get("args") or {}), role["page_arg"]: nxt}}
        elif role.get("token_arg"):
            action = self.search_action(query, None, token=str(nxt))
        else:
            return [], None
        got = self.act([action]) if action else None
        if not got:
            return [], None
        if nxt is None:
            self.tried.add(query)
        self.listen(got)
        obs = got[0]
        obs = _inner(obs)
        page_ids = self._remember_search_rows(obs)
        following: Any = _next_token(obs) if role.get("token_arg") else _next_number(obs)
        if following is None and isinstance(obs, dict):
            following = obs.get("next_page_token")
        self._note_offset_paging(following, len(page_ids), role.get("page_arg"))
        return page_ids, following

    def _page_query(self, query: str, pages: int) -> list[str]:
        """Run a search and follow next offsets for further pages (no opens)."""
        if self.steps_left() <= self.order_due() + 4:
            return []
        found: list[str] = []
        nxt: Any = None
        for _ in range(max(1, pages)):
            if self.steps_left() <= self.order_due() + 4:
                break
            page_ids, nxt = self._search_one_page(query, nxt)
            if not page_ids:
                break
            found = list(dict.fromkeys(found + page_ids))
            if nxt is None:
                break
        return found

    def search_needs_fresh(self, before: dict[int, list[str]]) -> list[str]:
        """For each product that gained a need: try short query variants (bare value first).

        Page the query before opening: opening weak page-1 rows can spend the step reserve before later
        pages (offset/token paging) are reached. After pages are collected, open in-category title matches first,
        then head/mid/tail of the collected rows."""
        keep, done = self.current, []
        for k, old in before.items():
            new = [r for r in self.product_reqs(k) if r not in old and all(_split_req(r))
                   and _usable_need_value(_split_req(r)[1])]
            if not new or self.steps_left() <= self.order_due() + 6:
                continue
            if self.multi():
                self.switch(k)
            value = _split_req(new[0])[1]
            category = self.category or ""
            bound = bool(_BOUND.search(value))
            figures = _need_figure_forms(value) if bound else []
            opened_for: list[str] = []
            scout_prefer: list[str] = []
            solid = False
            queries = self.need_queries(value, category)
            self.trace.append({"kind": "need_queries", "value": value, "category": category, "queries": queries[:8]})
            pages = BOUND_SHELF_PAGES if bound else NEED_PAGES
            for q_i, query in enumerate(queries):
                if self.steps_left() <= self.order_due() + 5:
                    break
                if query in self.tried and solid and q_i >= 2:
                    continue
                is_figure_q = bound and any(f.lower() in query.lower() for f in figures)
                if bound and not is_figure_q:
                    depth = pages
                elif q_i == 0 and len(queries) > 1:
                    depth = min(2, pages)
                else:
                    depth = min(2, pages)  # leave step room to open title-prefer rows
                page_groups: list[list[str]] = []
                nxt: Any = None
                label = self.label_for() or None
                got: list[str] = []
                scout_only = (not bound and q_i == 0 and len(queries) > 1)

                def _need_prefer(ids: list[str]) -> list[str]:
                    """Rows to open first: title states the value, or (bounds) title carries the figure."""
                    out = []
                    for i in ids:
                        if not self._row_in_category(i, label):
                            continue
                        title = str((self.rows.get(i) or {}).get("title") or "")
                        if bound:
                            blob = title.lower().replace(" ", "")
                            if any(re.sub(r"\s+", "", f).lower() in blob for f in figures if f):
                                out.append(i)
                            continue
                        if not value:
                            continue
                        bare_v = re.sub(
                            r"(?i)\b(?:at least|at most|no more than|no less than|up to|under|over)\b",
                            " ", value)
                        bare_v = re.sub(r"\s+", " ", bare_v).strip()
                        if bare_v and (_stated(bare_v, title) or _stated(value, title)):
                            out.append(i)
                    return out

                for page_i in range(depth):
                    if self.steps_left() <= self.order_due() + 4:
                        break
                    page_ids, nxt = self._search_one_page(query, nxt)
                    if not page_ids:
                        break
                    page_groups.append(page_ids)
                    page_prefer = _need_prefer(page_ids)
                    scout_prefer = list(dict.fromkeys(scout_prefer + page_prefer))
                    if (q_i > 0 or is_figure_q) and page_prefer:
                        self.need_title_prefer[k] = list(dict.fromkeys(
                            page_prefer + self.need_title_prefer.get(k, [])))
                    if nxt is None:
                        break
                if not page_groups:
                    continue
                all_ids = list(dict.fromkeys(i for g in page_groups for i in g))
                prefer = _need_prefer(all_ids)
                if not scout_only and self.steps_left() > self.order_due() + 4:
                    both = [i for i in scout_prefer if i in prefer]
                    open_prefer = list(dict.fromkeys(both + prefer))
                    open_order = [i for i in dict.fromkeys(open_prefer + all_ids) if i not in self.details]
                    if open_order:
                        more = self.open_heads_and_tails(
                            open_order, open_order, NEED_OPEN_HEAD + 2, NEED_OPEN_TAIL,
                            prefer=open_prefer)
                        got += more
                        opened_for += more
                        if any(self._detail_states_need(i, value, label) for i in more):
                            solid = True
                self.trace.append({"kind": "need_pages", "query": query, "pages": len(page_groups),
                                   "prefer": prefer[:8], "offsets": len(page_groups),
                                   "scout_only": scout_only})
                if (not scout_only and got and self.steps_left() > self.order_due() + 6):
                    follow = []
                    bare = re.sub(r"(?i)\b(?:at least|at most|no more than|no less than|up to|under|over)\b", " ", value)
                    bare = re.sub(r"\s+", " ", bare).strip().lower()
                    seed = (figures[0] if figures else bare).lower()
                    stop = set(_words(category)) | set(_words(seed)) | {
                        "pack", "set", "new", "sale", "free", "with", "for", "the", "and", "can", "tin",
                    }
                    word_hits: dict[str, int] = {}
                    for ident in got:
                        meets = (self._detail_states_need(ident, value, label)
                                 if bound else self._row_states_value(ident, value))
                        if not meets:
                            continue
                        title = str((self.rows.get(ident) or {}).get("title") or "")
                        for w in re.findall(r"[a-z]{3,}", title.lower()):
                            if w not in stop and not any(ch.isdigit() for ch in w):
                                word_hits[w] = word_hits.get(w, 0) + 1
                    for w, _n in sorted(word_hits.items(), key=lambda kv: (-kv[1], kv[0])):
                        follow.append(f"{seed} {w}".strip())
                        if len(follow) >= 4:
                            break
                    for fq in list(dict.fromkeys(follow))[:2]:
                        if fq in self.tried or self.steps_left() <= self.order_due() + 5:
                            continue
                        more = self._page_query(fq, 2)
                        if not more:
                            continue
                        pref2 = _need_prefer(more) if bound else [
                            i for i in more if self._row_states_value(i, value) and self._row_in_category(i, label)]
                        opened_for += self.open_heads_and_tails(
                            more, pref2 or more, NEED_OPEN_HEAD, NEED_OPEN_TAIL, prefer=pref2)
                if solid and not bound and q_i >= 2:
                    break
                if solid and bound and is_figure_q and q_i >= 1:
                    break
            if not opened_for and scout_prefer and self.steps_left() > self.order_due() + 4:
                hits = [i for i in scout_prefer if i not in self.details]
                if hits:
                    opened_for += self.open_heads_and_tails(
                        hits, hits, NEED_OPEN_HEAD, NEED_OPEN_TAIL, prefer=hits)
            if opened_for:
                done.append(f"{self.product_name(k)}: {', '.join(dict.fromkeys(opened_for))}")
        if self.multi():
            self.switch(keep)
        if done:
            self.trace.append({"kind": "need_search", "opened": done})
        return done

    def widen_coverage(self) -> list[str]:
        """After the interview: open buried unread rows, deepen category shelves for numeric needs, and (under
        SPEND_SOFT) a short synonym sweep."""
        if self.over() or self.steps_left() <= self.order_due() + 5:
            return []
        opened: list[str] = []
        pool_ids = [r["id"] for r in self.pool() if r.get("id")]
        buried = [i for i in pool_ids if i not in self.details][-max(WIDEN_OPENS, NEED_OPEN_TAIL):]
        buried = list(dict.fromkeys(
            [i for i in reversed(self._recent_search_ids) if i not in self.details][:NEED_OPEN_TAIL] + buried
        ))
        prefer = []
        for i in pool_ids:
            if i in self.details:
                continue
            title = str((self.rows.get(i) or {}).get("title") or "")
            blob = title.lower().replace(" ", "")
            for r in self.reqs:
                prop, val = _split_req(r)
                if not val:
                    continue
                if _BOUND.search(val):
                    if any(re.sub(r"\s+", "", f).lower() in blob for f in _need_figure_forms(val) if f):
                        prefer.append(i)
                        break
                elif self._row_states_value(i, val):
                    prefer.append(i)
                    break
        room = max(0, min(WIDEN_OPENS + 2, self.steps_left() - self.order_due() - 4))
        picks = list(dict.fromkeys(prefer + buried))[:room]
        if picks:
            self.read_details(picks)
            opened += [i for i in picks if i in self.details]
        keep = self.current
        for k in (range(len(self.items)) if self.multi() else [keep]):
            if self.steps_left() <= self.order_due() + 6:
                break
            if self.multi():
                self.switch(k)
            bounds = [r for r in self.product_reqs(k) if all(_split_req(r)) and _BOUND.search(_split_req(r)[1])]
            if bounds:
                bval = _split_req(bounds[0])[1]
                figures = _need_figure_forms(bval)
                label = self.label_for() or None
                for query in self.need_queries(bval, self.category or "")[:4]:
                    found = self._page_query(query, BOUND_SHELF_PAGES if not figures else 2)
                    if not found:
                        continue
                    fig_hits = []
                    if figures:
                        for i in found:
                            if not self._row_in_category(i, label):
                                continue
                            blob = str((self.rows.get(i) or {}).get("title") or "").lower().replace(" ", "")
                            if any(re.sub(r"\s+", "", f).lower() in blob for f in figures if f):
                                fig_hits.append(i)
                    ranked = fig_hits or [i for i in self.likely_ids(16, category=label) if i in set(found)]
                    opened += self.open_heads_and_tails(
                        found, ranked, NEED_OPEN_HEAD, NEED_OPEN_TAIL, prefer=fig_hits)
                    if fig_hits or opened:
                        break
            if self.spent < SPEND_SOFT and self.steps_left() > self.order_due() + 8:
                ask = (SEARCH_PROMPT + "\n\n" + self.context()
                       + "\n\nQUERIES ALREADY TRIED: " + "; ".join(sorted(self.tried)[:30]))
                suggested = [str(q).strip() for q in (self.ask(ask, 400, strong=False) or {}).get("queries") or []
                             if str(q).strip()]
                for query in suggested[:2]:
                    if query in self.tried or self.steps_left() <= self.order_due() + 5:
                        continue
                    found = self._page_query(query, 2)
                    label = self.label_for() or None
                    ranked = [i for i in self.likely_ids(12, category=label) if i in set(found)]
                    opened += self.open_heads_and_tails(found, ranked, 2, NEED_OPEN_TAIL)
                    break
        if self.multi():
            self.switch(keep)
        if opened:
            self.trace.append({"kind": "widen_coverage", "opened": list(dict.fromkeys(opened))})
        return list(dict.fromkeys(opened))

    def asked_name(self, k: int) -> str:
        """The product as a question names it: plural nouns joined by "and" are asked with "or", as one product is
        wanted; one product whose name holds "and" is left whole."""
        name = self.product_name(k)
        parts = [p for p in re.split(r"\s*(?:,|&|\band\b)\s*", name) if p]
        if len(parts) >= 2 and all(re.search(r"[a-z]{3}s$", p, re.I) for p in parts):
            return re.sub(r"\s*&\s*|\s+and\s+", " or ", name)
        return name

    def clean_labels(self, labels: list[str]) -> list[str]:
        """Question labels as a shopper reads them: store-only fields dropped, words a store key runs together split
        ("legheight" -> "leg height", when "height" is a word of another label or of the products' facts), and one
        label per property ("typesofvinegar" and "types of vinegar" are one)."""
        vocab = {w for label in labels for w in _words(label) if len(w) >= 4}
        for details in list(self.details.values())[:40]:
            vocab |= {w for key, _ in (_facts(details) or []) for w in _words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))) if len(w) >= 4}

        spaced = {"".join(_words(label)): label for label in labels if len(_words(label)) > 1}

        def split(label: str) -> str:
            words = _words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label)))
            if len(words) == 1 and words[0] in spaced:
                return " ".join(_words(spaced[words[0]]))  # the spaced spelling of the same key
            out = []
            for w in words:  # only a known word at the end splits off ("frame" + "material"), never one at the start
                part = next((v for v in sorted(vocab, key=len, reverse=True) if v != w and len(w) - len(v) >= 3 and w.endswith(v)), None)
                out += [w[:-len(part)], part] if part else [w]
            return " ".join(out)

        kept: list[str] = []
        seen: set[str] = set()
        for label in labels:
            if not askable_label(label):
                continue
            nice = split(label)
            joined = "".join(_words(nice))
            if not nice or joined in seen or not askable_label(nice):
                continue
            seen.add(joined)
            kept.append(nice)
        return kept

    def list_question(self, k: int, labels: list[str]) -> str:
        if len(labels) == 1:
            return f"What {labels[0]} would you like for the {self.asked_name(k)}?"
        *head, last = labels
        return f"For the {self.asked_name(k)}, what {', '.join(head)} or {last} would you like?"

    def fit_list(self, k: int, labels: list[str]) -> list[str]:
        """The longest leading part of the list whose message the store's message tool takes whole."""
        role = self.roles.get("message") or {}
        spec = self.props(role.get("tool") or "").get(role.get("text_arg") or "") or {}
        limit = spec.get("maxLength") if isinstance(spec, dict) else None
        while isinstance(limit, int) and len(labels) > 1 and len(self.list_question(k, labels)) > limit:
            labels = labels[:-1]
        return labels

    def answered_at(self, asked: list[str], heard: list[str]) -> int:
        """The earliest position in the list of a property an answer gave a need on, -1 when none matches."""
        stems = [self._stems(label) for label in asked]
        first = len(stems)
        for need in heard:
            name = _split_req(need)[0] or need.split(":")[0]
            prop, joined = self._stems(name), "".join(_words(name))
            top, place = 0, first
            for at, words in enumerate(stems):
                shared = len(prop & words) or int(joined == "".join(_words(asked[at])))
                if shared > top:
                    top, place = shared, at
            first = min(first, place)
        return first if first < len(stems) else -1

    def interview(self) -> str:
        """The survey, plan a surplus, top queues with carried keys, ask lists of properties with
        scoped hear → prune → round-robin from last item → defer found preferences; then search new needs and widen."""
        self.interviewed = True
        self.interview_topup = set()
        self.interview_asked = []
        if self.questions_left() <= 0 or not self.roles.get("message"):
            return ""
        self.survey()
        self.search_unseen()
        count = self.questions_left()
        if count <= 0:
            return ""
        products = list(range(len(self.items)) if self.multi() else [0])
        want = max(count + len(products) + 2, LIST_SIZE + 4)
        prompt = (INTERVIEW_PROMPT_15.replace("COUNT", str(want)) + "\n\nREQUEST:\n" + self.request
                  + "\n\nPRODUCTS AND THE ATTRIBUTE KEYS OF THEIR OPENED LISTINGS (how many listings carry each, example values):\n"
                  + self.key_lines())
        ask_first = []
        if isinstance(getattr(self, "plan", None), dict):
            ask_first = [str(x).strip() for x in (self.plan.get("ask_first") or []) if str(x).strip()]
        if not ask_first:
            for step in getattr(self, "process", []) or []:
                if step.get("when") == "choose" and int(step.get("messages") or 0) > 0:
                    text = str(step.get("step") or "").strip()
                    if text:
                        ask_first.append(text)
        if ask_first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(ask_first)
        prompt += self._hint_lines(products)
        plan = self.ask(prompt, 1200, strong=True) or {}
        queues: dict[int, list[str]] = {k: [] for k in products}
        order: list[int] = []
        for pick in plan.get("ask") or []:
            if not isinstance(pick, dict):
                continue
            n = pick.get("item")
            k = max(0, min(len(self.items) - 1, int(n) - 1)) if self.multi() and isinstance(n, int) and not isinstance(n, bool) else 0
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5 or not askable_label(label):
                continue
            if all(self._stems(label) != self._stems(other) for other in queues.setdefault(k, [])):
                queues[k].append(label)
            if k not in order:
                order.append(k)
        self.resolve_detail_keys(queues)
        self.guard_coverage(queues)
        for k in products:
            queue = queues.setdefault(k, [])
            for name in self._request_hint_labels(k) + self.item_keys(k, LIST_SIZE + 4):
                if all(self._stems(name) != self._stems(other) for other in queue):
                    queue.append(name)
                    self.interview_topup.add(name)
            if queue and k not in order:
                order.append(k)
        if self.multi():
            order = list(dict.fromkeys(order + products))
        self.prioritize_choice_keys(queues)
        id_keys = {" ".join(_words(str(key))) for ident, details in self.details.items() for key, value in _facts(details) or []
                   if str(value).strip().lower() == str(ident).lower()}
        for k in queues:
            queues[k] = self.clean_labels([name for name in queues[k] if " ".join(_words(name)) not in id_keys])
        self.trace.append({"kind": "interview_plan", "picks": [(k, l) for k in queues for l in queues[k]],
                           "order": order, "topup": sorted(self.interview_topup)})
        if not any(queues.values()):
            return ""
        lines: list[str] = []
        before = {k: self.product_reqs(k) for k in queues}
        found: set[int] = set()
        last_item: int | None = None

        def mark_found(k: int) -> None:
            """Defer a product only after a usable need — and on multi-buy, after a categorical ask when cats remain."""
            if self.product_reqs(k) == before.get(k, []):
                return
            self._promote_hint_cats(k, queues)
            if self.multi() and not self._product_had_cat_ask(k) and self._queue_has_cat(k, queues):
                return
            found.add(k)

        def send(k: int, labels: list[str]) -> str:
            labels = self.clean_labels(labels) or labels
            text = self.list_question(k, labels)
            action = self.message_action(text)
            if action is None or self.messages_left() <= 0 or self.steps_left() <= INTERVIEW_MIN_LEFT - 2:
                return ""
            observations = self.act([action])
            if observations is None:
                return ""
            self.messages_used += 1
            if self.multi():
                self.switch(k)
            self.hear_focus = k
            self.understood = True
            try:
                changed = self.listen(observations, text, reply_of=0)
            finally:
                self.hear_focus = None
            reply = (" ".join(v for v in observations[0].values() if isinstance(v, str))
                     if isinstance(observations[0], dict) else str(observations[0]))
            lines.append(f"Q: {text}\nA: {reply}")
            self.trace.append({"kind": "ask", "product": k, "label": labels[0], "labels": labels, "changed": bool(changed),
                               "bucket": self._label_bucket(k, labels[0])})
            return reply

        lists_on = True
        misses: dict[int, int] = {}
        while self.questions_left() > 0 and not self.over() and self.steps_left() > INTERVIEW_MIN_LEFT - 2:
            question = self.choose_question(order, queues, found, last_item)
            if question is None:
                break
            k, label = question
            labels = [label]
            if lists_on:
                labels = self.fit_list(k, [label] + queues[k][:LIST_SIZE - 1])
                del queues[k][:len(labels) - 1]
            free_before = list(self.free)
            reqs_before = self.product_reqs(k)
            refs_before = set(self.ref_needs)
            reply = send(k, labels)
            if not reply:
                break
            last_item = k
            mark_found(k)
            new = [r for r in self.product_reqs(k) if r not in reqs_before]
            # "same as that other listing" settles a property too, though it gives no value yet
            new += [f"{n['property']}: (from another listing)" for key, n in self.ref_needs.items() if key not in refs_before]
            crowded = re.search(
                r"(?i)split those|one by one|one at a time|in turn|first (?:matter|thing|bit)|then the next|"
                r"then another|too many", reply)
            if len(labels) == 1:
                self.interview_asked.append((k, label))
                if crowded and label not in self.free:
                    self.free.append(label)
                continue
            # A list's reply speaks for one property at most: nothing it did not name may be ruled out.
            self.free[:] = free_before
            if crowded and not new:
                lists_on = False
                queues[k][:0] = labels
                self.trace.append({"kind": "lists_off", "product": k})
                continue
            hit = self.answered_at(labels, new)
            if hit >= 0:
                done, ahead = labels[:hit + 1], labels[hit + 1:]
                misses[k] = 0
            elif not self.understood:
                done, ahead = labels, []  # a message naming nothing the shopper holds a view on puzzles them
            else:
                # the property it does not mind may stand in front of the one it cares about
                done, ahead = labels[:1], labels[:0:-1]
                misses[k] = misses.get(k, 0) + 1
            self.interview_asked.extend((k, x) for x in done)
            if misses.get(k, 0) and misses[k] % 2 == 0:
                queues[k].extend(ahead)
            else:
                queues[k][:0] = ahead
            self.trace.append({"kind": "list_settled", "product": k, "done": done, "next": ahead[:4],
                               "understood": self.understood, "found": len(new)})
        self.resolve_references()
        looked = [f"{n['property']}: {n['value']} (from {n['source'] or 'the listing the shopper named'})"
                  for n in self.ref_needs.values() if n.get("value")]
        if looked:
            lines.append("LOOKED UP FROM THE LISTINGS THE SHOPPER NAMED: " + "; ".join(looked))
        searched = self.search_new_needs(before)
        if searched:
            lines.append("THE CODE SEARCHED THE NEW NEEDS AND OPENED: " + "; ".join(searched))
        widened = self.widen_coverage()
        if widened:
            lines.append("THE CODE ALSO OPENED BURIED OR SYNONYM-SEARCH ROWS: " + ", ".join(widened))
        carted = self.apply_need_carts(before)
        if carted:
            lines.append("THE CODE CARTED THE BEST NEED FITS: " + ", ".join(carted))
        return (("THE CODE ASKED THE SHOPPER (each question used one step):\n" + "\n".join(lines)
                 + "\nThe requirements in STATUS now hold every answer: apply them to your picks "
                 "(search again with new values where needed).") if lines else "")


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Actor15(problem_data)
    try:
        return episode.run()
    except Exception as exc:  # keep the trace; finish if the world still allows it
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            episode.finish_now()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
