# Shopping agent for ORO world packs.
#
# An LLM controller drives the world's own tools, one store step per call. Code
# keeps what the controller would otherwise lose: every listing seen and what its
# record says, the shopper's requirements and later changes, and the step and
# message budgets. Questions to the shopper are planned separately, and every
# finish is checked before it is sent. The step budget decides the work.

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# each evaluation run's inference key has a USD cap ($2 qualifying, $5 race of 90 tasks, shared with the shopper
# simulation), so every role uses Haiku-class models: Sonnet-first roles spent ~$0.08 a task and emptied the race key
MAIN_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-flash"]
STRONG_MODELS = ["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "deepseek/deepseek-v4-flash"]
# the few judgements that decide a task (which properties to ask, what a reply states), while the task's spend is low
HEAVY_MODELS = ["anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
SPEND_SOFT = 0.055  # USD spent in a task after which optional model calls (notes, audits, fallback planning) stop
SPEND_HARD = 0.078  # USD spent in a task after which no Sonnet call is made; the walk goes on with the cheap model
SPEND_STOP = 0.095  # USD spent in a task at which the walk stops and the cart is ordered (race 15: finishing at SPEND_HARD left two thirds of the steps unused)
RATE_IN, RATE_OUT = 0.10e-6, 0.50e-6  # USD per token of the cheap models, when a response carries no cost

SEARCH_ROWS = 12  # rows a query shows before the system stops fetching its next pages (race 8 pages hold 4 rows)
AUTO_PAGES = 14  # most next pages the system fetches in one task
CHECKOUT_RESERVE = 7  # steps left when the first order is placed, so a refusal at checkout can still be handled
LLM_TIMEOUT = 70  # longest single inference call, seconds
MAX_TURNS = 90
MAX_CALLS_PER_TURN = 1  # tool calls of one controller turn that run: batches of 4 (as KYO) confused our controller (mock 20/27, practice 0.58)
RECENT_RESULTS = 6  # tool results kept in full in the controller history
NOTE_READS = 8  # most listing descriptions read by the notes reader per task
NOTE_TOOL = "note_to_self"

_env = ProxyClient(timeout=45, max_retries=1)
_llm = ProxyClient(timeout=75, max_retries=1)

SYSTEM_PROMPT = """You are a meticulous shopping assistant serving one shopper inside a simulated store. You act only by calling the supplied tools. Every store tool call uses one step of a limited budget. The step count decides when to finish.

The task scores 1 only if the finish is fully right, otherwise 0:
- the order holds exactly the wanted product(s), each in the wanted quantity, and every line meets every requirement: the ones stated in the request, the hidden ones the shopper revealed when asked, and every change the shopper announced later (a dropped requirement no longer counts; a new budget replaces the old one);
- the total is within the budget; when the request asks for the cheapest option, the order is the cheapest listing that meets everything, so every candidate must be checked, not only the first page;
- ordered items are in stock and can be bought;
- every procedure the request states is followed, including how to decline.
If nothing in the store meets every requirement after a thorough search, decline with the decline tool. A wrong decline scores 0 and most declines turn out wrong, so decline only after several different searches and after inspecting the plausible listings.

How to work:
1. Search with short queries of 1-4 distinctive words; long queries often return nothing. When the store has a tool that lists a category (HOW TO CALL THE KEY TOOLS: "browse"), list each wanted product's category with it first and page through it: keyword search misses many rows that the category list shows. Also try the product's plain name, synonyms and the search filters the tool offers (category or department, maker, price cap). Read the result count; page through every page when you need the cheapest option or when the results are cut off. When a store's result pages are short, the system fetches the next pages of each search for you (about 12 rows), so read all of them before searching again. A price-cap filter set just below a candidate's price is a quick way to look for anything cheaper.
2. The kind of product is the store's published category label, copied exactly from a tool's allowed values or from a result row. A neighbouring label does not count. Until a published label has been seen, search with the shopper's words, then switch to the exact label.
3. Requirements come in four forms: "property: value" (must have), "property: NOT value" (exclusion: a listing whose property states that value is out; a listing without that property is fine), "property: at least / at most N unit" (compare numbers after converting units; the edge counts), "property: one of A, B" (any listed value is fine, nothing else).
4. Inspect each plausible listing in full (its property record) before relying on it; do not reopen what the STATE already shows; titles mislead and the property record is what counts. A requirement is met when a property for it states the value: any key spelling, singular or plural, one element of a list, or a close synonym of the value counts, a more specific value of the same kind meets it ("forest green" for green). A blend or a second material does not meet the value alone ("80% cotton, 20% polyester" does not meet cotton). A different value or a broader class does not. When a listing has no property for a requirement, its title and description may settle it. When several keys could describe the property and they disagree (for example a plain key and a prefixed or nested key for the same property), prefer a listing where they agree or where the key named exactly like the requirement states the value; a listing where only one fitting key states the value still counts and is better than declining. Compare numbers after converting units; limits include the edge.
5. Text beats the record when it says so: a listing's description or notes may correct a property, or say that a property's true value is the one shown on a DIFFERENT listing (identified by its maker, seller, kind or other details), however that is worded. The STATE shows what such notes say. Then the other listing's value is the one that counts: find that listing with the details given, inspect it and use its value; follow chains. A need the shopper reveals may likewise point to another listing instead of naming a value: find that listing, read the named property and record the concrete value with the note_to_self tool. When the store has a separate tool that shows a listing's page (HOW TO CALL THE KEY TOOLS: "listing"), its notes appear only there: open the page of every listing you may order and of every listing a request, a shopper reply or a note points to, and when a page says to take the value from yet another listing, open that one's page too, until a page states the value. The store checks that the pages of such a chain were opened.
6. Store rules (for example that two values count as the same) apply when matching requirements, wherever they appear: the request, the opening, tool results or listing notes. When one wanted item must fit or go with another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code); if no accessory fits the chosen device, try another device before declining. When the request asks for the most or least of a measure (the lightest, the largest, the longest battery life), the order is the listing with the extreme value among those that meet everything.
7. The system plans and sends the questions to the shopper for you once you have inspected a few plausible listings (about six), and before anything goes into the cart; you will see them in the history. A reply of "no preference" or "I don't follow / say it again" means the shopper has no need on that property. The system writes every message to the shopper (questions, and a basket summary when a rule asks for one); you never message the shopper. Only needs the shopper stated count as known: an order that merely happens to fit an unstated need still fails.
8. Add the chosen item(s) to the cart in the right quantity, then place the order. The system checks the cart and the choice before the order goes out and tells you if something is wrong. If the order is refused (stock-out, price change, new budget, dropped requirement), read why, fix the cart or the choice, and order again.
9. Things change while you work: read SHOPPER UPDATES and TODO in the state.
10. note_to_self is free (no step): use it to record a resolved requirement value or a fact you will need later.
Each store words its tools, listings and rules its own way, and the shopper may act in ways these notes do not describe. When something is new, go back to what scores 1 above and act on that.

Keep your reasoning to a few short sentences, then make exactly one tool call."""

SETUP_PROMPT = """You prepare a shopping assistant for one task. Read the shopper's request and the store's tool list, then return ONE JSON object and nothing else, with these keys:
"products": list of {"label": the product's own name in 1-3 plain words, never the word "item", "category": the shopper's words for the kind, whole (keep every part of a phrase like "A or B" or "A and B"); the code replaces this with a category label the store itself publishes, copied exactly, "quantity": units wanted (integer), "requirements": list of every stated requirement on the product itself, each in one of these forms: "property: value" (must have), "property: NOT value" (an exclusion such as "cross off anything with ..." or "no ..."), "property: at least N unit" / "property: at most N unit" (a numeric limit such as "N inches or bigger"), "property: one of A, B" (a list such as "A or B, nothing else"); use the property name the request uses, "cheapest": true only if the request asks for the cheapest / lowest price option of it}. A product the request names in one phrase with one quantity (often one store department, even if the phrase says "and") is ONE product. Separate products only when each has its own quantity or its own requirements
"budget": the spending limit as a number in the store's price units, or null if none is stated
"budget_scope": "total" or "each"
"in_stock_required": true or false
"message_budget": how many messages the shopper allows (integer) or null
"step_budget": how many moves / steps / actions the shopper allows (integer) or null
"decline_rule": what the request says to do or say when nothing fits, close to its wording, or ""
"rules": list of every other procedure the request states, close to its wording
"reserve_messages": how many messages the rules force the assistant to send (for example "tell me before you buy" or "let me approve the final basket"), else 0
"ask_first": things the request explicitly says to check with the shopper before choosing, each {"product": label, "property": the property it is about}, else []
"pick": "cheapest" when the request asks for the lowest price, "most" or "least" when it asks for the extreme of a measure, else "any". Tight money is not a pick rule
"pick_measure": the measure maximised or minimised for "most" or "least", else ""
"process": steps the request requires besides finding and buying (approval, telling the shopper something, reading something first, using a particular tool), each {"step": what must be done in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checkout) or "end" (just before ending without a purchase), "messages": how many shopper messages it takes, 0 when none}. Requirements, the budget, stock, message and step limits, and how to decline are not steps. [] when there are none
"fits": when one wanted product must work with or fit another wanted product, each {"part": the accessory's label, "device": the other product's label}, else []
"hints": the story details that point at a property of a wanted product without stating its value, each {"product": label, "property": the property as store listings would likely name it, "detail": the detail in a few words}, likeliest first: what the item must match or suit, how it must look, what it must fit, who it is for, how it must feel, where or how it is used, or what about it is handled again and again. Weather, a job, a schedule or a place where the item is not used is background, never a hint; so is a purpose every listing of the kind serves, and a detail that only explains a stated requirement. [] when none
"calls": object describing how to call the store tools, using the placeholders <ID>, <QTY> and <TEXT>:
  "search": {"tool": name, "args": {"<free-text param>": "<TEXT>", ...fixed args needed for a keyword search}},
  "browse": {"tool": name of a tool that lists the listings of one category, section or department (the search tool itself when a mode argument makes it list a category), "args": {"<category param>": "<CATEGORY>", ...fixed args such as that mode}} or null,
  "details": {"tool": name of the tool that returns a listing's full property record, "args": {"<id param>": "<ID>"}},
  "listing": {"tool": name of another tool that shows one listing (or null), "args": {"<id param>": "<ID>"}},
  "cart_add": {"tool": name, "args": {"<id param>": "<ID>", "<quantity param>": "<QTY>", ...fixed args}},
  "cart_remove": {"tool": name, "args": {...}} or null,
  "cart_view": {"tool": name, "args": {}} or null,
  "message": {"tool": name, "args": {"<text param>": "<TEXT>"}} or null,
  "order": {"tool": name, "args": {...fixed args}},
  "decline": {"tool": name, "args": {"<remark param>": "<TEXT>", ...fixed args}}
Requirements cover what the request demands of the product itself (kind, properties, size, material, maker, condition, ...), not the budget, stock or procedures. Who asked, why the money is tight, and the shopper's schedule are not requirements. A named occasion or style the product itself is for is a requirement: "for a wedding" is "occasion: wedding", and "military-style" is "style: military". Never add a value the request does not contain. Words like "fitting", "suitable" or "one that works" just mean "meets the requirements" and are not requirements themselves. A store rule given in the opening or by the store (for example that two values count as the same) is how this store matches values: list it in "rules" as the store states it, and never write a rule that contradicts or overrides it; a value the store rule counts as equal is not a "substitute". Fixed args (such as a mode or route value) must be valid for the tool's schema. If one tool both orders and declines through an argument, use it for both with the right values. Use null for a call the store does not offer."""

INTERVIEW_PROMPT = """You choose the questions a shopping assistant sends to a shopper. The shopper holds back a few firm needs, each on one property of one wanted product. A question reveals a need only when it names that property, and messages are few, so the order matters.

You get the request and, for each wanted product, a numbered list of properties seen on its opened listings, widest coverage first: how many of those listings carry it, and example values. An unwritten need may be on a property no listed key names.

Choose in this order:
1. A property the request itself says to check with the shopper.
2. Hinted properties: the story often points at a hidden need without giving its value (who it is for or who will wear it, how it must look or what it must match, where or how it is used, what it must fit). Name it by a listed property of that product when one fits the hint. A hinted property that no listing carries comes after that product's three most carried properties.
3. Then the listed properties: favour the ones a buyer of that kind of product chooses between (its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind is bought by) and the ones carried by most of its listings. Once five or more listings were read, a property seen on only one or two of them comes after the widely carried ones. A property whose values bundle several features is asked last.
Never choose a second name for a property already chosen (keys that differ only by a prefix, a code or the product's own kind word name one property), a property the request already settles, a list of purchase variants, identifiers or model codes, or warranty, delivery, service and other terms of sale.
With several products, give each product its first pick before any product gets a second, then take turns, best first.
An unwritten need may also be on a property no opened listing shows: a hinted property missing from its product's list, and, once the listed properties worth asking run out, the properties buyers of that kind of product commonly choose by, may be named as {"pick": 0, "product": "the product", "label": "its plain name", "why": "..."}.
Return ONE JSON object and nothing else: {"ask": [{"pick": the property's number, or 0 for an unlisted property, "why": "a few words"}]} with COUNT picks when there are that many worth asking, best first."""

PLAN_PROMPT = """You choose the next question a shopping assistant sends to a shopper.
How this shopper behaves:
- The shopper may hold back hidden needs. Each hidden need is about one property of a wanted product; the property names are the ones in the store's property records.
- The shopper reveals a hidden need only when a message asks about that property, by its name, a close synonym or an example value. Questions about any other property get a non-answer: "no preference", or, from some shoppers, "I don't follow / say it again". Both mean the same thing: there is no hidden need on that property. Rewording does not help, so never ask about that property again.
- A reply asking for "one question at a time" usually means the message touched a property with a hidden need: the system then repeats the first property alone, so you do not need to.
- Messages are few, so each one should test the property most likely to hide a need. An order that happens to meet a hidden need the shopper never stated still fails, so every wanted product needs its hidden needs asked out of the shopper.

Choose one product and up to eight of its property names, the likeliest first. Use the property names from the records, in plain words (drop codes and prefixes). When several products are wanted, ask about the products with no revealed need yet first, and spread the questions so every product gets asked. Prefer properties that:
(a) the request does not already settle;
(b) appear in most records of the product's kind (core properties of that kind of product), including ones where most candidates share one value: a hidden need usually asks for an uncommon value, so it rules most candidates out;
(c) are about personal fit or taste for whoever the item is for (who it is made for, its size, what it is made of, how it looks), or are hinted at by the story in the request.
Only a few records may have been inspected, so you may also ask about a core property this kind of product commonly has even if it is not in the records yet.
Spread the questions over different kinds of property rather than near-duplicates of one (two keys for the same thing count as one).
Prefer properties whose values differ among the candidate listings; administrative fields (codes, identifiers, terms of sale) hide none.
Never ask about price, budget, quantity, stock or delivery. Never repeat a property already asked for the same product, in any wording.
Return ONE JSON object: {"product": label, "properties": [name, ...], "why": short reason}. Return {"wait": true} if another wanted product still has no property records and should be looked at first. Messages left unused are wasted, so always pick a property while messages remain."""

INTERPRET_PROMPT = """You help a shopping assistant understand what the shopper just said. Read the shopper's words with the context below and return ONE JSON object:
"reply_type": one of "reveal" (states a requirement or preference), "no_preference" (does not mind / any is fine), "not_understood" (says it did not follow or understand the message, or asks for it again or in simpler words; this shopper's way of saying it has no need there, so it is a non-answer too), "one_at_a_time" (objects to how many things the message asked about at once, however it is worded: wants the questions split up, spaced out, taken in turn or one per message, or only the first answered now; a reply that only says it did not understand is "not_understood"), "update" (announces a change: new budget, dropped requirement, item gone, price changed), "other"
"add": list of {"product": label of the wanted product it is about, "requirement": "property: value", kept close to the shopper's words; if, instead of naming a value, it says the wanted value is the one on another listing (however it is worded), copy that reference in full with every detail that identifies the listing, and keep its sense: "property: NOT the value on <reference>" when the shopper wants anything except that listing's value ("anything bar / but / other than what's written on ..."), "property: the value on <reference>" when it wants the same, "lookup": true if it points to another listing}. The details that pick out the other listing ("the page for X's sneakers, with closure slip-on and color red") only identify it: they belong inside the reference and are never requirements of their own, even when the message asked about those properties too
"drop": list of {"product": label, "property": the property the shopper no longer requires}; words like "<property> isn't worth the trouble anymore, so skip it", "forget the <property>" or "<property> doesn't matter now" drop that property only: the shopper still wants the product, and such words never cancel the purchase
"budget": the new spending limit as the plain number the shopper wrote, in the units the shopper used, without converting or rescaling, or null if unchanged
"stock_out": true if the chosen item is no longer available
"price_changed": true if the chosen item's price changed
"instructions": list of procedures the words add for the rest of the task, close to their wording (for example "confirm the basket with me before you buy", "tell me the total before ordering", "say why if you give up"); [] when none. Requirements on a product, budgets and stock news go in the keys above, not here
"messages_total": the new total number of messages the assistant may send over the whole task, if the words state one, else null
"messages_left": how many more messages may be sent from now on, if the words state that ("only 2 more messages"), else null
"steps_total": the new total number of steps / moves / actions allowed over the whole task, if the words state one, else null
"steps_left": how many more steps / moves / actions are allowed from now on, if the words state that ("closes in 10 more steps"), else null
"summary": one short line for the assistant's notes"""

VERIFY_PROMPT = """You are the final checker before a shopping assistant places an order. The task scores 1 only if every point holds:
1. The cart holds one line for each wanted product (no extra products), with the wanted quantity.
2. Each line's listing is in the store's published category for its product (the exact label, not a neighbouring one), and meets every active requirement of its product. Read the listing's records: a requirement is met when the listing's own property for it states the value (any key spelling, singular or plural, one element of a list, or a close synonym counts; a different value or a broader class does not); when the listing has no property for it, the title or description may settle it; the key that names that property decides, and a neighboring key (a handle colour is not the body's colour) decides nothing; when that key states a different value the listing fails. A range must lie wholly inside the limit. A requirement no record addresses makes ok false. A missing measure on a most or least pick is a hard fail. Numbers are compared after unit conversion, limits include the edge. Requirement forms: "property: value" must be stated; "property: NOT value" fails, and is a hard fail, when the listing's property states that value; "at least / at most N" compares the listing's number; "one of A, B" accepts any listed value. If a listing's notes say a property must be read from another listing, the other listing's value counts: use it if it is in the records or notes, otherwise that requirement is UNVERIFIED. A requirement that points to another listing is met only if that listing's value is known and matches.
3. The total (price x quantity, summed) is within the current budget, and each item is in stock.
4. If a product must be the cheapest option (or the most or least of a measure the request names), no cheaper (or more extreme) listing among the other listings seen could meet all its requirements. A cheaper listing that has not been inspected, or whose records leave a requirement open, must be checked first.
5. The stated procedures are followed. Store rules (for example that two values count as the same) are part of matching: a listing whose value the store rule counts as equal to the wanted one meets the requirement, and is not a substitute. A stock-out, a refused order or a dropped requirement never cancels the request: another listing that meets every requirement in force replaces the gone one, and is not the "substitute" a decline rule forbids (that means a listing that misses a requirement).
Judge only from the records; do not assume anything that is not shown. The listing's category or department counts toward what kind of product it is (a listing in a category named after the wanted product is that kind of product). Accept close variants of a value (a synonym, or the same value worded or shaped a little differently) and a more specific value of the same kind ("forest green" for green). A blend or a second material fails the value alone ("80% cotton, 20% polyester" does not meet cotton), and that is a hard fail when the record states the other material. Reject real differences. Name better listing ids in check_first. Return ONE JSON object: {"ok": true or false, "hard_fail": true only if the records clearly contradict a requirement (the listing's own property for it states a different value, including a blend where the value alone was required, or an excluded value), or the measure is missing on a most or least pick, or the price is over budget, or the item is shown out of stock; other missing or unclear information is not a hard fail, "problems": [short concrete problems], "check_first": [ids of listings to inspect or compare before ordering]}"""

SWAP_PROMPT = """A cart line just sold out. Pick a replacement that meets that product's requirements from the listings below. Return ONE JSON object: {"id": one of the listing ids, or "" when none fits}."""

GATE_CALL_PROMPT = """A step is due before this stage. A step counts as done only when you cite a RECORD line that shows it, and for a step about the cart that line must be after the cart last changed. Return ONE JSON object: {"steps": [{"step": the step, "done": true or false, "lines": [RECORD line numbers that show it]}], "call": {"tool": the exact tool name from TOOLS, "args": {}} or null when every due step is done}. A message to the shopper is that tool with the sentence in its text argument. Never name the order or the decline tool."""

UNMIX_PROMPT = """A product's requirements were read from one request. POINTERS take their value from a different listing. Words that only pick out that other listing (its kind, its seller, a property used to find it) are sometimes stored as if they were requirements of the product being bought. Using the request's wording, say which OTHER lines only describe that other listing. Return ONE JSON object: {"only_the_other": [those lines copied exactly], or [] when every other line is a real requirement of the product being bought}."""

SHAPE_PROMPT = """A store tool returned the result below. Name each part by the exact key the result or the tool's parameters use. Return ONE JSON object and nothing else:
"rows": the key of the list of listings
"count": the key of how many listings matched in all (not how many this page shows), or null
"next": the key of the value that asks for the following page, or null
"next_arg": the parameter that takes that value, or null
"price": the key, inside one listing, of its price, or null
"stock": the key, inside one listing, that says whether it can be bought now, or null"""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Declining wrongly scores 0, so check the listings seen against the requirements below, reading them like this: the listing must be in the store category named like the request's wording for the product; a requirement is met when the listing's property for it states the value (any key spelling, singular or plural, one element of a list or a close synonym counts, and a more specific value of the same kind meets it ("forest green" for green). A blend or a second material does not meet the value alone); the title or description counts when there is no such property; numbers are compared after unit conversion, limits include the edge; the price must be within budget and the item must not be out of stock. Only the listed requirements count.
Return ONE JSON object: {"may_fit": [ids of listings that meet or may meet everything, or that were never inspected but look like the wanted product], "search_more": [up to 3 short search queries that have not been tried and could find the product], "reason": short}"""

NOTES_PROMPT = """You read one store listing's description and notes for a shopping assistant. Report only what changes how the listing's properties must be read, however it is worded. Return ONE JSON object:
"defers": list of {"property": exactly one property name, "source": the other listing it must be read from, with every detail the text gives to find it (maker, seller, kind, title words, ...)}, for each property whose true value the text says is the one on a different listing;
"corrections": list of "property: value", for each property value the text states that differs from, or is missing in, the property record;
"rules": list of store rules the text states that apply beyond this listing (for example that two values count as the same).
A reference may be worded in many ways ("for the X, see ...", "take the X from ...", "X as on ...", "look to ... for X"); the property is the X it names, and the source is the other listing as the text describes it. For example, "For the strap colour, refer to the matching belt by Acme" gives {"property": "strap colour", "source": "the matching belt by Acme"}. Ordinary product description is none of these. Return empty lists when there is nothing of the kind."""

_ID_KEY = re.compile(r"(^|[_\s])(id|pid|sku|code|ref)$|^id$|_id$|^id_|item_?pid|product_?(id|code)|listing_?(id|code)|target", re.I)
_PRICE_KEY = re.compile(r"price|cost|amount", re.I)
_TOTAL_KEY = re.compile(r"total|sum", re.I)
_TITLE_KEY = re.compile(r"title|name|caption|heading|headline|label", re.I)
_STOCK_KEY = re.compile(r"stock|avail|can_?buy|buyable|purchasable|orderable|sold_?out|in_?stock|on_?hand", re.I)


_QTY_KEY = re.compile(r"qty|quantity|count|units|how_?many|number|amount", re.I)
_CATEGORY_KEY = re.compile(r"categor|department|shelf|aisle|section|^kind$|^type$", re.I)
_COUNT_KEY = re.compile(r"count|total|matched|hits|found|results?_?n", re.I)
_MORE_KEY = re.compile(r"next|continue|after|more|cursor|remaining|onward|follow|token", re.I)
_TOKEN_KEY = re.compile(r"next|continue|cursor|onward|after|follow|token", re.I)  # a value that asks for the next page
_PAGE_PARAM = re.compile(r"next|continue|cursor|page|offset|start|after|onward|token|skip|begin", re.I)
_LONG_TEXT_KEY = re.compile(r"descr|summary|about|note|know|detail|text|overview|blurb|info", re.I)
_REFERENCE = re.compile(r"\b(see|refer(s|red)? to|look to|go by|instead of|rather than|taken from|same as|comes? from|as (shown|stated|listed) on)\b", re.I)

_VOICE = re.compile(r"\b(I|I'm|I'll|I've|I'd|my|me|mine|you|your|yours)\b")
_ID_TOKEN = re.compile(r"\b[A-Za-z]{1,6}[-_][A-Za-z0-9]{5,40}\b")
_FINAL_WORDS = re.compile(r"\bfinal\b|no more actions|accepts no more|ends the|close out|closes the", re.I)
_ORDER_WORDS = re.compile(r"purchase|check ?out|order|buy|pay|settle|place", re.I)
_DECLINE_WORDS = re.compile(r"no purchase|without (buying|purchas)|give up|decline|abandon|abstain|refuse|cancel|nothing fits", re.I)
_SYSTEM_TEXT = re.compile(r"^(error|invalid|unknown|missing|expected|must be|not found)\b|https?://", re.I)
_APPROVAL = re.compile(
    r"approv|confirm|check with (me|the shopper)|sign off|okay the|run (it|the basket) by (me|the shopper)|before (you )?(buy|order|pay|check ?out|place)|"
    r"(tell|show) (me|the shopper) the (total|basket)",
    re.I,
)


# ---------------------------------------------------------------- which property to ask
# Order comes from this task's opened listings: a story hint first, then the property the most listings carry.
# Nothing here is fit on past episodes.
_CATCH_ALL = re.compile(
    r"^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b",
    re.I,
)
_MEASURE_VALUE = re.compile(r"^\s*[\d.,/x× -]+\s*[a-z]{0,6}\s*$", re.I)
_COVERAGE_GAP = 0.34
_TIGHT_MESSAGES = 2

_STEM_STOP = {"the", "and", "for", "you", "your", "would", "like", "which", "what", "have", "want", "with", "any"}


def _is_stock_key(key: str) -> bool:
    # a stock field; "available <something>" with other words is a property list ("available sizes"), not stock
    if not _STOCK_KEY.search(key):
        return False
    if re.search(r"stock|buy|sold|order|purchas|sale|qty|quant|unit|count", key, re.I):
        return True
    rest = [w for w in re.split(r"[^a-z]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", key).lower())
            if w and not re.fullmatch(r"avail(able|ability)?|is|now|on|in|hand|status|flag|state|note|info|text", w)]
    return not rest


def _not_a_need(key: str, values: list[str] | None = None, row_keys: set[str] | frozenset = frozenset()) -> bool:
    # a record field that is no shopper's need: listing metadata (the fields search rows carry, ids, prices, titles,
    # stock, category, free text), or values that are codes (one per listing), long text or lists of variants
    low = _norm_key(key)
    if not low or low in {_norm_key(k) for k in row_keys}:
        return True
    if re.fullmatch(r"(options?|variants?|variations?|choices?|selections?)( available)?", low):
        return True  # the listing's variant list ("Options: 6pcs"), not a property of the product
    # logistics are not a shopper's hidden need; asking them spends the message budget
    if re.search(r"\b(warrant\w*|shipping|delivery|postage|return|refund|seller|barcode|asin|upc)\b", low):
        return True
    if _ID_KEY.search(key) or _PRICE_KEY.search(key) or _TITLE_KEY.search(key) or _is_stock_key(key) \
            or _CATEGORY_KEY.search(key) or _LONG_TEXT_KEY.search(key):
        return True
    vals = [str(v).strip() for v in values or [] if str(v).strip()]
    if vals:
        if sum(len(v.split()) for v in vals) / len(vals) > 8:
            return True
        if len(vals) >= 3 and len(set(vals)) == len(vals) and all(re.search(r"\d", v) and " " not in v for v in vals):
            return True
        if sum(1 for v in vals if v.count(",") + v.count("|") + v.count(";") >= 2) * 2 > len(vals):
            return True
    return False


def _stems(text: str) -> set[str]:
    out = set()
    for word in re.findall(r"[a-z]+", str(text).lower()):
        if word in _STEM_STOP or len(word) < 3:
            continue
        for suffix in ("ies", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
                break
        out.add(word)
    return out


def _base(key: str) -> str:
    # one name per kind of property: the last two words of the normalised key, plurals folded
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in _norm_key(key).split()]
    return " ".join(words[-2:]) if words else str(key)


_GENERIC_KEY_WORDS = {"of", "the"}


def _fold_words(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in _norm_key(text).split())


def _concept(key: str, context: frozenset[str] = frozenset()) -> str:
    # the property a key names, without generic words and plurals: "types_of_sweeteners" and "sweetener type" are one;
    # without a leading code ("fa_pattern" is "pattern") or the product's own kind words ("clothing material" for clothing)
    words = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in _norm_key(key).split()]
    while len(words) > 1 and len(words[0]) <= 2:
        words = words[1:]
    plain = sorted({w for w in words if w not in _GENERIC_KEY_WORDS})
    core = sorted(w for w in plain if w not in context)
    return " ".join(core) or " ".join(plain) or " ".join(sorted(set(words))) or str(key)


def _key_groups(keys: Any, context: frozenset[str] = frozenset()) -> dict[str, list[str]]:
    # key -> every key naming the same property (same concept or same base), the key itself first
    keys = list(dict.fromkeys(keys))
    parent = {k: k for k in keys}

    def root(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for index, a in enumerate(keys):
        for b in keys[index + 1:]:
            if _concept(a, context) == _concept(b, context) or _base(a) == _base(b):
                parent[root(b)] = root(a)
    members: dict[str, list[str]] = {}
    for k in keys:
        members.setdefault(root(k), []).append(k)
    return {k: [k] + [m for m in members[root(k)] if m != k] for k in keys}


def _plain_key(keys: list[str]) -> str:
    # the plainest spelling among keys naming one property: fewest words, then shortest
    return min(keys, key=lambda k: (len(_norm_key(k).split()), len(k), k))


def _mostly_measured(values: list[str]) -> bool:
    return bool(values) and sum(1 for value in values if _MEASURE_VALUE.match(value)) * 2 > len(values)


def _rank_properties_scored(category: str, key_values: dict[str, dict[str, str]], n_listings: int) -> list[tuple[str, float]]:
    # (key, share of opened listings that carry it), widest first. The share is coverage of this task, not a chance
    # learned from past episodes. A key seen on one or two listings, once five or more were opened, comes after the
    # widely carried ones. A key that bundles several features comes last.
    groups = _key_groups(key_values, _fold_words(category))
    best: dict[str, tuple[float, str]] = {}
    merged: dict[str, dict[str, str]] = {}
    for key in key_values:
        group = groups[key]
        head = min(group)
        if head in merged:
            continue
        values: dict[str, str] = {}
        for member in sorted(group, key=lambda m: -len(key_values[m])):
            for ident, value in key_values[member].items():
                values.setdefault(ident, value)
        merged[head] = values
        widest = max(group, key=lambda m: (len(key_values[m]), -len(m)))
        best[head] = (len(values) / max(1, n_listings), widest)
    ranked = []
    for _head, (share, key) in best.items():
        vals = list(merged[_head].values())
        rare = 1 if n_listings >= 5 and len(merged[_head]) <= 2 else 0
        catch = 1 if _CATCH_ALL.search(_norm_key(key)) else 0
        ranked.append((catch, rare, -share, _mostly_measured(vals), -len(set(vals)), len(key), key, share))
    ranked.sort()
    return [(key, share) for *_, key, share in ranked]


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _brief(value: Any, text: int = 600, items: int = 30, depth: int = 0) -> Any:
    if isinstance(value, str):
        return value if len(value) <= text else value[:text] + "..."
    if isinstance(value, list):
        out = [_brief(v, text, items, depth + 1) for v in value[:items]]
        if len(value) > items:
            out.append(f"... {len(value) - items} more")
        return out
    if isinstance(value, dict):
        if depth > 6:
            return "{...}"
        return {str(k): _brief(v, text, items, depth + 1) for k, v in value.items()}
    return value


def _parse_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start : index + 1]
                    for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", re.sub(r"//[^\n\"]*\n", "\n", chunk))):
                        try:
                            parsed = json.loads(candidate)
                        except ValueError:
                            continue
                        if isinstance(parsed, dict):
                            return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number >= 0 else None


def _norm_key(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"\s+", " ", re.sub(r"[_\-.]+", " ", key)).strip().lower()


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", _norm_key(text)) if w}


def _model_codes(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", code).strip().lower() for code in re.findall(r"\b[A-Z][A-Za-z]{0,5} ?\d{3,6}\b", text)]


def _strings(value: Any, out: list[str], skip_listings: bool, depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            _strings(item, out, skip_listings, depth + 1)
    elif isinstance(value, dict):
        if skip_listings and _id_field(value):
            return
        for item in value.values():
            _strings(item, out, skip_listings, depth + 1)


def _id_field(record: dict[str, Any]) -> str | None:
    # The key holding a listing id, when the dict looks like a listing.
    id_key = None
    for key, value in record.items():
        if isinstance(value, (str, int)) and not isinstance(value, bool) and _ID_KEY.search(str(key)):
            if str(value).strip() and len(str(value)) <= 64:
                id_key = key
                break
    if id_key is None:
        return None
    for key, value in record.items():
        if key == id_key:
            continue
        if _PRICE_KEY.search(str(key)) or _TITLE_KEY.search(str(key)) or isinstance(value, dict):
            return id_key
    return None


def _limit_bound(value: str) -> tuple[str, float, str] | None:
    # "at most 101 cm" -> ("le", 101, "cm"). No unit conversion: a different unit is not this limit.
    text = value.strip().lower()
    match = re.match(r"(at least|at most|no less than|no more than|under|over|minimum|maximum|min|max)\s+(.+)", text)
    if not match:
        return None
    word, rest = match.group(1), match.group(2).strip()
    number = _number(rest)
    if number is None:
        return None
    unit_match = re.search(r"-?\d+(?:\.\d+)?\s*([a-z]{1,3})\b", rest.replace(",", ""))
    unit = unit_match.group(1) if unit_match else ""
    if word in ("at least", "no less than", "minimum", "min"):
        op = "ge"
    elif word == "over":
        op = "gt"
    elif word == "under":
        op = "lt"
    else:
        op = "le"
    return op, number, unit


def _inside_limit(op: str, have: float, limit: float) -> bool:
    if op == "ge":
        return have >= limit
    if op == "gt":
        return have > limit
    if op == "lt":
        return have < limit
    return have <= limit


_STORY_OCCASION = {
    "class", "fall", "winter", "summer", "spring", "campus", "today", "tonight",
    "week", "month", "year", "home", "work", "school", "morning", "evening",
    "night", "weekend", "semester",
}
_PROP_FILLER = {"type", "style", "kind", "general", "class", "family", "filter", "option"}


def _req_options(value: str) -> tuple[str, list[str]]:
    # "one of A, B" and a parenthetical ("none (sleeveless)") are alternatives. A limit stays a limit.
    if _limit_bound(value):
        return "limit", [value.strip()]
    form, wanted = _requirement_form(value)
    extras = [piece.strip() for piece in re.findall(r"\(([^)]+)\)", value) if piece.strip()]
    if form == "must" and extras:
        main = re.sub(r"\([^)]*\)", " ", wanted[0]).strip()
        options = [piece for piece in [main, *extras] if piece]
        return "one_of", options or wanted
    return form, wanted


def _content_words(text: str) -> set[str]:
    words = set()
    for word in _norm_key(text).replace(",", " ").split():
        word = re.sub(r"[^a-z]", "", word)
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) >= 3:
            words.add(word)
    return words


def _prop_stems(prop: str) -> set[str]:
    words = _content_words(prop)
    specific = {word for word in words if word not in _PROP_FILLER}
    return specific or words


def _matching_keys(props: dict[str, str], prop: str) -> list[str]:
    # The property key that names this requirement. A key with many extra words is a neighbour and does not decide.
    want = _prop_stems(prop)
    if not want:
        return []
    hits = []
    for key in props:
        have = _content_words(key)
        if want <= have and len(have - want) <= 2:
            hits.append((len(have - want), key))
    hits.sort()
    return [key for _extra, key in hits]


_NONE_WORDS = {"none", "no", "without", "nil", "n/a", "not applicable", "0", "zero"}


def _means_none(stated: str) -> bool:
    # a value that says there is none of the property: "none", "no", "without ...", "sleeveless"-style "-less" words
    low = stated.strip().lower()
    return low in _NONE_WORDS or bool(re.match(r"(no|without|non)\b", low)) or bool(re.search(r"[a-z]{3,}less\b", low))


def _value_hit(wanted: str, stated: str) -> bool:
    if wanted.strip().lower() in _NONE_WORDS:
        return _means_none(stated)
    numbers = set(re.findall(r"\d+(?:\.\d+)?", wanted))
    if numbers and not numbers <= set(re.findall(r"\d+(?:\.\d+)?", stated)):
        return False
    need = {word for word in _content_words(wanted) if word not in {"one", "none"}}
    if not need:
        return bool(numbers)
    # One word matches a short value with at most one extra word ("wrap skirts", "forest green").
    # A short phrase with more extras ("chiffon net yarn") is a different value.
    # A long string such as a title matches when the word is in it.
    if len(need) == 1:
        word = next(iter(need))
        for element in re.split(r"[,;/|]", stated):
            have = _content_words(element)
            extra = len(have - {word})
            if word in have and (extra <= 1 or len(have) > 4):
                return True
        return False
    return need <= _content_words(stated)


def _value_present(wanted: str, stated: str) -> bool:
    # The loose reading used before blocking an order: the wanted value's numbers and words all appear in the stated
    # value ("thermoplastic polyurethane (tpu)" holds tpu). _value_hit, which picks a line, stays strict.
    if wanted.strip().lower() in _NONE_WORDS:
        return _means_none(stated)
    numbers = set(re.findall(r"\d+(?:\.\d+)?", wanted))
    if numbers and not numbers <= set(re.findall(r"\d+(?:\.\d+)?", stated)):
        return False
    need = {word for word in _content_words(wanted) if word not in {"one", "none"}}
    if not need:
        return bool(numbers)
    return need <= _content_words(stated) or need <= _stems(stated)


def _value_in_request(value: str, stems: set[str], request: str) -> bool:
    bound = _limit_bound(value)
    if bound:
        _op, number, unit = bound
        shown = set(re.findall(r"\d+(?:\.\d+)?", request))
        if str(int(number)) not in shown and f"{number:g}" not in shown:
            return False
        return not unit or unit in request
    _form, options = _req_options(value)
    for option in options:
        need = {stem for stem in _stems(option) if stem not in {"one", "none"}}
        if need and need <= stems:
            return True
        numbers = set(re.findall(r"\d+(?:\.\d+)?", option))
        if numbers and numbers <= set(re.findall(r"\d+(?:\.\d+)?", request)):
            return True
    return False


def _requirement_form(value: str) -> tuple[str, list[str]]:
    # Split a requirement value into its form (must, not, at_least, at_most, one_of) and wanted values.
    value = value.strip().lower()
    if value.startswith("not "):
        return "not", [value[4:].strip()]
    match = re.match(r"(at least|at most|minimum|maximum|min|max)\s+(.+)", value)
    if match:
        return ("at_least" if match.group(1) in ("at least", "minimum", "min") else "at_most"), [match.group(2).strip()]
    if value.startswith("one of "):
        return "one_of", [v.strip() for v in re.split(r",|\bor\b", value[7:]) if v.strip()]
    return "must", [value]


def _form_check(form: str, wanted: list[str], related: list[str], text: str) -> str:
    # A rough code-side reading of one requirement against a listing; the checker model decides.
    if form == "not":
        bad = wanted[0]
        if any(bad in value for value in related):
            return f"EXCLUDED value '{bad}' is stated"
        return "excluded value not stated"
    if form in ("at_least", "at_most"):
        limit = _number(wanted[0])
        numbers = [n for n in (_number(v) for v in related) if n is not None]
        if limit is None or not numbers:
            return "no number to compare"
        ok = any(n >= limit for n in numbers) if form == "at_least" else any(n <= limit for n in numbers)
        return f"numbers {numbers[:4]} vs {form.replace('_', ' ')} {limit:g}: {'within' if ok else 'OUTSIDE'} (check units)"
    if form == "one_of":
        hit = [w for w in wanted if any(w in value for value in related) or w in text]
        return f"listed value found: {hit[0]}" if hit else "no listed value found"
    return "value text in record: " + ("yes" if wanted and wanted[0][:40] in text else "no")


def _failed(observation: Any) -> bool:
    # True when a store result reports that the call did not go through.
    if isinstance(observation, str):
        return bool(re.match(r"\s*(error|invalid|unknown|not allowed|refused|failed)", observation, re.I))
    if not isinstance(observation, dict):
        return False
    for key, value in observation.items():
        if re.search(r"^(error|errors|failure|problem)s?$", str(key), re.I) and value:
            return True
        if isinstance(value, bool) and not value and re.search(r"ok|success|applied|went|done|accepted|added", str(key), re.I):
            return True
    return False


def _main_record(observation: Any, ident: str | None) -> Any:
    # The listing record inside a details observation, unwrapping any envelope around it.
    if not isinstance(observation, dict):
        return observation
    if _id_field(observation):
        return observation
    found: list[dict[str, Any]] = []

    def visit(value: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(value, dict):
            key = _id_field(value)
            if key is not None:
                found.append(value)
                return
            for item in value.values():
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1)

    visit(observation, 0)
    for record in found:
        if ident is not None and str(record.get(_id_field(record) or "")) == str(ident):
            return record
    if found:
        return found[0]
    dicts = [v for v in observation.values() if isinstance(v, dict)]
    return max(dicts, key=lambda d: len(_compact(d))) if dicts else observation


# ---------------------------------------------------------------- string records
# Race 15's store returned every search row as one string ("ref_ref: p-x | sold_by_name: Annoka | price: 5498 |
# category: Google Pixel Cases | facts: color=brown; material=tpu | ...") and a page as {"value": "<such a string>"}.
# The memory layer reads dicts, so it saw no rows, no prices and no properties for the whole race, and the order gate
# (which needs a property record that states the requirements) declined every task: 8 passes in 180 episodes against
# 49 for the winner, whose code is equally blind but lets the model's checkout through. Observations are therefore
# turned into dicts as they arrive, before anything reads them.
_RECORD_SPLIT = re.compile(r"\s\|\s(?=[A-Za-z][\w .'-]{0,30}:\s)")  # a pipe followed by "key: " (a pipe inside a title is kept)
_RECORD_FIELD = re.compile(r"^([A-Za-z][\w .'-]{0,30}):\s?(.*)$", re.S)
_PAIR_FIELD = re.compile(r"^\s*([^=;]{1,40})=([^;]*)$")
_FACTS_KEY = re.compile(r"fact|spec|attr|propert|detail|trait|feature|datasheet|param", re.I)


def _scalar(text: str) -> Any:
    low = text.strip().lower()
    if low in ("none", "null", "n/a", "nil"):
        return None
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if re.fullmatch(r"-?\d{1,12}", low):
        return int(low)
    if re.fullmatch(r"-?\d{1,12}\.\d+", low):
        return float(low)
    return text.strip()


def _pairs_dict(text: str, key: str = "") -> dict[str, Any] | None:
    # "color=brown; material=tpu; options=" -> {"color": "brown", "material": "tpu"}. Needs two pairs, or one under a
    # key that names a property table, and every piece must be a pair.
    pieces = [p for p in text.split(";") if p.strip()]
    if not pieces:
        return None
    out: dict[str, Any] = {}
    for piece in pieces:
        match = _PAIR_FIELD.match(piece)
        if not match:
            return None
        name = match.group(1).strip()
        value = _scalar(match.group(2))
        if name and value is not None and value != "":
            out[name] = value
    if len(pieces) >= 2 or _FACTS_KEY.search(key):
        return out
    return None


def _record_dict(text: str) -> dict[str, Any] | None:
    # "key: value | key: value | ..." with at least three fields, the first at the start -> dict.
    if ": " not in text or "|" not in text or len(text) > 20000:
        return None
    pieces = _RECORD_SPLIT.split(text.strip())
    if len(pieces) < 3:
        return None
    out: dict[str, Any] = {}
    for piece in pieces:
        match = _RECORD_FIELD.match(piece.strip())
        if not match:
            return None
        name, value = match.group(1).strip(), match.group(2).strip()
        nested = _pairs_dict(value, name) if "=" in value else None
        out[name] = nested if nested is not None else _scalar(value)
    return out if len(out) >= 3 else None


def _structured(value: Any, key: str = "", depth: int = 0) -> Any:
    # Observations with records written as strings become the dicts the rest of the agent reads.
    if depth > 8:
        return value
    if isinstance(value, str):
        record = _record_dict(value)
        if record is not None:
            return record
        if "=" in value and ";" in value and len(value) <= 4000:
            pairs = _pairs_dict(value, key)
            if pairs:
                return pairs
        return value
    if isinstance(value, list):
        return [_structured(v, key, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _structured(v, str(k), depth + 1) for k, v in value.items()}
    return value


class Listing:
    def __init__(self, ident: str) -> None:
        self.id = ident
        self.fields: dict[str, Any] = {}
        self.record: Any = None
        self.inspected = False
        self.read_by: set[str] = set()  # tools whose record of this listing was read
        self.seen_step = 0
        self.notes_read = False
        self.paged = False  # its page was opened with the listing-page tool
        self.defers: list[tuple[str, str]] = []  # (property, the other listing its notes send that property to)
        self.reads: list[str] = []  # what the notes reader found in the description
        self.price_key: str | None = None  # the price field this store's rows actually use
        self.stock_key: str | None = None

    def merge(self, record: dict[str, Any], step: int) -> None:
        for key, value in record.items():
            if value is not None and value != "":
                self.fields[str(key)] = value
        self.seen_step = step

    @property
    def price(self) -> float | None:
        sources = (self.fields, self.record if isinstance(self.record, dict) else {})
        if self.price_key:
            for source in sources:
                if self.price_key in source and not isinstance(source[self.price_key], dict):
                    number = _number(source[self.price_key])
                    if number is not None:
                        return number
        for source in sources:
            for key, value in source.items():
                if _PRICE_KEY.search(key) and not _TOTAL_KEY.search(key):
                    number = _number(value) if not isinstance(value, dict) else None
                    if number is not None:
                        return number
        return None

    @property
    def title(self) -> str:
        for key, value in self.fields.items():
            if _TITLE_KEY.search(key) and isinstance(value, str):
                return value
        return ""

    @property
    def category(self) -> str:
        for key, value in self.fields.items():
            if _CATEGORY_KEY.search(key) and isinstance(value, str):
                return value
        return ""

    def _read_stock(self, key: str, value: Any) -> bool | None:
        negative = bool(re.search(r"out|sold", key, re.I))
        if isinstance(value, bool):
            return (not value) if negative else value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (value > 0) != negative
        if isinstance(value, str):
            low = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value).replace("_", " ").replace("-", " ").lower().strip()
            count = re.fullmatch(r"(\d+)", low) or re.search(r"(\d+)\s*(?:left|remaining|units?|pcs|pieces|in stock|available)\b", low) \
                or re.search(r"(?:stock|available|qty|quantity|units)\s*:?\s*(\d+)\b", low)
            if count:
                return (int(count.group(1)) > 0) != negative
            if re.search(r"\bout\b|unavail|sold|back ?order|pre ?order|discontinu|\bnone\b|^no\b|\bnot\b|false|empty", low):
                return negative
            if re.search(r"\bin\b|yes|avail|true|ready", low):
                return not negative
        return None

    @property
    def in_stock(self) -> bool | None:
        if self.fields.get("sold_out_now"):
            return False
        if self.stock_key and self.stock_key in self.fields:
            judged = self._read_stock(self.stock_key, self.fields[self.stock_key])
            if judged is not None:
                return judged
        for key, value in self.fields.items():
            if not _STOCK_KEY.search(key) or not _is_stock_key(key):
                continue  # "available sizes", "colors available": a property list, not stock
            judged = self._read_stock(key, value)
            if judged is not None:
                return judged
        return None

    def properties(self) -> dict[str, str]:
        # Flat property map from the inspected record (normalized keys).
        props: dict[str, str] = {}

        def visit(value: Any, prefix: str, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    visit(item, str(key), depth + 1)
            elif isinstance(value, (str, int, float)) and not isinstance(value, bool) and prefix:
                key = _norm_key(prefix)
                if _ID_KEY.search(prefix) or _PRICE_KEY.search(prefix) or _TITLE_KEY.search(prefix):
                    return
                if _is_stock_key(prefix) or (isinstance(value, str) and len(value) > 300):
                    return
                text = str(value).strip()
                if not text:
                    return
                if key not in props:
                    props[key] = text[:300]
                elif props[key] != text:
                    # Two spellings of one name can disagree ("clock shape" / "clock_shape"). Keep both.
                    n = 2
                    while f"{key} {n}" in props:
                        n += 1
                    props[f"{key} {n}"] = text[:300]
            elif isinstance(value, list) and prefix and all(isinstance(v, str) for v in value) and value:
                props.setdefault(_norm_key(prefix), ", ".join(value)[:300])

        visit(self.record if isinstance(self.record, dict) else self.fields, "", 0)
        return props

    def datasheet(self) -> dict[str, str]:
        # The listing's property sub-dict (traits_map, item_properties, facts, ...): the largest nested dict of plain values.
        #
        # Questions are chosen from these property keys, not from top-level fields such as shop or brand.
        best: dict[str, Any] = {}

        def visit(value: Any, depth: int) -> None:
            nonlocal best
            if depth > 3 or not isinstance(value, dict):
                return
            for item in value.values():
                if isinstance(item, dict):
                    plain = {k: v for k, v in item.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
                    if len(plain) >= 2 and len(plain) >= len(best) and len(plain) >= len(item) - 1:
                        best = plain
                    visit(item, depth + 1)

        visit(self.record if isinstance(self.record, dict) else {}, 0)
        return {_norm_key(k): str(v).strip()[:60] for k, v in best.items() if str(v).strip()}

    def notes(self) -> list[str]:
        texts: list[str] = []
        source = self.record if isinstance(self.record, dict) else self.fields
        for key, value in source.items():
            if isinstance(value, str) and len(value) > 30 and (_LONG_TEXT_KEY.search(key) or len(value) > 80):
                texts.append(value)
        return texts

    def references(self) -> list[str]:
        found = list(self.reads)
        # phrase match as a fallback for descriptions the notes reader did not get to
        for text in self.notes():
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if _REFERENCE.search(sentence) and len(sentence) < 400:
                    found.append(sentence.strip())
        return found


class Agent:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.opening = _structured(self.policy.get("opening"))
        self.tools: list[dict[str, Any]] = []
        for tool in self.policy.get("tools") or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(function, dict) and function.get("name"):
                self.tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": function["name"],
                            "description": function.get("description") or "",
                            "parameters": function.get("parameters") or {"type": "object", "properties": {}},
                        },
                    }
                )
        self.schemas = {t["function"]["name"]: t["function"]["parameters"] for t in self.tools}
        self.descriptions = {t["function"]["name"]: t["function"]["description"] for t in self.tools}
        self.note_tool = NOTE_TOOL if NOTE_TOOL not in self.schemas else NOTE_TOOL + "_x"
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.started = time.monotonic()
        self.late_asked = False
        self.one_question = False

        self.calls: dict[str, dict[str, Any]] = {}
        self.products: list[dict[str, Any]] = []
        self.budget: float | None = None
        self.budget_scope = "total"
        self.message_budget: int | None = None
        self.messages_used = 0
        self.reserve_messages = 0
        self.decline_rule = ""
        self.rules: list[str] = []

        self.listings: dict[str, Listing] = {}
        self.searches: list[str] = []
        self.cart: dict[str, int] | None = None
        self.cart_view_step = -1
        self.cart_edit_step = -1
        self.added: dict[str, int] = {}
        self.updates: list[str] = []
        self.todo: list[str] = []
        self.notes: list[str] = []
        self.heard: set[str] = set()
        self.questions: list[dict[str, Any]] = []
        self.pending_singles: list[tuple[str, str, bool]] = []  # (product, property, show two example values)
        self.repeated: set[tuple[str, str]] = set()  # (product, property) already re-asked after a crowded reply
        self.reply_kinds: dict[str, str] = {}  # reply text -> how the model read it, for replies that stated nothing
        self.ask_done = False
        self.ask_wait_until = 0
        self.interview: list[dict[str, Any]] | None = None  # the planner's ordered questions, None until planned
        self.cache_at = 0  # index of the last settled controller message (cache mark)
        self.ask_failures = 0
        self.ask_now = False
        self.ask_pushbacks = 0
        self.hints: list[dict[str, Any]] = []
        self.ask_first: list[dict[str, Any]] = []  # properties the request says to check with the shopper
        self.pick = "any"  # cheapest, most, least, or any
        self.pick_measure = ""
        self.process: list[dict[str, Any]] = []  # required steps, each with when it is due
        self.gate_ok: set[str] = set()  # a stage already held once for this cart
        self.shapes: dict[str, dict[str, Any] | None] = {}  # per tool: which result keys hold rows, the page token, price, stock
        self.learned_price: str | None = None
        self.learned_stock: str | None = None
        self.basket_check = False
        self.basket_sent_for: dict[str, int] | None = None
        self.decline_ordered = False
        self.decline_candidates: list[str] = []
        self.verify_rejections = 0
        self.decline_pushbacks = 0
        self.order_refusals = 0
        self.second_walk = False  # one longer loop after a refused finish, while steps remain
        self.walk_left = 0  # controller turns left in that second loop
        self.done_log: list[tuple[int, str]] = []  # (step, what was called) so a procedure can be proved after the cart changed
        self.fits: list[tuple[str, str]] = []  # (part label, device label) the request says must work together
        self.refused_ids: set[str] = set()  # listings known sold out (never ordered again)
        self.refused_carts: dict[str, int] = {}  # carts the store refused at checkout, by how often
        self.stock_recheck: list[str] = []  # cart lines to re-read after a stock-out the store did not pin on a line
        self.unread: list[tuple[list[str], str | None]] = []  # shopper lines whose reading failed
        self.unread_tries = 0
        self.query_keys: set[str] = set()  # distinct queries (arguments without the page value)
        self.decline_reads_done = False
        self.choice_opens = 0
        self.choice_searches = 0
        self.price_hunts: set[str] = set()
        self.verified_cart: dict[str, int] = {}  # the last cart the checker passed
        self.checkout_tried = False
        self.found_by: dict[str, dict[str, Any]] = {}
        self.chain_hops = 0
        self.followed: set[tuple[str, str]] = set()
        self.page_info: bool | None = None
        self.auto_pages = 0
        self.query_rows: dict[str, int] = {}  # rows shown so far per query (its arguments without the page value)
        self.pages_fetched: set[str] = set()
        self.row_keys: set[str] = set()  # field names seen in search rows  # does this store's listing page show fields its search rows lack (race 7 notes)?
        self.probed: set[tuple[str, float]] = set()
        self.note_calls = 0
        self.note_reads = 0
        self.read_cache: dict[str, dict[str, Any]] = {}
        self.llm_failures = 0
        self.spend = 0.0  # USD of inference this task
        # a message names several properties of one product until the shopper asks for one at a time
        self.singles = False
        self.list_on = True
        self.next_labels: dict[str, list[str]] = {}
        self.list_misses: dict[str, int] = {}
        self.preread: set[str] = set()  # carts whose lines the code already opened before the final check
        self.llm_dead = False  # the inference key was refused (cap reached or revoked)
        self.facts_from: dict[str, bool] = {}  # tool -> its listing records carry a property table (KYO)
        self.finish_refused_at = -1  # step of the last finish the store refused
        self.contradiction_holds = 0  # orders held because a record contradicts a line (at most 2 per task)
        self.story_props: dict[str, list[str]] = {}  # per product: properties the model names from the story when no record names any
        self.browsed: set[str] = set()  # products whose category was listed with the browse tool
        self.finish_refusal = ""
        self.last_heard: list[str] = []
        self.last_heard_step = -1

        self.history: list[dict[str, Any]] = []
        self.dialogue: list[dict[str, Any]] = []
        self.call_seq = 0

    # ------------------------------------------------------------------ infra

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def messages_left(self) -> int:
        budget = self.message_budget if self.message_budget is not None else 4  # race worlds allowed 4-5 when the request did not say
        return max(0, budget - self.messages_used)

    def call_cost(self, data: dict[str, Any], body: dict[str, Any], message: dict[str, Any]) -> float:
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        for key in ("cost", "total_cost"):
            if isinstance(usage.get(key), (int, float)) and not isinstance(usage.get(key), bool):
                return float(usage[key])
        prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int) and prompt > 0:
            return prompt * RATE_IN + completion * RATE_OUT
        sent = len(_compact(body.get("messages"))) + len(_compact(body.get("tools") or []))
        got = len(message.get("content") or "") + len(_compact(message.get("tool_calls") or []))
        return sent / 3.8 * RATE_IN + got / 3.8 * RATE_OUT

    def over_budget(self, soft: bool = True) -> bool:
        return self.spend >= (SPEND_SOFT if soft else SPEND_HARD)

    def infer(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None,
              cache_at: int = 0) -> dict[str, Any] | None:
        if self.llm_dead or self.spend >= SPEND_STOP + 0.01:
            return None
        if self.spend + 0.02 > SPEND_HARD:
            # the strong model is cut first; the cheap models keep walking to SPEND_STOP (Halo's judges())
            models = [m for m in models if m not in HEAVY_MODELS or m in MAIN_MODELS] or MAIN_MODELS
        for model in models:
            for attempt in range(2):
                # a call starts only when its result can still be used inside the 300s task
                room = 300 - (time.monotonic() - self.started) - 20
                if room < 8:
                    return None
                _llm.timeout = min(LLM_TIMEOUT, room)
                sent = messages
                if model.startswith("anthropic/"):
                    sent = [dict(m) for m in messages]
                    for index in range(min(2, len(sent))):
                        content = sent[index].get("content")
                        if isinstance(content, str) and len(content) > 400 and sent[index]["role"] in ("system", "user"):
                            sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                    if cache_at and 2 <= cache_at < len(sent) - 1 and isinstance(sent[cache_at].get("content"), str) and sent[cache_at]["content"]:
                        # KYO marks its last message: here the end of the settled history, so each turn reads it from cache
                        sent[cache_at]["content"] = [{"type": "text", "text": sent[cache_at]["content"], "cache_control": {"type": "ephemeral"}}]
                body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                result = _llm.post_verbose("/inference/chat/completions", body)
                data = result.data
                if isinstance(data, dict) and data.get("choices"):
                    message = data["choices"][0].get("message") or {}
                    self.spend += self.call_cost(data, body, message)
                    if message.get("content") or message.get("tool_calls"):
                        self.llm_failures = 0
                        return message
                    break
                status = (result.error or {}).get("status")
                self.dialogue.append({"kind": "inference_error", "model": model, "status": status})
                if status in (401, 402, 403):
                    # the run's key is spent or refused: every model shares it, so stop calling models
                    self.llm_dead = True
                    return None
                if (status == 429 or status is None or (isinstance(status, int) and status >= 500)) and attempt == 0 and self.steps_left() > 2:
                    time.sleep(2.5 if status == 429 else 1.0)
                    continue
                break
        self.llm_failures += 1
        if self.steps_left() > 4:
            # a provider outage often clears within seconds: back off instead of giving up the episode
            time.sleep(min(8.0, 1.5 * self.llm_failures))
        return None

    def ask_json(self, prompt: str, content: str, models: list[str] = STRONG_MODELS, max_tokens: int = 900) -> dict[str, Any] | None:
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
        message = self.infer(messages, models, max_tokens)
        parsed = _parse_json((message or {}).get("content") or "")
        if parsed is None and message is not None and self.steps_left() > 2:
            messages += [
                {"role": "assistant", "content": message.get("content") or ""},
                {"role": "user", "content": "Return only the JSON object, nothing else."},
            ]
            parsed = _parse_json((self.infer(messages, models, max_tokens) or {}).get("content") or "")
        return parsed

    def env_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if self.finished or self.dead or self.steps_left() <= 0:
            return None
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for attempt in range(4):
            _env.timeout = 40
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                if "observation" in data:
                    data["observation"] = _structured(data["observation"])
                self.dialogue.append({"step": self.step, "action": {"name": name, "args": args}, "observation": _brief(data.get("observation"), 300, 12)})
                self.done_log.append((self.step, f"{name} {_compact(args)[:160]}"))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.dialogue.append({"kind": "env_error", "status": status, "body": body[:200]})
            if status == 409:
                if "finish" in body.lower():
                    self.finished = True
                    return None
                match = re.search(r"expected\D{0,20}(\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status in (401, 402, 403):
                    self.dead = True
                return None
            time.sleep(0.4 * (attempt + 1))
        return None

    # ------------------------------------------------------------------ setup

    def setup(self) -> None:
        lines = []
        for tool in self.tools:
            function = tool["function"]
            lines.append(f"- {function['name']}: {function['description']} | parameters: {_compact(function['parameters'])}")
        content = "TOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.opening is not None:
            content += "\n\nOPENING OBSERVATION:\n" + _compact(_brief(self.opening, 1500))
        plan = self.ask_json(SETUP_PROMPT, content, HEAVY_MODELS, 1800) or {}
        self.dialogue.append({"kind": "setup", "plan": plan})

        calls = plan.get("calls") if isinstance(plan.get("calls"), dict) else {}
        for role, call in calls.items():
            if isinstance(call, dict) and call.get("tool") in self.schemas:
                args = call.get("args") if isinstance(call.get("args"), dict) else {}
                params = (self.schemas[call["tool"]].get("properties") or {}).keys()
                kept = {k: v for k, v in args.items() if k in params}
                # a pick whose schema cannot do the job is dropped and guessed below (race 11: "add" was the cart view)
                need = {"details": "<ID>", "listing": "<ID>", "cart_add": "<ID>", "cart_remove": "<ID>", "search": "<TEXT>", "message": "<TEXT>", "browse": "<CATEGORY>"}.get(role)
                if need and need not in kept.values():
                    self.dialogue.append({"kind": "setup_drop", "role": role, "tool": call["tool"]})
                    continue
                if role == "cart_view" and self.schemas[call["tool"]].get("required"):
                    continue
                self.calls[role] = {"tool": call["tool"], "args": kept}
        self.guess_calls()

        for item in plan.get("products") or []:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "").strip()
            label = str(item.get("label") or "").strip()
            if label.lower() in {"", "item", "product", "listing"}:
                label = category or "item"
            quantity = _int(item.get("quantity")) or 1
            reqs = [{"text": str(r), "source": "stated", "active": True} for r in item.get("requirements") or [] if r]
            self.products.append({"label": label, "category": category, "quantity": quantity, "cheapest": bool(item.get("cheapest")), "reqs": reqs})
        if not self.products:
            self.products.append({"label": "item", "quantity": 1, "cheapest": False, "reqs": []})
        self.adopt_category()
        self.budget = _number(plan.get("budget"))
        self.budget_scope = "each" if str(plan.get("budget_scope") or "").lower() == "each" else "total"
        self.message_budget = _int(plan.get("message_budget"))
        self.reserve_messages = min(_int(plan.get("reserve_messages")) or 0, 2)
        rule_text = " ".join(str(r) for r in plan.get("rules") or [] if r) + " " + self.request
        # race 6: every passing order sent the shopper a basket summary after the last cart change and before ordering
        self.basket_check = bool(_APPROVAL.search(rule_text))
        # races 10-11: KYO won both without a basket summary; a summary goes out only when the request asks for one, and
        # every other message asks about a property
        if self.basket_check:
            self.reserve_messages = max(self.reserve_messages, 1)
        self.hints = [h for h in plan.get("hints") or [] if isinstance(h, dict) and str(h.get("property") or "").strip()][:6]
        self.ask_first = []
        for item in plan.get("ask_first") or []:
            if isinstance(item, dict) and str(item.get("property") or "").strip():
                self.ask_first.append({"product": str(item.get("product") or ""), "property": str(item["property"]).strip()})
            elif isinstance(item, str) and item.strip():
                self.ask_first.append({"product": "", "property": item.strip()})
        pick = str(plan.get("pick") or "").lower()
        self.pick = pick if pick in ("cheapest", "most", "least") else "any"
        self.pick_measure = str(plan.get("pick_measure") or "").strip()[:40]
        if any(p["cheapest"] for p in self.products):
            self.pick = "cheapest"
        self.process = []
        for entry in plan.get("process") or []:
            if not isinstance(entry, dict) or not str(entry.get("step") or "").strip():
                continue
            when = str(entry.get("when") or "order")
            self.process.append({"step": str(entry["step"]).strip()[:80], "when": when if when in ("choose", "order", "end") else "order",
                                 "messages": _int(entry.get("messages")) or 0, "done": False, "at": -1})
        self.fits = []
        for pair in plan.get("fits") or []:
            if isinstance(pair, dict) and str(pair.get("part") or "").strip() and str(pair.get("device") or "").strip():
                self.fits.append((str(pair["part"]).strip(), str(pair["device"]).strip()))
        self.decline_rule = str(plan.get("decline_rule") or "")
        self.rules = [str(r) for r in plan.get("rules") or [] if r]
        if plan.get("in_stock_required") is not False:
            self.rules.append("Only order items shown as in stock / buyable.")
        steps = _int(plan.get("step_budget"))
        if steps and 0 < steps < self.max_steps:
            self.max_steps = steps
        self.unmix()
        self.ground_requirements()

    def ground_requirements(self) -> None:
        # A stated requirement whose value the request never uses was invented. A "for a wedding" occasion is kept.
        stems = _stems(self.request)
        request = self.request.lower()
        for product in self.products:
            for req in product["reqs"]:
                if req.get("source") != "stated" or not req["active"] or ":" not in req["text"]:
                    continue
                if not _value_in_request(req["text"].split(":", 1)[1], stems, request):
                    req["active"] = False
            if any(req["active"] and _prop_stems(req["text"].split(":", 1)[0]) == _prop_stems("occasion") for req in product["reqs"] if ":" in req["text"]):
                continue
            for match in re.finditer(r"\bfor an?\s+([a-z]{5,})\b", request):
                word = match.group(1)
                if word in _STORY_OCCASION or word not in stems:
                    continue
                product["reqs"].append({"text": f"occasion: {word}", "source": "use", "active": True})
                break

    def unmix(self) -> None:
        # A lookup requirement points at another listing. Details that only identify that listing are not requirements
        # of the product being bought; they move into the lookup's own text.
        if self.over_budget():
            return
        for product in self.products:
            live = [req for req in product["reqs"] if req["active"]]
            pointers = [req for req in live if _REFERENCE.search(req["text"]) or re.search(r"\b(value|same|that) on\b|another listing", req["text"], re.I)]
            rest = [req for req in live if req not in pointers]
            if not pointers or not rest:
                continue
            content = (
                f"REQUEST:\n{self.request}\n\nPRODUCT: {product['label']}\nPOINTERS:\n"
                + "\n".join(req["text"] for req in pointers)
                + "\nOTHER:\n" + "\n".join(req["text"] for req in rest)
            )
            models = HEAVY_MODELS if self.spend < SPEND_SOFT else STRONG_MODELS
            got = self.ask_json(UNMIX_PROMPT, content, models, 400) or {}
            moved = [req for req in rest if req["text"] in {str(item).strip() for item in got.get("only_the_other") or []}]
            if not moved:
                continue
            for req in moved:
                req["active"] = False
            pointers[0]["text"] += " (that listing: " + "; ".join(req["text"] for req in moved) + ")"
            self.dialogue.append({"kind": "unmix", "product": product["label"], "moved": [req["text"] for req in moved]})

    def guess_calls(self) -> None:
        # Fill tool roles the setup reading did not give, from names, descriptions and schemas.

        def text(name: str) -> str:
            return _norm_key(name) + " " + self.descriptions.get(name, "").lower()

        def params(name: str) -> dict[str, Any]:
            return self.schemas[name].get("properties") or {}

        def pick_param(name: str, pattern: str, kinds: tuple[str, ...]) -> str | None:
            for key, spec in params(name).items():
                if re.search(pattern, key, re.I) and (not isinstance(spec, dict) or spec.get("type", "string") in kinds):
                    return key
            return None

        def enum_value(name: str, pattern: str) -> dict[str, Any]:
            fixed = {}
            for key, spec in params(name).items():
                values = spec.get("enum") if isinstance(spec, dict) else None
                if values:
                    match = next((v for v in values if re.search(pattern, str(v), re.I)), values[0])
                    fixed[key] = match
            return fixed

        names = list(self.schemas)
        finals = [n for n in names if _FINAL_WORDS.search(self.descriptions.get(n, ""))]
        if "order" not in self.calls:
            for name in finals or names:
                if _ORDER_WORDS.search(text(name)) and not _DECLINE_WORDS.search(text(name)):
                    self.calls["order"] = {"tool": name, "args": enum_value(name, r"check|order|buy|purchase|pay|place")}
                    break
        order_tool = self.calls.get("order", {}).get("tool")
        if "decline" not in self.calls and order_tool:
            # one finishing tool that orders or declines through an enum argument
            for key, spec in params(order_tool).items():
                values = spec.get("enum") if isinstance(spec, dict) else None
                declining = [v for v in values or [] if re.search(r"declin|abstain|abandon|give|refus|cancel|none|skip|no.?(buy|order|purchase)", str(v), re.I)]
                if declining:
                    args = {k: v for k, v in self.calls["order"]["args"].items()}
                    args[key] = declining[0]
                    remark = pick_param(order_tool, r"remark|text|note|reason|message|body|comment", ("string",))
                    if remark:
                        args[remark] = "<TEXT>"
                    self.calls["decline"] = {"tool": order_tool, "args": args}
                    break
        if "decline" not in self.calls:
            for name in finals or names:
                if _DECLINE_WORDS.search(text(name)) or (finals and name in finals and name != self.calls.get("order", {}).get("tool")):
                    args = enum_value(name, r"declin|abstain|abandon|give|refus|cancel|none|skip|quit|walk|leave|without")
                    for key, spec in params(name).items():
                        values = spec.get("enum") if isinstance(spec, dict) else None
                        # no option names declining: take one that does not name buying (race 15's commit|skip)
                        if values and re.search(r"check|order|buy|purchas|pay|place|commit|confirm|accept|complete", str(args.get(key)), re.I):
                            args[key] = next((v for v in values if not re.search(r"check|order|buy|purchas|pay|place|commit|confirm|accept|complete", str(v), re.I)), args[key])
                    remark = pick_param(name, r"remark|text|note|reason|message|body|comment", ("string",))
                    if remark:
                        args[remark] = "<TEXT>"
                    self.calls["decline"] = {"tool": name, "args": args}
                    break
        if "order" not in self.calls and "decline" in self.calls:
            # one finishing tool, two modes (race 11: WrapUpRequest commit|none): the order takes the mode that is not the decline's
            tool = self.calls["decline"]["tool"]
            for key, spec in params(tool).items():
                values = spec.get("enum") if isinstance(spec, dict) else None
                chosen = self.calls["decline"]["args"].get(key)
                if values and chosen in values and len(values) >= 2:
                    rest = [v for v in values if v != chosen]
                    value = next((v for v in rest if re.search(r"check|order|buy|purchas|pay|place|commit|confirm|accept|complete", str(v), re.I)), rest[0])
                    self.calls["order"] = {"tool": tool, "args": {key: value}}
                    break
        if "message" not in self.calls:
            for name in names:
                if re.search(r"shopper|customer|client|buyer|question|ask|message|chat|pose|write", text(name)) and name not in finals:
                    body = pick_param(name, r"body|text|note|message|content|question", ("string",))
                    if body and not re.search(r"cart|bag|basket", text(name)):
                        self.calls["message"] = {"tool": name, "args": {body: "<TEXT>"}}
                        break
        def take_cart(name: str) -> bool:
            ident = pick_param(name, r"id|code|product|item|target|listing|chosen|sku|which|ref", ("string", "integer"))
            if not ident:
                return False
            args = enum_value(name, r"add|include|put|set|insert|keep|hold|place|plus")
            for key, spec in params(name).items():
                values = spec.get("enum") if isinstance(spec, dict) else None
                if values and re.search(r"remov|discard|drop|delete|take|clear|minus|out", str(args.get(key)), re.I):
                    # no value reads as adding: take one that does not read as removing
                    args[key] = next((v for v in values if not re.search(r"remov|discard|drop|delete|take|clear|minus|out", str(v), re.I)), values[0])
            args[ident] = "<ID>"
            qty = pick_param(name, r"qty|quant|count|many|units|number|amount|needed", ("integer", "number"))
            if qty:
                args[qty] = "<QTY>"
            self.calls["cart_add"] = {"tool": name, "args": args}
            if "cart_remove" not in self.calls and any(isinstance(s, dict) and s.get("enum") for s in params(name).values()):
                remove = enum_value(name, r"remov|discard|drop|delete|take|clear|minus")
                remove[ident] = "<ID>"
                self.calls["cart_remove"] = {"tool": name, "args": remove}
            return True

        def line_list(name: str) -> tuple[str, str, str | None] | None:
            # a cart tool that takes the whole basket as a list of {id, count} (race 13's hold_to_cart)
            for key, spec in params(name).items():
                if not isinstance(spec, dict) or spec.get("type") != "array":
                    continue
                item = spec.get("items") if isinstance(spec.get("items"), dict) else {}
                fields = item.get("properties") if isinstance(item.get("properties"), dict) else {}
                ident = next((k for k in fields if re.search(r"id|sku|item|listing|target|product", k, re.I)), None)
                qty = next((k for k in fields if k != ident and re.search(r"qty|quant|count|unit|amount|number|many", k, re.I)), None)
                if ident:
                    return key, ident, qty
            return None

        if "cart_add" not in self.calls:
            for name in names:
                if re.search(r"cart|bag|basket|trolley|hold", text(name)) and params(name) and name not in finals and take_cart(name):
                    break
        if "cart_add" not in self.calls:
            for name in names:
                if name in finals:
                    continue
                found = line_list(name)
                if found and re.search(r"cart|bag|basket|hold|trolley|stash|select|keep|pick", text(name)):
                    self.calls["cart_add"] = {"tool": name, "args": {found[0]: "<LINES>"}, "line": {"id": found[1], "qty": found[2]}}
                    break
        if "cart_add" not in self.calls:
            # no name says cart: a tool that takes a listing id with a count or a mode (race 11's Stash)
            taken = {c["tool"] for r, c in self.calls.items() if r in ("details", "listing", "search", "message", "order", "decline")}
            for name in names:
                if name in finals or name in taken:
                    continue
                if pick_param(name, r"section|department|categor|shelf|aisle|^group|collection", ("string",)):
                    continue  # lists a category: not a cart
                has_qty = pick_param(name, r"qty|quant|count|many|units|number|amount|needed", ("integer", "number"))
                has_mode = any(isinstance(spec, dict) and spec.get("enum") for spec in params(name).values())
                if (has_qty or has_mode) and take_cart(name):
                    break
        if "cart_view" not in self.calls:
            for name in names:
                if re.search(r"cart|bag|basket|trolley", text(name)) and not (self.schemas[name].get("required") or []) and name not in finals:
                    if name != self.calls.get("cart_add", {}).get("tool"):
                        self.calls["cart_view"] = {"tool": name, "args": {}}
                        break
        if "details" not in self.calls:
            for name in names:
                single = [k for k in params(name) if re.search(r"id|code|target|listing|item|sku", k, re.I)]
                if single and len(params(name)) <= 2 and re.search(r"propert|detail|datasheet|spec|trait|attribute|inspect|full|page|show|view|open", text(name)):
                    self.calls["details"] = {"tool": name, "args": {single[0]: "<ID>"}}
                    break
        if "search" not in self.calls:
            for name in names:
                if re.search(r"search|find|query|look", text(name)):
                    free = pick_param(name, r"text|query|q$|keyword|term|wanted|words|search", ("string",))
                    if free:
                        args = enum_value(name, r"typed|text|keyword|query")
                        args[free] = "<TEXT>"
                        self.calls["search"] = {"tool": name, "args": args}
                        break
        if "browse" not in self.calls:
            # a tool that lists one category's rows (race 15's listRange(itemSection)): the winner opened each item's
            # category with it 3 times per episode; keyword search there returned nothing for most queries
            search_tool = self.calls.get("search", {}).get("tool")
            taken = {c["tool"] for r, c in self.calls.items() if r in ("details", "listing", "message", "order", "decline", "cart_add", "cart_remove", "cart_view")}
            for name in names:
                if name in finals or name in taken:
                    continue
                shelf = pick_param(name, r"section|department|categor|shelf|aisle|^group|collection", ("string",))
                if not shelf or (name == search_tool and not any(isinstance(s, dict) and s.get("enum") for s in params(name).values())):
                    continue
                args = enum_value(name, r"field|filter|facet|brows|categor|list|section|department")
                args[shelf] = "<CATEGORY>"
                self.calls["browse"] = {"tool": name, "args": args}
                break

    def fill(self, role: str, ident: str = "", qty: int = 1, text: str = "") -> tuple[str, dict[str, Any]] | None:
        call = self.calls.get(role)
        if not call:
            return None
        args: dict[str, Any] = {}
        props = self.schemas[call["tool"]].get("properties") or {}
        for key, value in call["args"].items():
            if value == "<ID>":
                args[key] = ident
            elif value == "<QTY>":
                kind = (props.get(key) or {}).get("type") if isinstance(props.get(key), dict) else None
                args[key] = qty if kind != "string" else str(qty)
            elif value in ("<TEXT>", "<CATEGORY>"):
                limit = (props.get(key) or {}).get("maxLength") if isinstance(props.get(key), dict) else None
                args[key] = text[: int(limit)] if isinstance(limit, int) else text
            elif value == "<LINES>":
                line = call.get("line") or {}
                row: dict[str, Any] = {}
                if line.get("id"):
                    row[line["id"]] = ident
                if line.get("qty"):
                    row[line["qty"]] = qty
                args[key] = [row] if ident else []
            else:
                args[key] = value
        return call["tool"], args

    def role_of(self, name: str, args: dict[str, Any] | None = None) -> str | None:
        args = args or {}
        matches = [role for role, call in self.calls.items() if call["tool"] == name]
        if len(matches) <= 1:
            return matches[0] if matches else None
        # one tool, several roles told apart by a fixed argument (for example a route or a cart action)
        for role in matches:
            fixed = {k: v for k, v in self.calls[role]["args"].items() if v not in ("<ID>", "<QTY>", "<TEXT>")}
            if fixed and all(args.get(k) == v for k, v in fixed.items()):
                return role
        return matches[0]

    # ------------------------------------------------------------------ memory

    def single_listing(self, name: str, args: dict[str, Any]) -> str | None:
        # the listing id an unmapped tool is called with, when it takes just one listing
        if name not in self.schemas or name in {c["tool"] for c in self.calls.values()}:
            return None
        values = [str(v) for v in args.values() if isinstance(v, (str, int)) and not isinstance(v, bool)]
        hits = [v for v in values if v in self.listings]
        return hits[0] if len(hits) == 1 and len(args) <= 2 else None

    def note_facts(self, name: str, record: Any) -> None:
        # Learn which tool's records carry the property table, and give that tool the details role.
        #
        # Race 11 split a listing over a page tool (InspectDetails: text, fine print) and a property tool
        # (TechnicalData: properties_table); setup took the page for details and every property table was lost.
        # The role moves only once the details tool was seen to return no table.
        table = isinstance(record, dict) and any(
            isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, (str, int, float)) and not isinstance(x, bool)) >= 2
            for v in record.values())
        self.facts_from[name] = self.facts_from.get(name, False) or table
        details = (self.calls.get("details") or {}).get("tool")
        if details and self.facts_from.get(details) is not False:
            return
        target = next((t for t, ok in self.facts_from.items() if ok and t != details and t in self.schemas), None)
        if target is None:
            return
        listing_call = self.calls.get("listing") or {}
        ident = next((k for k, v in listing_call.get("args", {}).items() if v == "<ID>"), None) if listing_call.get("tool") == target else None
        props = self.schemas[target].get("properties") or {}
        if ident is None:
            ident = next((k for k in props if _ID_KEY.search(k) or re.search(r"id|ref|code|which|product|item|listing", k, re.I)), None)                 or next(iter(self.schemas[target].get("required") or []), None) or next(iter(props), None)
        if ident is None:
            return
        old = self.calls.get("details")
        self.calls["details"] = {"tool": target, "args": {ident: "<ID>"}}
        if old:
            self.calls["listing"] = old
        elif listing_call.get("tool") == target:
            self.calls.pop("listing", None)
        self.dialogue.append({"kind": "details_swap", "details": target, "listing": (old or {}).get("tool")})
        for listing in self.listings.values():
            read_by = listing.read_by
            if read_by:
                listing.paged = bool(old) and old["tool"] in read_by  # opened by what is now the page tool
                listing.inspected = target in read_by or bool(listing.datasheet())

    def probe_details(self) -> None:
        # The details tool returned no property table: try once each other tool that takes one listing, on a listing
        # already opened (KYO's facts_from). Race 11's property table sat behind the second tool the controller rarely used.
        details = (self.calls.get("details") or {}).get("tool")
        if not details or self.facts_from.get(details) is not False or self.steps_left() < 12 or self.finished:
            return
        if any(ok for tool, ok in self.facts_from.items() if tool != details):
            return
        target = next((l for l in self.listings.values() if details in l.read_by), None)
        if target is None:
            return
        mapped = {c["tool"] for role, c in self.calls.items() if role != "listing"}
        for name, schema in self.schemas.items():
            if name in mapped or name in self.facts_from:
                continue
            props = schema.get("properties") or {}
            ids = [k for k in props if _ID_KEY.search(k) or re.search(r"id|ref|code|which|product|item|listing", k, re.I)]
            if not ids or len(props) > 2 or set(schema.get("required") or []) - set(ids):
                continue
            args = {ids[0]: target.id}
            data = self.env_call(name, args)
            self.facts_from.setdefault(name, False)
            if data is None:
                return
            self.absorb(name, args, data.get("observation"))
            self.facts_from.setdefault(name, False)
            note = ""
            if (self.calls.get("details") or {}).get("tool") == name:
                note = (f"\n[The property records come from {name}: the system now uses it as the details tool. Read the plausible "
                        f"listings' properties with it; {details} shows only their page.]")
            self.inject(name, args, _compact(_brief(data.get("observation"))) + note)
            self.dialogue.append({"kind": "details_probe", "tool": name, "table": self.facts_from.get(name)})
            return

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> None:
        role = self.role_of(name, args)
        if role is None and self.single_listing(name, args):
            role = "listing"  # an unmapped tool showing one listing: keep its record (race 11's property table)
        # what was known of this listing before its page: walk() below merges the page in (shop_name reads as a title)
        # (field names of search rows count whatever their value: Listing.fields drops empty ones, and practice rows carry
        # "company": null, which made practice's view_listing look like it adds a field)
        page_known: set[str] | None = None
        if role == "listing" and self.page_info is None and self.row_keys:
            known = self.listings.get(str(next(iter(args.values()), "")))
            page_known = self.row_keys | set((known.record if known is not None and isinstance(known.record, dict) else {}) or {})

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                key = _id_field(value)
                if key is not None:
                    ident = str(value[key])
                    listing = self.listings.setdefault(ident, Listing(ident))
                    listing.merge({k: v for k, v in value.items() if not isinstance(v, (dict, list))}, self.step)
                for item in value.values():
                    walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        if role in ("search", "browse"):
            self.read_shape(name, observation)
            self.adopt_category()
        if role not in ("cart_view",):
            walk(observation, 0)
            self.stamp_learned(observation)
        ident = None
        if role in ("details", "listing"):
            for key, value in args.items():
                if isinstance(value, (str, int)) and str(value) and _ID_KEY.search(key) or (isinstance(value, str) and value in self.listings):
                    ident = str(value)
                    break
            if ident is None and args:
                ident = str(next(iter(args.values())))
            record = _main_record(observation, ident)
            if page_known is not None and isinstance(record, dict) and not _failed(record):
                self.page_info = bool(set(record) - page_known)
            if isinstance(record, dict) and not _failed(record):
                listing = self.listings.setdefault(ident, Listing(ident))
                if isinstance(listing.record, dict):
                    merged = dict(listing.record)
                    merged.update(record)
                    listing.record = merged
                else:
                    listing.record = record
                listing.merge({k: v for k, v in record.items() if not isinstance(v, (dict, list))}, self.step)
                listing.read_by.add(name)
                self.note_facts(name, record)
                role = self.role_of(name, args) or role
                if role == "details" or not self.calls.get("details") or self.facts_from.get(name):
                    listing.inspected = True
                if role == "listing":
                    listing.paged = True
                    listing.notes_read = False  # the page may carry notes the record lacked
        elif role in ("search", "browse"):
            count = None
            more = False
            if isinstance(observation, dict):
                for key, value in observation.items():
                    if _COUNT_KEY.search(key) and isinstance(value, (int, float)) and not isinstance(value, bool):
                        count = int(value)
                    if _MORE_KEY.search(key) and value not in (None, False, 0, "", []):
                        more = True
            rows = 0
            for record in self._listing_dicts(observation):
                rows += 1
                self.row_keys.update(str(k) for k in record)
                self.found_by.setdefault(str(record[_id_field(record)]), dict(args))
            self.searches.append(f"{_compact(args)[:120]} -> {rows} rows" + (f" of {count}" if count is not None else "") + (" (more pages)" if more else ""))
            page = self.page_param(name)
            self.query_keys.add(_compact({k: v for k, v in args.items() if k != page}))
        elif role == "cart_view":
            self.cart = self._parse_cart(observation)
            self.cart_view_step = self.step
            walk(observation, 0)
        elif role in ("cart_add", "cart_remove"):
            self.cart_edit_step = self.step
            ident = str(next((v for k, v in args.items() if self.calls[role]["args"].get(k) == "<ID>"), ""))
            qty = _int(next((v for k, v in args.items() if self.calls[role]["args"].get(k) == "<QTY>"), 1)) or 1
            if ident and not _failed(observation):
                if role == "cart_add":
                    self.added[ident] = qty
                else:
                    self.added.pop(ident, None)
        if isinstance(observation, dict) and any(re.search(r"cart|basket|selection|bag", str(key), re.I) for key in observation):
            parsed = self._parse_cart(observation)
            if parsed:
                self.cart = parsed
                self.added = dict(parsed)
                self.cart_view_step = self.step
                self.cart_edit_step = self.step
        elif role == "browse" or role is None:
            pass

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        # Which keys of a row-list result hold the rows, the match count, the next-page value, a row's price and its
        # stock flag. Named once per tool, then checked against the result. A miss leaves the regex readers in charge.
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict) or self.over_budget():
            return None
        lists = [key for key, value in observation.items() if isinstance(value, list) and len(value) >= 2 and all(isinstance(row, dict) for row in value)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {key: (value[:2] if key in lists else value) for key, value in observation.items()}
        content = f"TOOL: {name}\nPARAMETERS: {_compact(self.schemas.get(name) or {})}\n\nRESULT:\n{_compact(_brief(sample, 400, 8))}"
        got = self.ask_json(SHAPE_PROMPT, content, MAIN_MODELS, 400) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows or not observation[rows]:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {
            "rows": rows,
            "count": named(got.get("count"), observation),
            "next": named(got.get("next"), observation),
            "next_arg": named(got.get("next_arg"), props),
            "price": named(got.get("price"), first) if isinstance(first, dict) else None,
            "stock": named(got.get("stock"), first) if isinstance(first, dict) else None,
        }
        count_key = shape["count"]
        if count_key and (isinstance(observation[count_key], bool) or not isinstance(observation[count_key], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.learned_price = self.learned_price or shape["price"]
        self.learned_stock = self.learned_stock or shape["stock"]
        self.dialogue.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def stamp_learned(self, observation: Any) -> None:
        if not self.learned_price and not self.learned_stock:
            return
        for record in self._listing_dicts(observation):
            listing = self.listings.get(str(record.get(_id_field(record) or "")))
            if listing is None:
                continue
            if self.learned_price:
                listing.price_key = self.learned_price
            if self.learned_stock:
                listing.stock_key = self.learned_stock

    def page_from_shape(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        # (matches, rows shown, the parameter that asks for the next page, its value or None on the last page).
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def page_param(self, tool: str) -> str | None:
        # The parameter of a listing tool that asks for a later page of its results.
        own = {k for call in self.calls.values() if call["tool"] == tool for k, v in call["args"].items() if v in ("<TEXT>", "<ID>", "<QTY>")}
        for key, spec in (self.schemas.get(tool, {}).get("properties") or {}).items():
            if key in own or (isinstance(spec, dict) and spec.get("enum")):
                continue
            if _PAGE_PARAM.search(key) and not re.search(r"cap|max|price|budget|min|limit|size|per", key, re.I):
                return key
        # an unnamed position argument: the one integer argument that is no filter or limit (next_page_value then needs
        # an integer result field at offset + rows before it pages)
        ints = [key for key, spec in (self.schemas.get(tool, {}).get("properties") or {}).items()
                if key not in own and isinstance(spec, dict) and spec.get("type") == "integer" and not spec.get("enum")
                and not re.search(r"cap|max|min|price|budget|cost|limit|size|per|qty|quant|count|rating|star|year|number", key, re.I)]
        return ints[0] if len(ints) == 1 else None

    def next_page_value(self, tool: str, args: dict[str, Any], observation: Any, param: str, rows: int) -> Any:
        # What to pass as the page parameter for the next page, or None when the results end here.
        if not isinstance(observation, dict):
            return None
        spec = (self.schemas.get(tool, {}).get("properties") or {}).get(param)
        kind = spec.get("type") if isinstance(spec, dict) else None
        more = False
        for key, value in observation.items():
            if isinstance(value, (dict, list)) or value is None or not _MORE_KEY.search(key):
                continue
            if isinstance(value, bool):
                more = more or value
            elif _TOKEN_KEY.search(key) and (isinstance(value, (int, float)) or str(value).strip()):
                if kind == "integer" and isinstance(value, str):
                    return int(value) if value.strip().isdigit() else None
                return value
            elif isinstance(value, (int, float)) and value > 0:
                more = True  # rows remaining
        if not more and kind in ("integer", "number"):
            # an unnamed offset: an integer result field equal to where the next page starts
            current = _int(args.get(param)) or 0
            for key, value in observation.items():
                if isinstance(value, int) and not isinstance(value, bool) and not _COUNT_KEY.search(key) and value == current + rows and rows > 0:
                    return value
        if not more or kind not in ("integer", "number"):
            return None
        current = _int(args.get(param))
        if re.search(r"offset|start|skip|row", param, re.I):
            return (current or 0) + rows
        return (current if current is not None else 1) + 1

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        # Fetch the next pages of a search until the query has shown SEARCH_ROWS rows.
        #
        # Race 8 pages hold 4 rows: a quarter (LV) to a half (CHIDORI) of the passing orders were first seen on page 2+,
        # and KISAME, which seldom paged, declined wrongly on tasks LV passed.
        learned = self.page_from_shape(name, observation)
        param = (learned[2] if learned else None) or self.page_param(name)
        if not param:
            return ""
        # A cheapest request has to see every row the store will show. This store sorts matches by id and
        # caps the window at three short pages, so stopping at 12 rows hides the later ids.
        cheap = any(p.get("cheapest") or self.pick == "cheapest" for p in self.products)
        row_limit = 21 if cheap else SEARCH_ROWS
        key = name + _compact({k: v for k, v in args.items() if k != param})
        self.pages_fetched.add(key + _compact(args.get(param)))
        out: list[str] = []
        while True:
            rows = sum(1 for _ in self._listing_dicts(observation))
            self.query_rows[key] = self.query_rows.get(key, 0) + rows
            if rows == 0 or self.query_rows[key] >= row_limit or len(out) >= 2 or self.auto_pages >= AUTO_PAGES or self.steps_left() < 8:
                break
            learned = self.page_from_shape(name, observation)
            value = learned[3] if learned and learned[3] is not None else self.next_page_value(name, args, observation, param, rows)
            if value is None or key + _compact(value) in self.pages_fetched:
                break
            spec = (self.schemas.get(name, {}).get("properties") or {}).get(param) or {}
            maximum = spec.get("maximum") if isinstance(spec, dict) else None
            if isinstance(maximum, (int, float)) and isinstance(value, (int, float)) and not isinstance(value, bool) and value > maximum:
                break
            args = dict(args)
            args[param] = value
            data = self.env_call(name, args)
            if data is None:
                break
            self.auto_pages += 1
            self.pages_fetched.add(key + _compact(value))
            observation = data.get("observation")
            self.absorb(name, args, observation)
            out.append(f"{param}={_compact(value)}: " + _compact(_brief(observation, 220, 40)))
            if data.get("finished"):
                break
        if not out:
            return ""
        return "\n[the system also fetched the next page(s) of these results]\n" + "\n".join(out)

    def find_listing(self, source: str, exclude: str) -> Listing | None:
        # The seen listing a note's description of another listing names (seller, kind and values), when one stands out.
        words = _stems(source)
        low = source.lower()
        scored = []
        for listing in self.listings.values():
            if listing.id == exclude:
                continue
            # a short field value of the listing (its seller, maker, ...) written in the note
            named = any(isinstance(v, str) and 3 <= len(v) <= 40 and v.lower() != listing.title.lower() and v.lower() in low
                        and not _ID_KEY.search(k) and not _PRICE_KEY.search(k) for k, v in listing.fields.items())
            score = (3 if named else 0) + 2 * len(words & _stems(listing.category))
            score += len(words & _stems(listing.title)) + sum(1 for v in listing.properties().values() if len(v) > 2 and v.lower() in low)
            scored.append((score, listing))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] < 4 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][1]

    def follow_chains(self) -> str:
        # Race 7: open the page each reference chain on a required property passes through.
        #
        # A page note may send a property to another listing ("For closure, see the Canned Meat from Torvexzen"), whose
        # page may send it on again. The store checked that the chain's pages were opened (a skipped one failed as
        # process_requirement_missed, 0/51 passed), and the value is only on the last page.
        if not self.page_info or not self.calls.get("listing"):
            return ""
        wanted = {_base(r["text"].split(":")[0]) for p in self.products for r in p["reqs"] if r["active"] and ":" in r["text"]}
        wanted |= {_base(w) for p in self.products for r in p["reqs"] if r["active"] for w in [r["text"].split(":")[0]] if r.get("lookup")}
        done = []
        progress = True
        while progress:
            progress = False
            for listing in list(self.listings.values()):
                for prop, source in list(listing.defers):
                    if _base(prop) not in wanted or (listing.id, prop) in self.followed:
                        continue
                    if self.chain_hops >= 6 or self.steps_left() < 8:
                        return self._chain_note(done)
                    self.followed.add((listing.id, prop))
                    target = self.find_listing(source, listing.id) or self.opened_shop_page(source, listing.id)
                    if target is None and "search" in self.calls:
                        # the shop and the values in the note, then the capitalised names when that finds nothing
                        words = [w for w in re.findall(r"[A-Za-z0-9]{3,}", source) if w.lower() not in {"the", "and", "with", "from", "that", "listing", "page"}]
                        caps = [w for w in re.findall(r"[A-Za-z][\w'-]+", source) if w[0].isupper() and w.lower() not in {"the", "a", "an", "for", "from"}]
                        query = " ".join(words[:6]) or " ".join(caps[:4])
                        filled = self.fill("search", text=query)
                        if filled:
                            data = self.env_call(*filled)
                            if data is not None:
                                self.absorb(filled[0], filled[1], data.get("observation"))
                        target = self.find_listing(source, listing.id)
                    if target is None or target.paged:
                        continue
                    filled = self.fill("listing", ident=target.id)
                    if not filled:
                        return self._chain_note(done)
                    self.chain_hops += 1
                    data = self.env_call(*filled)
                    if data is None:
                        return self._chain_note(done)
                    self.absorb(filled[0], filled[1], data.get("observation"))
                    self.read_notes(target)
                    self.write_back(prop, target)
                    done.append(f"{prop} on {listing.id} -> page of {target.id} ({target.title[:60]})")
                    self.dialogue.append({"kind": "chain", "property": prop, "from": listing.id, "to": target.id})
                    progress = True
        return self._chain_note(done)

    def opened_shop_page(self, source: str, exclude: str) -> Listing | None:
        # An already opened page of the shop the note names, when its record contains the note's value words. Used before a search.
        low = source.lower()
        skip = {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for", "of"}
        opened: list[Listing] = []
        named: set[str] = set()
        for listing in self.listings.values():
            if listing.id == exclude or not (listing.paged or listing.inspected):
                continue
            names = [str(value).strip() for key, value in listing.fields.items()
                     if isinstance(value, str) and 2 < len(value.strip()) <= 40 and re.search(r"seller|shop|store|vendor|brand|maker|merchant", key, re.I)]
            if not any(re.search(r"(?<![a-z0-9])" + re.escape(name.lower()) + r"(?![a-z0-9])", low) for name in names):
                continue
            opened.append(listing)
            for name in names:
                named.update(re.findall(r"[a-z0-9]{3,}", name.lower()))
        values = {word for word in re.findall(r"[a-z0-9]+", low) if len(word) >= 3 or word.isdigit()} - named - skip
        for listing in opened:
            text = self.page_text(listing).lower()
            if values and all(re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) for word in values):
                return listing
        return None

    def write_back(self, prop: str, target: Listing) -> None:
        # The page that ends a reference chain states the value. Write that value onto the requirement.
        value = ""
        for key, stated in target.properties().items():
            if _words(key) & _words(prop) or _base(key) == _base(prop):
                value = str(stated).strip()
                break
        if not value:
            return
        for product in self.products:
            for req in product["reqs"]:
                if not req["active"] or ":" not in req["text"]:
                    continue
                head = req["text"].split(":", 1)[0]
                if _base(head) != _base(prop):
                    continue
                req["text"] = f"{head.strip()}: {value}"
                req["lookup"] = False
                req["fresh"] = True

    def _chain_note(self, done: list[str]) -> str:
        if not done:
            return ""
        return ("The system followed the reference chain(s) and opened these pages: " + "; ".join(done)
                + ". The value counts from the last page in each chain (see NOTES SAY / PAGE in the STATE); record it with "
                + f"{self.note_tool} and match the wanted product against it.")

    def read_notes(self, listing: Listing) -> None:
        # Ask once per inspected listing what its description says about reading its properties.
        if listing.notes_read or not (listing.inspected or listing.paged):
            return
        listing.notes_read = True
        text = "\n".join(listing.notes())[:2500]
        if not text:
            return
        result = self.read_cache.get(text)
        if result is None:
            if self.note_reads >= NOTE_READS or self.steps_left() < 6 or self.over_budget():
                return
            self.note_reads += 1
            content = (
                f"LISTING {listing.id}: {listing.title}\nPROPERTY RECORD: {_compact(listing.properties())}\n\n"
                f"DESCRIPTION AND NOTES:\n{text}"
            )
            # a note that sends a property to another listing is read by the strong model (the small one misread them)
            result = self.ask_json(NOTES_PROMPT, content, STRONG_MODELS if _REFERENCE.search(text) else MAIN_MODELS, 400) or {}
            self.read_cache[text] = result
            self.dialogue.append({"kind": "notes", "listing": listing.id, "read": result})
        for item in result.get("defers") or []:
            # one property per deferral; a reading that lists several is a misread
            if isinstance(item, dict) and item.get("property") and "," not in str(item["property"]) \
                    and _base(str(item["property"])) not in {_base(p) for p, _ in listing.defers}:
                read = f"its {item['property']} is the value on another listing: {str(item.get('source') or '?')[:200]}"
                if read not in listing.reads:
                    listing.reads.append(read)
                    listing.defers.append((str(item["property"]), str(item.get("source") or "")))
        for item in result.get("corrections") or []:
            if str(item).strip() and f"the notes state {str(item)[:120]}" not in listing.reads:
                listing.reads.append(f"the notes state {str(item)[:120]}")
        for rule in result.get("rules") or []:
            rule = f"Store rule (from notes of {listing.id}): {str(rule).strip()[:200]}"
            if rule not in self.rules:
                self.rules.append(rule)

    def _listing_dicts(self, value: Any, depth: int = 0):
        if depth > 6:
            return
        if isinstance(value, dict):
            if _id_field(value):
                yield value
            for item in value.values():
                yield from self._listing_dicts(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                yield from self._listing_dicts(item, depth + 1)

    def _parse_cart(self, observation: Any) -> dict[str, int]:
        cart: dict[str, int] = {}
        lines: list[tuple[dict[str, Any], str]] = []

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                key = _id_field(value) or next(
                    (k for k, v in value.items() if isinstance(v, str) and _ID_KEY.search(str(k)) and v.strip()), None
                )
                if key is not None:
                    lines.append((value, key))
                    return
                for item in value.values():
                    visit(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    visit(item, depth + 1)

        visit(observation, 0)
        for record, key in lines:
            qty = None
            for name, value in record.items():
                if _QTY_KEY.search(name) and not _PRICE_KEY.search(name) and isinstance(value, (int, float)) and not isinstance(value, bool):
                    qty = int(value)
                    break
            cart[str(record[key])] = qty if qty is not None else 1
        if not cart and self.added and any(ident in _compact(observation) for ident in self.added):
            return dict(self.added)
        return cart

    def shopper_lines(self, name: str, args: dict[str, Any], observation: Any) -> list[str]:
        role = self.role_of(name, args)
        strings: list[str] = []
        _strings(observation, strings, skip_listings=True)
        lines = []
        for text in strings:
            text = text.strip()
            # a reply to our message is read even when the shopper repeats a canned line (s11: the repeat went unread and
            # the model read the bare record as a dropped requirement)
            if not text or (text in self.heard and role != "message"):
                continue
            if role == "message" or (len(text) >= 20 and len(text.split()) >= 4 and not _SYSTEM_TEXT.search(text)):
                self.heard.add(text)
                lines.append(text)
        if lines:
            self.last_heard = lines
            self.last_heard_step = self.step
        return lines

    # ------------------------------------------------------------------ ledger

    def product(self, label: Any) -> dict[str, Any]:
        label = str(label or "").lower()
        for product in self.products:
            if product["label"].lower() == label:
                return product
        for product in self.products:
            if label and (label in product["label"].lower() or product["label"].lower() in label or _words(label) & _words(product["label"])):
                return product
        return self.products[0]

    def interpret(self, lines: list[str], question: str | None) -> dict[str, Any]:
        context = self.ledger_text()
        content = f"{context}\n\n"
        if question:
            content += f"THE ASSISTANT ASKED: {question}\n"
        content += "THE SHOPPER SAID:\n" + "\n".join(f"- {line}" for line in lines)
        # a reply to a question may carry a hidden need: read it with the strong model; change notices are simpler
        models = (HEAVY_MODELS if self.spend < SPEND_SOFT * 0.6 else STRONG_MODELS) if question else MAIN_MODELS
        result = self.ask_json(INTERPRET_PROMPT, content, models, 700)
        if result is None:
            # a failed reading would lose a reveal: read these lines again later (KISAME's catch_up)
            self.unread.append((lines, question))
            result = {}
        self.apply_interpretation(result, lines)
        return result

    def catch_up(self) -> str:
        # Read again the shopper lines whose reading failed (a lost reveal fails the order).
        if not self.unread or self.steps_left() < 4:
            return ""
        if self.unread_tries >= 3:
            self.unread.clear()
            return ""
        self.unread_tries += 1
        lines, question = self.unread.pop(0)
        result = self.interpret(lines, question)
        if not result:
            return ""
        return f"(The system read again a shopper reply whose reading had failed: {result.get('summary') or result.get('reply_type')}. The STATE shows the result.)"

    def apply_interpretation(self, result: dict[str, Any], lines: list[str]) -> None:
        for item in result.get("add") or []:
            if isinstance(item, dict) and item.get("requirement"):
                product = self.product(item.get("product"))
                text = str(item["requirement"])
                product["reqs"].append({"text": text, "source": "revealed", "active": True, "lookup": bool(item.get("lookup")), "fresh": True})
                if item.get("lookup"):
                    self.todo.append(
                        f"Resolve the {product['label']} requirement '{text[:160]}': find the listing it points to, "
                        + (f"open its page with {self.calls['listing']['tool']} (follow any note there that sends you to another listing, opening that page too), " if self.calls.get("listing") else "")
                        + f"read the value, then record it with {self.note_tool}"
                        + (" as 'property: NOT <value>' (the shopper wants anything except that value)." if re.search(r"\bNOT\b", text) else ".")
                    )
        # a requirement is dropped only on words that drop it ("skip it", "forget the ...", "doesn't matter now"), never on
        # a plain "no preference" to a question
        dropping = bool(re.search(r"skip|forget|drop|no longer|not (needed|necessary|required|important)|isn.?t worth|"
                                  r"(doesn.?t|does not|won.?t) matter|let (it|that|the \w+) go|never ?mind|give up on|leave (it|that) out|"
                                  r"cross (it|that) off|scrap|don.?t (need|bother|worry about)", " ".join(lines), re.I))
        for item in (result.get("drop") or []) if dropping else []:
            if isinstance(item, dict) and item.get("property"):
                product = self.product(item.get("product"))
                name = str(item["property"])
                prop = _words(name)
                active = [r for r in product["reqs"] if r["active"]]
                # the requirement on that property only: "size" never drops "display size", and "plug type" never drops
                # "collar type" (both have the base "type"); a looser match counts only when it is the one candidate
                said = _words(" ".join(lines))
                def named(req: dict[str, Any]) -> bool:
                    # the shopper's words name every word of the requirement's own property
                    head = _words(req["text"].split(":")[0])
                    return bool(head) and head <= said

                hits = [r for r in active if prop and (_words(r["text"].split(":")[0]) == prop or _base(r["text"].split(":")[0]) == _base(name))]
                if not hits:
                    loose = [r for r in active if prop and named(r)
                             and (prop <= _words(r["text"].split(":")[0]) or _words(r["text"].split(":")[0]) <= prop or prop <= _words(r["text"]))]
                    hits = loose if len(loose) == 1 else []
                for req in hits:
                    req["active"] = False
                product.setdefault("dropped", []).append(name)
        budget = _number(result.get("budget"))
        literal = [float(n.replace(",", "")) for line in lines for n in re.findall(r"\d[\d,]*(?:\.\d+)?", line)]
        if budget is not None and literal and budget not in literal:
            # models sometimes rescale ("1004 cts" read as 10.04): trust the figure the shopper wrote
            rescaled = [n for n in literal if abs(n - budget * 100) < 1]
            if rescaled or len(literal) == 1:
                budget = (rescaled or literal)[0]
        if budget is not None and budget > 0:
            self.budget = budget
        revealed = [product for product in self.products if any(req.get("fresh") for req in product["reqs"])]
        if revealed:
            found = self.value_search(revealed)
            for product in revealed:
                for req in product["reqs"]:
                    req.pop("fresh", None)
            if found:
                self.todo.append(found[:400])
        if result.get("stock_out"):
            # race refusals don't say which line ("that suggestion is showing as emptied out"): a one-line cart's line is
            # the gone one; with several lines, only a line the words name, else the code re-reads each line's stock
            cart = self.current_cart()
            said = " ".join(lines).lower()
            gone = [i for i in cart if (len(i) >= 5 and re.search(rf"(?<![\w-]){re.escape(i.lower())}(?![\w-])", said)) or self.title_named(i, said)]
            if not gone and len(cart) == 1:
                gone = list(cart)
            for ident in gone:
                self.mark_gone(ident)
            if gone:
                named = ", ".join(gone)
                self.todo.append(f"{named} is no longer available: remove it from the cart and choose another.")
            elif cart:
                named = "one of the cart lines"
                self.stock_recheck = [i for i in cart if i not in self.stock_recheck] + self.stock_recheck
            else:
                named = "the chosen item"
            result = dict(result)
            result["summary"] = f"{named} sold out ({result.get('summary') or ''})"[:200]
        if result.get("price_changed"):
            self.todo.append("A chosen item's price changed: view the cart or the listing for the new price, then recheck the budget and whether it is still the best choice.")
        # race 7 (KISAME's ledger): the shopper or the store may add a procedure or a new allowance midway; it binds from then on
        added = [str(i).strip()[:300] for i in result.get("instructions") or [] if isinstance(i, str) and str(i).strip()]
        added = [i for i in added if i not in self.rules][:4]
        if added:
            self.rules.extend(added)
            if _APPROVAL.search(" ".join(added)) and "message" in self.calls:
                self.basket_check = True
                self.basket_sent_for = None
                self.reserve_messages = max(self.reserve_messages, 1)
        messages_total = _int(result.get("messages_total"))
        messages_more = _int(result.get("messages_left"))
        if messages_more is not None and messages_more >= 0:
            messages_total = self.messages_used + messages_more
        if messages_total is not None and messages_total >= 0:
            self.message_budget = max(messages_total, self.messages_used)
        steps_total = _int(result.get("steps_total"))
        steps_more = _int(result.get("steps_left"))
        if steps_more is not None and steps_more >= 0:
            steps_total = self.step + steps_more
        if steps_total and self.step < steps_total < self.max_steps:
            self.max_steps = steps_total
        summary = str(result.get("summary") or "")
        if result.get("add") or result.get("drop") or budget or result.get("stock_out") or result.get("price_changed"):
            self.verified_cart = {}  # checked against requirements, budget or stock that changed
        if result.get("reply_type") in ("update",) or result.get("drop") or budget or result.get("stock_out") or result.get("price_changed") or added \
                or messages_total is not None or steps_total:
            self.updates.append(summary or " / ".join(lines)[:200])

    def title_named(self, ident: str, said: str) -> bool:
        listing = self.listings.get(ident)
        words = [w for w in re.findall(r"[a-z0-9]+", (listing.title if listing else "").lower()) if len(w) > 3][:5]
        return len(words) >= 2 and sum(1 for w in words if w in said) >= max(2, len(words) - 1)

    def mark_gone(self, ident: str) -> None:
        listing = self.listings.get(ident)
        if listing is not None:
            listing.fields["sold_out_now"] = True
        self.refused_ids.add(ident)
        self.verified_cart = {}

    def recheck_stock(self) -> str:
        # After a stock-out the store did not pin on one line: re-read each cart line's record (KISAME re-reads the
        # cart's stock), so the good lines of a multi-product basket stay in it.
        if not self.stock_recheck:
            return ""
        idents, self.stock_recheck = self.stock_recheck, []
        role = "details" if "details" in self.calls else ("listing" if "listing" in self.calls else None)
        gone, unknown = [], []
        for index, ident in enumerate(idents):
            filled = self.fill(role, ident=ident) if role else None
            # keep a step for each line still to read, a swap and the order (KISAME's refusal_queue)
            if not filled or self.steps_left() < len(idents) - index + 3:
                unknown.append(ident)
                continue
            data = self.env_call(*filled)
            if data is None:
                unknown.append(ident)
                continue
            observation = data.get("observation")
            self.absorb(filled[0], filled[1], observation)
            record = _main_record(observation, ident)
            fresh = Listing(ident)
            if isinstance(record, dict):
                fresh.merge({k: v for k, v in record.items() if not isinstance(v, (dict, list))}, self.step)
            if fresh.in_stock is False:
                gone.append(ident)
                self.mark_gone(ident)
            elif fresh.in_stock is None:
                unknown.append(ident)
        if gone:
            self.todo.append(f"{', '.join(gone)} is no longer available: remove it from the cart and choose another.")
            twins = self.twin_hint({i: 1 for i in gone})
            return (f"The store said a cart line sold out without saying which; the system re-read the cart lines' records: "
                    f"{', '.join(gone)} is now out of stock (the other lines are still available). Replace only that line."
                    + (f"\n{twins}" if twins else ""))
        if unknown:
            return (f"The store said a cart line sold out without saying which, and the re-read records don't show which "
                    f"(unclear: {', '.join(unknown)}). View the cart or open those listings to find the gone one.")
        return ("The store said a cart line sold out without saying which, but every line's record still reads in stock. "
                "Order the same cart again once; if the store refuses again, replace the line it names.")

    def ledger_text(self) -> str:
        parts = ["WANTED:"]
        for index, product in enumerate(self.products, 1):
            flags = " (must be the CHEAPEST option that meets everything)" if product["cheapest"] else ""
            parts.append(f"{index}. {product['label']} x{product['quantity']}{flags}")
            if product.get("category"):
                parts.append(f"   - [stated] kind: \"{product['category']}\" (only listings in the store category named like this count)")
            for req in product["reqs"]:
                if req["active"]:
                    state = ""
                elif req.get("resolved"):
                    state = " [STILL REQUIRED, in the resolved form below]"
                else:
                    state = " [DROPPED by the shopper, no longer required]"
                parts.append(f"   - [{req['source']}] {req['text']}{state}")
            if not product["reqs"]:
                parts.append("   - (no requirement recorded beyond the request text)")
        budget = "none stated" if self.budget is None else f"{self.budget:g} ({self.budget_scope})"
        parts.append(f"BUDGET: {budget}")
        if self.pick == "cheapest":
            parts.append("PICK: the cheapest listing that meets everything")
        elif self.pick in ("most", "least"):
            parts.append(f"PICK: the {self.pick} {self.pick_measure} among listings that meet everything".replace("  ", " "))
        if self.process:
            parts.append("PROCESS: " + " | ".join(f"{p['when']}: {p['step']}" + (" [done]" if p.get("done") else "") for p in self.process))
        if self.decline_rule:
            parts.append(f"DECLINE RULE: {self.decline_rule}")
        if self.rules:
            parts.append("RULES: " + " | ".join(self.rules))
        if self.updates:
            parts.append("SHOPPER UPDATES (latest last): " + " | ".join(self.updates[-8:]))
        if self.notes:
            parts.append("NOTES: " + " | ".join(self.notes[-12:]))
        return "\n".join(parts)

    def listing_line(self, listing: Listing, full: bool = False) -> str:
        price = listing.price
        stock = listing.in_stock
        bits = [listing.id, f"price {price:g}" if price is not None else "price ?", {True: "in stock", False: "OUT OF STOCK", None: "stock ?"}[stock]]
        if listing.category:
            bits.append(listing.category[:40])
        bits.append(listing.title[:90])
        line = " | ".join(bits)
        if listing.inspected:
            props = listing.properties()
            line += " | props: " + "; ".join(f"{k}={v}" for k, v in list(props.items())[: 40 if full else 18])
        else:
            line += " | (property record not inspected)" + (" | page opened" if listing.paged else "")
        if listing.inspected or listing.paged:
            refs = listing.references()
            if refs:
                line += " | NOTES SAY: " + " / ".join(refs)[:400]
            if listing.paged and not listing.inspected:
                page = " / ".join(listing.notes())[:300]
                if page:
                    line += f" | PAGE: {page}"
        return line

    def state_text(self) -> str:
        parts = [
            f"STATE at step {self.step} of {self.max_steps} ({self.steps_left()} steps left), "
            f"messages used {self.messages_used} of {self.message_budget if self.message_budget is not None else '?'}.",
            self.ledger_text(),
        ]
        if self.questions:
            parts.append("QUESTIONS ASKED: " + " | ".join(f"\"{q['text']}\" -> {q.get('type')}: {q.get('reply', '')[:160]}" for q in self.questions))
        cart = self.cart_lines_text()
        parts.append("CART: " + cart)
        inspected = [l for l in self.listings.values() if l.inspected]
        others = sorted((l for l in self.listings.values() if not l.inspected), key=lambda l: (l.price is None, l.price or 0))
        inspected.sort(key=lambda l: (l.price is None, l.price or 0))
        rows = [self.listing_line(l) for l in inspected[:30]]
        rows += [self.listing_line(l) for l in others[: max(8, 30 - len(rows))]]
        parts.append(f"LISTINGS SEEN ({len(self.listings)} total; inspected first, cheapest first):\n" + ("\n".join(rows) if rows else "none yet"))
        if self.searches:
            parts.append("SEARCHES DONE: " + " || ".join(self.searches[-12:]))
        if self.todo:
            parts.append("TODO: " + " | ".join(self.todo[-6:]))
        plan = self.fit_plan()
        if plan:
            parts.append(plan)
        if self.steps_left() <= len(self.products) + 3:
            parts.append("FEW STEPS LEFT: put the best verified choice in the cart and place the order now.")
        elif self.steps_left() <= 12:
            parts.append(
                f"{self.steps_left()} STEPS LEFT: stop broad exploring. Settle on the best choice that meets every requirement "
                "(adding, ordering and a possible retry need about 4 steps), or decline if a thorough search found nothing that fits."
            )
        return "\n\n".join(parts)

    def cart_lines_text(self) -> str:
        if self.cart is not None and self.cart_view_step >= self.cart_edit_step:
            items = self.cart
            source = f"as viewed at step {self.cart_view_step}"
        else:
            items = self.added
            source = "from your cart edits (not viewed since)"
        if not items:
            return f"empty ({source})"
        lines = []
        for ident, qty in items.items():
            listing = self.listings.get(ident)
            price = listing.price if listing else None
            lines.append(f"{ident} x{qty}" + (f" @ {price:g}" if price is not None else "") + (f" {listing.title[:60]}" if listing else ""))
        return "; ".join(lines) + f" ({source})"

    # ------------------------------------------------------------------ questions

    def property_table(self) -> str:
        groups: dict[str, dict[str, dict[str, int]]] = {}
        counts: dict[str, int] = {}
        for listing in self.listings.values():
            if not listing.inspected:
                continue
            group = listing.category or "listings"
            counts[group] = counts.get(group, 0) + 1
            table = groups.setdefault(group, {})
            for key, value in listing.properties().items():
                values = table.setdefault(key, {})
                values[value.lower()] = values.get(value.lower(), 0) + 1
        lines = []
        for group, table in groups.items():
            lines.append(f"[{group}] ({counts[group]} inspected listings)")
            for key, values in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
                shown = ", ".join(f"{v} ({n})" for v, n in sorted(values.items(), key=lambda vn: -vn[1])[:6])
                lines.append(f"  {key} [in {sum(values.values())} of {counts[group]}]: {shown}")
        return "\n".join(lines) if lines else "(no listing inspected yet)"

    def inspected_count(self) -> int:
        return sum(1 for l in self.listings.values() if l.inspected)

    def priority_plan(self) -> dict[str, Any] | None:
        # The request's own ask-first, then a story hint, before the properties the opened listings carry.
        for item in self.ask_first:
            product = self.product(item.get("product"))
            prop = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(item.get("property") or ""), maxsplit=1)[0].strip()
            if not prop or _base(prop) in self.asked_bases(product["label"]) or self.settled(product, prop, None):
                continue
            return {"product": product["label"], "properties": [prop], "key": prop, "keys": [prop], "why": "the request says to ask this"}
        return self.hint_plan()

    def maybe_ask(self) -> None:
        if self.ask_done or "message" not in self.calls:
            return
        force, self.ask_now = self.ask_now, False
        # ask once one listing of each product is open, or a few steps have passed: waiting for three per product
        # left multi-item tasks with no question at all
        if not force and self.inspected_count() < max(1, len(self.products)) and self.step < max(6, self.max_steps // 6):
            return
        if not force and self.inspected_count() < self.ask_wait_until and self.step < self.max_steps // 2:
            return
        while not self.finished:
            if self.messages_left() <= self.reserve_messages:
                self.ask_done = True
                return
            if self.steps_left() <= 6:
                self.ask_done = True
                return
            examples: list[str] = []
            if self.pending_singles:
                label, prop, show_examples = self.pending_singles.pop(0)
                props = [prop]
                examples = self.example_values(label, prop) if show_examples else []
                self.pending_key = self.pending_keys = None
            else:
                last = self.messages_left() - self.reserve_messages == 1
                plan = self.priority_plan() or self.next_planned(force) or self.code_plan(force)
                if last and not plan:
                    plan = self.hint_plan()
                plan = plan or self.plan_question()
                if plan is None:
                    self.ask_failures += 1
                    if self.ask_failures >= 2:
                        self.ask_done = True
                    return
                if plan.get("done"):
                    # unused messages are lost chances: a product with no stated need may still hide one
                    if self.unrevealed_products():
                        self.ask_failures += 1
                        if self.ask_failures < 2:
                            continue
                    self.ask_done = True
                    return
                if plan.get("wait") and not force:
                    self.ask_wait_until = self.inspected_count() + 2
                    return
                label = self.product(plan.get("product"))["label"]
                limit = 1 if self.singles or not self.list_on else 8
                props = [str(p).strip() for p in plan.get("properties") or [] if str(p).strip()][:limit]
                if len(props) < limit:
                    more = self.code_plan(True)
                    if more and self.product(more.get("product"))["label"] == label:
                        for prop in more.get("properties") or []:
                            prop = str(prop).strip()
                            if prop and prop.lower() not in {p.lower() for p in props}:
                                props.append(prop)
                            if len(props) >= limit:
                                break
                props = self.extend_props(label, props[:limit], limit)
                self.pending_key = plan.get("key")
                self.pending_keys = list(dict.fromkeys(list(plan.get("keys") or plan.get("properties") or []) + props))
                if not props:
                    self.ask_done = True
                    return
            text = self.question_text(label, props)
            self.send_question(label, props, text)
            if self.one_question:
                self.one_question = False
                self.ask_done = True
                return

    def extend_props(self, label: str, props: list[str], limit: int) -> list[str]:
        # A list question names several properties (Halo's LIST_SIZE): the shopper answers the first one it has a need
        # on, and a don't-mind rotates the rest. Race 15: no datasheet key was known, every plan gave one property, and
        # our 284 questions named 1.0 properties each against 5.2 for the winner. The list is filled from the story's
        # hints for the product, then from the model's own list of what a buyer of that kind of product chooses by.
        if limit <= 1 or len(props) >= limit:
            return props[:limit]
        product = self.product(label)
        out = list(props)

        def add(prop: Any) -> None:
            prop = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(prop or ""), maxsplit=1)[0].strip().rstrip("?.")
            if not prop or len(prop.split()) > 4 or _not_a_need(prop, None, self.row_keys):
                return
            if prop.lower() in {p.lower() for p in out} or _base(prop) in {_base(p) for p in out}:
                return
            if _base(prop) in self.asked_bases(label) or prop.lower() in self.asked_labels(label):
                return
            if _base(prop) in {_base(r["text"].split(":")[0]) for r in product["reqs"] if r["active"]} or self.settled(product, prop, None):
                return
            out.append(prop)

        for hint in self.hints:
            named = str(hint.get("product") or "").lower()
            if len(self.products) == 1 or (named and (named in label.lower() or label.lower() in named or _words(named) & _words(label))):
                add(hint.get("property"))
        if len(out) < limit and label not in self.story_props:
            self.story_props[label] = []
            plan = self.plan_question(label)
            if plan and self.product(plan.get("product"))["label"] == label:
                self.story_props[label] = [str(p).strip() for p in plan.get("properties") or [] if str(p).strip()]
        for prop in self.story_props.get(label, []):
            if len(out) >= limit:
                break
            add(prop)
        return out[:limit]

    def asked_name(self, label: str) -> str:
        # The shopper answers a question that names the product. "the item" is not a product.
        product = self.product(label)
        name = str(product.get("label") or label or "").strip()
        if name.lower() in {"", "item", "the item", "product", "listing"}:
            name = str(product.get("category") or "").strip()
        if not name or name.lower() in {"item", "product", "listing"}:
            name = "product"
        return name

    def question_text(self, label: str, props: list[str], examples: list[str] | None = None) -> str:
        # Several properties of one product, no example values. A value example is read as another question.
        name = self.asked_name(label)
        clean: list[str] = []
        for prop in props:
            one = re.split(r"\s+(?:and|or)\s+|[/,;(]", prop, maxsplit=1)[0].strip().rstrip("?.")
            if one and one.lower() not in {item.lower() for item in clean}:
                clean.append(one)
        while len(clean) > 1:
            text = f"What {', '.join(clean[:-1])} or {clean[-1]} would you like for the {name}?"
            if len(text) <= 240:
                return text
            clean.pop()
        prop = clean[0] if clean else "style"
        return f"What {prop} would you like for the {name}?"

    def example_values(self, label: str, prop: str) -> list[str]:
        # Two values the product's listings show for the property asked (most common first), none that a requirement names.
        product = self.product(label)
        listings = self.product_listings(product)
        key_values: dict[str, dict[str, str]] = {}
        for listing in listings:
            for key, value in (listing.datasheet() or listing.properties()).items():
                key_values.setdefault(key, {})[listing.id] = str(value).lower()
        groups = _key_groups(key_values, self.kind_words(product, listings))
        category = product.get("category") or ""
        wanted = prop.lower()
        settled = " ".join(r["text"].lower() for r in product["reqs"] if r["active"])
        counts: dict[str, int] = {}
        for key, members in groups.items():
            if self.question_label(_plain_key(members), category).lower() != wanted and _base(key) != _base(prop):
                continue
            for value in key_values[key].values():
                if 0 < len(value) <= 30 and value not in settled:
                    counts[value] = counts.get(value, 0) + 1
        return [v for v, _ in sorted(counts.items(), key=lambda vc: -vc[1])][:2]

    def unrevealed_products(self) -> list[str]:
        return [p["label"] for p in self.products if not any(r["source"] == "revealed" for r in p["reqs"])]

    def coverage_text(self) -> str:
        lines = []
        for product in self.products:
            label = product["label"]
            asked = [f"{', '.join(q['props'])} -> {q.get('type')}" for q in self.questions if q["product"] == label]
            revealed = sum(1 for r in product["reqs"] if r["source"] == "revealed")
            lines.append(f"- {label}: {revealed} need(s) revealed; asked: {'; '.join(asked) or 'nothing yet'}")
        return "\n".join(lines)

    def kind_words(self, product: dict[str, Any], listings: list[Listing]) -> frozenset[str]:
        # words naming the kind of product (its label, category and the store's categories of its listings)
        return _fold_words(f"{product.get('label') or ''} {product.get('category') or ''} " + " ".join({l.category for l in listings}))

    def asked_labels(self, label: str) -> set[str]:
        # after "one at a time" only the pair's first property counts as asked: the second is asked alone later
        return {str(p).lower() for q in self.questions if q["product"] == label
                for p in (q["props"][:1] if q.get("type") == "one_at_a_time" else q["props"])}

    def candidate_rows(self, product: dict[str, Any], limit: int) -> list[dict[str, Any]]:
        # The product's unasked, unsettled properties, widest coverage first, one row per property (all its key spellings).
        listings = self.product_listings(product)
        if not listings:
            return []
        key_values: dict[str, dict[str, str]] = {}
        for listing in listings:
            for key, value in (listing.datasheet() or listing.properties()).items():
                key_values.setdefault(key, {})[listing.id] = str(value).lower()
        category = product.get("category") or max((l.category for l in listings), key=lambda c: sum(1 for l in listings if l.category == c))
        context = self.kind_words(product, listings)
        groups = _key_groups(key_values, context)
        asked = self.asked_bases(product["label"])
        asked_texts = self.asked_labels(product["label"])
        stated = {_base(r["text"].split(":")[0]) for r in product["reqs"]}
        rows: list[dict[str, Any]] = []
        for key, chance in _rank_properties_scored(" ".join(sorted(context)), key_values, len(listings)):
            members = groups[key]
            if _not_a_need(key, list(key_values[key].values()), self.row_keys):
                continue
            label = self.question_label(_plain_key(members), category)
            if {_base(m) for m in members} & (asked | stated) or label.lower() in asked_texts:
                continue
            if any(self.settled(product, m, list(key_values[m].values())) for m in members):
                continue
            counts: dict[str, int] = {}
            carried: set[str] = set()
            for m in members:
                carried |= set(key_values[m])
                for value in key_values[m].values():
                    counts[value] = counts.get(value, 0) + 1
            values = [v for v, _ in sorted(counts.items(), key=lambda vc: -vc[1]) if len(v) <= 40][:4]
            rows.append({"product": product["label"], "label": label, "key": key, "keys": members, "chance": chance,
                         "carried": len(carried), "of": len(listings), "values": values,
                         "member_values": {m: list(key_values[m].values()) for m in members}})
            if len(rows) >= limit:
                break
        return rows

    def plan_interview(self) -> list[dict[str, Any]]:
        # One call orders every question over the properties the opened listings carry: the model reads the story's hints
        # and what a buyer of that kind of product chooses between, and drops second names and terms of sale.
        if self.over_budget() or "message" not in self.calls:
            return []
        count = max(1, self.messages_left() - self.reserve_messages) + 1
        table: dict[int, dict[str, Any]] = {}
        lines: list[str] = []
        for product in self.products:
            rows = self.candidate_rows(product, 12)
            if not rows:
                continue
            settles = "; ".join(r["text"] for r in product["reqs"] if r["active"]) or "nothing"
            lines.append(f"PRODUCT {product['label']!r} ({rows[0]['of']} opened listings; the request settles: {settles})")
            for row in rows:
                number = len(table) + 1
                table[number] = row
                lines.append(f"  {number}. {row['label']} [on {row['carried']} of {row['of']}]: {', '.join(row['values'])}")
        if not table:
            return []
        hints = "\n".join(f"- {h.get('product')}: {h.get('property')} ({str(h.get('detail') or '')[:80]})" for h in self.hints) or "(none)"
        content = (f"REQUEST:\n{self.request}\n\nCANDIDATE PROPERTIES (widest coverage first):\n" + "\n".join(lines)
                   + f"\n\nSTORY DETAILS THAT MAY HINT AT A PROPERTY:\n{hints}\n\nChoose up to {count}.")
        got = self.ask_json(INTERVIEW_PROMPT.replace("COUNT", str(count)), content, HEAVY_MODELS, 700) or {}
        picks: list[dict[str, Any]] = []
        named = 0
        for pick in got.get("ask") or []:
            if not isinstance(pick, dict):
                continue
            number = _int(pick.get("pick"))
            why = str(pick.get("why") or "")[:80]
            if number in table and all(p["keys"] != table[number]["keys"] for p in picks):
                picks.append(dict(table[number], why=why))
            elif number == 0 and str(pick.get("label") or "").strip():
                # a property no opened listing names: a hinted one, or one buyers of that kind of product choose by
                label = str(pick["label"]).strip()[:40]
                product = self.product(pick.get("product"))
                if not _not_a_need(label, None, self.row_keys) and not self.settled(product, label, None) \
                        and _base(label) not in self.asked_bases(product["label"]) \
                        and all(p["product"] != product["label"] or _base(p["label"]) != _base(label) for p in picks):
                    named += 1
                    picks.append({"product": product["label"], "label": label, "key": label, "keys": [label], "member_values": {}, "why": why})
        picks = self.lane_order(self.guard_common(picks))
        self.dialogue.append({"kind": "interview_plan", "count": count, "picks": [(p["product"], p["label"]) for p in picks],
                              "candidates": len(table)})
        return picks

    def lane_order(self, picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Each product's first property before any product's second, so one product cannot spend the message budget.
        buckets: list[list[dict[str, Any]]] = []
        for product in self.products:
            buckets.append([pick for pick in picks if pick["product"] == product["label"]])
        placed: list[dict[str, Any]] = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    placed.append(bucket.pop(0))
        placed.extend(pick for pick in picks if pick not in placed)
        return placed

    def guard_common(self, picks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # A property carried by far more of a product's listings than a non-hinted pick enters its first three questions.
        guarded = list(picks)
        for product in self.products:
            wide = sorted(
                (row for row in self.candidate_rows(product, 12) if row["of"] and not _CATCH_ALL.search(_norm_key(row["label"]))),
                key=lambda row: -row["carried"] / row["of"],
            )[:3]
            for row in wide:
                share = row["carried"] / row["of"]
                own = [index for index, pick in enumerate(guarded) if pick["product"] == product["label"]]
                if any(_base(guarded[index]["label"]) == _base(row["label"]) for index in own[:3]):
                    continue

                def shown(pick: dict[str, Any]) -> float:
                    return pick["carried"] / pick["of"] if pick.get("of") else 0.0

                def hinted(pick: dict[str, Any]) -> bool:
                    return bool(re.search(r"hint|story", str(pick.get("why") or ""), re.I))

                weak = next((
                    index for index in own[:3]
                    if not hinted(guarded[index]) and shown(guarded[index]) + _COVERAGE_GAP <= share
                ), None)
                if weak is None and len(own) >= 3:
                    continue
                at = own[weak] if weak is not None else (own[-1] + 1 if own else len(guarded))
                front = [guarded[index] for index in own[:3]]
                guarded.insert(at, {**row, "why": "carried by most listings"})
                after = [index for index, pick in enumerate(guarded) if pick["product"] == product["label"]]
                kept = {_base(guarded[index]["label"]) for index in after[:3]}
                if any(hinted(pick) and _base(pick["label"]) not in kept for pick in front):
                    guarded.pop(at)
        return guarded

    def value_search(self, products: list[dict[str, Any]]) -> str:
        # After a reveal, search the stated values together and open up to three rows of that product.
        if "search" not in self.calls or self.steps_left() < 8 or self.over_budget():
            return ""
        seen: list[str] = []
        for product in products:
            values = []
            for req in product["reqs"]:
                if req.get("source") != "revealed" or not req["active"] or req.get("lookup") or ":" not in req["text"]:
                    continue
                raw = req["text"].split(":", 1)[1].strip()
                if _limit_bound(raw):
                    prop = req["text"].split(":", 1)[0].strip()
                    if prop:
                        values.append(prop)
                    continue
                value = re.sub(r"^(?:at least|at most|under|over|one of|not|no)\s+", "", raw, flags=re.I).strip()
                if value and not re.search(r"\b(not|no|other than)\b", value, re.I):
                    values.append(value)
            if not values or self.steps_left() < 6:
                continue
            fresh = [req["text"].split(":", 1)[1].strip() for req in product["reqs"] if req.get("fresh") and ":" in req["text"]]
            tries = [values]
            if len(values) > 1:
                # the first search uses every value; if it opens nothing, drop one and search again
                stated_at = [index for index, value in enumerate(values) if value not in fresh]
                drop = stated_at[-1] if stated_at else 0
                shorter = values[:drop] + values[drop + 1:]
                if shorter:
                    tries.append(shorter)
            opened: list[str] = []
            for words in tries[:2]:
                if not words or self.steps_left() < 4:
                    break
                filled = self.fill("search", text=" ".join(words[:4])[:140])
                if not filled:
                    break
                data = self.env_call(*filled)
                if data is None:
                    break
                observation = data.get("observation")
                self.absorb(filled[0], filled[1], observation)
                found_ids = {str(record[_id_field(record)]) for record in self._listing_dicts(observation) if _id_field(record)}
                rows = []
                for ident in found_ids:
                    listing = self.listings.get(ident)
                    owner = self.owner(listing)
                    if listing is None or listing.inspected or listing.in_stock is False or owner not in (product, None):
                        continue
                    if listing.price is not None and self.budget is not None and listing.price * product["quantity"] > self.budget and self.budget_scope != "each":
                        continue
                    rows.append(listing)
                for listing in rows[:3]:
                    if self.steps_left() < 4 or "details" not in self.calls:
                        break
                    detail = self.fill("details", ident=listing.id)
                    if not detail:
                        break
                    opened_data = self.env_call(*detail)
                    if opened_data is None:
                        break
                    self.absorb(detail[0], detail[1], opened_data.get("observation"))
                    opened.append(listing.id)
                if opened:
                    break
            seen.append(product["label"] + ": " + (", ".join(opened) or "no fitting row"))
        if not seen:
            return ""
        self.dialogue.append({"kind": "value_search", "seen": seen})
        return "Searched the revealed values and opened: " + "; ".join(seen)

    def browse_category(self, product: dict[str, Any]) -> str:
        # List the product's category with the browse tool once (and read the next pages the system fetches), so the
        # interview and the picks see rows the keyword search misses. Race 15: the winner listed each item's category
        # 3 times per episode; our keyword searches returned no rows for most queries there.
        label = product["label"]
        if "browse" not in self.calls or label in self.browsed or self.steps_left() < 10 or self.finished:
            return ""
        category = str(product.get("category") or product.get("label") or "").strip()
        if not category:
            return ""
        self.browsed.add(label)
        filled = self.fill("browse", text=category)
        if not filled:
            return ""
        data = self.env_call(*filled)
        if data is None:
            return ""
        observation = data.get("observation")
        self.absorb(filled[0], filled[1], observation)
        rows = sum(1 for _ in self._listing_dicts(observation))
        text = _compact(_brief(observation, 220, 40))
        if rows and not data.get("finished"):
            text += self.more_pages(filled[0], filled[1], observation)
        self.inject(filled[0], filled[1], text[:9000])
        self.dialogue.append({"kind": "browse", "product": label, "category": category, "rows": rows})
        return f"The system listed the category {category!r} for the {label} ({rows} rows on the first page)."

    def widen(self) -> None:
        # KYO: before the interview, read more property records so the planner does not judge from two or three listings.
        # Up to two unread rows per wanted product that look like it (in stock, inside the budget; cheapest first when the
        # product must be the cheapest), four in all, only while 22+ steps remain.
        if "details" not in self.calls or self.steps_left() < 22 or self.over_budget():
            return
        opened = 0
        for product in self.products:
            room = max(0, min(2, 6 - len(self.product_listings(product))))
            if not room:
                continue
            wanted = _stems(f"{product.get('category') or ''} {product['label']}")
            rows = []
            for listing in self.listings.values():
                if listing.inspected or listing.id in self.refused_ids or listing.in_stock is False:
                    continue
                if listing.price is not None and self.budget is not None and listing.price * product["quantity"] > self.budget:
                    continue
                score = len(wanted & _stems(listing.category)) * 2 + len(wanted & _stems(listing.title))
                if score <= 0 or (len(self.products) > 1 and self.owner(listing) is not product):
                    continue
                cheap = listing.price if product.get("cheapest") and listing.price is not None else 0
                rows.append((cheap, -score, listing))
            rows.sort(key=lambda r: (r[0], r[1]))
            for _, _, listing in rows[:room]:
                if opened >= 4 or self.steps_left() < 22:
                    break
                filled = self.fill("details", ident=listing.id)
                if not filled:
                    return
                data = self.env_call(*filled)
                if data is None:
                    return
                self.absorb(filled[0], filled[1], data.get("observation"))
                self.inject(filled[0], filled[1], _compact(_brief(data.get("observation"), 300, 20)))
                opened += 1
        if opened:
            self.dialogue.append({"kind": "widen", "opened": opened})

    def next_planned(self, force: bool) -> dict[str, Any] | None:
        # The planner's next question that is still open (a reveal may have settled a later one); None hands over to
        # coverage order, which also waits while a wanted product has no opened listing.
        if self.interview is None:
            if not force and any(not self.product_listings(p) for p in self.products) and self.step < self.max_steps // 3:
                return None
            for product in self.products:
                if not self.product_listings(product):
                    self.browse_category(product)
            self.widen()
            self.interview = self.plan_interview()
        def open_item(item: dict[str, Any]) -> bool:
            product = self.product(item["product"])
            members = item["keys"]
            if {_base(m) for m in members} & (self.asked_bases(product["label"]) | {_base(r["text"].split(":")[0]) for r in product["reqs"]}):
                return False
            if item["label"].lower() in self.asked_labels(product["label"]):
                return False
            return not any(self.settled(product, m, item["member_values"].get(m)) for m in members)

        while self.interview:
            item = self.interview.pop(0)
            if not open_item(item):
                continue
            product = self.product(item["product"])
            members = list(item["keys"])
            return {"product": product["label"], "properties": [item["label"]], "key": item["keys"][0], "keys": members,
                    "why": "planner: " + item.get("why", "")}
        return None

    def published_labels(self) -> list[str]:
        # Category strings the store itself publishes: a search parameter's allowed values, then labels on result rows.
        labels: list[str] = []
        search = self.calls.get("search")
        props = (self.schemas.get(search["tool"]) or {}).get("properties") or {} if search else {}
        for key, spec in props.items():
            if not isinstance(spec, dict) or not _CATEGORY_KEY.search(key):
                continue
            for value in spec.get("enum") or []:
                if isinstance(value, str) and value.strip():
                    labels.append(value.strip())
        for listing in self.listings.values():
            if listing.category:
                labels.append(listing.category)
        return list(dict.fromkeys(labels))

    def adopt_category(self) -> None:
        # Replace the shopper's phrase with the published label that names the product. A narrower neighbour loses.
        labels = self.published_labels()
        if not labels:
            return
        request = self.request.lower()

        def kind_words(text: str) -> set[str]:
            words: set[str] = set()
            for word in re.findall(r"[a-z]+", text.lower()):
                if len(word) < 3:
                    continue
                words.add(word)
                if word.endswith("es") and not word.endswith("ss") and len(word) > 4:
                    words.add(word[:-2])
                elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                    words.add(word[:-1])
            return words

        for product in self.products:
            name = kind_words(product.get("label") or "")
            quoted = [label for label in labels if label.lower() in request and (not name or name & kind_words(label))]
            if quoted:
                product["category"] = max(quoted, key=len)
                continue
            scored = [(len(name & kind_words(label)), label) for label in labels if name & kind_words(label)]
            if not scored:
                continue
            best = max(score for score, _label in scored)
            top = [label for score, label in scored if score == best]
            full = [label for label in top if not any(kind_words(label) < kind_words(other) for other in top)]
            product["category"] = max(full or top, key=len)

    def product_listings(self, product: dict[str, Any]) -> list[Listing]:
        # Inspected listings that belong to this wanted product (by category or title words).
        inspected = [l for l in self.listings.values() if l.inspected and l.properties()]
        wanted = _stems(f"{product.get('category') or ''} {product['label']}")
        scored = [(len(wanted & _stems(l.category)) * 2 + len(wanted & _stems(l.title)), l) for l in inspected]
        # a listing whose title and category share no words with the product is not that product
        return [l for score, l in scored if score > 0]

    def owner(self, listing: Listing | None) -> dict[str, Any] | None:
        # The wanted product a listing belongs to, by category and title words. None when the words do not overlap.
        if not self.products or listing is None:
            return None
        scored = []
        for product in self.products:
            wanted = _stems(f"{product.get('category') or ''} {product['label']}")
            scored.append((len(wanted & _stems(listing.category)) * 2 + len(wanted & _stems(listing.title)), product["label"], product))
        scored.sort(key=lambda s: -s[0])
        if scored[0][0] == 0 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][2]

    def question_label(self, key: str, category: str) -> str:
        # Plain property name to ask about: datasheet key without codes or the category's own words.
        words = _norm_key(key).split()
        kept = list(words)
        while len(kept) > 1 and len(kept[0]) <= 2:  # leading code tokens: "m_top_neckline" -> "top neckline"
            kept = kept[1:]
        return " ".join(kept or words)

    def code_plan(self, force: bool) -> dict[str, Any] | None:
        # Next question from this task's opened listings, no inference call. A story hint comes first, then the
        # property the most listings carry. With two or fewer messages left and three or more listings open, only a
        # hint or a property on at least half the listings is asked. None leaves it to the model planner.
        # Every wanted product gets one question before any product gets a second.
        options = []
        for index, product in enumerate(self.products):
            listings = self.product_listings(product)
            if not listings:
                if not force and self.inspected_count() < 2 + len(self.products) and self.step < self.max_steps // 3:
                    return {"wait": True}
                continue
            key_values: dict[str, dict[str, str]] = {}
            for listing in listings:
                for key, value in (listing.datasheet() or listing.properties()).items():
                    key_values.setdefault(key, {})[listing.id] = str(value).lower()
            category = product.get("category") or max((l.category for l in listings), key=lambda c: sum(1 for l in listings if l.category == c))
            asked = self.asked_bases(product["label"])
            stated = {_base(r["text"].split(":")[0]) for r in product["reqs"]}
            context = self.kind_words(product, listings)
            groups = _key_groups(key_values, context)
            ranked = _rank_properties_scored(" ".join(sorted(context)), key_values, len(listings))
            asked_texts = self.asked_labels(product["label"])
            queued = [p for p in self.next_labels.get(product["label"], [])
                      if p.lower() not in asked_texts and _base(p) not in asked and _base(p) not in stated]
            if queued:
                limit = 1 if self.singles or not self.list_on else 8
                picks = [(p, 1.0) for p in queued[:limit]]
                revealed = any(r["source"] == "revealed" for r in product["reqs"])
                options.append((revealed, len(asked) > 0, False, -1.0, index, product, picks, category))
                continue
            hinted = {_concept(str(h.get("property") or "")) for h in self.hints if self.product(h.get("product")) is product}
            hinted |= {_base(str(h.get("property") or "")) for h in self.hints if self.product(h.get("product")) is product}

            def bases(key: str) -> set[str]:
                return {_base(m) for m in groups[key]}

            def is_hint(key: str) -> bool:
                return any(_concept(member) in hinted or _base(member) in hinted for member in groups[key])

            open_keys: list[tuple[str, float, bool]] = []
            for key, share in ranked:
                if _not_a_need(key, list(key_values[key].values()), self.row_keys):
                    continue
                if self.question_label(_plain_key(groups[key]), category).lower() in asked_texts:
                    continue
                if bases(key) & asked or bases(key) & stated:
                    continue
                if any(self.settled(product, member, list(key_values[member].values())) for member in groups[key]):
                    continue
                open_keys.append((key, share, is_hint(key)))
            if self.messages_left() - self.reserve_messages <= _TIGHT_MESSAGES and len(listings) >= 3:
                sure = [item for item in open_keys if item[2] or item[1] >= 0.5]
                if sure:
                    open_keys = sure
            open_keys.sort(key=lambda item: (not item[2], -item[1]))
            limit = 1 if self.singles or not self.list_on else 8
            picks = [(key, share) for key, share, _hinted in open_keys[:limit]]
            if picks:
                revealed = any(r["source"] == "revealed" for r in product["reqs"])
                options.append((revealed, len(asked) > 0, not open_keys[0][2], -picks[0][1], index, product, picks, category))
        if not options:
            return None
        # no reveal yet, then not yet asked, then a story hint, then the most widely carried property
        _revealed, _, _, _, _, product, picks, category = min(options, key=lambda o: o[:5])
        keys = [key for key, _ in picks]
        groups = _key_groups([k for l in self.product_listings(product) for k in (l.datasheet() or l.properties())] + keys,
                             self.kind_words(product, self.product_listings(product)))
        members = [m for key in keys for m in groups.get(key, [key])]
        return {"product": product["label"], "properties": [self.question_label(_plain_key(groups.get(key, [key])), category) for key in keys],
                "key": keys[0], "keys": members,
                "why": "coverage: " + ", ".join(f"{key} (on {share:.0%})" for key, share in picks)}

    def settled(self, product: dict[str, Any], key: str, values: list[str] | None) -> bool:
        # the request or a reveal already settles this property: its name holds a requirement's property name (or the
        # reverse: "packaging type" for "packaging: jar"), or its values include a required value
        words = _words(key)
        # the request's own description of the product ("lollipops in a jar") names a value of the property
        named = _words(f"{product.get('category') or ''} {product.get('label') or ''}")
        if values and any(len(_norm_key(v)) > 2 and _words(v) and _words(v) <= named for v in values):
            return True
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"]:
                continue
            head, value = req["text"].split(":", 1)
            head_words = _words(head)
            if head_words and words and (head_words <= words or words <= head_words):
                return True
            wanted = _norm_key(re.sub(r"^\s*(NOT|at least|at most|one of)\s+", "", value.strip(), flags=re.I))
            if wanted and values and any(_norm_key(v) == wanted for v in values):
                return True
        return False

    def asked_bases(self, label: str) -> set[str]:
        # Kinds of property already asked for a product; after "one at a time" only the first counts (it is re-asked alone).
        out = set()
        for q in self.questions:
            if q["product"] != label:
                continue
            keys = q.get("keys") or [q.get("key") or p for p in q["props"]]
            # singles: every key is a spelling of the one property asked
            out.update(_base(k) for k in (keys[:1] if q.get("type") == "one_at_a_time" and len(q["props"]) > 1 else keys))
        return out

    def hint_plan(self, label: str | None = None, taken: list[str] = ()) -> dict[str, Any] | None:
        # A story detail that points at a property, when no question has asked it yet.
        # With `label`, only a hint for that product other than the `taken` keys.
        for hint in self.hints:
            product = self.product(hint.get("product"))
            if label is not None and product["label"] != label:
                continue
            prop = str(hint.get("property") or "").strip()
            base = _base(prop)
            if not base or _not_a_need(prop, None, self.row_keys):
                continue
            asked = self.asked_bases(product["label"]) | {_base(k) for k in taken}
            stated = {_base(r["text"].split(":")[0]) for r in product["reqs"]}
            if base in asked or base in stated or self.settled(product, prop, None):
                continue
            # ask by the name the product's listings use for it, else by the plain name
            keys = {k for l in self.product_listings(product) for k in (l.datasheet() or l.properties())}
            key = next((k for k in sorted(keys, key=len) if _base(k) == base), None)
            label = self.question_label(key, product.get("category") or "") if key else _norm_key(prop)
            self.dialogue.append({"kind": "hint_question", "product": product["label"], "property": label, "detail": str(hint.get("detail") or "")[:120]})
            return {"product": product["label"], "properties": [label], "key": key or prop, "why": f"story hint: {hint.get('detail')}"}
        return None

    def plan_question(self, label: str | None = None) -> dict[str, Any] | None:
        # With `label`, the list for that one product (the first question's list decides the task, so it is allowed up
        # to the hard spend cap and does not wait for records).
        if self.over_budget() if label is None else self.spend + 0.02 > SPEND_HARD:
            return None
        asked = "\n".join(f"- \"{q['text']}\" -> {q.get('type')}: {q.get('reply', '')[:200]}" for q in self.questions) or "(none yet)"
        content = (
            f"REQUEST:\n{self.request}\n\n{self.ledger_text()}\n\n"
            f"PROPERTIES IN THE INSPECTED LISTINGS (value (count)):\n{self.property_table()}\n\n"
            f"QUESTIONS ALREADY ASKED:\n{asked}\n\nPER PRODUCT:\n{self.coverage_text()}\n\n"
            f"Messages left for questions: {self.messages_left() - self.reserve_messages}"
        )
        if label is not None:
            content += (f"\n\nChoose for the product {label!r} only: up to eight property names, likeliest first, even when no "
                        "record of it was inspected (then the properties buyers of that kind of product choose by). Do not return wait.")
        plan = self.ask_json(PLAN_PROMPT, content, HEAVY_MODELS, 400)
        if isinstance(plan, dict) and isinstance(plan.get("properties"), list):
            # drop a name that is not a shopper's need, already asked, or already settled
            product = self.product(plan.get("product"))
            asked = self.asked_bases(product["label"])
            stated = {_base(r["text"].split(":")[0]) for r in product["reqs"]}
            keep = [p for p in plan["properties"] if str(p).strip() and not _not_a_need(str(p), None, self.row_keys)
                    and _base(str(p)) not in asked and _base(str(p)) not in stated and not self.settled(product, str(p), None)]
            if not keep:
                return None
            plan["properties"] = keep
        return plan

    def read_reply(self, props: list[str], lines: list[str], question: str) -> dict[str, Any]:
        # Every reply to a question is read by the model (KYO reads every answer): fixed wordings missed most of the
        # ways race shoppers ask for one question at a time. Shoppers repeat canned lines, so a reply word for word
        # like one already read in this task reuses that reading when it stated nothing.
        key = re.sub(r"\s+", " ", " ".join(lines)).strip().lower()
        known = self.reply_kinds.get(key)
        if known:
            return {"reply_type": known, "summary": f"{known.replace('_', ' ')} on {' and '.join(props)}"}
        result = self.interpret(lines, question)
        kind = str(result.get("reply_type") or "")
        stated = result.get("add") or result.get("drop") or _number(result.get("budget")) or result.get("stock_out") \
            or result.get("price_changed") or result.get("instructions")
        if key and kind in ("no_preference", "not_understood", "one_at_a_time") and not stated:
            self.reply_kinds[key] = kind
        return result

    def answered_index(self, props: list[str], added: Any) -> int:
        # Which named property the reply answered. The shopper answers the earliest one it has a need for.
        reqs: list[str] = []
        if isinstance(added, list):
            for item in added:
                if isinstance(item, dict):
                    reqs.append(str(item.get("requirement") or ""))
                else:
                    reqs.append(str(item))
        reqs = [req for req in reqs if req]
        if not reqs:
            return -1
        hits = []
        for req in reqs:
            prop = req.split(":", 1)[0]
            scores = [len(_words(label) & _words(prop)) for label in props]
            if not scores or max(scores) == 0:
                scores = [len(_words(label) & _words(req)) for label in props]
            if scores and max(scores) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def settle_list(self, label: str, props: list[str], kind: str, added: Any) -> None:
        # A stated need keeps every property up to the one answered, and the rest lead the next message.
        # A reply that states nothing keeps only the first, and the rest come next in reverse so a don't-mind
        # cannot hide them. One-at-a-time, or a second empty reply, ends lists.
        if len(props) <= 1:
            return
        hit = self.answered_index(props, added)
        if kind == "one_at_a_time" or (kind == "not_understood" and hit < 0):
            self.list_on = False
            self.singles = True
            self.questions[-1]["props"] = []
            self.questions[-1]["keys"] = []
            self.questions[-1]["type"] = "one_at_a_time"
            self.next_labels[label] = [props[0]]
            return
        if hit >= 0:
            self.list_misses[label] = 0
            done = props[: hit + 1]
            ahead = props[hit + 1 :]
        else:
            self.list_misses[label] = self.list_misses.get(label, 0) + 1
            done = props[:1]
            ahead = list(reversed(props[1:]))
            if self.list_misses[label] >= 2:
                self.list_on = False
                ahead = props[1:]
        self.questions[-1]["props"] = done
        self.questions[-1]["keys"] = done
        self.next_labels[label] = ahead

    def send_question(self, label: str, props: list[str], text: str) -> None:
        filled = self.fill("message", text=text)
        if not filled:
            self.ask_done = True
            return
        name, args = filled
        data = self.env_call(name, args)
        if data is None:
            self.ask_done = True
            return
        observation = data.get("observation")
        reply_lines = self.shopper_lines(name, args, observation)
        reply = " ".join(reply_lines) or _compact(observation)[:300]
        if self.message_refused(observation):
            self.messages_used = self.message_budget or self.messages_used + 1
            self.ask_done = True
            self.inject(name, args, _compact(_brief(observation)))
            return
        self.messages_used += 1
        result = self.read_reply(props, reply_lines or [reply], text)
        kind = str(result.get("reply_type") or "other")
        self.questions.append({"product": label, "props": props, "text": text, "reply": reply, "type": kind,
                               "key": self.__dict__.pop("pending_key", None), "keys": self.__dict__.pop("pending_keys", None)})
        if kind == "one_at_a_time":
            self.singles = True
            self.list_on = False
        self.settle_list(label, props, kind, result.get("add"))
        if kind in ("one_at_a_time", "not_understood") and not result.get("add") and len(props) <= 1:
            self.repeated.add((label, props[0]))
        self.inject(name, args, _compact(_brief(observation)) + (f"\n[interpreted: {result.get('summary')}]" if result.get("summary") else ""))

    def questions_pending(self) -> bool:
        return (
            not self.ask_done
            and "message" in self.calls
            and self.messages_left() > self.reserve_messages
            and self.steps_left() > 8
        )

    def message_refused(self, observation: Any) -> bool:
        text = _compact(observation).lower()
        return bool(re.search(r"budget|limit|no more messages|exceed|quota|allowance|not allowed", text)) and len(text) < 400 and not _VOICE.search(_compact(observation))

    # ------------------------------------------------------------------ controller

    def inject(self, name: str, args: dict[str, Any], result: str) -> None:
        self.call_seq += 1
        call_id = f"call_{self.call_seq}"
        self.history.append({"role": "assistant", "content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": _compact(args)}}]})
        self.history.append({"role": "tool", "tool_call_id": call_id, "content": result})

    def note_tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.note_tool,
                "description": "Free (uses no store step). Record a resolved requirement value or a fact you need later. To replace a requirement that pointed to another listing with its concrete value, give product, requirement and replaces.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string"},
                        "product": {"type": "string"},
                        "requirement": {"type": "string", "description": "a concrete requirement, 'property: value'"},
                        "replaces": {"type": "string", "description": "the text of the requirement this one replaces"},
                    },
                    "required": ["note"],
                },
            },
        }

    def tools_for_turn(self) -> list[dict[str, Any]]:
        finish_roles = {"order", "decline", "cart_add", "cart_remove", "cart_view"}
        keep: set[str] = set(self.schemas)
        if self.steps_left() <= len(self.products) + 1:
            keep = {self.calls[r]["tool"] for r in finish_roles if r in self.calls}
        if "message" in self.calls:
            keep.discard(self.calls["message"]["tool"])  # the system writes every message to the shopper
        tools = [t for t in self.tools if t["function"]["name"] in keep] or self.tools
        if self.note_calls < 8:
            tools = tools + [self.note_tool_spec()]
        return tools

    def controller_messages(self) -> list[dict[str, Any]]:
        intro = f"SHOPPER'S REQUEST:\n{self.request}"
        if self.opening is not None:
            intro += "\n\nOPENING OBSERVATION:\n" + _compact(_brief(self.opening, 2000))
        intro += "\n\nHOW TO CALL THE KEY TOOLS: " + _compact(self.calls)
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": intro}]
        tool_indexes = [i for i, m in enumerate(self.history) if m["role"] == "tool"]
        old = set(tool_indexes[:-RECENT_RESULTS])
        self.cache_at = 0
        for index, message in enumerate(self.history):
            if index in old and len(message.get("content") or "") > 500:
                message = dict(message)
                message["content"] = message["content"][:500] + " ... (shortened; the STATE keeps what matters)"
            if index in old and message["role"] == "tool":
                self.cache_at = len(messages)  # old results never change again: the prefix up to here is cacheable
            messages.append(message)
        messages.append({"role": "user", "content": self.state_text() + "\n\nDecide the next step."})
        return messages

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.dialogue
        self.setup()
        idle = 0
        turn_cap = MAX_TURNS
        turn = 0
        while turn < turn_cap:
            turn += 1
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if time.monotonic() - self.started > 270:
                self.force_finish("time")
                break
            if self.llm_failures >= 12 or self.llm_dead or self.spend >= SPEND_STOP:
                self.force_finish("inference budget" if self.llm_dead or self.spend >= SPEND_STOP else "inference failures")
                break
            cart = self.current_cart()
            owed = int(bool(cart) and self.basket_check and self.basket_sent_for != cart and self.messages_left() > 0 and "message" in self.calls)
            if self.order_refusals and not self.second_walk:
                self.second_walk = True
                self.walk_left = 10
                self.history.append({"role": "user", "content": "The store refused the finish. Apply what it said, fix the cart, and check out again."})
            if self.second_walk and self.walk_left > 0 and self.steps_left() > 1:
                self.walk_left -= 1
            elif self.steps_left() <= ((1 if cart else 2) + owed):
                # only enough steps left to add (tell the shopper the basket) and order: finish now rather than run out
                self.force_finish("steps")
                break
            if self.listings and self.steps_left() >= 18:
                # Cheapest tasks lose the matching row once the step budget is spent on other searches. Look once, early.
                for product in self.products:
                    if (product.get("cheapest") or self.pick == "cheapest") and product["label"] not in self.price_hunts:
                        self._price_hunt(product)
                        break
            if not self.checkout_tried and self.order_refusals == 0 and self.max_steps >= 20 and self.query_keys and self.steps_left() <= CHECKOUT_RESERVE:
                self.checkout_tried = True
                note = self.early_checkout()
                if note:
                    self.history.append({"role": "user", "content": note})
                if self.finished:
                    break
            self.probe_details()
            for note in (self.catch_up(), self.recheck_stock()):
                if note:
                    self.history.append({"role": "user", "content": note})
            chain = self.follow_chains()
            if chain:
                self.history.append({"role": "user", "content": chain})
            self.maybe_ask()
            if self.finished or self.steps_left() <= 0:
                break
            prompt = self.controller_messages()
            message = self.infer(prompt, MAIN_MODELS, 700, self.tools_for_turn(), cache_at=self.cache_at)
            if message is None:
                if self.llm_dead or self.spend >= SPEND_STOP:
                    self.force_finish("inference budget")
                    break
                continue
            calls = message.get("tool_calls") or []
            content = message.get("content") or ""
            if not calls and content:
                calls = self.calls_in_text(content)
                if calls:
                    content = ""
            if not calls:
                idle += 1
                self.history.append({"role": "assistant", "content": content or "(no tool call)"})
                self.history.append({"role": "user", "content": "Reply with exactly one tool call."})
                if idle >= 3:
                    self.force_finish("no tool calls")
                    break
                continue
            idle = 0
            batch = []
            for call in calls[:MAX_CALLS_PER_TURN]:
                function = call.get("function") or {}
                raw = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except (ValueError, TypeError):
                    args = None
                batch.append((str(call.get("id") or f"call_x{self.call_seq}"), str(function.get("name") or ""), args))
                self.call_seq += 1
            self.history.append({"role": "assistant", "content": content or None, "tool_calls": [
                {"id": cid, "type": "function", "function": {"name": n, "arguments": _compact(a if isinstance(a, dict) else {})}} for cid, n, a in batch]})
            for call_id, name, args in batch:
                if not isinstance(args, dict):
                    result = "NOT SENT (no step used): the arguments were not a JSON object."
                elif self.finished or self.dead:
                    result = "NOT SENT: the episode has finished."
                elif len(batch) > 1 and self.role_of(name, args) in ("order", "decline"):
                    result = "NOT SENT (no step used): an order or decline must be the only call of its reply; read the other results first."
                else:
                    result = self.handle(name, args)
                self.history.append({"role": "tool", "tool_call_id": call_id, "content": result})
            if not self.finished and self.order_refusals and self.second_walk and turn + 1 >= turn_cap and self.steps_left() > 1:
                turn_cap = turn + 10
        if not self.finished and not self.dead and self.steps_left() > 0:
            self.force_finish("loop ended")
        return self.dialogue

    def calls_in_text(self, text: str) -> list[dict[str, Any]]:
        # tool calls a model wrote as text instead of real tool calls: a JSON object, or an <invoke> block
        found = self._json_calls(text)
        return found or self._xml_calls(text)

    def _call_from_text(self, name: str, args: dict[str, Any], index: int) -> dict[str, Any]:
        return {"id": f"call_t{self.call_seq}_{index}", "type": "function", "function": {"name": name, "arguments": _compact(args)}}

    def _json_calls(self, text: str) -> list[dict[str, Any]]:
        decoder = json.JSONDecoder()
        found, index = [], 0
        while len(found) < MAX_CALLS_PER_TURN:
            index = text.find("{", index)
            if index < 0:
                break
            try:
                item, end = decoder.raw_decode(text, index)
            except ValueError:
                index += 1
                continue
            index = end
            if isinstance(item, dict) and (item.get("name") in self.schemas or item.get("name") == self.note_tool) and isinstance(item.get("arguments", {}), dict):
                found.append(self._call_from_text(item["name"], item.get("arguments") or {}, len(found)))
        return found

    def _xml_calls(self, text: str) -> list[dict[str, Any]]:
        found = []
        for match in re.finditer(r"<invoke\b[^>]*\bname=[\"']([^\"']+)[\"'][^>]*>(.*?)</invoke>", text, re.I | re.S):
            name = match.group(1).strip()
            if name not in self.schemas and name != self.note_tool:
                continue
            args: dict[str, Any] = {}
            for param in re.finditer(r"<parameter\b[^>]*\bname=[\"']([^\"']+)[\"'][^>]*>(.*?)</parameter>", match.group(2), re.I | re.S):
                raw = param.group(2).strip()
                try:
                    args[param.group(1)] = json.loads(raw)
                except ValueError:
                    args[param.group(1)] = raw
            found.append(self._call_from_text(name, args, len(found)))
            if len(found) >= MAX_CALLS_PER_TURN:
                break
        return found

    def handle(self, name: str, args: dict[str, Any]) -> str:
        if name == self.note_tool:
            return self.handle_note(args)
        if name not in self.schemas:
            return f"NOT SENT (no step used): there is no tool named {name}."
        args, missing = self.coerce_args(name, args)
        if missing:
            return f"NOT SENT (no step used): {name} needs the argument(s) {', '.join(missing)}; see its parameters."
        role = self.role_of(name, args)
        if role == "message":
            # race 11: the controller's own questions revealed 11 needs in 173 messages, reworded non-answers and broke the cap
            return ("NOT SENT (no step used): the system writes every message to the shopper (one plain question per property, and "
                    "any summary a rule asks for). Carry on with the listings.")
        if role in ("order", "decline") and self.finish_refused_at == self.step and self.finish_refusal:
            return ("NOT SENT (no step used): the store refused the last finish and said: " + self.finish_refusal[:400]
                    + " Act on it first (a new budget: check the cart total against it, or search again within it; a gone item: "
                    "replace it; a dropped requirement: re-check the choice), then finish again.")
        if role == "message":
            if self.messages_left() <= 0:
                return "NOT SENT (no step used): no messages are left."
            if self.basket_check and self.messages_left() <= self.reserve_messages:
                # races 6-7: an order with no message after the last cart change passed under 1%; the system sends that one
                return "NOT SENT (no step used): the last message is kept for the basket summary the system sends before the order."
        if role == "cart_add":
            held = self.gate_hold("choose")
            if held:
                return held
            mismatch = self.fit_block(args)
            if mismatch:
                return mismatch
        if role in ("cart_add", "order") and self.questions_pending():
            if self.inspected_count() < 6 and self.ask_pushbacks == 0 and self.steps_left() > 14:
                self.ask_pushbacks += 1
                return (
                    "NOT SENT (no step used): before choosing, inspect more of the plausible listings (at least six in all); "
                    "the system then asks the shopper about hidden needs."
                )
            self.ask_now = True
            return "NOT SENT (no step used): the system asks the shopper about hidden needs first; then check your choice against any new requirement and call again."
        if role == "listing" and self.calls.get("details"):
            # race 7: the page alone carries the good-to-know notes (a property taken from another listing, store pairing
            # rules), and a skipped page on a reference chain failed as process_requirement_missed (0/51 passed)
            known = self.listings.get(str(next(iter(args.values()), "")))
            if known is not None and known.paged:
                return "NOT SENT (no step used): this listing's page was already opened; the STATE shows what it says."
            if self.page_info is False and known is not None and known.price is not None and known.in_stock is not None:
                return ("NOT SENT (no step used): in this store a listing's page shows nothing beyond its search row, which the STATE "
                        f"already holds: {_compact(_brief(known.fields, 600))}. Use {self.calls['details']['tool']} for its property record.")
        param = self.page_param(name) if role in ("search", "browse", None) else None
        if param and args.get(param) not in (None, "", 0, 1):
            key = name + _compact({k: v for k, v in args.items() if k != param})
            if key + _compact(args[param]) in self.pages_fetched:
                return "NOT SENT (no step used): that page of these results was already fetched; its rows are in the STATE."
        if role == "order" or self.order_mode(name, args):
            if not self.query_keys and self.inspected_count() == 0:
                return "NOT SENT (no step used): search the store before ordering."
            if not self.current_cart():
                return "NOT SENT (no step used): the cart has no lines. Add each product, then order. An order with no lines does not finish the task."
        if role == "order":
            return self.handle_order(name, args)
        if role == "decline":
            return self.handle_decline(name, args)
        return self.send(name, args)

    def coerce_args(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        # KISAME's ArgCheck, light: fit argument types to the tool's schema; list required arguments that are missing.
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out: dict[str, Any] = {}
        for key, value in args.items():
            spec = props.get(key)
            if not isinstance(spec, dict):
                if schema.get("additionalProperties") is not False:
                    out[key] = value
                continue
            kind = spec.get("type")
            if kind == "integer" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                value = int(float(value))
            elif kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
            elif kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                value = float(value)
            elif kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
            elif kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            enum = spec.get("enum")
            if enum and isinstance(value, str) and value not in enum:
                value = next((e for e in enum if str(e).lower() == value.strip().lower()), value)
            out[key] = value
        missing = [k for k in schema.get("required") or [] if k not in out]
        return out, missing

    def send(self, name: str, args: dict[str, Any], question: str | None = None) -> str:
        data = self.env_call(name, args)
        if data is None:
            if self.finished:
                return "The episode has finished."
            return "The store call failed (no observation). Try again or choose another step."
        observation = data.get("observation")
        role = self.role_of(name, args)
        self.absorb(name, args, observation)
        if role in ("details", "listing"):
            for listing in list(self.listings.values()):
                self.read_notes(listing)
        notes = ""
        lines = self.shopper_lines(name, args, observation)
        if role == "message":
            self.messages_used += 1
            if self.message_refused(observation):
                self.messages_used = max(self.messages_used, self.message_budget or self.messages_used)
            text = str(next((v for v in args.values() if isinstance(v, str)), ""))
            result = self.interpret(lines or [_compact(observation)[:300]], text)
            self.questions.append({"product": "", "props": [], "text": text[:160], "reply": " ".join(lines)[:300], "type": result.get("reply_type")})
            notes = f"\n[interpreted: {result.get('summary')}]" if result.get("summary") else ""
        elif lines:
            result = self.interpret(lines, None)
            if result.get("summary"):
                notes = f"\n[shopper: {result.get('summary')}]"
        limit = 1500 if role in ("details", "listing") else 220
        text = _compact(_brief(observation, limit, 40))
        if role not in ("details", "listing", "message", "cart_add", "cart_remove", "cart_view", "order", "decline") \
                and not data.get("finished") and any(True for _ in self._listing_dicts(observation)):
            text += self.more_pages(name, args, observation)
        if len(text) > 9000:
            text = text[:9000] + " ... (cut; see STATE)"
        if data.get("finished"):
            text += "\n[the episode has finished]"
        return text + notes

    def handle_note(self, args: dict[str, Any]) -> str:
        self.note_calls += 1
        note = str(args.get("note") or "").strip()
        if note:
            self.notes.append(note[:300])
        requirement = str(args.get("requirement") or "").strip()
        if requirement:
            product = self.product(args.get("product"))
            replaces = str(args.get("replaces") or "").strip().lower()
            for req in product["reqs"]:
                if replaces and req["active"] and (replaces[:40] in req["text"].lower() or req["text"].lower()[:40] in replaces):
                    req["active"] = False
                    req["resolved"] = True
                    req["text"] += f"  -> resolved as '{requirement}'"
            product["reqs"].append({"text": requirement, "source": "resolved", "active": True})
            self.todo = [t for t in self.todo if not (replaces and replaces[:30] in t.lower())]
        return "Noted (no step used)."

    # ------------------------------------------------------------------ finishing

    def records_text(self, idents: list[str]) -> str:
        blocks = []
        for ident in idents:
            listing = self.listings.get(ident)
            if listing is None:
                blocks.append(f"{ident}: no record seen")
                continue
            # search rows carry price, stock and category that a property record may lack: show both
            shown: dict[str, Any] = {"listing": _brief(listing.fields, 600, 40)}
            if listing.record is not None:
                shown["record"] = _brief(listing.record, 1500, 40)
            blocks.append(f"{ident} (inspected: {listing.inspected}): {_compact(shown)}")
        return "\n".join(blocks)

    def current_cart(self) -> dict[str, int]:
        if self.cart is not None and self.cart_view_step >= self.cart_edit_step:
            return dict(self.cart)
        return dict(self.added)

    def view_cart(self) -> str:
        filled = self.fill("cart_view")
        if not filled:
            return ""
        name, args = filled
        data = self.env_call(name, args)
        if data is None:
            return ""
        observation = data.get("observation")
        self.absorb(name, args, observation)
        lines = self.shopper_lines(name, args, observation)
        if lines:
            self.interpret(lines, None)
        return _compact(_brief(observation, 300, 30))

    def evidence(self, idents: list[str]) -> str:
        # Code-side evidence for each requirement: matching record keys and whether the value text appears.
        lines = []
        # each cart line is read against its own product's requirements only (a multi-product basket fails on one wrong line)
        owners = {ident: self.owner(self.listings.get(ident)) for ident in idents}
        for product in self.products:
            for req in product["reqs"]:
                if not req["active"] or ":" not in req["text"]:
                    continue
                prop, _, value = req["text"].partition(":")
                prop_words = _words(prop) - {"type", "the", "of"}
                value = value.strip().lower()
                form, wanted = _requirement_form(value)
                for ident in idents:
                    listing = self.listings.get(ident)
                    if listing is None or (owners[ident] is not None and owners[ident] is not product):
                        continue
                    props = listing.properties()
                    exact = _norm_key(prop).strip()
                    keys = [f"EXACT {k}={v}" for k, v in props.items() if k == exact]
                    keys += [f"{k}={v}" for k, v in props.items() if k != exact and prop_words and prop_words & _words(k)]
                    text = (_compact(listing.record) + " " + _compact(listing.fields)).lower()
                    related = [v.lower() for k, v in props.items() if k == exact or (prop_words and prop_words & _words(k))]
                    verdict = _form_check(form, wanted, related, text)
                    lines.append(f"- {product['label']} needs '{req['text'][:80]}' | {ident}: keys {'; '.join(keys)[:200] or 'none matching'} | code check: {verdict}")
        for product in self.products:
            for ident in idents:
                listing = self.listings.get(ident)
                if listing is not None and product.get("category") and owners[ident] in (None, product):
                    lines.append(f"- {product['label']} kind '{product['category']}' | {ident}: store category '{listing.category or 'unknown'}'")
        return "\n".join(lines)

    def verify(self, cart: dict[str, int]) -> dict[str, Any] | None:
        idents = list(cart)
        others = self.cheaper_rows(cart)
        content = (
            f"REQUEST:\n{self.request}\n\n{self.ledger_text()}\n\n"
            f"CART: " + "; ".join(f"{i} x{q}" for i, q in cart.items()) + "\n\n"
            f"RECORDS OF THE CART LISTINGS:\n{self.records_text(idents)}\n\n"
        )
        evidence = self.evidence(idents)
        if evidence:
            content += "EVIDENCE FOUND BY CODE (keys that look related to each requirement; check the meaning yourself):\n" + evidence + "\n\n"
        referenced = [l for l in self.listings.values() if l.inspected and l.id not in cart]
        if referenced:
            content += "OTHER INSPECTED LISTINGS (may be referenced by notes):\n" + "\n".join(self.listing_line(l) for l in referenced[:25]) + "\n\n"
        if others:
            content += "CHEAPER LISTINGS SEEN (for the cheapest check):\n" + "\n".join(others[:30]) + "\n"
        verdict = self.ask_json(VERIFY_PROMPT, content, HEAVY_MODELS if self.spend + 0.02 <= SPEND_HARD else STRONG_MODELS, 700)
        self.dialogue.append({"kind": "verify", "cart": cart, "verdict": verdict})
        if verdict is not None and verdict.get("ok") is True:
            self.verified_cart = dict(cart)
        return verdict

    def cheaper(self, cart: dict[str, int]) -> dict[str, list[Listing]]:
        # Per cart line of a cheapest product: in-stock listings of the same product priced below it. Matched per product,
        # so a cheap row of another product never counts against a line.
        out: dict[str, list[Listing]] = {}
        for ident in cart:
            line = self.listings.get(ident)
            product = self.owner(line)
            if line is None or line.price is None or product is None or not product["cheapest"]:
                continue
            rows = [
                l for l in self.listings.values()
                if l.id not in cart and l.id not in self.refused_ids and l.price is not None and l.price < line.price
                and l.in_stock is not False and self.owner(l) is product
            ]
            if rows:
                out[ident] = sorted(rows, key=lambda l: l.price or 0)
        return out

    def cheaper_rows(self, cart: dict[str, int]) -> list[str]:
        rows = []
        for ident, found in self.cheaper(cart).items():
            product = self.owner(self.listings.get(ident))
            rows.append(f"for the {product['label'] if product else 'item'} (cart line {ident}):")
            rows += ["  " + self.listing_line(l) for l in found[:15]]
        return rows

    def cap_param(self) -> str | None:
        call = self.calls.get("search")
        if not call:
            return None
        for key, spec in (self.schemas[call["tool"]].get("properties") or {}).items():
            kind = spec.get("type") if isinstance(spec, dict) else None
            if kind in ("number", "integer") and re.search(r"cap|max|price|budget|spend|limit|upto|under|ceiling", key, re.I):
                if not re.search(r"page|offset|start|cursor|row|min", key, re.I):
                    return key
        return None

    def cheaper_probe(self, cart: dict[str, int]) -> str:
        # One search step below the cart item's price; lists cheaper listings the controller has not ruled out.
        cap = self.cap_param()
        if not cap or not any(p["cheapest"] for p in self.products) or len(self.probed) >= 4:
            return ""
        for ident in cart:
            listing = self.listings.get(ident)
            price = listing.price if listing else None
            base = self.found_by.get(ident)
            product = self.owner(listing)
            if price is None or base is None or (ident, price) in self.probed or price <= 0 or product is None or not product["cheapest"]:
                continue
            self.probed.add((ident, price))
            args = {k: v for k, v in base.items() if not re.search(r"page|offset|start|cursor|row", k, re.I)}
            kind = (self.schemas[self.calls["search"]["tool"]].get("properties") or {}).get(cap, {}).get("type")
            args[cap] = int(price) - 1 if kind == "integer" or float(price).is_integer() else round(price - 0.01, 2)
            if args[cap] < 0:
                continue
            tool = self.calls["search"]["tool"]
            data = self.env_call(tool, args)
            if data is None:
                return ""
            observation = data.get("observation")
            self.absorb(tool, args, observation)
            cheaper = []
            for record in self._listing_dicts(observation):
                other = self.listings.get(str(record[_id_field(record)]))
                if other and other.id not in cart and other.id not in self.refused_ids and other.in_stock is not False and (other.price or 0) < price \
                        and self.owner(other) is product:
                    cheaper.append(other)
            if cheaper:
                opened = 0
                for other in sorted(cheaper, key=lambda l: l.price or 0):
                    if opened >= 3 or other.inspected or self.steps_left() < 6 or "details" not in self.calls:
                        break
                    detail = self.fill("details", ident=other.id)
                    if not detail:
                        break
                    opened_data = self.env_call(*detail)
                    if opened_data is None:
                        break
                    self.absorb(detail[0], detail[1], opened_data.get("observation"))
                    opened += 1
                rows = "\n".join(self.listing_line(l) for l in sorted(cheaper, key=lambda l: l.price or 0)[:12])
                return f"The system searched below the cart price ({_compact(args)}) and opened {opened} cheaper listing(s):\n{rows}"
        return ""

    def missing_products(self, cart: dict[str, int]) -> list[dict[str, Any]]:
        owners = [self.owner(self.listings.get(ident)) for ident in cart]
        return [product for product in self.products if not any(owner is product for owner in owners)]

    def plain_needs(self, product: dict[str, Any]) -> list[tuple[str, str]]:
        # Requirements a listing can state word for word. A limit or an exclusion is not one of these.
        needs = []
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"]:
                continue
            prop, _, value = req["text"].partition(":")
            words = _words(value)
            if not words or words & {"not", "no", "least", "most", "under", "over", "min", "max", "minimum", "maximum"}:
                continue
            needs.append((prop.strip(), value.strip()))
        return needs

    def states_plain(self, listing: Listing, product: dict[str, Any]) -> bool:
        # Every stated requirement is decided by the property with that name. The title does not stand in for a missing property.
        props = listing.properties()
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"]:
                continue
            prop, _, value = req["text"].partition(":")
            form, options = _req_options(value)
            if form == "limit":
                continue
            keys = _matching_keys(props, prop)

            def hits(stated: str) -> bool:
                if form == "one_of":
                    return any(_value_hit(option, stated) for option in options)
                return _value_hit(options[0], stated)

            # A use ("for a wedding") fails only when an occasion key states some other value.
            if req.get("source") == "use":
                if keys and not any(hits(props[key]) for key in keys):
                    return False
                continue
            if form == "not":
                stated = props[keys[0]] if keys else (listing.title or "")
                if _value_hit(options[0], stated):
                    return False
                continue
            if not keys:
                if not hits(listing.title or ""):
                    return False
                continue
            # "style" or "type" alone does not veto a more specific key that states the value ("skirt style: wrap").
            # A second key with the same words ("clock shape" and "clock_shape") can state the value the first missed.
            same = _prop_stems(keys[0])
            if hits(props[keys[0]]) or (
                _content_words(keys[0]) <= _PROP_FILLER and any(hits(props[key]) for key in keys[1:])
            ) or any(hits(props[key]) for key in keys[1:] if _prop_stems(key) == same):
                continue
            return False
        return True

    def meets_limits(self, listing: Listing, product: dict[str, Any]) -> bool:
        # A stated limit is met only by a number on that property. A listing that does not state the measure is out.
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"]:
                continue
            prop, _, value = req["text"].partition(":")
            bound = _limit_bound(value)
            if bound is None:
                continue
            op, limit, unit = bound
            found = False
            ok = False
            for key, stated in listing.properties().items():
                if not (_words(key) & _words(prop)):
                    continue
                raw = str(stated)
                if unit and re.search(r"\d\s*[a-z]", raw.lower()) and not re.search(rf"(?<![a-z]){re.escape(unit)}(?![a-z])", raw.lower()):
                    continue
                numbers = [_number(piece) for piece in re.findall(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))]
                numbers = [n for n in numbers if n is not None]
                if not numbers:
                    continue
                found = True
                if all(_inside_limit(op, n, limit) for n in numbers):
                    ok = True
                    break
            if not found or not ok:
                return False
        return True

    def best_line(self, product: dict[str, Any]) -> Listing | None:
        # The cheapest in-stock inspected listing that states this product's plain requirements.
        rows = [listing for listing in self.product_listings(product)
                if listing.in_stock is not False and listing.id not in self.refused_ids and listing.price is not None
                and self.states_plain(listing, product) and self.meets_limits(listing, product)]
        if self.budget is not None:
            if self.budget_scope == "each":
                rows = [listing for listing in rows if (listing.price or 0) <= self.budget]
            else:
                rows = [listing for listing in rows if (listing.price or 0) * product["quantity"] <= self.budget]
        if not rows:
            return None
        if self.pick in ("most", "least") and self.pick_measure:
            wanted = _words(self.pick_measure)
            measured: list[tuple[float, Listing]] = []
            for listing in rows:
                for key, value in listing.properties().items():
                    words = _words(key)
                    if wanted and (wanted <= words or wanted & words):
                        number = _number(value)
                        if number is not None:
                            measured.append((number, listing))
                            break
            if not measured:
                return None
            return (max if self.pick == "most" else min)(measured, key=lambda pair: pair[0])[1]
        return min(rows, key=lambda listing: listing.price or 0)

    def cart_states(self, cart: dict[str, int]) -> bool:
        # True when every line's own property record states that product's requirements and limits.
        if not cart:
            return False
        for ident in cart:
            listing = self.listings.get(ident)
            product = self.owner(listing) if listing else None
            if listing is None or product is None:
                return False
            if not self.states_plain(listing, product) or not self.meets_limits(listing, product):
                return False
        return True

    def line_contradicts(self, listing: Listing, product: dict[str, Any], soft: bool = False) -> list[str]:
        # Positive evidence against a cart line: a property named like a requirement states another value, an excluded
        # value is stated, or a number on that property lies outside its limit. A missing property, an unopened record
        # or a listing the code cannot place is no evidence: the order then goes out and the store decides.
        # With `soft`, a listing whose property record is open but names no key for a requirement that the title does
        # not state either is also reported: the order is then held (at most twice) so the controller can find a
        # listing that states it, since in a store with full records such a line fails far more often than it passes
        # (practice: a watch with no dial glass key for "sapphire", a sleeved jacket for "sleeveless").
        #
        # Race 15: the gate demanded proof (cart_states) and, with no parsed records, declined 152 of 180 episodes;
        # the winner's gate holds only on what the records show and placed an order in 147 of 180.
        props = listing.properties()
        faults: list[str] = []
        recorded = len(listing.datasheet()) >= 3
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"]:
                continue
            prop, _, value = req["text"].partition(":")
            form, options = _req_options(value)
            keys = _matching_keys(props, prop)
            bare = _content_words(prop) <= _PROP_FILLER
            if bare:
                # "type" or "style" alone: only a key of exactly that name decides ("style=sport & street" against
                # "style: youth fashion"); "milk type=hi-cal low fat" and "fa general styles=casual" (against "style:
                # wrap") were taken as evidence on listings the grader accepted, and a missing bare key proves nothing
                keys = [k for k in keys if _content_words(k) == _content_words(prop)]
            if form == "limit":
                bound = _limit_bound(value)
                if bound is None:
                    continue
                if not keys:
                    if soft and recorded and not bare:
                        faults.append(f"{listing.id}: no property states {req['text'][:60]}")
                    continue
                op, limit, unit = bound
                numbers: list[float] = []
                for key in keys:
                    raw = str(props[key]).lower()
                    if unit and re.search(r"\d\s*[a-z]", raw) and not re.search(rf"(?<![a-z]){re.escape(unit)}(?![a-z])", raw):
                        continue
                    numbers += [n for n in (_number(piece) for piece in re.findall(r"-?\d+(?:\.\d+)?", raw.replace(",", ""))) if n is not None]
                if numbers and not any(_inside_limit(op, n, limit) for n in numbers):
                    faults.append(f"{listing.id}: wants {req['text'][:60]} but {keys[0]}={str(props[keys[0]])[:40]}")
                continue

            def hits(stated: str) -> bool:
                return any(_value_present(option, stated) for option in options)

            if form == "not":
                where, stated = (keys[0], props[keys[0]]) if keys else ("title", listing.title or "")
                if any(_value_hit(option, stated) for option in options):
                    faults.append(f"{listing.id}: excluded '{options[0]}' is stated ({where}={str(stated)[:40]})")
                continue
            if not keys:
                if soft and recorded and not bare and not hits(listing.title or ""):
                    faults.append(f"{listing.id}: no property states {req['text'][:60]}")
                continue
            if not any(hits(props[key]) for key in keys):
                faults.append(f"{listing.id}: wants {req['text'][:60]} but {keys[0]}={str(props[keys[0]])[:40]}")
        return faults

    def cart_contradictions(self, cart: dict[str, int], soft: bool = False) -> list[str]:
        faults: list[str] = []
        for ident in cart:
            listing = self.listings.get(ident)
            product = self.owner(listing) if listing else None
            if listing is not None and product is not None:
                faults += self.line_contradicts(listing, product, soft)
            elif listing is not None and listing.category and all(p.get("category") for p in self.products):
                # a store label that shares no word with any wanted product: the wrong kind (a bomber jacket for a hoodie)
                faults.append(f"{ident} is in the category '{listing.category[:40]}', not a kind the shopper asked for")
        return faults

    def _open_for(self, product: dict[str, Any]) -> None:
        # Read a few cheap rows before the pick. One search uses the stated values when nothing open states them yet.
        if "details" not in self.calls or self.steps_left() < 6 or self.choice_opens >= 6:
            return
        if self.best_line(product) is None and not self.product_listings(product):
            self.browse_category(product)
        if self.best_line(product) is None and self.choice_searches < 2 and "search" in self.calls and self.steps_left() >= 8:
            self.choice_searches += 1
            terms: list[str] = []
            for req in product["reqs"]:
                if not req["active"] or ":" not in req["text"]:
                    continue
                prop, _, value = req["text"].partition(":")
                form, options = _req_options(value)
                if form == "limit":
                    terms.append(prop.strip())
                elif form != "not":
                    terms.extend(options[:2])
            filled = self.fill("search", text=" ".join([product["label"], *terms[:3]])[:140])
            if filled:
                data = self.env_call(*filled)
                if data is not None:
                    self.absorb(filled[0], filled[1], data.get("observation"))
        wanted = _stems(f"{product.get('category') or ''} {product['label']}")
        need: set[str] = set()
        for req in product["reqs"]:
            if req["active"] and ":" in req["text"]:
                _form, options = _req_options(req["text"].split(":", 1)[1])
                for option in options:
                    need |= _stems(option)
        best = self.best_line(product)
        cap = best.price if best is not None and (product.get("cheapest") or self.pick == "cheapest") else None
        rows = []
        for listing in self.listings.values():
            if listing.inspected or listing.in_stock is False or listing.id in self.refused_ids:
                continue
            if cap is not None and (listing.price is None or listing.price >= cap):
                continue
            if wanted and not (wanted & _stems(f"{listing.title} {listing.category}")):
                continue
            rows.append(listing)
        rows.sort(key=lambda listing: (0 if need and need & _stems(listing.title) else 1, listing.price or 0))
        for listing in rows[:3]:
            if self.choice_opens >= 6 or self.steps_left() < 6:
                break
            detail = self.fill("details", ident=listing.id)
            if not detail:
                break
            data = self.env_call(*detail)
            if data is None:
                break
            self.absorb(detail[0], detail[1], data.get("observation"))
            self.choice_opens += 1

    def _filter_args(self, category: str, cap: float) -> tuple[str, dict[str, Any]] | None:
        # A category search capped at a price. Keyword search ignores the cap and then sorts by id, so the
        # cheap rows of a category only show up on the filter search, and only while the cap fits in one window.
        if "search" not in self.calls:
            return None
        tool = self.calls["search"]["tool"]
        props = self.schemas.get(tool, {}).get("properties") or {}
        args: dict[str, Any] = {}
        for key, spec in props.items():
            enum = spec.get("enum") if isinstance(spec, dict) else None
            if not enum:
                continue
            filt = next((value for value in enum if re.search(r"field|filter|facet", str(value), re.I)), None)
            if filt is not None:
                args[key] = filt
                break
        if not args:
            return None
        dept = next((key for key, spec in props.items()
                     if isinstance(spec, dict) and spec.get("type") == "string" and re.search(r"department|category", key, re.I)), None)
        cap_key = self.cap_param()
        if dept is None or cap_key is None:
            return None
        limit = (props.get(dept) or {}).get("maxLength")
        args[dept] = category[: int(limit)] if isinstance(limit, int) else category
        kind = (props.get(cap_key) or {}).get("type")
        args[cap_key] = int(cap) if kind == "integer" or float(cap).is_integer() else round(float(cap), 2)
        if args[cap_key] < 0:
            return None
        return tool, args

    def _open_ids(self, rows: list[Listing], limit: int) -> None:
        opened = 0
        for listing in rows:
            if opened >= limit or self.steps_left() < 4 or "details" not in self.calls:
                break
            detail = self.fill("details", ident=listing.id)
            if not detail:
                break
            data = self.env_call(*detail)
            if data is None:
                break
            self.absorb(detail[0], detail[1], data.get("observation"))
            opened += 1

    def _price_hunt(self, product: dict[str, Any]) -> None:
        # Cheapest: the store shows 21 ids, not the 21 best matches. The cheap band of the published category
        # is the whole window when it fits. When it does not, the kind plus the required value is a smaller window.
        label = product["label"]
        if label in self.price_hunts or self.steps_left() < 8 or not self.listings:
            return
        if not (product.get("cheapest") or self.pick == "cheapest"):
            return
        self.price_hunts.add(label)
        category = str(product.get("category") or "").strip()
        values: list[str] = []
        for req in product["reqs"]:
            if not req["active"] or ":" not in req["text"] or req.get("source") == "use":
                continue
            form, options = _req_options(req["text"].split(":", 1)[1])
            if form not in ("limit", "not") and options:
                values.append(options[0])
        focus = {word for word in _content_words(values[0]) if len(word) >= 4} if values else set()
        seen = [listing.price for listing in self.listings.values()
                if listing.price is not None and category and listing.category.lower() == category.lower()]
        cap = min(seen) if seen else self.budget
        tight = False
        held = self.best_line(product)
        if category and cap is not None and "search" in self.calls and (held is None or (held.price or 0) > cap):
            for _ in range(2):
                filled = self._filter_args(category, cap)
                if filled is None or self.steps_left() < 8:
                    break
                tool, args = filled
                data = self.env_call(tool, args)
                if data is None:
                    break
                observation = data.get("observation")
                self.absorb(tool, args, observation)
                learned = self.page_from_shape(tool, observation)
                count = learned[0] if learned else None
                if count is not None and count <= 21 and self.steps_left() >= 8:
                    self.more_pages(tool, args, observation)
                if count is not None and count <= 21:
                    tight = True
                    break
                # The next cap is the cheapest price this window actually showed. Dropping one more unit
                # skips the row sitting on that price.
                prices = [listing.price for listing in self.listings.values()
                          if listing.price is not None and listing.seen_step == self.step and listing.category.lower() == category.lower()]
                nxt = min(prices) if prices else 0
                if nxt <= 0 or nxt >= cap:
                    break
                cap = nxt
        fresh = [listing for listing in self.listings.values()
                 if tight and not listing.inspected and listing.in_stock is not False and listing.id not in self.refused_ids
                 and listing.price is not None and listing.category.lower() == category.lower() and listing.price <= cap]
        if focus:
            # A cheap window whose titles never name the value cannot hold the match. Leave it for the value search.
            named = [listing for listing in fresh if focus <= _content_words(listing.title)]
            fresh = named
        fresh.sort(key=lambda listing: listing.price or 0)
        for listing in fresh[:4]:
            if self.best_line(product) is not None or self.steps_left() < 3:
                break
            self._open_ids([listing], 1)
        # A band that does not fit in 21 ids hides a higher id that states the value. Search the kind with the
        # value and open the cheapest titles that name it, until one states the requirement.
        if self.best_line(product) is not None or not focus or "search" not in self.calls or self.steps_left() < 12:
            return
        filled = self.fill("search", text=" ".join([label, values[0]])[:140])
        if not filled:
            return
        tool, args = filled
        data = self.env_call(tool, args)
        if data is None:
            return
        started = self.step
        observation = data.get("observation")
        self.absorb(tool, args, observation)
        if self.steps_left() >= 8:
            self.more_pages(tool, args, observation)
        titled = [listing for listing in self.listings.values()
                  if listing.seen_step >= started and not listing.inspected and listing.in_stock is not False
                  and listing.id not in self.refused_ids and listing.price is not None
                  and (self.budget is None or listing.price <= self.budget)
                  and focus <= _content_words(listing.title)]
        titled.sort(key=lambda listing: listing.price or 0)
        for listing in titled[:4]:
            if self.best_line(product) is not None or self.steps_left() < 3:
                break
            self._open_ids([listing], 1)

    def apply_choice(self) -> None:
        # The cart line is a listing whose record states the requirements. Cheapest means the lowest price among those.
        for product in self.products:
            if product.get("cheapest") or self.pick == "cheapest":
                self._price_hunt(product)
            if self.best_line(product) is None or product.get("cheapest") or self.pick == "cheapest":
                self._open_for(product)
        # A line is swapped only on evidence: it contradicts a requirement, or the pick is the cheapest and a cheaper
        # opened listing states every requirement. A line the records say nothing about stays, and a line that
        # contradicts with nothing better open stays too (handle_order names the fault; a decline is not the answer).
        target = dict(self.current_cart())
        changed = False
        for product in self.products:
            pick = self.best_line(product)
            owned = [ident for ident in list(target) if self.owner(self.listings.get(ident)) is product]
            if not owned:
                continue
            bad = [ident for ident in owned if self.line_contradicts(self.listings[ident], product)]
            good = [ident for ident in owned if ident not in bad]
            proven = [ident for ident in good if self.states_plain(self.listings[ident], product) and self.meets_limits(self.listings[ident], product)]
            cheapest = bool(product.get("cheapest") or self.pick == "cheapest")
            cheaper = pick is not None and pick.id not in good and good and (pick.price or 0) < min(self.listings[i].price or 0 for i in good)
            if pick is not None and (not proven or (cheapest and cheaper)):
                want = pick.id  # a proven listing beats an unproven line (practice: swaps after a stock-out picked unproven rows)
            elif good:
                want = min(proven or good, key=lambda ident: self.listings[ident].price or 0)
            else:
                continue
            if owned == [want] and target.get(want) == product["quantity"]:
                continue
            for ident in owned:
                target.pop(ident, None)
            target[want] = product["quantity"]
            changed = True
        if changed:
            self.set_cart(target)

    def fill_missing(self) -> list[str]:
        # One line per wanted product. A cart that is short a product fails even when the line it holds is right.
        added: list[str] = []
        for product in self.missing_products(self.current_cart()):
            if self.steps_left() <= 2:
                break
            listing = self.best_line(product)
            if listing is None or listing.id in self.current_cart():
                continue
            filled = self.fill("cart_add", ident=listing.id, qty=product["quantity"])
            if not filled:
                continue
            data = self.env_call(*filled)
            if data is None:
                continue
            self.absorb(filled[0], filled[1], data.get("observation"))
            added.append(product["label"])
        return added

    def chain_blocked(self) -> bool:
        # A required property whose page chain is still unopened. Checkout waits while there are steps to open it.
        if not self.page_info or not self.calls.get("listing") or self.steps_left() < 8 or self.chain_hops >= 6:
            return False
        wanted = {_base(req["text"].split(":")[0]) for product in self.products for req in product["reqs"] if req["active"] and ":" in req["text"]}
        for listing in self.listings.values():
            for prop, source in listing.defers:
                if _base(prop) not in wanted:
                    continue
                if (listing.id, prop) not in self.followed:
                    return True
                target = self.find_listing(source, listing.id)
                if target is not None and not target.paged:
                    return True
        return False

    def gate_hold(self, stage: str) -> str:
        # A process step due at this stage. One hold per cart: the system sends a shopper message when the step needs
        # one, otherwise the controller is told the step once and the next call proceeds.
        due = []
        for step in self.process:
            if step["when"] != stage:
                continue
            # a step about the cart counts only when it was done after the cart last changed
            if stage in ("order", "end") and step.get("at", -1) <= self.cart_edit_step:
                due.append(step)
            elif not step.get("done"):
                due.append(step)
        if not due:
            return ""
        key = stage + ":" + _compact(sorted(self.current_cart().items()))
        if key in self.gate_ok:
            return ""
        step = due[0]
        if _APPROVAL.search(step["step"]) and self.basket_check:
            step["done"] = True
            step["at"] = self.step
            return self.gate_hold(stage)
        ending = {call["tool"] for role, call in self.calls.items() if role in ("order", "decline") and call.get("tool")}
        tool_lines = [f"- {spec['function']['name']}: {spec['function'].get('description', '')[:160]}"
                      for spec in self.tools if spec["function"]["name"] not in ending]
        nudged = False
        for _round in range(3):
            if self.over_budget() or self.steps_left() <= 1:
                break
            start = max(1, len(self.done_log) - 19)
            numbered = [f"{index}. step {at}: {line}" for index, (at, line) in enumerate(self.done_log, 1) if index >= start]
            proof = {index for index, (at, _line) in enumerate(self.done_log, 1) if stage not in ("order", "end") or at > self.cart_edit_step}
            content = (
                f"STAGE: {stage}\nSTEPS DUE:\n" + "\n".join(f"- {item['step']}" for item in due)
                + f"\n\nCART LAST CHANGED AT STEP {self.cart_edit_step}\nRECORD:\n" + ("\n".join(numbered) or "none")
                + "\n\nTOOLS:\n" + "\n".join(tool_lines)
            )
            if nudged:
                content += "\n\nNo cited RECORD line after the cart changed shows the step. Name the one store call that does it."
            written = self.ask_json(GATE_CALL_PROMPT, content, HEAVY_MODELS if self.spend < SPEND_SOFT else STRONG_MODELS, 400) or {}
            claimed = [item for item in written.get("steps") or [] if isinstance(item, dict)]
            if not claimed and written.get("done") is True:
                claimed = [{"step": due[0]["step"], "done": True, "lines": written.get("lines") or []}]

            def cited(entry: dict[str, Any]) -> bool:
                if entry.get("done") is not True:
                    return False
                numbers = entry.get("lines") if isinstance(entry.get("lines"), list) else []
                return any((number := _int(item)) is not None and number in proof for item in numbers)

            if due and all(any(cited(entry) and item["step"] in str(entry.get("step") or "") for entry in claimed) for item in due):
                for item in due:
                    item["done"] = True
                    item["at"] = self.step
                self.dialogue.append({"kind": "gate", "stage": stage, "step": step["step"]})
                return f"The system did a required step ({step['step']}). Continue, then finish."
            call = written.get("call") if isinstance(written.get("call"), dict) else {}
            name = str(call.get("tool") or "")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if name not in self.schemas or name in ending:
                if not nudged:
                    nudged = True
                    continue
                break
            required = (self.schemas.get(name) or {}).get("required") or []
            role = self.role_of(name, args)
            filled = None
            if role == "message":
                text = next((str(value).strip() for value in args.values() if isinstance(value, str) and str(value).strip()), "")
                if text and self.messages_left() > 0:
                    filled = self.fill("message", text=text)
            elif all(key in args for key in required):
                filled = (name, args)
            if not filled:
                if not nudged:
                    nudged = True
                    continue
                break
            data = self.env_call(*filled)
            if data is None:
                break
            if role == "message":
                self.messages_used += 1
                reply = self.shopper_lines(filled[0], filled[1], data.get("observation"))
                if reply:
                    self.interpret(reply, text)
            self.absorb(filled[0], filled[1], data.get("observation"))
            self.inject(filled[0], filled[1], _compact(_brief(data.get("observation"))))
        return "NOT SENT (no step used): before you " + stage + ", the request requires: " + "; ".join(item["step"] for item in due) + ". The record does not show it yet. Do it, then call again."

    def swap_candidates(self, product: dict[str, Any], cart: dict[str, int], gone: set[str]) -> list[Listing]:
        # Quantity and the code match run before a sold-out line is replaced. The code is the other product still in the cart.
        rows = [listing for listing in self.product_listings(product)
                if listing.id not in self.refused_ids and listing.id not in cart and listing.in_stock is not False]
        qty = product["quantity"]
        if self.budget is not None and self.budget_scope == "each":
            rows = [listing for listing in rows if listing.price is None or listing.price <= self.budget]
        elif self.budget is not None:
            others = sum((self.listings[ident].price or 0) * held for ident, held in cart.items()
                         if ident not in gone and ident in self.listings and self.listings[ident].price is not None)
            rows = [listing for listing in rows if listing.price is None or others + listing.price * qty <= self.budget]
        index = self.fit_index()
        for part_label, device_label in self.fits:
            part, device = self.product(part_label), self.product(device_label)
            if product is part:
                codes: set[str] = set()
                for ident in cart:
                    if ident in gone or self.owner(self.listings.get(ident)) is not device:
                        continue
                    codes |= index["models"].get(ident, set())
                if codes:
                    rows = [listing for listing in rows if index["works"].get(listing.id, set()) & codes]
            elif product is device:
                needed: set[str] = set()
                for ident in cart:
                    if ident in gone or self.owner(self.listings.get(ident)) is not part:
                        continue
                    needed |= index["works"].get(ident, set())
                if needed:
                    rows = [listing for listing in rows if index["models"].get(listing.id, set()) & needed]
        return rows

    def swap_line(self, cart: dict[str, int]) -> str:
        # A sold-out cart line is replaced from the candidates that already passed the quantity and the code match.
        gone = [ident for ident in cart if ident in self.refused_ids or (self.listings.get(ident) is not None and self.listings[ident].in_stock is False)]
        if not gone or self.steps_left() < 3:
            return ""
        for ident in gone:
            self.mark_gone(ident)
        product = self.owner(self.listings.get(gone[0])) or (self.products[0] if len(self.products) == 1 else None)
        if product is None:
            return ""
        # The replacement is the cheapest open listing that states the requirements. The same title is often another id.
        stating = self.best_line(product)
        choice = stating.id if stating is not None and stating.id not in cart else ""
        rows = [] if choice else self.swap_candidates(product, cart, set(gone))
        if not choice and rows:
            stating_rows = [listing for listing in rows if self.states_plain(listing, product) and self.meets_limits(listing, product)]
            if stating_rows:
                rows = stating_rows
        if not choice and rows and not self.over_budget():
            shown = "\n".join(self.listing_line(listing) for listing in rows[:8])
            content = f"PRODUCT: {product['label']}\nREQUIREMENTS: " + "; ".join(req["text"] for req in product["reqs"] if req["active"]) + f"\n\nLISTINGS:\n{shown}"
            picked = self.ask_json(SWAP_PROMPT, content, HEAVY_MODELS if self.spend < SPEND_SOFT else STRONG_MODELS, 80) or {}
            ident = str(picked.get("id") or "")
            if any(listing.id == ident for listing in rows):
                choice = ident
        if not choice and rows:
            priced = [listing for listing in rows if listing.price is not None]
            choice = min(priced or rows, key=lambda listing: listing.price if listing.price is not None else 0).id
        if not choice:
            return ""
        kept = {ident: qty for ident, qty in cart.items() if ident not in self.refused_ids}
        kept[choice] = product["quantity"]
        self.set_cart(kept)
        self.dialogue.append({"kind": "swap", "gone": gone[0], "replacement": choice})
        return f"The sold-out line was replaced with {choice} at quantity {product['quantity']}. Order that cart."

    def fix_counts(self) -> list[str]:
        # A line held at the wrong quantity is set to the quantity the request asked for.
        cart = self.current_cart()
        fixed: list[str] = []
        for ident, qty in list(cart.items()):
            listing = self.listings.get(ident)
            product = self.owner(listing)
            if product is None or qty == product["quantity"] or listing is None or listing.in_stock is False:
                continue
            lines = [i for i in cart if self.owner(self.listings.get(i)) is product]
            if len(lines) != 1 or self.steps_left() <= 2:
                continue
            target = dict(self.current_cart())
            target[ident] = product["quantity"]
            self.set_cart(target)
            fixed.append(f"{ident} from {qty} to {product['quantity']}")
            cart = self.current_cart()
        return fixed

    def trim_extra(self) -> str:
        # Two lines for one product: keep the best in-stock line and drop the other.
        cart = self.current_cart()
        for product in self.products:
            lines = [ident for ident in cart if self.owner(self.listings.get(ident)) is product]
            if len(lines) <= 1 or self.steps_left() <= 2:
                continue
            keep = self.best_line(product)
            if keep is None or keep.id not in lines:
                continue
            target = {ident: qty for ident, qty in cart.items() if ident == keep.id or self.owner(self.listings.get(ident)) is not product}
            target[keep.id] = product["quantity"]
            self.set_cart(target)
            return f"Kept one line for {product['label']} ({keep.id}) and removed the extra. Order that cart."
        return ""

    def drop_strangers(self) -> str:
        # A cart line whose record is not one of the wanted products comes out.
        cart = self.current_cart()
        if len(self.products) <= 1 or self.steps_left() <= 2:
            return ""
        extra = [ident for ident in cart if self.owner(self.listings.get(ident)) is None]
        if not extra:
            return ""
        target = {ident: qty for ident, qty in cart.items() if ident not in extra}
        if not target:
            return ""
        self.set_cart(target)
        return f"Removed {', '.join(extra)}: not a kind the shopper asked for. Order that cart."

    def page_text(self, listing: Listing) -> str:
        chunks: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, str):
                chunks.append(value)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, str):
                        chunks.append(f"{key} {item}")
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(listing.fields)
        if listing.record not in (None, listing.fields):
            walk(listing.record)
        return " ".join(chunks)

    def opened_for(self, product: dict[str, Any]) -> list[Listing]:
        return [listing for listing in self.listings.values()
                if (listing.inspected or listing.paged) and listing.in_stock is not False and self.owner(listing) is product]

    def fit_index(self) -> dict[str, dict[str, set[str]]]:
        # A device page names its model. A part page names the models it works with.
        models: dict[str, set[str]] = {}
        works: dict[str, set[str]] = {}
        notes: dict[str, set[str]] = {}
        own: dict[str, str] = {}
        pages = [listing for listing in self.listings.values() if listing.inspected or listing.paged]
        for listing in pages:
            text = self.page_text(listing)
            model: list[str] = []
            for hit in re.finditer(r"\bmodel\s*(?:no\.?|number|#)\s*[:.]?\s*([A-Z][A-Za-z]{0,5} ?\d{3,6})\b", text, re.I):
                model.extend(_model_codes(hit.group(1)))
            for hit in re.finditer(r"\b(?:also sold as|also known as|sold also as)\s+([^.;]+)", text, re.I):
                model.extend(_model_codes(hit.group(1)))
            models[listing.id] = set(model)
            fitted: list[str] = []
            for hit in re.finditer(r"\b(?<!not )(?<!never )(?:works with|compatible with|fits)\s*:?\s+([^.;]+)", text, re.I):
                before = text[max(0, hit.start() - 40):hit.start()]
                if re.search(r"\bpart\s+[A-Z]{1,4}\d{3,6}\s+also\s+$", before, re.I):
                    continue
                fitted.extend(_model_codes(hit.group(1)))
            works[listing.id] = set(fitted)
            bracket = re.search(r"[\[(]\s*(?:part\s+)?([A-Z]{1,4}\d{3,6})\s*[\])]", text, re.I)
            if bracket:
                own[listing.id] = bracket.group(1).upper()
            for hit in re.finditer(r"\bpart\s+([A-Z]{1,4}\d{3,6})\s+also fits\s*:?\s*([^.;]+)", text, re.I):
                notes.setdefault(hit.group(1).upper(), set()).update(_model_codes(hit.group(2)))
        for ident, number in own.items():
            works.setdefault(ident, set()).update(notes.get(number, set()))
        return {"models": models, "works": works}

    def fit_faults(self, cart: dict[str, int]) -> str:
        # The cart's device is the one whose model the part page must name.
        if not self.fits:
            return ""
        index = self.fit_index()
        notes = []
        for part_label, device_label in self.fits:
            part, device = self.product(part_label), self.product(device_label)
            part_ids = [ident for ident in cart if self.owner(self.listings.get(ident)) is part]
            device_ids = [ident for ident in cart if self.owner(self.listings.get(ident)) is device]
            if len(device_ids) != 1 or not part_ids:
                continue
            codes = index["models"].get(device_ids[0], set())
            if not codes:
                continue
            for ident in part_ids:
                fits = index["works"].get(ident, set())
                if not fits or fits & codes:
                    continue
                good = [listing.id for listing in self.opened_for(part)
                        if listing.id not in cart and index["works"].get(listing.id, set()) & codes][:4]
                notes.append(
                    f"{ident} ({part['label']}) works with {', '.join(sorted(fits)[:4])}, not with {device_ids[0]} ({', '.join(sorted(codes)[:4])})"
                    + (f". Parts that do: {', '.join(good)}." if good else f". No seen {part['label']} does, so try another {device['label']}.")
                )
        return " ".join(notes)

    def fit_plan(self) -> str:
        # Before anything is carted: the codes the part pages name, and which seen device carries each.
        if not self.fits or self.current_cart():
            return ""
        index = self.fit_index()
        by_device: dict[str, list[str]] = {}
        for part_label, device_label in self.fits:
            by_device.setdefault(device_label, []).append(part_label)
        lines = []
        for device_label, part_labels in by_device.items():
            device = self.product(device_label)
            tally: dict[str, dict[str, int]] = {}
            for part_label in part_labels:
                for listing in self.opened_for(self.product(part_label)):
                    for code in index["works"].get(listing.id, set()):
                        tally.setdefault(code, {}).setdefault(part_label, 0)
                        tally[code][part_label] += 1
            ranked = sorted(tally, key=lambda code: (-len(tally[code]), -sum(tally[code].values())))[:4]
            bits = []
            for code in ranked:
                seen = [listing.id for listing in self.opened_for(device) if code in index["models"].get(listing.id, set())][:2]
                who = ", ".join(f"{count} {name}" for name, count in sorted(tally[code].items()))
                where = f"device seen: {', '.join(seen)}" if seen else "device not seen yet"
                bits.append(f"{code}: {who} ({where})")
            if bits:
                lines.append(f"codes the {device['label']} parts name -> " + "; ".join(bits))
        return "FIT: " + " | ".join(lines) if lines else ""

    def fit_block(self, args: dict[str, Any]) -> str:
        ident = next((str(value) for value in args.values() if str(value) in self.listings), "")
        listing = self.listings.get(ident)
        if listing is None:
            return ""
        index = self.fit_index()
        cart = self.current_cart()
        for part_label, device_label in self.fits:
            part, device = self.product(part_label), self.product(device_label)
            if self.owner(listing) is not part:
                continue
            fits = index["works"].get(ident, set())
            chosen = [i for i in cart if self.owner(self.listings.get(i)) is device]
            codes: set[str] = set()
            if chosen:
                for i in chosen:
                    codes |= index["models"].get(i, set())
            else:
                for other in self.opened_for(device):
                    codes |= index["models"].get(other.id, set())
            if not fits or not codes or fits & codes:
                continue
            good = [other.id for other in self.opened_for(part) if other.id != ident and index["works"].get(other.id, set()) & codes][:4]
            if good:
                return (f"NOT SENT (no step used): {ident} works with {', '.join(sorted(fits)[:4])}, "
                        f"and the {device['label']} model is {', '.join(sorted(codes)[:4])}. Parts that name it: {', '.join(good)}.")
        return ""

    def line_faults(self, cart: dict[str, int]) -> list[str]:
        # Read from the records: a missing product, an extra line, a wrong quantity, a sold-out line, an over-budget total.
        faults = []
        for product in self.products:
            lines = [ident for ident in cart if self.owner(self.listings.get(ident)) is product]
            if not lines:
                faults.append(f"{product['label']} has no line")
            elif len(lines) > 1:
                faults.append(f"{product['label']} has {len(lines)} lines")
            for ident in lines:
                listing = self.listings.get(ident)
                if listing is not None and listing.in_stock is False:
                    faults.append(f"{ident} is sold out")
                elif cart[ident] != product["quantity"]:
                    faults.append(f"{ident} is held at {cart[ident]}; {product['label']} wants {product['quantity']}")
        if len(self.products) > 1:
            for ident in cart:
                if self.owner(self.listings.get(ident)) is None:
                    faults.append(f"{ident} is not a kind the shopper asked for")
        if self.budget is not None:
            priced = []
            for ident, qty in cart.items():
                listing = self.listings.get(ident)
                if listing is None or listing.price is None:
                    priced = []
                    break
                priced.append((ident, listing.price, qty))
            if priced and self.budget_scope == "each":
                for ident, price, _qty in priced:
                    if self.budget < price <= 10 * self.budget:
                        faults.append(f"{ident} costs {price:g}, over the {self.budget:g} limit for each item")
            elif priced:
                total = sum(price * qty for _ident, price, qty in priced)
                if self.budget < total <= 10 * self.budget:
                    faults.append(f"the total is {total:g}, over the budget of {self.budget:g}")
        return faults

    def late_question(self) -> str:
        # A message kept for an ending is free when the order itself needs none. Spend it on one property.
        if self.late_asked or "message" not in self.calls or self.messages_left() <= 0 or self.steps_left() < 8:
            return ""
        if any(step["when"] == "order" and step.get("messages", 0) > 0 for step in self.process):
            return ""
        ending = sum(step.get("messages") or 0 for step in self.process if step["when"] == "end")
        if self.messages_left() <= max(0, self.reserve_messages - ending):
            return ""
        self.late_asked = True
        self.ask_done = False
        self.ask_now = True
        self.one_question = True
        asked = len(self.questions)
        self.maybe_ask()
        if len(self.questions) == asked:
            return ""
        return "Asked one more question with a message the order does not need. Check the cart against the answer, then order."

    def handle_order(self, name: str, args: dict[str, Any]) -> str:
        prefix = ""
        if "cart_view" in self.calls and self.cart_view_step < max(self.cart_edit_step, 0) and self.steps_left() >= 3:
            viewed = self.view_cart()
            if viewed:
                prefix = f"(The system viewed the cart first, one step: {viewed})\n"
        cart = self.current_cart()
        if cart:
            late = self.late_question()
            if late:
                return prefix + "NOT SENT (no step used): " + late
        self.repair_for_revealed()
        if self.steps_left() >= 6:
            self.cheaper_probe(self.current_cart())
        self.apply_choice()
        cart = self.current_cart()
        if not cart:
            return prefix + "NOT SENT (no step used): the cart has no lines. Add each product, then order."
        held = self.contradiction_hold(cart)
        if held:
            return prefix + held
        held = self.gate_hold("order")
        if held:
            return prefix + held
        if self.chain_blocked():
            note = self.follow_chains()
            if self.chain_blocked():
                return prefix + "ORDER NOT SENT: a reference page on a required property is still unopened. " + note
        missing = self.missing_products(cart)
        if missing and self.steps_left() > len(missing) + 2:
            added = self.fill_missing()
            still = [product["label"] for product in self.missing_products(self.current_cart())]
            if still:
                return prefix + "NOT SENT (no step used): the cart has no line for " + ", ".join(still) + ". Add one listing for each, then order."
            if added:
                return prefix + "The system added a line for " + ", ".join(added) + ". Check that cart, then order again."
            cart = self.current_cart()
        faults = self.line_faults(self.current_cart())
        if any("not a kind" in fault for fault in faults):
            dropped = self.drop_strangers()
            if dropped:
                return prefix + dropped
        if faults and all(" is held at " in fault for fault in faults):
            fixed = self.fix_counts()
            if fixed:
                return prefix + "The system set the quantity the request asked for: " + "; ".join(fixed) + ". Order that cart."
        if any("has " in fault and "lines" in fault for fault in faults):
            trimmed = self.trim_extra()
            if trimmed:
                return prefix + trimmed
        fit = self.fit_faults(self.current_cart())
        if fit and self.steps_left() > 4:
            return prefix + "ORDER NOT SENT: " + fit
        faults = self.line_faults(self.current_cart())
        if any("sold out" in fault for fault in faults):
            if self.steps_left() > 3:
                swapped = self.swap_line(dict(self.current_cart()))
                if swapped:
                    return prefix + swapped
            return prefix + "ORDER NOT SENT: " + "; ".join(fault for fault in faults if "sold out" in fault)
        over = [fault for fault in faults if "over the" in fault]
        if over:
            return prefix + "ORDER NOT SENT: " + "; ".join(over)
        if self.refused_carts.get(_compact(sorted(cart.items())), 0) >= 2:
            return prefix + ("NOT SENT (no step used): the store has refused this exact cart twice. Read what it said "
                             "(a line sold out, a new budget, a changed requirement) and change the cart before ordering again.")
        read = self.preorder_reads(cart)
        if read:
            prefix += f"(The system opened {', '.join(read)} before the check, {len(read)} step(s).)\n"
            self.apply_choice()
            cart = self.current_cart()
            held = self.contradiction_hold(cart)
            if held:
                return prefix + held
        held = self.basket_summary()
        if held:
            return prefix + held
        before = self.step
        result = self.send(name, args)
        if not self.finished and self.step > before:  # a call that failed in transit is no refusal
            self.note_refusal(cart)
            self.order_refusals += 1
            self.verify_rejections = 0
            swapped = self.swap_line(cart)
            if swapped:
                result += "\n" + swapped
            twins = self.twin_hint(self.current_cart())
            if twins:
                result += "\n" + twins
        return prefix + result

    def contradiction_hold(self, cart: dict[str, int]) -> str:
        # Hold the order while a record contradicts a line, at most twice per task; after that the order goes out,
        # since the controller may read the record differently and a decline earns nothing.
        faults = self.cart_contradictions(cart, soft=True)
        if not faults or self.contradiction_holds >= 2:
            return ""
        self.contradiction_holds += 1
        self.dialogue.append({"kind": "contradiction_hold", "faults": faults[:4]})
        return ("NOT SENT (no step used): the records do not support the cart: " + "; ".join(faults[:4])
                + ". Replace that line with a listing whose property states the value (search the value, open the row), "
                "or order again if you read the record differently.")

    def preorder_reads(self, cart: dict[str, int]) -> list[str]:
        # KYO reads its cart lines and cheaper rows before ordering: open the cart's unopened lines, and for a product
        # that must be the cheapest, up to two cheaper in-stock rows of the same kind that were never opened
        key = _compact(sorted(cart.items()))
        if key in self.preread or "details" not in self.calls:
            return []
        self.preread.add(key)
        todo = [i for i in cart if i in self.listings and not self.listings[i].inspected]
        for ident in cart:
            line = self.listings.get(ident)
            product = self.owner(line)
            if line is None or line.price is None or product is None or not product["cheapest"]:
                continue
            rows = [l for l in self.listings.values() if not l.inspected and l.id not in cart and l.id not in self.refused_ids
                    and l.in_stock is not False and l.price is not None and l.price < line.price and self.owner(l) is product]
            todo += [l.id for l in sorted(rows, key=lambda l: l.price or 0)[:2]]
        read = []
        for ident in todo[:4]:
            if self.steps_left() < 8:
                break
            filled = self.fill("details", ident=ident)
            data = self.env_call(*filled) if filled else None
            if data is None:
                break
            self.absorb(filled[0], filled[1], data.get("observation"))
            listing = self.listings.get(ident)
            if listing is not None:
                self.read_notes(listing)
            read.append(ident)
        return read

    def cart_sane(self, cart: dict[str, int]) -> bool:
        # the cart can still pass: within the budget and no line known sold out (a failed check alone is not a decline)
        total = sum((self.listings[i].price or 0) * q for i, q in cart.items() if i in self.listings)
        return bool(cart) and self.usable(cart) and (self.budget is None or self.budget_scope == "each" or total <= self.budget)

    def note_refusal(self, cart: dict[str, int]) -> None:
        # A refused order: remember the cart, not its listings. A refusal may carry a budget change or a dropped
        # requirement after which the same listing is still right; sold-out lines are excluded by mark_gone.
        key = _compact(sorted(cart.items()))
        self.refused_carts[key] = self.refused_carts.get(key, 0) + 1
        self.verified_cart = {}
        self.finish_refused_at = self.step
        self.finish_refusal = " ".join(self.last_heard)[:400] if self.last_heard_step == self.step else "(see its answer above)"

    def basket_summary(self, final: bool = False) -> str:
        # When the request asks to approve the basket, show the final cart once before ordering.
        #
        # Returns text for the controller when the shopper's reply changes something (the order is then held),
        # else "" and the order goes ahead. Product ids stay out of the message.
        cart = self.current_cart()
        if not self.basket_check or not cart or self.basket_sent_for == cart or self.messages_left() <= 0 or "message" not in self.calls:
            return ""
        parts, total = [], 0.0
        for ident, qty in cart.items():
            listing = self.listings.get(ident)
            price = listing.price if listing else None
            total += (price or 0) * qty
            parts.append(f"{qty} x {(listing.title if listing else 'item')[:80]}" + (f" at {price:g} each" if price is not None else ""))
        text = "Here is the basket before I place the order: " + "; ".join(parts) + f". Total {total:g}, all in stock. Shall I place the order?"
        filled = self.fill("message", text=text)
        if not filled:
            return ""
        data = self.env_call(*filled)
        self.basket_sent_for = dict(cart)
        if data is None:
            return ""
        self.messages_used += 1
        lines = self.shopper_lines(filled[0], filled[1], data.get("observation"))
        self.inject(filled[0], filled[1], _compact(_brief(data.get("observation"))))
        if lines and not final:
            # the model reads the whole reply: a change can be worded any way
            result = self.interpret(lines, text)
            if result.get("add") or result.get("drop") or _number(result.get("budget")) or result.get("stock_out") \
                    or result.get("price_changed") or result.get("reply_type") == "update" or result.get("instructions"):
                return "NOT SENT: the shopper answered the basket summary with a change: " + " | ".join(lines)[:400] + " Re-check the cart against it, then order again."
        return ""

    def same_title(self, listing: Listing) -> list[Listing]:
        key = re.sub(r"\W+", " ", listing.title.lower()).strip()
        return [
            l for l in self.listings.values()
            if l.id != listing.id and l.id not in self.refused_ids and l.in_stock is not False and re.sub(r"\W+", " ", l.title.lower()).strip() == key
        ]

    def twin_hint(self, cart: dict[str, int]) -> str:
        # After a stock-out at checkout: the same product is often listed again under another id, so search its exact title.
        lines = []
        for ident in cart:
            listing = self.listings.get(ident)
            if listing is None or listing.in_stock is not False or len(listing.title) < 8:
                continue
            twins = self.same_title(listing)
            if not twins and self.steps_left() > 4:
                filled = self.fill("search", text=listing.title[:150])
                if filled:
                    data = self.env_call(*filled)
                    if data is not None:
                        self.absorb(filled[0], filled[1], data.get("observation"))
                twins = self.same_title(listing)
            if twins:
                lines.append(
                    f"The system searched the sold-out {ident}'s title: listings with the same title (likely the same product listed again): "
                    + "; ".join(self.listing_line(l) for l in twins[:4])
                    + ". Inspect one; if its record meets every requirement, put it in the cart instead of the sold-out one."
                )
        return "\n".join(lines)

    def clean_remark(self, text: str) -> str:
        text = _ID_TOKEN.sub("", text)
        for ident in self.listings:
            text = text.replace(ident, "")
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text or "Sorry, nothing in the store meets all of your requirements."

    def decline_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        args = dict(args)
        call = self.calls.get("decline") or {}
        for key, value in (call.get("args") or {}).items():
            if value == "<TEXT>":
                current = args.get(key)
                args[key] = self.clean_remark(str(current) if isinstance(current, str) and current.strip() else "Sorry, nothing in the store meets all of your requirements.")
            elif key not in args:
                args[key] = value
        for key, value in list(args.items()):
            if isinstance(value, str):
                args[key] = self.clean_remark(value) if len(value) > 20 else value
        return args

    def handle_decline(self, name: str, args: dict[str, Any]) -> str:
        held = self.gate_hold("end")
        if held:
            return held
        enough_left = self.steps_left() >= 4
        if enough_left and self.decline_pushbacks < 2:
            if len(self.query_keys) < 3:
                self.decline_pushbacks += 1
                return "NOT SENT (no step used): only a few searches were tried. Before declining, search with other words, synonyms or filters and inspect the plausible listings."
            self.decline_reads()  # the audit then sees these records
            audit = self.audit_decline()
            may_fit = [str(i) for i in (audit or {}).get("may_fit") or []]
            self.decline_candidates = [i for i in may_fit if i in self.listings] or self.decline_candidates
            more = [str(q) for q in (audit or {}).get("search_more") or []]
            if may_fit or (more and self.decline_pushbacks == 0):
                self.decline_pushbacks += 1
                return (
                    "NOT SENT (no step used): a wrong decline scores 0. "
                    + (f"These listings may fit and need checking first: {', '.join(may_fit[:8])}. " if may_fit else "")
                    + (f"Untried searches: {'; '.join(more[:3])}. " if more else "")
                    + "If after that nothing fits, decline again."
                )
        self.repair_for_revealed()
        ordered = self.order_instead_of_decline()
        if ordered:
            return ordered
        if not self.current_cart():
            self.salvage_cart()
            ordered = self.order_instead_of_decline()
            if ordered:
                return ordered
        if self.current_cart() or self.listings:
            return ("NOT SENT (no step used): do not decline. The store has listings"
                    + (" and the cart has lines." if self.current_cart() else ".")
                    + " Match what the shopper stated and order.")
        before = self.step
        result = self.send(name, self.decline_args(name, args))
        if not self.finished and self.step > before:
            # race 11: a decline answered with a new budget ("could we go up to 806?") was sent again unchanged
            self.finish_refused_at = self.step
            self.finish_refusal = " ".join(self.last_heard)[:400] if self.last_heard_step == self.step else "(see its answer above)"
            result += ("\nThe store did not accept the decline. Read what it said: a new budget or a dropped requirement may make a "
                       "listing fit; act on it before finishing.")
        return result

    def order_instead_of_decline(self) -> str:
        # Order the cart rather than decline when the checker passes it.
        #
        # Race 6: declines passed 20/517 against 122/1007 for orders, and in 55 of 85 wrong declines the agent had
        # opened the listing that passed.
        cart = self.current_cart()
        # a partial basket is no pick: an order missing a product fails. Each product needs a line of its own; a line whose
        # product is unclear can stand for one product (two lines of one product do not cover a second)
        owners = [self.owner(self.listings.get(i)) for i in cart]
        uncovered = [p for p in self.products if not any(o is p for o in owners)]
        own_pick = bool(cart) and len(uncovered) <= sum(1 for o in owners if o is None)
        if not cart and "cart_add" in self.calls and self.steps_left() >= len(self.products) + 2:
            # nothing held: take the decline audit's likeliest fit for each wanted product that the store shows in stock,
            # all within budget (race 7: wrong declines on multi-product tasks had usually opened every line's fit)
            target: dict[str, int] = {}
            spent = 0.0
            for product in self.products:
                own = {l.id for l in self.product_listings(product)}
                for ident in self.decline_candidates:
                    listing = self.listings[ident]
                    if ident in target or (len(self.products) > 1 and ident not in own):
                        continue
                    cost = (listing.price or 0) * product["quantity"]
                    if listing.inspected and listing.in_stock is not False and ident not in self.refused_ids and listing.price is not None \
                            and (self.budget is None or spent + cost <= self.budget or (self.budget_scope == "each" and cost <= self.budget)):
                        target[ident] = product["quantity"]
                        spent += cost
                        break
            if len(target) == len(self.products):
                self.set_cart(target)
                cart = self.current_cart()
                own_pick = False
        if self.decline_ordered or not self.usable(cart) or "order" not in self.calls or self.steps_left() < 1:
            return ""
        # a cart the agent filled itself is ordered unless the checker finds a clear contradiction (race 7, top 4 agents:
        # declines passed 10/108, and 21% of wrong declines across races 1-7 had the passing listing in the cart). A cart
        # filled here from the audit needs a full pass: a listing that merely lacks a field for a requirement is a right
        # decline more often than not (practice t-70ef: an ordered hoodie with no style field failed where the decline was right)
        verdict = self.verify(cart) if self.steps_left() > 2 else None
        # a hard failure means the lines ignore a stated value: search that value and replace them, then check once more
        if verdict is not None and (verdict.get("hard_fail") or (not own_pick and verdict.get("ok") is not True)):
            self.repair_for_revealed()
            cart = self.current_cart()
            if not self.usable(cart):
                return ""
            verdict = self.verify(cart) if self.steps_left() > 2 else None
            if verdict is not None and (verdict.get("hard_fail") or (not own_pick and verdict.get("ok") is not True)):
                return ""
        # no check possible (no time or the model failed): the agent's own basket beats a decline, which nearly always fails
        if verdict is None and not own_pick:
            return ""
        self.decline_ordered = True
        filled = self.fill("order")
        if not filled:
            return ""
        held = self.basket_summary()
        if held or self.steps_left() <= 0:
            return "DECLINE NOT SENT: the checker passes the cart, so the system showed it to the shopper to order it. " + held
        before = self.step
        result = self.send(*filled)
        if not self.finished and self.step > before:
            self.note_refusal(cart)
            self.order_refusals += 1
            twins = self.twin_hint(cart)
            if twins:
                result += "\n" + twins
        return "DECLINE NOT SENT: the checker passes the cart, so the system ordered it. The store answered: " + result

    def decline_reads(self) -> list[str]:
        # Before the first decline audit, read the unread rows that look like a wanted product (KISAME's lacking_rows):
        # in races most wrong declines had already seen the passing listing in a search row.
        if self.decline_reads_done or "details" not in self.calls:
            return []
        self.decline_reads_done = True
        read: list[str] = []
        for product in self.products:
            if not self.product_listings(product):
                self.browse_category(product)
        for product in self.products:
            wanted = _stems(f"{product.get('category') or ''} {product['label']}")
            quantity = product["quantity"]
            rows = [
                l for l in self.listings.values()
                if not l.inspected and l.in_stock is not False and l.id not in self.refused_ids
                and (self.budget is None or l.price is None or l.price * quantity <= self.budget)
                and (wanted & _stems(l.category) or wanted & _stems(l.title))
            ]
            rows.sort(key=lambda l: (-(2 * len(wanted & _stems(l.category)) + len(wanted & _stems(l.title))), l.price or 0))
            for listing in rows[: 6 if len(self.products) == 1 else 3]:
                if len(read) >= 6 or self.steps_left() < 8:
                    break
                filled = self.fill("details", ident=listing.id)
                data = self.env_call(*filled) if filled else None
                if data is None:
                    break
                self.absorb(filled[0], filled[1], data.get("observation"))
                self.read_notes(listing)
                read.append(listing.id)
        if read:
            self.dialogue.append({"kind": "decline_reads", "ids": read})
        return read

    def audit_decline(self) -> dict[str, Any] | None:
        if self.over_budget(soft=False):
            return None
        rows = [self.listing_line(l, full=True) for l in sorted(self.listings.values(), key=lambda l: (not l.inspected, l.price or 0))[:45]]
        content = (
            f"REQUEST:\n{self.request}\n\n{self.ledger_text()}\n\nSEARCHES DONE:\n" + "\n".join(self.searches[-15:])
            + "\n\nLISTINGS SEEN:\n" + ("\n".join(rows) if rows else "none")
        )
        audit = self.ask_json(AUDIT_PROMPT, content, STRONG_MODELS, 600)
        self.dialogue.append({"kind": "audit", "verdict": audit})
        return audit

    def best_guess(self) -> list[tuple[str, int]]:
        # Items for a forced order: earlier cart choices first, else the cheapest affordable inspected listing.
        fresh = {i: q for i, q in self.added.items() if i not in self.refused_ids}
        if fresh and not self.missing_products(fresh):
            return list(fresh.items())
        picks = []
        for product in self.products:
            listing = self.best_line(product)
            if listing is not None and all(listing.id != ident for ident, _ in picks):
                picks.append((listing.id, product["quantity"]))
        return picks

    def usable(self, cart: dict[str, int]) -> bool:
        return bool(cart) and all(i not in self.refused_ids and (i not in self.listings or self.listings[i].in_stock is not False) for i in cart)

    def order_mode(self, name: str, args: dict[str, Any]) -> bool:
        call = self.calls.get("order") or {}
        if call.get("tool") != name:
            return False
        fixed = {k: v for k, v in (call.get("args") or {}).items() if not (isinstance(v, str) and v.startswith("<"))}
        return bool(fixed) and all(args.get(k) == v for k, v in fixed.items())

    def revealed_values(self, product: dict[str, Any]) -> list[str]:
        values = []
        for req in product["reqs"]:
            if req.get("source") != "revealed" or not req["active"] or ":" not in req["text"]:
                continue
            raw = req["text"].split(":", 1)[1].strip()
            if _limit_bound(raw):
                continue
            value = re.sub(r"^(?:at least|at most|under|over|one of|not|no)\s+", "", raw, flags=re.I).strip()
            if value:
                values.append(value)
        return values

    def has_limit(self, product: dict[str, Any]) -> bool:
        return any(req["active"] and ":" in req["text"] and _limit_bound(req["text"].split(":", 1)[1]) for req in product["reqs"])

    def states_values(self, listing: Listing, values: list[str]) -> bool:
        blob = f"{listing.title} {listing.category} " + " ".join(str(v) for v in listing.properties().values())
        words = _words(blob) | _stems(blob)
        for value in values:
            needed = _words(value) or _stems(value)
            if needed and not (needed & words):
                return False
        return True

    def repair_for_revealed(self) -> None:
        # A revealed value replaces a cart line that does not state it. Race 13 declined those carts.
        revealed = [product for product in self.products if self.revealed_values(product) or self.has_limit(product)]
        if not revealed:
            return
        for product in revealed:
            for req in product["reqs"]:
                if req.get("source") == "revealed" and req["active"]:
                    req["fresh"] = True
        if self.steps_left() > 4:
            self.value_search(revealed)
        target = dict(self.current_cart())
        changed = False
        for product in revealed:
            values = self.revealed_values(product)
            # the pool holds the product's own kind only (a nylon bomber jacket is not a nylon hoodie: practice
            # t-70ef000b3f), with no record that contradicts a requirement; a line with no replacement stays
            pool = [
                listing for listing in self.listings.values()
                if listing.in_stock is not False and listing.price is not None and listing.id not in self.refused_ids
                and (self.owner(listing) is product or (self.owner(listing) is None and not listing.category and len(self.products) == 1))
                and self.states_values(listing, values) and self.meets_limits(listing, product)
                and not self.line_contradicts(listing, product)
            ]
            if not pool:
                continue
            pool.sort(key=lambda listing: (0 if listing.inspected else 1, listing.price or 0))
            pick = pool[0]
            owned = [ident for ident in list(target) if self.owner(self.listings.get(ident)) is product]
            if owned == [pick.id] and target.get(pick.id) == product["quantity"]:
                continue
            if owned and all(self.states_values(self.listings[i], values) and self.meets_limits(self.listings[i], product)
                             and not self.line_contradicts(self.listings[i], product) for i in owned):
                continue  # the line held already states the revealed values
            for ident in owned:
                target.pop(ident, None)
            target[pick.id] = product["quantity"]
            changed = True
        if changed and target:
            self.set_cart(target)

    def salvage_cart(self) -> None:
        # One in-stock line per product from what the search already returned, so the order is not empty.
        if self.current_cart() or "cart_add" not in self.calls:
            return
        target: dict[str, int] = {}
        for product in self.products:
            wanted = _stems(f"{product.get('category') or ''} {product['label']}")
            values = self.revealed_values(product)
            rows = []
            for listing in self.listings.values():
                if listing.in_stock is False or listing.price is None or listing.id in self.refused_ids:
                    continue
                query = " ".join(str(v) for v in (self.found_by.get(listing.id) or {}).values())
                if wanted and not (wanted & _stems(f"{listing.title} {listing.category} {query}")):
                    continue
                if values and not self.states_values(listing, values):
                    continue
                if not self.meets_limits(listing, product) or not self.states_plain(listing, product):
                    continue
                rows.append(listing)
            if self.budget is not None and self.budget_scope != "each":
                affordable = [listing for listing in rows if (listing.price or 0) * product["quantity"] <= self.budget]
                rows = affordable or rows
            if not rows:
                continue
            rows.sort(key=lambda listing: (0 if listing.inspected else 1, listing.price or 0))
            target[rows[0].id] = product["quantity"]
        if target:
            self.set_cart(target)

    def set_cart(self, target: dict[str, int]) -> None:
        # Make the cart hold the target lines, keeping one step for the order.
        call = self.calls.get("cart_add") or {}
        if "<LINES>" in (call.get("args") or {}).values():
            line = call.get("line") or {}
            rows = []
            for ident, qty in target.items():
                row: dict[str, Any] = {}
                if line.get("id"):
                    row[line["id"]] = ident
                if line.get("qty"):
                    row[line["qty"]] = qty
                rows.append(row)
            key = next(k for k, v in call["args"].items() if v == "<LINES>")
            data = self.env_call(call["tool"], {key: rows})
            if data is not None:
                self.absorb(call["tool"], {key: rows}, data.get("observation"))
            return
        cart = self.current_cart()
        edits = [("cart_remove", i, q) for i, q in cart.items() if i not in target]
        edits += [("cart_add", i, q) for i, q in target.items() if i not in cart]
        if edits and self.steps_left() < len(edits) + 1:
            # too few steps to finish the change and still order: a removal alone would leave an empty cart and a decline
            # (practice t-13e82f2c08: the swap's add was skipped at the last step)
            self.dialogue.append({"kind": "cart_change_skipped", "edits": len(edits), "steps_left": self.steps_left()})
            return
        for role, ident, qty in edits:
            filled = self.fill(role, ident=ident, qty=qty)
            if filled:
                data = self.env_call(*filled)
                if data is not None:
                    self.absorb(filled[0], filled[1], data.get("observation"))

    def early_checkout(self) -> str:
        # Order while steps remain to handle a refusal: the store may report a change only when an order is tried.
        cart = self.current_cart()
        if cart != self.verified_cart and self.steps_left() > 4:
            verdict = self.verify(cart) if cart else None
            if not (verdict and verdict.get("ok") is True):
                if self.usable(self.verified_cart) and "cart_remove" in self.calls:
                    self.set_cart(self.verified_cart)
                elif verdict and verdict.get("hard_fail"):
                    return ""  # nothing checked to fall back on: leave the fix to the controller
        cart = self.current_cart()
        filled = self.fill("order") if cart else None
        if not filled or self.steps_left() <= 0:
            return ""
        held = self.basket_summary()
        if held or self.steps_left() <= 0:
            return "Steps are running low; the system showed the shopper the basket before ordering. " + held
        before = self.step
        result = self.send(*filled)
        if self.finished or self.step == before:
            return ""
        self.note_refusal(cart)
        self.order_refusals += 1
        self.verify_rejections = 0
        twins = self.twin_hint(cart)
        if twins:
            result += "\n" + twins
        return (
            "Steps are running low, so the system ordered the checked cart now. The store answered: " + result
            + "\nIf the store refused, handle what it said with the steps left (swap a listing that is gone, re-check a changed requirement), then order again."
        )

    def force_finish(self, reason: str) -> None:
        self.dialogue.append({"kind": "force_finish", "reason": reason, "spend_usd": round(self.spend, 4)})
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        cart = self.current_cart()
        if cart and set(cart) & self.refused_ids:
            if not self.swap_line(dict(cart)) and "cart_remove" in self.calls:
                for ident in [i for i in cart if i in self.refused_ids]:
                    if self.steps_left() <= 2:
                        break
                    filled = self.fill("cart_remove", ident=ident, qty=cart[ident])
                    if filled:
                        data = self.env_call(*filled)
                        if data is not None:
                            self.absorb(filled[0], filled[1], data.get("observation"))
            cart = self.current_cart()
        self.repair_for_revealed()
        self.apply_choice()
        if not self.current_cart() and self.steps_left() > 1:
            self.salvage_cart()
        cart = self.current_cart()
        # any cart is ordered unless the records show it cannot pass: every line known sold out (and no step left to
        # swap it: practice t-069a67a0c4 re-ordered the gone line) or contradicting a requirement with nothing better
        doomed = bool(cart) and (all(not self.usable({i: q}) for i, q in cart.items())
                                 or all(self.line_contradicts(self.listings[i], self.owner(self.listings.get(i))) for i in cart
                                        if i in self.listings and self.owner(self.listings.get(i)) is not None) and self.cart_contradictions(cart))
        if doomed:
            self.dialogue.append({"kind": "forced_decline", "faults": self.cart_contradictions(cart)[:4], "cart": cart})
            cart = {}
        if cart and self.steps_left() > 1:
            self.basket_summary(final=True)
        if cart and self.steps_left() > 0:
            filled = self.fill("order")
            if filled:
                data = self.env_call(*filled)
                if data is not None and not self.finished:
                    obs = data.get("observation")
                    if isinstance(obs, dict) and str(obs.get("request_fault") or "") == "NO_ITEMS":
                        return
                    lines = self.shopper_lines(filled[0], filled[1], obs)
                    if lines:
                        self.interpret(lines, None)
                    if set(cart) & self.refused_ids and self.steps_left() > 2 and self.swap_line(dict(cart)):
                        again = self.fill("order")
                        if again and self.steps_left() > 0:
                            self.env_call(*again)
        # decline only with an empty cart (nothing was ever worth holding) or a cart the records show cannot pass
        if not self.finished and self.steps_left() > 0 and (doomed or not self.current_cart()):
            filled = self.fill("decline", text="Sorry, nothing in the store meets all of your requirements.")
            if filled:
                self.env_call(filled[0], self.decline_args(*filled))


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    agent = Agent(problem_data)
    try:
        return agent.run()
    except Exception as error:  # keep the episode alive: finish with what is known
        agent.dialogue.append({"kind": "agent_error", "error": f"{type(error).__name__}: {error}"[:300]})
        try:
            if not agent.finished:
                agent.force_finish("error")
        except Exception as second:  # noqa: BLE001
            agent.dialogue.append({"kind": "agent_error", "error": f"{type(second).__name__}: {second}"[:300]})
        return agent.dialogue


__all__ = ["agent_main"]
