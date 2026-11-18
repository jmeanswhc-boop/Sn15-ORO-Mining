from __future__ import annotations

import concurrent.futures
import itertools
import math
import json
import re
import threading
import time
from collections import Counter
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"
# w13o: the question planner runs on Haiku. Picking the hidden need from the request and the listed properties, Haiku put
# it in its top 3 70% of the time against Sonnet's 69% and Opus's 68% (204 past tasks with their request text, mostly
# suite 146), at about a sixth of Sonnet's cost; the run key's credit then lasts more tasks.
PLAN_MODEL = "anthropic/claude-haiku-5.5"
# Claude 5.5 models think before answering a hard prompt, and the thinking counts against max_tokens: on this planner's
# prompt Haiku spent all of 3,000 tokens thinking and returned nothing. With 8,000 none of 204 past tasks came back
# empty, and the plan put the hidden need among its first 3 listed properties 73% of the time (63% with thinking off).
PLAN_TOKENS = 8000
CUTOFF_TOKENS_MAX = 16000  # a cut-off answer is retried with twice the limit, up to this
VALUE_MODEL = "anthropic/claude-sonnet-5.5"

TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
# OpenRouter credit one task may spend. On the network each run's model key is capped (about
# $5 for a 90-task run), and tasks still running when it runs out score 0. Calls billed to the
# miner's own provider key (BYOK) cost no credit, so normally none of this acts. Past each
# mark the task gets cheaper: Haiku in place of Sonnet, then wrap up, then finish at once.
CREDIT_CHEAP = 0.01
CREDIT_WRAP_UP = 0.03
CREDIT_FINISH = 0.045
OBS_CHARS = 16000
MAX_TURNS = 80
COMMIT_STEPS = 9

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.
- Search with the words of the required values (for example the material or pattern the shopper named) inside the requested category: listing values are searchable, property names are not, and guessed brand or model words only hide candidates. When a word search cannot be limited to the category and finds more rows than you can see, add a word the wanted listings are sure to carry, such as the requested product's own word next to the required value (for a striped wool rug, "striped wool" rather than "striped"), and open the hits in the requested category. When most visible rows belong to other categories, the words are too common or too many: search again with fewer, more distinctive words (the required value's key word with the product's own word) and keep the search whose rows are in the requested category. Results are not sorted by price or by measure, so for the cheapest or the best a list larger than you can see may hide the answer: split it with one more word at a time (a value seen on listings, such as a size, material or brand) and read each part, and for the best by a measure also search the required words with smaller (or larger) values of that measure. After a search, a [candidates] line tracks what is covered: whether a search made only of required values' words has been seen completely, and the in-stock, within-budget candidates by price with their check results. When it calls the list complete, rely on it. If changing the words does not change the reported total, that search mode ignores the words: narrow it with its filters (for the cheapest, lower the price limit) or use a mode that reads words.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down. They reveal one only when a message asks about that property. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, as a short, plain, friendly question to the shopper that names the property (an example value or two is fine). Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- When a planner is available, your questions are chosen and sent for you as soon as you have searched with the required values' words and opened one or two of the hits; do not open other listings just for this. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- If you must ask yourself: ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates; one property per message, named the way the listings name it, in one short plain question. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- A request can ask for several different products in one order. Then the order needs one cart line per product, each at that product's own quantity and meeting that product's own requirements; any missing line fails the whole task. The [item] part of the status says which product you are working on: search for it in its own category, open its listings, put its fitting listing in the cart, and then the next product comes up. Order only when every product has its line.
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set. A new requirement adds to the stated ones unless the shopper says it replaces one: never drop a stated requirement to meet a new one; when no listing meets both, decline.
- An order attempt can be REFUSED. The result then has a "refusal" field, even if the inner result looks successful. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again. A refusal announces all pending changes at once, so the next order attempt is usually final: make sure that pick meets every requirement and change before you send it.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- After you open a listing's details, a [check] line gives the exact result for each requirement under the scoring rule: a listing meets a requirement when one of its own property fields whose name contains the requirement's property word states the value (singular or plural, one element of a list is enough); its title or description counts only when it has no such field at all. MET and NOT MET are final. NO EVIDENCE means the listing does not show the value, so it does not fit. UNCLEAR (for example different units) is for you to decide from the details.
- A listing whose check says it fits every requirement is a valid choice: do not reject it for anything the request and the shopper did not ask. A listing whose check is contradicted (one field named for the property states the value and another field states something else) usually fits. Without a cheapest or best rule, prefer a listing without contradiction when one exists. Under a cheapest or best rule it is a candidate like any other: never pass over a cheaper (or better) one for that reason, and follow the [candidates] verdict.
- An excluded value ("leave out", "not", "nothing with") rules out every listing whose field states it, and "only X or Y" accepts nothing else.
- Requirements marked "not blocking" (things no listing can show, such as delivery time) never rule a listing out. Wishes only decide between valid listings.
- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.

FINISHING
- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.
- Before every order attempt: the cart holds exactly one line, the chosen item at the required quantity, and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.
- Decline only after a thorough search finds nothing that meets every requirement; while a listing fits, a decline is sent back. Follow the request's rule for declining exactly. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Order as soon as your choice is final (the questions are asked, its [check] says it fits, and for a cheapest or best request the [candidates] line proves it): every pending change shows up at your first order attempt, and a refused order costs one step and can be fixed. When 7 steps remain, only cart, stock-check and finishing tools stay available, and if no order was attempted when 9 steps remain, your best candidate is put in the cart and ordered for you. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words.
"quantity": how many units of the same listing the shopper explicitly asks you to buy (an integer). It is 1 unless the request literally asks for several units of the item itself. Numbers that count people, occasions, days, rooms, uses, earlier purchases, money, messages or steps are not a quantity. Something sold as one listing, such as a pair of shoes, a set or a pack, is quantity 1.
"quantity_quote": the exact words of the request that ask for that number of units, copied verbatim; "" when the quantity is 1.
"quantity_word": the number exactly as it is written inside quantity_quote (for example "3" or "three"); "" when the quantity is 1.
"requirements": a list of objects, one per firm stated requirement on the product, each {"property": ..., "op": ..., "values": [...], "unit": ..., "blocking": ..., "quote": ...}:
  "property": the product property in the request's own word for it, for example "material", "fit", "collar type". When the request names it only through a phrase (such as "made of"), use the plain property noun ("material"). When the request gives a material for a named part of the product ("the body has to be spruce"), the property is that part's material ("body material"). Otherwise never rename it to a narrower or different property, and never add words the request does not use.
  "op": "is" (it must be one of the values), "not" (it must be none of the values), "at_most", "at_least" or "between" (number limits).
  "values": the required values in the request's own words, each the plain term a listing would state (when the request stacks describing words on one property, such as "high, deep and plush", the value is the plain term "high"); numbers as numbers; for "between", [low, high]. A range written as one label (such as "a 15-30 minute assembly") is a value as written: op "is" with "15-30 minutes"; use "between" only when the request states two limits in words ("between 15 and 30 minutes").
  "unit": the unit the request writes next to a number, as written; "" otherwise.
  "blocking": false only for a statement that no product listing could show (a delivery time, the shop's service, the shopper's own plans); true otherwise. A value the shopper says they will check themselves ("I'll check it myself", "I'll look it over", "I don't take a label on faith") is still a requirement a listing shows: true.
  "quote": the exact words of the request that state it.
  A firm statement of what the product is for, such as an occasion or a use ("something I'd wear for a party and nothing else"), is a requirement on that property ("occasion": "party"), unless those words are part of the category's name.
  A requirement names a value a product listing could state. A sentence that only describes a benefit, a worry or a use without naming such a value (for example "it has to wash easily" or "it must keep going with no fuss") is a hint about something to ask, not a requirement.
  Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.
"price_cap": the maximum total price as a number written exactly as the request writes it (in the request's own unit, never converted), or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"priority_property": for "best", the listing property it is measured by, in the request's own word (for example "width"); "" otherwise.
"priority_direction": for "best", "least" when the smallest value wins (narrowest, lightest, shortest, lowest) or "most" when the largest wins; "" otherwise.
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"items": the products to buy, as a list in the order the request names them, each {"product": ..., "quantity": ..., "quantity_quote": ..., "quantity_word": ..., "requirements": [...]} with the same meaning as the keys above, for that product alone. A request for one product gives a list of one, the same as "product", "quantity" and "requirements". Several products are several entries only when the request wants each bought (each with its own listing in the order): products named together as one thing to buy, or a product and its parts, are one entry. Each requirement belongs to the product it is stated for.
"budget_scope": "each" when the price cap is a limit for every product separately; "total" when it covers the whole order (the usual case); null when there is no price cap.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": only these, which the agent cannot choose without: the budget when no price limit is stated; which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements); or a point the request explicitly tells you to confirm with the shopper. Never tastes, looks, style or what someone likes: those are hidden needs that the question planner finds. Usually empty.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying. One tool may fill both "order" and "decline" when an argument chooses between buying and not buying.
"order_args": an object with the argument values that make the "order" tool buy (such as the value of a mode or choice argument), using the exact argument names and allowed values from that tool's schema; {} when no such argument exists.
"decline_args": the same for the "decline" tool: the argument values that make it end without buying; {} when no such argument exists.
"cart_args": {"add": the argument values (other than the item and the amount) that make the "cart" tool add an item, "remove": the same for taking an item out}, using exact argument names and allowed values from its schema; {} for either when no such argument exists.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new or changed firm requirements on the product, each an object {"property": ..., "op": ..., "values": [...], "unit": ..., "quote": ...}: "property" is the product property in the shopper's own word for it (the plain property noun when they name it only through a phrase), never renamed or narrowed; "op" is "is", "not", "at_most", "at_least" or "between"; "values" are the shopper's values (numbers as numbers; [low, high] for "between"); "unit" is the unit written next to a number, or ""; "quote" is the shopper's exact words. A requirement that replaces an earlier one on the same property is listed here with its new value.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of the properties (as named under REQUIREMENTS SO FAR) whose requirement the shopper no longer needs, without a replacement.
"new_budget": the new maximum total price as a number, or null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"price_changed": true when the shopper says the price of an item has changed (up or down), else false.
"new_quantity": the number of units of the item the shopper now explicitly asks you to buy, as an integer, or null when they say nothing about how many units to buy. A number that counts people, days, occasions or money is not a quantity.
"quantity_quote": the shopper's exact words that ask for that number of units, copied verbatim; "" when new_quantity is null.
"quantity_word": the number exactly as it is written inside quantity_quote; "" when new_quantity is null.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".
"not_understood": true when the shopper only says they did not take in the assistant's message, in any wording (they did not follow it, it was too wordy or dense, it went over their head, please say it again, simpler, shorter or differently), without answering it; otherwise false. Such a line is not a question to reply to.
"one_at_a_time": true when the shopper does not answer because the message asked too many things at once and wants them sent separately, one after another, or only the first one, in any wording; else false. When this is true, not_understood is false.
One line can carry several changes: report every one. A requirement replaced by another on the same property is only a new need (with the new value); a replacement on a different property is a new need plus the old property in dropped. When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such an answer never drops or changes any other requirement."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant puts an item in the cart or orders it. The property requirements were already checked by exact rules; their results are listed under EXACT CHECKS and are final, so do not judge MET, NOT MET or NO EVIDENCE again. Judge ONLY these, from the records shown:
- The item is the requested kind of product: when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- Price times quantity is within the budget (equal is fine). If the records show it out of stock, it fails. When records disagree (for example on price or stock), the newest record counts.
- Each requirement marked UNCLEAR: decide from the records whether the item meets it (convert units when they differ; limits include the edge; a range counts only when the whole range meets the limit).
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a check above that fails]}. Use fits=false only for a check above that the records clearly fail; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is the value of one product property that the listings' details record, revealed only when a message asks about that property; a question about anything else gets a polite "don't mind" and is spent for nothing. Choose exactly QUESTION_COUNT different properties to ask about, most promising first. Never choose the brand, model, warranty, seller or condition, or anything about price, stock or delivery: these are never hidden needs. When the shopper did not take in a question, count that property as no need and choose a different one; never ask about the same property again in other words.
The request was written to hint at each hidden need without naming it: a story sentence that states no requirement points at one property through the part of the product it involves, the place or setting the product must suit, what it is used for or with, who uses it, or how it will be seen. Some story sentences are only colour (the shopper's job, mood or figures of speech) and point at nothing. Work in three steps:
1. list every story sentence of the request that states no requirement;
2. for each, write what the shopper would check on a listing to be satisfied, then name the property that records exactly that ("property") and the next most likely one ("second"), or "none". Prefer the property of the specific part or setting the sentence involves over a general property of the whole product: the part that hands, feet or body touch while using the product (comfort, grip, fit) points to that part's material or shape; a place the product must fit, stand or be used in points to the size, height or width that place limits; what it will be made into or used with points to the property naming that use or compatibility (such as type, style or form factor); the wearer's build or who it is for points to size class or gender; photographs, being seen or going with other things points to how the product looks (colour or pattern), even when the sentence names the body part that wears it. A sentence never points back at a property the request already settles;
3. choose: first anything under MUST BE LEARNED FROM THE SHOPPER that is not asked yet, then the hinted properties (strongest hint first, each sentence's property before its second guess; when the shopper did not mind a sentence's property, that sentence's second guess comes next), then, to fill the remaining questions, the properties a shopper of this product would most likely hold a fixed preference on. Only a few listings are opened, so a property whose values look alike among them can still be a hidden need (hidden needs often want an uncommon value); a property marked "differs among" the listings it names is about four times as likely to be a hidden need as one whose values look alike, but one whose values look alike can still be one. Hidden needs follow the kind of product: for a technical product, usually a core specification; for something people wear, handle or look at, usually how it looks, fits or is made; for something consumed, often how it is packed. Who it is for matters less often, and extras are rarely fixed. When a hint is weak, prefer those common kinds.
Use property names as the listings' details below show them (the plainest one when several name the same thing). A property that only the titles show is allowed when the titles show at least two different values of it: give it a plain name and add "values" with two or more of its values, each copied word for word from a different title. Another noun is allowed only for a clear hint that nothing listed or titled covers. Never choose a property the request already settles (its stated requirements, the price, stock, quantity or category), a property already asked, two names for one property, or a field that bundles several properties (such as variant labels mixing colour and size).
Return ONE JSON object and nothing else: {"hints": [{"sentence": "...", "check": "...", "property": "...", "second": "..."}], "questions": [{"property": "...", "question": "", "values": []}]}. Leave "values" empty for a property the listings' details show. Leave "question" empty for a product property, since that question is written for you; write "question" (one short plain sentence) only for an item under MUST BE LEARNED FROM THE SHOPPER that is not a product property, such as the budget."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. The listings below passed the exact property checks except for the requirements marked UNCLEAR. For each listing, decide from its record whether it meets its UNCLEAR requirements (convert units when they differ; limits include the edge), is not shown out of stock, and has a price within the budget.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

VALUES_PROMPT = """You check store listings against a shopper's requirements the way a strict store grades orders: by the terms a listing uses, not by loose similarity of meaning. Each numbered value below was found in a listing field named like the requirement's property, and it shares a word or word stem with the required value.
Answer true only when the value names the required thing itself in another form: another spelling or word form, an abbreviation, or the same thing with words added that keep it that thing (such as a blend that contains it). Answer false when the value names a different thing that merely shares a word or stem, such as an effect, a part, a related feature, a narrower or broader kind, or the opposite.
Return ONE JSON object and nothing else, with one entry per requirement label, for example {"R1": {"1": true, "2": false}}."""

VARIANTS_PROMPT = """Each field below comes from the property details of store listings, shown with values seen under it. Which fields hold variant labels, that is the option a listing is sold in, naming its colour, size, pack or version as a store shows a selected choice (such as "Black / XS" or "white")? A field named after one property of the product (such as a style, material, type or features field) is not a variant field, even when its values look alike.
Return ONE JSON object and nothing else: {"variant_fields": [field names exactly as shown]}."""

UNITS_PROMPT = """Values of one product property are written in different units. For each unit listed, give how many of the BASE unit one of that unit equals (for example, with BASE "gb", "tb" is 1000). Return ONE JSON object and nothing else: {"factors": {"unit": number}}, with null for a unit you cannot convert."""

MODES_PROMPT = """The store tool below ends a shopping task. Which of its argument values make it place the order, and which make it end without buying? Use only argument names and allowed values from its schema. Return ONE JSON object and nothing else: {"order_args": {...}, "decline_args": {...}}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _brief(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v, limit, depth + 1) for v in value]
    return value


_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
# A listing id read by its shape when no field is named like an id (one world named it "sku_code"): a short
# prefix, a dash and a long run of hex digits ("p-0098de9421fb").
_ID_SHAPE = re.compile(r"^[A-Za-z]{1,4}-[0-9a-f]{8,40}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


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
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
            if kind == "integer":
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
            if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                return int(number) if number.is_integer() else number
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema["items"], f"{path}[{index}]")
                fixed.append(item)
                problems += sub
            value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        fixed = {}
        for name, item in value.items():
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1)


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and ("id" in key.lower() and _ID_HINT.match(item) or _ID_SHAPE.match(item) and any(c.isdigit() for c in item)):
                out.add(item)
            else:
                _collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)


DECLINE_NOTE = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
_OPS = ("is", "not", "at_most", "at_least", "between")
_NUMBER = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z]*)")


def _words(value: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value if value is not None else "").lower())


def _spelling(word: str) -> str:
    return re.sub(r"our(s?)$", r"or\1", word) if len(word) > 4 else word


def _same_word(a: str, b: str) -> bool:
    a, b = _spelling(a), _spelling(b)
    return a == b or a + "s" == b or b + "s" == a or a + "es" == b or b + "es" == a


def _has_words(text: Any, wanted: list[str]) -> bool:
    have = _words(text)
    return bool(wanted) and all(any(_same_word(w, h) for h in have) for w in wanted)


def _has_phrase(text: Any, target: Any) -> bool:
    have, wanted = _words(text), _words(target)
    size = len(wanted)
    return size > 0 and any(all(_same_word(wanted[i], have[j + i]) for i in range(size)) for j in range(len(have) - size + 1))


def _says(text: Any, target: Any) -> bool:
    wanted = " ".join(_words(target))
    return bool(wanted) and f" {wanted} " in f" {' '.join(_words(text))} "


def _elements(value: Any) -> list[str]:
    return [part.strip() for part in re.split(r"[,;/|]", str(value)) if part.strip()]


def _states(value: Any, target: Any) -> bool:
    wanted = _words(target)
    if re.search(r"[,;/|]", str(target)):
        return _has_words(value, wanted)
    return any(_has_phrase(part, target) for part in _elements(value))


def _one_letter(a: str, b: str) -> bool:
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = sorted((a, b), key=len)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def _related(value: Any, target: Any, prop: Any) -> bool:
    named = _words(prop)
    have = _words(value)
    wanted = [w for w in _words(target) if len(w) >= 3 and not w.isdigit() and not any(_same_word(w, n) for n in named)][:1]
    return any(_same_word(w, h) or _one_letter(w, h) or (min(len(w), len(h)) >= 3 and (w.startswith(h) or h.startswith(w))) for w in wanted for h in have)


PLAN_TEMPERATURES = (0, 0.7, 0.7)
# Properties named in one message: lists while the shopper takes them (see LIST_START and LIST_MAX below), PER_MESSAGE
# after a reply asking for one thing at a time. Each task starts fresh, because nothing carries over between tasks.
# (The practice world never refuses a list, so practice runs can't show the refusals.)
PER_MESSAGE = 1
BUNDLE_SIZE = 3
# w13n: the planner ranks up to LIST_MAX properties once per product. Whatever the question limit, the first two
# questions name the top LIST_START properties still open, and from the third on, up to LIST_MAX led by the likeliest
# open property in turn (with three such questions, ranks 1, 4, 7, 10 first, then 2, 5, 8, 11, then 3, 6, 9, 12). A
# reply asking for one thing at a time turns lists off for the task. Worlds differ: in races that take lists a need was
# stated 79% of the time named first, 53% second, 38% third, 31% fourth and 24% later (races 2, 8, 14, 15, 17), but
# race 16 refused 13% of one-property questions, 43% of lists of 3-4 and 82% of lists of 9 or more. Modelled with
# those rates, averaged over races 1-18 (6 took lists, 12 refused at least sometimes), starting with lists of three
# reveals a ranked need 31 / 47 / 58 / 66 / 72% of the time for a limit of 1 / 2 / 3 / 4 / 5 questions, against
# 25 / 43 / 55 / 63 / 69% for one property at a time. In race 18 (2 questions, lists taken), miners asking lists in
# both questions revealed 54% of the needs, one property each 40%.
LIST_START = 3
LIST_MAX = 12


def _vote(samples: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Merge planner samples: non-property items from the first sample, then properties by summed rank points (1, 1/2, 1/3 ...)."""
    if not samples:
        return []
    extra = [q for q in samples[0].get("questions") or [] if isinstance(q, dict) and not _words(q.get("property")) and str(q.get("question") or "").strip()]
    points: dict[str, float] = {}
    values: dict[str, list[str]] = {}
    order: list[str] = []
    for sample in samples:
        items = [q for q in sample.get("questions") or [] if isinstance(q, dict) and _words(q.get("property"))]
        for rank, item in enumerate(items):
            name = str(item.get("property") or "").strip()
            key = next((k for k in order if len(_words(k)) == len(_words(name)) and _same_property(k, name)), None)
            if key is None:
                key = name
                order.append(key)
                points[key], values[key] = 0.0, []
            points[key] += 1 / (rank + 1)
            quoted = item.get("values") if isinstance(item.get("values"), list) else []
            values[key] += [str(v) for v in quoted if isinstance(v, (str, int, float)) and str(v) not in values[key]]
    ranked = sorted(order, key=lambda k: (-points[k], order.index(k)))
    return extra + [{"property": k, "question": "", "values": values[k]} for k in ranked]


_TITLE_KEY = re.compile(r"title|head|name|label")
# The store's own structure: the field naming a listing's category (the store's shelf), not a product property.
_SHELF_KEY = re.compile(r"categ|department|shelf|section|aisle")
# Properties that were never a hidden need in a race or in ORO Bench: a question about one is spent
# for nothing. Rechecked Nov 8 against all 1,509 known needs (races 1-17, ORO Bench, qualifying, practice): none is skipped;
# "sold by", "ships from" and "rank" were added then. A property is skipped when all its words are these (or filler such as "type").
NEVER_ASK = {"brand", "model", "warranty", "seller", "shop", "store", "condition", "sku", "price", "stock", "rating",
             "review", "shipping", "delivery", "manufacturer", "video", "url", "image", "link", "sold", "ships", "rank"}
FILLER_WORDS = {"type", "name", "number", "period", "policy", "info", "information", "classification", "status", "code", "id",
                "by", "from"}
VAGUE_WORDS = {"taste", "tastes", "preference", "preferences", "recipient", "like", "likes", "liking", "opinion", "feel", "vibe", "needs"}


def _row_title(row: dict[str, Any]) -> str | None:
    """A result row's product title, read by shape: of the string fields named like a title, the one with the
    most words (a product title is wordier than a shop, brand or category name), else the longest short
    multi-word string."""
    strings = [(str(k), " ".join(re.sub(r"<[^>]+>", " ", v).split())) for k, v in row.items() if isinstance(v, str)]
    named = [v for k, v in strings if _TITLE_KEY.search(k.lower()) and len(_words(v)) >= 2 and len(v) <= 300]
    if named:
        return max(named, key=lambda v: len(_words(v)))[:120]
    loose = [v for k, v in strings if not _SHELF_KEY.search(k.lower()) and len(_words(v)) >= 3 and len(v) <= 160]
    return max(loose, key=len)[:120] if loose else None


def _never_ask(prop: Any) -> bool:
    words = set(_words(prop))
    return bool(words & NEVER_ASK) and words <= NEVER_ASK | FILLER_WORDS


def _ask_text(prop: Any) -> str:
    return f"What is your preference for the {' '.join(_words(prop))}?"


def _ask_several(props: list[Any]) -> str:
    names = [" ".join(_words(p)) for p in props]
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " or " + names[-1]
    return f"Which {listed} would you like?"


# A question naming the hidden need was answered more often with two example values (82% against 76% of about 167,000
# such questions, races 1-15) and when short (81% at 12 words or fewer against 67%), so examples are added only while the
# question stays short.
EXAMPLE_WORDS = 14


def _example_ok(value: Any) -> bool:
    text = str(value).strip()
    return 0 < len(text) <= 24 and len(text.split()) <= 3 and not re.search(r"[,;/|]| or ", text) and not re.fullmatch(r"[A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*", text) or bool(re.fullmatch(r"\d+(\.\d+)?\s?[a-z]{1,4}", text))


def _with_examples(text: str, examples: list[str]) -> str:
    """The question with "such as A or B" before its question mark, or unchanged when that would make it long."""
    if len(examples) < 2 or not text.endswith("?"):
        return text
    longer = f"{text[:-1]}, such as {examples[0]} or {examples[1]}?"
    return longer if len(longer.split()) <= EXAMPLE_WORDS else text


def _same_property(a: Any, b: Any) -> bool:
    first, second = _words(a), _words(b)
    if not first or not second:
        return False
    small, large = sorted((first, second), key=len)
    return all(any(_same_word(w, h) for h in large) for w in small)


def _is_measure(target: Any) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*[a-z]*\s*", str(target).lower()))


def _same_measure(value: Any, target: Any, unit: str) -> bool:
    wanted = _measures(target)
    if len(wanted) != 1 or not _is_measure(target):
        return False
    number, want = wanted[0][0], wanted[0][1] or unit
    return any(n == number and (not want or not u or _same_unit(u, want)) for part in _elements(value) for n, u in _measures(part))


def _measures(value: Any) -> list[tuple[float, str]]:
    return [(float(number), unit) for number, unit in _NUMBER.findall(str(value).lower())]


def _same_unit(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return _same_word(a, b) or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)))


def _numeric(op: str, limits: list[float], unit: str, value: Any) -> str | None:
    found = _measures(value)
    if not found:
        return None
    units = [u for _, u in found if u]
    if unit and units and not _same_unit(units[-1], unit):
        return "unclear"
    low, high = min(n for n, _ in found), max(n for n, _ in found)
    if op == "at_most":
        met = high <= limits[0]
    elif op == "at_least":
        met = low >= limits[0]
    else:
        met = low >= min(limits[:2]) and high <= max(limits[:2])
    return "met" if met else "not met"


def _requirement(raw: Any, source: str) -> dict[str, Any] | None:
    if isinstance(raw, str):
        name, _, rest = raw.partition(":")
        raw = {"property": name, "op": "is", "values": [rest], "quote": raw} if rest.strip() else None
    if not isinstance(raw, dict):
        return None
    prop = " ".join(_words(raw.get("property")))
    op = str(raw.get("op") or "is").strip().lower().replace(" ", "_")
    op = op if op in _OPS else "is"
    values = raw.get("values") if isinstance(raw.get("values"), list) else [raw.get("values")]
    if op in ("at_most", "at_least", "between"):
        numbers = [n for v in values if not isinstance(v, bool) for n, _ in _measures(v)]
        values = numbers[:2] if op == "between" else numbers[:1]
        if len(values) < (2 if op == "between" else 1):
            return None
    else:
        values = [str(v).strip() for v in values if isinstance(v, (str, int, float)) and not isinstance(v, bool) and _words(v)]
    if not prop or not values:
        return None
    return {"property": prop, "op": op, "values": values, "unit": " ".join(_words(raw.get("unit"))), "blocking": raw.get("blocking") is not False, "source": source, "quote": str(raw.get("quote") or "")[:200]}


def _describe(req: dict[str, Any]) -> str:
    unit = f" {req['unit']}" if req["unit"] else ""
    if req["op"] in ("at_most", "at_least", "between"):
        numbers = [f"{v:g}" for v in req["values"]]
        limit = {"at_most": f"at most {numbers[0]}", "at_least": f"at least {numbers[0]}"}.get(req["op"]) or f"{numbers[0]} to {numbers[-1]}"
        return f"{req['property']}: {limit}{unit}"
    return f"{req['property']}: " + ("not " if req["op"] == "not" else "") + " or ".join(req["values"])


def _key_match(wanted: list[str], key: Any) -> int:
    have = _words(key)
    found = sum(1 for w in wanted if any(_same_word(w, h) or (len(w) >= 4 and h.startswith(w)) for h in have))
    return 2 if wanted and found == len(wanted) else (1 if found else 0)


def _family(req: dict[str, Any], facts: dict[str, Any], level: int = 2) -> dict[str, Any]:
    wanted = _words(req["property"])
    return {k: v for k, v in facts.items() if _key_match(wanted, k) == level}


def _wide_family(req: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    wanted = [w for w in _words(req["property"]) if len(w) >= 4] or _words(req["property"])
    return {k: v for k, v in facts.items() if any(_key_match([w], k) == 2 for w in wanted)}


def _evidence(req: dict[str, Any], facts: dict[str, Any], text: str, vocabulary: set[str], judged: dict[str, bool] | None = None, variants: set[str] | None = None) -> tuple[str, str]:
    judged = judged or {}
    family, near = _family(req, facts), _family(req, facts, 1)
    op, values = req["op"], req["values"]
    if op in ("at_most", "at_least", "between"):
        named = _words(req["property"])
        exact = {k: v for k, v in family.items() if len(_words(k)) == len(named) and all(_same_word(a, b) for a, b in zip(_words(k), named)) and _measures(v)}
        found = [(_numeric(op, values, req["unit"], part), f"{k}: {v}") for k, v in (exact or family).items() for part in _elements(v)]
        for state in ("met", "unclear", "not met"):
            hit = next((e for r, e in found if r == state), None)
            if hit:
                return state, hit
        near_hit = next((f"{k}: {v}" for k, v in near.items() if _measures(v)), None)
        return ("unclear", near_hit) if near_hit else ("no evidence", "")

    def literal(value: Any) -> bool:
        return any(_same_measure(value, t, req["unit"]) if _is_measure(t) else _states(value, t) for t in values)

    def in_text() -> bool:
        return any(_same_measure(text, t, req["unit"]) if _is_measure(t) else _says(text, t) for t in values)

    def same(value: Any) -> bool:
        return any(judged.get(part.lower()) for part in _elements(value))

    def open_question(value: Any) -> bool:
        return any(_related(value, t, req["property"]) for t in values) and not all(part.lower() in judged for part in _elements(value))

    hits = [(k, v, "") for k, v in family.items() if literal(v)] + [(k, v, " (same meaning)") for k, v in family.items() if not literal(v) and same(v)]
    labels = {k: v for k, v in facts.items() if " ".join(_words(k)) in (variants or set()) and k not in near}
    elsewhere = next((f"{k}: {v}" for k, v in labels.items() if literal(v)), None) if not family else None
    if op == "not":
        if hits:
            return "not met", f"{hits[0][0]}: {hits[0][1]}{hits[0][2]}"
        if not family and in_text():
            return "not met", "title or description"
        if elsewhere:
            return "not met", elsewhere
        return "met", ""
    if hits:
        named = _words(req["property"])
        exact = [(k, v, h) for k, v, h in hits if len(_words(k)) == len(named) and all(_same_word(a, b) for a, b in zip(_words(k), named))]
        key, value, how = (exact or hits)[0]
        others = [f"{k}: {v}" for k, v in _wide_family(req, facts).items() if not literal(v) and not same(v) and any(p.lower() in vocabulary for p in _elements(v))]
        return ("contradicted", f"{key}: {value}{how}; but {others[0]}") if others else ("met", f"{key}: {value}{how}")
    if family:
        pending = next(((k, v) for k, v in family.items() if open_question(v)), None)
        key, value = pending or next(iter(family.items()))
        return ("unclear" if pending else "not met"), f"{key}: {value}"
    if in_text():
        return "met", "title or description"
    if elsewhere:
        return "met", elsewhere
    near_hit = next((f"{k}: {v}" for k, v in near.items() if literal(v) or same(v)), None)
    return ("unclear", near_hit) if near_hit else ("no evidence", "")


def _quantity_from(value: Any, quote: Any, word: Any, source: Any) -> int | None:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 1:
        return None
    value = int(value)
    if value == 1:
        return 1
    quoted, spelled = _words(quote), _words(word)
    if not quoted or any(w not in set(_words(source)) for w in quoted):
        return None
    if len(spelled) != 1 or spelled[0] not in quoted or (spelled[0].isdigit() and int(spelled[0]) != value):
        return None
    return value


def _as_written(amount: Any, text: str) -> Any:
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return amount
    written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    if not written or float(amount) in written:
        return amount
    for factor in (100, 0.01):
        scaled = round(amount * factor, 6)
        if scaled in written:
            return int(scaled) if scaled == int(scaled) else scaled
    return amount


def _strings(value: Any, depth: int = 0) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if depth > 4:
        return []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return [s for item in items for s in _strings(item, depth + 1)]


def _unwrap(observation: Any) -> Any:
    if isinstance(observation, dict) and len(observation) == 2:
        inner = [v for v in observation.values() if isinstance(v, dict)]
        said = [v for v in observation.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
        if len(inner) == 1 and len(said) == 1:
            return inner[0]
    return observation


# Race 160/15 sends listings as one line of text: "ref_ref: p-0123abcd | title: A lamp | price: 900 | on_hand: True |
# category: Lamps | facts: color=silver; height=14 inch | item_blurb: ...". The field names change every race, so the line
# is read by its shape. A field is a name, a colon and a space after " | "; titles can hold " | " too.
_TEXT_FIELD = re.compile(r"(?:^|(?<= \| ))([A-Za-z_]\w*): ")
_TEXT_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def _text_pairs(text: str) -> dict[str, str] | None:
    if not re.match(r"^[^=;|]{1,60}=", text):
        return None
    pairs: dict[str, str] = {}
    last = None
    for part in text.split("; "):
        key, mark, value = part.partition("=")
        if mark and key.strip() and len(key) <= 60:
            last = key.strip()
            pairs[last] = value.strip()
        elif last:
            pairs[last] += "; " + part
    return {k: v for k, v in pairs.items() if v} or None


def _text_value(text: str) -> Any:
    text = text.strip()
    if text in ("", "None", "null"):
        return None
    if text in ("True", "true"):
        return True
    if text in ("False", "false"):
        return False
    if _TEXT_NUMBER.match(text):
        return float(text) if "." in text else int(text)
    return _text_pairs(text) or text


def _text_record(text: str) -> dict[str, Any] | None:
    if " | " not in text:
        return None
    marks = list(_TEXT_FIELD.finditer(text))

    def raw(mark: re.Match[str]) -> str:
        end = text.find(" | ", mark.end())
        return text[mark.end():end if end >= 0 else len(text)].strip()

    owner = next((m for m in marks if _ID_SHAPE.match(raw(m)) and any(c.isdigit() for c in raw(m))), None)
    if owner is None:
        return None
    capital = owner.group(1)[0].isupper()  # a title's "Model: X" is not a field when the fields are in lower case
    marks = [m for m in marks if m.group(1)[0].isupper() == capital]
    record: dict[str, Any] = {}
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() - 3 if index + 1 < len(marks) else len(text)
        record.setdefault(mark.group(1), _text_value(text[mark.end():end]))
    return record


def _decode(value: Any, depth: int = 0) -> Any:
    """A result with its listings as records, whichever way the world sent them."""
    if depth > 6:
        return value
    if isinstance(value, str):
        return _text_record(value) or value
    if isinstance(value, list):
        return [_decode(item, depth + 1) for item in value]
    if isinstance(value, dict):
        decoded = {k: _decode(v, depth + 1) for k, v in value.items()}
        if len(decoded) == 1:
            only = next(iter(value.values()))
            if isinstance(only, str) and isinstance(next(iter(decoded.values())), dict):  # {"value": "ref_ref: ..."}
                return next(iter(decoded.values()))
        return decoded
    return value


def _has_list(value: Any, depth: int = 0) -> bool:
    if isinstance(value, list):
        return True
    return isinstance(value, dict) and depth < 3 and any(_has_list(v, depth + 1) for v in value.values())


def _line_args(args: dict[str, Any]) -> dict[str, int] | None:
    """The lines a list-style cart call sets ({} empties the cart), or None for a cart that takes one listing."""
    for value in args.values():
        if isinstance(value, list) and all(isinstance(line, dict) for line in value):
            lines: dict[str, int] = {}
            for line in value:
                ident = next((v.strip() for v in line.values() if isinstance(v, str) and v.strip() and " " not in v.strip()), None)
                units = next((v for v in line.values() if isinstance(v, int) and not isinstance(v, bool)), 1)
                if ident:
                    lines[ident] = lines.get(ident, 0) + units
            return lines
    return None


def _line_shape(args: dict[str, Any]) -> tuple[str, str, str | None] | None:
    for key, value in args.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            ident = next((k for k, v in value[0].items() if isinstance(v, str)), None)
            units = next((k for k, v in value[0].items() if isinstance(v, int) and not isinstance(v, bool)), None)
            if ident:
                return key, ident, units
    return None


def _cart_accepted(observation: Any) -> bool:
    """Whether a cart call went through. Worlds answer with a yes/no flag, the cart, or the new line (an id and its
    count); a failure is an error field or a lone message."""
    # Shapes seen: {"accepted": true}; {"selection": [...]}; the new line as an id key plus a count key; {"error": ...};
    # a single message key and nothing else.
    top = _unwrap(observation)
    if not isinstance(top, dict) or not top or "error" in top:
        return False
    flags = [v for v in top.values() if isinstance(v, bool)]
    return all(flags) if flags else any(not isinstance(v, str) for v in top.values())


def _cart_lines(view: Any, known: set[str]) -> dict[str, int | None]:
    lines: dict[str, int | None] = {}

    def walk(value: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(value, dict):
            ids = {v for v in value.values() if isinstance(v, str) and v in known}
            if len(ids) == 1:
                numbers = sorted(v for v in value.values() if isinstance(v, int) and not isinstance(v, bool) and v > 0)
                unique = len(numbers) >= 2 and numbers[0] < numbers[1] and numbers[0] <= 99
                lines[ids.pop()] = numbers[0] if unique else None
                return
            for item in value.values():
                walk(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(view, 0)
    return lines


def _listing(record: Any, labels: set[str] | None = None) -> tuple[str, dict[str, Any], str] | None:
    record = _unwrap(record)
    if not isinstance(record, dict):
        return None
    facts = next((v for v in record.values() if isinstance(v, dict) and sum(isinstance(x, (str, int, float)) and not isinstance(x, bool) for x in v.values()) >= 2), None)
    item = _row_id(record)
    if facts is None or item is None:
        return None
    text = " ".join(v for v in record.values() if isinstance(v, str) and v != item and v.strip().lower() not in (labels or set()))
    return item, {str(k): v for k, v in facts.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}, text


def _row_id(row: dict[str, Any]) -> str | None:
    named = next((v for k, v in row.items() if isinstance(v, str) and "id" in str(k).lower() and _ID_HINT.match(v)), None)
    return named or next((v for v in row.values() if isinstance(v, str) and _ID_SHAPE.match(v) and any(c.isdigit() for c in v)), None)


def _shelf_key(rows: list[dict[str, Any]]) -> str | None:
    """The field naming the store's shelf on a page of results: one named like a shelf on most rows (a detail field
    whose name only looks like a shelf, on a few rows, doesn't count), else the short words-only field that most of the
    page's listings share (one world gave the shelf a name unlike any shelf word)."""
    if not rows:
        return None
    keys = Counter(k for row in rows for k in row)
    named = [k for k, n in keys.most_common() if _SHELF_KEY.search(str(k).lower()) and 2 * n >= len(rows)]
    if named or len(rows) < 3:
        return named[0] if named else None
    best, share = None, 0.6
    for k in keys:
        values = Counter(" ".join(v.split()) for row in rows for v in [row.get(k)] if isinstance(v, str) and v.strip())
        if not values or sum(values.values()) < 0.9 * len(rows) or _TITLE_KEY.search(str(k).lower()):
            continue
        top, n = values.most_common(1)[0]
        if n / len(rows) >= share and len(top) >= 4 and len(_words(top)) <= 5 and top[:1].isupper() and not re.search(r"\d", top):
            best, share = k, n / len(rows)
    return best


def _rows(observation: Any) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(value, list):
            for item in value:
                ident = _row_id(item) if isinstance(item, dict) else None
                if ident:
                    found.append((ident, item))
                else:
                    walk(item, depth + 1)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)

    walk(_unwrap(observation), 0)
    return found


def _price_stock(row: dict[str, Any]) -> tuple[float | None, bool | None]:
    numbers = [v for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    flags = [v for v in row.values() if isinstance(v, bool)]
    return (float(numbers[0]) if len(numbers) == 1 else None), (flags[0] if len(flags) == 1 else None)


class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.messages_used = 0
        self.message_budget: int | None = None
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checked = False
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.reqs: list[dict[str, Any]] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.menu_order: list[str] = []
        self.menu_names: list[str] = []
        self.splitting: set[str] = set()
        self.variant_names: dict[str, bool] = {}
        self.relaxed: dict[str, bool] = {}
        self.unit_factors: dict[str, dict[str, float | None]] = {}
        self.labels_cache: tuple[int, set[str]] = (-1, set())
        self.variant_failures = 0
        self.last_cart_view: Any = None
        self.cart_changed_at = 0
        self.cart_view_at = -1
        self.cart_state: dict[str, str] = {}
        self.line_shape: tuple[str, str, str | None] | None = None
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.planner_on = True
        self.audited = False
        self.qty_warned: set[str] = set()
        self.quantity = 1
        self.items: list[dict[str, Any]] = []  # several products in one order: each with its own state
        self.focus = 0
        self.menu_values: dict[str, Counter] = {}
        self.budget_scope = "total"
        self.held: dict[str, int | None] = {}
        self.cart_version = 0
        self.view_version = -1
        self.view_lines: dict[str, int | None] = {}
        self.qty_blocked = -1
        self.better_swapped = -1
        self.order_args: dict[str, Any] | None = {}
        self.decline_args: dict[str, Any] | None = {}
        self.mode_asked = False
        self.judged: dict[str, dict[str, bool]] = {}
        self.catalog: dict[str, dict[str, Any]] = {}
        self.shelf_field: str | None = None  # the shelf's field, from a page of results, for the listing pages too
        self.searches: dict[str, dict[str, Any]] = {}
        self.board_said = ""
        self.cheaper_asked: int | None = None
        self.best: dict[str, str] | None = None
        self.cart_modes: dict[str, dict[str, Any]] = {"add": {}, "remove": {}}
        self.decline_blocks = 0
        self.commit_wanted = False
        self.decline_hinted = False
        self.commit_tries = 0
        self.commit_failures = 0
        self.committing = False
        self.last_ask_step = -99
        self.last_ask_facts = 0
        self.ask_due = False
        self.bundling: bool | None = None  # decided at the first question; off once the shopper asks for one at a time
        self.bundle_refused = False
        # Decoys: a shopper answers a list question about one property only, so a property it does not mind (or a decoy)
        # named first can hide the ones after it. After such a list, only its first property counts as asked and the rest
        # lead the next question in reverse order; after a revealed need, the properties named after it lead (`lead`).
        self.lead: list[str] = []
        self.list_misses = 0
        self.list_plan: list[str] = []
        self.list_rounds = 0
        self.list_sent = 0
        self.list_stage = 0
        self.list_order: list[str] = []
        self.cutoffs = 0  # answers cut off at max_tokens (counted for the run's notes)
        self.need_found = False
        self.pending_reply = ""
        self.verdicts: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.cost = 0.0
        self.cost_lock = threading.Lock()
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    # What belongs to one product of a several-product order: kept per item and swapped in when the agent
    # moves to that item, so w12's one-product checks work unchanged on the product in hand.
    ITEM_STATE = {
        "free": list, "dropped": list, "records": list, "menu_order": list, "menu_names": list, "splitting": set, "relaxed": dict,
        "labels_cache": lambda: (-1, set()), "catalog": dict, "searches": dict, "board_said": str, "cheaper_asked": lambda: None,
        "asked": list, "planner_on": lambda: True, "audited": lambda: False, "last_ask_step": lambda: -99, "last_ask_facts": int,
        "ask_due": lambda: False, "decline_blocks": int, "commit_wanted": lambda: False,
        "lead": list, "list_misses": int, "list_plan": list, "list_rounds": int, "list_sent": int, "list_stage": int, "list_order": list,
        "need_found": lambda: False,
    }

    def multi(self) -> bool:
        return len(self.items) > 1

    def switch_item(self, index: int) -> None:
        """Keep the product in hand's state with it and take up another product's (fresh the first time)."""
        current = self.items[self.focus]
        current["state"] = {name: getattr(self, name) for name in self.ITEM_STATE}
        current["reqs"], current["quantity"] = self.reqs, self.quantity
        target = self.items[index]
        state = target.get("state") or {name: make() for name, make in self.ITEM_STATE.items()}
        for name, value in state.items():
            setattr(self, name, value)
        self.reqs, self.quantity, self.focus = target["reqs"], target["quantity"], index
        target.setdefault("messages_from", self.messages_used)
        self.plan["product"] = target["product"]
        self.ledger_version += 1
        self.trace.append({"kind": "switch_item", "item": index, "product": target["product"]})

    def item_done(self, line: str) -> str:
        """The product in hand has its cart line: note it and move to the next product without one."""
        item = self.items[self.focus]
        item["line"] = line
        item["line_price"] = (self.catalog.get(line) or {}).get("price")
        left = [k for k, other in enumerate(self.items) if not other.get("line")]
        if not left:
            return f"Every product now has its cart line ({', '.join(i['product'] for i in self.items)}): view the cart to check each line and its quantity, then order."
        self.switch_item(left[0])
        reqs = "; ".join(_describe(r) for r in self.reqs) or "no stated requirement"
        return (f"{item['product']} is in the cart ({line}). Next: {self.item_label()}; its requirements: {reqs}. "
                "Search for it in its own category and open its listings; its questions to the shopper are sent for you.")

    def item_removed(self, line: str) -> str:
        """A product's line left the cart (taken out, or sold out): that product is in hand again."""
        index = next((k for k, item in enumerate(self.items) if item.get("line") == line), None)
        if index is None:
            return ""
        self.items[index]["line"] = None
        self.items[index]["line_price"] = None
        if index != self.focus:
            self.switch_item(index)
        return f"{self.items[index]['product']} has no cart line now. Work on {self.item_label()} again: put a fitting listing for it in the cart."

    def item_label(self) -> str:
        item = self.items[self.focus]
        return f"item {self.focus + 1} of {len(self.items)}: {item['product']} (quantity {item['quantity']})"

    def item_budget(self) -> float | None:
        """The price cap for the product in hand: with a cap for the whole order, what the products already
        in the cart leave of it."""
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        if budget is None or not self.multi() or self.budget_scope == "each":
            return budget
        spent = sum((item.get("line_price") or 0) * (item.get("quantity") or 1) for k, item in enumerate(self.items) if k != self.focus and item.get("line"))
        return budget - spent

    def item_messages_left(self, limit: int) -> int:
        """Messages the product in hand may still use: an even share of what is left for the products not yet done."""
        if not self.multi():
            return limit - self.messages_used
        start = self.items[self.focus].get("messages_from", self.messages_used)
        remaining_items = sum(1 for k, item in enumerate(self.items) if k >= self.focus and not item.get("line"))
        share = max(1, -(-(limit - start) // max(1, remaining_items)))  # rounded up: earlier products ask first
        return min(limit - self.messages_used, start + share - self.messages_used)

    def time_left(self) -> float:
        left = TASK_SECONDS - (time.monotonic() - self.started)
        # Spending credit makes the task behave as if time were short, so every wrap-up rule applies.
        if self.cost >= CREDIT_FINISH:
            return min(left, FORCE_SECONDS - 1)
        if self.cost >= CREDIT_WRAP_UP:
            return min(left, WRAP_UP_SECONDS - 1)
        return left

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, temperature: float = 0) -> dict[str, Any] | None:
        if self.cost >= CREDIT_CHEAP:
            models = list(dict.fromkeys(MAIN_MODEL if "sonnet" in m else m for m in models))
        attempts = list(models)
        index = 0
        while index < len(attempts):
            model = attempts[index]
            index += 1
            left = self.time_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": temperature, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)) and usage["cost"] > 0:
                    with self.cost_lock:
                        before = self.cost
                        self.cost += float(usage["cost"])
                    for mark in (CREDIT_CHEAP, CREDIT_WRAP_UP, CREDIT_FINISH):
                        if before < mark <= self.cost:
                            self.trace.append({"kind": "credit_guard", "credit": round(self.cost, 4), "mark": mark, "model": model})
                message = data["choices"][0].get("message") or {}
                if data["choices"][0].get("finish_reason") == "length":
                    # Cut off at max_tokens (often by the model's own thinking): the answer is incomplete, so it counts as a
                    # failure; the next try gets twice the limit, and the last model gets one more try if none is left.
                    details = usage.get("completion_tokens_details") or {}
                    self.cutoffs += 1
                    self.trace.append({"kind": "cutoff", "model": model, "max_tokens": max_tokens, "reasoning": details.get("reasoning_tokens"),
                                       "visible": len(str(message.get("content") or "")), "tool_calls": bool(message.get("tool_calls"))})
                    if max_tokens < CUTOFF_TOKENS_MAX:
                        max_tokens = min(CUTOFF_TOKENS_MAX, max_tokens * 2)
                        if index == len(attempts):
                            attempts.append(model)
                    continue
                if message.get("content") or message.get("tool_calls"):
                    return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data:
                    data["observation"] = _decode(data["observation"])
                if isinstance(data.get("observations"), list):
                    data["observations"] = [_decode(o) for o in data["observations"]]
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if data.get("finished"):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                return None
        return None

    def setup(self) -> None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        static = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        variable = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            variable += "\n\nOPENING OBSERVATION:\n" + _compact(self.policy.get("opening"))
        content = [{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable}]
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 3000)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        if not plan and self.time_left() > TASK_SECONDS - 90:
            content[1]["text"] += "\n\nKeep every quote short (at most 12 words) so the whole JSON object fits."
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 4000)
            plan = _parse_json_object((message or {}).get("content") or "") or {}
            self.trace.append({"kind": "setup_retry", "parsed": bool(plan)})
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        self.order_args, self.decline_args = self.finish_modes(plan)
        measure, direction = " ".join(_words(plan.get("priority_property"))), str(plan.get("priority_direction") or "").strip().lower()
        self.best = {"property": measure, "direction": direction} if plan.get("priority") == "best" and measure and direction in ("least", "most") else None
        cart_args = plan.get("cart_args") if isinstance(plan.get("cart_args"), dict) else {}
        self.cart_modes = {k: self.mode_values(self.roles.get("cart"), cart_args.get(k)) for k in ("add", "remove")}
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        self.reqs = [r for r in (_requirement(raw, "stated") for raw in plan.get("requirements") or []) if r]
        self.quantity = _quantity_from(plan.get("quantity"), plan.get("quantity_quote"), plan.get("quantity_word"), self.policy.get("query")) or 1
        if plan.get("quantity") not in (None, 1, self.quantity):
            self.trace.append({"kind": "quantity", "proposed": plan.get("quantity"), "quote": plan.get("quantity_quote"), "kept": self.quantity})
        self.budget_now = _as_written(plan.get("price_cap"), str(self.policy.get("query") or ""))
        self.budget_scope = "each" if plan.get("budget_scope") == "each" else "total"
        raw_items = [i for i in plan.get("items") or [] if isinstance(i, dict) and str(i.get("product") or "").strip()]
        if len(raw_items) > 1:
            query = str(self.policy.get("query") or "")
            for raw in raw_items:
                reqs = [r for r in (_requirement(x, "stated") for x in raw.get("requirements") or []) if r]
                quantity = _quantity_from(raw.get("quantity"), raw.get("quantity_quote"), raw.get("quantity_word"), query) or 1
                self.items.append({"product": str(raw["product"]).strip(), "quantity": quantity, "reqs": reqs, "line": None})
            self.reqs, self.quantity, self.focus = self.items[0]["reqs"], self.items[0]["quantity"], 0
            self.items[0]["messages_from"] = 0
            self.plan["product"] = self.items[0]["product"]
            self.trace.append({"kind": "items", "items": [{"product": i["product"], "quantity": i["quantity"], "requirements": len(i["reqs"])} for i in self.items],
                               "budget_scope": self.budget_scope})
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)

    def mode_values(self, name: str | None, proposed: Any) -> dict[str, Any]:
        properties = (self.schemas.get(name or "") or {}).get("properties") or {}
        out: dict[str, Any] = {}
        for key, value in (proposed if isinstance(proposed, dict) else {}).items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue
            if "enum" in spec:
                match = next((option for option in spec["enum"] if str(option).lower() == str(value).strip().lower()), None)
                if match is not None:
                    out[key] = match
            elif spec.get("type") == "boolean" and isinstance(value, bool):
                out[key] = value
        return out

    def ask_modes(self, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        tool = next((t["function"] for t in self.tools if t["function"]["name"] == name), None)
        if tool is None:
            return None
        result = self.judge(MODES_PROMPT + "\n\nTOOL:\n" + _compact(tool), 300, [READ_MODEL, JUDGE_MODEL]) or {}
        order_args, decline_args = self.mode_values(name, result.get("order_args")), self.mode_values(name, result.get("decline_args"))
        if any(key in decline_args and decline_args[key] != value for key, value in order_args.items()):
            return order_args, decline_args
        return None

    def finish_modes(self, plan: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        order, decline = self.roles.get("order"), self.roles.get("decline")
        order_args = self.mode_values(order, plan.get("order_args"))
        decline_args = self.mode_values(decline, plan.get("decline_args"))
        if not order or order != decline:
            return order_args, decline_args
        if any(key in decline_args and decline_args[key] != value for key, value in order_args.items()):
            return order_args, decline_args
        asked = self.ask_modes(order)
        self.trace.append({"kind": "finish_modes", "source": "asked" if asked else "unknown", "modes": asked})
        return asked if asked else (None, None)

    def finish_kind(self, name: str | None, args: dict[str, Any]) -> str | None:
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if not name or name not in (order, decline):
            return None
        if order != decline:
            return "order" if name == order else "decline"
        for kind, mode in (("order", self.order_args), ("decline", self.decline_args)):
            if mode and all(args.get(key) == value for key, value in mode.items()):
                return kind
        return "unknown"

    def how(self, kind: str) -> str:
        name = self.roles.get(kind)
        mode = self.order_args if kind == "order" else self.decline_args
        return f"{name} with {_compact(mode)}" if name and mode else (name or f"the {kind} tool")

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if self.cart_line_spec():
            lines.append("- The cart tool sets the whole cart at each call: send every line you want to keep, not only the new one.")
        if self.multi():
            lines.append("- SEVERAL PRODUCTS, one cart line each (one at a time, in this order; you start with item 1):")
            for k, item in enumerate(self.items):
                reqs = "; ".join(_describe(r) for r in item["reqs"]) or "no stated requirement"
                lines.append(f"  item {k + 1}: {item['product']}, exactly {item['quantity']} unit(s) of one listing; {reqs}")
            if self.budget_now is not None:
                lines.append(f"- the price cap is for {'each product' if self.budget_scope == 'each' else 'the whole order (all lines together)'}")
        else:
            if plan.get("product"):
                lines.append(f"- product: {plan.get('product')}")
            lines.append(f"- quantity: exactly {self.quantity} unit(s) of one listing")
            for req in self.reqs:
                lines.append(f"- stated requirement: {_describe(req)}" + ("" if req["blocking"] else " (no listing can show this: not blocking)"))
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for item in plan.get("ask_first") or []:
            lines.append(f"- must be learned from the shopper first: {item}")
        for rule in plan.get("rules") or []:
            lines.append(f"- rule: {rule}")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.max_steps}")
        if self.roles.get("order") and self.roles.get("order") == self.roles.get("decline") and self.order_args and self.decline_args:
            lines.append(f"- finishing: {self.how('order')} places the order; {self.how('decline')} declines. Always set it explicitly.")
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        if self.multi():
            done = [f"{item['product']} ({item['line']})" for item in self.items if item.get("line")]
            text += f"\n[item] working on {self.item_label()}" + (f"; already in the cart: {', '.join(done)}" if done else "")
        text += "\n[requirements now] " + ("for this item: " if self.multi() else "") + self.ledger()
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += (" FINISH NOW: make sure the cart holds one verified line for each product, then order (or decline if a product has nothing that fits)."
                     if self.multi() else " FINISH NOW: fix the cart to one verified line, re-check its stock, then order (or decline if nothing fits).")
        elif self.order_attempts == 0 and (ready := self.safe_ready()) and left > COMMIT_STEPS + 8 and not self.safe_backups(ready):
            text += f" READY: {ready} is your choice, but no other opened listing is a verified fit yet, and the first pick often sells out at checkout: first open the details of one or two more candidates" + (" (the next cheapest ones)" if self.plan.get("priority") == "cheapest" else "") + f" so a backup is ready, then put {ready} in the cart and order it."
        elif self.order_attempts == 0 and ready:
            text += f" READY: {ready} is your choice; put it in the cart (if it is not there yet) and order it now. Every pending change shows up at the first order attempt, and a refused order can still be fixed."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        items = []
        for req in self.reqs:
            tag = "stated" if req["source"] == "stated" else "revealed by the shopper"
            items.append(f"{_describe(req)} ({tag}{'' if req['blocking'] else '; no listing can show this: not blocking'})")
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            if self.multi():
                text += f"; this item's price times quantity at most {self.item_budget()} {self.plan.get('currency') or ''}".rstrip()
            else:
                text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        text += f"; quantity exactly {self.quantity}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None, temperature: float = 0) -> dict[str, Any] | None:
        message = self.infer([{"role": "user", "content": prompt}], None, models or [JUDGE_MODEL, MAIN_MODEL], max_tokens, temperature) if temperature else self.infer([{"role": "user", "content": prompt}], None, models or [JUDGE_MODEL, MAIN_MODEL], max_tokens)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            return ""
        if question and result.get("one_at_a_time") is True and not result.get("needs"):
            self.bundling, self.bundle_refused = False, True
            self.trace.append({"kind": "one_at_a_time", "question": question[:120]})
            return "Meaning: the shopper wants one question at a time and did not answer this one. Its properties are still open."
        if question and result.get("not_understood") is True and not result.get("needs"):
            # Rewording never got an answer in a race (0 of 40); asking about another property did.
            self.trace.append({"kind": "not_understood", "question": question[:120]})
            return ("Meaning: the shopper did not take in the question. Count it as no need on that property "
                    "and ask about a different property next; do not reword it.")
        notes = []
        for drop in (result.get("dropped") or []) if question is None else []:
            removed = self.drop_requirement(drop)
            if removed:
                self.dropped.append(removed)
                notes.append(f"dropped: {removed}")
        for raw in result.get("needs") or []:
            req = _requirement(raw, "revealed")
            if req:
                self.set_requirement(req)
                notes.append(f"new requirement: {_describe(req)}")
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        budget = _as_written(result.get("new_budget"), "\n".join(fresh))
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        quantity = _quantity_from(result.get("new_quantity"), result.get("quantity_quote"), result.get("quantity_word"), "\n".join(fresh))
        if quantity and quantity != self.quantity:
            self.quantity = quantity
            notes.append(f"new quantity: exactly {quantity} unit(s) of one listing")
        first = self.cart_ids[0] if self.cart_ids else None
        if result.get("unavailable"):
            if first:
                self.catalog.setdefault(first, {"price": None, "stock": None, "text": ""})["stock"] = False
            notes.append(f"{first or 'an item'} became unavailable; remove it from the cart and do not order it")
        if result.get("price_changed") and first:
            self.catalog.setdefault(first, {"price": None, "stock": None, "text": ""})["stale"] = True
            notes.append(f"the price of {first} changed: re-read its price before ordering")
        if notes:
            self.ledger_version += 1
            recheck = self.recheck()
            if recheck:
                notes.append(recheck)
            board = self.safe_board()
            if board:
                notes.append(board)
        reply = str(result.get("reply") or "").strip()
        if reply and self.roles.get("message") and any("?" in line for line in fresh):
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def set_requirement(self, req: dict[str, Any]) -> None:
        self.reqs = [r for r in self.reqs if r["property"] != req["property"]] + [req]

    def drop_requirement(self, name: Any, keep_stated: bool = False) -> str:
        words = set(_words(str(name).partition(":")[0]))
        for req in self.reqs:
            have = set(_words(req["property"]))
            if keep_stated and req["source"] == "stated":
                continue
            if words and have and (have <= words or words <= have):
                self.reqs.remove(req)
                return _describe(req)
        return ""

    def label_values(self) -> set[str]:
        if self.labels_cache[0] == len(self.records):
            return self.labels_cache[1]
        labels = {str(v).strip().lower() for spec in self.schemas.values() for prop in ((spec or {}).get("properties") or {}).values() if isinstance(prop, dict) for v in prop.get("enum") or [] if isinstance(v, str)}
        seen: dict[tuple[str, str], set[str]] = {}
        for record in self.records:
            listing = _listing(record)
            top = _unwrap(record)
            if not listing or not isinstance(top, dict):
                continue
            for key, value in top.items():
                if isinstance(value, str) and value != listing[0] and 0 < len(value.split()) <= 4:
                    seen.setdefault((str(key), value.strip().lower()), set()).add(listing[0])
        labels |= {value for (_, value), items in seen.items() if len(items) >= 2}
        self.labels_cache = (len(self.records), labels)
        return labels

    def listings(self) -> dict[str, tuple[dict[str, Any], str]]:
        found: dict[str, tuple[dict[str, Any], str]] = {}
        labels = self.label_values()
        for record in self.records:
            listing = _listing(record, labels)
            if listing:
                found[listing[0]] = (listing[1], listing[2])
        for req in self.reqs:
            if not req["blocking"] and any(_family(req, facts) for facts, _ in found.values()):
                req["blocking"] = True
                self.trace.append({"kind": "grounded", "requirement": _describe(req)})
        return found

    def vocabulary(self, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> set[str]:
        if req["op"] != "is":
            return set()
        keys = {" ".join(_words(k)) for facts, _ in listings.values() for k, v in _wide_family(req, facts).items() if any(_states(v, t) for t in req["values"])}
        return {p.lower() for facts, _ in listings.values() for k, v in facts.items() if " ".join(_words(k)) in keys for p in _elements(v)}

    def evidence(self, req: dict[str, Any], item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[str, str]:
        facts, text = listings[item]
        shelf = self.catalog.get(item, {}).get("category")
        if req["op"] == "is" and req["source"] == "stated" and shelf and any(not _is_measure(t) and _states(shelf, t) for t in req["values"]):
            return "met", f"category: {shelf}"
        return _evidence(req, facts, text, self.vocabulary(req, listings), self.judged.get(_describe(req), {}), {n for n, v in self.variant_names.items() if v})

    def variant_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        names: list[str] = []
        for req in [r for r in self.reqs if r["blocking"] and r["op"] in ("is", "not")]:
            for item in items:
                if item not in listings or _family(req, listings[item][0]):
                    continue
                near = _family(req, listings[item][0], 1)
                for key, value in listings[item][0].items():
                    name = " ".join(_words(key))
                    if name and key not in near and name not in self.variant_names and name not in names and any(_states(value, t) for t in req["values"]):
                        names.append(name)
        return names[:20]

    def classify_variants(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        names = self.variant_asks(items, listings)
        if not names:
            return
        samples: dict[str, list[str]] = {n: [] for n in names}
        for facts, _ in listings.values():
            for key, value in facts.items():
                bucket = samples.get(" ".join(_words(key)))
                if bucket is not None and str(value)[:60] not in bucket and len(bucket) < 4:
                    bucket.append(str(value)[:60])
        prompt = VARIANTS_PROMPT + "\n\nFIELDS (name: values seen):\n" + "\n".join(f"- {n}: " + " | ".join(samples[n]) for n in names)
        result = self.judge(prompt, 300, [JUDGE_MODEL, READ_MODEL])
        if not isinstance(result, dict) and self.variant_failures < 1:
            self.variant_failures += 1
            return
        chosen = {" ".join(_words(x)) for x in (result or {}).get("variant_fields") or [] if isinstance(x, str)}
        for name in names:
            self.variant_names[name] = name in chosen
        self.trace.append({"kind": "variants", "asked": names, "variant": sorted(chosen & set(names))})

    def fails_elsewhere(self, item: str, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        for other in self.reqs:
            if other is req or not other["blocking"]:
                continue
            state, _ = self.evidence(other, item, listings)
            if state in ("not met", "no evidence"):
                return True
        return False

    def values_prompt(self, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> str:
        lines = []
        for number, (req, values) in enumerate(asks, 1):
            lines.append(f"R{number}. property: {req['property']}; required value: {' or '.join(str(v) for v in req['values'])}")
            lines += [f"  {index}. {value}   (field: {field})" for index, (value, field) in enumerate(values, 1)]
        return VALUES_PROMPT + "\n\n" + "\n".join(lines)

    def read_values(self, result: Any, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> None:
        for number, (req, values) in enumerate(asks, 1):
            answers = (result or {}).get(f"R{number}") if isinstance(result, dict) else None
            if not isinstance(answers, dict):
                continue
            cache = self.judged.setdefault(_describe(req), {})
            for index, (value, _) in enumerate(values, 1):
                if isinstance(answers.get(str(index)), bool):
                    cache[value] = answers[str(index)]

    def value_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], list[tuple[str, str]]]]:
        asks = []
        for req in [r for r in self.reqs if r["blocking"] and r["op"] in ("is", "not")]:
            cache = self.judged.get(_describe(req), {})
            values: list[tuple[str, str]] = []
            for item in items:
                if item not in listings or self.fails_elsewhere(item, req, listings):
                    continue
                state = self.evidence(req, item, listings)[0]
                if (req["op"] == "is" and state in ("met", "contradicted")) or (req["op"] == "not" and state == "not met"):
                    continue
                family = _family(req, listings[item][0])
                for key, value in family.items():
                    if any(_states(value, t) for t in req["values"]):
                        continue
                    for part in _elements(value):
                        name = part.lower()
                        if name not in cache and all(name != v for v, _ in values) and any(_related(part, t, req["property"]) for t in req["values"]):
                            values.append((name, key))
            if values:
                asks.append((req, values[:40]))
        return asks

    def judge_values(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        if self.time_left() < WRAP_UP_SECONDS + 30:
            return
        self.classify_variants(items, listings)
        asks = self.value_asks(items, listings)
        if not asks:
            return
        result = self.judge(self.values_prompt(asks), 1500, [VALUE_MODEL, JUDGE_MODEL])
        self.read_values(result, asks)
        self.trace.append({"kind": "values", "asked": sum(len(v) for _, v in asks), "answered": bool(result)})

    def assess(self, item: str, listings: dict[str, tuple[dict[str, Any], str]] | None = None, judge: bool = True) -> dict[str, Any] | None:
        blocking = [r for r in self.reqs if r["blocking"]]
        listings = self.listings() if listings is None else listings
        if not blocking or item not in listings:
            return None
        if judge:
            self.judge_values([item], listings)
        results = [(r, *self.evidence(r, item, listings)) for r in blocking]
        facts = listings[item][0]

        def loose(r: dict[str, Any], state: str) -> bool:
            if _describe(r) not in self.relaxed:
                return False
            return state == "not met" or (state == "no evidence" and (not self.relaxed[_describe(r)] or bool(_family(r, facts) or _family(r, facts, 1))))
        results = [(r, "unclear", "no opened listing states it in these words; decide from the details") if loose(r, state) else (r, state, evidence) for r, state, evidence in results]
        fails = [x for x in results if x[1] in ("not met", "no evidence")]
        unclear = [x for x in results if x[1] == "unclear"]
        return {"state": "fails" if fails else ("unclear" if unclear else "fits"), "results": results, "fails": fails, "contradicted": any(x[1] == "contradicted" for x in results)}

    def check_lines(self, assessment: dict[str, Any]) -> list[str]:
        return [f"{_describe(r)}: {state.upper()}" + (f" ({evidence[:90]})" if evidence else "") for r, state, evidence in assessment["results"]]

    def check_note(self, item: str) -> str:
        assessment = self.assess(item)
        if not assessment:
            return ""
        verdict = {"fits": "fits every requirement", "fails": "does not fit", "unclear": "fits unless an UNCLEAR item fails; decide it from the details"}[assessment["state"]]
        if assessment["state"] == "fits" and assessment["contradicted"]:
            verdict += " (its evidence is contradicted: it still counts under the cheapest or best rule, so follow the [candidates] verdict)" if self.plan.get("priority") in ("cheapest", "best") else " (its evidence is contradicted: prefer a clean listing when one exists)"
        return f"[check {item}] " + " | ".join(self.check_lines(assessment)) + f" => {verdict}"

    def recheck(self) -> str:
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: (self.assess(item, listings, judge=False) or {}).get("state") for item in listings}
        if not any(states.values()):
            return ""
        fits = [i for i, s in states.items() if s == "fits"]
        unclear = [i for i, s in states.items() if s == "unclear"]
        fails = sum(1 for s in states.values() if s == "fails")
        return "opened listings that fit the requirements now: " + (", ".join(fits[:8]) or "none") + (f"; unclear: {', '.join(unclear[:5])}" if unclear else "") + f"; not fitting: {fails}"

    def observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        top = _unwrap(observation)
        rows = _rows(top)
        single = _listing(top)
        if len(rows) >= 3:
            self.shelf_field = self.shelf_field or _shelf_key([row for _, row in rows])  # one stray page can't move it
        for item, row in rows + ([(single[0], top)] if single else []):
            entry = self.catalog.setdefault(item, {"price": None, "stock": None, "text": ""})
            entry.setdefault("labels", set()).update(v.strip().lower() for v in row.values() if isinstance(v, str) and v != item and 0 < len(v.strip()) <= 40)
            shelf = row.get(self.shelf_field) if self.shelf_field else None
            if not isinstance(shelf, str) or not 0 < len(shelf.strip()) <= 60:
                shelf = next((v for k, v in row.items() if isinstance(v, str) and 0 < len(v.strip()) <= 60 and _SHELF_KEY.search(str(k).lower())), None)
            title = _row_title(row)
            if title:
                entry["title"] = title
            if shelf:
                entry["category"] = shelf.strip().lower()
            price, stock = _price_stock(row)
            if price is not None:
                entry["price"], entry["stale"] = price, False
            if stock is not None:
                entry["stock"] = stock
        if not rows or name in {self.roles.get(r) for r in ("cart", "cart_view", "message", "order", "decline")}:
            return False
        props = (self.schemas.get(name) or {}).get("properties") or {}
        paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec}
        echoes = {v.strip().lower() for _, row in rows for v in row.values() if isinstance(v, str)}
        filters = {k for k, v in args.items() if k not in paging and isinstance(v, str) and v.strip().lower() in echoes}
        text = " ".join(str(v) for k, v in args.items() if k not in paging and k not in filters and isinstance(v, str) and "enum" not in (props.get(k) or {}))
        key = name + _compact({k: v for k, v in args.items() if k not in paging})
        query = self.searches.setdefault(key, {"text": text, "words": _words(text), "seen": [], "totals": [], "offsets": set(), "page_size": 0, "ended": False})
        query["seen"] += [item for item, _ in rows if item not in query["seen"]]
        query["offsets"].add(_compact({k: v for k, v in args.items() if k in paging}))
        numbers = [v for v in top.values() if isinstance(v, int) and not isinstance(v, bool)] if isinstance(top, dict) else []
        query["totals"].append(max(numbers) if numbers and max(numbers) >= len(rows) else None)
        query["ended"] = query["ended"] or (isinstance(top, dict) and any(v is None for v in top.values())) or (0 < len(rows) < query["page_size"])
        query["page_size"] = max(query["page_size"], len(rows))
        return True

    def complete(self, query: dict[str, Any]) -> bool:
        totals = query["totals"]
        if not totals or None in totals or len(set(totals)) != 1:
            return False
        return len(query["seen"]) == totals[0] and (len(query["offsets"]) >= 2 or query["ended"])

    def covering(self, query: dict[str, Any]) -> bool:
        choices = [r["values"] for r in self.reqs if r["blocking"] and r["op"] == "is"]
        if not query["words"] or not choices:
            return False
        for combo in itertools.product(*choices):
            allowed = [w for value in combo for w in _words(value)]
            if all(any(_same_word(w, a) for a in allowed) for w in query["words"]):
                return True
        return False

    def refines(self, query: dict[str, Any]) -> bool:
        choices = [r["values"] for r in self.reqs if r["blocking"] and r["op"] == "is"]
        if not query["words"] or not choices:
            return False
        for combo in itertools.product(*choices):
            allowed = [w for value in combo for w in _words(value)]
            if allowed and all(any(_same_word(a, w) for w in query["words"]) for a in allowed):
                return True
        return False

    def pool(self) -> set[str]:
        return {i for q in self.searches.values() if self.covering(q) or self.refines(q) for i in q["seen"]}

    def pool_rows(self) -> list[tuple[str, float, str]]:
        listings = self.listings()
        budget = self.item_budget()
        rows = []
        for item in self.pool():
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if price is None or entry.get("stock") is False or (budget is not None and price * self.quantity > budget):
                continue
            assessment = self.assess(item, listings, judge=False) if item in listings else None
            state = assessment["state"] if assessment else "not opened"
            rows.append((item, price, "may fit" if state == "fits" and assessment["contradicted"] else state))
        return sorted(rows, key=lambda r: (r[1], r[0]))

    def board(self) -> dict[str, Any] | None:
        if not any(r["blocking"] and r["op"] == "is" for r in self.reqs):
            return None
        queries = [q for q in self.searches.values() if self.covering(q)]
        complete = [q for q in queries if self.complete(q)]
        if not complete:
            return {"complete": False, "broadest": max(queries, key=lambda q: len(q["seen"]), default=None)}
        listings = self.listings()
        budget = self.item_budget()
        info: dict[str, Any] = {"complete": True, "query": min(complete, key=lambda q: len(q["seen"])), "rows": [], "out": [], "over": []}
        for item in set.intersection(*(set(q["seen"]) for q in complete)):
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if entry.get("stock") is False:
                info["out"].append(item)
            elif price is not None and budget is not None and price * self.quantity > budget:
                info["over"].append(item)
            else:
                assessment = self.assess(item, listings, judge=False) if item in listings else None
                state = assessment["state"] if assessment else "not opened"
                info["rows"].append({"item": item, "price": price, "state": "may fit" if state == "fits" and assessment["contradicted"] else state, "stale": bool(entry.get("stale"))})
        by_measure = self.plan.get("priority") == "best" and self.best
        for row in info["rows"]:
            row["measure"] = self.measure(row["item"], listings) if by_measure else None
        if by_measure:
            sign = 1 if self.best["direction"] == "least" else -1
            info["rows"].sort(key=lambda r: (r["measure"] is None, (r["measure"] or 0) * sign, r["price"] is None, r["price"] or 0, r["item"]))
        else:
            info["rows"].sort(key=lambda r: (r["price"] is None, r["price"] or 0, r["item"]))
        top = next((r for r in info["rows"] if r["state"] != "fails"), None)
        if top and (top["state"] not in ("fits", "may fit") or (by_measure and any(r["state"] == "not opened" for r in info["rows"]))):
            top = None
        info["cheapest_fit"] = top["item"] if top and top["state"] == "fits" else None
        info["cheapest_maybe"] = top["item"] if top and top["state"] == "may fit" else None
        info["top"] = top["item"] if top else None
        info["nothing_fits"] = all(r["state"] == "fails" for r in info["rows"])
        return info

    def raw_measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[float, str] | None:
        if not self.best or item not in listings:
            return None
        found = [m for value in _family({"property": self.best["property"]}, listings[item][0]).values() for m in _measures(value)]
        return found[0] if found else None

    def measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> float | None:
        raw = self.raw_measure(item, listings)
        if raw is None:
            return None
        units: list[str] = []
        for other in listings:
            seen = self.raw_measure(other, listings)
            if seen and not any(_same_unit(seen[1], u) for u in units):
                units.append(seen[1])
        if len(units) <= 1:
            return raw[0]
        counts = {u: sum(1 for other in listings if (m := self.raw_measure(other, listings)) and _same_unit(m[1], u)) for u in units}
        base = max(units, key=lambda u: counts[u])
        if _same_unit(raw[1], base):
            return raw[0]
        key = "|".join(sorted(units))
        if key not in self.unit_factors and self.time_left() > WRAP_UP_SECONDS + 20:
            prompt = UNITS_PROMPT + f"\n\nPROPERTY: {self.best['property']}\nBASE: {base}\nUNITS: " + ", ".join(u for u in units if u != base)
            result = self.judge(prompt, 200, [READ_MODEL, JUDGE_MODEL])
            factors = (result or {}).get("factors") if isinstance(result, dict) else None
            self.unit_factors[key] = {str(k).lower(): float(v) for k, v in (factors or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0}
            self.trace.append({"kind": "units", "base": base, "factors": self.unit_factors[key]})
        factor = next((v for k, v in (self.unit_factors.get(key) or {}).items() if _same_unit(k, raw[1])), None)
        return raw[0] * factor if factor else None

    def board_note(self) -> str:
        info = self.board()
        if not info:
            return ""
        priority = self.plan.get("priority")
        if not info["complete"]:
            broad = info.get("broadest")
            if broad:
                total = broad["totals"][-1] if broad["totals"] else None
                return f"[candidates] search '{broad['text']}' shows {len(broad['seen'])} of {total if total is not None else 'an unknown number of'} results: not a complete list yet" + (("; lower the price limit step by step until every result is visible (results are not sorted by price, so the cheapest can sit on any page), or split it with one more word at a time" if priority == "cheapest" else "; read its other pages, or split it with one more word at a time") + " (another required value's word, the product's own word, or a value seen on listings) and read every part" + (f", and search the required words with smaller or larger values of {self.best['property']}" if priority == "best" and self.best else "") if priority in ("cheapest", "best") else "") + self.pool_note()
            values = [r["values"][0] for r in self.reqs if r["blocking"] and r["op"] == "is"]
            return f"[candidates] a search made only of the required values' words ({' '.join(values)}) inside the requested category can list every candidate" if priority in ("cheapest", "best") else ""
        query = info["query"]
        excluded = [f"{len(info[k])} {label}" for k, label in (("out", "out of stock"), ("over", "over the budget")) if info[k]]
        text = f"[candidates] complete list from '{query['text']}': {len(query['seen'])} results, all seen; {len(info['rows'])} in stock and within budget" + (f" ({', '.join(excluded)})" if excluded else "")
        by_measure = priority == "best" and self.best
        order_name = f"{self.best['property']} ({'smallest' if self.best['direction'] == 'least' else 'largest'} first)" if by_measure else "price"
        shown = [f"{r['item']} {'price unknown' if r['price'] is None else format(r['price'], 'g')}" + (f" {self.best['property']} {'unknown' if r['measure'] is None else format(r['measure'], 'g')}" if by_measure else "") + f"{' (price changed: re-read)' if r['stale'] else ''} {r['state'].upper()}" for r in info["rows"][:8]]
        if shown:
            text += f"; by {order_name}: " + ", ".join(shown)
        clean = next((r["item"] for r in info["rows"] if r["state"] == "fits"), None)
        if info["nothing_fits"]:
            verdict = "none of the candidates fits"
        elif priority in ("cheapest", "best") and info["top"]:
            rank = "cheapest" if priority == "cheapest" else f"first by {order_name}"
            verdict = f"{info['top']} is the {rank} listing that fits (every better-ranked candidate failed)" if info["cheapest_fit"] else f"{info['top']} is the {rank} listing that may fit (its evidence is contradicted, yet such listings are usually accepted), so it is the choice" + (f"; {clean} is the best clean fit" if clean and clean != info["top"] else "")
        elif priority == "cheapest":
            verdict = "open the cheapest NOT OPENED candidates first"
        elif by_measure:
            verdict = f"open the NOT OPENED candidates: their {self.best['property']} is unknown until opened"
        elif priority == "best":
            verdict = "compare the fitting candidates on the measure asked"
        else:
            fit = next((r["item"] for r in info["rows"] if r["state"] == "fits"), None)
            maybe = next((r["item"] for r in info["rows"] if r["state"] == "may fit"), None)
            verdict = f"{fit} fits; any fitting listing is fine" if fit else (f"{maybe} may fit (its evidence is contradicted); a clean fit is safer" if maybe else "any fitting listing is fine")
        return text + " => " + verdict

    def pool_note(self) -> str:
        if self.plan.get("priority") != "cheapest":
            return ""
        cheap = [r for r in self.pool_rows() if r[2] != "fails"][:6]
        return ("; the cheapest rows seen so far across your searches for the required words (open the cheapest NOT OPENED ones first): " + ", ".join(f"{i} {format(p, 'g')} {st.upper()}" for i, p, st in cheap)) if cheap else ""

    def safe_observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        try:
            return self.observe(name, args, observation)
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return False

    def safe_board(self) -> str:
        try:
            text = self.board_note()
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return ""
        if not text or text == self.board_said:
            return ""
        self.board_said = text
        return text

    def ungrounded(self, listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], bool]]:
        if len(listings) < 3:
            return []
        found = []
        for req in [r for r in self.reqs if r["blocking"] and r["op"] == "is" and _describe(r) not in self.relaxed]:
            if any(self.evidence(req, item, listings)[0] in ("met", "contradicted") for item in listings):
                continue
            recorded = [{**_family(req, facts), **_family(req, facts, 1)} for facts, _ in listings.values()]
            recorded = [keys for keys in recorded if keys]
            reworded = any(_related(part, t, req["property"]) for keys in recorded for v in keys.values() for part in _elements(v) for t in req["values"])
            if reworded or not recorded:
                found.append((req, bool(recorded)))
        return found

    def cheaper_unopened(self) -> list[str]:
        if self.plan.get("priority") != "cheapest" or self.cheaper_asked == self.cart_version:
            return []
        held = [i for i in self.cart_ids if self.held.get(i, 1) != 0]
        chosen = self.catalog.get(held[-1]) if held else None
        if not chosen or chosen.get("price") is None:
            return []
        listings = self.listings()
        budget = self.item_budget()
        seen = self.pool()
        found = []
        for item in seen - set(listings) - set(held):
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if price is None or price >= chosen["price"] or entry.get("stock") is False or (budget is not None and price * self.quantity > budget):
                continue
            if entry.get("labels", set()) & chosen.get("labels", set()):
                found.append(item)
        self.cheaper_asked = self.cart_version
        return sorted(found, key=lambda i: (self.catalog[i]["price"], i))[:4]

    def failing_cart_swap(self) -> str:
        held = [i for i in self.cart_ids if self.held.get(i, 1) != 0]
        if not held or self.commit_failures >= 2:
            return ""
        assessment = self.assess(held[-1], judge=False)
        if not assessment or assessment["state"] != "fails":
            return ""
        best, state = self.best_candidate(relax=True)
        if not best or best == held[-1]:
            return ""
        self.commit_wanted = True
        why = "; ".join(f"{_describe(r)}: {state_}" for r, state_, _ in assessment["fails"])
        return f"{held[-1]} fails the exact check ({why}), so I will order {best} ({state}) instead."

    def better_cart_swap(self) -> str:
        priority = self.plan.get("priority")
        if priority == "best" or self.commit_failures >= 2 or self.better_swapped == self.cart_version:
            return ""
        held = [i for i in self.cart_ids if self.held.get(i, 1) != 0]
        if len(held) != 1:
            return ""
        best, state = self.best_candidate()
        if not best or best == held[0] or state not in ("fits", "may fit"):
            return ""
        mine = self.assess(held[0], judge=False)
        if not mine or mine["state"] == "fails":
            return ""
        price, current = self.catalog.get(best, {}).get("price"), self.catalog.get(held[0], {}).get("price")
        if priority == "cheapest":
            if price is None or current is None or price >= current:
                return ""
            why = f"{best} ({format(price, 'g')}) is cheaper than {held[0]} ({format(current, 'g')}) and its check says it {state}; under the cheapest rule a contradicted listing still counts"
        elif state == "fits" and mine["state"] == "fits" and mine["contradicted"]:
            why = f"{held[0]}'s evidence is contradicted, while {best} fits cleanly"
        else:
            return ""
        self.better_swapped = self.cart_version
        self.commit_wanted = True
        return why + f", so I will order {best} instead."

    def best_candidate(self, relax: bool = False) -> tuple[str | None, str]:
        listings = self.listings()
        budget = self.item_budget()
        pool = []
        for item in listings:
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if entry.get("stock") is False or (price is not None and budget is not None and price * self.quantity > budget) or self.verdicts.get(f"{item}|{self.ledger_version}"):
                continue
            assessment = self.assess(item, listings, judge=False)
            if not assessment or assessment["state"] == "fails":
                continue
            state = "may fit" if assessment["state"] == "fits" and assessment["contradicted"] else assessment["state"]
            pool.append((item, state, price, self.measure(item, listings)))
        if not pool:
            loose = self.ungrounded(listings) if relax else []
            if not loose:
                return None, ""
            self.relaxed.update({_describe(r): recorded for r, recorded in loose})
            self.verdicts = {k: v for k, v in self.verdicts.items() if not v.startswith("The exact check")}
            self.trace.append({"kind": "relaxed", "requirements": [_describe(r) for r, _ in loose]})
            return self.best_candidate()
        grade = {"fits": 0, "may fit": 1, "unclear": 2}
        held = set(self.cart_ids)
        priority = self.plan.get("priority")
        if priority == "cheapest":
            key = lambda c: (c[1] == "unclear", c[2] is None, c[2] or 0, grade[c[1]], c[0] not in held)
        elif priority == "best" and self.best:
            sign = 1 if self.best["direction"] == "least" else -1
            key = lambda c: (c[1] == "unclear", c[3] is None, (c[3] or 0) * sign, grade[c[1]], c[2] or 0)
        else:
            key = lambda c: (grade[c[1]], c[0] not in held, c[2] is None, c[2] or 0)
        item, state, _, _ = min(pool, key=key)
        return item, state

    def ready_choice(self) -> str | None:
        item, state = self.best_candidate()
        if not item:
            return None
        if self.plan.get("priority") in ("cheapest", "best"):
            info = self.board()
            return item if info and info.get("complete") and info.get("top") == item else None
        return item if state == "fits" else None

    def decline_blocker(self) -> str:
        item, state = self.best_candidate(relax=True)
        for _ in range(3):
            if not item or not self.verify_item(item, "ORDER"):
                break
            item, state = self.best_candidate(relax=True)
        if item and self.commit_failures < 2:
            self.decline_blocks += 1
            why = {"fits": "fits every requirement", "may fit": "may fit (its evidence is contradicted, yet such listings are usually accepted)", "unclear": "may still fit (no check proves its UNCLEAR requirement fails)"}[state]
            if self.decline_blocks >= 2:
                self.commit_wanted = True
                return f"{item} {why}, and a decline is right only when every candidate is proven wrong, so I will order {item} now."
            return f"{item} {why}, and a decline is right only when every candidate is proven wrong. Put it in the cart (quantity {self.quantity}) and send {self.how('order')}."
        info = self.board()
        if not item and info is not None and not info["complete"] and self.steps_left() > COMMIT_STEPS + 2 and not self.decline_hinted:
            self.decline_hinted = True
            return "no search has proven that nothing fits yet. " + (self.board_note() or "Search the requested category with the required values' words") + ". Search that way, then decline if it finds nothing."
        return ""

    def cart_line_spec(self) -> tuple[str, str, str | None] | None:
        """A cart that takes the whole basket as a list of lines: that list's argument and each line's id and amount keys."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        for key, spec in (schema.get("properties") or {}).items():
            if not isinstance(spec, dict) or "array" not in (spec.get("type") or []):
                continue
            inner = spec.get("items") if isinstance(spec.get("items"), dict) else {}
            fields = inner.get("properties") if isinstance(inner.get("properties"), dict) else {}
            ids = [k for k, f in fields.items() if isinstance(f, dict) and f.get("type") == "string" and "enum" not in f]
            amounts = [k for k, f in fields.items() if isinstance(f, dict) and f.get("type") == "integer"]
            if len(ids) == 1:
                return key, ids[0], amounts[0] if amounts else None
        return self.line_shape

    def line_quantity(self, line: str) -> int:
        item = next((x for x in self.items if x.get("line") == line), None)
        return (item or {}).get("quantity") or self.quantity

    def cart_lines_now(self) -> dict[str, int]:
        return {i: self.held.get(i) or self.line_quantity(i) for i in self.cart_ids if self.held.get(i, 1) != 0}

    def cart_set_call(self, lines: dict[str, int]) -> dict[str, Any] | None:
        spec = self.cart_line_spec()
        name = self.roles.get("cart")
        if not spec or not name:
            return None
        key, ident, units = spec
        rows = [{ident: item, **({units: count} if units else {})} for item, count in lines.items()]
        args, problems = _check({**(self.cart_modes.get("add") or {}), key: rows}, self.schemas[name], "args")
        return None if problems else {"name": name, "args": args}

    def set_cart(self, lines: dict[str, int], signature: str) -> str:
        """A list-style call sets the whole cart (every replay of races 2, 7, 13 and 15): what it leaves out is gone."""
        notes = []
        for item in [i for i in self.cart_ids if i not in lines]:
            self.held[item] = 0
            if self.multi():
                notes.append(self.item_removed(item))
        fresh = [i for i in lines if i not in self.cart_ids]
        self.cart_ids = [i for i in self.cart_ids if i in lines] + fresh
        self.held.update(lines)
        for item in lines:
            self.cart_state[item] = signature
        if self.multi():
            for item in lines:  # not only the new lines: one set aside earlier can take over a product that lost its line
                if any(x.get("line") == item for x in self.items):
                    continue
                left = [k for k, x in enumerate(self.items) if not x.get("line")]
                if not left:
                    break
                if self.items[self.focus].get("line"):
                    self.switch_item(left[0])
                notes.append(self.item_done(item))
        return next((n for n in reversed(notes) if n), "")

    def cart_call(self, item: str, mode: str) -> dict[str, Any] | None:
        if self.cart_line_spec():
            lines = self.cart_lines_now()
            if mode == "add":
                lines[item] = self.quantity
            else:
                lines.pop(item, None)
            return self.cart_set_call(lines)
        name = self.roles.get("cart")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        chosen = self.cart_modes.get(mode) or {}
        if not chosen and mode == "add" and self.add_signature:
            chosen = _parse_json_object(self.add_signature) or {}
        if any(isinstance(spec, dict) and "enum" in spec and k in (schema.get("required") or []) and k not in chosen for k, spec in props.items()):
            return None
        args = dict(chosen)
        ids = [k for k, spec in props.items() if isinstance(spec, dict) and (spec.get("type") == "string" or "string" in (spec.get("type") or [])) and "enum" not in spec and k not in args]
        if len(ids) != 1:
            return None
        args[ids[0]] = item
        amounts = [k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec]
        if amounts:
            args[amounts[0]] = self.quantity if mode == "add" else (self.held.get(item) or self.quantity)
        args, problems = _check(args, schema, "args")
        return None if problems else {"name": name, "args": args}

    def deadline_due(self) -> bool:
        if self.finished or self.dead or self.commit_failures >= 2 or not self.roles.get("order"):
            return False
        if self.commit_wanted:
            self.commit_wanted = False
            return True
        if self.order_attempts == 0:
            return self.steps_left() <= COMMIT_STEPS or self.time_left() < WRAP_UP_SECONDS
        return self.refusals > 0 and self.steps_left() <= 3

    def commit_now(self) -> bool:
        self.commit_tries += 1
        self.committing = True
        try:
            acted, ordered = self.commit_steps()
        finally:
            self.committing = False
        if not ordered:
            self.commit_failures += 1
        return acted

    def commit_steps(self) -> tuple[bool, bool]:
        item, state = self.best_candidate(relax=True)
        if item and self.verify_item(item, "ORDER"):
            item, state = self.best_candidate(relax=True)
        if not item:
            return False, False
        in_cart = [i for i in self.cart_ids if self.held.get(i, 1) != 0]
        # The lines of the other products of a several-product order stay.
        kept = {x["line"]: x["quantity"] for k, x in enumerate(self.items) if k != self.focus and x.get("line") in in_cart} if self.multi() else {}
        steps: list[dict[str, Any]] = []
        if self.cart_line_spec():
            wanted = {**kept, item: self.held.get(item) or self.quantity}
            if wanted != self.cart_lines_now():
                setting = self.cart_set_call(wanted)
                if setting is None:
                    return False, False
                steps.append(setting)
        else:
            for other in [i for i in in_cart if i != item and i not in kept]:
                removal = self.cart_call(other, "remove")
                if removal is None:
                    return False, False
                steps.append(removal)
            if item not in in_cart and self.roles.get("cart"):
                addition = self.cart_call(item, "add")
                if addition is None:
                    return False, False
                steps.append(addition)
        order = self.roles["order"]
        steps.append({"name": order, "args": self.neutral_args(order, "Order the cart.", self.order_args)})
        if len(steps) > self.steps_left():
            return False, False
        self.trace.append({"kind": "commit", "item": item, "state": state, "steps": steps})
        ordered = False
        for index, action in enumerate(steps):
            if action["name"] == order:
                self.order_checked = True
            call = {"id": f"commit_{self.step}_{index}", "type": "function", "function": {"name": action["name"], "arguments": json.dumps(action["args"])}}
            note = f"Steps are running out, so I will order my best candidate {item} ({state}) now." if index == 0 else "Continuing."
            self.messages.append({"role": "assistant", "content": note, "tool_calls": [call]})
            replies = self.run_calls([call], planned=True)
            for call_id, text in replies:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
            ordered = ordered or (action["name"] == order and bool(replies) and not any(text.startswith("Not run") for _, text in replies))
            if self.finished or any(text.startswith("Not run") for _, text in replies):
                break
        return True, ordered

    def safe_backups(self, chosen: str) -> list[str]:
        try:
            listings = self.listings()
            budget = self.item_budget()
            found = []
            for item in listings:
                entry = self.catalog.get(item, {})
                price = entry.get("price")
                if item == chosen or entry.get("stock") is False or (price is not None and budget is not None and price * self.quantity > budget):
                    continue
                if entry.get("category") != self.catalog.get(chosen, {}).get("category"):
                    continue
                assessment = self.assess(item, listings, judge=False) or {}
                if assessment.get("state") == "fits" and (not assessment.get("contradicted") or self.plan.get("priority") in ("cheapest", "best")):
                    found.append(item)
            return found
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return [chosen]

    def safe_ready(self) -> str | None:
        try:
            return self.ready_choice()
        except Exception as exc:
            self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
            return None

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()):
                    text = _compact(_brief(value, 400))[:2500]
                    if text not in found:
                        found.append(text)
                    return
                for v in value.values():
                    walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for record in self.records:
            walk(record, 0)
        return found[-4:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        return self.verify_item(self.cart_ids[-1], "ORDER")

    def verify_item(self, item: str, purpose: str) -> str:
        key = f"{item}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        assessment = self.assess(item)
        if assessment and assessment["state"] == "fails":
            verdict = f"The exact check says {item} does NOT meet every requirement: " + "; ".join(f"{_describe(r)}: {state}" + (f" ({evidence[:80]})" if evidence else "") for r, state, evidence in assessment["fails"]) + ". Choose another listing whose check fits."
            self.trace.append({"kind": "verify", "item": item, "exact": assessment["state"]})
            self.verdicts[key] = verdict
            return verdict
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if assessment:
            prompt += "\n\nEXACT CHECKS (final):\n" + "\n".join(self.check_lines(assessment))
        else:
            prompt += "\n\nEXACT CHECKS: this listing's property details were not opened, so treat every requirement above as UNCLEAR and decide it from the records."
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
        tools = self.tools
        if names is not None and order and cart:
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.planner_on and self.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        return tools

    def property_menu(self) -> tuple[str, int]:
        self.menu_values = {}
        counts: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        fact_sets = 0

        def short(value: Any) -> bool:
            return isinstance(value, str) and 0 < len(value) <= 70

        fits = self.fitting_items()
        fit_values: dict[str, set[str]] = {}
        for record in self.records:
            listing = _listing(record)
            strings = [(k, v) for k, v in listing[1].items() if short(v)] if listing else []
            if not strings:
                continue
            fact_sets += 1
            for key, item in strings:
                if re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item) and ("-" in item or "_" in item):
                    continue
                name = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                if _never_ask(name):
                    continue
                counts[name] = counts.get(name, 0) + 1
                bucket = samples.setdefault(name, [])
                if item.lower() not in bucket and len(bucket) < 5:
                    bucket.append(item.lower())
                self.menu_values.setdefault(name, Counter())[item.strip().lower()] += 1
                if listing[0] in fits:
                    fit_values.setdefault(name, set()).add(item.lower())
        # A property whose values differ is about four times as likely to be the hidden need (5.7% against 1.4% of listed
        # properties, races 1-15). The mark is read among the listings that meet the stated requirements when at least 3
        # do, else among all listings opened (w13n left it out then, so a first plan after 2 listings often had none).
        if len(fits) >= 3:
            splitting = {k for k, v in fit_values.items() if len(v) > 1}
            among = f"the {len(fits)} listings that meet the stated requirements"
        else:
            splitting = {k for k, c in self.menu_values.items() if len(c) > 1 and k in counts} if fact_sets >= 2 else set()
            among = f"the {fact_sets} listings opened"
        # Order: more listings first, a differing property counting about as much as twice the listings (from the same races,
        # a fitted model gave about -0.56 log-odds per natural-log step of a property's count rank and +0.55 when its values
        # differ).
        ranked = sorted(counts, key=lambda k: (-(math.log(counts[k]) + (0.7 if k in splitting else 0.0)), -counts[k]))[:28]
        self.menu_order = [k for k in ranked if k in splitting] + [k for k in ranked if k not in splitting and len(samples.get(k, [])) > 1]
        self.menu_names, self.splitting = list(ranked), splitting

        def note(k: str) -> str:
            return f" [differs among {among}]" if k in splitting else ""
        lines = [f"- {k} (in {counts[k]} listings){note(k)}: " + " | ".join(samples.get(k, [])) for k in ranked]
        return "\n".join(lines), fact_sets

    def search_titles(self, limit: int = 15) -> list[str]:
        pool = self.pool()
        seen = list(dict.fromkeys(i for q in self.searches.values() for i in q["seen"]))
        titles: list[str] = []
        for item in [i for i in seen if i in pool] + [i for i in seen if i not in pool]:
            title = self.catalog.get(item, {}).get("title")
            if title and title.lower() not in {t.lower() for t in titles}:
                titles.append(title)
            if len(titles) >= limit:
                break
        return titles

    def fitting_items(self) -> set[str]:
        if not any(r["blocking"] for r in self.reqs):
            return set()
        listings = self.listings()
        return {item for item in listings if (self.assess(item, listings, judge=False) or {}).get("state") not in (None, "fails")}

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and not (re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item)):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines[:12])

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target:
            return None
        if any(k != target for k in required):
            return None
        return {target: text}

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        if self.item_messages_left(limit) <= 0:
            # This product used its share; the next products still get theirs (w13k stopped asking for the whole task here).
            return False
        if self.need_found and self.multi():
            # This product's hidden need is known: its remaining questions go to the next products.
            return False
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 4 and self.step < 12 and self.steps_left() > 20 and not urgent and not force:
            return False
        if force:
            urgent = False
        if self.steps_left() <= 10 or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        if self.asked and not force and not self.ask_due and self.step < self.last_ask_step + 2 and fact_sets < self.last_ask_facts + 2 and self.steps_left() > 14:
            return False
        if self.bundling is None:
            self.bundling = True  # lists from the first question at any limit, until the shopper asks for one at a time
        # Asking one property at a time, the planner ranks as many properties as messages are left and the first is sent.
        size = BUNDLE_SIZE if self.bundling else PER_MESSAGE
        count = size if size > 1 else max(1, self.item_messages_left(limit))
        if self.bundling:
            planned = self.list_question(limit, menu)
            size = max(1, len(planned))
        else:
            planned = self.lead_first(self.plan_questions(count, menu))
        # A planned free question (no property, such as the budget) goes alone; otherwise the top properties together.
        free = [item for item in planned[:1] if item["question"] != _ask_text(item["property"])]
        named = [] if free else planned[:size]
        text = free[0]["question"] if free else (_ask_several([item["property"] for item in named]) if named else "")
        if text and named and self.multi():
            text = f"For the {self.items[self.focus]['product']}, " + text[0].lower() + text[1:]
        if text and len(named) == 1:
            text = _with_examples(text, self.examples_for(named[0]["property"]))
        calls = []
        labels = []
        args = self.message_args(text) if text else None
        if args:
            args, problems = _check(args, self.schemas[tool], "args")
            if not problems:
                calls.append({"id": f"ask_{self.messages_used + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
                labels = [(item["property"], text) for item in (named or free)]
        if not calls:
            # Only a question the tool can't take ends asking; nothing to ask yet (no properties seen) leaves it on.
            if text:
                self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        known_before = [r["property"] for r in self.reqs]
        self.bundle_refused = False
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index == 0:
                for prop, question in self.answered_labels(labels, known_before):
                    self.asked.append(f"- {prop}: asked \"{question}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        self.last_ask_step, self.last_ask_facts, self.ask_due = self.step, fact_sets, False
        self.planner_on = self.messages_used < limit
        return True

    def list_question(self, limit: int, menu: str) -> list[dict[str, str]]:
        """The next question while lists may be taken: one ranking per product; two lists of LIST_START, then up to
        LIST_MAX led by a different top guess each time (see LIST_MAX). A refused list turns lists off for the task
        (bundling), so the long lists only follow lists the shopper took."""
        if not self.list_plan:
            planned = self.plan_questions(LIST_MAX, menu)
            if planned and planned[0]["question"] != _ask_text(planned[0]["property"]):
                return planned[:1]  # a free question (such as the budget) goes alone; the ranking comes next time
            self.list_plan = [item["property"] for item in planned]
            self.list_stage, self.list_sent, self.list_rounds, self.list_order = 0, 0, 0, []
        asked = [line.split(":")[0].lstrip("- ") for line in self.asked]
        settled = asked + [str(r["property"]) for r in self.reqs]
        open_ = [(rank, p) for rank, p in enumerate(self.list_plan) if not any(_same_property(p, other) for other in settled)]
        if not open_:
            return []
        stage, self.list_stage = self.list_stage, self.list_stage + 1
        if stage <= 1:
            return [{"property": p, "question": _ask_text(p)} for _, p in open_[:LIST_START]]
        if not self.list_rounds:
            self.list_rounds = max(1, min(3, self.item_messages_left(limit)))
            self.list_order = [p for _, p in open_]  # the long lists take turns among what is still open, likeliest first
        open_ = [(rank, p) for rank, p in enumerate(self.list_order) if any(p == q for _, q in open_)]
        turn = self.list_sent % self.list_rounds
        lead = [p for rank, p in open_ if rank % self.list_rounds == turn]
        rest = [p for _, p in open_ if p not in lead]
        self.list_sent += 1
        return [{"property": p, "question": _ask_text(p)} for p in (lead + rest)[:LIST_MAX]]

    def lead_first(self, planned: list[dict[str, str]]) -> list[dict[str, str]]:
        """Put the properties a list question left open ahead of the planner's new picks (a free question, such as the
        budget, stays first)."""
        if not self.lead or (planned and planned[0]["question"] != _ask_text(planned[0]["property"])):
            return planned
        asked = [line.split(":")[0].lstrip("- ") for line in self.asked]
        settled = asked + [str(r["property"]) for r in self.reqs]
        lead = [p for p in self.lead if not any(_same_property(p, other) for other in settled)]
        rest = [item for item in planned if not any(_same_property(item["property"], p) for p in lead)]
        return [{"property": p, "question": _ask_text(p)} for p in lead] + rest

    def examples_for(self, prop: Any) -> list[str]:
        """Two values the opened listings show most often for this property, to make clear which property a question means."""
        name = " ".join(_words(prop))
        values = self.menu_values.get(name)
        if values is None:
            values = next((v for k, v in self.menu_values.items() if _same_property(k, prop)), Counter())
        return [v for v, _ in values.most_common() if _example_ok(v)][:2]

    def answered_labels(self, labels: list[tuple[str, str]], known_before: list[Any]) -> list[tuple[str, str]]:
        """The properties a question settled. A refused question settled none of the several it named. A shopper answers
        about one property a message, so when one states a need, the others named with it are still open."""
        fresh = [r["property"] for r in self.reqs if r["property"] not in known_before]
        if len(labels) < 2:
            if fresh and labels and any(_same_property(labels[0][0], f) for f in fresh):
                self.need_found = True
            return labels
        if self.bundle_refused:
            return []
        names = [label[0] for label in labels]
        hit = next((i for i, n in enumerate(names) if any(_same_property(n, f) for f in fresh)), -1)
        if hit >= 0:
            # The shopper answered the first property it holds: the ones named before it hold nothing, the ones after are open.
            self.lead, self.list_misses, self.need_found = names[hit + 1:], 0, True
            return labels[:hit + 1]
        if fresh:
            # A need on a property the list did not name: the first label is settled, the rest stay open in their order.
            self.lead, self.list_misses = names[1:], 0
            return labels[:1]
        # Nothing revealed: the first property is settled (a need there would have been answered); the rest come back in
        # reverse so whatever the shopper does not mind no longer stands in front of them. After a second miss in a row
        # the planner's own order leads again.
        self.list_misses += 1
        self.lead = names[:0:-1] if self.list_misses < 2 else []
        return labels[:1]

    def plan_questions(self, count: int, menu: str) -> list[dict[str, str]]:
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        titles = self.search_titles()
        if titles:
            prompt += "\n\nTITLES OF LISTINGS IN THE SEARCH RESULTS (most not opened yet; titles name main details such as who it is for, size or specification):\n" + "\n".join(f"- {t}" for t in titles)
        samples = self.plan_samples(prompt)
        self.trace.append({"kind": "plan_questions", "result": samples[0] if samples else None, "samples": len(samples), "titles": len(titles), "votes": [[str(q.get("property")) for q in (r.get("questions") or []) if isinstance(q, dict)] for r in samples]})
        asked = [line.split(":")[0].lstrip("- ") for line in self.asked]
        known = [set(_words(r["property"])) for r in self.reqs]
        voted = _vote(samples)
        if len(self.menu_names) >= 5:
            def listed(item: dict[str, str]) -> str | None:
                return next((n for n in self.menu_names if _same_property(item.get("property"), n)), None)

            def titled(item: dict[str, Any]) -> bool:
                if set(_words(item.get("property"))) & VAGUE_WORDS:
                    return False
                found = {" ".join(w for w in _words(v) if len(w) > 1) for v in item.get("values") or [] if any(_has_words(t, [w for w in _words(v) if len(w) > 1]) for t in titles)}
                return len(found - {""}) >= 2
            voted = [item for item in voted if listed(item) or titled(item) or str(item.get("question") or "").strip()]
        chosen = voted + [{"property": name} for name in self.menu_order]
        planned = []
        for item in chosen:
            prop, question = str(item.get("property") or "").strip(), str(item.get("question") or "").strip()
            if not question and (not _words(prop) or prop.lower() == "none" or set(_words(prop)) in known):
                continue
            if _never_ask(prop):
                continue
            if any(_same_property(prop, other) for other in asked + [str(p.get("property") or "") for p in planned]):
                continue
            planned.append({"property": prop or question, "question": _ask_text(prop) if _words(prop) else question})
            if len(planned) >= count:
                break
        return planned

    def plan_samples(self, prompt: str) -> list[dict[str, Any]]:
        temperatures = PLAN_TEMPERATURES if self.time_left() > WRAP_UP_SECONDS + 60 else PLAN_TEMPERATURES[:1]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(temperatures)) as pool:
            futures = [pool.submit(self.plan_one, prompt, t) for t in temperatures]
            results = [f.result() for f in futures]
        return [r for r in results if isinstance(r, dict)]

    def plan_one(self, prompt: str, temperature: float) -> dict[str, Any] | None:
        """One planner answer. A plan of up to LIST_MAX properties with its hint notes can run long, so the limit is
        PLAN_TOKENS (only the tokens used are paid for); an answer that isn't a readable plan, such as one cut off before
        its JSON closes, counts as a failure and the next model is tried."""
        for model in (PLAN_MODEL, JUDGE_MODEL):
            result = self.judge(prompt, PLAN_TOKENS, [model], temperature)
            if isinstance(result, dict) and isinstance(result.get("questions"), list) and result["questions"]:
                return result  # the plan is built from "questions" only, so hint notes without them count as no answer
        return None

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.message_budget if self.message_budget is not None else 3
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
        return True

    def cart_amount(self, name: str, args: dict[str, Any]) -> int:
        properties = self.schemas[name].get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                return value if isinstance(value, int) and not isinstance(value, bool) else 1
        return 1

    def cart_target(self, name: str, args: dict[str, Any]) -> str | None:
        properties = self.schemas[name].get("properties") or {}
        return next((v.strip() for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        listed = _line_args(args)
        if listed is not None:
            wanted = {x["quantity"] for x in self.items} if self.multi() else {self.quantity}
            wrong = [f"{count} unit(s) of {item}" for item, count in listed.items() if count not in wanted]
            if wrong:
                return f"this call sets {', '.join(wrong)}, but each line needs exactly {' or '.join(str(n) for n in sorted(wanted))} unit(s)."
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        amount = self.cart_amount(name, args)
        held = self.held.get(self.cart_target(name, args) or "", 0) or 0
        if held + amount == self.quantity:
            return ""
        if held:
            return f"this listing is already in the cart with {held} unit(s); adding {amount} more makes {held + amount}, but the order needs exactly {self.quantity} unit(s) of one listing."
        return f"the request needs exactly {self.quantity} unit(s) of one listing, but this call asks for {amount}."

    def cart_quantity_problem(self) -> str:
        if self.view_version != self.cart_version:
            return ""
        lines = self.view_lines
        chosen = self.cart_ids[-1] if self.cart_ids and self.cart_ids[-1] in lines else (next(iter(lines)) if len(lines) == 1 else None)
        count = lines.get(chosen) if chosen else None
        if count is None or count == self.quantity:
            return ""
        return f"the latest cart view shows {count} unit(s) of {chosen}, but the order needs exactly {self.quantity} unit(s) of one listing. Fix that line so it holds exactly {self.quantity} (adding a listing that is already in the cart raises its count), view the cart again, then order."

    def audit_decline(self) -> str:
        if self.audited:
            return ""
        self.audited = True
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: self.assess(item, listings, judge=False) for item in listings}
        fits = [item for item, a in states.items() if a and a["state"] == "fits"]
        if fits:
            return "The exact checks find that these opened listings meet every requirement: " + ", ".join(fits[:6]) + ". Re-check their stock and price, and buy the right one (the cheapest when the request asks for it); decline only if each one is out of stock or over the budget."
        unclear = [item for item, a in states.items() if a and a["state"] == "unclear"]
        if not unclear or self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        blocks = []
        for item in unclear[:8]:
            blocks.append(f"{item}: " + "\n".join(self.item_records(item)[-2:])[:2500] + "\nEXACT CHECKS: " + " | ".join(self.check_lines(states[item])))
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTINGS:\n" + "\n\n".join(blocks)
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."

    def neutral_args(self, name: str, text: str, mode: dict[str, Any] | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = dict(mode or {})
        for key in schema.get("required") or []:
            spec = properties.get(key) or {}
            if key in args:
                continue
            if "enum" in spec and spec["enum"]:
                args[key] = spec["enum"][0]
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = DECLINE_NOTE
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and "enum" not in spec:
                    cleaned[key] = neutral[: spec.get("maxLength", len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        return args

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        viable = [i for i in self.cart_ids if self.held.get(i, 1) != 0 and self.catalog.get(i, {}).get("stock") is not False]
        if order and self.roles.get("cart") and self.steps_left() >= 2 and not viable:
            try:
                item, _ = self.best_candidate(relax=True)
                addition = self.cart_call(item, "add") if item else None
            except Exception:
                item, addition = None, None
            if addition:
                reply = self.env_call([addition])
                self.trace.append({"kind": "forced_finish", "action": addition, "reply": reply})
                if reply and _cart_accepted(reply.get("observation")):
                    self.cart_used = True
                    self.cart_ids.append(item)
                    viable = [item]
        if order and (viable or (self.cart_used and not self.cart_ids)):
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.", self.order_args)}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, DECLINE_NOTE, self.decline_args)}
        else:
            return
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if not self.finished and self.finish_kind(action["name"], action["args"]) != "decline" and decline and self.steps_left() > 0 and self.time_left() >= 2:
            action = {"name": decline, "args": self.neutral_args(decline, DECLINE_NOTE, self.decline_args)}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            if observation.get("error") == "budget_exceeded":
                if name == self.roles.get("message") or self.message_budget is not None:
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That budget is used up; do not call this tool again.")
            if "refusal" in observation:
                self.refusals += 1
                self.order_checked = False
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids)
        return " ".join(notes)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        planned_messages = 0
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."))
                continue
            if not planned and name not in allowed:
                if name == message_tool and self.planner_on:
                    reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
                else:
                    reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
                replies.append((call_id, f"Not run (no step used): {reason}."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _parse_json_object(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = _check(args, self.schemas[name], "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])))
                continue
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if self.finish_kind(name, args) == "unknown" and self.order_args and self.decline_args and not self.mode_asked:
                self.mode_asked = True
                replies.append((call_id, f"Not run (no step used): say explicitly how this ends: {self.how('order')} places the order, {self.how('decline')} declines."))
                continue
            if name == message_tool and self.message_budget is not None and self.messages_used + planned_messages >= self.message_budget:
                replies.append((call_id, "Not run (no step used): the message budget is used up. Do not message the shopper again."))
                continue
            signature = name + _compact(args)
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if name == cart_tool and self.steps_left() > 3:
                properties = self.schemas[name].get("properties") or {}
                signature_now = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                if self.add_signature is None or signature_now == self.add_signature:
                    listed = _line_args(args)
                    targets = [self.cart_target(name, args)] if listed is None else [i for i in listed if i not in self.cart_lines_now()]
                    verdict = ""
                    for target in targets:
                        if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                            self.cart_checked.add(f"{target}|{self.ledger_version}")
                            verdict = self.verify_item(target, "ADD TO THE CART")
                            if verdict:
                                break
                    if verdict:
                        replies.append((call_id, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."))
                        continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            kind = self.finish_kind(name, args)
            missing = [item["product"] for item in self.items if not item.get("line")] if self.multi() else []
            # A cart with a line per product can go ahead even when a line couldn't be matched to its product here
            # (units added one by one, or products put in out of order).
            if kind == "order" and missing and len(self.cart_lines_now()) < len(self.items) and self.steps_left() > 2:
                replies.append((call_id, "Not run (no step used): the order needs a cart line for every product, and these have none yet: "
                                + ", ".join(missing) + f". Work on {self.item_label()} now."))
                return replies
            if kind == "order" and self.steps_left() > 2:
                problem = self.cart_quantity_problem()
                if problem and self.qty_blocked != self.cart_version:
                    self.qty_blocked = self.cart_version
                    replies.append((call_id, "Not run (no step used): " + problem))
                    return replies
            limit = self.message_budget if self.message_budget is not None else 3
            if kind == "order" and not self.committing and self.planner_on and self.asked and not self.ask_due and self.messages_used < limit and self.steps_left() > 11 and self.time_left() > WRAP_UP_SECONDS + 40:
                self.ask_due = True
                replies.append((call_id, "Not run yet (no step used): one more planned question to the shopper goes out first, since its answer may change the choice."))
                return replies
            if kind == "order" and not self.committing and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS + 20:
                cheaper = self.cheaper_unopened()
                if cheaper:
                    shown = ", ".join(f"{i} ({format(self.catalog[i]['price'], 'g')})" for i in cheaper)
                    replies.append((call_id, f"Not run yet (no step used). The request asks for the cheapest fitting listing, and your searches for the required values showed cheaper listings in the same category that you have not opened: {shown}. Open them and buy the cheapest one that fits; if none fits, send {self.how('order')} again."))
                    return replies
            if kind == "order" and not self.committing and self.steps_left() > 3:
                try:
                    better = self.better_cart_swap()
                except Exception as exc:
                    self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    better = ""
                if better:
                    self.trace.append({"kind": "better_swap", "note": better})
                    replies.append((call_id, "Not run (no step used): " + better))
                    return replies
            if kind == "order" and self.order_checked and not self.committing and self.steps_left() > 2:
                try:
                    swap = self.failing_cart_swap()
                except Exception as exc:
                    self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    swap = ""
                if swap:
                    replies.append((call_id, "Not run (no step used): " + swap))
                    return replies
            if kind == "order" and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS:
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                replies.append((call_id, f"Not run yet (no step used). Pre-order check: (1) the cart holds exactly one line, the chosen item at the required quantity, and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise send {self.how('order')} again."))
                return replies
            if kind == "decline":
                try:
                    blocker = self.decline_blocker()
                except Exception as exc:
                    self.trace.append({"kind": "board_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                    blocker = ""
                if blocker:
                    replies.append((call_id, "Not run (no step used): " + blocker))
                    return replies
                if not self.decline_checked and self.steps_left() > 6 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.decline_checked = True
                    audit = self.audit_decline()
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    replies.append((call_id, f"Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, send {self.how('decline')} again."))
                    return replies
                args = self.clean_decline(args, name)
                runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
                runnable = runnable[: max(0, keep)]
        for start in range(0, len(runnable), 16):
            chunk = runnable[start : start + 16]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "Not run: the episode is over or no steps are left."))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append((call_id, "Not run: the episode finished before this call."))
                    continue
                observation = observations[index]
                count_note = ""
                if name == message_tool:
                    self.messages_used += 1
                if self.finish_kind(name, args) == "order":
                    self.order_attempts += 1
                if name == cart_tool:
                    self.cart_used = True
                    self.order_checked = False
                if name in (cart_tool, message_tool) or name in finishing:
                    self.recent = []
                else:
                    self.recent = (self.recent + [name + _compact(args)])[-2:]
                self.records.append(observation)
                if name == cart_tool:
                    properties = self.schemas[name].get("properties") or {}
                    signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                    if self.add_signature is None:
                        self.add_signature = signature
                    for key, value in args.items():
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    target = self.cart_target(name, args)
                    if target:
                        if signature == self.add_signature and _cart_accepted(observation):
                            known = self.held.get(target, 0)
                            self.held[target] = None if known is None else known + self.cart_amount(name, args)
                            if self.held[target] not in (None, self.quantity):
                                count_note = f"The cart line for {target} now holds {self.held[target]} unit(s) (adding a listing that is already in the cart raises its count); the order needs exactly {self.quantity}."
                            elif self.multi() and not any(item.get("line") == target for item in self.items):
                                count_note = self.item_done(target)
                        else:
                            self.held[target] = None
                            if self.multi():
                                count_note = self.item_removed(target)
                    listed = _line_args(args)
                    if listed is not None and _cart_accepted(observation):
                        self.line_shape = self.line_shape or _line_shape(args)
                        count_note = self.set_cart(listed, signature)
                    if _cart_accepted(observation) and _has_list(_unwrap(observation)):
                        # The tool showed the cart (six races do): that beats what the call's arguments suggest.
                        shown = _cart_lines(_unwrap(observation), self.seen_ids | set(self.cart_state))
                        for item in self.held:
                            self.held[item] = shown.get(item, 0)
                        self.held.update(shown)
                        self.cart_ids = [i for i in self.cart_ids if i in shown] + [i for i in shown if i not in self.cart_ids]
                        for item in self.items if self.multi() else []:
                            if item.get("line") and item["line"] not in shown:
                                count_note = self.item_removed(item["line"]) or count_note
                    self.cart_version += 1
                    self.cart_changed_at = self.step
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                    if _has_list(_unwrap(observation)):
                        lines = _cart_lines(_unwrap(observation), self.seen_ids | set(self.cart_state))
                        for item in self.held:
                            self.held[item] = lines.get(item, 0)
                        self.held.update(lines)
                        self.view_lines = lines
                        self.view_version = self.cart_version
                note = self.note_observation(name, observation)
                if count_note:
                    note = (note + " " if note else "") + count_note
                heard: list[str] = []
                events: list[str] = []
                if name == message_tool:
                    if isinstance(observation, str):
                        heard = [observation]
                    elif isinstance(observation, dict) and "error" not in observation:
                        said = [v for v in observation.values() if isinstance(v, list) and v and all(isinstance(x, str) for x in v)]
                        answers = _strings([v for v in observation.values() if not any(v is x for x in said)])
                        if said and answers:
                            _sentences(said, events)
                            _sentences(answers, heard)
                        else:
                            _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                    question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                else:
                    question = None
                    if isinstance(observation, dict):
                        for value in observation.values():
                            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                _sentences(value, heard)
                            elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                                heard.append(value)
                if events:
                    meaning = self.interpret(events, None)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                if heard:
                    meaning = self.interpret(heard, question)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                listing = _listing(observation)
                if listing:
                    check = self.check_note(listing[0])
                    if check:
                        note = (note + " " if note else "") + check
                if self.safe_observe(name, args, observation) or listing:
                    board = self.safe_board()
                    if board:
                        note = (note + " " if note else "") + board
                text = _compact(observation)
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user += "\n\n" + brief
        user += "\n\n" + self.status()
        if opening is not None:
            self.records.append(opening)
            self.safe_observe("opening", {}, opening)
            _collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            try:
                committed = self.deadline_due() and self.commit_now()
            except Exception as exc:
                self.trace.append({"kind": "commit_error", "error": f"{type(exc).__name__}: {exc}"[:200]})
                committed = False
            if committed:
                continue
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            message = self.infer(self.messages, self.tools_for_turn(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    break
                continue
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls):
                if self.ask_planned(force=True):
                    continue
            if not calls:
                idle += 1
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not episode.finished and not episode.dead:
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
