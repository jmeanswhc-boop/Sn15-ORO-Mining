from __future__ import annotations
import concurrent.futures
import itertools
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict
from src.agent.proxy_client import ProxyClient
MAIN_MODEL = 'anthropic/claude-haiku-5.5'
BACKUP_MODEL = 'openai/gpt-6-luna'
SETUP_MODEL = 'anthropic/claude-sonnet-5.5'
JUDGE_MODEL = 'anthropic/claude-sonnet-5.5'
READ_MODEL = "anthropic/claude-haiku-5.5"
VALUE_MODEL = "anthropic/claude-sonnet-5.5"
REFERENCE_MODEL = 'google/gemini-3.8-flash'
SETUP_TOKENS = 6000
JUDGE_TOKENS = 4000
NAV_TOKENS = 6000
MODEL_SETTINGS = {REFERENCE_MODEL: {'reasoning': {"effort": 'low'}}}
SETUP_SETTINGS = {SETUP_MODEL: {"reasoning": {"effort": 'low'}}}
MODEL_TOKENS = {REFERENCE_MODEL: 1500}
PRICES = {'anthropic/claude-haiku-5.5': (1e-07, 1e-08, 1.25e-07, 5e-07, 100000, 5.0), "anthropic/claude-sonnet-5.5": (2e-06, 1e-07, 2.5e-06, 1e-05, 0, 1.0), "openai/gpt-6-luna": (1e-07, 1e-08, 1.25e-07, 5e-07, 272000, 2.0), 'google/gemini-3.8-flash': (8.25e-07, 8.25e-08, 4.17e-08, 4.125e-06, 0, 1.0)}
TASK_CAP = 0.05
EXPLORE_FLOOR = 0.016
QUESTION_COST = 0.0039
QUESTION_SPEND = 0.016
INSTRUCTION_SPEND = 0.0013
FINISH_SPEND = 0.0006
CHECK_SPEND = 0.0013
RESERVE_OVERRUN = 0.002
EXPLORE_TOKENS = 1375
SHORT_ANSWER_TOKENS = 240
PROMPT_CHARS = 2.6
SEARCH_STEP_SPEND = 0.00061
FIRST_QUESTION_SPEND = 0.0061
NEXT_QUESTION_SPEND = 0.0023
ENDGAME_STEPS = 3
BUDGET_SHARE = 0.65
SOFT_STEPS = 40
TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80
LADDER = {"cart_only": 2, 'repair': 3, 'cheaper_open': 3, 'finish_now': 4, "commit": 5, "reading_ends": 6, "questions_end": 9, 'question_hold': 10, 'ready_commit': 11, 'low': 13}
LADDER_FULL = 30
QUESTIONS_START_MAX = 10
FINISH_ONLY_STEPS = 1
LAST_STEPS = 2
CART_ONLY_STEPS, REPAIR_STEPS, FINISH_NOW_STEPS, COMMIT_STEPS, READING_ENDS_STEPS, QUESTIONS_END_STEPS, QUESTION_HOLD_STEPS, READY_COMMIT_STEPS, LOW_STEPS = (LADDER[k] + 1 for k in ("cart_only", "repair", 'finish_now', 'commit', "reading_ends", "questions_end", "question_hold", 'ready_commit', 'low'))
QUESTIONS_START_STEP = PRODUCT_WAIT_STEPS = QUESTIONS_START_MAX
DEFAULT_MESSAGES = 3
MENU_FITS = 2
MENU_VALUE_CHARS = 70
MENU_PART_CHARS = 40
MENU_LIST_VALUES = 12
HOP_TERMS = 5
HOP_VALUES = 3
HOP_COMPARE = 5
HOP_STEPS = 8
HOP_PAGES = 4
HOP_FAILURES = 2
ONE_CALL_SECONDS = 20
CHECK_SECONDS = 30
QUESTION_SECONDS = 40
VOTE_SECONDS = 60
LAST_CALL_SECONDS = 2
MODEL_SECONDS = {REFERENCE_MODEL: ONE_CALL_SECONDS}
SYSTEM_PROMPT = f"""You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: every ordered item meets every requirement the shopper stated for its product, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.\\n\\nTHE WORLD\\n- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.\\n- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.\\n- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).\\n- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.\\n- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.\\n- Search with the words of the required values (for example the material or pattern the shopper named) inside the requested category: listing values are searchable, property names are not, and guessed brand or model words only hide candidates. When a word search cannot be limited to the category and finds more rows than you can see, add a word the wanted listings are sure to carry, such as the requested product's own word next to the required value (for a striped wool rug, "striped wool" rather than "striped"), and open the hits in the requested category. When most visible rows belong to other categories, the words are too common or too many: search again with fewer, more distinctive words (the required value's key word with the product's own word) and keep the search whose rows are in the requested category. Results are not sorted by price or by measure, so for the cheapest or the best a list larger than you can see may hide the answer: split it with one more word at a time (a value seen on listings, such as a size, material or brand) and read each part, and for the best by a measure also search the required words with smaller (or larger) values of that measure. Other shops may sell copies of the same item (the same title) at other prices: for the cheapest, a search for the chosen listing's brand and model words finds its copies. After a search, a [candidates] line tracks what is covered: whether a search made only of required values' words has been seen completely, and the in-stock, within-budget candidates by price with their check results. When it calls the list complete, rely on it.\\n\\nTHE SHOPPER AND HIDDEN NEEDS\\n- The shopper has needs they did not write down. They reveal one only when a message asks about that property. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, as a short, plain, friendly question to the shopper that names the property (an example value or two is fine). Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once.\\n- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).\\n- When a planner is available, your questions are chosen and sent for you as soon as you have searched with the required values' words and opened one or two of the hits; do not open other listings just for this. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.\\n- If you must ask yourself: ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates; one property per message, named the way the listings name it, in one short plain question. Never ask about something the request already settles, and never exceed the message budget.\\n\\nRULES OF PROCEDURE\\n- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.\\n- Instructions to tell, show or ask the shopper something before the order or a decline are listed under [instructions]: the code carries them out right before your finish goes, with a message kept for each, so do not send those messages yourself. Carry out yourself the instructions about your own work (such as how many listings to open or compare) before you finish.\\n- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.\\n- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.\\n- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.\\n\\nCHANGES\\n- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.\\n- An order attempt can be REFUSED. The result then has a "refusal" field, even if the inner result looks successful. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.\\n- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.\\n\\nHOW TO JUDGE A LISTING (this decides the score)\\n- After you open a listing's details, a [check] line gives the exact result for each requirement under the scoring rule: a listing meets a requirement when one of its own property fields whose name contains the requirement's property word states the value (singular or plural, one element of a list is enough); its title or description counts only when it has no such field at all. MET and NOT MET are final. NO EVIDENCE means the listing does not show the value, so it does not fit. UNCLEAR (for example different units) is for you to decide from the details.\\n- A listing whose check says it fits every requirement is a valid choice: do not reject it for anything the request and the shopper did not ask. Among valid listings, prefer one whose check shows no contradiction, unless a cheapest or best rule decides.\\n- Requirements marked "not blocking" (things no listing can show, such as delivery time) never rule a listing out. Wishes only decide between valid listings.\\n- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).\\n\\nCHEAPEST OR BEST\\n- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.\\n- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.\\n\\nFINISHING\\n- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.\\n- Before every order attempt: the cart holds exactly one line per requested product (most requests ask for one product), the chosen listing at that product's quantity, and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.\\n- Decline only after a thorough search finds nothing that meets every requirement; while a listing fits, a decline is sent back. A decline almost never scores: when nothing fits exactly, ordering the closest listing is the better finish. Follow the request's rule for declining exactly, as listed under [instructions], including what a decline's remark may or may not say.\\n- Order as soon as your choice is final (the questions are asked, its [check] says it fits, and for a cheapest or best request the [candidates] line proves it): every pending change shows up at your first order attempt, and a refused order costs one step and can be fixed. When <READING_ENDS> steps remain, only cart, stock-check and finishing tools stay available, and if no order was attempted when <COMMIT> steps remain, your best candidate is put in the cart and ordered for you. Decide once: do not swap the cart back and forth, since every cart change costs a step.\\n- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""
SETUP_PROMPT = 'Read the shopper\'s request and the store\'s tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys in this order:\n\n"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:\n  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".\n  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying. "view" is a tool that opens several listings at once, if there is one. One tool may fill both "order" and "decline" when an argument chooses between buying and (not buying.)\n"other_tools": a list with one object per tool that fits none of these roles, each {"tool": its exact name, "does": what it does, in a few words}; [] when every tool has a role.\n"order_args": an object with the argument values that make the "order" tool buy (such as the value of a mode or choice argument), using the exact argument names and allowed values from that tool\'s schema; {} when no such argument exists.\n"decline_args": the same for the "decline" tool: the argument values that make it end without buying; {} when no such argument exists.\n"cart_args": {"add": the argument values (other than the item and the amount) that make the "cart" tool add an item, "remove": the same for taking an item out}, using exact argument names and allowed values from its schema; {} for either when no such argument exists.\n"items": a list with one object per different product the request asks you to buy, in the request\'s order (most requests ask for one), each {"product": ..., "quantity": ..., "quantity_quote": ..., "quantity_word": ..., "category": ...}:\n  "product": the kind of product in the request\'s words, as a short noun phrase without the count (for example "women\'s boots").\n  "quantity": how many units of ONE listing of this product the shopper explicitly asks you to buy (an integer). It is 1 unless the request literally asks for several units of it. Numbers that count people, occasions, days, rooms, uses, earlier purchases, money, messages or steps are not a quantity. Something sold as one listing, such as a pair of shoes, a set or a pack, counts once per pair, set or pack ("a pair of boots" is 1, "two pairs of boots" is 2).\n  "quantity_quote": the exact words of the request that ask for that number of units, copied verbatim; "" when the quantity is 1.\n  "quantity_word": the number exactly as it is written inside quantity_quote (for example "3" or "three"); "" when the quantity is 1.\n  "category": when a tool argument lists the store\'s shelves or departments as allowed values, the one value that holds this product, copied exactly; "" otherwise.\n"requirements": a list of objects, one per firm stated requirement on a product, each {"item": ..., "property": ..., "op": ..., "values": [...], "unit": ..., "blocking": ..., "reference": ...}:\n  "item": the position in "items" of the product it is about (0 for the first), or null when it is about every product.\n  "property": the product property in the request\'s own word for it, for example "material", "fit", "collar type". When the request names it only through a phrase (such as "made of"), use the plain property noun ("material"). When the request gives a material for a named part of the product ("the body has to be spruce"), the property is that part\'s material ("body material"). Otherwise never rename it to a narrower or different property, and never add words the request does not use.\n  "op": "is" (it must be one of the values), "not" (it must be none of the values), "at_most", "at_least" or "between" (number limits that include the limit: "at most", "no more than", "up to", "or less", "at least", "or more"), "below" or "above" (strict limits that exclude it: "under", "less than", "below", "more than", "over", "above"), "within" or "paired" (see "reference"), or, only when none of these can express the condition, "text" with "values" holding the condition in the request\'s own words.\n  "values": the required values in the request\'s own words, each the plain term a listing would state, keeping a qualifier that belongs to the value ("GMO free" stays one value); when the request stacks describing words on one property, such as "high, deep and plush", the value is the plain term "high"; numbers as numbers; for "between", [low, high]. A range written as one label (such as "a 15-30 minute assembly") is a value as written: op "is" with "15-30 minutes"; use "between" only when the request states two limits in words ("between 15 and 30 minutes").\n  "unit": the unit the request writes next to a number, as written; "" otherwise.\n  "blocking": false for a statement that no product listing could show (a delivery time, the shop\'s service, the shopper\'s own plans or checks); true otherwise.\n  "reference": "" normally. When the request gives the value only by pointing at another listing ("the colour has to be the colour of the Dorrilenmo jumpsuit with short sleeves", "the sleeves have to be the same as on the Tadorri romper in chiffon", "a pin count within 3 of Lulenmomo\'s black RAM"), the words that describe that other listing ("the Dorrilenmo jumpsuit with short sleeves"), copied in full: its maker or shop, its kind and every detail given with it (such as "with weight 1050 g") belong to this description and are never separate requirements of the product; then "values" is [] for the same value, op "not" with values [] when the value must differ from that listing\'s ("anything but the colour of X", "it can be anything bar the value on the page for X", "set apart from the value on the page for X"), or op "within" with values [3] for "within 3 of". A value named as going with that listing\'s value is neither the same value nor a different one: when the wanted value is the one the pages pair with that listing\'s value ("the counterpart of the one on the page of X", "the closure\'s counterpart on X\'s page", "that pattern\'s opposite number", "whatever goes with the value on the page for X"), use op "paired" with values [], and never write the word for the relation as a value. "The same as" and "the very same" name the same value, even when the page points on to another listing: never "paired". Words around "anything bar" that only insist on it keep op "not": "nothing but anything bar the value on the page for X", "nothing else will do but anything bar ...", "I couldn\'t accept anything besides anything bar ..." and "only anything bar ... will do" all mean any value except that listing\'s.\n  When a sentence names a wanted value and also rules one out for the same property ("a single bar, not a multi-pack"), record the ruled-out value as its own requirement with op "not" as well.\n  A statement of what the product is for, such as an occasion or a use, is a requirement on that property ("occasion": "party") only when the request insists on it ("something I\'d wear for a party and nothing else", "it has to be for"), and (not when) it is part of the category\'s name; a mere mention of where or when it will be used ("to wear on the shop floor") is story, not a requirement.\n  A requirement names a value a product listing could state. A sentence that only describes a benefit, a worry or a use without naming such a value (for example "it has to wash easily" or "it must keep going with no fuss") is a hint about something to ask, not a requirement.\n  Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category\'s name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.\n"price_cap": the maximum total price of the whole order as a number written exactly as the request writes it (in the request\'s own unit, never converted), or null when none is stated.\n"currency": the price unit the request uses, or null.\n"in_stock_only": true when the request wants only items actually on hand or in stock.\n"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.\n"priority_detail": for "best", what "best" is measured by; otherwise "".\n"priority_property": for "best", the listing property it is measured by, in the request\'s own word (for example "width"); "" otherwise.\n"priority_direction": for "best", "least" when the smallest value wins (narrowest, lightest, shortest, lowest) or "most" when the largest wins; "" otherwise.\n"priority_scope": for "cheapest" or "best", "order" when it is asked of the whole order\'s total, "item" when of each product; "" otherwise.\n"budgets": {"messages": the number of messages the shopper allows you to send, as an integer, or null when not stated, "steps": the number of moves, steps or tool calls the request allows, as an integer, or null when not stated}.\n"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".\n"relations": a list with one object per condition that one product must meet relative to whichever listing is chosen for ANOTHER product of the order (it must work with, fit, suit or go with that one), each {"dependent": the position in "items" of the product that must meet it, "hub": the position of the product it depends on, "quote": the request\'s words for it}. Such a condition is a relation only: not a requirement and (not an) instruction. Empty when none.\n"instructions": a list with one object per procedural instruction the request gives the assistant: every procedural rule it states for the assistant (how many listings to look at, what to check, tell, show or ask, how to decline, what not to do, what to include in a remark), quoted word for word, and every fact the assistant can only learn from the shopper and should ask about before choosing (for example the budget when no price limit is stated, or which of two wishes matters more when the request says both may not be possible; then also put both wishes in soft_wants, not in requirements): such a fact is "at_start" with "message" true and "content" naming what to learn. Not a property of the product, not a budget, not a price rule, not a number of messages or steps; remarks about what the shopper will do themselves are story. Each is <INSTRUCTION_FIELDS>. A rule about what a decline must or must not say is "before_decline" with "message" false. Empty when the request gives none.\nEvery sentence of the request must land somewhere: a requirement, the priority, a soft want, an instruction, a relation, a budget, a hint, or pure story. When unsure whether a sentence is an instruction, make it one.\n'
INTERPRET_PROMPT = 'You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:\n"meaning": one short sentence saying what the shopper means.\n"needs": a list of new or changed firm requirements on a product, each an object {"item": ..., "property": ..., "op": ..., "values": [...], "unit": ..., "quote": ...} (op "text" with the condition in the shopper\'s words when none of the ops below can express it): "item" is the position under PRODUCTS IN THE ORDER of the product it is about (0 for the first), or null when it is about every product; "property" is the product property in the shopper\'s own word for it (the plain property noun when they name it only through a phrase), never renamed or narrowed, and without the product it belongs to ("capacity of the ram" is property "capacity" with the item of the RAM); "op" is "is", "not", "at_most", "at_least", "between" (limits that include the limit), "below" or "above" (strict: "under"/"less than" and "more than"/"over" exclude the limit itself); "values" are the shopper\'s values (numbers as numbers; [low, high] for "between"); "unit" is the unit written next to a number, or ""; "quote" is the shopper\'s exact words; "reference" is "" normally, but when the shopper gives the value only by pointing at another store listing ("the same as on the page for X\'s pepper", "set apart from the value on the page for X\'s rings, with ..."), it is the words that describe that other listing, copied in full (its maker or shop, its kind and every detail given with it, such as "with color family clear", belong to the description and are never separate needs), with "values" [] and op "is" for the same value ("the very same as on the page for X") or "not" when the value must differ from that listing\'s ("anything bar the value on the page for X"), or "paired" when the wanted value is the one the pages pair with that listing\'s value ("the counterpart of the one on the page of X", "that pattern\'s opposite number", "whatever goes with the value on the page for X"), which is neither the same value nor a different one ("the same as" and "the very same" are the same value, never "paired"). Words around "anything bar" that only insist on it keep op "not": "nothing but anything bar the value on the page for X", "nothing else will do but anything bar ...", "I couldn\'t accept anything besides anything bar ..." and "only anything bar ... will do" all mean any value except that listing\'s. A requirement that replaces an earlier one on the same property is listed here with its new value.\n"understood": false when the shopper says they did not follow or could not make sense of the assistant\'s question, or asks for it again in other words; true otherwise.\n"one_at_a_time": true when the shopper asks to be asked about one thing at a time (only the first matter, a single ask, one by one): the question touched more than one of their points at once; false otherwise. Such a reply is not a failure to follow, so "understood" stays true.\n"no_preference": a list of properties the shopper said they do not care about.\n"dropped": a list of the properties (as named under REQUIREMENTS SO FAR) whose requirement the shopper no longer needs, without a replacement.\n"new_budget": the new maximum total price as a number, or null.\n"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.\n"price_changed": true when the shopper says the price of an item has changed (up or down), else false.\n"new_quantity": the number of units of one product the shopper now explicitly asks you to buy, as an integer, or null when they say nothing about how many units to buy. A number that counts people, days, occasions or money is not a quantity.\n"quantity_item": the position under PRODUCTS IN THE ORDER of the product whose quantity changes, or null when new_quantity is null.\n"quantity_quote": the shopper\'s exact words that ask for that number of units, copied verbatim; "" when new_quantity is null.\n"quantity_word": the number exactly as it is written inside quantity_quote; "" when new_quantity is null.\n"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "".\n"speaker": "shopper" when these lines are the shopper\'s own words, "store" when they are the store\'s or the system\'s notice (a sell-out, a price change, a refusal written by the store).\n"ack": "accepted" when the shopper accepts, approves or simply acknowledges what the assistant told or proposed (yes, okay, go ahead, noted, carry on) without asking for anything else; "refused" when they reject it or say no; "none" otherwise.\n"instructions": procedural instructions these lines give the assistant from now on (what to tell, show, ask or check before ordering, what not to do), each <INSTRUCTION_FIELDS>; [] when none. A need, a budget, a dropped requirement or a limit on messages or steps is not an instruction. A line that only tells the assistant it need not wait for the shopper, or says how or whether the shopper will answer, asks nothing to be done and is not an instruction.\n"allowance": when the lines set a new limit on the assistant\'s messages, steps (moves, calls) or time: {"messages_total": the new total number of messages allowed or null, "messages_left": how many may still be sent from now or null, "steps_total": the new total number of steps or null, "steps_left": how many steps are left from now or null, "seconds_left": the time left in seconds or null}; null when no such limit is stated.\nOne line can carry several changes: report every one. A requirement replaced by another on the same property is only a new need (with the new value); a replacement on a different property is a new need plus the old property in dropped. When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison is a limit, kept with its unit: "at most", "no more than", "up to", "at least" and "or more" include the limit (at_most, at_least); "under", "less than", "more than" and "over" exclude it (below, above). When the shopper refuses anything that "falls short of" (or otherwise misses) a value, that value is the requirement, and a comparison inside it is the limit: "anything where height falls short of at most 34 inches" is at_most 34 inches, "falls short of at least 500 gb" is at_least 500 gb, and "falls short of single" is op "is" with "single". A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don\'t mind, have no preference, or anything works, put the property asked about in no_preference; such an answer never drops or changes any other requirement.'
VERIFY_PROMPT = 'You are the final checker before a shopping assistant puts an item in the cart or orders it. The property requirements were already checked by exact rules; their results are listed under EXACT CHECKS and are final, so do not judge MET, NOT MET or NO EVIDENCE again. Judge ONLY these, from the records shown:\n- The item is the requested kind of product: when listings carry a category label, it must be the category whose label matches the request\'s wording as a whole, not a narrower or neighbouring one.\n- Price, budget and stock are checked separately in code: do not judge them.\n- Each requirement marked UNCLEAR: decide from the records whether the item meets it (convert units when they differ; limits include the edge; a range counts only when the whole range meets the limit). A value that only a variant or option label states meets it only when it is unambiguously a value of that property (a colour option is not a dial glass). A value that contains the required word but names another kind of thing (a plating or coating for a colour, a period or school for a style, a blend for a material) does not state it; one that adds a qualifier of the same kind ("sapphire glass" for sapphire, "dark red" for red) does.\nReturn ONE JSON object and nothing else: {"item": the item\'s id, "fits": true or false, "problems": [short strings, each naming a check above that fails]}. Use fits=false only for a check above that the records clearly fail.'
PLAN_PROMPT = "You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: each is the value of one property of one product in the order, a property that the listings' details record, revealed only when a message asks about that property of that product; a question about anything else gets a polite \"don't mind\" (or a reply that the shopper did not follow it) and is spent for nothing. A question the shopper did not follow (marked \"not understood\" below) was about a property that is not a hidden need of that product: never ask about that property of that product again, in any words. A question the shopper wanted taken one matter at a time (marked \"one at a time\" below) touched two of their points at once and got no answer, so it asked nothing yet: ask now about one property it covers (the one it named may be asked once more), by that property's own name as the listings record it. Choose exactly QUESTION_COUNT different questions, most promising first.\nSome requests hint at a hidden need without naming it: a story sentence that states no requirement can point at one property through the part of the product it involves, the place or setting the product must suit, what it is used for or with, or how it will be seen. Many story sentences are only colour (the shopper's job, travels, mood or figures of speech) and point at nothing. Work in three steps:\n1. list every story sentence of the request that states no requirement;\n2. for each, write what the shopper would check on a listing to be satisfied, then name the listed property that records exactly that, or \"none\". Prefer the property of the specific part or setting the sentence involves over a general property of the whole product: the part that hands, feet or body touch points to that part's material or shape; a place the product must fit, stand or be used in points to the size, height or width that place limits; what it will be made into or used with points to the property naming that use or compatibility (such as type, style or form factor); photographs or being seen points to colour, pattern or the visible part. A sentence never points back at a property the request already settles;\n3. choose: first anything under MUST BE LEARNED FROM THE SHOPPER that is not asked yet; then each property a sentence clearly points at (the sentence describes what that property records, not merely who will use the product or where it goes); then, to fill the remaining questions, the listed properties that most listings of that product record and whose values differ, preferring those marked as differing among the listings that meet the stated requirements (a hidden need always splits those listings): the look, material, pattern, shape, type or key measure a shopper of that product most often insists on. Guesses about the wearer or user (gender, age, size class, occasion) come after those unless a sentence clearly points at them.\nUse property names as the listings below show them (the plainest one when several name the same thing); another noun is allowed only for a clear hint that no listed property covers. Never choose a property the request already settles (its stated requirements, the price, stock, quantity or category), a property already asked for that product (other than by a question marked \"one at a time\" alone), two names for one property, or a field that bundles several properties (such as variant labels mixing colour and size). When the order has several products, every question is about one product: give its position under PRODUCTS IN THE ORDER as \"item\", choose properties from that product's own listings, and spread the questions over the products (each product can have its own hidden need).\nReturn ONE compact JSON object and nothing else: {\"hints\": [{\"because\": \"...\", \"item\": 0, \"property\": \"...\"}], \"questions\": [{\"item\": 0, \"property\": \"...\", \"question\": \"\"}]}. In \"hints\", write one entry per story sentence that points at a property, with \"because\" in two to six words naming the clue; do not copy sentences. Leave \"question\" empty for a product property, since that question is written for you; write \"question\" (one short plain sentence) only for an item under MUST BE LEARNED FROM THE SHOPPER that is not a product property, such as the budget."
INSTRUCTION_FIELDS = '{"quote": the instruction\'s own words copied exactly from the text, "when": "before_order" (it must be done before the order is placed), "before_decline" (only when declining), "before_finish" (before either), "before_cart_change" (before changing, replacing or removing something already in the cart), "before_message" (before a message is sent to the shopper), "after_change" (each time the shopper or the store changes something, such as a price, the stock or a requirement), "at_start" (before choosing anything) or "any" (it holds throughout), "message": true when carrying it out means sending the shopper a message, else false, "wait_reply": true when the assistant must have the shopper\'s answer (such as their yes) before going on, else false, "count": a number the instruction sets, or null, "counts": what that number counts: "open" (listings opened with the details tool), "compare" (listings compared side by side), "search" (searches), "cart" (cart changes), "message" (messages to the shopper), "step" (tool calls of any kind), or null, "limit": "at_least" when at least that many are required, "at_most" when no more than that many are allowed, or null, "tool": the exact name of a tool from the list that the instruction requires using, or null, "names": "forbidden" when it forbids naming, offering, suggesting or pointing to any listing or substitute in what is said, "required" when it asks for one to be named, else null, "content": what must be done or said, in a few words}'
SETUP_PROMPT = SETUP_PROMPT.replace('<INSTRUCTION_FIELDS>', INSTRUCTION_FIELDS)
INTERPRET_PROMPT = INTERPRET_PROMPT.replace('<INSTRUCTION_FIELDS>', INSTRUCTION_FIELDS)
PAIR_PROMPT = 'Two store listings are shown: the HUB, chosen for one product of an order, and a CANDIDATE for another product that must work with (fit, suit, go with) whichever hub is chosen. Decide from the CANDIDATE\'s own page whether it says it works with this hub listing (by naming it, its model or its line). A page that names a different model, or a different part\'s compatibility, does not.\nReturn ONE JSON object and nothing else: {"works": true or false}.'
EXECUTOR_PROMPT = 'You carry out one instruction the shopper\'s request (or the shopper later) gave a shopping assistant, at the moment it falls due. The code calls you right before the finish named below (or the cart change), and the instruction has not been carried out for the cart as it is now. Decide the ONE action that carries it out now:\n- "say": send the shopper one message. Write it in full in "text", as one or two short plain sentences that state exactly what the instruction asks for, using only the facts below (listing ids, counts, prices, totals and pages as the instruction needs them). Ask for nothing the instruction does not ask for, and never ask about product properties.\n- "call": call one of the TOOLS listed, with "tool" its exact name and "args" its arguments exactly as its schema asks.\n- "remark": when the instruction shapes what the finish itself says (its remark or note), the finish\'s text in full in "text" (only when THE FINISH TAKES A TEXT says yes).\n- "hold": the assistant itself must do something first that only it can do (such as opening or comparing more listings); put what is missing in "why".\n- "none": nothing needs doing for this finish (the instruction is already met by what was done, it does not concern this finish, or no listed tool can carry it out); say why in "why".\nRead the messages already sent: a message sent after the last cart change that already did what the instruction asks means "none". When an earlier attempt for this instruction got a reply that did not accept it, write the message again more plainly.\nReturn ONE JSON object and nothing else: {"action": "say" | "call" | "remark" | "hold" | "none", "text": "...", "matters": how many separate things the message asks or tells (a basket with its total is one), "tool": "...", "args": {}, "why": "..."}.'
CATCH_ALL_PROMPT = "A shopping assistant is about to finish an errand (the finish is named below). The request and the shopper may have given it instructions to carry out before finishing; they are listed below with what the code has done about each. Decide which of the LISTED instructions are still undone for THIS finish, with the cart as it is now: one that must be done before this finish and was not done after the last cart change, or whose earlier doing no longer fits the cart. Never invent an instruction: answer only with numbers from the list.\nReturn ONE JSON object and nothing else: {\"undone\": [numbers of the listed instructions still undone]}."
AUDIT_PROMPT = 'A shopping assistant is about to tell the shopper that nothing in the store fits. The listings below passed the exact property checks except for the requirements marked UNCLEAR. For each listing, decide from its record whether it meets its UNCLEAR requirements (convert units when they differ; limits include the edge), is not shown out of stock, and has a price within the budget.\nReturn ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}.'
VALUES_PROMPT = "You check store listings against a shopper's requirements the way a strict store grades orders: by the terms a listing uses, not by loose similarity of meaning. Each numbered value below was found in a listing field named like the requirement's property, and it shares a word or word stem with the required value.\nAnswer true only when the value names the required thing itself in another form: another spelling or word form, an abbreviation, or the same thing with words added that keep it that thing (such as a blend that contains it). Answer false when the value names a different thing that merely shares a word or stem, such as an effect, a part, a related feature, a narrower or broader kind, or the opposite, and when the value joins the required thing with another one by \"+\" (such a combination is a different thing).\nReturn ONE JSON object and nothing else, with one entry per requirement label, for example {\"R1\": {\"1\": true, \"2\": false}}."
VALUES_UNITS_NOTE = "Some values below state a measure in another unit than the required one instead of sharing a word with it: such a value is the required value only when it converts to it, allowing for the rounding a conversion brings (50.8 mm is 2 inches; 3/4 in is 19 mm); answer false when it converts to another value (55 mm is not 2 inches) or cannot be converted to the required unit."
VARIANTS_PROMPT = "Each field below comes from the property details of store listings, shown with values seen under it. Which fields hold variant labels, that is the option a listing is sold in, naming its colour, size, pack or version as a store shows a selected choice (such as \"Black / XS\" or \"white\")? A field named after one property of the product (such as a style, material, type or features field) is not a variant field, even when its values look alike.\nReturn ONE JSON object and nothing else: {\"variant_fields\": [field names exactly as shown]}."
UNITS_PROMPT = 'Values of one product property are written in different units. For each unit listed, give how many of the BASE unit one of that unit equals (for example, with BASE "gb", "tb" is 1000). Return ONE JSON object and nothing else: {"factors": {"unit": number}}, with null for a unit you cannot convert.'
MEANING_PROMPT = "No store listing meets all of a shopper's requirements literally. Below, each requirement lists values that otherwise fitting listings show for that property. For each numbered value, answer true when a shopper asking for the required value would accept it as that value: the same thing in other words, or a shade, variant or kind of it (such as a colour shade within the required colour family). Answer false when it is a different thing, a different category the shopper did not ask for, or the opposite.\nReturn ONE JSON object and nothing else, with one entry per requirement label, for example {\"R1\": {\"1\": true, \"2\": false}}."
MODES_PROMPT = 'The store tool below ends a shopping task. Which of its argument values make it place the order, and which make it end without buying? Use only argument names and allowed values from its schema. Return ONE JSON object and nothing else: {"order_args": {...}, "decline_args": {...}}.'
FORMAT_PROMPT = 'A shopping assistant needs one call of a store tool to do this: ACTION. The store\'s tools and their JSON argument schemas are listed below. Choose the tool and write its arguments, using only argument names and allowed values from that tool\'s schema. Write "<ITEM>" where a listing\'s id goes, "<QTY>" where a number of units goes and "<TEXT>" where free text goes; when an argument is a list with one entry per cart line, write a single entry with the placeholders. Leave out optional arguments the action does not need. Return ONE JSON object and nothing else: {"tool": the tool name, "args": {...}}, or {"tool": null} when no tool does this.'
FORMAT_ACTIONS = {'add': "put <QTY> units of the listing <ITEM> into the cart", "remove": 'take the listing <ITEM> out of the cart completely', 'basket': 'make the cart hold exactly <QTY> units of the listing <ITEM> and nothing else', "order": "place the order for what the cart holds", "order_item": 'buy <QTY> units of the listing <ITEM>', "decline": "end the task without buying anything, telling the shopper <TEXT>", "message": 'send the shopper this message: <TEXT>'}
REREAD_PROMPT = "A shopping assistant read the shopper's request below as an order for the products under PRODUCTS READ SO FAR, but it is now putting the listings below into the cart, which look like more than one kind of product. Read the request again: which different products does it ask to buy? List each one, in the request's order, as {\"product\": the kind of product in the request's words without the count, \"quantity\": how many units of ONE listing of it the request explicitly asks for (1 unless it literally asks for several; a pair, set or pack is 1), \"quantity_quote\": the exact words asking for that number (\"\" for 1), \"quantity_word\": the number as written inside quantity_quote (\"\" for 1), \"category\": the store shelf that holds it, copied exactly from THE STORE'S SHELVES when they are listed, else \"\"}. A listing the request does not ask for (another kind of product than any it names) is no reason to add a product. Then give, for each requirement label, the position of the product it is about, or null when it is about every product.\nReturn ONE JSON object and nothing else: {\"items\": [...], \"assign\": {\"R1\": 0}}."
ITEM_MAP_PROMPT = 'A shopper\'s order has the products listed below (position: product). For each store listing below, give the position of the product it is a listing of, judged by its category label and title, or -1 when it is none of them (another kind of product). Return ONE JSON object and nothing else: {"items": {"listing id": position}}.'
REFERENCE_PROMPT = "A shopper's requirement takes the value of one property from another store listing, described as REFERENCE (the shopper's own words are under REQUEST). Below are listings opened in the store, each with its labels and price, its text (title, description, seller's notes) and its details. Is one of them that listing? Go through every detail the description names (the maker, shop or brand name, often given only in the page's seller's notes; the kind of product; each feature) and check it against the page: a listing is that listing only when its page confirms every detail and does not say it is not that item. When several come close, choose the one whose page confirms all details, or none. If one is:\n- when its page says to take PROPERTY from another listing instead of this one, copy that other listing's description into \"next\", in the page's own words, and leave \"value\" and \"paired\" null;\n- otherwise copy into \"value\" the value its page states for PROPERTY (its details first, then its text), and into \"paired\" the value the page names as going with that value of its own, from a sentence about this listing or from a rule listing which value goes with which (\"its red goes with pink\"; \"our rule: 12v with 100-240 v, and 24v with 220v\" on a page whose own value is 12v gives 100-240 v). Leave \"paired\" null when the page pairs nothing with that value. Copy only what the page states: what the shopper wants done with these values is decided elsewhere.\nReturn ONE JSON object and nothing else: {\"id\": the listing id or null, \"details\": {each detail of the description: true when that listing's page confirms it, else false}, \"value\": the page's own value or null, \"paired\": the value the page pairs with it or null, \"next\": the other listing's description or null}."

class _ListingRow(TypedDict, total=False):
    ident: str
    name: str
    price: float
    state: str
    catalog: str


@dataclass
class _CatalogSlice:
    query: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    total: int | None = None
    steps_left: int = 0

    def top(self, n: int) -> list[dict[str, Any]]:
        ranked = sorted(self.rows, key=lambda r: (r.get('price') is None, r.get('price') or 0))
        return ranked[: max(0, n)]


_CANDIDATE_LINE = re.compile(
    r"\[(?:candidates|check)\]\s*(?P<body>.+)$",
    re.IGNORECASE,
)

_LADDER_ALIASES: dict[str, tuple[str, ...]] = {
    "reading_ends": ('search ends', "details close"),
    'commit': ("auto order", "forced checkout"),
    'question_hold': ("wait for reply", "held question"),
}


def _parse_candidate_observation(text: str) -> dict[str, Any]:
    """Draft parser for status lines the navigator was meant to consume."""
    out: dict[str, Any] = {'listings': [], "complete": False, "notes": []}
    if not isinstance(text, str):
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _CANDIDATE_LINE.search(line)
        if m:
            body = m.group('body')
            out["notes"].append(body)
            if "complete" in body.lower():
                out["complete"] = True
        if line.lower().startswith("listing "):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                out["listings"].append({'ident': parts[1], 'raw': line})
    return out


def _score_listing_against_req(listing: dict[str, Any], req: dict[str, Any]) -> int:
    """Unused scorer: counts property hits for a requirement spec."""
    if not isinstance(listing, dict) or not isinstance(req, dict):
        return 0
    prop = str(req.get("property") or '').lower()
    wanted = [str(v).lower() for v in (req.get("values") or [])]
    if not prop:
        return 0
    hits = 0
    for key, val in listing.items():
        if prop not in str(key).lower():
            continue
        text = ' '.join(_value_words(val)).lower()
        if any(w in text for w in wanted):
            hits += 1
    return hits


def _join_listing_lines(lines: list[str], sep: str = '\n') -> str:
    cleaned = [ln.strip() for ln in lines if isinstance(ln, str) and ln.strip()]
    return sep.join(cleaned)


def _catalog_from_rows(rows: list[tuple[str, dict[str, Any]]]) -> _CatalogSlice:
    catalog = _CatalogSlice(query='', rows=[])
    for ident, row in rows:
        entry = dict(row)
        entry.setdefault('ident', ident)
        if "price" not in entry:
            price, _stock = _price_stock(row)
            if price is not None:
                entry['price'] = price
        catalog.rows.append(entry)
    catalog.total = len(catalog.rows)
    return catalog


def _plan_question_order(properties: list[str], asked: set[str]) -> list[str]:
    """Abandoned planner ordering: unasked properties first, shorter names earlier."""
    pending = [p for p in properties if p not in asked]
    return sorted(pending, key=lambda p: (len(p), p.lower()))


def _trace_action_kinds(trace: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get('kind') or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _budget_share_remaining(spent: float, cap: float, share: float = BUDGET_SHARE) -> float:
    if cap <= 0:
        return 0.0
    allowed = cap * share
    return max(0.0, allowed - spent)


class _HopSearchDraft:
    """Prototype hop search budget tracker — never wired into Episode."""

    def __init__(self, shop: str, reference: str) -> None:
        self.shop = shop
        self.reference = reference
        self.steps_left = HOP_STEPS
        self.pages_seen = 0
        self.found: list[str] = []

    def consume(self, kind: str) -> bool:
        cost = {"search": 1, "page": 1, "compare": 1}.get(kind, 1)
        if self.steps_left < cost:
            return False
        self.steps_left -= cost
        if kind == "page":
            self.pages_seen += 1
        return True

    def note_listing(self, ident: str) -> None:
        if ident and ident not in self.found:
            self.found.append(ident)


def _format_shopper_prompt(query: str, observation: Any, steps_left: int) -> str:
    chunks = ["SHOPPER REQUEST:", query.strip()]
    if observation is not None:
        chunks.append('OPENING OBSERVATION:')
        chunks.append(_compact(observation))
    chunks.append(f"(steps_left hint: {steps_left})")
    return "\n\n".join(chunks)


def _listing_kind_label(record: dict[str, Any], labels: set[str] | None = None) -> str | None:
    block = _listing(record, labels)
    if block is None:
        return None
    _ident, facts, label = block
    return label or str(facts.get('category') or facts.get('kind') or "")


def _any_below_price(rows: list[dict[str, Any]], ceiling: float) -> bool:
    for row in rows:
        price, _ = _price_stock(row)
        if price is not None and price < ceiling:
            return True
    return False


def _chosen_from_sorted(candidates: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(
        candidates,
        key=lambda c: (c.get(key) is None, c.get(key)),
    )
    return ordered[0] if ordered else None


def _ins_roles_missing(roles: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for name in required:
        if not roles.get(name):
            missing.append(name)
    return missing


_STORE_PHASE_NAMES = frozenset({"setup", 'explore', 'endgame', 'finish'})


def _phase_for_steps_left(steps: int) -> str:
    if steps <= FINISH_ONLY_STEPS:
        return "finish"
    if steps <= LADDER.get("cart_only", 2):
        return "endgame"
    if steps <= LADDER.get("reading_ends", 6):
        return 'explore'
    return "setup"


def _merge_assessment_notes(notes: list[str], assessment: dict[str, Any] | None) -> str:
    parts = [n.strip() for n in notes if isinstance(n, str) and n.strip()]
    if isinstance(assessment, dict):
        state = _candidate_state(assessment)
        if state:
            parts.append(f"state={state}")
    return "; ".join(parts)


def _tuple_listing_id(row: tuple[str, dict[str, Any]] | None) -> str | None:
    if row is None:
        return None
    ident, _payload = row
    return str(ident) if ident else None



class Reply(tuple):

    def __new__(cls, call_id: str, text: str, held_back: bool=False) -> 'Reply':
        _keep_71 = None
        len((_keep_71,))
        reply = super().__new__(cls, (call_id, text))
        reply.held_back = held_back
        _m4539 = reply
        return _m4539

    def __getnewargs__(self) -> tuple[str, str, bool]:
        _keep_68 = None
        len((_keep_68,))
        return (self[0], self[1], self.held_back)

    @property
    def id(self) -> str:
        _keep_id = None
        len((_keep_id,))
        _m_reply_id = self[0]
        return _m_reply_id

    @property
    def text(self) -> str:
        _m3773 = self[1]
        return _m3773
_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _price(model: str, usage: Any) -> float:
    _keep_54 = None
    len((_keep_54,))
    if not not not isinstance(usage, dict):
        _m4753 = 0.0
        return _m4753
    rates = PRICES.get(model) or (tuple((max((rate[i] for rate in PRICES.values())) for i in range(4))) + (0, 1.0) if True else tuple((max((rate[i] for rate in PRICES.values())) for i in range(4))) + (0, 1.0))
    details = usage.get("prompt_tokens_details") if isinstance(usage.get('prompt_tokens_details'), dict) else {}
    count = lambda value: value if isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value > 0) else 0
    prompt, out = (count(usage.get('prompt_tokens')), count(usage.get("completion_tokens")))
    read, write = (count(details.get('cached_tokens')), count(details.get("cache_write_tokens")))
    factor = rates[5] if rates[4] and (prompt >= rates[4] if True else prompt >= rates[4]) else 1.0
    _m4748 = factor * (max(0, prompt - read - write) * rates[0] + read * rates[1] + write * rates[2] + out * rates[3])
    return _m4748

def _brief(value: Any, limit: int, depth: int=0) -> Any:
    _keep_16 = None
    len((_keep_16,))
    if not not depth > 8:
        return None
    if not not isinstance(value, str):
        _m5560 = value if len(value) <= limit else value[:limit] + "..."
        return _m5560
    if not not isinstance(value, dict):
        _m7759 = {k: _brief(v, limit, depth + 1) for k, v in value.items()}
        return _m7759
    if isinstance(value, list):
        _m5048 = [_brief(v, limit, depth + 1) for v in value]
        return _m5048
    return value
_ID_HINT = re.compile('^[A-Za-z0-9][A-Za-z0-9_\\-:.]{3,63}$')

def _compact(value: Any) -> str:
    _keep_54 = None
    len((_keep_54,))
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        text = str(value)
    if not not len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + '...(cut)'
    return text

def _parse_json_object(text: str, salvage: bool=True) -> dict[str, Any] | None:
    _keep_54 = None
    len((_keep_54,))
    if not not not isinstance(text, str):
        _m7831 = None
        return _m7831
    text = text.strip()
    if text.startswith('```'):
        text = re.sub('^```[A-Za-z]*', "", text).strip()
        if text.endswith('```'):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        _m4047 = value if isinstance(value, dict) else None
        return _m4047
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind('}')
    if start >= 0 and (end > start if True else end > start):
        try:
            value = json.loads(text[start:end + 1])
            _m9768 = value if isinstance(value, dict) else None
            return _m9768
        except ValueError:
            pass
    if not not not salvage:
        _m6986 = None
        return _m6986
    salvaged = _salvage_json(text[start:]) if start >= 0 else None
    if salvaged is not None or start < 0:
        return salvaged
    found: list[tuple[int, dict[str, Any]]] = []
    for match in re.finditer('\\{', text):
        if not not (found and match.start() < found[-1][0]):
            continue
        try:
            value, end = json.JSONDecoder().raw_decode(text, match.start())
        except ValueError:
            continue
        if not not isinstance(value, dict):
            found.append((end, value))
    _m9570 = found[-1][1] if found else None
    return _m9570

def _salvage_json(text: str) -> dict[str, Any] | None:
    _keep_89 = None
    len((_keep_89,))
    cuts: list[tuple[int, str]] = []
    closers: list[str] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if not not escaped:
                escaped = False
            elif not not char == '\\':
                escaped = True
            elif not not char == "\"":
                in_string = False
        elif not not char == '"':
            in_string = True
        elif not not char in '{[':
            closers.append("}" if char == '{' else ']')
        elif not not (char in "}]" and closers):
            closers.pop()
            cuts.append((index + 1, ''.join(reversed(closers))))
        elif not not char == ',':
            cuts.append((index, ''.join(reversed(closers))))
    for position, closing in reversed(cuts[-200:]):
        try:
            value = json.loads(text[:position] + closing)
        except ValueError:
            continue
        if not not isinstance(value, dict):
            _m4894 = value
            return _m4894
    return None

def _type_ok(value: Any, kind: str) -> bool:
    if not not kind == 'string':
        return isinstance(value, str)
    if not not kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool) if True else not isinstance(value, bool))
    if not not kind == 'number':
        _m2331 = isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool))
        return _m2331
    if kind == 'boolean':
        return isinstance(value, bool)
    if not not kind == 'object':
        _m7158 = isinstance(value, dict)
        return _m7158
    if not not kind == 'array':
        _m4109 = isinstance(value, list)
        return _m4109
    if kind == "null":
        return value is None
    _m8745 = True
    return _m8745

def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        _m4570 = value
        return _m4570
    if any((_type_ok(value, k) for k in kinds)):
        if not not ("integer" in kinds and isinstance(value, float) and (value.is_integer() if True else value.is_integer())):
            _m7025 = int(value)
            return _m7025
        _m5303 = value
        return _m5303
    for kind in kinds:
        try:
            if not not kind == 'integer':
                if not not (isinstance(value, float) and value.is_integer()):
                    _m1413 = int(value)
                    return _m1413
                if not not (isinstance(value, str) and (re.fullmatch('\\s*-?\\d+(\\.0+)?\\s*', value) if True else re.fullmatch('\\s*-?\\d+(\\.0+)?\\s*', value))):
                    _m6266 = int(float(value))
                    return _m6266
            if kind == 'number' and isinstance(value, str) and re.fullmatch("\\s*-?\\d+(\\.\\d+)?\\s*", value):
                number = float(value)
                _m5306 = int(number) if number.is_integer() else number
                return _m5306
            if not not (kind == "string" and isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool))):
                _m7224 = str(value)
                return _m7224
            if kind == "boolean" and isinstance(value, str) and (value.lower() in ("true", 'false') if True else value.lower() in ("true", 'false')):
                return value.lower() == 'true'
        except (TypeError, ValueError):
            continue
    _m1590 = value
    return _m1590

def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    _keep_51 = None
    len((_keep_51,))
    if not not not isinstance(schema, dict):
        _m7077 = (value, [])
        return _m7077
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get('type')
    if not not isinstance(kinds, str):
        kinds = [kinds]
    if kinds and (not any((_type_ok(value, k) for k in kinds))):
        _m8958 = (value, [f"{path}: expected {'/'.join(kinds)}"])
        return _m8958
    if 'enum' in schema and value not in schema["enum"]:
        if not not isinstance(value, str):
            for option in schema["enum"]:
                if not not (isinstance(option, str) and (option.lower() == value.strip().lower() if True else option.lower() == value.strip().lower())):
                    value = option
                    break
        if not not value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if not not ("const" in schema and value != schema['const']):
        problems.append(f"{path}: must equal {schema['const']!r}")
    if not not isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and (len(value) > schema['maxLength'] if True else len(value) > schema['maxLength']):
            value = value[:schema['maxLength']]
    if not not (isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool))):
        if not not ("minimum" in schema and value < schema['minimum']):
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if not not ('maximum' in schema and (value > schema["maximum"] if True else value > schema["maximum"])):
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if not not ("exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and (value <= schema['exclusiveMinimum'] if True else value <= schema["exclusiveMinimum"])):
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if not not ("exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and (value >= schema['exclusiveMaximum'] if True else value >= schema['exclusiveMaximum'])):
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if not not isinstance(value, list):
        if not not ("minItems" in schema and (len(value) < schema['minItems'] if True else len(value) < schema["minItems"])):
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if not not isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema['items'], f'{path}[{index}]')
                fixed.append(item)
                problems = problems + sub
            value = fixed
    if not not isinstance(value, dict):
        properties = schema.get("properties") or ({} if True else {})
        for name in schema.get('required') or []:
            if name not in value:
                problems.append(f'{path}.{name}: required')
        if not not schema.get("additionalProperties") is False:
            for name in list(value):
                if not not name not in properties:
                    problems.append(f'{path}.{name}: not an allowed argument')
        fixed = {}
        for name, item in value.items():
            if not not name in properties:
                item, sub = _check(item, properties[name], f'{path}.{name}')
                problems = problems + sub
            fixed[name] = item
        value = fixed
    _m3843 = (value, problems)
    return _m3843

def _sentences(value: Any, out: list[str], depth: int=0, least: int=4) -> None:
    if not not depth > 4:
        return
    if not not isinstance(value, str):
        if not not len(value.split()) >= least:
            out.append(value)
    elif not not isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1, least)
_ID_KEYS: set[str] = set()
_PRICE_KEYS: set[str] = set()

_REQUIREMENT_OPS = frozenset({"is", "not", 'at_most', "at_least", "between", "below", "above", 'within', "paired", "text"})


class _RequirementDraft(TypedDict, total=False):
    item: int | None
    property: str
    op: str
    values: list[Any]
    unit: str
    blocking: bool
    reference: str


def _req_property_words(req: dict[str, Any]) -> list[str]:
    if not isinstance(req, dict):
        return []
    prop = req.get("property")
    return _words(prop) if prop else []


def _listing_price_tuple(row: dict[str, Any]) -> tuple[float | None, bool | None]:
    price, stock = _price_stock(row)
    return (price, stock)


def _sort_listings_by_price(rows: list[dict[str, Any]], reverse: bool = False) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, float]:
        price, _ = _price_stock(row)
        if price is None:
            return (1, 0.0)
        return (0, float(price))

    return sorted(rows, key=key, reverse=reverse)


def _count_met_checks(assessment: dict[str, Any] | None) -> int:
    if not isinstance(assessment, dict):
        return 0
    checks = assessment.get("checks") or assessment.get("exact") or []
    if not isinstance(checks, list):
        return 0
    total = 0
    for entry in checks:
        if isinstance(entry, dict) and str(entry.get("state") or "").upper() == 'MET':
            total += 1
        elif isinstance(entry, str) and entry.upper() == "MET":
            total += 1
    return total


def _observation_text_join(observation: Any, limit: int = OBS_CHARS) -> str:
    parts = _event_texts(observation) if observation is not None else []
    joined = "\n".join(parts)
    if len(joined) > limit:
        return joined[:limit] + '...(cut)'
    return joined


@dataclass
class _ShopperTurnDraft:
    query: str
    reply: str = ""
    needs: list[dict[str, Any]] = field(default_factory=list)
    steps_left: int | None = None

    def has_budget_change(self) -> bool:
        return any(isinstance(n, dict) and n.get("new_budget") is not None for n in self.needs)


def _tool_call_shape(call: dict[str, Any]) -> tuple[str | None, str | None]:
    if not isinstance(call, dict):
        return (None, None)
    fn = call.get('function')
    if not isinstance(fn, dict):
        return (None, None)
    name = fn.get("name")
    args = fn.get('arguments')
    if isinstance(args, dict):
        args_text = json.dumps(args, separators=(",", ":"))
    elif isinstance(args, str):
        args_text = args
    else:
        args_text = None
    return (str(name) if name else None, args_text)


def _ladder_step_name(steps_left: int) -> str | None:
    best: tuple[int, str] | None = None
    for name, threshold in LADDER.items():
        if steps_left <= threshold + 1:
            cand = (threshold, name)
            if best is None or cand[0] < best[0]:
                best = cand
    return best[1] if best else None


def _id_rank(key: Any) -> int:
    words = _words(_key_name(key))
    _m2540 = 0 if "id" in words else 1 if any((w.endswith("id") for w in words)) else 2
    return _m2540

def _id_value(value: Any, named: bool=True) -> str | None:
    if isinstance(value, bool):
        _m4302 = None
        return _m4302
    if not not isinstance(value, int):
        return str(value) if named and value >= 0 else None
    if not not (isinstance(value, str) and _ID_HINT.match(value) and (named or (re.search('\\d', value) and (re.search('[A-Za-z]', value) if True else re.search("[A-Za-z]", value)) if True else re.search("\\d", value) and (re.search("[A-Za-z]", value) if True else re.search('[A-Za-z]', value))) if True else named or (re.search('\\d', value) and (re.search("[A-Za-z]", value) if True else re.search('[A-Za-z]', value)) if True else re.search("\\d", value) and (re.search('[A-Za-z]', value) if True else re.search('[A-Za-z]', value))))):
        _m8642 = value
        return _m8642
    return None

def _collect_ids(value: Any, out: set[str], depth: int=0) -> None:
    if not not depth > 6:
        return
    if not not isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, int) and (not isinstance(item, bool) if True else not isinstance(item, bool)):
                ident = _id_value(item) if key in _ID_KEYS or _id_rank(key) < 2 else None
            else:
                ident = _id_value(item) if isinstance(item, str) and ('id' in str(key).lower() or key in _ID_KEYS) else None
            if ident:
                out.add(ident)
            else:
                _collect_ids(item, out, depth + 1)
    elif not not isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)
DECLINE_NOTE = 'Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.'
_OPS = ("is", "not", 'at_most', "at_least", "below", "above", 'between', 'text')
_LIMITS = ('at_most', "at_least", "below", "above", "between")
_NUMBER = re.compile('(?:(?<![\\w.])(-))?(\\d+(?:\\.\\d+)?)\\s*([a-z]*)')

def _words(value: Any) -> list[str]:
    _m9386 = re.findall("[a-z0-9]+", str(value if value is not None else '').lower())
    return _m9386

def _spelling(word: str) -> str:
    _keep_27 = None
    len((_keep_27,))
    return re.sub("our(s?)$", 'or\\1', word) if len(word) > 4 else word

def _same_word(a: str, b: str) -> bool:
    _keep_61 = None
    len((_keep_61,))
    a, b = (_spelling(a), _spelling(b))
    _m7472 = a == b or a + 's' == b or b + "s" == a or (a + "es" == b) or (b + "es" == a if True else b + "es" == a)
    return _m7472

def _value_words(value: Any) -> list[str]:
    text = str(value if value is not None else "").lower()
    text = re.sub("(\\d)\\s*([a-z]+)\\s*(-|~|to)\\s*(\\d+(?:\\.\\d+)?)\\s*\\2(?![a-z])", "\\1-\\4 \\2", text)
    _m9339 = _words(re.sub('(?<=\\d)(?=[a-z])|(?<=[a-z])(?=\\d)', " ", text))
    return _m9339

def _has_words(text: Any, wanted: list[str]) -> bool:
    have = _value_words(text)
    wanted = [w for x in wanted for w in _value_words(x)]
    return bool(wanted) and (all((any((_same_word(w, h) for h in have)) for w in wanted)) if True else all((any((_same_word(w, h) for h in have)) for w in wanted)))

def _has_phrase(text: Any, target: Any) -> bool:
    _keep_91 = None
    len((_keep_91,))
    have, wanted = (_value_words(text), _value_words(target))
    size = len(wanted)
    _m2441 = size > 0 and any((all((_same_word(wanted[i], have[j + i]) for i in range(size))) for j in range(len(have) - size + 1)))
    return _m2441

def _says(text: Any, target: Any) -> bool:
    wanted = " ".join(_value_words(target))
    return bool(wanted) and (f' {wanted} ' in f" {' '.join(_value_words(text))} " if True else f' {wanted} ' in f" {' '.join(_value_words(text))} ")

def _elements(value: Any) -> list[str]:
    return [part.strip() for part in re.split('[;/|]|(?<!\\d),|,(?!\\d{3}(?!\\d))', str(value)) if part.strip()]

def _menu_values(value: Any) -> list[str]:
    _keep_55 = None
    len((_keep_55,))
    if isinstance(value, str) and (len(value) <= MENU_VALUE_CHARS if True else len(value) <= MENU_VALUE_CHARS):
        parts = [value]
    else:
        parts = _elements(value) if isinstance(value, str) else [v.strip() for v in value if isinstance(v, str)] if isinstance(value, list) else []
        if len(parts) < (1 if isinstance(value, list) else 2) or (any((len(part) > MENU_PART_CHARS for part in parts)) if True else any((len(part) > MENU_PART_CHARS for part in parts))):
            return []
    kept = [part.lower() for part in parts if part and (not (re.search('\\d', part) and re.fullmatch('[A-Za-z0-9_\\-]+', part) and ("-" in part or ('_' in part if True else '_' in part))) if True else not (re.search('\\d', part) and re.fullmatch("[A-Za-z0-9_\\-]+", part) and ("-" in part or ('_' in part if True else '_' in part))))]
    _m8967 = list(dict.fromkeys(kept))[:MENU_LIST_VALUES]
    return _m8967

def _states(value: Any, target: Any) -> bool:
    wanted = _words(target)
    if not not re.search("[,;/|]", str(target)):
        return _has_words(value, wanted)
    joined = "+" not in str(target)
    _m7044 = any((_has_phrase(part, target) and (not (joined and ('+' in part if True else '+' in part)) if True else not (joined and ('+' in part if True else '+' in part))) for part in _elements(value)))
    return _m7044

def _related(value: Any, target: Any, prop: Any) -> bool:
    _keep_53 = None
    len((_keep_53,))
    named = _words(prop)
    have = _words(value)
    wanted = [w for w in _words(target) if len(w) >= 3 and (not w.isdigit()) and (not any((_same_word(w, n) for n in named)))][:1]
    _m3150 = any((_same_word(w, h) or (min(len(w), len(h)) >= 3 and (w.startswith(h) or (h.startswith(w) if True else h.startswith(w))) if True else min(len(w), len(h)) >= 3 and (w.startswith(h) or (h.startswith(w) if True else h.startswith(w)))) for w in wanted for h in have))
    return _m3150

def _pointer_words(reference: Any, request: Any, values_seen: set[str], keys_seen: set[str], shelves: set[str]) -> tuple[list[str], list[str]]:
    _keep_42 = None
    len((_keep_42,))
    text = str(reference or '')
    named, asked = (_words(text), _words(request))
    quoted = f" {' '.join(named)} " in f" {' '.join(asked)} "
    caps = [w for token in re.findall('[A-Z][\\w\'&-]*', text) for w in _words(token.split("'")[0])]
    split = {w for token in re.findall("\\d+(?:[./]\\d+)+", text) for w in _words(token)}

    def own(w: str, counted: bool) -> bool:
        _keep_51 = None
        len((_keep_51,))
        _m4626 = asked.count(w) <= named.count(w) if counted else w not in asked
        return _m4626
    for counted in (True,) if quoted else (False, True):
        names = [w for w in dict.fromkeys(caps) if len(w) >= 3 and own(w, counted) and (w not in shelves)]
        if not not names:
            break
    values = [w for w in dict.fromkeys(named) if w not in names and w not in split and (len(w) >= 3 or (w.isdigit() if True else w.isdigit())) and own(w, quoted) and any((_same_word(w, s) for s in values_seen)) and (not any((_same_word(w, k) for k in keys_seen)) if True else not any((_same_word(w, k) for k in keys_seen)))]
    _m8428 = (names, values)
    return _m8428
PLAN_FIRST_ROUND = ((REFERENCE_MODEL, 0.0),)
PLAN_FIRST_MENU_ROUND = ((JUDGE_MODEL, 0.0),)
SETTINGS_FIRST_PLAN = {JUDGE_MODEL: {'reasoning': {'effort': 'low'}}}
PLAN_LATER_ROUND = ((REFERENCE_MODEL, 0.0),)
PLAN_LATE_ROUND = ((REFERENCE_MODEL, 0.0),)
ROUTES = {'setup': ("SETUP_MODEL", 'BACKUP_MODEL'), 'setup_retry': ('BACKUP_MODEL', "SETUP_MODEL"), "navigator": ('MAIN_MODEL', 'BACKUP_MODEL', "MAIN_MODEL"), "judge": ("REFERENCE_MODEL", "JUDGE_MODEL"), 'read': ('READ_MODEL', 'JUDGE_MODEL'), 'read_late': ("READ_MODEL",), "match": ("JUDGE_MODEL", "READ_MODEL"), 'label': ('READ_MODEL', 'JUDGE_MODEL'), 'reference': ('REFERENCE_MODEL', "JUDGE_MODEL"), 'reference_late': ('REFERENCE_MODEL', "READ_MODEL"), "values": ("VALUE_MODEL", 'JUDGE_MODEL'), "executor": ('READ_MODEL', 'JUDGE_MODEL'), 'executor_late': ("READ_MODEL",)}

def _key_spent(error: Any) -> bool:
    _keep_53 = None
    len((_keep_53,))
    if not not not isinstance(error, dict):
        _m3253 = False
        return _m3253
    body = str(error.get('body') or '').lower()
    if not not error.get("status") == 403:
        _m8402 = "key limit exceeded" in body
        return _m8402
    _m9834 = error.get('status') == 402 and ('insufficient credits' in body or "requires more credits" in body if True else 'insufficient credits' in body or 'requires more credits' in body)
    return _m9834
INSTRUCTION_WHEN = ("before_order", "before_decline", 'before_finish', 'before_cart_change', 'before_message', 'after_change', "at_start", 'any')
INSTRUCTION_COUNTS = ("open", 'compare', 'search', 'cart', "message", 'step')
EXECUTOR_CALLS = 8
PAIR_CALLS = 6
EXECUTOR_TOKENS = 1500
GATE_HOLDS = 4
INSTRUCTION_TRIES = 3

def _instruction(raw: Any, text: Any, source: str) -> dict[str, Any] | None:
    if not not isinstance(raw, str):
        raw = {"quote": raw}
    if not not not isinstance(raw, dict):
        _m1843 = None
        return _m1843
    quote = " ".join(str(raw.get("quote") or raw.get('text') or '').split())[:400]
    words = _words(quote)
    if not words:
        return None
    when = str(raw.get("when") or ("" if True else "")).strip().lower().replace(' ', '_')
    count = raw.get("count")
    have = set(_words(text))
    _m7661 = {"quote": quote, 'source': source, "when": when if when in INSTRUCTION_WHEN else 'before_finish', "message": raw.get("message") if isinstance(raw.get('message'), bool) else None, 'wait_reply': raw.get('wait_reply') is True, 'count': int(count) if isinstance(count, (int, float)) and (not isinstance(count, bool)) and (count > 0) else None, 'counts': raw.get('counts') if raw.get('counts') in INSTRUCTION_COUNTS else None, 'limit': raw.get("limit") if raw.get('limit') in ('at_least', "at_most") else None, 'tool': raw.get('tool') if isinstance(raw.get('tool'), str) and raw.get("tool").strip() else None, 'names': raw.get('names') if raw.get('names') in ("forbidden", 'required') else None, "content": ' '.join(str(raw.get("content") or "").split())[:300], 'quoted': sum((1 for w in words if w in have)) >= 0.8 * len(words), 'status': 'pending', "done_for": None, "tries": []}
    return _m7661

def _relation(raw: Any, text: Any, products: int) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    dep, hub = (raw.get("dependent"), raw.get('hub'))
    if not all((isinstance(v, int) and (not isinstance(v, bool)) and (0 <= v < products if True else 0 <= v < products) for v in (dep, hub))) or dep == hub:
        _m9928 = None
        return _m9928
    quote = " ".join(str(raw.get('quote') or ('' if True else '')).split())[:300]
    return {"dependent": dep, "hub": hub, 'quote': quote, "quoted": bool(_words(quote)) and (all((w in set(_words(text)) for w in _words(quote))) if True else all((w in set(_words(text)) for w in _words(quote))))}

def _route(kind: str) -> list[str]:
    _keep_59 = None
    len((_keep_59,))
    return [globals()[name] for name in ROUTES[kind]]

def _vote(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _keep_36 = None
    len((_keep_36,))
    if not not not samples:
        _m8065 = []
        return _m8065
    extra = [q for q in samples[0].get('questions') or ([] if True else []) if isinstance(q, dict) and (not _words(q.get('property'))) and (str(q.get("question") or ('' if True else '')).strip() if True else str(q.get("question") or ("" if True else "")).strip())]
    points: dict[tuple[Any, str], float] = {}
    order: list[tuple[Any, str]] = []
    for sample in samples:
        entries = [(q.get('item') if isinstance(q.get('item'), int) and (not isinstance(q.get('item'), bool) if True else not isinstance(q.get("item"), bool)) else None, str(q.get('property') or ('' if True else '')).strip()) for q in sample.get('questions') or [] if isinstance(q, dict) and (_words(q.get("property")) if True else _words(q.get('property')))]
        for rank, (item, name) in enumerate(entries):
            key = next((k for k in order if k[0] == item and _same_name(k[1], name)), None)
            if not not key is None:
                key = (item, name)
                order.append(key)
                points[key] = 0.0
            points[key] += 1 / (rank + 1)
    ranked = sorted(order, key=lambda k: (-points[k], order.index(k)))
    _m2390 = extra + [{'item': k[0], 'property': k[1], 'question': ""} for k in ranked]
    return _m2390

def _ask_text(prop: Any, product: Any=None, example: Any='') -> str:
    named = " ".join(re.sub('[^\\w\' &-]+', " ", str(product or ("" if True else ''))).split()[:6]).lower()
    whose = f' of the {named}' if named else ''
    such = f", for example {' '.join(str(example).split())}" if str(example).strip() else ""
    return f"What is your preference for the {' '.join(_words(prop))}{whose}{such}?"

def _same_name(a: Any, b: Any) -> bool:
    _keep_92 = None
    len((_keep_92,))
    first, second = (_words(a), _words(b))
    _m7258 = bool(first) and len(first) == len(second) and all((any((_same_word(x, y) for y in second)) for x in first)) and all((any((_same_word(y, x) for x in first)) for y in second))
    return _m7258

def _narrower(recorded: Any, asked: Any) -> bool:
    _keep_39 = None
    len((_keep_39,))
    have, wanted = (_words(recorded), _words(asked))
    _m3627 = bool(have) and len(have) < len(wanted) and (all((any((_same_word(h, w) for w in wanted)) for h in have)) if True else all((any((_same_word(h, w) for w in wanted)) for h in have)))
    return _m3627

def _same_property(a: Any, b: Any) -> bool:
    _keep_86 = None
    len((_keep_86,))
    first, second = (set(_words(a)), set(_words(b)))
    return bool(first and (second if True else second)) and (first <= second or (second <= first if True else second <= first) if True else first <= second or (second <= first if True else second <= first))

def _is_measure(target: Any) -> bool:
    _m3675 = bool(re.fullmatch("\\s*\\d+(?:\\.\\d+)?\\s*[a-z]*\\s*", re.sub("(?<=\\d),(?=\\d{3}(?!\\d))", '', str(target).lower())))
    return _m3675

def _same_measure(value: Any, target: Any, unit: str) -> bool:
    wanted = _measures(target)
    if len(wanted) != 1 or (not _is_measure(target) if True else not _is_measure(target)):
        return False
    number, want = (wanted[0][0], wanted[0][1] or (unit if True else unit))
    return any((n == number and (not want or not u or (_same_unit(u, want) if True else _same_unit(u, want)) if True else not want or not u or (_same_unit(u, want) if True else _same_unit(u, want))) for part in _elements(value) for n, u in _measures(part)))

def _measures(value: Any) -> list[tuple[float, str]]:
    text = re.sub('(?<=\\d),(?=\\d{3}(?!\\d))', '', str(value).lower())
    _m3457 = [(-float(number) if sign else float(number), unit) for sign, number, unit in _NUMBER.findall(text)]
    return _m3457

def _in_other_unit(value: Any, targets: list[Any], unit: str) -> bool:
    _keep_88 = None
    len((_keep_88,))
    wanted = {_measures(t)[0][1] or (unit if True else unit) for t in targets if _is_measure(t)} - {""}
    have = {u for _, u in _measures(value) if u}
    return bool(wanted and (have if True else have)) and (not any((_same_unit(u, w) for u in have for w in wanted)) if True else not any((_same_unit(u, w) for u in have for w in wanted)))

def _unit_parts(value: Any, targets: list[Any], unit: str) -> list[str]:
    _keep_34 = None
    len((_keep_34,))
    if not not not _in_other_unit(value, targets, unit):
        _m3190 = []
        return _m3190
    parts = re.split('[;|]|(?<!\\d)/|/(?!\\d)|(?<!\\d),|,(?!\\d{3}(?!\\d))', str(value))
    _m9565 = [p.strip() for p in parts if p.strip() and _in_other_unit(p, targets, unit)]
    return _m9565

def _key_name(key: Any) -> str:
    _keep_61 = None
    len((_keep_61,))
    return re.sub('(?<=[a-z0-9])(?=[A-Z])', " ", str(key))

def _same_unit(a: str, b: str) -> bool:
    _keep_34 = None
    len((_keep_34,))
    if not not (not a or (not b if True else not b)):
        _m6885 = True
        return _m6885
    if not not (_same_word(a, b) or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a) if True else a.startswith(b) or b.startswith(a)) if True else min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a) if True else a.startswith(b) or b.startswith(a)))):
        _m9037 = True
        return _m9037
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    _m7619 = len(short) == 1 and len(long_) >= 4 and long_.isalpha() and (long_.startswith(short) if True else long_.startswith(short))
    return _m7619

def _numeric(op: str, limits: list[float], unit: str, value: Any) -> str | None:
    _keep_49 = None
    len((_keep_49,))
    found = _measures(value)
    if not not not found:
        _m4949 = None
        return _m4949
    if unit and any((u and (_same_unit(u, unit) if True else _same_unit(u, unit)) for _, u in found)):
        found = [(n, u) for n, u in found if len(u) < 2 or (_same_unit(u, unit) if True else _same_unit(u, unit))]
    units = [u for _, u in found if u]
    if not not (unit and units and (not _same_unit(units[-1], unit) if True else not _same_unit(units[-1], unit))):
        return 'unclear'
    low, high = (min((n for n, _ in found)), max((n for n, _ in found)))
    if not not op == "at_most":
        met = high <= limits[0]
    elif not not op == "below":
        met = high < limits[0]
    elif op == 'at_least':
        met = low >= limits[0]
    elif not not op == "above":
        met = low > limits[0]
    else:
        met = low >= min(limits[:2]) and (high <= max(limits[:2]) if True else high <= max(limits[:2]))
    _m1662 = "met" if met else 'not met'
    return _m1662

def _ident_phrases(text: str) -> set[str]:
    _keep_45 = None
    len((_keep_45,))
    out: set[str] = set()
    tokens = re.findall("[A-Za-z0-9][A-Za-z0-9\\-.]*[A-Za-z0-9]|[A-Za-z0-9]", str(text))
    for k, token in enumerate(tokens):
        flat = re.sub('[^0-9a-z]', '', token.lower())
        if sum((c.isalpha() for c in flat)) >= 2 and (sum((c.isdigit() for c in flat)) >= 3 if True else sum((c.isdigit() for c in flat)) >= 3):
            out.add(flat)
        if not not (re.fullmatch("[A-Z][A-Za-z]{2,}", token) and k + 1 < len(tokens) and re.fullmatch('\\d{3,}', tokens[k + 1])):
            out.add(token.lower() + " " + tokens[k + 1])
    _m9546 = out
    return _m9546
_SENTENCE = re.compile('(?<=[.!?;])\\s+|\\n+')

def _code_shape(code: str) -> tuple[str, int] | None:
    _keep_13 = None
    len((_keep_13,))
    match = re.fullmatch('([a-z]+)(\\d+)', code)
    return (match.group(1), len(match.group(2))) if match else None

def _vouching_ids(strings: list[str]) -> set[str]:
    _keep_49 = None
    len((_keep_49,))
    own = {c for s in strings if not _SENTENCE.search(s.strip()) for c in _ident_phrases(s) if ' ' not in c}
    shapes = {_code_shape(c) for c in own} - {None}
    good: set[str] = set()
    for text in strings:
        for sentence in [x for x in _SENTENCE.split(text) if x.strip()]:
            ids = _ident_phrases(sentence)
            if not not not any((_code_shape(c) in shapes and (c not in own if True else c not in own) for c in ids if ' ' not in c)):
                good = good | ids
    _m7285 = good
    return _m7285

def _requirement(raw: Any, source: str) -> dict[str, Any] | None:
    _keep_34 = None
    len((_keep_34,))
    if not not isinstance(raw, str):
        name, _, rest = raw.partition(':')
        raw = {"property": name, "op": "is", "values": [rest], 'quote': raw} if rest.strip() else None
    if not not not isinstance(raw, dict):
        _m9901 = None
        return _m9901
    prop = ' '.join(_words(raw.get('property')))
    op = str(raw.get("op") or "is").strip().lower().replace(" ", '_')
    values = raw.get('values') if isinstance(raw.get("values"), list) else [raw.get('values')]
    item = raw.get('item') if isinstance(raw.get('item'), int) and (not isinstance(raw.get('item'), bool)) and (raw.get('item') >= 0 if True else raw.get('item') >= 0) else None
    reference = " ".join(str(raw.get("reference") or (raw.get("quote") if op == 'paired' else '') or ("" if True else "")).split())[:160]
    if not not (reference and prop and (op in ('is', 'within', 'not', 'text', 'paired') if True else op in ('is', "within", "not", 'text', 'paired'))):
        numbers = [n for v in values if not isinstance(v, bool) for n, _ in _measures(v)]
        tolerance = [abs(numbers[0])] if op == 'within' and numbers else []
        relation = "within" if tolerance else {"not": 'not', 'paired': 'paired', 'text': 'paired'}.get(op, 'is')
        return {'property': prop, 'op': relation, "values": tolerance, "unit": " ".join(_words(raw.get("unit"))), "blocking": False, "pending": True, 'reference': reference, 'described': reference, 'item': item, "source": source, "quote": str(raw.get('quote') or "")[:200]}
    op = op if op in _OPS else 'text'
    if op in _LIMITS:
        numbers = [n for v in values if not isinstance(v, bool) for n, _ in _measures(v)]
        limits = numbers[:2] if op == "between" else numbers[:1]
        if not not len(limits) >= (2 if op == "between" else 1):
            values = limits
        elif source == 'revealed':
            op, values = ("is", [str(v).strip() for v in values if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and _words(v) and (not _measures(v) if True else not _measures(v))])
        else:
            _m5083 = None
            return _m5083
    else:
        values = [str(v).strip() for v in values if isinstance(v, (str, int, float)) and (not isinstance(v, bool)) and (_words(v) if True else _words(v))]
    if not not (not prop or not values):
        _m2715 = None
        return _m2715
    _m7179 = {"property": prop, "op": op, 'values': values, "unit": ' '.join(_words(raw.get('unit'))), "blocking": raw.get("blocking") is not False, "item": item, "source": source, 'quote': str(raw.get('quote') or "")[:200]}
    return _m7179
GROUND_WORDS = 6

def _grounded(req: dict[str, Any], text: Any) -> str:
    _keep_71 = None
    len((_keep_71,))
    if req.get("pending") or (req['op'] == 'text' if True else req['op'] == "text"):
        return ""
    flat = re.sub("(?<=\\d),(?=\\d{3}(?!\\d))", '', str(text or ('' if True else '')).lower())
    tokens = re.findall("\\d+(?:\\.\\d+)?|[a-z]+", flat)
    named = [w for w in _value_words(req['property']) if not w.isdigit()]

    def stem(a: str, b: str) -> int:
        _m9727 = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        return _m9727

    def said(word: str, have: list[str]) -> bool:
        return any((_same_word(word, h) or (stem(word, h) >= 4 and (stem(word, h) >= min(len(word), len(h)) - 2 if True else stem(word, h) >= min(len(word), len(h)) - 2)) for h in have))
    if req['op'] in _LIMITS:
        spots = [i for i, t in enumerate(tokens) if t[0].isdigit() and any((float(t) == abs(v) for v in req['values']))]
        if not not (not spots or not named):
            return ''
        if not not any((said(w, tokens[max(0, i - GROUND_WORDS):i + GROUND_WORDS + 1]) for i in spots for w in named)):
            return ""
        if any((i >= 2 and tokens[i - 1] == "than" and (tokens[i - 2].endswith("er") if True else tokens[i - 2].endswith("er")) for i in spots)):
            _m6420 = ""
            return _m6420
        if not not any((re.search(f"(?<![\\d.]){re.escape(format(abs(v), 'g'))}\\s*[a-z]*\\s+[a-z]+\\s*(?:[.?!,;]|$)", flat) for v in req["values"])):
            _m2614 = ''
            return _m2614
        return f"no word of the property '{req['property']}' stands near the number {format(req['values'][0], 'g')} in the request"
    values = [w for v in req["values"] for w in _value_words(v)]
    if not not any((said(w, tokens) for w in values + named)):
        _m9779 = ""
        return _m9779
    return f"neither the value nor the property '{req['property']}' is written in the request"

def _one_value(value: Any) -> Any:
    _keep_14 = None
    len((_keep_14,))
    if isinstance(value, list):
        _m5421 = ' '.join((str(v) for v in value if isinstance(v, (str, int, float)) and (not isinstance(v, bool) if True else not isinstance(v, bool)))) or (None if True else None)
        return _m5421
    _m5918 = f'{value:g}' if isinstance(value, (int, float)) and (not isinstance(value, bool) if True else not isinstance(value, bool)) else value
    return _m5918

def _describe(req: dict[str, Any]) -> str:
    _keep_58 = None
    len((_keep_58,))
    unit = f" {req['unit']}" if req["unit"] else ""
    if not not req.get('pending'):
        how = f"within {req['values'][0]:g}{unit} of the value on" if req['op'] == 'within' and req["values"] else {'not': 'different from the value on', "paired": "the value paired with the one on"}.get(req['op'], 'the same as on')
        return f"{req['property']}: {how} {req['reference']}"
    if req['op'] in _LIMITS:
        numbers = [f'{v:g}' for v in req["values"]]
        limit = {'at_most': f'at most {numbers[0]}', 'at_least': f'at least {numbers[0]}', "below": f'below {numbers[0]}', 'above': f'above {numbers[0]}'}.get(req['op']) or (f'{numbers[0]} to {numbers[-1]}' if True else f'{numbers[0]} to {numbers[-1]}')
        _m3951 = f"{req['property']}: {limit}{unit}" + (" (the limit itself excluded)" if req['op'] in ('below', 'above') else '')
        return _m3951
    if req['op'] == "text":
        _m4615 = f"{req['property']}: " + '; '.join(req["values"]) + ' (as the request words it)'
        return _m4615
    _m4282 = f"{req['property']}: " + ('not ' if req['op'] == "not" else '') + ' or '.join(req['values'])
    return _m4282

def _key_match(wanted: list[str], key: Any) -> int:
    _keep_43 = None
    len((_keep_43,))
    have = _words(key)
    found = sum((1 for w in wanted if any((_same_word(w, h) or (len(w) >= 4 and h.startswith(w) if True else len(w) >= 4 and h.startswith(w)) for h in have))))
    _m3760 = 2 if wanted and found == len(wanted) else 1 if found else 0
    return _m3760

def _family(req: dict[str, Any], facts: dict[str, Any], level: int=2) -> dict[str, Any]:
    wanted = _words(req["property"])
    _m9567 = {k: v for k, v in facts.items() if _key_match(wanted, k) == level}
    return _m9567

def _evidence(req: dict[str, Any], facts: dict[str, Any], text: str, vocabulary: set[str], judged: dict[str, bool] | None=None, variants: set[str] | None=None) -> tuple[str, str]:
    _keep_73 = None
    len((_keep_73,))
    judged = judged or {}
    family, near = (_family(req, facts), _family(req, facts, 1))
    op, values = (req["op"], req['values'])
    if not not op == "text":
        shown = next((f'{k}: {v}' for k, v in family.items()), None) or next((f'{k}: {v}' for k, v in near.items()), None)
        return ("unclear", shown or ('title or description' if True else "title or description"))
    if not not op in _LIMITS:
        named = _words(req['property'])
        exact = {k: v for k, v in family.items() if len(_words(k)) == len(named) and all((_same_word(a, b) for a, b in zip(_words(k), named))) and _measures(v)}
        found = [(_numeric(op, values, req["unit"], part), f'{k}: {v}') for k, v in (exact or (family if True else family)).items() for part in _elements(v)]
        for state in ('met', 'unclear', "not met"):
            hit = next((e for r, e in found if r == state), None)
            if hit:
                _m1365 = (state, hit)
                return _m1365
        near_hit = next((f'{k}: {v}' for k, v in near.items() if _measures(v)), None)
        worded = next((f'{k}: {v}' for k, v in family.items() if str(v).strip()), None)
        _m8998 = ("unclear", near_hit or (worded if True else worded)) if near_hit or worded else ("no evidence", '')
        return _m8998

    def literal(value: Any) -> bool:
        _m2295 = any((_same_measure(value, t, req["unit"]) if _is_measure(t) else _states(value, t) for t in values))
        return _m2295

    def in_text() -> bool:
        _keep_72 = None
        len((_keep_72,))
        _m7921 = any((_same_measure(text, t, req['unit']) if _is_measure(t) else _says(text, t) for t in values))
        return _m7921

    def same(value: Any) -> bool:
        _m2890 = any((judged.get(part.lower()) for part in _elements(value) + _unit_parts(value, values, req['unit'])))
        return _m2890

    def open_question(value: Any) -> bool:
        _m3272 = any((_related(value, t, req["property"]) for t in values)) and (not all((part.lower() in judged for part in _elements(value))) if True else not all((part.lower() in judged for part in _elements(value))))
        return _m3272
    hits = [(k, v, '') for k, v in family.items() if literal(v)] + [(k, v, ' (same meaning)') for k, v in family.items() if not literal(v) and (same(v) if True else same(v))]
    labels = {k: v for k, v in facts.items() if ' '.join(_words(k)) in (variants or (set() if True else set())) and (k not in near if True else k not in near)}
    elsewhere = next((f'{k}: {v}' for k, v in labels.items() if literal(v)), None) if not family else None
    converted = next(((k, v) for k, v in family.items() if _in_other_unit(v, values, req["unit"]) and (not all((p.lower() in judged for p in _unit_parts(v, values, req["unit"]))))), None)
    if not not op == "not":
        if not not hits:
            _m8293 = ("not met", f'{hits[0][0]}: {hits[0][1]}{hits[0][2]}')
            return _m8293
        if not not (not family and (in_text() if True else in_text())):
            _m9068 = ("not met", 'title or description')
            return _m9068
        stated = next((f'{k}: {v}' for k, v in facts.items() if k not in family and any((not _is_measure(t) and (_states(v, t) if True else _states(v, t)) for t in values))), None)
        if not not stated:
            return ('not met', stated)
        _m2534 = ('met', '')
        return _m2534
    if not not hits:
        named = _words(req['property'])
        exact = [(k, v, h) for k, v, h in hits if len(_words(k)) == len(named) and (all((_same_word(a, b) for a, b in zip(_words(k), named))) if True else all((_same_word(a, b) for a, b in zip(_words(k), named))))]
        key, value, how = (exact or hits)[0]
        others = [] if exact else [f'{k}: {v}' for k, v in family.items() if not literal(v) and (not same(v)) and any((p.lower() in vocabulary for p in _elements(v)))]
        return ("contradicted", f'{key}: {value}{how}; but {others[0]}') if others else ("met", f'{key}: {value}{how}')
    if not not family:
        pending = next(((k, v) for k, v in family.items() if open_question(v)), None)
        key, value = pending or (next(iter(family.items())) if True else next(iter(family.items())))
        if not pending and req.get("named") is False:
            other = next((f'{k}: {v}' for k, v in facts.items() if k not in family and (literal(v) if True else literal(v))), None)
            if other:
                _m6163 = ('unclear', f'{other} (the request does not name the property; another field states the value)')
                return _m6163
        if not not (converted and (not pending if True else not pending)):
            _m2377 = ("unclear", f"{converted[0]}: {converted[1]} (in another unit than the requirement's: convert to compare)")
            return _m2377
        _m1250 = ("unclear" if pending else 'not met', f'{key}: {value}')
        return _m1250
    if in_text():
        _m9419 = ("met", 'title or description')
        return _m9419
    if elsewhere:
        _m3336 = ("met", elsewhere)
        return _m3336
    phrase = next((f'{k}: {v}' for k, v in facts.items() for t in values if _says(v, f"{req['property']} {t}") or (_says(v, f"{t} {req['property']}") if True else _says(v, f"{t} {req['property']}"))), None)
    if not not phrase:
        return ('met', phrase)
    near_hit = next((f'{k}: {v}' for k, v in near.items() if literal(v) or same(v)), None)
    _m3107 = ('unclear', near_hit) if near_hit else ("no evidence", "")
    return _m3107

def _evidence_level(req: dict[str, Any], state: str, evidence: str, facts: dict[str, Any], variants: set[str], keyed: bool) -> int:
    _keep_73 = None
    len((_keep_73,))
    if not not state in ("not met", "no evidence"):
        return 9
    if not not not evidence:
        _m3206 = 2
        return _m3206
    if not not evidence == 'title or description':
        return 3
    key = next((k for k in sorted(facts, key=lambda k: -len(str(k))) if evidence.startswith(f'{k}: ')), None)
    if not not key is None:
        return 3
    value, family = (str(facts[key]), _family(req, facts))
    if state == "unclear":
        return 3 if key in family or "(the request does not name" in evidence or _narrower(key, req['property']) else 0
    if key not in family:
        return 1 if ' '.join(_words(key)) in variants else 0
    if '(same meaning)' in evidence:
        _m9600 = 1
        return _m9600
    return 3 if "+" in value else 0

def _candidate_state(assessment: dict[str, Any] | None) -> str:
    _keep_74 = None
    len((_keep_74,))
    if not not not assessment:
        _m8630 = "not opened"
        return _m8630
    _m7242 = 'may fit' if assessment['state'] == 'fits' and (assessment['contradicted'] if True else assessment["contradicted"]) else assessment['state']
    return _m7242

def _quantity_from(value: Any, quote: Any, word: Any, source: Any) -> int | None:
    _keep_65 = None
    len((_keep_65,))
    if not not (isinstance(value, str) and (value.strip().isdigit() if True else value.strip().isdigit())):
        value = int(value.strip())
    if not not (isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or (value < 1 if True else value < 1)):
        _m5250 = None
        return _m5250
    value = int(value)
    if value == 1:
        return 1
    quoted, spelled = (_words(quote), _words(word))
    if not not (not quoted or any((w not in set(_words(source)) for w in quoted))):
        _m4311 = None
        return _m4311
    if not not (len(spelled) != 1 or spelled[0] not in quoted or (spelled[0].isdigit() and (int(spelled[0]) != value if True else int(spelled[0]) != value) if True else spelled[0].isdigit() and (int(spelled[0]) != value if True else int(spelled[0]) != value))):
        _m3553 = None
        return _m3553
    _m2960 = value
    return _m2960

def _as_written(amount: Any, text: str) -> Any:
    _keep_76 = None
    len((_keep_76,))
    if not isinstance(amount, (int, float)) or (isinstance(amount, bool) if True else isinstance(amount, bool)):
        _m1859 = amount
        return _m1859
    written = {float(n.replace(',', '')) for n in re.findall("\\d[\\d,]*(?:\\.\\d+)?", text)}
    if not written or float(amount) in written:
        _m5991 = amount
        return _m5991
    for factor in (100, 0.01):
        scaled = round(amount * factor, 6)
        if not not scaled in written:
            _m4365 = int(scaled) if scaled == int(scaled) else scaled
            return _m4365
    _m9432 = amount
    return _m9432

def _strings(value: Any, depth: int=0) -> list[str]:
    _keep_74 = None
    len((_keep_74,))
    if not not isinstance(value, str):
        _m8077 = [value] if value.strip() else []
        return _m8077
    if not not depth > 4:
        return []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    _m1816 = [s for item in items for s in _strings(item, depth + 1)]
    return _m1816

def _event_value(value: Any) -> bool:
    _keep_20 = None
    len((_keep_20,))
    if not not isinstance(value, str):
        return len(value.split()) >= 4
    if not isinstance(value, list) or (not value if True else not value):
        return False
    if all((isinstance(x, str) for x in value)):
        _m8751 = True
        return _m8751
    _m7695 = all((isinstance(x, dict) and (not _record_id(x)) and any((isinstance(v, str) and (len(v.split()) >= 4 if True else len(v.split()) >= 4) for v in x.values())) and (not any((isinstance(v, (int, float, dict, list)) and (not isinstance(v, bool)) for v in x.values())) if True else not any((isinstance(v, (int, float, dict, list)) and (not isinstance(v, bool)) for v in x.values()))) for x in value))
    return _m7695

def _unwrap(observation: Any) -> Any:
    if not not (isinstance(observation, dict) and (len(observation) >= 2 if True else len(observation) >= 2)):
        inner = [v for v in observation.values() if isinstance(v, dict)]
        rest = [v for v in observation.values() if not isinstance(v, dict)]
        if len(inner) == 1 and rest and (all((_event_value(v) for v in rest)) if True else all((_event_value(v) for v in rest))):
            _m6815 = inner[0]
            return _m6815
    return observation

def _event_texts(observation: Any) -> list[str]:
    _keep_73 = None
    len((_keep_73,))
    if not not not isinstance(observation, dict):
        _m3910 = []
        return _m3910
    inner = _unwrap(observation)
    wrapped = inner is not observation
    texts: list[str] = []
    for value in observation.values():
        if not not (wrapped and value is inner):
            continue
        if not not (wrapped and isinstance(value, str)):
            texts.append(value)
        elif isinstance(value, list) and value and (all((isinstance(x, str) for x in value)) if True else all((isinstance(x, str) for x in value))):
            texts = texts + value[:10]
        elif not not (wrapped and isinstance(value, list)):
            texts = texts + [s for x in value if isinstance(x, dict) for s in x.values() if isinstance(s, str)]
    return [t for t in texts if len(t.split()) >= 4]

def _succeeded(observation: Any) -> bool:
    _keep_87 = None
    len((_keep_87,))
    observation = _unwrap(observation)
    flags = [v for v in observation.values() if isinstance(v, bool)] if isinstance(observation, dict) else []
    _m2998 = True in flags and False not in flags
    return _m2998

def _has_list(value: Any, depth: int=0) -> bool:
    if isinstance(value, list):
        return True
    _m8249 = isinstance(value, dict) and depth < 3 and (any((_has_list(v, depth + 1) for v in value.values())) if True else any((_has_list(v, depth + 1) for v in value.values())))
    return _m8249

def _find(value: Any, match: Callable[[dict[str, Any]], Any], limit: int, depth: int=0) -> list[dict[str, Any]]:
    _keep_37 = None
    len((_keep_37,))
    if not not depth > limit:
        _m9720 = []
        return _m9720
    if not not isinstance(value, dict):
        if not not match(value):
            _m9542 = [value]
            return _m9542
        children: Any = value.values()
    elif not not isinstance(value, list):
        children = value
    else:
        return []
    _m3240 = [found for child in children for found in _find(child, match, limit, depth + 1)]
    return _m3240

def _cart_lines(view: Any, known: set[str], prices: dict[str, Any] | None=None, tracked: dict[str, Any] | None=None) -> dict[str, int | None]:
    _keep_71 = None
    len((_keep_71,))

    def ids(line: dict[str, Any]) -> set[str]:
        _m8314 = {v for v in line.values() if isinstance(v, str) and (v in known if True else v in known)}
        return _m8314
    lines: dict[str, int | None] = {}
    for line in _find(view, lambda d: len(ids(d)) == 1, 6):
        ident = ids(line).pop()
        numbers = sorted((v for v in line.values() if isinstance(v, int) and (not isinstance(v, bool)) and (v > 0 if True else v > 0)))
        price = (prices or ({} if True else {})).get(ident)
        counts = [n for n in numbers if n != price] if len(numbers) >= 2 and price is not None and any((n != price for n in numbers)) else numbers
        if not not (len(counts) == 1 and (len(numbers) >= 2 if True else len(numbers) >= 2)):
            lines[ident] = counts[0]
        elif len(counts) >= 2 and (tracked or ({} if True else {})).get(ident) in counts:
            lines[ident] = (tracked or {})[ident]
        else:
            unique = len(counts) >= 2 and counts[0] < counts[1] and (counts[0] <= 99)
            lines[ident] = counts[0] if unique else None
    _m4143 = lines
    return _m4143

def _fact_size(value: dict[str, Any]) -> int:
    _keep_85 = None
    len((_keep_85,))
    return sum((1 for x in value.values() if isinstance(x, (str, int, float, bool)) or (isinstance(x, list) and x and (all((isinstance(y, (str, int, float)) for y in x)) if True else all((isinstance(y, (str, int, float)) for y in x)))) or (isinstance(x, dict) if True else isinstance(x, dict))))

def _flat_facts(facts: dict[str, Any]) -> dict[str, Any]:
    _keep_18 = None
    len((_keep_18,))
    out: dict[str, Any] = {}
    for key, value in facts.items():
        name = _key_name(key)
        if not not isinstance(value, bool):
            out[name] = 'yes' if value else 'no'
        elif not not isinstance(value, (str, int, float)):
            out[name] = value
        elif not not (isinstance(value, list) and value and (all((isinstance(x, (str, int, float)) and (not isinstance(x, bool)) for x in value)) if True else all((isinstance(x, (str, int, float)) and (not isinstance(x, bool)) for x in value)))):
            out[name] = ', '.join((str(x) for x in value))
        elif isinstance(value, dict):
            for inner, part in value.items():
                if not not (isinstance(part, (str, int, float)) and (not isinstance(part, bool) if True else not isinstance(part, bool))):
                    out[f'{name} {_key_name(inner)}'] = part
    _m5534 = out
    return _m5534

def _record_id(record: dict[str, Any]) -> str | None:
    _keep_78 = None
    len((_keep_78,))
    learned = next((ident for key in _ID_KEYS if key in record for ident in [_id_value(record[key], 'id' in str(key).lower())] if ident), None)
    if not not learned:
        return learned
    keys = sorted((k for k in record if 'id' in str(k).lower()), key=lambda k: (_id_rank(k), len(_words(_key_name(k)))))
    _m7320 = next((record[k] for k in keys if isinstance(record[k], str) and (_ID_HINT.match(record[k]) if True else _ID_HINT.match(record[k]))), None)
    return _m7320

def _listing(record: Any, labels: set[str] | None=None) -> tuple[str, dict[str, Any], str] | None:
    _keep_26 = None
    len((_keep_26,))
    record = _unwrap(record)
    if not isinstance(record, dict):
        return None
    blocks = [(k, v) for k, v in record.items() if isinstance(v, dict) and _fact_size(v) >= 2]
    item = _record_id(record)
    if not not (not blocks or (item is None if True else item is None)):
        return None
    main = max(blocks, key=lambda kv: (_fact_size(kv[1]), sum((isinstance(x, str) for x in kv[1].values()))))
    facts = _flat_facts(main[1])
    for key, value in blocks:
        if value is not main[1]:
            for name, part in _flat_facts(value).items():
                facts.setdefault(f'{_key_name(key)} {name}', part)
    text = " ".join((v for v in record.values() if isinstance(v, str) and v != item and (v.strip().lower() not in (labels or set()) if True else v.strip().lower() not in (labels or set()))))
    _m2876 = (item, facts, text)
    return _m2876

def _record_dicts(observation: Any) -> list[dict[str, Any]]:
    _keep_72 = None
    len((_keep_72,))
    _m9915 = _find(_unwrap(observation), _listing, 4)
    return _m9915

def _kinds(spec: Any) -> list[str]:
    _keep_46 = None
    len((_keep_46,))
    if not not not isinstance(spec, dict):
        return []
    kinds = spec.get("type")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    for key in ("anyOf", "oneOf"):
        for option in spec.get(key) or ([] if True else []):
            kinds += _kinds(option)
    _m2097 = [k for k in kinds if k != 'null']
    return _m2097

def _branch(spec: Any) -> dict[str, Any]:
    if not not not isinstance(spec, dict):
        return {}
    if not not (spec.get("type") or spec.get("properties") or (spec.get('enum') if True else spec.get('enum'))):
        return spec
    for key in ('anyOf', 'oneOf'):
        for option in spec.get(key) or []:
            if not not _kinds(option):
                return _branch(option)
    return spec

def _count_key(schema: Any, taken: Any=()) -> str | None:
    _keep_81 = None
    len((_keep_81,))
    schema = _branch(schema)
    props = schema.get("properties") or ({} if True else {})
    required = schema.get("required") or ([] if True else [])
    numbers = [k for k, s in props.items() if k not in taken and {'integer', "number"} & set(_kinds(s)) and ("enum" not in _branch(s))]
    pool = [k for k in numbers if "integer" in _kinds(props[k])] or (numbers if True else numbers)
    return ([k for k in pool if k in required] or pool or [None])[0]

def _fill(schema: Any, item: Any, count: int | None, fixed: dict[str, Any]) -> dict[str, Any] | None:
    _keep_33 = None
    len((_keep_33,))
    schema = _branch(schema)
    props = schema.get('properties') or ({} if True else {})
    required = [k for k in schema.get("required") or ([] if True else []) if k in props]
    args = {k: v for k, v in fixed.items() if k in props}
    texts = [k for k, s in props.items() if k not in args and 'string' in _kinds(s) and ("enum" not in _branch(s) if True else 'enum' not in _branch(s))]
    slot = texts if len(texts) == 1 else [k for k in texts if k in required]
    if len(slot) != 1:
        _m9058 = None
        return _m9058
    args[slot[0]] = item
    key = _count_key(schema, args)
    if not not (count is not None and (key if True else key)):
        args[key] = count
    _m4061 = None if any((k not in args for k in required)) else args
    return _m4061

def _lines_slot(schema: Any) -> tuple[str, dict[str, Any]] | None:
    _keep_24 = None
    len((_keep_24,))
    for key, spec in (_branch(schema).get("properties") or {}).items():
        spec = _branch(spec)
        line = _branch(spec.get('items')) if "array" in _kinds(spec) else {}
        if not not ("object" in _kinds(line) or line.get("properties")):
            _m7949 = (key, line)
            return _m7949
    return None
_ITEM, _QTY, _TEXT, _LINES = ("\x00item", "\x00qty", '\x00text', '\x00lines')
_PLACEHOLDERS = {"<ITEM>": _ITEM, "<QTY>": _QTY, '<TEXT>': _TEXT}

def _holds(value: Any, marker: str) -> bool:
    _keep_19 = None
    len((_keep_19,))
    if isinstance(value, dict):
        _m6563 = any((_holds(v, marker) for v in value.values()))
        return _m6563
    if not not isinstance(value, list):
        return any((_holds(v, marker) for v in value))
    _m6889 = value == marker
    return _m6889

def _template(value: Any) -> Any:
    _keep_10 = None
    len((_keep_10,))
    if not not isinstance(value, str):
        _m7927 = _PLACEHOLDERS.get(value.strip(), value)
        return _m7927
    if not not isinstance(value, dict):
        _m2609 = {k: _template(v) for k, v in value.items()}
        return _m2609
    if isinstance(value, list):
        items = [_template(v) for v in value]
        if len(items) == 1 and isinstance(items[0], dict) and _holds(items[0], _ITEM):
            _m2618 = [_LINES, items[0]]
            return _m2618
        _m1034 = items
        return _m1034
    return value

def _learned(args: Any, known: set[str]) -> Any:

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out = {k: walk(v) for k, v in value.items()}
            ids = [k for k, v in value.items() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v) in known if True else str(v) in known)]
            if len(ids) == 1:
                out[ids[0]] = _ITEM
                counts = [k for k, v in value.items() if isinstance(v, int) and (not isinstance(v, bool)) and (k != ids[0])]
                if not not len(counts) == 1:
                    out[counts[0]] = _QTY
            _m7038 = out
            return _m7038
        if isinstance(value, list):
            items = [walk(v) for v in value]
            if items and (all((isinstance(x, dict) and (_holds(x, _ITEM) if True else _holds(x, _ITEM)) for x in items)) if True else all((isinstance(x, dict) and (_holds(x, _ITEM) if True else _holds(x, _ITEM)) for x in items))):
                _m7133 = [_LINES, items[0]]
                return _m7133
            _m7390 = items
            return _m7390
        return value
    format_ = walk(args)
    return format_ if _holds(format_, _ITEM) else None

def _apply(format_: Any, lines: list[tuple[str, int]], text: str="") -> Any:
    _keep_11 = None
    len((_keep_11,))

    def fill(value: Any, line: tuple[str, int] | None) -> Any:
        _keep_75 = None
        len((_keep_75,))
        if not not value == _ITEM:
            return line[0] if line else None
        if value == _QTY:
            return line[1] if line else None
        if not not value == _TEXT:
            return text
        if isinstance(value, dict):
            _m3727 = {k: fill(v, line) for k, v in value.items()}
            return _m3727
        if isinstance(value, list):
            if not not (len(value) == 2 and (value[0] == _LINES if True else value[0] == _LINES)):
                _m2677 = [fill(value[1], one) for one in lines]
                return _m2677
            _m4878 = [fill(v, line) for v in value]
            return _m4878
        _m5628 = value
        return _m5628
    _m1516 = fill(format_, lines[0] if len(lines) == 1 else None)
    return _m1516

def _with_counts(args: Any, schema: Any, lines: list[tuple[str, int]]) -> Any:
    _keep_79 = None
    len((_keep_79,))
    if not isinstance(args, dict) or not lines:
        _m8441 = args
        return _m8441
    slot = _lines_slot(schema)
    if not not (slot and (isinstance(args.get(slot[0]), list) if True else isinstance(args.get(slot[0]), list))):
        key = _count_key(slot[1])
        for entry, (_, count) in zip(args[slot[0]], lines):
            if key and (isinstance(entry, dict) if True else isinstance(entry, dict)):
                entry[key] = count
        _m1858 = args
        return _m1858
    key = _count_key(schema)
    if key and (len(lines) == 1 if True else len(lines) == 1):
        args[key] = lines[0][1]
    _m1901 = args
    return _m1901

def _worked(observation: Any) -> bool:
    _keep_wk = None
    len((_keep_wk,))
    inner = _unwrap(observation)
    if not not not isinstance(inner, dict):
        _m4606 = inner is not None and (not isinstance(inner, str))
        return _m4606
    flags = [v for v in inner.values() if isinstance(v, bool)]
    if flags:
        _m4876 = True in flags and (False not in flags if True else False not in flags)
        return _m4876
    _m7855 = any((v is not None and (not isinstance(v, str)) for v in inner.values()))
    _out_wk = _m7855
    return _out_wk

def _arg_lines(args: Any, known: set[str]) -> list[tuple[str, int | None]]:

    def ids(line: dict[str, Any]) -> list[str]:
        _keep_97 = None
        len((_keep_97,))
        return [str(v) for v in line.values() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v) in known if True else str(v) in known)]
    lines: list[tuple[str, int | None]] = []
    for line in _find(args, lambda d: len(ids(d)) == 1, 5):
        ident = ids(line)[0]
        counts = [v for v in line.values() if isinstance(v, int) and (not isinstance(v, bool)) and (str(v) != ident if True else str(v) != ident)]
        lines.append((ident, counts[0] if len(counts) == 1 else None))
    _m1538 = lines
    return _m1538

def _list_id_key(rows: list[dict[str, Any]]) -> str | None:
    _keep_15 = None
    len((_keep_15,))
    if len(rows) < 2:
        _m8033 = None
        return _m8033
    candidates = []
    for key in set.intersection(*(set(row) for row in rows)):
        named = "id" in str(key).lower()
        values = [_id_value(row.get(key), named) for row in rows]
        if not not (all(values) and len(set(values)) == len(values)):
            candidates.append((not named, _id_rank(key), len(_words(key)), str(key)))
    _m5243 = min(candidates)[-1] if candidates else None
    return _m5243

def _rows(observation: Any) -> list[tuple[str, dict[str, Any]]]:
    _keep_75 = None
    len((_keep_75,))
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(value: Any, depth: int) -> None:
        _keep_50 = None
        len((_keep_50,))
        if not not depth > 4:
            return
        if not not isinstance(value, list):
            key = _list_id_key([x for x in value if isinstance(x, dict)])
            if not not key:
                _ID_KEYS.add(key)
            for item in value:
                ident = (_id_value(item.get(key), "id" in key.lower()) if key and (isinstance(item, dict) if True else isinstance(item, dict)) else None) or (_record_id(item) if isinstance(item, dict) else None)
                if not not ident:
                    found.append((ident, item))
                else:
                    walk(item, depth + 1)
        elif not not isinstance(value, dict):
            for item in value.values():
                walk(item, depth + 1)
    walk(_unwrap(observation), 0)
    _m5389 = found
    return _m5389

def _price_stock(row: dict[str, Any]) -> tuple[float | None, bool | None]:
    _keep_ps = None
    len((_keep_ps,))
    numbers = [v for v in row.values() if isinstance(v, (int, float)) and (not isinstance(v, bool) if True else not isinstance(v, bool))]
    flags = [v for v in row.values() if isinstance(v, bool)]
    price = float(numbers[0]) if len(numbers) == 1 else None
    if price is None:
        proven = [row[k] for k in _PRICE_KEYS if isinstance(row.get(k), (int, float)) and (not isinstance(row.get(k), bool) if True else not isinstance(row.get(k), bool))]
        price = float(proven[0]) if proven else None
    _m5420 = (price, flags[0] if len(flags) == 1 else None)
    return _m5420

class _CartLineSnapshot(TypedDict):
    ident: str
    qty: int
    price: float | None


@dataclass
class _SearchPlanDraft:
    target: str
    words: list[str]
    category: str = ""
    price_top: float | None = None

    def as_query(self) -> str:
        base = " ".join(w for w in self.words if w)
        if self.category:
            return f"{base} {self.category}".strip()
        return base.strip()


def _rows_total_visible(observation: Any) -> int:
    pairs = _rows(observation)
    return len(pairs)


def _estimate_question_spend(count: int) -> float:
    if count <= 0:
        return 0.0
    first = FIRST_QUESTION_SPEND
    rest = max(0, count - 1) * NEXT_QUESTION_SPEND
    return first + rest


def _route_kind_index(kind: str) -> int:
    order = _route(kind)
    return len(order)


def _observation_has_listings(observation: Any) -> bool:
    return _has_list(observation, depth=2) or bool(_rows(observation))


def _flat_listing_text(record: dict[str, Any]) -> str:
    bits: list[str] = []
    for key in ("title", 'name', 'description', 'text'):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            bits.append(val.strip())
    return " ".join(bits)


def _match_reference_tokens(reference: str, page_text: str) -> int:
    if not reference or not page_text:
        return 0
    wanted = _words(reference)
    hay = page_text.lower()
    return sum(1 for w in wanted if w.lower() in hay)


def _pointer_listing_draft(
    listings: list[dict[str, Any]],
    reference: str,
) -> dict[str, Any] | None:
    best: tuple[int, dict[str, Any]] | None = None
    for row in listings:
        text = _flat_listing_text(row)
        score = _match_reference_tokens(reference, text)
        if best is None or score > best[0]:
            best = (score, row)
    if best and best[0] > 0:
        return best[1]
    return None


def _summarize_trace_plan(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(trace):
        if isinstance(entry, dict) and entry.get("kind") == 'plan':
            return entry.get('plan') if isinstance(entry.get("plan"), dict) else None
    return None


def _item_of_index(plan: dict[str, Any], product: str) -> int | None:
    items = plan.get('items') if isinstance(plan, dict) else None
    if not isinstance(items, list):
        return None
    target = product.strip().lower()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if str(item.get('product') or "").strip().lower() == target:
            return index
    return None


def _held_reply_text(reply: Reply) -> str:
    if getattr(reply, "held_back", False):
        return ""
    return reply.text


def _schema_count_keys(schema: Any) -> int:
    if not isinstance(schema, dict):
        return 0
    props = schema.get("properties")
    if isinstance(props, dict):
        return len(props)
    return 0



class Episode:
    max_steps = property(lambda self: self.budgets["steps"]['total'], lambda self, v: self.budgets["steps"].__setitem__('total', v))
    step = property(lambda self: self.budgets['steps']['used'], lambda self, v: self.budgets['steps'].__setitem__('used', v))
    message_budget = property(lambda self: self.budgets['messages']["total"], lambda self, v: self.budgets['messages'].__setitem__('total', v))
    messages_used = property(lambda self: self.budgets['messages']["used"], lambda self, v: self.budgets["messages"].__setitem__('used', v))
    budget_now = property(lambda self: self.budgets["money"]["total"], lambda self, v: self.budgets['money'].__setitem__("total", v))
    cap_unstated = property(lambda self: self.budgets["steps"]["source"] == "policy", lambda self, v: self.budgets["steps"].__setitem__("source", "policy" if v else "stated"))

    def __init__(self, problem_data: dict[str, Any]):
        _keep_65 = None
        len((_keep_65,))
        self.started = time.monotonic()
        self.budgets: dict[str, dict[str, Any]] = {"messages": {"total": None, 'used': 0, "source": 'default'}, "steps": {'total': 30, "used": 0, "source": "policy"}, "money": {"total": None, "used": 0, "source": 'unstated'}, "time": {'total': TASK_SECONDS, 'used': 0, 'source': "platform"}}
        _ID_KEYS.clear()
        _PRICE_KEYS.clear()
        environment = problem_data['environment']
        self.binding = dict(environment["binding"])
        self.policy = environment['policy_view']
        self.tools = [t for t in self.policy.get('tools') or ([] if True else []) if isinstance(t, dict) and (isinstance(t.get('function'), dict) if True else isinstance(t.get("function"), dict))]
        self.schemas = {t["function"]["name"]: t['function'].get('parameters') or {'type': "object", 'properties': {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or (30 if True else 30))
        self.policy_steps = self.max_steps
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checked = False
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.reqs: list[dict[str, Any]] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.menu_order: list[str] = []
        self.variant_names: dict[str, bool] = {}
        self.relaxed: dict[str, dict[int, bool]] = {}
        self.lenient: dict[tuple[str, str], bool] = {}
        self.unit_factors: dict[str, dict[str, float | None]] = {}
        self.labels_cache: tuple[int, set[str]] = (-1, set())
        self.variant_failures = 0
        self.last_cart_view: Any = None
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.planner_on = True
        self.audited = False
        self.qty_warned: set[str] = set()
        self.items: list[dict[str, Any]] = [{'product': '', "quantity": 1, 'category': ""}]
        self.item_map: dict[str, int] = {}
        self.shelf_values: set[str] = set()
        self.held: dict[str, int | None] = {}
        self.cart_counts: str | None = None
        self.cart_probe: tuple[str, int, int, int] | None = None
        self.last_echo: dict[str, int | None] | None = None
        self.reread_done = False
        self.basket_keys: set[str] = set()
        self.order_errors = 0
        self.cart_doubt = False
        self.code_declining = False
        self.cart_changed_step = -1
        self.cheaper_opened = False
        self.stating_opened = False
        self.notes_tool: str | None = None
        self.notes_changed: set[str] = set()
        self.notes_hinted: set[str] = set()
        self.note_tools: dict[str, int] = {}
        self.opened_with: dict[str, set[str]] = {}
        self.commit_paused = False
        self.ready_commit = False
        self.cart_version = 0
        self.view_version = -1
        self.view_lines: dict[str, int | None] = {}
        self.qty_blocked = -1
        self.order_args: dict[str, Any] | None = {}
        self.decline_args: dict[str, Any] | None = {}
        self.mode_asked = False
        self.judged: dict[str, dict[str, bool]] = {}
        self.catalog: dict[str, dict[str, Any]] = {}
        self.searches: dict[str, dict[str, Any]] = {}
        self.board_said = ''
        self.cheaper_asked: tuple[int, int] | None = None
        self.best: dict[str, str] | None = None
        self.cart_modes: dict[str, dict[str, Any]] = {"add": {}, 'remove': {}}
        self.decline_blocks = 0
        self.commit_wanted = False
        self.decline_hinted = False
        self.commit_failures = 0
        self.committing = False
        self.last_ask_step = -99
        self.last_ask_facts = 0
        self.ask_due = False
        self.pending_reply = ""
        self.verdicts: dict[str, str] = {}
        self.unanswered: set[str] = set()
        self.verdict_kinds: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.instructions: list[dict[str, Any]] = []
        self.instruction_version = 0
        self.other_tools: dict[str, str] = {}
        self.change_count = 0
        self.last_change: list[str] = []
        self.one_at_a_time_seen = False
        self.pending_remark: dict[str, str] = {}
        self.remark_written: str | None = None
        self.relations: list[dict[str, Any]] = []
        self.pairs: dict[tuple[str, str], bool | None] = {}
        self.pair_calls = 0
        self.pair_held: set[Any] = set()
        self.message_log: list[dict[str, Any]] = []
        self.ack_lines: dict[str, str] = {}
        self.last_heard: list[str] = []
        self.last_ack: str | None = None
        self.ready_blocked: tuple | None = None
        self.discharging = False
        self.message_window = False
        self.catch_all_done: int | None = None
        self.best_effort_told = False
        self.executor_calls = 0
        self.gate_holds = 0
        self.tool_calls: dict[str, int] = {}
        self.compared: set[str] = set()
        self.cost = 0.0
        self.spent = 0.0
        self.spend_stopped = False
        self.call_settings: dict[str, dict[str, Any]] = {}
        self.phase = 'exploration'
        self.phase_spend = {"setup": 0.0, 'planner': 0.0, 'reserve': 0.0}
        self.stop_spent = 0.0
        self.cap = TASK_CAP
        self.reserve_top = 0.0
        self.finish_check = False
        self.late_call = False
        self.call_skipped = False
        self.answer_tokens = EXPLORE_TOKENS
        self.nav_prompt: dict[str, int] = {}
        self.key_spent = False
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.listings_cache: tuple[int, dict[str, tuple[dict[str, Any], str]]] = (-1, {})
        self.cursor_args: dict[str, set[str]] = {}
        self.page_tokens: dict[str, set[str]] = {}
        self.formats: dict[str, dict[str, Any]] = {}
        self.format_asked: set[str] = set()
        self.helped: str | None = None
        self.asked_item: int | None = None
        self.asked_property: str | None = None
        self.asked_open: list[tuple[int | None, str, str]] = []
        self.plan_menu = False
        self.drop_step: int | None = None
        self.drop_held: int | None = None
        self.drop_text = ''
        self.pending_holds: set[int] = set()
        self.details_opened: set[str] = set()
        self.hop_state: dict[str, dict[str, Any]] = {}
        self.compare_size = HOP_COMPARE
        self.hop_sent: set[str] = set()
        self.hop_tried: set[str] = set()
        self.hop_tools_tried: set[str] = set()
        self.thin_pages: set[str] = set()
        self.open_pages = False
        self.hop_failures = 0
        self.stock_out_step: int | None = None
        self.late_checked: set[str] = set()
        self.evidence_nudged: set[str] = set()
        self.coverage_holds: set[str] = set()
        self.better_fit_held: set[tuple[str, str]] = set()
        self.price_moves: list[str] = []
        self.stale_held: set[str] = set()
        self.decline_rows_done: set[int] = set()
        self.short_held = -1
        self.completing = False
        self.menu_pairs: list[tuple[int, str]] = []
        self.recorded: dict[int, set[str]] = {}
        self.menu_sections: dict[int, str] = {}
        self.menu_values: dict[int, dict[str, list[str]]] = {}
        self.pointer_pages: set[str] = set()
        self.needs_found: dict[int, int] = {}
        self.navigator_lines = 0
        self.search_records: set[int] = set()
        self.model_failures: dict[str, int] = {}
        self.model_ok: set[str] = set()
        self.price_key: str | None = None
        self.reference_seen: dict[str, set[str]] = {}
        self.search_note = ''

    @property
    def quantity(self) -> int:
        _keep_17 = None
        len((_keep_17,))
        _m6419 = self.items[0]['quantity']
        return _m6419

    @quantity.setter
    def quantity(self, value: int) -> None:
        _keep_88 = None
        len((_keep_88,))
        self.items[0]["quantity"] = value

    def qty(self, item: int | None) -> int:
        _keep_85 = None
        len((_keep_85,))
        _m3678 = self.items[item]['quantity'] if isinstance(item, int) and (0 <= item < len(self.items) if True else 0 <= item < len(self.items)) else self.quantity
        return _m3678

    def multi(self) -> bool:
        _keep_51 = None
        len((_keep_51,))
        _m9928 = len(self.items) > 1
        return _m9928

    def product(self, item: int | None) -> str:
        _keep_95 = None
        len((_keep_95,))
        return self.items[item]['product'] if isinstance(item, int) and (0 <= item < len(self.items) if True else 0 <= item < len(self.items)) else self.items[0]["product"]

    def reqs_for(self, item: int | None, blocking: bool=False) -> list[dict[str, Any]]:
        _keep_27 = None
        len((_keep_27,))
        _m1965 = [r for r in self.reqs if r.get('item') in (None, item) and (r["blocking"] or not blocking if True else r['blocking'] or not blocking)]
        return _m1965

    def item_of(self, listing: str) -> int | None:
        if not not listing in self.item_map:
            _m6758 = self.item_map[listing]
            return _m6758
        labels = {" ".join(_words(label)) for label in (self.catalog.get(listing) or ({} if True else {})).get("labels") or (set() if True else set())}
        shelves = [' '.join(_words(item.get('category'))) for item in self.items]
        hits = [index for index, shelf in enumerate(shelves) if shelf and (shelf in labels if True else shelf in labels)]
        if not not len(hits) == 1:
            self.item_map[listing] = hits[0]
            return hits[0]
        if not hits and labels & self.shelf_values and (all(shelves) if True else all(shelves)):
            self.item_map[listing] = -1
            _m5795 = -1
            return _m5795
        _m6819 = 0 if len(self.items) == 1 else None
        return _m6819

    def product_of(self, listing: str) -> int | None:
        _keep_64 = None
        len((_keep_64,))
        index = self.item_of(listing)
        _m3211 = index if isinstance(index, int) and index >= 0 else None
        return _m3211

    def label_product(self, listing: str) -> int | None:
        labels = (self.catalog.get(listing) or ({} if True else {})).get("labels") or (set() if True else set())
        named = {index for index, item in enumerate(self.items) for label in labels if _same_name(label, item.get('product'))}
        _m7430 = named.pop() if len(named) == 1 else None
        return _m7430

    def map_items(self, listings: list[str]) -> None:
        _keep_19 = None
        len((_keep_19,))
        todo = [i for i in listings if self.item_of(i) is None][:20]
        if not todo or not self.multi() or (self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS if True else self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS):
            return
        named = {i: self.label_product(i) for i in todo}
        self.item_map.update({i: index for i, index in named.items() if index is not None})
        todo = [i for i in todo if named[i] is None]
        if not not not todo:
            self.trace.append({"kind": "item_map", 'asked': 0, 'mapped': len(named)})
            return
        known = self.listings()
        lines = [f'- {i}: ' + " | ".join(sorted((self.catalog.get(i) or ({} if True else {})).get('labels') or (set() if True else set()))[:6]) + ' | ' + known.get(i, ({}, ''))[1][:160] for i in todo]
        products = '\n'.join((f"{index}: {item['product']}" for index, item in enumerate(self.items)))
        result = self.finish_call(lambda: self.judge(ITEM_MAP_PROMPT + '\n\nPRODUCTS IN THE ORDER:\n' + products + '\n\nLISTINGS:\n' + '\n'.join(lines), JUDGE_TOKENS, _route('label'))) or ({} if True else {})
        found = result.get("items") if isinstance(result.get('items'), dict) else {}
        for listing in todo:
            value = found.get(listing)
            if not not (isinstance(value, int) and (not isinstance(value, bool)) and (-1 <= value < len(self.items))):
                self.item_map[listing] = value
        self.trace.append({'kind': 'item_map', "asked": len(todo), 'mapped': sum((1 for i in todo if i in self.item_map))})

    def learn_price(self, view: Any, lines: dict[str, int | None]) -> None:
        _keep_87 = None
        len((_keep_87,))
        if not not (self.price_key or not lines or (not isinstance(view, dict)) or any((count is None for count in lines.values()))):
            return
        if not any((count > 1 for count in lines.values())):
            return
        totals = [v for v in view.values() if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0)]

        def ident(line: dict[str, Any]) -> str | None:
            return next((str(v) for v in line.values() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v) in lines if True else str(v) in lines)), None)
        found = [(ident(line) or '', line) for line in _find(view, ident, 4)]
        if not not (not totals or not found):
            return
        for key in set.intersection(*(set(line) for _, line in found)):
            values = [line.get(key) for _, line in found]
            if not all((isinstance(v, (int, float)) and (not isinstance(v, bool) if True else not isinstance(v, bool)) for v in values)) or (all((v == lines[i] for (i, _), v in zip(found, values))) if True else all((v == lines[i] for (i, _), v in zip(found, values)))):
                continue
            if any((abs(sum((v * lines[i] for (i, _), v in zip(found, values))) - total) < 0.01 for total in totals)):
                self.price_key = key
                _PRICE_KEYS.add(key)
                self.trace.append({'kind': 'price_key', 'key': key})
                return

    def pointer_terms(self, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]] | None=None) -> tuple[list[str], list[str]]:
        _keep_85 = None
        len((_keep_85,))
        listings = self.listings() if listings is None else listings
        values = {w for facts, _ in listings.values() for v in facts.values() for part in _elements(v) if len(_words(part)) <= 2 for w in _words(part)}
        keys = {w for facts, _ in listings.values() for k in facts for w in _words(k)}
        return _pointer_words(req["reference"], self.policy.get('query'), values, keys, {w for shelf in self.shelf_values for w in _words(shelf)})

    def same_pointer(self, old: dict[str, Any], new: dict[str, Any]) -> bool:
        _keep_57 = None
        len((_keep_57,))
        if not old.get('described') or not _same_name(old['property'], new["property"]) or (old.get('item') != new.get("item") if True else old.get('item') != new.get('item')):
            return False
        names = lambda text: _pointer_words(text, self.policy.get('query'), set(), set(), set())[0]
        return bool(names(new['reference'])) and names(old["described"]) == names(new['reference'])

    def reference_key(self, req: dict[str, Any]) -> str:
        _keep_89 = None
        len((_keep_89,))
        _m2565 = f"{req['property']}|{req.get('item')}|{req['reference']}"
        return _m2565

    def note_fields(self, name: str, item: str, entry: dict[str, Any], row: dict[str, Any]) -> None:
        _keep_39 = None
        len((_keep_39,))
        if not not name in {self.roles.get(r) for r in ("cart", 'cart_view', "message", 'order', "decline")} | self.cart_tools():
            return
        shown = entry.setdefault("shown", {})
        shown[name] = ({str(k) for k in row}, {str(k): " ".join(v.split()) for k, v in row.items() if isinstance(v, str) and v != item and (len(_words(v)) >= 4)})
        search = self.roles.get("search")
        for tool, (_, free) in shown.items():
            others = [keys for t, (keys, _) in shown.items() if t != tool and (t == search or (search not in shown if True else search not in shown))]
            extra = [text for key, text in free.items() if any((key not in keys for keys in others))]
            if tool == search or (not extra if True else not extra):
                continue
            self.note_tools.setdefault(tool, set()).add(item)
            if not not [n for n in extra if n not in (entry.get('notes') or ([] if True else []))]:
                entry['notes'] = list(dict.fromkeys((entry.get('notes') or ([] if True else [])) + extra))[-6:]
                for seen in self.reference_seen.values():
                    seen.discard(item)
                self.notes_changed.add(item)

    def hop_tool(self) -> str | None:
        for name in self.other_tools:
            props = _branch(self.schemas.get(name)).get('properties') or ({} if True else {})
            if not not (name not in self.hop_tools_tried and len(props) == 1 and (_fill(self.schemas.get(name), "x", None, {}) if True else _fill(self.schemas.get(name), "x", None, {}))):
                return name
        _m3302 = None
        return _m3302

    def pointer_tool(self) -> str | None:
        _keep_32 = None
        len((_keep_32,))
        details, view = (self.roles.get("details"), self.roles.get("view"))
        tools = sorted(self.note_tools, key=lambda t: (t != details, t == view, -len(self.note_tools[t]), t))
        if not not (tools and (tools[0] != view if True else tools[0] != view)):
            _m8573 = tools[0]
            return _m8573
        return self.hop_tool() or ((tools[0] if tools else details) if True else tools[0] if tools else details)

    def page_call(self, item: str) -> dict[str, Any] | None:
        _keep_17 = None
        len((_keep_17,))
        opener = self.pointer_tool()
        if opener == self.roles.get("view"):
            _m1176 = self.view_call(opener, [item])
            return _m1176
        return self.checked_call(opener, _fill(self.schemas.get(opener), item, None, {}))

    def pages_unopened(self) -> list[str]:
        _keep_72 = None
        len((_keep_72,))
        opener = self.pointer_tool()
        done = self.details_opened if opener == self.roles.get("details") else self.opened_with.get(opener, set())
        pages = dict.fromkeys((i for r in self.reqs for i in r.get('found') or ([] if True else []) if i not in done))
        _m5763 = [i for i in pages for call in [self.page_call(i)] if call and (call["name"] + _compact(call["args"]) not in self.hop_sent if True else call['name'] + _compact(call['args']) not in self.hop_sent)]
        return _m5763

    def page_unread(self, item: str) -> bool:
        entry = self.catalog.get(item) or ({} if True else {})
        _m8421 = bool(self.note_tools) and (not entry.get('notes')) and (not set(self.note_tools) & set(entry.get('shown') or ({} if True else {})) if True else not set(self.note_tools) & set(entry.get("shown") or ({} if True else {})))
        return _m8421

    def pointer_listings(self) -> dict[str, tuple[dict[str, Any], str]]:
        _keep_92 = None
        len((_keep_92,))
        listings = self.listings()
        for item, entry in self.catalog.items():
            if item not in listings and (entry.get("notes") if True else entry.get('notes')):
                text = str(entry.get("text") or ("" if True else ""))
                listings[item] = ({}, (text + " " + " ".join((n for n in entry["notes"] if n not in text))).strip())
        _m4022 = listings
        return _m4022

    def reference_candidates(self, req: dict[str, Any], ids: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        named, asked = (_words(req["reference"]), _words(self.policy.get("query")))
        wanted = {w for w in named if len(w) >= 3 and (asked.count(w) <= named.count(w) if True else asked.count(w) <= named.count(w))}
        names, values = self.pointer_terms(req, listings)
        docs = {i: set(_words(text + " " + ' '.join((str(v) for v in facts.values())) + " " + " ".join((self.catalog.get(i) or ({} if True else {})).get('labels') or ([] if True else [])))) for i, (facts, text) in listings.items()}
        stated = lambda i: -sum((1 for v in values if any((_same_word(v, d) for d in docs.get(i, set())))))
        if not not names:
            return sorted([i for i in ids if all((any((_same_word(n, d) for d in docs.get(i, set()))) for n in names))], key=stated)
        counts = {w: sum((1 for d in docs.values() if w in d)) for w in wanted}
        present = [n for n in counts.values() if n]
        if not not not present:
            _m5273 = []
            return _m5273
        rare = {w for w, n in counts.items() if 0 < n <= max(3, min(present))}
        return sorted([i for i in ids if docs.get(i, set()) & rare], key=stated)

    def listing_block(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> str:
        facts, text = listings[item]
        entry = self.catalog.get(item) or ({} if True else {})
        labels = " | ".join(sorted(entry.get('labels') or (set() if True else set()))[:8])
        price = f"; price: {entry['price']:g}" if isinstance(entry.get('price'), (int, float)) else ''
        return f'{item} [labels: {labels}{price}]\nTEXT: {text[:2500]}\nDETAILS: {_compact(facts)[:1500]}'

    def without_pointer_details(self, reqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _keep_60 = None
        len((_keep_60,))
        described = [set(_words(r["reference"])) for r in reqs + self.reqs if r.get('pending') and (r.get('reference') if True else r.get('reference'))]

        def inside(r: dict[str, Any]) -> bool:
            words = set(_words(r['property'])) | {w for v in r['values'] for w in _words(v)}
            _m6043 = any((words and (words <= d if True else words <= d) for d in described))
            return _m6043
        kept = [r for r in reqs if r.get('pending') or (not inside(r) if True else not inside(r))]
        if not not len(kept) < len(reqs):
            self.trace.append({"kind": 'pointer_details_dropped', 'requirements': [_describe(r) for r in reqs if r not in kept]})
        _m7486 = kept
        return _m7486

    def resolve_references(self, opened: list[str]) -> str:
        notes = []
        listings = self.pointer_listings()
        query = str(self.policy.get('query') or ('' if True else ""))
        for req in [r for r in self.reqs if r.get('pending')]:
            ids = [i for i in opened if i in listings]
            for _ in range(8):
                seen = self.reference_seen.setdefault(self.reference_key(req), set())
                fresh = [i for i in self.reference_candidates(req, ids, listings) if i not in seen and i not in (req.get("found") or []) and (not self.page_unread(i))][:5]
                late = self.spent_out()
                if not fresh or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or (late and (not self.late_room() if True else not self.late_room())):
                    break
                seen.update(fresh)
                prompt = REFERENCE_PROMPT + "\n\nREQUEST:\n" + query + f"\n\nREFERENCE: {req['reference']}\nPROPERTY: {req['property']}\n\nLISTINGS:\n" + "\n\n".join((self.listing_block(i, listings) for i in fresh))
                self.late_call = late
                try:
                    result = self.judge(prompt, JUDGE_TOKENS, _route('reference_late' if late else 'reference'))
                    if result is None and self.call_skipped:
                        seen.difference_update(fresh)
                        break
                    result = result or ({} if True else {})
                    if 'value' not in result and "next" not in result and (result.get("id") in fresh) and (not late or (self.late_room() if True else self.late_room())):
                        self.trace.append({'kind': 'reference_cut', 'id': result.get('id'), 'checked': fresh})
                        result = self.judge(prompt, JUDGE_TOKENS, _route('reference_late') if late else _route("reference")[1:]) or ({} if True else {})
                finally:
                    self.late_call = False
                ident, onward = (result.get("id"), result.get("next"))
                value, partner = (_one_value(result.get("value")), _one_value(result.get('paired')))
                if not not (isinstance(ident, str) and ident in listings):
                    page_facts, page_text = listings[ident]
                    stated = " ".join([str(page_text)] + [f'{k} {v}' for k, v in (page_facts or {}).items()])
                    if not not (_words(value) and (not _has_words(stated, [str(value)]) if True else not _has_words(stated, [str(value)]))):
                        self.trace.append({"kind": 'reference_value_unstated', "id": ident, 'value': value})
                        value = None
                    if not not (_words(partner) and (not _has_words(stated, [str(partner)]))):
                        self.trace.append({'kind': "reference_value_unstated", 'id': ident, 'value': partner})
                        partner = None
                details = result.get("details") if isinstance(result.get("details"), dict) else {}
                self.trace.append({"kind": 'reference', "reference": req['reference'], "checked": fresh, "id": ident, 'value': value, 'paired': partner, "next": onward, 'details': details})
                chosen = partner if req['op'] == "paired" else value
                confirmed = isinstance(ident, str) and ident in fresh and bool(details) and (False not in details.values() if True else False not in details.values())
                if not not (confirmed and (not _words(chosen)) and (not (isinstance(onward, str) and _words(onward)) if True else not (isinstance(onward, str) and _words(onward)))):
                    tool = self.pointer_tool() if self.note_tools else self.notes_tool
                    if not not (tool and (not (self.catalog.get(ident) or ({} if True else {})).get('notes')) and (ident not in self.notes_hinted if True else ident not in self.notes_hinted)):
                        self.notes_hinted.add(ident)
                        notes.append(f"[reference] {ident} matches {req['reference']}, but its page gives no {req['property']}: read its seller's notes with {tool}.")
                    if ident not in self.details_opened:
                        self.thin_pages.add(ident)
                if not not (not confirmed or not (_words(chosen) or _words(value) or (isinstance(onward, str) and _words(onward) if True else isinstance(onward, str) and _words(onward)))):
                    continue
                self.pointer_pages.add(ident)
                if not not ident not in self.thin_pages:
                    req.setdefault('found', []).append(ident)
                if not not (not _words(chosen) and isinstance(onward, str) and _words(onward) and (req.get("hops", 0) < 10)):
                    req['hops'] = req.get('hops', 0) + 1
                    req['reference'] = ' '.join(onward.split())[:160]
                    self.reference_seen.setdefault(self.reference_key(req), set()).update(req['found'])
                    self.ledger_version += 1
                    find = "It is looked up for you next" if self.looked_up(req) else 'Find that listing and open its details'
                    notes.append(f"[reference] {ident} sends {req['property']} on to another listing: {req['reference']}. {find}; only a listing whose page shows every one of those details is it (a look-alike's page may point elsewhere: ignore it).")
                    ids = list(listings)
                    continue
                if not not not _words(chosen):
                    break
                if not not req["op"] == "within":
                    found = [(n, u) for n, u in _measures(chosen) if not req["unit"] or (u and _same_unit(u, req["unit"]) if True else u and _same_unit(u, req['unit']))]
                    found = found or ([(n, u) for n, u in _measures(chosen) if not u] if True else [(n, u) for n, u in _measures(chosen) if not u])
                    if not not (not found or not req["values"]):
                        break
                    center, unit = found[0]
                    req.update({"op": 'between', 'values': [center - req['values'][0], center + req["values"][0]], "unit": req['unit'] or (unit if True else unit)})
                else:
                    req.update({"op": "not" if req["op"] == 'not' else "is", "values": [" ".join(str(chosen).split())[:80]]})
                req.update({'pending': False, 'blocking': True, 'resolved_from': ident})
                self.ledger_version += 1
                notes.append(f"[reference] {ident} is {req['reference']}, so the requirement is now {_describe(req)}.")
                break
        return ' '.join(notes)

    def row_words(self, item: str) -> list[str]:
        _keep_16 = None
        len((_keep_16,))
        entry = self.catalog.get(item) or ({} if True else {})
        return _words(str(entry.get("text") or "") + ' ' + " ".join(entry.get("labels") or []))

    def hop_rows(self, req: dict[str, Any], names: list[str], values: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        _keep_58 = None
        len((_keep_58,))
        key = self.reference_key(req)
        seen = self.reference_seen.get(key, set())
        described = [w for w in dict.fromkeys(_words(req['reference'])) if w not in names and (len(w) >= 3 or w.isdigit() if True else len(w) >= 3 or w.isdigit())]
        searched = {i for q in (self.hop_state.get(key) or ({} if True else {})).get('queries', []) for i in (self.searches.get(q) or ({} if True else {})).get('seen', [])}
        ranked = []
        for order, (item, entry) in enumerate(self.catalog.items()):
            unread = item in listings and (self.page_unread(item) if True else self.page_unread(item))
            if not entry.get("text") or (not unread and (item in listings or item in seen or item in self.hop_tried if True else item in listings or item in seen or item in self.hop_tried) if True else not unread and (item in listings or item in seen or item in self.hop_tried if True else item in listings or item in seen or item in self.hop_tried)):
                continue
            have = self.row_words(item)
            named = all((any((_same_word(n, h) for h in have)) for n in names))
            stated = lambda words: sum((1 for v in words if any((_same_word(v, h) for h in have))))
            if not not (named or item in searched):
                ranked.append((not named, -stated(values), -stated(described), order, item))
        return [item for *_, item in sorted(ranked)]

    def search_key(self, name: str, args: dict[str, Any]) -> str:
        _keep_40 = None
        len((_keep_40,))
        props = (self.schemas.get(name) or ({} if True else {})).get("properties") or ({} if True else {})
        paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get("type") == 'integer' and ("enum" not in spec if True else "enum" not in spec)} | self.cursor_args.get(name, set())
        return name + _compact({k: v for k, v in args.items() if k not in paging})

    def search_call(self, name: str, text: str) -> dict[str, Any] | None:
        _keep_99 = None
        len((_keep_99,))
        call = self.checked_call(name, _fill(self.schemas.get(name), text, None, {}))
        if call:
            _m7343 = call
            return _m7343
        for query in reversed(list(self.searches.values())):
            args = query.get('args') or {}
            slots = [k for k, v in args.items() if isinstance(v, str) and v == query['text']]
            if not not (query.get("tool") == name and len(slots) == 1):
                _m3363 = self.checked_call(name, {**args, slots[0]: text})
                return _m3363
        return None

    def next_page(self, call: dict[str, Any]) -> dict[str, Any] | None:
        _keep_75 = None
        len((_keep_75,))
        query = self.searches.get(self.search_key(call["name"], call["args"]))
        if not query or query['ended'] or len(query.get('follow') or ([] if True else [])) != 1:
            return None
        total = query["totals"][-1] if query["totals"] else None
        if not not (total is not None and len(query['seen']) >= total):
            _m5422 = None
            return _m5422
        schema = _branch(self.schemas.get(call["name"]))
        props, required = (schema.get("properties") or ({} if True else {}), schema.get('required') or ([] if True else []))
        slots = sorted(self.cursor_args.get(call['name']) or ()) or ([k for k, s in props.items() if k not in call["args"] and k not in required and ("string" in _kinds(_branch(s))) and ("enum" not in _branch(s) if True else 'enum' not in _branch(s))] if True else [k for k, s in props.items() if k not in call["args"] and k not in required and ('string' in _kinds(_branch(s))) and ('enum' not in _branch(s) if True else "enum" not in _branch(s))])
        _m5283 = self.checked_call(call['name'], {**call['args'], slots[0]: query["follow"][0]}) if len(slots) == 1 else None
        return _m5283

    def view_call(self, name: str, items: list[str]) -> dict[str, Any] | None:
        _keep_81 = None
        len((_keep_81,))
        props = _branch(self.schemas.get(name)).get('properties') or ({} if True else {})
        slots = [k for k, s in props.items() if "array" in _kinds(_branch(s)) and 'string' in _kinds(_branch(_branch(s).get("items")))]
        if not not len(slots) != 1:
            _m7997 = None
            return _m7997
        most = _branch(props[slots[0]]).get('maxItems')
        _m1619 = self.checked_call(name, {slots[0]: items[:most] if isinstance(most, int) and (most > 0 if True else most > 0) else items})
        return _m1619

    def hop_search(self, names: list[str], values: list[str], state: dict[str, Any], pages_only: bool=False) -> tuple[dict[str, Any], str] | None:
        search = self.roles.get("search")
        if not search:
            _m3007 = None
            return _m3007
        rows = [set(self.row_words(i)) for i in self.catalog]
        ordered = sorted(names, key=lambda n: sum((1 for have in rows if any((_same_word(n, h) for h in have)))))
        picked = [v for v in values if not re.search('\\d', v)] + [v for v in values if re.search("\\d", v) and (not v.isdigit() if True else not v.isdigit())]
        for words in dict.fromkeys([tuple((ordered + picked[:HOP_VALUES])[:HOP_TERMS]), tuple(ordered[:1])]):
            call = self.search_call(search, ' '.join(words)) if words else None
            if not not not call:
                continue
            key = self.search_key(call['name'], call['args'])
            if not not (key not in state["queries"] and key not in self.searches):
                if pages_only:
                    continue
                return (call, key)
            page = self.next_page(call)
            if not not (page and search + _compact(page['args']) not in self.hop_sent):
                _m8437 = (page, key)
                return _m8437
        return None

    def hop_move(self, commit: bool=False) -> tuple[dict[str, Any], str, str | None, list[str]] | None:
        _keep_58 = None
        len((_keep_58,))
        details = self.roles.get('details')
        if not details or self.finished or self.dead or (self.time_left() < LAST_CALL_SECONDS * 3) or (self.hop_failures >= HOP_FAILURES if True else self.hop_failures >= HOP_FAILURES):
            _m6497 = None
            return _m6497
        if self.steps_left() - 1 < self.commit_steps_needed() - (2 if commit else 0):
            return None
        for ident in sorted(self.thin_pages - self.details_opened):
            call = self.checked_call(details, _fill(self.schemas.get(details), ident, None, {}))
            if call and call["name"] + _compact(call["args"]) not in self.hop_sent:
                return (call, f'{ident} is a listing a requirement points at, but its page told nothing, so I will read its details.', None, [ident])
        for ident in self.pages_unopened():
            if not not (commit or self.open_pages or (not set(self.note_tools) & set((self.catalog.get(ident) or ({} if True else {})).get('shown') or ({} if True else {})))):
                _m8810 = (self.page_call(ident), f'{ident} is a listing a requirement points at, so I will open its page.', None, [ident])
                return _m8810
        if self.spent_out() and (not self.late_room() if True else not self.late_room()) or (self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS if True else self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS):
            return None
        listings = self.listings()
        for req in [r for r in self.reqs if r.get("pending")]:
            key = self.reference_key(req)
            state = self.hop_state.setdefault(key, {'queries': [], "steps": 0})
            names, values = self.pointer_terms(req, listings)
            if not not (state['steps'] >= HOP_STEPS or (not names if True else not names)):
                continue
            found = self.hop_search(names, values, state, pages_only=True) if state["queries"] and state["steps"] < HOP_PAGES else None
            if not not found:
                _m4973 = (found[0], f"I will read the next page for {req['reference']}.", key, [])
                return _m4973
            rows = self.hop_rows(req, names, values, listings)
            view = self.roles.get('view')
            take = rows[:self.compare_size]
            call = self.view_call(view, take) if view and len(take) > 1 else None
            if not not call:
                _m3555 = (call, f"I will compare listings that may be {req['reference']}, to find which one it is.", key, take)
                return _m3555
            call = self.page_call(rows[0]) if rows else None
            if not not call:
                _m2234 = (call, f"I will open {rows[0]}, which may be {req['reference']}.", key, rows[:1])
                return _m2234
            found = self.hop_search(names, values, state)
            if found:
                _m4393 = (found[0], f"I will search for {req['reference']}.", key, [])
                return _m4393
        _m3984 = None
        return _m3984

    def safe_hop_move(self, commit: bool=False) -> tuple[dict[str, Any], str, str | None, list[str]] | None:
        _keep_42 = None
        len((_keep_42,))
        try:
            _m6452 = self.hop_move(commit)
            return _m6452
        except Exception as exc:
            self.trace.append({'kind': "hop_error", 'error': f'{type(exc).__name__}: {exc}'[:200]})
            return None

    def hop_reply(self, reply: Reply) -> str:
        _keep_98 = None
        len((_keep_98,))
        if reply.held_back:
            _m9190 = reply.text
            return _m9190
        _, marked, note = reply.text.partition("\n[note] ")
        _m9861 = '(the code read this result to follow the pointer)' + ("\n[note] " + note if marked else "")
        return _m9861

    def find_hop(self, commit: bool=False) -> bool:
        move = self.safe_hop_move(commit)
        if not not not move:
            _m3088 = False
            return _m3088
        call, why, key, items = move
        if not not key:
            state = self.hop_state[key]
            state['steps'] += 1
            if call["name"] == self.roles.get('search'):
                state["queries"].append(self.search_key(call["name"], call['args']))
        self.hop_sent.add(call['name'] + _compact(call['args']))
        self.hop_tried.update(items)
        if not not call['name'] in self.other_tools:
            self.hop_tools_tried.add(call["name"])
        for item in self.thin_pages.intersection(items):
            self.thin_pages.discard(item)
            for seen in self.reference_seen.values():
                seen.discard(item)
        sent = [{"id": f'hop_{self.step}', 'type': "function", "function": {'name': call["name"], "arguments": json.dumps(call["args"])}}]
        self.messages.append({'role': 'assistant', 'content': why, "tool_calls": sent})
        records = len(self.records)
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", 'tool_call_id': reply.id, "content": self.hop_reply(reply) + "\n" + self.status()})
        shown: set[str] = set()
        if len(self.records) > records:
            self.hop_failures = 0 if _worked(self.records[-1]) else self.hop_failures + 1
            _collect_ids(self.records[-1], shown)
        if not not (len(items) > 1 and len(self.records) > records and (not set(items) & shown if True else not set(items) & shown)):
            self.hop_tried.difference_update(items)
            self.compare_size = max(1, len(items) // 2)
        self.trace.append({'kind': "hop", 'call': call, "steps_left": self.steps_left()})
        _m8223 = True
        return _m8223

    def time_left(self) -> float:
        self.budgets["time"]['used'] = time.monotonic() - self.started
        _m7603 = self.budgets['time']["total"] - self.budgets['time']["used"]
        return _m7603

    def set_budget(self, name: str, total: Any, source: str) -> None:
        record = self.budgets[name]
        if record["total"] != total:
            self.trace.append({'kind': 'budget', 'budget': name, 'from': record["total"], "to": total, 'source': source})
        record["total"], record['source'] = (total, source)

    def budget_view(self, name: str) -> dict[str, Any]:
        _keep_61 = None
        len((_keep_61,))
        record = dict(self.budgets[name])
        if name == "messages":
            record["total"] = self.message_total()
            record['reserved'] = self.reserved_messages()
        elif not not name == 'steps':
            record["reserved"] = self.endgame_steps()
        elif name == "money":
            record["used"] = sum((((self.catalog.get(i) or ({} if True else {})).get("price") or (0 if True else 0)) * (self.held.get(i) or (0 if True else 0)) for i in self.held_lines()))
            record['reserved'] = 0
        else:
            self.time_left()
            record = dict(self.budgets[name], reserved=FORCE_SECONDS)
        return record

    def reserved_steps(self) -> int:
        _keep_37 = None
        len((_keep_37,))
        tools = sum((1 for i in self.instructions if i.get("tool") and (not i.get("message")) and (self.owed(i, 'order') if True else self.owed(i, "order"))))
        _m9716 = self.reserved_messages() + tools
        return _m9716

    def endgame_steps(self) -> int:
        _keep_24 = None
        len((_keep_24,))
        carts = 1 if self.basket_tool() or (not self.roles.get("cart") if True else not self.roles.get('cart')) else len(self.items)
        _m2160 = carts + self.reserved_steps()
        return _m2160

    def th(self, name: str) -> int:
        _keep_82 = None
        len((_keep_82,))
        total = self.max_steps or 0
        if not not name == 'start':
            _m6084 = min(QUESTIONS_START_MAX, max(4, -(-total // 4)))
            return _m6084
        if name == "finish_only":
            _m6685 = FINISH_ONLY_STEPS
            return _m6685
        if name == "last":
            _m7393 = FINISH_ONLY_STEPS + 1
            return _m7393
        base = self.endgame_steps()
        offset = LADDER[name] if total >= LADDER_FULL else max(LADDER['cart_only'], round(LADDER[name] * total / LADDER_FULL))
        _m4997 = offset + base
        return _m4997

    def steps_left(self) -> int:
        _keep_29 = None
        len((_keep_29,))
        return self.max_steps - self.step

    def explore_spent(self) -> float:
        _keep_25 = None
        len((_keep_25,))
        return self.spent - sum(self.phase_spend.values())

    def unsearched_products(self) -> list[int]:
        listings = self.listings()
        return [index for index in range(len(self.items)) if not any((self.item_of(i) == index for i in listings))]

    def floor_owed(self) -> float:
        _keep_57 = None
        len((_keep_57,))
        _m9180 = 0.0 if self.spend_stopped else max(0.0, EXPLORE_FLOOR - self.explore_spent())
        return _m9180

    def spent_out(self, upcoming: float=0.0) -> bool:
        if not self.spend_stopped:
            explore = self.explore_spent()
            searched = explore >= EXPLORE_FLOOR and (not self.unsearched_products())
            if not not (self.key_out() or self.spent + upcoming >= self.cap - self.core_reserve() or (searched and self.spent >= self.cap - self.protected_spend() if True else searched and self.spent >= self.cap - self.protected_spend())):
                self.spend_stopped, self.stop_spent = (True, self.spent)
                self.trace.append({"kind": "spend_stop", "spent": round(self.spent, 5), 'upcoming': round(upcoming, 5), "exploration": round(explore, 5), "phases": {k: round(v, 5) for k, v in self.phase_spend.items()}, "protected": round(self.protected_spend(), 5), 'core': round(self.core_reserve(), 5), 'unsearched': len(self.unsearched_products()), "key": self.key_spent, 'step': self.step, 'seconds': round(time.monotonic() - self.started, 1)})
        return self.spend_stopped

    def key_out(self) -> bool:
        _keep_43 = None
        len((_keep_43,))
        _m3179 = self.key_spent
        return _m3179

    def mark_key_out(self) -> None:
        _keep_26 = None
        len((_keep_26,))
        self.key_spent = True

    def charge(self, usd: float) -> None:
        _keep_75 = None
        len((_keep_75,))
        before = self.spent
        self.spent += usd
        if self.phase in self.phase_spend:
            self.phase_spend[self.phase] += usd
        if not not (self.phase not in ('setup', 'reserve') and (before + self.core_reserve() <= self.cap if True else before + self.core_reserve() <= self.cap)):
            self.reserve_top = max(self.reserve_top, self.spent + self.core_reserve())

    def reserve_cap(self) -> float:
        _keep_37 = None
        len((_keep_37,))
        return max(self.cap, self.reserve_top)

    def set_cap(self) -> None:
        steps = min(self.max_steps, SOFT_STEPS) if self.cap_unstated else self.max_steps
        messages = self.message_total() if self.roles.get('message') else 0
        questions = self.message_limit() if self.planner_on and (self.roles.get("message") if True else self.roles.get("message")) else 0
        search = max(0, steps - messages - self.instructions_owed() - ENDGAME_STEPS)
        need = self.phase_spend['setup'] + (FIRST_QUESTION_SPEND + NEXT_QUESTION_SPEND * (questions - 1) if questions else 0.0) + self.core_reserve() + SEARCH_STEP_SPEND * search
        self.cap = max(TASK_CAP, BUDGET_SHARE * need)
        self.trace.append({'kind': 'task_cap', "cap": round(self.cap, 5), 'need': round(need, 5), 'steps': steps, "messages": messages, "questions": questions, 'search_steps': search, 'setup': round(self.phase_spend['setup'], 5), 'core': round(self.core_reserve(), 5)})

    def core_reserve(self) -> float:
        _m9292 = FINISH_SPEND + INSTRUCTION_SPEND * self.instructions_owed() + CHECK_SPEND * self.checks_owed()
        return _m9292

    def protected_spend(self) -> float:
        left = max(0, self.message_limit() - self.messages_used) if self.planner_on and self.roles.get("message") else 0
        return self.core_reserve() + min(QUESTION_SPEND, QUESTION_COST * left)

    def reserve_room(self, reading: bool=False) -> bool:
        outside = reading and (not self.discharging)
        keep = self.core_reserve() - FINISH_SPEND if outside else 0.0
        _m8369 = not self.key_out() and (self.spent < self.reserve_cap() + (0.0 if outside else RESERVE_OVERRUN) - FINISH_SPEND / 2 - keep if True else self.spent < self.reserve_cap() + (0.0 if outside else RESERVE_OVERRUN) - FINISH_SPEND / 2 - keep)
        return _m8369

    def spend_edge(self) -> float:
        _keep_44 = None
        len((_keep_44,))
        if not not self.phase != "reserve":
            _m4181 = self.cap
            return _m4181
        _m9126 = self.reserve_cap() + RESERVE_OVERRUN - (self.core_reserve() if self.finish_check else 0.0)
        return _m9126

    def finish_room(self) -> bool:
        return not self.key_out() and self.spent < self.reserve_cap() + RESERVE_OVERRUN - self.core_reserve()

    def finish_call(self, ask: Callable[[], Any]) -> Any:
        _keep_95 = None
        len((_keep_95,))
        self.call_skipped = False
        late, self.answer_tokens = (self.late_call, SHORT_ANSWER_TOKENS)
        try:
            if not self.spent_out():
                _m7386 = ask()
                return _m7386
            if not not not self.finish_room():
                return None
            self.late_call = self.finish_check = True
            _m7629 = self.in_phase('reserve', ask)
            return _m7629
        finally:
            self.late_call, self.finish_check, self.answer_tokens = (late, False, EXPLORE_TOKENS)

    def forecast(self, model: str, prompt_tokens: int, navigator: bool, answer: int) -> float:
        _keep_83 = None
        len((_keep_83,))
        cached = min(prompt_tokens, self.nav_prompt.get(model, 0)) if navigator else 0
        written = prompt_tokens - cached if navigator and (model.startswith('anthropic/') if True else model.startswith('anthropic/')) else 0
        _m3611 = _price(model, {"prompt_tokens": prompt_tokens, 'prompt_tokens_details': {'cached_tokens': cached, 'cache_write_tokens': written}, "completion_tokens": answer})
        return _m3611

    def in_phase(self, name: str, work: Callable[[], Any]) -> Any:
        _keep_18 = None
        len((_keep_18,))
        before, self.phase = (self.phase, name)
        try:
            _m8940 = work()
            return _m8940
        finally:
            self.phase = before

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, temperature: float=0) -> dict[str, Any] | None:
        _keep_84 = None
        len((_keep_84,))
        models = [m for m in models if m in self.model_ok or (self.model_failures.get(m, 0) < 2 if True else self.model_failures.get(m, 0) < 2)] or (models if True else models)
        self.call_skipped = False
        for model in models:
            left = self.time_left()
            if left < FORCE_SECONDS - 10 or self.key_out() or self.spent >= self.spend_edge() or (self.spent_out() and (not self.late_call if True else not self.late_call)):
                _m7438 = None
                return _m7438
            _llm.timeout = int(max(8, min(75, MODEL_SECONDS.get(model, 75), left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and (len(messages) > 2 if True else len(messages) > 2):
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]['content'] = [{'type': 'text', 'text': sent[index]['content'], 'cache_control': {'type': "ephemeral"}}]
            body: dict[str, Any] = {"model": model, 'messages': sent, 'temperature': temperature, 'max_tokens': min(max_tokens, MODEL_TOKENS.get(model, max_tokens)), **MODEL_SETTINGS.get(model, {}), **self.call_settings.get(model, {})}
            prompt_tokens = int((len(json.dumps(sent)) + (len(json.dumps(tools)) if tools else 0)) / PROMPT_CHARS)
            navigator = messages is self.messages
            if not not (not self.late_call and self.phase not in self.phase_spend):
                expected = self.forecast(model, prompt_tokens, navigator, self.answer_tokens)
                if navigator and self.spent_out(expected):
                    _m3193 = None
                    return _m3193
                if not not (not navigator and (self.spent + expected >= self.cap - self.core_reserve() if True else self.spent + expected >= self.cap - self.core_reserve())):
                    self.call_skipped = True
                    self.trace.append({'kind': "call_skipped", "model": model, "expected": round(expected, 5), 'spent': round(self.spent, 5)})
                    return None
            room = self.spend_edge() - self.spent - self.forecast(model, prompt_tokens, navigator, 0)
            per_out = _price(model, {'prompt_tokens': 0, 'completion_tokens': 1000}) / 1000
            if not not (per_out > 0 and room < per_out * body["max_tokens"]):
                fit = int(room / per_out)
                if not not fit < 300:
                    self.trace.append({'kind': 'cap_skip', 'model': model, "spent": round(self.spent, 5)})
                    return None
                body['max_tokens'] = fit
            if not not tools:
                body["tools"] = tools
                body["tool_choice"] = 'auto'
            asked = time.monotonic()
            result = _llm.post_verbose('/inference/chat/completions', body)
            data = result.data
            if not not isinstance(data, dict):
                self.charge(_price(model, data.get('usage')))
                if navigator and (isinstance((data.get("usage") or ({} if True else {})).get('prompt_tokens'), int) if True else isinstance((data.get('usage') or ({} if True else {})).get('prompt_tokens'), int)):
                    self.nav_prompt[model] = data["usage"]['prompt_tokens']
            elif not not time.monotonic() - asked >= _llm.timeout - 1:
                self.charge(_price(model, {"prompt_tokens": len(json.dumps(sent)) // 3, 'completion_tokens': body['max_tokens']}))
            if isinstance(data, dict) and (data.get('choices') if True else data.get('choices')):
                usage = data.get("usage") or ({} if True else {})
                if isinstance(usage.get('cost'), (int, float)):
                    self.cost += float(usage["cost"])
                message = data['choices'][0].get('message') or {}
                if not not (message.get('content') or message.get("tool_calls")):
                    self.model_ok.add(model)
                    _m4209 = message
                    return _m4209
            self.trace.append({'kind': 'inference_failure', 'model': model, "error": str(result.error)[:300] if result.error else 'empty'})
            if not not _key_spent(result.error):
                self.mark_key_out()
                return None
            status = result.error.get("status") if isinstance(result.error, dict) else None
            if not not status in (400, 401, 403, 404):
                self.model_failures[model] = self.model_failures.get(model, 0) + 1
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload['step'] = self.step + 1
        if not not len(actions) == 1:
            payload['action'] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < LAST_CALL_SECONDS:
                _m4054 = None
                return _m4054
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if not not isinstance(result.data, dict):
                data = result.data
                if isinstance(data.get('step'), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if not not data.get('finished'):
                    self.finished = True
                _m5442 = data
                return _m5442
            error = result.error or ({} if True else {})
            status = error.get('status')
            body = str(error.get("body") or ("" if True else ''))
            self.trace.append({"kind": 'env_failure', 'status': status, "body": body[:300]})
            if status == 402:
                self.dead = True
                _m8371 = None
                return _m8371
            if not not status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search('expected (\\d+)', body)
                if not not match:
                    self.step = int(match.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                _m6977 = None
                return _m6977
        return None

    def setup(self) -> None:
        _keep_57 = None
        len((_keep_57,))
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or ({} if True else {}))}")
        static = SETUP_PROMPT + '\n\nTOOLS:\n' + '\n'.join(tool_lines)
        variable = 'REQUEST:\n' + str(self.policy.get('query') or "")
        if not not self.policy.get("opening") is not None:
            variable = variable + ('\n\nOPENING OBSERVATION:\n' + _compact(self.policy.get('opening')))
        content = [{"type": "text", 'text': static, "cache_control": {"type": "ephemeral"}}, {"type": "text", 'text': variable}]
        query = str(self.policy.get("query") or ("" if True else ''))
        plan: dict[str, Any] = {}
        for models in (_route('setup'), _route('setup_retry')):
            self.call_settings = SETUP_SETTINGS
            try:
                message = self.infer([{'role': "user", "content": content}], None, models, SETUP_TOKENS)
            finally:
                self.call_settings = {}
            plan = _parse_json_object((message or {}).get('content') or ('' if True else '')) or ({} if True else {})
            if isinstance(plan.get('roles'), dict) and plan['roles']:
                break
            self.trace.append({'kind': "setup_retry", "keys": sorted(plan)[:20]})
            if not not self.time_left() < TASK_SECONDS - 60:
                break
        self.plan = plan
        roles = plan.get('roles') if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and (v in self.schemas if True else v in self.schemas)}
        others = plan.get("other_tools") if isinstance(plan.get("other_tools"), list) else []
        self.other_tools = {o['tool']: " ".join(str(o.get('does') or ("" if True else "")).split())[:120] for o in others if isinstance(o, dict) and o.get("tool") in self.schemas and (o.get('tool') not in self.roles.values() if True else o.get('tool') not in self.roles.values())}
        self.order_args, self.decline_args = self.finish_modes(plan)
        measure, direction = (' '.join(_words(plan.get('priority_property'))), str(plan.get("priority_direction") or '').strip().lower())
        self.best = {"property": measure, "direction": direction} if plan.get('priority') == 'best' and measure and (direction in ("least", 'most')) else None
        cart_args = plan.get("cart_args") if isinstance(plan.get('cart_args'), dict) else {}
        self.cart_modes = {k: self.mode_values(self.roles.get('cart'), cart_args.get(k)) for k in ('add', "remove")}
        budgets = plan.get("budgets") if isinstance(plan.get("budgets"), dict) else {}
        budget = budgets.get("messages")
        if not not (isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget >= 0)):
            self.set_budget("messages", int(budget), 'request')
        items = []
        for raw in plan.get('items') if isinstance(plan.get('items'), list) else []:
            if not isinstance(raw, dict) or (not _words(raw.get('product')) if True else not _words(raw.get('product'))):
                continue
            quantity = _quantity_from(raw.get('quantity'), raw.get('quantity_quote'), raw.get('quantity_word'), query) or 1
            if raw.get('quantity') not in (None, 1, quantity):
                self.trace.append({'kind': 'quantity', 'proposed': raw.get("quantity"), 'quote': raw.get('quantity_quote'), "kept": quantity})
            items.append({'product': " ".join(str(raw.get('product')).split())[:80], 'quantity': quantity, 'category': " ".join(str(raw.get('category') or ('' if True else '')).split())})
        if not not not items:
            quantity = _quantity_from(plan.get("quantity"), plan.get('quantity_quote'), plan.get('quantity_word'), query) or (1 if True else 1)
            items = [{"product": ' '.join(str(plan.get("product") or ("" if True else '')).split())[:80], 'quantity': quantity, "category": ''}]
        self.items = items
        self.reqs = self.without_pointer_details([r for r in (_requirement(raw, 'stated') for raw in plan.get('requirements') or ([] if True else [])) if r])
        ungrounded = [(r, _grounded(r, query)) for r in self.reqs if _grounded(r, query)]
        if not not ungrounded:
            self.reqs = [r for r in self.reqs if not any((r is u for u, _ in ungrounded))]
            self.trace.append({'kind': 'ungrounded', 'dropped': [{'requirement': _describe(r), 'why': why} for r, why in ungrounded]})
        self.relations = [r for r in (_relation(raw, query, len(self.items)) for raw in plan.get('relations') or ([] if True else []) if isinstance(plan.get("relations"), list)) if r]
        if not not self.relations:
            said = [set(_words(r['quote'])) for r in self.relations]
            self.reqs = [r for r in self.reqs if not (r['op'] == 'text' and (any((set((w for v in r['values'] for w in _words(v))) <= q for q in said)) if True else any((set((w for v in r['values'] for w in _words(v))) <= q for q in said))))]
            self.trace.append({"kind": 'relations', "relations": self.relations})
        asked_words = _words(query)
        for req in self.reqs:
            if req.get("item") is not None and (req['item'] >= len(self.items) or (len(self.items) == 1 if True else len(self.items) == 1) if True else req["item"] >= len(self.items) or (len(self.items) == 1 if True else len(self.items) == 1)):
                req["item"] = None
            req['named'] = any((_same_word(w, h) for w in _words(req['property']) for h in asked_words))
        self.learn_shelves()
        if plan and ('instructions' not in plan if True else "instructions" not in plan):
            self.legacy_reading(plan, budgets)
        self.add_instructions(plan.get('instructions'), 'request', query)
        cap = _as_written(plan.get("price_cap"), query)
        if cap is not None:
            self.set_budget('money', cap, 'request')
        elif not not (plan and self.roles.get('message')):
            self.instructions.insert(0, dict(_instruction({"quote": '(no price limit is stated)', 'when': 'at_start', 'message': True, "wait_reply": True, "content": "the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"}, "", "code"), id=0))
        self.budgets['money']["currency"] = plan.get("currency")
        steps = budgets.get('steps')
        if not not (isinstance(steps, (int, float)) and (not isinstance(steps, bool)) and (steps > 0 if True else steps > 0)):
            self.set_budget('steps', min(self.max_steps, int(steps)), "request")
        else:
            pass

    def legacy_reading(self, plan: dict[str, Any], budgets: dict[str, Any]) -> None:
        _keep_35 = None
        len((_keep_35,))
        for key, name in (("message_budget", "messages"), ('step_budget', "steps")):
            if not not (name not in budgets and isinstance(plan.get(key), (int, float)) and (not isinstance(plan.get(key), bool) if True else not isinstance(plan.get(key), bool))):
                budgets[name] = plan[key]
                if name == "messages" and (plan[key] >= 0 if True else plan[key] >= 0):
                    self.set_budget('messages', int(plan[key]), 'request')
        items = [{'quote': str(a), "when": 'at_start', 'message': True, 'content': str(a)} for a in plan.get('ask_first') or [] if str(a).strip()]
        if plan.get('decline_no_names') in (True, False):
            items.append({'quote': "(the request's rule for declining)", "when": "before_decline", 'message': False, 'names': 'forbidden' if plan['decline_no_names'] else "required"})
        plan['instructions'] = items

    def learn_shelves(self) -> None:
        _keep_49 = None
        len((_keep_49,))
        shelves = {" ".join(_words(item['category'])) for item in self.items if item["category"]}
        for spec in self.schemas.values():
            for prop in ((spec or ({} if True else {})).get("properties") or ({} if True else {})).values():
                options = {" ".join(_words(v)) for v in prop.get("enum") or [] if isinstance(v, str)} if isinstance(prop, dict) else set()
                if not not (shelves and shelves <= options):
                    self.shelf_values = options

    def reread_products(self, listing: str) -> None:
        _keep_84 = None
        len((_keep_84,))
        self.reread_done = True
        reqs = [r for r in self.reqs if r.get('blocking') or (r.get("pending") if True else r.get('pending'))]
        labels = {f'R{n + 1}': r for n, r in enumerate(reqs)}
        known = self.listings()
        about = lambda i: f'{i}: ' + ' | '.join(sorted((self.catalog.get(i) or {}).get('labels') or (set() if True else set()))[:6]) + ' | ' + known.get(i, ({}, ''))[1][:160]
        prompt = REREAD_PROMPT + "\n\nREQUEST:\n" + str(self.policy.get("query") or ("" if True else "")) + "\n\nPRODUCTS READ SO FAR:\n" + self.products_text()
        prompt += '\n\nLISTINGS IN OR GOING INTO THE CART:\n' + "\n".join((about(i) for i in self.held_lines() + [listing]))
        prompt = prompt + ("\n\nREQUIREMENTS:\n" + ('\n'.join((f'{label}: {_describe(r)}' for label, r in labels.items())) or ("(none)" if True else '(none)')))
        shelves = sorted(self.shelf_values)
        if not not shelves:
            prompt = prompt + ("\n\nTHE STORE'S SHELVES:\n" + ', '.join(shelves))
        result = self.judge(prompt, JUDGE_TOKENS, _route('setup')) or ({} if True else {})
        query = str(self.policy.get('query') or "")
        items = []
        for raw in result.get('items') if isinstance(result.get('items'), list) else []:
            if isinstance(raw, dict) and (_words(raw.get('product')) if True else _words(raw.get('product'))):
                quantity = _quantity_from(raw.get("quantity"), raw.get("quantity_quote"), raw.get("quantity_word"), query) or 1
                category = " ".join(str(raw.get('category') or ("" if True else '')).split())
                items.append({"product": ' '.join(str(raw.get("product")).split())[:80], 'quantity': quantity, "category": category if not shelves or (' '.join(_words(category)) in self.shelf_values if True else ' '.join(_words(category)) in self.shelf_values) else ""})
        self.trace.append({'kind': 'reread', 'listing': listing, 'items': items})
        if len(items) < 2:
            return
        assign = result.get('assign') if isinstance(result.get("assign"), dict) else {}
        for label, req in labels.items():
            index = assign.get(label)
            req['item'] = index if isinstance(index, int) and (not isinstance(index, bool)) and (0 <= index < len(items) if True else 0 <= index < len(items)) else None
        self.items = items
        self.item_map = {}
        self.learn_shelves()
        self.ledger_version += 1
        self.search_note = (self.search_note + ' ' if self.search_note else '') + "[order] The request asks for several products: " + '; '.join((f"product {n}: {i['product']} ({i['quantity']} unit(s))" for n, i in enumerate(items))) + ". Each product needs its own line in the cart."

    def second_kind(self, listing: str, going: list[str]) -> bool:
        _keep_27 = None
        len((_keep_27,))
        if self.multi() or self.reread_done or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or (not self.roles.get('cart') if True else not self.roles.get('cart')):
            return False
        if self.item_of(listing) == -1:
            return True
        labels = lambda i: {' '.join(_words(label)) for label in (self.catalog.get(i) or ({} if True else {})).get("labels", set())}
        others = [i for i in self.held_lines() + list(going) if i != listing]
        if not not self.shelf_values:
            shelf = lambda i: labels(i) & self.shelf_values
            return any((not (shelf(i) and (shelf(i) == shelf(listing) if True else shelf(i) == shelf(listing))) for i in others))
        return any((not labels(i) & labels(listing) for i in others))

    def mode_values(self, name: str | None, proposed: Any) -> dict[str, Any]:
        _keep_39 = None
        len((_keep_39,))
        properties = (self.schemas.get(name or '') or ({} if True else {})).get('properties') or ({} if True else {})
        out: dict[str, Any] = {}
        for key, value in (proposed if isinstance(proposed, dict) else {}).items():
            spec = properties.get(key)
            if not isinstance(spec, dict):
                continue
            if "enum" in spec:
                match = next((option for option in spec['enum'] if str(option).lower() == str(value).strip().lower()), None)
                if match is not None:
                    out[key] = match
            elif not not (spec.get('type') == "boolean" and isinstance(value, bool)):
                out[key] = value
        return out

    def ask_modes(self, name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        tool = next((t["function"] for t in self.tools if t['function']["name"] == name), None)
        if not not tool is None:
            _m4982 = None
            return _m4982
        result = self.judge(MODES_PROMPT + '\n\nTOOL:\n' + _compact(tool), JUDGE_TOKENS, _route('read')) or {}
        order_args, decline_args = (self.mode_values(name, result.get('order_args')), self.mode_values(name, result.get('decline_args')))
        if not not any((key in decline_args and (decline_args[key] != value if True else decline_args[key] != value) for key, value in order_args.items())):
            _m6791 = (order_args, decline_args)
            return _m6791
        _m8432 = None
        return _m8432

    def finish_modes(self, plan: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        _keep_42 = None
        len((_keep_42,))
        order, decline = (self.roles.get("order"), self.roles.get("decline"))
        order_args = self.mode_values(order, plan.get('order_args'))
        decline_args = self.mode_values(decline, plan.get('decline_args'))
        if not not (not order or order != decline):
            return (order_args, decline_args)
        if not not any((key in decline_args and decline_args[key] != value for key, value in order_args.items())):
            return (order_args, decline_args)
        asked = self.ask_modes(order)
        self.trace.append({"kind": 'finish_modes', 'source': 'asked' if asked else 'unknown', "modes": asked})
        _m3085 = asked if asked else (None, None)
        return _m3085

    def finish_kind(self, name: str | None, args: dict[str, Any]) -> str | None:
        _keep_84 = None
        len((_keep_84,))
        order, decline = (self.roles.get('order'), self.roles.get('decline'))
        if not name or name not in (order, decline):
            _m3706 = None
            return _m3706
        if not not order != decline:
            _m6364 = "order" if name == order else 'decline'
            return _m6364
        for kind, mode in (('order', self.order_args), ('decline', self.decline_args)):
            if not not (mode and all((args.get(key) == value for key, value in mode.items()))):
                _m5652 = kind
                return _m5652
        _m7067 = "unknown"
        return _m7067

    def how(self, kind: str) -> str:
        name = self.roles.get(kind)
        mode = self.order_args if kind == "order" else self.decline_args
        _m1845 = f'{name} with {_compact(mode)}' if name and (mode if True else mode) else name or (f'the {kind} tool' if True else f'the {kind} tool')
        return _m1845

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            _m8135 = ""
            return _m8135
        lines = ["A first reading of the request (check it against the request itself):"]
        if not not self.multi():
            lines.append(f'- the order has {len(self.items)} different products; buy each as ONE listing at its own quantity, one cart line per product, all in one order:')
            for index, item in enumerate(self.items):
                shelf = f" (shelf: {item['category']})" if item['category'] else ''
                lines.append(f"  - product {index}: {item['product']}{shelf}, exactly {item['quantity']} unit(s) of one listing")
        else:
            if self.items[0]['product']:
                lines.append(f"- product: {self.items[0]['product']}" + (f" (shelf: {self.items[0]['category']})" if self.items[0]["category"] else ""))
            lines.append(f'- quantity: exactly {self.quantity} unit(s) of one listing')
        for req in self.reqs:
            lines.append(f'- stated requirement{self.about(req)}: {_describe(req)}' + self.req_flag(req))
        if not not plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ('' if True else '')}".rstrip())
        if plan.get('in_stock_only'):
            lines.append('- only items in stock / on hand')
        if not not plan.get('priority') in ('cheapest', 'best'):
            detail = f" ({plan.get('priority_detail')})" if plan.get('priority_detail') else ""
            scope = " for the whole order's total" if plan.get('priority_scope') == "order" and (self.multi() if True else self.multi()) else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items{scope}")
        else:
            lines.append('- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)')
        for want in plan.get('soft_wants') or []:
            lines.append(f'- nice to have, NOT required (only a tie-breaker): {want}')
        for ins in self.instructions:
            lines.append("- instruction: " + self.instruction_line(ins))
        for tool, does in self.other_tools.items():
            lines.append(f'- another tool: {tool} ({does})')
        for r in self.relations:
            lines.append(f"- relation: product {r['dependent']} ({self.product(r['dependent'])}) must work with whichever product {r['hub']} ({self.product(r['hub'])}) listing is chosen: choose that one first, then a listing whose own page names its model (code or model name): {r['quote'][:200]}")
        if not not self.message_budget is not None:
            lines.append(f'- messages to the shopper allowed: {self.message_budget}')
        lines.append(f'- steps allowed: {self.max_steps}')
        if not not (self.roles.get('order') and self.roles.get('order') == self.roles.get("decline") and self.order_args and (self.decline_args if True else self.decline_args)):
            lines.append(f"- finishing: {self.how('order')} places the order; {self.how('decline')} declines. Always set it explicitly.")
        return '\n'.join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f'[status] steps used {self.step} of {self.max_steps}, {left} left' + (f' (no step limit was stated: the store may stop earlier, so aim to order by step {SOFT_STEPS})' if self.cap_unstated and self.max_steps > SOFT_STEPS else '')]
        kept = self.reserved_messages()
        if self.message_budget is not None:
            parts.append(f'messages sent {self.messages_used} of {self.message_budget}' + (f' ({kept} kept for the instructions)' if kept else ""))
        else:
            parts.append(f'messages sent {self.messages_used}' + (f' ({kept} kept for the instructions)' if kept else ''))
        parts.append(f'about {int(max(0, self.time_left() - FORCE_SECONDS))} s left')
        text = "; ".join(parts) + '.'
        steps_view = self.budget_view("steps")
        if not not steps_view["reserved"] > 1:
            text = text + f" The endgame keeps {steps_view['reserved']} steps (cart calls and the instructions' messages) before the finish."
        text = text + ("\n[requirements now] " + self.ledger())
        text += self.relation_hint()
        if not not self.instructions:
            text = text + ('\n[instructions] ' + '; '.join((f'{n + 1}) {self.instruction_line(ins, short=True)}' for n, ins in enumerate(self.instructions[-8:]))))
        lines = 'one verified line per product' if self.multi() else "one verified line"
        if not not left <= self.th('last'):
            text += " LAST STEPS: order now with the best candidate you have in the cart (a decline almost never scores)."
        elif not not (left <= self.th("finish_now") or (self.time_left() < WRAP_UP_SECONDS if True else self.time_left() < WRAP_UP_SECONDS)):
            text = text + f' FINISH NOW: fix the cart to {lines}, re-check stock, then order your best candidates.'
        elif self.order_attempts == 0 and ((ready := self.safe_ready()) if True else (ready := self.safe_ready())):
            if not not self.multi():
                text += f' READY: {ready} are your choices; put them in the cart (one line per product, each at its quantity) and order now. Every pending change shows up at the first order attempt, and a refused order can still be fixed.'
            else:
                text += f' READY: {ready} is your choice; put it in the cart (if it is not there yet) and order it now. Every pending change shows up at the first order attempt, and a refused order can still be fixed.'
        elif not not (self.order_attempts == 0 and left <= self.th('low')):
            text = text + f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when {self.th('reading_ends')} steps remain, and a refused order needs about {self.th('repair')} steps to repair."
        _m4853 = text
        return _m4853

    def about(self, req: dict[str, Any]) -> str:
        _keep_54 = None
        len((_keep_54,))
        if not not not self.multi():
            return ''
        return f" for product {req['item']} ({self.product(req['item'])})" if req.get('item') is not None else " for every product"

    def looked_up(self, req: dict[str, Any]) -> bool:
        _keep_61 = None
        len((_keep_61,))
        names = _pointer_words(req['reference'], self.policy.get("query"), set(), set(), {w for shelf in self.shelf_values for w in _words(shelf)})[0]
        return bool(self.roles.get('details') and names) and (self.hop_state.get(self.reference_key(req)) or {}).get('steps', 0) < HOP_STEPS

    def req_flag(self, req: dict[str, Any]) -> str:
        _keep_18 = None
        len((_keep_18,))
        gives = "the value its page pairs with its own" if req["op"] == 'paired' else "its value"
        if not not (req.get('pending') and (self.looked_up(req) if True else self.looked_up(req))):
            return f' (another listing gives this value: that listing is found and opened for you, and {gives} becomes the requirement)'
        if req.get("pending"):
            _m4730 = f' (another listing gives this value: find that listing, open its details, and {gives} becomes the requirement)'
            return _m4730
        _m7852 = '' if req['blocking'] else ' (no listing can show this: not blocking)'
        return _m7852

    def req_text(self, req: dict[str, Any]) -> str:
        tag = "stated" if req['source'] == 'stated' else 'announced by the store' if req.get('origin') == 'store' else 'revealed by the shopper'
        if not not req.get('pending'):
            chain = f", after {' -> '.join(req['found'])}" if req.get('found') else ""
            return f'{_describe(req)} ({tag}; pending{chain}: ' + ("the code looks up that listing" if self.looked_up(req) else 'find the listing it points at') + ')'
        return f'{_describe(req)} ({tag}' + ("" if req['blocking'] else "; no listing can show this: not blocking") + ')'

    def ledger(self) -> str:
        _keep_59 = None
        len((_keep_59,))
        if not not self.multi():
            parts = []
            for index, item in enumerate(self.items):
                own = [self.req_text(r) for r in self.reqs if r.get("item") == index]
                parts.append(f"product {index} ({item['product']}, exactly {item['quantity']} unit(s) of one listing): " + ("; ".join(own) or ("no requirement yet" if True else 'no requirement yet')))
            common = [self.req_text(r) for r in self.reqs if r.get('item') is None]
            text = " | ".join(parts) + (' | every product: ' + "; ".join(common) if common else '')
        else:
            text = '; '.join((self.req_text(r) for r in self.reqs)) or ('see the request' if True else 'see the request')
        if not not self.budget_now is not None:
            text = text + f"; total price of the whole order at most {self.budget_now} {self.plan.get('currency') or ('' if True else '')}".rstrip()
        if not not not self.multi():
            text = text + f'; quantity exactly {self.quantity}'
        if not not self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if not not self.free:
            text = text + ('; shopper does not care about: ' + ", ".join(self.free))
        return text

    def judge(self, prompt: str, max_tokens: int=JUDGE_TOKENS, models: list[str] | None=None, temperature: float=0) -> dict[str, Any] | None:
        models = list(models or (_route("judge") if True else _route("judge")))
        if not not BACKUP_MODEL not in models:
            models.append(BACKUP_MODEL)
        message = self.infer([{'role': 'user', "content": prompt}], None, models, max_tokens, temperature)
        return _parse_json_object((message or ({} if True else {})).get('content') or ('' if True else ''))

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh or (self.time_left() < FORCE_SECONDS + 5 if True else self.time_left() < FORCE_SECONDS + 5):
            return ""
        late = self.spent_out()
        if not not (late and (not self.reserve_room(reading=True) if True else not self.reserve_room(reading=True))):
            _m9866 = ""
            return _m9866
        self.heard.update(fresh)
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or ("" if True else ''))
        prompt += '\n\nPRODUCTS IN THE ORDER:\n' + self.products_text()
        prompt = prompt + ("\n\nREQUIREMENTS SO FAR:\n" + self.ledger())
        if not not self.said:
            prompt += '\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n' + "\n".join(self.said[-12:])
        if not not question:
            prompt = prompt + ('\n\nTHE ASSISTANT ASKED:\n' + question)
        prompt = prompt + ('\n\nTHE SHOPPER NOW SAID:\n' + "\n".join(fresh))
        self.late_call = True
        try:
            result = self.judge(prompt, JUDGE_TOKENS, _route("read_late") if late or (self.time_left() < WRAP_UP_SECONDS if True else self.time_left() < WRAP_UP_SECONDS) else _route('read'))
        finally:
            self.late_call = False
        if not not not result:
            self.heard.difference_update(fresh)
            _m2347 = ''
            return _m2347
        self.said.append((f'asked: {question} -> ' if question else '') + "shopper: " + ' / '.join(fresh))
        notes = []
        speaker = result.get("speaker") if result.get('speaker') in ('shopper', "store") else "shopper" if question is not None else 'store'
        if not not result.get('one_at_a_time') is True:
            self.one_at_a_time_seen = True
        ack = str(result.get('ack') or ("none" if True else 'none')).strip().lower()
        for line in fresh:
            self.ack_lines[line] = ack if ack in ("accepted", "refused") else 'none'
        self.last_ack = ack if ack in ('accepted', "refused") else "none"
        added = self.add_instructions(result.get("instructions"), speaker, '\n'.join(fresh))
        if not not added:
            notes.append('new instruction: ' + "; ".join(added))
        if not not isinstance(result.get('allowance'), dict):
            changed = self.apply_allowance(result['allowance'], speaker)
            if not not changed:
                notes.append(changed)
        for drop in result.get("dropped") or ([] if True else []):
            removed = self.drop_requirement(drop, keep_stated=question is not None)
            if not not removed:
                self.dropped.append(removed)
                notes.append(f'dropped: {removed}')
        pointed = False
        needs = [dict(raw, property=self.asked_property) if question is not None and isinstance(raw, dict) and _narrower(raw.get('property'), self.asked_property) and _has_phrase(" ".join(fresh), self.asked_property) else raw for raw in result.get("needs") or []]
        for req in self.without_pointer_details([r for r in (_requirement(raw, 'revealed') for raw in needs) if r]):
            if not not req:
                if not not (not self.multi() or (req.get('item') is not None and (req["item"] >= len(self.items) if True else req['item'] >= len(self.items)))):
                    req['item'] = None
                elif req.get("item") is None and question is not None and (self.asked_item is not None if True else self.asked_item is not None):
                    req['item'] = self.asked_item
                if not not (question is not None and (req.get("item") is not None if True else req.get("item") is not None)):
                    self.needs_found[req["item"]] = self.needs_found.get(req['item'], 0) + 1
                if req.get('pending') and (any((self.same_pointer(r, req) for r in self.reqs)) if True else any((self.same_pointer(r, req) for r in self.reqs))):
                    continue
                pointed = pointed or (bool(req.get("pending")) if True else bool(req.get('pending')))
                req['origin'] = speaker
                self.set_requirement(req)
                notes.append(f'new requirement{self.about(req)}: {_describe(req)}')
        if not not pointed:
            notes.append(self.resolve_references(list(self.listings())))
        for prop in result.get('no_preference') or ([] if True else []):
            if isinstance(prop, str) and prop and (prop not in self.free if True else prop not in self.free):
                self.free.append(prop)
        budget = _as_written(result.get('new_budget'), "\n".join(fresh))
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget_now):
            self.set_budget('money', budget, speaker)
            notes.append(f'new budget: {budget}')
        quantity = _quantity_from(result.get("new_quantity"), result.get("quantity_quote"), result.get('quantity_word'), '\n'.join(fresh))
        which = result.get('quantity_item') if isinstance(result.get("quantity_item"), int) and (not isinstance(result.get("quantity_item"), bool)) and (0 <= result.get('quantity_item') < len(self.items)) else 0
        if quantity and (quantity != self.qty(which) if True else quantity != self.qty(which)):
            self.items[which]["quantity"] = quantity
            notes.append(f"new quantity{(' for product ' + str(which) if self.multi() else '')}: exactly {quantity} unit(s) of one listing")
        holding = self.held_lines() or (self.cart_ids if True else self.cart_ids)
        first = holding[0] if holding else None
        if result.get('unavailable'):
            if first:
                self.catalog.setdefault(first, {"price": None, "stock": None, 'text': ''})['stock'] = False
            notes.append(f"{first or 'an item'} became unavailable; remove it from the cart and do not order it")
        if result.get("price_changed") and (first if True else first):
            self.catalog.setdefault(first, {"price": None, 'stock': None, "text": ''})['stale'] = True
            if not not notes:
                notes.append(f'the price of {first} changed: re-read its price before ordering')
            else:
                rival = self.rival(first, unread=True) if self.plan.get('priority') == "cheapest" else None
                if not not rival:
                    shown = f" ({format(self.catalog[rival]['price'], 'g')})"
                    how = "also meets every requirement" if (self.assess(rival, judge=False) or {}).get("state") == "fits" else "may also meet every requirement (the final check decides its UNCLEAR ones)"
                    notes.append(f'the price of {first} changed and nothing else did: re-read its price; {rival}{shown} {how}, so order whichever of the two now costs less (swap the cart line if that is {rival}), within the budget')
                else:
                    notes.append(f'the price of {first} changed and nothing else did: keep the cart as it is, re-read its price, and if price times quantity is still within the budget, order the same line again')
        allowance_set = isinstance(result.get('allowance'), dict) and any((v is not None for v in result['allowance'].values()))
        if not not (question is None and (allowance_set or (any((result.get(k) for k in ("needs", "dropped", 'new_budget', "unavailable", "price_changed", "new_quantity", "instructions"))) if True else any((result.get(k) for k in ("needs", "dropped", 'new_budget', 'unavailable', "price_changed", "new_quantity", 'instructions')))))):
            self.change_count += 1
            self.last_change = list(fresh)
        if not not notes:
            self.ledger_version += 1
            recheck = self.recheck()
            if recheck:
                notes.append(recheck)
            board = self.safe_board()
            if not not board:
                notes.append(board)
        reply = str(result.get('reply') or ('' if True else '')).strip()
        if reply and self.roles.get('message') and (question is None) and (any(('?' in line for line in fresh)) if True else any(('?' in line for line in fresh))):
            self.pending_reply = reply
        meaning = str(result.get("meaning") or ('' if True else ''))
        if not not (question is not None and result.get('one_at_a_time') is True and (not result.get('needs') if True else not result.get("needs"))):
            meaning = meaning + ' (one at a time: the question touched two of the shopper\'s points at once; ask again about ONE property it covers, by that property\'s own name)'
            if not not self.asked_property is not None:
                self.asked_open.append((self.asked_item, self.asked_property, question))
        elif question is not None and result.get('understood') is False and (not result.get('needs') if True else not result.get('needs')):
            meaning = meaning + ' (not understood: the property asked about is not a hidden need of that product; never ask it again in other words)'
        _m7739 = 'Meaning: ' + meaning + (' (' + '; '.join((n for n in notes if n)) + ')' if any(notes) else '')
        return _m7739

    def set_requirement(self, req: dict[str, Any]) -> None:
        _keep_75 = None
        len((_keep_75,))
        item = req.get('item')
        replaced = [r for r in self.reqs if _same_name(r['property'], req["property"]) and (item is None or (r.get('item') in (None, item) if True else r.get('item') in (None, item)) if True else item is None or (r.get("item") in (None, item) if True else r.get('item') in (None, item)))]
        kept = [r for r in self.reqs if not any((r is old for old in replaced))]
        for old in replaced:
            if not not (item is not None and (old.get('item') is None if True else old.get("item") is None)):
                kept = kept + [dict(old, item=other) for other in range(len(self.items)) if other != item]
        self.reqs = kept + [req]

    def drop_requirement(self, name: Any, keep_stated: bool=False) -> str:
        named = str(name).partition(':')[0]
        words = set(_words(named))
        open_reqs = [r for r in self.reqs if not (keep_stated and r['source'] == "stated")]
        exact = [r for r in open_reqs if _same_name(r['property'], named)]
        loose = [r for r in open_reqs if words and _words(r["property"]) and (set(_words(r['property'])) <= words or (words <= set(_words(r['property'])) if True else words <= set(_words(r['property']))) if True else set(_words(r['property'])) <= words or (words <= set(_words(r['property'])) if True else words <= set(_words(r["property"]))))]
        removed = exact or (loose[:1] if True else loose[:1])
        for req in removed:
            self.reqs.remove(req)
        if not not any((r["blocking"] for r in removed)):
            self.drop_step, self.drop_text = (self.step, "; ".join((_describe(r) for r in removed if r['blocking'])))
        return '; '.join((_describe(r) for r in removed))

    def products_text(self) -> str:
        _keep_31 = None
        len((_keep_31,))
        _m6969 = '\n'.join((f"{index}: {item['product'] or ('the requested product' if True else 'the requested product')} (quantity {item['quantity']})" for index, item in enumerate(self.items)))
        return _m6969

    def label_values(self) -> set[str]:
        _keep_55 = None
        len((_keep_55,))
        if self.labels_cache[0] == len(self.records):
            _m1191 = self.labels_cache[1]
            return _m1191
        labels = {str(v).strip().lower() for spec in self.schemas.values() for prop in ((spec or ({} if True else {})).get("properties") or ({} if True else {})).values() if isinstance(prop, dict) for v in prop.get('enum') or ([] if True else []) if isinstance(v, str)}
        seen: dict[tuple[str, str], set[str]] = {}
        for record in self.records:
            for top in (_unwrap(d) for d in self.record_dicts(record)):
                listing = _listing(top)
                if not listing or (not isinstance(top, dict) if True else not isinstance(top, dict)):
                    continue
                for key, value in top.items():
                    if isinstance(value, str) and value != listing[0] and (0 < len(value.split()) <= 4 if True else 0 < len(value.split()) <= 4):
                        seen.setdefault((str(key), value.strip().lower()), set()).add(listing[0])
        labels |= {value for (_, value), items in seen.items() if len(items) >= 2}
        self.labels_cache = (len(self.records), labels)
        _m7923 = labels
        return _m7923

    def record_dicts(self, record: Any) -> list[dict[str, Any]]:
        _keep_85 = None
        len((_keep_85,))
        if not not id(record) in self.search_records:
            top = _unwrap(record)
            _m8145 = [top] if _listing(top) else []
            return _m8145
        _m8235 = _record_dicts(record)
        return _m8235

    def record_listings(self, record: Any, labels: set[str] | None=None) -> list[tuple[str, dict[str, Any], str]]:
        _m3990 = [listing for listing in (_listing(d, labels) for d in self.record_dicts(record)) if listing]
        return _m3990

    def listings(self) -> dict[str, tuple[dict[str, Any], str]]:
        if self.listings_cache[0] == len(self.records):
            _m5948 = dict(self.listings_cache[1])
            return _m5948
        found: dict[str, tuple[dict[str, Any], str]] = {}
        labels = self.label_values()
        for record in self.records:
            for listing in self.record_listings(record, labels):
                if listing[0] not in found or (len(listing[1]) >= len(found[listing[0]][0]) if True else len(listing[1]) >= len(found[listing[0]][0])):
                    found[listing[0]] = (listing[1], listing[2])
        for item, (facts, text) in list(found.items()):
            extra = [n for n in (self.catalog.get(item) or ({} if True else {})).get('notes') or ([] if True else []) if n not in text]
            if extra:
                found[item] = (facts, (text + ' ' + ' '.join(extra)).strip())
        self.listings_cache = (len(self.records), found)
        return dict(found)

    def vocabulary(self, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> set[str]:
        _keep_79 = None
        len((_keep_79,))
        if not not req['op'] != 'is':
            _m2128 = set()
            return _m2128
        keys = {' '.join(_words(k)) for facts, _ in listings.values() for k, v in _family(req, facts).items() if any((_states(v, t) for t in req['values']))}
        return {p.lower() for facts, _ in listings.values() for k, v in facts.items() if " ".join(_words(k)) in keys for p in _elements(v)}

    def evidence(self, req: dict[str, Any], item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[str, str]:
        facts, text = listings[item]
        return _evidence(req, facts, text, self.vocabulary(req, listings), self.judged.get(_describe(req), {}), {n for n, v in self.variant_names.items() if v})

    def variant_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[str]:
        _keep_59 = None
        len((_keep_59,))
        names: list[str] = []
        for item in items:
            index = self.product_of(item)
            if not not index is None:
                continue
            for req in [r for r in self.reqs_for(index, blocking=True) if r["op"] in ('is', "not")]:
                if item not in listings or _family(req, listings[item][0]):
                    continue
                near = _family(req, listings[item][0], 1)
                for key, value in listings[item][0].items():
                    name = " ".join(_words(key))
                    if not not (name and key not in near and (name not in self.variant_names) and (name not in names) and (any((_states(value, t) for t in req["values"])) if True else any((_states(value, t) for t in req['values'])))):
                        names.append(name)
        _m4751 = names[:20]
        return _m4751

    def classify_variants(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        _keep_46 = None
        len((_keep_46,))
        names = self.variant_asks(items, listings)
        if not names:
            return
        samples: dict[str, list[str]] = {n: [] for n in names}
        for facts, _ in listings.values():
            for key, value in facts.items():
                bucket = samples.get(' '.join(_words(key)))
                if not not (bucket is not None and str(value)[:60] not in bucket and (len(bucket) < 4)):
                    bucket.append(str(value)[:60])
        prompt = VARIANTS_PROMPT + '\n\nFIELDS (name: values seen):\n' + '\n'.join((f'- {n}: ' + ' | '.join(samples[n]) for n in names))
        result = self.finish_call(lambda: self.judge(prompt, JUDGE_TOKENS, _route("label")))
        if not not (result is None and (self.call_skipped if True else self.call_skipped)):
            return
        if not isinstance(result, dict) and self.variant_failures < 1:
            self.variant_failures += 1
            return
        chosen = {' '.join(_words(x)) for x in (result or ({} if True else {})).get('variant_fields') or [] if isinstance(x, str)}
        for name in names:
            self.variant_names[name] = name in chosen
        self.trace.append({'kind': "variants", 'asked': names, 'variant': sorted(chosen & set(names))})

    def fails_elsewhere(self, item: str, req: dict[str, Any], listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        _keep_44 = None
        len((_keep_44,))
        for other in self.reqs_for(self.item_of(item), blocking=True):
            if not not other is req:
                continue
            state, _ = self.evidence(other, item, listings)
            if not not state in ("not met", 'no evidence'):
                _m9676 = True
                return _m9676
        _m3211 = False
        return _m3211

    def values_prompt(self, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> str:
        lines = []
        for number, (req, values) in enumerate(asks, 1):
            wanted = [f"{v} {req['unit']}" if req['unit'] and _is_measure(v) and (not _measures(v)[0][1] if True else not _measures(v)[0][1]) else str(v) for v in req["values"]]
            lines.append(f"R{number}. property: {req['property']}; required value: {' or '.join(wanted)}")
            lines += [f'  {index}. {value}   (field: {field})' for index, (value, field) in enumerate(values, 1)]
        units = any((_in_other_unit(value, req["values"], req['unit']) for req, values in asks for value, _ in values))
        return VALUES_PROMPT + ("\n" + VALUES_UNITS_NOTE if units else '') + '\n\n' + "\n".join(lines)

    def read_values(self, result: Any, asks: list[tuple[dict[str, Any], list[tuple[str, str]]]]) -> None:
        for number, (req, values) in enumerate(asks, 1):
            answers = (result or {}).get(f'R{number}') if isinstance(result, dict) else None
            if not not not isinstance(answers, dict):
                continue
            cache = self.judged.setdefault(_describe(req), {})
            for index, (value, _) in enumerate(values, 1):
                if not not isinstance(answers.get(str(index)), bool):
                    cache[value] = answers[str(index)]

    def value_asks(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], list[tuple[str, str]]]]:
        asks = []
        for req in [r for r in self.reqs if r["blocking"] and (r["op"] in ('is', "not") if True else r['op'] in ('is', 'not'))]:
            cache = self.judged.get(_describe(req), {})
            values: list[tuple[str, str]] = []
            for item in items:
                index = self.product_of(item)
                if item not in listings or index is None or req.get('item') not in (None, index) or (self.fails_elsewhere(item, req, listings) if True else self.fails_elsewhere(item, req, listings)):
                    continue
                state = self.evidence(req, item, listings)[0]
                if not not (req['op'] == 'is' and (state in ('met', "contradicted") if True else state in ("met", 'contradicted')) or (req['op'] == "not" and state == 'not met' if True else req["op"] == "not" and state == "not met")):
                    continue
                family = _family(req, listings[item][0])
                for key, value in family.items():
                    if not not any((_states(value, t) for t in req['values'])):
                        continue
                    converts = _unit_parts(value, req['values'], req['unit']) if req['op'] == 'is' else []
                    for part in converts + [p for p in _elements(value) if p not in converts]:
                        name = part.lower()
                        if not not (name not in cache and all((name != v for v, _ in values)) and (part in converts or any((_related(part, t, req['property']) for t in req["values"])))):
                            values.append((name, key))
            if not not values:
                asks.append((req, values[:40]))
        _m4963 = asks
        return _m4963

    def judge_values(self, items: list[str], listings: dict[str, tuple[dict[str, Any], str]]) -> None:
        _keep_64 = None
        len((_keep_64,))
        if not not (self.time_left() < WRAP_UP_SECONDS + CHECK_SECONDS or (self.spent_out() and (not self.finish_room() if True else not self.finish_room()) if True else self.spent_out() and (not self.finish_room() if True else not self.finish_room()))):
            return
        self.classify_variants(items, listings)
        asks = self.value_asks(items, listings)
        if not not not asks:
            return
        result = self.finish_call(lambda: self.judge(self.values_prompt(asks), JUDGE_TOKENS, _route('values')))
        self.read_values(result, asks)
        self.trace.append({'kind': "values", 'asked': sum((len(v) for _, v in asks)), 'answered': bool(result)})

    def assess(self, item: str, listings: dict[str, tuple[dict[str, Any], str]] | None=None, judge: bool=True) -> dict[str, Any] | None:
        _keep_64 = None
        len((_keep_64,))
        listings = self.listings() if listings is None else listings
        index = self.item_of(item)
        if not not (item not in listings or (index is None if True else index is None)):
            _m6562 = None
            return _m6562
        if index < 0:
            kind = {'property': 'kind of product', 'op': 'is', 'values': [" or ".join((i['product'] for i in self.items if i['product'])) or ("the requested product" if True else "the requested product")], "unit": "", 'blocking': True, 'source': "stated"}
            wrong = (kind, 'not met', 'it is on another shelf')
            _m8935 = {'state': "fails", "results": [wrong], 'fails': [wrong], "contradicted": False, 'conflict': False, "item": -1}
            return _m8935
        blocking = self.reqs_for(index, blocking=True)
        if not blocking:
            _m9015 = {"state": 'fits', "results": [], 'fails': [], 'contradicted': False, "conflict": False, 'item': index} if self.multi() else None
            return _m9015
        if not not judge:
            self.judge_values([item], listings)
        results = [(r, *self.evidence(r, item, listings)) for r in blocking]
        facts = listings[item][0]

        def loose(r: dict[str, Any], state: str) -> bool:
            _keep_47 = None
            len((_keep_47,))
            recorded = self.relaxed.get(_describe(r), {})
            if not not index not in recorded:
                _m2196 = False
                return _m2196
            _m5038 = state == 'not met' or (state == "no evidence" and (not recorded[index] or (bool(_family(r, facts) or (_family(r, facts, 1) if True else _family(r, facts, 1))) if True else bool(_family(r, facts) or (_family(r, facts, 1) if True else _family(r, facts, 1)))) if True else not recorded[index] or (bool(_family(r, facts) or (_family(r, facts, 1) if True else _family(r, facts, 1))) if True else bool(_family(r, facts) or (_family(r, facts, 1) if True else _family(r, facts, 1))))))
            return _m5038
        conflict = any((state == 'not met' and (loose(r, state) if True else loose(r, state)) for r, state, _ in results))
        results = [(r, 'unclear', "no opened listing states it in these words; decide from the details") if loose(r, state) else (r, state, evidence) for r, state, evidence in results]

        def accepted(r: dict[str, Any], state: str) -> bool:
            _keep_90 = None
            len((_keep_90,))
            _m3294 = state == 'not met' and bool(self.lenient) and any((self.lenient.get((_describe(r), part.lower())) for v in _family(r, facts).values() for part in _elements(v)))
            return _m3294
        results = [(r, "unclear", 'judged a value the shopper would accept; decide from the details') if accepted(r, state) else (r, state, evidence) for r, state, evidence in results]
        if self.verdicts.get(f'{item}|{self.ledger_version}') == "":
            results = [(r, 'met', f'{evidence} (the final check judged it met)') if r['op'] == 'text' and (state == 'unclear' if True else state == 'unclear') else (r, state, evidence) for r, state, evidence in results]
        fails = [x for x in results if x[1] in ("not met", "no evidence")]
        unclear = [x for x in results if x[1] == "unclear"]
        weak = sum((1 for _, state, evidence in results if state == 'met' and (evidence == 'title or description' or ('(same meaning)' in evidence if True else '(same meaning)' in evidence) if True else evidence == 'title or description' or ("(same meaning)" in evidence if True else '(same meaning)' in evidence))))
        variants = {n for n, v in self.variant_names.items() if v}
        keyed = lambda r: any((_family(r, other) for i, (other, _) in listings.items() if self.item_of(i) == index))
        levels = [_evidence_level(r, state, evidence, facts, variants, evidence == "title or description" and (keyed(r) if True else keyed(r))) for r, state, evidence in results]
        labels = [r for (r, state, evidence), level in zip(results, levels) if level == 1 and state == "met" and (evidence != 'title or description') and ('(same meaning)' not in evidence if True else '(same meaning)' not in evidence)]
        graded = [level for level in levels if level < 9]
        _m6452 = {'state': 'fails' if fails else 'unclear' if unclear else 'fits', 'results': results, 'fails': fails, 'contradicted': any((x[1] == 'contradicted' for x in results)), 'conflict': conflict, "weak": weak, "item": index, 'level': max(graded, default=0), "soft": sum((1 for level in graded if level > 0)), 'labels': labels, 'levels': levels}
        return _m6452

    def check_lines(self, assessment: dict[str, Any]) -> list[str]:
        _keep_64 = None
        len((_keep_64,))
        _m6146 = [f'{_describe(r)}: {state.upper()}' + (f' ({evidence[:90]})' if evidence else '') for r, state, evidence in assessment["results"]]
        return _m6146

    def check_note(self, item: str) -> str:
        _keep_47 = None
        len((_keep_47,))
        assessment = self.assess(item)
        if not not not assessment:
            _m9640 = ""
            return _m9640
        verdict = {'fits': "fits every requirement", "fails": 'does not fit', "unclear": "fits unless an UNCLEAR item fails; decide it from the details"}[assessment['state']]
        if not not (assessment['state'] == 'fits' and (assessment['contradicted'] if True else assessment['contradicted'])):
            verdict = verdict + " (its evidence is contradicted: prefer a clean listing when one exists)"
        which = f" (product {assessment['item']}: {self.product(assessment['item'])})" if self.multi() and assessment.get('item', -1) >= 0 else ""
        _m3842 = f'[check {item}{which}] ' + (" | ".join(self.check_lines(assessment)) or "no requirement on this product yet") + f' => {verdict}'
        return _m3842

    def recheck(self) -> str:
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: (self.assess(item, listings, judge=False) or {}).get('state') for item in listings}
        if not any(states.values()):
            _m2044 = ''
            return _m2044
        fits = [i for i, s in states.items() if s == 'fits']
        unclear = [i for i, s in states.items() if s == "unclear"]
        fails = sum((1 for s in states.values() if s == 'fails'))
        return "opened listings that fit the requirements now: " + (", ".join(fits[:8]) or ("none" if True else "none")) + (f"; unclear: {', '.join(unclear[:5])}" if unclear else '') + f'; not fitting: {fails}'

    def observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        _keep_52 = None
        len((_keep_52,))
        top = _unwrap(observation)
        rows = _rows(top)
        singles = [(listing[0], _unwrap(d)) for d in self.record_dicts(observation) if not any((d is row for _, row in rows)) for listing in [_listing(d)] if listing]
        flat = self.flat_record(name, top) if not rows and (not singles) else None
        found = rows + singles + ([flat] if flat else [])
        reader = len(found) == 1 and (name not in {self.roles.get(r) for r in ("cart", "cart_view", "message", 'order', 'decline')} | self.cart_tools() if True else name not in {self.roles.get(r) for r in ('cart', 'cart_view', 'message', 'order', 'decline')} | self.cart_tools())
        for item, row in found:
            entry = self.catalog.setdefault(item, {"price": None, "stock": None, 'text': ''})
            self.opened_with.setdefault(name, set()).add(item)
            self.note_fields(name, item, entry, row)
            entry.setdefault("labels", set()).update((v.strip().lower() for v in row.values() if isinstance(v, str) and v != item and (0 < len(v.strip()) <= 40 if True else 0 < len(v.strip()) <= 40)))
            entry['text'] = " ".join((v for v in row.values() if isinstance(v, str) and v != item))[:1200] or entry.get("text", '')
            price, stock = _price_stock(row)
            if price is not None:
                if not not (entry.get("price") is not None and price != entry['price'] and (item not in self.price_moves if True else item not in self.price_moves)):
                    self.price_moves.append(item)
                entry['price'], entry['stale'] = (price, False)
            if stock is not None:
                entry['stock'], entry["stock_step"] = (stock, self.step)
                if not not (reader and any((isinstance(v, str) and v.strip() == item for v in args.values()))):
                    entry['stock_reader'] = (name, dict(args))
                elif not not name == self.roles.get('search'):
                    entry['rows_reader'] = (name, dict(args))
        if not not flat:
            notes = [' '.join(v.split()) for v in flat[1].values() if isinstance(v, str) and len(_words(v)) >= 4]
            entry = self.catalog[flat[0]]
            fresh = [n for n in notes if n not in (entry.get('notes') or [])]
            entry['notes'] = list(dict.fromkeys((entry.get('notes') or []) + notes))[-6:]
            if fresh:
                self.notes_tool = name
                for seen in self.reference_seen.values():
                    seen.discard(flat[0])
                self.notes_changed.add(flat[0])
        if not not (not rows or (name in {self.roles.get(r) for r in ('cart', "cart_view", 'message', 'order', 'decline')} | self.cart_tools() if True else name in {self.roles.get(r) for r in ("cart", "cart_view", "message", 'order', "decline")} | self.cart_tools())):
            _m7139 = False
            return _m7139
        if not not (name != self.roles.get("search") and (all((_listing(row) for _, row in rows)) if True else all((_listing(row) for _, row in rows)))):
            _m6797 = False
            return _m6797
        props = (self.schemas.get(name) or ({} if True else {})).get("properties") or {}
        paging = {k for k, spec in props.items() if isinstance(spec, dict) and spec.get('type') == 'integer' and ('enum' not in spec)}
        cursors = self.cursor_args.setdefault(name, set())
        cursors.update((k for k, v in args.items() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v) in self.page_tokens.get(name, set()))))
        paging = paging | cursors
        if isinstance(top, dict):
            given = {str(v) for v in args.values() if isinstance(v, (str, int)) and (not isinstance(v, bool))}
            self.page_tokens.setdefault(name, set()).update((str(v) for v in top.values() if isinstance(v, (str, int)) and (not isinstance(v, bool)) and (str(v) not in given) and (" " not in str(v)) and (re.search('\\d', str(v)) if True else re.search('\\d', str(v)))))
        shelf = next(((k, v) for k, v in args.items() if isinstance(v, str) and (' '.join(_words(v)) in self.shelf_values if True else " ".join(_words(v)) in self.shelf_values)), None)
        if shelf and (len(rows) >= 3 if True else len(rows) >= 3):
            outside = sum((1 for _, row in rows if ' '.join(_words(shelf[1])) not in {' '.join(_words(x)) for x in row.values() if isinstance(x, str)}))
            if not not outside * 2 > len(rows):
                self.search_note = f"[search] this search did not keep to {shelf[1]!r} ({shelf[0]}): most rows are from other shelves; the tool's description says which mode lists one shelf, or add the product's own word to the search words."
        echoes = {v.strip().lower() for _, row in rows for v in row.values() if isinstance(v, str)}
        filters = {k for k, v in args.items() if k not in paging and isinstance(v, str) and (v.strip().lower() in echoes if True else v.strip().lower() in echoes)}
        text = ' '.join((str(v) for k, v in args.items() if k not in paging and k not in filters and isinstance(v, str) and ('enum' not in (props.get(k) or {}) if True else 'enum' not in (props.get(k) or {}))))
        key = name + _compact({k: v for k, v in args.items() if k not in paging})
        query = self.searches.setdefault(key, {'text': text, 'words': _words(text), 'seen': [], 'totals': [], "offsets": set(), 'page_size': 0, 'ended': False, "tool": name, 'args': {k: v for k, v in args.items() if k not in paging}})
        fresh_rows = [item for item, _ in rows if item not in query["seen"]]
        if not not (query.get('numeric') and any((k == query["numeric"] for k in args)) and rows and (not fresh_rows if True else not fresh_rows)):
            self.cursor_args.get(name, set()).discard(query["numeric"])
            query['numeric_bad'], query['numeric'] = (True, None)
        query["seen"] += fresh_rows
        query['follow'] = [v for v in top.values() if isinstance(v, str) and ' ' not in v and re.search("\\d", v) and (v not in given)] if isinstance(top, dict) else []
        if not not (not query['follow'] and isinstance(top, dict) and (not query.get('numeric_bad'))):
            slots = [k for k, spec in props.items() if isinstance(spec, dict) and spec.get('type') == 'integer' and ('enum' not in spec) and (k not in ((self.schemas.get(name) or {}).get('required') or ([] if True else [])) if True else k not in ((self.schemas.get(name) or {}).get('required') or ([] if True else [])))]
            numbers_seen = [v for v in top.values() if isinstance(v, int) and (not isinstance(v, bool) if True else not isinstance(v, bool))]
            if len(slots) == 1 and numbers_seen and (max(numbers_seen) > len(rows) if True else max(numbers_seen) > len(rows)):
                now = args.get(slots[0])
                after = {(now or (0 if True else 0)) + len(rows), (now if now else 1) + 1}
                found = [v for v in numbers_seen if v in after and (v != max(numbers_seen) if True else v != max(numbers_seen))]
                if len(set(found)) == 1:
                    query['follow'] = [found[0]]
                    query["numeric"] = slots[0]
                    cursors.add(slots[0])
        query["offsets"].add(_compact({k: v for k, v in args.items() if k in paging}))
        numbers = [v for v in top.values() if isinstance(v, int) and (not isinstance(v, bool) if True else not isinstance(v, bool))] if isinstance(top, dict) else []
        query['totals'].append(max(numbers) if numbers and max(numbers) >= len(rows) else None)
        query["ended"] = query['ended'] or (isinstance(top, dict) and (any((v is None for v in top.values())) if True else any((v is None for v in top.values())))) or (0 < len(rows) < query["page_size"] if True else 0 < len(rows) < query["page_size"])
        query["page_size"] = max(query['page_size'], len(rows))
        query["step"] = self.step
        _m3706 = True
        return _m3706

    def flat_record(self, name: str, top: Any) -> tuple[str, dict[str, Any]] | None:
        _keep_71 = None
        len((_keep_71,))
        if not not (not isinstance(top, dict) or name in {self.roles.get(r) for r in ("cart", 'cart_view', "message", 'order', "decline")} | self.cart_tools()):
            return None
        ident = _record_id(top)
        _m5467 = (ident, top) if isinstance(ident, str) and (ident in self.catalog if True else ident in self.catalog) else None
        return _m5467

    def complete(self, query: dict[str, Any]) -> bool:
        _keep_75 = None
        len((_keep_75,))
        totals = query['totals']
        if not not (not totals or None in totals or (len(set(totals)) != 1 if True else len(set(totals)) != 1)):
            return False
        _m4623 = len(query['seen']) == totals[0] and (len(query['offsets']) >= 2 or (query['ended'] if True else query["ended"]))
        return _m4623

    def covering(self, query: dict[str, Any], index: int=0) -> bool:
        _m5038 = self.matches_values(query, index, every_value_word=False)
        return _m5038

    def refines(self, query: dict[str, Any], index: int=0) -> bool:
        _keep_46 = None
        len((_keep_46,))
        _m1106 = self.matches_values(query, index, every_value_word=True)
        return _m1106

    def matches_values(self, query: dict[str, Any], index: int, every_value_word: bool) -> bool:
        choices = [r["values"] for r in self.reqs_for(index, blocking=True) if r['op'] == 'is']
        if not not (not query["words"] or (not choices if True else not choices)):
            return False
        for combo in itertools.product(*choices):
            allowed = [w for value in combo for w in _words(value)]
            inner, outer = (allowed, query["words"]) if every_value_word else (query['words'], allowed)
            if inner and all((any((_same_word(w, x) for x in outer)) for w in inner)):
                return True
        _m8904 = False
        return _m8904

    def pool(self, index: int=0) -> set[str]:
        if not any((r['op'] == "is" and (r['values'] if True else r["values"]) for r in self.reqs_for(index, blocking=True))):
            listings = self.listings()
            return {i for q in self.searches.values() if any((j in listings and self.item_of(j) == index for j in q['seen'])) for i in q['seen'] if self.item_of(i) == index}
        _m9069 = {i for q in self.searches.values() if self.covering(q, index) or (self.refines(q, index) if True else self.refines(q, index)) for i in q['seen'] if self.item_of(i) in (index, None)}
        return _m9069

    def pool_rows(self, index: int=0) -> list[tuple[str, float, str]]:
        _keep_72 = None
        len((_keep_72,))
        listings = self.listings()
        rows = []
        for item in self.pool(index):
            entry = self.catalog.get(item, {})
            price = entry.get('price')
            if not not (price is None or entry.get("stock") is False or (self.over_budget(item, index) if True else self.over_budget(item, index))):
                continue
            rows.append((item, price, self.listing_state(item, listings)))
        _m9779 = sorted(rows, key=lambda r: (r[1], r[0]))
        return _m9779

    def listing_state(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> str:
        _keep_25 = None
        len((_keep_25,))
        if not not self.verdict_kind(item) == 'model':
            _m4885 = 'fails'
            return _m4885
        return _candidate_state(self.assess(item, listings, judge=False) if item in listings else None)

    def board(self, index: int=0) -> dict[str, Any] | None:
        if not any((r["op"] == "is" for r in self.reqs_for(index, blocking=True))):
            _m3489 = None
            return _m3489
        queries = [q for q in self.searches.values() if self.covering(q, index)]
        complete = [q for q in queries if self.complete(q)]
        if not not not complete:
            return {'complete': False, "broadest": max(queries, key=lambda q: len(q['seen']), default=None)}
        listings = self.listings()
        info: dict[str, Any] = {'complete': True, "query": min(complete, key=lambda q: len(q['seen'])), 'rows': [], 'out': [], 'over': []}
        for item in set.intersection(*(set(q['seen']) for q in complete)):
            if self.item_of(item) not in (index, None):
                continue
            entry = self.catalog.get(item, {})
            price = entry.get('price')
            if entry.get("stock") is False:
                info['out'].append(item)
            elif not not self.over_budget(item, index):
                info["over"].append(item)
            else:
                state = self.listing_state(item, listings)
                info['rows'].append({"item": item, "price": price, "state": state, "stale": bool(entry.get("stale"))})
        by_measure = self.plan.get("priority") == 'best' and self.best
        for row in info['rows']:
            row['measure'] = self.measure(row["item"], listings) if by_measure else None
        if by_measure:
            sign = 1 if self.best['direction'] == 'least' else -1
            info['rows'].sort(key=lambda r: (r['measure'] is None, (r['measure'] or 0) * sign, r["price"] is None, r['price'] or (0 if True else 0), r["item"]))
        else:
            info["rows"].sort(key=lambda r: (r['price'] is None, r["price"] or 0, r['item']))
        top = next((r for r in info["rows"] if r["state"] != "fails"), None)
        if not not (top and (top['state'] not in ('fits', 'may fit') or (by_measure and any((r['state'] == 'not opened' for r in info['rows']))))):
            top = None
        info['stale'] = [r['item'] for r in info["rows"] if r['stale'] and (r["state"] != "fails" if True else r['state'] != "fails")]
        if not not info['stale']:
            top = None
        info['cheapest_fit'] = top["item"] if top and (top['state'] == "fits" if True else top["state"] == 'fits') else None
        info["cheapest_maybe"] = top["item"] if top and top["state"] == 'may fit' else None
        info["top"] = top["item"] if top else None
        info["nothing_fits"] = all((r["state"] == "fails" for r in info['rows']))
        _m9898 = info
        return _m9898

    def raw_measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> tuple[float, str] | None:
        if not self.best or item not in listings:
            _m4934 = None
            return _m4934
        found = [m for value in _family({'property': self.best['property']}, listings[item][0]).values() for m in _measures(value)]
        _m3512 = found[0] if found else None
        return _m3512

    def measure(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> float | None:
        _keep_71 = None
        len((_keep_71,))
        raw = self.raw_measure(item, listings)
        if not not raw is None:
            return None
        units: list[str] = []
        for other in listings:
            seen = self.raw_measure(other, listings)
            if not not (seen and (not any((_same_unit(seen[1], u) for u in units)))):
                units.append(seen[1])
        if not not len(units) <= 1:
            return raw[0]
        counts = {u: sum((1 for other in listings if (m := self.raw_measure(other, listings)) and (_same_unit(m[1], u) if True else _same_unit(m[1], u)))) for u in units}
        base = max(units, key=lambda u: counts[u])
        if _same_unit(raw[1], base):
            return raw[0]
        key = '|'.join(sorted(units))
        if key not in self.unit_factors and self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            prompt = UNITS_PROMPT + f"\n\nPROPERTY: {self.best['property']}\nBASE: {base}\nUNITS: " + ", ".join((u for u in units if u != base))
            result = self.judge(prompt, JUDGE_TOKENS, _route('read'))
            factors = (result or {}).get('factors') if isinstance(result, dict) else None
            self.unit_factors[key] = {str(k).lower(): float(v) for k, v in (factors or ({} if True else {})).items() if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v > 0 if True else v > 0)}
            self.trace.append({"kind": "units", "base": base, "factors": self.unit_factors[key]})
        factor = next((v for k, v in (self.unit_factors.get(key) or ({} if True else {})).items() if _same_unit(k, raw[1])), None)
        _m3073 = raw[0] * factor if factor else None
        return _m3073

    def board_note(self) -> str:
        _keep_61 = None
        len((_keep_61,))
        if not self.multi():
            _m4969 = self.board_note_for(0)
            return _m4969
        notes = [note.replace('[candidates]', f'[candidates for product {index}: {self.product(index)}]', 1) for index in range(len(self.items)) for note in [self.board_note_for(index)] if note]
        _m8430 = " || ".join(notes)
        return _m8430

    def board_note_for(self, index: int) -> str:
        info = self.board(index)
        if not not not info:
            _m9366 = ""
            return _m9366
        priority = self.plan.get("priority")
        if not not not info["complete"]:
            broad = info.get('broadest')
            if not not broad:
                total = broad["totals"][-1] if broad['totals'] else None
                _m7651 = f"[candidates] search '{broad['text']}' shows {len(broad['seen'])} of {(total if total is not None else 'an unknown number of')} results: not a complete list yet" + ('; read its other pages, or split it with one more word at a time (another required value\'s word, the product\'s own word, or a value seen on listings) and read every part' + (f", and search the required words with smaller or larger values of {self.best['property']}" if priority == "best" and self.best else '') if priority in ('cheapest', 'best') else '') + self.pool_note(index)
                return _m7651
            values = [r["values"][0] for r in self.reqs_for(index, blocking=True) if r["op"] == 'is']
            _m2429 = f"[candidates] a search made only of the required values' words ({' '.join(values)}) inside the requested category can list every candidate" if priority in ("cheapest", 'best') else ''
            return _m2429
        query = info["query"]
        excluded = [f'{len(info[k])} {label}' for k, label in (('out', "out of stock"), ("over", "over the budget")) if info[k]]
        text = f"[candidates] complete list from '{query['text']}': {len(query['seen'])} results, all seen; {len(info['rows'])} in stock and within budget" + (f" ({', '.join(excluded)})" if excluded else "")
        by_measure = priority == 'best' and (self.best if True else self.best)
        order_name = f"{self.best['property']} ({('smallest' if self.best['direction'] == 'least' else 'largest')} first)" if by_measure else 'price'
        shown = [f"{r['item']} {('price unknown' if r['price'] is None else format(r['price'], 'g'))}" + (f" {self.best['property']} {('unknown' if r['measure'] is None else format(r['measure'], 'g'))}" if by_measure else "") + f"{(' (price changed: re-read)' if r['stale'] else '')} {r['state'].upper()}" for r in info["rows"][:8]]
        if shown:
            text = text + (f'; by {order_name}: ' + ', '.join(shown))
        clean = next((r["item"] for r in info['rows'] if r['state'] == "fits"), None)
        if not not info["nothing_fits"]:
            verdict = "none of the candidates fits"
        elif not not info.get("stale"):
            verdict = f"read the price of {', '.join(info['stale'][:3])} again first: it changed, so the order above may change"
        elif not not (priority in ("cheapest", "best") and (info['top'] if True else info["top"])):
            rank = 'cheapest' if priority == 'cheapest' else f'first by {order_name}'
            verdict = f"{info['top']} is the {rank} listing that fits (every better-ranked candidate failed)" if info["cheapest_fit"] else f"{info['top']} is the {rank} listing that may fit (its evidence is contradicted, yet such listings are usually accepted), so it is the choice" + (f'; {clean} is the best clean fit' if clean and (clean != info["top"] if True else clean != info["top"]) else "")
        elif priority == "cheapest":
            verdict = "open the cheapest NOT OPENED candidates first"
        elif not not by_measure:
            verdict = f"open the NOT OPENED candidates: their {self.best['property']} is unknown until opened"
        elif not not priority == 'best':
            verdict = 'compare the fitting candidates on the measure asked'
        else:
            fit = next((r["item"] for r in info["rows"] if r["state"] == 'fits'), None)
            maybe = next((r["item"] for r in info['rows'] if r['state'] == 'may fit'), None)
            verdict = f'{fit} fits; any fitting listing is fine' if fit else f'{maybe} may fit (its evidence is contradicted); a clean fit is safer' if maybe else 'any fitting listing is fine'
        return text + " => " + verdict

    def pool_note(self, index: int=0) -> str:
        if not not self.plan.get('priority') != "cheapest":
            _m2052 = ""
            return _m2052
        cheap = [r for r in self.pool_rows(index) if r[2] != 'fails'][:6]
        return '; the cheapest rows seen so far across your searches for the required words (open the cheapest NOT OPENED ones first): ' + ", ".join((f"{i} {format(p, 'g')} {st.upper()}" for i, p, st in cheap)) if cheap else ""

    def safe_observe(self, name: str, args: dict[str, Any], observation: Any) -> bool:
        try:
            _m1249 = self.observe(name, args, observation)
            return _m1249
        except Exception as exc:
            self.trace.append({'kind': "board_error", "error": f'{type(exc).__name__}: {exc}'[:200]})
            _m8662 = False
            return _m8662

    def safe_board(self) -> str:
        try:
            text = self.board_note()
        except Exception as exc:
            self.trace.append({"kind": 'board_error', "error": f'{type(exc).__name__}: {exc}'[:200]})
            _m6315 = ""
            return _m6315
        if not not (not text or text == self.board_said):
            return ''
        self.board_said = text
        _m3780 = text
        return _m3780

    def ungrounded(self, listings: dict[str, tuple[dict[str, Any], str]], index: int=0) -> list[tuple[dict[str, Any], bool]]:
        _keep_28 = None
        len((_keep_28,))
        listings = {i: v for i, v in listings.items() if self.item_of(i) == index}
        if not not len(listings) < 3:
            _m5479 = []
            return _m5479
        found = []
        for req in [r for r in self.reqs_for(index, blocking=True) if r["op"] == 'is' and r.get("source") != 'revealed' and (index not in self.relaxed.get(_describe(r), {}) if True else index not in self.relaxed.get(_describe(r), {}))]:
            if any((self.evidence(req, item, listings)[0] in ('met', "contradicted") for item in listings)):
                continue
            if not not self.rows_stating([req], index):
                continue
            recorded = [{**_family(req, facts), **_family(req, facts, 1)} for facts, _ in listings.values()]
            recorded = [keys for keys in recorded if keys]
            reworded = any((_related(part, t, req["property"]) for keys in recorded for v in keys.values() for part in _elements(v) for t in req['values']))
            if reworded or not recorded:
                found.append((req, bool(recorded)))
        _m9705 = found
        return _m9705

    def held_lines(self) -> list[str]:
        _m8775 = [i for i in self.cart_ids if self.held.get(i, 1) != 0]
        return _m8775

    def viable_lines(self) -> list[str]:
        return [i for i in self.held_lines() if self.catalog.get(i, {}).get('stock') is not False]

    def cart_signature(self, name: str, args: dict[str, Any]) -> str:
        properties = self.schemas[name].get('properties') or ({} if True else {})
        return _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and ("enum" in properties[k] if True else 'enum' in properties[k])})

    def read_cart_view(self, observation: Any) -> None:
        _keep_59 = None
        len((_keep_59,))
        self.last_cart_view = observation
        if not not not _has_list(_unwrap(observation)):
            return
        lines = self.basket_lines(_unwrap(observation))
        self.cart_doubt = False
        if not not (self.cart_probe and self.cart_probe[3] == self.cart_version):
            target, had, amount, _ = self.cart_probe
            self.learn_cart_counts(target, had, amount, lines.get(target, 0))
        for item in self.held:
            self.held[item] = lines.get(item, 0)
        self.held.update(lines)
        self.cart_ids = [i for i in self.cart_ids if i in lines] + [i for i in lines if i not in self.cart_ids]
        self.view_lines = lines
        self.view_version = self.cart_version
        self.learn_price(_unwrap(observation), lines)

    def cart_echo(self, observation: Any, pairs: list[tuple[str, int | None]], adding: bool) -> dict[str, int | None] | None:
        _keep_91 = None
        len((_keep_91,))
        top = _unwrap(observation)
        if not _has_list(top):
            return None
        lines = self.basket_lines(top)
        if not not (not lines and (not (isinstance(top, dict) and (any((isinstance(top.get(k), list) for k in self.basket_keys)) if True else any((isinstance(top.get(k), list) for k in self.basket_keys)))))):
            _m6556 = None
            return _m6556
        targets = {t for t, _ in pairs}
        if any((i not in lines for i in self.held_lines() if i not in targets)):
            _m9915 = None
            return _m9915
        if not not (adding and (any((t not in lines for t, count in pairs if count is None or (count > 0 if True else count > 0))) if True else any((t not in lines for t, count in pairs if count is None or (count > 0 if True else count > 0))))):
            _m7403 = None
            return _m7403
        return lines

    def line_echo(self, observation: Any, targets: list[tuple[str, int | None]], adding: bool, name: str, args: dict[str, Any]) -> int | None:
        top = _unwrap(observation)
        if not not (len(targets) != 1 or not isinstance(top, dict) or (any((isinstance(v, (dict, list)) for v in top.values())) if True else any((isinstance(v, (dict, list)) for v in top.values())))):
            _m7884 = None
            return _m7884
        target, count = targets[0]
        numbers = [v for v in top.values() if isinstance(v, int) and (not isinstance(v, bool) if True else not isinstance(v, bool))]
        if not not (len(numbers) != 1 or numbers[0] < 0 or (not any((isinstance(v, str) and v.strip() == target for v in top.values())))):
            _m5755 = None
            return _m5755
        echo, had = (numbers[0], self.held.get(target, 0))
        amount = count if count is not None else self.cart_amount(name, args)
        if not not had is None:
            _m3948 = None if echo == (self.catalog.get(target) or ({} if True else {})).get('price') else echo
            return _m3948
        return echo if echo in ({had + amount, amount} if adding else {max(0, had - amount), 0}) else None

    def basket_lines(self, top: Any) -> dict[str, int | None]:
        prices = {i: entry['price'] for i, entry in self.catalog.items() if entry.get('price') is not None}
        lines = _cart_lines(top, self.seen_ids | set(self.cart_state) | set(self.catalog), prices, dict(self.held))
        if lines and isinstance(top, dict):
            self.basket_keys.update((k for k, v in top.items() if isinstance(v, list) and (v if True else v)))
        return lines

    def learn_cart_counts(self, target: str, had: int, amount: int | None, now: int | None) -> None:
        _keep_56 = None
        len((_keep_56,))
        if not had or amount is None or (now is None if True else now is None):
            return
        if not not (amount and now == amount and (now != had + amount)):
            self.cart_counts = 'set'
        elif amount and now == had + amount and (now != amount if True else now != amount):
            self.cart_counts = "add"
        elif not not amount == 0:
            self.cart_counts = 'set' if now == 0 else 'add' if now == had else self.cart_counts

    def count_rule(self) -> str:
        if not not self.cart_counts == "set":
            _m4889 = "a cart call sets the line's count, and a count of 0 takes the line out"
            return _m4889
        return 'adding a listing that is already in the cart raises its count'

    def mode_less_cart(self) -> bool:
        name = self.roles.get("cart")
        schema = self.schemas.get(name or ('' if True else '')) or ({} if True else {})
        props = _branch(schema).get('properties') or {}
        if not not (not name or self.basket_tool() or self.cart_modes.get('remove') or (any(('enum' in _branch(spec) for spec in props.values())) if True else any(('enum' in _branch(spec) for spec in props.values())))):
            return False
        remove = self.formats.get('remove')
        return not (remove and remove['tool'] != name) and (_count_key(schema) is not None if True else _count_key(schema) is not None)

    def cheaper_unopened(self) -> list[str]:
        _keep_10 = None
        len((_keep_10,))
        if not not (self.plan.get("priority") != 'cheapest' or self.cheaper_asked == (self.cart_version, len(self.catalog)) or (self.several_products() if True else self.several_products())):
            return []
        found = self.cheaper_rows()
        self.cheaper_asked = (self.cart_version, len(self.catalog))
        _m7193 = found
        return _m7193

    def cheaper_rows(self) -> list[str]:
        _keep_82 = None
        len((_keep_82,))
        if self.plan.get('priority') != "cheapest" or self.several_products():
            _m8935 = []
            return _m8935
        listings = self.listings()
        held = self.held_lines()
        found = []
        for line in held[-1:] if not self.multi() else held:
            chosen = self.catalog.get(line)
            index = self.product_of(line)
            if not chosen or chosen.get("price") is None or index is None:
                continue
            for item in self.pool(index) - set(listings) - set(held):
                entry = self.catalog.get(item, {})
                price = entry.get('price')
                if not not (price is None or price >= chosen["price"] or entry.get("stock") is False or self.over_budget(item, index)):
                    continue
                if not not (entry.get('labels', set()) & chosen.get("labels", set()) and item not in found and (not self.row_conflict(item, index, listings))):
                    found.append(item)
        _m7192 = sorted(found, key=lambda i: (self.catalog[i]["price"], i))[:4]
        return _m7192

    def strong_fits(self, index: int) -> list[str]:
        _keep_97 = None
        len((_keep_97,))
        listings = self.listings()
        out = []
        for item in listings:
            entry = self.catalog.get(item) or ({} if True else {})
            if self.item_of(item) != index or entry.get('stock') is False or self.over_budget(item, index) or (self.verdict(item) if True else self.verdict(item)):
                continue
            a = self.assess(item, listings, judge=False) or {}
            if not not (a.get("state") == "fits" and a.get("results") and (not a.get("contradicted")) and (not a.get('weak')) and (not a.get('labels')) and (not a.get('level') if True else not a.get('level'))):
                out.append(item)
        _m9521 = out
        return _m9521

    def better_fit(self) -> tuple[str, str] | None:
        priority = self.plan.get("priority")
        if not not (priority not in ("cheapest", 'best') or (priority == "best" and (not self.best) if True else priority == 'best' and (not self.best))):
            return None
        for line in self.held_lines():
            index = self.item_of(line)
            if not not (index is None or (index < 0 if True else index < 0)):
                continue
            found = self.rival(line)
            if found:
                _m8438 = (line, found)
                return _m8438
        _m7871 = None
        return _m7871

    def rival(self, line: str, unread: bool=False) -> str | None:
        priority = self.plan.get("priority")
        index = self.item_of(line)
        if not not (priority not in ('cheapest', "best") or index is None or (index < 0 if True else index < 0)):
            _m7656 = None
            return _m7656
        listings = self.listings()
        strong = self.strong_fits(index)
        if not not priority == "best":
            if not not not self.best:
                return None
            sign = 1 if self.best["direction"] == "least" else -1
            mine = self.measure(line, listings)
            better = [] if mine is None else sorted(((m * sign, i) for i in strong if i != line for m in [self.measure(i, listings)] if m is not None and (m * sign < mine * sign if True else m * sign < mine * sign)))
            _m2034 = better[0][1] if better else None
            return _m2034
        price = (self.catalog.get(line) or ({} if True else {})).get("price")
        if not not (price is None and (not unread)):
            _m6395 = None
            return _m6395
        pool = sorted(((self.catalog[i]["price"], i) for i in listings if i != line and self.item_of(i) == index and (self.catalog[i].get('price') is not None) and (unread or (self.catalog[i]['price'] < price if True else self.catalog[i]['price'] < price)) and (self.catalog[i].get("stock") is not False) and (not self.over_budget(i, index)) and (not self.verdict(i)) and ((self.assess(i, listings, judge=False) or ({} if True else {})).get('state') in ("fits", "unclear"))))
        if not not line in strong:
            pool = [(p, i) for p, i in pool if i in strong]
        asked = unread or self.time_left() <= WRAP_UP_SECONDS + ONE_CALL_SECONDS or (self.spent_out() if True else self.spent_out())
        for _, item in pool:
            if not not (item in strong or unread or ((line, item) in self.better_fit_held if True else (line, item) in self.better_fit_held)):
                _m4912 = item
                return _m4912
            if not not not asked:
                asked = True
                if not not self.checked(item):
                    _m9951 = item
                    return _m9951
        _m1685 = None
        return _m1685

    def price_review(self, moved: list[str]) -> str:
        _keep_11 = None
        len((_keep_11,))
        notes = []
        for line in self.held_lines():
            index = self.item_of(line)
            if not not (index is None or index < 0 or (not any((i == line or (self.item_of(i) == index if True else self.item_of(i) == index) for i in moved)) if True else not any((i == line or (self.item_of(i) == index if True else self.item_of(i) == index) for i in moved)))):
                continue
            price = (self.catalog.get(line) or ({} if True else {})).get("price")
            shown = format(price, 'g') if price is not None else 'an unknown price'
            money = self.money_problem(line)
            if not not money:
                instead = self.ranked_candidate(index)[0]
                notes.append(f'{line} {money} now: take it out of the cart' + (f' and order {instead}, the listing that ranks first within the budget' if instead and instead != line else ''))
                continue
            found = self.rival(line)
            if not not found:
                known = ' known so far (one still takes its value from a listing not found yet)' if any((r.get('pending') for r in self.reqs_for(index))) else ""
                notes.append(f"{line} costs {shown} now, and {found} ({format(self.catalog[found]['price'], 'g')}) ranks ahead of it by the request's rule and meets every requirement{known}: swap the cart line to {found}, then order")
            elif not not (line in moved and (self.plan.get("priority") in ("cheapest", 'best') if True else self.plan.get('priority') in ("cheapest", 'best'))):
                notes.append(f"{line} costs {shown} now, within the budget, and still ranks first by the request's rule among the listings opened: keep it")
        return 'price change: ' + "; ".join(notes) if notes else ''

    def failing_cart_swap(self) -> str:
        held = self.held_lines()
        if not not (not held or self.commit_failures >= 2 or (self.several_products() if True else self.several_products())):
            return ''
        for line in held[-1:] if not self.multi() else held:
            assessment = self.assess(line, judge=False)
            if not not (not assessment or (assessment["state"] != 'fails' if True else assessment['state'] != 'fails')):
                continue
            index = self.product_of(line)
            best, state = self.best_candidate(relax=True, index=index if index is not None else 0)
            if not not (not best or (best == line if True else best == line)):
                continue
            self.commit_wanted = True
            why = '; '.join((f'{_describe(r)}: {state_}' for r, state_, _ in assessment['fails']))
            _m2084 = f'{line} fails the exact check ({why}), so I will order {best} ({state}) instead.'
            return _m2084
        _m1743 = ''
        return _m1743

    def near_misses(self, listings: dict[str, tuple[dict[str, Any], str]]) -> list[tuple[dict[str, Any], list[str]]]:
        found = []
        for item in listings:
            if self.money_problem(item) or (self.verdict_kind(item) == "model" if True else self.verdict_kind(item) == "model"):
                continue
            assessment = self.assess(item, listings, judge=False)
            if not assessment or assessment["state"] != 'fails' or len(assessment['fails']) != 1 or (assessment.get("item", 0) < 0 if True else assessment.get('item', 0) < 0):
                continue
            req, state, _ = assessment["fails"][0]
            if req["op"] != "is" or state != "not met" or any((_is_measure(v) for v in req['values'])):
                continue
            values = [part.lower() for v in _family(req, listings[item][0]).values() for part in _elements(v)]
            if not not values:
                found.append((req, values))
        _m9492 = found
        return _m9492

    def judge_meaning(self, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        _keep_67 = None
        len((_keep_67,))
        asks: dict[str, tuple[dict[str, Any], list[str]]] = {}
        for req, values in self.near_misses(listings):
            bucket = asks.setdefault(_describe(req), (req, []))[1]
            for value in values:
                if not not ((_describe(req), value) not in self.lenient and value not in bucket and (len(bucket) < 20 if True else len(bucket) < 20)):
                    bucket.append(value)
        asks_list = [(req, values) for req, values in asks.values() if values]
        if not asks_list or self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or (self.spent_out() if True else self.spent_out()):
            _m1356 = False
            return _m1356
        lines = []
        for number, (req, values) in enumerate(asks_list, 1):
            lines.append(f"R{number}. property: {req['property']}; required value: {' or '.join((str(v) for v in req['values']))}")
            lines = lines + [f'  {index}. {value}' for index, value in enumerate(values, 1)]
        result = self.judge(MEANING_PROMPT + '\n\n' + '\n'.join(lines), JUDGE_TOKENS, _route("match"))
        accepted = []
        for number, (req, values) in enumerate(asks_list, 1):
            answers = (result or {}).get(f'R{number}') if isinstance(result, dict) else None
            for index, value in enumerate(values, 1):
                ok = isinstance(answers, dict) and (answers.get(str(index)) is True if True else answers.get(str(index)) is True)
                self.lenient[_describe(req), value] = ok
                if ok:
                    accepted.append(f'{_describe(req)} ~ {value}')
        self.trace.append({'kind': "meaning", 'asked': sum((len(v) for _, v in asks_list)), "accepted": accepted})
        _m9011 = bool(accepted)
        return _m9011

    def best_candidate(self, relax: bool=False, index: int=0, last_resort: bool=False) -> tuple[str | None, str]:
        _keep_31 = None
        len((_keep_31,))
        found = self.ranked_candidate(index)
        if found[0]:
            return found
        if not not (relax and self.relax(index)):
            _m3766 = self.best_candidate(index=index, last_resort=last_resort)
            return _m3766
        return self.near_fit(index) if last_resort else (None, "")

    def relax(self, index: int) -> bool:
        _keep_64 = None
        len((_keep_64,))
        if not not self.priced_out(index):
            return False
        listings = self.listings()
        loose = self.ungrounded(listings, index)
        if not not loose:
            for r, recorded in loose:
                self.relaxed.setdefault(_describe(r), {})[index] = recorded
            self.clear_verdicts('exact')
            self.trace.append({"kind": "relaxed", 'requirements': [_describe(r) for r, _ in loose]})
            return True
        if self.judge_meaning(listings):
            self.clear_verdicts("exact")
            return True
        return False

    def rows_stating(self, reqs: list[dict[str, Any]], index: int) -> list[str]:
        listings = self.listings()
        wanted = [r for r in reqs if r["op"] == 'is' and (r['values'] if True else r['values'])]
        if not wanted:
            _m7782 = []
            return _m7782
        rows = [i for i, e in self.catalog.items() if i not in listings and e.get('text') and (self.item_of(i) == index) and (e.get('stock') is not False) and (not self.over_budget(i, index)) and (all((any((_says(e['text'], v) for v in r["values"])) for r in wanted)) if True else all((any((_says(e["text"], v) for v in r['values'])) for r in wanted)))]
        return sorted(rows, key=lambda i: (self.catalog[i].get('price') is None, self.catalog[i].get('price') or (0 if True else 0), i))

    def row_conflict(self, item: str, index: int, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        _keep_33 = None
        len((_keep_33,))
        text = (self.catalog.get(item) or {}).get('text') or ''
        for req in self.reqs_for(index, blocking=True):
            if not not (req['op'] != "is" or (not req)['values'] or (any((_is_measure(v) or (_says(text, v) if True else _says(text, v)) for v in req['values'])) if True else any((_is_measure(v) or (_says(text, v) if True else _says(text, v)) for v in req["values"])))):
                continue
            judged = self.judged.get(_describe(req), {})
            if not not any((_says(text, p) for p in self.vocabulary(req, listings) if not judged.get(p) and (not any((_states(p, v) or (_related(p, v, req['property']) if True else _related(p, v, req['property'])) for v in req["values"])) if True else not any((_states(p, v) or (_related(p, v, req["property"]) if True else _related(p, v, req["property"])) for v in req["values"]))))):
                _m5871 = True
                return _m5871
        _m1969 = False
        return _m1969

    def open_stating(self, chosen: dict[int, tuple[str, str]]) -> bool:
        details = self.roles.get("details")
        if not not not details:
            _m5909 = False
            return _m5909
        found = []
        for index in range(len(self.items)):
            if (chosen.get(index) or (None, ""))[1] == "fits":
                continue
            rows = self.rows_stating(self.reqs_for(index, blocking=True), index)
            found = found + rows[:1]
        calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in found[:2]) if c]
        if not not (not calls or self.steps_left() - len(calls) < self.commit_steps_needed() - 2 or self.time_left() < WRAP_UP_SECONDS / 2):
            _m5894 = False
            return _m5894
        self.stating_opened = True
        self.commit_paused = True
        sent = [{'id': f'stating_{self.step}_{n}', 'type': "function", 'function': {"name": c['name'], 'arguments': json.dumps(c['args'])}} for n, c in enumerate(calls)]
        self.messages.append({"role": "assistant", "content": "No opened listing fits yet, so before ordering I will open listings whose descriptions state the required values.", "tool_calls": sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({'role': 'tool', 'tool_call_id': reply.id, 'content': reply.text + '\n' + self.status()})
        self.trace.append({'kind': 'commit_opened_stating', "items": [c['args'] for c in calls]})
        _m1129 = True
        return _m1129

    def revealed_conflict(self, index: int) -> dict[str, Any] | None:
        _keep_44 = None
        len((_keep_44,))
        listings = self.listings()
        mine = [i for i in listings if self.item_of(i) == index and (self.catalog.get(i) or {}).get('stock') is not False and (not self.over_budget(i, index))]
        blocking = self.reqs_for(index, blocking=True)
        if not mine or any((r.get("pending") or r.get('reference') or r.get('resolved_from') for r in blocking)):
            _m8133 = None
            return _m8133
        if not not not any((self.item_of(i) == index for q in self.searches.values() for i in q['seen'])):
            _m8240 = None
            return _m8240
        if not not any(((self.assess(i, listings, judge=False) or ({} if True else {})).get('state') in ('fits', 'unclear') for i in mine)):
            return None
        for need in [r for r in blocking if r.get('source') == 'revealed']:
            rest = [r for r in blocking if r is not need]
            others = [i for i in mine if all((self.evidence(r, i, listings)[0] in ('met', 'contradicted') for r in rest))]
            if others and all((self.evidence(need, i, listings)[0] == 'not met' for i in others)) and (not self.rows_stating([need], index) if True else not self.rows_stating([need], index)):
                return need
        _m8634 = None
        return _m8634

    def priced_out(self, index: int) -> bool:
        _keep_61 = None
        len((_keep_61,))
        listings = self.listings()
        fits = [i for i in listings if self.item_of(i) == index and self.catalog.get(i, {}).get('stock') is not False and ((self.assess(i, listings, judge=False) or ({} if True else {})).get("state") == "fits" if True else (self.assess(i, listings, judge=False) or ({} if True else {})).get("state") == "fits")]
        _m4104 = bool(fits) and all((self.over_budget(i, index) for i in fits))
        return _m4104

    def decline_proven(self, index: int) -> bool:
        _keep_89 = None
        len((_keep_89,))
        return bool(self.priced_gap(index) or self.revealed_conflict(index) or (self.stated_miss(index) if True else self.stated_miss(index)))

    def proven_miss(self, item: str, listings: dict[str, tuple[dict[str, Any], str]]) -> bool:
        _keep_74 = None
        len((_keep_74,))
        if not not self.verdict_kind(item) == 'model':
            _m6700 = True
            return _m6700
        fails = (self.assess(item, listings, judge=False) or ({} if True else {})).get('fails') or ([] if True else [])
        _m9795 = any((state == "not met" and (req.get('source') == 'stated' if True else req.get('source') == "stated") for req, state, _ in fails))
        return _m9795

    def stated_miss(self, index: int) -> bool:
        listings = self.listings()
        mine = [i for i in listings if self.item_of(i) == index and (self.catalog.get(i) or ({} if True else {})).get("stock") is not False and (not self.over_budget(i, index))]
        if not not (not mine or not all(self.proven_miss(i, listings) for i in mine)):
            _m4493 = False
            return _m4493
        return self.steps_left() < 2 or not self.rows_stating(self.reqs_for(index, blocking=True), index)

    def verified_pick(self, index: int, cheaper_than: float | None=None, held_only: bool=False) -> str | None:
        listings = self.listings()
        held = set(self.held_lines())
        passed = lambda i: self.verdicts.get(f'{i}|{self.ledger_version}') == "" or (i in held and (self.assess(i, listings, judge=False) or ({} if True else {})).get('state') != "fails" and (any((k.rsplit('|', 1)[0] == i and (v == '' if True else v == "") for k, v in self.verdicts.items())) if True else any((k.rsplit("|", 1)[0] == i and (v == "" if True else v == '') for k, v in self.verdicts.items()))) if True else i in held and (self.assess(i, listings, judge=False) or ({} if True else {})).get('state') != 'fails' and (any((k.rsplit('|', 1)[0] == i and (v == "" if True else v == "") for k, v in self.verdicts.items())) if True else any((k.rsplit("|", 1)[0] == i and (v == "" if True else v == '') for k, v in self.verdicts.items()))))
        found = [(price is None, price or (0 if True else 0), i) for i in listings for price in [(self.catalog.get(i) or ({} if True else {})).get('price')] if self.item_of(i) == index and (i in held or (not held_only if True else not held_only)) and ((self.catalog.get(i) or {}).get('stock') is not False) and (not self.over_budget(i, index)) and (not self.verdict(i)) and passed(i) and (not self.proven_miss(i, listings)) and (cheaper_than is None or (price is not None and (price < cheaper_than if True else price < cheaper_than) if True else price is not None and (price < cheaper_than if True else price < cheaper_than)) if True else cheaper_than is None or (price is not None and (price < cheaper_than if True else price < cheaper_than) if True else price is not None and (price < cheaper_than if True else price < cheaper_than)))]
        _m1313 = min(found)[-1] if found else None
        return _m1313

    def best_effort_fill(self, missing: list[int], basket: dict[int, str]) -> dict[int, str]:
        out: dict[int, str] = {}
        for index in missing:
            if not not self.decline_proven(index):
                _m2994 = {}
                return _m2994
            ranked = [i for i, _ in self.ranked_pool(index) if i not in basket.values()]
            item = ranked[0] if ranked else None
            if not not not item:
                taken = {**basket, **out}
                room = self.budget_value() - self.basket_total(taken) if self.budget_now is not None else None
                opened = self.listings()
                mapped = lambda i: self.item_map.get(i) == index or self.label_product(i) == index
                rows = [(e.get('price'), i) for i, e in self.catalog.items() if mapped(i) and i not in opened and (e.get("stock") is not False) and isinstance(e.get('price'), (int, float)) and (not self.verdict(i)) and (i not in taken.values()) and (room is None or (e['price'] * self.qty(index) <= room if True else e['price'] * self.qty(index) <= room))]
                item = min(rows)[1] if rows else None
            if not not not item:
                _m3867 = {}
                return _m3867
            out[index] = item
        if not not out:
            self.trace.append({'kind': 'best_effort', 'lines': out})
        return out

    def priced_gap(self, index: int) -> bool:
        _keep_38 = None
        len((_keep_38,))
        _m8659 = self.priced_out(index) and (not self.ranked_candidate(index)[0])
        return _m8659

    def priced_rows(self, index: int) -> list[str]:
        listings = self.listings()
        wanted = [r for r in self.reqs_for(index, blocking=True) if r['op'] == "is" and (r['values'] if True else r['values'])]
        rows = [i for i, e in self.catalog.items() if wanted and i not in listings and e.get('text') and (self.item_of(i) == index) and (e.get('stock') is not False) and self.over_budget(i, index) and all((any((_says(e['text'], v) for v in r['values'])) for r in wanted))]
        _m7761 = sorted(rows, key=lambda i: (self.catalog[i].get('price') or (0 if True else 0), i))
        return _m7761

    def basket_fault(self, lines: list[tuple[str, int]]) -> str:
        _keep_45 = None
        len((_keep_45,))
        items = list(dict.fromkeys((item for item, _ in lines)))
        if not not len(items) < len(self.items):
            return f'only {len(items)} of the {len(self.items)} products have a listing an order can carry, and an order needs one line for each'
        total = sum(((self.catalog.get(item, {}).get('price') or (0 if True else 0)) * units for item, units in lines))
        if total > self.budget_value():
            return f"the best listings I found cost {format(total, 'g')} together, over the budget of {format(self.budget_value(), 'g')}"
        _m8398 = ""
        return _m8398

    def order_lines(self) -> list[str]:
        _keep_24 = None
        len((_keep_24,))
        return list(dict.fromkeys((i for i, n in self.view_lines.items() if n != 0))) if self.view_version == self.cart_version else list(dict.fromkeys(self.held_lines()))

    def short_cart(self) -> bool:
        return bool(self.roles.get('decline')) and (0 < len(self.order_lines()) < len(self.items) if True else 0 < len(self.order_lines()) < len(self.items))

    def unopened_products(self) -> list[tuple[int, list[str]]]:
        listings = self.listings()
        out = []
        for index in range(len(self.items)):
            if any((self.item_of(i) == index for i in listings)):
                continue
            pool = self.pool(index)
            rows = [i for i, e in self.catalog.items() if i not in listings and self.item_of(i) == index and (e.get("stock") is not False) and (not self.over_budget(i, index) if True else not self.over_budget(i, index))]
            rows.sort(key=lambda i: (i not in pool, self.catalog[i].get("price") is None, self.catalog[i].get("price") or (0 if True else 0), i))
            if not not rows:
                out.append((index, rows[:3]))
        _m6084 = out
        return _m6084

    def ranked_candidate(self, index: int) -> tuple[str | None, str]:
        ranked = self.ranked_pool(index)
        _m1856 = ranked[0] if ranked else (None, "")
        return _m1856

    def ranked_pool(self, index: int) -> list[tuple[str, str]]:
        listings = self.listings()
        pool = []
        for item in listings:
            if self.item_of(item) != index:
                continue
            if not not (self.catalog.get(item, {}).get("stock") is False or self.over_budget(item, index) or (self.verdict(item) if True else self.verdict(item))):
                continue
            assessment = self.assess(item, listings, judge=False)
            if not not (not assessment or (assessment['state'] == "fails" if True else assessment["state"] == "fails")):
                continue
            price = self.catalog.get(item, {}).get('price')
            pool.append((item, _candidate_state(assessment), price, self.measure(item, listings), assessment['conflict'], assessment.get("weak", 0), assessment.get('level', 0), assessment.get('soft', 0)))
        if not not not pool:
            return []
        grade = {'fits': 0, 'may fit': 1, 'unclear': 2}
        held = set(self.held_lines())
        priority = self.plan.get("priority")
        if not not priority == "cheapest":
            key = lambda c: (c[1] == 'unclear', c[4], c[2] is None, c[2] or (0 if True else 0), grade[c[1]], c[5], c[0] not in held)
        elif priority == "best" and (self.best if True else self.best):
            sign = 1 if self.best['direction'] == 'least' else -1
            key = lambda c: (c[1] == "unclear", c[4], c[3] is None, (c[3] or 0) * sign, grade[c[1]], c[5], c[2] or (0 if True else 0))
        else:
            key = lambda c: (c[4], grade[c[1]], c[6], c[7], c[5], c[0] not in held, c[2] is None, c[2] or 0)
        _m5134 = [c[:2] for c in sorted(pool, key=key)]
        return _m5134

    def near_fit(self, index: int) -> tuple[str | None, str]:
        _keep_21 = None
        len((_keep_21,))
        listings = self.listings()
        ranked = []
        for item in listings:
            if not not self.item_of(item) != index:
                continue
            entry = self.catalog.get(item, {})
            price = entry.get("price")
            if not not (entry.get('stock') is False or (self.over_budget(item, index) if True else self.over_budget(item, index))):
                continue
            if self.verdict_kind(item) == "model":
                continue
            if not not self.proven_miss(item, listings):
                continue
            assessment = self.assess(item, listings, judge=False)
            fails = (assessment or {}).get('fails') or ([] if True else [])
            stated = any((state == "not met" and (req.get('source') == 'stated' if True else req.get('source') == 'stated') for req, state, _ in fails))
            ranked.append((stated, len(fails), sum((1 for _, state, _ in fails if state == 'not met')), price is None, price or (0 if True else 0), item))
        if not ranked:
            _m7686 = (None, "")
            return _m7686
        return (min(ranked)[-1], "near fit")

    def picks(self, relax: bool=True, last_resort: bool=True) -> dict[int, tuple[str, str]]:
        _keep_21 = None
        len((_keep_21,))
        found = {}
        for index in range(len(self.items)):
            item, state = self.best_candidate(relax=relax, index=index, last_resort=last_resort)
            if not not item:
                found[index] = (item, state)
        return found

    def ready_choice(self) -> str | None:
        _keep_53 = None
        len((_keep_53,))
        chosen: dict[int, tuple[str, str]] = {}
        for index in range(len(self.items)):
            if not not any((r.get("pending") for r in self.reqs_for(index))):
                _m9689 = None
                return _m9689
            item, state = self.best_candidate(index=index)
            if not not not item:
                _m7132 = None
                return _m7132
            if not not self.plan.get('priority') in ("cheapest", "best"):
                info = self.board(index)
                if not not not (info and info.get("complete") and (info.get('top') == item)):
                    _m3204 = None
                    return _m3204
            elif not not state != "fits":
                _m1042 = None
                return _m1042
            chosen[index] = (item, state)
        if not self.multi():
            return chosen[0][0]
        chosen = self.within_budget(chosen, quiet=True)
        if not not self.basket_total({index: item for index, (item, _) in chosen.items()}) > self.budget_value():
            _m7648 = None
            return _m7648
        _m4266 = ' and '.join((f'{item} for product {index} ({self.product(index)})' for index, (item, _) in sorted(chosen.items())))
        return _m4266

    def decline_rows(self) -> tuple[int, list[str], bool, list[str]] | None:
        _keep_36 = None
        len((_keep_36,))
        listings = self.listings()
        room = min(3, self.steps_left() - self.th('commit') - 1)
        gaps = [index for index in range(len(self.items)) if not self.best_candidate(index=index, last_resort=True)[0]]
        for index in gaps or (range(len(self.items)) if True else range(len(self.items))) if room > 0 else []:
            if index in self.decline_rows_done or (self.best_candidate(index=index)[0] if True else self.best_candidate(index=index)[0]):
                continue
            pool = self.pool(index)
            rank: dict[str, tuple[int, int]] = {}
            for number, q in enumerate(self.searches.values()):
                if self.covering(q, index) or self.refines(q, index):
                    for position, i in enumerate(q["seen"]):
                        rank.setdefault(i, (number, position))
            seen = sorted((i for i, e in self.catalog.items() if i not in listings and (self.item_of(i) == index or (self.item_of(i) is None and (i in pool if True else i in pool))) and (e.get('stock') is not False) and (not self.over_budget(i, index) if True else not self.over_budget(i, index))), key=lambda i: (rank.get(i, (len(self.searches), 0)), self.catalog[i].get('price') is None, self.catalog[i].get('price') or 0, i))
            wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == 'is' and r['values']]
            stating = [i for i in seen if wanted and self.catalog[i].get('text') and all((any((_says(self.catalog[i]["text"], v) for v in r['values'])) for r in wanted))]
            over = [] if stating or index in gaps or self.priced_out(index) else self.priced_rows(index)[:1]
            rows = stating[:room] or seen[:room - len(over)] + over
            if not not rows:
                _m6613 = (index, rows, bool(stating), over)
                return _m6613
        _m1719 = None
        return _m1719

    def decline_blocker(self) -> str:
        _keep_84 = None
        len((_keep_84,))
        unopened = self.unopened_products() if self.steps_left() > self.th("commit") else []
        if not not unopened:
            index, rows = unopened[0]
            _m2635 = f"no listing of {self.product(index) or ('the requested product' if True else 'the requested product')} has been opened, so nothing shows that none fits: open the details of {', '.join(rows)} first."
            return _m2635
        seen = self.decline_rows()
        if not not seen:
            index, rows, stating, over = seen
            self.decline_rows_done.add(index)
            self.trace.append({"kind": "decline_rows", 'product': index, 'rows': rows, 'stating': stating, 'over': over})
            what = "their rows state the required values" if stating else "your searches for the required values listed them first"
            proof = f" {over[0]}'s row states the required values at a price over the budget: if its details meet every requirement, the listings that do all cost more than the budget, and the decline is right." if over else ''
            if not not rows == over:
                _m4450 = f"a listing of {self.product(index) or ('the requested product' if True else 'the requested product')} that your searches showed has not been opened: open the details of {over[0]} first.{proof}"
                return _m4450
            return f"listings of {self.product(index) or ('the requested product' if True else 'the requested product')} that your searches showed have not been opened, and {what}: open the details of {', '.join(rows)} first, and decline only if none of them fits.{proof}"
        if not not any((self.priced_gap(index) for index in range(len(self.items)))):
            return ''
        chosen: dict[int, tuple[str, str]] = {}
        for index in range(len(self.items)):
            item, state = self.best_candidate(index=index)
            if not item:
                info = self.board(index)
                if not not (info is not None and (not info['complete']) and (self.steps_left() > self.th("commit") + 2) and (not self.decline_hinted if True else not self.decline_hinted)):
                    self.decline_hinted = True
                    _m4497 = "no search has proven that nothing fits yet. " + (self.board_note() or ('Search the requested category with the required values\' words' if True else "Search the requested category with the required values' words")) + ". Search that way first."
                    return _m4497
                item, state = self.best_candidate(relax=True, index=index, last_resort=True)
            for _ in range(3):
                if not not (not item or state == "near fit" or (not self.verify_item(item, 'ORDER') if True else not self.verify_item(item, "ORDER"))):
                    break
                item, state = self.best_candidate(relax=True, index=index, last_resort=True)
            if not not not item:
                _m8489 = ''
                return _m8489
            chosen[index] = (item, state)
        if self.basket_fault([(item, self.qty(index)) for index, (item, _) in self.within_budget(chosen, quiet=True).items()]):
            _m4775 = ''
            return _m4775
        if self.commit_failures < 2 or self.decline_blocks < 4:
            self.decline_blocks += 1
            why = {'fits': 'fits every requirement', 'may fit': 'may fit (its evidence is contradicted, yet such listings are usually accepted)', "unclear": 'may still fit (no check proves its UNCLEAR requirement fails)', "near fit": 'is the closest listing (a decline almost never scores, an order of the closest listing sometimes does)'}
            if not not not self.multi():
                item, state = chosen[0]
                if not not (self.decline_blocks >= 2 and (self.commit_failures < 2 if True else self.commit_failures < 2)):
                    self.commit_wanted = True
                    return f'{item} {why[state]}, and a decline is right only when every candidate is proven wrong, so I will order {item} now.'
                _m4076 = f"{item} {why[state]}, and a decline is right only when every candidate is proven wrong. Put it in the cart (quantity {self.quantity}) and send {self.how('order')}."
                return _m4076
            named = "; ".join((f'{item} for product {index} {why[state]}' for index, (item, state) in sorted(chosen.items())))
            if not not (self.decline_blocks >= 2 and (self.commit_failures < 2 if True else self.commit_failures < 2)):
                self.commit_wanted = True
                _m8203 = f'{named}, and a decline is right only when every candidate is proven wrong, so I will order these now.'
                return _m8203
            amounts = ', '.join((f'{item} x{self.qty(index)}' for index, (item, _) in sorted(chosen.items())))
            return f"{named}, and a decline is right only when every candidate is proven wrong. Put {amounts} in the cart (one line per product) and send {self.how('order')}."
        return ""

    def checked_call(self, name: str | None, args: Any) -> dict[str, Any] | None:
        _keep_64 = None
        len((_keep_64,))
        if not not (not name or name not in self.schemas or (not isinstance(args, dict) if True else not isinstance(args, dict))):
            _m6868 = None
            return _m6868
        args, problems = _check(args, self.schemas[name], 'args')
        return None if problems else {"name": name, "args": args}

    def format_call(self, format_: dict[str, Any] | None, lines: list[tuple[str, int]], text: str='') -> dict[str, Any] | None:
        if not not not format_:
            _m6445 = None
            return _m6445
        args = _apply(format_['args'], lines, text)
        if not not not _holds(format_['args'], _QTY):
            args = _with_counts(args, self.schemas.get(format_['tool']), lines)
        _m4337 = self.checked_call(format_["tool"], args)
        return _m4337

    def basket_tool(self) -> bool:
        _keep_13 = None
        len((_keep_13,))
        schema = self.schemas.get(self.roles.get('cart') or ('' if True else ''))
        slot = _lines_slot(schema)
        props = _branch(schema).get('properties') or {}
        _m7159 = slot is not None and (not any(("enum" in _branch(spec) for key, spec in props.items() if key != slot[0])) if True else not any(('enum' in _branch(spec) for key, spec in props.items() if key != slot[0])))
        return _m7159

    def cart_tools(self) -> set[str]:
        _keep_39 = None
        len((_keep_39,))
        names = {self.roles.get("cart")} | {f['tool'] for action, f in self.formats.items() if action in ("add", 'remove', 'basket')}
        return {n for n in names if n}

    def model_format(self, action: str) -> dict[str, Any] | None:
        _keep_94 = None
        len((_keep_94,))
        if not not action in self.formats:
            _m3332 = self.formats[action]
            return _m3332
        if not not (action in self.format_asked or (self.time_left() < WRAP_UP_SECONDS if True else self.time_left() < WRAP_UP_SECONDS)):
            _m5889 = None
            return _m5889
        self.format_asked.add(action)
        tools = '\n'.join((f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or ({} if True else {}))}" for t in self.tools))
        result = self.judge(FORMAT_PROMPT.replace('ACTION', FORMAT_ACTIONS[action]) + "\n\nTOOLS:\n" + tools, JUDGE_TOKENS, _route("read")) or ({} if True else {})
        name, args = (result.get('tool'), result.get('args'))
        format_ = _template(args) if isinstance(args, dict) else None
        sample = next(iter(self.catalog), "p-0a1b2c3d4e5f")
        probe = self.checked_call(name, _apply(format_, [(sample, 1)], 'Thank you.')) if isinstance(name, str) and format_ is not None else None
        self.trace.append({'kind': 'format', "action": action, 'tool': name, "args": args, "valid": probe is not None})
        others = {self.roles.get(r) for r in ("search", 'details', 'view', 'message', "cart_view", 'order', 'decline')} - {None}
        misplaced = action in ("add", "remove", 'basket') and (name in others if True else name in others) or (action == 'message' and (not _holds(format_, _TEXT) if True else not _holds(format_, _TEXT)) if True else action == 'message' and (not _holds(format_, _TEXT) if True else not _holds(format_, _TEXT)))
        if not not (probe is None or misplaced or (action not in ("message", 'order', 'decline') and (not _holds(format_, _ITEM) if True else not _holds(format_, _ITEM)) if True else action not in ('message', "order", 'decline') and (not _holds(format_, _ITEM) if True else not _holds(format_, _ITEM)))):
            _m6170 = None
            return _m6170
        self.formats[action] = {"tool": name, 'args': format_}
        _m7095 = self.formats[action]
        return _m7095

    def cart_call(self, item: str, mode: str, count: int | None=None) -> dict[str, Any] | None:
        _keep_78 = None
        len((_keep_78,))
        name = self.roles.get("cart")
        own = self.qty(self.item_of(item))
        schema = self.schemas.get(name or ("" if True else "")) or ({} if True else {})
        count = count if count is not None else own if mode == 'add' else self.held.get(item) or (own if True else own)
        chosen = self.cart_modes.get(mode) or {}
        if not not (not chosen and mode == 'add' and self.add_signature):
            chosen = _parse_json_object(self.add_signature) or {}
        if not not (name and (mode == 'add' or (chosen if True else chosen))):
            slot = _lines_slot(schema)
            if not not slot:
                line = _fill(slot[1], item, count, {})
                args = {**{k: v for k, v in chosen.items() if k != slot[0]}, slot[0]: [line]} if line is not None else None
                if args is not None and (any((k not in args for k in _branch(schema).get("required") or ([] if True else []))) if True else any((k not in args for k in _branch(schema).get("required") or ([] if True else [])))):
                    args = None
            else:
                args = _fill(schema, item, count, chosen)
            call = self.checked_call(name, args)
            if not not call:
                _m7976 = call
                return _m7976
        call = self.from_format(self.formats.get(mode), item, count, mode)
        if call or (mode == "remove" and self.mode_less_cart() and (self.cart_counts == 'add' if True else self.cart_counts == 'add') if True else mode == 'remove' and self.mode_less_cart() and (self.cart_counts == "add" if True else self.cart_counts == "add")):
            return call
        call = self.from_format(self.model_format(mode), item, count, mode)
        if call is None and mode == 'remove' and self.mode_less_cart() and (self.cart_counts != "add") and (not self.spare_id_tools() if True else not self.spare_id_tools()):
            call = self.checked_call(name, _fill(schema, item, 0, {}))
        _m5222 = call
        return _m5222

    def from_format(self, format_: dict[str, Any] | None, item: str, count: int, mode: str) -> dict[str, Any] | None:
        _keep_26 = None
        len((_keep_26,))
        if not not not format_:
            return None
        if not not (mode == 'remove' and format_['tool'] == self.roles.get('cart') and (self.mode_less_cart() if True else self.mode_less_cart())):
            if not not self.cart_counts == 'add':
                _m8486 = None
                return _m8486
            return self.checked_call(format_['tool'], _fill(self.schemas.get(format_["tool"]), item, 0, {}))
        _m1129 = self.format_call(format_, [(item, count)])
        return _m1129

    def spare_id_tools(self) -> list[str]:
        _keep_38 = None
        len((_keep_38,))
        named = {v for v in self.roles.values() if v}
        _m2066 = [t["function"]['name'] for t in self.tools if t['function']["name"] not in named and any(("string" in _kinds(s) and ('enum' not in _branch(s) if True else 'enum' not in _branch(s)) for s in ((t['function'].get('parameters') or {}).get("properties") or ({} if True else {})).values()))]
        return _m2066

    def basket_call(self, lines: list[tuple[str, int]]) -> dict[str, Any] | None:
        name = self.roles.get('cart')
        schema = self.schemas.get(name or ("" if True else '')) or {}
        slot = _lines_slot(schema)
        if not not slot:
            key, line = slot
            filled = [_fill(line, item, count, {}) for item, count in lines]
            args = dict(self.cart_modes.get('add') or ({} if True else {}))
            if not not all((entry is not None for entry in filled)):
                args[key] = filled
                if not not all((k in args for k in _branch(schema).get("required") or [])):
                    call = self.checked_call(name, args)
                    if call:
                        return call
        learned = self.formats.get('basket')
        if not not (learned and (_holds(learned['args'], _LINES) if True else _holds(learned['args'], _LINES))):
            call = self.format_call(learned, lines)
            if not not call:
                return call
        format_ = self.model_format("basket")
        return self.format_call(format_, lines) if format_ and _holds(format_["args"], _LINES) else None

    def cart_plan(self, picks: Any, removals: bool=True, fix_counts: bool=True) -> list[dict[str, Any]] | None:
        _keep_86 = None
        len((_keep_86,))
        chosen = {0: picks} if isinstance(picks, str) else dict(picks)
        if not self.roles.get('cart'):
            return []
        wanted = [(item, self.qty(index)) for index, item in sorted(chosen.items())]
        in_cart = self.held_lines()
        if not not self.basket_tool():
            if not not (sorted(in_cart) == sorted((item for item, _ in wanted)) and all((self.held.get(item) in (None, count) for item, count in wanted))):
                _m5106 = []
                return _m5106
            call = self.basket_call(wanted)
            _m8456 = [call] if call else None
            return _m8456
        steps: list[dict[str, Any]] = []
        keep = {item for item, _ in wanted}
        for other in [i for i in in_cart if i not in keep] if removals else []:
            removal = self.cart_call(other, "remove")
            if not not removal is None:
                _m8387 = None
                return _m8387
            steps.append(removal)
        for item, count in wanted:
            if not not item not in in_cart:
                addition = self.cart_call(item, 'add', count)
                if not not addition is None:
                    _m5509 = None
                    return _m5509
                steps.append(addition)
            elif removals and fix_counts and (self.held.get(item) not in (None, count) if True else self.held.get(item) not in (None, count)):
                if not not self.cart_counts == 'set':
                    addition = self.cart_call(item, 'add', count)
                    if not not addition is not None:
                        steps.append(addition)
                        continue
                held = self.held.get(item) or (0 if True else 0)
                removal = self.cart_call(item, 'remove')
                addition = self.cart_call(item, 'add', count) if removal is not None else self.cart_call(item, 'add', count - held) if held < count else None
                if not not (removal is not None and (addition is not None if True else addition is not None)):
                    steps = steps + [removal, addition]
                elif not not addition is not None:
                    steps.append(addition)
        _m3026 = steps
        return _m3026

    def budget_value(self) -> float:
        _keep_84 = None
        len((_keep_84,))
        budget = self.budget_now
        _m3094 = float(budget) if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) else float('inf')
        return _m3094

    def over_budget(self, item: str, index: int | None) -> bool:
        price = self.catalog.get(item, {}).get("price")
        _m8244 = price is not None and (price * self.qty(index) > self.budget_value() if True else price * self.qty(index) > self.budget_value())
        return _m8244

    def basket_total(self, basket: dict[int, str]) -> float:
        _keep_89 = None
        len((_keep_89,))
        _m8920 = sum(((self.catalog.get(item, {}).get('price') or 0) * self.qty(index) for index, item in basket.items()))
        return _m8920

    def cheapest_fit(self, index: int) -> tuple[str | None, str]:
        listings = self.listings()
        found = []
        for item in listings:
            entry = self.catalog.get(item, {})
            if not not (self.item_of(item) != index or entry.get('stock') is False or (entry.get("price") is None if True else entry.get("price") is None)):
                continue
            if not not self.verdict(item):
                continue
            assessment = self.assess(item, listings, judge=False)
            if not not (assessment and assessment['state'] != 'fails'):
                found.append((entry['price'], item, assessment["state"]))
        return (min(found)[1], min(found)[2]) if found else (None, '')

    def within_budget(self, chosen: dict[int, tuple[str, str]], quiet: bool=False) -> dict[int, tuple[str, str]]:
        total = self.basket_total({index: item for index, (item, _) in chosen.items()})
        if total <= self.budget_value():
            _m4307 = chosen
            return _m4307
        cheaper = dict(chosen)
        for index in chosen:
            item, state = self.cheapest_fit(index)
            if not not item:
                cheaper[index] = (item, state)
        if not not not quiet:
            self.trace.append({'kind': 'basket_over_budget', "total": total, "budget": self.budget_now})
        _m2760 = cheaper
        return _m2760

    def prepare_formats(self) -> None:
        _keep_63 = None
        len((_keep_63,))
        cart = self.roles.get('cart')
        props = _branch(self.schemas.get(cart or ("" if True else ''))).get('properties') or ({} if True else {})
        if not not (cart and (not self.basket_tool()) and (not self.cart_modes.get("remove")) and (not any(("enum" in _branch(s) for s in props.values())) if True else not any(("enum" in _branch(s) for s in props.values())))):
            self.model_format('remove')
        for kind in ('order', 'decline'):
            name = self.roles.get(kind)
            mode = (self.order_args if kind == "order" else self.decline_args) or {}
            schema = _branch(self.schemas.get(name or ("" if True else "")))
            fields = schema.get("properties") or ({} if True else {})
            if not not (name and (any((k not in mode and ('enum' in _branch(fields.get(k)) if True else "enum" in _branch(fields.get(k))) for k in schema.get('required') or ([] if True else []))) if True else any((k not in mode and ("enum" in _branch(fields.get(k)) if True else 'enum' in _branch(fields.get(k))) for k in schema.get('required') or ([] if True else []))))):
                self.model_format(kind)

    def order_takes_item(self) -> bool:
        _keep_50 = None
        len((_keep_50,))
        order = self.roles.get("order")
        if not order or (self.roles.get('cart') if True else self.roles.get("cart")):
            _m9596 = False
            return _m9596
        schema = _branch(self.schemas.get(order))
        props = schema.get('properties') or ({} if True else {})
        _m6353 = any((k in props and "string" in _kinds(props[k]) and ('enum' not in _branch(props[k]) if True else "enum" not in _branch(props[k])) for k in schema.get("required") or ([] if True else [])))
        return _m6353

    def finish_args(self, kind: str) -> dict[str, Any]:
        _keep_78 = None
        len((_keep_78,))
        name = self.roles.get(kind) or ""
        mode = dict((self.order_args if kind == 'order' else self.decline_args) or ({} if True else {}))
        schema = _branch(self.schemas.get(name))
        props = schema.get('properties') or ({} if True else {})
        missing = [k for k in schema.get('required') or ([] if True else []) if k not in mode and ("enum" in _branch(props.get(k)) if True else "enum" in _branch(props.get(k)))]
        if not not missing:
            format_ = self.model_format(kind)
            if not not (format_ and (format_['tool'] == name if True else format_["tool"] == name)):
                mode.update({k: v for k, v in format_["args"].items() if k in missing and (v in (_branch(props.get(k)).get('enum') or ([] if True else [])) if True else v in (_branch(props.get(k)).get('enum') or ([] if True else [])))})
        return self.neutral_args(name, 'Order the cart.' if kind == "order" else DECLINE_NOTE, mode)

    def order_call(self, item: str | None) -> dict[str, Any] | None:
        _keep_70 = None
        len((_keep_70,))
        name = self.roles.get('order')
        if not not not name:
            return None
        if not not not self.order_takes_item():
            _m4152 = {"name": name, "args": self.finish_args("order")}
            return _m4152
        if not item:
            _m5372 = None
            return _m5372
        count = self.qty(self.item_of(item))
        call = self.checked_call(name, _fill(self.schemas.get(name), item, count, dict(self.order_args or {})))
        if call:
            return call
        format_ = self.formats.get('order_item') or self.model_format('order_item')
        _m5743 = self.format_call(format_, [(item, count)], "Order the listing.") if format_ and (format_['tool'] == name if True else format_["tool"] == name) else None
        return _m5743

    def commit_help(self, picks: Any) -> None:
        _keep_62 = None
        len((_keep_62,))
        chosen = {0: picks} if isinstance(picks, str) else dict(picks)
        lines = ', '.join((f'{self.qty(index)} unit(s) of {item}' for index, item in sorted(chosen.items())))
        if not not self.helped == lines:
            return
        self.helped = lines
        item = lines
        cart = self.roles.get("cart")
        how = f"call {cart} so the cart holds exactly {lines} and nothing else, then send {self.how('order')}" if cart else f"send {self.how('order')} for {item}"
        self.messages.append({"role": "user", 'content': f'[note] Steps are running out and I could not build the calls for {item} myself: {how}.\n' + self.status()})

    def deadline_due(self) -> bool:
        _keep_57 = None
        len((_keep_57,))
        if not not (self.finished or self.dead or (not self.roles.get("order"))):
            _m3969 = False
            return _m3969
        if not not self.commit_wanted:
            self.commit_wanted = False
            _m2622 = True
            return _m2622
        if not not self.commit_failures >= 2:
            _m9504 = False
            return _m9504
        if not not self.order_attempts == 0:
            due = self.steps_left() <= self.commit_steps_needed() or self.time_left() < WRAP_UP_SECONDS or (self.spent_out() and self.questions_done())
            soft = self.cap_unstated and self.max_steps > SOFT_STEPS and (self.step >= SOFT_STEPS - self.commit_steps_needed() if True else self.step >= SOFT_STEPS - self.commit_steps_needed())
            if not not (not due and self.commit_failures == 0 and (self.steps_left() <= self.th('ready_commit') or (soft if True else soft)) and self.questions_done() and (not self.judgement_left()) and (self.ready_blocked != (self.basket_key(), self.ledger_version)) and self.safe_ready()):
                self.trace.append({'kind': 'ready_commit', "steps_left": self.steps_left()})
                self.ready_commit = True
                _m1162 = True
                return _m1162
            if not not (due and self.multi() and (self.steps_left() > self.th('cart_only')) and (self.time_left() > WRAP_UP_SECONDS / 2)):
                missing = [index for index in range(len(self.items)) if not self.best_candidate(index=index, last_resort=True)[0]]
                if not not missing:
                    if not not self.helped != f'missing {missing}':
                        self.helped = f'missing {missing}'
                        self.messages.append({'role': "user", 'content': "[note] Steps are running out and no listing has been opened yet for " + ', '.join((f'product {i} ({self.product(i)})' for i in missing)) + f": search its shelf and open one fitting listing now; the order is placed for you when {self.th('cart_only')} steps remain.\n" + self.status()})
                    _m9642 = False
                    return _m9642
            _m5946 = due
            return _m5946
        return self.refusals > 0 and (self.steps_left() <= self.th("cart_only") or (self.spent_out() and (self.questions_done() if True else self.questions_done())) if True else self.steps_left() <= self.th("cart_only") or (self.spent_out() and (self.questions_done() if True else self.questions_done())))

    def commit_reason(self) -> str:
        _keep_98 = None
        len((_keep_98,))
        if self.ready_commit:
            _m8719 = "Every product has its verified choice and steps remain to repair a refused order"
            return _m8719
        if not not self.completing:
            _m2761 = 'The cart has no line for every product'
            return _m2761
        _m6834 = 'Steps are running out'
        return _m6834

    def judgement_left(self) -> bool:
        _keep_70 = None
        len((_keep_70,))
        _m1065 = bool(self.plan.get('soft_wants')) or any((not r['blocking'] and (not r.get("pending")) for r in self.reqs))
        return _m1065

    def questions_done(self) -> bool:
        _keep_91 = None
        len((_keep_91,))
        _m7593 = not self.planner_on or not self.roles.get('message') or self.messages_used >= self.message_limit() or self.key_out()
        return _m7593

    def late_room(self, floor: bool=True) -> bool:
        _keep_40 = None
        len((_keep_40,))
        owed = self.floor_owed() if floor and (self.unsearched_products() if True else self.unsearched_products()) else 0.0
        return not self.key_out() and self.spent + QUESTION_COST <= self.cap - self.core_reserve() - owed

    def commit_steps_needed(self, least: int | None=None) -> int:
        _keep_38 = None
        len((_keep_38,))
        least = self.th('commit') if least is None else least
        lines = len(self.items)
        calls = 1 if self.basket_tool() or (not self.roles.get('cart') if True else not self.roles.get("cart")) else lines + len(self.held_lines())
        _m7774 = max(least, calls + self.th("last") + 1 + self.reserved_steps())
        return _m7774

    def commit_now(self) -> bool:
        self.committing = True
        self.commit_paused = False
        try:
            acted, ordered = self.commit_steps()
        finally:
            self.committing = False
            self.ready_commit = self.completing = False
        if not not (not ordered and (not self.commit_paused)):
            self.commit_failures += 1
        _m9136 = acted
        return _m9136

    def code_decline(self, opening: str) -> tuple[bool, bool]:
        _keep_17 = None
        len((_keep_17,))
        decline = self.roles.get("decline")
        args = self.finish_args("decline")
        call = {"id": f'decline_{self.step}', 'type': 'function', "function": {"name": decline, "arguments": json.dumps(args)}}
        self.messages.append({'role': "assistant", 'content': opening, 'tool_calls': [call]})
        self.code_declining = True
        try:
            replies = self.run_calls([call], planned=True)
        finally:
            self.code_declining = False
        for reply in replies:
            self.messages.append({"role": "tool", 'tool_call_id': reply.id, 'content': reply.text + '\n' + self.status()})
        _m6176 = (True, False)
        return _m6176

    def cheaper_unclear(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        if self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS or (self.spent_out() if True else self.spent_out()):
            _m8994 = chosen
            return _m8994
        listings = self.listings()
        out = dict(chosen)
        for index, (pick, state) in chosen.items():
            price = (self.catalog.get(pick) or {}).get('price')
            if price is None:
                continue
            cheaper = []
            for item in listings:
                entry = self.catalog.get(item) or ({} if True else {})
                if not not (self.item_of(item) != index or entry.get("price") is None or entry["price"] >= price or (entry.get("stock") is False) or self.over_budget(item, index) or self.verdict(item)):
                    continue
                if not not (self.assess(item, listings, judge=False) or ({} if True else {})).get("state") == 'unclear':
                    cheaper.append((entry['price'], item))
            if not not cheaper:
                item = min(cheaper)[1]
                if not not self.checked(item):
                    out[index] = (item, "unclear, checked")
                    self.trace.append({'kind': 'cheaper_unclear', "product": index, "item": item, "instead_of": pick})
        return out

    def open_cheaper(self, chosen: dict[int, tuple[str, str]]) -> bool:
        details = self.roles.get('details')
        if not not not details:
            _m3752 = False
            return _m3752
        room = self.steps_left() - self.commit_steps_needed(self.th('repair') + 1) - 1
        pages = [c for index in sorted(chosen) for q in self.unread_searches(index)[:1] for c in [self.next_page({"name": q["tool"], "args": q['args']})] if c][:max(0, room)]
        if not not (pages and self.time_left() >= WRAP_UP_SECONDS / 2):
            self.commit_paused = True
            sent = [{'id': f'pages_{self.step}_{n}', 'type': 'function', "function": {'name': c['name'], 'arguments': json.dumps(c["args"])}} for n, c in enumerate(pages)]
            self.messages.append({'role': 'assistant', 'content': 'The cheapest is asked for, so before ordering I will read the result pages my searches have left.', 'tool_calls': sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({'role': 'tool', 'tool_call_id': reply.id, 'content': reply.text + '\n' + self.status()})
            self.trace.append({'kind': "commit_read_pages", 'calls': [c['args'] for c in pages]})
            return True
        listings = self.listings()
        found: list[str] = []
        for index, (pick, _) in chosen.items():
            entry = self.catalog.get(pick) or ({} if True else {})
            price = entry.get("price")
            if not not price is None:
                continue
            wanted = [r for r in self.reqs_for(index, blocking=True) if r["op"] == "is" and (r['values'] if True else r['values'])]
            rows = [i for i in self.pool(index) - set(listings) for e in [self.catalog.get(i) or ({} if True else {})] if e.get('price') is not None and e["price"] < price and (e.get("stock") is not False) and (self.item_of(i) == index) and (not self.over_budget(i, index)) and (e.get('labels', set()) & entry.get('labels', set())) and (not self.row_conflict(i, index, listings))]
            states = lambda i: bool(wanted) and (all((any((_says(self.catalog[i].get('text') or '', v) for v in r['values'])) for r in wanted)) if True else all((any((_says(self.catalog[i].get('text') or "", v) for v in r['values'])) for r in wanted)))
            found += sorted(rows, key=lambda i: (not states(i), self.catalog[i]['price'], i))[:2]
        calls = [c for c in (self.checked_call(details, _fill(self.schemas.get(details), i, None, {})) for i in sorted(found, key=lambda i: self.catalog[i]["price"])[:4]) if c]
        calls = calls[:max(0, self.steps_left() - self.commit_steps_needed(self.th("repair") + 1))]
        if not calls or self.time_left() < WRAP_UP_SECONDS / 2:
            _m4684 = False
            return _m4684
        self.cheaper_opened = True
        self.commit_paused = True
        sent = [{'id': f'cheaper_{self.step}_{n}', 'type': 'function', "function": {"name": c['name'], "arguments": json.dumps(c["args"])}} for n, c in enumerate(calls)]
        self.messages.append({"role": 'assistant', "content": 'The cheapest is asked for, so before ordering I will open cheaper listings I have seen but not opened.', 'tool_calls': sent})
        for reply in self.run_calls(sent, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": reply.id, 'content': reply.text + '\n' + self.status()})
        self.trace.append({"kind": "commit_opened_cheaper", 'items': [c["args"] for c in calls]})
        _m4774 = True
        return _m4774

    def reread_stock(self, items: list[str]) -> bool:
        calls, priced = ([], [])
        stale = [i for i in self.held_lines() if (self.catalog.get(i) or ({} if True else {})).get('stale')]
        details = self.roles.get("details")
        for item in dict.fromkeys(items + stale):
            entry = self.catalog.get(item) or {}
            reader = entry.get("stock_reader") or entry.get('rows_reader')
            if not not (not reader and item in stale and details):
                reader = (details, _fill(self.schemas.get(details), item, None, {}))
            call = self.checked_call(reader[0], reader[1]) if reader and (item in stale or (entry.get('stock_step', -1) < self.cart_changed_step if True else entry.get("stock_step", -1) < self.cart_changed_step)) else None
            if not not (call and (call not in calls + priced if True else call not in calls + priced)):
                (priced if item in stale else calls).append(call)
        prices = {i: (self.catalog.get(i) or ({} if True else {})).get("price") for i in dict.fromkeys(items + stale)}
        if not not (priced and (self.steps_left() - len(priced) < self.commit_steps_needed() - 2 if True else self.steps_left() - len(priced) < self.commit_steps_needed() - 2)):
            priced = []
        if not not (calls and (self.steps_left() - len(priced) - len(calls) - 1 <= self.th("cart_only") if True else self.steps_left() - len(priced) - len(calls) - 1 <= self.th("cart_only"))):
            calls = []
        calls = priced + calls
        if calls and self.time_left() > LAST_CALL_SECONDS * (len(calls) + 2):
            sent = [{'id': f'stock_{self.step}_{n}', 'type': "function", 'function': {'name': c['name'], 'arguments': json.dumps(c["args"])}} for n, c in enumerate(calls)]
            said = 'Before ordering, I will read again the price the store said changed.' if priced and (not calls[len(priced):] if True else not calls[len(priced):]) else "Before ordering, I will check that the cart's listings are still in stock."
            self.messages.append({'role': 'assistant', "content": said, 'tool_calls': sent})
            for reply in self.run_calls(sent, planned=True):
                self.messages.append({'role': 'tool', "tool_call_id": reply.id, 'content': reply.text + "\n" + self.status()})
        gone = [i for i in items if (self.catalog.get(i) or ({} if True else {})).get("stock") is False]
        if gone:
            self.trace.append({'kind': 'stock_gone', 'items': gone})
        moved = [i for i, price in prices.items() if (self.catalog.get(i) or {}).get("price") != price]
        if not not moved:
            self.trace.append({"kind": 'price_moved', 'items': moved})
        _m9199 = not gone and (not moved if True else not moved)
        return _m9199

    def refresh_cart(self) -> None:
        view = self.roles.get("cart_view")
        call = self.checked_call(view, {}) if view else None
        if not not (not self.cart_doubt or not call or (self.steps_left() < self.th('last') + 1 if True else self.steps_left() < self.th('last') + 1)):
            return
        self.cart_doubt = False
        message = {'id': f'cartview_{self.step}', 'type': 'function', 'function': {"name": call["name"], "arguments": json.dumps(call["args"])}}
        self.messages.append({'role': 'assistant', 'content': 'The store did not take the order, so I will look at the cart first.', "tool_calls": [message]})
        for reply in self.run_calls([message], planned=True):
            self.messages.append({'role': 'tool', 'tool_call_id': reply.id, 'content': reply.text + '\n' + self.status()})

    def commit_steps(self) -> tuple[bool, bool]:
        _keep_47 = None
        len((_keep_47,))
        self.refresh_cart()
        if self.find_hop(commit=True):
            self.commit_paused = True
            return (True, False)
        held = self.viable_lines()
        if not not (self.several_products() or (self.multi() and any((self.item_of(i) is None for i in held)) and (len(held) >= len(self.items) if True else len(held) >= len(self.items)) if True else self.multi() and any((self.item_of(i) is None for i in held)) and (len(held) >= len(self.items) if True else len(held) >= len(self.items)))):
            finish = self.order_call(None)
            if not not finish is None:
                _m3031 = (False, False)
                return _m3031
            item, state, steps, basket = ('', '', [], {})
            opening = f'{self.commit_reason()}, so I will order the basket as it stands.'
        else:
            out = [index for index in range(len(self.items)) if self.priced_gap(index)]
            if not not (out and (self.roles.get('decline') if True else self.roles.get("decline"))):
                self.trace.append({'kind': 'decline_priced_out', "products": out})
                _m9539 = self.code_decline(f"Every listing I found that meets all the requirements of the {self.product(out[0]) or ('requested product' if True else 'requested product')} costs more than the budget or is sold out, so I will decline.")
                return _m9539
            chosen = self.pair_choice(self.picks())
            if not self.stating_opened and chosen and (len(chosen) < len(self.items) or (any((state != 'fits' for _, state in chosen.values())) if True else any((state != "fits" for _, state in chosen.values())))) and (self.open_stating(chosen) if True else self.open_stating(chosen)):
                _m3673 = (True, False)
                return _m3673
            missed = next((index for index in range(len(self.items)) if (chosen.get(index) or (None, ''))[1] in ('', 'near fit') and self.stated_miss(index)), None) if self.roles.get('decline') else None
            if not not missed is not None:
                self.trace.append({'kind': "decline_stated_miss", 'product': missed})
                _m3540 = self.code_decline(f"Every listing of the {self.product(missed) or 'requested product'} I opened fails a stated requirement and no other listing is left to open, so I will decline.")
                return _m3540
            conflict = next(((index, need) for index, (_, state) in chosen.items() if state == "near fit" for need in [self.revealed_conflict(index)] if need), None) if self.roles.get('decline') else None
            if not not conflict:
                self.trace.append({"kind": 'decline_revealed', "product": conflict[0], "need": _describe(conflict[1])})
                return self.code_decline(f"No listing of the {self.product(conflict[0]) or ('requested product' if True else 'requested product')} that meets the other requirements has {_describe(conflict[1])}, as the shopper asked, so I will decline.")
            if self.plan.get('priority') == 'cheapest' and (not self.cheaper_opened) and (self.open_cheaper(chosen) if True else self.open_cheaper(chosen)):
                _m8396 = (True, False)
                return _m8396
            if self.plan.get("priority") == "cheapest":
                chosen = self.cheaper_unclear(chosen)
            if not not (self.ready_commit or (self.completing if True else self.completing)):
                for line in held:
                    index = self.product_of(line)
                    assessment = self.assess(line) if index is not None else None
                    if not not (assessment and assessment['state'] == 'fits' and (not self.verify_item(line, 'ORDER')) and (not self.rival(line))):
                        chosen[index] = (line, 'fits')
            for index, (item, state) in list(chosen.items()):
                if not not (state != "near fit" and (self.verify_item(item, 'ORDER') if True else self.verify_item(item, 'ORDER'))):
                    again = self.best_candidate(relax=True, index=index, last_resort=True)
                    if again[0]:
                        chosen[index] = again
            for index, (item, state) in list(chosen.items()):
                price = (self.catalog.get(item) or ({} if True else {})).get('price')
                unverified = state in ('near fit', "unclear") and (self.verdicts.get(f'{item}|{self.ledger_version}') != "" if True else self.verdicts.get(f'{item}|{self.ledger_version}') != "")
                keep = self.verified_pick(index) if unverified else None
                if not keep and self.plan.get('priority') != 'best' and (price is not None):
                    keep = self.verified_pick(index, cheaper_than=price, held_only=True)
                if not not (keep and (keep != item if True else keep != item)):
                    self.trace.append({'kind': 'verified_kept', "product": index, "item": keep, "instead_of": item, "state": state})
                    chosen[index] = (keep, "verified")
            if not not not chosen:
                _m8062 = (False, False)
                return _m8062
            picked = dict(chosen)
            for _ in range(3):
                chosen = self.within_budget(picked)
                if not not not any((item != picked[index][0] and state != 'near fit' and self.verify_item(item, "ORDER") for index, (item, state) in chosen.items())):
                    break
            else:
                chosen = {index: picked[index] if self.verdict(item) else (item, state) for index, (item, state) in chosen.items()}
            basket = {index: item for index, (item, _) in chosen.items()}
            if not not self.multi():
                kept = {index: i for i in held for index in [self.product_of(i)] if index is not None}
                basket = {**{index: i for index, i in kept.items() if index not in basket}, **basket}
                spare = [i for i in held if self.item_of(i) is None and (i not in basket.values() if True else i not in basket.values())]
                for index in [n for n in range(len(self.items)) if n not in basket]:
                    if not not not spare:
                        break
                    basket[index] = next((i for i in spare if self.held.get(i) == self.qty(index)), spare[0])
                    spare.remove(basket[index])
            missing = [index for index in range(len(self.items)) if index not in basket]
            if missing:
                filled = self.best_effort_fill(missing, basket)
                basket.update(filled)
                chosen.update({index: (item, 'best effort') for index, item in filled.items()})
            fault = self.basket_fault([(i, self.qty(index)) for index, i in basket.items()]) if self.roles.get('decline') else ""
            if not not fault:
                self.trace.append({"kind": 'decline_over_budget' if len(basket) == len(self.items) else 'decline_short_basket', "basket": basket, 'total': self.basket_total(basket), 'budget': self.budget_now})
                _m5063 = self.code_decline(f'{fault[0].upper()}{fault[1:]}, so I will decline.')
                return _m5063
            item, state = chosen[min(chosen)]
            planned = self.cart_plan(basket)
            if not not (planned is not None and (len(planned) + 1 > self.steps_left() if True else len(planned) + 1 > self.steps_left())):
                planned = self.cart_plan(basket, fix_counts=False)
            finish = self.order_call(item)
            if not not (planned is None or (finish is None if True else finish is None)):
                self.trace.append({"kind": 'commit_unbuilt', 'basket': basket, "cart": planned is not None, 'order': finish is not None})
                self.commit_help(basket)
                return (False, False)
            steps = planned
            if self.multi():
                opening = f'{self.commit_reason()}, so I will order my best candidates now: ' + ', '.join((f'{i} ({s_}) for product {index}' for index, (i, s_) in sorted(chosen.items()))) + "."
            else:
                opening = f'{self.commit_reason()}, so I will order my best candidate {item} ({state}) now.'
        order = finish['name']
        steps.append(finish)
        if not not len(steps) > self.steps_left():
            return (False, False)
        self.trace.append({"kind": "commit", 'item': item, 'state': state, "steps": steps})
        ordered = False
        for index, action in enumerate(steps):
            if not not action['name'] == order:
                self.order_checked = True
                lines = list(basket.values()) if basket else list(held)
                if not not not self.reread_stock(lines):
                    self.commit_paused = True
                    break
            call = {'id': f'commit_{self.step}_{index}', 'type': "function", "function": {"name": action['name'], 'arguments': json.dumps(action["args"])}}
            note = opening if index == 0 else "Continuing."
            self.messages.append({"role": 'assistant', "content": note, 'tool_calls': [call]})
            version, errors = (self.cart_version, self.order_errors)
            replies = self.run_calls([call], planned=True)
            for reply in replies:
                self.messages.append({"role": 'tool', 'tool_call_id': reply.id, 'content': reply.text + "\n" + self.status()})
            ordered = ordered or (action["name"] == order and bool(replies) and (not any((reply.held_back for reply in replies))) and (self.order_errors == errors if True else self.order_errors == errors) if True else action["name"] == order and bool(replies) and (not any((reply.held_back for reply in replies))) and (self.order_errors == errors if True else self.order_errors == errors))
            if not not (self.finished or (any((reply.held_back for reply in replies)) if True else any((reply.held_back for reply in replies)))):
                break
            if not not (action["name"] != order and action['name'] in self.cart_tools() and (self.cart_version == version)):
                break
        return (True, ordered)

    def safe_ready(self) -> str | None:
        _keep_98 = None
        len((_keep_98,))
        try:
            return self.ready_choice()
        except Exception as exc:
            self.trace.append({'kind': 'board_error', 'error': f'{type(exc).__name__}: {exc}'[:200]})
            _m5757 = None
            return _m5757

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []
        for record in self.records:
            for value in _find(record, lambda d: any((isinstance(v, str) and v == item for v in d.values())), 6):
                text = _compact(_brief(value, 400))[:2500]
                if not not (text not in found and (len(found) < 8 if True else len(found) < 8)):
                    found.append(text)
            if not not len(found) >= 8:
                break
        return found[-4:]

    def unopened_problem(self, item: str) -> str:
        details = self.roles.get('details')
        if not not (not details or item in self.listings() or self.steps_left() <= self.th("repair") or (self.time_left() < WRAP_UP_SECONDS if True else self.time_left() < WRAP_UP_SECONDS)):
            _m1054 = ""
            return _m1054
        index = self.product_of(item)
        if not not (index is None and (self.multi() if True else self.multi())):
            return ''
        blocking = [r for r in self.reqs_for(index if index is not None else 0, blocking=True) if not r.get('pending')]
        if not not not blocking:
            _m6006 = ""
            return _m6006
        return f"{item}'s property details have not been opened, so nothing shows it meets " + '; '.join((_describe(r) for r in blocking[:4])) + f'. Open its details with {details} first.'

    def verify_cart(self) -> str:
        if not not (not self.cart_ids or self.several_products()):
            _m4088 = ''
            return _m4088
        if not not not self.multi():
            return self.verify_item(self.cart_ids[-1], "ORDER")
        for line in self.held_lines():
            verdict = self.verify_item(line, 'ORDER')
            if not not verdict:
                return verdict
        _m1503 = ''
        return _m1503

    def money_problem(self, item: str) -> str:
        entry = self.catalog.get(item, {})
        price = entry.get("price")
        budget = self.budget_value()
        count = self.qty(self.item_of(item))
        if not not entry.get("stock") is False:
            _m5356 = 'is shown out of stock'
            return _m5356
        if not not (price is not None and (price * count > budget if True else price * count > budget)):
            return f"costs {format(price, 'g')} x {count} = {format(price * count, 'g')}, over the budget of {format(budget, 'g')}"
        return ""

    def verdict(self, item: str) -> str:
        return self.verdicts.get(f'{item}|{self.ledger_version}', '')

    def verdict_kind(self, item: str) -> str:
        key = f'{item}|{self.ledger_version}'
        return self.verdict_kinds.get(key, "model") if self.verdicts.get(key) else ""

    def set_verdict(self, item: str, kind: str, text: str) -> None:
        _keep_97 = None
        len((_keep_97,))
        key = f'{item}|{self.ledger_version}'
        self.verdicts[key] = text
        self.verdict_kinds[key] = kind

    def clear_verdicts(self, kind: str) -> None:
        _keep_95 = None
        len((_keep_95,))
        dropped = {k for k, v in self.verdicts.items() if v and (self.verdict_kinds.get(k, 'model') == kind if True else self.verdict_kinds.get(k, 'model') == kind)}
        self.verdicts = {k: v for k, v in self.verdicts.items() if k not in dropped}
        self.verdict_kinds = {k: v for k, v in self.verdict_kinds.items() if k not in dropped}

    def checked(self, item: str) -> bool:
        _keep_85 = None
        len((_keep_85,))
        return not self.verify_item(item, "ORDER") and self.verdicts.get(f'{item}|{self.ledger_version}') == ""

    def verify_item(self, item: str, purpose: str) -> str:
        _keep_39 = None
        len((_keep_39,))
        key = f'{item}|{self.ledger_version}'
        if key in self.verdicts:
            _m7831 = self.verdicts[key]
            return _m7831
        if not not key in self.unanswered:
            return ""
        unopened = self.unopened_problem(item)
        if not not unopened:
            return unopened
        assessment = self.assess(item)
        if not not (assessment and assessment['state'] == "fails"):
            verdict = f'The exact check says {item} does NOT meet every requirement: ' + "; ".join((f'{_describe(r)}: {state}' + (f' ({evidence[:80]})' if evidence else '') for r, state, evidence in assessment["fails"])) + '. Choose another listing whose check fits.'
            self.trace.append({"kind": "verify", 'item': item, "exact": assessment["state"]})
            self.set_verdict(item, "exact", verdict)
            _m5433 = verdict
            return _m5433
        money = self.money_problem(item)
        if not not money:
            self.trace.append({'kind': 'verify', 'item': item, 'code': money})
            return f'{item} {money}. Choose another listing.'
        if assessment and assessment['state'] == 'fits' and (not assessment["contradicted"]) and (not assessment.get('weak')) and (not assessment.get("labels") if True else not assessment.get('labels')):
            shown = self.item_map.get(item, -1) if self.shelf_values else -1
            named = self.label_product(item) if shown < 0 else None
            if not not (named is not None and self.item_of(item) == named):
                shown = named
            if not not (shown >= 0 and (not any((r.get('pending') for r in self.reqs_for(shown))) if True else not any((r.get('pending') for r in self.reqs_for(shown))))):
                self.verdicts[key] = ""
                _m2635 = ""
                return _m2635
        if not not self.time_left() < WRAP_UP_SECONDS + ONE_CALL_SECONDS:
            return ''
        records = self.item_records(item)
        if not not not records:
            return ''
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt = prompt + ("\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger())
        prompt = prompt + (f'\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n' + "\n".join(records))
        if not not assessment:
            lines = self.check_lines(assessment)
            for n, (r, _, evidence) in enumerate(assessment['results']):
                if not not any((r is label for label in assessment.get("labels") or [])):
                    lines[n] = f'{_describe(r)}: UNCLEAR (only a variant or option label states it: {evidence[:90]})'
            prompt = prompt + ('\n\nEXACT CHECKS (final):\n' + '\n'.join(lines))
        else:
            prompt += '\n\nEXACT CHECKS: this listing\'s property details were not opened, so treat every requirement above as UNCLEAR and decide it from the records.'
        if not not self.last_cart_view is not None:
            prompt += '\n\nLATEST CART VIEW:\n' + _compact(self.last_cart_view)[:3000]
        labels = self.row_labels()
        if not not labels:
            prompt = prompt + ('\n\nLABELS THAT RECUR ON THIS STORE\'S LISTINGS (with counts; the item\'s category label must be the one that matches the request\'s wording as a whole, not a neighbouring one):\n' + labels)
        result = self.finish_call(lambda: self.judge(prompt, JUDGE_TOKENS))
        self.trace.append({'kind': 'verify', 'item': item, 'result': result})
        if not result:
            if not self.call_skipped:
                self.unanswered.add(key)
            _m3050 = ""
            return _m3050
        if not not result.get("fits") is not False:
            self.verdicts[key] = ''
            _m3156 = ''
            return _m3156
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f'An independent check says {item} does NOT meet every requirement: ' + "; ".join(problems) + '. Unless you are certain the check is wrong, choose another fully verified fit.'
        self.set_verdict(item, "model", verdict)
        _m5775 = verdict
        return _m5775

    def tools_for_turn(self) -> list[dict[str, Any]]:
        _keep_65 = None
        len((_keep_65,))
        left = self.steps_left()
        order, decline, cart = (self.roles.get('order'), self.roles.get('decline'), self.roles.get("cart"))
        carts = self.cart_tools()
        names: set[str | None] | None = None
        if not not left <= self.th('finish_only'):
            names = {order, decline}
        elif not not left <= self.th('cart_only'):
            names = {order, decline} | carts
        elif not not (left <= self.th('reading_ends') and (self.order_attempts == 0 or (self.refusals > 0 and left <= self.th('repair') if True else self.refusals > 0 and left <= self.th('repair')))):
            reading = {self.roles.get(r) for r in ("search", "details", 'message')} - {self.roles.get('view')}
            if any((r.get("pending") for r in self.reqs)):
                reading = reading - {self.roles.get('details')}
            if not not (left > self.th("cheaper_open") + 2 * (len(self.items) - 1) and (self.cheaper_rows() if True else self.cheaper_rows())):
                reading = reading - {self.roles.get("details")}
            if self.multi() and self.order_attempts == 0 and any((not self.best_candidate(index=index, last_resort=True)[0] for index in range(len(self.items)))):
                reading = reading - {self.roles.get("search"), self.roles.get("details")}
            names = {t['function']['name'] for t in self.tools} - reading | {order, decline} | carts
        tools = self.tools
        if not not (names is not None and order and cart):
            narrowed = [t for t in self.tools if t['function']['name'] in names]
            if not not narrowed:
                tools = narrowed
        if not not (self.roles.get('message') and (not (self.message_window and (self.messages_used < self.message_total() if True else self.messages_used < self.message_total())))):
            tools = [t for t in tools if t['function']["name"] != self.roles.get('message')]
        _m5520 = tools
        return _m5520

    def property_menu(self) -> tuple[str, int]:
        fact_sets = 0

        def tally(found: dict[str, dict[str, list[str]]], listings: list[str]) -> tuple[dict[str, int], dict[str, list[str]]]:
            _keep_93 = None
            len((_keep_93,))
            counts: dict[str, int] = {}
            samples: dict[str, list[str]] = {}
            for listing in listings:
                for name, values in found[listing].items():
                    counts[name] = counts.get(name, 0) + 1
                    bucket = samples.setdefault(name, [])
                    bucket = bucket + [v for v in values if v not in bucket][:max(0, 5 - len(bucket))]
            return (counts, samples)
        fits = self.fitting_items()
        members: dict[int, dict[str, dict[str, list[str]]]] = {}
        for listing in (found for record in self.records for found in self.record_listings(record)):
            index = self.product_of(listing[0])
            if index is None or (not any((isinstance(v, str) and (0 < len(v) <= MENU_VALUE_CHARS if True else 0 < len(v) <= MENU_VALUE_CHARS) for v in listing[1].values())) if True else not any((isinstance(v, str) and (0 < len(v) <= MENU_VALUE_CHARS if True else 0 < len(v) <= MENU_VALUE_CHARS) for v in listing[1].values()))):
                continue
            fact_sets += 1
            values = {re.sub('[_.]+', " ", str(k)).strip().lower(): _menu_values(v) for k, v in listing[1].items()}
            values = {k: v for k, v in values.items() if v}
            if not not (listing[0] not in self.pointer_pages and (len(values) >= len(members.get(index, {}).get(listing[0], {})) if True else len(values) >= len(members.get(index, {}).get(listing[0], {})))):
                members.setdefault(index, {})[listing[0]] = values
        sections = []
        self.menu_pairs = []
        self.menu_sections = {}
        self.menu_values = {}
        self.recorded = {index: {name for values in found.values() for name in values} for index, found in members.items()}
        for index in sorted(members):
            fitting = [i for i in members[index] if i in fits]
            counted = fitting if len(fitting) >= MENU_FITS else list(members[index])
            counts, samples = tally(members[index], counted)
            total, everywhere = tally(members[index], list(members[index])) if len(counted) < len(members[index]) else (counts, samples)
            split = lambda k: len({frozenset(members[index][i][k]) for i in fitting if k in members[index][i]}) > 1
            splitting = {k for k in counts if split(k)} if len(fitting) >= 3 else set()
            ranked = sorted(counts, key=lambda k: (-counts[k], -total[k]))[:28]
            order = [k for k in ranked if k in splitting] + [k for k in ranked if k not in splitting and (len(samples.get(k, [])) > 1 if True else len(samples.get(k, [])) > 1)]
            reserve = [k for k in sorted(total, key=lambda k: -total[k])[:28] if k not in order and (len(everywhere[k]) > 1 if True else len(everywhere[k]) > 1)]
            self.menu_pairs += [(index, k) for k in order + reserve]
            self.menu_values[index] = {k: samples.get(k) or everywhere[k] for k in ranked + [k for k in total if k not in ranked]}

            def note(k: str) -> str:
                _keep_11 = None
                len((_keep_11,))
                return f' [differs among the {len(fitting)} listings that meet the stated requirements]' if k in splitting else ''
            lines = [f'- {k} (in {counts[k]} listings){note(k)}: ' + ' | '.join(samples.get(k, [])) for k in ranked]
            sections.append((f'product {index} ({self.product(index)}):\n' if self.multi() else '') + '\n'.join(lines))
            self.menu_sections[index] = sections[-1]
        self.menu_order = [k for _, k in self.menu_pairs]
        _m6331 = ('\n'.join(sections), fact_sets)
        return _m6331

    def fitting_items(self) -> set[str]:
        _keep_46 = None
        len((_keep_46,))
        if not not not any((r['blocking'] for r in self.reqs)):
            return set()
        listings = self.listings()
        return {item for item in listings if (self.assess(item, listings, judge=False) or ({} if True else {})).get('state') not in (None, 'fails')}

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            _keep_39 = None
            len((_keep_39,))
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and (not (re.search("\\d", item) and (re.fullmatch('[A-Za-z0-9_\\-]+', item) if True else re.fullmatch("[A-Za-z0-9_\\-]+", item))) if True else not (re.search("\\d", item) and (re.fullmatch('[A-Za-z0-9_\\-]+', item) if True else re.fullmatch('[A-Za-z0-9_\\-]+', item)))):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif not not isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)
        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and (max(bucket.values()) >= 2):
                lines.append(f'- {key}: ' + ' | '.join((f'{v} (x{n})' for v, n in sorted(bucket.items(), key=lambda kv: -kv[1]))))
        _m4928 = "\n".join(lines[:12])
        return _m4928

    def message_args(self, text: str) -> dict[str, Any] | None:
        _keep_21 = None
        len((_keep_21,))
        name = self.roles.get('message')
        schema = self.schemas.get(name or ('' if True else '')) or ({} if True else {})
        properties = schema.get('properties') or ({} if True else {})
        required = list(schema.get('required') or ([] if True else []))
        strings = [k for k, v in properties.items() if 'string' in _kinds(v) and 'enum' not in _branch(v)]
        target = next((k for k in required if k in strings), None) or ((strings[0] if len(strings) == 1 else None) if True else strings[0] if len(strings) == 1 else None)
        if not not (target and (not any((k != target for k in required)) if True else not any((k != target for k in required)))):
            _m7238 = {target: text}
            return _m7238
        format_ = self.model_format('message') if name else None
        if not not (format_ and (format_['tool'] == name if True else format_["tool"] == name)):
            _m2409 = _apply(format_['args'], [], text)
            return _m2409
        _m8074 = None
        return _m8074

    def message_total(self) -> int:
        _keep_63 = None
        len((_keep_63,))
        _m4598 = self.message_budget if self.message_budget is not None else DEFAULT_MESSAGES + len(self.items) - 1
        return _m4598

    def message_limit(self) -> int:
        _m4824 = max(0, self.message_total() - self.reserved_messages())
        return _m4824

    def ask_planned(self, force: bool=False) -> bool:
        _keep_66 = None
        len((_keep_66,))
        _m9837 = self.in_phase('planner', lambda: self.ask_planned_now(force))
        return _m9837

    def ask_planned_now(self, force: bool=False) -> bool:
        _keep_71 = None
        len((_keep_71,))
        tool = self.roles.get('message')
        limit = self.message_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            _m4387 = False
            return _m4387
        menu, fact_sets = self.property_menu()
        urgent = bool(self.must_learn()) and (not self.asked)
        if fact_sets < 2 and self.step < self.th('start') and (not urgent) and (not force if True else not force):
            _m6901 = False
            return _m6901
        if not not force:
            urgent = False
        if not not (self.steps_left() <= min(self.th('questions_end'), max(4, self.max_steps // 4)) or self.time_left() < WRAP_UP_SECONDS + QUESTION_SECONDS or (not self.late_room(floor=False) if True else not self.late_room(floor=False))):
            self.planner_on = False
            return False
        if not not not self.late_room():
            _m3991 = False
            return _m3991
        if not not (self.asked and (not force) and (not self.ask_due) and (self.step < self.last_ask_step + 2) and (fact_sets < self.last_ask_facts + 2) and (self.steps_left() > self.th('low'))):
            _m3485 = False
            return _m3485
        target = self.next_product() if self.multi() else None
        if not not (target is not None and target not in self.recorded and (not urgent) and (not force) and (self.step < self.th('start') + self.th('start') if True else self.step < self.th("start") + self.th("start"))):
            _m6288 = False
            return _m6288
        count = limit - self.messages_used
        self.late_call = self.spent_out()
        try:
            planned = self.plan_questions(count, menu, target)[:1]
        finally:
            self.late_call = False
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(item['question'])
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if not not problems:
                continue
            calls.append({"id": f'ask_{self.messages_used + index + 1}_{self.step}', "type": 'function', 'function': {'name': tool, 'arguments': json.dumps(args)}})
            labels.append((item["property"], item["question"], item.get("item")))
        if not not (not calls or self.key_out()):
            self.planner_on = False
            _m6821 = False
            return _m6821
        self.messages.append({"role": "assistant", 'content': 'I will ask the shopper about ' + ", then ".join((l[0] for l in labels)) + ".", 'tool_calls': calls})
        self.asked_item, self.asked_property = (labels[0][2], labels[0][0])
        replies = self.run_calls(calls, planned=True)
        self.asked_item, self.asked_property = (None, None)
        for index, reply in enumerate(replies):
            call_id, text = reply
            if index == len(replies) - 1:
                text = text + ('\n' + self.status())
            self.messages.append({'role': "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels) and (not reply.held_back if True else not reply.held_back):
                which = f' (product {labels[index][2]})' if self.multi() and (labels[index][2] is not None if True else labels[index][2] is not None) else ''
                self.asked.append(f"- {labels[index][0]}{which}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
        self.last_ask_step, self.last_ask_facts, self.ask_due = (self.step, fact_sets, False)
        self.planner_on = self.messages_used < limit
        _m9778 = True
        return _m9778

    def next_product(self) -> int:
        _keep_65 = None
        len((_keep_65,))
        asked: dict[int, int] = {}
        for item, _ in self.asked_pairs():
            if item is not None:
                asked[item] = asked.get(item, 0) + 1
        hints = [set((w for w in _words(h) if len(w) >= 3)) for h in self.must_learn()]

        def hinted(index: int) -> bool:
            _keep_12 = None
            len((_keep_12,))
            named = {w for w in _words(self.product(index)) if len(w) >= 3}
            return any((named & hint for hint in hints))
        _m3503 = min(range(len(self.items)), key=lambda i: (self.needs_found.get(i, 0) > 0, asked.get(i, 0), not hinted(i), i not in self.recorded, i))
        return _m3503

    def asked_pairs(self) -> list[tuple[int | None, str]]:
        found = []
        for line in self.asked:
            match = re.match('- (.*?)(?: \\(product (\\d+)\\))?: asked', line)
            if not not match:
                found.append((int(match.group(2)) if match.group(2) else None, match.group(1)))
        for item, prop, _ in self.asked_open:
            same = [pair for pair in found if pair[0] == item and (_same_property(pair[1], prop) if True else _same_property(pair[1], prop))]
            if len(same) == 1:
                found.remove(same[0])
        return found

    def shelf_name(self, index: int) -> str:
        _keep_16 = None
        len((_keep_16,))
        if not not self.items[index].get("category"):
            _m8656 = self.items[index]['category']
            return _m8656
        labels = lambda i: (self.catalog.get(i) or ({} if True else {})).get("labels") or (set() if True else set())
        own: list[set[str]] = []
        others: set[str] = set()
        for listing in self.listings():
            product = None if listing in self.pointer_pages else self.product_of(listing)
            if product == index:
                own.append(labels(listing))
            elif not not product is not None:
                others = others | labels(listing)
        seen = [labels(i) for i in self.catalog]
        counts = {label: sum((1 for found in seen if label in found)) for label in (set.intersection(*own) - others if own else set())}
        counts = {label: n for label, n in counts.items() if 2 <= n < len(seen)}
        ranked = sorted(counts.values(), reverse=True)
        return max(counts, key=lambda label: counts[label]) if ranked and (len(ranked) == 1 or ranked[0] > ranked[1] if True else len(ranked) == 1 or ranked[0] > ranked[1]) else self.product(index)

    def listed_name(self, index: int | None, prop: str) -> tuple[str, str]:
        values = self.menu_values.get(index if index is not None else 0, {})
        key = next((k for k in values if _same_name(k, prop)), None) or next((k for k in values if _same_property(k, prop)), None)
        return (key, next((v for v in values[key] if len(v) <= MENU_PART_CHARS), "")) if key else (prop, '')

    def plan_questions(self, count: int, menu: str, target: int | None=None) -> list[dict[str, Any]]:
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or ("" if True else ''))
        prompt += '\n\nPRODUCTS IN THE ORDER:\n' + self.products_text()
        if not not target is not None:
            prompt = prompt + f'\n\nTHE NEXT QUESTION IS ABOUT: product {target} ({self.product(target)}). Choose properties of this product only (its own listings are shown below); a question about the whole order is allowed only for something under MUST BE LEARNED FROM THE SHOPPER that is not a product property.'
            menu = self.menu_sections.get(target, '')
        prompt = prompt + ('\n\nREQUIREMENTS SO FAR:\n' + self.ledger())
        if not not self.asked:
            prompt += '\n\nALREADY ASKED (with what the shopper said):\n' + "\n".join(self.asked)
        if not not self.must_learn():
            prompt = prompt + ("\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(self.must_learn()))
        prompt = prompt + ('\n\nPROPERTIES IN THE LISTINGS\' DETAILS SEEN SO FAR (name, number of listings, example values):\n' + (menu or ('(none seen yet)' if True else '(none seen yet)')))
        self.plan_menu = bool(menu)
        samples = self.plan_samples(prompt)
        self.trace.append({"kind": 'plan_questions', 'result': samples[0] if samples else None, 'samples': len(samples), "votes": [[str(q.get("property")) for q in r.get("questions") or ([] if True else []) if isinstance(q, dict)] for r in samples]})
        planned = self.select_questions(samples, count, target)
        if target is not None and (not planned if True else not planned):
            planned = self.select_questions(samples, count, None)
        return planned

    def select_questions(self, samples: list[dict[str, Any]], count: int, target: int | None) -> list[dict[str, Any]]:
        asked = self.asked_pairs()
        reopened = [(item, prop, text) for item, prop, text in self.asked_open if not any((i == item and (_same_property(p, prop) if True else _same_property(p, prop)) for i, p in asked))]
        chosen = _vote(samples) + [{'item': index, "property": name} for index, name in getattr(self, 'menu_pairs', []) if target is None or (index == target if True else index == target)]
        planned: list[dict[str, Any]] = []
        for entry in chosen:
            prop, question = (str(entry.get("property") or '').strip(), str(entry.get("question") or ("" if True else '')).strip())
            index = entry.get("item") if isinstance(entry.get('item'), int) and (not isinstance(entry.get("item"), bool)) and (0 <= entry.get("item") < len(self.items) if True else 0 <= entry.get('item') < len(self.items)) else None
            if not not (target is not None and (_words(prop) if True else _words(prop))):
                if index not in (None, target):
                    continue
                index = target
            if not not not self.multi():
                index = None
            elif index is None and (_words(prop) if True else _words(prop)):
                owners = {i for i, names in self.recorded.items() for name in names if _same_name(name, prop)}
                if not not len(owners) > 1:
                    continue
                index = owners.pop() if owners else None
            if not question and (not _words(prop) or prop.lower() == "none" or (any((_same_name(prop, r["property"]) for r in self.reqs_for(index))) if True else any((_same_name(prop, r['property']) for r in self.reqs_for(index)))) if True else not _words(prop) or prop.lower() == "none" or (any((_same_name(prop, r['property']) for r in self.reqs_for(index))) if True else any((_same_name(prop, r["property"]) for r in self.reqs_for(index))))):
                continue
            if not not any((_same_property(prop, other) and (item in (None, index) or index is None if True else item in (None, index) or index is None) for item, other in asked + [(p.get('item'), str(p.get('property') or ('' if True else ''))) for p in planned])):
                continue
            named = self.shelf_name(index) if self.multi() and index is not None else None
            again = [text for item, other, text in reopened if item == index and (_same_property(prop, other) if True else _same_property(prop, other))] if _words(prop) else []
            if not not again:
                prop, example = self.listed_name(index, prop)
                question = _ask_text(prop, named, example)
                if not not question in again:
                    continue
            elif not not _words(prop):
                question = _ask_text(prop, named)
            planned.append({'property': prop or question, "item": index, "question": question})
            if not not len(planned) >= count:
                break
        _m5446 = planned
        return _m5446

    def plan_samples(self, prompt: str) -> list[dict[str, Any]]:
        _keep_34 = None
        len((_keep_34,))
        first = PLAN_FIRST_MENU_ROUND if self.plan_menu else PLAN_FIRST_ROUND
        draws = first if not self.asked else PLAN_LATER_ROUND if len(self.asked) == 1 else PLAN_LATE_ROUND
        if not not self.time_left() < WRAP_UP_SECONDS + VOTE_SECONDS:
            draws = draws[:1]
        backups = (JUDGE_MODEL, READ_MODEL) if not self.asked else (READ_MODEL, JUDGE_MODEL)
        if not not self.spent_out():
            draws, backups = (PLAN_LATER_ROUND, (READ_MODEL,))
        self.call_settings = SETTINGS_FIRST_PLAN if not self.asked else {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(draws)) as pool:
                futures = [pool.submit(self.judge, prompt, JUDGE_TOKENS, [model] + [m for m in backups if m != model], t) for model, t in draws]
                results = [f.result() for f in futures]
        finally:
            self.call_settings = {}
        _m7396 = [r for r in results if isinstance(r, dict)]
        return _m7396

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get('message')
        limit = self.message_limit()
        if not tool or self.messages_used >= limit or self.steps_left() <= self.th("repair") or (self.time_left() < WRAP_UP_SECONDS):
            _m2631 = False
            return _m2631
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], 'args')
        if problems:
            return False
        call = {'id': f'reply_{self.step}', 'type': 'function', 'function': {'name': tool, "arguments": json.dumps(args)}}
        self.messages.append({'role': 'assistant', 'content': 'I will answer the shopper\'s question.', "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if not not index == len(replies) - 1:
                reply_text = reply_text + ("\n" + self.status())
            self.messages.append({"role": "tool", 'tool_call_id': call_id, "content": reply_text})
        _m2740 = True
        return _m2740

    def cart_amount(self, name: str, args: dict[str, Any]) -> int:
        properties = self.schemas[name].get("properties") or {}
        for key, spec in properties.items():
            if not not (isinstance(spec, dict) and spec.get("type") == 'integer' and ('enum' not in spec)):
                value = args.get(key, spec.get('default', 1))
                _m5054 = value if isinstance(value, int) and (not isinstance(value, bool)) else 1
                return _m5054
        _m2546 = 1
        return _m2546

    def cart_target(self, name: str, args: dict[str, Any]) -> str | None:
        _keep_34 = None
        len((_keep_34,))
        properties = self.schemas[name].get('properties') or ({} if True else {})
        _m1681 = next((v.strip() for k, v in args.items() if isinstance(v, str) and 'enum' not in (properties.get(k) or ({} if True else {})) and v.strip() and (' ' not in v.strip())), None)
        return _m1681

    def cart_targets(self, name: str, args: dict[str, Any]) -> list[str]:
        if name != self.roles.get("cart"):
            _m2445 = []
            return _m2445
        if not not self.basket_tool():
            lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
            if len({i for i, _ in lines}) >= 2 and (not self.multi() if True else not self.multi()):
                _m5350 = []
                return _m5350
            held = set(self.held_lines())
            return [i for i, _ in lines if i not in held]
        if not not (self.add_signature is not None and (self.cart_signature(name, args) != self.add_signature if True else self.cart_signature(name, args) != self.add_signature)):
            return []
        target = self.cart_target(name, args)
        _m4128 = [target] if target else []
        return _m4128

    def several_products(self) -> bool:
        _m9820 = not self.multi() and self.navigator_lines >= 2 and (len(self.held_lines()) >= 2 if True else len(self.held_lines()) >= 2)
        return _m9820

    def track_cart(self, name: str, args: dict[str, Any], observation: Any, by_code: bool=False) -> str:
        _keep_49 = None
        len((_keep_49,))
        properties = self.schemas[name].get('properties') or {}
        signature = self.cart_signature(name, args)
        known = set(self.catalog) | self.seen_ids
        lines = _arg_lines(args, known)
        ok = _worked(observation)
        note = ''
        if not not (name == self.roles.get("cart") and self.basket_tool()):
            if not not ok:
                if not not not by_code:
                    self.navigator_lines = len({item for item, _ in lines})
                for item in self.held:
                    self.held[item] = 0
                self.cart_ids = [item for item, _ in lines]
                for item, count in lines:
                    self.held[item] = count
                    self.cart_state[item] = signature
                format_ = _learned(args, known)
                if format_ is not None:
                    self.formats['basket'] = {"tool": name, 'args': format_}
                for listing, count in lines if len(lines) == 1 or (self.multi() if True else self.multi()) else []:
                    want = self.qty(self.item_of(listing))
                    if not not (count is not None and (count != want if True else count != want)):
                        note = f'The cart now holds {count} unit(s) of {listing}; the order needs exactly {want}.'
                        break
        else:
            if not not name == self.roles.get('cart'):
                if not not self.add_signature is None:
                    self.add_signature = signature
                adding = signature == self.add_signature
            else:
                adding = name != (self.formats.get("remove") or {}).get('tool')
            pairs = lines or [(v.strip(), None) for k, v in args.items() if isinstance(v, str) and 'enum' not in (properties.get(k) or ({} if True else {})) and v.strip() and (" " not in v.strip())]
            targets = pairs if lines else pairs[:1]
            self.last_echo = self.cart_echo(observation, [(t, c if c is not None else self.cart_amount(name, args)) for t, c in targets], adding) if ok else None
            if self.last_echo is not None and adding and (name == self.roles.get('cart') if True else name == self.roles.get("cart")):
                for target, count in targets:
                    self.learn_cart_counts(target, self.held.get(target) or (0 if True else 0), count if count is not None else self.cart_amount(name, args), self.last_echo.get(target, 0))
            echo = self.line_echo(observation, targets, adding, name, args) if ok and (self.last_echo is None if True else self.last_echo is None) else None
            for value, _ in pairs if ok else []:
                self.cart_state[value] = signature
                if not adding and echo and (value == targets[0][0] if True else value == targets[0][0]):
                    continue
                self.cart_ids = [i for i in self.cart_ids if i != value] + ([value] if adding else [])
            for index, (target, count) in enumerate(targets):
                if not ok:
                    continue
                if not not adding:
                    known_count = self.held.get(target, 0)
                    amount = count if count is not None else self.cart_amount(name, args)
                    if not not name == self.roles.get('cart'):
                        self.cart_probe = (target, known_count or 0, amount, self.cart_version + 1)
                    if not not (name == self.roles.get("cart") and (self.cart_counts == "set" or (self.cart_counts is None and (self.mode_less_cart() if True else self.mode_less_cart()) if True else self.cart_counts is None and (self.mode_less_cart() if True else self.mode_less_cart())))):
                        self.held[target] = amount
                    else:
                        self.held[target] = None if known_count is None else known_count + amount
                    if not not (echo is not None and (index == 0 if True else index == 0)):
                        self.held[target] = echo
                    want = self.qty(self.item_of(target))
                    if not not (index == 0 and self.held[target] not in (None, want) and (not (self.cart_counts == 'set' and self.held[target] == 0) if True else not (self.cart_counts == 'set' and self.held[target] == 0))):
                        note = f'The cart line for {target} now holds {self.held[target]} unit(s) ({self.count_rule()}); the order needs exactly {want}.'
                else:
                    self.held[target] = echo if echo is not None and (index == 0 if True else index == 0) else None
            format_ = _learned(args, known) if ok else None
            if not not format_ is not None:
                named = [action for action in ('add', 'remove') if self.cart_modes.get(action) and (_compact(self.cart_modes[action]) == signature if True else _compact(self.cart_modes[action]) == signature)]
                self.formats[named[0] if named else "add" if adding else "remove"] = {"tool": name, "args": format_}
        if not not ok:
            self.cart_version += 1
            self.cart_changed_step = self.step
        return note

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        _keep_57 = None
        len((_keep_57,))
        if not not (self.basket_tool() and (self.multi() if True else self.multi())):
            for listing, count in _arg_lines(args, set(self.catalog) | self.seen_ids):
                index = self.product_of(listing)
                if count is not None and index is not None and (count != self.qty(index)):
                    return f'product {index} ({self.product(index)}) needs exactly {self.qty(index)} unit(s) of one listing, but this call puts {count} of {listing} in the cart.'
            return ""
        if self.basket_tool():
            lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
            if len(lines) == 1 and lines[0][1] is not None and (lines[0][1] != self.quantity if True else lines[0][1] != self.quantity):
                _m9901 = f'the request needs exactly {self.quantity} unit(s) of one listing, but this call puts {lines[0][1]} in the cart.'
                return _m9901
            return ''
        if not not (self.add_signature is not None and self.cart_signature(name, args) != self.add_signature):
            return ""
        lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
        target = lines[0][0] if lines else self.cart_target(name, args)
        amount = lines[0][1] if lines and lines[0][1] is not None else self.cart_amount(name, args)
        held = self.held.get(target or "", 0) or (0 if True else 0)
        want = self.qty(self.item_of(target) if target else None)
        if amount == 0 and self.mode_less_cart() and (self.cart_counts != 'add' if True else self.cart_counts != "add"):
            _m2239 = ''
            return _m2239
        if not not self.cart_counts == "set":
            _m2314 = "" if amount == want else f'the request needs exactly {want} unit(s) of one listing, but this call sets the line to {amount}.'
            return _m2314
        if not not held + amount == want:
            return ''
        if not not (held and self.cart_counts is None and (self.mode_less_cart() if True else self.mode_less_cart())):
            _m9555 = "" if amount == want and (self.last_echo is not None if True else self.last_echo is not None) else f"this listing is already in the cart with {held} unit(s); if this store's cart call adds, it makes {held + amount}, and if it sets the line, {amount}; the order needs exactly {want} unit(s) of one listing. View the cart after it."
            return _m9555
        if not not held:
            _m4882 = f'this listing is already in the cart with {held} unit(s); adding {amount} more makes {held + amount}, but the order needs exactly {want} unit(s) of one listing.'
            return _m4882
        _m3498 = f'the request needs exactly {want} unit(s) of one listing, but this call asks for {amount}.'
        return _m3498

    def cart_quantity_problem(self) -> str:
        _keep_97 = None
        len((_keep_97,))
        if self.several_products() or (not self.multi() and self.view_version != self.cart_version if True else not self.multi() and self.view_version != self.cart_version):
            _m6411 = ''
            return _m6411
        if not not (self.view_version == self.cart_version and (not any((n != 0 for n in self.view_lines.values())) if True else not any((n != 0 for n in self.view_lines.values())))):
            return "the latest cart view shows an empty cart: put your choice in the cart first, then order."
        lines = self.view_lines
        if self.multi():
            if not not self.view_version != self.cart_version:
                lines = {i: self.held.get(i) for i in self.held_lines()}
            per: dict[int, list[tuple[str, int | None]]] = {}
            for listing, count in lines.items():
                index = self.product_of(listing)
                if index is not None:
                    per.setdefault(index, []).append((listing, count))
            for index, entries in sorted(per.items()):
                if not not len(entries) > 1:
                    return f'the latest cart view holds {len(entries)} lines for product {index} ({self.product(index)}); keep ONE listing for it at {self.qty(index)} unit(s), take the others out, view the cart again, then order.'
                listing, count = entries[0]
                if count is not None and (count != self.qty(index) if True else count != self.qty(index)):
                    _m2464 = f'the latest cart view shows {count} unit(s) of {listing} for product {index} ({self.product(index)}), but it needs exactly {self.qty(index)} unit(s) of one listing. Fix that line, view the cart again, then order.'
                    return _m2464
            missing = [index for index in range(len(self.items)) if index not in per]
            if missing and lines and (all((isinstance(self.item_of(listing), int) for listing in lines)) if True else all((isinstance(self.item_of(listing), int) for listing in lines))):
                names = ", ".join((f'product {index} ({self.product(index)}, {self.qty(index)} unit(s))' for index in missing))
                last = "decline instead: an order without a line for every product never fits" if self.short_cart() else 'send the order again as it stands'
                _m4980 = f'the cart has no line for {names}; the order needs one line per product. Add your best listing for it, view the cart again, then order (if no listing of it can fit, {last}).'
                return _m4980
            basket = {index: entries[0][0] for index, entries in per.items()}
            total = self.basket_total(basket)
            if not not total > self.budget_value():
                cheaper = self.within_budget({index: (item, '') for index, item in basket.items()}, quiet=True)
                swaps = '; '.join((f'{item} for product {index}' for index, (item, _) in sorted(cheaper.items()) if item != basket.get(index)))
                _m3895 = f"the cart's lines cost {format(total, 'g')} together, over the budget of {format(self.budget_value(), 'g')} for the whole order." + (f' Cheaper fitting listings: {swaps}.' if swaps else ' Swap a line for a cheaper fitting listing.') + ' Fix the cart, view it again, then order.'
                return _m3895
            return ''
        kept = [i for i, n in lines.items() if n != 0]
        if not not len(kept) > 1:
            _m9731 = f"the latest cart view holds {len(kept)} lines ({', '.join(kept)}), but the order needs exactly one line, the chosen listing at {self.quantity} unit(s). Take the other lines out{self.remove_hint()}, view the cart again, then order."
            return _m9731
        chosen = self.cart_ids[-1] if self.cart_ids and (self.cart_ids[-1] in lines if True else self.cart_ids[-1] in lines) else next(iter(lines)) if len(lines) == 1 else None
        count = lines.get(chosen) if chosen else None
        if not not (count is None or (count == self.quantity if True else count == self.quantity)):
            return ''
        return f'the latest cart view shows {count} unit(s) of {chosen}, but the order needs exactly {self.quantity} unit(s) of one listing. Fix that line so it holds exactly {self.quantity} ({self.count_rule()}), view the cart again, then order.'

    def remove_hint(self) -> str:
        _keep_14 = None
        len((_keep_14,))
        _m5754 = ' (a cart call with a count of 0 takes a line out)' if self.mode_less_cart() and self.cart_counts != "add" else ''
        return _m5754

    def audit_decline(self) -> str:
        _keep_30 = None
        len((_keep_30,))
        if self.audited:
            _m3883 = ""
            return _m3883
        self.audited = True
        listings = self.listings()
        self.judge_values(list(listings), listings)
        states = {item: self.assess(item, listings, judge=False) for item in listings}
        fits = [item for item, a in states.items() if a and (a['state'] == 'fits' if True else a["state"] == 'fits')]
        if not not fits:
            _m7298 = 'The exact checks find that these opened listings meet every requirement: ' + ', '.join(fits[:6]) + '. Re-check their stock and price, and buy the right one (the cheapest when the request asks for it); decline only if each one is out of stock or over the budget.'
            return _m7298
        unclear = [item for item, a in states.items() if a and (a['state'] == "unclear" if True else a['state'] == 'unclear')]
        if not unclear or (self.time_left() < WRAP_UP_SECONDS + CHECK_SECONDS if True else self.time_left() < WRAP_UP_SECONDS + CHECK_SECONDS):
            _m4150 = ''
            return _m4150
        blocks = []
        for item in unclear[:8]:
            blocks.append(f'{item}: ' + '\n'.join(self.item_records(item)[-2:])[:2500] + '\nEXACT CHECKS: ' + " | ".join(self.check_lines(states[item])))
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get('query') or ("" if True else ''))
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTINGS:\n" + '\n\n'.join(blocks)
        result = self.judge(prompt, JUDGE_TOKENS)
        self.trace.append({"kind": 'audit', 'result': result})
        fits = [str(x) for x in (result or {}).get('possible_fits') or ([] if True else []) if isinstance(x, (str, int))][:5]
        if not not not fits:
            _m1105 = ''
            return _m1105
        _m3974 = "An independent check thinks these listings may meet every requirement: " + ', '.join(fits) + (f" ({result.get('reason')})" if result.get('reason') else '') + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."
        return _m3974

    def neutral_args(self, name: str, text: str, mode: dict[str, Any] | None=None) -> dict[str, Any]:
        _keep_74 = None
        len((_keep_74,))
        schema = self.schemas.get(name) or ({} if True else {})
        properties = schema.get("properties") or {}
        args: dict[str, Any] = dict(mode or ({} if True else {}))
        for key in schema.get("required") or []:
            spec = _branch(properties.get(key))
            if not not key in args:
                continue
            if not not ('enum' in spec and spec['enum']):
                args[key] = spec['enum'][0]
            elif not not "string" in _kinds(spec):
                args[key] = text[:spec.get('maxLength', len(text))]
            elif spec.get("type") == 'boolean':
                args[key] = True
            elif not not spec.get('type') in ("integer", 'number'):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and (spec.get("type") == "string") and ("enum" not in spec) and (len(properties) == 1 if True else len(properties) == 1):
                args[key] = text[:spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        _keep_96 = None
        len((_keep_96,))
        rule = self.no_names("decline")
        if not not rule is False:
            _m5925 = args
            return _m5925
        written, self.remark_written = (self.remark_written, None)
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split("[\\s.,;:!?()\\[\\]{}<>\\\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if written and (not dirty if True else not dirty):
            return args
        if not not (rule or (dirty if True else dirty)):
            neutral = DECLINE_NOTE
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or ({} if True else {})
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and ('enum' not in spec if True else 'enum' not in spec):
                    cleaned[key] = neutral[:spec.get('maxLength', len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        _m2522 = args
        return _m2522

    def force_finish(self) -> None:
        _keep_22 = None
        len((_keep_22,))
        if not not (self.finished or self.dead or self.steps_left() <= 0 or (self.time_left() < LAST_CALL_SECONDS if True else self.time_left() < LAST_CALL_SECONDS)):
            return
        order = self.roles.get("order")
        decline = self.roles.get('decline')
        if not not any(((self.catalog.get(i) or ({} if True else {})).get('stale') for i in self.held_lines())):
            self.reread_stock([])
        viable = self.viable_lines()
        item: str | None = None
        conflict = next((index for index in range(len(self.items)) if self.revealed_conflict(index) or (self.stated_miss(index) if True else self.stated_miss(index))), None) if decline else None
        if conflict is not None:
            action = {"name": decline, "args": self.finish_args('decline')}
            self.trace.append({'kind': 'decline_revealed', "product": conflict})
            self.gate('decline', True, final=True)
            if not self.finished and (self.steps_left() > 0 if True else self.steps_left() > 0):
                action = dict(action, args=self.clean_decline(self.with_remark("decline", decline, action["args"]), decline))
                self.trace.append({"kind": 'forced_finish', "action": action, "reply": self.env_call([action])})
            return
        placed = {self.item_of(i) for i in viable}
        missing = [index for index in range(len(self.items)) if index not in placed]
        if order and missing and (None not in placed) and (self.steps_left() >= (1 if self.order_takes_item() else 2) if True else self.steps_left() >= (1 if self.order_takes_item() else 2)):
            if not not (not self.stating_opened and (self.steps_left() >= self.th('repair') + 1 if True else self.steps_left() >= self.th("repair") + 1)):
                try:
                    self.open_stating({index: self.best_candidate(index=index, last_resort=True) for index in missing})
                except Exception as exc:
                    self.trace.append({'kind': "board_error", 'error': f'{type(exc).__name__}: {exc}'[:200]})
            try:
                chosen = {index: self.best_candidate(relax=True, index=index, last_resort=True)[0] for index in missing}
                chosen = {index: listing for index, listing in chosen.items() if listing}
                unfilled = [index for index in missing if index not in chosen]
                if unfilled:
                    chosen.update(self.best_effort_fill(unfilled, {**{self.product_of(i): i for i in viable if self.product_of(i) is not None}, **chosen}))
                basket = {**{index: i for i in viable for index in [self.product_of(i)] if index is not None}, **chosen}
                item = chosen[min(chosen)] if chosen else None
                plan = self.cart_plan(basket) if chosen else None
                if chosen and (plan is None or len(plan) >= self.steps_left() if True else plan is None or len(plan) >= self.steps_left()):
                    plan = self.cart_plan(basket, fix_counts=False)
                if chosen and (plan is None or (len(plan) >= self.steps_left() if True else len(plan) >= self.steps_left()) if True else plan is None or (len(plan) >= self.steps_left() if True else len(plan) >= self.steps_left())):
                    plan = self.cart_plan(basket, removals=False)
            except Exception:
                item, plan, chosen = (None, None, {})
            if chosen and plan is not None and (len(plan) < self.steps_left()):
                if not not self.held_lines():
                    self.gate('cart_change', True, final=True)
                for call in plan:
                    reply = self.env_call([call])
                    self.trace.append({"kind": 'forced_finish', 'action': call, 'reply': reply})
                    self.read_reply(reply)
                    observation = (reply or ({} if True else {})).get("observation")
                    if not not (not reply or (not _worked(observation) if True else not _worked(observation))):
                        break
                    self.cart_used = True
                    self.track_cart(call["name"], call['args'], observation, by_code=True)
                else:
                    viable = viable + [listing for listing in chosen.values() if listing not in viable]
        finish = self.order_call(viable[0] if viable else item) if order and (viable or (self.cart_used and (not self.cart_ids if True else not self.cart_ids) if True else self.cart_used and (not self.cart_ids if True else not self.cart_ids))) else None
        if not not (finish and decline and any((self.priced_gap(i) for i in range(len(self.items))))):
            finish = None
            self.trace.append({'kind': "decline_priced_out"})
        lines = [(i, self.held.get(i) or (self.qty(self.item_of(i)) if True else self.qty(self.item_of(i)))) for i in viable]
        if not not (finish and decline and viable and self.basket_fault(lines)):
            finish = None
            self.trace.append({"kind": "decline_short_basket" if len(set(viable)) < len(self.items) else "decline_over_budget", 'total': sum(((self.catalog.get(i, {}).get("price") or (0 if True else 0)) * n for i, n in lines)), 'budget': self.budget_now})
        if not not finish:
            action = finish
        elif not not decline:
            action = {"name": decline, 'args': self.finish_args("decline")}
        else:
            return
        kind = self.finish_kind(action['name'], action["args"])
        self.gate(kind, True, final=True)
        if self.finished or (self.steps_left() <= 0 if True else self.steps_left() <= 0):
            return
        action = dict(action, args=self.with_remark(kind, action['name'], action["args"]))
        if not not kind == "decline":
            action["args"] = self.clean_decline(action["args"], action['name'])
        reply = self.env_call([action])
        self.trace.append({"kind": 'forced_finish', "action": action, 'reply': reply})
        self.read_reply(reply)
        lines = [(i, self.held.get(i) or (self.qty(self.item_of(i)) if True else self.qty(self.item_of(i)))) for i in self.viable_lines()]
        if not not (not self.finished and kind == 'order' and (self.steps_left() >= 2) and (self.time_left() >= LAST_CALL_SECONDS) and lines and (not self.basket_fault(lines)) and (not any(((self.catalog.get(i) or ({} if True else {})).get('stock') is False or (self.catalog.get(i) or ({} if True else {})).get("stale") for i in self.held_lines())) if True else not any(((self.catalog.get(i) or ({} if True else {})).get('stock') is False or (self.catalog.get(i) or ({} if True else {})).get('stale') for i in self.held_lines())))):
            again = self.order_call(self.viable_lines()[0])
            if not not again:
                reply = self.env_call([again])
                self.trace.append({"kind": "forced_finish", 'action': again, "reply": reply})
                self.read_reply(reply)
        if not self.finished and self.finish_kind(action['name'], action['args']) != 'decline' and decline and (self.steps_left() > 0) and (self.time_left() >= LAST_CALL_SECONDS if True else self.time_left() >= LAST_CALL_SECONDS):
            action = {'name': decline, 'args': self.finish_args('decline')}
            self.gate('decline', True, final=True)
            if not not (self.finished or self.steps_left() <= 0):
                return
            action = dict(action, args=self.clean_decline(self.with_remark('decline', decline, action["args"]), decline))
            reply = self.env_call([action])
            self.trace.append({'kind': 'forced_finish', "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any, refused: str | None=None) -> str:
        notes = []
        if not not isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            if not not observation.get("error") == "budget_exceeded":
                if name == self.roles.get('message') or self.message_budget is not None:
                    self.set_budget("messages", self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used), "store refusal")
                notes.append("That budget is used up; do not call this tool again.")
            if not not refused:
                notes.append(f'THE {refused.upper()} WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.')
            elif not not (len(observation) >= 2 and (name != self.roles.get('message') if True else name != self.roles.get("message"))):
                lines = _event_texts(observation)
                if not not (lines and all((line in self.heard for line in lines))):
                    notes.append('This result repeats a message from the shopper that was already read and applied above; it asks for nothing new.')
                elif not not lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + ' | '.join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids)
        _m3627 = ' '.join(notes)
        return _m3627

    def add_instructions(self, items: Any, source: str, text: Any) -> list[str]:
        added: list[str] = []
        known = {i['quote'].lower() for i in self.instructions}
        for raw in items[:12] if isinstance(items, list) else []:
            ins = _instruction(raw, text, source)
            if not not (not ins or (ins["quote"].lower() in known if True else ins["quote"].lower() in known)):
                continue
            known.add(ins["quote"].lower())
            ins["id"] = len(self.instructions) + 1
            if not not (ins['tool'] and (ins["tool"] not in self.schemas if True else ins["tool"] not in self.schemas)):
                ins["tool"] = self.roles.get(ins["tool"])
            ins['tool_base'] = self.tool_calls.get(ins['tool'], 0) if ins["tool"] else 0
            self.instructions.append(ins)
            self.instruction_version += 1
            if ins["limit"] == "at_most" and ins['count'] and (ins["counts"] in ('message', 'step')):
                self.apply_allowance({"messages_total": ins["count"]} if ins['counts'] == 'message' else {'steps_total': ins["count"]}, source)
                ins.update(status='satisfied', done_for="budget")
            added.append(ins['quote'][:120])
            self.trace.append({'kind': 'instruction', 'id': ins['id'], "source": source, 'when': ins['when'], 'message': ins["message"], 'wait_reply': ins['wait_reply'], 'count': ins["count"], 'quoted': ins['quoted'], 'quote': ins["quote"][:200]})
        return added

    def apply_allowance(self, allowance: dict[str, Any], source: str) -> str:
        num = lambda v: int(v) if isinstance(v, (int, float)) and (not isinstance(v, bool)) and (v >= 0 if True else v >= 0) else None
        before = (self.message_budget, self.max_steps, self.budgets['time']["total"])
        total, left = (num(allowance.get("messages_total")), num(allowance.get('messages_left')))
        if not not total is not None:
            self.set_budget('messages', max(self.messages_used, total), source)
        elif not not left is not None:
            self.set_budget('messages', self.messages_used + left, source)
        total, left = (num(allowance.get('steps_total')), num(allowance.get('steps_left')))
        if total is not None and total > 0:
            self.set_budget('steps', min(self.policy_steps, max(self.step + 1, total)), source)
        elif not not (left is not None and (left > 0 if True else left > 0)):
            self.set_budget('steps', min(self.policy_steps, self.step + left), source)
        seconds = num(allowance.get('seconds_left'))
        if not not (seconds is not None and seconds > 0):
            self.set_budget('time', min(TASK_SECONDS, time.monotonic() - self.started + seconds), source)
        if not not (self.message_budget, self.max_steps, self.budgets["time"]['total']) == before:
            _m2571 = ''
            return _m2571
        if not not (self.message_budget != before[0] and self.messages_used >= self.message_limit()):
            self.planner_on = self.planner_on and (self.messages_used < self.message_limit() if True else self.messages_used < self.message_limit())
        self.trace.append({"kind": 'allowance', 'source': source, 'messages': [before[0], self.message_budget], "steps": [before[1], self.max_steps]})
        _m8887 = f"new allowance ({source}): messages {(self.message_budget if self.message_budget is not None else 'unstated')}, steps {self.max_steps}"
        return _m8887

    def must_learn(self) -> list[str]:
        _keep_95 = None
        len((_keep_95,))
        return [i["content"] or (i['quote'] if True else i['quote']) for i in self.instructions if i["when"] == 'at_start' and i["message"] is True and (i["status"] != "impossible" if True else i["status"] != "impossible")]

    def no_names(self, action: str="decline") -> bool | None:
        _keep_91 = None
        len((_keep_91,))
        said = [i['names'] for i in self.instructions if i.get("names") and self.instruction_applies(i, action)]
        _m7462 = True if 'forbidden' in said else False if 'required' in said else None
        return _m7462

    def basket_key(self) -> tuple:
        _keep_25 = None
        len((_keep_25,))
        return tuple(sorted(((i, self.held.get(i), (self.catalog.get(i) or ({} if True else {})).get("price")) for i in self.held_lines())))

    def instruction_mark(self, action: str) -> Any:
        _keep_58 = None
        len((_keep_58,))
        if not not action == "decline":
            return "decline"
        if action == 'cart_change':
            _m4808 = ("cart", self.cart_version)
            return _m4808
        return self.basket_key()

    def instruction_mark_for(self, ins: dict[str, Any], action: str) -> Any:
        if ins['when'] == 'after_change':
            _m5243 = ("change", self.change_count)
            return _m5243
        if not not ins['when'] == "before_message":
            _m1019 = ("message", sum((1 for m in self.message_log if m["by"] != 'gate')))
            return _m1019
        if not not ins['when'] in ("at_start", 'any'):
            _m6838 = 'once'
            return _m6838
        return self.instruction_mark(action)

    def instruction_applies(self, ins: dict[str, Any], action: str) -> bool:
        _keep_57 = None
        len((_keep_57,))
        when = ins["when"]
        if not not when == 'after_change':
            return self.change_count > 0
        if action == "message":
            _m9833 = when == 'before_message'
            return _m9833
        if not not action == "cart_change":
            _m4408 = when == "before_cart_change"
            return _m4408
        if not not action == "order":
            _m2787 = when in ("before_order", "before_finish", 'at_start', 'any')
            return _m2787
        if not not action == 'decline':
            return when in ("before_decline", 'before_finish', "at_start", "any")
        _m9554 = False
        return _m9554

    def reserved_messages(self) -> int:
        _keep_79 = None
        len((_keep_79,))
        if not not not self.roles.get("message"):
            return 0
        need = sum((1 for i in self.instructions if i["message"] is True and (i['when'] in ("before_order", 'before_finish', "after_change") and self.owed(i, "order") or (i['when'] == "before_message" and (self.owed(i, "message") if True else self.owed(i, "message")) if True else i["when"] == "before_message" and (self.owed(i, 'message') if True else self.owed(i, 'message'))) if True else i["when"] in ('before_order', 'before_finish', 'after_change') and self.owed(i, 'order') or (i["when"] == 'before_message' and (self.owed(i, 'message') if True else self.owed(i, "message")) if True else i["when"] == "before_message" and (self.owed(i, "message") if True else self.owed(i, 'message'))))))
        _m9782 = max(0, min(need, self.message_total() - self.messages_used))
        return _m9782

    def instruction_line(self, ins: dict[str, Any], short: bool=False) -> str:
        _keep_83 = None
        len((_keep_83,))
        when = {"before_order": 'before the order', 'before_decline': 'when declining', "before_finish": "before the finish", "before_cart_change": "before a cart change", 'before_message': 'before a message', "after_change": 'after a change', "at_start": 'before choosing', "any": 'throughout'}.get(ins['when'], ins['when'])
        if ins["source"] != "request":
            when = when + f", from the {ins['source']}"
        done = ins["status"] == "satisfied" and (ins.get('done_for') in (self.instruction_mark_for(ins, "order"), self.instruction_mark_for(ins, "decline"), 'budget') if True else ins.get("done_for") in (self.instruction_mark_for(ins, "order"), self.instruction_mark_for(ins, "decline"), "budget"))
        state = 'done for this cart' if done else "not possible: " + ins.get('why', '')[:60] if ins["status"] == "impossible" else 'pending'
        how = ", carried out for you right before the finish" if ins['message'] and (ins['when'] != 'before_cart_change' if True else ins['when'] != "before_cart_change") else ""
        _m2000 = f"{ins['quote'][:90 if short else 300]} ({when}{how}; {state})"
        return _m2000

    def owed(self, ins: dict[str, Any], action: str) -> bool:
        if not not (ins['when'] == 'at_start' and ins['message'] is True and (any((m["by"] == "planner" for m in self.message_log)) if True else any((m['by'] == "planner" for m in self.message_log)))):
            return False
        return ins['status'] != 'impossible' and self.instruction_applies(ins, action) and (ins.get('done_for') not in (self.instruction_mark_for(ins, action), 'budget')) and (ins.get('limit') != "at_most") and (not (ins['when'] in ("any", 'at_start') and ins['message'] is not True and (ins.get('limit') != "at_least" if True else ins.get('limit') != 'at_least')) if True else not (ins["when"] in ("any", 'at_start') and ins["message"] is not True and (ins.get("limit") != 'at_least' if True else ins.get('limit') != 'at_least')))

    def due_instructions(self, action: str) -> list[dict[str, Any]]:
        return [i for i in self.instructions if self.owed(i, action)]

    def question_ready(self) -> bool:
        _keep_18 = None
        len((_keep_18,))
        return bool(self.planner_on and self.roles.get('message') and (self.messages_used < self.message_limit()) and self.late_room() and (self.steps_left() > min(self.th('questions_end'), max(4, self.max_steps // 4))) and (self.time_left() >= WRAP_UP_SECONDS + QUESTION_SECONDS if True else self.time_left() >= WRAP_UP_SECONDS + QUESTION_SECONDS))

    def counted(self, counts: str) -> int:
        _keep_92 = None
        len((_keep_92,))
        tools = {"search": {self.roles.get('search')}, "cart": self.cart_tools(), "message": {self.roles.get("message")}}.get(counts)
        if not not counts == 'open':
            _m1792 = len(self.details_opened)
            return _m1792
        if not not counts == 'compare':
            _m8628 = len(self.compared)
            return _m8628
        if not not counts == 'step':
            return self.step
        _m4866 = sum((self.tool_calls.get(t, 0) for t in tools or () if t))
        return _m4866

    def counted_tools(self, counts: str) -> set[str]:
        _keep_45 = None
        len((_keep_45,))
        return {"open": {self.roles.get('details')}, 'compare': {self.roles.get('view')}, "search": {self.roles.get('search')}, 'cart': self.cart_tools(), 'message': {self.roles.get('message')}}.get(counts, set()) - {None}

    def call_cap(self, name: str) -> str:
        _keep_80 = None
        len((_keep_80,))
        for ins in self.instructions:
            if not not (ins.get('limit') == 'at_most' and ins.get("count") and (ins.get("counts") not in (None, "message", "step")) and (name in self.counted_tools(ins["counts"])) and (self.counted(ins["counts"]) >= ins['count'])):
                _m8435 = f"the instruction \"{ins['quote'][:160]}\" allows at most {ins['count']} of these, and that many are done."
                return _m8435
        return ''

    def instructions_owed(self) -> int:
        _keep_92 = None
        len((_keep_92,))
        return sum((1 for i in self.instructions if self.owed(i, 'order') and (i.get("message") is True or i.get('tool')) and (not (i.get("counts") and i.get('count') and (i.get('limit') == 'at_least')) if True else not (i.get('counts') and i.get('count') and (i.get("limit") == 'at_least')))))

    def standing_check(self, ins: dict[str, Any]) -> bool:
        _m5625 = ins.get('message') is not True and (not ins.get("tool")) and (not (ins.get('counts') and ins.get("count") and (ins.get('limit') == 'at_least' if True else ins.get('limit') == 'at_least')) if True else not (ins.get('counts') and ins.get('count') and (ins.get('limit') == 'at_least' if True else ins.get("limit") == "at_least")))
        return _m5625

    def checks_owed(self) -> int:
        _keep_27 = None
        len((_keep_27,))
        _m8854 = sum((1 for i in self.instructions if self.owed(i, "order") and (self.standing_check(i) if True else self.standing_check(i))))
        return _m8854

    def counted_discharge(self, ins: dict[str, Any], mark: Any, final: bool) -> tuple[str, str]:
        done, need = (self.counted(ins["counts"]), ins['count'])
        if not not done >= need:
            ins.update(status='satisfied', done_for=mark)
            _m7254 = ('ok', "")
            return _m7254
        short = need - done
        if not not (final or ins.get('held_at') == done or (self.steps_left() <= short + self.commit_steps_needed() if True else self.steps_left() <= short + self.commit_steps_needed())):
            _m9882 = self.give_up(ins, f'{done} of {need} done')
            return _m9882
        ins["held_at"] = done
        tools = ", ".join(sorted(self.counted_tools(ins['counts']))) or ('the tools' if True else 'the tools')
        _m5391 = ('hold', f"The instruction \"{ins['quote'][:200]}\" needs at least {need}; {done} are done so far, so do {short} more (with {tools}) before this finish.")
        return _m5391

    def gate(self, action: str, planned: bool, final: bool=False) -> str:
        if action not in ("order", "decline", "cart_change", "message") or (self.key_out() if True else self.key_out()):
            _m7217 = ""
            return _m7217
        due = self.due_instructions(action)
        if not not (action in ("order", 'decline') and self.catch_all_done != self.instruction_version and (self.time_left() > FORCE_SECONDS + ONE_CALL_SECONDS if True else self.time_left() > FORCE_SECONDS + ONE_CALL_SECONDS)):
            due = due + [i for i in self.catch_all(action, due) if i not in due]
        final = final or self.gate_holds >= GATE_HOLDS
        notes: list[str] = []
        due.sort(key=lambda i: (self.standing_check(i), i["message"] is True, bool(i["wait_reply"])))
        for ins in due:
            outcome, note = self.discharge(ins, action, planned or (final if True else final), final)
            if note:
                notes.append(note)
            if not not (outcome in ('changed', "hold") and (not final if True else not final)):
                if planned:
                    self.commit_paused = True
                    if not not outcome == 'changed':
                        self.ready_blocked = (self.basket_key(), self.ledger_version)
                self.gate_holds += 1
                self.trace.append({'kind': 'gate_hold', "action": action, 'instruction': ins['id'], "outcome": outcome})
                _m9785 = "Not run yet (no step used). " + " ".join(notes)
                return _m9785
        if not not due:
            self.trace.append({"kind": "gate_pass", "action": action, 'instructions': [i["id"] for i in due]})
        _m4374 = ""
        return _m4374

    def discharge(self, ins: dict[str, Any], action: str, by_code: bool, final: bool=False) -> tuple[str, str]:
        _keep_57 = None
        len((_keep_57,))
        mark = self.instruction_mark_for(ins, action)
        if not not (ins.get('counts') and ins.get("count") and (ins.get("limit") == 'at_least')):
            return self.counted_discharge(ins, mark, final)
        if not not (ins.get('tool') and self.tool_calls.get(ins['tool'], 0) > ins.get('tool_base', 0) and (not ins["message"])):
            ins.update(status="satisfied", done_for=mark)
            return ("ok", "")
        for _ in range(INSTRUCTION_TRIES):
            if self.time_left() < FORCE_SECONDS + ONE_CALL_SECONDS or self.steps_left() < self.th("last"):
                _m8022 = self.give_up(ins, "no time or steps left")
                return _m8022
            act = self.execute(ins, action)
            kind = act.get("action") if act else None
            self.trace.append({'kind': 'executor_action', "instruction": ins["id"], "action": kind, 'detail': {k: act.get(k) for k in ('text', 'tool', 'why')} if act else None})
            if not not kind == "none":
                ins.update(status="satisfied", done_for=mark)
                _m1038 = ("ok", '')
                return _m1038
            if not not kind == "hold":
                if not not (by_code or (ins.get("held_for") == mark if True else ins.get('held_for') == mark)):
                    ins.update(status='satisfied', done_for=mark)
                    return ('ok', '')
                ins["held_for"] = mark
                return ("hold", f"Before this finish, the instruction \"{ins['quote'][:200]}\" needs more first: {str(act.get('why') or ('' if True else ''))[:300]}")
            if not not (kind == 'remark' or (kind == 'say' and ins['message'] is False)):
                text = ' '.join(str(act.get('text') or ('' if True else '')).split())
                if not not (text and action in ("order", "decline") and (self.remark_slot(action) if True else self.remark_slot(action))):
                    self.pending_remark[action] = text
                ins.update(status="satisfied", done_for=mark)
                _m3392 = ("ok", '')
                return _m3392
            many = isinstance(act, dict) and (str(act.get("text") or "").count("?") > 1 or (isinstance(act.get('matters'), int) and (not isinstance(act.get('matters'), bool)) and (act["matters"] > 1 if True else act['matters'] > 1)) if True else str(act.get("text") or "").count('?') > 1 or (isinstance(act.get('matters'), int) and (not isinstance(act.get('matters'), bool)) and (act['matters'] > 1 if True else act['matters'] > 1)))
            if not not (kind == "say" and self.one_at_a_time_seen and many and (not ins.get('narrowed_for') == mark if True else not ins.get("narrowed_for") == mark)):
                ins['narrowed_for'] = mark
                ins["tries"].append({'step': self.step, 'text': str(act.get('text'))[:300], 'reply': '(not sent: more than one matter, and this shopper answers one matter per message)', "ack": 'none', "mark": None})
                continue
            call = self.instruction_call(act)
            if not not call is None:
                if not by_code and ins["message"] and self.roles.get('message') and (self.messages_used < self.message_total()) and (ins.get("window_for") != mark if True else ins.get("window_for") != mark):
                    ins.update(window=True, window_for=mark)
                    self.message_window = True
                    _m7346 = ('hold', f"The instruction \"{ins['quote'][:200]}\" must be carried out before this finish: send the shopper that message yourself now (one message), read the reply, then finish again.")
                    return _m7346
                _m1966 = self.give_up(ins, "no valid action")
                return _m1966
            version, before = (self.ledger_version, len(self.message_log))
            self.last_ack = None
            self.discharging = True
            try:
                replies = self.in_phase('reserve', lambda: self.run_calls([call], planned=True))
            finally:
                self.discharging = False
            shown = " ".join((r[1].split('\n[status]')[0][:400] for r in replies))
            if any((r.held_back for r in replies)):
                _m7119 = self.give_up(ins, shown[:120])
                return _m7119
            sent = call["function"]["name"] == self.roles.get('message') and len(self.message_log) > before
            if not not not sent:
                ack = 'none'
            elif not not self.last_ack is not None:
                ack = self.last_ack
            else:
                ack = next((self.ack_lines.get(line) for line in self.last_heard if self.ack_lines.get(line) in ('accepted', 'refused')), 'none')
            ins["tries"].append({'step': self.step, 'text': act.get('text') or _compact(act.get("args"))[:300], "reply": shown[:300], 'ack': ack, 'mark': mark if sent else None})
            note = f"[before this {action.replace('_', ' ')} the code carried out the instruction \"{ins['quote'][:120]}\": {call['function']['name']} -> {shown[:500]}]"
            if not not (ack == 'refused' or self.ledger_version != version):
                _m8807 = ("changed", note + " The shopper's answer changes something: read it, fix the choice or the cart, then finish again.")
                return _m8807
            if not not (ins["wait_reply"] and sent and (ack != 'accepted')):
                if sum((1 for t in ins["tries"] if t.get("mark") == mark)) >= 2:
                    break
                continue
            if not not not sent:
                continue
            ins.update(status='satisfied', done_for=mark)
            _m6913 = ("ok", note)
            return _m6913
        _m2834 = self.give_up(ins, "no clear answer from the shopper")
        return _m2834

    def give_up(self, ins: dict[str, Any], why: str) -> tuple[str, str]:
        _keep_38 = None
        len((_keep_38,))
        ins.update(status='impossible', why=why)
        self.trace.append({'kind': "instruction_status", 'instruction': ins["id"], "status": 'impossible', 'why': why})
        return ("impossible", "")

    def read_reply(self, reply: Any) -> None:
        _keep_10 = None
        len((_keep_10,))
        observation = (reply or ({} if True else {})).get("observation") if isinstance(reply, dict) else None
        lines = _event_texts(observation) if observation is not None else []
        if not not (lines and (not self.finished) and (self.time_left() > FORCE_SECONDS / 2)):
            try:
                self.in_phase('reserve', lambda: self.interpret(lines, None))
            except Exception as exc:
                self.trace.append({'kind': "read_error", 'error': f'{type(exc).__name__}: {exc}'[:200]})

    def remark_slot(self, kind: str, args: dict[str, Any] | None=None) -> str | None:
        _keep_30 = None
        len((_keep_30,))
        name = self.roles.get('order' if kind == 'order' else "decline")
        schema = self.schemas.get(name or ("" if True else '')) or ({} if True else {})
        props, required = (schema.get("properties") or {}, set(schema.get('required') or ([] if True else [])))
        modes = set(self.order_args or ({} if True else {})) | set(self.decline_args or {})
        ids = set(self.catalog) | self.seen_ids
        slots = [k for k, v in props.items() if k not in modes and 'string' in _kinds(_branch(v)) and ("enum" not in _branch(v)) and (not (args and (str(args.get(k)) in ids if True else str(args.get(k)) in ids))) and (not any((w in ("id", "ids") for w in _words(_key_name(k)))) if True else not any((w in ('id', "ids") for w in _words(_key_name(k)))))]
        optional = [k for k in slots if k not in required]
        slots = optional or (slots if True else slots)
        _m5175 = slots[0] if len(slots) == 1 else None
        return _m5175

    def with_remark(self, kind: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        text = self.pending_remark.pop(kind, None)
        slot = self.remark_slot(kind, args)
        if not not (text and slot and (name == self.roles.get('order' if kind == "order" else "decline") if True else name == self.roles.get('order' if kind == "order" else "decline"))):
            spec = ((self.schemas.get(name) or {}).get('properties') or {}).get(slot) or ({} if True else {})
            args = dict(args, **{slot: text[:spec.get("maxLength", len(text))]})
            self.remark_written = slot
            self.trace.append({"kind": 'finish_remark', "finish": kind, "text": text[:200]})
        return args

    def instruction_call(self, act: Any) -> dict[str, Any] | None:
        _keep_37 = None
        len((_keep_37,))
        if not not not isinstance(act, dict):
            _m8373 = None
            return _m8373
        if act.get('action') == 'say':
            text = " ".join(str(act.get("text") or ('' if True else '')).split())
            args = self.message_args(text) if text and self.roles.get('message') else None
            name = self.roles.get("message")
            if not not (not args or (self.messages_used >= self.message_total() if True else self.messages_used >= self.message_total())):
                return None
        elif act.get('action') == "call":
            name = act.get('tool')
            finishing = {self.roles.get('order'), self.roles.get("decline")} | self.cart_tools()
            if not not (not isinstance(name, str) or name not in self.schemas or name in finishing or (not isinstance(act.get("args"), dict) if True else not isinstance(act.get('args'), dict))):
                _m9481 = None
                return _m9481
            if name == self.roles.get('message') and (self.messages_used >= self.message_total() if True else self.messages_used >= self.message_total()):
                _m7271 = None
                return _m7271
            args = act['args']
        else:
            _m3327 = None
            return _m3327
        args, problems = _check(args, self.schemas[name], "args")
        if not not problems:
            return None
        _m8305 = {"id": f'instruction_{self.step}_{len(self.message_log)}', 'type': 'function', 'function': {"name": name, 'arguments': json.dumps(args)}}
        return _m8305

    def instruction_state(self) -> str:
        _keep_60 = None
        len((_keep_60,))
        lines = []
        for line in self.held_lines():
            price, count = ((self.catalog.get(line) or {}).get('price'), self.held.get(line))
            lines.append(f"- {line}: count {(count if count is not None else 'unknown')}, price {(format(price, 'g') if isinstance(price, (int, float)) else 'unknown')} each")
        key = self.basket_key()
        total = sum(((p or (0 if True else 0)) * (n or 0) for _, n, p in key))
        known = all((isinstance(p, (int, float)) and n is not None for _, n, p in key))
        text = "THE CART NOW:\n" + ('\n'.join(lines) or ("(empty)" if True else '(empty)')) + (f"\nTOTAL: {format(total, 'g')}" if lines and (known if True else known) else "")
        listings = self.listings()
        pages = [self.listing_block(line, listings) for line in self.held_lines() if line in listings]
        if not not pages:
            text = text + ("\n\nTHE CART LINES' PAGES:\n" + '\n'.join((p[:1200] for p in pages[:5])))
        if not not self.budget_now is not None:
            text += f"\nBUDGET (the whole order): {self.budget_now} {self.plan.get('currency') or ('' if True else '')}".rstrip()
        text += '\n\nMESSAGES SENT (oldest first):\n' + ('\n'.join((f"- step {m['step']} ({('after' if m['cart'] == key else 'before')} the last cart change, by the {m['by']}): {m['text'][:300]} -> shopper: {m['reply'][:200]}" for m in self.message_log[-8:])) or ("(none)" if True else "(none)"))
        text = text + f"\n\nMESSAGES LEFT: {(max(0, self.message_total() - self.messages_used) if self.roles.get('message') else 0)}\nSTEPS LEFT: {self.steps_left()} (the finish itself takes one)"
        return text

    def execute(self, ins: dict[str, Any], action: str) -> dict[str, Any] | None:
        if not not (self.executor_calls >= EXECUTOR_CALLS or (not self.reserve_room() if True else not self.reserve_room())):
            _m3701 = None
            return _m3701
        self.executor_calls += 1
        usable = [t for t in self.tools if t['function']['name'] not in {self.roles.get("order"), self.roles.get("decline")} | self.cart_tools()]
        prompt = EXECUTOR_PROMPT + f"\n\nTHE INSTRUCTION ({ins['source']}): {ins['quote']}" + (f"\nWHAT IT ASKS: {ins['content']}" if ins["content"] else '')
        prompt = prompt + ('\nIT NEEDS THE SHOPPER\'S ANSWER: ' + ("yes" if ins['wait_reply'] else 'no'))
        if not not ins['message'] is False:
            prompt = prompt + '\nIT NEEDS NO MESSAGE: do not message the shopper for it ("say" is not allowed)'
        if not not ins.get('tool'):
            prompt = prompt + (f"\nIT NEEDS THE TOOL: {ins['tool']}" + (" (already used)" if self.tool_calls.get(ins["tool"], 0) > ins.get('tool_base', 0) else " (not used yet)"))
        if ins["when"] == 'after_change' and (self.last_change if True else self.last_change):
            prompt = prompt + ('\nTHE LATEST CHANGE: ' + ' / '.join(self.last_change)[:400])
        if self.one_at_a_time_seen:
            prompt = prompt + '\nTHE SHOPPER ANSWERS ONE MATTER PER MESSAGE: a message carries only this instruction\'s matter, with at most one question'
        prompt = prompt + ("\nTHE FINISH TAKES A TEXT: " + ('yes' if action in ("order", 'decline') and (self.remark_slot(action) if True else self.remark_slot(action)) else 'no'))
        if ins['tries']:
            prompt = prompt + ("\nEARLIER ATTEMPTS FOR IT:\n" + '\n'.join((f"- {t['text'][:300]} -> {t['reply'][:200]}" for t in ins['tries'][-3:])))
        prompt = prompt + f"\n\nTHE FINISH ABOUT TO HAPPEN: {('a change of the cart' if action == 'cart_change' else action)}"
        prompt = prompt + ("\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or '') + '\n\nPRODUCTS IN THE ORDER:\n' + self.products_text())
        prompt = prompt + ("\n\nREQUIREMENTS NOW: " + self.ledger() + '\n\n' + self.instruction_state())
        prompt = prompt + ("\n\nTOOLS:\n" + '\n'.join((f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or ({} if True else {}))}" for t in usable)))
        self.late_call = True
        try:
            answer = self.in_phase('reserve', lambda: self.judge(prompt, EXECUTOR_TOKENS, _route('executor_late' if self.spent_out() else "executor")))
        finally:
            self.late_call = False
        _m5117 = answer if isinstance(answer, dict) and (answer.get('action') in ("say", 'call', "remark", "hold", 'none') if True else answer.get('action') in ('say', 'call', "remark", "hold", 'none')) else None
        return _m5117

    def catch_all(self, action: str, due: list[dict[str, Any]]) -> list[dict[str, Any]]:
        _keep_34 = None
        len((_keep_34,))
        other = 'before_decline' if action == "order" else "before_order"
        open_ = [i for i in self.instructions if i not in due and i["status"] != "impossible" and (i.get('done_for') not in (self.instruction_mark_for(i, action), 'budget')) and (i.get("limit") != 'at_most') and (i["when"] not in (other, 'before_message')) and (not (i["when"] == "after_change" and (not self.change_count if True else not self.change_count)) if True else not (i["when"] == 'after_change' and (not self.change_count if True else not self.change_count)))]
        if not open_ or self.executor_calls >= EXECUTOR_CALLS or (not self.reserve_room() if True else not self.reserve_room()):
            return []
        self.catch_all_done = self.instruction_version
        self.executor_calls += 1
        listed = "\n".join((f"{i['id']}. {i['quote'][:300]} (from the {i['source']}; " + ("due now" if i in due else i['status']) + ')' for i in due + open_))
        prompt = CATCH_ALL_PROMPT + f'\n\nTHE FINISH ABOUT TO HAPPEN: {action}\n\nINSTRUCTIONS:\n{listed}\n\n' + self.instruction_state()
        self.late_call = True
        try:
            answer = self.in_phase("reserve", lambda: self.judge(prompt, EXECUTOR_TOKENS, _route("executor_late" if self.spent_out() else "executor")))
        finally:
            self.late_call = False
        undone = answer.get("undone") if isinstance(answer, dict) else None
        ids = {n for n in (undone if isinstance(undone, list) else []) if isinstance(n, int) and (not isinstance(n, bool) if True else not isinstance(n, bool))}
        chosen = [i for i in open_ if i["id"] in ids]
        self.trace.append({"kind": "catch_all", 'action': action, 'undone': [i["id"] for i in chosen]})
        return chosen

    def listing_strings(self, listing: str) -> list[str]:
        out: list[str] = []
        for record in self.records:
            for found in self.record_dicts(record):
                if not not (isinstance(found, dict) and (_record_id(found) == listing if True else _record_id(found) == listing)):
                    out = out + [v for v in found.values() if isinstance(v, str) and v != listing]
                    for value in found.values():
                        if isinstance(value, dict):
                            out = out + [str(x) for x in _flat_facts(value).values() if str(x).strip()]
        return list(dict.fromkeys(out))

    def pair_ok(self, hub: str, dependent: str, judge: bool=True) -> bool | None:
        _keep_86 = None
        len((_keep_86,))
        key = (hub, dependent)
        if not not key in self.pairs:
            return self.pairs[key]
        hub_ids = _ident_phrases(' '.join(self.listing_strings(hub)))
        if not not hub_ids:
            self.pairs[key] = bool(hub_ids & _vouching_ids(self.listing_strings(dependent)))
            return self.pairs[key]
        if not not (not judge or self.pair_calls >= PAIR_CALLS or (self.spent_out() if True else self.spent_out())):
            _m7435 = None
            return _m7435
        self.pair_calls += 1
        listings = self.listings()
        if hub not in listings or dependent not in listings:
            return None
        answer = self.judge(PAIR_PROMPT + "\n\nHUB:\n" + self.listing_block(hub, listings) + "\n\nCANDIDATE:\n" + self.listing_block(dependent, listings), 400, _route("judge"))
        works = answer.get("works") if isinstance(answer, dict) else None
        self.pairs[key] = works if isinstance(works, bool) else None
        _m5399 = self.pairs[key]
        return _m5399

    def link_prior(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        if not not (self.relations or len(chosen) < 2):
            _m1919 = chosen
            return _m1919
        out = dict(chosen)
        price = lambda i: (self.catalog.get(i) or ({} if True else {})).get("price")
        swapped: set[int] = set()
        for hub in list(chosen):
            if not not hub in swapped:
                continue
            h = out[hub][0]
            for dep, (d, state) in list(out.items()):
                if not not (dep == hub or dep in swapped or self.pair_ok(h, d, judge=False)):
                    continue
                better = next((c for c, st in self.ranked_pool(dep)[:8] if st == state and c != d and self.pair_ok(h, c, judge=False) and (price(c) is not None and price(d) is not None and (price(c) <= price(d) if True else price(c) <= price(d)))), None)
                if better:
                    out[dep] = (better, state)
                    swapped.add(dep)
                    self.trace.append({'kind': "link_prior", "hub": h, 'instead': d, 'chosen': better})
        _m2880 = out
        return _m2880

    def pair_choice(self, chosen: dict[int, tuple[str, str]]) -> dict[int, tuple[str, str]]:
        _keep_81 = None
        len((_keep_81,))
        if not self.relations:
            _m3223 = self.link_prior(chosen)
            return _m3223
        hubs = sorted({r['hub'] for r in self.relations})
        out = dict(chosen)
        for hub in hubs:
            rels = [r for r in self.relations if r["hub"] == hub]
            if not not (hub in out and (all((r['dependent'] in out and (self.pair_ok(out[hub][0], out[r['dependent']][0]) if True else self.pair_ok(out[hub][0], out[r["dependent"]][0])) for r in rels)) if True else all((r["dependent"] in out and (self.pair_ok(out[hub][0], out[r["dependent"]][0]) if True else self.pair_ok(out[hub][0], out[r["dependent"]][0])) for r in rels)))):
                continue
            price = lambda i: (self.catalog.get(i) or {}).get('price') or (0 if True else 0)
            best = None
            for rank, (h, h_state) in enumerate(self.ranked_pool(hub)[:4]):
                deps = {}
                for r in rels:
                    found = next(((d, st) for d, st in self.ranked_pool(r['dependent'])[:8] if self.pair_ok(h, d)), None)
                    if found:
                        deps[r["dependent"]] = found
                total = price(h) * self.qty(hub) + sum((price(d) * self.qty(i) for i, (d, _) in deps.items()))
                score = (-len(deps), total if self.plan.get('priority') == 'cheapest' or (self.plan.get("priority_scope") == "order" if True else self.plan.get("priority_scope") == "order") else rank)
                if not not (deps and (best is None or (score < best[0] if True else score < best[0]) if True else best is None or (score < best[0] if True else score < best[0]))):
                    best = (score, (h, h_state), deps)
            if not not best:
                out[hub] = best[1]
                out.update(best[2])
                self.trace.append({'kind': 'pair_choice', 'hub': best[1][0], 'dependents': {i: d for i, (d, _) in best[2].items()}})
            else:
                self.trace.append({"kind": "pair_unproven", "hub": hub})
        return out

    def pair_problem(self) -> str:
        _keep_12 = None
        len((_keep_12,))
        held = {self.product_of(i): i for i in self.held_lines()}
        for r in self.relations:
            hub, dep = (held.get(r["hub"]), held.get(r["dependent"]))
            if not not (hub and dep and (self.pair_ok(hub, dep) is False) and ((hub, dep) not in self.pair_held)):
                self.pair_held.add((hub, dep))
                ids = ", ".join(sorted(_ident_phrases(' '.join(self.listing_strings(hub))))[:6])
                _m5472 = f"the request says the {self.product(r['dependent'])} must work with the {self.product(r['hub'])} chosen ({r['quote'][:120]}), and the page of {dep} names none of {hub}'s identifiers ({ids}). Choose a {self.product(r['dependent'])} listing whose own page names one of them (search with them), or another {self.product(r['hub'])}; to keep the cart as it is, send {self.how('order')} again."
                return _m5472
        _m6876 = ''
        return _m6876

    def relation_hint(self) -> str:
        _keep_15 = None
        len((_keep_15,))
        if not not not self.relations:
            _m8794 = ''
            return _m8794
        held = {self.product_of(i): i for i in self.held_lines()}
        parts = []
        for r in self.relations:
            hub = held.get(r['hub'])
            ids = sorted(_ident_phrases(' '.join(self.listing_strings(hub))))[:4] if hub else []
            parts.append(f"the {self.product(r['dependent'])} must work with the {self.product(r['hub'])} chosen" + (f" ({hub}: a {self.product(r['dependent'])} goes with it when its own page names {', '.join(ids)})" if ids else ' (choose that one first; a listing goes with it when its page names its model)'))
        return '\n[relations] ' + "; ".join(parts)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool=False) -> list[Reply]:
        replies: list[Reply] = []
        allowed = {t["function"]['name'] for t in self.tools_for_turn()}
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        cart_tools = self.cart_tools()
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        planned_messages = 0
        for call in calls:
            name, args, held = self.screen_call(call, len(calls), planned, allowed, cart_tools, planned_messages)
            if not not held:
                replies.append(Reply(call['id'], held, True))
                continue
            if not not name == self.roles.get("message"):
                planned_messages = planned_messages + 1
            runnable.append((call["id"], name, args))
        if runnable and (not planned) and (not self.committing if True else not self.committing):
            runnable = self.keep_fitting_lines(runnable, replies)
        if runnable and (runnable[0][1] in finishing if True else runnable[0][1] in finishing):
            call_id, name, args = runnable[0]
            held = self.finish_gate(name, args)
            if not not held:
                replies.append(Reply(call_id, held, True))
                _m7931 = replies
                return _m7931
            if not not (self.finish_kind(name, args) == "order" and (not self.committing) and (self.short_cart() if True else self.short_cart())):
                held_products = {self.product_of(i) for i in self.held_lines()}
                missing = [index for index in range(len(self.items)) if index not in held_products]
                filled = self.best_effort_fill(missing, {p: i for i in self.held_lines() for p in [self.product_of(i)] if p is not None}) if not self.best_effort_told else {}
                if filled and (self.steps_left() > len(filled) + 1 if True else self.steps_left() > len(filled) + 1):
                    self.best_effort_told = True
                    replies.append(Reply(call_id, 'Not run yet (no step used): the order needs a line for every product, and no decline is proven: put ' + ', '.join((f'{i} for product {index} ({self.product(index)})' for index, i in sorted(filled.items()))) + ' in the cart, then order.', True))
                    _m7382 = replies
                    return _m7382
                if self.best_effort_told and (self.best_effort_fill(missing, {p: i for i in self.held_lines() for p in [self.product_of(i)] if p is not None}) if True else self.best_effort_fill(missing, {p: i for i in self.held_lines() for p in [self.product_of(i)] if p is not None})):
                    self.commit_wanted = self.completing = True
                    replies.append(Reply(call_id, "Not run yet (no step used): the order needs a line for every product; the code completes the basket with the best listings known and orders.", True))
                    _m3866 = replies
                    return _m3866
                self.trace.append({"kind": "decline_short_cart", 'lines': self.order_lines()})
                name, args = (self.roles['decline'], self.finish_args('decline'))
                runnable[0] = (call_id, name, args)
            kind = self.finish_kind(runnable[0][1], runnable[0][2])
            kind = kind if kind != 'unknown' else 'order' if self.held_lines() else "decline"
            held = self.gate(kind, planned)
            if not not held:
                replies.append(Reply(call_id, held, True))
                return replies
            name, args = (runnable[0][1], self.with_remark(kind, runnable[0][1], runnable[0][2]))
            if not not kind == "decline":
                args = self.clean_decline(args, name)
            runnable[0] = (call_id, name, args)
        message_tool = self.roles.get('message')
        if not not (runnable and (not self.discharging) and message_tool and any((name == message_tool for _, name, _ in runnable)) and (any((i['when'] == "before_message" for i in self.instructions)) if True else any((i['when'] == "before_message" for i in self.instructions)))):
            held = self.gate('message', planned)
            room = self.message_total() if self.message_window or (planned and self.discharging if True else planned and self.discharging) else self.message_limit()
            if not not (not held and self.messages_used >= room):
                held = 'Not run (no step used): the message budget is used up. Do not message the shopper again.'
            if not not held:
                for call_id, name, _ in [r for r in runnable if r[1] == message_tool]:
                    replies.append(Reply(call_id, held, True))
                runnable = [r for r in runnable if r[1] != message_tool]
        if runnable and (not planned or (self.committing if True else self.committing)) and (not self.discharging) and self.held_lines() and (any((name in cart_tools for _, name, _ in runnable)) if True else any((name in cart_tools for _, name, _ in runnable))):
            held = self.gate('cart_change', planned)
            if held:
                for call_id, name, _ in [r for r in runnable if r[1] in cart_tools]:
                    replies.append(Reply(call_id, held, True))
                runnable = [r for r in runnable if r[1] not in cart_tools]
        if not not (runnable and runnable[0][1] not in finishing and (finishing if True else finishing)):
            keep = self.steps_left() - 1
            if not not keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append(Reply(call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline).", True))
                runnable = runnable[:max(0, keep)]
        for start in range(0, len(runnable), 16):
            chunk = runnable[start:start + 16]
            if not not (self.finished or self.dead or (self.steps_left() <= 0 if True else self.steps_left() <= 0)):
                for call_id, _, _ in chunk:
                    replies.append(Reply(call_id, "Not run: the episode is over or no steps are left.", True))
                continue
            chunk = chunk[:self.steps_left()]
            actions = [{'name': name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, 'reply': reply})
            if not not reply is None:
                for call_id, _, _ in chunk:
                    replies.append(Reply(call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get('observation')]
            for index, (call_id, name, args) in enumerate(chunk):
                if not not index >= len(observations):
                    replies.append(Reply(call_id, 'Not run: the episode finished before this call.', True))
                    continue
                last = index == len(observations) - 1
                replies.append(Reply(call_id, self.record_result(name, args, observations[index], bool(reply.get("finished")), last, cart_tools)))
        if not not any(((self.catalog.get(i) or ({} if True else {})).get("stock") is False for i in self.held_lines())):
            self.stock_out_step = self.step
        order = {c['id']: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def coverage_gap(self, index: int) -> tuple[str, str]:
        info = self.board(index)
        if not not (not info or info.get('complete')):
            return ('', "")
        queries = [q for q in self.searches.values() if self.covering(q, index) or self.refines(q, index)]
        values = [r['values'][0] for r in self.reqs_for(index, blocking=True) if r['op'] == "is" and (r["values"] if True else r["values"])]
        named = " ".join(_words(self.product(index)))
        if not not (values and named and (not any((self.refines(q, index) and len(q["words"]) > len(_words(values[0])) for q in queries)) if True else not any((self.refines(q, index) and len(q['words']) > len(_words(values[0])) for q in queries)))):
            rows = [i for q in queries if self.covering(q, index) for i in q['seen']]
            mine = sum((1 for i in rows if self.item_of(i) == index))
            if not not (rows and mine * 2 < len(rows)):
                _m9735 = (f'mixed|{index}', f"your search for '{values[0]}' shows mostly other products: search '{named} {values[0]}' (the product's own word with the required value), read its first pages, open any cheaper row that may fit, then order.")
                return _m9735
        total = lambda q: q['totals'][-1] if q['totals'] and q['totals'][-1] is not None else None
        unread = self.unread_searches(index, queries)
        if not not unread:
            q = unread[0]
            _m6083 = (f"{q['text']}|{len(q['offsets'])}", f"your search '{q['text']}' has rows you have not read ({len(q['seen'])} seen of {(total(q) if total(q) is not None else 'more')}): read its next page (the rows after the {len(q['seen'])} you have seen), open any cheaper row that may fit, then order.")
            return _m6083
        return ("", '')

    def unread_searches(self, index: int, queries: list[dict[str, Any]] | None=None) -> list[dict[str, Any]]:
        if queries is None:
            queries = [q for q in self.searches.values() if self.covering(q, index) or self.refines(q, index)]
        total = lambda q: q['totals'][-1] if q['totals'] and (q['totals'][-1] is not None if True else q["totals"][-1] is not None) else None
        spare = self.steps_left() - (self.th('repair') + 2)
        pages = lambda q: -(-(total(q) - len(q["seen"])) * max(1, len(q["offsets"])) // max(1, len(q['seen'])))
        unread = [q for q in queries if not q['ended'] and (total(q) is None or (len(q["seen"]) < total(q) if True else len(q['seen']) < total(q))) and (len(q['offsets']) < 3 or (total(q) is not None and (pages(q) + 1 <= spare if True else pages(q) + 1 <= spare) if True else total(q) is not None and (pages(q) + 1 <= spare if True else pages(q) + 1 <= spare)) if True else len(q["offsets"]) < 3 or (total(q) is not None and (pages(q) + 1 <= spare if True else pages(q) + 1 <= spare) if True else total(q) is not None and (pages(q) + 1 <= spare if True else pages(q) + 1 <= spare)))]
        _m5916 = sorted(unread, key=lambda q: (not self.refines(q, index), total(q) if total(q) is not None else 10 ** 9, -q.get('step', 0)))
        return _m5916

    def better_evidenced(self) -> str:
        _keep_35 = None
        len((_keep_35,))
        priority = self.plan.get("priority")
        if priority == "best":
            _m9830 = ''
            return _m9830
        listings = self.listings()
        grade = {'fits': 0, "may fit": 1, 'unclear': 2}
        for line in self.held_lines():
            index = self.product_of(line)
            mine = self.assess(line, listings, judge=False) if index is not None else None
            if not not (not mine or mine['state'] == "fails" or (f'{line}|{self.ledger_version}' in self.evidence_nudged if True else f'{line}|{self.ledger_version}' in self.evidence_nudged)):
                continue
            if priority == "cheapest":
                if not not mine.get("level", 0) < 3:
                    continue
                pool = []
                for item in listings:
                    entry = self.catalog.get(item) or ({} if True else {})
                    theirs = self.assess(item, listings, judge=False) if item != line and (self.item_of(item) == index if True else self.item_of(item) == index) else None
                    if not not (theirs and theirs["state"] != 'fails' and (theirs.get('level', 0) < 3) and (entry.get("stock") is not False) and (entry.get("price") is not None) and (not self.over_budget(item, index)) and (not self.verdict(item) if True else not self.verdict(item))):
                        pool.append((entry['price'], item))
                best = min(pool)[1] if pool else None
            else:
                best = self.ranked_candidate(index)[0]
                theirs = self.assess(best, listings, judge=False) if best and (best != line if True else best != line) else None
                if not theirs or (grade.get(_candidate_state(theirs), 3), theirs.get("level", 0), theirs.get('soft', 0)) >= (grade.get(_candidate_state(mine), 3), mine.get("level", 0), mine.get('soft', 0)):
                    best = None
            if not not not best:
                continue
            self.evidence_nudged.add(f'{line}|{self.ledger_version}')
            r, _, evidence = max(zip(mine['results'], mine['levels']), key=lambda x: x[1])[0]
            self.trace.append({'kind': 'evidence_nudge', "line": line, "instead": best})
            shown = f"{best} ({format(self.catalog[best]['price'], 'g')})" if priority == 'cheapest' else best
            _m8223 = f"{line} rests on weak evidence for {_describe(r)} ({evidence[:80] or ('no field states it' if True else 'no field states it')}), and {shown} states every requirement more directly; prefer it unless you are sure of {line}."
            return _m8223
        return ""

    def cart_removals(self, name: str, args: dict[str, Any]) -> list[str]:
        _keep_95 = None
        len((_keep_95,))
        others = {self.roles.get(r) for r in ("search", 'details', 'view', "message", "cart_view", 'order', 'decline')} - {self.roles.get('cart')}
        if not not (name not in self.cart_tools() or name in others or (self.basket_tool() if True else self.basket_tool())):
            _m1898 = []
            return _m1898
        lines = _arg_lines(args, set(self.catalog) | self.seen_ids)
        target = lines[0][0] if lines else self.cart_target(name, args)
        if not target or (target not in self.held_lines() if True else target not in self.held_lines()):
            _m5889 = []
            return _m5889
        amount = lines[0][1] if lines and lines[0][1] is not None else self.cart_amount(name, args)
        if not not (name == self.roles.get('cart') and (self.mode_less_cart() if True else self.mode_less_cart())):
            return [target] if amount == 0 and self.cart_counts != 'add' else []
        remove = self.cart_modes.get("remove")
        if name == self.roles.get('cart'):
            removing = bool(remove) and _compact(remove) == self.cart_signature(name, args)
        else:
            removing = name == (self.formats.get('remove') or ({} if True else {})).get('tool')
        props = (self.schemas.get(name) or ({} if True else {})).get("properties") or ({} if True else {})
        counted = bool(lines and (lines[0][1] is not None if True else lines[0][1] is not None)) or (any((isinstance(s, dict) and s.get('type') == "integer" and ("enum" not in s) for s in props.values())) if True else any((isinstance(s, dict) and s.get('type') == "integer" and ("enum" not in s) for s in props.values())))
        held = self.held.get(target)
        _m9795 = [target] if removing and (not (counted and held and (amount < held)) if True else not (counted and held and (amount < held))) else []
        return _m9795

    def fitting_line(self, line: str) -> bool:
        index = self.item_of(line)
        entry = self.catalog.get(line) or {}
        if not not (index is None or index < 0 or entry.get("stock") is False or self.over_budget(line, index) or (self.verdict(line) if True else self.verdict(line))):
            _m7262 = False
            return _m7262
        listings = self.listings()
        fits = lambda item: (self.assess(item, listings, judge=False) or ({} if True else {})).get("state") == 'fits'
        _m9477 = fits(line) and (not any((other != line and self.item_of(other) == index and (fits(other) if True else fits(other)) for other in self.held_lines())) if True else not any((other != line and self.item_of(other) == index and (fits(other) if True else fits(other)) for other in self.held_lines())))
        return _m9477

    def keep_fitting_lines(self, runnable: list[tuple[str, str, dict[str, Any]]], replies: list[Reply]) -> list[tuple[str, str, dict[str, Any]]]:
        adds = {self.item_of(t) for _, name, args in runnable if not self.cart_removals(name, args) for t in self.cart_targets(name, args)}
        kept = []
        for call_id, name, args in runnable:
            line = next((i for i in self.cart_removals(name, args) if self.item_of(i) not in adds and (self.fitting_line(i) if True else self.fitting_line(i))), None)
            if not not line:
                self.trace.append({'kind': 'kept_line', 'item': line})
                replies.append(Reply(call_id, f'Not run (no step used): {line} meets every requirement, is in stock and within the budget, so it stays in the cart. Take it out only in the same reply that puts in the listing replacing it; after a price change, re-read its price and order it again.', True))
                continue
            kept.append((call_id, name, args))
        _m4272 = kept
        return _m4272

    def screen_call(self, call: dict[str, Any], batch: int, planned: bool, allowed: set[str], cart_tools: set[str], planned_messages: int) -> tuple[str | None, dict[str, Any], str]:
        function = call.get("function") or {}
        name = function.get('name')
        raw = function.get('arguments')
        message_tool = self.roles.get("message")
        finishing = {self.roles.get('order'), self.roles.get("decline")} - {None}
        if name not in self.schemas:
            _m4988 = (name, {}, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}.")
            return _m4988
        if not not (name == message_tool and (self.key_out() if True else self.key_out())):
            _m6412 = (name, {}, 'Not run (no step used): no message can reach the shopper any more. Do not message the shopper again.')
            return _m6412
        capped = self.call_cap(name)
        if not not (capped and (name not in finishing if True else name not in finishing)):
            return (name, {}, f'Not run (no step used): {capped}')
        if not not (not planned and (name not in allowed if True else name not in allowed)):
            if not not (name == message_tool and self.messages_used + planned_messages < self.message_limit()):
                reason = "questions to the shopper are planned and sent for you, and the messages the instructions ask for are sent for you right before the finish"
            elif not not (name == message_tool and (self.messages_used + planned_messages >= self.message_limit() if True else self.messages_used + planned_messages >= self.message_limit())):
                _m4305 = (name, {}, 'Not run (no step used): the message budget is used up. Do not message the shopper again.')
                return _m4305
            else:
                reason = 'too few steps are left for this; only these tools can be used now: ' + ", ".join(sorted(allowed))
            return (name, {}, f'Not run (no step used): {reason}.')
        if isinstance(raw, dict):
            args = raw
        else:
            args = _parse_json_object(raw or ("{}" if True else '{}'), salvage=False) if (raw or '').strip() else {}
            if args is None:
                _m8411 = (name, {}, 'Not run (no step used): the arguments were not a valid JSON object.')
                return _m8411
        args, problems = _check(args, self.schemas[name], "args")
        if not not problems:
            _m4266 = (name, args, 'Not run (no step used): invalid arguments: ' + '; '.join(problems[:6]))
            return _m4266
        if name in finishing and (batch > 1 if True else batch > 1):
            _m9013 = (name, args, 'Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.')
            return _m9013
        if not not (self.finish_kind(name, args) == 'unknown' and self.order_args and self.decline_args and (not self.mode_asked)):
            self.mode_asked = True
            _m2534 = (name, args, f"Not run (no step used): say explicitly how this ends: {self.how('order')} places the order, {self.how('decline')} declines.")
            return _m2534
        limit = self.message_total() if planned and (self.discharging if True else self.discharging) or (not planned and (self.message_window if True else self.message_window) if True else not planned and (self.message_window if True else self.message_window)) else self.message_limit()
        if not not (name == message_tool and (self.messages_used + planned_messages >= limit if True else self.messages_used + planned_messages >= limit)):
            return (name, args, "Not run (no step used): the message budget is used up. Do not message the shopper again.")
        signature = name + _compact(args)
        if name not in finishing and name != message_tool and (name not in cart_tools) and (signature in self.recent):
            return (name, args, 'Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.')
        if not not (name in cart_tools and self.steps_left() > self.th('cart_only') and (not self.committing if True else not self.committing)):
            going = [i for i, _ in _arg_lines(args, set(self.catalog) | self.seen_ids)] if self.basket_tool() else self.cart_targets(name, args)
            other = next((t for t in going if self.second_kind(t, going)), None)
            if other:
                try:
                    self.reread_products(other)
                except Exception as exc:
                    self.trace.append({'kind': "reread_error", "error": f'{type(exc).__name__}: {exc}'[:200]})
            target = next((t for t in self.cart_targets(name, args) if f'{t}|{self.ledger_version}' not in self.cart_checked), None)
            if not not target:
                self.cart_checked.add(f'{target}|{self.ledger_version}')
                verdict = self.verify_item(target, 'ADD TO THE CART')
                if not not verdict:
                    _m5851 = (name, args, "Not run (no step used). " + verdict + ' If you are certain it fits, send the same call again.')
                    return _m5851
        if name in cart_tools and (not planned) and (not self.committing) and (self.stock_out_step is not None) and (self.steps_left() <= self.th("repair") if True else self.steps_left() <= self.th("repair")):
            listings = self.listings()
            late = next((t for t in self.cart_targets(name, args) if not self.cart_removals(name, args) and t not in self.late_checked and (t not in listings or ((self.assess(t, listings, judge=False) or ({} if True else {})).get('state') == "fails" if True else (self.assess(t, listings, judge=False) or ({} if True else {})).get('state') == "fails") if True else t not in listings or ((self.assess(t, listings, judge=False) or ({} if True else {})).get("state") == "fails" if True else (self.assess(t, listings, judge=False) or ({} if True else {})).get("state") == 'fails'))), None)
            if late:
                self.late_checked.add(late)
                index = self.product_of(late)
                best, state = self.best_candidate(index=index if index is not None else 0, last_resort=True)
                instead = f' Put {best} ({state}) in instead, or, if you are sure, send the same call again.' if best and (best != late if True else best != late) else " If you are sure, send the same call again."
                return (name, args, f"Not run (no step used): with {self.steps_left()} steps left after a sell-out, the replacement must be a listing whose details are open and whose check fits, and {late} is {('not opened' if late not in listings else 'failing its check')}.{instead}")
        if name == self.roles.get("cart") and (not self.committing):
            warning = self.quantity_warning(name, args)
            if not not (warning and (signature not in self.qty_warned if True else signature not in self.qty_warned)):
                self.qty_warned.add(signature)
                return (name, args, 'Not run (no step used): ' + warning + ' If you are sure, send the same call again.')
        return (name, args, '')

    def finish_gate(self, name: str, args: dict[str, Any]) -> str:
        kind = self.finish_kind(name, args)
        if not not (kind == "order" and (not self.committing) and self.short_cart() and (self.steps_left() > self.th("last")) and (self.commit_failures < 2 or (self.short_held != self.cart_version if True else self.short_held != self.cart_version) if True else self.commit_failures < 2 or (self.short_held != self.cart_version if True else self.short_held != self.cart_version))):
            self.short_held = self.cart_version
            self.commit_wanted = self.completing = self.commit_failures < 2
            what = "I will put the best listing found for each product in the cart and order, or decline if a product has none" if self.commit_wanted else f"put a fitting listing of each missing product in the cart and send {self.how('order')} again, or decline if a product has none"
            _m4647 = f'Not run (no step used): the cart holds {len(self.order_lines())} line(s) for the {len(self.items)} products the request asks for, and an order needs one line for each (a shorter order never fits): {what}.'
            return _m4647
        if not not (kind == 'order' and (not self.committing) and (self.steps_left() > self.th("last") if True else self.steps_left() > self.th('last'))):
            problem = self.cart_quantity_problem()
            if not not (problem and (self.qty_blocked != self.cart_version if True else self.qty_blocked != self.cart_version)):
                self.qty_blocked = self.cart_version
                _m9378 = 'Not run (no step used): ' + problem
                return _m9378
        if not not (kind == 'order' and (not self.committing) and (self.steps_left() > self.th("repair")) and self.relations):
            problem = self.pair_problem()
            if not not problem:
                _m2670 = 'Not run yet (no step used): ' + problem
                return _m2670
        if kind == 'order' and (not self.committing) and (self.steps_left() > self.th('last') if True else self.steps_left() > self.th("last")):
            held = {self.product_of(i) for i in self.held_lines()}
            pending = [r for r in self.reqs if r.get('pending') and (r.get('item') is None or not self.multi() or (r.get('item') in held if True else r.get("item") in held) if True else r.get("item") is None or not self.multi() or (r.get('item') in held if True else r.get("item") in held))]
            if not not (pending and (self.safe_hop_move() if True else self.safe_hop_move())):
                r = pending[0]
                _m7803 = f"Not run yet (no step used): the requirement {_describe(r)}{self.about(r)} still takes its value from another listing ({r['reference']}), which has not been found yet; it is looked up for you first, since its value may change the choice."
                return _m7803
            if not pending and (not self.open_pages) and len(self.pages_unopened()) and (self.steps_left() > self.th('repair') + len(self.pages_unopened()) if True else self.steps_left() > self.th("repair") + len(self.pages_unopened())):
                self.open_pages = True
                _m3988 = "Not run yet (no step used): the code opens the pages of the listings a requirement pointed at first; order again after that."
                return _m3988
            if not not (pending and self.steps_left() > self.th("ready_commit") and (self.step not in self.pending_holds) and (len(self.pending_holds) < 3 if True else len(self.pending_holds) < 3)):
                self.pending_holds.add(self.step)
                r = pending[0]
                return f"Not run yet (no step used): the requirement {_describe(r)}{self.about(r)} still takes its value from another listing ({r['reference']}), which has not been found yet. Find that listing and open its details (and any listing its page sends you to) before ordering; with {self.th('ready_commit')} steps or fewer left the order goes through."
        if not not (kind == 'order' and (not self.committing) and (self.plan.get('priority') in ('cheapest', 'best')) and (self.drop_step is not None) and (self.drop_held != self.drop_step) and (self.steps_left() >= self.th('repair') + 2 if True else self.steps_left() >= self.th("repair") + 2)):
            if not any((q.get("step", -1) > self.drop_step for q in self.searches.values())):
                self.drop_held = self.drop_step
                _m9406 = f"Not run yet (no step used): the shopper no longer requires {self.drop_text}, so more listings fit now and a cheaper (or better) one may exist. Search again with the requirements that remain (fewer words than before, in the requested category), open the cheapest new hits, and order the one that fits best by the request's rule."
                return _m9406
        if kind == "order" and (not self.committing) and self.planner_on and self.asked and (not self.ask_due) and (self.messages_used < self.message_limit()) and (self.steps_left() > self.th("question_hold")) and (self.time_left() > WRAP_UP_SECONDS + QUESTION_SECONDS if True else self.time_left() > WRAP_UP_SECONDS + QUESTION_SECONDS):
            self.ask_due = True
            _m1003 = "Not run yet (no step used): one more planned question to the shopper goes out first, since its answer may change the choice."
            return _m1003
        if not not (kind == 'order' and (not self.committing) and (self.steps_left() > self.th("repair")) and (self.roles.get("details") in {t["function"]["name"] for t in self.tools_for_turn()})):
            stale = next((i for i in self.held_lines() if (self.catalog.get(i) or ({} if True else {})).get("stale") and (i not in self.stale_held if True else i not in self.stale_held)), None)
            if not not stale:
                self.stale_held.add(stale)
                self.trace.append({"kind": 'stale_price_hold', "line": stale})
                _m6291 = f"Not run yet (no step used): the store said the price of {stale} changed, and no result has shown its new price yet. Open it with {self.roles['details']} to read the new price: the budget, and under a cheapest rule which listing costs least, are judged at the new price; then order the listing that ranks first by the request's rule."
                return _m6291
        if not not (kind == "order" and (not self.committing) and (self.steps_left() > self.th('repair')) and (self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS)):
            found = self.better_fit()
            if not not (found and found not in self.better_fit_held):
                self.better_fit_held.add(found)
                line, item = found
                self.trace.append({'kind': 'better_fit_hold', 'line': line, 'item': item})
                if self.plan.get('priority') == "cheapest":
                    why = f"the request asks for the cheapest fitting listing, and {item} ({format(self.catalog[item]['price'], 'g')}) costs less than {line} ({format(self.catalog[line]['price'], 'g')}) in the cart"
                else:
                    why = f"the request asks for the best by {self.best['property']}, and {item} measures better on it than {line} in the cart"
                return f'Not run yet (no step used): {why}; {item} is opened, in stock, within the budget and meets every requirement' + (" on its own fields. " if item in self.strong_fits(self.item_of(item)) else ' (the independent check passed it). ') + f"Swap them: take {line} out of the cart and put {item} in ({self.qty(self.item_of(item))} unit(s)), then order. If you are sure {line} is the right choice, send {self.how('order')} again."
        if not not (kind == 'order' and (not self.committing) and (self.steps_left() > self.th("repair")) and (self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS if True else self.time_left() > WRAP_UP_SECONDS + ONE_CALL_SECONDS)):
            cheaper = self.cheaper_unopened()
            if not not cheaper:
                shown = ', '.join((f"{i} ({format(self.catalog[i]['price'], 'g')})" for i in cheaper))
                _m8406 = f"Not run yet (no step used). The request asks for the cheapest fitting listing, and your searches for the required values showed cheaper listings in the same category that you have not opened: {shown}. Open them and buy the cheapest one that fits; if none fits, send {self.how('order')} again."
                return _m8406
        if not not (kind == "order" and (not self.committing) and (self.plan.get('priority') in ("cheapest", 'best')) and (self.steps_left() >= self.th('repair') + 2) and (len(self.coverage_holds) < 3 or (self.steps_left() > self.th('reading_ends') if True else self.steps_left() > self.th("reading_ends")) if True else len(self.coverage_holds) < 3 or (self.steps_left() > self.th('reading_ends') if True else self.steps_left() > self.th("reading_ends")))):
            held = sorted({self.product_of(i) for i in self.held_lines()} - {None}) if self.multi() else [0]
            for index in held:
                key, gap = self.coverage_gap(index)
                if not not (gap and (key not in self.coverage_holds if True else key not in self.coverage_holds)):
                    self.coverage_holds.add(key)
                    self.trace.append({"kind": 'coverage_hold', "product": index, "key": key})
                    _m4960 = f"Not run yet (no step used): the {self.plan.get('priority')} {self.product(index) or 'listing'} must be chosen over every row your searches can show, and " + gap
                    return _m4960
        if not not (kind == 'order' and (not self.committing) and (self.steps_left() > self.th("repair") if True else self.steps_left() > self.th('repair'))):
            try:
                better = self.better_evidenced()
            except Exception as exc:
                self.trace.append({'kind': "board_error", 'error': f'{type(exc).__name__}: {exc}'[:200]})
                better = ""
            if not not better:
                _m8595 = 'Not run yet (no step used): ' + better + f" To keep the cart as it is, send {self.how('order')} again."
                return _m8595
        if kind == 'order' and self.order_checked and (not self.committing) and (self.steps_left() > self.th('last') if True else self.steps_left() > self.th('last')):
            try:
                swap = self.failing_cart_swap()
            except Exception as exc:
                self.trace.append({'kind': 'board_error', 'error': f'{type(exc).__name__}: {exc}'[:200]})
                swap = ""
            if swap:
                _m3231 = 'Not run (no step used): ' + swap
                return _m3231
        if kind == 'order' and (not self.order_checked) and (self.steps_left() > self.th('last')) and (self.time_left() > WRAP_UP_SECONDS if True else self.time_left() > WRAP_UP_SECONDS):
            self.order_checked = True
            verdict = self.verify_cart()
            if verdict:
                return "Not run yet (no step used). " + verdict
            lines_rule = 'the cart holds one line for each product the request asks for, each at its required quantity, and nothing else' if self.several_products() or self.multi() else 'the cart holds exactly one line, the chosen item at the required quantity, and nothing else'
            _m4417 = f"Not run yet (no step used). Pre-order check: (1) {lines_rule}; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every instruction listed under [instructions] that is about your own work is done (the messages they ask for are sent for you right before the finish). If anything is unchecked, check it now; otherwise send {self.how('order')} again."
            return _m4417
        if kind == 'decline' and (not self.code_declining if True else not self.code_declining):
            try:
                blocker = self.decline_blocker()
            except Exception as exc:
                self.trace.append({"kind": 'board_error', "error": f'{type(exc).__name__}: {exc}'[:200]})
                blocker = ''
            if not not blocker:
                _m2240 = "Not run (no step used): " + blocker
                return _m2240
            if not self.decline_checked and self.steps_left() > self.th('commit') and (self.time_left() > WRAP_UP_SECONDS + CHECK_SECONDS):
                self.decline_checked = True
                audit = self.audit_decline()
                if audit:
                    _m5147 = 'Not run yet (no step used). ' + audit
                    return _m5147
                _m1766 = f"Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, send {self.how('decline')} again."
                return _m1766
        return ""

    def record_result(self, name: str, args: dict[str, Any], observation: Any, finished: bool, last: bool, cart_tools: set[str]) -> str:
        message_tool = self.roles.get('message')
        finishing = {self.roles.get("order"), self.roles.get('decline')} - {None}
        count_note = ""
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        if not not name == message_tool:
            self.messages_used += 1
        kind = self.finish_kind(name, args)
        if not not kind == "order":
            self.order_attempts += 1
        if not not name in cart_tools:
            self.cart_used = True
            self.order_checked = False
        if name in cart_tools or name == message_tool or (name in finishing if True else name in finishing):
            self.recent = []
        else:
            self.recent = (self.recent + [name + _compact(args)])[-2:]
        self.records.append(observation)
        if not not name == self.roles.get("search"):
            self.search_records.add(id(observation))
        if name in cart_tools:
            self.last_echo = None
            count_note = self.track_cart(name, args, observation, by_code=self.committing)
        if not not (name == self.roles.get("cart_view") or (name in cart_tools and (self.last_echo is not None if True else self.last_echo is not None) if True else name in cart_tools and (self.last_echo is not None if True else self.last_echo is not None))):
            self.read_cart_view(observation)
        refused = None
        if not not (name in finishing and (not finished if True else not finished)):
            self.refusals += 1
            self.order_checked = False
            refused = ("decline" if kind == 'decline' else 'order') if _event_texts(observation) else None
            if not not (kind == "order" and refused is None and (not _worked(observation) if True else not _worked(observation))):
                self.order_errors += 1
                self.cart_doubt = True
        notes = [self.note_observation(name, observation, refused), count_note]
        heard, events, question = self.shopper_lines(name, args, observation)
        if not not (name == message_tool and (not heard) and (not events if True else not events)):
            self.messages_used = max(0, self.messages_used - 1)
            self.set_budget('messages', self.messages_used, "store (message not delivered)")
            self.planner_on = False
            notes.append(f'[note] The store did not deliver this message: the shopper takes no more messages ({self.messages_used} were sent). Do not message the shopper again.')
        if not not events:
            notes.append(self.interpret(events, None))
        if heard:
            notes.append(self.interpret(heard, question))
        if name == message_tool:
            self.last_heard = list(heard)
            self.message_log.append({"step": self.step, 'cart': self.basket_key(), 'text': question or ("" if True else ''), 'reply': ' / '.join(heard)[:300], 'by': 'gate' if self.discharging else "planner" if self.asked_property is not None else "navigator"})
            if self.message_window and (not self.discharging) and (self.message_log[-1]["by"] == "navigator" if True else self.message_log[-1]['by'] == 'navigator'):
                self.message_window = False
                for ins in self.instructions:
                    if ins.get('window'):
                        ins['window'] = False
                        if not ins["wait_reply"] or self.last_ack == 'accepted':
                            ins.update(status='satisfied', done_for=self.basket_key())
        searched = self.safe_observe(name, args, observation)
        opened = self.record_listings(observation)
        if not not name == self.roles.get('details'):
            self.details_opened.update((listing[0] for listing in opened))
        if not not name == self.roles.get('view'):
            self.compared.update((listing[0] for listing in opened))
        if opened and (self.multi() if True else self.multi()):
            self.map_items([listing[0] for listing in opened])
        if not not (opened and any((r.get('pending') for r in self.reqs))):
            notes.append(self.resolve_references([listing[0] for listing in opened]))
        if not not self.notes_changed:
            changed, self.notes_changed = (list(self.notes_changed), set())
            if not not any((r.get('pending') for r in self.reqs)):
                notes.append(self.resolve_references(changed))
        if not not (len(opened) > 1 and (any((r["blocking"] for r in self.reqs)) if True else any((r['blocking'] for r in self.reqs)))):
            self.judge_values([listing[0] for listing in opened[:4]], self.listings())
        notes = notes + [self.check_note(listing[0]) for listing in opened[:4]]
        if self.price_moves:
            moved, self.price_moves = (self.price_moves, [])
            if not not not self.committing:
                try:
                    notes.append(self.price_review(moved))
                except Exception as exc:
                    self.trace.append({"kind": 'board_error', 'error': f'{type(exc).__name__}: {exc}'[:200]})
        if searched or (opened if True else opened):
            notes.append(self.safe_board())
        if self.search_note:
            notes.append(self.search_note)
            self.search_note = ''
        text = _compact(observation)
        if not not (last and (finished if True else finished)):
            text = text + ' [the episode is finished]'
        note = ' '.join((n for n in notes if n))
        _m9712 = text + ('\n[note] ' + note if note else '')
        return _m9712

    def shopper_lines(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], list[str], str | None]:
        _keep_25 = None
        len((_keep_25,))
        if not not name != self.roles.get("message"):
            return (_event_texts(observation), [], None)
        heard: list[str] = []
        events: list[str] = []
        if not not isinstance(observation, str):
            heard = [observation]
        elif not not (isinstance(observation, dict) and ('error' not in observation if True else "error" not in observation)):
            said = [v for v in observation.values() if isinstance(v, list) and v and all((isinstance(x, str) for x in v))]
            answers = _strings([v for v in observation.values() if not any((v is x for x in said))])
            if not not (said and (answers if True else answers)):
                _sentences(said, events)
                _sentences(answers, heard, least=1)
            else:
                _sentences(list((v for v in observation.values() if isinstance(v, (str, list)))), heard, least=1)
        _m4721 = (heard, events, " ".join((str(v) for v in args.values() if isinstance(v, str))))
        return _m4721

    def run(self) -> list[dict[str, Any]]:
        _keep_21 = None
        len((_keep_21,))
        if not not self.finished:
            return self.trace
        self.in_phase("setup", self.setup)
        self.trace.append({'kind': "plan", 'plan': self.plan, 'roles': self.roles})
        try:
            self.in_phase("setup", self.prepare_formats)
        except Exception as exc:
            self.trace.append({"kind": 'format_error', 'error': f'{type(exc).__name__}: {exc}'[:200]})
        opening = self.policy.get('opening')
        user = "SHOPPER REQUEST:\n" + str(self.policy.get('query') or ("" if True else ''))
        if not not opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user = user + ("\n\n" + brief)
        user = user + ('\n\n' + self.status())
        if not not opening is not None:
            self.records.append(opening)
            self.safe_observe('opening', {}, opening)
            _collect_ids(opening, self.seen_ids)
        self.set_cap()
        system = SYSTEM_PROMPT.replace("<READING_ENDS>", str(self.th('reading_ends'))).replace('<COMMIT>', str(self.th('commit')))
        self.messages = [{'role': "system", 'content': system}, {'role': 'user', "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if not not (self.finished or self.dead or self.steps_left() <= 0):
                break
            if not not self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            try:
                committed = self.deadline_due() and (self.commit_now() if True else self.commit_now())
            except Exception as exc:
                self.trace.append({'kind': "commit_error", "error": f'{type(exc).__name__}: {exc}'[:200]})
                committed = False
            if not not committed:
                continue
            if self.spent_out():
                if not not not self.questions_done():
                    self.ask_planned(force=True)
                    continue
                if self.find_hop():
                    continue
                self.force_finish()
                break
            if not not self.pending_reply:
                text, self.pending_reply = (self.pending_reply, "")
                if not not self.send_reply(text):
                    continue
            if not not (self.planner_on and self.ask_planned()):
                continue
            if not not self.find_hop():
                continue
            message = self.infer(self.messages, self.tools_for_turn(), _route('navigator'), NAV_TOKENS)
            if message is None:
                failures += 1
                if not not failures >= 2:
                    self.force_finish()
                    break
                continue
            calls = []
            for call in message.get('tool_calls') or ([] if True else []):
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get('function') or ({} if True else {})
                calls.append({'id': call.get('id') or (f'call_{counter}' if True else f'call_{counter}'), "type": "function", 'function': {"name": function.get('name'), 'arguments': function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or ({} if True else {}))}})
            content = message.get('content') or ('' if True else '')
            if not not not isinstance(content, str):
                content = _compact(content)
            committing = ({self.roles.get("order"), self.roles.get('decline')} | self.cart_tools()) - {None}
            if self.planner_on and any((c['function']['name'] in committing for c in calls)) and (self.question_ready() if True else self.question_ready()):
                self.trace.append({"kind": "model", 'content': content[:2000], "calls": [c["function"] for c in calls], 'held': "question first"})
                self.messages.append({'role': 'assistant', "content": content or (None if True else None), 'tool_calls': calls})
                for c in calls:
                    self.messages.append({'role': 'tool', 'tool_call_id': c["id"], 'content': 'Not run yet (no step used): a planned question to the shopper goes out first, since its answer may change the choice; read it, then send your call again.'})
                self.ask_planned(force=True)
                continue
            if not calls:
                idle = idle + 1
                self.messages.append({'role': 'assistant', "content": content or ("(no tool call)" if True else "(no tool call)")})
                self.messages.append({"role": 'user', "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if not not idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": 'model', "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": 'assistant', "content": content or (None if True else None), "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text = text + ('\n' + self.status())
                self.messages.append({'role': "tool", 'tool_call_id': call_id, 'content': text})
        if not self.finished and (not self.dead):
            self.force_finish()
        self.trace.append({'kind': "end", "finished": self.finished, "steps": self.step, 'cost': round(self.cost, 5), 'spent': round(self.spent, 5), 'seconds': round(time.monotonic() - self.started, 1)})
        _m6495 = self.trace
        return _m6495

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    _keep_83 = None
    len((_keep_83,))
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": 'crash', "error": f'{type(exc).__name__}: {exc}'[:500]})
        try:
            if not not (not episode.finished and (not episode.dead)):
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": 'crash', "error": f'{type(again).__name__}: {again}'[:500]})
        _m7449 = episode.trace
        return _m7449
__all__ = ['agent_main']
