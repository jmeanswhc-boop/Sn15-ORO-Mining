from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
ALL_CALLS = 60
# One scoped key pays for every episode of a run ($5 for a 90-task race) and an exhausted key zeroes every task still
# running, so each episode keeps to a purse: the heavy model only while the spend is under SPEND_SOFT (then the light
# one takes its jobs), the agent's loop stops exploring at SPEND_HARD, and no call starts whose estimated price would
# carry the episode past SPEND_STOP.
SPEND_SOFT = 0.030
SPEND_HARD = 0.044
SPEND_STOP = 0.054
# USD per token (input, output, cached input) for the estimate made before each call
PRICES = {LIGHT: (1e-7, 5e-7, 1e-8), HEAVY: (2e-6, 1e-5, 1e-7), SPARE: (1e-7, 5e-7, 1e-8)}
# The run kills an episode at 300 s. Seconds from the start after which the loop stops exploring and the finish runs
# (CLOCK_LOOP), after which no model call starts (CLOCK_CALLS), and the bound on any one call's wait (CALL_WAIT).
CLOCK_LOOP = 222.0
CLOCK_CALLS = 262.0
CALL_WAIT = 45.0
# waits before resending a failed model call; at most 45 s of an episode go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
ROW_CHARS = 260
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
FIRST_SLOTS = 2             # properties the first message names
ASK_SLOTS = 4               # properties one later message names (one, once the shopper asks for one question at a time). Shoppers differ between worlds: one answers the first property
                            # a list names among those it keeps track of (race 15); another refuses any message naming two
                            # or more of them and asks for one question at a time (race 16, where 8-name lists were turned
                            # down on 136 of 180 first messages). Short lists that drop to one property after the first
                            # such refusal found the most preferences across both kinds in simulation (tests/sim_rules.py).

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter. A keyword search with no rows mostly means one of its words is not on the listings, not that the store lacks the item: drop words (keep the product noun) or browse the item's category.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value as one unbroken phrase counts ("deep teal" meets teal, "extra roomy fit" meets roomy fit). A value that only shares some of its words does not count ("10k white gold" is not "10k gold", "swarovski zirconia" is not "swarovski crystal"), nor one with different words (an abbreviation or code is not the word it stands for). A family pattern a listing states ("fits every AB 35xx model") covers each model inside it. Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. When two attributes name it, the property's own name and the product's name followed by it (such as "style" and "skirt style" on a skirt), the listing meets the requirement when either states the value; a key with any other word in front ("clothing material", "general styles") is another property and decides nothing. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures. A requirement that no listing of that kind shows at all (the shopper may check it themselves) is no reason to reject a listing or to end without a purchase: judge the listings on what they do show.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). A sentence that names another part by its code ("part <code> ...") speaks for that other part, never for the listing it is printed on. When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better. Among listings you have already read that fit, prefer one whose attribute states the value plainly (the value alone, or as one of a list of alternatives such as "silver or black") over one that states it inside a mixed value naming another colour, material or kind ("alpine green and silver").
- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES WHILE YOU WORK
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.
- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end or what to say or offer when nothing fits are not steps: those go in "decline_words" and "quiet_decline".
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks to be told or offered the closest or nearest listing (or some alternative) when nothing fits, null when it says nothing.
"decline_words": the exact words the request tells the agent to say, or to start with, when nothing fits (copied exactly, without quotation marks), else "".
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You pick the questions a shopping agent sends to a shopper. Each question names up to SLOTS properties, the likeliest first (the shopper answers about the first property named among those it keeps track of, so a likely one named late is heard only when nothing before it is tracked). The shopper holds a few firm preferences the request does not state, each on one property of one requested product; in a request for several products some products hold none. A question finds a preference only when it names that property, and only a question naming it counts, so the picks must cover the properties most likely to hold one.

You get the request, the items with what the request already settles for each, and the attribute keys of the listings read so far for each item, with how many of those listings carry each key, how many values it takes, and example values.

A reply of "don't mind" means no preference sits there, a puzzled reply means the shopper never thought about that property, and either way the message is spent; so rank by how likely each property holds the preference, not by how safe it is to ask.

Pick, best first:
1. Each property the request asks to be asked about (listed under THE REQUEST ASKS TO BE ASKED ABOUT), with "asked": true.
2. Hinted properties. Read the request's story sentences (not its rules, budget or limits). A sentence that shows the shopper cares about some property of an item without giving its value points at ONE property of that item:
- someone else will use, wear or receive it: the key for gender or audience; a size that fitted wrongly: size or fit;
- a part of the body or of a picture that will be looked at (face and shoulders, feet, wrist): the key for that part of the product (neckline or collar, toe or heel, strap or case);
- how it looks, what it must go with or must not clash with, a wall or room it must suit: the looks key (colour, pattern or finish) that most of the item's read listings carry, colour when they carry them equally;
- what its label or tag says, what it is made of, what it holds or comes from: material, contents or source;
- where it sits or what it goes into or plugs into (a desk, a wall, a bay, a machine, a port): mounting, form factor, connector or interface;
- what it will be used for (an occasion, a sport, games, travel): the key for that use or occasion;
- how much gets used or how many are needed: pack size, quantity or travel size;
- one particular idea in mind, a look agreed with someone else: the key that says which kind of the product it is (its type, style or sub-kind).
Copy up to 8 of that sentence's own words into "hint". A sentence about a property the request already settles is no hint.
3. Then the properties a buyer of that kind of product chooses between first: which kind of it (its type, style, construction or sub-kind; often a key whose values echo the listings' titles and differ between them), then its form or fit, its size or capacity, its material, colour or pattern, whichever that kind of product is bought by. Among those, prefer keys that most read listings carry and whose values differ; a preference may also sit on a property no listed key names.
Never pick a property the request already settles for that item, a second name for one already picked, identifiers or model names, lists of purchase variants, condition, warranty or delivery.
"label" is the property's name as the key says it, in at most four words: all of a listed key's words except the store's codes or prefixes, the product's own name and a unit ("closure" for "jackets closure type", "display size" for "display size inch"); keep every other word, since a shorter name is another property (a "<part> type" key is not the "<part>" key; "lid colour", not "colour"); no values, no second property.
Pick up to COUNT in all, best first for each item. In a request for several items, list the picks in the order you would ask them, the item whose story hint is strongest first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "the story's words, or an empty string", "asked": true or false}]}"""


LISTEN = """A shopping agent asked a shopper one question about one or more properties of one or more requested products. Classify the answer and return ONE JSON object and nothing else:
{"kind": "need" when the answer states a firm preference (a value, a limit, an exclusion, a closed list, or a value to be read off another listing), "free" when the shopper does not mind or leaves it to the agent, "split" when the answer complains that too much was asked at once or wants one question at a time (in any words: asks to split the questions up, to take them in smaller pieces or one by one, to slow down, or to be asked just the first thing), "lost" when it only says it did not follow or asks for the question again or in simpler words, "other" otherwise,
"item": the number of the item the answer is about (the product it names, else the item of the property it answers), or null,
"matched": the asked property, copied exactly from PROPERTIES ASKED ABOUT, that the answer's property is (another name for the same property counts, as "packaging type" is "container type"), or "" when it is none of them,
"need": for a need, "property: constraint", with the property as the answer names it (else the first one asked about) and the value or limit word for word with its nouns and units ("at least 3 shelves", "not wicker"); for a value read off another listing, everything from the comparison words to the end of that listing's description, word for word ("within 2 cm either way of whatever's on the page for the lamps sold by Shopname that has colour red (or wherever that page points you)"); else "",
"points": true when the value is to be read off another listing, else false,
"budget": a new total budget the answer states, as a number, else null}
Folksy or roundabout wording that still names a value states that value, and an exclusion of everything but one value ("anything apart from V gets binned", "don't show me anything where P comes in other than V") requires that value: write it as "P: V". An answer that will not give a value up or let it move ("V isn't up for trade", "won't budge off V", "V's not up for haggling") wants V: write "P: V"; "not V" is only for an answer that turns V itself down. The property is the one the answer names, even when it names another product ("the X of the <product>"). Use exactly the keys kind, item, matched, need, points and budget."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value as one unbroken phrase counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside; a value that only shares some of the required value's words does not count, "10k white gold" is not "10k gold"; a family pattern such as "every AB 35xx model" covers each model inside it); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails (when both the property's own name and the product's name followed by it are keys, such as "style" and "skirt style" on a skirt, either stating the value is enough; a key with any other word in front, such as "clothing material", is another property); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (when both the property's own name and the product's name followed by it are keys, such as "style" and "skirt style" on a skirt, either stating the value is enough, while a key with any other word in front is another property; any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value as one unbroken phrase counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

LINK = """A shopper's requirement on a product's property takes its value from another listing, and a listing's page can send the reader on to a further listing for that property. Read the TEXT (the requirement itself, or a listing's page) for the PROPERTY and return ONE JSON object and nothing else:
{"next": when the text sends the reader to another listing for this property (a requirement of this kind always does; a page does when it says this property is to be taken from, or gone by, another product), {"shop": the proper name of that listing's shop or seller (the X in "sold by X", "from X", "X's ...", "the X <kind>"; empty only when no seller is named), "kind": its kind of product as written, "has": [{"property": a property given to pick it out, "value": its value as written}, ...]}, else null,
"value": when the text is a page that does not send the reader on for this property, that page's own value for the property (from its words or its attributes), else "",
"partner": when the page says what its value of this property goes together with or works alongside, that other value, else ""}
A page that sends the reader on for this property does so even when its own attributes show a value for it: that value is not the one wanted. A page that only says another version of it exists elsewhere, or says what it is not, does not send the reader on."""


RESTATE = """A shopper's requirement takes its value from another listing, and that listing's page has now given the value. Rewrite the requirement with the value in place of the description of the other listing, keeping the requirement's own comparison ("the same as" becomes the value alone; "within 2 cm either way of" stays; "no higher than" becomes "at most"; "equal to ... or above" becomes "at least"; "other than" stays; "works alongside" or "compatible with" becomes the value alone, since it is the value that goes with that listing). Return ONE JSON object and nothing else: {"requirement": "property: constraint"} with the requirement's own property name."""


CLOSEST = """A shopping agent is about to end without buying anything. That ending is right only when the store provably cannot sell what was asked; otherwise buying the closest listing of each requested item is better. From the records, name for each requested item the listing that comes closest to its requirements in force, and say whether there is such a proof.
Reading: a requirement fails only when the listing's own record states a different value for that property (an attribute whose key names the property, or the title or description when no attribute does); a property the record does not show is unknown, never failed; a value meets a requirement when it equals it or contains it as one unbroken phrase; a family pattern such as "every AB 35xx model" covers each model inside it; a ROW record shows only part of the listing. The closest listing has the fewest failed requirements, then the most met ones, is buyable and keeps the cart within the budget; for a cheapest pick, the cheapest of those.
A proof is one of: "contradiction" when the request's own requirements for one item cannot all hold at once; "budget" when even the cheapest listing of some item that meets its requirements costs more than the budget leaves; "absent" when the store shows no listing of that kind of product at all.
Return ONE JSON object and nothing else: {"items": [{"item": item number, "id": the closest listing's id, "fails": [{"requirement": a requirement its record contradicts, "record_says": the record's own words that contradict it, copied exactly}], "unknown": [every requirement in force its record does not show, copied as the requirements list it]}], "proof": null or {"kind": "contradiction" | "budget" | "absent", "item": item number, "why": "one sentence naming the prices or values"}}"""


MATCH = """Does this listing's page state, about the listing itself, each of the VALUES below? A value counts when the page states the same thing in other words or units ("2000 gb" is "2 tb", "0 inches" is "a display size of 0"), or a value that contains it ("1 gb" in "1gb & under"). A sentence saying the listing is not that, or that a version with that value exists elsewhere, does not count. Return ONE JSON object and nothing else: {"all": true or false}"""


TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value


def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = text.find("{"), text.rfind("}")
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def _words(key: str) -> str:
    # camel case splits ("lidColour" reads "lid colour"), but a unit after a figure stays whole ("250mL", "5GHz")
    key = re.sub(r"(\d*)([a-z]+)(?=[A-Z])", lambda m: m.group(0) if m.group(1) else m.group(0) + " ", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _same_prop(a: str, b: str) -> bool:
    """Two property names mean the same property: the same words, or one of two or more words inside the other
    ("system memory" in "system memory new"); a one-word name inside a longer one ("colour" in "lid colour") does not."""
    return _same_stems(_stem(a), _stem(b))


def _same_stems(x: frozenset[str], y: frozenset[str]) -> bool:
    if not x or not y:
        return False
    if x == y:
        return True
    small, large = (x, y) if len(x) <= len(y) else (y, x)
    return len(small) >= 2 and small <= large


def _plain_label(text: str) -> str:
    """A property's name fit for a question: one property (cut at "and", "or", "such as", a slash or a bracket), with
    no trailing numbering or "new" a store key may carry ("stone type 2", "graphics memory new")."""
    label = re.split(r"\s+(?:and|or|such as|for example|e\.g\.)\s+|[/,;(:]", text)[0]
    label = re.sub(r"(?:\s+(?:new|\d+))+$", "", _words(label)).strip().rstrip("?.")
    return label


def _squash(text: str) -> str:
    """Plain words with a number joined to the unit after it ("12 gb" and "12GB" both read "12gb"); a thousands
    separator is dropped first ("1,250" reads "1250": race 13's pointer walker rejected the real chain page over it)."""
    text = re.sub(r"(?<![\d,])(\d{1,3})[,\u2009](?=\d{3}(?![\d,]))", r"\1", str(text))
    # a decimal stays one figure ("1.5 kg" reads "1.5kg", "5.8GHz" reads "5.8ghz"): race 14's walker could not match
    # a decimal figure on a page, and took the digit after a decimal point for a whole figure
    text = re.sub(r"(\d)\.(?=\d)", lambda m: m.group(1) + "\u00b7", text)
    plain = " ".join(re.findall(r"[a-z0-9\u00b7]+", _words(text)))
    return re.sub(r"(\d)\s+(?=[a-z])", r"\1", plain).replace("\u00b7", ".")


def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _fits_kind(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value toward its JSON schema; list what cannot be fixed."""
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and not any(_fits_kind(value, k) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
                break
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
                break
            if kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
                break
        if not any(_fits_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f"{where}: allowed values are {options}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [_shape(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                faults.append(f"{where}.{name} is not an allowed argument")
        for name in schema.get("required") or []:
            if name not in value:
                faults.append(f"{where}.{name} is required")
        value = {k: _shape(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
    return value


def _items(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def _id_like(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    """The key that identifies rows: string values with a digit and no spaces, unique per row, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_id_like(v, "id" in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _in_rows(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether a list in the result holds a record carrying this id (a cart listing still showing the line)."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_in_rows(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_in_rows(v, ident, depth + 1) for v in node.values())
    return False


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


# ---------- listings sent as text ----------
# A store may send each listing as one string instead of a record: fields "key: value" joined by " | " (or one per
# line), the attribute block inside one field as "name=value; name=value". Every check here reads records, so such a
# string is turned into the record it spells out before anything else sees the result; other strings stay as they are.
_FIELD_START = re.compile(r"(?:^|(?<=\s\|\s))([a-z][a-z0-9_]{0,30}(?:[ _][a-z0-9]{1,15}){0,3}):(?=\s|$)")
_LINE_START = re.compile(r"^([A-Za-z][A-Za-z0-9_]{0,30}(?:[ _][A-Za-z0-9]{1,15}){0,3}):(?=\s|$)", re.M)
_PAIR_START = re.compile(r"(?:^|(?<=;\s))([A-Za-z][\w .&/()+-]{0,40}?)=")


def _scalar(text: str) -> Any:
    """A field's text as the value a record would hold: flags, nothing, whole numbers (not codes with leading zeros)
    and decimals; anything else stays text."""
    word = text.strip()
    if word in ("True", "true", "False", "false"):
        return word.lower() == "true"
    if word in ("None", "null", ""):
        return None if word else ""
    if re.fullmatch(r"-?(?:0|[1-9]\d{0,14})", word):
        return int(word)
    if re.fullmatch(r"-?\d{1,12}\.\d{1,6}", word):
        return float(word)
    return word


def _pairs_of(text: str) -> dict[str, Any] | None:
    """'a=1; b=two words; c=' as a dict, when the text opens with a pair and holds at least two (one pair counts when
    the whole text is that pair)."""
    starts = list(_PAIR_START.finditer(text.strip()))
    body = text.strip()
    if not starts or starts[0].start() != 0:
        return None
    out: dict[str, Any] = {}
    for at, found in enumerate(starts):
        end = starts[at + 1].start() if at + 1 < len(starts) else len(body)
        name = found.group(1).strip()
        if name in out:
            return None
        out[name] = body[found.end():end].rstrip().rstrip(";").strip()
    if len(out) == 1 and (";" in body or not re.fullmatch(r"[a-z][a-z0-9 _]{0,40}", next(iter(out)))):
        return None
    return out


def _text_listing(text: Any) -> dict[str, Any] | None:
    """The record a listing's text spells out, or None when the string is no listing (a reply, a note, a sentence
    with a colon in it): three or more fields, one of them an id-like value."""
    if not isinstance(text, str) or len(text) > 60000 or ":" not in text:
        return None
    if text.count(" | ") >= 2:
        starts = list(_FIELD_START.finditer(text))
    elif text.count("\n") >= 2:
        starts = list(_LINE_START.finditer(text))
    else:
        return None
    if len(starts) < 3 or text[:starts[0].start()].strip():
        return None
    record: dict[str, Any] = {}
    for at, found in enumerate(starts):
        end = starts[at + 1].start() if at + 1 < len(starts) else len(text)
        key = found.group(1)
        if key in record:
            return None
        value = text[found.end():end].strip()
        value = value[:-1].rstrip() if value.endswith("|") else value
        named = set(re.split(r"[\s_]+", key.lower()))
        block = _pairs_of(value) if "=" in value and not named & {"title", "name", "heading"} else None
        if named & {"id", "ref", "sku", "code", "number", "no"} and not block:
            record[key] = value if value not in ("None", "null") else None   # an id stays text ("100231", "007")
        else:
            record[key] = block if block else _scalar(value)
    if not any(_id_like(v) for v in record.values() if isinstance(v, str)):
        return None
    return record


def _untext(node: Any, depth: int = 0) -> Any:
    """A tool result with each listing sent as text read into its record (a result that is only a wrapper around one
    listing becomes that record); everything else as it was."""
    if depth > 7:
        return node
    if isinstance(node, str):
        record = _text_listing(node)
        return node if record is None else record
    if isinstance(node, list):
        return [_untext(v, depth + 1) for v in node]
    if isinstance(node, dict):
        if len(node) == 1:
            only = next(iter(node.values()))
            record = _text_listing(only)
            if record is not None:
                return record
        if len(node) >= 3 and any(_id_like(v) for v in node.values() if isinstance(v, str)):
            # already a record (a cart line, a structured row): its own texts are left alone, lists inside it are read
            return {k: v if isinstance(v, str) else _untext(v, depth + 1) for k, v in node.items()}
        return {k: _untext(v, depth + 1) for k, v in node.items()}
    return node


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Kyo:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.view = environment["policy_view"]
        if self.view.get("opening") is not None:
            self.view["opening"] = _untext(self.view["opening"])
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 30
        self.t0 = time.monotonic()
        self.cached_chars: dict[str, int] = {}   # per model: characters of the last call's prompt (its cached part next time)
        self.step = 0
        self.finished = bool(self.view.get("finished"))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.asked_props: dict[int, set[frozenset[str]]] = {}   # per item: properties asked about (stems)
        self.found_on: dict[int, int] = {}                      # per item: preferences the answers revealed
        self.asked_msgs: dict[int, int] = {}                    # per item: questions sent about it
        self.code_hits = 0                                      # word searches for a code that found rows
        self.code_misses = 0                                    # ... and that found none
        self.compat_holds = 0                                   # checkouts held for model codes no line pair shares
        self.parts_looked = 0                                   # code-made searches for parts a cart line's page names
        self.parts_seen: set[str] = set()
        self.compat_marks: dict[str, int] = {}
        self.model_memo: dict[str, tuple[Any, dict[str, set[str]]]] = {}  # model names and part series per listing
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ""
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.rescued = -1                    # ledger version at the last rescue review
        self.heavy_checks = 0
        self.check_said: dict[str, str] = {}
        self.check_repeats: dict[str, int] = {}
        self.decline_proven = -1
        self.link_chains: dict[str, list[str]] = {}   # requirement -> listings opened along its chain
        self.link_walked: set[str] = set()
        self.link_judged: dict[str, bool] = {}
        self.link_searched: set[str] = set()
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.decline_why = ""
        self.closest_named: tuple[str, str, str] | None = None
        self.plan_rank: dict[int, int] = {}
        self.cart_version = 0
        self.cart_sig = ""
        self.repicked = ""
        self.change_held = ""
        self.clash_holds = 0
        self.clash_marks: dict[str, int] = {}
        self.loop_cap, self.call_cap = LOOP_CALLS, ALL_CALLS
        self.spend_hard, self.spend_stop = SPEND_HARD, SPEND_STOP

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        """Tools that show a listing's page: the offer view when the store has one apart from the attributes view
        (an attributes list alone is not the page), else the single-listing view; and the side-by-side view."""
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        """The single-listing tool that shows attributes: one that has shown them, else the brief's attributes view."""
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        """Whether the side-by-side view shows attributes: True or False once seen, None before."""
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        """The side-by-side view is no way to read attributes: it showed none, or, not tried yet, the store has a separate
        attributes view (one that showed them, or one apart from the offer view)."""
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def seconds(self) -> float:
        return time.monotonic() - self.t0

    def spent(self) -> bool:
        """No model call may start any more: the purse or the clock is used up."""
        return self.cost >= self.spend_stop or self.seconds() >= CLOCK_CALLS

    def seat(self, careful: bool = True) -> list[str]:
        """Models for a judgement, first choice first: the heavy one for a careful judgement while the purse allows."""
        return [HEAVY, LIGHT] if careful and self.cost < SPEND_SOFT else [LIGHT, HEAVY]

    def guess_price(self, model: str, messages: list[dict[str, Any]], max_tokens: int) -> float:
        """A call's likely price: its prompt at about four characters a token (the part an earlier call to the same
        model already sent is billed at the cached rate) and its likely reply."""
        size = sum(len(_dump(m.get("content"))) for m in messages)
        reused = min(size, self.cached_chars.get(model, 0)) if model.startswith("anthropic/") and len(messages) > 1 else 0
        fresh_rate, out_rate, reuse_rate = PRICES.get(model, PRICES[HEAVY])
        # replies (with the model's own reasoning) run to about a thousand tokens whatever room they are given
        return (size - reused) / 4 * fresh_rate + reused / 4 * reuse_rate + min(max_tokens / 2, 1000) * out_rate

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.spent():
                return None
            if self.cost + self.guess_price(model, messages, max_tokens) > self.spend_stop:
                self.trace.append({"kind": "model_skipped", "model": model, "cost": round(self.cost, 4)})
                continue
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL \
                        or self.seconds() + WAITS[tries] >= CLOCK_CALLS:
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference call: the reply message, or None and the failure's status (0 for a reply without content)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        # a reply arrives in one piece, so the read timeout bounds the call: it may not run into the run's 300 s kill
        _llm.timeout = max(4.0, min(CALL_WAIT, 288.0 - self.seconds()))
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            else:
                # no price reported: the estimate stands in, so the purse still counts the call
                self.cost += self.guess_price(model, messages, max_tokens)
            self.cached_chars[model] = sum(len(_dump(m.get("content"))) for m in messages)
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error carried inside a 200 reply, with the provider's own code
            status = _count(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.trace.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        if name == (getattr(self, "roles", None) or {}).get("buy") and isinstance(getattr(self, "brief", None), dict) and self.is_buy(name, args):
            self.buy_tried = True
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data and name != self.roles.get("ask"):
                    # listings sent as text are read into records before the ledger, the checks or the model see them
                    # (a shopper's reply stays as it was said)
                    data["observation"] = _untext(data["observation"])
                    self.note_search(name, args, data["observation"])
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.finished = True
                    return None
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.step += 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    # ---------- reading the request ----------
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            self.items.append(Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts))
        if not self.items:
            self.items = [Item("the requested product", "", 1, [])]
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            # only a step at the order itself keeps messages back from the interview: what to say when nothing fits goes
            # into the ending's own text (race 13: a message kept for it cost the second question in 70 of 180 episodes,
            # and a "may I offer the closest listing" message never helped)
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, self.seat(), 2000) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get("describe_other"))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + " (that listing: " + "; ".join(moved) + ")"
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({"kind": "unmix", "item": number, "moved": moved})

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in item.musts + item.told if m not in item.dropped]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        pick = brief.get("pick")
        if pick == "cheapest":
            parts.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace("  ", " "))
        else:
            parts.append("pick: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        for wish in _items(brief.get("wishes")):
            parts.append(f"wish (tie-breaker only): {wish}")
        for rule in _items(brief.get("rules")):
            parts.append(f"rule from the request: {rule}")
        if brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        elif brief.get("quiet_decline") is False:
            parts.append("if nothing fits: the ending's text names the closest listing (its id and title) and what it misses")
        if str(brief.get("decline_words") or "").strip():
            parts.append(f"if nothing fits: the ending's text starts with the words {str(brief['decline_words']).strip()!r}")
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo writes it into the ending's text)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def status(self) -> str:
        left = self.left()
        text = f"[Kyo] step {self.step} of {self.max_steps} used, {left} left."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if not self.codes_searchable():
            text += f"\n[search] {self.code_misses} searches for a code found nothing: this store's search does not read codes. Search by kind and brand words or page through the category, and read the codes in the rows."
        compat = self.compat_note() if len(self.items) > 1 else ""
        if compat:
            text += "\n" + compat
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(_id_like(r.get(self.id_key), True) for r in rows):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    self.rows[ident].update(node)
                    if _fact_block(node):
                        self.full[ident] = dict(self.rows[ident])
                    elif ident in self.full:
                        self.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)

        walk(observation, 0)

    def labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get("enum") or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)

        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """(category, key name, listings carrying it, sample values), most carried first per category."""
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or "listings") if cat_key else "listings"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12:
                    continue
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, slot))
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def item_groups(self) -> list[list[str]]:
        """Listing ids per category, keeping the categories that match a requested item (largest first)."""
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or "") if cat_key else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """The listing records a result carries: the result itself, or the rows of its lists (two levels deep)."""
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def note_view(self, name: str, observation: Any) -> None:
        """Which fields each viewing tool shows, and which a listing has shown since the last change: a view whose
        fields were all shown already is not run again."""
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    # ---------- result pages ----------
    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """Which keys of a tool's row-list result hold the rows, the match count, the next-page value (and the parameter
        taking it), a row's price and the flag saying it can be bought: named once per tool by the light model, checked against the result."""
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        shape = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and (isinstance(observation[shape["count"]], bool) or not isinstance(observation[shape["count"]], int)):
            shape["count"] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape["price"]
        self.stock_key = self.stock_key or shape["stock"]
        self.trace.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the parameter that asks for the next page, its value or None on the last page)
        by the tool's named result shape; None when the result has no such parts."""
        shape = self.read_shape(name, observation)
        if not shape or not shape["count"] or not shape["next"] or not shape["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape["rows"]), observation.get(shape["count"]), observation.get(shape["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape["next_arg"], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.paged_obs = []
        pick = self.brief.get("pick")
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.price_sorted(observation):
            return ""
        # results come in relevance order, not by price, and stores cap the reported count: the next two pages are
        # read whatever the count says, as long as the episode's paging allowance and steps last
        pages = min(2, -(-(count - shown) // shown))
        # a rest of more than three pages (a whole category browsed without a filter: race 13 showed 800-1000 rows ten
        # at a time) is not read page by page; the agent narrows the search instead
        if self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12 or count > 4 * shown:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value), browse the category with a search's highest-price argument when one exists, or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.act(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 9 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under its plain product name alone (no values or adjectives: "
                + "; ".join(f"'{it.name}'" for it in self.items) + "), then other names for it (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def budget_over(self) -> str:
        """Why the cart breaks the budget by the prices Kyo has read, or "" when it does not (or when a price is
        unknown, or prices and budget plainly use different units)."""
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool) or self.budget <= 0:
            return ""
        view = _dump(self.cart_view) if self.cart_view is not None and self.cart_seen_at >= self.cart_at else ""
        # an id inside a longer id ("AB-12" in "AB-123") is not a cart line
        ids = [i for i in self.cart_ids() if not view or re.search(r'"' + re.escape(i) + r'"', view)]
        lines = []
        for ident in ids:
            price = self.price_of(ident)
            if price is None:
                return ""
            lines.append((ident, float(price), _count(self.cart_lines.get(ident)) or 1))
        if not lines:
            return ""
        each = self.brief.get("budget_scope") == "each"
        total = sum(p * u for _, p, u in lines)
        if total >= 20 * float(self.budget):
            # prices and budget in different units (cents against dollars): the model's check judges that cart
            return ""
        if each:
            over = [f"{i} at {p:g}" for i, p, _ in lines if p > float(self.budget) + 1e-9]
            return f"{', '.join(over)} costs more than the budget of {self.budget:g} for each item" if over else ""
        if total <= float(self.budget) + 1e-9:
            return ""
        return f"the cart totals {total:g} (price times units over its lines), over the budget of {self.budget:g}"

    def budget_hold(self) -> str:
        """A checkout of a cart over the budget, held in code (local run: a line bought in place of a sold-out one
        cost twice the budget, after the purse had used up the model's check). Holds a cart twice at most."""
        if self.left() < 3:
            return ""
        mark = self.cart_mark() + f"#{self.version}#budget"
        if self.clash_marks.get(mark, 0) >= 2:
            return ""
        why = self.budget_over()
        if not why:
            return ""
        self.clash_marks[mark] = self.clash_marks.get(mark, 0) + 1
        self.trace.append({"kind": "budget_hold", "why": why[:120]})
        return f"Kyo holds this checkout: {why}. Replace a line with a fitting listing that keeps the order inside the budget."

    def budget_rescue(self) -> str:
        """Before a code-made checkout of a cart over the budget: its costliest line is replaced by the code's own pick
        of that item inside what the budget leaves (local run: with the purse spent, the forced finish bought a
        replacement line at twice the budget). Once per episode; "" when nothing was ordered."""
        if getattr(self, "budget_redone", False) or not self.budget_over():
            return ""
        self.budget_redone = True
        cat_key = self.category_key()
        plan: dict[int, str] = {}
        for ident in self.cart_ids():
            number = self.item_for_label(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key and len(self.items) > 1 else 1
            if number and number not in plan:
                plan[number] = ident
        if not plan:
            return ""
        worst = max(plan, key=lambda n: (self.price_of(plan[n]) or 0.0) * self.items[n - 1].qty)
        others = {n: i for n, i in plan.items() if n != worst}
        pick = self.code_pick(worst, others)
        if not pick or pick == plan[worst]:
            return ""
        lines = {i: self.items[n - 1].qty for n, i in others.items()}
        lines[pick] = self.items[worst - 1].qty
        self.trace.append({"kind": "budget_rescue", "out": plan[worst], "in": pick})
        return self.order_lines(lines)

    def value_clash(self) -> str:
        """A cart line whose own attribute for a required property states another value: the attribute named for the
        property decides, whatever the title says (race 13: lines bought on a title's word while the attribute named
        another value). Checked in code before the model's check; holds one cart at most twice."""
        if self.clash_holds >= 3 or self.left() < len(self.items) + 4:
            return ""
        mark = self.cart_mark() + f"#{self.version}"
        seen = self.clash_marks.get(mark, 0)
        if seen >= 2:
            return ""
        self.clash_marks[mark] = seen + 1
        cat_key = self.category_key()
        pointers = {need for _, need in self.refs}
        found = []
        for ident in self.cart_ids():
            block = _fact_block(self.full.get(ident) or {}) or {}
            if not block:
                continue
            number = self.item_for_label(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key and len(self.items) > 1 else 1
            if not number:
                continue
            item = self.items[number - 1]
            for need in item.musts + item.told:
                if need in item.dropped or need in pointers or ":" not in need:
                    continue
                prop, value = (part.strip() for part in need.split(":", 1))
                wanted = _squash(value)
                own = _stem(f"{item.name} {item.shelf or ''}")
                keys = [k for k in block if _stem(k) == _stem(prop) or (_stem(k) - own == _stem(prop) and _stem(k) & own)]
                limit = self.limit_of(value)
                if limit and keys:
                    # a limit on a figure: the attribute's figure in the same unit word must keep inside it
                    figures = [self.figure_of(str(block[k]), limit[2]) for k in keys]
                    figures = [f for f in figures if f is not None]
                    if figures and not any((f >= limit[1] - 1e-9) if limit[0] == "min" else (f <= limit[1] + 1e-9) for f in figures):
                        found.append(f"{ident}'s own attribute {keys[0]}={str(block[keys[0]])[:60]!r} is outside the requirement '{need}'")
                    continue
                plain = bool(wanted) and not re.search(r"\d", wanted) and len(wanted.split()) <= 3 \
                    and not re.search(r"\b(?:not|no|non|without|except|other|unlike|same|match\w*|any|either|or|than|like)\b", wanted)
                if not plain and not limit:
                    # exclusions, comparisons and lists are left to the model's check
                    continue
                # a requirement the category label states is met by the category ("ready to drink" in "UHT Milk (Ready to
                # Drink)"), whatever an attribute says
                label = str((self.rows.get(ident) or self.full.get(ident) or {}).get(cat_key) or "") if cat_key else ""
                if plain and label and self.states(label, wanted):
                    continue
                if not keys:
                    # no attribute for the property while two or more other read listings of its kind carry one, and
                    # nothing else in the record states the value: the line is unproven (race 14: a line meeting a
                    # volume limit was swapped for one with no volume attribute at all)
                    carriers = [i for i in self.item_kind_rows(number) if i != ident and any(
                        _stem(k) == _stem(prop) for k in (_fact_block(self.full.get(i) or {}) or {}))]
                    text = f" {_squash(_dump(self.full.get(ident) or {}))} "
                    # a figure in free text is not the property's (a yield in a description is not the attribute):
                    # a limit needs the attribute; a plain word value may be stated by the title or description
                    stated = bool(plain and self.states(text, wanted))
                    if len(carriers) >= 2 and not stated:
                        found.append(f"{ident} has no attribute for {prop} (other listings of its kind show one, such as "
                                     f"{', '.join(carriers[:3])}), so nothing shows it meets '{need}'")
                    continue
                if any(self.states(str(block[k]), wanted) for k in keys):
                    continue
                found.append(f"{ident}'s own attribute {keys[0]}={str(block[keys[0]])[:60]!r} does not state the requirement '{need}'")
        if not found:
            return ""
        self.clash_holds += 1
        self.trace.append({"kind": "value_clash", "found": found[:3], "again": seen})
        if seen:
            return ("Kyo holds this checkout once more: " + "; ".join(found[:3]) + ". The key named for the property (or the product's name"
                    " followed by it) decides; a key with any other word in front is another property and cannot make up for it. Unless the"
                    " two values mean the same thing, replace the line with a listing whose own key states the value.")
        return ("; ".join(found[:3]) + " in those words. An attribute named for the property decides: unless that value means the same thing"
                " (another spelling or a plain synonym), the line fails, so replace it with a listing whose attribute states the wanted value"
                " (search with the value, open the candidates); if it does mean the same, check out again.")

    def compat_hold(self) -> str:
        """A cart whose lines name model codes none of which all of them fit (race 14: Cobalt bought 44 of 51 such
        carts with a part that named another device, or named the device only in a sentence about another part).
        Holds one cart at most twice, three carts in all, and offers the best basket the codes allow."""
        if len(self.items) < 2 or self.compat_holds >= 3 or self.left() < len(self.items) + 4:
            return ""
        plan: dict[int, str] = {}
        cat_key = self.category_key()
        for ident in self.cart_ids():
            number = self.item_for_label(str((self.rows.get(ident) or {}).get(cat_key) or "")) if cat_key else None
            if number and number not in plan:
                plan[number] = ident
        if len(plan) < 2:
            return ""
        mark = self.cart_mark() + f"#{self.version}"
        if self.compat_marks.get(mark, 0) >= 2:
            return ""
        why = self.compat_conflict(plan)
        if not why:
            return ""
        self.compat_marks[mark] = self.compat_marks.get(mark, 0) + 1
        self.compat_holds += 1
        baskets = self.compat_baskets()
        offer = ""
        if baskets:
            total, basket, code = baskets[0]
            offer = f" Lines that all name {code}: " + ", ".join(f"item {n} {i}" for n, i in sorted(basket.items())) + f" (total {total:g}); check each one's requirements on its page."
        self.trace.append({"kind": "compat_hold", "why": why[:200], "offer": offer[:200]})
        return ("Kyo holds this checkout: " + why + ". Items that must work together need one model code that every part's own"
                " page names for itself; a sentence about another part (\"part <code> ...\") speaks for that part only." + offer
                + " Replace the line that names no common code, or check out again if the request does not ask them to work together.")

    @staticmethod
    def limit_of(value: str) -> tuple[str, float, str] | None:
        """("min" or "max", figure, unit word) of a plain one-sided limit ("at least 320 ml", "no more than 4 ohms",
        "15.6 inches or more"), else None. Ranges, ties to another listing and exclusions are left to the model."""
        low = " ".join(str(value).lower().split())
        low = re.sub(r"(?<=\d),(?=\d{3}\b)", "", low)
        if re.search(r"\bof\b|whatever|listing|page|between|either|unlike|other than|\bnot\b(?! (?:more|less) than)|\bwithin\b|\binside\b", low):
            return None
        m = re.search(r"(\d+(?:\.\d+)?)\s*([a-z\"']*)", low)
        if not m:
            return None
        figure, unit = float(m.group(1)), m.group(2)
        # the upper bounds first, and no bare "min"/"max": a unit such as minutes must not read as a bound
        if re.search(r"\b(?:at most|maximum|no more than|not more than|up to)\b|\bor (?:less|below|lower|under|smaller)\b", low):
            return "max", figure, unit
        if re.search(r"\b(?:at least|minimum|no less than|not less than)\b|\bor (?:more|above|higher|greater|bigger|larger)\b", low):
            return "min", figure, unit
        return None

    @staticmethod
    def figure_of(text: str, unit: str) -> float | None:
        """The first figure in the text followed by the same unit word (or any figure when the limit has no unit)."""
        low = str(text).lower().replace(",", "")
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([a-z\"']*)", low):
            if not unit or m.group(2) == unit or (m.group(2) and (m.group(2).rstrip("s") == unit.rstrip("s"))):
                return float(m.group(1))
        return None

    @staticmethod
    def holds_words(value: str, wanted: str) -> bool:
        """Every wanted word is a word of the value, singular or plural only (no prefixes: "tan" is not "tangerine")."""
        have = {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _squash(value).split()}
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _squash(wanted).split()]
        return bool(words) and all(w in have for w in words)

    @staticmethod
    def states(value: str, wanted: str) -> bool:
        """The attribute value states the wanted words: each wanted word among the value's words, singular or plural,
        or one letter off for words of four letters or more ("grey" for "gray", "knit" for "knits")."""
        have = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _squash(value).split()]
        def near(a: str, b: str) -> bool:
            if a == b:
                return True
            # an initial or a shortened form ("l" for large, "zip" for zipper)
            if (len(b) <= 2 and a.startswith(b)) or (len(a) <= 2 and b.startswith(a)) or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))):
                return True
            if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 1:
                return False
            if len(a) == len(b):
                return sum(x != y for x, y in zip(a, b)) <= 1
            short, long = (a, b) if len(a) < len(b) else (b, a)
            return any(long[:i] + long[i + 1:] == short for i in range(len(long)))
        words = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in wanted.split()]
        return bool(words) and all(any(near(w, h) for h in have) for w in words)

    def req_sig(self) -> str:
        """The requirements and the budget as they stand: what a re-pick or a held checkout reacts to (a re-read price or
        stock figure moves the general version counter too, and must not hold a checkout again and again)."""
        return _dump([[i.musts, i.told, i.dropped] for i in self.items] + [self.budget])

    def change_hold(self) -> str:
        """Once per change: a checkout of a cart set before the requirements or the budget changed, under a cheapest,
        most or least pick, is held so the pick is redone for the rules now in force (race 13: Cobalt re-ordered the
        cart chosen for a requirement the shopper had just dropped)."""
        sig = self.req_sig()
        if self.brief.get("pick") not in ("cheapest", "most", "least") or not self.cart_touched or sig == self.cart_sig \
                or self.change_held == sig or self.left() < 6 + len(self.items):
            return ""
        self.change_held = sig
        rule = "lowest price" if self.brief.get("pick") == "cheapest" else f"{self.brief.get('pick')} {self.brief.get('pick_measure') or ''}".strip()
        return ("Not run yet (no step used). The requirements or the budget changed after this cart was set (see [changes so far] and the ledger),"
                f" and the pick is by {rule}: listings that failed a dropped requirement or the old budget may now fit and win."
                " Look again among every listing of each item's kind for the best one under the requirements in force (for the lowest price, browse the"
                " category with a price cap below the cart line's price and page through it), cart the winner, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + self.show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        if self.brief.get("pick") != "cheapest":
            return []
        mark = self.cart_mark() + f"#{self.version}"
        cat_key = self.category_key()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.price_of(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.buyable(i)
                    and str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.price_of(i) if self.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price_of(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.roles.get("compare")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        # a side-by-side view without attributes shows the planner no keys: the attributes view one by one instead
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.full)
        reply = self.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and self.left() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups, up to two each. The pages read here are what the question planner sees, so
        # typical rows go first (most words shared with the item and its required values), the price only breaking
        # ties: race 13's cheapest rows were often sparse pages without the key that held the preference
        lanes: list[list[str]] = []
        cat_key = self.category_key()
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            number = self.item_for_label(str((self.rows.get(ids[0]) or {}).get(cat_key) or "")) if ids and cat_key else None
            item = self.items[number - 1] if number else None
            want = set(_squash(f"{item.name} " + " ".join(m.split(':', 1)[-1] for m in item.musts)).split()) if item else set()
            pool.sort(key=lambda i: (-len(want & set(self.row_text(i).split())), self.price_of(i) if self.price_of(i) is not None else float("inf")))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
            return False
        # the planner names properties by what a buyer of that kind of product chooses by, so a few read listings per
        # item are enough; asking early lets every later search use the answers
        opened = len(self.full)
        # two read pages of every item at least (a planner seeing one item's keys only asks that item): race 13's
        # multi-item requests drew most questions toward the first item's keys
        each = all(self.item_reads(n) >= 2 for n in range(1, len(self.items) + 1))
        if opened >= 2 * len(self.items) + 1 and each:
            return True
        return self.step >= max(6, self.max_steps // 5)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("ask")
        schema = self.schemas.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    # ---------- the interview: one property per message, each chosen after the previous answer was read ----------
    # The shopper answers a question only about a property of the product that it holds a preference on, and the
    # grader counts a hidden preference as found only when some question asked about that property: so every message
    # names one property of one item, a property that drew no preference is never asked again under another name, and
    # an item whose preference was found gives its turn to the items still without one.

    def item_for_label(self, label: str) -> int | None:
        """The requested item a category label belongs to: the item whose shelf is that label, else the one whose
        name or shelf shares the most long words with it; None when none does. A request for one item owns them all."""
        if len(self.items) == 1:
            return 1
        scores = self.label_scores(label)
        best = max(scores, key=lambda n: scores[n]) if scores else None
        return best if best is not None and scores[best] > 0 else None

    def label_scores(self, label: str) -> dict[int, float]:
        """How strongly a category label names each requested item: its shelf exactly, else the words of three or
        more letters it shares with the item's name or shelf (an exact shelf counts above any word overlap)."""
        words = {w for w in _stem(" ".join(re.findall(r"[a-z0-9]+", _words(label)))) if len(w) >= 3}
        scores: dict[int, float] = {}
        for number, item in enumerate(self.items, 1):
            if item.shelf and _words(item.shelf) == _words(label):
                scores[number] = 100.0
                continue
            own = {w for w in _stem(" ".join(re.findall(r"[a-z0-9]+", _words(f"{item.shelf or ''} {item.name}")))) if len(w) >= 3}
            scores[number] = float(len(words & own))
        return scores

    def facets(self, number: int) -> list[tuple[str, int, int, list[str]]]:
        """The attribute keys of the item's listings whose attributes were read: (key, listings carrying it, distinct
        values, sample values). Keys that only identify a listing (a different long value on each) or list purchase
        variants are left out, and of two keys with the same value on every listing only the more carried stays."""
        cat_key = self.category_key()
        slots: dict[str, dict[str, str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            block = _fact_block(record) if record else None
            if not block:
                continue
            label = str(record.get(cat_key) or "") if cat_key else ""
            if self.item_for_label(label) != number:
                continue
            for key, value in block.items():
                name, text = _words(key), str(value).strip().lower()
                if name and text and len(name.split()) <= 5:
                    slots.setdefault(name, {})[ident] = text[:60]
        slots = self.fold_keys(slots)
        table: list[tuple[str, int, int, list[str]]] = []
        for name, per in sorted(slots.items(), key=lambda kv: -len(kv[1])):
            values = list(per.values())
            mean = sum(len(v) for v in values) / len(values)
            if mean > 40 or 2 * sum(1 for v in values if " / " in v) >= len(values):
                continue
            if len(values) >= 4 and len(set(values)) == len(values) and self.identifies(values):
                continue
            if any(set(per) <= set(slots[other]) and all(slots[other][i] == v for i, v in per.items()) for other, *_ in table):
                continue
            table.append((name, len(per), len(set(values)), list(dict.fromkeys(values))[:4]))
        return table

    @staticmethod
    def fold_keys(slots: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        """Spellings of one property folded into its shortest name before keys are counted ("battery type", "type of
        battery"; "display size", "display size inch"): one name's words inside another's, and, when the shorter name is
        a single word, the same value on every listing carrying both (a "lid colour" is not the "colour")."""
        names = sorted(slots, key=lambda n: (len(n.split()), len(n)))
        into: dict[str, str] = {}
        for long in names:
            for short in names:
                if short == long or into.get(short, short) != short or len(short.split()) >= len(long.split()):
                    continue
                small, large = _stem(short), _stem(long)
                if not small or not small < large:
                    continue
                both = [i for i in slots[short] if i in slots[long]]
                same = both and all(slots[short][i] == slots[long][i] or slots[short][i] in slots[long][i] or slots[long][i] in slots[short][i] for i in both)
                if len(small) >= 2 or same:
                    into[long] = short
                    break
        folded: dict[str, dict[str, str]] = {}
        for name in names:
            head = into.get(name, name)
            while head in into:
                head = into[head]
            target = folded.setdefault(head, {})
            for ident, value in slots[name].items():
                target.setdefault(ident, value)
        return folded

    @staticmethod
    def identifies(values: list[str]) -> bool:
        """Values that only tell listings apart (codes, model names, long free text), not a property a buyer chooses by:
        mostly digits, or long, or sharing no word with one another. A kind or style key also takes a different value
        on each of a few listings ("sideboard", "tv stand") but its values are short words that recur across a category."""
        text = "".join(values)
        digits = sum(ch.isdigit() for ch in text) / max(1, len(text))
        mean = sum(len(v) for v in values) / len(values)
        if digits > 0.3 or mean > 30:
            return True
        if mean <= 12:
            return False
        words = [{w for w in re.findall(r"[a-z]{3,}", v)} for v in values]
        shared = sum(1 for i, w in enumerate(words) if any(w & o for j, o in enumerate(words) if j != i))
        return 2 * shared < len(values)

    def item_reads(self, number: int) -> int:
        """Listings of the item whose attributes were read."""
        cat_key = self.category_key()
        return sum(1 for ident in self.seen_order if _fact_block(self.full.get(ident) or {})
                   and self.item_for_label(str((self.full.get(ident) or {}).get(cat_key) or "") if cat_key else "") == number)

    def say_name(self, number: int, label: str) -> str:
        """A property's name as the question says it. Race 13: "closure" drew the preference 69% of the times asked,
        "jackets closure type" never; "display size" 61%, "display size mobile" never: words of the product's own name
        and a leading short store code go. Every other word stays (race 14: hidden preferences sat on two-word keys
        whose last word was dropped by the shorter name; asking the shorter name found them only some of the time)."""
        words = _words(label).split()
        own = set(_stem(f"{self.items[number - 1].name} {self.items[number - 1].shelf or ''}"))
        # the product's own name goes only from a long label ("jackets closure type"); "skirt length" stays whole
        kept = ([w for w in words if _stem(w) and not (_stem(w) <= own)] or words) if len(words) >= 3 else words
        # a name left ending on a short joining word ("number of" from "number of shelves") is no name: keep it whole
        if kept != words and len(kept[-1]) <= 3 and not kept[-1].isdigit():
            kept = words
        if len(kept) >= 2 and len(kept[0]) <= 2 and not kept[0].isdigit():
            kept = kept[1:]
        return " ".join(kept)

    def value_slots(self, number: int) -> dict[str, dict[str, str]]:
        """Per attribute key of the item's read listings: listing id -> value."""
        cat_key = self.category_key()
        slots: dict[str, dict[str, str]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            block = _fact_block(record) if record else None
            if not block or self.item_for_label(str(record.get(cat_key) or "") if cat_key else "") != number:
                continue
            for key, value in block.items():
                name, text = _words(key), str(value).strip().lower()
                if name and text:
                    slots.setdefault(name, {})[ident] = text[:60]
        return slots

    def settled_props(self, number: int) -> list[frozenset[str]]:
        """Properties the item's requirements already decide (from the request, the answers, or a dropped one)."""
        item = self.items[number - 1]
        done = [_stem(m.split(":")[0]) for m in item.musts + item.told + item.dropped if m.split(":")[0].strip()]
        # a key whose read values hold a required value is settled too, whatever the requirement calls the property
        # ("print: animal print" settles a pattern key showing "animal print")
        wanted = [(_stem(m.split(":", 1)[0]), _squash(m.split(":", 1)[-1])) for m in item.musts + item.told if ":" in m and m not in item.dropped]
        wanted = [(prop, w) for prop, w in wanted if w and len(w) >= 3]
        if wanted:
            for name, per in self.value_slots(number).items():
                for prop, w in wanted:
                    hits = sum(1 for v in per.values() if f" {w} " in f" {_squash(v)} ")
                    if hits >= 2 and (len(w.split()) >= 2 or prop & _stem(name)):
                        done.append(_stem(name))
                        break
        return done

    def ranked_keys(self, number: int) -> list[str]:
        """The item's attribute keys in the order they are asked about when no hint picks them: keys whose values
        differ between listings first, then by how many listings carry them and how many values they take."""
        keep = set(self.carried_keys(number))
        rows = [r for r in self.facets(number) if r[0] in keep]
        rows.sort(key=lambda r: (r[2] >= 2 or r[1] < 2, r[1], r[2]), reverse=True)
        return [r[0] for r in rows]

    def carried_keys(self, number: int) -> list[str]:
        """The item's keys by how many read listings carry them, leaving out ones that list variants, identify the
        listing or name its condition, warranty, brand or shipping, and ones the requirements already settle."""
        junk = r"\b(?:options?|variants?|sku|ids?|codes?|warrant\w*|condition|brand|manufacturer|origin|availability|shipping|package\w*|url|images?)\b"
        done = self.settled_props(number)
        rows = [r for r in self.facets(number) if not re.search(junk, r[0])
                and not any(_names(d, _stem(r[0])) or _names(_stem(r[0]), d) for d in done)]
        return [r[0] for r in sorted(rows, key=lambda r: -r[1])]

    def plan_questions(self, room: int) -> dict[int, list[tuple[str, bool]]]:
        """The planner model's picks per item: (property, from a hint in the request or one it asks to be asked)."""
        settled = [f"item {n}: {it.name} - the request already settles: " + ("; ".join(it.musts) or "nothing")
                   for n, it in enumerate(self.items, 1)]
        keys = []
        for number in range(1, len(self.items) + 1):
            read = self.item_reads(number)
            for name, carried, distinct, values in self.facets(number)[:24]:
                keys.append(f"  item {number} - {name} (on {carried} of {read} read listings, {distinct} values): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(room)).replace("SLOTS", str(ASK_SLOTS)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nTHE REQUEST ASKS TO BE ASKED ABOUT:\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS OF THE LISTINGS READ SO FAR:\n" + ("\n".join(keys) or "(none read yet)")
        plan = self.ask_json(prompt, self.seat(), 2500) or {}
        said = set(_words(self.request).split())
        picks: dict[int, list[tuple[str, bool]]] = {n: [] for n in range(1, len(self.items) + 1)}
        self.plan_rank = {}
        for place, entry in enumerate(_items(plan.get("ask"))):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            label = _plain_label(str(entry.get("label") or ""))
            label = label if entry.get("asked") is True else self.say_name(number, label)
            if not label or len(label.split()) > 4 or any(_words(label) == _words(l) for l, _ in picks[number]):
                continue
            # a hint counts only when its words are the request's own (most of them)
            hint = [w for w in _words(str(entry.get("hint") or "")).split() if len(w) > 2]
            backed = bool(entry.get("asked")) or (len(hint) >= 2 and 2 * sum(w in said for w in hint) >= len(hint))
            picks[number].append((label, backed))
            # the items are asked in the planner's order, an item with a story-backed pick ahead of one without
            rank = place - (100 if backed else 0)
            self.plan_rank[number] = min(self.plan_rank.get(number, rank), rank)
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        return picks

    def question_queues(self, room: int) -> dict[int, list[str]]:
        """Per item, the properties to ask in order: the request's own pointers to a property first, then the planner's
        other picks in its order, then the keys it left out."""
        picks = self.plan_questions(room)
        queues: dict[int, list[str]] = {}
        for number in range(1, len(self.items) + 1):
            queue: list[str] = []

            def put(label: str) -> None:
                if label and all(not _same_prop(label, q) for q in queue):
                    queue.append(label)

            # the request's own pointers first, then the planner's order (what a buyer of that kind of product
            # chooses by), then the keys the planner left out, by how the read listings spread over them
            for label, backed in picks.get(number, []):
                if backed:
                    put(label)
            for label, backed in picks.get(number, []):
                put(label)
            for key in self.ranked_keys(number):
                put(self.say_name(number, _plain_label(key)))
            # The planner's picks keep their places (race 13: forcing the two most carried keys into a one-item
            # request's two questions asked construction and colour where material held the preference). One guard
            # only: the second slot goes to the item's best spread key when the planner's unhinted second pick is
            # carried by far fewer of the read listings.
            hinted = {_words(l) for l, b in picks.get(number, []) if b}
            spread = [self.say_name(number, _plain_label(k)) for k in self.ranked_keys(number)[:1]]
            if len(queue) >= 2 and spread and _words(queue[1]) not in hinted and not any(_same_prop(spread[0], q) for q in queue[:2]):
                share = self.key_share(number)
                if share.get(_words(spread[0]), 0.0) >= share.get(_words(queue[1]), 0.0) + 0.34:
                    queue.remove(spread[0]) if spread[0] in queue else None
                    queue.insert(1, spread[0])
            queues[number] = queue
        return queues

    def key_share(self, number: int) -> dict[str, float]:
        """Per spoken key name: the share of the item's read listings that carry it."""
        read = max(1, self.item_reads(number))
        share: dict[str, float] = {}
        for name, carried, _, _ in self.facets(number):
            spoken = _words(self.say_name(number, _plain_label(name)))
            share[spoken] = max(share.get(spoken, 0.0), carried / read)
        return share

    def next_pairs(self, queues: dict[int, list[str]], front: list[tuple[int, str]], room: int, quiet: dict[int, int]) -> list[tuple[int, str]]:
        """The next message's (item, property) pairs, in the order it names them: the names the last answer sent back
        first, in that order, then fresh names from the items' queues. One item's names fill a message; several items
        share one (the shopper answers the first property named that it keeps a view on, whichever item it belongs
        to, so a message about an item it keeps no view on is not spent for nothing). An item whose preference was
        found, or whose last names drew a puzzled reply, comes after the others."""
        for number, queue in queues.items():
            done = self.settled_props(number) + list(self.asked_props.get(number, []))
            queue[:] = [q for q in queue if not any(_same_stems(d, _stem(q)) for d in done)]
        live = [n for n in sorted(queues) if queues[n]]
        if not live:
            return []
        rank = getattr(self, "plan_rank", {}) or {}
        live.sort(key=lambda n: (bool(self.found_on.get(n)), quiet.get(n, 0), self.asked_msgs.get(n, 0), rank.get(n, 1000 + n), n))
        cap = room
        pairs: list[tuple[int, str]] = []
        for number, label in front:
            if len(pairs) < cap and label in queues.get(number, []) and (number, label) not in pairs:
                pairs.append((number, label))
        share = cap if len(live) == 1 else max(1, -(-cap // len(live)))
        # each item its share first, then whatever room is left to the items in order
        for limit in (share, cap):
            for number in live:
                slots = self.value_slots(number)
                mine = [l for n, l in pairs if n == number]
                for label in queues[number]:
                    if len(pairs) >= cap or len(mine) >= limit:
                        break
                    if label in mine or any(_same_prop(label, other) or self.same_values(slots, label, other) for other in mine):
                        continue
                    pairs.append((number, label))
                    mine.append(label)
        # a message names each item's properties together, the items in the order they first appear
        order = list(dict.fromkeys(n for n, _ in pairs))
        return [(n, l) for n in order for m, l in pairs if m == n]

    @staticmethod
    def same_values(slots: dict[str, dict[str, str]], a: str, b: str) -> bool:
        """Two keys that state the same value on every one of at least two listings carrying both: one property spelt
        twice ("closure" and "jackets closure type"), so one question slot names it."""
        def per(label: str) -> dict[str, str] | None:
            hit = slots.get(_words(label))
            if hit:
                return hit
            # a spoken label shortened from a key ("closure type" from "jackets closure type"): that key's values
            return next((v for k, v in slots.items() if k.endswith(" " + _words(label))), None)
        x, y = per(a), per(b)
        if not x or not y:
            return False
        both = [i for i in x if i in y]
        return len(both) >= 2 and all(x[i] == y[i] for i in both)

    def item_word(self, number: int) -> str:
        """How a question names a requested item: its name; one that joins several plural kinds with "and" ("door
        handles and locks") is one choice to the shopper, so "or" joins them."""
        name = self.items[number - 1].name
        parts = [p for p in re.split(r"\s*(?:,|&|\band\b)\s*", name) if p.strip()]
        if len(parts) >= 2 and all(re.search(r"[a-z]{3}(?<![su])s$", p.strip(), re.I) for p in parts):
            name = re.sub(r"\s*(?:&|\band\b)\s*", " or ", name)
        return name

    def question(self, pairs: list[tuple[int, str]]) -> str:
        """One plain question naming the properties of one item, or of several items one after another (race 13:
        "Which X would you like the Y to have?" drew the preference most often for a lone property)."""
        groups: list[tuple[int, list[str]]] = []
        for number, label in pairs:
            if groups and groups[-1][0] == number:
                groups[-1][1].append(label)
            else:
                groups.append((number, [label]))
        def names(labels: list[str]) -> str:
            return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"
        if len(groups) == 1:
            number, labels = groups[0]
            if len(labels) == 1:
                return f"What {labels[0]} would you like for the {self.item_word(number)}?"
            return f"For the {self.item_word(number)}, which {names(labels)} would you like?"
        parts = [f"for the {self.item_word(n)}, which {names(ls)}" for n, ls in groups]
        text = "; ".join(parts[:-1]) + "; and " + parts[-1] + " would you like?"
        return text[0].upper() + text[1:]

    def said_item(self, answer: str) -> int | None:
        """The one requested item an answer names ("... of the <item>"), else None."""
        if len(self.items) == 1:
            return 1
        said = f" {_words(answer)} "
        cat_key = self.category_key()
        found = []
        for number, item in enumerate(self.items, 1):
            names = {_words(item.name), _words(item.shelf or "")} - {""}
            names |= {_words(str(self.rows[i].get(cat_key) or "")) for i in self.item_kind_rows(number)[:1]} if cat_key else set()
            if any(len(n) >= 4 and f" of the {n}" in said for n in names):
                found.append(number)
        return found[0] if len(found) == 1 else None

    def answer_owner(self, pairs: list[tuple[int, str]], got: dict[str, Any], answer: str) -> int | None:
        """The item an answer is about among the message's items: the reader's item number when it is one of them,
        else the one item the answer's words name; None when neither says."""
        items = {n for n, _ in pairs}
        item = _count(got.get("item"))
        if item in items:
            return item
        named = self.said_item(answer)
        return named if named in items else None

    def answered_pair(self, pairs: list[tuple[int, str]], got: dict[str, Any], answer: str) -> int:
        """Where in the message the answer's property sits: the asked name the reader matched, else the first asked
        name sharing the property's words, on the answer's item when it is known; -1 when none does."""
        owner = self.answer_owner(pairs, got, answer)
        spots = [(at, label) for at, (number, label) in enumerate(pairs) if owner is None or number == owner]
        matched = _words(str(got.get("matched") or ""))
        for at, label in spots:
            if matched and _words(label) == matched:
                return at
        said = _stem(str(got.get("need") or "").split(":")[0])
        for at, label in spots:
            mine = _stem(label)
            if said and mine and (_same_stems(said, mine) or mine <= said or said <= mine):
                return at
        return -1

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        queues = self.question_queues(self.msgs_free() * ASK_SLOTS + 2)
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        room = ASK_SLOTS
        front: list[tuple[int, str]] = []      # names an answer sent back, asked next in this order
        again: set[tuple[int, str]] = set()     # names already sent back once
        quiet: dict[int, int] = {}              # per item: lists that drew a puzzled reply
        waved: list[tuple[int, str]] = []       # the last list a "don't mind" answered, while the next is out
        free_seen: set[tuple[int, str]] = set()  # every name of a message a "don't mind" answered
        last: list[tuple[int, str]] = []
        asked_texts: set[str] = set()
        repeats = 0

        def settle(done: list[tuple[int, str]]) -> None:
            for number, label in done:
                self.asked_props.setdefault(number, set()).add(_stem(label))

        def send_back(back: list[tuple[int, str]]) -> None:
            front[:0] = back

        while self.msgs_free() > 0 and not self.finished and self.left() > 6 and not self.spent():
            # the first message names few properties: two for one item, one when several items each may hold a
            # preference (two tracked properties in one message are refused by some shoppers)
            opening = FIRST_SLOTS if len(self.items) == 1 else 1
            pairs = self.next_pairs(queues, front, opening if not sent and room > 1 else room, quiet)
            front.clear()
            if pairs == last:
                # never the same message twice: its first name was surely heard
                settle(pairs[:1])
                pairs = pairs[1:]
            if not pairs:
                break
            last = list(pairs)
            number = pairs[0][0]
            labels = [l for _, l in pairs]
            text = self.question(pairs)
            if text in asked_texts:
                # a message is never sent twice in the same words: its first name was surely heard
                settle(pairs[:1])
                repeats += 1
                if repeats > 6:
                    break
                continue
            asked_texts.add(text)
            batch = self.send([(number, ", ".join(labels), text)])
            if not batch:
                break
            sent += batch
            answer = batch[0][2]
            for n in dict.fromkeys(n for n, _ in pairs):
                self.asked_msgs[n] = self.asked_msgs.get(n, 0) + 1
            got = self.listen(pairs, text, answer)
            kind = str(got.get("kind") or "")
            compound = re.search(r"\b(?:stack\w*|queue\w*|one at a time|one by one|at once|in one go|too many|several|top one|"
                                 r"the rest|pick the first|first one|questions)\b", answer, re.I)
            if kind == "split" and len(pairs) > 1 and room > 1 and (compound or str(got.get("kind")) == "split"):
                # a shopper who takes one property at a time: this list goes back and every later question names one
                # (the reader's "split" is trusted: that shopper's wording varies, "split them up", "one query", "in
                # smaller bites"). Such a shopper answers a message naming one tracked property, so a name from a
                # message it answered "don't mind" is most likely the free one: those names go last
                room = 1
                send_back([p for p in pairs if p not in free_seen] + [p for p in pairs if p in free_seen])
                self.trace.append({"kind": "ask_singly", "labels": labels})
                continue
            if kind == "split" and len(pairs) == 1:
                # one property and still "one at a time": the name matched two things the shopper tracks; it is
                # settled and the next question names another property
                settle(pairs)
                self.trace.append({"kind": "ask_split_single", "label": labels[0]})
                continue
            if kind != "free" or len(pairs) <= 1:
                # the two-list reasoning after "don't mind" holds only for two such answers in a row
                waved.clear()
            if kind == "need":
                at = self.answered_pair(pairs, got, answer)
                owner = pairs[at][0] if at >= 0 else (self.answer_owner(pairs, got, answer) or number)
                if self.take_need(owner, got, told, pairs[at][1] if at >= 0 else labels[0]):
                    self.found_on[owner] = self.found_on.get(owner, 0) + 1
                if at >= 0:
                    # the shopper answers the first property named that it keeps a view on: the names before it hold
                    # none, the names after it are still open and go next, in the same order
                    settle(pairs[:at + 1])
                    send_back(pairs[at + 1:])
                else:
                    # the answer names a property in words of its own: the later names go once more
                    settle(pairs[:1])
                    back = [p for p in pairs[1:] if p not in again]
                    again.update(back)
                    send_back(back)
                self.trace.append({"kind": "ask_hit", "at": at, "owner": owner})
            elif kind == "lost":
                # puzzled: the message named nothing the shopper keeps a view on
                settle(pairs)
                for n in dict.fromkeys(n for n, _ in pairs):
                    quiet[n] = quiet.get(n, 0) + 1
                self.trace.append({"kind": "ask_none", "names": len(pairs)})
            elif kind == "free" and len(pairs) > 1:
                free_seen.update(pairs)
                # "don't mind": the first property named that the shopper keeps a view on holds nothing, but which one
                # is unknown, most likely one named early. A list's first name is that one or one the shopper does not
                # track, so it goes; the later names are asked again, the last first. After a second such answer a
                # held property must sit after the first tracked name of both lists, so the names far from the front
                # of both go first (six of them, then fresh names): in a simulation of this shopper that finds a
                # property held behind two others 84-91% of the time, against 83-89% for dropping both lists' first
                # two names (which lost a held "gender" named second in a reversed list) and 73-79% for fresh names first
                settle(pairs[:1])
                if waved:
                    p1 = {pair: at for at, pair in enumerate(waved)}
                    p2 = {pair: at for at, pair in enumerate(pairs)}
                    done = {waved[0], pairs[0]}
                    old = [pair for pair in dict.fromkeys(waved + pairs) if pair not in done]
                    old.sort(key=lambda pair: (-min(p1.get(pair, 99), p2.get(pair, 99)), p2.get(pair, 99)))
                    back = old[:6]
                    waved.clear()
                else:
                    waved.extend(pairs)
                    back = list(reversed(pairs[1:]))
                send_back(back)
                self.trace.append({"kind": "ask_again", "back": len(back)})
            else:
                settle(pairs)
            if isinstance(got.get("budget"), (int, float)) and not isinstance(got.get("budget"), bool) and got["budget"] > 0:
                self.budget = got["budget"]
        if told:
            self.version += 1
        if not sent:
            return ""
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        walked = self.walk_links()
        if walked:
            summary += "\n" + walked
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def take_need(self, number: int, got: dict[str, Any], told: list[str], asked: str = "") -> bool:
        """Record a preference an answer stated on the item; True when it is new."""
        need = str(got.get("need") or "").strip()
        if need and ":" not in need:
            need = f"{got.get('property') or asked}: {need}".strip(": ")
        if not need and str(got.get("constraint") or got.get("value") or "").strip():
            need = f"{str(got.get('property') or '').strip() or 'property'}: {str(got.get('constraint') or got.get('value')).strip()}"
        if not need or ":" not in need:
            return False
        item = self.items[number - 1]
        if need in item.told:
            return False
        item.told.append(need)
        told.append(f"item {number}: {need}" if len(self.items) > 1 else need)
        if got.get("points") is True and (number, need) not in self.refs:
            self.refs.append((number, need))
        return True

    def listen(self, pairs: list[tuple[int, str]], question: str, answer: str) -> dict[str, Any]:
        """Read one answer (the light model first; the heavy one when the light one gives no reading)."""
        groups: dict[int, list[str]] = {}
        for number, label in pairs:
            groups.setdefault(number, []).append(label)
        asked = "\n".join(f"item {n} ({self.items[n - 1].name}): " + "; ".join(ls) for n, ls in groups.items())
        prompt = LISTEN + f"\n\nPROPERTIES ASKED ABOUT, BY ITEM:\n{asked}\nQUESTION: {question}\nANSWER: {answer}"
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        self.trace.append({"kind": "heard", "pairs": len(pairs), "got": _clip(got, 80)})
        return got

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.message_args(text)
            if not args:
                break
            reply = self.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            if isinstance(answer, dict):
                # a reply can come with notes beside it (lists of lines next to the one reply text): the reply is the
                # shopper's answer, the notes change the task the way a note on any other result does
                texts = [v for v in answer.values() if isinstance(v, str) and v.strip()]
                notes = [x for v in answer.values() if isinstance(v, list) for x in v if isinstance(x, str) and len(x.split()) >= 4]
                if len(texts) == 1 and notes:
                    words = texts[0]
                    fresh = [x for x in notes if x not in self.heard]
                    meaning = self.take_note(fresh) if fresh else ""
                    if meaning:
                        self.queued.append(f"(with a reply) {meaning}")
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = str(entry["need"]).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        # what the request wants said when nothing fits is written into the ending's own text (decline_text), never sent
        if stage == "end" or not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = (GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                # what a reply to the gate's own call asks for (earlier answers are the interviewer's to read)
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f"#{self.version}"))
                break
            if self.left() < 2 or (name == self.roles.get("ask") and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.schemas[name], "args", faults)
            if faults:
                break
            if name == self.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.absorb(seen)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""

    def spoken(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        return " ".join(found)[:800] if found else _dump(observation)[:400]

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = budget
            changed.append(f"budget now {budget}")
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(":")[0]) == _words(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
            if any(c.startswith(("budget now", "dropped", "added")) for c in changed):
                self.rearm()
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def rearm(self) -> None:
        """A change to the task (budget, a requirement added or dropped, a listing gone) opens the one-time checks
        again: the decline review, the search-more push, the late hold and the cheapest widening judge the new task."""
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, False, depth)
                if ident and _fact_block(node) and ident not in ids:
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)

        walk(observation, 0)
        needs = self.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f"(item {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    # ---------- checks ----------
    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get("pick") == "cheapest":
            # listings whose attributes were read first, each group cheapest first
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        limit = self.spend_hard if limit is None else limit
        if key in self.checked and self.verdicts.get(key) is False and self.check_said.get(key) and self.check_repeats.get(key, 0) < 1:
            # the same cart the check turned down comes back unchanged: the review stands
            self.check_repeats[key] = self.check_repeats.get(key, 0) + 1
            return self.check_said[key]
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        # the heavy model judges the episode's first cart while the purse is light; later checks go to the light one (a
        # second heavy check flip-flopped between two fitting carts until the steps ran out)
        got = self.ask_json(prompt, self.seat(self.heavy_checks < 1), 3000)
        self.heavy_checks += 1
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        if got:
            self.verdicts[key] = got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        text += ". Fix the cart (or show the review is wrong by re-reading the records), then check out."
        self.check_said[key] = text
        return text

    # ---------- requirements whose value is on another listing ----------
    # A need like "the same X as on the page for the <kind> sold by <shop> that has <...> (or wherever that page points
    # you)" is graded as met only when the order has the value of the LAST page of that chain AND every page of the
    # chain was opened with a full-page view. Kyo walks the chain itself as soon as the need is known: find the listing
    # the text describes, open its page, and either read the value there or follow the page on to the next listing.

    def search_call(self, words: str) -> tuple[str, dict[str, Any]] | None:
        """A call of the search tool with these words as its text, when its schema allows one with nothing else."""
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        wordy = [k for k in texts if set(_words(k).split()) & {"query", "phrase", "text", "keyword", "keywords", "q", "search", "terms", "words"}]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (wordy or texts or [None])[0]
        if not tool or not target:
            return None
        args: dict[str, Any] = {target: words}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    # a mode argument takes the value that names a word search ("words", "keyword"), else its first
                    wordish = [v for v in spec["enum"] if set(_words(str(v)).split()) & {"word", "words", "keyword", "keywords", "text", "search", "query", "terms"}]
                    args[key] = (wordish or spec["enum"])[0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else (tool, args)

    def note_search(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Whether this store's word search finds a code: a search whose words hold a model or part code counts as a
        hit when rows come back, else as a miss (race 15: 293 of 295 such searches returned nothing, while race 14's
        search found the parts that named a code)."""
        if name != self.roles.get("search") or (isinstance(observation, dict) and observation.get("error")):
            return
        words = " ".join(str(v) for v in args.values() if isinstance(v, str))
        if not re.search(self.CODE, words):
            return

        def rows(node: Any, depth: int) -> bool:
            if isinstance(node, list):
                return bool(node)
            return isinstance(node, dict) and depth < 2 and any(rows(v, depth + 1) for v in node.values())
        counted = [v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                   and re.search(r"count|total|match|found|number", str(k), re.I)] if isinstance(observation, dict) else []
        if rows(observation, 0) or any(v > 0 for v in counted):
            self.code_hits += 1
        else:
            self.code_misses += 1

    def codes_searchable(self) -> bool:
        return self.code_hits > 0 or self.code_misses < 2

    def part_refs(self, ident: str) -> list[str]:
        """Other parts a listing's page sends the reader to ("... part X ...", with no other code): a sentence
        naming another part's code and no other code."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        title_codes = {c.upper().replace("-", "") for k, v in record.items() if isinstance(v, str) and self.title_field(k)
                       for c in re.findall(self.CODE, v)}
        out: list[str] = []
        for sentence in re.split(r"(?<=[.;!?])\s+(?=[A-Z][a-z])|\n|•|\s\|\s", self.compat_text(ident)):
            codes = {c.upper().replace("-", "") for c in re.findall(self.CODE, sentence)} - title_codes
            named = [c.upper().replace("-", "") for c in re.findall(r"\bpart\s+(" + self.CODE + ")", sentence, re.I)]
            if named and codes <= set(named):
                out += [c for c in named if c not in title_codes and c not in out]
        return out

    def surface_parts(self) -> list[str]:
        """Before checking out several items that work together: look up each part a cart line's page sends the reader
        to (race 14: every passing multi-item run with such a line had searched or opened that part; 72 of 76 runs that
        had not failed). One word search per part, four at most per episode."""
        if len(self.items) < 2 or self.parts_looked >= 4 or not self.codes_searchable():
            return []
        done: list[str] = []
        for ident in self.cart_ids():
            for code in self.part_refs(ident):
                if code in self.parts_seen or self.parts_looked >= 4 or self.left() < len(self.items) + 5 or self.finished:
                    continue
                call = self.search_call(code)
                if not call:
                    return done
                self.parts_seen.add(code)
                self.parts_looked += 1
                reply = self.act(*call)
                if reply is None:
                    return done
                self.observe(call[0], reply.get("observation"))
                done.append(code)
        if done:
            self.trace.append({"kind": "parts_surfaced", "codes": done})
        return done

    def row_text(self, ident: str) -> str:
        """Everything known of a listing as lower-case words: its row, its attributes and its page."""
        return _squash(_dump(self.full.get(ident) or self.rows.get(ident) or {}))

    def seller_of(self, ident: str, shop: str) -> bool:
        """The listing names `shop` as one of its own field values (its shop, seller or brand), or in its title."""
        want = _words(shop)
        if not want:
            return False
        row = self.rows.get(ident) or {}
        if any(isinstance(v, str) and _words(v) == want for v in row.values()):
            return True
        return f" {want} " in f" {self.row_text(ident)} " and len(want) >= 5

    def described(self, target: dict[str, Any]) -> list[str]:
        """The values a link gives to pick its listing out, as plain words."""
        out = []
        for entry in _items(target.get("has")):
            # the value alone: a page says "running ios" for "operating system: ios"
            if isinstance(entry, dict):
                entry = entry.get("value") if entry.get("value") not in (None, "") else " ".join(str(v) for v in entry.values())
            text = _squash(str(entry))
            if text:
                out.append(text)
        return out

    def own_words(self, ident: str) -> tuple[str, str, str]:
        """A listing's record as (what its own sentences say of it, what they deny, its other fields): sentences of
        its free text that say what it is not, or that another version exists elsewhere, are the denied part."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        said, denied, fields = [], [], []
        for key, value in record.items():
            if isinstance(value, str) and len(value.split()) >= 6:
                for sentence in re.split(r"(?<=[.!?])\s+", value):
                    low = sentence.lower()
                    # a sentence sending the reader to another listing describes that one, not this
                    if re.search(r"\bnot (?:from|to|by|for|in|on) (?:this|the|that)\b|rather than (?:this|the|that)|\binstead\b|\brefer|\bgo by\b|"
                                 r"\blook to\b|\btaken? (?:it |its |the |\w+ )?from\b|\bcomes? from\b|\bsee (?:a|the|another)\b|\buse the\b|\bcheck the\b", low):
                        continue
                    if "elsewhere" in low:
                        denied.append(sentence)
                        continue
                    cue = re.search(r"\bnot\b|n't\b", sentence, re.I)
                    if cue:
                        said.append(sentence[:cue.start()])
                        denied.append(sentence[cue.end():])
                    else:
                        said.append(sentence)
            else:
                fields.append(f"{key} {value}" if not isinstance(value, (dict, list)) else _dump(value))
        return f" {_squash(' '.join(said))} ", f" {_squash(' '.join(denied))} ", f" {_squash(' '.join(fields))} "

    def link_fit(self, ident: str, target: dict[str, Any]) -> tuple[int, int, int]:
        """(described values the listing's own sentences state, values its fields state, values it denies), each
        value matched word by word as whole words. A value whose property the listing has an attribute for is judged
        on that attribute first: the attribute stating another value denies it (race 14: a page whose memory attribute
        said another figure was taken for the link). A value of three letters or fewer counts only from an attribute (it can
        be an ordinary word in free text)."""
        said, denied, fields = self.own_words(ident)
        block = _fact_block(self.full.get(ident) or {}) or {}

        def holds(text: str, value: str) -> bool:
            if all(f" {w} " in text for w in value.split()):
                return True
            # a value with figures matches by its figures, whatever unit word follows them ("1250 servings" against
            # "1,250 calories per serving" in race 13: the store's words for the unit differ from the shopper's)
            parts = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+)?", value.strip())
            if not parts:
                return False
            return bool(re.search(r"(?<![\w.])" + re.escape(parts.group(1)) + r"(?![\d.])", text))

        in_said = in_fields = in_denied = 0
        for prop, value in self.described_pairs(target):
            keys = [k for k in block if prop and _same_stems(_stem(k), _stem(prop))] if prop else []
            if keys:
                if any(holds(f" {_squash(str(block[k]))} ", value) for k in keys):
                    in_said += 1
                    in_fields += 1
                else:
                    in_denied += 1
                continue
            short = len(value.replace(" ", "")) <= 3 and not re.search(r"\d", value)
            in_said += 0 if short else holds(said, value)
            in_fields += holds(fields, value)
            in_denied += 0 if short else holds(denied, value)
        return in_said, in_fields, in_denied

    def described_pairs(self, target: dict[str, Any]) -> list[tuple[str, str]]:
        """The (property, value) pairs a link gives to pick its listing out, the value as plain words."""
        out = []
        for entry in _items(target.get("has")):
            prop = ""
            if isinstance(entry, dict):
                prop = str(entry.get("property") or "").strip()
                entry = entry.get("value") if entry.get("value") not in (None, "") else " ".join(str(v) for v in entry.values())
            text = _squash(str(entry))
            if text:
                out.append((prop, text))
        return out

    def exact_fit(self, ident: str, target: dict[str, Any]) -> bool:
        values = self.described(target)
        said, _, denied = self.link_fit(ident, target)
        return bool(values) and said == len(values) and not denied

    def qualifies(self, ident: str, target: dict[str, Any]) -> bool:
        """The page a link means restates every value the link describes in its own sentences and denies none. When
        the words differ (units, synonyms: "2000 gb" against "2 tb"), an opened page is judged once by the light model."""
        values = self.described(target)
        said, _, denied = self.link_fit(ident, target)
        if values and said == len(values) and not denied:
            return True
        if not values or denied or ident not in self.opened or not self.own_words(ident)[0].strip():
            return False
        mark = ident + "|" + "|".join(values)
        if mark not in self.link_judged:
            if len(self.link_judged) >= 24 or self.spent():
                return False
            prompt = (MATCH + "\n\nVALUES: " + "; ".join(str(v) for v in _items(target.get("has")))[:600]
                      + "\n\n" + self.page_text(ident))
            got = self.ask_json(prompt, [LIGHT], 1200) or {}
            self.link_judged[mark] = got.get("all") is True
        return self.link_judged[mark]

    def open_pages(self, ids: list[str]) -> None:
        """Open listings' full pages: two at a time in the side-by-side view when the store has one, else singly."""
        ids = [i for i in ids if i not in self.opened]
        many, joiner = self.compare_target(), self.roles.get("compare")
        if len(ids) >= 2 and many and joiner and self.compare_room() >= 2:
            faults: list[str] = []
            args = _shape({many: ids[:2]}, self.schemas.get(joiner), "args", faults)
            if not faults:
                reply = self.act(joiner, args)
                if reply is not None:
                    self.observe(joiner, reply.get("observation"))
                    self.note_view(joiner, reply.get("observation"))
                ids = [i for i in ids if i not in self.opened]
        for ident in ids:
            if self.left() <= self.walk_room() or self.finished:
                break
            self.open_page(ident)

    def link_score(self, ident: str, target: dict[str, Any]) -> float:
        """How well a listing's known record answers a link: its shop, its kind, and the words of what it has."""
        text = f" {_squash(self.row_text(ident))} "
        cat_key = self.category_key()
        label = str((self.rows.get(ident) or {}).get(cat_key) or "") if cat_key else ""
        kind = {w for w in _stem(" ".join(re.findall(r"[a-z0-9]+", _words(str(target.get("kind") or ""))))) if len(w) >= 3}
        shelf = _stem(" ".join(re.findall(r"[a-z0-9]+", _words(label))))
        score = 4.0 * self.seller_of(ident, str(target.get("shop") or ""))
        score += 2.0 * (len(kind & shelf) / len(kind) if kind and label else 0.0)
        def has(word: str) -> bool:
            if f" {word} " in text:
                return True
            # a figure with its unit ("0.73kg") is found by the figure alone when the page writes the unit apart or not
            # at all ("weight 0.73")
            figure = re.fullmatch(r"(\d+(?:\.\d+)?)[a-z]*", word)
            return bool(figure) and bool(re.search(r"(?<![\w.])" + re.escape(figure.group(1)) + r"(?![\d.])", text))
        for phrase in self.described(target):
            words = phrase.split()
            score += sum(1 for w in words if has(w)) / max(1, len(words))
        return score

    def page_text(self, ident: str) -> str:
        """A listing's page for the link reader: its words first (title, kind, seller, remarks, description), then
        its attributes, each value clipped."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        words = {k: v for k, v in record.items() if isinstance(v, str) and len(v) > 1}
        facts = _fact_block(record) or {}
        text = f"PAGE OF LISTING {ident}:\n" + "\n".join(f"{k}: {str(v)[:500]}" for k, v in words.items())
        if facts:
            text += "\nATTRIBUTES: " + "; ".join(f"{k}: {str(v)[:80]}" for k, v in facts.items())
        return text

    def read_link(self, prop: str, text: str) -> dict[str, Any]:
        """The light model's reading of a requirement or a page: the value it gives for the property, or the next
        listing it sends the reader to."""
        prompt = LINK + f"\n\nPROPERTY: {prop}\nTEXT:\n{text[:3000]}"
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        return got if isinstance(got, dict) else {}

    def find_link(self, target: dict[str, Any], tried: set[str]) -> str | None:
        """The listing a link describes. Candidates are the rows seen from the named shop (any row when no shop is
        named), searched for by shop and kind, then by kind and what it has, when none is known; they are checked
        best first by opening their pages (and reading their attributes) until one shows every described phrase.
        Without a full match the candidate showing the most of them is taken."""
        shop = str(target.get("shop") or "").strip()
        need = len(self.described(target))

        def pool() -> list[str]:
            ids = [i for i in self.rows if i not in tried and (not shop or self.seller_of(i, shop))]
            ids = [i for i in ids if self.link_score(i, target) >= (4.0 if shop else 1.0)]
            return sorted(ids, key=lambda i: (-self.exact_fit(i, target), -self.link_score(i, target)))

        has = " ".join(self.described(target))[:60]
        # a store's search on a shop's name alone tends to list that shop's listings only; its further pages are read
        # too (two at most) while they keep showing the shop's listings and none qualifies yet
        # the shop with the kind and the link's own values first (race 13: "<shop> denim jacket hood" listed the
        # hooded jackets of a shop selling 116, where the shop's name alone listed ten others a page)
        kind = str(target.get("kind") or "").strip()
        values = self.described(target)
        pointed = f"{shop} {kind} {values[0] if values else ''}".strip() if shop and values else ""
        queries = [pointed, shop, f"{shop} {kind}".strip(), f"{kind} {has}".strip()]
        queries = [q for q in dict.fromkeys(queries) if q]
        for index, words in enumerate(queries):
            known = pool()
            if any(self.exact_fit(i, target) for i in known) or (known and index >= 2) \
                    or self.left() <= self.walk_room() or self.finished:
                break
            call = self.search_call(words)
            for page in range(3):
                if not call or self.left() <= self.walk_room() or self.finished:
                    break
                # a search this walker already ran (and paged) at this state of the ledger is not run again
                mark = call[0] + _dump(call[1]) + f"#{self.version}"
                if mark in self.link_searched:
                    break
                self.link_searched.add(mark)
                reply = self.act(*call)
                if reply is None:
                    break
                seen = reply.get("observation")
                self.observe(call[0], seen)
                self.read_cache[call[0] + _dump(call[1])] = (self.version, self.step, self.show(seen))
                shown = [i for i, _ in self.records_in(seen)]
                if not shop or words != shop or any(self.exact_fit(i, target) for i in pool()) or not any(self.seller_of(i, shop) for i in shown):
                    break
                more = self.paging(call[0], seen)
                if not more or more[3] is None or more[0] <= more[1] * (page + 1):
                    break
                faults: list[str] = []
                nxt = _shape({**call[1], more[2]: more[3]}, self.schemas.get(call[0]), "args", faults)
                call = None if faults else (call[0], nxt)
        hit = next((i for i in pool() if self.exact_fit(i, target)), None)
        if hit:
            return hit
        if not need:
            # nothing to tell candidates apart by: the best named one, its page opened
            best = next(iter(pool()), None)
            if best:
                self.open_pages([best])
            return best
        checked: set[str] = set()
        for _ in range(4):
            ranked = [i for i in pool() if i not in checked]
            if not ranked or self.left() <= self.walk_room() or self.finished:
                break
            # the page shows what the row does not: the listing's own sentences about itself
            pair = ranked[:2]
            checked.update(pair)
            self.open_pages(pair)
            hit = next((i for i in pair if self.qualifies(i, target)), None)
            if hit:
                return hit
        # no page restates every value in its own sentences: one whose fields state them all and that denies none
        for ident in pool():
            said, fields, denied = self.link_fit(ident, target)
            if not denied and max(said, fields) >= need:
                return ident
        return None

    def open_page(self, ident: str) -> bool:
        """Open a listing's full page (the page view, else the side-by-side view); True when it was opened."""
        if ident in self.opened:
            return True
        for tool in [self.roles.get("offer") or self.roles.get("open"), self.roles.get("compare")]:
            if not tool or tool not in self.schemas:
                continue
            target = self.single_target(tool) if tool != self.roles.get("compare") else self.compare_target()
            if not target:
                continue
            value: Any = [ident] if tool == self.roles.get("compare") else ident
            faults: list[str] = []
            args = _shape({target: value}, self.schemas.get(tool), "args", faults)
            if faults:
                continue
            reply = self.act(tool, args)
            if reply is None:
                return False
            self.observe(tool, reply.get("observation"))
            self.note_view(tool, reply.get("observation"))
            return ident in self.opened
        return False

    def seller_named(self, ident: str, prop: str, kind: str) -> str:
        """The proper name a page's redirect sentence gives its next listing's seller when the reader gave none: the
        first capitalised word of the sentence naming the property that is not the sentence's first word, not a word
        of the next listing's kind and not the page's own seller."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        texts = [v for v in record.values() if isinstance(v, str) and len(v.split()) >= 6]
        skip = set(re.findall(r"[a-z0-9]+", _words(kind))) | set(re.findall(r"[a-z0-9]+", _words(prop)))
        own = {_words(v) for v in record.values() if isinstance(v, str) and len(v) <= 40}
        want = set(re.findall(r"[a-z0-9]+", _words(prop)))
        for text in texts:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if want and not want <= set(re.findall(r"[a-z0-9]+", sentence.lower())):
                    continue
                words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", sentence)
                if words and sentence.split()[:1] == [words[0]]:
                    words = words[1:]
                for word in words:
                    if word.lower() not in skip and _words(word) not in own and word.lower().rstrip("s") not in skip:
                        return word
        return ""

    def walk_link(self, number: int, need: str) -> str:
        """Follow one requirement's chain of pages; returns a line for the agent (empty when nothing was learnt)."""
        prop = need.split(":")[0].strip()
        first = self.read_link(prop, need)
        target = first.get("next") if isinstance(first.get("next"), dict) else None
        chain: list[str] = []
        value = ""
        tried: set[str] = set()
        while target and len(chain) < 5 and self.left() > self.walk_room() and not self.finished and not self.spent():
            ident = self.find_link(target, tried)
            if not ident:
                break
            tried.add(ident)
            if not self.open_page(ident):
                break
            chain.append(ident)
            got = self.read_link(prop, self.page_text(ident))
            target = got.get("next") if isinstance(got.get("next"), dict) else None
            if target is not None and not str(target.get("shop") or "").strip():
                target["shop"] = self.seller_named(ident, prop, str(target.get("kind") or ""))
            if target is None:
                value = str(got.get("value") or "").strip()
                partner = str(got.get("partner") or "").strip()
                if not value and not partner and self.facts_tool() and ident not in self.full:
                    self.fill_facts([ident], self.tail() + 2)
                    again = self.read_link(prop, self.page_text(ident))
                    value, partner = str(again.get("value") or "").strip(), str(again.get("partner") or "").strip()
                if partner and re.search(r"alongside|compatib|\bgo(?:es)?\b(?: \w+)? with|works? with\b|pairs? with\b|fits? with\b|made for|\bpartner", need, re.I):
                    value = partner
        self.trace.append({"kind": "link", "need": need[:200], "chain": chain, "value": value})
        if chain:
            self.link_chains[need] = chain
        if not value:
            if chain:
                return (f"[Kyo followed the listing '{need}' points at: opened {', '.join(chain)}; the value is not settled yet, "
                        f"open the next listing the last page names, if any, and read {prop} there.]")
            return ""
        self.ref_values[need] = value
        self.version += 1
        plain = self.restate(need, value)
        if plain:
            item = self.items[number - 1]
            # the requirement now reads with the value in place of the pointer, for the agent and the cart check
            item.told = [plain if t == need else t for t in item.told]
            item.musts = [plain if m == need else m for m in item.musts]
            self.ref_values[plain] = value
        return (f"[Kyo followed the listing '{need}' points at and opened each page of the chain ({' -> '.join(chain)}); "
                f"the last page gives {prop}: {value}." + (f" The requirement now reads: {plain}." if plain else
                " Apply that value with the requirement's own comparison words.") + "]")

    def restate(self, need: str, value: str) -> str:
        """The requirement with the value read off the chain in place of the pointer ("display size: within 0.02 inch
        either way of 10.9 inch"), or "" when the light model gives no clean reading."""
        prompt = (RESTATE + f"\n\nREQUIREMENT: {need}\nVALUE ON THE LAST PAGE: {value}")
        got = self.ask_json(prompt, [LIGHT], 1200) or {}
        text = str(got.get("requirement") or "").strip()
        prop = need.split(":")[0].strip().lower()
        if ":" not in text or text.split(":")[0].strip().lower() != prop or len(text) > 160:
            return ""
        return text

    def walk_room(self) -> int:
        """Steps the pointer walker leaves untouched: the finish, plus opening and carting a line for each item (race 13:
        a walker that ran to four steps from the end left a four-hop chain's basket unbought)."""
        return self.tail() + 4 + 2 * len(self.items)

    def walk_links(self) -> str:
        """Walk every chain not yet walked."""
        lines = []
        for number, need in list(self.refs):
            if need in self.ref_values or need in self.link_walked:
                continue
            self.link_walked.add(need)
            if self.left() <= self.walk_room() or self.spent():
                break
            line = self.walk_link(number, need)
            if line:
                lines.append(line)
        return "\n".join(lines)

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < 4 and self.cost < self.spend_hard:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:300])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 3000) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.act(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get("observation")
            self.offer_seen[ident] = self.step
            self.observe(tool, observation)
            # compare the stored record, so a re-read that arrives wrapped together with a note still shows its change
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.version += 1
        return "\n".join(lines)

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > self.spend_hard:
            return []
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.seat(), 3000)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        self.swap_fits = [i for _, i in sorted(fits, key=lambda f: f[0])]
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or self.cost > self.spend_hard:
            return ""
        self.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def relevant_ids(self, limit: int) -> list[str]:
        """Listings for a review of the whole search, the most telling first: per item in turn, listings of its kind
        holding the most of its wanted values (read pages before rows), then the newest seen. Race 14: a review window
        of the newest 30 dropped the one fitting row an early search had shown."""
        def wanted(number: int) -> list[str]:
            item = self.items[number - 1]
            out = []
            for need in item.musts + item.told:
                if need in item.dropped or ":" not in need:
                    continue
                value = _squash(need.split(":", 1)[1])
                if value and len(value.split()) <= 3:
                    out.append(value)
            return out

        newest = {ident: n for n, ident in enumerate(self.seen_order)}
        lanes: list[list[str]] = []
        for number in range(1, len(self.items) + 1):
            values = wanted(number)
            pool = [i for i in self.item_kind_rows(number) if not self.sold_out(i)]

            def score(ident: str) -> tuple[Any, ...]:
                text = f" {_squash(_dump(self.full.get(ident) or self.rows.get(ident) or {}))} "
                return (-sum(1 for v in values if f" {v} " in text), ident not in self.full, -newest.get(ident, 0))
            lanes.append(sorted(pool, key=score))
        out: list[str] = []
        for k in range(max((len(l) for l in lanes), default=0)):
            for lane in lanes:
                if k < len(lane) and lane[k] not in out:
                    out.append(lane[k])
        for ident in reversed(self.seen_order):
            if ident not in out and not self.sold_out(ident):
                out.append(ident)
        return out[:limit]

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        for ident in self.relevant_ids(30):
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (the most telling per item first):\n" + "\n".join(records)
        got = self.ask_json(prompt, self.seat(), 3000)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            # more rows when steps are plentiful: a decline is wrong whenever a fitting row was left unread
            plan = [(tool, {target: i}) for i in rows[:4 if self.left() >= 15 else 2]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.roles.get("cart") or ""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _kinds(s) and "enum" not in s]
        target = (next((k for t, k in self.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not self.cart_sets(name):
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.completed or not self.confirmed or self.sold_out(self.confirmed) or not cart or not buy:
            return ""
        self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), self.spend_hard)
        if not self.verdicts.get(self.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), self.spend_hard)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- an ending without a purchase must be proven ----------
    # In a race, ending without a purchase was wrong in about nineteen of twenty such endings: the agent gave up on a
    # listing whose record left a requirement unshown, or on a set it could not finish checking. So before any such
    # ending one review names, per item, the listing that comes closest, and Kyo buys those unless the review proves
    # the store cannot sell the request at all (the request contradicts itself, even the cheapest fitting listing
    # breaks the budget, or the store has nothing of that kind).

    def item_kind_rows(self, number: int) -> list[str]:
        """Listings seen whose category label belongs to the item."""
        cat_key = self.category_key()
        if not cat_key:
            return list(self.rows)
        return [i for i in self.rows if self.item_for_label(str(self.rows[i].get(cat_key) or "")) == number]

    def closest_call(self) -> dict[str, Any]:
        records = []
        ranked = self.relevant_ids(60)
        opened = [i for i in ranked if i in self.full]
        rows = [i for i in ranked if i not in self.full]
        for ident in opened[:24] + rows[:16]:
            record = self.full.get(ident) or self.rows.get(ident)
            records.append(("READ " if ident in self.full else "ROW ") + _dump(_clip(record, 160))[:1100])
        if not records:
            return {}
        prompt = CLOSEST + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nCART NOW:\n" + self.cart_text() + "\n\nLISTING RECORDS:\n" + "\n".join(records)
        # this review decides between buying and ending without a purchase: the heavy model whenever the purse holds
        got = self.ask_json(prompt, [HEAVY, LIGHT], 3000) or {}
        self.trace.append({"kind": "closest", "got": _clip(got, 150)})
        return got

    def proven_empty(self, got: dict[str, Any], plan: dict[int, str]) -> bool:
        """The review's proof that nothing can be bought, as far as code can test it."""
        proof = got.get("proof")
        if not isinstance(proof, dict):
            return False
        kind = str(proof.get("kind") or "").lower()
        number = min(max(1, _count(proof.get("item")) or 1), len(self.items))
        if kind == "contradiction":
            return True
        if kind == "absent":
            return not self.item_kind_rows(number)
        if kind == "budget" and isinstance(self.budget, (int, float)):
            # every listing of that kind read so far is dearer than the budget left once the other items are paid
            others = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in plan.items() if n != number)
            prices = [self.price_of(i) for i in self.item_kind_rows(number) if i in self.full and self.buyable(i)]
            prices = [p for p in prices if p is not None]
            room = float(self.budget) - (others if self.brief.get("budget_scope") != "each" else 0.0)
            # any row of the kind seen at a price inside the room, read or not, may still fit: no proof
            seen = [self.price_of(i) for i in self.item_kind_rows(number) if self.buyable(i)]
            if any(p is not None and p * self.items[number - 1].qty <= room + 1e-9 for p in seen):
                return False
            # a cheapest price tens of times the budget means the two are in different units: no proof
            return bool(prices) and room + 1e-9 < min(prices) * self.items[number - 1].qty < 20 * max(room, 1.0)
        return False

    def hand_over(self) -> bool:
        """The loop ends early so the finish can still buy: the cart lacks a requested item and only the steps that
        setting every line and checking out take are left."""
        if not self.roles.get("buy") or not self.roles.get("cart") or self.rescued == self.version or self.spent():
            return False
        if self.left() < len(self.items) + 2 or self.calls >= self.call_cap - 1:
            return False
        empty = not self.cart_ids() and not self.cart_lines
        return (empty or bool(self.cart_missing())) and self.left() <= len(self.items) + 3

    def cart_missing(self) -> list[int]:
        """Requested items without a cart line, judged by the category of each line (nothing when categories are unknown)."""
        cat_key = self.category_key()
        ids = self.cart_ids()
        if not cat_key or not ids:
            return []
        if len(self.items) == 1:
            return []
        # each line covers one item: lines are matched to items one to one, the best-named pairs first, so two
        # items of one category are both covered by two lines of it; a line that names no item makes the test void
        pairs = []
        for ident in ids:
            scores = self.label_scores(str((self.rows.get(ident) or {}).get(cat_key) or ""))
            if not any(v > 0 for v in scores.values()):
                return []
            pairs += [(score, ident, number) for number, score in scores.items() if score > 0]
        covered: set[int] = set()
        used: set[str] = set()
        for score, ident, number in sorted(pairs, key=lambda p: -p[0]):
            if ident not in used and number not in covered:
                used.add(ident)
                covered.add(number)
        return [n for n in range(1, len(self.items) + 1) if n not in covered]

    def nothing_fits(self, got: dict[str, Any]) -> bool:
        """The closest listing of some item, among at least four of its kind read in full, has a requirement its own
        record contradicts, quoted word for word: then no listing read fits and ending without a purchase stands."""
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            ident = str(entry.get("id") or "").strip()
            read = [i for i in self.item_kind_rows(number) if i in self.full]
            # six read listings of the kind at least: a closest listing picked from four left fitting ones unread
            if len(read) < 6 or ident not in self.rows:
                continue
            record = f" {_squash(_dump(self.full.get(ident) or self.rows.get(ident)))} "
            for fail in _items(entry.get("fails")):
                if not isinstance(fail, dict):
                    continue
                quote = _squash(str(fail.get("record_says") or ""))
                wanted = _squash(str(fail.get("requirement") or "").split(":", 1)[-1])
                # the quote is really in the record and does not itself state the wanted value
                if len(quote) >= 2 and f" {quote} " in record and not (wanted and f" {wanted} " in f" {quote} "):
                    return True
        return False

    def shows_property(self, ident: str, prop: str) -> bool:
        """The listing's record carries an attribute whose key is that property's own name (any spelling: case,
        separators, plural); a longer key ("hoodie style" beside "style") is another property."""
        want = _stem(prop)
        block = _fact_block(self.full.get(ident) or {}) or {}
        return bool(want) and any(_stem(k) == want for k in block)

    def unshown_needs(self, got: dict[str, Any], plan: dict[int, str]) -> list[str]:
        """Requirements the review left unknown on an item's closest listing although at least two other listings of
        that kind read in full carry an attribute for the property: such a requirement is checkable, and a listing
        that does not show it is not known to meet it. A requirement no listing of the kind shows (one the shopper
        says they will check) stays acceptable."""
        found: list[str] = []
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            ident = plan.get(number)
            if not ident:
                continue
            others = [i for i in self.item_kind_rows(number) if i in self.full and i != ident]
            for need in _items(entry.get("unknown")):
                prop = str(need).split(":")[0].strip()
                if not prop or self.shows_property(ident, prop):
                    continue
                if sum(1 for i in others if self.shows_property(i, prop)) >= 2:
                    found.append(f"item {number}: {need}")
        return found

    def rescue_buy(self, chosen: bool = False) -> str:
        """Instead of an ending without a purchase: buy the closest listing of each item unless the review proves the
        store has nothing to sell. Runs once per state of the requirements."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy or self.finished or self.rescued == self.version or self.left() < len(self.items) + 2:
            return ""
        self.rescued = self.version
        # with the purse or the clock used up no review can be had: the code picks alone (an ending without a
        # purchase scored on 0 of Cobalt's 79 race-13 tries, so a code-picked basket is the better bet)
        got = self.closest_call() if not self.spent() else {}
        self.note_closest(got)
        plan: dict[int, str] = {}
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            ident = str(entry.get("id") or "").strip()
            if ident in self.rows and not self.sold_out(ident) and self.buyable(ident) and number not in plan:
                plan[number] = ident
        reviewed = bool(plan)
        # listings read in full whose own attributes meet every requirement code can test outrank any review's proof
        # that nothing fits (race 14: declines with a fitting line in the cart, or with its page read)
        verified = self.verified_plan()
        if self.proven_empty(got, plan) or self.nothing_fits(got):
            if not verified:
                self.trace.append({"kind": "decline_proven", "proof": _clip(got.get("proof"), 80)})
                self.decline_proven = self.version
                return ""
            self.trace.append({"kind": "rescue_verified", "plan": verified, "over": "proof"})
            plan, reviewed = dict(verified), True
        elif not plan and verified:
            self.trace.append({"kind": "rescue_verified", "plan": verified, "over": "no review"})
            plan, reviewed = dict(verified), True
        # the lines the agent itself carted stand for items the review named nothing for, unless the agent took them
        # off again or the last check of a cart holding them found a problem
        rejected = self.rejected_lines()
        for ident in [i for i in self.built if i in self.rows and not self.sold_out(i) and self.buyable(i) and i not in self.dropped and i not in rejected]:
            number = self.item_for_label(str(self.rows[ident].get(self.category_key() or "") or "")) if len(self.items) > 1 else 1
            if number and number not in plan:
                plan[number] = ident
        # A requirement the closest listing does not show while other listings of its kind do show that property:
        # after a thorough search (six listings of the kind read) no read listing states the wanted value, and ending
        # without a purchase stands; after a short one it is unknown, and the closest listing is bought (race 13:
        # Cobalt v1's endings without a purchase came mostly from code, after five or so reads per item, none right).
        unshown = self.unshown_needs(got, plan)
        if unshown:
            thorough = all(sum(1 for i in self.item_kind_rows(n) if i in self.full) >= 6
                           for n in {min(max(1, _count(u.split(":")[0].replace("item", "")) or 1), len(self.items)) for u in unshown})
            self.trace.append({"kind": "decline_unshown" if thorough else "rescue_unshown", "needs": unshown[:4]})
            if thorough and not verified:
                self.decline_proven = self.version
                return ""
        # the code's own picks stand in only for an ending forced by the limits, or when a review found something: the
        # agent's own considered ending, after the decline review read the records and found no fit, is not overruled
        # by a pick the code made without a review
        considered = chosen and self.audited and not self.leads and not reviewed
        # a basket held together by model codes: it replaces a complete plan whose lines fit no one device, and fills
        # the items an incomplete plan lacks
        baskets = self.compat_baskets() if len(self.items) > 1 and not considered else []
        if baskets:
            basket = baskets[0][1]
            if len(plan) == len(self.items) and not self.compat_consistent(dict(plan)):
                self.trace.append({"kind": "rescue_compat", "plan": basket, "code": baskets[0][2]})
                plan = dict(basket)
            elif len(plan) < len(self.items):
                trial = {**basket, **plan}
                if self.compat_consistent(trial):
                    self.trace.append({"kind": "rescue_compat_fill", "plan": trial})
                    plan = trial
                else:
                    plan = dict(basket)
        # without a review, the replacements the sell-out review named stand in, cheapest first (a reviewed list beats
        # the code's own pick)
        if not reviewed and not considered:
            for ident in getattr(self, "swap_fits", []):
                number = self.item_for_label(str((self.rows.get(ident) or {}).get(self.category_key() or "") or "")) if len(self.items) > 1 else 1
                if number and number not in plan and self.buyable(ident) and not self.sold_out(ident) and ident not in self.dropped:
                    plan[number] = ident
        for number in range(1, len(self.items) + 1):
            if number not in plan and not considered:
                pick = self.code_pick(number, plan)
                if pick:
                    plan[number] = pick
        if len(plan) < len(self.items):
            self.trace.append({"kind": "rescue_short", "plan": plan})
            return ""
        lines = {ident: self.items[number - 1].qty for number, ident in plan.items()}
        if len(lines) < len(plan):
            return ""
        said = self.order_lines(lines)
        self.trace.append({"kind": "rescue", "lines": lines, "said": said[:200]})
        if not said:
            return ""
        return ("Not run (Kyo acted instead): ending without a purchase is right only when the store provably cannot sell the request, "
                "so Kyo put the closest listing of each item in the cart (" + ", ".join(lines) + ") and checked out. Store reply: " + said)

    def keyed_marks(self, ident: str, number: int) -> tuple[int, int]:
        """(requirements the listing's own attributes state, requirements an attribute named for them contradicts)."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        if not block:
            return 0, 0
        item = self.items[number - 1]
        own = _stem(f"{item.name} {item.shelf or ''}")
        met = against = 0
        for need in item.musts + item.told:
            if need in item.dropped or ":" not in need:
                continue
            prop, value = (part.strip() for part in need.split(":", 1))
            keys = [k for k in block if _stem(k) == _stem(prop) or (_stem(k) - own == _stem(prop) and _stem(k) & own)]
            if not keys:
                continue
            limit = self.limit_of(value)
            if limit:
                figs = [f for f in (self.figure_of(str(block[k]), limit[2]) for k in keys) if f is not None]
                if figs:
                    ok = any((f >= limit[1] - 1e-9) if limit[0] == "min" else (f <= limit[1] + 1e-9) for f in figs)
                    met, against = met + ok, against + (not ok)
                continue
            wanted = _squash(value)
            if not wanted or re.search(r"\d", wanted) or len(wanted.split()) > 3 \
                    or re.search(r"\b(?:not|no|non|without|except|other|unlike|same|match\w*|any|either|or|than|like)\b", wanted):
                continue
            ok = any(self.holds_words(str(block[k]), wanted) for k in keys)
            met, against = met + ok, against + (not ok)
        return met, against

    def line_meets(self, ident: str, number: int) -> bool:
        return self.line_tests(ident, number) > 0

    def line_tests(self, ident: str, number: int) -> int:
        """How many of the item's requirements the listing's own record meets, 0 when one fails or cannot be tested: a plain value stated by
        the attribute named for its property (or, without one, by the record), a one-sided limit kept by that
        attribute's figure. A requirement of any other form (a list, an exclusion, a tie to another listing) leaves
        the listing unproven."""
        record = self.full.get(ident)
        block = _fact_block(record) if record else None
        if not block or not self.buyable(ident):
            return 0
        tested = 0
        item = self.items[number - 1]
        own = _stem(f"{item.name} {item.shelf or ''}")
        text = f" {_squash(_dump(record))} "
        pointers = {need for _, need in self.refs}
        for need in item.musts + item.told:
            if need in item.dropped or ":" not in need:
                continue
            if need in pointers and need not in self.ref_values:
                return 0
            prop, value = (part.strip() for part in need.split(":", 1))
            wanted = _squash(value)
            keys = [k for k in block if _stem(k) == _stem(prop) or (_stem(k) - own == _stem(prop) and _stem(k) & own)]
            limit = self.limit_of(value)
            if limit:
                figs = [self.figure_of(str(block[k]), limit[2]) for k in keys]
                figs = [f for f in figs if f is not None]
                if not figs or not any((f >= limit[1] - 1e-9) if limit[0] == "min" else (f <= limit[1] + 1e-9) for f in figs):
                    return 0
                tested += 1
                continue
            plain = bool(wanted) and not re.search(r"\d", wanted) and len(wanted.split()) <= 3 \
                and not re.search(r"\b(?:not|no|non|without|except|other|unlike|same|match\w*|any|either|or|than|like)\b", wanted)
            if not plain:
                return 0
            # strict, since this outranks a review: without the property's own key, a key holding its words with others
            # in front ("fa pattern") must state the value, and free text must hold the value's words whole
            related = keys or [k for k in block if _stem(prop) and _stem(prop) < _stem(k)]
            if related:
                if not any(self.holds_words(str(block[k]), wanted) for k in related):
                    return 0
            elif f" {wanted} " not in text:
                return 0
            tested += 1
        return tested

    def verified_plan(self) -> dict[int, str]:
        """For every requested item, a listing read in full that meets every requirement by its own record (the
        cheapest such listing; the agent's own cart line first when it qualifies), inside the budget and fitting one
        model code when codes tie the items; {} when some item has none."""
        plan: dict[int, str] = {}
        cart = set(self.cart_ids()) | set(self.built)
        for number in range(1, len(self.items) + 1):
            rejected = self.rejected_lines()
            each = self.budget if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") == "each" else None
            pool = [i for i in self.item_kind_rows(number) if i in self.full and i not in self.dropped and i not in rejected
                    and (each is None or (self.price_of(i) or 0.0) * self.items[number - 1].qty <= float(each) + 1e-9)
                    and self.line_meets(i, number)]
            if not pool:
                return {}
            pool.sort(key=lambda i: (i not in cart and self.brief.get("pick") not in ("cheapest", "most", "least"),
                                     self.price_of(i) if self.price_of(i) is not None else 1e12))
            plan[number] = pool[0]
        if len(self.items) > 1 and not self.compat_consistent(plan):
            return {}
        if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each":
            total = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in plan.items())
            if total > float(self.budget) + 1e-9:
                return {}
        return plan

    def repick(self) -> str:
        """Once per state of the requirements: when they changed since the cart was set and the request picks by price
        or by a measure, the closest-listing review chooses again over every record read, and a different choice is
        carted and checked out. Returns the store's reply, or "" when nothing changed or nothing could be done."""
        sig = self.req_sig()
        if self.brief.get("pick") not in ("cheapest", "most", "least") or not self.cart_touched or sig == self.cart_sig \
                or self.repicked == sig or self.spent() or self.left() < len(self.items) + 2:
            return ""
        self.repicked = sig
        got = self.closest_call()
        plan: dict[int, str] = {}
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            ident = str(entry.get("id") or "").strip()
            fails = [f for f in _items(entry.get("fails")) if isinstance(f, dict)]
            # only a listing whose page was read and that shows every requirement replaces a carted line
            if ident in self.full and not fails and not _items(entry.get("unknown")) and not self.sold_out(ident) \
                    and self.buyable(ident) and number not in plan:
                plan[number] = ident
        cart = set(self.cart_ids())
        self.trace.append({"kind": "repick", "plan": plan, "cart": sorted(cart)})
        if len(plan) < len(self.items) or len(set(plan.values())) < len(plan) or set(plan.values()) == cart:
            return ""
        lines = {ident: self.items[number - 1].qty for number, ident in plan.items()}
        if self.check_cart(list(lines)):
            return ""
        said = self.order_lines(lines)
        return ("Kyo re-picked the cart for the changed requirements (" + ", ".join(lines) + ") and checked out. Store reply: " + said) if said else ""

    # ---------- model-code compatibility, worked out in code ----------
    # Race-13 multi-item baskets tie the lines together by model codes: a device's page says "Model: AB-1210", an
    # accessory's says "Compatible with: every AB-12xx model, except AB-1220" or lists the codes; a line "Part X also
    # fits: ..." speaks for part X, not for the listing. Cobalt v1 passed 3 of 36 such episodes, mostly ending with an
    # empty cart after running out of turns; the rows of a search already carry these sentences.
    CODE = r"[A-Z]{2,5}-?\d{2,6}[0-9xX]*"
    NAME_CODE = r"\b[A-Z][a-z]{2,11} \d{3,5}\b"

    def compat_text(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        facts = _fact_block(record) or {}
        return " | ".join([str(v) for v in record.values() if isinstance(v, str)] + [f"{k}: {v}" for k, v in facts.items()])

    def compat_codes(self, ident: str) -> tuple[set[str], list[tuple[str, set[str]]], set[str]]:
        """The listing's codes (see raw_codes) with each model's other names folded into one: a listing that names
        two codes as its own ("Model number: X. Also sold as Name 123.") makes them one model, written as its plain
        code, wherever any listing names either."""
        alias = self.code_aliases()
        own, families, listed = self.raw_codes(ident)
        fold = lambda codes: {alias.get(c, c) for c in codes}
        return fold(own), [(prefix, fold(skip) | skip) for prefix, skip in families], fold(listed)

    def code_aliases(self) -> dict[str, str]:
        """Model names with a number ("Name 123") mapped to the plain code ("AB-1230") of the listing that names both
        as its own. Only such names fold, each into its listing's one plain code, and a name two listings tie to
        different plain codes stays as it is; plain codes never fold into one another (a spec line names plain
        codes of its own that two different models share)."""
        mark = (len(self.rows), len(self.full))
        if getattr(self, "alias_mark", None) == mark:
            return self.alias_map
        norm = lambda c: re.sub(r"[\s-]", "", c.upper())
        ties: dict[str, set[str]] = {}
        for ident in self.seen_order:
            own = self.raw_codes(ident)[0]
            if len(own) < 2:
                continue
            text = self.compat_text(ident)
            written = {norm(c) for c in re.findall(self.CODE, text)}
            names = {norm(c) for c in re.findall(self.NAME_CODE, text)} - written
            spoken = {c for c in own if c in names}
            plain = own - spoken
            if len(plain) == 1:
                for name in spoken:
                    ties.setdefault(name, set()).update(plain)
        self.alias_mark = mark
        self.alias_map = {name: next(iter(plain)) for name, plain in ties.items() if len(plain) == 1}
        return self.alias_map

    def raw_codes(self, ident: str) -> tuple[set[str], list[tuple[str, set[str]]], set[str]]:
        """(codes the listing names about itself, families as (prefix, exceptions), all codes it names as fitting or
        being), read by the text's structure rather than its verbs (each world words this differently: "Model: X" and
        "Compatible with every X-45xx model, except ..." in one race, "Model ref. X" and "Goes with X, Y" in another).
        A sentence that names another part ("part <code>") speaks about that part, not this listing: its codes are
        skipped. Codes in the listing's own title are its own part or model names."""
        own: set[str] = set()
        families: list[tuple[str, set[str]]] = []
        listed: set[str] = set()
        norm = lambda c: re.sub(r"[\s-]", "", c.upper())
        record = self.full.get(ident) or self.rows.get(ident) or {}
        titles = [v for k, v in record.items() if isinstance(v, str) and self.title_field(k)] if record else []
        title_codes = {norm(c) for v in titles for c in re.findall(self.CODE, v)}

        def found(text: str, names: bool) -> list[str]:
            # plain codes, and model names with a number ("Name 123") where the sentence allows them
            hits = [(m.start(), norm(m.group(0))) for m in re.finditer(self.CODE, text)]
            if names:
                hits += [(m.start(), norm(m.group(0))) for m in re.finditer(self.NAME_CODE, text)]
            return [c for _, c in sorted(hits)]

        # a sentence ends at a stop followed by a capitalised word, never before a code ("Model ref. <code>.")
        before: list[str] = []
        for sentence in re.split(r"(?<=[.;!?])\s+(?=[A-Z][a-z])|\n|•|\s\|\s", self.compat_text(ident)):
            plain = re.findall(self.CODE, sentence)
            # a model name with a number is a code only beside a plain code, in the same sentence or right after a
            # sentence naming one ("Model number: X. Also sold as Name 123." reads as one listing's two names), or
            # right after a colon ("Fits: Name 123, X"); never in the listing's own title, where such words are a
            # product's name ("Brand 900 Pro")
            named = re.findall(self.NAME_CODE, sentence)
            listy = re.search(r":\s*" + self.NAME_CODE, sentence)
            in_title = any(sentence.strip() and sentence.strip() in t for t in titles)
            allow = bool(named) and not in_title and (bool(plain) or bool(listy) or (len(named) == 1 and len(before) == 1))
            before = [c for c in plain if norm(c) not in title_codes]
            codes = found(sentence, allow)
            if not codes:
                continue
            others = {norm(c) for c in re.findall(r"\bpart\s+(" + self.CODE + ")", sentence, re.I)} - title_codes
            if others:
                continue
            pieces = re.split(r"\bexcept\b", sentence, maxsplit=1, flags=re.I)
            head, tail = pieces[0], (pieces[1] if len(pieces) > 1 else "")
            tails = set(found(tail, allow))
            heads = [c for c in found(head, allow) if c not in title_codes]
            for code in heads:
                if re.search(r"X+$", code):
                    families.append((re.sub(r"X+$", "", code), tails))
                else:
                    listed.add(code)
            if len(heads) == 1 and not tails and not re.search(r"X+$", heads[0]):
                own.add(heads[0])
        return own, families, listed

    @staticmethod
    def title_field(key: str) -> bool:
        """A record field that names the listing itself (its title, name or heading)."""
        return bool(set(_words(key).split()) & {"title", "name", "heading", "headline"})

    @staticmethod
    def code_fits(code: str, families: list[tuple[str, set[str]]], listed: set[str]) -> bool:
        return code in listed or any(code.startswith(prefix) and code not in skip for prefix, skip in families)

    @staticmethod
    def code_family(code: str) -> str:
        return re.match(r"[A-Z]*", code).group(0)

    def shared_families(self, codes: dict[int, dict[str, tuple[set[str], list[tuple[str, set[str]]], set[str]]]]) -> set[str]:
        """Code families named on the listings of two requested items or more: a model number that only one kind of
        product carries (a garment's own model number) ties nothing together."""
        seen: dict[str, set[int]] = {}
        for number, per in codes.items():
            for _, fams, listed in per.values():
                for code in listed | {p for p, _ in fams}:
                    seen.setdefault(self.code_family(code), set()).add(number)
        return {fam for fam, items in seen.items() if fam and len(items) >= 2}

    def coded(self, entry: tuple[set[str], list[tuple[str, set[str]]], set[str]], fams: set[str]) -> tuple[list[tuple[str, set[str]]], set[str]]:
        _, families, listed = entry
        return [f for f in families if self.code_family(f[0]) in fams], {c for c in listed if self.code_family(c) in fams}

    def compat_baskets(self) -> list[tuple[float, dict[int, str], str]]:
        """Complete baskets whose coded lines all fit one model code: (total, item -> listing, code), the most required
        values held first, then cheapest; each line the best buyable seen listing of its item's kind that names that
        code (items whose listings name no shared code take their best listing)."""
        if len(self.items) < 2 or not self.category_key():
            return []
        linked = self.model_baskets()
        if linked:
            return linked
        codes = {n: {i: self.compat_codes(i) for i in self.item_kind_rows(n) if self.buyable(i)} for n in range(1, len(self.items) + 1)}
        fams = self.shared_families(codes)
        if not fams:
            return []
        coded_items = [n for n, per in codes.items() if per and 2 * sum(1 for e in per.values() if any(self.coded(e, fams))) >= len(per)]
        if len(coded_items) < 2:
            return []
        candidates = {c for n in coded_items for e in codes[n].values() for c in self.coded(e, fams)[1]}
        baskets = []
        for code in candidates:
            plan: dict[int, str] = {}
            total = 0.0
            for number in range(1, len(self.items) + 1):
                if number in coded_items:
                    pool = [i for i, e in codes[number].items() if self.code_fits(code, *self.coded(e, fams))]
                else:
                    pool = list(codes[number])
                pick = self.best_of(number, pool)
                if not pick:
                    break
                plan[number] = pick
                total += (self.price_of(pick) or 0.0) * self.items[number - 1].qty
            if len(plan) == len(self.items) and (not isinstance(self.budget, (int, float)) or total <= float(self.budget) + 1e-9
                                                 or self.brief.get("budget_scope") == "each"):
                baskets.append((total, plan, code))
        baskets.sort(key=lambda b: (-sum(self.met_count(n, i) for n, i in b[1].items()), b[0]))
        return baskets

    def met_count(self, number: int, ident: str) -> int:
        item = self.items[number - 1]
        text = f" {_squash(self.compat_text(ident))} "
        return sum(1 for m in item.musts + item.told if m not in item.dropped and (w := _squash(m.split(':', 1)[-1])) and f" {w} " in text)

    def best_of(self, number: int, pool: list[str]) -> str:
        """Of an item's candidate listings: the most required values held, then the cheapest."""
        pool = [i for i in pool if self.buyable(i)]
        return min(pool, key=lambda i: (-self.met_count(number, i), self.price_of(i) if self.price_of(i) is not None else 1e12)) if pool else ""

    def compat_note(self) -> str:
        """A line for the status: the best complete baskets the model codes allow, from what has been seen."""
        mark = (len(self.rows), len(self.full), self.version)
        if getattr(self, "compat_mark", None) == mark:
            return self.compat_said
        self.compat_mark, self.compat_said = mark, ""
        baskets = self.compat_baskets()
        if not baskets:
            return ""
        shown = []
        for total, plan, code in baskets[:2]:
            shown.append(f"code {code}: " + ", ".join(f"item {n} {i}" for n, i in sorted(plan.items())) + f" (total {total:g})")
        how = (" A word search for a model's code within an item's category lists the parts whose pages name it" if self.codes_searchable() else
               " This store's search finds no codes: page through each item's category and read the codes in its rows")
        self.compat_said = ("[model codes: complete baskets whose lines all fit one model, from the rows seen; check each line's requirements on its page."
                            + how + "] " + " | ".join(shown))
        return self.compat_said

    def compat_consistent(self, plan: dict[int, str]) -> bool:
        """Every coded line of the plan fits one model code (true when fewer than two lines name a shared code)."""
        if len(plan) < len(self.items):
            return False
        return not self.compat_conflict(plan)

    def compat_conflict(self, plan: dict[int, str]) -> str:
        """Why the plan's coded lines fit no one model code ("" when they do, or when codes are not in play)."""
        linked = self.model_conflict(plan)
        if linked:
            return linked
        everything = {n: {i: self.compat_codes(i) for i in self.item_kind_rows(n)} for n in range(1, len(self.items) + 1)}
        fams = self.shared_families(everything)
        if not fams:
            return ""
        lines = {n: self.coded(self.compat_codes(i), fams) for n, i in plan.items()}
        lines = {n: e for n, e in lines.items() if e[0] or e[1]}
        if len(lines) < 2:
            return ""
        pool = {c for _, listed in lines.values() for c in listed}
        if not pool:
            # every coded line names only families ("every AB-45xx model"): no concrete code to test them against
            return ""
        if any(all(self.code_fits(code, f, l) for f, l in lines.values()) for code in pool):
            return ""
        said = "; ".join(f"{plan[n]} names {', '.join(sorted(l)[:4]) or 'only code families'}" for n, (f, l) in sorted(lines.items()))
        return "these cart lines name no model code in common (" + said + ")"

    # ---------- named models and part series ----------
    # Race 16 ties baskets together two more ways: a device says "Model: AB 1230." and a part lists "Compatible with:
    # AB 1230, AB 1240." (a model name with a space, which CODE does not read); or a device says "Takes part series
    # XY4" and each part's title ends with its own code in brackets, "[XY4-1211]", whose part before the dash is its
    # series. A sentence "Part X also fits: ..." speaks for part X only. In race 16, 41 of 90 tasks used series and
    # 61 used model names; the field passed under 10% of the series ones.
    MODEL_NAME = r"\b(?:[A-Z]{2,5}[- ]?\d{3,5}|[A-Z][a-z]{2,11} \d{3,5})\b"
    SERIES = r"\b[A-Z]{1,4}\d{1,3}\b"

    def model_facts(self, ident: str) -> dict[str, set[str]]:
        """A listing's model names, the part series it takes, the series of its own part code, and the model names
        it fits (with exceptions)."""
        mark = (ident, ident in self.full, len(str(self.full.get(ident) or self.rows.get(ident) or "")))
        memo = self.model_memo.get(ident)
        if memo and memo[0] == mark:
            return memo[1]
        norm = lambda c: re.sub(r"[\s-]", "", c.upper())
        record = self.full.get(ident) or self.rows.get(ident) or {}
        titles = " ".join(v for k, v in record.items() if isinstance(v, str) and self.title_field(k)) if record else ""
        text = self.compat_text(ident)
        out: dict[str, set[str]] = {"names": set(), "takes": set(), "series": set(), "fits": set(), "except": set()}
        for code in re.findall(r"\[([A-Z]{1,4}\d{1,3})-\d{2,6}\]", titles):
            out["series"].add(code.upper())
        for sentence in re.split(r"(?<=[.;!?])\s+(?=[A-Z])|\n|\s\|\s", text):
            low = sentence.lower()
            if re.search(r"\bpart\s+[A-Z0-9-]{3,}\s+(?:also|too)\s+fits\b", sentence, re.I):
                continue
            if re.search(r"\bmodel(?:\s+(?:number|no\.?|name|ref\.?))?\s*:", low) or re.search(r"\balso sold as\b", low):
                out["names"].update(norm(c) for c in re.findall(self.MODEL_NAME, sentence))
            if re.search(r"\b(?:takes|accepts|uses|fits)\b[^.]*\bseries\b", low) or re.search(r"\bpart series\b", low):
                tail = re.split(r"(?i)\bseries\b", sentence, maxsplit=1)[-1]
                out["takes"].update(c.upper() for c in re.findall(self.SERIES, tail)[:3])
            if re.search(r"\b(?:compatible with|fits|works with|for use with)\b\s*:", low):
                body = re.split(r"(?i)\b(?:compatible with|fits|works with|for use with)\b\s*:", sentence, maxsplit=1)[-1]
                parts = re.split(r"(?i)\b(?:except|excluding|but not)\b", body, maxsplit=1)
                out["fits"].update(norm(c) for c in re.findall(self.MODEL_NAME, parts[0]))
                if len(parts) > 1:
                    out["except"].update(norm(c) for c in re.findall(self.MODEL_NAME, parts[1]))
        out["names"] -= out["fits"]
        self.model_memo[ident] = (mark, out)
        return out

    def model_fit(self, part: str, device: str) -> bool | None:
        """Does the part go with the device by model names or by part series? None when neither says."""
        p, d = self.model_facts(part), self.model_facts(device)
        if p["fits"] and d["names"]:
            if d["names"] & p["except"]:
                return False
            return bool(d["names"] & p["fits"])
        if d["takes"] and p["series"]:
            return bool(d["takes"] & p["series"])
        return None

    def model_conflict(self, plan: dict[int, str]) -> str:
        """Why two lines of the plan do not go together by model names or part series ("" when nothing says so)."""
        lines = list(plan.items())
        for a, ia in lines:
            for b, ib in lines:
                if a == b:
                    continue
                if self.model_fit(ia, ib) is False and self.model_fit(ib, ia) is not True:
                    pf, df = self.model_facts(ia), self.model_facts(ib)
                    if pf["fits"]:
                        why = f"{ia} is compatible with {', '.join(sorted(pf['fits'])[:4])}, not with {ib} ({', '.join(sorted(df['names'])[:2])})"
                    else:
                        why = f"{ib} takes part series {', '.join(sorted(df['takes']))}, but {ia} is series {', '.join(sorted(pf['series']))}"
                    return why
        return ""

    def model_baskets(self) -> list[tuple[float, dict[int, str], str]]:
        """Complete baskets whose lines go together by model names or part series: for each device listing seen, the
        best buyable listing of each other item that fits it."""
        if len(self.items) < 2 or not self.category_key():
            return []
        pools = {n: [i for i in self.item_kind_rows(n) if self.buyable(i)] for n in range(1, len(self.items) + 1)}
        out: list[tuple[float, dict[int, str], str]] = []
        for dn, devices in pools.items():
            for dev in devices[:20]:
                df = self.model_facts(dev)
                if not (df["names"] or df["takes"]):
                    continue
                plan = {dn: dev}
                tied = 0
                for n, pool in pools.items():
                    if n == dn:
                        continue
                    fitting = [i for i in pool if self.model_fit(i, dev) is True]
                    if fitting:
                        tied += 1
                        pick = self.best_of(n, fitting)
                    else:
                        if any(self.model_fit(i, dev) is False for i in pool):
                            pick = ""
                        else:
                            pick = self.best_of(n, pool)
                    if not pick:
                        break
                    plan[n] = pick
                if len(plan) != len(self.items) or not tied:
                    continue
                total = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in plan.items())
                if isinstance(self.budget, (int, float)) and self.brief.get("budget_scope") != "each" and total > float(self.budget) + 1e-9:
                    continue
                label = ", ".join(sorted(df["names"]) or [f"series {', '.join(sorted(df['takes']))}"])
                out.append((total, plan, label))
        out.sort(key=lambda b: (-sum(self.met_count(n, i) for n, i in b[1].items()), b[0]))
        return out

    def rejected_lines(self) -> set[str]:
        """Listings whose latest single-line check found a problem."""
        latest = {}
        for key, ok in self.verdicts.items():
            ids = [i for i in key.split("#")[0].split("|") if i]
            if len(ids) == 1:
                latest[ids[0]] = ok
        return {i for i, ok in latest.items() if not ok}

    def code_pick(self, number: int, plan: dict[int, str]) -> str:
        """The code's own choice for an item no review named: of the listings of its kind seen buyable, the one whose
        record holds the most of the item's required values, opened ones first, then the cheapest that keeps the cart
        inside the budget."""
        others = sum((self.price_of(i) or 0.0) * self.items[n - 1].qty for n, i in plan.items() if n != number)
        room = (float(self.budget) - (others if self.brief.get("budget_scope") != "each" else 0.0)) if isinstance(self.budget, (int, float)) else None
        item = self.items[number - 1]
        wants = [_squash(m.split(":", 1)[-1]) for m in item.musts + item.told if m not in item.dropped]
        best: tuple[Any, ...] | None = None
        # a line the agent took off the cart, or one a check found wrong, only when nothing else is left
        rejected = self.rejected_lines()
        for ident in self.item_kind_rows(number):
            if ident in plan.values() or not self.buyable(ident):
                continue
            price = self.price_of(ident)
            if room is not None and price is not None and price * item.qty > room + 1e-9:
                continue
            text = f" {_squash(_dump(self.full.get(ident) or self.rows.get(ident) or {}))} "
            met = sum(1 for w in wants if w and f" {w} " in text)
            # an attribute named for a required property that states another value rules the listing out before any
            # word count (race 14: a listing won on a wanted word in its description while its own attribute said another value)
            keyed, against = self.keyed_marks(ident, number)
            score = (ident not in self.dropped and ident not in rejected, not against, keyed, met, ident in self.full, -(price if price is not None else 1e12))
            if best is None or score > best[0]:
                best = (score, ident)
        if best:
            self.trace.append({"kind": "code_pick", "item": number, "id": best[1], "score": best[0][:3]})
        return best[1] if best else ""

    def note_closest(self, got: dict[str, Any]) -> None:
        """Keep what the closest-listing review said, for the text of an ending without a purchase."""
        proof = got.get("proof") if isinstance(got.get("proof"), dict) else None
        if proof and str(proof.get("why") or "").strip():
            self.decline_why = str(proof["why"]).strip()[:300]
        for entry in _items(got.get("items")):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            if ident not in self.rows:
                continue
            title = self.title_of(ident)
            fails = [str(f.get("requirement") or "") for f in _items(entry.get("fails")) if isinstance(f, dict) and f.get("requirement")]
            self.closest_named = (ident, title, "; ".join(fails[:2]))
            if not self.decline_why and fails:
                self.decline_why = f"No listing meets every requirement: the closest one misses {'; '.join(fails[:2])}."
            break

    def title_of(self, ident: str) -> str:
        """A listing's name as its search row gives it: the shortest text of three or more words in the row that is
        not its category label (a description runs longer than the name)."""
        row = self.rows.get(ident) or self.full.get(ident) or {}
        cat = self.category_key()
        texts = [str(v).strip() for k, v in row.items() if isinstance(v, str) and k not in (self.id_key, cat) and len(str(v).split()) >= 3]
        return min(texts, key=len)[:120] if texts else ident

    def order_lines(self, lines: dict[str, int]) -> str:
        """Set the cart to exactly these lines and check out (one retry after a refusal that leaves the lines buyable);
        returns the store's replies, or "" when nothing was ordered."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        drop = [i for i in self.cart_ids() if i not in lines]
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + 1 > self.left():
            return ""
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ""
            self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
        # a code-made basket looks up the parts its pages name too, as the model's checkout does
        if len(self.items) > 1 and self.left() > len(self.items) + 5:
            self.surface_parts()
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies: list[str] = []
        before = self.version
        bought = self.buys
        for attempt in range(2):
            # a refusal that changed nothing Kyo tracks (no new budget, requirement or stock) is retried once; one
            # that changed something goes back to the agent to act on
            if attempt and (self.left() < 1 or self.finished or self.version != before or any(self.sold_out(i) for i in lines)):
                break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go or self.left() < 1:
                break
            reply = self.act(buy, args)
            if reply is None:
                break
            self.buys += 1
            if not reply.get("finished"):
                self.refused_at = self.step
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        return " | then: ".join(replies) if self.buys > bought else ""

    # ---------- tool use ----------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        fixed = self.brief.get("buy_args") if isinstance(self.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        fixed = self.brief.get("refuse_args") if isinstance(self.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view")}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get("cart"), self.roles.get("cart_view"), self.roles.get("offer")}
        tools = [t for t in self.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The free-text arguments of an ending without a purchase, written by decline_text (the request's own words for
        it, the closest listing only when the request asks for it)."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        fixed = set(self.brief.get("refuse_args") or {}) | set(self.brief.get("buy_args") or {})
        texts = [k for k, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec and k not in fixed]
        required = set(schema.get("required") or [])
        for key in texts:
            # a required text, one the agent wrote, or the tool's only free text (race 13's ending's justification)
            if key in required or str(out.get(key) or "").strip() or len(texts) == 1:
                out[key] = self.decline_text(str(out.get(key) or ""))[: props[key].get("maxLength", 700)]
        return out

    def decline_text(self, said: str) -> str:
        """An ending's text. Race 13's correct endings without a purchase began with the words the request gave for it
        and, where the request asked for it, named the closest listing; a text with neither missed a stated step."""
        said = " ".join(said.split())
        words = str(self.brief.get("decline_words") or "").strip().strip("\"'").strip()
        mode = self.brief.get("quiet_decline")
        named = [i for i in self.rows if i in said]
        if mode is True and named:
            # the request forbids naming alternatives: keep only the sentences that name no listing
            kept = [s for s in re.split(r"(?<=[.;!?])\s+", said) if not any(i in s for i in self.rows)]
            said = " ".join(kept)
        if not said:
            said = self.decline_why or "No listing in the store meets every requirement."
        if mode is False and not any(i in said for i in self.rows) and self.closest_named:
            ident, title, misses = self.closest_named
            price = self.price_of(ident)
            said += f" Closest listing: {title} ({ident}" + (f", {price:g} {self.brief.get('budget_unit') or ''}".rstrip() if price is not None else "") + ")" \
                + (f", which misses {misses}." if misses and misses not in said else ".")
        if words and _words(words) not in _words(said[: len(words) + 40]):
            said = words + ("" if words[-1:] in ".:!?" else ".") + " " + said
        return said.strip()

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1 or (name != self.roles.get("cart") and (name, free[0]) not in self.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return value.strip(), ""
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any(k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)"

        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), "")
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        self.cart_version = self.version
        self.cart_sig = self.req_sig()
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any) -> str:
        if isinstance(observation, dict):
            # search rows keep their opening words (a row's codes and "fits" sentence come first); a listing's own
            # page keeps more, since its attributes decide the pick
            observation = {k: [_clip(r, ROW_CHARS) if isinstance(r, dict) else r for r in v] if isinstance(v, list) else v
                           for k, v in observation.items()}
        text = _dump(_clip(observation, SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + "...(cut)"
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        opener, joiner, target = self.roles.get("open"), self.roles.get("compare"), self.single_target(self.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        # only into a side-by-side view that shows what the opens would (attributes)
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.rows and ident not in self.gone:
                picks.append((index, ident))
        picks = picks[: self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({"id": call["id"], "joined": True})
            else:
                out.append(call)
        self.trace.append({"kind": "joined", "ids": [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                out.append((cid, "Shown in the side-by-side result above: Kyo opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Not run (no step used): too few steps are left for that; use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Not run (no step used): the arguments were not one JSON object."))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], "args", faults)
            if faults:
                out.append((cid, "Not run (no step used): " + "; ".join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get("open"), self.roles.get("offer")) and str(args.get(target)) in self.gone:
                out.append((cid, f"Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."))
                continue
            if name == self.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, "Not run (no step used): send the finishing call on its own, after reading your other results."))
                continue
            # until the first checkout was tried, two more steps stay free for it: a checkout a store note turns down
            # (a line sold out at the till) needs a cart change and a second checkout
            keep = self.tail() + (0 if getattr(self, "buy_tried", False) or name == self.roles.get("cart") else 2)
            if not ending and self.left() <= keep and (self.roles.get("buy") or self.roles.get("refuse")):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get("cart") and mark in self.recent:
                out.append((cid, "Not run (no step used): you just made this same call and nothing changed; its result is above."))
                continue
            readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and cached[0] == self.version:
                out.append((cid, f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.roles.get("compare") and self.shown_already(name, str(args.get(target))):
                out.append((cid, f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, "Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                                + "\n".join(f"{i}: {self.record_of(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    before = self.req_sig()
                    wider = self.wider_hold() or self.change_hold()
                    if wider:
                        out.append((cid, wider + "\nCart now: " + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3:
                        faults = []
                        view_args = _shape({}, self.schemas.get(self.roles["cart_view"]), "args", faults)
                        view = self.act(self.roles["cart_view"], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get("observation")
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles["cart_view"], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() + "\nApply the change, fix the cart if needed, then check out."))
                                continue
                    swap = self.swap_hint()
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    if self.req_sig() != before and self.left() > 3:
                        # a note read on Kyo's own re-reads changed the task (race 13: a dropped requirement and a new
                        # budget arrived there, and the old cart was bought anyway)
                        out.append((cid, "Not run yet (no step used). A store note arrived while Kyo re-read the cart: " + " | ".join(self.notes[-2:])
                                    + "\nApply the change to the picks (the ledger below is updated), fix the cart, then check out.\nCart now: " + self.cart_text()))
                        continue
                    clash = self.budget_hold() or self.value_clash() or self.compat_hold()
                    if clash:
                        out.append((cid, "Not run yet (no step used). " + clash + "\nCart now: " + self.cart_text()))
                        continue
                    surfaced = self.surface_parts()
                    if surfaced:
                        read.append(f"Kyo looked up the parts the cart's pages name ({', '.join(surfaced)}), one search each.")
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                elif self.is_refuse(name, args):
                    if self.left() > 5:
                        review = self.audit()
                        if review:
                            out.append((cid, "Not run yet (no step used). " + review))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads)})
                            out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = self.leads[0]
                            self.check_cart([first], self.spend_hard)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
                # too late for the review: a decline here usually throws away a built cart because time ran out
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart() or self.rescue_buy(chosen=True)
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if name == self.roles.get("compare"):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
            if name in readers - {self.roles.get("search")}:
                self.note_view(name, observation)
            # every job but searching: only a search result is paged
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            if self.single_target(name) and name not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask")):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get("open") and not self.facts_from[name]:
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
                # lines taken off by a zero or negative count (unless the reply still lists them), and the fullest
                # cart built so far, which is offered back when a late decline would throw it away
                counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.rows:
                        if counts[0] <= 0 and not _in_rows(observation, value):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get("cart_view"):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get("finished"):
                    self.refused_at = self.step
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.is_buy(name, args) and not reply.get("finished") and (self.note_sold or any(line in self.spoken(observation) for line in self.sold_lines)):
                seen = self.find_sold()
                if seen:
                    text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.finished or self.broke or self.left() <= 0:
            return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        # an empty cart or one missing a requested item cannot pass: the closest listings are bought instead, unless
        # the review proves nothing can be sold; an incomplete cart is never ordered
        if buy and (not holding or self.cart_missing() or any(self.sold_out(i) for i in self.cart_ids())):
            rescued = self.rescue_buy()
            if self.finished or self.left() <= 0:
                return ""
            if rescued and again:
                # the rescue's checkout was turned down with a change: the agent gets to act on it
                return rescued
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = (bool(self.cart_ids()) if fresh else bool(self.cart_lines)) and not self.cart_missing() \
                and not any(self.sold_out(i) for i in self.cart_ids())
        if buy and holding and not self.finished:
            # the requirements or the budget changed after the cart was set (race 13's endings were turned down with a
            # dropped requirement or a new budget): under a cheapest, most or least pick the cart chosen for the old
            # rules is re-picked before a code-made checkout re-buys it
            turned = self.repick()
            if turned and again:
                return turned
            if self.finished or self.left() <= 0:
                return ""
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and holding and not self.finished and self.left() >= 3 and self.budget_over():
            redone = self.budget_rescue()
            if self.finished or self.left() <= 0:
                return ""
            if redone and again:
                return redone
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        settled = self.decline_proven == self.version
        if buy and not settled and (holding or (self.cart_touched and not fresh and self.left() >= 2 and self.cart_lines
                                                and not self.cart_missing() and not any(self.sold_out(i) for i in self.cart_ids()))):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate("order")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.size_loop()
        self.work(self.loop_cap, self.spend_hard)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.spend_stop - 0.004)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def size_loop(self) -> None:
        """Limits sized to the task's step allowance. Race 13 gave 41 of 90 tasks 91-113 steps, and Cobalt v1's fixed
        42 turns and 0.042 USD ended them near step 40 with an empty cart (67 endings without a purchase forced by code,
        none right). Each step past 40 buys a few more turns and a little more purse (the loop's model reads its cached
        history, about 0.0005 USD a turn), while the race run's key holds 5 USD for 90 tasks, so the loop stops at
        0.052 and no call starts past 0.060 even on the longest task."""
        extra = max(0, min(self.max_steps, 110) - 40)
        self.loop_cap = min(max(LOOP_CALLS, self.max_steps), LOOP_CALLS + extra)
        self.call_cap = self.loop_cap + (ALL_CALLS - LOOP_CALLS) + extra // 6
        self.spend_hard = min(0.052, SPEND_HARD + 0.0002 * extra)
        self.spend_stop = min(0.060, self.spend_hard + (SPEND_STOP - SPEND_HARD))
        self.trace.append({"kind": "limits", "steps": self.max_steps, "loop": self.loop_cap, "calls": self.call_cap,
                           "hard": round(self.spend_hard, 4), "stop": round(self.spend_stop, 4)})

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap \
                and self.seconds() < CLOCK_LOOP and not self.hand_over():
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if any(need not in self.ref_values and need not in self.link_walked for _, need in self.refs) \
                    and (self.asked or not self.roles.get("ask") or self.msgs_free() <= 0):
                walked = self.walk_links()
                if walked:
                    self.chat.append({"role": "user", "content": walked + "\n" + self.status()})
                continue
            if self.calls >= self.call_cap - 3:
                break
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.loop_calls}_{number}"), "type": "function",
                              "function": {"name": str(fn.get("name") or ""), "arguments": _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _dump(content) if content else ""
            if not calls:
                idle += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Kyo(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
