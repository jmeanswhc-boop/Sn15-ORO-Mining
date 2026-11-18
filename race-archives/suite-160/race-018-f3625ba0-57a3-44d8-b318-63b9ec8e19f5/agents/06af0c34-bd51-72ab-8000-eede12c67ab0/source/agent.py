"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"

LOOP_CALLS = 42
TURNS_PER_STEP = 0.6
ALL_CALLS = 60
SPEND_SOFT = 0.055
SPEND_HARD = 0.078
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
TASK_SECONDS = 300.0
TIME_MARGIN = 20.0
CALL_FLOOR = 6.0
ASK_BY = 165.0
QUESTION_BY = 210.0
HUNT_BY = 200.0
WIND_DOWN = 200.0
LAST_TURN = 255.0
EMPTY_ROOM = 12.0
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
QUERY_NAMES = {"q", "query", "search", "text", "phrase", "keyword", "keywords", "term", "terms", "words", "input"}
JUDGE_BATCH = 6
CHEAPER_OPENS = 3

_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_short: dict[int, ProxyClient] = {}
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget (or the higher purchase limit the ledger gives); or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Some stores stop paging before the reported total: the listings past that point appear only for narrower words, never for more pages. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge a requirement on a property first from those attributes (WHAT COUNTS AS A FIT says when the title or description counts), so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Kyo adds bracketed lines to results ([evidence], [fit], unopened rows, searches so far, pages already opened). They are facts its code read from what the store showed: use them to choose what to open or search next. They never replace a listing's opened page, which is where a listing is judged. A listing under "fits" in [listings judged against the requirements now in force] was checked against its own page's words: treat it as fitting unless its page shows a requirement it breaks.\n- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that holds every word of the required value counts (its extra words only narrow it). A value with other words does not count (an abbreviation or code is not the word it stands for). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever keys for other properties or the title say. When the record also gives another value for that property (a second key naming it, the title, the options or a feature list), the record is mixed: the requirement still counts as met when a key naming the property states the value, but a mixed listing ranks below every fitting listing whose record agrees with itself, under any pick rule (with a cheapest pick, take it only when no listing whose record agrees fits). A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must be compatible with another (an accessory and the product it is for), choose the two together from the listings' own words: the accessory fits only when its own list of what it fits includes the other item's own model code (a series written with a wildcard counts, minus any exceptions it names). A line saying what a different part fits does not count for this listing, and a list continued on another part's page counts only once you have read that page. Model codes usually cannot be searched for, since they appear only in descriptions: browse the accessory's category and use the [fit] lines. When no accessory that meets its own requirements fits the product you picked, switch to another product that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists (only the values the request names). Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget, or the higher purchase limit the ledger gives when the request allows one; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check. Check that it worked: rows priced above it mean that search mode ignores it, so use a mode or filter that applies it.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: Kyo re-reads the cart's lines and looks at the cart itself when you check out, and holds the checkout with what changed.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short; name a listing only when the request asks that ending to put one forward (Kyo's review then fills it in).\n- Every reply must call at least one tool. Write at most one short sentence (the next move and why); several independent calls can go in one reply."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product. A product that must fit, suit or be compatible with another product of this order gets the must "compatibility: compatible with the chosen <that other item's name>", using exactly the "name" you give that other item, and that other item gets no must from it}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request's own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"buy_over": the share past the budget that the request lets the purchase itself cost, as a fraction (0.2 for 20 percent), or 0 when it allows none. A share allowed only for a listing named in an ending without a purchase does not count here.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else []; never a must tied to another requested item (those go under "relations").\n"relations": a list of the ties between two requested items, each {"item": the number of the item (counted from 1 in the order of "items") that carries the requirement, "to": the number of the other requested item, "kind": "fits" (it must work with that item), "same", "near" or "unlike" (a property compared with that item's), "property": the property compared, or "" for "fits", "need": that must, copied exactly from the item's musts}; [] when there is none.\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product, usually at most one per product. A question reveals a preference only when it names that exact property, and the shopper answers about one property at most. So each question is about one product and names properties of it that may hold an unwritten preference, likeliest first; how many is yours to decide (see below). Never put two products in one question.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nFor each item, list its open properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), named by the property it is about.\n1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. Name the properties each such sentence most likely points at: a hint about who will wear it points at gender, a hint about its looks at colour and finish.\n2. Then that item's listed keys: first the properties a buyer of that kind of product chooses between (often its size, capacity or dimensions, its type, its material or its colour), then the keys most of the item's listings carry; keys seen on only one or two listings come last.\nLeave out a property the request already settles for that item, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics, and a second name for a property already in the list.\nEach label is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets. List up to 12 labels per item.\nThen decide, from this request, how many labels each question names. A question can reveal a preference only on a property it names, so naming more covers more; but the shopper answers about one property at most, and a question crowded with names can be turned away or answered vaguely. Weigh how many messages there are (COUNT) for how many items, how many properties of the item are open and how sure you are of the first labels: a property the story clearly hints at needs little company, an item with many open properties and no hint needs more names, and when the messages are few for the items, each item may get only one question. The first question to an item names the first "first" labels; each later question to it names the next "then" labels.\nOne entry per item. When there are more items than COUNT, put first the items whose story hints at a preference, then the items with the most open properties. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "labels": ["...", "..."], "first": how many labels the first question names, "then": how many labels each later question names, "hint": "at most 8 of the story's words behind the first hinted label, or an empty string"}]}"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value comes from a different listing, else false, "to_item": the number of another requested item when the value or fit is about that item, else null, "tie": "fits", "same", "near" or "unlike" for such a need, else null}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "unclear": [numbers of the questions whose answer neither states a preference nor says the shopper does not mind: the shopper did not follow it or asks for it again], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). A question may name several properties; its answer is about one of them at most. Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.\n- A value the shopper insists on, refuses to give up or accepts nothing but is the value wanted, stated as that value; write a need as "not ..." only when the answer itself rules that value out."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint", "points": true when the value comes from a different listing (a value like, unlike or near the one on some shop's product), else false, "to_item": the number of another requested item when the value or fit is about that item, else null, "tie": "fits", "same", "near" or "unlike" for such a need, else null}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. A need whose value points at another listing is ONE need holding the note's own words from the comparison words to the end of that listing's description, copied exactly; the properties describing that other listing are not needs. Report every change in the note."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, singular or plural and plain synonyms count, a value holding every word of the required value counts but a value with other words does not, and an abbreviation or code is not the word it stands for; units are converted, limits include the edge, and a range must lie wholly inside a limit; when the record also gives another value for that property (a second key naming it, the title, the options or a feature list), it still counts as met, but the record is mixed and ranks below fitting listings whose record agrees with itself); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not, as a weight is not a capacity, nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; an item that must be compatible with another chosen item fits only when its own list of what it fits includes that item's own model code (a series written with a wildcard counts, minus its exceptions), and a line saying what a different part fits does not count; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings do not fit; a listing's price is weighed against the budget as the request's own words say, never past the purchase limit the requirements give."""

CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: """ + READING + """ Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock. The budget, and any share past it the request allows, apply as the request's own words say: read from them when that share may be used (for example only when nothing within the budget fits) and whether a dearer listing is then better; tags in square brackets give each price against the budget and the request's limit, worked out for you, and a cart past that limit is always wrong. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, or other listings are shown, also compare them: if one of them clearly fits every requirement and beats the cart's choice on that rule (or on the budget, as the request's words weigh it), the cart is wrong.\nA line marked VERIFIED ON ITS PAGE had each requirement it lists checked against that page's own words: take those as met unless a newer record of the listing shows otherwise, and spend the check on the rest (the lines and their quantities, the total, pairs, and the pick rule).\nA record marked MIXED RECORD gives another value for a requirement's property on its own page: it may still fit, but a fitting listing whose record agrees with itself is better under any pick rule (a cheapest pick included), so name such a listing under "better".\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first], "lines": {"each cart listing id": "fits" or "fails: the requirement it breaks"}}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nApart from that, read the request's own rule for an ending without a purchase: when it asks that ending to put a listing forward (one past the budget or short of a requirement, when its rule allows that), pick the one its rule prefers among the records shown; listings past the purchase limit the requirements give never go under "fits", and one past the budget goes there only when the request's own words allow that share now.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "fails": {"listing id": "the requirement it clearly breaks"} for OPENED listings the records clearly fail (at most 10), "open": [ids of ROW listings of a requested kind, inside the budget (or the purchase limit the requirements give), whose page could show that they fit, most promising first, at most 6], "ending": {"name": [the listing id that ending puts forward, or none], "say": "the ending's words, short, as the request asks"}, "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property's value itself (also when the requirement asks for a value other than or unlike that one: the page's value is what it is measured against), else false, "value": the property's value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (a description naming a pack form or size rules out the same shop's listing of that kind without it); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop's name and the description's values, as a few search words.\nList each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before that is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint", "points": true when the value comes from a different listing, else false, "to_item": the number of another requested item when the value or fit is about that item, else null, "tie": "fits", "same", "near" or "unlike" for such a need, else null} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """\nThe budget, and any share past it the request allows, apply as the request's own words say (for example a share used only when nothing within the budget fits); tags in square brackets give each price against them, worked out for you, and a listing past the purchase limit never goes under "fits".\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, best first by the pick rule in the requirements and the budget as the request's words weigh it, listings marked MIXED RECORD after every other fit, at most 6], "reason": "one sentence"}"""

JUDGE = """A shopping agent has opened the listing pages below. For each page, say which requested item it is a listing of, and judge it against each numbered requirement of that item, with these reading rules: """ + READING + """\nFor every judgment, copy the page's own words it rests on (a key with its value, or a phrase of the title or description), as short as they can be; a requirement the page gives no words for is "not shown".\nReturn ONE JSON object and nothing else: {"pages": {"listing id": {"item": the number of the requested item it is a listing of, or 0 when it is none of them, "needs": {"requirement number": {"v": "met", "broken" or "not shown", "quote": "the page's own words", "mixed": "the page's own words that give another value for that property (a second key naming it, the title, the options or a feature list), copied exactly, or \"\"", "partner_quote": "for a requirement judged against a partner listing, that listing's own words you compared with, else \"\""}}, "is": [the model codes or names the page gives for this product itself, each copied exactly], "fits": [what the page says this product works with, each copied exactly], "more": true when the page says that list goes on elsewhere}}}"""

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
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


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


def _on_page(quote: Any, record: Any) -> bool:
    """A quote from a model counts only when its text is on the listing's record: keys and values at any depth, any case,
    spacing, hyphens or punctuation (lidColour reads as lid colour; XR-200 as XR 200 and XR200)."""
    def flat(node: Any, depth: int = 0) -> str:
        if depth > 6:
            return ""
        if isinstance(node, dict):
            return " ".join(f"{k} {flat(v, depth + 1)}" for k, v in node.items())
        if isinstance(node, list):
            return " ".join(flat(v, depth + 1) for v in node)
        return str(node)

    def norm(text: str) -> str:
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
        return re.sub(r"[^a-z0-9.]+", " ", text).strip()

    said, page = norm(str(quote or "")), norm(flat(record))
    if len(said) < 2:
        return False
    return said in page or (len(said.replace(" ", "")) >= 3 and said.replace(" ", "") in page.replace(" ", ""))


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
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        steps = _count(self.view.get("max_steps"))
        if steps is None and isinstance(self.view.get("max_steps"), float):
            steps = int(self.view["max_steps"])
        self.max_steps = steps if steps and steps > 0 else 30
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
        self.started = time.monotonic()
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = ""
        self.audits = 0
        self.buy_over = 0.0
        self.ending: dict[str, Any] = {}
        self.price_held: set[str] = set()
        self.ref_pages: set[str] = set()
        self.store_words: dict[tuple[int, str], list[str]] = {}
        self.early_rounds = 0
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.cheap_held: set[str] = set()
        self.cap_modes: dict[tuple[str, str], dict[str, bool]] = {}
        self.cap_args: dict[tuple[str, str], tuple[dict[str, Any], set[str]]] = {}
        self.mixed: dict[str, tuple[str, list[str]]] = {}
        self.judge_flushed = ""
        self.fit_holds = 0
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
        self.judged: dict[str, tuple[str, str, str]] = {}
        self.proofs: dict[str, dict[str, tuple[str, str]]] = {}
        self.item_of: dict[str, int] = {}
        self.facts_of: dict[str, dict[str, Any]] = {}
        self.link_needs: set[tuple[int, str]] = set()
        self.relations: list[dict[str, Any]] = []
        self.judged_with: dict[str, str] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.links: dict[int, int] = {}
        self.ref_values: dict[str, str] = {}
        self.chains: dict[str, list[str]] = {}
        self.hop_searches: dict[str, int] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.ref_seen = 0
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.search_args: dict[str, dict[str, Any]] = {}
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.searched: list[str] = []
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.window = 0
        self.query_rows: dict[str, set[str]] = {}
        self.short_tries = 0
        self.short_dead: set[str] = set()
        self.query_pages: dict[str, set[int]] = {}
        self.query_sorted: dict[str, bool] = {}
        self.loop_cap, self.call_cap, self.spend_soft, self.spend_hard = LOOP_CALLS, ALL_CALLS, SPEND_SOFT, SPEND_HARD

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
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap:
                return None
            room = TASK_SECONDS - TIME_MARGIN - (time.monotonic() - self.started)
            if room < CALL_FLOOR:
                self.trace.append({"kind": "late_call_skipped", "at": round(time.monotonic() - self.started, 1)})
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                    break
                if TASK_SECONDS - TIME_MARGIN - (time.monotonic() - self.started) - WAITS[tries] < CALL_FLOOR:
                    break
                pause = WAITS[tries]
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
            marks = [0] + ([len(sent) - 2] if len(sent) > 2 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        room = TASK_SECONDS - TIME_MARGIN - (time.monotonic() - self.started)
        client = _llm
        if room < 50:
            limit = max(5, int(room) // 5 * 5)
            client = _llm_short.setdefault(limit, ProxyClient(timeout=limit, max_retries=1, retry_delay=1))
        began = time.monotonic()
        result = client.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            self.trace.append({"kind": "llm", "tag": str(messages[0].get("content") or "")[:24], "model": model,
                               "in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                               "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                               "cost": usage.get("cost"), "sec": round(time.monotonic() - began, 2)})
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
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
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
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

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
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
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 600) or {}
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
        for entry in _items(brief.get("relations")):
            if not isinstance(entry, dict):
                continue
            a, b = _count(entry.get("item")), _count(entry.get("to"))
            kind, need = str(entry.get("kind") or "").strip().lower(), str(entry.get("need") or "").strip()
            if (not a or not b or a == b or not 1 <= a <= len(self.items) or not 1 <= b <= len(self.items)
                    or kind not in ("fits", "same", "near", "unlike") or (need and need not in self.items[a - 1].musts)):
                continue
            self.relations.append({"item": a, "to": b, "kind": kind, "property": str(entry.get("property") or "").strip(), "need": need})
            if need:
                self.link_needs.add((a, need))
            if kind == "fits":
                self.links.setdefault(a, b)
        for number, item in enumerate(self.items if not self.relations else [], 1):
            for must in item.musts:
                prop, _, rest = must.partition(":")
                if "compat" not in prop.lower():
                    continue
                named = [n for n, other in enumerate(self.items, 1) if n != number and other.name.strip() and other.name.strip().lower() in rest.lower()]
                named = [n for n in named if not any(m != n and self.items[n - 1].name.strip().lower() in self.items[m - 1].name.strip().lower() for m in named)]
                if len(named) == 1:
                    self.links[number] = named[0]
                    self.link_needs.add((number, must))
                    self.relations.append({"item": number, "to": named[0], "kind": "fits", "property": "", "need": must})
        if self.links:
            self.trace.append({"kind": "links", "links": dict(self.links)})
        tied = {need for _, need in self.link_needs}
        for lookup in _items(brief.get("lookups")):
            text = str(lookup).strip()
            if text in tied:
                continue
            if not any(text in item.musts for item in self.items):
                key = _words(text.split(":")[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and text not in tied and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = budget
        over = brief.get("buy_over")
        if self.budget is not None and isinstance(over, (int, float)) and not isinstance(over, bool) and over > 0:
            sure = self.ask_json("REQUEST:\n" + self.request + "\n\nDoes this request let the purchase itself cost more than its budget? A share allowed only"
                                 " for a listing named in an ending without a purchase does not count. Return ONE JSON object and nothing else:"
                                 " {\"buy_over\": that share as a fraction, or 0}", [HEAVY], 300) or {}
            over = sure.get("buy_over", over)
        over = over / 100 if isinstance(over, (int, float)) and not isinstance(over, bool) and over > 1 else over
        self.buy_over = float(over) if isinstance(over, (int, float)) and not isinstance(over, bool) and 0 < over <= 1 else 0.0
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get("ask"):
            self.msg_limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        self.loop_cap = max(LOOP_CALLS, round(TURNS_PER_STEP * self.max_steps))
        self.call_cap = ALL_CALLS + (self.loop_cap - LOOP_CALLS)
        scale = min(2.0, self.loop_cap / LOOP_CALLS)
        self.spend_soft, self.spend_hard = SPEND_SOFT * scale, SPEND_HARD * scale
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.roles.get("ask"):
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
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
            got = self.ask_json(prompt, [HEAVY, LIGHT], 500) or {}
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
            parts.append((f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}"
                          + (f" (the request lets the purchase itself cost up to {self.budget * (1 + self.buy_over):g}, when its own words allow that share)" if self.buy_over else "")).replace("  ", " "))
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
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
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
        if time.monotonic() - self.started >= WIND_DOWN and left > self.tail() + 1:
            text += " TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        live_ties = [r for r in self.relations if not r["need"] or r["need"] in live]
        if live_ties:
            text += "\n[pairs] " + " | ".join(
                f"item {r['item']} ({self.items[r['item'] - 1].name}) " + ("must work with" if r["kind"] == "fits" else f"{r['kind']} {r['property']} as")
                + f" item {r['to']} ({self.items[r['to'] - 1].name}): choose item {r['to']} first, then judge item {r['item']} against that choice" for r in live_ties[:3])
        cat_key = self.category_key()
        typed = " ".join(self.searched).lower()
        unlike = []
        for number, item in enumerate(self.items, 1):
            kind = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            pages = [i for i in self.seen_order if i in self.full and (len(self.items) == 1 or (cat_key and {w for w in _stem(str(self.full[i].get(cat_key) or "")) if len(w) >= 4} & kind))]
            for need in item.musts + item.told:
                prop, _, value = need.partition(":")
                said = {w for w in _stem(value) if len(w) >= 4 and not w.isdigit()} - _stem(prop)
                if (need in item.dropped or need in self.ref_values or any(need == n for _, n in self.refs)
                        or (number, need) in self.link_needs or not said):
                    continue
                tally: dict[tuple[str, str], int] = {}
                for ident in pages:
                    for k, v in (_fact_block(self.full[ident]) or {}).items():
                        if isinstance(v, str) and re.search(r"[A-Za-z]", v) and _names(_stem(prop), _stem(k)):
                            tally[(k, v.strip()[:40])] = tally.get((k, v.strip()[:40]), 0) + 1
                store = {w for _, v in tally for w in _stem(v)} - _stem(prop)
                fresh = [f"{k}={v}" for (k, v), _ in sorted(tally.items(), key=lambda kv: -kv[1]) if v.lower() not in typed][:4]
                if tally and not said & store and fresh:
                    self.store_words[(number, need)] = [f.split("=", 1)[1] for f in fresh]
                    unlike.append((f"(item {number}) " if len(self.items) > 1 else "") + f"{need} -> this store writes {', '.join(fresh)}")
        if unlike:
            text += ("\n[worded unlike the shopper: the values opened pages give for these properties; searches read these words. If one of them means what the shopper asked, search with it] "
                     + " | ".join(unlike[:3]))
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        good = [i for i in self.pick_order() if self.judged.get(i, ("",))[0] == mark and self.judged[i][1] == "fits" and self.buyable(i) and i not in self.ref_pages][:8]
        bad = [i for i in reversed(self.seen_order) if self.judged.get(i, ("",))[0] == mark and self.judged[i][1] == "fails"][:6]
        if good or bad:
            text += ("\n[listings judged against the requirements now in force] fits: "
                     + (", ".join((f"{i} ({self.price_of(i):g}" if self.price_of(i) is not None else f"{i} (") + (", mixed record" if self.messy(i) else "") + ")" for i in good) or "none")
                     + " | fails: " + (", ".join(f"{i} ({self.judged[i][2]})" for i in bad) or "none"))
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.searched:
            text += "\n[searches so far, oldest first; a repeat shows nothing new] " + " | ".join(self.searched[-8:])
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

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
                bare = len(values) >= 4 and len(set(values)) == len(values) and mean > 12
                twin = any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept)
                if twin:
                    continue
                kept.append((name, {**slot, "values": []} if bare else slot))
            for name, slot in kept[:20]:
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

    def messy(self, ident: str) -> str:
        """The page words that make a listing's record mixed (another value for a requirement's property on its own page),
        under the requirements now in force; "" when its record agrees with itself or was not judged."""
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        at, words = self.mixed.get(ident, ("", []))
        return "; ".join(words) if at == mark and words else ""

    def price_note(self, ident: str) -> str:
        """A listing's price for its item's quantity against the budget and the request's limit, worked out by code for the
        model that weighs it (which reads from the request when a share past the budget may be used); "" with no price or budget."""
        p = self.price_of(ident)
        if p is None or self.budget is None:
            return ""
        n = self.items[self.item_for(ident) - 1].qty if self.item_for(ident) else 1
        cost, cap = p * n, self.budget * (1 + self.buy_over)
        what = f"price {p:g}" + (f" x {n} = {cost:g}" if n > 1 else "")
        if cost <= self.budget:
            return f"[{what}: within the budget {self.budget:g}] "
        if self.buy_over and cost <= cap:
            return f"[{what}: past the budget {self.budget:g}, within the request's limit {cap:g}] "
        return f"[{what}: past the purchase limit {cap:g}] "

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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
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
        linked = False
        if self.links:
            cat_key = self.category_key()
            labels = {str(r.get(cat_key) or "") for _, r in self.records_in(observation)} if cat_key else set()
            kinds = [{w for w in _stem(it.shelf or "") | _stem(it.name) if len(w) >= 4} for n, it in enumerate(self.items, 1) if n in self.links or n in self.links.values()]
            linked = any({w for w in _stem(label) if len(w) >= 4} & kind for label in labels for kind in kinds)
        plan = self.paging(name, observation) if pick in ("cheapest", "most", "least") or linked else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            return ""
        if pick == "cheapest" and self.price_sorted(observation) and not linked:
            return ""
        pages = -(-(count - shown) // shown)
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(param) or {}
        top = spec.get("maximum") if isinstance(spec.get("maximum"), int) and not isinstance(spec.get("maximum"), bool) else None
        beyond = 0
        if top is not None and isinstance(value, int) and not isinstance(value, bool):
            reach = max(0, top - value + 1)
            beyond = max(0, count - shown * (reach + 1))
            pages = min(pages, reach)
        if linked:
            pages = min(pages, 2)
        if pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
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
            if isinstance(value, int) and not isinstance(value, bool):
                self.query_pages.setdefault(name + _dump({k: v for k, v in args.items() if k != param}), set()).add(value)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        tail = (f"\n[Kyo] The store pages no further than {param}={top}: {beyond} more of the {count} matches appear only for narrower words"
                " (the requirement's value with the product noun, or a filter).") if texts and beyond else ""
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts) + tail) if texts else ""

    def cheapest_here(self, observations: list[Any], worded: bool = False) -> str:
        """The unopened rows of a search result (and the pages Kyo read on), most promising first by what the rows show."""
        cheapest = self.brief.get("pick") == "cheapest"
        cat_key = self.category_key()
        rows = list(dict.fromkeys(i for seen in observations for i, _ in self.records_in(seen)))
        mains = [i for i in dict.fromkeys(self.cart_ids() + list(reversed([j for j in self.seen_order if j in self.full])))][:20]
        kinds = [{w for w in _stem(it.shelf or "") | _stem(it.name) if len(w) >= 4} for it in self.items]
        label = lambda i: {w for w in _stem(str((self.rows.get(i) or {}).get(cat_key) or "")) if len(w) >= 4} if cat_key else set()
        orphans = [n for n, kind in enumerate(kinds, 1) if not any(label(i) & kind for i in self.seen_order)]
        lines = []
        for number, item in enumerate(self.items, 1):
            if worded and not cheapest and number not in self.links:
                continue
            kind = kinds[number - 1]
            values = [n.split(":", 1)[-1] for n in item.musts + item.told if n not in item.dropped]
            wanted = {w for v in values if not re.match(r"\s*(not|no|never)\b", v, re.I) for w in _stem(re.sub(r"[^\w]+", " ", v))
                      if len(w) >= 3 and w not in ("most", "least", "more", "less", "than", "and", "the", "with", "any")}
            scored = []
            for order, ident in enumerate(rows):
                record = self.rows.get(ident) or {}
                if ident in self.full or not self.buyable(ident):
                    continue
                if len(self.items) > 1 and not label(ident) & kind and not (orphans == [number] and not any(label(ident) & k for k in kinds)):
                    continue
                price = self.price_of(ident)
                if price is not None and self.budget is not None and price * item.qty > self.budget:
                    continue
                verdicts = [self.fit_of(ident, m) for m in mains if m != ident and self.item_for(m) == self.links[number]][:8] if number in self.links else []
                known = [v for v in verdicts if v != "?"]
                if known and all(v == "no" for v in known):
                    continue
                words = _stem(re.sub(r"[^\w]+", " ", " ".join(str(v) for v in record.values() if isinstance(v, str))))
                scored.append(((0 if "fit" in verdicts else 1), 0 if worded else -len(wanted & words), price if cheapest and price is not None else 0.0, order, ident, price, "fit" in verdicts))
            scored.sort()
            if scored:
                lines.append(f"item {number}: " + ", ".join(f"{s[4]}" + (f" ({s[5]:g}" if s[5] is not None else " (") + (", fits" if s[6] else "") + ")" for s in scored[:3]))
        if lines:
            return "\n[Kyo] unopened rows here, most promising first by what the rows show (open a page before judging it): " + " | ".join(lines)
        return ""

    def wider_hold(self, lacking: list[int] | None = None) -> str:
        """Once each: before a checkout with many steps left under a cheapest, most or least pick, and (any pick) before ending without a purchase
        while an item has no fitting listing and steps are left. A store's search reads its own wording, so listings
        worded unlike the shopper are missed until the walker writes its searches the way this store writes listings."""
        style = (" A store's search matches the store's own wording. First look at how the listings you have opened write this kind of product and these properties"
                 " (the product's own name in their titles, each property's key, and how its values are written: words, units and their spelling, word order),"
                 " then write each search the way this store would write what the shopper means, not in the shopper's words where they differ.")
        if lacking is not None:
            if self.pushed or self.left() <= self.tail() + 2 * len(self.items) + 4:
                return ""
            self.pushed = True
            where = f"for {', '.join(f'item {n}' for n in lacking)} (the review found fits for the other items)" if lacking else "for each requested item"
            return (f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending, {where}." + style
                    + " Also browse its category and page through it with the token or offset the store returns, and open the listings that might fit. End without a purchase only after that.")
        pick = self.brief.get("pick")
        if self.wider_held or pick not in ("cheapest", "most", "least") or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        if pick != "cheapest":
            measure = str(self.brief.get("pick_measure") or "").strip() or "the measure the request names"
            return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the {pick} {measure} among fitting listings, so the best one may be among fitting listings worded unlike your searches, still unseen." + style
                    + f" Listings that lead on {measure} are often worded alike, so also search the way the best ones on {measure} you have seen are worded."
                    + f" Run a few such searches for each item, open the listings whose rows could beat the cart's line on {measure}, swap in any that fits every requirement and beats it, then check out.")
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        cat_key = self.category_key()
        low = sorted((self.price_of(i), i) for i in self.seen_order if i not in self.cart_ids() and self.buyable(i) and self.price_of(i) is not None)[:3]
        cheap = [f"{i} at {p:g}" + (f" ({(self.rows.get(i) or {}).get(cat_key)})" if cat_key else "") for p, i in low]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest, and rows come in no price order, so fitting listings worded unlike your searches are still unseen." + style
                + " Cheaper listings of one kind are often worded alike (their size, pack or form), so also search the way the cheapest listings you have seen are worded"
                + (f" ({'; '.join(cheap)})" if cheap else "") + " to reach others like them."
                + " Run a few such searches for each item and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

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
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or time.monotonic() - self.started > ASK_BY:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

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

    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        self.trace.append({"kind": "interview", "limit": self.msg_limit, "used": self.msg_used, "reserve": self.reserve, "count": count})
        if count <= 0:
            return ""
        before =[len(item.told) for item in self.items]
        self.widen()
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values))
        prompt = PLAN.replace("COUNT", str(count)) +"\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 1800) or {}
        ordered: list[tuple[int, str]] = []
        rest: dict[int, list[str]] = {}
        ask_props = (self.schemas.get(self.roles.get("ask") or "") or {}).get("properties") or {}
        room = min((v["maxLength"] for v in ask_props.values() if isinstance(v, dict) and "string" in _kinds(v)
                    and "enum" not in v and isinstance(v.get("maxLength"), int)), default=0)
        join = lambda fit: fit[0] if len(fit) == 1 else ", ".join(fit[:-1]) + " or " + fit[-1]
        per: dict[int, int] = {}
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            if any(n == number for n, _ in ordered):
                continue
            names: list[str] = []
            for raw in _items(pick.get("labels")) or [pick.get("label")]:
                name = re.split(r"[/;(]", str(raw or ""))[0].strip().rstrip("?.")
                if name and len(name.split()) <= 5 and all(_words(name) != _words(n) for n in names):
                    names.append(name)
            if not names:
                continue
            fit = names[: max(1, _count(pick.get("first")) or len(names))]
            while len(fit) > 1 and room and len(self.question(number, join(fit))) > room:
                fit = fit[:-1]
            rest[number] = names[len(fit):]
            per[number] = max(1, _count(pick.get("then")) or len(fit))
            ordered.append((number, join(fit)))
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120)})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        head = ordered[:count]
        batch = self.send([(number, label, self.question(number, label)) for number, label in head])
        if not batch:
            return ""
        sent += batch
        heard = self.hear(batch)
        self.take_heard(heard, told)
        picks = head[: len(batch)]
        crowded = {i for i in (_count(x) for x in _items(heard.get("crowded"))) if i and 1 <= i <= len(picks)}
        unclear = [picks[i - 1] for i in (_count(x) for x in _items(heard.get("unclear"))) if i and 1 <= i <= len(picks) and i not in crowded]
        for i in sorted(crowded):
            number, label = picks[i - 1]
            rest[number] = re.split(r",\s*|\s+or\s+", label) + rest.get(number, [])
        revealed = {_count(e.get("item")) for e in _items(heard.get("needs")) if isinstance(e, dict)}
        while self.msgs_free() > 0 and not self.finished and time.monotonic() - self.started < QUESTION_BY:
            again = None
            while unclear and again is None:
                number, label = unclear.pop(0)
                parts = re.split(r",\s*|\s+or\s+", label)
                if len(parts) > 1:
                    turned = parts[1:] + parts[:1]
                    again = (number, ", ".join(turned[:-1]) + " or " + turned[-1])
            if again is None:
                for number in sorted(rest, key=lambda n: n in revealed):
                    if rest[number]:
                        fit = rest[number][: per.get(number, 1)]
                        while len(fit) > 1 and room and len(self.question(number, join(fit))) > room:
                            fit = fit[:-1]
                        rest[number] = rest[number][len(fit):]
                        again = (number, join(fit))
                        break
            if again is None:
                break
            batch = self.send([(again[0], again[1], self.question(again[0], again[1]))])
            if not batch:
                break
            sent += batch
            got = self.hear(batch)
            self.take_heard(got, told)
            revealed |= {_count(e.get("item")) for e in _items(got.get("needs")) if isinstance(e, dict)}
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        if told:
            cat_key = self.category_key()
            shown = []
            for number, item in enumerate(self.items, 1):
                new = [need for need in item.told[before[number - 1]:] if need not in item.dropped][:3]
                if not new:
                    continue
                kind = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
                pages = [i for i in reversed(self.seen_order) if i in self.full and not self.sold_out(i)
                         and (len(self.items) == 1 or (cat_key and {w for w in _stem(str(self.full[i].get(cat_key) or "")) if len(w) >= 4} & kind))][:8]
                for need in new:
                    prop = need.split(":", 1)[0].strip()
                    seen, bare = [], []
                    for ident in pages:
                        block = _fact_block(self.full.get(ident) or {}) or {}
                        hits = [f"{k}={str(v)[:40]}" for k, v in block.items() if _names(_stem(prop), _stem(k))][:2]
                        if hits:
                            seen.append(f"{ident} " + ", ".join(hits))
                        else:
                            bare.append(ident)
                    if bare:
                        seen.append(f"no {prop} key: " + ", ".join(bare))
                    if seen:
                        shown.append(f"item {number} '{need}': " + "; ".join(seen))
            if shown:
                summary += ("\nPages already opened, on the new requirements (attribute key=value as the page shows it; a key naming a different property decides nothing, and the title or description may still state the value): "
                            + " | ".join(shown))
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
            chain, _ = self.trail(early=True)
            self.ref_seen = len(self.refs)
            if chain:
                summary += "\n" + chain.strip()
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

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
            other = _count(entry.get("to_item"))
            tie = str(entry.get("tie") or "").strip().lower()
            if other and other != number and 1 <= other <= len(self.items):
                self.link_needs.add((number, need))
                self.relations.append({"item": number, "to": other, "kind": tie if tie in ("fits", "same", "near", "unlike") else "same",
                                       "property": need.split(":", 1)[0].strip(), "need": need})
                if tie == "fits":
                    self.links.setdefault(number, other)
            elif entry.get("points") is True and (number, need) not in self.refs:
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

    def hear(self, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        light = self.ask_json(prompt, [LIGHT], 1500)
        plain = light is not None and not any(_items(light.get(k)) for k in ("needs", "drop", "crowded", "unclear")) and light.get("budget") in (None, "", 0)
        if plain or self.no_credit or self.broke or self.calls >= self.call_cap:
            return light or {}
        return self.ask_json(prompt, [HEAVY], 1500) or light or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
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
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.take_heard({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.trace.append({"kind": "gate", "stage": stage, "call": _clip(call, 200), "steps": _clip(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
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
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        raised = False
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            raised = self.budget is not None and budget > self.budget
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
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            item = self.items[number - 1]
            need = str(entry["need"]).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            other = _count(entry.get("to_item"))
            tie = str(entry.get("tie") or "").strip().lower()
            if other and other != number and 1 <= other <= len(self.items):
                self.link_needs.add((number, need))
                self.relations.append({"item": number, "to": other, "kind": tie if tie in ("fits", "same", "near", "unlike") else "same",
                                       "property": need.split(":", 1)[0].strip(), "need": need})
                if tie == "fits":
                    self.links.setdefault(number, other)
            elif entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
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
        text = (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()
        if self.brief.get("pick") == "cheapest" and (drops or raised):
            again = [i for i in self.pick_order() if i in self.full and self.buyable(i) and i not in self.cart_ids()][:6]
            if drops:
                text += (". A requirement was dropped, so listings passed over before may fit now; the pick is still the cheapest of all that fit"
                         + (f". Re-check these read listings, cheapest first, against the requirements now in force: {', '.join(again)}" if again else ""))
            if raised:
                text += ". A higher limit is only a ceiling: the cheapest listing that fits still wins, so keep a fitting cart line unless a cheaper fit exists"
        return text

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
            if self.links:
                mine = self.item_for(ident)
                pool = [i for i in dict.fromkeys(self.cart_ids() + list(reversed([j for j in self.seen_order if j in self.full]))) if i != ident]
                others = [i for i in pool if mine in self.links and self.item_for(i) == self.links[mine]][:8]
                said = [f"fits {m}" if v == "fit" else f"its own list lacks {m}'s model" for m in others for v in [self.fit_of(ident, m)] if v != "?"][:4]
                deps = {d for d, m in self.links.items() if m == mine}
                fit_it = [x for x in reversed(self.seen_order) if x != ident and self.item_for(x) in deps and self.fit_of(x, ident) == "fit"][:4]
                if said:
                    parts.append("[fit] " + ", ".join(said))
                if fit_it:
                    parts.append("[fit] listings seen whose own list fits it: " + ", ".join(fit_it))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + " | ".join(lines)

    def item_for(self, ident: str) -> int | None:
        """Which requested item a listing is: JUDGE's reading when it judged the page; else the one item whose shelf or name
        shares a word with the listing's category label; None when that is not clear."""
        if ident in self.item_of:
            return self.item_of[ident]
        if len(self.items) == 1:
            return 1
        cat_key = self.category_key()
        label = {w for w in _stem(str((self.rows.get(ident) or {}).get(cat_key) or "")) if len(w) >= 4} if cat_key else set()
        hits = [n for n, it in enumerate(self.items, 1) if label & {w for w in _stem(it.shelf or "") | _stem(it.name) if len(w) >= 4}]
        return hits[0] if len(hits) == 1 else None

    def fit_of(self, dep: str, main: str) -> str:
        """Does listing dep fit listing main, by dep's own list of what it fits and main's own model code?
        "fit", "no" (dep's own list is complete and lacks the code) or "?" (no code, no list, or the list goes on elsewhere)."""
        fd, fm = self.facts_of.get(dep), self.facts_of.get(main)
        if fd is not None and fm is not None:
            flat = lambda t: re.sub(r"[^a-z0-9*]+", "", str(t).lower())
            listed, own = [flat(x) for x in fd.get("fits") or []], [flat(x) for x in fm.get("is") or []]
            if not listed or not own:
                return "?"

            def covers(entry: str, code: str) -> bool:
                wild = re.fullmatch(r"(.*\d)(x+|\*+)", entry)
                return entry == code or bool(wild and code.startswith(wild.group(1)) and len(code) == len(entry))
            if any(covers(e, c) for e in listed for c in own):
                return "fit"
            return "?" if fd.get("more") or not all(re.search(r"\d", e) for e in listed) else "no"

        def text(ident: str) -> str:
            record = self.full.get(ident) or self.rows.get(ident) or {}
            return " ".join(str(v) for v in record.values() if isinstance(v, str))
        code = re.search(r"\bmodel(?:\s*(?:no\.?|number|#))?\s*[:.]?\s*([A-Za-z0-9][A-Za-z0-9-]+\d)\b", text(main), re.I)
        own = re.search(r"\b(?<!not )(?:compatible with|works with|fits)\s*:?\s*([^.;]*)", text(dep), re.I)
        if not code or not own:
            return "?"
        model, listed = code.group(1).upper(), own.group(1)
        family = re.match(r"\s*every\s+([A-Za-z0-9-]*?\d)(x+)\s+model", listed, re.I)
        if not family and not any(len(c) >= 3 for c in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*\d", listed)):
            return "?"
        if family:
            stem = family.group(1).upper()
            excepted = {c.upper() for c in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*\d", listed[family.end():])}
            hit = model.startswith(stem) and len(model) == len(stem) + len(family.group(2)) and model not in excepted
        else:
            hit = model in {c.upper() for c in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*\d", listed)}
        if hit:
            return "fit"
        return "?" if re.search(r"\b(?:see|more (?:on|at)|details (?:on|at)|read)\s+(?:the\s+)?part\b", text(dep), re.I) else "no"

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def pick_order(self) -> list[str]:
        """Seen listings in the order a pick reads them: newest first; under a cheapest pick, opened listings first,
        inside the budget, cheapest first, and with several items each category's cheapest in turn."""
        order = list(reversed(self.seen_order))
        cheapest = self.brief.get("pick") == "cheapest"
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        over = lambda i: cheapest and self.budget is not None and price(i) > self.budget
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        known = lambda i: {"fits": 0, "fails": 2}.get(self.judged[i][1], 1) if self.judged.get(i, ("",))[0] == mark else 1
        if cheapest:
            clean = [price(i) for i in order if known(i) == 0 and not self.messy(i)]
            floor = min(clean or [price(i) for i in order if known(i) == 0] or [float("inf")])
            order.sort(key=lambda i: (over(i), known(i) == 2, bool(self.messy(i)), price(i) > floor, i not in self.full, price(i)))
        else:
            order.sort(key=lambda i: (known(i), bool(self.messy(i))))
        cat_key = self.category_key()
        if len(self.items) > 1 and cat_key:
            longs = [{w for w in _stem(it.shelf or "") | _stem(it.name) if len(w) >= 4} for it in self.items]
            names = {_words(it.shelf or "") for it in self.items} | {_words(it.name) for it in self.items}
            rank: dict[str, int] = {}
            count: dict[str, int] = {}
            stray: dict[str, bool] = {}
            for i in order:
                shelf = str((self.rows.get(i) or {}).get(cat_key) or "").strip().lower()
                rank[i] = count.get(shelf, 0)
                count[shelf] = rank[i] + 1
                stray[i] = not (_words(shelf) in names or any({w for w in _stem(shelf) if len(w) >= 4} & long for long in longs))
            order.sort(key=lambda i: (over(i), known(i) == 2, bool(self.messy(i)), stray[i], i not in self.full, rank[i]) if cheapest else (known(i), bool(self.messy(i)), stray[i], rank[i]))
        return order

    def others(self, skip: list[str], limit: int = 30) -> str:
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        lines = []
        for ident in self.pick_order():
            if ident in skip or self.sold_out(ident):
                continue
            fits = self.judged.get(ident, ("",))[0] == mark and self.judged[ident][1] == "fits"
            if len(lines) >= limit and not fits:
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(self.price_note(ident) + ("MIXED RECORD (" + self.messy(ident) + "). " if self.messy(ident) else "") + _dump(_clip(record, 160))[:900])
            if len(lines) >= 40:
                break
        return "\n".join(lines)

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def check_cart(self, ids: list[str] | None = None, limit: float | None = None) -> str:
        limit = self.spend_soft if limit is None else limit
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        prompt = CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        def verified(i: str) -> str:
            got = self.judged.get(i, ("", "", ""))
            if got[0] != mark or got[1] != "fits" or not str(got[2]).startswith("page review") or not self.proofs.get(i):
                return ""
            return "VERIFIED ON ITS PAGE (" + "; ".join(f"{need} -> {q!r}" for need, (v, q) in self.proofs[i].items() if q) + "). "
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it is unavailable). " if self.sold_out(i) else "") + self.price_note(i) + ("MIXED RECORD (" + self.messy(i) + "). " if self.messy(i) else "") + verified(i) + self.record_of(i) for i in ids)
        if self.budget is not None and None not in [self.price_of(i) for i in ids]:
            total = sum(self.price_of(i) * (self.cart_lines.get(i, 1) if not lead else self.items[(self.item_for(i) or 1) - 1].qty) for i in ids)
            cap = self.budget * (1 + self.buy_over)
            prompt += (f"\n\nCART TOTAL: {total:g} (" + ("within the budget" if total <= self.budget else "past the budget, within the request's limit" if total <= cap else "past the purchase limit")
                       + f"; budget {self.budget:g}" + (f", request's limit {cap:g}" if self.buy_over else "") + ")")
        pairs = []
        for r in self.relations:
            a = [i for i in ids if self.item_for(i) == r["item"]]
            b = [i for i in ids if self.item_for(i) == r["to"]]
            if len(a) != 1 or len(b) != 1:
                continue
            if r["kind"] == "fits":
                found = {"fit": "fit", "no": "no (its own complete list lacks the other's code)", "?": "not shown"}[self.fit_of(a[0], b[0])]
                pairs.append(f"item {r['item']} ({a[0]}) must work with item {r['to']} ({b[0]}): code check: {found}; judge it from both records above")
            else:
                shown = self.proofs.get(a[0], {}).get(r["need"], ("", ""))[1]
                pairs.append(f"item {r['item']} ({a[0]}) {r['kind']} {r['property']} as item {r['to']} ({b[0]})"
                             + (f" (its page: {shown!r})" if shown else "") + ": judge it from both records above")
        if pairs:
            prompt += "\n\nPAIRS:\n" + "\n".join(pairs)
        if self.brief.get("pick") in ("cheapest", "most", "least") or self.buy_over:
            order = "opened first, cheapest first, by category in turn" if self.brief.get("pick") == "cheapest" else "newest first, by category in turn" if len(self.items) > 1 else "newest first"
            prompt += f"\n\nOTHER LISTINGS SEEN ({order}):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        lines = got.get("lines") if isinstance((got or {}).get("lines"), dict) else {}
        for ident in ids:
            said = str(lines.get(ident) or "").strip()
            if said:
                self.judged[ident] = (mark, "fits" if said.lower().startswith("fit") else "fails", said[:80])
            elif got and got.get("ok") is not False:
                self.judged[ident] = (mark, "fits", "passed the cart check")
        for ident in [str(b) for b in _items((got or {}).get("better")) if str(b) in self.full]:
            if self.judged.get(ident, ("",))[0] != mark:
                self.judged[ident] = (mark, "fits", "named better by the cart check")
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        cap = None if self.budget is None else self.budget * (1 + self.buy_over)
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and self.buyable(str(b))
                  and (cap is None or self.price_of(str(b)) is None
                       or self.price_of(str(b)) * (self.items[self.item_for(str(b)) - 1].qty if self.item_for(str(b)) else 1) <= cap)][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def trail(self, early: bool = False) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        unread: list[str] = []
        read_for_value: set[str] = set()
        extra = 2 * max(0, len(self.refs) - 1)
        cap = 3 + extra if early else (6 if self.left() > 15 else 4) + extra
        fresh: set[str] = set()
        rounds, last_mark = 0, None
        while pending and self.cost < self.spend_hard and rounds < 12 and time.monotonic() - self.started <= HUNT_BY:
            mark_now = (len(self.opened), len(self.full), len(self.ref_values))
            if mark_now == last_mark or last_mark is None:
                if (self.early_rounds if early else self.trails) >= cap:
                    break
                if early:
                    self.early_rounds += 1
                else:
                    self.trails += 1
            last_mark = mark_now
            rounds += 1
            newest = list(reversed(self.seen_order))
            have = lambda i: i in self.opened and bool(self.full.get(i) or self.rows.get(i))
            chained = list(dict.fromkeys(i for need in pending for i in self.chains.get(need, []) if have(i)))
            long = self.ref_pages | fresh
            records = ["OPENED " + _dump(_clip(self.full.get(i) or self.rows.get(i), 1200))[:3000] for i in chained]
            records += [("OPENED " + _dump(_clip(self.full.get(i) or self.rows.get(i), 1200))[:3000]) if i in long
                        else ("OPENED " + _dump(_clip(self.full.get(i) or self.rows.get(i), 300))[:1400])
                        for i in newest if have(i) and i not in chained][: max(0, 45 - len(records))]
            records += ["ROW " + _dump(_clip(self.rows.get(i), 80))[:300] for i in newest if not have(i)][: max(0, 60 - len(records))]
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            known = [f"{need}: " + " -> ".join(self.chains[need]) for need in pending if self.chains.get(need)]
            if known:
                prompt += ("\n\nCHAIN SO FAR (listing ids in order, as an earlier reading found them; go on from the last one, or"
                           " correct it where the records show otherwise):\n" + "\n".join(known))
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 1800) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            unread = []
            refreshed = False
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                said = str(chain.get("need") or "").strip()
                starts = [p for p in pending if said and (p.startswith(said) or said.startswith(p))]
                same = [p for p in pending if said and _words(p.split(":")[0]) == _words(said.split(":")[0])]
                need = (next((p for p in pending if p == said), None) or (starts[0] if len(starts) == 1 else None)
                        or (same[0] if len(same) == 1 else None) or (pending[0] if len(pending) == 1 else None))
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                named = [i for i in ids if i]
                if len(named) != len(set(named)):
                    unread.append(f"{need} -> the pages point back at each other ({' -> '.join(named)}), so no page gives the value")
                    continue
                self.chains[need] = [i for i in named if i in self.rows]
                self.ref_pages.update(i for i in ids if i in self.rows)
                value, last = str(chain.get("value") or "").strip(), (ids[-1] if ids else "")
                if links and last in self.rows and all(i in self.opened for i in ids) and not str(links[-1].get("find") or "").strip():
                    page = _dump(self.full.get(last) or self.rows.get(last) or {}).lower()
                    figures, words = re.findall(r"\d+(?:\.\d+)?", value), re.findall(r"[a-z]{3,}", value.lower())
                    if value and all(n in page for n in figures) and (not words or any(w in page for w in words)):
                        self.ref_values[need] = value
                        self.version += 1
                        continue
                    if last not in self.full and last not in read_for_value and self.left() > self.tail() + 4:
                        read_for_value.add(last)
                        got_facts = self.fill_facts([last], self.tail() + 4)
                        if got_facts:
                            seen += got_facts
                            refreshed = True
                            continue
                    unread.append(f"{need} -> {last} was opened, but its record does not show the value" + (f" ({value!r} is not on it)" if value else ""))
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get("about") or "") or "the listing it points at"
                    words = str(gap.get("find") or "").strip()
                    lost.append(f"{need} -> find {about}" + (f" (search: {words})" if words else ""))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target:
                known_gaps: set[int] = set()
                for number, (about, words) in enumerate(gaps):
                    hop = _words(about)[:80]
                    if (self.left() <= 12 or self.finished or self.hop_searches.get(hop, 0) >= 2
                            or self.link_searches >= 8 + 2 * max(0, len(self.refs) - 1)):
                        continue
                    before = self.link_searches
                    found = self.find_link(about, words)
                    self.hop_searches[hop] = self.hop_searches.get(hop, 0) + (self.link_searches - before)
                    reach += [i for i in found if i not in reach]
                    if self.trace and self.trace[-1].get("kind") == "link_known":
                        seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                        known_gaps.add(number)
                lost = [line for n, line in enumerate(lost) if n not in known_gaps]
            if refreshed and not reach and not self.finished:
                continue
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 12 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                fresh.add(ident)
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        live = {need for _, need in self.needs_now()}
        open_refs = [need for _, need in self.refs if need in live and need not in self.ref_values]
        missing = lost + unread + [f"{need} -> its chain has not reached a page that shows the value yet" for need in open_refs
                                   if not any(m.startswith(need + " ->") for m in lost + unread)]
        if early and missing:
            text += "[requirements whose value is not read yet] " + " | ".join(missing[:3]) + "\n"
        hold = not early and bool(open_refs) and self.trail_blocks < (3 if self.left() > 15 else 2) and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing, and its value is read. Still to do: "
                     + " | ".join(missing[:3]) + ". For a listing not found, search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it) and open the row of that shop whose page shows those values;"
                     " for a page opened without the value, open it with the tool that shows its attributes. Apply the value its page gives, then check out.")
        return text, hold

    def find_link(self, about: str, words: str) -> list[str]:
        """A chain's link that no record matches: Kyo searches for it (the shop and the values its description gives) and
        returns up to two unopened rows of that shop; [] when an opened page of that shop already shows those values."""
        def shops(row: dict[str, Any]) -> list[str]:
            return [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40
                    and re.search(r"seller|sold|shop|store|vendor|brand|maker|merchant", str(k), re.I)]
        low = about.lower()

        def of_shop(ident: str) -> bool:
            return any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low) for n in shops(self.rows.get(ident) or {}))
        named = {w for i in self.rows if of_shop(i) for n in shops(self.rows[i]) for w in re.findall(r"[a-z0-9]{3,}", n.lower())}
        stop = {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for", "of", "product"}
        values = {w for w in re.findall(r"[a-z0-9]+", (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - stop
        for ident in [i for i in self.opened if of_shop(i)]:
            text = _dump(self.full.get(ident) or self.rows.get(ident) or {}).lower()
            if values and all(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text) for v in values):
                self.trace.append({"kind": "link_known", "id": ident})
                return []
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        field = next((k for k in texts if set(_words(k).split()) & QUERY_NAMES), texts[0] if len(texts) == 1 else None)
        if not tool or not field or not words.strip():
            return []
        found: list[str] = []
        tries = [words, " ".join(w for w in words.split() if w.lower().strip(",.'") not in named
                                 and not (not named and w[:1].isupper() and re.search(r"(?<![A-Za-z0-9])" + re.escape(w.strip(",.'")) + r"(?![A-Za-z0-9])", about)))]
        for text in dict.fromkeys(t.strip() for t in tries if t.strip()):
            memo = " ".join(text.lower().split())
            if memo in self.link_seen:
                found = self.link_seen[memo]
            else:
                if self.link_searches >= 8 + 2 * max(0, len(self.refs) - 1) or self.left() <= 12:
                    break
                last = self.search_args.get(tool) or {}
                args: dict[str, Any] = {field: text}
                for key in schema.get("required") or []:
                    if key != field:
                        spec = props.get(key) or {}
                        if key in last:
                            args[key] = last[key]
                        elif isinstance(spec.get("enum"), list) and spec["enum"]:
                            args[key] = spec["enum"][0]
                        else:
                            return []
                faults: list[str] = []
                args = _shape(args, schema, "args", faults)
                if faults:
                    return []
                self.link_searches += 1
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    return []
                observation = reply.get("observation")
                self.observe(tool, observation)
                found = self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                self.searched.append(f"{tool} {_dump(args)[:90]} -> {len(found)} rows (Kyo, for a page a requirement points at)")
                self.trace.append({"kind": "link_search", "words": text[:80], "rows": found[:6]})
            if any(of_shop(i) for i in found):
                break
        rows = [i for i in found if i not in self.opened and of_shop(i)]
        rows = [i for i in rows if values & set(re.findall(r"[a-z0-9]{3,}", _dump(self.rows.get(i) or {}).lower()))] or rows
        return (rows or [i for i in reversed(self.seen_order) if i not in self.opened and of_shop(i)])[:2]

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 6, self.refused_at)
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
            cart, units = self.roles.get("cart"), max(self.cart_lines.get(i, 1) for i in lost)
            lines = {**{j: self.cart_lines.get(j, 1) for j in self.cart_ids() if j not in lost}, picks[0]: units}
            calls = self.cart_calls(lines, lost) if cart else None
            done = 0
            if calls and len(calls) + 1 <= self.left():
                for args in calls:
                    faults: list[str] = []
                    args = _shape(args, self.schemas.get(cart), "args", faults)
                    if faults:
                        break
                    reply = self.act(cart, args)
                    if reply is None or self.finished:
                        break
                    self.note_cart(cart, args)
                    self.observe(cart, reply.get("observation"))
                    done += 1
            if calls and done == len(calls):
                self.trace.append({"kind": "swap_done", "lost": lost, "put": picks[0], "steps": done})
                return text + f" Few steps were left, so Kyo made the swap itself ({done} steps): {', '.join(lost)} out, {picks[0]} in at quantity {units}. Check out now."
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            listed = ""
            cat_key = self.category_key()
            shelf = str((self.rows.get(lost[0]) or {}).get(cat_key) or "").strip().lower() if cat_key else ""
            price = self.price_of(picks[0])
            tool = self.roles.get("search")
            page_key = (self.shapes.get(tool or "") or {}).get("next_arg")
            known = next(((arg, args) for (t, arg), (args, shelves) in self.cap_args.items() if t == tool and shelf and shelf in shelves), None)
            if known and price is not None and self.left() > self.tail() + 6:
                arg, args = known
                faults: list[str] = []
                call = _shape({**{k: v for k, v in args.items() if k != page_key}, arg: price}, self.schemas.get(tool), "args", faults)
                reply = None if faults else self.act(tool, call)
                if reply is not None and not self.finished:
                    seen = reply.get("observation")
                    self.observe(tool, seen)
                    self.trace.append({"kind": "cheaper_listed", "args": call})
                    listed = (f"\n[Kyo listed this shelf under {price:g} itself (one step), with the call whose {arg} kept every row under it]\n"
                              + self.show(seen) + self.cheapest_here([seen]))
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing." + listed
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
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        rest = sum((self.price_of(j) or 0) * n for j, n in self.cart_lines.items() if j not in lost)
        units = max((self.cart_lines.get(i, 1) for i in lost), default=1)
        room = None if self.budget is None else (self.budget if self.brief.get("budget_scope") == "each" else self.budget - rest)
        keep = [j for j in self.cart_ids() if j not in lost]
        def pair_no(i: str) -> bool:
            a = self.item_for(i)
            for j in keep:
                b = self.item_for(j)
                if a and b and ((self.links.get(a) == b and self.fit_of(i, j) == "no") or (self.links.get(b) == a and self.fit_of(j, i) == "no")):
                    return True
            return False
        lost_items = {self.item_for(i) for i in lost} - {None}
        known = [i for i in self.pick_order() if i not in lost and i not in self.cart_ids() and self.buyable(i)
                 and (len(self.items) == 1 or not lost_items or self.item_for(i) in lost_items)
                 and self.judged.get(i, ("",))[0] == mark and self.judged[i][1] == "fits"
                 and (room is None or self.price_of(i) is None or self.price_of(i) * units <= room) and not pair_no(i)]
        if known and self.brief.get("pick") not in ("most", "least"):
            floor = min((self.price_of(i) for i in known if self.price_of(i) is not None), default=None)
            unsure = [i for i in self.full if i not in lost and self.judged.get(i, ("",))[0] != mark and self.buyable(i) and floor is not None
                      and self.price_of(i) is not None and self.price_of(i) < floor]
            if self.brief.get("pick") != "cheapest" or not unsure:
                self.trace.append({"kind": "swap_known", "lost": lost, "fits": known[:4]})
                return known[:4]
        if self.cost > self.spend_soft:
            return []
        top = None if room is None else room + (self.budget or 0) * self.buy_over
        records = []
        for ident in self.pick_order():
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            price = self.price_of(ident)
            if ((top is not None and price is not None and price * units > top) or pair_no(ident)
                    or (len(self.items) > 1 and lost_items and self.item_for(ident) not in lost_items | {None})):
                continue
            records.append(self.price_note(ident) + ("MIXED RECORD (" + self.messy(ident) + "). " if self.messy(ident) else "") + _dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        order = "opened first, cheapest first, by category in turn" if self.brief.get("pick") == "cheapest" else "newest first, by category in turn" if len(self.items) > 1 else "newest first"
        prompt += f"\n\nOTHER LISTING RECORDS SEEN ({order}):\n" + "\n".join(records)
        paired = []
        for i in lost:
            a = self.item_for(i)
            for r in self.relations:
                other = r["to"] if r["item"] == a else r["item"] if r["to"] == a else None
                if a and other:
                    paired += [j for j in keep if self.item_for(j) == other]
        if paired:
            prompt += ("\n\nTHE LINE IT IS PAIRED WITH (a replacement must also work with it, or match it on the tied property):\n"
                       + "\n".join(f"{j}: {self.record_of(j)}" for j in dict.fromkeys(paired)))
        got = self.ask_json(prompt, [LIGHT], 2400)
        if not _items((got or {}).get("fits")) and not (self.no_credit or self.broke or self.calls >= self.call_cap):
            got = self.ask_json(prompt, [HEAVY], 1200) or got
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if (ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]
                    and not pair_no(ident)):
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        for _, ident in fits:
            self.judged[ident] = (mark, "fits", "replacement review")
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        if self.audits >= 2 or self.audited == mark or self.cost > self.spend_soft:
            return ""
        self.audited, self.audits = mark, self.audits + 1
        self.refuse_blocks, self.confirmed, self.completed = 0, None, False
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.rows and str(i) not in self.opened and not self.sold_out(str(i))]
            mains = [j for j in self.cart_ids() + [k for k in self.seen_order if k in self.full]][:20]

            def misfit(i: str) -> bool:
                d = self.item_for(i)
                known = [v for m in mains if d in self.links and m != i and self.item_for(m) == self.links[d] for v in [self.fit_of(i, m)] if v != "?"]
                return bool(known) and all(v == "no" for v in known)
            rows = [i for i in rows if self.buyable(i)
                    and (self.budget is None or self.price_of(i) is None or self.price_of(i) <= self.budget * (1 + self.buy_over))
                    and not misfit(i)]
            if rows and self.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        order = [i for i in self.pick_order() if not self.sold_out(i)]
        dear = [i for i in order[30:] if i in self.full and self.budget is not None and (self.price_of(i) or 0) > self.budget][:6]
        for ident in order[: 30 - len(dear)] + dear:
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + self.price_note(ident) + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + self.price_note(ident) + _dump(_clip(self.rows.get(ident), 120))[:500]))
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        ties = [f"item {r['item']} " + ("must work with" if r["kind"] == "fits" else f"{r['kind']} {r['property']} as") + f" item {r['to']}"
                for r in self.relations]
        if ties:
            prompt += "\n\nPAIRS (under \"fits\", list listings that work together for each pair): " + " | ".join(ties)
        order = "opened first, cheapest first, by category in turn" if self.brief.get("pick") == "cheapest" else "newest first, by category in turn" if len(self.items) > 1 else "newest first"
        prompt += f"\n\nLISTING RECORDS SEEN ({order}):\n" + "\n".join(records)
        got = self.ask_json(prompt, [HEAVY, LIGHT], 1800)
        self.trace.append({"kind": "audit", "got": got})
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        if isinstance((got or {}).get("fits"), dict) and self.budget is not None:
            cap = self.budget * (1 + self.buy_over)
            got["fits"] = {k: [i for i in _items(v) if (self.price_of(str(i)) or 0) * self.items[min(max(1, _count(k) or 1), len(self.items)) - 1].qty <= cap]
                           for k, v in got["fits"].items()}
        end = got.get("ending") if isinstance((got or {}).get("ending"), dict) else {}
        self.ending = {"mark": mark, "say": str(end.get("say") or "").strip(),
                       "name": [str(i) for i in _items(end.get("name")) if str(i) in self.rows and not self.sold_out(str(i))][:1]}
        named = (got or {}).get("fits")
        for ids in named.values() if isinstance(named, dict) else []:
            for ident in [str(i) for i in _items(ids) if str(i) in self.full]:
                self.judged[ident] = (mark, "fits", "decline review")
        failed = (got or {}).get("fails")
        for ident, why in failed.items() if isinstance(failed, dict) else []:
            if str(ident) in self.full:
                self.judged[str(ident)] = (mark, "fails", str(why)[:80])
        return got

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        cap = None if self.budget is None else self.budget * (1 + self.buy_over)
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))
                          and (cap is None or self.price_of(str(i)) is None or self.price_of(str(i)) <= cap)][:4] for k, v in fits.items()}
        mains: dict[int, list[int]] = {}
        for dep, main in self.links.items():
            mains.setdefault(main, []).append(dep)
        cheapest = self.brief.get("pick") == "cheapest"
        price = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        for main, deps in mains.items():
            ms, ds = named.get(str(main)) or [], {dep: named.get(str(dep)) or [] for dep in deps}
            if not ms or not all(ds.values()):
                continue
            good = [y for y in ms if all(any(self.fit_of(x, y) != "no" for x in ds[dep]) for dep in deps)]
            if cheapest:
                good.sort(key=lambda y: price(y) + sum(min(price(x) for x in ds[dep] if self.fit_of(x, y) != "no") for dep in deps))
            named[str(main)] = good
            for dep in deps:
                kept = [x for x in ds[dep] if any(self.fit_of(x, y) != "no" for y in good)]
                named[str(dep)] = sorted(kept, key=lambda x: (self.fit_of(x, good[0]) == "no", price(x) if cheapest else 0)) if good else []
            if ms != good:
                self.trace.append({"kind": "audit_pairs", "main": main, "leads": ms, "paired": good})
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:3 if self.left() >= 10 and self.cost < self.spend_soft else 2]]
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
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        plain = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."
        mark = _dump([self.needs_now(), self.budget, self.ref_values])
        end = self.ending if self.ending.get("mark") == mark and self.brief.get("quiet_decline") is not True else {}
        names = [i for i in end.get("name") or [] if not self.sold_out(i)]
        if names and names[0] not in self.full and self.left() >= 2:
            self.open_rows(names[:1])
        names = [i for i in names if i in self.full]
        say = str(end.get("say") or "").strip() if names else ""
        known = _dump(self.full.get(names[0]) or {}) + _dump([self.budget, self.budget * (1 + self.buy_over) if self.budget is not None else None]) if names else ""
        if say and any(n not in known for n in re.findall(r"\d+(?:\.\d+)?", say)):
            say = f"Nothing in the store meets all of your requirements; the nearest is {names[0]}."
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "").strip()
            cap = spec.get("maxLength", 200)
            if text in self.rows or (name, key) in self.id_args:
                if names:
                    out[key] = names[0]
                continue
            if say:
                out[key] = say[:cap]
                continue
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = plain[:cap]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.id_key:
            return args, ""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
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
            hunted = ""
            key = (self.shapes.get(name) or {}).get("next_arg")
            at = args.get(key) if key else None
            searching = name not in {tool for job, tool in self.roles.items() if job != "search"}
            if searching and isinstance(at, int) and not isinstance(at, bool) and not re.search(r"offset|start|skip|from|cursor|token", str(key), re.I):
                spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(key) or {}
                low = spec.get("minimum") if isinstance(spec.get("minimum"), int) else spec.get("default") if isinstance(spec.get("default"), int) else 1
                query = name + _dump({k: v for k, v in args.items() if k != key})
                read = self.query_pages.get(query, set())
                gap = next((p for p in range(low, at) if p not in read), None)
                if gap is not None and not self.query_sorted.get(query):
                    trimmed += (f"\n[Kyo] Sent {key}={gap} instead of {key}={at}: pages {gap}" + (f"-{at - 1}" if at - 1 > gap else "")
                                + " of this search are unread, and a search's fitting rows sit mostly on its first pages. Page on from here.")
                    self.trace.append({"kind": "page_order", "tool": name, "asked": at, "sent": gap})
                    args = {**args, key: gap}
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
            if not ending and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
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
                    wider = self.wider_hold()
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
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    if self.links and self.fit_holds < 2:
                        lines = self.cart_ids()
                        assign: dict[str, int] = {i: n for i in lines for n in [self.item_for(i)] if n}
                        rest_lines = [i for i in lines if i not in assign]
                        rest_items = [n for n in range(1, len(self.items) + 1) if n not in assign.values()]
                        if len(rest_lines) == 1 and len(rest_items) == 1:
                            assign[rest_lines[0]] = rest_items[0]
                        misfits = []
                        for dep, main in self.links.items():
                            ds = [i for i, n in assign.items() if n == dep]
                            ms = [i for i, n in assign.items() if n == main]
                            if len(ds) != 1 or len(ms) != 1 or self.fit_of(ds[0], ms[0]) != "no":
                                continue
                            d, m = ds[0], ms[0]
                            fits_m = [i for i in self.seen_order if i != d and self.item_for(i) == dep and not self.sold_out(i) and self.fit_of(i, m) == "fit"][:3]
                            fits_d = [j for j in self.seen_order if j != m and self.item_for(j) == main and not self.sold_out(j) and self.fit_of(d, j) == "fit"][:3]
                            misfits.append(f"{d} (item {dep}) must be compatible with {m} (item {main}), but its own list of what it fits does not include {m}'s model code"
                                           + (f"; item {dep} listings seen that fit {m}: {', '.join(fits_m)}" if fits_m else "")
                                           + (f"; item {main} listings seen that {d} fits: {', '.join(fits_d)}" if fits_d else ""))
                        if misfits:
                            self.fit_holds += 1
                            self.trace.append({"kind": "fit_hold", "misfits": misfits})
                            out.append((cid, "Not run yet (no step used). " + " | ".join(misfits) + ". Change one of the pair so that they fit (each still meeting its own requirements), then check out.\nCart now: " + self.cart_text()))
                            continue
                    limit = None if self.budget is None else self.budget * (1 + self.buy_over)
                    on_cart = self.cart_ids()
                    priced = {i: self.price_of(i) for i in on_cart}
                    if limit is not None and on_cart and None not in priced.values() and self.cart_mark() not in self.price_held:
                        units = {i: self.cart_lines.get(i, 1) for i in on_cart}
                        total = sum(priced[i] * units[i] for i in on_cart)
                        each = self.brief.get("budget_scope") == "each"
                        over = [i for i in on_cart if priced[i] * units[i] > limit] if each else (on_cart if total > limit else [])
                        if over and not all(priced[i] > 20 * limit for i in on_cart):
                            self.price_held.add(self.cart_mark())
                            self.trace.append({"kind": "price_hold", "total": total, "limit": limit, "lines": priced})
                            out.append((cid, "Not run yet (no step used). " + ("A line is over the per-item limit " if each else f"The cart's total {total:g} is over the limit ")
                                        + f"{limit:g}: " + "; ".join(f"{i} {priced[i]:g} x {units[i]}" for i in over)
                                        + ". Replace that line with a cheaper listing that meets every requirement, or end without a purchase when no such listing exists."
                                        + "\nCart now: " + self.cart_text()))
                            continue
                    cheapest = self.brief.get("pick") == "cheapest"
                    if (cheapest or any(self.messy(i) for i in self.cart_ids())) and self.cart_mark() not in self.cheap_held and self.left() > 3:
                        mark = _dump([self.needs_now(), self.budget, self.ref_values])
                        cart = self.cart_ids()
                        cap = None if self.budget is None else self.budget * (1 + self.buy_over)
                        each = self.brief.get("budget_scope") == "each"
                        cheaper = []
                        for line in cart:
                            p, n, it = self.price_of(line), self.cart_lines.get(line, 1), self.item_for(line)
                            if p is None or it is None or any(it in (r["item"], r["to"]) for r in self.relations):
                                continue
                            rest = sum((self.price_of(j) or 0) * self.cart_lines.get(j, 1) for j in cart if j != line)
                            room = None if cap is None else (cap if each else cap - rest)
                            fitting = [i for i in self.pick_order() if i not in cart and self.buyable(i) and self.item_for(i) == it
                                       and self.judged.get(i, ("",))[0] == mark and self.judged[i][1] == "fits"
                                       and self.price_of(i) is not None and (room is None or self.price_of(i) * n <= room)]
                            pool = [i for i in fitting if self.price_of(i) < p and (not self.messy(i) or self.messy(line))] if cheapest else []
                            clean = [i for i in fitting if not self.messy(i)] if self.messy(line) else []
                            if pool:
                                best = min(pool, key=lambda i: self.price_of(i))
                                cheaper.append(f"{best} at {self.price_of(best):g} ({self.judged[best][2]}) is cheaper than the cart's {line} at {p:g}")
                            elif clean:
                                best = min(clean, key=lambda i: self.price_of(i)) if cheapest else clean[0]
                                cheaper.append(f"{best} at {self.price_of(best):g} fits and its record agrees with itself, while the cart's {line} at {p:g} has a mixed record ({self.messy(line)})")
                        if cheaper:
                            self.cheap_held.add(self.cart_mark())
                            self.trace.append({"kind": "cheap_hold", "lines": cheaper})
                            out.append((cid, "Not run yet (no step used). A better listing judged to fit every requirement is known: "
                                        + "; ".join(cheaper) + ". Swap it in at the same quantity, or, when its own page shows a requirement it breaks, keep the cart and check out again."
                                        + "\nCart now: " + self.cart_text()))
                            continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > self.spend_soft:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.left() > 5:
                        if (self.audited != _dump([self.needs_now(), self.budget, self.ref_values]) and self.audits < 2 and self.cost <= self.spend_soft and self.roles.get("search") and self.left() > self.tail() + 8
                                and time.monotonic() - self.started <= HUNT_BY):
                            tool = self.roles["search"]
                            schema = self.schemas.get(tool) or {}
                            props = schema.get("properties") or {}
                            field = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v
                                          and set(_words(k).split()) & QUERY_NAMES), None)
                            last = self.search_args.get(tool) or {}
                            room = min(2 * len(self.items) + 2, (self.left() - self.tail()) // 3)
                            pointed = {need for _, need in self.refs} | {need for _, need in self.link_needs}
                            asked = []
                            for number, item in enumerate(self.items, 1):
                                values = [n.split(":", 1)[-1].strip() for n in item.musts + item.told if n not in item.dropped and n not in pointed
                                          and not re.search(r"\b(?:not|no|never|unlike|other than|at least|at most|under|over|below|above|between|within|less|more|than)\b",
                                                        n.split(":", 1)[-1], re.I)][:3]
                                base: dict[str, Any] = {}
                                ready = bool(field)
                                for key in schema.get("required") or []:
                                    if key == field:
                                        continue
                                    options = (props.get(key) or {}).get("enum")
                                    own = next((o for o in options if isinstance(o, str) and item.shelf and o.strip().lower() == item.shelf.strip().lower()), None) if isinstance(options, list) else None
                                    if own is not None:
                                        base[key] = own
                                    elif key in last:
                                        base[key] = last[key]
                                    else:
                                        ready = False
                                values += [w for need in item.musts + item.told if need not in item.dropped and need not in pointed
                                           for w in self.store_words.get((number, need), [])[:1]]
                                for query in dict.fromkeys([f"{v} {item.name}" for v in values if v] + [item.name]) if ready else []:
                                    if room <= 0 or self.finished:
                                        break
                                    faults = []
                                    call = _shape({**base, field: query}, schema, "args", faults)
                                    known = tool + _dump(call)
                                    said = _dump({field: query})[1:-1].lower()
                                    if faults or known in self.read_cache or any(said + "," in s.lower() or said + "}" in s.lower() for s in self.searched):
                                        continue
                                    reply = self.act(tool, call)
                                    room -= 1
                                    if reply is None:
                                        break
                                    seen = reply.get("observation")
                                    self.observe(tool, seen)
                                    got = len(self.records_in(seen))
                                    self.read_cache[known] = (self.version, self.step, self.show(seen))
                                    self.searched.append(f"{tool} {_dump(call)[:90]} -> {got} rows (Kyo, before ending)")
                                    asked.append(f"{query!r} -> {got} rows")
                            self.trace.append({"kind": "decline_hunt", "searches": asked})
                            if asked:
                                hunted = "[Kyo searched before ending, one step each: " + "; ".join(asked) + "]\n"
                                trimmed += "\n" + hunted.strip()
                        review = self.audit()
                        if review:
                            out.append((cid, ("Not run yet. " + hunted if hunted else "Not run yet (no step used). ") + review))
                            continue
                        missing = [n for n in range(1, len(self.items) + 1) if not self.lead_items.get(str(n))] if len(self.items) > 1 else []
                        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
                            self.refuse_blocks = 1
                            self.trace.append({"kind": "refuse_blocked", "leads": list(self.leads), "missing": missing})
                            if len(self.items) > 1:
                                found = "; ".join(f"item {k}: {', '.join(v)}" for k, v in sorted(self.lead_items.items(), key=lambda kv: _count(kv[0]) or 0))
                                out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + found + ". Cart one fitting line for each requested item at its quantity (respecting the pick rule)"
                                            + (f", and search on for {', '.join(f'item {n}' for n in missing)}, which the review found nothing for" if missing else "")
                                            + ", then check out. End without a purchase only if some item has no listing that meets its requirements; then say which item in your text and send the ending call again."))
                            else:
                                out.append((cid, "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."))
                            continue
                        if self.leads and self.left() > 4 and self.refuse_blocks == 1:
                            self.refuse_blocks = 2
                            first = ([ids[0] for _, ids in sorted(self.lead_items.items(), key=lambda kv: _count(kv[0]) or 0) if ids]
                                     if len(self.items) > 1 else self.leads[:1])
                            if first and not missing and len(first) == len(self.items):
                                self.check_cart(first, self.spend_hard)
                                if self.verdicts.get(self.check_key(first)) and all(i in self.leads for i in first):
                                    self.confirmed = first[0]
                                    self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                    out.append((cid, f"Not run yet (no step used). An independent check of {', '.join(first)} says "
                                                + ("they meet every requirement together. Add each at its item's quantity" if len(first) > 1 else "it meets every requirement. Add it to the cart at the requested quantity")
                                                + " and check out; end without a purchase only if a record clearly breaks a requirement."))
                                    continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
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
            lacking = [n for n in range(1, len(self.items) + 1) if not self.lead_items.get(str(n))] if len(self.items) > 1 and self.leads else []
            if self.is_refuse(name, args) and (not self.leads or lacking):
                wider = self.wider_hold(lacking)
                if wider:
                    out.append((cid, hunted + wider.replace("Not run yet (no step used). ", "Not run yet. ", 1) if hunted else wider))
                    continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
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
                if self.is_refuse(name, args):
                    self.ending = {}
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
            plain = {tool for job, tool in self.roles.items() if job != "search"}
            if name not in plain and not ending:
                key = (self.read_shape(name, observation) or {}).get("next_arg") if isinstance(observation, dict) else None
                if key:
                    spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(key) or {}
                    low = spec.get("minimum") if isinstance(spec.get("minimum"), int) else spec.get("default") if isinstance(spec.get("default"), int) else 1
                    query = name + _dump({k: v for k, v in args.items() if k != key})
                    page = args.get(key, low)
                    if isinstance(page, int) and not isinstance(page, bool):
                        self.query_pages.setdefault(query, set()).add(page)
                    if page == low:
                        self.query_sorted[query] = self.price_sorted(observation)
            paged = self.more_pages(name, args, observation) if name not in plain and not ending else ""
            capped = ""
            if name not in plain and not ending and isinstance(observation, dict) and not paged:
                shape = self.read_shape(name, observation) or {}
                props = (self.schemas.get(name) or {}).get("properties") or {}
                key = shape.get("next_arg")
                at = args.get(key) if key else None
                low = (props.get(key) or {}).get("minimum") if key and isinstance((props.get(key) or {}).get("minimum"), int) else 1
                total = observation.get(shape["count"]) if shape.get("count") else None
                now = self.records_in(observation)
                if (isinstance(at, int) and not isinstance(at, bool) and at > low and shape.get("next") and observation.get(shape["next"]) is None
                        and isinstance(total, int) and not isinstance(total, bool) and now and total > len(now) and self.left() > self.tail() + 4):
                    faults = []
                    first = _shape({**args, key: low}, self.schemas.get(name), "args", faults)
                    seen_low = low in self.query_pages.get(name + _dump({k: v for k, v in args.items() if k != key}), set())
                    again = None if faults or seen_low or name + _dump(first) in self.read_cache else self.act(name, first)
                    if again is not None:
                        seen = again.get("observation")
                        self.observe(name, seen)
                        new = self.records_in(seen)
                        if len(new) >= len(now) and not {i for i, _ in new} & {i for i, _ in now}:
                            shown_first = self.show(seen)
                            self.read_cache[name + _dump(first)] = (self.version, self.step, shown_first)
                            self.paged_obs.append(seen)
                            self.query_pages.setdefault(name + _dump({k: v for k, v in args.items() if k != key}), set()).add(low)
                            self.trace.append({"kind": "page_one", "tool": name, "from": at})
                            paged += (f"\n[Kyo] {key}={at} was the last page of this search ({total} matches, {len(now)} shown there). "
                                      f"The same search at {key}={low}, one step:\n" + shown_first)
                text_key = next((k for k, v in args.items() if isinstance(v, str) and len(v.split()) >= 3 and set(_words(k).split()) & QUERY_NAMES), None)
                if (text_key and not now and not self.paged_obs and self.short_tries < 3 and name not in self.short_dead
                        and args[text_key].strip().lower() not in self.labels() and self.left() > self.tail() + 8):
                    short = " ".join(args[text_key].split()[:2])
                    faults = []
                    call = _shape({**args, text_key: short}, self.schemas.get(name), "args", faults)
                    if not faults and name + _dump(call) not in self.read_cache and not any(_dump(call)[:90] in s for s in self.searched):
                        self.short_tries += 1
                        again = self.act(name, call)
                        if again is not None:
                            seen = again.get("observation")
                            self.observe(name, seen)
                            got = self.records_in(seen)
                            shown_short = self.show(seen)
                            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_short)
                            self.searched.append(f"{name} {_dump(call)[:90]} -> {len(got)} rows (Kyo, shorter)")
                            self.trace.append({"kind": "shorter", "from": args[text_key], "to": short, "rows": len(got)})
                            if got:
                                paged += f"\n[Kyo] No listing holds all those words; the same search with {short!r}, one step:\n" + shown_short
                            else:
                                self.short_dead.add(name)
                                paged += f"\n[Kyo] No listing holds all those words, and the same search with {short!r} found nothing either (one step)."
            if name not in plain and not ending and isinstance(observation, dict) and any(isinstance(v, list) for v in observation.values()):
                cat_key = self.category_key()
                found = [r for seen in [observation] + self.paged_obs for _, r in self.records_in(seen)]
                kinds: dict[str, int] = {}
                for r in found:
                    label = str(r.get(cat_key) or "?") if cat_key else "?"
                    kinds[label] = kinds.get(label, 0) + 1
                shown = ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])[:3])
                self.searched.append(f"{name} {_dump(args)[:90]} -> {len(found)} rows" + (f" ({shown})" if shown else ""))
                self.search_args[name] = dict(args)
                shape = self.read_shape(name, observation) or {}
                param = shape.get("next_arg")
                total = observation.get(shape["count"]) if shape.get("count") else None
                if isinstance(total, int) and not isinstance(total, bool):
                    query = name + _dump({k: v for k, v in args.items() if k != param})
                    fresh = query not in self.query_rows
                    ids = self.query_rows.setdefault(query, set())
                    ids.update(i for seen in [observation] + self.paged_obs for i, _ in self.records_in(seen))
                    tail = self.paging(name, self.paged_obs[-1] if self.paged_obs else observation)
                    page = args.get(param) if param else None
                    per_page = len(self.records_in(observation))
                    numbered = isinstance(page, int) and not isinstance(page, bool)
                    pages_in = -(-total // per_page) if per_page else 0
                    by_rows = numbered and (bool(re.search(r"offset|start|skip|from|cursor", str(param), re.I)) or page > pages_in)
                    reach = max(len(ids), (page + per_page if by_rows else page * per_page) if numbered else 0)
                    if tail and tail[3] is None and total > reach:
                        self.window = max(self.window, reach)
                    if fresh and self.window and total > self.window:
                        capped = (f"\n[Kyo] This search matched {total} listings, but this store shows at most {self.window} rows for one search"
                                  f" (its pages stop there), so {total - self.window} of them never appear for these words, however far you page."
                                  " To reach them, search with narrower words: a wanted value, or a type, size or pack word.")
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
            text = self.show(observation) + paged + trimmed + capped
            if name not in plain and not ending and isinstance(observation, dict):
                prices = [p for p in (self.price_of(i) for i, _ in self.records_in(observation)) if p is not None]
                page_key = (self.read_shape(name, observation) or {}).get("next_arg")
                props = (self.schemas.get(name) or {}).get("properties") or {}
                mode = _dump({k: v for k, v in sorted(args.items()) if isinstance((props.get(k) or {}).get("enum"), list)})
                for arg, v in args.items():
                    if arg == page_key or isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0 or len(prices) < 2:
                        continue
                    above, below = any(p > v for p in prices), any(p < v for p in prices)
                    if above and not below:
                        continue
                    seen = self.cap_modes.setdefault((name, arg), {})
                    seen[mode] = not above
                    if not above:
                        cat_key = self.category_key()
                        shelves = {str(r.get(cat_key) or "").strip().lower() for _, r in self.records_in(observation)} if cat_key else set()
                        self.cap_args[(name, arg)] = (dict(args), shelves - {""})
                    if above:
                        works = [m for m, ok in seen.items() if ok and m != mode]
                        text += (f"\n[Kyo] {arg}={v:g} limited nothing here: rows priced above and below it came back"
                                 + (f". With {works[0]} it did keep every row at or under it" if works else "") + ".")
            if name not in plain and not ending:
                fields = {str(v).strip().lower() for _, r in self.records_in(observation) for v in r.values() if isinstance(v, str) and len(v) <= 60}
                worded = any(isinstance(v, str) and v.strip() and v not in self.rows and v.strip().lower() not in fields for v in args.values())
                text += self.cheapest_here([observation] + self.paged_obs, worded)
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
            if len(self.refs) > self.ref_seen and not self.finished:
                self.ref_seen = len(self.refs)
                chain, _ = self.trail(early=True)
                if chain:
                    text += "\n" + chain.strip()
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
        ids = self.cart_ids()
        priced = [(self.price_of(i), self.cart_lines.get(i, 1)) for i in ids]
        limit = None if self.budget is None else self.budget * (1 + self.buy_over)
        over = (limit is not None and bool(ids) and all(p is not None for p, _ in priced)
                and not all(p > 20 * limit for p, _ in priced)
                and (any(p * n > limit for p, n in priced) if self.brief.get("budget_scope") == "each" else sum(p * n for p, n in priced) > limit)
                and self.check_key(ids) in self.checked and self.verdicts.get(self.check_key(ids)) is False)
        if over:
            self.trace.append({"kind": "forced_buy_skipped", "lines": ids, "limit": limit})
        if buy and not over and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if not faults:
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
                if reply is None and not self.finished and self.ending and self.left() > 0:
                    self.ending = {}
                    faults = []
                    args = _shape(self.quiet(refuse, dict(self.brief.get("refuse_args") or {})), self.schemas.get(refuse), "args", faults)
                    reply = self.act(refuse, args) if not faults else None
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
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)}; the status lists the ones judged so far, and opening a page that has not changed since shows its record at no step)"

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
        if self.refs:
            chain, _ = self.trail(early=True)
            self.ref_seen = len(self.refs)
            if chain:
                intro += "\n\n" + chain.strip()
        intro += ("\n\nStart with one reply that holds one search per requested item (several calls in one reply), then open several"
                  " listings of each item in full (one side-by-side call when the store has one), so the interviewer can see which properties"
                  " these listings carry.\n\n" + self.status())
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(self.loop_cap, self.spend_hard)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish."})
                self.work(self.loop_calls + 10, self.spend_hard + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        for _ in range(3):
            try:
                idle = 0
                while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap
                       and (time.monotonic() - self.started < LAST_TURN
                            or (not self.cart_ids() and TASK_SECONDS - TIME_MARGIN - (time.monotonic() - self.started) > EMPTY_ROOM))):
                    if self.ready_to_ask():
                        summary = self.interview()
                        if summary:
                            self.chat.append({"role": "user", "content": summary})
                        continue
                    self.trim()
                    self.loop_calls += 1
                    mark = _dump([self.needs_now(), self.budget, self.ref_values])
                    tie_to: dict[int, int] = {}
                    tie_of = {(r["item"], r["need"]): r for r in self.relations if r["need"]}
                    partners: dict[int, str] = {}
                    if self.relations:
                        for r in self.relations:
                            tie_to.setdefault(r["item"], r["to"])
                        on_cart = self.cart_ids()
                        for m in {r["to"] for r in self.relations}:
                            line = next((i for i in on_cart if self.item_for(i) == m), None)
                            lead = next((i for i in list(self.lead_items.get(str(m), [])) + [j for j in self.pick_order() if self.item_for(j) == m
                                         and self.judged.get(j, ("", ""))[1] == "fits"] if not self.sold_out(i)), None)
                            if line or lead:
                                partners[m] = line or lead
                            for page, was in list(self.judged_with.items()):
                                if tie_to.get(self.item_for(page) or 0) == m and ((line and was != line) or (not line and not was and m in partners)):
                                    self.judged.pop(page, None)
                                    self.judged_with.pop(page, None)
                    fresh = [i for i in reversed(self.seen_order) if i in self.full and self.buyable(i) and i not in self.ref_pages
                             and self.judged.get(i, ("",))[0] not in (mark, f"{mark}#{len(_dump(self.full[i]))}")][:10]
                    tied_fresh = sum(1 for i in fresh if tie_to.get(self.item_for(i) or 0) in partners)
                    cart_due = bool(fresh) and bool(self.cart_ids()) and self.judge_flushed != mark
                    if (len(fresh) >= JUDGE_BATCH or tied_fresh >= 3 or cart_due) and self.cost < self.spend_soft and self.calls < self.call_cap - 12:
                        if cart_due:
                            self.judge_flushed = mark
                        needs = self.needs_now()
                        def tie_line(n: int, need: str) -> str:
                            tied = tie_of.get((n, need))
                            other = partners.get(tied["to"]) if tied else None
                            if other:
                                return (f" (judge it against item {tied['to']}'s listing {other} under PARTNER LISTINGS, and give"
                                        " \"partner_quote\": that listing's own words you compared with)")
                            return " (judged with the other item: answer \"not shown\")" if (n, need) in self.link_needs else ""
                        numbered = "\n".join(f"requirement {k} (item {n}: {self.items[n - 1].name}): {need}"
                                             + (f" (the page it points at gives: {self.ref_values[need]})" if need in self.ref_values else "")
                                             + tie_line(n, need) for k, (n, need) in enumerate(needs, 1))
                        batch = fresh[: 10 if len(needs) <= 3 else 6]
                        prompt = (JUDGE + ("" if self.links else "\nLeave out is, fits and more.") + "\n\nREQUEST:\n" + self.request
                                  + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + "\n\nNUMBERED REQUIREMENTS:\n" + numbered
                                  + "\n\nOPENED LISTINGS:\n" + "\n".join(f"{i}: " + _dump(_clip(self.full[i], 200))[:1200] for i in batch)
                                  + ("\n\nPARTNER LISTINGS:\n" + "\n".join(f"{p}: " + _dump(_clip(self.full.get(p) or self.rows.get(p), 200))[:1200]
                                                                       for p in list(dict.fromkeys(partners.values()))[:2]) if partners else ""))
                        room = 2400 if self.links or len(needs) > 3 else 1800
                        got = (self.ask_json(prompt, [LIGHT], room * 2)
                               or (self.ask_json(prompt, [HEAVY], room * 2) if not (self.no_credit or self.broke or self.calls >= self.call_cap) else None)
                               or {})
                        pages = got.get("pages") if isinstance(got.get("pages"), dict) else {}
                        for ident in batch:
                            entry = pages.get(ident) if isinstance(pages.get(ident), dict) else {}
                            record = self.full.get(ident) or self.rows.get(ident) or {}
                            item = _count(entry.get("item"))
                            if item is None and entry and len(self.items) == 1:
                                item = 1
                            said = entry.get("needs") if isinstance(entry.get("needs"), dict) else {}
                            if self.links and item and (item in self.links or item in self.links.values()):
                                self.facts_of[ident] = {"is": [str(x) for x in _items(entry.get("is")) if _on_page(x, record)][:6],
                                                        "fits": [str(x) for x in _items(entry.get("fits")) if _on_page(x, record)][:30],
                                                        "more": entry.get("more") is True}
                            proof: dict[str, tuple[str, str]] = {}
                            broken, unshown = [], []
                            mixed_here: list[str] = []
                            used = ""
                            for k, (n, need) in enumerate(needs, 1):
                                tied = tie_of.get((n, need))
                                other = partners.get(tied["to"]) if tied else None
                                if n != item:
                                    continue
                                if (n, need) in self.link_needs and not other:
                                    if tied:
                                        unshown.append(f"{need} (judged once item {tied['to']} is chosen)")
                                    continue
                                e = said.get(str(k)) if isinstance(said.get(str(k)), dict) else {}
                                v, quote = str(e.get("v") or "").strip().lower(), str(e.get("quote") or "").strip()
                                shown = bool(quote) and _on_page(quote, record)
                                other_words = str(e.get("mixed") or "").strip()
                                if shown and v.startswith("met") and other_words and _on_page(other_words, record):
                                    mixed_here.append(f"{need} ({other_words[:60]})")
                                if other:
                                    used = other
                                    theirs = str(e.get("partner_quote") or "").strip()
                                    code = self.fit_of(ident, other) if tied["kind"] == "fits" else "?"
                                    if code in ("fit", "no"):
                                        v, shown = ("met" if code == "fit" else "broken"), True
                                        quote = quote if quote and _on_page(quote, record) else f"code check against {other}"
                                    elif (tied["kind"] == "same" and shown and theirs and _on_page(theirs, self.full.get(other) or self.rows.get(other) or {})
                                          and re.sub(r"[^a-z0-9]+", "", quote.split(":")[-1].lower()) == re.sub(r"[^a-z0-9]+", "", theirs.split(":")[-1].lower())):
                                        v = "met"
                                proof[need] = (v if shown else "not shown", quote[:120] if shown else "")
                                if shown and v.startswith("broken"):
                                    broken.append(f"{need} (page: {quote[:40]})")
                                elif not (shown and v.startswith("met")):
                                    unshown.append(need)
                            if not entry or item is None:
                                kind, why = "unknown", "not judged"
                            elif item == 0 or not 1 <= item <= len(self.items):
                                kind, why = "fails", "not a listing of a requested item"
                            elif broken:
                                kind, why = "fails", broken[0]
                            elif unshown:
                                kind, why = "unknown", "not shown: " + "; ".join(unshown)
                            else:
                                kind, why = "fits", "page review: " + "; ".join(q for _, q in proof.values() if q)[:60]
                            self.judged[ident] = (mark if kind != "unknown" else f"{mark}#{len(_dump(self.full[ident]))}", kind, why[:80])
                            self.mixed[ident] = (mark, mixed_here)
                            if proof:
                                self.proofs[ident] = proof
                            if item and 1 <= item <= len(self.items):
                                self.item_of[ident] = item
                            if item in tie_to:
                                self.judged_with[ident] = used
                        self.trace.append({"kind": "judge", "ids": batch, "got": _clip(pages, 80)})
                    status = self.status()
                    last = self.chat[-1]
                    if last.get("role") == "user":
                        sent = self.chat[:-1] + [{**last, "content": str(last.get("content") or "") + "\n" + status}]
                    else:
                        sent = self.chat + [{"role": "user", "content": status}]
                    message = self.think(sent, [LIGHT, SPARE], 1600, self.offered())
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
                        self.chat.append({"role": "user", "content": "Call one of the tools to continue."})
                        if idle >= 3:
                            break
                        continue
                    idle = 0
                    committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
                    if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > 8 and time.monotonic() - self.started <= ASK_BY:
                        self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                        for call in calls:
                            self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                        summary = self.interview()
                        self.chat.append({"role": "user", "content": summary or "The interviewer had nothing to ask."})
                        continue
                    self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
                    replies = self.run_calls(calls)
                    for cid, text in replies:
                        self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})

                return
            except Exception as exc:
                self.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
                at = next((n for n in range(len(self.chat) - 1, -1, -1) if self.chat[n].get("tool_calls")), None)
                if at is not None:
                    done = {m.get("tool_call_id") for m in self.chat[at + 1:]}
                    self.chat += [{"role": "tool", "tool_call_id": c["id"], "content": "The call may not have run: something went wrong on the way. Look at the state before you repeat it."}
                                  for c in self.chat[at]["tool_calls"] if c["id"] not in done]
                if self.finished or self.broke:
                    return

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
