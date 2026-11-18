"""Shopping agent: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

STEPS OF AN EPISODE (World.run) AND THE CLASS FOR EACH
  Step 0  read the request ................................ Intake
  Step 1  search and open listings before asking .......... the light model, helped by Browser
  Step 2  ask the shopper and read the answers ............ Talk
  Step 3  search again with the new needs ................. the light model, helped by Browser
  Step 4  choose, check and finish ........................ Checkout
  Every call the model sends:  Referee (admit, hold) -> Store (act) -> Results (learn, reply)
  At any step: a note from the shopper inside a result is read by Talk; the steps the request asks for along the
  way (an approval, a message) are carried out by Talk just before the order or the ending.

PARTS EVERY STEP USES
  Settings (fixed values) | Clock (steps, time) | Models (model calls, spend) | Store (store calls)
  Toolbox (the store's tools) | Order (what the shopper wants, the ledger) | Catalog (listings seen) | Cart
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
# shorter-timeout clients for the last minute of an episode, keyed by their timeout in seconds
_late_llm: dict[int, ProxyClient] = {}


def _llm_for(room: float) -> ProxyClient:
    """The model client for a call that has `room` seconds to finish in: the usual 50 s one while there is time, else
    one whose timeout is the room rounded down to whole 5 s."""
    if room >= 50:
        return _llm
    cap = max(5, int(room // 5) * 5)
    return _late_llm.setdefault(cap, ProxyClient(timeout=cap, max_retries=1, retry_delay=1))


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts even with other describing words around them. A value with different words does not count (an abbreviation or code is not the word it stands for, and two values that share only a describing word are different values). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (the property of one component says nothing about the same property of the product as a whole), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.
- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.
- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.
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

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "max_price": the most ONE unit of this product may cost when the request sets a price limit for that product alone (for example a limit "on each" of that kind), as a number in the budget's unit, else null, "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it (an occasion named in the request becomes an occasion must with that word; a measure named with a part becomes that part's measure, with its number and unit); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing (the property, the rule such as being unlike or the same as, and the full description of the other listing, all in one must): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. A closed list of allowed values ("A or B", "pick it from A and B only") is ONE must "property: A or B", never one must per value, since separate musts would each be required. A product that must fit something the shopper already owns, named by a model code, gets the must "compatible with: <that model code>". Never put price, budget, stock, quantity or delivery timing into musts. A must needs its value in the request (or on the listing it points at): a property whose value the shopper or someone else will only give later is not a must; it goes under "ask_first". A rule that ties this product's property to the property of ANOTHER requested product (it applies except when, or only when, the other product has some value) stays ONE must worded with its condition, and also goes under "conditions". The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself (a product said to be for a kind of wear or made for a kind of event, or an occasion the request says it has to have) IS a must on that property, valued with the request's word for the use, unless the store's category label already names it; an event, person or setting the product is bought for (a particular event it will be worn at, a person it is for), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"pointers": a list with one entry for each must whose value is not stated but must be read from another listing that is not part of the order (a listing on a page the request names, one a named shop sells, or one that someone already picked), each {"item": item number, "must": that must copied exactly, "property": the property whose value is read, "relation": "same" (the same value), "goes with" (the value that goes with or pairs with it), "within" (within some distance of it), "unlike" (a different value), "kind": the kind of product the other listing is (its own kind, which is often not the kind being bought), "shop": the shop named for it or null, "with": {its properties named in the request: values}, "follow": true when the request says to keep going through every listing a page sends the reader on to, else false}; [] when none. A rule tied to another product of this order is never a pointer.
"conditions": a list with one entry for each rule that ties one requested product's property to another requested product's property, each {"item": number of the product the rule constrains, "need": "property: value" it must have, "other_item": number of the product it depends on, "other": "property: value" of that product, "form": "except_when" when the need applies unless the other product has that value, "only_when" when the need applies only if the other product has that value}; [] when none.
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don't mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value. Read every story sentence for the situation it describes, and name the one property that situation would decide:
- a space the product must fit or fill (a gap, a wall, a corner, a measured spot, something it must line up with or cover): the dimension that space limits (width, depth or height);
- who puts it together and how fast: its assembly;
- being seen by others or on show: its looks (colour or pattern), or the visible part the story is about;
- being watched or read for a long time or from a distance: its screen or display;
- joining something the shopper already owns, or a socket: its connection (plug type, connector, impedance or power);
- interest in what is inside or how it works: its internals (processor or memory);
- how it is carried, stored, kept or poured: its packaging or pack type;
- a recipe, plan or container worked out in advance: its amount, size or variety;
- using it up, filling it or deleting things from it: its capacity (storage or volume);
- someone whose taste decides, the person meant to use or wear it: the property that person's view settles (gender, audience, style or fit).
A story's hints are usually about one product only: put that product's hinted property first in its list, and give that product the first message. A sentence saying the shopper has not decided yet, or does not mind anything else, hints at nothing.
Name a hinted property by the listed key of that item that is exactly that property, with all of its words (the key for the part the story is about, not a shorter key for the whole); a property no listing of the item carries comes after the item's three most carried keys.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; once five or more listings were read, avoid keys seen on only one or two of them.
Skip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling (a key written in joined or underscored words becomes those words separated by spaces, and is never shortened to its last word); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}, each item's picks in rank order"""


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "unclear": [numbers of the questions whose answer shows the message lost the shopper or wants it said once more], "budget": new total limit as a number or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit (a limit with its unit, a plain value, an exclusion). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units (a limit keeps the noun it counts).
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts even with other describing words around them, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (the property of one component says nothing about the same property of the product as a whole), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts even with other describing words around them, limits include the edge); a key for a different property, even a related one (the property of one component says nothing about the same property of the product as a whole), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property's value itself (also when the requirement asks for a value other than or unlike that one: the page's value is what it is measured against), else false, "value": the property's value as stated on that last page, or ""}]}
A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (a described pack size and pack kind rule out the same shop's listing that has the count but not the pack kind); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop's name and the description's values, as a few search words.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

POINT_READ = """A shopping agent has to read one property's value from another listing before it shops. The request describes the listing that holds the value (LOOKING FOR). A listing's note ("good to know" or similar) may say that the value is to be read from a further listing instead; then that further listing is the one to look for next.
Read each record below and return ONE JSON object and nothing else: {"readings": [{"id": the record's id, "match": true when the record is the listing LOOKING FOR describes: the same shop and the same kind of product, and nothing the record shows contradicts a detail LOOKING FOR gives (a detail the record does not show at all is no contradiction; a different value for it is), "value": the record's own value for PROPERTY when its note does not send the reader elsewhere for PROPERTY, else null, "pair": when the record pairs that value with another value (it "goes together with" one, or a rule pairs values), the value paired with it, else null, "next": when the record's note sends the reader to another listing for PROPERTY: {"shop": that listing's shop or null, "kind": its kind of product, "with": {property: value, for each property the note gives for it}, "said": the note's sentence that sends the reader there, copied word for word}, else null}]}
When LOOKING FOR carries "said", that sentence is how the previous page described this listing: the listing must fit every detail it gives.
The shop is the seller's name the note gives for that listing (a proper name, usually right before the kind of product or after "from" or "by"); give it whenever the note names one. A value can be stated in the record's attributes, its note, its title or its description. A record that contradicts any detail of the description is not the listing; a note about a different property than PROPERTY says nothing about PROPERTY; a note saying a version with some value exists elsewhere, or that this is not some version, only rules records out. A record whose own attributes state a value for PROPERTY while its note sends the reader elsewhere for PROPERTY gives "value": null and its "next"."""

POINT_PARSE = """A shopper answered a question by pointing at another listing for the value instead of stating it. Read the requirement below and return ONE JSON object and nothing else: {"property": the property whose value is to be read, "relation": "same" (the same value), "goes with" (the value that goes with or pairs with it), "within" (within some distance of it), "unlike" (a different value), "kind": the kind of product the other listing is, "shop": the shop or seller named for it or null, "with": {property: value, for each property the requirement gives for that other listing}}"""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

LINE_PICKS = """A shopping agent wants to end without buying a several-product order, because some product looked impossible to fill. Such an ending is scored right only when a product truly has no acceptable listing. For EACH requested item, pick the best listing among the records shown, with these reading rules: """ + READING + """
A listing is ruled out for an item only by its own record: another kind of product than the item, a stated value that breaks one of the item's requirements, or no stock. A requirement that no part of a record speaks to is open, not broken. Leave an item without a pick (null) only when every listing of its kind shown is ruled out.
Return ONE JSON object and nothing else: {"picks": {"item number": listing id or null}, "reason": "one sentence"}"""

FIT_JUDGE = """A shopping agent's cart holds a device and parts that must work with that device. Judge each PART LINE from its own record only.
- "yes": the part's own compatibility statement covers the device's model: it names that model, or names a series that covers it and does not exclude it. Model codes may be written with or without spaces or hyphens. In a series the letters x stand for digits in exactly those places and nowhere else: a series "AB12xx" covers AB1200 to AB1299, and not AB2210 or AB1300.
- "no": the part's own record has a compatibility statement and the device's model is not covered by it (or is excluded). A sentence about a different part number (that some other part also fits a model) is about that other part and does not add to this listing's list; a pointer to another part "for the rest" does not add models either.
- "unknown": only when the part's record says nothing at all about which models it works with.
Also name OTHER listings shown below whose own compatibility statement covers the device.
Return ONE JSON object and nothing else: {"lines": [{"id": the part line's listing id, "fits": "yes", "no" or "unknown", "why": "a few words"}], "others": [ids of other listings that work with the device, at most 4]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""

GUARD_KEYS = 3

GUARD_SHARE = 0.0

GUARD_GAP = 0.34

NOT_PROPERTY = "\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b"

CATCH_ALL = "^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b"

WIDEN_EACH = 6


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        # a listing sent as one line of text is cut field by field, so its attributes stay in view
        record = _flat_record(value)
        return _clip(record, limit, depth + 1) if record else value[:limit] + "..."
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

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search("(?:ch|sh|ss|x|z)es$", w) else w[:-1] if len(w) > 3 and w.endswith("s") and (not w.endswith("ss")) else w
        if len(w) >= 3 and w not in ("and", "for", "the", "with"):
            out.add(w)
    return out


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


def _flat_record(text: str) -> dict[str, Any] | None:
    """A listing written as one line of "key: value" parts joined by " | " (race 15 sent rows and pages so), as a record;
    a part whose value is "name=value; name=value" becomes a nested block. None for any other text."""
    if not isinstance(text, str) or text.count(" | ") < 2:
        return None
    record: dict[str, Any] = {}
    for part in text.split(" | "):
        key, sep, value = part.partition(": ")
        key, value = key.strip(), value.strip()
        if not sep or not key or len(key) > 40:
            continue
        block: dict[str, str] = {}
        last = None
        for piece in value.split("; ") if "=" in value else []:
            name, eq, got = piece.partition("=")
            if eq and name.strip() and len(name.split()) <= 4:
                last = name.strip(); block[last] = got.strip()
            elif last:
                block[last] += "; " + piece.strip()   # a value that itself holds "; "
        if len(block) >= 2 or (block and len(value) <= 80):
            record[key] = block
        elif value in ("True", "False"):
            record[key] = value == "True"
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            record[key] = float(value) if "." in value else int(value)
        else:
            record[key] = value
    return record if len(record) >= 3 else None


def _lift(node: Any, depth: int = 0) -> Any:
    """A store reply with its listings written as lines of text, turned into the same records a structured store
    sends, so every reader of the reply sees one form (race 15 sent its rows and pages as text)."""
    if depth <= 7 and isinstance(node, str):
        return _flat_record(node) or node
    if depth <= 7 and isinstance(node, dict):
        return dict(zip(node.keys(), (_lift(part, depth + 1) for part in node.values())))
    if depth <= 7 and isinstance(node, list):
        return [_lift(part, depth + 1) for part in node]
    return node


def _show(observation: Any) -> str:
    text = _dump(_clip(observation, Settings.SHOW_CHARS))
    if len(text) > Settings.OBS_LIMIT:
        text = text[:Settings.OBS_LIMIT] + "...(cut)"
    return text


# ==============================================================================================================
# THE PARTS EVERY STEP USES
# ==============================================================================================================


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.max_price: float | None = None   # a price limit on one unit of this item alone
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Settings:
    """Fixed values for one episode: models, limits on calls, steps and spend, waits, and how much is shown."""
    LIGHT = "anthropic/claude-haiku-5.5"
    HEAVY = "anthropic/claude-sonnet-5.5"
    SPARE = "openai/gpt-6-luna"

    LOOP_CALLS = 42
    ALL_CALLS = 60
    SPEND_SOFT = 0.055
    SPEND_HARD = 0.078
    HEAVY_ROOM = 0.02           # the heavy model is used only while this much spend is left under SPEND_HARD
    # waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
    WAITS = (4, 10, 20, 35)
    WAIT_TOTAL = 45
    SHOW_CHARS = 450
    OBS_LIMIT = 15000
    HISTORY_LIMIT = 150000
    GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
    AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
    CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
    # wall clock: the validator stops an episode at 300 s (race 15: 28 of 180 runs were cut off with no order, all on
    # tasks allowing 92+ steps), so Choji plans on finishing END_SPARE seconds before that
    RUN_SECONDS = 300.0
    END_SPARE = 20.0
    LEAST_CALL = 6.0            # no model call starts with fewer seconds of room than this
    INTERVIEW_UNTIL = 160.0     # no interview starts after this many seconds
    QUESTIONS_UNTIL = 205.0     # an interview sends no further question after this
    HURRY_FROM = 195.0          # from here the status line tells the model to check out
    LOOP_UNTIL = 250.0          # the model loop stops here and the code's own finish ends the episode
    # properties of one item named in one message. Race 15: a list naming a hidden property drew it in 37 of 45 replies
    # that were not puzzled, wherever it stood in the list (18 times not first); Choji's one property per message drew
    # 49% of hidden needs against 59% for an agent sending lists of 4-8
    LIST_KEYS = 7
    LIST_KEYS_AFTER_UNCLEAR = 4


class Clock:
    """Steps and time: steps used and allowed, model-loop turns, and seconds spent waiting on failed model calls."""
    def __init__(self, max_steps: int):
        self.max_steps = max_steps
        self.step = 0
        self.waited = 0
        self.loop_calls = 0
        self.misses = 0
        self.began = time.monotonic()

    def left(self) -> int:
        return self.max_steps - self.step

    def seconds(self) -> float:
        return time.monotonic() - self.began

    def room(self) -> float:
        """Seconds a model call may still take before the planned end."""
        return Settings.RUN_SECONDS - Settings.END_SPARE - self.seconds()


class Models:
    """Model calls: sends a chat to a model and resends it after a passing failure; counts calls and spend, and notes
    when the proxy has run out of credit. `halted` says whether the store has stopped the episode for lack of credit."""
    def __init__(self, clock: Clock, trace: list[dict[str, Any]], halted: Callable[[], bool]):
        self.clock = clock
        self.trace = trace
        self.halted = halted
        self.cost = 0.0
        self.calls = 0
        self.no_credit = False

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= Settings.ALL_CALLS:
                return None
            if self.clock.room() < Settings.LEAST_CALL:
                self.trace.append({"kind": "no_time_for_call", "at": round(self.clock.seconds(), 1)})
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most
                if not passing or tries >= (1 if status in (None, 408) else len(Settings.WAITS)) or self.clock.waited + Settings.WAITS[tries] > Settings.WAIT_TOTAL:
                    break
                if self.clock.room() - 1.2 * Settings.WAITS[tries] < Settings.LEAST_CALL:
                    break
                pause = Settings.WAITS[tries] * (0.8 + 0.4 * random.random())
                self.clock.waited += Settings.WAITS[tries]
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
        result = _llm_for(self.clock.room()).post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
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
            if self.halted() or self.no_credit or self.calls >= Settings.ALL_CALLS:
                return None
        return None

    def judges(self, limit: float = Settings.SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + Settings.HEAVY_ROOM <= Settings.SPEND_HARD:
            return [Settings.HEAVY, Settings.LIGHT]
        return [Settings.LIGHT] if self.cost <= Settings.SPEND_HARD else []


class Store:
    """Calls to the store: sends one tool call, keeps the step count in line with the store's, and notes when the
    episode has finished or the store has stopped it for lack of credit."""
    def __init__(self, binding: dict[str, Any], clock: Clock, toolbox: Toolbox, catalog: Catalog, trace: list[dict[str, Any]], finished: bool):
        self.binding = binding
        self.clock = clock
        self.toolbox = toolbox
        self.catalog = catalog
        self.trace = trace
        self.finished = finished
        self.broke = False
        self.recent: list[str] = []   # marks of the last calls since the last change: a repeat is not run

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.clock.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = dict(result.data)
                if isinstance(result.data.get("observation"), (str, list, dict)):
                    data["observation"] = _lift(result.data["observation"])
                self.clock.step = data["step"] if isinstance(data.get("step"), int) else self.clock.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name in self.toolbox.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.catalog.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
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
                    self.clock.step = int(found.group(1)) - 1
                    payload["step"] = self.clock.step + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.clock.step += 1
                    payload["step"] = self.clock.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None


class Toolbox:
    """The store's tools: schemas, the job each tool does, and what code has learned about their arguments and results
    (which view shows attributes, the shape of row lists). The order's brief names the jobs and the fixed arguments."""
    def __init__(self, view: dict[str, Any], models: Models, order: Order, trace: list[dict[str, Any]]):
        self.models = models
        self.order = order
        self.trace = trace
        self.tools = [t for t in view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        self.roles: dict[str, str] = {}
        self.id_args: set[tuple[str, str]] = set()
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.compare_ok = 0
        self.compare_bad = 0

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

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

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

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view."""
        return self.roles.get("offer") or self.roles.get("open")

    # ---------- tool use ----------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("buy"):
            return False
        fixed = self.order.brief.get("buy_args") if isinstance(self.order.brief.get("buy_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get("refuse"):
            return False
        fixed = self.order.brief.get("refuse_args") if isinstance(self.order.brief.get("refuse_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

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
        got = self.models.ask_json(prompt, [Settings.LIGHT, Settings.HEAVY], 2500) or {}
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

    def attributes_note(self) -> str:
        """The ledger's line on which tool shows listing attributes and which show none, once code knows it."""
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            return (f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                    + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return ""


class Order:
    """What the shopper wants: the request, its reading (brief), the items and their needs, budget, steps along the way,
    needs that point at other listings, and the changes the shopper has made; written out as the ledger the model sees.
    `tools_note` gives the ledger's line on which store tool shows listing attributes."""
    def __init__(self, view: dict[str, Any], models: Models, trace: list[dict[str, Any]], tools_note: Callable[[], str]):
        self.models = models
        self.trace = trace
        self.tools_note = tools_note
        self.view = view
        self.request = str(view.get("query") or "")
        self.brief: dict[str, Any] = {}
        self.items: list[Item] = []
        self.budget: Any = None
        self.process: list[dict[str, Any]] = []
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.dropped: set[str] = set()
        self.version = 0
        self.notes: list[str] = []
        self.bound: list[tuple[int, int]] = []
        # rules tying one item's need to another item's value: {"item", "need", "other_item", "other", "form"}
        self.conditions: list[dict[str, Any]] = []
        # which needs are only described in the shopper's words (no listing attribute states them); World wires it to the catalog
        self.described: Callable[[int, str], bool] = lambda number, need: False
        self.link_searches = 0

    def unmix(self) -> None:
        """Musts that only describe the listing a lookup points at (read as musts of the product) move into that lookup."""
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.models.ask_json(prompt, [Settings.HEAVY, Settings.LIGHT], 1500) or {}
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
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.described(number, m)]
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]})" if m in self.ref_values else "") for m in live if m not in told]
            head = f"item {number}: {item.name}" if many else f"item: {item.name}"
            if item.shelf:
                head += f" (store label: {item.shelf})"
            head += f", quantity {item.qty}"
            text = head + "; requirements: " + ("; ".join(needs) if needs else "none beyond the kind of product")
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + "; ".join(told)
            if item.free:
                text += "; shopper does not mind: " + ", ".join(item.free)
            if item.max_price is not None:
                text += f"; price: at most {item.max_price:g} for one unit of this item"
            parts.append(text)
        for rule in self.conditions:
            a, b = rule["item"], rule["other_item"]
            prop = rule["need"].partition(":")[0].strip()
            when = rule["other"].replace(":", " is", 1)
            if rule["form"] == "only_when":
                parts.append(f"rule tying two items: only when item {b}'s {when}, item {a} must have {rule['need']} (otherwise item {a}'s {prop} is free); an item {b} listing without that value avoids the rule")
            else:
                parts.append(f"rule tying two items: item {a} must have {rule['need']}, except when item {b}'s {when} (then item {a}'s {prop} is free); either way passes, and when no item {a} listing has that {prop}, the way through is an item {b} listing with that value")
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
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        note = self.tools_note()
        if note:
            parts.append(note)
        return "\n".join("- " + p for p in parts)

    def needs_now(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def check_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.version}"

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ""):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or "") | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)


class Catalog:
    """Listings seen: search rows and full records, how to read their id, price, stock and category label, what was
    opened, sold out or unavailable, and paging state. The tools say which keys hold what; the order gives the items."""
    def __init__(self, toolbox: Toolbox, order: Order, trace: list[dict[str, Any]]):
        self.toolbox = toolbox
        self.order = order
        self.trace = trace
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.opened: set[str] = set()
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.searched: list[dict[str, Any]] = []
        self.link_seen: dict[str, list[str]] = {}

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """A listing's id, read with the id key learned from result rows; None for anything else."""
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.toolbox.id_key and _id_like(node.get(self.toolbox.id_key), True):
            return node[self.toolbox.id_key].strip()
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
        if self.toolbox.id_key and all(_id_like(r.get(self.toolbox.id_key), True) for r in rows):
            return True
        found = _learn_id(rows)
        if found:
            self.toolbox.id_key = found
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
                node = [_flat_record(entry) or entry for entry in node]
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
            elif isinstance(node, str) and depth > 0:
                # a listing sent as one line of text is read like a structured record
                record = _flat_record(node)
                if record:
                    walk(record, depth + 1, in_list)

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

        for schema in self.toolbox.schemas.values():
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
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.order.items]
        matched = {label for label in groups if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted) or self.order.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: (-kv[1]["n"], len(kv[0]))):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                # all values different and long: free text, unless their words recur across listings
                recur = sum(len(set(re.findall("[a-z]{3,}", v)) & set(re.findall("[a-z]{3,}", " ".join(o for o in values if o is not v)))) > 0 for v in values)
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and recur * 2 < len(values):
                    continue
                # a second spelling of a kept key: its listings join the kept one
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot["per"].items():
                        twin["per"].setdefault(ident, text)
                        if text not in twin["values"]:
                            twin["values"].append(text)
                    twin["n"] = len(twin["per"])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]["n"])
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
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.order.items]
        matched = [ids for label, ids in groups.items() if any(_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted) or self.order.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[: max(1, len(self.order.items))]

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.toolbox.price_key) if self.toolbox.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.toolbox.stock_key) if self.toolbox.stock_key else None
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
            self.toolbox.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.toolbox.view_keys.get(ident, (-1, set()))
            self.toolbox.view_keys[ident] = (self.order.version, (have if version == self.order.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.toolbox.view_keys.get(ident, (-1, set()))
        keys = self.toolbox.tool_keys.get(name)
        return bool(keys) and version == self.order.version and keys <= have and ident not in self.gone

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.order.brief.get("pick") != "cheapest" or not self.toolbox.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def sold_out(self, ident: str) -> bool:
        """A note or a refusal named the listing as gone, or a flag of its own record turned from true to false."""
        return ident in self.unavailable or ident in self.gone

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else "(no record seen)"

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.order.brief.get("pick") == "cheapest":
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

    def cheaper_than(self, held: list[str]) -> str:
        """For a cheapest order's last check: per cart line, the in-stock listings of the same category priced below it,
        cheapest first; read ones with their attributes, unread ones as their search row (marked)."""
        label_key = self.category_key()

        def shelf_of(code: str) -> str:
            record = self.full.get(code) or self.rows.get(code) or {}
            return _words(str(record.get(label_key) or "")) if label_key else ""

        pool = list(dict.fromkeys(self.seen_order))
        blocks = []
        for line in held:
            cost, shelf = self.price_of(line), shelf_of(line)
            if cost is None or not shelf:
                continue
            below = sorted((code for code in pool if code not in held and not self.sold_out(code) and shelf_of(code) == shelf
                            and (self.price_of(code) or cost) < cost), key=self.price_of)
            read = [code for code in below if code in self.full][:8]
            unread = [code for code in below if code not in self.full][:4]
            text = [f"{code}: " + _dump(_clip(self.full[code], 160))[:900] for code in read]
            text += [f"{code} (search row only, attributes not read): " + _dump(_clip(self.rows.get(code), 100))[:300] for code in unread]
            blocks.append(f"CHEAPER LISTINGS OF THE KIND OF {line} ({cost:g}):\n" + ("\n".join(text) if text else "(none seen)"))
        return "\n\n".join(blocks) if blocks else "OTHER LISTINGS SEEN (newest first):\n" + self.others(held, 15)

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
        needs = self.order.needs_now()
        if not ids or not needs:
            return ""
        many = len(self.order.items) > 1
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

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall("[a-z0-9]+", x.lower()), re.findall("[a-z0-9]+", y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))

        both = [i for i in spec["per"] if i in slot["per"] and spec["per"][i].strip() and slot["per"][i].strip()]
        same = sum((1 for i in both if agree(spec["per"][i], slot["per"][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec["values"]) & set(slot["values"])) - {""})

    def described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.order.refs] or need in [str(x) for x in _items(self.order.brief.get("lookups"))]:
            return False
        prop, _, value = need.partition(":")
        if not prop.strip() or len(re.findall("[a-z0-9]+", value.lower())) < 4:
            return False
        if re.search("\\b(?:listing|page|shown|other than|unlike|same as)\\b", value, re.I):
            return False
        item = self.order.items[number - 1]
        if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.order.items if other is not item)):
            return False
        cat_key = self.category_key()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and (not self.order.owns(item, str(record.get(cat_key) or ""))):
                continue
            block = _fact_block(record) or {}
            if block:
                read += 1
                keys |= {_words(k) for k in block}
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.order.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.order.owns(item, str(record.get(cat_key) or "")) and _fact_block(record)))

    def item_ids(self, number: int) -> list[str]:
        cat_key = self.category_key()
        item = self.order.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.order.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ""))]

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [
                str(v).strip()
                for k, v in row.items()
                if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search("seller|shop|store|vendor|brand|maker|merchant", str(k), re.I)
            ]
            if ident not in self.opened and any((re.search("(?<![a-z0-9])" + re.escape(n.lower()) + "(?![a-z0-9])", low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [
                str(v).strip()
                for k, v in row.items()
                if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search("seller|shop|store|vendor|brand|maker|merchant", str(k), re.I)
            ]
            if any((re.search("(?<![a-z0-9])" + re.escape(n.lower()) + "(?![a-z0-9])", low) for n in names)):
                out.append(ident)
        return out


class Cart:
    """The cart and the finish: lines and quantities, when they changed or were viewed, checks run on them, sold-out
    lines, refusals and the holds used before an ending."""
    def __init__(self, catalog: Catalog, toolbox: Toolbox, clock: Clock):
        self.catalog = catalog
        self.toolbox = toolbox
        self.clock = clock
        self.lines: dict[str, int] = {}
        self.touched = False
        self.at = -1
        self.seen_at = -1
        self.view: Any = None
        self.said = ""
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.audited = False
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.note_named = False
        self.line_holds = 0
        self.fit_holds = 0
        self.held = 0
        self.refused_at = -1
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.late_held = False
        self.lines_offered = False   # the one hold of a several-product decline with a pick for every item
        self.confirmed: str | None = None
        self.completed = False
        self.buys = 0
        self.refusals = 0

    def mark(self) -> str:
        return _dump(sorted(self.lines.items()))

    def ids(self) -> list[str]:
        if self.view is not None and self.seen_at >= self.at:
            text = _dump(self.view)
            return [i for i in self.catalog.rows if i in text]
        return list(self.lines)

    # ---------- checks ----------
    def text(self) -> str:
        if self.view is not None and self.seen_at >= self.at:
            return _dump(_clip(self.view, 300))[:3000]
        # a cart view older than the last cart change would show lines that are gone
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.lines)

    def cart_sets(self, name: str) -> bool:
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.toolbox.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.touched = True
        self.at = self.clock.step
        spec = self.toolbox.schemas.get(name, {}).get("properties") or {}
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
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.catalog.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.catalog.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.catalog.rows:
                lines.append((value, loose_count))
        if whole:
            self.lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.lines[ident] = have
            else:
                self.lines.pop(ident, None)


# ==============================================================================================================
# STEP 0: READ THE REQUEST
# ==============================================================================================================


class Intake:
    """Step 0: reads the shopper's request once, before any store call, and sets up the episode from that reading:
    which tool does which job, the items and their needs, the budget, the message and step limits, and the steps
    the request asks for along the way."""
    def __init__(self, order: Order, toolbox: Toolbox, talk: Talk, clock: Clock, models: Models, trace: list[dict[str, Any]]):
        self.order = order
        self.toolbox = toolbox
        self.talk = talk
        self.clock = clock
        self.models = models
        self.trace = trace

    def read(self) -> None:
        brief = self.read_brief(self.prompt())
        self.order.brief = brief
        self.set_tools(brief)
        self.set_items(brief)
        self.set_limits(brief)

    def prompt(self) -> str:
        lines = []
        for tool in self.toolbox.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.order.request
        if self.order.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.order.view.get("opening"), 1500))
        return prompt

    def read_brief(self, prompt: str) -> dict[str, Any]:
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one (race 14: 3,467 reasoning tokens on one)
        brief = self.models.ask_json(prompt, [Settings.LIGHT], 6000) or {}
        if not self.models.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.models.ask_json(prompt, [Settings.HEAVY, Settings.LIGHT], 2500) or brief
        elif not self.models.no_credit and self.shared_shelf(brief):
            # two items on one store label: the heavy model reads the request again
            again = self.models.ask_json(prompt, [Settings.HEAVY], 2500) or {}
            if _items(again.get("items")) and isinstance(again.get("roles"), dict) and again["roles"].get("buy"):
                brief = again
        return brief

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get("shelf") or "")) for e in _items(brief.get("items")) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))


    def set_tools(self, brief: dict[str, Any]) -> None:
        """The job of each store tool, and the fixed arguments that make one tool check out or end without a purchase."""
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.toolbox.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.toolbox.schemas}
        for key in ("buy_args", "refuse_args"):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse")
        if buy and buy == refuse and (not brief["buy_args"] or not brief["refuse_args"] or brief["buy_args"] == brief["refuse_args"]):
            ask = (f"The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: "
                   + _dump(self.toolbox.schemas.get(buy)) + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, \"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
            fixed = self.models.ask_json(ask, [Settings.LIGHT, Settings.HEAVY], 1500) or {}
            if isinstance(fixed.get("buy_args"), dict) and isinstance(fixed.get("refuse_args"), dict) and fixed["buy_args"] != fixed["refuse_args"]:
                brief["buy_args"], brief["refuse_args"] = fixed["buy_args"], fixed["refuse_args"]

    def set_items(self, brief: dict[str, Any]) -> None:
        """The items with their firm needs, and the needs whose value comes from another listing."""
        for entry in _items(brief.get("items")):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get("quantity"))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get("musts")) if m]
            item = Item(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count, musts)
            cap = entry.get("max_price")
            if isinstance(cap, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", cap):
                cap = float(cap)
            if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
                item.max_price = float(cap)
            self.order.items.append(item)
        # items on one store label whose requirements do not clash are one item
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.order.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and _words(m.shelf) == _words(item.shelf)
                         and not any(_words(a.split(":")[0]) == _words(b.split(":")[0]) and _words(a.partition(":")[2]) != _words(b.partition(":")[2])
                                     for a in m.musts for b in item.musts)), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({"kind": "items_merged", "into": twin.name, "dropped": item.name})
        self.order.items = merged
        if not self.order.items:
            self.order.items = [Item("the requested product", "", 1, [])]
        # item references counted from 0 (a weaker reader does it): shift them all to count from 1
        refs = [(entry, key) for name, keys in (("fits", ("item", "with")), ("pointers", ("item",)), ("conditions", ("item", "other_item")))
                for entry in _items(brief.get(name)) if isinstance(entry, dict) for key in keys]
        if any(_count(entry.get(key)) == 0 for entry, key in refs):
            for entry, key in refs:
                if _count(entry.get(key)) is not None:
                    entry[key] = _count(entry.get(key)) + 1
        # a part that must work with another requested item (the device)
        for pair in _items(brief.get("fits")):
            if isinstance(pair, dict):
                part, main = became.get(_count(pair.get("item")) or 0), became.get(_count(pair.get("with")) or 0)
                if part and main and part != main and 1 <= part <= len(self.order.items) and 1 <= main <= len(self.order.items) and (part, main) not in self.order.bound:
                    self.order.bound.append((part, main))
        for rule in _items(brief.get("conditions")):
            if isinstance(rule, dict):
                number, other = became.get(_count(rule.get("item")) or 0), became.get(_count(rule.get("other_item")) or 0)
                need, value = str(rule.get("need") or "").strip(), str(rule.get("other") or "").strip()
                if number and other and number != other and need and value:
                    self.order.conditions.append({"item": number, "need": need, "other_item": other, "other": value,
                                                  "form": "only_when" if rule.get("form") == "only_when" else "except_when"})
        # a tied need left as a plain must would read as firm; it keeps its condition in the must's own words
        for rule in self.order.conditions:
            item = self.order.items[rule["item"] - 1]
            when = rule["other"].replace(":", " is", 1)
            worded = f"{rule['need']}, {'only when' if rule['form'] == 'only_when' else 'except when'} item {rule['other_item']}'s {when}"
            key, value = _words(rule["need"].split(":")[0]), _words(rule["need"].partition(":")[2])
            plain = [m for m in item.musts if _words(m.split(":")[0]) == key and value <= _words(m)
                     and not re.search(r"\b(?:except|unless|only|when|if)\b", m, re.I)]
            if plain or not any(value <= _words(m) for m in item.musts):
                item.musts = [m for m in item.musts if m not in plain] + [worded]
        brief["lookups"] = self.pointer_musts(brief)
        for lookup in brief["lookups"]:
            text = str(lookup).strip()
            if not any(text in item.musts for item in self.order.items):
                # the lookup copied loosely: the must on the same property stands for it
                key = _words(text.split(":")[0])
                text = next((m for item in self.order.items for m in item.musts if _words(m.split(":")[0]) == key), text)
            number = next((n for n, item in enumerate(self.order.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.order.refs:
                self.order.refs.append((number, text))
        self.order.unmix()

    def pointer_musts(self, brief: dict[str, Any]) -> list[str]:
        """The musts whose value is on another listing. A rule tied to another item of the order (race 14: "except
        when the other item's ...") is not one, even when the reading lists it: chasing it sends the model after a
        page that does not exist. Anything else the reading lists stays (shop names cannot be told apart by form)."""
        tied = [set(re.findall("[a-z0-9.]+", c["need"].partition(":")[2].lower())) for c in self.order.conditions]
        out = []
        for entry in _items(brief.get("pointers")) or _items(brief.get("lookups")):
            text = str(entry.get("must") or "") if isinstance(entry, dict) else str(entry)
            if not text.strip() and isinstance(entry, dict):
                # the reading left the must out: the item's must on the same property stands for it
                number, key = _count(entry.get("item")) or 0, _words(str(entry.get("property") or ""))
                pool = self.order.items[number - 1].musts if 1 <= number <= len(self.order.items) else [m for item in self.order.items for m in item.musts]
                text = next((m for m in pool if key and _words(m.split(":")[0]) == key), "")
                if text:
                    entry["must"] = text
            if not text.strip():
                continue
            pointing = re.search(r"\b(?:same|within|match\w*|unlike|alike|different|page|listing|lists|land on|picked|goes? with|pairs? with)\b", text, re.I)
            ruled = re.search(r"\b(?:except|unless|only if|only when|if the|when the|should the)\b", text, re.I)
            if ruled and not pointing or not pointing and any(t and t <= set(re.findall("[a-z0-9.]+", text.lower())) for t in tied):
                self.trace.append({"kind": "pointer_dropped", "must": text[:160]})
                continue
            out.append(text)
        return out

    def set_limits(self, brief: dict[str, Any]) -> None:
        """Budget, messages and steps allowed, the steps the request asks for along the way, and the messages kept for them."""
        budget = brief.get("budget")
        if isinstance(budget, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.order.budget = budget
        limit = _count(brief.get("messages"))
        if limit is not None and limit >= 0:
            self.talk.limit = limit
        elif self.toolbox.roles.get("ask"):
            self.talk.limit = 3
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.clock.max_steps:
            self.clock.max_steps = steps
        for entry in _items(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = str(entry.get("when") or "").strip().lower()
                self.order.process.append({"step": str(entry["step"]).strip(), "when": when if when in ("choose", "order", "end") else "order",
                                           "messages": max(0, _count(entry.get("messages")) or 0)})
        if self.toolbox.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.order.process if p["when"] in ("order", "end") and p["messages"]]
            # at least one message stays with the interviewer: race 15 had 12 Choji runs that sent the shopper nothing
            # (7 of them on tasks with a hidden need, 2 passed) against 2 for Kakashi
            room = self.talk.limit - 1 if self.talk.limit >= 2 else self.talk.limit
            self.talk.reserve = min(max(talk), room) if talk else 0


# ==============================================================================================================
# STEPS 1 AND 3: SEARCH AND OPEN LISTINGS (the model searches; Browser reads pages and attributes for it)
# ==============================================================================================================


class Browser:
    """Reads listings from the store for the model: further result pages, attributes of rows worth reading, wider
    searches, pages a pointer need leads through, fresh offers of cart lines, and several listings side by side.
    Every result passes through observe(), which files its listings and reads any note from the shopper in it."""
    def __init__(self, store: Store, toolbox: Toolbox, catalog: Catalog, order: Order, cart: Cart, talk: Talk, clock: Clock, models: Models, trace: list[dict[str, Any]]):
        self.store = store
        self.toolbox = toolbox
        self.catalog = catalog
        self.order = order
        self.cart = cart
        self.talk = talk
        self.clock = clock
        self.models = models
        self.trace = trace

    def observe(self, name: str, observation: Any) -> str:
        """Absorb a result of Kyo's own call and read any note in it; returns what the note changed."""
        self.catalog.absorb(observation)
        lines = self.talk.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.talk.take_note(lines)
        if meaning:
            self.talk.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that matched more listings than it showed, under a cheapest, most or least pick: Kyo reads a short
        rest itself (a step a page, no model turn); a long rest is named so the agent can narrow the search."""
        self.catalog.paged_obs = []
        pick = self.order.brief.get("pick")
        plan = self.toolbox.paging(name, observation) if pick in ("cheapest", "most", "least") else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            # the last page, or the agent is paging itself
            return ""
        if pick == "cheapest" and self.catalog.price_sorted(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.catalog.pages_read + 2 <= Settings.AUTO_PAGES and self.clock.left() - self.talk.tail() - 2 >= 12:
            pages = 2
        if pages > 2 or self.catalog.pages_read + pages > Settings.AUTO_PAGES or self.clock.left() - self.talk.tail() - pages < 12:
            return (f"\n[Kyo] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
                    f"search with narrower words (the requirement's value) or page on with {param}={_dump(value)}.")
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _shape({**args, param: value}, self.toolbox.schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.store.act(name, call)
            if reply is None or self.store.finished:
                break
            self.catalog.pages_read += 1
            seen = reply.get("observation")
            self.observe(name, seen)
            self.catalog.paged_obs.append(seen)
            shown_text = _show(seen)
            self.catalog.read_cache[name + _dump(call)] = (self.order.version, self.clock.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.toolbox.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        rest = (f"\n[Kyo] {count} listings matched and Kyo has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value "
                "or page on to see the rest.") if value is not None and texts else ""
        return ("\n[Kyo read the next pages of this search itself, one step a page]\n" + "\n".join(texts) + rest) if texts else ""

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.toolbox.facts_tool()
        target = self.toolbox.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.catalog.rows and i not in self.catalog.full and not self.catalog.sold_out(i)]:
            if self.clock.left() <= keep or self.store.finished or self.toolbox.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = _shape({target: ident}, self.toolbox.schemas.get(tool), "args", faults)
            if faults:
                break
            reply = self.store.act(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.catalog.note_view(tool, seen)
            if isinstance(seen, dict) and self.catalog.ident(seen):
                self.toolbox.facts_from[tool] = self.toolbox.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f"{ident}: " + _show(seen)[:1500])
        if texts:
            self.trace.append({"kind": "fill_facts", "n": len(texts)})
        return texts

    def cheaper_rows(self) -> list[str]:
        """Before a cheapest checkout: buyable rows of a cart line's own kind priced below it, whose attributes were
        never read, get read (a few, cheapest first) so the last check can weigh them against the cart's choice."""
        if self.order.brief.get("pick") != "cheapest":
            return []
        mark = self.cart.mark() + f"#{self.order.version}"
        cat_key = self.catalog.category_key()
        if mark in self.catalog.cheaper_marks or not cat_key:
            return []
        self.catalog.cheaper_marks.add(mark)
        cart = self.cart.ids()
        picks: list[str] = []
        for line in cart:
            price = self.catalog.price_of(line)
            kind = str((self.catalog.rows.get(line) or {}).get(cat_key) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.catalog.seen_order if i not in cart and i not in self.catalog.full and self.catalog.buyable(i)
                    and str((self.catalog.rows.get(i) or {}).get(cat_key) or "").strip().lower() == kind
                    and (self.catalog.price_of(i) if self.catalog.price_of(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.catalog.price_of(i) or 0.0)
            picks += [i for i in pool[:Settings.CHEAPER_OPENS] if i not in picks]
        # a few for each line: race 15 had a 3-line cheapest order bought with a $1313 laptop while a fitting $459 one sat
        # unread among the search rows, after three reads went to the other two lines
        return self.fill_facts(picks[:min(8, Settings.CHEAPER_OPENS * max(1, len(cart)))], self.talk.tail() + 4)

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.catalog.wider_held or self.order.brief.get("pick") != "cheapest" or self.clock.left() < 14 + 2 * len(self.order.items):
            return ""
        self.catalog.wider_held = True
        lines = [f"{i} at {self.catalog.price_of(i):g}" for i in self.cart.ids() if self.catalog.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.clock.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def widen(self) -> None:
        """Open more listings side by side (one step) so the key counts are not read off two or three listings."""
        tool = self.toolbox.roles.get("compare")
        schema = self.toolbox.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        # a side-by-side view without attributes shows the planner no keys: the attributes view one by one instead
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.toolbox.compare_bare():
            self.widen_singly()
            return
        if self.clock.left() <= 12:
            return
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.catalog.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.catalog.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _shape({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.catalog.full)
        reply = self.store.act(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.observe(tool, seen)
            self.catalog.note_view(tool, seen)
            rows = self.catalog.records_in(seen)
            if rows:
                self.toolbox.facts_from[tool] = self.toolbox.facts_from.get(tool, False) or any(_fact_block(r) for _, r in rows)
            self.trace.append({"kind": "widen", "ids": picks, "opened": len(self.catalog.full) - before})
            if rows and self.toolbox.facts_from.get(tool) is False:
                # the side-by-side view shows no attributes here: read them singly instead
                self.widen_singly()
                return
            if len(self.catalog.full) == before and len(picks) > 2 and self.clock.left() > 12:
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.store.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        tool = self.toolbox.facts_tool()
        target = self.toolbox.single_target(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.order.needs_now() for w in re.findall("[a-z0-9]{3,}", need.split(":", 1)[-1].lower())}
        each = WIDEN_EACH if len(self.order.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.catalog.item_groups():
            opened = sum((1 for i in ids if i in self.catalog.full))
            pool = [i for i in ids if i not in self.catalog.full and self.catalog.buyable(i)]
            if self.order.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.catalog.price_of(i) if self.catalog.price_of(i) is not None else float("inf"))
            pool.sort(key=lambda i: -len(wanted & set(re.findall("[a-z0-9]{3,}", _dump(self.catalog.rows.get(i) or {}).lower()))))
            lanes.append(pool[: max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[: max(0, 10 - len(read))], 30)

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.toolbox.roles.get("compare")
        tool = self.toolbox.facts_tool()
        target = self.toolbox.single_target(tool)
        if many and joiner and not self.toolbox.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.clock.left() < len(plan) + 5:
            return False
        before = len(self.catalog.full)
        for name, args in plan:
            faults: list[str] = []
            args = _shape(args, self.toolbox.schemas.get(name), "args", faults)
            if faults:
                return False
            reply = self.store.act(name, args)
            if reply is None or self.store.finished:
                break
            self.observe(name, reply.get("observation"))
        self.trace.append({"kind": "audit_open", "rows": rows, "opened": len(self.catalog.full) - before})
        return len(self.catalog.full) > before

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.toolbox.offer_tool()
        target = self.toolbox.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.clock.step - 2, self.cart.refused_at)
        stale = [i for i in self.cart.ids() if self.catalog.offer_seen.get(i, -99) < since and i not in self.catalog.gone]
        if not stale or self.clock.left() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _shape({target: ident}, self.toolbox.schemas.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.catalog.rows.get(ident) or {})
            reply = self.store.act(tool, args)
            if reply is None or self.store.finished:
                break
            observation = reply.get("observation")
            self.catalog.offer_seen[ident] = self.clock.step
            self.observe(tool, observation)
            # compare the stored record, so a re-read that arrives wrapped together with a note still shows its change
            after = self.catalog.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines)})
        if lines:
            self.order.version += 1
        return "\n".join(lines)

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        live = {need for _, need in self.order.needs_now()}
        pending = [need for _, need in self.order.refs if need in live and need not in self.order.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.toolbox.roles.get("offer") or self.toolbox.roles.get("open")
        target = self.toolbox.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.order.trails < (6 if self.clock.left() > 15 else 4) and self.models.cost < Settings.SPEND_HARD:
            self.order.trails += 1
            records = []
            for ident in reversed(self.catalog.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.catalog.full.get(ident) or self.catalog.rows.get(ident)) if ident in self.catalog.opened else None
                if record and len(records) < 45:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.catalog.seen_order):
                if len(records) >= 60:
                    break
                if ident not in self.catalog.opened or not (self.catalog.full.get(ident) or self.catalog.rows.get(ident)):
                    records.append("ROW " + _dump(_clip(self.catalog.rows.get(ident), 80))[:300])
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            got = self.models.ask_json(prompt, [Settings.LIGHT] + ([Settings.HEAVY] if self.models.cost + Settings.HEAVY_ROOM <= Settings.SPEND_HARD else []), 4500) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.catalog.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.order.ref_values[need] = str(chain["value"]).strip()
                    self.order.version += 1
                    continue
                # every link opened and nothing left to find: the last page holds the value
                if links and all(i in self.catalog.opened for i in ids) and not str(links[-1].get("find") or "").strip():
                    self.order.ref_values[need] = str(chain.get("value") or "").strip() or f"as stated on {ids[-1]}"
                    self.order.version += 1
                    continue
                closed = [i for i in ids if i in self.catalog.rows and i not in self.catalog.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.catalog.rows), {})
                    about = str(gap.get("about") or "") or "the listing it points at"
                    words = str(gap.get("find") or "").strip()
                    lost.append(f"{need} -> find {about}" + (f" (search: {words})" if words else ""))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.order.ref_values]
            if not reach and gaps and tool and target and self.clock.left() > 12 and self.order.link_searches < 3:
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get("kind") == "link_known":
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or not target or self.clock.left() < len(reach) + 4 or self.store.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _shape({target: ident}, self.toolbox.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.store.act(tool, args)
                if reply is None or self.store.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + _show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.order.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.order.ref_values.items()) + "\n"
        hold = bool(lost) and self.order.trail_blocks < (3 if self.clock.left() > 15 else 2) and self.clock.left() > 5
        if hold:
            self.order.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind "
                     "are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out.")
        return text, hold

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
        opener, joiner, target = self.toolbox.roles.get("open"), self.toolbox.roles.get("compare"), self.toolbox.single_target(self.toolbox.roles.get("open"))
        many = self.compare_target()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        # only into a side-by-side view that shows what the opens would (attributes)
        if self.toolbox.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_obj(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.catalog.rows and ident not in self.catalog.gone:
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

    def compare_target(self) -> str | None:
        schema = self.toolbox.schemas.get(self.toolbox.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work."""
        spec = ((self.toolbox.schemas.get(self.toolbox.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.toolbox.compare_bad:
            return max(2, min(self.toolbox.compare_bad - 1, max(self.toolbox.compare_ok, 2)))
        return max(4, self.toolbox.compare_ok)

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
        if name not in {self.toolbox.roles.get(r) for r in ("cart", "open", "offer", "compare")} - {None} or not self.toolbox.id_key:
            return args, ""
        props = (self.toolbox.schemas.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _kinds(v) or "array" in _kinds(v)) and "enum" not in v]
        # only the tool's one free argument, and only where it is known to take ids (the cart, or a call seen carrying one)
        if len(free) != 1 or (name != self.toolbox.roles.get("cart") and (name, free[0]) not in self.toolbox.id_args):
            return args, ""

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.catalog.rows:
                return value.strip(), ""
            if not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.catalog.rows.items() if any(k != self.toolbox.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.toolbox.id_key} field)"

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


# ==============================================================================================================
# STEP 2: ASK THE SHOPPER (and the steps along the way, and notes from the shopper)
# ==============================================================================================================

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.toolbox.roles.get("search")
        schema = self.toolbox.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and ("enum" not in v)]
        named = [k for k in texts if re.search("(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword", k, re.I)]
        for args in reversed(self.catalog.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search("[a-z]", args[k], re.I)]
            if not named:
                labels = {str(r.get(self.catalog.category_key()) or "").strip().lower() for r in self.catalog.rows.values()} - {""} if self.catalog.category_key() else set()
                fields = [
                    k
                    for k in fields
                    if len({str(a.get(k)) for a in self.catalog.searched if a.get(k)}) >= 2
                    and any((len(str(a.get(k)).split()) >= 2 for a in self.catalog.searched))
                    and (not any((str(a.get(k) or "").strip().lower() in labels for a in self.catalog.searched)))
                ]
            if fields and tool:
                field = fields[0]
                keep = {
                    k: v
                    for k, v in args.items()
                    if k != field and (isinstance((props.get(k) or {}).get("enum"), list) or k in (schema.get("required") or []))
                }
                cat_key = self.catalog.category_key()
                shelves = {_words(item.shelf or "") for item in self.order.items} | {_words(item.name) for item in self.order.items}
                shelves |= {_words(str(row.get(cat_key) or "")) for row in self.catalog.rows.values()} if cat_key else set()
                keep = {
                    k: v
                    for k, v in keep.items()
                    if k in (schema.get("required") or [])
                    or not isinstance(v, str)
                    or (
                        _words(v) not in shelves
                        and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s)))
                    )
                }
                worded = [a for a in self.catalog.searched if isinstance(a.get(field), str) and re.search("[a-z]", a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get("enum"), list):
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                return (tool, keep, field)
        return None

    def search_form(self) -> tuple[str, str, str | None, dict[str, Any]] | None:
        """The search tool's free-text field, its group field and its fixed arguments, read from the tool's own schema
        (before the model has searched, so no earlier search can show the form)."""
        tool = self.toolbox.roles.get("search")
        schema = self.toolbox.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        words = next((k for k in texts if re.search("q|query|search|text|phrase|keyword|term|word", k, re.I)), None)
        # the group field can be free text or a fixed list of the store's groups (some stores list their groups as fixed choices)
        strings = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v)]
        group = next((k for k in strings if k != words and re.search("group|categ|shelf|depart|aisle|section|kind", k, re.I)), None)
        if not tool or not words:
            return None
        fixed: dict[str, Any] = {}
        for k, v in props.items():
            options = v.get("enum") if isinstance(v, dict) else None
            if isinstance(options, list) and options and k != group:
                fixed[k] = next((o for o in options if isinstance(o, str) and re.search("word|text|keyword|search|query|phrase|match", o, re.I)), None)
                if fixed[k] is None:
                    if k in (schema.get("required") or []):
                        fixed[k] = options[0]
                    else:
                        fixed.pop(k)
        return tool, words, group, fixed

    def point_search(self, form: tuple[str, str, str | None, dict[str, Any]], words: str, group: str | None) -> list[tuple[str, dict[str, Any]]]:
        tool, field, gfield, fixed = form
        args = dict(fixed, **{field: words})
        options = ((self.toolbox.schemas.get(tool) or {}).get("properties") or {}).get(gfield or "", {}).get("enum") if gfield else None
        if gfield and group and isinstance(options, list):
            # a listed group: the option sharing the most words with the wanted kind, or none
            score = lambda o: len(_stem(str(o)) & _stem(group))
            best = max(options, key=score, default=None)
            if best is not None and score(best) > 0:
                args[gfield] = best
        elif gfield and group:
            args[gfield] = group
        faults: list[str] = []
        args = _shape(args, self.toolbox.schemas.get(tool), "args", faults)
        if faults or self.store.finished:
            return []
        reply = self.store.act(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get("observation"))
        if not self.toolbox.price_key and self.catalog.records_in(reply.get("observation")):
            # the result's parts (price, stock, paging) are learned from the first result Choji sees itself
            self.toolbox.read_shape(tool, reply.get("observation"))
        self.catalog.searched.append(args)
        return self.catalog.records_in(reply.get("observation"))

    def point_open(self, ident: str) -> dict[str, Any] | None:
        tool = self.toolbox.roles.get("open")
        target = self.toolbox.single_target(tool)
        if ident in self.catalog.opened and (self.catalog.full.get(ident) or self.catalog.rows.get(ident)):
            return self.catalog.full.get(ident) or self.catalog.rows.get(ident)
        if not tool or not target:
            return None
        faults: list[str] = []
        args = _shape({target: ident}, self.toolbox.schemas.get(tool), "args", faults)
        if faults:
            return None
        reply = self.store.act(tool, args)
        if reply is None or self.store.finished:
            return None
        observation = reply.get("observation")
        self.observe(tool, observation)
        # the page counts as shown, so a later open of it by the model is answered from the record without a step
        self.catalog.note_view(tool, observation)
        found = self.catalog.records_in(observation)
        return found[0][1] if found else (observation if isinstance(observation, dict) else None)

    def resolve_pointers(self) -> str:
        """Right after the reading of the request: each requirement whose value is on another listing is followed to
        the listing that states it, so the value can be searched with and asked around like any stated one. A page's
        note can send the reader on to a further listing; same-shop look-alikes that fit only part of a description are
        passed over, and when two listings fit, each one's value is kept. The value becomes the item's requirement by
        the request's relation: the same value, the paired value ("goes with"), a range ("within"), or an exclusion
        ("unlike"). What is not resolved within the budget stays a pointer for the model and the checkout's trail."""
        entries = [e for e in _items(self.order.brief.get("pointers")) if isinstance(e, dict) and str(e.get("must") or "").strip()]
        for e in entries:
            # the reading's must may have been reworded since (unmix): the live pointer on the same property stands for it
            key = _words(str(e["must"]).split(":")[0])
            e["must"] = next((need for _, need in self.order.refs if need == str(e["must"]).strip()), None) or \
                next((need for _, need in self.order.refs if _words(need.split(":")[0]) == key), "")
            prop = re.escape(str(e.get("property") or str(e["must"]).split(":")[0]).strip())
            told = re.search(prop + "[^.?!]{0,40}\\bgoes? with\\b", self.order.request, re.I) if prop else None
            if (told or re.search("\\bgoes with|\\bgo with|\\bpairs? with", e["must"], re.I)) and not str(e.get("relation") or "").lower().startswith("within"):
                e["relation"] = "goes with"
        entries = [e for e in entries if e["must"] and (str(e.get("shop") or "").strip() or e.get("with"))]
        spare = max(0, self.clock.max_steps - 24)
        budget = min(20 * len(entries), max(spare * 2 // 3, min(spare, 10)))
        return self.resolve_entries(entries, budget, 22, "[Choji followed the listings that requirements point at before searching; these are now plain requirements, searched and checked like any other] ")

    def resolve_answered(self, pointed: list[tuple[int, str]]) -> str:
        """Right after the answers: a need the shopper gave by pointing at another listing is read into the same form as
        a request's pointer and followed the same way, so its value is searched before the model shops on."""
        entries = []
        for number, need in pointed:
            got = self.models.ask_json(POINT_PARSE + "\n\nREQUIREMENT:\n" + need, [Settings.LIGHT, Settings.LIGHT], 1500) or {}
            if not (str(got.get("shop") or "").strip() or isinstance(got.get("with"), dict) and got["with"]):
                continue
            entry = dict(got, must=need, item=number)
            # the need's own label names the property (the reader can take a verb of the answer into it)
            if ":" in need and need.split(":", 1)[0].strip():
                entry["property"] = need.split(":", 1)[0].strip()
            if re.search("\\bgoes? with\\b|\\bpairs? with\\b", need, re.I) and not str(entry.get("relation") or "").lower().startswith("within"):
                entry["relation"] = "goes with"
            entries.append(entry)
        budget = min(16 * len(entries), max(0, self.clock.left() - 16))
        return self.resolve_entries(entries, budget, 16, "[Choji followed the listings the shopper's answers point at; these are now plain requirements] ")

    def resolve_entries(self, entries: list[dict[str, Any]], budget: int, floor: int, head: str) -> str:
        """Each entry's chain is walked to the listing that states the value (see resolve_pointers); `floor` is the
        number of steps that must stay free for the shopping that follows."""
        form = self.search_form()
        if not form or not entries or not self.toolbox.roles.get("open") or budget <= 0:
            return ""
        start = self.clock.step
        lines: list[str] = []
        settled: list[tuple[int, str]] = []
        for entry in entries:
            must = str(entry["must"]).strip()
            prop = str(entry.get("property") or must.split(":")[0]).strip()
            relation = str(entry.get("relation") or "same").strip().lower()
            # a shop named in the possessive ("Belramoan's") is searched by its name alone
            want = {"shop": re.sub(r"['\u2019]s$", "", str(entry.get("shop") or "").strip()) or None, "kind": str(entry.get("kind") or "").strip(), "with": entry.get("with") if isinstance(entry.get("with"), dict) else {}}
            values: list[str] = []
            pairs: list[str] = []
            chain: list[str] = []
            for _ in range(7):
                if self.clock.step - start >= budget or self.clock.left() < floor or self.models.cost > Settings.SPEND_SOFT or self.store.finished:
                    break
                described = [str(v) for v in want["with"].values()]
                shop = [want["shop"]] if want["shop"] else []
                group = want["kind"] or None
                # narrow to broad: a store search can need every word, and a described value is often worded differently
                # on the listing, so the shop's name alone (inside the kind, then anywhere) is the last try
                plans = [(" ".join(shop + described), group)] + [(" ".join(shop + [v]), group) for v in described[:3]]
                if shop:
                    plans += [(shop[0], group), (shop[0], None)]
                else:
                    plans += [(" ".join(described[:1]), None)]
                plans = [(w, g) for w, g in dict.fromkeys(plans) if w.strip()]
                rows: list[tuple[str, dict[str, Any]]] = []
                for words, group in plans:
                    rows = self.point_search(form, words, group)
                    shop_rows = [(i, r) for i, r in rows if not want["shop"] or want["shop"].lower() in _dump(r).lower()]
                    if shop_rows:
                        rows = shop_rows
                        break
                    rows = [] if want["shop"] else rows
                    if self.clock.step - start >= budget:
                        break
                if not rows:
                    break
                # the store's own order: its rows show titles only, so their words say little about the attributes
                readings: list[dict[str, Any]] = []
                looked: set[str] = set()
                for batch in (rows[:1], rows[1:3], rows[3:6]):
                    records = []
                    for ident, _row in batch:
                        if self.clock.step - start >= budget and ident not in self.catalog.opened:
                            break
                        record = self.point_open(ident)
                        if record:
                            records.append((ident, record))
                            looked.add(ident)
                    if not records:
                        break
                    prompt = (POINT_READ + f"\n\nPROPERTY: {prop}\nLOOKING FOR: " + _dump(want)
                              + "\n\nRECORDS:\n" + "\n".join(f"{i}: " + _dump(_clip(r, 400))[:2200] for i, r in records))
                    # an empty or broken reply is not "no match": the light model once more, then the heavy one
                    room = [Settings.HEAVY] if self.models.cost + Settings.HEAVY_ROOM <= Settings.SPEND_HARD else []
                    got = self.models.ask_json(prompt, [Settings.LIGHT, Settings.LIGHT] + room, 4000)
                    if got is None:
                        self.trace.append({"kind": "pointer_read_failed", "must": must[:80]})
                        break
                    readings = [r for r in _items(got.get("readings")) if isinstance(r, dict) and r.get("match") is True]
                    self.trace.append({"kind": "pointer_hop", "must": must[:80], "want": want, "read": _clip(readings, 60)})
                    # a page that fits but gives neither the value nor a next listing is a look-alike: try the shop's others
                    readings = [r for r in readings if r.get("value") or r.get("pair") or isinstance(r.get("next"), dict)]
                    if readings:
                        break
                if not readings:
                    break
                chain += [str(r.get("id")) for r in readings]
                values += [str(r["value"]).strip() for r in readings if r.get("value") and not r.get("next")]
                pairs += [str(r["pair"]).strip() for r in readings if r.get("pair") and not r.get("next")]
                onward = next((r["next"] for r in readings if isinstance(r.get("next"), dict)), None)
                if not onward:
                    break
                kind = str(onward.get("kind") or "").strip()
                shop = re.sub(r"['\u2019]s$", "", str(onward.get("shop") or "").strip()) or None
                if not shop and kind:
                    # the reader left the seller out: the proper name the note puts right before the kind of product
                    note = str(onward.get("said") or "") or " ".join(_dump(r) for i, r in records if i in {str(x.get("id")) for x in readings})
                    for word in kind.split()[:2]:
                        named = re.search("\\b([A-Z][a-z]{3,})(?:'s)?\\s+(?:[a-z]+\\s+)?" + re.escape(word), note, re.I)
                        if named and named.group(1)[0].isupper() and named.group(1).lower() not in {"this", "that", "from", "look", "refer", "check", "another", "each", "take"}:
                            shop = named.group(1)
                            break
                want = {"shop": shop, "kind": kind, "with": onward.get("with") if isinstance(onward.get("with"), dict) else {}}
                if str(onward.get("said") or "").strip():
                    want["said"] = str(onward["said"]).strip()[:300]
            found = pairs if relation.startswith("goes") and pairs else values
            if not found:
                lines.append(f"{must}: not resolved ({len(chain)} pages followed)")
                continue
            need = self.pointed_need(prop, relation, must, list(dict.fromkeys(found)))
            if not need:
                continue
            for number, item in enumerate(self.order.items, 1):
                if must in item.musts or must in item.told:
                    item.musts = [need if m == must else m for m in item.musts]
                    item.told = [need if m == must else m for m in item.told]
                    self.order.refs = [(n, m) for n, m in self.order.refs if m != must]
                    self.order.ref_values[need] = ", ".join(found)
                    self.order.notes.append(f"item {number}: {need} (read from {' -> '.join(chain[-3:])})")
                    lines.append(f"item {number} ({item.name}): {must} -> {need}")
                    if not relation.startswith(("within", "unlike", "different", "not")):
                        settled.append((number, found[0]))
            self.trace.append({"kind": "pointer_done", "must": must[:80], "need": need, "chain": chain})
        if not lines:
            return ""
        # the value is searched at once, so the preparation starts from listings that carry it
        for number, value in settled:
            if self.clock.left() < floor or self.store.finished:
                break
            item = self.order.items[number - 1]
            stated = [v for v in self.value_words(number) if _words(v) != _words(value)][:1]
            rows = self.point_search(form, " ".join([value] + stated), item.shelf or item.name) or self.point_search(form, value, item.shelf or item.name)
            opened = [i for i, _ in rows[:2] if self.point_open(i)]
            if opened:
                lines.append(f"searched item {number} with {' '.join([value] + stated)} and opened {', '.join(opened)}")
        return head + " | ".join(lines)

    @staticmethod
    def pointed_need(prop: str, relation: str, must: str, found: list[str]) -> str | None:
        """The requirement a resolved pointer becomes, by the request's relation."""
        if relation.startswith("within"):
            tol = re.search("(?:within|inside|up to)\\s*([0-9.]+)\\s*([a-z\"]*)", must, re.I)
            got = re.search("([0-9.]+)\\s*([a-z\"]*)", found[0], re.I)
            if not tol or not got:
                return None
            width, value = float(tol.group(1)), float(got.group(1))
            unit, have = tol.group(2).lower(), got.group(2).lower()
            if unit.startswith(("in", "\"")) and have.startswith("cm"):
                value /= 2.54
            elif unit.startswith("cm") and have.startswith(("in", "\"")):
                value *= 2.54
            return f"{prop}: between {max(0.0, value - width):.2f} and {value + width:.2f} {unit or have}".strip()
        if relation.startswith(("unlike", "different", "not")):
            return f"{prop}: not " + " and not ".join(found)
        return f"{prop}: " + " or ".join(found)

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.order.needs_now():
            if n != number or need in [m for _, m in self.order.refs] or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            if not value or re.search(
                "\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b",
                value,
                re.I,
            ):
                continue
            out.append(value)
        return out

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.clock.left() < 22:
            return ""
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.order.items[number - 1]
            fresh = [v for v in values if any((":" in t and t.split(":", 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            # every value together, then without the last stated one (race 14: a category word sent without the group filter
            # pushed the fitting rows off the page, 4 of 44 searches showed an accepted listing against 14 of 41 this way)
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1 :]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: " ".join(words)}, self.toolbox.schemas.get(tool), "args", faults)
                if faults or self.store.finished:
                    break
                reply = self.store.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get("observation")
                self.observe(tool, observation)
                cat_key = self.catalog.category_key()
                found = [i for i, _ in self.catalog.records_in(observation)]
                cap = self.order.budget if isinstance(self.order.budget, (int, float)) else None
                prices = [self.catalog.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [
                    i
                    for i in found
                    if self.catalog.buyable(i)
                    and (not cat_key or self.order.owns(item, str((self.catalog.rows.get(i) or {}).get(cat_key) or "")))
                    and (cap is None or (self.catalog.price_of(i) or 0) <= cap)
                ]
                seen.append(f"{_dump(args.get(field))}: " + (", ".join(rows[:6]) or "no row of that kind"))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.catalog.full][: min(3, budget)], self.talk.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append("opened: " + " | ".join(opened)[:2400])
                    break
        if not seen:
            return ""
        self.trace.append({"kind": "value_search", "seen": [s[:200] for s in seen]})
        return "[Choji searched each such item's requirement values together and read the first fitting-kind rows] " + "\n".join(seen)

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {
            w
            for ident in self.catalog.shop_rows(about, list(self.catalog.rows)) + self.catalog.opened_shop(about)
            for k, v in (self.catalog.rows.get(ident) or {}).items()
            if isinstance(v, str) and len(v.strip()) <= 40 and re.search("seller|shop|store|vendor|brand|maker|merchant", str(k), re.I)
            for w in re.findall("[a-z0-9]{3,}", v.lower())
        }
        values = (
            {w for w in re.findall("[a-z0-9]+", (words or about).lower()) if len(w) >= 3 or w.isdigit()}
            - named
            - {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for", "of"}
        )
        for ident in self.catalog.opened_shop(about):
            stack, leaves = ([self.catalog.full.get(ident) or self.catalog.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = " ".join(leaves).lower()
            if values and all((re.search("(?<![a-z0-9])" + re.escape(v) + "(?![a-z0-9])", text) for v in values)):
                self.trace.append({"kind": "link_known", "id": ident})
                return []
        plan = self.search_template()
        memo = " ".join(words.lower().split())
        if memo and memo in self.catalog.link_seen:
            self.order.link_searches += 1
            known = self.catalog.shop_rows(about, self.catalog.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _shape({**base, field: words}, self.toolbox.schemas.get(tool), "args", faults)
            if not faults:
                self.order.link_searches += 1
                reply = self.store.act(tool, args)
                if reply is not None:
                    observation = reply.get("observation")
                    self.observe(tool, observation)
                    self.catalog.link_seen[memo] = [i for i, _ in self.catalog.records_in(observation)]
                    known = self.catalog.shop_rows(about, self.catalog.link_seen[memo])
                    self.trace.append({"kind": "link_search", "words": words[:80], "rows": known[:4]})
        kind = (
            {w for w in re.findall("[a-z0-9]{3,}", (words or about).lower())}
            - named
            - {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for"}
        )

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall("[a-z0-9]{3,}", _dump(self.catalog.rows.get(i) or {}).lower()))]

        return (of_kind(known) or of_kind(self.catalog.shop_rows(about, list(reversed(self.catalog.seen_order)))))[:2]


class Talk:
    """Step 2 and the steps along the way: talking with the shopper. Decides when to ask, plans and sends the
    questions, reads the answers into new needs, carries out the steps the request asks for before the order or the
    ending (an approval, a message), and reads the shopper's notes that turn up inside store results. Keeps count of
    the messages allowed, sent and kept back."""
    def __init__(self, order: Order, catalog: Catalog, cart: Cart, store: Store, models: Models, toolbox: Toolbox, clock: Clock, trace: list[dict[str, Any]]):
        self.order = order
        self.catalog = catalog
        self.cart = cart
        self.store = store
        self.models = models
        self.toolbox = toolbox
        self.clock = clock
        self.trace = trace
        self.browser: Browser = None  # type: ignore[assignment]  # set once the Browser exists: asking widens the search first
        self.limit = 0
        self.used = 0
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.list_size = Settings.LIST_KEYS   # drops to 1 once the shopper wants single questions
        self.retried = False                  # an interview that sent nothing gets one later try
        self.qa: list[tuple[str, str]] = []
        self.heard: set[str] = set()
        self.queued: list[str] = []
        self.plan_order: list[int] = []

    def free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.limit - self.used - max(0, self.reserve - self.gate_msgs)

    # ---------- notes inside tool results ----------
    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        # listings sent as lines of text are records, not notes (race 15: every page and search row spent a model call here)
        prose = lambda text: len(text.split()) >= 5 and not _flat_record(text)
        if isinstance(observation, dict) and not self.catalog.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and prose(value):
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if prose(v))
                elif isinstance(value, dict) and not self.catalog.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.order.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.models.ask_json(prompt, [Settings.LIGHT, Settings.SPARE], 2500) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.catalog.unavailable.update(g for g in gone if g in self.catalog.rows)
        self.cart.note_sold = bool(gone) or got.get("sold_out") is True
        self.cart.note_named = any(g in self.catalog.rows for g in gone)
        if self.cart.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.cart.sold_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.order.budget:
            self.order.budget = budget
            changed.append(f"budget now {budget}")
        added: set[str] = set()
        drops = [e for e in _items(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _items(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            item = self.order.items[min(max(1, _count(entry.get("item")) or 1), len(self.order.items)) - 1]
            target = str(entry["need"]).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(":")[0]) == _words(target.split(":")[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        for entry in adds:
            item = self.order.items[min(max(1, _count(entry.get("item")) or 1), len(self.order.items)) - 1]
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
        if self.order.brief.get("pick") == "cheapest" and (drops or gone or self.cart.note_sold or any(c.startswith("budget now") for c in changed)):
            self.catalog.wider_held = False
            self.catalog.cheaper_marks = set()
        if changed or summary:
            self.order.version += 1
            self.store.recent = []
            self.order.notes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

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

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.order.process) and (stage, self.cart.mark() + f"#{self.order.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, and one more while a step due at the order needs a message."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.order.process)
        return 2 if talk and self.used < self.limit and self.gate_due("order") else 1

    def ready_to_ask(self) -> bool:
        if self.asked or not self.toolbox.roles.get("ask") or self.free() <= 0:
            return False
        if self.clock.left() <= 8 or self.clock.seconds() > Settings.INTERVIEW_UNTIL:
            return False
        opened = len(self.catalog.full)
        if opened >= 3 * len(self.order.items) + 1:
            return True
        return self.clock.step >= max(8, self.clock.max_steps // 4)

    def interview(self) -> str:
        self.asked = True
        if self.free() <= 0:
            return ""
        self.browser.widen()
        lanes = self.plan_lanes(self.free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.free() > 0 and (not self.store.finished) and (self.clock.left() > 6) and self.clock.seconds() < Settings.QUESTIONS_UNTIL:
            if follow:
                number, labels, shown = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used, self.list_size)
                shown = False
                if not labels:
                    break
            if len(labels) > 1:
                gave = self.ask_list(number, labels, used, found, spent, sent, told)
                if gave is None:
                    break
                continue
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], self.ask_text(number, labels, shown))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.order.items]
            heard = self.hear(batch, (Settings.LIGHT, Settings.HEAVY))
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.order.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_count(x) for x in _items(heard.get("crowded"))]:
                continue
            # a "one at a time" answer to a one-property question: the property's short name may still land, but the same
            # question with sample values mostly draws the same complaint again (races 11-13: 5% needs, 72% repeats),
            # so that message goes to the next property instead
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], False))
        # needs the shopper gave by pointing at another listing are followed now, while steps remain to shop with them
        fresh_refs = [(n, need) for n, need in self.order.refs if any(need in t for t in told)]
        resolved = self.browser.resolve_answered(fresh_refs) if fresh_refs else ""
        found_items = [n for n in found if found[n] > 0]
        searched = self.browser.value_search(found_items) if found_items else ""
        if told:
            self.order.version += 1
        if not sent and not self.retried and self.free() > 0:
            self.retried = True
            self.asked = False
            self.trace.append({"kind": "interview_empty", "at": self.clock.step})
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.order.refs if any((need in t for t in told))]
        if pointed:
            summary += (
                " These take their value from another listing: "
                + " | ".join(pointed)
                + ". Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened."
            )
        if resolved:
            summary += "\n" + resolved
        if searched:
            summary += "\n" + searched
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def question(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"What {label} would you like for the {self.order.items[number - 1].name}"
        return text + (f", for example {samples[0]} or {samples[1]}?" if samples else "?")

    def samples(self, number: int, label: str) -> list[str]:
        """Two values the listings show for this property: they say which property a question means."""
        want = _words(label)
        settled = " ".join(self.order.items[number - 1].musts).lower()
        for _, name, _, values in self.catalog.key_table():
            if name == want:
                return [v for v in values if v and v not in settled and len(v) <= 30][:2]
        return []

    def send(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.toolbox.roles["ask"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.store.finished or self.clock.left() <= floor or self.used >= self.limit:
                break
            args = self.toolbox.message_args(text)
            if not args:
                break
            reply = self.store.act(tool, args)
            if reply is None:
                break
            self.used += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.clock.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart.mark()})")
            self.done_steps.append(self.clock.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        """Add what the answers stated to the ledger; `told` collects the new requirements for the summary."""
        many = len(self.order.items) > 1
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.order.items))
            need = str(entry["need"]).strip()
            if need not in self.order.items[number - 1].told:
                self.order.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.order.refs:
                self.order.refs.append((number, need))
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.order.items[min(max(1, _count(entry.get("item")) or 1), len(self.order.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.order.items))
                self.order.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0:
            self.order.budget = heard["budget"]

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (Settings.HEAVY, Settings.LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.order.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.models.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.order.process if p["when"] == stage]
        if not steps or self.store.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.toolbox.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(Settings.GATE_ROUNDS):
            prompt = (GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.order.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.order.ledger() + "\n\nCART NOW: " + self.cart.text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.order.notes[-4:]) if self.order.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.limit - self.used)}   STEPS LEFT: {self.clock.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.models.ask_json(prompt, [Settings.HEAVY, Settings.LIGHT] if nudged else [Settings.LIGHT, Settings.HEAVY], 3500)
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
                self.order.version += 1
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
                    "The shopper's reply changes the task: " + "; ".join(told + ([hold] if hold else [])) + ". Fix the cart for it, then check out again.")
            name = str((call or {}).get("tool") or "")
            # a step called done must point at the record line that did it; at the order, a step that takes a
            # message is about the final cart, so its line must come after the cart last changed
            shown = [e for e in _items(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(p["messages"] > 0 for p in steps)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart.at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_count(n) or 0) in proof for n in _items(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.toolbox.roles.get("ask") and self.used < self.limit and self.clock.left() >= 2:
                nudged = True
                self.trace.append({"kind": "gate_unproven", "steps": _clip(unproven, 120)})
                self.done_log.append(f"(Kyo: no line made since the cart last changed at step {self.cart.at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
                                     + f" done, and {self.limit - self.used} message(s) and {self.clock.left()} steps are left: give the call that does it)")
                self.done_steps.append(-1)
                continue
            if not call or name not in self.toolbox.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart.mark() + f"#{self.order.version}"))
                break
            if self.clock.left() < 2 or (name == self.toolbox.roles.get("ask") and self.used >= self.limit):
                break
            faults: list[str] = []
            args = _shape(call.get("args") if isinstance(call.get("args"), dict) else {}, self.toolbox.schemas[name], "args", faults)
            if faults:
                break
            if name == self.toolbox.roles.get("ask"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.send([(1, "", text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.store.act(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.catalog.absorb(seen)
            self.done_log.append(f"step {self.clock.step}: {name} {_dump(args)} -> {_show(seen)[:300]} (cart then: {self.cart.mark()})")
            self.done_steps.append(self.clock.step)
            said.append(f"{name} {_dump(args)} -> {_show(seen)[:400]}")
        return True, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)) if said else ""


# ==============================================================================================================
# STEP 4: CHOOSE, CHECK AND FINISH
# ==============================================================================================================

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.catalog.key_table()
        settled = []
        for number, item in enumerate(self.order.items, 1):
            settled.append(f"item {number}: {item.name} - settled by the request: " + ("; ".join(item.musts) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(
                f"  {index}. {name} (on {carried}): "
                + " | ".join(values)
                + self.not_stated(label, name, values)
                + (" [several features in one: ask last]" if re.search(CATCH_ALL, name) else "")
            )
        prompt = PLAN.replace("COUNT", str(max(1, count))) + "\n\nREQUEST:\n" + self.order.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.order.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.models.ask_json(prompt, [Settings.HEAVY, Settings.LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.order.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.order.items))
            label = re.split("\\s+(?:and|or)\\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get("hint") or "").strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            read = max(1, self.catalog.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.order.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub("our\\b", "or", re.sub("(?<=[a-z]{3})s\\b", "", _words(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _words(label) in own:
                    return own[_words(label)]
                near = [s for k, s in own.items() if _names(_stem(label), _stem(k)) or _names(_stem(k), _stem(label))]
                return max(near) if near else 0.0

            hinted[number] = {_words(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((_words(spelled(pick)) != _words(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (not re.search(CATCH_ALL, k))][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
                    continue
                weak = next(
                    (i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)),
                    None,
                )
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _words(other) != _words(key)]
                if weak is not None and len(self.order.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((_words(x) in hinted[number] and all((_words(x) != _words(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((_words(key) != _words(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCH_ALL, _words(x))] + [
                x for x in lanes[number] if re.search(CATCH_ALL, _words(x))
            ]
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "lanes": {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.order.items[number - 1]
        settled = [_stem(m.split(":", 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.order.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match("^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$", v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.order.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if not re.search(CATCH_ALL, key)] + [key for key in keys if re.search(CATCH_ALL, key)]

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.order.owner_of(label)
        if not number:
            return ""
        for must in self.order.items[number - 1].musts:
            prop, _, value = must.partition(":")
            if re.search("\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b", value, re.I):
                continue
            words = set(re.findall("[a-z0-9]{3,}", value.lower()))
            shown = set(re.findall("[a-z0-9]{3,}", " ".join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f" [not the stated {prop.strip()}: another property]"
        return ""

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]], size: int = 1) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        # race 14: about half the items of an order hold an unwritten need; items with a pointer or a rule tied to them
        # held one more often (57% vs 46%), items with fewer stated requirements too, and the device the other items
        # must fit least often (35%); one question per item comes before a second one for any item
        devices = {main for _, main in self.order.bound}
        tied = {n for n, _ in self.order.refs} | {c["item"] for c in self.order.conditions} | {c["other_item"] for c in self.order.conditions}
        tied |= {n for n, item in enumerate(self.order.items, 1) if any(v for v in self.order.ref_values if v in item.musts)}
        def likely(n: int) -> tuple[int, int, int]:
            return (n in devices, n not in tied, len(self.order.items[n - 1].musts) if n <= len(self.order.items) else 9)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n) if n in first else len(first), likely(n), n)):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if size > 1:
                # one spelling per property in a list ("sleeve" after "sleeves"); of a name and a longer one ending in it
                # ("length", "pants length") the longer stays, in the shorter one's place; "resolution standard" is
                # another property than "resolution"
                def same(x: str, y: str) -> bool:
                    fold = lambda w: re.sub("our$", "or", w.rstrip("s"))
                    a, b = sorted((_words(x).split(), _words(y).split()), key=len)
                    return [fold(w) for w in a] == [fold(w) for w in b[len(b) - len(a):]]
                kept: list[str] = []
                for label in left:
                    if any(same(label, u) for u in used[number]):
                        continue
                    twin = next((i for i, k in enumerate(kept) if same(label, k)), None)
                    if twin is None:
                        kept.append(label)
                    elif len(label.split()) > len(kept[twin].split()):
                        kept[twin] = label
                left = kept
            if left:
                return (number, left[:size])
        return (0, [])

    def ask_list(self, number: int, labels: list[str], used: dict[int, set[str]], found: dict[int, int], spent: dict[int, int],
                 sent: list[tuple[int, str, str]], told: list[str]) -> bool | None:
        """One message naming several properties of one item. What its reply settles: a calm reply all of them (the
        shopper reads the whole list); a new need the properties up to the one it names, the rest are asked again; a
        puzzled reply none, and the next lists are shorter; a reply that wants single questions ends lists for this
        episode. None when no message went out."""
        name = self.order.items[number - 1].name
        text = f"For the {name}, do you need a particular {', '.join(labels[:-1])} or {labels[-1]}?"
        if not self.toolbox.message_args(text):
            return None
        batch = self.send([(number, labels[0], text)])
        if not batch:
            return None
        sent += batch
        spent[number] += 1
        counts = [len(item.told) for item in self.order.items]
        heard = self.hear(batch, (Settings.LIGHT, Settings.HEAVY))
        heard.pop("free", None)   # "does not mind" can not be pinned to one property of a list
        self.take_heard(heard, told)
        new = []
        for n, item in enumerate(self.order.items, 1):
            if len(item.told) > counts[n - 1]:
                new += item.told[counts[n - 1]:]
                found[n] = found.get(n, 0) + 1
        marked = lambda field: [_count(x) for x in _items(heard.get(field))]
        if new:
            # the last listed property a new need names; the ones after it were not answered
            hit = -1
            for need in new:
                prop = _stem(need.split(":", 1)[0])
                hit = max([hit] + [i for i, label in enumerate(labels) if _names(prop, _stem(label)) or _names(_stem(label), prop)])
            settled = labels if hit < 0 else labels[:hit + 1]
        elif 1 in marked("crowded"):
            self.list_size = 1
            settled = []
        elif 1 in marked("unclear"):
            self.list_size = min(self.list_size, Settings.LIST_KEYS_AFTER_UNCLEAR)
            settled = []
        else:
            settled = labels
        used[number].update(_words(label) for label in settled)
        self.trace.append({"kind": "list_reply", "item": number, "asked": labels, "settled": settled, "new": new, "size_next": self.list_size})
        return True

    def ask_text(self, number: int, labels: list[str], shown: bool = False) -> str:
        if len(labels) >= 2:
            return f"What {labels[0]} or {labels[1]} would you like for the {self.order.items[number - 1].name}?"
        return self.question(number, labels[0], self.samples(number, labels[0]) if shown else None)


class Checkout:
    """Step 4: choosing and finishing. Checks the cart against the needs with the heavy model before an order,
    replaces sold-out lines, audits the evidence before an ending without a purchase, completes a cart the model left
    unfinished, words a quiet decline, and makes the last checkout or ending."""
    def __init__(self, cart: Cart, catalog: Catalog, order: Order, toolbox: Toolbox, talk: Talk, browser: Browser, store: Store, clock: Clock, models: Models, trace: list[dict[str, Any]]):
        self.cart = cart
        self.catalog = catalog
        self.order = order
        self.toolbox = toolbox
        self.talk = talk
        self.browser = browser
        self.store = store
        self.clock = clock
        self.models = models
        self.trace = trace
        self.fit_seen: dict[str, dict[str, Any]] = {}   # judge answers per device and part lines, so one cart is judged once

    def check_cart(self, ids: list[str] | None = None, limit: float = Settings.SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart.ids()
        key = self.order.check_key(ids)
        if key in self.cart.checked or not self.models.judges(limit) or not ids:
            return ""
        self.cart.checked.add(key)
        prompt = CHECK + "\n\nREQUEST:\n" + self.order.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.order.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.order.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart.text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.catalog.sold_out(i) else "") + self.catalog.record_of(i) for i in ids)
        clash = self.key_clashes(ids)
        if clash:
            prompt += "\n\nATTRIBUTES THAT NAME A REQUIREMENT'S PROPERTY BUT GIVE ANOTHER VALUE (such an attribute decides, so the line fails that requirement unless its value is a plain synonym):\n" + "\n".join(clash)
        if self.order.brief.get("pick") == "cheapest":
            prompt += "\n\n" + self.catalog.cheaper_than(ids)
        elif self.order.brief.get("pick") in ("most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.catalog.others(ids, 15)
        judges = self.models.judges(limit)
        got = None
        ranked = self.order.brief.get("pick") in ("cheapest", "most", "least")
        if Settings.HEAVY in judges and not ranked:
            # the light judge first: on the race-14 cart bench it never passed a wrong cart, only raised false alarms
            # (5 of 18), so its pass stands and only its objection goes to the heavy judge, which decides
            got = self.models.ask_json(prompt, [Settings.LIGHT], 3000)
            self.trace.append({"kind": "check_light", "cart": ids, "ok": None if got is None else got.get("ok") is not False})
            judges = [Settings.HEAVY] if got is None or got.get("ok") is False else []
        if judges:
            got = self.models.ask_json(prompt, judges, 3000) or got
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.cart.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.catalog.leads = [i for i in self.catalog.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.catalog.rows and not self.catalog.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def key_clashes(self, ids: list[str]) -> list[str]:
        """Cart lines whose attribute key is the very name of a plain requirement's property while its value does not
        hold the required value. Race 15: a scrub with "ingredient: vitamin e" (glutathione only under another key) was
        bought for a glutathione requirement, and the heavy check passed it in 1 of 2 tries on the bench."""
        out = []
        for number, item in enumerate(self.order.items, 1):
            mine = set(self.catalog.item_ids(number))
            for need in item.musts:
                prop, _, value = need.partition(":")
                value = value.strip().lower()
                # plain values only: limits, ranges, exclusions, lists of choices and pointers are left to the judge
                if not value or re.search(r"\b(?:same|like|unlike|than|or|not|no|within|between|up to|above|below|over|under|at least|at most)\b|\d", value):
                    continue
                words = set(re.findall(r"[a-z]{3,}", value))
                for ident in ids:
                    if ident not in mine:
                        continue
                    block = _fact_block(self.catalog.full.get(ident) or {}) or {}
                    for key, got in block.items():
                        if _words(key) == _words(prop) and words and not words <= set(re.findall(r"[a-z]{3,}", str(got).lower())):
                            out.append(f"{ident}: {key} = {got!r}, while the requirement is {need!r}")
        return out[:6]

    def swap_hint(self) -> str:
        """A cart line whose newest record says it is sold out: name replacements among listings already read, once per line."""
        lost = [i for i in self.cart.ids() if i not in self.catalog.gone and self.catalog.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.catalog.gone[ident] = self.clock.step
        self.catalog.leads = [i for i in self.catalog.leads if i not in lost]
        self.order.version += 1
        picks = self.replacements(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.clock.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.order.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        """A refusal said a chosen listing sold out without naming it: a lone cart line is that listing; several lines
        are re-read with the offer view so their records show which one. Returns what the re-read showed."""
        ids = [i for i in self.cart.ids() if i not in self.catalog.gone]
        if any(self.catalog.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.catalog.unavailable.add(ids[0])
            return ""
        return self.browser.reread_offers() if ids else ""

    def replacements(self, lost: list[str]) -> list[str]:
        if not self.models.judges():
            return []
        records = []
        for ident in reversed(self.catalog.seen_order):
            if ident in lost or ident in self.catalog.gone or self.catalog.sold_out(ident):
                continue
            records.append(_dump(_clip(self.catalog.full.get(ident) or self.catalog.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.order.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.order.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.catalog.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart.text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.models.ask_json(prompt, self.models.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.catalog.rows and ident not in lost and ident not in self.catalog.gone and ident not in self.cart.ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.order.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.cart.audited or not self.models.judges():
            return ""
        self.cart.audited = True
        got = self.audit_call()
        named = self.audit_fits(got)
        if not named:
            # rows whose page was never read can still fit: open the promising ones (one step) and look again
            rows = [str(i) for i in _items((got or {}).get("open")) if str(i) in self.catalog.rows and str(i) not in self.catalog.opened and not self.catalog.sold_out(str(i))]
            if rows and self.browser.open_rows(rows):
                got = self.audit_call()
                named = self.audit_fits(got)
        if not named:
            return ""
        self.catalog.lead_items = named
        self.catalog.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self) -> dict[str, Any] | None:
        records = []
        # every listing seen counts, not only the newest: a fitting row first seen early is often the one left unopened
        # (race 14: declines with an accepted listing among the agent's own early search rows), so rows over the budget
        # make room for older ones
        budget = self.order.budget if isinstance(self.order.budget, (int, float)) and self.order.brief.get("budget_scope") != "each" else None
        for ident in reversed(self.catalog.seen_order):
            price = self.catalog.price_of(ident)
            if self.catalog.sold_out(ident) or (budget is not None and price is not None and budget < price <= 10 * budget):
                continue
            record = self.catalog.full.get(ident) if ident in self.catalog.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.catalog.rows.get(ident), 120))[:400]))
            if len(records) >= 60:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.order.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.order.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.models.ask_json(prompt, self.models.judges(), 4000)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def line_picks(self) -> dict[int, str]:
        """For an ending without a purchase on a several-product order: one listing per item that no record rules out,
        read by a judge from every in-stock listing seen; empty unless every item gets a pick and the lines fit the budget."""
        records = []
        for ident in reversed(self.catalog.seen_order):
            if not self.catalog.buyable(ident) or ident in self.catalog.gone:
                continue
            record = self.catalog.full.get(ident) or self.catalog.rows.get(ident)
            records.append(_dump(_clip(record, 200))[:1100])
            if len(records) >= 36:
                break
        judges = self.models.judges(Settings.SPEND_HARD)
        if not records or not judges:
            return {}
        prompt = LINE_PICKS + "\n\nREQUEST:\n" + self.order.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.order.ledger()
        prompt += "\n\nLISTING RECORDS (in stock; newest first):\n" + "\n".join(records)
        got = self.models.ask_json(prompt, judges, 2500) or {}
        self.trace.append({"kind": "line_picks", "got": _clip(got, 300)})
        raw = got.get("picks") if isinstance(got.get("picks"), dict) else {}
        picks = {}
        for key, ident in raw.items():
            number = _count(key)
            if number and 1 <= number <= len(self.order.items) and isinstance(ident, str) and ident in self.catalog.rows and self.catalog.buyable(ident):
                picks[number] = ident
        if len(picks) < len(self.order.items) or len(set(picks.values())) < len(picks):
            return {}
        if isinstance(self.order.budget, (int, float)):
            prices = [self.catalog.price_of(i) for i in picks.values()]
            if all(p is not None for p in prices):
                each = self.order.brief.get("budget_scope") == "each"
                cost = [p * (1 if each else self.order.items[n - 1].qty) for n, p in zip(picks, prices)]
                if (max(cost) if each else sum(cost)) > self.order.budget + 1e-09:
                    return {}
        return picks

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.catalog.rows and not self.catalog.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Arguments that make the cart hold `lines` and none of `drop`, read off the cart tool's schema; None when it cannot be said."""
        name = self.toolbox.roles.get("cart") or ""
        schema = self.toolbox.schemas.get(name) or {}
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
        target = (next((k for t, k in self.toolbox.id_args if t == name and k in texts), None)
                  or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None))
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart.lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.cart.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart.lines.get(ident, 0) if ident in self.cart.ids() else 0
            if have == want:
                continue
            if have and not self.cart.cart_sets(name):
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        cart, buy = self.toolbox.roles.get("cart"), self.toolbox.roles.get("buy")
        if self.cart.completed or not self.cart.confirmed or self.catalog.sold_out(self.cart.confirmed) or not cart or not buy:
            return ""
        self.cart.completed = True
        cat_key = self.catalog.category_key()

        def kind(ident: str) -> str:
            return str((self.catalog.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        plan: dict[str, int] = {}
        for number, ids in self.catalog.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if not self.catalog.sold_out(i)), None)
            if pick and index and 1 <= index <= len(self.order.items) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.order.items[index - 1].qty
        if self.cart.confirmed not in plan:
            return ""
        now = self.cart.ids()
        keep = {i: self.cart.lines.get(i, 1) for i in now if i not in plan and not self.catalog.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.order.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.clock.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        self.browser.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), Settings.SPEND_HARD)
        if not self.cart.verdicts.get(self.order.check_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, self.toolbox.schemas.get(cart), "args", faults)
            if faults:
                return ""
            reply = self.store.act(cart, args)
            if reply is None or self.store.finished:
                return ""
            self.cart.note_cart(cart, args)
            self.browser.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.order.brief.get("buy_args") or {}), self.toolbox.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.browser.reread_offers()
                self.check_cart(list(lines), Settings.SPEND_HARD)
                if (not self.cart.verdicts.get(self.order.check_key(list(lines))) or self.clock.left() < 1 or self.store.finished
                        or any(self.catalog.sold_out(i) for i in lines)):
                    break
            go, said = self.talk.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.store.act(buy, args)
            self.trace.append({"kind": "completed", "lines": lines, "drop": drop, "reply": _clip(reply, 200)})
            if reply is None:
                break
            self.cart.buys += 1
            if not reply.get("finished"):
                self.cart.refused_at = self.clock.step
            self.browser.observe(buy, reply.get("observation"))
            replies.append(_show(reply.get("observation")))
            if self.store.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.cart.confirmed + ", so Kyo set the cart (" + "; ".join(done)
                + ") and checked out. Store reply: " + " | then: ".join(replies))

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.toolbox.schemas.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.catalog.rows) or self.order.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode: check out a held cart, else end without a purchase. With `again`, a refused checkout is
        returned (as the store's reply) for the agent to deal with instead of being followed by an ending without a purchase."""
        if self.store.finished or self.store.broke or self.clock.left() <= 0:
            return ""
        buy, refuse = self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse")
        fresh = self.cart.view is not None and self.cart.seen_at >= self.cart.at
        holding = bool(self.cart.ids()) if fresh else bool(self.cart.lines)
        if buy and (holding or (self.cart.touched and not fresh and self.clock.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.order.brief.get("buy_args") or {}), self.toolbox.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.clock.left() >= 2:
                    self.talk.gate("order")
                if self.store.finished or self.clock.left() <= 0:
                    return ""
                reply = self.store.act(buy, args)
                self.trace.append({"kind": "forced_buy", "reply": _clip(reply, 200)})
                if self.store.finished or self.clock.left() <= 0:
                    return ""
                if reply is not None and again:
                    self.cart.buys += 1
                    self.cart.refused_at = self.clock.step
                    self.browser.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + _show(reply.get("observation"))
        if refuse:
            faults = []
            args = _shape(self.quiet(refuse, dict(self.order.brief.get("refuse_args") or {})), self.toolbox.schemas.get(refuse), "args", faults)
            if not faults:
                if self.clock.left() >= 2:
                    self.talk.gate("end")
                if self.store.finished or self.clock.left() <= 0:
                    return ""
                reply = self.store.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                if reply is not None and not self.store.finished:
                    self.browser.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + _show(reply.get("observation"))
        return ""


# ==============================================================================================================
# EVERY CALL THE MODEL SENDS: admit and hold (Referee), run (Store), learn and reply (Results)
# ==============================================================================================================

    def held_units(self, ident: str) -> int:
        if self.cart.view is not None and self.cart.seen_at >= self.cart.at:
            stack: list[Any] = [self.cart.view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [
                            _count(v)
                            for k, v in node.items()
                            if not isinstance(v, str) and _count(v) is not None and re.search("\\b(?:qty|quantity|count|units?)\\b", _words(k))
                        ]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart.lines.get(ident, 1)

    def line_faults(self) -> list[str]:
        cat_key = self.catalog.category_key()
        ids = [i for i in self.cart.ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.catalog.rows.get(i) or {}).get(cat_key) or "") for i in ids}
        known = {n for n, item in enumerate(self.order.items, 1) if any((self.order.owns(item, str(r.get(cat_key) or "")) for r in self.catalog.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.order.items, 1) if self.order.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.order.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f"item {n} ({item.name}) has no line in the cart")
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f"{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}")
        if len(known) == len(self.order.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.catalog.sold_out(i):
                faults.append(f"{i} is sold out")
        prices = {i: self.catalog.price_of(i) for i in ids}
        for n, item in enumerate(self.order.items, 1):
            if item.max_price is not None:
                faults += [f"{i} costs {prices[i]:g}, over the {item.max_price:g} limit for one unit of item {n} ({item.name})"
                           for i in ids if n in owner[i] and prices[i] is not None and prices[i] > item.max_price + 1e-09]
        if isinstance(self.order.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.order.brief.get("budget_scope") == "each":
                over = [i for i in ids if self.order.budget < prices[i] <= 10 * self.order.budget]
                faults += [f"{i} costs {prices[i]:g}, over the {self.order.budget:g} limit for each item" for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                if self.order.budget + 1e-09 < total <= 10 * self.order.budget:
                    faults.append(f"the total is {total:g}, over the budget of {self.order.budget:g}")
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.catalog.category_key()
        if not cat_key:
            return []
        out = []
        for ident in self.cart.ids():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.order.items, 1) if self.order.owns(item, str((self.catalog.rows.get(ident) or {}).get(cat_key) or ""))]
            mine = [
                i
                for i in self.cart.ids()
                if (self.cart.lines.get(i, 1) or 0) > 0
                and len(owners) == 1
                and self.order.owns(self.order.items[owners[0] - 1], str((self.catalog.rows.get(i) or {}).get(cat_key) or ""))
            ]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.order.items[owners[0] - 1].qty) and (not self.catalog.sold_out(ident)):
                out.append((ident, have, self.order.items[owners[0] - 1].qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.toolbox.roles.get("cart") or ""
        schema = self.toolbox.schemas.get(name) or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and ("enum" not in v)]
        target = (
            next((k for t, k in self.toolbox.id_args if t == name and k in texts), None)
            or next((k for k in schema.get("required") or [] if k in texts), None)
            or (texts[0] if len(texts) == 1 else None)
        )
        count = next((k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and (len(v["enum"]) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart.cart_sets(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]["enum"]]
                off = next((o for o in options if re.search("remov|delet|drop|clear|take", o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.clock.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, schema, "args", faults)
            if faults:
                break
            reply = self.store.act(name, args)
            if reply is None or self.store.finished:
                break
            self.cart.note_cart(name, args)
            self.cart.said = _dump(reply.get("observation"))
            self.browser.observe(name, reply.get("observation"))
            done.append(_dump(args) + " -> " + _show(reply.get("observation"))[:300])
        self.trace.append({"kind": "count_fix", "calls": done})
        return done

    def fit_faults(self) -> list[str]:
        """Parts in the cart that their own records say do not work with the device in the cart. A judge reads the records
        (compatibility lists, model series and their exclusions, notes about other part numbers); an unknown is not a fault."""
        if not self.order.bound:
            return []
        cart = self.cart.ids()
        faults = []
        for part, main in self.order.bound:
            devices = [i for i in cart if i in self.catalog.item_ids(main)]
            parts = [i for i in cart if i in self.catalog.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            key = devices[0] + ":" + ",".join(sorted(parts))
            got = self.fit_seen.get(key)
            if got is None:
                judges = self.models.judges(Settings.SPEND_HARD)
                if not judges:
                    continue
                others = [i for i in reversed(self.catalog.seen_order) if i in self.catalog.item_ids(part) and i not in cart and not self.catalog.sold_out(i)][:10]
                prompt = FIT_JUDGE + f"\n\nDEVICE ({self.order.items[main - 1].name}):\n{devices[0]}: {self.catalog.record_of(devices[0])}"
                prompt += f"\n\nPART LINES IN THE CART ({self.order.items[part - 1].name}):\n" + "\n".join(f"{i}: {self.catalog.record_of(i)}" for i in parts)
                if others:
                    prompt += "\n\nOTHER LISTINGS OF THAT KIND SEEN:\n" + "\n".join(f"{i}: {self.catalog.record_of(i)[:900]}" for i in others)
                got = self.models.ask_json(prompt, judges, 1500) or {}
                self.fit_seen[key] = got
                self.trace.append({"kind": "fit_judge", "device": devices[0], "parts": parts, "got": _clip(got, 300)})
            good = [str(i) for i in _items(got.get("others")) if str(i) in self.catalog.rows and str(i) not in cart and not self.catalog.sold_out(str(i))][:4]
            for entry in _items(got.get("lines")):
                if not isinstance(entry, dict) or str(entry.get("id")) not in parts or str(entry.get("fits")).lower() != "no":
                    continue
                faults.append(f"{entry['id']} ({self.order.items[part - 1].name}) does not work with {devices[0]}" + (f" ({entry.get('why')})" if entry.get("why") else "")
                              + (f"; seen listings of that kind that do: {', '.join(good)}" if good else "; no seen listing of that kind says it does, so look for one or switch the device"))
        return faults

class Call:
    """One tool call the model asked for, as the Referee admitted it: its arguments after checks, the mark that spots
    a repeat, and what the reply must add (trimmed lists, further pages)."""
    def __init__(self, cid: str, name: str, args: dict[str, Any], mark: str, many: str | None, target: str | None,
                 ending: bool, readers: set[str], trimmed: str):
        self.cid = cid
        self.name = name
        self.args = args
        self.mark = mark
        self.many = many
        self.target = target
        self.ending = ending
        self.readers = readers
        self.trimmed = trimmed
        self.paged = ""
        self.plain: set[str] = set()


class Referee:
    """Before a call reaches the store: which tools the model is offered now, whether a call may run at all (an
    unknown tool, broken arguments, a repeat, a view with nothing new to show), and whether a checkout or an ending
    is held back for a check, a sold-out line, a wider search, an audit or the steps the request asks for."""
    def __init__(self, toolbox: Toolbox, catalog: Catalog, cart: Cart, order: Order, store: Store, clock: Clock, models: Models,
                 talk: Talk, browser: Browser, checkout: Checkout, trace: list[dict[str, Any]]):
        self.toolbox = toolbox
        self.catalog = catalog
        self.cart = cart
        self.order = order
        self.store = store
        self.clock = clock
        self.models = models
        self.talk = talk
        self.browser = browser
        self.checkout = checkout
        self.trace = trace
        self.searches = 0            # searches the model sent (Choji's own lookups are not counted)
        self.search_held = False     # the one hold on a long search run for a single product

    def offered(self) -> list[dict[str, Any]]:
        left = self.clock.left()
        ask = self.toolbox.roles.get("ask")
        keep: set[str] | None = None
        ends = {self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse")} - {None}
        if left <= self.talk.tail():
            keep = set(ends)
        elif left <= self.talk.tail() + 3:
            keep = set(ends) | {self.toolbox.roles.get("cart"), self.toolbox.roles.get("cart_view")}
        elif left <= len(self.order.items) + 5 and self.cart.buys == 0:
            keep = set(ends) | {self.toolbox.roles.get("cart"), self.toolbox.roles.get("cart_view"), self.toolbox.roles.get("offer")}
        tools = [t for t in self.toolbox.tools if t["function"]["name"] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def admit(self, call: dict[str, Any], count: int) -> tuple[Call | None, str]:
        """The call ready to run, or None and the reason it is not run (no step used)."""
        calls_count = count
        allowed = {t["function"]["name"] for t in self.offered()}
        cid = call["id"]
        if call.get("joined"):
            return None, "Shown in the side-by-side result above: Kyo opened these listings together in one step."
        fn = call.get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments")
        if self.store.finished:
            return None, "Not run: the episode is over."
        if name == self.toolbox.roles.get("ask"):
            return None, "Not run (no step used): Kyo's interviewer handles every message to the shopper."
        if name not in self.toolbox.schemas:
            return None, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."
        if name not in allowed:
            return None, "Not run (no step used): too few steps are left for that; use: " + ", ".join(sorted(allowed)) + "."
        if isinstance(raw, dict):
            args = raw
        else:
            args = _json_obj(raw) if (raw or "").strip() else {}
            if args is None:
                return None, "Not run (no step used): the arguments were not one JSON object."
        faults: list[str] = []
        args = _shape(args, self.toolbox.schemas[name], "args", faults)
        if faults:
            return None, "Not run (no step used): " + "; ".join(faults[:6])
        args, wrong = self.browser.resolve_ids(name, args)
        if wrong:
            return None, "Not run (no step used): " + wrong + "."
        trimmed = ""
        many = self.browser.compare_target() if name == self.toolbox.roles.get("compare") else None
        if many and isinstance(args.get(many), list) and len(args[many]) > self.browser.compare_room() and self.toolbox.compare_bad:
            room = self.browser.compare_room()
            trimmed = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
            args = {**args, many: args[many][:room]}
        target = self.toolbox.single_target(name)
        if target and name in (self.toolbox.roles.get("open"), self.toolbox.roles.get("offer")) and str(args.get(target)) in self.catalog.gone:
            return None, f"Not run (no step used): {args[target]} sold out at step {self.catalog.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing."
        if name == self.toolbox.roles.get("cart") and any((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
            view = _dump(self.cart.view) if self.cart.seen_at >= self.cart.at else ""
            absent = [v for v in args.values() if isinstance(v, str) and v in self.catalog.rows and v not in self.cart.lines and v not in view and v not in self.cart.said]
            if absent:
                return None, f"Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: " + self.cart.text()
        ending = self.toolbox.is_buy(name, args) or self.toolbox.is_refuse(name, args)
        if ending and calls_count > 1:
            return None, "Not run (no step used): send the finishing call on its own, after reading your other results."
        if not ending and self.clock.left() <= self.talk.tail() and (self.toolbox.roles.get("buy") or self.toolbox.roles.get("refuse")):
            return None, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."
        mark = name + _dump(args)
        if not ending and name != self.toolbox.roles.get("cart") and mark in self.store.recent:
            return None, "Not run (no step used): you just made this same call and nothing changed; its result is above."
        readers = {self.toolbox.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
        cached = self.catalog.read_cache.get(mark)
        if name in readers and cached and cached[0] == self.order.version:
            return None, f"Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000]
        if target and name in readers and name != self.toolbox.roles.get("compare") and self.catalog.shown_already(name, str(args.get(target))):
            return None, f"Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: " + self.catalog.record_of(str(args[target]))
        if many and isinstance(args.get(many), list) and args[many]:
            keep = [i for i in args[many] if not self.catalog.shown_already(name, str(i))]
            if not keep:
                return None, ("Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n"
                            + "\n".join(f"{i}: {self.catalog.record_of(str(i))}" for i in args[many])[:4000])
            if len(keep) < len(args[many]):
                trimmed += f"\n[Kyo] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                args = {**args, many: keep}
                mark = name + _dump(args)
        if name == self.toolbox.roles.get("search"):
            # one product, six searches made and nothing carted: answers almost always show up by now (races 11-13:
            # 95% of the runs that ever saw one had seen it by the 6th search), so the seventh waits once for a choice
            if len(self.order.items) == 1 and self.searches >= 6 and not self.search_held and not self.cart.ids() and self.cart.buys == 0:
                self.search_held = True
                self.trace.append({"kind": "search_held", "searches": self.searches})
                return None, ("Not run (no step used): six searches are done and the cart is empty. Choose now among the listings you have seen: open the "
                              "promising ones you have not opened, cart the one that meets the requirements and check out. A requirement no listing states "
                              "does not rule a listing out. Search again only if every seen listing clearly breaks a requirement.")
            self.searches += 1
        return Call(cid, name, args, mark, many, target, ending, readers, trimmed), ""

    def hold(self, c: Call) -> str | None:
        """Why a checkout or an ending waits (code may have read or fixed things meanwhile), or None to let it run."""
        name, args, ending = c.name, c.args, c.ending
        if ending and self.clock.left() > 2:
            if self.toolbox.is_buy(name, args):
                wider = self.browser.wider_hold()
                if wider:
                    return wider + "\nCart now: " + self.cart.text()
                reread = self.browser.reread_offers()
                if self.cart.note_sold:
                    self.checkout.find_sold()
                if self.cart.touched and self.cart.seen_at < self.cart.at and self.toolbox.roles.get("cart_view") and self.clock.left() > 3:
                    faults = []
                    view_args = _shape({}, self.toolbox.schemas.get(self.toolbox.roles["cart_view"]), "args", faults)
                    view = self.store.act(self.toolbox.roles["cart_view"], view_args) if not faults else None
                    if view is not None:
                        self.cart.view = view.get("observation")
                        self.cart.seen_at = self.clock.step
                        change = self.browser.observe(self.toolbox.roles["cart_view"], self.cart.view)
                        if change:
                            self.talk.queued = []
                            return "Not run yet (no step used). Looking at the cart showed a change: " + change + "\nCart now: " + self.cart.text() + "\nApply the change, fix the cart if needed, then check out."
                swap = self.checkout.swap_hint()
                # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                held = [i for i in self.cart.ids() if i in self.catalog.gone] if self.cart.held < 2 else []
                if swap or held:
                    self.cart.held += 0 if swap else 1
                    return "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart.text()
                chain, hold = self.browser.trail()
                if hold:
                    return "Not run yet (no step used). " + chain + "\nCart now: " + self.cart.text()
                misfit = self.checkout.fit_faults() if self.cart.fit_holds < 2 and self.clock.left() > self.talk.tail() + 3 else []
                if misfit:
                    self.cart.fit_holds += 1
                    self.trace.append({"kind": "fit_hold", "faults": misfit[:4]})
                    return ("Not run yet (no step used). A part in the cart does not work with the device in the cart: " + "; ".join(misfit[:3])
                                + ". Swap the part (or the device), then check out.\nCart now: " + self.cart.text())
                faults = self.checkout.line_faults() if self.cart.line_holds < 2 and self.clock.left() > self.talk.tail() + 2 else []
                fixed = []
                if faults and all(" is held at " in f for f in faults):
                    fixed = self.checkout.fix_counts(self.checkout.count_fixes())
                    faults = self.checkout.line_faults() if fixed else faults
                    if fixed and not faults:
                        self.cart.line_holds += 1
                        return ("Not run yet (no step used). Kyo set the cart's quantities itself: " + "; ".join(fixed)
                                    + ". If the store refused any of these, fix those lines; then check out.\nCart now: " + self.cart.text())
                if faults:
                    self.cart.line_holds += 1
                    self.trace.append({"kind": "line_hold", "faults": faults[:6]})
                    return ("Not run yet (no step used). The cart does not match the request yet: " + "; ".join(faults[:6])
                                + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.cart.text())
                # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                read = self.browser.fill_facts(self.cart.ids(), self.talk.tail() + 3) + self.browser.cheaper_rows()
                verdict = self.checkout.check_cart()
                if verdict:
                    seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                    return "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart.text()
                if reread and self.models.cost > Settings.SPEND_SOFT:
                    return "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."
            else:
                if self.clock.left() > 5:
                    review = self.checkout.audit()
                    if review:
                        return "Not run yet (no step used). " + review
                    if self.catalog.leads and self.clock.left() > 4 and self.cart.refuse_blocks == 0:
                        self.cart.refuse_blocks = 1
                        self.trace.append({"kind": "refuse_blocked", "leads": list(self.catalog.leads)})
                        return "Not run yet (no step used). The review found listings that appear to fit: " + ", ".join(self.catalog.leads) + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again."
                    if self.catalog.leads and self.clock.left() > 4 and self.cart.refuse_blocks == 1:
                        self.cart.refuse_blocks = 2
                        first = self.catalog.leads[0]
                        self.checkout.check_cart([first], Settings.SPEND_HARD)
                        if self.cart.verdicts.get(self.order.check_key([first])) and first in self.catalog.leads:
                            self.cart.confirmed = first
                            self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                            return f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."
        if self.toolbox.is_refuse(name, args) and self.clock.left() <= 5 and not self.cart.late_held:
            # too late for the review: a decline here usually throws away a built cart because time ran out
            lines = [i for i in self.cart.built if not self.catalog.sold_out(i)]
            fresh = self.cart.view is not None and self.cart.seen_at >= self.cart.at
            missing = [i for i in lines if i not in self.cart.ids() or (i in self.order.dropped and not fresh)]
            if len(lines) >= len(self.order.items) and len(missing) + 1 <= self.clock.left():
                self.cart.late_held = True
                self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                return ("Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                            + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.")
        if self.toolbox.is_refuse(name, args) and not self.catalog.leads and not self.cart.pushed and self.clock.left() > self.clock.max_steps // 3:
            self.cart.pushed = True
            return f"Not run yet (no step used). {self.clock.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."
        if self.toolbox.is_refuse(name, args) and len(self.order.items) > 1 and not self.cart.lines_offered and self.clock.left() >= len(self.order.items) + 2:
            # several products: one hold that names a fitting listing for every item; a second decline goes through
            self.cart.lines_offered = True
            picks = self.checkout.line_picks()
            if picks:
                plan = "; ".join(f"item {n} ({self.order.items[n - 1].name}): {i} x{self.order.items[n - 1].qty}" for n, i in sorted(picks.items()))
                return ("Not run yet (no step used). Ending without a purchase is right only when some item has no acceptable listing, and a review of the "
                        f"records found one for every item: {plan}. Set the cart to exactly these lines and check out; end without a purchase only if one of "
                        "them clearly breaks a requirement on its own record.\nCart now: " + self.cart.text())
        if self.toolbox.is_refuse(name, args):
            done = self.checkout.complete_cart()
            if done:
                return done + (" [episode finished]" if self.store.finished else "")
            c.args = args = self.checkout.quiet(name, args)
        if ending and self.clock.left() >= 2:
            go, said = self.talk.gate("order" if self.toolbox.is_buy(name, args) else "end")
            if not go:
                return "Not run yet. " + said + "\nCart now: " + self.cart.text()
            if said:
                c.trimmed = "\n[Kyo] " + said + c.trimmed
        return None


class Results:
    """After a call: what its result teaches code (which view shows attributes, the cart's lines, offers seen), and
    the reply the model sees (the result, further pages, evidence lines, notes read, sold-out hints)."""
    def __init__(self, toolbox: Toolbox, catalog: Catalog, cart: Cart, order: Order, store: Store, clock: Clock,
                 talk: Talk, browser: Browser, checkout: Checkout):
        self.toolbox = toolbox
        self.catalog = catalog
        self.cart = cart
        self.order = order
        self.store = store
        self.clock = clock
        self.talk = talk
        self.browser = browser
        self.checkout = checkout

    def learn(self, c: Call, reply: dict[str, Any]) -> None:
        name, args, many, mark, ending, readers, trimmed = c.name, c.args, c.many, c.mark, c.ending, c.readers, c.trimmed
        observation = reply.get("observation")
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            shown = _dump(observation)
            if not any(str(i) in shown for i in args[many]):
                self.toolbox.compare_bad = min(self.toolbox.compare_bad or 999, len(args[many]))
                trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.browser.compare_room()} per call."
            else:
                self.toolbox.compare_ok = max(self.toolbox.compare_ok, len(args[many]))
        self.store.recent = (self.store.recent + [mark])[-3:]
        self.toolbox.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.catalog.rows for x in (v if isinstance(v, list) else [v])))
        self.catalog.absorb(observation)
        if name == self.toolbox.roles.get("compare"):
            rows = self.catalog.records_in(observation)
            if rows:
                self.toolbox.facts_from[name] = self.toolbox.facts_from.get(name, False) or any(_fact_block(r) for _, r in rows)
        if name in readers - {self.toolbox.roles.get("search")}:
            self.catalog.note_view(name, observation)
        # every job but searching: only a search result is paged
        plain = {tool for job, tool in self.toolbox.roles.items() if job != "search"}
        paged = self.browser.more_pages(name, args, observation) if name not in plain and not ending else ""
        if self.toolbox.single_target(name) and name not in (self.toolbox.roles.get("cart"), self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse"), self.toolbox.roles.get("ask")):
            listing = self.catalog.ident(observation) if isinstance(observation, dict) else None
            if listing:
                self.toolbox.facts_from[name] = self.toolbox.facts_from.get(name, False) or bool(_fact_block(observation))
                if name == self.toolbox.roles.get("open") and not self.toolbox.facts_from[name]:
                    other = next((t for t in self.toolbox.schemas if t != name and self.toolbox.single_target(t) and self.toolbox.facts_from.get(t) is not False
                                  and t not in (self.toolbox.roles.get("cart"), self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse"), self.toolbox.roles.get("ask"), self.toolbox.roles.get("search"))), None)
                    if other:
                        self.toolbox.roles["open"] = other
                if not self.toolbox.facts_from[name] and name != self.toolbox.roles.get("open") and not self.toolbox.roles.get("offer"):
                    self.toolbox.roles["offer"] = name
                if name == self.toolbox.offer_tool():
                    self.catalog.offer_seen[listing] = self.clock.step
        if name == self.toolbox.roles.get("cart"):
            self.cart.note_cart(name, args)
            self.cart.said = _dump(observation)
            self.store.recent = []
            # lines taken off by a zero or negative count (unless the reply still lists them), and the fullest
            # cart built so far, which is offered back when a late decline would throw it away
            counts = [_count(v) for v in args.values() if not isinstance(v, str) and _count(v) is not None]
            for value in args.values():
                if counts and isinstance(value, str) and value in self.catalog.rows:
                    if counts[0] <= 0 and not _in_rows(observation, value):
                        self.order.dropped.add(value)
                    elif counts[0] > 0:
                        self.order.dropped.discard(value)
            lines = [i for i in self.cart.ids() if not self.catalog.sold_out(i) and i not in self.order.dropped]
            if len(lines) >= len(self.cart.built):
                self.cart.built = lines
        if name == self.toolbox.roles.get("search") and self.catalog.records_in(observation):
            self.catalog.searched = (self.catalog.searched + [dict(args)])[-8:]
        if name == self.toolbox.roles.get("cart_view"):
            self.cart.view = observation
            self.cart.seen_at = self.clock.step
        if self.toolbox.is_buy(name, args):
            self.cart.buys += 1
            if not reply.get("finished"):
                self.cart.refused_at = self.clock.step
        c.trimmed, c.paged, c.plain = trimmed, paged, plain

    def text(self, c: Call, reply: dict[str, Any]) -> str:
        name, args, mark, ending, readers, trimmed, paged, plain = c.name, c.args, c.mark, c.ending, c.readers, c.trimmed, c.paged, c.plain
        observation = reply.get("observation")
        text = _show(observation) + paged + trimmed
        if name not in plain and not ending:
            text += self.catalog.cheapest_here([observation] + self.catalog.paged_obs) + self.cap_ignored(name, args, observation)
        base = text
        proof = self.catalog.evidence(observation)
        if proof:
            text += "\n" + proof
        if name == self.toolbox.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in self.catalog.rows and _in_rows(observation, v) for v in args.values()):
                text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
        self.cart.note_sold = False
        lines = self.talk.loose_lines(observation)
        if lines:
            meaning = self.talk.take_note(lines)
            if meaning:
                text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
        refused = self.toolbox.is_buy(name, args) and not reply.get("finished")
        again = refused and any(line in self.talk.spoken(observation) for line in self.cart.sold_lines)
        loose = self.cart.note_sold and (refused or (not self.cart.note_named and not self.toolbox.is_buy(name, args)))
        if (loose or again) and self.cart.ids() and not self.store.finished:
            seen = self.checkout.find_sold()
            if seen:
                text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
        self.cart.note_sold = False
        swap = self.checkout.swap_hint()
        if swap:
            text += "\n" + swap
        if self.talk.queued:
            text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.talk.queued)
            self.talk.queued = []
        if name in readers:
            self.catalog.read_cache[mark] = (self.order.version, self.clock.step, base)
        if reply.get("finished"):
            text += " [episode finished]"
        return text


# ==============================================================================================================
# THE EPISODE
# ==============================================================================================================

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.toolbox.schemas.get(name) or {}).get("properties") or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search("price|cost|amount|budget|spend|money", key + " " + str(spec.get("description") or ""), re.I))

        caps = [
            k
            for k, v in args.items()
            if k in props
            and isinstance(v, (int, float))
            and (not isinstance(v, bool))
            and re.search("max|most|under|below|limit|cap|upto|up_to", k, re.I)
            and priced(k)
        ]
        if not caps or not self.toolbox.price_key:
            return ""
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.catalog.records_in(observation) if (self.catalog.price_of(i) or 0) > cap]
        if not over:
            return ""
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get("enum"), list)}
        others = {k: [o for o in props[k]["enum"] if o != v] for k, v in modes.items()}
        hint = (
            "; list the category with "
            + ", ".join((f"{k}={_dump(o[0])}" for k, o in others.items() if o))
            + f" and the same {caps[0]} to see only cheaper listings"
            if any(others.values())
            else ""
        )
        return f"\n[Kyo] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}."


class World:
    def __init__(self, problem: dict[str, Any]):
        environment = problem["environment"]
        view = environment["policy_view"]
        steps = _count(view.get("max_steps"))
        if steps is None and isinstance(view.get("max_steps"), float):
            steps = int(view["max_steps"])
        self.trace: list[dict[str, Any]] = []
        self.clock = Clock(steps if steps and steps > 0 else 30)
        self.models = Models(self.clock, self.trace, lambda: self.store.broke)
        self.order = Order(view, self.models, self.trace, lambda: self.toolbox.attributes_note())
        self.toolbox = Toolbox(view, self.models, self.order, self.trace)
        self.catalog = Catalog(self.toolbox, self.order, self.trace)
        self.order.described = self.catalog.described
        self.store = Store(dict(environment["binding"]), self.clock, self.toolbox, self.catalog, self.trace, bool(view.get("finished")))
        self.cart = Cart(self.catalog, self.toolbox, self.clock)
        self.talk = Talk(self.order, self.catalog, self.cart, self.store, self.models, self.toolbox, self.clock, self.trace)
        self.intake = Intake(self.order, self.toolbox, self.talk, self.clock, self.models, self.trace)
        self.browser = Browser(self.store, self.toolbox, self.catalog, self.order, self.cart, self.talk, self.clock, self.models, self.trace)
        self.talk.browser = self.browser
        self.checkout = Checkout(self.cart, self.catalog, self.order, self.toolbox, self.talk, self.browser, self.store, self.clock, self.models, self.trace)
        self.referee = Referee(self.toolbox, self.catalog, self.cart, self.order, self.store, self.clock, self.models, self.talk, self.browser, self.checkout, self.trace)
        self.results = Results(self.toolbox, self.catalog, self.cart, self.order, self.store, self.clock, self.talk, self.browser, self.checkout)
        self.chat: list[dict[str, Any]] = []

    def status(self) -> str:
        left = self.clock.left()
        text = f"[Kyo] step {self.clock.step} of {self.clock.max_steps} used, {left} left."
        if left <= self.talk.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.order.items) + 5 and self.cart.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.cart.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.cart.buys == 0 and not self.cart.touched and self.clock.step >= self.clock.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.clock.seconds() >= Settings.HURRY_FROM and left > self.talk.tail() + 1:
            text += " TIME: the episode's time is almost gone, whatever steps remain. Put the chosen lines in the cart and check out in your next two turns; a run that is cut off by time scores nothing."
        live = {need for _, need in self.order.needs_now()}
        lookups = [need for _, need in self.order.refs if need in live and need not in self.order.ref_values]
        if lookups and self.cart.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.catalog.leads:
            text += "\n[review candidates] " + ", ".join(self.catalog.leads)
        if self.order.bound and self.cart.buys == 0:
            text += "\n[fit: each part must say on its own record that it works with the device's model; Choji checks this before the order]"
        if self.order.notes:
            text += "\n[changes so far] " + " | ".join(self.order.notes[-4:])
        text += "\n[ledger]\n" + self.order.ledger()
        return text

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= Settings.HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.catalog.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.store.finished:
            return self.trace
        # step 0: read the request
        self.intake.read()
        self.trace.append({"kind": "brief", "brief": self.order.brief})
        opening = self.order.view.get("opening")
        if opening is not None:
            self.catalog.absorb(opening)
        resolved = self.browser.resolve_pointers()
        intro = "SHOPPER'S REQUEST:\n" + self.order.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + _show(opening)
        if resolved:
            intro += "\n\n" + resolved
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        # steps 1 to 4: the light model works the store; Talk asks the shopper when it is time (step 2)
        self.work(Settings.LOOP_CALLS, Settings.SPEND_HARD)
        # the spend or turn cap reached with nothing in the cart: a short last round to cart what fits, rather than an
        # ending without a purchase that no search showed is right (race-15 replay: 2 of 4 wrong declines ended so)
        if not self.store.finished and not self.store.broke and not self.models.no_credit and not self.cart.ids() and self.clock.left() > 6 and self.clock.seconds() < Settings.LOOP_UNTIL:
            self.chat.append({"role": "user", "content": "[Choji] Little thinking budget is left. Put the best fitting listing you have read for each item in the cart now and check out; end without a purchase only for an item that no listing you read can fill.\n" + self.status()})
            self.trace.append({"kind": "last_round", "step": self.clock.step, "cost": round(self.models.cost, 4)})
            self.work(self.clock.loop_calls + 8, Settings.SPEND_HARD + 0.03)
        # step 4, the end: code checks out or ends when the model did not
        if not self.store.finished:
            turned = self.checkout.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.store.finished and not self.store.broke and not self.models.no_credit and self.clock.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.clock.loop_calls + 10, Settings.SPEND_HARD + 0.01)
            if not self.store.finished:
                self.checkout.finish()
        self.trace.append({"kind": "end", "finished": self.store.finished, "steps": self.clock.step, "cost": round(self.models.cost, 5), "calls": self.models.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.store.finished and not self.store.broke and self.clock.left() > 0 and self.clock.loop_calls < loop_cap and self.models.cost < spend_cap:
            # out of time: the code's finish checks out what the cart holds; an empty cart gets the model while room lasts
            if self.clock.seconds() >= Settings.LOOP_UNTIL and (self.cart.ids() or self.clock.room() < 12):
                self.trace.append({"kind": "loop_timed_out", "at": round(self.clock.seconds(), 1), "step": self.clock.step})
                break
            # step 2: once enough listings are open, or before the first cart or ending call
            if self.talk.ready_to_ask():
                summary = self.talk.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.clock.loop_calls += 1
            message = self.models.think(self.chat, [Settings.LIGHT, Settings.SPARE], 3000, self.referee.offered())
            if message is None:
                self.clock.misses += 1
                if self.models.no_credit or self.models.calls >= Settings.ALL_CALLS or self.clock.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else (_json_obj(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({"id": str(call.get("id") or f"k{self.clock.loop_calls}_{number}"), "type": "function",
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
            committing = {self.toolbox.roles.get("cart"), self.toolbox.roles.get("buy"), self.toolbox.roles.get("refuse")} - {None}
            if not self.talk.asked and self.toolbox.roles.get("ask") and self.talk.free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.clock.left() > 8 and self.clock.seconds() <= Settings.INTERVIEW_UNTIL:
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.talk.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            # every call: Referee admits or holds it, Store runs it, Results learns from it and answers
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Each call the model sent: admitted or not, held or not, run in the store, learned from, answered."""
        out: list[tuple[str, str]] = []
        calls = self.browser.join_opens(calls)
        for call in calls:
            c, refused = self.referee.admit(call, len(calls))
            if c is None:
                out.append((call["id"], refused))
                continue
            held = self.referee.hold(c)
            if held is not None:
                out.append((c.cid, held))
                continue
            reply = self.store.act(c.name, c.args)
            if reply is None:
                out.append((c.cid, "The call failed (network or server). You may try again."))
                continue
            self.results.learn(c, reply)
            out.append((c.cid, self.results.text(c, reply)))
        return out


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = World(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.store.finished:
                episode.checkout.finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.trace


__all__ = ["agent_main"]
