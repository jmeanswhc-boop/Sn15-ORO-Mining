from __future__ import annotations
import json
import random
import re
import time
from functools import lru_cache
from typing import Any
from src.agent.proxy_client import ProxyClient
SMALL_BRAIN = 'anthropic/claude-haiku-5.5'
BIG_BRAIN = 'anthropic/claude-sonnet-5.5'
BACKUP_BRAIN = 'openai/gpt-6-luna'
BIG_BRAIN_KIN = (BIG_BRAIN, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
CIRCUIT_RINGS = 42
CALL_CEILING = 60
OUTLAY_MILD = 0.052
OUTLAY_FIRM = 0.078
BIG_BRAIN_LEEWAY = 0.02
BACKOFF_SET = (4, 10, 20, 35)
BACKOFF_TOTAL = 45
PRESENT_CHARS = 450
SIGHT_CUTOFF = 15000
CONVO_CUTOFF = 150000
TOLL_ROUNDS = 3
SELF_LEAVES = 6
THRIFT_UNSEALS = 3
ROUND_SECONDS = 300
ROAM_SPARE = 50
CALL_SPARE = 15
SCARCE_MISSIVES = 2
REBOUND_PACES = 8
CROWDED_WAIT = 20
SCAN_PACES = 6
_model_client = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_realm = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
ORIENTATION = '''\
Your name is Orla. You shop for one customer inside an online store, and the only way to do anything there is to call the store's own tools. Each run is scored all or nothing: it scores only when it closes in the correct state. That state is either an order holding every requested item, in the requested quantity, where each line satisfies every requirement still standing at checkout and the total stays within budget; or, when the store really carries nothing that satisfies everything, a no-purchase ending done the way the request describes.

THE STORE AND ITS TOOLS
- One tool call uses one step and the steps run out; reasoning is free, but every reply of yours takes several seconds and the run also has a time limit. When many steps are left (more than about 40), send calls that do not depend on each other together in one reply (several opens at once, or searches for several items at once). Steps are spent all the same: open a listing only when its row (title, overview, price) leaves it a real candidate, and when steps are tight open just the likeliest few, cheapest first. This store names its tools, arguments and result fields in its own way: study each tool's description and schema before use, and send arguments with exactly the names and permitted values given there.
- Searching may be cramped: a filter by category, a keyword mode that might look at only the opening words, pages of just a few rows, a token or an offset for moving on. Whenever the answer hangs on seeing every match, set the total the store reports against the rows already seen and fetch further pages, handing back the token or offset just as the store gave it. Short keyword queries that lead with the wanted value and then name the product work best. When a filtered search comes back empty, the filter's value or the filter argument is usually off: try again with the label spelled as result rows spell it, or with a different filter argument.
- A search row is only a digest. Stores divide a listing's record in different ways: one view may give its attributes (its properties), another only the offer and a description, and a view that sets several listings side by side may give either. Once Orla has worked out which tool shows the attributes, the ledger says so. Every requirement about a property is judged from those attributes, so read them for any listing you might buy; a single call covering several listings saves steps when it carries what you need. Do not open a listing a second time when nothing about it has changed.
- Orla's interviewer looks after every message to the shopper; you never write to the shopper yourself. The replies show up in this conversation and are copied into the ledger.
- The request may also name things to do along the way: an approval to obtain, a message to send, something to check or read first. They appear in the ledger. Those due right before the order or the no-purchase ending are carried out by Orla at the moment you call checkout or the closing tool, so do not hold back for them and do not decline on their account; do the others yourself with the store's tools (anything about a product's properties goes through Orla's interviewer).

DECIDING WHETHER A LISTING FITS
- What kind of product a listing is comes from its category label: only a label naming the requested item as a whole will do, never a nearby one.
- To pass a property requirement, the listing needs an attribute for that property stating the value. Key spelling is irrelevant (spaces, underscores, camel case, prefixes alike), and one entry of a list split by commas or slashes is enough. Values are compared word by word: letter case, singular versus plural and plain synonyms make no difference, and a value holding all the words of the wanted one passes ("pale mint" passes for mint, "very soft grip" passes for soft grip). Other words do not pass (a code or an abbreviation stands for a word without being it, and "mint blue" is not "ice blue"). Numbers go into one unit before comparing, a limit includes its edge, and a range has to sit completely inside a limit.
- When an attribute's key names the requirement's own property (under that name or an ordinary synonym), that attribute is decisive: should it give some other value, the listing fails whatever its title or other keys say. A key about some other property, even a close relative (the lining's fabric is not the shell's fabric), decides nothing about that requirement. Under each opened listing Orla adds [evidence] lines that quote any key named like a requirement.
- Lacking any attribute that names the property, the wanted value written under a different attribute (say, a list of styles or features containing it) or in its title or description is enough. A figure of another kind will not do (a length is not a volume), and neither will a bare number whose text never says what it measures.
- Where the category label already carries a requirement's value (a season or an audience written into the label), the category itself satisfies that requirement.
- Instead of a value, a requirement may refer to some other listing ("matching the X of the item <shop> offers with <property>", "differing by at most N from the X listed on ..."). Settle it before anything else: track down that listing (searching on its category, brand, shop or wording), open its full page with the single-listing or side-by-side tool (a search row will not do), and should that page send you on to a further listing for the property, open that one as well; the last page in the chain supplies the value. A listing's shop and its descriptive text may show only in its offer view, so open the view that shows them. To find the listing a requirement points at, search the shop's name together with the kind of product and open the offer views of that shop's rows (several in one reply): their description restates each listing's values and carries any pointer, so reading the attributes view is not needed for that. When the requirement wants the "partner piece" of that value, it wants the value the last page pairs it with (a page saying its checkered pattern goes together with stripes makes stripes the wanted value; a pairing rule printed on the page works the same way on the value that listing has), never the named value itself; "identical" or "the very same" wants that value unchanged. A remark that another version of a listing exists elsewhere is not a pointer. Put nothing in the cart for such an item until its value is settled: the first listing carted must already fit. The order is valid only once every page along that chain has been opened. If the requirement asks for something other than that listing's value, only listings with precisely that value are ruled out, and on its own it is never grounds for declining.
- If a listing you mean to buy sends you to some other page for more about itself, read that page before buying.
- When one requested item has to suit another (an accessory for a device), choose the pair from what the listings themselves say (a list of what it suits, a model or series code printed on both pages). Should no accessory meeting its own requirements suit the device you picked, move to a different device that both meets its own requirements and is suited by such an accessory; do not decline while that move remains untried.
- Honour exclusions and closed lists. Wishes ("ideally", "it would be nice") only break ties.
- Never buy anything the store marks as sold out; the order total (each line's price times its quantity, added up) must not go over the budget, though matching it is fine.

PICKING AMONG FITS
- When the request wants the cheapest (lowest total, lowest price), the order has to hold the lowest-priced listing out of ALL that fit. Gather the whole fitting set (page through it: rows rarely arrive sorted by price, and Orla fetches further pages of a short result on its own), test candidates from the lowest price upwards, and take the first one that fits in full. If a search lets you set a top price, set it under the price of the best pick you hold, so that the category lists only cheaper candidates for you to test.
- When the request is after the highest or lowest value of some measure, rank all the fits by that measure and choose the extreme one; a listing that never states the measure cannot be chosen.
- Otherwise any listing meeting everything is correct: do not spend steps searching for a better one.
- Cart an item the moment you have opened a listing that satisfies all of its requirements (for a cheapest, most or least pick, once the fitting set has been compared), then go on to the next. Lines often sell out shortly after being carted, and carting early leaves steps to replace one.
- Different requested items each need a cart line of their own, every line at its requested quantity.
- While comparing, keep the runner-up fit of each item in mind: a carted line often sells out at checkout, and its replacement is then the next fit by the pick rule, which saves searching again.

WHEN THINGS SHIFT
- The shopper or the store may alter the task: a fresh budget, a requirement added or removed, a new price, a listing sold out. Such news turns up inside tool results, very often as a refusal at checkout; a note counts whichever tool returned it, and the ledger shows the requirements now in force. A refusal does not end the run: apply every change, repair the cart and check out once more.
- Read a chosen listing's record again shortly before checkout, as price and stock can shift without warning.

CLOSING THE RUN
- Leave exactly the chosen lines in the cart (take out anything else), look at the cart one time, and then check out with a call by itself.
- Decline only after genuine searching turned up no listing that meets everything. Being low on steps proves nothing of the sort: should that happen, buy whatever is the best cart you have put together. Word that ending the way the request asks; any remark it calls for stays short and mentions no listing.
- Every reply must include at least one tool call. Keep the text brief: the facts you hold, then what you will do.'''
TASK_DIGEST = '''\
Set up a shopping job for an agent. Study the request below along with the tools this store offers, then reply with ONE compact JSON object (no indentation, nothing around it) carrying these keys:

"items": one entry for each separate product the shopper wants, each {"name": that kind of product in the request's wording, "shelf": its store category label when some tool lists labels (copied as written), else "", "quantity": how many units of that single listing (integer; default 1; a set or pair that is one listing counts as 1), "musts": the product's firm requirements, each "property: constraint" holding its precise value, bound, exclusion or permitted set. Call the property by the request's own word ("sized for a narrow alcove" gives "size: narrow alcove", "a 40-litre tank" gives "tank capacity: 40 litre") rather than a related property. When a value has to be read off another listing, write ONE must holding the full portrayal of that listing ("strap colour: other than the strap colour of the watch Belvora sells with a steel case"): the words that single out the other listing (its shop, its kind, and whatever "with <property> <value> and <property> <value>" follows it) belong inside that single must; none of them becomes a must of this product. Keep the request's relation word as it stands: "partner piece" stays "partner piece" (it asks for the value that pairs with that listing's, never the same one), and "identical", "different from", "within N of" and "no less than" stay unchanged}. Price, budget, stock and quantity are never musts. The background story (who will use it, where it goes, what failed last time, the event behind the purchase) is context unless it spells out a property's value or limit, and words that are part of the product's own name (a season or an audience named with it) are never musts either: they stay in "name". A use or occasion the request turns into a property of the product ("trainers for trail running", "a lamp meant for reading", "a party is the occasion it is for") counts as a must on that property ("use: trail running") unless the store's category label already carries it; an event, a person or a setting the purchase serves ("for her graduation day", "for a neighbour"), and how it should sit, feel or look while in use, are still context. Products listed together in a single phrase linked by "and" (a single "I want ..." with one quantity, frequently what a store department itself is called) are ONE item; split them only where each carries a separate quantity or separate requirements.
"budget": the ceiling on the whole spend, written as a number in whatever unit the request uses, or null.
"budget_unit": the name of that unit (say "cents"), or null.
"budget_scope": "total" where the limit covers the whole order, "each" where it applies per item, null where none is given.
"in_stock_only": true unless the request plainly accepts unavailable items.
"pick": "cheapest" where the request wants the lowest price or the smallest total, "most" or "least" where it wants the extreme of a measure, otherwise "any". Remarks that money is short are not a pick rule.
"pick_measure": the measure to push up or down for "most"/"least", otherwise "".
"messages": the cap on messages the agent may send the shopper (integer), or null when no number is given.
"steps": how many steps, moves or tool calls are allowed (integer), or null.
"lookups": every must (copied exactly) whose value has to be fetched from another listing the request names; leave none of them out; otherwise [].
"rules": each instruction the request gives on how to proceed or how to finish, at most 12 words apiece.
"wishes": anything the shopper would merely like.
"ask_first": what the request explicitly wants confirmed with the shopper before choosing (empty when nothing).
"process": the actions the request demands beyond finding and purchasing the right products (obtaining the shopper's approval, informing the shopper of something or presenting it to them, confirming something with them, reading something first, working through a named tool), each {"step": the action in 15 words or fewer, "when": "choose" (before products are chosen), "order" (right before checkout) or "end" (right before a no-purchase ending), "messages": the messages to the shopper it costs (0 when it costs none)}; [] when no such action is asked for. Item requirements, the budget, stock, the message and step caps, and what to do when nothing fits are not actions.
"quiet_decline": true when the request bars offering or naming substitutes should nothing fit, false when it wants them offered, null when it says neither.
"fits": a list of {"item": the dependent item's number, "with": the number of the item it must suit} for each requested item that, per the request, has to work with, fit or suit ANOTHER requested item (whichever one the agent picks); [] when there is none.
"browse": the way one tool lists a single store category, where such a tool exists: {"tool": the tool's exact name, "category": its argument for the category label, "max_price": its argument for a maximum price, or null, "fixed": its remaining arguments that select that mode, else {}}; null if the store has no tool that lists a category.
"roles": for each job below, the exact name of the tool doing it, or null when none does: "search", "open" (displays one listing's attributes or properties; if two tools each show one listing, the one whose description mentions attributes, properties or specifications), "offer" (displays one listing's live offer, like price and stock, if that tool differs from "open"), "compare" (several listings at once, in full), "ask" (sends the shopper a message), "cart" (alters the cart), "cart_view" (displays the cart), "buy" (performs the checkout), "refuse" (closes with no purchase).
"buy_args": the fixed arguments under which the "buy" tool performs the checkout (the mode value, for a tool with a mode argument), else {}.
"refuse_args": the fixed arguments under which the "refuse" tool closes with no purchase, else {}.
'''
QUIZ_BLUEPRINT = '''\
Pick the questions a shopping agent will put to its shopper. A shopper keeps a few firm wishes that never made it into the request, each tied to one property of one requested product. A wish surfaces only when a question names its exact property; a property the shopper is indifferent to draws "don't mind", and one the shopper never considers draws a baffled reply, so each property named uses up part of a message.

What you receive: the request, the items with whatever the request already fixes for each, and a numbered table of attribute keys found on the opened listings of every item's category, noting on how many of those listings each key appears plus some sample values. A hidden wish may concern a property that no key in the table names.

Order the properties like this:
0. Anything the request itself says to confirm with the shopper (under ASK FIRST when that section is present), each expressed as the property concerned.
1. Properties the story hints at. The story in the request often gestures at a hidden wish without stating its value: a line about something the shopper cannot judge or hands over to you, somebody whose taste decides, how the product looks or the place it is meant for, who is going to use or wear it, or what goes inside it. For every such line, name the single property it points to and the item it is about, keeping the story's order.
Give a hinted property the name of a listed key of that item when one suits the hint (a hint about the wearer: the key for gender or audience; about appearance: the listed key for colour, pattern or finish found on most of the item's listings); a property absent from all the item's listings is placed after the item's three most frequent keys.
2. Next come the listed keys, best first: favour what a buyer of that sort of product settles on early (its type or style, how it is shaped or fits, its size, volume or measurements, what it is made of, its colour or pattern, whichever of these that sort of product is chosen by) and keys present on most of the item's listings; after five or more listings have been read, avoid keys that turned up on just one or two.
Skip: a property already pinned down by the request for that item (a requirement names it; a key whose values only echo a required value, such as a finish key displaying the required coating, is NOT fixed), another name for a property already chosen, a key where every value lists several purchase options together, identifiers, model names, warranty and any other logistics. Pack sizes, piece totals and quantities per pack are ordinary properties, and a key marked [not the stated ...] is a property separate from that requirement even if the two names share words. Order each item by itself: hinted properties first, its listed keys after.
"label" is the plain name of the property: for a key from the table, every word of that key spelled out in ordinary form ("strap width" for strapWidth, "sole material" for sole_material; never cut down to "width" or "material"); just the name, with no sample values, brackets or extra property.
Choose up to COUNT per item, best first. Reply with ONE JSON object and nothing more: {"ask": [{"item": the item's number, "label": "...", "hint": "no more than 8 words of the story behind it, or an empty string"}]}, keeping every item's choices in order'''
REPLY_SPEC = '''\
Questions went from a shopping agent to its shopper; the replies follow. Read every reply and return ONE JSON object, nothing else:
{"needs": [{"item": number of the item concerned, "need": "property: constraint", "points": true when the value must be looked up on some other listing, false when it need not}], "drop": [{"item": number of the item concerned, "property": "a requirement the shopper abandons"}], "free": [{"item": number of the item concerned, "property": "..."}], "crowded": [numbers of those questions whose reply protests that one message covered too many things, or asks to be asked one thing at a time; leave out replies that just say they were lost], "budget": a new limit for the whole order as a number, or null}
- "needs": every firm wish stated in a reply, carrying its precise value or limit and the unit ("no wider than 60 cm", "round", "anything but plastic"). Use the property named in the reply itself as the need's name, even where a different one was asked about; replies about one property together form one need. Take the value over word for word, keeping its nouns and units ("a minimum of 4 drawers", never "a minimum of 4").
- Sometimes a reply supplies the value by pointing to some other listing (a value equal to, different from or close to the one on a product a shop sells, the product identified through some of its properties): record ONE need on the property the reply mentions, with "points": true, and as its value the reply's wording from the comparing words to the end of that description, untouched, including whatever it says about a listing that page may mention. Properties used to describe that other product belong to it, not to this item.
- "free": for each question the shopper said they have no preference on, its property.
- A value said in a chatty or indirect way is still a value.
- A value is just what the property ought to be. Linking words between a property and its value (matching, set to, coming in as and the like) do not belong to it: "colour set to navy" states navy. A value that points at another listing is exempt and stays whole.'''
JOT_SPEC = '''\
Something turned up in a tool result while a shopping agent was busy; it may be from the shopper or from the store. Answer with ONE JSON object only:
{"kind": "change" if the note alters the job (the budget, the requirements, a price, what is available, a refused order), "error" if all it reports is a malformed or rejected call, "other" in every remaining case, "budget": the overall limit it sets, as a number in the request's unit, else null, "add": [{"item": which item (its number), "need": "property: constraint"}], "drop": [{"item": which item (its number), "need": "a requirement the shopper no longer wants"}], "gone": [ids of the listings it calls unavailable], "sold_out": true if it reports a picked listing (whether named or not) as gone, sold out or no longer obtainable, "price_moved": true when it says a price moved, "summary": "a single short sentence"}
Where one requirement takes another's place, the earlier one goes in drop and its successor in add. Every change the note makes must be reported.'''
CART_PROBE = '''\
Your job is the last look at a cart before the shopping agent pays. From the records supplied, decide whether the cart is exactly right.
Rules for reading records: a requirement is met when the listing has an attribute on that property showing the value (how keys are spelled does not matter; one entry of a list is enough; case, singular or plural and plain synonyms make no difference; a value holding all the required value's words passes, "pale mint" for mint; units converted; a limit includes its edge; a range must sit completely inside); an attribute keyed by the requirement's own property decides, so a different value there fails the listing; a key about another property, even a close relative (the lining's fabric is not the shell's fabric), decides nothing; with no attribute naming that property, the wanted value written under a different attribute or in its title or description passes (a figure of another kind will not do, a length is not a volume, and neither will a bare number whose text never says what it measures); a requirement already written in the category label is met by the category; the category label must name the requested item as a whole; a demand to be unlike some other listing's value holds unless the records show precisely that value. Each requested item must be a cart line of its own at the requested quantity, the cart must hold nothing more, no line may be shown as unavailable, and the lines' prices multiplied by their quantities must add up to no more than the budget. The latest record of a listing is the one that counts.
For a cheapest, most or least pick, also weigh the other listings shown: should one plainly meet every requirement and beat the cart's choice on that pick, the cart is wrong.
Reply with ONE JSON object and nothing more: {"ok": true or false, "problems": ["a brief, specific fault", ...], "better": [ids of listings that should take a cart line's place, best first]}. Say ok=false only for a problem the records plainly show, or for a requirement that no record speaks to.'''
MATCH_READING = '''\
a property is settled by an attribute whose key is that property's own name (key spelling is irrelevant; one entry of a list is enough; synonyms and converted units are fine; a value holding all the required value's words passes, "pale mint" for mint; a limit includes its edge); a key about another property, even a close relative (the lining's fabric is not the shell's fabric), decides nothing; when no attribute names the property, the wanted value written under a different attribute or in the title or description passes (a bare number whose text never says what it measures passes for nothing); a requirement already written in the category label is met by the category; the category label must name the item; where the value must differ from what some other listing has, it fails only if the records show that exact value; a listing out of stock or over the budget is not a fit.'''
DECLINE_REVIEW = '''\
A shopping agent means to finish without buying, believing that no listing fits. Hold the records shown up against the requirements, with these reading rules:''' + MATCH_READING + '''
Listings tagged ROW have only appeared as rows of a search result: nobody has read their page (with the attributes), so a property the row does not show can neither pass nor fail them.
The requirements were jotted down in a hurry, so first hold each one up against the REQUEST and the ANSWERS FROM THE SHOPPER. One that only restates the story (for whom, when, where or why the product gets used, an earlier mishap, the feel wanted) and is never demanded of the product by the request belongs in "not_firm". When a value carries extra words that merely tie the property to it goes under "corrected", stated plainly. When judging the listings, leave the not_firm requirements out and use the corrected values. Anything the request or an answer states outright remains in force even if no listing meets it; then stopping with no purchase is right.
Reply with ONE JSON object and nothing more: {"not_firm": [{"item": which item, by number, "need": the requirement copied exactly}], "corrected": [{"item": which item, by number, "need": the requirement word for word, "value": "property: the bare value"}], "fits": {"item number": [up to 4 ids of listings apparently meeting every requirement, lowest price first]}, "open": [ROW listing ids of a requested kind and inside the budget whose page could prove a fit, likeliest first, 6 at most], "search": [up to 4 entries of {"item": which item, by number, "query": "two or three keywords, not yet searched, that could bring up a fitting listing: another word for the product plus one required value (long queries tend to match nothing)"}], "reason": "one sentence"}'''
UNTANGLE_SPEC = '''\
The request below was turned into a set of firm requirements for one product. A few requirements get their value from another listing that the request mentions, and the words picking out that other listing (what it is, which shop sells it, the properties following "with ...") are now and then taken by mistake for requirements on the product to be bought. Judging from the request's wording, list which of the OTHER REQUIREMENTS below merely describe that other listing. Answer with ONE JSON object only: {"describe_other": [such requirements copied exactly; empty when all of them truly apply to the product being bought]}'''
ROUTE_SPEC = '''\
Checkout is the agent's next move. Some requirements take their value from a different listing instead of giving one, and the page of a listing may pass the reader along to yet another listing for that property. Until each listing along every such chain has had its full page opened, the order is not valid. Walk each requirement's chain beginning with the requirement's own wording, one link after the next, through what the pages themselves say, using the records below (OPENED: a page the agent has opened; ROW: met only as a search result row, page unread).
Reply with ONE JSON object and nothing more: {"chains": [{"need": the requirement, copied exactly from the list, "links": [{"id": the id of the record below that is this link, else "", "about": how the requirement or the page before portrays the listing, "find": "for a link still missing: search words, namely its shop plus the values in the portrayal"}], "done": true if the chain ends in an OPENED page giving the value of the property directly (this holds too when the requirement asks for something different from that value, which then serves as the yardstick), else false, "value": the value the requirement asks for, worked out from the last page: for "identical" or "same", the value that page gives; for "partner piece", the value that page pairs it with (its own words such as "goes together with", or a pairing rule printed on it applied to the value that listing has); for "different" or "not", "not " plus that value; for "within", "at least" or "no less than", the resulting limit; or ""}]}
Count a record as a link only where it agrees with the link's entire portrayal: the shop or brand, the product kind AND every property value named (for example "a 6 bottles multipack" rules out the same shop's listing holding 6 sachets and no multipack); where no record agrees with all of it the link is still missing, however many listings of that kind the shop sells. Every missing link needs its "find" words: the shop's name together with the portrayal's values.
Only wording that sends the requirement's property to another listing ("take it from", "refer to", "look to", "go by", "check") makes a link; a remark that another version exists elsewhere does not. Give the links of each chain in order, and stop as soon as a link is unfound or its page is not OPENED, since what it says is not known yet.'''
TOLL_SPEC = '''\
A shopping agent is about to STAGE. On the way there the shopper's request calls for the steps below. Using the RECORD (numbered lines of the agent's actions up to now), settle for every step that falls due before this STAGE whether it is complete for the cart as it stands. A step is complete only where some RECORD line shows it being carried out and, when the step touches the order or the cart, that line has to follow the newest cart change and contain a reply that agrees; questions about product properties never complete a step.
For a step still open, name the one store call that should carry it out next: a tool listed under TOOLS plus its arguments. Messages to the shopper use the messaging tool and hold a single short sentence or question about that step and nothing else, in ordinary words, mentioning no product property or value the step can do without. Name no call if all due steps are complete, or if the messages or steps still available are too few for the open one.
Answer with ONE JSON object only: {"steps": [{"step": that step, worded as above, "done": true or false, "lines": [which RECORD lines show it done]}], "call": {"tool": "...", "args": {...}} or null when no call is due, "hold": "anything a reply asks to change before going ahead, else empty", "needs": [{"item": which item (its number), "need": "property: constraint"} for each firm requirement that a reply of that kind states]}'''
SUBSTITUTE_SPEC = '''\
One of the lines in a shopping agent's cart has just gone out of stock. Among the remaining listing records shown, look for listings able to stand in for it, using these reading rules:''' + MATCH_READING + '''
Answer with ONE JSON object only: {"fits": [{"id": the listing's id, "price": the price, a number copied from the record} for each listing that seems to meet all requirements on the sold-out line's item, no more than 6], "reason": "one sentence"}'''
FORM_SPEC = '''\
Below is what a store tool sent back. Name its parts with the keys exactly as they appear in the output or among the tool's parameters. Answer with ONE JSON object only: {"rows": key under which the listings arrive, "count": key for how many listings match overall (rather than how many are shown), or null, "next": key with the value that loads the page after this one, or null, "next_arg": which parameter of the tool accepts that value, or null, "price": row key with the listing's price, or null, "stock": row key showing whether the listing can be bought right now, or null}'''
SENTRY_KEYS = 3
SENTRY_SHARE = 0.0
SENTRY_GAP = 0.34
NON_TRAIT = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
GRAB_BAG = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
BROADEN_EACH = 6
MATCH_MARK = '[A-Z][A-Za-z]{0,7}(?:[- ]?\\d{3,6}|[- ]?\\d{1,5}[xX]{1,4})'
PIECE_MARK = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
PIECE_RANGE = '[A-Z]{1,5}\\d{1,3}'
MATCH_TERMS = '(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|supports?)'
MATCH_SAVE = '\\b(?:except(?:\\s+for)?|excluding|other than|but not|apart from|not including|not for)\\b'

class _TextKit:

    @staticmethod
    def _title_names(trait: frozenset[str], attr_key: frozenset[str]) -> bool:
        if not trait or not attr_key:
            return False
        return attr_key == trait or attr_key >= trait or (len(attr_key) >= 2 and trait >= attr_key)

    @staticmethod
    def _bases(wording: str) -> set[str]:
        outside = set()
        for ww in _TextKit._terms(wording).split():
            ww = (ww if not (len(ww) > 3 and ww.endswith('s') and (not ww.endswith('ss'))) else ww[:-1]) if not re.search('(?:ch|sh|ss|x|z)es$', ww) else ww[:-2]
            if len(ww) >= 3 and ww not in ('and', 'for', 'the', 'with'):
                outside.add(ww)
        return outside

    @staticmethod
    def _stalk(wording: str) -> frozenset[str]:
        return frozenset((ww if not (len(ww) > 3 and ww.endswith('s')) else ww[:-1] for ww in _TextKit._terms(wording).split()))

    @staticmethod
    def _terms(attr_key: str) -> str:
        attr_key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(attr_key))
        return re.sub('[\\s_.\\-]+', ' ', attr_key).strip().lower()

    @staticmethod
    def _json_thing(wording: Any) -> dict[str, Any] | None:
        if not isinstance(wording, str):
            return None
        wording = wording.strip()
        barrier = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', wording, re.S)
        if barrier:
            wording = barrier.group(1)
        try:
            fetched = json.loads(wording)
            return None if not isinstance(fetched, dict) else fetched
        except ValueError:
            pass
        (remaining, correct) = (wording.find('{'), wording.rfind('}'))
        if 0 <= remaining < correct:
            try:
                fetched = json.loads(wording[remaining:correct + 1])
                return None if not isinstance(fetched, dict) else fetched
            except ValueError:
                return None
        return None

    @staticmethod
    def _snip(val: Any, cutoff: int, nesting: int=0) -> Any:
        if 7 < nesting:
            return '...'
        if isinstance(val, str):
            return val[:cutoff] + '...' if not len(val) <= cutoff else val
        if isinstance(val, dict):
            return {kw: _TextKit._snip(vw, cutoff, nesting + 1) for (kw, vw) in val.items()}
        if isinstance(val, list):
            return [_TextKit._snip(vw, cutoff, nesting + 1) for vw in val]
        return val

    @staticmethod
    def _render(val: Any) -> str:
        try:
            return json.dumps(val, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(val)

class _SchemaKit:

    @staticmethod
    def _number_in(val: Any) -> int | None:
        if isinstance(val, bool):
            return None
        if isinstance(val, int):
            return val
        if isinstance(val, float) and val.is_integer():
            return int(val)
        if isinstance(val, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', val):
            return int(float(val))
        return None

    @staticmethod
    def _as_list(val: Any) -> list[Any]:
        if isinstance(val, list):
            return val
        if isinstance(val, (str, dict)) and val:
            return [val]
        return []

    @staticmethod
    def _form(val: Any, layout: Any, wherever: str, flaws: list[str]) -> Any:
        if not isinstance(layout, dict):
            return val
        types_of = _SchemaKit._types_of(layout)
        if types_of and (not any((_SchemaKit._matches_up_type_of(val, kw) for kw in types_of))):
            for type_of in types_of:
                if 'integer' == type_of and isinstance(val, float) and val.is_integer():
                    val = int(val)
                    break
                if type_of in ('integer', 'number') and isinstance(val, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val):
                    digit = float(val)
                    val = digit if not ('integer' == type_of or digit.is_integer()) else int(digit)
                    break
                if 'string' == type_of and isinstance(val, (int, float)) and (not isinstance(val, bool)):
                    val = str(val)
                    break
                if 'boolean' == type_of and isinstance(val, str) and (val.strip().lower() in ('true', 'false')):
                    val = val.strip().lower() == 'true'
                    break
                if 'array' == type_of and (not isinstance(val, list)) and (val is not None):
                    val = [val]
                    break
            if not any((_SchemaKit._matches_up_type_of(val, kw) for kw in types_of)):
                flaws.append(f"{wherever} should be {'/'.join(types_of)}")
                return val
        choices = layout.get('enum')
        if isinstance(choices, list) and val not in choices:
            if isinstance(val, str):
                short_of = val.strip().lower()
                match = [ow for ow in choices if isinstance(ow, str) and ow.lower() == short_of]
                if match:
                    val = match[0]
            if val not in choices:
                flaws.append(f'{wherever}: allowed values are {choices}')
        if isinstance(val, str) and isinstance(layout.get('maxLength'), int) and (len(val) > layout['maxLength']):
            val = val[:layout['maxLength']]
        if isinstance(val, str) and isinstance(layout.get('minLength'), int) and (len(val) < layout['minLength']):
            flaws.append(f'{wherever} is too short')
        if isinstance(val, (int, float)) and (not isinstance(val, bool)):
            if isinstance(layout.get('minimum'), (int, float)) and val < layout['minimum']:
                flaws.append(f"{wherever} is below {layout['minimum']}")
            if isinstance(layout.get('maximum'), (int, float)) and val > layout['maximum']:
                flaws.append(f"{wherever} is above {layout['maximum']}")
        if isinstance(val, list):
            if isinstance(layout.get('maxItems'), int) and len(val) > layout['maxItems']:
                flaws.append(f"{wherever} allows at most {layout['maxItems']} entries")
            if isinstance(layout.get('minItems'), int) and len(val) < layout['minItems']:
                flaws.append(f"{wherever} needs at least {layout['minItems']} entries")
            if isinstance(layout.get('items'), dict):
                val = [_SchemaKit._form(vw, layout['items'], f'{wherever}[{iw}]', flaws) for (iw, vw) in enumerate(val)]
        if isinstance(val, dict):
            traits = {} if not isinstance(layout.get('properties'), dict) else layout.get('properties')
            if layout.get('additionalProperties') is False:
                for title_name in [nw for nw in val if nw not in traits]:
                    flaws.append(f'{wherever}.{title_name} is not an allowed argument')
            for title_name in layout.get('required') or []:
                if title_name not in val:
                    flaws.append(f'{wherever}.{title_name} is required')
            val = {kw: vw if not kw in traits else _SchemaKit._form(vw, traits.get(kw), f'{wherever}.{kw}', flaws) for (kw, vw) in val.items()}
        return val

    @staticmethod
    def _matches_up_type_of(val: Any, type_of: str) -> bool:
        if 'string' == type_of:
            return isinstance(val, str)
        if 'integer' == type_of:
            return isinstance(val, int) and (not isinstance(val, bool))
        if 'number' == type_of:
            return isinstance(val, (int, float)) and (not isinstance(val, bool))
        if 'boolean' == type_of:
            return isinstance(val, bool)
        if 'array' == type_of:
            return isinstance(val, list)
        if 'object' == type_of:
            return isinstance(val, dict)
        if 'null' == type_of:
            return val is None
        return True

    @staticmethod
    def _types_of(layout: dict[str, Any]) -> list[str]:
        type_of = layout.get('type')
        if isinstance(type_of, str):
            return [type_of]
        return [kw for kw in type_of or [] if isinstance(kw, str)]

class _RowKit:

    @staticmethod
    def _attr_slab(unit: Any) -> dict[str, Any] | None:
        if not isinstance(unit, dict):
            return None
        finest = None
        for val in unit.values():
            if isinstance(val, dict) and len(val) >= 2:
                plain_vals = {kw: vw for (kw, vw) in val.items() if isinstance(vw, (str, int, float)) and (not isinstance(vw, bool))}
                if len(plain_vals) >= 2 and (finest is None or len(plain_vals) > len(finest)):
                    finest = plain_vals
        return finest

    @staticmethod
    def _inside_entry_rows(unit: Any, mark_id: str, nesting: int=0) -> bool:
        if 5 < nesting:
            return False
        if isinstance(unit, list):
            return any((isinstance(vw, dict) and mark_id in [str(xw).strip() for xw in vw.values() if isinstance(xw, str)] for vw in unit)) or any((_RowKit._inside_entry_rows(vw, mark_id, nesting + 1) for vw in unit))
        if isinstance(unit, dict):
            return any((_RowKit._inside_entry_rows(vw, mark_id, nesting + 1) for vw in unit.values()))
        return False

    @staticmethod
    def _study_code_id(entry_rows: list[dict[str, Any]]) -> str | None:
        finest = None
        for attr_key in entry_rows[0]:
            values = [entry_row.get(attr_key) for entry_row in entry_rows]
            if not all((_RowKit._code_id_resembling(vw, 'id' in str(attr_key).lower()) for vw in values)):
                continue
            values = [vw.strip() for vw in values]
            if len(entry_rows) > 1 and len(set(values)) < len(values):
                continue
            grade = ('id' in str(attr_key).lower(), sum((len(vw) for vw in values)) / len(values))
            if finest is None or grade > finest[0]:
                finest = (grade, attr_key)
        return None if not finest else finest[1]

    @staticmethod
    def _code_id_resembling(val: Any, titled: bool=False) -> bool:
        if not isinstance(val, str) or not 3 <= len(val.strip()) <= 64 or ' ' in val.strip():
            return False
        return titled or bool(re.search('\\d', val))

class _FitKit:

    @staticmethod
    def _match_up_ruling(piece: dict[str, Any], gadget: dict[str, Any], permitted: set[str]) -> bool | None:
        if gadget.get('series') and re.match(PIECE_RANGE + '-', piece.get('own') or ''):
            return any((piece['own'] == sw or piece['own'].startswith(sw + '-') for sw in gadget['series']))
        if not permitted or not gadget.get('codes'):
            return None
        clans = {_FitKit._mark_code_clan(cw) for cw in permitted}
        mark_codes = [cw for cw in gadget['codes'] if _FitKit._mark_code_clan(cw) in clans]
        if not mark_codes:
            return None
        if any((_FitKit._mark_code_title_names(dw, cw) for dw in piece.get('deny') or [] for cw in mark_codes)):
            return False
        return any((_FitKit._mark_code_title_names(aw, cw) for aw in permitted for cw in mark_codes))

    @staticmethod
    @lru_cache(maxsize=4096)
    def _match_up_ticks(wording: str) -> dict[str, Any]:
        owned = re.search('[\\[(]\\s*(?:part\\s+)?(' + PIECE_MARK + ')\\s*[\\])]', wording, re.I)
        mark_codes: list[str] = []
        for strike in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(' + MATCH_MARK + ')\\b', wording, re.I):
            mark_codes += _FitKit._mark_codes(strike.group(1))
        for strike in re.finditer('\\b(?:also sold as|also known as|sold also as|sold as)\\s+([^.;]+)', wording, re.I):
            mark_codes += _FitKit._mark_codes(strike.group(1))
        toils: list[str] = []
        withhold: list[str] = []
        for strike in re.finditer('\\b(?<!not )(?<!never )' + MATCH_TERMS + '\\s*:?\\s+([^.;]+)', wording, re.I):
            if re.search('\\bpart\\s+' + PIECE_MARK + '\\s+(?:also\\s+)?$', wording[max(0, strike.start() - 40):strike.start()], re.I) or re.match('part\\s+series', strike.group(1), re.I):
                continue
            flanks = re.split(MATCH_SAVE, strike.group(1), maxsplit=1, flags=re.I)
            toils += _FitKit._mark_codes(flanks[0])
            if len(flanks) > 1:
                withhold += _FitKit._mark_codes(flanks[1])
        too = [(strike.group(1).upper(), _FitKit._mark_codes(strike.group(2))) for strike in re.finditer('\\bpart\\s+(' + PIECE_MARK + ')\\s+also fits\\s*:?\\s*([^.;]+)', wording, re.I)]
        range_set: list[str] = []
        for strike in re.finditer('\\bfits?\\s+part\\s+series\\s*:?\\s*([^.;]+)', wording, re.I):
            range_set += [cw.upper() for cw in re.findall('\\b' + PIECE_RANGE + '\\b', strike.group(1))]
        return {'own': '' if not owned else owned.group(1).upper(), 'codes': mark_codes, 'works': toils, 'deny': withhold, 'also': too, 'series': range_set}

    @staticmethod
    def _mark_code_clan(mark_code: str) -> str:
        spotted = re.match('[a-z]+', mark_code)
        return '' if not spotted else spotted.group(0)

    @staticmethod
    def _mark_code_title_names(declared: str, mark_code: str) -> bool:
        rear = re.search('(?<=\\d)x+$', declared)
        if not rear:
            return mark_code == declared
        return mark_code.startswith(declared[:rear.start()]) and len(mark_code) == len(declared)

    @staticmethod
    def _mark_codes(wording: Any) -> list[str]:
        return [re.sub('[\\s-]+', '', cw).lower() for cw in re.findall('\\b' + MATCH_MARK + '\\b', str(wording or ''))]

class ShoppingLine:

    def __init__(self, title_name: str, rack: str, howmany: int, requisites: list[str]):
        self.title_name = title_name
        self.rack = rack
        self.howmany = howmany
        self.requisites = requisites
        self.relayed: list[str] = []
        self.shed_off: list[str] = []
        self.spare_one: list[str] = []

class _Turns:

    def launch_rings(self, rings: list[dict[str, Any]]) -> list[tuple[str, str]]:
        outside: list[tuple[str, str]] = []
        rings = self.merge_in_unseals(rings)
        for (index, ring) in enumerate(rings):
            permitted = {tw['function']['name'] for tw in self.quoted()}
            ring_id = ring['id']
            if ring.get('joined'):
                outside.append((ring_id, 'Shown in the side-by-side result above: Orla opened these listings together in one step.'))
                continue
            fun = ring.get('function') or {}
            title_name = fun.get('name')
            untouched = fun.get('arguments')
            if self.wrapped_up:
                outside.append((ring_id, 'Not run: the episode is over.'))
                continue
            if title_name == self.jobs.get('ask'):
                outside.append((ring_id, "Not run (no step used): Orla's interviewer handles every message to the shopper."))
                continue
            if title_name not in self.layouts:
                outside.append((ring_id, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(permitted))}."))
                continue
            if title_name not in permitted:
                outside.append((ring_id, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(permitted)) + '.'))
                continue
            if not isinstance(untouched, dict):
                argset = {} if not (untouched or '').strip() else _TextKit._json_thing(untouched)
                if argset is None:
                    outside.append((ring_id, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            else:
                argset = untouched
            flaws: list[str] = []
            argset = _SchemaKit._form(argset, self.layouts[title_name], 'args', flaws)
            if flaws:
                outside.append((ring_id, 'Not run (no step used): ' + '; '.join(flaws[:6])))
                continue
            (argset, off_mark) = self.untangle_code_ids(title_name, argset)
            if off_mark:
                outside.append((ring_id, 'Not run (no step used): ' + off_mark + '.'))
                continue
            shaved = ''
            numerous = None if not title_name == self.jobs.get('compare') else self.contrast_aim()
            if numerous and isinstance(argset.get(numerous), list) and (len(argset[numerous]) > self.contrast_leeway()) and self.contrast_poor:
                leeway = self.contrast_leeway()
                shaved = f"\n[Orla] Sent the first {leeway} listings only: this store shows at most {leeway} side by side. Not shown yet: {', '.join(map(str, argset[numerous][leeway:]))}."
                argset = {**argset, numerous: argset[numerous][:leeway]}
            aim = self.sole_aim(title_name)
            if aim and title_name in (self.jobs.get('open'), self.jobs.get('offer')) and (str(argset.get(aim)) in self.departed):
                outside.append((ring_id, f'Not run (no step used): {argset[aim]} sold out at step {self.departed[str(argset[aim])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if title_name == self.jobs.get('cart') and any(((_SchemaKit._number_in(vw) or 0) < 0 for vw in argset.values() if not isinstance(vw, str))):
                look = '' if not self.trolley_upon <= self.trolley_noticed_upon else _TextKit._render(self.trolley_look)
                lacking = [vw for vw in argset.values() if isinstance(vw, str) and vw in self.entry_rows and (vw not in self.trolley_row_lines) and (vw not in look) and (vw not in self.trolley_stated_out)]
                if lacking:
                    outside.append((ring_id, f'Not run (no step used): {lacking[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.trolley_wording()))
                    continue
            closeout = self.be_purchase(title_name, argset) or self.be_turn_down(title_name, argset)
            if closeout and len(rings) > 1:
                outside.append((ring_id, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not closeout and self.remaining() <= self.rear() and (self.jobs.get('buy') or self.jobs.get('refuse')):
                outside.append((ring_id, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            tick = title_name + _TextKit._render(argset)
            if not closeout and title_name != self.jobs.get('cart') and (tick in self.latest):
                outside.append((ring_id, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            perusers = {self.jobs.get(rw) for rw in ('search', 'open', 'offer', 'compare')} - {None}
            stashed = self.peruse_stash.get(tick)
            if title_name in perusers and stashed and (stashed[0] == self.edition):
                outside.append((ring_id, f'Not run (no step used): the same call at step {stashed[1]} returned this, and nothing has changed since:\n' + stashed[2][:3000]))
                continue
            if aim and title_name in perusers and (title_name != self.jobs.get('compare')) and self.presented_out_so_far(title_name, str(argset.get(aim))):
                outside.append((ring_id, f'Not run (no step used): {argset[aim]} has shown everything this view gives since the last change. Its record: ' + self.file_rec_from_of(str(argset[aim]))))
                continue
            if numerous and isinstance(argset.get(numerous), list) and argset[numerous]:
                retain = [iw for iw in argset[numerous] if not self.presented_out_so_far(title_name, str(iw))]
                if not retain:
                    outside.append((ring_id, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{iw}: {self.file_rec_from_of(str(iw))}' for iw in argset[numerous]))[:4000]))
                    continue
                if len(retain) < len(argset[numerous]):
                    shaved += f"\n[Orla] Left out {', '.join((str(iw) for iw in argset[numerous] if iw not in retain))}: shown in full already, nothing changed since."
                    argset = {**argset, numerous: retain}
                    tick = title_name + _TextKit._render(argset)
            if aim and title_name in (self.jobs.get('open'), self.jobs.get('offer')) and (3 > self.sequence_turnstiles) and (self.remaining() > self.rear() + 4):
                mark_id = str(argset.get(aim))
                cls_attr_key = self.genre_attr_key()
                tag_r = '' if not cls_attr_key else str((self.entry_rows.get(mark_id) or {}).get(cls_attr_key) or '')
                digit = 0 if not tag_r else self.holder_from_of(tag_r)
                ongoing = {needful for (_, needful) in self.needfuls_at_present()}
                open_ended = [] if not digit else [needful for (nw, needful) in self.cites if digit == nw and needful in ongoing and (needful not in self.cite_vals)]
                holder = {str(vw).strip() for (kw, vw) in (self.entry_rows.get(mark_id) or {}).items() if isinstance(vw, str) and re.search('seller|shop|store|merchant|vendor|brand', str(kw), re.I)}
                outlets = set() if not open_ended else self.sequence_outlets(digit)
                if open_ended and (not holder & outlets) and (6 > self.routes) and (self.timing_remaining() > CALL_SPARE + 40):
                    self.route()
                    open_ended = [needful for needful in open_ended if needful not in self.cite_vals]
                if open_ended and (not holder & outlets):
                    self.sequence_turnstiles += 1
                    outside.append((ring_id, f'Not run (no step used): {mark_id} would be a candidate for item {digit}, but its requirement "{open_ended[0]}" takes its value from another listing and that value is not settled, so no candidate can be judged yet. Settle it first: search for the listing it points at by its shop and kind' + ('' if not outlets else f" (shops of this chain so far: {', '.join(sorted(outlets))})") + ', open the offer view of the one whose description matches, and follow every listing that page sends you to.'))
                    continue
            if title_name == self.jobs.get('cart') and (not closeout) and (2 > self.sequence_keeps_hold) and (self.remaining() > self.rear() + 6) and (self.timing_remaining() > CALL_SPARE + 40) and any(((_SchemaKit._number_in(vw) or 0) > 0 for vw in argset.values() if not isinstance(vw, str))):
                ongoing = {needful for (_, needful) in self.needfuls_at_present()}
                if any((needful in ongoing and needful not in self.cite_vals for (_, needful) in self.cites)):
                    self.sequence_keeps_hold += 1
                    (sequence, keep_hold) = self.route()
                    fixed_up = '; '.join((f'{kw} -> {vw}' for (kw, vw) in self.cite_vals.items()))
                    outside.append((ring_id, 'Not run yet (no step used): a requirement takes its value from the page of another listing, and the first listing put in the cart must already meet it. ' + (sequence or '') + ('Find and open that page first.' if not fixed_up else 'Settled values: ' + fixed_up + '. Check this pick against them, then put the fitting listing in the cart.') + '\nCart now: ' + self.trolley_wording()))
                    continue
            if title_name == self.jobs.get('cart') and (not closeout) and (self.digest.get('pick') == 'cheapest') and (self.remaining() > self.rear() + 8) and (self.timing_remaining() > ROAM_SPARE + 30) and any(((_SchemaKit._number_in(vw) or 0) > 0 for vw in argset.values() if not isinstance(vw, str))):
                choose = next((vw for vw in argset.values() if isinstance(vw, str) and vw in self.entry_rows), None)
                cls_attr_key = self.genre_attr_key()
                tag_r = '' if not (choose and cls_attr_key) else str((self.entry_rows.get(choose) or {}).get(cls_attr_key) or '')
                digit = (0 if not len(self.items) == 1 else 1) if not tag_r else self.holder_from_of(tag_r)
                fee = None if not choose else self.fee_from_of(choose)
                kept_hold = [iw for iw in self.trolley_code_ids() if cls_attr_key and self.holder_from_of(str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or '')) == digit]
                if digit and fee is not None and (digit not in self.initial_probed) and (choose not in self.trolley_code_ids()) and (not kept_hold):
                    self.initial_probed.add(digit)
                    spotted = self.series_genre(digit, fee * 0.9 if not 1 < fee else int(fee) - 1, 1)
                    thriftier = [iw for iw in self.type_of_entry_rows(digit, spotted or [], fee) if choose != iw and iw not in self.whole_rec][:3]
                    unsealed = [] if not thriftier else self.pad_attrs(thriftier, self.rear() + 6)
                    if unsealed:
                        outside.append((ring_id, 'Not run yet (no step used): the pick is the cheapest, and the first listing put in the cart for an item must already be the right one. Orla listed that category below the price of ' + choose + ' and read its cheapest rows:\n' + '\n'.join(unsealed) + '\nPut in the cart the cheapest listing that meets every requirement (' + choose + ' if none of these does).'))
                        continue
            if closeout and self.remaining() > 2 and (self.timing_remaining() > CALL_SPARE + 25):
                if self.be_purchase(title_name, argset):
                    broader = self.broader_keep_hold()
                    if broader:
                        outside.append((ring_id, broader + '\nCart now: ' + self.trolley_wording()))
                        continue
                    reperuse = self.reperuse_quotes()
                    if self.jot_soldout:
                        self.seek_soldout()
                    if self.trolley_handled and self.trolley_upon > self.trolley_noticed_upon and self.jobs.get('cart_view') and (self.remaining() > 3):
                        flaws = []
                        look_argset = _SchemaKit._form({}, self.layouts.get(self.jobs['cart_view']), 'args', flaws)
                        look = None if flaws else self.do_step(self.jobs['cart_view'], look_argset)
                        if look is not None:
                            self.trolley_look = look.get('observation')
                            self.trolley_noticed_upon = self.pace
                            shift = self.notice(self.jobs['cart_view'], self.trolley_look)
                            if shift:
                                self.lined_up = []
                                outside.append((ring_id, 'Not run yet (no step used). Looking at the cart showed a change: ' + shift + '\nCart now: ' + self.trolley_wording() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    switch = self.switch_cue()
                    kept_hold = [] if not 2 > self.kept_hold else [iw for iw in self.trolley_code_ids() if iw in self.departed]
                    if switch or kept_hold:
                        self.kept_hold += 1 if not switch else 0
                        outside.append((ring_id, 'Not run yet (no step used). ' + (switch or f"{', '.join(kept_hold)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.trolley_wording()))
                        continue
                    (sequence, keep_hold) = self.route()
                    if keep_hold:
                        outside.append((ring_id, 'Not run yet (no step used). ' + sequence + '\nCart now: ' + self.trolley_wording()))
                        continue
                    mismatch = [] if not (2 > self.match_up_keeps_hold and self.remaining() > self.rear() + 3) else self.match_up_flaws()
                    if mismatch:
                        self.match_up_keeps_hold += 1
                        self.record_path.append({'kind': 'fit_hold', 'faults': mismatch[:4]})
                        outside.append((ring_id, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(mismatch[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.trolley_wording()))
                        continue
                    flaws = [] if not (2 > self.row_line_keeps_hold and self.remaining() > self.rear() + 2) else self.row_line_flaws()
                    mended = []
                    if flaws and all((' is held at ' in fw for fw in flaws)):
                        mended = self.mend_numbers_of(self.number_of_mends())
                        flaws = flaws if not mended else self.row_line_flaws()
                        if mended and (not flaws):
                            self.row_line_keeps_hold += 1
                            outside.append((ring_id, "Not run yet (no step used). Orla set the cart's quantities itself: " + '; '.join(mended) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.trolley_wording()))
                            continue
                    if flaws:
                        self.row_line_keeps_hold += 1
                        self.record_path.append({'kind': 'line_hold', 'faults': flaws[:6]})
                        outside.append((ring_id, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(flaws[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.trolley_wording()))
                        continue
                    peruse = self.pad_attrs(self.trolley_code_ids(), self.rear() + 3) + self.thriftier_entry_rows()
                    ruling = self.probe_trolley()
                    if ruling:
                        noticed = '' if not peruse else "Orla read these listings' attributes for the check (one step each):\n" + '\n'.join(peruse) + '\n'
                        outside.append((ring_id, 'Not run yet. ' + sequence + noticed + ruling + '\nCart now: ' + self.trolley_wording()))
                        continue
                    if reperuse and OUTLAY_MILD < self.charge:
                        outside.append((ring_id, "Not run yet (no step used). Orla re-read the cart's listings just now:\n" + reperuse + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.remaining() > 5:
                    inspect = self.sweep_check()
                    if inspect:
                        outside.append((ring_id, 'Not run yet (no step used). ' + inspect))
                        continue
                    if self.pointers_r and self.remaining() > 4 and (0 == self.turn_down_slabs):
                        self.turn_down_slabs = 1
                        self.record_path.append({'kind': 'refuse_blocked', 'leads': list(self.pointers_r)})
                        outside.append((ring_id, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.pointers_r) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.pointers_r and self.remaining() > 4 and (1 == self.turn_down_slabs):
                        self.turn_down_slabs = 2
                        initial = self.pointers_r[0]
                        self.probe_trolley([initial], OUTLAY_FIRM)
                        if self.rulings.get(self.probe_attr_key([initial])) and initial in self.pointers_r:
                            self.affirmed = initial
                            self.record_path.append({'kind': 'refuse_blocked', 'confirmed': initial})
                            outside.append((ring_id, f"Not run yet (no step used). An independent check of {initial}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.be_turn_down(title_name, argset) and self.remaining() <= 5 and (not self.tardy_kept_hold):
                row_lines = [iw for iw in self.assembled if not self.soldout_outside(iw)]
                newly = self.trolley_look is not None and self.trolley_upon <= self.trolley_noticed_upon
                wanting = [iw for iw in row_lines if iw not in self.trolley_code_ids() or (iw in self.shed_off and (not newly))]
                if len(row_lines) >= len(self.items) and len(wanting) + 1 <= self.remaining():
                    self.tardy_kept_hold = True
                    self.record_path.append({'kind': 'late_decline_held', 'lines': row_lines, 'missing': wanting})
                    do_over = 'Check out the cart as it is.' if not wanting else f"Put {', '.join(wanting)} back in the cart at the requested quantities, then check out."
                    outside.append((ring_id, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + do_over + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.be_turn_down(title_name, argset) and (not self.pointers_r) and (not self.shoved) and (self.remaining() > self.most_paces // 3) and (self.timing_remaining() > ROAM_SPARE + 30):
                self.shoved = True
                outside.append((ring_id, f"Not run yet (no step used). {self.remaining()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.be_turn_down(title_name, argset):
                through = self.fulfil_trolley()
                if through:
                    outside.append((ring_id, through + ('' if not self.wrapped_up else ' [episode finished]')))
                    continue
                argset = self.muted(title_name, argset)
            if closeout and self.remaining() >= 2 and (self.timing_remaining() > CALL_SPARE + 15):
                (proceed, stated_out) = self.toll('end' if not self.be_purchase(title_name, argset) else 'order')
                if not proceed:
                    outside.append((ring_id, 'Not run yet. ' + stated_out + '\nCart now: ' + self.trolley_wording()))
                    continue
                if stated_out:
                    shaved = '\n[Orla] ' + stated_out + shaved
            response = self.do_step(title_name, argset)
            if response is None:
                outside.append((ring_id, 'The call failed (network or server). You may try again.'))
                continue
            sighting = response.get('observation')
            if numerous and isinstance(argset.get(numerous), list) and (len(argset[numerous]) > 1):
                presented_out = _TextKit._render(sighting)
                if any((str(iw) in presented_out for iw in argset[numerous])):
                    self.contrast_fine_one = max(self.contrast_fine_one, len(argset[numerous]))
                else:
                    self.contrast_poor = min(self.contrast_poor or 999, len(argset[numerous]))
                    shaved += f'\n[Orla] The side-by-side view showed nothing for {len(argset[numerous])} listings: send at most {self.contrast_leeway()} per call.'
            self.latest = (self.latest + [tick])[-3:]
            self.code_id_argset.update(((title_name, kw) for (kw, vw) in argset.items() if ([vw] if not isinstance(vw, list) else vw) and all((isinstance(xw, str) and xw in self.entry_rows for xw in ([vw] if not isinstance(vw, list) else vw)))))
            self.soak_up(sighting)
            if title_name == self.jobs.get('compare'):
                entry_rows = self.file_recs_inside(sighting)
                if entry_rows:
                    self.attrs_via[title_name] = self.attrs_via.get(title_name, False) or any((_RowKit._attr_slab(rw) for (_, rw) in entry_rows))
            if title_name in perusers - {self.jobs.get('search')}:
                self.jot_look(title_name, sighting)
            simple = {utensil for (duty, utensil) in self.jobs.items() if 'search' != duty}
            leafed = '' if not (title_name not in simple and (not closeout)) else self.extra_leaves_set(title_name, argset, sighting)
            if self.sole_aim(title_name) and title_name not in (self.jobs.get('cart'), self.jobs.get('buy'), self.jobs.get('refuse'), self.jobs.get('ask')):
                entry_rec = None if not isinstance(sighting, dict) else self.mark_id(sighting)
                if entry_rec:
                    self.attrs_via[title_name] = self.attrs_via.get(title_name, False) or bool(_RowKit._attr_slab(sighting))
                    if title_name == self.jobs.get('open') and (not self.attrs_via[title_name]):
                        another = next((tw for tw in self.layouts if title_name != tw and self.sole_aim(tw) and (self.attrs_via.get(tw) is not False) and (tw not in (self.jobs.get('cart'), self.jobs.get('buy'), self.jobs.get('refuse'), self.jobs.get('ask'), self.jobs.get('search')))), None)
                        if another:
                            self.jobs['open'] = another
                    if not self.attrs_via[title_name] and title_name != self.jobs.get('open') and (not self.jobs.get('offer')):
                        self.jobs['offer'] = title_name
                    if title_name == self.quote_utensil():
                        self.quote_noticed[entry_rec] = self.pace
            if title_name == self.jobs.get('cart'):
                self.jot_trolley(title_name, argset)
                self.trolley_stated_out = _TextKit._render(sighting)
                self.latest = []
                numbers_of = [_SchemaKit._number_in(vw) for vw in argset.values() if not isinstance(vw, str) and _SchemaKit._number_in(vw) is not None]
                for val in argset.values():
                    if numbers_of and isinstance(val, str) and (val in self.entry_rows):
                        if numbers_of[0] <= 0 and (not _RowKit._inside_entry_rows(sighting, val)):
                            self.shed_off.add(val)
                        elif numbers_of[0] > 0:
                            self.shed_off.discard(val)
                row_lines = [iw for iw in self.trolley_code_ids() if not self.soldout_outside(iw) and iw not in self.shed_off]
                if len(row_lines) >= len(self.assembled):
                    self.assembled = row_lines
            if title_name == self.jobs.get('search') and self.file_recs_inside(sighting):
                self.hunted = (self.hunted + [dict(argset)])[-8:]
            if title_name == self.jobs.get('cart_view'):
                self.trolley_look = sighting
                self.trolley_noticed_upon = self.pace
            if self.be_purchase(title_name, argset):
                self.purchases += 1
                if not response.get('finished'):
                    self.turned_down_upon = self.pace
            wording = self.present_r(sighting) + leafed + shaved
            if title_name not in simple and (not closeout):
                wording += self.thriftiest_present_here([sighting] + self.leafed_sight) + self.lid_overlooked(title_name, argset, sighting)
                wording += self.unbidden_unseal([sighting] + self.leafed_sight)
            footing = wording
            evidence_bit = self.proofs(sighting)
            if evidence_bit:
                wording += '\n' + evidence_bit
            if title_name == self.jobs.get('cart') and any((_SchemaKit._number_in(vw) == 0 for vw in argset.values() if not isinstance(vw, str))):
                if any((isinstance(vw, str) and vw in self.entry_rows and _RowKit._inside_entry_rows(sighting, vw) for vw in argset.values())):
                    wording += '\n[Orla] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.jot_soldout = False
            row_lines = self.stray_row_lines(sighting)
            if row_lines:
                sense = self.grab_jot(row_lines)
                if sense:
                    wording += '\n[note read] ' + sense + '. Apply this change to your picks before checking out.'
            turned_down = self.be_purchase(title_name, argset) and (not response.get('finished'))
            once_more = turned_down and any((row_line in self.voiced(sighting) for row_line in self.soldout_row_lines))
            stray = self.jot_soldout and (turned_down or (not self.jot_titled and (not self.be_purchase(title_name, argset))))
            if (stray or once_more) and self.trolley_code_ids() and (not self.wrapped_up):
                noticed = self.seek_soldout()
                if noticed:
                    wording += "\n[Orla re-read the cart's lines after the refusal]\n" + noticed
            self.jot_soldout = False
            switch = self.switch_cue()
            if switch:
                wording += '\n' + switch
            if self.lined_up:
                wording += "\n[notes seen on Orla's own calls] " + ' | '.join(self.lined_up)
                self.lined_up = []
            if title_name in perusers:
                self.peruse_stash[tick] = (self.edition, self.pace, footing)
            if response.get('finished'):
                wording += ' [episode finished]'
            outside.append((ring_id, wording))
        return outside

class _Wrapup:

    def wrap_up(self, once_more: bool=False) -> str:
        if self.wrapped_up or self.no_funds or self.remaining() <= 0:
            return ''
        (purchase, turn_down) = (self.jobs.get('buy'), self.jobs.get('refuse'))
        newly = self.trolley_look is not None and self.trolley_upon <= self.trolley_noticed_upon
        keeping_hold = bool(self.trolley_row_lines) if not newly else bool(self.trolley_code_ids())
        if purchase and (keeping_hold or (self.trolley_handled and (not newly) and (self.remaining() >= 2))):
            flaws: list[str] = []
            argset = _SchemaKit._form(dict(self.digest.get('buy_args') or {}), self.layouts.get(purchase), 'args', flaws)
            if not flaws:
                if self.remaining() >= 2 and self.timing_remaining() > CALL_SPARE + 15:
                    self.toll('order')
                if self.wrapped_up or self.remaining() <= 0:
                    return ''
                response = self.do_step(purchase, argset)
                self.record_path.append({'kind': 'forced_buy', 'reply': _TextKit._snip(response, 200)})
                if self.wrapped_up or self.remaining() <= 0:
                    return ''
                if response is not None and once_more:
                    self.purchases += 1
                    self.turned_down_upon = self.pace
                    self.notice(purchase, response.get('observation'))
                    return 'the checkout was refused: ' + self.present_r(response.get('observation'))
        if turn_down:
            flaws = []
            argset = _SchemaKit._form(self.muted(turn_down, dict(self.digest.get('refuse_args') or {})), self.layouts.get(turn_down), 'args', flaws)
            if not flaws:
                if self.remaining() >= 2 and self.timing_remaining() > CALL_SPARE + 15:
                    self.toll('end')
                if self.wrapped_up or self.remaining() <= 0:
                    return ''
                response = self.do_step(turn_down, argset)
                self.record_path.append({'kind': 'forced_refuse', 'reply': _TextKit._snip(response, 200)})
                if response is not None and (not self.wrapped_up):
                    self.notice(turn_down, response.get('observation'))
                    return 'ending without a purchase was refused: ' + self.present_r(response.get('observation'))
        return ''

    def fulfil_trolley(self) -> str:
        (trolley, purchase) = (self.jobs.get('cart'), self.jobs.get('buy'))
        if self.fulfilled or not self.affirmed or self.soldout_outside(self.affirmed) or (not trolley) or (not purchase):
            return ''
        self.fulfilled = True
        cls_attr_key = self.genre_attr_key()

        def type_of(mark_id: str) -> str:
            return mark_id if not cls_attr_key else str((self.entry_rows.get(mark_id) or {}).get(cls_attr_key) or '').strip().lower()
        blueprint: dict[str, int] = {}
        for (digit, code_ids) in self.pointer_products.items():
            index = _SchemaKit._number_in(digit)
            choose = next((iw for iw in code_ids if not self.soldout_outside(iw)), None)
            if choose and index and (1 <= index <= len(self.items)) and (type_of(choose) not in {type_of(pw) for pw in blueprint}):
                blueprint[choose] = self.items[index - 1].howmany
        if self.affirmed not in blueprint:
            return ''
        at_present = self.trolley_code_ids()
        retain = {iw: self.trolley_row_lines.get(iw, 1) for iw in at_present if iw not in blueprint and (not self.soldout_outside(iw)) and (type_of(iw) not in {type_of(pw) for pw in blueprint})}
        shed = [iw for iw in at_present if iw not in blueprint and iw not in retain]
        if len(blueprint) + len({type_of(iw) for iw in retain}) < len(self.items):
            return ''
        row_lines = {**retain, **blueprint}
        rings = self.trolley_rings(row_lines, shed)
        if not rings or len(rings) + 1 > self.remaining():
            return ''
        self.pad_attrs(list(row_lines), len(rings) + 3)
        self.probe_trolley(list(row_lines), OUTLAY_FIRM)
        if not self.rulings.get(self.probe_attr_key(list(row_lines))):
            return ''
        through = []
        for argset in rings:
            flaws: list[str] = []
            argset = _SchemaKit._form(argset, self.layouts.get(trolley), 'args', flaws)
            if flaws:
                return ''
            response = self.do_step(trolley, argset)
            if response is None or self.wrapped_up:
                return ''
            self.jot_trolley(trolley, argset)
            self.notice(trolley, response.get('observation'))
            through.append(_TextKit._render(argset))
        flaws = []
        argset = _SchemaKit._form(dict(self.digest.get('buy_args') or {}), self.layouts.get(purchase), 'args', flaws)
        if flaws:
            return ''
        responses = []
        for trial in range(2):
            if trial:
                self.reperuse_quotes()
                self.probe_trolley(list(row_lines), OUTLAY_FIRM)
                if not self.rulings.get(self.probe_attr_key(list(row_lines))) or self.remaining() < 1 or self.wrapped_up or any((self.soldout_outside(iw) for iw in row_lines)):
                    break
            (proceed, stated_out) = self.toll('order')
            if stated_out:
                responses.append(stated_out)
            if not proceed:
                break
            response = self.do_step(purchase, argset)
            self.record_path.append({'kind': 'completed', 'lines': row_lines, 'drop': shed, 'reply': _TextKit._snip(response, 200)})
            if response is None:
                break
            self.purchases += 1
            if not response.get('finished'):
                self.turned_down_upon = self.pace
            self.notice(purchase, response.get('observation'))
            responses.append(self.present_r(response.get('observation')))
            if self.wrapped_up or response.get('finished'):
                break
        if not responses:
            return ''
        return 'Not run (Orla acted instead): the independent check confirmed ' + self.affirmed + ', so Orla set the cart (' + '; '.join(through) + ') and checked out. Store reply: ' + ' | then: '.join(responses)

class _DeclineReview:

    def sweep_check_matches_up(self, fetched: dict[str, Any] | None) -> dict[str, list[str]]:
        matches_up = (fetched or {}).get('fits')
        if not isinstance(matches_up, dict):
            return {}
        titled = {str(kw): [str(iw) for iw in _SchemaKit._as_list(vw) if str(iw) in self.entry_rows and (not self.soldout_outside(str(iw)))][:4] for (kw, vw) in matches_up.items()}
        return {kw: vw for (kw, vw) in titled.items() if vw}

    def grab_inspect(self, fetched: dict[str, Any] | None) -> None:
        fetches_up = {needful for (_, needful) in self.cites}
        shifted = []
        apart: set[int] = set()
        for slot_item in _SchemaKit._as_list((fetched or {}).get('not_firm')):
            if isinstance(slot_item, dict):
                (digit, needful) = (_SchemaKit._number_in(slot_item.get('item')) or 0, str(slot_item.get('need') or '').strip())
                if 1 <= digit <= len(self.items) and digit not in apart and (needful in self.items[digit - 1].requisites) and (needful not in self.items[digit - 1].shed_off) and (needful not in fetches_up):
                    apart.add(digit)
                    self.items[digit - 1].shed_off.append(needful)
                    shifted.append(f'set aside {needful}')
        for slot_item in _SchemaKit._as_list((fetched or {}).get('corrected')):
            if isinstance(slot_item, dict):
                (digit, needful, val) = (_SchemaKit._number_in(slot_item.get('item')) or 0, str(slot_item.get('need') or '').strip(), str(slot_item.get('value') or '').strip())
                if 1 <= digit <= len(self.items) and needful in self.items[digit - 1].requisites + self.items[digit - 1].relayed and (needful not in fetches_up) and (':' in val):
                    self.items[digit - 1].shed_off.append(needful)
                    self.items[digit - 1].relayed.append(val)
                    shifted.append(f'{needful} read as {val}')
        if shifted:
            self.edition += 1
            self.record_path.append({'kind': 'review', 'changed': shifted})
            self.jots.append('a review of the request: ' + '; '.join(shifted))

    def sweep_check_ring(self) -> dict[str, Any] | None:
        lid = None if not isinstance(self.purse, (int, float)) else self.purse
        cls_attr_key = self.genre_attr_key()
        unsealed: list[str] = []
        entry_rows: list[str] = []
        for stern in (True, False):
            for mark_id in reversed(self.noticed_purchase_order):
                tag_r = '' if not cls_attr_key else str((self.entry_rows.get(mark_id) or {}).get(cls_attr_key) or '')
                if self.soldout_outside(mark_id) or (stern and (lid is not None and (self.fee_from_of(mark_id) or 0) > lid or (tag_r and (not any((self.holds_up(product, tag_r) for product in self.items)))))):
                    continue
                file_rec = None if not mark_id in self.whole_rec else self.whole_rec.get(mark_id)
                if file_rec and len(unsealed) < 20:
                    unsealed.append('OPENED ' + _TextKit._render(_TextKit._snip(file_rec, 200))[:1000])
                elif not file_rec and len(entry_rows) < 15:
                    entry_rows.append('ROW ' + _TextKit._render(_TextKit._snip(self.entry_rows.get(mark_id), 120))[:500])
            if unsealed or entry_rows:
                break
        if not unsealed and (not entry_rows):
            return None
        prompt_text = DECLINE_REVIEW + '\n\nREQUEST:\n' + self.petition + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        retorts = [row_line for row_line in self.through_journal if 'message to the shopper' in row_line]
        if retorts:
            prompt_text += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(retorts)
        essayed = [str(vw) for argset in self.hunted for vw in argset.values() if isinstance(vw, str)]
        if essayed:
            prompt_text += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(essayed))
        prompt_text += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(unsealed + entry_rows)
        fetched = self.pose_json(prompt_text, self.assesses(), 4000)
        self.record_path.append({'kind': 'audit', 'got': fetched})
        return fetched

    def sweep_check(self) -> str:
        if self.sweep_checked or not self.assesses():
            return ''
        self.sweep_checked = True
        fetched = self.sweep_check_ring()
        self.grab_inspect(fetched)
        titled = self.sweep_check_matches_up(fetched)
        wanting = [nw for nw in range(1, len(self.items) + 1) if not titled.get(str(nw))]
        if wanting:
            entry_rows = [str(iw) for iw in _SchemaKit._as_list((fetched or {}).get('open')) if str(iw) in self.entry_rows and str(iw) not in self.unsealed and (not self.soldout_outside(str(iw)))]
            unsealed = bool(entry_rows) and self.unseal_entry_rows(entry_rows)
            if self.rebound_hunt(fetched, wanting) or unsealed:
                fetched = self.sweep_check_ring()
                self.grab_inspect(fetched)
                titled = self.sweep_check_matches_up(fetched) or titled
        if not titled:
            return ''
        self.pointer_products = titled
        self.pointers_r = [iw for code_ids in titled.values() for iw in code_ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {kw}: {', '.join(vw)}" for (kw, vw) in titled.items())) + ('' if not (fetched and fetched.get('reason')) else f" ({fetched.get('reason')})") + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

class _Shortages:

    def substitutes(self, mislaid: list[str]) -> list[str]:
        if not self.assesses():
            return []
        file_recs = []
        for mark_id in reversed(self.noticed_purchase_order):
            if mark_id in mislaid or mark_id in self.departed or self.soldout_outside(mark_id):
                continue
            file_recs.append(_TextKit._render(_TextKit._snip(self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id), 200))[:1200])
            if len(file_recs) >= 25:
                break
        if not file_recs:
            return []
        prompt_text = SUBSTITUTE_SPEC + '\n\nREQUEST:\n' + self.petition + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        prompt_text += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{iw}: {self.file_rec_from_of(iw)}' for iw in mislaid)) + '\n\nCART NOW:\n' + self.trolley_wording()
        prompt_text += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(file_recs)
        fetched = self.pose_json(prompt_text, self.assesses(), 3500)
        matches_up: list[tuple[float, str]] = []
        for slot_item in _SchemaKit._as_list((fetched or {}).get('fits')):
            mark_id = str(slot_item if not isinstance(slot_item, dict) else slot_item.get('id'))
            fee = None if not isinstance(slot_item, dict) else slot_item.get('price')
            if mark_id in self.entry_rows and mark_id not in mislaid and (mark_id not in self.departed) and (mark_id not in self.trolley_code_ids()) and (mark_id not in [iw for (_, iw) in matches_up]):
                matches_up.append((float('inf') if not (isinstance(fee, (int, float)) and (not isinstance(fee, bool))) else float(fee), mark_id))
        ordered_by = [iw for (_, iw) in matches_up] if not self.digest.get('pick') == 'cheapest' else [iw for (_, iw) in sorted(matches_up, key=lambda fw: fw[0])]
        self.record_path.append({'kind': 'swap', 'lost': mislaid, 'fits': matches_up})
        return ordered_by[:4]

    def seek_soldout(self) -> str:
        code_ids = [iw for iw in self.trolley_code_ids() if iw not in self.departed]
        if any((self.soldout_outside(iw) for iw in code_ids)):
            return ''
        if len(code_ids) == 1:
            self.not_offered.add(code_ids[0])
            return ''
        return '' if not code_ids else self.reperuse_quotes()

    def switch_cue(self) -> str:
        mislaid = [iw for iw in self.trolley_code_ids() if iw not in self.departed and self.soldout_outside(iw)]
        if not mislaid:
            return ''
        for mark_id in mislaid:
            self.departed[mark_id] = self.pace
        self.pointers_r = [iw for iw in self.pointers_r if iw not in mislaid]
        self.edition += 1
        chooses = self.substitutes(mislaid)
        wording = f"[Orla] Sold out now: {', '.join(mislaid)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if chooses and self.remaining() <= 6:
            return wording + f" Few steps are left: in this turn add {chooses[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(chooses[1:]) or 'none'}."
        if chooses and self.digest.get('pick') == 'cheapest':
            return wording + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(chooses)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {chooses[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if chooses:
            return wording + f" Listings already read that appear to fit that item instead: {', '.join(chooses)}. In this turn take the sold-out line out of the cart and add {chooses[0]} at the requested quantity, then check out."
        return wording + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def soldout_outside(self, mark_id: str) -> bool:
        return mark_id in self.not_offered or mark_id in self.departed

    def reperuse_quotes(self) -> str:
        utensil = self.quote_utensil()
        aim = self.sole_aim(utensil)
        if not utensil or not aim:
            return ''
        as_of = max(self.pace - 2, self.turned_down_upon)
        dated = [iw for iw in self.trolley_code_ids() if self.quote_noticed.get(iw, -99) < as_of and iw not in self.departed]
        if not dated or self.remaining() < len(dated) + 3:
            return ''
        row_lines = []
        for mark_id in dated:
            flaws: list[str] = []
            argset = _SchemaKit._form({aim: mark_id}, self.layouts.get(utensil), 'args', flaws)
            if flaws:
                break
            prior = dict(self.entry_rows.get(mark_id) or {})
            response = self.do_step(utensil, argset)
            if response is None or self.wrapped_up:
                break
            sighting = response.get('observation')
            self.quote_noticed[mark_id] = self.pace
            self.notice(utensil, sighting)
            past = self.entry_rows.get(mark_id) or {}
            relocated = [kw for (kw, vw) in past.items() if kw in prior and prior[kw] != vw and (not isinstance(vw, (dict, list)))]
            if relocated:
                row_lines.append(f"{mark_id} changed ({', '.join(relocated[:4])}): {_TextKit._render(_TextKit._snip(sighting, 300))[:900]}")
        self.record_path.append({'kind': 'reread', 'ids': dated, 'changed': len(row_lines)})
        if row_lines:
            self.edition += 1
        return '\n'.join(row_lines)

class _Routes:

    def seek_hop(self, touching: str, terms: str) -> list[str]:
        learned: list[str] = []
        titled = {ww for mark_id in self.store_entry_rows(touching, list(self.entry_rows)) + self.unsealed_store(touching) for (kw, vw) in (self.entry_rows.get(mark_id) or {}).items() if isinstance(vw, str) and len(vw.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kw), re.I) for ww in re.findall('[a-z0-9]{3,}', vw.lower())}
        values = {ww for ww in re.findall('[a-z0-9]+', (terms or touching).lower()) if len(ww) >= 3 or ww.isdigit()} - titled - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for mark_id in self.unsealed_store(touching):
            (pile, quits) = ([self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id) or {}], [])
            while pile:
                unit = pile.pop()
                if not isinstance(unit, (dict, list)):
                    quits.append(str(unit))
                else:
                    pile += unit if not isinstance(unit, dict) else list(unit.values())
            wording = ' '.join(quits).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(vw) + '(?![a-z0-9])', wording) for vw in values)):
                self.record_path.append({'kind': 'link_known', 'id': mark_id})
                return []
        blueprint = self.hunt_pattern_tpl()
        jotting = ' '.join(terms.lower().split())
        if jotting and jotting in self.hop_noticed:
            self.hop_hunts += 1
            learned = self.store_entry_rows(touching, self.hop_noticed[jotting])
        elif blueprint and terms:
            (utensil, footing, column) = blueprint
            flaws: list[str] = []
            argset = _SchemaKit._form({**footing, column: terms}, self.layouts.get(utensil), 'args', flaws)
            if not flaws:
                self.hop_hunts += 1
                response = self.do_step(utensil, argset)
                if response is not None:
                    sighting = response.get('observation')
                    self.notice(utensil, sighting)
                    self.hop_noticed[jotting] = [iw for (iw, _) in self.file_recs_inside(sighting)]
                    learned = self.store_entry_rows(touching, self.hop_noticed[jotting])
                    self.record_path.append({'kind': 'link_search', 'words': terms[:80], 'rows': learned[:4]})
        type_of = {ww for ww in re.findall('[a-z0-9]{3,}', (terms or touching).lower())} - titled - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def from_of_type_of(code_ids: list[str]) -> list[str]:
            return [iw for iw in code_ids if not type_of or type_of & set(re.findall('[a-z0-9]{3,}', _TextKit._render(self.entry_rows.get(iw) or {}).lower()))]
        return (from_of_type_of(learned) or from_of_type_of(self.store_entry_rows(touching, list(reversed(self.noticed_purchase_order)))))[:2]

    def unsealed_store(self, touching: str) -> list[str]:
        short_of = touching.lower()
        outside = []
        for mark_id in self.unsealed:
            entry_row = self.entry_rows.get(mark_id) or {}
            title_names = [str(vw).strip() for (kw, vw) in entry_row.items() if isinstance(vw, str) and 2 < len(vw.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kw), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(nw.lower()) + '(?![a-z0-9])', short_of) for nw in title_names)):
                outside.append(mark_id)
        return outside

    def store_entry_rows(self, touching: str, code_ids: list[str]) -> list[str]:
        short_of = touching.lower()
        outside = []
        for mark_id in code_ids:
            entry_row = self.entry_rows.get(mark_id) or {}
            title_names = [str(vw).strip() for (kw, vw) in entry_row.items() if isinstance(vw, str) and 2 < len(vw.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kw), re.I)]
            if mark_id not in self.unsealed and any((re.search('(?<![a-z0-9])' + re.escape(nw.lower()) + '(?![a-z0-9])', short_of) for nw in title_names)):
                outside.append(mark_id)
        return outside

    def sequence_outlets(self, digit: int) -> set[str]:
        ongoing = {needful for (_, needful) in self.needfuls_at_present()}
        wordings = [needful for (nw, needful) in self.cites if digit == nw and needful in ongoing and (needful not in self.cite_vals)]
        if not wordings:
            return set()
        holders: dict[str, list[str]] = {}
        for (mark_id, entry_row) in self.entry_rows.items():
            for (attr_key, val) in entry_row.items():
                if isinstance(val, str) and 2 < len(val.strip()) <= 40 and re.search('seller|shop|store|merchant|vendor|brand', str(attr_key), re.I):
                    holders.setdefault(val.strip(), []).append(mark_id)
        spotted: set[str] = set()
        for _ in range(6):
            lump = ' '.join(wordings).lower()
            newly = {title_name for title_name in holders if title_name not in spotted and re.search('(?<![a-z0-9])' + re.escape(title_name.lower()) + '(?![a-z0-9])', lump)}
            if not newly:
                break
            spotted |= newly
            for title_name in newly:
                for mark_id in holders[title_name]:
                    if mark_id in self.unsealed:
                        file_rec = self.entry_rows.get(mark_id) or {}
                        wordings += [str(vw) for vw in file_rec.values() if isinstance(vw, str) and len(vw) > 40]
        return spotted

    def route(self) -> tuple[str, bool]:
        ongoing = {needful for (_, needful) in self.needfuls_at_present()}
        open_ended = [needful for (_, needful) in self.cites if needful in ongoing and needful not in self.cite_vals]
        utensil = self.jobs.get('offer') or self.jobs.get('open')
        aim = self.sole_aim(utensil)
        noticed: list[str] = []
        mislaid: list[str] = []
        while open_ended and self.routes < (4 if not self.remaining() > 15 else 6) and (OUTLAY_FIRM > self.charge):
            self.routes += 1
            file_recs = []
            for mark_id in reversed(self.noticed_purchase_order):
                file_rec = None if not mark_id in self.unsealed else self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id)
                if file_rec and len(file_recs) < 45:
                    file_recs.append('OPENED ' + _TextKit._render(_TextKit._snip(file_rec, 300))[:1400])
            for mark_id in reversed(self.noticed_purchase_order):
                if len(file_recs) >= 60:
                    break
                if mark_id not in self.unsealed or not (self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id)):
                    file_recs.append('ROW ' + _TextKit._render(_TextKit._snip(self.entry_rows.get(mark_id), 80))[:300])
            prompt_text = ROUTE_SPEC + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(open_ended)
            prompt_text += '\n\nRECORDS (newest first):\n' + '\n'.join(file_recs)
            fetched = self.pose_json(prompt_text, [SMALL_BRAIN] + ([] if not self.charge + BIG_BRAIN_LEEWAY <= OUTLAY_FIRM else [BIG_BRAIN]), 4500) or {}
            self.record_path.append({'kind': 'trail', 'pending': open_ended, 'got': _TextKit._snip(fetched, 150)})
            range_r: list[str] = []
            mislaid = []
            spreads: list[tuple[str, str]] = []
            for sequence in _SchemaKit._as_list(fetched.get('chains')):
                if not isinstance(sequence, dict):
                    continue
                needful = next((pw for pw in open_ended if pw == str(sequence.get('need') or '').strip()), None) or (None if not len(open_ended) == 1 else open_ended[0])
                if not needful:
                    continue
                hops = [lw for lw in _SchemaKit._as_list(sequence.get('links')) if isinstance(lw, dict)]
                code_ids = [str(lw.get('id') or '').strip() for lw in hops]
                if sequence.get('done') is True and hops and all((iw in self.unsealed for iw in code_ids)) and str(sequence.get('value') or '').strip():
                    self.cite_vals[needful] = str(sequence['value']).strip()
                    self.edition += 1
                    continue
                if hops and all((iw in self.unsealed for iw in code_ids)) and (not str(hops[-1].get('find') or '').strip()):
                    self.cite_vals[needful] = str(sequence.get('value') or '').strip() or f'as stated on {code_ids[-1]}'
                    self.edition += 1
                    continue
                shut = [iw for iw in code_ids if iw in self.entry_rows and iw not in self.unsealed]
                if not shut:
                    spread = next((lw for (lw, iw) in zip(hops, code_ids) if iw not in self.entry_rows), {})
                    touching = str(spread.get('about') or '') or 'the listing it points at'
                    terms = str(spread.get('find') or '').strip()
                    mislaid.append(f'{needful} -> find {touching}' + ('' if not terms else f' (search: {terms})'))
                    spreads.append((touching, terms))
                else:
                    range_r += [iw for iw in shut if iw not in range_r]
            open_ended = [pw for pw in open_ended if pw not in self.cite_vals]
            if not range_r and spreads and utensil and aim and (self.remaining() > 12) and (3 > self.hop_hunts):
                range_r = self.seek_hop(*spreads[0])
                if self.record_path and self.record_path[-1].get('kind') == 'link_known':
                    noticed.append(f"{self.record_path[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    mislaid = mislaid[1:]
            if not range_r or not utensil or (not aim) or (self.remaining() < len(range_r) + 4) or self.wrapped_up:
                break
            for mark_id in range_r[:max(0, min(3, 6 - len(noticed)))]:
                flaws: list[str] = []
                argset = _SchemaKit._form({aim: mark_id}, self.layouts.get(utensil), 'args', flaws)
                if flaws:
                    break
                response = self.do_step(utensil, argset)
                if response is None or self.wrapped_up:
                    break
                self.notice(utensil, response.get('observation'))
                noticed.append(f'{mark_id}: ' + self.present_r(response.get('observation'))[:1500])
        wording = ''
        if noticed:
            wording += '[Orla opened the pages a requirement points at]\n' + '\n'.join(noticed) + '\n'
        if self.cite_vals:
            wording += '[values read from the pages requirements point at] ' + '; '.join((f'{kw} -> {vw}' for (kw, vw) in self.cite_vals.items())) + '\n'
        keep_hold = bool(mislaid) and self.route_slabs < (2 if not self.remaining() > 15 else 3) and (self.remaining() > 5)
        if keep_hold:
            self.route_slabs += 1
            wording += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(mislaid[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (wording, keep_hold)

class _CartProbe:

    def probe_trolley(self, code_ids: list[str] | None=None, cutoff: float=OUTLAY_MILD) -> str:
        pointer = code_ids is not None
        code_ids = self.trolley_code_ids() if not pointer else code_ids
        attr_key = self.probe_attr_key(code_ids)
        if attr_key in self.probed or not self.assesses(cutoff) or (not code_ids):
            return ''
        self.probed.add(attr_key)
        prompt_text = CART_PROBE + '\n\nREQUEST:\n' + self.petition + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        if not pointer:
            prompt_text += '\n\nCART:\n' + self.trolley_wording()
        else:
            amount_of = '; '.join((f'item {nw}: quantity {that_one.howmany}' for (nw, that_one) in enumerate(self.items, 1)))
            prompt_text += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + amount_of + '):\n' + ', '.join(code_ids)
        prompt_text += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{iw}: ' + ('' if not self.soldout_outside(iw) else 'SOLD OUT (the store says it can no longer be bought). ') + self.file_rec_from_of(iw) for iw in code_ids))
        if self.digest.get('pick') in ('cheapest', 'most', 'least'):
            prompt_text += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.the_rest(code_ids, 15)
        fetched = self.pose_json(prompt_text, self.assesses(cutoff), 3000)
        self.record_path.append({'kind': 'check', 'cart': code_ids, 'got': fetched})
        self.rulings[attr_key] = bool(fetched) and fetched.get('ok') is not False
        if not fetched or fetched.get('ok') is not False:
            return ''
        self.pointers_r = [iw for iw in self.pointers_r if iw not in code_ids]
        snags = [str(pw) for pw in _SchemaKit._as_list(fetched.get('problems'))][:5]
        finer = [str(bw) for bw in _SchemaKit._as_list(fetched.get('better')) if str(bw) in self.entry_rows and (not self.soldout_outside(str(bw)))][:4]
        wording = 'A careful review says this cart is not right yet: ' + '; '.join(snags or ['see the requirements'])
        if finer:
            wording += '. Listings that look better: ' + ', '.join(finer)
        return wording + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def probe_attr_key(self, code_ids: list[str]) -> str:
        return '|'.join(sorted(code_ids)) + f'#{self.edition}'

class _Counts:

    def mend_numbers_of(self, off_mark: list[tuple[str, int, int]]) -> list[str]:
        title_name = self.jobs.get('cart') or ''
        layout = self.layouts.get(title_name) or {}
        traits = layout.get('properties') or {}
        wordings = [kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'string' in _SchemaKit._types_of(vw) and ('enum' not in vw)]
        aim = next((kw for (tw, kw) in self.code_id_argset if title_name == tw and kw in wordings), None) or next((kw for kw in layout.get('required') or [] if kw in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        count = next((kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'integer' in _SchemaKit._types_of(vw)), None)
        style_mode = next((kw for (kw, vw) in traits.items() if isinstance(vw, dict) and isinstance(vw.get('enum'), list) and (len(vw['enum']) >= 2)), None)
        if not title_name or not aim or (not count):
            return []
        rings: list[dict[str, Any]] = []
        for (mark_id, own_it, desire) in off_mark:
            if self.trolley_places(title_name):
                rings.append({aim: mark_id, count: desire})
            elif not style_mode:
                return []
            else:
                choices = [str(ow) for ow in traits[style_mode]['enum']]
                away = next((ow for ow in choices if re.search('remov|delet|drop|clear|take', ow, re.I)), choices[1])
                onto = next((ow for ow in choices if away != ow), choices[0])
                rings += [{aim: mark_id, style_mode: away, count: own_it}, {aim: mark_id, style_mode: onto, count: desire}]
        if len(rings) + 3 > self.remaining():
            return []
        through = []
        for argset in rings:
            flaws: list[str] = []
            argset = _SchemaKit._form(argset, layout, 'args', flaws)
            if flaws:
                break
            response = self.do_step(title_name, argset)
            if response is None or self.wrapped_up:
                break
            self.jot_trolley(title_name, argset)
            self.trolley_stated_out = _TextKit._render(response.get('observation'))
            self.notice(title_name, response.get('observation'))
            through.append(_TextKit._render(argset) + ' -> ' + self.present_r(response.get('observation'))[:300])
        self.record_path.append({'kind': 'count_fix', 'calls': through})
        return through

    def number_of_mends(self) -> list[tuple[str, int, int]]:
        cls_attr_key = self.genre_attr_key()
        if not cls_attr_key:
            return []
        outside = []
        for mark_id in self.trolley_code_ids():
            own_it = self.kept_hold_pieces_n(mark_id) or 0
            holders = [nw for (nw, product) in enumerate(self.items, 1) if self.holds_up(product, str((self.entry_rows.get(mark_id) or {}).get(cls_attr_key) or ''))]
            my_own = [iw for iw in self.trolley_code_ids() if (self.trolley_row_lines.get(iw, 1) or 0) > 0 and len(holders) == 1 and self.holds_up(self.items[holders[0] - 1], str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or ''))]
            if 0 < own_it and len(holders) == 1 and (len(my_own) == 1) and (own_it != self.items[holders[0] - 1].howmany) and (not self.soldout_outside(mark_id)):
                outside.append((mark_id, own_it, self.items[holders[0] - 1].howmany))
        return outside

class _Matching:

    def row_line_flaws(self) -> list[str]:
        cls_attr_key = self.genre_attr_key()
        code_ids = [iw for iw in self.trolley_code_ids() if (self.kept_hold_pieces_n(iw) or 0) > 0]
        if not code_ids or not cls_attr_key:
            return []
        tag_r = {iw: str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or '') for iw in code_ids}
        learned = {nw for (nw, product) in enumerate(self.items, 1) if any((self.holds_up(product, str(rw.get(cls_attr_key) or '')) for rw in self.entry_rows.values()))}
        holder = {iw: [nw for (nw, product) in enumerate(self.items, 1) if self.holds_up(product, tag_r[iw])] for iw in code_ids}
        flaws: list[str] = []
        for (nw, product) in enumerate(self.items, 1):
            if nw not in learned:
                continue
            row_lines = [iw for iw in code_ids if nw in holder[iw]]
            if not row_lines:
                flaws.append(f'item {nw} ({product.title_name}) has no line in the cart')
            elif len([iw for iw in row_lines if holder[iw] == [nw]]) > 1:
                flaws.append(f"item {nw} ({product.title_name}) has {len(row_lines)} lines ({', '.join(row_lines)}); keep one")
            for iw in row_lines:
                own_it = self.kept_hold_pieces_n(iw)
                if holder[iw] == [nw] and product.howmany != own_it:
                    flaws.append(f'{iw} is held at {own_it} unit(s); item {nw} ({product.title_name}) wants {product.howmany}')
        if len(learned) == len(self.items):
            for iw in code_ids:
                if not holder[iw]:
                    flaws.append(f"{iw} ({tag_r[iw] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for iw in code_ids:
            if self.soldout_outside(iw):
                flaws.append(f'{iw} is sold out')
        fees = {iw: self.fee_from_of(iw) for iw in code_ids}
        if isinstance(self.purse, (int, float)) and all((pw is not None for pw in fees.values())):
            if not self.digest.get('budget_scope') == 'each':
                sum_total = sum((fees[iw] * self.kept_hold_pieces_n(iw) for iw in code_ids))
                if self.purse + 1e-09 < sum_total <= 10 * self.purse:
                    flaws.append(f'the total is {sum_total:g}, over the budget of {self.purse:g}')
            else:
                beyond = [iw for iw in code_ids if self.purse < fees[iw] <= 10 * self.purse]
                flaws += [f'{iw} costs {fees[iw]:g}, over the {self.purse:g} limit for each item' for iw in beyond]
        return flaws

    def match_up_blueprint(self, brief_one: bool=False) -> str:
        if not self.edge_val:
            return ''
        index = self.match_up_pos()
        row_lines = []
        for chief in sorted({mw for (_, mw) in self.edge_val}):
            pieces = [pw for (pw, mw) in self.edge_val if chief == mw]
            reckon: dict[str, dict[int, int]] = {}
            for pw in pieces:
                for iw in self.product_code_ids(pw):
                    if self.soldout_outside(iw):
                        continue
                    owned = (index['marks'].get(iw) or {}).get('own') or ''
                    titled = self.match_up_place(iw, index) | (set() if not re.match(PIECE_RANGE + '-', owned) else {owned.split('-')[0]})
                    for mark_code in titled:
                        reckon.setdefault(mark_code, {}).setdefault(pw, 0)
                        reckon[mark_code][pw] += 1
            ordered_by = sorted(reckon.items(), key=lambda pair_kv: (-len(pair_kv[1]), -sum(pair_kv[1].values())))[:4]
            gadgets = {}
            for (mark_code, _) in ordered_by:
                ticks = {iw: index['marks'].get(iw) or {} for iw in self.product_code_ids(chief)}
                gadgets[mark_code] = [iw for (iw, mw) in ticks.items() if any((_FitKit._mark_code_title_names(mark_code, cw) for cw in mw.get('codes') or [])) or mark_code in (mw.get('series') or [])][:2]
            wording = '; '.join((f'{mark_code}: ' + ', '.join((f'{nw} {self.items[pw - 1].title_name}' for (pw, nw) in sorted(a_head.items()))) + (' (device not seen yet)' if not gadgets[mark_code] else f" (device seen: {', '.join(gadgets[mark_code])})") for (mark_code, a_head) in ordered_by))
            if wording:
                row_lines.append(f"codes the {self.items[chief - 1].title_name} candidates' parts name -> {wording}")
        if not row_lines:
            return ''
        return "[fit plan: a device fits a part when the part's text names the device's model code, its series or another name it is sold as, or when the device's text names the part's series; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(row_lines) if not brief_one else ' | '.join(row_lines)

    def matches_up_gadget(self, mark_id: str, gadget: dict[str, Any], index: dict[str, Any]) -> bool | None:
        return _FitKit._match_up_ruling(index['marks'].get(mark_id) or {}, gadget, self.match_up_place(mark_id, index))

    def match_up_flaws(self) -> list[str]:
        if not self.edge_val:
            return []
        index = self.match_up_pos()
        trolley = self.trolley_code_ids()
        flaws = []
        for (piece, chief) in self.edge_val:
            gadgets = [iw for iw in trolley if iw in self.product_code_ids(chief)]
            pieces = [iw for iw in trolley if iw in self.product_code_ids(piece)]
            if len(gadgets) != 1 or not pieces:
                continue
            gadget = index['marks'].get(gadgets[0]) or {}
            for row_line in pieces:
                if self.matches_up_gadget(row_line, gadget, index) is False:
                    wanted_item = [iw for iw in self.product_code_ids(piece) if self.matches_up_gadget(iw, gadget, index) and (not self.soldout_outside(iw)) and (iw not in trolley)][:4]
                    titled = sorted(self.match_up_place(row_line, index))[:4] or [(index['marks'].get(row_line) or {}).get('own') or 'other models']
                    flaws.append(f"{row_line} ({self.items[piece - 1].title_name}) works with {', '.join(titled)}, not with {gadgets[0]} ({', '.join(sorted(gadget.get('codes') or gadget.get('series') or []))})" + ('' if not wanted_item else f"; seen parts of that kind that work with it: {', '.join(wanted_item)}"))
        return flaws

    def product_code_ids(self, digit: int) -> list[str]:
        cls_attr_key = self.genre_attr_key()
        product = self.items[digit - 1]
        return [iw for iw in self.noticed_purchase_order if cls_attr_key and self.holds_up(product, str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or ''))]

    def match_up_place(self, mark_id: str, index: dict[str, Any]) -> set[str]:
        tick = index['marks'].get(mark_id) or {}
        spotted = set(tick.get('works') or [])
        if tick.get('own'):
            spotted |= index['notes'].get(tick['own'], set())
        return spotted

    def match_up_pos(self) -> dict[str, Any]:
        ticks: dict[str, dict[str, Any]] = {}
        jots: dict[str, set[str]] = {}
        for mark_id in self.noticed_purchase_order:
            file_rec = self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id) or {}
            wording = ' '.join((str(vw) for vw in file_rec.values() if isinstance(vw, str)))
            wording += ' ' + '. '.join((f'{kw} {vw}' for slab in file_rec.values() if isinstance(slab, dict) for (kw, vw) in slab.items() if isinstance(vw, str)))
            ticks[mark_id] = _FitKit._match_up_ticks(wording)
            for (piece, mark_codes) in ticks[mark_id]['also']:
                jots.setdefault(piece, set()).update(mark_codes)
        return {'marks': ticks, 'notes': jots}

class _Basket:

    def trolley_rings(self, row_lines: dict[str, int], shed: list[str]) -> list[dict[str, Any]] | None:
        title_name = self.jobs.get('cart') or ''
        layout = self.layouts.get(title_name) or {}
        traits = layout.get('properties') or {}
        for (attr_key, spec_sheet) in traits.items():
            form = None if not (isinstance(spec_sheet, dict) and 'array' in _SchemaKit._types_of(spec_sheet)) else spec_sheet.get('items')
            if isinstance(form, dict) and isinstance(form.get('properties'), dict):
                inward = form['properties']
                code_id_attr_key = next((kw for (kw, sw) in inward.items() if isinstance(sw, dict) and 'string' in _SchemaKit._types_of(sw) and ('enum' not in sw)), None)
                number_of_attr_key = next((kw for (kw, sw) in inward.items() if isinstance(sw, dict) and 'integer' in _SchemaKit._types_of(sw)), None)
                if not code_id_attr_key:
                    return None
                return [{attr_key: [{code_id_attr_key: iw, **({} if not number_of_attr_key else {number_of_attr_key: nw})} for (iw, nw) in row_lines.items()]}]
        wordings = [kw for (kw, sw) in traits.items() if isinstance(sw, dict) and 'string' in _SchemaKit._types_of(sw) and ('enum' not in sw)]
        aim = next((kw for (tw, kw) in self.code_id_argset if title_name == tw and kw in wordings), None) or next((kw for kw in layout.get('required') or [] if kw in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        count = next((kw for (kw, sw) in traits.items() if isinstance(sw, dict) and 'integer' in _SchemaKit._types_of(sw)), None)
        style_mode = next((kw for (kw, sw) in traits.items() if isinstance(sw, dict) and isinstance(sw.get('enum'), list) and (len(sw['enum']) >= 2)), None)
        if not aim:
            return None
        rings: list[dict[str, Any]] = []
        for mark_id in shed:
            own_it = self.trolley_row_lines.get(mark_id, 1)
            if style_mode:
                rings.append({aim: mark_id, style_mode: traits[style_mode]['enum'][1], **({} if not count else {count: own_it})})
            elif count and self.trolley_places(title_name):
                rings.append({aim: mark_id, count: 0})
            elif not (count and isinstance(traits[count].get('minimum'), int) and (traits[count]['minimum'] <= -own_it)):
                return None
            else:
                rings.append({aim: mark_id, count: -own_it})
        for (mark_id, desire) in row_lines.items():
            own_it = 0 if not mark_id in self.trolley_code_ids() else self.trolley_row_lines.get(mark_id, 0)
            if desire == own_it:
                continue
            if own_it and (not self.trolley_places(title_name)):
                return None
            rings.append({aim: mark_id, **({} if not style_mode else {style_mode: traits[style_mode]['enum'][0]}), **({} if not count else {count: desire})})
        return rings

    def jot_trolley(self, title_name: str, argset: dict[str, Any]) -> None:
        self.trolley_handled = True
        self.trolley_upon = self.pace
        spec_sheet = self.layouts.get(title_name, {}).get('properties') or {}
        taking_off = False
        for (attr_key, val) in argset.items():
            choices = (spec_sheet.get(attr_key) or {}).get('enum')
            if isinstance(choices, list) and len(choices) >= 2 and (val in choices[1:]):
                taking_off = True
        stray_number_of = next((_SchemaKit._number_in(vw) for vw in argset.values() if _SchemaKit._number_in(vw) is not None and (not isinstance(vw, str))), None)
        row_lines: list[tuple[str, int | None]] = []
        complete_one = False
        for (attr_key, val) in argset.items():
            if isinstance(val, list):
                form = (spec_sheet.get(attr_key) or {}).get('items') or {}
                if not val and isinstance(form, dict) and (form.get('type') == 'object' or 'properties' in form):
                    complete_one = True
                for row_line in val:
                    if isinstance(row_line, dict):
                        complete_one = True
                        mark_id = next((vw for vw in row_line.values() if isinstance(vw, str) and vw in self.entry_rows), None)
                        count = next((_SchemaKit._number_in(vw) for vw in row_line.values() if _SchemaKit._number_in(vw) is not None and (not isinstance(vw, str))), 1)
                        if mark_id:
                            row_lines.append((mark_id, count))
                    elif isinstance(row_line, str) and row_line in self.entry_rows:
                        row_lines.append((row_line, stray_number_of))
            elif isinstance(val, str) and val in self.entry_rows:
                row_lines.append((val, stray_number_of))
        if complete_one:
            self.trolley_row_lines = {iw: cw for (iw, cw) in row_lines if cw and 0 < cw}
            return
        places = self.trolley_places(title_name)
        for (mark_id, count) in row_lines:
            own_it = self.trolley_row_lines.get(mark_id, 0)
            if taking_off:
                own_it = 0 if not count else own_it - count
            elif not places:
                own_it = own_it + (1 if not count is not None else count)
            else:
                own_it = 1 if not count is not None else count
            if not 0 < own_it:
                self.trolley_row_lines.pop(mark_id, None)
            else:
                self.trolley_row_lines[mark_id] = own_it

    def trolley_places(self, title_name: str) -> bool:
        traits = self.layouts.get(title_name, {}).get('properties') or {}
        if any((isinstance(sw, dict) and isinstance(sw.get('enum'), list) and (len(sw['enum']) >= 2) for sw in traits.values())):
            return False
        return any((isinstance(sw, dict) and 'integer' in _SchemaKit._types_of(sw) and (sw.get('minimum') == 0) for sw in traits.values()))

    def untangle_code_ids(self, title_name: str, argset: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if title_name not in {self.jobs.get(rw) for rw in ('cart', 'open', 'offer', 'compare')} - {None} or not self.code_id_attr_key:
            return (argset, '')
        traits = (self.layouts.get(title_name) or {}).get('properties') or {}
        spare_one = [kw for (kw, vw) in traits.items() if isinstance(vw, dict) and ('string' in _SchemaKit._types_of(vw) or 'array' in _SchemaKit._types_of(vw)) and ('enum' not in vw)]
        if len(spare_one) != 1 or (title_name != self.jobs.get('cart') and (title_name, spare_one[0]) not in self.code_id_argset):
            return (argset, '')

        def mend(val: Any) -> tuple[Any, str]:
            if isinstance(val, str) and val.strip() in self.entry_rows:
                return (val.strip(), '')
            if not isinstance(val, str) or _RowKit._code_id_resembling(val, True) or len(val.strip()) < 8:
                return (val, '')
            short_of = val.strip().lower()
            strikes = [iw for (iw, entry_row) in self.entry_rows.items() if any((self.code_id_attr_key != kw and isinstance(vw, str) and vw.strip().lower().startswith(short_of) for (kw, vw) in entry_row.items()))]
            if len(strikes) == 1:
                return (strikes[0], '')
            if strikes:
                return (val, f"several listings carry the text {val!r} ({', '.join(strikes[:6])}); pass one listing id")
            return (val, f'{val!r} is not a listing id; pass the id from the result rows (their {self.code_id_attr_key} field)')
        outside = dict(argset)
        for attr_key in spare_one:
            if attr_key not in argset:
                continue
            mended = [mend(argset[attr_key])] if not isinstance(argset[attr_key], list) else [mend(vw) for vw in argset[attr_key]]
            snag = next((pw for (_, pw) in mended if pw), '')
            if snag:
                return (argset, snag)
            outside[attr_key] = mended[0][0] if not isinstance(argset[attr_key], list) else [vw for (vw, _) in mended]
        return (outside, '')

    def the_rest(self, omit: list[str], cutoff: int=30) -> str:
        row_lines = []
        purchase_order = list(reversed(self.noticed_purchase_order))
        if self.digest.get('pick') == 'cheapest':
            purchase_order.sort(key=lambda iw: (iw not in self.whole_rec, float('inf') if not self.fee_from_of(iw) is not None else self.fee_from_of(iw)))
        for mark_id in purchase_order:
            if mark_id in omit or self.soldout_outside(mark_id):
                continue
            file_rec = self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id)
            row_lines.append(_TextKit._render(_TextKit._snip(file_rec, 160))[:900])
            if len(row_lines) >= cutoff:
                break
        return '\n'.join(row_lines)

    def file_rec_from_of(self, mark_id: str) -> str:
        file_rec = self.whole_rec.get(mark_id) or self.entry_rows.get(mark_id)
        return '(no record seen)' if not file_rec else _TextKit._render(_TextKit._snip(file_rec, 400))[:3000]

    def trolley_code_ids(self) -> list[str]:
        if self.trolley_look is not None and self.trolley_upon <= self.trolley_noticed_upon:
            wording = _TextKit._render(self.trolley_look)
            return [iw for iw in self.entry_rows if iw in wording]
        return list(self.trolley_row_lines)

    def kept_hold_pieces_n(self, mark_id: str) -> int:
        if self.trolley_look is not None and self.trolley_upon <= self.trolley_noticed_upon:
            pile: list[Any] = [self.trolley_look]
            while pile:
                unit = pile.pop()
                if isinstance(unit, dict):
                    if mark_id in [vw.strip() for vw in unit.values() if isinstance(vw, str)]:
                        numbers_of = [_SchemaKit._number_in(vw) for (kw, vw) in unit.items() if not isinstance(vw, str) and _SchemaKit._number_in(vw) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _TextKit._terms(kw))]
                        if len(numbers_of) == 1:
                            return numbers_of[0]
                    pile += [vw for vw in unit.values() if isinstance(vw, (dict, list))]
                elif isinstance(unit, list):
                    pile += unit
        return self.trolley_row_lines.get(mark_id, 1)

    def trolley_wording(self) -> str:
        if self.trolley_look is not None and self.trolley_upon <= self.trolley_noticed_upon:
            return _TextKit._render(_TextKit._snip(self.trolley_look, 300))[:3000]
        return "lines by Orla's record of the cart calls (listing: units): " + _TextKit._render(self.trolley_row_lines)

    def trolley_tick(self) -> str:
        return _TextKit._render(sorted(self.trolley_row_lines.items()))

    def plain_kind_products(self) -> int:
        if len(self.items) < 2:
            return 0
        cls_attr_key = self.genre_attr_key()
        kept_hold = set() if not cls_attr_key else {self.holder_from_of(str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or '')) for iw in self.trolley_code_ids()}
        return sum((1 for nw in range(1, len(self.items) + 1) if nw not in kept_hold))

class _Tolls:

    def toll_pending_due(self, phase: str) -> bool:
        return any((pw['when'] == phase for pw in self.procedure)) and (phase, self.trolley_tick() + f'#{self.edition}') not in self.toll_fine_one

    def toll(self, phase: str) -> tuple[bool, str]:
        paces = [pw for pw in self.procedure if pw['when'] == phase]
        if not paces or self.wrapped_up or (not self.toll_pending_due(phase)):
            return (True, '')
        closeout = {self.jobs.get('buy'), self.jobs.get('refuse')} - {None}
        utensils = [f"- {tw['function']['name']}: {tw['function'].get('description', '')} | parameters: {_TextKit._render(tw['function'].get('parameters') or {})}" for tw in self.utensils if tw['function']['name'] not in closeout]
        stated_out: list[str] = []
        prodded = False
        for _ in range(TOLL_ROUNDS):
            prompt_text = TOLL_SPEC.replace('STAGE', 'end without a purchase' if not 'order' == phase else 'place the order') + '\n\nREQUEST:\n' + self.petition + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {pw['step']}" for pw in paces)) + '\n\nLEDGER:\n' + self.register() + '\n\nCART NOW: ' + self.trolley_wording() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{iw}. {row_line}' for (iw, row_line) in enumerate(self.through_journal, 1))) or '(nothing yet)') + ('' if not self.jots else '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.jots[-4:])) + f"\n\nMESSAGES LEFT: {max(0, self.mv_cutoff - self.mv_spent_up)}   STEPS LEFT: {self.remaining()} (the {('ending' if not 'order' == phase else 'order')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(utensils)
            fetched = self.pose_json(prompt_text, [SMALL_BRAIN, BIG_BRAIN] if not prodded else [BIG_BRAIN, SMALL_BRAIN], 3500)
            if fetched is None:
                break
            relayed: list[str] = []
            keep_hold = ''
            if stated_out:
                self.grab_listened({'needs': fetched.get('needs')}, relayed)
                keep_hold = str(fetched.get('hold') or '').strip()
            ring = None if not isinstance(fetched.get('call'), dict) else fetched.get('call')
            self.record_path.append({'kind': 'gate', 'stage': phase, 'call': _TextKit._snip(ring, 200), 'steps': _TextKit._snip(fetched.get('steps'), 120), 'hold': keep_hold[:200], 'told': relayed})
            if relayed or keep_hold:
                self.edition += 1
                return (False, ('' if not stated_out else 'Before going on, Orla did the steps the request asks for:\n' + '\n'.join(stated_out) + '\n') + ("The shopper's reply changes the task: " + '; '.join(relayed + ([] if not keep_hold else [keep_hold])) + '. Fix the cart for it, then check out again.'))
            title_name = str((ring or {}).get('tool') or '')
            presented_out = [ew for ew in _SchemaKit._as_list(fetched.get('steps')) if isinstance(ew, dict)]
            chatter = 'order' == phase and any((pw['messages'] > 0 for pw in paces))
            row_lines = range(1, len(self.through_journal) + 1)
            evidence_bit = {iw for iw in row_lines if not chatter or (iw <= len(self.through_paces) and self.through_paces[iw - 1] > self.trolley_upon)}
            untested = [ew for ew in presented_out if ew.get('done') is not True or not any(((_SchemaKit._number_in(nw) or 0) in evidence_bit for nw in _SchemaKit._as_list(ew.get('lines'))))]
            if not ring and untested and (not stated_out) and (not prodded) and self.jobs.get('ask') and (self.mv_cutoff > self.mv_spent_up) and (self.remaining() >= 2):
                prodded = True
                self.record_path.append({'kind': 'gate_unproven', 'steps': _TextKit._snip(untested, 120)})
                self.through_journal.append(f'(Orla: no line made since the cart last changed at step {self.trolley_upon} shows ' + '; '.join((str(ew.get('step') or '') for ew in untested))[:200] + f' done, and {self.mv_cutoff - self.mv_spent_up} message(s) and {self.remaining()} steps are left: give the call that does it)')
                self.through_paces.append(-1)
                continue
            if not ring or title_name not in self.layouts or title_name in closeout:
                if not ring:
                    self.toll_fine_one.add((phase, self.trolley_tick() + f'#{self.edition}'))
                break
            if self.remaining() < 2 or (title_name == self.jobs.get('ask') and self.mv_cutoff <= self.mv_spent_up):
                break
            flaws: list[str] = []
            argset = _SchemaKit._form({} if not isinstance(ring.get('args'), dict) else ring.get('args'), self.layouts[title_name], 'args', flaws)
            if flaws:
                break
            if title_name == self.jobs.get('ask'):
                wording = next((vw for vw in argset.values() if isinstance(vw, str)), '')
                lot = self.post([(1, '', wording)], 1)
                if not lot:
                    break
                self.toll_mvs += 1
                stated_out.append(f'Q: {wording}\nA: {lot[0][2]}')
                continue
            response = self.do_step(title_name, argset)
            if response is None:
                break
            noticed = response.get('observation')
            self.soak_up(noticed)
            self.through_journal.append(f'step {self.pace}: {title_name} {_TextKit._render(argset)} -> {self.present_r(noticed)[:300]} (cart then: {self.trolley_tick()})')
            self.through_paces.append(self.pace)
            stated_out.append(f'{title_name} {_TextKit._render(argset)} -> {self.present_r(noticed)[:400]}')
        return (True, '' if not stated_out else 'Before going on, Orla did the steps the request asks for:\n' + '\n'.join(stated_out))

class _Replies:

    def notice(self, title_name: str, sighting: Any) -> str:
        self.soak_up(sighting)
        row_lines = self.stray_row_lines(sighting)
        if not row_lines:
            return ''
        sense = self.grab_jot(row_lines)
        if sense:
            self.lined_up.append(f'(seen on {title_name}) {sense}')
        return sense

    def grab_jot(self, row_lines: list[str]) -> str:
        self.listened.update(row_lines)
        prompt_text = JOT_SPEC + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.register() + '\n\nTHE NOTE:\n' + '\n'.join(row_lines)
        fetched = self.pose_json(prompt_text, [SMALL_BRAIN, BACKUP_BRAIN], 2500) or {}
        departed = [str(gw) for gw in _SchemaKit._as_list(fetched.get('gone')) if isinstance(gw, (str, int))]
        self.not_offered.update((gw for gw in departed if gw in self.entry_rows))
        self.jot_soldout = bool(departed) or fetched.get('sold_out') is True
        self.jot_titled = any((gw in self.entry_rows for gw in departed))
        if self.jot_soldout and len(row_lines) == 1:
            self.soldout_row_lines.update(row_lines)
        if fetched.get('kind') not in (None, 'change'):
            return ''
        shifted = []
        purse = fetched.get('budget')
        if isinstance(purse, (int, float)) and (not isinstance(purse, bool)) and (0 < purse) and (self.purse != purse):
            self.purse = purse
            shifted.append(f'budget now {purse}')
        joined_in: set[str] = set()
        sheds = [ew for ew in _SchemaKit._as_list(fetched.get('drop')) if isinstance(ew, dict) and str(ew.get('need') or '').strip()]
        joins_in = [ew for ew in _SchemaKit._as_list(fetched.get('add')) if isinstance(ew, dict) and str(ew.get('need') or '').strip()]
        for slot_item in sheds:
            product = self.items[min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items)) - 1]
            aim = str(slot_item['need']).strip()
            ongoing = [mw for mw in product.requisites + product.relayed if mw not in product.shed_off]
            strike = [mw for mw in ongoing if aim == mw]
            if not strike:
                strike = [mw for mw in ongoing if _TextKit._terms(mw.split(':')[0]) == _TextKit._terms(aim.split(':')[0])][:1]
            product.shed_off.extend(strike or [aim])
            shifted.append(f'dropped {aim}')
        for slot_item in joins_in:
            product = self.items[min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items)) - 1]
            needful = str(slot_item['need']).strip()
            if needful in product.shed_off:
                product.shed_off.remove(needful)
            if needful not in product.requisites + product.relayed:
                product.relayed.append(needful)
            joined_in.add(needful)
            shifted.append(f'added {needful}')
        if departed:
            shifted.append('unavailable: ' + ', '.join(departed))
        if fetched.get('price_moved') is True:
            shifted.append('a price changed')
        abstract = str(fetched.get('summary') or '').strip()
        if self.digest.get('pick') == 'cheapest' and (sheds or departed or self.jot_soldout or any((cw.startswith('budget now') for cw in shifted))):
            self.broader_kept_hold = False
            self.thriftier_ticks = set()
        if shifted or abstract:
            self.edition += 1
            self.latest = []
            self.jots.append(abstract or '; '.join(shifted))
        return (abstract + ('' if not shifted else ' (' + '; '.join(shifted) + ')')).strip()

    def stray_row_lines(self, sighting: Any) -> list[str]:
        row_lines: list[str] = []
        if isinstance(sighting, dict) and (not self.mark_id(sighting)):
            for (attr_key, val) in sighting.items():
                if isinstance(val, str) and len(val.split()) >= 5:
                    row_lines.append(val)
                elif isinstance(val, list) and val and all((isinstance(vw, str) for vw in val)):
                    row_lines.extend((vw for vw in val if len(vw.split()) >= 5))
                elif isinstance(val, dict) and (not self.mark_id(val, False, 1)):
                    for inward in val.values():
                        if isinstance(inward, str) and len(inward.split()) >= 6:
                            row_lines.append(inward)
        return [row_line for row_line in row_lines if row_line not in self.listened]

    def voiced(self, sighting: Any) -> str:
        if isinstance(sighting, str):
            return sighting
        spotted: list[str] = []

        def stroll(unit: Any, nesting: int) -> None:
            if 4 < nesting:
                return
            if isinstance(unit, str) and len(unit.split()) >= 2:
                spotted.append(unit)
            elif isinstance(unit, dict):
                for val in unit.values():
                    stroll(val, nesting + 1)
            elif isinstance(unit, list):
                for val in unit:
                    stroll(val, nesting + 1)
        stroll(sighting, 0)
        return _TextKit._render(sighting)[:400] if not spotted else ' '.join(spotted)[:800]

    def listen(self, posted: list[tuple[int, str, str]], brains: tuple[str, ...]=(BIG_BRAIN, SMALL_BRAIN)) -> dict[str, Any]:
        prompt_text = REPLY_SPEC + '\n\nITEMS:\n' + '\n'.join((f'item {nw}: {that_one.title_name}' for (nw, that_one) in enumerate(self.items, 1)))
        prompt_text += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{iw}. (about item {nw}) Q: {qw}\nA: {aw}' for (iw, (nw, qw, aw)) in enumerate(posted, 1)))
        return self.pose_json(prompt_text, list(brains), 2500) or {}

    def grab_listened(self, listened: dict[str, Any], relayed: list[str]) -> None:
        numerous = len(self.items) > 1
        for slot_item in _SchemaKit._as_list(listened.get('needs')):
            if not isinstance(slot_item, dict) or not str(slot_item.get('need') or '').strip():
                continue
            digit = min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items))
            needful = str(slot_item['need']).strip()
            if needful not in self.items[digit - 1].relayed:
                self.items[digit - 1].relayed.append(needful)
                relayed.append(needful if not numerous else f'item {digit}: {needful}')
            if slot_item.get('points') is True and (digit, needful) not in self.cites:
                self.cites.append((digit, needful))
        for slot_item in _SchemaKit._as_list(listened.get('drop')):
            if isinstance(slot_item, dict) and str(slot_item.get('property') or '').strip():
                product = self.items[min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items)) - 1]
                departed = _TextKit._terms(str(slot_item['property']).split(':')[0])
                strike = [mw for mw in product.requisites + product.relayed if mw not in product.shed_off and _TextKit._terms(mw.split(':')[0]) == departed]
                product.shed_off.extend(strike)
                relayed.extend((f'dropped {mw}' for mw in strike))
        for slot_item in _SchemaKit._as_list(listened.get('free')):
            if isinstance(slot_item, dict) and slot_item.get('property'):
                digit = min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items))
                self.items[digit - 1].spare_one.append(str(slot_item['property']))
        if isinstance(listened.get('budget'), (int, float)) and (not isinstance(listened.get('budget'), bool)) and (listened['budget'] > 0):
            self.purse = listened['budget']

class _ValueHunt:

    def rebound_hunt(self, fetched: dict[str, Any] | None, wanting: list[int]) -> bool:
        retain = self.rear() + len(self.items) + 2
        leeway = min(REBOUND_PACES, self.remaining() - retain)
        if 3 > leeway or self.timing_remaining() < ROAM_SPARE + 25:
            return False
        prior = len(self.whole_rec)
        final = self.pace + leeway
        hunting = self.pace + max(1, leeway // 2)
        blueprint = self.hunt_pattern_tpl()
        inquiries: list[tuple[int, str]] = []
        for slot_item in _SchemaKit._as_list((fetched or {}).get('search')):
            if isinstance(slot_item, dict) and str(slot_item.get('query') or '').strip():
                inquiries.append((min(max(1, _SchemaKit._number_in(slot_item.get('item')) or 1), len(self.items)), str(slot_item['query']).strip()))
        inquiries = [qw for qw in inquiries if qw[0] in wanting] or [(nw, ' '.join(self.val_terms(nw)[:2] + [self.items[nw - 1].title_name])) for nw in wanting]
        spotted: dict[int, list[str]] = {nw: [] for nw in wanting}
        for digit in wanting[:3]:
            if hunting > self.pace:
                lid = None if not (isinstance(self.purse, (int, float)) and self.roam and self.roam['price']) else self.purse
                spotted[digit] += self.series_genre(digit, lid) or []
        for (digit, terms) in inquiries[:4]:
            if blueprint and hunting > self.pace:
                spotted.setdefault(digit, [])
                spotted[digit] += self.hunt_terms(blueprint[0], blueprint[1], blueprint[2], terms)
        for (digit, entry_rows) in spotted.items():
            chooses = [iw for iw in self.type_of_entry_rows(digit, entry_rows) if iw not in self.whole_rec][:2]
            self.pad_attrs(chooses[:max(0, final - self.pace)], retain)
        self.record_path.append({'kind': 'recovery', 'missing': wanting, 'queries': inquiries[:4], 'opened': len(self.whole_rec) - prior})
        return len(self.whole_rec) > prior

    def hunt_terms(self, utensil: str, footing: dict[str, Any], column: str, terms: str) -> list[str]:
        flaws: list[str] = []
        argset = _SchemaKit._form({**footing, column: terms}, self.layouts.get(utensil), 'args', flaws)
        if flaws or self.wrapped_up:
            return []
        response = self.do_step(utensil, argset)
        if response is None:
            return []
        self.notice(utensil, response.get('observation'))
        self.hunted = (self.hunted + [dict(argset)])[-8:]
        return [iw for (iw, _) in self.file_recs_inside(response.get('observation'))]

    def val_hunt(self, digits: list[int]) -> str:
        blueprint = self.hunt_pattern_tpl()
        if not blueprint or self.remaining() < 22 or self.timing_remaining() < ROAM_SPARE + 25:
            return ''
        (utensil, footing, column) = blueprint
        noticed: list[str] = []
        purse = 8
        for digit in digits:
            values = self.val_terms(digit)
            product = self.items[digit - 1]
            newly = [vw for vw in values if any((':' in tw and tw.split(':', 1)[1].strip() == vw for tw in product.relayed))]
            if not newly or 2 >= purse:
                continue
            declared = [iw for (iw, vw) in enumerate(values) if vw not in newly]
            shed = 0 if not declared else declared[-1]
            essays = [values] if not len(values) > 1 else [values, values[:shed] + values[shed + 1:]]
            for terms in essays[:2]:
                if not terms:
                    continue
                flaws: list[str] = []
                argset = _SchemaKit._form({**footing, column: ' '.join(terms)}, self.layouts.get(utensil), 'args', flaws)
                if flaws or self.wrapped_up:
                    break
                response = self.do_step(utensil, argset)
                purse -= 1
                if response is None:
                    break
                sighting = response.get('observation')
                self.notice(utensil, sighting)
                cls_attr_key = self.genre_attr_key()
                spotted = [iw for (iw, _) in self.file_recs_inside(sighting)]
                lid = None if not isinstance(self.purse, (int, float)) else self.purse
                fees = [self.fee_from_of(iw) for iw in spotted]
                if lid is not None and fees and all((pw is not None and pw > 10 * lid for pw in fees)):
                    lid = None
                entry_rows = [iw for iw in spotted if self.obtainable(iw) and (not cls_attr_key or self.holds_up(product, str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or ''))) and (lid is None or (self.fee_from_of(iw) or 0) <= lid)]
                noticed.append(f'{_TextKit._render(argset.get(column))}: ' + (', '.join(entry_rows[:6]) or 'no row of that kind'))
                if entry_rows:
                    unsealed = self.pad_attrs([iw for iw in entry_rows if iw not in self.whole_rec][:min(3, purse)], self.rear() + 16)
                    purse -= len(unsealed)
                    if unsealed:
                        noticed.append('opened: ' + ' | '.join(unsealed)[:2400])
                    break
        if not noticed:
            return ''
        self.record_path.append({'kind': 'value_search', 'seen': [sw[:200] for sw in noticed]})
        return "[Orla searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(noticed)

    def val_terms(self, digit: int) -> list[str]:
        outside = []
        for (nw, needful) in self.needfuls_at_present():
            if digit != nw or needful in [mw for (_, mw) in self.cites] or ':' not in needful:
                continue
            val = needful.split(':', 1)[1].strip()
            if not val or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            outside.append(val)
        return outside

    def hunt_pattern_tpl(self) -> tuple[str, dict[str, Any], str] | None:
        utensil = self.jobs.get('search')
        layout = self.layouts.get(utensil or '') or {}
        traits = layout.get('properties') or {}
        wordings = [kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'string' in _SchemaKit._types_of(vw) and ('enum' not in vw)]
        titled = [kw for kw in wordings if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', kw, re.I)]
        for argset in reversed(self.hunted):
            columns = [kw for kw in titled or wordings if isinstance(argset.get(kw), str) and re.search('[a-z]', argset[kw], re.I)]
            if not titled:
                tags = set() if not self.genre_attr_key() else {str(rw.get(self.genre_attr_key()) or '').strip().lower() for rw in self.entry_rows.values()} - {''}
                columns = [kw for kw in columns if len({str(aw.get(kw)) for aw in self.hunted if aw.get(kw)}) >= 2 and any((len(str(aw.get(kw)).split()) >= 2 for aw in self.hunted)) and (not any((str(aw.get(kw) or '').strip().lower() in tags for aw in self.hunted)))]
            if columns and utensil:
                column = columns[0]
                retain = {kw: vw for (kw, vw) in argset.items() if column != kw and (isinstance((traits.get(kw) or {}).get('enum'), list) or kw in (layout.get('required') or []))}
                cls_attr_key = self.genre_attr_key()
                racks = {_TextKit._terms(product.rack or '') for product in self.items} | {_TextKit._terms(product.title_name) for product in self.items}
                racks |= set() if not cls_attr_key else {_TextKit._terms(str(entry_row.get(cls_attr_key) or '')) for entry_row in self.entry_rows.values()}
                retain = {kw: vw for (kw, vw) in retain.items() if kw in (layout.get('required') or []) or not isinstance(vw, str) or (_TextKit._terms(vw) not in racks and (not any(({ww for ww in _TextKit._stalk(vw) if len(ww) >= 4} & {ww for ww in _TextKit._stalk(sw) if len(ww) >= 4} for sw in racks if sw))))}
                phrased = [aw for aw in self.hunted if isinstance(aw.get(column), str) and re.search('[a-z]', aw[column], re.I)]
                for kw in list(retain):
                    if isinstance((traits.get(kw) or {}).get('enum'), list):
                        spent_up = [_TextKit._render(aw[kw]) for aw in phrased if kw in aw]
                        if spent_up:
                            finest = max(spent_up, key=spent_up.count)
                            retain[kw] = next((aw[kw] for aw in phrased if kw in aw and _TextKit._render(aw[kw]) == finest))
                return (utensil, retain, column)
        return None

class _Questioning:

    def post(self, quiz_lines: list[tuple[int, str, str]], minimum: int=6) -> list[tuple[int, str, str]]:
        utensil = self.jobs['ask']
        posted: list[tuple[int, str, str]] = []
        for (digit, _, wording) in quiz_lines:
            if self.wrapped_up or self.remaining() <= minimum or self.mv_cutoff <= self.mv_spent_up:
                break
            argset = self.missive_argset(wording)
            if not argset:
                break
            response = self.do_step(utensil, argset)
            if response is None:
                break
            self.mv_spent_up += 1
            retort = response.get('observation')
            terms = self.voiced(retort)
            posted.append((digit, wording, terms))
            self.through_journal.append(f'step {self.pace}: message to the shopper {wording!r} -> reply {terms[:300]!r} (cart then: {self.trolley_tick()})')
            self.through_paces.append(self.pace)
            self.record_path.append({'kind': 'ask', 'q': wording, 'a': terms[:300]})
            if isinstance(retort, dict) and retort.get('error'):
                break
        return posted

    def quiz_line(self, digit: int, tag_r: str) -> str:
        return f'What {tag_r} would you like for the {self.items[digit - 1].title_name}?'

    def shaky(self, slot_item: Any) -> bool:
        if not isinstance(slot_item, dict):
            return False
        (trait, _, val) = str(slot_item.get('need') or '').partition(':')
        terms = set(re.findall('[a-z0-9]+', val.lower()))
        if slot_item.get('points') is True:
            return True
        if len(terms) < 2:
            return False
        noticed: set[str] = set()
        for file_rec in self.whole_rec.values():
            for (attr_key, presented_out) in (_RowKit._attr_slab(file_rec) or {}).items():
                if _TextKit._title_names(_TextKit._stalk(trait), _TextKit._stalk(attr_key)) or _TextKit._title_names(_TextKit._stalk(attr_key), _TextKit._stalk(trait)):
                    noticed |= set(re.findall('[a-z0-9]+', str(presented_out).lower()))
        return bool(terms - noticed)

    def blueprint_tracks(self, count: int) -> dict[int, list[str]]:
        board = self.attr_key_board()
        fixed_up = []
        for (digit, product) in enumerate(self.items, 1):
            fixed_up.append(f'item {digit}: {product.title_name} - settled by the request: ' + ('; '.join(product.requisites) or 'nothing'))
        entry_rec = []
        final = None
        for (index, (tag_r, title_name, borne, values)) in enumerate(board, 1):
            if final != tag_r:
                entry_rec.append(f'category {tag_r!r}:')
                final = tag_r
            entry_rec.append(f'  {index}. {title_name} (on {borne}): ' + ' | '.join(values) + self.non_declared(tag_r, title_name, values) + ('' if not re.search(GRAB_BAG, title_name) else ' [several features in one: ask last]'))
        prompt_text = QUIZ_BLUEPRINT.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.petition + '\n\nITEMS:\n' + '\n'.join(fixed_up)
        initial = [str(xw) for xw in _SchemaKit._as_list(self.digest.get('ask_first')) if str(xw).strip()]
        if initial:
            prompt_text += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(initial)
        prompt_text += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(entry_rec) or '(no full listing records yet)')
        blueprint = self.pose_json(prompt_text, [BIG_BRAIN, SMALL_BRAIN], 3500, profound=True) or {}
        tracks: dict[int, list[str]] = {nw: [] for nw in range(1, len(self.items) + 1)}
        cued: dict[int, set[str]] = {nw: set() for nw in tracks}
        for choose in _SchemaKit._as_list(blueprint.get('ask')):
            if not isinstance(choose, dict):
                continue
            digit = min(max(1, _SchemaKit._number_in(choose.get('item') or 1) or 1), len(self.items))
            tag_r = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(choose.get('label') or ''))[0].strip().rstrip('?.')
            if not tag_r or len(tag_r.split()) > 5:
                continue
            if all((_TextKit._terms(tag_r) != _TextKit._terms(another) for another in tracks[digit])):
                tracks[digit].append(tag_r)
            if str(choose.get('hint') or '').strip():
                cued[digit].add(_TextKit._terms(tag_r))
                if digit not in self.blueprint_purchase_order:
                    self.blueprint_purchase_order.append(digit)
        for digit in tracks:
            keys = self.track_attr_keys(digit, board)
            peruse = max(1, self.file_recs_from_of_product(digit))
            owned: dict[str, float] = {}
            for (tag_r, attr_key, borne, _) in board:
                if self.holder_from_of(tag_r) == digit:
                    owned[attr_key] = max(owned.get(attr_key, 0.0), borne / peruse)

            def spelt(tag_r: str) -> str:
                simple = lambda wording: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _TextKit._terms(wording)))
                return next((kw for kw in owned if simple(kw) == simple(tag_r)), tag_r)

            def portion_from_of(tag_r: str) -> float:
                if _TextKit._terms(tag_r) in owned:
                    return owned[_TextKit._terms(tag_r)]
                nearby = [sw for (kw, sw) in owned.items() if _TextKit._title_names(_TextKit._stalk(tag_r), _TextKit._stalk(kw)) or _TextKit._title_names(_TextKit._stalk(kw), _TextKit._stalk(tag_r))]
                return 0.0 if not nearby else max(nearby)
            cued[digit] = {_TextKit._terms(spelt(hw)) for hw in cued[digit]}
            track = []
            for choose in tracks[digit]:
                if all((_TextKit._terms(spelt(choose)) != _TextKit._terms(another) for another in track)):
                    track.append(spelt(choose))
            for attr_key in [kw for kw in keys if owned.get(kw, 0.0) >= SENTRY_SHARE and (not re.search(GRAB_BAG, kw))][:SENTRY_KEYS]:
                if any((_TextKit._terms(attr_key) == _TextKit._terms(another) for another in track[:3])):
                    continue
                frail = next((iw for (iw, another) in enumerate(track[:3]) if _TextKit._terms(another) not in cued[digit] and portion_from_of(another) + SENTRY_GAP <= owned.get(attr_key, 0.0)), None)
                if frail is None and len(track) >= 3:
                    continue
                set_down = [another for another in track if _TextKit._terms(another) != _TextKit._terms(attr_key)]
                if not (frail is not None and len(self.items) > 1):
                    set_down.insert(min(len(set_down), len(set_down) if not frail is not None else frail), attr_key)
                else:
                    relocated = set_down[frail]
                    set_down[frail] = attr_key
                    set_down.insert(min(3, len(set_down)), relocated)
                if any((_TextKit._terms(xw) in cued[digit] and all((_TextKit._terms(xw) != _TextKit._terms(yw) for yw in set_down[:3])) for xw in track[:3])):
                    continue
                track = set_down
            tracks[digit] = track
            for attr_key in keys:
                if all((_TextKit._terms(attr_key) != _TextKit._terms(another) for another in tracks[digit])):
                    tracks[digit].append(attr_key)
            tracks[digit] = [xw for xw in tracks[digit] if not re.search(GRAB_BAG, _TextKit._terms(xw))] + [xw for xw in tracks[digit] if re.search(GRAB_BAG, _TextKit._terms(xw))]
            if SCARCE_MISSIVES >= count and 3 <= peruse:
                certain = [xw for xw in tracks[digit] if _TextKit._terms(xw) in cued[digit] or portion_from_of(xw) >= 0.5]
                tracks[digit] = certain + [xw for xw in tracks[digit] if xw not in certain]
        self.record_path.append({'kind': 'plan', 'ask': _TextKit._snip(blueprint.get('ask'), 120), 'lanes': {nw: vw[:8] for (nw, vw) in tracks.items()}})
        return tracks

    def file_recs_from_of_product(self, digit: int) -> int:
        cls_attr_key = self.genre_attr_key()
        product = self.items[digit - 1]
        return sum((1 for file_rec in self.whole_rec.values() if cls_attr_key and self.holds_up(product, str(file_rec.get(cls_attr_key) or '')) and _RowKit._attr_slab(file_rec)))

    def holds_up(self, product: ShoppingLine, tag_r: str) -> bool:
        return bool(tag_r) and product in self.items and (self.holder_from_of(tag_r) == self.items.index(product) + 1)

    def holder_from_of(self, tag_r: str) -> int:
        if not tag_r:
            return 0
        terms = _TextKit._terms(tag_r)
        lengthy = _TextKit._bases(tag_r)
        (finest, upmost) = (0, 0.0)
        for (digit, product) in enumerate(self.items, 1):
            if terms == _TextKit._terms(product.rack or ''):
                grade = 3.0
            elif not terms == _TextKit._terms(product.title_name):
                my_own = _TextKit._bases(product.rack or '') | _TextKit._bases(product.title_name)
                grade = len(lengthy & my_own) / max(1, len(lengthy | my_own))
            else:
                grade = 2.0
            if upmost < grade:
                (finest, upmost) = (digit, grade)
        return finest

    def track_attr_keys(self, digit: int, board: list[tuple[str, str, int, list[str]]]) -> list[str]:
        product = self.items[digit - 1]
        fixed_up = [_TextKit._stalk(mw.split(':', 1)[0]) for mw in product.requisites]
        entry_rows = [(tag_r, attr_key, borne, len(values)) for (tag_r, attr_key, borne, values) in board if self.holder_from_of(tag_r) == digit]
        metered = {attr_key: sum((1 for vw in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', vw))) * 2 > len(values) for (_, attr_key, _, values) in board}
        if len(self.items) == 1 and (not entry_rows):
            entry_rows = [(tag_r, attr_key, borne, len(values)) for (tag_r, attr_key, borne, values) in board]
        entry_rows.sort(key=lambda rw: (-rw[2], metered.get(rw[1], False), -rw[3]))
        keys = [attr_key for (_, attr_key, _, _) in entry_rows if not any((_TextKit._title_names(trait, _TextKit._stalk(attr_key)) for trait in fixed_up)) and (not re.search(NON_TRAIT, attr_key))]
        return [attr_key for attr_key in keys if not re.search(GRAB_BAG, attr_key)] + [attr_key for attr_key in keys if re.search(GRAB_BAG, attr_key)]

    def following_pose(self, tracks: dict[int, list[str]], spotted: dict[int, int], expended: dict[int, int], spent_up: dict[int, set[str]]) -> tuple[int, list[str]]:
        initial = {nw: iw for (iw, nw) in enumerate(self.blueprint_purchase_order)}
        for digit in sorted(tracks, key=lambda nw: (spotted.get(nw, 0) > 0, expended.get(nw, 0), initial.get(nw, len(initial) + nw))):
            remaining = [tag_r for tag_r in tracks[digit] if _TextKit._terms(tag_r) not in spent_up[digit]]
            if remaining:
                return (digit, remaining[:1])
        return (0, [])

    def canvass(self) -> str:
        self.posed = True
        if self.mvs_spare_one() <= 0:
            return ''
        if self.timing_remaining() > ROAM_SPARE + 30:
            self.broaden()
        tracks = self.blueprint_tracks(self.mvs_spare_one())
        spotted = {nw: 0 for nw in tracks}
        expended = {nw: 0 for nw in tracks}
        spent_up: dict[int, set[str]] = {nw: set() for nw in tracks}
        trace_on: list[tuple[int, list[str]]] = []
        posted: list[tuple[int, str, str]] = []
        relayed: list[str] = []
        while self.mvs_spare_one() > 0 and (not self.wrapped_up) and (self.remaining() > 6) and (self.timing_remaining() > ROAM_SPARE + 20):
            if not trace_on:
                (digit, tags) = self.following_pose(tracks, spotted, expended, spent_up)
                if not tags:
                    break
            else:
                (digit, tags) = trace_on.pop(0)
            spent_up[digit].update((_TextKit._terms(tag_r) for tag_r in tags))
            lot = self.post([(digit, tags[0], self.quiz_line(digit, tags[0]))])
            if not lot:
                break
            posted += lot
            expended[digit] += 1
            held_before = [len(product.relayed) for product in self.items]
            listened = self.listen(lot, (SMALL_BRAIN, BIG_BRAIN))
            if self.assesses()[:1] == [BIG_BRAIN] and any((self.shaky(slot_item) for slot_item in _SchemaKit._as_list(listened.get('needs')))):
                listened = self.listen(lot, (BIG_BRAIN,)) or listened
            self.grab_listened(listened, relayed)
            newly = [nw for (nw, product) in enumerate(self.items, 1) if len(product.relayed) > held_before[nw - 1]]
            for nw in newly:
                spotted[nw] = spotted.get(nw, 0) + 1
            if not any(spotted.values()):
                break
            if newly or 1 not in [_SchemaKit._number_in(xw) for xw in _SchemaKit._as_list(listened.get('crowded'))]:
                continue
            front = tags[0].split()[-1]
            if len(tags[0].split()) >= 2 and _TextKit._terms(front) not in spent_up[digit]:
                trace_on.insert(0, (digit, [front]))
        spotted_products = [nw for nw in spotted if spotted[nw] > 0]
        hunted = '' if not spotted_products else self.val_hunt(spotted_products)
        if relayed:
            self.edition += 1
        row_lines = [f'Q: {qw}\nA: {aw}' for (_, qw, aw) in posted]
        abstract = "Orla's interviewer asked the shopper (each question used one step):\n" + '\n'.join(row_lines)
        abstract += '\nNew firm requirements from these answers: ' + ('none' if not relayed else '; '.join(relayed))
        abstract += '. Apply them to every pick (search again with the new values where needed).'
        directed = [needful for (_, needful) in self.cites if any((needful in tw for tw in relayed))]
        if directed:
            abstract += ' These take their value from another listing: ' + ' | '.join(directed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if hunted:
            abstract += '\n' + hunted
        if self.lined_up:
            abstract += '\n[notes seen meanwhile] ' + ' | '.join(self.lined_up)
            self.lined_up = []
        return abstract

    def missive_argset(self, wording: str) -> dict[str, Any] | None:
        title_name = self.jobs.get('ask')
        layout = self.layouts.get(title_name or '') or {}
        traits = layout.get('properties') or {}
        wordings = [kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'string' in _SchemaKit._types_of(vw) and ('enum' not in vw)]
        aim = next((kw for kw in layout.get('required') or [] if kw in wordings), None) or (None if not wordings else wordings[0])
        if not aim:
            return None
        argset: dict[str, Any] = {aim: wording}
        for attr_key in layout.get('required') or []:
            if attr_key not in argset:
                spec_sheet = traits.get(attr_key) or {}
                if not (isinstance(spec_sheet.get('enum'), list) and spec_sheet['enum']):
                    return None
                else:
                    argset[attr_key] = spec_sheet['enum'][0]
        flaws: list[str] = []
        argset = _SchemaKit._form(argset, layout, 'args', flaws)
        return argset if not flaws else None

    def set_up_toward_pose(self) -> bool:
        if self.posed or not self.jobs.get('ask') or self.mvs_spare_one() <= 0:
            return False
        if self.remaining() <= 8 or self.timing_remaining() < ROAM_SPARE + 40:
            return False
        unsealed = len(self.whole_rec)
        if unsealed >= 3 * len(self.items) + 1:
            return True
        return self.pace >= max(8, self.most_paces // 4)

class _Reading:

    def contrast_leeway(self) -> int:
        spec_sheet = ((self.layouts.get(self.jobs.get('compare') or '') or {}).get('properties') or {}).get(self.contrast_aim() or '') or {}
        if isinstance(spec_sheet.get('maxItems'), int):
            return spec_sheet['maxItems']
        if self.contrast_poor:
            return max(2, min(self.contrast_poor - 1, max(self.contrast_fine_one, 2)))
        return max(4, self.contrast_fine_one)

    def contrast_aim(self) -> str | None:
        layout = self.layouts.get(self.jobs.get('compare') or '') or {}
        traits = layout.get('properties') or {}
        aim = next((kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'array' in _SchemaKit._types_of(vw)), None)
        if not aim or any((aim != kw for kw in layout.get('required') or [])):
            return None
        return aim

    def merge_in_unseals(self, rings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        (unsealer, merger, aim) = (self.jobs.get('open'), self.jobs.get('compare'), self.sole_aim(self.jobs.get('open')))
        numerous = self.contrast_aim()
        if not unsealer or not merger or (not aim) or (not numerous) or (merger == unsealer):
            return rings
        if self.contrast_plain_kind():
            return rings
        chooses: list[tuple[int, str]] = []
        for (index, ring) in enumerate(rings):
            fun = ring.get('function') or {}
            argset = None if not isinstance(fun.get('arguments'), str) else _TextKit._json_thing(fun.get('arguments'))
            mark_id = str((argset or {}).get(aim) or '').strip()
            if fun.get('name') == unsealer and argset and (set(argset) == {aim}) and (mark_id in self.entry_rows) and (mark_id not in self.departed):
                chooses.append((index, mark_id))
        chooses = chooses[:self.contrast_leeway()]
        if len(chooses) < 2:
            return rings
        initial = chooses[0][0]
        outside = []
        for (index, ring) in enumerate(rings):
            if initial == index:
                outside.append({'id': ring['id'], 'type': 'function', 'function': {'name': merger, 'arguments': _TextKit._render({numerous: [iw for (_, iw) in chooses]})}})
            elif not index in {iw for (iw, _) in chooses}:
                outside.append(ring)
            else:
                outside.append({'id': ring['id'], 'joined': True})
        self.record_path.append({'kind': 'joined', 'ids': [iw for (_, iw) in chooses]})
        return outside

    def unseal_entry_rows(self, entry_rows: list[str]) -> bool:
        (numerous, merger) = (self.contrast_aim(), self.jobs.get('compare'))
        utensil = self.attrs_utensil()
        aim = self.sole_aim(utensil)
        if numerous and merger and (not self.contrast_plain_kind()):
            blueprint = [(merger, {numerous: entry_rows[:self.contrast_leeway()]})]
        elif not (utensil and aim):
            return False
        else:
            blueprint = [(utensil, {aim: iw}) for iw in entry_rows[:3]]
        if self.remaining() < len(blueprint) + 5:
            return False
        prior = len(self.whole_rec)
        for (title_name, argset) in blueprint:
            flaws: list[str] = []
            argset = _SchemaKit._form(argset, self.layouts.get(title_name), 'args', flaws)
            if flaws:
                return False
            response = self.do_step(title_name, argset)
            if response is None or self.wrapped_up:
                break
            self.notice(title_name, response.get('observation'))
        self.record_path.append({'kind': 'audit_open', 'rows': entry_rows, 'opened': len(self.whole_rec) - prior})
        return len(self.whole_rec) > prior

    def broaden_solely(self) -> None:
        utensil = self.attrs_utensil()
        aim = self.sole_aim(utensil)
        if not utensil or not aim:
            return
        desired = {ww for (_, needful) in self.needfuls_at_present() for ww in re.findall('[a-z0-9]{3,}', needful.split(':', 1)[-1].lower())}
        every = 4 if not len(self.items) <= 2 else BROADEN_EACH
        tracks: list[list[str]] = []
        fronts: list[int] = []
        for code_ids in self.product_bunches():
            unsealed = sum((1 for iw in code_ids if iw in self.whole_rec))
            heap = [iw for iw in code_ids if iw not in self.whole_rec and self.obtainable(iw)]
            if self.digest.get('pick') == 'cheapest':
                heap.sort(key=lambda iw: float('inf') if not self.fee_from_of(iw) is not None else self.fee_from_of(iw))
            heap.sort(key=lambda iw: -len(desired & set(re.findall('[a-z0-9]{3,}', _TextKit._render(self.entry_rows.get(iw) or {}).lower()))))
            tracks.append(heap[:max(0, every - unsealed)])
            fronts.append(max(0, 4 - unsealed))
        initial = [track[kw] for kw in range(4) for (track, front) in zip(tracks, fronts) if kw < min(front, len(track))]
        extra = [track[kw] for kw in range(every) for (track, front) in zip(tracks, fronts) if front <= kw < len(track)]
        lid = max(5, 2 * len(tracks) + 2)
        peruse = self.pad_attrs(initial[:lid][:8], 22)
        self.pad_attrs((initial[lid:] + extra)[:max(0, 10 - len(peruse))], 30)

    def broaden(self) -> None:
        utensil = self.jobs.get('compare')
        layout = self.layouts.get(utensil or '') or {}
        traits = layout.get('properties') or {}
        aim = next((kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'array' in _SchemaKit._types_of(vw)), None)
        if not utensil or not aim or any((aim != kw for kw in layout.get('required') or [])) or self.contrast_plain_kind():
            self.broaden_solely()
            return
        if self.remaining() <= 12:
            return
        lid = 4 if not isinstance(traits[aim].get('maxItems'), int) else traits[aim].get('maxItems')
        ordered_by = self.product_bunches()
        portion = max(1, lid // max(1, len(ordered_by)))
        chooses: list[str] = []
        for code_ids in ordered_by:
            chooses += [iw for iw in code_ids if iw not in self.whole_rec][:portion]
        chooses = chooses[:lid]
        if len(chooses) < 2:
            return
        flaws: list[str] = []
        argset = _SchemaKit._form({aim: chooses}, layout, 'args', flaws)
        if flaws:
            return
        prior = len(self.whole_rec)
        response = self.do_step(utensil, argset)
        if response is not None:
            noticed = response.get('observation')
            self.notice(utensil, noticed)
            self.jot_look(utensil, noticed)
            entry_rows = self.file_recs_inside(noticed)
            if entry_rows:
                self.attrs_via[utensil] = self.attrs_via.get(utensil, False) or any((_RowKit._attr_slab(rw) for (_, rw) in entry_rows))
            self.record_path.append({'kind': 'widen', 'ids': chooses, 'opened': len(self.whole_rec) - prior})
            if entry_rows and self.attrs_via.get(utensil) is False:
                self.broaden_solely()
                return
            if len(self.whole_rec) == prior and len(chooses) > 2 and (self.remaining() > 12):
                flaws = []
                argset = _SchemaKit._form({aim: chooses[:2]}, layout, 'args', flaws)
                if not flaws:
                    response = self.do_step(utensil, argset)
                    if response is not None:
                        self.notice(utensil, response.get('observation'))

    def unbidden_unseal(self, sightings: list[Any]) -> str:
        if self.remaining() < self.rear() + 40 or self.timing_remaining() < ROAM_SPARE + 60 or 12 <= self.unbidden_unsealed:
            return ''
        cls_attr_key = self.genre_attr_key()
        spotted = [iw for noticed in sightings for (iw, _) in self.file_recs_inside(noticed)]
        chooses: list[str] = []
        for (digit, product) in enumerate(self.items, 1):
            ongoing = {needful for (nw, needful) in self.needfuls_at_present() if digit == nw}
            if any((needful in ongoing and needful not in self.cite_vals for (nw, needful) in self.cites if digit == nw)):
                continue
            desired = {ww for ww in re.findall('[a-z0-9]{3,}', ' '.join(self.val_terms(digit)).lower())}
            if not desired:
                continue
            for mark_id in self.type_of_entry_rows(digit, spotted):
                terms = set(re.findall('[a-z0-9]{3,}', _TextKit._render(self.entry_rows.get(mark_id) or {}).lower()))
                if mark_id not in self.whole_rec and mark_id not in chooses and desired & terms and (not cls_attr_key or self.holds_up(product, str((self.entry_rows.get(mark_id) or {}).get(cls_attr_key) or ''))):
                    chooses.append(mark_id)
                if len(chooses) >= 3:
                    break
            if len(chooses) >= 3:
                break
        if not chooses:
            return ''
        unsealed = self.pad_attrs(chooses[:3], self.rear() + 30)
        self.unbidden_unsealed += len(unsealed)
        return '' if not unsealed else '\n[Orla opened the likeliest rows of this result for you (one step each)]\n' + '\n'.join(unsealed)

    def pad_attrs(self, code_ids: list[str], retain: int) -> list[str]:
        utensil = self.attrs_utensil()
        aim = self.sole_aim(utensil)
        if not utensil or not aim:
            return []
        wordings = []
        for mark_id in [iw for iw in code_ids if iw in self.entry_rows and iw not in self.whole_rec and (not self.soldout_outside(iw))]:
            if self.remaining() <= retain or self.wrapped_up or self.attrs_via.get(utensil) is False or (self.timing_remaining() < CALL_SPARE + 10):
                break
            flaws: list[str] = []
            argset = _SchemaKit._form({aim: mark_id}, self.layouts.get(utensil), 'args', flaws)
            if flaws:
                break
            response = self.do_step(utensil, argset)
            if response is None:
                break
            noticed = response.get('observation')
            self.notice(utensil, noticed)
            self.jot_look(utensil, noticed)
            if isinstance(noticed, dict) and self.mark_id(noticed):
                self.attrs_via[utensil] = self.attrs_via.get(utensil, False) or bool(_RowKit._attr_slab(noticed))
            wordings.append(f'{mark_id}: ' + self.present_r(noticed)[:1500])
        if wordings:
            self.record_path.append({'kind': 'fill_facts', 'n': len(wordings)})
        return wordings

class _Leafing:

    def extra_leaves_set(self, title_name: str, argset: dict[str, Any], sighting: Any) -> str:
        self.leafed_sight = []
        choose = self.digest.get('pick')
        blueprint = None if not choose in ('cheapest', 'most', 'least') else self.leafing(title_name, sighting)
        if not blueprint:
            return ''
        (count, presented_out, arg_slot, val) = blueprint
        handed = argset.get(arg_slot)
        afterward = bool(str(handed or '').strip()) if not (isinstance(handed, int) and (not isinstance(handed, bool))) else handed not in (None, 0, 1)
        if val is None or afterward or presented_out >= count:
            return ''
        if 'cheapest' == choose and self.fee_arranged(sighting):
            return ''
        leaves_set = -(-(count - presented_out) // presented_out)
        if 2 < leaves_set and self.leaves_set_peruse + 2 <= SELF_LEAVES and (self.remaining() - self.rear() - 2 >= 12):
            leaves_set = 2
        if 2 < leaves_set or self.leaves_set_peruse + leaves_set > SELF_LEAVES or self.remaining() - self.rear() - leaves_set < 12:
            return f"\n[Orla] This search matched {count} listings and showed {presented_out}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {arg_slot}={_TextKit._render(val)}."
        wordings = []
        for _ in range(leaves_set):
            flaws: list[str] = []
            ring = _SchemaKit._form({**argset, arg_slot: val}, self.layouts.get(title_name), 'args', flaws)
            if flaws:
                break
            response = self.do_step(title_name, ring)
            if response is None or self.wrapped_up:
                break
            self.leaves_set_peruse += 1
            noticed = response.get('observation')
            self.notice(title_name, noticed)
            self.leafed_sight.append(noticed)
            presented_out_wording = self.present_r(noticed)
            self.peruse_stash[title_name + _TextKit._render(ring)] = (self.edition, self.pace, presented_out_wording)
            wordings.append(f'{arg_slot}={_TextKit._render(val)}: {presented_out_wording}')
            past = self.leafing(title_name, noticed)
            val = None if not past else past[3]
            if val is None:
                break
        self.record_path.append({'kind': 'pages', 'tool': title_name, 'read': len(wordings)})
        balance = '' if not (val is not None and wordings) else f"\n[Orla] {count} listings matched and Orla has read {presented_out * (len(wordings) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest."
        return '' if not wordings else '\n[Orla read the next pages of this search itself, one step a page]\n' + '\n'.join(wordings) + balance

    def leafing(self, title_name: str, sighting: Any) -> tuple[int, int, str, Any] | None:
        form = self.peruse_form(title_name, sighting)
        if not form or not form['count'] or (not form['next']) or (not form['next_arg']) or (not isinstance(sighting, dict)):
            return None
        (entry_rows, count, val) = (sighting.get(form['rows']), sighting.get(form['count']), sighting.get(form['next']))
        if not isinstance(entry_rows, list) or not entry_rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if val is not None and (isinstance(val, bool) or not isinstance(val, (str, int))):
            return None
        return (count, len(entry_rows), form['next_arg'], val)

class _Prices:

    def thriftier_entry_rows(self) -> list[str]:
        if self.digest.get('pick') != 'cheapest':
            return []
        tick = self.trolley_tick() + f'#{self.edition}'
        cls_attr_key = self.genre_attr_key()
        if tick in self.thriftier_ticks or not cls_attr_key:
            return []
        self.thriftier_ticks.add(tick)
        trolley = self.trolley_code_ids()
        chooses: list[str] = []
        for row_line in trolley:
            fee = self.fee_from_of(row_line)
            type_of = str((self.entry_rows.get(row_line) or {}).get(cls_attr_key) or '').strip().lower()
            if fee is None or not type_of:
                continue
            heap = [iw for iw in self.noticed_purchase_order if iw not in trolley and iw not in self.whole_rec and self.obtainable(iw) and (str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or '').strip().lower() == type_of) and ((float('inf') if not self.fee_from_of(iw) is not None else self.fee_from_of(iw)) < fee)]
            heap.sort(key=lambda iw: self.fee_from_of(iw) or 0.0)
            chooses += [iw for iw in heap[:2 if not len(trolley) == 1 else THRIFT_UNSEALS] if iw not in chooses]
        return self.pad_attrs(chooses[:THRIFT_UNSEALS], self.rear() + 2)

    def thriftier_scan(self) -> bool:
        cls_attr_key = self.genre_attr_key()
        blueprint = self.hunt_pattern_tpl()
        tick = self.trolley_tick() + f'#{self.edition}'
        leeway = self.remaining() - self.rear() - len(self.items) - 3
        if tick in self.scanned or 1 > leeway or self.timing_remaining() < ROAM_SPARE + 25 or (not cls_attr_key):
            return False
        self.scanned.add(tick)
        cutoff = self.pace + min(SCAN_PACES, leeway)
        executed = False
        thriftier: list[str] = []
        for row_line in self.trolley_code_ids():
            fee = self.fee_from_of(row_line)
            tag_r = str((self.entry_rows.get(row_line) or {}).get(cls_attr_key) or '')
            digit = next((nw for (nw, product) in enumerate(self.items, 1) if tag_r and self.holds_up(product, tag_r)), 0)
            if fee is None or not digit or cutoff <= self.pace:
                continue
            spotted = self.series_genre(digit, fee * 0.9 if not 1 < fee else int(fee) - 1, 0 if not cutoff - self.pace >= 2 else 1)
            if spotted is not None:
                executed = True
            elif blueprint:
                footing = self.val_terms(digit)[:2]
                for another in self.another_vals(digit)[:3]:
                    if cutoff <= self.pace:
                        break
                    spotted = (spotted or []) + self.hunt_terms(blueprint[0], blueprint[1], blueprint[2], ' '.join(footing + [another, self.items[digit - 1].title_name]))
                    executed = True
            thriftier += [iw for iw in self.type_of_entry_rows(digit, spotted or [], fee) if row_line != iw and iw not in self.whole_rec]
        self.record_path.append({'kind': 'sweep', 'ran': executed, 'cheaper': thriftier[:6]})
        return executed

    def another_vals(self, digit: int) -> list[str]:
        board = self.attr_key_board()
        keys = self.track_attr_keys(digit, board)[:3]
        stated_out = ' '.join(self.val_terms(digit)).lower() + ' ' + self.items[digit - 1].title_name.lower()
        tracks = []
        for attr_key in keys:
            values = next((vw for (tag_r, title_name, _, vw) in board if attr_key == title_name and (len(self.items) == 1 or self.holder_from_of(tag_r) == digit)), [])
            tracks.append([vw for vw in values if vw and len(vw) <= 20 and (vw not in stated_out) and re.search('[a-z0-9]', vw)][:2])
        return [track[kw] for kw in range(2) for track in tracks if kw < len(track)]

    def broader_keep_hold(self) -> str:
        if self.broader_kept_hold or self.digest.get('pick') != 'cheapest':
            return ''
        self.broader_kept_hold = True
        if self.thriftier_scan() or self.remaining() < 14 + 2 * len(self.items):
            return ''
        row_lines = [f'{iw} at {self.fee_from_of(iw):g}' for iw in self.trolley_code_ids() if self.fee_from_of(iw) is not None]
        return f"Not run yet (no step used). {self.remaining()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + ('' if not row_lines else f" ({'; '.join(row_lines)})") + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def thriftiest_present_here(self, sightings: list[Any]) -> str:
        if self.digest.get('pick') != 'cheapest' or not self.fee_attr_key:
            return ''
        spotted = {iw for noticed in sightings for (iw, _) in self.file_recs_inside(noticed)}
        entry_rows = [(self.fee_from_of(iw), iw) for iw in spotted if self.obtainable(iw) and self.fee_from_of(iw) is not None]
        if len(entry_rows) < 2:
            return ''
        return '\n[Orla] buyable rows here, cheapest first: ' + ', '.join((f'{iw} ({pw:g})' for (pw, iw) in sorted(entry_rows)[:6]))

    def lid_overlooked(self, title_name: str, argset: dict[str, Any], sighting: Any) -> str:
        traits = (self.layouts.get(title_name) or {}).get('properties') or {}

        def feed(attr_key: str) -> bool:
            spec_sheet = {} if not isinstance(traits.get(attr_key), dict) else traits.get(attr_key)
            return bool(re.search('price|cost|amount|budget|spend|money', attr_key + ' ' + str(spec_sheet.get('description') or ''), re.I))
        lids = [kw for (kw, vw) in argset.items() if kw in traits and isinstance(vw, (int, float)) and (not isinstance(vw, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', kw, re.I) and feed(kw)]
        if not lids or not self.fee_attr_key:
            return ''
        lids.sort(key=lambda kw: argset[kw])
        lid = argset[lids[0]]
        beyond = [iw for (iw, _) in self.file_recs_inside(sighting) if (self.fee_from_of(iw) or 0) > lid]
        if not beyond:
            return ''
        style_modes = {kw: vw for (kw, vw) in argset.items() if isinstance(vw, str) and isinstance((traits.get(kw) or {}).get('enum'), list)}
        the_rest = {kw: [ow for ow in traits[kw]['enum'] if vw != ow] for (kw, vw) in style_modes.items()}
        cue = '' if not any(the_rest.values()) else '; list the category with ' + ', '.join((f'{kw}={_TextKit._render(ow[0])}' for (kw, ow) in the_rest.items() if ow)) + f' and the same {lids[0]} to see only cheaper listings'
        return f'\n[Orla] This search did not apply {lids[0]}={lid:g}: {len(beyond)} of its rows cost more{cue}.'

    def fee_arranged(self, sighting: Any) -> bool:
        fees = [self.fee_from_of(iw) for (iw, _) in self.file_recs_inside(sighting)]
        return len(fees) >= 3 and None not in fees and all((bw >= aw for (aw, bw) in zip(fees, fees[1:])))

    def obtainable(self, mark_id: str) -> bool:
        if self.soldout_outside(mark_id):
            return False
        val = None if not self.inventory_attr_key else (self.entry_rows.get(mark_id) or {}).get(self.inventory_attr_key)
        return not (val is False or (isinstance(val, (int, float)) and (not isinstance(val, bool)) and (0 >= val)))

    def fee_from_of(self, mark_id: str) -> float | None:
        val = None if not self.fee_attr_key else (self.entry_rows.get(mark_id) or {}).get(self.fee_attr_key)
        if isinstance(val, dict):
            val = next((vw for vw in val.values() if isinstance(vw, (int, float)) and (not isinstance(vw, bool))), None)
        if isinstance(val, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', val):
            val = float(val)
        return None if not (isinstance(val, (int, float)) and (not isinstance(val, bool))) else float(val)

class _KeySheet:

    @staticmethod
    def doubles(title_name: str, spec_sheet: dict[str, Any], another: str, niche: dict[str, Any]) -> bool:
        (aw, bw) = (set(title_name.split()), set(another.split()))
        if not aw or not bw or (not aw & bw):
            return False

        def concur(xw: str, yw: str) -> bool:
            (wxa, wya) = (re.findall('[a-z0-9]+', xw.lower()), re.findall('[a-z0-9]+', yw.lower()))
            (brief_one, lengthy) = (wya, wxa) if not len(wxa) <= len(wya) else (wxa, wya)
            return yw == xw or (bool(brief_one) and set(brief_one) <= set(lengthy) and (len(lengthy) <= 2 * len(brief_one) + 2))
        pair_of = [iw for iw in spec_sheet['per'] if iw in niche['per'] and spec_sheet['per'][iw].strip() and niche['per'][iw].strip()]
        identical = sum((1 for iw in pair_of if concur(spec_sheet['per'][iw], niche['per'][iw])))
        if not (bw >= aw or aw >= bw):
            return len(pair_of) >= 2 and identical == len(pair_of)
        if pair_of:
            return identical * 2 > len(pair_of)
        return bool((set(spec_sheet['values']) & set(niche['values'])) - {''})

    def non_declared(self, tag_r: str, title_name: str, values: list[str]) -> str:
        digit = self.holder_from_of(tag_r)
        if not digit:
            return ''
        for demand in self.items[digit - 1].requisites:
            (trait, _, val) = demand.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', val, re.I):
                continue
            terms = set(re.findall('[a-z0-9]{3,}', val.lower()))
            presented_out = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if trait.strip() and _TextKit._terms(trait) != title_name and _TextKit._stalk(trait) & _TextKit._stalk(title_name) and terms and (not presented_out >= terms):
                return f' [not the stated {trait.strip()}: another property]'
        return ''

    def attr_key_board(self) -> list[tuple[str, str, int, list[str]]]:
        cls_attr_key = self.genre_attr_key()
        groups: dict[str, dict[str, Any]] = {}
        for mark_id in self.noticed_purchase_order:
            file_rec = self.whole_rec.get(mark_id)
            if not file_rec:
                continue
            slab = _RowKit._attr_slab(file_rec) or {}
            tag_r = 'listings' if not cls_attr_key else str(file_rec.get(cls_attr_key) or 'listings')
            group = groups.setdefault(tag_r, {'n': 0, 'keys': {}})
            group['n'] += 1
            through: set[str] = set()
            for (attr_key, val) in slab.items():
                title_name = _TextKit._terms(attr_key)
                if not title_name or title_name in through:
                    continue
                through.add(title_name)
                niche = group['keys'].setdefault(title_name, {'n': 0, 'values': [], 'per': {}})
                niche['n'] += 1
                wording = str(val).strip().lower()
                niche['per'][mark_id] = wording
                if wording not in niche['values']:
                    niche['values'].append(wording)
        desired = [(_TextKit._terms(product.rack or ''), _TextKit._terms(product.title_name), {ww for ww in _TextKit._stalk(product.rack or '') | _TextKit._stalk(product.title_name) if len(ww) >= 4}) for product in self.items]
        paired_up = {tag_r for tag_r in groups if any((_TextKit._terms(tag_r) in (rack, title_name) or {ww for ww in _TextKit._stalk(tag_r) if len(ww) >= 4} & lengthy for (rack, title_name, lengthy) in desired)) or self.holder_from_of(tag_r)}
        if paired_up:
            groups = {tag_r: group for (tag_r, group) in groups.items() if tag_r in paired_up}
        board = []
        for (tag_r, group) in sorted(groups.items(), key=lambda pair_kv: -pair_kv[1]['n']):
            retained: list[tuple[str, dict[str, Any]]] = []
            for (title_name, niche) in sorted(group['keys'].items(), key=lambda pair_kv: (-pair_kv[1]['n'], len(pair_kv[0]))):
                values = list(niche['per'].values())
                if not values:
                    continue
                avg = sum((len(vw) for vw in values)) / len(values)
                if 40 < avg or sum((1 for vw in values if ' / ' in vw)) * 2 >= len(values):
                    continue
                repeat = sum((len(set(re.findall('[a-z]{3,}', vw)) & set(re.findall('[a-z]{3,}', ' '.join((ow for ow in values if ow is not vw))))) > 0 for vw in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (12 < avg) and (repeat * 2 < len(values)):
                    continue
                double = next((spec_sheet for (another, spec_sheet) in retained if self.doubles(another, spec_sheet, title_name, niche)), None)
                if double is not None:
                    for (mark_id, wording) in niche['per'].items():
                        double['per'].setdefault(mark_id, wording)
                        if wording not in double['values']:
                            double['values'].append(wording)
                    double['n'] = len(double['per'])
                    continue
                retained.append((title_name, niche))
            retained.sort(key=lambda pair_kv: -pair_kv[1]['n'])
            for (title_name, niche) in retained[:14]:
                board.append((tag_r, title_name, niche['n'], [vw[:40] for vw in niche['values'][:4]]))
        return board

class _Shelving:

    def type_of_entry_rows(self, digit: int, spotted: list[str], beneath: float | None=None) -> list[str]:
        cls_attr_key = self.genre_attr_key()
        product = self.items[digit - 1]
        desired = {ww for ww in re.findall('[a-z0-9]{3,}', ' '.join(self.val_terms(digit)).lower())}
        entry_rows = [iw for iw in dict.fromkeys(spotted) if self.obtainable(iw) and (not cls_attr_key or self.holds_up(product, str((self.entry_rows.get(iw) or {}).get(cls_attr_key) or '')))]
        if beneath is not None:
            entry_rows = [iw for iw in entry_rows if (beneath if not self.fee_from_of(iw) is not None else self.fee_from_of(iw)) < beneath]
        return sorted(entry_rows, key=lambda iw: (iw in self.whole_rec, -len(desired & set(re.findall('[a-z0-9]{3,}', _TextKit._render(self.entry_rows.get(iw) or {}).lower()))), self.fee_from_of(iw) or 0))

    def series_genre(self, digit: int, lid: float | None, extra: int=0) -> list[str] | None:
        spec_sheet = self.roam
        tag_r = self.rack_from_of(digit)
        if not spec_sheet or not tag_r or 2 <= self.roam_lapses or (lid is not None and (not spec_sheet['price'])):
            return None
        argset = {**spec_sheet['fixed'], spec_sheet['category']: tag_r}
        if lid is not None:
            argset[spec_sheet['price']] = lid
        flaws: list[str] = []
        argset = _SchemaKit._form(argset, self.layouts.get(spec_sheet['tool']), 'args', flaws)
        if flaws or self.wrapped_up:
            self.roam_lapses += 1
            return None
        response = self.do_step(spec_sheet['tool'], argset)
        if response is None:
            self.roam_lapses += 1
            return None
        noticed = response.get('observation')
        self.notice(spec_sheet['tool'], noticed)
        spotted = [iw for (iw, _) in self.file_recs_inside(noticed)]
        if isinstance(noticed, dict) and noticed.get('error'):
            self.roam_lapses += 1
            return None
        for _ in range(extra):
            leaf = self.leafing(spec_sheet['tool'], noticed)
            if not leaf or leaf[3] is None or self.wrapped_up:
                break
            flaws = []
            ring = _SchemaKit._form({**argset, leaf[2]: leaf[3]}, self.layouts.get(spec_sheet['tool']), 'args', flaws)
            response = None if flaws else self.do_step(spec_sheet['tool'], ring)
            if response is None:
                break
            noticed = response.get('observation')
            self.notice(spec_sheet['tool'], noticed)
            spotted += [iw for (iw, _) in self.file_recs_inside(noticed)]
        return spotted

    def rack_from_of(self, digit: int) -> str:
        cls_attr_key = self.genre_attr_key()
        product = self.items[digit - 1]
        tags = [] if not cls_attr_key else [str(rw.get(cls_attr_key) or '') for rw in self.entry_rows.values()]
        tags = [tag_r for tag_r in tags if tag_r and self.holds_up(product, tag_r)]
        return product.rack if not tags else max(set(tags), key=tags.count)

    def sole_aim(self, utensil: str | None) -> str | None:
        layout = self.layouts.get(utensil or '') or {}
        traits = layout.get('properties') or {}
        wordings = [kw for (kw, vw) in traits.items() if isinstance(vw, dict) and 'string' in _SchemaKit._types_of(vw) and ('enum' not in vw)]
        aim = next((kw for kw in layout.get('required') or [] if kw in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        if not utensil or not aim or any((aim != kw for kw in layout.get('required') or [])):
            return None
        return aim

    def product_bunches(self) -> list[list[str]]:
        cls_attr_key = self.genre_attr_key()
        groups: dict[str, list[str]] = {}
        for mark_id in self.noticed_purchase_order:
            tag_r = '' if not cls_attr_key else str(self.entry_rows[mark_id].get(cls_attr_key) or '')
            groups.setdefault(tag_r, []).append(mark_id)
        desired = [(_TextKit._terms(product.rack or ''), _TextKit._terms(product.title_name), {ww for ww in _TextKit._stalk(product.rack or '') | _TextKit._stalk(product.title_name) if len(ww) >= 4}) for product in self.items]
        paired_up = [code_ids for (tag_r, code_ids) in groups.items() if any((_TextKit._terms(tag_r) in (rack, title_name) or {ww for ww in _TextKit._stalk(tag_r) if len(ww) >= 4} & lengthy for (rack, title_name, lengthy) in desired)) or self.holder_from_of(tag_r)]
        heap = paired_up or list(groups.values())
        return sorted(heap, key=lambda code_ids: -len(code_ids))[:max(1, len(self.items))]

    def genre_attr_key(self) -> str | None:
        numbers_of: dict[str, set[str]] = {}
        exhibit: dict[str, int] = {}
        for entry_row in self.entry_rows.values():
            for (attr_key, val) in entry_row.items():
                if isinstance(val, str) and 0 < len(val) <= 40 and (not re.search('\\d{3}', val)):
                    numbers_of.setdefault(attr_key, set()).add(val)
                    exhibit[attr_key] = exhibit.get(attr_key, 0) + 1
        sum_total = max(1, len(self.entry_rows))
        learned = self.tags()
        finest = None
        for (attr_key, values) in numbers_of.items():
            if exhibit[attr_key] < 0.8 * sum_total or len(values) > 40:
                continue
            enumerated = sum((1 for vw in values if vw.strip().lower() in learned)) / len(values)
            avg = sum((len(vw) for vw in values)) / len(values)
            grade = (-enumerated, len(values), -exhibit[attr_key], -avg)
            if finest is None or grade < finest[0]:
                finest = (grade, attr_key)
        return None if not finest else finest[1]

    def tags(self) -> set[str]:
        spotted: set[str] = set()

        def stroll(layout: Any, nesting: int) -> None:
            if 6 < nesting or not isinstance(layout, dict):
                return
            for choice_r in layout.get('enum') or []:
                if isinstance(choice_r, str):
                    spotted.add(choice_r.strip().lower())
            for val in layout.values():
                if isinstance(val, dict):
                    stroll(val, nesting + 1)
        for layout in self.layouts.values():
            stroll(layout, 0)
        return spotted

class _RowMemory:

    def peruse_form(self, title_name: str, sighting: Any) -> dict[str, Any] | None:
        if title_name in self.forms:
            return self.forms[title_name]
        if not isinstance(sighting, dict):
            return None
        serieses = [kw for (kw, vw) in sighting.items() if isinstance(vw, list) and len(vw) >= 2 and all((isinstance(rw, dict) for rw in vw))]
        if not serieses:
            return None
        self.forms[title_name] = None
        sample = {kw: vw if not kw in serieses else vw[:3] for (kw, vw) in sighting.items()}
        prompt_text = FORM_SPEC + '\n\nTOOL: ' + title_name + ' | parameters: ' + _TextKit._render(self.layouts.get(title_name) or {}) + '\n\nRESULT (first rows only):\n' + _TextKit._render(_TextKit._snip(sample, 120))[:4000]
        fetched = self.pose_json(prompt_text, [SMALL_BRAIN, BIG_BRAIN], 2500) or {}
        entry_rows = None if not fetched.get('rows') in serieses else fetched.get('rows')
        if not entry_rows:
            return None
        initial = sighting[entry_rows][0]
        traits = (self.layouts.get(title_name) or {}).get('properties') or {}

        def titled(attr_key: Any, wherever: dict[str, Any]) -> str | None:
            return None if not (isinstance(attr_key, str) and attr_key in wherever) else attr_key
        form = {'rows': entry_rows, 'count': titled(fetched.get('count'), sighting), 'next': titled(fetched.get('next'), sighting), 'next_arg': titled(fetched.get('next_arg'), traits), 'price': titled(fetched.get('price'), initial), 'stock': titled(fetched.get('stock'), initial)}
        if form['count'] and (isinstance(sighting[form['count']], bool) or not isinstance(sighting[form['count']], int)):
            form['count'] = None
        self.forms[title_name] = form
        self.fee_attr_key = self.fee_attr_key or form['price']
        self.inventory_attr_key = self.inventory_attr_key or form['stock']
        self.record_path.append({'kind': 'shape', 'tool': title_name, 'shape': form})
        return form

    def presented_out_so_far(self, title_name: str, mark_id: str) -> bool:
        (edition, own_it) = self.look_attr_keys.get(mark_id, (-1, set()))
        keys = self.utensil_attr_keys.get(title_name)
        return bool(keys) and self.edition == edition and (own_it >= keys) and (mark_id not in self.departed)

    def jot_look(self, title_name: str, sighting: Any) -> None:
        for (mark_id, file_rec) in self.file_recs_inside(sighting):
            keys = set(file_rec)
            self.utensil_attr_keys.setdefault(title_name, set()).update(keys)
            (edition, own_it) = self.look_attr_keys.get(mark_id, (-1, set()))
            self.look_attr_keys[mark_id] = (self.edition, (set() if not self.edition == edition else own_it) | keys)

    def file_recs_inside(self, sighting: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(sighting, dict):
            mark_id = self.mark_id(sighting)
            if mark_id:
                return [(mark_id, sighting)]
            spotted: list[tuple[str, dict[str, Any]]] = []
            for val in sighting.values():
                if isinstance(val, list):
                    spotted += [(iw, rw) for rw in val if isinstance(rw, dict) for iw in [self.mark_id(rw, True)] if iw]
                elif isinstance(val, dict):
                    spotted += [(iw, rw) for vw in val.values() if isinstance(vw, list) for rw in vw if isinstance(rw, dict) for iw in [self.mark_id(rw, True)] if iw]
            return spotted
        if isinstance(sighting, list):
            return [(iw, rw) for rw in sighting if isinstance(rw, dict) for iw in [self.mark_id(rw, True)] if iw]
        return []

    def soak_up(self, sighting: Any) -> None:

        def stroll(unit: Any, nesting: int, inside_series: bool=False) -> None:
            if 6 < nesting:
                return
            if isinstance(unit, dict):
                mark_id = self.mark_id(unit, inside_series, nesting)
                if mark_id:
                    if mark_id not in self.entry_rows:
                        self.noticed_purchase_order.append(mark_id)
                        self.entry_rows[mark_id] = {}
                    prior_one = self.entry_rows[mark_id]
                    if any((prior_one.get(kw) is True and vw is False for (kw, vw) in unit.items())):
                        self.not_offered.add(mark_id)
                    elif any((prior_one.get(kw) is False and vw is True for (kw, vw) in unit.items())):
                        self.not_offered.discard(mark_id)
                    self.entry_rows[mark_id].update(unit)
                    if _RowKit._attr_slab(unit):
                        self.whole_rec[mark_id] = dict(self.entry_rows[mark_id])
                    elif mark_id in self.whole_rec:
                        self.whole_rec[mark_id].update(unit)
                    return
                for val in unit.values():
                    stroll(val, nesting + 1)
            elif isinstance(unit, list):
                entry_rows = self.study_entry_rows(unit)
                for val in unit:
                    stroll(val, nesting + 1, entry_rows)
        stroll(sighting, 0)

    def study_entry_rows(self, unit: list[Any]) -> bool:
        entry_rows = [nw for nw in unit if isinstance(nw, dict) and len(nw) >= 3]
        if not entry_rows or len(entry_rows) < len(unit) / 2:
            return False
        if self.code_id_attr_key and all((_RowKit._code_id_resembling(rw.get(self.code_id_attr_key), True) for rw in entry_rows)):
            return True
        spotted = _RowKit._study_code_id(entry_rows)
        if spotted:
            self.code_id_attr_key = spotted
            return True
        return False

    def mark_id(self, unit: Any, inside_series: bool=False, nesting: int=0) -> str | None:
        if not isinstance(unit, dict) or len(unit) < 3:
            return None
        if self.code_id_attr_key and _RowKit._code_id_resembling(unit.get(self.code_id_attr_key), True):
            return unit[self.code_id_attr_key].strip()
        nestled = any((isinstance(vw, dict) for vw in unit.values()))
        if not (inside_series or 0 == nesting or nestled):
            return None
        spotted = _RowKit._study_code_id([unit])
        if spotted and ('id' in str(spotted).lower() or unit[spotted].strip() in self.entry_rows):
            return unit[spotted].strip()
        return None

class _Register:

    def present_r(self, sighting: Any) -> str:
        wording = _TextKit._render(_TextKit._snip(sighting, PRESENT_CHARS))
        if len(wording) > SIGHT_CUTOFF:
            wording = wording[:SIGHT_CUTOFF] + '...(cut)'
        return wording

    def proofs(self, sighting: Any) -> str:
        code_ids: list[str] = []

        def stroll(unit: Any, nesting: int) -> None:
            if 6 < nesting:
                return
            if isinstance(unit, dict):
                mark_id = self.mark_id(unit, False, nesting)
                if mark_id and _RowKit._attr_slab(unit) and (mark_id not in code_ids):
                    code_ids.append(mark_id)
                    return
                for val in unit.values():
                    stroll(val, nesting + 1)
            elif isinstance(unit, list):
                for val in unit:
                    stroll(val, nesting + 1)
        stroll(sighting, 0)
        needfuls = self.needfuls_at_present()
        if not code_ids or not needfuls:
            return ''
        numerous = len(self.items) > 1
        row_lines = []
        for mark_id in code_ids[:12]:
            slab = _RowKit._attr_slab(self.whole_rec.get(mark_id) or {}) or {}
            pieces = []
            for (digit, needful) in needfuls[:8]:
                trait = needful.split(':', 1)[0].strip()
                strikes = [f'{kw}={str(vw)[:50]}' for (kw, vw) in slab.items() if _TextKit._title_names(_TextKit._stalk(trait), _TextKit._stalk(kw))]
                badge = '' if not numerous else f'(item {digit}) '
                if strikes:
                    pieces.append(f'{badge}{trait} -> ' + ', '.join(strikes[:3]))
            if pieces:
                row_lines.append(f'{mark_id}: ' + '; '.join(pieces))
        if not row_lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(row_lines)

    def needfuls_at_present(self) -> list[tuple[int, str]]:
        outside = []
        for (digit, product) in enumerate(self.items, 1):
            for needful in product.requisites + product.relayed:
                if needful not in product.shed_off:
                    outside.append((digit, needful))
        return outside

    def standing(self) -> str:
        remaining = self.remaining()
        wording = f'[Orla] step {self.pace} of {self.most_paces} used, {remaining} left.'
        if remaining <= self.rear() + 1:
            wording += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif self.plain_kind_products() and remaining <= len(self.items) + 5:
            wording += f' Few steps left and {self.plain_kind_products()} item(s) still have no cart line (an order missing one fails): put the best fitting listing you have read for each in the cart, opening another only if none fits, then check out.'
        elif 6 >= remaining or (remaining <= len(self.items) + 5 and 0 == self.purchases):
            wording += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif 0 == self.purchases and 12 >= remaining:
            wording += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif 0 == self.purchases and (not self.trolley_handled) and (self.pace >= self.most_paces // 2):
            wording += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        ongoing = {needful for (_, needful) in self.needfuls_at_present()}
        fetches_up = [needful for (_, needful) in self.cites if needful in ongoing and needful not in self.cite_vals]
        if fetches_up and 0 == self.purchases:
            wording += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(fetches_up[:3])
        if self.pointers_r:
            wording += '\n[review candidates] ' + ', '.join(self.pointers_r)
        blueprint = '' if not (self.edge_val and 0 == self.purchases) else self.match_up_blueprint()
        if blueprint:
            wording += '\n' + blueprint
        if self.jots:
            wording += '\n[changes so far] ' + ' | '.join(self.jots[-4:])
        if self.timing_remaining() < ROAM_SPARE + 45:
            wording += f'\n[time] about {max(0, int(self.timing_remaining()))} s of this episode are left: settle on your picks and check out within the next few calls.'
        wording += '\n[ledger]\n' + self.register()
        return wording

    def register(self) -> str:
        pieces = []
        numerous = len(self.items) > 1
        for (digit, product) in enumerate(self.items, 1):
            ongoing = [mw for mw in product.requisites + product.relayed if mw not in product.shed_off]
            relayed = [mw for mw in ongoing if self.portrayed(digit, mw)]
            needfuls = [mw + ('' if not mw in self.cite_vals else f' (the page it points at gives: {self.cite_vals[mw]})') for mw in ongoing if mw not in relayed]
            front = f'item: {product.title_name}' if not numerous else f'item {digit}: {product.title_name}'
            tag_r = product.rack or self.rack_from_of(digit)
            if tag_r:
                front += f' (store label: {tag_r})'
            front += f', quantity {product.howmany}'
            wording = front + '; requirements: ' + ('none beyond the kind of product' if not needfuls else '; '.join(needfuls))
            if relayed:
                wording += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(relayed)
            if product.spare_one:
                wording += '; shopper does not mind: ' + ', '.join(product.spare_one)
            pieces.append(wording)
        digest = self.digest
        if self.purse is not None:
            range_of = 'in total' if not digest.get('budget_scope') == 'each' else 'for each item'
            pieces.append(f"budget: at most {self.purse} {digest.get('budget_unit') or ''} {range_of}".replace('  ', ' '))
        choose = digest.get('pick')
        if 'cheapest' == choose:
            pieces.append('pick: the cheapest listing that fits (lowest price)')
        elif not choose in ('most', 'least'):
            pieces.append('pick: any listing that fits everything')
        else:
            pieces.append(f"pick: the {choose} {digest.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        if digest.get('in_stock_only', True) is not False:
            pieces.append('in stock only')
        for yearning in _SchemaKit._as_list(digest.get('wishes')):
            pieces.append(f'wish (tie-breaker only): {yearning}')
        for regulation in _SchemaKit._as_list(digest.get('rules')):
            pieces.append(f'rule from the request: {regulation}')
        if digest.get('quiet_decline') is True:
            pieces.append('if nothing fits: end without naming or suggesting any listing')
        for slot_item in self.procedure:
            whenever = {'choose': 'before choosing', 'order': 'at the order (Orla does it when you check out)', 'end': 'at an ending without a purchase (Orla does it then)'}[slot_item['when']]
            pieces.append(f"step the request asks for, {whenever}: {slot_item['step']}")
        attrs = self.attrs_utensil()
        if attrs and (self.attrs_via.get(attrs) is True or self.jobs.get('offer')):
            plain_kind = sorted((tw for tw in self.leaf_utensils() | {self.jobs.get('offer') or ''} if tw and self.attrs_via.get(tw) is False))
            leaves_set = sorted((tw for tw in self.leaf_utensils() if attrs != tw))
            pieces.append(f'listing attributes come from {attrs}' + ('' if not plain_kind else f"; {', '.join(plain_kind)} show none") + ('' if not leaves_set else f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(leaves_set)}"))
        return '\n'.join(('- ' + pw for pw in pieces))

class _TaskReading:

    def portrayed(self, digit: int, needful: str) -> bool:
        if needful in [mw for (_, mw) in self.cites] or needful in [str(xw) for xw in _SchemaKit._as_list(self.digest.get('lookups'))]:
            return False
        (trait, _, val) = needful.partition(':')
        if not trait.strip() or len(re.findall('[a-z0-9]+', val.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', val, re.I):
            return False
        product = self.items[digit - 1]
        if any(({ww for ww in _TextKit._stalk(another.title_name) if len(ww) >= 4} & _TextKit._stalk(val) for another in self.items if another is not product)):
            return False
        cls_attr_key = self.genre_attr_key()
        keys: set[str] = set()
        peruse = 0
        for file_rec in self.whole_rec.values():
            if cls_attr_key and (not self.holds_up(product, str(file_rec.get(cls_attr_key) or ''))):
                continue
            slab = _RowKit._attr_slab(file_rec) or {}
            if slab:
                peruse += 1
                keys |= {_TextKit._terms(kw) for kw in slab}
        return 3 <= peruse and (not any((_TextKit._stalk(kw) & _TextKit._stalk(trait) for kw in keys)))

    def disentangle(self) -> None:
        for (digit, product) in enumerate(self.items, 1):
            directions = [needful for (nw, needful) in self.cites if digit == nw and needful in product.requisites]
            balance = [mw for mw in product.requisites if mw not in directions]
            if not directions or not balance:
                continue
            prompt_text = UNTANGLE_SPEC + '\n\nREQUEST:\n' + self.petition + f'\n\nPRODUCT: {product.title_name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(directions) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(balance)
            fetched = self.pose_json(prompt_text, [BIG_BRAIN, SMALL_BRAIN], 1500) or {}
            relocated = [mw for mw in balance if mw in [str(xw).strip() for xw in _SchemaKit._as_list(fetched.get('describe_other'))]]
            if not relocated:
                continue
            product.requisites = [mw for mw in product.requisites if mw not in relocated]
            merged_in = directions[0] + ' (that listing: ' + '; '.join(relocated) + ')'
            product.requisites = [mw if not mw == directions[0] else merged_in for mw in product.requisites]
            self.cites = [(nw, needful if not (nw, needful) == (digit, directions[0]) else merged_in) for (nw, needful) in self.cites]
            self.record_path.append({'kind': 'unmix', 'item': digit, 'moved': relocated})

    @staticmethod
    def pooled_rack(digest: dict[str, Any]) -> bool:
        racks = [_TextKit._terms(str(ew.get('shelf') or '')) for ew in _SchemaKit._as_list(digest.get('items')) if isinstance(ew, dict)]
        racks = [sw for sw in racks if sw]
        return len(racks) != len(set(racks))

    def peruse_roam(self, spec_sheet: Any) -> None:
        if not isinstance(spec_sheet, dict) or spec_sheet.get('tool') not in self.layouts:
            return
        traits = self.layouts[spec_sheet['tool']].get('properties') or {}
        (genre, fee) = (spec_sheet.get('category'), spec_sheet.get('max_price'))
        if genre not in traits:
            return
        mended = {} if not isinstance(spec_sheet.get('fixed'), dict) else spec_sheet.get('fixed')
        self.roam = {'tool': spec_sheet['tool'], 'category': genre, 'price': None if not fee in traits else fee, 'fixed': {kw: vw for (kw, vw) in mended.items() if kw in traits and kw not in (genre, fee)}}

    def peruse_petition(self) -> None:
        row_lines = []
        for utensil in self.utensils:
            fun = utensil['function']
            row_lines.append(f"- {fun['name']}: {fun.get('description', '')} | parameters: {_TextKit._render(fun.get('parameters') or {})}")
        prompt_text = TASK_DIGEST + '\nTOOLS:\n' + '\n'.join(row_lines) + '\n\nREQUEST:\n' + self.petition
        if self.look.get('opening') is not None:
            prompt_text += '\n\nOPENING NOTE FROM THE STORE:\n' + _TextKit._render(_TextKit._snip(self.look.get('opening'), 1500))
        digest = self.pose_json(prompt_text, [SMALL_BRAIN], 4000) or {}
        if not self.none_of_funds and (not _SchemaKit._as_list(digest.get('items')) or not isinstance(digest.get('roles'), dict) or (not digest['roles'].get('buy'))):
            digest = self.pose_json(prompt_text, [BIG_BRAIN, SMALL_BRAIN], 2500) or digest
        elif not self.none_of_funds and self.pooled_rack(digest):
            once_more = self.pose_json(prompt_text, [BIG_BRAIN], 2500) or {}
            if _SchemaKit._as_list(once_more.get('items')) and isinstance(once_more.get('roles'), dict) and once_more['roles'].get('buy'):
                digest = once_more
        self.digest = digest
        self.peruse_roam(digest.get('browse'))
        jobs = {} if not isinstance(digest.get('roles'), dict) else digest.get('roles')
        self.jobs = {kw: vw for (kw, vw) in jobs.items() if isinstance(vw, str) and vw in self.layouts}
        for attr_key in ('buy_args', 'refuse_args'):
            if not isinstance(digest.get(attr_key), dict):
                digest[attr_key] = {}
        (purchase, turn_down) = (self.jobs.get('buy'), self.jobs.get('refuse'))
        if purchase and turn_down == purchase and (not digest['buy_args'] or not digest['refuse_args'] or digest['buy_args'] == digest['refuse_args']):
            pose = f'The store tool {purchase} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _TextKit._render(self.layouts.get(purchase)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            mended = self.pose_json(pose, [SMALL_BRAIN, BIG_BRAIN], 1500) or {}
            if isinstance(mended.get('buy_args'), dict) and isinstance(mended.get('refuse_args'), dict) and (mended['buy_args'] != mended['refuse_args']):
                (digest['buy_args'], digest['refuse_args']) = (mended['buy_args'], mended['refuse_args'])
        for slot_item in _SchemaKit._as_list(digest.get('items')):
            if not isinstance(slot_item, dict):
                continue
            amount_of = _SchemaKit._number_in(slot_item.get('quantity'))
            count = 1 if not (amount_of and 0 < amount_of) else amount_of
            requisites = [str(mw) for mw in _SchemaKit._as_list(slot_item.get('musts')) if mw]
            self.items.append(ShoppingLine(str(slot_item.get('name') or 'the product'), str(slot_item.get('shelf') or ''), count, requisites))
        unified: list[ShoppingLine] = []
        turned_to: dict[int, int] = {}
        for (prior_one, product) in enumerate(self.items, 1):
            double = next((mw for mw in unified if product.rack.strip() and product.howmany == mw.howmany and (_TextKit._terms(mw.rack) == _TextKit._terms(product.rack)) and (not any((_TextKit._terms(aw.split(':')[0]) == _TextKit._terms(bw.split(':')[0]) and _TextKit._terms(aw.partition(':')[2]) != _TextKit._terms(bw.partition(':')[2]) for aw in mw.requisites for bw in product.requisites)))), None)
            if not double is None:
                turned_to[prior_one] = unified.index(double) + 1
                double.requisites += [mw for mw in product.requisites if mw not in double.requisites]
                self.record_path.append({'kind': 'items_merged', 'into': double.title_name, 'dropped': product.title_name})
            else:
                unified.append(product)
                turned_to[prior_one] = len(unified)
        self.items = unified
        if not self.items:
            self.items = [ShoppingLine('the requested product', '', 1, [])]
        for duo in _SchemaKit._as_list(digest.get('fits')):
            if isinstance(duo, dict):
                (piece, chief) = (turned_to.get(_SchemaKit._number_in(duo.get('item')) or 0), turned_to.get(_SchemaKit._number_in(duo.get('with')) or 0))
                if piece and chief and (chief != piece) and (1 <= piece <= len(self.items)) and (1 <= chief <= len(self.items)) and ((piece, chief) not in self.edge_val):
                    self.edge_val.append((piece, chief))
        for fetch_up in _SchemaKit._as_list(digest.get('lookups')):
            wording = str(fetch_up).strip()
            if not any((wording in product.requisites for product in self.items)):
                attr_key = _TextKit._terms(wording.split(':')[0])
                wording = next((mw for product in self.items for mw in product.requisites if _TextKit._terms(mw.split(':')[0]) == attr_key), wording)
            digit = next((nw for (nw, product) in enumerate(self.items, 1) if wording in product.requisites), 1)
            if wording and (digit, wording) not in self.cites:
                self.cites.append((digit, wording))
        for (digit, product) in enumerate(self.items, 1):
            for demand in product.requisites:
                val = demand if not ':' in demand else demand.split(':', 1)[1]
                if re.search('\\b(?:listing|sells|sold by|pointed to|on the page|page of|page for)\\b', val, re.I) and demand not in [mw for (_, mw) in self.cites]:
                    self.cites.append((digit, demand))
        self.disentangle()
        purse = digest.get('budget')
        if isinstance(purse, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', purse):
            purse = float(purse)
        if isinstance(purse, (int, float)) and (not isinstance(purse, bool)) and (0 < purse):
            self.purse = purse
        cutoff = _SchemaKit._number_in(digest.get('messages'))
        if cutoff is not None and 0 <= cutoff:
            self.mv_cutoff = cutoff
        elif self.jobs.get('ask'):
            self.mv_cutoff = 3
        paces = _SchemaKit._number_in(digest.get('steps'))
        if paces is not None and 10 <= paces < self.most_paces:
            self.most_paces = paces
        for slot_item in _SchemaKit._as_list(digest.get('process')):
            if isinstance(slot_item, dict) and str(slot_item.get('step') or '').strip():
                whenever = str(slot_item.get('when') or '').strip().lower()
                self.procedure.append({'step': str(slot_item['step']).strip(), 'when': 'order' if not whenever in ('choose', 'order', 'end') else whenever, 'messages': max(0, _SchemaKit._number_in(slot_item.get('messages')) or 0)})
        if self.jobs.get('ask'):
            chatter = [pw['messages'] for pw in self.procedure if pw['when'] in ('order', 'end') and pw['messages']]
            self.set_aside = 0 if not chatter else min(max(chatter), self.mv_cutoff)

class _StoreLine:

    def do_step(self, title_name: str, argset: dict[str, Any]) -> dict[str, Any] | None:
        cargo = dict(self.linkage)
        cargo['step'] = self.pace + 1
        cargo['action'] = {'name': title_name, 'args': argset}
        _realm.timeout = max(10, min(95, int(self.timing_remaining() - 5)))
        _realm.max_retries = 1 if not self.timing_remaining() > ROAM_SPARE else 3
        for _ in range(4):
            yield_out = _realm.post_verbose('/environment/call', cargo)
            if isinstance(yield_out.data, dict):
                data = yield_out.data
                self.pace = self.pace + 1 if not isinstance(data.get('step'), int) else data['step']
                if data.get('finished'):
                    self.wrapped_up = True
                noticed = data.get('observation')
                if title_name in self.leaf_utensils() and (not (isinstance(noticed, dict) and noticed.get('error'))):
                    for val in argset.values():
                        self.unsealed.update((vw.strip() for vw in ([val] if not isinstance(val, list) else val) if isinstance(vw, str)))
                return data
            error = yield_out.error or {}
            standing = error.get('status')
            main_text = str(error.get('body') or '')
            self.record_path.append({'kind': 'env_fail', 'status': standing, 'body': main_text[:200]})
            if 409 == standing:
                short_of = main_text.lower()
                if 'finish' in short_of or 'quarantin' in short_of or 'finaliz' in short_of:
                    self.wrapped_up = True
                    return None
                spotted = re.search('expected (\\d+)', main_text)
                if spotted:
                    self.pace = int(spotted.group(1)) - 1
                    cargo['step'] = self.pace + 1
                    continue
                if 'reuse' in short_of or 'different action' in short_of:
                    self.pace += 1
                    cargo['step'] = self.pace + 1
                    continue
                return None
            if standing in (400, 401, 402, 403, 404, 422):
                if 402 == standing:
                    self.no_funds = True
                return None
        return None

class _BrainLine:

    def pose_json(self, prompt_text: str, brains: list[str], most_pieces_tok: int=1800, profound: bool=False) -> dict[str, Any] | None:
        for brain in brains:
            missive = self.ponder([{'role': 'user', 'content': prompt_text}], [brain], most_pieces_tok, profound=profound)
            fetched = _TextKit._json_thing((missive or {}).get('content'))
            if fetched is not None:
                return fetched
            if self.no_funds or self.none_of_funds or CALL_CEILING <= self.rings:
                return None
        return None

    def ring_brain(self, brain: str, missives: list[dict[str, Any]], most_pieces_tok: int, utensils: list[dict[str, Any]] | None, exertion: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        posted = missives
        if brain.startswith('anthropic/') and len(missives) > 1:
            posted = [dict(mw) for mw in missives]
            ticks = [0] + ([] if not len(posted) > 1 else [len(posted) - 1])
            for index in ticks:
                matter = posted[index].get('content')
                if isinstance(matter, str) and matter:
                    posted[index]['content'] = [{'type': 'text', 'text': matter, 'cache_control': {'type': 'ephemeral'}}]
        main_text: dict[str, Any] = {'model': brain, 'messages': posted, 'max_tokens': most_pieces_tok, 'temperature': 0}
        if utensils:
            main_text['tools'] = utensils
            main_text['tool_choice'] = 'auto'
        if exertion:
            main_text['reasoning'] = {'effort': exertion}
        yield_out = _model_client.post_verbose('/inference/chat/completions', main_text)
        data = yield_out.data
        if isinstance(data, dict) and data.get('choices'):
            usage_info = data.get('usage') or {}
            if isinstance(usage_info.get('cost'), (int, float)):
                self.charge += float(usage_info['cost'])
            missive = data['choices'][0].get('message') or {}
            if missive.get('content') or missive.get('tool_calls'):
                return (missive, 200)
            self.record_path.append({'kind': 'llm_fail', 'model': brain, 'status': 0})
            return (None, 0)
        error = yield_out.error or {}
        standing = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            standing = _SchemaKit._number_in(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.record_path.append({'kind': 'llm_fail', 'model': brain, 'status': standing, 'body': str(error.get('body') or '')[:200]})
        return (None, standing)

    def ponder(self, missives: list[dict[str, Any]], brains: list[str], most_pieces_tok: int, utensils: list[dict[str, Any]] | None=None, profound: bool=False) -> dict[str, Any] | None:
        if self.timing_remaining() < ROAM_SPARE + 15:
            brains = list(dict.fromkeys((mw if not BIG_BRAIN == mw else SMALL_BRAIN for mw in brains)))
        for brain in brains:
            for choice in (brain,) if not BIG_BRAIN == brain else BIG_BRAIN_KIN:
                if CALL_CEILING <= self.rings or self.timing_remaining() < CALL_SPARE:
                    return None
                if time.monotonic() < self.brain_loaded.get(choice, 0):
                    continue
                self.rings += 1
                essays = 0
                while True:
                    _model_client.timeout = max(8, min(50, int(self.timing_remaining() - CALL_SPARE)))
                    exertion = None if not (BIG_BRAIN == brain and BIG_BRAIN != choice and (not profound)) else 'low'
                    (missive, standing) = self.ring_brain(choice, missives, most_pieces_tok, utensils, exertion)
                    if missive is not None:
                        return missive
                    if 429 == standing and BIG_BRAIN == brain:
                        self.brain_loaded[choice] = time.monotonic() + CROWDED_WAIT
                        self.rings -= 1
                        break
                    in_passing = standing is None or standing in (402, 408, 425, 429) or (isinstance(standing, int) and 500 <= standing)
                    if not in_passing or essays >= (len(BACKOFF_SET) if not standing in (None, 408) else 1) or self.lingered + BACKOFF_SET[essays] > BACKOFF_TOTAL:
                        break
                    if self.timing_remaining() < BACKOFF_SET[essays] + CALL_SPARE + 10:
                        break
                    halt = BACKOFF_SET[essays] * (0.8 + 0.4 * random.random())
                    self.lingered += BACKOFF_SET[essays]
                    essays += 1
                    time.sleep(halt)
                if 402 == standing:
                    self.none_of_funds = True
                    return None
                if 429 != standing:
                    break
        return None

class _ToolSense:

    def muted(self, title_name: str, argset: dict[str, Any]) -> dict[str, Any]:
        layout = self.layouts.get(title_name) or {}
        traits = layout.get('properties') or {}
        outside = dict(argset)
        for (attr_key, spec_sheet) in traits.items():
            if not isinstance(spec_sheet, dict) or 'string' not in _SchemaKit._types_of(spec_sheet) or 'enum' in spec_sheet:
                continue
            val = outside.get(attr_key)
            if val is None and attr_key not in (layout.get('required') or []):
                continue
            wording = str(val or '')
            titled = any((mark_id in wording for mark_id in self.entry_rows)) or self.digest.get('quiet_decline') is not False
            if titled or not wording:
                outside[attr_key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec_sheet.get('maxLength', 200)]
        return outside

    def quoted(self) -> list[dict[str, Any]]:
        remaining = self.remaining()
        pose = self.jobs.get('ask')
        retain: set[str] | None = None
        closeouts = {self.jobs.get('buy'), self.jobs.get('refuse')} - {None}
        plain_kind = self.plain_kind_products()
        if remaining <= self.rear():
            retain = set(closeouts)
        elif remaining <= self.rear() + (3 if not plain_kind else plain_kind):
            retain = set(closeouts) | {self.jobs.get('cart'), self.jobs.get('cart_view')}
        elif not plain_kind and remaining <= len(self.items) + 5 and (0 == self.purchases):
            retain = set(closeouts) | {self.jobs.get('cart'), self.jobs.get('cart_view'), self.jobs.get('offer')}
        utensils = [tw for tw in self.utensils if tw['function']['name'] != pose]
        if retain is not None and closeouts:
            cut_back = [tw for tw in utensils if tw['function']['name'] in retain]
            utensils = cut_back or utensils
        return utensils

    def be_turn_down(self, title_name: str, argset: dict[str, Any]) -> bool:
        if title_name != self.jobs.get('refuse'):
            return False
        mended = {} if not isinstance(self.digest.get('refuse_args'), dict) else self.digest.get('refuse_args')
        return all((argset.get(kw) == vw for (kw, vw) in mended.items()))

    def be_purchase(self, title_name: str, argset: dict[str, Any]) -> bool:
        if title_name != self.jobs.get('buy'):
            return False
        mended = {} if not isinstance(self.digest.get('buy_args'), dict) else self.digest.get('buy_args')
        return all((argset.get(kw) == vw for (kw, vw) in mended.items()))

    def quote_utensil(self) -> str | None:
        return self.jobs.get('offer') or self.jobs.get('open')

    def contrast_plain_kind(self) -> bool:
        noticed = self.contrast_attrs()
        return noticed is False or (noticed is None and (self.attrs_via.get(self.attrs_utensil() or '') is True or bool(self.jobs.get('offer'))))

    def contrast_attrs(self) -> bool | None:
        return self.attrs_via.get(self.jobs.get('compare') or '')

    def attrs_utensil(self) -> str | None:
        learned = next((tw for (tw, fine_one) in self.attrs_via.items() if fine_one and self.sole_aim(tw)), None)
        if learned:
            return learned
        utensil = self.jobs.get('open')
        return None if not (self.attrs_via.get(utensil or '') is not False and self.sole_aim(utensil)) else utensil

    def leaf_utensils(self) -> set[str]:
        leaf = self.jobs.get('offer') or self.jobs.get('open')
        return {tw for tw in (leaf, self.jobs.get('compare')) if tw}

class _Clock:

    def assesses(self, cutoff: float=OUTLAY_MILD) -> list[str]:
        if cutoff >= self.charge and self.charge + BIG_BRAIN_LEEWAY <= OUTLAY_FIRM:
            return [BIG_BRAIN, SMALL_BRAIN]
        return [] if not OUTLAY_FIRM >= self.charge else [SMALL_BRAIN]

    def shave(self) -> None:
        measure = sum((len(_TextKit._render(mw.get('content'))) for mw in self.convo))
        if CONVO_CUTOFF >= measure:
            return
        utensils = [iw for (iw, mw) in enumerate(self.convo) if mw.get('role') == 'tool']
        for index in utensils[:max(1, len(utensils) // 2)]:
            matter = self.convo[index].get('content') or ''
            if len(matter) > 600:
                code_ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', matter)
                code_ids = [iw for iw in dict.fromkeys(code_ids) if iw in self.entry_rows][:20]
                self.convo[index]['content'] = matter[:400] + f"...(older result shortened; listings in it: {', '.join(code_ids)})"

    def rear(self) -> int:
        chatter = any((pw['when'] == 'order' and pw['messages'] > 0 for pw in self.procedure))
        backup = 0 if not 0 == self.purchases else 1
        return backup + (1 if not (chatter and self.mv_cutoff > self.mv_spent_up and self.toll_pending_due('order')) else 2)

    def mvs_spare_one(self) -> int:
        return self.mv_cutoff - self.mv_spent_up - max(0, self.set_aside - self.toll_mvs)

    def timing_remaining(self) -> float:
        return ROUND_SECONDS - (time.monotonic() - self.begun)

    def remaining(self) -> int:
        return self.most_paces - self.pace

class Orla(_Turns, _Wrapup, _DeclineReview, _Shortages, _Routes, _CartProbe, _Counts, _Matching, _Basket, _Tolls, _Replies, _ValueHunt, _Questioning, _Reading, _Leafing, _Prices, _KeySheet, _Shelving, _RowMemory, _Register, _TaskReading, _StoreLine, _BrainLine, _ToolSense, _Clock):

    def __init__(self, snag: dict[str, Any]):
        self.begun = time.monotonic()
        realm_full = snag['environment']
        self.linkage = dict(realm_full['binding'])
        self.look = realm_full['policy_view']
        self.petition = str(self.look.get('query') or '')
        self.utensils = [tw for tw in self.look.get('tools') or [] if isinstance(tw, dict) and isinstance(tw.get('function'), dict) and isinstance(tw['function'].get('name'), str)]
        self.layouts = {tw['function']['name']: {'type': 'object', 'properties': {}} if not isinstance(tw['function'].get('parameters'), dict) else tw['function'].get('parameters') for tw in self.utensils}
        paces = _SchemaKit._number_in(self.look.get('max_steps'))
        if paces is None and isinstance(self.look.get('max_steps'), float):
            paces = int(self.look['max_steps'])
        self.most_paces = 30 if not (paces and 0 < paces) else paces
        self.pace = 0
        self.wrapped_up = bool(self.look.get('finished'))
        self.no_funds = False
        self.none_of_funds = False
        self.lingered = 0
        self.lapses = 0
        self.charge = 0.0
        self.rings = 0
        self.circuit_rings = 0
        self.record_path: list[dict[str, Any]] = []
        self.digest: dict[str, Any] = {}
        self.jobs: dict[str, str] = {}
        self.items: list[ShoppingLine] = []
        self.purse: Any = None
        self.mv_cutoff = 0
        self.mv_spent_up = 0
        self.procedure: list[dict[str, Any]] = []
        self.set_aside = 0
        self.toll_mvs = 0
        self.toll_fine_one: set[tuple[str, str]] = set()
        self.through_journal: list[str] = []
        self.through_paces: list[int] = []
        self.posed = False
        self.quiz: list[tuple[str, str]] = []
        self.entry_rows: dict[str, dict[str, Any]] = {}
        self.whole_rec: dict[str, dict[str, Any]] = {}
        self.noticed_purchase_order: list[str] = []
        self.listened: set[str] = set()
        self.jots: list[str] = []
        self.edition = 0
        self.trolley_handled = False
        self.trolley_upon = -1
        self.trolley_noticed_upon = -1
        self.trolley_look: Any = None
        self.jot_titled = False
        self.trolley_row_lines: dict[str, int] = {}
        self.probed: set[str] = set()
        self.sweep_checked = False
        self.quote_noticed: dict[str, int] = {}
        self.departed: dict[str, int] = {}
        self.not_offered: set[str] = set()
        self.soldout_row_lines: set[str] = set()
        self.jot_soldout = False
        self.kept_hold = 0
        self.code_id_argset: set[tuple[str, str]] = set()
        self.trolley_stated_out = ''
        self.turned_down_upon = -1
        self.pointers_r: list[str] = []
        self.turn_down_slabs = 0
        self.shoved = False
        self.assembled: list[str] = []
        self.shed_off: set[str] = set()
        self.tardy_kept_hold = False
        self.pointer_products: dict[str, list[str]] = {}
        self.rulings: dict[str, bool] = {}
        self.affirmed: str | None = None
        self.fulfilled = False
        self.code_id_attr_key: str | None = None
        self.attrs_via: dict[str, bool] = {}
        self.lined_up: list[str] = []
        self.purchases = 0
        self.turndowns = 0
        self.latest: list[str] = []
        self.convo: list[dict[str, Any]] = []
        self.unsealed: set[str] = set()
        self.cites: list[tuple[int, str]] = []
        self.cite_vals: dict[str, str] = {}
        self.routes = 0
        self.route_slabs = 0
        self.sequence_keeps_hold = 0
        self.sequence_turnstiles = 0
        self.unbidden_unsealed = 0
        self.initial_probed: set[int] = set()
        self.contrast_fine_one = 0
        self.contrast_poor = 0
        self.utensil_attr_keys: dict[str, set[str]] = {}
        self.look_attr_keys: dict[str, tuple[int, set[str]]] = {}
        self.peruse_stash: dict[str, tuple[int, int, str]] = {}
        self.leaves_set_peruse = 0
        self.forms: dict[str, dict[str, Any] | None] = {}
        self.fee_attr_key: str | None = None
        self.inventory_attr_key: str | None = None
        self.thriftier_ticks: set[str] = set()
        self.broader_kept_hold = False
        self.leafed_sight: list[Any] = []
        self.roam: dict[str, Any] | None = None
        self.roam_lapses = 0
        self.brain_loaded: dict[str, float] = {}
        self.scanned: set[str] = set()
        self.blueprint_purchase_order: list[int] = []
        self.row_line_keeps_hold = 0
        self.edge_val: list[tuple[int, int]] = []
        self.match_up_keeps_hold = 0
        self.hunted: list[dict[str, Any]] = []
        self.hop_hunts = 0
        self.hop_noticed: dict[str, list[str]] = {}

    def launch(self) -> list[dict[str, Any]]:
        if self.wrapped_up:
            return self.record_path
        self.peruse_petition()
        self.record_path.append({'kind': 'brief', 'brief': self.digest})
        unsealing = self.look.get('opening')
        if unsealing is not None:
            self.soak_up(unsealing)
        preface = "SHOPPER'S REQUEST:\n" + self.petition
        if unsealing is not None:
            preface += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.present_r(unsealing)
        preface += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.standing()
        self.convo = [{'role': 'system', 'content': ORIENTATION}, {'role': 'user', 'content': preface}]
        self.toil(CIRCUIT_RINGS, OUTLAY_FIRM)
        if not self.wrapped_up:
            flipped = self.wrap_up(once_more=True)
            if flipped and (not self.wrapped_up) and (not self.no_funds) and (not self.none_of_funds) and (self.remaining() > 1) and (self.timing_remaining() > ROAM_SPARE):
                self.convo.append({'role': 'user', 'content': '[Orla] Orla tried to end the episode and ' + flipped + '\nApply what the store says, then finish.\n' + self.standing()})
                self.toil(self.circuit_rings + 10, OUTLAY_FIRM + 0.01)
            if not self.wrapped_up:
                self.wrap_up()
        self.record_path.append({'kind': 'end', 'finished': self.wrapped_up, 'steps': self.pace, 'cost': round(self.charge, 5), 'calls': self.rings})
        return self.record_path

    def toil(self, circuit_lid: int, expend_lid: float) -> None:
        slack = 0
        while not self.wrapped_up and (not self.no_funds) and (self.remaining() > 0) and (circuit_lid > self.circuit_rings) and (expend_lid > self.charge) and (self.timing_remaining() > ROAM_SPARE):
            if self.set_up_toward_pose():
                abstract = self.canvass()
                if abstract:
                    self.convo.append({'role': 'user', 'content': abstract + '\n' + self.standing()})
                continue
            self.shave()
            self.circuit_rings += 1
            missive = self.ponder(self.convo, [SMALL_BRAIN, BACKUP_BRAIN], 3000, self.quoted())
            if missive is None:
                self.lapses += 1
                if self.none_of_funds or CALL_CEILING <= self.rings or 2 < self.lapses:
                    break
                continue
            rings = []
            for (digit, ring) in enumerate(missive.get('tool_calls') or []):
                if not isinstance(ring, dict):
                    continue
                fun = {} if not isinstance(ring.get('function'), dict) else ring.get('function')
                untouched = fun.get('arguments')
                read_in = ({} if not (isinstance(untouched, str) and untouched.strip()) else _TextKit._json_thing(untouched)) if not isinstance(untouched, dict) else untouched
                rings.append({'id': str(ring.get('id') or f'k{self.circuit_rings}_{digit}'), 'type': 'function', 'function': {'name': str(fun.get('name') or ''), 'arguments': _TextKit._render({} if not isinstance(read_in, dict) else read_in)}})
            matter = missive.get('content')
            if not isinstance(matter, str):
                matter = '' if not matter else _TextKit._render(matter)
            if not rings:
                slack += 1
                self.convo.append({'role': 'assistant', 'content': matter or '(no call)'})
                self.convo.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.standing()})
                if 3 <= slack:
                    break
                continue
            slack = 0
            locking_in = {self.jobs.get('cart'), self.jobs.get('buy'), self.jobs.get('refuse')} - {None}
            if not self.posed and self.jobs.get('ask') and (self.mvs_spare_one() > 0) and any((cw['function']['name'] in locking_in for cw in rings)) and (self.remaining() > 8):
                self.convo.append({'role': 'assistant', 'content': matter or None, 'tool_calls': rings})
                for ring in rings:
                    self.convo.append({'role': 'tool', 'tool_call_id': ring['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                abstract = self.canvass()
                self.convo.append({'role': 'user', 'content': (abstract or 'The interviewer had nothing to ask.') + '\n' + self.standing()})
                continue
            self.convo.append({'role': 'assistant', 'content': matter or None, 'tool_calls': rings})
            self.record_path.append({'kind': 'turn', 'text': matter[:300], 'calls': [cw['function'] for cw in rings]})
            responses = self.launch_rings(rings)
            for (index, (ring_id, wording)) in enumerate(responses):
                if index == len(responses) - 1:
                    wording += '\n' + self.standing()
                self.convo.append({'role': 'tool', 'tool_call_id': ring_id, 'content': wording})

def agent_main(snag_payload_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        round_r = Orla(snag_payload_data)
    except Exception as err:
        return [{'kind': 'crash', 'error': f'{type(err).__name__}: {err}'[:400]}]
    try:
        return round_r.launch()
    except Exception as err:
        round_r.record_path.append({'kind': 'crash', 'error': f'{type(err).__name__}: {err}'[:400]})
        try:
            if not round_r.wrapped_up:
                round_r.wrap_up()
        except Exception as once_more:
            round_r.record_path.append({'kind': 'crash', 'error': f'{type(once_more).__name__}: {once_more}'[:400]})
        return round_r.record_path
__all__ = ['agent_main']