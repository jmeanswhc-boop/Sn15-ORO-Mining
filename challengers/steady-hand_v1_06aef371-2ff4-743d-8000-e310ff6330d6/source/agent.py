"""Orrin: one shopper's buyer inside an ORO store world.

Built on the public YesYesYes agent (itself built on Know Thyself v1 and KYO v6). A light model walks the store with
the world's own tools; code keeps the ledger, questions the shopper and guards the finish; a heavy model reads the
request and checks the cart before an order and the evidence before a decline.

Changes from YesYesYes:
- After a finish the world turns down, the agent re-picks from the listings it knows under the new terms and checks
  out; a decline then needs a fresh review showing an item with no fitting listing under those terms.
- A decline needs proof: the review of the evidence, a recovery search (late in an episode, a shelf already read),
  and an item that no known listing fits. Otherwise the best known basket is ordered, also when the loop ends with
  an empty or incomplete cart. A lowest-price line that sells out is re-picked at its own price first.
- A basket that would leave out a requested item with no fitting listing ends without a purchase instead.
- Listings printed as text lines and JSON held in strings are read into records like any other row.
- Codes printed inside listings are refused as listing ids before a step is spent; detail pages keep more text.
- Every basket listing is read with each detail view the store offers before checkout.
- A shopper who turns a question back is asked again only when a look-alike key can be set apart.
- Compatibility between a part and its device is judged by the model from the records, not by fixed phrases; plain
  values for searches come from the model's reading and the episode's own listings.
- Lower spending caps per episode.
"""
from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
LIGHT = 'anthropic/claude-haiku-5.5'
HEAVY = 'anthropic/claude-sonnet-5.5'
SPARE = 'openai/gpt-6-luna'
HEAVY_KIN = (HEAVY, 'anthropic/claude-sonnet-5', 'anthropic/claude-sonnet-4.6')
LOOP_CALLS = 42
ALL_CALLS = 60
SPEND_SOFT = 0.028
SPEND_HARD = 0.055
HEAVY_ROOM = 0.014
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
PAGE_CHARS = 1400
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3
AUTO_PAGES = 6
SHORT_PAGES = 4
CHEAPER_OPENS = 3
EPISODE_SECONDS = 288
EXPLORE_RESERVE = 50
CALL_RESERVE = 15
TIGHT_MESSAGES = 2
RECOVERY_STEPS = 8
BUSY_SECONDS = 20
SWEEP_STEPS = 6
ID_SHARE = 0.9
CACHE_CHARS = 2000
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = 'You are Orrin. You buy for a single shopper in an online store, and calling the store\'s tools is the only way you can act. Scoring is all or nothing per episode. Full marks need either (a) an order containing each requested item at its requested quantity, every one of them satisfying every requirement still in force at checkout, with the total inside the budget, or (b) when the store genuinely carries nothing that satisfies everything, the no-purchase ending, performed the way the request describes.\n\nSTORE MECHANICS\n1. Steps are the scarce resource: each tool call costs one, reasoning costs none. Tool names, argument names and result layouts belong to this store alone; take them from each tool\'s description and schema, spelled exactly, using only values the schema allows.\n2. Expect narrow searches: category filters, keyword modes that look at the opening words only, small pages, and a token or offset to reach later pages. If the right answer needs the full result, check the reported total against the rows in hand and request later pages with precisely the token or offset the store supplied. Brief keyword queries work best: wanted value first, then the product noun. An empty filtered result usually signals a wrong filter value or parameter; retry using the label as rows print it, or another filter parameter.\n3. Rows are summaries. Each store splits a listing differently: attributes (properties) in one view, offer details such as price, stock, seller notes and description in another, perhaps a side-by-side view carrying either. The ledger names the attribute view once Orrin has found it. Requirements on properties are judged on those attributes, so read them for anything you may buy; a call covering several listings is cheaper when it shows what you need. Reopening an unchanged listing wastes a step.\n4. Part, model and series codes printed in titles, overviews or notes are not listing ids. Only the id in a result row opens a listing; a code is something to search for.\n5. Orrin\'s interviewer handles all contact with the shopper, so you never message the shopper. Replies land in the conversation and in the ledger.\n6. Extra steps the request demands (an approval, a message, something to read or check first) appear in the ledger. Those due right before ordering or before the no-purchase ending are performed by Orrin when you call checkout or that ending, so do not wait for them and do not abandon the purchase over them; perform any other such step with store tools (product questions still go through the interviewer).\n\nFIT RULES\n1. A listing counts as the requested kind only when its category label names that item as a whole; neighbouring categories never do.\n2. A property requirement is satisfied when the listing\'s attribute for that property carries the value. Ignore how keys are spelled (spacing, underscores, camel case, prefixes); one member of a comma- or slash-separated list suffices. Values are judged word by word. Letter case, one-or-many forms and everyday synonyms are irrelevant, and a value containing every required word is a match ("pale olive" for olive, "wide grip handle" for wide grip). Other words are no match, however close: an abbreviation or code is not the word behind it, and "olive drab" is not "olive green". Numbers are compared after unit conversion, boundaries count as inside a limit, and a range qualifies only if all of it lies within the limit.\n3. When a key names the requirement\'s property (directly or by an ordinary synonym), that key is the judge: a different value there fails the listing regardless of the title or other keys. Keys about another property, even a neighbouring one (the colour of a strap is not the colour of the bag), are no evidence either way. Under opened listings Orrin prints [evidence] lines with the keys whose names match a requirement.\n4. Without any attribute for the property, the value counts if another attribute (a features or styles list, say) or the title or description states it. A number measuring something else does not count (weight is not volume), nor does a number whose text never says what it measures.\n5. A category label that itself states a requirement\'s value (an audience or season in the label) satisfies that requirement.\n6. Some requirements refer to another listing instead of naming a value ("the same Y as <shop>\'s listing with <property>", "within N of the Y on the page of ..."). Resolve these before anything else: locate that listing (by its category, brand, shop or words), open its full page with a single-listing or side-by-side tool (a row is not enough), and if that page directs you to yet another listing for the property, locate and open that one too; the value comes from the last page. The order is valid only if every page along that chain was opened. If the requirement asks for anything other than that listing\'s value, it only rules out listings having exactly that value, and it can never by itself justify ending without a purchase.\n7. If a listing you plan to buy refers to another page about itself, read that page first.\n8. If one requested item has to work with another (an accessory for a device), pick the two as a pair from what the listings themselves say (a list of compatible models, a model or series code on both pages). If no accessory meeting its own requirements works with your chosen device, change to another device that meets its requirements and that a qualifying accessory works with; ending without a purchase is not allowed while that change is untried.\n9. Exclusions and closed lists are binding. Soft wishes ("ideally", "would be nice") serve only to break ties.\n10. Never buy what the store shows as sold out. The order total (price times quantity over all lines) must stay at or under the budget.\n\nCHOICE RULES\n1. "Cheapest" requests (lowest price, smallest total) require the lowest-priced listing among ALL fitting ones. Gather the full fitting set (page through it, since rows arrive in no price order; Orrin pulls the next pages of a short result itself), test candidates from cheapest upward, and take the first one that fits completely. If the search accepts a maximum price, list the category below the price of your current best candidate so that only cheaper ones remain.\n2. "Most" or "least" requests: compare all fitting listings on that measure and take the extreme; a listing that omits the measure cannot win.\n3. Otherwise the first listing that meets everything is correct; do not spend steps on a better one.\n4. Cart an item as soon as an opened listing meets all of its requirements (for cheapest, most or least, after comparing the fitting set), then move to the next item. Carted lines often sell out soon after, and carting early leaves room to replace them.\n5. Several different items mean one cart line per item, each at its quantity.\n\nCHANGES\n1. The shopper or the store may alter the task midway: budget, added or dropped requirements, prices, sold-out listings. Such notes ride inside tool results, often as a refusal when you try to finish; a note applies whichever tool returned it, and the ledger shows the requirements now in force. A refusal does not end the episode: apply every change, choose again among listings you already know (move to the next fitting one when a pick sold out or no longer meets the terms), correct the cart and check out again.\n2. A refused finish is never evidence that the store lacks a fit. After a refusal, buy the best listing that meets the new terms; end without a purchase only when those terms leave some requested item with no fitting listing in the store.\n3. Price and stock can change silently, so re-read a chosen listing\'s record shortly before checkout.\n\nENDING\n1. Leave only the chosen lines in the cart (remove the rest), view the cart once, then send checkout as a call of its own.\n2. The no-purchase ending is right only after real searching proves that at least one requested item lacks any listing satisfying all of its terms. A shortage of steps proves nothing like that: check out the best cart you have instead. Word that ending as the request asks; if a remark is needed, keep it short and name no listing.\n3. Each reply must include at least one tool call. Keep your text brief: what you know, then your next move.'
BRIEF = """Turn a shopper's request and a store's tool list into a task sheet for a shopping agent. Reply with ONE compact JSON object (no indentation, nothing around it) holding these keys:

"items": one entry for each separate product the shopper wants bought, each {"name": that kind of product in the request's own words, "shelf": the store's category label for it when some tool lists labels (copied exactly), else "", "quantity": units of that single listing (an integer, 1 when unstated; a pair or set sold as one listing is 1), "musts": the firm requirements on that product, each written "property: constraint" with the exact value, limit, exclusion or closed list, "plain": the musts (copied exactly) whose constraint is a single value, or a short list of alternatives, that a listing would print in its own words (leave out limits, ranges, exclusions, comparisons and values taken from another listing)}. Call the property by the request's own word for it ("meant for a beach outing" gives "occasion: beach", "a 30-litre tank" gives "tank capacity: 30 litre"); never trade it for a neighbouring property. A must whose value comes from another listing is ONE must holding that listing's whole description ("strap colour: other than the strap colour of the backpack Talvero sells with a laptop sleeve"): words describing that other listing (its shop, its kind, the "with <property> <value>" phrases after it) stay inside that single must and never become musts of this product. Price, budget, stock and quantity never go into musts. The story around the request (who will use it, where it goes, what went wrong before, the event behind it) is background and not a must, unless it gives a property's value or limit outright; words that are part of what the product is called (a season or an audience named with the product) go into "name", not into musts. A use or occasion the request turns into a property of the product itself ("boots for hiking", "a gown made for evening events", "the occasion must be formal") IS a must on that property ("occasion: hiking"), unless the store's category label already names it; an event, person or place the product is bought for ("to wear at the reunion", "as a present for my aunt"), and how it should sit, feel or look in use, stay background. Products named in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) form ONE item; make separate items only when the request gives each its own quantity or its own requirements.
"budget": the most the shopper will pay in total, as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it applies per item, null when there is no limit.
"in_stock_only": true unless the request plainly accepts listings that are not available.
"pick": "cheapest" when the request wants the lowest price or the smallest total, "most" or "least" when it wants the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": the measure to maximise or minimise for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when unstated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": every instruction the request gives on how to work or how to end, each in 12 words or fewer.
"wishes": things the shopper would merely like.
"ask_first": things the request explicitly says to check with the shopper before choosing (empty when none).
"process": [] unless the request makes the agent do something beyond finding and buying the right items, such as winning the shopper's approval, telling or showing them something, checking a point with them, reading something first or using a named tool. For each such action give {"step": the action in at most 15 words, "when": "choose" if it comes before any product is picked, "order" if it comes just before checkout, "end" if it comes just before a no-purchase ending, "messages": the number of shopper messages it needs, 0 if none}. Item requirements, the budget, stock, the message and step limits and the way to end when nothing fits are not actions of this kind.
"quiet_decline": true if the request forbids suggesting or naming alternatives when nothing fits, false if it wants alternatives offered, null if it says neither.
"fits": [] unless the request says a requested item has to work with, fit or suit ANOTHER requested item (whichever one the agent picks); then one {"item": that item's number, "with": the other item's number} per such pair.
"browse": how one tool lists a single store category, when one can: {"tool": exact tool name, "category": the argument taking the category label, "max_price": the argument taking a maximum price, or null, "fixed": the other arguments that put the tool in that mode, else {}}; null when no tool lists a category.
"query_arg": the argument of the search tool that takes free search words, or null.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when there are two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as price, stock and seller notes, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}."""
PLAN = 'Task: draft the shopper interview for a buying agent.\n\nModel of the shopper: besides what the request says, the shopper holds a few unspoken conditions, each fixing one property of one requested product. A question uncovers a condition only when it names precisely that property. A named property the shopper is indifferent to gets a "don\'t mind"; one the shopper has never considered gets a bewildered answer. Each named property costs a share of a message.\n\nYou receive: the request; the product items and the conditions the request already sets on each; and, for each product\'s category, a numbered catalogue of fields (ATTRIBUTE KEYS SEEN) found on the listings opened so far, every field shown with how many listings bear it and a few example values. Unspoken conditions can also sit on properties absent from the catalogue.\n\nOrder of priority:\n(1) Whatever the request asks the agent to confirm with the shopper (shown under ASK FIRST if any), expressed as the property in question.\n(2) Story-hinted properties. A request\'s narrative often points at an unspoken condition while withholding its value: a remark about something the shopper cannot assess or leaves to the agent, somebody whose taste settles it, the product\'s appearance or the place or occasion it is meant for, whoever will wear or use it, or its contents. Turn every such remark into a single property plus the product it belongs to, in narrative order. If a catalogue field of that product matches the hint, use that field (hints about the wearer map to a gender or audience field; hints about appearance map to whichever colour, pattern or finish field most of its listings bear). A hinted property that no listing of the product bears goes after that product\'s three most widespread fields.\n(3) Then catalogue fields, best first: the decisions a buyer of such a product makes first (usually type or style, form or fit, size, capacity or measurements, material, colour or pattern, as suits the product) and fields borne by most of the product\'s listings. Once five or more listings have been read, pass over fields seen on just one or two.\nNever pick: a property the request has already settled for that product (careful: a field whose values merely restate a demanded value, for instance a print field echoing the demanded print, is still open); a synonym of a property already picked; a field each of whose values bundles several purchase options; identifiers, model names, warranty or other logistics. Pack size, counts and pieces per pack are ordinary properties. A field marked [not the stated ...] differs from that condition no matter how alike the names look. A field marked [many features in one] comes last. Treat each product on its own: hinted properties first, catalogue fields after.\nNaming: "label" gives the property\'s plain name; for a catalogue field, all of its words in normal spelling ("strap colour" for strapColour, "buckle finish" for buckle_finish, never trimmed to "colour" or "finish"); nothing else goes in it: no examples, brackets or extra property.\nReturn at most as many picks per product as PICKS PER PRODUCT allows below, best first, as ONE JSON object and nothing more: {"ask": [{"item": item number, "label": "...", "hint": "the story\'s words behind it, eight at most, or an empty string"}]}, listing every product\'s picks from best to worst'
HEAR = 'Below are questions a shopping agent sent to a shopper, with the replies. Output ONE JSON object only:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true if the value must be read from another listing else false, "plain": true if the constraint is one value (or a few alternatives) that a listing would print word for word, false for limits, ranges, exclusions and comparisons}], "drop": [{"item": item number, "property": "a requirement the shopper is giving up"}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers whose reply complains the message covered too much or wants one question at a time; not replies that merely did not understand], "budget": new total limit as a number, or null}\nRules:\n- needs: list each firm condition a reply sets, with the precise value or bound and its unit ("no wider than 40 cm", "round", "not leather"). The property is whichever one the reply mentions, even when another was asked about; repeated replies on one property merge into a single need. Quote the reply\'s words for the value, keeping nouns and units ("at least 4 drawers", not "at least 4").\n- A reply may set the value by reference to another listing (like, unlike or close to the value on some shop\'s product, identified by a few of its properties). That is a single need on the property the reply names, with "points": true, and its value is the reply\'s wording copied verbatim from the comparison words to the end of that description, including anything it says about a listing that page may name. Do not shorten or paraphrase it. Properties used to identify the other product are not needs of this item.\n- free: the property of each question the shopper said they don\'t mind about.\n- Colloquial or indirect wording that names a value still states it.\n- The value is only what the property should be; linking words between property and value (matching, set to, coming in as) are not part of it: "colour set to navy" means navy. A value given by reference to another listing is the exception and stays whole.'
NOTE = """While a shopping agent was working, the store showed the note below (it may come from the shopper or from the store). Reply with ONE JSON object and nothing else:
{"kind": "change" when the note alters the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only says a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement that is no longer wanted"}], "gone": [ids of listings said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or can no longer be bought (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A requirement that is replaced counts as a drop of the old one plus an add of the new one. Report every change the note makes."""
READING = """a property's requirement is decided by the listing attribute whose key names that same property (any key spelling; one entry of a list is enough; synonyms and unit conversions count; a value that contains the required value's words counts, as "pale olive" does for olive; limits include their edge); a key for another property, even a related one (a strap colour is not the bag's colour), decides nothing; when no attribute names the property, the wanted value written under another attribute or in the title or description counts (a bare number that does not say what it measures counts for nothing); a requirement stated by the category label itself is met by that category; the category label must match the item; a rule of differing from another listing's value counts as met unless the records show that same value; listings out of stock or over the budget do not fit."""
CHECK = """You are the final check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
How to read the records: """ + READING + """ The newest record of a listing wins.
Each requested item must be its own cart line at the requested quantity, nothing else may be in the cart, nothing may be shown out of stock, and price times quantity summed over all lines must stay within the budget. When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them plainly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Reply with ONE JSON object and nothing else: {"ok": true or false, "problems": ["a short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Set ok=false only for a problem the records plainly show or for a requirement that no record addresses."""
AUDIT = 'Before a shopping agent gives up and ends with no purchase, review the evidence: hold the listing records shown against the requirements. Reading rules: ' + READING + '\nROW records were only seen as search results: their attributes are unread, so a property the row does not show can neither pass nor fail them.\nThe requirements were noted down in haste, so first compare each with the REQUEST and the ANSWERS FROM THE SHOPPER. Put under "not_firm" any requirement that only echoes the story (who it is for, when, where or why it gets used, past trouble, how it should feel) and that the request does not demand of the product. Put under "corrected", with the plain value, any value that absorbed words which merely link a property to its value. Judge the listings with the not_firm requirements left out and the corrected values used. A requirement that the request or an answer states outright stays even if no listing meets it; then ending with no purchase is correct.\nOutput ONE JSON object only: {"not_firm": [{"item": item number, "need": the requirement copied exactly}], "corrected": [{"item": item number, "need": the requirement copied exactly, "value": "property: plain value"}], "fits": {"item number": [ids of listings that seem to meet every requirement, cheapest first, at most 4]}, "near": {"item number": [ids of the listings closest to fitting, fewest failed requirements first, at most 3]}, "open": [ids of ROW listings of a requested kind and inside the budget whose page might show a fit, most promising first, at most 6], "search": [up to 4 of {"item": item number, "query": "an untried keyword search of two or three words that might surface a fitting listing: another name for the product plus one required value (long queries tend to match nothing)"}], "reason": "one sentence"}'
UNMIX = """A shopper's request was turned into firm requirements on a product. Some requirements take their value from another listing the request points to, and the words that single out that other listing (its kind, its shop, the properties listed after it with "with ...") are sometimes misread as requirements on the product being bought. Going by the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Reply with ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when each is a real requirement on the product]}"""
TRAIL = 'A shopping agent is close to checking out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can pass the reader on to a further listing for that property. The order only counts when the agent has opened in full every listing along each such chain. Trace each requirement\'s chain from the requirement\'s own description, one link at a time, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page unread).\nReply with ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found yet: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that itself states the property\'s value (also when the requirement wants a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as that last page states it, or "", "relation": how the listing to buy must relate to that value: "same", "exclude" (any value except that one), "paired" (the value the pages say goes together with it), "window" (within a stated distance of it), "decoy" (the page only says a version exists elsewhere and fixes nothing), or "" when unclear, "wanted": the value the listing to buy must show when the relation gives one ("same": the value itself; "paired": the partner value the pages name), else ""}]}\nMatch a link to a record only on its whole description: the shop or brand, the kind of product and each property value given (so "a 3 tier stand finished in walnut" is not that shop\'s 2 tier stand, walnut or not). If no record carries all of it, the link counts as not found, however many other listings of that kind the shop has, and its "find" holds a few search words: the shop\'s name plus the values from the description.\nGive each chain\'s links in order, ending at the first link that is not found or whose page is not OPENED: its words stay unknown until then.'
GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Using the numbered RECORD of what the agent has done, decide for each step due before this STAGE whether it is done for the cart as it stands now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions about a product's properties do not do a step.
When a step is still missing, give the single store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Reply with ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""
SWAP = """A listing in a shopping agent's cart has just sold out. Among the other listing records shown, find replacements for that cart line. How to read the records: """ + READING + """
Reply with ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number, as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""
SHAPE = """A store tool returned the result below. Name its parts by the exact keys used in the result and in the tool's parameters. Reply with ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in total (not how many are shown), or null, "next": the key holding the value that requests the following page, or null, "next_arg": the tool parameter that accepts that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, telling whether the listing can be bought now, or null}"""
FITS = """A shopping agent's cart holds a device together with parts that have to work with it. Judge from the listing records below, for each part, whether the records say it works with this device: a list of devices or models it works with, a model or series code both records carry, or a statement that it does not work with certain models. Reply with ONE JSON object and nothing else: {"parts": [{"id": the part's listing id, "works": "yes", "no" or "unknown", "why": "the few words of the records that decide it"}], "others": [ids of other part listings below whose records say they work with this device, at most 4]}"""
ALIKE = """A shopper answered a one-property question as if it had asked about two things at once. Below are the property asked about, the product, and the attribute keys this product's listings carry. Name the one key the shopper most likely heard as a second subject of that question (a neighbouring property easy to mix up with it), or "" when none is. Reply with ONE JSON object and nothing else: {"key": the key exactly as listed, or ""}"""
MODES = """A store tool changes a shopping cart, and its arguments include a mode chosen from a fixed list. From the schema below, say which mode adds units of a listing and which takes them out. Reply with ONE JSON object and nothing else: {"argument": the mode argument's name, "add": the mode value that adds, "remove": the mode value that removes, or null}"""
FIXED_HEADS = (BRIEF, PLAN, HEAR, NOTE, CHECK, AUDIT, UNMIX, TRAIL, GATE, SWAP, SHAPE, FITS, MODES, ALIKE)
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
WIDEN_EACH = 6
OWNER_KEY = 'seller|shop|store|vendor|brand|maker|merchant'

def _scalar(text: str) -> Any:
    """A value printed in a text row, read back as the number or flag it spells when it spells one."""
    word = text.strip()
    if word in ('True', 'False'):
        return word == 'True'
    if word == 'None':
        return None
    if re.fullmatch('-?\\d+', word):
        return int(word)
    if re.fullmatch('-?\\d+\\.\\d+', word):
        return float(word)
    return word

def _row_text(text: Any) -> dict[str, Any] | None:
    """A listing printed as one line of 'key: value' parts joined by a separator, as some stores print rows. A part
    whose value is itself several 'name=value' pairs becomes a nested record. Punctuation alone decides."""
    if not isinstance(text, str) or len(text) < 12 or ': ' not in text:
        return None
    best: dict[str, Any] | None = None
    for sep in (' | ', '\n', ' ; ', ' || '):
        if text.count(sep) < 2:
            continue
        record: dict[str, Any] = {}
        last = None
        for part in text.split(sep):
            key, mark, value = part.partition(': ')
            key = key.strip()
            if mark and key and len(key) <= 40 and len(key.split()) <= 4 and key not in record:
                record[key] = value.strip()
                last = key
            elif last is not None:
                record[last] = f'{record[last]}{sep}{part}'
            else:
                record = {}
                break
        if len(record) >= 3 and (best is None or len(record) > len(best)):
            best = record
    if best is None:
        return None
    out: dict[str, Any] = {}
    for key, value in best.items():
        pairs = [p.partition('=') for p in re.split('\\s*;\\s*', value)] if value.count('=') >= 2 else []
        if pairs and all((eq and k.strip() and len(k.strip()) <= 40 for k, eq, _ in pairs)):
            out[key] = {k.strip(): v.strip() for k, _, v in pairs}
        else:
            out[key] = _scalar(value)
    return out

def _unfold(node: Any, depth: int=0, slot: bool=True) -> Any:
    """The observation with listings printed as text lines turned into records and JSON held in strings parsed, so
    every row format reaches the ledger in one shape. A text line is read as a listing only where a listing can sit:
    a list entry, the whole result, or the value of a small wrapper; a text field inside a record stays text."""
    if depth > 8:
        return node
    if isinstance(node, str):
        text = node.strip()
        if len(text) >= 2 and text[0] in '{[' and text[-1] in '}]':
            try:
                return _unfold(json.loads(text), depth + 1)
            except ValueError:
                pass
        row = _row_text(node) if slot else None
        return row if row is not None else node
    if isinstance(node, dict):
        return {k: _unfold(v, depth + 1, len(node) <= 2) for k, v in node.items()}
    if isinstance(node, list):
        return [_unfold(v, depth + 1) for v in node]
    return node

def _id_shape(ident: str) -> tuple[int, str]:
    lead = re.match('[^0-9]*', ident)
    return (len(ident), (lead.group(0) if lead else '').lower())

def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def _clip(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: _clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit, depth + 1) for v in value]
    return value

def _json_obj(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None

def _words(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def _stem(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in _words(text).split()))

def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def _fits_kind(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True

def _shape(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = _kinds(schema)
    if kinds and (not any((_fits_kind(value, k) for k in kinds))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((_fits_kind(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [_shape(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: _shape(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
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
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def _id_like(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def _learn_id(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((_id_like(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def _in_rows(node: Any, ident: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((_in_rows(v, ident, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((_in_rows(v, ident, depth + 1) for v in node.values()))
    return False

def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

class Item:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []
        self.plain: set[str] = set()
        self.flagged: set[str] = set()

class Orrin:

    def __init__(self, problem: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = _count(self.view.get('max_steps'))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.step = 0
        self.finished = bool(self.view.get('finished'))
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
        self.note_named = False
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
        self.cart_said = ''
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.lead_items: dict[str, list[str]] = {}
        self.near_items: dict[str, list[str]] = {}
        self.better: list[str] = []
        self.verdicts: dict[str, bool] = {}
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
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.browse: dict[str, Any] | None = None
        self.browse_misses = 0
        self.model_busy: dict[str, float] = {}
        self.swept: set[str] = set()
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}
        self.turned = 0
        self.proof: dict[int, int] = {}
        self.reviewed_at = -1
        self.ordered_best = 0
        self.viewed: set[tuple[str, str]] = set()
        self.bundled: set[str] = set()
        self.alike_asked: set[str] = set()
        self.fit_seen: dict[str, list[str]] = {}
        self.modes: dict[str, Any] | None = None
        self.query_arg: str | None = None
        self.relations: dict[str, str] = {}
        self.code_refusals = 0
        self.gap_done = False
        self.decline_turned = False
        self.listed_total: int | None = None
        self.reviewed = False
        self.vocab: set[str] = set()
        self.vocab_at = -1
        self.note_kind = ''
        self.row_edits = 0
        self.cat_memo: tuple[Any, str | None] | None = None

    def left(self) -> int:
        return self.max_steps - self.step

    @staticmethod
    def complaint(observation: Any) -> bool:
        """Whether a result only objects to the call itself: an error entry, or a lone message and nothing else."""
        if not isinstance(observation, dict) or not observation:
            return False
        if observation.get('error'):
            return True
        return len(observation) == 1 and isinstance(next(iter(observation.values())), str)

    def detail_tools(self) -> set[str]:
        """Every store view that shows one listing by its id: the ones the brief named, and any other tool seen taking
        a listing id that only shows."""
        busy = {self.roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask', 'search', 'cart_view')} - {None}
        named = {self.roles.get(r) for r in ('open', 'offer')} - {None}
        named |= {tool for tool, _ in self.id_args}
        return {t for t in named if t not in busy and self.single_target(t)}

    def foreign_code(self, value: Any) -> bool:
        """Whether a value passed as a listing id is shaped unlike the ids this store's rows carry, as a part, model or
        series code printed inside a listing is. Learned from the episode's own rows; silent until most rows agree."""
        if not isinstance(value, str) or not value.strip() or value.strip() in self.rows or len(self.rows) < 5:
            return False
        shapes: dict[tuple[int, str], int] = {}
        for ident in self.rows:
            shape = _id_shape(ident)
            shapes[shape] = shapes.get(shape, 0) + 1
        top, count = max(shapes.items(), key=lambda kv: kv[1])
        return count >= ID_SHARE * len(self.rows) and _id_shape(value.strip()) != top

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or '') is True or bool(self.roles.get('offer'))))

    def gate_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_mark() + f'#{self.version}') not in self.gate_ok

    def tail(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        spare = 1 if self.buys == 0 else 0
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1)

    def judges(self, limit: float=SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None, deep: bool=False) -> dict[str, Any] | None:
        """One model reply. Providers limit requests per minute for each model, so a limited heavy model hands the call
        to the next of its kin at once, and rests a while; what is left of a limit does not count against the calls.
        Kin answer at low reasoning effort, which keeps a handed-over call as cheap and quick as a normal one, unless the
        call is `deep`."""
        if self.time_left() < EXPLORE_RESERVE + 15:
            models = list(dict.fromkeys((LIGHT if m == HEAVY else m for m in models)))
        for model in models:
            for choice in (HEAVY_KIN if model == HEAVY else (model,)):
                if self.calls >= ALL_CALLS or self.time_left() < CALL_RESERVE + 8:
                    return None
                if time.monotonic() < self.model_busy.get(choice, 0):
                    continue
                self.calls += 1
                tries = 0
                while True:
                    _llm.timeout = max(8, min(50, int(self.time_left() - CALL_RESERVE)))
                    effort = 'low' if model == HEAVY and choice != HEAVY and not deep else None
                    message, status = self.call_model(choice, messages, max_tokens, tools, effort)
                    if message is not None:
                        return message
                    if status == 402:
                        break
                    if status == 429 and model == HEAVY:
                        self.model_busy[choice] = time.monotonic() + BUSY_SECONDS
                        self.calls -= 1
                        break
                    passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                    if not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL:
                        break
                    if self.time_left() < WAITS[tries] + CALL_RESERVE + 10:
                        break
                    pause = WAITS[tries] * (0.8 + 0.4 * random.random())
                    self.waited += WAITS[tries]
                    tries += 1
                    time.sleep(pause)
                if status == 402:
                    self.no_credit = True
                    return None
                if status != 429:
                    break
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, effort: str | None=None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        if effort:
            body['reasoning'] = {'effort': effort}
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = _count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def ask_json(self, prompt: str, models: list[str], max_tokens: int=1800, deep: bool=False) -> dict[str, Any] | None:
        """One JSON object from the first model that returns one. Fixed instructions long enough for the provider to
        cache go in their own leading message, so it can reuse them across the calls of a run; the rest of the prompt
        is this call's own and goes as a plain part, which the cache marks leave alone. Shorter prompts go whole."""
        head = next((fixed for fixed in FIXED_HEADS if len(fixed) >= CACHE_CHARS and len(prompt) > len(fixed) and prompt.startswith(fixed)), '')
        chat = [{'role': 'system', 'content': head}, {'role': 'user', 'content': [{'type': 'text', 'text': prompt[len(head):].lstrip('\n') or prompt}]}] if head else [{'role': 'user', 'content': prompt}]
        for model in models:
            message = self.think(chat, [model], max_tokens, deep=deep)
            got = _json_obj((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        _env.timeout = max(5, min(95, int(self.time_left() - 3)))
        _env.max_retries = 3 if self.time_left() > EXPLORE_RESERVE else 1
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                return self.took(name, args, dict(result.data))
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
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
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + _dump(_clip(self.view.get('opening'), 1500))
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        elif not self.no_credit and self.shared_shelf(brief):
            again = self.ask_json(prompt, [HEAVY], 2500) or {}
            if _items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        self.read_browse(brief.get('browse'))
        query = brief.get('query_arg')
        search_props = (self.schemas.get(str((brief.get('roles') or {}).get('search') if isinstance(brief.get('roles'), dict) else '')) or {}).get('properties') or {}
        self.query_arg = query if isinstance(query, str) and query in search_props else None
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'One store tool, {buy}, either places the order or ends with no purchase, depending on its arguments. Schema: ' + _dump(self.schemas.get(buy)) + '\nReply with ONE JSON object and nothing else: {"buy_args": the fixed arguments that place the order, "refuse_args": the fixed arguments that end with no purchase} (only the arguments that select the mode).'
            fixed = self.ask_json(ask, [LIGHT, HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in _items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = _count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in _items(entry.get('musts')) if m]
            item = Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts)
            if isinstance(entry.get('plain'), list):
                said = {_words(str(m)) for m in entry['plain']}
                item.plain = {m for m in musts if _words(m) in said}
                item.flagged = set(musts)
            self.items.append(item)
        merged: list[Item] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and _words(m.shelf) == _words(item.shelf) and (not any((_words(a.split(':')[0]) == _words(b.split(':')[0]) and _words(a.partition(':')[2]) != _words(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                twin.plain |= item.plain
                twin.flagged |= item.flagged
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        if not self.items:
            self.items = [Item('the requested product', '', 1, [])]
        for pair in _items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(_count(pair.get('item')) or 0), became.get(_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
        for lookup in _items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = _words(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if _words(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.unmix()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = _count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = _count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in _items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, _count(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def read_browse(self, spec: Any) -> None:
        if not isinstance(spec, dict) or spec.get('tool') not in self.schemas:
            return
        props = self.schemas[spec['tool']].get('properties') or {}
        category, price = (spec.get('category'), spec.get('max_price'))
        if category not in props:
            return
        fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
        self.browse = {'tool': spec['tool'], 'category': category, 'price': price if price in props else None, 'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, price)}}

    @staticmethod
    def shared_shelf(brief: dict[str, Any]) -> bool:
        shelves = [_words(str(e.get('shelf') or '')) for e in _items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.ask_json(prompt, [HEAVY, LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in _items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.refs] or need in [str(x) for x in _items(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        item = self.items[number - 1]
        if need in item.plain:
            return False
        if any(({w for w in _stem(other.name) if len(w) >= 4} & _stem(value) for other in self.items if other is not item)):
            return False
        cat_key = self.category_key()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and (not self.owns(item, str(record.get(cat_key) or ''))):
                continue
            block = _fact_block(record) or {}
            if block:
                read += 1
                keys |= {_words(k) for k in block}
        return read >= 3 and (not any((_stem(k) & _stem(prop) for k in keys)))

    def fit_args(self, tool: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Arguments coerced to a tool's schema, or None when they cannot be made to fit it."""
        trouble: list[str] = []
        fixed = _shape(dict(args), self.schemas.get(tool), 'args', trouble)
        return None if trouble else fixed

    def visit(self, tool: str, args: dict[str, Any], read: bool=True) -> dict[str, Any] | None:
        """One store call the agent makes on its own: arguments fitted to the schema, the call sent, its result
        read into the ledger. The reply carries the arguments sent under '_args'; None when nothing was sent or the
        call did not go through."""
        fixed = self.fit_args(tool, args)
        if fixed is None or self.finished:
            return None
        reply = self.act(tool, fixed)
        if reply is None:
            return None
        reply['_args'] = fixed
        if read:
            self.observe(tool, reply.get('observation'))
        return reply

    def took(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Book a store reply: the step, the finish flag, and which listings each view has shown."""
        step = data.get('step')
        self.step = step if isinstance(step, int) and (not isinstance(step, bool)) else self.step + 1
        if data.get('finished'):
            self.finished = True
        seen = _unfold(data.get('observation'))
        data['observation'] = seen
        if not self.complaint(seen):
            asked = [v.strip() for value in args.values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)]
            if name in self.page_tools():
                self.opened.update(asked)
            if name in self.detail_tools():
                self.viewed.update(((name, v) for v in asked))
        return data

    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.described(number, m)]
            needs = [m + (f' (read from the page it points at: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store category: {item.shelf})'
            head += f', {item.qty} unit(s)'
            text = head + '; must have: ' + ('; '.join(needs) if needs else 'nothing beyond being this kind of product')
            if told:
                text += "; said in the shopper's own words, which no listing attribute carries (settles nothing): " + '; '.join(told)
            if item.free:
                text += '; the shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'per item' if brief.get('budget_scope') == 'each' else 'for the whole order'
            parts.append(f"budget: no more than {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('choice rule: the lowest-priced listing that fits')
        elif pick in ('most', 'least'):
            parts.append(f"choice rule: the {pick} {brief.get('pick_measure') or ''} among the fitting listings".replace('  ', ' '))
        else:
            parts.append('choice rule: any listing that meets everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('only listings in stock')
        for wish in _items(brief.get('wishes')):
            parts.append(f'wish (breaks ties only): {wish}')
        for rule in _items(brief.get('rules')):
            parts.append(f'the request says: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('when nothing fits: end without naming or proposing any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Orrin does it when you check out)', 'end': 'when ending without a purchase (Orrin does it then)'}[entry['when']]
            parts.append(f"a step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.page_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.page_tools() if t != facts))
            parts.append(f'{facts} shows listing attributes' + (f"; {', '.join(bare)} shows none" if bare else '') + (f"; {' or '.join(pages)} shows a listing's page (what a chain of pages or a page check refers to)" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def status(self) -> str:
        left = self.left()
        text = f'[Orrin] {self.step} of {self.max_steps} steps spent, {left} remain.'
        if left <= self.tail() + 1:
            text += ' LAST STEPS: check out now if the cart has a line for every item; end without a purchase only if your searches showed that some item has no fitting listing (running out of steps shows no such thing).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps remain: set the cart to your chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are getting short: settle your picks, fill the cart and check out within a few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half of the steps are gone and the cart is still empty: cart each item's fitting listing now (the store can still sell one out)."
        if self.turned:
            text += ' The store turned a finish down: apply its change and check out the best cart under the new terms; a no-purchase ending needs a requested item that those terms leave with no fitting listing at all.'
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[settle these first: open the listing each one points at and every listing its page sends you to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[listings the review thinks fit] ' + ', '.join(self.leads)
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        if self.time_left() < EXPLORE_RESERVE + 45:
            text += f'\n[clock] roughly {max(0, int(self.time_left()))} s of this episode remain: settle your picks and check out within the next few calls.'
        text += '\n[ledger]\n' + self.ledger()
        return text

    def ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((_id_like(r.get(self.id_key), True) for r in rows)):
            return True
        found = _learn_id(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    self.row_edits += 1
                    if ident not in self.rows:
                        self.seen_order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
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
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def category_key(self) -> str | None:
        """The row key that names a listing's category, worked out again only after the rows change."""
        stamp = (id(self.rows), len(self.rows), self.row_edits)
        if self.cat_memo is not None and self.cat_memo[0] == stamp:
            return self.cat_memo[1]
        self.cat_memo = (stamp, self.find_category_key())
        return self.cat_memo[1]

    def find_category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record:
                continue
            block = _fact_block(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _words(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                if sum((1 for v in values if len([p for p in re.split('[,;+]', v) if p.strip()]) >= 2)) * 2 >= len(values):
                    self.bundled.add(name)
                twin = next((spec for other, spec in kept if self.twins(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if not self.plain_need(number, must):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and _words(prop) != name and _stem(prop) & _stem(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def item_groups(self) -> list[list[str]]:
        cat_key = self.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.seen_order:
            label = str(self.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(_words(item.shelf or ''), _words(item.name), {w for w in _stem(item.shelf or '') | _stem(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((_words(label) in (shelf, name) or {w for w in _stem(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def single_target(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_of(self, ident: str) -> float | None:
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
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
        for ident, record in self.records_in(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(ident, (-1, set()))
            self.view_keys[ident] = (self.version, (have if version == self.version else set()) | keys)

    def shown_already(self, name: str, ident: str) -> bool:
        version, have = self.view_keys.get(ident, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (ident not in self.gone)

    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + _dump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + _dump(_clip(sample, 120))[:4000]
        got = self.ask_json(prompt, [LIGHT, HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def paging(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.price_of(i) for i, _ in self.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.paging(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        whole = pages <= SHORT_PAGES and self.pages_read + pages <= AUTO_PAGES + SHORT_PAGES and self.left() - self.tail() - pages >= 12
        if not whole and pages > 2 and self.pages_read + 2 <= AUTO_PAGES and (self.left() - self.tail() - 2 >= 12):
            pages = 2
        if not whole and (pages > 2 or self.pages_read + pages > AUTO_PAGES or self.left() - self.tail() - pages < 12):
            return f"\n[Orrin] {count} listings matched and {shown} are shown, in no price order. The choice rule weighs every fitting listing: narrow the search with the requirement's value, or fetch the next page with {param}={_dump(value)}."
        texts = []
        for _ in range(pages):
            reply = self.visit(name, {**args, param: value})
            if reply is None:
                break
            call = reply['_args']
            self.pages_read += 1
            seen = reply.get('observation')
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={_dump(value)}: {shown_text}')
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[Orrin] Of {count} matching listings, {shown * (len(texts) + 1)} rows are read so far; narrow the search with the requirement's value or fetch more pages for the rest." if value is not None and texts else ''
        return '\n[Orrin fetched the following pages of this search itself, one step per page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}
        limit_arg = self.browse['price'] if self.browse and self.browse.get('tool') == name else None
        cap = args.get(limit_arg) if limit_arg else None
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or (not self.price_key):
            return ''
        over = [i for i, _ in self.records_in(observation) if (self.price_of(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; to see only cheaper rows, list the category with ' + ', '.join((f'{k}={_dump(o[0])}' for k, o in others.items() if o)) + f' and keep {limit_arg}' if any(others.values()) else ''
        return f'\n[Orrin] {len(over)} rows here cost more than {limit_arg}={cap:g}, so this search ignored that limit{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Orrin] rows here that can be bought, lowest price first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and (not self.sold_out(i))]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or self.time_left() < CALL_RESERVE + 10:
                break
            reply = self.visit(tool, {target: ident})
            if reply is None:
                break
            seen = reply.get('observation')
            self.note_view(tool, seen)
            if isinstance(seen, dict) and self.ident(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(_fact_block(seen))
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def price_cap(self, price: float, same: bool=False) -> float:
        """A maximum-price value at the store's precision: the price itself (`same`) or the step just under it, kept
        whole when the store's price argument takes whole numbers."""
        whole = float(price).is_integer()
        cap: float = (int(price) if whole else price) if same else (int(price) - 1 if whole else round(price - 0.01, 2))
        spec = ((self.schemas.get(self.browse['tool']) or {}).get('properties') or {}).get(self.browse['price'] or '') if self.browse else None
        if isinstance(spec, dict) and 'integer' in _kinds(spec) and not float(cap).is_integer():
            cap = int(cap) + (1 if same else 0)
        return cap

    def descend(self, number: int, price: float, stop: int, same: bool=False) -> list[str] | None:
        """The item's category listed below a price, the bound lowered (to the cheapest quarter of the prices seen
        under it) until the listing is short enough to read to its last page, so the cheapest rows are seen whatever
        order the store lists them in. With `same` (a line that sold out), the listing runs up to that price itself
        and its first pages are read as they come. Returns every row read, or None when the store lists no category
        by price."""
        cap = self.price_cap(price, same)
        rows: list[str] | None = None
        for _ in range(1 if same else 3):
            if cap <= 0:
                break
            more = min(1, stop - self.step - 2) if same else min(SHORT_PAGES - 1, stop - self.step - 2)
            got = self.list_category(number, cap, max(0, more), whole=not same)
            if got is None:
                break
            rows = (rows or []) + got
            total = self.listed_total
            if same or total is None or total <= len(got) or self.step >= stop - 2:
                break
            below = sorted((p for p in (self.price_of(i) for i in rows if self.kind_of(i) == number) if p is not None and p < cap))
            cap = min(below[len(below) // 4] if len(below) >= 4 else cap / 2, cap * 0.8)
            cap = int(cap) if float(price).is_integer() else round(cap, 2)
        return rows

    def floor_sweep(self, lines: list[str] | None=None) -> str:
        """For a lowest-price pick, before checkout: list each cart line's category below the line's price (to the
        last page when the result is short; a value search where the store lists no category) and read the cheapest
        unread rows of that kind, so the final check weighs every cheaper candidate. Runs again after any change of
        the cart or the task. Given `lines` that sold out, it looks at their own price too, for a re-pick at that
        price. Returns words for the loop model only when no priced listing was possible."""
        if self.brief.get('pick') != 'cheapest':
            return ''
        mark = (self.cart_mark() if lines is None else _dump(sorted(lines))) + f'#{self.version}'
        room = self.left() - self.tail() - len(self.items) - 3
        if mark in self.swept or room < 2 or self.time_left() < EXPLORE_RESERVE + 25:
            return ''
        self.swept.add(mark)
        stop = self.step + min(SWEEP_STEPS + 2, room)
        lower: list[str] = []
        listed = False
        for line in self.cart_ids() if lines is None else lines:
            price, number = (self.price_of(line), self.kind_of(line))
            if price is None or not number or self.step >= stop:
                continue
            same = self.sold_out(line)
            rows = self.descend(number, price, stop, same)
            if rows is None and self.search_template():
                tool, base, field = self.search_template()
                rows = self.search_words(tool, base, field, ' '.join(self.value_words(number)[:2] + [self.items[number - 1].name]))
            else:
                listed = listed or rows is not None
            lower += [i for i in self.kind_rows(number, rows or [], price + 1e-06 if same else price) if i != line]
        unread = [i for i in dict.fromkeys(lower) if i not in self.full][:CHEAPER_OPENS]
        if lines is not None:
            unread = unread[:max(0, self.left() - self.tail() - 2)]
        if unread and self.left() >= self.tail() + len(unread) + 2:
            self.fill_facts(unread, self.tail() + 2)
        self.trace.append({'kind': 'floor_sweep', 'listed': listed, 'lower': lower[:8], 'opened': [i for i in unread if i in self.full], 'lines': lines})
        if lines is not None or listed or self.wider_held or self.left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        held = [f'{i} at {self.price_of(i):g}' for i in self.cart_ids() if self.price_of(i) is not None]
        return f"Not run yet (no step used). The choice rule is the lowest price and {self.left()} steps remain. Searches match titles, so a required value (which often sits only in the attributes) brings up few of the fitting listings, and rows arrive in no price order. Before checking out, search each item a few more times under other names for the product (its plain one-word name, synonyms, words from titles of fitting listings) and keep only rows that can be bought and cost less than the cart's line" + (f" ({'; '.join(held)})" if held else '') + '; read the attributes of the cheapest of those, swap in any that fits, then check out.'

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def widen(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.compare_bare():
            self.widen_singly()
            return
        if self.left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.item_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        if self.fit_args(tool, {target: picks}) is None:
            return
        before = len(self.full)
        reply = self.visit(tool, {target: picks})
        if reply is not None:
            seen = reply.get('observation')
            self.note_view(tool, seen)
            rows = self.records_in(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((_fact_block(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.widen_singly()
                return
            if len(self.full) == before and len(picks) > 2 and (self.left() > 12):
                faults = []
                args = _shape({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get('observation'))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = WIDEN_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.item_groups():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.fill_facts(first[:cap][:8], 22)
        self.fill_facts((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= 8 or self.time_left() < EXPLORE_RESERVE + 40:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, 'args', faults)
        return None if faults else args

    def interview(self) -> str:
        self.asked = True
        if self.msgs_free() <= 0:
            return ''
        if self.time_left() > EXPLORE_RESERVE + 30:
            self.widen()
        lanes = self.plan_lanes(self.msgs_free())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        follow: list[tuple[int, list[str], str]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        while self.msgs_free() > 0 and (not self.finished) and (self.left() > 6) and (self.time_left() > EXPLORE_RESERVE + 20):
            if follow:
                number, labels, worded = follow.pop(0)
            else:
                number, labels = self.next_ask(lanes, found, spent, used)
                worded = ''
                if not labels:
                    break
            used[number].update((_words(label) for label in labels))
            batch = self.send([(number, labels[0], worded or self.question(number, labels[0]))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            had = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if self.judges()[:1] == [HEAVY] and any((self.doubtful(entry) for entry in _items(heard.get('needs')))):
                heard = self.hear(batch, (HEAVY,)) or heard
            self.take_heard(heard, told)
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            if fresh or 1 not in [_count(x) for x in _items(heard.get('crowded'))]:
                continue
            if worded:
                continue
            sibling = self.look_alike(number, labels[0]) if self.msgs_free() >= 2 else ''
            if sibling and _words(labels[0]) not in self.alike_asked:
                self.alike_asked.add(_words(labels[0]))
                follow.insert(0, (number, [labels[0]], self.apart_question(number, labels[0], sibling)))
                continue
            head = labels[0].split()[-1]
            if len(labels[0].split()) >= 2 and _words(head) not in used[number]:
                follow.insert(0, (number, [head], ''))
        found_items = [n for n in found if found[n] > 0]
        searched = self.value_search(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = 'Questions the interviewer sent to the shopper, one step apiece:\n' + '\n'.join(lines)
        summary += '\nFirm requirements the answers added: ' + ('; '.join(told) if told else 'none')
        summary += '. Hold every pick to them (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' Their value comes from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, then open each listing that page sends you to; the last page gives the value, and the order only counts once all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes that arrived meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if k == self.query_arg]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.category_key()) or '').strip().lower() for r in self.rows.values()} - {''} if self.category_key() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.category_key()
                shelves = {_words(item.shelf or '') for item in self.items} | {_words(item.name) for item in self.items}
                shelves |= {_words(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (_words(v) not in shelves and (not any(({w for w in _stem(v) if len(w) >= 4} & {w for w in _stem(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [_dump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and _dump(a[k]) == best))
                return (tool, keep, field)
        return None

    def vocabulary(self) -> set[str]:
        """Words this episode's listings print as short attribute values (titles and descriptions left out)."""
        if self.vocab_at != len(self.rows) + len(self.full):
            words: set[str] = set()
            for record in list(self.rows.values()) + list(self.full.values()):
                stack: list[Any] = [record]
                while stack:
                    node = stack.pop()
                    if isinstance(node, dict):
                        stack += list(node.values())
                    elif isinstance(node, list):
                        stack += node
                    elif isinstance(node, str) and len(node.split()) <= 6:
                        words |= set(re.findall('[a-z0-9]{2,}', node.lower()))
            self.vocab = words
            self.vocab_at = len(self.rows) + len(self.full)
        return self.vocab

    def plain_need(self, number: int, need: str) -> bool:
        """Whether a requirement's value is a plain value a listing would print: as the model judged it when it read
        the requirement; when it gave no judgment, whether every word of the value is one this episode's listings
        print as an attribute value."""
        item = self.items[number - 1]
        if need in item.flagged:
            return need in item.plain
        if need in [m for _, m in self.refs] or ':' not in need:
            return False
        words = set(re.findall('[a-z0-9]{2,}', need.split(':', 1)[1].lower()))
        return bool(words) and words <= self.vocabulary()

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if value and self.plain_need(n, need):
                out.append(value)
        for need, wanted in self.relations.items():
            if any((n == number and m == need for n, m in self.refs)) and wanted and wanted not in out:
                out.append(wanted)
        return out

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22 or self.time_left() < EXPLORE_RESERVE + 25:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                if self.finished or self.fit_args(tool, {**base, field: ' '.join(words)}) is None:
                    break
                reply = self.visit(tool, {**base, field: ' '.join(words)})
                budget -= 1
                if reply is None:
                    break
                args = reply['_args']
                observation = reply.get('observation')
                cat_key = self.category_key()
                found = [i for i, _ in self.records_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Orrin searched each of these items with its required values together and read the first rows of the right kind] " + '\n'.join(seen)

    def next_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _words(label) not in used[number]]
            if left:
                return (number, left[:1])
        return (0, [])

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [_stem(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((_names(prop, _stem(key)) for prop in settled)) and (not re.search(NOT_PROPERTY, key))]
        return [key for key in keys if key not in self.bundled] + [key for key in keys if key in self.bundled]

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = _words(label)
        long = _roots(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == _words(item.shelf or ''):
                score = 3.0
            elif words == _words(item.name):
                score = 2.0
            else:
                mine = _roots(item.shelf or '') | _roots(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_of(label) == self.items.index(item) + 1)

    def records_of_item(self, number: int) -> int:
        cat_key = self.category_key()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.owns(item, str(record.get(cat_key) or '')) and _fact_block(record)))

    def plan_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.key_table()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - already fixed by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.not_stated(label, name, values) + (' [many features in one]' if name in self.bundled else ''))
        prompt = PLAN + f'\n\nPICKS PER PRODUCT: {max(1, count)}' + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in _items(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request wants these checked with the shopper):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.ask_json(prompt, [HEAVY, LIGHT], 3500, deep=True) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((_words(label) != _words(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(_words(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_keys(number, table)
            read = max(1, self.records_of_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_of(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _words(text)))
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
            picked = {_words(x) for x in lane}
            for key in [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and (k not in self.bundled)][:GUARD_KEYS]:
                if any((_words(key) == _words(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _words(other) not in hinted[number] and share_of(other) + GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _words(other) != _words(key)]
                if weak is not None and len(self.items) > 1:
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
            lanes[number] = [x for x in lanes[number] if _words(x) not in self.bundled or _words(x) in picked] + [x for x in lanes[number] if _words(x) in self.bundled and _words(x) not in picked]
            if count <= TIGHT_MESSAGES and read >= 3:
                sure = [x for x in lanes[number] if _words(x) in hinted[number] or share_of(x) >= 0.5]
                lanes[number] = sure + [x for x in lanes[number] if x not in sure]
        self.trace.append({'kind': 'plan', 'ask': _clip(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def doubtful(self, entry: Any) -> bool:
        """Whether a stated answer deserves a second reading by the heavy model: it points at another listing (a
        value copied word for word), or a long value carries words no opened listing shows for that property."""
        if not isinstance(entry, dict):
            return False
        prop, _, value = str(entry.get('need') or '').partition(':')
        words = set(re.findall('[a-z0-9]+', value.lower()))
        if entry.get('points') is True:
            return True
        if len(words) < 4:
            return False
        seen: set[str] = set()
        for record in self.full.values():
            for key, shown in (_fact_block(record) or {}).items():
                if _names(_stem(prop), _stem(key)) or _names(_stem(key), _stem(prop)):
                    seen |= set(re.findall('[a-z0-9]+', str(shown).lower()))
        return bool(words - seen)

    def question(self, number: int, label: str) -> str:
        return f'What {label} would you like for the {self.items[number - 1].name}?'

    def look_alike(self, number: int, label: str) -> str:
        """Another key of this item's listings whose name shares a word with the asked property (so a shopper may hear
        two subjects in one question), taken from this episode's own keys; the most carried one."""
        mine = set(_words(label).split())
        best = ('', 0)
        keys = []
        for shelf, key, carried, _ in self.key_table():
            if self.owner_of(shelf) != number and len(self.items) > 1:
                continue
            if key != _words(label):
                keys.append(key)
            if key != _words(label) and set(key.split()) & mine and carried > best[1]:
                best = (key, carried)
        if best[0] or not keys or self.cost >= SPEND_SOFT:
            return best[0]
        got = self.ask_json(ALIKE + f'\n\nPROPERTY ASKED: {label}\nPRODUCT: {self.items[number - 1].name}\nKEYS: ' + ' | '.join(keys[:14]), [LIGHT], 300) or {}
        pick = _words(str(got.get('key') or ''))
        return pick if pick in keys else ''

    def apart_question(self, number: int, label: str, sibling: str) -> str:
        return f'Not counting the {sibling}, what {label} would you like for the {self.items[number - 1].name}?'

    def send(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
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
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: sent the shopper {text!r}, who answered {words[:300]!r} (cart at that point: {self.cart_mark()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if isinstance(entry.get('plain'), bool) and entry.get('points') is not True:
                self.items[number - 1].flagged.add(need)
                if entry['plain']:
                    self.items[number - 1].plain.add(need)
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in _items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
                gone = _words(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in _items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, _count(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished or (not self.gate_due(stage)):
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(GATE_ROUNDS):
            prompt = GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() + '\n\nRECORD (messages and step calls up to now):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES REMAINING: {max(0, self.msg_limit - self.msg_used)}   STEPS REMAINING: {self.left()} (the {('order' if stage == 'order' else 'ending')} itself uses one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': _clip(call, 200), 'steps': _clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('First Orrin carried out the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's answer changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Adjust the cart to it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in _items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((_count(n) or 0) in proof for n in _items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': _clip(unproven, 120)})
                self.done_log.append(f'(Orrin: since the cart last changed at step {self.cart_at}, no line shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' as done, while {self.msg_limit - self.msg_used} message(s) and {self.left()} steps remain: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.version}'))
                break
            if self.left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = _shape(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, '', text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.act(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.absorb(seen)
            self.done_log.append(f'step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {_dump(args)} -> {self.show(seen)[:400]}')
        return (True, 'First Orrin carried out the steps the request asks for:\n' + '\n'.join(said) if said else '')

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
        return ' '.join(found)[:800] if found else _dump(observation)[:400]

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.ident(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 2500) or {}
        self.note_kind = str(got.get('kind') or '')
        gone = [str(g) for g in _items(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in _items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in _items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if _words(m.split(':')[0]) == _words(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, _count(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

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
                if ident and _fact_block(node) and (ident not in ids):
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
            return ''
        many = len(self.items) > 1
        lines = []
        for ident in ids[:12]:
            block = _fact_block(self.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if _names(_stem(prop), _stem(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names carry a requirement's property; a key for a different property (another part or aspect) settles nothing, and a key under another name can still hold the same property] " + ' | '.join(lines)

    def cart_text(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return _dump(_clip(self.cart_view, 300))[:3000]
        return "lines as Orrin logged the cart calls (listing: units): " + _dump(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_count(v) for k, v in node.items() if isinstance(v, int) and (not isinstance(v, bool)) and k != self.price_key]
                        if len(counts) == 1 and counts[0] is not None:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def cart_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = _dump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def record_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        return _dump(_clip(record, 400))[:3000] if record else '(no record seen)'

    def others(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.price_of(i) if self.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.sold_out(ident):
                continue
            record = self.full.get(ident) or self.rows.get(ident)
            lines.append(_dump(_clip(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def item_ids(self, number: int) -> list[str]:
        """Listings seen so far whose category makes them the given requested item, oldest first."""
        return [i for i in self.seen_order if self.category_key() and self.kind_of(i) == number]

    def fit_faults(self) -> list[str]:
        """Parts in the cart that the records say do not work with the device in the cart, judged by the light model
        from both records (lists of models, shared codes, stated exceptions); once per cart state."""
        if not self.bound:
            return []
        cart = self.cart_ids()
        faults: list[str] = []
        for part, main in self.bound:
            kin = self.item_ids(part)
            devices = [i for i in cart if self.kind_of(i) == main]
            parts = [i for i in cart if i in kin]
            if len(devices) != 1 or not parts:
                continue
            mark = devices[0] + '|' + '|'.join(sorted(parts)) + f'#{self.version}'
            if mark not in self.fit_seen:
                pool = [i for i in reversed(kin) if i not in parts and not self.sold_out(i)][:12]
                prompt = FITS + '\n\nDEVICE:\n' + f'{devices[0]}: {self.record_of(devices[0])}' + '\n\nPARTS IN THE CART:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in parts))
                if pool:
                    prompt += '\n\nOTHER PART LISTINGS SEEN:\n' + '\n'.join((f'{i}: ' + _dump(_clip(self.full.get(i) or self.rows.get(i), 160))[:700] for i in pool))
                got = self.ask_json(prompt, [LIGHT], 1500) if self.cost < SPEND_HARD else None
                said: list[str] = []
                others = [str(o) for o in _items((got or {}).get('others')) if str(o) in self.rows and str(o) not in parts and not self.sold_out(str(o))][:4]
                for entry in _items((got or {}).get('parts')):
                    if isinstance(entry, dict) and str(entry.get('id') or '') in parts and str(entry.get('works') or '').lower() == 'no':
                        why = str(entry.get('why') or '').strip()[:120]
                        said.append(f"{entry['id']} ({self.items[part - 1].name}) does not work with {devices[0]}" + (f' ({why})' if why else '') + (f"; part listings seen that do: {', '.join(others)}" if others else ''))
                self.fit_seen[mark] = said
                self.trace.append({'kind': 'fit_check', 'device': devices[0], 'parts': parts, 'faults': said[:3]})
            faults += self.fit_seen[mark]
        return faults

    def line_faults(self) -> list[str]:
        cat_key = self.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'the cart has no line for item {n} ({item.name})')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) takes up {len(lines)} lines ({', '.join(lines)}); keep only one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} sits at {have} unit(s) while item {n} ({item.name}) needs {item.qty}')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is no kind of product the shopper asked for: remove it")
        for i in ids:
            if self.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, more than the per-item limit of {self.budget:g}' for i in over]
            else:
                total = sum((prices[i] * self.held_units(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the total of {total:g} is above the budget of {self.budget:g}')
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.category_key()
        if not cat_key:
            return []
        out = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.owns(item, str((self.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.items[owners[0] - 1].qty) and (not self.sold_out(ident)):
                out.append((ident, have, self.items[owners[0] - 1].qty))
        return out

    def cart_modes(self, name: str, mode: str) -> tuple[Any, Any]:
        """The values of a cart tool's mode argument that add and that take out units: read once by the light model
        from the schema, else the schema's first and second options."""
        options = (((self.schemas.get(name) or {}).get('properties') or {}).get(mode) or {}).get('enum') or [None, None]
        options = list(options) + [options[-1]] * max(0, 2 - len(options))
        if self.modes is None:
            self.modes = {}
            got = self.ask_json(MODES + '\n\nTOOL: ' + name + '\nSCHEMA: ' + _dump(self.schemas.get(name) or {}), [LIGHT], 600) or {}
            if got.get('argument') == mode and got.get('add') in options and got.get('remove') in options and got.get('add') != got.get('remove'):
                self.modes = {'add': got['add'], 'remove': got['remove']}
        if self.modes:
            return (self.modes['add'], self.modes['remove'])
        return (options[0], options[1])

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in _kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.cart_sets(name):
                calls.append({target: ident, count: want})
            elif mode:
                on, off = self.cart_modes(name, mode)
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _shape(args, schema, 'args', faults)
            if faults:
                break
            reply = self.act(name, args)
            if reply is None or self.finished:
                break
            self.note_cart(name, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(_dump(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def check_cart(self, ids: list[str] | None=None, limit: float=SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or not self.judges(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        if lead:
            quantity = '; '.join((f'item {n}: {it.qty} unit(s)' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned: judge these listings as if carted at the requested quantities, ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.sold_out(i) else '') + self.record_of(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 10)
        got = self.ask_json(prompt, self.judges(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        self.better = []
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get('problems'))][:5]
        better = [str(b) for b in _items(got.get('better')) if str(b) in self.rows and (not self.sold_out(str(b)))][:4]
        self.better = better
        text = 'A second reading of the records says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look like better picks: ' + ', '.join(better)
        return text + '. Fix the cart (or re-read the records if the reading is wrong), then check out.'

    def trail(self) -> tuple[str, bool]:
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.left() > 15 else 4) and (self.cost < SPEND_HARD):
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                record = self.full.get(ident) or self.rows.get(ident) if ident in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + _dump(_clip(record, 300))[:1400])
            for ident in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if ident not in self.opened or not (self.full.get(ident) or self.rows.get(ident)):
                    records.append('ROW ' + _dump(_clip(self.rows.get(ident), 80))[:300])
            prompt = TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.ask_json(prompt, [LIGHT] + ([HEAVY] if self.cost + HEAVY_ROOM <= SPEND_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': _clip(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = self.related(need, str(chain['value']).strip(), chain)
                    self.version += 1
                    continue
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = self.related(need, str(chain.get('value') or '').strip(), chain) or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.left() > 12) and (self.link_searches < 3):
                reach = self.find_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page from the shop this chain names, showing the values its description gives; take the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.left() < len(reach) + 4) or self.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                reply = self.visit(tool, {target: ident})
                if reply is None:
                    break
                seen.append(f'{ident}: ' + self.show(reply.get('observation'), True)[:1500])
                if self.finished:
                    break
        text = ''
        if seen:
            text += '[Orrin opened the pages these requirements point to]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values taken from the pages that requirements point to] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.left() > 15 else 2) and (self.left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order only counts once every page a requirement points to is opened in full (a search row does not count), following each page on to the next listing it names. Not found and opened yet: ' + ' | '.join(lost[:3]) + ". Search for it with the shop's name plus the values its description gives (that shop's other listings of the same kind are not it), open the row from that shop whose page shows those values, apply the value that page gives, then check out."
        return (text, hold)

    def related(self, need: str, value: str, chain: dict[str, Any]) -> str:
        """The value a resolved chain gives, with the relation the model read from the pages applied in words: the
        same value, any value but it, its stated partner, a window around it, or nothing at all for a decoy."""
        if not value:
            return value
        relation = str(chain.get('relation') or '').strip().lower()
        wanted = str(chain.get('wanted') or '').strip()
        self.trace.append({'kind': 'relation', 'need': need[:120], 'relation': relation, 'value': value[:80], 'wanted': wanted[:80]})
        if relation == 'same':
            self.relations[need] = wanted or value
            return f'{value} (as Orrin reads the pages, the same {wanted or value} on the listing to buy; the requirement\'s own words decide)'
        if relation == 'paired' and wanted:
            self.relations[need] = wanted
            return f'{value} (as Orrin reads the pages, {value} goes with {wanted}, so {wanted} on the listing to buy; the requirement\'s own words decide)'
        if relation == 'exclude':
            return f'{value} (as Orrin reads the pages, any value but {value} on the listing to buy; the requirement\'s own words decide)'
        if relation == 'window':
            return f'{value} (as Orrin reads the pages, a value within the stated distance of {value}; the requirement\'s own words decide)'
        if relation == 'decoy':
            return f'{value} (as Orrin reads the pages, that page only says a version exists elsewhere and fixes no value)'
        return value

    def shop_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(OWNER_KEY, str(k), re.I)]
            if ident not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def opened_shop(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.opened:
            row = self.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(OWNER_KEY, str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def find_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for ident in self.shop_rows(about, list(self.rows)) + self.opened_shop(about) for k, v in (self.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search(OWNER_KEY, str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'its', 'for', 'of'}
        for ident in self.opened_shop(about):
            stack, leaves = ([self.full.get(ident) or self.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.search_template()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.shop_rows(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            if self.fit_args(tool, {**base, field: words}) is not None:
                self.link_searches += 1
                reply = self.visit(tool, {**base, field: words})
                if reply is not None:
                    observation = reply.get('observation')
                    self.link_seen[memo] = [i for i, _ in self.records_in(observation)]
                    known = self.shop_rows(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.shop_rows(about, list(reversed(self.seen_order)))))[:2]

    def offer_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            if self.fit_args(tool, {target: ident}) is None:
                break
            before = dict(self.rows.get(ident) or {})
            reply = self.visit(tool, {target: ident})
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[ident] = self.step
            after = self.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_dump(_clip(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        self.floor_sweep(lost)
        picks = self.replacements(lost)
        text = f"[Orrin] Now sold out: {', '.join(lost)} (the store shows it can no longer be bought). It will not return, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Only a few steps remain: in this turn add {picks[0]} at the requested quantity and take out the sold-out line (two calls), then check out. Other listings already read that seem to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Already-read listings that seem to fit that item instead, lowest price first: {', '.join(picks)}. The choice rule is the lowest price, so unless you have seen the whole fitting set already, run one search below {picks[0]}'s price (with the search's maximum price, if it takes one) before replacing the sold-out line with the cheapest fitting listing."
        if picks:
            return text + f" Already-read listings that seem to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Take that line out of the cart and add the best other listing that fits; search for one when none you have read fits.'

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any((self.sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        """Listings to take a sold-out line's place: first from the ledger (the review's fits, then opened listings of
        that item whose own keys break no plain requirement and that meet the terms now in force), and only when the
        ledger has none, from the heavy model's reading of the records seen."""
        known: list[str] = []
        for ident in lost:
            number = self.kind_of(ident)
            if number:
                known += [i for i in self.candidates(number) if i not in lost and i not in self.cart_ids() and i not in known and (i in self.lead_items.get(str(number), []) or (i in self.full and self.judged(i, number)[1] == 0))]
        if known:
            if self.brief.get('pick') == 'cheapest':
                known.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
            self.trace.append({'kind': 'swap_ledger', 'lost': lost, 'picks': known[:4]})
            return known[:4]
        if not self.judges():
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
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.record_of(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.ask_json(prompt, self.judges(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and (ident not in self.gone) and (ident not in self.cart_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def audit(self) -> str:
        """The evidence review before a decline: the heavy model checks every record against the requirements, a
        recovery search runs for each item it finds no fit for, and the review looks again. Items still without a fit
        after that are the decline's proof; the others get their fitting and nearest listings on record."""
        if self.audited or not self.judges():
            return ''
        self.audited = True
        self.reviewed_at = self.version
        got = self.audit_call()
        self.take_review(got)
        named = self.audit_fits(got)
        missing = [n for n in range(1, len(self.items) + 1) if not named.get(str(n))]
        if missing:
            rows = [str(i) for i in _items((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.sold_out(str(i)))]
            opened = bool(rows) and self.open_rows(rows)
            searched = self.recovery_search(got, missing)
            if searched or opened:
                again = self.audit_call()
                if again is not None:
                    got = again
                    self.take_review(got)
                    named = self.audit_fits(got) or named
        self.near_items = self.audit_fits({'fits': (got or {}).get('near')})
        if named or (isinstance(got, dict) and 'fits' in got):
            self.lead_items = named
            self.leads = [i for ids in named.values() for i in ids]
        if not named:
            return ''
        return 'An evidence review found listings that may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them where needed and buy whichever truly fit; a no-purchase ending is acceptable only once each of them has been shown to fail.'

    def audit_call(self) -> dict[str, Any] | None:
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        cat_key = self.category_key()
        opened: list[str] = []
        rows: list[str] = []
        for strict in (True, False):
            for ident in reversed(self.seen_order):
                label = str((self.rows.get(ident) or {}).get(cat_key) or '') if cat_key else ''
                if self.sold_out(ident) or (strict and ((cap is not None and (self.price_of(ident) or 0) > cap) or (label and (not any((self.owns(item, label) for item in self.items)))))):
                    continue
                record = self.full.get(ident) if ident in self.full else None
                if record and len(opened) < 16:
                    opened.append('OPENED ' + _dump(_clip(record, 150))[:850])
                elif not record and len(rows) < 15:
                    rows.append('ROW ' + _dump(_clip(self.rows.get(ident), 100))[:400])
            if opened or rows:
                break
        if not opened and not rows:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        answers = [line for line in self.done_log if 'sent the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers)
        tried = [str(v) for args in self.searched for v in args.values() if isinstance(v, str)]
        if tried:
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(tried))
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        got = self.ask_json(prompt, self.judges(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        self.reviewed = self.reviewed or (isinstance(got, dict) and 'fits' in got)
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        """Apply what the decline review found wrong with the ledger: story set aside, values corrected. A requirement
        the shopper stated in an answer can be corrected but not set aside, and a lookup is left alone."""
        lookups = {need for _, need in self.refs}
        changed = []
        aside: set[int] = set()
        for entry in _items((got or {}).get('not_firm')):
            if isinstance(entry, dict):
                number, need = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip())
                if 1 <= number <= len(self.items) and number not in aside and need in self.items[number - 1].musts and need not in self.items[number - 1].dropped and need not in lookups:
                    aside.add(number)
                    self.items[number - 1].dropped.append(need)
                    changed.append(f'set aside {need}')
        for entry in _items((got or {}).get('corrected')):
            if isinstance(entry, dict):
                number, need, value = (_count(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip())
                if 1 <= number <= len(self.items) and need in self.items[number - 1].musts + self.items[number - 1].told and need not in lookups and ':' in value:
                    self.items[number - 1].dropped.append(need)
                    self.items[number - 1].told.append(value)
                    changed.append(f'{need} read as {value}')
        if changed:
            self.version += 1
            self.trace.append({'kind': 'review', 'changed': changed})
            self.notes.append('a review of the request: ' + '; '.join(changed))

    def shelf_of(self, number: int) -> str:
        """The category label of a requested item: the one most of its rows show, else the brief's."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        labels = [str(r.get(cat_key) or '') for r in self.rows.values()] if cat_key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def list_category(self, number: int, cap: float | None, more: int=0, whole: bool=False) -> list[str] | None:
        """The rows the store's category listing shows for an item, under a price limit when one is given, with up to
        `more` further pages (with `whole`, only when those pages reach the listing's end). The listing's reported
        total is kept in `listed_total`. None when the store has no such listing or it does not answer."""
        spec = self.browse
        label = self.shelf_of(number)
        if not spec or not label or self.browse_misses >= 2 or (cap is not None and (not spec['price'])):
            return None
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        reply = self.visit(spec['tool'], args)
        if reply is None:
            self.browse_misses += 1
            return None
        args = reply['_args']
        seen = reply.get('observation')
        found = [i for i, _ in self.records_in(seen)]
        if isinstance(seen, dict) and seen.get('error'):
            self.browse_misses += 1
            return None
        first = self.paging(spec['tool'], seen)
        self.listed_total = first[0] if first else None
        if whole and first and first[0] > first[1] * (more + 1):
            more = 0
        for _ in range(more):
            page = self.paging(spec['tool'], seen)
            if not page or page[3] is None or self.finished:
                break
            reply = self.visit(spec['tool'], {**args, page[2]: page[3]})
            if reply is None:
                break
            seen = reply.get('observation')
            found += [i for i, _ in self.records_in(seen)]
        return found

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        reply = self.visit(tool, {**base, field: words})
        if reply is None:
            return []
        self.searched = (self.searched + [dict(reply['_args'])])[-8:]
        return [i for i, _ in self.records_in(reply.get('observation'))]

    def kind_rows(self, number: int, found: list[str], below: float | None=None) -> list[str]:
        """Buyable rows of the item's kind, unread first, most of the requirement's words first, then cheapest."""
        cat_key = self.category_key()
        item = self.items[number - 1]
        wanted = {w for w in re.findall('[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower())}
        rows = [i for i in dict.fromkeys(found) if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))]
        if below is not None:
            rows = [i for i in rows if (self.price_of(i) if self.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in self.full, -len(wanted & set(re.findall('[a-z0-9]{3,}', _dump(self.rows.get(i) or {}).lower()))), self.price_of(i) or 0))

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        """Before a decline stands: list each unfitted item's category and run the searches the review proposes, then
        read the most promising new rows, so that the second look at the evidence has something new to judge. Half of
        the steps it may use are kept for reading; the cart and the checkout stay possible. With no steps to spare,
        an item whose shelf this episode already read (four of its rows seen, two of them opened) counts as searched."""
        keep = self.tail() + len(self.items) + 2
        room = min(RECOVERY_STEPS, self.left() - keep)
        if room < 3 or self.time_left() < EXPLORE_RESERVE + 25:
            for number in missing:
                kin = [i for i in self.seen_order if self.kind_of(i) == number]
                if len(kin) >= 4 and sum((1 for i in kin if i in self.full)) >= 2:
                    self.proof[number] = self.proof.get(number, 0) + 1
            return False
        before = len(self.full)
        last = self.step + room
        searching = self.step + max(1, room // 2)
        plan = self.search_template()
        queries: list[tuple[int, str]] = []
        for entry in _items((got or {}).get('search')):
            if isinstance(entry, dict) and str(entry.get('query') or '').strip():
                queries.append((min(max(1, _count(entry.get('item')) or 1), len(self.items)), str(entry['query']).strip()))
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_words(n)[:2] + [self.items[n - 1].name])) for n in missing]
        for number in missing:
            values = self.value_words(number)
            for k in range(len(values) if len(values) > 1 else 0):
                loose = ' '.join(values[:k] + values[k + 1:][:1] + [self.items[number - 1].name])
                if (number, loose) not in queries:
                    queries.append((number, loose))
            self.proof[number] = self.proof.get(number, 0) + 1
        found: dict[int, list[str]] = {n: [] for n in missing}
        for number in missing[:3]:
            if self.step < searching:
                cap = self.budget if isinstance(self.budget, (int, float)) and self.browse and self.browse['price'] else None
                found[number] += self.list_category(number, cap) or []
        for number, words in queries[:5]:
            if plan and self.step < searching:
                found.setdefault(number, [])
                found[number] += self.search_words(plan[0], plan[1], plan[2], words)
        for number, rows in found.items():
            picks = [i for i in self.kind_rows(number, rows) if i not in self.full][:2]
            self.fill_facts(picks[:max(0, last - self.step)], keep)
        self.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(self.full) - before})
        return len(self.full) > before

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and (not self.sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = (self.compare_target(), self.roles.get('compare'))
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and (not self.compare_bare()):
            plan = [(joiner, {many: rows[:self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:3]]
        else:
            return False
        if self.left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            if self.fit_args(name, args) is None:
                return False
            reply = self.visit(name, args)
            if reply is None or self.finished:
                break
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in _kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in _kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in _kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: self.cart_modes(name, mode)[1], **({count: have} if count else {})})
            elif count and self.cart_sets(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_sets(name)):
                return None
            calls.append({target: ident, **({mode: self.cart_modes(name, mode)[0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def record_finish(self, name: str, args: dict[str, Any], reply: dict[str, Any] | None) -> bool:
        """Book a finishing call's outcome. Returns whether the world turned the finish down: the episode goes on and
        the reply carries more than an objection to the call itself (or a note that changes the task)."""
        if reply is None:
            return False
        if self.is_buy(name, args):
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
        if reply.get('finished') or self.finished:
            return False
        seen = reply.get('observation')
        if self.complaint(seen) and self.note_kind != 'change':
            return False
        self.turned += 1
        if self.is_refuse(name, args) and not self.is_buy(name, args):
            self.decline_turned = True
            self.gap_done = False
        else:
            self.audited = self.reviewed = self.gap_done = False
            self.proof = {}
        self.trace.append({'kind': 'turned_down', 'tool': name, 'count': self.turned})
        return True

    def judged(self, ident: str, number: int) -> tuple[int, int]:
        """How many of an item's plain requirements a listing's own keys meet and break: a key naming the property
        whose value holds every word of the required value meets it, one sharing none of its words breaks it."""
        block = _fact_block(self.full.get(ident) or {}) or {}
        met = broke = 0
        for n, need in self.needs_now():
            if n != number or not self.plain_need(n, need):
                continue
            prop, _, value = need.partition(':')
            want = set(re.findall('[a-z0-9]+', value.lower()))
            hits = [str(v).lower() for k, v in block.items() if _names(_stem(prop), _stem(k)) or _names(_stem(k), _stem(prop))]
            if not want or not hits:
                continue
            have = set(re.findall('[a-z0-9]+', ' '.join(hits)))
            if want <= have:
                met += 1
            elif not want & have:
                broke += 1
        return (met, broke)

    def kind_of(self, ident: str) -> int:
        cat_key = self.category_key()
        if not cat_key:
            return 1 if len(self.items) == 1 else 0
        return self.owner_of(str((self.rows.get(ident) or {}).get(cat_key) or ''))

    def missing_items(self) -> list[int]:
        """Requested items with no line in the cart, when the store's rows show which item a listing is."""
        if not self.category_key():
            return []
        held = {self.kind_of(i) for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and not self.sold_out(i)}
        return [n for n in range(1, len(self.items) + 1) if n not in held]

    def usable(self, ident: str, number: int) -> bool:
        """Whether a listing can still be bought for an item under the terms now in force: known, in stock, not sold
        out, and inside the budget at the item's quantity."""
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        price = self.price_of(ident)
        qty = self.items[number - 1].qty if self.brief.get('budget_scope') != 'each' else 1
        return ident in self.rows and self.buyable(ident) and (cap is None or price is None or price * qty <= cap * (1 + 1e-09))

    def candidates(self, number: int) -> list[str]:
        """Known listings for an item, best first: the review's fits, then its nearest misses, then opened listings
        by how many plain requirements their keys meet (fewest broken first), cheapest first for a cheapest pick."""
        cheap = self.brief.get('pick') == 'cheapest'
        fits = [i for i in self.lead_items.get(str(number), []) if self.usable(i, number)]
        if cheap:
            fits.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float('inf'))
        near = [i for i in self.near_items.get(str(number), []) if self.usable(i, number) and i not in fits]
        rest = [i for i in self.seen_order if i in self.full and self.kind_of(i) == number and self.usable(i, number) and i not in fits and i not in near]
        shelf = _words(self.shelf_of(number))
        cat_key = self.category_key()

        def rank(ident: str) -> tuple[Any, ...]:
            met, broke = self.judged(ident, number)
            label = _words(str((self.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else shelf
            return (label != shelf, broke, -met, self.price_of(ident) if cheap and self.price_of(ident) is not None else 0.0)
        rest.sort(key=rank)
        return fits + near + rest

    def basket_plan(self) -> tuple[dict[str, int], list[int]]:
        """The best known basket: each item keeps its cart line when it has one, else takes its first candidate.
        Returns the lines and the items left without any listing."""
        plan: dict[str, int] = {}
        empty: list[int] = []
        cart = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and not self.sold_out(i)]
        for number, item in enumerate(self.items, 1):
            mine = [i for i in cart if self.kind_of(i) == number and i not in plan and self.usable(i, number) and self.judged(i, number)[1] == 0]
            pool = mine + self.candidates(number)
            pool = [i for i in pool if i not in plan]
            if not pool:
                empty.append(number)
                continue
            plan[pool[0]] = item.qty
        return (plan, empty)

    def decline_proven(self, numbers: list[int] | None=None) -> bool:
        """Some requested item has no fitting listing on record: the evidence review ran under the terms now in force,
        the item was searched again (or its shelf was already read when no step was left for that), and the review
        still names no listing that fits it."""
        if self.decline_turned or not self.audited or not self.reviewed:
            return False
        items = range(1, len(self.items) + 1) if numbers is None else numbers
        return any((str(n) not in self.lead_items and self.proof.get(n, 0) > 0 for n in items))

    def review_silent(self) -> bool:
        """The evidence review ran but gave no verdict (no reply or an unreadable one), and no ending without a purchase
        was turned down earlier: then the agent's own decline stands, as it would without the review."""
        return self.audited and not self.reviewed and not self.decline_turned

    def order_best(self, why: str) -> str:
        """Set the cart to the best known basket and check out, instead of ending without a purchase. One swap is
        taken from the final check when it names a better listing; the order goes ahead either way, since a decline
        the store does not support earns nothing."""
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if not cart or not buy or self.finished or self.ordered_best >= 2 or self.left() < 2:
            return ''
        plan, empty = self.basket_plan()
        if empty or not plan:
            self.trace.append({'kind': 'order_best_none', 'why': why, 'empty': empty})
            return ''
        loose = [i for i in plan if i not in self.cart_ids()]
        if loose and self.brief.get('pick') == 'cheapest' and self.left() > 2 * len(plan) + 8 and self.time_left() > EXPLORE_RESERVE + 30:
            self.floor_sweep(loose)
            swept, gaps = self.basket_plan()
            plan = swept if swept and not gaps else plan
        self.ordered_best += 1
        unread = [i for i in plan if i not in self.full]
        if unread and self.left() > len(plan) + 4:
            self.fill_facts(unread, len(plan) + 3)
        reviewed = all((i in self.lead_items.get(str(self.kind_of(i)), []) or i in self.cart_ids() for i in plan))
        if not reviewed and self.left() > len(plan) + 3 and self.time_left() > CALL_RESERVE + 30:
            self.better = []
            self.check_cart(list(plan))
            swap = [b for b in self.better if b not in plan and self.kind_of(b)]
            if swap:
                kind = self.kind_of(swap[0])
                old = next((i for i in plan if self.kind_of(i) == kind), None)
                if old is not None and old not in self.cart_ids():
                    plan = {(swap[0] if i == old else i): n for i, n in plan.items()}
        drop = [i for i in self.cart_ids() if i not in plan]
        calls = self.cart_calls(plan, drop)
        if calls is None or len(calls) + 1 > self.left():
            self.trace.append({'kind': 'order_best_blocked', 'why': why, 'plan': plan})
            return ''
        done = []
        for args in calls:
            args = self.fit_args(cart, args)
            if args is None:
                return ''
            reply = self.act(cart, args)
            if reply is None or self.finished:
                return ''
            self.note_cart(cart, args)
            self.cart_said = _dump(reply.get('observation'))
            self.observe(cart, reply.get('observation'))
            done.append(_dump(args))
        args = self.fit_args(buy, dict(self.brief.get('buy_args') or {}))
        if args is None or self.left() < 1:
            return ''
        said = ''
        if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
            go, said = self.gate('order')
            if not go:
                return 'Orrin set the cart to the best known basket (' + '; '.join(done) + ') because ' + why + '; then: ' + said
        reply = self.act(buy, args)
        self.trace.append({'kind': 'order_best', 'why': why, 'lines': plan, 'drop': drop, 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        self.note_kind = ''
        self.observe(buy, reply.get('observation'))
        self.record_finish(buy, args, reply)
        text = 'Not run (Orrin acted instead): ' + why + ', so Orrin set the cart to the best known basket (' + ('; '.join(done) or 'as it was') + ') and checked out. Store reply: ' + self.show(reply.get('observation'))
        return (said + '\n' if said else '') + text


    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def offered(self) -> list[dict[str, Any]]:
        left = self.left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in _kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, no listing in this store meets every one of your requirements, so I am not placing an order.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        named = {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None}
        if name not in named | self.detail_tools() or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in _kinds(v) or 'array' in _kinds(v)) and ('enum' not in v)]
        if len(free) != 1:
            return (args, '')
        learned = name == self.roles.get('cart') or (name, free[0]) in self.id_args
        sample = next(iter(self.rows), '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if self.foreign_code(value):
                self.code_refusals += 1
                return (value, f'{value!r} is a code printed inside a listing, not a listing id (ids here look like {sample!r}); search for the code with the search tool, then open the id of the row it finds')
            if not learned or not isinstance(value, str) or _id_like(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"the text {value!r} appears on several listings ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is no listing id; pass an id from the result rows (the {self.id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_sets(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in _kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2:
                removal = removal or (value == self.cart_modes(name, key)[1] if self.modes else value in options[1:])
        loose_count = next((_count(v) for v in args.values() if _count(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((_count(v) for v in line.values() if _count(v) is not None and (not isinstance(v, str))), 1)
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

    def show(self, observation: Any, wide: bool=False) -> str:
        """The result as the loop model reads it. A single listing's page keeps long texts (compatibility lists and
        notes often sit past the first few hundred characters); result lists keep the short clip."""
        single = wide or (isinstance(observation, dict) and bool(self.ident(observation)))
        text = _dump(_clip(observation, PAGE_CHARS if single else SHOW_CHARS))
        if len(text) > OBS_LIMIT:
            text = text[:OBS_LIMIT] + '...(cut)'
        return text

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in _kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.compare_target() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.single_target(self.roles.get('open')))
        many = self.compare_target()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.compare_bare():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = _json_obj(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.rows) and (ident not in self.gone):
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': _dump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'See the side-by-side result above: Orrin opened these listings together in a single step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): every message to the shopper goes through Orrin's interviewer."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): no such tool here. Available: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): not enough steps remain for that; use one of: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_obj(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments must be a single JSON object.'))
                    continue
            faults: list[str] = []
            args = _shape(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = self.compare_target() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.compare_room()) and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Orrin] Only the first {room} listings went out: this store shows no more than {room} side by side. Still unseen: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.single_target(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not return. Give its cart line to another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = _dump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so nothing can be taken off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): read your other results first, then send the finishing call by itself.'))
                continue
            if not ending and self.left() <= self.tail() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the remaining steps are reserved for checking out (with the shopper's approval of the basket when that is due) or for ending without a purchase."))
                continue
            mark = name + _dump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): this exact call was just made and nothing has changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): this call already ran at step {cached[1]} and nothing has changed since; it returned:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.shown_already(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): this view has already shown all it has for {args[target]} since the last change. Its record: ' + self.record_of(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.shown_already(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): this view has already shown all it has for these listings since the last change. Their records:\n' + '\n'.join((f'{i}: {self.record_of(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Orrin] Skipped {', '.join((str(i) for i in args[many] if i not in keep))}: already shown in full, and nothing has changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2 and self.time_left() > CALL_RESERVE + 25:
                if self.is_buy(name, args):
                    wider = self.floor_sweep()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.cart_text()))
                        continue
                    reread = self.reread_offers()
                    covered = self.cover_views()
                    if self.note_sold:
                        self.find_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.left() > 3):
                        view_args = self.fit_args(self.roles['cart_view'], {})
                        view = self.act(self.roles['cart_view'], view_args) if view_args is not None else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.observe(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). The cart view carried a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply it, fix the cart where needed, then check out.'))
                                continue
                    swap = self.swap_hint()
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart has sold out: replace that line first.") + '\nCart now: ' + self.cart_text()))
                        continue
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.cart_text()))
                        continue
                    misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). The records say a part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Change the part (or the device), then check out.\nCart now: ' + self.cart_text()))
                        continue
                    faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
                    fixed = []
                    if faults and all((' sits at ' in f for f in faults)):
                        fixed = self.fix_counts(self.count_fixes())
                        faults = self.line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). Orrin corrected the cart's quantities itself: " + '; '.join(fixed) + '. Fix any line the store refused; then check out.\nCart now: ' + self.cart_text()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (with the cart tool's own way of adding or taking off units), then check out.\nCart now: " + self.cart_text()))
                        continue
                    gap = self.missing_items() if self.line_holds >= 1 or self.left() <= self.tail() + 2 else []
                    if gap:
                        settled = self.settle_gap(gap)
                        if settled:
                            out.append((cid, settled + (' [episode finished]' if self.finished else '\n' + self.status())))
                            continue
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3)
                    verdict = self.check_cart()
                    if verdict:
                        seen = "Orrin read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.cart_text()))
                        continue
                    if (reread or covered) and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Orrin just re-read the cart's listings:\n" + '\n'.join((x for x in (reread, covered) if x)) + '\nCheck out again if every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
            if self.is_refuse(name, args) and not self.finished:
                held = self.hold_decline()
                if held:
                    out.append((cid, held + (' [episode finished]' if self.finished else '')))
                    continue
            if self.is_refuse(name, args) and self.turned and (not self.finished) and self.left() >= 2 and (not self.review_silent()) and (not self.decline_proven()):
                done = self.order_best('the store turned an earlier finish down, and no item is shown to lack a fitting listing under the new terms')
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                again = self.fit_args(self.roles.get('buy') or '', dict(self.brief.get('buy_args') or {}))
                if self.cart_ids() and self.roles.get('buy') and again is not None:
                    self.trace.append({'kind': 'decline_to_checkout', 'turned': self.turned})
                    name, args = (self.roles['buy'], again)
                    trimmed += '\n[Orrin] The store turned a finish down and nothing shows that the new terms leave an item without a fitting listing, so Orrin checked out the cart as it stands.'
            if self.is_refuse(name, args):
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Orrin] ' + said + trimmed
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, 'The call did not go through (network or server trouble). You can try it again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = _dump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Orrin] The side-by-side view returned nothing for {len(args[many])} listings at once: send {self.compare_room()} or fewer per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.absorb(observation)
            if name == self.roles.get('compare'):
                rows = self.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((_fact_block(r) for _, r in rows))
            if name in readers - {self.roles.get('search')}:
                self.note_view(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain and (not ending) else ''
            if self.single_target(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(_fact_block(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                self.note_cart(name, args)
                self.cart_said = _dump(observation)
                self.recent = []
            if name == self.roles.get('search') and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            text = self.show(observation, name in self.detail_tools()) + paged + trimmed
            if name not in plain and (not ending):
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values())):
                    text += '\n[Orrin] The line is still in the cart after a count of 0: this tool adds the count to the line, so a negative count removes units.'
            self.note_sold = False
            self.note_kind = ''
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Bring your picks in line with this change before checking out.'
            if ending and self.record_finish(name, args, reply):
                text += '\n[Orrin] The store turned this finish down. Apply the change, pick again among the listings you know (the next fitting one if a pick broke the new terms), fix the cart and check out; a refusal alone never shows that nothing fits.'
            refused = self.is_buy(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spoken(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_buy(name, args))))
            if (loose or again) and self.cart_ids() and (not self.finished):
                seen = self.find_sold()
                if seen:
                    text += "\n[Orrin re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes that came with Orrin's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def hold_decline(self) -> str:
        """What happens instead when the loop model asks to end without a purchase, or '' to let the ending go.
        The evidence is reviewed (with a recovery search while steps allow), and the ending goes ahead when some
        requested item is shown to have no fitting listing, or when the review gives no verdict at all; when it names
        fits, or after the store has turned an ending down once, the best known basket is ordered. After a turned-down
        checkout the review is run afresh under the new terms and its verdict acts at once; otherwise the model first
        gets one chance to buy what the review found."""
        if self.turned or self.left() <= 3 or self.time_left() < CALL_RESERVE + 25:
            if not self.audited and self.left() >= 2 and self.time_left() > CALL_RESERVE + 40:
                self.audit()
            if self.decline_proven() or self.review_silent():
                self.trace.append({'kind': 'decline_proven' if self.reviewed else 'decline_unreviewed', 'fits': self.lead_items, 'proof': self.proof, 'turned': self.turned})
                return ''
            return self.order_best('the store turned an earlier finish down, and nothing shows that the terms now in force leave an item without a fitting listing' if self.turned else 'the steps or the clock ran short, which does not show that nothing fits')
        review = self.audit() if self.left() > 3 else ''
        if review:
            return 'Not run yet (no step used). ' + review
        if self.leads and self.refuse_blocks == 0 and self.left() > 4:
            self.refuse_blocks = 1
            self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
            return 'Not run yet (no step used). The evidence review found listings that seem to fit: ' + ', '.join(self.leads) + '. Open them if needed, cart the best fitting one (keeping to the choice rule) and check out. Only if each of them plainly breaks a requirement may you end without a purchase: name that requirement in your text and send the ending again.'
        if not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.time_left() > EXPLORE_RESERVE + 30:
            self.pushed = True
            return f"Not run yet (no step used). {self.left()} steps remain, so search in ways not tried yet before ending: list each requested item's category and page through it with the token or offset the store returns, try one-word value queries and other names for the property, and open the listings that might fit. End without a purchase only after that."
        if self.decline_proven() or self.review_silent():
            self.trace.append({'kind': 'decline_proven' if self.reviewed else 'decline_unreviewed', 'fits': self.lead_items, 'proof': self.proof})
            return ''
        return self.order_best('no requested item has been shown to lack a fitting listing' if not self.leads else 'the evidence review found fitting listings')

    def distinct_shelf(self, numbers: list[int]) -> bool:
        """Whether each of these items is its own kind of product, apart from every line in the cart, so that a cart
        without it is plainly short (and not one product the request split in two)."""
        carted = {_words(str((self.rows.get(i) or {}).get(self.category_key() or '') or '')) for i in self.cart_ids()} - {''}
        return all((_words(self.shelf_of(n)) and _words(self.shelf_of(n)) not in carted for n in numbers))

    def settle_gap(self, gap: list[int]) -> str:
        """A checkout that would leave requested items out of the order: fill them from the known listings, after an
        evidence review when none is on record. When the review shows that such an item has no fitting listing, end
        without a purchase, since an order missing a requested item cannot be right."""
        if self.gap_done:
            return ''
        self.gap_done = True
        if not self.audited and self.left() >= 2 and self.time_left() > CALL_RESERVE + 40:
            self.audit()
        if all((self.lead_items.get(str(n)) for n in gap)):
            return self.order_best('the cart left out a requested item that a known listing fits')
        if self.decline_proven(gap) and self.distinct_shelf(gap):
            return self.decline_now('a requested item has no fitting listing in this store, and an order without it cannot be right')
        return self.order_best('the cart left out a requested item, and an order without it cannot be right')

    def decline_now(self, why: str) -> str:
        refuse = self.roles.get('refuse')
        if not refuse or self.finished or self.left() < 1:
            return ''
        args = self.fit_args(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {})))
        if args is None:
            return ''
        said = ''
        if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
            go, said = self.gate('end')
            if not go:
                self.gap_done = False
                return 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()
        reply = self.act(refuse, args)
        self.trace.append({'kind': 'gap_decline', 'why': why, 'reply': _clip(reply, 200)})
        if reply is None:
            return ''
        self.note_kind = ''
        self.observe(refuse, reply.get('observation'))
        self.record_finish(refuse, args, reply)
        return (said + '\n' if said else '') + 'Not run (Orrin acted instead): ' + why + ', so Orrin ended without a purchase. Store reply: ' + self.show(reply.get('observation'))

    def cover_views(self) -> str:
        """Before checkout, open every cart listing with each single-listing view of the store it has not been read
        with yet (a view of seller notes or pointers often holds what the order is judged on)."""
        tools = sorted(self.detail_tools())
        todo = [(t, i) for i in self.cart_ids() if i not in self.gone and not self.sold_out(i) for t in tools if (t, i) not in self.viewed]
        if not todo or self.left() < len(todo) + self.tail() + 3 or self.time_left() < EXPLORE_RESERVE + 20:
            return ''
        lines = []
        for tool, ident in todo[:6]:
            target = self.single_target(tool)
            reply = self.visit(tool, {target: ident}) if target else None
            if reply is None:
                if self.finished:
                    break
                continue
            seen = reply.get('observation')
            self.viewed.add((tool, ident))
            self.note_view(tool, seen)
            if tool == self.offer_tool():
                self.offer_seen[ident] = self.step
            lines.append(f'{ident} ({tool}): ' + self.show(seen, True)[:1500])
        if lines:
            self.trace.append({'kind': 'cover_views', 'n': len(lines)})
        return '\n'.join(lines)

    def finish(self, again: bool=False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        gap = self.missing_items() if holding else list(range(1, len(self.items) + 1))
        if gap and buy and not self.audited and self.left() >= 2 and self.time_left() > CALL_RESERVE + 45 and (not self.no_credit):
            self.audit()
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
            gap = self.missing_items() if holding else gap
        proven = self.decline_proven(gap)
        if buy and gap and (not proven) and (not self.no_credit) and self.left() >= 2:
            done = self.order_best('the work loop ended with ' + ('items missing from the cart' if holding else 'an empty cart') + ', which does not show that nothing fits')
            if self.finished or self.left() <= 0:
                return ''
            if done and again:
                return 'the checkout was refused: ' + done
            fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
            holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
            gap = self.missing_items() if holding else gap
        short = holding and bool(gap) and proven and self.distinct_shelf(gap)
        if buy and (not short) and (holding or (self.cart_touched and (not fresh) and (self.left() >= 2))):
            args = self.fit_args(buy, dict(self.brief.get('buy_args') or {}))
            if args is not None:
                if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                    self.gate('order')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': _clip(reply, 200)})
                if self.finished or self.left() <= 0:
                    return ''
                if reply is not None:
                    self.note_kind = ''
                    self.observe(buy, reply.get('observation'))
                    self.record_finish(buy, args, reply)
                    if again:
                        return 'the checkout was refused: ' + self.show(reply.get('observation'))
                    proven = self.decline_proven(gap)
        if self.turned and buy and not proven and not self.finished and self.left() >= 1:
            done = self.order_best('the store turned a finish down, and nothing shows that the new terms leave an item without a fitting listing') if self.left() >= 2 else ''
            if not done and (self.cart_ids() if fresh else self.cart_lines) and self.left() >= 1:
                args = self.fit_args(buy, dict(self.brief.get('buy_args') or {}))
                reply = self.act(buy, args) if args is not None else None
                self.trace.append({'kind': 'forced_buy_again', 'reply': _clip(reply, 200)})
                if reply is not None:
                    self.record_finish(buy, args, reply)
            if self.finished or self.left() <= 0 or done:
                return ''
        if refuse:
            args = self.fit_args(refuse, self.quiet(refuse, dict(self.brief.get('refuse_args') or {})))
            if args is not None:
                if self.left() >= 2 and self.time_left() > CALL_RESERVE + 15:
                    self.gate('end')
                if self.finished or self.left() <= 0:
                    return ''
                reply = self.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': _clip(reply, 200)})
                if reply is not None and (not self.finished):
                    self.note_kind = ''
                    self.observe(refuse, reply.get('observation'))
                    self.record_finish(refuse, args, reply)
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        size = sum((len(_dump(m.get('content'))) for m in self.chat))
        if size <= HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(earlier result trimmed; its listings: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.absorb(_unfold(opening))
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nTHE STORE OPENS WITH:\n' + self.show(opening)
        intro += '\n\nBegin by searching each item in its category and opening several of its listings in full, so the interviewer learns which properties those listings carry.\n\n' + self.status()
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.left() > 1) and (self.time_left() > EXPLORE_RESERVE):
                self.chat.append({'role': 'user', 'content': '[Orrin] Orrin tried to finish the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_calls + 10, SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls, 'turned': self.turned})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished and (not self.broke) and (self.left() > 0) and (self.loop_calls < loop_cap) and (self.cost < spend_cap) and (self.time_left() > EXPLORE_RESERVE):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [LIGHT, SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= ALL_CALLS or self.misses > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else _json_obj(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': _dump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = _dump(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Continue by calling one of the tools.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.msgs_free() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.left() > 8):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper gets asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer found nothing worth asking.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Orrin(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.finish()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
