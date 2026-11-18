from __future__ import annotations
import json
import random
import re
import threading
from typing import Any
from src.agent.proxy_client import ProxyClient
NIMBLE = 'anthropic/claude-haiku-5.5'
WEIGHTY = 'anthropic/claude-sonnet-5.5'
RESERVE = 'openai/gpt-6-luna'
CYCLE_DISPATCHES = 42
TOTAL_DISPATCHES = 60
OUTLAY_MILD = 0.055
OUTLAY_FIRM = 0.078
WEIGHTY_HEADROOM = 0.02
LINGERINGS = (4, 10, 20, 35)
LINGER_SUM = 45
EXHIBIT_GLYPHS = 450
SIGHTING_BOUND = 15000
ANNALS_BOUND = 150000
BARRIER_LAPS = 3
SELF_FOLIOS = 6
THRIFT_UNFOLDS = 3
SPAN_WINDUP = 180
SPAN_HALT = 220
SPAN_FINAL = 245
_model = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_habitat = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
HANDBOOK = '''\
Kyo is your name: you shop on behalf of a single shopper in an online store, and calls to the store's tools are the only way you can act. Scoring is all or nothing (1 or 0), and the 1 goes only to a finish that is exactly correct. That means one of two outcomes. Either the order holds each item asked for, in the quantity asked for, every one of them satisfying all the requirements that apply at checkout time, with the whole order within budget; or, if truly no listing in the store satisfies everything, the episode closes through the no-purchase tool, used the way the request says.

THE STORE AND ITS TOOLS
- Steps are counted: each tool call uses one and the supply is limited, while reasoning is free. Every store names its tools and arguments and shapes its results its own way, so study each tool's description and schema, and pass argument names and allowed values exactly as written there.
- Searching may be restricted: a filter by category, a keyword mode that perhaps reads only the opening words of your query, pages of only a few rows, a token or offset that fetches the next page. When the answer hinges on seeing every match, check the total the store reports against the rows seen so far and keep paging, passing the token or offset exactly as the store hands it to you. Short keyword queries that lead with the value do best (the value you want, then the product noun). If a filtered search comes back empty, the cause is usually a wrong filter value or the wrong filter parameter: try again with the label spelled exactly as it appears in result rows, or switch to a different filter parameter.
- A search row is only a summary. How a listing's record is split up varies by store: one view may carry its attributes (properties), another only the offer and description, while a side-by-side view could hold either one. Once Kyo has learnt which tool shows attributes, the ledger names it. Since every property requirement is judged on those attributes, read them for any candidate you might buy; when a single call covering several listings shows what you need, it saves steps. If nothing has changed since you read a listing, do not open it a second time.
- Kyo's interviewer deals with all questions to the shopper; you never send the shopper a message yourself. Its answers show up in the conversation and get added to the ledger.
- Besides the purchase, the request may require steps on the way, such as an approval, a message, or something to check or read beforehand; the ledger lists these. Kyo itself performs the ones due right before the order or the ending, at the moment you call the checkout or ending tool, so do not wait for them, and never let them be the reason you end without buying. Any other such step is yours to do with store tools, except that product questions are routed through Kyo's interviewer.

DECIDING WHETHER A LISTING FITS
- The listing's category label settles what kind of product it is. Only a label matching the requested item as a whole is valid; a neighbouring label never is.
- A property requirement is satisfied when the listing's attribute for that property gives the value. How the key is spelled is irrelevant (spaces, underscores, camel case, prefixes), and it is enough for one element of a comma or slash list to match. Values are compared by their words: differences of case, singular versus plural, and plain synonyms are all acceptable, and so is a value containing the words of the required value ("extra roomy fit" satisfies roomy, "deep teal" satisfies teal). Different words do not match: "teal green" is not "sea green", and a code or abbreviation does not equal the word it stands for. Convert units before comparing numbers; a limit includes its edge value, and a range has to fall entirely within a limit.
- If an attribute's key names the requirement's property (by that name or a plain synonym), that attribute is decisive: should it state something else, the listing fails, no matter what other keys or the title say. A key about a different property, however closely related (the colour of a handle is not the colour of the body), settles nothing for the requirement. After each opened listing Kyo appends [evidence] lines listing the keys whose names match a requirement.
- If the listing has no attribute for the property, the wanted value counts when it is stated under some other attribute (a list of features or styles that includes it, say) or in the title or description. A number of a different kind does not count (a weight is not a capacity), and neither does a bare figure whose text fails to say what it measures.
- A requirement whose value appears in the category label itself (a season or audience in the label, for instance) is met by the category.
- Instead of giving a value, a requirement may point at another listing (phrasings like "the X matching that of the listing <shop> sells with <property>" or "no more than N away from the X shown on the page of ..."). Resolve such a requirement first: locate that listing (search by its category, brand, shop or words), open its whole page using the side-by-side or single-item tool (search rows are not enough), and if that page in turn sends you on to yet another listing for the property, locate and open that one likewise; the value comes from the final page. The order only counts once every page in that chain has been opened. If the wanted value is anything other than that listing's value, the rule excludes just the listings with exactly that value, and that rule alone is never a reason to end without buying.
- If the page of a listing you plan to buy points you to a further page about it, read that page before buying.
- If one requested item has to fit another (an accessory for a device), decide the pair together from the listings' own wording (a stated list of what it fits, or a model or series code that appears on both pages). When none of the accessories meeting their own requirements fits the device you chose, switch to a different device, one which satisfies its own requirements and which an accessory of that kind fits; do not end without buying while that switch remains untried.
- Wishes ("ideally", "would be nice") only break ties; exclusions and closed lists must be respected.
- A listing the store shows as sold out is never bought. The total, meaning price times quantity added up over all lines, must not go over the budget; landing exactly on it is fine.

PICKING AMONG FITS
- If the request wants the cheapest (lowest total, smallest price), the order has to contain the lowest-priced listing out of ALL fitting listings. Gather the complete fitting set by paging through it (result rows often come in no price order, and Kyo reads the following pages of a short result on its own), check candidates from the cheapest up, and take the first that fits completely. If a search accepts a maximum price, use it: list the category under your current best pick's price, so only cheaper candidates remain to be checked.
- If the request wants the most or the least of some measure, compare all fitting listings on that measure and choose the extreme; no listing can win on a measure it does not state.
- In any other case every listing meeting everything is correct: spend no steps looking for a better one.
- As soon as you have opened a listing meeting every requirement of an item, cart it (for a cheapest, most or least pick, only after comparing the fitting set) and go on to the next. Carted lines often sell out soon after being added, and carting early leaves steps for a replacement.
- When the request covers several different items, each one gets its own cart line at its requested quantity.

WHEN THINGS CHANGE
- The shopper or the store may change things along the way: a new budget, a requirement added or removed, a different price, a listing that sold out. Such changes come inside tool results, frequently as a refusal when you try to check out; a note counts no matter which tool's result carries it, and the ledger lists the requirements currently in force. A refusal does not end the episode: apply all the changes, repair the cart and check out once more.
- Price and stock can change without notice, so re-read the record of a chosen listing shortly before you check out.

ENDING
- The cart must hold exactly the chosen lines (remove everything else); view the cart once, then check out in a call by itself.
- Ending without a purchase is allowed only after a genuine search has turned up no listing meeting everything. Being short of steps proves no such thing; in that case check out the best cart you have. For that ending, follow the wording of the request, and if a remark is needed, keep it brief and name no listing.
- Each reply has to call at least one tool. Keep the text short: what you know, then your next move.'''
DIGEST = '''\
Your job is to turn a shopper's request into a task an agent can work from. Using the request and the list of store tools, reply with a single JSON object and nothing more, written compactly without indentation, holding the keys below.

"items": one entry for every distinct product the shopper wants to buy, each of the form {"name": the product's kind, in the words the request uses, "shelf": the category label the store uses for it, copied exactly, when some tool lists labels, otherwise "", "quantity": how many units of that single listing (an integer; 1 if the request does not say; a pair or a set sold as one listing is 1), "musts": the firm requirements on that product as a list, each written "property: constraint" carrying the exact value, limit, exclusion or closed list}.
How to fill "musts":
- Call the property by the word the request itself uses ("a 12-inch blade" becomes "blade length: 12 inch", "meant for a lakeside occasion" becomes "occasion: lakeside"); never substitute a neighbouring property.
- When a must takes its value from another listing, it is ONE must containing the full description of that listing ("strap colour: unlike the strap colour of the backpack Telvane sells with a padded back"). Words describing that other listing (its shop, its kind, a trailing "with <property> <value> and <property> <value>") stay inside that single must and never become musts of this product.
- Price, budget, stock and quantity never go into musts.
- Background in the request (the intended user, where the product is to be placed, earlier problems, the event prompting the purchase) counts as context rather than a must, except where it plainly gives a property's value or limit. Words forming part of the product's name (for instance a season or audience given with the product) are never musts either; they belong in "name".
- If the request turns a use or occasion into a property of the product itself, that IS a must on the property ("the occasion has to be party", "a dress made for parties", "blazers for business wear" give such musts, e.g. "occasion: business"), unless the category label of the store already names it. Any event, person or setting the purchase is meant for ("for a colleague", "to wear at his wedding") stays context, as does the way it should look, feel or sit when used.
- Products named in one phrase joined by "and" (a single "I want ..." with a single quantity, often the name of one store department) form ONE item; split them into separate items only when each product has a separate quantity or separate requirements.
"budget": the shopper's maximum total spend, as a number in the unit the request uses, or null.
"budget_unit": that unit spelled out in words (e.g. "cents"), or null.
"budget_scope": "total" if the limit applies to the whole order, "each" if it applies per item, null if there is no limit.
"in_stock_only": true, except when the request plainly accepts unavailable items.
"pick": "cheapest" if the request asks for the smallest total or the lowest price; "most" or "least" if it asks for the extreme of some measure; otherwise "any". Comments that money is tight do not make a pick rule.
"pick_measure": for "most"/"least", the thing being maximised or minimised; otherwise "".
"messages": the number of messages the shopper lets the agent send (integer), or null if the request does not say.
"steps": the number of steps, moves or tool calls the request permits (integer), or null.
"lookups": the musts (copied exactly) whose value has to be read first from another listing the request points at; [] if none.
"rules": all instructions in the request on the way to work or the way to end, each in 12 words or fewer.
"wishes": things the shopper would merely like.
"ask_first": matters the request expressly wants confirmed with the shopper before anything is chosen ([] when there are none).
"process": the steps the request demands other than finding and buying the right items, e.g. obtaining approval from the shopper, showing or telling them something, confirming something with them, first reading something, using some specific tool. Each is {"step": the action required, in 15 words or fewer, "when": "choose" (before the products are chosen), "order" (just before checkout) or "end" (right before ending with no purchase), "messages": the number of messages to the shopper it needs (0 if none)}; [] if the request asks for no such step. These are not steps: item requirements, budget, stock, message and step limits, or the way to end if nothing fits.
"quiet_decline": true when the request forbids suggesting or naming alternatives if nothing fits, false when it requests alternatives, null when it does not address this.
"fits": one {"item": item number, "with": item number} entry per requested item that, per the request, has to suit, fit or work with ANOTHER requested item (whichever one the agent picks); [] if none.
"roles": an object giving the exact tool for each job, with null for any job no tool does. The jobs: "search"; "open", the tool showing a single listing's attributes or properties (with two single-listing tools, choose the one whose description says it returns attributes, properties or specifications); "offer", a tool separate from "open" that gives one listing's present offer, e.g. its price and stock; "compare", showing several listings in full; "ask", messaging the shopper; "cart", modifying the cart; "cart_view", displaying the cart; "buy", the checkout; "refuse", ending with no purchase.
"buy_args": arguments with fixed values that switch the "buy" tool to checking out (when the tool takes a mode argument, the value of that mode); {} if there are none.
"refuse_args": likewise, the fixed-value arguments that switch the "refuse" tool to ending without a purchase; {} if there are none.
'''
BLUEPRINT = '''\
Your task: pick which questions a shopping agent will put to a shopper. Each shopper has a handful of firm preferences left out of the request, every one concerning a single property of a single requested product. Only a question naming that very property brings such a preference out. Asking about a property the shopper holds no opinion on returns "don't mind", and asking about one the shopper never considers returns a confused answer, so each property you name uses up part of a message.

Your input: the request; the items, each with what the request already fixes for it; and a numbered list of the attribute keys found on opened listings in each item's category, giving how many of those listings have the key plus some example values. Bear in mind that an unwritten preference can concern a property none of the listed keys names.

Rank the properties as follows.
0. First, whatever the request itself tells you to check with the shopper (shown under ASK FIRST if present), each called by the property it concerns.
1. Next, hinted properties. The story in a request often alludes to an unwritten preference without stating its value: a sentence about something the shopper is unable to judge or hands over to you, a person whose taste is decisive, how the product looks or the setting it goes in, the person who will use or wear it, or what it contains. Further allusions of this kind: a past purchase that looked the part yet turned out to be something else underneath, or curiosity about how the thing was put together or came to be, points at what it really is or is made from, named by that item's own key for it (its construction or material for goods, its source or main ingredient for food), never at its looks; a figure the shopper copies from the label or logs every day points at the per-serving amount the label states; a recipient whose fit or range is guesswork points at the gender or age group; a place of use that keeps changing points at the occasion. For every such sentence give the single property it alludes to and the item concerned, following the story's order.
Call a hinted property by one of that item's listed keys if one matches the hint (a hint about the wearer: the gender or audience key; a hint about looks: whichever listed colour, pattern or finish key is carried by most of its listings). A property that no listing of the item has goes after that item's three most widely carried keys.
2. After those, the listed keys in ranked order. Prefer properties that buyers of this kind of product name when they choose it (frequently its type or style when the request leaves it open, colour, material, the wearer's gender or age group, construction, closure, pattern, flavour or main ingredient, network or connector type, depending on what that kind of product is chosen by) and keys that most of the item's listings carry. When a listed type, style or kind key shows values that name different kinds of product (a wardrobe next to a chest of drawers, a sideboard next to a TV stand), the category is broad and the kind itself is open: unless the request fixes the kind, that key is the item's first pick. A plain measurement (width, height, depth, length, size, weight, capacity, an amount per serving) is a real candidate too, roughly one hidden preference in four sits on one: rank it by the same tests, but not ahead of the item's best named property unless the story talks about space, fit or amounts. When five or more listings have been read, steer clear of keys that showed up on just one or two.
Leave out: a property the request already fixes for that item (some requirement names it; but a key whose values merely echo a required value, e.g. a pattern key displaying the required decoration, is NOT fixed); a second name for a property already picked; a key each of whose values bundles several purchase options together; also identifiers, model names, warranty and similar logistics. Treat pack sizes, counts and per-pack quantities as ordinary properties. A key tagged [not the stated ...] is a property distinct from that requirement, however many words their names have in common. Rank each item separately: its hinted properties come first, its listed keys after.
"label" holds the plain name of the property. When it comes from a listed key, use all of that key's words in ordinary spelling (lidColour becomes "lid colour", rim_finish becomes "rim finish"; never shorten them to "colour" or "finish"). Only the name of the property goes there: leave out example values, brackets and any second property.
Pick at most COUNT per item, strongest first. Reply with one JSON object only: {"ask": [{"item": item number, "label": "...", "hint": "up to 8 words of the story it is drawn from, or an empty string"}]}, keeping each item's picks in ranked order'''
LISTEN = '''\
Below are questions a shopping agent put to a shopper, together with the replies. Go through every reply, then answer with a single JSON object and no other text:
{"needs": [{"need": "property: constraint", "item": the item's number, "points": true if the value has to be taken from another listing, otherwise false}], "drop": [{"property": "a requirement the shopper has now given up", "item": the item's number}], "free": [{"property": "...", "item": the item's number}], "crowded": [numbers of the questions whose reply objects that the message covered too many things or asks for one question at a time; leave out replies that merely say they did not understand it], "budget": a new total limit as a number, or null}
How to fill it:
- "needs" holds each firm preference stated in a reply, with the precise value or limit plus its unit ("oval", "not wicker", "at least 30 cm"). Each need is named after the property the reply itself names, even when that differs from the property asked about; several replies about the same property give one need. Copy the value verbatim as the reply puts it, nouns and units included ("at least 3 shelves", not "at least 3").
- A reply may give the value by referring to another listing (a value that resembles, differs from or is close to the one on a product some shop sells, the product being picked out by a few of its properties). Record that as ONE need on the property the reply names, with "points": true. Its value is the reply's own wording copied exactly, from the comparison words through the end of the description, along with whatever it says about a listing that the page might name; do not shorten or paraphrase it. None of the properties describing the other product is a need of this item.
- "free": for every question whose reply says the shopper does not mind, the property it asked about.
- Homely or roundabout phrasing that names a value still states that value.'''
JOTTING = '''\
The note below appeared in the store while a shopping agent was working; its author may be the shopper or the store. Reply with one JSON object and nothing beside it:
{"kind": "change" if the note alters the task (the budget, the requirements, a price, availability, or the order being refused), "error" if all it says is that a call was malformed or rejected, "other" in every remaining case, "budget": the new overall limit, a number in the unit the request uses, or null, "add": [{"item": the item's number, "need": "property: constraint"}], "drop": [{"item": the item's number, "need": "the requirement that is no longer wanted"}], "gone": [ids of listings the note calls unavailable], "sold_out": true if the note reports that a chosen listing has gone, sold out or stopped being available (even if it does not say which), "price_moved": true if the note says some price changed, "summary": "a single short sentence"}
The "summary" states what the note changes, as a fact the agent must apply; it makes no remark about who wrote the note or whether it can be trusted. List every change the note contains. When a requirement is replaced, record the old one under "drop" and the new one under "add".'''
PROBE = '''\
You are the final gate before a shopping agent checks out: judge from the records shown whether the cart is exactly right.
How to read the records: a requirement holds when the listing's attribute for that property gives the value. Key spelling does not matter, a single element of a list suffices, and case, singular or plural and plain synonyms are accepted, as is a value that contains the words of the required value (such as "deep teal" for teal); convert units, treat limits as including their edge, and require a range to fall entirely inside. Where an attribute's key names the requirement's own property, that attribute decides, and the listing fails if it states something different; a key about a different property, however closely related (the colour of a handle is not the colour of the body), settles nothing. Lacking an attribute for that property, a listing can still show the wanted value in the title, the description or under some other attribute, and that counts (but a number of another kind does not, since a weight is not a capacity, and neither does a bare figure whose text fails to say what it measures). The category label has to fit the requested item taken as a whole, and when that label itself states a requirement's value, the category meets that requirement. A rule that the value must be unlike another listing's value is met unless the records show that very value.
The cart must contain each requested item as its own line at the requested quantity and nothing else; no line may be shown out of stock; and price times quantity summed over all lines must stay within the budget. Where a listing has several records, the newest one counts.
If the pick rule is cheapest, most or least, look at the other listings shown as well: the cart is wrong if one of them plainly meets every requirement and does better than the cart's choice by that rule.
Return just one JSON object, with no other text: {"ok": true or false, "problems": ["a short, concrete problem", ...], "better": [ids of listings that should replace a cart line, best first]}. Only use ok=false when the records plainly show a problem, or when no record addresses some requirement.'''
PERUSAL = '''\
if an attribute's key names the requirement's property, that attribute settles it (spell the key any way, a single list element suffices, synonyms and converted units are accepted, and so is a value that contains the required value's words, such as "deep teal" for teal; limits include their edge); a key about some other property, however closely related (the colour of a handle is not the colour of the body), settles nothing; where no attribute names the property, the wanted value counts if it appears in the title, the description or under another attribute (a bare figure counts for nothing when its text does not say what it measures); a category label that itself states a requirement's value meets that requirement; the category label has to match the item; a rule that the value be unlike another listing's is met unless the records show that very value; listings that are out of stock or that break the budget do not fit.'''
INSPECTION = '''\
Believing that no listing fits, a shopping agent is about to end without a purchase. Test the listing records shown against the requirements, reading them by these rules: ''' + PERUSAL + '''
A record marked ROW is only a search result row; its page, with the attributes, has not been read, so on any property the row does not show it can neither pass nor fail.
Give back a single JSON object and nothing else: {"fits": {"item number": [ids of listings that seem to meet every requirement, cheapest first, no more than 4]}, "open": [the ids of ROW records for a requested kind of product and within the budget whose page might show that they fit, most promising first, no more than 6], "reason": "a single sentence"}'''
UNTANGLE = '''\
Firm requirements on a product were extracted from a shopper's request. A few of them get their value from a different listing the request refers to, and the words identifying that listing (its kind, its shop, the properties listed after it with "with ...") sometimes get mistaken for requirements on the product being purchased. Going by how the request is worded, tell which of the OTHER REQUIREMENTS below do nothing but describe that other listing. Output a single JSON object and nothing else: {"describe_other": [each such requirement, copied exactly; an empty list if all of them really are requirements on the product]}'''
WAKE = '''\
A shopping agent is on the point of checking out. Certain requirements, rather than stating a value, take it from another listing, and the page of a listing may in turn send the reader on to yet another listing for that property. Only if the agent has opened, in full, every listing on each such chain does the order count. Starting from each requirement's own description, trace its chain one link at a time through the wording of the pages, working from the records below (OPENED: a page the agent opened; ROW: known only from a search result row, its page unread).
Return a lone JSON object with nothing around it: {"chains": [{"need": the exact text of the requirement, "links": [{"id": the id of the listing if one of the records below is this link, otherwise "", "about": the way the requirement or the preceding page describes this listing, "find": "for a link not yet found: words to search for (its shop plus the values the description gives)"}], "done": true if the final link is an OPENED page that itself states the property's value (this holds too when the requirement wants a value different from or unlike that one, since the page's value is the yardstick), otherwise false, "value": the property's value as that final page states it, or ""}]}
A record counts as a link only if it fits the link's entire description: shop or brand, kind of product, AND each property value the description mentions (so "a 10 pieces multipack" excludes a listing from the same shop with 10 capsules that is not a multipack). If no record fits all of it, the link has not been found yet, even when that shop sells other listings of the same kind; for such a link give "find": a few search words made of the shop's name and the values in the description.
List the links of every chain in order, stopping at the earliest link that is either not found or whose page is not OPENED, since its words are not known yet.'''
BARRIER = '''\
A shopping agent is ready to STAGE. Along the way, the shopper's request requires the steps listed below. Using the numbered RECORD of the agent's actions so far, decide for every step that falls due before the agent may STAGE whether it has been done for the cart in its current state. A step is done only if some RECORD line shows it happening; for a step that concerns the cart or the order, that line must also come after the most recent cart change, and the reply in it must grant the step. Questions about a product's properties never accomplish a step.
If a step is still outstanding, give the single store call that would do it next: a tool from TOOLS plus its arguments. Messages to the shopper go through the messaging tool, as one short sentence or question about that step only, in plain language, mentioning no product properties or values that the step does not require. Give no call when all due steps are done, or when the remaining messages or steps cannot accomplish the missing one.
Give one JSON object, nothing more: {"steps": [{"step": the step itself, "done": true or false, "lines": [numbers of the RECORD lines showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": whatever a reply asks to change before continuing, or "", "needs": [{"item": the item's number, "need": "property: constraint"} for every firm requirement such a reply states]}'''
EXCHANGE = '''\
One of the listings in a shopping agent's cart sold out a moment ago. Look among the other listing records shown for listings that could replace that cart line, reading them by these rules: ''' + PERUSAL + '''
Reply with just one JSON object and no other text: {"fits": [{"id": the listing's id, "price": the listing's price as a number, as its record gives it} for each listing that seems to satisfy every requirement of the item on the sold-out line, no more than 6], "reason": "a single sentence"}'''
SILHOUETTE = '''\
Below is a result returned by a store tool. Identify its parts using exactly the keys that appear in the result and in the tool's parameters. Output a single JSON object and nothing else: {"rows": key of the list that holds the listings, "count": key giving the total number of matching listings (the full match count, not the number displayed), or null, "next": key whose value requests the next page, or null, "next_arg": the tool's parameter that accepts that value, or null, "price": key within a row for the listing's price, or null, "stock": key within a row telling whether the listing is buyable right now, or null}'''
SENTRY_CLEFS = 3
SENTRY_PART = 0.0
SENTRY_SPACING = 0.34
NON_TRAIT = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
SNARE_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
BROADEN_EVERY = 6
CONFORM_TOKEN = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PIECE_TOKEN = '[A-Z]{1,4}\\d{3,6}'

class _Text:

    @staticmethod
    def _json_object(wording: Any) -> dict[str, Any] | None:
        if not isinstance(wording, str):
            return None
        wording = wording.strip()
        barrier = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', wording, re.S)
        if barrier:
            wording = barrier.group(1)
        try:
            gained = json.loads(wording)
            return None if not isinstance(gained, dict) else gained
        except ValueError:
            pass
        (leftover, correct) = (wording.find('{'), wording.rfind('}'))
        if 0 <= leftover < correct:
            try:
                gained = json.loads(wording[leftover:correct + 1])
                return None if not isinstance(gained, dict) else gained
            except ValueError:
                return None
        return None

    @staticmethod
    def _serialize(worth: Any) -> str:
        try:
            return json.dumps(worth, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(worth)

    @staticmethod
    def _shear(worth: Any, boundary: int, deepness: int=0) -> Any:
        if 7 < deepness:
            return '...'
        if isinstance(worth, str):
            return worth[:boundary] + '...' if not len(worth) <= boundary else worth
        if isinstance(worth, dict):
            return {kk: _Text._shear(vv, boundary, deepness + 1) for (kk, vv) in worth.items()}
        if isinstance(worth, list):
            return [_Text._shear(vv, boundary, deepness + 1) for vv in worth]
        return worth

    @staticmethod
    def _radicals(wording: str) -> set[str]:
        away = set()
        for ww in _Text._vocables(wording).split():
            ww = (ww if not (len(ww) > 3 and ww.endswith('s') and (not ww.endswith('ss'))) else ww[:-1]) if not re.search('(?:ch|sh|ss|x|z)es$', ww) else ww[:-2]
            if len(ww) >= 3 and ww not in ('and', 'for', 'the', 'with'):
                away.add(ww)
        return away

    @staticmethod
    def _trunk(wording: str) -> frozenset[str]:
        return frozenset((ww if not (len(ww) > 3 and ww.endswith('s')) else ww[:-1] for ww in _Text._vocables(wording).split()))

    @staticmethod
    def _vocables(clef: str) -> str:
        clef = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(clef))
        return re.sub('[\\s_.\\-]+', ' ', clef).strip().lower()

class _Kinds:

    @staticmethod
    def _quantity_n(worth: Any) -> int | None:
        if isinstance(worth, bool):
            return None
        if isinstance(worth, int):
            return worth
        if isinstance(worth, float) and worth.is_integer():
            return int(worth)
        if isinstance(worth, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', worth):
            return int(float(worth))
        return None

    @staticmethod
    def _entries_of(worth: Any) -> list[Any]:
        if isinstance(worth, list):
            return worth
        if isinstance(worth, (str, dict)) and worth:
            return [worth]
        return []

    @staticmethod
    def _silhouette(worth: Any, outline: Any, whereat: str, flaws: list[str]) -> Any:
        if not isinstance(outline, dict):
            return worth
        genera = _Kinds._genera(outline)
        if genera and (not any((_Kinds._conforms_genus(worth, kk) for kk in genera))):
            for genus in genera:
                if 'integer' == genus and isinstance(worth, float) and worth.is_integer():
                    worth = int(worth)
                    break
                if genus in ('integer', 'number') and isinstance(worth, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', worth):
                    digit = float(worth)
                    worth = digit if not ('integer' == genus or digit.is_integer()) else int(digit)
                    break
                if 'string' == genus and isinstance(worth, (int, float)) and (not isinstance(worth, bool)):
                    worth = str(worth)
                    break
                if 'boolean' == genus and isinstance(worth, str) and (worth.strip().lower() in ('true', 'false')):
                    worth = worth.strip().lower() == 'true'
                    break
                if 'array' == genus and (not isinstance(worth, list)) and (worth is not None):
                    worth = [worth]
                    break
            if not any((_Kinds._conforms_genus(worth, kk) for kk in genera)):
                flaws.append(f"{whereat} should be {'/'.join(genera)}")
                return worth
        alternatives = outline.get('enum')
        if isinstance(alternatives, list) and worth not in alternatives:
            if isinstance(worth, str):
                scant = worth.strip().lower()
                match = [oo for oo in alternatives if isinstance(oo, str) and oo.lower() == scant]
                if match:
                    worth = match[0]
            if worth not in alternatives:
                flaws.append(f'{whereat}: allowed values are {alternatives}')
        if isinstance(worth, str) and isinstance(outline.get('maxLength'), int) and (len(worth) > outline['maxLength']):
            worth = worth[:outline['maxLength']]
        if isinstance(worth, str) and isinstance(outline.get('minLength'), int) and (len(worth) < outline['minLength']):
            flaws.append(f'{whereat} is too short')
        if isinstance(worth, (int, float)) and (not isinstance(worth, bool)):
            if isinstance(outline.get('minimum'), (int, float)) and worth < outline['minimum']:
                flaws.append(f"{whereat} is below {outline['minimum']}")
            if isinstance(outline.get('maximum'), (int, float)) and worth > outline['maximum']:
                flaws.append(f"{whereat} is above {outline['maximum']}")
        if isinstance(worth, list):
            if isinstance(outline.get('maxItems'), int) and len(worth) > outline['maxItems']:
                flaws.append(f"{whereat} allows at most {outline['maxItems']} entries")
            if isinstance(outline.get('minItems'), int) and len(worth) < outline['minItems']:
                flaws.append(f"{whereat} needs at least {outline['minItems']} entries")
            if isinstance(outline.get('items'), dict):
                worth = [_Kinds._silhouette(vv, outline['items'], f'{whereat}[{ii}]', flaws) for (ii, vv) in enumerate(worth)]
        if isinstance(worth, dict):
            characteristics = {} if not isinstance(outline.get('properties'), dict) else outline.get('properties')
            if outline.get('additionalProperties') is False:
                for moniker in [nn for nn in worth if nn not in characteristics]:
                    flaws.append(f'{whereat}.{moniker} is not an allowed argument')
            for moniker in outline.get('required') or []:
                if moniker not in worth:
                    flaws.append(f'{whereat}.{moniker} is required')
            worth = {kk: vv if not kk in characteristics else _Kinds._silhouette(vv, characteristics.get(kk), f'{whereat}.{kk}', flaws) for (kk, vv) in worth.items()}
        return worth

    @staticmethod
    def _conforms_genus(worth: Any, genus: str) -> bool:
        if 'string' == genus:
            return isinstance(worth, str)
        if 'integer' == genus:
            return isinstance(worth, int) and (not isinstance(worth, bool))
        if 'number' == genus:
            return isinstance(worth, (int, float)) and (not isinstance(worth, bool))
        if 'boolean' == genus:
            return isinstance(worth, bool)
        if 'array' == genus:
            return isinstance(worth, list)
        if 'object' == genus:
            return isinstance(worth, dict)
        if 'null' == genus:
            return worth is None
        return True

    @staticmethod
    def _genera(outline: dict[str, Any]) -> list[str]:
        genus = outline.get('type')
        if isinstance(genus, str):
            return [genus]
        return [kk for kk in genus or [] if isinstance(kk, str)]

    @staticmethod
    def _monikers(aspect: frozenset[str], clef: frozenset[str]) -> bool:
        if not aspect or not clef:
            return False
        return clef == aspect or clef >= aspect or (len(clef) >= 2 and aspect >= clef)

    @staticmethod
    def _conform_tag_marks(wording: Any) -> dict[str, Any]:
        wording = str(wording or '')
        native = re.search('[\\[(]\\s*(?:part\\s+)?(' + PIECE_TOKEN + ')\\s*[\\])]', wording, re.I)
        token_codes: list[str] = []
        for catch in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + CONFORM_TOKEN + ')\\b', wording, re.I):
            token_codes += _Kinds._token_codes(catch.group(1))
        for catch in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', wording, re.I):
            token_codes += _Kinds._token_codes(catch.group(1))
        labours: list[str] = []
        for catch in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', wording, re.I):
            if not re.search('\\bpart\\s+' + PIECE_TOKEN + '\\s+also\\s+$', wording[max(0, catch.start() - 40):catch.start()], re.I):
                labours += _Kinds._token_codes(catch.group(1))
        as_well = [(catch.group(1).upper(), _Kinds._token_codes(catch.group(2))) for catch in re.finditer('\\bpart\\s+(' + PIECE_TOKEN + ')\\s+also fits\\s*:?\\s*([^.;]+)', wording, re.I)]
        extra = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PIECE_TOKEN + ')', wording, re.I)
        return {'own': '' if not native else native.group(1).upper(), 'codes': token_codes, 'works': labours, 'also': as_well, 'more': '' if not extra else extra.group(1).upper()}

    @staticmethod
    def _token_codes(wording: Any) -> list[str]:
        return [re.sub('\\s+', ' ', cc).strip().lower() for cc in re.findall('\\b' + CONFORM_TOKEN + '\\b', str(wording or ''))]

class _Rows:

    @staticmethod
    def _datum_slab(knot: Any) -> dict[str, Any] | None:
        if not isinstance(knot, dict):
            return None
        finest = None
        for worth in knot.values():
            if isinstance(worth, dict) and len(worth) >= 2:
                primitives = {kk: vv for (kk, vv) in worth.items() if isinstance(vv, (str, int, float)) and (not isinstance(vv, bool))}
                if len(primitives) >= 2 and (finest is None or len(primitives) > len(finest)):
                    finest = primitives
        return finest

    @staticmethod
    def _inside_lines(knot: Any, identifier: str, deepness: int=0) -> bool:
        if 5 < deepness:
            return False
        if isinstance(knot, list):
            return any((isinstance(vv, dict) and identifier in [str(xx).strip() for xx in vv.values() if isinstance(xx, str)] for vv in knot)) or any((_Rows._inside_lines(vv, identifier, deepness + 1) for vv in knot))
        if isinstance(knot, dict):
            return any((_Rows._inside_lines(vv, identifier, deepness + 1) for vv in knot.values()))
        return False

    @staticmethod
    def _acquire_id_ident(tiers: list[dict[str, Any]]) -> str | None:
        finest = None
        for clef in tiers[0]:
            values = [tier.get(clef) for tier in tiers]
            if not all((_Rows._ident_resembling(vv, 'id' in str(clef).lower()) for vv in values)):
                continue
            values = [vv.strip() for vv in values]
            if len(tiers) > 1 and len(set(values)) < len(values):
                continue
            points = ('id' in str(clef).lower(), sum((len(vv) for vv in values)) / len(values))
            if finest is None or points > finest[0]:
                finest = (points, clef)
        return None if not finest else finest[1]

    @staticmethod
    def _ident_resembling(worth: Any, titled: bool=False) -> bool:
        if not isinstance(worth, str) or not 3 <= len(worth.strip()) <= 64 or ' ' in worth.strip():
            return False
        return titled or bool(re.search('\\d', worth))

class _Metronome:

    def __init__(self) -> None:
        self.beats = 0
        self.over_with = threading.Event()
        threading.Thread(target=self.count, daemon=True).start()

    def count(self) -> None:
        while not self.over_with.wait(0.25):
            self.beats += 1

    def secs(self) -> float:
        return self.beats * 0.25

    def cease(self) -> None:
        self.over_with.set()

class Article:

    def __init__(self, moniker: str, rack: str, units_n: int, obligations: list[str]):
        self.moniker = moniker
        self.rack = rack
        self.units_n = units_n
        self.obligations = obligations
        self.related_to: list[str] = []
        self.let_gone: list[str] = []
        self.unbound: list[str] = []

class _Driver:

    def conclude(self, once_more: bool=False) -> str:
        if self.concluded or self.snapped or self.leftover() <= 0:
            return ''
        (acquire, spurn) = (self.functions.get('buy'), self.functions.get('refuse'))
        newfound = self.trolley_outlook is not None and self.trolley_upon <= self.trolley_spotted_upon
        gripping = bool(self.trolley_strands) if not newfound else bool(self.trolley_numbers())
        if acquire and (gripping or (self.trolley_handled and (not newfound) and (self.leftover() >= 2))):
            flaws: list[str] = []
            operands = _Kinds._silhouette(dict(self.digest.get('buy_args') or {}), self.outlines.get(acquire), 'args', flaws)
            if not flaws:
                if self.leftover() >= 2:
                    self.barrier_check('order')
                if self.concluded or self.leftover() <= 0:
                    return ''
                retort = self.operate(acquire, operands)
                self.footprints.append({'kind': 'forced_buy', 'reply': _Text._shear(retort, 200)})
                if self.concluded or self.leftover() <= 0:
                    return ''
                if retort is not None and once_more:
                    self.acquires += 1
                    self.vetoed_upon = self.pace
                    self.watch(acquire, retort.get('observation'))
                    return 'the checkout was refused: ' + self.exhibit(retort.get('observation'))
        if spurn:
            flaws = []
            operands = _Kinds._silhouette(self.silent(spurn, dict(self.digest.get('refuse_args') or {})), self.outlines.get(spurn), 'args', flaws)
            if not flaws:
                if self.leftover() >= 2:
                    self.barrier_check('end')
                if self.concluded or self.leftover() <= 0:
                    return ''
                retort = self.operate(spurn, operands)
                self.footprints.append({'kind': 'forced_refuse', 'reply': _Text._shear(retort, 200)})
                if retort is not None and (not self.concluded):
                    self.watch(spurn, retort.get('observation'))
                    return 'ending without a purchase was refused: ' + self.exhibit(retort.get('observation'))
        return ''

    def duration_trolley(self) -> None:
        (trolley, acquire) = (self.functions.get('cart'), self.functions.get('buy'))
        if self.concluded or self.trolley_strands or (not trolley) or (not acquire) or (self.leftover() < 3):
            return
        if not self.clue_items:
            self.inspection()
        design: dict[str, int] = {}
        for (digit, serials) in self.clue_items.items():
            index = _Kinds._quantity_n(digit)
            elect = next((ii for ii in serials if not self.vended_away(ii)), None)
            if elect and index and (1 <= index <= len(self.items)) and (elect not in design):
                design[elect] = self.items[index - 1].units_n
        if not design:
            return
        self.footprints.append({'kind': 'time_cart', 'plan': design})
        for operands in self.trolley_dispatches(design, []) or []:
            flaws: list[str] = []
            operands = _Kinds._silhouette(operands, self.outlines.get(trolley), 'args', flaws)
            if flaws or self.leftover() < 2:
                return
            retort = self.operate(trolley, operands)
            if retort is None or self.concluded:
                return
            self.jot_trolley(trolley, operands)
            self.watch(trolley, retort.get('observation'))

    def execute_dispatches(self, dispatches: list[dict[str, Any]]) -> list[tuple[str, str]]:
        away: list[tuple[str, str]] = []
        dispatches = self.unite_unfolds_n(dispatches)
        for (index, dispatch) in enumerate(dispatches):
            permitted = {tt['function']['name'] for tt in self.proffered()}
            cident = dispatch['id']
            if dispatch.get('joined'):
                away.append((cident, 'Shown in the side-by-side result above: Kyo opened these listings together in one step.'))
                continue
            fun = dispatch.get('function') or {}
            moniker = fun.get('name')
            untreated = fun.get('arguments')
            if self.concluded:
                away.append((cident, 'Not run: the episode is over.'))
                continue
            if moniker == self.functions.get('ask'):
                away.append((cident, "Not run (no step used): Kyo's interviewer handles every message to the shopper."))
                continue
            if moniker not in self.outlines:
                away.append((cident, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(permitted))}."))
                continue
            if moniker not in permitted:
                away.append((cident, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(permitted)) + '.'))
                continue
            if not isinstance(untreated, dict):
                operands = {} if not (untreated or '').strip() else _Text._json_object(untreated)
                if operands is None:
                    away.append((cident, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            else:
                operands = untreated
            flaws: list[str] = []
            operands = _Kinds._silhouette(operands, self.outlines[moniker], 'args', flaws)
            if flaws:
                away.append((cident, 'Not run (no step used): ' + '; '.join(flaws[:6])))
                continue
            (operands, mistaken) = self.work_out_numbers(moniker, operands)
            if mistaken:
                away.append((cident, 'Not run (no step used): ' + mistaken + '.'))
                continue
            clipped = ''
            numerous = None if not moniker == self.functions.get('compare') else self.collate_destination()
            if numerous and isinstance(operands.get(numerous), list) and (len(operands[numerous]) > self.collate_headroom()) and self.collate_poor:
                headroom = self.collate_headroom()
                clipped = f"\n[Kyo] Sent the first {headroom} listings only: this store shows at most {headroom} side by side. Not shown yet: {', '.join(map(str, operands[numerous][headroom:]))}."
                operands = {**operands, numerous: operands[numerous][:headroom]}
            destination = self.singular_destination(moniker)
            if destination and moniker in (self.functions.get('open'), self.functions.get('offer')) and (str(operands.get(destination)) in self.departed):
                away.append((cident, f'Not run (no step used): {operands[destination]} sold out at step {self.departed[str(operands[destination])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if moniker == self.functions.get('cart') and any(((_Kinds._quantity_n(vv) or 0) < 0 for vv in operands.values() if not isinstance(vv, str))):
                outlook = '' if not self.trolley_upon <= self.trolley_spotted_upon else _Text._serialize(self.trolley_outlook)
                missing_out = [vv for vv in operands.values() if isinstance(vv, str) and vv in self.tiers and (vv not in self.trolley_strands) and (vv not in outlook) and (vv not in self.trolley_uttered)]
                if missing_out:
                    away.append((cident, f'Not run (no step used): {missing_out[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.trolley_wording()))
                    continue
            terminating = self.be_acquire(moniker, operands) or self.be_spurn(moniker, operands)
            if terminating and len(dispatches) > 1:
                away.append((cident, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not terminating and self.leftover() <= self.end_part() and (self.functions.get('buy') or self.functions.get('refuse')):
                away.append((cident, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            tag_mark = moniker + _Text._serialize(operands)
            if not terminating and moniker != self.functions.get('cart') and (tag_mark in self.latest_few):
                away.append((cident, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            perusers = {self.functions.get(rr) for rr in ('search', 'open', 'offer', 'compare')} - {None}
            hoarded = self.peruse_hoard.get(tag_mark)
            if moniker in perusers and hoarded and (hoarded[0] == self.edition):
                away.append((cident, f'Not run (no step used): the same call at step {hoarded[1]} returned this, and nothing has changed since:\n' + hoarded[2][:3000]))
                continue
            if destination and moniker in perusers and (moniker != self.functions.get('compare')) and self.exhibited_previously(moniker, str(operands.get(destination))):
                away.append((cident, f'Not run (no step used): {operands[destination]} has shown everything this view gives since the last change. Its record: ' + self.ledger_from(str(operands[destination]))))
                continue
            if numerous and isinstance(operands.get(numerous), list) and operands[numerous]:
                hold_on = [ii for ii in operands[numerous] if not self.exhibited_previously(moniker, str(ii))]
                if not hold_on:
                    away.append((cident, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{ii}: {self.ledger_from(str(ii))}' for ii in operands[numerous]))[:4000]))
                    continue
                if len(hold_on) < len(operands[numerous]):
                    clipped += f"\n[Kyo] Left out {', '.join((str(ii) for ii in operands[numerous] if ii not in hold_on))}: shown in full already, nothing changed since."
                    operands = {**operands, numerous: hold_on}
                    tag_mark = moniker + _Text._serialize(operands)
            if terminating and self.leftover() > 2:
                if self.be_acquire(moniker, operands):
                    broader = self.broader_grip()
                    if broader:
                        away.append((cident, broader + '\nCart now: ' + self.trolley_wording()))
                        continue
                    reperuse = self.reperuse_proffers()
                    if self.jot_vended:
                        self.discover_vended()
                    if self.trolley_handled and self.trolley_upon > self.trolley_spotted_upon and self.functions.get('cart_view') and (self.leftover() > 3):
                        flaws = []
                        outlook_operands = _Kinds._silhouette({}, self.outlines.get(self.functions['cart_view']), 'args', flaws)
                        outlook = None if flaws else self.operate(self.functions['cart_view'], outlook_operands)
                        if outlook is not None:
                            self.trolley_outlook = outlook.get('observation')
                            self.trolley_spotted_upon = self.pace
                            shift = self.watch(self.functions['cart_view'], self.trolley_outlook)
                            if shift:
                                self.lined_up = []
                                away.append((cident, 'Not run yet (no step used). Looking at the cart showed a change: ' + shift + '\nCart now: ' + self.trolley_wording() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    exchange = self.exchange_signal()
                    withheld = [] if not 2 > self.withheld else [ii for ii in self.trolley_numbers() if ii in self.departed]
                    if exchange or withheld:
                        self.withheld += 1 if not exchange else 0
                        away.append((cident, 'Not run yet (no step used). ' + (exchange or f"{', '.join(withheld)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.trolley_wording()))
                        continue
                    (linkage, grip) = self.wake()
                    if grip:
                        away.append((cident, 'Not run yet (no step used). ' + linkage + '\nCart now: ' + self.trolley_wording()))
                        continue
                    mismatch = [] if not (2 > self.conform_persists and self.leftover() > self.end_part() + 3) else self.conform_flaws()
                    if mismatch:
                        self.conform_persists += 1
                        self.footprints.append({'kind': 'fit_hold', 'faults': mismatch[:4]})
                        away.append((cident, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(mismatch[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.trolley_wording()))
                        continue
                    flaws = [] if not (2 > self.strand_persists and self.leftover() > self.end_part() + 2) else self.strand_flaws()
                    stable = []
                    if flaws and all((' is held at ' in ff for ff in flaws)):
                        stable = self.mend_quantities(self.quantity_n_mends())
                        flaws = flaws if not stable else self.strand_flaws()
                        if stable and (not flaws):
                            self.strand_persists += 1
                            away.append((cident, "Not run yet (no step used). Kyo set the cart's quantities itself: " + '; '.join(stable) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.trolley_wording()))
                            continue
                    if flaws:
                        self.strand_persists += 1
                        self.footprints.append({'kind': 'line_hold', 'faults': flaws[:6]})
                        away.append((cident, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(flaws[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.trolley_wording()))
                        continue
                    peruse = self.stuff_particulars(self.trolley_numbers(), self.end_part() + 3) + self.lower_cost_lines()
                    finding = self.probe_trolley()
                    if finding:
                        spotted = '' if not peruse else "Kyo read these listings' attributes for the check (one step each):\n" + '\n'.join(peruse) + '\n'
                        away.append((cident, 'Not run yet. ' + linkage + spotted + finding + '\nCart now: ' + self.trolley_wording()))
                        continue
                    if reperuse and OUTLAY_MILD < self.expense:
                        away.append((cident, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reperuse + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.leftover() > 5:
                    reappraisal = self.inspection()
                    if reappraisal:
                        away.append((cident, 'Not run yet (no step used). ' + reappraisal))
                        continue
                    if self.clues and self.leftover() > 4 and (0 == self.spurn_segments):
                        self.spurn_segments = 1
                        self.footprints.append({'kind': 'refuse_blocked', 'leads': list(self.clues)})
                        away.append((cident, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.clues) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.clues and self.leftover() > 4 and (1 == self.spurn_segments):
                        self.spurn_segments = 2
                        initial = self.clues[0]
                        self.probe_trolley([initial], OUTLAY_FIRM)
                        if self.findings.get(self.probe_clef([initial])) and initial in self.clues:
                            self.ratified = initial
                            self.footprints.append({'kind': 'refuse_blocked', 'confirmed': initial})
                            away.append((cident, f"Not run yet (no step used). An independent check of {initial}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.be_spurn(moniker, operands) and self.leftover() <= 5 and (not self.tardy_withheld):
                strands = [ii for ii in self.assembled if not self.vended_away(ii)]
                newfound = self.trolley_outlook is not None and self.trolley_upon <= self.trolley_spotted_upon
                lacking = [ii for ii in strands if ii not in self.trolley_numbers() or (ii in self.let_gone and (not newfound))]
                if len(strands) >= len(self.items) and len(lacking) + 1 <= self.leftover():
                    self.tardy_withheld = True
                    self.footprints.append({'kind': 'late_decline_held', 'lines': strands, 'missing': lacking})
                    rework = 'Check out the cart as it is.' if not lacking else f"Put {', '.join(lacking)} back in the cart at the requested quantities, then check out."
                    away.append((cident, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + rework + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.be_spurn(moniker, operands) and (not self.tardy_salvage) and (not self.trolley_strands) and (3 <= self.leftover() <= 5):
                self.tardy_salvage = True
                self.duration_trolley()
                if self.trolley_strands and (not self.concluded):
                    self.footprints.append({'kind': 'late_rescue', 'lines': dict(self.trolley_strands)})
                    away.append((cident, "Not run yet (no step used). Ending without a purchase earns nothing when a listing comes close, so Kyo put the review's closest listing for each item in the cart: " + self.trolley_wording() + '. Check out now; end without a purchase only if one of these clearly breaks a requirement the request makes firm.'))
                    continue
            if self.be_spurn(moniker, operands) and (not self.clues) and (not self.shoved) and (self.leftover() > self.upper_paces // 3):
                self.shoved = True
                away.append((cident, f"Not run yet (no step used). {self.leftover()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.be_spurn(moniker, operands):
                over_with = self.entire_trolley()
                if over_with:
                    away.append((cident, over_with + ('' if not self.concluded else ' [episode finished]')))
                    continue
                operands = self.silent(moniker, operands)
            if terminating and self.leftover() >= 2:
                (proceed, uttered) = self.barrier_check('end' if not self.be_acquire(moniker, operands) else 'order')
                if not proceed:
                    away.append((cident, 'Not run yet. ' + uttered + '\nCart now: ' + self.trolley_wording()))
                    continue
                if uttered:
                    clipped = '\n[Kyo] ' + uttered + clipped
            retort = self.operate(moniker, operands)
            if retort is None:
                away.append((cident, 'The call failed (network or server). You may try again.'))
                continue
            sighting = retort.get('observation')
            if numerous and isinstance(operands.get(numerous), list) and (len(operands[numerous]) > 1):
                exhibited = _Text._serialize(sighting)
                if any((str(ii) in exhibited for ii in operands[numerous])):
                    self.collate_fine = max(self.collate_fine, len(operands[numerous]))
                else:
                    self.collate_poor = min(self.collate_poor or 999, len(operands[numerous]))
                    clipped += f'\n[Kyo] The side-by-side view showed nothing for {len(operands[numerous])} listings: send at most {self.collate_headroom()} per call.'
            self.latest_few = (self.latest_few + [tag_mark])[-3:]
            self.ident_operands.update(((moniker, kk) for (kk, vv) in operands.items() if ([vv] if not isinstance(vv, list) else vv) and all((isinstance(xx, str) and xx in self.tiers for xx in ([vv] if not isinstance(vv, list) else vv)))))
            self.soak_up(sighting)
            if moniker == self.functions.get('compare'):
                tiers = self.ledgers_inside(sighting)
                if tiers:
                    self.particulars_of[moniker] = self.particulars_of.get(moniker, False) or any((_Rows._datum_slab(rr) for (_, rr) in tiers))
            if moniker in perusers - {self.functions.get('search')}:
                self.jot_outlook(moniker, sighting)
            simple = {utensil for (chore, utensil) in self.functions.items() if 'search' != chore}
            leafed = '' if not (moniker not in simple and (not terminating)) else self.extra_folios(moniker, operands, sighting)
            if self.singular_destination(moniker) and moniker not in (self.functions.get('cart'), self.functions.get('buy'), self.functions.get('refuse'), self.functions.get('ask')):
                posting = None if not isinstance(sighting, dict) else self.identifier(sighting)
                if posting:
                    self.particulars_of[moniker] = self.particulars_of.get(moniker, False) or bool(_Rows._datum_slab(sighting))
                    if moniker == self.functions.get('open') and (not self.particulars_of[moniker]):
                        different = next((tt for tt in self.outlines if moniker != tt and self.singular_destination(tt) and (self.particulars_of.get(tt) is not False) and (tt not in (self.functions.get('cart'), self.functions.get('buy'), self.functions.get('refuse'), self.functions.get('ask'), self.functions.get('search')))), None)
                        if different:
                            self.functions['open'] = different
                    if not self.particulars_of[moniker] and moniker != self.functions.get('open') and (not self.functions.get('offer')):
                        self.functions['offer'] = moniker
                    if moniker == self.proffer_utensil():
                        self.proffer_spotted[posting] = self.pace
            if moniker == self.functions.get('cart'):
                self.jot_trolley(moniker, operands)
                self.trolley_uttered = _Text._serialize(sighting)
                self.latest_few = []
                quantities = [_Kinds._quantity_n(vv) for vv in operands.values() if not isinstance(vv, str) and _Kinds._quantity_n(vv) is not None]
                for worth in operands.values():
                    if quantities and isinstance(worth, str) and (worth in self.tiers):
                        if quantities[0] <= 0 and (not _Rows._inside_lines(sighting, worth)):
                            self.let_gone.add(worth)
                        elif quantities[0] > 0:
                            self.let_gone.discard(worth)
                strands = [ii for ii in self.trolley_numbers() if not self.vended_away(ii) and ii not in self.let_gone]
                if len(strands) >= len(self.assembled):
                    self.assembled = strands
            if moniker == self.functions.get('search') and self.ledgers_inside(sighting):
                self.combed = (self.combed + [dict(operands)])[-8:]
            if moniker == self.functions.get('cart_view'):
                self.trolley_outlook = sighting
                self.trolley_spotted_upon = self.pace
            if self.be_acquire(moniker, operands):
                self.acquires += 1
                if not retort.get('finished'):
                    self.vetoed_upon = self.pace
            wording = self.exhibit(sighting) + leafed + clipped
            if moniker not in simple and (not terminating):
                wording += self.thriftiest_hereabouts([sighting] + self.leafed_observation) + self.lid_overlooked(moniker, operands, sighting)
            footing = wording
            attestation = self.support(sighting)
            if attestation:
                wording += '\n' + attestation
            if moniker == self.functions.get('cart') and any((_Kinds._quantity_n(vv) == 0 for vv in operands.values() if not isinstance(vv, str))):
                if any((isinstance(vv, str) and vv in self.tiers and _Rows._inside_lines(sighting, vv) for vv in operands.values())):
                    wording += '\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.jot_vended = False
            strands = self.lax_strands(sighting)
            if strands:
                sense = self.seize_jot(strands)
                if sense:
                    wording += '\n[note read] ' + sense + '. Apply this change to your picks before checking out.'
            vetoed = self.be_acquire(moniker, operands) and (not retort.get('finished'))
            once_more = vetoed and any((strand in self.voiced(sighting) for strand in self.vended_strands))
            lax = self.jot_vended and (vetoed or (not self.jot_titled and (not self.be_acquire(moniker, operands))))
            if (lax or once_more) and self.trolley_numbers() and (not self.concluded):
                spotted = self.discover_vended()
                if spotted:
                    wording += "\n[Kyo re-read the cart's lines after the refusal]\n" + spotted
            self.jot_vended = False
            exchange = self.exchange_signal()
            if exchange:
                wording += '\n' + exchange
            if self.lined_up:
                wording += "\n[notes seen on Kyo's own calls] " + ' | '.join(self.lined_up)
                self.lined_up = []
            if moniker in perusers:
                self.peruse_hoard[tag_mark] = (self.edition, self.pace, footing)
            if retort.get('finished'):
                wording += ' [episode finished]'
            away.append((cident, wording))
        return away

class _Checkout:

    def unite_unfolds_n(self, dispatches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        (unfolder, uniter, destination) = (self.functions.get('open'), self.functions.get('compare'), self.singular_destination(self.functions.get('open')))
        numerous = self.collate_destination()
        if not unfolder or not uniter or (not destination) or (not numerous) or (uniter == unfolder):
            return dispatches
        if self.collate_naked():
            return dispatches
        elects: list[tuple[int, str]] = []
        for (index, dispatch) in enumerate(dispatches):
            fun = dispatch.get('function') or {}
            operands = None if not isinstance(fun.get('arguments'), str) else _Text._json_object(fun.get('arguments'))
            identifier = str((operands or {}).get(destination) or '').strip()
            if fun.get('name') == unfolder and operands and (set(operands) == {destination}) and (identifier in self.tiers) and (identifier not in self.departed):
                elects.append((index, identifier))
        elects = elects[:self.collate_headroom()]
        if len(elects) < 2:
            return dispatches
        initial = elects[0][0]
        away = []
        for (index, dispatch) in enumerate(dispatches):
            if initial == index:
                away.append({'id': dispatch['id'], 'type': 'function', 'function': {'name': uniter, 'arguments': _Text._serialize({numerous: [ii for (_, ii) in elects]})}})
            elif not index in {ii for (ii, _) in elects}:
                away.append(dispatch)
            else:
                away.append({'id': dispatch['id'], 'joined': True})
        self.footprints.append({'kind': 'joined', 'ids': [ii for (_, ii) in elects]})
        return away

    def collate_headroom(self) -> int:
        details = ((self.outlines.get(self.functions.get('compare') or '') or {}).get('properties') or {}).get(self.collate_destination() or '') or {}
        if isinstance(details.get('maxItems'), int):
            return details['maxItems']
        if self.collate_poor:
            return max(2, min(self.collate_poor - 1, max(self.collate_fine, 2)))
        return max(4, self.collate_fine)

    def collate_destination(self) -> str | None:
        outline = self.outlines.get(self.functions.get('compare') or '') or {}
        characteristics = outline.get('properties') or {}
        destination = next((kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'array' in _Kinds._genera(vv)), None)
        if not destination or any((destination != kk for kk in outline.get('required') or [])):
            return None
        return destination

    def work_out_numbers(self, moniker: str, operands: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if moniker not in {self.functions.get(rr) for rr in ('cart', 'open', 'offer', 'compare')} - {None} or not self.ident_clef:
            return (operands, '')
        characteristics = (self.outlines.get(moniker) or {}).get('properties') or {}
        unbound = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and ('string' in _Kinds._genera(vv) or 'array' in _Kinds._genera(vv)) and ('enum' not in vv)]
        if len(unbound) != 1 or (moniker != self.functions.get('cart') and (moniker, unbound[0]) not in self.ident_operands):
            return (operands, '')

        def mend(worth: Any) -> tuple[Any, str]:
            if isinstance(worth, str) and worth.strip() in self.tiers:
                return (worth.strip(), '')
            if not isinstance(worth, str) or _Rows._ident_resembling(worth, True) or len(worth.strip()) < 8:
                return (worth, '')
            scant = worth.strip().lower()
            finds = [ii for (ii, tier) in self.tiers.items() if any((self.ident_clef != kk and isinstance(vv, str) and vv.strip().lower().startswith(scant) for (kk, vv) in tier.items()))]
            if len(finds) == 1:
                return (finds[0], '')
            if finds:
                return (worth, f"several listings carry the text {worth!r} ({', '.join(finds[:6])}); pass one listing id")
            return (worth, f'{worth!r} is not a listing id; pass the id from the result rows (their {self.ident_clef} field)')
        away = dict(operands)
        for clef in unbound:
            if clef not in operands:
                continue
            stable = [mend(operands[clef])] if not isinstance(operands[clef], list) else [mend(vv) for vv in operands[clef]]
            snag = next((pp for (_, pp) in stable if pp), '')
            if snag:
                return (operands, snag)
            away[clef] = stable[0][0] if not isinstance(operands[clef], list) else [vv for (vv, _) in stable]
        return (away, '')

    def silent(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any]:
        outline = self.outlines.get(moniker) or {}
        characteristics = outline.get('properties') or {}
        away = dict(operands)
        for (clef, details) in characteristics.items():
            if not isinstance(details, dict) or 'string' not in _Kinds._genera(details) or 'enum' in details:
                continue
            worth = away.get(clef)
            if worth is None and clef not in (outline.get('required') or []):
                continue
            wording = str(worth or '')
            titled = any((identifier in wording for identifier in self.tiers)) or self.digest.get('quiet_decline') is not False
            if titled or not wording:
                away[clef] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:details.get('maxLength', 200)]
        return away

    def proffered(self) -> list[dict[str, Any]]:
        leftover = self.leftover()
        petition = self.functions.get('ask')
        hold_on: set[str] | None = None
        closes = {self.functions.get('buy'), self.functions.get('refuse')} - {None}
        if leftover <= self.end_part():
            hold_on = set(closes)
        elif leftover <= self.end_part() + 3:
            hold_on = set(closes) | {self.functions.get('cart'), self.functions.get('cart_view')}
        elif leftover <= len(self.items) + 5 and 0 == self.acquires:
            hold_on = set(closes) | {self.functions.get('cart'), self.functions.get('cart_view'), self.functions.get('offer')}
        utensils = [tt for tt in self.utensils if tt['function']['name'] != petition]
        if hold_on is not None and closes:
            thinned = [tt for tt in utensils if tt['function']['name'] in hold_on]
            utensils = thinned or utensils
        return utensils

    def be_spurn(self, moniker: str, operands: dict[str, Any]) -> bool:
        if moniker != self.functions.get('refuse'):
            return False
        stable = {} if not isinstance(self.digest.get('refuse_args'), dict) else self.digest.get('refuse_args')
        return all((operands.get(kk) == vv for (kk, vv) in stable.items()))

    def be_acquire(self, moniker: str, operands: dict[str, Any]) -> bool:
        if moniker != self.functions.get('buy'):
            return False
        stable = {} if not isinstance(self.digest.get('buy_args'), dict) else self.digest.get('buy_args')
        return all((operands.get(kk) == vv for (kk, vv) in stable.items()))

    def entire_trolley(self) -> str:
        (trolley, acquire) = (self.functions.get('cart'), self.functions.get('buy'))
        if self.wrapped_up or not self.ratified or self.vended_away(self.ratified) or (not trolley) or (not acquire):
            return ''
        self.wrapped_up = True
        genre_clef = self.classification_clef()

        def genus(identifier: str) -> str:
            return identifier if not genre_clef else str((self.tiers.get(identifier) or {}).get(genre_clef) or '').strip().lower()
        design: dict[str, int] = {}
        for (digit, serials) in self.clue_items.items():
            index = _Kinds._quantity_n(digit)
            elect = next((ii for ii in serials if not self.vended_away(ii)), None)
            if elect and index and (1 <= index <= len(self.items)) and (genus(elect) not in {genus(pp) for pp in design}):
                design[elect] = self.items[index - 1].units_n
        if self.ratified not in design:
            return ''
        presently = self.trolley_numbers()
        hold_on = {ii: self.trolley_strands.get(ii, 1) for ii in presently if ii not in design and (not self.vended_away(ii)) and (genus(ii) not in {genus(pp) for pp in design})}
        let_go = [ii for ii in presently if ii not in design and ii not in hold_on]
        if len(design) + len({genus(ii) for ii in hold_on}) < len(self.items):
            return ''
        strands = {**hold_on, **design}
        dispatches = self.trolley_dispatches(strands, let_go)
        if not dispatches or len(dispatches) + 1 > self.leftover():
            return ''
        self.stuff_particulars(list(strands), len(dispatches) + 3)
        self.probe_trolley(list(strands), OUTLAY_FIRM)
        if not self.findings.get(self.probe_clef(list(strands))):
            return ''
        over_with = []
        for operands in dispatches:
            flaws: list[str] = []
            operands = _Kinds._silhouette(operands, self.outlines.get(trolley), 'args', flaws)
            if flaws:
                return ''
            retort = self.operate(trolley, operands)
            if retort is None or self.concluded:
                return ''
            self.jot_trolley(trolley, operands)
            self.watch(trolley, retort.get('observation'))
            over_with.append(_Text._serialize(operands))
        flaws = []
        operands = _Kinds._silhouette(dict(self.digest.get('buy_args') or {}), self.outlines.get(acquire), 'args', flaws)
        if flaws:
            return ''
        retorts = []
        for essay in range(2):
            if essay:
                self.reperuse_proffers()
                self.probe_trolley(list(strands), OUTLAY_FIRM)
                if not self.findings.get(self.probe_clef(list(strands))) or self.leftover() < 1 or self.concluded or any((self.vended_away(ii) for ii in strands)):
                    break
            (proceed, uttered) = self.barrier_check('order')
            if uttered:
                retorts.append(uttered)
            if not proceed:
                break
            retort = self.operate(acquire, operands)
            self.footprints.append({'kind': 'completed', 'lines': strands, 'drop': let_go, 'reply': _Text._shear(retort, 200)})
            if retort is None:
                break
            self.acquires += 1
            if not retort.get('finished'):
                self.vetoed_upon = self.pace
            self.watch(acquire, retort.get('observation'))
            retorts.append(self.exhibit(retort.get('observation')))
            if self.concluded or retort.get('finished'):
                break
        if not retorts:
            return ''
        return 'Not run (Kyo acted instead): the independent check confirmed ' + self.ratified + ', so Kyo set the cart (' + '; '.join(over_with) + ') and checked out. Store reply: ' + ' | then: '.join(retorts)

    def trolley_dispatches(self, strands: dict[str, int], let_go: list[str]) -> list[dict[str, Any]] | None:
        moniker = self.functions.get('cart') or ''
        outline = self.outlines.get(moniker) or {}
        characteristics = outline.get('properties') or {}
        for (clef, details) in characteristics.items():
            silhouette = None if not (isinstance(details, dict) and 'array' in _Kinds._genera(details)) else details.get('items')
            if isinstance(silhouette, dict) and isinstance(silhouette.get('properties'), dict):
                inside = silhouette['properties']
                ident_clef = next((kk for (kk, ss) in inside.items() if isinstance(ss, dict) and 'string' in _Kinds._genera(ss) and ('enum' not in ss)), None)
                quantity_n_clef = next((kk for (kk, ss) in inside.items() if isinstance(ss, dict) and 'integer' in _Kinds._genera(ss)), None)
                if not ident_clef:
                    return None
                return [{clef: [{ident_clef: ii, **({} if not quantity_n_clef else {quantity_n_clef: nn})} for (ii, nn) in strands.items()]}]
        wordings = [kk for (kk, ss) in characteristics.items() if isinstance(ss, dict) and 'string' in _Kinds._genera(ss) and ('enum' not in ss)]
        destination = next((kk for (tt, kk) in self.ident_operands if moniker == tt and kk in wordings), None) or next((kk for kk in outline.get('required') or [] if kk in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        count = next((kk for (kk, ss) in characteristics.items() if isinstance(ss, dict) and 'integer' in _Kinds._genera(ss)), None)
        fashion = next((kk for (kk, ss) in characteristics.items() if isinstance(ss, dict) and isinstance(ss.get('enum'), list) and (len(ss['enum']) >= 2)), None)
        if not destination:
            return None
        dispatches: list[dict[str, Any]] = []
        for identifier in let_go:
            possess = self.trolley_strands.get(identifier, 1)
            if fashion:
                dispatches.append({destination: identifier, fashion: characteristics[fashion]['enum'][1], **({} if not count else {count: possess})})
            elif count and self.trolley_sets(moniker):
                dispatches.append({destination: identifier, count: 0})
            elif not (count and isinstance(characteristics[count].get('minimum'), int) and (characteristics[count]['minimum'] <= -possess)):
                return None
            else:
                dispatches.append({destination: identifier, count: -possess})
        for (identifier, craving) in strands.items():
            possess = 0 if not identifier in self.trolley_numbers() else self.trolley_strands.get(identifier, 0)
            if craving == possess:
                continue
            if possess and (not self.trolley_sets(moniker)):
                return None
            dispatches.append({destination: identifier, **({} if not fashion else {fashion: characteristics[fashion]['enum'][0]}), **({} if not count else {count: craving})})
        return dispatches

class _Audit:

    def unfold_lines(self, tiers: list[str]) -> bool:
        (numerous, uniter) = (self.collate_destination(), self.functions.get('compare'))
        utensil = self.particulars_utensil()
        destination = self.singular_destination(utensil)
        if numerous and uniter and (not self.collate_naked()):
            design = [(uniter, {numerous: tiers[:self.collate_headroom()]})]
        elif not (utensil and destination):
            return False
        else:
            design = [(utensil, {destination: ii}) for ii in tiers[:2]]
        if self.leftover() < len(design) + 5:
            return False
        prior_to = len(self.complete_)
        for (moniker, operands) in design:
            flaws: list[str] = []
            operands = _Kinds._silhouette(operands, self.outlines.get(moniker), 'args', flaws)
            if flaws:
                return False
            retort = self.operate(moniker, operands)
            if retort is None or self.concluded:
                break
            self.watch(moniker, retort.get('observation'))
        self.footprints.append({'kind': 'audit_open', 'rows': tiers, 'opened': len(self.complete_) - prior_to})
        return len(self.complete_) > prior_to

    def inspection_conforms(self, gained: dict[str, Any] | None) -> dict[str, list[str]]:
        conforms = (gained or {}).get('fits')
        if not isinstance(conforms, dict):
            return {}
        titled = {str(kk): [str(ii) for ii in _Kinds._entries_of(vv) if str(ii) in self.tiers and (not self.vended_away(str(ii)))][:4] for (kk, vv) in conforms.items()}
        return {kk: vv for (kk, vv) in titled.items() if vv}

    def inspection_dispatch(self) -> dict[str, Any] | None:
        ledgers = []
        for identifier in reversed(self.spotted_purchase):
            if self.vended_away(identifier):
                continue
            entry_rec = None if not identifier in self.complete_ else self.complete_.get(identifier)
            ledgers.append('ROW ' + _Text._serialize(_Text._shear(self.tiers.get(identifier), 120))[:500] if not entry_rec else 'OPENED ' + _Text._serialize(_Text._shear(entry_rec, 200))[:1200])
            if len(ledgers) >= 30:
                break
        if not ledgers:
            return None
        instruction = INSPECTION + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        instruction += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(ledgers)
        gained = self.petition_json(instruction, self.assessors(), 4000)
        self.footprints.append({'kind': 'audit', 'got': gained})
        return gained

    def inspection(self) -> str:
        if self.inspected or not self.assessors():
            return ''
        self.inspected = True
        gained = self.inspection_dispatch()
        titled = self.inspection_conforms(gained)
        if not titled:
            tiers = [str(ii) for ii in _Kinds._entries_of((gained or {}).get('open')) if str(ii) in self.tiers and str(ii) not in self.unfolded and (not self.vended_away(str(ii)))]
            if tiers and self.unfold_lines(tiers):
                gained = self.inspection_dispatch()
                titled = self.inspection_conforms(gained)
        if not titled:
            return ''
        self.clue_items = titled
        self.clues = [ii for serials in titled.values() for ii in serials]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {kk}: {', '.join(vv)}" for (kk, vv) in titled.items())) + ('' if not gained.get('reason') else f" ({gained.get('reason')})") + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

class _Offers:

    def substitutions(self, mislaid: list[str]) -> list[str]:
        if not self.assessors():
            return []
        ledgers = []
        for identifier in reversed(self.spotted_purchase):
            if identifier in mislaid or identifier in self.departed or self.vended_away(identifier):
                continue
            ledgers.append(_Text._serialize(_Text._shear(self.complete_.get(identifier) or self.tiers.get(identifier), 200))[:1200])
            if len(ledgers) >= 25:
                break
        if not ledgers:
            return []
        instruction = EXCHANGE + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        instruction += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{ii}: {self.ledger_from(ii)}' for ii in mislaid)) + '\n\nCART NOW:\n' + self.trolley_wording()
        instruction += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(ledgers)
        gained = self.petition_json(instruction, self.assessors(), 3500)
        conforms: list[tuple[float, str]] = []
        for item_rec in _Kinds._entries_of((gained or {}).get('fits')):
            identifier = str(item_rec if not isinstance(item_rec, dict) else item_rec.get('id'))
            tariff = None if not isinstance(item_rec, dict) else item_rec.get('price')
            if identifier in self.tiers and identifier not in mislaid and (identifier not in self.departed) and (identifier not in self.trolley_numbers()) and (identifier not in [ii for (_, ii) in conforms]):
                conforms.append((float('inf') if not (isinstance(tariff, (int, float)) and (not isinstance(tariff, bool))) else float(tariff), identifier))
        ordered_by = [ii for (_, ii) in conforms] if not self.digest.get('pick') == 'cheapest' else [ii for (_, ii) in sorted(conforms, key=lambda ff: ff[0])]
        self.footprints.append({'kind': 'swap', 'lost': mislaid, 'fits': conforms})
        return ordered_by[:4]

    def discover_vended(self) -> str:
        serials = [ii for ii in self.trolley_numbers() if ii not in self.departed]
        if any((self.vended_away(ii) for ii in serials)):
            return ''
        if len(serials) == 1:
            self.not_offered.add(serials[0])
            return ''
        return '' if not serials else self.reperuse_proffers()

    def exchange_signal(self) -> str:
        mislaid = [ii for ii in self.trolley_numbers() if ii not in self.departed and self.vended_away(ii)]
        if not mislaid:
            return ''
        for identifier in mislaid:
            self.departed[identifier] = self.pace
        self.clues = [ii for ii in self.clues if ii not in mislaid]
        self.edition += 1
        elects = self.substitutions(mislaid)
        wording = f"[Kyo] Sold out now: {', '.join(mislaid)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if elects and self.leftover() <= 6:
            return wording + f" Few steps are left: in this turn add {elects[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(elects[1:]) or 'none'}."
        if elects and self.digest.get('pick') == 'cheapest':
            return wording + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(elects)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {elects[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if elects:
            return wording + f" Listings already read that appear to fit that item instead: {', '.join(elects)}. In this turn take the sold-out line out of the cart and add {elects[0]} at the requested quantity, then check out."
        return wording + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def vended_away(self, identifier: str) -> bool:
        return identifier in self.not_offered or identifier in self.departed

    def reperuse_proffers(self) -> str:
        utensil = self.proffer_utensil()
        destination = self.singular_destination(utensil)
        if not utensil or not destination:
            return ''
        given_that = max(self.pace - 2, self.vetoed_upon)
        musty = [ii for ii in self.trolley_numbers() if self.proffer_spotted.get(ii, -99) < given_that and ii not in self.departed]
        if not musty or self.leftover() < len(musty) + 3:
            return ''
        strands = []
        for identifier in musty:
            flaws: list[str] = []
            operands = _Kinds._silhouette({destination: identifier}, self.outlines.get(utensil), 'args', flaws)
            if flaws:
                break
            prior_to = dict(self.tiers.get(identifier) or {})
            retort = self.operate(utensil, operands)
            if retort is None or self.concluded:
                break
            sighting = retort.get('observation')
            self.proffer_spotted[identifier] = self.pace
            self.watch(utensil, sighting)
            following = self.tiers.get(identifier) or {}
            migrated = [kk for (kk, vv) in following.items() if kk in prior_to and prior_to[kk] != vv and (not isinstance(vv, (dict, list)))]
            if migrated:
                strands.append(f"{identifier} changed ({', '.join(migrated[:4])}): {_Text._serialize(_Text._shear(sighting, 300))[:900]}")
        self.footprints.append({'kind': 'reread', 'ids': musty, 'changed': len(strands)})
        if strands:
            self.edition += 1
        return '\n'.join(strands)

    def proffer_utensil(self) -> str | None:
        return self.functions.get('offer') or self.functions.get('open')

class _Pointers:

    def discover_tie(self, concerning: str, vocables: str) -> list[str]:
        noted: list[str] = []
        titled = {ww for identifier in self.emporium_lines(concerning, list(self.tiers)) + self.unfolded_emporium(concerning) for (kk, vv) in (self.tiers.get(identifier) or {}).items() if isinstance(vv, str) and len(vv.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kk), re.I) for ww in re.findall('[a-z0-9]{3,}', vv.lower())}
        values = {ww for ww in re.findall('[a-z0-9]+', (vocables or concerning).lower()) if len(ww) >= 3 or ww.isdigit()} - titled - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for identifier in self.unfolded_emporium(concerning):
            (pile, quits) = ([self.complete_.get(identifier) or self.tiers.get(identifier) or {}], [])
            while pile:
                knot = pile.pop()
                if not isinstance(knot, (dict, list)):
                    quits.append(str(knot))
                else:
                    pile += knot if not isinstance(knot, dict) else list(knot.values())
            wording = ' '.join(quits).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(vv) + '(?![a-z0-9])', wording) for vv in values)):
                self.footprints.append({'kind': 'link_known', 'id': identifier})
                return []
        design = self.search_pattern()
        note = ' '.join(vocables.lower().split())
        if note and note in self.tie_spotted:
            self.tie_hunts += 1
            noted = self.emporium_lines(concerning, self.tie_spotted[note])
        elif design and vocables:
            (utensil, footing, column) = design
            flaws: list[str] = []
            operands = _Kinds._silhouette({**footing, column: vocables}, self.outlines.get(utensil), 'args', flaws)
            if not flaws:
                self.tie_hunts += 1
                retort = self.operate(utensil, operands)
                if retort is not None:
                    sighting = retort.get('observation')
                    self.watch(utensil, sighting)
                    self.tie_spotted[note] = [ii for (ii, _) in self.ledgers_inside(sighting)]
                    noted = self.emporium_lines(concerning, self.tie_spotted[note])
                    self.footprints.append({'kind': 'link_search', 'words': vocables[:80], 'rows': noted[:4]})
        genus = {ww for ww in re.findall('[a-z0-9]{3,}', (vocables or concerning).lower())} - titled - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def from_genus(serials: list[str]) -> list[str]:
            return [ii for ii in serials if not genus or genus & set(re.findall('[a-z0-9]{3,}', _Text._serialize(self.tiers.get(ii) or {}).lower()))]
        return (from_genus(noted) or from_genus(self.emporium_lines(concerning, list(reversed(self.spotted_purchase)))))[:2]

    def unfolded_emporium(self, concerning: str) -> list[str]:
        scant = concerning.lower()
        away = []
        for identifier in self.unfolded:
            tier = self.tiers.get(identifier) or {}
            monikers = [str(vv).strip() for (kk, vv) in tier.items() if isinstance(vv, str) and 2 < len(vv.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kk), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', scant) for nn in monikers)):
                away.append(identifier)
        return away

    def emporium_lines(self, concerning: str, serials: list[str]) -> list[str]:
        scant = concerning.lower()
        away = []
        for identifier in serials:
            tier = self.tiers.get(identifier) or {}
            monikers = [str(vv).strip() for (kk, vv) in tier.items() if isinstance(vv, str) and 2 < len(vv.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(kk), re.I)]
            if identifier not in self.unfolded and any((re.search('(?<![a-z0-9])' + re.escape(nn.lower()) + '(?![a-z0-9])', scant) for nn in monikers)):
                away.append(identifier)
        return away

    def wake(self) -> tuple[str, bool]:
        current = {necessity for (_, necessity) in self.necessities_presently()}
        outstanding = [necessity for (_, necessity) in self.allusions if necessity in current and necessity not in self.allusion_worths]
        utensil = self.functions.get('offer') or self.functions.get('open')
        destination = self.singular_destination(utensil)
        spotted: list[str] = []
        mislaid: list[str] = []
        while outstanding and self.wakes < (4 if not self.leftover() > 15 else 6) and (OUTLAY_FIRM > self.expense):
            self.wakes += 1
            ledgers = []
            for identifier in reversed(self.spotted_purchase):
                entry_rec = None if not identifier in self.unfolded else self.complete_.get(identifier) or self.tiers.get(identifier)
                if entry_rec and len(ledgers) < 45:
                    ledgers.append('OPENED ' + _Text._serialize(_Text._shear(entry_rec, 300))[:1400])
            for identifier in reversed(self.spotted_purchase):
                if len(ledgers) >= 60:
                    break
                if identifier not in self.unfolded or not (self.complete_.get(identifier) or self.tiers.get(identifier)):
                    ledgers.append('ROW ' + _Text._serialize(_Text._shear(self.tiers.get(identifier), 80))[:300])
            instruction = WAKE + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(outstanding)
            instruction += '\n\nRECORDS (newest first):\n' + '\n'.join(ledgers)
            gained = self.petition_json(instruction, [NIMBLE] + ([] if not self.expense + WEIGHTY_HEADROOM <= OUTLAY_FIRM else [WEIGHTY]), 4500) or {}
            self.footprints.append({'kind': 'trail', 'pending': outstanding, 'got': _Text._shear(gained, 150)})
            span: list[str] = []
            mislaid = []
            spacings: list[tuple[str, str]] = []
            for linkage in _Kinds._entries_of(gained.get('chains')):
                if not isinstance(linkage, dict):
                    continue
                necessity = next((pp for pp in outstanding if pp == str(linkage.get('need') or '').strip()), None) or (None if not len(outstanding) == 1 else outstanding[0])
                if not necessity:
                    continue
                ties = [ll for ll in _Kinds._entries_of(linkage.get('links')) if isinstance(ll, dict)]
                serials = [str(ll.get('id') or '').strip() for ll in ties]
                if linkage.get('done') is True and ties and all((ii in self.unfolded for ii in serials)) and str(linkage.get('value') or '').strip():
                    self.allusion_worths[necessity] = str(linkage['value']).strip()
                    self.edition += 1
                    continue
                if ties and all((ii in self.unfolded for ii in serials)) and (not str(ties[-1].get('find') or '').strip()):
                    self.allusion_worths[necessity] = str(linkage.get('value') or '').strip() or f'as stated on {serials[-1]}'
                    self.edition += 1
                    continue
                shut = [ii for ii in serials if ii in self.tiers and ii not in self.unfolded]
                if not shut:
                    spacing = next((ll for (ll, ii) in zip(ties, serials) if ii not in self.tiers), {})
                    concerning = str(spacing.get('about') or '') or 'the listing it points at'
                    vocables = str(spacing.get('find') or '').strip()
                    mislaid.append(f'{necessity} -> find {concerning}' + ('' if not vocables else f' (search: {vocables})'))
                    spacings.append((concerning, vocables))
                else:
                    span += [ii for ii in shut if ii not in span]
            outstanding = [pp for pp in outstanding if pp not in self.allusion_worths]
            if not span and spacings and utensil and destination and (self.leftover() > 12) and (3 > self.tie_hunts):
                span = self.discover_tie(*spacings[0])
                if self.footprints and self.footprints[-1].get('kind') == 'link_known':
                    spotted.append(f"{self.footprints[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    mislaid = mislaid[1:]
            if not span or not utensil or (not destination) or (self.leftover() < len(span) + 4) or self.concluded:
                break
            for identifier in span[:max(0, min(3, 6 - len(spotted)))]:
                flaws: list[str] = []
                operands = _Kinds._silhouette({destination: identifier}, self.outlines.get(utensil), 'args', flaws)
                if flaws:
                    break
                retort = self.operate(utensil, operands)
                if retort is None or self.concluded:
                    break
                self.watch(utensil, retort.get('observation'))
                spotted.append(f'{identifier}: ' + self.exhibit(retort.get('observation'))[:1500])
        wording = ''
        if spotted:
            wording += '[Kyo opened the pages a requirement points at]\n' + '\n'.join(spotted) + '\n'
        if self.allusion_worths:
            wording += '[values read from the pages requirements point at] ' + '; '.join((f'{kk} -> {vv}' for (kk, vv) in self.allusion_worths.items())) + '\n'
        grip = bool(mislaid) and self.wake_segments < (2 if not self.leftover() > 15 else 3) and (self.leftover() > 5)
        if grip:
            self.wake_segments += 1
            wording += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(mislaid[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (wording, grip)

class _Fitting:

    def probe_trolley(self, serials: list[str] | None=None, boundary: float=OUTLAY_MILD) -> str:
        clue = serials is not None
        serials = self.trolley_numbers() if not clue else serials
        clef = self.probe_clef(serials)
        if clef in self.probed or not self.assessors(boundary) or (not serials):
            return ''
        self.probed.add(clef)
        instruction = PROBE + '\n\nREQUEST:\n' + self.appeal + '\n\nREQUIREMENTS IN FORCE:\n' + self.register()
        if not clue:
            instruction += '\n\nCART:\n' + self.trolley_wording()
        else:
            headcount = '; '.join((f'item {nn}: quantity {that.units_n}' for (nn, that) in enumerate(self.items, 1)))
            instruction += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + headcount + '):\n' + ', '.join(serials)
        instruction += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{ii}: ' + ('' if not self.vended_away(ii) else 'SOLD OUT (the store says it can no longer be bought). ') + self.ledger_from(ii) for ii in serials))
        if self.digest.get('pick') in ('cheapest', 'most', 'least'):
            instruction += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.remaining(serials, 15)
        gained = self.petition_json(instruction, self.assessors(boundary), 3000)
        self.footprints.append({'kind': 'check', 'cart': serials, 'got': gained})
        self.findings[clef] = bool(gained) and gained.get('ok') is not False
        if not gained or gained.get('ok') is not False:
            return ''
        self.clues = [ii for ii in self.clues if ii not in serials]
        snags = [str(pp) for pp in _Kinds._entries_of(gained.get('problems'))][:5]
        superior = [str(bb) for bb in _Kinds._entries_of(gained.get('better')) if str(bb) in self.tiers and (not self.vended_away(str(bb)))][:4]
        wording = 'A careful review says this cart is not right yet: ' + '; '.join(snags or ['see the requirements'])
        if superior:
            wording += '. Listings that look better: ' + ', '.join(superior)
        return wording + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def probe_clef(self, serials: list[str]) -> str:
        return '|'.join(sorted(serials)) + f'#{self.edition}'

    def mend_quantities(self, mistaken: list[tuple[str, int, int]]) -> list[str]:
        moniker = self.functions.get('cart') or ''
        outline = self.outlines.get(moniker) or {}
        characteristics = outline.get('properties') or {}
        wordings = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'string' in _Kinds._genera(vv) and ('enum' not in vv)]
        destination = next((kk for (tt, kk) in self.ident_operands if moniker == tt and kk in wordings), None) or next((kk for kk in outline.get('required') or [] if kk in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        count = next((kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'integer' in _Kinds._genera(vv)), None)
        fashion = next((kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and isinstance(vv.get('enum'), list) and (len(vv['enum']) >= 2)), None)
        if not moniker or not destination or (not count):
            return []
        dispatches: list[dict[str, Any]] = []
        for (identifier, possess, craving) in mistaken:
            if self.trolley_sets(moniker):
                dispatches.append({destination: identifier, count: craving})
            elif not fashion:
                return []
            else:
                alternatives = [str(oo) for oo in characteristics[fashion]['enum']]
                halted = next((oo for oo in alternatives if re.search('remov|delet|drop|clear|take', oo, re.I)), alternatives[1])
                onto = next((oo for oo in alternatives if halted != oo), alternatives[0])
                dispatches += [{destination: identifier, fashion: halted, count: possess}, {destination: identifier, fashion: onto, count: craving}]
        if len(dispatches) + 3 > self.leftover():
            return []
        over_with = []
        for operands in dispatches:
            flaws: list[str] = []
            operands = _Kinds._silhouette(operands, outline, 'args', flaws)
            if flaws:
                break
            retort = self.operate(moniker, operands)
            if retort is None or self.concluded:
                break
            self.jot_trolley(moniker, operands)
            self.trolley_uttered = _Text._serialize(retort.get('observation'))
            self.watch(moniker, retort.get('observation'))
            over_with.append(_Text._serialize(operands) + ' -> ' + self.exhibit(retort.get('observation'))[:300])
        self.footprints.append({'kind': 'count_fix', 'calls': over_with})
        return over_with

    def quantity_n_mends(self) -> list[tuple[str, int, int]]:
        genre_clef = self.classification_clef()
        if not genre_clef:
            return []
        away = []
        for identifier in self.trolley_numbers():
            possess = self.withheld_pieces_n(identifier) or 0
            holders = [nn for (nn, article) in enumerate(self.items, 1) if self.holds(article, str((self.tiers.get(identifier) or {}).get(genre_clef) or ''))]
            my_own = [ii for ii in self.trolley_numbers() if (self.trolley_strands.get(ii, 1) or 0) > 0 and len(holders) == 1 and self.holds(self.items[holders[0] - 1], str((self.tiers.get(ii) or {}).get(genre_clef) or ''))]
            if 0 < possess and len(holders) == 1 and (len(my_own) == 1) and (possess != self.items[holders[0] - 1].units_n) and (not self.vended_away(identifier)):
                away.append((identifier, possess, self.items[holders[0] - 1].units_n))
        return away

    def strand_flaws(self) -> list[str]:
        genre_clef = self.classification_clef()
        serials = [ii for ii in self.trolley_numbers() if (self.withheld_pieces_n(ii) or 0) > 0]
        if not serials or not genre_clef:
            return []
        caption = {ii: str((self.tiers.get(ii) or {}).get(genre_clef) or '') for ii in serials}
        noted = {nn for (nn, article) in enumerate(self.items, 1) if any((self.holds(article, str(rr.get(genre_clef) or '')) for rr in self.tiers.values()))}
        holder = {ii: [nn for (nn, article) in enumerate(self.items, 1) if self.holds(article, caption[ii])] for ii in serials}
        flaws: list[str] = []
        for (nn, article) in enumerate(self.items, 1):
            if nn not in noted:
                continue
            strands = [ii for ii in serials if nn in holder[ii]]
            if not strands:
                flaws.append(f'item {nn} ({article.moniker}) has no line in the cart')
            elif len([ii for ii in strands if holder[ii] == [nn]]) > 1:
                flaws.append(f"item {nn} ({article.moniker}) has {len(strands)} lines ({', '.join(strands)}); keep one")
            for ii in strands:
                possess = self.withheld_pieces_n(ii)
                if holder[ii] == [nn] and article.units_n != possess:
                    flaws.append(f'{ii} is held at {possess} unit(s); item {nn} ({article.moniker}) wants {article.units_n}')
        if len(noted) == len(self.items):
            for ii in serials:
                if not holder[ii]:
                    flaws.append(f"{ii} ({caption[ii] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for ii in serials:
            if self.vended_away(ii):
                flaws.append(f'{ii} is sold out')
        costs = {ii: self.tariff_from(ii) for ii in serials}
        if isinstance(self.purse, (int, float)) and all((pp is not None for pp in costs.values())):
            if not self.digest.get('budget_scope') == 'each':
                grand_total = sum((costs[ii] * self.withheld_pieces_n(ii) for ii in serials))
                if self.purse + 1e-09 < grand_total <= 10 * self.purse:
                    flaws.append(f'the total is {grand_total:g}, over the budget of {self.purse:g}')
            else:
                atop = [ii for ii in serials if self.purse < costs[ii] <= 10 * self.purse]
                flaws += [f'{ii} costs {costs[ii]:g}, over the {self.purse:g} limit for each item' for ii in atop]
        return flaws

    def conform_design(self, curt: bool=False) -> str:
        if not self.edge:
            return ''
        index = self.conform_index()
        strands = []
        for central in sorted({mm for (_, mm) in self.edge}):
            components = [pp for (pp, mm) in self.edge if central == mm]
            reckon: dict[str, dict[int, int]] = {}
            for pp in components:
                for ii in self.article_numbers(pp):
                    if self.vended_away(ii):
                        continue
                    for token_code in self.conform_fix(ii, index):
                        reckon.setdefault(token_code, {}).setdefault(pp, 0)
                        reckon[token_code][pp] += 1
            ordered_by = sorted(reckon.items(), key=lambda pairing: (-len(pairing[1]), -sum(pairing[1].values())))[:4]
            gadgets = {token_code: [ii for ii in self.article_numbers(central) if token_code in (index['marks'].get(ii, {}).get('codes') or [])][:2] for (token_code, _) in ordered_by}
            wording = '; '.join((f'{token_code}: ' + ', '.join((f'{nn} {self.items[pp - 1].moniker}' for (pp, nn) in sorted(apiece.items()))) + (' (device not seen yet)' if not gadgets[token_code] else f" (device seen: {', '.join(gadgets[token_code])})") for (token_code, apiece) in ordered_by))
            if wording:
                strands.append(f"codes the {self.items[central - 1].moniker} candidates' parts name -> {wording}")
        if not strands:
            return ''
        return "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(strands) if not curt else ' | '.join(strands)

    def conform_flaws(self) -> list[str]:
        if not self.edge:
            return []
        index = self.conform_index()
        trolley = self.trolley_numbers()
        flaws = []
        for (piece, central) in self.edge:
            gadgets = [ii for ii in trolley if ii in self.article_numbers(central)]
            components = [ii for ii in trolley if ii in self.article_numbers(piece)]
            if len(gadgets) != 1 or not components:
                continue
            token_codes = set(index['marks'].get(gadgets[0], {}).get('codes') or [])
            if not token_codes:
                continue
            for strand in components:
                conforms = self.conform_fix(strand, index)
                if conforms and (not conforms & token_codes):
                    sound = [ii for ii in self.article_numbers(piece) if self.conform_fix(ii, index) & token_codes and (not self.vended_away(ii)) and (ii not in trolley)][:4]
                    flaws.append(f"{strand} ({self.items[piece - 1].moniker}) works with {', '.join(sorted(conforms)[:4])}, not with {gadgets[0]} ({', '.join(sorted(token_codes))})" + ('; no seen part of that kind works with it, so look for one or switch the device: ' + self.conform_design(curt=True) if not sound else f"; seen parts of that kind that work with it: {', '.join(sound)}"))
        return flaws

    def article_numbers(self, digit: int) -> list[str]:
        genre_clef = self.classification_clef()
        article = self.items[digit - 1]
        return [ii for ii in self.spotted_purchase if genre_clef and self.holds(article, str((self.tiers.get(ii) or {}).get(genre_clef) or ''))]

    def conform_fix(self, identifier: str, index: dict[str, Any]) -> set[str]:
        tag_mark = index['marks'].get(identifier) or {}
        discovered = set(tag_mark.get('works') or [])
        if tag_mark.get('own'):
            discovered |= index['notes'].get(tag_mark['own'], set())
        return discovered

    def conform_index(self) -> dict[str, Any]:
        tag_marks: dict[str, dict[str, Any]] = {}
        jottings: dict[str, set[str]] = {}
        for identifier in self.spotted_purchase:
            entry_rec = self.complete_.get(identifier) or self.tiers.get(identifier) or {}
            wording = ' '.join((str(vv) for vv in entry_rec.values() if isinstance(vv, str)))
            wording += ' ' + '. '.join((f'{kk} {vv}' for slab in entry_rec.values() if isinstance(slab, dict) for (kk, vv) in slab.items() if isinstance(vv, str)))
            tag_marks[identifier] = _Kinds._conform_tag_marks(wording)
            for (piece, token_codes) in tag_marks[identifier]['also']:
                jottings.setdefault(piece, set()).update(token_codes)
        return {'marks': tag_marks, 'notes': jottings}

class _CartView:

    def exhibit(self, sighting: Any) -> str:
        wording = _Text._serialize(_Text._shear(sighting, EXHIBIT_GLYPHS))
        if len(wording) > SIGHTING_BOUND:
            wording = wording[:SIGHTING_BOUND] + '...(cut)'
        return wording

    def jot_trolley(self, moniker: str, operands: dict[str, Any]) -> None:
        self.trolley_handled = True
        self.trolley_upon = self.pace
        details = self.outlines.get(moniker, {}).get('properties') or {}
        taking_out = False
        for (clef, worth) in operands.items():
            alternatives = (details.get(clef) or {}).get('enum')
            if isinstance(alternatives, list) and len(alternatives) >= 2 and (worth in alternatives[1:]):
                taking_out = True
        lax_quantity_n = next((_Kinds._quantity_n(vv) for vv in operands.values() if _Kinds._quantity_n(vv) is not None and (not isinstance(vv, str))), None)
        strands: list[tuple[str, int | None]] = []
        entire = False
        for (clef, worth) in operands.items():
            if isinstance(worth, list):
                silhouette = (details.get(clef) or {}).get('items') or {}
                if not worth and isinstance(silhouette, dict) and (silhouette.get('type') == 'object' or 'properties' in silhouette):
                    entire = True
                for strand in worth:
                    if isinstance(strand, dict):
                        entire = True
                        identifier = next((vv for vv in strand.values() if isinstance(vv, str) and vv in self.tiers), None)
                        count = next((_Kinds._quantity_n(vv) for vv in strand.values() if _Kinds._quantity_n(vv) is not None and (not isinstance(vv, str))), 1)
                        if identifier:
                            strands.append((identifier, count))
                    elif isinstance(strand, str) and strand in self.tiers:
                        strands.append((strand, lax_quantity_n))
            elif isinstance(worth, str) and worth in self.tiers:
                strands.append((worth, lax_quantity_n))
        if entire:
            self.trolley_strands = {ii: cc for (ii, cc) in strands if cc and 0 < cc}
            return
        groupings = self.trolley_sets(moniker)
        for (identifier, count) in strands:
            possess = self.trolley_strands.get(identifier, 0)
            if taking_out:
                possess = 0 if not count else possess - count
            elif not groupings:
                possess = possess + (1 if not count is not None else count)
            else:
                possess = 1 if not count is not None else count
            if not 0 < possess:
                self.trolley_strands.pop(identifier, None)
            else:
                self.trolley_strands[identifier] = possess

    def trolley_sets(self, moniker: str) -> bool:
        characteristics = self.outlines.get(moniker, {}).get('properties') or {}
        if any((isinstance(ss, dict) and isinstance(ss.get('enum'), list) and (len(ss['enum']) >= 2) for ss in characteristics.values())):
            return False
        return any((isinstance(ss, dict) and 'integer' in _Kinds._genera(ss) and (ss.get('minimum') == 0) for ss in characteristics.values()))

    def remaining(self, bypass: list[str], boundary: int=30) -> str:
        strands = []
        purchase = list(reversed(self.spotted_purchase))
        if self.digest.get('pick') == 'cheapest':
            purchase.sort(key=lambda ii: (ii not in self.complete_, float('inf') if not self.tariff_from(ii) is not None else self.tariff_from(ii)))
        for identifier in purchase:
            if identifier in bypass or self.vended_away(identifier):
                continue
            entry_rec = self.complete_.get(identifier) or self.tiers.get(identifier)
            strands.append(_Text._serialize(_Text._shear(entry_rec, 160))[:900])
            if len(strands) >= boundary:
                break
        return '\n'.join(strands)

    def ledger_from(self, identifier: str) -> str:
        entry_rec = self.complete_.get(identifier) or self.tiers.get(identifier)
        return '(no record seen)' if not entry_rec else _Text._serialize(_Text._shear(entry_rec, 400))[:3000]

    def trolley_numbers(self) -> list[str]:
        if self.trolley_outlook is not None and self.trolley_upon <= self.trolley_spotted_upon:
            wording = _Text._serialize(self.trolley_outlook)
            return [ii for ii in self.tiers if ii in wording]
        return list(self.trolley_strands)

    def withheld_pieces_n(self, identifier: str) -> int:
        if self.trolley_outlook is not None and self.trolley_upon <= self.trolley_spotted_upon:
            pile: list[Any] = [self.trolley_outlook]
            while pile:
                knot = pile.pop()
                if isinstance(knot, dict):
                    if identifier in [vv.strip() for vv in knot.values() if isinstance(vv, str)]:
                        quantities = [_Kinds._quantity_n(vv) for (kk, vv) in knot.items() if not isinstance(vv, str) and _Kinds._quantity_n(vv) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', _Text._vocables(kk))]
                        if len(quantities) == 1:
                            return quantities[0]
                    pile += [vv for vv in knot.values() if isinstance(vv, (dict, list))]
                elif isinstance(knot, list):
                    pile += knot
        return self.trolley_strands.get(identifier, 1)

    def trolley_wording(self) -> str:
        if self.trolley_outlook is not None and self.trolley_upon <= self.trolley_spotted_upon:
            return _Text._serialize(_Text._shear(self.trolley_outlook, 300))[:3000]
        return "lines by Kyo's record of the cart calls (listing: units): " + _Text._serialize(self.trolley_strands)

class _Talk:

    def support(self, sighting: Any) -> str:
        serials: list[str] = []

        def stroll(knot: Any, deepness: int) -> None:
            if 6 < deepness:
                return
            if isinstance(knot, dict):
                identifier = self.identifier(knot, False, deepness)
                if identifier and _Rows._datum_slab(knot) and (identifier not in serials):
                    serials.append(identifier)
                    return
                for worth in knot.values():
                    stroll(worth, deepness + 1)
            elif isinstance(knot, list):
                for worth in knot:
                    stroll(worth, deepness + 1)
        stroll(sighting, 0)
        necessities = self.necessities_presently()
        if not serials or not necessities:
            return ''
        numerous = len(self.items) > 1
        strands = []
        for identifier in serials[:12]:
            slab = _Rows._datum_slab(self.complete_.get(identifier) or {}) or {}
            components = []
            for (digit, necessity) in necessities[:8]:
                aspect = necessity.split(':', 1)[0].strip()
                finds = [f'{kk}={str(vv)[:50]}' for (kk, vv) in slab.items() if _Kinds._monikers(_Text._trunk(aspect), _Text._trunk(kk))]
                badge = '' if not numerous else f'(item {digit}) '
                if finds:
                    components.append(f'{badge}{aspect} -> ' + ', '.join(finds[:3]))
            if components:
                strands.append(f'{identifier}: ' + '; '.join(components))
        if not strands:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(strands)

    def necessities_presently(self) -> list[tuple[int, str]]:
        away = []
        for (digit, article) in enumerate(self.items, 1):
            for necessity in article.obligations + article.related_to:
                if necessity not in article.let_gone:
                    away.append((digit, necessity))
        return away

    def seize_jot(self, strands: list[str]) -> str:
        self.listened.update(strands)
        instruction = JOTTING + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.register() + '\n\nTHE NOTE:\n' + '\n'.join(strands)
        gained = self.petition_json(instruction, [NIMBLE, RESERVE], 2500) or {}
        departed = [str(gg) for gg in _Kinds._entries_of(gained.get('gone')) if isinstance(gg, (str, int))]
        self.not_offered.update((gg for gg in departed if gg in self.tiers))
        self.jot_vended = bool(departed) or gained.get('sold_out') is True
        self.jot_titled = any((gg in self.tiers for gg in departed))
        if self.jot_vended and len(strands) == 1:
            self.vended_strands.update(strands)
        if gained.get('kind') not in (None, 'change'):
            return ''
        altered = []
        purse = gained.get('budget')
        if isinstance(purse, (int, float)) and (not isinstance(purse, bool)) and (0 < purse) and (self.purse != purse):
            self.purse = purse
            altered.append(f'budget now {purse}')
        tacked_on: set[str] = set()
        releases = [ee for ee in _Kinds._entries_of(gained.get('drop')) if isinstance(ee, dict) and str(ee.get('need') or '').strip()]
        joins = [ee for ee in _Kinds._entries_of(gained.get('add')) if isinstance(ee, dict) and str(ee.get('need') or '').strip()]
        for item_rec in releases:
            article = self.items[min(max(1, _Kinds._quantity_n(item_rec.get('item')) or 1), len(self.items)) - 1]
            destination = str(item_rec['need']).strip()
            current = [mm for mm in article.obligations + article.related_to if mm not in article.let_gone]
            catch = [mm for mm in current if destination == mm]
            if not catch:
                catch = [mm for mm in current if _Text._vocables(mm.split(':')[0]) == _Text._vocables(destination.split(':')[0])][:1]
            article.let_gone.extend(catch or [destination])
            altered.append(f'dropped {destination}')
        for item_rec in joins:
            article = self.items[min(max(1, _Kinds._quantity_n(item_rec.get('item')) or 1), len(self.items)) - 1]
            necessity = str(item_rec['need']).strip()
            if necessity in article.let_gone:
                article.let_gone.remove(necessity)
            if necessity not in article.obligations + article.related_to:
                article.related_to.append(necessity)
            tacked_on.add(necessity)
            altered.append(f'added {necessity}')
        if departed:
            altered.append('unavailable: ' + ', '.join(departed))
        if gained.get('price_moved') is True:
            altered.append('a price changed')
        synopsis = str(gained.get('summary') or '').strip()
        if self.digest.get('pick') == 'cheapest' and (releases or departed or self.jot_vended or any((cc.startswith('budget now') for cc in altered))):
            self.broader_withheld = False
            self.lower_cost_tag_marks = set()
        if altered or synopsis:
            self.edition += 1
            self.latest_few = []
            self.jottings.append(synopsis or '; '.join(altered))
        return (synopsis + ('' if not altered else ' (' + '; '.join(altered) + ')')).strip()

    def lax_strands(self, sighting: Any) -> list[str]:
        strands: list[str] = []
        if isinstance(sighting, dict) and (not self.identifier(sighting)):
            for (clef, worth) in sighting.items():
                if isinstance(worth, str) and len(worth.split()) >= 5:
                    strands.append(worth)
                elif isinstance(worth, list) and worth and all((isinstance(vv, str) for vv in worth)):
                    strands.extend((vv for vv in worth if len(vv.split()) >= 5))
                elif isinstance(worth, dict) and (not self.identifier(worth, False, 1)):
                    for inside in worth.values():
                        if isinstance(inside, str) and len(inside.split()) >= 6:
                            strands.append(inside)
        return [strand for strand in strands if strand not in self.listened]

    def voiced(self, sighting: Any) -> str:
        if isinstance(sighting, str):
            return sighting
        discovered: list[str] = []

        def stroll(knot: Any, deepness: int) -> None:
            if 4 < deepness:
                return
            if isinstance(knot, str) and len(knot.split()) >= 2:
                discovered.append(knot)
            elif isinstance(knot, dict):
                for worth in knot.values():
                    stroll(worth, deepness + 1)
            elif isinstance(knot, list):
                for worth in knot:
                    stroll(worth, deepness + 1)
        stroll(sighting, 0)
        return _Text._serialize(sighting)[:400] if not discovered else ' '.join(discovered)[:800]

    def barrier_check(self, phase: str) -> tuple[bool, str]:
        paces = [pp for pp in self.procedure if pp['when'] == phase]
        if not paces or self.concluded or (not self.barrier_check_payable(phase)):
            return (True, '')
        terminating = {self.functions.get('buy'), self.functions.get('refuse')} - {None}
        utensils = [f"- {tt['function']['name']}: {tt['function'].get('description', '')} | parameters: {_Text._serialize(tt['function'].get('parameters') or {})}" for tt in self.utensils if tt['function']['name'] not in terminating]
        uttered: list[str] = []
        prodded = False
        for _ in range(BARRIER_LAPS):
            instruction = BARRIER.replace('STAGE', 'end without a purchase' if not 'order' == phase else 'place the order') + '\n\nREQUEST:\n' + self.appeal + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {pp['step']}" for pp in paces)) + '\n\nLEDGER:\n' + self.register() + '\n\nCART NOW: ' + self.trolley_wording() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{ii}. {strand}' for (ii, strand) in enumerate(self.over_with_logbook, 1))) or '(nothing yet)') + ('' if not self.jottings else '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.jottings[-4:])) + f"\n\nMESSAGES LEFT: {max(0, self.dispatch_note_boundary - self.dispatch_note_spent_on)}   STEPS LEFT: {self.leftover()} (the {('ending' if not 'order' == phase else 'order')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(utensils)
            gained = self.petition_json(instruction, [NIMBLE, WEIGHTY] if not prodded else [WEIGHTY, NIMBLE], 3500)
            if gained is None:
                break
            related_to: list[str] = []
            grip = ''
            if uttered:
                self.seize_listened({'needs': gained.get('needs')}, related_to)
                grip = str(gained.get('hold') or '').strip()
            dispatch = None if not isinstance(gained.get('call'), dict) else gained.get('call')
            self.footprints.append({'kind': 'gate', 'stage': phase, 'call': _Text._shear(dispatch, 200), 'steps': _Text._shear(gained.get('steps'), 120), 'hold': grip[:200], 'told': related_to})
            if related_to or grip:
                self.edition += 1
                return (False, ('' if not uttered else 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(uttered) + '\n') + ("The shopper's reply changes the task: " + '; '.join(related_to + ([] if not grip else [grip])) + '. Fix the cart for it, then check out again.'))
            moniker = str((dispatch or {}).get('tool') or '')
            exhibited = [ee for ee in _Kinds._entries_of(gained.get('steps')) if isinstance(ee, dict)]
            chatter = 'order' == phase and any((pp['messages'] > 0 for pp in paces))
            strands = range(1, len(self.over_with_logbook) + 1)
            attestation = {ii for ii in strands if not chatter or (ii <= len(self.over_with_paces) and self.over_with_paces[ii - 1] > self.trolley_upon)}
            unconfirmed = [ee for ee in exhibited if ee.get('done') is not True or not any(((_Kinds._quantity_n(nn) or 0) in attestation for nn in _Kinds._entries_of(ee.get('lines'))))]
            if not dispatch and unconfirmed and (not uttered) and (not prodded) and self.functions.get('ask') and (self.dispatch_note_boundary > self.dispatch_note_spent_on) and (self.leftover() >= 2):
                prodded = True
                self.footprints.append({'kind': 'gate_unproven', 'steps': _Text._shear(unconfirmed, 120)})
                self.over_with_logbook.append(f'(Kyo: no line made since the cart last changed at step {self.trolley_upon} shows ' + '; '.join((str(ee.get('step') or '') for ee in unconfirmed))[:200] + f' done, and {self.dispatch_note_boundary - self.dispatch_note_spent_on} message(s) and {self.leftover()} steps are left: give the call that does it)')
                self.over_with_paces.append(-1)
                continue
            if not dispatch or moniker not in self.outlines or moniker in terminating:
                if not dispatch:
                    self.barrier_check_fine.add((phase, self.trolley_tag_mark() + f'#{self.edition}'))
                break
            if self.leftover() < 2 or (moniker == self.functions.get('ask') and self.dispatch_note_boundary <= self.dispatch_note_spent_on):
                break
            flaws: list[str] = []
            operands = _Kinds._silhouette({} if not isinstance(dispatch.get('args'), dict) else dispatch.get('args'), self.outlines[moniker], 'args', flaws)
            if flaws:
                break
            if moniker == self.functions.get('ask'):
                wording = next((vv for vv in operands.values() if isinstance(vv, str)), '')
                lot = self.transmit([(1, '', wording)], 1)
                if not lot:
                    break
                self.barrier_check_dispatch_notes += 1
                uttered.append(f'Q: {wording}\nA: {lot[0][2]}')
                continue
            retort = self.operate(moniker, operands)
            if retort is None:
                break
            spotted = retort.get('observation')
            self.soak_up(spotted)
            self.over_with_logbook.append(f'step {self.pace}: {moniker} {_Text._serialize(operands)} -> {self.exhibit(spotted)[:300]} (cart then: {self.trolley_tag_mark()})')
            self.over_with_paces.append(self.pace)
            uttered.append(f'{moniker} {_Text._serialize(operands)} -> {self.exhibit(spotted)[:400]}')
        return (True, '' if not uttered else 'Before going on, Kyo did the steps the request asks for:\n' + '\n'.join(uttered))

    def listen(self, transmitted: list[tuple[int, str, str]], machines: tuple[str, ...]=(WEIGHTY, NIMBLE)) -> dict[str, Any]:
        instruction = LISTEN + '\n\nITEMS:\n' + '\n'.join((f'item {nn}: {that.moniker}' for (nn, that) in enumerate(self.items, 1)))
        instruction += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{ii}. (about item {nn}) Q: {qq}\nA: {aa}' for (ii, (nn, qq, aa)) in enumerate(transmitted, 1)))
        return self.petition_json(instruction, list(machines), 2500) or {}

    def seize_listened(self, listened: dict[str, Any], related_to: list[str]) -> None:
        numerous = len(self.items) > 1
        for item_rec in _Kinds._entries_of(listened.get('needs')):
            if not isinstance(item_rec, dict) or not str(item_rec.get('need') or '').strip():
                continue
            digit = min(max(1, _Kinds._quantity_n(item_rec.get('item')) or 1), len(self.items))
            necessity = str(item_rec['need']).strip()
            if necessity not in self.items[digit - 1].related_to:
                self.items[digit - 1].related_to.append(necessity)
                related_to.append(necessity if not numerous else f'item {digit}: {necessity}')
            if item_rec.get('points') is True and (digit, necessity) not in self.allusions:
                self.allusions.append((digit, necessity))
        for item_rec in _Kinds._entries_of(listened.get('drop')):
            if isinstance(item_rec, dict) and str(item_rec.get('property') or '').strip():
                article = self.items[min(max(1, _Kinds._quantity_n(item_rec.get('item')) or 1), len(self.items)) - 1]
                departed = _Text._vocables(str(item_rec['property']).split(':')[0])
                catch = [mm for mm in article.obligations + article.related_to if mm not in article.let_gone and _Text._vocables(mm.split(':')[0]) == departed]
                article.let_gone.extend(catch)
                related_to.extend((f'dropped {mm}' for mm in catch))
        for item_rec in _Kinds._entries_of(listened.get('free')):
            if isinstance(item_rec, dict) and item_rec.get('property'):
                digit = min(max(1, _Kinds._quantity_n(item_rec.get('item')) or 1), len(self.items))
                self.items[digit - 1].unbound.append(str(item_rec['property']))
        if isinstance(listened.get('budget'), (int, float)) and (not isinstance(listened.get('budget'), bool)) and (listened['budget'] > 0):
            self.purse = listened['budget']

    def transmit(self, queries: list[tuple[int, str, str]], minimum: int=6) -> list[tuple[int, str, str]]:
        utensil = self.functions['ask']
        transmitted: list[tuple[int, str, str]] = []
        for (digit, _, wording) in queries:
            if self.concluded or self.leftover() <= minimum or self.dispatch_note_boundary <= self.dispatch_note_spent_on:
                break
            operands = self.missive_operands(wording)
            if not operands:
                break
            retort = self.operate(utensil, operands)
            if retort is None:
                break
            self.dispatch_note_spent_on += 1
            rejoinder = retort.get('observation')
            vocables = self.voiced(rejoinder)
            transmitted.append((digit, wording, vocables))
            self.over_with_logbook.append(f'step {self.pace}: message to the shopper {wording!r} -> reply {vocables[:300]!r} (cart then: {self.trolley_tag_mark()})')
            self.over_with_paces.append(self.pace)
            self.footprints.append({'kind': 'ask', 'q': wording, 'a': vocables[:300]})
            if isinstance(rejoinder, dict) and rejoinder.get('error'):
                break
        return transmitted

class _Lanes:

    def specimens(self, digit: int, caption: str) -> list[str]:
        craving = _Text._vocables(caption)
        fixed_up = ' '.join(self.items[digit - 1].obligations).lower()
        for (_, moniker, _, values) in self.clef_matrix():
            if craving == moniker:
                return [vv for vv in values if vv and vv not in fixed_up and (len(vv) <= 30)][:2]
        return []

    def query(self, digit: int, caption: str, specimens: list[str] | None=None) -> str:
        wording = f'What {caption} would you like for the {self.items[digit - 1].moniker}'
        return wording + ('?' if not specimens else f', for example {specimens[0]} or {specimens[1]}?')

    def design_tracks(self, count: int) -> dict[int, list[str]]:
        matrix = self.clef_matrix()
        fixed_up = []
        for (digit, article) in enumerate(self.items, 1):
            fixed_up.append(f'item {digit}: {article.moniker} - settled by the request: ' + ('; '.join(article.obligations) or 'nothing'))
        posting = []
        latest = None
        for (index, (caption, moniker, borne, values)) in enumerate(matrix, 1):
            if latest != caption:
                posting.append(f'category {caption!r}:')
                latest = caption
            posting.append(f'  {index}. {moniker} (on {borne}): ' + ' | '.join(values) + ('' if not (len(values) <= 1 and 3 <= borne) else ' [one value on every listing read]') + self.non_voiced(caption, moniker, values) + ('' if not re.search(SNARE_ALL, moniker) else ' [several features in one: ask last]'))
        instruction = BLUEPRINT.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.appeal + '\n\nITEMS:\n' + '\n'.join(fixed_up)
        initial = [str(xx) for xx in _Kinds._entries_of(self.digest.get('ask_first')) if str(xx).strip()]
        if initial:
            instruction += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(initial)
        instruction += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(posting) or '(no full listing records yet)')
        design = self.petition_json(instruction, [WEIGHTY, NIMBLE], 3500) or {}
        tracks: dict[int, list[str]] = {nn: [] for nn in range(1, len(self.items) + 1)}
        signalled: dict[int, set[str]] = {nn: set() for nn in tracks}
        for elect in _Kinds._entries_of(design.get('ask')):
            if not isinstance(elect, dict):
                continue
            digit = min(max(1, _Kinds._quantity_n(elect.get('item') or 1) or 1), len(self.items))
            caption = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(elect.get('label') or ''))[0].strip().rstrip('?.')
            if not caption or len(caption.split()) > 5:
                continue
            if re.search(NON_TRAIT, _Text._vocables(caption)) and (not str(elect.get('hint') or '').strip()):
                continue
            if all((_Text._vocables(caption) != _Text._vocables(different) for different in tracks[digit])):
                tracks[digit].append(caption)
            if str(elect.get('hint') or '').strip():
                signalled[digit].add(_Text._vocables(caption))
                if digit not in self.design_purchase:
                    self.design_purchase.append(digit)
        for digit in tracks:
            keys = self.track_clefs(digit, matrix)
            peruse = max(1, self.ledgers_from_article(digit))
            native: dict[str, float] = {}
            for (caption, clef, borne, _) in matrix:
                if self.holder_from(caption) == digit:
                    native[clef] = max(native.get(clef, 0.0), borne / peruse)

            def lettered(caption: str) -> str:
                simple = lambda wording: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', _Text._vocables(wording)))
                return next((kk for kk in native if simple(kk) == simple(caption)), caption)

            def part_from(caption: str) -> float:
                if _Text._vocables(caption) in native:
                    return native[_Text._vocables(caption)]
                nigh = [ss for (kk, ss) in native.items() if _Kinds._monikers(_Text._trunk(caption), _Text._trunk(kk)) or _Kinds._monikers(_Text._trunk(kk), _Text._trunk(caption))]
                return 0.0 if not nigh else max(nigh)
            signalled[digit] = {_Text._vocables(lettered(hh)) for hh in signalled[digit]}
            track = []
            for elect in tracks[digit]:
                if all((_Text._vocables(lettered(elect)) != _Text._vocables(different) for different in track)):
                    track.append(lettered(elect))
            for clef in [kk for kk in keys if native.get(kk, 0.0) >= SENTRY_PART and (not re.search(SNARE_ALL, kk))][:SENTRY_CLEFS]:
                if any((_Text._vocables(clef) == _Text._vocables(different) for different in track[:3])):
                    continue
                feeble = next((ii for (ii, different) in enumerate(track[:3]) if _Text._vocables(different) not in signalled[digit] and part_from(different) + SENTRY_SPACING <= native.get(clef, 0.0)), None)
                if feeble is None and len(track) >= 3:
                    continue
                sited = [different for different in track if _Text._vocables(different) != _Text._vocables(clef)]
                if not (feeble is not None and len(self.items) > 1):
                    sited.insert(min(len(sited), len(sited) if not feeble is not None else feeble), clef)
                else:
                    migrated = sited[feeble]
                    sited[feeble] = clef
                    sited.insert(min(3, len(sited)), migrated)
                if any((_Text._vocables(xx) in signalled[digit] and all((_Text._vocables(xx) != _Text._vocables(yy) for yy in sited[:3])) for xx in track[:3])):
                    continue
                track = sited
            tracks[digit] = track
            for clef in keys:
                if all((_Text._vocables(clef) != _Text._vocables(different) for different in tracks[digit])):
                    tracks[digit].append(clef)
            tracks[digit] = [xx for xx in tracks[digit] if not re.search(SNARE_ALL, _Text._vocables(xx))] + [xx for xx in tracks[digit] if re.search(SNARE_ALL, _Text._vocables(xx))]
        self.footprints.append({'kind': 'plan', 'ask': _Text._shear(design.get('ask'), 120), 'lanes': {nn: vv[:8] for (nn, vv) in tracks.items()}})
        return tracks

    def ledgers_from_article(self, digit: int) -> int:
        genre_clef = self.classification_clef()
        article = self.items[digit - 1]
        return sum((1 for entry_rec in self.complete_.values() if genre_clef and self.holds(article, str(entry_rec.get(genre_clef) or '')) and _Rows._datum_slab(entry_rec)))

    def holds(self, article: Article, caption: str) -> bool:
        return bool(caption) and article in self.items and (self.holder_from(caption) == self.items.index(article) + 1)

    def holder_from(self, caption: str) -> int:
        if not caption:
            return 0
        vocables = _Text._vocables(caption)
        lengthy = _Text._radicals(caption)
        (finest, summit) = (0, 0.0)
        for (digit, article) in enumerate(self.items, 1):
            if vocables == _Text._vocables(article.rack or ''):
                points = 3.0
            elif not vocables == _Text._vocables(article.moniker):
                my_own = _Text._radicals(article.rack or '') | _Text._radicals(article.moniker)
                points = len(lengthy & my_own) / max(1, len(lengthy | my_own))
            else:
                points = 2.0
            if summit < points:
                (finest, summit) = (digit, points)
        return finest

    def track_clefs(self, digit: int, matrix: list[tuple[str, str, int, list[str]]]) -> list[str]:
        article = self.items[digit - 1]
        fixed_up = [_Text._trunk(mm.split(':', 1)[0]) for mm in article.obligations]
        tiers = [(caption, clef, borne, len(values)) for (caption, clef, borne, values) in matrix if self.holder_from(caption) == digit]
        gauged = {clef: sum((1 for vv in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', vv))) * 2 > len(values) for (_, clef, _, values) in matrix}
        if len(self.items) == 1 and (not tiers):
            tiers = [(caption, clef, borne, len(values)) for (caption, clef, borne, values) in matrix]
        tiers.sort(key=lambda rr: (rr[3] <= 1 and rr[2] >= 3, -rr[2], gauged.get(rr[1], False), -rr[3]))
        keys = [clef for (_, clef, _, _) in tiers if not any((_Kinds._monikers(aspect, _Text._trunk(clef)) for aspect in fixed_up)) and (not re.search(NON_TRAIT, clef))]
        return [clef for clef in keys if not re.search(SNARE_ALL, clef)] + [clef for clef in keys if re.search(SNARE_ALL, clef)]

class _Interview:

    def petition_wording(self, digit: int, captions: list[str], exhibited: bool=False) -> str:
        if len(captions) >= 2:
            return f'What {captions[0]} or {captions[1]} would you like for the {self.items[digit - 1].moniker}?'
        return self.query(digit, captions[0], None if not exhibited else self.specimens(digit, captions[0]))

    def subsequent_petition(self, tracks: dict[int, list[str]], discovered: dict[int, int], expended: dict[int, int], spent_on: dict[int, set[str]]) -> tuple[int, list[str]]:
        initial = {nn: ii for (ii, nn) in enumerate(self.design_purchase)}
        lodgers = {central for (_, central) in self.edge}
        for digit in sorted(tracks, key=lambda nn: (discovered.get(nn, 0) > 0, nn not in lodgers, expended.get(nn, 0), initial.get(nn, len(initial) + nn))):
            leftover = [caption for caption in tracks[digit] if _Text._vocables(caption) not in spent_on[digit]]
            if leftover:
                return (digit, leftover[:1])
        return (0, [])

    def worth_search(self, digits: list[int]) -> str:
        design = self.search_pattern()
        if not design or self.leftover() < 22:
            return ''
        (utensil, footing, column) = design
        spotted: list[str] = []
        purse = 8
        for digit in digits:
            values = self.worth_vocables(digit)
            article = self.items[digit - 1]
            newfound = [vv for vv in values if any((':' in tt and tt.split(':', 1)[1].strip() == vv for tt in article.related_to))]
            if not newfound or 2 >= purse:
                continue
            declared = [ii for (ii, vv) in enumerate(values) if vv not in newfound]
            let_go = 0 if not declared else declared[-1]
            tests = [values] if not len(values) > 1 else [values, values[:let_go] + values[let_go + 1:]]
            for vocables in tests[:2]:
                if not vocables:
                    continue
                flaws: list[str] = []
                operands = _Kinds._silhouette({**footing, column: ' '.join(vocables)}, self.outlines.get(utensil), 'args', flaws)
                if flaws or self.concluded:
                    break
                retort = self.operate(utensil, operands)
                purse -= 1
                if retort is None:
                    break
                sighting = retort.get('observation')
                self.watch(utensil, sighting)
                genre_clef = self.classification_clef()
                discovered = [ii for (ii, _) in self.ledgers_inside(sighting)]
                lid = None if not isinstance(self.purse, (int, float)) else self.purse
                costs = [self.tariff_from(ii) for ii in discovered]
                if lid is not None and costs and all((pp is not None and pp > 10 * lid for pp in costs)):
                    lid = None
                tiers = [ii for ii in discovered if self.obtainable(ii) and (not genre_clef or self.holds(article, str((self.tiers.get(ii) or {}).get(genre_clef) or ''))) and (lid is None or (self.tariff_from(ii) or 0) <= lid)]
                spotted.append(f'{_Text._serialize(operands.get(column))}: ' + (', '.join(tiers[:6]) or 'no row of that kind'))
                if tiers:
                    unfolded = self.stuff_particulars([ii for ii in tiers if ii not in self.complete_][:min(3, purse)], self.end_part() + 16)
                    purse -= len(unfolded)
                    if unfolded:
                        spotted.append('opened: ' + ' | '.join(unfolded)[:2400])
                    break
        if not spotted:
            return ''
        self.footprints.append({'kind': 'value_search', 'seen': [ss[:200] for ss in spotted]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(spotted)

    def worth_vocables(self, digit: int) -> list[str]:
        away = []
        for (nn, necessity) in self.necessities_presently():
            if digit != nn or necessity in [mm for (_, mm) in self.allusions] or ':' not in necessity:
                continue
            worth = necessity.split(':', 1)[1].strip()
            if not worth or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', worth, re.I):
                continue
            away.append(worth)
        return away

    def search_pattern(self) -> tuple[str, dict[str, Any], str] | None:
        utensil = self.functions.get('search')
        outline = self.outlines.get(utensil or '') or {}
        characteristics = outline.get('properties') or {}
        wordings = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'string' in _Kinds._genera(vv) and ('enum' not in vv)]
        titled = [kk for kk in wordings if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', kk, re.I)]
        for operands in reversed(self.combed):
            pillars = [kk for kk in titled or wordings if isinstance(operands.get(kk), str) and re.search('[a-z]', operands[kk], re.I)]
            if not titled:
                captions = set() if not self.classification_clef() else {str(rr.get(self.classification_clef()) or '').strip().lower() for rr in self.tiers.values()} - {''}
                pillars = [kk for kk in pillars if len({str(aa.get(kk)) for aa in self.combed if aa.get(kk)}) >= 2 and any((len(str(aa.get(kk)).split()) >= 2 for aa in self.combed)) and (not any((str(aa.get(kk) or '').strip().lower() in captions for aa in self.combed)))]
            if pillars and utensil:
                column = pillars[0]
                hold_on = {kk: vv for (kk, vv) in operands.items() if column != kk and (isinstance((characteristics.get(kk) or {}).get('enum'), list) or kk in (outline.get('required') or []))}
                genre_clef = self.classification_clef()
                racks = {_Text._vocables(article.rack or '') for article in self.items} | {_Text._vocables(article.moniker) for article in self.items}
                racks |= set() if not genre_clef else {_Text._vocables(str(tier.get(genre_clef) or '')) for tier in self.tiers.values()}
                hold_on = {kk: vv for (kk, vv) in hold_on.items() if kk in (outline.get('required') or []) or not isinstance(vv, str) or (_Text._vocables(vv) not in racks and (not any(({ww for ww in _Text._trunk(vv) if len(ww) >= 4} & {ww for ww in _Text._trunk(ss) if len(ww) >= 4} for ss in racks if ss))))}
                couched = [aa for aa in self.combed if isinstance(aa.get(column), str) and re.search('[a-z]', aa[column], re.I)]
                for kk in list(hold_on):
                    if isinstance((characteristics.get(kk) or {}).get('enum'), list):
                        spent_on = [_Text._serialize(aa[kk]) for aa in couched if kk in aa]
                        if spent_on:
                            finest = max(spent_on, key=spent_on.count)
                            hold_on[kk] = next((aa[kk] for aa in couched if kk in aa and _Text._serialize(aa[kk]) == finest))
                return (utensil, hold_on, column)
        return None

    def enquiry(self) -> str:
        self.petitioned = True
        if self.dispatch_notes_loose() <= 0:
            return ''
        self.broaden()
        tracks = self.design_tracks(self.dispatch_notes_loose())
        discovered = {nn: 0 for nn in tracks}
        expended = {nn: 0 for nn in tracks}
        spent_on: dict[int, set[str]] = {nn: set() for nn in tracks}
        trail_after: list[tuple[int, list[str], bool]] = []
        transmitted: list[tuple[int, str, str]] = []
        related_to: list[str] = []
        while self.dispatch_notes_loose() > 0 and (not self.concluded) and (self.leftover() > 6) and (self.metronome.secs() < SPAN_HALT):
            if not trail_after:
                (digit, captions) = self.subsequent_petition(tracks, discovered, expended, spent_on)
                exhibited = False
                if not captions:
                    break
            else:
                (digit, captions, exhibited) = trail_after.pop(0)
            spent_on[digit].update((_Text._vocables(caption) for caption in captions))
            lot = self.transmit([(digit, captions[0], self.petition_wording(digit, captions, exhibited))])
            if not lot:
                break
            transmitted += lot
            expended[digit] += 1
            owned = [len(article.related_to) for article in self.items]
            listened = self.listen(lot, (NIMBLE, WEIGHTY))
            self.seize_listened(listened, related_to)
            newfound = [nn for (nn, article) in enumerate(self.items, 1) if len(article.related_to) > owned[nn - 1]]
            for nn in newfound:
                discovered[nn] = discovered.get(nn, 0) + 1
            if newfound or 1 not in [_Kinds._quantity_n(xx) for xx in _Kinds._entries_of(listened.get('crowded'))]:
                continue
            front = captions[0].split()[-1]
            if len(captions[0].split()) >= 2 and _Text._vocables(front) not in spent_on[digit]:
                trail_after.insert(0, (digit, [front], False))
        discovered_items = [nn for nn in discovered if discovered[nn] > 0]
        combed = '' if not discovered_items else self.worth_search(discovered_items)
        if related_to:
            self.edition += 1
        strands = [f'Q: {qq}\nA: {aa}' for (_, qq, aa) in transmitted]
        synopsis = "Kyo's interviewer asked the shopper (each question used one step):\n" + '\n'.join(strands)
        synopsis += '\nNew firm requirements from these answers: ' + ('none' if not related_to else '; '.join(related_to))
        synopsis += '. Apply them to every pick (search again with the new values where needed).'
        indicated = [necessity for (_, necessity) in self.allusions if any((necessity in tt for tt in related_to))]
        if indicated:
            synopsis += ' These take their value from another listing: ' + ' | '.join(indicated) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if combed:
            synopsis += '\n' + combed
        if self.lined_up:
            synopsis += '\n[notes seen meanwhile] ' + ' | '.join(self.lined_up)
            self.lined_up = []
        return synopsis

    def missive_operands(self, wording: str) -> dict[str, Any] | None:
        moniker = self.functions.get('ask')
        outline = self.outlines.get(moniker or '') or {}
        characteristics = outline.get('properties') or {}
        wordings = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'string' in _Kinds._genera(vv) and ('enum' not in vv)]
        destination = next((kk for kk in outline.get('required') or [] if kk in wordings), None) or (None if not wordings else wordings[0])
        if not destination:
            return None
        operands: dict[str, Any] = {destination: wording}
        for clef in outline.get('required') or []:
            if clef not in operands:
                details = characteristics.get(clef) or {}
                if not (isinstance(details.get('enum'), list) and details['enum']):
                    return None
                else:
                    operands[clef] = details['enum'][0]
        flaws: list[str] = []
        operands = _Kinds._silhouette(operands, outline, 'args', flaws)
        return operands if not flaws else None

    def set_up_toward_petition(self) -> bool:
        if self.petitioned or not self.functions.get('ask') or self.dispatch_notes_loose() <= 0 or (self.metronome.secs() >= SPAN_WINDUP):
            return False
        if self.leftover() <= 8:
            return False
        unfolded = len(self.complete_)
        if unfolded >= 3 * len(self.items) + 1:
            return True
        return self.pace >= max(8, self.upper_paces // 4)

class _Browsing:

    def broaden_one_by_one(self) -> None:
        utensil = self.particulars_utensil()
        destination = self.singular_destination(utensil)
        if not utensil or not destination:
            return
        craved = {ww for (_, necessity) in self.necessities_presently() for ww in re.findall('[a-z0-9]{3,}', necessity.split(':', 1)[-1].lower())}
        every = 4 if not len(self.items) <= 2 else BROADEN_EVERY
        tracks: list[list[str]] = []
        prefixes: list[int] = []
        for serials in self.article_bands():
            unfolded = sum((1 for ii in serials if ii in self.complete_))
            reservoir = [ii for ii in serials if ii not in self.complete_ and self.obtainable(ii)]
            if self.digest.get('pick') == 'cheapest':
                reservoir.sort(key=lambda ii: float('inf') if not self.tariff_from(ii) is not None else self.tariff_from(ii))
            reservoir.sort(key=lambda ii: -len(craved & set(re.findall('[a-z0-9]{3,}', _Text._serialize(self.tiers.get(ii) or {}).lower()))))
            tracks.append(reservoir[:max(0, every - unfolded)])
            prefixes.append(max(0, 4 - unfolded))
        initial = [track[kk] for kk in range(4) for (track, front) in zip(tracks, prefixes) if kk < min(front, len(track))]
        extra = [track[kk] for kk in range(every) for (track, front) in zip(tracks, prefixes) if front <= kk < len(track)]
        lid = max(5, 2 * len(tracks) + 2)
        peruse = self.stuff_particulars(initial[:lid][:8], 22)
        self.stuff_particulars((initial[lid:] + extra)[:max(0, 10 - len(peruse))], 30)

    def broaden(self) -> None:
        utensil = self.functions.get('compare')
        outline = self.outlines.get(utensil or '') or {}
        characteristics = outline.get('properties') or {}
        destination = next((kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'array' in _Kinds._genera(vv)), None)
        if not utensil or not destination or any((destination != kk for kk in outline.get('required') or [])) or self.collate_naked():
            self.broaden_one_by_one()
            return
        if self.leftover() <= 12:
            return
        lid = 4 if not isinstance(characteristics[destination].get('maxItems'), int) else characteristics[destination].get('maxItems')
        ordered_by = self.article_bands()
        portion = max(1, lid // max(1, len(ordered_by)))
        elects: list[str] = []
        for serials in ordered_by:
            elects += [ii for ii in serials if ii not in self.complete_][:portion]
        elects = elects[:lid]
        if len(elects) < 2:
            return
        flaws: list[str] = []
        operands = _Kinds._silhouette({destination: elects}, outline, 'args', flaws)
        if flaws:
            return
        prior_to = len(self.complete_)
        retort = self.operate(utensil, operands)
        if retort is not None:
            spotted = retort.get('observation')
            self.watch(utensil, spotted)
            self.jot_outlook(utensil, spotted)
            tiers = self.ledgers_inside(spotted)
            if tiers:
                self.particulars_of[utensil] = self.particulars_of.get(utensil, False) or any((_Rows._datum_slab(rr) for (_, rr) in tiers))
            self.footprints.append({'kind': 'widen', 'ids': elects, 'opened': len(self.complete_) - prior_to})
            if tiers and self.particulars_of.get(utensil) is False:
                self.broaden_one_by_one()
                return
            if len(self.complete_) == prior_to and len(elects) > 2 and (self.leftover() > 12):
                flaws = []
                operands = _Kinds._silhouette({destination: elects[:2]}, outline, 'args', flaws)
                if not flaws:
                    retort = self.operate(utensil, operands)
                    if retort is not None:
                        self.watch(utensil, retort.get('observation'))

    def watch(self, moniker: str, sighting: Any) -> str:
        self.soak_up(sighting)
        strands = self.lax_strands(sighting)
        if not strands:
            return ''
        sense = self.seize_jot(strands)
        if sense:
            self.lined_up.append(f'(seen on {moniker}) {sense}')
        return sense

    def lower_cost_lines(self) -> list[str]:
        if self.digest.get('pick') != 'cheapest':
            return []
        tag_mark = self.trolley_tag_mark() + f'#{self.edition}'
        genre_clef = self.classification_clef()
        if tag_mark in self.lower_cost_tag_marks or not genre_clef:
            return []
        self.lower_cost_tag_marks.add(tag_mark)
        trolley = self.trolley_numbers()
        elects: list[str] = []
        for strand in trolley:
            tariff = self.tariff_from(strand)
            genus = str((self.tiers.get(strand) or {}).get(genre_clef) or '').strip().lower()
            if tariff is None or not genus:
                continue
            reservoir = [ii for ii in self.spotted_purchase if ii not in trolley and ii not in self.complete_ and self.obtainable(ii) and (str((self.tiers.get(ii) or {}).get(genre_clef) or '').strip().lower() == genus) and ((float('inf') if not self.tariff_from(ii) is not None else self.tariff_from(ii)) < tariff)]
            reservoir.sort(key=lambda ii: self.tariff_from(ii) or 0.0)
            elects += [ii for ii in reservoir[:2] if ii not in elects]
        return self.stuff_particulars(elects[:THRIFT_UNFOLDS], self.end_part() + 4)

    def stuff_particulars(self, serials: list[str], hold_on: int) -> list[str]:
        utensil = self.particulars_utensil()
        destination = self.singular_destination(utensil)
        if not utensil or not destination:
            return []
        wordings = []
        for identifier in [ii for ii in serials if ii in self.tiers and ii not in self.complete_ and (not self.vended_away(ii))]:
            if self.leftover() <= hold_on or self.concluded or self.particulars_of.get(utensil) is False:
                break
            flaws: list[str] = []
            operands = _Kinds._silhouette({destination: identifier}, self.outlines.get(utensil), 'args', flaws)
            if flaws:
                break
            retort = self.operate(utensil, operands)
            if retort is None:
                break
            spotted = retort.get('observation')
            self.watch(utensil, spotted)
            self.jot_outlook(utensil, spotted)
            if isinstance(spotted, dict) and self.identifier(spotted):
                self.particulars_of[utensil] = self.particulars_of.get(utensil, False) or bool(_Rows._datum_slab(spotted))
            wordings.append(f'{identifier}: ' + self.exhibit(spotted)[:1500])
        if wordings:
            self.footprints.append({'kind': 'fill_facts', 'n': len(wordings)})
        return wordings

    def broader_grip(self) -> str:
        if self.broader_withheld or self.digest.get('pick') != 'cheapest' or self.leftover() < 14 + 2 * len(self.items):
            return ''
        self.broader_withheld = True
        strands = [f'{ii} at {self.tariff_from(ii):g}' for ii in self.trolley_numbers() if self.tariff_from(ii) is not None]
        return f"Not run yet (no step used). {self.leftover()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + ('' if not strands else f" ({'; '.join(strands)})") + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def thriftiest_hereabouts(self, sightings: list[Any]) -> str:
        if self.digest.get('pick') != 'cheapest' or not self.tariff_clef:
            return ''
        discovered = {ii for spotted in sightings for (ii, _) in self.ledgers_inside(spotted)}
        tiers = [(self.tariff_from(ii), ii) for ii in discovered if self.obtainable(ii) and self.tariff_from(ii) is not None]
        if len(tiers) < 2:
            return ''
        return '\n[Kyo] buyable rows here, cheapest first: ' + ', '.join((f'{ii} ({pp:g})' for (pp, ii) in sorted(tiers)[:6]))

    def lid_overlooked(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        characteristics = (self.outlines.get(moniker) or {}).get('properties') or {}

        def costed(clef: str) -> bool:
            details = {} if not isinstance(characteristics.get(clef), dict) else characteristics.get(clef)
            return bool(re.search('price|cost|amount|budget|spend|money', clef + ' ' + str(details.get('description') or ''), re.I))
        lids = [kk for (kk, vv) in operands.items() if kk in characteristics and isinstance(vv, (int, float)) and (not isinstance(vv, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', kk, re.I) and costed(kk)]
        if not lids or not self.tariff_clef:
            return ''
        lids.sort(key=lambda kk: operands[kk])
        lid = operands[lids[0]]
        atop = [ii for (ii, _) in self.ledgers_inside(sighting) if (self.tariff_from(ii) or 0) > lid]
        if not atop:
            return ''
        fashions = {kk: vv for (kk, vv) in operands.items() if isinstance(vv, str) and isinstance((characteristics.get(kk) or {}).get('enum'), list)}
        remaining = {kk: [oo for oo in characteristics[kk]['enum'] if vv != oo] for (kk, vv) in fashions.items()}
        signal = '' if not any(remaining.values()) else '; list the category with ' + ', '.join((f'{kk}={_Text._serialize(oo[0])}' for (kk, oo) in remaining.items() if oo)) + f' and the same {lids[0]} to see only cheaper listings'
        return f'\n[Kyo] This search did not apply {lids[0]}={lid:g}: {len(atop)} of its rows cost more{signal}.'

    def extra_folios(self, moniker: str, operands: dict[str, Any], sighting: Any) -> str:
        self.leafed_observation = []
        elect = self.digest.get('pick')
        design = None if not elect in ('cheapest', 'most', 'least') else self.leafing(moniker, sighting)
        if not design:
            return ''
        (count, exhibited, parameter, worth) = design
        granted = operands.get(parameter)
        afterward = bool(str(granted or '').strip()) if not (isinstance(granted, int) and (not isinstance(granted, bool))) else granted not in (None, 0, 1)
        if worth is None or afterward or exhibited >= count:
            return ''
        if 'cheapest' == elect and self.tariff_arranged(sighting):
            return ''
        folios = -(-(count - exhibited) // exhibited)
        if 2 < folios and self.folios_peruse + 2 <= SELF_FOLIOS and (self.leftover() - self.end_part() - 2 >= 12):
            folios = 2
        if 2 < folios or self.folios_peruse + folios > SELF_FOLIOS or self.leftover() - self.end_part() - folios < 12:
            return f"\n[Kyo] This search matched {count} listings and showed {exhibited}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {parameter}={_Text._serialize(worth)}."
        wordings = []
        for _ in range(folios):
            flaws: list[str] = []
            dispatch = _Kinds._silhouette({**operands, parameter: worth}, self.outlines.get(moniker), 'args', flaws)
            if flaws:
                break
            retort = self.operate(moniker, dispatch)
            if retort is None or self.concluded:
                break
            self.folios_peruse += 1
            spotted = retort.get('observation')
            self.watch(moniker, spotted)
            self.leafed_observation.append(spotted)
            exhibited_wording = self.exhibit(spotted)
            self.peruse_hoard[moniker + _Text._serialize(dispatch)] = (self.edition, self.pace, exhibited_wording)
            wordings.append(f'{parameter}={_Text._serialize(worth)}: {exhibited_wording}')
            following = self.leafing(moniker, spotted)
            worth = None if not following else following[3]
            if worth is None:
                break
        self.footprints.append({'kind': 'pages', 'tool': moniker, 'read': len(wordings)})
        balance = '' if not (worth is not None and wordings) else f"\n[Kyo] {count} listings matched and Kyo has read {exhibited * (len(wordings) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest."
        return '' if not wordings else '\n[Kyo read the next pages of this search itself, one step a page]\n' + '\n'.join(wordings) + balance

    def tariff_arranged(self, sighting: Any) -> bool:
        costs = [self.tariff_from(ii) for (ii, _) in self.ledgers_inside(sighting)]
        return len(costs) >= 3 and None not in costs and all((bb >= aa for (aa, bb) in zip(costs, costs[1:])))

    def leafing(self, moniker: str, sighting: Any) -> tuple[int, int, str, Any] | None:
        silhouette = self.peruse_silhouette(moniker, sighting)
        if not silhouette or not silhouette['count'] or (not silhouette['next']) or (not silhouette['next_arg']) or (not isinstance(sighting, dict)):
            return None
        (tiers, count, worth) = (sighting.get(silhouette['rows']), sighting.get(silhouette['count']), sighting.get(silhouette['next']))
        if not isinstance(tiers, list) or not tiers or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if worth is not None and (isinstance(worth, bool) or not isinstance(worth, (str, int))):
            return None
        return (count, len(tiers), silhouette['next_arg'], worth)

    def trolley_tag_mark(self) -> str:
        return _Text._serialize(sorted(self.trolley_strands.items()))

    def collate_naked(self) -> bool:
        spotted = self.collate_particulars()
        return spotted is False or (spotted is None and (self.particulars_of.get(self.particulars_utensil() or '') is True or bool(self.functions.get('offer'))))

    def collate_particulars(self) -> bool | None:
        return self.particulars_of.get(self.functions.get('compare') or '')

    def particulars_utensil(self) -> str | None:
        noted = next((tt for (tt, fine) in self.particulars_of.items() if fine and self.singular_destination(tt)), None)
        if noted:
            return noted
        utensil = self.functions.get('open')
        return None if not (self.particulars_of.get(utensil or '') is not False and self.singular_destination(utensil)) else utensil

    def folio_utensils(self) -> set[str]:
        folio = self.functions.get('offer') or self.functions.get('open')
        return {tt for tt in (folio, self.functions.get('compare')) if tt}

class _Records:

    def peruse_silhouette(self, moniker: str, sighting: Any) -> dict[str, Any] | None:
        if moniker in self.silhouettes:
            return self.silhouettes[moniker]
        if not isinstance(sighting, dict):
            return None
        sequences = [kk for (kk, vv) in sighting.items() if isinstance(vv, list) and len(vv) >= 2 and all((isinstance(rr, dict) for rr in vv))]
        if not sequences:
            return None
        self.silhouettes[moniker] = None
        sample = {kk: vv if not kk in sequences else vv[:3] for (kk, vv) in sighting.items()}
        instruction = SILHOUETTE + '\n\nTOOL: ' + moniker + ' | parameters: ' + _Text._serialize(self.outlines.get(moniker) or {}) + '\n\nRESULT (first rows only):\n' + _Text._serialize(_Text._shear(sample, 120))[:4000]
        gained = self.petition_json(instruction, [NIMBLE, WEIGHTY], 2500) or {}
        tiers = None if not gained.get('rows') in sequences else gained.get('rows')
        if not tiers:
            return None
        initial = sighting[tiers][0]
        characteristics = (self.outlines.get(moniker) or {}).get('properties') or {}

        def titled(clef: Any, whereat: dict[str, Any]) -> str | None:
            return None if not (isinstance(clef, str) and clef in whereat) else clef
        silhouette = {'rows': tiers, 'count': titled(gained.get('count'), sighting), 'next': titled(gained.get('next'), sighting), 'next_arg': titled(gained.get('next_arg'), characteristics), 'price': titled(gained.get('price'), initial), 'stock': titled(gained.get('stock'), initial)}
        if silhouette['count'] and (isinstance(sighting[silhouette['count']], bool) or not isinstance(sighting[silhouette['count']], int)):
            silhouette['count'] = None
        self.silhouettes[moniker] = silhouette
        self.tariff_clef = self.tariff_clef or silhouette['price']
        self.inventory_clef = self.inventory_clef or silhouette['stock']
        self.footprints.append({'kind': 'shape', 'tool': moniker, 'shape': silhouette})
        return silhouette

    def exhibited_previously(self, moniker: str, identifier: str) -> bool:
        (edition, possess) = self.outlook_clefs.get(identifier, (-1, set()))
        keys = self.utensil_clefs.get(moniker)
        return bool(keys) and self.edition == edition and (possess >= keys) and (identifier not in self.departed)

    def jot_outlook(self, moniker: str, sighting: Any) -> None:
        for (identifier, entry_rec) in self.ledgers_inside(sighting):
            keys = set(entry_rec)
            self.utensil_clefs.setdefault(moniker, set()).update(keys)
            (edition, possess) = self.outlook_clefs.get(identifier, (-1, set()))
            self.outlook_clefs[identifier] = (self.edition, (set() if not self.edition == edition else possess) | keys)

    def ledgers_inside(self, sighting: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(sighting, dict):
            identifier = self.identifier(sighting)
            if identifier:
                return [(identifier, sighting)]
            discovered: list[tuple[str, dict[str, Any]]] = []
            for worth in sighting.values():
                if isinstance(worth, list):
                    discovered += [(ii, rr) for rr in worth if isinstance(rr, dict) for ii in [self.identifier(rr, True)] if ii]
                elif isinstance(worth, dict):
                    discovered += [(ii, rr) for vv in worth.values() if isinstance(vv, list) for rr in vv if isinstance(rr, dict) for ii in [self.identifier(rr, True)] if ii]
            return discovered
        if isinstance(sighting, list):
            return [(ii, rr) for rr in sighting if isinstance(rr, dict) for ii in [self.identifier(rr, True)] if ii]
        return []

    def obtainable(self, identifier: str) -> bool:
        if self.vended_away(identifier):
            return False
        worth = None if not self.inventory_clef else (self.tiers.get(identifier) or {}).get(self.inventory_clef)
        return not (worth is False or (isinstance(worth, (int, float)) and (not isinstance(worth, bool)) and (0 >= worth)))

    def tariff_from(self, identifier: str) -> float | None:
        worth = None if not self.tariff_clef else (self.tiers.get(identifier) or {}).get(self.tariff_clef)
        if isinstance(worth, dict):
            worth = next((vv for vv in worth.values() if isinstance(vv, (int, float)) and (not isinstance(vv, bool))), None)
        if isinstance(worth, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', worth):
            worth = float(worth)
        return None if not (isinstance(worth, (int, float)) and (not isinstance(worth, bool))) else float(worth)

class _Ledger:

    def singular_destination(self, utensil: str | None) -> str | None:
        outline = self.outlines.get(utensil or '') or {}
        characteristics = outline.get('properties') or {}
        wordings = [kk for (kk, vv) in characteristics.items() if isinstance(vv, dict) and 'string' in _Kinds._genera(vv) and ('enum' not in vv)]
        destination = next((kk for kk in outline.get('required') or [] if kk in wordings), None) or (None if not len(wordings) == 1 else wordings[0])
        if not utensil or not destination or any((destination != kk for kk in outline.get('required') or [])):
            return None
        return destination

    def article_bands(self) -> list[list[str]]:
        genre_clef = self.classification_clef()
        groups: dict[str, list[str]] = {}
        for identifier in self.spotted_purchase:
            caption = '' if not genre_clef else str(self.tiers[identifier].get(genre_clef) or '')
            groups.setdefault(caption, []).append(identifier)
        craved = [(_Text._vocables(article.rack or ''), _Text._vocables(article.moniker), {ww for ww in _Text._trunk(article.rack or '') | _Text._trunk(article.moniker) if len(ww) >= 4}) for article in self.items]
        paired_up = [serials for (caption, serials) in groups.items() if any((_Text._vocables(caption) in (rack, moniker) or {ww for ww in _Text._trunk(caption) if len(ww) >= 4} & lengthy for (rack, moniker, lengthy) in craved)) or self.holder_from(caption)]
        reservoir = paired_up or list(groups.values())
        return sorted(reservoir, key=lambda serials: -len(serials))[:max(1, len(self.items))]

    @staticmethod
    def doubles(moniker: str, details: dict[str, Any], different: str, berth: dict[str, Any]) -> bool:
        (aa, bb) = (set(moniker.split()), set(different.split()))
        if not aa or not bb or (not aa & bb):
            return False

        def concur(xx: str, yy: str) -> bool:
            (wxa, wya) = (re.findall('[a-z0-9]+', xx.lower()), re.findall('[a-z0-9]+', yy.lower()))
            (curt, lengthy) = (wya, wxa) if not len(wxa) <= len(wya) else (wxa, wya)
            return yy == xx or (bool(curt) and set(curt) <= set(lengthy) and (len(lengthy) <= 2 * len(curt) + 2))
        pair_of = [ii for ii in details['per'] if ii in berth['per'] and details['per'][ii].strip() and berth['per'][ii].strip()]
        identical = sum((1 for ii in pair_of if concur(details['per'][ii], berth['per'][ii])))
        if not (bb >= aa or aa >= bb):
            return len(pair_of) >= 2 and identical == len(pair_of)
        if pair_of:
            return identical * 2 > len(pair_of)
        return bool((set(details['values']) & set(berth['values'])) - {''})

    def non_voiced(self, caption: str, moniker: str, values: list[str]) -> str:
        digit = self.holder_from(caption)
        if not digit:
            return ''
        for obligation in self.items[digit - 1].obligations:
            (aspect, _, worth) = obligation.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', worth, re.I):
                continue
            vocables = set(re.findall('[a-z0-9]{3,}', worth.lower()))
            exhibited = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if aspect.strip() and _Text._vocables(aspect) != moniker and _Text._trunk(aspect) & _Text._trunk(moniker) and vocables and (not exhibited >= vocables):
                return f' [not the stated {aspect.strip()}: another property]'
        return ''

    def clef_matrix(self) -> list[tuple[str, str, int, list[str]]]:
        genre_clef = self.classification_clef()
        groups: dict[str, dict[str, Any]] = {}
        for identifier in self.spotted_purchase:
            entry_rec = self.complete_.get(identifier)
            if not entry_rec:
                continue
            slab = _Rows._datum_slab(entry_rec) or {}
            caption = 'listings' if not genre_clef else str(entry_rec.get(genre_clef) or 'listings')
            group = groups.setdefault(caption, {'n': 0, 'keys': {}})
            group['n'] += 1
            over_with: set[str] = set()
            for (clef, worth) in slab.items():
                moniker = _Text._vocables(clef)
                if not moniker or moniker in over_with:
                    continue
                over_with.add(moniker)
                berth = group['keys'].setdefault(moniker, {'n': 0, 'values': [], 'per': {}})
                berth['n'] += 1
                wording = str(worth).strip().lower()
                berth['per'][identifier] = wording
                if wording not in berth['values']:
                    berth['values'].append(wording)
        craved = [(_Text._vocables(article.rack or ''), _Text._vocables(article.moniker), {ww for ww in _Text._trunk(article.rack or '') | _Text._trunk(article.moniker) if len(ww) >= 4}) for article in self.items]
        paired_up = {caption for caption in groups if any((_Text._vocables(caption) in (rack, moniker) or {ww for ww in _Text._trunk(caption) if len(ww) >= 4} & lengthy for (rack, moniker, lengthy) in craved)) or self.holder_from(caption)}
        if paired_up:
            groups = {caption: group for (caption, group) in groups.items() if caption in paired_up}
        matrix = []
        for (caption, group) in sorted(groups.items(), key=lambda pairing: -pairing[1]['n']):
            held_on: list[tuple[str, dict[str, Any]]] = []
            for (moniker, berth) in sorted(group['keys'].items(), key=lambda pairing: (-pairing[1]['n'], len(pairing[0]))):
                values = list(berth['per'].values())
                if not values:
                    continue
                signify = sum((len(vv) for vv in values)) / len(values)
                if 40 < signify or sum((1 for vv in values if ' / ' in vv)) * 2 >= len(values):
                    continue
                repeat = sum((len(set(re.findall('[a-z]{3,}', vv)) & set(re.findall('[a-z]{3,}', ' '.join((oo for oo in values if oo is not vv))))) > 0 for vv in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (12 < signify) and (repeat * 2 < len(values)):
                    continue
                double = next((details for (different, details) in held_on if self.doubles(different, details, moniker, berth)), None)
                if double is not None:
                    for (identifier, wording) in berth['per'].items():
                        double['per'].setdefault(identifier, wording)
                        if wording not in double['values']:
                            double['values'].append(wording)
                    double['n'] = len(double['per'])
                    continue
                held_on.append((moniker, berth))
            held_on.sort(key=lambda pairing: -pairing[1]['n'])
            for (moniker, berth) in held_on[:14]:
                matrix.append((caption, moniker, berth['n'], [vv[:40] for vv in berth['values'][:4]]))
        return matrix

    def classification_clef(self) -> str | None:
        quantities: dict[str, set[str]] = {}
        at_hand: dict[str, int] = {}
        for tier in self.tiers.values():
            for (clef, worth) in tier.items():
                if isinstance(worth, str) and 0 < len(worth) <= 40 and (not re.search('\\d{3}', worth)):
                    quantities.setdefault(clef, set()).add(worth)
                    at_hand[clef] = at_hand.get(clef, 0) + 1
        grand_total = max(1, len(self.tiers))
        noted = self.captions()
        finest = None
        for (clef, values) in quantities.items():
            if at_hand[clef] < 0.8 * grand_total or len(values) > 40:
                continue
            enumerated = sum((1 for vv in values if vv.strip().lower() in noted)) / len(values)
            signify = sum((len(vv) for vv in values)) / len(values)
            points = (-enumerated, len(values), -at_hand[clef], -signify)
            if finest is None or points < finest[0]:
                finest = (points, clef)
        return None if not finest else finest[1]

    def captions(self) -> set[str]:
        discovered: set[str] = set()

        def stroll(outline: Any, deepness: int) -> None:
            if 6 < deepness or not isinstance(outline, dict):
                return
            for alternative in outline.get('enum') or []:
                if isinstance(alternative, str):
                    discovered.add(alternative.strip().lower())
            for worth in outline.values():
                if isinstance(worth, dict):
                    stroll(worth, deepness + 1)
        for outline in self.outlines.values():
            stroll(outline, 0)
        return discovered

    def soak_up(self, sighting: Any) -> None:

        def stroll(knot: Any, deepness: int, inside_sequence: bool=False) -> None:
            if 6 < deepness:
                return
            if isinstance(knot, dict):
                identifier = self.identifier(knot, inside_sequence, deepness)
                if identifier:
                    if identifier not in self.tiers:
                        self.spotted_purchase.append(identifier)
                        self.tiers[identifier] = {}
                    earlier = self.tiers[identifier]
                    if any((earlier.get(kk) is True and vv is False for (kk, vv) in knot.items())):
                        self.not_offered.add(identifier)
                    elif any((earlier.get(kk) is False and vv is True for (kk, vv) in knot.items())):
                        self.not_offered.discard(identifier)
                    self.tiers[identifier].update(knot)
                    if _Rows._datum_slab(knot):
                        self.complete_[identifier] = dict(self.tiers[identifier])
                    elif identifier in self.complete_:
                        self.complete_[identifier].update(knot)
                    return
                for worth in knot.values():
                    stroll(worth, deepness + 1)
            elif isinstance(knot, list):
                tiers = self.acquire_id_lines(knot)
                for worth in knot:
                    stroll(worth, deepness + 1, tiers)
        stroll(sighting, 0)

    def acquire_id_lines(self, knot: list[Any]) -> bool:
        tiers = [nn for nn in knot if isinstance(nn, dict) and len(nn) >= 3]
        if not tiers or len(tiers) < len(knot) / 2:
            return False
        if self.ident_clef and all((_Rows._ident_resembling(rr.get(self.ident_clef), True) for rr in tiers)):
            return True
        discovered = _Rows._acquire_id_ident(tiers)
        if discovered:
            self.ident_clef = discovered
            return True
        return False

    def identifier(self, knot: Any, inside_sequence: bool=False, deepness: int=0) -> str | None:
        if not isinstance(knot, dict) or len(knot) < 3:
            return None
        if self.ident_clef and _Rows._ident_resembling(knot.get(self.ident_clef), True):
            return knot[self.ident_clef].strip()
        embedded = any((isinstance(vv, dict) for vv in knot.values()))
        if not (inside_sequence or 0 == deepness or embedded):
            return None
        discovered = _Rows._acquire_id_ident([knot])
        if discovered and ('id' in str(discovered).lower() or knot[discovered].strip() in self.tiers):
            return knot[discovered].strip()
        return None

    def standing(self) -> str:
        leftover = self.leftover()
        wording = f'[Kyo] step {self.pace} of {self.upper_paces} used, {leftover} left.'
        if self.metronome.secs() >= SPAN_WINDUP and (not self.concluded):
            wording += ' TIME IS NEARLY UP: within your next two or three calls set the cart to your best pick for every item and check out; end without a purchase only when your search showed that an item has no fitting listing.'
        elif leftover <= self.end_part() + 1:
            wording += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif 6 >= leftover or (leftover <= len(self.items) + 5 and 0 == self.acquires):
            wording += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif 0 == self.acquires and 12 >= leftover:
            wording += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif 0 == self.acquires and (not self.trolley_handled) and (self.pace >= self.upper_paces // 2):
            wording += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        current = {necessity for (_, necessity) in self.necessities_presently()}
        retrievals = [necessity for (_, necessity) in self.allusions if necessity in current and necessity not in self.allusion_worths]
        if retrievals and 0 == self.acquires:
            wording += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(retrievals[:3])
        if self.clues:
            wording += '\n[review candidates] ' + ', '.join(self.clues)
        design = '' if not (self.edge and 0 == self.acquires) else self.conform_design()
        if design:
            wording += '\n' + design
        if self.jottings:
            wording += '\n[changes so far] ' + ' | '.join(self.jottings[-4:])
        wording += '\n[ledger]\n' + self.register()
        return wording

    def register(self) -> str:
        components = []
        numerous = len(self.items) > 1
        for (digit, article) in enumerate(self.items, 1):
            current = [mm for mm in article.obligations + article.related_to if mm not in article.let_gone]
            related_to = [mm for mm in current if self.depicted(digit, mm)]
            necessities = [mm + ('' if not mm in self.allusion_worths else f' (the page it points at gives: {self.allusion_worths[mm]})') for mm in current if mm not in related_to]
            front = f'item: {article.moniker}' if not numerous else f'item {digit}: {article.moniker}'
            if article.rack:
                front += f' (store label: {article.rack})'
            front += f', quantity {article.units_n}'
            wording = front + '; requirements: ' + ('none beyond the kind of product' if not necessities else '; '.join(necessities))
            if related_to:
                wording += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(related_to)
            if article.unbound:
                wording += '; shopper does not mind: ' + ', '.join(article.unbound)
            components.append(wording)
        digest = self.digest
        if self.purse is not None:
            ambit = 'in total' if not digest.get('budget_scope') == 'each' else 'for each item'
            components.append(f"budget: at most {self.purse} {digest.get('budget_unit') or ''} {ambit}".replace('  ', ' '))
        elect = digest.get('pick')
        if 'cheapest' == elect:
            components.append('pick: the cheapest listing that fits (lowest price)')
        elif not elect in ('most', 'least'):
            components.append('pick: any listing that fits everything')
        else:
            components.append(f"pick: the {elect} {digest.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        if digest.get('in_stock_only', True) is not False:
            components.append('in stock only')
        for desire in _Kinds._entries_of(digest.get('wishes')):
            components.append(f'wish (tie-breaker only): {desire}')
        for canon in _Kinds._entries_of(digest.get('rules')):
            components.append(f'rule from the request: {canon}')
        if digest.get('quiet_decline') is True:
            components.append('if nothing fits: end without naming or suggesting any listing')
        for item_rec in self.procedure:
            whenever = {'choose': 'before choosing', 'order': 'at the order (Kyo does it when you check out)', 'end': 'at an ending without a purchase (Kyo does it then)'}[item_rec['when']]
            components.append(f"step the request asks for, {whenever}: {item_rec['step']}")
        particulars = self.particulars_utensil()
        if particulars and (self.particulars_of.get(particulars) is True or self.functions.get('offer')):
            naked = sorted((tt for tt in self.folio_utensils() | {self.functions.get('offer') or ''} if tt and self.particulars_of.get(tt) is False))
            folios = sorted((tt for tt in self.folio_utensils() if particulars != tt))
            components.append(f'listing attributes come from {particulars}' + ('' if not naked else f"; {', '.join(naked)} show none") + ('' if not folios else f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(folios)}"))
        return '\n'.join(('- ' + pp for pp in components))

class _Intake:

    def depicted(self, digit: int, necessity: str) -> bool:
        if necessity in [mm for (_, mm) in self.allusions] or necessity in [str(xx) for xx in _Kinds._entries_of(self.digest.get('lookups'))]:
            return False
        (aspect, _, worth) = necessity.partition(':')
        if not aspect.strip() or len(re.findall('[a-z0-9]+', worth.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', worth, re.I):
            return False
        article = self.items[digit - 1]
        if any(({ww for ww in _Text._trunk(different.moniker) if len(ww) >= 4} & _Text._trunk(worth) for different in self.items if different is not article)):
            return False
        genre_clef = self.classification_clef()
        keys: set[str] = set()
        peruse = 0
        for entry_rec in self.complete_.values():
            if genre_clef and (not self.holds(article, str(entry_rec.get(genre_clef) or ''))):
                continue
            slab = _Rows._datum_slab(entry_rec) or {}
            if slab:
                peruse += 1
                keys |= {_Text._vocables(kk) for kk in slab}
        return 3 <= peruse and (not any((_Text._trunk(kk) & _Text._trunk(aspect) for kk in keys)))

    def untangle(self) -> None:
        for (digit, article) in enumerate(self.items, 1):
            indicators = [necessity for (nn, necessity) in self.allusions if digit == nn and necessity in article.obligations]
            balance = [mm for mm in article.obligations if mm not in indicators]
            if not indicators or not balance:
                continue
            instruction = UNTANGLE + '\n\nREQUEST:\n' + self.appeal + f'\n\nPRODUCT: {article.moniker}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(indicators) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(balance)
            gained = self.petition_json(instruction, [WEIGHTY, NIMBLE], 1500) or {}
            migrated = [mm for mm in balance if mm in [str(xx).strip() for xx in _Kinds._entries_of(gained.get('describe_other'))]]
            if not migrated:
                continue
            article.obligations = [mm for mm in article.obligations if mm not in migrated]
            fused_ = indicators[0] + ' (that listing: ' + '; '.join(migrated) + ')'
            article.obligations = [mm if not mm == indicators[0] else fused_ for mm in article.obligations]
            self.allusions = [(nn, necessity if not (nn, necessity) == (digit, indicators[0]) else fused_) for (nn, necessity) in self.allusions]
            self.footprints.append({'kind': 'unmix', 'item': digit, 'moved': migrated})

    @staticmethod
    def common_rack(digest: dict[str, Any]) -> bool:
        racks = [_Text._vocables(str(ee.get('shelf') or '')) for ee in _Kinds._entries_of(digest.get('items')) if isinstance(ee, dict)]
        racks = [ss for ss in racks if ss]
        return len(racks) != len(set(racks))

    def peruse_appeal(self) -> None:
        strands = []
        for utensil in self.utensils:
            fun = utensil['function']
            strands.append(f"- {fun['name']}: {fun.get('description', '')} | parameters: {_Text._serialize(fun.get('parameters') or {})}")
        instruction = DIGEST + '\nTOOLS:\n' + '\n'.join(strands) + '\n\nREQUEST:\n' + self.appeal
        if self.outlook.get('opening') is not None:
            instruction += '\n\nOPENING NOTE FROM THE STORE:\n' + _Text._serialize(_Text._shear(self.outlook.get('opening'), 1500))
        digest = self.petition_json(instruction, [NIMBLE], 4000) or {}
        if not self.zero_allowance and (not _Kinds._entries_of(digest.get('items')) or not isinstance(digest.get('roles'), dict) or (not digest['roles'].get('buy'))):
            digest = self.petition_json(instruction, [WEIGHTY, NIMBLE], 2500) or digest
        elif not self.zero_allowance and self.common_rack(digest):
            once_more = self.petition_json(instruction, [WEIGHTY], 2500) or {}
            if _Kinds._entries_of(once_more.get('items')) and isinstance(once_more.get('roles'), dict) and once_more['roles'].get('buy'):
                digest = once_more
        self.digest = digest
        functions = {} if not isinstance(digest.get('roles'), dict) else digest.get('roles')
        self.functions = {kk: vv for (kk, vv) in functions.items() if isinstance(vv, str) and vv in self.outlines}
        for clef in ('buy_args', 'refuse_args'):
            if not isinstance(digest.get(clef), dict):
                digest[clef] = {}
        (acquire, spurn) = (self.functions.get('buy'), self.functions.get('refuse'))
        if acquire and spurn == acquire and (not digest['buy_args'] or not digest['refuse_args'] or digest['buy_args'] == digest['refuse_args']):
            petition = f'The store tool {acquire} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _Text._serialize(self.outlines.get(acquire)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            stable = self.petition_json(petition, [NIMBLE, WEIGHTY], 1500) or {}
            if isinstance(stable.get('buy_args'), dict) and isinstance(stable.get('refuse_args'), dict) and (stable['buy_args'] != stable['refuse_args']):
                (digest['buy_args'], digest['refuse_args']) = (stable['buy_args'], stable['refuse_args'])
        for item_rec in _Kinds._entries_of(digest.get('items')):
            if not isinstance(item_rec, dict):
                continue
            headcount = _Kinds._quantity_n(item_rec.get('quantity'))
            count = 1 if not (headcount and 0 < headcount) else headcount
            obligations = [str(mm) for mm in _Kinds._entries_of(item_rec.get('musts')) if mm]
            self.items.append(Article(str(item_rec.get('name') or 'the product'), str(item_rec.get('shelf') or ''), count, obligations))
        blended: list[Article] = []
        turned_into: dict[int, int] = {}
        for (earlier, article) in enumerate(self.items, 1):
            double = next((mm for mm in blended if article.rack.strip() and _Text._vocables(mm.rack) == _Text._vocables(article.rack) and (not any((_Text._vocables(aa.split(':')[0]) == _Text._vocables(bb.split(':')[0]) and _Text._vocables(aa.partition(':')[2]) != _Text._vocables(bb.partition(':')[2]) for aa in mm.obligations for bb in article.obligations)))), None)
            if not double is None:
                turned_into[earlier] = blended.index(double) + 1
                double.obligations += [mm for mm in article.obligations if mm not in double.obligations]
                self.footprints.append({'kind': 'items_merged', 'into': double.moniker, 'dropped': article.moniker})
            else:
                blended.append(article)
                turned_into[earlier] = len(blended)
        self.items = blended
        if not self.items:
            self.items = [Article('the requested product', '', 1, [])]
        for duo in _Kinds._entries_of(digest.get('fits')):
            if isinstance(duo, dict):
                (piece, central) = (turned_into.get(_Kinds._quantity_n(duo.get('item')) or 0), turned_into.get(_Kinds._quantity_n(duo.get('with')) or 0))
                if piece and central and (central != piece) and (1 <= piece <= len(self.items)) and (1 <= central <= len(self.items)) and ((piece, central) not in self.edge):
                    self.edge.append((piece, central))
        for retrieve in _Kinds._entries_of(digest.get('lookups')):
            wording = str(retrieve).strip()
            if not any((wording in article.obligations for article in self.items)):
                clef = _Text._vocables(wording.split(':')[0])
                wording = next((mm for article in self.items for mm in article.obligations if _Text._vocables(mm.split(':')[0]) == clef), wording)
            digit = next((nn for (nn, article) in enumerate(self.items, 1) if wording in article.obligations), 1)
            if wording and (digit, wording) not in self.allusions:
                self.allusions.append((digit, wording))
        self.untangle()
        purse = digest.get('budget')
        if isinstance(purse, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', purse):
            purse = float(purse)
        if isinstance(purse, (int, float)) and (not isinstance(purse, bool)) and (0 < purse):
            self.purse = purse
        boundary = _Kinds._quantity_n(digest.get('messages'))
        if boundary is not None and 0 <= boundary:
            self.dispatch_note_boundary = boundary
        elif self.functions.get('ask'):
            self.dispatch_note_boundary = 3
        paces = _Kinds._quantity_n(digest.get('steps'))
        if paces is not None and 10 <= paces < self.upper_paces:
            self.upper_paces = paces
        for item_rec in _Kinds._entries_of(digest.get('process')):
            if isinstance(item_rec, dict) and str(item_rec.get('step') or '').strip():
                whenever = str(item_rec.get('when') or '').strip().lower()
                self.procedure.append({'step': str(item_rec['step']).strip(), 'when': 'order' if not whenever in ('choose', 'order', 'end') else whenever, 'messages': max(0, _Kinds._quantity_n(item_rec.get('messages')) or 0)})
        if self.functions.get('ask'):
            chatter = [pp['messages'] for pp in self.procedure if pp['when'] == 'order' and pp['messages']]
            self.fallback = 0 if not chatter else min(max(chatter), self.dispatch_note_boundary)

class _StoreLine:

    def operate(self, moniker: str, operands: dict[str, Any]) -> dict[str, Any] | None:
        freight = dict(self.fastening)
        freight['step'] = self.pace + 1
        freight['action'] = {'name': moniker, 'args': operands}
        for _ in range(4):
            consequence = _habitat.post_verbose('/environment/call', freight)
            if isinstance(consequence.data, dict):
                data = consequence.data
                self.pace = self.pace + 1 if not isinstance(data.get('step'), int) else data['step']
                if data.get('finished'):
                    self.concluded = True
                spotted = data.get('observation')
                if moniker in self.folio_utensils() and (not (isinstance(spotted, dict) and spotted.get('error'))):
                    for worth in operands.values():
                        self.unfolded.update((vv.strip() for vv in ([worth] if not isinstance(worth, list) else worth) if isinstance(vv, str)))
                return data
            error = consequence.error or {}
            standing = error.get('status')
            main_text = str(error.get('body') or '')
            self.footprints.append({'kind': 'env_fail', 'status': standing, 'body': main_text[:200]})
            if 409 == standing:
                scant = main_text.lower()
                if 'finish' in scant or 'quarantin' in scant or 'finaliz' in scant:
                    self.concluded = True
                    return None
                discovered = re.search('expected (\\d+)', main_text)
                if discovered:
                    self.pace = int(discovered.group(1)) - 1
                    freight['step'] = self.pace + 1
                    continue
                if 'reuse' in scant or 'different action' in scant:
                    self.pace += 1
                    freight['step'] = self.pace + 1
                    continue
                return None
            if standing in (400, 401, 402, 403, 404, 422):
                if 402 == standing:
                    self.snapped = True
                return None
        return None

class _ModelLine:

    def petition_json(self, instruction: str, machines: list[str], upper_counters: int=1800) -> dict[str, Any] | None:
        for machine in machines:
            missive = self.ponder([{'role': 'user', 'content': instruction}], [machine], upper_counters)
            gained = _Text._json_object((missive or {}).get('content'))
            if gained is not None:
                return gained
            if self.snapped or self.zero_allowance or TOTAL_DISPATCHES <= self.dispatches:
                return None
        return None

    def dispatch_machine(self, machine: str, missives: list[dict[str, Any]], upper_counters: int, utensils: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        transmitted = missives
        if machine.startswith('anthropic/') and len(missives) > 1:
            transmitted = [dict(mm) for mm in missives]
            tag_marks = [0] + ([] if not len(transmitted) > 1 else [len(transmitted) - 1])
            for index in tag_marks:
                matter = transmitted[index].get('content')
                if isinstance(matter, str) and matter:
                    transmitted[index]['content'] = [{'type': 'text', 'text': matter, 'cache_control': {'type': 'ephemeral'}}]
        main_text: dict[str, Any] = {'model': machine, 'messages': transmitted, 'max_tokens': upper_counters, 'temperature': 0}
        if utensils:
            main_text['tools'] = utensils
            main_text['tool_choice'] = 'auto'
        consequence = _model.post_verbose('/inference/chat/completions', main_text)
        data = consequence.data
        if isinstance(data, dict) and data.get('choices'):
            consumption = data.get('usage') or {}
            if isinstance(consumption.get('cost'), (int, float)):
                self.expense += float(consumption['cost'])
            missive = data['choices'][0].get('message') or {}
            if missive.get('content') or missive.get('tool_calls'):
                return (missive, 200)
            self.footprints.append({'kind': 'llm_fail', 'model': machine, 'status': 0})
            return (None, 0)
        error = consequence.error or {}
        standing = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            standing = _Kinds._quantity_n(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.footprints.append({'kind': 'llm_fail', 'model': machine, 'status': standing, 'body': str(error.get('body') or '')[:200]})
        return (None, standing)

    def ponder(self, missives: list[dict[str, Any]], machines: list[str], upper_counters: int, utensils: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for machine in machines:
            if TOTAL_DISPATCHES <= self.dispatches or self.metronome.secs() >= SPAN_FINAL:
                return None
            self.dispatches += 1
            tests = 0
            while True:
                (missive, standing) = self.dispatch_machine(machine, missives, upper_counters, utensils)
                if missive is not None:
                    return missive
                clearing = standing is None or standing in (402, 408, 425, 429) or (isinstance(standing, int) and 500 <= standing)
                if not clearing or tests >= (len(LINGERINGS) if not standing in (None, 408) else 1) or self.lingered + LINGERINGS[tests] > LINGER_SUM:
                    break
                halt = LINGERINGS[tests] * (0.8 + 0.4 * random.random())
                self.lingered += LINGERINGS[tests]
                tests += 1
                threading.Event().wait(halt)
            if 402 == standing:
                self.zero_allowance = True
                return None
        return None

class _Allowance:

    def clip(self) -> None:
        dimension = sum((len(_Text._serialize(mm.get('content'))) for mm in self.converse))
        if ANNALS_BOUND >= dimension:
            return
        utensils = [ii for (ii, mm) in enumerate(self.converse) if mm.get('role') == 'tool']
        for index in utensils[:max(1, len(utensils) // 2)]:
            matter = self.converse[index].get('content') or ''
            if len(matter) > 600:
                serials = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', matter)
                serials = [ii for ii in dict.fromkeys(serials) if ii in self.tiers][:20]
                self.converse[index]['content'] = matter[:400] + f"...(older result shortened; listings in it: {', '.join(serials)})"

    def assessors(self, boundary: float=OUTLAY_MILD) -> list[str]:
        if boundary >= self.expense and self.expense + WEIGHTY_HEADROOM <= OUTLAY_FIRM:
            return [WEIGHTY, NIMBLE]
        return [] if not OUTLAY_FIRM >= self.expense else [NIMBLE]

    def end_part(self) -> int:
        chatter = any((pp['when'] == 'order' and pp['messages'] > 0 for pp in self.procedure))
        return 1 if not (chatter and self.dispatch_note_boundary > self.dispatch_note_spent_on and self.barrier_check_payable('order')) else 2

    def barrier_check_payable(self, phase: str) -> bool:
        return any((pp['when'] == phase for pp in self.procedure)) and (phase, self.trolley_tag_mark() + f'#{self.edition}') not in self.barrier_check_fine

    def dispatch_notes_loose(self) -> int:
        return self.dispatch_note_boundary - self.dispatch_note_spent_on - max(0, self.fallback - self.barrier_check_dispatch_notes)

    def leftover(self) -> int:
        return self.upper_paces - self.pace

class Steward(_Driver, _Checkout, _Audit, _Offers, _Pointers, _Fitting, _CartView, _Talk, _Lanes, _Interview, _Browsing, _Records, _Ledger, _Intake, _StoreLine, _ModelLine, _Allowance):

    def __init__(self, snag: dict[str, Any]):
        habitat = snag['environment']
        self.fastening = dict(habitat['binding'])
        self.outlook = habitat['policy_view']
        self.appeal = str(self.outlook.get('query') or '')
        self.utensils = [tt for tt in self.outlook.get('tools') or [] if isinstance(tt, dict) and isinstance(tt.get('function'), dict) and isinstance(tt['function'].get('name'), str)]
        self.outlines = {tt['function']['name']: {'type': 'object', 'properties': {}} if not isinstance(tt['function'].get('parameters'), dict) else tt['function'].get('parameters') for tt in self.utensils}
        paces = _Kinds._quantity_n(self.outlook.get('max_steps'))
        if paces is None and isinstance(self.outlook.get('max_steps'), float):
            paces = int(self.outlook['max_steps'])
        self.upper_paces = 30 if not (paces and 0 < paces) else paces
        self.pace = 0
        self.concluded = bool(self.outlook.get('finished'))
        self.snapped = False
        self.zero_allowance = False
        self.lingered = 0
        self.metronome = _Metronome()
        self.omissions = 0
        self.expense = 0.0
        self.dispatches = 0
        self.cycle_dispatches = 0
        self.footprints: list[dict[str, Any]] = []
        self.digest: dict[str, Any] = {}
        self.functions: dict[str, str] = {}
        self.items: list[Article] = []
        self.purse: Any = None
        self.dispatch_note_boundary = 0
        self.dispatch_note_spent_on = 0
        self.procedure: list[dict[str, Any]] = []
        self.fallback = 0
        self.barrier_check_dispatch_notes = 0
        self.barrier_check_fine: set[tuple[str, str]] = set()
        self.over_with_logbook: list[str] = []
        self.over_with_paces: list[int] = []
        self.petitioned = False
        self.quiz: list[tuple[str, str]] = []
        self.tiers: dict[str, dict[str, Any]] = {}
        self.complete_: dict[str, dict[str, Any]] = {}
        self.spotted_purchase: list[str] = []
        self.listened: set[str] = set()
        self.jottings: list[str] = []
        self.edition = 0
        self.trolley_handled = False
        self.trolley_upon = -1
        self.trolley_spotted_upon = -1
        self.trolley_outlook: Any = None
        self.jot_titled = False
        self.trolley_strands: dict[str, int] = {}
        self.probed: set[str] = set()
        self.inspected = False
        self.proffer_spotted: dict[str, int] = {}
        self.departed: dict[str, int] = {}
        self.not_offered: set[str] = set()
        self.vended_strands: set[str] = set()
        self.jot_vended = False
        self.withheld = 0
        self.ident_operands: set[tuple[str, str]] = set()
        self.trolley_uttered = ''
        self.vetoed_upon = -1
        self.clues: list[str] = []
        self.spurn_segments = 0
        self.shoved = False
        self.assembled: list[str] = []
        self.let_gone: set[str] = set()
        self.tardy_withheld = False
        self.tardy_salvage = False
        self.clue_items: dict[str, list[str]] = {}
        self.findings: dict[str, bool] = {}
        self.ratified: str | None = None
        self.wrapped_up = False
        self.ident_clef: str | None = None
        self.particulars_of: dict[str, bool] = {}
        self.lined_up: list[str] = []
        self.acquires = 0
        self.vetoes = 0
        self.latest_few: list[str] = []
        self.converse: list[dict[str, Any]] = []
        self.unfolded: set[str] = set()
        self.allusions: list[tuple[int, str]] = []
        self.allusion_worths: dict[str, str] = {}
        self.wakes = 0
        self.wake_segments = 0
        self.collate_fine = 0
        self.collate_poor = 0
        self.utensil_clefs: dict[str, set[str]] = {}
        self.outlook_clefs: dict[str, tuple[int, set[str]]] = {}
        self.peruse_hoard: dict[str, tuple[int, int, str]] = {}
        self.folios_peruse = 0
        self.silhouettes: dict[str, dict[str, Any] | None] = {}
        self.tariff_clef: str | None = None
        self.inventory_clef: str | None = None
        self.lower_cost_tag_marks: set[str] = set()
        self.broader_withheld = False
        self.leafed_observation: list[Any] = []
        self.design_purchase: list[int] = []
        self.strand_persists = 0
        self.edge: list[tuple[int, int]] = []
        self.conform_persists = 0
        self.combed: list[dict[str, Any]] = []
        self.tie_hunts = 0
        self.tie_spotted: dict[str, list[str]] = {}

    def execute(self) -> list[dict[str, Any]]:
        if self.concluded:
            return self.footprints
        self.peruse_appeal()
        self.footprints.append({'kind': 'brief', 'brief': self.digest})
        debut = self.outlook.get('opening')
        if debut is not None:
            self.soak_up(debut)
        preface = "SHOPPER'S REQUEST:\n" + self.appeal
        if debut is not None:
            preface += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.exhibit(debut)
        preface += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.standing()
        self.converse = [{'role': 'system', 'content': HANDBOOK}, {'role': 'user', 'content': preface}]
        self.toil(CYCLE_DISPATCHES, OUTLAY_FIRM)
        self.duration_trolley()
        if not self.concluded:
            flipped = self.conclude(once_more=True)
            if flipped and (not self.concluded) and (not self.snapped) and (not self.zero_allowance) and (self.leftover() > 1):
                self.converse.append({'role': 'user', 'content': '[Kyo] Kyo tried to end the episode and ' + flipped + '\nApply what the store says, then finish.\n' + self.standing()})
                self.toil(self.cycle_dispatches + 10, OUTLAY_FIRM + 0.01)
            if not self.concluded:
                self.conclude()
        self.footprints.append({'kind': 'end', 'finished': self.concluded, 'steps': self.pace, 'cost': round(self.expense, 5), 'calls': self.dispatches})
        return self.footprints

    def toil(self, cycle_lid: int, outlay_lid: float) -> None:
        inactive = 0
        while not self.concluded and (not self.snapped) and (self.leftover() > 0) and (cycle_lid > self.cycle_dispatches) and (outlay_lid > self.expense) and (self.metronome.secs() < SPAN_HALT):
            if self.set_up_toward_petition():
                synopsis = self.enquiry()
                if synopsis:
                    self.converse.append({'role': 'user', 'content': synopsis + '\n' + self.standing()})
                continue
            self.clip()
            self.cycle_dispatches += 1
            missive = self.ponder(self.converse, [NIMBLE, RESERVE], 3000, self.proffered())
            if missive is None:
                self.omissions += 1
                if self.zero_allowance or TOTAL_DISPATCHES <= self.dispatches or 2 < self.omissions:
                    break
                continue
            dispatches = []
            for (digit, dispatch) in enumerate(missive.get('tool_calls') or []):
                if not isinstance(dispatch, dict):
                    continue
                fun = {} if not isinstance(dispatch.get('function'), dict) else dispatch.get('function')
                untreated = fun.get('arguments')
                interpreted = ({} if not (isinstance(untreated, str) and untreated.strip()) else _Text._json_object(untreated)) if not isinstance(untreated, dict) else untreated
                dispatches.append({'id': str(dispatch.get('id') or f'k{self.cycle_dispatches}_{digit}'), 'type': 'function', 'function': {'name': str(fun.get('name') or ''), 'arguments': _Text._serialize({} if not isinstance(interpreted, dict) else interpreted)}})
            matter = missive.get('content')
            if not isinstance(matter, str):
                matter = '' if not matter else _Text._serialize(matter)
            if not dispatches:
                inactive += 1
                self.converse.append({'role': 'assistant', 'content': matter or '(no call)'})
                self.converse.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.standing()})
                if 3 <= inactive:
                    break
                continue
            inactive = 0
            pledging = {self.functions.get('cart'), self.functions.get('buy'), self.functions.get('refuse')} - {None}
            if not self.petitioned and self.functions.get('ask') and (self.dispatch_notes_loose() > 0) and any((cc['function']['name'] in pledging for cc in dispatches)) and (self.leftover() > 8):
                self.converse.append({'role': 'assistant', 'content': matter or None, 'tool_calls': dispatches})
                for dispatch in dispatches:
                    self.converse.append({'role': 'tool', 'tool_call_id': dispatch['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                synopsis = self.enquiry()
                self.converse.append({'role': 'user', 'content': (synopsis or 'The interviewer had nothing to ask.') + '\n' + self.standing()})
                continue
            self.converse.append({'role': 'assistant', 'content': matter or None, 'tool_calls': dispatches})
            self.footprints.append({'kind': 'turn', 'text': matter[:300], 'calls': [cc['function'] for cc in dispatches]})
            retorts = self.execute_dispatches(dispatches)
            for (index, (cident, wording)) in enumerate(retorts):
                if index == len(retorts) - 1:
                    wording += '\n' + self.standing()
                self.converse.append({'role': 'tool', 'tool_call_id': cident, 'content': wording})

def agent_main(snag_info: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        sortie = Steward(snag_info)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(failure).__name__}: {failure}'[:400]}]
    try:
        return sortie.execute()
    except Exception as exc:
        sortie.footprints.append({'kind': 'crash', 'error': f'{type(failure).__name__}: {failure}'[:400]})
        try:
            if not sortie.concluded:
                sortie.conclude()
        except Exception as again:
            sortie.footprints.append({'kind': 'crash', 'error': f'{type(once_more).__name__}: {once_more}'[:400]})
        return sortie.footprints
    finally:
        sortie.metronome.cease()
__all__ = ['agent_main']