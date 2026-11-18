from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient


class Models:
    LIGHT = "anthropic/claude-haiku-5.5"
    HEAVY = "anthropic/claude-sonnet-5.5"
    SPARE = "openai/gpt-6-luna"


class Caps:
    LOOP_CALLS = 42
    ALL_CALLS = 60
    # a race run's key holds about $5 for its 90 tasks, the shopper's costs included, and tasks run in id order: in race 12 both
    # of this agent's runs ran dry about two-thirds through, and the last 24-31 tasks scored 0. Past the soft limit a task goes
    # on with the cheap model only (past the hard limit the work stops)
    SPEND_SOFT = 0.045
    SPEND_HARD = 0.07
    HEAVY_TOKEN = 2e-6          # the heavy model's price per input token, to see a call coming
    REPLY_COST = 0.001          # about what one reply of the cheap model costs
    # the light model's hidden thinking counts against a reply's limit and costs next to nothing: at 1,500 tokens 6 of 6 reviews
    # came back with no answer, and at 1,800 about half of the chain reads would have been cut; with room they took up to 5,800
    LIGHT_ROOM = 8000
    CHAIN_SPEND = 0.025


class Retry:
    # waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
    WAITS = (4, 10, 20, 35)
    WAIT_TOTAL = 45


class Views:
    SHOW_CHARS = 450
    ROW_CHARS = 300             # each text of a results page: every call resends what earlier calls showed
    FADED_CHARS = 60            # each text of a row shown in full before, or one that cannot be the purchase
    STATE_EVERY = 6             # replies between repeats of an unchanged ledger
    OBS_LIMIT = 15000
    HISTORY_LIMIT = 150000


class Moves:
    GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
    AUTO_PAGES = 6              # further search pages Acer reads itself for a cheapest, most or least pick
    CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Acer reads before a cheapest checkout
    CHAIN_ROUNDS = 12           # readings of the records while following pointer chains
    CHAIN_SEARCHES = 8          # searches Acer makes itself for a chain's next listing
    CHAIN_SHOWN = 6             # pages of a chain shown to a review
    # one message can name several properties. Race 14: the first name was told as often as a lone question (90% against 89%), the
    # next ones 67, 54, 40, 54, then under 20%, and known needs told rose to 92% on one-item tasks (66-84% with lone questions)
    ASK_NAMES = 8
    PLAN_PICKS = 5              # picks the planner ranks per item; the item's other listed keys fill the rest of a message
    CHAIN_SIBLINGS = 3          # other listings of a link's shop Acer opens when the one it opened leads nowhere


class Looks:
    LONG_VALUE_WORDS = 2.5      # words per value from which an attribute key's values count as long
    FIGURE_VALUE_SHARE = 0.5    # share of values carrying a digit from which they count as figures
    # a key showing one value on every listing read so far sits lower: over races 5-11 this put more hidden needs in a
    # category's first four keys in 6 of 7 races (+1.6 to +5.6 points) and changed nothing in the other
    SAME_VALUE_ODDS = 0.3


class Wires:
    # one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
    llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
    env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


class Wording:
    GUIDE = """You are Acer, buying carefully on behalf of a single shopper inside an online store, and calling the store's tools is the only way you can act. Each episode is graded 1 or 0, and it gets the 1 only for an exactly right finish: either a purchase holding every requested item in its requested quantity, within the budget, with each item satisfying every requirement still in force at checkout; or, if the store really has nothing that satisfies everything, an ending through the no-purchase tool, in the manner the request asks for.

HOW THIS STORE BEHAVES
- Thinking costs nothing, but every tool call spends one of a limited number of steps. Tools, their arguments and the shape of what they return are specific to this store: go by each tool's description and schema, and use precisely the argument names and permitted values they define.
- Searching can be restrictive: a category filter, keyword matching that may look only at the first few words you type, pages of only a handful of rows, and a token or offset for moving to the next page. Whenever the answer needs every row, compare the store's reported total with how many rows you have, and continue paging using precisely the token or offset the store returns. Keyword queries do best when short and led by the value (the wanted value followed by the product noun). Zero rows from a filtered search usually signal a bad filter value or a bad filter parameter: repeat it with the label copied exactly from the result rows, or switch to some other filter parameter.
- Rows in search results are only summaries, and each store splits a listing's record its own way: its attributes (properties) may sit in one view, its offer and description in another, and a side-by-side view may hold either one. The ledger records which tool shows attributes as soon as Acer has found it. Requirements about properties are decided by those attributes, so fetch them for every listing you might buy; when one call returns what you need for several listings, it saves steps. A listing whose situation has not changed since you read it should not be opened a second time.
- You never write to the shopper yourself: Acer's interviewer asks the shopper's questions for you. Its answers show up in the conversation and go into the ledger.
- A request may also want steps done on the way, such as an approval, a message, or something to verify or look at beforehand; the ledger lists these. Whatever is due right before ordering or right before ending, Acer handles at the moment you call the checkout tool or the ending tool. Those steps are therefore no reason to wait and no reason to end without buying; carry out every other step by means of the store's tools (questions about products go through Acer's interviewer).

WHEN A LISTING FITS
- A listing's category label settles what kind of product it is: the label has to match the requested item taken as a whole, and a neighbouring label never does. On a matching label a listing counts as that kind even when its title or description names some other kind of product: judge it on the requirements, never on what its name looks like, and never swap it for a dearer one that only looks more like the request.
- If the listing's attribute for a property states the required value, that requirement is met. Spacing, underscores, camelCase or prefixes in the key make no difference, and matching one element of a list split by commas or slashes suffices. Values are matched word by word: differences of case, of singular versus plural, and plain synonyms are allowed, and a value holding the required value's words passes (teal is satisfied by "deep teal", roomy by "extra roomy fit"). A value in other words fails (an abbreviation or code does not equal the word it abbreviates, and "sea green" is not met by "teal green"). Convert units before comparing numbers; the edge value itself is within a limit, and a range passes only when it lies wholly inside it.
- The decision belongs to the attribute keyed by the requirement's property, whether under the property's own name or a plain synonym of it: if that attribute gives a different value, the listing fails, regardless of the title or any other key. A key that is about some other property says nothing either way about the requirement, even when the two are related (the colour of a handle is not the body's colour). After each opened listing, Acer adds [evidence] lines showing the keys named like a requirement.
- Lacking an attribute for that property, a listing can still pass on the wanted value when another attribute states it (a styles or features list that contains it, for instance) or when its title or description does. A figure of a different kind never passes (a weight does not serve as a capacity), and neither does a number standing alone, with nothing saying what it measures.
- A requirement is met by the category when its value appears in the category label (say, a season or an audience the label names).
- Some requirements name another listing rather than a value ("the same X as the listing that <shop> sells with <property>", "no more than N away from the X shown on the page of ..."). Resolve these first. Find the listing in question (searching by its category, brand, shop or words) and open its complete page with the single-item or side-by-side tool, because a search row is not enough; should that page refer you on to yet another listing for that property, track it down and open it likewise, and take the value from the final page. Until all the pages in that chain have been opened, the order does not count. If what is wanted is merely a value different from that listing's, the rule shuts out only listings carrying that exact value, and by itself it never justifies ending without a purchase.
- If a listing you are on the point of buying points to a further page with more about it, view that page first.
- If one requested item has to be compatible with another (an accessory and its device), pick the pair together, going by what the listings themselves say (a fits-list, or a model or series code printed on both pages). If no accessory satisfying its own requirements works with the device you chose, change to a different device that satisfies its own requirements and accepts such an accessory; ending without a purchase is not allowed before that change has been tried.
- Exclusions and closed lists must be obeyed. Wishes ("ideally", "would be nice") serve only to break ties.
- Anything the store marks as sold out is off limits. The total, meaning price times quantity added up over every line, must not go over the budget; landing exactly on it is fine.

PICKING
- For the cheapest (smallest price, lowest total), the order must include the lowest-priced of ALL the fitting listings. Collect every fit (page through, because row order is often unrelated to price; for a short result, Acer itself reads on to the following pages), check candidates starting from the lowest price, and the first candidate that fits completely is the one to take. When a search lets you set a maximum price, use that: list the category below what your current best pick costs, which leaves just the cheaper candidates for checking.
- When the request is about the most or least of a measure, line up all fitting listings by it and choose the extreme; a listing silent on the measure cannot win.
- In every other case, any listing meeting everything is correct, so spend no steps looking for a better one.
- As soon as an opened listing meets every requirement of an item, cart it and move to the next item (for a cheapest, most or least pick, only after comparing the fitting set). When several opened listings meet every requirement and nothing known tells them apart, take the cheaper one: among such ties the right listing was the cheaper one about three times out of four. That tie-break comes after this one: when one fitting listing's attributes state the wanted words themselves and a cheaper one only implies them (an abbreviation such as L for large, a neighbouring key such as pack size for pack type, or the title alone), the one stating them is the pick. Stores frequently sell out a line shortly after it is carted, and carting early keeps steps in hand for a replacement.
- When several different items are requested, each needs its own cart line at the quantity requested for it.

WHEN THINGS CHANGE
- Things can change on the shopper's side or the store's: the budget may be replaced, a requirement may be added or removed, a price may move, a listing may sell out. Such changes turn up inside tool results, often as a refusal at your checkout attempt; a note counts no matter which tool's result carries it, and the ledger shows the requirements currently in force. A refusal does not end the episode: apply all the changes, fix whatever the cart needs and run the checkout again.
- Because prices and stock may shift without any notice, read a chosen listing's record again shortly before checkout.

ENDING THE EPISODE
- Make the cart hold exactly the chosen lines (take out anything else), view the cart once, and finally run the checkout as a call of its own.
- Ending without a purchase is only for when a genuine search turns up no listing meeting everything. Being low on steps proves no such thing: in that case, buy the best cart you have assembled. For a no-purchase ending, follow the wording the request gives, and if a remark is needed, make it brief and mention no listing.
- Call at least one tool in each reply. Write little: state what you know, then the next move."""

    BRIEF = """A shopping agent is about to work on the shopper's request below, in the store whose tools are listed. Prepare its task sheet: answer with exactly ONE JSON object, compact (no indentation) and with nothing around it, that holds the following keys.

Tools for the jobs:
"roles": an object giving, for each job, the exact name of the tool that does it, or null when no tool does it: "search", "open" (displays the attributes or properties of a single listing; when there are two single-listing tools, the one whose description promises attributes, properties or specifications), "offer" (displays a single listing's current offer, such as price and stock, when a tool other than "open" does this), "compare" (several listings in full), "ask" (sends the shopper a message), "cart" (changes the cart), "cart_view" (displays the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments with which the "buy" tool checks out (the mode, for a tool that takes a mode argument), otherwise {}.
"refuse_args": the fixed arguments with which the "refuse" tool ends without a purchase, otherwise {}.
"cart_modes": when the "cart" tool takes a choice argument, {"add": the option that puts units in, "remove": the option that takes units out}, otherwise {}.

Limits:
"budget": as a number in the request's own unit, the most the shopper will pay in total; null if there is none.
"budget_unit": the name of that unit in words (for example "cents"); null if there is none.
"budget_scope": "each" when the limit is per item, "total" when it covers the whole order, null when there is no limit.
"in_stock_only": true, unless the request clearly accepts items that are unavailable.
"messages": as an integer, how many messages the agent may send the shopper; null if the request does not say.
"steps": as an integer, how many steps, moves or tool calls the request allows; null otherwise.

Choosing:
"pick": "cheapest" for a request that wants the lowest price or the smallest total; "most" or "least" for one that wants the extreme of some measure; "any" in every other case. Saying money is tight is not a pick rule.
"pick_measure": for "most"/"least", the measure to maximise or minimise; otherwise "".

The products:
"items": a list holding one entry per distinct product to be bought. Each entry is {"name": the kind of product as the request words it, "shelf": when some tool lists the store's category labels, the label for this product copied exactly, otherwise "", "quantity": an integer count of units of that one listing (1 when the request names none; a pair or a set sold as a single listing counts as 1), "musts": the product's firm requirements as a list of "property: constraint" strings, each with the exact value, limit, exclusion or closed list}. Rules for the items:
- In a must, the property keeps the request's own word for it, never a neighbouring property's name: "meant for a lakeside occasion" gives "occasion: lakeside", and "a 12-inch blade" gives "blade length: 12 inch".
- When a value is to be read off another listing, that is ONE must containing the whole description of the other listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"). The words that describe that other listing (its shop, its kind, and any "with <property> <value> and <property> <value>" after it) remain inside this one must and never turn into musts of the product.
- Musts never hold price, budget, stock or quantity.
- Background in the request (the person who will use it, where it will stand, what went wrong earlier, the occasion behind it) is context and not a must, unless it plainly states the value or limit of a property. Words that belong to the product's own name (a season, an audience or a use named together with it) are also never musts; they are part of "name".
- One phrase that joins products with "and" (a single "I want ..." with a single quantity, often a store department's name) is ONE item. Several items are right only when each product comes with its own quantity or its own requirements.

Other parts of the request:
"lookups": copied exactly, the musts whose value has to be read first from another listing the request points at; [] when there are none.
"ask_first": the things the request explicitly says to check with the shopper before choosing; empty when there are none.
"process": [] unless the request requires steps beyond finding and buying the right items, such as getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first or using a particular tool. Give each as {"step": the thing to do in at most 15 words, "when": "choose" (before the products are chosen), "order" (just before checking out) or "end" (just before ending without a purchase), "messages": the number of messages to the shopper it takes (0 when it takes none)}. Item requirements, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"rules": every instruction in the request about how to work or how to end, at most 12 words apiece.
"wishes": what the shopper would merely like.
"quiet_decline": null when the request says nothing on it, true when it says not to suggest or name alternatives if nothing fits, false when it asks for alternatives.
"""

    PLAN = """Your job: decide which questions a shopping agent puts to a shopper. On top of what the request says, the shopper holds a few firm preferences they never wrote down, each tied to one property of one product they want. Only a message naming that very property draws such a preference out. One message may name several properties of an item, but the earlier a property stands in the message the likelier the answer covers it, so what you rank first matters most.

What you receive: the request; the items, each with the points the request already decides for it; and, for each item's category, a numbered list of the attribute keys that turned up on opened listings, with the count of those listings showing the key and a few example values. The property behind an unwritten preference is usually among the listed keys, though not always.

The order to choose in:
0. Anything the request itself wants you to confirm with the shopper leads (you will find it under ASK FIRST when present); name each by the property it concerns.
1. Go on to hinted properties. Often the story in a request hints that an unwritten preference exists while keeping its value back, in a sentence about: a matter the shopper is unable to judge or hands over to you; someone whose taste is the deciding one; how the product looks or the place it belongs in; the person who will use or wear it; or its contents. Turn each such sentence into the one property it hints at plus the item concerned, keeping the story's order.
Name a hinted property by the listed key that carries it: a gender key when the hint concerns who will wear the item, a colour or finish key (the one the request leaves open) when it concerns the item's looks. Only when the item has no such key use a plain name of your own, such as colour; a property none of its listings shows rarely draws an answer, so prefer the listed keys over gender, finish or size for an item whose listings don't show them.
2. Finish with the listed keys, best first: the properties shoppers for that kind of product choose between before anything else (often its type, its material or its colour, whichever that kind of product is bought by) rank high, and so do keys most of the item's listings carry; avoid a key seen on just one or two of those listings.
Drop from consideration: a property the request already decides for the item; another name for something already chosen; a key whose values are the purchasable variants (sizes, colours and pack counts mixed together); identifiers; warranty and similar logistics. A pack type or packaging key is neither a variant nor logistics: it stays. Rank each item's picks on their own, its hinted properties first; the agent names them in your order.
Write "label" as the plain name of the property and nothing else, so no example values, no brackets and no second property; when it comes from a listed key, spell out all of that key's words plainly ("lid colour" for lidColour, "rim finish" for rim_finish), never trimming them to "colour" or "finish".
Up to COUNT picks for each item, the strongest first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "the story words it is drawn from, 8 at most, or an empty string"}]}"""

    HEAR = """Below are a shopping agent's questions and the shopper's answers. Work out what each answer means for the purchase.
What goes where:
- "needs": a firm preference stated in an answer, giving the exact value or limit together with its unit ("at least 30 cm", "oval", "not wicker"). Its property is the one the answer itself names, which can differ from the property asked about; answers that all concern one property make a single need. The value is copied word for word from the answer, nouns and units included ("at least 3 shelves", not "at least 3").
- When an answer states the value by pointing to a different listing (a value like, unlike or near the one on some shop's product, that product identified by a few of its properties), record ONE need on the property the answer names and set "points" to true. For its value copy the answer's own words exactly, from the comparison words to the end of that description, with whatever it says about a listing that page may name; never shorten it and never reword it. The other product's describing properties are not needs of this item.
- "drop": a requirement the shopper now gives up.
- "free": the property of a question that the shopper said they do not mind about.
- "crowded": the numbers of the questions whose answer objects to being asked too much at once or wants one thing at a time.
- "budget": a new total limit, as a number in the unit of CURRENT BUDGET; null otherwise. When an answer's figure is already in that unit or another spelling of it, copy the figure unchanged and never rescale it.
- A value named in folksy or roundabout words is still a stated value.
Return ONE JSON object and nothing else: {"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "the requirement given up"}], "free": [{"item": item number, "property": "..."}], "crowded": [question numbers], "budget": a number or null}"""

    NOTE = """Some text appeared while a shopping agent was busy: a note from either the shopper or the store. Classify it and answer with ONE JSON object and nothing else:
{"kind": "change" if it changes the task (budget, requirements, a price, availability, or the order being refused), "error" if it merely says a call was malformed or was not accepted, "other" otherwise, "budget": the new limit on the total, as a number in the request's own unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement that is no longer wanted"}], "gone": [ids of listings it says are unavailable], "sold_out": true if it says a chosen listing has sold out, is gone or can no longer be had (even unnamed), "price_moved": true if it says a price has moved, "summary": "one brief sentence"}
Report each change the note makes; a requirement that gets replaced is a drop of the old one together with an add of the new one."""

    CHECK = """Before a shopping agent checks out, you give the cart its last review: judge only from the records shown whether it is exactly right.
Reading the records: the listing's attribute for a property settles a requirement on it by stating the value; key spelling is irrelevant, one element of a list suffices, case, singular or plural and plain synonyms are fine, and a value holding the required value's words also counts (teal is met by "deep teal"); units are converted, a limit includes its edge, and a range passes only if all of it lies inside. When an attribute's key names the requirement's own property, that attribute decides, and a different value there fails the listing. An attribute about some other property, even a close one (the colour of a handle is not the colour of the body), decides nothing. Where the listing has no attribute for the property, the wanted value may be found under another attribute or in the title or description, but a number of another sort does not count (a weight is no capacity), nor does a lone figure whose text never says what it measures. The category label must match the requested item as a whole, and a requirement the label itself states is met by that category; on a matching label, a title or description naming another kind of product changes nothing. A demand to differ from another listing's value is met unless the records show exactly that value. Each requested item needs a cart line of its own at the requested quantity, nothing else may sit in the cart, nothing in it may be shown out of stock, and the sum of price times quantity over the lines may not exceed the budget. When records of a listing conflict, the latest one counts.
With a cheapest, most or least pick rule, look at the other listings shown too: if one of them clearly meets every requirement and beats the cart's pick on that rule, the cart is wrong.
Answer with ONE JSON object and nothing else: {"ok": true or false, "problems": ["a short, concrete problem", ...], "better": [ids of listings to put in place of a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement that no record speaks to."""

    READING = """listings that are out of stock or that would break the budget never fit; the category label decides a listing's kind, whatever its title says; a property is settled by the attribute whose key names it (spell the key any way, let one element of a list do, accept synonyms and converted units, and let a value holding the required words do, e.g. teal is met by "deep teal"; a limit includes its edge), while an attribute about some other property, however close (the colour of a handle is not the colour of the body), settles nothing; where no attribute names the property, the wanted value may come from another attribute, the title or the description, though a lone figure that never says what it measures proves nothing; the category label must match the item, and a requirement the label states outright is met by that category; a demand to be unlike another listing's value is met unless the records show exactly that value; when one listing's attributes state the wanted words themselves, a cheaper one that only implies them (an abbreviation, a neighbouring key, the title alone) is not the pick."""

    AUDIT = """The shopping agent wants to stop without buying, convinced that nothing fits. Hold the listing records shown up against the requirements, applying these reading rules: """ + READING + """
A record marked ROW was only seen as a search result row and its page (the attributes) was never read, so a property the row does not show can neither pass it nor fail it. A record marked PAGE had its page opened, but no attributes were read from it: everything it says counts, and a property it does not show can neither pass it nor fail it.
Answer with ONE JSON object and nothing else: {"fits": {"item number": [ids of listings that look as if they meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings that are of a requested kind and inside the budget and whose page could reveal a fit, most promising first, at most 6], "reason": "a single sentence"}"""

    UNMIX = """Firm requirements on a product were read out of a shopper's request. A few of them take their value from another listing the request points to, and the words that single out that other listing (what kind it is, which shop sells it, properties named after it with "with ...") can be mistaken for requirements on the product being bought. Check the request's wording and report which of the OTHER REQUIREMENTS below do nothing but describe that other listing. Answer with ONE JSON object and nothing else: {"describe_other": [each such requirement, copied exactly; leave the list empty if every one is a genuine requirement on the product]}"""

    TRAIL = """A shopping agent is working on an order. Some requirements give no value of their own but point at another listing, and that listing's page may in turn send the reader to a further listing for the property. Only once the agent has fully opened every listing on every such chain does the order count. Trace each chain from the requirement's own description, link after link, by what the pages themselves say, with the records below (OPENED means the agent opened that page; ROW means it was only seen as a search result row and its page was never read). A page's words and its attributes both count as what it states. When several records could be one link, name the record that best fits the description; an OPENED page that contradicts the description is not the link, so pass over it for another listing of that shop.
Answer with ONE JSON object and nothing else: {"chains": [{"need": the requirement exactly as written, "links": [{"id": the listing id when one of the records below is this link, else "", "about": the way the requirement or the preceding page refers to this listing, "search": "up to six words a text search would find this listing by: its shop, its kind of product and the most distinctive value stated for it", "short": "the same without the value: its shop's name first, then its kind of product"}], "done": true when the last link is an OPENED page stating the property's value itself, else false, "value": that value as the last page states it, with any rule the pages give for relating values, or ""}]}
Keep each chain's links in order, and stop at the first link whose page is either not OPENED or not found: its words remain unknown."""

    GATE = """Along the way the shopper's request asks for the steps below, and a shopping agent is now about to STAGE. For every step due before this STAGE, judge from the agent's numbered RECORD whether it is done for the cart as it stands. Only a RECORD line that shows the step being carried out makes it done; when the step is about the cart or the order, that line must also follow the most recent cart change, and the reply in it must grant the step. Asking about a product's properties never completes a step.
For a step that is still missing, supply the one store call to make next: a tool taken from TOOLS, with its arguments. Anything for the shopper goes through the messaging tool, as one short plain sentence or question on that step and nothing else, leaving out product properties and values the step does not require. Supply no call if all due steps are done or if the messages or steps that remain cannot carry out the missing step.
Answer with ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [RECORD line numbers showing it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": a change a reply asks for before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement a reply of that sort states]}"""

    SWAP = """A cart line of a shopping agent has just sold out. Search the other listing records shown for listings that could take its place, applying these reading rules: """ + READING + """
Answer with ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": the price shown in its record, as a number} for each listing that looks as if it meets every requirement of the sold-out line's item, at most 6], "reason": "a single sentence"}"""

    SHAPE = """The result below came back from a store tool. Map its parts, naming each by the exact key that the result or the tool's parameters use. Answer with ONE JSON object and nothing else: {"rows": the key under which the listings are listed, "count": the key giving the total number of matching listings (not the number on show), or null, "next": the key whose value fetches the following page, or null, "next_arg": the parameter of the tool that this value goes into, or null, "price": the key within a row that gives the listing's price, or null, "stock": the key within a row that says whether the listing is buyable at this moment, or null}"""


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


def _tokens(text: Any) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower()))


def _same_text(a: str, b: str) -> bool:
    return re.findall(r"[a-z0-9]+", a.lower()) == re.findall(r"[a-z0-9]+", b.lower())


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


def _value_look(values: list[str]) -> tuple[bool, bool]:
    if not values:
        return False, False
    long = sum(len(v.split()) for v in values) / len(values) >= Looks.LONG_VALUE_WORDS
    figures = sum(1 for v in values if re.search(r"\d", v)) / len(values) >= Looks.FIGURE_VALUE_SHARE
    return long, figures


def _ask_odds(values: list[str]) -> float:
    # keyed by (long, figures)
    odds = {(False, False): 1.0, (False, True): 0.6, (True, False): 0.65, (True, True): 0.27}[_value_look(values)]
    return odds if len(set(values)) >= 2 else odds * Looks.SAME_VALUE_ODDS


class Item:
    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Acer:
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
        self.key_looks: dict[str, tuple[bool, bool]] = {}      # attribute key -> (long values, figures), from key_table
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
        self.shown_full: set[str] = set()
        self.audited = False
        self.late_reviewed = False
        self.trusted = False
        self.end_refused = False
        self.refused_held = False
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
        self.carted_at: dict[str, int] = {}
        self.dropped: set[str] = set()
        self.late_held = False
        self.shape_held = False
        self.state_shown = ""
        self.state_at = 0
        self.lead_items: dict[str, list[str]] = {}
        self.lead_version = 0
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
        self.trail_blocks = 0
        self.chain_spent = 0.0
        self.chain_ids: set[str] = set()
        self.chain_end_held = False
        self.searches_made: list[dict[str, Any]] = []
        self.link_tried: set[str] = set()
        self.link_rows: list[str] = []
        self.link_shop = ""
        self.link_opened = 0
        self.trail_state: tuple[Any, ...] | None = None
        self.trail_lost: list[str] = []
        self.last_model = ""
        self.shown_count = 0
        self.last_shown: dict[str, int] = {}
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

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_mark(self) -> str:
        return _dump(sorted(self.cart_lines.items()))

    def page_tools(self) -> set[str]:
        page = self.roles.get("offer") or self.roles.get("open")
        return {t for t in (page, self.roles.get("compare")) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.single_target(t)), None)
        if known:
            return known
        tool = self.roles.get("open")
        return tool if self.facts_from.get(tool or "") is not False and self.single_target(tool) else None

    def compare_facts(self) -> bool | None:
        return self.facts_from.get(self.roles.get("compare") or "")

    def compare_bare(self) -> bool:
        seen = self.compare_facts()
        return seen is False or (seen is None and (self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))))

    def gate_due(self, stage: str) -> bool:
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        return 2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= Caps.ALL_CALLS:
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
                if not passing or tries >= (1 if status in (None, 408) else len(Retry.WAITS)) or self.waited + Retry.WAITS[tries] > Retry.WAIT_TOTAL:
                    break
                pause = Retry.WAITS[tries] * (0.8 + 0.4 * random.random())
                self.waited += Retry.WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model == Models.LIGHT and len(messages) > 1:
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
        result = Wires.llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                self.last_model = model
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
        if Models.HEAVY in models and self.cost + len(prompt) / 4 * Caps.HEAVY_TOKEN > Caps.SPEND_SOFT:
            models = [m for m in models if m != Models.HEAVY] or [Models.LIGHT]
            # a cheap reply that can't be read gets one heavy try while money is left
            if self.cost < Caps.SPEND_HARD:
                models.append(Models.HEAVY)
        for model in models:
            message = self.think([{"role": "user", "content": prompt}], [model], max(max_tokens, Caps.LIGHT_ROOM) if model == Models.LIGHT else max_tokens)
            got = _json_obj((message or {}).get("content"))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= Caps.ALL_CALLS:
                return None
        return None

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = Wires.env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if name == self.roles.get("search") and not (isinstance(seen, dict) and seen.get("error")):
                    self.searches_made.append(dict(args))
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
        prompt = Wording.BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [Models.LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 2500) or brief
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
            fixed = self.ask_json(ask, [Models.LIGHT, Models.HEAVY], 600) or {}
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
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any
            talk = [p["messages"] for p in self.process if p["when"] in ("order", "end") and p["messages"]]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    def unmix(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = (Wording.UNMIX + "\n\nREQUEST:\n" + self.request + f"\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n"
                      + "\n".join(pointers) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 500) or {}
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
        for entry in self.process:
            when = {"choose": "before choosing", "order": "at the order (Acer does it when you check out)", "end": "at an ending without a purchase (Acer does it then)"}[entry["when"]]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get("offer")):
            bare = sorted(t for t in self.page_tools() | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
            pages = sorted(t for t in self.page_tools() if t != facts)
            parts.append(f"listing attributes come from {facts}" + (f"; {', '.join(bare)} show none" if bare else "")
                         + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def replies_left(self) -> int:
        # at least a few: the short round after a refused forced ending goes past the usual count
        replies = Caps.LOOP_CALLS - self.loop_calls
        if self.cost >= Caps.SPEND_SOFT:
            replies = min(replies, int((Caps.SPEND_HARD - self.cost) / Caps.REPLY_COST))
        return max(replies, 2)

    def status(self) -> str:
        left = self.left()
        text = f"[Acer] step {self.step} of {self.max_steps} used, {left} left."
        # on long tasks the replies Acer can afford run out long before the steps, so the pace goes by whichever is nearer
        replies = self.replies_left()
        room = min(left, replies)
        if room < left:
            text += f" Plan by your replies, not the steps: about {room} more are possible."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif room <= 6 or (room <= len(self.items) + 5 and self.buys == 0):
            text += " Little room left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and room <= 12:
            text += " Room is running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and (self.step >= self.max_steps // 2 or replies <= Caps.LOOP_CALLS // 2):
            text += " Half the room is used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        # the state goes along when it changes and every few replies: each call resends all that earlier calls showed, and
        # repeating it on every reply made up a quarter of the history
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        state = ""
        if lookups and self.buys == 0:
            state += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            state += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            state += "\n[changes so far] " + " | ".join(self.notes[-4:])
        state += "\n[ledger]\n" + self.ledger()
        if state != self.state_shown or self.loop_calls - self.state_at >= Views.STATE_EVERY:
            self.state_shown, self.state_at = state, self.loop_calls
            return text + state
        return text

    # ---------- listings ----------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and _id_like(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _learn_id([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.rows):
            # a reply with its own id beside a list of listings, and no text of a listing, is the page of results
            rows = [[r for r in v if isinstance(r, dict) and len(r) >= 3] for v in node.values() if isinstance(v, list)]
            if any(r and _learn_id(r) for r in rows) and not any(isinstance(v, str) and len(v.split()) >= 3 for v in node.values()):
                return None
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
                    self.shown_count += 1
                    self.last_shown[ident] = self.shown_count
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
            for name, slot in group["keys"].items():
                self.key_looks[name] = _value_look(list(slot["per"].values()))
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"] * _ask_odds(list(kv[1]["per"].values()))):
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
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        value = (self.rows.get(ident) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

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
        return bool(keys) and version == self.version and keys <= have and ident not in self.gone

    # ---------- result pages ----------
    def read_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = (Wording.SHAPE + "\n\nTOOL: " + name + " | parameters: " + _dump(self.schemas.get(name) or {})
                  + "\n\nRESULT (first rows only):\n" + _dump(_clip(sample, 120))[:4000])
        got = self.ask_json(prompt, [Models.LIGHT, Models.HEAVY], 1500) or {}
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
        pages = -(-(count - shown) // shown)
        if pages > 2 or self.pages_read + pages > Moves.AUTO_PAGES or self.left() - self.tail() - pages < 12:
            return (f"\n[Acer] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: "
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
            shown_text = self.show(seen, name)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Acer read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Acer] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        if self.wider_held or self.brief.get("pick") != "cheapest" or min(self.left(), self.replies_left()) < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. A search finds only some listings,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches of two to five words for each item under other names for the product (a plain name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
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
        return self.fill_facts(picks[:Moves.CHEAPER_OPENS], self.tail() + 4)

    def observe(self, name: str, observation: Any) -> str:
        self.absorb(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f"(seen on {name}) {meaning}")
        return meaning

    def widen(self) -> None:
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
                # stores limit these calls to 2-6 ids, whatever the schema says; half as many, still taking turns between the items
                lanes = [[i for i in ids if i in picks] for ids in ranked]
                fewer = [lane[k] for k in range(len(picks)) for lane in lanes if k < len(lane)][: max(2, len(picks) // 2)]
                faults = []
                args = _shape({target: fewer}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    def widen_singly(self) -> None:
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return
        # turns between the item groups, up to two each; with a cheapest pick the cheapest buyable rows first
        lanes: list[list[str]] = []
        for ids in self.item_groups():
            opened = sum(1 for i in ids if i in self.full)
            pool = [i for i in ids if i not in self.full and self.buyable(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            lanes.append(pool[: max(0, min(2, 8 - opened))])
        queue = [lane[k] for k in range(2) for lane in lanes if k < len(lane)]
        # while steps are plentiful: 22 stay for the rest of the episode
        self.fill_facts(queue[:5], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= 8:
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
        if count <= 0:
            return ""
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
        # one pick more than the messages allow, in case one can't be asked
        # only that many questions go out in all, so an item's queue needs its share and one more
        share = max(Moves.PLAN_PICKS, min(max(1, count), (count + len(self.items) - 1) // max(1, len(self.items)) + 1))
        prompt = Wording.PLAN.replace("COUNT", str(share)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 3500) or {}
        ordered: list[tuple[int, str]] = []
        kept_in_place: set[tuple[int, str]] = set()
        asked_for = {_words(x) for x in first}
        for pick in _items(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _count(pick.get("item") or 1) or 1), len(self.items))
            # one property per question: a label naming two is cut to the first
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(_words(label) != _words(other) for n, other in ordered if n == number):
                ordered.append((number, label))
                if str(pick.get("hint") or "").strip() or _words(label) in asked_for:
                    kept_in_place.add((number, label))
        # a key whose values are long figures drew a preference least often (0 of 102 questions in race 11): unless the
        # story hints at it or the request asks about it, it waits behind every other pick
        late = [pair for pair in ordered if pair not in kept_in_place and self.key_looks.get(_words(pair[1])) == (True, True)]
        ordered = [pair for pair in ordered if pair not in late] + late
        # one queue per item: the planner's picks for it, then the item's other listed keys, so no item runs out of questions
        # while messages remain
        lanes: dict[int, list[str]] = {}
        for number, label in ordered:
            lanes.setdefault(number, []).append(label)
        wanted = [(_words(item.shelf or ""), _words(item.name), {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}) for item in self.items]
        for label, name, carried, values in table:
            for number, (shelf, item_name, long) in enumerate(wanted, 1):
                if not (_words(label) in (shelf, item_name) or {w for w in _stem(label) if len(w) >= 4} & long):
                    continue
                item = self.items[number - 1]
                settled = {_words(m.split(":")[0]) for m in item.musts + item.told}
                lane = lanes.setdefault(number, [])
                if len(lane) < count + Moves.ASK_NAMES and _words(name) not in settled and all(_words(name) != _words(x) for x in lane) and self.key_looks.get(_words(name)) != (True, True):
                    lane.append(name)
        order = list(dict.fromkeys(n for n, _ in ordered))
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        self.trace.append({"kind": "plan", "ask": _clip(plan.get("ask"), 120), "late": late, "lanes": {n: lane[:6] for n, lane in lanes.items()}})
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        lists_ok = True
        # one message at a time, its answer read before the next: an item with no preference found yet comes first, then the
        # one asked least, then the planner's order (an answer of no preference leaves its item at the front for its next key)
        while self.msgs_free() > 0 and not self.finished and self.left() > 6:
            open_items = [n for n, lane in lanes.items() if lane]
            if not open_items:
                break
            number = min(open_items, key=lambda n: (found[n] > 0, spent[n], order.index(n) if n in order else len(order)))
            names = lanes[number][:Moves.ASK_NAMES if lists_ok else 1]
            batch = self.send([(number, names[0], self.question(number, names))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            known = len(self.items[number - 1].told)
            # an answer that carries a need names the asked property or product (95% of 6,498, races 1-12) or holds a figure, and 98%
            # of the stock lines do neither: the heavy model reads the first kind and the light one the rest, which catches the others
            models = (Models.HEAVY, Models.LIGHT) if self.names_asked(batch[0]) else (Models.LIGHT, Models.HEAVY)
            heard = self.hear(batch, models)
            if len(names) > 1:
                # "I don't mind" after a list doesn't say which of its properties that covers
                heard.pop("free", None)
            self.take_heard(heard, told)
            if len(self.items[number - 1].told) > known:
                found[number] += 1
            if len(names) > 1 and _items(heard.get("crowded")):
                # this shopper wants one thing at a time: the same first property goes alone next, and so does every later one
                lists_ok = False
            else:
                lanes[number].pop(0)
            # a property an answer has settled isn't named again, under any spelling
            said = [_tokens(need.split(":")[0]) for need in self.items[number - 1].told]
            lanes[number] = [x for x in lanes[number] if not any(k and (k <= _tokens(x) or _tokens(x) <= k) for k in said)]
        if not sent:
            return ""
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Acer's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else "none")
        summary += ". Apply them to every pick (search again with the new values where needed)."
        # most listings were read before the answers came: what they already show about the new requirements
        fresh = [(n, need) for n, need in self.needs_now() if any(need in t for t in told)]
        shown = self.evidence_of([i for i in reversed(self.seen_order) if i in self.full and not self.sold_out(i)], fresh)
        if shown:
            summary += "\nListings already read: " + shown
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if self.chain_pending():
            walked, _ = self.trail(self.chain_spare())
            if walked:
                summary += "\n" + walked.rstrip()
            pointed = [need for need in pointed if need not in self.ref_values]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    def names_asked(self, sent: tuple[int, str, str]) -> bool:
        number, text, answer = sent
        asked = {w for w in (_tokens(text) - _tokens("what would you like for the")) | _tokens(self.items[number - 1].name) if len(w) >= 3}
        return bool(asked & _tokens(answer)) or bool(re.search(r"\d", answer))

    def question(self, number: int, names: list[str]) -> str:
        item = self.items[number - 1].name
        if len(names) == 1:
            return f"What {names[0]} would you like for the {item}?"
        return f"For the {item}, what {', '.join(names[:-1])} or {names[-1]} would you like?"

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
            # a change line that comes with an answer goes the way every other change line goes (budget, sold out and the
            # rest), and the answer is read without it
            if isinstance(answer, dict):
                said = {k: v for k, v in answer.items() if not (isinstance(v, list) and v and all(isinstance(x, str) for x in v))}
                extra = [x for v in answer.values() if v and isinstance(v, list) and all(isinstance(x, str) for x in v) for x in v]
                if extra and any(isinstance(v, str) and len(v.split()) >= 2 for v in said.values()):
                    fresh = [line for line in extra if line not in self.heard]
                    if fresh:
                        meaning = self.take_note(fresh)
                        if meaning:
                            self.queued.append(f"(seen on {tool}) {meaning}")
                    answer = said
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append(f"step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def plain_need(self, number: int, need: str) -> str:
        # shoppers say "closure of the pumps": attribute keys are named after the property alone
        prop, sep, value = need.partition(":")
        tail = re.search(r"\s+(?:of|for|on)\s+(?:the\s+)?(\S.*)$", prop)
        item = self.items[number - 1]
        if sep and tail and _tokens(tail.group(1)) and _tokens(tail.group(1)) <= (_tokens(item.name) | _tokens(item.shelf or "")):
            return prop[:tail.start()].strip() + sep + value
        return need

    def take_heard(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in _items(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.items[min(max(1, _count(entry.get("item")) or 1), len(self.items)) - 1]
                gone = _words(str(entry["property"]).split(":")[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and _words(m.split(":")[0]) == gone]
                item.dropped.extend(hit)
                told.extend(f"dropped {m}" for m in hit)
        for entry in _items(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
            need = self.plain_need(number, str(entry["need"]).strip())
            if not any(_same_text(need, known) for known in self.items[number - 1].told):
                self.items[number - 1].told.append(need)
                told.append(f"item {number}: {need}" if many else need)
            if entry.get("points") is True and not any(n == number and _same_text(need, known) for n, known in self.refs):
                self.refs.append((number, need))
        for entry in _items(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _count(entry.get("item")) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry["property"]))
        if isinstance(heard.get("budget"), (int, float)) and not isinstance(heard.get("budget"), bool) and heard["budget"] > 0 and heard["budget"] != self.budget:
            self.budget = heard["budget"]
            self.version += 1
            self.recent = []

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (Models.HEAVY, Models.LIGHT)) -> dict[str, Any]:
        prompt = Wording.HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        unit = self.brief.get("budget_unit") or ""
        prompt += "\n\nCURRENT BUDGET: " + (f"{self.budget} {unit}".strip() if self.budget is not None else "none stated")
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.ask_json(prompt, list(models), 2500) or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.gate_due(stage):
            return True, ""
        ending = {self.roles.get("buy"), self.roles.get("refuse")} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools if t["function"]["name"] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(Moves.GATE_ROUNDS):
            prompt = (Wording.GATE.replace("STAGE", "place the order" if stage == "order" else "end without a purchase")
                      + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE NOW:\n" + "\n".join(f"- {p['step']}" for p in steps)
                      + "\n\nLEDGER:\n" + self.ledger() + "\n\nCART NOW: " + self.cart_text()
                      + "\n\nRECORD (messages and step calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.done_log, 1)) or "(nothing yet)")
                      + ("\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT] if nudged else [Models.LIGHT, Models.HEAVY], 2000)
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
                return False, ("Before going on, Acer did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
                self.done_log.append(f"(Acer: no line made since the cart last changed at step {self.cart_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200]
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
            self.observe(name, seen)
            self.done_log.append(f"step {self.step}: {name} {_dump(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})")
            self.done_steps.append(self.step)
            said.append(f"{name} {_dump(args)} -> {self.show(seen)[:400]}")
        return True, ("Before going on, Acer did the steps the request asks for:\n" + "\n".join(said)) if said else ""

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
        prompt = Wording.NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [Models.LIGHT, Models.SPARE], 1200) or {}
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
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

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
        return self.evidence_of(ids, self.needs_now())

    def evidence_of(self, ids: list[str], needs: list[tuple[int, str]]) -> str:
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
        return "lines by Acer's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def check_cart(self, ids: list[str] | None = None, limit: float = Caps.SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        self.checked.add(key)
        prompt = Wording.CHECK + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + self.chain_text(self.chain_pages())
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 1500)
        self.trace.append({"kind": "check", "cart": ids, "got": got})
        self.verdicts[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _items(got.get("problems"))][:5]
        better = [str(b) for b in _items(got.get("better")) if str(b) in self.rows and not self.sold_out(str(b))][:4]
        text = "A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Listings that look better: " + ", ".join(better)
        return text + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def chain_pending(self) -> list[str]:
        live = {need for _, need in self.needs_now()}
        return [need for _, need in self.refs if need in live and need not in self.ref_values]

    def chain_pages(self) -> list[str]:
        # while a pointed-at value is still open, the pages followed so far (the newest, which hold the value): a review that cannot see them calls the requirement unreadable
        if not self.chain_pending():
            return []
        return [i for i in self.seen_order if i in self.chain_ids and i in self.opened and (self.full.get(i) or self.rows.get(i))][-Moves.CHAIN_SHOWN:]

    def chain_text(self, pages: list[str]) -> str:
        if not pages:
            return ""
        records = [_dump(_clip(self.full.get(i) or self.rows.get(i), 350))[:1800] for i in pages]
        return "\n\nPAGES THE AGENT OPENED WHILE FOLLOWING A REQUIREMENT THAT POINTS AT ANOTHER LISTING (what they say about that requirement counts; they can be on any shelf at any price):\n" + "\n".join(records)

    def chain_spare(self) -> int:
        # what buying still takes once the value is known: a search and a read per item, the cart, the checkout
        return 6 + 2 * len(self.items)

    def link_search(self, words: str) -> bool:
        # a chain's next listing, searched the way the agent's own word searches went. Races differ: a required mode, a
        # shelf name beside the words, a text page token. True when the search showed any listing
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        paging = (self.shapes.get(tool or "") or {}).get("next_arg")
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v and k != paging]
        modes = [k for k, v in props.items() if isinstance(v, dict) and "enum" in v and k in required]
        cat_key = self.category_key()
        shelves = {_words(str(r.get(cat_key))) for r in self.rows.values() if r.get(cat_key)} if cat_key else set()

        def word_arg(args: dict[str, Any]) -> str | None:
            keys = [k for k in texts if isinstance(args.get(k), str) and args[k].strip() and (len(texts) == 1 or _words(args[k]) not in shelves)]
            return max(keys, key=lambda k: (k in required, len(args[k].split())), default=None)

        usable = [(a, word_arg(a)) for a in self.searches_made]
        usable = [(a, k) for a, k in usable if k]
        if not tool or not usable or len(self.link_tried) >= Moves.CHAIN_SEARCHES:
            return False
        self.link_tried.add(words.lower())
        used = [tuple(str(a.get(m)) for m in modes) for a, _ in usable]
        usual = max(set(used), key=used.count)
        shaped, key = next((a, k) for a, k in reversed(usable) if tuple(str(a.get(m)) for m in modes) == usual)
        # when a required argument picks the mode, any other fixed choice in that call was a filter
        call = {k: v for k, v in shaped.items() if k == key or k in required or (not modes and "enum" in (props.get(k) or {}))}
        call[key] = words
        faults: list[str] = []
        call = _shape(call, schema, "args", faults)
        if faults:
            return False
        before = self.shown_count
        reply = self.act(tool, call)
        if reply is None or self.finished:
            return False
        self.observe(tool, reply.get("observation"))
        self.trace.append({"kind": "link_search", "words": words, "shown": self.shown_count - before})
        if self.shown_count > before:
            self.link_rows = sorted((i for i, n in self.last_shown.items() if n > before), key=lambda i: self.last_shown[i])
            self.link_opened = 0
        return self.shown_count > before

    def next_sibling(self) -> str | None:
        # the opened listing leads nowhere: another listing of that shop from the same search may be the link
        if self.link_opened >= Moves.CHAIN_SIBLINGS or not self.link_shop:
            return None
        return next((i for i in self.link_rows if i not in self.opened and self.link_shop in _dump(self.rows.get(i) or {}).lower()), None)

    def trail(self, spare: int = 3, gate: bool = False) -> tuple[str, bool]:
        # in race 11, agents told such a need passed 24 of 28 episodes after opening the whole chain and 0 of 60 otherwise,
        # with steps to spare and with 38 of those 60 giving up, so the chain is walked here rather than left to the model
        pending = self.chain_pending()
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost = list(self.trail_lost) if pending else []
        while (pending and self.trails < Moves.CHAIN_ROUNDS and self.chain_spent < Caps.CHAIN_SPEND and self.cost < Caps.SPEND_HARD and not self.finished
               and self.left() > spare):
            # nothing opened or shown since the last reading: the same records give the same answer
            state = (len(self.opened), self.shown_count, len(self.full), tuple(pending))
            if state == self.trail_state:
                break
            self.trails += 1
            records, room = [], 9000
            # the chain's own pages first, then the others by how recently they were shown
            pages = sorted((i for i in self.seen_order if i in self.opened), key=lambda i: (i not in self.chain_ids, -self.last_shown.get(i, 0)))
            for ident in pages:
                # an opened page is the page view's record (with the attributes when they were read too)
                record = self.full.get(ident) or self.rows.get(ident)
                if record and room > 0:
                    records.append("OPENED " + _dump(_clip(record, 350))[:1800])
                    room -= len(records[-1])
            # the rows shown most recently, so a listing the last search found again is among them
            rows = sorted((i for i in self.seen_order if i not in self.opened and self.rows.get(i)), key=lambda i: -self.last_shown.get(i, 0))[:15]
            records += ["ROW " + _dump(_clip(self.rows.get(i), 80))[:600] for i in rows]
            prompt = Wording.TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS:\n" + "\n".join(records)
            before = self.cost
            got = self.ask_json(prompt, [Models.LIGHT], 1800) or {}
            self.chain_spent += self.cost - before
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150), "model": self.last_model, "spent": round(self.cost - before, 5)})
            if not got:
                break
            self.trail_state = state
            reach: list[str] = []
            finds: list[str] = []
            ends: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                self.chain_ids.update(i for i in ids if i)
                # a chain starts at the shop the requirement names: a value reached from elsewhere is not this requirement's
                start = str(links[0].get("short") or "").split()[:1] if links else []
                anchored = not start or (start[0].lower() in need.lower() and start[0].lower() in _dump(self.rows.get(ids[0]) or {}).lower())
                if chain.get("done") is True and anchored and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                    continue
                absent = next((l for l, i in zip(links, ids) if i not in self.rows), None)
                if absent:
                    about = str(absent.get("about") or "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
                    # the store wants every word to match, so an empty search falls back to fewer words: shop and kind, then shop
                    short = str(absent.get("short") or "").strip()
                    self.link_shop = short.split()[0].lower() if short else self.link_shop
                    tries = [str(absent.get("search") or "").strip(), short, short.split()[0] if short else ""]
                    finds += [w for w in tries if w and w.lower() not in self.link_tried and w not in finds]
                elif ids:
                    # every link is open and none states the value: it may be among the last page's attributes
                    ends.append(ids[-1])
                    short = str(links[-1].get("short") or "").strip()
                    self.link_shop = short.split()[0].lower() if short else self.link_shop
                    lost.append(f"{need} -> find the listing its last page points at")
            pending = [p for p in pending if p not in self.ref_values]
            self.trail_lost = list(lost)
            moved = False
            if tool and target:
                for ident in reach[: max(0, min(3, 8 - len(seen)))]:
                    faults: list[str] = []
                    args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                    if faults or self.left() <= spare:
                        break
                    reply = self.act(tool, args)
                    if reply is None or self.finished:
                        break
                    self.observe(tool, reply.get("observation"))
                    seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
                    moved = True
            for words in finds:
                if moved or self.left() <= spare or self.finished:
                    break
                moved = self.link_search(words)
            if not moved and ends:
                moved = bool(self.fill_facts(ends, spare))
            if not moved and ends and tool and target and self.left() > spare:
                sibling = self.next_sibling()
                faults = []
                args = _shape({target: sibling}, self.schemas.get(tool), "args", faults) if sibling else {}
                if sibling and not faults:
                    self.link_opened += 1
                    reply = self.act(tool, args)
                    if reply is not None and not self.finished:
                        self.observe(tool, reply.get("observation"))
                        seen.append(f"{sibling}: " + self.show(reply.get("observation"))[:1500])
                        moved = True
            if not moved and ends and self.link_shop and self.left() > spare and self.link_shop not in self.link_tried:
                # no other listing of that shop on offer: the shop's name alone shows them all
                moved = self.link_search(self.link_shop)
            if not moved:
                break
        text = ""
        if seen:
            text += "[Acer opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(f"{k} -> {v}" for k, v in self.ref_values.items()) + "\n"
        hold = gate and bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_tool(self) -> str | None:
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < max(since, self.carted_at.get(i, -99)) and i not in self.gone]
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
        return ident in self.unavailable or ident in self.gone

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.gone and self.sold_out(i)]
        if not lost:
            return ""
        for ident in lost:
            self.gone[ident] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        text = f"[Acer] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        twins = self.twins(lost)
        if twins and self.left() > 6:
            text += f" Nearly the same listing, worth opening before anything else: {', '.join(twins)}."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.gone]
        if any(self.sold_out(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ""
        return self.reread_offers() if ids else ""

    def review_order(self) -> list[str]:
        # a window over the newest first-seen listings dropped pages read early (the right item on 103 of 240 reviews, races 3-11)
        wanted: set[str] = set()
        for _, need in self.needs_now():
            wanted |= _tokens(need.split(":", 1)[-1])

        def carried(ident: str) -> int:
            return 0 if ident in self.full else len(wanted & _tokens(" ".join(str(v) for v in (self.rows.get(ident) or {}).values() if isinstance(v, str))))

        return sorted(self.seen_order, key=lambda i: (i not in self.full, -carried(i), -self.last_shown.get(i, 0)))

    def twins(self, lost: list[str]) -> list[str]:
        # the same product under another listing: where one was left in the store after a sell-out, a passer bought it on 15 of 24 give-ups (4 races)
        out = []
        shown: dict[str, list[str]] = {}
        for row in self.rows.values():
            for k, v in row.items():
                if isinstance(v, str):
                    shown.setdefault(k, []).append(v)
        own = {k for k, v in shown.items() if len(set(v)) >= 0.5 * len(v)}
        for gone in lost:
            texts = {k: _tokens(v) for k, v in (self.rows.get(gone) or {}).items() if k in own and isinstance(v, str) and len(v.split()) >= 3}
            for ident in reversed(self.seen_order):
                row = self.rows.get(ident) or {}
                if ident in lost or ident in out or self.sold_out(ident) or ident in self.cart_ids():
                    continue
                if any(len(words & _tokens(row[k])) >= 0.8 * max(len(words), len(_tokens(row[k]))) for k, words in texts.items() if isinstance(row.get(k), str)):
                    out.append(ident)
        return out[:3]

    def replacements(self, lost: list[str]) -> list[str]:
        if self.cost > Caps.SPEND_SOFT:
            return []
        records = []
        for ident in self.review_order():
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = Wording.SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (pages read first):\n" + "\n".join(records)
        got = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 2500)
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get("pick") == "cheapest" else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        return ranked[:4]

    def audit(self) -> str:
        if self.audited or self.cost > Caps.SPEND_HARD:
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
        self.lead_version = self.version
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def audit_call(self, opened_only: bool = False) -> dict[str, Any] | None:
        records = []
        chain = self.chain_pages()
        for ident in self.review_order():
            if ident in chain or self.sold_out(ident) or self.out_of_reach(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            page = not record and ident in self.opened
            if opened_only and not record and not page:
                continue
            if record:
                records.append("OPENED " + _dump(_clip(record, 200))[:1200])
            elif page:
                records.append("PAGE " + _dump(_clip(self.rows.get(ident), 200))[:1200])
            else:
                records.append("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records and not chain:
            return None
        prompt = Wording.AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger() + self.chain_text(chain)
        prompt += "\n\nLISTING RECORDS SEEN (pages read first):\n" + "\n".join(records)
        # (the heavy model's thinking ran out of 1,500 tokens on 2 of 5 of these; on real requests it read them better than the light one)
        got = self.ask_json(prompt, [Models.HEAVY, Models.LIGHT], 2500)
        self.trace.append({"kind": "audit", "got": got})
        return got

    def late_review(self) -> None:
        self.late_reviewed = True
        self.review_picks()

    def review_picks(self) -> bool:
        # one review of the opened listings; when it names a fit for every item, those become the picks the ending path
        # buys (complete_cart with trust)
        if self.cost >= Caps.SPEND_HARD + 0.01:
            return False
        named = self.audit_fits(self.audit_call(opened_only=True))
        named = {k: [i for i in v if i in self.full or i in self.opened] for k, v in named.items()}
        named = {k: v for k, v in named.items() if v}
        self.trace.append({"kind": "late_review", "fits": named})
        if not named or len(named) < max(1, len(self.items)):
            return False
        self.lead_items = named
        self.lead_version = self.version
        self.leads = [i for ids in named.values() for i in ids]
        return True

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
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

    def cart_mode(self, name: str, key: str, job: str) -> Any:
        # the task sheet says which option adds and which takes out; the listed order stands in only when it doesn't
        options = ((self.schemas.get(name, {}).get("properties") or {}).get(key) or {}).get("enum") or []
        sheet = self.brief.get("cart_modes") if isinstance(self.brief.get("cart_modes"), dict) else {}
        add, take = sheet.get("add"), sheet.get("remove")
        if add in options and take in options and add != take:
            return add if job == "add" else take
        return options[0 if job == "add" else 1]

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
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
                calls.append({target: ident, mode: self.cart_mode(name, mode, "remove"), **({count: have} if count else {})})
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
                # a verb cart adds and removes units; without a verb nothing shows whether a count adds or sets
                if not mode or not count:
                    return None
                calls.append({target: ident, mode: self.cart_mode(name, mode, "add" if want > have else "remove"), count: abs(want - have)})
                continue
            calls.append({target: ident, **({mode: self.cart_mode(name, mode, "add")} if mode else {}), **({count: want} if count else {})})
        return calls

    def fits_budget(self, lines: dict[str, int]) -> bool:
        # True when a price or the budget is unknown: nothing shows the cart breaks it
        if not isinstance(self.budget, (int, float)) or isinstance(self.budget, bool):
            return True
        prices = {i: self.price_of(i) for i in lines}
        if None in prices.values():
            return True
        if self.brief.get("budget_scope") == "each":
            return all(p <= self.budget for p in prices.values())
        return sum(p * lines[i] for i, p in prices.items()) <= self.budget

    def complete_cart(self, trust: bool = False) -> str:
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if (self.trusted if trust else self.completed) or not cart or not buy:
            return ""
        if not trust and (not self.confirmed or self.sold_out(self.confirmed)):
            return ""
        if trust:
            self.trusted = True
        else:
            self.completed = True
        cat_key = self.category_key()

        def kind(ident: str) -> str:
            return str((self.rows.get(ident) or {}).get(cat_key) or "").strip().lower() if cat_key else ident

        if trust and self.chain_pending():
            self.trail(self.tail() + 2 * len(self.lead_items))
        if trust and self.lead_version != self.version:
            # picks named before a change (or before a chain's value) may no longer fit
            self.lead_items = {}
            self.leads = []
            self.review_picks()
        if trust:
            rows = [i for ids in self.lead_items.values() for i in ids[:1] if i not in self.full and not self.sold_out(i)]
            if rows:
                self.open_rows(rows)
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = _count(number)
            pick = next((i for i in ids if i not in plan and not self.sold_out(i) and (not trust or i in self.full or i in self.opened)), None)
            if pick and index and 1 <= index <= len(self.items):
                plan[pick] = self.items[index - 1].qty
        if (not trust and self.confirmed not in plan) or (not plan and len(self.cart_ids()) < len(self.items)):
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if calls is None or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to the hard spend cap
        if not trust:
            self.fill_facts(list(lines), len(calls) + 3)
            self.check_cart(list(lines), Caps.SPEND_HARD)
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
        needs_then = self.needs_now()
        budget_then = self.budget
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check;
        # with no check to rely on, one that changed the requirements or leaves the cart over budget goes back to the model
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                if not trust:
                    self.check_cart(list(lines), Caps.SPEND_HARD)
                if ((not trust and not self.verdicts.get(self.check_key(list(lines)))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines) or (trust and (self.needs_now() != needs_then or not self.fits_budget(lines) or (self.budget != budget_then and any(self.price_of(i) is None for i in lines))))):
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
                self.end_refused = True
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
        if not replies:
            return ""
        why = ("the cart holds a line for every item and ending without a purchase rarely passes" if not plan
               else "the review named a listing that fits for every item" if trust
               else "the independent check confirmed " + self.confirmed)
        return ("Not run (Acer acted instead): " + why + (", so Acer set the cart (" + "; ".join(done) + ") and checked out" if done else ", so Acer checked out the cart as it stood")
                + ". Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------
    def cart_shape(self) -> str:
        # failed orders one plain fix from a passer's cart, on every race: a line missing or extra, or a wrong count
        lines = self.cart_lines
        if not lines or not all(isinstance(n, int) for n in lines.values()):
            return ""
        units, wanted = sum(lines.values()), sum(item.qty for item in self.items)
        if len(lines) == len(self.items) and units == wanted:
            return ""
        asked = "; ".join(f"{item.qty} of {item.name}" for item in self.items)
        return f"By Acer's record the cart holds {len(lines)} line(s) and {units} unit(s), and the request asks for {asked}. Fix the cart if that is a mistake; otherwise check out again."

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
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _kinds(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            named = any(ident in text for ident in self.rows) or self.brief.get("quiet_decline") is not False
            if named or not text:
                out[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
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
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    def whole_line(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # stores subtract a removal's count (add 2, remove 1, and the line holds 1), so a count of 0 would leave the line in the cart
        spec = (self.schemas.get(name) or {}).get("properties") or {}
        if name != self.roles.get("cart") or not any(isinstance((spec.get(k) or {}).get("enum"), list) and v in spec[k]["enum"] and v != self.cart_mode(name, k, "add") for k, v in args.items()):
            return args
        have = next((self.cart_lines[v] for v in args.values() if isinstance(v, str) and v in self.cart_lines), 1)
        return {k: (have if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0 else v) for k, v in args.items()}

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options and value != self.cart_mode(name, key, "add"):
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
        for ident, _ in lines:
            # a read from before the cart call says nothing about the line now: it may sell out right after it is carted
            self.carted_at[ident] = self.step
            self.view_keys.pop(ident, None)
            self.read_cache = {mark: got for mark, got in self.read_cache.items() if ident not in mark}
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_sets(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - (count if count else 1)
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def show(self, observation: Any, tool: str = "") -> str:
        # only a results page is cut to row length: a side-by-side read is a list of records too, and its tables and notes are what it was opened for
        page = tool == self.roles.get("search") and isinstance(observation, dict) and any(isinstance(v, list) and len(v) >= 3 and all(isinstance(r, dict) for r in v) for v in observation.values())
        if page:
            observation = self.fade_rows(observation)
        text = _dump(_clip(observation, Views.ROW_CHARS if page else Views.SHOW_CHARS))
        if len(text) > Views.OBS_LIMIT:
            text = text[:Views.OBS_LIMIT] + "...(cut)"
        return text

    def fade_rows(self, observation: dict[str, Any]) -> dict[str, Any]:
        # every later call resends a page: a row shown in full before, or one that cannot be the purchase, keeps its fields but only
        # short texts. Done as a page first arrives, because rewriting older messages would lose the prompt cache
        out: dict[str, Any] = {}
        for key, value in observation.items():
            if not (isinstance(value, list) and value and all(isinstance(r, dict) for r in value)):
                out[key] = value
                continue
            rows = []
            for row in value:
                ident = self.ident(row, True)
                if ident and (ident in self.shown_full or self.out_of_reach(ident)):
                    rows.append(_clip(row, Views.FADED_CHARS))
                else:
                    rows.append(row)
                    if ident:
                        self.shown_full.add(ident)
            out[key] = rows
        return out

    def out_of_reach(self, ident: str) -> bool:
        # dearer than the whole budget, or on a shelf no item names. A pointer requirement can lead to any shelf or price, so it turns
        # this off; so does a budget no seen listing meets or an item no shelf lines up with: there the reading may be off, not the store
        record = self.rows.get(ident)
        if self.refs or not record:
            return False
        price = self.price_of(ident)
        if isinstance(self.budget, (int, float)) and price is not None and price > self.budget:
            if any((self.price_of(i) or self.budget + 1) <= self.budget for i in self.rows):
                return True
        cat_key = self.category_key()
        label = str(record.get(cat_key) or "") if cat_key else ""
        if not label:
            return False

        def named(item: Item, text: str) -> bool:
            own = {w for w in _stem(item.shelf or "") | _stem(item.name) if len(w) >= 4}
            return _words(text) in (_words(item.shelf or ""), _words(item.name)) or bool({w for w in _stem(text) if len(w) >= 4} & own)

        if any(named(item, label) for item in self.items):
            return False
        labels = {str(r.get(cat_key) or "") for r in self.rows.values()} - {""}
        return all(any(named(item, other) for other in labels) for item in self.items)

    def compare_target(self) -> str | None:
        schema = self.schemas.get(self.roles.get("compare") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _kinds(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def compare_room(self) -> int:
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                out.append((cid, "Shown in the side-by-side result above: Acer opened these listings together in one step."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.finished:
                out.append((cid, "Not run: the episode is over."))
                continue
            if name == self.roles.get("ask"):
                out.append((cid, "Not run (no step used): Acer's interviewer handles every message to the shopper."))
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
            args = self.whole_line(name, args)
            if wrong:
                out.append((cid, "Not run (no step used): " + wrong + "."))
                continue
            trimmed = ""
            many = self.compare_target() if name == self.roles.get("compare") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Acer] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
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
                    trimmed += f"\n[Acer] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _dump(args)
            if ending and self.left() > 2:
                if self.is_buy(name, args):
                    if not self.shape_held:
                        shape = self.cart_shape()
                        if shape:
                            self.shape_held = True
                            out.append((cid, "Not run yet (no step used). " + shape + "\nCart now: " + self.cart_text()))
                            continue
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
                    # the checkout is held back at most twice for a sold-out line, so a wrong cart record cannot stall the episode
                    held = [i for i in self.cart_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, "Not run yet (no step used). " + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + "\nCart now: " + self.cart_text()))
                        continue
                    known = len(self.ref_values)
                    chain, hold = self.trail(gate=True)
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    if len(self.ref_values) > known:
                        out.append((cid, "Not run yet. " + chain + "Apply that value to the cart if it matters, then check out again.\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    verdict = self.check_cart()
                    if verdict:
                        seen = ("Acer read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    if reread and self.cost > Caps.SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Acer re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    if self.chain_pending() and not self.chain_end_held:
                        self.chain_end_held = True
                        # this late the picks are read already: buying takes a cart call per item and the checkout
                        walked, _ = self.trail(1 + len(self.items))
                        if walked:
                            out.append((cid, "Not run yet. " + walked.rstrip() + "\nA requirement takes its value from those pages: apply it, put the best fitting listing for each item in the cart and check out; end without a purchase only if every listing clearly fails."))
                            continue
                    if self.end_refused and not self.refused_held:
                        # giving up after an ending drew a change line passed 0 of 193 times across races 3-11, and buying 730 of 1,157
                        self.refused_held = True
                        if not self.lead_items:
                            self.review_picks()
                        picks = "; ".join(f"item {k}: {', '.join(v)}" for k, v in self.lead_items.items())
                        out.append((cid, "Not run yet (no step used). Your last ending call did not end the task: it brought a change (the ledger shows it), and the next ending call is likely the last. Apply the change, put the best fitting listing for each item in the cart"
                                    + (f" (a review of the opened listings suggests {picks})" if picks else "") + " and check out."))
                        continue
                    if self.left() <= 5 and not self.late_reviewed and not self.lead_items:
                        self.late_review()
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
                            self.check_cart([first], Caps.SPEND_HARD)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and not self.late_held:
                # a decline with a full cart usually throws it away: on 51 of 276 such endings across races 3-11 it was the very cart a passer bought
                lines = [i for i in self.built if not self.sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not fresh)]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.left() and (not missing or self.left() <= 5):
                    self.late_held = True
                    self.trace.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else "Check out the cart as it is."
                    out.append((cid, "Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                                + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again."))
                    continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3 and self.replies_left() > 6:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try queries of two to five words (the value, then the product's plain name) and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if not done and self.lead_items:
                    done = self.complete_cart(trust=True)
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.is_buy(name, args) and self.queued and self.left() >= 2:
                news = " | ".join(self.queued)
                self.queued = []
                out.append((cid, "Not run yet. Acer's own reads just showed: " + news + "\nApply it if it matters, then check out again.\nCart now: " + self.cart_text()))
                continue
            if ending and self.left() >= 2:
                go, said = self.gate("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Acer] " + said + trimmed
            if ending and self.is_buy(name, args) and self.queued and self.left() >= 2:
                # a change seen on Acer's own calls (a re-read, a chain, attributes, the gate) has not reached the model yet
                news = " | ".join(self.queued)
                self.queued = []
                out.append((cid, "Not run yet. Acer's own reads just showed: " + news + "\nApply it if it matters, then check out again.\nCart now: " + self.cart_text()))
                continue
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if ending and not self.finished and not reply.get("finished"):
                self.end_refused = True       # the ending brought a change line instead (giving up after that passed 0 of 193 times, races 3-11)
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f"\n[Acer] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
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
            text = self.show(observation, name) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Acer] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if not reply.get("finished") and not self.finished and (self.note_sold or (self.is_buy(name, args) and any(line in self.spoken(observation) for line in self.sold_lines))):
                seen = self.find_sold()
                if seen:
                    text += "\n[Acer re-read the cart's lines after that note]\n" + seen
            swap = self.swap_hint()
            if swap:
                text += "\n" + swap
            if self.queued:
                text += "\n[notes seen on Acer's own calls] " + " | ".join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        if self.finished or self.broke or self.left() <= 0:
            return ""
        if self.chain_pending() and self.left() > 3:
            self.trail(self.tail() + max(0, len(self.items) - len(self.cart_ids())))
            if self.finished or self.left() <= 0:
                return ""
        buy, refuse = self.roles.get("buy"), self.roles.get("refuse")
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
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
                    self.end_refused = True
                    self.refused_at = self.step
                    self.observe(buy, reply.get("observation"))
                    return "the checkout was refused: " + self.show(reply.get("observation"))
        if refuse:
            # an ending the code forces (turn, spend or time limit) gets the same last look as one the model asks for
            if not self.lead_items:
                self.review_picks()
            if self.lead_items:
                bought = self.complete_cart(trust=True)
                if bought or self.finished:
                    return "" if self.finished else bought
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
                    self.end_refused = True
                    self.observe(refuse, reply.get("observation"))
                    return "ending without a purchase was refused: " + self.show(reply.get("observation"))
        return ""

    def trim(self) -> None:
        size = sum(len(_dump(m.get("content"))) for m in self.chat)
        if size <= Views.HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get("role") == "tool"]
        for index in tools[: max(1, len(tools) // 2)]:
            content = self.chat[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"
                self.shown_full.clear()

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
        self.chat = [{"role": "system", "content": Wording.GUIDE}, {"role": "user", "content": intro}]
        self.work(Caps.LOOP_CALLS, Caps.SPEND_HARD)
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Acer] Acer tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, Caps.SPEND_HARD + 0.01)
            if not self.finished:
                self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap and self.cost < spend_cap):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            self.trim()
            self.loop_calls += 1
            message = self.think(self.chat, [Models.LIGHT, Models.SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= Caps.ALL_CALLS or self.misses > 2:
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
        episode = Acer(problem_data)
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
