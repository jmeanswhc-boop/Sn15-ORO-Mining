from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

def cushion(fallback: Any) -> Any:
    """A side step. On failure, record the original step name and keep the episode going."""

    def wrap(method: Any) -> Any:
        origin = ORIGIN.get(method.__name__, method.__name__)

        def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as exc:
                self.spine.vault.scroll.leaf.trace.append({'kind': 'soft_crash', 'where': origin, 'error': f'{type(exc).__name__}: {exc}'[:300]})
                return fallback() if callable(fallback) else fallback
        guarded.__name__ = method.__name__
        guarded.__doc__ = method.__doc__
        return guarded
    return wrap

class Charter:
    LIGHT = 'anthropic/claude-haiku-5.5'
    HEAVY = 'anthropic/claude-sonnet-5.5'
    SPARE = 'openai/gpt-6-luna'
    LOOP_CALLS = 42
    ALL_CALLS = 60
    SPEND_SOFT = 0.055
    SPEND_HARD = 0.078
    HEAVY_ROOM = 0.02
    WAITS = (4, 10, 20, 35)
    WAIT_TOTAL = 45
    BRIEF_BY = 120.0
    BRIEF_GAP = 10.0
    OUTAGE_BY = 150.0
    SHOW_CHARS = 450
    HUNT_STEPS = 24
    ENV_GAP = 0.25
    OPEN_GAP = 0.5
    TABLE_KEYS = 18
    CHEAPER_SHOWN = 4
    QUIET_WORD = 4
    OPEN_LATER = 2
    PAGE_ATTR_CHARS = 700
    PAGE_REST_CHARS = 300
    NO_NAME = ('none', 'null', 'n/a', 'unknown')
    ASK_TURNS = 2
    MESSAGE_FIELD = '(?:^|_)(?:texts?|body|messages?|notes?|contents?|questions?|msg|say)(?:_|$)'
    NOT_MESSAGE = '(?:^|_)(?:id|ids|type|format|code|lang|language|kind)$'
    TRAIL_ROOM = 15
    FIT_PAGES = 2
    CRASH_TURNS = 3
    OPEN_MIN_STEPS = 20
    OPEN_ROWS = 5
    OPEN_ITEMS = 2
    OPEN_SHARE = 5
    OPEN_EACH = 6
    OPEN_NOT = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|listing)\\b'
    SHORT_KEEP = 8
    SET_SHOWN = 2
    STOCK_OUT = '^(?:no|none|false|0|n)$|out[ _-]?of[ _-]?stock|sold[ _-]?out|unavailable|not[ _-]?available'
    HUNT_WHOLE = 49
    HUNT_GRACE = 25.0
    HUNT_VALUE_CHARS = 30
    ASK_WORDS = '\\b(?:one|each|single|first|time|many|much|several|list|once|separate\\w*|split|short\\w*|questions?|ask\\w*|turn|lot|too|all)\\b'
    HUNT_ROUNDS = 2
    HUNT_KEEP = 10
    HUNT_IDLE = 4
    HUNT_PAGE_NEW = 5
    HUNT_FIT_WORDS = 4
    HUNT_FIT_OPENS = 4
    HUNT_COMMON = 0.6
    HUNT_BY = 200.0
    TITLE_CHARS = 200
    MIX_LABELS = 3
    TEXT_CHARS = 3000
    OBS_LIMIT = 15000
    HISTORY_LIMIT = 150000
    GATE_ROUNDS = 3
    AUTO_PAGES = 6
    CHEAPER_OPENS = 3
    TASK_SECONDS = 300.0
    TIME_MARGIN = 20.0
    CALL_FLOOR = 6.0
    ASK_BY = 165.0
    QUESTION_BY = 210.0
    WIND_DOWN = 200.0
    LAST_TURN = 255.0
    EMPTY_ROOM = 12.0
    LATE_BY = 190.0
    LATE_STEPS = 8
    SCARCE_MESSAGES = 2
    LIST_SIZE = 12
    LIST_MIXED = 10
    LIST_STEPS = (1, 3)  # labels in the episode's first and second question, before the full list size
    LIST_STEPS_MIXED = (3, 3)  # the same on several items, where each question goes to another item (simulated: 3,3,8 level or above 1,3,8 on 2-3 items)
    LIST_FULL_MESSAGES = 3  # with this many messages or fewer, every question names the full list from the first one (race 17, 2 messages: 6-11 labels drew a need in 15% of messages, 1-3 labels in 10%; no one-at-a-time complaint in 2340 episodes)
    PUZZLED = '\\b(?:did not (?:follow|catch|understand|land|grasp|get)|didn.?t (?:follow|catch|understand|land|grasp|get|quite)|could not (?:tell|follow|make out|parse|catch|grasp)|couldn.?t (?:tell|follow|catch|grasp|make)|not (?:sure|clear) what you|lost (?:me|you|on that)|adrift|all at sea|muddle|tangled|explain(?:ed)? (?:it |that )?again|(?:say|send|run|tell me|give me) (?:that|it) (?:again|over)|over again|once more|try (?:that |it )?again|another pass|try (?:that|it) another way|another way|word (?:that|it) differently|differently|put it another way|break (?:it|that) down|simpler(?: words| terms)?|plain(?:er)? (?:words|english|terms)|everyday language|sailed past|over my head|none of that landed|did not help sort|shuffled deck|unfamiliar cards|sheet music|clef|jargon|gibberish|garbled|confus\\w*|baffl\\w*|puzzl\\w*|no idea what|what are you asking|what do you mean|what (?:that|it) was getting at|come again|slowly|gentler|not taking it in|in one ear|through a sieve|missed your meaning|repeat (?:it|that)|rephrase|clearer|sorry to be slow)\\b'  # a reply saying the message was not understood: none of the properties it named is one the shopper tracks (race 17 and 18 and practice personas; new idioms fall to the model)
    FREE_REPLY = '\\b(?:no horse|don.?t mind|do not mind|no (?:great |strong |particular |real )?(?:opinion|preference|view|feelings|lean|notion|ties|stake|skin)|not (?:fussed|bothered|picky|attached|particular|fussy)|either way|whichever|whatever (?:you|works|suits|comes|looks)|up to you|your call|any (?:option|of (?:them|those)|one|will do|is fine|would)|suits? me|sit well with me|hinges on|decide for me|choose freely|pick any|leave (?:it|that|that one) (?:wide )?open|happy to leave|content with anything|anything at all|agreeable|equally happy|zero feelings|leaves me cold|lose sleep over|nothing springs to mind|never thought about|surprise me|indifferent|doesn.?t matter|does not matter|not important to me|open on that|open ground|flexible|fine by me|all the same to me|not something I|care to pin|wind blows|dandy with me|no particular filling|plain-bun|toss a coin|whatever looks best|(?:got|have|had) (?:a|no) lean|lean on that)\\b'  # a reply saying the shopper does not mind: the first tracked property the message named is free
    COVER_ROWS = 60  # a search result set this small with unseen rows is listed in the status with its next-page call
    COVER_LINES = 4
    BATCH_OPENS = 8  # listing pages opened in one grouped environment call when a model turn asks for several (the store runs them as consecutive steps; saves each call's gap and round trip)
    CAP_ARG = '(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)'
    CAP_NOT = 'result|row|page|count|size|num|item|step'
    CAP_PRICED = 'price|cost|budget|spend|money|amount'  # a limit argument's name, when no search has shown one holding yet
    CAP_LOW = '(?:^|_)(?:min|minimum|low|lowest|least|from|start|offset|floor)(?:_|$)'
    EARLY_TRAILS = 3  # chain rounds the early follow may spend; the checkout keeps its own
    CAP_CHANCE = 0.05  # how rarely rows drawn at random from their labels would all sit under a value, for that value to count as a limit held
    WALK_ROOM = 10  # steps beyond the finish a walk listing after the first needs: a refusal at the next checkout still needs a swap
    WALK_CALLS = 4  # capped shelf listings per cart line at one checkout (race 16: 456 rows on a shelf, 6 a page)
    SHELF_ARG = 'section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$'
    LIST_MODE = 'field|filter|facet|brows|categor|list|structured'
    AMOUNT = '\\d[\\d.,]*\\s*[a-z"\']{1,12}'
    FLOOR = '\\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\\s+(' + AMOUNT + ')\\s*$'
    QUERY_WORDS = '(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword'
    GUIDE = 'You are God, a careful buyer working for one shopper in an online store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nHOW THE STORE WORKS\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun); a store\'s search can read attribute values and shop names as well as titles and may want every word, so a few required values plus the noun can list nearly the whole fitting set, and a model or part code may find nothing (one try tells). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once God knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by God\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. God carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through God\'s interviewer).\n\nWHAT COUNTS AS A FIT\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. God adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category. A listing under the requested label is that kind of product even when its title or description names something else, so judge it on the requirements, price and stock.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value; when the requirement asks for what goes with that listing\'s value (its partner, its counterpart) and the page names what its value pairs with, the requirement is that other value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nCHOOSING\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and God reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nCHANGES WHILE YOU WORK\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Price and stock can change silently: God re-reads the cart\'s lines and looks at the cart when you call the checkout, so do not spend a turn on either.\n\nFINISHING\n- Put exactly the chosen lines in the cart (remove anything else), then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
    BRIEF = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; so is a use or occasion the shopper says they will check or double-check on what you find ("they\'re for a wedding (I\'ll double-check)" gives "occasion: wedding"); a particular person\'s event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"search_words": {"tool": the tool that searches listings by free words (when one tool lists a category and another takes words, the one that takes words), "field": its parameter that takes the words, "args": the fixed arguments that make it search by words (for a tool with a mode argument, that mode), else {}}.\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
    PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
    HEAR = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them asked one at a time; not answers that only say they did not follow it or want it in simpler words], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"); the value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else", "at least 30 cm" for "equal to at least 30 cm").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write \"not X\" only when the answer rejects X itself.'
    NOTE = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
    CHECK = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails, though a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
    READING = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a value under a key that adds the item\'s own name to the property (a bag\'s "bag shape") counts even when the bare key ("shape") says something else; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
    AUDIT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READING + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
    UNMIX = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
    TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: its page not read, seen only as a search result row or by its attributes).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page; when the requirement asks for what goes with that value and the page names what its value pairs with, that other value; or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
    SENSE = 'A shopping agent asked a shopper which values it wants for some properties of a product, and the shopper replied as below. Say what the reply does. Return ONE JSON object and nothing else: {"kind": "puzzled" when the shopper says it did not understand or could not follow the message and asks for it again or in other words; "free" when the shopper says it has no preference and leaves the choice to the agent; "other" for anything else (a value or limit it wants, a change, a question back, a complaint about asking several things at once)}'
    GATE = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
    SWAP = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READING + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
    SHAPE = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
    GUARD_KEYS = 3
    GUARD_SHARE = 0.0
    GUARD_GAP = 0.34
    NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
    CATCH_ALL = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
    WIDEN_EACH = 6
    FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?(?:\\d{3,6}|\\d{1,5}[xX]{1,3})'
    PART_CODE = '[A-Z]{1,5}\\d{0,3}-?\\d{3,7}'
    VALUE_ROOM = 12  # steps beyond the finish before a value search runs at all (race 16: limit 35, answers at about step 15)
    VALUE_KEEP = 8  # steps a value search leaves beyond the finish, for opening and buying what it found
    LOOK_ALIKES = 2  # keys set aside in one question (race 16: one or two)
    LINK_SEARCHES = 3  # series searched per item in a decline hunt (race 16: up to 9 series)
    LINK_ROWS = 3  # rows of one item carrying a series code before the codes count as a system (race 16 covers: nearly every row)
    LINK_FAMILIES = 12  # more distinct series than this are model numbers, not a system (race 16: 9)
    PAIR_MARKS = (': ', ' = ', '=', ' -> ', ' - ')
    FIELD_MARKS = (' | ', '\n', ' || ', ' ; ', '; ', '\t', ' // ', ' · ', ' • ', '|', ';', ', ')
    FIELD_NAME = "[A-Za-z][A-Za-z0-9_ ./&'()-]{0,38}"
    LIST_BULLET = '^\\s*(?:[-*•·]|\\d{1,3}[.)])\\s+'
    RECORD_FIELDS = 3
    NOTE_FIELDS = 5
    FIELD_SHARE = 0.15
    PAIR_SHARE = 0.7
    BLOCK_NAME = 60
    BLOCK_VALUE = 40
    TAG_TEXT = 20000
    PAIR_KEY = 'name|key|label|attr|prop|field|spec|title|^k$'
    ID_FIELD = '(?:^|[_ -])(?:id|idx|sku|pid|ref|asin|ident\\w*)(?:$|[_ -])|^id(?-i:[A-Z0-9])|(?-i:Ident)|(?<![a-z])(?!(?:paid|valid|grid|solid|rigid|fluid|liquid|hybrid|rapid|void|acid|humid|android|said|laid)$)[a-z]*id$'
    FAIL_KEY = 'error|fail|(?<!de)fault|trouble(?!shoot)|denied|refus|invalid'

class Pipes:
    llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
    llm_short: dict[int, ProxyClient] = {}
    env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

    @staticmethod
    def pipe_for(seconds: float) -> ProxyClient:
        """The model client whose timeout fits the time left: the usual one, or one with a timeout in whole 5 s steps."""
        if seconds >= 50:
            return Pipes.llm
        limit = max(5, int(seconds) // 5 * 5)
        if limit not in Pipes.llm_short:
            Pipes.llm_short[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
        return Pipes.llm_short[limit]

class GlyphKit:
    """Codes and the words they sit beside."""

    @staticmethod
    def harvest(text: Any) -> list[str]:
        found = re.findall('\\b' + Charter.FIT_CODE + '\\b', str(text or ''))
        return [re.sub('[\\s-]+', ' ', c).strip().lower() for c in found if not re.fullmatch('[A-Z][a-z]+ (?:19|20)\\d\\d', c)]

    @staticmethod
    def covers(pattern: str, code: str) -> bool:
        """Whether a code a part names covers a device's code: the same letters and digits, or a series written with trailing
    x's ("abc12xx" covers "abc1234"); spaces and hyphens do not count."""
        wide, plain = (re.sub('[\\s-]+', '', pattern), re.sub('[\\s-]+', '', code))
        stem = wide.rstrip('x')
        if stem != wide and stem[-1:].isdigit():
            return plain.startswith(stem) and len(plain) == len(wide)
        return wide == plain

    @staticmethod
    def marks(text: Any) -> dict[str, Any]:
        text = str(text or '')
        own = re.search('[\\[(]\\s*(?:part\\s+)?(' + Charter.PART_CODE + ')\\s*[\\])]', text, re.I)
        codes: list[str] = []
        for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)?\\s*[:.]?\\s*(?!(?:no|number)\\b)(' + Charter.FIT_CODE + ')\\b', text, re.I):
            codes += GlyphKit.harvest(hit.group(1))
        for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
            codes += GlyphKit.harvest(hit.group(1))
        works: list[str] = []
        bars: list[str] = []
        for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
            if not re.search('\\bpart\\s+' + Charter.PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
                cut = re.split('(?i)\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1)
                works += GlyphKit.harvest(cut[0])
                bars += GlyphKit.harvest(cut[1]) if len(cut) > 1 else []
        also = [(hit.group(1).upper(), GlyphKit.harvest(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + Charter.PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
        more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + Charter.PART_CODE + ')', text, re.I)
        fam, named = GlyphKit.series_bits(text)
        return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'bars': bars, 'also': also, 'more': more.group(1).upper() if more else '', 'fam': fam, 'named': named}

    @staticmethod
    def series_bits(text: str) -> tuple[list[str], list[str]]:
        """The series a record's own codes start with ("AB1" of "AB1-2345") and the bare series-like words it names outside a clause that
    excludes them. Which bare words are series is decided against the codes other records carry (God.link_system)."""
        fam = {hit.upper() for hit in re.findall('\\b([A-Za-z]{1,5}\\d{1,3})-\\d{2,7}\\b', text)}
        rest = re.sub('\\b[A-Za-z]{1,5}\\d{1,3}-\\d{2,7}\\b', ' ', text)
        rest = re.sub('(?i)\\b(?:except|excluding|but not|not for|not compatible with|does not (?:fit|take))\\b[^.;]*', ' ', rest)
        return sorted(fam), sorted({hit.upper() for hit in re.findall('\\b[A-Za-z]{1,5}\\d{1,3}\\b', rest)})

    @staticmethod
    def speak(key: str) -> str:
        key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
        return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

    @staticmethod
    def snake(key: str) -> str:
        return GlyphKit.speak(key).replace(' ', '_')

    @staticmethod
    def stemset(text: str) -> frozenset[str]:
        return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in GlyphKit.speak(text).split()))

    @staticmethod
    def glued_text(text: str) -> str:
        """A number and its short unit as one word, however it is spelled: "65 W" and "65W" are one value, "45 W" another."""
        return re.sub('(\\d)\\s+([a-z]{1,3})(?![a-z0-9])', '\\1\\2', text.lower())

    @staticmethod
    def text_wordset(text: str) -> set[str]:
        """The words of a row or a search, a number and its short unit also as one word."""
        low = text.lower()
        return set(re.findall('[a-z0-9]+', low)) | set(re.findall('[a-z0-9]+', GlyphKit.glued_text(low)))

class BlockKit:
    """One field, one row, or a small block of text."""

    @staticmethod
    def attr_slab(text: str) -> dict[str, Any] | None:
        """A listing's attributes written inside one field: "a=b; c=d", a brace block, bulleted "name: value" lines, or short
    "name: value" parts between commas. Prose stays text whatever colons it holds (a description reading "Fits: A, B. Condition:
    new" is no block), and a name may be any short text, since attribute names come as the catalog has them."""
        bare = text.strip()
        if len(bare) < 5:
            return None
        if bare[0] in '{[' and bare[-1] in '}]':
            try:
                parsed = json.loads(bare)
                return parsed if isinstance(parsed, dict) else None
            except (ValueError, RecursionError):
                found = dict(re.findall('[\'"]?([A-Za-z][\\w ./-]{0,38})[\'"]?\\s*[:=]\\s*[\'"]?([^,\'"}\\]]*)', bare))
                return {k.strip(): BlockKit.scalar_of(v) for k, v in found.items()} if len(found) >= 2 else None
        if '=' in bare:
            out: dict[str, Any] = {}
            last = ''
            parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', bare) if p.strip()]
            for part in parts:
                name, found, value = part.partition('=')
                if found and 0 < len(name.strip()) <= Charter.BLOCK_NAME and name.strip() not in out:
                    last = name.strip()
                    out[last] = BlockKit.slab_scalar(value)
                elif last:
                    out[last] = f'{out[last]}; {part.strip()}'
                else:
                    return None
            return out if len(out) >= 2 or (out and ':' not in last and len(last) <= 30 and len(bare) <= 90) else None
        lines = [line for line in bare.splitlines() if line.strip()]
        if len(lines) >= 2 and all((re.match(Charter.LIST_BULLET, line) for line in lines)):
            pairs = [BlockKit.cut_pair(line.strip(), ': ') for line in lines]
            if all(pairs):
                return {name: BlockKit.scalar_of(value) for name, value in pairs}
        for between in (', ', '; '):
            if between in bare and ': ' in bare:
                pairs = [BlockKit.cut_pair(part.strip(), ': ') for part in bare.split(between)]
                if len(pairs) >= 2 and all(pairs) and all(('.' not in name and len(value) <= Charter.BLOCK_VALUE for name, value in pairs)):
                    return {name: BlockKit.scalar_of(value) for name, value in pairs}
        return None

    @staticmethod
    def block_of(text: str) -> Any:
        """A text field of a row that already has its own fields: read as a record only when wholly written as name=value parts."""
        if len(text) < 5 or '=' not in text or not re.search('; | \\| |\\n', text):
            return text
        parts = [p for p in re.split(';\\s+|;$|\\s\\|\\s|\\n', text.strip()) if p.strip()]
        if len(parts) < 2 or not all((re.fullmatch('[^=:]{1,40}=[^=]{0,200}', p.strip()) for p in parts)):
            return text
        return BlockKit.attr_slab(text) or text

    @staticmethod
    def cut_pair(text: str, mark: str) -> tuple[str, str] | None:
        """(name, value) when the text opens with a field name and this mark."""
        head, found, rest = text.partition(mark)
        if not found and text.endswith(mark.strip()):
            head, found, rest = (text[:-len(mark.strip())], mark, '')
        name = re.sub(Charter.LIST_BULLET, '', head).strip()
        if not found or not re.fullmatch(Charter.FIELD_NAME, name) or len(name.split()) > 5:
            return None
        return (name, rest.strip())

    @staticmethod
    def failed_only(observation: Any) -> bool:
        """Whether a result only reports that the call failed: a small object with a key saying so, or one bare code."""
        if not isinstance(observation, dict) or not 0 < len(observation) <= 3:
            return False
        if any((value and re.search(Charter.FAIL_KEY, str(key), re.I) for key, value in observation.items())):
            return True
        only = next(iter(observation.values()))
        return len(observation) == 1 and isinstance(only, str) and 0 < len(only) <= 40 and ' ' not in only

    @staticmethod
    def field_cuts(text: str, between: str, mark: str) -> tuple[list[tuple[str, str]], int]:
        """The fields of a text cut at `between`, a part with no name joining the field before it (a value may hold the
    separator), with how many parts carried a name. A name closed by a colon with nothing after it heads the bulleted
    or indented lines below it, kept as one field holding those lines."""
        parts = [p for p in text.split(between) if p.strip()]
        fields: list[tuple[str, str]] = []
        named = 0
        heading = -1
        if parts and not BlockKit.cut_pair(parts[0].strip(), mark):
            return ([], 0)
        for part in parts:
            bare = part.strip()
            if between == '\n' and heading >= 0 and (re.match(Charter.LIST_BULLET, part) or part[:1] in ' \t'):
                name, value = fields[heading]
                fields[heading] = (name, (value + '\n' if value else '') + bare)
                continue
            heading = -1
            if between == '\n' and bare.endswith(':') and re.fullmatch(Charter.FIELD_NAME, bare[:-1].strip()):
                fields.append((bare[:-1].strip(), ''))
                heading = len(fields) - 1
                named += 1
                continue
            pair = BlockKit.cut_pair(bare, mark)
            if pair:
                fields.append(pair)
                named += 1
            elif fields:
                name, value = fields[-1]
                fields[-1] = (name, value + between + bare)
        return (fields, named) if len(parts) else ([], 0)

    @staticmethod
    def pairs_of(rows: list[Any], in_record: bool) -> dict[str, Any] | None:
        """Attributes written as a list: objects of a name and a value, [name, value] lists, or, inside a record, short "name: value"
    texts. A list of offers, of highlights or of notes is no such thing."""
        if len(rows) < 2:
            return None
        out: dict[str, Any] = {}
        first = rows[0]
        if isinstance(first, dict):
            if any((not isinstance(r, dict) or set(r) != set(first) or len(r) != 2 for r in rows)):
                return None
            keys = list(first)
            if not re.search(Charter.PAIR_KEY, str(keys[0]), re.I):
                return None
            for row in rows:
                name, value = row[keys[0]], row[keys[1]]
                if not isinstance(name, str) or not 0 < len(name) <= Charter.BLOCK_NAME or isinstance(value, (dict, list)):
                    return None
                out[name.strip()] = value
            return out
        if isinstance(first, (list, tuple)):
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str) or isinstance(row[1], (dict, list)) or not 0 < len(row[0]) <= Charter.BLOCK_NAME:
                    return None
                out[row[0].strip()] = row[1]
            return out
        if in_record and all((isinstance(r, str) for r in rows)):
            for row in rows:
                pair = BlockKit.cut_pair(row.strip(), ': ') or BlockKit.cut_pair(row.strip(), '=')
                if not pair or len(pair[1]) > Charter.BLOCK_VALUE or len(pair[0].split()) > 3:
                    return None
                out[pair[0]] = BlockKit.scalar_of(pair[1])
            return out
        return None

    @staticmethod
    def record_slab(text: str) -> dict[str, Any] | None:
        """The record a text spells out field by field, whichever separators it uses; None when it reads as plain prose. The
    separators are found by trying them, the outer field mark first and then the pair mark giving the most named fields,
    and a value spelled the same way becomes a record of its own (a listing's attributes written inside one field)."""
        best: tuple[int, list[tuple[str, str]]] | None = None
        for between in Charter.FIELD_MARKS:
            if between not in text:
                continue
            pieces = len([p for p in text.split(between) if p.strip()])
            for mark in Charter.PAIR_MARKS:
                fields, named = BlockKit.field_cuts(text, between, mark)
                if named < Charter.RECORD_FIELDS or named < (Charter.PAIR_SHARE if between == ', ' else Charter.FIELD_SHARE) * pieces:
                    continue
                if len({name for name, _ in fields}) * 2 <= len(fields):
                    continue
                if best is None or named > best[0]:
                    best = (named, fields)
            if best:
                break
        if best is None:
            return None
        record: dict[str, Any] = {}
        for name, value in best[1]:
            if name in record:
                continue
            inner = BlockKit.attr_slab(value)
            record[name] = inner if inner else value.strip() if re.search(Charter.ID_FIELD, name, re.I) else BlockKit.scalar_of(value)
        return record

    @staticmethod
    def scalar_of(text: str) -> Any:
        """A field's text as the value a structured result would hold: a boolean, nothing, a number, else the text."""
        bare = text.strip()
        low = bare.lower()
        if low in ('true', 'false'):
            return low == 'true'
        if low in ('none', 'null', ''):
            return None
        money = re.fullmatch('[$€£]?\\s?((?:\\d{1,3}(?:,\\d{3})+|\\d+)(?:\\.\\d+)?)', bare)
        if money and not re.match('0\\d', money.group(1)):
            number = float(money.group(1).replace(',', ''))
            return int(number) if number.is_integer() and '.' not in money.group(1) else number
        return bare

    @staticmethod
    def slab_scalar(text: str) -> Any:
        """A value inside an attribute block: "attachment method=none" is a value, and a block only keeps text and numbers."""
        found = BlockKit.scalar_of(text)
        return text.strip() if isinstance(found, bool) or (found is None and text.strip()) else found

    @staticmethod
    def table_slab(text: str) -> list[dict[str, Any]] | None:
        """Rows of a text table: a heading line of names, then lines cut into as many cells (markdown rules are skipped)."""
        lines = [line.strip().strip('|').strip() for line in text.splitlines() if line.strip() and not re.fullmatch('[\\s|:+=-]+', line)]
        if len(lines) < 2:
            return None
        for between in ('|', '\t', ';', ','):
            cells = [[c.strip() for c in line.split(between)] for line in lines]
            size = len(cells[0])
            if size < 3 or any((len(row) != size for row in cells)) or not all((re.fullmatch(Charter.FIELD_NAME, c) for c in cells[0])):
                continue
            if any((re.fullmatch('[\\d.,$]+', c) for c in cells[0])):
                continue
            return [{name: BlockKit.scalar_of(c) for name, c in zip(cells[0], row)} for row in cells[1:]]
        return None

    @staticmethod
    def rootset(text: str) -> set[str]:
        out = set()
        for w in GlyphKit.speak(text).split():
            w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
            if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
                out.add(w)
        return out

    @staticmethod
    def spellset(text: str) -> frozenset[str]:
        """A name's words with plural -s and British -our folded, so "colours" and "color" compare equal."""
        return frozenset((re.sub('our$', 'or', w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w) for w in GlyphKit.speak(text).split()))

class NestKit:
    """Nested records, restored pages, and the first spelling of a value."""

    first_spelling: dict[str, str] = {}

    @staticmethod
    def column_map(node: dict[str, Any]) -> dict[str, Any]:
        """A result that names its columns once and gives each row as a list comes back with each row as an object."""
        heads = [k for k, v in node.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(c, str) and re.fullmatch(Charter.FIELD_NAME, c) for c in v))]
        for head in heads:
            names = node[head]
            for key, rows in node.items():
                if key != head and isinstance(rows, list) and rows and all((isinstance(r, (list, tuple)) and len(r) == len(names) for r in rows)):
                    out = {k: v for k, v in node.items() if k != head}
                    out[key] = [dict(zip(names, row)) for row in rows]
                    return out
        return node

    @staticmethod
    def flat_of(node: dict[str, Any]) -> dict[str, Any]:
        """An attribute block with groups inside it, as one level: {"size": {"width": 3}} gives {"size width": 3}."""
        if not any((isinstance(v, dict) for v in node.values())):
            return node
        out: dict[str, Any] = {}
        for key, value in node.items():
            if isinstance(value, dict):
                out.update({f'{key} {inner}': v for inner, v in value.items() if not isinstance(v, (dict, list))})
            else:
                out[key] = value
        return out

    @staticmethod
    def fold_pairs(node: Any, depth: int=0) -> Any:
        """One line per property: a block spelling a key twice keeps the first spelling, with both values when they differ, so that
    one element of a list is enough ("clock shape: rectangle" and "clock_shape: rectangular, square" hold square). About a tenth
    of the listings of races 13-15 and of a practice world hold such pairs with different values; read as a conflict, they
    lost listings the store's answer accepted. Every repeated line also costs tokens on each later model turn."""
        if depth > 5:
            return node
        if isinstance(node, list):
            return [NestKit.fold_pairs(v, depth + 1) for v in node]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        seen: dict[str, str] = {}
        for key, value in node.items():
            name = GlyphKit.speak(key)
            first = seen.get(name)
            if first is None:
                seen[name] = key
                out[key] = NestKit.fold_pairs(value, depth + 1)
            elif out[first] == value:
                continue
            elif isinstance(out[first], str) and isinstance(value, str) and not re.search(Charter.ID_FIELD, name, re.I):
                out[first] = NestKit.join_pair(out[first], value)
            else:
                out[key] = NestKit.fold_pairs(value, depth + 1)
        return out if len(out) >= 2 or len(node) < 2 else {k: NestKit.fold_pairs(v, depth + 1) for k, v in node.items()}

    @staticmethod
    def holds_ident(node: dict[str, Any]) -> bool:
        return any((re.search(Charter.ID_FIELD, str(k), re.I) for k in node))

    @staticmethod
    def id_text_map(record: dict[str, Any]) -> dict[str, Any]:
        """A record's numeric identifier as text, the form the agent keys listings by (schemas turn it back when a tool wants a number)."""
        if not any((isinstance(v, str) and ' ' in v for v in record.values())):
            return record
        return {k: str(v) if isinstance(v, int) and not isinstance(v, bool) and re.search(Charter.ID_FIELD, k, re.I) and abs(v) >= 100 else v for k, v in record.items()}

    @staticmethod
    def join_pair(held: str, value: str) -> str:
        """Two spellings' values as one list, each element once; the first spelling's value stays known, for measures of how a key's values read."""
        between = '(?<!\\d),|,(?!\\d)'
        have = {part.strip().lower() for part in re.split(between, held)}
        extra: list[str] = []
        for part in re.split(between, value):
            if part.strip() and part.strip().lower() not in have:
                have.add(part.strip().lower())
                extra.append(part.strip())
        if not extra:
            return held
        joined = (f'{held}, ' if held.strip() else '') + ', '.join(extra)
        NestKit.first_spelling[joined] = NestKit.first_spelling.get(held, held)
        return joined

    @staticmethod
    def keyed_of(node: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Rows given as an object keyed by listing code, each a record with the same fields, as a list of rows with their code."""
        rows = list(node.values())
        if len(rows) < 2 or not all((isinstance(r, dict) and len(r) >= Charter.RECORD_FIELDS for r in rows)) or not all((re.search('\\d', str(k)) and ' ' not in str(k) for k in node)):
            return None
        if len(set(rows[0]).intersection(*rows[1:])) < Charter.RECORD_FIELDS:
            return None
        return [r if any((re.search(Charter.ID_FIELD, str(x), re.I) for x in r)) else {'id': str(k), **r} for k, r in node.items()]

    @staticmethod
    def records_of(text: str) -> Any:
        """The structured form of a text a store tool returned, or None when it holds no records."""
        bare = text.strip()
        if len(bare) < 12:
            return None
        if bare[0] in '{[':
            try:
                return NestKit.restore_tree(json.loads(bare))
            except (ValueError, RecursionError):
                pass
        for parts in (re.split('\\n\\s*\\n', bare), bare.splitlines()):
            if len(parts) >= 2:
                rows = [BlockKit.record_slab(p) for p in parts]
                good = [r for r in rows if r]
                if len(good) >= 2 and len(good) * 5 >= len(parts) * 4 and len({next(iter(r)) for r in good}) == 1:
                    return good
        found = BlockKit.table_slab(bare) if '\n' in bare and BlockKit.record_slab(bare) is None else BlockKit.record_slab(bare)
        if found is None and len(bare) <= Charter.TAG_TEXT and re.search('<[a-zA-Z/][^<>]{0,40}>', bare):
            plain = re.sub('</?(?:b|i|em|strong|span|a|u|small)\\b[^<>]{0,40}>', '', bare, flags=re.I)
            plain = re.sub('</t[dh]>\\s*<t[dh][^<>]{0,40}>', ': ', plain, flags=re.I)
            plain = re.sub('\\s*(?:<[^<>]{1,40}>\\s*)+', '\n', plain).strip()
            if plain != bare and len(plain) >= 12:
                return NestKit.records_of(plain)
        if found is None and '\n' in bare:
            lines = [re.sub('(?:[ ]{2,}|\t)', ': ', line.strip(), count=1) for line in bare.splitlines() if line.strip()]
            spaced = '\n'.join(lines)
            if spaced != bare:
                return BlockKit.record_slab(spaced)
        return found

    @staticmethod
    def restore_one(value: Any, depth: int, carded: bool, page: bool=True) -> Any:
        """A field's value restored; a text only becomes a record when it reads as a listing, by an id, by a code among its values
    or by many fields, so that a store's note written with colons and semicolons stays a note."""
        found = NestKit.restore_tree(value, depth + 1, carded, page)
        if isinstance(value, str) and isinstance(found, dict) and len(found) < Charter.NOTE_FIELDS and not NestKit.holds_ident(found) and not any((FormKit.id_shaped(v) for v in found.values())):
            return value
        return found

    @staticmethod
    def restore_tree(value: Any, depth: int=0, in_record: bool=False, page: bool=True) -> Any:
        """A store result in the one form the rest of the agent reads: lists of row objects, a listing's attributes as a nested
    object. Worlds spell results differently (race 15 gave each record as one text, its attributes "a=b; c=d" inside it, and
    a result the agent cannot read leaves every record-based check blind), so the spelling is undone here, once, for all."""
        if depth > 7:
            return value
        if isinstance(value, str):
            found = NestKit.records_of(value)
            return found if found is not None else value
        if isinstance(value, list):
            block = BlockKit.pairs_of(value, in_record and page)
            if block:
                return block
            return [NestKit.restore_one(v, depth, False, page) for v in value]
        if isinstance(value, dict):
            node = NestKit.id_text_map(NestKit.column_map(value))
            carded = len(node) >= Charter.RECORD_FIELDS or NestKit.holds_ident(node)
            keyed = {k: NestKit.keyed_of(v) if isinstance(v, dict) else None for k, v in node.items()}
            node = {k: keyed[k] or v for k, v in node.items()}
            listing = NestKit.holds_ident(node) or any((FormKit.id_shaped(v) for v in node.values()))
            out = {k: NestKit.restore_one(v, depth, carded, page) if not isinstance(v, str) or len(node) <= 2 else (BlockKit.block_of(v) if listing else v) for k, v in node.items()}
            if depth <= 3:
                out = {k: NestKit.flat_of(v) if isinstance(v, dict) and any((isinstance(x, dict) for x in v.values())) and not any((isinstance(x, list) for x in v.values())) and isinstance(next(iter(v.values()), None), (dict, str, int, float)) and len(node) >= Charter.RECORD_FIELDS and not NestKit.holds_ident(v) else v for k, v in out.items()}
            if len(out) == 1:
                only = next(iter(out.values()))
                if isinstance(only, dict) and len(only) >= Charter.RECORD_FIELDS and isinstance(next(iter(value.values())), str):
                    return only
            return out
        return value

    @staticmethod
    def value_pieces(value: str) -> list[str]:
        """A value's words a search or a row can hold: long ones, else short ones of two characters holding a letter (XL, 4g, 65w), else its numbers."""
        low = GlyphKit.glued_text(value)
        return re.findall('[a-z0-9]{3,}', low) or re.findall('[0-9]+[a-z]|[0-9]*[a-z][a-z0-9]+', low) or re.findall('[0-9]+', low)

    @staticmethod
    def same_name(prop: frozenset[str], key: frozenset[str]) -> bool:
        if not prop or not key:
            return False
        return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

class FormKit:
    """Model JSON, schema checks, and the arguments of a call."""

    @staticmethod
    def pack(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def shorten(value: Any, limit: int, depth: int=0) -> Any:
        if depth > 7:
            return '...'
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + '...'
        if isinstance(value, dict):
            return {k: FormKit.shorten(v, limit, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [FormKit.shorten(v, limit, depth + 1) for v in value]
        return value

    @staticmethod
    def object_of(text: Any) -> dict[str, Any] | None:
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

    @staticmethod
    def fact_map(node: Any) -> dict[str, Any] | None:
        if not isinstance(node, dict):
            return None
        best = None
        for value in node.values():
            if isinstance(value, dict) and len(value) >= 2:
                scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
                if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                    best = scalars
        return best

    @staticmethod
    def filler(spec: dict[str, Any]) -> Any:
        """A value for a required field the agent has nothing to say in; None when the schema gives no safe one."""
        if isinstance(spec.get('enum'), list) and spec['enum']:
            return spec['enum'][0]
        for word in ('const', 'default'):
            if word in spec:
                return spec[word]
        kinds = FormKit.kind_list(spec)
        if 'integer' in kinds or 'number' in kinds:
            return spec['minimum'] if isinstance(spec.get('minimum'), (int, float)) else 1
        if 'boolean' in kinds:
            return False
        return 'question'[:spec.get('maxLength') or 8] if 'string' in kinds else None

    @staticmethod
    def strings_of(node: Any) -> list[str]:
        """The texts a call's arguments hold, at any depth."""
        if isinstance(node, dict):
            return [s for v in node.values() for s in FormKit.strings_of(v)]
        if isinstance(node, list):
            return [s for v in node for s in FormKit.strings_of(v)]
        return [node] if isinstance(node, str) else []

    @staticmethod
    def says_text(node: Any, text: str) -> bool:
        """Whether a call's arguments carry `text` whole, however deep."""
        if isinstance(node, dict):
            return any((FormKit.says_text(v, text) for v in node.values()))
        if isinstance(node, list):
            return any((FormKit.says_text(v, text) for v in node))
        return node == text

    @staticmethod
    def as_count(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
            return int(float(value))
        return None

    @staticmethod
    def as_items(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, (str, dict)) and value:
            return [value]
        return []

    @staticmethod
    def fit_value(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
        if not isinstance(schema, dict):
            return value
        kinds = FormKit.kind_list(schema)
        if kinds and (not any((FormKit.matches_kind(value, k) for k in kinds))):
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
            if not any((FormKit.matches_kind(value, k) for k in kinds)):
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
                value = [FormKit.fit_value(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
        if isinstance(value, dict):
            props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
            if schema.get('additionalProperties') is False:
                for name in [n for n in value if n not in props]:
                    faults.append(f'{where}.{name} is not an allowed argument')
            for name in schema.get('required') or []:
                if name not in value:
                    faults.append(f'{where}.{name} is required')
            value = {k: FormKit.fit_value(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
        return value

    @staticmethod
    def id_shaped(value: Any, named: bool=False) -> bool:
        if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
            return False
        return named or bool(re.search('\\d', value))

    @staticmethod
    def ident_inside(node: Any, ident: str, depth: int=0) -> bool:
        if depth > 5:
            return False
        if isinstance(node, list):
            return any((isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((FormKit.ident_inside(v, ident, depth + 1) for v in node))
        if isinstance(node, dict):
            return any((FormKit.ident_inside(v, ident, depth + 1) for v in node.values()))
        return False

    @staticmethod
    def kind_list(schema: dict[str, Any]) -> list[str]:
        kind = schema.get('type')
        if isinstance(kind, str):
            return [kind]
        return [k for k in kind or [] if isinstance(k, str)]

    @staticmethod
    def learn_ident(rows: list[dict[str, Any]]) -> str | None:
        best = None
        for key in rows[0]:
            values = [row.get(key) for row in rows]
            named = bool(re.search(Charter.ID_FIELD, str(key), re.I))
            if not all((FormKit.id_shaped(v, named) for v in values)):
                continue
            values = [v.strip() for v in values]
            if len(rows) > 1 and len(set(values)) < len(values):
                continue
            score = (named, sum((len(v) for v in values)) / len(values))
            if best is None or score > best[0]:
                best = (score, key)
        return best[1] if best else None

    @staticmethod
    def matches_kind(value: Any, kind: str) -> bool:
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

class Ribbon(list):
    """A trace whose entries say when they happened."""

    def __init__(self, clock: Any):
        super().__init__()
        self.clock = clock

    def append(self, entry: dict[str, Any]) -> None:
        entry.setdefault('at', round(self.clock(), 1))
        super().append(entry)

class SlipLine:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class Pocket:
    """A bare bag of fields. The vault hangs these at the leaves."""

class Clockwork:
    def __init__(self) -> None:
        self.pace = Pocket()

class Counter:
    def __init__(self) -> None:
        self.front = Pocket()

class FolioOrder:
    def __init__(self) -> None:
        self.sheet = Pocket()

class FolioVoice:
    def __init__(self) -> None:
        self.tape = Pocket()

class FolioBasket:
    def __init__(self) -> None:
        self.bin = Pocket()

class Folio:
    def __init__(self) -> None:
        self.order = FolioOrder()
        self.voice = FolioVoice()
        self.basket = FolioBasket()

class StackPages:
    def __init__(self) -> None:
        self.book = Pocket()

class StackProbe:
    def __init__(self) -> None:
        self.log = Pocket()

class StackChains:
    def __init__(self) -> None:
        self.web = Pocket()

class Stacks:
    def __init__(self) -> None:
        self.pages = StackPages()
        self.probe = StackProbe()
        self.chains = StackChains()

class Scroll:
    def __init__(self) -> None:
        self.leaf = Pocket()

class Vault:
    """Episode memory, one pocket deeper than a flat agent object."""

    def __init__(self) -> None:
        self.clockwork = Clockwork()
        self.counter = Counter()
        self.folio = Folio()
        self.stacks = Stacks()
        self.scroll = Scroll()

class Switch:
    """Every desk call is reached by its new name, so a desk never names a sibling."""

    def __init__(self) -> None:
        self._fn: dict[str, Any] = {}

    def attach(self, desk: Any) -> None:
        for name, value in type(desk).__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(value, staticmethod):
                self._fn[name] = value.__func__
            elif callable(value):
                self._fn[name] = getattr(desk, name)

    def __getattr__(self, name: str) -> Any:
        found = self._fn.get(name)
        if found is None:
            raise AttributeError(name)
        return found

ORIGIN = {
    'steps_open': 'left',
    'elapsed': 'clock',
    'model_window': 'model_room',
    'pace_pinched': 'rushed',
    'notes_spare': 'msgs_free',
    'reserve_steps': 'tail',
    'model_order': 'judges',
    'ask_model': 'think',
    'post_model': 'call_model',
    'ask_object': 'ask_json',
    'book_result': 'took',
    'call_batch': 'act_many',
    'prime_opens': 'prefetch_opens',
    'call_store': 'act',
    'mend_text': 'unspelled',
    'take_brief': 'read_request',
    'shelf_clash': 'shared_shelf',
    'peel_foreign': 'unmix',
    'foreign_need': 'described',
    'must_sheet': 'ledger',
    'cover_lines': 'coverage',
    'now_text': 'status',
    'basket_sig': 'cart_mark',
    'stage_pending': 'gate_due',
    'find_ident': 'ident',
    'learn_row_keys': 'learn_rows',
    'take_obs': 'absorb',
    'shelf_labels': 'labels',
    'shelf_field': 'category_key',
    'spec_census': 'key_table',
    'aside_mark': 'not_stated',
    'slot_match': 'twins',
    'shelf_groups': 'item_groups',
    'one_id_arg': 'single_target',
    'price_num': 'price_of',
    'in_stock': 'buyable',
    'rows_inside': 'records_in',
    'mark_view': 'note_view',
    'seen_before': 'shown_already',
    'learn_shape': 'read_shape',
    'page_span': 'paging',
    'price_order': 'price_sorted',
    'page_names': 'page_tools',
    'spec_tool': 'facts_tool',
    'compare_has_specs': 'compare_facts',
    'compare_is_thin': 'compare_bare',
    'page_dump': 'page_text',
    'render_obs': 'observe',
    'show_obs': 'show',
    'proof_lines': 'evidence',
    'record_blob': 'record_of',
    'side_records': 'others',
    'title_code_ids': 'row_ids',
    'next_page_note': 'more_pages',
    'cap_miss_note': 'cap_ignored',
    'low_price_note': 'cheapest_here',
    'wide_hold_note': 'wider_hold',
    'mark_spec_tool': 'note_facts',
    'tool_has_rows': 'lists_things',
    'open_for_specs': 'fill_facts',
    'open_for_links': 'show_links',
    'cap_form': 'cap_template',
    'cap_names': 'cap_keys',
    'cap_mode_map': 'cap_modes',
    'search_under': 'list_below',
    'lower_ids': 'cheaper_rows',
    'under_basket_note': 'below_cart',
    'title_slot': 'title_key',
    'hunt_phrase': 'hunt_words',
    'item_row_count': 'rows_of_item',
    'hunt_lower': 'hunt_cheaper',
    'hunt_one': 'hunt_item',
    'hunt_matches': 'hunt_fits',
    'open_unseen': 'unread_pages',
    'call_sig': 'call_mark',
    'mark_read': 'note_read',
    'query_fresh': 'query_is_new',
    'flip_last_page': 'last_page_first',
    'query_has_code': 'names_code',
    'tune_first_query': 'first_look',
    'short_query_note': 'shorter_search',
    'open_pass': 'first_moves',
    'rows_until_keys': 'more_rows_for_keys',
    'word_form': 'search_template',
    'value_bits': 'value_words',
    'value_form': 'value_template',
    'search_values': 'value_search',
    'own_shelf': 'home_label',
    'spread_asks': 'widen',
    'spread_one': 'widen_singly',
    'due_to_ask': 'ready_to_ask',
    'note_args': 'message_args',
    'run_questions': 'interview',
    'late_ask': 'late_question',
    'ask_batches': 'groups_for',
    'batch_questions': 'ask_group',
    'close_ask': 'finish_ask',
    'list_limit': 'list_cap',
    'list_size_now': 'list_now',
    'next_lane': 'next_ask',
    'already_said': 'told_about',
    'near_labels': 'look_alikes',
    'ask_line': 'ask_text',
    'trim_ask': 'fit_ask',
    'upcoming_asks': 'ask_next',
    'land_answer': 'settle',
    'reply_tag': 'reply_kind',
    'reply_read': 'reply_sense',
    'free_index': 'free_hit',
    'answered_at': 'answered_label',
    'lane_for': 'lane_keys',
    'take_labels': 'adopt_labels',
    'shelf_owner': 'owner_of',
    'label_owned': 'owns',
    'opened_for': 'records_of_item',
    'build_lanes': 'plan_lanes',
    'name_bits': 'name_parts',
    'item_words': 'item_said',
    'one_ask': 'question',
    'sample_vals': 'samples',
    'dispatch_asks': 'send',
    'apply_reply': 'take_heard',
    'read_replies': 'hear',
    'reply_blank': 'quiet_reply',
    'pass_stage': 'gate',
    'voice_text': 'spoken',
    'note_bits': 'loose_lines',
    'read_note': 'take_note',
    'live_needs': 'needs_now',
    'basket_text': 'cart_text',
    'units_of': 'held_units',
    'basket_ids': 'cart_ids',
    'compat_index': 'fit_index',
    'codes_of': 'fit_set',
    'ids_of_item': 'item_ids',
    'series_web': 'link_system',
    'series_hit': 'link_holds',
    'series_miss': 'link_faults',
    'series_list': 'series_for',
    'compat_miss': 'fit_faults',
    'compat_plan': 'fit_plan',
    'cheap_pairs': 'cheapest_sets',
    'items_absent': 'missing_items',
    'ref_spent': 'pointer_dead',
    'meets_needs': 'states_needs',
    'can_fill_short': 'fill_short_cart',
    'split_shelves': 'part_shelves',
    'richer_shelf': 'fuller_label',
    'shelf_for_item': 'named_label',
    'line_misses': 'line_faults',
    'qty_fixes': 'count_fixes',
    'apply_qty': 'fix_counts',
    'judge_key': 'check_key',
    'judge_basket': 'check_cart',
    'code_picks': 'code_leads',
    'short_picks': 'short_leads',
    'basket_compat': 'whole_cart_fits',
    'trail_on_refuse': 'refuse_trail',
    'trail_soon': 'early_trail',
    'page_opened': 'page_read',
    'walk_chains': 'trail',
    'skip_maker': 'not_maker_keys',
    'shop_ids': 'shop_rows',
    'opened_shop_ids': 'opened_shop',
    'seek_link': 'find_link',
    'offer_name': 'offer_tool',
    'reload_offers': 'reread_offers',
    'is_gone': 'sold_out',
    'swap_note': 'swap_hint',
    'reread_gone': 'find_sold',
    'swap_ids': 'replacements',
    'audit_empty': 'audit',
    'audit_body': 'audit_call',
    'fits_of_audit': 'audit_fits',
    'open_row_ids': 'open_rows',
    'basket_calls': 'cart_calls',
    'build_basket': 'complete_cart',
    'call_checks_out': 'is_buy',
    'call_declines': 'is_refuse',
    'quiet_args': 'quiet',
    'fix_id_args': 'resolve_ids',
    'sets_units': 'cart_sets',
    'fill_remove': 'whole_removal',
    'apply_basket': 'note_cart',
    'tools_now': 'offered',
    'compare_field': 'compare_target',
    'compare_width': 'compare_room',
    'fold_opens': 'join_opens',
    'play_calls': 'run_calls',
    'decline_ready': 'can_refuse',
    'end_episode': 'finish',
    'shrink_chat': 'trim',
    'drive': 'run',
    'drive_guarded': 'work',
    'seal_calls': 'answer_open_calls',
    'cycle': 'turns',
}

class ReedBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def basket_text(self) -> str:
        if self.spine.vault.folio.basket.bin.cart_view is not None and self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at:
            return FormKit.pack(FormKit.shorten(self.spine.vault.folio.basket.bin.cart_view, 300))[:3000]
        return "lines by God's record of the cart calls (listing: units): " + FormKit.pack(self.spine.vault.folio.basket.bin.cart_lines)

    def series_miss(self, part: int, main: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.spine.route.series_web(part, main, index)
        if not system:
            return []
        carrier, _, families = system
        marks = index['marks']
        faults = []
        for line in lines:
            carried, namer = (line, device) if carrier == part else (device, line)
            if self.spine.route.series_hit(carried, namer, families, index) is not False:
                continue
            wanted = set(marks[namer].get('named') or []) & families
            good = [i for i in self.spine.route.ids_of_item(carrier) if set(marks[i].get('fam') or []) & wanted and not self.spine.route.is_gone(i) and i != carried][:4]
            faults.append(f"{carried} is of series {', '.join(sorted(set(marks[carried].get('fam') or []) & families))}, while {namer} takes {', '.join(sorted(wanted))}" + (f"; seen listings of that series: {', '.join(good)}" if good else f"; search that series with the item's words, or take another {self.spine.vault.folio.order.sheet.items[main - 1].name} whose series the wanted lines carry"))
        return faults

    def can_fill_short(self) -> bool:
        """Before a forced order: add, for each item with no line, the cheapest opened listing that states its requirements.
        False when an item stays uncovered, so the episode ends by refusing rather than with a short order."""
        cart = self.spine.vault.folio.order.sheet.roles.get('cart') or ''
        bound = {n for pair in self.spine.vault.stacks.chains.web.bound for n in pair}
        for number in self.spine.route.items_absent():
            item = self.spine.vault.folio.order.sheet.items[number - 1]
            dead = any((self.spine.route.ref_spent(number, need) for n, need in self.spine.vault.stacks.chains.web.refs if n == number))
            fits = [i for i in self.spine.route.ids_of_item(number) if i in self.spine.vault.stacks.pages.book.full and i not in self.spine.vault.folio.basket.bin.cart_lines and self.spine.route.in_stock(i) and self.spine.route.price_num(i) is not None and self.spine.route.meets_needs(i, number, dead)]
            fits.sort(key=lambda i: self.spine.route.price_num(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.spine.vault.folio.order.sheet.budget, (int, float)):
                held = sum(((self.spine.route.price_num(i) or 0.0) * (self.spine.route.units_of(i) or 0) for i in self.spine.route.basket_ids()))
                cost = (self.spine.route.price_num(pick) or 0.0) * (1 if self.spine.vault.folio.order.sheet.brief.get('budget_scope') == 'each' else item.qty)
                if cost + (0.0 if self.spine.vault.folio.order.sheet.brief.get('budget_scope') == 'each' else held) > self.spine.vault.folio.order.sheet.budget + 1e-09:
                    pick = None
            calls = self.spine.route.basket_calls({**{i: self.spine.vault.folio.basket.bin.cart_lines[i] for i in self.spine.route.basket_ids() if i in self.spine.vault.folio.basket.bin.cart_lines}, pick: item.qty}, []) if pick else None
            if not calls or len(calls) + 1 > self.spine.route.steps_open():
                return False
            for args in calls:
                faults: list[str] = []
                args = FormKit.fit_value(args, self.spine.vault.counter.front.schemas.get(cart), 'args', faults)
                if faults or self.spine.route.call_store(cart, args) is None:
                    return False
                self.spine.route.apply_basket(cart, args)
            self.spine.vault.scroll.leaf.trace.append({'kind': 'filled_line', 'item': number, 'line': pick})
        return True

    def judge_basket(self, ids: list[str] | None=None, limit: float=Charter.SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.spine.route.basket_ids()
        key = self.spine.route.judge_key(ids)
        if key in self.spine.vault.folio.basket.bin.checked or not self.spine.route.model_order(limit) or (not ids):
            return ''
        self.spine.vault.folio.basket.bin.checked.add(key)
        prompt = Charter.CHECK + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.spine.route.must_sheet()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.spine.vault.folio.order.sheet.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.spine.route.basket_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.spine.route.is_gone(i) else '') + self.spine.route.record_blob(i) for i in ids))
        if self.spine.vault.folio.order.sheet.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.spine.route.side_records(ids, 15)
        judges = self.spine.route.model_order(limit)
        got = None
        if len(judges) > 1 and not lead and not self.spine.vault.folio.basket.bin.audited and self.spine.vault.folio.order.sheet.brief.get('pick') not in ('cheapest', 'most', 'least'):
            got = self.spine.route.ask_object(prompt, [Charter.LIGHT], 3000)
            if got:
                judges = [Charter.HEAVY] if got.get('ok') is False and Charter.HEAVY in self.spine.route.model_order(limit) else []
        if judges and not self.spine.vault.clockwork.pace.no_credit:
            got = self.spine.route.ask_object(prompt, judges, 3000) or got
        self.spine.vault.scroll.leaf.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.spine.vault.folio.basket.bin.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.spine.vault.folio.basket.bin.leads = [i for i in self.spine.vault.folio.basket.bin.leads if i not in ids]
        problems = [str(p) for p in FormKit.as_items(got.get('problems'))][:5]
        better = [str(b) for b in FormKit.as_items(got.get('better')) if str(b) in self.spine.vault.stacks.pages.book.rows and (not self.spine.route.is_gone(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def skip_maker(self) -> tuple[Any, ...]:
        """The row keys that cannot hold the name of who sells or makes a listing: its id, its label and its name. Any other
        text field can, and the value must still appear word for word in the description being matched."""
        return (self.spine.vault.stacks.pages.book.id_key, self.spine.route.shelf_field(), self.spine.route.title_slot())

    def reread_gone(self) -> str:
        ids = [i for i in self.spine.route.basket_ids() if i not in self.spine.vault.folio.basket.bin.gone]
        if any((self.spine.route.is_gone(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.spine.vault.folio.basket.bin.unavailable.add(ids[0])
            return ''
        return self.spine.route.reload_offers() if ids else ''

    def call_checks_out(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.spine.vault.folio.order.sheet.roles.get('buy'):
            return False
        fixed = self.spine.vault.folio.order.sheet.brief.get('buy_args') if isinstance(self.spine.vault.folio.order.sheet.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    @staticmethod
    def shelf_clash(brief: dict[str, Any]) -> bool:
        shelves = [GlyphKit.speak(str(e.get('shelf') or '')) for e in FormKit.as_items(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def find_ident(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.spine.vault.stacks.pages.book.id_key and FormKit.id_shaped(node.get(self.spine.vault.stacks.pages.book.id_key), True):
            return node[self.spine.vault.stacks.pages.book.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = FormKit.learn_ident([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.spine.vault.stacks.pages.book.rows):
            return node[found].strip()
        return None

    def shelf_groups(self) -> list[list[str]]:
        cat_key = self.spine.route.shelf_field()
        groups: dict[str, list[str]] = {}
        for ident in self.spine.vault.stacks.pages.book.seen_order:
            label = str(self.spine.vault.stacks.pages.book.rows[ident].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(ident)
        wanted = [(GlyphKit.speak(item.shelf or ''), GlyphKit.speak(item.name), {w for w in GlyphKit.stemset(item.shelf or '') | GlyphKit.stemset(item.name) if len(w) >= 4}) for item in self.spine.vault.folio.order.sheet.items]
        matched = [ids for label, ids in groups.items() if any((GlyphKit.speak(label) in (shelf, name) or {w for w in GlyphKit.stemset(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.spine.route.shelf_owner(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.spine.vault.folio.order.sheet.items))]

    def page_span(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.spine.route.learn_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def show_obs(self, observation: Any) -> str:
        text = FormKit.pack(FormKit.shorten(observation, Charter.SHOW_CHARS if self.spine.route.rows_inside(observation) else Charter.TEXT_CHARS))
        if len(text) > Charter.OBS_LIMIT:
            text = text[:Charter.OBS_LIMIT] + '...(cut)'
        return text

    def notes_spare(self) -> int:
        return self.spine.vault.folio.order.sheet.msg_limit - self.spine.vault.folio.order.sheet.msg_used - max(0, self.spine.vault.folio.order.sheet.reserve - self.spine.vault.folio.order.sheet.gate_msgs)

    def title_code_ids(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing tool called with a code that one seen row's title holds gets that row's id; anything else stays as it was sent."""
        if name not in {self.spine.vault.folio.order.sheet.roles.get(job) for job in ('open', 'offer', 'compare', 'cart')} or not self.spine.vault.stacks.pages.book.rows:
            return args
        title = self.spine.route.title_slot()
        if not title:
            return args

        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}

        def known(value: Any, key: str) -> Any:
            if not isinstance(value, str) or value.strip() in self.spine.vault.stacks.pages.book.rows or not 3 <= len(value.strip()) <= 30 or not re.search('\\d', value) or 'enum' in (props.get(key) or {}):
                return value
            pattern = '(?<![a-z0-9])' + re.escape(value.strip().lower()) + '(?![a-z0-9])'
            hits = [i for i, row in self.spine.vault.stacks.pages.book.rows.items() if re.search(pattern, str(row.get(title) or '').lower())]
            if len(hits) == 1:
                self.spine.vault.scroll.leaf.trace.append({'kind': 'row_id', 'sent': value, 'id': hits[0]})
                return hits[0]
            return value
        return {key: [known(v, key) for v in value] if isinstance(value, list) else known(value, key) for key, value in args.items()}

    def open_for_links(self, ids: list[str], keep: int) -> None:
        """Run the part code a cart line sends the reader to through the search, once per code, and open
        the page it finds when steps allow; the result feeds the ledger, not the model. Multi-line carts only."""
        plan = self.spine.route.value_form()
        if not plan or (len(self.spine.vault.folio.order.sheet.items) < 2 and len(ids) < 2):
            return
        tool, base, field = plan
        for ident in ids:
            marks = self.spine.route.compat_index()['marks']
            code = (marks.get(ident) or {}).get('more')
            if not code or code in self.spine.vault.stacks.chains.web.link_tried or any((mark.get('own') == code for mark in marks.values())):
                continue
            if self.spine.route.steps_open() <= keep or self.spine.vault.clockwork.pace.finished or self.spine.route.pace_pinched():
                return
            self.spine.vault.stacks.chains.web.link_tried.add(code)
            faults: list[str] = []
            args = FormKit.fit_value({**base, field: code}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
            if faults:
                return
            reply = self.spine.route.call_store(tool, args)
            if reply is None:
                return
            self.spine.route.render_obs(tool, reply.get('observation'))
            self.spine.vault.scroll.leaf.trace.append({'kind': 'show_link', 'line': ident, 'code': code})
            marks = self.spine.route.compat_index()['marks']
            self.spine.route.open_for_specs([i for i, mark in marks.items() if mark.get('own') == code], max(keep, self.spine.route.reserve_steps() + 6))

    def hunt_phrase(self, number: int) -> tuple[str, list[str]]:
        """The item's noun, a word its store label holds that no other item's label holds (a search reads labels too, so it
        keeps the rows to this item), with the words a further search can add, most promising first: a word on most rows finds
        nothing new and a word on one row finds little else, so the words kept sit between."""
        cat_key = self.spine.route.shelf_field()
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        title = self.spine.route.title_slot()
        labels = [str(row.get(cat_key) or '') for row in self.spine.vault.stacks.pages.book.rows.values() if self.spine.route.label_owned(item, str(row.get(cat_key) or ''))] if cat_key else []
        names = [str(row.get(title) or '') for row in self.spine.vault.stacks.pages.book.rows.values() if title and cat_key and self.spine.route.label_owned(item, str(row.get(cat_key) or ''))]
        for record in self.spine.vault.stacks.pages.book.full.values():
            if cat_key and self.spine.route.label_owned(item, str(record.get(cat_key) or '')):
                names.append(' '.join((str(v) for v in (FormKit.fact_map(record) or {}).values() if isinstance(v, str) and len(v) <= Charter.HUNT_VALUE_CHARS)))
        others = {w for n, other in enumerate(self.spine.vault.folio.order.sheet.items, 1) if n != number for w in re.findall('[a-z]{4,}', ((other.shelf or '') + ' ' + other.name).lower())}
        label = max(labels, key=labels.count).lower() if labels else (item.shelf or item.name).lower()
        nouns = [w for w in re.findall('[a-z]{4,}', label) if w not in others]
        noun = '' if number in self.spine.vault.stacks.probe.log.hunt_no_noun else (max(nouns, key=len) if nouns else '')
        if not names:
            return (noun, [])
        counts: dict[str, int] = {}
        for name in names:
            for word in {w.lower() for w in re.findall("[A-Za-z][A-Za-z'-]{2,}", name)}:
                counts[word] = counts.get(word, 0) + 1
        own = BlockKit.rootset(item.shelf or '') | BlockKit.rootset(item.name)
        ceiling = max(2, int(len(names) * Charter.HUNT_COMMON))
        words = [w for w, n in counts.items() if 2 <= n <= ceiling and w != noun and BlockKit.rootset(w) and not BlockKit.rootset(w) & own]
        words.sort(key=lambda w: (-counts[w], w))
        return (noun, words)

    def query_fresh(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this search (its arguments apart from numbers) is the first of its kind, and remember it."""
        mark = FormKit.pack({k: v for k, v in args.items() if isinstance(v, bool) or not isinstance(v, int)})
        seen = self.spine.vault.stacks.pages.book.page_queries.setdefault(name, set())
        fresh = mark not in seen
        seen.add(mark)
        return fresh

    def value_bits(self, number: int) -> list[tuple[str, bool]]:
        """Search words for each requirement value of the item a words search can match, each with whether an answer
        gave it: the value itself, or for an answer's lower numeric limit with a unit its number and
        unit, since such a hidden limit often sits on the value of a listing that fits."""
        out = []
        told = self.spine.vault.folio.order.sheet.items[number - 1].told
        for n, need in self.spine.route.live_needs():
            if n != number or need in [m for _, m in self.spine.vault.stacks.chains.web.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            edge = re.search(Charter.FLOOR, value, re.I)
            if edge:
                if need in told:
                    out.append((edge.group(1).strip(), True))
                continue
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append((value, need in told))
        return out

    def call_batch(self, name: str, batch: list[dict[str, Any]]) -> bool:
        """Run several calls of one tool as one grouped environment call (the store runs them as consecutive steps) and hold
        each result for act() to return; False when the store did not take the group, and the calls then run one by one."""
        payload = dict(self.spine.vault.counter.front.binding)
        payload['step'] = self.spine.vault.clockwork.pace.step + 1
        payload['actions'] = [{'name': name, 'args': args} for args in batch]
        rest = self.spine.vault.clockwork.pace.env_at + self.spine.vault.clockwork.pace.env_gap - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        self.spine.vault.clockwork.pace.env_at = time.monotonic()
        began = time.monotonic()
        result = Pipes.env.post_verbose('/environment/call', payload)
        self.spine.vault.clockwork.pace.env_spent += time.monotonic() - began
        self.spine.vault.clockwork.pace.env_count += 1
        data = result.data
        if not isinstance(data, dict) or not isinstance(data.get('observations'), list) or not data['observations']:
            error = result.error or {}
            status = error.get('status')
            self.spine.vault.scroll.leaf.trace.append({'kind': 'batch_fail', 'status': status, 'body': str(error.get('body') or '')[:200]})
            if status in (400, 404, 405, 422) or isinstance(data, dict):
                self.spine.vault.stacks.probe.log.batch_ok = False
            if status == 402:
                self.spine.vault.clockwork.pace.broke = True
            if status == 409 and any((w in str(error.get('body') or '').lower() for w in ('finish', 'quarantin', 'finaliz'))):
                self.spine.vault.clockwork.pace.finished = True
            return False
        seen = data['observations']
        first = data['step'] - len(seen) if isinstance(data.get('step'), int) else self.spine.vault.clockwork.pace.step
        for index, (args, observation) in enumerate(zip(batch, seen)):
            last = index == len(seen) - 1
            reply = {'observation': observation, 'finished': bool(data.get('finished')) and last, 'step': first + index + 1}
            self.spine.vault.stacks.probe.log.prefetched[name + FormKit.pack(args)] = self.spine.route.book_result(name, args, reply)
        self.spine.vault.scroll.leaf.trace.append({'kind': 'batch', 'tool': name, 'calls': len(batch), 'ran': len(seen)})
        return True

    def decline_ready(self) -> bool:
        refuse = self.spine.vault.folio.order.sheet.roles.get('refuse')
        if not refuse:
            return False
        faults: list[str] = []
        FormKit.fit_value(self.spine.route.quiet_args(refuse, dict(self.spine.vault.folio.order.sheet.brief.get('refuse_args') or {})), self.spine.vault.counter.front.schemas.get(refuse), 'args', faults)
        return not faults

    def spread_asks(self) -> None:
        tool = self.spine.vault.folio.order.sheet.roles.get('compare')
        schema = self.spine.vault.counter.front.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in FormKit.kind_list(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.spine.route.compare_is_thin():
            self.spine.route.spread_one()
            return
        if self.spine.route.steps_open() <= 12:
            return
        cap = min(props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4, self.spine.route.compare_width())
        ranked = sorted(self.spine.route.shelf_groups(), key=lambda ids: sum((1 for i in ids if i in self.spine.vault.stacks.pages.book.full)))
        lanes = [[i for i in ids if i not in self.spine.vault.stacks.pages.book.full] for ids in ranked]
        picks: list[str] = []
        for round_of in range(cap):
            picks += [lane[round_of] for lane in lanes if round_of < len(lane)]
        picks = picks[:cap]
        if len(picks) < 2:
            self.spine.route.open_for_specs(picks, 12)
            return
        faults: list[str] = []
        args = FormKit.fit_value({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.spine.vault.stacks.pages.book.full)
        reply = self.spine.route.call_store(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.spine.route.render_obs(tool, seen)
            self.spine.route.mark_view(tool, seen)
            rows = self.spine.route.rows_inside(seen)
            if rows:
                self.spine.vault.stacks.pages.book.facts_from[tool] = self.spine.vault.stacks.pages.book.facts_from.get(tool, False) or any((FormKit.fact_map(r) for _, r in rows))
                self.spine.vault.stacks.chains.web.compare_ok = max(self.spine.vault.stacks.chains.web.compare_ok, len(rows))
            else:
                self.spine.vault.stacks.chains.web.compare_bad = min(self.spine.vault.stacks.chains.web.compare_bad or 999, len(picks))
            self.spine.vault.scroll.leaf.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.spine.vault.stacks.pages.book.full) - before})
            if rows and self.spine.vault.stacks.pages.book.facts_from.get(tool) is False:
                self.spine.route.spread_one()
                return
            if len(self.spine.vault.stacks.pages.book.full) == before and len(picks) > 2 and (self.spine.route.steps_open() > 12):
                faults = []
                args = FormKit.fit_value({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.spine.route.call_store(tool, args)
                    if reply is not None:
                        self.spine.route.render_obs(tool, reply.get('observation'))
                        self.spine.vault.stacks.chains.web.compare_ok = max(self.spine.vault.stacks.chains.web.compare_ok, len(self.spine.route.rows_inside(reply.get('observation'))))

    def close_ask(self, number: int, labels: list[str], shown: bool, batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Read a sent question's answer into the items' needs and settle the labels it named."""
        self.spine.vault.folio.voice.tape.lane_spent[number] = self.spine.vault.folio.voice.tape.lane_spent.get(number, 0) + 1
        self.spine.vault.folio.voice.tape.list_asks += 1
        had = [len(item.told) for item in self.spine.vault.folio.order.sheet.items]
        heard = self.spine.route.read_replies(batch, (Charter.LIGHT, Charter.HEAVY))
        answer = batch[0][2] if batch else ''
        free_hit = self.spine.route.free_index(labels, heard, answer) if len(labels) > 1 else -1
        if len(labels) > 1:
            heard.pop('free', None)
        self.spine.route.apply_reply(heard, told)
        new = [need for item, size in zip(self.spine.vault.folio.order.sheet.items, had) for need in item.told[size:]]
        for n, (item, size) in enumerate(zip(self.spine.vault.folio.order.sheet.items, had), 1):
            if len(item.told) > size:
                self.spine.vault.folio.voice.tape.lane_found[n] = self.spine.vault.folio.voice.tape.lane_found.get(n, 0) + 1
        crowded = any((FormKit.as_count(x) == 1 for x in FormKit.as_items(heard.get('crowded'))))
        self.spine.route.land_answer(number, labels, shown, new, crowded, answer, free_hit)

    def upcoming_asks(self, told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """Send the next question (a pending follow-up first, else the next planned labels), read its answer into the
        items' needs (appending each new one to `told`) and settle the labels it named. Returns the sent batch, or []
        when nothing was sent."""
        for _ in range(Charter.LIST_SIZE):
            self.spine.vault.folio.voice.tape.follow_up = [f for f in self.spine.vault.folio.voice.tape.follow_up if not self.spine.route.already_said(f[0], f[1][0])]
            if self.spine.vault.folio.voice.tape.follow_up:
                number, labels, shown = self.spine.vault.folio.voice.tape.follow_up.pop(0)
            else:
                number, labels = self.spine.route.next_lane(self.spine.vault.folio.voice.tape.lanes, self.spine.vault.folio.voice.tape.lane_found, self.spine.vault.folio.voice.tape.lane_spent, self.spine.vault.folio.voice.tape.lane_used)
                shown = False
            if not labels:
                return []
            fitted, text = self.spine.route.trim_ask(number, labels, shown)
            if not fitted and shown:
                fitted, text = self.spine.route.trim_ask(number, labels, False)
            if fitted:
                break
            self.spine.vault.folio.voice.tape.lane_used.setdefault(number, set()).add(GlyphKit.speak(labels[0]))
            self.spine.vault.scroll.leaf.trace.append({'kind': 'ask_too_long', 'item': number, 'label': labels[0]})
        else:
            return []
        labels = fitted
        batch = self.spine.route.dispatch_asks([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.spine.route.close_ask(number, labels, shown, batch, told)
        return batch

    def shelf_owner(self, label: str) -> int:
        if not label:
            return 0
        if label in self.spine.vault.folio.voice.tape.adopted:
            return self.spine.vault.folio.voice.tape.adopted[label]
        known = self.spine.vault.stacks.pages.book.owners_seen.get((label, len(self.spine.vault.folio.order.sheet.items)))
        if known is not None:
            return known
        words = GlyphKit.speak(label)
        long = BlockKit.rootset(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            if words == GlyphKit.speak(item.shelf or ''):
                score = 3.0
            elif words == GlyphKit.speak(item.name):
                score = 2.0
            else:
                mine = BlockKit.rootset(item.shelf or '') | BlockKit.rootset(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        self.spine.vault.stacks.pages.book.owners_seen[(label, len(self.spine.vault.folio.order.sheet.items))] = best
        return best

    def dispatch_asks(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.spine.vault.folio.order.sheet.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.spine.vault.clockwork.pace.finished or self.spine.route.steps_open() <= floor or self.spine.vault.folio.order.sheet.msg_used >= self.spine.vault.folio.order.sheet.msg_limit:
                break
            args = self.spine.route.note_args(text)
            if not args:
                break
            reply = self.spine.route.call_store(tool, args)
            if reply is None:
                break
            self.spine.vault.folio.order.sheet.msg_used += 1
            answer = reply.get('observation')
            words = self.spine.route.voice_text(answer)
            sent.append((number, text, words))
            self.spine.vault.folio.order.sheet.done_log.append(f'step {self.spine.vault.clockwork.pace.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.spine.route.basket_sig()})')
            self.spine.vault.folio.order.sheet.done_steps.append(self.spine.vault.clockwork.pace.step)
            self.spine.vault.scroll.leaf.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

class ClayBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def units_of(self, ident: str) -> int:
        if self.spine.vault.folio.basket.bin.cart_view is not None and self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at:
            stack: list[Any] = [self.spine.vault.folio.basket.bin.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [FormKit.as_count(v) for k, v in node.items() if not isinstance(v, str) and FormKit.as_count(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', GlyphKit.speak(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.spine.vault.folio.basket.bin.cart_lines.get(ident, 1)

    def series_list(self, number: int) -> list[str]:
        """The series to search a linked item by, most useful first: those the cart's lines of the linked items hold, then those most rows of the
        linked items carry or name. [] when the item is linked by no series system."""
        index = self.spine.route.compat_index()
        marks = index['marks']
        counts: dict[str, int] = {}
        held: list[str] = []
        cart = self.spine.route.basket_ids()
        for part, main in self.spine.vault.stacks.chains.web.bound:
            system = self.spine.route.series_web(part, main, index) if number in (part, main) else None
            if not system:
                continue
            for member in (part, main):
                for i in self.spine.route.ids_of_item(member):
                    tags = set((marks.get(i) or {}).get('fam') or []) | set((marks.get(i) or {}).get('named') or [])
                    for family in tags & system[2]:
                        counts[family] = counts.get(family, 0) + 1
                        if i in cart and family not in held:
                            held.append(family)
        return (held + [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if f not in held])[:Charter.LINK_SEARCHES]

    def split_shelves(self) -> None:
        """An item that shares its shelf with another requested item takes the store label that is its own name, when one is seen
        and no item holds it yet."""
        cat_key = self.spine.route.shelf_field()
        labels = sorted({str(row.get(cat_key) or '').strip() for row in self.spine.vault.stacks.pages.book.rows.values()} - {''}) if cat_key else []
        moved = False
        for item in self.spine.vault.folio.order.sheet.items:
            name = BlockKit.rootset(item.name)
            if not name or not item.shelf.strip() or not any((other is not item and GlyphKit.speak(other.shelf) == GlyphKit.speak(item.shelf) for other in self.spine.vault.folio.order.sheet.items)):
                continue
            if any((other is not item and BlockKit.rootset(other.name) == name for other in self.spine.vault.folio.order.sheet.items)):
                continue
            own = [label for label in labels if BlockKit.rootset(label) == name and GlyphKit.speak(label) != GlyphKit.speak(item.shelf)]
            if len(own) == 1 and not any((GlyphKit.speak(other.shelf) == GlyphKit.speak(own[0]) for other in self.spine.vault.folio.order.sheet.items)):
                self.spine.vault.scroll.leaf.trace.append({'kind': 'shelf_parted', 'item': item.name, 'from': item.shelf, 'to': own[0]})
                item.shelf = own[0]
                moved = True
        if moved:
            self.spine.vault.stacks.pages.book.owners_seen = {}
            self.spine.vault.stacks.chains.web.fit_cached = None

    def code_picks(self) -> dict[str, list[str]]:
        """Candidates the code can vouch for before an ending without a purchase: for each item with no cart line, the read,
        buyable listings inside the budget whose attributes state every plain requirement of the item (a requirement whose
        pointed-at page was not found is left to the check), cheapest first, at most 3; {} when such an item has none.
        Race 17: 41 of God's 45 endings without a purchase were wrong and in 9 of them it had read a listing another
        agent passed with."""
        cat_key = self.spine.route.shelf_field()
        cap = self.spine.vault.folio.order.sheet.budget if isinstance(self.spine.vault.folio.order.sheet.budget, (int, float)) and (not isinstance(self.spine.vault.folio.order.sheet.budget, bool)) else None
        held = [str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '') for i in self.spine.route.basket_ids() if (self.spine.route.units_of(i) or 0) > 0 and (not self.spine.route.is_gone(i))] if cat_key else []
        out: dict[str, list[str]] = {}
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            if any((self.spine.route.label_owned(item, label) for label in held)):
                continue
            dead = any((self.spine.route.ref_spent(number, need) for n, need in self.spine.vault.stacks.chains.web.refs if n == number))
            fits = [i for i in self.spine.route.ids_of_item(number) if i in self.spine.vault.stacks.pages.book.full and self.spine.route.in_stock(i) and (not self.spine.route.is_gone(i)) and self.spine.route.meets_needs(i, number, dead)]
            if cap is not None:
                units = 1 if self.spine.vault.folio.order.sheet.brief.get('budget_scope') == 'each' else item.qty
                fits = [i for i in fits if self.spine.route.price_num(i) is not None and self.spine.route.price_num(i) * units <= cap + 1e-09]
            fits.sort(key=lambda i: self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf'))
            if not fits:
                return {}
            out[str(number)] = fits[:3]
        return out

    def shop_ids(self, about: str, ids: list[str]) -> list[str]:
        not_names = self.spine.route.skip_maker()
        low = about.lower()
        out = []
        for ident in ids:
            row = self.spine.vault.stacks.pages.book.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in Charter.NO_NAME]
            if ident not in self.spine.vault.stacks.pages.book.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def swap_ids(self, lost: list[str]) -> list[str]:
        if not self.spine.route.model_order():
            return []
        records = []
        for ident in reversed(self.spine.vault.stacks.pages.book.seen_order):
            if ident in lost or ident in self.spine.vault.folio.basket.bin.gone or self.spine.route.is_gone(ident):
                continue
            records.append(FormKit.pack(FormKit.shorten(self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = Charter.SWAP + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.spine.route.must_sheet()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.spine.route.record_blob(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.spine.route.basket_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.spine.route.ask_object(prompt, self.spine.route.model_order()[::-1], 3500)
        fits: list[tuple[float, str]] = []
        for entry in FormKit.as_items((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.spine.vault.stacks.pages.book.rows and ident not in lost and (ident not in self.spine.vault.folio.basket.bin.gone) and (ident not in self.spine.route.basket_ids()) and (ident not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.spine.vault.scroll.leaf.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        self.spine.vault.folio.basket.bin.swap_fits = ranked[:4]
        return ranked[:4]

    def call_declines(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.spine.vault.folio.order.sheet.roles.get('refuse'):
            return False
        fixed = self.spine.vault.folio.order.sheet.brief.get('refuse_args') if isinstance(self.spine.vault.folio.order.sheet.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def peel_foreign(self) -> None:
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            pointers = [need for n, need in self.spine.vault.stacks.chains.web.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = Charter.UNMIX + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.spine.route.ask_object(prompt, [Charter.HEAVY, Charter.LIGHT], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in FormKit.as_items(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.spine.vault.stacks.chains.web.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.spine.vault.stacks.chains.web.refs]
            self.spine.vault.scroll.leaf.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def learn_row_keys(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.spine.vault.stacks.pages.book.id_key and all((FormKit.id_shaped(r.get(self.spine.vault.stacks.pages.book.id_key), True) for r in rows)):
            return True
        found = FormKit.learn_ident(rows)
        if found:
            self.spine.vault.stacks.pages.book.id_key = found
            return True
        return False

    def one_id_arg(self, tool: str | None) -> str | None:
        schema = self.spine.vault.counter.front.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def price_order(self, observation: Any) -> bool:
        prices = [self.spine.route.price_num(i) for i, _ in self.spine.route.rows_inside(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def proof_lines(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.spine.route.find_ident(node, False, depth)
                if ident and FormKit.fact_map(node) and (ident not in ids):
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.spine.route.live_needs()
        if not ids or not needs:
            return ''
        many = len(self.spine.vault.folio.order.sheet.items) > 1
        lines = []
        for ident in ids[:12]:
            block = FormKit.fact_map(self.spine.vault.stacks.pages.book.full.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:80]}' for k, v in block.items() if NestKit.same_name(GlyphKit.stemset(prop), GlyphKit.stemset(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)

    def reserve_steps(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.spine.vault.folio.order.sheet.process))
        return 2 if talk and self.spine.vault.folio.order.sheet.msg_used < self.spine.vault.folio.order.sheet.msg_limit and self.spine.route.stage_pending('order') else 1

    def tool_has_rows(self, tool: str) -> bool:
        """Whether a tool lists a shelf or takes search words rather than a listing's id."""
        target = self.spine.route.one_id_arg(tool) or ''
        return bool(re.search(Charter.SHELF_ARG, GlyphKit.snake(target), re.I) or re.search(Charter.QUERY_WORDS, GlyphKit.snake(target), re.I)) or tool in self.spine.vault.stacks.pages.book.shapes

    @cushion('')
    def under_basket_note(self) -> str:
        """The buyable rows priced below a cart line, of its label, that nobody has read yet."""
        cat_key = self.spine.route.shelf_field()
        cart = self.spine.route.basket_ids()
        lines = []
        for line in cart:
            price = self.spine.route.price_num(line)
            kind = str((self.spine.vault.stacks.pages.book.rows.get(line) or {}).get(cat_key) or '').strip().lower() if cat_key else ''
            if price is None or not kind:
                continue
            pool = [i for i in self.spine.vault.stacks.pages.book.seen_order if i not in cart and i not in self.spine.vault.stacks.pages.book.full and self.spine.route.in_stock(i) and str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind and (self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: self.spine.route.price_num(i) or 0.0)
            lines += [f'{i} at {self.spine.route.price_num(i):g}' for i in pool[:Charter.CHEAPER_SHOWN]]
        return 'Still unread and below a cart line: ' + ', '.join(lines) + '.' if lines else ''

    def call_sig(self, name: str, args: dict[str, Any]) -> str:
        """One spelling per call: words in any case and spacing, a number at its first value left out."""
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}
        kept: dict[str, Any] = {}
        for key, value in sorted(args.items()):
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            first = value in (spec.get('minimum'), spec.get('default')) or (value == 1 and key == self.spine.vault.stacks.pages.book.page_args.get(name) and spec.get('minimum') is None and spec.get('default') is None)
            if isinstance(value, int) and not isinstance(value, bool) and first:
                continue
            kept[key] = ' '.join(value.lower().split()) if isinstance(value, str) else value
        return name + FormKit.pack(kept)

    def rows_until_keys(self, tool: str, args: dict[str, Any], seen: Any, stop_at: int, number: int) -> list[str]:
        """One further page when a first page shows too few of the item's listings to tell what they carry; what it showed."""
        page = self.spine.route.page_span(tool, seen)
        if not page or page[3] is None or self.spine.route.item_row_count(number) >= Charter.OPEN_EACH or self.spine.vault.clockwork.pace.finished or self.spine.vault.clockwork.pace.step >= stop_at:
            return []
        faults: list[str] = []
        call = FormKit.fit_value({**args, page[2]: page[3]}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
        if faults:
            return []
        reply = self.spine.route.call_store(tool, call)
        if reply is None:
            return []
        self.spine.route.render_obs(tool, reply.get('observation'))
        return [f'next page: ' + self.spine.route.show_obs(reply.get('observation'))]

    def value_form(self) -> tuple[str, dict[str, Any], str] | None:
        """The search a value search copies: search_template(), or, when every search so far listed a category without
        words, the tool's one query field with its one mode argument set to read it (the option naming words or a query,
        else the other option of a required two-option mode). None when the schema leaves a doubt."""
        plan = self.spine.route.word_form()
        tool = self.spine.vault.folio.order.sheet.roles.get('search')
        if plan or not tool:
            return plan
        schema = self.spine.vault.counter.front.schemas.get(tool) or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        told = self.spine.vault.folio.order.sheet.brief.get('search_words') if isinstance(self.spine.vault.folio.order.sheet.brief.get('search_words'), dict) else {}
        said, fixed = (told.get('field') if isinstance(told.get('field'), str) else None, told.get('args') if isinstance(told.get('args'), dict) else {})
        named_tool = told.get('tool') if isinstance(told.get('tool'), str) and told.get('tool') in self.spine.vault.counter.front.schemas else tool
        spec = (self.spine.vault.counter.front.schemas.get(named_tool) or {}).get('properties') or {}
        if said in spec and 'string' in FormKit.kind_list(spec[said]) and 'enum' not in spec[said]:
            faults: list[str] = []
            base = FormKit.fit_value({**fixed, said: 'x'}, self.spine.vault.counter.front.schemas.get(named_tool), 'args', faults)
            if not faults:
                base.pop(said, None)
                return (named_tool, base, said)
        named = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and 'enum' not in v and re.search(Charter.QUERY_WORDS, GlyphKit.snake(k), re.I)]
        last = self.spine.vault.stacks.probe.log.searched[-1] if self.spine.vault.stacks.probe.log.searched else {}
        enums = {k: v['enum'] for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2}
        modes = []
        for key, options in enums.items():
            reads = [o for o in options if isinstance(o, str) and re.search(Charter.QUERY_WORDS, o, re.I)]
            others = [o for o in options if o != last.get(key)]
            if len(reads) == 1:
                modes.append((key, reads[0]))
            elif key in required and len(options) == 2 and len(others) == 1:
                modes.append((key, others[0]))
        if len(named) != 1 or len(modes) > 1 or (enums and not modes):
            return None
        base = {k: last[k] for k in required if k in last and k != named[0]}
        base.update(modes)
        return (tool, base, named[0])

    def prime_opens(self, calls: list[dict[str, Any]]) -> None:
        """When a model turn opens several listings with the single-listing tool, open them in one grouped call before the
        calls run one by one (agent12's batching): each call then finds its result held. Only calls that would run as sent
        are grouped: the known tool, arguments its schema takes, a listing not shown already, not sold out, and no more
        than leave the tool offered afterwards (offered() hides it near the end, and a held result would then be refused)."""
        tool = self.spine.vault.folio.order.sheet.roles.get('open')
        target = self.spine.route.one_id_arg(tool)
        if not self.spine.vault.stacks.probe.log.batch_ok or not tool or not target or self.spine.vault.clockwork.pace.finished:
            return
        allowed = {t['function']['name'] for t in self.spine.route.tools_now()}
        batch: list[dict[str, Any]] = []
        marks: set[str] = set()
        idents: set[str] = set()
        for call in calls:
            fn = call.get('function') or {}
            if call.get('joined') or fn.get('name') != tool or tool not in allowed:
                continue
            raw = fn.get('arguments')
            args = raw if isinstance(raw, dict) else FormKit.object_of(raw) if isinstance(raw, str) and raw.strip() else {}
            if not isinstance(args, dict):
                continue
            faults: list[str] = []
            args = FormKit.fit_value(args, self.spine.vault.counter.front.schemas[tool], 'args', faults)
            if faults:
                continue
            args, wrong = self.spine.route.fix_id_args(tool, args)
            if wrong:
                continue
            args = self.spine.route.title_code_ids(tool, args)
            ident = str(args.get(target))
            mark = tool + FormKit.pack(args)
            cached = self.spine.vault.stacks.pages.book.read_cache.get(mark)
            if mark in marks or ident in idents or mark in self.spine.vault.folio.basket.bin.recent or (cached and cached[0] == self.spine.vault.stacks.pages.book.version) or ident in self.spine.vault.folio.basket.bin.gone or self.spine.route.seen_before(tool, ident) or (tool, ident) in self.spine.vault.stacks.pages.book.factless:
                continue
            marks.add(mark)
            idents.add(ident)
            batch.append(args)
        floor = max(self.spine.route.reserve_steps() + 3, len(self.spine.vault.folio.order.sheet.items) + 5 if self.spine.vault.folio.basket.bin.buys == 0 else 0)
        others = sum((1 for call in calls if not call.get('joined') and (call.get('function') or {}).get('name') != tool))
        batch = batch[:max(0, min(Charter.BATCH_OPENS, self.spine.route.steps_open() - floor - others - 1))]
        if len(batch) < 2:
            return
        self.spine.route.call_batch(tool, batch)

    def play_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.spine.route.fold_opens(calls)
        self.spine.route.prime_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.spine.route.tools_now()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: God opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.spine.vault.clockwork.pace.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.spine.vault.folio.order.sheet.roles.get('ask') and self.spine.route.note_args('x' * 200) is not None:
                out.append((cid, "Not run (no step used): God's interviewer handles every message to the shopper."))
                continue
            if name not in self.spine.vault.counter.front.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = FormKit.object_of(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = FormKit.fit_value(args, self.spine.vault.counter.front.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.spine.route.fix_id_args(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            if name == self.spine.vault.folio.order.sheet.roles.get('cart'):
                args = self.spine.route.fill_remove(name, args)
            trimmed = ''
            many = self.spine.route.compare_field() if name == self.spine.vault.folio.order.sheet.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.spine.route.compare_width()) and self.spine.vault.stacks.chains.web.compare_bad:
                room = self.spine.route.compare_width()
                trimmed = f"\n[God] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.spine.route.one_id_arg(name)
            if target and name in (self.spine.vault.folio.order.sheet.roles.get('open'), self.spine.vault.folio.order.sheet.roles.get('offer')) and (str(args.get(target)) in self.spine.vault.folio.basket.bin.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.spine.vault.folio.basket.bin.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.spine.vault.folio.order.sheet.roles.get('cart') and any(((FormKit.as_count(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = FormKit.pack(self.spine.vault.folio.basket.bin.cart_view) if self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.spine.vault.stacks.pages.book.rows and (v not in self.spine.vault.folio.basket.bin.cart_lines) and (v not in view) and (v not in self.spine.vault.folio.basket.bin.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.spine.route.basket_text()))
                    continue
            ending = self.spine.route.call_checks_out(name, args) or self.spine.route.call_declines(name, args)
            plain_jobs = {tool for job, tool in self.spine.vault.folio.order.sheet.roles.items() if job != 'search'}
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.spine.route.steps_open() <= self.spine.route.reserve_steps() and (self.spine.vault.folio.order.sheet.roles.get('buy') or self.spine.vault.folio.order.sheet.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            fresh = name not in plain_jobs and self.spine.route.query_fresh(name, args)
            asked_for = args
            args = self.spine.route.tune_first_query(name, args, fresh)
            if args != asked_for:
                trimmed += f"\n[God] This is a new search, so God asked for its first page ({FormKit.pack(asked_for)} was the call you sent)."
            args = self.spine.route.title_code_ids(name, args)
            if name == self.spine.vault.folio.order.sheet.roles.get('search') and self.spine.vault.stacks.probe.log.codes_dead and self.spine.route.query_has_code(args):
                out.append((cid, 'Not run (no step used): this store\'s search found nothing for a model or part code. Search by words and values, list the category, and read the codes on the rows.'))
                continue
            mark = name + FormKit.pack(args)
            held = mark in self.spine.vault.stacks.probe.log.prefetched
            if not ending and name != self.spine.vault.folio.order.sheet.roles.get('cart') and (mark in self.spine.vault.folio.basket.bin.recent) and (not held):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.spine.vault.folio.order.sheet.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.spine.vault.stacks.pages.book.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.spine.vault.stacks.pages.book.version) and (not held):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.spine.vault.folio.order.sheet.roles.get('compare')) and (not held) and (self.spine.route.seen_before(name, str(args.get(target))) or (name, str(args.get(target))) in self.spine.vault.stacks.pages.book.factless):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.spine.route.record_blob(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.spine.route.seen_before(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.spine.route.record_blob(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[God] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + FormKit.pack(args)
            if ending and self.spine.route.steps_open() > 2:
                if self.spine.route.call_checks_out(name, args):
                    hunted = self.spine.route.hunt_lower()
                    wider = self.spine.route.wide_hold_note()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    reread = self.spine.route.reload_offers()
                    if self.spine.vault.folio.basket.bin.note_sold:
                        self.spine.route.reread_gone()
                    if self.spine.vault.folio.basket.bin.cart_touched and self.spine.vault.folio.basket.bin.cart_seen_at < self.spine.vault.folio.basket.bin.cart_at and self.spine.vault.folio.order.sheet.roles.get('cart_view') and (self.spine.route.steps_open() > 5) and not self.spine.route.pace_pinched():
                        faults = []
                        view_args = FormKit.fit_value({}, self.spine.vault.counter.front.schemas.get(self.spine.vault.folio.order.sheet.roles['cart_view']), 'args', faults)
                        view = self.spine.route.call_store(self.spine.vault.folio.order.sheet.roles['cart_view'], view_args) if not faults else None
                        if view is not None:
                            self.spine.vault.folio.basket.bin.cart_view = view.get('observation')
                            self.spine.vault.folio.basket.bin.cart_seen_at = self.spine.vault.clockwork.pace.step
                            change = self.spine.route.render_obs(self.spine.vault.folio.order.sheet.roles['cart_view'], self.spine.vault.folio.basket.bin.cart_view)
                            if change:
                                self.spine.vault.folio.basket.bin.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.spine.route.basket_text() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.spine.route.swap_note()
                    held = [i for i in self.spine.route.basket_ids() if i in self.spine.vault.folio.basket.bin.gone] if self.spine.vault.folio.basket.bin.held < 2 else []
                    if swap or held:
                        self.spine.vault.folio.basket.bin.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    chain, hold = self.spine.route.walk_chains()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    misfit = self.spine.route.compat_miss() if self.spine.vault.stacks.chains.web.fit_holds < 2 and self.spine.route.steps_open() > self.spine.route.reserve_steps() + 3 else []
                    if misfit:
                        self.spine.vault.stacks.chains.web.fit_holds += 1
                        self.spine.vault.scroll.leaf.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    faults = self.spine.route.line_misses() if self.spine.vault.stacks.chains.web.line_holds < 2 and self.spine.route.steps_open() > self.spine.route.reserve_steps() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.spine.route.apply_qty(self.spine.route.qty_fixes())
                        faults = self.spine.route.line_misses() if fixed else faults
                        if fixed and (not faults):
                            self.spine.vault.stacks.chains.web.line_holds += 1
                            out.append((cid, "Not run yet (no step used). God set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.spine.route.basket_text()))
                            continue
                    if faults:
                        self.spine.vault.stacks.chains.web.line_holds += 1
                        self.spine.vault.scroll.leaf.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.spine.route.basket_text()))
                        continue
                    lines = self.spine.route.basket_ids()
                    read = self.spine.route.open_for_specs(lines, self.spine.route.reserve_steps() + (1 if len(lines) >= 2 else 3))
                    self.spine.route.open_for_links(lines, self.spine.route.reserve_steps() + 1)
                    read += self.spine.route.lower_ids()
                    verdict = self.spine.route.judge_basket()
                    if verdict:
                        seen = "God read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        seen = (hunted + ' ' + self.spine.route.under_basket_note() + '\n' if hunted else '') + seen
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    if reread and self.spine.vault.clockwork.pace.cost > Charter.SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). God re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.spine.route.steps_open() > 5:
                    pointed = self.spine.route.trail_on_refuse()
                    if pointed:
                        out.append((cid, 'Not run yet (no step used). ' + pointed))
                        continue
                    self.spine.route.hunt_matches()
                    review = self.spine.route.audit_empty()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if not self.spine.vault.folio.basket.bin.leads:
                        found = self.spine.route.code_picks()
                        if found:
                            self.spine.vault.folio.basket.bin.lead_items = found
                            self.spine.vault.folio.basket.bin.leads = [i for ids in found.values() for i in ids]
                            self.spine.vault.scroll.leaf.trace.append({'kind': 'code_leads', 'leads': found})
                    whole = self.spine.route.basket_compat()
                    if whole:
                        out.append((cid, 'Not run yet (no step used). ' + whole + '\nCart now: ' + self.spine.route.basket_text()))
                        continue
                    if self.spine.vault.folio.basket.bin.leads and self.spine.route.steps_open() > 4 and (self.spine.vault.folio.basket.bin.refuse_blocks == 0):
                        self.spine.vault.folio.basket.bin.refuse_blocks = 1
                        self.spine.vault.scroll.leaf.trace.append({'kind': 'refuse_blocked', 'leads': list(self.spine.vault.folio.basket.bin.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.spine.vault.folio.basket.bin.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.spine.vault.folio.basket.bin.leads and self.spine.route.steps_open() > 4 and (self.spine.vault.folio.basket.bin.refuse_blocks == 1):
                        self.spine.vault.folio.basket.bin.refuse_blocks = 2
                        first = self.spine.vault.folio.basket.bin.leads[0]
                        self.spine.route.judge_basket([first], Charter.SPEND_HARD)
                        if self.spine.vault.folio.basket.bin.verdicts.get(self.spine.route.judge_key([first])) and first in self.spine.vault.folio.basket.bin.leads:
                            self.spine.vault.folio.basket.bin.confirmed = first
                            self.spine.vault.scroll.leaf.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.spine.route.call_declines(name, args) and 2 <= self.spine.route.steps_open() <= 5 and (not self.spine.vault.folio.basket.bin.completed) and (not self.spine.vault.folio.basket.bin.confirmed):
                found = self.spine.route.short_picks()
                first = next((i for ids in found.values() for i in ids), None)
                if first:
                    self.spine.vault.folio.basket.bin.lead_items = found
                    self.spine.vault.folio.basket.bin.leads = [i for ids in found.values() for i in ids]
                    self.spine.vault.folio.basket.bin.confirmed = first
                    self.spine.vault.scroll.leaf.trace.append({'kind': 'short_leads', 'leads': found, 'left': self.spine.route.steps_open()})
            if self.spine.route.call_declines(name, args) and self.spine.route.steps_open() <= 5 and (not self.spine.vault.folio.basket.bin.late_held):
                lines = [i for i in self.spine.vault.folio.basket.bin.built if not self.spine.route.is_gone(i)]
                fresh = self.spine.vault.folio.basket.bin.cart_view is not None and self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at
                missing = [i for i in lines if i not in self.spine.route.basket_ids() or (i in self.spine.vault.folio.basket.bin.dropped and (not fresh))]
                if len(lines) >= len(self.spine.vault.folio.order.sheet.items) and len(missing) + 1 <= self.spine.route.steps_open():
                    self.spine.vault.folio.basket.bin.late_held = True
                    self.spine.vault.scroll.leaf.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.spine.route.call_declines(name, args) and (not self.spine.vault.folio.basket.bin.leads) and (not self.spine.vault.folio.basket.bin.pushed) and (self.spine.route.steps_open() > self.spine.vault.clockwork.pace.max_steps // 3):
                self.spine.vault.folio.basket.bin.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.spine.route.steps_open()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.spine.route.call_declines(name, args):
                done = self.spine.route.build_basket()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.spine.vault.clockwork.pace.finished else '')))
                    continue
                args = self.spine.route.quiet_args(name, args)
            if self.spine.route.call_checks_out(name, args):
                late = self.spine.route.late_ask()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.spine.route.basket_text()))
                    continue
            if ending and self.spine.route.steps_open() >= 2:
                go, said = self.spine.route.pass_stage('order' if self.spine.route.call_checks_out(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.spine.route.basket_text()))
                    continue
                if said:
                    trimmed = '\n[God] ' + said + trimmed
            reply = self.spine.route.call_store(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = FormKit.pack(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.spine.vault.stacks.chains.web.compare_bad = min(self.spine.vault.stacks.chains.web.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[God] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.spine.route.compare_width()} per call.'
                else:
                    self.spine.vault.stacks.chains.web.compare_ok = max(self.spine.vault.stacks.chains.web.compare_ok, len(args[many]))
            self.spine.vault.folio.basket.bin.recent = (self.spine.vault.folio.basket.bin.recent + [mark])[-3:]
            self.spine.vault.stacks.pages.book.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.spine.vault.stacks.pages.book.rows for x in (v if isinstance(v, list) else [v])))))
            self.spine.route.take_obs(observation)
            if name == self.spine.vault.folio.order.sheet.roles.get('compare'):
                rows = self.spine.route.rows_inside(observation)
                if rows:
                    self.spine.vault.stacks.pages.book.facts_from[name] = self.spine.vault.stacks.pages.book.facts_from.get(name, False) or any((FormKit.fact_map(r) for _, r in rows))
            if name in readers - {self.spine.vault.folio.order.sheet.roles.get('search')}:
                self.spine.route.mark_view(name, observation)
            plain = {tool for job, tool in self.spine.vault.folio.order.sheet.roles.items() if job != 'search'}
            paged = self.spine.route.next_page_note(name, args, observation) if name not in plain and (not ending) else ''
            paged += self.spine.route.flip_last_page(name, args, observation) if fresh and (not ending) else ''
            paged += self.spine.route.short_query_note(name, args, observation) if name not in plain and (not ending) else ''
            if self.spine.route.one_id_arg(name) and name not in (self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse'), self.spine.vault.folio.order.sheet.roles.get('ask')):
                listing = self.spine.route.find_ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.spine.route.mark_spec_tool(name, bool(FormKit.fact_map(observation)))
                    if name == self.spine.vault.folio.order.sheet.roles.get('open') and (self.spine.vault.stacks.pages.book.facts_from.get(name) is False):
                        other = next((t for t in self.spine.vault.counter.front.schemas if t != name and self.spine.route.one_id_arg(t) and (self.spine.vault.stacks.pages.book.facts_from.get(t) is not False) and (t not in (self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse'), self.spine.vault.folio.order.sheet.roles.get('ask'), self.spine.vault.folio.order.sheet.roles.get('search'))) and not self.spine.route.tool_has_rows(t)), None)
                        if other:
                            self.spine.vault.folio.order.sheet.roles['open'] = other
                    if self.spine.vault.stacks.pages.book.facts_from.get(name) is False and name != self.spine.vault.folio.order.sheet.roles.get('open') and (not self.spine.vault.folio.order.sheet.roles.get('offer')):
                        self.spine.vault.folio.order.sheet.roles['offer'] = name
                    if name == self.spine.route.offer_name():
                        self.spine.vault.folio.basket.bin.offer_seen[listing] = self.spine.vault.clockwork.pace.step
            if name == self.spine.vault.folio.order.sheet.roles.get('cart'):
                self.spine.route.apply_basket(name, args)
                self.spine.vault.folio.basket.bin.cart_said = FormKit.pack(observation)
                self.spine.vault.folio.basket.bin.recent = []
                counts = [FormKit.as_count(v) for v in args.values() if not isinstance(v, str) and FormKit.as_count(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.spine.vault.stacks.pages.book.rows):
                        if counts[0] <= 0 and (not FormKit.ident_inside(observation, value)):
                            self.spine.vault.folio.basket.bin.dropped.add(value)
                        elif counts[0] > 0:
                            self.spine.vault.folio.basket.bin.dropped.discard(value)
                lines = [i for i in self.spine.route.basket_ids() if not self.spine.route.is_gone(i) and i not in self.spine.vault.folio.basket.bin.dropped]
                if len(lines) >= len(self.spine.vault.folio.basket.bin.built):
                    self.spine.vault.folio.basket.bin.built = lines
            if name == self.spine.vault.folio.order.sheet.roles.get('search') and self.spine.route.query_has_code(args):
                self.spine.vault.stacks.probe.log.code_tries.append(bool(self.spine.route.rows_inside(observation)))
                self.spine.vault.stacks.probe.log.codes_dead = len(self.spine.vault.stacks.probe.log.code_tries) >= 2 and not any(self.spine.vault.stacks.probe.log.code_tries)
            if name == self.spine.vault.folio.order.sheet.roles.get('search') and self.spine.route.rows_inside(observation):
                self.spine.vault.stacks.probe.log.searched = (self.spine.vault.stacks.probe.log.searched + [dict(args)])[-8:]
            if name == self.spine.vault.folio.order.sheet.roles.get('cart_view'):
                self.spine.vault.folio.basket.bin.cart_view = observation
                self.spine.vault.folio.basket.bin.cart_seen_at = self.spine.vault.clockwork.pace.step
            if self.spine.route.call_checks_out(name, args):
                self.spine.vault.folio.basket.bin.buys += 1
                if not reply.get('finished'):
                    self.spine.vault.folio.basket.bin.refused_at = self.spine.vault.clockwork.pace.step
            text = self.spine.route.show_obs(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.spine.route.low_price_note([observation] + self.spine.vault.stacks.probe.log.paged_obs) + self.spine.route.cap_miss_note(name, args, observation)
            base = text
            proof = self.spine.route.proof_lines(observation)
            if proof:
                text += '\n' + proof
            if name == self.spine.vault.folio.order.sheet.roles.get('cart') and any((FormKit.as_count(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.spine.vault.stacks.pages.book.rows and FormKit.ident_inside(observation, v) for v in args.values())):
                    text += '\n[God] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.spine.vault.folio.basket.bin.note_sold = False
            lines = self.spine.route.note_bits(observation)
            if lines:
                meaning = self.spine.route.read_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.spine.route.call_checks_out(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.spine.route.voice_text(observation) for line in self.spine.vault.folio.basket.bin.sold_lines))
            loose = self.spine.vault.folio.basket.bin.note_sold and (refused or (not self.spine.vault.folio.basket.bin.note_named and (not self.spine.route.call_checks_out(name, args))))
            if (loose or again) and self.spine.route.basket_ids() and (not self.spine.vault.clockwork.pace.finished):
                seen = self.spine.route.reread_gone()
                if seen:
                    text += "\n[God re-read the cart's lines after the refusal]\n" + seen
            self.spine.vault.folio.basket.bin.note_sold = False
            swap = self.spine.route.swap_note()
            if swap:
                text += '\n' + swap
            if name == self.spine.vault.folio.order.sheet.roles.get('cart') and self.spine.vault.stacks.chains.web.bound and not self.spine.vault.clockwork.pace.finished:
                misfit = self.spine.route.compat_miss()
                if misfit:
                    text += '\n[God] These lines do not work together yet: ' + '; '.join(misfit)
            if self.spine.vault.folio.basket.bin.queued:
                text += "\n[notes seen on God's own calls] " + ' | '.join(self.spine.vault.folio.basket.bin.queued)
                self.spine.vault.folio.basket.bin.queued = []
            if name in readers:
                self.spine.vault.stacks.pages.book.read_cache[mark] = (self.spine.vault.stacks.pages.book.version, self.spine.vault.clockwork.pace.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        if self.spine.vault.stacks.probe.log.prefetched:
            extra = []
            for mark in list(self.spine.vault.stacks.probe.log.prefetched):
                reply = self.spine.vault.stacks.probe.log.prefetched.pop(mark)
                observation = reply.get('observation')
                self.spine.route.take_obs(observation)
                extra.append(self.spine.route.show_obs(observation)[:1500])
            if extra and out:
                cid, text = out[-1]
                out[-1] = (cid, text + '\n[God also opened these in the same grouped call]\n' + '\n'.join(extra))
            self.spine.vault.scroll.leaf.trace.append({'kind': 'batch_leftover', 'n': len(extra)})
        return out

    def late_ask(self) -> str:
        """Before a checkout, ask the next planned question with a message the order itself does not need.

        The interview leaves out every message the request keeps for a process step, whether the step comes with the
        order or with an ending without a purchase. When God is about to buy, the messages kept for the ending are
        free. Race 13 allowed two messages: the base asked one question in 28 of 180 episodes and left a hidden need
        undisclosed in 12 of them (no episode with an undisclosed need passed). Returns the summary for the loop model
        when the answer adds a requirement, else ''."""
        if self.spine.vault.folio.voice.tape.late_asked or not self.spine.vault.folio.order.sheet.roles.get('ask') or self.spine.vault.folio.order.sheet.msg_used >= self.spine.vault.folio.order.sheet.msg_limit:
            return ''
        if any((p['when'] == 'order' and p['messages'] > 0 for p in self.spine.vault.folio.order.sheet.process)):
            return ''
        if self.spine.route.steps_open() < Charter.LATE_STEPS or self.spine.route.elapsed() > Charter.LATE_BY or not self.spine.route.model_order():
            return ''
        self.spine.vault.folio.voice.tape.late_asked = True
        if not self.spine.vault.folio.voice.tape.lanes:
            self.spine.vault.folio.voice.tape.lanes = self.spine.route.build_lanes(self.spine.route.list_limit() if self.spine.vault.stacks.chains.web.listing else 1)
            self.spine.vault.folio.voice.tape.lane_found = {n: 0 for n in self.spine.vault.folio.voice.tape.lanes}
            self.spine.vault.folio.voice.tape.lane_spent = {n: 0 for n in self.spine.vault.folio.voice.tape.lanes}
            self.spine.vault.folio.voice.tape.lane_used = {n: set() for n in self.spine.vault.folio.voice.tape.lanes}
        told: list[str] = []
        budget = self.spine.vault.folio.order.sheet.budget
        batch = self.spine.route.upcoming_asks(told, Charter.LATE_STEPS - 2)
        if not batch:
            self.spine.vault.scroll.leaf.trace.append({'kind': 'late_ask', 'skipped': 'no property left to ask or no message sent'})
            return ''
        if self.spine.vault.folio.order.sheet.budget != budget:
            told.append(f'budget now {self.spine.vault.folio.order.sheet.budget}')
        number, question, answer = batch[0]
        self.spine.vault.scroll.leaf.trace.append({'kind': 'late_ask', 'q': question, 'a': answer[:200], 'told': told})
        if not told:
            return ''
        self.spine.vault.stacks.pages.book.version += 1
        summary = f'God asked the shopper one more question before checking out (one step):\nQ: {question}\nA: {answer}\nNew firm requirements from this answer: ' + '; '.join(told) + '. The cart was not checked out: check each line against them (search again with the new values where needed), fix the cart, then check out.'
        pointed = [need for _, need in self.spine.vault.stacks.chains.web.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value.'
        searched = self.spine.route.search_values([number])
        return summary + ('\n' + searched if searched else '')

    def near_labels(self, number: int, label: str) -> list[str]:
        """Up to two keys of the item a reader may take for `label` as well: keys sharing a word with it, or values written under both on the
        listings read. A word several of the item's keys carry says nothing, and neither does a value of one letter or digit."""
        mine = GlyphKit.speak(label)
        seen: dict[str, set[str]] = {}
        for ident in self.spine.route.ids_of_item(number):
            for key, value in (FormKit.fact_map(self.spine.vault.stacks.pages.book.full.get(ident) or {}) or {}).items():
                seen.setdefault(GlyphKit.speak(key), set()).update((w for w in BlockKit.rootset(str(value)) if len(w) > 1))
        counts: dict[str, int] = {}
        for name in seen:
            for word in set(name.split()):
                counts[word] = counts.get(word, 0) + 1
        plain = {w for w in mine.split() if counts.get(w, 0) <= 2}
        scored = []
        for order, (name, values) in enumerate(seen.items()):
            if name != mine and name:
                shared = plain & set(name.split())
                common = seen.get(mine, set()) & values
                if shared or common:
                    scored.append((-(3 * len(shared) + min(len(common), 3)), order, name))
        return [name for _, _, name in sorted(scored)[:Charter.LOOK_ALIKES]]

    def reply_read(self, answer: str) -> str:
        """reply_kind() by its markers, and for an unknown phrasing (each race's shopper has its own idioms) one small model
        reading, cached per reply text; '' when neither says."""
        text = ' '.join(str(answer or '').split())
        kind = self.spine.route.reply_tag(text)
        if kind or len(text.split()) < 3 or text.startswith('{'):
            return kind
        cache = self.__dict__.setdefault('sense_cache', {})
        if text in cache:
            return cache[text]
        got = self.spine.route.ask_object(Charter.SENSE + '\n\nREPLY:\n' + text[:500], [Charter.LIGHT], 400) or {} if self.spine.route.model_window() > Charter.CALL_FLOOR + 5 and self.spine.vault.clockwork.pace.cost < Charter.SPEND_HARD else {}
        kind = str(got.get('kind') or '').strip().lower()
        kind = kind if kind in ('puzzled', 'free') else ''
        cache[text] = kind
        self.spine.vault.scroll.leaf.trace.append({'kind': 'reply_sense', 'a': text[:80], 'sense': kind})
        return kind

    def label_owned(self, item: SlipLine, label: str) -> bool:
        return bool(label) and item in self.spine.vault.folio.order.sheet.items and (self.spine.route.shelf_owner(label) == self.spine.vault.folio.order.sheet.items.index(item) + 1 or GlyphKit.speak(label) == GlyphKit.speak(item.shelf or ''))

    def apply_reply(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.spine.vault.folio.order.sheet.items) > 1
        for entry in FormKit.as_items(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, FormKit.as_count(entry.get('item')) or 1), len(self.spine.vault.folio.order.sheet.items))
            need = str(entry['need']).strip()
            if need not in self.spine.vault.folio.order.sheet.items[number - 1].told:
                self.spine.vault.folio.order.sheet.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.spine.vault.stacks.chains.web.refs:
                self.spine.vault.stacks.chains.web.refs.append((number, need))
        for entry in FormKit.as_items(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.spine.vault.folio.order.sheet.items[min(max(1, FormKit.as_count(entry.get('item')) or 1), len(self.spine.vault.folio.order.sheet.items)) - 1]
                gone = GlyphKit.speak(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and GlyphKit.speak(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in FormKit.as_items(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, FormKit.as_count(entry.get('item')) or 1), len(self.spine.vault.folio.order.sheet.items))
                self.spine.vault.folio.order.sheet.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.spine.vault.folio.order.sheet.budget = heard['budget']

class SaltBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def basket_ids(self) -> list[str]:
        if self.spine.vault.folio.basket.bin.cart_view is not None and self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at:
            text = FormKit.pack(self.spine.vault.folio.basket.bin.cart_view)
            return [i for i in self.spine.vault.stacks.pages.book.rows if i in text]
        return list(self.spine.vault.folio.basket.bin.cart_lines)

    def compat_miss(self) -> list[str]:
        if not self.spine.vault.stacks.chains.web.bound:
            return []
        index = self.spine.route.compat_index()
        cart = self.spine.route.basket_ids()
        faults = []
        for part, main in self.spine.vault.stacks.chains.web.bound:
            devices = [i for i in cart if i in self.spine.route.ids_of_item(main)]
            parts = [i for i in cart if i in self.spine.route.ids_of_item(part)]
            if len(devices) != 1 or not parts:
                continue
            linked = self.spine.route.series_miss(part, main, devices[0], parts, index)
            if linked:
                faults += linked
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            barred = lambda i: set(index['marks'].get(i, {}).get('bars') or [])
            covers = lambda i: any((GlyphKit.covers(p, c) for p in self.spine.route.codes_of(i, index) for c in codes)) and not any((GlyphKit.covers(p, c) for p in barred(i) for c in codes))
            for line in parts:
                fits = self.spine.route.codes_of(line, index)
                if fits and (not covers(line)):
                    good = [i for i in self.spine.route.ids_of_item(part) if covers(i) and (not self.spine.route.is_gone(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.spine.vault.folio.order.sheet.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.spine.route.compat_plan(short=True)))
        return faults

    def richer_shelf(self, number: int, label: str) -> str:
        cat_key = self.spine.route.shelf_field()
        mine = BlockKit.rootset(self.spine.vault.folio.order.sheet.items[number - 1].name)
        if len(mine) < 2 or mine <= BlockKit.rootset(label) or not cat_key or (self.spine.vault.folio.order.sheet.items[number - 1].shelf and BlockKit.rootset(label) == BlockKit.rootset(self.spine.vault.folio.order.sheet.items[number - 1].shelf)):
            return ''
        return next((seen for seen in sorted({str(r.get(cat_key) or '') for r in self.spine.vault.stacks.pages.book.rows.values()}) if seen and mine <= BlockKit.rootset(seen)), '')

    def short_picks(self) -> dict[str, list[str]]:
        """Candidates for an ending with too few steps left for the leads flow (block, check, block, complete): the review's
        leads, else the code's, else the last swap's fits, each under its item number; {} when none. Practice run
        2026-11-09: three declines with 2-5 steps left came after a sell-out while the audit or the swap had named a fit."""
        if self.spine.vault.folio.basket.bin.leads and self.spine.vault.folio.basket.bin.lead_items:
            return {k: [i for i in v if i in self.spine.vault.stacks.pages.book.rows and not self.spine.route.is_gone(i)] for k, v in self.spine.vault.folio.basket.bin.lead_items.items() if any((i in self.spine.vault.stacks.pages.book.rows and not self.spine.route.is_gone(i) for i in v))}
        found = self.spine.route.code_picks()
        if found:
            return found
        cat_key = self.spine.route.shelf_field()
        out: dict[str, list[str]] = {}
        for ident in self.spine.vault.folio.basket.bin.swap_fits:
            if ident not in self.spine.vault.stacks.pages.book.rows or self.spine.route.is_gone(ident) or ident in self.spine.vault.folio.basket.bin.gone or ident in self.spine.route.basket_ids():
                continue
            number = self.spine.route.shelf_owner(str((self.spine.vault.stacks.pages.book.rows.get(ident) or {}).get(cat_key) or '')) if cat_key else 0
            number = number or (1 if len(self.spine.vault.folio.order.sheet.items) == 1 else 0)
            if number:
                out.setdefault(str(number), []).append(ident)
        return out

    def opened_shop_ids(self, about: str) -> list[str]:
        not_names = self.spine.route.skip_maker()
        low = about.lower()
        out = []
        for ident in self.spine.vault.stacks.pages.book.opened:
            row = self.spine.vault.stacks.pages.book.rows.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in Charter.NO_NAME]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(ident)
        return out

    def audit_empty(self) -> str:
        if self.spine.vault.folio.basket.bin.audited or not self.spine.route.model_order():
            return ''
        self.spine.vault.folio.basket.bin.audited = True
        got = self.spine.route.audit_body()
        named = self.spine.route.fits_of_audit(got)
        if not named:
            rows = [str(i) for i in FormKit.as_items((got or {}).get('open')) if str(i) in self.spine.vault.stacks.pages.book.rows and str(i) not in self.spine.vault.stacks.pages.book.opened and (not self.spine.route.is_gone(str(i)))]
            if rows and self.spine.route.open_row_ids(rows):
                got = self.spine.route.audit_body()
                named = self.spine.route.fits_of_audit(got)
        if not named:
            return ''
        self.spine.vault.folio.basket.bin.lead_items = named
        self.spine.vault.folio.basket.bin.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.spine.vault.counter.front.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in FormKit.kind_list(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((ident in text for ident in self.spine.vault.stacks.pages.book.rows)) or self.spine.vault.folio.order.sheet.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def foreign_need(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.spine.vault.stacks.chains.web.refs] or need in [str(x) for x in FormKit.as_items(self.spine.vault.folio.order.sheet.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        if any(({w for w in GlyphKit.stemset(other.name) if len(w) >= 4} & GlyphKit.stemset(value) for other in self.spine.vault.folio.order.sheet.items if other is not item)):
            return False
        cat_key = self.spine.route.shelf_field()
        keys: set[str] = set()
        read = 0
        for record in self.spine.vault.stacks.pages.book.full.values():
            if cat_key and (not self.spine.route.label_owned(item, str(record.get(cat_key) or ''))):
                continue
            block = FormKit.fact_map(record) or {}
            if block:
                read += 1
                keys |= {GlyphKit.speak(k) for k in block}
        return read >= 3 and (not any((GlyphKit.stemset(k) & GlyphKit.stemset(prop) for k in keys)))

    def take_obs(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.spine.route.find_ident(node, in_list, depth)
                if ident:
                    if ident not in self.spine.vault.stacks.pages.book.rows:
                        self.spine.vault.stacks.pages.book.seen_order.append(ident)
                        self.spine.vault.stacks.pages.book.rows[ident] = {}
                    old = self.spine.vault.stacks.pages.book.rows[ident]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.spine.vault.folio.basket.bin.unavailable.add(ident)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.spine.vault.folio.basket.bin.unavailable.discard(ident)
                    self.spine.vault.stacks.pages.book.rows[ident].update(node)
                    if FormKit.fact_map(node):
                        self.spine.vault.stacks.pages.book.full[ident] = dict(self.spine.vault.stacks.pages.book.rows[ident])
                    elif ident in self.spine.vault.stacks.pages.book.full:
                        self.spine.vault.stacks.pages.book.full[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.spine.route.learn_row_keys(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def price_num(self, ident: str) -> float | None:
        value = (self.spine.vault.stacks.pages.book.rows.get(ident) or {}).get(self.spine.vault.stacks.pages.book.price_key) if self.spine.vault.stacks.pages.book.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str):
            figure = re.fullmatch('\\s*[^\\d\\s]{0,3}\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s*[A-Za-z]{0,6}\\.?\\s*', value)
            value = float(figure.group(1).replace(',', '')) if figure else None
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def page_names(self) -> set[str]:
        page = self.spine.vault.folio.order.sheet.roles.get('offer') or self.spine.vault.folio.order.sheet.roles.get('open')
        return {t for t in (page, self.spine.vault.folio.order.sheet.roles.get('compare')) if t}

    def record_blob(self, ident: str) -> str:
        record = self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident)
        return FormKit.pack(FormKit.shorten(record, 400))[:3000] if record else '(no record seen)'

    def model_order(self, limit: float=Charter.SPEND_SOFT) -> list[str]:
        if self.spine.vault.clockwork.pace.cost <= limit and self.spine.vault.clockwork.pace.cost + Charter.HEAVY_ROOM <= Charter.SPEND_HARD:
            return [Charter.HEAVY, Charter.LIGHT]
        return [Charter.LIGHT] if self.spine.vault.clockwork.pace.cost <= Charter.SPEND_HARD else []

    def next_page_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.spine.vault.stacks.probe.log.paged_obs = []
        pick = self.spine.vault.folio.order.sheet.brief.get('pick')
        plan = self.spine.route.page_span(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.spine.route.price_order(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.spine.vault.stacks.pages.book.pages_read + 2 <= Charter.AUTO_PAGES and (self.spine.route.steps_open() - self.spine.route.reserve_steps() - 2 >= 12):
            pages = 2
        if pages > 2 or self.spine.vault.stacks.pages.book.pages_read + pages > Charter.AUTO_PAGES or self.spine.route.steps_open() - self.spine.route.reserve_steps() - pages < 12:
            return f"\n[God] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={FormKit.pack(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = FormKit.fit_value({**args, param: value}, self.spine.vault.counter.front.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.spine.route.call_store(name, call)
            if reply is None or self.spine.vault.clockwork.pace.finished:
                break
            self.spine.vault.stacks.pages.book.pages_read += 1
            seen = reply.get('observation')
            self.spine.route.render_obs(name, seen)
            self.spine.vault.stacks.probe.log.paged_obs.append(seen)
            shown_text = self.spine.route.show_obs(seen)
            self.spine.vault.stacks.pages.book.read_cache[name + FormKit.pack(call)] = (self.spine.vault.stacks.pages.book.version, self.spine.vault.clockwork.pace.step, shown_text)
            texts.append(f'{param}={FormKit.pack(value)}: {shown_text}')
            after = self.spine.route.page_span(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.spine.vault.scroll.leaf.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[God] {count} listings matched and God has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[God read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_form(self) -> tuple[str, dict[str, Any], str, str] | None:
        """The listing tool that takes a shelf and a maximum price, with the mode it needs: the mode an earlier listing by
        shelf used, else the one option naming a listing. None when no tool or mode is clear."""
        for tool, schema in self.spine.vault.counter.front.schemas.items():
            props = (schema or {}).get('properties') or {}
            caps = self.spine.route.cap_names(tool)
            shelves = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and 'enum' not in v and re.search(Charter.SHELF_ARG, GlyphKit.snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            learned = [(dict(mode), held) for (key, mode), held in self.spine.route.cap_mode_map(tool).items() if key == caps[0]]
            options_args = [k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2]
            single = len(options_args) == 1 and options_args[0] in ((schema or {}).get('required') or [])
            base: dict[str, Any] = {}
            for key, spec in props.items():
                options = spec.get('enum') if isinstance(spec, dict) else None
                if not isinstance(options, list) or len(options) < 2:
                    continue
                pick = next((mode[key] for mode, held in learned if held and key in mode), None)
                if pick is None:
                    pick = next((a[key] for a in reversed(self.spine.vault.stacks.probe.log.searched) if shelves[0] in a and key in a and not any(k != shelves[0] and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and isinstance(v, str) and v.strip() and re.search(Charter.QUERY_WORDS, GlyphKit.snake(k), re.I) for k, v in a.items())), None)
                    named = [o for o in options if isinstance(o, str) and re.search(Charter.LIST_MODE, o, re.I)]
                    worded = {a[key] for a in self.spine.vault.stacks.probe.log.searched if key in a and any(isinstance(v, str) and v.strip() and re.search(Charter.QUERY_WORDS, GlyphKit.snake(k), re.I) for k, v in a.items())}
                    unworded = [o for o in options if o not in worded]
                    pick = pick if pick is not None else named[0] if len(named) == 1 else unworded[0] if single and worded and len(unworded) == 1 else None
                spoiled = {mode[key] for mode, held in learned if not held and key in mode} if single else set()
                if spoiled and (pick is None or pick in spoiled):
                    rest = [o for o in options if o not in spoiled]
                    pick = rest[0] if len(rest) == 1 else None
                if pick is None:
                    return None
                base[key] = pick
            return (tool, base, shelves[0], caps[0])
        return None

    def item_row_count(self, number: int) -> int:
        cat_key = self.spine.route.shelf_field()
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        return sum((1 for row in self.spine.vault.stacks.pages.book.rows.values() if not cat_key or self.spine.route.label_owned(item, str(row.get(cat_key) or ''))))

    @cushion('')
    def flip_last_page(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A new search that opened on a later page shows few rows and no next page, which reads as all there is: race 15's models
        asked the browse tool for its highest page in 1716 of 2845 calls, seeing 7 of 342 rows and "next: none". When such a call
        reports more rows than it shows, the same call at the lowest value of its numeric argument tells: rows with other ids,
        at least as many, mean that argument is the page."""
        if name in self.spine.vault.stacks.pages.book.page_args or not isinstance(observation, dict):
            return ''
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}
        numeric = [k for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool) and k in props and v > (props[k].get('minimum') if isinstance(props[k].get('minimum'), int) else 1)]
        if len(numeric) != 1 or self.spine.route.steps_open() <= self.spine.route.reserve_steps() + 4:
            return ''
        shape = self.spine.route.learn_shape(name, observation)
        if not shape or not shape.get('count') or shape.get('next') is None or observation.get(shape['next']) is not None:
            return ''
        rows = observation.get(shape['rows'])
        count = observation.get(shape['count'])
        if not isinstance(rows, list) or not isinstance(count, int) or count <= len(rows):
            return ''
        key = numeric[0]
        low = props[key].get('minimum') if isinstance(props[key].get('minimum'), int) else 1
        faults: list[str] = []
        call = FormKit.fit_value({**args, key: low}, self.spine.vault.counter.front.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.spine.route.call_store(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.spine.route.render_obs(name, seen)
        new = seen.get(shape['rows']) if isinstance(seen, dict) else None
        if not isinstance(new, list) or len(new) < len(rows) or {i for i, _ in self.spine.route.rows_inside(seen)} & {i for i, _ in self.spine.route.rows_inside(observation)}:
            return ''
        self.spine.vault.stacks.pages.book.page_args[name] = key
        self.spine.vault.scroll.leaf.trace.append({'kind': 'page_arg', 'tool': name, 'arg': key})
        return f'\n[God] {key} is the page number of this search: the call above showed its last page. Page {low} of the same search:\n' + self.spine.route.show_obs(seen)

    def search_values(self, numbers: list[int]) -> str:
        plan = self.spine.route.value_form()
        if not plan or self.spine.route.steps_open() < self.spine.route.reserve_steps() + Charter.VALUE_ROOM:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = min(8, self.spine.route.steps_open() - self.spine.route.reserve_steps() - Charter.VALUE_KEEP)
        for number in numbers:
            pairs = self.spine.route.value_bits(number)
            item = self.spine.vault.folio.order.sheet.items[number - 1]
            values = [v for v, _ in pairs]
            fresh = [v for v, told in pairs if told]
            if not fresh or budget <= 2:
                continue
            lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(Charter.AMOUNT, v, re.I)]
            values = lead + [v for v in values if v not in lead]
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [[values[0], item.name], values] if lead else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = FormKit.fit_value({**base, field: ' '.join(words)}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults or self.spine.vault.clockwork.pace.finished:
                    break
                reply = self.spine.route.call_store(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.spine.route.render_obs(tool, observation)
                cat_key = self.spine.route.shelf_field()
                found = [i for i, _ in self.spine.route.rows_inside(observation)]
                cap = self.spine.vault.folio.order.sheet.budget if isinstance(self.spine.vault.folio.order.sheet.budget, (int, float)) else None
                prices = [self.spine.route.price_num(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.spine.route.in_stock(i) and (not cat_key or self.spine.route.label_owned(item, str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.spine.route.price_num(i) or 0) <= cap)]
                home = self.spine.route.own_shelf(number)
                if cat_key and home:
                    rows.sort(key=lambda i: GlyphKit.speak(str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '')) != home)
                seen.append(f'{FormKit.pack(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.spine.route.open_for_specs([i for i in rows if i not in self.spine.vault.stacks.pages.book.full][:min(3, budget)], self.spine.route.reserve_steps() + Charter.VALUE_KEEP)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.spine.vault.scroll.leaf.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[God searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def call_store(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        held = self.spine.vault.stacks.probe.log.prefetched.pop(name + FormKit.pack(args), None)
        if held is not None:
            return held
        payload = dict(self.spine.vault.counter.front.binding)
        payload['step'] = self.spine.vault.clockwork.pace.step + 1
        payload['action'] = {'name': name, 'args': args}
        rest = self.spine.vault.clockwork.pace.env_at + self.spine.vault.clockwork.pace.env_gap - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        self.spine.vault.clockwork.pace.env_at = time.monotonic()
        for _ in range(4):
            began = time.monotonic()
            result = Pipes.env.post_verbose('/environment/call', payload)
            self.spine.vault.clockwork.pace.env_spent += time.monotonic() - began
            self.spine.vault.clockwork.pace.env_count += 1
            if isinstance(result.data, dict):
                return self.spine.route.book_result(name, args, result.data)
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.spine.vault.scroll.leaf.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.spine.vault.clockwork.pace.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.spine.vault.clockwork.pace.step = int(found.group(1)) - 1
                    payload['step'] = self.spine.vault.clockwork.pace.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.spine.vault.clockwork.pace.step += 1
                    payload['step'] = self.spine.vault.clockwork.pace.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.spine.vault.clockwork.pace.broke = True
                return None
        return None

    def end_episode(self, again: bool=False) -> str:
        if self.spine.vault.clockwork.pace.finished or self.spine.vault.clockwork.pace.broke or self.spine.route.steps_open() <= 0:
            return ''
        buy, refuse = (self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse'))
        fresh = self.spine.vault.folio.basket.bin.cart_view is not None and self.spine.vault.folio.basket.bin.cart_seen_at >= self.spine.vault.folio.basket.bin.cart_at
        holding = bool(self.spine.route.basket_ids()) if fresh else bool(self.spine.vault.folio.basket.bin.cart_lines)
        whole = True
        if holding and len(self.spine.vault.folio.order.sheet.items) >= 2:
            whole = self.spine.route.can_fill_short() or not self.spine.route.decline_ready()
            if whole and len(self.spine.route.basket_ids()) >= 2:
                self.spine.route.open_for_specs(self.spine.route.basket_ids(), self.spine.route.reserve_steps())
                if self.spine.route.steps_open() >= 3 and self.spine.route.model_window() > 5:
                    self.spine.route.open_for_links(self.spine.route.basket_ids(), self.spine.route.reserve_steps())
            if not whole:
                self.spine.vault.scroll.leaf.trace.append({'kind': 'short_cart_refused', 'lines': self.spine.route.basket_ids()})
        if self.spine.vault.clockwork.pace.finished or self.spine.route.steps_open() <= 0:
            return ''
        if whole and buy and (holding or (self.spine.vault.folio.basket.bin.cart_touched and (not fresh) and (self.spine.route.steps_open() >= 2))):
            faults: list[str] = []
            args = FormKit.fit_value(dict(self.spine.vault.folio.order.sheet.brief.get('buy_args') or {}), self.spine.vault.counter.front.schemas.get(buy), 'args', faults)
            if not faults:
                if self.spine.route.steps_open() >= 2:
                    self.spine.route.pass_stage('order')
                if self.spine.vault.clockwork.pace.finished or self.spine.route.steps_open() <= 0:
                    return ''
                reply = self.spine.route.call_store(buy, args)
                self.spine.vault.scroll.leaf.trace.append({'kind': 'forced_buy', 'reply': FormKit.shorten(reply, 200)})
                if self.spine.vault.clockwork.pace.finished or self.spine.route.steps_open() <= 0:
                    return ''
                if reply is not None and again:
                    self.spine.vault.folio.basket.bin.buys += 1
                    self.spine.vault.folio.basket.bin.refused_at = self.spine.vault.clockwork.pace.step
                    self.spine.route.render_obs(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.spine.route.show_obs(reply.get('observation'))
        if refuse:
            faults = []
            args = FormKit.fit_value(self.spine.route.quiet_args(refuse, dict(self.spine.vault.folio.order.sheet.brief.get('refuse_args') or {})), self.spine.vault.counter.front.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.spine.route.steps_open() >= 2:
                    self.spine.route.pass_stage('end')
                if self.spine.vault.clockwork.pace.finished or self.spine.route.steps_open() <= 0:
                    return ''
                reply = self.spine.route.call_store(refuse, args)
                self.spine.vault.scroll.leaf.trace.append({'kind': 'forced_refuse', 'reply': FormKit.shorten(reply, 200)})
                if reply is not None and (not self.spine.vault.clockwork.pace.finished):
                    self.spine.route.render_obs(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.spine.route.show_obs(reply.get('observation'))
        return ''

    def spread_one(self) -> None:
        tool = self.spine.route.spec_tool()
        target = self.spine.route.one_id_arg(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.spine.route.live_needs() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = Charter.WIDEN_EACH if len(self.spine.vault.folio.order.sheet.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.spine.route.shelf_groups():
            opened = sum((1 for i in ids if i in self.spine.vault.stacks.pages.book.full))
            pool = [i for i in ids if i not in self.spine.vault.stacks.pages.book.full and self.spine.route.in_stock(i)]
            if self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', FormKit.pack(self.spine.vault.stacks.pages.book.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.spine.route.open_for_specs(first[:cap][:8], 22)
        self.spine.route.open_for_specs((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def list_limit(self) -> int:
        return Charter.LIST_SIZE if len(self.spine.vault.folio.order.sheet.items) == 1 else Charter.LIST_MIXED

    def land_answer(self, number: int, labels: list[str], shown: bool, new: list[str], crowded: bool, answer: str='', free_hit: int=-1) -> None:
        """Mark what an answer showed about the labels its question named.

        One label: it is asked; a reply asking for a narrower question gets the base's follow-up (one product of an
        item name joining several, the property's head word, or example values). A list: the shopper answers about
        the first property of a message it tracks, and its reply to a message naming none of its properties is the
        same as to a property it does not mind. So a disclosed need settles its label and every label before it, and
        the labels after it lead the item's next list; a reply that discloses nothing settles only the first label
        (a need there would have been answered) and the others come next in reverse order, so whatever the shopper
        does not mind no longer stands in front of them; after a second such reply in a row the labels not yet named lead, since the
        order would otherwise return to the first list's. A reply asking for one question at a time ends lists for
        the episode and the list's first label is asked alone."""
        used = self.spine.vault.folio.voice.tape.lane_used.setdefault(number, set())
        if len(labels) == 1:
            used.add(GlyphKit.speak(labels[0]))
            if crowded and self.spine.vault.stacks.chains.web.listing:
                self.spine.vault.stacks.chains.web.listing = False
                self.spine.vault.scroll.leaf.trace.append({'kind': 'lists_off', 'item': number})
            if new or not crowded:
                return
            if number not in self.spine.vault.folio.voice.tape.one_product and self.spine.route.name_bits(number):
                self.spine.vault.folio.voice.tape.one_product.add(number)
                self.spine.vault.scroll.leaf.trace.append({'kind': 'one_product', 'item': number, 'said': self.spine.route.item_words(number)})
                self.spine.vault.folio.voice.tape.follow_up.insert(0, (number, labels, shown))
                return
            if not self.spine.vault.folio.voice.tape.apart:
                alike = self.spine.route.near_labels(number, labels[0])
                if alike:
                    self.spine.vault.folio.voice.tape.apart.setdefault(labels[0], {})[number] = alike
                    self.spine.vault.scroll.leaf.trace.append({'kind': 'apart', 'item': number, 'label': labels[0], 'from': alike})
                    self.spine.vault.folio.voice.tape.follow_up.insert(0, (number, labels, False))
            return
        if crowded and not new:
            self.spine.vault.stacks.chains.web.listing = False
            self.spine.vault.scroll.leaf.trace.append({'kind': 'lists_off', 'item': number})
            self.spine.vault.folio.voice.tape.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.spine.route.answered_at(labels, new) if new else -1
        if hit < 0 and free_hit >= 0:
            hit = free_hit
        rest = [label for label in self.spine.vault.folio.voice.tape.lanes.get(number) or [] if label not in labels]
        if hit < 0 and self.spine.route.reply_read(answer) == 'puzzled':
            self.spine.vault.folio.voice.tape.lane_misses[number] = self.spine.vault.folio.voice.tape.lane_misses.get(number, 0) + 1
            self.spine.vault.folio.voice.tape.lanes[number] = rest + labels
            self.spine.vault.scroll.leaf.trace.append({'kind': 'list_settled', 'item': number, 'puzzled': labels, 'next': rest[:4]})
            return
        ahead = labels[hit + 1:] if hit >= 0 else labels[:0:-1]
        used.update((GlyphKit.speak(label) for label in labels[:max(hit, 0) + 1]))
        self.spine.vault.folio.voice.tape.lane_misses[number] = 0 if hit >= 0 else self.spine.vault.folio.voice.tape.lane_misses.get(number, 0) + 1
        if self.spine.vault.folio.voice.tape.lane_misses[number] and self.spine.vault.folio.voice.tape.lane_misses[number] % 2 == 0:
            ahead = [label for _, _, label in sorted((abs(i - (len(ahead) - 1) / 2), i, label) for i, label in enumerate(ahead))]
            self.spine.vault.folio.voice.tape.lanes[number] = rest + ahead
        else:
            self.spine.vault.folio.voice.tape.lanes[number] = ahead + rest
        self.spine.vault.scroll.leaf.trace.append({'kind': 'list_settled', 'item': number, 'done': labels[:hit + 1], 'next': ahead[:4]})

    def opened_for(self, number: int) -> int:
        cat_key = self.spine.route.shelf_field()
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        return sum((1 for record in self.spine.vault.stacks.pages.book.full.values() if cat_key and self.spine.route.label_owned(item, str(record.get(cat_key) or '')) and FormKit.fact_map(record)))

    def read_replies(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(Charter.HEAVY, Charter.LIGHT)) -> dict[str, Any]:
        if self.spine.route.reply_blank(sent):
            self.spine.vault.scroll.leaf.trace.append({'kind': 'quiet_reply', 'a': sent[0][2][:80]})
            return {}
        self.spine.vault.folio.voice.tape.hear_reads += 1
        prompt = Charter.HEAR + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.spine.vault.folio.order.sheet.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f"{i}. ({f'about item {n}' if n else 'about several items'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1)))
        return self.spine.route.ask_object(prompt, list(models), 2500) or {}

class WoolBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def compat_index(self) -> dict[str, Any]:
        state = (len(self.spine.vault.stacks.pages.book.seen_order), len(self.spine.vault.stacks.pages.book.full), sum((len(v) for v in self.spine.vault.stacks.pages.book.views.values())))
        if self.spine.vault.stacks.chains.web.fit_cached and self.spine.vault.stacks.chains.web.fit_cached[0] == state:
            return self.spine.vault.stacks.chains.web.fit_cached[1]
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.spine.vault.stacks.pages.book.seen_order:
            record = self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[ident] = GlyphKit.marks(text)
            for part, codes in marks[ident]['also']:
                notes.setdefault(part, set()).update(codes)
        self.spine.vault.stacks.chains.web.fit_cached = (state, {'marks': marks, 'notes': notes})
        return self.spine.vault.stacks.chains.web.fit_cached[1]

    def compat_plan(self, short: bool=False) -> str:
        if not self.spine.vault.stacks.chains.web.bound:
            return ''
        index = self.spine.route.compat_index()
        lines = []
        for main in sorted({m for _, m in self.spine.vault.stacks.chains.web.bound}):
            parts = [p for p, m in self.spine.vault.stacks.chains.web.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.spine.route.ids_of_item(p):
                    if self.spine.route.is_gone(i):
                        continue
                    for code in self.spine.route.codes_of(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            for p in parts:
                system = self.spine.route.series_web(p, main, index)
                for i in self.spine.route.ids_of_item(p) if system and system[0] == p else []:
                    for code in set((index['marks'].get(i) or {}).get('fam') or []) & system[2] if not self.spine.route.is_gone(i) else ():
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
            devices = {code: [i for i in self.spine.route.ids_of_item(main) if code in ((index['marks'].get(i) or {}).get('named') or []) or any((GlyphKit.covers(code, c) for c in index['marks'].get(i, {}).get('codes') or []))][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.spine.vault.folio.order.sheet.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.spine.vault.folio.order.sheet.items[main - 1].name} candidates' parts name -> {text}")
            sets = self.spine.route.cheap_pairs(main, parts, index) if self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest' else ''
            if sets:
                lines.append(sets)
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as, or when the device's text names the series of the part's code; try one search of a code or series with the item's words, and if it finds nothing list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def shelf_for_item(self, number: int, label: str) -> str:
        """The store label the request named for this item, when a listing seen carries it and `label` is another one."""
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        cat_key = self.spine.route.shelf_field()
        wanted = GlyphKit.speak(item.shelf or '')
        if not wanted or not cat_key or GlyphKit.speak(label) == wanted:
            return ''
        here = next((str(r.get(cat_key)) for r in self.spine.vault.stacks.pages.book.rows.values() if GlyphKit.speak(str(r.get(cat_key) or '')) == wanted), '')
        return here

    def basket_compat(self) -> str:
        """Once, before an ending without a purchase while the cart holds a line for every item: the independent check of
        that cart; the text that holds the ending when the check finds the cart right, else ''."""
        ids = [i for i in self.spine.route.basket_ids() if (self.spine.route.units_of(i) or 0) > 0]
        if self.spine.vault.stacks.probe.log.whole_held or not ids or len(ids) < len(self.spine.vault.folio.order.sheet.items) or self.spine.route.items_absent() or any((self.spine.route.is_gone(i) for i in ids)):
            return ''
        self.spine.vault.stacks.probe.log.whole_held = True
        self.spine.route.judge_basket()
        if not self.spine.vault.folio.basket.bin.verdicts.get(self.spine.route.judge_key(self.spine.route.basket_ids())):
            return ''
        self.spine.vault.scroll.leaf.trace.append({'kind': 'whole_cart_held', 'cart': ids})
        return "An independent check of the cart's records says every line meets its requirements, and ending without a purchase earns nothing when the cart fits: check out instead. End without a purchase only if a requirement clearly fails for every listing of some item; then say which one in your text and send the ending call again."

    def seek_link(self, about: str, words: str) -> list[str]:
        not_names = self.spine.route.skip_maker()
        known: list[str] = []
        named = {w for ident in self.spine.route.shop_ids(about, list(self.spine.vault.stacks.pages.book.rows)) + self.spine.route.opened_shop_ids(about) for k, v in (self.spine.vault.stacks.pages.book.rows.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and k not in not_names and v.strip().lower() not in Charter.NO_NAME for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for ident in self.spine.route.opened_shop_ids(about):
            stack, leaves = ([self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.spine.vault.scroll.leaf.trace.append({'kind': 'link_known', 'id': ident})
                return []
        plan = self.spine.route.value_form()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.spine.vault.stacks.chains.web.link_seen:
            known = self.spine.route.shop_ids(about, self.spine.vault.stacks.chains.web.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            for query in dict.fromkeys((words, ' '.join(words.split()[:2]))):
                faults: list[str] = []
                args = FormKit.fit_value({**base, field: query}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults:
                    break
                self.spine.vault.stacks.chains.web.link_searches += 1
                reply = self.spine.route.call_store(tool, args)
                if reply is None:
                    break
                observation = reply.get('observation')
                self.spine.route.render_obs(tool, observation)
                self.spine.vault.stacks.chains.web.link_seen[memo] = [i for i, _ in self.spine.route.rows_inside(observation)]
                known = self.spine.route.shop_ids(about, self.spine.vault.stacks.chains.web.link_seen[memo])
                self.spine.vault.scroll.leaf.trace.append({'kind': 'link_search', 'words': query[:80], 'rows': known[:4]})
                if self.spine.vault.stacks.chains.web.link_seen[memo]:
                    break
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', FormKit.pack(self.spine.vault.stacks.pages.book.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.spine.route.shop_ids(about, list(reversed(self.spine.vault.stacks.pages.book.seen_order)))))[:2]

    def audit_body(self) -> dict[str, Any] | None:
        records = []
        for ident in sorted(reversed(self.spine.vault.stacks.pages.book.seen_order), key=lambda i: i not in self.spine.vault.stacks.pages.book.full):
            if self.spine.route.is_gone(ident):
                continue
            record = self.spine.vault.stacks.pages.book.full.get(ident) if ident in self.spine.vault.stacks.pages.book.full else None
            records.append('OPENED ' + FormKit.pack(FormKit.shorten(record, 200))[:1200] if record else 'ROW ' + FormKit.pack(FormKit.shorten(self.spine.vault.stacks.pages.book.rows.get(ident), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = Charter.AUDIT + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.spine.route.must_sheet()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        judges = self.spine.route.model_order()
        got = self.spine.route.ask_object(prompt, [Charter.LIGHT] if judges else [], 4000)
        if Charter.HEAVY in judges and not self.spine.vault.clockwork.pace.no_credit and not self.spine.route.fits_of_audit(got):
            got = self.spine.route.ask_object(prompt, [Charter.HEAVY], 4000) or got
        self.spine.vault.scroll.leaf.trace.append({'kind': 'audit', 'got': got})
        return got

    def fix_id_args(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.spine.vault.folio.order.sheet.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.spine.vault.stacks.pages.book.id_key:
            return (args, '')
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in FormKit.kind_list(v) or 'array' in FormKit.kind_list(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.spine.vault.folio.order.sheet.roles.get('cart') and (name, free[0]) not in self.spine.vault.stacks.pages.book.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.spine.vault.stacks.pages.book.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or FormKit.id_shaped(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.spine.vault.stacks.pages.book.rows.items() if any((k != self.spine.vault.stacks.pages.book.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.spine.vault.stacks.pages.book.id_key} field)')
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

    def must_sheet(self) -> str:
        parts = []
        many = len(self.spine.vault.folio.order.sheet.items) > 1
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.spine.route.foreign_need(number, m)]
            needs = [m + (f' (the page it points at gives: {self.spine.vault.stacks.chains.web.ref_values[m]})' if m in self.spine.vault.stacks.chains.web.ref_values else ' (the listing it points at was not found in this store after a search for it: it decides nothing, judge the other requirements)' if self.spine.route.ref_spent(number, m) else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            parts.append(text)
        brief = self.spine.vault.folio.order.sheet.brief
        if self.spine.vault.folio.order.sheet.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.spine.vault.folio.order.sheet.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in FormKit.as_items(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in FormKit.as_items(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.spine.vault.folio.order.sheet.process:
            when = {'choose': 'before choosing', 'order': 'at the order (God does it when you check out)', 'end': 'at an ending without a purchase (God does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.spine.route.spec_tool()
        if facts and (self.spine.vault.stacks.pages.book.facts_from.get(facts) is True or self.spine.vault.folio.order.sheet.roles.get('offer')):
            bare = sorted((t for t in self.spine.route.page_names() | {self.spine.vault.folio.order.sheet.roles.get('offer') or ''} if t and self.spine.vault.stacks.pages.book.facts_from.get(t) is False))
            pages = sorted((t for t in self.spine.route.page_names() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def shelf_labels(self) -> set[str]:
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
        for schema in self.spine.vault.counter.front.schemas.values():
            walk(schema, 0)
        return found

    def in_stock(self, ident: str) -> bool:
        if self.spine.route.is_gone(ident):
            return False
        value = (self.spine.vault.stacks.pages.book.rows.get(ident) or {}).get(self.spine.vault.stacks.pages.book.stock_key) if self.spine.vault.stacks.pages.book.stock_key else None
        if isinstance(value, str) and re.search(Charter.STOCK_OUT, value.strip(), re.I):
            return False
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def spec_tool(self) -> str | None:
        known = next((t for t, ok in self.spine.vault.stacks.pages.book.facts_from.items() if ok and self.spine.route.one_id_arg(t)), None)
        if known:
            return known
        tool = self.spine.vault.folio.order.sheet.roles.get('open')
        return tool if self.spine.vault.stacks.pages.book.facts_from.get(tool or '') is not False and self.spine.route.one_id_arg(tool) else None

    def side_records(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.spine.vault.stacks.pages.book.seen_order))
        if self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.spine.vault.stacks.pages.book.full, self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.spine.route.is_gone(ident):
                continue
            record = self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident)
            lines.append(FormKit.pack(FormKit.shorten(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def pass_stage(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.spine.vault.folio.order.sheet.process if p['when'] == stage]
        if not steps or self.spine.vault.clockwork.pace.finished or (not self.spine.route.stage_pending(stage)):
            return (True, '')
        ending = {self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {FormKit.pack(t['function'].get('parameters') or {})}" for t in self.spine.vault.counter.front.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        for _ in range(Charter.GATE_ROUNDS):
            prompt = Charter.GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.spine.route.must_sheet() + '\n\nCART NOW: ' + self.spine.route.basket_text() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.spine.vault.folio.order.sheet.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.spine.vault.stacks.pages.book.notes[-4:]) if self.spine.vault.stacks.pages.book.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.spine.vault.folio.order.sheet.msg_limit - self.spine.vault.folio.order.sheet.msg_used)}   STEPS LEFT: {self.spine.route.steps_open()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.spine.route.ask_object(prompt, [Charter.HEAVY, Charter.LIGHT] if nudged else [Charter.LIGHT, Charter.HEAVY], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.spine.route.apply_reply({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.spine.vault.scroll.leaf.trace.append({'kind': 'gate', 'stage': stage, 'call': FormKit.shorten(call, 200), 'steps': FormKit.shorten(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.spine.vault.stacks.pages.book.version += 1
                return (False, ('Before going on, God did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in FormKit.as_items(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any((p['messages'] > 0 for p in steps))
            lines = range(1, len(self.spine.vault.folio.order.sheet.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.spine.vault.folio.order.sheet.done_steps) and self.spine.vault.folio.order.sheet.done_steps[i - 1] > self.spine.vault.folio.basket.bin.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((FormKit.as_count(n) or 0) in proof for n in FormKit.as_items(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.spine.vault.folio.order.sheet.roles.get('ask') and (self.spine.vault.folio.order.sheet.msg_used < self.spine.vault.folio.order.sheet.msg_limit) and (self.spine.route.steps_open() >= 2):
                nudged = True
                self.spine.vault.scroll.leaf.trace.append({'kind': 'gate_unproven', 'steps': FormKit.shorten(unproven, 120)})
                self.spine.vault.folio.order.sheet.done_log.append(f'(God: no line made since the cart last changed at step {self.spine.vault.folio.basket.bin.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.spine.vault.folio.order.sheet.msg_limit - self.spine.vault.folio.order.sheet.msg_used} message(s) and {self.spine.route.steps_open()} steps are left: give the call that does it)')
                self.spine.vault.folio.order.sheet.done_steps.append(-1)
                continue
            if not call or name not in self.spine.vault.counter.front.schemas or name in ending:
                if not call:
                    self.spine.vault.folio.order.sheet.gate_ok.add((stage, self.spine.route.basket_sig() + f'#{self.spine.vault.stacks.pages.book.version}'))
                break
            if self.spine.route.steps_open() < 2 or (name == self.spine.vault.folio.order.sheet.roles.get('ask') and self.spine.vault.folio.order.sheet.msg_used >= self.spine.vault.folio.order.sheet.msg_limit):
                break
            faults: list[str] = []
            args = FormKit.fit_value(call.get('args') if isinstance(call.get('args'), dict) else {}, self.spine.vault.counter.front.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.spine.vault.folio.order.sheet.roles.get('ask'):
                text = max(FormKit.strings_of(args), key=len, default='')
                if not text:
                    break
                batch = self.spine.route.dispatch_asks([(1, '', text)], 1)
                if not batch:
                    break
                self.spine.vault.folio.order.sheet.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.spine.route.call_store(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.spine.route.take_obs(seen)
            self.spine.vault.folio.order.sheet.done_log.append(f'step {self.spine.vault.clockwork.pace.step}: {name} {FormKit.pack(args)} -> {self.spine.route.show_obs(seen)[:300]} (cart then: {self.spine.route.basket_sig()})')
            self.spine.vault.folio.order.sheet.done_steps.append(self.spine.vault.clockwork.pace.step)
            said.append(f'{name} {FormKit.pack(args)} -> {self.spine.route.show_obs(seen)[:400]}')
        return (True, 'Before going on, God did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def cap_miss_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        limits = self.spine.route.cap_names(name)
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and (re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k) or k in limits[:1] and len(limits) == 1)]
        if not caps or not self.spine.vault.stacks.pages.book.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.spine.route.rows_inside(observation) if (self.spine.route.price_num(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={FormKit.pack(o[0])}' for k, o in others.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[God] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cap_names(self, tool: str) -> list[str]:
        """The numeric arguments of `tool` that may cap a listing's price: those a search has shown holding a limit, else those whose
        name says a maximum, else those whose name says a price or budget."""
        props = (self.spine.vault.counter.front.schemas.get(tool) or {}).get('properties') or {}
        numbers = [k for k, v in props.items() if isinstance(v, dict) and {'integer', 'number'} & set(FormKit.kind_list(v)) and not re.search(Charter.CAP_NOT, k, re.I) and not re.search(Charter.CAP_LOW, GlyphKit.snake(k), re.I)]
        held = {key for (key, _), ok in self.spine.route.cap_mode_map(tool).items() if ok}
        named = [k for k in numbers if re.search(Charter.CAP_ARG, GlyphKit.snake(k), re.I)]
        priced = [k for k in numbers if re.search(Charter.CAP_PRICED, GlyphKit.snake(k), re.I)]
        for group in ([k for k in numbers if k in held], [k for k in named if k in priced], priced, named):
            if group:
                return group
        return []

    @cushion('')
    def hunt_lower(self) -> str:
        """Search on, without the model, while the pick is the cheapest and the store shows only part of what it holds.

        Searches stop at a page cap while reporting the whole total (race 14: 96 rows of 199; race 15: 49 rows of up to 1000, no
        price order, no price filter), so the lowest price of a kind can sit where no first search reaches. Race 15's search reads
        titles, categories, brands, sellers and attribute VALUES and wants every word (22642 of 91849 matched words were found in
        an attribute value only), so all of an item's required values together list nearly its whole fitting set, and adding a
        further word splits a set too long for the cap. After a requirement was dropped the fitting set widens: that decided 7
        tasks of race 15, nearly every agent buying a dearer listing, and a rare passing run had kept trying new words (9 cents
        after 98 and 20). Each search is one step and no model call; cheaper_rows then opens the lowest rows."""
        mark = FormKit.pack([self.spine.route.live_needs(), self.spine.vault.folio.order.sheet.budget, sorted(self.spine.vault.folio.basket.bin.gone)])
        if self.spine.vault.folio.order.sheet.brief.get('pick') != 'cheapest' or self.spine.vault.stacks.probe.log.hunts >= Charter.HUNT_ROUNDS or self.spine.vault.stacks.probe.log.hunt_mark == mark:
            return ''
        plan = self.spine.route.value_form()
        if not plan or not self.spine.vault.stacks.pages.book.price_key or self.spine.route.elapsed() > Charter.HUNT_BY:
            return ''
        tool, base, field = plan
        room = min(Charter.HUNT_STEPS, self.spine.route.steps_open() - self.spine.route.reserve_steps() - Charter.HUNT_KEEP - 2 * len(self.spine.vault.folio.order.sheet.items))
        if room < 3:
            return ''
        self.spine.vault.stacks.probe.log.hunts += 1
        self.spine.vault.stacks.probe.log.hunt_mark = mark
        cart = self.spine.route.basket_ids()
        numbers = list(range(1, len(self.spine.vault.folio.order.sheet.items) + 1))
        share = max(3, room // max(1, len(numbers)))
        spent = 0
        fresh_total = 0
        for number in numbers:
            if spent >= room or self.spine.vault.clockwork.pace.finished:
                break
            used, gained = self.spine.route.hunt_one(number, tool, base, field, min(share, room - spent))
            spent += used
            fresh_total += gained
        self.spine.vault.scroll.leaf.trace.append({'kind': 'hunt', 'steps': spent, 'new_rows': fresh_total, 'cart': cart})
        if spent:
            self.spine.vault.stacks.probe.log.wider_held = True
        return f'God ran {spent} more searches for the lowest price, one step each, and they showed {fresh_total} more rows.' if spent else ''

    @cushion(False)
    def query_has_code(self, args: dict[str, Any]) -> bool:
        """Whether a search names a model or part code some listing's text gave as a code."""
        said = {c for v in args.values() if isinstance(v, str) for c in GlyphKit.harvest(v)}
        if not said:
            return False
        known: set[str] = set()
        for mark in self.spine.route.compat_index()['marks'].values():
            known.update(mark.get('codes') or [])
            known.update(mark.get('works') or [])
        return bool(said & known)

    def own_shelf(self, number: int) -> str:
        """The category label (in _words form) of the item's own listings: the request's shelf name when a listing
        seen carries it, else the label most of the item's opened listings (or, with none opened, its search rows)
        carry; '' when none is known."""
        cat_key = self.spine.route.shelf_field()
        if not cat_key:
            return ''
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        for pool in (self.spine.vault.stacks.pages.book.full, self.spine.vault.stacks.pages.book.rows):
            counts: dict[str, int] = {}
            for record in pool.values():
                label = str(record.get(cat_key) or '')
                if label and self.spine.route.label_owned(item, label):
                    counts[GlyphKit.speak(label)] = counts.get(GlyphKit.speak(label), 0) + 1
            for name in (item.shelf, item.name):
                if GlyphKit.speak(name) in counts:
                    return GlyphKit.speak(name)
            if counts:
                return max(counts, key=lambda label: counts[label])
        return ''

    def mend_text(self, name: str, observation: Any) -> Any:
        """The result in the form the agent reads (_restore); on a listing view, a record whose attributes sit among its own
        keys gets them gathered into one object: the keys the search rows do not have, holding plain values."""
        views = {self.spine.vault.folio.order.sheet.roles.get('open'), self.spine.vault.folio.order.sheet.roles.get('offer'), self.spine.vault.folio.order.sheet.roles.get('compare')}
        out = NestKit.fold_pairs(NestKit.restore_tree(observation, page=name in views or name != self.spine.vault.folio.order.sheet.roles.get('search')))
        if name not in views:
            return out
        self.spine.vault.stacks.pages.book.row_keys.update((k for ident, row in self.spine.vault.stacks.pages.book.rows.items() if ident not in self.spine.vault.stacks.pages.book.opened for k in row))
        known = self.spine.vault.stacks.pages.book.row_keys
        if not known:
            return out

        def gathered(record: Any) -> Any:
            if not isinstance(record, dict) or FormKit.fact_map(record) or not self.spine.route.find_ident(record, True):
                return record
            extra = {k: v for k, v in record.items() if k not in known and isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(extra) < Charter.RECORD_FIELDS:
                return record
            return {**{k: v for k, v in record.items() if k not in extra}, 'attributes': extra}
        if isinstance(out, dict) and self.spine.route.find_ident(out):
            return gathered(out)
        if isinstance(out, dict):
            return {k: [gathered(r) for r in v] if isinstance(v, list) else v for k, v in out.items()}
        return out

    def shrink_chat(self) -> None:
        size = sum((len(FormKit.pack(m.get('content'))) for m in self.spine.vault.folio.voice.tape.chat))
        if size <= Charter.HISTORY_LIMIT:
            return
        tools = [i for i, m in enumerate(self.spine.vault.folio.voice.tape.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.spine.vault.folio.voice.tape.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.spine.vault.stacks.pages.book.rows][:20]
                self.spine.vault.folio.voice.tape.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def ask_batches(self, free: int) -> list[list[int]]:
        """Which items each of the free messages asks about when the items outnumber them: consecutive shares as even as they
        can be, so no item goes unasked. Race 15: 19 of 90 tasks wanted 4 or 5 items and allowed 3 messages, and an agent naming
        two items in one message drew a need on either of them (10 needs in 37 such messages of one agent)."""
        numbers = [n for n in self.spine.vault.folio.voice.tape.plan_order if n in self.spine.vault.folio.voice.tape.lanes] + [n for n in self.spine.vault.folio.voice.tape.lanes if n not in self.spine.vault.folio.voice.tape.plan_order]
        numbers = [n for n in numbers if self.spine.vault.folio.voice.tape.lanes.get(n)]
        if free <= 0 or len(numbers) <= free:
            return []
        out: list[list[int]] = []
        start = 0
        for slot in range(free):
            size = (len(numbers) - start + (free - slot) - 1) // (free - slot)
            out.append(numbers[start:start + size])
            start += size
        return out

    def ask_line(self, number: int, labels: list[str], shown: bool=False) -> str:
        if len(labels) >= 2:
            return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {self.spine.route.item_words(number)}?"
        alike = self.spine.vault.folio.voice.tape.apart.get(labels[0], {}).get(number)
        if alike:
            return f"Apart from the {' and the '.join(alike)}, which {labels[0]} would you like for the {self.spine.route.item_words(number)}?"
        return self.spine.route.one_ask(number, labels[0], self.spine.route.sample_vals(number, labels[0]) if shown else None)

    def lane_for(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        settled = [GlyphKit.stemset(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.spine.route.shelf_owner(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.spine.vault.folio.order.sheet.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        every = [key for _, key, _, _ in rows if not any((NestKit.same_name(prop, GlyphKit.stemset(key)) for prop in settled))]
        keys = [key for key in every if not re.search(Charter.NOT_PROPERTY, key) or re.search('\\w[ _]models?\\b', key)]
        return [key for key in keys if not re.search(Charter.CATCH_ALL, key)] + [key for key in keys if re.search(Charter.CATCH_ALL, key)] + [key for key in every if key not in keys]

    def one_ask(self, number: int, label: str, samples: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.spine.route.item_words(number)}'
        return text + (f', for example {samples[0]} or {samples[1]}?' if samples else '?')

class FlintBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def codes_of(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(ident) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    @cushion('')
    def cheap_pairs(self, main: int, parts: list[int], index: dict[str, Any]) -> str:
        """The lowest totals of one device of the item with one part of each bound item its codes fit, among buyable rows
        seen, requirements not checked: the pick counts the whole order, not each line on its own."""
        options = []
        systems = {part: self.spine.route.series_web(part, main, index) for part in parts}
        shelves = {part: sorted(((self.spine.route.price_num(i), i, self.spine.route.codes_of(i, index), index['marks'].get(i, {}).get('bars') or []) for i in self.spine.route.ids_of_item(part) if self.spine.route.price_num(i) is not None and self.spine.route.in_stock(i)), key=lambda o: (o[0], o[1])) for part in parts}
        for device in self.spine.route.ids_of_item(main):
            codes = index['marks'].get(device, {}).get('codes') or []
            price = self.spine.route.price_num(device)
            if not (codes or (index['marks'].get(device, {}).get('named'))) or price is None or not self.spine.route.in_stock(device):
                continue
            total = price * self.spine.vault.folio.order.sheet.items[main - 1].qty
            picks = [device]
            for part in parts:
                system = systems[part]
                fitting = next((i for _, i, fits, bars in shelves[part] if (any((GlyphKit.covers(p, c) for p in fits for c in codes)) and not any((GlyphKit.covers(p, c) for p in bars for c in codes))) or (system and system[0] == part and self.spine.route.series_hit(i, device, system[2], index) is True)), None)
                if not fitting:
                    picks = []
                    break
                picks.append(fitting)
                total += (self.spine.route.price_num(fitting) or 0.0) * self.spine.vault.folio.order.sheet.items[part - 1].qty
            if picks:
                options.append((total, picks))
        options.sort(key=lambda o: o[0])
        if not options:
            return ''
        return 'lowest whole sets seen, requirements not yet checked: ' + '; '.join((f"{' + '.join(p)} = {t:g}" for t, p in options[:Charter.SET_SHOWN]))

    def line_misses(self) -> list[str]:
        self.spine.route.take_labels()
        cat_key = self.spine.route.shelf_field()
        ids = [i for i in self.spine.route.basket_ids() if (self.spine.route.units_of(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if any((self.spine.route.label_owned(item, str(r.get(cat_key) or '')) for r in self.spine.vault.stacks.pages.book.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if self.spine.route.label_owned(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.spine.route.units_of(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
                fuller = self.spine.route.richer_shelf(n, label[i]) or self.spine.route.shelf_for_item(n, label[i])
                if fuller:
                    faults.append(f'{i} is from {label[i]}, but the store has the label {fuller}, which names item {n} ({item.name}) in full: take the line from there')
        if len(known) == len(self.spine.vault.folio.order.sheet.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.spine.route.is_gone(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.spine.route.price_num(i) for i in ids}
        if isinstance(self.spine.vault.folio.order.sheet.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.spine.vault.folio.order.sheet.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.spine.vault.folio.order.sheet.budget < prices[i] <= 10 * self.spine.vault.folio.order.sheet.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.spine.vault.folio.order.sheet.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.spine.route.units_of(i) for i in ids))
                if self.spine.vault.folio.order.sheet.budget + 1e-09 < total <= 10 * self.spine.vault.folio.order.sheet.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.spine.vault.folio.order.sheet.budget:g}')
        return faults

    def trail_on_refuse(self) -> str:
        """Once, before a decline with room left: follow what a requirement points at; what that showed, '' when nothing."""
        live = {need for _, need in self.spine.route.live_needs()}
        if self.spine.vault.stacks.chains.web.refuse_trailed or self.spine.route.steps_open() < Charter.TRAIL_ROOM or self.spine.route.elapsed() > Charter.HUNT_BY or not any((need in live and need not in self.spine.vault.stacks.chains.web.ref_values for _, need in self.spine.vault.stacks.chains.web.refs)):
            return ''
        self.spine.vault.stacks.chains.web.refuse_trailed = True
        text, _ = self.spine.route.walk_chains(Charter.HUNT_BY)
        return text + 'Search for those values now and buy what fits; end without a purchase only if nothing does.' if text else ''

    def offer_name(self) -> str | None:
        return self.spine.vault.folio.order.sheet.roles.get('offer') or self.spine.vault.folio.order.sheet.roles.get('open')

    def fits_of_audit(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in FormKit.as_items(v) if str(i) in self.spine.vault.stacks.pages.book.rows and (not self.spine.route.is_gone(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def sets_units(self, name: str) -> bool:
        props = self.spine.vault.counter.front.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in FormKit.kind_list(s) and (s.get('minimum') == 0) for s in props.values()))

    @cushion([])
    def cover_lines(self) -> list[str]:
        """Searches whose result set is small enough to page and still has rows not seen, each with the exact call for its
        next page, oldest first. Pages of one query are grouped by their arguments apart from the page argument; a query
        whose next page was already read is followed to its frontier."""
        groups: dict[str, dict[str, Any]] = {}
        for mark, (name, args) in self.spine.vault.stacks.probe.log.read_args.items():
            seen = self.spine.vault.stacks.probe.log.read_calls.get(mark)
            page = self.spine.route.page_span(name, seen)
            if not page:
                continue
            count, _, param, value = page
            base = {k: v for k, v in args.items() if k != param}
            key = name + FormKit.pack(sorted(base.items(), key=lambda kv: kv[0]))
            group = groups.setdefault(key, {'name': name, 'base': base, 'param': param, 'count': 0, 'ids': set(), 'asked': set(), 'next': []})
            group['count'] = max(group['count'], count)
            group['ids'].update((i for i, _ in self.spine.route.rows_inside(seen)))
            if param in args:
                group['asked'].add(FormKit.pack(args[param]))
            if value is not None:
                group['next'].append(value)
        lines = []
        for group in groups.values():
            unseen = group['count'] - len(group['ids'])
            if unseen <= 0 or group['count'] > Charter.COVER_ROWS:
                continue
            frontier = [v for v in group['next'] if FormKit.pack(v) not in group['asked']]
            if not frontier:
                continue
            call = FormKit.pack({**group['base'], group['param']: frontier[-1]})
            lines.append(f"- {group['name']} {call}: {len(group['ids'])} of {group['count']} rows seen")
        return lines[:Charter.COVER_LINES]

    def shelf_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.spine.vault.stacks.pages.book.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.spine.vault.stacks.pages.book.rows))
        known = self.spine.route.shelf_labels()
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

    def rows_inside(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.spine.route.find_ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.spine.route.find_ident(r, True)] if i]
                elif isinstance(value, dict):
                    inner = self.spine.route.find_ident(value, False, 1)
                    found += [(inner, value)] if inner else [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.spine.route.find_ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.spine.route.find_ident(r, True)] if i]
        return []

    def compare_has_specs(self) -> bool | None:
        return self.spine.vault.stacks.pages.book.facts_from.get(self.spine.vault.folio.order.sheet.roles.get('compare') or '')

    def steps_open(self) -> int:
        return self.spine.vault.clockwork.pace.max_steps - self.spine.vault.clockwork.pace.step

    def voice_text(self, observation: Any) -> str:
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
        return ' '.join(found)[:800] if found else FormKit.pack(observation)[:400]

    def low_price_note(self, observations: list[Any]) -> str:
        if self.spine.vault.folio.order.sheet.brief.get('pick') != 'cheapest' or not self.spine.vault.stacks.pages.book.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.spine.route.rows_inside(seen)}
        rows = [(self.spine.route.price_num(i), i) for i in found if self.spine.route.in_stock(i) and self.spine.route.price_num(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[God] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def cap_mode_map(self, tool: str) -> dict[tuple[str, tuple[tuple[str, Any], ...]], bool]:
        """What the searches of `tool` so far show of each numeric argument as a price limit, by argument and mode (the call's option
        arguments): False once a call's rows held one priced above its value; True when a call's rows all sat at or below it and rows drawn
        at random from the known rows of their labels would rarely have (CAP_CHANCE), so the limit, not the query, kept dearer rows out."""
        props = (self.spine.vault.counter.front.schemas.get(tool) or {}).get('properties') or {}
        cat_key = self.spine.route.shelf_field()
        out: dict[tuple[str, tuple[tuple[str, Any], ...]], bool] = {}
        prices: dict[str, list[float]] = {}
        for ident, row in self.spine.vault.stacks.pages.book.rows.items():
            price = self.spine.route.price_num(ident)
            if price is not None:
                prices.setdefault(str(row.get(cat_key) or '').strip().lower() if cat_key else '', []).append(price)
        for mark, (name, args) in self.spine.vault.stacks.probe.log.read_args.items():
            if name != tool:
                continue
            rows = [i for i, _ in self.spine.route.rows_inside(self.spine.vault.stacks.probe.log.read_calls.get(mark))]
            shown = [self.spine.route.price_num(i) for i in rows]
            if not rows or None in shown:
                continue
            mode = tuple(sorted(((k, v) for k, v in args.items() if isinstance(props.get(k), dict) and isinstance(props[k].get('enum'), list) and isinstance(v, (str, int, float)))))
            labels = {str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '').strip().lower() for i in rows} if cat_key else set()
            for key, value in args.items():
                if key not in props or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                pool = [p for label in labels for p in prices.get(label, [])]
                if max(shown) > value:
                    out[(key, mode)] = False
                elif out.get((key, mode)) is not False and pool and (sum((p <= value for p in pool)) / len(pool)) ** len(rows) < Charter.CAP_CHANCE:
                    out[(key, mode)] = True
        return out

    def hunt_one(self, number: int, tool: str, base: dict[str, Any], field: str, room: int) -> tuple[int, int]:
        """Up to `room` searches for one item: all its required values together, then split by further words."""
        tried = self.spine.vault.stacks.probe.log.hunt_tried.setdefault(number, set())
        noun, _ = self.spine.route.hunt_phrase(number)
        values = [value for value, _ in self.spine.route.value_bits(number)]
        spent = gained = idle = 0
        issued = refused = False

        def run(query: str) -> bool:
            """Search `query` to its last page or the room, reading again no page this episode has read; whether it found any row."""
            nonlocal spent, gained, idle, issued, refused
            tried.add(query.lower())
            faults: list[str] = []
            args = FormKit.fit_value({**base, field: query}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
            refused = refused or bool(faults)
            if faults or spent >= room or self.spine.vault.clockwork.pace.finished:
                return False
            issued = True
            holds_values = bool(values) and all((v.lower() in query.lower() for v in values))
            found = False
            walked: set[str] = set()
            while not self.spine.vault.clockwork.pace.finished and self.spine.route.elapsed() <= Charter.HUNT_BY + Charter.HUNT_GRACE:
                mark = self.spine.route.call_sig(tool, args)
                if mark in walked:
                    return found
                walked.add(mark)
                if mark in self.spine.vault.stacks.probe.log.read_calls:
                    seen = self.spine.vault.stacks.probe.log.read_calls[mark]
                    new = Charter.HUNT_PAGE_NEW
                elif spent >= room:
                    return found
                else:
                    before = self.spine.route.item_row_count(number)
                    reply = self.spine.route.call_store(tool, args)
                    spent += 1
                    if reply is None:
                        return found
                    seen = reply.get('observation')
                    self.spine.route.render_obs(tool, seen)
                    new = self.spine.route.item_row_count(number) - before
                    gained += new
                    idle = 0 if new else idle + 1
                rows = self.spine.route.rows_inside(seen)
                found = found or bool(rows)
                if holds_values:
                    self.spine.vault.stacks.pages.book.value_rows.update((i for i, _ in rows))
                page = self.spine.route.page_span(tool, seen)
                if not page or page[3] is None or (new < Charter.HUNT_PAGE_NEW and page[0] > Charter.HUNT_WHOLE):
                    return found
                faults = []
                args = FormKit.fit_value({**args, page[2]: page[3]}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults:
                    return found
            return found
        mark = (number, tuple(values))
        stem = self.spine.vault.stacks.probe.log.hunt_stem.get(mark)
        if stem is None:
            tries = ([' '.join((values + [noun])).strip(), ' '.join(values)] if values else []) + ([' '.join((values[:-1] + [noun])).strip()] if len(values) >= 2 else []) + ([noun] if noun else [])
            stem = ''
            for query in dict.fromkeys(tries):
                issued = False
                if run(query):
                    stem = query
                    break
                if not issued:
                    return (spent, gained)
                if query == noun:
                    self.spine.vault.stacks.probe.log.hunt_no_noun.add(number)
                    noun = ''
            self.spine.vault.stacks.probe.log.hunt_stem[mark] = stem
        while spent < room and idle < Charter.HUNT_IDLE and not refused and not self.spine.vault.clockwork.pace.finished and self.spine.route.elapsed() <= Charter.HUNT_BY + Charter.HUNT_GRACE:
            _, words = self.spine.route.hunt_phrase(number)
            words = [w for w in words if f'{w} {stem}'.strip().lower() not in tried and w not in stem.lower().split()]
            if not words:
                break
            run(f'{words[0]} {stem}'.strip())
        return (spent, gained)

    def tune_first_query(self, name: str, args: dict[str, Any], fresh: bool) -> dict[str, Any]:
        """A new search starts on its first page once the tool's page argument is known."""
        key = self.spine.vault.stacks.pages.book.page_args.get(name)
        if not key or not fresh or not isinstance(args.get(key), int):
            return args
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}
        low = props.get(key, {}).get('minimum') if isinstance(props.get(key, {}).get('minimum'), int) else 1
        return {**args, key: low}

    def ask_model(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.spine.vault.clockwork.pace.calls >= Charter.ALL_CALLS:
                return None
            if self.spine.route.model_window() < Charter.CALL_FLOOR:
                self.spine.vault.scroll.leaf.trace.append({'kind': 'late_call_skipped', 'at': round(self.spine.route.elapsed(), 1)})
                return None
            self.spine.vault.clockwork.pace.calls += 1
            tries = 0
            while True:
                message, status = self.spine.route.post_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(Charter.WAITS)) or self.spine.vault.clockwork.pace.waited + Charter.WAITS[tries] > Charter.WAIT_TOTAL:
                    break
                if self.spine.route.model_window() - Charter.WAITS[tries] * 1.2 < Charter.CALL_FLOOR:
                    break
                pause = Charter.WAITS[tries] * (0.8 + 0.4 * random.random())
                self.spine.vault.clockwork.pace.waited += Charter.WAITS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.spine.vault.clockwork.pace.no_credit = True
                return None
        return None

    def tools_now(self) -> list[dict[str, Any]]:
        left = self.spine.route.steps_open()
        ask = self.spine.vault.folio.order.sheet.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse')} - {None}
        if left <= self.spine.route.reserve_steps():
            keep = set(ends)
        elif left <= self.spine.route.reserve_steps() + 3:
            keep = set(ends) | {self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('cart_view')}
        elif left <= len(self.spine.vault.folio.order.sheet.items) + 5 and self.spine.vault.folio.basket.bin.buys == 0:
            keep = set(ends) | {self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('cart_view'), self.spine.vault.folio.order.sheet.roles.get('offer')}
        tools = [t for t in self.spine.vault.counter.front.tools if t['function']['name'] != ask or self.spine.route.note_args('x' * 200) is None]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def drive(self) -> list[dict[str, Any]]:
        if self.spine.vault.clockwork.pace.finished:
            return self.spine.vault.scroll.leaf.trace
        self.spine.route.take_brief()
        self.spine.vault.scroll.leaf.trace.append({'kind': 'brief', 'brief': self.spine.vault.folio.order.sheet.brief})
        opening = self.spine.vault.counter.front.view.get('opening')
        if opening is not None:
            opening = NestKit.restore_tree(opening)
            self.spine.route.take_obs(opening)
        moved = self.spine.route.open_pass() if not self.spine.vault.clockwork.pace.finished else ''
        self.spine.vault.clockwork.pace.env_gap = Charter.ENV_GAP
        intro = "SHOPPER'S REQUEST:\n" + self.spine.vault.counter.front.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.spine.route.show_obs(opening)
        if moved:
            intro += '\n\nGod has searched for each item and read some of its listings (one step each):\n' + moved + '\nJudge these against the requirements and search further only for what is still missing.\n\n' + self.spine.route.now_text()
        else:
            intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.spine.route.now_text()
        self.spine.vault.folio.voice.tape.chat = [{'role': 'system', 'content': Charter.GUIDE}, {'role': 'user', 'content': intro}]
        self.spine.route.drive_guarded(Charter.LOOP_CALLS, Charter.SPEND_HARD)
        if not self.spine.vault.clockwork.pace.finished:
            turned = self.spine.route.end_episode(again=True)
            if turned and (not self.spine.vault.clockwork.pace.finished) and (not self.spine.vault.clockwork.pace.broke) and (not self.spine.vault.clockwork.pace.no_credit) and (self.spine.route.steps_open() > 1):
                self.spine.vault.folio.voice.tape.chat.append({'role': 'user', 'content': '[God] God tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.spine.route.now_text()})
                self.spine.route.drive_guarded(self.spine.vault.clockwork.pace.loop_calls + 10, Charter.SPEND_HARD + 0.01)
            if not self.spine.vault.clockwork.pace.finished:
                self.spine.route.end_episode()
        self.spine.vault.scroll.leaf.trace.append({'kind': 'end', 'finished': self.spine.vault.clockwork.pace.finished, 'steps': self.spine.vault.clockwork.pace.step, 'cost': round(self.spine.vault.clockwork.pace.cost, 5), 'calls': self.spine.vault.clockwork.pace.calls})
        return self.spine.vault.scroll.leaf.trace

    def due_to_ask(self) -> bool:
        if self.spine.vault.folio.voice.tape.asked or not self.spine.vault.folio.order.sheet.roles.get('ask') or self.spine.route.notes_spare() <= 0:
            return False
        if self.spine.route.steps_open() <= 8 or self.spine.route.elapsed() > Charter.ASK_BY:
            return False
        opened = len(self.spine.vault.stacks.pages.book.full)
        if self.spine.vault.clockwork.pace.loop_calls < Charter.ASK_TURNS and not all((self.spine.route.opened_for(n) for n in range(1, len(self.spine.vault.folio.order.sheet.items) + 1))):
            return False
        if opened >= 3 * len(self.spine.vault.folio.order.sheet.items) + 1:
            return True
        return self.spine.vault.clockwork.pace.step >= max(8, self.spine.vault.clockwork.pace.max_steps // 4)

    def list_size_now(self) -> int:
        """Labels the next list names: the full list when the shopper allows few messages (each message must name as many
        properties as it can, since the reply is about the first tracked one), else few in the episode's first questions,
        since nothing shows yet how this shopper takes a list."""
        if self.spine.vault.folio.order.sheet.msg_limit <= Charter.LIST_FULL_MESSAGES:
            return self.spine.route.list_limit()
        steps = Charter.LIST_STEPS if len(self.spine.vault.folio.order.sheet.items) == 1 else Charter.LIST_STEPS_MIXED
        return min(steps[self.spine.vault.folio.voice.tape.list_asks], self.spine.route.list_limit()) if self.spine.vault.folio.voice.tape.list_asks < len(steps) else self.spine.route.list_limit()

    @staticmethod
    def reply_tag(answer: str) -> str:
        """What a reply that discloses no need says. 'puzzled': the shopper could not follow the message, so none of the
        properties it named is one the shopper tracks (race 17: a message naming no tracked property drew such a reply, a
        tracked one the shopper does not mind drew a don't-mind phrase); those properties go to the end of the item's lane
        rather than out of it, since a shopper that refuses lists (race 16) says much the same. 'free': the shopper does
        not mind, so the first tracked property the message named is free. '' when the reply says neither."""
        text = ' '.join(str(answer or '').lower().split())
        if re.search(Charter.PUZZLED, text):
            return 'puzzled'
        if re.search(Charter.FREE_REPLY, text):
            return 'free'
        return ''

    def build_lanes(self, count: int) -> dict[int, list[str]]:
        self.spine.route.take_labels()
        table = self.spine.route.spec_census()
        settled = []
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.spine.route.aside_mark(label, name, values) + (' [several features in one: ask last]' if re.search(Charter.CATCH_ALL, name) else ''))
        prompt = Charter.PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in FormKit.as_items(self.spine.vault.folio.order.sheet.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.spine.route.ask_object(prompt, [Charter.HEAVY, Charter.LIGHT], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.spine.vault.folio.order.sheet.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in FormKit.as_items(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, FormKit.as_count(pick.get('item') or 1) or 1), len(self.spine.vault.folio.order.sheet.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((GlyphKit.speak(label) != GlyphKit.speak(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(GlyphKit.speak(label))
                if number not in self.spine.vault.folio.voice.tape.plan_order:
                    self.spine.vault.folio.voice.tape.plan_order.append(number)
        for number in lanes:
            keys = self.spine.route.lane_for(number, table)
            read = max(1, self.spine.route.opened_for(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.spine.route.shelf_owner(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', GlyphKit.speak(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if GlyphKit.speak(label) in own:
                    return own[GlyphKit.speak(label)]
                near = [s for k, s in own.items() if NestKit.same_name(GlyphKit.stemset(label), GlyphKit.stemset(k)) or NestKit.same_name(GlyphKit.stemset(k), GlyphKit.stemset(label))]
                return max(near) if near else 0.0
            hinted[number] = {GlyphKit.speak(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((GlyphKit.speak(spelled(pick)) != GlyphKit.speak(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= Charter.GUARD_SHARE and (not re.search(Charter.CATCH_ALL, k))][:Charter.GUARD_KEYS]:
                if any((GlyphKit.speak(key) == GlyphKit.speak(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if GlyphKit.speak(other) not in hinted[number] and share_of(other) + Charter.GUARD_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if GlyphKit.speak(other) != GlyphKit.speak(key)]
                if weak is not None and len(self.spine.vault.folio.order.sheet.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((GlyphKit.speak(x) in hinted[number] and all((GlyphKit.speak(x) != GlyphKit.speak(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((GlyphKit.speak(key) != GlyphKit.speak(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(Charter.CATCH_ALL, GlyphKit.speak(x))] + [x for x in lanes[number] if re.search(Charter.CATCH_ALL, GlyphKit.speak(x))]
        self.spine.vault.scroll.leaf.trace.append({'kind': 'plan', 'ask': FormKit.shorten(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    @cushion(False)
    def reply_blank(self, sent: list[tuple[int, str, str]]) -> bool:
        """Whether an answer needs no reading: it names nothing the questions, the items, the listings' keys or their values name
        and holds no number, and it asks nothing about how to ask. Race 15: 5112 of 5160 non-need replies are such (race 14: 5050 of 5055) and
        the check misses few need replies once the keys count (race 15: 35 of 1355 by question words alone). A reading costs a
        model call, and a race model call about 5-10 s of a 300 s episode. The first answer of an episode is always read: it
        tells whether this shopper takes lists."""
        if self.spine.vault.folio.voice.tape.hear_reads < 1:
            return False
        known: frozenset[str] = frozenset()
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            known |= BlockKit.spellset(item.name) | BlockKit.spellset(item.shelf or '')
            for need in item.musts + item.told:
                known |= BlockKit.spellset(' '.join(re.findall('[A-Za-z0-9]+', need.split(':', 1)[0])))
        for label, name, _, _ in self.spine.route.spec_census():
            known |= BlockKit.spellset(name) | BlockKit.spellset(label)
        for record in self.spine.vault.stacks.pages.book.full.values():
            for value in (FormKit.fact_map(record) or {}).values():
                known |= {w for w in BlockKit.spellset(' '.join(re.findall('[A-Za-z0-9]+', str(value)))) if len(w) >= Charter.QUIET_WORD}
        for _, question, answer in sent:
            asked = re.sub('\\b(?:what|which|would|you|like|for|the|please|tell|me|and|or)\\b', ' ', question, flags=re.I)
            mine = known | BlockKit.spellset(' '.join(re.findall('[A-Za-z0-9]+', asked)))
            said = BlockKit.spellset(' '.join(re.findall('[A-Za-z0-9]+', answer)))
            if re.search('\\d', answer) or said & mine or re.search(Charter.ASK_WORDS, answer, re.I):
                return False
        return True

class LinenBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def ids_of_item(self, number: int) -> list[str]:
        cat_key = self.spine.route.shelf_field()
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        return [i for i in self.spine.vault.stacks.pages.book.seen_order if cat_key and self.spine.route.label_owned(item, str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or ''))]

    def items_absent(self) -> list[int]:
        """Items with no line in a cart holding fewer lines than items; items no seen listing is labelled for are not counted."""
        cat_key = self.spine.route.shelf_field()
        ids = [i for i in self.spine.route.basket_ids() if (self.spine.route.units_of(i) or 0) > 0]
        if not cat_key or len(ids) >= len(self.spine.vault.folio.order.sheet.items):
            return []
        labels = [str(r.get(cat_key) or '') for r in self.spine.vault.stacks.pages.book.rows.values()]
        held = [str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '') for i in ids]
        return [n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if any((self.spine.route.label_owned(item, label) for label in labels)) and (not any((self.spine.route.label_owned(item, label) for label in held)))]

    def qty_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.spine.route.shelf_field()
        if not cat_key:
            return []
        out = []
        for ident in self.spine.route.basket_ids():
            have = self.spine.route.units_of(ident) or 0
            owners = [n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if self.spine.route.label_owned(item, str((self.spine.vault.stacks.pages.book.rows.get(ident) or {}).get(cat_key) or ''))]
            mine = [i for i in self.spine.route.basket_ids() if (self.spine.vault.folio.basket.bin.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.spine.route.label_owned(self.spine.vault.folio.order.sheet.items[owners[0] - 1], str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.spine.vault.folio.order.sheet.items[owners[0] - 1].qty) and (not self.spine.route.is_gone(ident)):
                out.append((ident, have, self.spine.vault.folio.order.sheet.items[owners[0] - 1].qty))
        return out

    def trail_soon(self) -> str:
        """Follow the chains of requirements that point at another listing, once, while there is room to act on what they say."""
        live = {need for _, need in self.spine.route.live_needs()}
        if self.spine.route.steps_open() < Charter.TRAIL_ROOM or self.spine.route.elapsed() > Charter.HUNT_BY or not any((need in live and need not in self.spine.vault.stacks.chains.web.ref_values for _, need in self.spine.vault.stacks.chains.web.refs)):
            return ''
        kept = (self.spine.vault.stacks.chains.web.trails, self.spine.vault.stacks.chains.web.trail_blocks)
        self.spine.vault.stacks.chains.web.trails, self.spine.vault.stacks.chains.web.trail_room = 0, Charter.EARLY_TRAILS
        try:
            text, _ = self.spine.route.walk_chains(Charter.HUNT_BY)
        finally:
            (self.spine.vault.stacks.chains.web.trails, self.spine.vault.stacks.chains.web.trail_blocks), self.spine.vault.stacks.chains.web.trail_room = kept, None
        self.spine.vault.scroll.leaf.trace.append({'kind': 'early_trail', 'left': self.spine.route.steps_open(), 'values': list(self.spine.vault.stacks.chains.web.ref_values.values())[:3]})
        return text

    def reload_offers(self) -> str:
        tool = self.spine.route.offer_name()
        target = self.spine.route.one_id_arg(tool)
        if not tool or not target:
            return ''
        since = max(self.spine.vault.clockwork.pace.step - 2, self.spine.vault.folio.basket.bin.refused_at)
        stale = [i for i in self.spine.route.basket_ids() if self.spine.vault.folio.basket.bin.offer_seen.get(i, -99) < since and i not in self.spine.vault.folio.basket.bin.gone]
        if not stale or self.spine.route.steps_open() < len(stale) + 5:
            return ''
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = FormKit.fit_value({target: ident}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
            if faults or self.spine.route.pace_pinched():
                break
            before = dict(self.spine.vault.stacks.pages.book.rows.get(ident) or {})
            reply = self.spine.route.call_store(tool, args)
            if reply is None or self.spine.vault.clockwork.pace.finished:
                break
            observation = reply.get('observation')
            self.spine.vault.folio.basket.bin.offer_seen[ident] = self.spine.vault.clockwork.pace.step
            self.spine.route.render_obs(tool, observation)
            after = self.spine.vault.stacks.pages.book.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {FormKit.pack(FormKit.shorten(observation, 300))[:900]}")
        self.spine.vault.scroll.leaf.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.spine.vault.stacks.pages.book.version += 1
        return '\n'.join(lines)

    def open_row_ids(self, rows: list[str]) -> bool:
        many, joiner = (self.spine.route.compare_field(), self.spine.vault.folio.order.sheet.roles.get('compare'))
        tool = self.spine.route.spec_tool()
        target = self.spine.route.one_id_arg(tool)
        if many and joiner and (not self.spine.route.compare_is_thin()):
            plan = [(joiner, {many: rows[:self.spine.route.compare_width()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        cart_steps = len(self.spine.route.basket_calls({f'#{n}': 1 for n, _ in enumerate(self.spine.vault.folio.order.sheet.items)}, []) or self.spine.vault.folio.order.sheet.items)
        if self.spine.route.steps_open() < len(plan) + max(5, self.spine.route.reserve_steps() + cart_steps):
            return False
        before = len(self.spine.vault.stacks.pages.book.full)
        for name, args in plan:
            faults: list[str] = []
            args = FormKit.fit_value(args, self.spine.vault.counter.front.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.spine.route.call_store(name, args)
            if reply is None or self.spine.vault.clockwork.pace.finished:
                break
            self.spine.route.render_obs(name, reply.get('observation'))
        self.spine.vault.scroll.leaf.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.spine.vault.stacks.pages.book.full) - before})
        return len(self.spine.vault.stacks.pages.book.full) > before

    def fill_remove(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A removal of a held line sent without a count names the units held, since the store would take off only one."""
        spec = self.spine.vault.counter.front.schemas.get(name, {}).get('properties') or {}
        count = next((k for k, v in spec.items() if isinstance(v, dict) and 'integer' in FormKit.kind_list(v)), None)
        removal = any((isinstance((spec.get(k) or {}).get('enum'), list) and v in spec[k]['enum'][1:] for k, v in args.items()))
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.spine.vault.folio.basket.bin.cart_lines), None)
        return {**args, count: min(self.spine.vault.folio.basket.bin.cart_lines[ident], spec[count].get('maximum', self.spine.vault.folio.basket.bin.cart_lines[ident]))} if count and removal and ident and count not in args else args

    def now_text(self) -> str:
        left = self.spine.route.steps_open()
        text = f'[God] step {self.spine.vault.clockwork.pace.step} of {self.spine.vault.clockwork.pace.max_steps} used, {left} left.'
        if left <= self.spine.route.reserve_steps() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.spine.vault.folio.order.sheet.items) + 5 and self.spine.vault.folio.basket.bin.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.spine.vault.folio.basket.bin.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.spine.vault.folio.basket.bin.buys == 0 and (not self.spine.vault.folio.basket.bin.cart_touched) and (self.spine.vault.clockwork.pace.step >= self.spine.vault.clockwork.pace.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        if self.spine.route.elapsed() >= Charter.WIND_DOWN and left > self.spine.route.reserve_steps() + 1:
            text += ' TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an episode that runs out of time earns nothing.'
        live = {need for _, need in self.spine.route.live_needs()}
        lookups = [need for _, need in self.spine.vault.stacks.chains.web.refs if need in live and need not in self.spine.vault.stacks.chains.web.ref_values]
        if lookups and self.spine.vault.folio.basket.bin.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.spine.vault.folio.basket.bin.leads:
            text += '\n[review candidates] ' + ', '.join(self.spine.vault.folio.basket.bin.leads)
        plan = self.spine.route.compat_plan() if self.spine.vault.stacks.chains.web.bound and self.spine.vault.folio.basket.bin.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.spine.vault.stacks.pages.book.notes:
            text += '\n[changes so far] ' + ' | '.join(self.spine.vault.stacks.pages.book.notes[-4:])
        cover = self.spine.route.cover_lines() if self.spine.vault.folio.basket.bin.buys == 0 and left > self.spine.route.reserve_steps() + 3 else []
        if cover:
            text += '\n[search coverage: result sets with rows not yet seen; the call shown fetches the next page] ' + ' | '.join(cover)
        text += '\n[ledger]\n' + self.spine.route.must_sheet()
        return text

    def spec_census(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.spine.route.shelf_field()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.spine.vault.stacks.pages.book.seen_order:
            record = self.spine.vault.stacks.pages.book.full.get(ident)
            if not record:
                continue
            block = FormKit.fact_map(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = GlyphKit.speak(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}, 'plain': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][ident] = text
                slot['plain'][ident] = NestKit.first_spelling.get(str(value).strip(), text).strip().lower()
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(GlyphKit.speak(item.shelf or ''), GlyphKit.speak(item.name), {w for w in GlyphKit.stemset(item.shelf or '') | GlyphKit.stemset(item.name) if len(w) >= 4}) for item in self.spine.vault.folio.order.sheet.items]
        matched = {label for label in groups if any((GlyphKit.speak(label) in (shelf, name) or {w for w in GlyphKit.stemset(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.spine.route.shelf_owner(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['plain'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.spine.route.slot_match(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot['per'].items():
                        twin['per'].setdefault(ident, text)
                        twin['plain'].setdefault(ident, slot['plain'][ident])
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:Charter.TABLE_KEYS]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def mark_view(self, name: str, observation: Any) -> None:
        for ident, record in self.spine.route.rows_inside(observation):
            keys = set(record)
            self.spine.vault.stacks.pages.book.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.spine.vault.stacks.pages.book.view_keys.get(ident, (-1, set()))
            self.spine.vault.stacks.pages.book.view_keys[ident] = (self.spine.vault.stacks.pages.book.version, (have if version == self.spine.vault.stacks.pages.book.version else set()) | keys)

    def compare_is_thin(self) -> bool:
        seen = self.spine.route.compare_has_specs()
        return seen is False or (seen is None and (self.spine.vault.stacks.pages.book.facts_from.get(self.spine.route.spec_tool() or '') is True or bool(self.spine.vault.folio.order.sheet.roles.get('offer'))))

    def elapsed(self) -> float:
        return time.monotonic() - self.spine.vault.clockwork.pace.started

    def note_bits(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.spine.route.find_ident(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.spine.route.find_ident(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.spine.vault.stacks.pages.book.heard]

    def wide_hold_note(self) -> str:
        if self.spine.vault.stacks.probe.log.wider_held or self.spine.vault.folio.order.sheet.brief.get('pick') != 'cheapest' or self.spine.route.steps_open() < 14 + 2 * len(self.spine.vault.folio.order.sheet.items):
            return ''
        self.spine.vault.stacks.probe.log.wider_held = True
        lines = [f'{i} at {self.spine.route.price_num(i):g}' for i in self.spine.route.basket_ids() if self.spine.route.price_num(i) is not None]
        return f"Not run yet (no step used). {self.spine.route.steps_open()} steps are left and the pick is the cheapest. Searches may read only part of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def search_under(self, cart: list[str]) -> None:
        plan = self.spine.route.cap_form()
        cat_key = self.spine.route.shelf_field()
        if not plan or not cat_key:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price = self.spine.route.price_num(line)
            label = str((self.spine.vault.stacks.pages.book.rows.get(line) or {}).get(cat_key) or '')
            if price is None or price < 2 or not label or (line, price) in self.spine.vault.stacks.probe.log.capped:
                continue
            self.spine.vault.stacks.probe.log.capped.add((line, price))
            limit = int(price) - 1 if float(price).is_integer() else int(price)
            tried: set[int] = set()
            for _ in range(Charter.WALK_CALLS):
                if self.spine.route.steps_open() <= self.spine.route.reserve_steps() + (Charter.WALK_ROOM if tried else 7) or self.spine.vault.clockwork.pace.finished or self.spine.route.pace_pinched():
                    return
                if limit < 1 or limit in tried:
                    break
                tried.add(limit)
                faults: list[str] = []
                args = FormKit.fit_value({**base, shelf: label, cap: limit}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults:
                    return
                reply = self.spine.route.call_store(tool, args)
                if reply is None:
                    return
                seen = reply.get('observation')
                self.spine.route.render_obs(tool, seen)
                rows = self.spine.route.rows_inside(seen)
                self.spine.vault.scroll.leaf.trace.append({'kind': 'list_below', 'line': line, 'price': price, 'limit': limit, 'rows': len(rows)})
                page = self.spine.route.page_span(tool, seen)
                if not rows or any(((self.spine.route.price_num(i) or 0) > limit for i, _ in rows)) or not page or page[0] <= page[1] or self.spine.route.price_order(seen):
                    break
                if page[0] <= 2 * page[1]:
                    call = FormKit.fit_value({**args, page[2]: page[3]}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults) if page[3] is not None else None
                    more = None if faults or call is None or self.spine.route.steps_open() <= self.spine.route.reserve_steps() + 7 or self.spine.route.pace_pinched() else self.spine.route.call_store(tool, call)
                    if more is not None:
                        self.spine.route.render_obs(tool, more.get('observation'))
                    break
                low = min((p for i, _ in rows for p in [self.spine.route.price_num(i)] if p is not None), default=None)
                if low is None:
                    break
                limit = int(low) - 1 if float(low).is_integer() else int(low)

    @cushion(list)
    def hunt_matches(self) -> list[str]:
        """Search on, without the model, before ending without a purchase.

        Race 15: 495 of 2340 episodes of 13 leaders ended without a purchase on a task that has a right answer, and on 433 of them
        another agent held a cart the store's answer set accepted, so a fit existed within reach; the same holds in practice worlds
        (a fit past the page cap, "no in-stock jeans list a slim fit"). A requirement's value and its words find listings the
        first searches did not show; the rows naming them are opened so that the audit judges their attributes. Returns what
        was read."""
        if self.spine.vault.stacks.probe.log.fit_hunted or self.spine.route.elapsed() > Charter.HUNT_BY:
            return []
        plan = self.spine.route.value_form()
        if not plan:
            return []
        tool, base, field = plan
        keep = self.spine.route.reserve_steps() + len(self.spine.vault.folio.order.sheet.items) + 1
        spare = self.spine.route.steps_open() - keep
        if spare < 3:
            return []
        self.spine.vault.stacks.probe.log.fit_hunted = True
        cat_key = self.spine.route.shelf_field()
        held = [str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '') for i in self.spine.route.basket_ids() if (self.spine.route.units_of(i) or 0) > 0] if cat_key else []
        numbers = [n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if not any((self.spine.route.label_owned(item, label) for label in held))] or list(range(1, len(self.spine.vault.folio.order.sheet.items) + 1))
        wanted = {number: {w for value, _ in self.spine.route.value_bits(number) for w in NestKit.value_pieces(value)} for number in numbers}
        paged = self.spine.route.open_unseen(tool, wanted, min(Charter.FIT_PAGES, spare // 2))
        room = min(Charter.HUNT_STEPS, self.spine.route.steps_open() - self.spine.route.reserve_steps() - Charter.HUNT_KEEP)
        spent = 0
        for number in numbers if room >= 3 else []:
            item = self.spine.vault.folio.order.sheet.items[number - 1]
            pairs = self.spine.route.value_bits(number)
            noun, words = self.spine.route.hunt_phrase(number)
            tries = [f'{family} {value} {noun}'.strip() for family in self.spine.route.series_list(number) for value in ([v for v, _ in pairs][:1] or [''])]
            tries += [f'{value} {noun}'.strip() for value, _ in pairs] + [value for value, _ in pairs if noun] + [f'{word} {noun}'.strip() for word in words[:Charter.HUNT_FIT_WORDS]] + ([noun] if noun else [])
            share = max(2, room // max(1, len(numbers)))
            used = 0
            for query in dict.fromkeys(tries):
                if spent >= room or used >= share or self.spine.vault.clockwork.pace.finished or self.spine.route.elapsed() > Charter.HUNT_BY + Charter.HUNT_GRACE:
                    break
                faults: list[str] = []
                args = FormKit.fit_value({**base, field: query}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults:
                    continue
                reply = self.spine.route.call_store(tool, args)
                spent += 1
                used += 1
                if reply is None:
                    return []
                self.spine.route.render_obs(tool, reply.get('observation'))
        picks: list[str] = []
        for number in numbers:
            item = self.spine.vault.folio.order.sheet.items[number - 1]
            pool = [i for i in self.spine.vault.stacks.pages.book.seen_order if i not in self.spine.vault.stacks.pages.book.full and self.spine.route.in_stock(i) and (not cat_key or self.spine.route.label_owned(item, str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '')))]
            cap = self.spine.vault.folio.order.sheet.budget if isinstance(self.spine.vault.folio.order.sheet.budget, (int, float)) else None
            pool = [i for i in pool if cap is None or (self.spine.route.price_num(i) or 0) <= cap]
            words = wanted.get(number) or set()
            pool.sort(key=lambda i: (-len(words & GlyphKit.text_wordset(FormKit.pack(self.spine.vault.stacks.pages.book.rows.get(i) or {}))), self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf')))
            picks += pool[:Charter.HUNT_FIT_OPENS]
        read: list[str] = []
        for start in range(0, len(picks), Charter.HUNT_FIT_OPENS):
            if self.spine.route.elapsed() > Charter.HUNT_BY + Charter.HUNT_GRACE:
                break
            read += self.spine.route.open_for_specs(picks[start:start + Charter.HUNT_FIT_OPENS], keep)
        self.spine.vault.scroll.leaf.trace.append({'kind': 'hunt_fits', 'pages': paged, 'searches': spent, 'opened': len(read)})
        return read

    @cushion('')
    def short_query_note(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A long words search that found nothing is run again with its first words, no model turn needed. Race 15: 54% of
        six-word queries and 42% of five-word ones came back empty (search wants every word), 10% of one-word ones."""
        plan = self.spine.route.value_form()
        if not plan or plan[0] != name or self.spine.route.steps_open() <= self.spine.route.reserve_steps() + Charter.SHORT_KEEP or self.spine.vault.clockwork.pace.finished or not isinstance(observation, dict):
            return ''
        field = plan[2]
        text = args.get(field)
        if not isinstance(text, str) or self.spine.route.rows_inside(observation) or len(text.split()) < 3:
            return ''
        for value in observation.values():
            if (isinstance(value, (list, str)) and value) or (isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0):
                return ''
        short = ' '.join([w for w in text.split() if not GlyphKit.harvest(w)][:2])
        if not short:
            return ''
        faults: list[str] = []
        call = FormKit.fit_value({k: v for k, v in {**args, field: short}.items() if isinstance(v, bool) or not isinstance(v, int)}, self.spine.vault.counter.front.schemas.get(name), 'args', faults)
        if faults:
            return ''
        reply = self.spine.route.call_store(name, call)
        if reply is None:
            return ''
        seen = reply.get('observation')
        self.spine.route.render_obs(name, seen)
        self.spine.vault.scroll.leaf.trace.append({'kind': 'shorter', 'from': text, 'to': short, 'rows': len(self.spine.route.rows_inside(seen))})
        return f'\n[God] Nothing has all those words; the same search with {short!r} shows:\n' + self.spine.route.show_obs(seen)

    def post_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
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
        result = Pipes.pipe_for(self.spine.route.model_window()).post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.spine.vault.clockwork.pace.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.spine.vault.scroll.leaf.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = FormKit.as_count(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.spine.vault.scroll.leaf.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def compare_field(self) -> str | None:
        schema = self.spine.vault.counter.front.schemas.get(self.spine.vault.folio.order.sheet.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in FormKit.kind_list(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def drive_guarded(self, loop_cap: int, spend_cap: float) -> None:
        for _ in range(Charter.CRASH_TURNS):
            try:
                self.spine.route.cycle(loop_cap, spend_cap)
                return
            except Exception as exc:
                self.spine.vault.scroll.leaf.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
                self.spine.route.seal_calls()

    def note_args(self, text: str) -> dict[str, Any] | None:
        name = self.spine.vault.folio.order.sheet.roles.get('ask')
        schema = self.spine.vault.counter.front.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        required = schema.get('required') or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and 'enum' not in v and 'const' not in v]
        pool = [k for k in required if k in texts] or texts
        args: dict[str, Any] = {}
        if pool:
            args[max(pool, key=lambda k: (not re.search(Charter.NOT_MESSAGE, GlyphKit.snake(k), re.I), bool(re.search(Charter.MESSAGE_FIELD, GlyphKit.snake(k), re.I)), props[k].get('maxLength') or 10 ** 6))] = text
        else:
            for key, spec in props.items():
                inner = next((k for k, v in (spec.get('properties') or {}).items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v)), None) if isinstance(spec, dict) else None
                if isinstance(spec, dict) and 'array' in FormKit.kind_list(spec):
                    args[key] = [text]
                    break
                if inner:
                    args[key] = {inner: text}
                    break
        if not args:
            return None
        for key in required:
            if key not in args:
                args[key] = FormKit.filler(props.get(key) or {})
                if args[key] is None:
                    return None
        faults: list[str] = []
        args = FormKit.fit_value(args, schema, 'args', faults)
        return None if faults else args

    def next_lane(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.spine.vault.folio.voice.tape.plan_order)}
        scarce = self.spine.vault.folio.order.sheet.msg_limit <= Charter.SCARCE_MESSAGES and (not self.spine.vault.stacks.chains.web.listing)
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, 0 if scarce else spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if GlyphKit.speak(label) not in used[number] and (not self.spine.route.already_said(number, label))]
            if left:
                return (number, left[:self.spine.route.list_size_now() if self.spine.vault.stacks.chains.web.listing else 1])
        return (0, [])

    @staticmethod
    def free_index(labels: list[str], heard: dict[str, Any], answer: str) -> int:
        """Index of the earliest label a list's reply said the shopper does not mind about, when the reply itself names
        that property (a reading that only guesses the label is not trusted); -1 otherwise."""
        said = BlockKit.spellset(' '.join(re.findall('[A-Za-z0-9]+', str(answer or ''))))
        props = [str(e.get('property') or '') for e in FormKit.as_items(heard.get('free')) if isinstance(e, dict) and str(e.get('property') or '').strip()]
        hits = []
        for prop in props:
            if not BlockKit.spellset(prop) & said:
                continue
            scores = [len(BlockKit.spellset(label) & BlockKit.spellset(prop)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def name_bits(self, number: int) -> list[str]:
        """The products an item name joins with "and" or commas when each is a plural noun ("door handles and locks"),
        else []; a single product named with "and" ("salt and pepper grinder", "usb and lightning cables", "his and
        hers towels") gives []."""
        name = self.spine.vault.folio.order.sheet.items[number - 1].name
        if not re.search('\\band\\b', name, re.I):
            return []
        parts = [p.strip() for p in re.split('\\s*(?:,|\\band\\b)\\s*', name, flags=re.I) if p.strip()]
        return parts if len(parts) >= 2 and all((re.search('[a-z]{3}(?<![su])s$', p, re.I) for p in parts)) else []

class CopperBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def series_web(self, part: int, main: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """(item carrying series codes, item naming them, the series) of two items bound to each other, or None when the rows read so far
        show no such system: most rows of one item carry a code of few series, and rows of the other name at least one of those series."""
        marks = index['marks']
        for carrier, namer in ((part, main), (main, part)):
            rows = self.spine.route.ids_of_item(carrier)
            tagged = [i for i in rows if (marks.get(i) or {}).get('fam')]
            families = {f for i in tagged for f in marks[i]['fam']}
            if len(tagged) < Charter.LINK_ROWS or 2 * len(tagged) < len(rows) or len(families) > Charter.LINK_FAMILIES:
                continue
            if any((f in families for i in self.spine.route.ids_of_item(namer) for f in (marks.get(i) or {}).get('named') or [])):
                return (carrier, namer, families)
        return None

    def ref_spent(self, number: int, need: str) -> bool:
        """Whether a requirement that points at another listing stays unresolved after the chain was followed as far as the
        store allows (the pre-decline follow ran, or the checkout holds for it are spent): it then decides nothing."""
        return (number, need) in self.spine.vault.stacks.chains.web.refs and need not in self.spine.vault.stacks.chains.web.ref_values and (self.spine.vault.stacks.chains.web.refuse_trailed or self.spine.vault.stacks.chains.web.trail_blocks >= 2)

    def apply_qty(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.spine.vault.folio.order.sheet.roles.get('cart') or ''
        schema = self.spine.vault.counter.front.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and ('enum' not in v)]
        target = next((k for t, k in self.spine.vault.stacks.pages.book.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in FormKit.kind_list(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.spine.route.sets_units(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.spine.route.steps_open():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = FormKit.fit_value(args, schema, 'args', faults)
            if faults:
                break
            reply = self.spine.route.call_store(name, args)
            if reply is None or self.spine.vault.clockwork.pace.finished:
                break
            self.spine.route.apply_basket(name, args)
            self.spine.vault.folio.basket.bin.cart_said = FormKit.pack(reply.get('observation'))
            self.spine.route.render_obs(name, reply.get('observation'))
            done.append(FormKit.pack(args) + ' -> ' + self.spine.route.show_obs(reply.get('observation'))[:300])
        self.spine.vault.scroll.leaf.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def page_opened(self, ident: str) -> bool:
        """Whether a listing's page was read: opened, and through the store's page view when it has one apart from the attribute view."""
        page = self.spine.vault.folio.order.sheet.roles.get('offer')
        return ident in self.spine.vault.stacks.pages.book.opened and (not page or not self.spine.route.one_id_arg(page) or page in self.spine.vault.stacks.pages.book.views.get(ident, ()))

    def is_gone(self, ident: str) -> bool:
        return ident in self.spine.vault.folio.basket.bin.unavailable or ident in self.spine.vault.folio.basket.bin.gone

    def basket_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.spine.vault.folio.order.sheet.roles.get('cart') or ''
        schema = self.spine.vault.counter.front.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in FormKit.kind_list(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in FormKit.kind_list(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in FormKit.kind_list(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in FormKit.kind_list(s) and ('enum' not in s)]
        target = next((k for t, k in self.spine.vault.stacks.pages.book.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in FormKit.kind_list(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.spine.vault.folio.basket.bin.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.spine.route.sets_units(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.spine.vault.folio.basket.bin.cart_lines.get(ident, 0) if ident in self.spine.route.basket_ids() else 0
            if have == want:
                continue
            if have and (not self.spine.route.sets_units(name)):
                return None
            calls.append({target: ident, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def apply_basket(self, name: str, args: dict[str, Any]) -> None:
        self.spine.vault.folio.basket.bin.cart_touched = True
        self.spine.vault.folio.basket.bin.cart_at = self.spine.vault.clockwork.pace.step
        spec = self.spine.vault.counter.front.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((FormKit.as_count(v) for v in args.values() if FormKit.as_count(v) is not None and (not isinstance(v, str))), None)
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
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.spine.vault.stacks.pages.book.rows), None)
                        count = next((FormKit.as_count(v) for v in line.values() if FormKit.as_count(v) is not None and (not isinstance(v, str))), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.spine.vault.stacks.pages.book.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.spine.vault.stacks.pages.book.rows:
                lines.append((value, loose_count))
        if whole:
            self.spine.vault.folio.basket.bin.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.spine.route.sets_units(name)
        for ident, count in lines:
            have = self.spine.vault.folio.basket.bin.cart_lines.get(ident, 0)
            if removal:
                have = have - (count or 1) if any(('integer' in FormKit.kind_list(v) for v in spec.values() if isinstance(v, dict))) else have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.spine.vault.folio.basket.bin.cart_lines[ident] = have
            else:
                self.spine.vault.folio.basket.bin.cart_lines.pop(ident, None)

    def basket_sig(self) -> str:
        return FormKit.pack(sorted(self.spine.vault.folio.basket.bin.cart_lines.items()))

    def aside_mark(self, label: str, name: str, values: list[str]) -> str:
        number = self.spine.route.shelf_owner(label)
        if not number:
            return ''
        for must in self.spine.vault.folio.order.sheet.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and GlyphKit.speak(prop) != name and GlyphKit.stemset(prop) & GlyphKit.stemset(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    def seen_before(self, name: str, ident: str) -> bool:
        version, have = self.spine.vault.stacks.pages.book.view_keys.get(ident, (-1, set()))
        keys = self.spine.vault.stacks.pages.book.tool_keys.get(name)
        return bool(keys) and version == self.spine.vault.stacks.pages.book.version and (keys <= have) and (ident not in self.spine.vault.folio.basket.bin.gone)

    def page_dump(self, ident: str, record: dict[str, Any]) -> str:
        """One read listing for the model: its attributes first, since those are what it judges."""
        block = FormKit.fact_map(record) or {}
        rest = {k: v for k, v in record.items() if v is not block and not isinstance(v, (dict, list))}
        return f'{ident}: attributes ' + FormKit.pack(FormKit.shorten(block, 120))[:Charter.PAGE_ATTR_CHARS] + ' | ' + FormKit.pack(FormKit.shorten(rest, 200))[:Charter.PAGE_REST_CHARS]

    def model_window(self) -> float:
        return Charter.TASK_SECONDS - Charter.TIME_MARGIN - self.spine.route.elapsed()

    def read_note(self, lines: list[str]) -> str:
        self.spine.vault.stacks.pages.book.heard.update(lines)
        prompt = Charter.NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.spine.route.must_sheet() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.spine.route.ask_object(prompt, [Charter.LIGHT, Charter.SPARE], 2500) or {}
        gone = [str(g) for g in FormKit.as_items(got.get('gone')) if isinstance(g, (str, int))]
        self.spine.vault.folio.basket.bin.unavailable.update((g for g in gone if g in self.spine.vault.stacks.pages.book.rows))
        self.spine.vault.folio.basket.bin.note_sold = bool(gone) or got.get('sold_out') is True
        self.spine.vault.folio.basket.bin.note_named = any((g in self.spine.vault.stacks.pages.book.rows for g in gone))
        if self.spine.vault.folio.basket.bin.note_sold and len(lines) == 1:
            self.spine.vault.folio.basket.bin.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.spine.vault.folio.order.sheet.budget):
            self.spine.vault.folio.order.sheet.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in FormKit.as_items(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in FormKit.as_items(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.spine.vault.folio.order.sheet.items[min(max(1, FormKit.as_count(entry.get('item')) or 1), len(self.spine.vault.folio.order.sheet.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if GlyphKit.speak(m.split(':')[0]) == GlyphKit.speak(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.spine.vault.folio.order.sheet.items[min(max(1, FormKit.as_count(entry.get('item')) or 1), len(self.spine.vault.folio.order.sheet.items)) - 1]
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
        if self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest' and (drops or gone or self.spine.vault.folio.basket.bin.note_sold or any((c.startswith('budget now') for c in changed))):
            self.spine.vault.stacks.probe.log.wider_held = False
            self.spine.vault.stacks.probe.log.cheaper_marks = set()
        if changed or summary:
            self.spine.vault.stacks.pages.book.version += 1
            self.spine.vault.folio.basket.bin.recent = []
            self.spine.vault.stacks.pages.book.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def mark_spec_tool(self, tool: str, shown: bool) -> None:
        """What a single-listing view showed: it shows attributes once any page had them, and none after two pages without."""
        if shown:
            self.spine.vault.stacks.pages.book.facts_from[tool] = True
        elif self.spine.vault.stacks.pages.book.facts_from.get(tool) is not True:
            self.spine.vault.stacks.pages.book.bare_pages[tool] = self.spine.vault.stacks.pages.book.bare_pages.get(tool, 0) + 1
            if self.spine.vault.stacks.pages.book.bare_pages[tool] >= 2:
                self.spine.vault.stacks.pages.book.facts_from[tool] = False

    def lower_ids(self) -> list[str]:
        if self.spine.vault.folio.order.sheet.brief.get('pick') != 'cheapest':
            return []
        mark = self.spine.route.basket_sig() + f'#{self.spine.vault.stacks.pages.book.version}'
        cat_key = self.spine.route.shelf_field()
        if mark in self.spine.vault.stacks.probe.log.cheaper_marks or not cat_key:
            return []
        self.spine.vault.stacks.probe.log.cheaper_marks.add(mark)
        cart = self.spine.route.basket_ids()
        self.spine.route.search_under(cart)
        picks: list[str] = []
        room = self.spine.route.compare_width() if self.spine.route.compare_has_specs() is True else 0
        for line in cart:
            price = self.spine.route.price_num(line)
            kind = str((self.spine.vault.stacks.pages.book.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.spine.vault.stacks.pages.book.seen_order if i not in cart and i not in self.spine.vault.stacks.pages.book.full and self.spine.route.in_stock(i) and (str((self.spine.vault.stacks.pages.book.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.spine.route.price_num(i) if self.spine.route.price_num(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.spine.route.price_num(i) or 0.0)
            lowest = [i for i in pool if i in self.spine.vault.stacks.pages.book.value_rows][:1]
            pool = lowest + [i for i in pool if i not in lowest]
            picks += [i for i in pool[:room or 2] if i not in picks]
        return self.spine.route.open_for_specs(picks[:room * len(cart) if room else min(6, Charter.CHEAPER_OPENS * max(1, len(cart)))], self.spine.route.reserve_steps() + 6)

    def open_unseen(self, tool: str, wanted: dict[int, set[str]], room: int) -> int:
        """Up to `room` further pages of the earliest searches that hold a required value, each item its share first, before any
        new words; the steps it took. A search whose rows all belong to another item is left. In a practice world accepted rows
        sat on unread pages of the episode's own searches in nearly every wrong decline, and in races 13-15 about 23% of passing
        carts were first seen on page 2 or later."""
        spent = 0
        cat_key = self.spine.route.shelf_field()
        share = max(1, room // max(1, len(wanted)))
        for limit in (share, room):
            for number, words in wanted.items():
                stop = min(room, spent + limit)
                for mark, (name, args) in list(self.spine.vault.stacks.probe.log.read_args.items()):
                    text = ' '.join((v for v in args.values() if isinstance(v, str))).lower()
                    if name != tool or not words & GlyphKit.text_wordset(text):
                        continue
                    seen = self.spine.vault.stacks.probe.log.read_calls[mark]
                    owners = {self.spine.route.shelf_owner(str(record.get(cat_key) or '')) for _, record in self.spine.route.rows_inside(seen)} if cat_key else set()
                    if owners and 0 not in owners and number not in owners:
                        continue
                    call, walked = args, {mark}
                    while spent < stop and not self.spine.vault.clockwork.pace.finished:
                        page = self.spine.route.page_span(tool, seen)
                        if not page or page[3] is None:
                            break
                        faults: list[str] = []
                        call = FormKit.fit_value({**call, page[2]: page[3]}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                        now = self.spine.route.call_sig(tool, call) if not faults else ''
                        if faults or now in walked:
                            break
                        walked.add(now)
                        if now in self.spine.vault.stacks.probe.log.read_calls:
                            seen = self.spine.vault.stacks.probe.log.read_calls[now]
                            continue
                        reply = self.spine.route.call_store(tool, call)
                        spent += 1
                        if reply is None:
                            return spent
                        seen = reply.get('observation')
                        self.spine.route.render_obs(tool, seen)
        return spent

    @cushion('')
    def open_pass(self) -> str:
        """Search each item once and read a few of its listings before the model's first turn, so the interview can start at once.

        A race model turn costs 5-10 s (race 15: 207 s mean per task, 16 of 180 and 30 of 180 episodes of two agents killed at 300 s,
        all on 3+ items) and the first turns only do this; one-item tasks have time, and keep the model's own first searches. Each
        item's search is its required values with its noun, then its name when that finds nothing. Returns what the store showed; '' when the search tool cannot be used without a model's reading of its
        modes, and the model then starts as usual."""
        plan = self.spine.route.value_form()
        if not plan or self.spine.vault.clockwork.pace.max_steps < Charter.OPEN_MIN_STEPS or len(self.spine.vault.folio.order.sheet.items) < Charter.OPEN_ITEMS:
            return ''
        tool, base, field = plan
        self.spine.vault.clockwork.pace.env_gap = Charter.OPEN_GAP
        shown: list[str] = []
        stop_at = self.spine.vault.clockwork.pace.step + self.spine.route.steps_open() - self.spine.route.reserve_steps() - self.spine.route.notes_spare() - Charter.OPEN_LATER * len(self.spine.vault.folio.order.sheet.items) - Charter.HUNT_KEEP // 2
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            wanted = [m.split(':', 1)[1].strip() for m in item.musts if ':' in m and m not in [need for _, need in self.spine.vault.stacks.chains.web.refs]]
            wanted = [w for w in wanted if w and len(w.split()) <= 3 and not re.search(Charter.OPEN_NOT, w, re.I)]
            noun = item.name if len(item.name.split()) <= 3 else ' '.join(item.name.split()[-2:])
            tries = list(dict.fromkeys(([' '.join(wanted[:2] + [noun])] if wanted else []) + [noun]))
            for query in tries:
                faults: list[str] = []
                args = FormKit.fit_value({**base, field: query}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults or self.spine.vault.clockwork.pace.finished or self.spine.vault.clockwork.pace.step >= stop_at:
                    break
                reply = self.spine.route.call_store(tool, args)
                if reply is None:
                    return '\n'.join(shown)
                seen = reply.get('observation')
                self.spine.route.render_obs(tool, seen)
                if self.spine.route.rows_inside(seen):
                    self.spine.vault.stacks.probe.log.searched = (self.spine.vault.stacks.probe.log.searched + [dict(args)])[-8:]
                    shown.append(f'{query}: ' + self.spine.route.show_obs(seen))
                    shown += self.spine.route.rows_until_keys(tool, args, seen, stop_at, number)
                    if self.spine.route.item_row_count(number) >= Charter.OPEN_EACH:
                        break
                else:
                    shown.append(f"{query}: {'the store refused the call: ' + self.spine.route.show_obs(seen) if BlockKit.failed_only(seen) else 'no rows'}")
        if self.spine.vault.stacks.pages.book.rows and self.spine.vault.folio.order.sheet.roles.get('ask') and self.spine.route.notes_spare() > 0:
            start = self.spine.vault.clockwork.pace.step
            found = [n for n in range(1, len(self.spine.vault.folio.order.sheet.items) + 1) if self.spine.route.item_row_count(n)]
            while self.spine.vault.clockwork.pace.step - start < self.spine.vault.clockwork.pace.max_steps // Charter.OPEN_SHARE and self.spine.vault.clockwork.pace.step < stop_at and not self.spine.vault.clockwork.pace.finished:
                if min((self.spine.route.opened_for(n) for n in found), default=Charter.OPEN_EACH) >= Charter.OPEN_EACH:
                    break
                had = len(self.spine.vault.stacks.pages.book.full)
                self.spine.route.spread_asks()
                if len(self.spine.vault.stacks.pages.book.full) == had:
                    break
        self.spine.vault.clockwork.pace.env_gap = Charter.ENV_GAP
        self.spine.vault.scroll.leaf.trace.append({'kind': 'first_moves', 'rows': len(self.spine.vault.stacks.pages.book.rows), 'opened': len(self.spine.vault.stacks.pages.book.full)})
        shown += ['read in full: ' + self.spine.route.page_dump(ident, record) for ident, record in self.spine.vault.stacks.pages.book.full.items()]
        proof = self.spine.route.proof_lines(list(self.spine.vault.stacks.pages.book.full.values()))
        return '\n'.join(shown + ([proof] if proof else []))

    def ask_object(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.spine.route.ask_model([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = FormKit.object_of((message or {}).get('content'))
            if got is not None:
                return got
            if self.spine.vault.clockwork.pace.broke or self.spine.vault.clockwork.pace.no_credit or self.spine.vault.clockwork.pace.calls >= Charter.ALL_CALLS:
                return None
        return None

    def compare_width(self) -> int:
        spec = ((self.spine.vault.counter.front.schemas.get(self.spine.vault.folio.order.sheet.roles.get('compare') or '') or {}).get('properties') or {}).get(self.spine.route.compare_field() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.spine.vault.stacks.chains.web.compare_bad:
            return max(2, min(self.spine.vault.stacks.chains.web.compare_bad - 1, max(self.spine.vault.stacks.chains.web.compare_ok, 2)))
        return max(4, self.spine.vault.stacks.chains.web.compare_ok)

    def seal_calls(self) -> None:
        """Give every call of the last model turn a reply, so that the conversation stays one the model can continue."""
        at = next((n for n in range(len(self.spine.vault.folio.voice.tape.chat) - 1, -1, -1) if self.spine.vault.folio.voice.tape.chat[n].get('tool_calls')), None)
        if at is None:
            return
        done = {m.get('tool_call_id') for m in self.spine.vault.folio.voice.tape.chat[at + 1:]}
        for call in self.spine.vault.folio.voice.tape.chat[at]['tool_calls']:
            if call['id'] not in done:
                self.spine.vault.folio.voice.tape.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'The call may not have run: something went wrong on the way. Look at the state before you repeat it.'})
        self.spine.vault.folio.voice.tape.chat.append({'role': 'user', 'content': self.spine.route.now_text()})

    def run_questions(self) -> str:
        self.spine.vault.folio.voice.tape.asked = True
        if self.spine.route.notes_spare() <= 0:
            return ''
        self.spine.route.spread_asks()
        count = self.spine.vault.folio.order.sheet.msg_limit - self.spine.vault.folio.order.sheet.msg_used
        self.spine.vault.folio.voice.tape.lanes = self.spine.route.build_lanes(max(count, self.spine.route.list_limit()) if self.spine.vault.stacks.chains.web.listing else count)
        self.spine.vault.folio.voice.tape.lane_found = {n: 0 for n in self.spine.vault.folio.voice.tape.lanes}
        self.spine.vault.folio.voice.tape.lane_spent = {n: 0 for n in self.spine.vault.folio.voice.tape.lanes}
        self.spine.vault.folio.voice.tape.lane_used = {n: set() for n in self.spine.vault.folio.voice.tape.lanes}
        self.spine.vault.folio.voice.tape.follow_up = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        groups = self.spine.route.ask_batches(self.spine.route.notes_spare()) if self.spine.vault.stacks.chains.web.listing else []
        while self.spine.route.notes_spare() > 0 and (not self.spine.vault.clockwork.pace.finished) and (self.spine.route.steps_open() > 6) and (self.spine.route.elapsed() < Charter.QUESTION_BY):
            batch = []
            while groups and self.spine.vault.stacks.chains.web.listing and not batch:
                batch = self.spine.route.batch_questions(groups.pop(0), told)
            batch = batch or self.spine.route.upcoming_asks(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.spine.vault.folio.voice.tape.lane_found if self.spine.vault.folio.voice.tape.lane_found[n] > 0]
        chain = self.spine.route.trail_soon()
        searched = self.spine.route.search_values(found_items) if found_items else ''
        if told:
            self.spine.vault.stacks.pages.book.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "God's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.spine.vault.stacks.chains.web.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if chain:
            summary += '\n' + chain
        if searched:
            summary += '\n' + searched
        if self.spine.vault.folio.basket.bin.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.spine.vault.folio.basket.bin.queued)
            self.spine.vault.folio.basket.bin.queued = []
        return summary

    def already_said(self, number: int, label: str) -> bool:
        """Whether an answer already gave a need on this property of the item: the label's words are held by the told
        property's name ("colour" after "shade colour") or hold a told name of two or more words; colour and color,
        singular and plural alike. A one-word told name ("type") hides no longer label ("bulb type")."""
        mine = BlockKit.spellset(label)
        return any((NestKit.same_name(mine, BlockKit.spellset(t.split(':', 1)[0])) for t in self.spine.vault.folio.order.sheet.items[number - 1].told))

    def answered_at(self, labels: list[str], new: list[str]) -> int:
        """Index of the earliest label a list's reply answered. Each new need maps to the label sharing most words with
        its property (colour and color alike), else to the label whose values on opened listings share most words with
        its value; -1 when no new need maps to a label (the reply is then settled like one that disclosed nothing)."""
        seen: dict[str, set[str]] = {}
        for _, key, _, values in self.spine.route.spec_census():
            seen.setdefault(GlyphKit.speak(key), set()).update((w for v in values for w in BlockKit.rootset(v)))
        hits = []
        for need in new:
            prop, _, value = need.partition(':')
            scores = [len(BlockKit.spellset(label) & BlockKit.spellset(prop)) for label in labels]
            if max(scores, default=0) == 0:
                scores = [len(seen.get(GlyphKit.speak(label), set()) & BlockKit.rootset(value)) for label in labels]
            if max(scores, default=0) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def item_words(self, number: int) -> str:
        """The item as a question names it. A name joining plural nouns with "and" is asked with "or" (race 13, 10
        agents: "door handles and locks" drew one-at-a-time in 81 of 155 questions; names joined with "or" in 0 of 50,
        "dresser or cabinet" disclosing a need in 10 of 28). After a one-at-a-time reply about the item, one of the
        products its name joins: the one whose head noun most of the item's own rows name outside their category label
        (with no such rows, the first)."""
        item = self.spine.vault.folio.order.sheet.items[number - 1]
        parts = self.spine.route.name_bits(number)
        if not parts:
            return item.name
        if number not in self.spine.vault.folio.voice.tape.one_product:
            return re.sub('\\band\\b', 'or', item.name, flags=re.I)
        cat_key = self.spine.route.shelf_field()
        rows = [row for row in self.spine.vault.stacks.pages.book.rows.values() if not cat_key or self.spine.route.label_owned(item, str(row.get(cat_key) or ''))]
        named = [BlockKit.rootset(' '.join((str(v) for k, v in row.items() if k != cat_key and isinstance(v, str)))) for row in rows]
        hits = {p: sum((1 for words in named if BlockKit.rootset(p.split()[-1]) & words)) for p in parts}
        return max(parts, key=lambda p: (hits[p], -parts.index(p)))

class GlassBraid:
    """Steps drawn from several jobs, mixed so the split does not follow the other agent."""

    def __init__(self, spine: 'Spine') -> None:
        self.spine = spine

    def series_hit(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        """Whether a row's own series is one the other row names; None when either row says nothing about it."""
        own = set((index['marks'].get(carried) or {}).get('fam') or []) & families
        named = set((index['marks'].get(namer) or {}).get('named') or []) & families
        return bool(own & named) if own and named else None

    def meets_needs(self, ident: str, number: int, skip_refs: bool=False) -> bool:
        """Whether a listing's attributes state each of the item's requirements word for word; a limit or an exclusion is
        not read here and counts as not stated. A requirement pointing at another listing is judged by the value read from
        that page, left out when `skip_refs` and unresolved, else not stated."""
        block = FormKit.fact_map(self.spine.vault.stacks.pages.book.full.get(ident) or {}) or {}
        for owner, need in self.spine.route.live_needs():
            if owner != number:
                continue
            prop, _, value = need.partition(':')
            if need in self.spine.vault.stacks.chains.web.ref_values:
                value = self.spine.vault.stacks.chains.web.ref_values[need]
            elif any((n == number and r == need for n, r in self.spine.vault.stacks.chains.web.refs)):
                if skip_refs:
                    continue
                return False
            split = lambda text: set(re.sub('[,/;|]', ' ', GlyphKit.speak(text)).split())
            words = split(value)
            if not words or words & {'not', 'no', 'least', 'most', 'under', 'over', 'unlike', 'different', 'min', 'max', 'minimum', 'maximum'}:
                return False
            if not any((NestKit.same_name(BlockKit.spellset(prop), BlockKit.spellset(k)) and words <= split(str(v)) for k, v in block.items())):
                return False
        return True

    def judge_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.spine.vault.stacks.pages.book.version}'

    def walk_chains(self, stop: float | None=None) -> tuple[str, bool]:
        live = {need for _, need in self.spine.route.live_needs()}
        pending = [need for _, need in self.spine.vault.stacks.chains.web.refs if need in live and need not in self.spine.vault.stacks.chains.web.ref_values]
        tool = self.spine.vault.folio.order.sheet.roles.get('offer') or self.spine.vault.folio.order.sheet.roles.get('open')
        target = self.spine.route.one_id_arg(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.spine.vault.stacks.chains.web.trails < min(6 if self.spine.route.steps_open() > 15 else 4, self.spine.vault.stacks.chains.web.trail_room or 6) and (self.spine.vault.clockwork.pace.cost < Charter.SPEND_HARD) and (stop is None or self.spine.route.elapsed() <= stop):
            self.spine.vault.stacks.chains.web.trails += 1
            records = []
            for ident in reversed(self.spine.vault.stacks.pages.book.seen_order):
                record = self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident) if self.spine.route.page_opened(ident) else None
                if record and len(records) < 45:
                    records.append('OPENED ' + FormKit.pack(FormKit.shorten(record, 300))[:1400])
            for ident in reversed(self.spine.vault.stacks.pages.book.seen_order):
                if len(records) >= 60:
                    break
                if not self.spine.route.page_opened(ident) or not (self.spine.vault.stacks.pages.book.full.get(ident) or self.spine.vault.stacks.pages.book.rows.get(ident)):
                    records.append('ROW ' + (FormKit.pack(FormKit.shorten(self.spine.vault.stacks.pages.book.full[ident], 120))[:700] if ident in self.spine.vault.stacks.pages.book.full else FormKit.pack(FormKit.shorten(self.spine.vault.stacks.pages.book.rows.get(ident), 80))[:300]))
            prompt = Charter.TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.spine.route.ask_object(prompt, [Charter.LIGHT] + ([Charter.HEAVY] if self.spine.vault.clockwork.pace.cost + Charter.HEAVY_ROOM <= Charter.SPEND_HARD else []), 4500) or {}
            self.spine.vault.scroll.leaf.trace.append({'kind': 'trail', 'pending': pending, 'got': FormKit.shorten(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in FormKit.as_items(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in FormKit.as_items(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((self.spine.route.page_opened(i) for i in ids)) and str(chain.get('value') or '').strip():
                    self.spine.vault.stacks.chains.web.ref_values[need] = str(chain['value']).strip()
                    self.spine.vault.stacks.pages.book.version += 1
                    continue
                if links and all((self.spine.route.page_opened(i) for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.spine.vault.stacks.chains.web.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.spine.vault.stacks.pages.book.version += 1
                    continue
                closed = [i for i in ids if i in self.spine.vault.stacks.pages.book.rows and not self.spine.route.page_opened(i)]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.spine.vault.stacks.pages.book.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.spine.vault.stacks.chains.web.ref_values]
            if not reach and gaps and tool and target and (self.spine.route.steps_open() > 12) and (self.spine.vault.stacks.chains.web.link_searches < 3):
                reach = self.spine.route.seek_link(*gaps[0])
                if self.spine.vault.scroll.leaf.trace and self.spine.vault.scroll.leaf.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.spine.vault.scroll.leaf.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.spine.route.steps_open() < len(reach) + 4) or self.spine.vault.clockwork.pace.finished:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = FormKit.fit_value({target: ident}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
                if faults or self.spine.route.pace_pinched():
                    break
                reply = self.spine.route.call_store(tool, args)
                if reply is None or self.spine.vault.clockwork.pace.finished:
                    break
                self.spine.route.render_obs(tool, reply.get('observation'))
                seen.append(f'{ident}: ' + self.spine.route.show_obs(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[God opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.spine.vault.stacks.chains.web.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.spine.vault.stacks.chains.web.ref_values.items())) + '\n'
        hold = bool(lost) and self.spine.vault.stacks.chains.web.trail_blocks < (3 if self.spine.route.steps_open() > 15 else 2) and (self.spine.route.steps_open() > 5)
        if hold:
            self.spine.vault.stacks.chains.web.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def swap_note(self) -> str:
        lost = [i for i in self.spine.route.basket_ids() if i not in self.spine.vault.folio.basket.bin.gone and self.spine.route.is_gone(i)]
        if not lost:
            return ''
        for ident in lost:
            self.spine.vault.folio.basket.bin.gone[ident] = self.spine.vault.clockwork.pace.step
        self.spine.vault.folio.basket.bin.leads = [i for i in self.spine.vault.folio.basket.bin.leads if i not in lost]
        self.spine.vault.stacks.pages.book.version += 1
        picks = self.spine.route.swap_ids(lost)
        text = f"[God] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.spine.route.steps_open() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.spine.vault.folio.order.sheet.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def build_basket(self) -> str:
        cart, buy = (self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('buy'))
        if self.spine.vault.folio.basket.bin.completed or not self.spine.vault.folio.basket.bin.confirmed or self.spine.route.is_gone(self.spine.vault.folio.basket.bin.confirmed) or (not cart) or (not buy):
            return ''
        self.spine.vault.folio.basket.bin.completed = True
        cat_key = self.spine.route.shelf_field()

        def kind(ident: str) -> str:
            return str((self.spine.vault.stacks.pages.book.rows.get(ident) or {}).get(cat_key) or '').strip().lower() if cat_key else ident
        plan: dict[str, int] = {}
        for number, ids in self.spine.vault.folio.basket.bin.lead_items.items():
            found = re.search('\\d+', str(number))
            index = int(found.group()) if found else (1 if len(self.spine.vault.folio.order.sheet.items) == 1 else None)
            pick = next((i for i in ids if not self.spine.route.is_gone(i)), None)
            if pick and index and (1 <= index <= len(self.spine.vault.folio.order.sheet.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.spine.vault.folio.order.sheet.items[index - 1].qty
        if self.spine.vault.folio.basket.bin.confirmed not in plan:
            return ''
        now = self.spine.route.basket_ids()
        keep = {i: self.spine.vault.folio.basket.bin.cart_lines.get(i, 1) for i in now if i not in plan and (not self.spine.route.is_gone(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.spine.vault.folio.order.sheet.items):
            return ''
        lines = {**keep, **plan}
        calls = self.spine.route.basket_calls(lines, drop)
        if not calls or len(calls) + 1 > self.spine.route.steps_open():
            return ''
        self.spine.route.open_for_specs(list(lines), len(calls) + 3)
        self.spine.route.open_for_links(list(lines), len(calls) + 3)
        self.spine.route.judge_basket(list(lines), Charter.SPEND_HARD)
        if not self.spine.vault.folio.basket.bin.verdicts.get(self.spine.route.judge_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = FormKit.fit_value(args, self.spine.vault.counter.front.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.spine.route.call_store(cart, args)
            if reply is None or self.spine.vault.clockwork.pace.finished:
                return ''
            self.spine.route.apply_basket(cart, args)
            self.spine.route.render_obs(cart, reply.get('observation'))
            done.append(FormKit.pack(args))
        faults = []
        args = FormKit.fit_value(dict(self.spine.vault.folio.order.sheet.brief.get('buy_args') or {}), self.spine.vault.counter.front.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.spine.route.reload_offers()
                self.spine.route.judge_basket(list(lines), Charter.SPEND_HARD)
                if not self.spine.vault.folio.basket.bin.verdicts.get(self.spine.route.judge_key(list(lines))) or self.spine.route.steps_open() < 1 or self.spine.vault.clockwork.pace.finished or any((self.spine.route.is_gone(i) for i in lines)):
                    break
            go, said = self.spine.route.pass_stage('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.spine.route.call_store(buy, args)
            self.spine.vault.scroll.leaf.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': FormKit.shorten(reply, 200)})
            if reply is None:
                break
            self.spine.vault.folio.basket.bin.buys += 1
            if not reply.get('finished'):
                self.spine.vault.folio.basket.bin.refused_at = self.spine.vault.clockwork.pace.step
            self.spine.route.render_obs(buy, reply.get('observation'))
            replies.append(self.spine.route.show_obs(reply.get('observation')))
            if self.spine.vault.clockwork.pace.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (God acted instead): the independent check confirmed ' + self.spine.vault.folio.basket.bin.confirmed + ', so God set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def take_brief(self) -> None:
        lines = []
        for tool in self.spine.vault.counter.front.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {FormKit.pack(fn.get('parameters') or {})}")
        prompt = Charter.BRIEF + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.spine.vault.counter.front.request
        if self.spine.vault.counter.front.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + FormKit.pack(FormKit.shorten(self.spine.vault.counter.front.view.get('opening'), 1500))
        brief = self.spine.route.ask_object(prompt, [Charter.LIGHT], 4000) or {}
        if not self.spine.vault.clockwork.pace.no_credit and (not FormKit.as_items(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.spine.route.ask_object(prompt, [Charter.HEAVY, Charter.LIGHT], 2500) or brief
        elif not self.spine.vault.clockwork.pace.no_credit and self.spine.route.shelf_clash(brief):
            again = self.spine.route.ask_object(prompt, [Charter.HEAVY], 2500) or {}
            if FormKit.as_items(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        while not brief and not self.spine.vault.clockwork.pace.no_credit and self.spine.vault.clockwork.pace.calls < Charter.ALL_CALLS and self.spine.route.elapsed() < Charter.BRIEF_BY:
            time.sleep(Charter.BRIEF_GAP)
            brief = self.spine.route.ask_object(prompt, [Charter.LIGHT, Charter.HEAVY], 2500) or {}
        if brief:
            self.spine.vault.clockwork.pace.waited = 0
        self.spine.vault.folio.order.sheet.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.spine.vault.folio.order.sheet.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.spine.vault.counter.front.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + FormKit.pack(self.spine.vault.counter.front.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.spine.route.ask_object(ask, [Charter.LIGHT, Charter.HEAVY], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in FormKit.as_items(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = FormKit.as_count(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in FormKit.as_items(entry.get('musts')) if m]
            self.spine.vault.folio.order.sheet.items.append(SlipLine(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[SlipLine] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and m.qty == item.qty and GlyphKit.speak(m.shelf) == GlyphKit.speak(item.shelf) and (not any((GlyphKit.speak(a.split(':')[0]) == GlyphKit.speak(b.split(':')[0]) and GlyphKit.speak(a.partition(':')[2]) != GlyphKit.speak(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.spine.vault.scroll.leaf.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.spine.vault.folio.order.sheet.items = merged
        if not self.spine.vault.folio.order.sheet.items:
            self.spine.vault.folio.order.sheet.items = [SlipLine('the requested product', '', 1, [])]
        for pair in FormKit.as_items(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(FormKit.as_count(pair.get('item')) or 0), became.get(FormKit.as_count(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.spine.vault.folio.order.sheet.items)) and (1 <= main <= len(self.spine.vault.folio.order.sheet.items)) and ((part, main) not in self.spine.vault.stacks.chains.web.bound):
                    self.spine.vault.stacks.chains.web.bound.append((part, main))
        for lookup in FormKit.as_items(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.spine.vault.folio.order.sheet.items)):
                key = GlyphKit.speak(text.split(':')[0])
                text = next((m for item in self.spine.vault.folio.order.sheet.items for m in item.musts if GlyphKit.speak(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.spine.vault.folio.order.sheet.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.spine.vault.stacks.chains.web.refs:
                self.spine.vault.stacks.chains.web.refs.append((number, text))
        self.spine.route.peel_foreign()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.spine.vault.folio.order.sheet.budget = budget
        limit = FormKit.as_count(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.spine.vault.folio.order.sheet.msg_limit = limit
        elif self.spine.vault.folio.order.sheet.roles.get('ask'):
            self.spine.vault.folio.order.sheet.msg_limit = 3
        steps = FormKit.as_count(brief.get('steps'))
        if steps is not None and 10 <= steps < self.spine.vault.clockwork.pace.max_steps:
            self.spine.vault.clockwork.pace.max_steps = steps
        for entry in FormKit.as_items(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.spine.vault.folio.order.sheet.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, FormKit.as_count(entry.get('messages')) or 0)})
        if self.spine.vault.folio.order.sheet.roles.get('ask'):
            talk = [p['messages'] for p in self.spine.vault.folio.order.sheet.process if p['when'] in ('order', 'end') and p['messages']]
            self.spine.vault.folio.order.sheet.reserve = min(max(talk), self.spine.vault.folio.order.sheet.msg_limit) if talk else 0

    def stage_pending(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.spine.vault.folio.order.sheet.process)) and (stage, self.spine.route.basket_sig() + f'#{self.spine.vault.stacks.pages.book.version}') not in self.spine.vault.folio.order.sheet.gate_ok

    @staticmethod
    def slot_match(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['plain'] if i in slot['plain'] and spec['plain'][i].strip() and slot['plain'][i].strip()]
        same = sum((1 for i in both if agree(spec['plain'][i], slot['plain'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def learn_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.spine.vault.stacks.pages.book.shapes:
            return self.spine.vault.stacks.pages.book.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.spine.vault.stacks.pages.book.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = Charter.SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + FormKit.pack(self.spine.vault.counter.front.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + FormKit.pack(FormKit.shorten(sample, 120))[:4000]
        got = self.spine.route.ask_object(prompt, [Charter.LIGHT, Charter.HEAVY], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.spine.vault.counter.front.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.spine.vault.stacks.pages.book.shapes[name] = shape
        self.spine.vault.stacks.pages.book.price_key = self.spine.vault.stacks.pages.book.price_key or shape['price']
        self.spine.vault.stacks.pages.book.stock_key = self.spine.vault.stacks.pages.book.stock_key or shape['stock']
        self.spine.vault.scroll.leaf.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def render_obs(self, name: str, observation: Any) -> str:
        self.spine.route.take_obs(observation)
        lines = self.spine.route.note_bits(observation)
        if not lines:
            return ''
        meaning = self.spine.route.read_note(lines)
        if meaning:
            self.spine.vault.folio.basket.bin.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def pace_pinched(self) -> bool:
        """No room for a read the order does not need: that read and the order itself, at this run's store speed, must end inside the margin."""
        pace = self.spine.vault.clockwork.pace.env_spent / self.spine.vault.clockwork.pace.env_count if self.spine.vault.clockwork.pace.env_count else 1.0
        return self.spine.route.model_window() <= 2 * max(pace, 1.0)

    def live_needs(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.spine.vault.folio.order.sheet.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def open_for_specs(self, ids: list[str], keep: int) -> list[str]:
        tool = self.spine.route.spec_tool()
        target = self.spine.route.one_id_arg(tool)
        if (not tool or not target) and self.spine.route.compare_has_specs() is not True:
            return []
        texts = []
        wanted = [i for i in ids if i in self.spine.vault.stacks.pages.book.rows and i not in self.spine.vault.stacks.pages.book.full and (not self.spine.route.is_gone(i))]
        side = self.spine.vault.folio.order.sheet.roles.get('compare')
        many = self.spine.route.compare_field()
        unread: list[str] = []
        while side and many and self.spine.route.compare_has_specs() is True and len(wanted) >= 2 and self.spine.route.steps_open() > keep and not self.spine.vault.clockwork.pace.finished and not self.spine.route.pace_pinched():
            room = self.spine.route.compare_width()
            batch, wanted = (wanted[:room], wanted[room:])
            faults_side: list[str] = []
            args_side = FormKit.fit_value({many: batch}, self.spine.vault.counter.front.schemas.get(side), 'args', faults_side)
            if faults_side:
                wanted = batch + wanted
                break
            reply = self.spine.route.call_store(side, args_side)
            if reply is None:
                wanted = batch + wanted
                break
            seen = reply.get('observation')
            self.spine.route.render_obs(side, seen)
            self.spine.route.mark_view(side, seen)
            rows = self.spine.route.rows_inside(seen)
            for ident, record in rows:
                texts.append(f'{ident}: ' + self.spine.route.show_obs(record)[:1500])
            if rows:
                self.spine.vault.stacks.chains.web.compare_ok = max(self.spine.vault.stacks.chains.web.compare_ok, len(rows))
                unread += [i for i in batch if i not in self.spine.vault.stacks.pages.book.full]
                continue
            self.spine.vault.stacks.chains.web.compare_bad = min(self.spine.vault.stacks.chains.web.compare_bad or 999, len(batch))
            wanted = batch + wanted
            if self.spine.route.compare_width() >= len(batch):
                break
        if not tool or not target:
            return texts
        for ident in unread + wanted:
            if self.spine.route.steps_open() <= keep or self.spine.vault.clockwork.pace.finished or self.spine.vault.stacks.pages.book.facts_from.get(tool) is False or self.spine.route.pace_pinched():
                break
            if (tool, ident) in self.spine.vault.stacks.pages.book.factless:
                continue
            faults: list[str] = []
            args = FormKit.fit_value({target: ident}, self.spine.vault.counter.front.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.spine.route.call_store(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.spine.route.render_obs(tool, seen)
            self.spine.route.mark_view(tool, seen)
            if isinstance(seen, dict) and self.spine.route.find_ident(seen):
                self.spine.route.mark_spec_tool(tool, bool(FormKit.fact_map(seen)))
            texts.append(f'{ident}: ' + self.spine.route.show_obs(seen)[:1500])
        if texts:
            self.spine.vault.scroll.leaf.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def title_slot(self) -> str | None:
        """The row key holding a listing's name: of the short texts most rows carry and few rows share, with more than one word
        on average and about as many different values as any such key has (a seller's name repeats where a listing's does not),
        the shortest."""
        cat_key = self.spine.route.shelf_field()
        totals: dict[str, list[Any]] = {}
        for row in self.spine.vault.stacks.pages.book.rows.values():
            for key, value in row.items():
                if key != cat_key and isinstance(value, str) and len(value) <= Charter.TITLE_CHARS:
                    slot = totals.setdefault(key, [0, 0, 0, set()])
                    slot[0] += len(value.split())
                    slot[1] += 1
                    slot[2] += len(value)
                    slot[3].add(value)
        kept = [(slot[2] / slot[1], key, len(slot[3])) for key, slot in totals.items() if slot[1] * 5 >= len(self.spine.vault.stacks.pages.book.rows) * 4 and slot[0] >= slot[1] * 2 and len(slot[3]) * 2 >= slot[1]]
        most = max((distinct for _, _, distinct in kept), default=0)
        ranked = [(mean, key) for mean, key, distinct in kept if distinct * 10 >= most * 9]
        return min(ranked)[1] if ranked else None

    def mark_read(self, name: str, args: dict[str, Any], seen: Any) -> None:
        """Remember what a search or listing call showed, whoever made it."""
        if name in self.spine.route.page_names() or name in {self.spine.vault.folio.order.sheet.roles.get(r) for r in ('ask', 'cart', 'cart_view', 'buy', 'refuse')}:
            return
        self.spine.route.query_fresh(name, args)
        if isinstance(seen, (dict, list)) and not BlockKit.failed_only(seen):
            self.spine.vault.stacks.probe.log.read_calls[self.spine.route.call_sig(name, args)] = seen
            self.spine.vault.stacks.probe.log.read_args[self.spine.route.call_sig(name, args)] = (name, dict(args))

    def word_form(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.spine.vault.folio.order.sheet.roles.get('search')
        schema = self.spine.vault.counter.front.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in FormKit.kind_list(v) and ('enum' not in v)]
        named = [k for k in texts if re.search(Charter.QUERY_WORDS, GlyphKit.snake(k), re.I)]
        for args in reversed(self.spine.vault.stacks.probe.log.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                labels = {str(r.get(self.spine.route.shelf_field()) or '').strip().lower() for r in self.spine.vault.stacks.pages.book.rows.values()} - {''} if self.spine.route.shelf_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.spine.vault.stacks.probe.log.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.spine.vault.stacks.probe.log.searched)) and (not any((str(a.get(k) or '').strip().lower() in labels for a in self.spine.vault.stacks.probe.log.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.spine.route.shelf_field()
                shelves = {GlyphKit.speak(item.shelf or '') for item in self.spine.vault.folio.order.sheet.items} | {GlyphKit.speak(item.name) for item in self.spine.vault.folio.order.sheet.items}
                shelves |= {GlyphKit.speak(str(row.get(cat_key) or '')) for row in self.spine.vault.stacks.pages.book.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (GlyphKit.speak(v) not in shelves and (not any(({w for w in GlyphKit.stemset(v) if len(w) >= 4} & {w for w in GlyphKit.stemset(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.spine.vault.stacks.probe.log.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [FormKit.pack(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and FormKit.pack(a[k]) == best))
                return (tool, keep, field)
        return None

    def book_result(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        """Bookkeeping for one step's result, whichever call carried it: the step count, the end of the episode, the result
        in the form the agent reads, what was read and which pages were opened."""
        self.spine.vault.clockwork.pace.step = data['step'] if isinstance(data.get('step'), int) else self.spine.vault.clockwork.pace.step + 1
        if data.get('finished'):
            self.spine.vault.clockwork.pace.finished = True
        if name != self.spine.vault.folio.order.sheet.roles.get('ask'):
            data['observation'] = self.spine.route.mend_text(name, data.get('observation'))
        seen = data.get('observation')
        self.spine.route.mark_read(name, args, seen)
        if name == self.spine.route.spec_tool() and isinstance(seen, dict) and not BlockKit.failed_only(seen) and self.spine.route.find_ident(seen) and not FormKit.fact_map(seen):
            self.spine.vault.stacks.pages.book.factless.add((name, self.spine.route.find_ident(seen)))
        if name in self.spine.route.page_names() and (not (isinstance(seen, dict) and seen.get('error'))) and not BlockKit.failed_only(seen):
            shown = FormKit.pack(seen)
            for value in args.values():
                named = [v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and (not isinstance(value, list) or v.strip() in shown)]
                self.spine.vault.stacks.pages.book.opened.update(named)
                for ident in named:
                    self.spine.vault.stacks.pages.book.views.setdefault(ident, set()).add(name)
        return data

    def fold_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.spine.vault.folio.order.sheet.roles.get('open'), self.spine.vault.folio.order.sheet.roles.get('compare'), self.spine.route.one_id_arg(self.spine.vault.folio.order.sheet.roles.get('open')))
        many = self.spine.route.compare_field()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.spine.route.compare_is_thin():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = FormKit.object_of(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (ident in self.spine.vault.stacks.pages.book.rows) and (ident not in self.spine.vault.folio.basket.bin.gone):
                picks.append((index, ident))
        picks = picks[:self.spine.route.compare_width()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': FormKit.pack({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.spine.vault.scroll.leaf.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def cycle(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.spine.vault.clockwork.pace.finished and (not self.spine.vault.clockwork.pace.broke) and (self.spine.route.steps_open() > 0) and (self.spine.vault.clockwork.pace.loop_calls < loop_cap) and (self.spine.vault.clockwork.pace.cost < spend_cap) and (self.spine.route.elapsed() < Charter.LAST_TURN or (not self.spine.route.basket_ids() and self.spine.route.model_window() > Charter.EMPTY_ROOM)):
            if self.spine.route.due_to_ask():
                summary = self.spine.route.run_questions()
                if summary:
                    self.spine.vault.folio.voice.tape.chat.append({'role': 'user', 'content': summary + '\n' + self.spine.route.now_text()})
                continue
            self.spine.route.shrink_chat()
            self.spine.vault.clockwork.pace.loop_calls += 1
            message = self.spine.route.ask_model(self.spine.vault.folio.voice.tape.chat, [Charter.LIGHT, Charter.SPARE], 3000, self.spine.route.tools_now())
            if message is None:
                self.spine.vault.clockwork.pace.misses += 1
                if self.spine.vault.clockwork.pace.no_credit or self.spine.vault.clockwork.pace.calls >= Charter.ALL_CALLS:
                    break
                if self.spine.vault.clockwork.pace.misses > 2:
                    if self.spine.route.elapsed() >= Charter.OUTAGE_BY:
                        break
                    time.sleep(Charter.BRIEF_GAP)
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else FormKit.object_of(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.spine.vault.clockwork.pace.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': FormKit.pack(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = FormKit.pack(content) if content else ''
            if not calls:
                idle += 1
                self.spine.vault.folio.voice.tape.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.spine.vault.folio.voice.tape.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.spine.route.now_text()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.spine.vault.folio.order.sheet.roles.get('cart'), self.spine.vault.folio.order.sheet.roles.get('buy'), self.spine.vault.folio.order.sheet.roles.get('refuse')} - {None}
            if not self.spine.vault.folio.voice.tape.asked and self.spine.vault.folio.order.sheet.roles.get('ask') and (self.spine.route.notes_spare() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.spine.route.steps_open() > 8) and (self.spine.route.elapsed() <= Charter.ASK_BY):
                self.spine.vault.folio.voice.tape.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.spine.vault.folio.voice.tape.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.spine.route.run_questions()
                self.spine.vault.folio.voice.tape.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.spine.route.now_text()})
                continue
            self.spine.vault.folio.voice.tape.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.spine.vault.scroll.leaf.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.spine.route.play_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.spine.route.now_text()
                self.spine.vault.folio.voice.tape.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def batch_questions(self, group: list[int], told: list[str], floor: int=6) -> list[tuple[int, str, str]]:
        """One message naming a few planned properties of each item of the group, the same list rules applying: the reply is
        about the first tracked property named, so each item's best keys lead its part and the first item's first label is the
        only one a reply disclosing nothing settles."""
        share = max(1, self.spine.route.list_size_now() // len(group)) if self.spine.route.list_size_now() < self.spine.route.list_limit() else max(Charter.MIX_LABELS, self.spine.route.list_limit() // len(group))
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            left = [label for label in self.spine.vault.folio.voice.tape.lanes.get(number) or [] if GlyphKit.speak(label) not in self.spine.vault.folio.voice.tape.lane_used.setdefault(number, set()) and (not self.spine.route.already_said(number, label))]
            if left:
                parts.append((number, left[:share]))
        if not parts:
            return []
        if len(parts) == 1:
            number, labels = parts[0]
            fitted, text = self.spine.route.trim_ask(number, labels, False)
            if not fitted:
                return []
            batch = self.spine.route.dispatch_asks([(number, labels[0], text)], floor)
            if batch:
                self.spine.route.close_ask(number, fitted, False, batch, told)
            return batch
        text = ''
        while True:
            for size in range(share, 0, -1):
                pieces = [f"{', '.join(labels[:size][:-1])} or {labels[:size][-1]}" if len(labels[:size]) >= 2 else labels[0] for _, labels in parts]
                text = ', and '.join((f'what {piece} for the {self.spine.route.item_words(number)}' for piece, (number, _) in zip(pieces, parts)))
                text = 'W' + text[1:].replace(' for the ', ' would you like for the ', 1) + '?'
                args = self.spine.route.note_args(text)
                if args and FormKit.says_text(args, text):
                    parts = [(number, labels[:size]) for number, labels in parts]
                    break
            else:
                parts = parts[:-1]
                if not parts:
                    return []
                continue
            break
        first = parts[0][0]
        batch = self.spine.route.dispatch_asks([(first, parts[0][1][0], text)], floor)
        if not batch:
            return []
        for number, _ in parts:
            self.spine.vault.folio.voice.tape.lane_spent[number] = self.spine.vault.folio.voice.tape.lane_spent.get(number, 0) + 1
        self.spine.vault.folio.voice.tape.list_asks += 1
        had = [len(item.told) for item in self.spine.vault.folio.order.sheet.items]
        heard = self.spine.route.read_replies([(0, text, batch[0][2])], (Charter.LIGHT, Charter.HEAVY))
        heard.pop('free', None)
        self.spine.route.apply_reply(heard, told)
        answered = False
        for number, labels in parts:
            new = self.spine.vault.folio.order.sheet.items[number - 1].told[had[number - 1]:]
            if new:
                answered = True
                self.spine.vault.folio.voice.tape.lane_found[number] = self.spine.vault.folio.voice.tape.lane_found.get(number, 0) + 1
                self.spine.route.land_answer(number, labels, False, list(new), False)
        crowded = any((FormKit.as_count(x) == 1 for x in FormKit.as_items(heard.get('crowded'))))
        if crowded and not answered:
            self.spine.vault.stacks.chains.web.listing = False
            self.spine.vault.scroll.leaf.trace.append({'kind': 'lists_off', 'item': first})
            return batch
        if not answered:
            if self.spine.route.reply_read(batch[0][2]) == 'puzzled':
                for number, labels in parts:
                    self.spine.vault.folio.voice.tape.lanes[number] = [label for label in self.spine.vault.folio.voice.tape.lanes.get(number) or [] if label not in labels] + labels
            else:
                self.spine.vault.folio.voice.tape.lane_used.setdefault(first, set()).add(GlyphKit.speak(parts[0][1][0]))
        self.spine.vault.scroll.leaf.trace.append({'kind': 'mixed_ask', 'parts': parts, 'answered': answered, 'reply': self.spine.route.reply_tag(batch[0][2])})
        return batch

    def trim_ask(self, number: int, labels: list[str], shown: bool) -> tuple[list[str], str]:
        """The longest leading part of `labels` whose question the message tool's schema takes whole (the schema shaper
        cuts a text past a length cap, so a list too long drops labels from the end instead), with that question;
        ([], '') when not even one label fits."""
        for size in range(len(labels), 0, -1):
            text = self.spine.route.ask_line(number, labels[:size], shown)
            args = self.spine.route.note_args(text)
            if args and FormKit.says_text(args, text):
                return (labels[:size], text)
        return ([], '')

    def take_labels(self) -> None:
        cat_key = self.spine.route.shelf_field()
        if len(self.spine.vault.folio.order.sheet.items) < 2 or not cat_key:
            return
        self.spine.route.split_shelves()
        labels = {str(record.get(cat_key) or '') for record in self.spine.vault.stacks.pages.book.full.values()} - {''}
        strays = [label for label in labels if not self.spine.route.shelf_owner(label)]
        orphans = [n for n in range(1, len(self.spine.vault.folio.order.sheet.items) + 1) if not any((self.spine.route.shelf_owner(label) == n and label not in self.spine.vault.folio.voice.tape.adopted for label in labels))]
        if strays and len(orphans) == 1:
            self.spine.vault.folio.voice.tape.adopted.update({label: orphans[0] for label in strays})
            self.spine.vault.scroll.leaf.trace.append({'kind': 'adopted', 'item': orphans[0], 'labels': strays})

    def sample_vals(self, number: int, label: str) -> list[str]:
        want = GlyphKit.speak(label)
        settled = ' '.join(self.spine.vault.folio.order.sheet.items[number - 1].musts).lower()
        for _, name, _, values in self.spine.route.spec_census():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

class Board:
    """Eight mixed braids. A call still goes through the switch."""

    def __init__(self, spine: 'Spine') -> None:
        self.reed = ReedBraid(spine)
        self.clay = ClayBraid(spine)
        self.salt = SaltBraid(spine)
        self.wool = WoolBraid(spine)
        self.flint = FlintBraid(spine)
        self.linen = LinenBraid(spine)
        self.copper = CopperBraid(spine)
        self.glass = GlassBraid(spine)

    def each(self) -> list[Any]:
        return [self.reed, self.clay, self.salt, self.wool, self.flint, self.linen, self.copper, self.glass]

class Spine:
    """One episode. The vault holds the memory; the braids hold the steps."""

    def __init__(self, problem: dict[str, Any]):
        self.vault = Vault()
        self.board = Board(self)
        self.route = Switch()
        for desk in self.board.each():
            self.route.attach(desk)
        environment = problem['environment']
        self.vault.counter.front.binding = dict(environment['binding'])
        self.vault.counter.front.view = environment['policy_view']
        self.vault.counter.front.request = str(self.vault.counter.front.view.get('query') or '')
        self.vault.counter.front.tools = [t for t in self.vault.counter.front.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.vault.counter.front.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.vault.counter.front.tools}
        steps = FormKit.as_count(self.vault.counter.front.view.get('max_steps'))
        if steps is None and isinstance(self.vault.counter.front.view.get('max_steps'), float):
            steps = int(self.vault.counter.front.view['max_steps'])
        self.vault.clockwork.pace.max_steps = steps if steps and steps > 0 else 30
        self.vault.clockwork.pace.started = time.monotonic()
        self.vault.clockwork.pace.step = 0
        self.vault.clockwork.pace.finished = bool(self.vault.counter.front.view.get('finished'))
        self.vault.clockwork.pace.broke = False
        self.vault.clockwork.pace.no_credit = False
        self.vault.clockwork.pace.waited = 0
        self.vault.clockwork.pace.misses = 0
        self.vault.clockwork.pace.cost = 0.0
        self.vault.clockwork.pace.calls = 0
        self.vault.clockwork.pace.loop_calls = 0
        self.vault.scroll.leaf.trace: list[dict[str, Any]] = Ribbon(self.route.elapsed)
        self.vault.folio.order.sheet.brief: dict[str, Any] = {}
        self.vault.folio.order.sheet.roles: dict[str, str] = {}
        self.vault.folio.order.sheet.items: list[SlipLine] = []
        self.vault.folio.order.sheet.budget: Any = None
        self.vault.folio.order.sheet.msg_limit = 0
        self.vault.folio.order.sheet.msg_used = 0
        self.vault.folio.order.sheet.process: list[dict[str, Any]] = []
        self.vault.folio.order.sheet.reserve = 0
        self.vault.folio.order.sheet.gate_msgs = 0
        self.vault.folio.order.sheet.gate_ok: set[tuple[str, str]] = set()
        self.vault.folio.order.sheet.done_log: list[str] = []
        self.vault.folio.order.sheet.done_steps: list[int] = []
        self.vault.folio.voice.tape.asked = False
        self.vault.folio.voice.tape.qa: list[tuple[str, str]] = []
        self.vault.stacks.pages.book.rows: dict[str, dict[str, Any]] = {}
        self.vault.stacks.pages.book.full: dict[str, dict[str, Any]] = {}
        self.vault.stacks.pages.book.seen_order: list[str] = []
        self.vault.stacks.pages.book.heard: set[str] = set()
        self.vault.stacks.pages.book.notes: list[str] = []
        self.vault.stacks.pages.book.version = 0
        self.vault.folio.basket.bin.cart_touched = False
        self.vault.folio.basket.bin.cart_at = -1
        self.vault.folio.basket.bin.cart_seen_at = -1
        self.vault.folio.basket.bin.cart_view: Any = None
        self.vault.folio.basket.bin.note_named = False
        self.vault.folio.basket.bin.cart_lines: dict[str, int] = {}
        self.vault.folio.basket.bin.checked: set[str] = set()
        self.vault.folio.basket.bin.audited = False
        self.vault.folio.basket.bin.offer_seen: dict[str, int] = {}
        self.vault.folio.basket.bin.gone: dict[str, int] = {}
        self.vault.folio.basket.bin.unavailable: set[str] = set()
        self.vault.folio.basket.bin.sold_lines: set[str] = set()
        self.vault.folio.basket.bin.note_sold = False
        self.vault.folio.basket.bin.held = 0
        self.vault.stacks.pages.book.id_args: set[tuple[str, str]] = set()
        self.vault.folio.basket.bin.cart_said = ''
        self.vault.folio.basket.bin.refused_at = -1
        self.vault.folio.basket.bin.leads: list[str] = []
        self.vault.folio.basket.bin.refuse_blocks = 0
        self.vault.folio.basket.bin.pushed = False
        self.vault.folio.basket.bin.built: list[str] = []
        self.vault.folio.basket.bin.dropped: set[str] = set()
        self.vault.folio.basket.bin.late_held = False
        self.vault.folio.basket.bin.lead_items: dict[str, list[str]] = {}
        self.vault.folio.basket.bin.verdicts: dict[str, bool] = {}
        self.vault.folio.basket.bin.confirmed: str | None = None
        self.vault.folio.basket.bin.completed = False
        self.vault.stacks.chains.web.link_tried: set[str] = set()
        self.vault.folio.voice.tape.lane_misses: dict[int, int] = {}
        self.vault.folio.voice.tape.adopted: dict[str, int] = {}
        self.vault.stacks.probe.log.capped: set[tuple[str, float]] = set()
        self.vault.stacks.pages.book.factless: set[tuple[str, str]] = set()
        self.vault.stacks.chains.web.trail_room: int | None = None
        self.vault.stacks.pages.book.id_key: str | None = None
        self.vault.stacks.pages.book.facts_from: dict[str, bool] = {}
        self.vault.folio.basket.bin.queued: list[str] = []
        self.vault.folio.basket.bin.buys = 0
        self.vault.folio.basket.bin.refusals = 0
        self.vault.folio.basket.bin.recent: list[str] = []
        self.vault.folio.voice.tape.chat: list[dict[str, Any]] = []
        self.vault.stacks.pages.book.opened: set[str] = set()
        self.vault.stacks.pages.book.views: dict[str, set[str]] = {}
        self.vault.stacks.chains.web.fit_cached: tuple[Any, dict[str, Any]] | None = None
        self.vault.stacks.pages.book.owners_seen: dict[tuple[str, int], int] = {}
        self.vault.stacks.chains.web.refs: list[tuple[int, str]] = []
        self.vault.stacks.chains.web.ref_values: dict[str, str] = {}
        self.vault.stacks.chains.web.trails = 0
        self.vault.stacks.chains.web.trail_blocks = 0
        self.vault.stacks.chains.web.compare_ok = 0
        self.vault.stacks.chains.web.compare_bad = 0
        self.vault.stacks.pages.book.tool_keys: dict[str, set[str]] = {}
        self.vault.stacks.pages.book.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.vault.stacks.pages.book.read_cache: dict[str, tuple[int, int, str]] = {}
        self.vault.stacks.pages.book.pages_read = 0
        self.vault.stacks.pages.book.shapes: dict[str, dict[str, Any] | None] = {}
        self.vault.stacks.pages.book.price_key: str | None = None
        self.vault.stacks.pages.book.stock_key: str | None = None
        self.vault.stacks.probe.log.cheaper_marks: set[str] = set()
        self.vault.stacks.probe.log.wider_held = False
        self.vault.stacks.probe.log.paged_obs: list[Any] = []
        self.vault.folio.voice.tape.plan_order: list[int] = []
        self.vault.folio.voice.tape.lanes: dict[int, list[str]] = {}
        self.vault.folio.voice.tape.lane_found: dict[int, int] = {}
        self.vault.folio.voice.tape.lane_spent: dict[int, int] = {}
        self.vault.folio.voice.tape.lane_used: dict[int, set[str]] = {}
        self.vault.folio.voice.tape.late_asked = False
        self.vault.folio.voice.tape.one_product: set[int] = set()
        self.vault.folio.voice.tape.list_asks = 0
        self.vault.folio.voice.tape.apart: dict[str, dict[int, list[str]]] = {}
        self.vault.folio.voice.tape.follow_up: list[tuple[int, list[str], bool]] = []
        self.vault.stacks.chains.web.listing = True
        self.vault.stacks.chains.web.line_holds = 0
        self.vault.stacks.chains.web.bound: list[tuple[int, int]] = []
        self.vault.stacks.chains.web.fit_holds = 0
        self.vault.stacks.probe.log.searched: list[dict[str, Any]] = []
        self.vault.stacks.chains.web.link_searches = 0
        self.vault.stacks.chains.web.link_seen: dict[str, list[str]] = {}
        self.vault.stacks.probe.log.hunts = 0
        self.vault.clockwork.pace.env_gap = Charter.ENV_GAP
        self.vault.clockwork.pace.env_spent = 0.0
        self.vault.clockwork.pace.env_count = 0
        self.vault.stacks.probe.log.read_calls: dict[str, Any] = {}
        self.vault.stacks.probe.log.read_args: dict[str, tuple[str, dict[str, Any]]] = {}
        self.vault.stacks.pages.book.value_rows: set[str] = set()
        self.vault.stacks.pages.book.row_keys: set[str] = set()
        self.vault.stacks.chains.web.refuse_trailed = False
        self.vault.clockwork.pace.env_at = 0.0
        self.vault.stacks.probe.log.hunt_stem: dict[tuple[int, tuple[str, ...]], str] = {}
        self.vault.stacks.probe.log.hunt_no_noun: set[int] = set()
        self.vault.stacks.pages.book.bare_pages: dict[str, int] = {}
        self.vault.stacks.probe.log.code_tries: list[bool] = []
        self.vault.stacks.probe.log.codes_dead = False
        self.vault.folio.voice.tape.hear_reads = 0
        self.vault.stacks.probe.log.fit_hunted = False
        self.vault.stacks.probe.log.hunt_mark = ''
        self.vault.stacks.probe.log.hunt_tried: dict[int, set[str]] = {}
        self.vault.stacks.pages.book.page_args: dict[str, str] = {}
        self.vault.stacks.pages.book.page_queries: dict[str, set[str]] = {}
        self.vault.stacks.probe.log.whole_held = False
        self.vault.stacks.probe.log.prefetched: dict[str, dict[str, Any]] = {}
        self.vault.stacks.probe.log.batch_ok = True
        self.vault.folio.basket.bin.swap_fits: list[str] = []

class QuarryRun:
    """The agent the store calls. It builds one spine and drives it."""

    def __init__(self, problem_data: dict[str, Any]) -> None:
        self.spine = Spine(problem_data)

    def play(self) -> list[dict[str, Any]]:
        return list(self.spine.route.drive())

    def salvage(self, exc: Exception) -> list[dict[str, Any]]:
        leaf = self.spine.vault.scroll.leaf
        leaf.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not self.spine.vault.clockwork.pace.finished:
                self.spine.route.end_episode()
        except Exception as again:
            leaf.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return list(leaf.trace)

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        quarry = QuarryRun(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return quarry.play()
    except Exception as exc:
        return quarry.salvage(exc)
__all__ = ['agent_main']
