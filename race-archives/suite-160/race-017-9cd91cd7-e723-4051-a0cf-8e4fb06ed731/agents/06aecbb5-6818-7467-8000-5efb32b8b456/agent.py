from __future__ import annotations
import json
import random
import re
import threading
from typing import Any
from src.agent.proxy_client import ProxyClient
SWIFT_MODEL = 'anthropic/claude-haiku-5.5'
DEEP_MODEL = 'anthropic/claude-sonnet-5.5'
FALLBACK_MODEL = 'openai/gpt-6-luna'
TURN_FLOOR = 42
CALL_FLOOR = 60
COST_GENTLE = 0.055
COST_FIRM = 0.078
COST_HALT = 0.085
COST_SWIFT = 0.035
PLANNED_SECONDS = 275.0
WINDUP_SECONDS = 70.0
FINAL_CALL_SECONDS = 28.0
ASKING_SECONDS = 45.0
BACKOFFS = (4, 10, 20, 35)
BACKOFF_SUM = 45
DISPLAY_LEN = 450
VIEW_CAP = 15000
DIALOGUE_CAP = 150000
CHECKPOINT_TURNS = 3
OWN_SHEETS = 6
NOISE_FIELDS = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition|mobile)\\b'
BATCH_WIDTH = 8
TAIL_ROSTERS = 1
NOISE_FIELDS_SOLO = NOISE_FIELDS[:-3] + '|size class|filter)\\b'
REPEAT_READS = 4
LOWCOST_EXPANDS = 3
_lm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_world = ProxyClient(timeout=50, max_retries=3, retry_delay=1)
MANUAL = '''\
Sable is your name; you shop for a single customer inside an online store, and nothing happens there except through the store's own tools. The grade is binary. A run scores 1 only if it closes in exactly the right state, otherwise 0. The right state is either (a) an order that contains each requested item at its requested quantity, where each line satisfies all requirements live at checkout time and the bill fits the budget, or (b), only when the store genuinely stocks nothing that satisfies them all, a run closed through the no-purchase tool in the manner the request prescribes.

TOOLS AND STEPS
- Each call to a tool uses up one step, steps run out, and thinking is free of charge. In this store, tool names, argument names and the shape of results are its own: go over each tool's description and schema, then send arguments under exactly those names with exactly the permitted values.
- Searching may be cramped. Typical limits: filtering by category, keyword matching that perhaps reads only the first few words you type, very small pages, and a token or offset for getting the next page. If your answer depends on having every match in front of you, check the store's reported total against how many rows you hold, and fetch more pages using the very token or offset the store returned. Keywords work best few and value-led: the wanted value, then the noun for the product. Zero rows from a keyword search most likely means one of your words never appears on the listings, so trim words (the noun stays) or switch to browsing the item's category; zero rows is never proof that the store does not sell the item. Too many rows from a keyword search that matched every word? Pack all of the item's required values into one query to cut it down. Zero rows from a filtered search most likely means the filter's value or the filter's parameter is off: retry using the label in the exact form result rows display, or try another filter parameter.
- Some facts, like which products an item is made for or the codes it carries, may be written only in descriptions, which keyword matching may skip. Then browse the category instead, sliced by brand or capped by price if the store lets you, and read what the rows' descriptions say.
- Rows in search results are abridged. How a listing's record is split varies by store: one view may carry the attributes (the properties), another just the offer plus description, and a view comparing listings side by side could carry either. Once Sable has learned which tool reveals attributes, the ledger says so. Property requirements are judged against those attributes, hence get them for every listing you might buy; a call that reveals several listings at once is cheaper when it reveals what you need. If a listing has not changed since you read it, leave it closed.
- Talking to the shopper is not your job: Sable's interviewer does all questioning. What the shopper answers appears in the chat and is written into the ledger.
- A request may also prescribe things to do on the way, such as an approval to get, a message to send, or a thing to verify or read beforehand. These appear in the ledger. Whatever is due immediately before the order or the no-purchase ending is done by Sable itself as soon as you invoke the tool for checkout or for ending, so neither wait on those nor give up on buying because of them; the remaining ones you do through the store's tools (product questions are for Sable's interviewer).

MATCHING
- Product kind comes from the category label. A label passes only when it matches the requested item in full; a label for a nearby kind never passes.
- A property requirement passes when the attribute for that property on the listing states the value. Ignore how the key is formatted (spaces, underscores, camel case, prefixes); one member of a comma- or slash-separated list is enough. Compare values word against word: upper or lower case, singular or plural, and variant spellings of one word are fine, and a value including every word of the requirement passes (value "<P> <Q>" passes requirement "<Q>"). Any differing word fails, even a near-synonym: no synonyms, no close kinds; a code or abbreviation is not the word behind it, and one word in common proves nothing. Bring numbers to one unit before comparing; the boundary itself is within a limit, and a range must fall completely inside it.
- The deciding attribute is the one whose key names the requirement's property (as is, or via a plain synonym): when its value differs, the listing fails, no matter what the title or other keys claim. Keys for other properties decide nothing here, related ones included (a key on one part of the product is silent about another part). Under each opened listing Sable adds [evidence] lines that list keys whose names match some requirement.
- Lacking any attribute for the property, a listing may still pass if the wanted value is written in another attribute (say, a list-type attribute that includes it), in the title, or in the description. Figures of a different quantity do not pass (measuring one thing is not measuring another), nor do figures whose text never tells what they measure.
- A requirement whose value is printed in the category label itself is passed by that category.
- A requirement may name no value and point to another listing instead (same as, different from, or near the value found on some other listing the request describes). Resolve those before anything else: track that listing down (search by its category, brand, shop or wording), open its complete page through the single-item tool or the side-by-side tool (rows from a search are insufficient), and if the page sends you on to a further listing for this property, track down and open that one likewise; whatever the last page states is the value. All pages along the chain have to be opened prior to checkout. Should the wanted value be anything but that listing's value, the requirement excludes just the listings holding that exact value, and by itself it never justifies a no-purchase ending. Should the request tie the value to a rule published by the store, the rule's output decides: apply the rule to the letter as the page gives it.
- If the listing you mean to buy refers to another page for more about itself, check that page first.
- If one requested item must suit another (accessory and device, say), choose the pair together on the strength of what the listings say (a fits-list, or a model or series code appearing on both pages). In case no accessory meeting its own requirements suits your chosen device, swap the device for another one meeting its own requirements that such an accessory does suit; ending without buying is not allowed while this swap remains untried.
- Exclusions and closed lists are binding. Wishes ("ideally", "would be nice") serve only to break ties.
- Never buy anything the store shows as sold out. Price times quantity, totalled over every line, may equal the budget but never exceed it.

SELECTION
- Cheapest requested (lowest price, smallest total): your order must carry the lowest-priced listing among ALL that pass. Collect the full set of passing listings (keep paging: rows are often unsorted by price, and Sable itself pulls the following pages of a short result), go through candidates from cheapest up, and stop at the first that passes completely. Where the search takes a price ceiling, set it: list the category beneath the price of the best pick so far, leaving only cheaper candidates to examine.
- Most or least of a measure requested: line up every passing listing on that measure and pick the extreme; listings silent on the measure are out of the running.
- Anything else: whichever listing passes everything is right, and searching for a better one only burns steps.
- Cart an item right after opening a listing that passes all of its requirements (or, for a cheapest, most or least choice, after the passing set is compared), then turn to the next item. Stores frequently sell out a line soon after it is carted; carting early leaves steps for a replacement.
- Several different items requested: one cart line each, each at its requested quantity.

MOVING TARGETS
- Things can change under you, from the shopper or from the store: a new budget, a requirement gained or lost, a price shift, a sell-out. You will find such news inside tool results, often as a refusal of an attempted checkout; a note matters whichever tool returned it, and the ledger shows which requirements currently hold. A refusal does not finish the run: take in every change, repair the cart, and try the checkout again.
- Price and stock may shift silently, so reread your chosen listing's record just before checkout.

CLOSING
- Leave exactly your chosen lines in the cart (take everything else out), look at the cart one time, then call checkout as a lone call.
- Ending without buying requires that a real search found no listing passing everything. Running low on steps is not that proof; with few steps left, buy the best cart you have assembled. Close that kind of run in the wording the request asks for; if some remark is needed, make it short and name no listing.
- Each reply needs at least one tool call. Write little: what you know, then what you do next.'''
TASK_SPEC = '''\
Turn the shopper's request below into a structured brief for a buying agent, using the store's tool list for the tool fields. Your whole reply is ONE JSON object, compact (no indentation), holding these keys:

"items": a list with an entry per distinct product to be bought; entry = {"name": the product kind, phrased as in the request, "shelf": exact copy of the store's category label for it if a tool exposes such labels, else "", "quantity": integer count of units of that one listing (1 if unstated; a pair or set sold as a single listing is 1), "musts": list of hard requirements on that product, each in the form "property: constraint" giving the precise value, limit, exclusion or closed list}.
Notes on musts. The property is named with the request's own term (if the request calls "<value>" its "<property>", write "<property>: <value>"); never replace it with a neighbouring property. The property is whatever words name it ahead of the value, even if the value's unit seems a strange fit for it: "<property> per <unit> of at least 5 <units>" becomes "<property> per <unit>: at least 5 <units>", and never "<units>: at least 5". A must that gets its value from some other listing stays ONE must containing that listing's entire description, with the relation kept in the request's words ("<property>: <relation> the <property> of the <kind> sold by <shop> with <feature>"; <relation> = the request's exact wording for how the values relate): words that describe the other listing (its kind, its shop, any "with <property> <value> and <property> <value>" that trails it) belong to that must alone; they are not requirements on the item itself. Musts never mention price, budget, stock or quantity. Story details (who it is for, where it goes, what went wrong earlier, the occasion) are context, not musts, unless they directly state a property's value or limit; words that are part of the product's name are not musts either, they go in "name". When several products appear in one "and"-linked phrase (one "I want ..." carrying one quantity, frequently a store department's title), treat them as ONE item; split only when each product has a quantity or requirements of its own.
"budget": the shopper's spending ceiling as a number, in whatever unit the request states; null if none.
"budget_unit": the request's word(s) for that unit; null if none.
"budget_scope": "total" for a cap on the whole order, "each" for a per-item cap, null when uncapped.
"in_stock_only": true by default; false only if the request clearly tolerates unavailable items.
"pick": "cheapest" when the request is after the minimum price or the smallest total; "most" or "least" for a request wanting the extreme of some measure; "any" in all other cases. Saying money is tight is no pick rule.
"pick_measure": the measure whose extreme is wanted under "most"/"least"; "" in other cases.
"messages": integer cap on messages the agent may send to the shopper; null if unstated.
"steps": integer cap on steps, moves or tool calls stated in the request; null if none.
"lookups": those musts (exact copies) whose value has to be fetched first from another listing the request points to; [] if none. Not a lookup: a requirement that one item of this order must suit or fit another item of the same order. Keep that as a must of the item worded as the request has it ("<property>: <relation> the <other item> of this order"), and keep it out of lookups.
"rules": each instruction in the request on how to work or how to finish, 12 words apiece at most.
"wishes": nice-to-haves the shopper mentions without insisting.
"ask_first": things the request explicitly wants checked with the shopper before choosing ([] if none).
"process": actions the request requires on top of finding and buying the right items (get the shopper's approval, tell or show the shopper something, check something with them, read something first, use a particular tool); each {"step": the action in 15 words at most, "when": "choose" (before products are chosen), "order" (immediately before checkout) or "end" (immediately before a no-purchase ending), "messages": messages to the shopper it costs (0 when none)}; [] if no such action is required. Not actions: requirements on items, the budget, stock, caps on messages and steps, how to finish when nothing fits.
"quiet_decline": true if the request says alternatives must not be suggested or named in case nothing fits, false if it wants alternatives offered, null if it is silent.
"fallback_buy": true if the request, for the case where nothing satisfies everything, wants the closest candidate (the one failing the fewest requirements, possibly within some price margin) offered or bought rather than a no-purchase ending; false otherwise.
"roles": object mapping each job to its exact tool name (null where no tool does the job): "search", "open" (shows a single listing's attributes or properties; if two single-listing tools exist, prefer the tool whose description advertises attributes, properties or specifications), "offer" (shows a single listing's current offer, e.g. price and stock, if that is a tool other than "open"), "compare" (shows several listings in full), "ask" (messages the shopper), "cart" (edits cart contents), "cart_view" (displays what is in the cart), "buy" (performs the checkout), "refuse" (closes the run with no purchase).
"buy_args": fixed arguments that put the "buy" tool into checking out (with a mode argument: that mode); {} if none.
"refuse_args": fixed arguments that put the "refuse" tool into ending with no purchase; {} if none.
'''
QUESTION_SPEC = '''\
Decide which questions a shopping agent should put to its shopper. A shopper keeps a handful of firm, unstated preferences, each concerning exactly one property of exactly one of the requested products. A preference surfaces only when the question names that precise property, and the shopper responds to a question solely regarding the first property in it; so each question spends exactly one property.

Inputs: the request; the items and, for each, what the request has already decided; and a numbered table of attribute keys that appeared in opened listings within each item's category, with the number of listings carrying each key and a few sample values. A hidden preference may also sit on a property no key in the table names.

Priority of properties:
0. Whatever the request says to check with the shopper (under ASK FIRST, if that section exists), each given as the property it concerns.
1. Hinted properties: a sentence in the request alludes to a key in the table yet gives no value for it; take that key, for the item the sentence talks about.
2. The remaining table keys in table order (widest coverage first); skip keys found on just one or two listings of the item.
Never pick: a property the request already decides for that item; an alias of a property already picked; a key whose values enumerate multiple buying choices at once; a key that differs on every listing (an identifier or a code); keys about sale conditions instead of the product. With more than one item, rotate among the items.
"label": the plain name of the property. For a table key, write out all its words in normal spelling (firstSecond or first_second -> "<first> <second>"; never cut to a single word of it); give the name only, without sample values, brackets, or a second property.
Give COUNT picks, best first (fewer only if the table has fewer usable keys). Your whole reply is ONE JSON object: {"ask": [{"item": number of the item, "label": "...", "hint": "the request-story words it stems from, 8 at most, or an empty string"}]}'''
ANSWER_SPEC = '''\
Here are questions a shopping agent put to its shopper, each followed by the shopper's answer. Interpret every answer; your whole reply is ONE JSON object:
{"needs": [{"item": number of the item, "need": "property: constraint", "points": true if its value has to be fetched from another listing, false if not}], "drop": [{"item": number of the item, "property": "a requirement the shopper has now abandoned"}], "free": [{"item": number of the item, "property": "..."}], "crowded": [question numbers where the answer protests that the message asked too much at once or demands one question per message; not answers merely saying they could not follow], "budget": the new total cap as a number, or null}
- "needs": each firm preference stated in any answer, with its exact value or limit and unit (e.g. "not <value>", "<value>", "at least <number> <unit>"). Name it by the property the answer itself uses, even if a different property was asked; several answers on one property form one need. Copy the value verbatim from the answer, nouns and units kept: "at least <number> <noun>" is right, "at least <number>" alone is not.
- An answer may give a value by pointing to another listing (the value is said to be like, unlike, or near the one some shop's product has, that product being identified through a few of its properties). That gives ONE need, filed under the property the answer mentions and marked "points": true. Its value is the answer's wording, copied exactly, starting at the comparison words and ending where that description ends, including whatever it states regarding any listing that page might name; no shortening, no rewording. Properties describing that other product never become needs of this item.
- "free": per question, the property the shopper declared themselves indifferent to.
- Phrasing that is folksy or roundabout still states a value when it names one.'''
NOTICE_SPEC = '''\
A note turned up while a shopping agent was working (from the shopper or from the store; either is possible). Your whole reply is ONE JSON object:
{"kind": "change" when the note modifies the task (budget, requirements, a price, availability, the order being refused), "error" when it merely says a call was malformed or rejected, "other" for everything else, "budget": the new total cap expressed in the request's unit as a number, else null, "add": [{"item": number of the item, "need": "property: constraint"}], "drop": [{"item": number of the item, "need": "the requirement now unwanted"}], "gone": [ids of listings stated to be unavailable], "sold_out": true when a listing picked for the cart is reported gone, sold out or unavailable (named or not), "price_moved": true when it reports that a price changed, "summary": "one brief sentence"}
Report a replaced requirement as two entries: a drop for the old one, an add for the new one. Leave no change in the note unreported.'''
VERIFY_SPEC = '''\
You are the last gate ahead of a shopping agent's checkout. Using only the records shown, rule whether this cart is exactly right.
Reading the records. Property requirements are decided by the listing's attribute for that property stating the value: key format is irrelevant, one item of a list suffices, case, singular or plural and spelling variants of one word pass, a value carrying each word of the required value passes, any differing word fails even with a close meaning, units get converted, the boundary is inside a limit, and a range must sit fully inside its limit. If the request ties a requirement to a rule the store gives, the rule's resulting value satisfies it. The attribute keyed to the requirement's own property governs, and a different value there fails the listing. Keys on other properties, related ones too (a key on one part of the product is silent on another part), decide nothing. With no attribute for the property, the wanted value written in another attribute, the title or the description passes (figures of another quantity do not, and nor do figures whose text never tells what they measure). If the category label itself states the required value, that category satisfies it. The category label must correspond to the requested item in full. A requirement to differ from another listing's value passes unless that exact value appears in the records.
Cart rules: one separate line per requested item at its requested quantity, no extra lines, no line shown as out of stock, and the lines' price times quantity, added up, inside the budget. When one listing has several records, trust the latest.
Category: should the CATEGORIES section show a store category named with the item's own words while some cart line's category is not, that line belongs to a nearby category and is wrong. Pairs: if the request or a requirement wants one item to work with, fit or suit some other item of the request (or a model named by the shopper), the two listings' own text must tie them together (a code or name on both records, or one record stating it covers the other); a pair with no such tie in its records is wrong.
Pick rule cheapest, most or least: also scan the other listings shown; if one plainly passes all requirements while doing better than the cart's choice under that rule, flag the cart as wrong.
Your whole reply is ONE JSON object: {"ok": true or false, "problems": ["brief concrete problem", ...], "better": [listing ids to swap in for a cart line, best first]}. Say false only when the records clearly show the problem, or when a requirement is covered by no record.'''
MATCH_RULES = '''\
an attribute decides a property when its key names that same property (key format is irrelevant, one item of a list suffices, case, plural, spelling variants and converted units all pass, a value carrying each word of the required value passes, any differing word fails even with a close meaning, and the boundary is inside a limit); if the request ties a requirement to a rule the store gives, the rule's resulting value satisfies it; keys on other properties, related ones too (a key on one part of the product is silent on another part), decide nothing; with no attribute naming the property, the wanted value written in another attribute, the title or the description passes (a figure whose text never tells what it measures passes nothing); if the category label itself states the required value, that category satisfies it; the category label must correspond to the item; a requirement to differ from another listing's value passes unless that exact value appears in the records; listings out of stock, or that would break the budget, are not fits.'''
NOFIT_SPEC = '''\
A shopping agent intends to close without a purchase, convinced nothing fits. Go over the listing records shown, requirement by requirement, under these reading rules: ''' + MATCH_RULES + '''
Records tagged ROW come from search result rows only: their page with the attributes was never read, so they cannot pass or fail on a property missing from the row.
Your whole reply is ONE JSON object: {"fits": {"item number": [ids of listings apparently meeting every requirement, cheapest first, 4 at most]}, "open": [ids of ROW listings worth opening (a requested kind, within budget) because their page might show a fit, best bets first, 6 at most], "reason": "one sentence"}'''
SPLIT_SPEC = '''\
A shopper's request has been broken down into firm requirements on one product. Some of them borrow a value from a further listing that the request indicates, and the words that single out that further listing (its kind, its shop, properties given after it through "with ...") are at times misread as requirements on the item actually being bought. Judging by how the request is worded, name the OTHER REQUIREMENTS below that merely describe that further listing. Your whole reply is ONE JSON object: {"describe_other": [those requirements, copied exactly; [] when each of them is a genuine requirement on the product]}'''
CHAIN_SPEC = '''\
Checkout is next for a shopping agent. Some requirements give no value of their own and borrow it from another listing, and one page may refer onward to yet another listing for that same property. Before the order can count, every listing on every such chain must have been opened in full by the agent. Starting at each requirement's own description, walk its chain one link after another, through the words of the pages themselves, using these records (OPENED: a page the agent opened; ROW: only seen as a search result row, page unread).
Your whole reply is ONE JSON object: {"chains": [{"need": requirement, copied exactly, "links": [{"id": that listing's id when a record below is this link, "" if none is, "about": how the requirement or the page before describes this listing}], "done": true if the chain's final page is OPENED and spells out the property's value itself, false if not, "value": that value as the last page states it, or "", "rule_value": if a rule printed on that page gives a value for this requirement, that value, else ""}]}
List links in chain order; break off at the earliest link whose page is not OPENED or cannot be found, as its wording is still unknown.'''
DUTY_SPEC = '''\
A shopping agent's next action is to STAGE. Its shopper's request demands certain steps on the way, listed below. Go through the numbered RECORD of everything the agent has done and settle, for each step that is due before it can STAGE, whether that step is done for the cart as it currently stands. Only a RECORD line showing the step actually performed makes it done; when the step concerns the cart or the order, the line must also postdate the latest cart change and carry a reply granting it. Asking about a product's properties does not perform any step.
For a step not yet done, name the one store call to make next for it: a tool out of TOOLS plus arguments. Messages for the shopper travel through the messaging tool: one short plain-worded sentence or question on that step alone, mentioning no product property or value beyond what the step needs. Name no call if every due step is done, or if the messages or steps remaining cannot accomplish the missing one.
Your whole reply is ONE JSON object: {"steps": [{"step": the step as listed, "done": true or false, "lines": [each RECORD line number where it got done]}], "hold": "" unless a reply asks for something to change before continuing, then that change, "call": {"tool": "...", "args": {...}} or null, "needs": [{"item": number of the item, "need": "property: constraint"} one per firm requirement stated in such a reply]}'''
STANDIN_SPEC = '''\
A shopping agent has lost a cart line: its listing just sold out. Among the other listing records shown, look for replacements for that line, under these reading rules: ''' + MATCH_RULES + '''
Your whole reply is ONE JSON object: {"fits": [{"id": a listing id, "price": the number the record shows as its price} for every listing apparently meeting all requirements of the sold-out line's item, 6 at most], "reason": "one sentence"}'''
LAYOUT_SPEC = '''\
The output of a store tool is shown below. Point out its parts by the very keys used in that output and in the tool's parameters. Your whole reply is ONE JSON object: {"rows": key holding the list of listings, "count": key holding the overall number of matching listings (all matches, not just those displayed), or null, "next": key holding the value that fetches the next page, or null, "next_arg": name of the tool parameter that receives that value, or null, "price": the row-level key carrying the listing's cost, or null, "stock": the row-level key that tells whether the listing is buyable at present, or null}'''

class _Prose:

    @staticmethod
    def _label_names(feature: frozenset[str], field: frozenset[str]) -> bool:
        if not feature or not field:
            return False
        return field == feature or field >= feature or (len(field) >= 2 and feature >= field)

    @staticmethod
    def _root_word(prose: str) -> frozenset[str]:
        return frozenset((wq if not (len(wq) > 3 and wq.endswith('s')) else wq[:-1] for wq in _Prose._wordings(prose).split()))

    @staticmethod
    def _wordings(field: str) -> str:
        field = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(field))
        return re.sub('[\\s_.\\-]+', ' ', field).strip().lower()

    @staticmethod
    def _json_struct(prose: Any) -> dict[str, Any] | None:
        if not isinstance(prose, str):
            return None
        prose = prose.strip()
        limit_line = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', prose, re.S)
        if limit_line:
            prose = limit_line.group(1)
        try:
            obtained = json.loads(prose)
            return None if not isinstance(obtained, dict) else obtained
        except ValueError:
            pass
        (residual, proper) = (prose.find('{'), prose.rfind('}'))
        if 0 <= residual < proper:
            try:
                obtained = json.loads(prose[residual:proper + 1])
                return None if not isinstance(obtained, dict) else obtained
            except ValueError:
                return None
        return None

    @staticmethod
    def _cut_down(val_text: Any, ceiling_num: int, recursion: int=0) -> Any:
        if 7 < recursion:
            return '...'
        if isinstance(val_text, str):
            return val_text[:ceiling_num] + '...' if not len(val_text) <= ceiling_num else val_text
        if isinstance(val_text, dict):
            return {kq: _Prose._cut_down(vq, ceiling_num, recursion + 1) for (kq, vq) in val_text.items()}
        if isinstance(val_text, list):
            return [_Prose._cut_down(vq, ceiling_num, recursion + 1) for vq in val_text]
        return val_text

    @staticmethod
    def _to_text(val_text: Any) -> str:
        try:
            return json.dumps(val_text, ensure_ascii=False, separators=(',', ':'))
        except (TypeError, ValueError):
            return str(val_text)

class _SchemaFit:

    @staticmethod
    def _tally_of(val_text: Any) -> int | None:
        if isinstance(val_text, bool):
            return None
        if isinstance(val_text, int):
            return val_text
        if isinstance(val_text, float) and val_text.is_integer():
            return int(val_text)
        if isinstance(val_text, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', val_text):
            return int(float(val_text))
        return None

    @staticmethod
    def _entries_list(val_text: Any) -> list[Any]:
        if isinstance(val_text, list):
            return val_text
        if isinstance(val_text, (str, dict)) and val_text:
            return [val_text]
        return []

    @staticmethod
    def _outline_form(val_text: Any, blueprint: Any, at_place: str, defects: list[str]) -> Any:
        if not isinstance(blueprint, dict):
            return val_text
        sorts = _SchemaFit._sorts(blueprint)
        if sorts and (not any((_SchemaFit._suits_variety(val_text, kq) for kq in sorts))):
            for variety in sorts:
                if 'integer' == variety and isinstance(val_text, float) and val_text.is_integer():
                    val_text = int(val_text)
                    break
                if variety in ('integer', 'number') and isinstance(val_text, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', val_text):
                    numeral = float(val_text)
                    val_text = numeral if not ('integer' == variety or numeral.is_integer()) else int(numeral)
                    break
                if 'string' == variety and isinstance(val_text, (int, float)) and (not isinstance(val_text, bool)):
                    val_text = str(val_text)
                    break
                if 'boolean' == variety and isinstance(val_text, str) and (val_text.strip().lower() in ('true', 'false')):
                    val_text = val_text.strip().lower() == 'true'
                    break
                if 'array' == variety and (not isinstance(val_text, list)) and (val_text is not None):
                    val_text = [val_text]
                    break
            if not any((_SchemaFit._suits_variety(val_text, kq) for kq in sorts)):
                defects.append(f"{at_place} should be {'/'.join(sorts)}")
                return val_text
        alts = blueprint.get('enum')
        if isinstance(alts, list) and val_text not in alts:
            if isinstance(val_text, str):
                minor = val_text.strip().lower()
                match = [oq for oq in alts if isinstance(oq, str) and oq.lower() == minor]
                if match:
                    val_text = match[0]
            if val_text not in alts:
                defects.append(f'{at_place}: allowed values are {alts}')
        if isinstance(val_text, str) and isinstance(blueprint.get('maxLength'), int) and (len(val_text) > blueprint['maxLength']):
            val_text = val_text[:blueprint['maxLength']]
        if isinstance(val_text, str) and isinstance(blueprint.get('minLength'), int) and (len(val_text) < blueprint['minLength']):
            defects.append(f'{at_place} is too short')
        if isinstance(val_text, (int, float)) and (not isinstance(val_text, bool)):
            if isinstance(blueprint.get('minimum'), (int, float)) and val_text < blueprint['minimum']:
                defects.append(f"{at_place} is below {blueprint['minimum']}")
            if isinstance(blueprint.get('maximum'), (int, float)) and val_text > blueprint['maximum']:
                defects.append(f"{at_place} is above {blueprint['maximum']}")
        if isinstance(val_text, list):
            if isinstance(blueprint.get('maxItems'), int) and len(val_text) > blueprint['maxItems']:
                defects.append(f"{at_place} allows at most {blueprint['maxItems']} entries")
            if isinstance(blueprint.get('minItems'), int) and len(val_text) < blueprint['minItems']:
                defects.append(f"{at_place} needs at least {blueprint['minItems']} entries")
            if isinstance(blueprint.get('items'), dict):
                val_text = [_SchemaFit._outline_form(vq, blueprint['items'], f'{at_place}[{iq}]', defects) for (iq, vq) in enumerate(val_text)]
        if isinstance(val_text, dict):
            features = {} if not isinstance(blueprint.get('properties'), dict) else blueprint.get('properties')
            if blueprint.get('additionalProperties') is False:
                for label_name in [nq for nq in val_text if nq not in features]:
                    defects.append(f'{at_place}.{label_name} is not an allowed argument')
            for label_name in blueprint.get('required') or []:
                if label_name not in val_text:
                    defects.append(f'{at_place}.{label_name} is required')
            val_text = {kq: vq if not kq in features else _SchemaFit._outline_form(vq, features.get(kq), f'{at_place}.{kq}', defects) for (kq, vq) in val_text.items()}
        return val_text

    @staticmethod
    def _suits_variety(val_text: Any, variety: str) -> bool:
        if 'string' == variety:
            return isinstance(val_text, str)
        if 'integer' == variety:
            return isinstance(val_text, int) and (not isinstance(val_text, bool))
        if 'number' == variety:
            return isinstance(val_text, (int, float)) and (not isinstance(val_text, bool))
        if 'boolean' == variety:
            return isinstance(val_text, bool)
        if 'array' == variety:
            return isinstance(val_text, list)
        if 'object' == variety:
            return isinstance(val_text, dict)
        if 'null' == variety:
            return val_text is None
        return True

    @staticmethod
    def _sorts(blueprint: dict[str, Any]) -> list[str]:
        variety = blueprint.get('type')
        if isinstance(variety, str):
            return [variety]
        return [kq for kq in variety or [] if isinstance(kq, str)]

class _RecordScan:

    @staticmethod
    def _trait_panel(element: Any) -> dict[str, Any] | None:
        if not isinstance(element, dict):
            return None
        top_pick = None
        for val_text in element.values():
            if isinstance(val_text, dict) and len(val_text) >= 2:
                atoms = {kq: vq for (kq, vq) in val_text.items() if isinstance(vq, (str, int, float)) and (not isinstance(vq, bool))}
                if len(atoms) >= 2 and (top_pick is None or len(atoms) > len(top_pick)):
                    top_pick = atoms
        return top_pick

    @staticmethod
    def _within_result_rows(element: Any, key_ident: str, recursion: int=0) -> bool:
        if 5 < recursion:
            return False
        if isinstance(element, list):
            return any((isinstance(vq, dict) and key_ident in [str(xq).strip() for xq in vq.values() if isinstance(xq, str)] for vq in element)) or any((_RecordScan._within_result_rows(vq, key_ident, recursion + 1) for vq in element))
        if isinstance(element, dict):
            return any((_RecordScan._within_result_rows(vq, key_ident, recursion + 1) for vq in element.values()))
        return False

    @staticmethod
    def _pick_up_key_id(result_rows: list[dict[str, Any]]) -> str | None:
        top_pick = None
        for field in result_rows[0]:
            values = [result_row.get(field) for result_row in result_rows]
            if not all((_RecordScan._key_id_alike(vq, 'id' in str(field).lower()) for vq in values)):
                continue
            values = [vq.strip() for vq in values]
            if len(result_rows) > 1 and len(set(values)) < len(values):
                continue
            mark_score = ('id' in str(field).lower(), sum((len(vq) for vq in values)) / len(values))
            if top_pick is None or mark_score > top_pick[0]:
                top_pick = (mark_score, field)
        return None if not top_pick else top_pick[1]

    @staticmethod
    def _key_id_alike(val_text: Any, labelled: bool=False) -> bool:
        if not isinstance(val_text, str) or not 3 <= len(val_text.strip()) <= 64 or ' ' in val_text.strip():
            return False
        return labelled or bool(re.search('\\d', val_text))

class _Stopclock:

    def __init__(self) -> None:
        self.beats = 0
        self.halted = threading.Event()
        threading.Thread(target=self.tick, daemon=True).start()

    def tick(self) -> None:
        while not self.halted.wait(0.25):
            self.beats += 1

    def elapsed(self) -> float:
        return self.beats * 0.25

    def halt(self) -> None:
        self.halted.set()

class WantedGood:

    def __init__(self, label_name: str, aisle_shelf: str, units: int, demands: list[str]):
        self.label_name = label_name
        self.aisle_shelf = aisle_shelf
        self.units = units
        self.demands = demands
        self.informed: list[str] = []
        self.discarded: list[str] = []
        self.unused: list[str] = []

class _Loop:

    def carry_on_invokes(self, invokes: list[dict[str, Any]]) -> list[tuple[str, str]]:
        outward: list[tuple[str, str]] = []
        invokes = self.combine_expands(invokes)
        self.combine_missed = False
        self.bundle_remark = False
        for (index, invoke) in enumerate(invokes):
            granted = {tq['function']['name'] for tq in self.dealt()}
            call_ref = invoke['id']
            if invoke.get('joined'):
                if not self.combine_missed:
                    outward.append((call_ref, 'Shown in the side-by-side result above: Sable opened these listings together in one step.'))
                else:
                    outward.append((call_ref, 'Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own.'))
                continue
            func = invoke.get('function') or {}
            label_name = func.get('name')
            crude = func.get('arguments')
            if self.closed_out:
                outward.append((call_ref, 'Not run: the episode is over.'))
                continue
            if label_name == self.duties.get('ask'):
                outward.append((call_ref, "Not run (no step used): Sable's interviewer handles every message to the shopper."))
                continue
            if label_name not in self.blueprints:
                outward.append((call_ref, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(granted))}."))
                continue
            if label_name not in granted:
                outward.append((call_ref, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(granted)) + '.'))
                continue
            if self.bundle_remark:
                outward.append((call_ref, 'Not run (no step used): the reply to an earlier call in this turn carried a note that changes the task. Read the note, adjust the picks and the cart, then call again.'))
                continue
            if not isinstance(crude, dict):
                params = {} if not (crude or '').strip() else _Prose._json_struct(crude)
                if params is None:
                    outward.append((call_ref, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            else:
                params = crude
            defects: list[str] = []
            params = _SchemaFit._outline_form(params, self.blueprints[label_name], 'args', defects)
            if defects:
                outward.append((call_ref, 'Not run (no step used): ' + '; '.join(defects[:6])))
                continue
            (params, incorrect) = self.settle_on_key_ids(label_name, params)
            if incorrect:
                outward.append((call_ref, 'Not run (no step used): ' + incorrect + '.'))
                continue
            pared = ''
            plenty = None if not label_name == self.duties.get('compare') else self.set_side_by_objective()
            if plenty and isinstance(params.get(plenty), list) and (len(params[plenty]) > self.set_side_by_space()) and self.set_side_by_faulty:
                space = self.set_side_by_space()
                pared = f"\n[Sable] Sent the first {space} listings only: this store shows at most {space} side by side. Not shown yet: {', '.join(map(str, params[plenty][space:]))}."
                params = {**params, plenty: params[plenty][:space]}
            objective = self.lone_objective(label_name)
            if objective and label_name in (self.duties.get('open'), self.duties.get('offer')) and (str(params.get(objective)) in self.vanished):
                outward.append((call_ref, f'Not run (no step used): {params[objective]} sold out at step {self.vanished[str(params[objective])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if label_name == self.duties.get('cart') and any(((_SchemaFit._tally_of(vq) or 0) < 0 for vq in params.values() if not isinstance(vq, str))):
                snapshot = '' if not self.basket_by <= self.basket_observed_by else _Prose._to_text(self.basket_snapshot)
                not_there = [vq for vq in params.values() if isinstance(vq, str) and vq in self.result_rows and (vq not in self.basket_entry_lines) and (vq not in snapshot) and (vq not in self.basket_remarked)]
                if not_there:
                    outward.append((call_ref, f'Not run (no step used): {not_there[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.basket_prose()))
                    continue
            wind_up = self.exists_checkout(label_name, params) or self.exists_decline(label_name, params)
            if wind_up and len(invokes) > 1:
                outward.append((call_ref, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not wind_up and self.residual() <= self.end_reserve() and (self.duties.get('buy') or self.duties.get('refuse')):
                outward.append((call_ref, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            stamp = label_name + _Prose._to_text(params)
            if not wind_up and label_name != self.duties.get('cart') and (stamp in self.newest):
                outward.append((call_ref, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            scanners = {self.duties.get(rq) for rq in ('search', 'open', 'offer', 'compare')} - {None}
            kept_back = self.scan_keep_store.get(stamp)
            if label_name in scanners and kept_back and (kept_back[0] == self.revision):
                outward.append((call_ref, f'Not run (no step used): the same call at step {kept_back[1]} returned this, and nothing has changed since:\n' + kept_back[2][:3000]))
                continue
            if objective and label_name in scanners and (label_name != self.duties.get('compare')) and self.displayed_by_now(label_name, str(params.get(objective))):
                outward.append((call_ref, f'Not run (no step used): {params[objective]} has shown everything this view gives since the last change. Its record: ' + self.log_entry_belonging(str(params[objective]))))
                continue
            if plenty and isinstance(params.get(plenty), list) and params[plenty]:
                preserve = [iq for iq in params[plenty] if not self.displayed_by_now(label_name, str(iq))]
                if not preserve:
                    outward.append((call_ref, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{iq}: {self.log_entry_belonging(str(iq))}' for iq in params[plenty]))[:4000]))
                    continue
                if len(preserve) < len(params[plenty]):
                    pared += f"\n[Sable] Left out {', '.join((str(iq) for iq in params[plenty] if iq not in preserve))}: shown in full already, nothing changed since."
                    params = {**params, plenty: preserve}
                    stamp = label_name + _Prose._to_text(params)
            if self.exists_checkout(label_name, params) and len(self.items) >= 2:
                lacking = self.absent_now_items()
                if lacking:
                    self.part_way_retentions += 1
                    if 1 == self.part_way_retentions and self.residual() > 3:
                        outward.append((call_ref, 'Not run (no step used). The cart has no line for: ' + '; '.join(lacking) + '. An order that leaves out a requested item is not accepted, even where the request lets a sold-out item go: put a fitting listing of every missing item in the cart, or end without a purchase when one of them has no fitting listing.\nCart now: ' + self.basket_prose()))
                        continue
                    decline = self.duties.get('refuse')
                    if decline:
                        defects = []
                        traded = _SchemaFit._outline_form(dict(self.outline.get('refuse_args') or {}), self.blueprints.get(decline), 'args', defects)
                        if not defects:
                            self.record_trail.append({'kind': 'partial_declined', 'missing': lacking})
                            (label_name, params) = (decline, traded)
            if wind_up and self.residual() > 2 and (self.clock_residual() > WINDUP_SECONDS - 25):
                if self.exists_checkout(label_name, params):
                    logged_rev = self.revision
                    roomier = self.roomier_retain()
                    if roomier:
                        outward.append((call_ref, roomier + '\nCart now: ' + self.basket_prose()))
                        continue
                    rescan = self.rescan_deals()
                    if self.basket_altered_cart and self.basket_by > self.basket_observed_by and self.duties.get('cart_view') and (self.residual() > 3):
                        defects = []
                        snapshot_params = _SchemaFit._outline_form({}, self.blueprints.get(self.duties['cart_view']), 'args', defects)
                        snapshot = None if defects else self.perform(self.duties['cart_view'], snapshot_params)
                        if snapshot is not None:
                            self.basket_snapshot = snapshot.get('observation')
                            self.basket_observed_by = self.move
                            alteration = self.view_out(self.duties['cart_view'], self.basket_snapshot)
                            if alteration:
                                self.backlogged = []
                                outward.append((call_ref, 'Not run yet (no step used). Looking at the cart showed a change: ' + alteration + '\nCart now: ' + self.basket_prose() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    interchange = self.interchange_clue()
                    retained = [] if not 2 > self.retained else [iq for iq in self.basket_key_ids() if iq in self.vanished]
                    if interchange or retained:
                        self.retained += 1 if not interchange else 0
                        outward.append((call_ref, 'Not run yet (no step used). ' + (interchange or f"{', '.join(retained)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.basket_prose()))
                        continue
                    (link_series, retain) = self.pursuit()
                    if retain:
                        outward.append((call_ref, 'Not run yet (no step used). ' + link_series + '\nCart now: ' + self.basket_prose()))
                        continue
                    scan = self.top_up_traits(self.basket_key_ids(), self.end_reserve() + 3) + self.less_costly_result_rows()
                    if logged_rev != self.revision and self.remarks and (self.residual() > 2):
                        self.backlogged = []
                        outward.append((call_ref, 'Not run yet (no step used). A note arrived while Sable re-read the cart: ' + self.remarks[-1] + '\nCart now: ' + self.basket_prose() + '\nApply the change to the picks (a dropped requirement or a new budget can make another listing the right one), fix the cart, then check out.'))
                        continue
                    judgement = self.verify_basket()
                    if judgement:
                        observed = '' if not scan else "Sable read these listings' attributes for the check (one step each):\n" + '\n'.join(scan) + '\n'
                        outward.append((call_ref, 'Not run yet. ' + link_series + observed + judgement + '\nCart now: ' + self.basket_prose()))
                        continue
                    if rescan and COST_GENTLE < self.spend_sum:
                        outward.append((call_ref, "Not run yet (no step used). Sable re-read the cart's listings just now:\n" + rescan + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.clock_residual() > WINDUP_SECONDS + 15:
                    if self.residual() > 5:
                        look_over = self.review_pass()
                        if look_over:
                            outward.append((call_ref, 'Not run yet (no step used). ' + look_over))
                            continue
                        if self.guide_refs and self.residual() > 4 and (0 == self.decline_panels):
                            self.decline_panels = 1
                            self.record_trail.append({'kind': 'refuse_blocked', 'leads': list(self.guide_refs)})
                            outward.append((call_ref, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.guide_refs) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                            continue
                        if self.guide_refs and self.residual() > 4 and (1 == self.decline_panels):
                            self.decline_panels = 2
                            earliest = self.guide_refs[0]
                            self.verify_basket([earliest], COST_FIRM)
                            if self.judgements.get(self.verify_field([earliest])) and earliest in self.guide_refs:
                                self.approved = earliest
                                self.record_trail.append({'kind': 'refuse_blocked', 'confirmed': earliest})
                                outward.append((call_ref, f"Not run yet (no step used). An independent check of {earliest}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.exists_decline(label_name, params) and self.residual() <= 5 and (not self.belated_retained) and (self.clock_residual() > WINDUP_SECONDS + 15):
                entry_lines = [iq for iq in self.made_up if not self.out_of_stock_outward(iq)]
                fresher = self.basket_snapshot is not None and self.basket_by <= self.basket_observed_by
                absent_now = [iq for iq in entry_lines if iq not in self.basket_key_ids() or (iq in self.discarded and (not fresher))]
                if len(entry_lines) >= len(self.items) and len(absent_now) + 1 <= self.residual():
                    self.belated_retained = True
                    self.record_trail.append({'kind': 'late_decline_held', 'lines': entry_lines, 'missing': absent_now})
                    retry_pass = 'Check out the cart as it is.' if not absent_now else f"Put {', '.join(absent_now)} back in the cart at the requested quantities, then check out."
                    outward.append((call_ref, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + retry_pass + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.exists_decline(label_name, params) and (not self.guide_refs) and (not self.nudged_on) and (self.residual() > self.upper_moves // 3) and (self.clock_residual() > WINDUP_SECONDS + 15):
                self.nudged_on = True
                outward.append((call_ref, f"Not run yet (no step used). {self.residual()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.exists_decline(label_name, params) and self.outline.get('fallback_buy') is True and self.duties.get('buy'):
                if 2 > self.backstop_retentions and self.residual() > 3 and (self.clock_residual() > WINDUP_SECONDS):
                    self.backstop_retentions += 1
                    self.record_trail.append({'kind': 'fallback_hold', 'n': self.backstop_retentions})
                    outward.append((call_ref, 'Not run yet (no step used). This request wants the closest listing bought when nothing meets every requirement, so ending without a purchase is wrong here: put the candidate that misses the fewest requirements (stock and budget count as requirements; keep within any price margin the request allows) in the cart at the requested quantity, then check out. The request already gives that permission, so do not ask the shopper for it.\nCart now: ' + self.basket_prose()))
                    continue
                if not self.basket_key_ids() and (not self.basket_entry_lines):
                    self.basket_closest_match()
                if self.basket_key_ids() or self.basket_entry_lines:
                    defects = []
                    checkout_params = _SchemaFit._outline_form(dict(self.outline.get('buy_args') or {}), self.blueprints.get(self.duties['buy']), 'args', defects)
                    if not defects and self.exists_checkout(self.duties['buy'], checkout_params):
                        self.record_trail.append({'kind': 'fallback_checkout', 'cart': self.basket_key_ids()})
                        (label_name, params) = (self.duties['buy'], checkout_params)
            if self.exists_decline(label_name, params):
                wrapped = self.finish_off_basket()
                if wrapped:
                    outward.append((call_ref, wrapped + ('' if not self.closed_out else ' [episode finished]')))
                    continue
                params = self.hushed(label_name, params)
            if wind_up and self.residual() >= 2 and (self.clock_residual() > FINAL_CALL_SECONDS + 12):
                (advance, remarked) = self.checkpoint('end' if not self.exists_checkout(label_name, params) else 'order')
                if not advance:
                    outward.append((call_ref, 'Not run yet. ' + remarked + '\nCart now: ' + self.basket_prose()))
                    continue
                if remarked:
                    pared = '\n[Sable] ' + remarked + pared
            comeback = self.perform(label_name, params)
            if comeback is None:
                outward.append((call_ref, 'The call failed (network or server). You may try again.'))
                continue
            viewing = comeback.get('observation')
            if plenty and isinstance(params.get(plenty), list) and (len(params[plenty]) > 1):
                displayed = _Prose._to_text(viewing)
                if any((str(iq) in displayed for iq in params[plenty])):
                    self.set_side_by_approved_one = max(self.set_side_by_approved_one, len(params[plenty]))
                    self.combine_missed = False
                else:
                    if all((str(iq) in self.result_rows for iq in params[plenty])):
                        self.set_side_by_faulty = min(self.set_side_by_faulty or 999, len(params[plenty]))
                    self.combine_missed = True
                    pared += f'\n[Sable] The side-by-side view showed nothing for {len(params[plenty])} listings: send at most {self.set_side_by_space()} per call.'
            self.newest = (self.newest + [stamp])[-3:]
            self.key_id_params.update(((label_name, kq) for (kq, vq) in params.items() if ([vq] if not isinstance(vq, list) else vq) and all((isinstance(xq, str) and xq in self.result_rows for xq in ([vq] if not isinstance(vq, list) else vq)))))
            self.take_in(viewing)
            if label_name == self.duties.get('compare'):
                result_rows = self.log_entries_within(viewing)
                if result_rows:
                    self.traits_out_of[label_name] = self.traits_out_of.get(label_name, False) or any((_RecordScan._trait_panel(rq) for (_, rq) in result_rows))
            if label_name in scanners - {self.duties.get('search')}:
                self.remark_snapshot(label_name, viewing)
            unadorned = {instrument for (assignment, instrument) in self.duties.items() if 'search' != assignment}
            sheeted = '' if not (label_name not in unadorned and (not wind_up)) else self.further_sheets(label_name, params, viewing)
            if self.lone_objective(label_name) and label_name not in (self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse'), self.duties.get('ask')):
                offer_rec = None if not isinstance(viewing, dict) else self.key_ident(viewing)
                if offer_rec:
                    self.traits_out_of[label_name] = self.traits_out_of.get(label_name, False) or bool(_RecordScan._trait_panel(viewing))
                    if label_name == self.duties.get('open') and (not self.traits_out_of[label_name]):
                        alternative = next((tq for tq in self.blueprints if label_name != tq and self.lone_objective(tq) and (self.traits_out_of.get(tq) is not False) and (tq not in (self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse'), self.duties.get('ask'), self.duties.get('search')))), None)
                        if alternative:
                            self.duties['open'] = alternative
                    if not self.traits_out_of[label_name] and label_name != self.duties.get('open') and (not self.duties.get('offer')):
                        self.duties['offer'] = label_name
                    if label_name == self.deal_instrument():
                        self.deal_observed[offer_rec] = self.move
            if label_name == self.duties.get('cart'):
                self.enter_in_basket(label_name, params, viewing)
                self.basket_remarked = _Prose._to_text(viewing)
                self.newest = []
                tallies = [_SchemaFit._tally_of(vq) for vq in params.values() if not isinstance(vq, str) and _SchemaFit._tally_of(vq) is not None]
                for val_text in params.values():
                    if tallies and isinstance(val_text, str) and (val_text in self.result_rows):
                        if tallies[0] <= 0 and (not _RecordScan._within_result_rows(viewing, val_text)):
                            self.discarded.add(val_text)
                        elif tallies[0] > 0:
                            self.discarded.discard(val_text)
                entry_lines = [iq for iq in self.basket_key_ids() if not self.out_of_stock_outward(iq) and iq not in self.discarded]
                if len(entry_lines) >= len(self.made_up):
                    self.made_up = entry_lines
            if label_name == self.duties.get('cart_view'):
                self.basket_snapshot = viewing
                self.basket_observed_by = self.move
            if self.exists_checkout(label_name, params):
                self.checkouts += 1
                if not comeback.get('finished'):
                    self.declined_by = self.move
                    self.reset_caps()
            elif self.exists_decline(label_name, params) and (not comeback.get('finished')):
                self.reset_caps()
            prose = self.display(viewing) + sheeted + pared
            if label_name not in unadorned and (not wind_up):
                prose += self.least_costly_at_hand([viewing] + self.sheeted_seen_text)
            if label_name == self.duties.get('search') and (not wind_up):
                prose += self.repeat_hitting_result_rows(params, viewing)
            groundwork = prose
            backing = self.grounds(viewing)
            if backing:
                prose += '\n' + backing
            if label_name == self.duties.get('cart') and any((_SchemaFit._tally_of(vq) == 0 for vq in params.values() if not isinstance(vq, str))):
                if any((isinstance(vq, str) and vq in self.result_rows and _RecordScan._within_result_rows(viewing, vq) for vq in params.values())):
                    prose += '\n[Sable] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.remark_out_of_stock = False
            entry_lines = self.free_floating_entry_lines(viewing)
            if entry_lines:
                gist = self.accept_remark(entry_lines)
                if gist:
                    self.bundle_remark = index < len(invokes) - 1
                    prose += '\n[note read] ' + gist + '. Apply this change to your picks before checking out.'
            if self.exists_checkout(label_name, params) and (not comeback.get('finished')) and (self.remark_out_of_stock or any((entry_line in self.uttered_text(viewing) for entry_line in self.out_of_stock_entry_lines))):
                observed = self.locate_out_of_stock()
                if observed:
                    prose += "\n[Sable re-read the cart's lines after the refusal]\n" + observed
            interchange = self.interchange_clue()
            if interchange:
                prose += '\n' + interchange
            if self.backlogged:
                prose += "\n[notes seen on Sable's own calls] " + ' | '.join(self.backlogged)
                self.backlogged = []
            if label_name in scanners:
                self.scan_keep_store[stamp] = (self.revision, self.move, groundwork)
            if comeback.get('finished'):
                prose += ' [episode finished]'
            outward.append((call_ref, prose))
        return outward

class _Closing:

    def close_out(self, anew: bool=False) -> str:
        if self.closed_out or self.went_broke or self.residual() <= 0:
            return ''
        (checkout, decline) = (self.duties.get('buy'), self.duties.get('refuse'))
        fresher = self.basket_snapshot is not None and self.basket_by <= self.basket_observed_by
        retaining = bool(self.basket_entry_lines) if not fresher else bool(self.basket_key_ids())
        if checkout and (not retaining) and (not self.basket_altered_cart) and (self.residual() >= 3):
            retaining = self.basket_closest_match()
        lacking = [] if not (len(self.items) >= 2 and self.outline.get('fallback_buy') is not True) else self.absent_now_items()
        if lacking:
            self.record_trail.append({'kind': 'partial_forced_refuse', 'missing': lacking})
        if checkout and (not lacking) and (retaining or (self.basket_altered_cart and (not fresher) and (self.residual() >= 2))):
            defects: list[str] = []
            params = _SchemaFit._outline_form(dict(self.outline.get('buy_args') or {}), self.blueprints.get(checkout), 'args', defects)
            if not defects:
                if self.residual() >= 2 and self.clock_residual() > FINAL_CALL_SECONDS + 12:
                    self.checkpoint('order')
                if self.closed_out or self.residual() <= 0:
                    return ''
                comeback = self.perform(checkout, params)
                self.record_trail.append({'kind': 'forced_buy', 'reply': _Prose._cut_down(comeback, 200)})
                if self.closed_out or self.residual() <= 0:
                    return ''
                if comeback is not None and anew:
                    self.checkouts += 1
                    self.declined_by = self.move
                    self.view_out(checkout, comeback.get('observation'))
                    return 'the checkout was refused: ' + self.display(comeback.get('observation'))
        if decline:
            defects = []
            params = _SchemaFit._outline_form(self.hushed(decline, dict(self.outline.get('refuse_args') or {})), self.blueprints.get(decline), 'args', defects)
            if not defects:
                if self.residual() >= 2 and self.clock_residual() > FINAL_CALL_SECONDS + 12:
                    self.checkpoint('end')
                if self.closed_out or self.residual() <= 0:
                    return ''
                comeback = self.perform(decline, params)
                self.record_trail.append({'kind': 'forced_refuse', 'reply': _Prose._cut_down(comeback, 200)})
                if comeback is not None and (not self.closed_out):
                    self.view_out(decline, comeback.get('observation'))
                    return 'ending without a purchase was refused: ' + self.display(comeback.get('observation'))
        return ''

    def basket_closest_match(self) -> bool:
        basket = self.duties.get('cart')
        entry_lines = {} if not basket else self.closest_match_entry_lines()
        invokes = None if not entry_lines else self.basket_invokes(entry_lines, [])
        if not invokes or len(invokes) + 1 > self.residual():
            return False
        for params in invokes:
            defects: list[str] = []
            params = _SchemaFit._outline_form(params, self.blueprints.get(basket), 'args', defects)
            if defects:
                return False
            comeback = self.perform(basket, params)
            if comeback is None or self.closed_out:
                return False
            self.enter_in_basket(basket, params, comeback.get('observation'))
            self.view_out(basket, comeback.get('observation'))
        self.record_trail.append({'kind': 'nearest_carted', 'lines': entry_lines})
        return bool(self.basket_entry_lines)

    def closest_match_entry_lines(self) -> dict[str, int]:
        if self.outline.get('fallback_buy') is not True or not self.items:
            return {}
        catgroup_field = self.section_field()
        each_unit_entry_line = self.outline.get('budget_scope') == 'each'
        ceiling = None if not (isinstance(self.money_limit, (int, float)) and (not isinstance(self.money_limit, bool)) and (0 < self.money_limit)) else self.money_limit
        entry_lines: dict[str, int] = {}
        aggregate = 0.0
        for good in self.items:
            aisle_shelf_wish_for = (good.aisle_shelf or '').strip().lower()
            headword = {wq for wq in re.findall('[a-z0-9]{4,}', good.label_name.lower())}
            values = [re.findall('[a-z0-9]{3,}', mq.split(':', 1)[-1].lower()) for mq in good.demands + good.informed if mq not in good.discarded]
            graded = []
            for key_ident in dict.fromkeys(self.guide_refs + list(self.entire_set)):
                cost_of = self.cost_of_belonging(key_ident)
                if key_ident in entry_lines or not self.for_sale(key_ident) or cost_of is None or (ceiling and cost_of * max(1, good.units) > ceiling):
                    continue
                aisle_shelf = '' if not catgroup_field else str((self.result_rows.get(key_ident) or {}).get(catgroup_field) or '').strip().lower()
                if aisle_shelf_wish_for and catgroup_field and (aisle_shelf not in ('', aisle_shelf_wish_for)):
                    continue
                prose = _Prose._to_text(self.entire_set.get(key_ident) or self.result_rows.get(key_ident)).lower()
                if len(self.items) > 1 and (not aisle_shelf_wish_for) and headword and (not any((wq in (aisle_shelf or prose) for wq in headword))):
                    continue
                displayed = sum((1 for wordings in values if wordings and all((wq in prose for wq in wordings))))
                graded.append((displayed + (0 if not key_ident in self.guide_refs else 100), cost_of, key_ident))
            if not graded:
                continue
            top_pick = min(graded, key=lambda rq: (-rq[0], rq[1]))
            duplicates = max(1, good.units)
            if ceiling and (not each_unit_entry_line) and (aggregate + top_pick[1] * duplicates > ceiling):
                continue
            entry_lines[top_pick[2]] = duplicates
            aggregate += top_pick[1] * duplicates
        return entry_lines

    def finish_off_basket(self) -> str:
        (basket, checkout) = (self.duties.get('cart'), self.duties.get('buy'))
        if self.finished_off or not self.approved or self.out_of_stock_outward(self.approved) or (not basket) or (not checkout):
            return ''
        self.finished_off = True
        catgroup_field = self.section_field()

        def variety(key_ident: str) -> str:
            return key_ident if not catgroup_field else str((self.result_rows.get(key_ident) or {}).get(catgroup_field) or '').strip().lower()
        scheme: dict[str, int] = {}
        for (numeral, key_ids) in self.guide_ref_items.items():
            index = _SchemaFit._tally_of(numeral)
            pick_out = next((iq for iq in key_ids if not self.out_of_stock_outward(iq)), None)
            if pick_out and index and (1 <= index <= len(self.items)) and (variety(pick_out) not in {variety(pq) for pq in scheme}):
                scheme[pick_out] = self.items[index - 1].units
        if self.approved not in scheme:
            return ''
        currently = self.basket_key_ids()
        preserve = {iq: self.basket_entry_lines.get(iq, 1) for iq in currently if iq not in scheme and (not self.out_of_stock_outward(iq)) and (variety(iq) not in {variety(pq) for pq in scheme})}
        let_fall = [iq for iq in currently if iq not in scheme and iq not in preserve]
        if len(scheme) + len({variety(iq) for iq in preserve}) < len(self.items):
            return ''
        entry_lines = {**preserve, **scheme}
        invokes = self.basket_invokes(entry_lines, let_fall)
        if not invokes or len(invokes) + 1 > self.residual():
            return ''
        self.top_up_traits(list(entry_lines), len(invokes) + 3)
        self.verify_basket(list(entry_lines), COST_FIRM)
        if not self.judgements.get(self.verify_field(list(entry_lines))):
            return ''
        wrapped = []
        for params in invokes:
            defects: list[str] = []
            params = _SchemaFit._outline_form(params, self.blueprints.get(basket), 'args', defects)
            if defects:
                return ''
            comeback = self.perform(basket, params)
            if comeback is None or self.closed_out:
                return ''
            self.enter_in_basket(basket, params, comeback.get('observation'))
            self.view_out(basket, comeback.get('observation'))
            wrapped.append(_Prose._to_text(params))
        defects = []
        params = _SchemaFit._outline_form(dict(self.outline.get('buy_args') or {}), self.blueprints.get(checkout), 'args', defects)
        if defects:
            return ''
        comebacks = []
        for go_at in range(2):
            if go_at:
                self.rescan_deals()
                self.verify_basket(list(entry_lines), COST_FIRM)
                if not self.judgements.get(self.verify_field(list(entry_lines))) or self.residual() < 1 or self.closed_out or any((self.out_of_stock_outward(iq) for iq in entry_lines)):
                    break
            (advance, remarked) = self.checkpoint('order')
            if remarked:
                comebacks.append(remarked)
            if not advance:
                break
            comeback = self.perform(checkout, params)
            self.record_trail.append({'kind': 'completed', 'lines': entry_lines, 'drop': let_fall, 'reply': _Prose._cut_down(comeback, 200)})
            if comeback is None:
                break
            self.checkouts += 1
            if not comeback.get('finished'):
                self.declined_by = self.move
            self.view_out(checkout, comeback.get('observation'))
            comebacks.append(self.display(comeback.get('observation')))
            if self.closed_out or comeback.get('finished'):
                break
        if not comebacks:
            return ''
        return 'Not run (Sable acted instead): the independent check confirmed ' + self.approved + ', so Sable set the cart (' + '; '.join(wrapped) + ') and checked out. Store reply: ' + ' | then: '.join(comebacks)

    def absent_now_items(self) -> list[str]:
        key_ids = self.basket_key_ids()
        if not key_ids:
            return []
        covered_set: set[int] = set()
        for key_ident in key_ids:
            values = [vq for vq in (self.result_rows.get(key_ident) or {}).values() if isinstance(vq, str) and 0 < len(vq) <= 60]
            match_many = {nq for (nq, good) in enumerate(self.items) if any((self.heading_suit(vq, good) >= 2 for vq in values))}
            if not match_many:
                return []
            covered_set |= match_many
        return [good.label_name for (nq, good) in enumerate(self.items) if good.aisle_shelf and nq not in covered_set]

class _Review:

    def review_pass_suits(self, obtained: dict[str, Any] | None) -> dict[str, list[str]]:
        suits = (obtained or {}).get('fits')
        if not isinstance(suits, dict):
            return {}
        labelled = {str(kq): [str(iq) for iq in _SchemaFit._entries_list(vq) if str(iq) in self.result_rows and (not self.out_of_stock_outward(str(iq)))][:4] for (kq, vq) in suits.items()}
        return {kq: vq for (kq, vq) in labelled.items() if vq}

    def review_pass_invoke(self) -> dict[str, Any] | None:
        log_entries = []
        for key_ident in reversed(self.observed_buy_order):
            if self.out_of_stock_outward(key_ident):
                continue
            log_entry = None if not key_ident in self.entire_set else self.entire_set.get(key_ident)
            log_entries.append('ROW ' + _Prose._to_text(_Prose._cut_down(self.result_rows.get(key_ident), 120))[:500] if not log_entry else 'OPENED ' + _Prose._to_text(_Prose._cut_down(log_entry, 200))[:1200])
            if len(log_entries) >= 30:
                break
        if not log_entries:
            return None
        cue = NOFIT_SPEC + '\n\nREQUEST:\n' + self.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.account()
        cue += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(log_entries)
        obtained = self.inquire_json(cue, self.verifier(), 1500)
        self.record_trail.append({'kind': 'audit', 'got': obtained})
        return obtained

    def review_pass(self) -> str:
        if self.reviewed or COST_GENTLE < self.spend_sum:
            return ''
        self.reviewed = True
        obtained = self.review_pass_invoke()
        labelled = self.review_pass_suits(obtained)
        if not labelled:
            result_rows = [str(iq) for iq in _SchemaFit._entries_list((obtained or {}).get('open')) if str(iq) in self.result_rows and str(iq) not in self.expanded and (not self.out_of_stock_outward(str(iq)))]
            if result_rows and self.expand_result_rows(result_rows):
                obtained = self.review_pass_invoke()
                labelled = self.review_pass_suits(obtained)
        if not labelled:
            return ''
        self.guide_ref_items = labelled
        self.guide_refs = [iq for key_ids in labelled.values() for iq in key_ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {kq}: {', '.join(vq)}" for (kq, vq) in labelled.items())) + ('' if not obtained.get('reason') else f" ({obtained.get('reason')})") + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

class _StockOuts:

    def standins(self, gone_missing: list[str]) -> list[str]:
        if COST_GENTLE < self.spend_sum:
            return []
        log_entries = []
        for key_ident in reversed(self.observed_buy_order):
            if key_ident in gone_missing or key_ident in self.vanished or self.out_of_stock_outward(key_ident):
                continue
            log_entries.append(_Prose._to_text(_Prose._cut_down(self.entire_set.get(key_ident) or self.result_rows.get(key_ident), 200))[:1200])
            if len(log_entries) >= 25:
                break
        if not log_entries:
            return []
        cue = STANDIN_SPEC + '\n\nREQUEST:\n' + self.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.account()
        cue += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{iq}: {self.log_entry_belonging(iq)}' for iq in gone_missing)) + '\n\nCART NOW:\n' + self.basket_prose()
        cue += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(log_entries)
        obtained = self.inquire_json(cue, self.verifier(), 1200)
        suits: list[tuple[float, str]] = []
        for record_item in _SchemaFit._entries_list((obtained or {}).get('fits')):
            key_ident = str(record_item if not isinstance(record_item, dict) else record_item.get('id'))
            cost_of = None if not isinstance(record_item, dict) else record_item.get('price')
            if key_ident in self.result_rows and key_ident not in gone_missing and (key_ident not in self.vanished) and (key_ident not in self.basket_key_ids()) and (key_ident not in [iq for (_, iq) in suits]):
                suits.append((float('inf') if not (isinstance(cost_of, (int, float)) and (not isinstance(cost_of, bool))) else float(cost_of), key_ident))
        graded = [iq for (_, iq) in suits] if not self.outline.get('pick') == 'cheapest' else [iq for (_, iq) in sorted(suits, key=lambda fq: fq[0])]
        self.record_trail.append({'kind': 'swap', 'lost': gone_missing, 'fits': suits})
        return graded[:4]

    def locate_out_of_stock(self) -> str:
        key_ids = [iq for iq in self.basket_key_ids() if iq not in self.vanished]
        if any((self.out_of_stock_outward(iq) for iq in key_ids)):
            return ''
        if len(key_ids) == 1:
            self.unbuyable.add(key_ids[0])
            return ''
        return '' if not key_ids else self.rescan_deals()

    def interchange_clue(self) -> str:
        gone_missing = [iq for iq in self.basket_key_ids() if iq not in self.vanished and self.out_of_stock_outward(iq)]
        if not gone_missing:
            return ''
        for key_ident in gone_missing:
            self.vanished[key_ident] = self.move
        self.guide_refs = [iq for iq in self.guide_refs if iq not in gone_missing]
        self.revision += 1
        picks_out = self.standins(gone_missing)
        prose = f"[Sable] Sold out now: {', '.join(gone_missing)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks_out and self.residual() <= 6:
            return prose + f" Few steps are left: in this turn add {picks_out[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks_out[1:]) or 'none'}."
        if picks_out and self.outline.get('pick') == 'cheapest':
            return prose + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks_out)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks_out[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks_out:
            return prose + f" Listings already read that appear to fit that item instead: {', '.join(picks_out)}. In this turn take the sold-out line out of the cart and add {picks_out[0]} at the requested quantity, then check out."
        return prose + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def out_of_stock_outward(self, key_ident: str) -> bool:
        return key_ident in self.unbuyable or key_ident in self.vanished

    def rescan_deals(self) -> str:
        instrument = self.deal_instrument()
        objective = self.lone_objective(instrument)
        if not instrument or not objective:
            return ''
        because = max(self.move - 2, self.declined_by)
        outdated = [iq for iq in self.basket_key_ids() if self.deal_observed.get(iq, -99) < because and iq not in self.vanished]
        if not outdated or self.residual() < len(outdated) + 3:
            return ''
        entry_lines = []
        for key_ident in outdated:
            defects: list[str] = []
            params = _SchemaFit._outline_form({objective: key_ident}, self.blueprints.get(instrument), 'args', defects)
            if defects:
                break
            ahead_of = dict(self.result_rows.get(key_ident) or {})
            comeback = self.perform(instrument, params)
            if comeback is None or self.closed_out:
                break
            viewing = comeback.get('observation')
            self.deal_observed[key_ident] = self.move
            self.view_out(instrument, viewing)
            once = self.result_rows.get(key_ident) or {}
            shifted = [kq for (kq, vq) in once.items() if kq in ahead_of and ahead_of[kq] != vq and (not isinstance(vq, (dict, list)))]
            if shifted:
                entry_lines.append(f"{key_ident} changed ({', '.join(shifted[:4])}): {_Prose._to_text(_Prose._cut_down(viewing, 300))[:900]}")
        self.record_trail.append({'kind': 'reread', 'ids': outdated, 'changed': len(entry_lines)})
        if entry_lines:
            self.revision += 1
        return '\n'.join(entry_lines)

class _Pointers:

    def pursuit(self) -> tuple[str, bool]:
        active = {requirement for (_, requirement) in self.requirements_currently()}
        waiting = [requirement for (_, requirement) in self.pointer_refs if requirement in active and requirement not in self.pointer_ref_values]
        instrument = self.duties.get('offer') or self.duties.get('open')
        objective = self.lone_objective(instrument)
        observed: list[str] = []
        gone_missing: list[str] = []
        while waiting and 6 > self.chain_set and (COST_FIRM > self.spend_sum) and (self.clock_residual() > WINDUP_SECONDS):
            self.chain_set += 1
            log_entries = []
            for key_ident in reversed(self.observed_buy_order):
                log_entry = None if not key_ident in self.expanded else self.entire_set.get(key_ident) or self.result_rows.get(key_ident)
                if not log_entry:
                    log_entries.append('ROW ' + _Prose._to_text(_Prose._cut_down(self.result_rows.get(key_ident), 80))[:300])
                else:
                    log_entries.append('OPENED ' + _Prose._to_text(_Prose._cut_down(log_entry, 300))[:1400])
                if len(log_entries) >= 60:
                    break
            cue = CHAIN_SPEC + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(waiting)
            cue += '\n\nRECORDS (newest first):\n' + '\n'.join(log_entries)
            obtained = self.inquire_json(cue, [SWIFT_MODEL, DEEP_MODEL], 1800) or {}
            self.record_trail.append({'kind': 'trail', 'pending': waiting, 'got': _Prose._cut_down(obtained, 150)})
            extent: list[str] = []
            gone_missing = []
            for link_series in _SchemaFit._entries_list(obtained.get('chains')):
                if not isinstance(link_series, dict):
                    continue
                requirement = next((pq for pq in waiting if pq == str(link_series.get('need') or '').strip()), None) or (None if not len(waiting) == 1 else waiting[0])
                if not requirement:
                    continue
                chain_links = [lq for lq in _SchemaFit._entries_list(link_series.get('links')) if isinstance(lq, dict)]
                key_ids = [str(lq.get('id') or '').strip() for lq in chain_links]
                if link_series.get('done') is True and chain_links and all((iq in self.expanded for iq in key_ids)) and str(link_series.get('value') or '').strip():
                    self.pointer_ref_values[requirement] = str(link_series['value']).strip()
                    if str(link_series.get('rule_value') or '').strip():
                        self.pointer_ref_directives[requirement] = str(link_series['rule_value']).strip()
                    self.revision += 1
                    continue
                sealed = [iq for iq in key_ids if iq in self.result_rows and iq not in self.expanded]
                if not sealed:
                    regarding = next((str(lq.get('about') or '') for (lq, iq) in zip(chain_links, key_ids) if iq not in self.result_rows), '') or 'the listing it points at'
                    gone_missing.append(f'{requirement} -> find {regarding}')
                else:
                    extent += [iq for iq in sealed if iq not in extent]
            waiting = [pq for pq in waiting if pq not in self.pointer_ref_values]
            if not extent or not instrument or (not objective) or (self.residual() < len(extent) + 4) or self.closed_out:
                break
            for key_ident in extent[:max(0, min(3, 6 - len(observed)))]:
                defects: list[str] = []
                params = _SchemaFit._outline_form({objective: key_ident}, self.blueprints.get(instrument), 'args', defects)
                if defects:
                    break
                comeback = self.perform(instrument, params)
                if comeback is None or self.closed_out:
                    break
                self.view_out(instrument, comeback.get('observation'))
                observed.append(f'{key_ident}: ' + self.display(comeback.get('observation'))[:1500])
        prose = ''
        if observed:
            prose += '[Sable opened the pages a requirement points at]\n' + '\n'.join(observed) + '\n'
        if self.pointer_ref_values:
            prose += '[values read from the pages requirements point at] ' + '; '.join((f'{kq} -> {vq}' for (kq, vq) in self.pointer_ref_values.items())) + '\n'
        retain = bool(gone_missing) and 3 > self.pursuit_panels and (self.residual() > 5) and (self.clock_residual() > WINDUP_SECONDS)
        if retain:
            self.pursuit_panels += 1
            prose += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(gone_missing[:3]) + '. Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.'
        return (prose, retain)

class _Verifying:

    def verify_basket(self, key_ids: list[str] | None=None, ceiling_num: float=COST_GENTLE) -> str:
        guide_ref = key_ids is not None
        key_ids = self.basket_key_ids() if not guide_ref else key_ids
        field = self.verify_field(key_ids)
        if field in self.verified or ceiling_num < self.spend_sum or (not key_ids):
            return ''
        self.verified.add(field)
        cue = VERIFY_SPEC + '\n\nREQUEST:\n' + self.ask_text + '\n\nREQUIREMENTS IN FORCE:\n' + self.account()
        if not guide_ref:
            cue += '\n\nCART:\n' + self.basket_prose()
        else:
            amount = '; '.join((f'item {nq}: quantity {this.units}' for (nq, this) in enumerate(self.items, 1)))
            cue += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + amount + '):\n' + ', '.join(key_ids)
        cue += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{iq}: ' + ('' if not self.out_of_stock_outward(iq) else 'SOLD OUT (the store says it can no longer be bought). ') + self.log_entry_belonging(iq) for iq in key_ids))
        aisles = self.aisle_entry_lines(key_ids)
        if aisles:
            cue += "\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + aisles
        if self.outline.get('pick') in ('cheapest', 'most', 'least'):
            cue += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.alternatives(key_ids, 15)
        obtained = self.inquire_json(cue, self.verifier(), 1500)
        self.record_trail.append({'kind': 'check', 'cart': key_ids, 'got': obtained})
        self.judgements[field] = bool(obtained) and obtained.get('ok') is not False
        if not obtained or obtained.get('ok') is not False:
            return ''
        self.guide_refs = [iq for iq in self.guide_refs if iq not in key_ids]
        issues = [str(pq) for pq in _SchemaFit._entries_list(obtained.get('problems'))][:5]
        preferable = [str(bq) for bq in _SchemaFit._entries_list(obtained.get('better')) if str(bq) in self.result_rows and (not self.out_of_stock_outward(str(bq)))][:4]
        prose = 'A careful review says this cart is not right yet: ' + '; '.join(issues or ['see the requirements'])
        if preferable:
            prose += '. Listings that look better: ' + ', '.join(preferable)
        return prose + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def verify_field(self, key_ids: list[str]) -> str:
        return '|'.join(sorted(key_ids)) + f'#{self.revision}'

class _CartBook:

    def settle_on_key_ids(self, label_name: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if label_name not in {self.duties.get(rq) for rq in ('cart', 'open', 'offer', 'compare')} - {None} or not self.key_id_field:
            return (params, '')
        features = (self.blueprints.get(label_name) or {}).get('properties') or {}
        unused = [kq for (kq, vq) in features.items() if isinstance(vq, dict) and ('string' in _SchemaFit._sorts(vq) or 'array' in _SchemaFit._sorts(vq)) and ('enum' not in vq)]
        if len(unused) != 1 or (label_name != self.duties.get('cart') and (label_name, unused[0]) not in self.key_id_params):
            return (params, '')

        def repair(val_text: Any) -> tuple[Any, str]:
            if isinstance(val_text, str) and val_text.strip() in self.result_rows:
                return (val_text.strip(), '')
            if not isinstance(val_text, str) or _RecordScan._key_id_alike(val_text, True) or len(val_text.strip()) < 8:
                return (val_text, '')
            minor = val_text.strip().lower()
            match_many = [iq for (iq, result_row) in self.result_rows.items() if any((self.key_id_field != kq and isinstance(vq, str) and vq.strip().lower().startswith(minor) for (kq, vq) in result_row.items()))]
            if len(match_many) == 1:
                return (match_many[0], '')
            if match_many:
                return (val_text, f"several listings carry the text {val_text!r} ({', '.join(match_many[:6])}); pass one listing id")
            return (val_text, f'{val_text!r} is not a listing id; pass the id from the result rows (their {self.key_id_field} field)')
        outward = dict(params)
        for field in unused:
            if field not in params:
                continue
            repaired = [repair(params[field])] if not isinstance(params[field], list) else [repair(vq) for vq in params[field]]
            issue = next((pq for (_, pq) in repaired if pq), '')
            if issue:
                return (params, issue)
            outward[field] = repaired[0][0] if not isinstance(params[field], list) else [vq for (vq, _) in repaired]
        return (outward, '')

    def enter_in_basket(self, label_name: str, params: dict[str, Any], viewing: Any) -> None:
        self.basket_altered_cart = True
        self.basket_by = self.move
        situation = self.align_basket(viewing, label_name, params)
        if 'synced' == situation:
            return
        if 'none' == situation and self.basket_displays_entry_lines:
            self.record_trail.append({'kind': 'cart_unchanged', 'args': _Prose._cut_down(params, 60)})
            return
        self.remark_basket(label_name, params)

    def align_basket(self, viewing: Any, label_name: str | None=None, params: dict[str, Any] | None=None) -> str:
        rosters: list[list[Any]] = []

        def traverse(element: Any, recursion: int) -> None:
            if 3 < recursion:
                return
            if isinstance(element, list):
                if all((isinstance(rq, dict) for rq in element)):
                    rosters.append(element)
                return
            if isinstance(element, dict):
                for val_text in element.values():
                    traverse(val_text, recursion + 1)
        if not isinstance(viewing, dict) or self.key_ident(viewing):
            return 'none'
        traverse(viewing, 0)
        altered_cart = [vq for val_text in (params or {}).values() for vq in ([val_text] if not isinstance(val_text, list) else val_text) if isinstance(vq, str) and vq in self.result_rows]
        alongside_key_ids = [result_rows for result_rows in rosters if result_rows and any((self.key_ident(rq, True) for rq in result_rows))]
        result_rows = next((rq for rq in alongside_key_ids if any((self.key_ident(xq, True) in altered_cart for xq in rq))), None if not alongside_key_ids else alongside_key_ids[0])
        if result_rows is None:
            if not (len(rosters) == 1 and (not rosters[0])):
                return 'none'
            else:
                result_rows = rosters[0]
        key_idents = [self.key_ident(rq, True) for rq in result_rows]
        if not all(key_idents):
            return 'unreadable'
        field = self.count_field
        if not field and result_rows:
            keys = [kq for kq in result_rows[0] if self.cost_of_field != kq and all((isinstance(rq.get(kq), int) and (not isinstance(rq.get(kq), bool)) and (0 <= rq[kq] <= 99) for rq in result_rows))]
            if len(keys) > 1 and label_name and (params is not None):
                spared = dict(self.basket_entry_lines)
                self.remark_basket(label_name, params)
                anticipated = dict(self.basket_entry_lines)
                self.basket_entry_lines = spared
                keys = [kq for kq in keys if all((rq.get(kq) == anticipated.get(iq, 0) for (rq, iq) in zip(result_rows, key_idents) if iq in altered_cart))]
            if len(keys) != 1:
                return 'unreadable'
            field = keys[0]
            self.count_field = field
        outward: dict[str, int] = {}
        for (result_row, key_ident) in zip(result_rows, key_idents):
            count = None if not field else result_row.get(field)
            if not isinstance(count, int) or isinstance(count, bool):
                return 'unreadable'
            if 0 < count:
                outward[key_ident] = count
        self.basket_entry_lines = outward
        self.basket_snapshot = viewing
        self.basket_observed_by = self.move
        self.basket_displays_entry_lines = True
        return 'synced'

    def remark_basket(self, label_name: str, params: dict[str, Any]) -> None:
        self.basket_altered_cart = True
        self.basket_by = self.move
        specification = self.blueprints.get(label_name, {}).get('properties') or {}
        deletion = False
        for (field, val_text) in params.items():
            alts = (specification.get(field) or {}).get('enum')
            if isinstance(alts, list) and len(alts) >= 2 and (val_text in alts[1:]):
                deletion = True
        free_floating_count = next((_SchemaFit._tally_of(vq) for vq in params.values() if _SchemaFit._tally_of(vq) is not None and (not isinstance(vq, str))), None)
        entry_lines: list[tuple[str, int | None]] = []
        entirety = False
        for (field, val_text) in params.items():
            if isinstance(val_text, list):
                outline_form = (specification.get(field) or {}).get('items') or {}
                if not val_text and isinstance(outline_form, dict) and (outline_form.get('type') == 'object' or 'properties' in outline_form):
                    entirety = True
                for entry_line in val_text:
                    if isinstance(entry_line, dict):
                        entirety = True
                        key_ident = next((vq for vq in entry_line.values() if isinstance(vq, str) and vq in self.result_rows), None)
                        count = next((_SchemaFit._tally_of(vq) for vq in entry_line.values() if _SchemaFit._tally_of(vq) is not None and (not isinstance(vq, str))), 1)
                        if key_ident:
                            entry_lines.append((key_ident, count))
                    elif isinstance(entry_line, str) and entry_line in self.result_rows:
                        entry_lines.append((entry_line, free_floating_count))
            elif isinstance(val_text, str) and val_text in self.result_rows:
                entry_lines.append((val_text, free_floating_count))
        if entirety:
            self.basket_entry_lines = {iq: cq for (iq, cq) in entry_lines if cq and 0 < cq}
            return
        assignments = self.basket_assignments(label_name)
        for (key_ident, count) in entry_lines:
            carry = self.basket_entry_lines.get(key_ident, 0)
            if deletion:
                carry = carry - (1 if not count else count)
            elif not assignments:
                carry = carry + (1 if not count is not None else count)
            else:
                carry = 1 if not count is not None else count
            if not 0 < carry:
                self.basket_entry_lines.pop(key_ident, None)
            else:
                self.basket_entry_lines[key_ident] = carry

    def basket_assignments(self, label_name: str) -> bool:
        features = self.blueprints.get(label_name, {}).get('properties') or {}
        if any((isinstance(sq, dict) and isinstance(sq.get('enum'), list) and (len(sq['enum']) >= 2) for sq in features.values())):
            return False
        return any((isinstance(sq, dict) and 'integer' in _SchemaFit._sorts(sq) and (sq.get('minimum') == 0) for sq in features.values()))

    def basket_invokes(self, entry_lines: dict[str, int], let_fall: list[str]) -> list[dict[str, Any]] | None:
        label_name = self.duties.get('cart') or ''
        blueprint = self.blueprints.get(label_name) or {}
        features = blueprint.get('properties') or {}
        for (field, specification) in features.items():
            outline_form = None if not (isinstance(specification, dict) and 'array' in _SchemaFit._sorts(specification)) else specification.get('items')
            if isinstance(outline_form, dict) and isinstance(outline_form.get('properties'), dict):
                interior = outline_form['properties']
                key_id_field = next((kq for (kq, sq) in interior.items() if isinstance(sq, dict) and 'string' in _SchemaFit._sorts(sq) and ('enum' not in sq)), None)
                count_field = next((kq for (kq, sq) in interior.items() if isinstance(sq, dict) and 'integer' in _SchemaFit._sorts(sq)), None)
                if not key_id_field:
                    return None
                return [{field: [{key_id_field: iq, **({} if not count_field else {count_field: nq})} for (iq, nq) in entry_lines.items()]}]
        proses = [kq for (kq, sq) in features.items() if isinstance(sq, dict) and 'string' in _SchemaFit._sorts(sq) and ('enum' not in sq)]
        objective = next((kq for (tq, kq) in self.key_id_params if label_name == tq and kq in proses), None) or next((kq for kq in blueprint.get('required') or [] if kq in proses), None) or (None if not len(proses) == 1 else proses[0])
        count = next((kq for (kq, sq) in features.items() if isinstance(sq, dict) and 'integer' in _SchemaFit._sorts(sq)), None)
        manner = next((kq for (kq, sq) in features.items() if isinstance(sq, dict) and isinstance(sq.get('enum'), list) and (len(sq['enum']) >= 2)), None)
        if not objective:
            return None
        invokes: list[dict[str, Any]] = []
        for key_ident in let_fall:
            carry = self.basket_entry_lines.get(key_ident, 1)
            if manner:
                invokes.append({objective: key_ident, manner: features[manner]['enum'][1], **({} if not count else {count: carry})})
            elif count and self.basket_assignments(label_name):
                invokes.append({objective: key_ident, count: 0})
            elif not (count and isinstance(features[count].get('minimum'), int) and (features[count]['minimum'] <= -carry)):
                return None
            else:
                invokes.append({objective: key_ident, count: -carry})
        for (key_ident, wish_for) in entry_lines.items():
            carry = 0 if not key_ident in self.basket_key_ids() else self.basket_entry_lines.get(key_ident, 0)
            if wish_for == carry:
                continue
            if carry and (not self.basket_assignments(label_name)):
                return None
            invokes.append({objective: key_ident, **({} if not manner else {manner: features[manner]['enum'][0]}), **({} if not count else {count: wish_for})})
        return invokes

    def alternatives(self, pass_over: list[str], ceiling_num: int=30) -> str:
        entry_lines = []
        buy_order = list(reversed(self.observed_buy_order))
        if self.outline.get('pick') == 'cheapest':
            buy_order.sort(key=lambda iq: (iq not in self.entire_set, float('inf') if not self.cost_of_belonging(iq) is not None else self.cost_of_belonging(iq)))
        for key_ident in buy_order:
            if key_ident in pass_over or self.out_of_stock_outward(key_ident):
                continue
            log_entry = self.entire_set.get(key_ident) or self.result_rows.get(key_ident)
            entry_lines.append(_Prose._to_text(_Prose._cut_down(log_entry, 160))[:900])
            if len(entry_lines) >= ceiling_num:
                break
        return '\n'.join(entry_lines)

    def log_entry_belonging(self, key_ident: str) -> str:
        log_entry = self.entire_set.get(key_ident) or self.result_rows.get(key_ident)
        return '(no record seen)' if not log_entry else _Prose._to_text(_Prose._cut_down(log_entry, 400))[:3000]

    def basket_key_ids(self) -> list[str]:
        if self.basket_snapshot is not None and self.basket_by <= self.basket_observed_by:
            prose = _Prose._to_text(self.basket_snapshot)
            return [iq for iq in self.result_rows if iq in prose]
        return list(self.basket_entry_lines)

    def basket_prose(self) -> str:
        if self.basket_snapshot is not None and self.basket_by <= self.basket_observed_by:
            return _Prose._to_text(_Prose._cut_down(self.basket_snapshot, 300))[:3000]
        return "lines by Sable's record of the cart calls (listing: units): " + _Prose._to_text(self.basket_entry_lines)

    def basket_stamp(self) -> str:
        return _Prose._to_text(sorted(self.basket_entry_lines.items()))

class _Duties:

    def checkpoint_owing(self, point_in_time: str) -> bool:
        return any((pq['when'] == point_in_time for pq in self.routine)) and (point_in_time, self.basket_stamp() + f'#{self.revision}') not in self.checkpoint_approved_one

    def checkpoint(self, point_in_time: str) -> tuple[bool, str]:
        moves = [pq for pq in self.routine if pq['when'] == point_in_time]
        if not moves or self.closed_out or (not self.checkpoint_owing(point_in_time)):
            return (True, '')
        wind_up = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        instruments = [f"- {tq['function']['name']}: {tq['function'].get('description', '')} | parameters: {_Prose._to_text(tq['function'].get('parameters') or {})}" for tq in self.instruments if tq['function']['name'] not in wind_up]
        remarked: list[str] = []
        pushed_on = False
        for _ in range(CHECKPOINT_TURNS):
            cue = DUTY_SPEC.replace('STAGE', 'end without a purchase' if not 'order' == point_in_time else 'place the order') + '\n\nREQUEST:\n' + self.ask_text + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {pq['step']}" for pq in moves)) + '\n\nLEDGER:\n' + self.account() + '\n\nCART NOW: ' + self.basket_prose() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{iq}. {entry_line}' for (iq, entry_line) in enumerate(self.wrapped_diary, 1))) or '(nothing yet)') + ('' if not self.remarks else '\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.remarks[-4:])) + f"\n\nMESSAGES LEFT: {max(0, self.mesg_ceiling_num - self.mesg_consumed)}   STEPS LEFT: {self.residual()} (the {('ending' if not 'order' == point_in_time else 'order')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(instruments)
            obtained = self.inquire_json(cue, [SWIFT_MODEL, DEEP_MODEL] if not pushed_on else [DEEP_MODEL, SWIFT_MODEL], 2000)
            if obtained is None:
                break
            informed: list[str] = []
            retain = ''
            if remarked:
                self.accept_answered({'needs': obtained.get('needs')}, informed)
                retain = str(obtained.get('hold') or '').strip()
            invoke = None if not isinstance(obtained.get('call'), dict) else obtained.get('call')
            self.record_trail.append({'kind': 'gate', 'stage': point_in_time, 'call': _Prose._cut_down(invoke, 200), 'steps': _Prose._cut_down(obtained.get('steps'), 120), 'hold': retain[:200], 'told': informed})
            if informed or retain:
                self.revision += 1
                return (False, ('' if not remarked else 'Before going on, Sable did the steps the request asks for:\n' + '\n'.join(remarked) + '\n') + ("The shopper's reply changes the task: " + '; '.join(informed + ([] if not retain else [retain])) + '. Fix the cart for it, then check out again.'))
            label_name = str((invoke or {}).get('tool') or '')
            displayed = [eq for eq in _SchemaFit._entries_list(obtained.get('steps')) if isinstance(eq, dict)]
            discuss = 'order' == point_in_time and any((pq['messages'] > 0 for pq in moves))
            entry_lines = range(1, len(self.wrapped_diary) + 1)
            backing = {iq for iq in entry_lines if not discuss or (iq <= len(self.wrapped_moves) and self.wrapped_moves[iq - 1] > self.basket_by)}
            unattested = [eq for eq in displayed if eq.get('done') is not True or not any(((_SchemaFit._tally_of(nq) or 0) in backing for nq in _SchemaFit._entries_list(eq.get('lines'))))]
            if not invoke and unattested and (not remarked) and (not pushed_on) and self.duties.get('ask') and (self.mesg_ceiling_num > self.mesg_consumed) and (self.residual() >= 2):
                pushed_on = True
                self.record_trail.append({'kind': 'gate_unproven', 'steps': _Prose._cut_down(unattested, 120)})
                self.wrapped_diary.append(f'(Sable: no line made since the cart last changed at step {self.basket_by} shows ' + '; '.join((str(eq.get('step') or '') for eq in unattested))[:200] + f' done, and {self.mesg_ceiling_num - self.mesg_consumed} message(s) and {self.residual()} steps are left: give the call that does it)')
                self.wrapped_moves.append(-1)
                continue
            if not invoke or label_name not in self.blueprints or label_name in wind_up:
                if not invoke:
                    self.checkpoint_approved_one.add((point_in_time, self.basket_stamp() + f'#{self.revision}'))
                break
            if self.residual() < 2 or (label_name == self.duties.get('ask') and self.mesg_ceiling_num <= self.mesg_consumed):
                break
            defects: list[str] = []
            params = _SchemaFit._outline_form({} if not isinstance(invoke.get('args'), dict) else invoke.get('args'), self.blueprints[label_name], 'args', defects)
            if defects:
                break
            if label_name == self.duties.get('ask'):
                prose = next((vq for vq in params.values() if isinstance(vq, str)), '')
                bundle = self.dispatch([(1, '', prose)], 1)
                if not bundle:
                    break
                self.checkpoint_mesgs += 1
                remarked.append(f'Q: {prose}\nA: {bundle[0][2]}')
                continue
            comeback = self.perform(label_name, params)
            if comeback is None:
                break
            observed = comeback.get('observation')
            self.take_in(observed)
            self.wrapped_diary.append(f'step {self.move}: {label_name} {_Prose._to_text(params)} -> {self.display(observed)[:300]} (cart then: {self.basket_stamp()})')
            self.wrapped_moves.append(self.move)
            remarked.append(f'{label_name} {_Prose._to_text(params)} -> {self.display(observed)[:400]}')
        return (True, '' if not remarked else 'Before going on, Sable did the steps the request asks for:\n' + '\n'.join(remarked))

class _Answers:

    def view_out(self, label_name: str, viewing: Any) -> str:
        self.take_in(viewing)
        entry_lines = self.free_floating_entry_lines(viewing)
        if not entry_lines:
            return ''
        gist = self.accept_remark(entry_lines)
        if gist:
            self.backlogged.append(f'(seen on {label_name}) {gist}')
        return gist

    def accept_remark(self, entry_lines: list[str]) -> str:
        self.answered.update(entry_lines)
        cue = NOTICE_SPEC + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.account() + '\n\nTHE NOTE:\n' + '\n'.join(entry_lines)
        obtained = self.inquire_json(cue, [SWIFT_MODEL, FALLBACK_MODEL], 1200) or {}
        vanished = [str(gq) for gq in _SchemaFit._entries_list(obtained.get('gone')) if isinstance(gq, (str, int))]
        self.unbuyable.update((gq for gq in vanished if gq in self.result_rows))
        self.remark_out_of_stock = bool(vanished) or obtained.get('sold_out') is True
        if self.remark_out_of_stock and len(entry_lines) == 1:
            self.out_of_stock_entry_lines.update(entry_lines)
        if obtained.get('kind') not in (None, 'change'):
            return ''
        amended = []
        money_limit = obtained.get('budget')
        if isinstance(money_limit, (int, float)) and (not isinstance(money_limit, bool)) and (0 < money_limit) and (self.money_limit != money_limit):
            self.money_limit = money_limit
            amended.append(f'budget now {money_limit}')
        put_in: set[str] = set()
        discards = [eq for eq in _SchemaFit._entries_list(obtained.get('drop')) if isinstance(eq, dict) and str(eq.get('need') or '').strip()]
        puts_in = [eq for eq in _SchemaFit._entries_list(obtained.get('add')) if isinstance(eq, dict) and str(eq.get('need') or '').strip()]
        for record_item in discards:
            good = self.items[min(max(1, _SchemaFit._tally_of(record_item.get('item')) or 1), len(self.items)) - 1]
            objective = str(record_item['need']).strip()
            active = [mq for mq in good.demands + good.informed if mq not in good.discarded]
            match_one = [mq for mq in active if objective == mq]
            if not match_one:
                match_one = [mq for mq in active if _Prose._wordings(mq.split(':')[0]) == _Prose._wordings(objective.split(':')[0])][:1]
            good.discarded.extend(match_one or [objective])
            amended.append(f'dropped {objective}')
        for record_item in puts_in:
            good = self.items[min(max(1, _SchemaFit._tally_of(record_item.get('item')) or 1), len(self.items)) - 1]
            requirement = str(record_item['need']).strip()
            if requirement in good.discarded:
                good.discarded.remove(requirement)
            if requirement not in good.demands + good.informed:
                good.informed.append(requirement)
            put_in.add(requirement)
            amended.append(f'added {requirement}')
        if vanished:
            amended.append('unavailable: ' + ', '.join(vanished))
        if obtained.get('price_moved') is True:
            amended.append('a price changed')
        recap = str(obtained.get('summary') or '').strip()
        if amended or recap:
            self.revision += 1
            self.newest = []
            self.remarks.append(recap or '; '.join(amended))
            self.reset_caps()
        return (recap + ('' if not amended else ' (' + '; '.join(amended) + ')')).strip()

    def free_floating_entry_lines(self, viewing: Any) -> list[str]:
        entry_lines: list[str] = []
        if isinstance(viewing, dict) and (not self.key_ident(viewing)):
            for (field, val_text) in viewing.items():
                if isinstance(val_text, str) and len(val_text.split()) >= 5:
                    entry_lines.append(val_text)
                elif isinstance(val_text, list) and val_text and all((isinstance(vq, str) for vq in val_text)):
                    entry_lines.extend((vq for vq in val_text if len(vq.split()) >= 5))
                elif isinstance(val_text, dict) and (not self.key_ident(val_text, False, 1)):
                    for interior in val_text.values():
                        if isinstance(interior, str) and len(interior.split()) >= 6:
                            entry_lines.append(interior)
        return [entry_line for entry_line in entry_lines if entry_line not in self.answered]

    def uttered_text(self, viewing: Any) -> str:
        if isinstance(viewing, str):
            return viewing
        located: list[str] = []

        def traverse(element: Any, recursion: int) -> None:
            if 4 < recursion:
                return
            if isinstance(element, str) and len(element.split()) >= 2:
                located.append(element)
            elif isinstance(element, dict):
                for val_text in element.values():
                    traverse(val_text, recursion + 1)
            elif isinstance(element, list):
                for val_text in element:
                    traverse(val_text, recursion + 1)
        traverse(viewing, 0)
        return _Prose._to_text(viewing)[:400] if not located else ' '.join(located)[:800]

    def take_answers(self, dispatched: list[tuple[int, str, str]], engines: tuple[str, ...]=(DEEP_MODEL, SWIFT_MODEL)) -> dict[str, Any]:
        cue = ANSWER_SPEC + '\n\nITEMS:\n' + '\n'.join((f'item {nq}: {this.label_name}' for (nq, this) in enumerate(self.items, 1)))
        cue += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{iq}. (about item {nq}) Q: {qz}\nA: {aq}' for (iq, (nq, qz, aq)) in enumerate(dispatched, 1)))
        return self.inquire_json(cue, list(engines), 1500) or {}

    def accept_answered(self, answered: dict[str, Any], informed: list[str]) -> None:
        plenty = len(self.items) > 1
        for record_item in _SchemaFit._entries_list(answered.get('needs')):
            if not isinstance(record_item, dict) or not str(record_item.get('need') or '').strip():
                continue
            numeral = min(max(1, _SchemaFit._tally_of(record_item.get('item')) or 1), len(self.items))
            requirement = str(record_item['need']).strip()
            if requirement not in self.items[numeral - 1].informed:
                self.items[numeral - 1].informed.append(requirement)
                informed.append(requirement if not plenty else f'item {numeral}: {requirement}')
            if record_item.get('points') is True and (numeral, requirement) not in self.pointer_refs:
                self.pointer_refs.append((numeral, requirement))
        for record_item in _SchemaFit._entries_list(answered.get('drop')):
            if isinstance(record_item, dict) and str(record_item.get('property') or '').strip():
                good = self.items[min(max(1, _SchemaFit._tally_of(record_item.get('item')) or 1), len(self.items)) - 1]
                vanished = _Prose._wordings(str(record_item['property']).split(':')[0])
                match_one = [mq for mq in good.demands + good.informed if mq not in good.discarded and _Prose._wordings(mq.split(':')[0]) == vanished]
                good.discarded.extend(match_one)
                informed.extend((f'dropped {mq}' for mq in match_one))
        for record_item in _SchemaFit._entries_list(answered.get('free')):
            if isinstance(record_item, dict) and record_item.get('property'):
                numeral = min(max(1, _SchemaFit._tally_of(record_item.get('item')) or 1), len(self.items))
                self.items[numeral - 1].unused.append(str(record_item['property']))
        if isinstance(answered.get('budget'), (int, float)) and (not isinstance(answered.get('budget'), bool)) and (answered['budget'] > 0):
            self.money_limit = answered['budget']

class _Asking:

    def dispatch(self, query_texts: list[tuple[int, str, str]], lower_bound: int=6) -> list[tuple[int, str, str]]:
        instrument = self.duties['ask']
        dispatched: list[tuple[int, str, str]] = []
        for (numeral, _, prose) in query_texts:
            if self.closed_out or self.residual() <= lower_bound or self.mesg_ceiling_num <= self.mesg_consumed:
                break
            params = self.note_msg_params(prose)
            if not params:
                break
            comeback = self.perform(instrument, params)
            if comeback is None:
                break
            self.mesg_consumed += 1
            response = comeback.get('observation')
            wordings = self.uttered_text(response)
            dispatched.append((numeral, prose, wordings))
            self.wrapped_diary.append(f'step {self.move}: message to the shopper {prose!r} -> reply {wordings[:300]!r} (cart then: {self.basket_stamp()})')
            self.wrapped_moves.append(self.move)
            self.record_trail.append({'kind': 'ask', 'q': prose, 'a': wordings[:300]})
            self.mesg_declined = isinstance(response, dict) and bool(response.get('error'))
            if self.mesg_declined:
                break
        return dispatched

    def query_text(self, numeral: int, heading: str) -> str:
        return f'What {heading} would you like for the {self.items[numeral - 1].label_name}?'

    def upcoming_pick_out(self, buy_order: list[int], backlogs: dict[int, list[str]], located: set[int], closing_one_good: int | None) -> tuple[int, str] | None:
        for (numeral, backlog) in backlogs.items():
            if numeral <= len(self.items):
                good = self.items[numeral - 1]
                informed = [_Prose._root_word(tq.split(':')[0]) for tq in good.informed]
                asserted = [_Prose._root_word(tq.split(':')[0]) for tq in good.demands]
                backlog[:] = [heading for heading in backlog if not any((kq and kq <= _Prose._root_word(heading) for kq in informed)) and (not (heading in self.refill and any((kq and (kq <= _Prose._root_word(heading) or _Prose._root_word(heading) <= kq) for kq in asserted))))]
                inquired = [_Prose._root_word(heading) for (nq, heading) in self.inquired_headings if numeral == nq]
                backlog[:] = [heading for heading in backlog if not any((_Prose._root_word(heading) <= aq for aq in inquired))]
        for (numeral, backlog) in backlogs.items():
            if numeral <= len(self.items):
                repaired = {_Prose._wordings(mq.split(':', 1)[0]) for mq in self.items[numeral - 1].demands}
                backlog[:] = [heading for heading in backlog if _Prose._wordings(heading) not in repaired and _Prose._wordings(heading).rstrip('s') not in {fq.rstrip('s') for fq in repaired}]
        active = [nq for nq in buy_order if backlogs.get(nq)]
        if not active:
            return None
        collection = [nq for nq in active if nq not in located] or active
        start = 0 if not closing_one_good in buy_order else buy_order.index(closing_one_good) + 1
        for move in range(len(buy_order)):
            numeral = buy_order[(start + move) % len(buy_order)]
            if numeral in collection:
                return (numeral, backlogs[numeral].pop(0))
        return None

    def upcoming_roster(self, buy_order: list[int], backlogs: dict[int, list[str]], located: set[int], used_up: dict[int, int], money_limit: int) -> tuple[int, list[tuple[int, str]], str] | None:
        for (numeral, backlog) in backlogs.items():
            if numeral <= len(self.items):
                good = self.items[numeral - 1]
                informed = [_Prose._root_word(tq.split(':')[0]) for tq in good.informed]
                asserted = [_Prose._root_word(tq.split(':')[0]) for tq in good.demands]
                backlog[:] = [heading for heading in backlog if not any((kq and kq <= _Prose._root_word(heading) for kq in informed)) and (not (heading in self.refill and any((kq and (kq <= _Prose._root_word(heading) or _Prose._root_word(heading) <= kq) for kq in asserted))))]
                inquired = [_Prose._root_word(heading) for (nq, heading) in self.inquired_headings if numeral == nq]
                backlog[:] = [heading for heading in backlog if not any((_Prose._root_word(heading) <= aq for aq in inquired))]
        for (numeral, backlog) in backlogs.items():
            if numeral <= len(self.items):
                repaired = {_Prose._wordings(mq.split(':', 1)[0]) for mq in self.items[numeral - 1].demands}
                backlog[:] = [heading for heading in backlog if _Prose._wordings(heading) not in repaired and _Prose._wordings(heading).rstrip('s') not in {fq.rstrip('s') for fq in repaired}]
        active = [nq for nq in buy_order if backlogs.get(nq) or len(self.closing_one_rosters.get(nq, [])) > 1]
        if not active:
            return None
        if len(self.items) > money_limit and (not used_up):
            asked_pairs = [(nq, backlogs[nq].pop(0)) for nq in active]
            return (active[0], asked_pairs, ' '.join((self.query_text(nq, heading) for (nq, heading) in asked_pairs)))
        collection = [nq for nq in active if nq not in located] or active
        numeral = min(collection, key=lambda nq: (used_up.get(nq, 0), buy_order.index(nq)))
        good = self.items[numeral - 1]
        informed = [_Prose._root_word(tq.split(':')[0]) for tq in good.informed]
        headings = [heading for heading in reversed(self.closing_one_rosters.get(numeral, [])[1:]) if not any((kq and (kq <= _Prose._root_word(heading) or _Prose._root_word(heading) <= kq) for kq in informed))][:BATCH_WIDTH - 2]
        while backlogs[numeral] and len(headings) < BATCH_WIDTH:
            heading = backlogs[numeral].pop(0)
            if all((_Prose._wordings(heading) != _Prose._wordings(alternative) for alternative in headings)):
                headings.append(heading)
        self.closing_one_rosters[numeral] = headings
        return (numeral, [(numeral, heading) for heading in headings], self.roster_query_text(numeral, headings))

    def roster_query_text(self, numeral: int, headings: list[str]) -> str:
        good = self.items[numeral - 1]
        labels_text = []
        if good.aisle_shelf and _Prose._wordings(good.aisle_shelf) != _Prose._wordings(good.label_name):
            labels_text.append(good.aisle_shelf)
        values = [mq.split(':', 1)[-1].strip() for mq in good.demands if mq not in good.discarded]
        values = [vq for vq in values if vq and len(vq) <= 30]
        if values:
            labels_text.append(', '.join(values[:3]))
        lead_word = f'For the {good.label_name}' + ('' if not labels_text else f" ({'; '.join(labels_text)})")
        return f"{lead_word}, what {', '.join(headings)} would you like?"

    def questioning(self) -> str:
        self.inquired = True
        if len(self.items) < 2:
            self.roster_manner = False
        count = self.mesgs_unused()
        self.lone_good_rosters = not self.roster_manner and 3 <= count
        if 0 >= count or self.clock_residual() <= WINDUP_SECONDS + ASKING_SECONDS or (not self.duties.get('ask')):
            return ''
        self.enlarge()
        grid = self.field_grid()
        decided = []
        for (numeral, good) in enumerate(self.items, 1):
            decided.append(f'item {numeral}: {good.label_name} - settled by the request: ' + ('; '.join(good.demands) or 'nothing'))
        offer_rec = []
        closing_one = None
        for (index, (heading, label_name, held_by, values)) in enumerate(grid, 1):
            if closing_one != heading:
                offer_rec.append(f'category {heading!r}:')
                closing_one = heading
            offer_rec.append(f'  {index}. {label_name} (on {held_by}): ' + ' | '.join(values))
        wish_for = count + len(self.items) + 2
        if self.lone_good_rosters:
            wish_for += BATCH_WIDTH
        if self.roster_manner:
            wish_for = max(wish_for, min(2 * BATCH_WIDTH * len(self.items), 5 * BATCH_WIDTH))
        cue = QUESTION_SPEC.replace('COUNT', str(wish_for))
        if self.lone_good_rosters:
            cue += '\nThe picks after the first few feed one closing message that lists several properties, so give the full number: when the usable keys run out, add the everyday properties people weigh when they shop for that kind of thing, even where no table key names them.'
        if self.roster_manner:
            cue += "\nThese picks feed lists of several properties per message, so give the full number for every item: when an item's usable keys run out, add the everyday properties people weigh when they shop for that kind of thing, even where no table key names them."
        cue += '\n\nREQUEST:\n' + self.ask_text + '\n\nITEMS:\n' + '\n'.join(decided)
        earliest = [str(xq) for xq in _SchemaFit._entries_list(self.outline.get('ask_first')) if str(xq).strip()]
        if earliest:
            cue += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(earliest)
        cue += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(offer_rec) or '(no full listing records yet)')
        scheme = self.inquire_json(cue, [DEEP_MODEL, SWIFT_MODEL], 1800) or {}
        backlogs: dict[int, list[str]] = {}
        buy_order: list[int] = []
        for pick_out in _SchemaFit._entries_list(scheme.get('ask')):
            if not isinstance(pick_out, dict):
                continue
            numeral = min(max(1, _SchemaFit._tally_of(pick_out.get('item') or 1) or 1), len(self.items))
            heading = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick_out.get('label') or ''))[0].strip().rstrip('?.')
            if not heading or len(heading.split()) > 5:
                continue
            if re.search(NOISE_FIELDS_SOLO if not self.roster_manner else NOISE_FIELDS, _Prose._wordings(heading)):
                continue
            backlog = backlogs.setdefault(numeral, [])
            if all((_Prose._wordings(heading) != _Prose._wordings(alternative) for alternative in backlog)):
                backlog.append(heading)
            if numeral not in buy_order:
                buy_order.append(numeral)
        for numeral in range(1, len(self.items) + 1):
            backlog = backlogs.setdefault(numeral, [])
            for label_name in self.good_keys(numeral, grid, 8 if not (self.roster_manner or self.lone_good_rosters) else 2 * BATCH_WIDTH):
                if re.search(NOISE_FIELDS_SOLO if not self.roster_manner else NOISE_FIELDS, _Prose._wordings(label_name)):
                    continue
                if all((_Prose._wordings(label_name) != _Prose._wordings(alternative) for alternative in backlog)):
                    backlog.append(label_name)
                    self.refill.add(label_name)
            if backlog and numeral not in buy_order:
                buy_order.append(numeral)
        self.record_trail.append({'kind': 'plan', 'ask': _Prose._cut_down(scheme.get('ask'), 120), 'queues': _Prose._cut_down(backlogs, 60)})
        dispatched: list[tuple[int, str, str]] = []
        informed: list[str] = []
        located: set[int] = set()
        closing_one_good: int | None = None
        used_up: dict[int, int] = {}
        while self.mesgs_unused() > 0 and (not self.closed_out) and (self.clock_residual() > WINDUP_SECONDS + ASKING_SECONDS):
            if not (self.roster_manner or (self.lone_good_rosters and used_up and (self.mesgs_unused() <= TAIL_ROSTERS))):
                single_pick = self.upcoming_pick_out(buy_order, backlogs, located, closing_one_good)
                rostered = None if not single_pick else (single_pick[0], [single_pick], self.query_text(single_pick[0], single_pick[1]))
            else:
                rostered = self.upcoming_roster(buy_order, backlogs, located, used_up, count)
            if rostered is None:
                break
            (numeral, asked_pairs, prose) = rostered
            bundle = self.dispatch([(numeral, asked_pairs[0][1], prose)])
            if not bundle:
                break
            closing_one_good = numeral
            used_up[numeral] = used_up.get(numeral, 0) + 1
            self.inquired_headings.extend(asked_pairs)
            if self.mesg_declined:
                self.mesg_consumed -= 1
                self.mesg_ceiling_num = self.mesg_consumed
                break
            dispatched += bundle
            answered = self.take_answers(bundle, (SWIFT_MODEL, DEEP_MODEL))
            if (self.roster_manner or self.lone_good_rosters) and len(asked_pairs) > 1 and (1 in [_SchemaFit._tally_of(xq) for xq in _SchemaFit._entries_list(answered.get('crowded'))]):
                self.roster_manner = False
                self.lone_good_rosters = False
                self.record_trail.append({'kind': 'lists_refused', 'q': prose[:200]})
                for (nq, heading) in reversed(asked_pairs):
                    backlogs.setdefault(nq, []).insert(0, heading)
                self.inquired_headings = [pq for pq in self.inquired_headings if pq not in asked_pairs]
            ahead_of = len(informed)
            self.accept_answered(answered, informed)
            if len(informed) > ahead_of:
                located.add(numeral)
        if not dispatched:
            return ''
        if informed:
            self.revision += 1
        entry_lines = [f'Q: {qz}\nA: {aq}' for (_, qz, aq) in dispatched]
        recap = "Sable's interviewer asked the shopper (each question used one step):\n" + '\n'.join(entry_lines)
        recap += '\nNew firm requirements from these answers: ' + ('none' if not informed else '; '.join(informed))
        recap += '. Apply them to every pick (search again with the new values where needed).'
        aimed = [requirement for (_, requirement) in self.pointer_refs if any((requirement in tq for tq in informed))]
        if aimed:
            recap += ' These take their value from another listing: ' + ' | '.join(aimed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if self.backlogged:
            recap += '\n[notes seen meanwhile] ' + ' | '.join(self.backlogged)
            self.backlogged = []
        return recap

    def note_msg_params(self, prose: str) -> dict[str, Any] | None:
        label_name = self.duties.get('ask')
        blueprint = self.blueprints.get(label_name or '') or {}
        features = blueprint.get('properties') or {}
        proses = [kq for (kq, vq) in features.items() if isinstance(vq, dict) and 'string' in _SchemaFit._sorts(vq) and ('enum' not in vq)]
        objective = next((kq for kq in blueprint.get('required') or [] if kq in proses), None) or (None if not proses else proses[0])
        if not objective:
            return None
        params: dict[str, Any] = {objective: prose}
        for field in blueprint.get('required') or []:
            if field not in params:
                specification = features.get(field) or {}
                if not (isinstance(specification.get('enum'), list) and specification['enum']):
                    return None
                else:
                    params[field] = specification['enum'][0]
        defects: list[str] = []
        params = _SchemaFit._outline_form(params, blueprint, 'args', defects)
        return params if not defects else None

    def prepared_into_inquire(self) -> bool:
        if self.inquired or not self.duties.get('ask') or self.mesgs_unused() <= 0:
            return False
        if self.residual() <= 8:
            return False
        expanded = len(self.entire_set)
        if expanded >= 3 * len(self.items) + 1:
            return True
        return self.move >= max(8, self.upper_moves // 4)

class _Opening:

    def set_side_by_space(self) -> int:
        specification = ((self.blueprints.get(self.duties.get('compare') or '') or {}).get('properties') or {}).get(self.set_side_by_objective() or '') or {}
        ceiling = None if not isinstance(specification.get('maxItems'), int) else specification['maxItems']
        if self.set_side_by_faulty:
            space = max(2, self.set_side_by_approved_one, min(self.set_side_by_faulty - 1, max(2, self.set_side_by_faulty // 2)))
            return space if not ceiling else min(space, ceiling)
        if ceiling:
            return ceiling
        return max(4, self.set_side_by_approved_one)

    def set_side_by_objective(self) -> str | None:
        blueprint = self.blueprints.get(self.duties.get('compare') or '') or {}
        features = blueprint.get('properties') or {}
        objective = next((kq for (kq, vq) in features.items() if isinstance(vq, dict) and 'array' in _SchemaFit._sorts(vq)), None)
        if not objective or any((objective != kq for kq in blueprint.get('required') or [])):
            return None
        return objective

    def combine_expands(self, invokes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        (expander, combiner, objective) = (self.duties.get('open'), self.duties.get('compare'), self.lone_objective(self.duties.get('open')))
        plenty = self.set_side_by_objective()
        if not expander or not combiner or (not objective) or (not plenty) or (combiner == expander):
            return invokes
        if self.set_side_by_unadorned_kind():
            return invokes
        picks_out: list[tuple[int, str]] = []
        for (index, invoke) in enumerate(invokes):
            func = invoke.get('function') or {}
            params = None if not isinstance(func.get('arguments'), str) else _Prose._json_struct(func.get('arguments'))
            key_ident = str((params or {}).get(objective) or '').strip()
            if func.get('name') == expander and params and (set(params) == {objective}) and (key_ident in self.result_rows) and (key_ident not in self.vanished):
                picks_out.append((index, key_ident))
        picks_out = picks_out[:self.set_side_by_space()]
        if len(picks_out) < 2:
            return invokes
        earliest = picks_out[0][0]
        outward = []
        for (index, invoke) in enumerate(invokes):
            if earliest == index:
                outward.append({'id': invoke['id'], 'type': 'function', 'function': {'name': combiner, 'arguments': _Prose._to_text({plenty: [iq for (_, iq) in picks_out]})}})
            elif not index in {iq for (iq, _) in picks_out}:
                outward.append(invoke)
            else:
                outward.append({'id': invoke['id'], 'joined': True})
        self.record_trail.append({'kind': 'joined', 'ids': [iq for (_, iq) in picks_out]})
        return outward

    def expand_result_rows(self, result_rows: list[str]) -> bool:
        (plenty, combiner) = (self.set_side_by_objective(), self.duties.get('compare'))
        instrument = self.traits_instrument()
        objective = self.lone_objective(instrument)
        if plenty and combiner and (not self.set_side_by_unadorned_kind()):
            scheme = [(combiner, {plenty: result_rows[:self.set_side_by_space()]})]
        elif not (instrument and objective):
            return False
        else:
            space = 2 if not (self.residual() >= 15 and self.clock_residual() > WINDUP_SECONDS + 30) else 4
            scheme = [(instrument, {objective: iq}) for iq in result_rows[:space]]
        if self.residual() < len(scheme) + 5:
            return False
        ahead_of = len(self.entire_set)
        for (label_name, params) in scheme:
            defects: list[str] = []
            params = _SchemaFit._outline_form(params, self.blueprints.get(label_name), 'args', defects)
            if defects:
                return False
            comeback = self.perform(label_name, params)
            if comeback is None or self.closed_out:
                break
            self.view_out(label_name, comeback.get('observation'))
            if combiner == label_name and isinstance(params.get(plenty), list) and (len(params[plenty]) > 1):
                if not any((str(iq) in _Prose._to_text(comeback.get('observation')) for iq in params[plenty])):
                    self.set_side_by_faulty = min(self.set_side_by_faulty or 999, len(params[plenty]))
                else:
                    self.set_side_by_approved_one = max(self.set_side_by_approved_one, len(params[plenty]))
        self.record_trail.append({'kind': 'audit_open', 'rows': result_rows, 'opened': len(self.entire_set) - ahead_of})
        return len(self.entire_set) > ahead_of

    def enlarge_one_at_a_time(self) -> None:
        instrument = self.traits_instrument()
        objective = self.lone_objective(instrument)
        if not instrument or not objective:
            return
        row_lanes: list[list[str]] = []
        for key_ids in self.good_groups():
            expanded = sum((1 for iq in key_ids if iq in self.entire_set))
            collection = [iq for iq in key_ids if iq not in self.entire_set and self.for_sale(iq)]
            if self.outline.get('pick') == 'cheapest':
                collection.sort(key=lambda iq: float('inf') if not self.cost_of_belonging(iq) is not None else self.cost_of_belonging(iq))
            row_lanes.append(collection[:max(0, min(2 if not len(self.items) == 1 else 4, 8 - expanded))])
        backlog = [row_lane[kq] for kq in range(4) for row_lane in row_lanes if kq < len(row_lane)]
        self.top_up_traits(backlog[:min(8, max(4, 2 * len(row_lanes)))], 22)

    def enlarge(self) -> None:
        instrument = self.duties.get('compare')
        blueprint = self.blueprints.get(instrument or '') or {}
        features = blueprint.get('properties') or {}
        objective = next((kq for (kq, vq) in features.items() if isinstance(vq, dict) and 'array' in _SchemaFit._sorts(vq)), None)
        if not instrument or not objective or any((objective != kq for kq in blueprint.get('required') or [])) or self.set_side_by_unadorned_kind():
            self.enlarge_one_at_a_time()
            return
        if self.residual() <= 12:
            return
        ceiling = self.set_side_by_space()
        if 2 > ceiling:
            self.enlarge_one_at_a_time()
            return
        graded = self.good_groups()
        fraction = max(1, ceiling // max(1, len(graded)))
        picks_out: list[str] = []
        for key_ids in graded:
            picks_out += [iq for iq in key_ids if iq not in self.entire_set][:fraction]
        picks_out = picks_out[:ceiling]
        if len(picks_out) < 2:
            return
        defects: list[str] = []
        params = _SchemaFit._outline_form({objective: picks_out}, blueprint, 'args', defects)
        if defects:
            return
        ahead_of = len(self.entire_set)
        comeback = self.perform(instrument, params)
        if comeback is not None:
            observed = comeback.get('observation')
            self.view_out(instrument, observed)
            self.remark_snapshot(instrument, observed)
            result_rows = self.log_entries_within(observed)
            if result_rows:
                self.traits_out_of[instrument] = self.traits_out_of.get(instrument, False) or any((_RecordScan._trait_panel(rq) for (_, rq) in result_rows))
            self.record_trail.append({'kind': 'widen', 'ids': picks_out, 'opened': len(self.entire_set) - ahead_of})
            if any((str(iq) in _Prose._to_text(observed) for iq in picks_out)):
                self.set_side_by_approved_one = max(self.set_side_by_approved_one, len(picks_out))
            else:
                self.set_side_by_faulty = min(self.set_side_by_faulty or 999, len(picks_out))
            if result_rows and self.traits_out_of.get(instrument) is False:
                self.enlarge_one_at_a_time()
                return
            if len(self.entire_set) == ahead_of and len(picks_out) > 2 and (self.residual() > 12):
                defects = []
                params = _SchemaFit._outline_form({objective: picks_out[:2]}, blueprint, 'args', defects)
                if not defects:
                    comeback = self.perform(instrument, params)
                    if comeback is not None:
                        self.view_out(instrument, comeback.get('observation'))
                        if any((str(iq) in _Prose._to_text(comeback.get('observation')) for iq in picks_out[:2])):
                            self.set_side_by_approved_one = max(self.set_side_by_approved_one, 2)

    def repeat_hitting_result_rows(self, params: dict[str, Any], viewing: Any) -> str:
        search_key = _Prose._to_text({kq: vq for (kq, vq) in sorted(params.items()) if isinstance(vq, str)})
        for (key_ident, _) in self.log_entries_within(viewing):
            self.result_row_search_keys.setdefault(key_ident, set()).add(search_key)
        if REPEAT_READS <= self.repeat_hit_expanded or self.residual() <= self.end_reserve() + 8:
            return ''
        ceiling = None if not (isinstance(self.money_limit, (int, float)) and (not isinstance(self.money_limit, bool))) else self.money_limit
        collection = [iq for (iq, search_keys) in self.result_row_search_keys.items() if len(search_keys) >= 2 and iq in self.result_rows and (iq not in self.entire_set) and self.for_sale(iq) and (ceiling is None or (self.cost_of_belonging(iq) or 0) <= ceiling)]
        collection.sort(key=lambda iq: (-len(self.result_row_search_keys[iq]), self.cost_of_belonging(iq) or 0))
        proses = self.top_up_traits(collection[:min(2, REPEAT_READS - self.repeat_hit_expanded)], self.end_reserve() + 8)
        self.repeat_hit_expanded += len(proses)
        if not proses:
            return ''
        return '\n[Sable read the attributes of rows that several of your searches returned; a title can hide a fit]\n' + '\n'.join(proses)

    def top_up_traits(self, key_ids: list[str], preserve: int) -> list[str]:
        instrument = self.traits_instrument()
        objective = self.lone_objective(instrument)
        if not instrument or not objective:
            return []
        proses = []
        for key_ident in [iq for iq in key_ids if iq in self.result_rows and iq not in self.entire_set and (not self.out_of_stock_outward(iq))]:
            if self.residual() <= preserve or self.closed_out or self.traits_out_of.get(instrument) is False:
                break
            defects: list[str] = []
            params = _SchemaFit._outline_form({objective: key_ident}, self.blueprints.get(instrument), 'args', defects)
            if defects:
                break
            comeback = self.perform(instrument, params)
            if comeback is None:
                break
            observed = comeback.get('observation')
            self.view_out(instrument, observed)
            self.remark_snapshot(instrument, observed)
            if isinstance(observed, dict) and self.key_ident(observed):
                self.traits_out_of[instrument] = self.traits_out_of.get(instrument, False) or bool(_RecordScan._trait_panel(observed))
            proses.append(f'{key_ident}: ' + self.display(observed)[:1500])
        if proses:
            self.record_trail.append({'kind': 'fill_facts', 'n': len(proses)})
        return proses

class _Paging:

    def roomier_retain(self) -> str:
        if self.roomier_retained or self.outline.get('pick') != 'cheapest' or self.residual() < 14 + 2 * len(self.items):
            return ''
        self.roomier_retained = True
        entry_lines = [f'{iq} at {self.cost_of_belonging(iq):g}' for iq in self.basket_key_ids() if self.cost_of_belonging(iq) is not None]
        return f"Not run yet (no step used). {self.residual()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + ('' if not entry_lines else f" ({'; '.join(entry_lines)})") + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def further_sheets(self, label_name: str, params: dict[str, Any], viewing: Any) -> str:
        self.sheeted_seen_text = []
        pick_out = self.outline.get('pick')
        scheme = None if not pick_out in ('cheapest', 'most', 'least') else self.sheeting(label_name, viewing)
        if not scheme:
            return ''
        (count, displayed, arg_name, val_text) = scheme
        supplied = params.get(arg_name)
        subsequently = bool(str(supplied or '').strip()) if not (isinstance(supplied, int) and (not isinstance(supplied, bool))) else supplied not in (None, 0, 1)
        if val_text is None or subsequently or displayed >= count:
            return ''
        if 'cheapest' == pick_out and self.cost_of_ordered(viewing):
            return ''
        sheets = -(-(count - displayed) // displayed)
        if 2 < sheets or self.sheets_scan + sheets > OWN_SHEETS or self.residual() - self.end_reserve() - sheets < 12:
            return f"\n[Sable] This search matched {count} listings and showed {displayed}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {arg_name}={_Prose._to_text(val_text)}."
        proses = []
        for _ in range(sheets):
            defects: list[str] = []
            invoke = _SchemaFit._outline_form({**params, arg_name: val_text}, self.blueprints.get(label_name), 'args', defects)
            if defects:
                break
            comeback = self.perform(label_name, invoke)
            if comeback is None or self.closed_out:
                break
            self.sheets_scan += 1
            observed = comeback.get('observation')
            self.view_out(label_name, observed)
            self.sheeted_seen_text.append(observed)
            displayed_prose = self.display(observed)
            self.scan_keep_store[label_name + _Prose._to_text(invoke)] = (self.revision, self.move, displayed_prose)
            proses.append(f'{arg_name}={_Prose._to_text(val_text)}: {displayed_prose}')
            once = self.sheeting(label_name, observed)
            val_text = None if not once else once[3]
            if val_text is None:
                break
        self.record_trail.append({'kind': 'pages', 'tool': label_name, 'read': len(proses)})
        return '' if not proses else '\n[Sable read the rest of this search itself, one step a page]\n' + '\n'.join(proses)

    def sheeting(self, label_name: str, viewing: Any) -> tuple[int, int, str, Any] | None:
        outline_form = self.scan_outline_form(label_name, viewing)
        if not outline_form or not outline_form['count'] or (not outline_form['next']) or (not outline_form['next_arg']) or (not isinstance(viewing, dict)):
            return None
        (result_rows, count, val_text) = (viewing.get(outline_form['rows']), viewing.get(outline_form['count']), viewing.get(outline_form['next']))
        if not isinstance(result_rows, list) or not result_rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if val_text is not None and (isinstance(val_text, bool) or not isinstance(val_text, (str, int))):
            return None
        return (count, len(result_rows), outline_form['next_arg'], val_text)

class _Pricing:

    def less_costly_result_rows(self) -> list[str]:
        if self.outline.get('pick') != 'cheapest':
            return []
        stamp = self.basket_stamp() + f'#{self.revision}'
        catgroup_field = self.section_field()
        if stamp in self.less_costly_stamps or not catgroup_field:
            return []
        self.less_costly_stamps.add(stamp)
        basket = self.basket_key_ids()
        picks_out: list[str] = []
        for entry_line in basket:
            cost_of = self.cost_of_belonging(entry_line)
            variety = str((self.result_rows.get(entry_line) or {}).get(catgroup_field) or '').strip().lower()
            if cost_of is None or not variety:
                continue
            collection = [iq for iq in self.observed_buy_order if iq not in basket and iq not in self.entire_set and self.for_sale(iq) and (str((self.result_rows.get(iq) or {}).get(catgroup_field) or '').strip().lower() == variety) and ((float('inf') if not self.cost_of_belonging(iq) is not None else self.cost_of_belonging(iq)) < cost_of)]
            collection.sort(key=lambda iq: self.cost_of_belonging(iq) or 0.0)
            picks_out += [iq for iq in collection[:2] if iq not in picks_out]
        return self.top_up_traits(picks_out[:LOWCOST_EXPANDS], self.end_reserve() + 4)

    def least_costly_at_hand(self, viewings: list[Any]) -> str:
        if self.outline.get('pick') != 'cheapest' or not self.cost_of_field:
            return ''
        located = {iq for observed in viewings for (iq, _) in self.log_entries_within(observed)}
        result_rows = [(self.cost_of_belonging(iq), iq) for iq in located if self.for_sale(iq) and self.cost_of_belonging(iq) is not None]
        if len(result_rows) < 2:
            return ''
        return '\n[Sable] buyable rows here, cheapest first: ' + ', '.join((f'{iq} ({pq:g})' for (pq, iq) in sorted(result_rows)[:6]))

    def cost_of_ordered(self, viewing: Any) -> bool:
        costs_of = [self.cost_of_belonging(iq) for (iq, _) in self.log_entries_within(viewing)]
        return len(costs_of) >= 3 and None not in costs_of and all((bq >= aq for (aq, bq) in zip(costs_of, costs_of[1:])))

    def for_sale(self, key_ident: str) -> bool:
        if self.out_of_stock_outward(key_ident):
            return False
        log_entry = self.result_rows.get(key_ident) or {}
        field = self.supply_field or next((kq for (kq, vq) in log_entry.items() if re.search('stock|availab', str(kq).lower()) and isinstance(vq, (bool, int, float))), None)
        val_text = None if not field else log_entry.get(field)
        return not (val_text is False or (isinstance(val_text, (int, float)) and (not isinstance(val_text, bool)) and (0 >= val_text)))

    def cost_of_belonging(self, key_ident: str) -> float | None:
        log_entry = self.result_rows.get(key_ident) or {}
        field = self.cost_of_field or next((kq for (kq, vq) in log_entry.items() if re.search('(?:^|[^a-z])(?:price|cost)(?:$|[^a-z])', str(kq).lower()) and isinstance(vq, (int, float, str, dict)) and (not isinstance(vq, bool))), None)
        val_text = None if not field else log_entry.get(field)
        if isinstance(val_text, dict):
            val_text = next((vq for vq in val_text.values() if isinstance(vq, (int, float)) and (not isinstance(vq, bool))), None)
        if isinstance(val_text, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', val_text):
            val_text = float(val_text)
        return None if not (isinstance(val_text, (int, float)) and (not isinstance(val_text, bool))) else float(val_text)

class _KeyTable:

    def good_keys(self, numeral: int, grid: list[tuple[str, str, int, list[str]]], ceiling_num: int=4) -> list[str]:
        good = self.items[numeral - 1]
        decided = ' '.join(good.demands).lower()
        suits = {heading: self.heading_suit(heading, good) for (heading, _, _, _) in grid}
        top_pick = max(suits.values(), default=0)
        joint: list[str] = []
        lone: list[str] = []
        for (heading, label_name, held_by, values) in grid:
            close_by = 0 < top_pick and suits[heading] == top_pick
            if close_by and 2 <= held_by and (label_name not in decided) and (label_name not in joint + lone):
                (lone if not self.field_unlike.get((heading, label_name), len(values)) < held_by else joint).append(label_name)
        return (joint + lone)[:ceiling_num]

    def blend_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:

        def level(prose: str) -> str:
            return re.sub('[^a-z0-9]+', '', prose)
        label_names = sorted(keys, key=lambda nq: (-len(nq.split()), -keys[nq]['n'], nq))
        fused: dict[str, dict[str, Any]] = {}
        for label_name in label_names:
            position = keys[label_name]
            ours = _Prose._root_word(label_name)
            objective = None
            for (large, specification) in fused.items():
                their_own = _Prose._root_word(large)
                the_two = [iq for iq in position['per'] if iq in specification['per']]
                accord = sum((1 for iq in the_two if level(position['per'][iq]) == level(specification['per'][iq])))
                fraction = bool({level(vq) for vq in position['values']} & {level(vq) for vq in specification['values']})
                if (their_own >= ours or ours >= their_own) and (accord or fraction) and (accord == len(the_two)):
                    objective = large
                    break
                if ours & their_own and len(the_two) >= 3 and (accord == len(the_two)):
                    objective = large
                    break
            if objective is None:
                fused[label_name] = {'n': position['n'], 'values': list(position['values']), 'per': dict(position['per'])}
                continue
            specification = fused[objective]
            for (key_ident, prose) in position['per'].items():
                specification['per'].setdefault(key_ident, prose)
            for val_text in position['values']:
                if val_text not in specification['values']:
                    specification['values'].append(val_text)
            specification['n'] = len(specification['per'])
        return fused

    def field_grid(self) -> list[tuple[str, str, int, list[str]]]:
        catgroup_field = self.section_field()
        groups: dict[str, dict[str, Any]] = {}
        for key_ident in self.observed_buy_order:
            log_entry = self.entire_set.get(key_ident)
            if not log_entry:
                continue
            panel = _RecordScan._trait_panel(log_entry) or {}
            heading = 'listings' if not catgroup_field else str(log_entry.get(catgroup_field) or 'listings')
            group = groups.setdefault(heading, {'n': 0, 'keys': {}})
            group['n'] += 1
            wrapped: set[str] = set()
            for (field, val_text) in panel.items():
                label_name = _Prose._wordings(field)
                if not label_name or label_name in wrapped:
                    continue
                wrapped.add(label_name)
                position = group['keys'].setdefault(label_name, {'n': 0, 'values': [], 'per': {}})
                position['n'] += 1
                prose = str(val_text).strip().lower()
                position['per'][key_ident] = prose
                if prose not in position['values']:
                    position['values'].append(prose)
        fitted_to = {heading for heading in groups if any((self.heading_suit(heading, good) for good in self.items))}
        if fitted_to:
            groups = {heading: group for (heading, group) in groups.items() if heading in fitted_to}
        grid = []
        for (heading, group) in sorted(groups.items(), key=lambda field_value: -field_value[1]['n']):
            group['keys'] = self.blend_keys(group['keys'])
            preserved: list[tuple[str, dict[str, Any]]] = []
            for (label_name, position) in sorted(group['keys'].items(), key=lambda field_value: -field_value[1]['n']):
                values = list(position['per'].values())
                if not values:
                    continue
                average = sum((len(vq) for vq in values)) / len(values)
                if 40 < average or sum((1 for vq in values if ' / ' in vq)) * 2 >= len(values):
                    continue
                if len(values) >= 4 and len(set(values)) == len(values) and (12 < average):
                    continue
                near_twin = any((label_name.startswith(alternative + ' ') and position['per'] == specification['per'] for (alternative, specification) in preserved))
                if near_twin:
                    continue
                preserved.append((label_name, position))
            for (label_name, position) in preserved[:14]:
                grid.append((heading, label_name, position['n'], [vq[:40] for vq in position['values'][:4]]))
                self.field_unlike[heading, label_name] = len(set(position['per'].values()))
        return grid

class _Categories:

    def aisle_entry_lines(self, key_ids: list[str]) -> str:
        field = self.section_field()
        if not field:
            return ''
        headings: list[str] = []
        for key_ident in self.observed_buy_order:
            heading = str((self.result_rows.get(key_ident) or {}).get(field) or '').strip()
            if heading and heading not in headings:
                headings.append(heading)
        outward = []
        for key_ident in key_ids:
            heading = str((self.result_rows.get(key_ident) or {}).get(field) or '').strip()
            if heading:
                outward.append(f'{key_ident}: category {heading!r}')
        for (numeral, good) in enumerate(self.items, 1):
            close_by = sorted((heading for heading in headings if self.heading_suit(heading, good)), key=lambda heading: -self.heading_suit(heading, good))
            if close_by:
                outward.append(f'item {numeral} ({good.label_name}): categories seen whose names share its words: ' + ', '.join((repr(lq) for lq in close_by[:4])))
        return '\n'.join(outward)

    def lone_objective(self, instrument: str | None) -> str | None:
        blueprint = self.blueprints.get(instrument or '') or {}
        features = blueprint.get('properties') or {}
        proses = [kq for (kq, vq) in features.items() if isinstance(vq, dict) and 'string' in _SchemaFit._sorts(vq) and ('enum' not in vq)]
        objective = next((kq for kq in blueprint.get('required') or [] if kq in proses), None) or (None if not len(proses) == 1 else proses[0])
        if not instrument or not objective or any((objective != kq for kq in blueprint.get('required') or [])):
            return None
        return objective

    def good_groups(self) -> list[list[str]]:
        catgroup_field = self.section_field()
        groups: dict[str, list[str]] = {}
        for key_ident in self.observed_buy_order:
            heading = '' if not catgroup_field else str(self.result_rows[key_ident].get(catgroup_field) or '')
            groups.setdefault(heading, []).append(key_ident)
        fitted_to = [key_ids for (heading, key_ids) in groups.items() if any((self.heading_suit(heading, good) for good in self.items))]
        collection = fitted_to or list(groups.values())
        return sorted(collection, key=lambda key_ids: -len(key_ids))[:max(1, len(self.items))]

    def heading_suit(self, heading: str, good: WantedGood) -> int:
        ours = _Prose._root_word(good.aisle_shelf or '') | _Prose._root_word(good.label_name)
        their_own = _Prose._root_word(heading)
        if not their_own:
            return 0
        if _Prose._wordings(heading) in (_Prose._wordings(good.aisle_shelf or ''), _Prose._wordings(good.label_name)):
            return 3
        if ours >= their_own:
            return 2
        if {wq for wq in their_own if len(wq) >= 4} & {wq for wq in ours if len(wq) >= 4}:
            return 1
        return 0

    def section_field(self) -> str | None:
        tallies: dict[str, set[str]] = {}
        shown_now: dict[str, int] = {}
        for result_row in self.result_rows.values():
            for (field, val_text) in result_row.items():
                if isinstance(val_text, str) and 0 < len(val_text) <= 40 and (not re.search('\\d{3}', val_text)):
                    tallies.setdefault(field, set()).add(val_text)
                    shown_now[field] = shown_now.get(field, 0) + 1
        aggregate = max(1, len(self.result_rows))
        familiar = self.headings()
        top_pick = None
        for (field, values) in tallies.items():
            if shown_now[field] < 0.8 * aggregate or len(values) > 40:
                continue
            rostered = sum((1 for vq in values if vq.strip().lower() in familiar)) / len(values)
            average = sum((len(vq) for vq in values)) / len(values)
            mark_score = (-rostered, len(values), -shown_now[field], -average)
            if top_pick is None or mark_score < top_pick[0]:
                top_pick = (mark_score, field)
        return None if not top_pick else top_pick[1]

    def headings(self) -> set[str]:
        located: set[str] = set()

        def traverse(blueprint: Any, recursion: int) -> None:
            if 6 < recursion or not isinstance(blueprint, dict):
                return
            for alt in blueprint.get('enum') or []:
                if isinstance(alt, str):
                    located.add(alt.strip().lower())
            for val_text in blueprint.values():
                if isinstance(val_text, dict):
                    traverse(val_text, recursion + 1)
        for blueprint in self.blueprints.values():
            traverse(blueprint, 0)
        return located

class _RowKnowledge:

    def scan_outline_form(self, label_name: str, viewing: Any) -> dict[str, Any] | None:
        if label_name in self.outline_forms:
            return self.outline_forms[label_name]
        if not isinstance(viewing, dict):
            return None
        rosters = [kq for (kq, vq) in viewing.items() if isinstance(vq, list) and len(vq) >= 2 and all((isinstance(rq, dict) for rq in vq))]
        if not rosters:
            return None
        self.outline_forms[label_name] = None
        sample = {kq: vq if not kq in rosters else vq[:3] for (kq, vq) in viewing.items()}
        cue = LAYOUT_SPEC + '\n\nTOOL: ' + label_name + ' | parameters: ' + _Prose._to_text(self.blueprints.get(label_name) or {}) + '\n\nRESULT (first rows only):\n' + _Prose._to_text(_Prose._cut_down(sample, 120))[:4000]
        obtained = self.inquire_json(cue, [SWIFT_MODEL, DEEP_MODEL], 1500) or {}
        result_rows = None if not obtained.get('rows') in rosters else obtained.get('rows')
        if not result_rows:
            return None
        earliest = viewing[result_rows][0]
        features = (self.blueprints.get(label_name) or {}).get('properties') or {}

        def labelled(field: Any, at_place: dict[str, Any]) -> str | None:
            return None if not (isinstance(field, str) and field in at_place) else field
        outline_form = {'rows': result_rows, 'count': labelled(obtained.get('count'), viewing), 'next': labelled(obtained.get('next'), viewing), 'next_arg': labelled(obtained.get('next_arg'), features), 'price': labelled(obtained.get('price'), earliest), 'stock': labelled(obtained.get('stock'), earliest)}
        if outline_form['count'] and (isinstance(viewing[outline_form['count']], bool) or not isinstance(viewing[outline_form['count']], int)):
            outline_form['count'] = None
        self.outline_forms[label_name] = outline_form
        self.cost_of_field = self.cost_of_field or outline_form['price']
        self.supply_field = self.supply_field or outline_form['stock']
        self.record_trail.append({'kind': 'shape', 'tool': label_name, 'shape': outline_form})
        return outline_form

    def displayed_by_now(self, label_name: str, key_ident: str) -> bool:
        (revision, carry) = self.snapshot_keys.get(key_ident, (-1, set()))
        keys = self.instrument_keys.get(label_name)
        return bool(keys) and self.revision == revision and (carry >= keys) and (key_ident not in self.vanished)

    def remark_snapshot(self, label_name: str, viewing: Any) -> None:
        for (key_ident, log_entry) in self.log_entries_within(viewing):
            keys = set(log_entry)
            self.instrument_keys.setdefault(label_name, set()).update(keys)
            (revision, carry) = self.snapshot_keys.get(key_ident, (-1, set()))
            self.snapshot_keys[key_ident] = (self.revision, (set() if not self.revision == revision else carry) | keys)

    def log_entries_within(self, viewing: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(viewing, dict):
            key_ident = self.key_ident(viewing)
            if key_ident:
                return [(key_ident, viewing)]
            located: list[tuple[str, dict[str, Any]]] = []
            for val_text in viewing.values():
                if isinstance(val_text, list):
                    located += [(iq, rq) for rq in val_text if isinstance(rq, dict) for iq in [self.key_ident(rq, True)] if iq]
                elif isinstance(val_text, dict):
                    located += [(iq, rq) for vq in val_text.values() if isinstance(vq, list) for rq in vq if isinstance(rq, dict) for iq in [self.key_ident(rq, True)] if iq]
            return located
        if isinstance(viewing, list):
            return [(iq, rq) for rq in viewing if isinstance(rq, dict) for iq in [self.key_ident(rq, True)] if iq]
        return []

    def take_in(self, viewing: Any) -> None:

        def traverse(element: Any, recursion: int, within_roster: bool=False) -> None:
            if 6 < recursion:
                return
            if isinstance(element, dict):
                key_ident = self.key_ident(element, within_roster, recursion)
                if key_ident:
                    if key_ident not in self.result_rows:
                        self.observed_buy_order.append(key_ident)
                        self.result_rows[key_ident] = {}
                    former = self.result_rows[key_ident]
                    if any((former.get(kq) is True and vq is False for (kq, vq) in element.items())):
                        self.unbuyable.add(key_ident)
                    elif any((former.get(kq) is False and vq is True for (kq, vq) in element.items())):
                        self.unbuyable.discard(key_ident)
                    self.result_rows[key_ident].update(element)
                    if _RecordScan._trait_panel(element):
                        self.entire_set[key_ident] = dict(self.result_rows[key_ident])
                    elif key_ident in self.entire_set:
                        self.entire_set[key_ident].update(element)
                    return
                for val_text in element.values():
                    traverse(val_text, recursion + 1)
            elif isinstance(element, list):
                result_rows = self.pick_up_result_rows(element)
                for val_text in element:
                    traverse(val_text, recursion + 1, result_rows)
        traverse(viewing, 0)

    def pick_up_result_rows(self, element: list[Any]) -> bool:
        result_rows = [nq for nq in element if isinstance(nq, dict) and len(nq) >= 3]
        if not result_rows or len(result_rows) < len(element) / 2:
            return False
        if self.key_id_field and all((_RecordScan._key_id_alike(rq.get(self.key_id_field), True) for rq in result_rows)):
            return True
        located = _RecordScan._pick_up_key_id(result_rows)
        if located:
            self.key_id_field = located
            return True
        return False

    def key_ident(self, element: Any, within_roster: bool=False, recursion: int=0) -> str | None:
        if not isinstance(element, dict) or len(element) < 3:
            return None
        if self.key_id_field and _RecordScan._key_id_alike(element.get(self.key_id_field), True):
            return element[self.key_id_field].strip()
        enclosed = any((isinstance(vq, dict) for vq in element.values()))
        if not (within_roster or 0 == recursion or enclosed):
            return None
        located = _RecordScan._pick_up_key_id([element])
        if located and ('id' in str(located).lower() or element[located].strip() in self.result_rows):
            return element[located].strip()
        return None

class _LedgerView:

    def display(self, viewing: Any) -> str:
        prose = _Prose._to_text(_Prose._cut_down(viewing, DISPLAY_LEN))
        if len(prose) > VIEW_CAP:
            prose = prose[:VIEW_CAP] + '...(cut)'
        return prose

    def grounds(self, viewing: Any) -> str:
        key_ids: list[str] = []

        def traverse(element: Any, recursion: int) -> None:
            if 6 < recursion:
                return
            if isinstance(element, dict):
                key_ident = self.key_ident(element, False, recursion)
                if key_ident and _RecordScan._trait_panel(element) and (key_ident not in key_ids):
                    key_ids.append(key_ident)
                    return
                for val_text in element.values():
                    traverse(val_text, recursion + 1)
            elif isinstance(element, list):
                for val_text in element:
                    traverse(val_text, recursion + 1)
        traverse(viewing, 0)
        requirements = self.requirements_currently()
        if not key_ids or not requirements:
            return ''
        plenty = len(self.items) > 1
        entry_lines = []
        for key_ident in key_ids[:12]:
            panel = _RecordScan._trait_panel(self.entire_set.get(key_ident) or {}) or {}
            portions = []
            for (numeral, requirement) in requirements[:8]:
                feature = requirement.split(':', 1)[0].strip()
                match_many = [f'{kq}={str(vq)[:50]}' for (kq, vq) in panel.items() if _Prose._label_names(_Prose._root_word(feature), _Prose._root_word(kq))]
                marker = '' if not plenty else f'(item {numeral}) '
                if match_many:
                    portions.append(f'{marker}{feature} -> ' + ', '.join(match_many[:3]))
            if portions:
                entry_lines.append(f'{key_ident}: ' + '; '.join(portions))
        if not entry_lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(entry_lines)

    def requirements_currently(self) -> list[tuple[int, str]]:
        outward = []
        for (numeral, good) in enumerate(self.items, 1):
            for requirement in good.demands + good.informed:
                if requirement not in good.discarded:
                    outward.append((numeral, requirement))
        return outward

    def state_line(self) -> str:
        residual = self.residual()
        prose = f'[Sable] step {self.move} of {self.upper_moves} used, {residual} left.'
        if residual <= self.end_reserve() + 1:
            prose += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif 6 >= residual or (residual <= len(self.items) + 5 and 0 == self.checkouts):
            prose += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif 0 == self.checkouts and 12 >= residual:
            prose += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif 0 == self.checkouts and (not self.basket_altered_cart) and (self.move >= self.upper_moves // 2):
            prose += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        seconds = self.clock_residual()
        if seconds < WINDUP_SECONDS + 50:
            prose += f' Time is short: about {max(0, int(seconds))} seconds are left for this task. Cart the fitting listings you have and check out now.'
        active = {requirement for (_, requirement) in self.requirements_currently()}
        look_ups = [requirement for (_, requirement) in self.pointer_refs if requirement in active and requirement not in self.pointer_ref_values]
        if look_ups and 0 == self.checkouts:
            prose += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(look_ups[:3])
        if self.guide_refs:
            prose += '\n[review candidates] ' + ', '.join(self.guide_refs)
        if self.remarks:
            prose += '\n[changes so far] ' + ' | '.join(self.remarks[-4:])
        prose += '\n[ledger]\n' + self.account()
        return prose

    def account(self) -> str:
        portions = []
        plenty = len(self.items) > 1
        for (numeral, good) in enumerate(self.items, 1):
            requirements = [mq + ('' if not mq in self.pointer_ref_values else f' (value on the page it points at: {self.pointer_ref_values[mq]}' + ('' if not mq in self.pointer_ref_directives else f'; a rule stated on that page gives: {self.pointer_ref_directives[mq]}') + ')') for mq in good.demands + good.informed if mq not in good.discarded]
            lead_word = f'item: {good.label_name}' if not plenty else f'item {numeral}: {good.label_name}'
            if good.aisle_shelf:
                lead_word += f' (store label: {good.aisle_shelf})'
            lead_word += f', quantity {good.units}'
            prose = lead_word + '; requirements: ' + ('none beyond the kind of product' if not requirements else '; '.join(requirements))
            if good.unused:
                prose += '; shopper does not mind: ' + ', '.join(good.unused)
            portions.append(prose)
        outline = self.outline
        if self.money_limit is not None:
            coverage = 'in total' if not outline.get('budget_scope') == 'each' else 'for each item'
            portions.append(f"budget: at most {self.money_limit} {outline.get('budget_unit') or ''} {coverage}".replace('  ', ' '))
        pick_out = outline.get('pick')
        if 'cheapest' == pick_out:
            portions.append('pick: the cheapest listing that fits (lowest price)')
        elif not pick_out in ('most', 'least'):
            portions.append('pick: any listing that fits everything')
        else:
            portions.append(f"pick: the {pick_out} {outline.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        if outline.get('in_stock_only', True) is not False:
            portions.append('in stock only')
        for hope in _SchemaFit._entries_list(outline.get('wishes')):
            portions.append(f'wish (tie-breaker only): {hope}')
        for directive in _SchemaFit._entries_list(outline.get('rules')):
            portions.append(f'rule from the request: {directive}')
        if outline.get('quiet_decline') is True:
            portions.append('if nothing fits: end without naming or suggesting any listing')
        for record_item in self.routine:
            at_time = {'choose': 'before choosing', 'order': 'at the order (Sable does it when you check out)', 'end': 'at an ending without a purchase (Sable does it then)'}[record_item['when']]
            portions.append(f"step the request asks for, {at_time}: {record_item['step']}")
        traits = self.traits_instrument()
        if traits and (self.traits_out_of.get(traits) is True or self.duties.get('offer')):
            unadorned_kind = sorted((tq for tq in self.sheet_instruments() | {self.duties.get('offer') or ''} if tq and self.traits_out_of.get(tq) is False))
            sheets = sorted((tq for tq in self.sheet_instruments() if traits != tq))
            portions.append(f'listing attributes come from {traits}' + ('' if not unadorned_kind else f"; {', '.join(unadorned_kind)} show none") + ('' if not sheets else f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(sheets)}"))
        return '\n'.join(('- ' + pq for pq in portions))

class _RequestReading:

    def separate(self) -> None:
        for (numeral, good) in enumerate(self.items, 1):
            aims = [requirement for (nq, requirement) in self.pointer_refs if numeral == nq and requirement in good.demands]
            remainder = [mq for mq in good.demands if mq not in aims]
            if not aims or not remainder:
                continue
            cue = SPLIT_SPEC + '\n\nREQUEST:\n' + self.ask_text + f'\n\nPRODUCT: {good.label_name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(aims) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(remainder)
            obtained = self.inquire_json(cue, self.verifier(), 500) or {}
            shifted = [mq for mq in remainder if mq in [str(xq).strip() for xq in _SchemaFit._entries_list(obtained.get('describe_other'))]]
            if not shifted:
                continue
            good.demands = [mq for mq in good.demands if mq not in shifted]
            combined = aims[0] + ' (that listing: ' + '; '.join(shifted) + ')'
            good.demands = [mq if not mq == aims[0] else combined for mq in good.demands]
            self.pointer_refs = [(nq, requirement if not (nq, requirement) == (numeral, aims[0]) else combined) for (nq, requirement) in self.pointer_refs]
            self.record_trail.append({'kind': 'unmix', 'item': numeral, 'moved': shifted})

    def scan_ask_text(self) -> None:
        entry_lines = []
        for instrument in self.instruments:
            func = instrument['function']
            entry_lines.append(f"- {func['name']}: {func.get('description', '')} | parameters: {_Prose._to_text(func.get('parameters') or {})}")
        cue = TASK_SPEC + '\nTOOLS:\n' + '\n'.join(entry_lines) + '\n\nREQUEST:\n' + self.ask_text
        if self.snapshot.get('opening') is not None:
            cue += '\n\nOPENING NOTE FROM THE STORE:\n' + _Prose._to_text(_Prose._cut_down(self.snapshot.get('opening'), 1500))
        outline = self.inquire_json(cue, [SWIFT_MODEL], 4000) or {}
        if not self.nil_balance and (not _SchemaFit._entries_list(outline.get('items')) or not isinstance(outline.get('roles'), dict) or (not outline['roles'].get('buy'))):
            outline = self.inquire_json(cue, [DEEP_MODEL, SWIFT_MODEL], 2500) or outline
        self.outline = outline
        if outline.get('pick') not in ('cheapest', 'most', 'least'):
            outline['pick'] = 'cheapest'
        duties = {} if not isinstance(outline.get('roles'), dict) else outline.get('roles')
        self.duties = {kq: vq for (kq, vq) in duties.items() if isinstance(vq, str) and vq in self.blueprints}
        for field in ('buy_args', 'refuse_args'):
            if not isinstance(outline.get(field), dict):
                outline[field] = {}
        (checkout, decline) = (self.duties.get('buy'), self.duties.get('refuse'))
        if checkout and decline == checkout and (not outline['buy_args'] or not outline['refuse_args'] or outline['buy_args'] == outline['refuse_args']):
            inquire = f'The store tool {checkout} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + _Prose._to_text(self.blueprints.get(checkout)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            repaired = self.inquire_json(inquire, [SWIFT_MODEL, DEEP_MODEL], 600) or {}
            if isinstance(repaired.get('buy_args'), dict) and isinstance(repaired.get('refuse_args'), dict) and (repaired['buy_args'] != repaired['refuse_args']):
                (outline['buy_args'], outline['refuse_args']) = (repaired['buy_args'], repaired['refuse_args'])
        for record_item in _SchemaFit._entries_list(outline.get('items')):
            if not isinstance(record_item, dict):
                continue
            amount = _SchemaFit._tally_of(record_item.get('quantity'))
            count = 1 if not (amount and 0 < amount) else amount
            demands = [str(mq) for mq in _SchemaFit._entries_list(record_item.get('musts')) if mq]
            self.items.append(WantedGood(str(record_item.get('name') or 'the product'), str(record_item.get('shelf') or ''), count, demands))
        if not self.items:
            self.items = [WantedGood('the requested product', '', 1, [])]
        for look_up in _SchemaFit._entries_list(outline.get('lookups')):
            prose = str(look_up).strip()
            if not any((prose in good.demands for good in self.items)):
                field = _Prose._wordings(prose.split(':')[0])
                prose = next((mq for good in self.items for mq in good.demands if _Prose._wordings(mq.split(':')[0]) == field), prose)
            numeral = next((nq for (nq, good) in enumerate(self.items, 1) if prose in good.demands), 1)
            if prose and (numeral, prose) not in self.pointer_refs:
                self.pointer_refs.append((numeral, prose))
        self.separate()
        money_limit = outline.get('budget')
        if isinstance(money_limit, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', money_limit):
            money_limit = float(money_limit)
        if isinstance(money_limit, (int, float)) and (not isinstance(money_limit, bool)) and (0 < money_limit):
            self.money_limit = money_limit
        ceiling_num = _SchemaFit._tally_of(outline.get('messages'))
        if ceiling_num is not None and 0 <= ceiling_num:
            self.mesg_ceiling_num = ceiling_num
        elif self.duties.get('ask'):
            self.mesg_ceiling_num = 3
        moves = _SchemaFit._tally_of(outline.get('steps'))
        if moves is not None and 10 <= moves < self.upper_moves:
            self.upper_moves = moves
        for record_item in _SchemaFit._entries_list(outline.get('process')):
            if isinstance(record_item, dict) and str(record_item.get('step') or '').strip():
                at_time = str(record_item.get('when') or '').strip().lower()
                self.routine.append({'step': str(record_item['step']).strip(), 'when': 'order' if not at_time in ('choose', 'order', 'end') else at_time, 'messages': max(0, _SchemaFit._tally_of(record_item.get('messages')) or 0)})
        if self.duties.get('ask'):
            discuss = [pq['messages'] for pq in self.routine if pq['when'] == 'order' and pq['messages']]
            self.held_back = 0 if not discuss else min(max(discuss), self.mesg_ceiling_num)

class _StoreLink:

    def perform(self, label_name: str, params: dict[str, Any]) -> dict[str, Any] | None:
        packet = dict(self.link_info)
        packet['step'] = self.move + 1
        packet['action'] = {'name': label_name, 'args': params}
        for _ in range(4):
            outcome = _world.post_verbose('/environment/call', packet)
            if isinstance(outcome.data, dict):
                data = outcome.data
                self.move = self.move + 1 if not isinstance(data.get('step'), int) else data['step']
                if data.get('finished'):
                    self.closed_out = True
                observed = data.get('observation')
                if label_name in self.sheet_instruments() and (not (isinstance(observed, dict) and observed.get('error'))):
                    for val_text in params.values():
                        self.expanded.update((vq.strip() for vq in ([val_text] if not isinstance(val_text, list) else val_text) if isinstance(vq, str)))
                return data
            error = outcome.error or {}
            state_line = error.get('status')
            payload_text = str(error.get('body') or '')
            self.record_trail.append({'kind': 'env_fail', 'status': state_line, 'body': payload_text[:200]})
            if 409 == state_line:
                minor = payload_text.lower()
                if 'finish' in minor or 'quarantin' in minor or 'finaliz' in minor:
                    self.closed_out = True
                    return None
                located = re.search('expected (\\d+)', payload_text)
                if located:
                    self.move = int(located.group(1)) - 1
                    packet['step'] = self.move + 1
                    continue
                if 'reuse' in minor or 'different action' in minor:
                    self.move += 1
                    packet['step'] = self.move + 1
                    continue
                return None
            if state_line in (400, 401, 402, 403, 404, 422):
                if 402 == state_line:
                    self.went_broke = True
                return None
        return None

class _ModelLink:

    def inquire_json(self, cue: str, engines: list[str], upper_word_units: int=1800) -> dict[str, Any] | None:
        for engine in engines:
            note_msg = self.reason([{'role': 'user', 'content': cue}], [engine], upper_word_units)
            obtained = _Prose._json_struct((note_msg or {}).get('content'))
            if obtained is not None:
                return obtained
            if self.went_broke or self.nil_balance or self.invoke_ceiling <= self.invokes or self.outward_belonging_clock or (COST_HALT <= self.spend_sum):
                return None
        return None

    def invoke_engine(self, engine: str, note_msgs: list[dict[str, Any]], upper_word_units: int, instruments: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        dispatched = note_msgs
        if engine.startswith('anthropic/') and len(note_msgs) > 1:
            dispatched = [dict(mq) for mq in note_msgs]
            stamps = [0] + ([] if not len(dispatched) > 1 else [len(dispatched) - 1])
            for index in stamps:
                body_text = dispatched[index].get('content')
                if isinstance(body_text, str) and body_text:
                    dispatched[index]['content'] = [{'type': 'text', 'text': body_text, 'cache_control': {'type': 'ephemeral'}}]
        payload_text: dict[str, Any] = {'model': engine, 'messages': dispatched, 'max_tokens': upper_word_units, 'temperature': 0}
        if instruments:
            payload_text['tools'] = instruments
            payload_text['tool_choice'] = 'auto'
        outcome = _lm.post_verbose('/inference/chat/completions', payload_text)
        data = outcome.data
        if isinstance(data, dict) and data.get('choices'):
            use_info = data.get('usage') or {}
            if isinstance(use_info.get('cost'), (int, float)):
                self.spend_sum += float(use_info['cost'])
            note_msg = data['choices'][0].get('message') or {}
            if note_msg.get('content') or note_msg.get('tool_calls'):
                return (note_msg, 200)
            self.record_trail.append({'kind': 'llm_fail', 'model': engine, 'status': 0})
            return (None, 0)
        error = outcome.error or {}
        state_line = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            state_line = _SchemaFit._tally_of(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.record_trail.append({'kind': 'llm_fail', 'model': engine, 'status': state_line, 'body': str(error.get('body') or '')[:200]})
        return (None, state_line)

    def reason(self, note_msgs: list[dict[str, Any]], engines: list[str], upper_word_units: int, instruments: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for engine in engines:
            if self.invoke_ceiling <= self.invokes or COST_HALT <= self.spend_sum:
                return None
            if self.clock_residual() < FINAL_CALL_SECONDS:
                self.outward_belonging_clock = True
                return None
            self.invokes += 1
            attempts = 0
            while True:
                (note_msg, state_line) = self.invoke_engine(engine, note_msgs, upper_word_units, instruments)
                if note_msg is not None:
                    return note_msg
                transient = state_line is None or state_line in (402, 408, 425, 429) or (isinstance(state_line, int) and 500 <= state_line)
                if not transient or attempts >= (len(BACKOFFS) if not state_line in (None, 408) else 1) or self.waited_out + BACKOFFS[attempts] > BACKOFF_SUM:
                    break
                if self.clock_residual() - BACKOFFS[attempts] < FINAL_CALL_SECONDS:
                    self.outward_belonging_clock = True
                    break
                wait_for = BACKOFFS[attempts] * (0.8 + 0.4 * random.random())
                self.waited_out += BACKOFFS[attempts]
                attempts += 1
                threading.Event().wait(wait_for)
            if 402 == state_line:
                self.nil_balance = True
                return None
        return None

class _ToolKinds:

    def hushed(self, label_name: str, params: dict[str, Any]) -> dict[str, Any]:
        blueprint = self.blueprints.get(label_name) or {}
        features = blueprint.get('properties') or {}
        outward = dict(params)
        for (field, specification) in features.items():
            if not isinstance(specification, dict) or 'string' not in _SchemaFit._sorts(specification) or 'enum' in specification:
                continue
            val_text = outward.get(field)
            if val_text is None and field not in (blueprint.get('required') or []):
                continue
            prose = str(val_text or '')
            labelled = any((key_ident in prose for key_ident in self.result_rows)) or self.outline.get('quiet_decline') is not False
            if labelled or not prose:
                outward[field] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:specification.get('maxLength', 200)]
        return outward

    def dealt(self) -> list[dict[str, Any]]:
        residual = self.residual()
        inquire = self.duties.get('ask')
        preserve: set[str] | None = None
        wind_ups = {self.duties.get('buy'), self.duties.get('refuse')} - {None}
        if residual <= self.end_reserve():
            preserve = set(wind_ups)
        elif residual <= self.end_reserve() + 3:
            preserve = set(wind_ups) | {self.duties.get('cart'), self.duties.get('cart_view')}
        elif residual <= len(self.items) + 5 and 0 == self.checkouts:
            preserve = set(wind_ups) | {self.duties.get('cart'), self.duties.get('cart_view'), self.duties.get('offer')}
        instruments = [tq for tq in self.instruments if tq['function']['name'] != inquire]
        if preserve is not None and wind_ups:
            tightened = [tq for tq in instruments if tq['function']['name'] in preserve]
            instruments = tightened or instruments
        return instruments

    def exists_decline(self, label_name: str, params: dict[str, Any]) -> bool:
        if label_name != self.duties.get('refuse'):
            return False
        repaired = {} if not isinstance(self.outline.get('refuse_args'), dict) else self.outline.get('refuse_args')
        return all((params.get(kq) == vq for (kq, vq) in repaired.items()))

    def exists_checkout(self, label_name: str, params: dict[str, Any]) -> bool:
        if label_name != self.duties.get('buy'):
            return False
        repaired = {} if not isinstance(self.outline.get('buy_args'), dict) else self.outline.get('buy_args')
        return all((params.get(kq) == vq for (kq, vq) in repaired.items()))

    def deal_instrument(self) -> str | None:
        return self.duties.get('offer') or self.duties.get('open')

    def set_side_by_unadorned_kind(self) -> bool:
        observed = self.set_side_by_traits()
        return observed is False or (observed is None and (self.traits_out_of.get(self.traits_instrument() or '') is True or bool(self.duties.get('offer'))))

    def set_side_by_traits(self) -> bool | None:
        return self.traits_out_of.get(self.duties.get('compare') or '')

    def traits_instrument(self) -> str | None:
        familiar = next((tq for (tq, approved_one) in self.traits_out_of.items() if approved_one and self.lone_objective(tq)), None)
        if familiar:
            return familiar
        instrument = self.duties.get('open')
        return None if not (self.traits_out_of.get(instrument or '') is not False and self.lone_objective(instrument)) else instrument

    def sheet_instruments(self) -> set[str]:
        sheet = self.duties.get('offer') or self.duties.get('open')
        return {tq for tq in (sheet, self.duties.get('compare')) if tq}

class _Budgets:

    def pare(self) -> None:
        bulk = sum((len(_Prose._to_text(mq.get('content'))) for mq in self.dialogue))
        if DIALOGUE_CAP >= bulk:
            return
        instruments = [iq for (iq, mq) in enumerate(self.dialogue) if mq.get('role') == 'tool']
        for index in instruments[:max(1, len(instruments) // 2)]:
            body_text = self.dialogue[index].get('content') or ''
            if len(body_text) > 600:
                key_ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', body_text)
                key_ids = [iq for iq in dict.fromkeys(key_ids) if iq in self.result_rows][:20]
                self.dialogue[index]['content'] = body_text[:400] + f"...(older result shortened; listings in it: {', '.join(key_ids)})"

    def end_reserve(self) -> int:
        discuss = any((pq['when'] == 'order' and pq['messages'] > 0 for pq in self.routine))
        return 1 if not (discuss and self.mesg_ceiling_num > self.mesg_consumed and self.checkpoint_owing('order')) else 2

    def mesgs_unused(self) -> int:
        return self.mesg_ceiling_num - self.mesg_consumed - max(0, self.held_back - self.checkpoint_mesgs)

    def reset_caps(self) -> None:
        self.reviewed = False
        self.nudged_on = False
        self.decline_panels = 0
        self.belated_retained = False
        self.roomier_retained = False

    def assign_ceilings(self) -> None:
        self.cycle_ceiling = max(TURN_FLOOR, min(110, self.upper_moves))
        self.invoke_ceiling = max(CALL_FLOOR, self.cycle_ceiling + 25)

    def verifier(self) -> list[str]:
        return [SWIFT_MODEL, DEEP_MODEL] if not COST_SWIFT > self.spend_sum else [DEEP_MODEL, SWIFT_MODEL]

    def clock_residual(self) -> float:
        return PLANNED_SECONDS - self.launched.elapsed()

    def residual(self) -> int:
        return self.upper_moves - self.move

class Sable(_Loop, _Closing, _Review, _StockOuts, _Pointers, _Verifying, _CartBook, _Duties, _Answers, _Asking, _Opening, _Paging, _Pricing, _KeyTable, _Categories, _RowKnowledge, _LedgerView, _RequestReading, _StoreLink, _ModelLink, _ToolKinds, _Budgets):

    def __init__(self, issue: dict[str, Any]):
        surroundings = issue['environment']
        self.link_info = dict(surroundings['binding'])
        self.snapshot = surroundings['policy_view']
        self.ask_text = str(self.snapshot.get('query') or '')
        self.instruments = [tq for tq in self.snapshot.get('tools') or [] if isinstance(tq, dict) and isinstance(tq.get('function'), dict) and isinstance(tq['function'].get('name'), str)]
        self.blueprints = {tq['function']['name']: {'type': 'object', 'properties': {}} if not isinstance(tq['function'].get('parameters'), dict) else tq['function'].get('parameters') for tq in self.instruments}
        moves = _SchemaFit._tally_of(self.snapshot.get('max_steps'))
        if moves is None and isinstance(self.snapshot.get('max_steps'), float):
            moves = int(self.snapshot['max_steps'])
        self.upper_moves = 30 if not (moves and 0 < moves) else moves
        self.launched = _Stopclock()
        self.cycle_ceiling = TURN_FLOOR
        self.invoke_ceiling = CALL_FLOOR
        self.combine_missed = False
        self.outward_belonging_clock = False
        self.basket_displays_entry_lines = False
        self.count_field: str | None = None
        self.mesg_declined = False
        self.refill: set[str] = set()
        self.pointer_ref_directives: dict[str, str] = {}
        self.inquired_headings: list[tuple[int, str]] = []
        self.result_row_search_keys: dict[str, set[str]] = {}
        self.part_way_retentions = 0
        self.repeat_hit_expanded = 0
        self.roster_manner = True
        self.lone_good_rosters = False
        self.closing_one_rosters: dict[int, list[str]] = {}
        self.field_unlike: dict[tuple[str, str], int] = {}
        self.move = 0
        self.closed_out = bool(self.snapshot.get('finished'))
        self.went_broke = False
        self.nil_balance = False
        self.waited_out = 0
        self.failures = 0
        self.spend_sum = 0.0
        self.invokes = 0
        self.cycle_invokes = 0
        self.record_trail: list[dict[str, Any]] = []
        self.outline: dict[str, Any] = {}
        self.duties: dict[str, str] = {}
        self.items: list[WantedGood] = []
        self.money_limit: Any = None
        self.mesg_ceiling_num = 0
        self.mesg_consumed = 0
        self.routine: list[dict[str, Any]] = []
        self.held_back = 0
        self.checkpoint_mesgs = 0
        self.checkpoint_approved_one: set[tuple[str, str]] = set()
        self.wrapped_diary: list[str] = []
        self.wrapped_moves: list[int] = []
        self.inquired = False
        self.question_answer: list[tuple[str, str]] = []
        self.result_rows: dict[str, dict[str, Any]] = {}
        self.entire_set: dict[str, dict[str, Any]] = {}
        self.observed_buy_order: list[str] = []
        self.answered: set[str] = set()
        self.remarks: list[str] = []
        self.revision = 0
        self.basket_altered_cart = False
        self.basket_by = -1
        self.basket_observed_by = -1
        self.basket_snapshot: Any = None
        self.basket_entry_lines: dict[str, int] = {}
        self.verified: set[str] = set()
        self.reviewed = False
        self.deal_observed: dict[str, int] = {}
        self.vanished: dict[str, int] = {}
        self.unbuyable: set[str] = set()
        self.out_of_stock_entry_lines: set[str] = set()
        self.remark_out_of_stock = False
        self.retained = 0
        self.key_id_params: set[tuple[str, str]] = set()
        self.basket_remarked = ''
        self.declined_by = -1
        self.guide_refs: list[str] = []
        self.decline_panels = 0
        self.backstop_retentions = 0
        self.nudged_on = False
        self.made_up: list[str] = []
        self.discarded: set[str] = set()
        self.belated_retained = False
        self.guide_ref_items: dict[str, list[str]] = {}
        self.judgements: dict[str, bool] = {}
        self.approved: str | None = None
        self.finished_off = False
        self.key_id_field: str | None = None
        self.traits_out_of: dict[str, bool] = {}
        self.backlogged: list[str] = []
        self.checkouts = 0
        self.rejections = 0
        self.newest: list[str] = []
        self.dialogue: list[dict[str, Any]] = []
        self.expanded: set[str] = set()
        self.pointer_refs: list[tuple[int, str]] = []
        self.pointer_ref_values: dict[str, str] = {}
        self.chain_set = 0
        self.pursuit_panels = 0
        self.set_side_by_approved_one = 0
        self.set_side_by_faulty = 0
        self.instrument_keys: dict[str, set[str]] = {}
        self.snapshot_keys: dict[str, tuple[int, set[str]]] = {}
        self.scan_keep_store: dict[str, tuple[int, int, str]] = {}
        self.sheets_scan = 0
        self.outline_forms: dict[str, dict[str, Any] | None] = {}
        self.cost_of_field: str | None = None
        self.supply_field: str | None = None
        self.less_costly_stamps: set[str] = set()
        self.roomier_retained = False
        self.sheeted_seen_text: list[Any] = []

    def carry_on(self) -> list[dict[str, Any]]:
        if self.closed_out:
            return self.record_trail
        self.scan_ask_text()
        self.record_trail.append({'kind': 'brief', 'brief': self.outline})
        launch_view = self.snapshot.get('opening')
        if launch_view is not None:
            self.take_in(launch_view)
        opener_text = "SHOPPER'S REQUEST:\n" + self.ask_text
        if launch_view is not None:
            opener_text += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.display(launch_view)
        opener_text += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.state_line()
        self.dialogue = [{'role': 'system', 'content': MANUAL}, {'role': 'user', 'content': opener_text}]
        self.assign_ceilings()
        self.operate(self.cycle_ceiling, COST_FIRM)
        if not self.closed_out:
            converted = self.close_out(anew=True)
            if converted and (not self.closed_out) and (not self.went_broke) and (not self.nil_balance) and (self.residual() > 1) and (self.clock_residual() > WINDUP_SECONDS):
                self.dialogue.append({'role': 'user', 'content': '[Sable] Sable tried to end the episode and ' + converted + '\nApply what the store says, then finish.\n' + self.state_line()})
                self.operate(self.cycle_invokes + 10, COST_FIRM + 0.01)
            if not self.closed_out:
                self.close_out()
        self.record_trail.append({'kind': 'end', 'finished': self.closed_out, 'steps': self.move, 'cost': round(self.spend_sum, 5), 'calls': self.invokes})
        return self.record_trail

    def operate(self, cycle_ceiling: int, outlay_ceiling: float) -> None:
        quiet_turns = 0
        while not self.closed_out and (not self.went_broke) and (self.residual() > 0) and (cycle_ceiling > self.cycle_invokes) and (outlay_ceiling > self.spend_sum) and (self.clock_residual() > WINDUP_SECONDS):
            if self.prepared_into_inquire():
                recap = self.questioning()
                if recap:
                    self.dialogue.append({'role': 'user', 'content': recap + '\n' + self.state_line()})
                continue
            self.pare()
            self.cycle_invokes += 1
            note_msg = self.reason(self.dialogue, [SWIFT_MODEL, FALLBACK_MODEL], 3000, self.dealt())
            if note_msg is None:
                self.failures += 1
                if self.nil_balance or self.invoke_ceiling <= self.invokes or 2 < self.failures or self.outward_belonging_clock or (COST_HALT <= self.spend_sum):
                    break
                continue
            invokes = []
            for (numeral, invoke) in enumerate(note_msg.get('tool_calls') or []):
                if not isinstance(invoke, dict):
                    continue
                func = {} if not isinstance(invoke.get('function'), dict) else invoke.get('function')
                crude = func.get('arguments')
                decoded = ({} if not (isinstance(crude, str) and crude.strip()) else _Prose._json_struct(crude)) if not isinstance(crude, dict) else crude
                invokes.append({'id': str(invoke.get('id') or f'k{self.cycle_invokes}_{numeral}'), 'type': 'function', 'function': {'name': str(func.get('name') or ''), 'arguments': _Prose._to_text({} if not isinstance(decoded, dict) else decoded)}})
            body_text = note_msg.get('content')
            if not isinstance(body_text, str):
                body_text = '' if not body_text else _Prose._to_text(body_text)
            if not invokes:
                quiet_turns += 1
                self.dialogue.append({'role': 'assistant', 'content': body_text or '(no call)'})
                self.dialogue.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.state_line()})
                if 3 <= quiet_turns:
                    break
                continue
            quiet_turns = 0
            binding_to = {self.duties.get('cart'), self.duties.get('buy'), self.duties.get('refuse')} - {None}
            if not self.inquired and self.duties.get('ask') and (self.mesgs_unused() > 0) and any((cq['function']['name'] in binding_to for cq in invokes)) and (self.residual() > 8):
                self.dialogue.append({'role': 'assistant', 'content': body_text or None, 'tool_calls': invokes})
                for invoke in invokes:
                    self.dialogue.append({'role': 'tool', 'tool_call_id': invoke['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                recap = self.questioning()
                self.dialogue.append({'role': 'user', 'content': (recap or 'The interviewer had nothing to ask.') + '\n' + self.state_line()})
                continue
            self.dialogue.append({'role': 'assistant', 'content': body_text or None, 'tool_calls': invokes})
            self.record_trail.append({'kind': 'turn', 'text': body_text[:300], 'calls': [cq['function'] for cq in invokes]})
            comebacks = self.carry_on_invokes(invokes)
            for (index, (call_ref, prose)) in enumerate(comebacks):
                if index == len(comebacks) - 1:
                    prose += '\n' + self.state_line()
                self.dialogue.append({'role': 'tool', 'tool_call_id': call_ref, 'content': prose})

def agent_main(issue_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session = Sable(issue_data)
    except Exception as raised:
        return [{'kind': 'crash', 'error': f'{type(raised).__name__}: {raised}'[:400]}]
    try:
        return session.carry_on()
    except Exception as raised:
        session.record_trail.append({'kind': 'crash', 'error': f'{type(raised).__name__}: {raised}'[:400]})
        try:
            if not session.closed_out:
                session.close_out()
        except Exception as anew:
            session.record_trail.append({'kind': 'crash', 'error': f'{type(anew).__name__}: {anew}'[:400]})
        return session.record_trail
    finally:
        session.launched.halt()
__all__ = ['agent_main']