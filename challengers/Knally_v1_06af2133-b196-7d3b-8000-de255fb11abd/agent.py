from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from src.agent.proxy_client import ProxyClient
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
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 36000
GATE_ROUNDS = 3
AUTO_PAGES = 12
CHEAPER_OPENS = 3
FIRST_READS = 6
# long multi-item tasks: at least LONG_STEPS steps and LONG_ITEM_STEPS per requested product. Arena r18 world, ours b04/b07/b08
# against Halo b40 on its 74-120-step tasks: we ran 20-27 model turns of ~12 s and used 29-59 steps (1.3-2.2 steps a turn), Halo
# 50-61 steps in 17-21 turns (2.3-2.9) because its desk paged model searches and opened rows in code; 9 of the 27 paid lines Halo
# first saw sat on pages 2-4 (ours: 3 of 24 on page 2, none past it). No 37-step task meets the gate (2-3 products)
LONG_STEPS = 60
LONG_ITEM_STEPS = 15
# further pages of one model search (the store reached 4 pages of 10 rows), and auto pages per product on a long task
LONG_PAGES = 3
LONG_ITEM_PAGES = 9
# Desk row reads per episode on a long task (two per search as on short tasks)
LONG_FIRST_READS = 16
# on a long task Desk pages and reads only while this many seconds remain before CLOCK_STOP (two model turns)
LONG_CLOCK = 30
# price-capped category listing at the first checkout of a cheapest pick: needs this many steps left, this many
# seconds before the clock stop, and reads at most this many pages
CAPPED_ROOM = 18
CAPPED_CLOCK = 90
CAPPED_PAGES = 5
# the episode lasts 300 s; past CLOCK_STOP no model turn starts and the held cart is checked out (or the episode ended)
CLOCK_STOP = 268
EPISODE_SECONDS = 300
# before this many seconds the two question-plan samples are requested together (no call timeout is cut yet)
SIDE_BY_SIDE_UNTIL = 150
# a search paged this often without a new row of a requested kind holding its own words is not paged further
EMPTY_PAGES = 3
# model searches that listed no row and that Desk sends again with fewer words, per episode (at most two per item)
ZERO_RETRIES = 3
# search arguments filtering by seller or maker (matched on the argument name). Real r8-r14: after such a filter came
# back empty, the same filter with another name found rows 5 times in 581 (category filters 52%)
SELLER_ARG = re.compile('vendor|maker|brand|seller|dealer|merchant|manufactur', re.I)
# once the store refused an unbought close, a further such close is held back (twice at most)
NO_SECOND_END = True
# the engine profile follows the number of requested products (see SizeProfile)
SIZE_PROFILES = True
# question plan: a second PLAN sample from the other model; properties chosen by both go first
P1_ON = True
# question plan: past the first 14 rows the key table keeps every key seen on 2+ read listings (no purchase-option fields)
P2_ON = True
# question plan: with at most 2 messages in all, a hinted key with measure values does not take the first one
P3_ON = True
# adaptive L2 asking (11-08): a list names up to LIST_KEYS keys of one item, the first LIST_FRONT fresh; a grouped
# message names GROUP_KEYS keys per item (slot 3+ of a grouped message drew a need .15, real r15/r17)
LIST_KEYS = 8
LIST_FRONT = 3
GROUP_KEYS = 2
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
GUIDE = (
    "Your role: a shopping agent working for one shopper inside an online shop, able to act only through the shop's tools. Grading is all or nothing, and a run passes "
    'in exactly two ways.\n'
    '(a) An order goes through in which each requested item appears at the quantity asked for, every line meets all requirements in effect at the moment of checkout, '
    'and the order total does not exceed the budget.\n'
    '(b) The shop truly carries no listing meeting everything, and you finish without buying, in the way the request prescribes.\n'
    '\n'
    '== WORKING WITH THE SHOP ==\n'
    '- Thinking costs nothing; each tool call spends one step from a limited supply. Shops differ in what they call their tools, arguments and result fields, so study a '
    "tool's description and schema first and pass argument names and permitted values exactly as given.\n"
    '- Search may be restrictive: filtering by category, matching keywords only at the start of a title, returning few rows per page, handing back a token or offset for '
    'the next page. If the decision rests on all matches, check the reported total against the rows seen so far and go on paging with the token or offset exactly as '
    'returned. Keep queries short, with the wanted value first and the product noun at the end. An empty answer to a filtered search usually means a wrong filter value '
    'or parameter; try again with the label spelled the way the rows spell it, or filter on another parameter.\n'
    "- Search rows are only summaries. Shops divide a listing's record in different ways: one view can carry the attributes (properties), another just the offer plus "
    'description, and a comparison view can carry either. Once Desk has identified the tool that shows attributes, the ledger says which one. Property requirements are '
    'decided on those attributes, so read them for every listing you may buy; a single call covering several listings saves steps if it shows what you need. Never read '
    'a listing again while nothing about it has changed.\n'
    "- You do not write to the shopper directly; all questions travel via Desk's interviewer. Replies show up in the conversation and get copied into the ledger.\n"
    '- The request can demand intermediate steps (an approval, a message, a check, or reading something first); the ledger lists them. Desk carries out the ones due '
    'right before the order or the ending by itself when you call the checkout or ending tool, so neither wait for those nor give up the purchase over them. Any other '
    "such step is yours to do with the shop's tools (questions about products travel via Desk's interviewer).\n"
    '\n'
    '== WHEN A LISTING QUALIFIES ==\n'
    '- Product kind comes from the category label. Only a label matching the requested item as a whole counts, never a merely similar one. A label that itself carries a '
    "requirement's value (say an audience or season inside the label) meets that requirement.\n"
    "- A property requirement is met once the listing's attribute for that property gives the value. Key spelling is irrelevant (spaces, underscores, camelCase, "
    'prefixes) and one entry of a comma- or slash-separated list suffices. Values are compared word by word with case, singular versus plural and everyday synonyms '
    'ignored; a value containing the required words still counts ("deep teal" is teal, "extra roomy fit" is roomy). Different words do not: a code or abbreviation '
    'differs from the word it abbreviates, and "teal green" is not "sea green". Numbers are compared after unit conversion; a limit includes its end point; a range must '
    'lie wholly inside a limit.\n'
    "- The decisive attribute is whichever key names the requirement's property (by that name or a common synonym). Should it state something else, the listing fails, "
    'no matter what the title or other keys claim. Keys about some other property decide nothing, however near (a handle colour tells nothing about the body colour). '
    'Desk adds [evidence] lines after opened listings, naming keys that match a requirement.\n'
    '- Only where no attribute covers the property may the wanted value come from elsewhere: another attribute (for instance a features or styles list holding it), the '
    'title, or the description. A figure for some other quantity never qualifies (weight is not capacity), and neither does a bare number whose text leaves unsaid what '
    'it measures.\n'
    '- Certain requirements carry no value and instead point at another listing ("the same X as the listing that <shop> sells with <property>", "within N of the X shown '
    'on the page of ..."). Resolve those before anything else: find that listing (search by its category, brand, shop or words) and open its full page through the '
    'single-listing or side-by-side tool, since a search row is not enough; should that page in turn point to a further listing for the property, find and open it the '
    'same way. The value is read off the last page. Where that page states a rule pairing values of the property (one value goes with another), the requirement takes '
    "whichever value that rule pairs with the page's own value. An order is valid only when every page along the chain was opened. If the wanted value is anything other "
    "than that listing's value, the rule excludes only listings having exactly that value, and alone it is never a reason to end without a purchase.\n"
    '- If a listing you plan to buy refers to another page for more about itself, read that page first.\n'
    '- Where one requested item must fit another (an accessory for a device), choose the two together from what the listings themselves say (a list of compatible '
    'models, a model or series code present on both pages). When no accessory meeting its own requirements fits the chosen device, switch to another device that meets '
    'its requirements and that such an accessory fits, before any ending without a purchase.\n'
    '- Exclusions and closed lists bind. Wishes ("ideally", "would be nice") serve only to break ties.\n'
    '- A listing shown as sold out is never bought. The order total, meaning price times quantity added up over the lines, may not exceed the budget (being equal is '
    'fine).\n'
    '\n'
    '== PICKING AMONG QUALIFYING LISTINGS ==\n'
    '- Cheapest (lowest total, smallest price): the right line is the lowest-priced listing of the ENTIRE qualifying set. Build that set by paging through it (rows are '
    'seldom ordered by price, and Desk pulls further pages of a short result by itself), try candidates from the lowest price up, and take the first one that fully '
    'qualifies. When search takes a maximum price, use it: list the category under the price of your current best pick so only cheaper candidates remain.\n'
    '- Most or least of some measure: compare every qualifying listing on that measure and take the extreme; a listing silent on the measure cannot win.\n'
    '- Any other request: the first listing meeting everything is correct; spend no steps hunting for a better one.\n'
    '- The moment an opened listing meets every requirement of an item (for cheapest, most or least: once the qualifying set has been compared), cart it and move on. '
    'Cart lines often sell out soon afterwards, and carting early leaves steps for a replacement.\n'
    '- Each distinct requested item takes a cart line of its own at its own quantity.\n'
    '\n'
    '== CHANGES DURING THE RUN ==\n'
    '- Either the shopper or the shop may alter the task: a new budget, a requirement added or dropped, a new price, a listing sold out. Such changes come inside tool '
    "results, frequently as a refused checkout; a note counts whichever tool's result carries it, and the ledger keeps the requirements currently in force. A refusal is "
    'not the end of the run: apply every change, fix the cart and check out again.\n'
    '- Just before checkout, read the record of each chosen listing once more; price and stock may change without warning.\n'
    '\n'
    '== FINISHING ==\n'
    '- The cart must end up holding exactly the chosen lines (remove anything else); view it once, then check out in a separate call.\n'
    '- Ending without buying is allowed only after a real search turned up no listing meeting everything. Being short of steps proves no such thing; in that situation '
    'check out the best cart you have. Phrase that ending as the request asks, and if a remark is needed keep it short and name no listing.\n'
    '- Each reply needs at least one tool call. Keep your own words few: what you know, then your next move.'
)
BRIEF = (
    "From the shopper's request and the shop's tool list given below, build the task sheet a shopping agent will follow. The answer is a single compact JSON object only "
    '(no indentation, nothing before or after it) with these fields.\n'
    '\n'
    'PRODUCTS\n'
    '"items" - one entry per distinct product: {"name": what kind of product it is, in the request\'s wording, "shelf": the category label exactly as a tool lists it, '
    'or "" when no tool lists labels, "quantity": how many whole units of that one listing (1 when unstated; a pair or set sold as one listing counts as 1), "musts": '
    'the firm requirements, each written "property: constraint" carrying the exact value, limit, exclusion or closed list}.\n'
    'Rules for musts:\n'
    '- Name the property with the word the request uses ("a 12-inch blade" -> "blade length: 12 inch"; "meant for a lakeside occasion" -> "occasion: lakeside"); do not '
    'substitute a neighbouring property.\n'
    '- When a value has to be read off another listing, write exactly ONE must that carries the full description of that listing ("rim colour: unlike the rim colour of '
    'the kettle Ovarno sells with a glass lid"); the shop, the kind and any trailing "with <property> <value> and <property> <value>" of that listing remain inside it '
    'and never become musts of their own.\n'
    '- Never a must: price, budget, stock, quantity.\n'
    '- Background story (who uses it, where, earlier trouble, the reason for buying) gives no must unless it clearly supplies a property value or limit; words forming '
    'part of the product\'s name (a season or audience said together with the product) go into "name".\n'
    '- When a use or occasion is turned into a property of the product ("blazers for business wear", "a dress made for parties", "the occasion has to be party"), it is '
    'a must on that property ("occasion: business"), except when the shop\'s category label already carries it; the event, person or place it is meant for ("to wear at '
    'his wedding", "for a colleague") and how it should sit, feel or look while used are only context.\n'
    '- A single "and"-joined phrase naming products (one "I want ..." with one quantity, frequently the name of a shop department) is ONE item; split into several items '
    'only where each has a quantity or requirements of its own.\n'
    '\n'
    'TOOLS\n'
    '"roles" - for each job, the exact tool name that does it, or null when none does: "search"; "open" (the attributes or properties of one listing; given two '
    'single-listing tools, take the one that promises attributes, properties or specifications); "offer" (the present offer of one listing - price, stock - when a tool '
    'other than "open" supplies it); "compare" (several listings shown in full); "ask" (a message to the shopper); "cart" (alters the cart); "cart_view" (displays the '
    'cart); "buy" (checkout); "refuse" (finishes without a purchase).\n'
    '"buy_args" - fixed arguments that make the "buy" tool carry out the checkout (its mode, if the tool has a mode argument), otherwise {}.\n'
    '"refuse_args" - fixed arguments that make the "refuse" tool finish without a purchase, otherwise {}.\n'
    '\n'
    'MONEY AND CHOICE\n'
    '"budget" - the highest total spend, a number in the request\'s unit; null when no budget is given.\n'
    '"budget_unit" - that unit in words (such as "cents") or null.\n'
    '"budget_scope" - "total" for a limit on the entire order, "each" for a per-item limit, null for no limit.\n'
    '"in_stock_only" - true, except when unavailable items are clearly acceptable.\n'
    '"pick" - "cheapest" (the lowest price or smallest total is wanted), "most" / "least" (an extreme of some measure), otherwise "any"; remarks that money is short set '
    'no pick rule.\n'
    '"pick_measure" - with most/least, the measure to push up or down; otherwise "".\n'
    '\n'
    'LIMITS AND CONDUCT\n'
    '"messages" - how many messages (whole number) the agent may send to the shopper, or null.\n'
    '"steps" - how many steps / moves / tool calls (whole number) are allowed, or null.\n'
    '"lookups" - those musts (copied exactly) whose value must first be read from another listing the request refers to; otherwise [].\n'
    '"rules" - every instruction about how to work or how to finish, at most 12 words each.\n'
    '"wishes" - things that would merely be nice.\n'
    '"ask_first" - points to settle with the shopper before choosing, only when the request explicitly asks for it; otherwise [].\n'
    '"process" - actions required beyond finding and buying the right products (approval by the shopper, telling or showing the shopper something, checking something '
    'with them, reading something first, using a particular tool), each {"step": the action in 15 words at most, "when": "choose" (before products are chosen) | "order" '
    '(right before checkout) | "end" (right before ending without a purchase), "messages": shopper messages it takes (0 when none)}; otherwise []. Item requirements, '
    'the budget, stock, message or step limits and the nothing-fits ending are not steps.\n'
    '"quiet_decline" - true when alternatives may not be named or offered if nothing fits, false when they are wanted, null when the request says nothing.\n'
    '"fits" - [{"item": item number, "with": item number}] for every requested item said to fit, suit or work with ANOTHER requested item (chosen by the agent); '
    'otherwise [].\n'
)
PLAN = (
    'Choose which properties a shopping agent should raise with the shopper.\n'
    'How shoppers behave: every shopper keeps a few firm wishes the request leaves out, and each wish belongs to one property of one requested product. A wish comes out '
    'only when a question names exactly its property. A property the shopper does not care about gets "don\'t mind"; one they never thought about gets confusion; and '
    'each property named costs part of a message.\n'
    'What you are given: the request; the items, each with what the request already settles for it; and per item category, a numbered list of attribute keys found on '
    'opened listings, with how many listings show each key and a few sample values. A hidden wish can also be about a property missing from that list.\n'
    '\n'
    'Order of preference:\n'
    '1. Whatever the request tells the agent to confirm with the shopper (the ASK FIRST section, when present), stated as the property it is about.\n'
    '2. Hints in the story. A story often points at a hidden wish without giving its value away - a comment about something the shopper cannot judge or leaves to you, '
    'another person whose taste decides, how the product or its place looks, who is going to use or wear it, or what goes inside it. For each hint take the one property '
    "it points at, with its item, keeping the story's order. Favour a listed key of that item that fits the hint (wearer -> a gender/audience key; looks -> the listed "
    "colour, pattern or finish key carried by most of its listings). A hinted property carried by none of the item's listings comes after the three keys that item's "
    'listings carry most.\n'
    '3. After that, the listed keys in order: the properties buyers of this product weigh first (usually type/style, form/fit, size/capacity/dimensions, material, '
    "colour/pattern - whatever drives the choice) and the keys most of the item's listings carry; once 5 or more listings have been read, skip keys found on only 1-2 of "
    'them.\n'
    'Leave out: properties the request already settles for that item (named in a requirement - though a key whose values only echo a required value, such as a pattern '
    'key showing the required decoration, is NOT settled), alternative names of a property already chosen, keys whose values all list several purchase options, '
    'identifiers, model names, warranty and other logistics. Pack sizes, counts and per-pack quantities are normal properties; a key tagged [not the stated ...] is not '
    'the property of that requirement, however similar the names. Rank each item on its own: hints first, listed keys after.\n'
    'Put only the bare property name in "label"; for a listed key spell out every word plainly ("lid colour" for lidColour, "rim finish" for rim_finish - never '
    'shortened to "colour"/"finish"); just the name, without sample values, brackets or a second property.\n'
    'At most @N@ picks per item, strongest first. The answer is one JSON object and nothing more: {"ask": [{"item": item number, "label": "...", "hint": "up to 8 words '
    'of the story it came from, or an empty string"}]}, with each item\'s picks in rank order'
)
HEAR = (
    'A shopping agent put questions to a shopper; both the questions and the replies follow below. Convert the replies into a single JSON object and output nothing '
    'besides it:\n'
    '{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value must be looked up on a different listing, else false}],\n'
    ' "drop": [{"item": item number, "property": "a requirement the shopper gives up"}],\n'
    ' "free": [{"item": item number, "property": "..."}],\n'
    ' "crowded": [numbers of the questions whose reply complained that too much was asked in one go, or wanted one question at a time - a plain "I do not understand" is '
    'not such a reply],\n'
    ' "budget": the new overall limit as a number, else null}\n'
    'How to fill it:\n'
    '- needs: one entry per firm wish, with its exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"), filed under the property the reply names '
    '(which may differ from the one asked about); each property gets one need only, even if more than one reply touches it. Values are copied word for word, nouns and '
    'units too ("at least 3 shelves", never "at least 3"). For a plainly stated value keep only the wanted part, without emphasis or excluding-everything-else phrasing '
    '("it must be oval, oval only" turns into "oval"); limits and explicit exclusions stay ("at least 30 cm", "not wicker").\n'
    "- If a reply sets a value by reference to another listing (like, unlike or close to the value of some shop's product that it identifies by a few properties), write "
    'ONE need under the property the reply names, set "points": true, and use the reply\'s own wording as the value, from the comparing word through to the end of that '
    'description, together with anything it says about a listing named on that page; do not shorten or paraphrase. Properties that only describe the other product are '
    'not needs of this item.\n'
    '- free: for each question, the property the shopper does not care about.\n'
    '- A value said in a casual or roundabout way is still a stated value.'
)
NOTE = (
    'While the task was under way, the shop showed the message quoted below; its author may be the shopper or the shop. Decide what sort of message it is and extract '
    'each change it carries.\n'
    'The answer is one JSON object and nothing more:\n'
    '{"kind": "change" (some part of the task shifted - budget, a requirement, a price, stock, or a refused order), "error" (just a faulty or rejected call) or '
    '"other",\n'
    ' "budget": the new overall limit as a number in the request\'s unit, else null,\n'
    ' "add": [{"item": item number, "need": "property: constraint"}],\n'
    ' "drop": [{"item": item number, "need": "the requirement that stops applying"}],\n'
    ' "gone": [ids the message calls unobtainable],\n'
    ' "sold_out": true if the message says a listing already chosen (named or not) is sold out, gone or unavailable,\n'
    ' "price_moved": true if the message says a price has changed,\n'
    ' "summary": "one short sentence"}\n'
    'Miss no change. A requirement replaced by another appears twice: the old one in drop, the new one in add.'
)
CHECK = (
    'You are the final review before a shopping agent pays. Decide, from the records shown and nothing else, whether the cart is exactly right.\n'
    'The cart is right when it holds one line of its own for every requested item at the requested quantity and nothing else, no line is shown out of stock, and price x '
    'quantity summed stays within the budget. If a listing has more than one record, the latest one applies.\n'
    'Reading the records:\n'
    "- The attribute for a requirement's property meets it when it gives the value; key spelling does not matter, one member of a list suffices, case, singular/plural "
    'and common synonyms are fine, and so is a value that contains the required words ("deep teal" for teal); convert units, treat limits as including their end points, '
    'and require a range to sit wholly inside.\n'
    "- Whatever the attribute keyed by the requirement's property says is final - a different value there fails the listing; keys for other properties, related or not, "
    'settle nothing (handle colour vs body colour). When several keys spell the same property (disregarding case, spaces, underscores, punctuation), one that gives the '
    'value decides, whatever the others say; a key about another aspect of the same thing never overrules it. A bare figure under that key supplies no word value, '
    'whatever the title or description claim.\n'
    '- Only where no attribute covers the property does the wanted value count from another attribute, the title or the description; a figure of another kind does not '
    '(weight vs capacity), nor an unlabelled bare number.\n'
    '- A category label stating a requirement meets it; the category label has to match the requested item as a whole.\n'
    '- "unlike another listing\'s value" holds unless the records show precisely that value; a value taken from another listing\'s page is the one that page\'s pairing '
    "rule sets against the page's own value, if the page gives such a rule for the property.\n"
    "Under pick rule cheapest / most / least, weigh the other listings shown as well: if one clearly meets everything and beats the cart's choice on that rule, the cart "
    'is wrong.\n'
    'A cart listing\'s page may refer the reader to another listing, by its code, for the rest of its own details; list such codes under "refs" (not codes of devices or '
    'parts it merely fits, works with or replaces).\n'
    'The answer is one JSON object and nothing more: {"ok": true or false, "problems": ["short concrete problem", ...], "better": [ids of listings that should replace a '
    'cart line, best first], "refs": [codes]}. Set ok=false only for a problem the records plainly show, or for a requirement no record covers.'
)
READING = (
    'the key naming the property is what decides it - how the key is spelled does not matter, one list member suffices, synonyms and converted units count, and so does '
    'a value containing the required words ("deep teal" for teal); limits include their end points; where several keys spell the same property, one of them giving the '
    'value is decisive; a bare figure under such a key never supplies a word value. Keys about other properties settle nothing, related ones included (handle colour vs '
    'body colour). Only if no such key exists may the wanted value come from another attribute, the title or the description (an unlabelled bare number never does). A '
    'category label stating a requirement meets it, and the category label has to match the item. "unlike another listing\'s value" holds unless the records show '
    "precisely that value. A value taken over from another listing's page is the one that page's pairing rule matches with the page's own value, if the page gives such "
    'a rule for the property. A listing shown sold out or over budget never fits.'
)
AUDIT = (
    'The shopping agent is about to finish without buying, believing nothing fits. Go over the listing records below against the requirements one more time. How to read '
    'the records: ' + READING + '\n'
    'A ROW record is merely a search-result row whose page (attributes) has not been read; any property missing from the row counts as neither met nor failed.\n'
    'Those requirements came from a hurried first reading, so check each REQUEST requirement against the REQUEST wording first. Put under "not_firm" any that '
    'merely tells the backstory (the user, the place, the occasion or purpose, an earlier mishap, the feel wanted) without the request asking it of the product. '
    'Put under "corrected" any whose value carries connecting words besides the value, giving the bare value for the same property. Needs from the shopper\'s '
    'answers or from store notes always stand. Judge the records leaving out the not_firm ones and using the corrected values. A requirement the request plainly '
    'states stays even if nothing meets it; finishing without a purchase is then the right call.\n'
    'The answer is one JSON object and nothing more: {"not_firm": [{"item": item number, "need": the requirement exactly as listed}], "corrected": [{"item": '
    'item number, "need": the requirement exactly as listed, "value": "property: bare value"}], "fits": {"item number": [ids that appear to meet every '
    'requirement, cheapest first, no more than 4]}, "open": [ids of ROW listings of a requested kind, within budget, whose page might prove to fit, most likely '
    'first, no more than 6], "search": [no more than 3 of {"item": item number, "query": "a two- or three-word search not yet run, such as another name for '
    'the product plus one required value; long searches seldom match"}], "reason": "one sentence"}'
)
CLOSEST = (
    'No offer meets every requirement, so the shopping agent is going to finish without buying, and the request asks for the nearest candidate to be named. For each '
    "listing record shown, state which requested item it is a case of and which of that item's current requirements the record does not show as met. How to read the "
    'records: ' + READING + '\n'
    'Leave stock and price aside; the agent deals with those. The answer is one JSON object and nothing more: {"listings": [{"id": listing id, "item": number of the '
    'requested item it is a case of (0 if none), "misses": [the property name of each requirement not shown as met]}]}'
)
UNMIX = (
    "The requirements below come from a shopper's request. A few take their value from another listing the request refers to, and the words that identify that other "
    'listing (its kind, the shop selling it, the "with ..." properties after it) sometimes end up extracted as if they were requirements of the product to buy.\n'
    "Going by the request's wording, name the OTHER REQUIREMENTS below that only describe that other listing.\n"
    'The answer is one JSON object and nothing more: {"describe_other": [those requirements, copied exactly character by character; an empty list if every one truly '
    'belongs to the product]}'
)
TRAIL = (
    "Some requirements have no value of their own but take it from another listing, and that listing's page can again refer to a further listing for the property. The "
    'shopping agent is about to check out, yet its order is valid only once every listing along each such chain has been opened in full. Follow the chain of each '
    "requirement one link at a time - starting from the requirement's own description, then the pages' words - with the records below (OPENED = a page the agent opened; "
    'ROW = just a search-result row, its page unread).\n'
    'A record counts as a link only if it matches the WHOLE description: shop or brand, product kind AND every stated property value (a described two-pack, for example, '
    'rules out the same shop\'s single unit). Without a full match the link is not found, even if that shop sells other listings of that kind; give it a "find": the '
    "shop name plus the description's values as a few search words.\n"
    "Give each chain's links in order, stopping at the first link that is not OPENED or not found - its words are still unknown.\n"
    'A page whose own words send the reader on to a different listing for the property (the value is to be read there, not from this item) can never be the final link, even '
    'when it also gives a value for the property: the listing it names comes next. OPENED records tagged [sends on] are pages of that sort.\n'
    "Copy the value completely, exactly as the final page gives it. That is the page's own value, unless the request's own words for the requirement (REQUEST'S WORDS "
    'below) call for a counterpart of it - a value that pairs with or complements it instead of the same one: only in that case, when the page gives a rule pairing '
    "values of the property, is it the value the rule pairs with the page's own value. Words that ask for the same, an identical or a matching value, or just for what "
    "that page says, take the page's own value.\n"
    'The answer is one JSON object and nothing more: {"chains": [{"need": the requirement, verbatim, "links": [{"id": the listing id if one of the records below is this '
    'link, else "", "about": how the requirement or the page before describes this listing, "find": "for a link not found: search words (its shop and the description\'s '
    'values)"}], "done": true if the final link is an OPENED page that itself gives the value of the property (also when the requirement asks for a value different from '
    'or unlike it - that page\'s value is the reference), else false, "value": the value the final page gives, as explained above, or ""}]}'
)
GATE = (
    "Before a shopping agent may @ACT@, the shopper's request demands the steps listed below. For each step due before it can @ACT@, decide from the numbered RECORD of "
    "the agent's actions whether that step is done for the cart as it stands now. A step is done when some RECORD line shows it done; for a step about the cart or the "
    'order, that line must also come after the latest cart change, and the reply to it must grant it. A question about product properties never completes a step.\n'
    'For a step not yet done, give the single store call that carries it out next: a tool named in TOOLS with its arguments. Whatever goes to the shopper uses the '
    'messaging tool, as one short sentence or question about that step only, in plain words, without product properties or values the step does not need. Give no call '
    'when every due step is done, or when the messages or steps left cannot carry out the missing one.\n'
    'The answer is one JSON object and nothing more: {"steps": [{"step": the step, "done": true or false, "lines": [numbers of the RECORD lines that show it done]}], '
    '"call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: '
    'constraint"} for each firm requirement such a reply states]}'
)
SWAP = (
    "One listing in the shopping agent's cart has sold out. Look through the other listing records for replacements. How to read the records: " + READING + '\n'
    'The answer is one JSON object and nothing more: {"fits": [{"id": listing id, "price": its price as a number, copied from the record} - one entry for each listing '
    'that appears to meet every requirement of the item the sold-out line served, no more than 6], "reason": "one sentence"}'
)
SHAPE = (
    "A store tool returned the text below. Describe how it is built, naming every part by a key that appears literally in the text or among the tool's parameters.\n"
    'The answer is one JSON object and nothing more:\n'
    '{"rows": the key of the list holding the listings,\n'
    ' "count": the key for the total number of matches (not the number on this page), else null,\n'
    ' "next": the key whose value fetches the next page, else null,\n'
    ' "next_arg": the tool parameter taking that value, else null,\n'
    ' "price": the key within a row holding the price, else null,\n'
    ' "stock": the key within a row telling whether the listing can be bought now, else null}'
)
GUARD_KEYS = 3
GUARD_SHARE = 0.0
GUARD_GAP = 0.34
# a key whose shown values are mostly lists of several phrases (a features or contents field), judged by shape alone
LIST_PARTS, LIST_PART_CHARS, LIST_VALUE_CHARS = 3, 10, 60
NOT_PROPERTY = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
# NOT_PROPERTY's purchase-option and identifier part (model, condition and video stay in the table)
TABLE_SKIP = '\\b(?:options?|variants?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|url|links?|images?|photos?)\\b'
TABLE_FIRST = 14
# a key naming the purchase options or variants a listing comes in (fit reviews leave such keys out)
CHOICE_KEY = '\\b(?:options?|variants?)\\b'
# a number followed by at most a short unit word (shape only)
MEASURE_VALUE = '^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$'
WIDEN_EACH = 6
FIT_CODE = '[A-Z][A-Za-z]{0,5}[ -]?\\d{3,6}'
# a model family written with trailing x's ("every QX12xx unit"): any code of that prefix and length
FIT_WILD = '[A-Z][A-Za-z]{0,5}[ -]?\\d{1,5}[xX]{1,4}'
PART_CODE = '[A-Z]{1,4}\\d{3,6}'
BOUND_WORDS = '\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|same|than|like|either|unless|if|only)\\b'
# grammar endings stripped when two spellings of one word are compared
INFLECTIONS = ('s', 'es', 'ed', 'd', 'ing', 'er', 'ers', 'est', 'ies', 'ied', 'ly')
# inflection endings (grammar, not vocabulary): a word and its -ing/-ed/-s forms are one word
FORM_ENDS = ('ing', 'ies', 'ied', 'ed', 'es', 's', 'e', 'd')
NUMERAL = '\\s*-?\\d+(\\.\\d+)?\\s*'
SHOP_FIELD = 'seller|shop|store|vendor|brand|maker|merchant'
LIMIT_ARG = 'max|most|under|below|limit|cap|upto|up_to'
TITLE_FIELD = 'title|name|heading'
_NO = object()

def model_codes(text: Any, wild: bool = False) -> list[str]:
    """Model codes in a text, normalised."""
    pattern = FIT_WILD + '|' + FIT_CODE if wild else FIT_CODE
    codes = []
    for hit in re.findall('\\b(?:' + pattern + ')\\b', str(text or '')):
        codes.append(re.sub('[\\s-]+', '', hit).lower())
    return codes

def codes_named(marks: list[str], known: set[str]) -> set[str]:
    """Known codes a set of fit marks names."""
    hits: set[str] = set()
    barred: set[str] = set()
    for mark in marks:
        if mark.startswith('!'):
            barred.add(mark[1:])
        elif not mark.endswith('x'):
            hits.add(mark)
        else:
            # a wildcard mark ("ab12xx") names each known code of its length with digits in the x places
            stem = mark.rstrip('x')
            hits.update(code for code in known if len(code) == len(mark) and code.startswith(stem) and code[len(stem):].isdigit())
    return hits - barred

def fit_codes(text: str) -> list[str]:
    """Model codes a page gives as its own name: after "model no." first, then after "also sold as"."""
    model_rx = '\\bmodel\\s*(?:(?:no\\.?|number|#)\\s*[:.]?|:)\\s*(' + FIT_CODE + ')\\b'
    sold_as = '\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)'
    return [code for rx in (model_rx, sold_as) for hit in re.finditer(rx, text, re.I) for code in model_codes(hit.group(1))]

def fit_works(text: str) -> list[str]:
    """Codes a page says it works with ('!' before excepted ones), then the part series its own code belongs to."""
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits|goes with|for use with)\\s*:?\\s+([^.;|]+)', text, re.I):
        if re.search('\\bpart\\s+' + PART_CODE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            continue
        pieces = re.split('\\b(?:except|excluding|but not)\\b', hit.group(1), maxsplit=1, flags=re.I)
        works += model_codes(pieces[0], True)
        works += ['!' + code for code in model_codes(''.join(pieces[1:]))]
    # a part whose own code is a series, a dash and a number ("[XY5-1234]") carries that series as a mark, matching a
    # device that takes the series (Arena r16: sofa / cover pairs)
    works += ['series:' + found.lower() for found in re.findall('[\\[(]\\s*([A-Z]{1,5}\\d{1,3})-\\d{3,6}\\s*[\\])]', text)]
    return works

def fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    upper_of = lambda rx: next((hit.group(1).upper() for hit in [re.search(rx, text, re.I)] if hit), '')
    also = [(part.upper(), model_codes(listed))
            for part, listed in re.findall('\\bpart\\s+(' + PART_CODE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    # a device taking a series of parts ("Takes part series XY5") shares that series mark with each such part
    takes = ['series:' + found.lower() for found in re.findall('\\btakes\\s+(?:parts?\\s+)?(?:of\\s+)?(?:the\\s+)?series\\s+([A-Z]{1,5}\\d{1,3})\\b', text, re.I)]
    return {'takes': takes, 'own': upper_of('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE + ')\\s*[\\])]'), 'codes': fit_codes(text),
            'works': fit_works(text), 'also': also,
            'more': upper_of('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE + ')')}

def dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, limit, depth + 1) for v in value]
    return value

def parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', body, re.S)
    if fenced:
        body = fenced.group(1)
    start, end = body.find('{'), body.rfind('}')
    # the whole text first, then the span from its first brace to its last
    pieces = [body, body[start:end + 1]] if 0 <= start < end else [body]
    for piece in pieces:
        try:
            got = json.loads(piece)
        except ValueError:
            continue
        return got if isinstance(got, dict) else None
    return None

def plain_key(key: str) -> str:
    spaced = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', spaced).strip().lower()

def stems(text: str) -> frozenset[str]:
    trim = lambda w: w[:-1] if len(w) > 3 and w.endswith('s') else w
    return frozenset(map(trim, plain_key(text).split()))

def roots(text: str) -> set[str]:
    found = set()
    for word in plain_key(text).split():
        # plural endings off: -es after a sibilant, a lone -s otherwise (never the -ss of "glass")
        if re.search('(?:ch|sh|ss|x|z)es$', word):
            word = word[:-2]
        elif len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
            word = word[:-1]
        if len(word) >= 3 and word not in ('and', 'for', 'the', 'with'):
            found.add(word)
    return found

def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop <= key or (len(key) >= 2 and key <= prop)

def worded(text: str) -> bool:
    """Search text holds words (not a bare code)."""
    text = str(text or '')
    if not re.search('[a-z]', text, re.I):
        return False
    if not re.search('\\d', text):
        return True
    return any(re.fullmatch('[a-z]+', token.strip('.,;:!?"\'()[]'), re.I) for token in text.split())

def plain_words(text: Any) -> list[str]:
    words = re.findall('[a-z0-9]+(?:\\.[0-9]+)?', str(text).lower())
    return [w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in words]

def spread(value: str, stated: Any) -> str:
    """'phrase', 'apart' or '' for a multi-word value."""
    want = plain_words(value)
    if len(want) < 2 or re.search(BOUND_WORDS, value, re.I):
        return ''
    width = len(want)
    parts = [plain_words(p) for p in re.split('[,/;|]', str(stated))]
    for words in parts + [plain_words(stated)]:
        if any(words[i:i + width] == want for i in range(len(words) - width + 1)):
            return 'phrase'
    wanted = set(want)
    return next(('apart' for words in parts if wanted <= set(words)), '')

def word_bases(word: str) -> dict[str, str]:
    bases = {word: ''}
    for end in FORM_ENDS:
        if not word.endswith(end) or len(word) - len(end) < 3 or (end == 's' and word.endswith('ss')):
            continue
        base = word[:-len(end)]
        if end in ('ies', 'ied'):
            base += 'y'
        bases.setdefault(base, end)
    return bases

def consonants(word: str) -> str:
    return re.sub('(.)\\1', '\\1', re.sub('[aeiou]', '', word))

def form_kind(a: str, b: str) -> str:
    """'same', 'form', 'vowel' or '' for two words."""
    if a == b:
        return 'same'
    if re.search('\\d', a + b):
        return ''
    left, right = word_bases(a), word_bases(b)
    for base in left.keys() & right.keys():
        # a bare final e is no inflection by itself (plan / plane): one side must carry a real ending
        if len(base) >= 4 and {left[base], right[base]} - {'', 'e'}:
            return 'form'
    for x, y in ((a, b), (b, a)):
        if not x.endswith('en') or y.endswith('en') or min(len(x), len(y)) < 5 or x[0] != y[0]:
            continue
        core = consonants(x[:-2])
        if len(core) >= 2 and core == consonants(y[:-1] if y.endswith('e') else y):
            return 'vowel'
    return ''

def joined_forms(want: list[str], got: list[str], kinds: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """Word-by-word match allowing joins and forms."""
    # depth first: at each word pair the same-or-form step is tried before a join, and the first full match wins
    pending = [(want, got, kinds)]
    while pending:
        want, got, kinds = pending.pop()
        if not want or not got:
            if not want and not got:
                return kinds
            continue
        head, other = want[0], got[0]
        kind = form_kind(head, other)
        options = [(want[1:], got[1:], kinds + (kind,))] if kind else []
        if not re.search('\\d', head + other):
            if len(other) >= 3 and len(head) - len(other) >= 3 and head.startswith(other):
                options.append(([head[len(other):]] + want[1:], got[1:], kinds + ('join',)))
            elif len(head) >= 3 and len(other) - len(head) >= 3 and other.startswith(head):
                options.append((want[1:], [other[len(head):]] + got[1:], kinds + ('join',)))
        pending.extend(reversed(options))
    return None

def same_in_other_form(value: str, stated: Any) -> bool:
    """Value stated in another spelling of its words."""
    pieces = '[a-z]+|[0-9]+(?:\\.[0-9]+)?'
    want = re.findall(pieces, value.lower())
    if re.search(BOUND_WORDS, value, re.I) or len(''.join(want)) < 4:
        return False
    text = str(stated)
    if any(re.findall(pieces, part.lower()) == want for part in re.split('[,/;|]', text)):
        return False
    # only the whole attribute: a list of several values does not state one of them
    kinds = set(joined_forms(want, re.findall(pieces, text.lower())) or ())
    if not kinds - {'same'}:
        return False
    return 'vowel' not in kinds or bool(kinds & {'same', 'join'})

def restore_value(request: str, must: str) -> str:
    """Give back a value word the reader cut short."""
    prop, sep, value = must.partition(':')
    words, named = (re.findall('[a-z0-9]+', part.lower()) for part in (value, prop))
    if not (sep and named and words) or len(words) > 3 or re.search('\\d', value) or re.search(BOUND_WORDS, value, re.I):
        return must
    tokens = re.findall('[a-z0-9]+', request.lower())
    width = len(named)
    # the three request words after each place the property's own words appear
    windows = [tokens[i + width:i + width + 3] for i in range(len(tokens) - width + 1) if tokens[i:i + width] == named]
    completions = lambda word, near: {t for t in near if t.startswith(word) and t[len(word):] not in INFLECTIONS}
    fixed = value
    for word in (w for w in words if len(w) >= 3):
        # the first window that already holds the word, or that grows it in exactly one way, settles it
        settled = next((near for near in windows if word in near or len(completions(word, near)) == 1), None)
        if settled is not None and word not in settled:
            fixed = re.sub('\\b' + re.escape(word) + '\\b', completions(word, settled).pop(), fixed, count=1, flags=re.I)
    return must if fixed == value else prop + sep + fixed

def type_names(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

TYPE_TESTS = {'string': lambda v: isinstance(v, str), 'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
              'number': is_number, 'boolean': lambda v: isinstance(v, bool), 'array': lambda v: isinstance(v, list),
              'object': lambda v: isinstance(v, dict), 'null': lambda v: v is None}

def to_integer(v: Any) -> Any:
    if isinstance(v, float):
        return int(v) if v.is_integer() else _NO
    if isinstance(v, str) and re.fullmatch(NUMERAL, v):
        return int(float(v))
    return _NO

def to_number(v: Any) -> Any:
    if not (isinstance(v, str) and re.fullmatch(NUMERAL, v)):
        return _NO
    figure = float(v)
    return int(figure) if figure.is_integer() else figure

def to_boolean(v: Any) -> Any:
    if isinstance(v, str) and v.strip().lower() in ('true', 'false'):
        return v.strip().lower() == 'true'
    return _NO

# JSON-schema type -> how a near miss becomes that type
COERCE = {'integer': to_integer, 'number': to_number, 'string': lambda v: str(v) if is_number(v) else _NO,
          'boolean': to_boolean, 'array': lambda v: _NO if isinstance(v, list) or v is None else [v]}

def int_field(schema: dict[str, Any], key: str) -> Any:
    got = schema.get(key)
    return got if isinstance(got, int) else None

def conform_text(value: str, schema: dict[str, Any], where: str, faults: list[str]) -> str:
    longest = int_field(schema, 'maxLength')
    if longest is not None:
        value = value[:longest]
    shortest = int_field(schema, 'minLength')
    if shortest is not None and len(value) < shortest:
        faults.append(f'{where} is too short')
    return value

def conform_number(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    low, high = schema.get('minimum'), schema.get('maximum')
    if is_number(low) and value < low:
        faults.append(f'{where} is below {low}')
    if is_number(high) and value > high:
        faults.append(f'{where} is above {high}')
    return value

def conform_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least = int_field(schema, 'maxItems'), int_field(schema, 'minItems')
    if most is not None and len(value) > most:
        faults.append(f'{where} allows at most {most} entries')
    if least is not None and len(value) < least:
        faults.append(f'{where} needs at least {least} entries')
    inner = schema.get('items')
    if not isinstance(inner, dict):
        return value
    return [conform(v, inner, f'{where}[{i}]', faults) for i, v in enumerate(value)]

def conform_record(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    props = schema.get('properties')
    if not isinstance(props, dict):
        props = {}
    if schema.get('additionalProperties') is False:
        for name in value:
            if name not in props:
                faults.append(f'{where}.{name} is not an allowed argument')
    for name in schema.get('required') or []:
        if name not in value:
            faults.append(f'{where}.{name} is required')
    fitted: dict[str, Any] = {}
    for key, item in value.items():
        fitted[key] = conform(item, props[key], f'{where}.{key}', faults) if key in props else item
    return fitted

# value test -> the fitter that checks a value of that kind against its schema, first test that holds wins
FITTERS = ((lambda v: isinstance(v, str), conform_text), (is_number, conform_number), (lambda v: isinstance(v, list), conform_list),
           (lambda v: isinstance(v, dict), conform_record))

def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Fit a value to a JSON schema, noting faults."""
    if not isinstance(schema, dict):
        return value
    kinds = type_names(schema)
    typed = lambda v: any(TYPE_TESTS.get(k, lambda _: True)(v) for k in kinds)
    if kinds and not typed(value):
        value = next((turned for turned in (COERCE[k](value) for k in kinds if k in COERCE) if turned is not _NO), value)
        if not typed(value):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    allowed = schema.get('enum')
    if isinstance(allowed, list) and value not in allowed:
        lowered = value.strip().lower() if isinstance(value, str) else None
        if lowered is not None:
            value = next((o for o in allowed if isinstance(o, str) and o.lower() == lowered), value)
        if value not in allowed:
            faults.append(f'{where}: allowed values are {allowed}')
    fitter = next((fit for test, fit in FITTERS if test(value)), None)
    return value if fitter is None else fitter(value, schema, where, faults)

def shaped(args: Any, schema: Any) -> dict[str, Any] | None:
    """Schema-fitted args, or None on any fault."""
    faults: list[str] = []
    fitted = conform(args, schema, 'args', faults)
    return None if faults else fitted

def units_after(have: int, count: int | None, removal: bool, sets: bool) -> int:
    """Line units after a cart call names it."""
    if removal:
        return have - count if count else 0
    step = 1 if count is None else count
    return step if sets else have + step

def free_text(values: list[str]) -> bool:
    """Values too free-form to ask about."""
    count = len(values)
    if not count:
        return True
    mean = sum(map(len, values)) / count
    if mean > 40 or 2 * sum(' / ' in v for v in values) >= count:
        return True
    if count < 4 or len(set(values)) != count or mean <= 12:
        return False
    vocab = lambda text: set(re.findall('[a-z]{3,}', text))
    overlapping = sum(1 for v in values if vocab(v) & vocab(' '.join(o for o in values if o is not v)))
    return 2 * overlapping < count

def cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat copy with cache marks on the ends."""
    if len(messages) < 2:
        return messages
    marked = [dict(m) for m in messages]
    for at in (0, -1):
        text = marked[at].get('content')
        if isinstance(text, str) and text:
            marked[at]['content'] = [{'type': 'text', 'text': text, 'cache_control': {'type': 'ephemeral'}}]
    return marked

def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if value and isinstance(value, (str, dict)) else []

def whole(value: Any) -> int | None:
    if isinstance(value, int):
        return None if isinstance(value, bool) else value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def pick_number(value: Any, size: int) -> int:
    """An entry's item number, clamped."""
    return min(max(1, whole(value) or 1), size)

def as_stated(record: Any) -> Any:
    """Record without purchase-option keys."""
    # real race 15: lines whose required value sat only in such a key (or there and as bare title
    # figures) failed 36 of 36
    if not isinstance(record, dict):
        return record
    stated = {}
    for key, value in record.items():
        if isinstance(value, dict):
            value = {a: b for a, b in value.items() if not (isinstance(b, str) and re.search(CHOICE_KEY, plain_key(a)))}
        stated[key] = value
    return stated

def id_like(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    bare = value.strip()
    if len(bare) < 3 or len(bare) > 64 or ' ' in bare:
        return False
    return named or bool(re.search('\\d', value))

def has_id_text(values: Any) -> bool:
    return any(id_like(v, True) and re.search('\\d', v) for v in values)

def pairs_block(value: str) -> dict[str, Any] | None:
    """A "name=value; ..." text as a block, or None."""
    pairs = [p.partition('=') for p in value.split('; ')]
    keyed = [bool(eq and name.strip()) for name, eq, _ in pairs]
    if len(pairs) < 2 or not pairs[0][1] or 2 * sum(keyed) <= len(pairs):
        return None
    entries: list[list[str]] = []
    for (name, _, rest), starts in zip(pairs, keyed):
        if starts:
            entries.append([name.strip(), rest.strip()])
        elif entries:
            entries[-1][1] += '; ' + name
    # a purchase-options key lists choices, not what this listing states (real race 15: order lines whose required
    # value sat only there failed 36 of 36)
    offered = lambda n, v: isinstance(v, str) and re.search(CHOICE_KEY, plain_key(n))
    return {n: 'choices offered, not stated specs: ' + v if offered(n, v) else v for n, v in dict(entries).items()}

def text_listing(text: str) -> dict[str, Any] | None:
    """A " | "-joined key: value line as a record."""
    if not isinstance(text, str) or text.count(' | ') < 2:
        return None
    record: dict[str, Any] = {}
    last = None
    for part in text.split(' | '):
        hit = re.match('^\\s*([A-Za-z_][A-Za-z0-9_ ]{0,29}):(?:\\s+|$)(.*)$', part, re.S)
        key = hit.group(1).strip() if hit else None
        if key is None or key in record:
            # a part without a new key belongs to the field before it; one before any field spoils the line
            if last is None:
                return None
            record[last] += ' | ' + part
        else:
            last = key
            record[key] = hit.group(2).strip()
    if len(record) < 3 or not has_id_text(record.values()):
        return None
    words = {'True': True, 'False': False, 'None': None}
    scalar = lambda raw: int(raw) if re.fullmatch('-?\\d{1,12}', raw) else float(raw) if re.fullmatch('-?\\d{1,12}\\.\\d+', raw) else raw
    typed = lambda raw, block: block if block is not None else words[raw] if raw in words else scalar(raw)
    return {key: typed(raw, pairs_block(raw)) for key, raw in record.items()}

def as_records(node: Any, depth: int = 0) -> Any:
    """Observation with text listings parsed."""
    if depth > 3:
        return node
    if isinstance(node, str):
        return text_listing(node) or node
    deeper = lambda v: as_records(v, depth + 1)
    if isinstance(node, list):
        return list(map(deeper, node))
    # a dict that itself names a listing in its text values stays as it came
    if not isinstance(node, dict) or has_id_text(v for v in node.values() if isinstance(v, str)):
        return node
    return {k: deeper(v) for k, v in node.items()}

def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    best, best_rank = None, None
    for key in rows[0]:
        named = 'id' in str(key).lower()
        values = [row.get(key) for row in rows]
        if not all(id_like(v, named) for v in values):
            continue
        trimmed = [v.strip() for v in values]
        if len(rows) > 1 and len(set(trimmed)) != len(trimmed):
            continue
        # an id-named key first, then the longest values on average; the first key wins a tie
        rank = (named, sum(map(len, trimmed)) / len(trimmed))
        if best_rank is None or rank > best_rank:
            best, best_rank = key, rank
    return best

def mentions_row(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, dict):
        return any(mentions_row(v, ident, depth + 1) for v in node.values())
    if not isinstance(node, list):
        return False
    for row in node:
        if isinstance(row, dict) and ident in [str(x).strip() for x in row.values() if isinstance(x, str)]:
            return True
    return any(mentions_row(v, ident, depth + 1) for v in node)

def listy(values: list[str]) -> bool:
    """Values shaped like feature lists."""
    if not values:
        return False
    shaped = 0
    for value in values:
        text = str(value)
        parts = [x.strip() for x in re.split('[,;|]', text) if x.strip()]
        if len(text) >= LIST_VALUE_CHARS or (len(parts) >= LIST_PARTS and sum(map(len, parts)) >= LIST_PART_CHARS * len(parts)):
            shaped += 1
    return 2 * shaped >= len(values)

def fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if not isinstance(value, dict) or len(value) < 2:
            continue
        # the largest nested dict of plain values (two or more), the first one on a tie
        block = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        if len(block) >= 2 and (best is None or len(block) > len(best)):
            best = block
    return best

def has_bound(value: str, extra: str = '') -> bool:
    return bool(re.search('\\b(?:not|no|other than|unlike|' + extra + 'at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I))

class ItemSpec:
    """A requested product and its requirements."""

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.added, self.dropped, self.free = [], [], []

    def live(self) -> list[str]:
        return [m for m in self.musts + self.added if m not in self.dropped]

class SizeProfile:
    """Message policy for one task size (1, 2, 3, 4+ items)."""
    # real races 10-13: one question plan wins at every size; the size only decides whether a message may
    # wait for a step due at an unbought ending (each product keeps one question and one more). An unsure
    # item count keeps the earlier behaviour

    def __init__(self, count: int, sure: bool):
        count = max(1, count)
        self.count = count
        self.size = '4+' if count > 3 else str(count)
        self.active = SIZE_PROFILES and sure

    def end_room(self, limit: int, held: int) -> bool:
        """Whether held messages may wait for an ending step."""
        return not self.active or limit - held >= self.count + 1

class Call:
    """One model tool call on its way through the desk."""

    def __init__(self, name: str, args: dict[str, Any], ending: bool, many: str | None):
        self.name = name
        self.args = args
        self.ending = ending
        self.many = many
        self.searching = False
        self.extra, self.chain, self.reread = '', '', ''

    def mark(self) -> str:
        return self.name + dumps(self.args)

    def listed(self) -> list[Any] | None:
        """Ids a side-by-side call names, if it is one."""
        ids = self.args.get(self.many) if self.many else None
        return ids if isinstance(ids, list) else None

class Episode:
    """Shared episode record plus its services."""

    def __init__(self, problem: dict[str, Any]):
        self._take_view(problem['environment'])
        self._clear_state()
        self.began = time.monotonic()
        self.profile = SizeProfile(1, False)
        self.llm, self.store = ModelDesk(self), StoreGateway(self)
        self.mem, self.scout, self.req, self.talk, self.news = ListingMemory(), SearchScout(), RequestReader(), Interviewer(), ChangeWatcher()
        self.judge, self.chains, self.basket = Checker(), ChainWalker(), CartDesk()
        self.gates, self.close, self.screen = Gatekeeper(), Closer(), Screener()
        self.loop = TurnLoop(self)

    def _take_view(self, environment: dict[str, Any]) -> None:
        """The binding, request, tools and step cap the policy view hands over."""
        self.binding = dict(environment['binding'])
        view = self.view = environment['policy_view']
        self.request = str(view.get('query') or '')
        usable = lambda t: isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)
        self.tools = list(filter(usable, view.get('tools') or []))
        blank = {'type': 'object', 'properties': {}}
        self.schemas = {}
        for tool in self.tools:
            name, params = tool['function']['name'], tool['function'].get('parameters')
            self.schemas[name] = blank if not isinstance(params, dict) else params
        cap = view.get('max_steps')
        steps = whole(cap)
        if steps is None and isinstance(cap, float):
            steps = int(cap)
        self.max_steps = 30 if not steps or steps <= 0 else steps
        self.finished = bool(view.get('finished'))

    def _clear_state(self) -> None:
        """Counters, ledgers and lookup tables at their starting values."""
        for name in ('step', 'version', 'msg_cap', 'msg_sent', 'msg_reserve', 'checkouts', 'end_refusals', 'side_ok', 'side_bad'):
            setattr(self, name, 0)
        self.cap_stated, self.refused_at = False, -1
        self.budget = self.confirmed = None
        self.brief, self.roles = {}, {}
        # the trace and running notes, leads, queued note readings and recent results sit among these lists
        for name in ('items', 'trace', 'process', 'notes', 'leads', 'queued', 'recent', 'refs', 'bound'):
            setattr(self, name, [])
        for name in ('lead_items', 'fact_tools', 'ref_values', 'ref_pages', 'cache', 'offer_seen', 'added_at', 'gone'):
            setattr(self, name, {})
        for name in ('page_seen', 'unavailable', 'opened', 'id_args', 'repick'):
            setattr(self, name, set())

    # -- clock, steps and messages --
    def elapsed(self) -> float:
        return time.monotonic() - self.began

    def past_stop(self) -> bool:
        """Past the clock stop: no new turns."""
        return self.elapsed() > CLOCK_STOP

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def clock_room(self) -> int:
        """Steps the clock still allows at this pace."""
        if self.step < 6:
            return 10 ** 6
        pace = max(3.0, self.elapsed() / self.step)
        return max(0, int((CLOCK_STOP - self.elapsed()) / pace))

    def free_messages(self) -> int:
        return self.msg_cap - self.msg_sent - max(0, self.msg_reserve - self.gates.sent)

    def stage_open(self, stage: str) -> bool:
        if all(p['when'] != stage for p in self.process):
            return False
        mark = self.cart_stamp() + f'#{self.version}'
        return (stage, mark) not in self.gates.cleared

    def reserve(self) -> int:
        """Steps held back for the close."""
        talk = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process)
        held = 1
        if talk and self.msg_sent < self.msg_cap and self.stage_open('order'):
            held += 1
        return held + self.offers_due() + self.refusal_room()

    def refusal_room(self) -> int:
        """Steps kept free to answer a refused first ending."""
        # Arena r16 noask b02/b03: 11-13 of 36-44 first checkouts were refused by a store change (sold out, new price or
        # budget); 4 came with no step left to answer. Two steps stay free for that answer until the first ending of
        # either kind was refused: real races 12-16, after a refused unbought ending, episodes that went on to act passed
        # 16-52% vs 4-17% for a second ending (ours ended again 13 times, 0 passed; the kept room had left only the ending)
        return 2 if self.checkouts == 0 and self.end_refusals == 0 and self.max_steps >= 30 else 0

    def page_cap(self) -> int:
        """Search pages Desk may read on its own in this episode."""
        # Arena r16 noask b10: in a 35-step four-item task Desk read 12 later pages of two searches, so the first listing
        # was opened at step 22 and later reads were refused for want of steps; real races 12-16: our episodes spending 30%+
        # of their steps on later pages passed 1 of 17
        return min(AUTO_PAGES, max(2, self.max_steps // 6))

    def offer_view(self) -> str | None:
        """A single-listing offer tool apart from the attribute view, or None."""
        tool = self.roles.get('offer')
        if tool and tool != self.roles.get('open') and self.id_named(tool):
            return tool
        # the brief can leave the offer role empty (Arena r12 b08: 36 of 66 briefs; 2 of 10 'process requirement missed'
        # orders never read a line's own page, KING v11 passed them reading it after carting): a single-listing tool no
        # role took, taking one id-named argument, is that page
        taken = set(filter(None, self.roles.values()))
        spare = []
        for name in self.schemas:
            if name in taken or name == self.attr_tool() or self.fact_tools.get(name) is True:
                continue
            if re.search('id|code|number|sku|listing|item|product', self.id_param(name) or '', re.I):
                spare.append(name)
        return spare[0] if len(spare) == 1 else None

    def unoffered(self) -> list[str]:
        """Cart lines whose offer page was not read since they were carted."""
        if not self.offer_view():
            return []
        due = []
        for ident in basket_ids(self):
            if ident not in self.gone and self.offer_seen.get(ident, -1) < self.added_at.get(ident, 0):
                due.append(ident)
        return due

    def offers_due(self) -> int:
        """Offer reads due before checkout: one for each item not yet carted and each cart line not yet read."""
        # Arena r16 noask b01: 15/15 'process requirement missed' orders checked out with a cart line whose offer page was
        # never read (tight step ends); every paid order had read each line's offer page after carting it
        if not self.offer_view():
            return 0
        uncarted = max(0, len(self.items) - len(basket_ids(self)))
        return min(4, len(self.unoffered()) + uncarted)

    def cart_stamp(self) -> str:
        return dumps(sorted(self.basket.lines.items()))

    # -- tools --
    def id_param(self, tool: str | None) -> str | None:
        """A single-listing tool's id argument."""
        if not tool:
            return None
        schema = self.schemas.get(tool) or {}
        needed = schema.get('required') or []
        texts = [k for k, v in (schema.get('properties') or {}).items() if isinstance(v, dict) and 'string' in type_names(v) and 'enum' not in v]
        # a required text argument, else the only text argument; any other required argument rules the tool out
        target = next((k for k in needed if k in texts), None)
        if not target and len(texts) == 1:
            target = texts[0]
        return target if target and all(k == target for k in needed) else None

    def side_param(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        arrays = [k for k, v in (schema.get('properties') or {}).items() if isinstance(v, dict) and 'array' in type_names(v)]
        target = arrays[0] if arrays else None
        if not target:
            return None
        return target if all(k == target for k in schema.get('required') or []) else None

    def side_room(self) -> int:
        props = (self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}
        cap = (props.get(self.side_param() or '') or {}).get('maxItems')
        # stores refuse fewer ids than their schema allows (real r8: every call of 6+ ids, r9/r12: 3+), so a size
        # once refused caps the schema's own limit
        if isinstance(cap, int):
            return cap if not self.side_bad else min(cap, max(1, self.side_bad - 1))
        if not self.side_bad:
            return max(4, self.side_ok)
        return max(2, min(self.side_bad - 1, max(self.side_ok, 2)))

    def page_tools(self) -> set[str]:
        single = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (single, self.roles.get('compare')) if t}

    def id_named(self, tool: str | None) -> bool:
        """True for a single-listing tool, not the search tool, whose only argument carries an id-like name."""
        if not tool or tool == self.roles.get('search'):
            return False
        return bool(re.search('id|code|number|sku|listing|item|product', self.id_param(tool) or '', re.I))

    def offer_tool(self) -> str | None:
        # Arena r13 b09: the brief set offer to the search tool; Desk's offer reads then searched listing ids as text
        offer = self.roles.get('offer')
        return offer if self.id_named(offer) else self.roles.get('open')

    def attr_tool(self) -> str | None:
        for tool, ok in self.fact_tools.items():
            if ok and self.id_param(tool):
                return tool
        tool = self.roles.get('open')
        if self.fact_tools.get(tool or '') is False or not self.id_param(tool):
            return None
        return tool

    def side_view_bare(self) -> bool:
        seen = self.fact_tools.get(self.roles.get('compare') or '')
        if seen is not None:
            return seen is False
        return self.fact_tools.get(self.attr_tool() or '') is True or bool(self.roles.get('offer'))

    def reading_tools(self) -> set[str]:
        return {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}

    def closing_tools(self) -> set[str]:
        return {tool for tool in (self.roles.get('buy'), self.roles.get('refuse')) if tool}

    def role_tools(self, *skip: str) -> set[Any]:
        """Tools of all roles except the skipped ones."""
        return {tool for role, tool in self.roles.items() if role not in skip}

    def mode_ok(self, role: str, name: str, args: dict[str, Any]) -> bool:
        """Call is the role's tool in the brief's mode."""
        if name != self.roles.get(role):
            return False
        fixed = self.brief.get(role + '_args')
        if not isinstance(fixed, dict):
            return True
        for key, wanted in fixed.items():
            if args.get(key) != wanted:
                return False
        return True

    # -- requirements and records --
    def live_needs(self) -> list[tuple[int, str]]:
        return [(number, need) for number, item in enumerate(self.items, 1) for need in item.live()]

    def open_pointers(self) -> list[str]:
        """Pointer requirements still unresolved."""
        live = set(need for _, need in self.live_needs())
        pending = []
        for _, need in self.refs:
            if need in live and need not in self.ref_values:
                pending.append(need)
        return pending

    def is_gone(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def render(self, observation: Any) -> str:
        text = dumps(clip(observation, SHOW_CHARS))
        return text[:OBS_LIMIT] + '...(cut)' if len(text) > OBS_LIMIT else text

    def log(self, kind: str, **fields: Any) -> dict[str, Any]:
        entry = {'kind': kind, **fields}
        self.trace.append(entry)
        return entry

    def ingest(self, name: str, observation: Any) -> str:
        """Absorb a result and read any note in it."""
        mem_absorb(self, observation)
        req_label_met(self)
        lines = news_free_lines(self, observation)
        meaning = news_read_note(self, lines) if lines else ''
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def cheapest(self) -> bool:
        return self.brief.get('pick') == 'cheapest'

    def price_rank(self, ident: str) -> float:
        price = mem_price(self, ident)
        return price if price is not None else float('inf')

    def numeric_budget(self) -> Any:
        return self.budget if is_number(self.budget) else None

def chat_body(model: str, messages: list[dict[str, Any]], max_tokens: int,
              tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The proxy request body: cache marks for Anthropic models, tools in loop turns."""
    anthropic = model.startswith('anthropic/')
    body: dict[str, Any] = {'model': model, 'messages': messages, 'max_tokens': max_tokens, 'temperature': 0}
    if anthropic:
        body['messages'] = cache_marked(messages)
    if tools:
        body['tools'], body['tool_choice'] = tools, 'auto'
        # Arena r16 noask b04: tool-loop turns ran ~1.2-2.5k completion tokens for ~150 visible (default thinking),
        # 7.5-12 s each; 27/81 episodes hit the clock with ~205 s of loop calls. Loop turns think at low effort (as
        # terrya2 r7-r9 and YesYesYes r16 send it through the race proxy); readings and checks keep full thinking
        if anthropic:
            body['reasoning'] = {'effort': 'low'}
    return body

def timed_post(client: Any, path: str, body: dict[str, Any], room: float, floor: int) -> Any:
    """Send the post with a client wait short enough that every retry finishes inside room seconds, floor at least."""
    usual = getattr(client, 'timeout', None)
    if isinstance(usual, (int, float)):
        retries = max(0, int(getattr(client, 'max_retries', 0) or 0))
        client.timeout = max(floor, int(min(usual, room / (1 + retries))))
        try:
            return client.post_verbose(path, body)
        finally:
            client.timeout = usual
    return client.post_verbose(path, body)

def failure_of(result: Any, data: dict[str, Any] | None) -> tuple[Any, Any]:
    """(status, text) of a failed chat call: the reply's own error object first, else the transport error."""
    failed = data.get('error') if data else None
    if not isinstance(failed, dict):
        error = result.error or {}
        return error.get('status'), error.get('body')
    return whole(failed.get('code')), failed.get('message')

class ModelDesk:
    """Model calls: choice, retries, time cuts, JSON."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.calls = 0
        self.waited = 0
        self.cost = 0.0
        self.out_of_credit = False

    def models_for(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.cost <= limit and self.cost + HEAVY_ROOM <= SPEND_HARD:
            return [HEAVY, LIGHT]
        return [LIGHT] if self.cost <= SPEND_HARD else []

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int,
              tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            # past the clock stop a fallback model would start a call the 300 s limit cuts off
            if self.calls >= ALL_CALLS or (model != models[0] and self.ep.past_stop()):
                return None
            self.calls += 1
            message, status = self.retried(model, messages, max_tokens, tools)
            if message is not None or status == 402:
                # an exhausted key (402) stops every later model too
                self.out_of_credit = self.out_of_credit or message is None
                return message
        return None

    def retried(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
                tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One model's posts, repeated after a transient failure while the wait budget and clock allow."""
        tries = 0
        while True:
            message, status = self.post(model, messages, max_tokens, tools)
            if message is not None:
                return message, status
            transient = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
            if not transient or tries >= (1 if status in (None, 408) else len(WAITS)):
                return None, status
            pause = WAITS[tries]
            if self.waited + pause > WAIT_TOTAL or self.ep.past_stop():
                return None, status
            self.waited += pause
            time.sleep(pause * (0.75 + 0.05 * (self.calls % 11)))
            tries += 1

    def post(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
             tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One proxy chat call: (message, 200) or (None, status)."""
        ep = self.ep
        body = chat_body(model, messages, max_tokens, tools)
        # each call must end inside the episode's 300 s, so its wait is cut to the seconds left (Arena run 13021: a call
        # started near the clock stop ran past the limit and the episode ended unfinished)
        room = EPISODE_SECONDS - ep.elapsed() - 6
        if room < 4:
            ep.log('llm_skip', model=model, room=round(room, 1))
            return None, 0
        result = timed_post(_llm, '/inference/chat/completions', body, room, 4)
        data = result.data if isinstance(result.data, dict) else None
        choices = data.get('choices') if data else None
        if not choices:
            status, text = failure_of(result, data)
            ep.log('llm_fail', model=model, status=status, body=str(text or '')[:200])
            return None, status
        spent = (data.get('usage') or {}).get('cost')
        self.cost += float(spent) if isinstance(spent, (int, float)) else 0.0
        message = choices[0].get('message') or {}
        if message.get('content') or message.get('tool_calls'):
            return message, 200
        ep.log('llm_fail', model=model, status=0)
        return None, 0

    def ask(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        """First JSON object any of the models returns."""
        for model in models:
            reply = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens) or {}
            got = parse_object(reply.get('content'))
            if got is not None or self.stopped():
                return got
        return None

    def stopped(self) -> bool:
        return self.ep.store.blocked or self.out_of_credit or self.calls >= ALL_CALLS

def store_note_in_answer(ep, name: str, observation: Any) -> None:
    """A reply to Desk's one question that carries more than one passage: the others are the store talking, read like any
    store note (the answer itself goes to the interviewer as before)."""
    # Arena r18 noask b02: on 11 of 84 tasks a reply to a question also carried a store note in a field of its own (a need
    # the shopper let go); no change reader saw it, the dropped need stayed live (4 of 11 passed, 35 of 73 others)
    if not isinstance(observation, dict):
        return
    # Arena 18357: the note came as a list of sentences ({"note-field": ["..."], "answer-field": "..."}), never a plain string
    texts = [x for v in observation.values() for x in (v if isinstance(v, list) else [v])]
    passages = [v for v in texts if isinstance(v, str) and len(v.split()) >= 5]
    if len(passages) >= 2:
        ep.log('note_in_answer', passages=len(passages))
        ep.ingest(name, observation)

class StoreGateway:
    """Store actions with step bookkeeping."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.blocked = False

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        ep = self.ep
        payload = {**ep.binding, 'step': ep.step + 1, 'action': {'name': name, 'args': args}}
        attempts = 0
        while attempts < 4:
            attempts += 1
            # each store call waits at most for the episode's remaining time shared over the client's retries: a slow
            # store near the end must not hold the episode past its limit without a finish
            result = timed_post(_env, '/environment/call', payload, EPISODE_SECONDS - ep.elapsed() - 3, 3)
            if isinstance(result.data, dict):
                data = self.accept(name, args, result.data)
                if name == ep.roles.get('ask'):
                    store_note_in_answer(ep, name, data.get('observation'))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            ep.log('env_fail', status=status, body=body[:200])
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.blocked = True
                return None
            if status != 409:
                continue
            if not self.step_resync(body):
                return None
            payload['step'] = ep.step + 1
        return None

    def step_resync(self, body: str) -> bool:
        """On a step conflict adopt the step count the store reports; False if the episode has ended or the cause is unclear."""
        ep = self.ep
        low = body.lower()
        if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
            ep.finished = True
            return False
        expected = re.search('expected (\\d+)', body)
        if expected is None and 'reuse' not in low and 'different action' not in low:
            return False
        ep.step = int(expected.group(1)) - 1 if expected else ep.step + 1
        return True

    def accept(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        ep = self.ep
        told = data.get('step')
        ep.step = told if isinstance(told, int) else ep.step + 1
        if data.get('finished'):
            ep.finished = True
        if 'observation' in data:
            data['observation'] = as_records(data['observation'])
        seen = data.get('observation')
        viewed = name in ep.page_tools() or name == ep.offer_view()
        if not viewed or (isinstance(seen, dict) and seen.get('error')):
            return data
        # a view that came back without the listing (a refused side-by-side size) has not opened it
        shown = dumps(seen)
        given = [v for value in args.values() for v in (value if isinstance(value, list) else [value])]
        ep.opened.update(v.strip() for v in given if isinstance(v, str) and v.strip() in shown)
        return data

class RequestReader:
    """Request reading and the ledger text."""

    def __init__(self) -> None:
        self.routed: tuple[int, bool] | None = None
        self.label_met: dict[str, str] = {}
        self.labels_seen: set[str] = set()

def req_read(ep) -> None:
    """Brief, roles, items, limits, message reserve."""
    ep.brief = req_first_reading(ep)
    named = ep.brief.get('roles')
    ep.roles = {}
    for job, tool in (named.items() if isinstance(named, dict) else ()):
        if isinstance(tool, str) and tool in ep.schemas:
            ep.roles[job] = tool
    req_repair_search(ep, named if isinstance(named, dict) else {})
    req_settle_end_modes(ep)
    became = req_collect_items(ep, as_list(ep.brief.get('items')))
    req_collect_links(ep, became)
    req_collect_limits(ep)
    profile = req_profile_now(ep)
    ep.msg_reserve = req_close_reserve(ep, profile)

def req_repair_search(ep, named: dict[str, Any]) -> None:
    """The brief named a search tool the store does not have: the one unassigned tool that takes free text and says it
    searches takes the role."""
    # Arena r18 noask b02: the brief named a search tool the store lacks on 55 of 84 tasks (a generic name, not the store's); with no
    # search role every code search, page read and whole-word cut stood down, and model queries cut mid-word at the
    # 27-character field came back empty (one task: 11 of 37 steps)
    wanted = named.get('search')
    if 'search' in ep.roles or not isinstance(wanted, str) or not wanted.strip():
        return
    taken = set(ep.roles.values())
    fits = [t['function']['name'] for t in ep.tools if t['function']['name'] not in taken and req_takes_query(t['function'])]
    if len(fits) == 1:
        ep.roles['search'] = fits[0]
        ep.log('search_role_repaired', named=wanted, tool=fits[0])

def req_takes_query(spec: dict[str, Any]) -> bool:
    """A tool whose name or description says it searches and that takes a free-text argument."""
    props = (spec.get('parameters') or {}).get('properties') or {}
    free = [k for k, v in props.items() if isinstance(v, dict) and v.get('type') == 'string' and not v.get('enum')]
    said = f"{spec.get('name', '')} {spec.get('description', '')}"
    return bool(free) and re.search('(?i)search|keyword|query|look ?up|seek|\\bfind', said) is not None

def req_label_met(ep) -> list[str]:
    """Request musts the store label of their item already states (the value's words plus the label's last word): set aside,
    with the label noted for the ledger."""
    # Arena r18 needs-in-request: a must built from the requested category named a key whose value on the paying listing
    # (filed under that very label) was another word; the listing was read early and refused on every later turn
    cat_key = mem_kind_key(ep)
    if not cat_key or not ep.items:
        return []
    labels = {str(r.get(cat_key) or '').strip() for r in ep.mem.rows.values()} - {''}
    fresh = labels - ep.req.labels_seen
    if not fresh:
        return []
    ep.req.labels_seen |= fresh
    pointers = {m for _, m in ep.refs} | {str(x) for x in as_list(ep.brief.get('lookups'))}
    words = lambda text: [w for w in plain_words(text) if len(w) > 2 and w not in ('and', 'the', 'any')]
    notes = []
    for number, item in enumerate(ep.items, 1):
        mine = sorted(label for label in labels if mem_owner(ep, label) == number)
        for need in [m for m in item.musts if m not in item.dropped and m not in pointers]:
            value = need.partition(':')[2]
            want = set(words(value))
            if not want or re.search('[0-9]', value) or re.search('\\b(?:not|no|without|except|other than|unlike)\\b', value, re.I):
                continue
            for label in mine:
                have = words(label)
                if have and want <= set(have) and set(have) - want <= {have[-1]}:
                    item.dropped.append(need)
                    ep.req.label_met[need] = label
                    notes.append(f'{need} <- {label}')
                    break
    if notes:
        ep.version += 1
        ep.log('label_met', changed=notes)
    return notes

def req_profile_now(ep) -> SizeProfile:
    """Profile for the item count (re-chosen on change)."""
    merged = any(t.get('kind') == 'items_merged' for t in ep.trace)
    sure = not req_shared_shelf(ep.brief) and not merged
    stamp = (len(ep.items), sure)
    if ep.req.routed != stamp:
        ep.req.routed = stamp
        ep.profile = SizeProfile(*stamp)
        ep.log('profile', size=ep.profile.size, sure=sure)
    return ep.profile

def req_tool_digest(ep, skip: set[str] | None = None) -> str:
    lines = []
    for tool in ep.tools:
        spec = tool['function']
        if skip and spec['name'] in skip:
            continue
        lines.append(f"- {spec['name']}: {spec.get('description', '')} | parameters: {dumps(spec.get('parameters') or {})}")
    return '\n'.join(lines)

def req_first_reading(ep) -> dict[str, Any]:
    opening = ep.view.get('opening')
    parts = [BRIEF, '\nTOOLS:\n', req_tool_digest(ep), '\n\nREQUEST:\n', ep.request]
    if opening is not None:
        parts += ['\n\nOPENING NOTE FROM THE STORE:\n', dumps(clip(opening, 1500))]
    prompt = ''.join(parts)
    usable = lambda got: bool(as_list(got.get('items'))) and isinstance(got.get('roles'), dict) and bool(got['roles'].get('buy'))
    first = ep.llm.ask(prompt, [LIGHT], 4000) or {}
    # a usable sheet is read again by the heavy model only when two of its items share one shelf
    if ep.llm.out_of_credit or (usable(first) and not req_shared_shelf(first)):
        return first
    if not usable(first):
        return ep.llm.ask(prompt, [HEAVY, LIGHT], 2500) or first
    second = ep.llm.ask(prompt, [HEAVY], 2500) or {}
    return second if usable(second) else first

def req_settle_end_modes(ep) -> None:
    """Separate buy and refuse modes of one tool."""
    for key in ('buy_args', 'refuse_args'):
        if not isinstance(ep.brief.get(key), dict):
            ep.brief[key] = {}
    tool = ep.roles.get('buy')
    buying, ending = ep.brief['buy_args'], ep.brief['refuse_args']
    if not tool or tool != ep.roles.get('refuse') or (buying and ending and buying != ending):
        return
    prompt = (f'The store tool {tool} either checks out or ends without a purchase, according to the arguments it gets. Schema: '
              + dumps(ep.schemas.get(tool)) + '\nAnswer with ONE JSON object only: {"buy_args": the fixed arguments that make it check out, '
              '"refuse_args": the fixed arguments that make it end without a purchase} (just the arguments that choose the mode).')
    modes = ep.llm.ask(prompt, [LIGHT, HEAVY], 1500) or {}
    buy_mode, end_mode = modes.get('buy_args'), modes.get('refuse_args')
    if isinstance(buy_mode, dict) and isinstance(end_mode, dict) and buy_mode != end_mode:
        ep.brief['buy_args'], ep.brief['refuse_args'] = buy_mode, end_mode

def musts_clash(one: ItemSpec, other: ItemSpec) -> bool:
    """Two items set one property to different values."""
    halves = lambda must: (plain_key(must.split(':')[0]), plain_key(must.partition(':')[2]))
    theirs: dict[str, set[str]] = {}
    for key, value in map(halves, other.musts):
        theirs.setdefault(key, set()).add(value)
    return any(theirs.get(key, set()) - {value} for key, value in map(halves, one.musts))

def req_item_of(ep, entry: dict[str, Any]) -> ItemSpec:
    """One brief entry as an item, its values restored to the request's wording."""
    written = [str(m) for m in as_list(entry.get('musts')) if m]
    restored = [restore_value(ep.request, m) for m in written]
    if restored != written:
        ep.log('request_value', read=[m for m in written if m not in restored], now=[m for m in restored if m not in written])
    name = str(entry.get('name') or 'the product')
    units = whole(entry.get('quantity')) or 0
    units = units if units > 1 else 1
    # the brief counts "a pair or set sold as one listing" as 1, and with a size value beside it also turned an
    # explicit "N x <item>" into 1 (Arena r17 b11 and KING; real r17: orders holding fewer units than the request's
    # written count passed 0/19, matching ones about a third): a count the request writes before the item's name wins
    # over a brief quantity of 1; the pack size stays a must
    counted = req_written_count(ep.request, name)
    if units == 1 and (counted or 0) > 1:
        ep.log('request_count', item=name, brief=units, now=counted)
        units = counted
    return ItemSpec(name, str(entry.get('shelf') or ''), units, restored)

def req_words(text: str) -> set[str]:
    """Roots of the words once possessive endings and punctuation are gone ("the lamps's colour," -> lamp, colour)."""
    return roots(re.sub("[^a-z0-9]+", ' ', re.sub("['\u2019]s?(?![a-z])", '', str(text).lower())))

def req_written_count(request: str, name: str) -> int | None:
    """The single count written right before the item's name in the request ("N x <item>", "N <item>"), or None."""
    sequence = lambda words: [r for word in words for r in sorted(req_words(word))]
    want = sequence(re.findall('[a-z0-9]+', str(name).lower()))
    if not want:
        return None
    tokens = re.findall("[a-z0-9]+(?:['\u2019][a-z]+)?|[^\\sa-z0-9]", str(request).lower())
    # English words that tie a number to a property or a bound rather than to the item ("the set size is 6 cups",
    # "a box of 10 forks", "at most 3 rugs"): grammar, not product vocabulary. A number written that way anywhere
    # in the request is a value, not the item's count (real r17: pack-size numbers repeat as "30 hangers")
    linking = {'is', 'are', 'was', 'be', 'of', 'per', 'with', 'has', 'have', 'holds', 'contains', 'includes', 'in', 'than', 'to',
               'by', 'from', 'least', 'most', 'up', 'over', 'under', 'about', 'around', 'between', 'each', 'every'}
    numbers = {}
    for at, token in enumerate(tokens):
        hit = re.fullmatch('(\\d{1,3})(x|\u00d7)?', token)
        if hit:
            numbers[at] = hit
    valued = {int(hit.group(1)) for at, hit in numbers.items() if at and (tokens[at - 1] in linking or tokens[at - 1] in ('.', '$'))}

    def names_item(at: int, hit: Any) -> bool:
        # the item's own words follow the count, in their order ("3 x tea cups where ...")
        rest = tokens[at + 1:]
        if not hit.group(2) and rest[:1] and rest[0] in ('x', '\u00d7', '*'):
            rest = rest[1:]
        window = rest[:len(want) + 3]
        stop = next((k for k, word in enumerate(window) if not re.match('[a-z0-9&]', word)), len(window))
        return sequence(window[:stop])[:len(want)] == want
    found = {int(hit.group(1)) for at, hit in numbers.items()
             if int(hit.group(1)) not in valued and 1 <= int(hit.group(1)) <= 99 and names_item(at, hit)}
    return next(iter(found)) if len(found) == 1 else None

def req_collect_items(ep, entries: list[Any]) -> dict[int, int]:
    """Items from the brief; entry number -> item number."""
    became: dict[int, int] = {}
    for number, entry in enumerate([e for e in entries if isinstance(e, dict)], 1):
        item = req_item_of(ep, entry)
        home = 0
        if item.shelf.strip():
            # an entry on the shelf of an earlier item, with no property set differently, joins that item
            shelf = plain_key(item.shelf)
            home = next((n for n, kept in enumerate(ep.items, 1) if plain_key(kept.shelf) == shelf and not musts_clash(kept, item)), 0)
        if not home:
            ep.items.append(item)
            became[number] = len(ep.items)
            continue
        kept = ep.items[home - 1]
        for must in item.musts:
            if must not in kept.musts:
                kept.musts.append(must)
        ep.log('items_merged', into=kept.name, dropped=item.name)
        became[number] = home
    if not ep.items:
        ep.items.append(ItemSpec('the requested product', '', 1, []))
    return became

def req_collect_links(ep, became: dict[int, int]) -> None:
    """Part-device pairs and pointer requirements."""
    req_fit_pairs(ep, became)
    musts = [(n, m) for n, item in enumerate(ep.items, 1) for m in item.musts]
    req_lookup_refs(ep, musts)
    req_must_pointers(ep, musts)
    req_untangle(ep)

def req_fit_pairs(ep, became: dict[int, int]) -> None:
    """Item pairs the brief says must fit each other, in brief order, each once."""
    count = len(ep.items)
    for entry in filter(lambda e: isinstance(e, dict), as_list(ep.brief.get('fits'))):
        first, second = (became.get(whole(entry.get(side)) or 0, 0) for side in ('item', 'with'))
        if first == second or not (0 < first <= count and 0 < second <= count):
            continue
        if (first, second) not in ep.bound:
            ep.bound.append((first, second))

def req_lookup_refs(ep, musts: list[tuple[int, str]]) -> None:
    """The brief's lookups as (item, need) pointers, unless they only name another requested item."""
    # possessive endings and punctuation do not hide the name: "unless the floor lamp's colour is white" names the
    # floor lamp (Arena r17 b11: 12 tasks carry such a clause, ours 0/12; read as a chain, it spent TRAIL rounds,
    # searches and checkout holds hunting for a listing that is the agent's own pick)
    plain_names = [(n, req_words(item.name)) for n, item in enumerate(ep.items, 1)]
    for text in filter(None, (str(x).strip() for x in as_list(ep.brief.get('lookups')))):
        head = plain_key(text.split(':')[0])
        exact = [pair for pair in musts if pair[1] == text]
        owner = (exact or [pair for pair in musts if plain_key(pair[1].split(':')[0]) == head] or [(1, text)])[0]
        value = text.partition(':')[2]
        said = req_words(value)
        # a value that hangs on what is picked for ANOTHER requested item and names no shop or listing is a condition
        # inside this order, not a page to find: no chain is held for it (Arena r15: such a hold spent three checkouts)
        inner = any(n != owner[0] and words and words <= said for n, words in plain_names)
        if inner and re.search('\\b[A-Z][a-z]', value) is None:
            ep.log('condition_not_lookup', need=text[:120])
        elif owner not in ep.refs:
            ep.refs.append(owner)

def req_must_pointers(ep, musts: list[tuple[int, str]]) -> None:
    """Musts whose value describes a listing by a capitalised shop or maker name, even when lookups left them out."""
    # Arena r15: 1 of 33 pointer tasks per full run had no lookup, and that episode never opened the pointed page; a
    # value naming ANOTHER requested item is a fit or condition inside this order, not a page to find
    item_roots = [roots(item.name) for item in ep.items]
    for number, need in musts:
        if (number, need) in ep.refs:
            continue
        value = need.partition(':')[2]
        held = roots(value)
        if any(words and words <= held for n, words in enumerate(item_roots, 1) if n != number):
            continue
        if re.search('\\b(?:listing|page)\\b', value, re.I) and re.search('\\b[A-Z][A-Za-z]{3,}', value):
            ep.refs.append((number, need))
            ep.log('pointer_from_must', need=need[:120])

def req_collect_limits(ep) -> None:
    """Budget, caps and the required process steps."""
    brief = ep.brief
    budget = brief.get('budget')
    numeral = isinstance(budget, str) and budget.strip().replace('.', '', 1).isdecimal()
    budget = float(budget) if numeral else budget
    if is_number(budget) and budget > 0:
        ep.budget = budget
    cap = whole(brief.get('messages'))
    if cap is None or cap < 0:
        if ep.roles.get('ask'):
            ep.msg_cap = 3
    else:
        ep.msg_cap, ep.cap_stated = cap, True
    steps = whole(brief.get('steps'))
    if steps is not None and 10 <= steps < ep.max_steps:
        ep.max_steps = steps
    for entry in as_list(brief.get('process')):
        step = str(entry.get('step') or '').strip() if isinstance(entry, dict) else ''
        if not step:
            continue
        when = str(entry.get('when') or '').strip().lower()
        ep.process.append({'step': step, 'when': when if when in ('choose', 'order', 'end') else 'order',
                           'messages': max(0, whole(entry.get('messages')) or 0)})

def req_close_reserve(ep, profile: SizeProfile) -> int:
    """Messages reserved for closing steps."""
    if not ep.roles.get('ask'):
        return 0
    peak = {'order': 0, 'end': 0}
    for step in ep.process:
        if step['when'] in peak:
            peak[step['when']] = max(peak[step['when']], step['messages'])
    # the unbought ending's need wins only when it is larger and the profile leaves room for it
    if peak['end'] > peak['order'] and profile.end_room(ep.msg_cap, peak['end']):
        return min(peak['end'], ep.msg_cap)
    return min(peak['order'], ep.msg_cap)

def req_shared_shelf(brief: dict[str, Any]) -> bool:
    shelves = []
    for entry in as_list(brief.get('items')):
        shelf = plain_key(str(entry.get('shelf') or '')) if isinstance(entry, dict) else ''
        if shelf:
            shelves.append(shelf)
    return len(shelves) != len(set(shelves))

def req_untangle(ep) -> None:
    """Fold a pointer's description back into it."""
    for number, item in enumerate(ep.items, 1):
        pointers = [need for n, need in ep.refs if n == number and need in item.musts]
        rest = [m for m in item.musts if m not in pointers]
        if not pointers or not rest:
            continue
        sections = [UNMIX, '\n\nREQUEST:\n', ep.request, f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n',
                    '\n'.join(pointers), '\nOTHER REQUIREMENTS:\n', '\n'.join(rest)]
        reply = ep.llm.ask(''.join(sections), [HEAVY, LIGHT], 1500) or {}
        named = {str(x).strip() for x in as_list(reply.get('describe_other'))}
        moved = [m for m in rest if m in named]
        if not moved:
            continue
        lead = pointers[0]
        joined = f"{lead} (that listing: {'; '.join(moved)})"
        item.musts = [joined if m == lead else m for m in item.musts if m not in moved]
        ep.refs = [(number, joined) if pair == (number, lead) else pair for pair in ep.refs]
        ep.log('unmix', item=number, moved=moved)

def req_is_description(ep, number: int, need: str) -> bool:
    """Shopper's own wording no read key carries."""
    prop, _, value = need.partition(':')
    item = ep.items[number - 1]
    if req_value_refers(ep, item, need, prop, value):
        return False
    blocks = req_kind_blocks(ep, item)
    if len(blocks) < 3:
        return False
    wanted = stems(prop)
    return all(not stems(k) & wanted for k in {plain_key(k) for block in blocks for k in block})

def req_value_refers(ep, item: ItemSpec, need: str, prop: str, value: str) -> bool:
    """A need that is a pointer or contrast, too short to be a description, or that names another requested item."""
    lookups = [str(x) for x in as_list(ep.brief.get('lookups'))]
    if need in lookups or need in [m for _, m in ep.refs]:
        return True
    if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4 or re.search('\\b(?:other than|unlike)\\b', value, re.I):
        return True
    # a value with a figure is a measure, not the shopper's description
    if re.search('[0-9]', value):
        return True
    # a likeness to something no listing was found for (a request requirement, not a lookup) is the shopper's own
    # description when no listing states that property
    likened = need in item.musts and need not in item.added and re.search('\\bsame as\\b', value, re.I)
    if re.search('\\b(?:listing|page|shown|same as)\\b', value, re.I) and not likened:
        return True
    value_stems = stems(value)
    return any(len(w) >= 4 and w in value_stems for other in ep.items if other is not item for w in stems(other.name))

def req_kind_blocks(ep, item: ItemSpec) -> list[dict[str, Any]]:
    """Attribute blocks of the read pages whose category belongs to the item (all pages when no category key)."""
    cat_key = mem_kind_key(ep)
    records = list(ep.mem.pages.values())
    if cat_key:
        records = [r for r in records if mem_belongs(ep, item, str(r.get(cat_key) or ''))]
    return [block for block in map(fact_block, records) if block]

def req_ledger(ep) -> str:
    """Requirements in force, as the model reads them."""
    many = len(ep.items) > 1
    lines = ['- ' + req_item_line(ep, n, item, many) for n, item in enumerate(ep.items, 1)]
    lines += ['- ' + line for line in req_terms_lines(ep)]
    return '\n'.join(lines)

def req_item_line(ep, number: int, item: ItemSpec, many: bool) -> str:
    live = item.live()
    own_words = [m for m in live if req_is_description(ep, number, m)]
    needs = [m + f' (value read on the page it refers to: {ep.ref_values[m]})' if m in ep.ref_values else m for m in live if m not in own_words]
    head = (f'item {number}: ' if many else 'item: ') + item.name + (f' (store label: {item.shelf})' if item.shelf else '')
    parts = [f'{head}, quantity {item.qty}', 'must have: ' + ('; '.join(needs) or 'only that it is this kind of product')]
    if own_words:
        parts.append("the shopper's own wording, which no listing attribute states (it decides nothing): " + '; '.join(own_words))
    shown = [m for m in item.dropped if m in ep.req.label_met]
    if shown:
        parts.append('met by the store label itself, whatever a listing under it says on that key: '
                     + '; '.join(f"{m} (label '{ep.req.label_met[m]}')" for m in shown))
    if item.free:
        parts.append('no preference on: ' + ', '.join(item.free))
    return '; '.join(parts)

def req_terms_lines(ep) -> list[str]:
    brief = ep.brief
    pick = brief.get('pick')
    lines = []
    if ep.budget is not None:
        scope = 'per item' if brief.get('budget_scope') == 'each' else 'for the whole order'
        lines.append(f"budget: no more than {ep.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
    if pick in ('most', 'least'):
        lines.append(f"pick: among fitting listings, the {pick} {brief.get('pick_measure') or ''}".replace('  ', ' ').rstrip())
    elif pick == 'cheapest':
        lines.append('pick: the lowest-priced listing that fits')
    else:
        lines.append('pick: any listing that meets everything')
    if brief.get('in_stock_only', True) is not False:
        lines.append('only listings in stock')
    for title, key in (('wish, only to break ties: ', 'wishes'), ('request rule: ', 'rules')):
        lines.extend(title + f'{entry}' for entry in as_list(brief.get(key)))
    if brief.get('quiet_decline') is True:
        lines.append('when nothing fits: end naming or suggesting no listing')
    stages = {'choose': 'before choosing', 'order': 'at the order (Desk does it when you check out)', 'end': 'at an unbought ending (Desk does it then)'}
    for step in ep.process:
        lines.append(f"required step, {stages[step['when']]}: {step['step']}")
    facts = ep.attr_tool()
    if facts and (ep.fact_tools.get(facts) is True or ep.roles.get('offer')):
        lines.append(req_attr_line(ep, facts))
    return lines

def req_attr_line(ep, facts: str) -> str:
    views = ep.page_tools()
    text = f'attributes are shown by {facts}'
    bare = sorted(t for t in views | {ep.roles.get('offer') or ''} if t and ep.fact_tools.get(t) is False)
    if bare:
        text += f"; {', '.join(bare)} show no attributes"
    pages = sorted(views - {facts})
    if pages:
        text += f"; {' or '.join(pages)} shows a listing's page (what a chain of pages or a page check refers to)"
    return text

class ListingMemory:
    """Rows and read pages of every listing seen."""

    def __init__(self) -> None:
        self.rows, self.pages, self.tool_keys, self.view_keys = {}, {}, {}, {}
        self.order: list[str] = []
        self.id_key = self.price_key = self.stock_key = None
        self.catch_stamp: tuple[int, int] | None = None
        self.catch_keys: set[str] = set()

def mem_identify(ep, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
    if not isinstance(node, dict) or len(node) < 3:
        return None
    known = ep.mem.id_key
    if known and id_like(node.get(known), True):
        return node[known].strip()
    if not (in_list or depth == 0 or any(isinstance(v, dict) for v in node.values())):
        return None
    found = guess_id_key([node])
    if not found:
        return None
    ident = node[found].strip()
    return ident if 'id' in str(found).lower() or ident in ep.mem.rows else None

def mem_rows_keyed(ep, node: list[Any]) -> bool:
    rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
    if not rows or 2 * len(rows) < len(node):
        return False
    known = ep.mem.id_key
    if not (known and all(id_like(r.get(known), True) for r in rows)):
        # the rows name a new id key, or they are not listings
        found = guess_id_key(rows)
        if not found:
            return False
        ep.mem.id_key = found
    return True

def mem_keep_row(ep, ident: str, node: dict[str, Any]) -> None:
    """Merge one listing row, tracking stock flips and read pages."""
    mem = ep.mem
    old = mem.rows.get(ident)
    if old is None:
        mem.order.append(ident)
        old = mem.rows[ident] = {}
    # boolean fields that changed value: any true -> false marks the listing unavailable, else a false -> true clears it
    flips = [old.get(k) for k, v in node.items() if isinstance(v, bool) and isinstance(old.get(k), bool) and old.get(k) is not v]
    if True in flips:
        ep.unavailable.add(ident)
    elif flips:
        ep.unavailable.discard(ident)
    old.update(node)
    page = mem.pages.get(ident)
    if fact_block(node):
        mem.pages[ident] = dict(old)
    elif page is not None:
        page.update(node)

def mem_walk(ep, node: Any, depth: int, in_list: bool = False) -> None:
    # depth first in document order: a list's own keyed test runs before any of its entries is read
    pending = [(node, depth, in_list)]
    while pending:
        node, depth, in_list = pending.pop()
        if depth > 6:
            continue
        if isinstance(node, dict):
            ident = mem_identify(ep, node, in_list, depth)
            if ident:
                mem_keep_row(ep, ident, node)
            else:
                pending.extend((v, depth + 1, False) for v in reversed(list(node.values())))
        elif isinstance(node, list):
            keyed = mem_rows_keyed(ep, node)
            pending.extend((v, depth + 1, keyed) for v in reversed(node))

def mem_absorb(ep, observation: Any) -> None:
    mem_walk(ep, observation, 0)

def mem_enum_labels(ep) -> set[str]:
    found: set[str] = set()
    level = list(ep.schemas.values())
    # enum values anywhere in the tool schemas, down to seven levels of nesting
    for _ in range(7):
        below = []
        for schema in (s for s in level if isinstance(s, dict)):
            found.update(o.strip().lower() for o in schema.get('enum') or [] if isinstance(o, str))
            below += [v for v in schema.values() if isinstance(v, dict)]
        level = below
    return found

def mem_kind_key(ep) -> str | None:
    rows = ep.mem.rows
    short = [(k, v) for row in rows.values() for k, v in row.items() if isinstance(v, str) and 0 < len(v) <= 40 and not re.search('\\d{3}', v)]
    seen: dict[str, list[str]] = {}
    for key, value in short:
        seen.setdefault(key, []).append(value)
    total = max(1, len(rows))
    known = mem_enum_labels(ep)
    best, best_rank = None, None
    for key, texts in seen.items():
        values = set(texts)
        if len(texts) < 0.8 * total or len(values) > 40:
            continue
        # most values a tool lists as labels, then fewest distinct values, most carriers, shortest values
        listed = sum(v.strip().lower() in known for v in values) / len(values)
        rank = (-listed, len(values), -len(texts), -sum(map(len, values)) / len(values))
        if best_rank is None or rank < best_rank:
            best, best_rank = key, rank
    return best

def mem_label(ep, ident: str, key: str | None) -> str:
    return str((ep.mem.rows.get(ident) or {}).get(key) or '')

def mem_is_catch_all(ep, key: str) -> bool:
    """Key whose values look like feature bundles."""
    mem = ep.mem
    stamp = (len(mem.pages), ep.version)
    if mem.catch_stamp != stamp:
        mem.catch_stamp = stamp
        mem.catch_keys = {plain_key(row[1]) for row in mem_key_table(ep) if listy(row[3])}
    return plain_key(key) in mem.catch_keys

def mem_key_table(ep) -> list[tuple[str, str, int, list[str]]]:
    """Planner rows: (category, key, carriers, samples)."""
    table: list[tuple[str, str, int, list[str]]] = []
    groups = mem_wanted_groups(ep, mem_key_groups(ep))
    for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
        for rank, (name, slot) in enumerate(mem_kept_keys(group)):
            # a need can sit on a key few listings carry: past the first rows every key seen on two or more stays
            if rank >= TABLE_FIRST and not (P2_ON and slot['n'] >= 2 and not re.search(TABLE_SKIP, name)):
                continue
            table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
    return table

def mem_key_groups(ep) -> dict[str, dict[str, Any]]:
    """Per category: pages read and keys with values."""
    cat_key = mem_kind_key(ep)
    groups: dict[str, dict[str, Any]] = {}
    for ident in ep.mem.order:
        record = ep.mem.pages.get(ident)
        if not record:
            continue
        group = groups.setdefault(str(record.get(cat_key) or 'listings') if cat_key else 'listings', {'n': 0, 'keys': {}})
        group['n'] += 1
        # a key spelled several ways in one block keeps the value of its first spelling
        keyed = [(plain_key(k), v) for k, v in (fact_block(record) or {}).items()]
        for name in dict.fromkeys(n for n, _ in keyed if n):
            text = str(next(v for n, v in keyed if n == name)).strip().lower()
            slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
            slot['n'] += 1
            slot['per'][ident] = text
            slot['values'] += [] if text in slot['values'] else [text]
    return groups

def mem_asked_label(ep, label: str) -> bool:
    """Category label of a requested kind."""
    long = lambda text: {w for w in stems(text) if len(w) >= 4}
    key, words = plain_key(label), long(label)
    asked = next((item for item in ep.items if key in (plain_key(item.shelf or ''), plain_key(item.name))
                  or words & (long(item.shelf or '') | long(item.name))), None)
    return asked is not None or bool(mem_owner(ep, label))

def mem_wanted_groups(ep, groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Groups of requested kinds (all if none match)."""
    return {label: group for label, group in groups.items() if mem_asked_label(ep, label)} or groups

def mem_kept_keys(group: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    kept: list[tuple[str, dict[str, Any]]] = []
    for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
        if free_text(list(slot['per'].values())):
            continue
        twin = next((spec for other, spec in kept if mem_twins(other, spec, name, slot)), None)
        if twin is None:
            kept.append((name, slot))
            continue
        # a twin key folds into the earlier one: its listings and any values not yet listed
        for ident, text in slot['per'].items():
            twin['per'].setdefault(ident, text)
        twin['values'] += [text for text in dict.fromkeys(slot['per'].values()) if text not in twin['values']]
        twin['n'] = len(twin['per'])
    return sorted(kept, key=lambda kv: -kv[1]['n'])

def mem_stated_clash(ep, label: str, name: str, values: list[str]) -> str:
    """Mark keys that are not a stated property."""
    number = mem_owner(ep, label)
    if not number:
        return ''
    words = lambda text: set(re.findall('[a-z0-9]{3,}', text.lower()))
    shown = words(' '.join(values))
    for must in ep.items[number - 1].musts:
        prop, _, value = must.partition(':')
        # a required property sharing a word with this key, whose value the key's values do not show
        if has_bound(value) or not prop.strip() or plain_key(prop) == name or not stems(prop) & stems(name):
            continue
        said = words(value)
        if said and not said <= shown:
            return f' [not the stated {prop.strip()}: another property]'
    return ''

def mem_twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
    a, b = set(name.split()), set(other.split())
    if not a & b:
        return False
    mine, theirs = spec['per'], slot['per']
    both = [i for i in mine if i in theirs and mine[i].strip() and theirs[i].strip()]
    agreeing = 0
    for x, y in ((mine[i], theirs[i]) for i in both):
        if x != y:
            # differing texts still agree when the shorter one's words all sit in a not much longer one
            short, longer = sorted((re.findall('[a-z0-9]+', t.lower()) for t in (x, y)), key=len)
            if not (short and set(short) <= set(longer) and len(longer) <= 2 * len(short) + 2):
                continue
        agreeing += 1
    if not (a <= b or b <= a):
        return len(both) >= 2 and agreeing == len(both)
    if not both:
        return bool((set(spec['values']) & set(slot['values'])) - {''})
    return 2 * agreeing > len(both)

def mem_kind_pools(ep) -> list[list[str]]:
    """Seen ids per requested kind, largest first."""
    cat_key = mem_kind_key(ep)
    label_of = (lambda i: str(ep.mem.rows[i].get(cat_key) or '')) if cat_key else (lambda i: '')
    groups: dict[str, list[str]] = {}
    for ident in ep.mem.order:
        label = label_of(ident)
        if label not in groups:
            groups[label] = []
        groups[label].append(ident)
    pools = [ids for label, ids in groups.items() if mem_asked_label(ep, label)] or list(groups.values())
    pools.sort(key=len, reverse=True)
    return pools[:max(1, len(ep.items))]

def mem_price(ep, ident: str) -> float | None:
    key = ep.mem.price_key
    if not key:
        return None
    value = (ep.mem.rows.get(ident) or {}).get(key)
    if isinstance(value, str):
        return float(value) if re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value) else None
    if isinstance(value, dict):
        value = next((v for v in value.values() if is_number(v)), None)
    return float(value) if is_number(value) else None

def mem_can_buy(ep, ident: str) -> bool:
    if ep.is_gone(ident):
        return False
    key = ep.mem.stock_key
    if not key:
        return True
    value = (ep.mem.rows.get(ident) or {}).get(key)
    return value is not False and not (is_number(value) and value <= 0)

def mem_listings_in(ep, observation: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(observation, list):
        groups = [observation]
    elif isinstance(observation, dict):
        ident = mem_identify(ep, observation)
        if ident:
            return [(ident, observation)]
        # lists one level down, or inside a dict one level down
        groups = []
        for value in observation.values():
            inner = [value] if isinstance(value, list) else list(value.values()) if isinstance(value, dict) else []
            groups += [rows for rows in inner if isinstance(rows, list)]
    else:
        return []
    found = []
    for row in (r for rows in groups for r in rows if isinstance(r, dict)):
        ident = mem_identify(ep, row, True)
        if ident:
            found.append((ident, row))
    return found

def mem_note_keys(ep, name: str, observation: Any) -> None:
    for ident, record in mem_listings_in(ep, observation):
        keys = set(record)
        if name not in ep.mem.tool_keys:
            ep.mem.tool_keys[name] = set()
        ep.mem.tool_keys[name] |= keys
        # keys seen at an older version start over; at this version they add up
        seen_at, have = ep.mem.view_keys.get(ident, (-1, set()))
        ep.mem.view_keys[ident] = (ep.version, set(keys) if seen_at != ep.version else have | keys)

def mem_fully_shown(ep, name: str, ident: str) -> bool:
    keys = ep.mem.tool_keys.get(name)
    if not keys or ident in ep.gone:
        return False
    seen_at, have = ep.mem.view_keys.get(ident, (-1, set()))
    return seen_at == ep.version and keys <= have

def mem_owner(ep, label: str) -> int:
    """Item number a category label belongs to (0: none)."""
    if not label:
        return 0
    words, long = plain_key(label), roots(label)
    best, top = 0, 0.0
    for number, item in enumerate(ep.items, 1):
        # the exact shelf beats the exact name, which beats the best share of word roots; the first item wins a tie
        if words == plain_key(item.shelf or ''):
            score = 3.0
        elif words == plain_key(item.name):
            score = 2.0
        else:
            mine = roots(item.shelf or '') | roots(item.name)
            score = len(long & mine) / max(1, len(long | mine))
        if score > top:
            best, top = number, score
    return best

def mem_belongs(ep, item: ItemSpec, label: str) -> bool:
    if not label or item not in ep.items:
        return False
    return mem_owner(ep, label) == ep.items.index(item) + 1

def mem_read_count(ep, number: int) -> int:
    cat_key = mem_kind_key(ep)
    item = ep.items[number - 1]
    if not cat_key:
        return 0
    return len([r for r in ep.mem.pages.values() if mem_belongs(ep, item, str(r.get(cat_key) or '')) and fact_block(r)])

def mem_ids_of_item(ep, number: int) -> list[str]:
    cat_key = mem_kind_key(ep)
    item = ep.items[number - 1]
    if not cat_key:
        return []
    return [i for i in ep.mem.order if mem_belongs(ep, item, mem_label(ep, i, cat_key))]

def mem_record_text(ep, ident: str, review: bool = False) -> str:
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident)
    if record:
        return dumps(clip(as_stated(record) if review else record, 400))[:3000]
    return '(no record seen)'

def mem_other_records(ep, skip: list[str], limit: int = 30) -> str:
    order = list(reversed(ep.mem.order))
    if ep.cheapest():
        order.sort(key=lambda i: (i not in ep.mem.pages, ep.price_rank(i)))
    chosen = [i for i in order if i not in skip and not ep.is_gone(i)]
    texts = []
    for ident in chosen[:max(1, limit)]:
        record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident)
        texts.append(dumps(clip(as_stated(record), 160))[:900])
    return '\n'.join(texts)

class SearchScout:
    """Searching and paging done on the desk's own."""

    def __init__(self) -> None:
        self.word_rule: str | None = None
        self.capped_done, self.capped_more, self.wider_done = False, False, False
        self.pages_read = self.first_reads = self.loose_notes = 0
        self.cheaper_marks: set[str] = set()
        self.rule_log, self.search_log, self.searched, self.widened = [], [], [], []
        self.extra_pages, self.kind_calls, self.cap_calls = [], [], []
        # pages Desk read on its own per item number on a long task
        self.long_pages: dict[int, int] = {}
        # (item number, query) of each zero-row search Desk sent again with fewer words
        self.retried: list[tuple[int, str]] = []
        self.shapes: dict[str, dict[str, Any] | None] = {}
        # paging learned in code from results: tool -> (arg that moves through the rows, result key of the next value)
        self.page_ways: dict[str, tuple[str, str | None]] = {}
        # next values the store handed back, per tool and query (the arguments without the paging one)
        self.next_given: dict[str, list[Any]] = {}

def scout_values_shown(ep, ident: str, needs: list[str]) -> int:
    """How many required plain values (limits excluded) have all their words in a listing's row or page."""
    words = set(plain_words(dumps(ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {})))
    count = 0
    for need in needs:
        _, colon, value = need.partition(':')
        if not colon or has_bound(value):
            continue
        wanted = set(plain_words(value))
        if wanted and wanted <= words:
            count += 1
    return count

def scout_kinds(records: Any, cat_key: str) -> set[str]:
    """The records' kind labels, lower-cased, empty ones left out."""
    return {str(r.get(cat_key) or '').strip().lower() for r in records} - {''}

def scout_open_items(ep) -> list[int]:
    """Item numbers with no cart line yet."""
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return list(range(1, len(ep.items) + 1))
    held = {mem_owner(ep, mem_label(ep, i, cat_key)) for i in basket_ids(ep)}
    return [n for n, _ in enumerate(ep.items, 1) if n not in held]

def scout_long_task(ep) -> bool:
    """A long multi-item task: steps to spare for Desk's own paging and reads (see LONG_STEPS)."""
    count = len(ep.items)
    return count >= 2 and ep.max_steps >= max(LONG_STEPS, LONG_ITEM_STEPS * count)

def scout_long_room(ep, steps: int) -> bool:
    """Whether Desk may spend steps more on its own on a long task: the clock leaves two model turns and the steps left,
    after the close reserve and these steps, keep four model steps per item still lacking a line."""
    if not scout_long_task(ep) or ep.elapsed() > CLOCK_STOP - LONG_CLOCK:
        return False
    room = min(ep.steps_left(), ep.clock_room()) - ep.reserve() - steps
    return room >= 12 + 4 * len(scout_open_items(ep))

def scout_search_item(ep, name: str, args: dict[str, Any]) -> int:
    """The item a search's words are for, 0 when unclear."""
    return scout_query_item(ep, scout_search_words(ep, name, args).split()) if ep.items else 0

def scout_long_item(ep, name: str, args: dict[str, Any]) -> int:
    """On a long task, the item a model search is for when that item has no cart line yet, else 0."""
    if not scout_long_task(ep) or name != ep.roles.get('search'):
        return 0
    number = scout_search_item(ep, name, args)
    return number if number in scout_open_items(ep) else 0

def scout_long_pages(ep, plan: Any, args: dict[str, Any], number: int, observation: Any) -> int:
    """Further pages to read of a first page on a long task, whatever the pick: up to the store's reach (LONG_PAGES), within
    the item's page budget and the room left. Arena r18 world, task with 4 products: one of Halo's paid lines sat on page 4
    of the very query we sent (36 rows), another on page 3 of a query of 18 rows; we read page 1 of both."""
    if not plan:
        return 0
    count, shown, param, value = plan
    given = args.get(param)
    numeric = isinstance(given, int) and not isinstance(given, bool)
    later = given not in (None, 0, 1) if numeric else bool(str(given or '').strip())
    if later or value is None or shown <= 0 or shown >= count:
        return 0
    if ep.cheapest() and scout_price_sorted(ep, observation):
        return 0
    used = ep.scout.long_pages.get(number, 0)
    pages = min(LONG_PAGES, (count - 1) // shown, LONG_ITEM_PAGES - used)
    return pages if pages > 0 and scout_long_room(ep, pages) else 0

def scout_first_reads(ep, seen: Any) -> str:
    """Once the model has searched, open in one go the best unread rows for items that have no cart line yet."""
    # Arena noask b10: the model asked for 30 store calls per episode against the king's 20 at the same step count (the
    # king's own code made about half its store calls), and 158 of 393 episodes ran past 255 s; a Desk read costs a step
    # but no model turn. Not before the interview starts, so the question timing stays as it is
    scout = ep.scout
    # long tasks: more reads per episode while the clock leaves room (LONG_FIRST_READS); the searched item goes first
    long = scout_long_task(ep)
    limit = LONG_FIRST_READS if long else FIRST_READS
    if scout.first_reads >= limit or (ep.roles.get('ask') and not ep.talk.started) or ep.past_stop():
        return ''
    if long and ep.elapsed() > CLOCK_STOP - LONG_CLOCK:
        return ''
    cat_key = mem_kind_key(ep)
    missing = scout_open_items(ep)
    room = min(ep.steps_left(), ep.clock_room()) - ep.reserve()
    if not (cat_key and missing) or room < 12 + 4 * len(missing):
        return ''
    if long and scout.searched and ep.roles.get('search'):
        aimed = scout_search_item(ep, ep.roles['search'], scout.searched[-1])
        missing = sorted(missing, key=lambda n: n != aimed)
    cap = ep.numeric_budget()
    # the caller hands over the search's pages as a list; read whole, that list names no listing (Arena r18 world runs
    # 18357/18591/18615: Desk opened no row in 108 episodes). Long tasks take the rows of each page; short tasks keep the
    # behaviour their passes were measured with
    pages = seen if long and isinstance(seen, list) else [seen]
    listed = list(dict.fromkeys(i for page in pages for i, _ in mem_listings_in(ep, page)))
    picks: list[str] = []
    for number in missing:
        item = ep.items[number - 1]
        needs = item.live()
        rows = []
        for i in listed:
            if i in ep.mem.pages or not mem_can_buy(ep, i) or ep.is_gone(i):
                continue
            if mem_belongs(ep, item, mem_label(ep, i, cat_key)) and (cap is None or (mem_price(ep, i) or 0) <= cap):
                rows.append(i)
        cheap = ep.cheapest()
        shown = {i: scout_values_shown(ep, i, needs) for i in rows}
        ranked = sorted(rows, key=lambda i: (-shown[i], ep.price_rank(i) if cheap else 0))
        picks += [i for i in ranked if cheap or shown[i] > 0][:2]
    picks = list(dict.fromkeys(picks))[:min(2, max(0, limit - scout.first_reads))]
    if not picks:
        return ''
    texts = scout_read_pages(ep, picks, ep.reserve() + 10)
    scout.first_reads += len(texts)
    if not texts:
        return ''
    ep.log('first_reads', ids=picks[:len(texts)])
    return '\n'.join(['\n[Desk opened, one step each, the most promising unread rows of this result for items that have no cart '
                      'line yet]', *texts])

def scout_fetch(ep, tool: str, args: Any) -> tuple[bool, Any]:
    """One store call with its observation taken into memory: (whether the call went through, the observation)."""
    reply = None if args is None else ep.store.call(tool, args)
    if reply is None:
        return False, None
    seen = reply.get('observation')
    ep.ingest(tool, seen)
    return True, seen

def scout_paging_arg(ep, tool: str) -> Any:
    named = (ep.scout.shapes.get(tool) or {}).get('next_arg')
    if named:
        return named
    learned = ep.scout.page_ways.get(tool)
    return learned[0] if learned else None

def scout_next_key(ep, tool: str) -> Any:
    """Result key holding the next page's value."""
    shape = ep.scout.shapes.get(tool) or {}
    if shape.get('next_arg'):
        return shape.get('next')
    learned = ep.scout.page_ways.get(tool)
    return learned[1] if learned else None

def scout_loose_numbers(ep, tool: str) -> list[str]:
    """Optional numeric arguments the schema does not explain: no description, no choices and no tie to price."""
    schema = ep.schemas.get(tool) or {}
    required = schema.get('required') or []
    out = []
    for k, spec in (schema.get('properties') or {}).items():
        if not isinstance(spec, dict) or k in required or 'enum' in spec:
            continue
        if not {'integer', 'number'} & set(type_names(spec)) or str(spec.get('description') or '').strip():
            continue
        # a number named as a cap or limit bounds the rows (real r13 listing filter: a cap beside a start row); never a start
        if re.search('price|cost|amount|budget|spend|money', k, re.I) or re.search(LIMIT_ARG, k, re.I):
            continue
        out.append(k)
    return out

def scout_paging_found(loose: list[str], sent: dict[str, int], ints: dict[str, int], shown: int) -> tuple[str, str | None, int] | None:
    """(start argument, next-value key or None, start sent) when the result proves a start position, else None."""
    for key, start in sent.items():
        nxt = [k for k, v in ints.items() if shown and v == start + shown]
        if len(nxt) == 1:
            return key, nxt[0], start
        if not shown and any(0 < v <= start for v in ints.values()):
            return key, None, start
    if sent or len(loose) != 1 or not shown:
        return None
    nxt = [k for k, v in ints.items() if v == shown]
    return (loose[0], nxt[0], 0) if len(nxt) == 1 and any(v > shown for v in ints.values()) else None

def scout_learn_paging(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Find out from a search result which unexplained number walks through its rows (a start position). The proof is
    structural: the next value equals the start sent plus the rows shown; or a search shows no rows while its match
    count is at or below the start sent; or, with nothing sent and just one such argument, the next value equals the
    rows shown while more matched. Real race 17 (all agents): episodes whose new queries carried a start position
    passed .16, against .37 for the rest."""
    if name != ep.roles.get('search') or not isinstance(observation, dict) or scout_paging_arg(ep, name):
        return ''
    loose = scout_loose_numbers(ep, name)
    is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)
    ints: dict[str, int] = {}
    lists = []
    for k, v in observation.items():
        if isinstance(v, list):
            lists.append(v)
        elif is_int(v):
            ints[k] = v
    if len(lists) != 1 or not loose:
        return ''
    sent = {k: args[k] for k in loose if is_int(args.get(k)) and args[k] > 0}
    found = scout_paging_found(loose, sent, ints, len(lists[0]))
    if found is None:
        return ''
    key, nxt, start = found
    ep.scout.page_ways[name] = (key, nxt)
    ep.log('paging_learned', tool=name, arg=key, next=nxt, start=start)
    return (f"\n[Desk] In this search's results {key} is the start position: this page began after row {start}, so the "
            f'earlier rows stayed hidden. Leave {key} out to start a new query at its first row; page on only with the value '
            'the store returns.') if start else ''

def scout_note_next(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Keep the next value a search handed back, per query."""
    paging = scout_paging_arg(ep, name)
    key = scout_next_key(ep, name)
    if not (paging and key) or not isinstance(observation, dict) or key not in observation:
        return
    query = name + dumps({k: v for k, v in args.items() if k != paging})
    given = ep.scout.next_given
    given[query] = [*given.get(query, []), observation[key]][-6:]

def scout_reopen_cheapest(ep) -> None:
    """Reopen the cheaper-fit hunt after a change."""
    ep.scout.wider_done = False
    ep.scout.cheaper_marks = set()

def scout_shape_of(got: dict[str, Any], observation: dict[str, Any], lists: list[str], params: dict[str, Any]) -> dict[str, Any] | None:
    """Result shape from a SHAPE reply; every key it names is checked in the place it must occur."""
    rows = got.get('rows')
    if not rows or rows not in lists:
        return None
    first = observation[rows][0]
    places = {'count': observation, 'next': observation, 'next_arg': params, 'price': first, 'stock': first}
    shape: dict[str, Any] = {'rows': rows}
    for part, where in places.items():
        said = got.get(part)
        shape[part] = None if not isinstance(said, str) or said not in where else said
    total = observation.get(shape['count']) if shape['count'] else None
    shape['count'] = shape['count'] if isinstance(total, int) and not isinstance(total, bool) else None
    return shape

def scout_learn_shape(ep, name: str, observation: Any) -> dict[str, Any] | None:
    """Learn a list result's keys once per tool."""
    shapes = ep.scout.shapes
    if name in shapes:
        return shapes[name]
    if not isinstance(observation, dict):
        return None
    lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) > 1 and all(isinstance(r, dict) for r in v)]
    if not lists:
        return None
    shapes[name] = None
    schema = ep.schemas.get(name) or {}
    sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
    prompt = (SHAPE + '\n\nTOOL: ' + name + ' | parameters: ' + dumps(schema) + '\n\nRESULT (first rows only):\n'
              + dumps(clip(sample, 120))[:4000])
    got = ep.llm.ask(prompt, [LIGHT, HEAVY], 2500) or {}
    shape = scout_shape_of(got, observation, lists, schema.get('properties') or {})
    if shape is None:
        return None
    shapes[name] = shape
    mem = ep.mem
    mem.price_key = mem.price_key or shape['price']
    mem.stock_key = mem.stock_key or shape['stock']
    ep.log('shape', tool=name, shape=shape)
    return shape

def scout_page_plan(ep, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
    """(matched, shown, paging arg, next value) or None."""
    shape = scout_learn_shape(ep, name, observation)
    if not shape or not isinstance(observation, dict) or not all(shape[k] for k in ('count', 'next', 'next_arg')):
        return None
    rows, count, value = (observation.get(shape[k]) for k in ('rows', 'count', 'next'))
    integral = lambda v: isinstance(v, int) and not isinstance(v, bool)
    if not isinstance(rows, list) or not rows or not integral(count):
        return None
    if value is not None and not integral(value) and not isinstance(value, str):
        return None
    return count, len(rows), shape['next_arg'], value

def scout_price_sorted(ep, observation: Any) -> bool:
    prices = [mem_price(ep, i) for i, _ in mem_listings_in(ep, observation)]
    if None in prices or len(prices) < 3:
        return False
    return not any(prices[k] > prices[k + 1] for k in range(len(prices) - 1))

def scout_auto_pages(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Read more pages for extreme picks, or advise."""
    ep.scout.extra_pages = []
    pick = ep.brief.get('pick')
    extreme = pick in ('cheapest', 'most', 'least')
    pointed = not extreme and scout_names_pointed_seller(ep, name, args)
    number = 0 if pointed else scout_long_item(ep, name, args)
    plan = scout_page_plan(ep, name, observation) if extreme or pointed or number else None
    scout_remember_kind_search(ep, name, args, observation)
    further = scout_long_pages(ep, plan, args, number, observation) if number else 0
    if further:
        return scout_long_read(ep, name, args, plan, number, further)
    due = (scout_pages_due(ep, plan, args, observation, pick, worded(scout_search_words(ep, name, args))) if extreme
           else scout_pointed_pages(plan, args))
    if isinstance(due, str) or not due:
        return due or ''
    count, shown, param, value = plan
    texts, value = scout_read_more(ep, name, args, param, value, due)
    ep.log('pages', tool=name, read=len(texts))
    if not texts:
        return ''
    lines = ['\n[Desk read the next pages of this search on its own, one step per page]', *texts]
    if value is not None:
        lines.append(f'[Desk] Of the {count} matching listings Desk has read {shown * (len(texts) + 1)} rows so far; run a new '
                     'search with other words, or page on for the rest.')
    return '\n'.join(lines)

def scout_long_read(ep, name: str, args: dict[str, Any], plan: Any, number: int, pages: int) -> str:
    """Read the further pages of a model search on a long task, all as compact rows; stops at the clock margin."""
    count, shown, param, value = plan
    go = lambda: ep.elapsed() <= CLOCK_STOP - LONG_CLOCK
    texts, value = scout_read_more(ep, name, args, param, value, pages, compact=True, go=go)
    ep.scout.long_pages[number] = ep.scout.long_pages.get(number, 0) + len(texts)
    ep.log('long_pages', tool=name, item=number, read=len(texts))
    if not texts:
        return ''
    lines = ['\n[Desk read the further pages of this search on its own, one step per page; their rows follow in short]', *texts]
    if value is not None:
        lines.append(f'[Desk] Of the {count} matching listings Desk has read {shown * (len(texts) + 1)} rows so far; page on with '
                     f'{param}={dumps(value)} for the rest, or run a new search with other words.')
    return '\n'.join(lines)

def scout_names_pointed_seller(ep, name: str, args: dict[str, Any]) -> bool:
    """Whether the words of a search carry a capitalised name that a pointer requirement uses (the shop or the maker
    of the listing the request points to)."""
    texts = ' '.join([need for _, need in ep.refs] + [str(x) for x in as_list(ep.brief.get('lookups'))])
    names = set(re.findall('[A-Z][A-Za-z]{3,}', texts))
    if not names:
        return False
    return any(w in names for w in re.findall('[A-Z][A-Za-z]{3,}', scout_search_words(ep, name, args)))

def scout_pointed_pages(plan: Any, args: dict[str, Any]) -> int:
    """Further pages to read after a first page that searched the shop of a pointed listing; all of them when at most
    two remain. Real race 15: the pointed listing sat on page 2 of an 8-row shop search; 21 of the 22 passers read and
    opened it, and none of the 56 agents that missed it passed."""
    if not plan:
        return 0
    count, shown, param, value = plan
    if value is None or shown <= 0 or count <= shown or args.get(param) not in (None, '', 0, 1):
        return 0
    rest = (count - 1) // shown
    return rest if rest <= 2 else 0

def scout_search_words(ep, name: str, args: dict[str, Any]) -> str:
    """Text arguments that are not labels or choices."""
    props = (ep.schemas.get(name) or {}).get('properties') or {}
    cat_key = mem_kind_key(ep)
    labels = scout_kinds(ep.mem.rows.values(), cat_key) if cat_key else set()
    choice = lambda key: isinstance(props.get(key), dict) and 'enum' in props[key]
    return ' '.join(v for k, v in args.items()
                    if isinstance(v, str) and v.strip() and not choice(k) and v.strip().lower() not in labels)

def scout_pages_due(ep, plan: Any, args: dict[str, Any], observation: Any, pick: Any, has_words: bool = True) -> int | str:
    """Pages to read here, or a note instead."""
    if not plan:
        return ''
    count, shown, param, value = plan
    given = args.get(param)
    numeric = isinstance(given, int) and not isinstance(given, bool)
    later = given not in (None, 0, 1) if numeric else bool(str(given or '').strip())
    if later or value is None or shown >= count:
        return ''
    if pick == 'cheapest' and scout_price_sorted(ep, observation):
        return ''
    # cheapest picks, real races 13-15: a later page of a category listing or of a number-led search rarely showed
    # a paying listing (0.4-4.7%, r15 .008), a later page of a worded search 4-9% (r15 short results of at most
    # 5 pages .23, a new search's first page .16). Searches led by a number and a noun count as worded: r15 later
    # pages of results up to 56 rows showed a paying listing .356 of calls for them, .145 for plain words, and 30
    # of 49 first sights of a paid listing by passers came on a later page (one task: page 3 of 54 rows for all)
    rest = (count - 1) // shown
    cap = ep.page_cap()
    # a store reaching 7 pages of 7 rows leaves at most 6 to read, whatever count it reports; the page budget is
    # shared by the items: one item's search may not use up another's
    pages = min(1 if rest > 7 else min(rest, 6), max(1, cap // max(1, len(ep.items)))) if has_words else 0
    roomy = ep.steps_left() - ep.reserve() - pages >= 12
    if pages and ep.scout.pages_read + pages <= cap and roomy:
        return pages
    return (f"\n[Desk] This search matched {count} listings and shows {shown}, in no order of price; each fitting one "
            "counts for the pick. New words (a synonym, the product's plain name, title words of fitting listings seen so far) "
            "find fitting listings far more often than further pages do; once no new words are left, keep paging with "
            f"{param}={dumps(value)}.")

def scout_read_more(ep, name: str, args: dict[str, Any], param: str, value: Any, pages: int, compact: bool = False,
                    go: Any = None) -> tuple[list[str], Any]:
    """Next pages of a search: (page texts, the next value). compact: every page as compact rows; go: checked before
    each page, False stops the reading."""
    texts: list[str] = []
    schema = ep.schemas.get(name)
    scout = ep.scout
    while value is not None and len(texts) < pages and (go is None or go()):
        call = shaped({**args, param: value}, schema)
        if call is None:
            break
        reply = ep.store.call(name, call)
        if reply is None or ep.finished:
            break
        scout.pages_read += 1
        seen = reply.get('observation')
        ep.ingest(name, seen)
        scout_note_next(ep, name, call, seen)
        scout.extra_pages.append(seen)
        rendered = ep.render(seen)
        ep.cache[name + dumps(call)] = (ep.version, ep.step, rendered)
        # the first extra page is shown in full, later ones as compact rows
        texts.append(f'{param}={dumps(value)}: ' + (scout_short_rows(ep, seen) if texts or compact else rendered))
        after = scout_page_plan(ep, name, seen)
        value = after[3] if after else None
    return texts, value

def scout_short_rows(ep, observation: Any) -> str:
    """Compact one-line-per-row page text."""
    rows = []
    for ident, record in mem_listings_in(ep, observation):
        blurb = next((v for v in record.values() if isinstance(v, str) and v.strip() != ident and len(v.split()) >= 3), '')[:90]
        price = mem_price(ep, ident)
        sold = ' | SOLD OUT' if ep.is_gone(ident) else ''
        shown = '?' if price is None else price
        rows.append((float('inf') if price is None else price, f'{ident} | price {shown}{sold} | {blurb}'))
    if not rows:
        return ep.render(observation)
    rows.sort(key=lambda row: row[0])
    return '(compact rows, cheapest first) ' + '; '.join(text for _, text in rows)

def scout_price_caps(args: dict[str, Any], props: dict[str, Any]) -> list[str]:
    """Numeric limit arguments the schema ties to price, lowest value first."""
    about = lambda key: str(props[key].get('description') or '') if isinstance(props[key], dict) else ''
    picked = [key for key, value in args.items() if key in props and is_number(value) and re.search(LIMIT_ARG, key, re.I)
              and re.search('price|cost|amount|budget|spend|money', f'{key} {about(key)}', re.I)]
    return sorted(picked, key=args.get)

def scout_ignored_cap(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Note for a price cap the search ignored."""
    props = (ep.schemas.get(name) or {}).get('properties') or {}
    caps = scout_price_caps(args, props)
    if not caps or not ep.mem.price_key:
        return ''
    head = caps[0]
    cap = args[head]
    over = sum(1 for i, _ in mem_listings_in(ep, observation) if (mem_price(ep, i) or 0) > cap)
    if not over:
        return ''
    others = {k: [o for o in props[k]['enum'] if o != v] for k, v in args.items()
              if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
    named = ', '.join(f'{k}={dumps(rest[0])}' for k, rest in others.items() if rest)
    hint = f'; to see only cheaper listings, list the category with {named} and the same {head}' if named else ''
    return f'\n[Desk] {head}={cap:g} was not applied by this search: {over} of its rows cost more{hint}.'

def scout_cheapest_note(ep, observations: list[Any]) -> str:
    if not ep.mem.price_key or not ep.cheapest():
        return ''
    found = {i for seen in observations for i, _ in mem_listings_in(ep, seen)}
    rows = sorted((mem_price(ep, i), i) for i in found if mem_can_buy(ep, i) and mem_price(ep, i) is not None)
    if len(rows) < 2:
        return ''
    return '\n[Desk] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in rows[:6])

def scout_remember_kind_search(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Keep a kind search as a capped-listing base."""
    paging = scout_paging_arg(ep, name)
    cat_key = mem_kind_key(ep)
    rows = mem_listings_in(ep, observation)
    if not (rows and cat_key) or (paging and args.get(paging) not in (None, 0, '')):
        return
    kinds = scout_kinds((record for _, record in rows), cat_key)
    calls = ep.scout.kind_calls
    if kinds and not any(c[0] == name and c[1] == args for c in calls):
        ep.scout.kind_calls = [*calls, (name, dict(args), kinds)][-12:]

def scout_cap_guesses(ep) -> list[tuple[str, str, dict[str, Any], set[str]]]:
    """Calls that may list a category under a price."""
    guesses = list(ep.scout.cap_calls)
    for name, args, kinds in ep.scout.kind_calls:
        props = (ep.schemas.get(name) or {}).get('properties') or {}
        skip = {scout_paging_arg(ep, name), *args}
        numeric = [key for key, spec in props.items()
                   if key not in skip and isinstance(spec, dict) and {'number', 'integer'} & set(type_names(spec))]
        guesses += [(name, key, dict(args), kinds) for key in numeric
                    if re.search(LIMIT_ARG, key, re.I) and not re.search('min|least|page|size|offset|start|row|count', key, re.I)]
    return guesses

def scout_capped_listing(ep) -> str:
    """List each cart line's category under its price."""
    scout = ep.scout
    if scout.capped_done or ep.finished or not ep.cheapest():
        return ''
    scout.capped_done = True
    late = lambda: ep.elapsed() > CLOCK_STOP - CAPPED_CLOCK
    if late() or ep.steps_left() < CAPPED_ROOM:
        ep.log('capped_skip', left=ep.steps_left(), spent=round(ep.elapsed()))
        return ''
    cat_key = mem_kind_key(ep)
    scout.capped_more = False
    texts: list[str] = []
    for line in basket_ids(ep):
        price = mem_price(ep, line)
        kind = mem_label(ep, line, cat_key).strip().lower() if cat_key else ''
        if price is None or not kind or late() or ep.steps_left() < ep.reserve() + 8:
            continue
        # the guess that lists only this kind, with the fewest other arguments, goes first
        ranked = sorted((g for g in scout_cap_guesses(ep) if kind in g[3]), key=lambda g: (g[3] != {kind}, len(g[2])))
        if ranked:
            texts.extend(scout_list_under(ep, line, price, ranked[0]))
    return '\n'.join(texts)

def scout_capped_pages(ep, tool: str, call: dict[str, Any], paging: str | None) -> tuple[list[Any], list[str], Any]:
    """Page through the capped listing; returns the pages, the texts shown and the last paging plan."""
    pages, shown, plan = [], [], None
    late = CLOCK_STOP - CAPPED_CLOCK / 2
    for _ in range(CAPPED_PAGES):
        fitted = shaped(call, ep.schemas.get(tool))
        if fitted is None or ep.steps_left() < ep.reserve() + 6 or ep.elapsed() > late:
            break
        ok, seen = scout_fetch(ep, tool, fitted)
        if not ok:
            break
        pages.append(seen)
        text, label = ep.render(seen), dumps(fitted)
        ep.cache[tool + label] = (ep.version, ep.step, text)
        shown.append(f'[{tool} {label}, one step]\n{text}')
        plan = scout_page_plan(ep, tool, seen) if paging else None
        if not plan or plan[3] is None or ep.finished or not mem_listings_in(ep, seen):
            break
        call = {**call, plan[2]: plan[3]}
    return pages, shown, plan

def scout_rows_below(ep, pages: list[Any], price: float) -> tuple[set[str], list[tuple[float, str]]]:
    """Every row id on the pages, and the unopened buyable rows priced under the line, cheapest first."""
    found = {ident for page in pages for ident, _ in mem_listings_in(ep, page)}
    cart = basket_ids(ep)
    priced = ((mem_price(ep, i), i) for i in found if i not in cart and i not in ep.mem.pages)
    return found, sorted((c, i) for c, i in priced if c is not None and c < price and mem_can_buy(ep, i))

def scout_open_below(ep, line: str, cheaper: list[tuple[float, str]], shown: list[str]) -> list[tuple[float, str]]:
    """The cheapest rows under the line are opened here, so the pick and the review read their attributes."""
    roomy = ep.steps_left() >= ep.reserve() + 8
    if not (cheaper and roomy) or ep.elapsed() > CLOCK_STOP - CAPPED_CLOCK / 2:
        return cheaper
    tried = [ident for _, ident in cheaper[:CHEAPER_OPENS]]
    opened = scout_read_pages(ep, tried, ep.reserve() + 6)
    if opened:
        read = [i for i in tried if i in ep.mem.pages]
        proof = judge_evidence(ep, [ep.mem.pages[i] for i in read])
        shown.append('\n'.join(['[Desk opened the cheapest of these rows itself, one step each]', *opened] + ([proof] if proof else [])))
        ep.log('capped_open', line=line, opened=[i for i in tried if i in ep.mem.pages])
    return [(c, i) for c, i in cheaper if i not in ep.mem.pages]

def scout_list_under(ep, line: str, price: float, guess: tuple[str, str, dict[str, Any], set[str]]) -> list[str]:
    """One cart line's capped listing."""
    tool, key, args, _ = guess
    paging = scout_paging_arg(ep, tool)
    first = {k: v for k, v in args.items() if k not in (key, paging)}
    first[key] = price if not float(price).is_integer() else int(price)
    pages, shown, plan = scout_capped_pages(ep, tool, first, paging)
    if not pages:
        return []
    scout_learn_cap(ep, tool, first, pages[0])
    found, cheaper = scout_rows_below(ep, pages, price)
    cheaper = scout_open_below(ep, line, cheaper, shown)
    paged_on = bool(plan) and plan[3] is not None
    more = plan[0] - sum(len(mem_listings_in(ep, page)) for page in pages) if paged_on else 0
    ep.log('capped_listing', line=line, tool=tool, pages=len(pages), rows=len(found), unread=len(cheaper), more=max(0, more))
    notes = shown[:]
    if cheaper:
        listed = ', '.join(f'{i} ({p:g})' for p, i in cheaper[:12])
        tail = f' and {len(cheaper) - 12} more' if len(cheaper) > 12 else ''
        notes.append(f'[Desk] Rows under the cart line price ({price:g}) that nobody opened yet, lowest price first: '
                     f'{listed}{tail}. Choose the lowest-priced one among them whose attributes satisfy every '
                     'requirement, if one does.')
    if more > 0:
        ep.scout.capped_more = True
        notes.append(f'[Desk] This listing still has {more} rows nobody read: continue paging with {plan[2]}={dumps(plan[3])}.')
    return ['\n'.join(notes)]

def scout_side_pages(ep, ids: list[str], keep: int) -> list[str]:
    """Listings read in side-view calls of up to the view's room, once that view is known to show full pages; texts of the
    pages that arrived. Real race 18: two single reads in a row came 3.9 times an episode for us, 1.9 for the leader."""
    tool, many = ep.roles.get('compare'), ep.side_param()
    if not (tool and many) or ep.side_view_bare() or ep.fact_tools.get(tool) is not True or len(ids) < 2:
        return []
    size = max(2, ep.side_room())
    texts: list[str] = []
    while len(ids) >= 2 and ep.steps_left() > keep and not ep.finished:
        batch, ids = ids[:size], ids[size:]
        ok, seen = scout_fetch(ep, tool, shaped({many: batch}, ep.schemas.get(tool)))
        if not ok:
            break
        mem_note_keys(ep, tool, seen)
        arrived = [i for i in batch if i in ep.mem.pages]
        if not arrived:
            ep.side_bad = min(ep.side_bad or 999, len(batch))
            break
        texts += [f'{i}: ' + dumps(clip(as_stated(ep.mem.pages[i]), 200))[:1500] for i in arrived]
    if texts:
        ep.log('side_pages', n=len(texts))
    return texts

def scout_read_pages(ep, ids: list[str], keep: int) -> list[str]:
    """Open listings while more than keep steps remain."""
    tool = ep.attr_tool()
    target = ep.id_param(tool)
    if not (tool and target):
        return []
    mem = ep.mem
    todo = [i for i in ids if i in mem.rows and i not in mem.pages and not ep.is_gone(i)]
    texts = scout_side_pages(ep, todo, keep)
    todo = [i for i in todo if i not in mem.pages]
    for ident in todo:
        if ep.finished or ep.steps_left() <= keep or ep.fact_tools.get(tool) is False:
            break
        ok, seen = scout_fetch(ep, tool, shaped({target: ident}, ep.schemas.get(tool)))
        if not ok:
            break
        mem_note_keys(ep, tool, seen)
        if isinstance(seen, dict) and mem_identify(ep, seen):
            ep.fact_tools[tool] = ep.fact_tools.get(tool, False) or bool(fact_block(seen))
        texts.append(f'{ident}: ' + ep.render(seen)[:1500])
    if texts:
        ep.log('fill_facts', n=len(texts))
    return texts

def scout_cheaper_rows(ep) -> list[str]:
    """Open unread rows cheaper than cart lines."""
    if not ep.cheapest():
        return []
    cat_key = mem_kind_key(ep)
    mark = f'{ep.cart_stamp()}#{ep.version}'
    marks = ep.scout.cheaper_marks
    if not cat_key or mark in marks:
        return []
    marks.add(mark)
    mem, cart = ep.mem, basket_ids(ep)
    kind_of = lambda ident: mem_label(ep, ident, cat_key).strip().lower()
    picks: list[str] = []
    for line in cart:
        kind, price = kind_of(line), mem_price(ep, line)
        if price is None or not kind:
            continue
        owner = mem_owner(ep, kind)
        needs = [] if not 0 < owner <= len(ep.items) else ep.items[owner - 1].live()
        pool = [i for i in mem.order if i not in cart and i not in mem.pages and mem_can_buy(ep, i)
                and kind_of(i) == kind and ep.price_rank(i) < price]
        pool.sort(key=lambda i: (-scout_values_shown(ep, i, needs), mem_price(ep, i) or 0.0))
        for ident in pool[:3]:
            if ident not in picks:
                picks.append(ident)
    # Desk reads cost a step but no model turn: with room left, read more of the cheaper rows (Arena b10: 9 of 34
    # 'not the best option' orders ended with 6+ steps and 70+ s unused while a cheaper valid line sat unread; real
    # r16 cheapest episodes using a price cap passed 18% vs 9%)
    room = min(ep.steps_left(), ep.clock_room()) - ep.reserve() - 6
    return scout_read_pages(ep, picks[:max(CHEAPER_OPENS, min(6, room))], ep.reserve() + 4)

def scout_learn_cap(ep, name: str, args: dict[str, Any], observation: Any) -> None:
    """Remember a numeric argument acting as price cap."""
    rows = mem_listings_in(ep, observation)
    prices = [mem_price(ep, ident) for ident, _ in rows]
    if len(prices) < 3 or None in prices:
        return
    paging = scout_paging_arg(ep, name)
    cat_key = mem_kind_key(ep)
    kinds = scout_kinds((record for _, record in rows), cat_key) if cat_key else set()
    top = max(prices)
    # an argument at or above every shown price may be a cap; one call per tool, argument and other arguments is kept
    for key, value in args.items():
        if key == paging or isinstance(value, bool) or not isinstance(value, (int, float)) or value < top:
            continue
        rest = {k: v for k, v in args.items() if k not in (key, paging)}
        kept = any(c[0] == name and c[1] == key and {k: v for k, v in c[2].items() if k not in (key, paging)} == rest
                   for c in ep.scout.cap_calls)
        if not kept:
            ep.scout.cap_calls = [*ep.scout.cap_calls, (name, key, dict(args), kinds)][-12:]

def scout_cheaper_plan(ep, number: int, cat_key: str | None) -> tuple[str, str, dict[str, Any], float] | None:
    """(tool, cap argument, call, floor) that lists the item's category at the lowest affordable price seen."""
    qty = max(1, ep.items[number - 1].qty)
    ids = mem_ids_of_item(ep, number)
    kinds = {mem_label(ep, i, cat_key).strip().lower() for i in ids} - {''} if cat_key else set()
    costs = (mem_price(ep, i) for i in ids if mem_can_buy(ep, i))
    prices = [c for c in costs if c is not None and (ep.budget is None or c * qty <= ep.budget)]
    calls = [c for c in ep.scout.cap_calls if kinds and c[3] & kinds]
    if not (prices and calls):
        return None
    tool, key, args, _ = min(calls, key=lambda c: len(c[2]))
    paging = scout_paging_arg(ep, tool)
    floor = min(prices)
    whole_floor = floor if not float(floor).is_integer() else int(floor)
    fitted = shaped({**{k: v for k, v in args.items() if k != paging}, key: whole_floor}, ep.schemas.get(tool))
    return None if fitted is None else (tool, key, fitted, floor)

def scout_cheaper_search(ep) -> str:
    """Capped listing for items a note re-opened."""
    # a listing that failed only the dropped requirement may never have been looked at,
    # and rows come in no price order
    due = sorted(ep.repick)
    ep.repick = set()
    if not due or ep.finished or not ep.cheapest():
        return ''
    if not ep.scout.cap_calls:
        return scout_widened_search(ep, due)
    cat_key = mem_kind_key(ep)
    texts = []
    for number in due:
        if ep.finished or number > len(ep.items) or ep.steps_left() < ep.reserve() + 3:
            break
        planned = scout_cheaper_plan(ep, number, cat_key)
        if planned is None:
            continue
        tool, key, call, floor = planned
        mark = tool + dumps(call)
        cached = ep.cache.get(mark)
        if cached and cached[0] == ep.version:
            continue
        ok, seen = scout_fetch(ep, tool, call)
        if not ok:
            break
        shown = ep.render(seen)
        ep.cache[mark] = (ep.version, ep.step, shown)
        listed = mem_listings_in(ep, seen)
        if any(floor < (mem_price(ep, i) or 0) for i, _ in listed):
            ep.scout.cap_calls = [c for c in ep.scout.cap_calls if (c[0], c[1]) != (tool, key)]
        ep.log('cheaper_search', item=number, tool=tool, args=call, rows=len(listed))
        texts.append(f'[Desk listed this category up to {floor:g}, the lowest price seen for item {number}, since with a '
                     f'requirement dropped a listing nobody looked at may be the cheapest fit ({tool} {dumps(call)}, one step)]\n'
                     + shown + scout_cheapest_note(ep, [seen]))
    return '\n'.join(texts)

def scout_reach_pages(ep, tool: str, args: dict[str, Any], seen: Any) -> list[Any]:
    """Further pages of a widened search up to the store's reach (three at most) while eight spare steps remain."""
    # after a change the paying row sat on a later page of a short search (real race 18: page 4 of one query, page 3 of another)
    plan = scout_page_plan(ep, tool, seen)
    if not plan or plan[3] is None or plan[1] <= 0:
        return []
    more = min(3, -(-(plan[0] - plan[1]) // plan[1]), ep.steps_left() - ep.reserve() - 8)
    if more <= 0:
        return []
    texts, _ = scout_read_more(ep, tool, args, plan[2], plan[3], more)
    return ep.scout.extra_pages[-len(texts):] if texts else []

def scout_widened_search(ep, due: list[int]) -> str:
    """No price-capped listing: every re-opened item is searched again with the values still in force."""
    # Arena r15 noask b04: 18 'not the best option' orders, 17 under a cheapest pick; after a refusal dropped a
    # requirement the model checked the old pick out again, never looking at listings that only the dropped value had kept out
    plan = scout_search_template(ep)
    if plan is None:
        return ''
    tool, base, field = plan
    texts = []
    for number in due:
        if ep.finished or ep.steps_left() < ep.reserve() + 4 or number > len(ep.items) or ep.past_stop():
            break
        # figures left out: an item whose needs are all limits or figures is searched by its bare noun (Arena r18
        # needs-in-request: figure words in a query emptied the result on an every-word store)
        values = [v for v in scout_value_words(ep, number) if not scout_figure(v)]
        noun = scout_title_noun(ep, number) or ep.items[number - 1].name
        words = scout_value_tries(values, values[0], noun)[0] if values else noun
        args = shaped(scout_whole_words(ep, tool, {**base, field: words}), ep.schemas.get(tool))
        if args is None:
            continue
        ok, seen = scout_fetch(ep, tool, args)
        if not ok:
            break
        ep.log('widened_search', item=number, words=words)
        pages = [seen] + scout_reach_pages(ep, tool, args, seen)
        line = min([mem_price(ep, i) or float('inf') for i in basket_ids(ep)] or [float('inf')])
        under = [i for i in judge_requested_unread(ep, [i for p in pages for i, _ in mem_listings_in(ep, p)]) if ep.price_rank(i) < line][:2]
        read = scout_read_pages(ep, under, ep.reserve() + 4) if under else []
        texts.append(f'[Desk ran item {number} as a fresh search keeping only the requirements still in force ({tool} {dumps(args)}, '
                     f'{len(pages)} page(s)): the store changed something, so a listing once excluded may now be the cheapest fit. The '
                     'cart line stays unless a listing meeting every requirement now in force costs less; read the cheapest rows of the '
                     'right kind before you check out]\n' + ep.render(seen) + ''.join('\n' + scout_short_rows(ep, p) for p in pages[1:])
                     + ('\n[cheaper rows Desk read]\n' + '\n'.join(read) if read else '') + scout_cheapest_note(ep, pages))
    return '\n'.join(texts)

def scout_wider_hint(ep) -> str:
    """Capped listing or a wider-search push, once."""
    if ep.scout.wider_done or not ep.cheapest() or ep.steps_left() < 14 + 2 * len(ep.items):
        return ''
    ep.scout.wider_done = True
    listed = scout_capped_listing(ep)
    if listed:
        rest = ' and keep paging through the rows nobody read while steps allow' if ep.scout.capped_more else ''
        return ("Not run yet. The pick is the cheapest, so Desk listed the cart's category up to the price of its line by itself:\n"
                + listed + f'\nOpen the lowest-priced of these rows that might fit{rest}; put the cheapest one meeting every '
                'requirement in place of the line, then check out.')
    priced = '; '.join(f'{i} at {mem_price(ep, i):g}' for i in basket_ids(ep) if mem_price(ep, i) is not None)
    return (f"Not run yet (no step used). {ep.steps_left()} steps remain and the pick is the cheapest. Rows come in no order "
            "of price, and this store's search matches titles only while a required value often appears only among the "
            "attributes, so a search by value skips many listings that fit. Before checking out, search each item one more "
            "time under different product names (synonyms, the plain one-word noun, title words of fitting listings already "
            "seen), keeping to buyable rows priced below the cart's line" + (f' ({priced})' if priced else '')
            + '; open the cheapest of those rows, put any that fits in place of the line, then check out.')

def scout_widen(ep) -> None:
    """Read listings of each kind before planning."""
    tool, target = ep.roles.get('compare'), ep.side_param()
    if not (tool and target) or ep.side_view_bare():
        return scout_widen_singly(ep)
    if ep.steps_left() <= 12:
        return None
    schema = ep.schemas.get(tool) or {}
    limit = schema['properties'][target].get('maxItems')
    cap = limit if isinstance(limit, int) else 4
    pools = mem_kind_pools(ep)
    share = max(1, cap // max(1, len(pools)))
    picks = [x for ids in pools for x in [i for i in ids if i not in ep.mem.pages][:share]][:cap]
    args = shaped({target: picks}, schema) if len(picks) >= 2 else None
    if args is None:
        return None
    before = len(ep.mem.pages)
    ok, seen = scout_fetch(ep, tool, args)
    if not ok:
        return None
    mem_note_keys(ep, tool, seen)
    rows = mem_listings_in(ep, seen)
    if rows and not ep.fact_tools.get(tool, False):
        ep.fact_tools[tool] = any(fact_block(r) for _, r in rows)
    ep.scout.widened.extend(filter(ep.mem.pages.__contains__, picks))
    ep.log('widen', ids=picks, opened=len(ep.mem.pages) - before)
    if rows and ep.fact_tools.get(tool) is False:
        return scout_widen_singly(ep)
    if len(ep.mem.pages) != before or len(picks) < 3 or ep.steps_left() <= 12:
        return None
    # the compare view opened nothing: try a pair once
    pair = picks[:2]
    if scout_fetch(ep, tool, shaped({target: pair}, schema))[0]:
        ep.scout.widened += [i for i in pair if i in ep.mem.pages]
    return None

def scout_widen_singly(ep) -> None:
    tool = ep.attr_tool()
    if not tool or not ep.id_param(tool):
        return
    pages, rows = ep.mem.pages, ep.mem.rows
    wanted = {w for _, need in ep.live_needs() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
    overlap = lambda i: len(wanted & set(re.findall('[a-z0-9]{3,}', dumps(rows.get(i) or {}).lower())))
    each = WIDEN_EACH if len(ep.items) <= 2 else 4
    built = []
    for ids in mem_kind_pools(ep):
        # unread buyable listings of one kind, best first, and how many may go in the first round
        opened = sum(1 for i in ids if i in pages)
        pool = [i for i in ids if i not in pages and mem_can_buy(ep, i)]
        if ep.cheapest():
            pool.sort(key=ep.price_rank)
        pool.sort(key=lambda i: -overlap(i))
        built.append((pool[:max(0, each - opened)], max(0, 4 - opened)))
    # round robin over the kinds: the first-round share of each, then the rest
    first = [lane[k] for k in range(4) for lane, head in built if k < min(head, len(lane))]
    more = [lane[k] for k in range(each) for lane, head in built if head <= k < len(lane)]
    cap = max(5, 2 * len(built) + 2)
    before = set(pages)
    read = scout_read_pages(ep, first[:min(cap, 8)], 22)
    scout_read_pages(ep, (first[cap:] + more)[:max(0, 10 - len(read))], 30)
    ep.scout.widened.extend(i for i in pages if i not in before)

def scout_whole_words(ep, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Search text above its field's maxLength cut back by whole words; a number leaves together with its unit."""
    # real race 18: the search field held 27 characters, every agent's longest query was cut there mid-word, and our
    # 27-character queries came back empty 116 of 188 times (.62; shorter ones .11-.26)
    if tool != ep.roles.get('search'):
        return args
    props = (ep.schemas.get(tool) or {}).get('properties') or {}
    fitted = dict(args)
    for key, text in args.items():
        spec = props.get(key)
        room = spec.get('maxLength') if isinstance(spec, dict) else None
        if not (isinstance(text, str) and isinstance(room, int) and 0 < room < len(text.strip())):
            continue
        words = text.split()
        kept: list[str] = []
        while len(kept) < len(words) and len(' '.join(kept + words[len(kept):len(kept) + 1])) <= room:
            kept.append(words[len(kept)])
        after = words[len(kept)] if len(kept) < len(words) else ''
        if kept and re.fullmatch('[0-9.,/]+', kept[-1]) and re.fullmatch('[A-Za-z"]{1,6}[.]?', after):
            kept.pop()
        while kept and kept[-1].lower() in {'and', 'or', 'with', 'for', 'of', 'the', 'a', 'in', '&', '-'}:
            kept.pop()
        if kept:
            fitted[key] = ' '.join(kept)
    return fitted

def scout_search_template(ep) -> tuple[str, dict[str, Any], str] | None:
    """Tool, kept args and text field for new searches."""
    tool = ep.roles.get('search')
    if not tool:
        return None
    schema = ep.schemas.get(tool or '') or {}
    props = schema.get('properties') or {}
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'enum' not in v and 'string' in type_names(v)]
    word_field = re.compile('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', re.I)
    # a camel-case name is read word by word as well (Arena 18550: a camel-case text field matched no pattern, and with one
    # phrase paged the field went unnamed, so every Desk search, the cheapest sweep among them, returned before sending one)
    spaced = lambda k: re.search('\\b(?:q|query|search|text|phrase|keywords?|terms?|words?|string)\\b', plain_key(k)) is not None
    named = [k for k in texts if word_field.search(k) or spaced(k)]
    worded = lambda args, key: isinstance(args.get(key), str) and re.search('[a-z]', args[key], re.I) is not None
    searched = ep.scout.searched
    for args in searched[::-1]:
        fields = [k for k in named or texts if worded(args, k)]
        # without a field named for words, only one filled with varied phrases counts as the text field
        field = next((k for k in fields if named or scout_free_field(ep, k)), None)
        if field is not None:
            return tool, scout_kept_args(ep, args, field, schema, props, [a for a in searched if worded(a, field)]), field
    return None

def scout_free_field(ep, key: str) -> bool:
    """Text field filled with varied phrases."""
    sent = [a.get(key) for a in ep.scout.searched]
    if len({str(v) for v in sent if v}) < 2 or not any(len(str(v).split()) >= 2 for v in sent):
        return False
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return True
    labels = scout_kinds(ep.mem.rows.values(), cat_key)
    return all(str(v or '').strip().lower() not in labels for v in sent)

def scout_kept_args(ep, args: dict[str, Any], field: str, schema: dict[str, Any], props: dict[str, Any], worded_args: list[dict[str, Any]]) -> dict[str, Any]:
    """Search arguments worth keeping."""
    required = schema.get('required') or []
    choice = lambda key: isinstance((props.get(key) or {}).get('enum'), list)
    cat_key = mem_kind_key(ep)
    names = [plain_key(item.shelf or '') for item in ep.items] + [plain_key(item.name) for item in ep.items]
    if cat_key:
        names += [plain_key(str(row.get(cat_key) or '')) for row in ep.mem.rows.values()]
    shelves = set(names)
    long = lambda text: {w for w in stems(text) if len(w) >= 4}
    # required arguments stay; an optional choice stays unless its value only names the product kind
    keep = {}
    for k, v in args.items():
        if k == field:
            continue
        if k not in required:
            if not choice(k):
                continue
            if isinstance(v, str) and (plain_key(v) in shelves or any(long(v) & long(s) for s in shelves if s)):
                continue
        keep[k] = v
    for key in [k for k in keep if choice(k)]:
        values = [a[key] for a in worded_args if key in a]
        if not values:
            continue
        marks = [dumps(v) for v in values]
        keep[key] = values[marks.index(max(marks, key=marks.count))]
    return keep

def scout_value_words(ep, number: int) -> list[str]:
    pointers = {m for _, m in ep.refs}
    out = []
    for n, need in ep.live_needs():
        if n != number or need in pointers or ':' not in need:
            continue
        value = scout_bare_value(need.partition(':')[2])
        if value and not has_bound(value, 'alongside|listing|page|shown|'):
            out.append(value)
    return out

HEDGES = frozenset('in the a region area range neighbourhood of around about roughly approximately near close to something like somewhere along lines'.split())

def scout_bare_value(value: str) -> str:
    """A value without the hedge words in front of it, so a search starts with the value itself. Real race 18: our value
    searches sent 'in the sauce', 'in the shoes' and 'in the mints' when a shopper put a hedge before the value."""
    words = value.strip().split()
    while len(words) > 1 and words[0].lower().strip(',') in HEDGES:
        words.pop(0)
    return ' '.join(words)

def scout_title_noun(ep, number: int) -> str:
    """Word of the item name titles carry most."""
    item = ep.items[number - 1]
    words = [w for w in re.findall('[a-z]{3,}', item.name.lower()) if w not in ('and', 'the', 'for', 'with')]
    if not words:
        return ''
    cat_key = mem_kind_key(ep)
    titles = [' '.join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(TITLE_FIELD, k, re.I)).lower()
              for record in ep.mem.rows.values() if not cat_key or mem_belongs(ep, item, str(record.get(cat_key) or ''))]
    stem = lambda w: w[:-1] if w.endswith('s') and len(w) > 4 else w
    return max(words, key=lambda w: (sum(1 for t in titles if stem(w) in t), words.index(w)))

WORD_RULE_ADVICE = {
    'first2': 'Only the first two words of a query count in this store; any later words are ignored. Lead with the wanted '
              'value, then the product noun: the two words a fitting title would have.',
    'and': 'In this store a row must contain every query word, so each added word removes rows. Use two or three words '
           'a fitting title would have: the product noun and the wanted value.',
    'or': 'In this store one matching query word is enough for a row, so each added word brings in rows of other kinds. '
          'Lead with the wanted value and use two or three words a fitting title would have.'}

def scout_pair_votes(query: tuple[str, tuple[str, ...], tuple[str, ...], Any], entry: tuple[str, tuple[str, ...], tuple[str, ...], Any],
                     apart: Any) -> list[str]:
    """Votes from one earlier search with the same other arguments and different words."""
    other, words, ids, hits = query
    e_other, e_words, e_ids, e_hits = entry
    if e_other != other or e_words == words:
        return []
    out = []
    if words[2:] != e_words[2:] and words[:2] == e_words[:2] and min(len(words), len(e_words)) > 2:
        if ids != e_ids:
            out.append('not first2')
        elif apart(set(words[2:]) ^ set(e_words[2:])):
            out.append('first2')
    # one query extends the other at its end: fewer hits -> words narrow, more -> they widen; past the second
    # word, the same rows and count -> only the first two words count
    if words[:len(e_words)] == e_words:
        pair = ((e_words, e_hits, e_ids), (words, hits, ids))
    elif e_words[:len(words)] == words:
        pair = ((words, hits, ids), (e_words, e_hits, e_ids))
    else:
        return out
    (short, s_hits, s_ids), (longer, l_hits, l_ids) = pair
    if s_hits is None or l_hits is None:
        return out
    moved = l_hits != s_hits
    if moved:
        out.append('or' if l_hits > s_hits else 'and')
    changed = moved or l_ids != s_ids
    if len(short) >= 2 and (changed or apart(longer[len(short):])):
        out.append('not first2' if changed else 'first2')
    return out

def scout_rule_votes(log: list[tuple[str, tuple[str, ...], tuple[str, ...], Any]],
                     query: tuple[str, tuple[str, ...], tuple[str, ...], Any], heads: list[str]) -> dict[str, int]:
    """Evidence on the query-word rule from comparing this search with earlier ones."""
    trim = lambda word: word[:-1] if len(word) > 4 and word.endswith('s') else word
    # a word every shown title carries leaves the rows unchanged under any rule: no evidence either way
    # (an every-word store answers "round toe wedge" and "round toe wedge pumps" alike when all rows are pumps)
    apart = lambda extra: any(trim(w) not in head for head in heads for w in extra)
    votes: dict[str, int] = {}
    for key in (k for entry in log for k in scout_pair_votes(query, entry, apart)):
        votes[key] = votes.get(key, 0) + 1
    return votes

def scout_learn_rule(ep, name: str, args: dict[str, Any], observation: Any) -> str:
    """Infer the store's query-word rule."""
    plan = scout_search_template(ep)
    paging = scout_paging_arg(ep, name)
    later_page = paging and args.get(paging) not in (None, 0, '', 1)
    if later_page or not plan or plan[0] != name:
        return ''
    field = plan[-1]
    words = tuple(str(args.get(field) or '').lower().split())
    if not words:
        return ''
    counted = observation.items() if isinstance(observation, dict) else ()
    hits = next((v for k, v in counted if isinstance(v, int) and not isinstance(v, bool)
                 and re.search('hit|count|total|found|match|tally', k, re.I)), None)
    found = mem_listings_in(ep, observation)
    rest = sorted((k, dumps(v)) for k, v in args.items() if k not in (field, paging))
    query = (dumps(rest), words, tuple(ident for ident, _ in found), hits)
    heads = [' '.join(str(v) for k, v in record.items() if isinstance(v, str) and re.search(TITLE_FIELD, k, re.I)).lower()
             for _, record in found]
    votes = scout_rule_votes(ep.scout.rule_log, query, heads)
    ep.scout.rule_log = [*ep.scout.rule_log, query][-16:]
    # an extra word that changes nothing is also what an any-word store shows, so a first-two-words reading also
    # needs a narrowing seen; one pair sharing its first two words with other rows rules it out
    narrow, wide = votes.get('and', 0), votes.get('or', 0)
    first2 = narrow and votes.get('first2', 0) and not votes.get('not first2', 0)
    rule = 'or' if wide > narrow else 'first2' if first2 else 'and' if narrow > wide else None
    if rule is None or rule == ep.scout.word_rule:
        return ''
    ep.scout.word_rule = rule
    ep.log('word_rule', rule=rule, votes=votes)
    seen = sum(1 for t in ep.trace if t.get('kind') == 'word_rule')
    return '' if seen > 3 else '\n[Desk, from the searches so far] ' + WORD_RULE_ADVICE[rule]

def scout_value_tries(values: list[str], lead: str, noun: str) -> list[str]:
    """Queries for one new value: at most two, distinct and non-empty."""
    # real races 11-14: 2-3 word searches led by words a fitting title carries surfaced the paying listing
    # first most often (.21-.39 per call vs .06-.18 for 4+ words): one value, then the product noun
    head = ' '.join(lead.split()[:2])
    with_noun = head + ' ' + noun if noun and noun not in head.lower() else head
    # within a task (all agents, r11/r13/r14), a page 1 led by two required values listed a paid listing
    # .51/.49/.42 vs .47/.43/.32 for one value and the noun; one value alone did worse than either
    partner = next((v for v in values if v != lead and len(v.split()) == 1 and v.lower() not in head.lower()
                    and not re.search('\\d', v)), None)
    tries = [with_noun, head] if partner is None or len(head.split()) != 1 else [head + ' ' + partner, with_noun]
    return [w for w in dict.fromkeys(tries) if w][:2]

def scout_value_rows(ep, item: ItemSpec, found: list[str]) -> list[str]:
    """Rows of the item's kind that can be bought within the budget (the budget is ignored when all rows are far above it)."""
    cat_key = mem_kind_key(ep)
    cap = ep.numeric_budget()
    far = cap is not None and found and all(p is not None and p > 10 * cap for p in (mem_price(ep, i) for i in found))
    keep = []
    for ident in found:
        if not mem_can_buy(ep, ident):
            continue
        if cat_key and not mem_belongs(ep, item, mem_label(ep, ident, cat_key)):
            continue
        if far or cap is None or (mem_price(ep, ident) or 0) <= cap:
            keep.append(ident)
    return keep

def scout_value_search(ep, numbers: list[int]) -> str:
    """Search the new values with the product noun."""
    plan = scout_search_template(ep)
    if not plan or ep.steps_left() < 22:
        return ''
    tool, base, field = plan
    seen: list[str] = []
    budget = 8
    schema = ep.schemas.get(tool)
    for number in numbers:
        values = scout_value_words(ep, number)
        item = ep.items[number - 1]
        settled = {t.partition(':')[2].strip() for t in item.added if ':' in t}
        lead = next(filter(settled.__contains__, values), None)
        if lead is None or budget <= 2:
            continue
        for words in scout_value_tries(values, lead, scout_title_noun(ep, number)):
            args = shaped(scout_whole_words(ep, tool, {**base, field: words}), schema)
            if ep.finished or args is None:
                break
            budget -= 1
            ok, observation = scout_fetch(ep, tool, args)
            if not ok:
                break
            rows = scout_value_rows(ep, item, [ident for ident, _ in mem_listings_in(ep, observation)])
            shown = ', '.join(rows[:6]) or 'no row of that kind'
            seen.append(f'{dumps(args.get(field))}: {shown}')
            if rows:
                opened = scout_read_pages(ep, [i for i in rows if i not in ep.mem.pages][:min(3, budget)], ep.reserve() + 16)
                budget -= len(opened)
                seen += ['opened: ' + ' | '.join(opened)[:2400]] if opened else []
                break
    if not seen:
        return ''
    ep.log('value_search', seen=[s[:200] for s in seen])
    return ('[Desk searched each newly required value together with the product noun and opened the top rows of the right '
            'kind] ' + '\n'.join(seen))

def scout_figure(word: str) -> bool:
    """A word holding a figure: digits alone, joined to letters, or a range."""
    return re.search('\\d', word) is not None

def scout_word_set(text: str) -> set[str]:
    """Words of a text with plural endings off, two-letter ones kept, linking words left out."""
    return {w for w in plain_words(text) if len(w) > 1 and w not in ('and', 'or', 'the', 'for', 'with', 'of', 'in')}

def scout_query_item(ep, words: list[str]) -> int:
    """The item whose name the query words name most, else the one whose plain required values they name; 0 on none or a tie."""
    said = {w for w in scout_word_set(' '.join(words)) if not scout_figure(w)}
    hits = [len(said & scout_word_set(item.name)) for item in ep.items]
    if not any(hits):
        hits = [len(said & scout_word_set(' '.join(scout_value_words(ep, n)))) for n in range(1, len(ep.items) + 1)]
    best = max(hits, default=0)
    return hits.index(best) + 1 if best and hits.count(best) == 1 else 0

def scout_reduced_query(ep, text: str) -> tuple[str, int]:
    """A search that found no row, cut down to the words a fitting title carries: figures (with the unit word after a bare
    number) and words naming a required property go; the words of the item's name part the query names stay (one part of
    a compound name), with at most one other word, the item's own plain value first. Returns the reduced text ('' when
    nothing is left) and the item it is for (0 when the query names none)."""
    # Arena r18 needs-in-request (runs 18508/18550): the store's search needs every query word in the title, and our model's
    # queries added limit figures (a number with its unit, joined or apart), property names or the second noun of a compound
    # item name: over runs 18357/18508/18550, 204 of 870 first-page model searches listed no row (11 of 21 on one task); the
    # reduction changes 91 of them within the retry caps, and the 10 reduced queries some later call sent all listed rows;
    # the top agent found the paying rows with short noun-plus-value queries
    words = [w.strip(',;:()[]{}"\'') for w in text.split()]
    number = scout_query_item(ep, words)
    names = scout_word_set(' '.join(item.name for item in ep.items))
    plain = lambda n: scout_word_set(' '.join(v for v in scout_value_words(ep, n) if not scout_figure(v)))
    stated = set().union(*(plain(n) for n in range(1, len(ep.items) + 1)))
    props = scout_word_set(' '.join(need.partition(':')[0] for _, need in ep.live_needs() if ':' in need)) - names - stated
    kept: list[str] = []
    after_number = False
    for at, word in enumerate(words):
        bare, own = word.lower(), scout_word_set(word)
        if scout_figure(bare):
            after_number = re.fullmatch('[0-9.,/-]+', bare) is not None
            continue
        # a short letters-only word right after a bare number is its unit, unless an item name or value has it; as the
        # query's last word only up to two letters (a closing noun after a number stays)
        unit = (after_number and re.fullmatch('[a-z"/]{1,6}[.]?', bare) is not None and not own & (names | stated)
                and (at < len(words) - 1 or len(bare) <= 2))
        after_number = False
        if own and not unit and not own <= props:
            kept.append(word)
    if not number:
        return ' '.join(kept if len(kept) <= 2 else [kept[0], kept[-1]]), 0
    item = ep.items[number - 1]
    noun = scout_title_noun(ep, number) or item.name
    said = scout_word_set(' '.join(kept))
    parts = [p for p in re.split('\\s*(?:&|\\band\\b|/|,|\\+)\\s*', item.name) if scout_word_set(p)] or [item.name]
    part = max(parts, key=lambda p: (len(said & scout_word_set(p)), bool(scout_word_set(noun) & scout_word_set(p))))
    named = [w for w in kept if scout_word_set(w) & scout_word_set(part)]
    others = [w for w in kept if not scout_word_set(w) & scout_word_set(item.name)]
    value = next((w for w in others if scout_word_set(w) & plain(number)), others[0] if others else None)
    if not named:
        return ' '.join(filter(None, [value, noun])), number
    return ' '.join(dict.fromkeys(w for w in kept if w == value or w in named)), number

def scout_no_rows(ep, observation: Any) -> bool:
    """A search result that lists no row: an empty row list or a zero count, with no listing in it."""
    if not isinstance(observation, dict) or mem_listings_in(ep, observation):
        return False
    counted = [v for k, v in observation.items() if is_count(v) and re.search('hit|count|total|found|match|tally|overall', k, re.I)]
    return any(v == 0 for v in counted) or any(isinstance(v, list) and not v for v in observation.values())

def scout_query_field(ep, tool: str, args: dict[str, Any]) -> str | None:
    """The argument holding a search's free words."""
    plan = scout_search_template(ep)
    if plan and plan[0] == tool and isinstance(args.get(plan[2]), str):
        return plan[2]
    props = (ep.schemas.get(tool) or {}).get('properties') or {}
    free = [k for k, v in args.items() if isinstance(v, str) and re.search('[a-z]', v, re.I)
            and not (isinstance(props.get(k), dict) and 'enum' in props[k]) and not SELLER_ARG.search(k)]
    return free[0] if len(free) == 1 else None

def scout_zero_retry(ep, job: Call, observation: Any) -> str:
    """A model search that listed no row is sent again once, in code, with the reduced query; its rows go back with the result."""
    scout, tool = ep.scout, job.name
    if (tool != ep.roles.get('search') or not scout_no_rows(ep, observation) or scout.word_rule == 'or' or ep.finished
            or len(scout.retried) >= ZERO_RETRIES or ep.steps_left() - ep.reserve() < 4 or ep.past_stop()):
        return ''
    paging = scout_paging_arg(ep, tool)
    field = scout_query_field(ep, tool, job.args)
    if field is None or paging and not screen_start_value(job.args.get(paging)):
        return ''
    said = ' '.join(str(job.args[field]).lower().split())
    words, number = scout_reduced_query(ep, said)
    if not words or sum(1 for n, _ in scout.retried if n == number) >= 2:
        return ''
    args = shaped(scout_whole_words(ep, tool, {**job.args, field: words}), ep.schemas.get(tool))
    query = ' '.join(str((args or {}).get(field) or '').lower().split())
    sent = {' '.join(str(a.get(field) or '').lower().split()) for name, a, _, _ in ep.screen.page_log if name == tool}
    sent |= {' '.join(str(a.get(field) or '').lower().split()) for a in scout.searched} | {q for _, q in scout.retried}
    if args is None or not query or query == said or query in sent:
        return ''
    scout.retried = [*scout.retried, (number, query)]
    ok, seen = scout_fetch(ep, tool, args)
    ep.log('zero_retry', item=number, sent=said, words=query, rows=len(mem_listings_in(ep, seen)) if ok else None)
    if not ok:
        return ''
    found = mem_listings_in(ep, seen)
    if found:
        scout.searched = list(scout.searched)[-7:] + [dict(args)]
    shown = ep.render(seen)
    ep.cache[tool + dumps(args)] = (ep.version, ep.step, shown)
    head = (f'\n[Desk retried this search once (one step) with fewer words, since one word no title carries can empty a result: '
            f'{field}={query!r}, figures and property names left out]')
    if not found:
        return head + ' It listed no row either: search the product noun alone or with one plain value.'
    return head + '\n' + shown

class Interviewer:
    """Questions to the shopper and their answers."""

    def __init__(self) -> None:
        self.started = False
        self.plan_order: list[int] = []

def talk_due(ep) -> bool:
    if ep.talk.started or not ep.roles.get('ask') or ep.free_messages() <= 0 or ep.steps_left() <= 8:
        return False
    return ep.step >= max(8, ep.max_steps // 4) or len(ep.mem.pages) > 3 * len(ep.items)

def talk_msg_args(ep, text: str) -> dict[str, Any] | None:
    schema = ep.schemas.get(ep.roles.get('ask') or '') or {}
    props = schema.get('properties') or {}
    needed = schema.get('required') or []
    texts = [k for k, v in props.items() if isinstance(v, dict) and 'enum' not in v and 'string' in type_names(v)]
    target = next((k for k in needed if k in texts), None) or (texts[0] if texts else None)
    if not target:
        return None
    # every other required argument takes its first allowed value; one without a value list leaves no valid call
    args: dict[str, Any] = {target: text}
    for key in needed:
        if key in args:
            continue
        options = (props.get(key) or {}).get('enum')
        if not options or not isinstance(options, list):
            return None
        args[key] = options[0]
    return shaped(args, schema)

def talk_tally(ctx: dict[str, Any], batch: list[tuple[int, str, str]], heard: dict[str, Any]) -> list[int]:
    ep, found = ctx['ep'], ctx['found']
    before = [len(item.added) for item in ep.items]
    talk_apply_answers(ep, heard, ctx['told'], ' '.join(reply for *_, reply in batch))
    fresh = [n for n, (had, item) in enumerate(zip(before, ep.items), 1) if had < len(item.added)]
    found.update({n: found.get(n, 0) + 1 for n in fresh})
    return fresh

def talk_regroup(ctx: dict[str, Any]) -> list[list[int]]:
    """When products with no need yet outnumber the messages left, they are asked together, least asked first."""
    ep, found = ctx['ep'], ctx['found']
    used = ctx['used']
    waiting = [n for n, lane in ctx['lanes'].items() if not found.get(n, 0) and any(plain_key(x) not in used[n] for x in lane)]
    left = ep.free_messages()
    if not ctx['listing'] or not ctx['spare'] or left < 1 or len(waiting) <= left:
        return []
    rank = {n: i for i, n in enumerate(ep.talk.plan_order)}
    spent = ctx['spent']
    waiting.sort(key=lambda n: (spent.get(n, 0), rank.get(n, len(rank) + n)))
    return [waiting[:(len(waiting) + left - 1) // left]]

def talk_group_turn(ctx: dict[str, Any], groups: list[tuple[int, list[str]]]) -> bool:
    """One message naming a few keys of several products; False when nothing went out."""
    ep = ctx['ep']
    text = talk_group_text(ep, groups)
    if text not in (talk_msg_args(ep, text) or {}).values():
        # the message tool caps its text: one key per product
        groups = [(n, labels[:1]) for n, labels in groups]
        text = talk_group_text(ep, groups)
    for n, labels in groups:
        ctx['spent'][n] += 1
        ctx['used'][n].update(map(plain_key, labels))
    batch = talk_send(ep, [(0, text)])
    if not batch:
        return False
    ctx['sent'].extend(batch)
    # a reply answers one of the items at most, so a "don't mind" cannot be pinned to an item
    heard = dict(talk_read_answers(ep, batch), free=[])
    split = any(whole(x) == 1 for x in as_list(heard.get('crowded')))
    fresh = talk_tally(ctx, batch, heard)
    if split:
        # the shopper wants one thing per message: single questions from now on, the first unanswered item first
        ctx['rounds'], ctx['listing'] = [], False
        first, labels = next(((n, x) for n, x in groups if n not in fresh), (0, []))
        if first:
            ctx['used'][first].discard(plain_key(labels[0]))
            ctx['follow'].insert(0, (first, labels[:1]))
        ep.log('items_split', items=[n for n, _ in groups], found=fresh)
    return True

def talk_single_turn(ctx: dict[str, Any], number: int, labels: list[str]) -> bool:
    """One message about one product (a single key or a list); False when nothing went out."""
    ep, lane = ctx['ep'], ctx['lanes'][number]
    text = talk_text_for(ep, number, labels)
    while len(labels) > 3 and text not in (talk_msg_args(ep, text) or {}).values():
        # the message tool caps its text: the longest list it takes, best-ranked properties kept
        labels = labels[:-1]
        text = talk_text_for(ep, number, labels)
    ctx['rested'] = 0
    keys = [plain_key(x) for x in labels]
    taken, answered = ctx['used'][number], ctx['answered'][number]
    taken.update(keys)
    ctx['fronted'][number].update(keys[:LIST_FRONT])
    batch = talk_send(ep, [(number, text)])
    if not batch:
        return False
    ctx['sent'].extend(batch)
    ctx['spent'][number] += 1
    heard = talk_read_answers(ep, batch)
    # the refusal is read before the need is counted: a crowded reply that also gives a need still ends the lists
    split = any(whole(x) == 1 for x in as_list(heard.get('crowded')))
    if len(labels) > 2:
        # a reply to a list answers one of its properties at most: a "don't mind" names none of them
        heard = dict(heard, free=[])
        heads = (stems(str(e.get('need') or '').split(':', 1)[0]) for e in as_list(heard.get('needs')) if isinstance(e, dict))
        for prop in filter(None, heads):
            answered.update(plain_key(x) for x in lane if names_same(prop, stems(x)) or names_same(stems(x), prop))
    if not split and len(labels) > 1:
        # the keys after the one that drew a need, then those before it, lead the next list; with no need the first
        # key is settled and the others lead
        hit = next((i for i, x in enumerate(labels) if plain_key(x) in answered), 0)
        rest = [x for x in ctx['missed'][number] if x not in labels]
        ctx['missed'][number] = labels[hit + 1:] + labels[:hit] + rest
    fresh = talk_tally(ctx, batch, heard)
    if ctx['listing'] and split and len(labels) > 1:
        # the shopper wants one property per message: single questions from now on, the list's head first
        ctx['listing'] = False
        taken.difference_update(keys[1:])
        if number not in fresh:
            ctx['follow'].insert(0, (number, labels[:1]))
        ep.log('list_split', item=number, found=bool(fresh))
        return True
    if fresh:
        return True
    words = labels[0].split()
    # after a reply wanting one question per message only the head word of a longer label earns a second try
    # (.28 in real races 10-13); the same label again, with or without examples, almost never draws a value
    if split and len(words) > 1 and plain_key(words[-1]) not in taken:
        ctx['follow'].insert(0, (number, words[-1:]))
        return True
    blank = bool(heard) and heard.get('budget') is None and not any(as_list(heard.get(k)) for k in ('needs', 'drop', 'free'))
    if ctx['many'] and (split or blank):
        # a reply with nothing in it, or wanting the message split, seldom turns into a value when the same item
        # is asked again: the next message goes to another item still without a value, and this one waits a turn
        ctx['rested'] = number
        ep.log('ask_rest', item=number, split=split)
    return True

def talk_run(ep) -> str:
    """Plan, ask and read the answers; returns a summary."""
    # adaptive L2 (user OK 11-08): lists at every cap; on 1 product a single first (strict-safe), then 5 or 3 keys, then
    # 8; on 2+ products an 8-key list per product, or grouped messages of 2 keys per product when products outnumber the
    # messages; any refusal turns the rest into single questions. Real r17 (lenient, cap 2): list askers missed a need in
    # .21-.27 of episodes vs .47-.51 for single askers (ours .47); the same agent passed .36 with lists vs .28 with singles.
    # Strict r16: single askers .15-.17 vs list openers .10-.12 (review 2026-11-08-1720-asking-engine-review.md)
    ep.talk.started = True
    if ep.free_messages() <= 0:
        return ''
    req_profile_now(ep)
    scout_widen(ep)
    lanes = talk_plan(ep, ep.free_messages())
    # a list names the item's own listing keys after the planned picks, most carried first, in the store's spelling
    # (real r15: "packaging type" drew nothing where the need was "pack type"; both keys were listed)
    table = mem_key_table(ep)
    for n, lane in lanes.items():
        have = {plain_key(x) for x in lane}
        lanes[n] = lane + [k for k in talk_own_keys(ep, n, table) if plain_key(k) not in have]
    free, many = ep.free_messages(), len(ep.items) > 1
    # an unknown cap is planned as 2 messages: a third may not exist
    plan_cap = free if ep.cap_stated else min(free, 2)
    # keys a list named without drawing a need lead the item's next list from place 1 (re-listed at place 1 they drew
    # the need .95-1.00: real r15 18/19 and 16/16, r17 8/8; asklab: lenient caps 3/4 +7/+11 of 141 tasks)
    ctx: dict[str, Any] = {'ep': ep, 'lanes': lanes, 'found': dict.fromkeys(lanes, 0), 'spent': dict.fromkeys(lanes, 0),
                           'used': {n: set() for n in lanes}, 'answered': {n: set() for n in lanes},
                           'fronted': {n: set() for n in lanes}, 'missed': {n: [] for n in lanes},
                           'follow': [], 'sent': [], 'told': [], 'listing': True, 'rested': 0, 'many': many,
                           'rounds': talk_spread_rounds(ep, lanes, plan_cap) if many else [],
                           # a late regroup only when messages outnumber the products at the start; products =
                           # messages: one product each
                           'spare': many and sum(1 for v in lanes.values() if v) < plan_cap}
    ep.log('ask_form', form='grouped' if ctx['rounds'] else 'list', messages=ep.free_messages(), plan_cap=plan_cap,
           items=len(lanes))
    found, spent, used = ctx['found'], ctx['spent'], ctx['used']
    while not ep.finished and ep.free_messages() > 0 and ep.steps_left() > 6:
        follow = ctx['follow']
        if not (follow or ctx['rounds']):
            ctx['rounds'] = talk_regroup(ctx)
        groups = []
        if ctx['rounds'] and not follow:
            for n in ctx['rounds'].pop(0):
                labels = [x for x in lanes[n] if plain_key(x) not in used[n]][:GROUP_KEYS]
                if labels:
                    groups.append((n, labels))
        if len(groups) > 1:
            if talk_group_turn(ctx, groups):
                continue
            break
        if follow:
            number, labels = follow.pop(0)
        elif groups:
            # a lone product in its slice: one list of its best keys
            number = groups[0][0]
            labels = talk_list_pick(lanes[number], used[number], ctx['answered'][number], ctx['fronted'][number],
                                    ctx['missed'][number])
        else:
            number, labels = talk_choose_next(ep, lanes, found, spent, used, ctx['answered'], ctx['fronted'],
                                              ctx['rested'], ctx['listing'], ctx['missed'])
            if not labels:
                break
            if ctx['listing']:
                labels = labels[:talk_list_size(spent.get(number, 0), plan_cap, many)]
        if not talk_single_turn(ctx, number, labels):
            break
    return talk_summary(ep, found, ctx['sent'], ctx['told'])

def talk_list_size(asked: int, plan_cap: int, many: bool) -> int:
    """List length for an item already asked `asked` times. With 2+ products: 8. With 1 product: a single key first
    (a strict reader turns down an opening list: r16 first lists of 3-4 properties drew a need .08, singles .28), then
    5 keys when there are 2 messages or 3 when there are more, then 8; a single message overall gets one 8-key list."""
    if not many and plan_cap > 1 and asked < 2:
        return 1 if not asked else 5 if plan_cap <= 2 else 3
    return LIST_KEYS

def talk_choose_next(ep, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]],
                answered: dict[int, set[str]], fronted: dict[int, set[str]], rested: int, listing: bool,
                missed: dict[int, list[str]] | None = None) -> tuple[int, list[str]]:
    """Item and labels for the next question."""
    # a resting item gives way only to items with nothing drawn yet: real r8-r14 multi-product
    # tasks, asking a product that already gave a value drew another .003-.043, one still
    # without a value after an empty reply .11-.42
    waiting = {n: labels for n, labels in lanes.items() if n != rested and not found.get(n, 0)}
    number, labels = talk_next_single(ep, waiting, found, spent, used) if waiting else (0, [])
    if not labels:
        number, labels = talk_next_single(ep, lanes, found, spent, used)
    if not listing:
        return number, labels
    ahead = (lambda n: None) if missed is None else (lambda n: missed.get(n, []))
    pick = lambda n: talk_list_pick(lanes[n], used[n], answered[n], fronted[n], ahead(n))
    if not labels:
        # every planned property went out once: list again the ones that drew no answer
        ranked = sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0)))
        number = next((n for n in ranked if pick(n)), 0)
    return (number, pick(number)) if number else (number, [])

def talk_summary(ep, found: dict[int, int], sent: list[tuple[int, str, str]], told: list[str]) -> str:
    drawn = [n for n, hits in found.items() if hits > 0]
    read_before = list(ep.mem.pages)
    searched = scout_value_search(ep, drawn) if drawn else ''
    if told:
        ep.version += 1
    parts = ["Desk's interviewer put these questions to the shopper (one step each):\n"
             + '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in sent)
             + '\nFirm requirements added by these answers: ' + ('; '.join(told) or 'none')
             + '. Each pick has to meet them (search again with the new values where needed).']
    pointed = [need for _, need in ep.refs if any(need in t for t in told)]
    if pointed:
        parts.append(' Their value sits in another listing: ' + ' | '.join(pointed) + '. The order only counts after all of these '
                     'are open: find that listing, open its full page and each listing it points on to; the value is on the last page.')
    order = list(dict.fromkeys(ep.scout.widened))
    widened = set(order)
    ep.scout.widened = []
    pages = [ep.mem.pages[i] for i in order if i in ep.mem.pages]
    if pages:
        # pages read so the questions fit the store: the model has not seen them in its own results
        body = '\n'.join(ep.render(page)[:1500] for page in pages)
        parts.append('\n[pages Desk read while getting the questions ready, which you have not seen before (one step each)]\n'
                     + body[:8000])
        proof = judge_evidence(ep, pages)
        if proof:
            parts.append('\n' + proof)
    again = talk_recheck_read(ep, [i for i in read_before if i not in widened]) if told else ''
    parts += ['\n' + extra for extra in (again, searched) if extra]
    if ep.queued:
        parts.append('\n[notes seen meanwhile] ' + ' | '.join(ep.queued))
        ep.queued = []
    return ''.join(parts)

def talk_recheck_read(ep, ids: list[str]) -> str:
    """Evidence on pages read before the answers."""
    # real races 5-14: of failed picks that had opened the paid listing, .35-.57 opened it
    # only before the hidden need was drawn and never judged it against that need
    kinds = {i for n in range(1, len(ep.items) + 1) for i in mem_ids_of_item(ep, n)}
    wanted = set(ids)
    pool = [i for i in ep.mem.order[::-1]
            if i in wanted and (not kinds or i in kinds) and mem_can_buy(ep, i) and i not in ep.gone]
    cheapest = ep.cheapest()
    if cheapest:
        pool.sort(key=ep.price_rank)
    proof = judge_evidence(ep, [ep.mem.pages[i] for i in pool[:8]], False)
    if not proof:
        return ''
    return (f"[listings you opened before these answers{', cheapest first' if cheapest else ''}, keys named for a requirement in "
            'force: check them again against the new requirements before any further search] ' + proof)

def talk_next_single(ep, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
    rank = {n: i for i, n in enumerate(ep.talk.plan_order)}
    for n in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), rank.get(n, len(rank) + n))):
        taken = used[n]
        first = next((label for label in lanes[n] if plain_key(label) not in taken), None)
        if first is not None:
            return n, [first]
    return 0, []

def talk_list_pick(lane: list[str], used: set[str], answered: set[str], fronted: set[str], ahead: list[str] | None = None) -> list[str]:
    """Labels of one list question: first the keys an earlier list missed (ahead, when given), then unused keys, then the others."""
    # real race 14 lists: a property in places 1-3 drew its need .79, in 4+ .43; one named only
    # further back before drew .58 when moved to the front, .33 when left behind
    pool: dict[str, str] = {}
    for label in lane:
        word = plain_key(label)
        if word not in answered and word not in pool and not re.fullmatch('options?', word):
            pool[word] = label
    labels = list(pool.values())
    if ahead is None:
        held = used - fronted
        front = [label for word, label in pool.items() if word not in used]
        front = (front + [label for word, label in pool.items() if word in held])[:LIST_FRONT]
        ordered = front + [x for x in labels if x not in front]
    else:
        lead = [x for x in ahead if x in labels]
        fresh = [x for x in labels if plain_key(x) not in used and x not in lead]
        ordered = lead + fresh + [x for x in labels if x not in lead and x not in fresh]
    return ordered[:LIST_KEYS]

def talk_spread_rounds(ep, lanes: dict[int, list[str]], cap: int | None = None) -> list[list[int]]:
    """Spread items over too few messages."""
    items = [n for n in sorted(lanes) if lanes[n]]
    free = cap if cap is not None else ep.free_messages()
    if free <= 0 or len(items) <= free:
        return []
    size, extra = divmod(len(items), free)
    rounds, start = [], 0
    for k in range(free):
        end = start + size + (1 if k < extra else 0)
        if end > start:
            rounds.append(items[start:end])
        start = end
    return rounds

def talk_group_text(ep, groups: list[tuple[int, list[str]]]) -> str:
    """One message asking several items, each with its own keys."""
    parts = []
    for n, labels in groups:
        spoken = ' or '.join(re.sub('[_]+', ' ', label).strip() for label in labels)
        parts.append(f"On the {ep.items[n - 1].name}: which {spoken} should it have?")
    return ' '.join(parts)

def talk_text_for(ep, number: int, labels: list[str]) -> str:
    item = ep.items[number - 1]
    spoken = [re.sub('[_]+', ' ', label).strip() for label in labels]
    if len(spoken) <= 2:
        return f"Which {' or '.join(spoken)} should the {item.name} have?"
    where = item.name
    if item.shelf and plain_key(item.shelf) != plain_key(item.name):
        where = f'{item.name} ({item.shelf})'
    return f"For the {where}, what {', '.join(spoken[:-1])} or {spoken[-1]} do you want?"

def talk_own_keys(ep, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
    """Item's unsettled listed keys, best first."""
    settled = [stems(m.split(':', 1)[0]) for m in ep.items[number - 1].musts]
    mine = [row for row in table if mem_owner(ep, row[0]) == number] or (list(table) if len(ep.items) == 1 else [])
    measured = {key: sum(1 for v in values if re.match(MEASURE_VALUE, v)) * 2 > len(values) for _, key, _, values in table}
    ranked = sorted(mine, key=lambda r: (-r[2], measured.get(r[1], False), -len(r[3])))
    keys = [key for _, key, _, _ in ranked
            if not re.search(NOT_PROPERTY, key) and all(not names_same(prop, stems(key)) for prop in settled)]
    return sorted(keys, key=lambda k: mem_is_catch_all(ep, k))

def talk_plan_prompt(ep, count: int, table: list[tuple[str, str, int, list[str]]]) -> str:
    settled = '\n'.join(f"item {n}: {item.name} - settled by the request: {'; '.join(item.musts) or 'nothing'}"
                        for n, item in enumerate(ep.items, 1))
    listing = []
    for index, (label, name, carried, values) in enumerate(table, 1):
        if index == 1 or table[index - 2][0] != label:
            listing.append(f'category {label!r}:')
        clash = mem_stated_clash(ep, label, name, values)
        mixed = ' [several features in one: ask last]' if listy(values) else ''
        listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + clash + mixed)
    blocks = [PLAN.replace('@N@', str(max(1, count))), 'REQUEST:\n' + ep.request, 'ITEMS:\n' + settled]
    asks = [str(x) for x in as_list(ep.brief.get('ask_first')) if str(x).strip()]
    if asks:
        blocks.append('ASK FIRST (the request asks to be asked about these):\n' + '\n'.join(asks))
    blocks.append('ATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)'))
    return '\n\n'.join(blocks)

def talk_read_picks(ep, got: dict[str, Any]) -> tuple[dict[int, list[str]], dict[int, set[str]], list[int]]:
    size = len(ep.items)
    picked: dict[int, list[str]] = {n: [] for n in range(1, size + 1)}
    hints: dict[int, set[str]] = {n: set() for n in picked}
    order: list[int] = []
    for pick in as_list(got.get('ask')):
        if not isinstance(pick, dict):
            continue
        number = pick_number(pick.get('item') or 1, size)
        label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
        # a listing's bare purchase-option field is no property of the product: the question would earn a shrug
        if not label or len(label.split()) > 5 or (P2_ON and re.fullmatch('options?', plain_key(label))):
            continue
        key = plain_key(label)
        lane = picked[number]
        if all(plain_key(x) != key for x in lane):
            lane.append(label)
        if str(pick.get('hint') or '').strip():
            hints[number].add(key)
            if number not in order:
                order.append(number)
    return picked, hints, order

def talk_plan_samples(ep, prompt: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The question plan and, when affordable, a second sample from the other model."""
    llm = ep.llm
    ask = lambda models: llm.ask(prompt, models, 3500)
    if not P1_ON:
        return ask([HEAVY, LIGHT]) or {}, None
    # the same PLAN prompt to both models: a property both choose is a surer first question than one only one picks
    if ep.elapsed() < SIDE_BY_SIDE_UNTIL:
        # both samples are requested together, so the second model costs no wait of its own (r15 Arena logs:
        # the two ran one after the other on every task, roughly 8 s per episode)
        with ThreadPoolExecutor(max_workers=2) as workers:
            jobs = [workers.submit(llm.ask, prompt, [model], 3500) for model in (HEAVY, LIGHT)]
            first, other = (job.result() for job in jobs)
        return (first, other) if first is not None else (other or {}, None)
    first = ask([HEAVY])
    if first is None:
        return ({}, None) if llm.stopped() else (ask([LIGHT]) or {}, None)
    return first, ask([LIGHT]) if llm.cost <= SPEND_SOFT else None

def talk_carried(ep, number: int, table: list[tuple[str, str, int, list[str]]]) -> dict[str, float]:
    """Share of the item's read listings carrying each of its keys."""
    read = max(1, mem_read_count(ep, number))
    own: dict[str, float] = {}
    for label, key, carried, _ in table:
        if mem_owner(ep, label) == number:
            own[key] = max(own.get(key, 0.0), carried / read)
    return own

def talk_add_unique(lane: list[str], picks: list[str]) -> list[str]:
    """The lane followed by each pick whose plain spelling it does not hold yet."""
    known = {plain_key(x): None for x in lane}
    extra = [known.setdefault(plain_key(p), p) for p in picks if plain_key(p) not in known]
    return lane + extra

def talk_plan(ep, count: int) -> dict[int, list[str]]:
    """Ordered properties to ask about, per item."""
    table = mem_key_table(ep)
    plan, second = talk_plan_samples(ep, talk_plan_prompt(ep, count, table))
    lanes, hinted, order = talk_read_picks(ep, plan)
    lanes2, hinted2, order2 = talk_read_picks(ep, second) if second else ({}, {}, [])
    known = ep.talk.plan_order
    for n in [*order, *order2]:
        if n not in known:
            known.append(n)
    for number, picks in list(lanes.items()):
        keys = talk_own_keys(ep, number, table)
        own = talk_carried(ep, number, table)
        hints = {plain_key(talk_spelled(h, own)) for h in hinted[number] | hinted2.get(number, set())}
        lane = talk_add_unique([], [talk_spelled(p, own) for p in picks])
        if lanes2.get(number):
            lane = talk_merge_samples(lane, lanes2[number], own)
        guarded = talk_add_unique(talk_guard_lane(ep, lane, keys, own, hints), keys)
        lanes[number] = lane = sorted(guarded, key=lambda x: mem_is_catch_all(ep, x))
        if P3_ON and lane and count <= 2 and plain_key(lane[0]) in hints:
            talk_measure_swap(ep, number, lane, hints, keys, table, max(1, count // len(ep.items)))
    ep.log('plan', ask=clip(plan.get('ask'), 120), lanes={n: v[:8] for n, v in lanes.items()},
           **({'ask2': clip(second.get('ask'), 120)} if second else {}))
    return lanes

def talk_spelled(label: str, own: dict[str, float]) -> str:
    """Item's own key for an alternate spelling."""
    def plain(text: str) -> str:
        return re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', plain_key(text)))
    wanted = plain(label)
    for k in own:
        if plain(k) == wanted:
            return k
    return label

def talk_share_of(label: str, own: dict[str, float]) -> float:
    if plain_key(label) in own:
        return own[plain_key(label)]
    mine = stems(label)
    near = [s for k, s in own.items() if names_same(mine, stems(k)) or names_same(stems(k), mine)]
    return max(near) if near else 0.0

def talk_merge_samples(lane: list[str], other: list[str], own: dict[str, float]) -> list[str]:
    """Merge the two planner samples."""
    spelled = [talk_spelled(pick, own) for pick in other]
    theirs = {plain_key(x) for x in spelled}
    both = [x for x in lane if plain_key(x) in theirs]
    seen = [plain_key(x) for x in lane]
    extra: list[str] = []
    for name in spelled:
        if plain_key(name) not in seen:
            extra.append(name)
            seen.append(plain_key(name))
    extra.sort(key=lambda x: -talk_share_of(x, own))
    return both + [x for x in lane if x not in both] + extra

def talk_guard_place(lane: list[str], key: str, weak: int | None, many: bool) -> list[str]:
    """Lane with the key inserted at the weak place (on a multi-item task the pick there moves to fourth), otherwise at the end."""
    target = plain_key(key)
    placed = [other for other in lane if plain_key(other) != target]
    if weak is None:
        return placed + [key]
    if not many:
        return placed[:weak] + [key] + placed[weak:]
    placed[weak], moved = key, placed[weak]
    placed.insert(min(3, len(placed)), moved)
    return placed

def talk_guard_lane(ep, lane: list[str], keys: list[str], own: dict[str, float], hinted: set[str]) -> list[str]:
    """Let widely carried keys into the top three."""
    strong = [k for k in keys if own.get(k, 0.0) >= GUARD_SHARE and not mem_is_catch_all(ep, k)][:GUARD_KEYS]
    many = len(ep.items) > 1
    for key in strong:
        top = lane[:3]
        if plain_key(key) in {plain_key(x) for x in top}:
            continue
        need = own.get(key, 0.0)
        weak = next((place for place, other in enumerate(top)
                     if plain_key(other) not in hinted and talk_share_of(other, own) + GUARD_GAP <= need), None)
        if weak is None and len(lane) > 2:
            continue
        placed = talk_guard_place(lane, key, weak, many)
        kept = {plain_key(x) for x in placed[:3]}
        if not any(plain_key(x) in hinted and plain_key(x) not in kept for x in top):
            lane = placed
    return lane

def talk_measure_swap(ep, number: int, lane: list[str], hinted: set[str], keys: list[str], table: list[tuple[str, str, int, list[str]]], room: int) -> None:
    """Move a measure-valued hinted key off first place."""
    measure = lambda vals: bool(vals) and sum(1 for v in vals if re.match(MEASURE_VALUE, v)) * 2 > len(vals)
    mine = [(key, values) for label, key, _, values in table if mem_owner(ep, label) == number]
    head = plain_key(lane[0])
    if not measure(next((values for key, values in mine if key == head), [])):
        return
    shown = dict(mine)
    going = {plain_key(x) for x in lane[:room]}
    for k in keys:
        pk = plain_key(k)
        if pk not in hinted and pk not in going and not measure(shown.get(k, [])) and not mem_is_catch_all(ep, k):
            lane[:] = [k, *(x for x in lane if plain_key(x) != pk)]
            ep.log('measure_first', item=number, first=k, moved=lane[1])
            return

def talk_send(ep, questions: list[tuple[int, str]], floor: int = 6) -> list[tuple[int, str, str]]:
    tool = ep.roles['ask']
    sent: list[tuple[int, str, str]] = []
    queue = list(questions)
    while queue and ep.msg_sent < ep.msg_cap and not ep.finished and ep.steps_left() > floor:
        number, text = queue.pop(0)
        args = talk_msg_args(ep, text)
        reply = ep.store.call(tool, args) if args else None
        if reply is None:
            break
        ep.msg_sent += 1
        answer = reply.get('observation')
        words = talk_words_of(answer)
        sent.append((number, text, words))
        gates_note_line(ep, f'step {ep.step}: asked the shopper {text!r}, reply {words[:300]!r} (cart at that time: {ep.cart_stamp()})',
                        ep.step)
        ep.log('ask', q=text, a=words[:300])
        # a reply carrying an error ends the round of questions
        if isinstance(answer, dict) and answer.get('error'):
            break
    return sent

def talk_budget_as_written(ep, value: float, said: str) -> Any:
    """Budget as the answers actually write it."""
    figures = {float(raw.replace(',', '')) for raw in re.findall('\\d[\\d,]*(?:\\.\\d+)?', said)} - {0.0}
    if any(abs(value - f) <= 1e-09 * max(1.0, f) for f in figures):
        return value
    # a figure off by a power of ten is the same budget read in another unit
    shifted = [f for f in figures if any(abs(f / (value * 10 ** k) - 1) < 0.002 for k in (-3, -2, -1, 1, 2, 3))]
    if not shifted:
        ep.log('budget_unwritten', read=value)
        return ep.budget
    if len(shifted) > 1:
        return value
    written = shifted[0]
    old = ep.budget if is_number(ep.budget) and ep.budget > 0 else None
    if old and max(written / old, old / written) >= max(value / old, old / value):
        return value
    ep.log('budget_written', read=value, written=written)
    return written if not written.is_integer() else int(written)

def talk_apply_answers(ep, heard: dict[str, Any], told: list[str], said: str = '') -> None:
    size = len(ep.items)
    entries = lambda field: [e for e in as_list(heard.get(field)) if isinstance(e, dict)]
    for entry in entries('needs'):
        need = str(entry.get('need') or '').strip()
        if not need:
            continue
        number = pick_number(entry.get('item'), size)
        added = ep.items[number - 1].added
        if need not in added:
            added.append(need)
            told.append(f'item {number}: {need}' if size > 1 else need)
        if entry.get('points') is True and (number, need) not in ep.refs:
            ep.refs.append((number, need))
    for entry in entries('drop'):
        if not str(entry.get('property') or '').strip():
            continue
        item = ep.items[pick_number(entry.get('item'), size) - 1]
        head = plain_key(str(entry['property']).split(':')[0])
        hit = [m for m in item.live() if plain_key(m.split(':')[0]) == head]
        item.dropped += hit
        told.extend(f'dropped {m}' for m in hit)
    for entry in entries('free'):
        if entry.get('property'):
            ep.items[pick_number(entry.get('item'), size) - 1].free.append(str(entry['property']))
    budget = heard.get('budget')
    if is_number(budget) and budget > 0:
        ep.budget = talk_budget_as_written(ep, budget, said)

def talk_read_answers(ep, sent: list[tuple[int, str, str]]) -> dict[str, Any]:
    items = '\n'.join(f'item {n}: {it.name}' for n, it in enumerate(ep.items, 1))
    asked = '\n'.join(f"{i}. ({f'about item {n}' if n else 'about the items it names'}) Q: {q}\nA: {a}"
                      for i, (n, q, a) in enumerate(sent, 1))
    blocks = [HEAR, 'ITEMS:\n' + items, 'QUESTIONS AND ANSWERS:\n' + asked]
    return ep.llm.ask('\n\n'.join(blocks), [LIGHT, HEAVY], 2500) or {}

def talk_texts_in(node: Any, depth: int = 0) -> list[str]:
    """Multi-word strings of a nested reply, depth first, at most four levels down."""
    if depth <= 4 and isinstance(node, str):
        return [node] * (len(node.split()) > 1)
    if depth > 4 or not isinstance(node, (dict, list)):
        return []
    out: list[str] = []
    for child in (node.values() if isinstance(node, dict) else node):
        out.extend(talk_texts_in(child, depth + 1))
    return out

def talk_words_of(observation: Any) -> str:
    """Multi-word texts of a reply, in order."""
    if isinstance(observation, str):
        return observation
    found = talk_texts_in(observation)
    if not found:
        return dumps(observation)[:400]
    return ' '.join(found)[:800]

class ChangeWatcher:
    """Notes inside results and what they change."""

    def __init__(self) -> None:
        self.sold_flag = self.named_gone = False
        self.heard: set[str] = set()
        self.sold_lines: set[str] = set()

def news_free_lines(ep, observation: Any) -> list[str]:
    """Unread sentences of a result."""
    if not isinstance(observation, dict) or mem_identify(ep, observation):
        return []
    heard = ep.news.heard
    lines = []
    for value in observation.values():
        if isinstance(value, str):
            found = [value] * (len(value.split()) >= 5)
        elif isinstance(value, list):
            found = [v for v in value if len(v.split()) >= 5] if value and all(isinstance(v, str) for v in value) else []
        elif isinstance(value, dict) and not mem_identify(ep, value, False, 1):
            found = [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        else:
            continue
        lines += [line for line in found if line not in heard]
    return lines

def news_reopen_items(ep, got: dict[str, Any], gone: list[str], known: list[str]) -> None:
    """Items whose requirement was dropped, or whose basket line sold out, go back to the pick."""
    count = len(ep.items)
    ep.repick.update(pick_number(e.get('item'), count) for e in as_list(got.get('drop'))
                     if isinstance(e, dict) and str(e.get('need') or '').strip())
    # a sold-out line re-opens its item the same way: the next cheapest fit may be a row never read (r15 b05: 3 of 18
    # not-best orders followed a sold-out refusal)
    cat_key = mem_kind_key(ep)
    # a moved price can hand the cheapest fit to another listing (real race 18: drop+price refusals paid .02 field-wide;
    # the paying row was a cheap one nobody had shown, e.g. a twin listing still at the old price)
    if cat_key and got.get('price_moved') is True:
        ep.repick.update(filter(None, (mem_owner(ep, mem_label(ep, i, cat_key)) for i in basket_ids(ep))))
    if not cat_key or not (gone or ep.news.sold_flag):
        return
    lost = [i for i in basket_ids(ep) if ep.is_gone(i)] or known
    ep.repick.update(filter(None, (mem_owner(ep, mem_label(ep, i, cat_key)) for i in lost)))

def news_hunt_again(ep, gone: list[str], changed: list[str]) -> bool:
    """A lower budget keeps the cheapest fit cheapest while its line stays inside it (real races 5-14: a still-fitting
    pick kept after a budget note passed .93, dropped .67), so only a line it breaks reopens the hunt."""
    if gone or ep.news.sold_flag or any(c.startswith('dropped ') for c in changed):
        return True
    if not any(c.startswith('budget now') for c in changed):
        return False
    held, cap = basket_ids(ep), ep.numeric_budget()
    if not held or cap is None:
        return True
    prices = [mem_price(ep, i) for i in held]
    return None in prices or sum(p * max(1, basket_units(ep, i)) for i, p in zip(held, prices)) > cap

def news_read_note(ep, lines: list[str]) -> str:
    """Read a note with a model and apply it."""
    news = ep.news
    news.heard.update(lines)
    prompt = '\n\n'.join([NOTE, 'ITEMS AND REQUIREMENTS NOW:\n' + req_ledger(ep), 'THE NOTE:\n' + '\n'.join(lines)])
    got = ep.llm.ask(prompt, [LIGHT, SPARE], 2500) or {}
    gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
    known = [g for g in gone if g in ep.mem.rows]
    ep.unavailable.update(known)
    news.sold_flag = bool(gone) or got.get('sold_out') is True
    news.named_gone = bool(known)
    if news.sold_flag and len(lines) == 1:
        news.sold_lines.update(lines)
    if got.get('kind') not in (None, 'change'):
        return ''
    changed = news_new_budget(ep, got.get('budget')) + news_new_needs(ep, got)
    tails = (('unavailable: ' + ', '.join(gone), bool(gone)), ('a price changed', got.get('price_moved') is True))
    changed += [text for text, on in tails if on]
    summary = str(got.get('summary') or '').strip()
    if ep.cheapest() and ep.items:
        news_reopen_items(ep, got, gone, known)
    if ep.cheapest() and news_hunt_again(ep, gone, changed):
        scout_reopen_cheapest(ep)
    joined = '; '.join(changed)
    if changed or summary:
        ep.version += 1
        ep.recent = []
        ep.notes.append(summary or joined)
    return f'{summary} ({joined})'.strip() if changed else summary

def news_new_budget(ep, budget: Any) -> list[str]:
    if is_number(budget) and 0 < budget != ep.budget:
        ep.budget = budget
        return [f'budget now {budget}']
    return []

def drop_targets(live: list[str], need: str) -> list[str]:
    """Which live requirements a dropped one cancels: an identical text, otherwise the first on that property, otherwise the drop itself."""
    if need in live:
        return [m for m in live if m == need]
    head = plain_key(need.split(':')[0])
    return next(([m] for m in live if plain_key(m.split(':')[0]) == head), [need])

def news_new_needs(ep, got: dict[str, Any]) -> list[str]:
    """Apply dropped and added requirements."""
    count = len(ep.items)
    lists: dict[str, list[tuple[Any, str]]] = {'drop': [], 'add': []}
    for key, out in lists.items():
        for e in as_list(got.get(key)):
            need = str(e.get('need') or '').strip() if isinstance(e, dict) else ''
            if need:
                out.append((ep.items[pick_number(e.get('item'), count) - 1], need))
    for item, need in lists['drop']:
        item.dropped.extend(drop_targets(item.live(), need))
    for item, need in lists['add']:
        if need in item.dropped:
            item.dropped.remove(need)
        if need not in item.musts and need not in item.added:
            item.added.append(need)
    return [f'{verb} {need}' for verb, key in (('dropped', 'drop'), ('added', 'add')) for _, need in lists[key]]

def news_store_note(ep, seen: Any) -> str:
    ep.news.sold_flag = False
    lines = news_free_lines(ep, seen)
    meaning = news_read_note(ep, lines) if lines else ''
    return f'[store note] {meaning}. Bring your picks in line with this change before checking out.' if meaning else ''

def news_after_refusal(ep, job: Call, reply: dict[str, Any]) -> str:
    """Re-read cart lines after a refusal."""
    news = ep.news
    buying = ep.mode_ok('buy', job.name, job.args)
    refused = buying and not reply.get('finished')
    if not news.sold_flag:
        due = refused and any(line in talk_words_of(reply.get('observation')) for line in news.sold_lines)
    else:
        due = refused or not (news.named_gone or buying)
    found = ''
    if due and basket_ids(ep) and not ep.finished:
        found = news_locate_sold(ep)
    news.sold_flag = False
    return "[Desk re-read the cart's lines after the refusal]\n" + found if found else ''

def news_flush_queue(ep) -> str:
    queued = ep.queued
    if not queued:
        return ''
    ep.queued = []
    return "[notes seen on Desk's own calls] " + ' | '.join(queued)

def news_swap_hint(ep) -> str:
    """Mark sold-out lines and suggest swaps."""
    lost = [i for i in basket_ids(ep) if ep.is_gone(i) and i not in ep.gone]
    if not lost:
        return ''
    ep.gone.update(dict.fromkeys(lost, ep.step))
    ep.leads = [i for i in ep.leads if i not in lost]
    ep.version += 1
    picks = news_replacements(ep, lost)
    text = (f"[Desk] Now sold out: {', '.join(lost)} (the store marks it as not buyable). It will not come back, so do not "
            'open it again.')
    if not picks:
        return text + (' Remove that line from the cart and add the best other fitting listing in its place; search for one if '
                       'nothing read so far fits.')
    first, rest = picks[0], picks[1:]
    listed = ', '.join(picks)
    if ep.steps_left() <= 6:
        return text + (f' Few steps remain: in this turn add {first} to the cart at the requested quantity and remove the sold-out '
                       f"line (two calls), then check out. Other read listings that look like fits: {', '.join(rest) or 'none'}.")
    if ep.cheapest():
        return text + (f' Read listings that look like fits for that item, cheapest first: {listed}. As the pick is the cheapest, '
                       'unless every fitting listing has already been seen, run one search below the price of '
                       f"{first} (using the search's maximum price when it has one) before swapping the cheapest fitting "
                       'listing in for the sold-out line.')
    return text + (f' Read listings that look like they fit that item instead: {listed}. In this turn remove the sold-out line '
                   f'from the cart and add {first} at the requested quantity, then check out.')

def news_locate_sold(ep) -> str:
    ids = [i for i in basket_ids(ep) if i not in ep.gone]
    if ids and not any(ep.is_gone(i) for i in ids):
        if len(ids) != 1:
            return news_refresh_offers(ep)
        ep.unavailable.add(ids[0])
    return ''

def news_replacements(ep, lost: list[str]) -> list[str]:
    """Replacement ids for sold-out lines."""
    models = ep.llm.models_for()
    if not models:
        return []
    mem = ep.mem
    candidates = (i for i in reversed(mem.order) if i not in lost and i not in ep.gone and not ep.is_gone(i))
    pool = [ident for _, ident in zip(range(25), candidates)]
    if not pool:
        return []
    sold = '\n'.join(f'{i}: {mem_record_text(ep, i, True)}' for i in lost)
    records = '\n'.join(dumps(clip(as_stated(mem.pages.get(i) or mem.rows.get(i)), 200))[:1200] for i in pool)
    parts = [SWAP, 'REQUEST:\n' + ep.request, 'REQUIREMENTS IN FORCE:\n' + req_ledger(ep), 'SOLD-OUT CART LINE:\n' + sold,
             'CART NOW:\n' + basket_describe(ep), 'OTHER LISTING RECORDS SEEN (newest first):\n' + records]
    got = ep.llm.ask('\n\n'.join(parts), models, 3500) or {}
    taken = {*lost, *ep.gone, *basket_ids(ep)}
    first: dict[str, Any] = {}
    for entry in as_list(got.get('fits')):
        named = isinstance(entry, dict)
        ident = str(entry.get('id')) if named else str(entry)
        if ident not in mem.rows or ident in taken:
            continue
        first.setdefault(ident, entry.get('price') if named else None)
    fits = [(float(price) if is_number(price) else float('inf'), ident) for ident, price in first.items()]
    ranked = sorted(fits, key=lambda f: f[0]) if ep.cheapest() else fits
    ep.log('swap', lost=lost, fits=fits)
    return [ident for _, ident in ranked[:4]]

def page_reads(ep, tool: str, target: str, ids: list[str]) -> Any:
    """Yield (id, observation) for each id opened with a single-listing tool, stopping when a call fails or the episode is over."""
    schema = ep.schemas.get(tool)
    for ident in ids:
        args = shaped({target: ident}, schema)
        if args is None:
            return
        reply = ep.store.call(tool, args)
        if ep.finished or reply is None:
            return
        yield ident, reply.get('observation')

def news_refresh_offers(ep) -> str:
    """Re-read stale cart lines; changed ones."""
    tool = ep.offer_tool()
    target = ep.id_param(tool)
    if not (tool and target):
        return ''
    rows = ep.mem.rows
    # a line read since its own carting and the last refusal is current: b09 re-read every line at each close
    # attempt (Arena r12: 3 extra steps per attempt in clock-bound episodes, a last line then bought unread at the clock)
    carted = lambda i: min(ep.step - 2, ep.added_at.get(i, ep.basket.changed_at))
    stale = [i for i in basket_ids(ep) if i not in ep.gone and ep.offer_seen.get(i, -99) < max(carted(i), ep.refused_at)]
    if ep.steps_left() < len(stale) + 3 or not stale:
        return ''
    lines = []
    for ident, observation in page_reads(ep, tool, target, stale):
        # the store call itself leaves the rows alone, so this is still the row as it was before the re-read
        before = dict(rows.get(ident) or {})
        ep.offer_seen[ident] = ep.step
        ep.ingest(tool, observation)
        after = rows.get(ident) or {}
        moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
        if not moved:
            continue
        lines.append(f"{ident} changed ({', '.join(moved[:4])}): " + dumps(clip(observation, 300))[:900])
    ep.log('reread', ids=stale, changed=len(lines))
    if lines:
        ep.version += 1
    return '\n'.join(lines)

class Checker:
    """Judgements on listings and carts."""

    def __init__(self) -> None:
        self.audited = self.hunted = False
        # cart keys reviewed and their verdicts; (listing, requirement) pairs whose word form was traced
        self.checked, self.verdicts, self.form_seen = set(), {}, set()
        # the closest-match ranking per item (None before the first) and the pages it has seen
        self.closest: dict[int, list[tuple[int, float, str, list[str]]]] | None = None
        self.closest_ids: set[str] = set()
        self.refs: list[str] = []
        self.refs_read: set[str] = set()

def judge_evidence(ep, observation: Any, head: bool = True) -> str:
    """Per read listing, the keys that match a requirement."""
    ids: list[str] = []
    stack = [(observation, 0)]
    while stack:
        node, depth = stack.pop()
        if depth > 6:
            continue
        if isinstance(node, dict):
            ident = mem_identify(ep, node, False, depth)
            if ident and fact_block(node) and ident not in ids:
                ids.append(ident)
                continue
            node = list(node.values())
        if isinstance(node, list):
            stack.extend((child, depth + 1) for child in reversed(node))
    needs = ep.live_needs()
    if not ids or not needs:
        return ''
    many = len(ep.items) > 1
    tagged = [(f'(item {number}) ' if many else '', need) for number, need in needs[:8]]
    lines = [line for line in (judge_evidence_line(ep, ident, tagged) for ident in ids[:12]) if line]
    if not lines:
        return ''
    lead = ('[evidence: keys whose name carries a requirement\'s property. A key about some other property (another part or '
            'aspect) settles nothing; a key named another way may still hold that same property] ') if head else ''
    return lead + ' | '.join(lines)

def judge_evidence_line(ep, ident: str, tagged: list[tuple[str, str]]) -> str:
    """One read listing's keys per tagged requirement, plus values written in another form; '' when there are none."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    parts = []
    for tag, need in tagged:
        prop = need.split(':', 1)[0].strip()
        hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if names_same(stems(prop), stems(k))]
        if hits:
            off = judge_apart_needs(ep, ident, [need])
            parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3])
                         + (' [not the required value: its words stand apart or in another order there]' if off else ''))
        parts.extend(tag + said for said in judge_form_said(ep, ident, need))
    if not parts:
        return ''
    return f'{ident}: ' + '; '.join(parts)

def judge_form_said(ep, ident: str, need: str) -> list[str]:
    """Attributes that give the required value written in another form."""
    prop, _, raw = need.partition(':')
    value = raw.strip()
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    if not value or not block or judge_apart_needs(ep, ident, [need]):
        return []
    want = plain_words(value)
    plain_hit = lambda v: plain_words(str(v)) == want or any(plain_words(part) == want for part in re.split('[,/;|]', str(v)))
    if any(map(plain_hit, block.values())):
        return []
    hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if same_in_other_form(value, v)]
    kind, note = 'form_match', ' [this attribute gives the required value: the same words, only joined, hyphenated or inflected differently]'
    if not hits:
        hits = judge_form_within(prop, value, block)
        kind, note = 'form_within', ' [has the required words whole inside a longer value: this attribute gives that value]'
        if not hits:
            return []
    judge_form_logged(ep, kind, ident, need, hits)
    return [f'{prop.strip()} "{value}" -> ' + ', '.join(hits[:2]) + note]

def judge_form_logged(ep, kind: str, ident: str, need: str, hits: list[str]) -> None:
    """Trace a form finding once per listing and requirement."""
    pair = (ident, need)
    if pair in ep.judge.form_seen:
        return
    ep.judge.form_seen.add(pair)
    ep.log(kind, id=ident, need=need, hits=hits[:2])

def judge_form_within(prop: str, value: str, block: dict[str, Any]) -> list[str]:
    """Values of the property's own key that hold the required words whole and in order inside a longer value."""
    # real r16: 6/6 paid pages for "sd" said "micro sd"; hyphenated compounds ("x-free", "non-x") never count
    whole = lambda text: re.findall('[a-z0-9]+(?:-[a-z0-9]+)*', str(text).lower())
    wanted = whole(value)
    if not wanted or re.search('\\d', value) or has_bound(value):
        return []
    width = len(wanted)
    inside = lambda words: len(words) > width and wanted in (words[at:at + width] for at in range(len(words) - width + 1))
    return [f'{k}={str(v)[:50]}' for k, v in block.items() if names_same(stems(prop), stems(k)) and inside(whole(v))]

def judge_form_lines(ep, ids: list[str]) -> list[str]:
    many = len(ep.items) > 1
    out = []
    for ident in ids:
        found = [(f'(item {n}) ' if many else '') + line for n, need in ep.live_needs()[:8] for line in judge_form_said(ep, ident, need)]
        if found:
            out.append(f'{ident}: ' + '; '.join(found))
    return out

def judge_apart_needs(ep, ident: str, needs: list[str] | None = None) -> list[str]:
    """The requirements whose value the page holds only as separated or reordered words."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    if needs is None:
        needs = [n for _, n in ep.live_needs()]
    found = []
    for need in needs:
        prop, _, value = need.partition(':')
        shapes = set()
        for k, v in block.items():
            if names_same(stems(prop), stems(k)):
                shapes.add(spread(value.strip(), v))
        if 'apart' in shapes and 'phrase' not in shapes:
            found.append(need)
    return found

def judge_against(ep, ident: str, needs: list[str]) -> bool:
    """Whether the page contradicts a plain requirement: its key for the property shares no word with the value."""
    block = fact_block(ep.mem.pages.get(ident) or {}) or {}
    for prop, _, value in (need.partition(':') for need in needs):
        if re.search('\\d|\\b(?:not|no|other than|unlike|same|than|like|only)\\b', value, re.I):
            continue
        words = set(plain_words(value)) - {'or', 'either', 'and'}
        if not words:
            continue
        held = [v for k, v in block.items() if names_same(stems(prop), stems(k))]
        if held and not any(words & set(plain_words(v)) for v in held):
            return True
    return False

def judge_fit_text(ep, ident: str) -> str:
    """A record's text for fit codes."""
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {}
    loose, inner = [], []
    for v in record.values():
        if isinstance(v, str):
            loose.append(v)
        elif isinstance(v, dict):
            inner.extend(f'{k} {x}' for k, x in v.items() if isinstance(x, str))
    return ' '.join(loose) + ' ' + '. '.join(inner)

def judge_fit_index(ep) -> dict[str, Any]:
    """Fit marks of every seen listing, with the part notes, the known codes and the sale-name groups."""
    marks = {ident: fit_marks(judge_fit_text(ep, ident)) for ident in ep.mem.order}
    notes: dict[str, set[str]] = {}
    names: dict[str, set[str]] = {}
    known: set[str] = set()
    for mark in marks.values():
        for part, codes in mark['also']:
            notes.setdefault(part, set()).update(codes)
        # one device sold under several names ("Model number: X. Also sold as Y"): each name maps to all of them
        group = set(mark['codes'])
        for code in mark['codes']:
            group |= names.get(code, set())
        names.update((code, group) for code in group)
        known.update(mark['codes'])
    return {'marks': marks, 'notes': notes, 'known': known, 'names': names}

def judge_fit_set(ident: str, index: dict[str, Any]) -> set[str]:
    mark = index['marks'].get(ident) or {}
    own = mark.get('own')
    named = codes_named(mark.get('works') or [], index.get('known') or set())
    return named | index['notes'].get(own, set()) if own else named

def judge_fit_faults(ep) -> list[str]:
    if not ep.bound:
        return []
    index = judge_fit_index(ep)
    cart = basket_ids(ep)
    faults = []
    for part, main in ep.bound:
        device_ids, part_ids = mem_ids_of_item(ep, main), mem_ids_of_item(ep, part)
        devices = [i for i in cart if i in device_ids]
        if len(devices) != 1:
            continue
        codes = judge_fit_codes(index, devices[0])
        if not codes:
            continue
        for line in (i for i in cart if i in part_ids):
            text = judge_fit_line(ep, index, cart, (part, main), (devices[0], codes), line)
            if text:
                faults.append(text)
    return faults

def judge_fit_codes(index: dict[str, Any], ident: str) -> set[str]:
    """Codes a listing carries or takes, by the fit index."""
    mark = index['marks'].get(ident, {})
    return set(mark.get('codes') or []).union(mark.get('takes') or [])

def judge_fit_line(ep, index: dict[str, Any], cart: list[str], pair: tuple[int, int], device: tuple[str, set[str]], line: str) -> str:
    """Why a carted part line fails the carted device, with the seen alternatives; '' when it fits."""
    fits = judge_fit_set(line, index)
    host, codes = device
    if not fits or fits & codes:
        return ''
    part, main = pair
    item = ep.items[part - 1]
    said = f'{host} ({", ".join(sorted(codes))})'
    listed = ', '.join(sorted(fits)[:4])
    rest = (index['marks'].get(line) or {}).get('more')
    if rest and all(m.get('own') != rest for m in index['marks'].values()):
        return judge_fit_redirect(ep, index, cart, part, codes, f'{line} ({item.name}) does not name {said} on its own page, which lists '
                                  f'{listed} and sends the rest to part {rest}')
    good = list(filter(lambda i: judge_fit_set(i, index) & codes and not ep.is_gone(i) and i not in cart, mem_ids_of_item(ep, part)))[:4]
    hosts = list(filter(lambda i: judge_fit_codes(index, i) & fits and not ep.is_gone(i), mem_ids_of_item(ep, main)))[:4]
    if good:
        advice = f"seen parts of that kind that work with it: {', '.join(good)}"
    else:
        advice = 'no part of that kind seen so far works with it, so find one or switch the device: ' + judge_fit_plan(ep, short=True)
    text = f'{line} ({item.name}) works with {listed}, not with {said}; {advice}'
    return text + (f"; seen {ep.items[main - 1].name} listings this part works with: {', '.join(hosts)}" if hosts else '')

def judge_fit_redirect(ep, index: dict[str, Any], cart: list[str], part: int, codes: set[str], head: str) -> str:
    """For a part whose own page leaves the device out: the seen parts naming it on their own page; '' when none."""
    # a part naming the device on its own page is surer (real r13: 13/94 paid vs 3/86 via an unseen page)
    needs = ep.items[part - 1].live()
    sure = lambda i: (i in ep.mem.pages and judge_fit_set(i, index) & codes and mem_can_buy(ep, i) and i not in ep.gone
                      and i not in ep.basket.dropped and not judge_apart_needs(ep, i, needs) and not judge_against(ep, i, needs))
    named = [i for i in mem_ids_of_item(ep, part) if sure(i) and i not in cart][:4]
    if not named:
        return ''
    return (f"{head}; seen parts of that kind naming it on their own page: {', '.join(named)}. If one of them also meets the "
            "item's other requirements, keep the device and replace this part with it")

def judge_fit_plan(ep, short: bool = False) -> str:
    """Devices that part listings name, the most-named first."""
    if not ep.bound:
        return ''
    index = judge_fit_index(ep)
    mains = sorted({m for _, m in ep.bound})
    lines = [text for text in (judge_fit_plan_line(ep, index, main, short) for main in mains) if text]
    joined = ' | '.join(lines)
    if short or not lines:
        return joined
    return ("[fit plan: the part item with the fewest listings left under its own requirements picks the device. A part works with a "
            "device when the part's text names the device's model code or one of its sale names (names joined by = are one device). "
            "Searches miss codes, so list the device category and check every row for codes] " + joined)

def judge_fit_tally(ep, index: dict[str, Any], main: int) -> dict[str, dict[int, list[str]]]:
    """Per device code group, the seen fitting listings of each part item bound to the device item."""
    names = index.get('names') or {}
    tally: dict[str, dict[int, list[str]]] = {}
    for p, m in ep.bound:
        if m != main:
            continue
        needs = ep.items[p - 1].live()
        for i in mem_ids_of_item(ep, p):
            if ep.is_gone(i) or judge_against(ep, i, needs):
                continue
            groups = {' = '.join(sorted(names.get(code) or {code})) for code in judge_fit_set(i, index)}
            for key in groups:
                tally.setdefault(key, {}).setdefault(p, []).append(i)
    return tally

def judge_fit_plan_line(ep, index: dict[str, Any], main: int, short: bool) -> str:
    """The codes one device item's parts name, most-named first, each with the devices seen for it."""
    tally = judge_fit_tally(ep, index, main)
    weight = lambda kv: (-len(kv[1]), -sum(len(v) for v in kv[1].values()))
    tail = (lambda ids: '') if short else (lambda ids: f" ({', '.join(ids[-2:])})")
    chunks = []
    for key, per in sorted(tally.items(), key=weight)[:4]:
        wanted = set(key.split(' = '))
        hosts = [i for i in mem_ids_of_item(ep, main) if wanted & judge_fit_codes(index, i)][:2]
        counts = ', '.join(f'{len(ids)} {ep.items[p - 1].name}{tail(ids)}' for p, ids in sorted(per.items()))
        seen = f"(device seen: {', '.join(hosts)})" if hosts else '(device not seen yet)'
        chunks.append(f'{key}: {counts} {seen}')
    if not chunks:
        return ''
    return f"codes the {ep.items[main - 1].name} candidates' parts name -> " + '; '.join(chunks)

def judge_line_faults(ep) -> list[str]:
    """Cart line problems that code can find."""
    cat_key = mem_kind_key(ep)
    held = [i for i in basket_ids(ep) if (basket_units(ep, i) or 0) > 0]
    if not held or not cat_key:
        return []
    labels = {str(r.get(cat_key) or '') for r in ep.mem.rows.values()}
    kind_of: dict[str, str] = {}
    claims: dict[str, list[int]] = {}
    for i in held:
        kind_of[i] = mem_label(ep, i, cat_key)
        claims[i] = [n for n, item in enumerate(ep.items, 1) if mem_belongs(ep, item, kind_of[i])]
    listed = [n for n, item in enumerate(ep.items, 1) if any(mem_belongs(ep, item, label) for label in labels)]
    out = []
    for n in listed:
        out += judge_item_faults(ep, n, held, claims)
    if len(listed) == len(ep.items):
        out += [f"{i} ({kind_of[i] or 'unknown kind'}) is no kind the shopper asked for: take it out" for i in held if not claims[i]]
    else:
        out += judge_stand_in_faults(ep, held, claims, kind_of, listed)
    out += judge_shelf_faults(ep, held, claims, kind_of, labels)
    out += [f'{i} is sold out' for i in held if ep.is_gone(i)]
    out += judge_budget_faults(ep, held)
    return out

def judge_stand_in_faults(ep, held: list[str], claims: dict[str, list[int]], kind_of: dict[str, str], listed: list[int]) -> list[str]:
    """Lines whose category shares no word with any item, while an item without a line has shown no row of its own
    category yet: the line stands in for that item; the store's own category for it is still to be found."""
    # Arena OB6 runs 16000+: 10 of 4,595 orders held a line whose category shares no word with the request, every one
    # unpaid (ours and the leading agent's alike); 0 of 2,498 paid orders held such a line. Each time the item's own
    # category had not been listed yet, so the gate above (every item listed) never ran
    stray = [i for i in held if kind_of[i] and not claims[i]]
    owned = {n for i in held for n in claims[i]}
    unseen = [n for n in range(1, len(ep.items) + 1) if n not in listed and n not in owned]
    if not (stray and unseen):
        return []
    names = ' or '.join(f'item {n} ({ep.items[n - 1].name})' for n in unseen)
    return [f"{i} sits under {kind_of[i]!r}, a category that matches no requested item; it may stand in for {names}, whose own "
            f"category no search has shown yet: search the item's plain name to find that category, then cart a listing from it"
            for i in stray]

def judge_item_faults(ep, n: int, held: list[str], claims: dict[str, list[int]]) -> list[str]:
    """Item n's cart faults: no line, more than one line claimed by it alone, a line at the wrong count."""
    item = ep.items[n - 1]
    mine = [i for i in held if n in claims[i]]
    if not mine:
        return [f'item {n} ({item.name}) has no cart line']
    only = [i for i in mine if claims[i] == [n]]
    out = []
    if len(only) > 1:
        out.append(f"item {n} ({item.name}) has {len(mine)} lines ({', '.join(mine)}); keep one")
    for i in only:
        units = basket_units(ep, i)
        if units != item.qty:
            out.append(f'{i} is held at {units} unit(s); item {n} ({item.name}) wants {item.qty}')
    return out

def judge_shelf_faults(ep, held: list[str], claims: dict[str, list[int]], kind_of: dict[str, str], labels: set[str]) -> list[str]:
    """Lines of an item carted from some label other than the store's own category for that item."""
    # real r15: every passing cart took each item from the store's own label; 51 neighbour-label orders all failed
    spelled = {plain_key(label): label for label in labels if label}
    out = []
    for i in (i for i in held if len(claims[i]) == 1):
        n = claims[i][0]
        item = ep.items[n - 1]
        # real race 18: two orders of ours came from a neighbouring label while every paying row sat under the label spelled like the item
        shelf = plain_key(item.shelf or '') or (plain_key(item.name) if len(item.name.split()) > 1 else '')
        if shelf and shelf in spelled and plain_key(kind_of[i]) not in (shelf, plain_key(item.name)):
            out.append(f"{i} sits under {kind_of[i]!r} rather than {spelled[shelf]!r}, which is the store's category for item {n} "
                       f'({item.name}): swap in a listing from that category')
    return out

def judge_budget_faults(ep, held: list[str]) -> list[str]:
    prices = [mem_price(ep, i) for i in held]
    cap = ep.budget
    if not isinstance(cap, (int, float)) or any(p is None for p in prices):
        return []
    pairs = list(zip(held, prices))
    if ep.brief.get('budget_scope') == 'each':
        return [f'{i} costs {p:g}, over the per-item limit of {cap:g}' for i, p in pairs if cap < p <= 10 * cap]
    total = sum(p * basket_units(ep, i) for i, p in pairs)
    if not cap + 1e-09 < total <= 10 * cap:
        return []
    return [f'the total of {total:g} is over the budget of {cap:g}']

def judge_check_key(ep, ids: list[str]) -> str:
    return f"{'|'.join(sorted(ids))}#{ep.version}"

def judge_check_cart(ep, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
    """Review a cart once per cart and version; '' when it passes or no review runs."""
    judge = ep.judge
    lead = ids is not None
    if not lead:
        ids = basket_ids(ep)
    key = judge_check_key(ep, ids)
    if not ids or key in judge.checked:
        return ''
    apart = []
    for i in ids if ep.steps_left() > ep.reserve() + 4 else ():
        apart.extend((i, need) for need in judge_apart_needs(ep, i))
    if not (apart or ep.llm.models_for(limit)):
        return ''
    judge.checked.add(key)
    if apart:
        judge.verdicts[key] = False
        ep.leads = list(filter(lambda i: i not in ids, ep.leads))
        ep.log('check_apart', cart=ids, needs=apart[:4])
        said = []
        for ident, need in apart[:3]:
            bits = [part.strip() for part in need.split(':', 1)]
            said.append('{}: its {} has the words of "{}" only split up or in another order, so it is another value'.format(ident, *bits))
        return ('A close review finds this cart still wrong: ' + '; '.join(said) + '. Find a listing whose attribute gives the value exactly '
                'as written (search the value with the product noun), put it in place of that line, then check out.')
    got = ep.llm.ask(judge_check_prompt(ep, ids, lead), ep.llm.models_for(limit), 3000)
    ep.log('check', cart=ids, got=got)
    if got and not lead:
        fresh = []
        for r in as_list(got.get('refs')):
            hit = re.search('[A-Za-z]{1,6}[- ]?\\d{2,7}', str(r))
            if hit and hit.group(0) not in judge.refs:
                fresh.append(hit.group(0))
        judge.refs += fresh[:4]
    if got and got.get('ok') is False:
        judge.verdicts[key] = False
        return judge_check_refusal(ep, ids, got)
    judge.verdicts[key] = bool(got)
    return ''

def judge_check_refusal(ep, ids: list[str], got: dict[str, Any]) -> str:
    """The review's refusal note; the cart's listings stop being leads."""
    ep.leads = list(filter(lambda i: i not in ids, ep.leads))
    problems = list(map(str, as_list(got.get('problems'))))[:5] or ['see the requirements']
    better: list[str] = []
    for b in map(str, as_list(got.get('better'))):
        if len(better) < 4 and b in ep.mem.rows and not ep.is_gone(b):
            better.append(b)
    shown = f'. Listings that look better: {", ".join(better)}' if better else ''
    return (f'A careful review says this cart is not right yet: {"; ".join(problems)}{shown}. Fix the cart, or re-read the records '
            'to show the review is wrong, and then check out.')

def judge_chain_records(ep, ids: list[str]) -> str:
    """The pages the live requirements point at that are not cart lines, each chain led by the page giving its value."""
    # the value of a chain comes from its last page, and a rule there may relate it to the cart's value, so the review
    # sees those pages (Arena b10 r14; real r12-r16: lines offering alternatives on the chain's property passed 5/122)
    live = {need for _, need in ep.live_needs()}
    chains = [(need, held) for need, held in ((n, [i for i in c if i in ep.mem.pages]) for n, c in ep.ref_pages.items()) if need in live and held]
    ends_first = [held[-1] for _, held in chains] + [i for _, held in chains for i in held[-2::-1]]
    linked = [i for i in dict.fromkeys(ends_first) if i not in ids][:4]
    if not linked:
        return ''
    ends = '; '.join(f'{need} -> ' + ' -> '.join(held) + f' (value given by {held[-1]}: {ep.ref_values.get(need) or "see its page"})' for need, held in chains)
    return ('RECORDS OF THE PAGES THOSE REQUIREMENTS POINT AT. A page that only forwards the reader to another one supplies no value; '
            'each chain stops on the page that does: ' + ends + ". The requirement's own wording decides how the cart's value must relate "
            'to that value. When it asks for that value itself, the cart line must state it in full: an attribute offering options ("A or '
            'B") or a list naming it among others does not supply the single value B. A pairing rule found on that page fixes the value '
            "only if the request's own words ask for a counterpart to the page's value (something that pairs with or complements it); when "
            "the request wants the same, an identical or a matching value, or just what that page states, the page's own value is the one "
            'that counts.\n' + '\n'.join(f'{i}: ' + mem_record_text(ep, i, True) for i in linked))

def judge_check_prompt(ep, ids: list[str], lead: bool) -> str:
    if lead:
        wanted = '; '.join(f'item {n}: quantity {it.qty}' for n, it in enumerate(ep.items, 1))
        cart = (f'CART (planned, not added yet: judge these listings as though the cart already held them at the requested quantities: '
                f'{wanted}):\n' + ', '.join(ids))
    else:
        cart = f'CART:\n{basket_describe(ep)}'
    blocks = [CHECK, f'REQUEST:\n{ep.request}', f'REQUIREMENTS IN FORCE:\n{req_ledger(ep)}']
    blocks.append(cart)
    gone = lambda i: 'SOLD OUT (the store says it can no longer be bought). ' if ep.is_gone(i) else ''
    blocks.append("RECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(f'{i}: {gone(i)}' + mem_record_text(ep, i, True) for i in ids))
    forms = judge_form_lines(ep, ids)
    blocks += list(filter(None, [judge_chain_records(ep, ids)]))
    if ep.brief.get('pick') in ('cheapest', 'most', 'least'):
        blocks.append('OTHER LISTINGS SEEN (newest first):\n' + mem_other_records(ep, ids, 15))
        forms += judge_form_lines(ep, [i for i in ep.mem.pages if i not in ids and not ep.is_gone(i)])
    if forms:
        blocks.append('WORD FORMS (attributes Desk read as stating a required value written another way):\n' + '\n'.join(forms[:8]))
    return '\n\n'.join(blocks)

def judge_audit(ep, keep: int | None = None) -> str:
    """Before an ending without a purchase, ask the review for listings that may still fit."""
    judge = ep.judge
    if judge.audited or not ep.llm.models_for():
        return ''
    judge.audited = True
    got = judge_audit_call(ep)
    revised = judge_apply_review(ep, got)
    named = judge_audit_fits(ep, got)
    if not named:
        # an empty review first runs its own short searches, then looks again (real race 18, 49 agents: none of 123
        # unbought endings that followed a carted paying listing paid)
        shown = judge_review_searches(ep, got, keep)
        offered = map(str, as_list((got or {}).get('open')))
        unread = [i for i in offered if i in ep.mem.rows and i not in ep.opened and not ep.is_gone(i)]
        unread += [i for i in judge_requested_unread(ep, shown) if i not in unread][:2]
        opened = bool(unread) and judge_open_rows(ep, unread, keep)
        if not (opened or revised or shown):
            return ''
        got = judge_audit_call(ep)
        judge_apply_review(ep, got)
        named = judge_audit_fits(ep, got)
        if not named:
            return ''
    ep.lead_items = named
    ep.leads = [i for ids in named.values() for i in ids]
    listed = '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items())
    why = got.get('reason')
    return (f'A careful review thinks these listings may fit: {listed}' + (f' ({why})' if why else '')
            + '. Open each of them and buy whichever truly fits; end without a purchase only if none of them does.')

def judge_apply_review(ep, got: dict[str, Any] | None) -> list[str]:
    """The review's reading of the request: per item at most one backstory-only request requirement goes, and a value
    padded with connecting words becomes its bare value for the same property. Needs from answers or notes and pointer
    lookups stay as they are."""
    pointers = {need for _, need in ep.refs} | {str(x) for x in as_list(ep.brief.get('lookups'))}

    def target(entry: Any) -> tuple[Any, str]:
        if not isinstance(entry, dict):
            return None, ''
        number = judge_item_number(ep, entry.get('item'))
        return (ep.items[number - 1] if 1 <= number <= len(ep.items) else None), str(entry.get('need') or '').strip()
    notes: list[str] = []
    quota: set[int] = set()
    for entry in as_list((got or {}).get('not_firm')):
        item, need = target(entry)
        if item is None or need not in item.musts or need in item.added or need in item.dropped:
            continue
        own = judge_own_likeness(need)
        if need in pointers and not own:
            continue
        # a measured requirement is never backstory (forum, race-13 Cairn fork: a demoted thickness figure bought a wrong
        # listing); likenesses to the shopper's own things stay exempt
        if not own and re.search('[0-9]', need.partition(':')[2]):
            continue
        # Arena 17784 (race-18 world): four "matching the style of our ..." likenesses kept a read fitting listing out after
        # the review had named it. Likenesses to the shopper's own things may all go; any other must keeps the one-per-item
        # limit, and an item always keeps one plain request must
        if not own and id(item) in quota:
            continue
        if not [m for m in item.musts if m not in item.dropped and m != need and m not in pointers]:
            continue
        if not own:
            quota.add(id(item))
        item.dropped.append(need)
        notes.append(f'backstory set aside: {need}')
    for entry in as_list((got or {}).get('corrected')):
        item, need = target(entry)
        bare = str(entry.get('value') or '').strip() if isinstance(entry, dict) else ''
        if item is None or need in pointers or need not in item.live() or ':' not in bare or bare == need:
            continue
        if plain_key(bare.partition(':')[0]) == plain_key(need.partition(':')[0]):
            item.dropped.append(need)
            item.added.append(bare)
            notes.append(f'{need} -> {bare}')
    if notes:
        ep.version += 1
        ep.log('review', changed=notes)
    return notes

def judge_own_likeness(need: str) -> bool:
    """A likeness to the shopper's own belongings (our/my ...) naming no shop and no listing code: no listing can carry its
    value, so a review may drop it as it drops backstory."""
    value = need.partition(':')[2]
    if re.search('[A-Z][a-z]{3,}', value) or re.search('[A-Z]{2,}[-]?[0-9]', value):
        return False
    return re.search('\\b(?:our|my|mine|ours|we|us)\\b', value, re.I) is not None

def judge_review_searches(ep, got: dict[str, Any] | None, keep: int | None) -> list[str]:
    """Two at most of the review's short searches not run yet, leaving room for the cart and the order; the ids shown."""
    plan = scout_search_template(ep)
    if plan is None or ep.finished or ep.past_stop():
        return []
    tool, base, field = plan
    spare = ep.steps_left() - (ep.reserve() + len(ep.items) + 2 if keep is None else keep) - 2
    done = {str((args or {}).get(field) or '').strip().lower() for _, args, _ in ep.scout.search_log}
    asked = [str(e.get('query') or '').strip() for e in as_list((got or {}).get('search')) if isinstance(e, dict)]
    shown: list[str] = []
    for words in [q for q in dict.fromkeys(asked) if q and q.lower() not in done][:2]:
        if spare < 2 or ep.finished or ep.past_stop():
            break
        args = shaped(scout_whole_words(ep, tool, {**base, field: words}), ep.schemas.get(tool))
        ok, seen = scout_fetch(ep, tool, args)
        if not ok:
            break
        ep.scout.search_log = list(ep.scout.search_log)[-11:] + [(tool, dict(args), seen)]
        shown += [i for i, _ in mem_listings_in(ep, seen)]
        spare -= 1
    if shown:
        ep.log('review_search', queries=asked[:2], rows=len(shown))
    return shown

def judge_requested_unread(ep, ids: list[str]) -> list[str]:
    """Unread buyable rows of a requested kind among ids: ranked by required values shown, cheaper ahead on a tie."""
    cat_key = mem_kind_key(ep)
    ranked = []
    for ident in dict.fromkeys(ids):
        if ident in ep.mem.pages or not mem_can_buy(ep, ident):
            continue
        owner = mem_owner(ep, mem_label(ep, ident, cat_key)) if cat_key else 1
        if 0 < owner <= len(ep.items):
            ranked.append((-scout_values_shown(ep, ident, ep.items[owner - 1].live()), ep.price_rank(ident), ident))
    return [ident for *_, ident in sorted(ranked)]

def judge_audit_call(ep) -> dict[str, Any] | None:
    # read pages (which the review can pass) and unread rows (which it can send Desk to open) each keep their room
    mem = ep.mem
    live = [i for i in reversed(mem.order) if not ep.is_gone(i)]
    read = [i for i in live if i in mem.pages][:20]
    records = ['OPENED ' + dumps(clip(as_stated(mem.pages[i]), 200))[:1200] for i in read]
    # unread rows of a requested kind within budget first, then by required values shown, then price (real r12-r16: 114
    # paid rows were seen and never opened)
    cat_key, cap = mem_kind_key(ep), ep.numeric_budget()
    count = len(ep.items)

    def order(i: str) -> tuple[bool, bool, int, float]:
        owner = mem_owner(ep, mem_label(ep, i, cat_key)) if cat_key else 0
        mine = 0 < owner <= count
        price = mem_price(ep, i)
        shown = scout_values_shown(ep, i, ep.items[owner - 1].live()) if mine else 0
        return not mine, not (cap is None or (price or 0) <= cap), -shown, price or 0.0
    unread = sorted((i for i in live if i not in mem.pages), key=order)
    records += ['ROW ' + dumps(clip(as_stated(mem.rows.get(i)), 120))[:500] for i in unread[:30]]
    if not records:
        return None
    stood = [f'item {n}: {need}' for n, item in enumerate(ep.items, 1) for need in item.added if need not in item.dropped]
    ran = list(dict.fromkeys(str(v) for _, args, _ in ep.scout.search_log for v in (args or {}).values() if isinstance(v, str)))
    prompt = (f'{AUDIT}\n\nREQUEST:\n{ep.request}\n\nREQUIREMENTS IN FORCE:\n{req_ledger(ep)}\n\n'
              "NEEDS FROM THE SHOPPER'S ANSWERS AND STORE NOTES (these stand):\n" + ('\n'.join(stood) or 'none') + '\n\n'
              'SEARCHES ALREADY RUN:\n' + (' | '.join(ran[-12:]) or 'none') + '\n\n'
              'LISTING RECORDS SEEN (read pages first, then rows; newest first in each):\n' + '\n'.join(records))
    got = ep.llm.ask(prompt, ep.llm.models_for(), 4000)
    ep.log('audit', got=got)
    return got

def judge_audit_fits(ep, got: dict[str, Any] | None) -> dict[str, list[str]]:
    """The audit's fits, per item number."""
    fits = (got or {}).get('fits')
    if not isinstance(fits, dict):
        return {}
    named: dict[str, list[str]] = {}
    for key, ids in fits.items():
        slot = named.setdefault(str(judge_item_number(ep, key) or key), [])
        held = list(slot)
        for i in map(str, as_list(ids)):
            if len(slot) < 4 and i in ep.mem.rows and not ep.is_gone(i) and i not in held:
                slot.append(i)
    return {k: v for k, v in named.items() if v}

def judge_item_number(ep, key: Any) -> int:
    """The item number a key names (0: none)."""
    count = len(ep.items)
    text = str(key)
    number = whole(key)
    if number is None and re.fullmatch('\\s*(?:item\\s*)?#?\\s*\\d+\\s*', text, re.I):
        digits = re.findall('\\d+', text)
        number = int(digits[0]) if len(digits) == 1 else None
    if number is not None:
        return number if 1 <= number <= count else 0
    said = plain_key(text)
    if said:
        for n, item in enumerate(ep.items, 1):
            if said in (plain_key(item.name), plain_key(item.shelf or '')) or mem_belongs(ep, item, text):
                return n
    return 1 if count == 1 else 0

def judge_row_plan(ep, rows: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """The calls that would read these rows' pages ([] when no tool can)."""
    many, joiner = ep.side_param(), ep.roles.get('compare')
    if many and joiner and not ep.side_view_bare():
        return [(joiner, {many: rows[:ep.side_room()]})]
    tool = ep.attr_tool()
    target = ep.id_param(tool)
    if not tool or not target:
        return []
    # one at a time: 4 rows while steps and clock are plentiful (an unread row may be the fit), else 2
    plenty = ep.steps_left() >= 15 and ep.elapsed() <= CLOCK_STOP - 60
    return [(tool, {target: i}) for i in rows[:4 if plenty else 2]]

def judge_open_rows(ep, rows: list[str], keep: int | None = None) -> bool:
    plan = judge_row_plan(ep, rows)
    if keep is None:
        need = 5
    else:
        # a late review opens only what still leaves room for its cart lines and the order
        plan = plan[:max(0, ep.steps_left() - keep)]
        need = keep
    if not plan or ep.steps_left() < len(plan) + need:
        return False
    before = len(ep.mem.pages)
    for name, args in plan:
        call = shaped(args, ep.schemas.get(name))
        if call is None:
            return False
        reply = ep.store.call(name, call)
        if reply is None or ep.finished:
            break
        ep.ingest(name, reply.get('observation'))
    gained = len(ep.mem.pages) - before
    ep.log('audit_open', rows=rows, opened=gained)
    return gained > 0

def judge_fit_picks(ep) -> dict[int, str]:
    """Per item, a lead or a cart line that passes the checks."""
    cat_key = mem_kind_key(ep)
    carted = basket_ids(ep) if cat_key else []
    picks: dict[int, str] = {}
    for number, item in enumerate(ep.items, 1):
        mine = [i for i in carted if mem_belongs(ep, item, mem_label(ep, i, cat_key))]
        for i in ep.lead_items.get(str(number), []) + mine:
            if i in ep.mem.rows and not ep.is_gone(i) and not judge_apart_needs(ep, i):
                if i:
                    picks[number] = i
                break
    return picks

def judge_offer_note(ep, room: int) -> str:
    """The ending remark naming the nearest matches."""
    if ep.brief.get('quiet_decline') is not False:
        return ''
    picks = judge_fit_picks(ep)
    named, missing = [], []
    for n, item in enumerate(ep.items, 1):
        if n in picks:
            named.append(f'{picks[n]} for the {item.name}')
        else:
            missing.append(item.name)
    if not named:
        return judge_closest_note(ep, room)
    more = ' No ' + ' or '.join(missing) + ' listing I read satisfies all of its requirements.' if missing else ''
    size = len(named)
    while size:
        text = 'Ending with no purchase: no listing here satisfies the full order. Nearest matches: ' + '; '.join(named[:size]) + ' (each one satisfies its own item).'
        if len(text + more) <= room:
            return text + more
        if len(text) <= room:
            return text
        size -= 1
    return ''

def closest_target(entry: Any, pages: list[str], count: int) -> tuple[str, int] | None:
    """(id, item number) a CLOSEST reply entry names, when both are valid."""
    if not isinstance(entry, dict):
        return None
    ident, item = str(entry.get('id')), entry.get('item')
    if ident not in pages:
        return None
    number = whole(item) or (1 if item is None and count == 1 else 0)
    if not 1 <= number <= count:
        return None
    return ident, number

def judge_rank_closest(ep, only: list[str] | None = None) -> dict[int, list[tuple[int, float, str, list[str]]]]:
    """Read listings per item, fewest misses first, then cheapest."""
    judge, pages_seen = ep.judge, ep.mem.pages
    cat_key = mem_kind_key(ep)
    if only is None:
        ranked, pool = {}, pages_seen
    else:
        ranked = {n: list(rows) for n, rows in (judge.closest or {}).items()}
        pool = only
    pages = sorted((i for i in pool if i in pages_seen and i in ep.mem.rows), key=ep.price_rank)[:16]
    models = ep.llm.models_for(SPEND_HARD)
    if not pages or not models or not ep.items:
        return judge.closest or {}
    stock = lambda i: '' if mem_can_buy(ep, i) else 'SOLD OUT '
    records = [stock(i) + dumps(clip(as_stated(pages_seen[i]), 200))[:1000] for i in pages]
    prompt = f'{CLOSEST}\n\nREQUEST:\n{ep.request}\n\nREQUIREMENTS IN FORCE:\n{req_ledger(ep)}\n\nLISTING RECORDS:\n' + '\n'.join(records)
    got = ep.llm.ask(prompt, models, 8000) or {}
    judge.closest_ids.update(pages)
    for entry in as_list(got.get('listings')):
        found = judge_closest_row(ep, entry, pages, cat_key)
        if not found:
            continue
        number, row = found
        kept = [r for r in ranked.get(number, []) if r[2] != row[2]]
        ranked[number] = kept + [row]
    for n in ranked:
        ranked[n] = sorted(ranked[n], key=lambda r: r[:2])
    judge.closest = ranked
    ep.log('closest', got={n: [r[:3] for r in rows[:4]] for n, rows in ranked.items()})
    return ranked

def judge_closest_row(ep, entry: Any, pages: list[str], cat_key: str | None) -> tuple[int, tuple[int, float, str, list[str]]] | None:
    """A ranked listing as its item number and (miss count, price, id, misses); None when it is no item's listing."""
    target = closest_target(entry, pages, len(ep.items))
    if not target:
        return None
    ident, number = target
    if cat_key:
        label = mem_label(ep, ident, cat_key)
        if label and mem_owner(ep, label) not in (0, number):
            return None
    price = mem_price(ep, ident)
    misses = list(map(str, as_list(entry.get('misses'))))
    misses += [] if mem_can_buy(ep, ident) else ['in stock']
    over = price is not None and is_number(ep.budget) and price * max(1, ep.items[number - 1].qty) > ep.budget
    misses += ['budget'] if over else []
    return number, (len(misses), float('inf') if price is None else price, ident, misses)

def closest_phrase(name: str, row: tuple[int, float, str, list[str]], unit: str) -> str:
    missed, price, ident, misses = row
    cost = f'{price:g}{unit}, ' if price != float('inf') else ''
    detail = f"it misses only: {'; '.join(misses)[:120]}" if missed else 'the closest I found'
    return f'{ident} for the {name} ({cost}{detail})'

def judge_closest_note(ep, room: int) -> str:
    judge = ep.judge
    # the ranking is a long model call: near the clock it could outlast the episode and lose the ending itself
    if ep.past_stop() or ep.elapsed() > CLOCK_STOP - 40:
        ranked = judge.closest or {}
    elif judge.closest is None:
        ranked = judge_rank_closest(ep)
    else:
        fresh = [i for i in ep.mem.pages if i not in judge.closest_ids]
        ranked = judge_rank_closest(ep, fresh) if fresh else judge.closest
    unit = ep.brief.get('budget_unit')
    unit = f' {unit}' if unit else ''
    named = [closest_phrase(item.name, ranked[n][0], unit) for n, item in enumerate(ep.items, 1) if ranked.get(n)]
    if not named:
        return ''
    # too long for the room: drop every detail in brackets, then the last names one by one
    bare = [n.split(' (', 1)[0] for n in named]
    for names in [named] + [bare[:size] for size in range(len(bare), 0, -1)]:
        text = 'Ending with no purchase: no listing satisfies every requirement. Nearest matches: ' + '; '.join(names) + '.'
        if len(text) <= room:
            return text
    return ''

def judge_capped_try(ep, number: int, kind: str, floor: float) -> bool:
    """Search the item's kind with the price capped at the floor; False when the store did not reply."""
    matching = sorted((c for c in ep.scout.cap_calls if kind and kind in c[3]), key=lambda c: len(c[2]))
    if not matching:
        return True
    tool, key, args, _ = matching[0]
    paging = scout_paging_arg(ep, tool)
    bound = int(floor) if float(floor).is_integer() else floor
    call = shaped({**{k: v for k, v in args.items() if k != paging}, key: bound}, ep.schemas.get(tool))
    if call is None:
        return True
    memo = tool + dumps(call)
    if memo in ep.cache:
        return True
    reply = ep.store.call(tool, call)
    if reply is not None:
        seen = reply.get('observation')
        ep.ingest(tool, seen)
        ep.cache[memo] = (ep.version, ep.step, ep.render(seen))
        ep.log('closest_search', item=number, args=call, rows=len(mem_listings_in(ep, seen)))
    return reply is not None

def judge_closest_hunt(ep) -> None:
    """Before naming a near miss, look for cheaper ones."""
    # rows come in no price order and the offer is judged by price among the closest
    judge = ep.judge
    if judge.hunted or ep.brief.get('quiet_decline') is not False or judge_fit_picks(ep):
        return
    if ep.finished or ep.elapsed() > CLOCK_STOP - 60:
        return
    judge.hunted = True
    if not ep.mem.price_key:
        # learn from the searches so far where rows keep price and stock, and which searches capped the price
        searches = ep.scout.search_log
        for name, _, seen in searches:
            scout_learn_shape(ep, name, seen)
        for name, args, seen in searches:
            scout_learn_cap(ep, name, args, seen)
    ranked = judge_rank_closest(ep)
    cat_key = mem_kind_key(ep)
    read: list[str] = []
    for number, rows in sorted(ranked.items()):
        got = judge_hunt_item(ep, number, rows, cat_key)
        if got is None:
            return
        read += got
    if read and not ep.past_stop():
        judge_rank_closest(ep, read)

def judge_hunt_item(ep, number: int, rows: list[tuple[int, float, str, list[str]]], cat_key: str | None) -> list[str] | None:
    """One item's price-capped search, then reads of cheaper unread rows of the nearest kind; None when the store failed."""
    if not rows or not cat_key or ep.steps_left() < ep.reserve() + 5 or ep.elapsed() > CLOCK_STOP - 30:
        return []
    fewest = rows[0][0]
    priced = [(price, ident) for misses, price, ident, _ in rows if misses == fewest and price != float('inf')]
    if not priced:
        return []
    floor = min(price for price, _ in priced)
    kind = mem_label(ep, priced[0][1], cat_key).strip().lower()
    if not judge_capped_try(ep, number, kind, floor):
        return None
    same_kind = lambda i: mem_label(ep, i, cat_key).strip().lower() == kind
    pool = []
    for i in ep.mem.order:
        if i not in ep.mem.pages and mem_can_buy(ep, i) and same_kind(i) and ep.price_rank(i) < floor:
            pool.append(i)
    pool.sort(key=lambda i: mem_price(ep, i) or 0.0)
    cheaper = pool[:CHEAPER_OPENS]
    scout_read_pages(ep, cheaper, ep.reserve() + 2)
    return [i for i in cheaper if i in ep.mem.pages]

class ChainWalker:
    """Requirements whose value sits on another page."""

    def __init__(self) -> None:
        # counters: trail rounds, holds, link searches, cart adds held back for an open pointer
        self.rounds = self.holds = self.searches = self.cart_holds = 0
        # per link query: rows shown, the next-page call (None at the end) and the further pages Desk read
        self.link_seen, self.link_next, self.link_paged = {}, {}, {}
        # pages already opened when the last trail round was asked; the lost-link marks of earlier holds
        self.read_at, self.hold_marks = set(), set()
        # the last answered trail round and what it saw (pending needs, pages read, rows seen), reused while unchanged
        self.last_got: dict[str, Any] | None = None
        self.last_pending: tuple[str, ...] = ()
        self.last_pages, self.last_rows = set(), set()

def chains_trace(ep, pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, tuple[str, ...]]]]:
    """One TRAIL round: ask how each pending chain runs, then settle what it can."""
    walker = ep.chains
    walker.rounds += 1
    # Arena b10 noask: 63/461 trail rounds saw nothing new of a chain's shops; such a round reuses the last answer
    if walker.last_got is not None and not chains_changed(ep, pending):
        ep.log('trail_reuse', pending=pending)
        return chains_read_chains(ep, walker.last_got, pending)
    heavy_ok = ep.llm.cost + HEAVY_ROOM <= SPEND_HARD
    walker.read_at = set(ep.opened)
    seen_pages = ep.opened | ep.page_seen
    seen_rows = set(ep.mem.rows)
    got = ep.llm.ask(chains_prompt(ep, pending), [LIGHT, HEAVY] if heavy_ok else [LIGHT], 4500) or {}
    ep.log('trail', pending=pending, got=clip(got, 150))
    walker.last_got = got if any(isinstance(c, dict) for c in as_list(got.get('chains'))) else None
    walker.last_pending = tuple(pending)
    walker.last_pages = seen_pages
    walker.last_rows = seen_rows
    return chains_read_chains(ep, got, pending)

def chains_changed(ep, pending: list[str]) -> bool:
    """Whether a trail round would see new input: other pending needs, or a newly read page or seen row of a chain's listings."""
    walker = ep.chains
    if not set(pending) <= set(walker.last_pending):
        return True
    # the words naming a chain's listings and shops: the pending needs and the last answer's link descriptions
    links: list[dict[str, Any]] = []
    for chain in as_list((walker.last_got or {}).get('chains')):
        if isinstance(chain, dict):
            links.extend(link for link in as_list(chain.get('links')) if isinstance(link, dict))
    ids = {code_listing(str(link.get('id') or '').strip(), ep.mem.rows) for link in links}
    # plus what a read link's own page says about the next one
    sends = [chains_onward(ep, i, need) for i in ids if i in ep.mem.pages for need in pending]
    described = [f"{link.get('about') or ''} {link.get('find') or ''}" for link in links]
    text = '\n'.join([*pending, *described, *sends])
    fresh = ((ep.opened | ep.page_seen) - walker.last_pages) | (set(ep.mem.rows) - walker.last_rows)
    return any(i in ids or chains_names_shop(ep, text, i) for i in fresh)

def walk_report(pages: list[str], values: dict[str, str], lost: list[str], hold: bool) -> str:
    """Desk's note after a walk: the pages opened, the values read and, under a hold, the links still to open."""
    out = []
    if pages:
        out.append('[Desk opened the pages a requirement points at]\n' + '\n'.join(pages) + '\n')
    if values:
        out.append('[values read from the pages requirements point at] ' + '; '.join(f'{k} -> {v}' for k, v in values.items()) + '\n')
    if hold:
        out.append('No order counts until every page a requirement names, and each page that one leads on to, has been opened in full '
                   '(a search row does not do). Not yet found and opened: ' + ' | '.join(lost[:3]) + ". Search the shop's name along "
                   "with the values its description gives (other listings of that kind from the same shop are not it), open the shop "
                   'row whose page shows those values, use the value stated there, then check out.')
    return ''.join(out)

def chains_walk_hold(ep, lost: list[str]) -> bool:
    """Whether the walk holds the order for links still unopened, counting the hold."""
    left = ep.steps_left()
    hold = bool(lost) and left > 5 and ep.chains.holds < (3 if left > 15 else 2)
    mark = '|'.join(lost)
    if hold:
        # a hold naming exactly the links of an earlier hold says nothing new
        if mark in ep.chains.hold_marks:
            ep.log('chain_hold_repeat', lost=lost[:3])
            hold = False
        else:
            ep.chains.hold_marks.add(mark)
    ep.chains.holds += hold
    return hold

def chains_walk(ep, room: int = 6) -> tuple[str, bool]:
    """Follow pointer requirements to their pages; returns (note, hold the order)."""
    pending = ep.open_pointers()
    tool = ep.offer_view() or ep.offer_tool()
    target = ep.id_param(tool)
    pages, lost = [], []
    # chains of 3-4 hops (Arena r13: Voshibel -> Motormo -> Morisano -> Rarilen) need a round and a search per hop
    left = ep.steps_left()
    limit = min(room + 2, 8) if left > 25 else min(room, 4 if left <= 15 else 6)
    while True:
        if not pending or ep.llm.cost >= SPEND_HARD or ep.chains.rounds >= limit or ep.past_stop():
            break
        reach, lost, gaps = chains_trace(ep, pending)
        pending = list(filter(lambda need: need not in ep.ref_values, pending))
        stuck = gaps and not reach
        if stuck and tool and target and chains_search_room(ep):
            reach = chains_find_link(ep, *gaps[0])
            # chains_link_opened's trace event: the link is a page already opened
            known = ep.trace[-1]
            if known.get('kind') == 'link_known':
                lost.pop(0)
                pages.append(f"{known['id']}: use the value on an opened page of the shop this chain names, the one showing the values its description gives")
        # the round's candidates open together: in one side-by-side step where that view shows pages, else one read each
        side = chains_side_view(ep) is not None
        room_now = chains_batch_room(ep) if side else max(0, min(3, 6 - len(pages)))
        batch = reach[:room_now]
        if ep.finished or not batch or not (tool and target):
            break
        if ep.steps_left() < (1 if side else len(batch)) + 4:
            break
        pages += chains_open_links(ep, tool, target, batch)
    hold = chains_walk_hold(ep, lost)
    return walk_report(pages, ep.ref_values, lost, hold), hold

def chains_prompt(ep, pending: list[str]) -> str:
    rows, pages = ep.mem.rows, ep.mem.pages
    record = lambda i: pages.get(i) or rows.get(i)
    full: list[str] = []
    bare: list[str] = []
    for i in reversed(ep.mem.order):
        (full if i in ep.opened and record(i) else bare).append(i)
    # an opened page whose words send the reader on carries that sentence first (Arena b10 r14/r13; records cut at 1400)
    records = []
    for i in full[:45]:
        mark = ''
        for need in pending:
            sentence = chains_onward(ep, i, need)
            if sentence:
                mark = f'[sends on for {need.split(":", 1)[0].strip()}: "{sentence}"] '
                break
        records.append('OPENED ' + mark + dumps(clip(record(i), 300))[:1400])
    records += ['ROW ' + dumps(clip(rows.get(i), 80))[:300] for i in bare][:60 - len(records)]
    lines = [TRAIL, 'REQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)]
    said = [f'{need}\n  request: "{words}"' for need in pending for words in [chains_request_words(ep, need)] if words]
    if said:
        lines.append("REQUEST'S WORDS FOR THEM (the shopper's own sentence for each):\n" + '\n'.join(said))
    lines.append('RECORDS (newest first):\n' + '\n'.join(records))
    return '\n\n'.join(lines)

def chains_request_words(ep, need: str) -> str:
    """The shopper's own sentence behind a pointer requirement (the one naming the shops it names), else ''."""
    # Arena b10 (2302, ece5, 0e86): whether a pairing rule applies is read from the shopper's own words
    proper = lambda text: {w.lower() for w in re.findall("(?<![A-Za-z])[A-Z][a-z]{3,}", text)}
    bag = lambda text: roots(re.sub('[^a-z0-9]+', ' ', text.lower()))
    names = proper(need)
    low_need = need.lower()
    for row in ep.mem.rows.values():
        names |= {n.lower() for n in shop_values(row) if len(n) > 3 and n.lower() in low_need}
    prop = bag(need.split(':', 1)[0])
    best = (0, '')
    for raw in re.split('(?<=[.!?;])\\s+|\\n+', ep.request):
        sentence = raw.strip()
        shared = names & proper(sentence) if sentence else set()
        if shared:
            best = max(best, (len(shared) * 3 + len(prop & bag(sentence)), sentence))
    return ' '.join(best[1].split())[:500]

def code_pattern(code: str) -> Any:
    """A regex for a part or model code, matching it with or without a dash or space ("AB1234", "AB-1234"); None for other text."""
    hit = re.fullmatch('([A-Za-z]{1,6})[- ]?(\\d{2,7})', code.strip())
    if hit is None:
        return None
    letters, digits = hit.groups()
    return re.compile(f'(?<![A-Za-z0-9]){letters}[- ]?{digits}(?![0-9])', re.I)

def code_listing(code: str, rows: dict[str, Any]) -> str:
    """The listing id a chain link names by its own part or model code; the code itself when no seen row carries it."""
    # Arena r13 noask b07: TRAIL wrote links as the codes the pages name each other by ("LU5557") where the records show
    # them in titles ("... [LU5557]"); such a link never counted as found or opened, and 15/77 orders missed the process
    if code in rows:
        return code
    rx = code_pattern(code)
    if rx is None:
        # Arena r16 b10 (f788): "AB2-3456" counts as one whole token, parts joined by a dash, a space or nothing
        parts = re.findall('[A-Za-z0-9]+', code)
        mixed = re.search('\\d', code) and re.search('[A-Za-z]', code)
        if not (4 <= len(code) <= 16 and mixed and len(parts) <= 3):
            return code
        if any(len(i) == len(code) and i[:2] == code[:2] for i in rows):
            return code
        rx = re.compile('(?<![A-Za-z0-9])' + '[- ]?'.join(map(re.escape, parts)) + '(?![A-Za-z0-9])', re.I)
    titled = []
    for i, row in rows.items():
        if any(isinstance(v, str) and re.search(TITLE_FIELD, k, re.I) and rx.search(str(v)) for k, v in row.items()):
            titled.append(i)
    hits = titled or [i for i, row in rows.items() if rx.search(dumps(row))]
    return hits[0] if len(hits) == 1 else code

def chain_need(named: str, pending: list[str]) -> str:
    """The pending requirement a traced chain names; '' when it is unclear."""
    if named in pending:
        return named
    if len(pending) == 1:
        return pending[0]
    # Arena r16 b10 (2302): TRAIL echoed two needs without their 'property:' head, and both chains were dropped
    tail = lambda p: p.split(':', 1)[-1].strip()
    near = [p for p in pending if named in p or tail(p) in named] if named else []
    return near[0] if len(near) == 1 else ''

def chain_outcome(chain: Any, pending: list[str], opened: set[str], rows: dict[str, Any], onward: Any = None) -> tuple[Any, ...]:
    """Verdict of one traced chain: ('value', need, value, ids), ('open', ids), ('find', need, about, words, ids) or ().
    onward(id, need) returns the opened page's sentence that points elsewhere for the need's property, or ''."""
    need = chain_need(str(chain.get('need') or '').strip(), pending) if isinstance(chain, dict) else ''
    if not need:
        return ()
    links = list(filter(lambda step: isinstance(step, dict), as_list(chain.get('links'))))
    ids = list(map(lambda step: code_listing(str(step.get('id') or '').strip(), rows), links))
    value = str(chain.get('value') or '').strip()
    unread = [i for i in ids if i not in opened]
    if links and not unread and (chain.get('done') is True and value or not str(links[-1].get('find') or '').strip()):
        sentence = onward(ids[-1], need) if onward else ''
        if sentence:
            # the last page read sends the reader on for this property: its own words describe the next link
            return 'find', need, sentence, chains_onward_words(sentence), tuple(ids)
        return 'value', need, value or f'as stated on {ids[-1]}', ids
    closed = [i for i in unread if i in rows]
    if closed:
        return 'open', closed
    pos = next((n for n, i in enumerate(ids) if i not in rows), -1)
    gap, code = (links[pos], ids[pos] if code_pattern(ids[pos]) else '') if pos >= 0 else ({}, '')
    told = str(gap.get('about') or '')
    words = str(gap.get('find') or '').strip()
    found = tuple(i for i in ids if i and i in rows)
    if not code:
        return 'find', need, told or 'the listing its value comes from', words, found
    # a link named only by its code: its description (shop, kind, values) is the search, and the code picks the row
    return 'find', need, f"{told or 'the listing its value comes from'} [{code}]", words or re.sub('\\([^)]*\\)', ' ', told).strip(), found

def chains_read_chains(ep, got: dict[str, Any], pending: list[str]) -> tuple[list[str], list[str], list[tuple[str, str, tuple[str, ...]]]]:
    """From a trail answer: the values read, the ids to open and the links still missing."""
    reach, lost, gaps = [], [], []
    # a page's notes show only in a separate page view, so a link read via attributes alone opens again (Arena b09)
    read = ep.page_seen if ep.offer_view() else ep.opened
    for chain in as_list(got.get('chains')):
        outcome = chain_outcome(chain, pending, read, ep.mem.rows, lambda ident, need: chains_onward(ep, ident, need))
        if not outcome:
            continue
        tag, *rest = outcome
        if tag == 'open':
            reach.extend(i for i in dict.fromkeys(rest[0]) if i not in reach)
        elif tag == 'value':
            need, value, ids = rest
            ep.ref_values[need], ep.ref_pages[need] = value, ids
            ep.version += 1
        else:
            need, about, words, ids = rest
            if ids and about == chains_onward(ep, ids[-1], need):
                ep.log('trail_onward', need=need, page=ids[-1], words=words)
            search = f' (search: {words})' if words else ''
            lost.append(f'{need} -> find {about}{search}')
            gaps.append((about, words, ids))
    return reach, lost, gaps

def chains_side_open(ep, side: tuple[str, str], ids: list[str]) -> tuple[list[str], bool]:
    """One side-by-side step over the first ids: (texts of the listings it showed, whether the walk may go on)."""
    name, many = side
    batch = ids[:ep.side_room()]
    width = len(batch)
    args = shaped({many: batch}, ep.schemas.get(name))
    if args is None:
        return [], False
    reply = ep.store.call(name, args)
    if reply is None or ep.finished:
        return [], False
    seen = reply.get('observation')
    ep.ingest(name, seen)
    listed = dict(mem_listings_in(ep, seen))
    texts = [f'{i}: ' + ep.render(listed[i])[:1500] for i in batch if i in listed]
    if texts:
        ep.side_ok = max(ep.side_ok, width)
    else:
        ep.side_bad = min(ep.side_bad or 999, width)
    ep.log('chain_batch', ids=batch, shown=len(texts))
    return texts, True

def chains_open_links(ep, tool: str, target: str, ids: list[str]) -> list[str]:
    """Open listings, several at once in a side-by-side step when that view can read chain pages, otherwise through the page tool."""
    side = chains_side_view(ep) if len(ids) > 1 else None
    shown, go_on = chains_side_open(ep, side, ids) if side else ([], True)
    if shown or not go_on:
        return shown
    if side:
        ids = ids[:2] if ep.steps_left() > 8 else []
    for ident, observation in page_reads(ep, tool, target, ids):
        if tool == ep.offer_view():
            ep.page_seen.add(ident)
        ep.ingest(tool, observation)
        shown.append(f'{ident}: ' + ep.render(observation)[:1500])
    return shown

def shop_values(row: dict[str, Any]) -> list[str]:
    """Stripped short text values a row files under a shop-like key."""
    return [v.strip() for k, v in row.items() if isinstance(v, str) and re.search(SHOP_FIELD, str(k), re.I) and len(v.strip()) <= 40]

def chains_names_shop(ep, about: str, ident: str) -> bool:
    low = about.lower()
    return any(len(n) > 2 and word_in(n.lower(), low) for n in shop_values(ep.mem.rows.get(ident) or {}))

def chains_shop_rows(ep, about: str, ids: list[str]) -> list[str]:
    return [i for i in ids if i not in ep.opened and chains_names_shop(ep, about, i)]

def chains_opened_shop(ep, about: str) -> list[str]:
    return [i for i in ep.opened if chains_names_shop(ep, about, i)]

def leaf_texts(node: Any) -> list[str]:
    """Every scalar of a nested record, as text, in reading order."""
    if not isinstance(node, (dict, list)):
        return [str(node)]
    out: list[str] = []
    for value in (node.values() if isinstance(node, dict) else node):
        out += leaf_texts(value)
    return out

def chains_search_room(ep) -> bool:
    """Whether a walk may still send a link search: steps to spare and searches left under the cap."""
    left = ep.steps_left()
    return left > 12 and ep.chains.searches < (5 if left > 25 else 3)

def chains_link_search(ep, plan: tuple[str, dict[str, Any], str], words: str, memo: str, about: str, shop: set[str],
                       values: set[str] = frozenset()) -> list[str]:
    """Search the store for a chain's missing link; the shop rows found."""
    tool, base, field = plan
    short, alone = chains_short_query(ep, words, shop)
    # real r14-r17: shop queries of 6+ words came back empty .16-.96 of the time, 3-5 words showed the listing .18-.85;
    # the shop's name and kind words search, the described values rank the rows
    text = words
    if short and len(words.split()) >= 6:
        text = short
    sent = chains_link_call(ep, tool, base, field, text, shop)
    if sent is None:
        return []
    args, observation = sent
    ids = [i for i, _ in mem_listings_in(ep, observation)]
    # no row of the shop at all: once more with fewer words (1-2-word shop queries showed the listing .77 / .33 / .94)
    if text != short:
        fewer = short
    else:
        fewer = alone if alone != short else ''
    if fewer and not any(chains_names_shop(ep, about, i) for i in ids) and chains_search_room(ep):
        again = chains_link_call(ep, tool, base, field, fewer, shop)
        if again is not None:
            args, observation = again
            ids = list(dict.fromkeys([*ids, *(i for i, _ in mem_listings_in(ep, observation))]))
            ep.log('link_fewer', words=fewer[:80])
    links = ep.chains
    links.link_seen[memo] = ids
    links.link_next[memo] = chains_next_page(ep, tool, args, observation)
    # a short result set is read whole before a row is chosen, as in scout_pointed_pages (real r16/r17: the pointed
    # shop's listings first showed on page 2 or later in 163 of 944 and 163 of 564 showings)
    for _ in range(scout_pointed_pages(chains_page_plan(ep, tool, observation), args)):
        if not links.link_next.get(memo) or ep.steps_left() <= 12 or chains_strong_row(ep, about, links.link_seen[memo], values):
            break
        chains_link_more(ep, memo, about)
    found = chains_shop_rows(ep, about, links.link_seen[memo])
    ep.log('link_search', words=words[:80], rows=found[:4])
    return found

def chains_link_call(ep, tool: str, base: dict[str, Any], field: str, text: str, shop: set[str]) -> tuple[dict[str, Any], Any] | None:
    """One link search: (arguments sent, result) or None."""
    args = shaped(scout_whole_words(ep, tool, {**base, field: chains_kind_last(ep, tool, field, text, shop)}), ep.schemas.get(tool))
    if args is not None:
        ep.chains.searches += 1
        reply = ep.store.call(tool, args)
        if reply is not None:
            observation = reply.get('observation')
            ep.ingest(tool, observation)
            return args, observation
    return None

def chains_short_query(ep, words: str, shop: set[str]) -> tuple[str, str]:
    """A link query cut down to the shop's name plus the product-kind words, and the shop's name alone ('' when no name
    turns up). A name is a word of a seen shop field, or else a capitalised word naming no category; kind words are
    those of the store's own category labels."""
    cat_key = mem_kind_key(ep)
    kinds = chains_kind_roots(ep, cat_key) if cat_key else set()
    plain = lambda t: re.sub('[^a-z0-9]+', '', t.lower())
    kind = lambda t: bool(roots(plain(t))) and roots(plain(t)) <= kinds
    tokens = list(filter(lambda t: re.search('[A-Za-z0-9]', t), words.split()))
    if any(plain(t) in shop for t in tokens):
        named = lambda t: plain(t) in shop
    else:
        named = lambda t: not kind(t) if re.fullmatch('[A-Z][a-z]{3,}', t.strip('.,;:()')) else False
    names = [t for t in dict.fromkeys(tokens) if named(t)]
    if not names:
        return '', ''
    return ' '.join(t for t in dict.fromkeys(tokens) if named(t) or kind(t)), ' '.join(names)

def chains_page_plan(ep, tool: str, observation: Any) -> tuple[int, int, str, Any] | None:
    """(matched, shown, paging arg, next value) for a link query's result page, or None."""
    param, key = scout_paging_arg(ep, tool), scout_next_key(ep, tool)
    if not param or not key or not isinstance(observation, dict):
        return None
    shown = len(list(mem_listings_in(ep, observation)))
    total = chains_match_count(ep, tool, observation)
    if total is None:
        # without a learned count key, the largest other number of the result stands for the match count
        total = max((v for k, v in observation.items() if k != key and is_count(v)), default=None)
    return (total, shown, param, observation.get(key)) if is_count(total) and shown else None

def chains_match_count(ep, tool: str, observation: dict[str, Any]) -> int | None:
    """The result's match count under the count key learned for the tool, when it is a whole number."""
    shape = ep.scout.shapes.get(tool) or {}
    total = observation.get(shape['count']) if shape.get('count') else None
    return total if is_count(total) else None

def is_count(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)

def chains_strong_row(ep, about: str, ids: list[str], values: set[str]) -> bool:
    """Whether some unread row of the named shop among ids carries every described value as a whole word."""
    if values:
        for i in chains_shop_rows(ep, about, ids):
            text = dumps(ep.mem.rows.get(i) or {}).lower()
            if all(word_in(v, text) for v in values):
                return True
    return False

def chains_next_page(ep, tool: str, args: dict[str, Any], observation: Any) -> tuple[str, dict[str, Any], str, Any] | None:
    """The call for a search's next result page, from the paging value the store handed back; None at the end."""
    param, key = scout_paging_arg(ep, tool), scout_next_key(ep, tool)
    if not param or not key or not isinstance(observation, dict) or not mem_listings_in(ep, observation):
        return None
    value = observation.get(key)
    if value is None or value == '' or isinstance(value, bool) or value == args.get(param):
        return None
    if is_count(value):
        total = chains_match_count(ep, tool, observation)
        if total is not None:
            ended = value >= total
        else:
            # without a known match count, a start value at or past every other number of the result means no rows are left
            others = [v for k, v in observation.items() if k != key and is_count(v)]
            ended = bool(others) and max(others) <= value
        if ended:
            return None
    return tool, dict(args), param, value

def chains_link_more(ep, memo: str, about: str) -> list[str]:
    """The next result page of a link query (two at most per query); the shop rows it shows."""
    # also used when the trail asks again for a link whose seen shop rows of the kind are all read
    links = ep.chains
    step, paged = links.link_next.get(memo), links.link_paged.get(memo, 0)
    if paged >= 2 or not step:
        return []
    tool, args, param, value = step
    call = shaped({**args, param: value}, ep.schemas.get(tool))
    reply = ep.store.call(tool, call) if call is not None else None
    links.link_paged[memo] = paged + 1
    observation = reply.get('observation') if reply is not None else None
    if reply is None:
        links.link_next[memo] = None
        return []
    ep.ingest(tool, observation)
    scout_note_next(ep, tool, call, observation)
    ids = [i for i, _ in mem_listings_in(ep, observation)]
    links.link_seen[memo] = list(dict.fromkeys(links.link_seen.get(memo, []) + ids))
    links.link_next[memo] = chains_next_page(ep, tool, call, observation)
    found = chains_shop_rows(ep, about, ids)
    ep.log('link_page', words=memo[:80], page=dumps(value), rows=found[:4])
    return found

def word_in(word: str, text: str) -> bool:
    """Whether word stands whole in text: no letter or digit touches it on either side."""
    return re.search('(?<![a-z0-9])' + re.escape(word) + '(?![a-z0-9])', text) is not None

def chains_find_link(ep, about: str, words: str, before: tuple[str, ...] = ()) -> list[str]:
    """Rows that may be a chain's missing link, best first."""
    record = lambda ident: dumps(ep.mem.rows.get(ident) or {})
    named: set[str] = set()
    for ident in chains_shop_rows(ep, about, list(ep.mem.rows)) + chains_opened_shop(ep, about):
        for v in shop_values(ep.mem.rows.get(ident) or {}):
            named.update(re.findall('[a-z0-9]{3,}', v.lower()))
    # English function words and the words any pointer phrase uses for a page (grammar, not product vocabulary)
    common = named | {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}
    described = (words or about).lower()
    values = set(filter(lambda w: len(w) >= 3 or w.isdigit(), re.findall('[a-z0-9]+', described))) - common
    if chains_link_opened(ep, about, before, values):
        return []
    memo = ' '.join(words.lower().split())
    plan = scout_search_template(ep)
    kind_words = set(re.findall('[a-z0-9]{3,}', described)) - common
    known: list[str] = []
    if memo and memo in ep.chains.link_seen:
        ep.chains.searches += 1
        known = chains_shop_rows(ep, about, ep.chains.link_seen[memo])
        # the same link asked for again with every seen row of the shop and kind read: the query's next page
        unread_kind = chains_link_kind(ep, about, words, kind_words)
        if not any(map(unread_kind, chains_shop_rows(ep, about, ep.mem.order))):
            known = chains_link_more(ep, memo, about) or known
    elif plan and words:
        known = chains_link_search(ep, plan, words, memo, about, named, values)
    of_kind = chains_link_kind(ep, about, words, kind_words)
    newest = ep.mem.order[::-1]
    # the shop's seen rows of the kind, search hits first, each group led by rows showing more described values
    shown = lambda ident: sum(word_in(v, record(ident).lower()) for v in values)
    found = [i for i in known if of_kind(i)]
    seen = [i for i in chains_shop_rows(ep, about, newest) if of_kind(i) and i not in found]
    picks = sorted(found, key=shown, reverse=True) + sorted(seen, key=shown, reverse=True)
    patterns = [rx for rx in map(code_pattern, re.findall('\\[([A-Za-z]{1,6}[- ]?\\d{2,7})\\]', about)) if rx]
    carry = [i for i in known + newest if any(rx.search(record(i)) for rx in patterns)]
    return list(dict.fromkeys(carry + picks))[:chains_batch_room(ep)]

def chains_link_opened(ep, about: str, before: tuple[str, ...], values: set[str]) -> bool:
    """Whether an opened page of the shop (not an earlier link) already shows every described value, so the link is known."""
    # an earlier link of the chain is the page naming this one: its words describe the link, it is not it
    candidates = [i for i in chains_opened_shop(ep, about) if i not in before] if values else []
    for ident in candidates:
        record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {}
        text = ' '.join(leaf_texts(record)).lower()
        if all(word_in(v, text) for v in values):
            ep.log('link_known', id=ident)
            return True
    return False

def chains_link_kind(ep, about: str, words: str, kind: set[str]):
    """A test for rows of the link's kind: a kind word in the row, else a category label the description names."""
    # a row without the values still counts when its category label is one the link's description names
    cat_key = mem_kind_key(ep)
    told = roots(re.sub('[^a-z0-9]+', ' ', (about + ' ' + words).lower()))
    row_words = lambda ident: set(re.findall('[a-z0-9]{3,}', dumps(ep.mem.rows.get(ident) or {}).lower()))
    label_roots = lambda ident: roots(re.sub('[^a-z0-9]+', ' ', mem_label(ep, ident, cat_key).lower())) if cat_key else set()
    named_label = lambda label: bool(label) and label <= told
    return lambda ident: not kind or bool(kind & row_words(ident)) or named_label(label_roots(ident))

def chains_side_view(ep) -> tuple[str, str] | None:
    """The compare tool with its id-list argument when it can read chain pages (the store has no separate offer view); else None."""
    # a store with an offer view apart from the attributes shows a page's notes only there (chains_read_chains)
    tool, many = ep.roles.get('compare'), ep.side_param()
    usable = tool and many and not ep.offer_view() and not ep.side_view_bare()
    return (tool, many) if usable else None

def chains_batch_room(ep) -> int:
    """How many listings a chain search passes on: as many as one side-by-side step shows, otherwise two (one read per step)."""
    # real r12-r16 pointer needs: opening none of the named shop's seen rows passed 0/306, one 125/830, two or more
    # 336/3135; only the one-step side view opens more than two
    return max(2, ep.side_room()) if chains_side_view(ep) else 2

def chains_kind_roots(ep, cat_key: str) -> set[str]:
    """Word roots of every category label the seen rows carry."""
    out: set[str] = set()
    for row in ep.mem.rows.values():
        out |= roots(re.sub('[^a-z0-9]+', ' ', str(row.get(cat_key) or '').lower()))
    return out

def chains_kind_last(ep, tool: str, field: str, words: str, shop: set[str]) -> str:
    """For a short text field, the words naming only a kind go last."""
    spec = ((ep.schemas.get(tool) or {}).get('properties') or {}).get(field)
    cap = spec.get('maxLength') if isinstance(spec, dict) else None
    cat_key = mem_kind_key(ep)
    if not isinstance(cap, int) or len(words) <= cap or not cat_key:
        return words
    kinds = chains_kind_roots(ep, cat_key)
    # the words naming only a category (and no shop) move to the end, each group keeping its own order
    front, back = [], []
    for token in words.split():
        low = re.sub('[^a-z0-9]+', ' ', token.lower())
        base = roots(low)
        only_kind = bool(base) and base <= kinds and not shop.intersection(re.findall('[a-z0-9]{3,}', low))
        (back if only_kind else front).append(token)
    return ' '.join(front + back)

def chains_recheck(ep) -> str:
    """A trail round over the pages opened since the previous round, kept outside the checkout's own round and hold budget."""
    walker = ep.chains
    if ep.opened <= walker.read_at or ep.llm.cost >= SPEND_HARD or ep.past_stop():
        return ''
    saved = (walker.rounds, walker.holds)
    walker.rounds = 0
    try:
        text, _ = chains_walk(ep, 1)
    finally:
        walker.rounds, walker.holds = saved
    ep.log('cart_trail', step=ep.step, values=dict(ep.ref_values))
    return text

def chains_early(ep, fresh: set[str]) -> str:
    """Walk a chain when its shop page gets opened."""
    if ep.chains.rounds or ep.finished or ep.past_stop() or ep.steps_left() <= 15:
        return ''
    pending = ep.open_pointers()
    # Arena r16 noask b01: pointed pages sat unopened in search rows while no chain round ran before the close; once a
    # worded search shows the query field, the first round searches the pointed shop itself
    go = pending and ep.step >= 3 and scout_search_template(ep) is not None
    if not go and fresh:
        go = any(not fresh.isdisjoint(chains_opened_shop(ep, need)) for need in pending)
    if not go:
        return ''
    ep.log('early_trail', step=ep.step, pending=pending)
    text, _ = chains_walk(ep, 4)
    return text

def chains_onward(ep, ident: str, need: str) -> str:
    """The sentence of a listing's own text that sends the reader on to a different listing for the need's property, or ''."""
    # Arena b10 r14/r13: chains closed on a start page whose note takes the property from another listing failed; such a
    # sentence names the property, then another shop's name, and negates this page (r12-r16: no false hit in 1,000+)
    record = ep.mem.pages.get(ident) or ep.mem.rows.get(ident) or {}
    prop = roots(need.split(':', 1)[0])
    if not prop or not record:
        return ''
    own = {w.lower() for name in shop_values(record) for w in re.findall('[A-Za-z]+', name)}
    plain = lambda text: roots(re.sub('[^a-z0-9]+', ' ', text.lower()))
    sentences = (part for text in leaf_texts(record) for part in re.split('(?<=[.!?;])\\s+', text))
    for sentence in sentences:
        if len(sentence) > 300 or not prop <= plain(sentence):
            continue
        if not re.search("\\b(?:not|n't|instead|rather)\\b(?:\\W+\\w+){0,2}?\\W+(?:this|here)\\b|\\b(?:another|different)\\b", sentence.lower()):
            continue
        other = (m.start() for m in re.finditer('(?<![A-Za-z])[A-Z][a-z]{3,}', sentence) if m.start() > 0 and m.group(0).lower() not in own)
        first = next(other, None)
        if first is not None and prop <= plain(sentence[:first]):
            return sentence.strip()
    return ''

def chains_onward_words(sentence: str) -> str:
    """Search words for the listing an onward sentence points to: the text from its first name on, minus the words that point back."""
    start = next((m.start() for m in re.finditer('(?<![A-Za-z])[A-Z][a-z]{3,}', sentence) if m.start() > 0), 0)
    head, *_ = re.split("[,;]?\\s+\\b(?:not|instead|rather)\\b", sentence[start:])
    # English function words a pointer phrase uses around the description (grammar, not product vocabulary)
    filler = {'the', 'and', 'with', 'from', 'that', 'comes', 'which', 'has', 'its', 'for', 'this', 'item', 'product', 'listing'}
    kept = []
    for w in re.findall("[A-Za-z0-9][A-Za-z0-9.:/-]*", re.sub("'s\\b", '', head)):
        if len(w) > 2 and w.lower() not in filler:
            kept.append(w)
    return ' '.join(kept[:12])

class CartDesk:
    """Cart record, cart calls and own purchases."""

    def __init__(self) -> None:
        self.lines: dict[str, int] = {}
        self.view: Any = None
        self.viewed_at = self.changed_at = -1
        self.reply = ''
        self.touched = self.completed = False
        self.built: list[str] = []
        self.built_needs, self.dropped = set(), set()

def basket_fresh_view(ep) -> bool:
    return ep.basket.view is not None and ep.basket.viewed_at >= ep.basket.changed_at

def basket_describe(ep) -> str:
    if not basket_fresh_view(ep):
        return "lines per Desk's record of the cart calls (listing: units): " + dumps(ep.basket.lines)
    return dumps(clip(ep.basket.view, 300))[:3000]

def basket_units(ep, ident: str) -> int:
    if not basket_fresh_view(ep):
        return ep.basket.lines.get(ident, 1)
    stack: list[Any] = [ep.basket.view]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        if any(isinstance(v, str) and v.strip() == ident for v in node.values()):
            counts = [whole(v) for k, v in node.items()
                      if not (isinstance(v, str) or whole(v) is None) and re.search('\\b(?:qty|quantity|count|units?)\\b', plain_key(k))]
            if len(counts) == 1:
                return counts[0]
        stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
    return ep.basket.lines.get(ident, 1)

def basket_ids(ep) -> list[str]:
    if not basket_fresh_view(ep):
        return list(ep.basket.lines)
    shown = dumps(ep.basket.view)
    return list(filter(shown.__contains__, ep.mem.rows))

def basket_count_fixes(ep) -> list[tuple[str, int, int]]:
    """(line, held, wanted) for each line at a wrong quantity."""
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return []
    belongs = lambda item, i: mem_belongs(ep, item, mem_label(ep, i, cat_key))
    out = []
    for ident in basket_ids(ep):
        owners = [n for n, item in enumerate(ep.items, 1) if belongs(item, ident)]
        if len(owners) != 1:
            continue
        item = ep.items[owners[0] - 1]
        have = basket_units(ep, ident) or 0
        if have <= 0 or have == item.qty or ep.is_gone(ident):
            continue
        same = sum(1 for i in basket_ids(ep) if (ep.basket.lines.get(i, 1) or 0) > 0 and belongs(item, i))
        if same == 1:
            out.append((ident, have, item.qty))
    return out

def basket_fields(ep, name: str) -> tuple[dict[str, Any], dict[str, Any], str | None, str | None, str | None]:
    """The cart tool's schema and properties, and the names of its id, count and mode arguments."""
    schema = ep.schemas.get(name) or {}
    props = schema.get('properties') or {}
    texts: list[str] = []
    count = mode = None
    for k, v in props.items():
        if not isinstance(v, dict):
            continue
        kinds = type_names(v)
        if 'string' in kinds and 'enum' not in v:
            texts.append(k)
        if count is None and 'integer' in kinds:
            count = k
        if mode is None and isinstance(v.get('enum'), list) and len(v['enum']) >= 2:
            mode = k
    target = next((k for t, k in ep.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None)
    if not target:
        target = texts[0] if len(texts) == 1 else None
    return schema, props, target, count, mode

def basket_count_plan(ep, tool: str, props: dict[str, Any], fields: tuple[str, str, str | None], wrong: list[tuple[str, int, int]]) -> list[dict[str, Any]] | None:
    """Cart calls that bring each wrong line to its wanted count; None when the tool has no way to."""
    target, count, mode = fields
    if basket_sets_units(ep, tool):
        return [{target: ident, count: want} for ident, _, want in wrong]
    if not wrong:
        return []
    if not mode:
        return None
    # a tool without a set-count mode takes the line out and adds it again at the wanted count
    choices = [str(c) for c in props[mode]['enum']]
    drop = next(filter(lambda c: re.search('remov|delet|drop|clear|take', c, re.I), choices), choices[1])
    add = next((c for c in choices if c != drop), choices[0])
    return [call for ident, have, want in wrong for call in ({target: ident, mode: drop, count: have}, {target: ident, mode: add, count: want})]

def basket_fix_counts(ep, wrong: list[tuple[str, int, int]]) -> list[str]:
    """Correct wrong quantities through the cart tool; the calls made with their replies."""
    tool = ep.roles.get('cart') or ''
    schema, props, target, count, mode = basket_fields(ep, tool)
    if not (tool and target and count):
        return []
    plan = basket_count_plan(ep, tool, props, (target, count, mode), wrong)
    if plan is None or len(plan) + 3 > ep.steps_left():
        return []
    done: list[str] = []
    for fitted in map(lambda args: shaped(args, schema), plan):
        reply = ep.store.call(tool, fitted) if fitted is not None else None
        if reply is None or ep.finished:
            break
        seen = reply.get('observation')
        ep.basket.reply = dumps(seen)
        basket_record(ep, tool, fitted)
        ep.ingest(tool, seen)
        done.append(dumps(fitted) + ' -> ' + ep.render(seen)[:300])
    ep.log('count_fix', calls=done)
    return done

def basket_drop_call(ep, tool: str, props: dict[str, Any], fields: tuple[str, str | None, str | None], ident: str) -> dict[str, Any] | None:
    """The cart call taking one line out, or None when the tool cannot."""
    target, count, mode = fields
    have = ep.basket.lines.get(ident, 1)
    if mode:
        return {target: ident, mode: props[mode]['enum'][1], **({count: have} if count else {})}
    if not count:
        return None
    if basket_sets_units(ep, tool):
        return {target: ident, count: 0}
    low = props[count].get('minimum')
    return {target: ident, count: -have} if isinstance(low, int) and low <= -have else None

def basket_calls_for(ep, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
    """The cart calls that leave exactly these lines, or None."""
    tool = ep.roles.get('cart') or ''
    _, props, target, count, mode = basket_fields(ep, tool)
    whole_call = basket_whole_cart_call(props, lines)
    if whole_call is not _NO:
        return whole_call
    if not target:
        return None
    plan = [basket_drop_call(ep, tool, props, (target, count, mode), ident) for ident in drop]
    if None in plan:
        return None
    in_cart = basket_ids(ep) if lines else []
    add_mode = {mode: props[mode]['enum'][0]} if mode else {}
    for ident, want in lines.items():
        have = ep.basket.lines.get(ident, 0) if ident in in_cart else 0
        if have != want and have and not basket_sets_units(ep, tool):
            return None
        if have != want:
            plan.append({target: ident, **add_mode, **({count: want} if count else {})})
    return plan

def basket_whole_cart_call(props: dict[str, Any], lines: dict[str, int]) -> Any:
    """The one whole-cart call of a tool that takes a list of lines; _NO when the tool takes none."""
    for key, spec in props.items():
        if not isinstance(spec, dict) or 'array' not in type_names(spec):
            continue
        shape = spec.get('items')
        inner = shape.get('properties') if isinstance(shape, dict) else None
        if not isinstance(inner, dict):
            continue
        specs = [(k, s) for k, s in inner.items() if isinstance(s, dict)]
        id_key = next((k for k, s in specs if 'string' in type_names(s) and 'enum' not in s), None)
        if not id_key:
            return None
        count_key = next((k for k, s in specs if 'integer' in type_names(s)), None)
        entry = (lambda i, n: {id_key: i, count_key: n}) if count_key else (lambda i, n: {id_key: i})
        return [{key: [entry(i, n) for i, n in lines.items()]}]
    return _NO

def basket_complete_cart(ep) -> str:
    """Set and buy a confirmed lead's cart directly."""
    cart, buy = ep.roles.get('cart'), ep.roles.get('buy')
    basket = ep.basket
    if basket.completed or not ep.confirmed or ep.is_gone(ep.confirmed) or not (cart and buy):
        return ''
    basket.completed = True
    lines, drop = basket_lead_cart(ep)
    if not lines:
        return ''
    calls = basket_calls_for(ep, lines, drop)
    if not calls or len(calls) >= ep.steps_left():
        return ''
    ids = list(lines)
    scout_read_pages(ep, ids, len(calls) + 3)
    if not basket_passes(ep, ids):
        return ''
    placed = basket_place(ep, cart, calls)
    replies = basket_buy_confirmed(ep, buy, ids) if placed else []
    if not replies:
        return ''
    return (f'Not run (Desk acted instead): with {ep.confirmed} confirmed by the independent check, Desk filled the cart ('
            + '; '.join(placed) + ') and checked out. Store reply: ' + ' | then: '.join(replies))

def basket_lead_cart(ep) -> tuple[dict[str, int], list[str]]:
    """For a lead: the lines to set and the lines to take out."""
    key = mem_kind_key(ep)
    kind = lambda ident: mem_label(ep, ident, key).strip().lower() if key else ident
    picks: dict[str, int] = {}
    taken: set[str] = set()
    for number, ids in ep.lead_items.items():
        index = whole(number) or 0
        pick = next((i for i in ids if not ep.is_gone(i)), None)
        if not pick or not 1 <= index <= len(ep.items) or kind(pick) in taken:
            continue
        taken.add(kind(pick))
        picks[pick] = ep.items[index - 1].qty
    if ep.confirmed not in picks:
        return {}, []
    now = basket_ids(ep)
    keep = {i: ep.basket.lines.get(i, 1) for i in now if i not in picks and not ep.is_gone(i) and kind(i) not in taken}
    if len(picks) + len(set(map(kind, keep))) < len(ep.items):
        return {}, []
    merged = dict(keep)
    merged.update(picks)
    return merged, [i for i in now if i not in merged]

def basket_passes(ep, ids: list[str]) -> bool:
    """Independent check of these lines as the cart."""
    judge_check_cart(ep, ids, SPEND_HARD)
    verdict = ep.judge.verdicts.get(judge_check_key(ep, ids))
    return bool(verdict)

def basket_place(ep, cart: str, calls: list[dict[str, Any]]) -> list[str]:
    """Run cart calls; [] when one fails."""
    placed: list[str] = []
    for args in calls:
        fitted = shaped(args, ep.schemas.get(cart))
        if fitted is None:
            return []
        reply = ep.store.call(cart, fitted)
        if reply is None or ep.finished:
            return []
        basket_record(ep, cart, fitted)
        ep.ingest(cart, reply.get('observation'))
        placed.append(dumps(fitted))
    return placed

def basket_buy_confirmed(ep, buy: str, ids: list[str]) -> list[str]:
    """Check out; once more after a refusal."""
    args = shaped(dict(ep.brief.get('buy_args') or {}), ep.schemas.get(buy))
    if args is None:
        return []
    replies: list[str] = []
    for again in (False, True):
        if again:
            news_refresh_offers(ep)
            if not basket_passes(ep, ids) or ep.steps_left() < 1 or ep.finished or any(map(ep.is_gone, ids)):
                break
        go, said = gates_run(ep, 'order')
        replies.extend(filter(None, [said]))
        if not go:
            break
        reply = ep.store.call(buy, args)
        ep.log('completed', lines=ids, reply=clip(reply, 200))
        if reply is None:
            break
        ep.checkouts += 1
        seen = reply.get('observation')
        ended = bool(reply.get('finished'))
        if not ended:
            ep.refused_at = ep.step
        ep.ingest(buy, seen)
        replies.append(ep.render(seen))
        if ended or ep.finished:
            break
    return replies

def basket_sets_units(ep, name: str) -> bool:
    """Cart tool sets units rather than adding."""
    specs = list(filter(lambda s: isinstance(s, dict), (ep.schemas.get(name, {}).get('properties') or {}).values()))
    modal = any(isinstance(s.get('enum'), list) and len(s['enum']) >= 2 for s in specs)
    return not modal and any(s.get('minimum') == 0 and 'integer' in type_names(s) for s in specs)

def basket_record(ep, name: str, args: dict[str, Any]) -> None:
    """Keep the cart lines current after a cart call."""
    basket = ep.basket
    basket.touched = True
    basket.changed_at = ep.step
    spec = ep.schemas.get(name, {}).get('properties') or {}
    enums = (((spec.get(k) or {}).get('enum'), v) for k, v in args.items())
    removal = any(isinstance(c, list) and len(c) >= 2 and v in c[1:] for c, v in enums)
    named, whole_cart = basket_lines_named(ep, spec, args)
    if whole_cart:
        basket.lines = dict((i, c) for i, c in named if c and c > 0)
        ep.added_at = {i: ep.added_at.get(i, ep.step) for i in basket.lines}
        return
    sets = basket_sets_units(ep, name)
    for ident, count in named:
        have = units_after(basket.lines.get(ident, 0), count, removal, sets)
        if have <= 0:
            basket.lines.pop(ident, None)
            ep.added_at.pop(ident, None)
            continue
        basket.lines[ident] = have
        ep.added_at.setdefault(ident, ep.step)

def basket_lines_named(ep, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
    """The (id, units) pairs a cart call names, and whether the call states the whole cart."""
    rows = ep.mem.rows
    number = lambda values: next((whole(v) for v in values if whole(v) is not None and not isinstance(v, str)), None)
    loose = number(args.values())
    found: list[tuple[str, int | None]] = []
    whole_cart = False
    for key, value in args.items():
        if isinstance(value, str):
            entries = [value]
        elif isinstance(value, list):
            shape = (spec.get(key) or {}).get('items') or {}
            if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                whole_cart = True
            entries = value
        else:
            entries = []
        for line in entries:
            if isinstance(line, dict):
                whole_cart = True
                ident = next((v for v in line.values() if isinstance(v, str) and v in rows), None)
                if ident:
                    got = number(line.values())
                    found.append((ident, 1 if got is None else got))
            elif isinstance(line, str) and line in rows:
                found.append((line, loose))
    return found, whole_cart

def basket_late_pick(ep, number: int, held: list[str], index: dict[str, Any]) -> str | None:
    """A read page of item number that passes the code checks; None when none does."""
    kind, marks = mem_kind_key(ep), index['marks']
    needs = ep.items[number - 1].live()
    owner = lambda i: mem_owner(ep, mem_label(ep, i, kind))
    codes = set().union(*map(model_codes, needs))
    mains = {m for p, m in ep.bound if p == number}
    for i in held:
        if owner(i) in mains:
            codes.update(marks.get(i, {}).get('codes') or [])
    owns = {m.get('own') for m in marks.values()}
    blocked = lambda i: i in held or not mem_can_buy(ep, i) or i in ep.gone or i in ep.basket.dropped
    clash = lambda fits, more: bool(codes and fits) and not fits & codes and (not more or more in owns)
    picks = [i for i in reversed(list(ep.mem.pages)) if owner(i) == number and not blocked(i) and not judge_apart_needs(ep, i, needs)
             and not clash(judge_fit_set(i, index), (marks.get(i) or {}).get('more'))]
    if ep.cheapest():
        picks.sort(key=ep.price_rank)
    return picks[0] if picks else None

def judge_proves(ep, ident: str, needs: list[str]) -> bool:
    """The read record states every requirement as written: the value's words in order under the property's own key, else in
    another attribute, else in a short top-level text such as the title. Limits, exclusions and unresolved pointers prove
    nothing, and a listing with no checkable requirement proves nothing either."""
    page = ep.mem.pages.get(ident) or {}
    block = fact_block(page) or {}
    if not block or not needs:
        return False
    open_refs = set(ep.open_pointers())
    heads = [str(v) for v in page.values() if isinstance(v, str) and 0 < len(v) <= 160]
    for need in needs:
        prop, _, value = need.partition(':')
        value = value.strip()
        words = plain_words(value)
        if need in open_refs or not words or has_bound(value) or re.search('\\b(?:not|no|without|except|other than|unlike)\\b', value, re.I):
            return False
        own = [v for k, v in block.items() if names_same(stems(prop), stems(k))]
        said = lambda text: (lambda got: any(got[i:i + len(words)] == words for i in range(len(got) - len(words) + 1)))(plain_words(text))
        if not (any(map(said, own)) or (not own and (any(said(v) for v in block.values()) or any(map(said, heads))))):
            return False
    return not judge_against(ep, ident, needs) and not judge_apart_needs(ep, ident, needs)

def basket_proven_picks(ep) -> tuple[list[str], dict[int, str]] | None:
    """The cart's held lines and, per lacking item, its cheapest read buyable listing that proves the item's requirements;
    None when the cart lacks nothing or one lacking item has no such listing. No store call."""
    kind = mem_kind_key(ep)
    if not kind:
        return None
    held = [i for i in basket_ids(ep) if (basket_units(ep, i) or 0) > 0 and not ep.is_gone(i)]
    owners = {mem_owner(ep, mem_label(ep, i, kind)) for i in held}
    missing = [n for n in range(1, len(ep.items) + 1) if n not in owners]
    if not missing or 0 in owners:
        return None
    picks: dict[int, str] = {}
    for n in missing:
        needs = ep.items[n - 1].live()
        found = [i for i in ep.mem.pages if i not in held and mem_can_buy(ep, i) and i not in ep.basket.dropped
                 and mem_owner(ep, mem_label(ep, i, kind)) == n and judge_proves(ep, i, needs)]
        if not found:
            return None
        picks[n] = min(found, key=ep.price_rank)
    return held, picks

def basket_proven_fill(ep) -> bool:
    """Desk's own ending: every item the cart lacks gets the cheapest read, buyable listing that proves its requirements, all
    or none, inside the budget and the steps; True when lines were placed."""
    # the race-15 and race-17 crowns, and the leading needs-in-request run of race 18, fill a short cart in code before
    # their own ending and decline only when an item has no such listing; Arena r18 b02: 17 of our 20 declines came from Desk's
    # ending with an empty or short cart, 7 of them on tasks whose answer was an order
    tool = ep.roles.get('cart')
    if ep.finished or ep.store.blocked or not (tool and ep.roles.get('buy')):
        return False
    found = basket_proven_picks(ep)
    if found is None:
        ep.log('proven_fill', why='nothing lacking, or an item with no proving read listing')
        return False
    held, picks = found
    missing = list(picks)
    lines = {i: basket_units(ep, i) for i in held}
    lines.update((picks[n], ep.items[n - 1].qty) for n in missing)
    cap = ep.numeric_budget()
    prices = [mem_price(ep, i) for i in lines]
    if cap is not None:
        if None in prices:
            return False
        each = ep.brief.get('budget_scope') == 'each'
        if (max(prices) if each else sum(p * lines[i] for p, i in zip(prices, lines))) > cap + 1e-09:
            ep.log('proven_fill', picks=picks, why='budget')
            return False
    calls = basket_calls_for(ep, {i: q for i, q in lines.items() if i not in held}, [])
    if not calls or len(calls) + 2 > ep.steps_left() or EPISODE_SECONDS - ep.elapsed() < 6 + 2 * len(calls):
        ep.log('proven_fill', picks=picks, why='no room')
        return False
    placed = basket_place(ep, tool, calls)
    ep.log('proven_fill', picks=picks, placed=bool(placed))
    return bool(placed)

def basket_late_add(ep) -> None:
    """P8: at the clock stop, fill the items a cart still lacks."""
    roles = [ep.roles.get(role) for role in ('cart', 'buy', 'refuse')]
    cat_key = mem_kind_key(ep)
    stopped = ep.finished or ep.store.blocked or ep.llm.out_of_credit
    if stopped or not ep.past_stop() or not (all(roles) and cat_key):
        return
    held, owners = [], []
    for i in basket_ids(ep):
        if (basket_units(ep, i) or 0) > 0 and not ep.is_gone(i):
            held.append(i)
            owners.append(mem_owner(ep, mem_label(ep, i, cat_key)))
    missing = [n for n in range(1, len(ep.items) + 1) if n not in owners]
    if not (held and missing) or 0 in owners or len(set(owners)) < len(held):
        return
    index = judge_fit_index(ep)
    picks = {n: basket_late_pick(ep, n, held, index) for n in missing}
    entry = ep.log('late_add', step=ep.step, cart=held, picks=picks)
    if not all(picks.values()):
        entry['why'] = 'no page passes'
        return
    lines = dict(zip(held, (basket_units(ep, i) for i in held)))
    lines.update((str(p), ep.items[n - 1].qty) for n, p in picks.items())
    prices = [mem_price(ep, i) for i in lines]
    cap = ep.budget
    over = False
    if isinstance(cap, (int, float)) and None not in prices:
        each = ep.brief.get('budget_scope') == 'each'
        over = max(prices) > cap if each else sum(p * lines[i] for p, i in zip(prices, lines)) > cap + 1e-09
    calls: Any = []
    room = 0.0
    if over:
        why = 'budget'
    else:
        calls = basket_calls_for(ep, lines, [])
        room = EPISODE_SECONDS - ep.elapsed() - 8
        why = '' if calls and len(calls) + 3 <= ep.steps_left() and room >= 6 else 'no room'
    if why:
        entry['why'] = why
        return
    entry['checked'] = ok = basket_late_check(ep, list(lines), room)
    if ok and EPISODE_SECONDS - ep.elapsed() >= 6:
        entry['placed'] = basket_place(ep, roles[0], calls)

def basket_late_check(ep, ids: list[str], room: float) -> bool:
    """At most one model check, with its timeout cut to the time left; afterwards the call count is put back."""
    llm = ep.llm
    before = llm.calls
    timeout = getattr(_llm, 'timeout', None)
    timed = isinstance(timeout, (int, float))
    llm.calls = floor = max(before, ALL_CALLS - 1)
    if timed:
        _llm.timeout = max(4, int(min(timeout, room)))
    try:
        return basket_passes(ep, ids)
    finally:
        spent = max(0, llm.calls - floor)
        llm.calls = before + spent
        if timed:
            _llm.timeout = timeout

class Gatekeeper:
    """Required steps at the order or the ending."""

    def __init__(self) -> None:
        self.cleared: set[tuple[str, str]] = set()
        self.record: list[str] = []
        self.record_steps: list[int] = []
        self.sent = 0

def gates_note_line(ep, line: str, step: int) -> None:
    ep.gates.record.append(line)
    ep.gates.record_steps.append(step)

def gates_run(ep, stage: str) -> tuple[bool, str]:
    """Do the steps due at a stage: (go on, report)."""
    due = [p for p in ep.process if p['when'] == stage]
    if not due or ep.finished or not ep.stage_open(stage):
        return True, ''
    done: list[str] = []
    nudged = False
    rounds = 0
    while rounds < GATE_ROUNDS:
        rounds += 1
        got = ep.llm.ask(gates_prompt(ep, stage, due), [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 3500)
        if got is None:
            break
        call, change = gates_read_reply(ep, stage, got, done)
        if change:
            return False, change
        if not (call or done or nudged) and gates_unproven(ep, stage, due, got):
            nudged = True
            continue
        line = gates_do_call(ep, stage, call)
        if line is None:
            break
        done.append(line)
    return True, gates_said(done) if done else ''

def gates_read_reply(ep, stage: str, got: dict[str, Any], done: list[str]) -> tuple[dict[str, Any] | None, str]:
    """The gate model's call, and the report when the shopper's answers change the task ('' when they do not)."""
    told: list[str] = []
    hold = ''
    if done:
        # answers count only once a gate step has run (the first reply has nothing to answer yet)
        talk_apply_answers(ep, {'needs': got.get('needs')}, told)
        hold = str(got.get('hold') or '').strip()
    call = got.get('call') if isinstance(got.get('call'), dict) else None
    ep.log('gate', stage=stage, call=clip(call, 200), steps=clip(got.get('steps'), 120), hold=hold[:200], told=told)
    if not (told or hold):
        return call, ''
    ep.version += 1
    changes = '; '.join(told + [hold] * bool(hold))
    lead = gates_said(done) + '\n' if done else ''
    return call, f"{lead}What the shopper replied alters the task: {changes}. Adjust the cart to it, then send the checkout once more."

def gates_said(done: list[str]) -> str:
    return 'Before going on, Desk carried out the steps the request requires:\n' + '\n'.join(done)

def gates_prompt(ep, stage: str, due: list[dict[str, Any]]) -> str:
    buying = stage == 'order'
    record = '\n'.join(f'{n}. {line}' for n, line in enumerate(ep.gates.record, 1)) or '(nothing yet)'
    if ep.notes:
        record += '\n\nSTORE NOTES SEEN:\n' + '\n'.join(ep.notes[-4:])
    sections = (('', GATE.replace('@ACT@', 'place the order' if buying else 'end without a purchase')),
                ('REQUEST:\n', ep.request),
                ('STEPS DUE NOW:\n', '\n'.join(f"- {p['step']}" for p in due)),
                ('LEDGER:\n', req_ledger(ep)),
                ('CART NOW: ', basket_describe(ep)),
                ('RECORD (messages and step calls so far):\n', record),
                ('MESSAGES LEFT: ', f"{max(0, ep.msg_cap - ep.msg_sent)}   STEPS LEFT: {ep.steps_left()} (the {'order' if buying else 'ending'} itself takes one)"),
                ('TOOLS:\n', req_tool_digest(ep, ep.closing_tools())))
    return '\n\n'.join(head + body for head, body in sections)

def gates_unproven(ep, stage: str, due: list[dict[str, Any]], got: dict[str, Any]) -> bool:
    """Flag claimed steps no record line backs."""
    room = ep.msg_cap - ep.msg_sent
    if not ep.roles.get('ask') or room <= 0 or ep.steps_left() < 2:
        return False
    changed = ep.basket.changed_at
    # record and record_steps grow together (gates_note_line); a messaging step at the order counts only lines made since
    # the cart last changed
    fresh_only = stage == 'order' and any(p['messages'] > 0 for p in due)
    backed = {n for n, at in enumerate(ep.gates.record_steps, 1) if not fresh_only or at > changed}
    unproven = [claim for claim in as_list(got.get('steps')) if isinstance(claim, dict)
                and not (claim.get('done') is True and backed & {whole(n) or 0 for n in as_list(claim.get('lines'))})]
    if not unproven:
        return False
    ep.log('gate_unproven', steps=clip(unproven, 120))
    names = '; '.join(str(e.get('step') or '') for e in unproven)[:200]
    gates_note_line(ep, f'(Desk: since the cart last changed at step {changed}, no line shows {names} done; {room} '
                   f'message(s) and {ep.steps_left()} steps are left: name the call that does it)', -1)
    return True

def gates_do_call(ep, stage: str, call: dict[str, Any] | None) -> str | None:
    """Run the gate model's call; record line or None."""
    if not call:
        ep.gates.cleared.add((stage, f'{ep.cart_stamp()}#{ep.version}'))
        return None
    tool = str(call.get('tool') or '')
    asking = tool == ep.roles.get('ask')
    usable = tool in ep.schemas and tool not in ep.closing_tools() and ep.steps_left() >= 2 and not (asking and ep.msg_sent >= ep.msg_cap)
    given = call.get('args') if usable else None
    args = shaped(given if isinstance(given, dict) else {}, ep.schemas[tool]) if usable else None
    if args is None:
        return None
    if asking:
        text = next((v for v in args.values() if isinstance(v, str)), '')
        sent = talk_send(ep, [(1, text)], 1)
        if sent:
            ep.gates.sent += 1
            return f'Q: {text}\nA: {sent[0][2]}'
        return None
    reply = ep.store.call(tool, args)
    if reply is None:
        return None
    seen = reply.get('observation')
    mem_absorb(ep, seen)
    head = f'{tool} {dumps(args)} -> '
    shown = ep.render(seen)
    gates_note_line(ep, f'step {ep.step}: {head}{shown[:300]} (cart at that time: {ep.cart_stamp()})', ep.step)
    return head + shown[:400]

class Closer:
    """Holds on endings and the desk's own endings."""

    def __init__(self) -> None:
        self.variant_held = False
        self.swept = False
        self.sold_holds = 0
        self.lead_holds = 0
        self.end_holds = 0
        self.line_holds = 0
        self.fit_holds = 0
        self.pushed = False
        self.late_held = False
        self.refusal_ordered = False
        self.spare_read = False

def close_policy(ep, job: Call) -> str:
    """Ending holds in plan order: the first one that answers holds the call back; '' lets the call go through."""
    if not job.ending:
        return ''
    for name in close_plan(ep, job):
        hold, with_call = CLOSE_STEPS[name]
        said = hold(ep, job) if with_call else hold(ep)
        if said:
            return said
    return ''

def close_plan(ep, job: Call):
    """The ordered hold names for an ending call; a checkout re-checks the steps left once the offer pass is done."""
    if ep.mode_ok('buy', job.name, job.args):
        yield 'offer'
        yield 'spare'
        yield 'sweep'
        if ep.steps_left() > 2:
            yield from ('variant', 'wider', 'changed', 'sold', 'chain', 'misfit', 'lines', 'review', 'stale')
        yield 'gate'
        return
    yield 'after_refusal'
    if ep.steps_left() > 5:
        yield from ('audit', 'leads', 'second_end', 'late_end')
    else:
        yield from ('second_end', 'late_end', 'late_review')
    yield from ('unsearched', 'closest', 'by_lead', 'gate')

def close_offer_pass(ep) -> str:
    """Offer pages of the cart lines unread since carting: read each, and hold only when one changed and steps remain."""
    due = ep.unoffered()
    tool = ep.offer_view()
    target = ep.id_param(tool) if tool else None
    if not (due and target) or ep.steps_left() < len(due) + 1:
        return ''
    lines = []
    for ident, observation in page_reads(ep, tool, target, due):
        ep.offer_seen[ident] = ep.step
        ep.page_seen.add(ident)
        old = dict(ep.mem.rows.get(ident) or {})
        ep.ingest(tool, observation)
        new = ep.mem.rows.get(ident) or {}
        moved = [k for k, v in new.items() if k in old and old[k] != v and not isinstance(v, (dict, list))]
        if not moved and not ep.is_gone(ident):
            continue
        what = ', '.join(moved[:4]) or 'availability'
        lines.append(f'{ident} changed ({what}): {dumps(clip(observation, 300))[:900]}')
    ep.log('offer_pass', ids=due, changed=len(lines))
    if not lines:
        return ''
    ep.version += 1
    # with two steps or fewer left only a sold-out line is worth the hold
    if ep.steps_left() > 2 or any(ep.is_gone(i) for i in due):
        return 'Desk read the offer page of each cart line before checkout; these changed:\n' + '\n'.join(lines) + close_cart_now(ep)
    return ''

def close_cart_now(ep) -> str:
    return '\nCart now: ' + basket_describe(ep)

def close_hold_wider(ep) -> str:
    wider = scout_wider_hint(ep)
    return wider + close_cart_now(ep) if wider else ''

def close_hold_changed(ep, job: Call) -> str:
    """Re-read lines, view the cart, hold on change."""
    basket = ep.basket
    job.reread = news_refresh_offers(ep)
    if ep.news.sold_flag:
        news_locate_sold(ep)
    tool = ep.roles.get('cart_view')
    if not tool or not basket.touched or basket.viewed_at >= basket.changed_at or ep.steps_left() <= 3:
        return ''
    args = shaped({}, ep.schemas.get(tool))
    view = None if args is None else ep.store.call(tool, args)
    if view is None:
        return ''
    seen = view.get('observation')
    basket.view, basket.viewed_at = seen, ep.step
    change = ep.ingest(tool, seen)
    if change:
        ep.queued = []
        return ('Not run yet (no step used). Viewing the cart showed a change: ' + change + close_cart_now(ep)
                + '\nApply the change, fix the cart if needed, then check out.')
    return ''

def close_hold_sold(ep) -> str:
    swap = news_swap_hint(ep)
    if swap:
        return 'Not run yet (no step used). ' + swap + close_cart_now(ep)
    held = [i for i in basket_ids(ep) if i in ep.gone] if ep.close.sold_holds < 2 else []
    if not held:
        return ''
    ep.close.sold_holds += 1
    return f"Not run yet (no step used). {', '.join(held)} in the cart is sold out: replace that line first." + close_cart_now(ep)

def close_hold_chain(ep, job: Call) -> str:
    job.chain, hold = chains_walk(ep)
    return 'Not run yet (no step used). ' + job.chain + close_cart_now(ep) if hold else ''

def close_hold_misfit(ep) -> str:
    if ep.close.fit_holds >= 2 or ep.steps_left() <= ep.reserve() + 3:
        return ''
    misfit = judge_fit_faults(ep)
    if not misfit:
        return ''
    ep.close.fit_holds += 1
    ep.log('fit_hold', faults=misfit[:4])
    return ('Not run yet (no step used). A carted part does not work with the carted device: ' + '; '.join(misfit[:3])
            + '. Replace the part (or the device), then check out.' + close_cart_now(ep))

def close_hold_lines(ep) -> str:
    """Hold for wrong lines; fix quantities here."""
    faults = judge_line_faults(ep) if ep.close.line_holds < 2 and ep.steps_left() > ep.reserve() + 2 else []
    if not faults:
        return ''
    ep.close.line_holds += 1
    # faults that are only counts are fixed by Desk itself; a hold follows only when some fault is left after that
    fixed = basket_fix_counts(ep, basket_count_fixes(ep)) if all(' is held at ' in f for f in faults) else []
    if fixed:
        faults = judge_line_faults(ep)
        if not faults:
            return ("Not run yet (no step used). Desk itself set the quantities in the cart: " + '; '.join(fixed)
                    + '. If the store turned any of these down, fix those lines; then check out.' + close_cart_now(ep))
    ep.log('line_hold', faults=faults[:6])
    return ('Not run yet (no step used). The cart and the request still differ: ' + '; '.join(faults[:6])
            + ". Fix those lines by adding or taking off units as the cart tool allows, then check out." + close_cart_now(ep))

def close_hold_review(ep, job: Call) -> str:
    read = [*scout_read_pages(ep, basket_ids(ep), ep.reserve() + 3), *scout_cheaper_rows(ep)]
    verdict = judge_check_cart(ep)
    if verdict:
        seen = "For the check Desk read the attributes of these listings (one step each):\n" + '\n'.join(read) + '\n' if read else ''
        return 'Not run yet. ' + job.chain + seen + verdict + close_cart_now(ep)
    close_ref_pass(ep)
    return ''

def close_ref_pass(ep) -> list[str]:
    """One hop: open the listings that a cart line's own page points to for the rest of that line's details."""
    # real races 12-16 (all agents, same lines seen both ways): a final cart line whose page sent the reader on to another
    # listing passed 35/53 when that listing was opened vs 42/201 when not (r14 17/17 vs 38/160, r13 17/32 vs 22/102); the
    # passers searched the named code and opened the hit, one hop (Arena Cairn traces). At most two such reads per checkout
    judge = ep.judge
    held = set(basket_ids(ep))
    shown: list[str] = []
    todo = [c for c in judge.refs if c not in judge.refs_read][:2]
    while todo and not (ep.finished or ep.past_stop() or ep.steps_left() < ep.reserve() + 3):
        code = todo.pop(0)
        judge.refs_read.add(code)
        ident = code_listing(code, ep.mem.rows)
        ident = ident if ident in ep.mem.rows else scout_code_row(ep, code)
        fresh = ident and ident not in held and ident not in ep.mem.pages
        if fresh and ep.steps_left() >= ep.reserve() + 2:
            shown += scout_read_pages(ep, [ident], ep.reserve() + 1)
    if shown:
        ep.log('ref_pass', refs=judge.refs, read=len(shown))
    return shown

def scout_code_row(ep, code: str) -> str:
    """One free-text search for a listing code; the row carrying the code in its title (else its record), or ''."""
    plan = scout_search_template(ep)
    if not plan or code_pattern(code) is None:
        return ''
    tool, base, field = plan
    args = shaped({**base, field: code}, ep.schemas.get(tool))
    reply = ep.store.call(tool, args) if args is not None else None
    if reply is None:
        return ''
    seen = reply.get('observation')
    ep.ingest(tool, seen)
    listed = [i for i, _ in mem_listings_in(ep, seen)]
    if not listed:
        return ''
    rows = ep.mem.rows
    return code_listing(code, {i: rows[i] for i in listed if i in rows})

def close_hold_stale(ep, job: Call) -> str:
    if not job.reread or ep.llm.cost <= SPEND_SOFT:
        return ''
    return ("Not run yet (no step used). Desk has just re-read the listings in the cart:\n" + job.reread
            + '\nSend the checkout again if each line still shows stock and a price within the budget; if one does not, replace it first.')

def close_hold_audit(ep) -> str:
    review = judge_audit(ep)
    return 'Not run yet (no step used). ' + review if review else ''

def close_hold_leads(ep) -> str:
    """Hold an ending while review leads exist."""
    if not ep.leads or ep.steps_left() <= 4 or ep.close.lead_holds > 1:
        return ''
    ep.close.lead_holds += 1
    if ep.close.lead_holds > 1:
        first = ep.leads[0]
        # the check may change the leads, so the lead is looked up again after it
        if not basket_passes(ep, [first]) or first not in ep.leads:
            return ''
        ep.confirmed = first
        ep.log('refuse_blocked', confirmed=first)
        return (f"Not run yet (no step used). Checked separately, the records of {first} meet every requirement. Cart it at the "
                'requested quantity and check out; end unbought only if its record clearly breaks a requirement.')
    ep.log('refuse_blocked', leads=list(ep.leads))
    return ('Not run yet (no step used). The review found listings that seem to fit: ' + ', '.join(ep.leads) + '. Read whichever you '
            'need, cart the best fit under the pick rule and check out. Ending unbought is allowed only when every one of them plainly '
            'breaks a requirement: name that requirement in your text and send the ending call again.')

def close_after_refusal(ep) -> str:
    """Once the store has refused an ending of either kind, an unbought ending makes way for an order: the last complete
    cart when the independent check passes it under today's requirements, else a confirmed review lead."""
    # real race 18 (49 fetched agents): unbought endings after a refused ending paid 1 in 120 and no first ending was
    # refused on a decline task; ordering after a refused decline paid .26 of 160. Race 16: .40 of 10 against 0 of 6
    close = ep.close
    if close.refusal_ordered or not (ep.end_refusals or ep.refused_at >= 0) or not ep.roles.get('buy') or ep.basket.completed:
        return ''
    close.refusal_ordered = True
    cart = [i for i in ep.basket.built if not ep.is_gone(i)]
    if len(cart) >= len(ep.items) and cart and ep.steps_left() >= len(cart) + 2 and basket_passes(ep, cart):
        done = basket_reorder(ep, cart)
        if done:
            ep.log('after_refusal_cart', lines=cart)
            return done + (' [episode finished]' if ep.finished else '')
    if not ep.leads and ep.steps_left() > len(ep.items) + 3:
        judge_audit(ep, len(ep.items) + 1)
    if ep.leads and basket_passes(ep, ep.leads[:1]):
        ep.confirmed = ep.leads[0]
        done = basket_complete_cart(ep)
        if done:
            ep.log('after_refusal_lead', lead=ep.confirmed)
            return done + (' [episode finished]' if ep.finished else '')
    return ''

def basket_reorder(ep, cart: list[str], why: str = 'an ending had already been refused and ') -> str:
    """Set a checked complete cart again, each line at its item's quantity, and check out."""
    tool, buy = ep.roles.get('cart'), ep.roles.get('buy')
    if not (tool and buy):
        return ''
    key = mem_kind_key(ep)
    wanted = {}
    for ident in cart:
        owner = mem_owner(ep, mem_label(ep, ident, key)) if key else 0
        wanted[ident] = ep.items[owner - 1].qty if 0 < owner <= len(ep.items) else max(1, ep.basket.lines.get(ident, 1))
    held = {i: basket_units(ep, i) for i in basket_ids(ep)}
    surplus = [i for i in held if i not in wanted]
    change = {i: q for i, q in wanted.items() if held.get(i) != q}
    calls = basket_calls_for(ep, change, surplus) if change or surplus else []
    if calls is None or len(calls) + 1 > ep.steps_left():
        return ''
    ep.basket.completed = True
    placed = basket_place(ep, tool, calls) if calls else ['cart already as checked']
    replies = basket_buy_confirmed(ep, buy, list(wanted)) if placed else []
    if not replies:
        return ''
    return (f'Not run (Desk acted instead): {why}the independent check passed the last complete '
            f"cart ({', '.join(wanted)}), so Desk set it again ({'; '.join(placed)}) and checked out. Store reply: " + ' | then: '.join(replies))

def close_spare_page(ep) -> str:
    """First checkout of a cheapest pick with ten or more steps to spare: the next page of the search that showed a cart
    line, read once; the checkout waits only when a cheaper buyable row of that kind appears."""
    # real race 18, 1-product tasks with every need known: our paying listing showed .75 of the time against the leaders'
    # .86 while we used 24.2 of a 34-step cap (leader 26.5); a page 2 gave a first paying row .10-.15 of calls
    close = ep.close
    if close.spare_read or ep.checkouts or not ep.cheapest() or ep.steps_left() - ep.reserve() < 10 or ep.past_stop():
        return ''
    close.spare_read = True
    carted = basket_ids(ep)
    source = next(((n, a, o) for n, a, o in reversed(ep.scout.search_log) if set(carted) & {i for i, _ in mem_listings_in(ep, o)}), None)
    plan = scout_page_plan(ep, source[0], source[2]) if source else None
    if not plan or plan[3] is None:
        return ''
    texts, _ = scout_read_more(ep, source[0], source[1], plan[2], plan[3], 1)
    if not texts:
        return ''
    line = min([mem_price(ep, i) or float('inf') for i in carted] or [float('inf')])
    under = [i for i in judge_requested_unread(ep, [i for i, _ in mem_listings_in(ep, ep.scout.extra_pages[-1])]) if ep.price_rank(i) < line][:2]
    ep.log('spare_page', tool=source[0], cheaper=under)
    if not under:
        return ''
    read = scout_read_pages(ep, under, ep.reserve() + 3)
    return ('Not run yet (no step used). With the cheapest as the pick and steps to spare, Desk read the next page of the search that '
            'showed the cart line, and cheaper rows of that kind appeared:\n' + '\n'.join(read or texts)
            + '\nSwap one in only if it meets every requirement in force; if not, check out the cart unchanged.' + close_cart_now(ep))

def close_cheapest_sweep(ep) -> str:
    """First checkout of a cheapest pick with ten or more spare steps: each carted item searched under the seller names its rows
    carry; the checkout waits once, and only when cheaper unread rows of that kind appeared (the two cheapest are read)."""
    # Arena r18 needs-in-request (fix loop round 1): on 7 cheapest picks the paid row never showed for us; the run that
    # solved them searched seller name + value + noun after its first cart line and found it at steps 21-31, while we
    # checked out with 9-16 steps unused
    close = ep.close
    spare = ep.steps_left() - ep.reserve()
    if close.swept or ep.checkouts or not ep.cheapest() or spare < 6 or ep.elapsed() > CLOCK_STOP - 60:
        return ''
    plan, kind = scout_search_template(ep), mem_kind_key(ep)
    if plan is None or not kind:
        return ''
    tool, base, field = plan
    sent = {str(a.get(field) or '').strip().lower() for a in ep.scout.searched}
    lines = []
    for line in basket_ids(ep):
        number = mem_owner(ep, mem_label(ep, line, kind)) or (1 if len(ep.items) == 1 else 0)
        if 0 < number <= len(ep.items):
            lines.append((line, number, scout_title_noun(ep, number) or ep.items[number - 1].name))
    # with 6-9 spare steps only while no bare-noun search was sent (Arena 18550: a first checkout with 6 left never swept)
    if not lines or (spare < 10 and all(noun.strip().lower() in sent for _, _, noun in lines)):
        return ''
    close.swept = True
    room = min(5, spare - 4)
    cheaper: list[str] = []

    def run(words: str, number: int, top: float, paged: bool) -> bool:
        nonlocal room
        args = shaped(scout_whole_words(ep, tool, {**base, field: words}), ep.schemas.get(tool))
        said = str((args or {}).get(field) or '').strip().lower()
        if args is None or not said or said in sent:
            return True
        sent.add(said)
        room -= 1
        ok, seen = scout_fetch(ep, tool, args)
        if not ok:
            return False
        pages = [seen] + (scout_reach_pages(ep, tool, args, seen) if paged else [])
        ids = [i for page in pages for i, _ in mem_listings_in(ep, page)]
        cheaper.extend(i for i in judge_requested_unread(ep, ids)
                       if ep.price_rank(i) < top and mem_owner(ep, mem_label(ep, i, kind)) in (number, 0 if len(ep.items) == 1 else -1))
        return True

    for line, number, noun in lines:
        top = mem_price(ep, line) or float('inf')
        rows = [ep.mem.rows[i] for i in mem_ids_of_item(ep, number) if i in ep.mem.rows] or [ep.mem.rows.get(line) or {}]
        titles = {w for i in mem_ids_of_item(ep, number) for w in plain_words(mem_title(ep, i))}
        named: dict[str, int] = {}
        for v in (v for row in rows for v in shop_values(row) if re.search('[a-z]{3}', v, re.I) and v.lower() not in ('none', 'null')):
            named[v] = named.get(v, 0) + 1
        # a word search matches titles: names the item's titles carry go first (Arena 18550: seller names no title holds
        # used the room while the paid row's brand, seen in titles, waited)
        carried = lambda v: all(w in titles for w in plain_words(v))
        order = sorted(named, key=lambda v: (not carried(v), -named[v]))
        # the bare product noun first when never searched (Arena 18550: two paid rows sat on later pages of the bare noun)
        if room > 0 and not ep.finished and not run(noun, number, top, True):
            break
        values = scout_value_words(ep, number)
        for seller in order:
            if room <= 0 or ep.finished or ep.past_stop():
                break
            if not run(' '.join(w for w in (seller, values[0] if values else '', noun) if w), number, top, False):
                room = 0
    under = list(dict.fromkeys(cheaper))
    under.sort(key=ep.price_rank)
    ep.log('cheapest_sweep', cheaper=under[:4], left=ep.steps_left())
    if not under:
        return ''
    read = scout_read_pages(ep, under[:2], ep.reserve() + 3)
    if not read:
        return ''
    return ('Not run yet (no step used). Price decides this pick and steps were left over, so Desk searched each item under the seller '
            'names its rows show, and cheaper rows of that kind appeared; Desk read the cheapest:\n' + '\n'.join(read)
            + '\nPut one in place of its item\'s line only if it meets every requirement in force; otherwise check out the cart unchanged.'
            + close_cart_now(ep))

def title_lists_variants(title: str) -> bool:
    """A title naming two or more different figures with the same suffix: a listing of several variants."""
    seen: dict[str, set[str]] = {}
    for figure, suffix in re.findall('(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s?([a-z]{1,4})\\b', str(title).lower()):
        seen.setdefault(suffix, set()).add(figure)
    return any(len(v) >= 2 for v in seen.values())

def mem_title(ep, ident: str) -> str:
    row = ep.mem.rows.get(ident) or ep.mem.pages.get(ident) or {}
    return next((v for k, v in row.items() if isinstance(v, str) and re.search('title|name', str(k), re.I)
                 and not re.search(SHOP_FIELD, str(k), re.I)), '')

def close_hold_variants(ep) -> str:
    """Once, before checkout: a cart line whose title lists several variants, while a read single-variant listing of that item
    proves every requirement (not dearer on a cheapest pick), is named with that listing."""
    # real race 18: orders holding a line whose title lists several variants paid .099 of 5,123 against .445 of 23,802
    close, kind = ep.close, mem_kind_key(ep)
    if close.variant_held or not kind:
        return ''
    close.variant_held = True
    swaps = []
    for line in basket_ids(ep):
        if not title_lists_variants(mem_title(ep, line)):
            continue
        number = mem_owner(ep, mem_label(ep, line, kind))
        if not 0 < number <= len(ep.items):
            continue
        top = ep.price_rank(line)
        needs = ep.items[number - 1].live()
        better = [i for i in ep.mem.pages if i != line and i not in basket_ids(ep) and mem_can_buy(ep, i)
                  and mem_owner(ep, mem_label(ep, i, kind)) == number and not title_lists_variants(mem_title(ep, i))
                  and (not ep.cheapest() or ep.price_rank(i) <= top) and judge_proves(ep, i, needs)]
        if better:
            swaps.append((line, min(better, key=ep.price_rank)))
    ep.log('variant_hold', swaps=swaps)
    if not swaps:
        return ''
    named = '; '.join(f'{line} lists several variants in its title, while {alt} states one and its record shows every requirement'
                      for line, alt in swaps)
    return ('Not run yet (no step used). ' + named + '. Orders holding a line that lists several variants seldom pass; put the '
            'named listing in place of that line unless its record breaks a requirement, then check out.' + close_cart_now(ep))

def close_hold_second_end(ep) -> str:
    if NO_SECOND_END and ep.end_refusals and ep.close.end_holds < 2:
        ep.close.end_holds += 1
        ep.log('second_end_held', refused=ep.end_refusals)
        return ('Not run (no step used): the store has already refused an ending without a purchase in this episode, so a second one '
                "would fail as well. Act on the store's reply: cart the best fitting listing read so far for every item at its requested "
                'quantity, and check out.' + close_cart_now(ep))
    return ''

def close_hold_late_end(ep) -> str:
    """Put a once-complete cart back, once."""
    # real races 5-14: 118 failed episodes dropped a carted paid listing still in stock and in
    # budget, then ended unbought, against at most 37 right endings after a cart
    if ep.close.late_held:
        return ''
    basket = ep.basket
    lines = [i for i in basket.built if not ep.is_gone(i)]
    if ep.steps_left() > 5:
        # before the last steps only a cart no requirement was added to since, inside the budget in force
        cap = ep.numeric_budget()
        grown = {need for _, need in ep.live_needs()} - basket.built_needs
        over = cap is not None and sum((mem_price(ep, i) or 0) * max(1, basket_units(ep, i)) for i in lines) > cap
        # real race 18: putting back every once-complete cart would win 41 orders and lose 44 right declines
        if (grown or over) and not (lines and len(lines) >= len(ep.items) and basket_passes(ep, lines)):
            return ''
    stale = set() if basket_fresh_view(ep) else basket.dropped
    carted = basket_ids(ep)
    missing = [i for i in lines if i not in carted or i in stale]
    if len(lines) < len(ep.items) or len(missing) >= ep.steps_left():
        return ''
    ep.close.late_held = True
    if (ep.steps_left() <= 5 and not ep.basket.completed and EPISODE_SECONDS - ep.elapsed() > 25
            and len(missing) + 1 <= ep.steps_left() and basket_passes(ep, lines)):
        # the Halo fork's late finish: with the last steps, Desk puts the once-complete cart back and orders it itself
        # instead of asking for more turns (Arena r18 b02: the clock ended 26 of 84 episodes past 250 s)
        done = basket_reorder(ep, lines, 'the cart once held every item; ')
        if done:
            ep.log('late_finish', lines=lines)
            return done + (' [episode finished]' if ep.finished else '')
    ep.log('late_decline_held', lines=lines, missing=missing)
    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
    return ('Not run yet (no step used). Ending unbought scores nothing when a fitting listing exists, and running short of steps does '
            'not show that none exists. ' + redo + ' End without a purchase only if your search turned up a requested item for which no '
            'listing meets its requirements; in that case send the ending call again.')

def close_hold_unsearched(ep) -> str:
    room = min(ep.steps_left(), ep.clock_room())
    if not (ep.leads or ep.close.pushed) and room > ep.max_steps // 3:
        ep.close.pushed = True
        return (f'Not run yet (no step used). {room} steps are left; spend them on searches not tried yet: open the category list of each '
                "requested item and page on with the store's offset or token, query single value words and other names of the property, "
                'and open every listing that might fit. Only after that end without a purchase.')
    return ''

def close_hold_late_review(ep) -> str:
    done = close_decline_review(ep)
    return done + (' [episode finished]' if ep.finished else '') if done else ''

def close_decline_review(ep) -> str:
    """One review before an unreviewed ending."""
    # real races 11-14: most unbought endings of the know line on payable tasks came with 6 steps
    # left or fewer; love's own study put 7 of 9 at exactly 5
    barred = (ep.judge.audited or ep.basket.completed or ep.finished or not (ep.roles.get('cart') and ep.roles.get('buy'))
              or ep.steps_left() < len(ep.items) + 2 or ep.elapsed() > CLOCK_STOP - 30 or ep.llm.cost > SPEND_HARD)
    if barred:
        return ''
    judge_audit(ep, len(ep.items) + 1)
    ep.log('decline_review', left=ep.steps_left(), leads=list(ep.leads))
    if not ep.leads:
        return ''
    ep.confirmed = ep.leads[0]
    return basket_complete_cart(ep)

def screen_added_lines(ep, job: Call) -> tuple[list[str], set[str]]:
    """Listings a cart call adds units of that have no cart line yet, and the ids the cart holds."""
    carted = set(basket_ids(ep))
    named, _ = basket_lines_named(ep, (ep.schemas.get(job.name) or {}).get('properties') or {}, job.args)
    return [ident for ident, units in named if (units is None or units > 0) and ident not in carted], carted

def close_hold_closest(ep) -> str:
    judge_closest_hunt(ep)
    return ''

def close_end_by_lead(ep, job: Call) -> str:
    """Buy a confirmed lead, else quiet the ending."""
    done = basket_complete_cart(ep)
    if not done:
        job.args = close_quiet(ep, job.name, job.args)
        return ''
    return done + (' [episode finished]' if ep.finished else '')

def close_hold_gate(ep, job: Call) -> str:
    if ep.steps_left() < 2:
        return ''
    stage = 'order' if ep.mode_ok('buy', job.name, job.args) else 'end'
    go, said = gates_run(ep, stage)
    if go:
        if said:
            job.extra = '\n[Desk] ' + said + job.extra
        return ''
    return 'Not run yet. ' + said + close_cart_now(ep)

def close_quiet(ep, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Free-text ending arguments."""
    schema = ep.schemas.get(name) or {}
    required = schema.get('required') or []
    out = dict(args)
    for key, spec in (schema.get('properties') or {}).items():
        if isinstance(spec, dict) and 'string' in type_names(spec) and 'enum' not in spec:
            text = close_quiet_text(ep, spec, out.get(key), key in required)
            if text is not None:
                out[key] = text
    return out

def close_quiet_text(ep, spec: dict[str, Any], value: Any, required: bool) -> str | None:
    """The text for one free-text ending field: the offer note, the plain decline, or None to keep the model's words."""
    cap = spec.get('maxLength')
    offer = judge_offer_note(ep, cap if isinstance(cap, int) else 400)
    if offer:
        ep.log('offer_note', text=offer)
        return offer
    if value is None and not required:
        return None
    text = str(value or '')
    if text and ep.brief.get('quiet_decline') is False and not any(ident in text for ident in ep.mem.rows):
        return None
    return 'Ending with no purchase: no listing here satisfies every requirement you gave.'[:spec.get('maxLength', 200)]

# hold name -> (hold, whether it takes the ending call); close_plan gives the order
CLOSE_STEPS = {
    'offer': (close_offer_pass, False), 'wider': (close_hold_wider, False), 'changed': (close_hold_changed, True),
    'sold': (close_hold_sold, False), 'chain': (close_hold_chain, True), 'misfit': (close_hold_misfit, False),
    'lines': (close_hold_lines, False), 'review': (close_hold_review, True), 'stale': (close_hold_stale, True),
    'audit': (close_hold_audit, False), 'leads': (close_hold_leads, False), 'second_end': (close_hold_second_end, False),
    'late_end': (close_hold_late_end, False), 'late_review': (close_hold_late_review, False),
    'unsearched': (close_hold_unsearched, False), 'closest': (close_hold_closest, False), 'by_lead': (close_end_by_lead, True),
    'gate': (close_hold_gate, True), 'after_refusal': (close_after_refusal, False), 'spare': (close_spare_page, False),
    'sweep': (close_cheapest_sweep, False), 'variant': (close_hold_variants, False),
}

def close_finish(ep, again: bool = False) -> str:
    """The desk's own ending of the episode."""
    if ep.finished or ep.store.blocked or ep.steps_left() <= 0:
        return ''
    basket = ep.basket
    buy, refuse = ep.roles.get('buy'), ep.roles.get('refuse')
    if refuse and len(basket_ids(ep)) < len(ep.items):
        basket_proven_fill(ep)
        if ep.finished or ep.steps_left() <= 0:
            return ''
    fresh = basket_fresh_view(ep)
    holding = bool(basket_ids(ep) if fresh else basket.lines)
    # a purchase missing a requested item earns nothing: such a cart ends unbought instead
    short = holding and refuse is not None and len(basket_ids(ep)) < len(ep.items)
    if short:
        ep.log('short_cart_end', cart=basket_ids(ep))
    if buy and not short and (holding or (basket.touched and not fresh and ep.steps_left() >= 2)):
        args = shaped(dict(ep.brief.get('buy_args') or {}), ep.schemas.get(buy))
        broken = [] if args is None else close_blind_broken(ep)
        if broken and refuse:
            ep.log('forced_buy_blocked', ids=broken)
            short = True
        elif args is not None:
            said = close_forced_buy(ep, buy, args, again)
            if said is not None:
                return said
    if refuse and (short or not holding) and (ep.end_refusals or ep.refused_at >= 0):
        done = close_after_refusal(ep)
        if ep.finished or done:
            return '' if ep.finished else done
    if refuse and (short or not holding):
        done = close_decline_review(ep)
        if ep.finished:
            return ''
        if done:
            return done
    return close_forced_refuse(ep, refuse) if refuse else ''

def screen_start_value(value: Any) -> bool:
    """A paging value that names the first page (or none)."""
    return value in (None, '', 0, '0', 1, '1') or isinstance(value, bool)

def screen_asked_before(ep, args: dict[str, Any], skip: Any) -> bool:
    """A search with these arguments, the skipped keys aside, was sent already."""
    return any({k: v for k, v in s.items() if k not in skip} == args for s in ep.scout.searched)

def close_blind_broken(ep) -> list[str]:
    """Open the cart lines never read; return those whose page breaks a requirement of their item."""
    # the desk's own checkout (clock or step stop) first reads lines never opened and each unread line's offer
    # page: store reads are quick, and an order with an unread line misses the process
    blind = [i for i in basket_ids(ep) if i not in ep.mem.pages]
    if not blind or ep.steps_left() < len(blind) + 2:
        return []
    scout_read_pages(ep, blind, 2)
    cat_key = mem_kind_key(ep)
    broken: list[str] = []
    for ident in blind if cat_key else []:
        owner = mem_owner(ep, mem_label(ep, ident, cat_key))
        if 0 < owner <= len(ep.items) and judge_against(ep, ident, ep.items[owner - 1].live()):
            broken.append(ident)
    return broken

def close_forced_buy(ep, buy: str, args: dict[str, Any], again: bool) -> str | None:
    """The desk's checkout after the offer pass and the order gate; None passes the ending on to the decline."""
    close_offer_pass(ep)
    if not ep.finished and ep.steps_left() >= 2:
        gates_run(ep, 'order')
    if ep.finished or ep.steps_left() <= 0:
        return ''
    reply = ep.store.call(buy, args)
    ep.log('forced_buy', reply=clip(reply, 200))
    if ep.finished or ep.steps_left() <= 0:
        return ''
    if reply is None or not again:
        return None
    ep.checkouts += 1
    ep.refused_at = ep.step
    observation = reply.get('observation')
    ep.ingest(buy, observation)
    return 'the checkout was refused: ' + ep.render(observation)

def close_forced_refuse(ep, refuse: str) -> str:
    args = shaped(close_quiet(ep, refuse, dict(ep.brief.get('refuse_args') or {})), ep.schemas.get(refuse))
    if args is not None and ep.steps_left() >= 2:
        gates_run(ep, 'end')
    if args is None or ep.finished or ep.steps_left() <= 0:
        return ''
    reply = ep.store.call(refuse, args)
    ep.log('forced_refuse', reply=clip(reply, 200))
    if reply is None or ep.finished:
        return ''
    ep.end_refusals += 1
    ep.refused_at = ep.step
    observation = reply.get('observation')
    ep.ingest(refuse, observation)
    return 'ending without a purchase was refused: ' + ep.render(observation)

class Screener:
    """Screens model calls before they run."""

    def __init__(self) -> None:
        self.page_log: list[tuple[str, dict[str, Any], set[str], bool]] = []
        self.dead: dict[tuple[str, str], tuple[str, int] | None] = {}
        self.code_misses = 0
        self.code_found = False
        self.cheaper_held: set[str] = set()
        self.blind_held: set[str] = set()
        self.refused_run = 0

def screen_screen(ep, fn: dict[str, Any], batch: int) -> Call | str:
    """A Call, or the answer instead of running it."""
    if ep.finished:
        return 'Not run: the episode is over.'
    name = fn.get('name')
    refused = screen_tool_refusal(ep, name)
    if refused:
        return refused
    args, fault = screen_parse_args(ep, name, fn.get('arguments'))
    if fault:
        return 'Not run (no step used): ' + fault
    job = Call(name, args, ep.mode_ok('buy', name, args) or ep.mode_ok('refuse', name, args),
               ep.side_param() if name == ep.roles.get('compare') else None)
    rules = (screen_fit_room, screen_first_page, screen_page_value, screen_removal_count, screen_gone_target, screen_absent_line,
             screen_empty_checkout,
             lambda ep, job: ('Not run (no step used): go through your other results first, then send the finishing call by itself.'
                              if job.ending and batch > 1 else ''),
             screen_tail_kept, screen_pointer_hold, screen_unread_line, screen_cheaper_seen, screen_repeated, screen_cached, screen_shown,
             screen_page_stop, screen_dead_filter, screen_dead_code)
    return next(filter(None, (rule(ep, job) for rule in rules)), job)

def screen_tool_refusal(ep, name: Any) -> str:
    """Why a call to this tool is not run now ('' when it may run)."""
    allowed = sorted(t['function']['name'] for t in ep.loop.offered())
    listed = ', '.join(allowed)
    if name == ep.roles.get('ask'):
        return "Not run (no step used): Desk's interviewer sends every message to the shopper."
    if name not in ep.schemas:
        return f'Not run (no step used): no such tool. Tools: {listed}.'
    if name in allowed:
        ep.screen.refused_run = 0
        return ''
    # a paused search or read the model has asked for twice in a row runs once while an item has no line and the steps
    # still hold it plus a cart line and the checkout (Arena r18 noask b02: up to 18 refused turns in a row ran the clock out)
    ep.screen.refused_run += 1
    missing = len(ep.items) - len(basket_ids(ep))
    if (ep.screen.refused_run >= 3 and missing > 0 and name in (ep.roles.get('search'), ep.attr_tool())
            and ep.steps_left() >= missing + 3 and not ep.past_stop()):
        ep.screen.refused_run = 0
        ep.log('paused_call_run', tool=name, left=ep.steps_left())
        return ''
    # Arena b10: the short form made the model believe its tools were gone (a decline with steps left; turns without a
    # call); say what the remaining steps are kept for and which calls work now
    return (f'Not run (no step used): {ep.steps_left()} steps are left and Desk keeps them for finishing the '
            f'order, so this tool is paused (nothing is broken). Calls that work now: {listed}. Use them: cart a '
            'line that meets every requirement and check out, or end the way the request asks when nothing fits.')

def screen_parse_args(ep, name: str, raw: Any) -> tuple[dict[str, Any], str]:
    """The call's arguments fitted to its schema with titles mapped to ids, and the fault that stops it ('' when none)."""
    if not isinstance(raw, dict):
        raw = parse_object(raw) if (raw or '').strip() else {}
        if raw is None:
            return {}, 'the arguments were not a single JSON object.'
    faults: list[str] = []
    args = conform(scout_whole_words(ep, name, raw), ep.schemas[name], 'args', faults)
    if faults:
        return args, '; '.join(faults[:6])
    args, wrong = screen_resolve_ids(ep, name, args)
    return args, f'{wrong}.' if wrong else ''

def screen_fit_room(ep, job: Call) -> str:
    """Trim side-by-side calls to the known room."""
    ids, room = job.listed(), ep.side_room()
    if ids is None or len(ids) <= room or not ep.side_bad:
        return ''
    rest = ', '.join(map(str, ids[room:]))
    job.extra = f"\n[Desk] Sent only the first {room} listings: this store shows at most {room} side by side. Not shown yet: {rest}."
    job.args = {**job.args, job.many: ids[:room]}
    return ''

def screen_first_page(ep, job: Call) -> str:
    """Start a new query from its first page."""
    searching = job.name == ep.roles.get('search')
    paging = scout_paging_arg(ep, job.name) if searching else None
    if searching and not paging:
        return screen_loose_numbers(ep, job)
    value = job.args.get(paging) if paging else None
    if screen_start_value(value):
        return ''
    base = {k: v for k, v in job.args.items() if k != paging}
    if screen_asked_before(ep, base, {paging}):
        return ''
    schema = ep.schemas.get(job.name) or {}
    if paging in (schema.get('required') or []):
        spec = (schema.get('properties') or {}).get(paging) or {}
        kinds = set(type_names(spec))
        if 'integer' not in kinds and 'number' not in kinds:
            return ''
        low = spec.get('minimum')
        base[paging] = low if isinstance(low, int) else 0
    job.args = base
    job.extra += f'\n[Desk] Sent from the first page: {paging} {value!r} was from an earlier query, and a new query begins at its best rows.'
    return ''

def screen_loose_numbers(ep, job: Call) -> str:
    """Until a result shows what they do, a new query goes out without the numbers its schema does not explain: one may
    be a start position skipping the best rows (Arena race 17: 274 searches showed an empty page for a query that matched
    listings, in 35 of 47 failed tasks; real race 17: those episodes passed .16 against .37)."""
    dropped = {k: job.args[k] for k in scout_loose_numbers(ep, job.name)
               if job.args.get(k) not in (None, 0) and not isinstance(job.args.get(k), bool)}
    kept = {k: v for k, v in job.args.items() if k not in dropped}
    if not dropped or screen_asked_before(ep, kept, dropped):
        return ''
    job.args = kept
    ep.log('loose_dropped', step=ep.step, args=dropped)
    scout = ep.scout
    if scout.loose_notes < 2:
        scout.loose_notes += 1
        sent = ', '.join(f'{k}={dumps(v)}' for k, v in dropped.items())
        job.extra += f'\n[Desk] Sent without {sent}: the tool does not say what that number does, and a new query starts at its first row.'
    return ''

def screen_page_value(ep, job: Call) -> str:
    """Page on with the value the store handed back for this query."""
    paging = scout_paging_arg(ep, job.name) if job.name == ep.roles.get('search') else None
    value = job.args.get(paging) if paging else None
    if screen_start_value(value):
        return ''
    query = job.name + dumps({k: v for k, v in job.args.items() if k != paging})
    given = [v for v in ep.scout.next_given.get(query, []) if v is not None]
    if not given or dumps(value) in map(dumps, given):
        return ''
    *_, last = given
    job.args = {**job.args, paging: last}
    ep.log('page_value', step=ep.step, sent=value, used=last)
    job.extra += f'\n[Desk] Paged with {paging}={dumps(last)}, the value this search handed back ({dumps(value)} would skip rows).'
    return ''

def screen_removal_count(ep, job: Call) -> str:
    """A removal without a count is meant to take the whole line, so it carries the units Desk holds (Arena r17 b11: the
    store removed one unit, Desk took the line for gone and the model removed it again in 12 tasks, 9 failed; real r17:
    those episodes passed 4/23 against .33)."""
    if job.name != ep.roles.get('cart'):
        return ''
    spec = (ep.schemas.get(job.name) or {}).get('properties') or {}
    removal = any(isinstance(options, list) and len(options) >= 2 and v in options[1:]
                  for k, v in job.args.items() for options in [(spec.get(k) or {}).get('enum')])
    if not removal:
        return ''
    counts = [k for k, v in spec.items() if isinstance(v, dict) and {'integer', 'number'} & set(type_names(v)) and 'enum' not in v]
    ids = [v for v in job.args.values() if isinstance(v, str) and v in ep.basket.lines]
    if len(counts) != 1 or len(ids) != 1 or job.args.get(counts[0]) is not None:
        return ''
    (count,), (ident,) = counts, ids
    have = ep.basket.lines.get(ident) or 0
    if have > 1:
        job.args = {**job.args, count: have}
        ep.log('removal_count', step=ep.step, line=ident, units=have)
    return ''

def screen_gone_target(ep, job: Call) -> str:
    target = ep.id_param(job.name)
    if not target or job.name not in (ep.roles.get('open'), ep.roles.get('offer')):
        return ''
    ident = str(job.args.get(target))
    if ident not in ep.gone:
        return ''
    return f'Not run (no step used): {job.args[target]} sold out at step {ep.gone[ident]} for good. Put another fitting listing in its cart line instead.'

def screen_empty_checkout(ep, job: Call) -> str:
    """A checkout sent with an empty cart: the last complete cart when the check passes it today, else no step spent."""
    # Arena r18 noask b02: 4 checkouts reached the store with an empty cart (error E60, a step each); two of those tasks had
    # carted an accepted listing earlier, emptied the cart and ended unbought
    if not ep.mode_ok('buy', job.name, job.args) or basket_ids(ep):
        return ''
    cart = [i for i in ep.basket.built if not ep.is_gone(i)]
    if (cart and len(cart) >= len(ep.items) and not ep.basket.completed and ep.steps_left() >= len(cart) + 1
            and basket_passes(ep, cart)):
        done = basket_reorder(ep, cart, 'the checkout came with an empty cart; ')
        if done:
            ep.log('empty_checkout_cart', lines=cart)
            return done + (' [episode finished]' if ep.finished else '')
    ep.log('empty_checkout_held', built=cart)
    earlier = f" (the last complete cart held {', '.join(cart)})" if cart else ''
    return f'Not run (no step used): the cart is empty, so a checkout cannot go through{earlier}. Cart a line for every item, then check out.'

def screen_absent_line(ep, job: Call) -> str:
    """Units taken off a line not in the cart."""
    if job.name != ep.roles.get('cart'):
        return ''
    values = job.args.values()
    if all((whole(v) or 0) >= 0 for v in values if not isinstance(v, str)):
        return ''
    basket = ep.basket
    places = (basket.lines, dumps(basket.view) if basket.viewed_at >= basket.changed_at else '', basket.reply)
    for v in values:
        if isinstance(v, str) and v in ep.mem.rows and all(v not in place for place in places):
            return f'Not run (no step used): {v} has no line in the cart, so there is nothing to take off. Cart now: ' + basket_describe(ep)
    return ''

def screen_tail_kept(ep, job: Call) -> str:
    if job.ending or ep.steps_left() > ep.reserve() or not ep.closing_tools():
        return ''
    return "Not run (no step used): the remaining steps are kept for the checkout (plus the shopper's basket approval when due) or for an ending without a purchase."

def screen_pointer_hold(ep, job: Call) -> str:
    """A new cart line waits while its item's requirement still depends on a listing the request points at."""
    # real race 15, a pointer task: a first cart add made after the pointed listing was opened passed 7 of 7, one made
    # before it 15 of 33, and only after re-carting; never reaching the chain's last page passed 0 of 66 (Arena: an
    # early add led the model to treat the chain as settled and check out the early line)
    if job.ending or job.name != ep.roles.get('cart') or ep.checkouts or not ep.refs:
        return ''
    pending = ep.open_pointers()
    waiting = set()
    for n, need in ep.refs:
        if need in pending:
            waiting.add(n)
    if not waiting:
        return ''
    added, _ = screen_added_lines(ep, job)
    kind = mem_kind_key(ep)
    every = waiting >= set(range(1, len(ep.items) + 1))
    owners = [mem_owner(ep, mem_label(ep, ident, kind)) if kind else 0 for ident in added]
    held = [ident for ident, owner in zip(added, owners) if (owner in waiting if owner else every)]
    if not held or min(ep.steps_left(), ep.clock_room()) <= len(ep.items) + 6 or ep.chains.cart_holds >= 2 or ep.past_stop():
        return ''
    ep.chains.cart_holds += 1
    found = chains_recheck(ep)
    now_open = ep.open_pointers()
    still = [need for n, need in ep.refs if n in waiting and need in now_open]
    ep.log('pointer_hold', ids=held, pending=still, step=ep.step)
    ids = ', '.join(held)
    if still:
        chain = ' | '.join(still[:3])
        why = (f'{ids} is for an item whose requirement gets its value from another listing, and that chain is not resolved yet: '
               f'{chain}. Walk it: open the named listing and each listing its page leads on to, read the value off the last page, '
               'and cart a listing that has it (other items can go in the cart already).')
    else:
        known = '; '.join(f'{need} -> {ep.ref_values[need]}' for need in pending if need in ep.ref_values)
        why = (f'{ids} was picked before the pointed value was found. That value is known now: {known}. Cart a listing whose '
               'own page carries it (if this one does, send the same call again).')
    return f'Not run (no step used): {why}\n' + found

def screen_unread_line(ep, job: Call) -> str:
    """One hold for a new cart line whose listing was never read, while there is still room to read it."""
    # Arena r12-r16 b09 vs passing runs: lines carted straight from a search row (one by the fallback model) broke a
    # requirement or missed the process; the passing runs read every line before carting it
    tool = ep.attr_tool()
    readable = tool and ep.steps_left() > ep.reserve() + 2
    blind = []
    if readable and not job.ending and job.name == ep.roles.get('cart'):
        added, _ = screen_added_lines(ep, job)
        blind = [i for i in added if i not in ep.mem.pages and i not in ep.screen.blind_held]
    if not blind:
        return ''
    ep.screen.blind_held.update(blind)
    ep.log('unread_line_hold', ids=blind)
    return (f"Not run (no step used): {', '.join(blind)} has not been opened, so its attributes are unchecked. First open it with {tool}; "
            'cart it only if it meets every requirement of its item (if it does, send the same call again).')

def screen_cheaper_seen(ep, job: Call) -> str:
    """With a lowest-price rule, one hold for a new cart line when a cheaper listing of its kind shows no clash."""
    # Arena r16 noask b02: 6 'not the best option' orders; in tight 35-step episodes the checkout came with 1-2 steps left,
    # so no close review ran, and a cheaper opened listing meeting every requirement stayed out of the cart
    held = ep.screen.cheaper_held
    if job.ending or job.name != ep.roles.get('cart') or not ep.cheapest() or len(held) >= 2:
        return ''
    cat_key = mem_kind_key(ep)
    if not cat_key:
        return ''
    added, carted = screen_added_lines(ep, job)
    notes = []
    for ident in added:
        price = None if ident in held else mem_price(ep, ident)
        note = '' if price is None else screen_cheaper_line(ep, ident, price, cat_key, carted)
        if note:
            held.add(ident)
            notes.append(note)
    if not notes:
        return ''
    ep.log('cheaper_hold', notes=notes)
    return ('Not run (no step used): the lowest total wins here. ' + ' | '.join(notes) + '. Open whichever of those pages you have not read and '
            'check each against every requirement of the item; cart the cheapest one that passes them all. If none passes, name in your '
            'text the requirement each one breaks and repeat this same call.')

def screen_cheaper_line(ep, ident: str, price: float, cat_key: str, carted: set[str]) -> str:
    """For one new line: the note naming cheaper listings of its kind (opened, or only seen in results); '' if none."""
    kind = mem_label(ep, ident, cat_key).strip().lower()
    owner = mem_owner(ep, kind) if kind else 0
    item = ep.items[owner - 1] if 0 < owner <= len(ep.items) else None
    needs = [n for _, n in ep.live_needs()] if item is None else item.live()
    pointers = {m for _, m in ep.refs}
    stated = [n.split(':', 1) for n in needs if n not in pointers and ':' in n]
    props = [stems(key) for key, _ in stated]
    line = (kind, cat_key, price)
    peer = lambda i: i != ident and i not in carted and mem_can_buy(ep, i) and not ep.is_gone(i)
    opened = sorted((i for i in ep.mem.pages if peer(i) and screen_clean_peer(ep, i, line, needs, props)), key=ep.price_rank)
    unread = screen_unread_peers(ep, peer, line, stated)
    if ep.steps_left() <= ep.reserve() + len(unread) + 2:
        unread = []
    groups = (('opened listings of its kind cost less: {}', opened[:3]),
              ('rows of its kind not read yet cost less: {} (those showing more required values first)', unread))
    parts = [label.format(', '.join(f'{i} ({ep.price_rank(i):g})' for i in ids)) for label, ids in groups if ids]
    return f"{ident} costs {price:g}, yet " + '; '.join(parts) if parts else ''

def screen_clean_peer(ep, i: str, line: tuple[str, str, float], needs: list[str], props: list[Any]) -> bool:
    """An opened listing of the line's kind (kind, category key, price) that costs less and clashes with no requirement."""
    kind, cat_key, price = line
    if (i in ep.basket.dropped or mem_label(ep, i, cat_key).strip().lower() != kind or not ep.price_rank(i) < price
            or judge_apart_needs(ep, i, needs) or judge_against(ep, i, needs)):
        return False
    # a page that leaves a required property out is turned down by the model anyway (Arena r14 b06: holds named
    # cheaper pages without the required depth, the model kept its pick and the paid cheaper fit went unnamed)
    block = fact_block(ep.mem.pages.get(i) or {}) or {}
    return all(any(names_same(prop, stems(k)) for k in block) for prop in props)

def screen_unread_peers(ep, peer: Any, line: tuple[str, str, float], stated: list[list[str]]) -> list[str]:
    """Up to three cheaper rows of the line's kind seen in results but never read."""
    # rows of its kind seen in results but never read, cheaper than the line: the best fit under a lowest-price
    # rule sat unread in an earlier result in 6 of 42 traced failures (Arena r12-r16 b09 vs passing runs)
    # most required values shown in the row first, then price (real r12-r16: among rows seen but never opened,
    # cheapest-first put the paid row first in 9% and in the top 3 in 30% of 114 misses)
    kind, cat_key, price = line
    wanted = [set(plain_words(value)) for _, value in stated if not has_bound(value)]
    found = []
    for i in ep.mem.order:
        if i in ep.mem.pages or not peer(i) or mem_label(ep, i, cat_key).strip().lower() != kind or not ep.price_rank(i) < price:
            continue
        words = set(plain_words(dumps(ep.mem.rows.get(i) or {})))
        found.append((-sum(1 for w in wanted if w and w <= words), ep.price_rank(i), len(found), i))
    return [i for *_, i in sorted(found)[:3]]

def screen_repeated(ep, job: Call) -> str:
    if job.ending or job.name == ep.roles.get('cart') or job.mark() not in ep.recent:
        return ''
    return 'Not run (no step used): the identical call ran just now and nothing has changed since; its result is above.'

def screen_cached(ep, job: Call) -> str:
    if job.name not in ep.reading_tools():
        return ''
    hit = ep.cache.get(job.mark())
    if not hit or hit[0] != ep.version:
        return ''
    _, at, text = hit
    return f'Not run (no step used): nothing has changed since the same call returned this at step {at}:\n' + text[:3000]

def screen_shown(ep, job: Call) -> str:
    """Skip views showing nothing new."""
    target = ep.id_param(job.name)
    if target and job.name in ep.reading_tools() and job.name != ep.roles.get('compare'):
        ident = str(job.args.get(target))
        if mem_fully_shown(ep, job.name, ident):
            return (f'Not run (no step used): since the last change, {job.args[target]} has shown all this view offers. Its record: '
                    + mem_record_text(ep, ident))
    ids = job.listed()
    if not ids:
        return ''
    shown = [mem_fully_shown(ep, job.name, str(i)) for i in ids]
    if all(shown):
        records = '\n'.join(f'{i}: {mem_record_text(ep, str(i))}' for i in ids)
        return ('Not run (no step used): since the last change, these listings have shown all this view offers. Their records:\n'
                + records[:4000])
    if any(shown):
        job.extra += f"\n[Desk] Left out {', '.join(str(i) for i, done in zip(ids, shown) if done)}: already shown in full, nothing changed since."
        job.args = {**job.args, job.many: [i for i, done in zip(ids, shown) if not done]}
    return ''

def screen_page_stop(ep, job: Call) -> str:
    """P7: stop paging a search gone dry."""
    if job.ending or job.name in ep.role_tools('search'):
        return ''
    paged = screen_page_of(ep, job)
    if not paged:
        return ''
    key, base = paged
    empty = 0
    # trailing pages of this query that brought nothing
    for name, args, _, found in reversed(ep.screen.page_log):
        if name != job.name or dumps({k: v for k, v in args.items() if k != key}) != base:
            continue
        if found:
            break
        empty += 1
    if empty < EMPTY_PAGES:
        return ''
    ep.log('page_stop', step=ep.step, tool=job.name, args=clip(job.args, 80), empty=empty)
    return (f"Not run (no step used): past this point the search only pages through rows of the store that do not relate; none of its "
            f"last {empty} pages brought a listing of a requested kind with these words. Use other words: search the values the listing "
            "should show, alone or with its seller's name.")

def screen_dead_filter(ep, job: Call) -> str:
    """Refuse a seller filter that never found rows."""
    for key, value in job.args.items():
        dead = ep.screen.dead.get((job.name, key)) if isinstance(value, str) and value.strip() else None
        if not dead:
            continue
        ep.log('dead_filter', step=ep.step, tool=job.name, key=key)
        word, at = dead
        return (f'Not run (no step used): at step {at} the {key} filter of {job.name} found no rows for {word!r}; a store where it finds '
                f"nothing for one name finds nothing for others either. Search without the {key} filter: the seller's or maker's name as "
                "words along with the product's kind, or the category.")
    return ''

def screen_code_words(ep, job: Call) -> set[str]:
    """Known model codes in a search."""
    said: set[str] = set()
    if job.name == ep.roles.get('search'):
        for value in job.args.values():
            if isinstance(value, str):
                said.update(model_codes(value))
    if not said:
        return said
    index = judge_fit_index(ep)
    works = [w for m in index['marks'].values() for w in m['works'] if not w.startswith('!')]
    known = set(index['known']) | set(model_codes(ep.request)) | codes_named(works, said)
    return said & known

def screen_dead_code(ep, job: Call) -> str:
    """Refuse code searches this store cannot answer."""
    codes = screen_code_words(ep, job) if not ep.screen.code_found and ep.screen.code_misses >= 2 else set()
    if not codes:
        return ''
    ep.log('dead_code', step=ep.step, codes=sorted(codes))
    plan = judge_fit_plan(ep, short=True)
    return (f"Not run (no step used): this store's search does not index model codes ({ep.screen.code_misses} code searches returned "
            "nothing). Search the product's kind with plain values, or list its category and page through it, then read the codes in the "
            "rows' text; Desk matches the codes in rows by itself." + (f' Codes the seen parts name: {plan}' if plan else ''))

def screen_own_words(ep, args: dict[str, Any], skip: str | None) -> set[str]:
    """Search words not naming a requested kind."""
    kind: set[str] = set().union(*(roots(item.name) | roots(item.shelf or '') for item in ep.items))
    cat_key = mem_kind_key(ep)
    if cat_key:
        labels = {str(r.get(cat_key) or '') for r in ep.mem.rows.values()}
        kind.update(*(roots(label) for label in labels if mem_owner(ep, label)))
    text = ' '.join(str(v) for k, v in args.items() if k != skip and isinstance(v, str))
    return roots(re.sub('[^A-Za-z0-9]+', ' ', text)) - kind

def screen_note_page(ep, job: Call, before: set[str]) -> None:
    """Record what a search brought."""
    seen, screen = ep.loop.last_obs, ep.screen
    values = seen.values() if isinstance(seen, dict) else ()
    shown = {dumps(v) for v in values if isinstance(v, (str, int)) and not isinstance(v, bool)}
    paged = screen_page_of(ep, job)
    words = screen_own_words(ep, job.args, paged and paged[0])
    kind = mem_kind_key(ep)
    listed = mem_listings_in(ep, seen)
    # whether a row new to this search is of a requested kind and carries the search's own words
    found = False
    for ident, record in listed:
        if ident in before:
            continue
        row = ep.mem.rows.get(ident) or record
        if kind and not mem_owner(ep, str(row.get(kind) or '')):
            continue
        if not words or words & roots(re.sub('[^A-Za-z0-9]+', ' ', dumps(row))):
            found = True
            break
    screen.page_log = [*screen.page_log, (job.name, dict(job.args), shown, found)][-40:]
    if screen_code_words(ep, job):
        # a search holding a model code read on a fit text or in the request: real r11 and r15 stores never index such
        # codes (after two came back empty, later ones listed rows 5 times in 2,496), r8, r12 and r13 stores do (.67-.94)
        screen.code_found |= bool(listed)
        screen.code_misses += not listed
    fields = (ep.schemas.get(job.name) or {}).get('properties') or {}
    slots = [(job.name, key) for key, value in job.args.items()
             if isinstance(value, str) and value.strip() and SELLER_ARG.search(key) and not (fields.get(key) or {}).get('enum')]
    if listed:
        # a seller filter that found rows is alive again
        screen.dead.update(dict.fromkeys(slots))
    else:
        for slot in slots:
            screen.dead.setdefault(slot, (job.args[slot[1]], ep.step))

def screen_page_of(ep, job: Call) -> tuple[str, str] | None:
    """(paging arg, base) if this pages a search."""
    known = scout_paging_arg(ep, job.name)
    for name, args, offered, _ in reversed(ep.screen.page_log):
        diff = [k for k in set(args) | set(job.args) if args.get(k) != job.args.get(k)] if name == job.name else []
        if len(diff) != 1 or screen_start_value(job.args.get(diff[0])):
            continue
        key, = diff
        if key == known or dumps(job.args[key]) in offered:
            return key, dumps({k: v for k, v in job.args.items() if k != key})
    return None

def screen_resolve_ids(ep, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Map titles given as ids to listing ids."""
    takers = {ep.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None}
    props = (ep.schemas.get(name) or {}).get('properties') or {} if name in takers and ep.mem.id_key else {}
    free = [k for k, v in props.items() if isinstance(v, dict) and {'string', 'array'} & set(type_names(v)) and 'enum' not in v]
    key = free[0] if len(free) == 1 else None
    if key is None or name != ep.roles.get('cart') and (name, key) not in ep.id_args:
        return args, ''
    if key not in args:
        return dict(args), ''
    value = args[key]
    many = isinstance(value, list)
    fixed = [screen_listing_named(ep, v) for v in (value if many else [value])]
    problem = next((p for _, p in fixed if p), '')
    if problem:
        return args, problem
    ids = [v for v, _ in fixed]
    return {**args, key: ids if many else ids[0]}, ''

def screen_listing_named(ep, value: Any) -> tuple[Any, str]:
    mem = ep.mem
    if not isinstance(value, str):
        return value, ''
    text = value.strip()
    if text in mem.rows:
        return text, ''
    if id_like(value, True) or len(text) < 8:
        return value, ''
    start = text.lower()
    hits = [i for i, row in mem.rows.items() if any(k != mem.id_key and isinstance(v, str) and v.strip().lower().startswith(start) for k, v in row.items())]
    if not hits:
        return value, f'{value!r} is not a listing id; pass the id from the result rows (their {mem.id_key} field)'
    if len(hits) > 1:
        return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
    return hits[0], ''

def status_unfound(ep) -> str:
    """A hint for items no row of whose kind turned up after several searches."""
    # Arena b10: items with no candidate after repeated searches (the same words resent); real r12-r16: the paid row was
    # first seen through a query led by the request's values 61% of the time, a kind-only query 15%
    cat_key = mem_kind_key(ep)
    if not cat_key or len(ep.scout.searched) < 2 or ep.checkouts:
        return ''
    labels = [label for label in (mem_label(ep, i, cat_key) for i in ep.mem.rows) if label]
    empty = [str(n) for n in scout_open_items(ep) if not any(mem_belongs(ep, ep.items[n - 1], label) for label in labels)]
    if not empty:
        return ''
    return (f"[no listing of item {', '.join(empty)}'s kind seen yet] search its stated values along with the product noun; try each "
            'number both with and without a space before its unit; or use a noun that the titles of rows in that category carry.')

# TurnLoop.work states
THINK, ASK, ACT, DEFER, IDLE, MISS, CLOCK, HALT = 'think', 'ask', 'act', 'defer', 'idle', 'miss', 'clock', 'halt'

class TurnLoop:
    """The model's turns and the calls they make."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self.chat: list[dict[str, Any]] = []
        self.turns = 0
        self.misses = 0
        self.last_obs: Any = None
        self.pace: dict[str, Any] = {}
        self.pending: tuple[str, list[dict[str, Any]]] = ('', [])

    def status(self) -> str:
        # the step count is not the binding limit on long tasks: real r15 multi-product episodes of the whole KT line
        # ended near step 48 of 100+, and Arena r15 had 24 of 49 failures ending at the clock with steps unused
        ep = self.ep
        steps = ep.steps_left()
        left = min(steps, ep.clock_room())
        unbought = not ep.checkouts
        text = f'[Desk] step {ep.step} of {ep.max_steps} used, {steps} left.'
        if left < steps:
            text += f' At this pace the clock allows only about {left} more steps, so plan for {left}.'
        text += self.urgency(left, steps, unbought)
        lookups = ep.open_pointers() if unbought else []
        blocks = []
        if lookups:
            blocks.append('[resolve first: open the listing each of these points at, then every page that one points on to; the value is '
                          "read on the final page, or set by a pairing rule there against that page's own value] " + ' | '.join(lookups[:3]))
        if ep.leads:
            blocks.append('[review candidates] ' + ', '.join(ep.leads))
        blocks.append(status_unfound(ep))
        blocks.append(judge_fit_plan(ep) if ep.bound and unbought else '')
        if ep.notes:
            blocks.append('[changes so far] ' + ' | '.join(ep.notes[-4:]))
        return text + ''.join('\n' + block for block in blocks if block) + '\n[ledger]\n' + req_ledger(ep)

    def urgency(self, left: int, steps: int, unbought: bool) -> str:
        """The status line's warning for the steps the clock and the step count leave ('' while there is room)."""
        ep = self.ep
        if left <= ep.reserve() + 1:
            return (' FINAL STEPS: once every item has a cart line, check out now. Ending unbought needs your search to have turned up '
                    'an item that no listing fits; running short of steps shows nothing of the sort.')
        if left <= 6 or unbought and left <= len(ep.items) + 5:
            hint = ' Few steps left: set the cart to the chosen lines and check out now.'
            if not self.read_room():
                return hint
            gap = 'An item has no cart line yet' if basket_ids(ep) else 'No line is in the cart yet'
            which = 'the cheapest such row' if ep.cheapest() else 'one whose row shows the wanted values'
            return hint + (f' {gap}: with {ep.attr_tool()}, read the unread listing that looks most promising ({which}); cart it if it meets '
                           'every requirement, then check out.')
        if unbought and left <= 12:
            return ' Only a few steps remain: settle each pick, cart it, and complete the checkout before the steps run out.'
        if unbought and not ep.basket.touched and (ep.step >= ep.max_steps // 2 or steps > left <= ep.step):
            return (' Half the steps or the time are gone and the cart is still empty: add each item\'s fitting listing now, as the store '
                    'may still sell one out.')
        return ''

    def offered(self) -> list[dict[str, Any]]:
        """Tools offered now (narrowed near the end)."""
        ep, roles = self.ep, self.ep.roles
        left, reserve = ep.steps_left(), ep.reserve()
        ends = {roles.get('buy'), roles.get('refuse')} - {None}
        tools = [t for t in ep.tools if t['function']['name'] != roles.get('ask')]
        wide = left > reserve + 3 and (left > len(ep.items) + 5 or ep.checkouts)
        if wide or not ends:
            return tools
        if left <= reserve:
            # Arena r17 b11: at 3 left (1 + the 2-step refusal room) a missing line just read, or the swap a final review
            # named, could not be carted (11 refused cart calls, 9 failed); real r17: first endings refused with <= 5 steps
            # left passed 1/8, so the refusal room yields the cart tool while no ending was tried
            also: tuple[str, ...] = ('cart',) if ep.refusal_room() and left >= 2 else ()
        else:
            also = ('cart', 'cart_view') + (('offer',) if left > reserve + 3 else ())
        keep = ends | set(map(roles.get, also))
        if ep.unoffered():
            keep.add(ep.offer_view())
        if (ep.end_refusals or ep.refused_at >= 0) and left >= 2:
            # a refused ending is answered by a cart change (Arena r13 noask b10: refused decline at step 48 of 50, the
            # store dropping a requirement; only the ending was offered and the model wrote that it had no cart tool)
            keep.update([roles.get('cart'), roles.get('cart_view')] + ([ep.attr_tool()] if left >= 3 else []))
        room = self.read_room()
        # an item without a cart line may need one listing read before it is carted; the narrowed tools left only the
        # ending (Arena r11/r14: the model stalled there and the episode ended unbought). Arena b27-b32: 8 of 17 unbought
        # endings on order tasks came after turns without a call while the narrowed tools held no search
        if room:
            keep.add(ep.attr_tool())
        # Arena r18 noask b02: with one item lineless at 7 of 37 left, search was held for the 2-step refusal room too; the
        # model asked for it 6-18 more times (no step, one model turn each) until the clock ended the episode unbought:
        # 13 tasks, 3 passed. A search, a read and a cart line per missing item still leave the close its steps
        if room and left >= reserve - ep.refusal_room() + 2 + room:
            keep.add(roles.get('search'))
        narrowed = [t for t in tools if t['function']['name'] in keep]
        return narrowed or tools

    def read_room(self) -> int:
        """Missing items one read each can still fill."""
        ep = self.ep
        missing = max(0, len(ep.items) - len(basket_ids(ep)))
        if not missing or not ep.attr_tool():
            return 0
        # reads for items without a line may use the refusal room: without a line for every item no checkout can pass
        # (Arena r16 noask b10: 3 items uncarted at 10 steps left got only the cart and ending tools)
        return missing if ep.steps_left() >= ep.reserve() - ep.refusal_room() + 1 + missing else 0

    def trim(self) -> None:
        if sum(len(dumps(m.get('content'))) for m in self.chat) <= HISTORY_LIMIT:
            return
        # Arena r16 noask b01: loop prompts ran a median 20.9k tokens per call (hayate 3.0k), 7.2 s per call against 4.6 s;
        # 25 of 81 episodes ran into the clock stop. Older results go to stubs; the last few stay whole
        mem = self.ep.mem
        results = [m for m in self.chat if m.get('role') == 'tool']
        for message in results[:max(1, len(results) - 6)]:
            content = message.get('content') or ''
            # a stub is not shortened again (its id list sits past the first 400 characters)
            if len(content) <= 600 or '...(older result shortened;' in content:
                continue
            ids = [i for i in dict.fromkeys(re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)) if i in mem.rows][:20]
            # a shortened page read keeps the values it showed for the requirements (Arena r17 b11: in 19 tasks the model
            # took a checked page for one without the value once its result was cut; 14 failed)
            pages = [mem.pages[i] for i in ids if i in mem.pages] if len(ids) <= 2 else []
            said = judge_evidence(self.ep, pages, False)[:300] if pages else ''
            tail = f'; values it showed: {said}' if said else ''
            message['content'] = f"{content[:400]}...(older result shortened; listings in it: {', '.join(ids)}{tail})"
        # b02 still ran 16-30k prompt tokens a call: Desk's older notes stay as stubs too (the request, the newest turns and
        # the status line each turn carries keep what is current)
        notes = [m for i, m in enumerate(self.chat) if i > 1 and m.get('role') == 'user' and isinstance(m.get('content'), str)]
        for message in notes[:-3]:
            if len(message['content']) > 700:
                message['content'] = message['content'][:500] + '...(older note shortened)'

    def run(self) -> list[dict[str, Any]]:
        ep = self.ep
        if ep.finished:
            return ep.trace
        req_read(ep)
        ep.log('brief', brief=ep.brief)
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': self.opening()}]
        self.work(LOOP_CALLS, SPEND_HARD)
        if not ep.finished:
            self.wrap_up()
        ep.log('end', finished=ep.finished, steps=ep.step, cost=round(ep.llm.cost, 5), calls=ep.llm.calls)
        return ep.trace

    def opening(self) -> str:
        """The first user message: the request, what the store shows first, how to begin, and the status line."""
        ep = self.ep
        first = ep.view.get('opening')
        parts = ["SHOPPER'S REQUEST:\n" + ep.request]
        if first is not None:
            mem_absorb(ep, first)
            parts.append('WHAT THE STORE SHOWS FIRST:\n' + ep.render(first))
        parts.append('Begin with one free-text search per item: the product noun preceded by one or two of its stated values that a fitting '
                     'title would carry (filter by category only with a label the store has shown already). Then read a few listings of '
                     'the right kind in full, so the interviewer learns which properties such listings carry.')
        return '\n\n'.join(parts) + '\n\n' + self.status()

    def wrap_up(self) -> None:
        """After the loop: the desk's own ending; when the store turns it down, ten more turns to act on the reply."""
        ep = self.ep
        if ep.roles.get('refuse') and len(basket_ids(ep)) < len(ep.items):
            # the code fill first: it needs no model call
            basket_proven_fill(ep)
        basket_late_add(ep)
        turned = close_finish(ep, again=True)
        stuck = not turned or ep.finished or ep.store.blocked or ep.llm.out_of_credit or ep.steps_left() <= 1 or ep.past_stop()
        if not stuck:
            listed = scout_cheaper_search(ep)
            self.chat.append({'role': 'user', 'content': '[Desk] Desk tried to end the episode and ' + turned + ('\n' + listed if listed else '')
                              + '\nApply what the store says, then finish.\n' + self.status()})
            self.work(self.turns + 10, SPEND_HARD + 0.01)
        if not ep.finished:
            close_finish(ep)

    def parse_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        listed = [(number, call) for number, call in enumerate(message.get('tool_calls') or []) if isinstance(call, dict)]
        fns = [call['function'] if isinstance(call.get('function'), dict) else {} for _, call in listed]
        raws = [fn.get('arguments') for fn in fns]
        parsed = [parse_object(raw) if isinstance(raw, str) and raw.strip() else raw if isinstance(raw, dict) else {} for raw in raws]
        return [{'id': str(call.get('id') or f'k{self.turns}_{number}'), 'type': 'function',
                 'function': {'name': str(fn.get('name') or ''), 'arguments': dumps(args if isinstance(args, dict) else {})}}
                for (number, call), fn, args in zip(listed, fns, parsed)]

    def work(self, loop_cap: int, spend_cap: float) -> None:
        """State machine: each state's handler does its work and returns an outcome; next_state turns the state and its
        outcome into the next state until HALT."""
        self.pace = {'idle': 0, 'retried': False, 'nudged': False}
        handlers = {THINK: self.on_think, ASK: self.on_ask, ACT: self.on_act, DEFER: self.on_defer, IDLE: self.on_idle,
                    MISS: self.on_miss, CLOCK: self.on_clock}
        state = self.gate(loop_cap, spend_cap)
        while state != HALT:
            state = self.next_state(state, handlers[state](), loop_cap, spend_cap)

    def next_state(self, state: str, outcome: Any, loop_cap: int, spend_cap: float) -> str:
        """THINK's outcome is the model's reply, sorted into a state; any other outcome says whether to go on."""
        if state == THINK:
            return self.read_reply(outcome)
        return self.gate(loop_cap, spend_cap) if outcome else HALT

    def gate(self, loop_cap: int, spend_cap: float) -> str:
        """The state before any model turn: stop, clock stop, interview, or a model turn."""
        ep = self.ep
        if ep.finished or ep.store.blocked or ep.steps_left() <= 0 or self.turns >= loop_cap or not ep.llm.cost < spend_cap:
            return HALT
        if ep.past_stop():
            return CLOCK
        if (ep.elapsed() > CLOCK_STOP - 15 and ep.roles.get('refuse') and len(basket_ids(ep)) < len(ep.items)
                and basket_proven_picks(ep) is not None):
            # a short cart whose lacking items all have proving read listings: the loop gives Desk's ending the time to
            # cart them (Arena 18357: the ending began at 280-290 s and the code fill had no time left)
            ep.log('clock_stop_for_fill', step=ep.step)
            return CLOCK
        return ASK if talk_due(ep) else THINK

    def read_reply(self, message: dict[str, Any] | None) -> str:
        """The state a model reply leads to; its text and calls wait in self.pending for that state's handler."""
        if message is None:
            return MISS
        ep = self.ep
        calls = self.parse_calls(message)
        content = message.get('content')
        self.pending = (content if isinstance(content, str) else dumps(content) if content else '', calls)
        if not calls:
            return IDLE
        committing = {ep.roles.get(role) for role in ('cart', 'buy', 'refuse')} - {None}
        if ep.talk.started or not ep.roles.get('ask') or ep.free_messages() <= 0 or ep.steps_left() <= 8:
            return ACT
        return DEFER if any(c['function']['name'] in committing for c in calls) else ACT

    def on_think(self) -> dict[str, Any] | None:
        self.trim()
        self.turns += 1
        return self.ep.llm.think(self.chat, [LIGHT, SPARE], 3000, self.offered())

    def on_clock(self) -> bool:
        self.ep.log('clock_stop', step=self.ep.step)
        return False

    def on_ask(self) -> bool:
        summary = talk_run(self.ep)
        if summary:
            self.say(summary)
        return True

    def on_miss(self) -> bool:
        self.misses += 1
        llm = self.ep.llm
        return not (llm.out_of_credit or llm.calls >= ALL_CALLS or self.misses > 2)

    def say(self, text: str) -> None:
        self.chat.append({'role': 'user', 'content': text + '\n' + self.status()})

    def on_idle(self) -> bool:
        """A turn without a call; after three in a row, one note naming the open tools, or the end of the run."""
        ep, pace = self.ep, self.pace
        pace['idle'] += 1
        self.chat.append({'role': 'assistant', 'content': self.pending[0] or '(no call)'})
        self.say('Call one of the tools to continue.')
        if pace['idle'] < 3:
            return True
        names = [t['function']['name'] for t in self.offered()]
        narrowed = len(names) < len([t for t in ep.tools if t['function']['name'] != ep.roles.get('ask')])
        if not pace['nudged'] and self.read_room() and narrowed:
            # turns without a call on the last steps' narrowed tools while an item has no cart line (Arena b27-b32: 8
            # of 17 unbought endings on order tasks): once, the open tools are named
            pace.update(nudged=True, idle=0)
            note = (f"[Desk] No tool was called in your last turns. Tools you can use now: {', '.join(names)}. One item still "
                    'lacks a cart line: with them, find and read a fitting listing, cart it and check out; end unbought only '
                    'if your search showed that nothing fits.')
            kind, fields = 'idle_tail_nudge', {'tools': names}
        elif pace['retried'] or ep.mem.pages or ep.step * 2 >= ep.max_steps or ep.steps_left() <= 2:
            return False
        else:
            # turns without a call this early (no page read, under half the steps used) come from a model that wrongly
            # believes the tools are gone: one more round with the tool names before the ending
            pace.update(retried=True, idle=0)
            listed = ', '.join(t['function']['name'] for t in self.offered())
            note = (f'[Desk] No tool was called in your last turns. The store tools are working and open to you now: {listed}. '
                    'Call one this turn; no listing has been read in full yet, so open listings in each category.')
            kind, fields = 'idle_retry', {}
        self.chat.pop()
        self.say(note)
        ep.log(kind, step=ep.step, **fields)
        return True

    def on_defer(self) -> bool:
        text, calls = self.pending
        self.pace['idle'] = 0
        self.chat.append({'role': 'assistant', 'content': text or None, 'tool_calls': calls})
        self.chat += [{'role': 'tool', 'tool_call_id': c['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for c in calls]
        self.say(talk_run(self.ep) or 'The interviewer had nothing to ask.')
        return True

    def on_act(self) -> bool:
        text, calls = self.pending
        self.pace['idle'] = 0
        self.chat.append({'role': 'assistant', 'content': text or None, 'tool_calls': calls})
        self.ep.log('turn', text=text[:300], calls=[c['function'] for c in calls])
        # a non-empty turn always yields one reply per call; the status line rides on the last one
        *done, (last_id, last_said) = self.run_calls(calls)
        self.chat += [{'role': 'tool', 'tool_call_id': cid, 'content': said} for cid, said in done]
        self.chat.append({'role': 'tool', 'tool_call_id': last_id, 'content': last_said + '\n' + self.status()})
        return True

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        """Answer every tool call of one turn."""
        ep = self.ep
        batch = self.join_opens(calls)
        size = len(batch)
        before = set(ep.opened)
        replies = [(call['id'], self.handle(call, size)) for call in batch]
        if replies:
            # a pointer chain the turn's reads opened gets its note under the last reply
            early = chains_early(ep, ep.opened - before)
            if early:
                cid, said = replies.pop()
                replies.append((cid, said + '\n' + early))
        return replies

    def handle(self, call: dict[str, Any], batch: int) -> str:
        if call.get('joined'):
            return 'Desk opened these listings together in one step; their side-by-side result is above.'
        ep = self.ep
        job = screen_screen(ep, call.get('function') or {}, batch)
        if isinstance(job, str):
            return job
        known = set(ep.mem.rows)
        self.last_obs = None
        said = close_policy(ep, job)
        if not said:
            said = self.perform(job)
        if job.searching:
            screen_note_page(ep, job, known)
        if self.last_obs is not None:
            said += scout_zero_retry(ep, job, self.last_obs)
        return said

    def second_read(self, ident: str) -> str:
        """An unread buyable row of ident's kind among the newest results: most required values shown, then cheapest."""
        ep = self.ep
        key = mem_kind_key(ep)
        label = mem_label(ep, ident, key).strip().lower() if key else ''
        owner = mem_owner(ep, mem_label(ep, ident, key)) if key else 0
        needs = ep.items[owner - 1].live() if 0 < owner <= len(ep.items) else []
        cap, carted = ep.numeric_budget(), set(basket_ids(ep))
        fits = lambda i: (i != ident and i not in ep.mem.pages and i not in carted and mem_can_buy(ep, i)
                          and (not key or mem_label(ep, i, key).strip().lower() == label) and (cap is None or (mem_price(ep, i) or 0) <= cap))
        pool = sorted(filter(fits, list(reversed(ep.mem.order))[:24]),
                      key=lambda i: (-scout_values_shown(ep, i, needs), ep.price_rank(i) if ep.cheapest() else 0))
        return pool[0] if pool else ''

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge single opens into one side view."""
        ep = self.ep
        opener, joiner = ep.roles.get('open'), ep.roles.get('compare')
        target, many = ep.id_param(opener), ep.side_param()
        ready = all((opener, joiner, target, many)) and opener != joiner
        if not ready or ep.side_view_bare():
            return calls
        picked = []
        for at, call in enumerate(calls):
            fn = call.get('function') or {}
            text = fn.get('arguments')
            args = parse_object(text) if isinstance(text, str) and fn.get('name') == opener else None
            ident = str(args[target] or '').strip() if args and set(args) == {target} else None
            if ident is not None and ident in ep.mem.rows and ident not in ep.gone:
                picked.append((at, ident))
        picked = picked[:ep.side_room()]
        if len(picked) == 1 and ep.fact_tools.get(joiner) is True and ep.side_room() >= 2:
            partner = self.second_read(picked[0][1])
            if partner:
                ep.log('paired_read', ids=[picked[0][1], partner])
                picked.append((picked[0][0], partner))
        if len(picked) < 2:
            return calls
        ids = [ident for _, ident in picked]
        (lead, _), *others = picked
        others = [(at, ident) for at, ident in others if at != lead]
        # the first open becomes the side view (calls come from parse_calls: id, type, function); the others point at it
        swaps = {at: {'id': calls[at]['id'], 'joined': True} for at, _ in others}
        swaps[lead] = {**calls[lead], 'function': {'name': joiner, 'arguments': dumps({many: ids})}}
        ep.log('joined', ids=list(ids))
        return [swaps.get(at, call) for at, call in enumerate(calls)]

    def perform(self, job: Call) -> str:
        """Run the call and write its answer."""
        ep = self.ep
        reply = ep.store.call(job.name, job.args)
        if reply is None:
            return 'The call failed (network or server). You may try again.'
        seen = reply.get('observation')
        self.last_obs = seen
        self.side_view_seen(job, seen)
        ep.recent = [*ep.recent, job.mark()][-3:]
        rows = ep.mem.rows
        # an argument whose every value is a known listing id takes ids
        ep.id_args |= {(job.name, key) for key, value in job.args.items() if (not isinstance(value, list) or value)
                       and all(isinstance(v, str) and v in rows for v in (value if isinstance(value, list) else [value]))}
        mem_absorb(ep, seen)
        paged = self.learn_view(job, seen)
        self.learn_cart(job, seen)
        closed = bool(reply.get('finished'))
        if ep.mode_ok('refuse', job.name, job.args) and not (closed or ep.finished):
            ep.end_refusals += 1
        if ep.mode_ok('buy', job.name, job.args):
            ep.checkouts += 1
            if not closed:
                # a checkout the store did not close was refused
                ep.refused_at = ep.step
        return self.answer(job, reply, paged)

    def side_view_seen(self, job: Call, seen: Any) -> None:
        """Learn the side view room from its result."""
        ep = self.ep
        ids = job.listed() or []
        count = len(ids)
        if count < 2:
            return
        shown = dumps(seen)
        if any(str(i) in shown for i in ids):
            ep.side_ok = max(ep.side_ok, count)
        else:
            ep.side_bad = min(ep.side_bad or 999, count)
            job.extra += f'\n[Desk] The side-by-side view showed nothing for {count} listings: send at most {ep.side_room()} per call.'

    def learn_view(self, job: Call, seen: Any) -> str:
        """Learn tool roles and search facts from a call."""
        ep = self.ep
        roles, scout, tool, args = ep.roles, ep.scout, job.name, job.args
        if tool == roles.get('compare') and (rows := mem_listings_in(ep, seen)):
            ep.fact_tools[tool] = ep.fact_tools.get(tool, False) or any(fact_block(r) for _, r in rows)
        if tool != roles.get('search') and tool in ep.reading_tools():
            mem_note_keys(ep, tool, seen)
        job.searching = not job.ending and tool not in ep.role_tools('search')
        extra = ''
        if job.searching:
            # paging, next-page values and caps from a result that lists rows
            extra = scout_learn_paging(ep, tool, args, seen)
            scout_note_next(ep, tool, args, seen)
            extra += scout_auto_pages(ep, tool, args, seen)
            scout_learn_cap(ep, tool, args, seen)
            if ep.brief.get('quiet_decline') is False:
                scout.search_log = list(scout.search_log)[-11:] + [(tool, dict(args), seen)]
        fixed = {roles.get(r) for r in ('cart', 'buy', 'refuse', 'ask')}
        if isinstance(seen, dict) and tool not in fixed and ep.id_param(tool):
            found = mem_identify(ep, seen)
            if found:
                self.learn_listing_tool(tool, found, seen, fixed)
        # the search role is read again: the calls above may have settled it
        if tool == roles.get('search') and mem_listings_in(ep, seen):
            extra += scout_learn_rule(ep, tool, args, seen)
            scout.searched = list(scout.searched)[-7:] + [dict(args)]
            extra += scout_first_reads(ep, [seen] + scout.extra_pages)
        return extra

    def learn_listing_tool(self, tool: str, found: str, seen: Any, fixed: set[Any]) -> None:
        """What a view of one listing teaches about the tool that showed it."""
        ep, roles = self.ep, self.ep.roles
        shows = ep.fact_tools.get(tool, False) or bool(fact_block(seen))
        ep.fact_tools[tool] = shows
        if tool == roles.get('open') and not shows:
            # a single-listing view without attributes: another id tool may show them; this one still shows the offer
            skip = fixed | {roles.get('search')}
            better = (t for t in ep.schemas if t != tool and t not in skip and ep.id_param(t) and ep.fact_tools.get(t) is not False)
            roles['open'] = next(better, None) or roles['open']
        if not shows and tool != roles.get('open') and not roles.get('offer'):
            roles['offer'] = tool
        if tool == ep.offer_tool():
            ep.offer_seen[found] = ep.step
        if tool == ep.offer_view():
            ep.page_seen.add(found)

    def learn_cart(self, job: Call, seen: Any) -> None:
        """Cart bookkeeping after a call."""
        ep, basket = self.ep, self.ep.basket
        if job.name == ep.roles.get('cart_view'):
            basket.view, basket.viewed_at = seen, ep.step
        if job.name != ep.roles.get('cart'):
            return
        basket_record(ep, job.name, job.args)
        basket.reply, ep.recent = dumps(seen), []
        first = next((n for n in (whole(v) for v in job.args.values() if not isinstance(v, str)) if n is not None), None)
        named = [v for v in job.args.values() if isinstance(v, str) and v in ep.mem.rows] if first is not None else []
        if named and first > 0:
            basket.dropped -= set(named)
        elif named:
            # a zero count the store's reply does not show as kept takes the line out
            basket.dropped |= {i for i in named if not mentions_row(seen, i)}
        kept = [i for i in basket_ids(ep) if not ep.is_gone(i) and i not in basket.dropped]
        if len(kept) < len(basket.built):
            return
        basket.built, basket.built_needs = kept, {need for _, need in ep.live_needs()}

    def answer(self, job: Call, reply: dict[str, Any], paged: str) -> str:
        """Result text with the notes below it."""
        ep = self.ep
        seen = reply.get('observation')
        body = [ep.render(seen), paged, job.extra]
        if job.searching:
            body += [scout_cheapest_note(ep, [seen] + ep.scout.extra_pages), scout_ignored_cap(ep, job.name, job.args, seen)]
        plain = ''.join(body)
        notes = '\n'.join(filter(None, [judge_evidence(ep, seen), self.zero_count_hint(job, seen), news_store_note(ep, seen),
                                        news_after_refusal(ep, job, reply), news_swap_hint(ep), news_flush_queue(ep), scout_cheaper_search(ep)]))
        if job.name in ep.reading_tools():
            ep.cache[job.mark()] = (ep.version, ep.step, plain)
        return plain + ('\n' + notes if notes else '') + (' [episode finished]' if reply.get('finished') else '')

    def zero_count_hint(self, job: Call, seen: Any) -> str:
        ep = self.ep
        values = list(job.args.values())
        zeroed = job.name == ep.roles.get('cart') and any(whole(v) == 0 for v in values if not isinstance(v, str))
        if zeroed and any(isinstance(v, str) and v in ep.mem.rows and mentions_row(seen, v) for v in values):
            return '[Desk] A count of 0 left that line in the cart: the tool adds the count to the line, and units come off with a negative count.'
        return ''

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Episode(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.loop.run()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                close_finish(episode)
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
