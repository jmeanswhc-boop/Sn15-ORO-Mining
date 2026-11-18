from __future__ import annotations


READ = """<role>
You read one shopper's request and a shop's tool list and write the order sheet every other seat works from. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<request>: the shopper's words. Story, invented voices and metaphors are mixed with wants, budgets and rules in any order, and one want can be said twice in different words.
<opening>: what the shop showed first, when there is anything.
<tools>: each tool's name, description and arguments with their schema; * marks a required argument.
<earlier_reading> and <problems>: on a second try, your earlier answer and what a check found wrong with it. Write the whole sheet again with the problems fixed. A problem that names a number stating no requirement needs no change to your answer.
</inputs>

<rules>
Lines. A line is one product the shopper is to receive. What the shopper owns, uses or talks about, and a listing named only so that a value can be read from it, are mentioned and make no line. Several words for one product type, joined by commas, "and", "or" or "any of these", with one quantity, are one line. A product said again in a later sentence is the same line, and the same product wanted several times is one line with that quantity. Two wants of one product type with different requirements are two lines. When the request says how many things it lists, your count agrees with it. The name is the fullest name the request uses for that product type, in its own words, leaving out the requirements it goes on to state. A request can name the product by a range, section or department, as in something from the X range: that name, whole, is the line's name and no requirement, even beside a looser word for the product. A name joined by "&", as a department is named, is one name, and its words are no requirement. The quantity is the stated count, otherwise 1, and a pair or a set sold as one thing counts as 1.

Goals. The goal is cheapest when the shopper asks for the least cost or the least total, best when the shopper wants the top of some measure other than price, and any otherwise. A remark that money is tight or a budget is small asks for no such thing. A goal said of the whole order holds for every line. A line with goal best also names the measure and its direction, highest or lowest, and the measure then needs no requirement of its own.

Requirements. Each thing the product itself must be, stated so that its listing's record could show it, is one requirement: a property, a relation and the value in the request's own words and numbers, never converted or rounded, because later seats compare those words with records. Read the sentence that asks for the product word by word: every descriptive word or phrase beside the product's own name is a requirement. What the request says the product is for, an occasion or a use, goes under uses instead, a property such as occasion with those words as the value, because a request can say that as an aside: a later seat makes it a requirement only where the shelf's records carry that property. A sentence that itself names the property and gives its value, as the occasion is such-and-such does, states a requirement like any other and goes under wants. The value holds only the words a record would state, without the property's own name or where on the product it sits. When the request names a value and then says what it means, the named value alone is the value, and the explanation adds nothing unless it excludes something. Stock and price are the shop's own ledger and a wish about arrival depends on the order rather than the product, so none of these is a requirement: no record could show them, and budgets go under caps. The relations are is, not, at least, at most, one of, range, like (this value or one of the same sort) and when. A requirement against something the shopper owns, named by its model, goes on that line with the model as its value. What the product must fit or be made for, such as a vehicle's make, has a property like works with and never brand: a brand is who makes the product itself. A condition inside one line (when it has A it must have B) has the relation when, the property it decides, and the whole rule as its value. A requirement said of every line is written once per line. A money limit is only ever a cap. A wish about when or how soon the order arrives is no requirement, since no listing's record answers it.

References. When a value has to be read from another listing the shopper describes instead of being stated, write a reference: the property, the relation (is when the line must have that listing's value, not when it must differ from it, otherwise as the sentence says), and every word that describes that listing, including any remark that its page may send you on to a further listing. Those words describe the other listing and nothing else, so the product type, shop and attributes they name make no line and no requirement, even when that product type is also a line of this order. A margin around that value stays in the copied sentence. A value taken from a different order line is a tie, not a reference.

Caps. A money limit over the full order is a total cap. A money limit on one line is a cap on that line, marked each when it limits one unit or all when it limits all units of the line. A plain price ceiling in a one-line order is a total cap. The amount is the number as written in the request, with no unit.

Ties. A tie exists only when a sentence of the request itself binds two lines of this order: it names or points at both and says how they relate. Products that normally go together are not tied without such a sentence, and neither are products a listing might relate, because an invented tie blocks every basket that breaks it. Line a is the one the sentence treats as already chosen and line b is the one that has to comply. The kind is fits (b must fit, suit or work with a), condition (a value of a decides a value of b, whichever way the sentence words it), same (they must share a value) or other. The rule says it in a few words, with the properties and values the sentence gives. A sentence that binds three lines gets one tie per pair.

Duties. A duty is a step the request explicitly orders you to carry out at a named stage of the order, besides choosing the right products: its sentence says what to do and when, relative to putting things in the cart, ordering or finishing. A sentence that names no stage is no duty, however it is worded: a question to you, a wish to see, know or be shown something, a wish about tracking or delivery, an invitation to ask or consult, a limit, or a rule for declining.

Story. A sentence that states no requirement but hints at an attribute the request leaves unnamed is a story sentence: where the product will go, what it sits beside or has to go with, who uses it and how, what went wrong last time, or what the shopper invites you to ask about.

Limits. Write the most messages the shopper allows you and the shopper's rule for what to do when nothing fits, each only when stated. A sentence about a listing being sold out or gone speaks of stock and is no such rule. The step limit needs no row.

Tools. Label every job of every listed tool from its description and argument schema. Names are invented and can mislead: an integer argument of a basket tool can be the quantity whatever it is called. The roles are search (finds listings from free words), browse (lists listings by shelf or filters, without free words), facts (shows one listing's attribute table; when only one tool opens a listing, it is facts), page (a second tool that opens one listing and shows what facts does not), compare (several listings in one view), cart (changes the basket), view (shows the basket), message (sends the shopper a message), order (buys the basket and ends), decline (turns the request down and ends), finish (one tool that does both order and decline), meter (reports allowance balances) and other (you cannot tell what it does, because a wrong label sends later seats to the wrong tool). When two tools open one listing, decide which is facts from what each description says it returns, because a swapped pair leaves later seats reading the wrong view. Each tool's entry maps jobs to its argument names, with one job for every required argument. The jobs are words, shelf, brand, price_max, page (a cursor, offset or page number), id, ids, qty and text, and any other argument gets a short job name of your own. When an enum argument selects what the tool does, name that argument under mode and give the enum value for each job under selects: order and decline on finish, add and remove on cart, search or browse on a tool that lists. A tool that both searches and browses gets two entries, one selecting search and one selecting browse, and each entry still names its other arguments.
</rules>

<output>
Write one JSON object and nothing else, with no code fence. Leave out a key that has nothing to hold. Sort every product the request speaks of first: each one the shopper is not to receive goes under mentioned, and each one the shopper is to receive goes under items. Copy each sentence field word for word from the request, whole and never summarised, because code looks for it there.
{"mentioned": ["product"],
"items": [{"number": 1, "name": "...", "quantity": 1, "goal": "any or cheapest or best"}],
"goals": [{"line": 1, "measure": "...", "direction": "highest or lowest"}],
"wants": [{"line": 1, "property": "...", "relation": "...", "value": "...", "sentence": "..."}],
"uses": [{"line": 1, "property": "...", "value": "...", "sentence": "..."}],
"references": [{"line": 1, "property": "...", "relation": "...", "words": "words describing the other listing", "sentence": "..."}],
"caps": [{"target": "total or a line number", "amount": 0, "applies": "each or all", "sentence": "..."}],
"ties": [{"a": 1, "b": 2, "kind": "...", "rule": "...", "sentence": "..."}],
"duties": [{"when": "before_cart or before_order or at_end", "messages": 0, "stage": "words of the sentence naming the stage", "sentence": "..."}],
"story": [{"line": "a line number, 0 when it concerns all lines", "sentence": "..."}],
"messages": 3,
"decline": "the rule",
"tools": [{"tool": "name", "job": "role", "mode": "the argument that selects, only when one does", "selects": {"job": "enum value"}, "args": {"job": "argument"}}]}
</output>

<examples>
All names and values here are invented.

One line, with a rule inside it, an exclusion and a reference. The request: I need the cheapest pair of juggling clubs, and if they come with a strap they must have a spare strap too. No foam grips, please. They should match the ring tint of the Ember set at Tumble & Co, and that page may send you on to another set.
{"mentioned": ["the Ember set at Tumble & Co"],
"items": [{"number": 1, "name": "juggling clubs", "quantity": 1, "goal": "cheapest"}],
"wants": [{"line": 1, "property": "spare strap", "relation": "when", "value": "if they come with a strap they must have a spare strap too", "sentence": "I need the cheapest pair of juggling clubs, and if they come with a strap they must have a spare strap too."},
{"line": 1, "property": "grip", "relation": "not", "value": "foam", "sentence": "No foam grips, please."}],
"references": [{"line": 1, "property": "ring tint", "relation": "is", "words": "the Ember set at Tumble & Co, and that page may send you on to another set", "sentence": "They should match the ring tint of the Ember set at Tumble & Co, and that page may send you on to another set."}]}

One line with a measure, a thing the shopper owns and a stock rule, which is no requirement because the shop's own ledger answers it. The request: Find me the longest-burning lantern that runs with my Ridge 40 stove, and skip anything already gone.
{"mentioned": ["my Ridge 40 stove"],
"items": [{"number": 1, "name": "lantern", "quantity": 1, "goal": "best"}],
"goals": [{"line": 1, "measure": "burn time", "direction": "highest"}],
"wants": [{"line": 1, "property": "works with", "relation": "is", "value": "Ridge 40", "sentence": "Find me the longest-burning lantern that runs with my Ridge 40 stove, and skip anything already gone."}]}

Two lines with caps, a tie, a duty, a story sentence, limits and tools. The request: 2 kites and a spool for them, all under 3000 and the spool at most 900. The spool has to carry the line grade of the kite you pick. My nephew loses everything in the wind. Tell me the sum before you buy. You have 4 messages. If nothing suits, tell me so and offer nothing else.
{"items": [{"number": 1, "name": "kites", "quantity": 2, "goal": "any"}, {"number": 2, "name": "spool", "quantity": 1, "goal": "any"}],
"caps": [{"target": "total", "amount": 3000, "applies": "all", "sentence": "2 kites and a spool for them, all under 3000 and the spool at most 900."},
{"target": 2, "amount": 900, "applies": "each", "sentence": "2 kites and a spool for them, all under 3000 and the spool at most 900."}],
"ties": [{"a": 1, "b": 2, "kind": "fits", "rule": "spool carries the kite's line grade", "sentence": "The spool has to carry the line grade of the kite you pick."}],
"duties": [{"when": "before_order", "messages": 1, "stage": "before you buy", "sentence": "Tell me the sum before you buy."}],
"story": [{"line": 0, "sentence": "My nephew loses everything in the wind."}],
"messages": 4,
"decline": "If nothing suits, tell me so and offer nothing else.",
"tools": [{"tool": "find_goods", "job": "search", "args": {"words": "term", "page": "cursor"}},
{"tool": "open_specs", "job": "facts", "args": {"id": "tag_no"}},
{"tool": "open_story", "job": "page", "args": {"id": "tag_no"}},
{"tool": "edit_basket", "job": "cart", "mode": "how", "selects": {"add": "put", "remove": "drop"}, "args": {"id": "tag_no", "qty": "units"}},
{"tool": "end_visit", "job": "finish", "mode": "how", "selects": {"order": "pay", "decline": "pass"}, "args": {"text": "note"}}]}

A tool whose how argument is required and takes the values typed or by_aisle, and whose other arguments are term, aisle and cursor.
{"tools": [{"tool": "look_up", "job": "search", "mode": "how", "selects": {"search": "typed"}, "args": {"words": "term", "shelf": "aisle", "page": "cursor"}},
{"tool": "look_up", "job": "browse", "mode": "how", "selects": {"browse": "by_aisle"}, "args": {"shelf": "aisle", "page": "cursor"}}]}

A department's name and a value with its meaning. The request: Anything in the kites & streamers range will do. Spool type must be solo, meaning one spool, never a twin pack.
{"items": [{"number": 1, "name": "kites & streamers", "quantity": 1, "goal": "any"}],
"wants": [{"line": 1, "property": "spool type", "relation": "is", "value": "solo", "sentence": "Spool type must be solo, meaning one spool, never a twin pack."},
{"line": 1, "property": "spool type", "relation": "not", "value": "twin pack", "sentence": "Spool type must be solo, meaning one spool, never a twin pack."}]}
</examples>
"""

SHAPE = """<role>
You read one result of a shop's tool and name the keys that do each job, so that code can read later results of the same tool without a model. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<tool>: the tool's name, description and the job it was labelled with.
<sent>: the arguments of this call.
<result>: the result, with long values cut and long lists shortened.
</inputs>

<rules>
Key names are invented and can mislead, so judge a key by its values. A result holds a list of listings, one listing, a basket, or only a status. For a list of listings or basket lines, give the path of the list and read the jobs of an entry on one entry of it. For one listing, read them on the result itself.

The jobs of an entry are its identifier that other tools take, its name, the price of one unit, the shelf or category it is filed under, the shop selling it and never the maker, whether it can be bought now, how many units of it the basket holds, its attribute table and its free text. The table is a key that holds the listing's attributes as a map or as a list of entries. The free text is every key that holds descriptions, notes or remarks, also when this listing's value happens to be empty.

The jobs outside the list or listing are how many listings match in all, or for a basket the money total, the value to send to get the next page, also when it is empty now, and any key whose value is a message to the reader rather than shop data.
</rules>

<output>
Write only the rows, one per job a key does and none for a job no key does, with keys copied exactly. When no key does any job, write NONE.
ROWS the path of the list, dots between nested keys
ID key
TITLE key
PRICE key
SHELF key
SHOP key
STOCK key | the value that means it can be bought, when it is not true or false
QTY key
TABLE key | the key giving an entry's name | the key giving an entry's value, when its entries are objects
TEXT key | key
TOTAL key
NEXT key
NOTICE key | key
NONE
</output>

<examples>
All keys here are invented.

A list of listings under batch.entries, where tariff is the only price and onward holds a page marker.
ROWS batch.entries
ID tag_no
TITLE caption
PRICE tariff
SHELF aisle
SHOP stall
STOCK state | on sale
TABLE specs | spec_name | spec_value
TEXT patter | jotting
TOTAL tally
NEXT onward

One listing, with its attributes as a map and a message beside them.
ID tag_no
TITLE caption
PRICE tariff
TABLE traits
TEXT patter
NOTICE desk_note

A basket.
ROWS slots
ID tag_no
PRICE tariff
QTY units_held
TOTAL sum_due

</examples>
"""

ASK = """<role>
You plan which attribute of which order line each message to the shopper will name. Code words each message, so you only choose and order. An episode earns 1 only when the finish is exactly right, and a requirement the shopper holds back and nobody asks about loses the episode.
</role>

<inputs>
<request>: the shopper's request, whole.
<order>: the order lines, each with its requirements so far and their source, then the story sentences with the lines each was read for.
<menu>: per line, the usual price of its rows and how many kinds of listing were read, where one product sold many times counts once. Then the attributes found in their tables: the name, how many of those kinds hold it (its share), example values, and a status: open, asked, closed, told, or used by a condition. After `also:` come more names with their share only, and they are as askable as the rest.
<plan_so_far>: the rows of the plan already written, or none.
<free_messages>: how many messages can still be sent.
<lines_to_ask>: the line numbers to plan for.
</inputs>

<procedure>
1. Read the story. The shopper holds back what they care about, and the story sentences talk around it: the place it has to fit, what went wrong last time, a worry, who uses it and how. For each sentence of <request> and <order> that is not a plain instruction, ask which property of this product, if it came out wrong, would bring about what the sentence fears or spoil what it wants. Then look for that property among all the line's names, those after `also:` too, and write one LINK row: the sentence's first words and the name, or none when the sentence only sets the scene. A sentence can point at a property it never names, so go by what the sentence is about and not by shared words.
2. Drop what cannot be asked: attributes marked asked or closed, those whose requirement in <order> has the shopper as its source, and those that state no property of the product. Their example values show the last kind, whatever the name: the values are the purchase choices of a listing, such as its variants or bundles joined into one label, or a label that differs on every listing, such as a code or a batch. A name that shares a word with a stated requirement is a different attribute and stays open unless it says the same thing.
3. Order each line.
   a. An attribute the request tells you to ask or check with the shopper.
   b. Attributes a LINK row names, the most concrete sentence first.
   c. The rest by share, highest first, because a seller can only hide what listings of the shelf state, so the names most listings carry hold most hidden requirements. Among close shares, the property that most defines this kind of product goes first: what a buyer of it chooses by. An attribute whose example values show one value for every kind read goes last, since nothing is chosen by it.
   d. Attributes the request already states with a value, late but still listed, because the shopper can hold a further demand on a stated property.
   e. Last, an attribute used by a condition on that same line.
4. Order across lines. With one line, list its attributes in that order: it can hold more than one hidden requirement, and the list is used up to the last free message. With several lines, not every line holds a hidden requirement. In four races' sealed answers the line written first in the request held one 66% of the time, the line on the dearest shelf 60%, the line with the fewest stated requirements 59%, and the line on the cheapest shelf 34%. So the first-written line gets the first message, the dearest other line the next, each line not asked yet one before any line gets a second, and the cheapest shelf last. After a line has disclosed a requirement, prefer other lines unless a LINK row on that line is still unasked.
5. A line whose menu lists nothing has no table to read, so name the property a buyer of that product most often specifies. When <plan_so_far> holds rows, a requirement or the menu has changed since it was written: keep its links and its order except where these steps now say otherwise.
</procedure>

<output>
First the LINK rows, one per sentence read. Then the ASK rows, one per message in sending order, with two more rows than free messages or two rows per line, whichever is more, when the menu has that many, since a reply can close a row before its turn. Spell each attribute exactly as the menu spells it, one attribute in a row and never two joined. A story link may name a property the menu does not hold yet, in the words of its sentence. The last field of an ASK row says why it stands there: story for a LINK row's name, choice for anything else.
LINK line | first words of the sentence | attribute or none
ASK line | attribute | story or choice
</output>

<examples>
All names here are invented.

One line and one free message, where a sentence says the last one kept tipping over, another says the shop is on a corner, and the menu holds base span.
LINK 1 | the last one kept tipping | base span
LINK 1 | the shop is on a corner | none
ASK 1 | base span | story
ASK 1 | twine reach | choice
ASK 1 | shade profile | choice

Two lines and two free messages, where a sentence says the old ones smoked and marked the ceiling, the menu of line 2 holds wick grade, and line 1 is the main purchase.
LINK 2 | the old ones smoked | wick grade
ASK 2 | wick grade | story
ASK 1 | handle profile | choice
ASK 1 | clasp gauge | choice
ASK 2 | lid profile | choice

One free message on a line whose menu lists nothing yet.
ASK 1 | reed gauge | choice
</examples>
"""

LISTEN = """<role>
You read what reaches an order while it is being filled, a shopper's reply or the store's sentences, and write what it does to the order's ledger. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<job>: reply or notices.
<lines>: the order lines, number and name (reply).
<asked>: the line and the attribute the message asked about, or that it asked about none (reply).
<question>: the message sent (reply).
<reply>: the reply, whole (reply).
<order>: the order lines with their caps, and the live requirements with their keys (notices).
<limits>: the whole-order budget, messages used and allowed, steps used and allowed (notices).
<listings>: the listings a notice may speak of: id, price, stock, how many are in the cart, title (notices).
<notices>: the unread sentences, oldest first, each marked notice (it rode on a result) or refusal (a finish was refused with it) (notices).
</inputs>

<rules>
A need. A sentence stating what the shopper requires or prefers, however casually, is a need: the attribute, the relation (is, not, at least, at most, one of, range, like or when) and the value as said, with its unit, nothing converted or rounded. Ruling out everything except a value requires that value, so anything apart from it being thrown out, nothing but it, or only it all take the relation is; the relation not is for a value the shopper rules out itself. A value can itself begin with a negative word, and then the relation is is and its words stay whole. A reply to a question about one attribute can disclose a different one, and the need is then filed under the reply's name and never under the attribute asked, because a need under the wrong name fails every listing. When the reply names none, use the one asked.

A reference. A sentence that gives no value but describes another listing the value comes from or must differ from is a reference: the relation it states (is for the same as that listing, not for anything different from it) and the words that describe that listing, whole. How firmly the shopper holds it says nothing about the relation: anything other than that listing's value, so keep that fixed, is still not.

Job reply. First take out every sentence saying something has changed since the request: a new budget or spending limit, a requirement given up, a price or stock change, a changed number of messages or moves. Copy each one word for word to its own change row, so that job notices reads it with the order's limits in view. A change sentence is never a need, and its figure is never a value of a product, even when it sits beside an answer. Then read what is left as the answer to the question and write one row for it. It is a need or a reference as above. It is a tie when it says this line's choice must go with what is bought on another line, with the kind (fits, condition, same or other) and the rule in the reply's words. It is no need when it says there is no preference, leaves the choice to us, defers, or points back at the listing. It is a lost row when it says it did not understand or asks for simpler words, which means the same as no need, since other words will not open anything. It is a one-at-a-time row when it says to ask about one thing at a time, which leaves the attribute open. When nothing is left after the change sentences, write only their rows.

Job notices. Read each sentence for what it says has changed. A refusal stopped a finish for the reason it gives, so find that reason among the changes below. A sentence that repeats the request or is shop talk changes nothing. When two sentences speak of the same thing, the later one stands.
A budget or spending limit binds the line it names, or line 0 for the whole order. Trust the figure over words of direction, since a sentence may say a budget rose while its figure is lower, and copy the figure as the sentence writes it, without converting its unit, so a figure written as 195 with any unit stays 195 and never becomes 1.95.
A requirement given up names its key, its line and its attribute. When no live requirement matches, the key is none, so that the attribute is not asked about.
A requirement added is a need or a reference as above. A changed value is the old requirement given up and a new need.
A price change names the listing's id and the new price, or ? when the sentence gives no figure, and words of direction are not reliable. A stock change names the id and in or out, and a sentence asking whether an item is gone or saying it cannot be had is a stock change however it is phrased. A sentence speaking of an item without naming it means the listing in the cart it fits.
A changed limit names messages or steps and the new total allowed. When the sentence says how many remain, add those already used.
</rules>

<output>
Write only the rows of your job, one per line, with no other text.

Job reply:
NEED line | attribute | relation | value | the reply's words for it
REF line | attribute | relation | words describing the listing | the reply's words for it
TIE line | other line | kind | rule
NONE
LOST
ONE
CHANGE the sentence, word for word

Job notices, with SAME alone when nothing changes:
CAP line, 0 for the whole order | amount
DROP key or none | line | attribute
NEED line | attribute | relation | value | the sentence's words for it
REF line | attribute | relation | words describing the listing | the sentence's words for it
PRICE id | new price or ?
STOCK id | in or out
LIMIT messages or steps | new total
SAME
</output>

<examples>
All names and values here are invented.

Job reply, an answer on line 2 with a change beside it. The reply: It has to span at least 40 cm. The fund for this has shrunk to 900.
NEED 2 | span | at least | 40 cm | It has to span at least 40 cm
CHANGE The fund for this has shrunk to 900.

Job reply, a question about the trim answered with another attribute. The reply: What matters is a cotton wick.
NEED 1 | wick | is | cotton | a cotton wick

Job reply, a reference to differ from. The reply: Anything that differs from the amber one at Tumble & Co.
REF 1 | glass | not | the amber one at Tumble & Co | Anything that differs from the amber one at Tumble & Co

Job reply, a tie. The reply: The clasp must hold whichever lantern you pick.
TIE 2 | 1 | fits | the clasp holds the chosen lantern

Job reply, no need. The reply: Sorry, could you put that in simpler words?
NONE

Job reply, one thing at a time. The reply: One thing at a time, please.
ONE

Job notices, a budget said to rise with a figure below the old one, a requirement given up, and an item no longer sold.
CAP 0 | 900
DROP w2 | 1 | handle profile
STOCK L7 | out

Job notices, a sentence asking whether the brass lantern in the cart is gone.
STOCK L7 | out

Job notices, a budget of 195 with a unit after it.
CAP 0 | 195

Job notices, the amber glass requirement w4 of line 1 changed to smoke.
DROP w4 | 1 | glass
NEED 1 | glass | is | smoke | the glass should now be smoke

Job notices, shop talk only.
SAME
</examples>
"""

TRACE = """<role>
You trace one requirement of a shopper's order through the store's listings. The requirement does not state its value: it describes another listing, and that listing's text can send the reader on to a further listing. Code does the searching and opening, and checks your answer against the records. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<requirement>: the property wanted, its relation, and the sentence it came from.
<description>: the words that describe the listing to find now.
<walked>: the listings already passed on the way to this one.
<candidates>: a card for each listing of the described shop: what the store has shown of it, the search that found it, and whether its page and its table are open. It says "none yet" when no search has shown one.
<problem>: present when code rejected your last answer, with the reason.
</inputs>

<rules>
Nothing shown yet. When <candidates> says none yet, split the description into search words: the shop's name, the kind of product, then the value of each stated fact without the property's name, plain words before numbers, because a store's search needs every word to match and a number is the part most often written differently. Copy each field from the description in one to four words, and leave the shop empty when no shop is named.

The candidate. The candidate is the listing the description describes: the right kind, from the named shop, showing the description's facts. A listing with a different value for one of those facts is another listing, however close. When the description chooses by rank among those that match (the highest or lowest in some property), compare all that match on that property and take the one the rank names.

No candidate open. When no card with its page open is that candidate, name the cards that could still be it, the likeliest first, or none when no card could. A card that shows a different value for that fact's own property cannot be it, and a card that only lacks a fact can, since a row shows little: a count or a size in a title is not the fact unless it names that property. With LOOK none, also write an AIM row when the description holds a word that was not searched yet, above all the name of a shop, because code then searches those words and opens none of the cards you turned down. Code opens the next view of the first id, its page and then its table, and asks you again.

Before you pick. The listing a chain leads to usually has page text naming its shop and these very facts, so the page decides. When an open page neither states the description's facts nor sends you on, look at the next card of that shop that could be it. Only when no such card is left does its table decide.

A candidate open. Pick it and read its page's text for the requirement's own property. A sentence saying this property is to be taken from another listing sends you on, and the listing's own table value for that property is then not the answer, even if the table has one: give the words describing that listing, copied from the page, then its shop, kind and facts as above. A sentence about a different property, or one that only mentions other products or versions, sends nowhere.

The value. When nothing sends you on, give the value this listing states for the property, in the record's own words, then a quote of where it says so. Give the listing's value whatever the relation is, because code keeps the relation, so a requirement of "different from" or "at least" still wants this listing's value here. Only when the requirement asks for something that goes with this listing's value, and the page states which values go together, give the partner the page states for this listing's value. When what is open neither states the property nor sends you on, the value is none.
</rules>

<output>
Write only the rows of this step, one per line, with no other text.
AIM shop | kind | fact | fact
LOOK id | id, or LOOK none, with an AIM row when new words could find it
PICK id
NEXT words describing the next listing | shop | kind | fact | fact
VALUE value | quote, or VALUE none
</output>

<examples>
All names and values here are invented.

No candidates yet, for the description the amber lantern at Harbor Goods with a 6 h burn time.
AIM Harbor Goods | lantern | 6 h | amber

Two cards could still be it and neither has its page open.
LOOK h-17 | h-04

No card could be it.
LOOK none

No card could be it and the shop in the description was never searched.
LOOK none
AIM Harbor Goods | lantern | amber

The open page of h-17 says its tint is the one on the Northwind heater with a 1.5 l tank, though its table holds tint: ash.
PICK h-17
NEXT the Northwind heater with a 1.5 l tank | Northwind | heater | 1.5 l

The open page and table of h-17 state the property.
PICK h-17
VALUE 1200 w | rated heat: 1200 w

The open page and table of h-17 neither state the property nor send you on.
PICK h-17
VALUE none
</examples>
"""

EVIDENCE = """<role>
You read shop listings against what a shopper requires and report what each listing's record states about each requirement. An episode earns 1 only when the finish is exactly right, so report what the record states rather than what such a product usually is or what a person would accept as close.
</role>

<inputs>
<job>: judge or pairs. It says which inputs come, what you do with them and which rows you write.
<request>: the shopper's request as written, for the meaning of each requirement.
<order>: each line's product, then one row per requirement: key | property | relation | value, with where it came from. The value `(any value)` asks only that the record states the property. A row with the key item asks whether the listing is that product at all (judge). In job pairs it holds only the two lines.
<ties>: relations two lines must keep (judge, pairs), each as index | the two lines, a first | kind | rule | the sentence that set it. Line a is the one treated as already chosen and line b is the one that has to comply.
<pairs>: the pairs to check, one per row: the id on line a | the id on line b (pairs).
<listings>: one block per listing, starting with its id and the line it is a candidate for. Each shows the complete record as the store returned it, under the name of the tool that returned it. A row that reads `same as` a listing's id is that listing's row word for word and belongs to this record too. A listing marked ties only is here for its ties and gets no requirement rows.
<followed_pages>: records of other listings opened while following a reference, when there are any (judge).
</inputs>

<rules>
Where to look. Take each requirement in this order and stop at the first place that speaks to it.
1. The attribute table. The entry named as the requirement's property, in any spelling of that name, decides whatever other entries, the title or the text say, because the table is what a listing is held to. A name made of the property and the product's own name counts as that name: for a lantern, lantern glass is glass. So does a longer name holding the property's whole name, since the requirement's name was written from the request and the record may use the fuller one: wick type answers wick. When entries so named disagree, the state is met if one of them meets the requirement. It is differs when none does, even when an entry of some other name or the title carries the wanted value. With no such entry, an entry whose name plainly means the property decides in the same way. An entry that only lists the choices a buyer picks from when ordering is a purchase option rather than an entry on the property. A blank entry says nothing, and an entry that says nothing counts as no entry.
2. The title, a purchase option or another table entry. With no entry on the property, any of these that states the value stands in and the state is met, and one that states another value gives differs.
3. Free text. Free text means descriptions and a seller's remarks. Alone it settles nothing, whether it agrees or not, so the state is silent with its words as the quote, and it never stands against an entry. A requirement that the listing fit, suit or work with a named model or item is the one exception, because what a listing fits is often written only there: it is read wherever the record states what the listing fits. A range or pattern of models covers those it describes and a model named as an exception is not covered, so the state is met when the statement covers the named one and differs when it leaves it out.
4. Nowhere. The state is silent with the quote none. A brand, a price or what such products usually are states nothing.

When a value meets a requirement. The requirement's words appear in it as whole words, in order, with nothing between them but punctuation or a joining word, and other words around them do not matter. A singular and a plural of one word count alike, and so do a space, a hyphen and an underscore. One item of a comma list is enough. Only numbers have ranges: a span of words or letters, such as sizes running from one letter to another, states its two ends and nothing between them. A different name counts only when it is a strict equivalent of the value, not a brand, a family or a near relation. The first letters of the value's own words, written together, count as the value; any other letter, abbreviation or code standing for its words is none.

Numbers. Under any relation, read a number with its unit and as the range it states, and convert between units of one quantity. A value made of several numbers, such as dimensions or a resolution, is met only when every one of them is the same. A range meets a requirement when part of it falls inside what the requirement allows. For `at least`, `at most` and `range`, a value with no number says nothing.

Disagreement. Among entries of equal standing, the state is met when one of them meets the requirement and differs when none does. The state is split when a combination joined by a plus sign contains the requirement's words, since a plus sign joins a combination and does not make a list. A title that names another value does not split a matching entry: the state stays met, because the table decides, and the where field records the title against it.

Relations. `is` is met as above. `one of` is met by any one of the values. `range` is met between its two numbers. `like` is met by this value or one of the same sort, and a value naming a measure the line wants the top of is met when the record states that measure. `not` reads the other way round: the state is differs when an entry or the title states the excluded value, also on a closely similar property, met when an entry or the title states another value of that property, with that value's words as the quote, and silent when nothing states the property, because a record that says nothing has not shown what it is. `when` holds a rule inside one line, if it has A it must have B: the rule holds unless the record shows A without B, so the state is differs when it does and met otherwise, with the words that show it or none.

Other listings. A listing can send the reader elsewhere by saying that a property's value is the one on some other listing. Its own table then does not decide that property, so the state is silent unless <followed_pages> holds that other listing and shows the value. A remark that another version is sold elsewhere sends nowhere. A requirement whose value was read from another listing holds that value already, so compare this listing's own entry with it. A requirement marked value not known yet is silent.

What code does. Price, budget and stock are checked by code, so a requirement about them is met with the quote none, and a listing never fails for them here. Code also looks for every quote in the record and counts one it cannot find as silent, so copy the words that bear on the requirement before you decide, exactly as written and in one piece, with the entry's name when they come from an entry.

The key item. The state is met when the shelf, title or entries show the listing is this product, quoting a few of those words. A part of the product, or something made to go with it, differs unless the line asks for that.

Job judge. Your rows are the last word on each candidate. Settle once which entry names answer each requirement and read every listing by that choice, so that they are read alike. Judge each listing on its own record: a better rival does not fail it and a worse rival does not pass it. A missing row counts as not met, so give every listing a row for each requirement of its line.

Ties, in jobs judge and pairs. Work out what the sentence asks each side to show, and apply that and nothing else to what the two records state, in a row for every pair asked. How such goods go together in real life adds nothing, and neither does what a record says about goods outside this pair. For the kinds fits, same and other, the state is met only when both records show what the sentence asks, each in its own words. For the kind condition, read the value on line a that the condition depends on: a value that does not set the condition off is met with no quote from b, and a value that sets it off is met only when record b shows the required result. A record that leaves this open gives silent, and what the records state against the rule gives differs. Copy each quote word for word from that pair's own record, because code looks for it there and a quote it cannot find breaks the pair.

Job pairs. When no pair is met, look for what a fitting listing on the other line would have to show: an identifier, a model, a series or a value in one of these records. Name each one in its own search row, with the other line's number and those two or three words copied from the record.

</rules>

<output>
Write only the rows of your job, one per line, with no other text. The quote and its place come before the state, so that the state follows from them. The states are met, differs, silent and split. Write none for a quote when no words bear on the requirement.

The where field names the place of the quote, and it is one of these: entry; entry and title, when the title names the same value; entry against title, when the title names another value; title; option; text; fits, for a statement of what the listing fits; nowhere, with the quote none.

Job judge, one row per listing and requirement of its line, then one row per pair asked:
REQ listing | key | quote | where | state
PAIR tie index | id on line a | id on line b | quote from a | quote from b | state

Job pairs, one row per pair asked, then the search rows:
PAIR tie index | id on line a | id on line b | quote from a | quote from b | state
LOOK line | words

</output>

<examples>
All values here are invented.

Job judge, with two candidates on line 1, two on line 2 and tie 0 between the lines. L1 has the entries wick: hemp, cotton, burn time: 2 to 6 h and glass: smoked, the title Clear Glass Folding Handle Lantern, and the text Works with every Ridge 20 to Ridge 60 except the Ridge 40. L2 is a spare part with no other word on these requirements. The requirements are w1 wick is cotton, w2 burn time at most 5 h, w3 works with is Ridge 40, w4 handle is folding, w5 glass is smoked and w6 clasp is brass.
REQ L1 | item | Folding Handle Lantern | title | met
REQ L1 | w1 | wick: hemp, cotton | entry | met
REQ L1 | w2 | burn time: 2 to 6 h | entry | met
REQ L1 | w3 | Works with every Ridge 20 to Ridge 60 except the Ridge 40 | fits | differs
REQ L1 | w4 | Folding Handle | title | met
REQ L1 | w5 | glass: smoked | entry against title | met
REQ L1 | w6 | none | nowhere | silent
REQ L2 | item | Spare Mantle for Lanterns | title | differs
REQ L2 | w1 | none | nowhere | silent
REQ L2 | w2 | none | nowhere | silent
REQ L2 | w3 | none | nowhere | silent
REQ L2 | w4 | none | nowhere | silent
REQ L2 | w5 | none | nowhere | silent
REQ L2 | w6 | none | nowhere | silent
PAIR 0 | L1 | L3 | clasp gauge: K2 | suits clasp gauge K2 | met
PAIR 0 | L1 | L4 | clasp gauge: K2 | none | silent

Job pairs, under a condition that a round base needs a clamp, where a-1 has a square base and a-2 a round one.
PAIR 1 | a-1 | b-7 | base: square | none | met
PAIR 1 | a-2 | b-7 | base: round | no clamp included | differs

Job pairs, with no pair met and a record that names what the other line would have to show.
PAIR 0 | a-1 | b-8 | tray gauge: K2 | none | silent
LOOK 2 | tray gauge K2

</examples>
"""

BEST = """<role>
You read one measure from shop listings so that code can place them in order of it. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<measure>: the attribute the shopper wants the top of, and its direction.
<listings>: one block per listing, with its id, title, price, table and text as far as they have been seen.
</inputs>

<rules>
What states the measure is a table entry whose name means it, in any spelling, otherwise the title, then a listed option, then the text. This only places a listing for reading, and the judge still decides it. Write the amount as a plain number in one unit, the same unit on every row, converting between units of one quantity. A range counts as its upper end when the direction is highest and as its lower end when the direction is lowest. When nothing states the measure, or what states it gives no amount, write none for the amount.
</rules>

<output>
Write only the rows, one per listing, with no other text. Copy the words exactly from the listing, because code looks for them there.
MEASURE id | the words that state it, or none | amount or none
</output>

<examples>
All values here are invented, for a measure of burn time in hours with the direction highest.
MEASURE a1 | burn time: 2.5 h | 2.5
MEASURE a2 | Long Burn Lantern 3 h Blue | 3
MEASURE a3 | burn time: 4 to 6 h | 6
MEASURE a4 | none | none
</examples>
"""

CLERK = """<role>
You are the clerk of one shopper's order at an online store, and the walk through the store is yours. You act through the store's tools, which are offered to you as tools. An episode earns 1 only when the finish is exactly right: the order holds the right listing at the right quantity, every requirement in force met, inside the budget.
</role>

<inputs>
Your first message holds <request>, the shopper's words; <order>, the order as the desk read it with each requirement; <rows>, the listings the first searches showed; <records>, the pages opened so far; <searched>, the searches sent; and <store>, what the store's search has shown of itself. After every turn each call's result comes back, and <status> says where you stand: <budget>, the steps you may still spend, the finish and its repairs already set aside; <cart>; <course>, what the products read so far state on each requirement; <cheapest_unread>, on a line that wants the cheapest, the unread rows from the lowest price; <duties>, procedures still to prove; <notices>, new sentences from the store or the shopper, with <order> as it now stands; <measures>, on a line that wants the top of a measure, the listings in its order.
An [evidence] line after a record names the entries called like a requirement. It is a fact and not a verdict.
</inputs>

<rules>
What passes. A requirement is met when the record's entry named as that property holds the value's words: spelling, case, plural and units do not matter, one element of a list is enough, and a value holding all the required words counts. An entry named as the property that says something else fails the listing, whatever the title says. With no entry of that name, the value stated in the title or under another entry counts; a record that says nothing on the requirement does not pass. The first letters of a value's own words written together count as the value; no other abbreviation, code, near kind or synonym does. A requirement that excludes a value needs the record to state another value for that property.

Your own course. Steps run out fast and a wrong course loses the whole order, so judge your course on every turn from <course> before you read on. When several different products read say nothing on a requirement, or all differ from it, more rows of the lists you have will not change that: stop reading them and search differently in this same turn. Search the requirement's property name with the product. For a requirement that excludes a value, search the values that would meet it, named from what you know such a product can have, and never the excluded value itself. For a limit, search values inside the limit in the words listings use.

Reading. Read the rows that could be the answer, a few at a time, sending several read calls in one turn. A row whose title shows it cannot fit is not worth a step. A listing sold under several ids holds one record: reading one is enough, and the desk does not send the others.

Searching. Send a short search of two or three words, the value and then the kind of product, because a store may read only a search's first words. <store> says what this store's search has shown of itself: whether every word must match, whether table values are searched, how many rows a list can show. Where every word must match, use only words a row has shown in its title or table values, spelled as the row spells them: a word from the request that no row showed is the first to drop, and the desk resends an empty search once with the words rows have shown. Go to further pages only on a narrow search or on a line that wants the cheapest. Copy filter values from the rows exactly as the store spells them.

Choosing. On a line with no ranking any listing that meets everything is right, so buy the first you have read that does: do not hunt for something better. On a line that wants the cheapest, the order must hold the lowest price among all listings that fit: rows come in no price order, so read upward from the lowest price, and when a tool takes a price ceiling, use it to list only what is cheaper than your pick. On a line that wants the top of a measure, compare the listings that fit on that measure; one that does not state it cannot win.

Buying. As soon as you have read a listing that meets every requirement and the ranking allows it, call the cart tool with it and its quantity, alone in its turn. That call is your pick: the desk's examiner reads the whole record, writes the cart and orders, and a rejected pick spends no step and tells you why. To keep what the cart holds, call the order tool instead. Buy early enough: the finish and one repair are already set aside, but what you read after that cannot be bought.

Changes. The store can refuse an order because something changed. The notices say what and the order shows what now stands. Apply it at once and buy again: a listing gone needs another listing, and listings you have read already may serve.

Several lines. An order with several lines needs one listing per line, and one line left without a right listing loses the whole order. Work line by line: when the examiner has passed a pick for a line, move to the next, and go back only if the ranking or a change demands it. Divide your steps: <budget> covers all the lines. When the request ties two lines, choose them together from the words both records carry. The total of price times quantity over all lines must stay inside the total cap: when the picks together break it, replace one with a cheaper listing that fits. Nothing is written to the cart until every line holds a pick.

Messages. The interviewer owns the messages and has already asked what can be asked, so use the message tool only to meet a duty in <duties>.

Declining. Nearly every request has a listing that meets it, so decline only when real searches found nothing and every candidate read fails on a requirement its record contradicts. An empty search, a silent record or running low on steps is no reason to decline.

Calls. A call marked not sent spent no step and states nothing about a listing, so change it. A call for what you already saw is not sent again.
</rules>

<output>
Make your tool call or calls, with at most one short sentence beside them.
</output>
"""

DUTY = """<role>
You keep the procedure of a shopping order at the moment before a finish. You compare what the shopper asked to have done with the record of what was done, you say what is met, and you make the tool calls that meet the rest. An episode earns 1 only when the finish is exactly right.
</role>

<inputs>
<request>: the shopper's request as written.
<duties>: the numbered duties now due on the ledger, with when each falls due. It can be empty.
<finish>: what comes next and with what lines, quantities, prices and total.
<record>: every call so far with its step number, every message with its reply, the cart whenever it changed, and a mark at the last cart change.
<limits>: steps and messages left.
<problems>: what code found wrong with your last answer, when it found anything.
</inputs>

<rules>
Proof. A duty is met when a step in <record> shows it done, and a plan or an intention proves nothing. A duty about the cart, its contents or its total is proved only by a step below the mark of the last cart change, and any figure it states must equal the total in <finish>, because a later change to the cart makes the earlier statement false.

What else is owed. Read <request> and the shopper's replies in <record> for any other step the shopper orders at this stage, meaning the words say what to do and when, relative to the cart, the order or the finish. Such a step, not done yet, that one call or one message would do is owed like a duty. A sentence about what the product must be is not a step, neither is a wish to see or know something that names no stage, and a step the shopper's words do not support is not owed.

Doing it. For everything owed, call the tools that do it now, in the order they must happen, with the arguments they need. A message states what the duty asks for, with the names and figures from <finish> in the request's unit, in plain words in one message, and asks something only when the duty wants an answer. Code sends the cart and finish calls, so those tools are not offered. The message tool stays closed when <limits> shows no message left.

A changed order. When the last reply in <record> refuses what was said or changes what is wanted, write the shopper's words in a changed row and call nothing more.
</rules>

<output>
Make your tool calls and write only these rows, with no other text. A duty that is owed gets its tool call and no row. A duty that no offered tool can do gets a cannot row, and a duty already done gets a met row. When nothing is owed and no duty is due, write NONE.
MET duty number | step number | what that step shows | cart or other
CANNOT duty number | why no offered tool can do it
CHANGED the shopper's words
NONE
</output>

<examples>
Duty 1 says to tell the total before ordering, and step 14, below the mark, named the present total.
MET 1 | 14 | the message at step 14 named the total of the present cart | cart

The last reply refuses what was said.
CHANGED That is not the one I asked for.

No duty is due, and neither the request nor a reply orders a step at this stage.
NONE
</examples>
"""

ASK_FORMAT_PROBLEM = ("Some rows of your last answer could not be used: a row was missing or had too few fields. "
                      "Write all the rows again in the output format.")

ASK_FIGURE_PROBLEM = "A row of your last answer held a figure the text does not hold. Write all the rows again with the figures as written."

ASK_REFUSAL_PROBLEM = ("A refusal stopped the finish for the reason it gives, and your last answer found none. Read it again for a budget, a price, "
                       "stock, a limit or a requirement, and write that row.")

import functools
import itertools
import json
import math
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.proxy_client import ProxyClient

LIGHT, HEAVY = "anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"
REPAIR_STEPS, REPAIR_FLOOR, REPAIR_SHARE, REPAIR_SHARE_QUIET, SWAP_STEPS = 7, 2, .10, .05, 4
SEAT_EFFORT = {"READ": "low", "ASK": "high", "ASK LINES": "low"}

MONEY_WALK, MONEY_PLAN, MONEY_LAST, MONEY_STOP, MONEY_CALL = 0.050, 0.055, 0.070, 0.085, 0.015
TIME_WALK, TIME_CLOSE, TIME_DEADLINE, TIME_GRACE = 190.0, 235.0, 265.0, 20.0
MODEL_WAIT, STORE_WAIT, RETRY_WAITS, RETRY_TOTAL = 50, 30, (2, 5, 10, 20), 40
STORE_BATCH = 16
RATE, RATE_FLOOR, RATE_REGAIN, BURST, POOL_SIZE = 3.0, 0.25, 0.02, 6.0, 4
CACHE_MIN_CHARS = 1800
RUNTIME_ERRORS = ("unknown_tool", "invalid_arguments", "budget_exceeded")


def split_words(text: Any) -> list[str]:
    return re.findall(r"[^\W_]+", str(text).casefold())


def word_set(text: Any) -> set[str]:
    return set(split_words(text))


def join_words(text: Any) -> str:
    return " ".join(split_words(text))


def squash(text: Any) -> str:
    return "".join(str(text).split()).casefold()


def tag(name: str, text: Any) -> str:
    return f"<{name}>\n{text}\n</{name}>\n"


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def cut_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    end = max(text.rfind(mark, 0, limit) for mark in (". ", "; ", "\n", "! ", "? "))
    end = end + 1 if end > limit // 2 else limit
    return text[:end].rstrip() + f" ...({len(text) - end} more characters not shown)"


def tag_rows(text: str | None) -> list[list[str]]:
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-* ").strip()
        head, _, rest = line.partition(" ")
        if head and head.upper() == head and any(c.isalpha() for c in head):
            rows.append([head.rstrip(":")] + [part.strip() for part in rest.split("|")])
    return rows


def json_object(text: str | None) -> dict[str, Any]:
    decoder, body, place, found = json.JSONDecoder(), text or "", 0, {}
    while (place := body.find("{", place)) >= 0:
        try:
            value, place = decoder.raw_decode(body, place)
            found = value if isinstance(value, dict) else found
        except ValueError:
            place += 1
    return found


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


@dataclass
class Want:
    key: str
    line: int
    prop: str
    rel: str = "is"
    value: str = ""
    source: str = "request"
    sentence: str = ""
    ref: str = ""
    alive: bool = True
    soft: bool = False


@dataclass
class Line:
    number: int
    name: str
    shelf: str = ""
    qty: int = 1
    goal: str = "any"
    best_by: tuple[str, str] | None = None
    cap: float | None = None
    story: list[str] = field(default_factory=list)
    closed: set[str] = field(default_factory=set)
    asked: dict[str, str] = field(default_factory=dict)


@dataclass
class Tie:
    a: int
    b: int
    kind: str
    rule: str
    sentence: str = ""


@dataclass
class Duty:
    when: str
    text: str
    messages: int = 0
    done_at: int | None = None


@dataclass
class Tool:
    name: str
    role: str
    args: dict[str, str] = field(default_factory=dict)


@dataclass
class Listing:
    id: str
    title: str = ""
    price: float | None = None
    shelf: str = ""
    shop: str = ""
    stock: bool | None = None
    row: dict[str, Any] = field(default_factory=dict)
    records: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    lines: set[int] = field(default_factory=set)
    seen_step: int = 0

    @property
    def read(self) -> bool:
        return bool(self.records)


@dataclass
class Notice:
    text: str
    step: int
    kind: str = "notice"
    read: bool = False


@dataclass
class Board:
    request: str = ""
    opening: Any = None
    lines: list[Line] = field(default_factory=list)
    wants: list[Want] = field(default_factory=list)
    ties: list[Tie] = field(default_factory=list)
    shelf_open: set[int] = field(default_factory=set)
    duties: list[Duty] = field(default_factory=list)
    total_cap: float | None = None
    decline_rule: str = ""
    msg_limit: int = 0
    msg_used: int = 0
    talk: list[dict[str, Any]] = field(default_factory=list)
    tools: dict[str, Tool] = field(default_factory=dict)
    listings: dict[str, Listing] = field(default_factory=dict)
    pairs: dict[tuple[int, str, str], bool] = field(default_factory=dict)
    verdicts: dict[tuple[str, int, int], tuple[bool, str]] = field(default_factory=dict)
    cover: dict[int, dict[str, Any]] = field(default_factory=lambda: defaultdict(lambda: {"queries": [], "ended": [], "floor": None}))
    notices: list[Notice] = field(default_factory=list)
    version: int = 0
    cart: dict[str, int] = field(default_factory=dict)
    cart_total: float | None = None
    cart_version: int = 0
    memo: str = ""

    def line(self, number: int) -> Line:
        return next(item for item in self.lines if item.number == number)

    def wants_of(self, number: int) -> list[Want]:
        return [w for w in self.wants if w.alive and w.line == number and not w.soft]

    def soft_of(self, number: int) -> list[Want]:
        return [w for w in self.wants if w.alive and w.line == number and w.soft]

    def add_want(self, line: int, prop: str, rel: str = "is", value: str = "", source: str = "request", sentence: str = "", ref: str = "") -> Want:
        want = Want(f"w{len(self.wants) + 1}", line, prop, rel, value, source, sentence, ref)
        self.wants.append(want)
        self.version += 1
        return want

    def tool(self, role: str) -> Tool | None:
        return self.tools.get(role)

    def msgs_free(self) -> int:
        owed = sum(d.messages for d in self.duties if d.done_at is None)
        return max(0, self.msg_limit - self.msg_used - owed)


@dataclass
class Reply:
    text: str
    calls: list[dict[str, Any]]
    model: str


class Ops:

    def __init__(self, binding: dict[str, Any], policy: dict[str, Any]):
        self.binding = binding
        self.max_steps = int(policy.get("max_steps") or 0)
        self.schemas = {t["function"]["name"]: t["function"] for t in policy.get("tools") or []}
        self.tool_list = policy.get("tools") or []
        self.store_client, self.model_client = ProxyClient(max_retries=1), ProxyClient(max_retries=1)
        self.started = self.tokens_at = time.monotonic()
        self.step = 0
        self.finished, self.no_credit = bool(policy.get("finished")), False
        self.spent = 0.0
        self.waited = 0.0
        self.tokens, self.rate = BURST, RATE
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(POOL_SIZE)
        self.seats: dict[str, list[float]] = defaultdict(lambda: [0, 0.0, 0.0, 0, 0, 0])
        self.trace: list[dict[str, Any]] = []

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def note(self, kind: str, **fields: Any) -> None:
        self.trace.append({"kind": kind, "step": self.step, "at": round(self.elapsed(), 1), **fields})

    def pace(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(BURST, self.tokens + (now - self.tokens_at) * self.rate)
                self.tokens_at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait * (0.9 + 0.3 * random.random()))

    def rate_seen(self, status: Any) -> None:
        with self.lock:
            self.rate = max(RATE_FLOOR, self.rate / 2) if status == 429 else min(RATE, self.rate + RATE_REGAIN)

    def may_think(self, kind: str) -> bool:
        if self.no_credit:
            return False
        if kind == "walk":
            return self.spent < MONEY_WALK and self.elapsed() < TIME_WALK
        if kind == "plan":
            return self.spent < MONEY_PLAN and self.elapsed() < TIME_CLOSE
        return self.spent < MONEY_STOP and self.elapsed() < TIME_DEADLINE - 8

    def think(self, seat: str, fixed: str, variable: str | tuple[str, str] | list[dict[str, Any]], kind: str = "walk", heavy: bool = False, tools: list[dict[str, Any]] | None = None) -> Reply | None:
        if not self.may_think(kind):
            return None
        wants_heavy = heavy and (self.spent + MONEY_CALL <= (MONEY_PLAN if kind == "plan" else MONEY_LAST))
        for model in (HEAVY, LIGHT) if wants_heavy else (LIGHT, HEAVY):
            reply = self.call_model(seat, model, fixed, variable, tools)
            if reply is not None or self.no_credit or not self.may_think(kind):
                return reply
        return None

    def call_model(self, seat: str, model: str, fixed: str, variable: str | tuple[str, str] | list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Reply | None:
        system: Any = fixed
        if len(fixed) >= CACHE_MIN_CHARS:
            system = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}]
        user: Any = variable
        if isinstance(variable, tuple):
            user = [{"type": "text", "text": variable[0], "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable[1]}]
        messages = [{"role": "user", "content": user}]
        if isinstance(variable, list):
            messages = [dict(message) for message in variable]
            if isinstance(messages[-1].get("content"), str) and messages[-1]["content"]:
                messages[-1]["content"] = [{"type": "text", "text": messages[-1]["content"], "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": [{"role": "system", "content": system}, *messages]}
        if tools:
            body["tools"], body["tool_choice"] = tools, "auto"
        if seat == "CARD":
            body["reasoning"] = {"enabled": False}
        if seat in SEAT_EFFORT:
            body["reasoning"] = {"effort": SEAT_EFFORT[seat]}
        for attempt in range(len(RETRY_WAITS) + 1):
            self.pace()
            began = time.monotonic()
            self.model_client.timeout = max(5, int(min(MODEL_WAIT, TIME_DEADLINE - self.elapsed() - 3)))
            result = self.model_client.post_verbose("/inference/chat/completions", body)
            data = result.data
            status = (result.error or {}).get("status")
            self.rate_seen(status)
            with self.lock:
                record = self.seats[seat]
                record[0] += 1
                record[2] += time.monotonic() - began
                if isinstance(data, dict):
                    usage = data.get("usage") or {}
                    cost = (number(usage.get("cost")) or 0.0) + ((number((usage.get("cost_details") or {}).get("upstream_inference_cost")) or 0.0) if usage.get("is_byok") else 0.0)
                    self.spent += cost
                    record[1] += cost
                    record[3] += usage.get("prompt_tokens") or 0
                    record[4] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    record[5] += usage.get("completion_tokens") or 0
            if isinstance(data, dict) and data.get("choices"):
                choice = data["choices"][0]
                message = choice.get("message") or {}
                if choice.get("finish_reason") != "length" and (message.get("content") or message.get("tool_calls")):
                    return Reply(message.get("content") or "", message.get("tool_calls") or [], model)
                self.note("model_empty", seat=seat, model=model)
                return None
            if isinstance(data, dict) and isinstance(data.get("error"), dict):
                status = number(data["error"].get("code"))
            self.note("model_fail", seat=seat, model=model, status=status, body=str((result.error or {}).get("body") or "")[:200])
            if status == 402:
                self.no_credit = True
                return None
            passing = status is None or status in (408, 425, 429) or status >= 500
            if not passing or attempt >= len(RETRY_WAITS) or self.waited + RETRY_WAITS[attempt] > RETRY_TOTAL or TIME_DEADLINE - self.elapsed() < RETRY_WAITS[attempt] + 8:
                return None
            self.waited += RETRY_WAITS[attempt]
            time.sleep(RETRY_WAITS[attempt] * (0.8 + 0.4 * random.random()))
        return None

    def fan(self, jobs: list[Callable[[], Any]]) -> list[Any]:
        if not jobs:
            return []
        results = [self.safely(jobs[0])]
        results += list(self.pool.map(self.safely, jobs[1:]))
        return results

    def safely(self, job: Callable[[], Any]) -> Any:
        try:
            return job()
        except Exception as exc:
            self.note("job_crash", error=f"{type(exc).__name__}: {exc}"[:300])
            return None

    def act_many(self, calls: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
        self.pace()
        calls = calls[:min(STORE_BATCH, self.steps_left())]
        for attempt in range(4):
            if self.finished or not calls or TIME_DEADLINE - self.elapsed() < 1:
                return []
            self.store_client.timeout = max(3, int(min(STORE_WAIT, TIME_DEADLINE - self.elapsed())))
            first = self.step
            result = self.store_client.post_verbose("/environment/call", {**self.binding, "step": first + 1, "actions": [{"name": name, "args": args} for name, args in calls]})
            if isinstance(result.data, dict):
                self.rate_seen(200)
                data = result.data
                seen = data.get("observations") if isinstance(data.get("observations"), list) else [data.get("observation")]
                self.step = data["step"] if isinstance(data.get("step"), int) else first + len(seen)
                self.finished = bool(data.get("finished")) or self.finished
                return [{"observation": item, "finished": place == len(seen) - 1 and bool(data.get("finished")), "step": first + place + 1} for place, item in enumerate(seen)]
            status = (result.error or {}).get("status")
            self.rate_seen(status)
            body = str((result.error or {}).get("body") or "")
            self.note("store_fail", tool=calls[0][0], status=status, body=body[:200], batch=len(calls))
            if status == 409 and re.search(r"expected (\d+)", body):
                self.step = int(re.search(r"expected (\d+)", body).group(1)) - 1
                continue
            if status in (400, 401, 402, 403, 404, 409, 422):
                return []
            time.sleep(2.0 ** attempt * (0.4 + 0.2 * random.random()))
        return []

    def act(self, name: str, args: dict[str, Any], ending: bool = False) -> dict[str, Any] | None:
        grace = TIME_GRACE if ending else 0.0
        self.pace()
        for attempt in range(4):
            if self.finished or self.steps_left() <= 0 or TIME_DEADLINE + grace - self.elapsed() < 1:
                return None
            self.store_client.timeout = max(3, int(min(STORE_WAIT, TIME_DEADLINE + grace - self.elapsed())))
            payload = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
            result = self.store_client.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                self.rate_seen(200)
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                self.finished = bool(data.get("finished")) or self.finished
                seen = data.get("observation")
                if isinstance(seen, dict) and "budget_exceeded" in seen.values():
                    self.note("budget_exceeded", tool=name)
                return data
            status = (result.error or {}).get("status")
            self.rate_seen(status)
            body = str((result.error or {}).get("body") or "")
            self.note("store_fail", tool=name, status=status, body=body[:200])
            if status == 409:
                low = body.lower()
                if "finish" in low or "finaliz" in low or "quarantin" in low:
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    continue
                self.step += 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                return None
            time.sleep(2.0 ** attempt * (0.4 + 0.2 * random.random()))
        return None

READ_ROLES = ("search", "browse", "facts", "page", "compare", "cart", "view", "message", "order", "decline", "finish", "meter")
READ_RELATIONS, READ_GOALS = ("is", "not", "at least", "at most", "one of", "range", "like", "when"), ("any", "cheapest", "best")
READ_DIRECTIONS = ("highest", "lowest")
READ_KINDS, READ_WHENS = ("fits", "condition", "same", "other"), ("before_cart", "before_order", "at_end")
READ_FIELDS = ("lines", "wants", "ties", "duties", "total_cap", "decline_rule", "msg_limit", "tools", "version")
READ_NUMBER = r"\d[\d,]*(?:\.\d+)?"
READ_LISTS = ("items", "goals", "wants", "uses", "references", "caps", "ties", "duties", "story", "tools")


def read_numbers(text: str) -> set[float]:
    return {number(found) for found in re.findall(READ_NUMBER, text)}


def read_first_number(cell: str) -> float | None:
    found = re.search(READ_NUMBER, cell)
    return number(found.group()) if found else None


def read_source(quote: str, request: str) -> str:
    words = join_words(quote)
    if words.count(" ") < 2 or words not in join_words(request):
        return ""
    return next((sentence for sentence in re.split(r"(?<=[.!?])\s+", request) if words in join_words(sentence)), quote)


def read_tool_cards(ops: Ops) -> str:
    cards = []
    for name, schema in ops.schemas.items():
        params = schema.get("parameters") or {}
        required = params.get("required") or []
        properties = (params.get("properties") or {}).items()
        arguments = "; ".join(f"{arg}{'*' if arg in required else ''} {dump(spec)}" for arg, spec in properties)
        cards.append(f"{name}: {schema.get('description') or ''}\n  arguments: {arguments or 'none'}")
    return "\n".join(cards)


def read_field(entry: dict[str, Any], key: str) -> str:
    value = entry.get(key)
    return "" if value is None else str(value).strip()


def read_line(sheet: Board, entry: dict[str, Any], key: str = "line") -> Line | None:
    target = read_first_number(read_field(entry, key))
    return next((line for line in sheet.lines if line.number == target), None)


def read_items(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    for entry in entries:
        index, name = read_first_number(read_field(entry, "number")), read_field(entry, "name")
        count, goal = read_first_number(read_field(entry, "quantity") or "1"), read_field(entry, "goal").lower()
        if not index or not name or any(line.number == index for line in sheet.lines):
            problems.append(f"items {index} {name}: a line needs a new number from 1 up and a name")
            continue
        if count is None or goal not in READ_GOALS:
            problems.append(f"items {index} {name}: the quantity must be a number and the goal any, cheapest or best")
        sheet.lines.append(Line(int(index), name, qty=max(1, int(count or 1)), goal=goal if goal in READ_GOALS else "any"))
    if not sheet.lines:
        problems.append("there is no entry under items")


def read_joined(sheet: Board, request: str, problems: list[str]) -> None:
    names = [re.escape(line.name) for line in sheet.lines]
    if len(names) > 1 and re.search(r"(?:,\s*|,?\s+(?:and|or)\s+)".join(names), request, re.IGNORECASE):
        problems.append(f"items {', '.join(line.name for line in sheet.lines)}: the request joins these names in one phrase with one quantity, so they are one line unless the request gives each its own quantity")


def read_goals(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    named = set()
    for entry in entries:
        line, measure, direction = read_line(sheet, entry), read_field(entry, "measure"), read_field(entry, "direction").lower()
        if line is None or not measure or direction not in READ_DIRECTIONS:
            problems.append(f"goals {entry}: it needs a known line, a measure and highest or lowest")
            continue
        named.add(line.number)
        if line.goal != "cheapest":
            line.goal, line.best_by = "best", (measure, direction)
    problems.extend(f"items {line.number} {line.name}: goal best needs an entry under goals" for line in sheet.lines
                    if line.goal == "best" and line.number not in named)


def read_wants(sheet: Board, answer: dict[str, Any], problems: list[str]) -> None:
    known = {line.number for line in sheet.lines} | {0}
    for key, field_name in (("wants", "value"), ("references", "words")):
        for entry in answer.get(key) or []:
            line, prop, relation = read_first_number(read_field(entry, "line")), read_field(entry, "property"), read_field(entry, "relation").lower()
            value, sentence = read_field(entry, field_name), read_source(read_field(entry, "sentence"), sheet.request)
            if line not in known or relation not in READ_RELATIONS or not prop or not value or not sentence:
                problems.append(f"{key} {line} {prop} {relation} {value}: it needs a known line, a property, a listed relation, a value and a sentence copied from the request")
            if prop and value:
                sheet.add_want(int(line) if line in known else 0, prop, relation if relation in READ_RELATIONS else "is",
                               "" if key == "references" else value, "request", sentence, value if key == "references" else "")
    for entry in answer.get("uses") or []:
        line, prop, value = read_first_number(read_field(entry, "line")), read_field(entry, "property"), read_field(entry, "value")
        if line in known and prop and value:
            sentence = read_source(read_field(entry, "sentence"), sheet.request)
            sheet.add_want(int(line), prop, "is", value, "request", sentence).soft = f" {join_words(prop)} " not in f" {join_words(sentence)} "


def read_caps(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    for entry in entries:
        target, amount, line = read_field(entry, "target"), read_first_number(read_field(entry, "amount")), read_line(sheet, entry, "target")
        if amount is None:
            problems.append(f"caps {target}: the amount must be a number")
        elif line:
            line.cap = amount / line.qty if read_field(entry, "applies").lower() == "all" else amount
        else:
            sheet.total_cap = amount
            if target.lower() not in ("total", "0"):
                problems.append(f"caps {target}: name a known line or total")


def read_ties(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    for entry in entries:
        first, second, kind = read_line(sheet, entry, "a"), read_line(sheet, entry, "b"), read_field(entry, "kind").lower()
        sentence, rule = read_source(read_field(entry, "sentence"), sheet.request), read_field(entry, "rule")
        if first and second and first is not second and sentence:
            sheet.ties.append(Tie(first.number, second.number, kind if kind in READ_KINDS else "other", rule, sentence))
        else:
            problems.append(f"ties {read_field(entry, 'a')} {read_field(entry, 'b')} {rule}: copy the request's sentence that binds these two different lines word for word, or drop the entry")


def read_duties(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    for entry in entries:
        sentence, stage, when = read_source(read_field(entry, "sentence"), sheet.request), join_words(read_field(entry, "stage")), read_field(entry, "when").lower()
        if sentence and stage and stage in join_words(sentence) and when in READ_WHENS:
            sheet.duties.append(Duty(when, sentence, int(read_first_number(read_field(entry, "messages")) or 0)))
        else:
            problems.append(f"duties {when} {stage}: a duty needs its stage, the words of the sentence naming that stage, and the sentence copied word for word; a sentence that names no stage is no duty")


def read_story(sheet: Board, entries: list[dict[str, Any]], problems: list[str]) -> None:
    known = {line.number for line in sheet.lines}
    for entry in entries:
        target, sentence = read_first_number(read_field(entry, "line")), read_source(read_field(entry, "sentence"), sheet.request)
        if not sentence:
            problems.append(f"story {read_field(entry, 'sentence')}: copy the sentence word for word from the request, or drop the entry")
            continue
        for line in sheet.lines:
            if (target == line.number or target not in known) and sentence not in line.story:
                line.story.append(sentence)


def read_tool_args(schema: dict[str, Any], entry: dict[str, Any], problems: list[str]) -> dict[str, str]:
    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    name, role, mode = read_field(entry, "tool"), read_field(entry, "job").lower(), read_field(entry, "mode")
    named, selects = [value if isinstance(value, dict) else {} for value in (entry.get("args"), entry.get("selects"))]
    choices = [str(choice) for choice in (properties.get(mode) or {}).get("enum") or []]
    jobs = {job.lower(): str(arg) for job, arg in named.items() if str(arg) in properties}
    selected = {job.lower(): str(value) for job, value in selects.items() if str(value) in choices}
    args = {**jobs, **selected, **({"mode": mode} if mode in properties else {})}
    unmapped = [arg for arg in params.get("required") or [] if arg not in jobs.values() and not (arg == mode and selected)]
    unsettled = role == "finish" and not {"mode", "order", "decline"} <= args.keys()
    if unmapped or len(jobs) < len(named) or len(selected) < len(selects) or unsettled:
        missing = f" The required {', '.join(unmapped)} get no value: name the argument that selects what the tool does under mode and give the enum value for {role} under selects." if unmapped else ""
        problems.append(f"TOOL {name} | {role} | {dump(entry)}: every job must name its schema's argument or an enum value of mode, every required argument needs a job, finish needs mode, order and decline.{missing}")
    return args


def read_tools(sheet: Board, entries: list[dict[str, Any]], ops: Ops, problems: list[str]) -> None:
    labelled = set()
    for entry in entries:
        name, role = read_field(entry, "tool"), read_field(entry, "job").lower()
        schema = ops.schemas.get(name)
        if schema is None or role not in READ_ROLES + ("other",):
            problems.append(f"TOOL {name} | {role}: name a listed tool and a listed role")
            continue
        labelled.add(name)
        if role == "other":
            continue
        args = read_tool_args(schema, entry, problems)
        if role in sheet.tools and sheet.tools[role].name != name:
            problems.append(f"TOOL {name}: role {role} is held by {sheet.tools[role].name}; a role belongs to the one tool that fits it best")
        else:
            sheet.tools[role] = Tool(name, role, args)
    if "finish" in sheet.tools:
        sheet.tools.setdefault("order", sheet.tools["finish"])
        sheet.tools.setdefault("decline", sheet.tools["finish"])
    problems.extend(f"tool {name} has no entry" for name in ops.schemas.keys() - labelled)
    if not {"order", "decline"} <= sheet.tools.keys():
        problems.append("no tool holds the order and decline jobs")


def read_check_numbers(answer: dict[str, Any], ops: Ops, request: str, problems: list[str]) -> None:
    stated = read_numbers(request)
    for key, name in (("items", "quantity"), ("wants", "value"), ("references", "words"), ("caps", "amount")):
        for entry in answer.get(key) or []:
            for value in read_numbers(read_field(entry, name)) - stated - ({1.0} if key == "items" else set()):
                problems.append(f"{key} {entry}: {value:g} is not among the request's numbers; write it as the request does")
    written = " ".join(str(value) for key, entries in answer.items() if key not in ("tools", "mentioned") for entry in (entries if isinstance(entries, list) else [entries]) for value in (entry.values() if isinstance(entry, dict) else [entry]))
    for value in stated - read_numbers(written) - {float(ops.max_steps)}:
        problems.append(f"the request's number {value:g} is in no entry: add one when it states a line or a requirement")


def read_keep_scope(answer: dict[str, Any], ops: Ops, request: str) -> None:
    whole = f" {join_words(request)} "
    spans = [(read_first_number(read_field(entry, "line")), f" {join_words(read_field(entry, 'words'))} ") for entry in answer.get("references") or []]

    def described(text: str, skip: float | None = None) -> bool:
        words = f" {join_words(text)} "
        return bool(words.strip()) and sum(span.count(words) for line, span in spans if line != skip) >= whole.count(words) > 0

    items = answer.get("items") or []
    gone = {read_first_number(read_field(entry, "number")) for entry in items if described(read_field(entry, "name"), read_first_number(read_field(entry, "number")))} - {None}
    if len(gone) >= len(items):
        gone = set()
    for key, names in (("items", ("number",)), ("goals", ("line",)), ("wants", ("line",)), ("uses", ("line",)), ("references", ("line",)), ("caps", ("target",)), ("story", ("line",)), ("ties", ("a", "b"))):
        answer[key] = [entry for entry in answer.get(key) or [] if not {read_first_number(read_field(entry, name)) for name in names} & gone]
    kept = [entry for entry in answer["wants"] if not described(read_field(entry, "value"))]
    if gone or len(kept) < len(answer["wants"]):
        ops.note("read_scope", lines=sorted(gone), wants=[read_field(entry, "property") for entry in answer["wants"] if entry not in kept])
    answer["wants"] = kept


def read_sheet(text: str, ops: Ops, request: str) -> tuple[Board, list[str]]:
    answer, sheet = json_object(text), Board(request=request)
    problems = [] if answer else ["the answer was not one JSON object"]
    answer = {**{key: answer.get(key) for key in ("messages", "decline")}, **{key: [entry for entry in answer.get(key) or [] if isinstance(entry, dict)] if isinstance(answer.get(key), list) else [] for key in READ_LISTS}}
    read_keep_scope(answer, ops, request)
    read_items(sheet, answer["items"], problems)
    read_joined(sheet, request, problems)
    read_goals(sheet, answer["goals"], problems)
    read_wants(sheet, answer, problems)
    read_caps(sheet, answer["caps"], problems)
    read_ties(sheet, answer["ties"], problems)
    read_duties(sheet, answer.get("duties") or [], problems)
    read_story(sheet, answer["story"], problems)
    read_tools(sheet, answer.get("tools") or [], ops, problems)
    read_check_numbers(answer, ops, request, problems)
    stated = read_first_number(read_field(answer, "messages"))
    sheet.msg_limit = int(stated) if stated is not None else 3 * max(1, len(sheet.lines))
    sheet.decline_rule = read_field(answer, "decline")
    return sheet, problems


def read_walk_problem(problem: str) -> bool:
    return problem.startswith("TOOL ") and problem.split(" | ")[1:2] in (["search"], ["browse"], ["cart"])


def read_order(ops: Ops, board: Board) -> bool:
    opening = tag("opening", dump(board.opening)) if board.opening else ""
    view = tag("request", board.request) + opening + tag("tools", read_tool_cards(ops))
    reply = ops.think("READ", READ, view, kind="plan", heavy=True)
    if reply is None:
        return False
    sheet, problems = read_sheet(reply.text, ops, board.request)
    if not all(map(read_walk_problem, problems)):
        again = ops.think("READ", READ, view + tag("earlier_reading", reply.text) + tag("problems", "\n".join(problems)), kind="plan", heavy=True)
        if again is not None:
            other, left = read_sheet(again.text, ops, board.request)
            if (not other.lines, len(left)) <= (not sheet.lines, len(problems)):
                other.tools = {**sheet.tools, **other.tools}
                sheet, problems = other, left
    ops.note("read", lines=[[line.number, line.name, line.qty, line.goal] for line in sheet.lines], ties=len(sheet.ties), problems=problems)
    for name in READ_FIELDS:
        setattr(board, name, getattr(sheet, name))
    return bool(board.lines)

STORE_RAW_CHARS, STORE_TEXT_CHARS, STORE_ROW_TEXT_CHARS = 6000, 4000, 300
STORE_SAMPLE_ROWS, STORE_SAMPLE_CHARS, STORE_SHAPE_TRIES, STORE_CART_ROUNDS = 2, 200, 2, 3
STORE_ARG_BYTES = 4096
STORE_CHOICE_TRIES, STORE_AND_DROP, STORE_PAGES_TOGETHER = 3, .8, 4
STORE_SENTENCE_WORDS, STORE_TEXT_WORDS, STORE_TITLE_WORDS, STORE_WRAPPER_KEYS, STORE_COMPARE_GUESS = 3, 8, 40, 4, 4
STORE_ARG_JOBS = {"words", "shelf", "brand", "price_max", "page", "id", "ids", "qty", "text", "mode"}
STORE_JOBS = ("rows", "id", "title", "price", "shelf", "shop", "stock", "qty", "table", "text", "total", "next", "notice")
STORE_SEATLESS = {"message", "order", "decline", "finish", "meter"}
STORE_FINISHES, STORE_CARTS, STORE_OPENERS = {"order", "decline", "finish"}, {"cart", "view"}, {"facts", "page"}


@dataclass
class StoreShape:
    tool: str
    top: set[str] = field(default_factory=set)
    rows: str | None = None
    keys: dict[str, str] = field(default_factory=dict)
    tables: list[list[str]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    in_stock: str = ""
    empty: set[str] = field(default_factory=set)
    strays: dict[str, int] = field(default_factory=dict)


def store_get(holder: Any, path: str) -> Any:
    if not path:
        return holder
    if isinstance(holder, dict) and path in holder:
        return holder[path]
    for part in path.split("."):
        holder = holder.get(part) if isinstance(holder, dict) else None
    return holder


def store_field(shape: StoreShape, job: str, holder: Any) -> Any:
    key = shape.keys.get(job)
    return store_get(holder, key) if key else None


def store_is_amount(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def store_is_sentence(text: Any) -> bool:
    return isinstance(text, str) and len(text.split()) >= STORE_SENTENCE_WORDS


def store_sentences(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if store_is_sentence(value) else []
    items = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else []
    return [sentence for item in items for sentence in store_sentences(item)]


def store_only_sentences(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(store_only_sentences(inner) for inner in value.values())
    return isinstance(value, list) and bool(value) and all(store_is_sentence(item) for item in value)


def store_envelope(response: dict[str, Any]) -> tuple[Any, Any]:
    observation = response.get("observation")
    if isinstance(observation, dict) and "refusal" in observation and "observation" in observation:
        return observation["observation"], observation["refusal"]
    return observation, response.get("refusal")


def store_text_value(text: str) -> Any:
    word = text.strip()
    pairs: list[list[str]] = []
    for part in word.split(";") if "=" in word else []:
        found = re.match(r"\s*([^=;]{1,60}?)\s*=\s*(.*?)\s*$", part)
        if found:
            pairs.append([found.group(1).strip(), found.group(2)])
        elif pairs and part.strip():
            pairs[-1][1] += ";" + part
    if len(pairs) >= 2:
        return {name: held.strip() for name, held in pairs}
    low = word.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", word):
        return int(word)
    if re.fullmatch(r"-?\d+\.\d+", word):
        return float(word)
    return word


def store_text_record(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or ": " not in text:
        return None
    best: dict[str, Any] = {}
    for sep in (" | ", "\n", "; "):
        pairs: list[list[str]] = []
        for piece in (part.strip() for part in text.split(sep)):
            if not piece:
                continue
            found = re.match(r"([A-Za-z_][\w .\-/()%]{0,40}):\s(.*)", piece, re.S)
            if found:
                pairs.append([found.group(1).strip(), found.group(2)])
            elif pairs:
                pairs[-1][1] += sep + piece
        if len(pairs) >= 2 and len(pairs) > len(best):
            best = {key: store_text_value(value) for key, value in pairs}
    return best or None


def store_row_lists(value: Any, path: str = "", depth: int = 3) -> dict[str, list]:
    found: dict[str, list] = {}
    if isinstance(value, list):
        if not value or any(isinstance(item, dict) for item in value):
            found[path] = value
    elif isinstance(value, dict) and depth:
        for key, inner in value.items():
            found.update(store_row_lists(inner, f"{path}.{key}" if path else key, depth - 1))
    return found


def store_id_values(values: list[Any]) -> bool:
    def usable(value: Any) -> bool:
        return isinstance(value, (str, int)) and not isinstance(value, bool) and len(str(value).split()) == 1
    return all(usable(value) for value in values) and len({str(value) for value in values}) == len(values)


def store_flat_table(value: Any, name_key: str = "", value_key: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(inner, list) and all(not isinstance(item, (dict, list)) for item in inner):
                flat[key] = ", ".join(str(item) for item in inner)
            elif isinstance(inner, (dict, list)):
                nested = store_flat_table(inner, name_key, value_key)
                flat.update({f"{key} {name}": text for name, text in nested.items()})
            elif inner not in (None, ""):
                flat[key] = str(inner)
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                fields = list(entry)
                name, held = entry.get(name_key), entry.get(value_key)
                if name is None and fields:
                    name, held = entry[fields[0]], entry[fields[1]] if len(fields) > 1 else ""
                flat.update(store_flat_table({str(name): held}))
            elif isinstance(entry, str):
                name, _, held = entry.partition(":")
                flat[name.strip()] = held.strip()
    return {key: text for key, text in flat.items() if text}


def store_table_keys(record: dict[str, Any]) -> list[list[str]]:
    filled = [key for key, value in record.items() if (isinstance(value, dict) and value)
              or (isinstance(value, list) and any(isinstance(item, dict) for item in value))]
    return [[key] for key in filled[:1]]


def store_sample(value: Any) -> Any:
    if isinstance(value, str):
        return cut_text(value, STORE_SAMPLE_CHARS)
    if isinstance(value, dict):
        return {key: store_sample(inner) for key, inner in value.items()}
    if isinstance(value, list):
        left = len(value) - STORE_SAMPLE_ROWS
        return [store_sample(item) for item in value[:STORE_SAMPLE_ROWS]] + ([f"...({left} more entries)"] if left > 0 else [])
    return value


def store_reply(body: Any) -> str:
    if isinstance(body, str):
        return body
    texts = [value for value in (body.values() if isinstance(body, dict) else []) if isinstance(value, str) and value.strip()]
    return "\n".join(texts) if texts else dump(body)


def store_types(spec: dict[str, Any]) -> list[str]:
    kinds = spec.get("type") or []
    return [kinds] if isinstance(kinds, str) else list(kinds)


def store_fit_typed(value: Any, kind: str, spec: dict[str, Any]) -> Any:
    if kind == "string":
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"{dump(value)[:60]} is not text")
        text, limit = str(value).strip(), spec.get("maxLength")
        if limit is not None and len(text) > limit:
            head = text[:limit + 1].rsplit(" ", 1)[0] if " " in text[:limit + 1] else ""
            if not head:
                raise ValueError(f"{len(text)} characters where {limit} fit")
            text = head.rstrip()
        if len(text) < (spec.get("minLength") or 0):
            raise ValueError("the text is empty")
        return text
    if kind in ("integer", "number"):
        amount = number(value)
        if amount is None or (kind == "integer" and amount != int(amount)):
            raise ValueError(f"{dump(value)[:60]} is not {'a whole number' if kind == 'integer' else 'a number'}")
        if ("minimum" in spec and amount < spec["minimum"]) or ("maximum" in spec and amount > spec["maximum"]):
            raise ValueError(f"{amount:g} is outside {spec.get('minimum', '')} to {spec.get('maximum', '')}")
        return int(amount) if amount == int(amount) else amount
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        word = str(value).strip().lower()
        if word in ("true", "1", "false", "0"):
            return word in ("true", "1")
        raise ValueError(f"{dump(value)[:60]} is not true or false")
    if kind == "null":
        if value is None:
            return None
        raise ValueError("only null fits")
    if kind == "array":
        items = value if isinstance(value, list) else [value]
        if len(items) < spec.get("minItems", 0) or ("maxItems" in spec and len(items) > spec["maxItems"]):
            raise ValueError(f"{len(items)} entries where {spec.get('minItems', 0)} to {spec.get('maxItems', '')} fit")
        return [store_fit_value(item, spec.get("items") or {}) for item in items]
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{dump(value)[:60]} is not an object")
        return store_fit_object(value, spec)
    return value


def store_fit_value(value: Any, spec: dict[str, Any]) -> Any:
    if "const" in spec:
        return spec["const"]
    if spec.get("enum") is not None:
        wanted = str(value).strip().casefold()
        for choice in spec["enum"] if value is not None else []:
            if str(choice).casefold() == wanted:
                return choice
        raise ValueError(f"{dump(value)[:60]} is not one of {dump(spec['enum'])[:200]}")
    kinds = store_types(spec)
    why = ""
    for kind in kinds:
        try:
            return store_fit_typed(value, kind, spec)
        except ValueError as exc:
            why = str(exc)
    if kinds:
        raise ValueError(why)
    return value


def store_fit_object(args: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    props, required = spec.get("properties") or {}, spec.get("required") or []
    fitted = {}
    for key, value in args.items():
        inner = props.get(key)
        if inner is None:
            if not props and spec.get("additionalProperties", True) is not False:
                fitted[key] = value
        elif key in required or value not in (None, "") or "null" in store_types(inner):
            try:
                fitted[key] = store_fit_value(value, inner)
            except ValueError as exc:
                raise ValueError(f"argument {key}: {exc}") from None
    for key in required:
        inner = props.get(key) or {}
        if key not in fitted and "const" in inner:
            fitted[key] = inner["const"]
        elif key not in fitted and len(inner.get("enum") or []) == 1:
            fitted[key] = inner["enum"][0]
        elif key not in fitted:
            raise ValueError(f"required argument {key} is missing")
    return fitted


def store_trim_to_bytes(args: dict[str, Any]) -> None:
    while len(dump(args).encode("utf-8", "replace")) > STORE_ARG_BYTES:
        spaced = [key for key, value in args.items() if isinstance(value, str) and " " in value]
        if not spaced:
            raise ValueError(f"the arguments are over {STORE_ARG_BYTES} bytes")
        longest = max(spaced, key=lambda key: len(args[key]))
        args[longest] = args[longest][:len(args[longest]) // 2].rsplit(" ", 1)[0]


def store_tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


class Store:

    def __init__(self, ops: Ops, board: Board):
        self.ops, self.board = ops, board
        self.purse = Purse(ops, board, self)
        self.last_fault, self.last_refusal = "", []
        self.shapes: dict[str, StoreShape] = {}
        self._vocab: dict[str, list[str]] = {}
        self._known_keys: set[str] = set()
        self._heard: set[str] = set()
        self._seen_text = dump(board.opening) if board.opening else ""
        self._last: dict[str, Any] = {}
        self._shut: set[str] = set()
        self._shelves: dict[str, str] = {}
        self._sent: dict[tuple[str, str], dict[str, Any]] = {}
        self._bags: dict[tuple[str, frozenset[str], str], dict[str, Any]] = {}
        self._word_queries: list[dict[str, Any]] = []
        self._direction: str | None = None
        self._seen_words: set[str] = set()
        self._seen_keys: set[str] = set()
        self._fields: str | None = None
        self._price_ignored: set[tuple[str, str]] = set()
        self._words_ignored: set[tuple[str, str]] = set()
        self._shows: dict[str, set[str]] = {}
        self._unreadable: set[str] = set()
        self._blanks: dict[str, set[str]] = {}
        self._meaning, self._cart_shows, self._cart_unsure = "", "", False
        self._planned: dict[str, int] | None = None
        self._settled: set[tuple[str, str]] = set()


    def call(self, tool: str, args: dict[str, Any], ending: bool = False) -> dict[str, Any] | None:
        self.last_fault = ""
        fitted = self._fit(tool, args or {}, ending)
        if fitted is None:
            return None
        response = self.ops.act(tool, fitted, ending)
        if response is None:
            self._fail(f"{tool}: no store call could be made")
            return None
        self.absorb(tool, fitted, response)
        return response

    def _fail(self, why: str) -> None:
        self.last_fault = why
        self.ops.note("store_fault", fault=why[:300])

    def _roles(self, tool: str) -> set[str]:
        return {role for role, item in self.board.tools.items() if item.name == tool}

    def _params(self, tool: str) -> dict[str, Any]:
        return (self.ops.schemas.get(tool) or {}).get("parameters") or {}

    def _props(self, tool: str) -> dict[str, Any]:
        return self._params(tool).get("properties") or {}

    def _fit(self, tool: str, args: dict[str, Any], ending: bool) -> dict[str, Any] | None:
        if tool not in self.ops.schemas:
            self._fail(f"{tool}: the store has no such tool")
        elif tool in self._shut and not ending:
            self._fail(f"{tool}: its budget is used up, so the tool stays shut")
        elif "message" in self._roles(tool) and self.board.msg_used >= self.board.msg_limit:
            self._fail(f"{tool}: the {self.board.msg_limit} messages allowed are used up")
        elif self.ops.steps_left() == 1 and not ending and not self._roles(tool) & STORE_FINISHES:
            self._fail(f"{tool}: the last step belongs to the finish")
        else:
            try:
                fitted = store_fit_object(args, self._params(tool))
                store_trim_to_bytes(fitted)
            except (ValueError, OverflowError) as exc:
                self._fail(f"{tool}: {exc}")
                return None
            unseen = self._unseen_id(tool, fitted)
            if not unseen:
                return fitted
            self._fail(f"{tool}: {unseen} was in no result, so it is not a listing the store has shown")
        return None

    def _unseen_id(self, tool: str, args: dict[str, Any]) -> str:
        for item in self.board.tools.values():
            if item.name == tool:
                for job in ("id", "ids"):
                    held = args.get(item.args.get(job, ""))
                    for value in held if isinstance(held, list) else [] if held is None else [held]:
                        if not isinstance(value, (dict, list)) and str(value) not in self._seen_text:
                            return str(value)
        return ""

    def known_id(self, value: Any) -> str:
        text = str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""
        if not text or len(text.split()) != 1:
            return ""
        if text in self.board.listings:
            return text
        shaped = len(text) >= 4 and (re.search(r"\d", text) or "-" in text or "_" in text) and not re.fullmatch(r"-?\d+(\.\d+)?", text)
        if not shaped or not re.search(rf"(?<![\w-]){re.escape(text)}(?![\w-])", self._seen_text):
            return ""
        self.board.listings.setdefault(text, Listing(text, seen_step=self.ops.step))
        self.ops.note("id_adopted", id=text)
        return text

    def _arg(self, tool: Tool, job: str) -> str:
        props = self._props(tool.name)
        if tool.args.get(job, "") in props:
            return tool.args[job]
        if job == "words" and tool.role != "search":
            return ""
        if job in ("shelf", "price_max", "brand"):
            return next((other.args[job] for other in self.board.tools.values() if other.name == tool.name and other.args.get(job, "") in props), "")
        claimed = {arg for label, arg in tool.args.items() if label in STORE_ARG_JOBS and not (job == "words" and label == "text")}
        free = {arg: spec for arg, spec in props.items() if arg not in claimed and "enum" not in spec}
        required = self._params(tool.name).get("required") or []
        strings = [arg for arg, spec in free.items() if "string" in store_types(spec)]
        strings.sort(key=lambda arg: arg not in required)
        if job == "ids":
            return next((arg for arg, spec in free.items() if "array" in store_types(spec)), "")
        if job == "qty":
            return next((arg for arg, spec in free.items() if {"integer", "number"} & set(store_types(spec))), "")
        if job == "page":
            whole = next((arg for arg, spec in free.items() if "integer" in store_types(spec)), "")
            short = sorted((arg for arg in strings if arg not in required), key=lambda arg: free[arg].get("maxLength") or 10 ** 6)
            return whole or (short[0] if short else "")
        return strings[0] if strings and job in ("id", "text", "words") else ""

    def _mode_args(self, tool: Tool, job: str) -> dict[str, Any]:
        mode = tool.args.get("mode", "")
        choices = [str(choice) for choice in self._props(tool.name).get(mode, {}).get("enum") or []]
        if tool.args.get(job) in choices:
            return {mode: tool.args[job]}
        spare = [value for key, value in tool.args.items() if key != "mode" and value in choices]
        return {mode: spare[0]} if spare and tool.role not in STORE_FINISHES else {}

    def _open_choice(self, tool: Tool) -> str:
        required, props = self._params(tool.name).get("required") or [], self._props(tool.name)
        mode = tool.args.get("mode", "")
        if mode in required and not self._mode_args(tool, tool.role):
            return mode
        held = set(tool.args.values())
        return next((arg for arg in required if arg not in held and (props.get(arg) or {}).get("enum")), "")

    def _settle_choice(self, tool: Tool, words: str, shelf: str) -> None:
        choice = self._open_choice(tool)
        if not choice or (tool.name, tool.role) in self._settled:
            return
        self._settled.add((tool.name, tool.role))
        taken = {other.args.get(other.role) for other in self.board.tools.values() if other.name == tool.name}
        tool.args["mode"] = choice
        answers = []
        for value in [str(item) for item in self._props(tool.name)[choice]["enum"] if str(item) not in taken][:STORE_CHOICE_TRIES]:
            if not self._walk_room():
                break
            tool.args[tool.role] = value
            response = self.call(tool.name, self._search_args(tool, words, shelf, None))
            rows = self._last["ids"] if response is not None and not self.last_fault else []
            if rows:
                answers.append((self._last["total"] or len(rows), value))
        if answers:
            tool.args[tool.role] = min(answers)[1]
        else:
            tool.args.pop(tool.role, None)
        self.ops.note("choice_settled", tool=tool.name, argument=choice, answers=answers)

    def absorb(self, tool: str, args: dict[str, Any], response: dict[str, Any]) -> None:
        self._last, self.last_fault, self.last_refusal = {"ids": [], "total": None, "next": None}, "", []
        self._seen_text += dump(response)
        observation, refusal = store_envelope(response)
        if refusal:
            response["refusal"] = refusal
            self.last_refusal = store_sentences(refusal) or [dump(refusal)]
            self._hear(self.last_refusal, "refusal")
        body, riding = self._peel(tool, observation)
        self._hear(riding)
        if response.get("finished"):
            return
        roles = self._roles(tool)
        fault = self._fault_text(tool, body)
        if fault:
            self._fail(f"{tool}: {fault}")
            if isinstance(body, dict) and body.get("error") == "budget_exceeded":
                if "message" in roles:
                    self.board.msg_limit = self.board.msg_used
                else:
                    self._shut.add(tool)
            return
        if roles & STORE_FINISHES:
            if not self.last_refusal:
                self._fail(f"{tool}: the call was accepted but did not end the episode")
            return
        shape = self.shapes.get(tool) or self._learn(tool, body, args)
        if "message" in roles:
            self._file_reply(args, body)
        if shape is None:
            self._absorb_nothing(tool, roles, body)
            return
        self._hear(self._extras(shape, body))
        items = self._items(shape, body)
        self._check(shape, items if items is not None else [body], body)
        if roles & STORE_CARTS:
            self._absorb_cart(roles, shape, body, items, args)
        elif items is not None:
            self._file_rows(tool, roles, shape, items, body)
        elif "message" not in roles and isinstance(body, dict):
            self._file_record(tool, roles, shape, body, args)

    def _file_rows(self, tool: str, roles: set[str], shape: StoreShape, items: list[dict[str, Any]], body: Any) -> None:
        as_record = bool(roles & (STORE_OPENERS | {"compare"})) or bool(shape.tables)
        for item in items:
            item_id = self._item_id(shape, item)
            if item_id:
                self._file_item(tool, shape, item, item_id, as_record)
                self._last["ids"].append(item_id)
        self._last["total"] = number(store_field(shape, "total", body))
        self._last["next"] = store_field(shape, "next", body)

    def _file_record(self, tool: str, roles: set[str], shape: StoreShape, body: dict[str, Any], args: dict[str, Any]) -> None:
        names = [self._arg(item, "id") for item in self.board.tools.values() if item.name == tool]
        item_id = self._item_id(shape, body) or next((str(args[name]) for name in names if name in args), "")
        if item_id and (shape.keys or shape.tables or roles & STORE_OPENERS):
            self._file_item(tool, shape, body, item_id, True)
            self._last["ids"] = [item_id]

    def _hear(self, sentences: list[str], kind: str = "notice") -> None:
        for text in sentences:
            key = " ".join(text.lower().split())
            if key and key not in self._heard:
                self._heard.add(key)
                self.board.notices.append(Notice(text.strip(), self.ops.step, kind))

    def _is_normal(self, tool: str, value: dict[str, Any]) -> bool:
        shape = self.shapes.get(tool)
        return bool(shape and shape.top) and 2 * len(shape.top & set(value)) >= len(shape.top)

    def _wrapper_body(self, tool: str, value: dict[str, Any]) -> str:
        rest = [key for key, inner in value.items() if not store_only_sentences(inner)]
        if len(rest) != 1 or len(value) < 2 or len(value) > STORE_WRAPPER_KEYS:
            return ""
        held = value[rest[0]]
        bodied = isinstance(held, (dict, list)) or ("message" in self._roles(tool) and isinstance(held, str)) or bool(store_text_record(held))
        return rest[0] if bodied else ""

    def _peel(self, tool: str, value: Any) -> tuple[Any, list[str]]:
        riding: list[str] = []
        while isinstance(value, dict) and not self._is_normal(tool, value):
            inner = next(iter(value.values())) if len(value) == 1 else None
            if isinstance(inner, dict) and self._wrapper_body(tool, inner):
                value = inner
                continue
            body = self._wrapper_body(tool, value)
            if not body:
                break
            riding += [sentence for key, held in value.items() if key != body for sentence in store_sentences(held)]
            value = value[body]
        return self._as_records(tool, value), riding

    def _as_records(self, tool: str, value: Any, depth: int = 3) -> Any:
        if not depth or "message" in self._roles(tool):
            return value
        if isinstance(value, str):
            return store_text_record(value) or value
        if isinstance(value, list):
            if value and all(isinstance(item, str) for item in value):
                records = [store_text_record(item) for item in value]
                if 2 * sum(record is not None for record in records) >= len(records):
                    return [record if record is not None else {"text": item} for record, item in zip(records, value)]
            return value
        if isinstance(value, dict):
            sole = next(iter(value.values())) if len(value) == 1 else None
            if isinstance(sole, str) and (record := store_text_record(sole)):
                return record
            return {key: self._as_records(tool, inner, depth - 1) if isinstance(inner, (dict, list)) else inner for key, inner in value.items()}
        return value

    def _fault_text(self, tool: str, body: Any) -> str:
        if isinstance(body, dict) and body.get("error") in RUNTIME_ERRORS:
            return " ".join(str(value) for value in (body["error"], body.get("detail")) if value)
        if "message" in self._roles(tool):
            return ""
        if body in (None, {}, [], ""):
            return "the result is empty"
        if isinstance(body, str):
            return body
        if not isinstance(body, dict):
            return ""
        scalars = [value for value in body.values() if not isinstance(value, (dict, list))]
        small = len(scalars) == len(body) <= 2
        names_listing = any(str(value) in self.board.listings for value in scalars)
        if small and not names_listing and any(value is False for value in scalars):
            return dump(body)
        shape = self.shapes.get(tool)
        if shape and shape.top:
            return "" if shape.top & set(body) else dump(body)
        bare = small and not names_listing and all(isinstance(value, str) or value is None for value in scalars)
        return dump(body) if bare else ""

    def _extras(self, shape: StoreShape, body: Any) -> list[str]:
        if not isinstance(body, dict):
            return []
        labelled = set(shape.keys.values()) | {table[0] for table in shape.tables} | set(shape.texts)
        record = shape.rows is None and "id" in shape.keys
        found: list[str] = []
        for key, value in body.items():
            if key not in shape.top:
                shape.strays[key] = shape.strays.get(key, 0) + 1
                if record and shape.strays[key] > 1:
                    shape.top.add(key)
                    shape.texts.append(key)
                else:
                    found += store_sentences(value)
            elif key in shape.empty and value not in (None, "", [], {}):
                shape.empty.discard(key)
                if not record and key not in labelled:
                    found += store_sentences(value)
        return found

    def _items(self, shape: StoreShape, body: Any) -> list[dict[str, Any]] | None:
        if shape.rows is None:
            return None
        rows = store_get(body, shape.rows)
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []

    def _item_id(self, shape: StoreShape, item: Any) -> str:
        value = store_field(shape, "id", item)
        return str(value) if value not in (None, "") else ""


    def _learn(self, tool: str, body: Any, args: dict[str, Any]) -> StoreShape | None:
        if not isinstance(body, (dict, list)):
            return None
        roles = self._roles(tool)
        opens_one = bool(roles & STORE_OPENERS) and isinstance(body, dict)
        lists = {} if opens_one or roles & STORE_SEATLESS else store_row_lists(body)
        if lists and not any(lists.values()):
            return None
        nested = any(isinstance(value, (dict, list)) for value in body.values()) if isinstance(body, dict) else True
        plain = not lists and len(body) < 3 and not nested
        labels, whole = ({}, False) if plain or roles & STORE_SEATLESS else self._labels(tool, body, lists, args)
        asked = labels.get("rows", [[""]])[-1][0]
        rows = asked if asked in lists else max(lists, key=lambda path: len(lists[path])) if lists else None
        shape = StoreShape(tool, top=set(body) if isinstance(body, dict) else set(), rows=rows)
        items = self._items(shape, body)
        sample = items if items is not None else [body]
        if not roles & STORE_SEATLESS:
            self._take_labels(shape, labels, sample, body)
            self._guess(shape, sample, body, args, thorough=not whole and not plain)
        if isinstance(body, dict):
            shape.empty = {key for key in shape.top if body[key] in (None, "", [], {})}
        self.shapes[tool] = shape
        self._remember(shape, sample)
        self.ops.note("shape", tool=tool, rows=rows, keys=dict(shape.keys), tables=list(shape.tables), texts=list(shape.texts),
                      seat=whole)
        if roles & (STORE_OPENERS | {"search", "browse", "compare"}) and "id" not in shape.keys:
            self.ops.note("shape_blind", tool=tool, keys=sorted(sample[0]) if sample and isinstance(sample[0], dict) else [])
        return shape

    def _labels(self, tool: str, body: Any, lists: dict[str, list],
                args: dict[str, Any]) -> tuple[dict[str, list[list[str]]], bool]:
        rows = max(lists.values(), key=len, default=[])
        sample = next((row for row in rows if isinstance(row, dict)), body if isinstance(body, dict) else {})
        outer = {path.split(".")[0] for path in lists}
        keys = set(sample) | (set(body) - outer if isinstance(body, dict) else set())
        known: dict[str, list[list[str]]] = {}
        for key in keys:
            job, *cells = self._vocab.get(key, ["", ""])
            if job:
                known.setdefault(job, []).append([key] + cells)
        if keys <= self._known_keys:
            return known, bool(known)
        description = (self.ops.schemas.get(tool) or {}).get("description") or ""
        view = tag("tool", f"{tool}: {description}\nlabelled: {', '.join(sorted(self._roles(tool))) or 'unknown'}")
        view += tag("sent", dump(args)) + tag("result", dump(store_sample(body)))
        labels: dict[str, list[list[str]]] = {}
        for _ in range(STORE_SHAPE_TRIES):
            reply = self.ops.think("SHAPE", SHAPE, view, kind="plan")
            if reply is None:
                break
            rows = tag_rows(reply.text)
            for head, *cells in rows:
                clean = [cell.strip("`*'\" ") for cell in cells]
                job = head.strip("*:#").lower()
                if job in STORE_JOBS and clean and clean[0]:
                    labels.setdefault(job, []).append(clean)
            if labels:
                return labels, True
            if any(row[0] == "NONE" for row in rows):
                break
        return known, False

    def _holds(self, shape: StoreShape, job: str, key: str, items: list[dict[str, Any]], body: Any) -> bool:
        if job in ("total", "next"):
            value = store_get(body, key)
            if job == "next" or value is None:
                return not isinstance(value, (dict, list))
            amount = number(value)
            return amount is not None and (amount >= len(items) or bool(self._roles(shape.tool) & STORE_CARTS))
        values = [store_get(item, key) for item in items]
        present = [value for value in values if value not in (None, "")]
        if job == "id":
            return store_id_values(values)
        if job in ("price", "qty"):
            return all(number(value) is not None for value in present)
        if job in ("title", "shelf", "shop"):
            return all(isinstance(value, str) for value in present)
        return all(not isinstance(value, (dict, list)) for value in present)

    def _take_labels(self, shape: StoreShape, labels: dict[str, list[list[str]]], items: list[dict[str, Any]], body: Any) -> None:
        first = items[0] if items else {}
        for job, rows in labels.items():
            if job == "table":
                held = [(cells, store_get(first, cells[0])) for cells in rows]
                shape.tables += [cells[:3] for cells, value in held if isinstance(value, (dict, list))
                                 or (value is None and cells[0] in first)]
            elif job == "text":
                named = [key for cells in rows for key in cells]
                shape.texts += [key for key in named if key in first and isinstance(first[key], (str, type(None)))]
            elif job == "notice":
                for key in (key for cells in rows for key in cells):
                    if store_sentences(store_get(body, key)):
                        shape.top.discard(key)
            elif job != "rows":
                key = rows[-1][0]
                holder = body if job in ("total", "next") else first
                known = isinstance(holder, dict) and (key in holder or store_get(holder, key) is not None)
                if known and self._holds(shape, job, key, items, body):
                    shape.keys[job] = key
                    if job == "stock" and len(rows[-1]) > 1:
                        shape.in_stock = rows[-1][1]
                else:
                    self.ops.note("shape_reject", tool=shape.tool, job=job, key=key)

    def _guess_id(self, items: list[dict[str, Any]], args: dict[str, Any]) -> str:
        if not items:
            return ""
        sent = {str(value) for held in args.values() for value in (held if isinstance(held, list) else [held])}
        ranked = []
        for key in items[0]:
            values = [item.get(key) for item in items]
            if store_id_values(values):
                tied = any(str(value) in sent or str(value) in self.board.listings for value in values)
                text = str(values[0])
                ranked.append((tied, isinstance(values[0], str), any(char.isdigit() for char in text), len(text), key))
        best = max(ranked, default=None)
        return best[-1] if best and (best[0] or len(items) > 1) else ""

    def _guess(self, shape: StoreShape, items: list[dict[str, Any]], body: Any, args: dict[str, Any], thorough: bool) -> None:
        keys = shape.keys
        if "id" not in keys and (found := self._guess_id(items, args)):
            keys["id"] = found
        first = items[0] if items and isinstance(items[0], dict) else {}
        amounts = [key for key, value in first.items() if store_is_amount(value) and key != keys.get("id")]
        if "price" not in keys and (priced := [key for key in amounts if self._vocab.get(key, [""])[0] == "price"]):
            keys["price"] = priced[0]
        if self._roles(shape.tool) & STORE_CARTS and "id" in keys and "qty" not in keys:
            counts = sorted((key for key in amounts if key != keys.get("price")), key=lambda key: first[key])
            if counts:
                keys["qty"] = counts[0]
        if thorough:
            self._guess_rest(shape, first, amounts, body)

    def _guess_rest(self, shape: StoreShape, first: dict[str, Any], amounts: list[str], body: Any) -> None:
        keys = shape.keys
        left = [key for key in amounts if key not in keys.values()]
        if "price" not in keys and len(left) == 1:
            keys["price"] = left[0]
        flags = [key for key, value in first.items() if isinstance(value, bool)]
        if "stock" not in keys and len(flags) == 1:
            keys["stock"] = flags[0]
        words = {key: len(value.split()) for key, value in first.items() if isinstance(value, str) and key != keys.get("id")}
        titles = [key for key, count in words.items() if 2 <= count <= STORE_TITLE_WORDS]
        if "title" not in keys and titles:
            keys["title"] = max(titles, key=lambda key: words[key])
        shape.texts += [key for key, count in words.items() if count >= STORE_TEXT_WORDS and key != keys.get("title")
                        and key not in shape.texts]
        if not shape.tables:
            shape.tables += store_table_keys(first)
        if shape.rows and isinstance(body, dict):
            outer = shape.rows.split(".")[0]
            numbers = [key for key, value in body.items() if key != outer and store_is_amount(value)]
            if numbers and "total" not in keys:
                keys["total"] = max(numbers, key=lambda key: body[key])
            others = [key for key in body if key not in (outer, keys.get("total"))]
            marks = [key for key in others if not isinstance(body[key], (dict, list, bool))]
            if marks and "next" not in keys:
                keys["next"] = marks[0]

    def _remember(self, shape: StoreShape, items: list[Any]) -> None:
        for job, key in shape.keys.items():
            self._vocab[key] = [job]
        for table in shape.tables:
            self._vocab[table[0]] = ["table"] + table[1:]
        for key in shape.texts:
            self._vocab[key] = ["text"]
        self._known_keys |= shape.top | (set(items[0]) if items and isinstance(items[0], dict) else set())

    def _check(self, shape: StoreShape, items: list[Any], body: Any) -> None:
        for job, key in list(shape.keys.items()):
            if not self._holds(shape, job, key, items, body):
                del shape.keys[job]
                self.ops.note("shape_drop", tool=shape.tool, job=job, key=key)
                if job == "id" and (found := self._guess_id(items, {})):
                    shape.keys["id"] = found


    def _note_view(self, tool: str, shape: StoreShape, item: dict[str, Any]) -> None:
        jobs = set(shape.keys.values())
        if not shape.tables:
            shape.tables += store_table_keys(item)
        own = shape.top if shape.rows is None else set()
        shape.texts += [key for key in own & set(item) - jobs - set(shape.texts)
                        if isinstance(item[key], str) and len(item[key].split()) >= STORE_TEXT_WORDS]
        shown = self._shows.setdefault(tool, set())
        if any(isinstance(store_get(item, table[0]), (dict, list)) for table in shape.tables):
            shown.add("table")
        if any(key in item and key not in jobs for key in shape.texts):
            shown.add("text")
        shown.update(job for job in ("price", "stock") if store_field(shape, job, item) is not None)

    def shows_table(self, tool: str) -> bool:
        return "table" in self._shows.get(tool, ())

    def shows_text(self, tool: str) -> bool:
        return "text" in self._shows.get(tool, ())

    def _text_keys(self, tool: str) -> set[str]:
        shape = self.shapes.get(tool)
        return set(shape.texts) - set(shape.keys.values()) if shape else set()

    def _shows_all_text(self, tool: str, other: str) -> bool:
        return other in self._shows and self.shows_text(tool) and self._text_keys(other) <= self._text_keys(tool)

    def _file_item(self, tool: str, shape: StoreShape, item: dict[str, Any], item_id: str, as_record: bool) -> None:
        listing = self.board.listings.setdefault(item_id, Listing(item_id, seen_step=self.ops.step))
        old = listing.records.get(tool) if as_record else (listing.row or None)
        self._apply_fields(listing, shape, item, old, not as_record)
        if as_record:
            self._note_view(tool, shape, item)
            self._note_changes(listing, shape, old, item)
            listing.records[tool] = item
        else:
            listing.row = item
        for table in shape.tables:
            listing.facts.update(store_flat_table(store_get(item, table[0]), *table[1:3]))
        if as_record and not shape.tables:
            head = set(shape.keys.values()) | set(shape.texts)
            listing.facts.update({key: str(value) for key, value in item.items()
                                  if key not in head and value not in (None, "") and not isinstance(value, (dict, list))})
        self._seen_words |= store_tokens(" ".join(str(value) for value in [*item.values(), *listing.facts.values()] if not isinstance(value, (dict, list))))
        self._seen_keys |= store_tokens(" ".join(listing.facts))
        self._refresh_notes(listing)

    def _apply_fields(self, listing: Listing, shape: StoreShape, item: dict[str, Any], old: Any, is_row: bool) -> None:
        for job in ("title", "shelf", "shop"):
            value = store_field(shape, job, item)
            if isinstance(value, str) and value.strip():
                setattr(listing, job, value)
                if job == "shelf":
                    self._shelves[value.casefold()] = value
        for job in ("price", "stock"):
            value = store_field(shape, job, item)
            stale = is_row and old is None and listing.records
            if value is None or stale or (old is not None and store_field(shape, job, old) == value):
                continue
            new = number(value) if job == "price" else self._stock_of(shape, value)
            held = getattr(listing, job)
            if new is not None and new != held:
                setattr(listing, job, new)
                if held is not None:
                    self._hear([f"Listing {listing.id} ({listing.title}) now shows {job} {new}; it showed {held}."])

    def _stock_of(self, shape: StoreShape, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        return str(value).casefold() == shape.in_stock.casefold() if shape.in_stock else None

    def _note_changes(self, listing: Listing, shape: StoreShape, old: dict[str, Any] | None, item: dict[str, Any]) -> None:
        if old is None:
            return
        settled = {shape.keys.get("price"), shape.keys.get("stock")}
        for key, value in item.items():
            if key not in settled and value != old.get(key):
                reads = cut_text(dump(value), STORE_ROW_TEXT_CHARS)
                self._hear([f"Listing {listing.id} ({listing.title}): {key} now reads {reads}."])

    def _refresh_notes(self, listing: Listing) -> None:
        texts = sorted({key for key, entry in self._vocab.items() if entry[0] == "text"}
                       | {key for shape in self.shapes.values() for key in shape.texts})
        parts: list[str] = []
        for source in [*listing.records.values(), listing.row]:
            for key in texts:
                text = source.get(key)
                if isinstance(text, str) and text.strip() and not any(text.strip() in part for part in parts):
                    parts.append(text.strip())
        listing.notes = "\n".join(parts)

    def _link(self, line: int, ids: list[str]) -> list[str]:
        seen = self.board.cover[line].setdefault("seen", set())
        fresh = [item_id for item_id in ids if item_id not in seen]
        seen.update(fresh)
        if line > 0:
            for item_id in ids:
                self.board.listings[item_id].lines.add(line)
        return fresh


    def _file_reply(self, args: dict[str, Any], body: Any) -> None:
        item = self.board.tool("message")
        reply = store_reply(body)
        self.board.msg_used += 1
        text = args.get(self._arg(item, "text"), "") if item else ""
        self.board.talk.append({"line": None, "attr": "", "text": text, "reply": reply, "kind": ""})
        self._last["reply"] = reply

    def say(self, text: str) -> str | None:
        item = self.board.tool("message")
        if item is None:
            self._fail("the store has no message tool")
            return None
        response = self.call(item.name, {self._arg(item, "text"): text})
        return None if response is None or self.last_fault else self._last.get("reply")


    def _walk_room(self) -> bool:
        if self.purse.room() > 0:
            return True
        self._fail("the steps left are held for the finish")
        return False

    def _shelf_spelling(self, tool: Tool, shelf: str) -> str:
        spec = self._props(tool.name).get(self._arg(tool, "shelf"))
        if spec is None or not shelf:
            return ""
        wanted = shelf.strip().casefold()
        if spec.get("enum"):
            return next((str(choice) for choice in spec["enum"] if str(choice).casefold() == wanted), "")
        return self._shelves.get(wanted, "")

    def shelves_seen(self) -> list[str]:
        return sorted(self._shelves.values())

    def unseen_shelf(self, name: str, args: dict[str, Any]) -> str:
        for tool in (tool for tool in self.board.tools.values() if tool.name == name and tool.args.get("shelf") in args):
            asked = str(args[tool.args["shelf"]])
            if asked and not self._shelf_spelling(tool, asked):
                return asked
        return ""

    def _search_tool(self, words: str, shelf: str, price_max: float | None) -> Tool | None:
        tools = [tool for tool in map(self.board.tool, ("search", "browse")) if tool]
        if words:
            return next((tool for tool in tools if self._arg(tool, "words") and (tool.name, self._mode_of(tool)) not in self._words_ignored), None)
        return next((tool for tool in reversed(tools) if self._shelf_spelling(tool, shelf)), None)

    def _mode_of(self, tool: Tool) -> str:
        return str(self._mode_args(tool, tool.role).get(tool.args.get("mode", ""), ""))

    def _search_args(self, tool: Tool, words: str, shelf: str, price_max: float | None) -> dict[str, Any]:
        args = self._mode_args(tool, tool.role)
        if words:
            arg = self._arg(tool, "words")
            limit = self._props(tool.name)[arg].get("maxLength")
            over = limit and len(words) > limit and " " in words[:limit + 1]
            args[arg] = words[:limit + 1].rsplit(" ", 1)[0] if over else words
        spelled = self._shelf_spelling(tool, shelf)
        if spelled:
            args[self._arg(tool, "shelf")] = spelled
        elif shelf and self._arg(tool, "shelf"):
            self.ops.note("shelf_unseen", shelf=shelf, tool=tool.name)
        arg = self._arg(tool, "price_max")
        if price_max is not None and arg in self._props(tool.name) and (tool.name, tool.role) not in self._price_ignored:
            args[arg] = int(price_max) if "integer" in store_types(self._props(tool.name)[arg]) else price_max
        return args

    def search(self, line: int, words: str, shelf: str = "", price_max: float | None = None) -> list[str]:
        words = " ".join(words.split())
        own = next((item.shelf for item in self.board.lines if item.number == line), "")
        tool = self._search_tool(words, shelf or own, price_max)
        if tool is None:
            self._fail("no search tool can take this query")
            return []
        self._settle_choice(tool, words, shelf or (own if not words else ""))
        args = self._search_args(tool, words, shelf or (own if not words else ""), price_max)
        key = (tool.name, dump(args).casefold())
        bag = (tool.name, frozenset(store_tokens(words)), dump({name: value for name, value in args.items() if name != self._arg(tool, "words")}).casefold())
        limit_new = words and key in self._sent and price_max is not None and self._arg(tool, "price_max") not in args
        if limit_new and self._search_tool("", shelf or own, price_max):
            return self.search(line, "", shelf or own, price_max)
        cover = self.board.cover[line]
        sent = self._sent.get(key) or self._bags.get(bag)
        if sent is not None:
            if sent not in cover["queries"]:
                cover["queries"].append(sent)
            return self._link(line, sent["rows"])
        if not self._walk_room():
            return []
        response = self.call(tool.name, args)
        if response is None:
            return []
        rows, total = self._last["ids"], self._last["total"]
        query = {"words": args.get(self._arg(tool, "words"), ""), "shelf": args.get(tool.args.get("shelf", ""), ""),
                 "price_max": args.get(self._arg(tool, "price_max")), "tool": tool.name, "role": tool.role, "args": args,
                 "total": total if total is not None else len(rows), "pages_read": 1, "ended": False, "next": None, "size": len(rows),
                 "rows": list(rows), "capped": False, "fault": self.last_fault, "all_seen": False}
        cover["queries"].append(query)
        self._sent[key] = self._bags[bag] = query
        self.note_words_ignored(query)
        self._after_page(line, query, tool, rows)
        if rows and any(other is not query and other["rows"][:len(rows)] == rows for other in cover["queries"]):
            self._end_query(line, query)
        self._note_filter(tool, query, rows)
        self._note_direction(query)
        self._note_fields(query, rows)
        self._note_shelf(line, rows)
        kept = self._pruned(line, words) if not rows and words and not shelf and price_max is None and self._direction != "or" else ""
        if kept:
            self.ops.note("search_pruned", words=words, kept=kept)
            return self.search(line, kept, shelf, price_max)
        return self._link(line, rows)

    def _pruned(self, line: int, words: str) -> str:
        wanted = set().union(*(store_tokens(want.value) for want in self.board.wants_of(line) if want.value), set())
        own = next((store_tokens(f"{item.name} {item.shelf}") for item in self.board.lines if item.number == line), set())
        unseen = [word for word in words.split() if not store_tokens(word) <= self._seen_words]
        is_own = lambda word: any(len(a) >= 4 and (a.startswith(b) or b.startswith(a)) for a in own for b in store_tokens(word))
        names = [word for word in words.split() if store_tokens(word) <= self._seen_keys and not is_own(word)]
        stages = ([word for word in words.split() if word in names or (word in unseen and not store_tokens(word) <= wanted)], names + unseen)
        for dropped in stages:
            kept = " ".join(word for word in words.split() if word not in dropped)
            tool = self._search_tool(kept, "", None) if kept and dropped else None
            if tool and (tool.name, dump(self._search_args(tool, kept, "", None)).casefold()) not in self._sent:
                return kept
        return ""

    def more_many(self, line: int, query_index: int, pages: int) -> list[str]:
        queries = self.board.cover[line]["queries"]
        query = queries[query_index]
        tool = self.board.tool(query.get("role", ""))
        page = self._arg(tool, "page") if tool else ""
        spec = self._props(tool.name).get(page) or {} if tool else {}
        if pages < 2 or query.get("ended") or not isinstance(query.get("next"), int) or "integer" not in store_types(spec) or not query["rows"]:
            return self.more(line, query_index)
        size = len(query["rows"]) // max(1, query["pages_read"])
        starts = [query["next"] + place * (size if spec.get("minimum") == 0 else 1) for place in range(pages)]
        reach = query["total"] if spec.get("minimum") == 0 and query.get("total") else None
        starts = [start for start in starts if (reach is None or start < reach) and (not isinstance(spec.get("maximum"), int) or start <= spec["maximum"])]
        room = self.purse.room()
        calls = [fitted for fitted in (self._fit(tool.name, {**query["args"], page: start}, False) for start in starts[:max(1, room)]) if fitted is not None]
        if len(calls) < 2:
            return self.more(line, query_index)
        found: list[str] = []
        for fitted, response in zip(calls, self.ops.act_many([(tool.name, fitted) for fitted in calls])):
            self.absorb(tool.name, fitted, response)
            rows = self._last["ids"]
            fresh = [item_id for item_id in rows if item_id not in set(query["rows"])]
            query["pages_read"] += 1
            query["rows"] += fresh
            found += self._link(line, rows)
            if self.last_fault or not fresh:
                self._end_query(line, query)
                break
            self._after_page(line, query, tool, rows)
        return found

    def more(self, line: int, query_index: int = -1) -> list[str]:
        queries = self.board.cover[line]["queries"]
        if not -len(queries) <= query_index < len(queries) or not queries:
            self._fail(f"line {line} has no query {query_index}")
            return []
        query = queries[query_index]
        tool = self.board.tool(query.get("role", ""))
        if query.get("ended") or tool is None or not self._walk_room():
            return []
        page = self._arg(tool, "page")
        response = self.call(tool.name, {**query["args"], page: query["next"]}) if page else None
        if response is None:
            self._end_query(line, query)
            return []
        rows = self._last["ids"]
        known = set(query["rows"])
        fresh = [item_id for item_id in rows if item_id not in known]
        query["pages_read"] += 1
        query["rows"] += fresh
        if self.last_fault or not fresh:
            self._end_query(line, query)
        else:
            self._after_page(line, query, tool, rows)
        return self._link(line, rows)

    def _after_page(self, line: int, query: dict[str, Any], tool: Tool, rows: list[str]) -> None:
        shape = self.shapes.get(tool.name)
        page = self._arg(tool, "page")
        spec = self._props(tool.name).get(page) or {}
        if not rows or self.last_fault or not page:
            value = None
        elif shape and "next" in shape.keys:
            value = None if self._last["next"] in (None, "", query["next"]) else self._last["next"]
        elif "integer" not in store_types(spec) or (shape and "total" in shape.keys and len(query["rows"]) >= query["total"]) or len(rows) < query.get("size", 0):
            value = None
        else:
            value = len(query["rows"]) if spec.get("minimum") == 0 else query["pages_read"] + 1
        if isinstance(value, int) and isinstance(spec.get("maximum"), int) and value > spec["maximum"]:
            value = None
        query["next"] = value
        if value is None:
            self._end_query(line, query)

    def _end_query(self, line: int, query: dict[str, Any]) -> None:
        query["ended"] = True
        keys = self.shapes[query["tool"]].keys if query["tool"] in self.shapes else {}
        said = not query["rows"] or query["pages_read"] > 1 or "total" in keys or "next" in keys
        reached = len(query["rows"]) >= (query["total"] or 0)
        query["all_seen"] = said and reached and not self.last_fault
        cover = self.board.cover[line]
        index = cover["queries"].index(query) if query in cover["queries"] else -1
        if index not in cover["ended"]:
            cover["ended"].append(index)

    def _note_filter(self, tool: Tool, query: dict[str, Any], rows: list[str]) -> None:
        limit = query["price_max"]
        if limit is not None and any((self.board.listings[item_id].price or 0) > limit for item_id in rows):
            self._price_ignored.add((tool.name, tool.role))
            self.ops.note("price_filter_ignored", tool=tool.name, role=tool.role)

    def _note_direction(self, query: dict[str, Any]) -> None:
        if query["role"] != "search" or not query["words"] or query["fault"] or query["shelf"] or query["price_max"] is not None:
            return
        words = store_tokens(query["words"])
        for other in self._word_queries:
            theirs = store_tokens(other["words"])
            short, long = (other, query) if theirs < words else (query, other) if words < theirs else (None, None)
            if short and short["total"] and (long["total"] > short["total"] or long["total"] < STORE_AND_DROP * short["total"]):
                seen = "or" if long["total"] > short["total"] else "and"
                if self._direction not in (None, seen, "mixed"):
                    self.ops.note("direction_dropped", words=query["words"])
                self._direction = seen if self._direction in (None, seen) else "mixed"
        self._word_queries.append(query)

    def _note_shelf(self, line: int, rows: list[str]) -> None:
        target = next((item for item in self.board.lines if item.number == line), None)
        shelves = sorted({self.board.listings[item_id].shelf for item_id in rows if self.board.listings[item_id].shelf})
        if target is None or not shelves:
            return
        name, request = entry_words(target.name), entry_words(self.board.request)
        fit = lambda shelf: (entry_words(shelf) == name, len(name & entry_words(shelf)), len(request & entry_words(shelf)))
        known = fit(target.shelf) if target.shelf else (False, 0, 0)
        best = max(shelves, key=fit)
        if fit(best)[1] and fit(best) > known:
            target.shelf = best
        if target.shelf:
            rivals = [shelf for shelf in shelves if shelf != target.shelf and fit(shelf) == fit(target.shelf)]
            (self.board.shelf_open.add if rivals else self.board.shelf_open.discard)(line)

    def search_direction(self) -> str | None:
        return self._direction if self._direction in ("and", "or") else None

    def _note_fields(self, query: dict[str, Any], rows: list[str]) -> None:
        if self._fields or query["role"] != "search" or not query["words"]:
            return
        for item_id in rows[:STORE_SAMPLE_ROWS * 4]:
            listing = self.board.listings.get(item_id)
            if listing is None or not listing.facts:
                continue
            plain = store_tokens(" ".join(str(value) for value in (listing.row or {}).values() if not isinstance(value, (dict, list))) + f" {listing.title} {listing.shelf}")
            if any(word in store_tokens(" ".join(map(str, listing.facts.values()))) and word not in plain for word in store_tokens(query["words"])):
                self._fields = "values"
                self.ops.note("search_fields", learned="values", words=query["words"])
                return

    def search_notes(self) -> str:
        direction = self.search_direction()
        notes = ["Every word of a search must match a listing: added words narrow the list, and a word no listing carries empties it." if direction == "and"
                 else "Any word of a search may match: added words widen the list." if direction == "or"
                 else "Whether every word must match is not known yet: a search with one word more or fewer than another teaches it."]
        if self._fields == "values":
            notes.append("Table values are searched: a value a row shows in its table finds the listing.")
        queries = [query for cover in self.board.cover.values() for query in cover["queries"] if query.get("role") == "search" and query.get("rows")]
        if queries:
            widest = max(queries, key=lambda query: len(query["rows"]) // max(1, query["pages_read"]))
            reach, size = self.reach(widest), len(widest["rows"]) // max(1, widest["pages_read"])
            if reach != float("inf"):
                notes.append(f"A list shows at most {int(reach)} rows ({size} a page): rows past them never show, so narrow the search rather than page on.")
        empties = [f'"{query["words"]}"' for cover in self.board.cover.values() for query in cover["queries"] if query.get("words") and not query.get("total")]
        if empties:
            notes.append("Searches that matched nothing: " + ", ".join(empties[-6:]) + ".")
        for name, mode in sorted(self._words_ignored):
            reads = [self._mode_of(tool) for tool in self.board.tools.values() if tool.name == name and self._arg(tool, "words") and (name, self._mode_of(tool)) not in self._words_ignored]
            notes.append(f"{name} with {mode!r} lists the whole shelf whatever the words (different words gave the same rows)" + (f"; words count only with {reads[0]!r}." if reads else "."))
        return "\n".join(notes)

    def has_price_filter(self) -> bool:
        tools = [tool for tool in map(self.board.tool, ("search", "browse")) if tool]
        usable = [tool for tool in tools if (tool.name, tool.role) not in self._price_ignored]
        return any(self._arg(tool, "price_max") in self._props(tool.name) for tool in usable)

    def under_price(self, line: int, price: float) -> bool:
        queries = self.board.cover[line]["queries"]
        found = [query for query in queries if query.get("role") == "search" and query.get("words") and query.get("total")]
        if not self.has_price_filter() or not found:
            return False
        words = min(found, key=lambda query: query["total"])["words"]
        self.search(line, words, "", price - 0.01)
        return any(query.get("words") == words and query.get("price_max") is not None for query in queries)

    def note_words_ignored(self, query: dict[str, Any]) -> None:
        tool = next((item for item in self.board.tools.values() if item.name == query["tool"] and self._arg(item, "words")), None)
        if tool is None or not query.get("total"):
            return
        words_arg, page = self._arg(tool, "words"), self._arg(tool, "page")
        rest = {name: value for name, value in query["args"].items() if name not in (words_arg, page)}
        words = str(query["args"].get(words_arg, ""))
        for other in (item for cover in self.board.cover.values() for item in cover["queries"] if item is not query and item["tool"] == query["tool"]):
            theirs = str(other["args"].get(words_arg, ""))
            if (words or theirs) and theirs != words and other.get("total") == query["total"] and {n: v for n, v in other["args"].items() if n not in (words_arg, page)} == rest:
                mode = str(query["args"].get(tool.args.get("mode", ""), ""))
                if (tool.name, mode) not in self._words_ignored:
                    self._words_ignored.add((tool.name, mode))
                    self.ops.note("words_ignored", tool=tool.name, mode=mode)
                return

    def reach(self, query: dict[str, Any]) -> float:
        spec = self._props(query["tool"]).get(self._arg(self.board.tool(query["role"]), "page")) or {}
        last, size = spec.get("maximum"), max(1, len(query["rows"]) // max(1, query["pages_read"]))
        return float("inf") if last is None else last + size if spec.get("minimum") == 0 else last * size

    def _seen_ids(self, ids: list[str], line: int | None) -> list[str]:
        known = [item_id for item_id in dict.fromkeys(ids) if item_id in self.board.listings]
        if len(known) < len(set(ids)):
            self.ops.note("read_unseen", ids=[item_id for item_id in ids if item_id not in self.board.listings][:8])
        if line is not None:
            self._link(line, known)
        return known

    def _attribute_tool(self) -> Tool | None:
        facts, page = self.board.tool("facts"), self.board.tool("page")
        return page if page and self.shows_table(page.name) and self.shows_text(page.name) else facts or page

    def has_record(self, item_id: str) -> bool:
        records, opener = self.board.listings[item_id].records, self._attribute_tool()
        views = [tool.name for tool in map(self.board.tool, ("facts", "page", "compare")) if tool]
        return bool(opener and opener.name in records) or any(name in records and self.shows_table(name) for name in views)

    def page_open(self, item_id: str) -> bool:
        listing, page, facts = self.board.listings.get(item_id), self.board.tool("page"), self.board.tool("facts")
        if listing is None:
            return False
        if page:
            return page.name in listing.records or any(self._shows_all_text(name, page.name) for name in listing.records)
        return any(self.shows_text(name) or not facts or name == facts.name for name in listing.records)

    def _compare_room(self, wanted: int) -> int:
        compare = self.board.tool("compare")
        if compare is None or wanted < 2 or not self._arg(compare, "ids") or (compare.name in self._shows and not self.shows_table(compare.name)):
            return 0
        spec = self._props(compare.name).get(self._arg(compare, "ids")) or {}
        stated = [int(found) for found in re.findall(r"\d+", (self.ops.schemas.get(compare.name) or {}).get("description") or "") if int(found) >= 2]
        room = min(stated + ([spec["maxItems"]] if "maxItems" in spec else []), default=STORE_COMPARE_GUESS)
        return min(room, wanted) if room >= 2 else 0

    def _read_batch(self, compare: Tool, batch: list[str]) -> None:
        self.call(compare.name, {self._arg(compare, "ids"): batch})

    def read(self, ids: list[str], line: int | None = None) -> list[str]:
        targets = self._seen_ids(ids, line)
        while True:
            todo = [item_id for item_id in targets if not self.has_record(item_id) and item_id not in self._unreadable]
            if not todo or not self._walk_room():
                break
            size = self._compare_room(len(todo))
            opener = self._attribute_tool()
            if size:
                self._read_batch(self.board.tool("compare"), todo[:size])
            elif opener is None:
                self._fail("the store has no tool that opens a listing")
                break
            elif self.call(opener.name, {self._arg(opener, "id"): todo[0]}) is None:
                break
            elif self._attribute_tool() is opener and not self.has_record(todo[0]):
                self._unreadable.add(todo[0])
        return [item_id for item_id in targets if self.has_record(item_id)]

    def read_page(self, ids: list[str]) -> list[str]:
        done = []
        for item_id in self._seen_ids(ids, None):
            tool = self.board.tool("page") or self.board.tool("facts")
            while tool and not self.page_open(item_id) and self._walk_room():
                self.call(tool.name, {self._arg(tool, "id"): item_id})
                moved = self.board.tool("page") or self.board.tool("facts")
                tool = moved if moved is not tool else None
            if self.page_open(item_id):
                done.append(item_id)
        return done

    def read_pages(self, ids: list[str]) -> list[str]:
        compare, page = self.board.tool("compare"), self.board.tool("page")
        todo = [item_id for item_id in self._seen_ids(ids, None) if not self.page_open(item_id)]
        untried = compare and page and (compare.name not in self._shows or self._shows_all_text(compare.name, page.name))
        if untried and len(todo) > 1 and self._arg(compare, "ids") and self._walk_room():
            self._read_batch(compare, todo[:STORE_PAGES_TOGETHER])
        return self.read_page(todo[:1])

    def _fresh_view(self) -> Tool | None:
        def shows(tool: Tool) -> float:
            seen = self._shows.get(tool.name)
            return 1.5 if seen is None else len(seen & {"price", "stock"})
        return max((tool for tool in map(self.board.tool, ("page", "facts")) if tool), key=shows, default=None)

    def look_again(self, ids: list[str]) -> None:
        for item_id in self._seen_ids(ids, None):
            tool = self._fresh_view()
            if tool and self._walk_room():
                self.call(tool.name, {self._arg(tool, "id"): item_id})


    def _cart_mode(self, tool: Tool) -> str:
        props = self._props(tool.name)
        if tool.args.get("mode") in props:
            return tool.args["mode"]
        return next((arg for arg, spec in props.items() if spec.get("enum")), "")

    def _cart_mode_value(self, tool: Tool, removing: bool) -> Any:
        choices = self._props(tool.name).get(self._cart_mode(tool), {}).get("enum") or [None]
        labelled = tool.args.get("remove" if removing else "add")
        return labelled if labelled in choices else choices[-1 if removing else 0]

    def _cart_list_keys(self, tool: Tool) -> tuple[str, str, str]:
        arg = next((arg for arg, spec in self._props(tool.name).items() if "array" in store_types(spec)), "")
        fields = ((self._props(tool.name).get(arg) or {}).get("items") or {}).get("properties") or {}
        id_key = next((key for key, spec in fields.items() if "string" in store_types(spec)), "")
        qty_key = next((key for key, spec in fields.items() if {"integer", "number"} & set(store_types(spec))), "")
        return arg, id_key, qty_key

    def _cart_meaning(self, tool: Tool) -> str:
        if not self._meaning:
            if self._cart_list_keys(tool)[0]:
                self._meaning = "list"
            elif self._cart_mode(tool):
                self._meaning = "add"
            else:
                self._meaning = "set" if self._arg(tool, "qty") else "add"
        return self._meaning

    def _cart_each(self, tool: Tool) -> int:
        spec = self._props(tool.name).get(self._arg(tool, "qty"))
        return 1 if spec is None else int(spec.get("maximum") or 10 ** 6)

    def _cart_effect(self, tool: Tool, args: dict[str, Any]) -> dict[str, int]:
        meaning = self._cart_meaning(tool)
        cart: dict[str, int] = defaultdict(int)
        if meaning == "list":
            arg, id_key, qty_key = self._cart_list_keys(tool)
            for entry in args.get(arg) or []:
                held = entry if isinstance(entry, dict) else {id_key: entry}
                cart[str(held.get(id_key))] += int(number(held.get(qty_key)) or 1)
            return dict(cart)
        cart.update(self.board.cart)
        item_id = str(args.get(self._arg(tool, "id")))
        count = int(number(args.get(self._arg(tool, "qty"))) or (0 if meaning == "set" else 1))
        mode = self._cart_mode(tool)
        removing = bool(mode) and args.get(mode) == self._cart_mode_value(tool, True)
        if meaning == "set":
            cart[item_id] = 0 if removing else count
        else:
            cart[item_id] += -count if removing else count
        return {key: value for key, value in cart.items() if value > 0}

    def _cart_seen(self, cart: dict[str, int], total: float | None) -> None:
        moved = cart != self.board.cart or (None not in (total, self.board.cart_total) and total != self.board.cart_total)
        self.board.cart, self.board.cart_total = cart, total
        if moved:
            self.board.cart_version += 1

    def _cart_from_lines(self, shape: StoreShape, items: list[dict[str, Any]]) -> dict[str, int] | None:
        cart: dict[str, int] = defaultdict(int)
        for line in items:
            count = number(store_field(shape, "qty", line))
            item_id = self._item_id(shape, line)
            if count is None or not item_id:
                self.ops.note("cart_unreadable", tool=shape.tool, keys=shape.keys)
                return None
            listing = self.board.listings.setdefault(item_id, Listing(item_id, seen_step=self.ops.step))
            shown = line if count == 1 else {key: value for key, value in line.items() if key != shape.keys.get("price")}
            self._apply_fields(listing, shape, shown, None, False)
            if count > 0:
                cart[item_id] += int(count)
        return dict(cart)

    def _absorb_cart(self, roles: set[str], shape: StoreShape, body: Any, items: list[Any] | None, args: dict[str, Any]) -> None:
        cart = self._cart_from_lines(shape, items) if items is not None else None
        if cart is not None:
            self._cart_seen(cart, number(store_field(shape, "total", body)))
            self._cart_unsure = False
            if "cart" in roles:
                self._cart_shows = "cart"
            return
        writer = self.board.tool("cart")
        if "cart" not in roles or writer is None:
            return
        count, item_id = number(store_field(shape, "qty", body)), self._item_id(shape, body)
        if items is None and count is not None and item_id:
            seen = {**self.board.cart, item_id: int(count)}
            self._cart_seen({key: value for key, value in seen.items() if value > 0}, None)
            self._cart_shows = "line"
        else:
            self._cart_seen(self._cart_effect(writer, args), None)
            self._cart_shows = "nothing"
        self._cart_unsure = True

    def _absorb_nothing(self, tool: str, roles: set[str], body: Any) -> None:
        if isinstance(body, dict):
            empty = self._blanks.setdefault(tool, {key for key, value in body.items() if value in (None, "", [], {})})
            self._hear([sentence for key in empty for sentence in store_sentences(body.get(key))])
        if roles & STORE_CARTS and isinstance(body, dict):
            amounts = [value for value in body.values() if store_is_amount(value)]
            self._cart_seen({}, amounts[0] if len(amounts) == 1 else None)
            self._cart_unsure = False
            if "cart" in roles:
                self._cart_shows = "cart"
        elif "message" not in roles:
            self._last["total"] = 0

    def _cart_look(self) -> None:
        view = self.board.tool("view")
        if self._cart_unsure and view:
            self.call(view.name, {})

    def _cart_args(self, tool: Tool, item_id: str, count: int, removing: bool) -> dict[str, Any]:
        args: dict[str, Any] = {self._arg(tool, "id"): item_id}
        if self._arg(tool, "qty"):
            args[self._arg(tool, "qty")] = count
        if self._cart_mode(tool):
            args[self._cart_mode(tool)] = self._cart_mode_value(tool, removing)
        return args

    def _cart_write(self, tool: Tool, args: dict[str, Any]) -> bool:
        return self.call(tool.name, args) is not None and not self.last_fault

    def _cart_write_all(self, tool: Tool, want: dict[str, int]) -> bool:
        meaning = self._cart_meaning(tool)
        if meaning == "list":
            arg, id_key, qty_key = self._cart_list_keys(tool)
            lines: list[Any] = []
            for item_id, count in want.items():
                lines += [{id_key: item_id, qty_key: count}] if qty_key else [item_id] * count
            return self._cart_write(tool, {arg: lines})
        each = self._cart_each(tool)
        for item_id in sorted(self.board.cart.keys() | want.keys(), key=lambda key: key in want):
            have, target = self.board.cart.get(item_id, 0), want.get(item_id, 0)
            if have != target and meaning == "set":
                if not self._cart_write(tool, self._cart_args(tool, item_id, target, not target and bool(self._cart_mode(tool)))):
                    return False
            elif have != target:
                left = target - have
                while left:
                    part = min(abs(left), each)
                    if not self._cart_write(tool, self._cart_args(tool, item_id, part, left < 0)):
                        return False
                    left -= part if left > 0 else -part
        return True

    def cart_set(self, want: dict[str, int]) -> bool:
        tool = self.board.tool("cart")
        if tool is None:
            self._fail("the store has no cart tool")
            return False
        target = {item_id: int(count) for item_id, count in want.items() if count > 0}
        self._planned = None
        for _ in range(STORE_CART_ROUNDS):
            if self.ops.steps_left() >= self.steps_for(target) + (self._cart_writes(tool, target) > 0):
                self._cart_look()
            if self.board.cart == target:
                return True
            if self.ops.steps_left() < self._cart_writes(tool, target) + 1:
                self._fail("too few steps left to change the cart")
                return False
            written = self._cart_write_all(tool, target)
            fault = self.last_fault
            if self.ops.steps_left() > 1:
                self._cart_look()
            if self.board.cart == target:
                return True
            if not written:
                self.last_fault = fault
                return False
        return False

    def _cart_writes(self, tool: Tool, target: dict[str, int]) -> int:
        meaning, cart = self._cart_meaning(tool), self.board.cart
        moves = [abs(cart.get(item_id, 0) - target.get(item_id, 0)) for item_id in cart.keys() | target.keys()]
        moves = [move for move in moves if move]
        if meaning == "list":
            return int(bool(moves))
        if meaning == "set":
            return len(moves)
        return sum(-(-move // self._cart_each(tool)) for move in moves)

    def steps_for(self, want: dict[str, int]) -> int:
        tool = self.board.tool("cart")
        if tool is None:
            return 1
        writes = self._cart_writes(tool, {item_id: int(count) for item_id, count in want.items() if count > 0})
        look = self.board.tool("view") and (self._cart_unsure or (writes and self._cart_shows != "cart"))
        return writes + (1 if look else 0) + 1

    def plan_cart(self, want: dict[str, int] | None) -> None:
        self._planned = want

    def _lines_cart(self) -> dict[str, int]:
        cart, listings, want = self.board.cart, self.board.listings, {}
        for line in self.board.lines:
            held = [item_id for item_id, count in cart.items() if count == line.qty and item_id not in want
                    and line.number in getattr(listings.get(item_id), "lines", ())]
            want[held[0] if held else f"line {line.number}"] = line.qty
        return want

    def steps_to_finish(self) -> int:
        return self.steps_for(self._lines_cart() if self._planned is None else self._planned)


    def _finish_args(self, tool: Tool, job: str, text: str) -> dict[str, Any]:
        args = self._mode_args(tool, job)
        text_arg = self._arg(tool, "text")
        if text and text_arg:
            args[text_arg] = text
        for name in self._params(tool.name).get("required") or []:
            spec = self._props(tool.name).get(name) or {}
            kinds = set(store_types(spec))
            if name in args or spec.get("enum"):
                continue
            if "boolean" in kinds:
                args[name] = True
            elif {"integer", "number"} & kinds and self.board.cart_total is not None:
                args[name] = self.board.cart_total
            elif "string" in kinds:
                args[name] = text or job
        return args

    def order(self) -> dict[str, Any] | None:
        tool = self.board.tool("order")
        if tool is None:
            self._fail("the store has no order tool")
            return None
        return self.call(tool.name, self._finish_args(tool, "order", ""), ending=True)

    def decline(self, text: str) -> dict[str, Any] | None:
        tool = self.board.tool("decline")
        if tool is None:
            self._fail("the store has no decline tool")
            return None
        for item_id in sorted(self.board.listings, key=len, reverse=True):
            text = text.replace(item_id, "")
        return self.call(tool.name, self._finish_args(tool, "decline", " ".join(text.split())), ending=True)


    def render(self, tool: str, response: dict[str, Any] | None) -> str:
        if response is None:
            return f"NOT SENT: {self.last_fault or 'no call was made'}. No step was spent."
        observation, refusal = store_envelope(response)
        body, riding = self._peel(tool, observation)
        roles, shape = self._roles(tool), self.shapes.get(tool)
        refused = (store_sentences(refusal) or [dump(refusal)]) if refusal else []
        out = [f"REFUSED: {sentence}" for sentence in refused]
        out += [f"NOTICE: {sentence}" for sentence in riding if sentence not in refused]
        if shape and isinstance(body, dict):
            strays = [value for key, value in body.items() if key not in shape.top]
            out += [f"NOTICE: {sentence}" for sentence in store_sentences(strays)]
        if refused:
            out.append("The finish did not go through and the episode goes on; the sentences above changed something.")
        if response.get("finished"):
            return "\n".join(out + ["FINISHED: the episode is over."])
        fault = self._fault_text(tool, body)
        if fault:
            shut = isinstance(body, dict) and body.get("error") == "budget_exceeded"
            hint = "This tool stays shut; do not call it again." if shut else "A step was spent. Change the call before resending it."
            return "\n".join(out + [f"FAULT: {cut_text(fault, STORE_RAW_CHARS)}", hint])
        if "message" in roles:
            return "\n".join(out + [f"REPLY: {store_reply(body)}"])
        items = self._items(shape, body) if shape else None
        if roles & STORE_CARTS and (shape is None or items is not None or "qty" in shape.keys):
            out += self._render_cart(shape, body, items)
        elif items is not None and "id" in shape.keys:
            out += self._render_rows(shape, body, items)
        elif shape and isinstance(body, dict) and "id" in shape.keys and not roles & STORE_FINISHES:
            out += self._render_record(shape, body)
        elif store_row_lists(body) and not any(store_row_lists(body).values()):
            out.append("NO ROWS for this query. That says nothing about the catalog; other words may find rows.")
        else:
            out.append(f"RESULT: {cut_text(dump(body), STORE_RAW_CHARS)}")
            if "cart" in roles and self.board.tool("view"):
                out.append("The write's result does not show the basket; the view tool does.")
        return "\n".join(out)

    def _render_cart(self, shape: StoreShape | None, body: Any, items: list[dict[str, Any]] | None) -> list[str]:
        if shape is None or items is None:
            return [f"BASKET: {dump(self.board.cart) if self.board.cart else 'empty'}; total {self.board.cart_total}"]
        out = [f"BASKET: {len(items)} lines, total {store_field(shape, 'total', body)}"]
        for line in items:
            cells = [store_field(shape, job, line) for job in ("id", "qty", "price", "title")]
            out.append(f"{cells[0]} | units {cells[1]} | price {cells[2]} | {cells[3]}")
        return out

    def _render_rows(self, shape: StoreShape, body: Any, items: list[dict[str, Any]]) -> list[str]:
        if not items:
            return ["NO ROWS for this query. That says nothing about the catalog; other words may find rows."]
        total, shown = store_field(shape, "total", body), store_field(shape, "next", body)
        counted = total if total is not None else "an unknown number"
        paging = "a next page exists" if shown not in (None, "") else "no next page shown"
        out = [f"ROWS: {len(items)} shown of {counted}; {paging}"]
        out.append("id | price | in stock | shelf | shop | title")
        for item in items:
            cells = [store_field(shape, job, item) for job in ("id", "price", "stock", "shelf", "shop", "title")]
            out.append(" | ".join("" if cell is None else str(cell) for cell in cells))
            for key in shape.texts:
                text = store_get(item, key)
                if isinstance(text, str) and text.strip():
                    out.append(f"  {key}: {cut_text(text, STORE_ROW_TEXT_CHARS)}")
            for table in shape.tables:
                out += [f"  {name}: {value}" for name, value in store_flat_table(store_get(item, table[0]), *table[1:3]).items()]
        return out

    def _render_record(self, shape: StoreShape, body: dict[str, Any]) -> list[str]:
        out = [f"LISTING {self._item_id(shape, body)}"]
        shown = {shape.keys.get("id")}
        for job in ("title", "price", "stock", "shelf", "shop"):
            value = store_field(shape, job, body)
            if value is not None:
                out.append(f"{job}: {value}")
                shown.add(shape.keys[job])
        for table in shape.tables:
            shown.add(table[0])
            out += [f"{name}: {value}" for name, value in store_flat_table(store_get(body, table[0]), *table[1:3]).items()]
        for key in shape.texts:
            shown.add(key)
            text = store_get(body, key)
            if isinstance(text, str) and text.strip():
                out.append(f"{key}: {cut_text(text, STORE_TEXT_CHARS)}")
        rest = [key for key in body if key not in shown and key in shape.top and body[key] not in (None, "", [], {})]
        return out + [f"{key}: {cut_text(dump(body[key]), STORE_ROW_TEXT_CHARS)}" for key in rest]

class Purse:

    def __init__(self, ops: Ops, board: Board, store: Store):
        self.ops, self.board, self.store = ops, board, store
        self.closer: Any = None
        self.basket: Callable[[], dict[int, str] | None] = lambda: None
        self.closing = self.refused = self.quiet = False

    def owed(self) -> int:
        return self.closer.steps_owed() if self.closer else 0

    def repair(self) -> int:
        if self.refused or any(notice.kind == "refusal" for notice in self.board.notices):
            return 0
        if len(self.board.lines) < 2:
            return REPAIR_STEPS
        return max(REPAIR_FLOOR, math.ceil((REPAIR_SHARE_QUIET if self.quiet else REPAIR_SHARE) * self.ops.max_steps))

    def close_cost(self) -> int:
        basket = self.basket()
        if basket and all(basket.values()) and self.closer:
            pages = sum(not self.store.page_open(item_id) for item_id in set(basket.values()))
            return self.store.steps_for(self.closer.plan_of(basket)) + pages + len(set(basket.values()))
        coming = {f"line {line.number}": line.qty for line in self.board.lines}
        return self.store.steps_for(coming) + 2 * len(coming)

    def reserves(self) -> dict[str, int]:
        return {"close": self.close_cost(), "owed": self.owed(), "repair": self.repair()}

    def walk(self) -> int:
        return self.ops.steps_left() - sum(self.reserves().values())

    def close_room(self) -> int:
        return self.ops.steps_left() - self.store.steps_to_finish() - self.owed()

    def room(self) -> int:
        return self.close_room() if self.closing else self.walk()

CARDS_PER_CALL = 60
CARD_TEXT_CHARS, CARD_FIELD_CHARS, CARD_TOP_ROWS, CARD_SPLIT_WORDS = 12000, 60, 11, 3
VIEW_TITLE_CHARS, VIEW_QUOTE_CHARS = 140, 90
MENU_ROWS, MENU_VALUES, MENU_VALUE_CHARS = 30, 3, 50


EVIDENCE_STATES = {"met": "HAS", "differs": "OTHER", "silent": "SILENT", "split": "CONFLICT"}
EVIDENCE_PLACES = ("entry", "entry and title", "entry against title", "title", "option", "text", "fits", "nowhere")


@functools.lru_cache(maxsize=8192)
def show_money(value: float | None) -> str:
    return "?" if value is None else f"{value:.10g}"


def readable_wants(board: Board, line: int) -> list[Want]:
    return [w for w in board.wants_of(line) if w.value or not w.ref]


def record_table(listing: Listing) -> str:
    if not listing.facts and not listing.notes and listing.records:
        return cut_text(dump(listing.records), CARD_TEXT_CHARS)
    entries = dict(listing.facts)
    for source in (listing.row, *listing.records.values()):
        for name, value in source.items() if isinstance(source, dict) else ():
            text = str(value)
            known = text in (listing.id, listing.title, *listing.notes.split("\n"))
            known = known or (listing.price is not None and number(value) == listing.price)
            if type(value) in (str, int, float) and 0 < len(text) <= CARD_FIELD_CHARS and not known:
                entries.setdefault(name, text)
    return "\n".join(f"{name}: {value}" for name, value in sorted(entries.items()))


def record_sheet(listing: Listing) -> str:
    return (f"title: {listing.title}\nprice: {show_money(listing.price)}\ntable:\n{record_table(listing)}\n"
            f"text: {cut_text(listing.notes, CARD_TEXT_CHARS)}")


def evidence_row(row: list[str]) -> tuple[str, str, str, str, str, str]:
    cells = row[1:] + ["", ""]
    closed = [place for place in range(3, len(cells)) if join_words(cells[place]) in EVIDENCE_STATES]
    state = ([place for place in closed if join_words(cells[place - 1]) in EVIDENCE_PLACES] or closed or [2])[-1]
    where = state - 1 if state > 3 and join_words(cells[state - 1]) in EVIDENCE_PLACES else state
    quote = " | ".join(cells[2:where])
    return (cells[0], cells[1], "" if close_none(quote) else quote, join_words(cells[where]) if where < state else "",
            join_words(cells[state]), " | ".join(filter(None, cells[state + 1:])))


def product_words(board: Board, line: int) -> set[str]:
    item = board.line(line)
    return entry_words(f"{item.name} {item.shelf}")


def entry_words(name: Any) -> set[str]:
    words = split_words(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name)))
    stems = (re.sub(r"ies$", "y", re.sub(r"(ch|sh|x|ss|z)es$", r"\1", word)) for word in words if len(word) > 2)
    return {stem[:-1] if len(stem) > 3 and stem.endswith("s") and not stem.endswith("ss") else stem for stem in stems}


def named_as(name: Any, want: Want, product: set[str]) -> bool:
    words, prop = entry_words(name), entry_words(want.prop)
    return bool(prop) and (words == prop or words - product == prop - product != set())


def value_words(value: Any) -> set[str]:
    return entry_words(value) | {word for word in split_words(value) if len(word) <= 2}


def named_entries(listing: Listing, want: Want, product: set[str]) -> list[str]:
    fuller = entry_words(want.prop)
    return [str(value) for key, value in listing.facts.items() if str(value).strip() and (named_as(key, want, product) or (bool(fuller) and fuller <= entry_words(key)))]


def named_against(listing: Listing, want: Want, quote: str, product: set[str]) -> str:
    fuller = entry_words(want.prop)
    if want.rel not in ("is", "one of") or named_as(quote.partition(":")[0], want, product) or (bool(fuller) and fuller <= entry_words(quote.partition(":")[0])):
        return ""
    named = named_entries(listing, want, product)
    forms = [form for form in map(value_words, want_forms(want)) if form]
    return next(iter(named), "") if forms and not any(form <= value_words(value) for form in forms for value in named) else ""


def singular(word: str) -> str:
    lower = word.casefold()
    if len(word) > 4 and lower.endswith("ies"):
        return f"{word[:-3]}y"
    if re.search(r"(ch|sh|x|ss|z)es$", lower):
        return word[:-2]
    return word[:-1] if len(word) > 3 and lower.endswith("s") and not lower.endswith("ss") else word


def product_heads(line: Line) -> list[str]:
    parts = [part.strip() for part in re.split(r",|\bor\b|\band\b", line.name) if part.strip()]
    parts = [side.strip() for part in parts for side in (part.split("&") if all(side.strip().casefold().endswith("s") for side in part.split("&")) else [part])]
    heads = list(dict.fromkeys(" ".join([*part.split()[:-1], singular(part.split()[-1])]) for part in parts if part.split()))
    story = split_words(" ".join(line.story))
    return sorted(heads, key=lambda head: -sum(word.startswith(split_words(head)[-1]) for word in story)) if heads else [line.name]


def value_first(head: str, values: list[str]) -> str:
    words = head.split()
    return " ".join([*sorted(values, key=lambda value: len(value.split())), *words[-1:], *words[:-1]])


def squeezed(values: list[str]) -> list[str]:
    return [" ".join(max(reversed(value.split()), key=len) for value in values)] * (sum(" " in value for value in values) > 1)


def want_values(want: Want) -> list[str]:
    if want.rel == "one of" or re.search(r"\bor\b", want.value):
        return [part.strip() for part in re.split(r",|\bor\b|\band\b", want.value) if part.strip()]
    return [want.value]


def want_forms(want: Want) -> list[str]:
    parts = want_values(want)
    name = entry_words(want.prop)
    bare = [" ".join(word for word in re.split(r"[\s-]+", part) if entry_words(word) - name or not entry_words(word)) for part in parts]
    return list(dict.fromkeys(form for form in parts + bare if form))


def shelf_holds(listing: Listing, want: Want) -> bool:
    forms = [entry_words(part) for part in want_values(want)] if want.rel in ("is", "one of") else []
    return any(form and form <= entry_words(listing.shelf) for form in forms)


def promote_uses(ops: Ops, board: Board, line: int) -> None:
    product = product_words(board, line)
    for want in board.soft_of(line):
        forms = entry_words(want.value)
        carried = any(line in item.lines and not off_shelf(board.line(line), item) and named_as(key, want, product) and forms <= entry_words(value)
                      for item in board.listings.values() for key, value in item.facts.items())
        if forms and carried:
            want.soft = False
            board.version += 1
            ops.note("use_became_want", line=line, prop=want.prop, value=want.value)


def price_bars(board: Board, ln: Line, item: Listing) -> list[str]:
    over_line = item.price is not None and ln.cap is not None and item.price > ln.cap
    over_order = item.price is not None and board.total_cap is not None and item.price * ln.qty > board.total_cap
    return ["over cap"] * (over_line or over_order) + ["out of stock"] * (item.stock is False)


def off_shelf(ln: Line, item: Listing) -> bool:
    return bool(ln.shelf and item.shelf) and squash(ln.shelf) != squash(item.shelf)


def shelf_bars(board: Board, ln: Line, item: Listing) -> bool:
    return off_shelf(ln, item) and ln.number not in board.shelf_open


def _view_title_shows(board: Board, ln: Line, item: Listing) -> tuple[list[str], list[str]]:
    title, wanted, excluded = set(split_words(item.title)), set(split_words(ln.name)), set()
    for want in board.wants_of(ln.number):
        (excluded if want.rel == "not" else wanted).update(split_words(want.value))
    return sorted(title & wanted), sorted(title & excluded)


def rank(board: Board, line: int) -> list[str]:
    ln = board.line(line)

    def place(lid: str) -> tuple:
        item = board.listings[lid]
        price = float("inf") if item.price is None else item.price
        shown, excluded = _view_title_shows(board, ln, item)
        hits = len(shown) - len(excluded)
        failed = board.verdicts.get((lid, line, board.version), (True,))[0] is False
        order = (price, -hits) if ln.goal == "cheapest" else (-hits, price)
        return (bool(price_bars(board, ln, item)), off_shelf(ln, item), failed, *order, lid)

    return sorted((lid for lid, item in board.listings.items() if line in item.lines), key=place)


def promise(board: Board, line: int, lid: str, groups: list[list[str]], every: bool) -> int:
    title = squash(board.listings[lid].title)
    lists = [(squash(query.get("words") or ""), query["rows"].index(lid)) for query in board.cover[line]["queries"] if lid in query["rows"]]

    def shown(form: str) -> bool:
        return bool(form) and (form in title or any(form == words or (form in words and (every or place < CARD_TOP_ROWS)) for words, place in lists))

    return sum(any(shown(squash(value)) for value in group) for group in groups)


def twin_ids(board: Board, line: int, lid: str) -> list[str]:
    title = join_words(board.listings[lid].title)
    return [i for i, item in board.listings.items() if i != lid and line in item.lines and title and join_words(item.title) == title]


def twin_state(board: Board, line: int, lid: str) -> int:
    twins = twin_ids(board, line, lid)
    verdicts = [board.verdicts[key][0] for key in ((i, line, board.version) for i in twins) if key in board.verdicts]
    return -1 if any(verdicts) else 1 if verdicts or any(board.listings[i].read for i in twins) else 0


def named_values(board: Board, line: int, item: Listing) -> dict[str, list[str]]:
    product = product_words(board, line)
    return {want.key: named_entries(item, want, product) for want in readable_wants(board, line)}


def row_holds(board: Board, lid: str, groups: list[list[str]]) -> int:
    item = board.listings[lid]
    text = f" {join_words(' '.join([item.title, *(value for value in item.row.values() if isinstance(value, str))]))} " if isinstance(item.row, dict) else f" {join_words(item.title)} "
    return sum(any(f" {join_words(value)} " in text for value in group) for group in groups)


def reading_key(board: Board, line: int, lid: str, groups: list[list[str]], every: bool) -> tuple:
    item, queries = board.listings[lid], board.cover[line]["queries"]
    place = min((query["rows"].index(lid) for query in queries if lid in query["rows"]), default=float("inf"))
    return twin_state(board, line, lid) > 0, -max(row_holds(board, lid, groups), promise(board, line, lid, groups, every)), (item.price if item.price is not None else float("inf")) if board.line(line).goal == "cheapest" else place


def _view_lines(board: Board, lines: list[int] | None) -> list[Line]:
    return [ln for ln in board.lines if lines is None or ln.number in lines]


def line_head(ln: Line) -> str:
    cap = f", line cap {show_money(ln.cap)}" if ln.cap is not None else ""
    shelf = f", shelf {ln.shelf}" if ln.shelf else ""
    return f"line {ln.number}: {ln.name}, qty {ln.qty}, goal {ln.goal}{cap}{shelf}"


def want_text(want: Want) -> str:
    value = want.value or (f"as on another listing: {want.ref} (open)" if want.ref else "any value")
    return f"{want.key} {want.prop} {want.rel} {value} ({want.source})"


def _view_status(board: Board, line: int, lid: str) -> str:
    verdict = board.verdicts.get((lid, line, board.version))
    return "row" if not board.listings[lid].read else "read" + (f", judge {'pass' if verdict[0] else 'fail'}" if verdict else "")


def _view_flags(board: Board, ln: Line, item: Listing) -> str:
    flags = price_bars(board, ln, item) + [f"other shelf {item.shelf}"] * off_shelf(ln, item)
    return ", ".join(flags + [f"in cart {board.cart.get(item.id)}"] * (item.id in board.cart))


def _view_title_note(board: Board, ln: Line, item: Listing) -> str:
    shown, excluded = _view_title_shows(board, ln, item)
    good = [f"title shows {' '.join(shown)}"] * bool(shown)
    return "; ".join(good + [f"title shows excluded {' '.join(excluded)}"] * bool(excluded))


def _view_cheaper_unread(board: Board, ln: Line) -> tuple[str | None, list[Listing]]:
    line, ids = ln.number, rank(board, ln.number)
    carted = [lid for lid in ids if lid in board.cart]
    passed = [lid for lid in ids if board.verdicts.get((lid, line, board.version), (False,))[0]]
    pick = next(iter(carted + passed), None)
    ceiling = board.listings[pick].price if pick else None
    rows = [board.listings[lid] for lid in ids]
    cheaper = [item for item in rows if not item.read and item.price is not None and not price_bars(board, ln, item)
               and not off_shelf(ln, item) and (ceiling is None or item.price < ceiling)]
    return pick, sorted(cheaper, key=lambda item: (item.price, item.id))


def skim_view(board: Board, line: int, limit: int = 30) -> str:
    ln = board.line(line)
    ids = rank(board, line)
    out = [f"{line_head(ln)}, {len(ids)} rows"]
    for lid in ids[:limit]:
        item = board.listings[lid]
        parts = [lid, show_money(item.price), _view_status(board, line, lid), _view_title_note(board, ln, item),
                 _view_flags(board, ln, item), cut_text(item.title, VIEW_TITLE_CHARS)]
        out.append("  " + " | ".join(part for part in parts if part))
    return "\n".join(out + [f"  ({len(ids) - limit} more rows not shown)"] * (len(ids) > limit))


def _view_menu_status(board: Board, ln: Line, names: set[str]) -> str:
    marks = ["asked"] * bool(names & {join_words(name) for name in ln.asked})
    marks += ["closed"] * bool(names & {join_words(name) for name in ln.closed})
    marks += ["told"] * bool(names & {join_words(w.prop) for w in board.wants_of(ln.number)})
    ties = [t for t in board.ties if t.kind == "condition" and ln.number in (t.a, t.b)]
    conditions = [f" {join_words(t.rule + ' ' + t.sentence)} " for t in ties]
    marks += ["used by a condition"] * any(f" {name} " in text for name in names for text in conditions)
    return ", ".join(marks)


def cover_view(board: Board, lines: list[int] | None = None) -> str:
    out = []
    for ln in _view_lines(board, lines):
        out.append(line_head(ln))
        for query in board.cover[ln.number]["queries"]:
            pages = f"pages read {query.get('pages_read')}, {'ended' if query.get('ended') else 'more pages left'}"
            out.append(f"  search \"{query.get('words')}\": total {query.get('total')}, {pages}")
        pick, cheaper = _view_cheaper_unread(board, ln)
        lead = f"pick {pick} at {show_money(board.listings[pick].price)}" if pick else "no pick yet"
        tail = f"{len(cheaper)} unread rows cheaper, lowest {show_money(cheaper[0].price)} ({cheaper[0].id})" if cheaper else ""
        out.append(f"  {lead}: {tail or 'no unread row cheaper'}")
    return "\n".join(out)

ASK_MENU_ONE, ASK_MENU_KINDS, ASK_MENU_WIDE, ASK_WIDE_STEPS, ASK_MENU_PAGES, ASK_READ_SHARE = 6, 3, 5, 20, 1, 3
ASK_MENU_ROWS, ASK_MENU_VALUES, ASK_VALUE_CHARS, NOTICE_LISTINGS = 10, 2, 50, 40
ASK_KIND_OVERLAP, ASK_PART_CHARS = 0.5, 5
ASK_NO_ATTRIBUTE = ("", "duty")

ASK_ROWS = {"ASK": 3}
ASK_NAMES, ASK_CLOSE_NAMES, ASK_ONE_VALUE_KINDS = 5, 1, 8
HEAR_ROWS = {"NEED": 5, "REF": 5, "TIE": 5, "NONE": 1, "LOST": 1, "ONE": 1, "CHANGE": 2}
NOTICE_ROWS = {"CAP": 3, "DROP": 4, "PRICE": 3, "STOCK": 3, "LIMIT": 3, "NEED": 5, "REF": 5, "SAME": 1}


@dataclass
class AskAttribute:
    spelling: str
    names: set[str]
    held: int
    listings: int
    values: list[str]


def ask_figures_hold(value: str, source: str) -> bool:
    digits = re.sub(r"\D", "", source)
    return all(run.rstrip("0") in digits for run in re.findall(r"\d+", value))


def ask_figure_problem(row: list[str], source: str) -> str:
    place = {"NEED": 4, "REF": 4, "CAP": 2, "PRICE": 2}.get(row[0])
    return "" if place is None or ask_figures_hold(row[place], source) else ASK_FIGURE_PROBLEM


def ask_words(attribute: str) -> str:
    return " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", attribute.replace("_", " ")).lower().split())


def ask_question(line: Line, names: list[str]) -> str:
    words = [ask_words(name) for name in names]
    listed = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " or " + words[-1]
    return f"Which {listed} would you like for the {line.shelf or line.name}?"


def ask_stands_in(words: list[str], others: list[str]) -> bool:
    joined = "".join(others)
    return all(word in others or (len(word) >= ASK_PART_CHARS and word in joined) for word in words)


def ask_same_family(one: str, other: str) -> bool:
    words, others = split_words(one), split_words(other)
    return bool(words and others) and (ask_stands_in(words, others) or ask_stands_in(others, words))


def ask_one_attribute(one: str, other: str, tables: list[dict[str, str]]) -> bool:
    agree = all(table[one] == table[other] for table in tables if one in table and other in table)
    return squash(one) == squash(other) or (agree and ask_same_family(one, other))


def ask_alike(one: set[str], other: set[str]) -> bool:
    return bool(one | other) and len(one & other) >= ASK_KIND_OVERLAP * len(one | other)


def ask_kinds(items: list[Listing]) -> list[list[Listing]]:
    kinds: list[list[Listing]] = []
    for item in items:
        words = set(split_words(item.title))
        home = next((kind for kind in kinds if ask_alike(words, set(split_words(kind[0].title)))), None)
        if home is None:
            kinds.append([item])
        else:
            home.append(item)
    return kinds


class Interviewer:
    def __init__(self, ops: Ops, board: Board, store: Store):
        self.ops, self.board, self.store = ops, board, store
        self.plan: list[tuple[int, str, bool]] = []
        self.planned_on: tuple[Any, ...] | None = None
        self.kinds_wanted = ASK_MENU_KINDS
        self.read_room = 0
        self.tried: set[str] = set()
        self.dry: set[int] = set()
        self.pages: dict[int, int] = defaultdict(int)
        self.refused_many = False
        self.bundled: dict[int, int] = defaultdict(int)

    def interview(self, lines: list[int] | None = None, messages: int | None = None) -> None:
        known = [line.number for line in self.board.lines]
        targets = known if lines is None else [line_no for line_no in lines if line_no in known]
        lines_held = len(self.board.lines)
        sent, self.kinds_wanted = 0, ASK_MENU_ONE if lines_held == 1 else ASK_MENU_WIDE if self.ops.max_steps / lines_held >= ASK_WIDE_STEPS else ASK_MENU_KINDS
        self.read_notices()
        self.read_room = self.ask_read_room()
        while targets and (messages is None or sent < messages) and self.ask_may_send():
            self.ask_fill(targets)
            free = self.board.msgs_free()
            self.ask_plan(targets, free if messages is None else min(messages - sent, free))
            pick = self.ask_pick(targets)
            if not pick:
                self.kinds_wanted += ASK_MENU_KINDS
                if self.ask_fill(targets):
                    continue
                break
            names = self.ask_names(pick[0], pick[1])
            text = ask_question(self.board.line(pick[0]), names)
            reply = self.store.say(text)
            if reply is None:
                break
            sent += 1
            self.hear(pick[0], names, text, reply)
            self.read_notices()

    def ask_names(self, line_no: int, first: str) -> list[str]:
        names = [first]
        for name in [row[1] for row in self.plan if row[0] == line_no] + self.ask_attributes(line_no):
            if len(names) < ASK_NAMES and self.ask_free(line_no, name) and not any(ask_same_family(name, held) for held in names):
                names.append(name)
        return names[:1] if self.refused_many else names

    def hear(self, line: int | None, names: list[str] | str, question: str, reply: str) -> None:
        names = [names] if isinstance(names, str) else names
        attr = names[0]
        shelves = "\n".join(f"{item.number}: {item.shelf or item.name}" for item in self.board.lines)
        named = attr not in ASK_NO_ATTRIBUTE
        asked = (f"line {line}, attribute {ask_words(attr)}" + (f", also naming {', '.join(map(ask_words, names[1:]))}" if len(names) > 1 else "")) if named else "no attribute: the message served another purpose"
        view = tag("job", "reply") + tag("lines", shelves) + tag("asked", asked) + tag("question", question) + tag("reply", reply)
        rows = self.ask_seat("HEAR", LISTEN, view, HEAR_ROWS, lambda row: ask_figure_problem(row, reply), ("plan", "decide"))
        kind = rows[0][0].lower() if rows else "unread"
        self.ask_entry(question).update(line=line, attr=attr, names=names, kind=kind, changed=any(row[0] == "CHANGE" for row in rows or []))
        heard = self.ask_line(line)
        told = any(row[0] in ("NEED", "REF") for row in rows or [])
        if heard and named:
            for name in names[:1 if told else ASK_CLOSE_NAMES]:
                heard.asked[name] = reply
                if not told:
                    heard.closed.add(name)
        if len(names) > 1 and not told:
            self.bundled[line or 0] += 1
        for row in rows or []:
            self.ask_apply(row, reply, line or 0, attr)
        self.ops.note("heard", line=line, attr=attr, names=names, reading=kind)

    def read_notices(self) -> bool:
        version = self.board.version
        self.ask_hear_talk()
        unread = [item for item in self.board.notices if not item.read]
        if not unread:
            return self.board.version != version
        source = " ".join(item.text for item in unread)
        refused = any(item.kind == "refusal" for item in unread)
        check = lambda row: ASK_REFUSAL_PROBLEM if refused and row[0] == "SAME" else ask_figure_problem(row, source)
        rows = self.ask_seat("NOTICE", LISTEN, self.ask_notice_view(unread), NOTICE_ROWS, check, ("decide",), True)
        if rows is None:
            return self.board.version != version
        for item in unread:
            item.read = True
        self.ops.note("notices_read", rows=rows)
        changes = [self.ask_apply(row, source) for row in rows]
        return any(changes) or self.board.version != version

    def ask_hear_talk(self) -> None:
        for entry in [entry for entry in self.board.talk if entry.get("kind") in ("", "unread") and entry.get("reply")]:
            self.hear(entry.get("line"), entry.get("attr") or "", entry["text"], entry["reply"])
            if entry["kind"] != "unread" and not entry.get("changed"):
                for item in self.board.notices:
                    if not item.read and item.kind == "notice" and join_words(item.text) in join_words(entry["reply"]):
                        item.read = True

    def ask_may_send(self) -> bool:
        return (self.board.msgs_free() > 0 and self.ops.may_think("decide") and self.ops.elapsed() < TIME_CLOSE
                and self.store.purse.close_room() > 0)

    def ask_seat(self, seat: str, prompt: str, view: str, heads: dict[str, int], check: Callable[[list[str]], str] = lambda row: "",
                 kinds: tuple[str, ...] = ("plan",), heavy: bool = False) -> list[list[str]] | None:
        rows = None
        for attempt in range(2):
            reply = next(filter(None, (self.ops.think(seat, prompt, view, kind, heavy) for kind in kinds)), None)
            if reply is None:
                return rows
            seen = [row for row in tag_rows(reply.text) if row[0] in heads]
            rows = [row for row in seen if len(row) >= heads[row[0]]]
            problems = list(filter(None, map(check, rows)))
            if attempt or (rows and len(rows) == len(seen) and not problems):
                break
            view += tag("format_problem", problems[0] if problems else ASK_FORMAT_PROBLEM)
        if not rows:
            self.ops.note("seat_unusable", seat=seat)
        return rows

    def ask_read_room(self) -> int:
        return (self.store.purse.close_room() - self.board.msgs_free()) // ASK_READ_SHARE

    def ask_fill(self, targets: list[int]) -> bool:
        grown = False
        while self.read_room > 0 and not self.ops.finished:
            sizes = {line_no: len(ask_kinds(self.ask_sample(line_no))) for line_no in targets if line_no not in self.dry}
            thin = [line_no for line_no, size in sizes.items() if size < self.kinds_wanted]
            if not thin:
                break
            line_no, start = min(thin, key=sizes.get), self.ops.step
            rows = self.ask_new_kinds(line_no)
            if rows:
                batch = rows[:self.kinds_wanted - sizes[line_no]] if self.board.tool("compare") else rows[:1]
                self.tried.update(batch)
                grown = bool(self.store.read(batch, line_no)) or grown
            elif not self.ask_more_rows(line_no):
                self.dry.add(line_no)
            self.read_room -= self.ops.step - start
        return grown

    def ask_sample(self, line_no: int) -> list[Listing]:
        line = self.board.line(line_no)
        held = [item for item in self.board.listings.values() if line_no in item.lines and item.facts]
        return [item for item in held if not off_shelf(line, item)] or held

    def ask_new_kinds(self, line_no: int) -> list[str]:
        line, picked = self.board.line(line_no), []
        seen = [set(split_words(item.title)) for item in self.ask_sample(line_no)]
        for lid in rank(self.board, line_no):
            item, words = self.board.listings[lid], set(split_words(self.board.listings[lid].title))
            fresh = not (item.facts or lid in self.tried or off_shelf(line, item))
            if fresh and not any(ask_alike(words, other) for other in seen):
                seen.append(words)
                picked.append(lid)
        return picked

    def ask_more_rows(self, line_no: int) -> bool:
        if self.pages[line_no] >= ASK_MENU_PAGES:
            return False
        self.pages[line_no] += 1
        line, queries = self.board.line(line_no), self.board.cover[line_no]["queries"]
        if not queries:
            return bool(self.store.search(line_no, line.shelf or line.name, line.shelf))
        wide = [place for place, query in enumerate(queries) if not query.get("ended")]
        return bool(wide) and bool(self.store.more(line_no, max(wide, key=lambda place: queries[place].get("total") or 0)))

    def ask_menu(self, line_no: int) -> list[AskAttribute]:
        tables: list[dict[str, str]] = []
        spellings: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        values: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        holders: dict[str, set[int]] = defaultdict(set)
        for place, kind in enumerate(ask_kinds(self.ask_sample(line_no))):
            for item in kind:
                table = {}
                for key, value in item.facts.items():
                    name, text = join_words(key), str(value or "").strip()
                    if name and text:
                        table[name] = squash(text)
                        spellings[name][key] += 1
                        values[name][text] += 1
                        holders[name].add(place)
                tables.append(table)
        groups: list[set[str]] = []
        for name in sorted(holders, key=lambda name: (-len(holders[name]), name)):
            home = next((group for group in groups if all(ask_one_attribute(name, other, tables) for other in group)), None)
            if home is None:
                groups.append({name})
            else:
                home.add(name)
        menu = []
        for names in groups:
            written, said = defaultdict(int), defaultdict(int)
            for name in names:
                for key, count in spellings[name].items():
                    written[key] += count
                for text, count in values[name].items():
                    said[text] += count
            held = len(set().union(*(holders[name] for name in names)))
            menu.append(AskAttribute(max(sorted(written, key=lambda key: (len(ask_words(key).split()), key)), key=written.get), names, held, sum(written.values()),
                                     sorted(said, key=lambda text: (-said[text], text))))
        menu = [entry for entry in menu if not (len(entry.values) == 1 and entry.held >= ASK_ONE_VALUE_KINDS)]
        return sorted(menu, key=lambda entry: (len(entry.values) == 1, -entry.held, -entry.listings, entry.spelling))

    def ask_menu_view(self, targets: list[int]) -> str:
        out = []
        for line_no in targets:
            line = self.board.line(line_no)
            kinds = len(ask_kinds(self.ask_sample(line_no)))
            prices = sorted(item.price for item in self.board.listings.values()
                            if line_no in item.lines and item.price is not None and not off_shelf(line, item))
            usual = f", usual price {show_money(prices[len(prices) // 2])}" if prices else ""
            out.append(f"{line_head(line)}{usual}, {kinds} kinds of listing read: attribute | held by | example values | status")
            marked = [(_view_menu_status(self.board, line, entry.names), entry) for entry in self.ask_menu(line_no)]
            marked.sort(key=lambda pair: bool(pair[0]))
            for status, entry in marked[:ASK_MENU_ROWS]:
                said = ", ".join(cut_text(value, ASK_VALUE_CHARS) for value in entry.values[:ASK_MENU_VALUES])
                out.append(f"  {entry.spelling} | {entry.held}/{kinds} | {said} | {status or 'open'}")
            rest = [f"{entry.spelling} {entry.held}/{kinds}{f' {status}' if status else ''}" for status, entry in marked[ASK_MENU_ROWS:]]
            out += [f"  also: {'; '.join(rest)}"] * bool(rest)
        return "\n".join(out)

    def ask_plan(self, targets: list[int], room: int) -> None:
        menus = tuple(frozenset(entry.spelling for entry in self.ask_menu(line_no)) for line_no in targets)
        state = (tuple(targets), tuple(want.key for want in self.board.wants if want.alive and want.source != "shopper"), menus)
        if state == self.planned_on:
            return
        so_far = "\n".join(f"{row[0]} | {row[1]}" for row in self.plan) or "none"
        view = (tag("request", self.board.request) + tag("order", self.ask_order_view() + self.ask_story_view())
                + tag("menu", self.ask_menu_view(targets)) + tag("plan_so_far", so_far)
                + tag("free_messages", room) + tag("lines_to_ask", ", ".join(map(str, targets))))
        rows = self.ask_seat("ASK" if len(self.board.lines) == 1 else "ASK LINES", ASK, view, ASK_ROWS, heavy=self.planned_on is None)
        self.planned_on = state
        if rows:
            picks = [pick for pick in (self.ask_resolve(row, targets) for row in rows) if pick]
            self.plan = sorted(picks, key=lambda pick: self.ask_stated(pick[0], pick[1]))
        self.ops.note("ask_plan", plan=list(self.plan), room=room)

    def ask_resolve(self, row: list[str], targets: list[int]) -> tuple[int, str, bool] | None:
        line_no = self.ask_number(row[1], None)
        if line_no not in targets:
            return None
        menu, said = self.ask_menu(line_no), join_words(row[2])
        found = [entry for entry in menu if said in entry.names]
        found = found or [entry for entry in menu if any(f" {name} " in f" {said} " for name in entry.names)]
        found = found or [entry for entry in menu if any(ask_same_family(said, name) for name in entry.names)]
        shared = [entry for entry in menu if any(len(word) >= ASK_PART_CHARS and word in split_words(" ".join(entry.names)) for word in split_words(said))]
        found = found or shared[:1] if len(shared) == 1 else found
        linked = len(row) > 3 and row[3].casefold().startswith("story")
        attribute = found[0].spelling if found else "" if menu and not linked else row[2]
        return (line_no, attribute, linked) if attribute and self.ask_free(line_no, attribute) else None

    def ask_pick(self, targets: list[int]) -> tuple[int, str] | None:
        remaining = {line_no: self.ask_attributes(line_no) for line_no in targets}
        rows = [row for row in self.plan if row[0] in remaining and self.ask_free(row[0], row[1])]
        waiting = {row[0] for row in rows} | {line_no for line_no in targets if remaining[line_no]}
        if not waiting:
            return None
        place = {line_no: (sum(want.source == "shopper" for want in self.board.wants_of(line_no)), len(self.board.line(line_no).asked) + self.bundled[line_no], line_no)
                 for line_no in targets}
        turn = min(place[line_no] for line_no in waiting)
        unasked = sum(not place[line_no][1] for line_no in waiting)
        jump = self.board.msgs_free() > unasked
        pick = next((row for row in rows if (row[2] and (jump or not place[row[0]][1])) or place[row[0]] == turn), rows[0] if rows and turn[1] else None)
        if pick is None and rows:
            pick = min(rows, key=lambda row: (place[row[0]], rows.index(row)))
        if pick:
            self.plan.remove(pick)
            return pick[:2]
        ranked = [(place[line_no], order, line_no, attribute) for line_no in targets for order, attribute in enumerate(remaining[line_no])]
        return min(ranked)[2:]

    def ask_attributes(self, line_no: int) -> list[str]:
        roles = " ".join(join_words(f"{tie.rule} {tie.sentence}") for tie in self.board.ties
                         if tie.kind == "condition" and line_no in (tie.a, tie.b))
        ranked = sorted(self.ask_menu(line_no), key=lambda entry: (any(f" {name} " in f" {roles} " for name in entry.names),
                                                                   self.ask_stated(line_no, entry.spelling), -entry.held, -entry.listings, entry.spelling))
        return [entry.spelling for entry in ranked if self.ask_free(line_no, entry.spelling)]

    def ask_stated(self, line_no: int, attribute: str) -> bool:
        return any(want.value and ask_same_family(want.prop, attribute) for want in self.board.wants_of(line_no))

    def ask_free(self, line_no: int, attribute: str) -> bool:
        line = self.board.line(line_no)
        spent = {join_words(item) for item in (*line.closed, *line.asked)}
        spent |= {join_words(want.prop) for want in self.board.wants_of(line_no) if want.source == "shopper"}
        return not any(ask_same_family(attribute, name) for name in spent)

    def ask_line(self, line_no: int | None) -> Line | None:
        return next((line for line in self.board.lines if line.number == line_no), None)

    def ask_number(self, text: str, fallback: int | None) -> int | None:
        found = re.search(r"\d+", text)
        line_no = int(found.group()) if found else fallback
        return line_no if line_no == 0 or self.ask_line(line_no) else fallback

    def ask_entry(self, text: str) -> dict[str, Any]:
        return next((entry for entry in reversed(self.board.talk) if entry.get("text") == text), {})

    def ask_listing(self, text: str) -> Listing | None:
        listings, cart = self.board.listings, self.board.cart
        return listings.get(text) or (listings.get(next(iter(cart))) if len(cart) == 1 else None)

    def ask_want(self, row: list[str], line: int, attr: str, source: str) -> bool:
        owner, rel, value = self.ask_number(row[1], line), row[3] or "is", row[4]
        prop = self.ask_named(owner, row[2] or attr, source)
        refers = row[0] == "REF"
        quote = row[5] if len(row) > 5 and row[5] and join_words(row[5]) in join_words(source) else source
        for want in self.board.wants_of(owner):
            same = bool(want.ref) if refers else join_words(want.value or want.ref) == join_words(value)
            if same and join_words(want.prop) == join_words(prop) and want.rel == rel:
                return False
        self.board.add_want(owner, prop, rel, "" if refers else value, "shopper", quote, value if refers else "")
        return True

    def ask_named(self, line: int, prop: str, reply: str) -> str:
        said = f" {join_words(reply)} "
        if f" {join_words(prop)} " in said:
            return prop
        named = {key for item in self.board.listings.values() if line in item.lines for key in item.facts if f" {join_words(key)} " in said}
        return max(sorted(named), key=len, default=prop)

    def ask_tie(self, row: list[str], line: int, reply: str) -> bool:
        owner, other = self.ask_number(row[1], line), self.ask_number(row[2], line)
        if owner == other:
            return False
        self.board.ties.append(Tie(owner, other, row[3], row[4], reply))
        self.board.version += 1
        return True

    def ask_file_change(self, sentence: str, reply: str) -> None:
        text = sentence if sentence and join_words(sentence) in join_words(reply) else reply
        name = join_words(text)
        if not any(name in join_words(item.text) or join_words(item.text) in name for item in self.board.notices):
            self.board.notices.append(Notice(text, self.ops.step))

    def ask_order_view(self) -> str:
        rows = [f"whole order: {want_text(want)}" for want in self.board.wants_of(0)]
        for line in self.board.lines:
            rows.append(line_head(line))
            rows += [f"  {want_text(want)}" for want in self.board.wants_of(line.number)]
        return "\n".join(rows)

    def ask_story_view(self) -> str:
        readers: dict[str, list[str]] = defaultdict(list)
        for line in self.board.lines:
            for sentence in line.story:
                readers[sentence].append(str(line.number))
        return "".join(f"\nstory for line {', '.join(numbers)}: {sentence}" for sentence, numbers in readers.items())

    def ask_notice_view(self, unread: list[Notice]) -> str:
        board = self.board
        words = set(split_words(" ".join(item.text for item in unread)))
        shared = lambda item: len(words & set(split_words(f"{item.id} {item.title}")))
        near = sorted(board.listings.values(), key=lambda item: (item.id not in board.cart, -shared(item)))
        rows = [f"{item.id} | price {item.price} | in stock {item.stock} | in cart {board.cart.get(item.id, 0)} | {item.title}"
                for item in near[:NOTICE_LISTINGS]]
        limits = (f"whole-order budget {board.total_cap}; messages used {board.msg_used} of {board.msg_limit}; "
                  f"steps used {self.ops.step} of {self.ops.max_steps}")
        sentences = "\n".join(f"[{item.kind}, step {item.step}] {item.text}" for item in unread)
        return (tag("job", "notices") + tag("order", self.ask_order_view()) + tag("limits", limits)
                + tag("listings", "\n".join(rows)) + tag("notices", sentences))

    def ask_apply(self, row: list[str], source: str, line: int = 0, attr: str = "") -> bool:
        head, board = row[0], self.board
        if head in ("NEED", "REF"):
            return self.ask_want(row, line, attr or row[2], source)
        if head == "TIE":
            return self.ask_tie(row, line, source)
        if head == "CHANGE":
            self.ask_file_change(row[1], source)
        elif head == "ONE":
            self.refused_many = True
        elif head in ("NONE", "LOST") and attr not in ASK_NO_ATTRIBUTE and self.ask_line(line):
            board.line(line).closed.add(attr)
        elif head == "CAP":
            written = [number(found) for found in re.findall(r"\d[\d,]*(?:\.\d+)?", source)]
            amount = number(row[2])
            return self.ask_set_cap(self.ask_number(row[1], None), written[0] if amount not in written and len(written) == 1 else amount)
        elif head == "DROP":
            want = next((item for item in board.wants if item.key == row[1] and item.alive), None)
            owner = self.ask_line(want.line if want else self.ask_number(row[2], None))
            if owner:
                owner.closed.add(want.prop if want else row[3])
            if want:
                want.alive = False
                board.version += 1
                return True
        elif head in ("PRICE", "STOCK"):
            item = self.ask_listing(row[1])
            if item:
                if head == "PRICE":
                    item.price = number(row[2])
                else:
                    item.stock = row[2].casefold() == "in"
                return True
        elif head == "LIMIT":
            found = re.search(r"\d+", row[2])
            if found:
                if row[1].casefold() == "messages":
                    board.msg_limit = int(found.group())
                else:
                    self.ops.max_steps = int(found.group())
                return True
        return False

    def ask_set_cap(self, owner: int | None, amount: float | None) -> bool:
        if owner is None or amount is None:
            return False
        if owner:
            self.board.line(owner).cap = amount
        else:
            self.board.total_cap = amount
        self.board.version += 1
        return True

TRACE_LINE = 0
TRACE_CARDS, TRACE_ASKS = 12, 16
TRACE_ROW_CHARS, TRACE_RECORD_CHARS = 1500, 8000
TRACE_SEARCH_STEPS = 2
TRACE_COPIED = 0.7


def trace_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {trace_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(map(trace_text, value))
    return "" if value is None else str(value)


def trace_share(text: str, words: str) -> float:
    wanted = word_set(words)
    return len(wanted & word_set(text)) / len(wanted) if wanted else 0.0


def trace_holds(text: str, words: str) -> bool:
    return trace_share(text, words) == 1.0


def trace_fields(seen: Any) -> str:
    if not isinstance(seen, dict):
        return dump(seen)
    fields = [(key, item if isinstance(item, str) else dump(item)) for key, item in seen.items() if item not in (None, "")]
    return "\n".join(f"{key}: {text}" for key, text in sorted(fields, key=lambda pair: len(pair[1])))


def trace_row(rows: list[list[str]], head: str) -> list[str]:
    return next((row[1:] for row in reversed(rows) if row[0] == head), [])


def trace_says(fields: list[str]) -> bool:
    return bool(fields) and fields[0].lower() not in ("", "none")


def trace_ids(fields: list[str], shown: list[str]) -> list[str]:
    cells = [cell.strip(".:") for cell in re.split(r"[\s,;|\"'`*()\[\]]+", " ".join(fields))]
    return [cell for cell in dict.fromkeys(cells) if cell in shown]


@dataclass
class TraceLink:
    words: str
    rejected: set[str]
    aimed: bool = False
    shop: str = ""
    kind: str = ""
    facts: list[str] = field(default_factory=list)
    narrow: list[str] = field(default_factory=list)
    wide: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)
    found: dict[str, str] = field(default_factory=dict)
    sight: list[tuple[str, bool, bool]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    latest: str = ""
    tabled: set[str] = field(default_factory=set)
    answered: bool = False


class Tracer:

    def __init__(self, ops: Ops, board: Board, store: Store):
        self.ops, self.board, self.store = ops, board, store
        self.pages: dict[str, list[str]] = {}
        self.ahead: dict[str, TraceLink] = {}
        self.ended: set[str] = set()

    def open_chains(self) -> list[Want]:
        return [want for want in self.board.wants if want.alive and want.ref and not want.value and want.key not in self.ended]

    def follow(self, want: Want, steps: int) -> bool:
        began, stop = self.ops.step, self.ops.step + steps
        for walked in self.pages.get(want.key, []):
            if self._left(stop) > 0 and not self.store.page_open(walked):
                self.store.read_page([walked])
        while not want.value and want.key not in self.ended:
            link = self.ahead.setdefault(want.key, TraceLink(want.ref, set(self.pages.get(want.key, []))))
            if not (link.aimed or self._ask_aim(want, link)) or not self._walk_link(want, link, stop):
                break
        self.ops.note("chain_state", want=want.key, state=self.chain_state(want), pages=self.pages.get(want.key, []),
                      steps=self.ops.step - began)
        return bool(want.value)

    def chain_state(self, want: Want) -> str:
        if want.value and self._links_open(want):
            return "resolved"
        return "exhausted" if want.key in self.ended else "unresolved"

    def chain_pages_open(self, want: Want) -> bool:
        return self._links_open(want) and self.chain_state(want) != "unresolved"

    def all_open(self) -> bool:
        return all(self.chain_pages_open(want) for want in self.board.wants if want.alive and want.ref)

    def _links_open(self, want: Want) -> bool:
        links = self.pages.get(want.key, [])
        return bool(links) and all(map(self.store.page_open, links))

    def _left(self, stop: int) -> int:
        if self.ops.elapsed() >= TIME_CLOSE:
            return 0
        return min(stop - self.ops.step, self.store.purse.room())

    def _ask_aim(self, want: Want, link: TraceLink) -> bool:
        reason = ""
        for _ in range(2):
            rows = self._ask(want, link, [], reason)
            if rows is None:
                return False
            self._aim(link, trace_row(rows, "AIM"), link.words)
            if link.aimed:
                return True
            reason = "Write one AIM row with words copied from the description."
        return False

    def _aim(self, link: TraceLink, fields: list[str], source: str) -> None:
        link.shop, link.kind, *facts = [text if trace_holds(source, text) else "" for text in fields + ["", ""]]
        link.facts = [text for text in facts if text]
        plain = [fact for fact in link.facts if not any(ch.isdigit() for ch in fact)]
        narrow = [" ".join([link.shop, *facts]) for facts in (plain, link.facts) if facts]
        narrow += [f"{link.shop} {fact}" for fact in sorted(link.facts, key=lambda fact: fact not in plain)]
        narrow = [" ".join(dict.fromkeys(words.split())) for words in narrow]
        wide = [f"{link.shop} {link.kind}", link.shop, " ".join([*link.facts, link.kind]),
                *(f"{link.kind} {fact}" for fact in link.facts)]
        link.narrow = list(dict.fromkeys(words.strip() for words in narrow))
        link.wide = [words for words in dict.fromkeys(words.strip() for words in wide) if words and words not in link.narrow]
        link.aimed = bool(link.narrow or link.wide)

    def _frontier(self, link: TraceLink) -> list[str]:
        listings, found = self.board.listings, list(link.found)
        if link.shop:
            found = ([i for i, item in listings.items() if trace_holds(item.shop, link.shop) or trace_holds(link.shop, item.shop)]
                     or [i for i, item in listings.items() if trace_holds(trace_text(item.row), link.shop)])
        texts = {i: self._record(i) for i in found if i not in link.rejected}
        first = {i: place for place, i in enumerate(link.found)}
        return sorted(texts, key=lambda i: (-sum(trace_share(texts[i], text) for text in [link.kind, *link.facts]),
                                            first.get(i, len(first))))[:TRACE_CARDS]

    def _turnable(self, link: TraceLink) -> int | None:
        queries = self.board.cover[TRACE_LINE]["queries"]
        return next((place for place in reversed(range(len(queries))) if not queries[place].get("ended")
                     and any(trace_holds(sent, queries[place].get("words", "")) for sent in link.sent)), None)

    def _can_widen(self, link: TraceLink) -> bool:
        return bool(link.narrow or link.wide) or self._turnable(link) is not None

    def _widen(self, want: Want, link: TraceLink, stop: int) -> bool:
        if self._left(stop) < TRACE_SEARCH_STEPS:
            return False
        place = None if link.narrow else self._turnable(link)
        queries, before = self.board.cover[TRACE_LINE]["queries"], set(self._frontier(link))
        if place is not None:
            words = queries[place]["words"]
            new = self.store.more(TRACE_LINE, place)
            if not (set(self._frontier(link)) - before):
                queries[place]["ended"] = True
        else:
            words = (link.narrow or link.wide).pop(0)
            link.sent.append(words)
            before_step = self.ops.step
            new = self.store.search(TRACE_LINE, words)
            link.answered = link.answered or self.ops.step > before_step
        for row_id in new:
            link.found.setdefault(row_id, words)
        return True

    def _table_done(self, link: TraceLink, chosen: str) -> bool:
        return chosen in link.tabled or self.store.has_record(chosen)

    def _view_left(self, link: TraceLink, chosen: str) -> bool:
        return not self.store.page_open(chosen) or not self._table_done(link, chosen)

    def _read_table(self, link: TraceLink, chosen: str) -> None:
        link.tabled.add(chosen)
        self.store.read([chosen])

    def _read_next(self, link: TraceLink, chosen: str, stop: int) -> bool:
        if self._left(stop) < 1:
            return False
        if self.store.page_open(chosen):
            self._read_table(link, chosen)
            return True
        link.latest = chosen
        self.store.read_pages(list(dict.fromkeys([chosen, *link.order, *self._frontier(link)])))
        if not self.store.page_open(chosen) and self._left(stop) > 0:
            link.rejected.add(chosen)
        return True

    def _advance(self, want: Want, link: TraceLink, shown: list[str], stop: int) -> bool:
        target = next((i for i in link.order if i in shown and self._view_left(link, i)), "")
        rest = [i for i in shown if not self.store.page_open(i)]
        if not target and self._can_widen(link) and (self._left(stop) >= TRACE_SEARCH_STEPS or not rest):
            return self._widen(want, link, stop)
        target = target or next(iter(rest), "")
        if target:
            return self._read_next(link, target, stop)
        if not link.answered:
            self.ops.note("chain_unanswered", want=want.key, fault=self.store.last_fault[:200])
            return False
        self.ended.add(want.key)
        self.ops.note("chain_end", want=want.key, pages=self.pages.get(want.key, []))
        return True

    def _record(self, chosen: str) -> str:
        item = self.board.listings[chosen]
        return trace_text([item.title, item.shop, item.shelf, item.row, item.records, item.facts, item.notes])

    def _card(self, link: TraceLink, chosen: str) -> str:
        item, page = self.board.listings[chosen], self.board.tool("page")
        views = [("page" if page and view == page.name else "table", seen) for view, seen in item.records.items()]
        views = views or [("row", item.row)]
        limit = TRACE_RECORD_CHARS if item.records else TRACE_ROW_CHARS
        shown = "\n".join(f"[{label}]\n{cut_text(trace_fields(seen), limit)}" for label, seen in views)
        opened = ("open" if done else "not open" for done in (self.store.page_open(chosen), self._table_done(link, chosen)))
        state = "page {}, table {}".format(*opened)
        found = f" | found by the search {link.found[chosen]}" if chosen in link.found else ""
        return f"{chosen} | shop {item.shop} | shelf {item.shelf} | {state}{found}\n{shown}\n"

    def _ask(self, want: Want, link: TraceLink, shown: list[str], reason: str) -> list[list[str]] | None:
        walked = [self.board.listings[i] for i in self.pages.get(want.key, [])]
        requirement = f"property: {want.prop}\nrelation: {want.rel}\nsentence: {want.sentence}"
        variable = (tag("requirement", requirement) + tag("description", link.words)
                    + tag("walked", "\n".join(f"{item.id}: {item.shop}, {item.title}" for item in walked) or "none")
                    + tag("candidates", "\n".join(self._card(link, i) for i in shown) or "none yet")
                    + (tag("problem", reason) if reason else ""))
        reply = self.ops.think("TRACE", TRACE, variable, kind="plan", heavy=bool(reason))
        if reply is None:
            return None
        return [[row[0].strip("*:"), *(cell.strip(" `\"'*") for cell in row[1:])] for row in tag_rows(reply.text)]

    def _walk_link(self, want: Want, link: TraceLink, stop: int) -> bool:
        reason = ""
        for _ in range(TRACE_ASKS):
            shown = self._frontier(link)
            whole = any(all(trace_holds(self._record(i), fact) for fact in link.facts) for i in shown)
            if not link.sent and not whole and self._can_widen(link):
                if not self._widen(want, link, stop):
                    return False
                continue
            sight = [(i, self.store.page_open(i), self._table_done(link, i)) for i in shown]
            if len(shown) == 1 and not sight[0][1]:
                link.order = shown
            elif shown and (reason or sight != link.sight):
                act, reason, link.order = self._consult(want, link, shown, reason, stop)
                if act == "wait":
                    return False
                link.sight = sight
                if act == "passed":
                    return True
                if act == "again":
                    continue
            if not self._advance(want, link, [i for i in shown if i not in link.rejected], stop):
                return False
            if want.key in self.ended:
                return True
        return False

    def _consult(self, want: Want, link: TraceLink, shown: list[str], reason: str, stop: int) -> tuple[str, str, list[str]]:
        rows = self._ask(want, link, shown, reason)
        if rows is None:
            return "wait", "", []
        pick, look, onward, value = (trace_row(rows, head) for head in ("PICK", "LOOK", "NEXT", "VALUE"))
        named, opened = trace_ids(pick, shown), [i for i in shown if self.store.page_open(i)]
        latest = link.latest if link.latest in opened else next(iter(opened), "")
        chosen = next(iter(named), latest if onward or value else "")
        if chosen in opened and (onward or value):
            return self._settle(want, link, chosen, onward, value, bool(reason), stop)
        missing = f"The page of {chosen} is open: write its NEXT row or its VALUE row." if chosen in opened else ""
        if not (pick or look):
            missing = "Write a LOOK row, or a PICK row with its NEXT row or its VALUE row."
        if missing:
            return ("wait", "", []) if reason else ("again", missing, [])
        order = list(dict.fromkeys(named + trace_ids(look, shown)))
        link.rejected.update(i for i in opened if i not in order or not self._view_left(link, i))
        if not order and shown and not opened:
            order = shown[:1]
        if not order:
            link.rejected.update(shown)
            aim = trace_row(rows, "AIM")
            if aim:
                self._aim(link, aim, link.words)
                link.narrow = [words for words in link.narrow if words not in link.sent]
                link.wide = [words for words in link.wide if words not in link.sent]
        return "go", "", order

    def _settle(self, want: Want, link: TraceLink, chosen: str, onward: list[str], value: list[str], retried: bool,
                stop: int) -> tuple[str, str, list[str]]:
        if not trace_says(onward) and not self._table_done(link, chosen):
            if self._left(stop) > 0:
                self._read_table(link, chosen)
                return "again", "", []
            if not trace_says(value):
                return "wait", "", []
        problem = self._problem(link, chosen, onward, value, retried)
        if not problem:
            self._pass_link(want, chosen, onward, value)
            return "passed", "", []
        if retried:
            link.rejected.add(chosen)
            problem = ""
        return "again", problem, []

    def _problem(self, link: TraceLink, chosen: str, onward: list[str], value: list[str], retried: bool) -> str:
        record = self._record(chosen)
        if not retried and link.facts and not any(trace_holds(record, fact) for fact in link.facts):
            return (f"The record of {chosen} shows none of the description's facts ({'; '.join(link.facts)}). "
                    "Find the card that does, else write LOOK with the cards still possible.")
        if trace_says(onward):
            if trace_share(record, onward[0]) < TRACE_COPIED:
                return f"The NEXT words must be copied from the page of {chosen}; not found there: {onward[0]}."
            return ""
        if not trace_says(value):
            return (f"You read {chosen} as neither giving the property nor sending on. "
                    "Reread its page for a sentence about the property, else write LOOK with the cards still possible.")
        copied = trace_holds(record, value[0])
        return "" if copied else f"The VALUE must be copied from the record of {chosen}; not found there: {value[0]}."

    def _pass_link(self, want: Want, chosen: str, onward: list[str], value: list[str]) -> None:
        walked = self.pages.setdefault(want.key, [])
        walked.append(chosen)
        if trace_says(onward):
            link = TraceLink(" | ".join(part for part in onward if part), set(walked))
            self._aim(link, onward[1:], self._record(chosen))
            self.ahead[want.key] = link
            return
        want.value, want.source = value[0], "chain"
        said = f": {value[1]}" if len(value) > 1 else ""
        want.sentence = f"{want.sentence} [value from listing {chosen}{said}]".strip()
        self.board.version += 1
        self.ops.note("chain_value", want=want.key, pages=walked, value=want.value)

RECORD_CHARS, RECORD_ARG_CHARS = 30000, 200
DUTY_TURNS, FINISH_TRIES, FIT_COMBINATIONS = 3, 3, 200000
SHARED_ROW_CHARS = 40
CLOSE_SECOND_LOOK, CLOSE_LOOKS, CLOSE_FINAL_TRIES = 2, 3, 2
CLOSE_PAGES = 3
JUDGE_STATES = {"met": "PASS", "differs": "FAIL", "silent": "NOT STATED", "split": "CONFLICT"}


def close_flat(value: Any, section: str = "", name: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        inner = f"{section}.{name}" if section and name else section or name
        return [row for key, item in value.items() for row in close_flat(item, inner, str(key))]
    if isinstance(value, list):
        return [row for place, item in enumerate(value, 1)
                for row in close_flat(item, section, f"{name} {place}" if isinstance(item, dict) else name)]
    return [] if value in (None, "") else [(section, f"{name}: {value}" if name else str(value))]


def close_original(listing: Listing) -> str:
    seen, parts = set(), []
    for source, record in (("search row", listing.row), *listing.records.items()):
        rows, at = [], ""
        for section, row in sorted(close_flat(record), key=lambda item: item[0]):
            words = section, join_words(row)
            if words not in seen and not row.endswith(f": {listing.id}"):
                seen.add(words)
                rows += [f"{section}:"] * (section != at) + [row]
                at = section
        if rows:
            parts.append(f"[{source}]\n" + "\n".join(rows))
    return cut_text("\n".join(parts) or listing.title, RECORD_CHARS)


def close_once(records: list[tuple[str, str]]) -> list[str]:
    first: dict[str, str] = {}
    out = []
    for name, record in records:
        rows = record.split("\n")
        for place, row in enumerate(rows):
            if len(row) >= SHARED_ROW_CHARS and first.setdefault(row, name) != name:
                head, colon, _ = row.partition(": ")
                rows[place] = f"{head}: same as {first[row]}" if colon else f"same as {first[row]}"
        out.append("\n".join(rows))
    return out


def close_none(quote: str) -> bool:
    return join_words(quote) in ("", "none")


def close_found(quote: str, words: str) -> bool:
    def there(part: str) -> bool:
        return any(join_words(form) and f" {join_words(form)} " in f" {words} " for form in (part, part.partition(":")[2]))
    parts = [part for part in re.split(r"\.\.\.|…|;", quote) if join_words(part)]
    return not close_none(quote) and all(there(part) for part in parts)


def close_id(cell: str) -> str:
    return cell.strip(" `'\"*")


def close_tie_row(index: int, tie: Tie) -> str:
    return f"{index} | line {tie.a} and line {tie.b} | {tie.kind} | {tie.rule} | {tie.sentence}"


def close_money(value: float | None) -> str:
    return f"{value or 0:.12g}"


def close_judge_rows(text: str, ids: dict[str, str]) -> tuple[dict[tuple[str, str], tuple[str, str, str]],
                                                              dict[tuple[int, str, str], bool]]:
    reads, pairs = {}, {}
    real = lambda cell: ids.get(close_id(cell), close_id(cell))
    for row in tag_rows(text):
        if row[0] == "REQ" and len(row) >= 4:
            listing, key, quote, where, state, _ = evidence_row(row)
            reads[real(listing), close_id(key)] = (quote, JUDGE_STATES.get(state, "NO VERDICT"), where)
        elif row[0] == "PAIR" and len(row) >= 5 and read_first_number(row[1]) is not None:
            tie, met = int(read_first_number(row[1])), join_words(row[-1]) == "met"
            pairs[tie, real(row[2]), real(row[3])] = met
            pairs.setdefault((tie, real(row[3]), real(row[2])), met)
    return reads, pairs


class Closer:
    def __init__(self, ops: Ops, board: Board, store: Store, tracer: Tracer):
        self.ops, self.board, self.store, self.tracer = ops, board, store, tracer
        self.calls: list[dict[str, Any]] = []
        self.breaks: dict[tuple[str, int, int], tuple[str, str]] = {}
        self.judged: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self.reread: set[tuple[Any, ...]] = set()
        self.reread_tries: dict[tuple[str, int], int] = defaultdict(int)
        self.looks = 0
        self.last_model = ""
        self.pairs_read: set[tuple[int, str, str]] = set()
        self.held: float | None = None
        self.record_calls()

    def record_calls(self) -> None:
        call = self.store.call

        def recorded(tool: str, args: dict[str, Any], ending: bool = False) -> dict[str, Any] | None:
            response = call(tool, args, ending)
            message = self.board.tool("message")
            if response is not None:
                reply = dump(response.get("observation")) if message and tool == message.name else ""
                self.calls.append({"step": self.ops.step, "tool": tool, "args": args, "reply": reply,
                                   "fault": self.store.last_fault, "cart": self.cart_words(), "version": self.board.cart_version})
            return response

        self.store.call = recorded

    def wants(self, line: int) -> list[Want]:
        return [want for want in self.board.wants_of(line) + self.board.wants_of(0) if want.value or not want.ref]

    def requirements(self, lines: list[int]) -> str:
        rows = []
        for number_ in lines:
            line = self.board.line(number_)
            shelf = f", on the store's shelf {line.shelf}" if line.shelf else ""
            rows.append(f"line {number_}: {line.name}{shelf}, quantity {line.qty}, goal {line.goal}")
            rows.append("item | the listing is this product")
            for want in self.wants(number_):
                ref = f"; the value is read from: {want.ref}" if want.ref else ""
                value = want.value or "value not known yet"
                said = "" if want.sentence in self.board.request else f": {want.sentence}"
                rows.append(f"{want.key} | {want.prop} | {want.rel} | {value} | from the {want.source}{said}{ref}")
        return "\n".join(rows)

    def listed(self, line: int) -> list[str]:
        return [id_ for id_, listing in self.board.listings.items() if line in listing.lines]

    def followed_pages(self) -> list[Listing]:
        ids = dict.fromkeys(id_ for walked in self.tracer.pages.values() for id_ in walked)
        return [self.board.listings[id_] for id_ in ids if id_ in self.board.listings]

    def buyable(self, id_: str, line: int) -> bool:
        listing, cap = self.board.listings.get(id_), self.board.line(line).cap
        return bool(listing) and listing.stock is not False and (cap is None or listing.price is None or listing.price <= cap)

    def plan_of(self, basket: dict[int, str]) -> dict[str, int]:
        qty = {line.number: line.qty for line in self.board.lines}
        if not basket or any(line not in qty or id_ not in self.board.listings for line, id_ in basket.items()):
            return {}
        plan: dict[str, int] = {}
        for line, id_ in basket.items():
            plan[id_] = plan.get(id_, 0) + qty[line]
        return plan

    def plan_total(self, plan: dict[str, int]) -> float:
        return sum((self.board.listings[id_].price or 0) * qty for id_, qty in plan.items() if id_ in self.board.listings)

    def cart_words(self) -> str:
        b = self.board
        total = "" if b.cart_total is None else f", total {close_money(b.cart_total)}"
        return ", ".join(f"{id_} x{qty}" for id_, qty in b.cart.items()) + total if b.cart else "empty"

    def order_words(self, plan: dict[str, int]) -> str:
        b, rows = self.board, []
        if not plan:
            return "No basket has been chosen yet."
        for id_, qty in plan.items():
            listing = b.listings.get(id_) or Listing(id_, id_)
            rows.append(f"{qty} x {listing.title or id_} at {close_money(listing.price)} each")
        total = b.cart_total if b.cart == plan and b.cart_total is not None else self.plan_total(plan)
        return f"The order: {'; '.join(rows)}. Total {close_money(total)}."

    def decline_words(self) -> str:
        names = ", ".join(line.name for line in self.board.lines)
        return f"I could not find anything that meets the request for {names}, so I am not placing an order."

    def spare_steps(self) -> int:
        return max(0, self.store.purse.close_room())

    def within_room(self, ids: list[str]) -> list[str]:
        room, kept = self.store.purse.room(), []
        for id_ in ids:
            if self.store.has_record(id_) or room > 0:
                kept.append(id_)
                room -= not self.store.has_record(id_)
        return kept

    def open_views(self, basket: dict[int, str]) -> None:
        for line, id_ in basket.items():
            self.store.read([id_], line)
        self.store.read_page(list(basket.values()))

    def decide(self, seat: str, fixed: str, variable: str | tuple[str, str], complete: Callable[[str], bool], heavy: bool = True) -> str:
        reply = self.ops.think(seat, fixed, variable, "decide", heavy=heavy)
        self.last_model = reply.model if reply else ""
        if reply is None or complete(reply.text):
            return reply.text if reply else ""
        if not self.ops.may_think("decide"):
            return reply.text
        reminder = tag("format", "Your last answer left rows out. Write every row of the output format.")
        asked = (variable[0], variable[1] + reminder) if isinstance(variable, tuple) else variable + reminder
        again = self.ops.call_model(seat, LIGHT if reply.model == HEAVY else HEAVY, fixed, asked, None)
        return reply.text + "\n" + (again.text if again else "")

    def read_again(self, picks: dict[int, list[str]]) -> bool:
        b, due = self.board, {}
        for line, ids in picks.items():
            for id_ in ids:
                basis = self.verdict_basis(line, id_)
                if (id_, line, basis) not in self.reread and self.reread_tries[id_, line] < CLOSE_FINAL_TRIES:
                    self.reread_tries[id_, line] += 1
                    self.reread.add((id_, line, basis))
                    b.verdicts.pop((id_, line, b.version), None)
                    self.judged.pop((id_, line, basis), None)
                    due.setdefault(line, []).append(id_)
        if not due:
            return False
        for t, tie in enumerate(b.ties):
            if tie.a in due and tie.b in due:
                self.pairs_read -= {(t, x, y) for x in due[tie.a] for y in due[tie.b]}
        self.judge(due, heavy=True)
        self.ops.note("final_read", model=self.last_model, listings=[[id_, line] for line, ids in due.items() for id_ in ids])
        return True

    def judge(self, candidates: dict[int, list[str]], heavy: bool = False) -> dict[tuple[int, str], bool]:
        b = self.board
        self.carry_verdicts()
        lines = {line.number for line in b.lines}
        known = {n: self.within_room(self.worth_reading(n, [id_ for id_ in ids if id_ in b.listings]))
                 for n, ids in candidates.items() if n in lines}
        for line, ids in known.items():
            self.store.read(ids, line)
        version = b.version
        due_pairs = [(t, x, y) for t, tie in enumerate(b.ties) if tie.a in known and tie.b in known
                     for x in known[tie.a] for y in known[tie.b] if (t, x, y) not in self.pairs_read]
        rounds = []
        for group in self.tied_groups(list(known)):
            pairs = [pair for pair in due_pairs if b.ties[pair[0]].a in group]
            ids = {n: [id_ for id_ in known[n] if (id_, n, version) not in b.verdicts] for n in group}
            if pairs or any(ids.values()):
                rounds.append((ids, pairs))
        read = sum(len(group) for ids, _ in rounds for group in ids.values())
        self.ops.note("judge_asked", asked=sum(map(len, candidates.values())), read=read)
        rows = self.ops.fan([lambda job=job: self.judge_round(*job, heavy) for job in rounds])
        for (ids, pairs), found in zip(rounds, rows):
            self.file_verdicts(ids, pairs, found or ({}, {}), version)
        return {(n, id_): b.verdicts.get((id_, n, version), (False, ""))[0] for n, ids in candidates.items() for id_ in ids}

    def carry_verdicts(self) -> None:
        b, version = self.board, self.board.version
        for id_, line in {key[:2] for key in self.judged}:
            held = self.judged.get((id_, line, self.verdict_basis(line, id_)))
            if held and (id_, line, version) not in b.verdicts:
                b.verdicts[id_, line, version] = held[0]
                if held[1]:
                    self.breaks[id_, line, version] = held[1]

    def verdict_basis(self, line: int, id_: str) -> tuple[str, str, str]:
        pages = "\n".join(close_original(page) for page in self.followed_pages()) if self.sends_away(line, id_) else ""
        return self.requirements([line]), close_original(self.board.listings[id_]), pages

    def sends_away(self, line: int, id_: str) -> bool:
        return True

    def worth_reading(self, n: int, ids: list[str]) -> list[str]:
        b, version = self.board, self.board.version
        if len(ids) < 2:
            return ids
        judged = [id_ for id_ in ids if (id_, n, version) in b.verdicts]
        line = b.line(n)
        passers = [id_ for id_ in self.listed(n) if b.verdicts.get((id_, n, version), (False,))[0]
                   and not price_bars(b, line, b.listings[id_])]
        fresh = [id_ for id_ in ids if id_ not in judged and self.can_change_pick(line, id_, passers)]
        return judged + fresh

    def can_change_pick(self, line: Line, id_: str, passers: list[str]) -> bool:
        b, n = self.board, line.number
        if not passers or line.goal == "best" or any(n in (tie.a, tie.b) for tie in b.ties):
            return True
        price = b.listings[id_].price
        return price is None or price < min(b.listings[i].price or 0 for i in passers)

    def tied_groups(self, lines: list[int]) -> list[list[int]]:
        groups = [{n} for n in lines]
        for tie in self.board.ties:
            hit = [group for group in groups if tie.a in group or tie.b in group]
            if len(hit) == 2:
                groups = [group for group in groups if group not in hit] + [hit[0] | hit[1]]
        return [sorted(group) for group in groups]

    def judge_round(self, ids: dict[int, list[str]], pairs: list[tuple[int, str, str]], heavy: bool = False) -> tuple[dict, dict]:
        b = self.board
        shown = {n: list(group) for n, group in ids.items()}
        for t, x, y in pairs:
            for n, id_ in ((b.ties[t].a, x), (b.ties[t].b, y)):
                shown[n] += [id_] * (id_ not in shown[n])
        slots = [(n, id_) for n, group in shown.items() for id_ in group]
        names = {id_: f"L{place}" for place, id_ in enumerate(dict.fromkeys(id_ for _, id_ in slots), 1)}
        records = close_once([(names[id_], close_original(b.listings[id_])) for _, id_ in slots])
        heads = (f"id {names[id_]}, candidate for line {n}" + ", ties only" * (id_ not in ids[n]) for n, id_ in slots)
        ties = [(t, b.ties[t]) for t in sorted({pair[0] for pair in pairs})]
        rows = "\n".join(close_tie_row(t, tie) for t, tie in ties)
        view = tag("ties", rows) if ties else ""
        view += tag("listings", "".join(tag("listing", f"{head}\n{record}") for head, record in zip(heads, records)))
        if any(self.sends_away(n, id_) for n, group in ids.items() for id_ in group):
            pages = "".join(tag("page", f"id {page.id}\n{close_original(page)}") for page in self.followed_pages())
            view += tag("followed_pages", pages) if pages else ""
        need = [(id_, key) for n, group in ids.items() for id_ in group
                for key in ["item"] + [want.key for want in self.wants(n)]]
        real = {name: id_ for id_, name in names.items()}

        def complete(text: str) -> bool:
            reads, pair_reads = close_judge_rows(text, real)
            return all(item in reads for item in need) and all(pair in pair_reads for pair in pairs)

        stable = tag("job", "judge") + tag("request", b.request) + tag("order", self.requirements([line.number for line in b.lines]))
        return close_judge_rows(self.decide("JUDGE", EVIDENCE, (stable, view), complete, heavy), real)

    def file_verdicts(self, ids: dict[int, list[str]], pairs: list[tuple[int, str, str]], rows: tuple[dict, dict],
                      version: int) -> None:
        b = self.board
        reads, pair_reads = rows
        pages = "\n".join(close_original(page) for page in self.followed_pages())
        for line, listing_ids in ids.items():
            for id_ in listing_ids:
                words = join_words(close_original(b.listings[id_]) + "\n" + pages)
                faults, broke, answered = [], None, True
                for key, want in [("item", None)] + [(want.key, want) for want in self.wants(line)]:
                    quote, state, where = reads.get((id_, key), ("", "NO VERDICT", ""))
                    if want is None and b.line(line).shelf and b.listings[id_].shelf and line not in b.shelf_open:
                        state, quote = ("FAIL", b.listings[id_].shelf) if shelf_bars(b, b.line(line), b.listings[id_]) else ("PASS", b.listings[id_].shelf)
                    answered = answered and (id_, key) in reads
                    silence_passes = want is None or want.rel == "when" or want.line == 0
                    if want is not None and where == "text" and state in ("PASS", "FAIL"):
                        state = "PASS" if silence_passes else "NOT STATED"
                    if state == "PASS" and not (silence_passes or close_found(quote, words)):
                        state = "NOT STATED"
                    against = named_against(b.listings[id_], want, quote, product_words(b, line)) if want is not None and state == "PASS" and where != "fits" else ""
                    if against:
                        state, quote = "FAIL", against
                    if want is not None and state != "PASS" and shelf_holds(b.listings[id_], want):
                        state = "PASS"
                    if state == "FAIL" and close_found(quote, words):
                        broke = broke or (key, quote)
                    if state != "PASS":
                        faults.append(f"{want.prop if want else key} {state}" + ("" if close_none(quote) else f' "{quote}"'))
                b.verdicts[(id_, line, version)] = (not faults, "; ".join(faults) or "every requirement stated")
                if broke:
                    self.breaks[id_, line, version] = broke
                if answered:
                    self.judged[id_, line, self.verdict_basis(line, id_)] = (b.verdicts[id_, line, version], broke)
                self.ops.note("judge", line=line, id=id_, passed=not faults, why=faults)
        for pair in pairs:
            b.pairs[pair] = pair_reads.get(pair, False)
            self.pairs_read.add(pair)

    def close(self, basket: dict[int, str]) -> str:
        self.store.purse.closing = True
        try:
            outcome = self.close_basket(basket)
        finally:
            self.store.purse.closing = False
        self.store.purse.refused = self.store.purse.refused or outcome == "refused"
        return outcome

    def close_basket(self, basket: dict[int, str]) -> str:
        b, store = self.board, self.store
        if self.ops.finished:
            return "done"
        plan = self.plan_of(basket)
        if not plan:
            return self.stop("the basket names no line and listing that exist")
        store.plan_cart(plan)
        if self.ops.steps_left() < store.steps_to_finish():
            return self.order_held(f"{self.ops.steps_left()} steps left and this basket needs {store.steps_to_finish()}")
        if self.spend_spare(basket):
            plan = self.plan_of(basket)
            store.plan_cart(plan)
        before = len(b.notices), b.version
        had = {id_: b.listings[id_].notes for id_ in basket.values() if id_ in b.listings}
        self.open_views(basket)
        self.carry_verdicts()
        last = {line: [id_] for line, id_ in basket.items()
                if (id_, line, b.version) not in b.verdicts or b.verdicts[id_, line, b.version][0] or (id_ in had and b.listings[id_].notes != had[id_])}
        if self.ops.elapsed() < TIME_CLOSE:
            self.read_again(last)
        failed = [ids[0] for line, ids in last.items() if not b.verdicts.get((ids[0], line, b.version), (False,))[0]]
        broken = [t for t, tie in enumerate(b.ties) if b.pairs.get((t, basket.get(tie.a), basket.get(tie.b))) is False]
        if failed or broken:
            return self.stop(f"the final reading fails {failed} and ties {broken}")
        self.follow_chains()
        due = self.duties_due("before_cart")
        if due:
            self.settle(due, self.order_words(plan))
        if before != (len(b.notices), b.version):
            return "refused"
        if not store.cart_set(plan) or b.cart != plan:
            return self.stop(f"the cart shows {self.cart_words()} where the plan is {plan}")
        fault = self.cart_fault(basket, plan)
        if fault:
            return self.stop(fault)
        if self.spare_steps() >= 2 * len(plan) - 1:
            store.look_again(list(plan))
        if any(b.listings[id_].stock is False for id_ in plan if id_ in b.listings):
            return "refused"
        due = self.duties_due("before_order") + self.duties_due("at_end")
        if due:
            self.settle(due, self.order_words(plan))
        if before != (len(b.notices), b.version):
            return "refused"
        outcome = self.finish_outcome(store.order())
        self.ops.note("close", plan=plan, outcome=outcome)
        return outcome

    def spend_spare(self, basket: dict[int, str]) -> bool:
        b, store = self.board, self.store
        if self.ops.elapsed() >= TIME_CLOSE or any(notice.kind == "refusal" and not notice.read for notice in b.notices) or not self.ops.may_think("decide"):
            return False
        changed = False
        for line, id_ in list(basket.items()):
            ln, pick = b.line(line), b.listings.get(id_)
            if ln.goal != "cheapest" or pick is None or pick.price is None:
                continue
            unread = [item.id for item in _view_cheaper_unread(b, ln)[1] if item.price is not None and item.price < pick.price]
            unjudged = sorted((item.id for item in b.listings.values() if line in item.lines and item.read and item.price is not None
                               and item.price < pick.price and self.buyable(item.id, line) and (item.id, line, b.version) not in b.verdicts
                               and not shelf_bars(b, ln, item)), key=lambda i: (b.listings[i].price, i))
            pages = sum(not store.page_open(item_id) for item_id in set(basket.values()))
            room = store.purse.close_room() - pages - 2
            capped = not unread and room >= 3 and store.under_price(line, pick.price)
            if capped:
                unread = [item.id for item in _view_cheaper_unread(b, ln)[1] if item.price is not None and item.price < pick.price]
                room = store.purse.close_room() - pages - 2
            if not unread and room >= 3 and not capped:
                queries = b.cover[line]["queries"]
                index = next((i for i in range(len(queries) - 1, -1, -1) if not queries[i].get("ended")), None)
                if index is not None:
                    turned = store.more_many(line, index, min(CLOSE_PAGES, room - 2))
                    unread = [item.id for item in _view_cheaper_unread(b, ln)[1] if item.price is not None and item.price < pick.price]
                    room = store.purse.close_room() - pages - 2
                    self.ops.note("reserve_paged", line=line, query=index, rows=len(turned), cheaper=len(unread))
            read = store.read(unread[:room], line) if unread and room > 0 else []
            due = [i for i in [*read, *unjudged] if i not in basket.values()]
            self.ops.note("reserve_spent", line=line, read=read, unjudged=unjudged, room=room)
            if due:
                self.read_again({line: due})
            passed = [item.id for item in b.listings.values() if line in item.lines and item.price is not None and item.price < pick.price
                      and b.verdicts.get((item.id, line, b.version), (False,))[0] and self.buyable(item.id, line) and not shelf_bars(b, ln, item)]
            if passed:
                basket[line] = min(passed, key=lambda i: (b.listings[i].price or 0, i))
                changed = True
        return changed

    def stop(self, why: str) -> str:
        self.ops.note("close_stop", why=why)
        return "failed"

    def order_held(self, why: str) -> str:
        self.stop(why)
        held = self.cart_basket()
        if not held or not self.ops.steps_left() or self.cart_fault(held, self.plan_of(held)):
            return "failed"
        return self.finish_outcome(self.store.order())

    def cart_basket(self) -> dict[int, str]:
        b, basket = self.board, {}
        for line in b.lines:
            held = next((id_ for id_, count in b.cart.items() if count == line.qty and id_ not in basket.values()
                         and id_ in b.listings and line.number in b.listings[id_].lines), None)
            if held is None:
                return {}
            basket[line.number] = held
        return basket if len(basket) == len(b.cart) else {}

    def finish_outcome(self, response: dict[str, Any] | None) -> str:
        if self.ops.finished:
            return "done"
        if response and self.store.last_refusal:
            if self.ops.steps_left() > SWAP_STEPS:
                self.store.look_again(list(self.board.cart))
            return "refused"
        return self.stop(self.store.last_fault or "the finish did not end the episode")

    def follow_chains(self) -> None:
        for want in self.board.wants:
            if want.alive and want.ref and self.tracer.chain_state(want) == "unresolved" and self.spare_steps():
                self.tracer.follow(want, self.spare_steps())

    def cart_fault(self, basket: dict[int, str], plan: dict[str, int]) -> str:
        b = self.board
        for line, id_ in basket.items():
            if not self.buyable(id_, line):
                return f"{id_} is out of stock or over the cap of line {line}"
        made = self.plan_total(plan)
        total = made if b.cart_total is None else b.cart_total
        if b.total_cap is not None and total > b.total_cap:
            return f"the cart total {close_money(total)} is over the cap {close_money(b.total_cap)}"
        if abs(total - made) > 0.5 and self.held != total:
            self.held = total
            return f"the cart total {close_money(total)} differs from the sum of listed prices {close_money(made)}"
        return ""

    def duty_proved(self, duty: Duty) -> bool:
        return duty.done_at is not None and (duty.when == "before_cart" or duty.done_at == self.board.cart_version)

    def duties_due(self, when: str) -> list[Duty]:
        for duty in self.board.duties:
            if not self.duty_proved(duty):
                duty.done_at = None
        return [duty for duty in self.board.duties if duty.when == when and duty.done_at is None]

    def steps_owed(self) -> int:
        return sum(max(1, duty.messages) for duty in self.board.duties if not self.duty_proved(duty))

    def settle(self, duties: list[Duty], finish: str) -> bool:
        b, still, problems, acted = self.board, list(duties), "", False
        for _ in range(DUTY_TURNS):
            view = self.duty_view(still, finish, problems)
            reply = self.ops.think("DUTY", DUTY, view, "decide", heavy=True, tools=self.duty_tools())
            if reply is None:
                break
            changed, problems = self.duty_rows(reply.text, still)
            if changed:
                return False
            if not reply.calls and (acted or not still):
                break
            if not reply.calls:
                problems = problems or "Each duty needs a MET or CANNOT row or a tool call."
                continue
            talked = len(b.talk)
            results = [self.duty_call(call) for call in reply.calls]
            acted = acted or any(results)
            if len(b.talk) != talked and Interviewer(self.ops, b, self.store).read_notices():
                return False
            if all(results) and len(b.talk) == talked:
                break
            problems = "\n".join(filter(None, [problems, self.store.last_fault]))
        by_code = bool(still) and not acted
        if by_code:
            view = self.board.tool("view")
            if any(duty.messages for duty in still):
                self.store.say(finish)
            elif view:
                self.store.call(view.name, {})
        for duty in still:
            duty.done_at = b.cart_version
        return not by_code

    def duty_view(self, duties: list[Duty], finish: str, problems: str) -> str:
        b = self.board
        rows = [f"{b.duties.index(duty) + 1} | {duty.when} | {duty.messages} messages | {duty.text}" for duty in duties]
        steps, messages = self.ops.steps_left(), max(0, b.msg_limit - b.msg_used)
        limits = f"{steps} steps left, {self.store.steps_to_finish()} of them for the finish; {messages} messages left"
        view = tag("request", b.request) + tag("duties", "\n".join(rows)) + tag("finish", finish)
        view += tag("record", self.record_view()) + tag("limits", limits)
        return view + (tag("problems", problems) if problems else "")

    def record_view(self) -> str:
        rows, version, last = [], 0, self.last_cart_step()
        for call in self.calls:
            args = dump(call["args"])
            row = f"step {call['step']} {call['tool']} {args if call['reply'] else cut_text(args, RECORD_ARG_CHARS)}"
            if call["reply"]:
                row += f" -> reply: {call['reply']}"
            if call["fault"]:
                row += f" -> failed: {call['fault']}"
            if call["version"] != version:
                version = call["version"]
                row += f" -> cart now: {call['cart']}"
            rows.append(row)
            if call["step"] == last:
                rows.append("--- the cart last changed here: only what follows proves a duty about the cart ---")
        return "\n".join(rows)

    def last_cart_step(self) -> int:
        version = self.board.cart_version
        return next((call["step"] for call in self.calls if call["version"] == version), self.ops.step) if version else 0

    def closed_tools(self) -> set[str]:
        return {tool.name for role, tool in self.board.tools.items() if role in ("order", "decline", "finish", "cart")}

    def duty_tools(self) -> list[dict[str, Any]]:
        closed = self.closed_tools()
        return [tool for tool in self.ops.tool_list if tool["function"]["name"] not in closed]

    def duty_rows(self, text: str, still: list[Duty]) -> tuple[bool, str]:
        b, changed, problems = self.board, False, []
        last, steps = self.last_cart_step(), {call["step"] for call in self.calls}
        for row in tag_rows(text):
            if row[0] == "CHANGED" and len(row) > 1:
                b.notices.append(Notice(" | ".join(row[1:]), self.ops.step))
                changed = True
            elif row[0] in ("MET", "CANNOT") and len(row) > 1:
                index = int(read_first_number(row[1]) or 0)
                duty = b.duties[index - 1] if 1 <= index <= len(b.duties) else None
                if duty not in still:
                    continue
                if row[0] == "MET":
                    step = read_first_number(row[2]) if len(row) > 2 else None
                    on_cart = len(row) <= 3 or row[-1].strip().lower() != "other"
                    if step not in steps or (on_cart and step <= last):
                        problems.append(f"Duty {index}: step {step} proves nothing about the present cart. Make the call.")
                        continue
                still.remove(duty)
                duty.done_at = b.cart_version
        return changed, "\n".join(problems)

    def duty_call(self, call: dict[str, Any]) -> bool:
        function = call.get("function") or {}
        name = function.get("name") or ""
        if name in self.closed_tools() or self.ops.steps_left() <= self.store.steps_to_finish():
            return False
        try:
            args = json.loads(function.get("arguments") or "{}")
        except ValueError:
            args = {}
        message, sent = self.board.tool("message"), self.board.msg_used
        if message and name == message.name:
            strings = [value for value in args.values() if isinstance(value, str)]
            text = args.get(message.args.get("text", "")) or (strings[0] if strings else "")
            if text:
                self.store.say(str(text))
            return self.board.msg_used > sent
        return self.store.call(name, args) is not None and not self.store.last_fault

    def decline_proved(self) -> bool:
        b = self.board
        if (b.msgs_free() and b.tool("message")) or not self.tracer.all_open():
            return False
        for line in b.lines:
            allowed = set(join_words(f"{line.name} {line.shelf}").split())
            allowed |= {word for want in b.wants_of(line.number) for word in join_words(want.value).split()}
            whole = [query for query in b.cover[line.number]["queries"] if query.get("all_seen") and query.get("rows") and query.get("price_max") is None
                     and set(join_words(query.get("words") or "").split()) <= allowed]
            for query in whole:
                rows = [id_ for id_ in query["rows"] if self.buyable(id_, line.number)]
                if rows and all(b.listings[id_].read and self.broken(line.number, id_) for id_ in rows):
                    self.ops.note("decline_proved", line=line.number, words=query.get("words"), rows=len(rows))
                    return True
        return False

    def wrong_kind(self, id_: str, line: int) -> tuple[str, str] | None:
        return next((found for key, found in self.breaks.items() if key[:2] == (id_, line) and found[0] == "item"), None)

    def broken(self, line: int, id_: str) -> tuple[str, str] | None:
        b = self.board
        verdict = b.verdicts.get((id_, line, b.version))
        if verdict and verdict[0]:
            return None
        return self.wrong_kind(id_, line) or self.breaks.get((id_, line, b.version))

    def standing(self, id_: str, line: int) -> int:
        b = self.board
        verdict = b.verdicts.get((id_, line, b.version))
        if verdict and verdict[0]:
            return 0
        return 2 if self.broken(line, id_) else 1

    def choices(self, line: int) -> list[str]:
        b = self.board

        def place(id_: str) -> tuple[int, bool, bool]:
            return self.standing(id_, line), not b.listings[id_].read, (id_, line, b.version) in b.verdicts

        target = b.line(line)
        return sorted((id_ for id_ in rank(b, line) if self.buyable(id_, line) and not self.wrong_kind(id_, line)
                       and not shelf_bars(b, target, b.listings[id_])), key=place)

    def fit_basket(self, choices: dict[int, list[str]], proven: bool = False) -> dict[int, str]:
        b = self.board
        lines = [n for n, ids in choices.items() if ids]
        qty = {line.number: line.qty for line in b.lines}
        width = max(2, int(FIT_COMBINATIONS ** (1 / max(1, len(lines)))))
        best, best_key = {}, None
        for picks in itertools.product(*(list(enumerate(choices[n][:width])) for n in lines)):
            basket = {n: pick[1] for n, pick in zip(lines, picks)}
            total = sum((b.listings[id_].price or 0) * qty[n] for n, id_ in basket.items())
            if len(set(basket.values())) < len(basket) or (b.total_cap is not None and total > b.total_cap):
                continue
            pairs = [b.pairs.get((t, basket[tie.a], basket[tie.b])) for t, tie in enumerate(b.ties)
                     if tie.a in basket and tie.b in basket]
            if False in pairs or (proven and None in pairs):
                continue
            key = (pairs.count(None), sum(pick[0] for pick in picks), total)
            if best_key is None or key < best_key:
                best, best_key = basket, key
        return best

    def best_basket(self) -> dict[int, str]:
        choices = {line.number: self.choices(line.number) for line in self.board.lines}
        if any(self.listed(n) and not ids for n, ids in choices.items()):
            return {}
        floor = {n: min((self.standing(id_, n) for id_ in ids), default=0) for n, ids in choices.items()}
        for worst in range(3):
            kept = {n: [id_ for id_ in ids if self.standing(id_, n) <= max(worst, floor[n])] for n, ids in choices.items()}
            basket = self.fit_basket(kept, proven=not worst) if all(kept[n] or not self.listed(n) for n in kept) else {}
            if basket:
                return basket
        return {}

    def decline(self) -> str:
        b = self.board
        before = len(b.notices), b.version
        words = self.decline_words()
        finish = f"A decline will be sent with these words: {words} The rule for declining: {b.decline_rule or 'none stated'}"
        self.settle(self.duties_due("at_end"), finish)
        if before != (len(b.notices), b.version):
            return "refused"
        return self.finish_outcome(self.store.decline(words))

    def last_basket(self) -> dict[int, str]:
        basket = self.best_basket()
        if basket and all(self.standing(id_, n) < 2 for n, id_ in basket.items()):
            return basket
        return {} if self.decline_proved() else basket

    def finish_now(self) -> None:
        b, last, examined = self.board, None, False
        for _ in range(FINISH_TRIES):
            if self.ops.finished:
                return
            if any(not notice.read for notice in b.notices):
                Interviewer(self.ops, b, self.store).read_notices()
            basket = self.last_basket()
            if basket and not examined and self.ops.may_think("decide") and self.ops.elapsed() < TIME_CLOSE:
                examined = True
                unjudged = {n: [i] for n, i in basket.items() if (i, n, b.version) not in b.verdicts}
                if unjudged:
                    self.read_again(unjudged)
                    basket = self.last_basket()
            outcome = self.close(basket) if basket else self.decline()
            if outcome == "done" or (outcome == "failed" and basket == last):
                break
            last = basket
        if self.ops.finished:
            return
        if b.cart:
            self.store.order()
        else:
            self.store.decline(self.decline_words())

CLERK_RESULT_CHARS, CLERK_MEASURE_ROWS, CLERK_MEASURE_CHARS, CLERK_MEASURE_SHOWN = 8000, 30, 1500, 12
CLERK_COMMITS = ("cart", "order", "decline", "finish", "message")
CLERK_FAULTS = ("FAIL", "NOT STATED", "CONFLICT")
CLERK_TURNS, CLERK_IDLE, CLERK_CHAT_CHARS, CLERK_OLD_CHARS, CLERK_SHOWN_ROWS, CLERK_LAST_JUDGED, CLERK_FINISH_TRIES = 90, 2, 150000, 400, 8, 8, 3


class Clerk:

    def __init__(self, ops: Ops, board: Board, store: Store, interviewer: Interviewer, tracer: Tracer, closer: Closer, scout: Scout):
        self.ops, self.board, self.store = ops, board, store
        self.interviewer, self.tracer, self.closer, self.scout = interviewer, tracer, closer, scout
        self.chat: list[dict[str, Any]] = []
        self.picks: dict[int, str] = {}
        self.chosen: set[str] = set()
        self.results: list[str] = []
        self.facts: list[str] = []
        self.fired: set[tuple[Any, ...]] = set()
        self.served: dict[tuple[str, str], tuple[tuple[int, int, int], str]] = {}
        self.bags: dict[tuple[str, str, frozenset[str]], str] = {}
        self.notices_shown = 0
        self.focus = board.lines[0].number
        self.closing = self.refused = self.walk_over = False
        self.measures: dict[str, float] = {}
        self.measured: dict[str, str] = {}
        self.kept_version, self.kept_wants = board.version, self.wants_in_force()
        store.purse.basket = lambda: {line.number: self.picks.get(line.number) for line in board.lines}

    def run(self) -> None:
        safely = self.ops.safely
        safely(self.read_measures)
        safely(self.open_chat)
        for _ in range(CLERK_FINISH_TRIES):
            safely(self.walk)
            if not self.ops.finished:
                safely(self.finish)
            if self.ops.finished or not self.chat or not self.can_walk():
                break
            self.chat.append({"role": "user", "content": "No listing read passes the examiner on every line. Search other words or a price ceiling, read what the rows show, then pick." + self.status()})
        if not self.ops.finished:
            self.closer.finish_now()

    def read_measures(self) -> None:
        listings = self.board.listings
        for line in self.board.lines:
            if not self.is_best(line):
                continue
            sheets = {i: cut_text(record_sheet(listings[i]), CLERK_MEASURE_CHARS) for i in rank(self.board, line.number)}
            todo = [i for i, sheet in sheets.items() if self.measured.get(i) != sheet][:CLERK_MEASURE_ROWS]
            blocks = "".join(tag("listing", f"id {i}\n{sheets[i]}") for i in todo)
            view = tag("measure", f"{line.best_by[0]}, {line.best_by[1]} wanted") + tag("listings", blocks)
            for note in ("", tag("note", "Your last answer lacked MEASURE rows. Write one row per listing.")) if todo else ():
                reply = self.ops.think("BEST", BEST, view + note, kind="plan")
                rows = [row + [""] * 3 for row in tag_rows(reply.text) if row[0] == "MEASURE"] if reply else []
                for _, i, words, amount, *_ in rows:
                    if i in sheets and squash(words) and squash(words) in squash(sheets[i]):
                        self.measures[i] = number(amount)
                self.measured.update({i: sheets[i] for i in todo} if rows else {})
                if rows or not reply:
                    break

    def measures_view(self) -> str:
        rows = []
        for line in filter(self.is_best, self.board.lines):
            placed = [i for i in rank(self.board, line.number) if self.worth(line, i) is not None]
            shown = sorted(placed, key=lambda i: self.worth(line, i))
            rows.append(f"line {line.number}, {line.best_by[0]}, {line.best_by[1]} wanted: "
                        + ", ".join(f"{i} {self.measures[i]:g}" for i in shown[:CLERK_MEASURE_SHOWN]))
        return "\n".join(rows)

    def worth(self, line: Line, i: str) -> float | None:
        if line.goal == "cheapest":
            return self.board.listings[i].price
        value = self.measures.get(i)
        return None if value is None else -value if line.best_by[1] == "highest" else value

    def is_best(self, line: Line) -> bool:
        return line.goal == "best" and bool(getattr(line, "best_by", None))

    def ranked(self, line: Line) -> bool:
        return line.goal == "cheapest" or self.is_best(line)

    def buying_order(self, line: Line) -> list[str]:
        board, line_no, listings = self.board, line.number, self.board.listings
        order = [i for i in rank(board, line_no) if not price_bars(board, line, listings[i]) and not shelf_bars(board, line, listings[i])]
        if self.is_best(line):
            order.sort(key=lambda i: (self.worth(line, i) is None, self.worth(line, i) or 0))
        held = [i for i in order if i in board.cart or i in self.chosen]
        if not self.ranked(line) and held:
            return held + [i for i in order if i not in held]
        return order

    def combine(self, passers: dict[int, list[str]]) -> dict[int, str] | None:
        board = self.board
        for combo in itertools.product(*passers.values()):
            basket = dict(zip(passers, combo))
            total = sum((board.listings[i].price or 0) * board.line(line).qty for line, i in basket.items())
            tied = all(board.pairs.get((place, basket.get(tie.a), basket.get(tie.b)), True)
                       for place, tie in enumerate(board.ties))
            if tied and (board.total_cap is None or total <= board.total_cap):
                return basket
        return None

    def verdict(self, i: str, line: int) -> bool | None:
        if self.board.version != self.kept_version:
            self.carry_verdicts()
        found = self.board.verdicts.get((i, line, self.board.version))
        return found[0] if found else None

    def carry_verdicts(self) -> None:
        board, old, version, breaks = self.board, self.kept_version, self.board.version, self.closer.breaks
        self.closer.carry_verdicts()
        wants = self.wants_in_force()
        added = [row for key, row in wants.items() if self.kept_wants.get(key) != row]
        standing = [row for row in wants.values() if row not in added]
        for (i, line, seen), (passed, why) in list(board.verdicts.items()) if wants != self.kept_wants else ():
            if seen != old:
                continue
            props = ["item"] + [row[1] for row in standing if row[0] in (line, 0)]
            faulted = any(f"{prop} {state}" in why for prop in props for state in CLERK_FAULTS)
            if not any(row[0] in (line, 0) for row in added) if passed else faulted:
                board.verdicts.setdefault((i, line, version), (passed, why))
                if (i, line, old) in breaks:
                    breaks.setdefault((i, line, version), breaks[i, line, old])
        self.kept_version, self.kept_wants = version, wants

    def wants_in_force(self) -> dict[str, tuple[int, str, str, str]]:
        return {want.key: (want.line, want.prop, want.rel, want.value) for want in self.board.wants
                if want.alive and (want.value or not want.ref)}

    def spare(self) -> int:
        return 0 if self.ops.elapsed() >= TIME_CLOSE else max(0, self.store.purse.walk())

    def can_walk(self) -> bool:
        return not self.walk_over and self.spare() > 0 and self.ops.may_think("walk")

    def budget(self) -> str:
        return f"steps you may still spend: {self.spare()}\nmessages used: {self.board.msg_used} of {self.board.msg_limit}"

    def state(self) -> tuple[int, int, int]:
        return self.board.version, self.board.cart_version, len(self.board.notices)

    def hold(self, door: str, *condition: Any) -> bool:
        key = (door, *condition)
        if key in self.fired:
            return False
        self.fired.add(key)
        self.ops.note("door", door=door, condition=str(condition)[:200])
        return True

    def kept_result(self, name: str, args: dict[str, Any]) -> str:
        if "view" in self.store._roles(name) and not self.store._cart_unsure:
            return f"The cart holds: {self.closer.cart_words()}"
        if self.search_key(name, args) in self.store._sent:
            return skim_view(self.board, self.focus)
        ids = self.ids_in(args)
        record = self.board.listings[ids[0]].records.get(name) if len(ids) == 1 else None
        return "" if record is None else cut_text(self.store.render(name, {"observation": record}), CLERK_RESULT_CHARS)

    def search_key(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        tool = self.tool_of(name)
        page = tool.args.get("page", "") if tool else ""
        return name, dump({key: value for key, value in args.items() if key != page or value}).casefold()

    def send(self, name: str, args: dict[str, Any]) -> None:
        response = self.store.call(name, args)
        text = cut_text(self.store.render(name, response), CLERK_RESULT_CHARS)
        self.results.append(f"{name} {dump(args)}\n{text}")
        if response is not None:
            self.served[(name, dump(args))] = (self.state(), text)
            bag = self.search_bag(name, args)
            if bag:
                self.bags[bag] = dump(args)
            self.file_result(name, args)

    def file_result(self, name: str, args: dict[str, Any]) -> None:
        listings = self.board.listings
        ids = self.store._last.get("ids", [])
        if self.store._roles(name) & {"search", "browse"} and name in self.store.shapes and not self.store.last_fault:
            self.note_search(name, args, ids)
        for i in ids:
            if not listings[i].lines:
                listings[i].lines.add(self.line_of_row(listings[i]))

    def line_of_row(self, item: Listing) -> int:
        words = entry_words(f"{item.shelf} {item.title}")
        fit = {line.number: len(entry_words(line.shelf or line.name) & words) for line in self.board.lines}
        return self.focus if fit[self.focus] == max(fit.values()) else max(fit, key=fit.get)

    def note_search(self, name: str, args: dict[str, Any], rows: list[str]) -> None:
        store, tool = self.store, self.tool_of(name)
        owners = {i: self.line_of_row(self.board.listings[i]) for i in rows}
        line = Counter(owners.values()).most_common(1)[0][0] if owners else self.focus
        page = store._arg(tool, "page")
        asked = {key: value for key, value in args.items() if key != page}
        queries = self.board.cover[line]["queries"]
        query = next((item for item in queries if item["tool"] == name and item["args"] == asked), None)
        for owner in set(owners.values()):
            store._link(owner, [i for i, held in owners.items() if held == owner])
        if query is not None and args.get(page) is None:
            return
        if query is None:
            total = store._last["total"]
            query = {"words": str(args.get(store._arg(tool, "words"), "")), "shelf": "", "price_max": None, "tool": name,
                     "role": tool.role, "args": asked, "total": len(rows) if total is None else total, "pages_read": 1,
                     "ended": False, "next": None, "rows": list(rows), "capped": False, "fault": ""}
            queries.append(query)
            store._sent[self.search_key(name, asked)] = query
            store.note_words_ignored(query)
            store._note_direction(query)
        else:
            query["pages_read"] += 1
            query["rows"] += [i for i in rows if i not in query["rows"]]
        store._after_page(line, query, tool, rows)
        store._note_shelf(line, rows)

    def ids_in(self, value: Any) -> list[str]:
        if isinstance(value, str):
            known = self.store.known_id(value)
            return [known] if known else []
        if isinstance(value, dict):
            value = [*value, *value.values()]
        return [i for item in value for i in self.ids_in(item)] if isinstance(value, list) else []

    def search_bag(self, name: str, args: dict[str, Any]) -> tuple[str, str, frozenset[str]] | None:
        tool = self.tool_of(name)
        words_arg = self.store._arg(tool, "words") if tool else ""
        if not words_arg or not args.get(words_arg):
            return None
        return name, dump({key: value for key, value in args.items() if key != words_arg}).casefold(), frozenset(store_tokens(str(args[words_arg])))

    def send_read(self, name: str, args: dict[str, Any]) -> None:
        bag = self.search_bag(name, args)
        call = self.bags.get(bag, dump(args)) if bag else dump(args)
        invented = self.store.unseen_shelf(name, args)
        if invented:
            self.facts.append(f"Not sent, no step spent: {name} {call}. The store has shown no shelf called {invented}; "
                              f"shelves seen so far: {', '.join(self.store.shelves_seen()) or 'none yet'}.")
            return
        tool = self.tool_of(name)
        if bag and tool and (name, str(args.get(tool.args.get("mode", ""), ""))) in self.store._words_ignored:
            self.facts.append(f"Not sent, no step spent: {name} {call}. With this mode the store lists the whole shelf whatever the words, as <store> says: send the words with the mode that reads them.")
            return
        state, text = self.served.get((name, call), (None, ""))
        kept = text if state == self.state() else self.kept_result(name, args)
        if kept and (self.hold("repeat", name, call, self.state()) or not self.closer.steps_owed()):
            self.results.append(f"{name} {call}: already on the board, not sent, no step spent.\n{kept}")
        elif self.spare() > 0:
            self.send(name, args)
        else:
            self.facts.append(f"Not sent: {name}. The steps left stay back for the finish.")

    def run_call(self, name: str, args: dict[str, Any]) -> None:
        role = self.role(name)
        if role == "finish":
            role = self.finish_way(name, args)
        if role == "decline":
            self.decline_door()
        elif role == "order":
            self.order_call()
        elif role == "message":
            self.message_call(name, args)
        elif role == "cart":
            self.cart_call(name, args)
        else:
            self.read_call(name, args)

    def role(self, name: str) -> str:
        roles = self.store._roles(name)
        if "finish" in roles or {"order", "decline"} <= roles:
            return "finish"
        return next((role for role in CLERK_COMMITS if role in roles), next(iter(roles), ""))

    def tool_of(self, name: str) -> Tool | None:
        return next((tool for tool in self.board.tools.values() if tool.name == name), None)

    def finish_way(self, name: str, args: dict[str, Any]) -> str:
        labels = self.tool_of(name).args
        way = args.get(labels.get("mode", ""))
        if way is not None and way in (labels.get("order"), labels.get("decline")):
            return "order" if way == labels.get("order") else "decline"
        return "order" if self.board.cart else "decline"

    def message_call(self, name: str, args: dict[str, Any]) -> None:
        text = args.get(self.store._arg(self.tool_of(name), "text"), "")
        due = [duty for when in ("before_cart", "before_order", "at_end") for duty in self.closer.duties_due(when)]
        if text and any(duty.messages for duty in due):
            reply = self.store.say(text)
            self.results.append(f"You said: {text}\nThe shopper replied: {reply}" if reply else self.store.render(name, None))
            if reply:
                self.interviewer.hear(self.focus, "duty", text, reply)
                self.refresh_cards()
        elif self.board.msgs_free() > 0 and self.hold("message", self.board.version):
            self.interviewer.interview()
            self.refresh_cards()
            self.facts.append("Your message was not sent: the interviewer owns the messages and has just asked "
                              "what the listings read so far suggest. See the order table.")
        else:
            self.facts.append("Not sent, no step spent: the interviewer owns the messages and has nothing more to ask, "
                              "and no duty needs one now.")

    def decline_door(self) -> None:
        if not self.closer.decline_proved():
            self.facts.append("Not declined: no line is proved empty. A decline is right only when every row a search of the product and its "
                              "required values showed has been read and fails on a requirement its record contradicts. Search and read first.")
            self.walk_over = False
            self.closing = self.can_walk()
            return
        outcome = self.closer.decline()
        self.refused = self.refused or outcome == "refused"
        self.read_changes()
        self.closing = not self.ops.finished

    def read_changes(self) -> None:
        if any(not notice.read for notice in self.board.notices) and self.interviewer.read_notices():
            self.refresh_cards()
        if self.spare() > 0:
            self.scout.search_told()

    def parse_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        parsed = []
        for call in calls:
            function = call.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except ValueError:
                continue
            if isinstance(args, dict) and function.get("name"):
                parsed.append((function["name"], args))
        return parsed

    def refresh_cards(self) -> None:
        for line in self.board.lines:
            promote_uses(self.ops, self.board, line.number)

    def evidence(self, item: Listing) -> str:
        rows = []
        for number_ in sorted(item.lines):
            held = named_values(self.board, number_, item)
            for want in readable_wants(self.board, number_):
                named = held[want.key]
                rows.append(f"[evidence] line {number_} {want_text(want)}: " + (f"entry says {' / '.join(named[:2])}" if named else "no entry of that name"))
        return "\n".join(rows)

    def order_text(self) -> str:
        board = self.board
        rows = [f"whole order: total cap {show_money(board.total_cap)}"] if board.total_cap is not None else []
        rows += [f"tie lines {tie.a} and {tie.b} ({tie.kind}): {tie.rule}" for tie in board.ties]
        for line in board.lines:
            rows += [line_head(line)] + [f"  {want_text(want)}" for want in board.wants_of(line.number)]
            rows += [f"  the shopper has no preference on: {', '.join(sorted(line.closed))}"] if line.closed else []
        return "\n".join(rows)

    def open_chat(self) -> None:
        board = self.board
        records = [f"{item.id}\n{record_sheet(item)}\n{self.evidence(item)}" for item in board.listings.values() if item.read]
        rows = "\n\n".join(skim_view(board, line.number) for line in board.lines)
        facts, page = board.tool("facts"), board.tool("page")
        tools = (f"A listing's attributes come from {facts.name}, so read with that tool." if facts else "") + (f" {page.name} shows its offer, and the desk opens it before ordering." if page else "")
        self.notes_shown = self.store.search_notes()
        text = (tag("request", board.request) + tag("order", self.order_text()) + tag("rows", rows) + tag("records", "\n\n".join(records)) + tag("searched", cover_view(board))
                + tag("store", self.notes_shown) + tag("tools", tools))
        self.chat = [{"role": "user", "content": text + self.status()}]

    def course(self) -> str:
        board, rows = self.board, []
        for line in board.lines:
            number_ = line.number
            items: dict[str, Listing] = {}
            for item in board.listings.values():
                if number_ in item.lines and item.read:
                    items.setdefault(join_words(item.title) or item.id, item)
            pick = self.picks.get(number_)
            passed = [i for i in rank(board, number_) if board.verdicts.get((i, number_, board.version), (False,))[0]]
            rows.append(f"line {number_}: {len(items)} different products read, " + (f"pick {pick} passed" if pick else f"judge passed: {', '.join(passed[:6])}" if passed else "no pick yet"))
            held = [named_values(board, number_, item) for item in items.values()]
            for want in readable_wants(board, number_):
                named = [values[want.key] for values in held]
                forms = [form for form in map(value_words, want_forms(want)) if form] if want.rel in ("is", "one of") else []
                holding = sum(any(form <= value_words(value) for form in forms for value in entries) for entries in named)
                rows.append(f"  {want_text(want)}: {sum(map(bool, named))} have an entry of that name" + (f", {holding} holding the value" if forms else "") + f", {sum(not entries for entries in named)} have none")
        return "\n".join(rows)

    def cheaper_rows(self, line: Line) -> list[Listing]:
        board, number_ = self.board, line.number
        passed = [board.listings[i].price for i in board.cart if i in board.listings and number_ in board.listings[i].lines] + [item.price for item in board.listings.values() if board.verdicts.get((item.id, number_, board.version), (False,))[0]]
        ceiling = min((price for price in passed if price is not None), default=None)
        rows = [item for item in board.listings.values() if number_ in item.lines and not item.read and item.price is not None and not price_bars(board, line, item)
                and not off_shelf(line, item) and not self.twin_spent(item.id) and (ceiling is None or item.price < ceiling)]
        return sorted(rows, key=lambda item: (item.price, item.id))

    def twin_spent(self, lid: str) -> bool:
        board = self.board
        self.verdict(lid, 0)
        return any(twin_state(board, number_, lid) > 0 and not any(board.listings[i].stock is False for i in twin_ids(board, number_, lid)) for number_ in board.listings[lid].lines)

    def status(self) -> str:
        board = self.board
        fresh = [notice.text for notice in board.notices[self.notices_shown:]]
        self.notices_shown = len(board.notices)
        blocks = {"budget": self.budget(), "cart": dump(board.cart) if board.cart else "empty", "order": self.order_text() if fresh else "", "course": self.course(),
                  "duties": "\n".join(f"{duty.when}: {duty.text}" for duty in board.duties if not self.closer.duty_proved(duty)), "notices": "\n".join(fresh), "measures": self.measures_view()}
        cheap = [f"line {line.number}: {item.id} {show_money(item.price)} {cut_text(item.title, VIEW_TITLE_CHARS)}" for line in board.lines if line.goal == "cheapest" for item in self.cheaper_rows(line)[:CLERK_SHOWN_ROWS]]
        blocks["cheapest_unread"] = "\n".join(cheap)
        notes = self.store.search_notes()
        blocks["store"] = notes if notes != getattr(self, "notes_shown", None) else ""
        self.notes_shown = notes
        return tag("status", "".join(tag(name, text) for name, text in blocks.items() if text))

    def read_call(self, name: str, args: dict[str, Any]) -> None:
        ids = self.ids_in(args)
        if ids and self.role(name) in ("facts", "page") and all(self.twin_spent(i) for i in ids):
            self.facts.append(f"Not sent, no step spent: {name} {dump(args)}. It is the same product as a listing you have read that has not passed, so its record says the same. Look at <course> and change your search.")
            return
        self.send_read(name, args)
        for i in ids:
            if self.board.listings[i].read:
                self.results.append(self.evidence(self.board.listings[i]))

    def cart_call(self, name: str, args: dict[str, Any]) -> None:
        ids = self.ids_in(args)
        if ids:
            self.chosen.update(ids)
            self.closing = True
        else:
            self.facts.append("Not sent, no step spent: that call names no listing the store has shown.")

    def order_call(self) -> None:
        self.closing = True

    def run_turn(self, calls: list[dict[str, Any]]) -> None:
        answers = []
        lone = len(calls) == 1
        for call in calls:
            parsed = self.parse_calls([call])
            self.results, self.facts = [], []
            if not parsed:
                self.facts.append("Not run, no step spent: the arguments were not one JSON object.")
            elif self.role(parsed[0][0]) in CLERK_COMMITS and self.role(parsed[0][0]) != "cart" and not lone:
                self.facts.append("Not sent, no step spent: an order, a decline and a message each go alone in their turn.")
            elif not self.ops.finished:
                self.ops.safely(lambda parsed=parsed: self.run_call(*parsed[0]))
            answers.append("\n".join(self.results + self.facts) or "Done.")
        self.results, self.facts = [], []
        self.ops.safely(self.read_changes)
        if self.closing and not self.ops.finished:
            self.closing = False
            answers[-1] += "\n" + self.buy()
        if not self.ops.finished:
            self.chat += [{"role": "tool", "tool_call_id": call.get("id"), "content": answer} for call, answer in zip(calls, answers)]
            self.chat[-1]["content"] += "\n" + "\n".join(self.results + self.facts) + self.status()

    def examine(self, pick: str) -> str:
        board = self.board
        item = board.listings[pick]
        number_ = self.line_of_row(item) if not item.lines or len(board.lines) == 1 else next((n for n in sorted(item.lines) if n not in self.picks), min(item.lines))
        line = board.line(number_)
        if item.stock is False:
            return f"Not picked, no step spent: {pick} is out of stock. Choose another."
        if line.goal == "cheapest" and self.spare() >= 2:
            cheaper = [row for row in self.cheaper_rows(line) if item.price is not None and row.price < item.price]
            if cheaper and self.hold("cheaper", pick):
                return (f"Not picked yet, no step spent. {len(cheaper)} rows cheaper than {pick} have not been read, from {show_money(cheaper[0].price)}, and line {number_} must hold the cheapest listing that fits. "
                        "Read the cheapest of them that could fit, or search another word for the product or a price ceiling, then pick again. Picking the same listing again goes through.")
        self.closer.judge({number_: [pick]})
        if not self.verdict(pick, number_):
            why = (board.verdicts.get((pick, number_, board.version)) or (False, "the record does not show it"))[1]
            self.ops.note("clerk_rejected", id=pick, line=number_, why=str(why)[:200])
            return f"The examiner rejects {pick} for line {number_}, no step spent: {why}. Find a listing that states what it lacks."
        self.picks[number_] = pick
        return f"The examiner passes {pick} for line {number_}."

    def buy(self) -> str:
        board = self.board
        said = [self.examine(pick) for pick in sorted(self.chosen) if pick in board.listings]
        self.chosen = set()
        for number_, pick in list(self.picks.items()):
            if not self.closer.buyable(pick, number_) or not self.verdict(pick, number_):
                del self.picks[number_]
                said.append(f"Line {number_} no longer holds {pick}: its stock, a cap or what now stands rules it out.")
        waiting = [line.number for line in board.lines if line.number not in self.picks]
        if waiting:
            return " ".join(said + [f"Lines {waiting} still need a pick."])
        return " ".join(said + [self.close_picks()])

    def close_picks(self) -> str:
        board = self.board
        basket = dict(self.picks)
        self.closer.judge({number_: [pick] for number_, pick in basket.items()}, heavy=True)
        if self.combine({number_: [pick] for number_, pick in basket.items()}) is None:
            broken = [f"lines {tie.a} and {tie.b} ({tie.rule})" for place, tie in enumerate(board.ties) if board.pairs.get((place, basket.get(tie.a), basket.get(tie.b))) is False]
            total = sum((board.listings[i].price or 0) * board.line(n).qty for n, i in basket.items())
            return f"The basket does not hold together, no step spent: {'; '.join(broken) or f'its total {show_money(total)} is over the cap {show_money(board.total_cap)}'}. Change a pick."
        outcome = self.closer.close(basket)
        self.store.plan_cart(None)
        self.ops.note("clerk_close", basket=basket, outcome=outcome)
        if outcome == "done" or self.ops.finished:
            return ""
        self.read_changes()
        if outcome == "refused":
            return "The store refused the order: <notices> say what changed. Change the pick that no longer holds with a cart call, or call the order tool to keep them all."
        return f"The order did not go out: the cart shows {board.cart}, total {board.cart_total}, or a cap stands in the way."

    def trim(self) -> None:
        total = sum(len(str(message.get("content"))) for message in self.chat)
        for message in self.chat:
            if total <= CLERK_CHAT_CHARS:
                return
            if message["role"] == "tool" and len(message["content"]) > CLERK_OLD_CHARS:
                total -= len(message["content"]) - CLERK_OLD_CHARS
                message["content"] = message["content"][:CLERK_OLD_CHARS] + " ... (older result cut)"

    def walk(self) -> None:
        idle = 0
        for _ in range(CLERK_TURNS):
            if self.ops.finished or not self.chat or not self.can_walk():
                return
            self.focus = next((line.number for line in self.board.lines if line.number not in self.picks), self.focus)
            self.trim()
            reply = self.ops.think("CLERK", CLERK, self.chat, kind="walk", tools=self.ops.tool_list)
            if reply is None:
                return
            self.chat.append({"role": "assistant", "content": reply.text or None, **({"tool_calls": reply.calls} if reply.calls else {})})
            if not reply.calls:
                idle += 1
                if idle >= CLERK_IDLE:
                    return
                self.chat.append({"role": "user", "content": "Make a tool call now." + self.status()})
                continue
            idle = 0
            self.run_turn(reply.calls)

    def finish(self) -> None:
        board = self.board
        for _ in range(CLERK_FINISH_TRIES):
            basket = {}
            for line in board.lines:
                number_ = line.number
                order = [i for i in self.buying_order(line) if board.listings[i].read and self.closer.buyable(i, number_)]
                passed_before = {i for (i, n, version), verdict in board.verdicts.items() if n == number_ and version < board.version and verdict[0]}
                order.sort(key=lambda i: i not in passed_before)
                passed = [i for i in order if self.verdict(i, number_)]
                if not passed and self.ops.may_think("decide"):
                    self.closer.judge({number_: [i for i in order if self.verdict(i, number_) is None][:CLERK_LAST_JUDGED]})
                    passed = [i for i in order if self.verdict(i, number_)]
                if passed:
                    basket[number_] = passed[0]
            if len(basket) < len(board.lines):
                return
            outcome = self.closer.close(basket)
            self.store.plan_cart(None)
            if outcome == "done" or self.ops.finished:
                return
            self.read_changes()

SCOUT_READS, SCOUT_LINE_SEARCHES, SCOUT_BETTER_READS, SCOUT_STACKS, SCOUT_INTERVIEWS = 5, 6, 30, 4, 2
SCOUT_TOLD_SEARCHES, SCOUT_TOLD_READS, SCOUT_WORDS_AFTER, SCOUT_TOP_ROWS, SCOUT_VALUE_WORDS, SCOUT_GAP_WALKS = 3, 3, 2, 33, 6, 2
SCOUT_FLOOR, SCOUT_CHAIN, SCOUT_USE_WORDS = 3, 18, 3
SCOUT_SWEEP_END, SCOUT_PER_MESSAGE, SCOUT_MARGIN = 85.0, 6.0, 10.0


def common_words(one: str, other: str) -> int:
    theirs = split_words(other)
    return sum(any(a == b or (min(len(a), len(b)) > 2 and (a.startswith(b) or b.startswith(a))) for b in theirs)
               for a in split_words(one))


def same_words(one: str, other: str) -> bool:
    return common_words(one, other) == len(split_words(one)) and common_words(other, one) == len(split_words(other))


class Scout:

    def __init__(self, ops: Ops, board: Board, store: Store, interviewer: Interviewer, tracer: Tracer, closer: Closer):
        self.ops, self.board, self.store = ops, board, store
        self.interviewer, self.tracer, self.closer = interviewer, tracer, closer
        self.order: list[Line] = []
        self.share: dict[int, int] = {}
        self.used: dict[int, int] = defaultdict(int)
        self.tried: dict[int, set[str]] = defaultdict(set)
        self.look: dict[int, list[str]] = defaultdict(list)
        self.words: dict[int, list[str]] = defaultdict(list)
        self.worded: dict[int, str] = {}
        self.told: dict[int, set[tuple[str, str]]] = {}
        self.dead: dict[str, set[str]] = defaultdict(set)
        self.skip: set[str] = set()
        self.chain_tries: dict[str, int] = defaultdict(int)
        self.chain_steps = self.chain_held = 0
        self.asked = 0
        self.failed_reads = 0
        self.leave = ""
        self.read_cost = 0.25 if board.tool("compare") else 1.0


    def first_look(self) -> None:
        self.settle_ties()
        self.plan(self.board.lines)
        if self.free() + self.chain_held >= len(self.board.lines):
            self.follow_chains()
            self.sweep()
            self.ask()
            self.settle_ties()

    def settle_ties(self) -> None:
        board = self.board
        numbers = {line.number for line in board.lines}
        board.ties[:] = [t for t in board.ties if t.sentence and t.a != t.b and {t.a, t.b} <= numbers]
        self.order = sorted(board.lines, key=lambda line: (-self.anchor_score(line.number), line.number))

    def anchor_score(self, number_: int) -> int:
        return sum(2 * (tie.a == number_) + (tie.b == number_) for tie in self.board.ties)

    def free(self) -> int:
        board, ops = self.board, self.ops
        waiting = board.msgs_free() if self.asked < SCOUT_INTERVIEWS else 0
        steps = self.store.purse.walk() - waiting - self.chain_held
        seconds = TIME_WALK - SCOUT_MARGIN - ops.elapsed() - SCOUT_PER_MESSAGE * waiting
        return min(steps, int(seconds * ops.rate / (1 + 1 / self.read_cost)))

    def plan(self, lines: list[Line]) -> None:
        waiting = [want for want in self.tracer.open_chains() if not self.chain_tries[want.key]]
        self.chain_held = 0
        room = max(0, self.free())
        self.chain_steps = min(SCOUT_CHAIN, max(SCOUT_FLOOR, room // (len(self.board.lines) + len(waiting))))
        self.chain_held = min(room, self.chain_steps * len(waiting))
        room -= self.chain_held
        weights = {line.number: 2 + (self.anchor_score(line.number) > 0) + (line.goal == "cheapest") for line in lines}
        floor = max(1, min(SCOUT_FLOOR, room // len(lines)))
        for line in lines:
            part = room * weights[line.number] // sum(weights.values())
            self.share[line.number] = self.used[line.number] + max(floor, part)

    def sweep(self) -> None:
        for line in self.order:
            number_ = line.number
            hurry = self.ops.elapsed() > SCOUT_SWEEP_END or self.share[number_] <= 2 * SCOUT_FLOOR
            wanted = 1 if hurry else 1 + bool(self.value_groups(line))
            stop = self.share[number_] * 2 // 3
            hits = tries = 0
            while hits < wanted and tries < wanted + 2 and self.used[number_] < stop and self.free() > 0:
                forms = self.queries(line)
                if not forms:
                    break
                tries += 1
                self.search(line, forms[0])
                hits += bool(self.last_total(number_))
            rows = self.stated_first(line, self.unread(line))
            if rows and not self.leave and self.free() > 0:
                limit = min(ASK_MENU_KINDS, max(1, int((stop - self.used[number_]) / self.read_cost)))
                self.read_rows(line, rows[:limit], behind=True)

    def stated_first(self, line: Line, rows: list[str]) -> list[str]:
        phrases = [squash(value) for group in self.value_groups(line) for value in group]
        items = self.board.listings
        shown = {lid: sum(phrase in squash(f"{items[lid].title} {dump(items[lid].row)}") for phrase in phrases) for lid in rows}
        ordered = sorted(rows, key=lambda lid: -shown[lid])
        first = {kind[0].id for kind in ask_kinds([items[lid] for lid in ordered])}
        return sorted(ordered, key=lambda lid: (-shown[lid], lid not in first))

    def last_total(self, number_: int) -> int | None:
        queries = self.board.cover[number_]["queries"]
        return queries[-1].get("total") if queries else 0

    def value_groups(self, line: Line) -> list[list[str]]:
        groups = []
        for want in reversed(self.board.wants_of(line.number)):
            if want.value and want.rel in ("is", "one of", "like"):
                groups.append([part for part in want_forms(want) if len(part.split()) <= SCOUT_VALUE_WORDS])
        return [group for group in groups if group]

    def alive(self, value: str) -> bool:
        return not any(not self.matches_nothing(name) for name in self.dead.get(value.casefold(), ()))

    def stacks(self, line: Line) -> list[str]:
        if self.store.search_direction() != "and":
            return []
        groups = [[value for value in group if self.alive(value)] for group in self.value_groups(line)]
        kept = [group for group in groups if group]
        combos = itertools.islice(itertools.product(*kept), SCOUT_STACKS) if kept else ()
        return [value_first(product_heads(line)[0], list(combo)) for combo in combos]

    def singles(self, line: Line) -> list[str]:
        head = product_heads(line)[0]
        return [value_first(head, [value]) for group in self.value_groups(line) for value in group if self.alive(value)]

    def queries(self, line: Line) -> list[str]:
        heads = product_heads(line)
        head = heads[0]
        values = [value for group in self.value_groups(line) for value in group]
        bounds = [value_first(head, [want.value]) for want in self.board.wants_of(line.number) if want.rel in ("at least", "at most") and want.value and len(want.value.split()) <= SCOUT_VALUE_WORDS]
        browse = [""] if self.board.tool("browse") and line.shelf else []
        direction = self.store.search_direction()
        if direction == "or":
            forms = [value_first(head, [group[0] for group in self.value_groups(line)])] * bool(values) + [value_first(head, [value]) for value in values] + bounds + heads + browse + [line.shelf, line.name] + values
        else:
            stacks, singles = self.stacks(line), self.singles(line)
            forms = stacks[:1] + (singles + stacks[1:] if stacks and self.matches_nothing(stacks[0]) else stacks[1:] + singles)
            forms += bounds + browse + [*heads, line.name, line.shelf] + [value_first(head, [value]) for value in values if not self.alive(value)]
        forms += [f"{want.value} {head}" for want in self.board.soft_of(line.number) if len(want.value.split()) <= SCOUT_USE_WORDS]
        forms[1:1] = squeezed([group[0] for group in self.value_groups(line)])
        sent = self.board.cover[line.number]["queries"]
        forms = [head, value_first(head, [group[0] for group in self.value_groups(line)])][:1 + bool(values)] + forms if not sent else forms
        teach = (values[:1] if values else [head] if sent else []) if direction is None else []
        worded = [] if direction == "or" else self.words[line.number]
        forms = worded + forms if worded else forms
        if sent and not any(query.get("total") for query in sent):
            words = split_words(line.name)
            forms += [" ".join(words[:2]), " ".join(words[:1])] + values
        tried, found = self.tried[line.number], []
        for form in [*self.look[line.number], *forms[:1 + (not sent)], *teach, *forms[1 + (not sent):]]:
            if (form or browse) and not any(same_words(form, other) for other in [*tried, *found]) and not self.matches_nothing(form):
                found.append(form)
        return found

    def matches_nothing(self, form: str) -> bool:
        if self.store.search_direction() != "and":
            return False
        sent = (query for cover in self.board.cover.values() for query in cover["queries"])
        empty = (set(split_words(query.get("words"))) for query in sent if query.get("total") == 0 and not query.get("shelf")
                 and query.get("price_max") is None)
        return any(words and words <= set(split_words(form)) for words in empty)

    def search(self, line: Line, words: str, below: float | None = None) -> list[str]:
        number_, start = line.number, self.ops.step
        self.tried[number_].add(words)
        bare = bool(words) and below is None and self.store.search_direction() is None and len(self.tried[number_]) <= 2
        shelf, price_max = ("", None) if bare else (line.shelf, below)
        if not bare and below is None and self.store.has_price_filter():
            ceiling = self.ceiling(line)
            price_max = ceiling if ceiling and ceiling > 0 else None
        ids = self.store.search(number_, words, shelf, price_max) or []
        if words and below is None and (shelf or price_max is not None) and self.last_total(number_) == 0 and self.free() > 0:
            ids = self.store.search(number_, words) or []
        self.used[number_] += self.ops.step - start
        head = product_heads(line)[0]
        for value in (value for group in self.value_groups(line) for value in group):
            if self.last_total(number_) == 0 and words == value_first(head, [value]):
                self.dead[value.casefold()].add(head)
        return ids

    def bound(self, line: Line) -> float | None:
        caps = [] if line.cap is None else [line.cap]
        return min(caps + ([] if self.board.total_cap is None else [self.board.total_cap / line.qty]), default=None)

    def ceiling(self, line: Line) -> float | None:
        board, bound = self.board, self.bound(line)
        if board.total_cap is None:
            return bound
        others = sum(other.qty * min((item.price for item in board.listings.values() if other.number in item.lines and item.price is not None and not price_bars(board, other, item)), default=0)
                     for other in board.lines if other is not line)
        return min(bound, (board.total_cap - others) / line.qty)

    def head_words(self, line: Line) -> set[str]:
        bounds = (want.value for want in self.board.wants_of(line.number) if want.rel in ("at least", "at most"))
        return set(split_words(f"{line.name} {line.shelf} {' '.join(product_heads(line))} {' '.join(bounds)}"))

    def narrow_rows(self, line: Line) -> dict[str, float]:
        head, found, every = self.head_words(line), {}, self.store.search_direction() == "and"
        for query in self.board.cover[line.number]["queries"]:
            if set(split_words(query.get("words"))) - head:
                for lid in (query.get("rows") or [])[:None if every else SCOUT_TOP_ROWS]:
                    found[lid] = min(found.get(lid, float("inf")), query.get("total") or 0)
        return found

    def unread(self, line: Line, any_shelf: bool = False) -> list[str]:
        board, bound, ceiling = self.board, self.bound(line), self.ceiling(line)
        groups, every = self.value_groups(line), self.store.search_direction() != "or"
        rows = []
        for lid in rank(board, line.number):
            item = board.listings[lid]
            over = bound is not None and item.price is not None and item.price > bound
            stray = off_shelf(line, item) and not any_shelf
            if not (item.read or over or stray or item.stock is False or lid in self.skip):
                rows.append(lid)
        return sorted(rows, key=lambda lid: (ceiling is not None and (board.listings[lid].price or 0) > ceiling,
                                             *reading_key(board, line.number, lid, groups, every)))

    def named_rows(self, line: Line) -> list[str]:
        board, look, narrow = self.board, [squash(words) for words in self.look[line.number]], self.narrow_rows(line)
        phrases = look + [squash(value) for group in self.value_groups(line) for value in group]
        rows = []
        for lid in self.unread(line):
            text = squash(f"{board.listings[lid].title} {board.listings[lid].notes}")
            if lid in narrow or any(phrase and phrase in text for phrase in phrases):
                rows.append((not any(phrase and phrase in text for phrase in look), lid))
        return [lid for _, lid in sorted(rows, key=lambda row: row[0])]

    def read_rows(self, line: Line, ids: list[str], behind: bool = False) -> bool:
        number_, start = line.number, self.ops.step
        done = self.store.read(ids, number_) or []
        spent = self.ops.step - start
        self.used[number_] += spent
        self.skip.update(set(ids) - set(done))
        self.failed_reads = 0 if done else self.failed_reads + 1
        if self.failed_reads >= 2:
            self.leave = "records could not be read"
        if done and spent:
            self.read_cost = spent / len(done)
        return True

    def follow_chains(self, tries: int = 1) -> bool:
        moved = False
        for want in self.tracer.open_chains():
            if self.chain_tries[want.key] >= tries:
                continue
            self.chain_tries[want.key] += 1
            steps = min(self.chain_steps, self.chain_held + max(0, self.free()))
            self.chain_held = max(0, self.chain_held - self.chain_steps)
            start = self.ops.step
            if steps > 0:
                self.tracer.follow(want, steps)
            moved = moved or self.ops.step > start
        return moved

    def ask(self) -> None:
        self.asked += 1
        for line in self.board.lines:
            self.told.setdefault(line.number, self.line_told(line))
        if self.board.msgs_free():
            self.interviewer.interview()
            self.search_told()

    def line_told(self, line: Line) -> set[tuple[str, str]]:
        return {(want.prop, want.value) for want in self.board.wants_of(line.number) if want.value and want.rel in ("is", "one of", "like")}

    def search_told(self) -> None:
        for line in self.order:
            told, number_ = self.line_told(line), line.number
            if told == self.told.get(number_, told) or self.free() <= 0:
                continue
            for _ in range(SCOUT_TOLD_SEARCHES):
                forms = [form for form in self.queries(line) if form not in self.look[number_]]
                if not forms or self.free() <= 0:
                    break
                self.search(line, forms[0])
                self.told[number_] = told
                if self.last_total(number_):
                    break
            rows = self.named_rows(line)[:min(SCOUT_TOLD_READS, self.free())]
            if rows:
                self.read_rows(line, rows)

def run_office(ops: Ops, board: Board) -> Closer:
    read_order(ops, board)
    if not board.lines:
        board.lines.append(Line(1, cut_text(board.request, 200)))
    store = Store(ops, board)
    interviewer = Interviewer(ops, board, store)
    tracer = Tracer(ops, board, store)
    closer = Closer(ops, board, store, tracer)
    board.closer = store.purse.closer = closer
    scout = Scout(ops, board, store, interviewer, tracer, closer)
    ops.safely(scout.first_look)
    if not ops.finished:
        Clerk(ops, board, store, interviewer, tracer, closer, scout).run()
    return closer


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    environment = problem_data["environment"]
    policy = environment["policy_view"]
    ops = Ops(environment["binding"], policy)
    board = Board(request=policy.get("query") or "", opening=policy.get("opening"))
    try:
        run_office(ops, board)
    except Exception as exc:
        ops.note("crash", where="office", error=f"{type(exc).__name__}: {exc}"[:400])
    try:
        if not ops.finished and getattr(board, "closer", None):
            board.closer.finish_now()
    except Exception as exc:
        ops.note("crash", where="finish", error=f"{type(exc).__name__}: {exc}"[:400])
    ops.note("seats", seats={name: [record[0], round(record[1], 5), round(record[2], 1), record[3], record[4], record[5]] for name, record in ops.seats.items()}, spent=round(ops.spent, 5), finished=ops.finished)
    ops.pool.shutdown(wait=False, cancel_futures=True)
    return ops.trace


__all__ = ["agent_main"]
