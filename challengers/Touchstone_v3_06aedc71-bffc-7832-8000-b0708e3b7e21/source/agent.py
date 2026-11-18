"""Kyo: buys for one shopper inside an ORO store world.

A light model walks the store with the world's own tools. Code keeps the
ledger (items, needs, budget, listings seen), plans the questions to the
shopper, and stops wasted steps. A heavy model reads the request once and
checks the cart before an order and the evidence before a decline.

k1 (a fork of Kyo v6) changes three things:
- the interview asks one question at a time and reads each answer before the
  next, so the remaining messages go to items no answer has settled yet, and a
  question the shopper found crowded is asked again with example values;
- the episode keeps to its 300 s: the loop stops taking turns at LOOP_SECONDS,
  late model calls are not resent, and the request's own steps at the order or
  the ending are skipped when no time is left for them;
- a loop that stops on its limits with an empty cart does not end without a
  purchase straight away: a review names fitting listings among those seen and
  Kyo carts and checks out the ones a fresh check passes.

k3 (headstone v6 candidate, from k2 = headstone v5) changes, from race 12's episodes:
- the chain check takes a listing as the one a requirement points at only when
  its opened page shows every property the description names (a shop's near
  twin that leaves one out is not it); rows that name a shop or brand from the
  requirement are listed first, rows are long enough to show the shop, and the
  check has room to think; the code search for a missing link skips pages
  already read;
- the review before an ending without a purchase runs up to the hard spend
  line (the soft line skipped it in costly episodes);
- the sold-out replacement review sees only the sold-out line's category, at
  most 20 records.

k4 (headstone v6 candidate, from k3) moves spend off a check that rarely acts.
In six practice runs the checkout check on tasks without a pick rule flagged 2
of 242 carts, neither to any gain, for about a fifth of the spend, and a run
that crosses the key's spend line can be stopped mid-race. That check now runs
only on cheapest / most / least picks, where its flags were the useful ones.
The checks before a purchase the code makes and before an ending without a
purchase stay.

k5 (from k4) keeps a run under the inference key's spend line. In races 12 and
13 headstone v5 spent about $0.055-0.062 a task, so runs crossed the key's $5
line and were stopped with episodes still running. The heavy model's reviews
(the cart check, the sold-out replacement review, the review before an ending
without a purchase) cost $0.015-0.023 a call, the light model's about a
thirteenth of that. They run on the heavy model while the episode has spent
less than LEAN_LINE; past it the light model decides, with room for its longer
thinking. The question plan stays on the heavy model.

k6 (from k5), from races 11 and 12's episodes:
- the fits a sold-out replacement review or a review before an ending without
  a purchase names are taken cheapest first on every task but a most or least
  pick: the races graded the dearer of two fitting orders "not the best
  option" (538 orders dearer than the passing ones on their task, 19 cheaper);
- a one-property question the shopper answers with "one question at a time"
  is asked next about the same property on another requested item: such
  re-asks drew a requirement 0.91 (race 11) and 0.58 (race 12) of the time,
  the same question with example values 0.12 (race 10), 0.07 and 0.02. When
  that answer brings no requirement (race 10, where the other item drew
  nothing and a simpler question on the same item 0.60-1.00), or when there is
  no other item, the item is asked once more in plain words without its name;
- the plan skips gender for an item whose key table shows no gender key.
  Over races 10-12 every gender value a shopper gave (139) came from a task
  whose listings carried one, and 710 gender questions on tasks without one
  drew nothing (482 of them in race 12). On 49 recorded practice plans that
  asked gender, the rule dropped it in 22 of 22 without a key and kept it in
  23 of 27 with one (the old prompt sent again: 24 of 27);
- the plan also skips the listing's condition, a listing detail like the
  warranty: over races 9-12 the field asked it 1,579 times and drew one
  requirement (the KYO line 63 times, none).

k7 (from k6): the fits a review before an ending without a purchase names are
read by item number even when the key is text. The light model copied the
prompt's "item number" as the key (practice: every light review that named a
fit), and reviews of multi-item tasks wrote "item 1" (local race runs: 3 of 7
reviews that named a fit); such keys were dropped where Kyo buys the reviewed
lines itself (after the loop's limits, after a refused ending), and the loop
read "item item number". A key with one number is that item, a key without a
number is item 1 on a one-item task or the one requested item it names.

k8 (from k7): one message to the shopper is held back at the interview (when
no step known yet takes a message and three or more are free) until the
store's first cart answer, or an ending, has been read. On practice mutants a
procedure that only a store notice on the first cart answer states ("before
checkout, message the customer the final cart and its total") was kept 0 of 15
times by the KYO line, because the interview had spent every message before
the first cart answer came; stated in the request it was kept 14 of 14. When
nothing claimed the held message, it asks the plan's next question then (the
4th-5th questions drew a requirement 0.06-0.14 of the time in races 11-12).

k9 (from k8): a step the request asks for before anything is taken out of the
cart or swapped ("ask me first") is done by Kyo: before a cart call that
lowers or drops a line, when no message went out since the cart last changed,
Kyo asks the shopper (one message, kept back from the interview) and follows
the answer; with no message left the line stays. The loop's own messages are
never run, so it could not do this step itself: on practice mutants the KYO
line kept it in 6-9 of 15 episodes (k7 0.333, k8 0.600 against 0.733-0.867).

k10 (from k9): an enum argument is read by its words, not its position. A cart
mode whose value says remove, withdraw, discard, delete, drop, release, clear
or take is a removal (the first value counted as the add and every other as a
removal); Kyo's own cart calls pick the add and remove values the same way;
a required choice on the message tool takes the value naming the shopper
(customer, shopper, user, buyer, client) when there is one. The sold-out
replacement review on the heavy model gets 2,500 tokens (1,200 cut it short in
about one call in eighteen, and the light model then decided it again).

k11 (from k10): the plan ranks a product's type or style and its form or fit
first among the listed keys, then size, material and looks (the order Know
Thyself's plan uses). Across races 9-13 a type, style or form property held the
hidden need on 26-37% of the tasks where one was drawn, the most of any family
in four races, while looks held 4-22%; in race 13 (two messages) v5 asked such
a property first in 7% of its runs and Know Thyself in 29%, at need rates of
about 0.5 either way. On the RaceSim question bench (races 10 and 11, two
samples each) the change named 67 of 100 hidden needs within the message budget
against 62 for the old order, with the first question no worse.

k12 (from k11): a checkout the store turned down with a note that only moves
the budget (the cart's total still inside it) or drops requirements is sent
again by Kyo for the same cart, once (after the steps due at the order). Over
races 11-13, runs whose cart still fit after such a note passed 0.18-0.84 when
they kept it and 0.00-0.22 when they changed it; in race 13 the note's wording
("a mere N cents") led the KYO line to change the cart in 4 of 11-13 such runs.
Not when the note sells a line out or adds a requirement, and not when it drops
one on a cheapest, most or least pick (a dropped requirement can let a better
listing in).

k13 (from k12): a step the request asks for just before an ending without a
purchase no longer holds a message back from the interview; steps at the order
still do, and the ending's step is done when a message is left then. In race 13
(two messages a task) the KYO line asked a single property question in 41-61%
of its runs against 21% for Know Thyself, keeping the other message for such a
step ("say so if nothing fits"), and spent it on "may I offer the closest
listing?" before 153 declines; none of the race's 839 declines passed. Over
races 9-13 an ending without a purchase passed on 0-2 tasks a race.

k14 (from k13): the checkout check runs on every task again, read by the light
model with room to think, up to the hard spend line, and its reading rules count
the words of the requested item's name where a listing's own attribute takes
them as a value. In race 13, on single-item tasks where the shopper's need came
out, v5 had opened a listing that passed for others in 19 runs and passed 12 of
them, against 20-23 of 20-24 for the three best versions: it bought a near twin
(a ready-to-drink shake for a protein powder, a "high pile solid" rug for a shag
rug) or a dearer fit. Its check was skipped above the soft spend line, and from
k4 it ran only on cheapest, most or least picks. On those race-13 records the
light model with the added sentence failed both twins and the dearer pick and
passed the right listings and three odd-looking listings that really passed
(24 of 27 calls); the heavy model passed both twins. Also from Know Thyself
(public as the overall #1): after the interview, each item with a new
requirement is searched once with its requirement values together, and the
first rows of its kind are opened (value_search).

k15 (from k14c): the question plan is asked of the light model too, on the same
prompt, and the two plans are merged by points for their places (eight for a
first place, seven for a second, ...; the heavy plan's points count 1.5 times;
the same property in other words is one pick). On the question bench of races
10 and 11 (100 hidden needs, 85 tasks), with two samples of each plan, the vote
named 2.5 more needs per 100 within one question, 2.5 within two and 2 within
the race's messages than the heavy plan alone; the light plan alone named fewer
within the race's messages (61-63 against 67) but more with its first pick
(36-37 against 33-34). Not when the request names what to ask first, which the
heavy plan leads with, nor late in the episode.

k15b (from k15): with two messages or fewer for the interview on a task of
several items, an item not asked about yet gets the next question before a
second question goes to an item already asked (with three or more, the plan's
order stands). On the question bench (races 10 and 11, k15's vote) this named
43.0 needs per 100 within two questions against 41.5, all on tasks of several
items, and 51.5 against 52.0 within three; in race 13 (two messages) Know
Thyself, which spreads its questions over the items, passed 5 of 48 runs of
three-item tasks with one hidden need against 2 for v5.

k15c (from k15b): a checkout the store holds until another call places it
("call confirm_purchase with this confirmation_code") makes that tool part of
the ending: the last steps, which allow only the ending tools, may still call
it, and one more step is kept for the finish from then on. On the practice
mutant with such a two-step checkout, k14b ended 2 of 15 episodes without an
order because, with one step left, the confirming call was turned away as not
an ending and the agent asked to check out again; Kyo v6, which does not narrow
the last steps, lost none.

k15d (from k15c): a listing's properties given as a list of name/value pairs
([{"name": "colour", "value": "red"}, ...]) are read as its attribute block,
and a money object ({"amount": 804, "unit": "cts"}) never is; a price written
with its unit ("145 cts", "$12.50") is read as a number. On the practice mutant
with pairs, money objects and nested paging, k15c's question plan saw only the
keys "amount" and "unit" (the price object) and none of the listings'
properties, and two tasks it passes otherwise lost the shopper's need.

k15e (from k15d): stock written as a word ("sold out", "out of stock",
"unavailable") counts like a false stock flag: such a listing is not buyable,
and a record whose short stock word turns into one of these is gone. Before,
only a false flag or a count of zero or less made a listing unbuyable in code.

k15f (from k15e): the call that places a held checkout (k15c) also passes the
second guard on the last steps, which turned away every call but the checkout
and the ending once the steps left were down to those kept for the finish; with
k15c's extra step kept, a confirming call two steps before the end was turned
away there.

k16 (from k15f): cost safety and a stock re-read.
- The light model's second plan (k15's vote) is asked only while the episode
  has spent less than VOTE_SPEND before it. On practice tasks the spend there
  is $0.009-0.016, so the vote always runs; in a world that costs 2.5-3 times
  as much (race 13 was about 2.5) it is skipped, which saves about $0.16 a run
  at three times the practice cost.
- The hard spend line is 0.065 (was 0.078). Replaying a practice run's calls at
  three times their cost (judges on the heavy model moved to the light one past
  the lean line), a 90-task run cost $5.20 with k15f's settings and $4.83 with
  these; the line cuts one more episode of 30 there and none at practice cost.
- A remark that says something sold out without naming the listing (a side
  remark in a shopper's answer, or a store note outside a checkout or an
  ending) makes Kyo re-read, before the next checkout, each cart line whose
  stock no record has shown since. The re-read uses a single-listing view that
  shows stock: when the offer or attributes view has shown none, another
  single-listing view of the store. A line the store's record then shows as
  sold out is replaced before the checkout; a line it shows in stock is
  bought (the store's record decides, as in races 11-12, where sell-out
  refusals of listings still in stock were probes). The practice watch task
  failed this way in 1 of 18 k-line runs: the remark came right after the
  line went into the cart, the model checked the line with an attributes view
  that has no stock field, and the pre-checkout re-read used that same view.
- k16b: that other single-listing view must look like a listing reader (its
  one parameter named like an id; no word of ending, ordering, the cart or
  messages in its name or description), so an ending tool with one text
  parameter is never called with a listing id when the brief left the ending
  role empty.

k17 (Touchstone v2 candidate, from k16b). Each change has its own switch among
the constants below the model names; with every switch off, k17 acts as k16b.
- Q1 (MENU, MENU_EVERY, MENU_HARD): a question may name up to eight of an
  item's properties ("What A, B or C would you like for the item?"). Its lead
  is the next pick in the plan's order, as a single question would be; after
  it come the item's picks no question has named yet, among the keys its
  opened listings carry with more than one value, leaving out a second
  spelling of a property already named ("size" and "size class") and the
  properties the light model finds hard for an everyday shopper (technical
  measurements, jargon). So a shopper who answers only a question's first
  property is asked as with single questions, and one who reads every property
  hears many more of them. Once a reply gives the value of a property named
  after the lead (and none for the lead), each next question leads with a
  property no question has named. A crowded reply turns menus off for the
  episode and asks the lead alone next (MENU_CONFUSED_OFF: a confused one
  too; off, since confusion stays on a property and the next question moves
  on). No pick whose property an answer has settled is asked again. Not with
  fewer than MENU_MIN_MSGS messages. (With MENU_EVERY off, only an item's
  first question is a menu; MENU_COVERS, off, counts a menu the shopper did not
  mind as a question on each property it names, which lost practice tasks.)
  Evidence: in race 14 (three messages) the shopper accepted menus (no crowded
  reply); a listed hidden property drew its value 64% of the time (74% with no
  hard key listed, 12% with three or more), the first listed hidden property
  0.85 at the first place, 0.67 at the second and third and 0.43 at the fourth
  to eighth, and the winning version's eight-key menus drew 0.36 needs a
  question against 0.19-0.27 for single questions (0.98 needs an episode
  against 0.56-0.81). Other shoppers answer only the first property: on the
  practice pack a hidden value came 25 of 25 times it was a menu's first
  property and 0 of 3 times it was later (a hoodie's material at the second or
  third place drew "I don't mind"; asked alone, "nylon" 4 of 4 times); in races
  10-11 a hidden key named first among two drew 61-77% against 81-88% alone,
  second 34-41%. In race 13 13% of the questions joining two things with "and"
  drew "one at a time" (5% of the others).
- k15b's spread (an item not asked about yet takes the next question) now acts
  with up to three messages (was two): in race 14 (three messages) the k line
  spent its first two questions on one item in 22% of multi-item runs, against
  4% for the versions that spread them, and drew 0.53 needs a run against the
  winner's 0.96.
- Q2: the listings read before the first question are k16b's (widen: up to two
  of each item's buyable rows, five at most), unchanged.
- Q3 (PART_CODES): the plan may ask the model code of a part the product holds,
  comes with or fits; only a code naming the listing itself is an identifier.
- F1 (ANSWER_SEARCH): after the interview, the words of the shopper's new
  values that no listing row has shown are searched with the item's kind noun,
  and up to two rows of that kind that the search shows for the first time are
  read; when there is no such word, or that search brings no new row, Know
  Thyself's search with the values together (k14) reads its rows cheapest
  first. Not past ANSWER_SEARCH_SPEND.
- F2 (MORE_OPENS, off): up to six more of each item's first twelve rows (eight
  in all) read after the interview; k4 found that more views were no lever.
- C1 (COVER): after the interview, the opened pages that show the most of each
  item's requirement words in their attributes are named. At the first
  checkout, once, a cart line whose attributes lack a word of the shopper's
  answers while another opened page of its kind shows them all (the cheapest
  such page), or another opened page of its kind that shows as many of the
  words for less, holds the checkout; the model decides. Not on cheapest, most
  or least picks.
- C2 (REQ_AUDIT): after the checkout check passes, a strict review marks each
  requirement met, broken or unstated on the cart lines of items with values
  from the shopper's answers and on up to four opened pages of their kind that
  show every word of those answers. A
  line with a broken requirement, or with an answer's requirement not met,
  holds the checkout when such a page meets every requirement (once per cart,
  twice an episode at most, below SPEND_SOFT).
- E1 (END_AUDIT): the review before an ending without a purchase (and the
  reviews after a refused ending, after a sold-out line and at a forced
  finish) marks each requirement on up to ten candidates of each item the cart
  lacks: its opened pages, then the rows that carry its requirement words.
  Listings meeting every requirement are the fits; when there are none,
  listings with nothing broken that show every requirement word are offered,
  with the rule that a requirement a page leaves unstated is no reason to end
  without a purchase (a fresh check still decides before Kyo buys one).
- E2 (END_SURE): when the brief names no tool for ending without a purchase,
  the light model is asked once which tool and arguments do it; the attributes
  view is switched only to a tool that reads like a listing view (k16b's guard:
  the switch could pick an ending tool while that role was empty); a forced
  ending that the schema check or the store turns down is fixed once from the
  reason and sent again, and a run that has still not ended tries its finish
  twice more.
- E3 (BUDGET_RULE): a note that moves the budget comes with the rule that a
  higher budget is no reason to switch to a dearer listing, and a lower one
  changes only the lines it rules out, each to its cheapest fitting listing.
- R1 (CALL_FIELDS): a 422 that asks for a call id, an idempotency key or a turn
  adds that field to every store call from then on; a result wrapped as
  {"calls": [{"observation": {"observation", "done", "error"}}]} is unwrapped.
  (k16b's two call ids are the chat's tool_call_id, on the model side.)
- R2 (CATEGORY_FIX): a field about stock or availability, or one whose values
  are all stock words, is never taken for the listings' category. With stock
  written as words ("in stock", "sold out") and no category list in the tools,
  the two stock words beat the real categories on the fewest-values rule, so
  items, key tables and the sold-out replacement review were grouped by stock
  (on the stock-word practice mutant k17a sent 4 menus in 15 tasks against
  13-14 on the other mutants).
- $: the hard spend line is 0.060 (was 0.065). Replaying two k17 practice runs'
  calls at three times their cost (judges on the heavy model moved to the light
  one past their lean lines), a 90-task run cost $4.92 at 0.065 and $4.78 at
  0.060; at 2.5 times, $4.25 and $4.20. The lower line cuts one more of 60
  episodes that had passed before its cart was complete, at 2.5 and at 3 times
  the practice cost, and none at practice cost (the dearest practice episode
  spent $0.058).
- $: the strict reviews (C2, E1) run on the heavy model only below AUDIT_LEAN,
  on the light model past it.

k17j (from race 14's episodes; local runs serve as collapse guards only):
- Questions:
  - After a reply that gives one value, the properties the menu named after it
    lead the item's next menu (a reply names one wanted value, and an item may
    hold several).
  - A confused menu is not sent again: the item's next question names only
    its next two properties, once.
  - Properties the light model finds hard for a shopper go to the back of the
    item's picks, for menus and single questions alike. In race 14 a menu with
    a hard key and no hidden one drew confusion 96% of the time, and with a
    hidden one the value came 74% with no hard key and 12% with three or more.
  - With as many messages as items or more, every item is asked once before
    any is asked twice (race 14, three messages: the k line spent its first two
    questions on one item in 22% of multi-item runs; multi-item tasks were 8
    of the winner's 15 extra passes). With more items than messages, the first
    question names one property of each item.
  - Picks that run out with messages left are followed by the item's most
    carried keys no question has named (in race 14 the k line sent two
    questions instead of three in 23% of its runs).
  - A reply asking for one thing a question is read from its words too.
  - Menu tails put keys that two listings share before keys with a value per
    listing.
  - One property spelled several ways is one key, in menus and in the plan's
    key table (love's merge_keys; 33% of the winner's menus named one property
    twice).
- Opens: F2 is on and aimed at the answers. For an item with a value from the
  shopper, its unread rows that carry the value's words are read first, then
  up to four more. In race 14 the k line opened the page carrying the value
  0.75 of the time after a need, against 0.90-0.92 for the versions opening
  about twelve pages.
- Endings:
  - An ending with an empty cart first runs the strict review of the pages
    read; when listings meet every requirement, Kyo buys them after a fresh
    check.
  - An ending with a line for every item and nothing broken in their records
    is held once.
  - An item's line is changed once unless something shows it wrong.
  - The coverage hint tells the agent to cart a page that shows every
    requirement word.
  - The race-14 case: a decline after fitting towers were opened, and one
    after two swaps over a delivery requirement no page could state, on tasks
    the other runs passed 0.83 and 0.93 of the time.
- From love v20 (public as the overall #1):
  - rearm: a change or a refused finish re-opens the one-shot checks,
    including k17's.
  - The checkout check sees each cart line's category and the neighbouring
    ones, with its Category and Pairs rules.
  - The cart is read from the cart tool's own reply when it lists the lines;
    otherwise the k line's arithmetic applies.

k17k (from k17j): the plan's picks are capped at 12 (16 in k17j; the menus'
fillers come from the key table at no model cost), and PLAN_LIGHT_LINE, off,
would have the light model plan first once the episode has spent that much
before the interview. Replaying 120 recorded episodes, a line at 0.0055 left a
world at 1.4 times practice cost untouched (spend before the plan p90 0.0039)
and saved $0.14-0.24 a 90-task run at 2.5-3 times; it moved 89 of 120 plans to
the light model at 3 times, whose plans named about 6 fewer needs per 100.

k17m (from k17k, for worlds not seen yet; local runs are collapse guards only):
- From love v20 (public as the overall #1):
  - A1 (SEARCH_TIPS): the guide says a keyword search with no rows means a
    word is missing from the listings (drop words, keep the product noun, or
    browse the category; no rows never shows the store lacks the item), that
    a search matching every word with too many rows takes all the required
    values in one query, and that details only in descriptions (what an item
    suits, its codes) are found by browsing the category and reading the
    rows.
  - A2 (STORE_RULES): a value the request ties to a rule the store states is
    the value that rule gives: one sentence in the guide, the reading rules
    and the checkout check; the chain check names it in "paired" (k2's
    pairing-rule field, so one field serves both).
  - A3 (PAIR_MUSTS): a requirement that one requested item suit or fit
    another item of the same order is a must of that item, not a lookup.
  - A4 (WIDEN_FOUR): a one-item task reads four of its listings before the
    questions (two before); key counts from three cards are mostly ties.
  - A5 (COMPARE_STEP): after a side-by-side call that showed nothing, the size
    steps down by half at most and never below what has worked (a stated
    limit can be wrong); a failure counts only when every id was a listing the
    store had shown; the opens joined into a failed call are not reported as
    shown.
- World shapes:
  - B1 (PEEL): an answer wrapped in an envelope ({"envelope": {"payload":
    ...}, "trace": ...}) is read as what it holds; notes, errors and other
    fields beside a wrapper keep it whole. On the wrapped-answer practice
    mutant the line scored 0.733: rows three levels down were missed by the
    row readers.
  - B2 (NESTED_PAGING): paging fields one level down (a "paging" object with
    the total and the next page) are named by their path and read, so the
    cheapest pick's own paging still runs.
  - B3 (OPENS_CAP): a cap the request sets on listing details is read by the
    brief and kept: the reads before the questions take a third of it at
    most (the opens after the answers are the aimed ones); Kyo's own opens
    (the widening, the review's row opens, the reads of listings about to be
    judged) leave two for the end; the chain and the re-read before a
    checkout may use those; the more-opens step is off; the status line
    shows how many are used; and a store answer that the allowance is used up
    stops every open Kyo makes. Every listing a detail or side-by-side call
    names counts, the model's own calls too.
  - B4 (TIGHT_STEPS): at 30 steps or fewer, no more listings are read after
    the interview and the value search gets half its steps.
  - B6 (MSG_CUT): a reply that says how many more messages the shopper will
    read ("I can only read one more message"), or a refused message, lowers
    the allowance; the interview then asks single questions on the plan's
    next property.
- V1 (VOTE_KEYS): the k15 vote merges the heavy and light plans by their
  labels' words, so one property named by its store key in one plan and by a
  plain word in the other ("body colour", "colour") split its points and a
  generic property led. On a practice task whose hidden need the story
  hinted, k17m lost 5 of 12 mutant runs that way (k17j shows the same split).
  The vote now counts two labels naming one store key on the item's opened
  listings as one pick, worded as the heavy plan words it; a label naming no
  key keeps its words.
- From Secretarybird v1 and lllll v1 (both public), for the k-line's known
  weak spots:
  - C1 (CODE_END): in race 13 a key ran out of credit and 210 calls failed.
    With the models gone (a 402, or three loop turns without a reply), no
    call is sent again. Roles a brief did not name come from the tools' names
    (lllll v1's guess_roles). With no brief, the budget comes from the
    request's figures. Answers no model could read are kept and read again
    when a model answers; the summary no longer calls them "none". After the
    loop, Kyo fills the cart in code (Secretarybird v1's code finish): per
    item, the seen listing that can be bought inside the budget and shows the
    most requirement and answer words, opened pages first, then the cheapest.
    It then checks out. A sold-out refusal swaps that line for the next one,
    four carts at most. The forced finish fills required arguments in code.
  - C2 (SEARCH_KIND): two first-page searches whose queries extend one
    another show the store's word rule. More matches with more words means any
    word matches; fewer means every word is required. Equal results from
    queries agreeing on their first n words show that only n words are read.
    Each search gets a short note on what that means for the next query. A1's
    "all the values in one query" now comes only in a store that needs every
    word. In such a store, when every match is shown, the rows found carry the
    query's words; the answer-aimed opens and the strict review count them.
  - C3 (OPEN_EACH): the questions are planned from the item's opened pages,
    so in a several-item task each item gets two opened listings of its kind
    before them. An item with no rows gets its kind noun searched. An item
    still without one is asked about after the others.
  - C4 (STUCK_SPARE 3): after three loop turns in a row that used no store
    step, the spare model leads, until a turn moves again.
  - C5 (PRICE_AGAIN): a checkout refused over a moved price alone is sent
    again (k12's re-checkout) when the re-read cart still fits the budget, on
    any pick but cheapest, most or least.
- Left out on purpose: Secretarybird's wider spend lines for several items
  (past $5 a run at 3x), lllll's at-once decline of three-item tasks (a
  race-14 version passed 10 of 34 of them), and Secretarybird's merging of
  "A & B" categories into two items (no evidence).
- k17m2 (LEVER_GUARD), a safety treatment and no new lever: k16b had three
  exception handlers, and the levers added since (about 800 lines: the
  menus, the plan's adjustments, the answer-word search, the more opens, the
  coverage hint and hold, the two strict reviews, the decline guards, the
  search rule, the code finish, the reads before the questions, the spare
  model, the price re-checkout, rearm, the cart sync and the A and B parts)
  each ran unguarded. Practice and its mutants raised nothing, but a world
  never seen could reach a rare path. Each lever's entry point is now
  guarded: an error turns that lever off for the rest of the episode (a
  "lever_error" trace entry) and the path without it runs (single questions
  for the menus, Kyo's own cart arithmetic for the sync, k16b's side-by-side
  size, no hold for a guard, and so on). The code finish (C1) is the last
  safety net, so an error in it does nothing.
- k17m3 (SOLD_BUY): no ending without a purchase after a sell-out. A cart
  line that sold out leaves the agent searching for a replacement, and when it
  finds none it ends without a purchase. That ending did not pass once in race
  11 (0 of 39, while a replaced line passed 127 of 237) nor in this line's
  practice runs (0 of 30, while 291 of 342 orders after a sell-out passed).
  So once a line has sold out, an ending without a purchase is taken over
  once: Kyo buys per item the replacements the sold-out review named that are
  still buyable inside the budget, in its order, else the code finish's pick
  (C1: the seen listing of the item a strict review found fitting, else the
  one with the most requirement words, opened pages first, then the
  cheapest; never one a strict review marked broken). The request's steps at the order run
  first, and a line that sells out again is swapped for the next. When that
  checkout does not go through, the ending is held once with the cart as Kyo
  set it. The forced finish does the same with an empty cart.
- k17m3 (SOLD_SEEK): the takeover can only buy what was found, and in the
  practice runs the forced buys failed where the fitting replacement had not
  been seen. So when a cart line sells out and no listing read so far
  replaces it, Kyo runs k2's value searches at once (each plain value with
  the item's kind noun, the next page when the first shows nothing new; up
  to three new rows opened) instead of only before an ending with eight
  steps left, then asks the sold-out review again. Only with 12 steps or
  more left.
- k17m4 (SOLD_REVIEW): after a sell-out the replacement is checked out mostly
  unreviewed: k17's strict review (C2) needs answer words, takes two carts at
  most and stops at the soft spend line. Replacements a strict review had
  found fitting passed 45 of 49 in this line's practice runs, unreviewed ones
  501 of 578 (74 of 100 when the model chose a listing the sold-out review
  had not named), and in race 11 110 of 237 replaced lines failed. So in an
  episode with a sell-out, each cart that passes the checkout check gets the
  strict review once (up to the hard spend line): every line, requirement by
  requirement, with the sold-out reviews' named fits and the opened listings
  of its kind beside it. A line with a requirement broken while another
  listing meets every one holds the checkout once, naming that listing; on a
  cheapest pick a cheaper listing meeting every one does the same.
- k17m4 (EMPTY_SEEK 12): k2's value searches run before an ending only with
  more than eight steps left, and on cheapest picks the agent spends its
  steps checking prices and ends late: of the 31 cheapest-pick endings without
  a purchase and without a sell-out in this line's practice runs (none
  passed), 26 had seen no fitting listing and most had not run the searches.
  So when 12 steps or fewer are left with an empty cart, no checkout tried and
  no reviewed lead, the searches run once and what they opened is named.
- k17m5 (COSTLY): at three times practice prices this line's 90-task run
  costs $5.10 by the by-model replay (hard line 0.060), and in race 12 runs of
  $5.0-5.1 were killed about half the time. The spend through the first
  question-plan call tells a costly world: 0.007-0.018 at practice prices, so a
  line of 0.040 is passed by no episode at 1.4 or 2 times practice prices, by 6
  of 330 at 2.5 times and by 95 at 3 times. Such an episode's hard spend line
  drops from 0.060 to 0.048, never closer than 0.004 above what it had spent
  when read (replay over 405 episodes: 3x $5.10 -> $4.83, 2.5x $4.56 ->
  $4.54, 1.4x and 2x the same; procedure-heavy runs alone 3x $5.18 -> $4.92). Only the line moves; every place that used the hard line uses the
  episode's own, and the loop that runs the interview lowers its cap with it.
  An episode without a question plan keeps 0.060.
- k17m6 (SPARE_TELL): race 7 graded a message with the basket and its total
  (passing orders sent one every time, violating ones 61%), and this line
  sends one only when a stated step asks for it. At a checkout the interview
  and the held message are over and the loop sends no messages of its own,
  so a message still free then is one no question will ever use: it tells
  the shopper the final cart (units, name, id, price) and its total, right
  after the order gate. Only when the request stated its allowance, no step at
  the order takes messages (the gate tells the cart for those), the cart has
  changed since the last such message and the steps left keep the checkout.
  A reply that states a new requirement holds the checkout once.
- k17m7 (SWAP_RESERVE 2): on cheapest picks the agent opens listings in
  price order until the steps run out, and a sell-out at the checkout then
  leaves no step to swap the line: of 82 cheapest-pick practice episodes with
  a sell-out, 18 used every step and 14 of those failed (21 of the 64 with
  steps left). A swap after a turned-down checkout takes three steps at the
  least (the line out, the replacement in, the checkout again). So on a
  cheapest pick whose cart holds a line for every item, before any sell-out,
  the steps kept for the finish grow by the swap's two cart calls and the
  checkout again; the first sell-out gives them back for the swap.
- k17m11 (ROTATE): after a menu reply that drew nothing (calm or confused),
  the item's next question is a menu again: it leads with a property no menu
  has named yet (the plan's next pick), and the last menu's properties after
  its lead fill the rest in reverse order (a hidden key at places 4-8 was
  drawn 0.21-0.30 of the time in race 15, against 0.63-0.69 at places 1-3).
  The message held for the store's first cart answer asks such a menu too.
  The plan names up to 16 properties, so later menus have fresh ones. From
  races 14-15: a fresh lead drew more needs than a re-listed one in every
  version; single re-asks of a menu's property drew 0.03-0.06.
- k17m11 (TIME_BOUND): a model call late in the task gets a client whose
  timeout fits the task's 300 s less CALL_MARGIN (whole 5 s steps, one
  attempt: the runtime client's max_retries counts attempts); with under 5 s
  of room no call is sent. The usual client (one 50 s attempt) could run a
  call started at 279 s to 329 s, past the network's stop.
- k17m10 (HEALTH): run-time checks for a store whose listings the code cannot
  read. After BLIND_AFTER answers of the reading tools that carry text but no
  listing record, Kyo counts itself blind: the loop model is told to read the
  raw results itself; texts holding listing ids keep room in its view; the
  levers that need records stop spending steps (EMPTY_SEEK, the decline-time
  seek, the reference trail); the interview starts after a few steps; an
  ending without a purchase is held once for a re-read. Always on: the forced
  finish also trusts Kyo's own record of its cart calls when no listing was
  read (it used to end without buying with lines in the cart); long or id-
  bearing listing texts are not read as store notes; the interview starts
  with fewer steps left when the allowance is small; a plan that names
  nothing asks PLAIN_PICKS; a record with no nested block but three or more
  plain attribute fields counts as read in full; a brief with no messaging
  tool gets one guessed from the tool names; a messaging tool's other
  required fields are filled plainly.
- k17m9 (TEXT_ROWS, KEYLESS_MENU): race 15 sent every listing as one text
  line ("ref_ref: p-.. | title: .. | price: 44999 | facts: color=silver;
  ram=16gb | item_blurb: .."). The ledger reads dict records only, so it held
  no listing: no attribute keys, so the menus (which name only keys the opened
  listings carry) fell back to single questions (11% of questions, against
  91-98% for the race's top three), and the loop model saw each line cut at
  450 characters, before most of its properties. A run that did not draw
  every hidden key passed none of its tasks. TEXT_ROWS reads such a line as a
  record where every store result comes in (act) and in the store's first
  view: the ledger, the menus and the model's view then see fields, each cut
  on its own. A record sent as a JSON string is decoded too, and a long
  listing text that no reader takes in keeps TEXT_ROOM (1,600) characters in
  the model's view instead of 450. KEYLESS_MENU: when no attribute key of an item's opened
  listings shows two values or more (no key read, in a shape no reader here
  knows, or a single listing opened when the questions start), a question
  names the plan's own next picks for the item, hard ones left out, as the
  race-15 winners' menus did (5-8 properties each).
- k17m8 (SOLD_TWIN): when a cart line sells out, the sold-out review names
  replacements only among the listings read so far, and SOLD_SEEK's value
  searches run only with 12 or more steps left and once an episode. In this
  line's practice runs the review named no fit at 48 first sell-outs (65%
  passed, 80% when it named one); at 6 of them the catalogue held a listing
  with the sold-out one's very name, in stock and inside the budget, that the
  episode had not seen, and all 6 failed. So when the review names nothing,
  Kyo searches the sold-out listing's name, opens up to two in-stock,
  in-budget listings of its kind whose name is the same or shares most words,
  and asks the review again (past the soft spend line, where the review does
  not run, the same-name ones are named by code). Once per sold-out line.
- B5 (procedures): the interview, the fillers and the held message all count
  only the messages the request's steps leave free (msgs_free), and the
  message held back for the first cart answer (k8) keeps a store notice's
  procedure possible with three messages: unchanged, checked on the four
  procedure mutants.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

LIGHT = "anthropic/claude-haiku-5.5"
HEAVY = "anthropic/claude-sonnet-5.5"
SPARE = "openai/gpt-6-luna"
# k2: models tried for the question plan before HEAVY. Gemini 3.1 Pro led on the plan bench's synthetic hints, but on
# the calibrated real-style requests every planner drew out the same share, and in full RaceSim runs it drew out fewer
# needs than HEAVY (all disclosed 0.46 vs 0.49 race 10, 0.49 vs 0.54 race 11): the plan is Kyo's again
PLANNERS: list[str] = []
# k15: the light model's plan joins the heavy one in a vote (see the module notes); the heavy plan's places count this
# many times, and no second plan is asked for after VOTE_SECONDS
VOTE_WEIGHT = 1.5
VOTE_SECONDS = 150
# k15b: at most this many messages for the interview, an item not asked about yet comes first (see the module notes)
SPREAD_MESSAGES = 3         # k17: was 2 (race 14 had three messages; see the module notes)

LOOP_CALLS = 42
ALL_CALLS = 64             # Kyo v6's 60, plus the answer reads of a one-at-a-time interview and k15's second plan
SPEND_SOFT = 0.055
SPEND_HARD = 0.060          # k17: was 0.065 (k16: 0.078; see the module notes)
# k16: the light model's second plan is asked only below this episode spend (see the module notes)
VOTE_SPEND = 0.025
# k17 switches (see the module notes): each one turns its mechanism off by itself
MENU = True                 # Q1: questions name several properties of an item (see the module notes)
MENU_EVERY = True           # every question may name several (its lead in the plan's order, then properties no question named)
MENU_HARD = True            # properties the light model finds hard for a shopper stay out of menus after their lead
MENU_CONFUSED_OFF = False   # a confused reply turns menus off too (only a crowded one does when False)
MENU_MIN_MSGS = 2           # no menu with fewer messages than this for the interview
MENU_MIN_PROPS = 3          # a menu names at least this many properties
MENU_MAX_PROPS = 8          # and at most this many
MENU_COVERS = False         # a menu the shopper understood counts as a question on each property it names
PART_CODES = True           # Q3: the plan may ask a part's model code
ANSWER_SEARCH = True        # F1: answer words no row has shown, searched with the item's kind noun
ANSWER_SEARCH_SPEND = 0.04
MORE_OPENS = True           # F2: more of each item's rows read after the interview (k17j: aimed at the answers' values)
MORE_OPENS_SPEND = 0.05
COVER = True                # C1: pages ranked by the requirement words their attributes show; one checkout held
REQ_AUDIT = True            # C2: the strict review of the cart and fuller pages before a checkout
END_AUDIT = True            # E1: the strict review before an ending without a purchase
END_SURE = True             # E2: the ending tool found, the attributes view guarded, a turned-down ending fixed once
BUDGET_RULE = True          # E3: a budget change moves only the lines it rules out
CALL_FIELDS = True          # R1: fields the store asks every call to carry; wrapped results
CATEGORY_FIX = True         # R2: a stock or availability field written as words is never taken for the category
MENU_FILL = True            # k17j: picks run out with messages left: the items' most carried keys no question named
KEY_MERGE = True            # k17j: one property spelled several ways is one key (love's merge_keys)
CHECK_DEPTS = True          # k17j: the checkout check sees each line's category and the neighbouring ones (love)
REARM = True                # k17j: a change or a refused finish re-opens the one-shot checks (love's rearm)
CART_SYNC = True            # k17j: the cart from the cart tool's own reply when it lists the lines (love)
EMPTY_GUARD = True          # k17j: an ending with an empty cart: the strict review of the pages read, Kyo buys a fit
FULL_GUARD = True           # k17j: an ending with a line for every item and nothing broken is held once
SWAP_ONCE = True            # k17j: an item's line is changed once unless something shows it wrong
PLAN_LIGHT_LINE = 0.0       # k17k: past this spend before the plan the light model plans first (0: off; see the notes)
# k17m switches (see the module notes)
SEARCH_TIPS = True          # A1: the guide's search advice (love)
STORE_RULES = True          # A2: a value tied to a rule the store states decides (love)
PAIR_MUSTS = True           # A3: one item fitting another of the same order is a must, not a lookup (love)
WIDEN_FOUR = True           # A4: a one-item task reads four listings before its questions (love)
COMPARE_STEP = True         # A5: side-by-side sizing after a failed call (love)
PEEL = True                 # B1: an answer in an envelope is read as what it holds
NESTED_PAGING = True        # B2: paging fields one level down are read
OPENS_CAP = True            # B3: a stated cap on listing details is kept
OPENS_KEEP = 2              # B3: details Kyo's own opens leave for the end
TIGHT_STEPS = 30            # B4: at this step allowance or below, no more opens after the interview
MSG_CUT = True              # B6: a cut message allowance is obeyed
VOTE_KEYS = True            # V1: two plans naming one store key in other words are one pick in the vote
CODE_END = True             # C1: with the models gone, roles guessed, unread answers kept, and the cart filled and bought in code
CODE_END_ROUNDS = 4         # C1: carts the code finish tries (a sold-out line is swapped for the next listing)
SEARCH_KIND = True          # C2: the store's word rule read from two searches; a note on each search; value hits in an every-word store
OPEN_EACH = True            # C3: every item of a several-item task has two opened listings before the questions
STUCK_SPARE = 3             # C4: after this many loop turns in a row that used no store step the spare model leads (0: off)
PRICE_AGAIN = True          # C5: a checkout refused only over a changed price is sent again when the cart still fits
LEVER_GUARD = True          # k17m2: an error in a lever turns it off for the episode and the path without it runs
SOLD_BUY = True             # k17m3: after a sell-out, an ending without a purchase becomes a purchase of the best replacement
SOLD_SEEK = True            # k17m3: a sell-out no listing read replaces: Kyo's own value searches run at once
SOLD_REVIEW = True          # k17m4: after a sell-out, each cart the checkout check passes gets the strict review once
EMPTY_SEEK = 12             # k17m4: with this many steps left, an empty cart and no reviewed lead, Kyo's value searches run (0: off)
COSTLY_LINE = 0.040         # k17m5: spend through the first question plan above this marks a costly world
COSTLY_HARD = 0.048         # k17m5: the hard spend line of an episode in a costly world (0: off)
COSTLY_ROOM = 0.004         # k17m5: a lowered line stays at least this far above the spend when it was read
SPARE_TELL = True           # k17m6: a message no question will use tells the shopper the final cart and its total
SWAP_RESERVE = 2            # k17m7: a full cheapest-pick cart keeps a swap's two cart calls (and the checkout again) free (0: off)
SOLD_TWIN = True            # k17m8: a sell-out no listing read replaces: the sold-out listing's own name is searched for twins
TEXT_ROWS = True            # k17m9: a listing sent as one text line of "key: value" fields is read as a record (race 15)
TEXT_ROOM = 1600            # k17m9: characters the model sees of a long listing text no reader took in (others: SHOW_CHARS)
KEYLESS_MENU = True         # k17m9: no attribute key on an item's opened listings: menus from the plan's own picks
HEALTH = True               # k17m10: run-time checks for a store whose listings the code cannot read (see the module notes)
ROTATE = True               # k17m11: after a menu that drew nothing, a menu led by an unnamed property (see the module notes)
TIME_BOUND = True           # k17m11: a late model call's timeout fits the task's time left (see the module notes)
TASK_LIMIT = 300.0          # k17m11: seconds the network gives a task, finished or not
CALL_MARGIN = 20.0          # k17m11: seconds kept after the last model call for the store calls that finish the task
BLIND_AFTER = 3             # k17m10: reading answers with text but no listing record read before Kyo counts itself blind
PLAIN_PICKS = ("type", "size", "colour", "material", "style")   # k17m10: asked when the question plan names nothing
RECIPIENT_WORDS = re.compile(r"\b(?:recipient|to|target|audience|who|addressee|receiver)\b", re.I)
SUBJECT_WORDS = re.compile(r"\b(?:subject|title|topic|headline|summary)\b", re.I)
BLIND_NOTE = ("[Kyo] Kyo's own code cannot read the listing records in this store's answers (they come in a form it does "
              "not know), so its checks see no listings and its cart record is only the cart calls made so far. Read each "
              "listing's id, name, price, stock and properties from the raw results yourself, keep every requirement in "
              "mind, open the listings that might fit, and buy the ones that fit every requirement.")
AUDIT_LEAN = 0.02           # the strict reviews go to the light model past this episode spend
# k5: the heavy model's reviews run only below this episode spend; past it the light model decides (see the module notes)
TAKE_OFF = re.compile(r"remov|withdraw|discard|delet|drop|releas|clear|take|minus|decreas|reduc|subtract|unset|\bout\b|\bdel\b", re.I)
# k15e: a short stock word that means the listing cannot be bought (see the module notes)
SOLD_WORD = re.compile(r"sold[ -]?out|out of stock|\bunavailable|not available|none left|no stock|discontinued", re.I)
# k16b: what an ending, ordering, cart or message tool says of itself, and a parameter named like a listing id
NOT_READER = re.compile(r"final|no more actions|give[ _-]?up|abandon|\bend(?:s|ing)?\b|finish|close|conclu|check[ _-]?out|purchas|"
                        r"\border|cart|bag\b|basket|message|shopper|customer|remark|refus|declin|quit|\bstop", re.I)
ID_PARAM = re.compile(r"(?:^|[_\-.])(?:id|ids|pid|sku|item|product|listing|article|code|ref|key|target)(?:$|[_\-.])|id$", re.I)
# k16: a remark that something sold out or is gone, in a shopper's answer or a store note (see the module notes)
SOLD_NOTE = re.compile(r"sold[ -]?out|out of stock|\bunavailable|not available|none left|no stock|discontinued|\bgone\b|"
                       r"no longer (?:available|in stock|for sale|sold|offered|carried)|\b(?:ran|run|running) out\b|snapped up", re.I)
SHOPPER_SIDE = re.compile(r"custom|shopper|\buser\b|buyer|client", re.I)
REMOVAL = re.compile(r"\b(?:remov\w*|take\w*\s+(?:\w+\s+){0,3}?out|swap\w*|replac\w*|substitut\w*|withdraw\w*|drop\w*|discard\w*)", re.I)
LEAN_LINE = 0.025
LIGHT_ROOM = 6000           # the light model thinks about 2,000 tokens on a review: a smaller limit cut its replies short
LIGHT_SECONDS = 255         # later than this a review goes to the heavy model, which answers in about half the time
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
WAITS = (4, 10, 20, 35)
WAIT_TOTAL = 45
SHOW_CHARS = 450
OBS_LIMIT = 15000
HISTORY_LIMIT = 150000
GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
# the episode has 300 s from the child process start, and the agent's clock starts 1-3 s later
LOOP_SECONDS = 240          # the loop and the interview take no new turn after this
RESCUE_SECONDS = 250        # an empty cart at a forced finish is reviewed and bought only before this
GATE_SECONDS = 265          # the request's steps at the order or the ending are skipped after this
CALL_SECONDS = 280          # no model call starts after this
RESCUE_CALLS = 3            # model calls the empty-cart rescue may make beyond ALL_CALLS
AUTO_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_OPENS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout
# k2: a wished value that is not a plain search word (an exclusion, a limit, a choice between values, a comparison)
PLAIN_SKIP = re.compile(r"\b(not|no|never|without|unlike|except|other|different|under|over|below|above|least|most|less|more|"
                        r"than|up|between|or|any|either|same|compatible|match|matching)\b|[<>=\d]", re.I)
# k2: the same for the decline-time value search, where an exact figure ("2.4", "1 pair", "4k") is a good search word
VALUE_SKIP = re.compile(r"\b(not|no|never|without|unlike|except|other|different|under|over|below|above|least|most|less|more|"
                        r"than|up|between|or|any|either|same|compatible|match|matching|within|at)\b|[<>=]", re.I)
# k2: capitalised words that start a description rather than name a shop or brand
CHASE_STARTERS = {"For", "The", "Its", "This", "That", "These", "Those", "From", "Check", "Take", "Look", "With", "When",
                  "Where", "Versions", "Read", "See", "Note", "Please", "Could", "Compatibility", "Listing", "Page"}

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_llm_bounded: dict[int, Any] = {}


def _llm_for(room: float) -> Any:
    """k17m11 (TIME_BOUND): the model client for `room` seconds: the usual one with 50 s or more, else one whose timeout
    is the room in whole 5 s steps (one attempt; the runtime client's max_retries is its number of attempts)."""
    if room >= 50:
        return _llm
    limit = max(5, int(room) // 5 * 5)
    if limit not in _llm_bounded:
        _llm_bounded[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
    return _llm_bounded[limit]
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


GUIDE = """You are Kyo, a careful buyer working for one shopper in an online store. You can act only by calling the store's tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.

HOW THE STORE WORKS
- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.
- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.
- Search rows are summaries. Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once Kyo knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.
- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.
- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).

WHAT COUNTS AS A FIT
- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.
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
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Prices that a few searches happened to show prove nothing about the budget either: before ending because the items cost too much together, find each item's cheapest fitting listing (search its kind alone, with the store's price filter or order when it has one). Only the ledger's requirements count: a fact the shopper never gave (which device or person it is for, a model or size they did not state) is no requirement, so never end without a purchase for want of it. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""


BRIEF = """You prepare a shopping task for an agent. Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's words, "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright, or the shopper says they will check or double-check that part themselves, which marks it as wanted (a trip said to be for hiking, which the shopper will verify, gives "occasion: hiking"); and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to "name". A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": a list of things the shopper would merely like.
"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).
"process": a list of the steps the request (or a store tool's description) requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""


PLAN = """You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.

Choose the properties in this order:
0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them, except that the wearer's gender comes before a look property when both apply to an item.
Name a hinted property by its plain name: a hint about who will wear it points at gender; a hint about its looks points at whichever look property (colour, pattern, decoration or finish) the request leaves open and the most of that item's listings carry in the key table; a hint about where or how it will be used (the weather or water it meets, the device it must work with, the power or connection at hand) points at the property that decides that use (water resistance, compatibility or form factor, power source, connectivity), named as the key table names it.
2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item's listings; avoid keys seen on only one or two of them.
Skip a property the request already settles for that item, gender when the key table shows no gender key for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty, condition and other logistics. With several items, put every item's hinted properties first, then take turns between the items.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property's name, no example values, no brackets, no second property.
Choose up to COUNT, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}"""


# k17 (Q3): see the module notes
# k17 (Q1, MENU_HARD): see the module notes
HARD = ("A shopping assistant is about to ask a shopper which values they want for several properties of a product at"
        " once. Name only the properties that take technical knowledge to answer: engineering measurements,"
        " specifications or codes that an ordinary buyer could not judge (such as an amplifier's impedance or a lens's"
        " focal ratio). Everyday properties of looks, fit, size, material, style, closure, season, use or the person it"
        " is for are never hard, and most lists have none. Return ONE JSON object and nothing else: {\"hard\": [the"
        " property names copied exactly; empty when none]}")
PLAN_PART = (" A code counts as an identifier only when it names the listing itself; the model code of a part that the"
             " product holds, comes with or fits is one of its properties and may be asked.")


HEAR = """A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null, "steps": [{"step": a procedure an answer asks the agent to follow besides choosing the items (getting the shopper's approval, telling the shopper the cart or its total, checking something with them), in at most 15 words, "when": "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes}]}
- "steps": only procedures an answer itself asks for; [] when none.
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Words of firmness keep the value wanted even when they are worded negatively: saying that X is not negotiable, not to be traded away or not to be changed asks for X; a need is a negation only when the answer turns the value itself away (wanting anything except X, or no X). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value."""


NOTE = """While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "steps": [{"step": a procedure the note asks the agent to follow besides choosing the items (getting the shopper's approval, telling the shopper the cart or its total, checking something with them), in at most 15 words, "when": "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes}], "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note. "steps" lists only procedures the note itself asks for; [] when none."""


CHECK = """You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.
Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing's value counts as met unless the records show that same value; a rule of going with, working alongside, pairing with or being compatible with another listing's value is met by the value the store's pairing rule pairs with that listing's value (the requirement's note gives it when known), not by that listing's own value unless the rule pairs that value with itself. The words of the requested item's name are requirements too where a listing's own attribute takes them as a value: a style, type, form or closure the name states (shag, velcro) fails a listing whose attribute for that property says something else, whatever its title says. A word the category label also states (a season or an audience in the department's name, a product form the department names) is met by that category, and a word that only names the kind of product (dresser, jacket, sneakers) is met by any listing of that kind. A requirement that takes its value from another listing's page is judged only against the value its note gives (the page it points at gives ...); while no such value is given it is not yours to judge, since the agent reads that chain of pages itself, so never fail a cart because the records do not show that other page. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget (a listing's price figure is in the budget's unit unless its record names another unit: a price of 7 under a budget of 99 cents is 7 cents, inside it). The newest record of a listing wins. A detail of the shopper's story that names no property value (who the purchase is for, what it will be used with) is not a requirement by itself: the requirements in force list what the shopper asked for.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses."""


READING = """the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; a rule of going with, working alongside, pairing with or being compatible with another listing's value is met by the value the store's pairing rule pairs with that listing's value (the requirement's note gives it when known), not by that listing's own value unless the rule pairs that value with itself; out-of-stock listings and listings that break the budget do not fit (a listing's price figure is in the budget's unit unless its record names another unit); a detail of the shopper's story that names no property value (who the purchase is for, what it will be used with) is not a requirement by itself."""

AUDIT = """A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).
A link is a listing only when its page shows everything the description says about it: its shop or brand, its kind, and every property and value named after it ("that has finish matte, width 40 cm and handle brass" names three). A shop often sells near twins of that listing: an OPENED page that leaves one of those properties out, or shows another value for it, is not the link. A ROW shows too little to rule a listing out, so give the ROW that best fits the shop or brand and kind as the link to be opened next. When no record can be the link, give its id as "" and describe it, and Kyo searches further.
Some requirements ask for a value that goes with, works alongside, pairs with or is compatible with the value on the listing they point at, not for that value itself. For such a requirement, also find the store's pairing rule for that property in the OPENED pages' words (a sentence saying which values go together, such as "A goes with B" or "pairs A with B"; often on the last page itself) and give the value it pairs with the last page's value as "paired" (the same value when the rule pairs it with itself). "paired" is "" when no such rule is in the records, or when the requirement asks for the same value, a different value or a limit.
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing}], "done": true when the last link is an OPENED page that states the property's value itself, else false, "value": the property's value as stated on that last page, or "", "paired": the value the store's pairing rule pairs with it, or ""}]}
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

SWAP = """A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: """ + READING + """
Return ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], "reason": "one sentence"}"""

# k17 (C2, E1): the strict review, one requirement at a time
REQS = """You review listings for a shopping agent strictly, one requirement at a time. For each listing below and each requirement of its item, give "met" when the record states the required value for that property, "broken" when it states another value for that property, and "unstated" when the record says nothing either way. Reading rules: """ + READING + """
Records marked ROW were seen only as a search result row: mark them on what the row shows.
Return ONE JSON object and nothing else: {"marks": [{"id": listing id, "req": {"<the requirement copied exactly>": "met" or "broken" or "unstated"}}]}"""

# k17 (E2): the ending tool when the brief named none, and a turned-down ending's arguments
ENDING = """A shopping agent may have to end its task without buying anything. Name the one store tool below that ends the task with no purchase, and the fixed arguments that make it end that way (only the arguments that choose that way of ending; {} when none do). Return ONE JSON object and nothing else: {"tool": the tool's exact name, or null when no tool ends the task without a purchase, "args": {...}}"""

REFIX = """A shopping agent called a store tool to end its task without a purchase, and the call was turned down for the reason below. Using the tool's schema, give arguments that make the call valid; keep any remark short and name no listing. Return ONE JSON object and nothing else: {"args": {...}}"""

# k17j: from love v20 (public), for the checkout check (see the module notes)
CHECK_CATS = ("Category: when the CATEGORIES section shows a store category whose name carries the item's own words while a"
              " cart line's category does not, that line comes from a neighbouring category and is wrong. Pairs: when the"
              " request or a requirement says one item must work with, fit or suit another requested item (or a model the"
              " shopper names), the two listings' own words must connect them (a code or name that both records carry, or a"
              " statement on one record that covers the other); a pair whose records show no such link is wrong.")
CHECK_ANCHOR = "When the pick rule is cheapest, most or least, also look at the other listings shown"
# k17j: a reply that asks for one thing a question, in its own words
CROWDED_WORDS = re.compile(r"one (?:thing |question |item )?at a time|single file|one by one|space (?:them|those|these|it) out|"
                           r"(?:ask|send) (?:them |each )?separately|too many (?:things|questions)|piling (?:up|on)", re.I)
# k17j: keys that are store logistics or identifiers, never filler questions
FILL_SKIP = re.compile(r"warranty|condition|shipping|delivery|returns?\b|price|cost|stock|availab|seller|vendor|\bstore\b|"
                       r"\bids?\b|sku|upc|ean|barcode|rating|reviews?\b", re.I)
# k17 (E3): see the module notes
BUDGET_UP = "(A higher budget is no reason to switch to a dearer listing: keep every line that still meets its requirements.)"
BUDGET_DOWN = "(With a lower budget, change only the lines it rules out, each to the cheapest listing that meets every requirement.)"
# k17 (C1): words of a requirement value that say nothing of the product, and values that are no plain words
COVER_SKIP = {"the", "and", "for", "with", "that", "this", "any", "one", "some", "very", "really", "just", "like", "kind", "type",
              "style", "please", "would", "want", "need", "prefer", "something", "thing", "only", "also", "but", "are", "has",
              "have", "its", "all", "from", "made", "will", "can", "should", "must", "able"}
# k17 (R2): a field about stock or availability, and values that are stock words
STOCK_FIELD = re.compile(r"stock|avail|inventor", re.I)
STOCK_WORDS = re.compile(r"\b(?:in|out of|low|no)\s+stock\b|sold[ -]?out|\b(?:un)?available\b|backorder|discontinued|none left", re.I)
COVER_VALUE_SKIP = re.compile(r"\b(?:not|no|never|without|other than|unlike|except|alongside|listing|page|shown|at most|at least|"
                              r"under|over|below|above|less|more|between|within|up to|either)\b|[<>=]", re.I)


SHAPE = """A store tool returned the result below. Name its parts by the exact keys the result and the tool's parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing's price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}"""


# k17m (A1-A3, B3): sentences from love v20 (public) and the brief's detail cap, each behind its switch
SEARCH_TIPS_TEXT = (" A keyword search that returns no rows usually means one of its words is not on the listings: drop words"
                    " (keep the product noun) or browse the item's category instead; no rows never shows that the store lacks"
                    " the item.")
# k17m (C2): the narrowing advice holds only where every word is required; with C2 it comes in the note after a search
SEARCH_NARROW = (" When a keyword search matches every word and returns too many rows to see, put all the item's required values"
                 " in one query to narrow it.")
if not SEARCH_KIND:
    SEARCH_TIPS_TEXT += SEARCH_NARROW
DESC_TIP = ("\n- Some details (what other products an item suits, codes it carries) may appear only in a listing's description,"
            " which a keyword search may not read: browse the item's category, sliced by brand or a price ceiling when the"
            " store allows, and read the rows' descriptions.")
RULE_GUIDE = (" When the request ties the value to a rule the store states, the value that rule gives decides: apply the rule"
              " exactly as the page states it.")
RULE_READ = "a requirement the request ties to a rule the store states is met by the value that rule gives; "
RULE_TRAIL = ("Likewise, when the request ties the requirement's value to another rule the store states on those pages, give the"
              " value that rule gives as \"paired\". ")
PAIR_LOOKUP = (" A requirement that one requested item must suit or fit another item of this same order is not a lookup: keep it"
               " as a must of that item in the request's own words (\"<property>: <relation> the <other item> in this order\")"
               " and leave it out of lookups.")
OPENS_FIELD = ("\"opens\": how many listings' full details the request allows the agent to open in all (integer), or null when"
               " it sets no such limit.\n")
if SEARCH_TIPS:
    GUIDE = GUIDE.replace(" A filtered search that returns no rows", SEARCH_TIPS_TEXT + " A filtered search that returns no rows", 1)
    GUIDE = GUIDE.replace("\n- Search rows are summaries.", DESC_TIP + "\n- Search rows are summaries.", 1)
if STORE_RULES:
    GUIDE = GUIDE.replace("and never end without a purchase because of it alone.",
                          "and never end without a purchase because of it alone." + RULE_GUIDE, 1)
    # the reading rules sit inside the review prompts already built from them
    READING, AUDIT, SWAP, REQS = (text.replace("limits include the edge); ", "limits include the edge); " + RULE_READ, 1)
                                  for text in (READING, AUDIT, SWAP, REQS))
    CHECK = CHECK.replace("a range must lie wholly inside); ", "a range must lie wholly inside); " + RULE_READ, 1)
    TRAIL = TRAIL.replace("\"paired\" is \"\" when no such rule is in the records", RULE_TRAIL + "\"paired\" is \"\" when no such rule is in the records", 1)
if PAIR_MUSTS:
    BRIEF = BRIEF.replace("whose value must first be read from another listing the request points at, else [].",
                          "whose value must first be read from another listing the request points at, else []." + PAIR_LOOKUP, 1)
if OPENS_CAP:
    BRIEF = BRIEF.replace("\"lookups\":", OPENS_FIELD + "\"lookups\":", 1)
if NESTED_PAGING:
    SHAPE = SHAPE.replace(" Return ONE JSON object", " A count or next-page key inside a nested object is named by its path, outer key"
                          " first (\"paging.total\")." + " Return ONE JSON object", 1)
# k17m (B1): keys that wrap an answer, and the bookkeeping fields that may stand beside a wrapper (a notice or a note
# beside it keeps the answer whole)
WRAP_KEYS = {"envelope", "payload", "data", "result", "response", "body", "wrapper", "content"}
WRAP_META = re.compile(r"^(?:trace|trace_?id|request_?id|ts|timestamp|seq|version|meta)$", re.I)
# k17m (B3, B6): an answer that a detail or message allowance is used up (not a size limit of one call), and a reply
# that cuts the messages
ALLOWANCE = re.compile(r"allowance|used up|no more (?:listings|details|pages|views)", re.I)
MSG_OUT = re.compile(r"allowance|used up|no more messages|will not read", re.I)
# only with a verb of reading or taking messages ("I can only read one more message"), so "just one more message: ..." or
# "no more messages about colour" in an ordinary reply cut nothing
_READS = r"(?:read|take|answer|get|see|handle|reply to|look at)"
MSG_CUT_RE = re.compile(rf"(?:(?:only|just)\s+{_READS}|{_READS}\s+(?:only|just))\s+(one|two|three|four|five|\d+)\s+more\s+messages?"
                        rf"|{_READS}\s+no\s+more\s+messages?|(?:won't|will not|cannot|can't)\s+{_READS}\s+(?:any\s+)?more\s+messages?", re.I)
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


# k17m (C1, after lllll v1's guess_roles): the words of a tool's name that tell its job, for when no brief named the roles
CART_NAME_WORDS = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}
ROLE_TESTS = [
    ("refuse", lambda w: bool(w & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"})
     or any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w),
    ("cart_view", lambda w: bool(w & (CART_NAME_WORDS | {"items", "order"}))
     and bool(w & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})),
    ("cart", lambda w: bool(w & CART_NAME_WORDS) or (bool(w & {"add", "put", "set", "remove", "rewrite", "update"})
                                                     and bool(w & (CART_NAME_WORDS | {"items", "order", "all"})))),
    ("buy", lambda w: bool(w & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute",
                                "finalize", "submit"})),
    ("ask", lambda w: bool(w & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"})),
    ("open", lambda w: bool(w & {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec",
                                 "specs", "describe", "examine"})),
    ("search", lambda w: bool(w & {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"})),
]
MONEY = re.compile(r"(?:\$\s*(\d+(?:\.\d+)?))|(?:(\d+(?:\.\d+)?)\s*(?:cents|dollars|bucks|usd)\b)", re.I)
REQUEST_CAP = re.compile(r"(?:under|below|at most|no more than|up to|budget|spend|max|cap|limit|tops)\D{0,30}?(?:\$\s*)?"
                         r"(\d+(?:\.\d+)?)\s*(?:cents|dollars|bucks|usd)\b|(?:under|below|at most|no more than|up to|budget|spend|max|cap|"
                         r"limit)\D{0,30}?\$\s*(\d+(?:\.\d+)?)", re.I)
BUY_OPTION = re.compile(r"\b(?:buy|order|purchase|confirm\w*|checkout|yes|place|go)\b", re.I)
END_OPTION = re.compile(r"\b(?:declin\w*|refus\w*|none|nothing|no|cancel\w*|abandon\w*)\b", re.I)
# k17m (C2): a search argument that asks for a later page
PAGE_KEY = re.compile(r"page|offset|cursor|token|start|skip|after", re.I)


def _extends(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    return len(longer) > len(shorter) and longer[: len(shorter)] == shorter


def _word_verdict(words: tuple[str, ...], total: int, old_words: tuple[str, ...], old_total: int, cut: int | None) -> str:
    """k17m (C2, Secretarybird v1's _i_word_verdict): "any" or "all" when one query extends the other (more matches with
    more words: any word matches), "subset" when one word set holds the other and matches fewer, else ""."""
    if _extends(words, old_words):
        return "any" if total > old_total else "all"
    if _extends(old_words, words):
        return "any" if total < old_total else "all"
    new_set, old_set = set(words[:cut]), set(old_words[:cut])
    if (new_set > old_set and total < old_total) or (old_set > new_set and old_total < total):
        return "subset"
    return ""


def _common_lead(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _lever(name: str, fallback: Any = None) -> Any:
    """k17m2: a lever's entry point. Its own code runs while the lever works; an error in it turns the lever off for the
    rest of the episode (a "lever_error" trace entry, see Kyo.lever_fail) and gives the path without it: `fallback`
    (a value, copied when it is a list, set or dict), or what a callable fallback gives for the same arguments. The
    code itself is the wrapper's `impl` (the scripted checks swap it for one that raises)."""
    def wrap(fn: Any) -> Any:
        def run(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not LEVER_GUARD:
                return run.impl(self, *args, **kwargs)
            if name not in self.levers_off:
                try:
                    return run.impl(self, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    self.lever_fail(name, exc, fn.__name__)
            if callable(fallback):
                return fallback(self, *args, **kwargs)
            return type(fallback)(fallback) if isinstance(fallback, (list, set, dict)) else fallback
        run.impl = fn
        run.lever = name
        run.__name__, run.__qualname__, run.__doc__ = fn.__name__, fn.__qualname__, fn.__doc__
        return run
    return wrap


def _at(node: Any, path: str | None) -> Any:
    """k17m (B2): the value at a key or a dotted path one level down ("paging.total")."""
    for part in str(path or "").split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


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


def _name_words(name: Any) -> set[str]:
    """k17m8: a listing name's words (lower case, two characters or more), without bracketed promotion text."""
    text = re.sub(r"\u3010[^\u3011]*\u3011|\[[^\]]*\]|\([^)]*\)", " ", str(name or ""))
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


def _words(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _stem(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _words(text).split())


def _toks(text: Any) -> set[str]:
    """k17: a text's words in lower case, a plural s cut ("Rose-Gold Straps" -> {"rose", "gold", "strap"})."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", _words(str(text))):
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w)
    return out


def _scalars(node: Any, depth: int = 0) -> list[Any]:
    """k17: the keys and scalar values inside a record, nested blocks and lists included (four levels at most)."""
    if depth > 3:
        return []
    if isinstance(node, dict):
        return [k for k in node if isinstance(k, str)] + [v for value in node.values() for v in _scalars(value, depth + 1)]
    if isinstance(node, list):
        return [v for value in node for v in _scalars(value, depth + 1)]
    return [node] if isinstance(node, (str, int, float)) and not isinstance(node, bool) else []


def _vote(first: list[tuple[int, str]], second: list[tuple[int, str]], weight: float, depth: int = 8, *,
          same: Any = None) -> list[tuple[int, str]]:
    """k15: two question plans as one order. Each pick (item number, property) gets depth minus its place in each plan,
    the first plan's points times `weight`; ties keep the earlier place. The same property in other words is one pick,
    worded as the first plan words it; picks past the depth follow in the first plan's order."""
    score: dict[tuple[int, frozenset[str]], float] = {}
    place: dict[tuple[int, frozenset[str]], int] = {}
    label: dict[tuple[int, frozenset[str]], tuple[int, str]] = {}
    for picks, points in ((first, weight), (second, 1.0)):
        keys: list[tuple[int, frozenset[str]]] = []
        for number, text in picks:
            # k17m (V1): with `same`, the store key the label names (see the module notes)
            key = (number, (same(number, text) if same else None) or _stem(text))
            if key not in keys:
                keys.append(key)
                label.setdefault(key, (number, text))
        for pos, key in enumerate(keys[:depth]):
            score[key] = score.get(key, 0.0) + points * (depth - pos)
            place.setdefault(key, pos)
    out = [label[k] for k in sorted(score, key=lambda k: (-score[k], place[k]))]
    return out + [p for p in first if p not in out]


def _kind_noun(name: str) -> str:
    """The head noun of a product's name or store label, singular: "Point & Shoot Cameras" -> "camera"."""
    words = re.findall(r"[a-z]+", str(name).lower())
    if not words:
        return ""
    w = words[-1]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _names(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _roots(text: str) -> set[str]:
    out = set()
    for w in _words(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out


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


_PAIR_NAMES = ("name", "key", "label", "property", "attribute", "field")
_PAIR_VALUES = ("value", "val", "values", "detail", "text")
_MONEY_KEYS = {"amount", "unit", "units", "currency", "cents", "value", "price", "minor_units", "precision"}


def _pairs(value: Any) -> dict[str, Any] | None:
    """k15d: properties given as a list of name/value pairs, as one dict (None for any other list)."""
    if not isinstance(value, list) or len(value) < 2:
        return None
    out: dict[str, Any] = {}
    for entry in value:
        if not isinstance(entry, dict):
            return None
        name = next((entry[k] for k in _PAIR_NAMES if isinstance(entry.get(k), str) and entry[k].strip()), None)
        got = next((entry[k] for k in _PAIR_VALUES if isinstance(entry.get(k), (str, int, float, list)) and not isinstance(entry.get(k), bool)), None)
        if name is None or got is None:
            return None
        out[name] = ", ".join(str(v) for v in got) if isinstance(got, list) else got
    return out if len(out) >= 2 else None


# k17m9 (TEXT_ROWS): a field starts after " | " (or a line break) where a key and ": " follow; a title may hold " | ".
# When the line's first key is lower case, only lower-case keys start fields ("... | Color: Black" stays in a title).
_ROW_SPLIT = re.compile(r"\s+\|\s+(?=[A-Za-z_][\w ]{0,30}?:\s)|\n+(?=[A-Za-z_][\w ]{0,30}?:\s)")
_ROW_SPLIT_LOW = re.compile(r"\s+\|\s+(?=[a-z_][a-z0-9_ ]{0,30}?:\s)|\n+(?=[a-z_][a-z0-9_ ]{0,30}?:\s)")
_ROW_FIELD = re.compile(r"([A-Za-z_][\w ]{0,30}?):[ \t]*(.*)", re.S)
# a "name=value" pair; a value runs to the next "; name=" (a value may hold ";" itself: "features=a;b;c")
_ROW_PAIR = re.compile(r"\s*([^=;|\n]{1,48}?)\s*=(.*?)(?=;[^=;|\n]{1,48}=|$)", re.S)


def _row_value(text: str) -> Any:
    """k17m9 (TEXT_ROWS): a text field's value: None, true / false, a whole or decimal number, a run of "name=value"
    pairs (a dict), else the text itself."""
    text = text.strip()
    if text in ("None", "null"):
        return None
    if text in ("True", "False", "true", "false"):
        return text.lower() == "true"
    if re.fullmatch(r"-?\d{1,15}", text):
        return int(text)
    if re.fullmatch(r"-?\d{1,12}\.\d{1,6}", text):
        return float(text)
    pairs = _ROW_PAIR.findall(text)
    if len(pairs) >= 2 and text.count("=") == len(pairs):
        return {k.strip(): v.strip() for k, v in pairs if k.strip()}
    return text


def _text_row(text: Any) -> dict[str, Any] | None:
    """k17m9 (TEXT_ROWS): a listing written as one text line of "key: value" fields joined by " | " (or one field a
    line) as a dict; None for other text. Four fields at least, no key twice, and one value shaped like a listing id
    (letters and digits, no spaces)."""
    if not isinstance(text, str) or not 24 <= len(text) <= 20000:
        return None
    head = _ROW_FIELD.match(text.strip())
    if not head:
        return None
    parts = (_ROW_SPLIT_LOW if head.group(1) == head.group(1).lower() else _ROW_SPLIT).split(text.strip())
    if len(parts) < 4:
        return None
    record: dict[str, Any] = {}
    for part in parts:
        found = _ROW_FIELD.fullmatch(part.strip())
        if not found or found.group(1).strip() in record:
            return None
        record[found.group(1).strip()] = _row_value(found.group(2))
    if not any(isinstance(v, str) and _id_like(v) and re.search(r"[A-Za-z]", v) for v in record.values()):
        return None
    return record


def _text_rows(node: Any, depth: int = 0) -> Any:
    """k17m9 (TEXT_ROWS): a store result with each listing text line in it read as a record; the rest unchanged."""
    if depth > 6:
        return node
    if isinstance(node, str):
        row = _text_row(node)
        if row is None and node[:1] in "{[" and node.rstrip()[-1:] in "}]":
            # a record (or a list of them) sent as a JSON string is read as what it encodes
            try:
                inner = json.loads(node)
            except ValueError:
                inner = None
            if isinstance(inner, (dict, list)) and inner:
                return _text_rows(inner, depth + 1)
        return node if row is None else row
    if isinstance(node, list):
        return [_text_rows(v, depth + 1) for v in node]
    if isinstance(node, dict):
        return {k: _text_rows(v, depth + 1) for k, v in node.items()}
    return node


_ROW_LIKE = re.compile(r"(?:^|\s\|\s|\n|;\s|,\s)[A-Za-z_][\w ]{0,30}?:\s")


def _clip_room(value: Any, limit: int, depth: int = 0, wide: bool = False) -> Any:
    """k17m9 (TEXT_ROWS): _clip, except that a long text standing as an item of its own (a list element or a wrapper's
    value, not a field of a listing record) holding four "key: value" fields or more (a listing in a shape no reader here
    knows) keeps TEXT_ROOM characters: a 450-character cut hid most of race 15's properties. Fields of records, such as a
    description that lists "Material: ..; Colour: ..", keep the usual cut."""
    if isinstance(value, str) and len(value) > limit and len(_ROW_LIKE.findall(value)) >= 4:
        return value if len(value) <= TEXT_ROOM else value[:TEXT_ROOM] + "..."
    if wide and isinstance(value, str) and len(value) > limit and _ID_TOKEN.search(value):
        # k17m10 (HEALTH): in a blind store, a text holding listing ids keeps room (several ids: several listings)
        room = TEXT_ROOM if len(_ID_TOKEN.findall(value)) < 3 else 4 * TEXT_ROOM
        return value if len(value) <= room else value[:room] + "..."
    if depth > 7 or not isinstance(value, (dict, list)):
        return _clip(value, limit, depth)
    if isinstance(value, dict):
        if len(value) >= 4 and any(isinstance(v, str) and _id_like(v) and re.search(r"[A-Za-z]", v) for v in value.values()):
            return _clip(value, limit, depth)      # a listing record: its fields keep the usual cut
        return {k: _clip_room(v, limit, depth + 1, wide) for k, v in value.items()}
    return [_clip_room(v, limit, depth + 1, wide) for v in value]


# k17m10 (HEALTH): a listing-like id inside text ("p-0a1b2c3d", "SKU-12AB34"); the fields a record's attributes are not;
# the argument names a cart call puts a listing id in
_ID_TOKEN = re.compile(r"\b[A-Za-z]{1,4}[-_](?=\w*\d)[0-9A-Za-z]{1,16}\b")
_CORE_FIELD = re.compile(r"(?:^|_)(?:[a-z]?ids?|ref|sku|code|title|name|price|cost|amount|currency|stock|avail\w*|on|hand|in|"
                         r"qty|quantity|count|total|category|shelf|group|department|aisle|shop|seller|merchant|store|vendor|"
                         r"company|sold|by|desc\w*|about|blurb|remarks?|comments?|notes?|url|link|image|rating|reviews?|"
                         r"page|next|score|rank)(?:$|_)", re.I)
_ID_ARG = re.compile(r"(?:^|_)(?:id|ids|sku|ref|listing|product|item|code|number|offer|article)(?:$|_)", re.I)


def _flat_facts(node: dict[str, Any]) -> dict[str, Any] | None:
    """k17m10 (HEALTH): a listing record with no nested attribute block but three or more short plain fields that are not
    its id, name, price, stock, category, shop or texts: those fields are its attributes (a store that sends them flat)."""
    if not any(isinstance(v, str) and _id_like(v) and re.search(r"[A-Za-z]", v) for v in node.values()):
        return None
    facts = {k: v for k, v in node.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
             and not _CORE_FIELD.search(_words(str(k)).replace(" ", "_")) and len(str(v)) <= 80}
    return facts if len(facts) >= 3 else None


def _arg_id(args: dict[str, Any]) -> str | None:
    """k17m10 (HEALTH): the listing id a cart call (or one of its line objects) names: a string argument whose name says
    id / listing / product / item, else the only id-like string value."""
    named = [v.strip() for k, v in args.items() if isinstance(v, str) and v.strip()
             and _ID_ARG.search(_words(str(k)).replace(" ", "_"))]
    if named:
        return named[0]
    plain = [v.strip() for v in args.values() if isinstance(v, str) and _id_like(v) and re.search(r"[A-Za-z]", v)]
    return plain[0] if len(plain) == 1 else None


def _listing_text(text: str) -> bool:
    """k17m10 (HEALTH): text that reads as a listing rather than a store note: long, or past a short line and holding a
    listing-like id."""
    return len(text) > 300 or (len(text) > 120 and bool(_ID_TOKEN.search(text)))


def _fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, list):
            # k15d: name/value pairs (see the module notes)
            value = _pairs(value)
        if isinstance(value, dict) and {str(k).lower() for k in value} <= _MONEY_KEYS:
            # k15d: a money object is no attribute block
            continue
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    if best is None and HEALTH:
        # k17m10 (HEALTH): attributes sent as the record's own plain fields
        best = _flat_facts(node)
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


class Kyo:
    def __init__(self, problem: dict[str, Any]):
        self.levers_off: set[str] = set()                   # k17m2: levers that raised in this episode
        self.iv_sent: list[tuple[int, str, str]] = []       # k17m2: the questions the interview has sent
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
        self.held_msg = 0                                   # k8: a message held back until the first cart answer
        self.later: list[tuple[int, str, str]] = []         # k8: the interview's remaining picks, for the held message
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
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        # k16: whether a single-listing tool's result showed a stock key, the step a record last showed a listing's
        # stock, and the step of a sell-out remark that named no listing (-1: none pending)
        self.stock_from: dict[str, bool] = {}
        self.stock_seen: dict[str, int] = {}
        self.doubt_at = -1
        self.note_gone: list[str] = []
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
        self.sought = False         # k2: the code search for items with no lead before an ending without a purchase
        self.answered_by = ""       # k2: the model whose reply ask_json used last
        self.recheckouts = 0                                # k12
        self.change_now: list[str] = []  # k2: what the last note changed (budget, requirements)
        self.searched: list[dict[str, Any]] = []  # k14: the loop's search calls that returned listings (last 8)
        self.rethought = 0          # k2: reviews after a refused ending without a purchase
        self.swap_ended = False     # k2: the review before an ending without a purchase after a line sold out
        self.next_page: tuple[str, Any] | None = None  # k2: (parameter, value) of the last code search's next page
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.follow_ups: set[str] = set()                   # k15c: tools that place a held checkout
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
        self.ref_paired: dict[str, str] = {}   # k2: the value a store pairing rule pairs with the pointed page's value
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
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.t0 = time.monotonic()
        self.call_cap = ALL_CALLS
        # k17
        self.call_fields: dict[str, str] = {}               # R1: field -> "action" or "top"
        self.last_fail = ""                                 # R1/E2: the body of the last store call that failed
        self.refixed = False                                # E2: a turned-down ending was fixed once
        self.cover_done = False                             # C1: the checkout coverage hold, once
        self.req_marks: set[str] = set()                    # C2: carts given the strict review
        self.budget_moved = ""                              # E3: "up" or "down" after a note moved the budget
        self.menus: list[dict[str, Any]] = []               # Q1: each menu and how it was answered
        self.hard: dict[int, set[str]] = {}                 # Q1: per item, properties hard for a shopper (MENU_HARD)
        self.key_n: dict[tuple[int, frozenset[str]], int] = {}       # k17j: listings carrying an item's key
        self.key_name: dict[tuple[int, frozenset[str]], str] = {}    # k17j: the key's longest spelling
        self.empty_tried = False                            # k17j: EMPTY_GUARD, once (re-armed by a change)
        self.full_held = False                              # k17j: FULL_GUARD, once (re-armed by a change)
        self.swaps: dict[int, int] = {}                     # k17j: an item's line changes made while the line was fine
        self.swap_held: set[int] = set()                    # k17j: items whose second change was held once
        self.count_key: str | None = None                   # k17j: the line count's key in the cart tool's reply (love)
        self.cart_shows_lines = False                       # k17j: the cart tool's replies list the cart's lines (love)
        self.asked_props: set[tuple[int, str]] = set()      # k17j: (item, property) every interview question named
        self.opens_cap: int | None = None                   # k17m (B3): listing details the request allows in all
        self.opens_used = 0                                 # k17m (B3): listing details asked for so far
        self.opens_out = False                              # k17m (B3): the store said the allowance is used up
        self.msg_cut = False                                # k17m (B6): a reply cut the message allowance
        self.join_failed = False                            # k17m (A5): the last side-by-side call showed nothing
        self.brief_failed = False                           # k17m (C1): no model read the request
        self.hear_failed = False                            # k17m (C1): the last answer reading got no reply
        self.unheard: list[tuple[int, str, str]] = []       # k17m (C1): answers no model could read yet
        self.word_seen: list[tuple[Any, ...]] = []          # k17m (C2): (text key and mode, other arguments, words, total, rows)
        self.word_kind = ""                                 # k17m (C2): "all", "any" or "" (not known)
        self.lead_bounds: dict[Any, list[Any]] = {}         # k17m (C2): per text key and mode, words read at least / maybe
        self.lead_words = 0                                 # k17m (C2): the search reads only this many first words (0: not known)
        self.hit_words: dict[str, set[str]] = {}            # k17m (C2): words an every-word search found in a row's record
        self.thin: set[int] = set()                         # k17m (C3): items with no opened listing before the questions
        self.stuck = 0                                      # k17m (C4): loop turns in a row that used no store step
        self.price_moved_now = False                        # k17m (C5): the last note said only that a price moved
        self.swap_named: list[str] = []                     # k17m3: replacements the sold-out reviews named, newest first
        self.sold_bought = False                            # k17m3: the takeover after a sell-out has run
        self.marks_seen: dict[str, dict[str, str]] = {}     # k17m3: the strict reviews' latest marks per listing
        self.sold_reviewed: set[str] = set()                # k17m4: carts given the strict review after a sell-out
        self.spend_hard = SPEND_HARD                        # k17m5: this episode's hard spend line
        self.costly_read = False                            # k17m5: the world's cost has been read
        self.told_mark = ""                                 # k17m6: the cart the last cart-and-total message named
        self.reserve_noted = False                          # k17m7: the swap reserve was noted in the trace
        self.twin_sought: set[str] = set()                  # k17m8: sold-out lines whose name has been searched
        self.blind = False                                  # k17m10: no listing record read after BLIND_AFTER reading answers
        self.blind_told = False                             # k17m10: the loop model was told to read the raw results
        self.blind_held = False                             # k17m10: an ending without a purchase held once while blind
        self.reader_answers = 0                             # k17m10: reading tools' answers with text and no error
        self.own_cart: dict[str, int] = {}                  # k17m10: the cart by Kyo's own cart calls, any id

    # ---------- plumbing ----------
    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        """Seconds since this episode's agent started (time spent inside the episode only)."""
        return time.monotonic() - self.t0

    def msgs_free(self) -> int:
        """Messages the interviewer may send: the ones the request's steps at the order or the ending take stay back."""
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs) - self.held_msg

    def cart_mark(self) -> str:
        # k17m10 (HEALTH): with no listing read, the cart changes only in Kyo's own record of its calls
        own = _dump(sorted(self.own_cart.items())) if HEALTH and not self.rows else ""
        return _dump(sorted(self.cart_lines.items())) + own

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
        # steps due at this stage that the gate has not passed for the cart as it is now
        return any(p["when"] == stage for p in self.process) and (stage, self.cart_mark() + f"#{self.version}") not in self.gate_ok

    def tail(self) -> int:
        """Steps kept for the finish: the checkout, one more while a step due at the order needs a message, and (k15c)
        one more where the store holds a checkout until another call places it; (k17m7) on a cheapest pick with a full
        cart and no sell-out yet, the swap's two cart calls and the checkout again (see the module notes)."""
        talk = any(p["when"] == "order" and p["messages"] > 0 for p in self.process)
        kept = (2 if talk and self.msg_used < self.msg_limit and self.gate_due("order") else 1) + (1 if self.follow_ups else 0)
        if SWAP_RESERVE and self.swap_room():
            kept += SWAP_RESERVE + 1
        return kept

    @_lever("swap_reserve", False)
    def swap_room(self) -> bool:
        """k17m7 (SWAP_RESERVE): a cheapest pick whose cart holds a line for every requested item, before any line has
        sold out (a sell-out gives the reserve back for the swap)."""
        if self.gone or self.brief.get("pick") != "cheapest" or not self.items:
            return False
        lines = self.cart_ids()
        if not lines:
            return False
        held = {self.line_item(i) for i in lines}
        if not all(n in held for n in range(1, len(self.items) + 1)):
            return False
        if not self.reserve_noted:
            self.reserve_noted = True
            self.trace.append({"kind": "swap_reserve", "left": self.left(), "cart": list(lines)})
        return True

    def note_follow_up(self, seen: Any) -> None:
        """k15c: a checkout answer that names another tool which confirms, places or completes the order (a held
        checkout): that tool is part of the ending from now on (see the module notes)."""
        text = seen if isinstance(seen, str) else _dump(seen)
        ends = {self.roles.get("buy"), self.roles.get("refuse")}
        for tool in self.tools:
            fn = tool.get("function") or {}
            name = str(fn.get("name") or "")
            if not name or name in ends or name in self.follow_ups or name not in text:
                continue
            if re.search(r"confirm|place|finali[sz]|complet|commit|submit", name + " " + str(fn.get("description") or ""), re.I):
                self.follow_ups.add(name)
                self.trace.append({"kind": "follow_up", "tool": name})

    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if CODE_END and self.no_credit:
            # k17m (C1): a key out of credit stays so for the episode: no call is sent again
            return None
        for model in models:
            if self.calls >= self.call_cap or self.elapsed() > CALL_SECONDS:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.call_model(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors pass: wait and send the same call again
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                # a timeout already took the client's 50 s: it is sent once more at most, and not at all late in the episode
                if (not passing or tries >= (1 if status in (None, 408) else len(WAITS)) or self.waited + WAITS[tries] > WAIT_TOTAL
                        or self.elapsed() > LOOP_SECONDS):
                    break
                pause = WAITS[tries] * (0.8 + 0.4 * random.random())
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
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if model.startswith("google/"):
            # k2: these models think before answering and spend a short allowance on it; a short think is asked for,
            # and the allowance leaves room for the answer if the setting does not reach the provider
            body["reasoning"] = {"effort": "low"}
            body["max_tokens"] = max(max_tokens, 6000)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        client = _llm
        if TIME_BOUND:
            # k17m11 (TIME_BOUND): see the module notes
            room = TASK_LIMIT - CALL_MARGIN - self.elapsed()
            if room < 5:
                self.trace.append({"kind": "llm_late_skip", "model": model, "at": round(self.elapsed(), 1)})
                return None, 408
            client = _llm_for(room)
        result = client.post_verbose("/inference/chat/completions", body)
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
                self.answered_by = model
                return got
            if message is not None:
                # k2: a reply that holds no JSON object (cut short, or prose) is recorded, then the next model is asked
                self.trace.append({"kind": "llm_fail", "model": model, "status": "no json",
                                   "body": str((message or {}).get("content") or "")[:160]})
            if self.broke or self.no_credit or self.calls >= self.call_cap:
                return None
        return None

    def judge(self, prompt: str, max_tokens: int, kind: str) -> dict[str, Any] | None:
        """k5: a review on the heavy model while the episode has spent less than LEAN_LINE; past it, or when the heavy
        model gives no answer, the light model decides with room to think. Late in the episode the heavy model decides
        either way: on a review the light one takes about 15 s, the heavy one about 8."""
        late = self.elapsed() > LIGHT_SECONDS
        if self.cost < LEAN_LINE or late:
            got = self.ask_json(prompt, [HEAVY], max_tokens)
            if got is not None or late or self.broke or self.no_credit or self.calls >= self.call_cap:
                return got
        else:
            self.trace.append({"kind": "lean", "judge": kind, "cost": round(self.cost, 4)})
        return self.ask_json(prompt, [LIGHT], LIGHT_ROOM)

    def check_judge(self, prompt: str) -> dict[str, Any] | None:
        """k14: the checkout check is read by the light model with room to think (on race-13 records it failed the near
        twins the heavy model passed); late in the episode the heavy model answers, being faster."""
        late = self.elapsed() > LIGHT_SECONDS
        got = self.ask_json(prompt, [HEAVY] if late else [LIGHT], 1500 if late else LIGHT_ROOM)
        if got is None and not late and not (self.broke or self.no_credit or self.calls >= self.call_cap):
            got = self.ask_json(prompt, [HEAVY], 1500)
        return got

    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        self.last_fail = ""
        if OPENS_CAP and "opens_cap" not in self.levers_off and name in self.detail_tools():
            # k17m (B3): every listing a detail view is asked for counts against a cap the request sets
            self.opens_used += max(1, sum(len(v) for v in args.values() if isinstance(v, list)))
        self.add_fields(payload)
        for _ in range(4):
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                # k17 (R1): a result wrapped in a list of calls is read as the plain one (see the module notes)
                data = result.data
                if CALL_FIELDS and "call_fields" not in self.levers_off:
                    try:
                        data = self.unwrap(result.data)
                    except Exception as exc:  # noqa: BLE001
                        self.lever_fail("call_fields", exc, "unwrap")
                if PEEL and "peel" not in self.levers_off and isinstance(data.get("observation"), dict):
                    try:
                        # k17m (B1): an answer in an envelope is read as what it holds (see the module notes)
                        data = {**data, "observation": self.peel(data["observation"])}
                    except Exception as exc:  # noqa: BLE001
                        self.lever_fail("peel", exc, "peel")
                if TEXT_ROWS and "text_rows" not in self.levers_off and data.get("observation") is not None:
                    try:
                        # k17m9 (TEXT_ROWS): listings sent as text lines are read as records (see the module notes)
                        data = {**data, "observation": _text_rows(data["observation"])}
                    except Exception as exc:  # noqa: BLE001
                        self.lever_fail("text_rows", exc, "act")
                if OPENS_CAP and "opens_cap" not in self.levers_off and name in self.detail_tools():
                    try:
                        seen_now = data.get("observation")
                        if isinstance(seen_now, dict) and seen_now.get("error") and ALLOWANCE.search(str(seen_now.get("error"))):
                            # k17m (B3): the store's detail allowance is used up: Kyo opens nothing more
                            self.opens_out = True
                            self.trace.append({"kind": "opens_out", "used": self.opens_used, "cap": self.opens_cap})
                    except Exception as exc:  # noqa: BLE001
                        self.lever_fail("opens_cap", exc, "act")
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                seen = data.get("observation")
                if (isinstance(seen, dict) and not seen.get("error") and self.single_target(name) and self.ident(seen)
                        and name not in {self.roles.get(r) for r in ("cart", "buy", "refuse", "ask")}):
                    # k16: whether this single-listing view shows stock (see the module notes)
                    self.stock_from[name] = self.stock_from.get(name, False) or self.shows_stock(seen)
                if name in self.page_tools() and not (isinstance(seen, dict) and seen.get("error")):
                    # listings whose full page the agent has opened (a search row is not that)
                    for value in args.values():
                        self.opened.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                if name == self.roles.get("buy") and not self.finished:
                    self.note_follow_up(seen)
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_fail", "status": status, "body": body[:200]})
            self.last_fail = body[:600]
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
            if status == 422 and CALL_FIELDS and self.learn_fields(body):
                # k17 (R1): the store wants a field on every call (see the module notes)
                self.add_fields(payload)
                continue
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    @staticmethod
    def peel(node: Any, depth: int = 0) -> Any:
        """k17m (B1): {"envelope": {"payload": {...}, "trace": ...}} read as the payload: a dict whose one wrapper key
        (envelope, payload, data, result, ...) holds a dict, beside nothing but bookkeeping fields and no error, is
        replaced by what that key holds, a few levels deep."""
        while depth < 4 and isinstance(node, dict):
            inner = [k for k, v in node.items() if isinstance(v, dict) and str(k).lower() in WRAP_KEYS]
            rest = [k for k in node if k not in inner]
            if len(inner) != 1 or node.get("error") or any(not WRAP_META.match(str(k)) and not (k == "error" and not node[k]) for k in rest):
                break
            node = node[inner[0]]
            depth += 1
        return node

    @_lever('opens_cap', set())
    def detail_tools(self) -> set[str]:
        """k17m (B3): the tools that open listings' full details: the attributes view, the offer view, the side by side."""
        return {t for t in (self.roles.get("open"), self.roles.get("offer"), self.roles.get("compare")) if t}

    @_lever('opens_cap', True)
    def may_open(self, count: int = 1, keep: int = OPENS_KEEP) -> bool:
        """k17m (B3): whether Kyo may open `count` more listings itself, leaving `keep` of a stated cap for the end."""
        if not OPENS_CAP:
            return True
        if self.opens_out:
            return False
        return not self.opens_cap or self.opens_used + count <= self.opens_cap - keep

    @_lever('opens_cap', lambda self, want: want)
    def pre_room(self, want: int) -> int:
        """k17m (B3): how many listings Kyo reads before the questions: a third of a stated cap at most, since the
        opens after the answers are the ones aimed at the wanted values."""
        if not OPENS_CAP or not (self.opens_cap or self.opens_out):
            return want
        if self.opens_out:
            return 0
        return max(0, min(want, self.opens_cap // 3 - self.opens_used))

    @_lever('msg_cut', None)
    def read_cut(self, answer: Any, words: str) -> None:
        """k17m (B6): a reply saying how many more messages the shopper will read, or a message the store turned down as
        past the allowance, lowers the allowance to that."""
        error = str(answer.get("error") or "") if isinstance(answer, dict) else ""
        cut = None
        if error and MSG_OUT.search(error):
            cut = self.msg_used
        else:
            found = MSG_CUT_RE.search(words)
            if found:
                said = (found.group(1) or "0").lower()
                cut = self.msg_used + (NUM_WORDS.get(said) or (_count(said) or 0))
        if cut is not None and cut < self.msg_limit:
            self.msg_limit = cut
            self.msg_cut = True
            self.trace.append({"kind": "msg_cut", "limit": self.msg_limit, "used": self.msg_used})

    @_lever('call_fields', None)
    def add_fields(self, payload: dict[str, Any]) -> None:
        """k17 (R1): the fields the store asked every call to carry: a turn number, or a fresh id for anything else."""
        tag = f"k{payload.get('step')}-{random.randrange(16 ** 10):010x}"
        for field, where in self.call_fields.items():
            value = payload.get("step") if re.search(r"turn", field, re.I) else tag
            (payload["action"] if where == "action" else payload)[field] = value

    @_lever('call_fields', False)
    def learn_fields(self, body: str) -> bool:
        """k17 (R1): a 422 that says a call id, an idempotency key or a turn is required or missing; True when it named
        a field not added yet."""
        if not re.search(r"requir|missing|must|expected", body, re.I):
            return False
        found = False
        for m in re.finditer(r"\b(call_?id|callId|idempotency_?key|idempotencyKey|idempotency|turn_?id|turnId|turn)\b", body, re.I):
            field = m.group(1)
            if field in self.call_fields:
                continue
            self.call_fields[field] = "action" if re.search(r"action", body[max(0, m.start() - 40):m.start()], re.I) else "top"
            found = True
        if found:
            self.trace.append({"kind": "call_fields", "fields": dict(self.call_fields)})
        return found

    @staticmethod
    def unwrap(data: dict[str, Any]) -> dict[str, Any]:
        """k17 (R1): {"calls": [{"observation": {"observation": ..., "done": ..., "error": ...}}]} read as the plain
        {"observation", "finished", "step"} result; any other result as it is."""
        wrapped = data.get("calls")
        if "observation" in data or not isinstance(wrapped, list) or not wrapped or not isinstance(wrapped[0], dict):
            return data
        head = wrapped[0]
        inner = head.get("observation")
        if not isinstance(inner, dict) or "observation" not in inner:
            return data
        seen = inner.get("observation")
        if inner.get("error"):
            seen = {**seen, "error": seen.get("error") or inner["error"]} if isinstance(seen, dict) else {"error": inner["error"], "detail": seen}
        out = {k: v for k, v in data.items() if k != "calls"}
        out["observation"] = seen
        out["finished"] = bool(data.get("finished") or inner.get("done") or head.get("done"))
        if not isinstance(out.get("step"), int) and isinstance(head.get("step"), int):
            out["step"] = head["step"]
        return out

    # ---------- reading the request ----------
    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {_dump(fn.get('parameters') or {})}")
        prompt = BRIEF + "\nTOOLS:\n" + "\n".join(lines) + "\n\nREQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + _dump(_clip(self.view.get("opening"), 1500))
        # the light model reads the request; the heavy one only when that reading names no item or no tool for a job
        # room for the light model's thinking: a reply cut at its limit sent a tenth of race 6's briefs to the heavy one
        brief = self.ask_json(prompt, [LIGHT], 4000) or {}
        if not self.no_credit and (not _items(brief.get("items")) or not isinstance(brief.get("roles"), dict) or not brief["roles"].get("buy")):
            brief = self.ask_json(prompt, [HEAVY, LIGHT], 2500) or brief
        self.brief = brief
        roles = brief.get("roles") if isinstance(brief.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        try:
            if CODE_END:
                # k17m (C1, after lllll v1): no brief, or one without the cart or the checkout: the tools' names tell the jobs
                # (asking the shopper only when no brief came at all)
                self.brief_failed = not brief
                if not brief or not self.roles.get("buy") or not self.roles.get("cart"):
                    guess = self.guess_roles()
                    for job, tool in guess.items():
                        if job not in self.roles and tool not in self.roles.values() and (job != "ask" or not brief):
                            self.roles[job] = tool
                    self.trace.append({"kind": "roles_guessed", "roles": dict(self.roles), "brief": bool(brief)})
                if not brief:
                    # k17m (C1): no brief: a money figure in the request after a word of limit, else the only one there
                    figure = REQUEST_CAP.search(self.request)
                    money = MONEY.findall(self.request)
                    value = (figure.group(1) or figure.group(2)) if figure else (next(a or b for a, b in money) if len(money) == 1 else None)
                    if value:
                        brief["budget"] = float(value)
            if HEALTH and not self.roles.get("ask"):
                # k17m10 (HEALTH): the brief named no messaging tool: the tools' names may (no question can go out without it)
                asker = self.guess_roles().get("ask")
                if asker and asker not in self.roles.values():
                    self.roles["ask"] = asker
                    self.trace.append({"kind": "ask_role_guessed", "tool": asker})
        except Exception as exc:  # noqa: BLE001
            # k17m2: the role guess and the budget figure are left out (the brief's own roles and budget stand)
            self.lever_fail("code_end", exc, "read_request")
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
        if END_SURE and not self.roles.get("refuse") and not self.no_credit:
            # k17 (E2): see the module notes
            self.find_refuse()
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
        if OPENS_CAP:
            # k17m (B3): a cap the request sets on listing details (see the module notes)
            cap = _count(brief.get("opens"))
            self.opens_cap = cap if cap and cap > 0 else None
        steps = _count(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        self.add_steps(brief.get("process"))

    def add_steps(self, entries: Any) -> list[str]:
        """Take in procedures (from the request, a tool's description, an answer or a note); returns the new ones."""
        new = []
        for entry in _items(entries):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                step = str(entry["step"]).strip()
                if any(_words(step) == _words(p["step"]) for p in self.process):
                    continue
                when = str(entry.get("when") or "").strip().lower()
                self.process.append({"step": step, "when": when if when in ("choose", "order", "end") else "order",
                                     "messages": max(0, _count(entry.get("messages")) or 0)})
                new.append(step)
        if self.roles.get("ask"):
            # the messages the steps at the order and the ending take, one at least for each of the two that needs any;
            # one more for a step before the order that takes a message, while a question still fits: a refused checkout
            # changes the cart, and the step is then due again for the new cart
            # k13: only the steps at the order hold messages back (see the module notes)
            talk = [p["messages"] for p in self.process if p["when"] == "order" and p["messages"]]
            keep = max(talk) if talk else 0
            if any(p["when"] == "order" and p["messages"] for p in self.process) and self.msg_limit - keep - 1 >= 1:
                keep += 1
            # k9: one message for asking before a line comes out of the cart
            if self.removal_rule() and self.msg_limit - keep - 1 >= 1:
                keep += 1
            self.reserve = min(keep, self.msg_limit)
            if self.reserve:
                self.held_msg = 0
        return new

    @_lever('ending', None)
    def find_refuse(self) -> None:
        """k17 (E2): the brief named no tool for ending without a purchase: the light model names it once. Not a tool
        the brief gave another job, nor the checkout tool unless its fixed arguments tell the two endings apart."""
        lines = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_dump(t['function'].get('parameters') or {})}"
                 for t in self.tools]
        got = self.ask_json(ENDING + "\n\nTOOLS:\n" + "\n".join(lines), [LIGHT], 600) or {}
        tool = got.get("tool")
        args = got.get("args") if isinstance(got.get("args"), dict) else {}
        busy = {self.roles.get(r) for r in ("search", "open", "offer", "compare", "ask", "cart", "cart_view")} - {None}
        ok = isinstance(tool, str) and tool in self.schemas and tool not in busy
        if ok and tool == self.roles.get("buy"):
            ok = bool(args) and bool(self.brief.get("buy_args")) and args != self.brief.get("buy_args")
        if ok:
            self.roles["refuse"] = tool
            self.brief["refuse_args"] = args
        self.trace.append({"kind": "ending_found", "tool": tool if ok else None, "said": _clip(got, 80)})

    @_lever('ending', None)
    def refix(self, tool: str, args: dict[str, Any], why: str) -> dict[str, Any] | None:
        """k17 (E2): arguments for an ending without a purchase that the schema check or the store turned down, fixed once
        by the light model from the reason and the tool's schema; None when no valid arguments came back."""
        if self.refixed or self.no_credit or self.broke:
            return None
        self.refixed = True
        prompt = (REFIX + f"\n\nTOOL: {tool} | parameters: {_dump(self.schemas.get(tool) or {})}\nARGUMENTS SENT: {_dump(args)}"
                  + f"\nWHY IT WAS TURNED DOWN: {why[:800]}")
        got = self.ask_json(prompt, [LIGHT, HEAVY], 800) or {}
        fixed = got.get("args") if isinstance(got.get("args"), dict) else None
        faults: list[str] = []
        if fixed is not None:
            fixed = _shape(self.quiet(tool, fixed), self.schemas.get(tool), "args", faults)
        self.trace.append({"kind": "ending_fixed", "args": _clip(fixed, 80), "faults": faults[:3], "why": why[:160]})
        return None if fixed is None or faults else fixed

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
            needs = [m + (f" (the page it points at gives: {self.ref_values[m]}" + (f"; the store's pairing rule pairs that with: {self.ref_paired[m]}" if m in self.ref_paired else "") + ")" if m in self.ref_values else "")
                     for m in item.musts + item.told if m not in item.dropped]
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
            when = {"choose": "before choosing", "order": "at the order (Kyo does it when you check out)", "end": "at an ending without a purchase (Kyo does it then)"}[entry["when"]]
            if entry is self.removal_rule():
                # k9
                when = "before a line comes out of the cart (Kyo asks the shopper when you send such a cart call; so put a listing in the cart only once it is the one you will buy)"
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
        if OPENS_CAP and self.opens_out:
            text += " The store allows no more listing details: work with the records you have."
        elif OPENS_CAP and self.opens_cap:
            text += f" Listing details opened: {min(self.opens_used, self.opens_cap)} of the {self.opens_cap} the request allows."
        if left <= self.tail() + 1:
            text += " FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that)."
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += " Few steps left: set the cart to the chosen lines and check out now."
        elif self.buys == 0 and left <= 12:
            text += " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        elif self.buys == 0 and not self.cart_touched and self.step >= self.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        live = {need for _, need in self.needs_now()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += "\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] " + " | ".join(lookups[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        if self.notes:
            text += "\n[changes so far] " + " | ".join(self.notes[-4:])
        text += "\n[ledger]\n" + self.ledger()
        return text

    # ---------- listings ----------
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
                    # a flag of the listing's own record turning from true to false is read as "can no longer be bought"
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()) or any(
                            isinstance(old.get(k), str) and isinstance(v, str) and len(v) <= 40 and SOLD_WORD.search(v)
                            and not SOLD_WORD.search(old[k]) for k, v in node.items()):
                        # k15e: a short stock word turning into "sold out" counts too
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    if self.shows_stock(node):
                        self.stock_seen[ident] = self.step
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
            if CATEGORY_FIX and (key == self.stock_key or STOCK_FIELD.search(str(key)) or all(STOCK_WORDS.search(v) for v in values)):
                # k17 (R2): stock written as words is no category (see the module notes)
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
            if KEY_MERGE:
                # k17j: one property spelled several ways is one key (love's merge_keys; see the module notes)
                group["keys"] = self.merge_keys(group["keys"])
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: -kv[1]["n"]):
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

    def price_of(self, ident: str) -> float | None:
        """The listing's price as its newest record shows it, under the price key the model named for result rows."""
        value = (self.rows.get(ident) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str):
            # k15d: a price written with its unit ("145 cts", "$12.50") counts as its number
            got = re.fullmatch(r"\s*[$€£]?\s*(\d+(?:\.\d+)?)\s*(?:[A-Za-z][A-Za-z.\-]{0,11})?\s*", value)
            value = float(got.group(1)) if got else value
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def reader_like(self, tool: str) -> bool:
        """k16b: a tool the brief named for no job may stand in as a listing view only when its one parameter is named
        like an id and its name and description say nothing of ending, ordering, the cart or messages."""
        about = tool + " " + str(next((t.get("function", {}).get("description") for t in self.tools
                                       if isinstance(t, dict) and t.get("function", {}).get("name") == tool), "") or "")
        return bool(ID_PARAM.search(self.single_target(tool) or "")) and not NOT_READER.search(about)

    def shows_stock(self, node: Any) -> bool:
        """k16: a record carries the stock key the model named for result rows, or a key about stock or availability."""
        if not isinstance(node, dict):
            return False
        return any(k == self.stock_key or re.search(r"stock|avail", str(k), re.I) for k in node)

    def buyable(self, ident: str) -> bool:
        """Not sold out by any record: no note or refusal named it, and its stock flag (the key the model named) is not no."""
        if self.sold_out(ident):
            return False
        record = self.rows.get(ident) or {}
        value = record.get(self.stock_key) if self.stock_key else None
        # k15e: stock written as a word (see the module notes)
        if any(isinstance(v, str) and len(v) <= 40 and SOLD_WORD.search(v) for k, v in record.items()
               if k == self.stock_key or (not self.stock_key and re.search(r"stock|avail", str(k), re.I))):
            return False
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
        got = self.ask_json(prompt, [LIGHT, HEAVY], 1500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None

        def deep(key: Any) -> str | None:
            # k17m (B2): a paging field one level down ("paging": {"total": ...}) is named by its path
            found = named(key, observation)
            if found or not NESTED_PAGING or not isinstance(key, str) or "nested_paging" in self.levers_off:
                return found
            try:
                tail = key.split(".")[-1]
                return next((f"{outer}.{tail}" for outer, inner in observation.items() if isinstance(inner, dict) and tail in inner), None)
            except Exception as exc:  # noqa: BLE001
                self.lever_fail("nested_paging", exc, "read_shape")
                return found

        shape = {"rows": rows, "count": deep(got.get("count")), "next": deep(got.get("next")),
                 "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if shape["count"] and (isinstance(_at(observation, shape["count"]), bool) or not isinstance(_at(observation, shape["count"]), int)):
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
        rows, count, value = observation.get(shape["rows"]), _at(observation, shape["count"]), _at(observation, shape["next"])
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
            shown_text = self.show(seen)
            self.read_cache[name + _dump(call)] = (self.version, self.step, shown_text)
            texts.append(f"{param}={_dump(value)}: {shown_text}")
            after = self.paging(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({"kind": "pages", "tool": name, "read": len(texts)})
        return ("\n[Kyo read the rest of this search itself, one step a page]\n" + "\n".join(texts)) if texts else ""

    def cheapest_here(self, observations: list[Any]) -> str:
        """Under a cheapest pick: the buyable rows of a search result (and the pages Kyo read on) in price order, since
        stores list them in no such order."""
        if self.brief.get("pick") != "cheapest" or not self.price_key:
            return ""
        found = {i for seen in observations for i, _ in self.records_in(seen)}
        rows = [(self.price_of(i), i) for i in found if self.buyable(i) and self.price_of(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Kyo] buyable rows here, cheapest first: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def wider_hold(self) -> str:
        """Once, before a cheapest checkout with many steps left: a store's keyword search reads titles, so fitting
        listings whose titles use other words are missed by searches for a requirement's value."""
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        lines = [f"{i} at {self.price_of(i):g}" for i in self.cart_ids() if self.price_of(i) is not None]
        return (f"Not run yet (no step used). {self.left()} steps are left and the pick is the cheapest. Searches match a listing's title,"
                " so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order."
                " Before checking out, run a few more searches for each item under other names for the product (its plain one-word name,"
                " synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line"
                + (f" ({'; '.join(lines)})" if lines else "") + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        """Read the attributes of listings about to be judged that no view has shown yet, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.rows and i not in self.full and not self.sold_out(i)]:
            if self.left() <= keep or self.finished or self.facts_from.get(tool) is False or not self.may_open():
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
        picks = picks[: self.pre_room(cap)]
        if len(picks) < 2 or not self.may_open(len(picks)):
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
                faults = []
                args = _shape({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.act(tool, args)
                    if reply is not None:
                        self.observe(tool, reply.get("observation"))

    @_lever('read_each', None)
    def read_each(self) -> None:
        """k17m (C3, after Secretarybird v1's pre_read and lllll v1's menus by kind): the questions are planned from the
        keys of the item's opened pages, so in a task for several items each item gets two opened listings of its kind
        before them (one at a small step allowance): its seen rows that can be bought first, else the rows of a search
        for its kind noun. An item still without one is asked about after the others."""
        if len(self.items) < 2 or self.finished:
            return
        tight = self.max_steps <= TIGHT_STEPS
        need = 1 if tight else 2
        read: list[str] = []
        for number, item in enumerate(self.items, 1):
            opened = [i for i in self.full if self.code_mine(i, number)]
            if len(opened) >= need:
                continue
            pool = [i for i in self.seen_order if i not in self.full and self.code_mine(i, number) and not self.sold_out(i)]
            if not pool and self.roles.get("search") and not tight and self.left() > 24:
                self.search_words(_kind_noun(item.shelf or item.name) or item.name)
                pool = [i for i in self.seen_order if i not in self.full and self.code_mine(i, number) and not self.sold_out(i)]
            pool.sort(key=lambda i: not self.buyable(i))
            take = pool[: need - len(opened)]
            take = take[: self.pre_room(len(take))]
            if take:
                read += [t.split(":", 1)[0] for t in self.fill_facts(take, 20)]
            if not any(self.code_mine(i, number) for i in self.full):
                self.thin.add(number)
        if read or self.thin:
            self.trace.append({"kind": "read_each", "read": read, "thin": sorted(self.thin)})

    def widen_singly(self) -> None:
        """No side-by-side view that shows attributes: read a few more listings' attributes singly."""
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
            # k17m (A4, love): a single item gets four of its listings read (key counts from three cards are mostly ties)
            lanes.append(pool[: max(0, min(4 if WIDEN_FOUR and len(self.items) == 1 else 2, 8 - opened))])
        queue = [lane[k] for k in range(4) for lane in lanes if k < len(lane)]
        queue = queue[: min(8, max(4, 2 * len(lanes)))] if WIDEN_FOUR else queue[:5]
        # while steps are plentiful: 22 stay for the rest of the episode (k17m, B3: a third of a stated detail cap)
        self.fill_facts(queue[: self.pre_room(len(queue))], 22)

    def ready_to_ask(self) -> bool:
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.left() <= self.ask_room():
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        if HEALTH and self.blind and self.step >= 4:
            return True                     # k17m10: a blind store: no opened record will ever count
        return self.step >= max(8, self.max_steps // 4)

    def ask_room(self) -> int:
        """k17m10 (HEALTH): steps that must be left for the interview to start: 8, fewer under a small step allowance
        (with 16 steps or fewer it never started)."""
        return min(8, max(3, self.max_steps // 4)) if HEALTH else 8

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
                    # k10: the value naming the shopper when one does, else the first
                    args[key] = next((v for v in spec["enum"] if SHOPPER_SIDE.search(str(v))), spec["enum"][0])
                elif HEALTH:
                    # k17m10 (HEALTH): another required field of the messaging tool, filled plainly (no question went out)
                    kinds, low = _kinds(spec), _words(key)
                    if "string" in kinds:
                        args[key] = "customer" if RECIPIENT_WORDS.search(low) else ("Question about your order" if SUBJECT_WORDS.search(low) else text)
                    elif "integer" in kinds or "number" in kinds:
                        args[key] = 1
                    elif "boolean" in kinds:
                        args[key] = True
                    else:
                        return None
                else:
                    return None
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        return None if faults else args

    def plan_picks(self, plan: dict[str, Any]) -> list[tuple[int, str]]:
        """A question plan's picks as (item number, property), in its order, without repeats."""
        ordered: list[tuple[int, str]] = []
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
        return ordered

    @_lever('interview', lambda self: self.interview_salvage())
    def interview(self) -> str:
        self.asked = True
        count = self.msgs_free()
        if count <= 0:
            return ""
        if OPEN_EACH:
            # k17m (C3): every item has opened listings of its kind before the plan
            self.read_each()
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
        # two picks more than the messages allow: messages an answered item no longer needs go to the next picks
        # k17 (Q1): a menu names several picks of one item, so the plan names more of them; (Q3) a part's model code
        picks = max(count + 2, min(12, MENU_MAX_PROPS * len(self.items) + count)) if MENU and count >= MENU_MIN_MSGS else count + 2
        if ROTATE and MENU and count >= MENU_MIN_MSGS:
            # k17m11 (ROTATE): more picks, so a later menu can lead with a property no menu has named
            picks = max(picks, min(16, MENU_MAX_PROPS * len(self.items) + 2 * count))
        plan_text = PLAN.replace("other logistics.", "other logistics." + PLAN_PART) if PART_CODES else PLAN
        prompt = plan_text.replace("COUNT", str(picks)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in _items(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        planners = PLANNERS + [HEAVY, LIGHT]
        if PLAN_LIGHT_LINE and self.cost >= PLAN_LIGHT_LINE:
            # k17k: a world that has already cost this much before the interview gets the light model's plan first
            planners = [LIGHT, HEAVY]
            self.trace.append({"kind": "plan_light", "cost": round(self.cost, 4)})
        plan = self.ask_json(prompt, planners, LIGHT_ROOM if planners[0] == LIGHT else 1800) or {}
        # k17m5 (COSTLY): the spend through the first plan tells a costly world (see the module notes)
        self.read_costly()
        ordered = self.plan_picks(plan)
        self.trace.append({"kind": "plan", "model": self.answered_by, "ask": _clip(plan.get("ask"), 120)})
        # k15: the light model's plan on the same prompt, merged with the heavy plan by a vote (see the module notes);
        # not when the request names what to ask first, which the heavy plan leads with
        if (ordered and self.answered_by == HEAVY and not first and self.elapsed() < VOTE_SECONDS and self.cost < VOTE_SPEND
                and not (self.broke or self.no_credit) and self.calls < self.call_cap - 8):
            second = self.ask_json(prompt, [LIGHT], LIGHT_ROOM) or {}
            other = self.plan_picks(second)
            if other:
                ordered = _vote(ordered, other, VOTE_WEIGHT, same=self.vote_key if VOTE_KEYS else None)
            self.trace.append({"kind": "plan2", "ask": _clip(second.get("ask"), 120), "order": [list(p) for p in ordered]})
        if HEALTH and not ordered:
            # k17m10 (HEALTH): the plan named nothing (no reply, or every label dropped): the properties most products are
            # chosen by, so the messages are not left unused (a run that drew no hidden need passed nothing in race 15)
            ordered = [(number, label) for number in range(1, len(self.items) + 1) for label in PLAIN_PICKS]
            self.trace.append({"kind": "plain_picks", "items": len(self.items)})
        sent: list[tuple[int, str, str]] = []
        self.iv_sent = sent                         # k17m2: what to show if the interview stops on an error
        told: list[str] = []
        # one question at a time, each answer read (by the light model) before the next: while some item has no
        # requirement from an answer yet, the next message goes to that item's next pick; a question whose answer wanted
        # a single question is asked once more, named with example values
        queue = [(number, label, self.question(number, label)) for number, label in ordered]
        if OPEN_EACH and self.thin:
            # k17m (C3): an item with no opened listing is asked about after the others
            queue = [q for q in queue if q[0] not in self.thin] + [q for q in queue if q[0] in self.thin]
        # k8: one message waits for the store's first cart answer (see the module notes)
        if count >= 3 and not self.reserve and self.roles.get("cart") and not self.cart_touched:
            self.held_msg = 1
        answered: set[int] = set()
        again: set[tuple[int, str]] = set()
        asked: set[tuple[int, str]] = set()
        urgent: list[tuple[int, str, str]] = []     # k6: re-asks, sent before the planned questions
        fallback: dict[tuple[int, str], tuple[int, str, str]] = {}
        # k17j: with as many messages as items or more, every item is asked once before any is asked twice
        spread = len(self.items) > 1 and (count <= SPREAD_MESSAGES or (MENU and count >= len(self.items)))
        # k17 (Q1): questions naming several properties (see the module notes)
        menus = MENU and count >= MENU_MIN_MSGS and "menu" not in self.levers_off
        across = MENU and len(self.items) > count >= 1 and "menu" not in self.levers_off   # k17j: more items than messages
        menued: set[int] = set()
        listed: set[tuple[int, str]] = set()        # every (item, property) a menu has named
        reads_all = False                            # a reply gave the value of a property a menu named after its lead
        carry: dict[int, list[tuple[int, str, str]]] = {}   # the rest of a menu after the property its reply answered
        short: dict[int, list[tuple[int, str, str]]] = {}   # after a confused menu, its next two properties
        rotate: dict[int, list[tuple[int, str, str]]] = {}  # k17m11: after a menu that drew nothing, its later properties
        shorted: set[int] = set()
        if MENU and MENU_HARD:
            # properties hard for a shopper go to the back of each item's picks (menus and single questions)
            queue = self.hard_last(queue)
        while self.msgs_free() > 0 and not self.finished and self.elapsed() < LOOP_SECONDS:
            members: list[tuple[int, str, str]] = []
            several: list[tuple[int, str, str]] = []
            if MENU:
                # a pick whose property an answer has already settled for its item is not asked
                queue = [q for q in queue if not self.settled(q[0], q[1])]
                if not urgent and not queue and MENU_FILL:
                    # k17j: messages left over: the items' most carried keys no question has named yet
                    queue = self.fillers(answered, asked, listed)
                    if queue:
                        self.trace.append({"kind": "menu_fill", "picks": [q[1] for q in queue][:8]})
            if not urgent and not queue:
                break
            if urgent:
                pick = urgent.pop(0)
            else:
                if across:
                    across = False
                    several = self.across_picks(queue)
                if several:
                    pick = several[0]
                else:
                    pick = next((q for q in queue if q[0] not in answered), queue[0])
                    if spread:
                        # k15b: an item not asked about yet first (see the module notes)
                        pick = next((q for q in queue if q[0] not in answered and all(q[0] != n for n, _ in asked)
                                     and not (OPEN_EACH and q[0] in self.thin)), pick)
                    lead_item = pick[0]
                    carried: list[tuple[int, str, str]] = []
                    relisted: list[tuple[int, str, str]] = []
                    if menus and MENU_EVERY and short.get(lead_item):
                        # after a confused menu: its next two properties alone, not the same menu again
                        two = [q for q in short.pop(lead_item) if q in queue]
                        if len(two) >= 2:
                            pick, members = two[0], two
                    elif menus and MENU_EVERY and carry.get(lead_item):
                        # after a value: the properties the menu named after it come first again
                        carried = [q for q in carry.pop(lead_item) if q in queue]
                        pick = carried[0] if carried else pick
                    elif menus and MENU_EVERY and ROTATE and rotate.get(lead_item):
                        # k17m11 (ROTATE): after a menu that drew nothing, a property no menu has named leads; the last
                        # menu's later properties (reversed) fill the rest
                        relisted = [q for q in rotate.pop(lead_item) if q in queue]
                        pick = next((q for q in queue if q[0] == lead_item and (q[0], _words(q[1])) not in listed),
                                    relisted[0] if relisted else pick)
                    elif menus and MENU_EVERY and reads_all:
                        # this shopper reads every property a question names: lead with one no menu has named yet
                        pick = next((q for q in queue if q[0] == lead_item and (q[0], _words(q[1])) not in listed), pick)
                queue.remove(pick)
                if several or members:
                    pass
                elif menus and MENU_EVERY:
                    members = self.menu_every(pick, queue, listed, carried, relisted)
                elif menus and pick[0] not in menued:
                    members = self.menu_members(pick, queue)
            number, label, _ = pick
            # without MENU_EVERY only an item's first question may be a menu
            menued.add(number)
            if several:
                listed.update((m[0], _words(m[1])) for m in several)
                batch = self.send([(number, label, self.across_question(several))])
            elif members:
                listed.update((m[0], _words(m[1])) for m in members)
                batch = self.send([(number, label, self.menu_question(number, [m[1] for m in members]))])
            else:
                batch = self.send([pick])
            if not batch:
                break
            sent += batch
            for m in several or [pick]:
                asked.add((m[0], _words(m[1])))
            before = [len(item.told) for item in self.items]
            heard = self.hear(batch, (LIGHT, HEAVY))
            if CODE_END and self.hear_failed:
                # k17m (C1): the answers are kept for another reading (and for the code finish's words)
                self.unheard += batch
            self.take_heard(heard, told)
            answered.update(n for n, item in enumerate(self.items, 1) if len(item.told) > before[n - 1])
            if MSG_CUT and self.msg_cut:
                # k17m (B6): the messages left go to single questions on the plan's next properties
                menus = across = False
            try:
                if several:
                    outcome = self.menu_outcome(heard, number, before, batch[0][2], [m[0] for m in several])
                    came = [n for n, item in enumerate(self.items, 1) if len(item.told) > before[n - 1]]
                    self.trace.append({"kind": "menu_across", "labels": [f"{m[0]}:{m[1]}" for m in several], "outcome": outcome, "items_told": came})
                    if any(n != several[0][0] for n in came):
                        reads_all = True
                        self.trace.append({"kind": "menu_reads_all", "item": 0})
                    if outcome == "crowded":
                        menus = False
                        if not came:
                            urgent.insert(0, pick)
                    continue
                if members:
                    outcome = self.menu_outcome(heard, number, before, batch[0][2])
                    drew = len(self.items[number - 1].told) - before[number - 1]
                    props = [n.split(":", 1)[0].strip() for n in self.items[number - 1].told[before[number - 1]:]]
                    self.menus.append({"item": number, "labels": [m[1] for m in members], "outcome": outcome, "needs": drew, "said": props})
                    self.trace.append({"kind": "menu", **self.menus[-1]})
                    keys = self.item_keys(number)
                    hits: list[int] = []
                    if outcome == "accepted":
                        named = [self.key_of(m[1], keys) for m in members]
                        said = {self.key_of(p, keys) for p in props}
                        hits = [i for i, key in enumerate(named) if key is not None and key in said]
                        for p in props if not hits else []:
                            # the value's property in other words: the member sharing the most words with it
                            share = [len(_stem(p) & _stem(m[1])) / max(1, len(_stem(p) | _stem(m[1]))) for m in members]
                            if max(share, default=0) > 0:
                                hits.append(share.index(max(share)))
                    if MENU_EVERY and outcome == "accepted" and not reads_all and hits and min(hits) > 0:
                        # a value for a property after the lead (and none for the lead) shows a shopper who reads them all
                        reads_all = True
                        self.trace.append({"kind": "menu_reads_all", "item": number})
                    if outcome == "crowded" or (outcome == "confused" and MENU_CONFUSED_OFF):
                        # one property a question from here on; the menu's first property alone next when nothing came
                        menus = False
                        if not drew:
                            urgent.insert(0, pick)
                    elif ROTATE and MENU_EVERY and outcome in ("declined", "confused"):
                        # k17m11 (ROTATE): nothing drawn: the lead is settled (a need there is answered most of the time);
                        # the properties after it wait, reversed, behind the item's next unnamed property
                        rotate[number] = members[1:][::-1]
                    elif outcome == "confused":
                        # a confused reply stays confused on the same property: the next question for this item names
                        # only the menu's next two properties (once), not the same menu again
                        if MENU_EVERY and number not in shorted and len(members) >= 3:
                            shorted.add(number)
                            short[number] = members[1:3]
                    elif outcome == "accepted" and MENU_EVERY:
                        # one wanted value a reply: the properties named after the answered one may hold another
                        rest = members[max(hits) + 1:] if hits else members[1:]
                        if rest:
                            carry[number] = rest
                    elif MENU_COVERS:
                        # a menu the shopper does not mind covers every property it names; one that drew a value (without
                        # MENU_EVERY) covers them up to the last one answered, and the ones after it are asked again
                        named = [self.key_of(m[1], keys) for m in members]
                        if outcome == "accepted":
                            named = named[:max(hits) + 1] if hits else named[:1]
                        covered = set(named) - {None}
                        for q in [q for q in queue if q[0] == number and self.key_of(q[1], keys) in covered]:
                            queue.remove(q)
                            asked.add((q[0], _words(q[1])))
                    continue
            except Exception as exc:  # noqa: BLE001
                # k17m2: the menu's bookkeeping raised: one property a question from here on
                self.lever_fail("menu", exc, "interview")
                menus = across = False
                continue
            if (number, _words(label)) in fallback:
                # k6: the re-ask on another item brought no requirement: the first item is asked once more in plain words
                retry = fallback.pop((number, _words(label)))
                if len(self.items[number - 1].told) == before[number - 1]:
                    urgent.insert(0, retry)
            elif _items(heard.get("crowded")) and (number, label) not in again:
                # k6: "one question at a time" for a one-property question (see the module notes): the same property on
                # another item first (races 11 and 12), then the item itself in plain words without its name (race 10)
                again.add((number, label))
                plain = (number, label, f"Just one question: what {label} would you like?")
                other = next((m for m in range(1, len(self.items) + 1) if m != number and (m, _words(label)) not in asked), None)
                if other:
                    queue = [q for q in queue if not (q[0] == other and _words(q[1]) == _words(label))]
                    urgent.insert(0, (other, label, self.question(other, label)))
                    fallback[(other, _words(label))] = plain
                else:
                    urgent.insert(0, plain)
        # k8: the picks left, in the order the interview would have taken them (re-asks first)
        self.later = urgent + sorted(queue, key=lambda q: q[0] in answered)
        # k17m11 (ROTATE): what the held message's late question builds its menu from
        self.iv_listed, self.iv_rotate, self.iv_carry, self.iv_menus = listed, rotate, carry, bool(menus and MENU_EVERY)
        self.asked_props = asked | (listed if MENU else set())
        if not sent:
            return ""
        # k14: Know Thyself's search with the new requirement values (see the module notes)
        searched = self.value_search(sorted(answered)) if told and answered else ""
        # k17 (F2, C1): more listings read, then the opened pages that show the most requirement words (see the module notes)
        more = self.open_more()
        hint = self.cover_hint()
        if CODE_END and self.unheard:
            # k17m (C1): answers the reader missed are read again now
            told += self.reread_unheard()
        if told:
            self.version += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Kyo's interviewer asked the shopper (each question used one step):\n" + "\n".join(lines)
        summary += "\nNew firm requirements from these answers: " + ("; ".join(told) if told else (
            "not read (the answer reader failed): read the answers above yourself and apply each value they want"
            if CODE_END and self.unheard else "none"))
        summary += ". Apply them to every pick (search again with the new values where needed)."
        pointed = [need for _, need in self.refs if any(need in t for t in told)]
        if pointed:
            summary += (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its page in full,"
                        " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if searched:
            summary += "\n" + searched
        if more:
            summary += "\n" + more
        if hint:
            summary += "\n" + hint
        if self.queued:
            summary += "\n[notes seen meanwhile] " + " | ".join(self.queued)
            self.queued = []
        return summary

    @_lever('menu', {})
    def item_keys(self, number: int) -> dict[frozenset[str], set[str]]:
        """k17 (Q1): the attribute keys on an item's opened listings (of its own kind when rows carry a category), each
        with the values those listings show. k17j: one property spelled several ways is one key (love's merge_keys,
        KEY_MERGE); how many listings carry each key and its longest spelling are kept for the menus."""
        cat = self.category_key()
        slots: dict[str, dict[str, Any]] = {}
        for ident in self.seen_order:
            record = self.full.get(ident)
            if not record or (cat and not self.of_item(ident, number)):
                continue
            done: set[str] = set()
            for name, value in (_fact_block(record) or {}).items():
                word = _words(name)
                if not word or word in done:
                    continue
                done.add(word)
                slot = slots.setdefault(word, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        if KEY_MERGE:
            slots = self.merge_keys(slots)
        out: dict[frozenset[str], set[str]] = {}
        for word, slot in slots.items():
            key = _stem(word)
            out.setdefault(key, set()).update(slot["values"])
            if slot["n"] >= self.key_n.get((number, key), 0):
                self.key_n[(number, key)] = slot["n"]
                self.key_name[(number, key)] = word
        return out

    @_lever('key_merge', lambda self, keys: keys)
    def merge_keys(self, keys: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """From love v20 (public as the overall #1): stores spell one property several ways across listings (a
        shorter or longer key, a run-together word). Two keys are one property when one key's words are all in the
        other's and their values agree on a listing both carry or share a value, or when they share a word and agree
        on every one of at least three listings both carry. The merged key counts every listing carrying either and is
        named by its longest spelling (the one a question can name without also naming a neighbour)."""
        def flat(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text)

        names = sorted(keys, key=lambda n: (-len(n.split()), -keys[n]["n"], n))
        merged: dict[str, dict[str, Any]] = {}
        for name in names:
            slot = keys[name]
            mine = _stem(name)
            target = None
            for big, spec in merged.items():
                theirs = _stem(big)
                both = [i for i in slot["per"] if i in spec["per"]]
                agree = sum(1 for i in both if flat(slot["per"][i]) == flat(spec["per"][i]))
                share = bool({flat(v) for v in slot["values"]} & {flat(v) for v in spec["values"]})
                if (mine <= theirs or theirs <= mine) and (agree or share) and agree == len(both):
                    target = big
                    break
                if mine & theirs and len(both) >= 3 and agree == len(both):
                    target = big
                    break
            if target is None:
                merged[name] = {"n": slot["n"], "values": list(slot["values"]), "per": dict(slot["per"])}
                continue
            spec = merged[target]
            for ident, text in slot["per"].items():
                spec["per"].setdefault(ident, text)
            for value in slot["values"]:
                if value not in spec["values"]:
                    spec["values"].append(value)
            spec["n"] = len(spec["per"])
        return merged

    @_lever('menu', [])
    def menu_members(self, lead: tuple[int, str, str], queue: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
        """k17 (Q1): the picks one menu names: the lead and the item's next picks in the plan's order, each a key the
        item's opened listings carry with more than one value; [] when the lead is no such key or fewer than
        MENU_MIN_PROPS qualify (a single question then)."""
        number = lead[0]
        keys = self.item_keys(number)

        def varying(label: str) -> frozenset[str] | None:
            key = self.key_of(label, keys)
            return key if key is not None and len(keys[key]) >= 2 else None

        first = varying(lead[1])
        if first is None:
            return []
        members, used = [lead], {first}
        for q in queue:
            if len(members) >= MENU_MAX_PROPS:
                break
            key = varying(q[1]) if q[0] == number else None
            if key is not None and key not in used:
                members.append(q)
                used.add(key)
        return members if len(members) >= MENU_MIN_PROPS else []

    @_lever('menu', [])
    def menu_every(self, lead: tuple[int, str, str], queue: list[tuple[int, str, str]], listed: set[tuple[int, str]],
                   carried: list[tuple[int, str, str]] | None = None,
                   relisted: list[tuple[int, str, str]] | None = None) -> list[tuple[int, str, str]]:
        """k17 (Q1, MENU_EVERY): the picks one question names: its lead (the next pick in the plan's order, so a
        shopper who answers only a question's first property is asked as with single questions), then the item's picks
        no question has named yet, each a key the item's opened listings carry with more than one value, not a second
        spelling of a property already named in it ("size" and "size class"), and not one the light model finds hard
        for a shopper; [] (a single question) when fewer than MENU_MIN_PROPS - 1 such picks are left. k17j: the
        properties the last menu named after the one its reply answered come first after the lead (a reply names one
        wanted value, so another may sit among them); among the others, keys whose values two listings share come
        before keys with a value per listing (an identifier or a list of purchase options; love's item_keys)."""
        number = lead[0]
        keys = self.item_keys(number)
        carried = [q for q in (carried or []) if q != lead]
        # k17m11 (ROTATE): properties an earlier menu named, after the fresh ones
        relisted = [q for q in (relisted or []) if q != lead and q not in carried]
        if KEYLESS_MENU and not any(len(v) >= 2 for v in keys.values()) and "keyless" not in self.levers_off:
            # k17m9 (KEYLESS_MENU): no attribute key on the item's opened listings with two values or more (none read, or
            # one listing opened): the plan's own picks (see the module notes)
            return self.keyless_members(lead, queue, listed, carried, relisted)

        def unique(q: tuple[int, str, str]) -> bool:
            key = self.key_of(q[1], keys)
            carriers = self.key_n.get((number, key), 0) if key is not None else 0
            return carriers >= 3 and len(keys.get(key, ())) >= carriers

        def pick_tail(hard: set[str]) -> list[tuple[int, str, str]]:
            used = {self.key_of(lead[1], keys)} - {None}
            names = [_stem(lead[1])]
            tail: list[tuple[int, str, str]] = []
            fresh = [q for q in queue if q[0] == number and q not in carried and (q[0], _words(q[1])) not in listed]
            fresh = [q for q in fresh if not unique(q)] + [q for q in fresh if unique(q)]
            for q in [*carried, *fresh, *relisted]:
                if len(tail) >= MENU_MAX_PROPS - 1:
                    break
                if q not in queue or _words(q[1]) in hard:
                    continue
                key, words = self.key_of(q[1], keys), _stem(q[1])
                if key is None or len(keys[key]) < 2 or key in used or any(words <= n or n <= words for n in names):
                    continue
                tail.append(q)
                used.add(key)
                names.append(words)
            return tail

        # the light model is asked about hard properties only when a menu can be formed at all
        if len(pick_tail(set())) < MENU_MIN_PROPS - 1:
            return []
        tail = pick_tail(self.hard_labels(number, queue))
        return [lead] + tail if len(tail) >= MENU_MIN_PROPS - 1 else []

    @_lever('keyless', [])
    def keyless_members(self, lead: tuple[int, str, str], queue: list[tuple[int, str, str]], listed: set[tuple[int, str]],
                        carried: list[tuple[int, str, str]],
                        relisted: list[tuple[int, str, str]] | None = None) -> list[tuple[int, str, str]]:
        """k17m9 (KEYLESS_MENU): the picks one question names when no attribute key of the item's opened listings shows
        two values or more: its lead, then the item's next picks in the plan's order (the ones carried from the last
        menu first), none the light
        model finds hard for a shopper, none a menu has named, and no second spelling of one already in it; [] (a single
        question) when fewer than MENU_MIN_PROPS - 1 are left."""
        number = lead[0]
        hard = self.hard_labels(number, queue)
        names = [_stem(lead[1])]
        flats = {_words(lead[1]).replace(" ", "")}           # "sleevestyle" and "sleeve style" are one property
        tail: list[tuple[int, str, str]] = []
        fresh = [q for q in queue if q[0] == number and q not in carried and (q[0], _words(q[1])) not in listed]
        for q in [*carried, *fresh, *[r for r in (relisted or []) if r not in carried]]:
            if len(tail) >= MENU_MAX_PROPS - 1:
                break
            if q == lead or q not in queue or q[0] != number or _words(q[1]) in hard:
                continue
            words = _stem(q[1])
            flat = _words(q[1]).replace(" ", "")
            if not words or flat in flats or any(words <= n or n <= words for n in names):
                continue
            tail.append(q)
            names.append(words)
            flats.add(flat)
        if len(tail) < MENU_MIN_PROPS - 1:
            return []
        self.trace.append({"kind": "keyless_menu", "item": number, "picks": [lead[1]] + [q[1] for q in tail]})
        return [lead] + tail

    @_lever('menu', [])
    def across_picks(self, queue: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
        """k17j (more requested items than messages): one question names one property of each item, the first of its
        picks in the plan's order (hard ones are at the back already); [] for fewer than two items."""
        picks: list[tuple[int, str, str]] = []
        for q in queue:
            if all(q[0] != p[0] for p in picks):
                picks.append(q)
        return picks[:MENU_MAX_PROPS] if len(picks) >= 2 else []

    @_lever('menu', lambda self, picks: self.question(picks[0][0], picks[0][1]))
    def across_question(self, picks: list[tuple[int, str, str]]) -> str:
        """k17j: one question naming one property of each of several items."""
        first = f"Which {picks[0][1]} would you like for the {self.items[picks[0][0] - 1].name}"
        rest = [f"which {label} for the {self.items[number - 1].name}" for number, label, _ in picks[1:]]
        return first + (", " + ", ".join(rest[:-1]) if len(rest) > 1 else "") + " and " + rest[-1] + "?"

    @_lever('menu', lambda self, queue: queue)
    def hard_last(self, queue: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
        """k17j (MENU_HARD): within the plan's picks, the ones the light model finds hard for a shopper go last."""
        hard = {number: self.hard_labels(number, queue) for number in {q[0] for q in queue}}
        easy = [q for q in queue if _words(q[1]) not in hard[q[0]]]
        return easy + [q for q in queue if q not in easy]

    @_lever('menu', [])
    def fillers(self, answered: set[int], asked: set[tuple[int, str]], listed: set[tuple[int, str]]) -> list[tuple[int, str, str]]:
        """k17j (MENU_FILL): when the plan's picks run out with messages left, the keys the items' opened listings
        carry most (on two listings at least, with more than one value) that no question has named, items with no
        requirement from an answer first; no store logistics, nothing hard for a shopper or already settled."""
        out: list[tuple[int, str, str]] = []
        for number, _ in sorted(enumerate(self.items, 1), key=lambda pair: pair[0] in answered):
            keys = self.item_keys(number)
            hard = self.hard.get(number, set())
            for key in sorted(keys, key=lambda k: -self.key_n.get((number, k), 0)):
                label = self.key_name.get((number, key))
                if (not label or self.key_n.get((number, key), 0) < 2 or len(keys[key]) < 2 or FILL_SKIP.search(label)
                        or (number, _words(label)) in asked or (number, _words(label)) in listed or _words(label) in hard
                        or self.settled(number, label)):
                    continue
                out.append((number, label, self.question(number, label)))
        return out[:8]

    @_lever('menu', set())
    def hard_labels(self, number: int, queue: list[tuple[int, str, str]]) -> set[str]:
        """k17 (Q1, MENU_HARD): the item's planned properties the light model finds hard for an everyday shopper to
        answer (technical measurements, jargon), kept out of menus after their lead; asked once per item."""
        if not MENU_HARD:
            return set()
        if number not in self.hard:
            labels = [q[1] for q in queue if q[0] == number]
            self.hard[number] = set()
            if len(labels) >= 2 and not (self.broke or self.no_credit) and self.calls < self.call_cap - 6:
                got = self.ask_json(HARD + f"\n\nPRODUCT: {self.items[number - 1].name}\nPROPERTIES:\n" + "\n".join(labels), [LIGHT], 600) or {}
                mine = {_words(x) for x in labels}
                self.hard[number] = {_words(str(x)) for x in _items(got.get("hard")) if _words(str(x)) in mine}
                self.trace.append({"kind": "menu_hard", "item": number, "hard": sorted(self.hard[number])})
        return self.hard[number]

    @_lever('menu', False)
    def settled(self, number: int, label: str) -> bool:
        """k17 (Q1): an answer has already given the item a requirement on this property (the same store key, or the
        same words)."""
        item = self.items[number - 1]
        keys = self.item_keys(number)
        mine, words = self.key_of(label, keys), _stem(label)
        for need in item.told:
            if need in item.dropped or ":" not in need:
                continue
            prop = need.split(":", 1)[0]
            if _stem(prop) == words or (mine is not None and self.key_of(prop, keys) == mine):
                return True
        return False

    @staticmethod
    def key_of(label: str, keys: dict[frozenset[str], set[str]]) -> frozenset[str] | None:
        """k17 (Q1): the store key a plan label names: the same words, else the closest key whose words hold the label's
        ("colour" -> "body colour")."""
        words = _stem(label)
        if words in keys:
            return words
        near = [key for key in keys if _names(words, key)]
        return min(near, key=len) if near else None

    @_lever('vote', None)
    def vote_key(self, number: int, label: str) -> frozenset[str] | None:
        """k17m (V1): the store key a plan label names on the item's opened listings ("colour" and "body colour" are
        the key "body colour"), so the vote adds up both plans' places for it; None when the label names no key."""
        return self.key_of(label, self.item_keys(number))

    @_lever('menu', lambda self, number, labels: self.question(number, labels[0]))
    def menu_question(self, number: int, labels: list[str]) -> str:
        """k17 (Q1): one question naming several properties of one item."""
        names = ", ".join(labels[:-1]) + " or " + labels[-1]
        return f"What {names} would you like for the {self.items[number - 1].name}?"

    @_lever('menu', "declined")
    def menu_outcome(self, heard: dict[str, Any], number: int, before: list[int], answer: str = "",
                     numbers: list[int] | None = None) -> str:
        """k17 (Q1): "crowded" (the shopper wants one thing a question; k17j: also read from the reply's own words),
        "accepted" (a value came), "declined" (the shopper does not mind those properties) or "confused" (none of
        these: the shopper did not follow it)."""
        if _items(heard.get("crowded")) or (answer and CROWDED_WORDS.search(answer)):
            return "crowded"
        numbers = numbers or [number]
        if any(len(self.items[n - 1].told) > before[n - 1] for n in numbers):
            return "accepted"
        if any(isinstance(e, dict) and min(max(1, _count(e.get("item")) or 1), len(self.items)) in numbers for e in _items(heard.get("free"))):
            return "declined"
        return "confused"

    def question(self, number: int, label: str) -> str:
        return f"What {label} would you like for the {self.items[number - 1].name}?"

    # k14: from Know Thyself (public as the overall #1), copied unchanged
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

    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in _kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
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

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, need in self.needs_now():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def value_search(self, numbers: list[int]) -> str:
        plan = self.search_template()
        if not plan or self.left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8 if self.max_steps > TIGHT_STEPS else 4  # k17m (B4)
        novel_all: dict[int, list[str]] = {}
        for number in numbers:
            values = self.value_words(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [(values, False), (values[:drop] + values[drop + 1:], False)] if len(values) > 1 else [(values, False)]
            # k17 (F1): first the answer's words that no row has shown yet, with the item's kind noun; of that search only
            # the rows it shows for the first time are read (see the module notes)
            novel = self.novel_words(fresh) if ANSWER_SEARCH and self.cost < ANSWER_SEARCH_SPEND else []
            if novel:
                noun = _kind_noun(item.shelf or item.name)
                novel_all[number] = novel
                tries = [(novel + ([noun] if noun and noun not in novel else []), True)] + tries[:1]
            for words, first_seen in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _shape({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                known = set(self.rows)
                reply = self.act(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.observe(tool, observation)
                cat_key = self.category_key()
                found = [i for i, _ in self.records_in(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_of(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.buyable(i) and (not cat_key or self.owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.price_of(i) or 0) <= cap)]
                if first_seen:
                    rows = [i for i in rows if i not in known]
                elif ANSWER_SEARCH and self.brief.get('pick') not in ('most', 'least'):
                    rows.sort(key=self.dearness)
                seen.append(f'{_dump(args.get(field))}: ' + (', '.join(rows[:6]) or ('no new row of that kind' if first_seen else 'no row of that kind')))
                if rows:
                    opened = self.fill_facts([i for i in rows if i not in self.full][:min(2 if first_seen else 3, budget)], self.tail() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen], 'novel': novel_all})
        if novel_all:
            return "[Kyo searched the store with each such item's new requirement words and read the first rows of its kind] " + '\n'.join(seen)
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    @_lever('answer_search', [])
    def novel_words(self, values: list[str]) -> list[str]:
        """k17 (F1): the words of the shopper's new values that no listing row seen so far shows (three at most)."""
        shown: set[str] = set()
        for row in self.rows.values():
            for value in _scalars(row):
                shown |= _toks(value)
        out: list[str] = []
        for value in values:
            for w in re.findall(r"[a-z0-9]+", _words(value)):
                if len(w) >= 3 and w not in COVER_SKIP and not _toks(w) <= shown and w not in out:
                    out.append(w)
        return out[:3]

    @_lever('menu', [])
    def late_menu(self, pick: tuple[int, str, str]) -> list[tuple[int, str, str]]:
        """k17m11 (ROTATE): the held message's question names several properties of the pick's item when the
        interview's menus were on: the properties after the last answered one first, else one no menu has named, then
        the last menu's later properties reversed; [] for a single question (menus off, a shopper who wanted one thing
        at a time, or too few properties left)."""
        if not getattr(self, "iv_menus", False) or "menu" in self.levers_off or (MSG_CUT and self.msg_cut):
            return []
        number = pick[0]
        listed = getattr(self, "iv_listed", set())
        carried = [q for q in (getattr(self, "iv_carry", {}) or {}).get(number, []) if q in self.later]
        relisted = [q for q in (getattr(self, "iv_rotate", {}) or {}).get(number, []) if q in self.later]
        lead = carried[0] if carried else next((q for q in self.later if q[0] == number and (q[0], _words(q[1])) not in listed),
                                               relisted[0] if relisted else pick)
        members = self.menu_every(lead, self.later, listed, carried, relisted)
        if members:
            self.trace.append({"kind": "late_menu", "item": number, "labels": [m[1] for m in members]})
        return members

    def late_question(self) -> str:
        """k8: release the message held back at the interview. A step that takes a message and was learned meanwhile
        keeps it (the gate sends it); otherwise it asks the next pick, for an item no answer has settled yet first.
        Returns what the answer added ("" when nothing was asked)."""
        self.held_msg = 0
        if MENU and MENU_FILL and not self.later and not self.finished:
            # k17j: no pick left for the held message: the items' most carried keys no question has named
            self.later = self.fillers({n for n, item in enumerate(self.items, 1) if item.told}, self.asked_props, set())
        if self.msgs_free() <= 0 or not self.later or self.finished or self.left() <= 6 or self.elapsed() > LOOP_SECONDS:
            return ""
        settled = {n for n, item in enumerate(self.items, 1) if item.told}
        pick = next((q for q in self.later if q[0] not in settled), self.later[0])
        members = self.late_menu(pick) if ROTATE else []
        if members:
            pick = members[0]
        self.later = [q for q in self.later if q is not pick]
        batch = self.send([(pick[0], pick[1], self.menu_question(pick[0], [m[1] for m in members]))] if members else [pick])
        if not batch:
            return ""
        told: list[str] = []
        heard = self.hear(batch, (LIGHT, HEAVY))
        if CODE_END and self.hear_failed:
            # k17m (C1): kept for another reading
            self.unheard += batch
        self.take_heard(heard, told)
        if CODE_END and self.unheard and not self.hear_failed:
            told += self.reread_unheard()
        if told:
            self.version += 1
        _, q, a = batch[0]
        self.trace.append({"kind": "late_question", "q": q, "told": told})
        return (f"Kyo's interviewer asked one more question (held back until the store's first cart answer): Q: {q} A: {a}"
                + ("\nNew firm requirements: " + "; ".join(told) + ". Apply them to the cart before checking out." if told else ""))

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
            if MSG_CUT:
                self.read_cut(answer, words)
            if SOLD_NOTE.search(words):
                # k16: a sell-out remark in the answer (see the module notes)
                self.doubt_at = self.step
                self.trace.append({"kind": "doubt", "at": self.step, "from": "answer"})
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
            if entry.get("points") is True and (number, need) not in self.refs:
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
        told.extend(f"the shopper asks: {step}" for step in self.add_steps(heard.get("steps")))

    def hear(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (HEAVY, LIGHT)) -> dict[str, Any]:
        prompt = HEAR + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it.name}" for n, it in enumerate(self.items, 1))
        prompt += "\n\nQUESTIONS AND ANSWERS:\n" + "\n".join(f"{i}. (about item {n}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.ask_json(prompt, list(models), 1500)
        # k17m (C1): no reply at all is not "no requirement": the caller keeps the answers to read again
        self.hear_failed = got is None
        return got or {}

    def gate(self, stage: str) -> tuple[bool, str]:
        """Right before a checkout ("order") or an ending without a purchase ("end"): the steps the request asks for at
        that point, done by the model's choice of store calls (a message to the shopper, a page, any store tool) until
        it finds none missing for the cart as it is now. Returns whether to go on, and what to tell the agent."""
        steps = [p for p in self.process if p["when"] == stage]
        # too late in the episode for these steps: going on still lands the order (or the ending) inside the limit
        if not steps or self.finished or not self.gate_due(stage) or self.elapsed() > GATE_SECONDS:
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
            # the light model decides; after a nudge, the heavy one
            got = self.ask_json(prompt, [HEAVY, LIGHT] if nudged else [LIGHT, HEAVY], 2000)
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
                return False, ("Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said) + "\n" if said else "") + (
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
        # k17m10 (HEALTH): a listing text no reader took in is no store note (each one cost a note model call)
        return [line for line in lines if line not in self.heard and not (HEALTH and _listing_text(line))]

    def take_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        self.price_moved_now = False
        prompt = NOTE + "\n\nITEMS AND REQUIREMENTS NOW:\n" + self.ledger() + "\n\nTHE NOTE:\n" + "\n".join(lines)
        got = self.ask_json(prompt, [LIGHT, SPARE], 1200) or {}
        gone = [str(g) for g in _items(got.get("gone")) if isinstance(g, (str, int))]
        self.unavailable.update(g for g in gone if g in self.rows)
        self.note_gone = [g for g in gone if g in self.rows]
        self.note_sold = bool(gone) or got.get("sold_out") is True
        if self.note_sold and len(lines) == 1:
            # only a note that is nothing but the sell-out is remembered: the same words later mean the same thing
            self.sold_lines.update(lines)
        asked = self.add_steps(got.get("steps"))
        if asked:
            self.version += 1
            self.notes.append("steps asked: " + "; ".join(asked))
        if got.get("kind") not in (None, "change"):
            return ("The note asks for these steps: " + "; ".join(asked)) if asked else ""
        changed = []
        budget = got.get("budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            if BUDGET_RULE and isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
                # k17 (E3): which way the budget moved (see the module notes)
                self.budget_moved = "up" if budget > self.budget else "down"
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
            # k17m (C5): a moved price and nothing else (the checkout may be sent again)
            self.price_moved_now = not (gone or drops or adds or got.get("sold_out") is True)
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or "; ".join(changed))
            if REARM:
                self.rearm()
        self.change_now = [c for c in changed if c.startswith(("budget now", "dropped", "added"))]
        rule = {"up": BUDGET_UP, "down": BUDGET_DOWN}.get(self.budget_moved, "")
        self.budget_moved = ""
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "") + (" " + rule if rule else "")).strip()

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
        if HEALTH and not self.rows and not self.cart_lines and self.own_cart:
            # k17m10 (HEALTH): no listing read: the cart by Kyo's own calls
            return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.own_cart)
        return "lines by Kyo's record of the cart calls (listing: units): " + _dump(self.cart_lines)

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

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_ids()
        key = self.check_key(ids)
        if key in self.checked or self.cost > limit or not ids:
            return ""
        # k14: the checkout check runs on every task again, on the light model (see the module notes)
        self.checked.add(key)
        check = CHECK.replace(CHECK_ANCHOR, CHECK_CATS + "\n" + CHECK_ANCHOR) if CHECK_DEPTS else CHECK
        prompt = check + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        if lead:
            quantity = "; ".join(f"item {n}: quantity {it.qty}" for n, it in enumerate(self.items, 1))
            prompt += "\n\nCART (planned, judge these listings as if they were carted at the requested quantities: " + quantity + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nCART:\n" + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (the store says it can no longer be bought). " if self.sold_out(i) else "") + self.record_of(i) for i in ids)
        depts = self.dept_lines(ids) if CHECK_DEPTS else ""
        if depts:
            # k17j: from love v20 (see the module notes)
            prompt += "\n\nCATEGORIES (each cart line's store category; per item, the store categories seen whose names share the item's words):\n" + depts
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            prompt += "\n\nOTHER LISTINGS SEEN (newest first):\n" + self.others(ids, 15)
        got = self.check_judge(prompt)
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

    # ---------- k17: pages by their requirement words, and the strict review ----------
    def of_item(self, ident: str, number: int) -> bool:
        """k17: the listing is of that requested item's kind (its category goes with that item; with no category on the
        rows, any listing of a one-item task)."""
        cat = self.category_key()
        if not cat:
            return len(self.items) <= 1
        return self.owner_of(str((self.rows.get(ident) or {}).get(cat) or "")) == number

    def line_item(self, ident: str) -> int:
        """k17: the requested item a cart line is for (0 when its category goes with none)."""
        if len(self.items) <= 1:
            return 1
        return next((n for n, _ in enumerate(self.items, 1) if self.of_item(ident, n)), 0)

    def same_kind(self, ident: str, other: str) -> bool:
        cat = self.category_key()
        if not cat:
            return True
        return (str((self.rows.get(ident) or {}).get(cat) or "").strip().lower()
                == str((self.rows.get(other) or {}).get(cat) or "").strip().lower())

    def in_budget(self, ident: str) -> bool:
        known, limit = self.price_of(ident), self.budget
        if known is None or isinstance(limit, bool) or not isinstance(limit, (int, float)):
            return True
        return known <= limit

    def row_words(self, ident: str) -> set[str]:
        out: set[str] = set()
        for value in _scalars(self.rows.get(ident) or {}):
            out |= _toks(value)
        # k17m (C2): words an every-word search that showed all its matches found somewhere in the row's record
        return out | self.hit_words.get(ident, set()) if SEARCH_KIND else out

    def search_keys(self, args: dict[str, Any]) -> tuple[list[tuple[str, tuple[str, ...]]], tuple[Any, ...], bool]:
        """k17m (C2, after Secretarybird v1): a search call's free-text arguments with their words, its choice arguments
        (the mode), and whether it asks for the first page."""
        props = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        shape = self.shapes.get(self.roles.get("search") or "") or {}
        texts: list[tuple[str, tuple[str, ...]]] = []
        mode: list[tuple[str, str]] = []
        first = True
        for key, value in args.items():
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            paging = key == shape.get("next_arg") or bool(PAGE_KEY.search(key))
            if isinstance(value, str):
                if spec.get("enum"):
                    mode.append((key, value))
                elif paging:
                    first = first and not value.strip()
                else:
                    words = tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", value.lower())))
                    if words:
                        texts.append((key, words))
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and paging:
                floor = spec.get("minimum") if isinstance(spec.get("minimum"), (int, float)) else (1 if "page" in key.lower() else 0)
                if value > floor:
                    first = False
        return texts, tuple(sorted(mode)), first

    def word_rule(self, args: dict[str, Any], total: int, marks: tuple[str, ...]) -> None:
        """k17m (C2, Secretarybird v1's word_rule and lead_rule): two first-page searches with one text argument and the
        same other arguments: when one query extends the other, more matches with more words show that any word matches,
        fewer that every word is required; equal results from queries that agree on their first n words, beside
        different ones from queries agreeing on n - 1, show that only n words are read."""
        texts, mode, first = self.search_keys(args)
        if not first:
            return
        for key, words in texts:
            ctx = (key, mode)
            others = tuple(sorted((k, str(v)) for k, v in args.items() if k != key and not PAGE_KEY.search(k)))
            subset = False
            bounds = self.lead_bounds.setdefault(ctx, [-1, set()])
            for old_ctx, old_others, old_words, old_total, old_marks in self.word_seen:
                if old_ctx != ctx or old_others != others or old_words == words:
                    continue
                if old_total != total:
                    verdict = _word_verdict(words, total, old_words, old_total, self.lead_words or None)
                    if verdict == "subset":
                        subset = True
                    elif verdict:
                        self.word_kind = verdict
                common = _common_lead(words, old_words)
                if old_total != total or old_marks != marks:
                    bounds[0] = max(bounds[0], common)
                elif common > 0 and len(set(marks)) >= 3 and common == min(len(words), len(old_words)):
                    bounds[1].add(common)
            if subset and not self.word_kind:
                self.word_kind = "all"
            self.word_seen.append((ctx, others, words, total, marks))
        claims = []
        for low, highs in self.lead_bounds.values():
            above = sorted(h for h in highs if h > low)
            if above and above[0] == low + 1:
                claims.append(above[0])
        self.lead_words = min(claims) if claims else 0

    @_lever('search_kind', "")
    def search_note(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> str:
        """k17m (C2, after Secretarybird v1's search_note): after a search, what the store's word rule means for the
        next query (A1's "all the values in one query" holds only where every word is required), and in such a store,
        when every match is shown, the rows found by the query's words are marked as carrying them."""
        rows = [i for i, _ in self.records_in(observation)]
        shape = self.read_shape(name, observation) if len(rows) >= 2 else self.shapes.get(name)
        count = _at(observation, shape.get("count")) if shape and shape.get("count") else None
        total = count if isinstance(count, int) and not isinstance(count, bool) and count >= len(rows) else len(rows)
        before = (self.word_kind, self.lead_words)
        self.word_rule(args, total, tuple(rows))
        if (self.word_kind, self.lead_words) != before:
            self.trace.append({"kind": "search_kind", "words": self.word_kind, "lead": self.lead_words})
        notes: list[str] = []
        if self.lead_words and before[1] != self.lead_words:
            notes.append(f"This store's keyword search reads only the first {self.lead_words} words: put a required value and the"
                         " item kind first, and search other values separately.")
        if not rows:
            return " ".join(notes)
        if total > len(rows):
            if self.word_kind == "all":
                notes.append(f"{total} listings matched and {len(rows)} are shown: here every word is required, so adding a"
                             " required value's word narrows the search.")
            elif self.word_kind == "any":
                notes.append(f"{total} listings matched and {len(rows)} are shown: here any one word matches, so adding words"
                             " widens the search; search a required value's distinctive word with the kind noun alone, or read further pages.")
        elif self.word_kind == "all":
            texts, _, first = self.search_keys(args)
            words = [w for _, ws in texts for w in (ws[: self.lead_words] if self.lead_words else ws)]
            if first and words:
                found = set().union(*(_toks(w) for w in words))
                for ident in rows:
                    self.hit_words.setdefault(ident, set()).update(found)
                fresh = [i for i in rows if i not in self.full and self.buyable(i) and self.in_budget(i)][:6]
                if fresh:
                    notes.append("Every word is required here and every match is shown, so each of these rows carries the searched"
                                 " words somewhere in its record (often in fields the title hides): " + ", ".join(fresh) + ".")
        return " ".join(notes)

    def page_words(self, ident: str) -> set[str]:
        """k17 (C1): the words of an opened page's attribute names and values."""
        out: set[str] = set()
        for key, value in (_fact_block(self.full.get(ident) or {}) or {}).items():
            out |= _toks(key) | _toks(value)
        return out

    def cover(self, ident: str, words: list[str]) -> int:
        have = self.page_words(ident)
        return sum(1 for w in words if _toks(w) and _toks(w) <= have)

    def req_words(self, number: int, told_only: bool = False) -> list[str]:
        """k17 (C1): the words of an item's plain requirement values (with told_only, those the shopper or the store
        added during the episode); limits, exclusions and values read from another listing give none."""
        item = self.items[number - 1]
        pointers = {need for _, need in self.refs}
        out: list[str] = []
        for need in (item.told if told_only else item.musts + item.told):
            if need in item.dropped or need in pointers or ":" not in need:
                continue
            value = need.split(":", 1)[1]
            if COVER_VALUE_SKIP.search(value):
                continue
            for w in re.findall(r"[a-z0-9]+", _words(value)):
                if (len(w) >= 3 or w.isdigit()) and w not in COVER_SKIP and w not in out:
                    out.append(w)
        return out

    def dearness(self, ident: str) -> float:
        known = self.price_of(ident)
        return float("inf") if known is None else known

    @_lever('more_opens', "")
    def open_more(self) -> str:
        """k17 (F2), k17j aimed at the answers: for each item with a value from the shopper's answers, its unread
        buyable rows inside the budget whose words carry that value come first (most of its words first), then up to
        four more of its first rows; six an item, eight in all (see the module notes)."""
        if (not MORE_OPENS or self.cost >= MORE_OPENS_SPEND or self.left() < 26 or self.finished or self.max_steps <= TIGHT_STEPS
                or (OPENS_CAP and (self.opens_cap or self.opens_out))):
            return ""
        picks: list[str] = []
        for number, _ in enumerate(self.items, 1):
            words = self.req_words(number, True)
            if not words:
                continue
            pool = [i for i in self.seen_order if self.of_item(i, number) and i not in self.full and i not in picks
                    and self.buyable(i) and self.in_budget(i)]
            score = {i: sum(1 for w in words if _toks(w) <= self.row_words(i)) for i in pool}
            hits = sorted((i for i in pool if score[i]), key=lambda i: -score[i])
            picks += (hits + [i for i in pool if not score[i]][:4])[:6]
        texts = self.fill_facts(picks[:8], 18)
        if not texts:
            return ""
        self.trace.append({"kind": "more_opens", "n": len(texts)})
        return f"[Kyo read {len(texts)} more listings of the requested kinds: " + ", ".join(t.split(":", 1)[0] for t in texts) + "]"

    @_lever('cover', "")
    def cover_hint(self) -> str:
        """k17 (C1): the opened pages of each item whose attributes show the most of its requirement words."""
        if not COVER or self.brief.get("pick") in ("cheapest", "most", "least"):
            return ""
        parts = []
        for number, _ in enumerate(self.items, 1):
            words = self.req_words(number)
            if not words:
                continue
            pool = [i for i in self.seen_order if i in self.full and self.of_item(i, number) and self.buyable(i)
                    and self.in_budget(i) and self.cover(i, words)]
            pool.sort(key=lambda i: (-self.cover(i, words), self.dearness(i)))
            if pool:
                parts.append((f"item {number} " if len(self.items) > 1 else "") + f"({', '.join(words[:8])}): "
                             + ", ".join(f"{i} {self.cover(i, words)}/{len(words)}" for i in pool[:3])
                             # k17j: a page showing every word is carted now (the guide's early cart, pushed by code)
                             + (f"; {pool[0]} shows every requirement word: cart it now if its record meets each requirement"
                                if self.cover(pool[0], words) == len(words) else ""))
        if not parts:
            return ""
        self.trace.append({"kind": "cover_hint", "parts": parts})
        return ("[Kyo] Opened listings whose attributes show the most of the requirement words (cheapest first among equals): "
                + " | ".join(parts) + ". The words only point the way: judge each requirement on the attribute for that property.")

    @_lever('cover', "")
    def cover_hold(self) -> str:
        """k17 (C1): at the first checkout, once (see the module notes); "" to go on."""
        if not COVER or self.cover_done or self.brief.get("pick") in ("cheapest", "most", "least"):
            return ""
        self.cover_done = True
        cart = self.cart_ids()
        notes = []
        for line in cart:
            number = self.line_item(line)
            if not 1 <= number <= len(self.items) or line not in self.full:
                continue
            told, words = self.req_words(number, True), self.req_words(number)
            pool = [i for i in self.seen_order if i in self.full and i not in cart and self.of_item(i, number) and self.buyable(i)
                    and self.in_budget(i) and self.same_kind(i, line)]
            have = self.page_words(line)
            lacks = [w for w in told if not _toks(w) <= have]
            if lacks:
                full = sorted((i for i in pool if self.cover(i, told) == len(told)), key=self.dearness)
                if full:
                    notes.append(f"{line}'s attributes do not show {', '.join(lacks)} from the shopper's answers, while {full[0]}'s show all of them"
                                 + (f" (so do {', '.join(full[1:3])})" if len(full) > 1 else ""))
                    continue
            mine, price = self.cover(line, words), self.price_of(line)
            if mine and price is not None:
                cheaper = sorted((i for i in pool if self.cover(i, words) >= mine and self.dearness(i) < price), key=self.dearness)
                if cheaper:
                    notes.append(f"{cheaper[0]} shows as many of the requirement words as {line} ({mine} of {len(words)}) and costs less"
                                 f" ({self.dearness(cheaper[0]):g} against {price:g})")
        if not notes or self.left() < self.tail() + 4:
            return ""
        self.trace.append({"kind": "cover_hold", "notes": notes})
        return ("Not run yet. Kyo compared the cart with the other opened listings of its kind: " + "; ".join(notes)
                + ". If that listing meets every requirement (judge each one on its attribute for that property), put it in the cart in"
                " place of the line at the requested quantity and check out; if it does not, check out again as you are.")

    def judge_lean(self, prompt: str, max_tokens: int, kind: str) -> dict[str, Any] | None:
        """k17 ($): the strict reviews on the heavy model below AUDIT_LEAN, on the light model (with room) past it; late
        in the episode the heavy model, being faster."""
        late = self.elapsed() > LIGHT_SECONDS
        if self.cost < AUDIT_LEAN or late:
            got = self.ask_json(prompt, [HEAVY], max_tokens)
            if got is not None or late or self.broke or self.no_credit or self.calls >= self.call_cap:
                return got
        else:
            self.trace.append({"kind": "lean", "judge": kind, "cost": round(self.cost, 4)})
        return self.ask_json(prompt, [LIGHT], LIGHT_ROOM)

    def mark_reqs(self, pairs: list[tuple[str, int]], kind: str) -> dict[str, dict[str, str]]:
        """k17 (C2, E1): the strict review of (listing, item number) pairs: each live requirement of the item marked met,
        broken or unstated."""
        needs: dict[int, list[str]] = {}
        for _, number in pairs:
            needs[number] = [m for n, m in self.needs_now() if n == number]
        prompt = REQS + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nREQUIREMENTS TO MARK, BY ITEM:\n" + "\n".join(f"item {n}: " + (" | ".join(r) or "(none beyond the kind of product)") for n, r in needs.items())
        prompt += "\n\nLISTINGS TO MARK:\n" + "\n".join(
            f"{i} (item {n}): " + ("SOLD OUT. " if self.sold_out(i) else "")
            + (("OPENED " + _dump(_clip(self.full[i], 200))[:1400]) if i in self.full else ("ROW " + _dump(_clip(self.rows.get(i), 120))[:600]))
            for i, n in pairs)
        got = self.judge_lean(prompt, 2500, kind) or {}
        out: dict[str, dict[str, str]] = {}
        for entry in _items(got.get("marks")):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            req = entry.get("req")
            if ident in self.rows and isinstance(req, dict):
                out[ident] = {str(k).strip(): str(v).strip().lower() for k, v in req.items()}
        self.trace.append({"kind": kind, "pairs": [list(p) for p in pairs], "marks": _clip(out, 60)})
        self.marks_seen.update(out)                 # k17m3: for the code's own picks (SOLD_BUY, C1)
        return out

    @staticmethod
    def mark_of(marks: dict[str, str], need: str) -> str:
        """k17: a requirement's mark, found by its text or else by its property name; "unstated" when not marked."""
        if need in marks:
            return marks[need]
        prop = _words(need.split(":", 1)[0])
        return next((v for k, v in marks.items() if _words(str(k).split(":", 1)[0]) == prop), "unstated")

    @_lever('req_audit', "")
    def req_audit(self) -> str:
        """k17 (C2): after the checkout check passed (see the module notes); "" to go on."""
        if not REQ_AUDIT or self.cost > SPEND_SOFT or len(self.req_marks) >= 2 or self.left() < self.tail() + 4:
            return ""
        cart = self.cart_ids()
        key = self.check_key(cart)
        if not cart or key in self.req_marks:
            return ""
        alts: dict[str, list[str]] = {}
        for line in cart:
            number = self.line_item(line)
            if not 1 <= number <= len(self.items):
                continue
            words = self.req_words(number, True)
            if not words:
                continue
            pool = [i for i in self.seen_order if i in self.full and i not in cart and self.of_item(i, number) and self.buyable(i)
                    and self.in_budget(i) and self.same_kind(i, line) and self.cover(i, words) == len(words)]
            if self.brief.get("pick") not in ("most", "least"):
                pool.sort(key=self.dearness)
            if pool:
                alts[line] = pool[:4]
        if not alts:
            return ""
        self.req_marks.add(key)
        pairs = [(line, self.line_item(line)) for line in alts] + [(a, self.line_item(line)) for line, ids in alts.items() for a in ids]
        marks = self.mark_reqs(pairs, "req_audit")
        notes = []
        for line, ids in alts.items():
            number = self.line_item(line)
            item = self.items[number - 1]
            reqs = [m for n, m in self.needs_now() if n == number]
            got = marks.get(line) or {}
            broken = [r for r in reqs if self.mark_of(got, r) == "broken"]
            short = [r for r in item.told if r not in item.dropped and self.mark_of(got, r) != "met"]
            if not broken and not short:
                continue
            fine = [a for a in ids if marks.get(a) and all(self.mark_of(marks[a], r) == "met" for r in reqs)]
            if fine:
                notes.append(f"{line}: " + "; ".join([f"{r} broken" for r in broken] + [f"{r} not stated" for r in short if r not in broken])
                             + f", while {fine[0]} meets every requirement")
        if not notes:
            return ""
        self.trace.append({"kind": "req_hold", "notes": notes})
        return ("Not run yet. A strict review of the cart, requirement by requirement, against opened listings that show every word of the"
                " shopper's answers: " + " | ".join(notes) + ". Put that listing in the cart in place of the line at the requested quantity"
                " unless its record clearly fails a requirement, then check out; if the review is wrong, check out again as you are.")

    @_lever("empty_seek", "")
    def empty_seek(self) -> str:
        """k17m4 (EMPTY_SEEK): EMPTY_SEEK steps or fewer are left, the cart is empty, no checkout was tried and no review
        has named a lead: k2's value searches run now instead of only before an ending with more than eight steps left
        (each plain value with the item's kind noun, the next page when the first shows nothing new, then the item's
        name; up to three new rows opened). What they opened, for the agent ("" when nothing)."""
        if self.sought or self.finished or self.lead_items or self.left() < 8:
            return ""
        before = set(self.full)
        self.seek_missing()
        fresh = [i for i in self.full if i not in before]
        self.trace.append({"kind": "empty_seek", "opened": fresh, "left": self.left()})
        return (f"[Kyo] {self.left()} steps are left and the cart is empty, so Kyo searched each requirement value with the item's kind "
                f"and opened {', '.join(fresh)}. If one of them meets every requirement, put it in the cart now and check out.") if fresh else ""

    @_lever("sold_review", "")
    def sold_review(self) -> str:
        """k17m4 (SOLD_REVIEW): in an episode with a sell-out, a cart the checkout check passed is reviewed requirement by
        requirement once (see the module notes): each line beside up to three listings of its kind that can be bought
        inside the budget (the sold-out reviews' named fits first, then the opened ones showing most requirement words).
        A line with a requirement broken while one of them meets every one, or on a cheapest pick a cheaper one meeting
        every one, holds the checkout once. "" to go on."""
        if not SOLD_REVIEW or not self.gone or self.cost > self.spend_hard or self.left() < self.tail() + 4:
            return ""
        cart = self.cart_ids()
        key = self.check_key(cart)
        if not cart or key in self.sold_reviewed:
            return ""
        self.sold_reviewed.add(key)
        cheapest = "cheapest" == self.brief.get("pick")
        alts: dict[str, list[str]] = {}
        for line in cart:
            number = self.line_item(line)
            if not 1 <= number <= len(self.items):
                continue
            words = self.req_words(number)
            others = [i for i in self.swap_named + [j for j in self.seen_order if j in self.full] if i not in cart and i != line
                      and i not in self.gone and self.buyable(i) and self.in_budget(i) and self.same_kind(i, line)]
            others = list(dict.fromkeys(others))
            named = [i for i in others if i in self.swap_named]
            rest = sorted((i for i in others if i not in named), key=lambda i: (-self.cover(i, words), self.dearness(i)))
            pool = (named + rest)[:3]
            if cheapest:
                pool.sort(key=self.dearness)
            alts[line] = pool
        if not any(alts.values()):
            return ""
        pairs = [(line, self.line_item(line)) for line in alts] + [(a, self.line_item(line)) for line, ids in alts.items() for a in ids]
        marks = self.mark_reqs(pairs, "sold_review")
        notes = []
        for line, ids in alts.items():
            number = self.line_item(line)
            reqs = [m for n, m in self.needs_now() if n == number]
            if not reqs:
                continue
            got = marks.get(line) or {}
            broken = [r for r in reqs if self.mark_of(got, r) == "broken"]
            fine = [a for a in ids if marks.get(a) and all(self.mark_of(marks[a], r) == "met" for r in reqs)]
            if not fine:
                continue
            if broken:
                notes.append(f"{line}: " + "; ".join(f"{r} broken" for r in broken) + f", while {fine[0]} meets every requirement")
            elif cheapest:
                own = self.price_of(line)
                cheaper = [a for a in fine if own is not None and self.price_of(a) is not None and self.price_of(a) < own]
                if cheaper:
                    notes.append(f"{line} costs {own:g}, while {cheaper[0]} at {self.price_of(cheaper[0]):g} meets every requirement")
        self.trace.append({"kind": "sold_review", "cart": cart, "alts": _clip(alts, 40), "notes": notes})
        if not notes:
            return ""
        return ("Not run yet. After the sell-out, a strict review of the cart, requirement by requirement, against the other "
                "listings of its kind: " + " | ".join(notes) + ". Put that listing in the cart in place of the line at the requested "
                "quantity unless its record clearly fails a requirement, then check out; if the review is wrong, check out again as you are.")

    @_lever("costly", None)
    def read_costly(self) -> None:
        """k17m5 (COSTLY): once, right after the first question plan (an episode without one keeps the plain line): a spend
        above COSTLY_LINE marks a costly world, and the episode's hard line drops to COSTLY_HARD, never closer than
        COSTLY_ROOM above the spend now (see the module notes). The loop running the interview lowers its cap with it."""
        if self.costly_read or not COSTLY_HARD:
            return None
        self.costly_read = True
        if self.cost > COSTLY_LINE:
            self.spend_hard = min(SPEND_HARD, max(COSTLY_HARD, self.cost + COSTLY_ROOM))
            self.trace.append({"kind": "costly", "cost": round(self.cost, 4), "hard": round(self.spend_hard, 4)})
        return None

    def gate_tell(self, stage: str) -> tuple[bool, str]:
        """k17m6: the gate for this stage, then at the order (nothing holding it) SPARE_TELL's cart-and-total message."""
        go, said = self.gate(stage)
        if not go or stage != "order" or not SPARE_TELL:
            return go, said
        told = self.tell_cart()
        if told.startswith("HOLD:"):
            return False, (said + "\n" if said else "") + told[5:]
        return go, (said + "\n" + told) if said and told else (said or told)

    @_lever("spare_tell", "")
    def tell_cart(self) -> str:
        """k17m6 (SPARE_TELL): a message no question will use tells the shopper the final cart and its total (see the
        module notes). "" when nothing was sent; "HOLD:" and why when the reply states a new requirement."""
        if not SPARE_TELL or self.finished or not self.roles.get("ask") or _count(self.brief.get("messages")) is None:
            return ""
        if any(p["when"] == "order" and p["messages"] for p in self.process):
            return ""
        if self.held_msg or self.msgs_free() <= 0 or self.elapsed() > GATE_SECONDS or self.left() < self.tail() + 2:
            return ""
        cart = self.cart_ids()
        mark = self.cart_mark()
        if not cart or mark == self.told_mark:
            return ""
        parts, total, priced = [], 0.0, True
        for ident in cart:
            units = self.cart_lines.get(ident) or 1
            each = self.price_of(ident)
            priced = priced and each is not None
            parts.append(f"{units} x {self.short_name(ident)} ({ident})" + (f" at {each:g} each" if each is not None else ""))
            total += (each or 0) * units
        # the total in the store's own price figures (a budget unit word could name another scale)
        text = ("Before I place your order, here is the final cart: " + "; ".join(parts)
                + (f". Total: {total:g}." if priced else "."))
        batch = self.send([(1, "", text)], self.tail() + 1)
        if not batch:
            return ""
        self.told_mark = mark
        told: list[str] = []
        if not self.model_down() and self.cost < self.spend_hard + 0.02:
            self.take_heard(self.hear(batch, (LIGHT, HEAVY)), told)
        _, q, a = batch[0]
        self.trace.append({"kind": "spare_tell", "q": q[:300], "a": a[:300], "told": told})
        if told:
            self.version += 1
            return ("HOLD:Kyo told the shopper the final cart and its total, and the reply adds: " + "; ".join(told)
                    + ". Fix the cart for it, then check out again.")
        return "Kyo told the shopper the final cart and its total (a message no question needed). Reply: " + a[:300]

    def name_of(self, ident: str) -> str:
        """k17m8: a listing's full name, from its record's name or title field (not a shop's or brand's)."""
        row = self.full.get(ident) or self.rows.get(ident) or {}
        return next((str(v) for k, v in row.items() if isinstance(v, str) and re.search(r"name|title|heading", str(k), re.I)
                     and not re.search(r"brand|shop|seller|maker|vendor|store|company|categ|depart", str(k), re.I)), "")

    @_lever("sold_twin", [])
    def sold_twin(self, lost: list[str]) -> list[str]:
        """k17m8 (SOLD_TWIN): a cart line sold out and the review named nothing among the listings read: Kyo searches each
        sold-out listing's name, opens up to two listings of its kind, in stock and inside the budget, whose name is the
        same or shares most words (same name first), and asks the review again; past the soft spend line, where the
        review does not run, the same-name listings are named by code. Once per sold-out line, only while the steps for
        the search, the opens and a swap are left (see the module notes). The replacements named."""
        if not SOLD_TWIN or self.finished or self.left() < self.tail() + 6 or self.elapsed() > LOOP_SECONDS - 20:
            return []
        same_name: list[str] = []
        found: list[str] = []
        for line in lost:
            if line in self.twin_sought:
                continue
            self.twin_sought.add(line)
            name = self.name_of(line)
            mine = _name_words(name)
            if len(mine) < 2:
                continue
            plain = re.sub(r"\u3010[^\u3011]*\u3011|\[[^\]]*\]|\([^)]*\)", " ", name)
            query = " ".join(list(dict.fromkeys(w for w in re.findall(r"[A-Za-z0-9]+", plain) if len(w) > 1))[:6])
            ids = self.search_words(query)
            ranked = []
            for ident in ids:
                if ident == line or ident in lost or ident in self.gone or ident in self.cart_ids() or self.sold_out(ident):
                    continue
                if not self.buyable(ident) or not self.in_budget(ident) or not self.same_kind(ident, line):
                    continue
                theirs = _name_words(self.name_of(ident))
                share = len(mine & theirs) / max(1, len(mine | theirs))
                if theirs == mine or share >= 0.6:
                    ranked.append((0 if theirs == mine else 1, -share, self.dearness(ident), ident))
            ranked.sort()
            twins = [r[3] for r in ranked]
            fresh = [i for i in twins if i not in self.opened][:2]
            if fresh:
                self.open_rows(fresh)
            found.extend(twins[:3])
            same_name += [r[3] for r in ranked if r[0] == 0][:2]
            self.trace.append({"kind": "sold_twin", "lost": line, "query": query, "twins": twins[:4], "opened": fresh})
        if not found:
            return []
        picks = self.replacements(lost)
        if not picks and self.cost > SPEND_SOFT and same_name:
            # the review does not run past the soft spend line: the same-name listings are the replacements
            picks = [i for i in dict.fromkeys(same_name) if not self.sold_out(i)][:2]
            self.swap_named = picks + [i for i in self.swap_named if i not in picks]
            self.trace.append({"kind": "sold_twin", "named": picks, "by": "code"})
        return picks

    def trail(self) -> tuple[str, bool]:
        """Before a checkout: each requirement whose value is on another listing has every page of its chain opened.
        Kyo opens the links it can name itself; a link nobody has found yet holds the checkout back (twice at most).
        Returns what it saw and whether to hold the checkout."""
        if HEALTH and self.blind:
            # k17m10 (HEALTH): no listing record to follow a chain with: the loop model follows it in the raw results
            return "", False
        live = {need for _, need in self.needs_now()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        # a chain's links are pages: the page view (the offer view when it is apart from the attributes view)
        tool = self.roles.get("offer") or self.roles.get("open")
        target = self.single_target(tool)
        seen: list[str] = []
        lost: list[str] = []
        chased = 0
        while pending and self.trails < 6 and self.cost < self.spend_hard:
            self.trails += 1
            records = []
            for ident in reversed(self.seen_order):
                # an opened page is the page view's record (with the attributes when they were read too)
                record = (self.full.get(ident) or self.rows.get(ident)) if ident in self.opened else None
                if record:
                    records.append("OPENED " + _dump(_clip(record, 300))[:1400])
                else:
                    # k3: 450, not 300: a row's shop or brand field often came last and was cut, so the chain check
                    # could not tell which rows a shop sells
                    records.append("ROW " + _dump(_clip(self.rows.get(ident), 80))[:450])
                if len(records) >= 60:
                    break
            prompt = TRAIL + "\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending)
            # k3: the records naming a shop or brand the requirements name come first, in a block of their own (the light
            # model picked rows of a look-alike shop when they were mixed in with the rest)
            marks = {w.lower() for need in pending for w in re.findall(r"\b[A-Z][a-z]{3,}\b", need) if w not in CHASE_STARTERS}
            named = [r for r in records if marks and any(m in r.lower() for m in marks)]
            if named:
                prompt += "\n\nRECORDS THAT NAME A SHOP OR BRAND FROM THE REQUIREMENTS (newest first):\n" + "\n".join(named)
                prompt += "\n\nOTHER RECORDS (newest first):\n" + "\n".join(r for r in records if r not in named)
            else:
                prompt += "\n\nRECORDS (newest first):\n" + "\n".join(records)
            # k3: room for the light model's thinking on a chain with several named properties (1800 cut it short)
            got = self.ask_json(prompt, [LIGHT, HEAVY], 4000) or {}
            self.trace.append({"kind": "trail", "pending": pending, "got": _clip(got, 150)})
            reach: list[str] = []
            lost = []
            for chain in _items(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _items(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.opened for i in ids) and str(chain.get("value") or "").strip():
                    self.ref_values[need] = str(chain["value"]).strip()
                    # k2: a value wanted to go with the page's value (the store's pairing rule), not the value itself
                    if str(chain.get("paired") or "").strip():
                        self.ref_paired[need] = str(chain["paired"]).strip()
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    about = next((str(l.get("about") or "") for l, i in zip(links, ids) if i not in self.rows), "") or "the listing it points at"
                    lost.append(f"{need} -> find {about}")
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and lost and pending and chased < 3 and self.left() > 8 and not self.finished:
                # k2: a link nobody has seen yet is searched for in code (its shop or brand and its kind), and the
                # chain is read again with that page opened
                found = False
                for item in lost[:2]:
                    chased += 1
                    found = self.chase(item.split(" -> find ", 1)[-1]) or found
                if found:
                    continue
            if not reach or not tool or not target or self.left() < len(reach) + 4 or self.finished:
                break
            for ident in reach[: max(0, min(3, 6 - len(seen)))]:
                if not self.may_open(1, 0):
                    break
                faults: list[str] = []
                args = _shape({target: ident}, self.schemas.get(tool), "args", faults)
                if faults:
                    break
                reply = self.act(tool, args)
                if reply is None or self.finished:
                    break
                self.observe(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.show(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n"
        if self.ref_values:
            text += "[values read from the pages requirements point at] " + "; ".join(
                f"{k} -> {v}" + (f" (the store's pairing rule pairs it with {self.ref_paired[k]})" if k in self.ref_paired else "")
                for k, v in self.ref_values.items()) + "\n"
        hold = bool(lost) and self.trail_blocks < 2 and self.left() > 5
        if hold:
            self.trail_blocks += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def search_words(self, text: str, page: tuple[str, Any] | None = None) -> list[str]:
        """k2: run the store's word search for `text`, the call built in code from the tool's schema (its text argument,
        within its length limit, and its word-search mode when the tool has modes); the ids it shows. `page` is the
        (parameter, value) that asks for a later page; the next page's pair is left in `self.next_page`."""
        self.next_page = None
        tool = self.roles.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        key = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v
                    and re.search(r"query|phrase|text|^q|keyword|search|words|string|find|term", k, re.I)), None)
        if not tool or not key or self.left() < 3 or self.finished:
            return []
        limit = props[key].get("maxLength") if isinstance(props[key].get("maxLength"), int) else 60
        cut = ""
        for w in text.split():
            if len(cut) + len(w) + 1 > limit:
                break
            cut = f"{cut} {w}".strip()
        args: dict[str, Any] = {key: cut}
        for k, v in props.items():
            if isinstance(v, dict) and v.get("enum") and k in (schema.get("required") or []):
                args[k] = next((e for e in v["enum"] if re.search(r"text|word|keyword|search|query|phrase", str(e), re.I)), v["enum"][0])
        if page and page[0] in props:
            args[page[0]] = page[1]
        faults: list[str] = []
        args = _shape(args, schema, "args", faults)
        if faults:
            return []
        reply = self.act(tool, args)
        if reply is None:
            return []
        seen = reply.get("observation")
        self.observe(tool, seen)
        plan = self.paging(tool, seen)
        if plan and plan[3] is not None and plan[0] > plan[1]:
            self.next_page = (plan[2], plan[3])
        return [ident for ident, _ in self.records_in(seen)]

    def seek_missing(self) -> bool:
        """k2: before an ending without a purchase, each requested item that has no fitting lead is searched for in code:
        each plain value it must have (stated in the request or told by the shopper) with the item's kind noun, one value
        a query, and the next page when the first shows no unread row; then the item's name alone. The unread rows found
        are opened. (sweepsim.py: one-value queries showed 7 of the 10 race-11 answers agents never saw and 8 of 12 in
        race 10, about half of them on page 2.) True when a page was opened."""
        if self.sought:
            return False
        self.sought = True
        opened = False
        pointers = {need for _, need in self.refs}
        tried: list[str] = []
        for k, item in enumerate(self.items, 1):
            if self.lead_items.get(str(k)) or self.left() < 7 or item.name == "the requested product":
                continue
            # only plain values are search words: an exclusion, a limit, a choice or a pointer would find the wrong rows
            live = [m for m in item.musts + item.told if ":" in m and m not in pointers and m not in item.dropped]
            values = [v for v in (m.split(":", 1)[-1].strip() for m in live) if 0 < len(v.split()) <= 3 and not VALUE_SKIP.search(v)]
            noun = _kind_noun(item.shelf or item.name)
            queries = [" ".join(dict.fromkeys(f"{v} {noun}".lower().split())) for v in dict.fromkeys(values)][:3] + [item.name]
            for query in dict.fromkeys(queries):
                if self.left() < 7:
                    break
                tried.append(query)
                ids = self.search_words(query)
                rows = [i for i in ids if i not in self.opened and not self.sold_out(i)]
                if not rows and self.next_page and self.left() >= 8:
                    ids = self.search_words(query, self.next_page)
                    rows = [i for i in ids if i not in self.opened and not self.sold_out(i)]
                if rows and self.open_rows(rows[:3]):
                    opened = True
                    break
        self.trace.append({"kind": "decline_held", "searched": True, "opened": opened, "queries": tried})
        return opened

    def chase(self, about: str) -> bool:
        """k2: find and open the listing a chain link describes ("the Strut from Ratorno (make isuzu, ...)"): search its
        shop or brand name with its kind, keep the rows that name that shop or brand, open the one sharing most words
        with the description. True when a page was opened."""
        names = [w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", about) if w not in CHASE_STARTERS]
        known = {str(v).lower() for r in self.rows.values() if isinstance(r, dict) for k, v in r.items()
                 if isinstance(v, str) and re.search(r"brand|vendor|maker|seller|shop|merchant", k, re.I)}
        names.sort(key=lambda w: w.lower() not in known)
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not names or not tool or not target:
            return False
        about_words = set(re.findall(r"[a-z0-9]+", about.lower()))
        for name in names[:2]:
            rest = [w for w in re.findall(r"[a-z]+", re.sub(name, " ", about).lower()) if len(w) > 3
                    and w not in ("from", "with", "page", "listing", "product", "item", "this", "that", "points", "look")]
            for query in (f"{name} {' '.join(rest[:2])}", name):
                ids = self.search_words(query)
                # k3: a page already opened was read by the chain check and is not the link; a near twin of the same
                # shop is, so only unopened rows are candidates
                hits = [i for i in ids if name.lower() in json.dumps(self.rows.get(i) or {}).lower() and i not in self.opened]
                if not hits:
                    continue
                best = max(hits, key=lambda i: len(about_words & set(re.findall(r"[a-z0-9]+", json.dumps(self.rows.get(i) or {}).lower()))))
                faults: list[str] = []
                args = _shape({target: best}, self.schemas.get(tool), "args", faults)
                if faults:
                    return False
                reply = self.act(tool, args)
                if reply is None:
                    return False
                self.observe(tool, reply.get("observation"))
                self.trace.append({"kind": "chase", "about": about[:160], "query": query, "opened": best})
                return True
        self.trace.append({"kind": "chase", "about": about[:160], "found": None})
        return False

    def offer_tool(self) -> str | None:
        """Tool for re-reading a listing's live offer: the offer view when the store has one, else the single-listing view.
        k16: when that view has shown no stock, the other of the two, else another single-listing view of the store
        (not a cart, order, message or search tool) that has not shown itself without stock."""
        tool = self.roles.get("offer") or self.roles.get("open")
        if not tool or self.stock_from.get(tool) is not False:
            return tool
        busy = {self.roles.get(r) for r in ("search", "cart", "cart_view", "buy", "refuse", "ask", "compare")} - {None}
        for other in [self.roles.get("open"), self.roles.get("offer")] + sorted(self.schemas):
            if (other and other != tool and other not in busy and self.single_target(other) and self.stock_from.get(other) is not False
                    and (other in (self.roles.get("open"), self.roles.get("offer")) or self.reader_like(other))):
                return other
        return tool

    def reread_offers(self) -> str:
        """Re-read each cart line with the store's offer view just before checkout: stock and price can change unannounced."""
        tool = self.offer_tool()
        target = self.single_target(tool)
        if not tool or not target:
            return ""
        since = max(self.step - 2, self.refused_at)
        # k16: after a sell-out remark that named no listing, a line whose stock no record has shown since is stale too
        stale = [i for i in self.cart_ids() if (self.offer_seen.get(i, -99) < since or 0 <= self.doubt_at and self.stock_from.get(tool) is not False
                                                and self.stock_seen.get(i, -99) < self.doubt_at) and i not in self.gone]
        if not stale or self.left() < len(stale) + 3 or (SWAP_RESERVE and self.swap_room() and self.left() - len(stale) <= self.tail()):
            # k17m7: the re-reads would eat a cheapest pick's swap reserve
            return ""
        doubt = self.doubt_at
        self.doubt_at = -1
        lines = []
        for ident in stale:
            if not self.may_open(1, 0):
                break
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
        if doubt >= 0:
            # k16: a line the re-read shows unbuyable (a stock key new to its record included) is sold out
            for ident in stale:
                if self.stock_seen.get(ident, -99) >= doubt and not self.buyable(ident):
                    self.unavailable.add(ident)
        self.trace.append({"kind": "reread", "ids": stale, "changed": len(lines), "doubt": doubt, "tool": tool})
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
        if SOLD_TWIN and not picks:
            # k17m8 (SOLD_TWIN): no listing read replaces the line: its own name is searched for twins (see the module notes)
            picks = self.sold_twin(lost)
        if SOLD_SEEK and not picks:
            # k17m3 (SOLD_SEEK): no listing read replaces the line: Kyo searches for its values now (see the module notes)
            picks = self.sold_seek(lost)
        text = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get("pick") == "cheapest":
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."

    @_lever("sold_seek", [])
    def sold_seek(self, lost: list[str]) -> list[str]:
        """k17m3 (SOLD_SEEK): a cart line sold out and no listing read so far replaces it. Kyo runs k2's value searches
        now, while steps remain, instead of only before an ending (each plain value with the item's kind noun, one value
        a query, the next page when the first shows nothing new, then the item's name; up to three new rows opened), and
        asks the sold-out review again. In the race-10 and race-11 replays such one-value queries showed 7 of 10 and 8 of
        12 answers agents never saw. The replacements the second review names."""
        if self.sought or self.left() < 12 + 2 * len(self.items) or self.elapsed() > LOOP_SECONDS - 20:
            return []
        for key in list(self.lead_items):
            # the sold-out line no longer leads its item, so its item is searched for
            self.lead_items[key] = [i for i in self.lead_items[key] if i not in lost]
            if not self.lead_items[key]:
                del self.lead_items[key]
        if not self.seek_missing():
            return []
        picks = self.replacements(lost)
        self.trace.append({"kind": "sold_seek", "lost": lost, "picks": picks[:4]})
        return picks

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
        if self.cost > SPEND_SOFT:
            return []
        # k3: a listing of another category cannot replace the line, so in a multi-item task only the sold-out line's
        # own category is shown (when the store's rows carry a category); the newest 20 of those
        cat = self.category_key()
        lost_cats = {str((self.rows.get(i) or {}).get(cat) or "").strip().lower() for i in lost} - {""} if cat else set()
        records = []
        for ident in reversed(self.seen_order):
            if ident in lost or ident in self.gone or self.sold_out(ident):
                continue
            own = str((self.rows.get(ident) or {}).get(cat) or "").strip().lower() if lost_cats else ""
            if own and own not in lost_cats:
                continue
            records.append(_dump(_clip(self.full.get(ident) or self.rows.get(ident), 200))[:1200])
            if len(records) >= 20:
                break
        if not records:
            return []
        prompt = SWAP + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nSOLD-OUT CART LINE:\n" + "\n".join(f"{i}: {self.record_of(i)}" for i in lost) + "\n\nCART NOW:\n" + self.cart_text()
        prompt += "\n\nOTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.judge(prompt, 2500, "swap")
        fits: list[tuple[float, str]] = []
        for entry in _items((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.rows and ident not in lost and ident not in self.gone and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        # sorted here: the model names fits well but orders them by price unreliably. k6: on every task but a most or least
        # pick (see the module notes); a fit whose price the review left out is priced from its record

        def cost(fit: tuple[float, str]) -> float:
            known = self.price_of(fit[1]) if fit[0] == float("inf") else fit[0]
            return known if known is not None else float("inf")

        ranked = [i for _, i in sorted(fits, key=cost)] if self.brief.get("pick") not in ("most", "least") else [i for _, i in fits]
        self.trace.append({"kind": "swap", "lost": lost, "fits": fits})
        # k17m3 (SOLD_BUY): kept for a takeover of a later ending without a purchase
        self.swap_named = ranked + [i for i in self.swap_named if i not in ranked]
        return ranked[:4]

    def audit(self) -> str:
        # k3: the review before an ending without a purchase runs up to the hard spend line, not the soft one. In race 12
        # (about $0.055 a task) declines came late in costly episodes, where the soft line skipped this review; v5's wrong
        # declines on tasks others passed by ordering (3 runs, about 2 orders) include one with the fitting listing among
        # its own unread search rows
        if self.audited or self.cost > self.spend_hard:
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
        if not named and END_AUDIT:
            # k17 (E1): nothing met every requirement: listings with nothing broken that show every requirement word
            soft: dict[str, list[str]] = {}
            for k, v in ((got or {}).get("soft") or {}).items():
                soft.setdefault(self.item_key(k), []).extend(i for i in v if not self.sold_out(i))
            soft = {k: v[:4] for k, v in soft.items() if v}
            if soft:
                self.lead_items = soft
                self.leads = [i for ids in soft.values() for i in ids]
                self.trace.append({"kind": "end_audit_soft", "leads": soft})
                return ("A strict review found no listing whose record states every requirement, but these state nothing against one and"
                        " show every requirement word: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in soft.items())
                        + ". A requirement a page leaves unstated is no reason to end without a purchase: open them if needed and buy the"
                        " best one; end without a purchase only if each one clearly breaks a requirement.")
        if not named:
            return ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review thinks these listings may fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them and buy what truly fits; end without a purchase only if each one fails."

    def recheckout(self, name: str, args: dict[str, Any]) -> str:
        """k12: the checkout was turned down with a note that only moved the budget or dropped requirements, and the cart
        still meets the change: Kyo sends the same checkout again (see the module notes)."""
        kinds = {c.split(" ", 1)[0] for c in self.change_now}
        # k17m (C5, after lllll v1): a refusal over a moved price alone is no reason to change a cart that still fits
        kinds |= self.price_kinds()
        if not kinds or not kinds <= {"budget", "dropped", "price"} or self.note_sold or self.recheckouts >= 1:
            return ""
        if kinds & {"dropped", "price"} and self.brief.get("pick") in ("cheapest", "most", "least"):
            return ""
        if "price" in kinds and self.left() > len(self.cart_ids()) + 3:
            # the lines' prices now: each line re-read (every one is stale since the refusal)
            self.reread_offers()
        ids = self.cart_ids()
        if not ids or self.left() < 2 or any(self.sold_out(i) or i in self.gone for i in ids):
            return ""
        prices = [self.price_of(i) for i in ids]
        if any(p is None for p in prices):
            return ""
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool):
            if self.brief.get("budget_scope") == "each":
                if any(p > self.budget for p in prices):
                    return ""
            elif sum(p * max(1, _count(self.cart_lines.get(i, 1)) or 1) for p, i in zip(prices, ids)) > self.budget:
                return ""
        self.recheckouts += 1
        go, said = self.gate_tell("order")
        if not go:
            return "[Kyo] " + said
        reply = self.act(name, args)
        self.trace.append({"kind": "recheckout", "change": list(self.change_now), "cart": ids})
        if reply is None:
            return ""
        self.buys += 1
        head = ("[Kyo checked out the same cart again: the note only " + " and ".join(sorted(kinds)).replace("budget", "moved the budget")
                .replace("dropped", "dropped requirements").replace("price", "changed a price") + ", and the cart still meets it]")
        if reply.get("finished") or self.finished:
            return head + " [episode finished]"
        self.refused_at = self.step
        observation = reply.get("observation")
        self.observe(name, observation)
        return head + " The store answered: " + self.show(observation)

    def rethink(self) -> str:
        """k2: the store turned down an ending without a purchase and changed the task (a new budget, a requirement
        dropped or added). In races 10 and 11 sending that ending again almost never passed, and agents nearly always
        did: the listings seen are reviewed under the change (opened pages first), and Kyo carts and checks out the
        reviewed lines itself when a fresh check passes them; otherwise the guards before an ending start over."""
        self.rethought += 1
        self.audited, self.sought, self.pushed, self.refuse_blocks, self.late_held = True, False, False, 0, False
        self.leads, self.lead_items, self.confirmed = [], {}, None
        head = "[Kyo] The store did not accept ending without a purchase and changed the task (" + "; ".join(self.change_now) + ")."
        if self.rethought > 2:
            return head + " Look again at the listings you have seen under the changed task and buy what fits."
        named, done = self.review_buy()
        self.trace.append({"kind": "rethink", "change": self.change_now, "named": named, "bought": bool(done)})
        if not named:
            return head + (" No listing seen so far fits under it: search again under the changed task (open the listings the old "
                           "task ruled out) and buy what fits; end without a purchase only after that.")
        if done:
            return head + " " + done
        return head + (" Under the changed task a careful review thinks these listings fit: "
                       + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
                       + ". Put the best fitting one per item in the cart at the requested quantity and check out; end without "
                       "a purchase only if each one clearly fails.")

    def review_buy(self) -> tuple[dict[str, list[str]], str]:
        """k2: review the listings read (opened pages first) and, when the review names one fitting listing per item and
        a fresh check passes them, cart and check out (Kyo acts itself). Returns what the review named and the purchase
        report ("" when Kyo did not buy)."""
        if self.cost > self.spend_hard + 0.02:
            return {}, ""
        named = self.audit_fits(self.audit_call(opened_first=True))
        if not named:
            return {}, ""
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        self.completed = False
        return named, self.buy_leads(None, self.spend_hard + 0.03)

    @_lever('end_review', None)
    def strict_end_review(self) -> dict[str, Any] | None:
        """k17 (E1): the strict review of up to ten candidates for each item the cart lacks: its opened pages (those
        showing the most requirement words first), then rows of its kind carrying every requirement word. Returns the old
        review's shape ("fits": every requirement met, "open": rows with nothing broken) and "soft" (nothing broken and
        every requirement word shown); None when no item has a candidate (the old review runs then)."""
        held = {self.line_item(i) for i in self.cart_ids() if not self.sold_out(i)}
        missing = [n for n, _ in enumerate(self.items, 1) if n not in held] or [n for n, _ in enumerate(self.items, 1)]
        pairs: list[tuple[str, int]] = []
        for number in missing:
            words = self.req_words(number)
            opened = [i for i in reversed(self.seen_order) if i in self.full and self.of_item(i, number) and not self.sold_out(i)]
            opened.sort(key=lambda i: -self.cover(i, words))
            rows = [i for i in reversed(self.seen_order) if i not in self.full and self.of_item(i, number) and not self.sold_out(i)
                    and words and all(_toks(w) <= self.row_words(i) for w in words)]
            pairs += [(i, number) for i in (opened + rows)[:10]]
        if not pairs:
            return None
        marks = self.mark_reqs(pairs, "end_audit")
        fits: dict[str, list[str]] = {}
        soft: dict[str, list[str]] = {}
        rows_open: list[str] = []
        for ident, number in pairs:
            if ident not in marks or not self.buyable(ident) or not self.in_budget(ident):
                continue
            states = [self.mark_of(marks[ident], r) for n, r in self.needs_now() if n == number]
            if all(s == "met" for s in states):
                fits.setdefault(str(number), []).append(ident)
            elif "broken" not in states:
                have = self.page_words(ident) if ident in self.full else self.row_words(ident)
                if ident not in self.full:
                    rows_open.append(ident)
                if all(_toks(w) <= have for w in self.req_words(number)):
                    soft.setdefault(str(number), []).append(ident)
        return {"fits": fits, "open": rows_open[:6], "soft": soft, "reason": "strict review, requirement by requirement"}

    def audit_call(self, opened_first: bool = False) -> dict[str, Any] | None:
        if END_AUDIT:
            # k17 (E1): the strict review in its place (see the module notes); the old one when it has no candidate
            got = self.strict_end_review()
            if got is not None:
                return got
        records = []
        order = list(reversed(self.seen_order))
        if opened_first:
            # k2: every opened page before the rows, so a listing opened early is not cut by the 30-record limit
            order = [i for i in order if i in self.full] + [i for i in order if i not in self.full]
        for ident in order:
            if self.sold_out(ident):
                continue
            record = self.full.get(ident) if ident in self.full else None
            records.append(("OPENED " + _dump(_clip(record, 200))[:1200]) if record else ("ROW " + _dump(_clip(self.rows.get(ident), 120))[:500]))
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS IN FORCE:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(records)
        got = self.judge(prompt, 1500, "audit")
        self.trace.append({"kind": "audit", "got": got})
        return got

    def item_key(self, key: Any) -> str:
        """k7: the item number a review keyed its fits with (see the module notes); the key as given when it says none."""
        text = str(key).strip()
        found = re.findall(r"\d+", text)
        if len(found) == 1 and 1 <= int(found[0]) <= max(1, len(self.items)):
            return str(int(found[0]))
        if not found and len(self.items) <= 1:
            return "1"
        if not found:
            low = text.lower()
            hits = [n for n, item in enumerate(self.items, 1) if item.name and item.name.lower() in low]
            if len(hits) == 1:
                return str(hits[0])
        return text

    def audit_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if isinstance(fits, list) and len(self.items) <= 1:
            fits = {"1": fits}
        if not isinstance(fits, dict):
            return {}
        named: dict[str, list[str]] = {}
        for k, v in fits.items():
            number = self.item_key(k)
            ids = named.setdefault(number, [])
            ids += [str(i) for i in _items(v) if str(i) in self.rows and not self.sold_out(str(i)) and str(i) not in ids]
        named = {k: v[:4] for k, v in named.items()}
        if self.brief.get("pick") not in ("most", "least"):
            # k6: the cheapest named fit first, since the first one is bought (the review is asked for that order too)
            def cost(ident: str) -> float:
                known = self.price_of(ident)
                return known if known is not None else float("inf")

            named = {k: sorted(v, key=cost) for k, v in named.items()}
        return {k: v for k, v in named.items() if v}

    def open_rows(self, rows: list[str]) -> bool:
        """Open listings seen only as search rows: side by side in one step when the store can, else the first two one by one."""
        many, joiner = self.compare_target(), self.roles.get("compare")
        tool = self.facts_tool()
        target = self.single_target(tool)
        if many and joiner and not self.compare_bare():
            plan = [(joiner, {many: rows[: self.compare_room()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        wanted = sum(len(v) if isinstance(v, list) else 1 for _, a in plan for v in a.values())
        if self.left() < len(plan) + 5 or not self.may_open(wanted):
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
                calls.append({target: ident, mode: self.mode_values(props[mode]["enum"])[1], **({count: have} if count else {})})
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
            calls.append({target: ident, **({mode: self.mode_values(props[mode]["enum"])[0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def complete_cart(self) -> str:
        """The agent ends without buying although the independent check confirmed a reviewed listing: Kyo carts the reviewed
        lines itself (one per item, keeping cart lines of the other items) and checks out, when a fresh check passes that cart."""
        if self.completed or not self.confirmed or self.sold_out(self.confirmed):
            return ""
        return self.buy_leads(self.confirmed, self.spend_hard)

    def lever_fail(self, lever: str, exc: BaseException, where: str) -> None:
        """k17m2: a lever's code raised: the lever is off for the rest of the episode and the path without it runs."""
        if not LEVER_GUARD:
            raise exc
        self.levers_off.add(lever)
        self.trace.append({"kind": "lever_error", "lever": lever, "where": where, "error": f"{type(exc).__name__}: {exc}"[:300]})

    def interview_salvage(self) -> str:
        """k17m2: the interview stopped on an error: what it asked so far, for the agent to read itself. The interview
        counts as held (it is not started again before every cart call)."""
        self.asked = True
        if not self.iv_sent:
            return ""
        told = [(f"item {n}: " if len(self.items) > 1 else "") + need for n, item in enumerate(self.items, 1) for need in item.told]
        return ("Kyo's interviewer asked the shopper (each question used one step):\n"
                + "\n".join(f"Q: {q}\nA: {a}" for _, q, a in self.iv_sent)
                + "\nRequirements from the answers so far: " + ("; ".join(told) if told else
                                                                 "read the answers above yourself and apply each value they want") + ".")

    def compare_room_k16(self) -> int:
        """k17m2: the side-by-side size as k16b chose it (the fallback of A5's sizing)."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    @_lever('spare', lambda self: [LIGHT, SPARE])
    def loop_models(self) -> list[str]:
        """k17m (C4, after Secretarybird v1): after STUCK_SPARE turns in a row that used no store step the spare model leads."""
        if STUCK_SPARE and self.stuck >= STUCK_SPARE:
            self.trace.append({"kind": "stuck_spare", "turns": self.stuck})
            return [SPARE, LIGHT]
        return [LIGHT, SPARE]

    @_lever('price_again', set())
    def price_kinds(self) -> set[str]:
        """k17m (C5, after lllll v1): "price" when the last note said only that a price moved."""
        return {"price"} if PRICE_AGAIN and self.price_moved_now else set()

    def model_down(self) -> bool:
        """k17m (C1): the models cannot be reached: no credit (402), or three loop turns without a reply."""
        return self.no_credit or self.misses > 2

    @_lever('code_end', {})
    def guess_roles(self) -> dict[str, str]:
        """k17m (C1, after lllll v1's guess_roles): each job's tool from the words of the tools' names, the first match
        per job, in the order refuse, cart view, cart, checkout, message (one free-text argument), details, search."""
        def words(name: str) -> set[str]:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
            return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}

        def one_text(name: str) -> bool:
            props = (self.schemas.get(name) or {}).get("properties") or {}
            return len([k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]) == 1

        guessed: dict[str, str] = {}
        taken: set[str] = set()
        for job, test in ROLE_TESTS:
            for tool in self.tools:
                name = str((tool.get("function") or {}).get("name") or "")
                if name and name not in taken and test(words(name)) and (job != "ask" or one_text(name)):
                    guessed[job] = name
                    taken.add(name)
                    break
        return guessed

    @_lever('code_end', lambda self, tool, given, text, buying: dict(given))
    def plain_args(self, tool: str, given: dict[str, Any], text: str, buying: bool) -> dict[str, Any]:
        """k17m (C1): the fixed arguments, and each other required one filled in code: a choice gets the option naming
        that ending (else the first), a text `text`, a flag true, a count 1."""
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        args = dict(given)
        want = BUY_OPTION if buying else END_OPTION
        for key in schema.get("required") or []:
            if key in args:
                continue
            spec = props.get(key) or {}
            kinds = _kinds(spec)
            if isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = next((o for o in spec["enum"] if want.search(str(o))), spec["enum"][0])
            elif "string" in kinds:
                args[key] = text
            elif "boolean" in kinds:
                args[key] = True
            elif "integer" in kinds or "number" in kinds:
                args[key] = 1
        return args

    @_lever('code_end', [])
    def reread_unheard(self) -> list[str]:
        """k17m (C1): the answers no model could read are read again; the requirements they add (none, and they stay
        kept, when the reader fails again)."""
        if not self.unheard or self.model_down():
            return []
        heard = self.hear(self.unheard, (LIGHT, HEAVY))
        if self.hear_failed:
            return []
        told: list[str] = []
        self.take_heard(heard, told)
        self.trace.append({"kind": "reheard", "n": len(self.unheard), "told": told})
        self.unheard = []
        if told:
            self.version += 1
        return told

    def code_words(self, number: int) -> list[str]:
        """k17m (C1): the words a listing for the item should show: its requirement values, the words of answers no model
        read, and with neither (no brief) the request's own longer words."""
        words = list(self.req_words(number))
        for asked_item, _, answer in self.unheard:
            if asked_item == number or len(self.items) == 1:
                words += [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) >= 4 and w not in COVER_SKIP and w not in words]
        if not words and self.brief_failed:
            words = [w for w in dict.fromkeys(re.findall(r"[a-z]{4,}", self.request.lower())) if w not in COVER_SKIP][:12]
        return words

    def code_mine(self, ident: str, number: int) -> bool:
        """k17m (C1, C3): the listing is for the item: its category goes with the item, else (no category, or one going
        with no item) the item's kind noun is among the row's words."""
        if self.of_item(ident, number):
            return True
        if any(self.of_item(ident, m) for m in range(1, len(self.items) + 1)):
            return False
        if len(self.items) == 1 and self.brief_failed:
            # no brief: the one item is only "the requested product", and every listing seen is a candidate
            return True
        noun = _kind_noun(self.items[number - 1].shelf or self.items[number - 1].name)
        return bool(noun) and _toks(noun) <= self.row_words(ident)

    def code_pool(self, number: int, avoid: set[str], taken: set[str]) -> list[str]:
        # k17m3: a listing a strict review marked broken on a live requirement of the item is no candidate
        return [i for i in self.seen_order if i in self.rows and i not in avoid and i not in taken and i not in self.gone
                and i not in self.unavailable and self.code_mine(i, number) and self.buyable(i) and self.in_budget(i)
                and self.review_mark(i, number) != "broken"]

    def review_mark(self, ident: str, number: int) -> str:
        """k17m3: what the strict reviews last said of the listing for the item's live requirements: "broken" when one is
        marked broken, "met" when all are marked met, else "" (not reviewed, or some unstated)."""
        marks = self.marks_seen.get(ident)
        if not marks:
            return ""
        said = [self.mark_of(marks, need) for n, need in self.needs_now() if n == number]
        if "broken" in said:
            return "broken"
        return "met" if said and all(s == "met" for s in said) else ""

    def code_picks(self, avoid: set[str], prefer: list[str] | None = None) -> dict[str, int]:
        """k17m (C1, after Secretarybird v1's code_picks): one listing per item: of the seen listings of the item that
        can be bought inside the budget (the item's kind noun searched when there are none), the one showing the most
        of `code_words` on its row and page, opened pages first, then the cheapest (on a cheapest pick the cheapest
        before opened pages); the best two unopened ones are opened first when steps allow. When a total budget cannot
        hold the picks, the cheapest of each item; empty when even that fails or an item has none."""
        keep = len(self.items) + 3
        plan: dict[str, int] = {}
        pools: dict[int, list[str]] = {}
        prefer_rank = {ident: n for n, ident in enumerate(prefer or [])}   # k17m3: a review's names first, in its order
        for number, item in enumerate(self.items, 1):
            pool = self.code_pool(number, avoid, set(plan))
            if not pool and self.roles.get("search") and self.left() > keep + 4:
                noun = _kind_noun(item.shelf or item.name) or item.name
                extra = [w for w in self.code_words(number) if w != noun][:2]
                self.search_words(" ".join(extra + [noun]))
                pool = self.code_pool(number, avoid, set(plan))
                if not pool and extra and self.left() > keep + 3:
                    self.search_words(noun)
                    pool = self.code_pool(number, avoid, set(plan))
            if not pool:
                return {}
            words = self.code_words(number)

            def rank(ident: str) -> tuple[Any, ...]:
                have = self.row_words(ident) | self.page_words(ident)
                hits = sum(1 for w in words if _toks(w) and _toks(w) <= have)
                first = prefer_rank[ident] if ident in prefer_rank else len(prefer_rank)
                unmet = self.review_mark(ident, number) != "met"     # k17m3: a listing a strict review found fitting next
                if self.brief.get("pick") == "cheapest":
                    return (first, unmet, -hits, self.dearness(ident))
                return (first, unmet, -hits, ident not in self.full, self.dearness(ident))
            pool.sort(key=rank)
            unread = [i for i in pool[:2] if i not in self.full]
            if unread and self.left() > keep + len(unread) and self.may_open(len(unread), 0):
                self.fill_facts(unread, keep)
                pool = [i for i in pool if self.buyable(i) and self.in_budget(i)]
                pool.sort(key=rank)
            if not pool:
                return {}
            pools[number] = pool
            plan[pool[0]] = max(1, item.qty)
        if isinstance(self.budget, (int, float)) and not isinstance(self.budget, bool) and self.brief.get("budget_scope") != "each":
            def total(lines: dict[str, int]) -> float:
                return sum(self.dearness(i) * q for i, q in lines.items() if self.price_of(i) is not None)
            if total(plan) > self.budget:
                plan = {min(pools[n], key=self.dearness): max(1, item.qty) for n, item in enumerate(self.items, 1)}
                if total(plan) > self.budget:
                    return {}
        return plan

    @_lever('code_end', "")
    def code_buy(self, prefer: list[str] | None = None, why: str = "models gone") -> str:
        """k17m (C1, after Secretarybird v1's code finish): the models are gone, so Kyo finishes in code: the cart is
        set to `code_picks` and checked out; after a refusal the lines are re-read (a lone line named sold out is that
        listing), sold-out lines are swapped for the next listing, and a cart refused twice as it was is left; four
        carts at most. "bought" when the episode ended with the checkout."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if self.finished or self.broke or not cart or not buy or self.left() < 3 or self.elapsed() > RESCUE_SECONDS + 20:
            return ""
        self.trace.append({"kind": "code_end", "why": why, "no_credit": self.no_credit, "misses": self.misses, "brief": not self.brief_failed,
                           "cart": self.cart_ids(), "rows": len(self.rows), "opened": len(self.full)})
        avoid = set(self.gone) | set(self.unavailable)
        last: dict[str, int] = {}
        repeats = 0
        for _ in range(CODE_END_ROUNDS):
            if self.finished or self.left() < 2:
                break
            plan = self.code_picks(avoid, prefer)
            if not plan:
                break
            repeats = repeats + 1 if plan == last else 0
            if repeats >= 2:
                avoid |= set(plan)
                continue
            last = dict(plan)
            now = self.cart_ids()
            if set(now) != set(plan) or any(self.cart_lines.get(i) != q for i, q in plan.items()):
                calls = self.cart_calls(plan, [i for i in now if i not in plan])
                if not calls or len(calls) + 1 > self.left():
                    break
                for args in calls:
                    faults: list[str] = []
                    args = _shape(args, self.schemas.get(cart), "args", faults)
                    if faults:
                        return ""
                    reply = self.act(cart, args)
                    if reply is None or self.finished:
                        return ""
                    if CART_SYNC:
                        self.book_cart(cart, args, reply.get("observation"))
                    else:
                        self.note_cart(cart, args)
                    self.observe(cart, reply.get("observation"))
            faults = []
            args = _shape(self.plain_args(buy, dict(self.brief.get("buy_args") or {}), "Please place the order for the cart as it stands.", True),
                          self.schemas.get(buy), "args", faults)
            if faults or self.left() < 1:
                break
            if not self.model_down() and self.left() >= 3:
                # k17m3: with the models up, the request's steps at the order come first (a message, a page)
                go, _ = self.gate_tell("order")
                if not go or self.finished or self.left() < 1:
                    break
            reply = self.act(buy, args)
            self.trace.append({"kind": "code_buy", "plan": dict(plan), "reply": _clip(reply, 160)})
            if reply is None:
                break
            self.buys += 1
            if self.finished or reply.get("finished"):
                return "bought"
            self.refused_at = self.step
            seen = reply.get("observation")
            self.observe(buy, seen)
            if len(plan) == 1 and SOLD_NOTE.search(self.spoken(seen)):
                self.unavailable.add(next(iter(plan)))
            elif self.left() > len(plan) + 2:
                self.reread_offers()
            avoid |= {i for i in plan if self.sold_out(i) or i in self.unavailable or i in self.gone}
        return ""

    @_lever("sold_buy", "")
    def sold_buy(self) -> str:
        """k17m3 (SOLD_BUY): a cart line sold out in this episode and the agent ends without a purchase (or the forced
        finish has an empty cart). No such ending passed in race 11 (0 of 39) nor in this line's practice runs (0 of 30).
        Kyo buys once: per item the replacements the sold-out reviews named, still buyable inside the budget, in their
        order, else the code finish's pick; a line that sells out again is swapped for the next. "bought" when the
        episode ended with the checkout."""
        if (not SOLD_BUY or self.sold_bought or self.finished or self.broke or not self.gone
                or self.left() < len(self.items) + 2 or self.elapsed() > RESCUE_SECONDS + 20):
            return ""
        self.sold_bought = True
        prefer = [i for i in self.swap_named if i not in self.gone and i not in self.unavailable and not self.sold_out(i)]
        done = self.code_buy(prefer, "sold out, then an ending without a purchase")
        self.trace.append({"kind": "sold_buy", "prefer": prefer[:6], "bought": done == "bought", "cart": self.cart_ids()})
        return done

    def rescue(self) -> None:
        """The loop stopped on its limits (calls, spend, steps or time) with nothing in the cart. An ending without a
        purchase there is rarely right, so before it a review names fitting listings among those seen (when no review
        has yet), and Kyo carts and checks out one per item when a fresh check passes that cart."""
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_ids()) if fresh else bool(self.cart_lines)
        if (self.finished or self.broke or self.no_credit or self.completed or holding or self.left() < 4
                or self.elapsed() > RESCUE_SECONDS or not self.roles.get("cart") or not self.roles.get("buy")):
            return
        self.call_cap = ALL_CALLS + RESCUE_CALLS
        if not self.lead_items:
            named = self.audit_fits(self.audit_call())
            if named:
                self.lead_items = named
                self.leads = [i for ids in named.values() for i in ids]
        self.trace.append({"kind": "rescue", "leads": _clip(self.lead_items, 80)})
        if self.lead_items:
            self.buy_leads(None, self.spend_hard + 0.02)

    def buy_leads(self, require: str | None, limit: float) -> str:
        """Cart the reviewed lines (one per item, keeping cart lines of the other items; `require` must be among them) and
        check out, when a fresh check passes that cart."""
        cart, buy = self.roles.get("cart"), self.roles.get("buy")
        if not cart or not buy:
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
        if not plan or (require and require not in plan):
            return ""
        now = self.cart_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and not self.sold_out(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ""
        lines = {**keep, **plan}
        if drop and self.removal_rule():
            # k9: taking a line out needs the shopper's word first: the loop does it through the guarded cart calls
            return ""
        calls = self.cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.left():
            return ""
        # acting costs no model call; a cart the checks have not passed yet gets one check, allowed up to `limit`
        self.fill_facts(list(lines), len(calls) + 3)
        self.check_cart(list(lines), limit)
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
            if CART_SYNC:
                self.book_cart(cart, args, reply.get("observation"))
            else:
                self.note_cart(cart, args)
            self.observe(cart, reply.get("observation"))
            done.append(_dump(args))
        faults = []
        args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        # a refused checkout (a price, budget or stock change) gets one retry after re-reading the lines and a fresh check
        for attempt in range(2):
            if attempt:
                self.reread_offers()
                self.check_cart(list(lines), limit)
                if (not self.verdicts.get(self.check_key(list(lines))) or self.left() < 1 or self.finished
                        or any(self.sold_out(i) for i in lines)):
                    break
            go, said = self.gate_tell("order")
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
            self.note_sold = False
            self.observe(buy, reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if self.finished or reply.get("finished"):
                break
            # k2: a sell-out refusal marks the sold line (a lone line is that listing, several are re-read), so the retry
            # below does not check out the same listing again (a RaceSim race-11 episode did, and then ended without a purchase)
            if self.note_sold or any(line in self.spoken(reply.get("observation")) for line in self.sold_lines):
                self.find_sold()
                hint = self.swap_hint()
                if hint:
                    replies.append(hint)
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + (require or ", ".join(lines)) + ", so Kyo set the cart ("
                + "; ".join(done) + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------
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
        ends = ({self.roles.get("buy"), self.roles.get("refuse")} | self.follow_ups) - {None}
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
        """A listing given by its title where a tool takes listing ids: use the id when one seen row has that title,
        otherwise say what is wrong, so no step is spent on a call the store cannot match."""
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
        """A cart tool whose count may be exactly 0 and that has no mode for taking lines off sets a line's quantity: 0 removes it."""
        props = self.schemas.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _kinds(s) and s.get("minimum") == 0 for s in props.values())

    @staticmethod
    def takes_off(options: list[Any], value: Any) -> bool:
        """k10: a cart mode value that takes a line out, read by its words; by position (any but the first) only when no
        value of the list says either way."""
        worded = [o for o in options if TAKE_OFF.search(str(o))]
        if worded and len(worded) < len(options):
            return value in worded
        return value in options[1:]

    @classmethod
    def mode_values(cls, options: list[Any]) -> tuple[Any, Any]:
        """k10: (the value that adds a line, the value that takes one out) of a cart tool's mode enum."""
        off = next((o for o in options if cls.takes_off(options, o)), options[1])
        on = next((o for o in options if not cls.takes_off(options, o)), options[0])
        return on, off

    def removal_rule(self) -> dict[str, Any] | None:
        """k9: a step the request asks for before anything is taken out of the cart or swapped, taking a message."""
        return next((p for p in self.process if p["when"] == "choose" and p["messages"] and REMOVAL.search(p["step"])), None)

    def lowered_by(self, name: str, args: dict[str, Any]) -> list[str]:
        """k9: cart lines this cart call would lower or drop (read the way note_cart reads it, on a copy)."""
        saved = (dict(self.cart_lines), self.cart_touched, self.cart_at)
        before = dict(self.cart_lines)
        self.note_cart(name, args)
        after = dict(self.cart_lines)
        self.cart_lines, self.cart_touched, self.cart_at = saved
        return [i for i, n in before.items() if after.get(i, 0) < n]

    def ask_before_removal(self, lowered: list[str]) -> str:
        """k9: the request asks to be asked before a line comes out of the cart. Returns why the cart call is held ("" to
        go on); a message goes out only when none did since the cart last changed."""
        sent_at = [int(m.group(1)) for line in self.done_log for m in [re.match(r"step (\d+): message to the shopper", line)] if m]
        if any(s > self.cart_at for s in sent_at):
            return ""
        names = "; ".join(f"{self.short_name(i)} ({i})" for i in lowered)
        if self.msg_used >= self.msg_limit or self.left() < 3 or not self.roles.get("ask"):
            return (f"Not run (no step used): the request asks you to ask the shopper before anything comes out of the cart, and no "
                    f"message is left to ask with, so {names} stays in the cart. Add what else the order needs without taking it out.")
        text = f"Before I change the cart: may I take {names} out of it?"
        batch = self.send([(1, "", text)], 2)
        if not batch:
            return "Not run (no step used): the message to the shopper did not go through, so the cart stays as it is."
        told: list[str] = []
        heard = self.hear(batch, (LIGHT, HEAVY))
        self.take_heard(heard, told)
        self.trace.append({"kind": "removal_ask", "lines": lowered, "reply": batch[0][2][:200], "told": told})
        if told:
            self.version += 1
            return (f"Not run yet (no step used). Kyo asked first: Q: {text} A: {batch[0][2]}\nThe answer adds: " + "; ".join(told)
                    + ". Apply it, then change the cart.")
        return ""

    @_lever('rearm', None)
    def rearm(self) -> None:
        """From love v20 (public as the overall #1): a change (new budget, requirement added or dropped, a refused
        finish) re-opens the one-shot checks so they run again on the new task: the decline review, the search-more
        push, the refuse blocks, the late hold and the cheapest widening; k17j adds its own (the coverage hold, the
        decline guards and the swap holds)."""
        self.audited = False
        self.pushed = False
        self.refuse_blocks = 0
        self.late_held = False
        self.wider_held = False
        self.cover_done = False
        self.empty_tried = False
        self.full_held = False
        self.swap_held = set()
        self.trace.append({"kind": "rearm", "step": self.step})

    @_lever('decline_guard', "")
    def decline_guard(self) -> str:
        """k17j, before an ending without a purchase. EMPTY_GUARD: with an empty cart, the strict review of the pages
        read; when listings meet every requirement, Kyo carts and buys them after a fresh check. FULL_GUARD: with a
        line for every requested item, the strict review of those lines; when it finds nothing broken, the ending is
        held once (a requirement no page can state is no reason to end without a purchase). "" to go on."""
        if self.finished or self.left() <= 5:
            return ""
        cart = [line for line in self.cart_ids() if line not in self.gone and not self.sold_out(line)]
        if EMPTY_GUARD and not cart and not self.empty_tried and self.cost < self.spend_hard:
            self.empty_tried = True
            named = self.audit_fits(self.audit_call(opened_first=True))
            self.audited = True
            self.trace.append({"kind": "empty_guard", "named": _clip(named, 60)})
            if named:
                self.lead_items, self.leads = named, [i for ids in named.values() for i in ids]
                done = self.buy_leads(None, self.spend_hard + 0.02)
                if done:
                    return done
            return ""
        lines = {self.line_item(i) for i in cart}
        if (FULL_GUARD and cart and not self.full_held and all(n in lines for n, _ in enumerate(self.items, 1))
                and self.cost < SPEND_SOFT):
            self.full_held = True
            marks = self.mark_reqs([(i, self.line_item(i)) for i in cart], "full_guard")
            broken = [i for i in cart if any(self.mark_of(marks.get(i) or {}, r) == "broken" for n, r in self.needs_now() if n == self.line_item(i))]
            if marks and not broken:
                return ("Not run yet (no step used). Every requested item has a line in the cart, and a strict review of their records "
                        "finds no requirement broken. A requirement no page can state (delivery, service terms and the like) is no "
                        "reason to end without a purchase: check out the cart as it is.")
        return ""

    @_lever('swap_once', "")
    def swap_hold(self, name: str, args: dict[str, Any]) -> str:
        """k17j (SWAP_ONCE): a cart call that takes out an item's line when that item's line was already changed once,
        while the line is still buyable, inside the budget and its last check found nothing wrong, is held once."""
        lowered = self.lowered_by(name, args)
        free = [i for i in lowered if not self.sold_out(i) and self.buyable(i) and self.in_budget(i)
                and self.verdicts.get(self.check_key(self.cart_ids())) is not False]
        items = {self.line_item(i) for i in free} - {0}
        again = sorted(n for n in items if self.swaps.get(n, 0) >= 1 and n not in self.swap_held)
        if again:
            self.swap_held.update(again)
            self.trace.append({"kind": "swap_hold", "items": again, "lines": free})
            return ("Not run (no step used): " + ", ".join(f"item {n}" for n in again) + " already had its line changed once, and "
                    "nothing shows the line in the cart breaking a requirement. Keep it, unless its record clearly breaks one "
                    "(then say which in your text and send the call again).")
        for n in items:
            self.swaps[n] = self.swaps.get(n, 0) + 1
        return ""

    @_lever('check_depts', 0)
    def label_fit(self, label: str, item: Item) -> int:
        """From love v20 (public): how well a store category label fits a requested item: 3 the label is the item's
        shelf or name, 2 every word of the label is among the item's words (a short label such as an abbreviation), 1
        they share a word of four letters or more, 0 not at all."""
        mine = _stem(item.shelf or "") | _stem(item.name)
        theirs = _stem(label)
        if not theirs:
            return 0
        if _words(label) in (_words(item.shelf or ""), _words(item.name)):
            return 3
        if theirs <= mine:
            return 2
        if {w for w in theirs if len(w) >= 4} & {w for w in mine if len(w) >= 4}:
            return 1
        return 0

    @_lever('check_depts', "")
    def dept_lines(self, ids: list[str]) -> str:
        """From love v20 (public): each cart line's store category, and per item the categories seen whose names share
        the item's words: a neighbouring category (a near kind of product on another shelf) shows up next to the right
        one."""
        cat = self.category_key()
        if not cat:
            return ""
        labels: list[str] = []
        for ident in self.seen_order:
            label = str((self.rows.get(ident) or {}).get(cat) or "").strip()
            if label and label not in labels:
                labels.append(label)
        out = []
        for ident in ids:
            label = str((self.rows.get(ident) or {}).get(cat) or "").strip()
            if label:
                out.append(f"{ident}: category {label!r}")
        for number, item in enumerate(self.items, 1):
            near = sorted((label for label in labels if self.label_fit(label, item)), key=lambda label: -self.label_fit(label, item))
            if near:
                out.append(f"item {number} ({item.name}): categories seen whose names share its words: " + ", ".join(repr(l) for l in near[:4]))
        return "\n".join(out)

    @_lever('cart_sync', "failed")
    def sync_cart(self, observation: Any, name: str | None = None, args: dict[str, Any] | None = None) -> str:
        """From love v20 (public): read the cart from a cart tool's own reply when it lists the cart's lines (listing
        records, each with one small count): "synced" when that list was read and is now Kyo's record, "unreadable"
        when such a list is there but its counts cannot be told apart from other small numbers, "none" when the reply
        carries no such list."""
        lists: list[list[Any]] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(node, list):
                if all(isinstance(r, dict) for r in node):
                    lists.append(node)
                return
            if isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)

        if not isinstance(observation, dict) or self.ident(observation):
            return "none"
        walk(observation, 0)
        touched = [v for value in (args or {}).values() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str) and v in self.rows]
        with_ids = [rows for rows in lists if rows and any(self.ident(r, True) for r in rows)]
        rows = next((r for r in with_ids if any(self.ident(x, True) in touched for x in r)), with_ids[0] if with_ids else None)
        if rows is None:
            if len(lists) == 1 and not lists[0]:
                rows = lists[0]
            else:
                return "none"
        idents = [self.ident(r, True) for r in rows]
        if not all(idents):
            return "unreadable"
        key = self.count_key
        if not key and rows:
            keys = [k for k in rows[0] if k != self.price_key
                    and all(isinstance(r.get(k), int) and not isinstance(r.get(k), bool) and 0 <= r[k] <= 99 for r in rows)]
            if len(keys) > 1 and name and args is not None:
                # several small numbers on each line: the one that matches Kyo's own arithmetic for the changed line
                saved = dict(self.cart_lines)
                self.note_cart(name, args)
                expected = dict(self.cart_lines)
                self.cart_lines = saved
                keys = [k for k in keys if all(r.get(k) == expected.get(i, 0) for r, i in zip(rows, idents) if i in touched)]
            if len(keys) != 1:
                return "unreadable"
            key = keys[0]
            self.count_key = key
        out: dict[str, int] = {}
        for row, ident in zip(rows, idents):
            count = row.get(key) if key else None
            if not isinstance(count, int) or isinstance(count, bool):
                return "unreadable"
            if count > 0:
                out[ident] = count
        self.cart_lines = out
        self.cart_view = observation
        self.cart_seen_at = self.step
        self.cart_shows_lines = True
        return "synced"

    @_lever('cart_sync', lambda self, name, args, observation: self.note_cart(name, args))
    def book_cart(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """From love v20 (public): record a cart call from the reply's own line list when it can be read; a store that
        lists the cart after every change it made changed nothing when a reply has no list of listings at all (a
        refusal); otherwise (no list, or one that cannot be read) Kyo's own arithmetic (note_cart)."""
        self.cart_touched = True
        self.cart_at = self.step
        if HEALTH:
            self.own_cart_call(name, args)
        state = self.sync_cart(observation, name, args)
        if state == "synced":
            self.trace.append({"kind": "cart_sync", "lines": dict(self.cart_lines)})
            return
        if state == "none" and self.cart_shows_lines:
            self.trace.append({"kind": "cart_unchanged", "args": _clip(args, 60)})
            return
        self.note_cart(name, args)

    @_lever('health', None)
    def health(self, name: str, observation: Any) -> None:
        """k17m10 (HEALTH): count the reading tools' answers that carry text (no error, no failed call); after BLIND_AFTER
        of them with no listing record read, Kyo is blind (see the module notes). The loop model is told at its next turn."""
        readers = {self.roles.get(r) for r in ("search", "open", "offer", "compare")} - {None}
        if self.blind or name not in readers:
            return
        if isinstance(observation, dict) and (observation.get("error") or observation.get("call_failure")):
            return
        if len(_dump(observation)) < 120:
            return
        self.reader_answers += 1
        if self.rows or self.reader_answers < BLIND_AFTER:
            return
        self.blind = True
        self.trace.append({"kind": "blind_store", "answers": self.reader_answers, "step": self.step})

    @_lever('health', None)
    def own_cart_call(self, name: str, args: dict[str, Any]) -> None:
        """k17m10 (HEALTH): Kyo's own record of the cart by its cart calls, whatever the ids look like (note_cart keeps
        only listings it has read): a list of line objects sets the whole cart; otherwise the call's listing id is added
        (or set) with its count, or taken off when the call removes it or sets 0."""
        lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
        if lists:
            cart: dict[str, int] = {}
            for line in lists[0]:
                ident = _arg_id(line)
                count = next((_count(v) for v in line.values() if _count(v) is not None and not isinstance(v, str)), 1)
                if ident and (count is None or count > 0):
                    cart[ident] = count or 1
            self.own_cart = cart
            return
        ident = _arg_id(args)
        if not ident:
            return
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = any(self.takes_off((spec.get(k) or {}).get("enum"), v) for k, v in args.items()
                      if isinstance((spec.get(k) or {}).get("enum"), list) and len(spec[k]["enum"]) >= 2 and v in spec[k]["enum"])
        count = next((_count(v) for v in args.values() if _count(v) is not None and not isinstance(v, str)), None)
        if removal or (count is not None and count <= 0):
            self.own_cart.pop(ident, None)
        elif self.cart_sets(name):
            self.own_cart[ident] = count or 1
        else:
            self.own_cart[ident] = self.own_cart.get(ident, 0) + (count or 1)

    def short_name(self, ident: str) -> str:
        row = self.rows.get(ident) or {}
        name = next((v for k, v in row.items() if isinstance(v, str) and k.lower() in ("name", "title", "title_text", "name_text", "product_name")), "")
        return " ".join(str(name).split()[:8]) or "this listing"

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options:
                removal = self.takes_off(options, value)
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
        # k17m9 (TEXT_ROWS): a long listing text no reader took in keeps more room (see _clip_room)
        text = _dump(_clip_room(observation, SHOW_CHARS, 0, HEALTH and self.blind) if TEXT_ROWS else _clip(observation, SHOW_CHARS))
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

    @_lever('compare_step', lambda self: self.compare_room_k16())
    def compare_room(self) -> int:
        """How many listings one side-by-side call may carry: the schema's limit, else what the store has shown to work.
        k17m (A5, love v20): once a call has failed, what the store has shown to work (a stated limit can be wrong),
        stepping down from the failed size by half at most, never below what has worked or below two."""
        spec = ((self.schemas.get(self.roles.get("compare") or "") or {}).get("properties") or {}).get(self.compare_target() or "") or {}
        cap = spec["maxItems"] if isinstance(spec.get("maxItems"), int) else None
        if COMPARE_STEP and self.compare_bad:
            room = max(2, self.compare_ok, min(self.compare_bad - 1, max(2, self.compare_bad // 2)))
            return min(room, cap) if cap else room
        if cap:
            return cap
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call: one step instead of several."""
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
        self.join_failed = False
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.offered()}
            cid = call["id"]
            if call.get("joined"):
                if COMPARE_STEP and self.join_failed:
                    # k17m (A5, love): the side-by-side call these opens were joined into showed nothing
                    out.append((cid, "Not shown: the side-by-side call above showed nothing for these listings (this store shows fewer at once). Open this one on its own."))
                else:
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
            if (ROTATE and self.held_msg and self.is_refuse(name, args) and not self.cart_ids() and not self.cart_touched
                    and not self.finished and self.left() >= 4):
                # k17m11 (ROTATE): an ending without a purchase before any cart call: the message held for the store's
                # first cart answer would go unused (the decline guards below answer first, and the episode often ends
                # before they let the ending through). Races 14-15: 36 of 41 and 29 of 31 of the k line's two-question
                # runs ended so, never carted (1 pass in 72); other versions sent all three messages. It asks now.
                late = self.late_question()
                if late and "New firm requirements" in late:
                    out.append((cid, "Not run yet (no step used). " + late + "\nSearch with this requirement before ending "
                                "without a purchase."))
                    continue
            if not ending and name not in self.follow_ups and self.left() <= self.tail() and (self.roles.get("buy") or self.roles.get("refuse")):
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
                    if (self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get("cart_view") and self.left() > 3
                            and not (SWAP_RESERVE and self.swap_room() and self.left() <= self.tail() + 1)):
                        # k17m7: not when the look would eat a cheapest pick's swap reserve
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
                    chain, hold = self.trail()
                    if hold:
                        out.append((cid, "Not run yet (no step used). " + chain + "\nCart now: " + self.cart_text()))
                        continue
                    # the check judges attributes: read the cart lines' own (and, for a cheapest pick, a few cheaper rows')
                    read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
                    seen = ("Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n") if read else ""
                    # k17: the coverage hold (C1) before the check, the strict review (C2) after it (see the module notes)
                    covered = self.cover_hold()
                    if covered:
                        out.append((cid, covered + "\n" + chain + seen + "Cart now: " + self.cart_text()))
                        continue
                    verdict = self.check_cart(limit=self.spend_hard)
                    if verdict:
                        out.append((cid, "Not run yet. " + chain + seen + verdict + "\nCart now: " + self.cart_text()))
                        continue
                    strict = self.req_audit()
                    if strict:
                        out.append((cid, strict + "\n" + chain + seen + "Cart now: " + self.cart_text()))
                        continue
                    if SOLD_REVIEW:
                        # k17m4: after a sell-out, the replacement's cart gets the strict review once (see the module notes)
                        strict = self.sold_review()
                        if strict:
                            out.append((cid, strict + "\n" + chain + seen + "Cart now: " + self.cart_text()))
                            continue
                    if reread and self.cost > SPEND_SOFT:
                        out.append((cid, "Not run yet (no step used). Kyo re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first."))
                        continue
                else:
                    # k17j: an ending with an empty cart, or with a line for every item (see the module notes)
                    stop = self.decline_guard()
                    if stop:
                        out.append((cid, stop + (" [episode finished]" if self.finished else "")))
                        continue
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
                            self.check_cart([first], self.spend_hard)
                            if self.verdicts.get(self.check_key([first])) and first in self.leads:
                                self.confirmed = first
                                self.trace.append({"kind": "refuse_blocked", "confirmed": first})
                                out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                                continue
            if self.is_refuse(name, args) and self.gone and not self.swap_ended and self.left() >= 2:
                # k2: a line sold out and the agent now ends without a purchase. In race 11 no ending without a purchase
                # after a sold-out checkout passed (0 of 39) while a replaced line passed 127 of 237: the listings read
                # are reviewed and Kyo buys one fitting listing per item when a fresh check passes them
                self.swap_ended = True
                named, done = self.review_buy()
                self.trace.append({"kind": "swap_end", "named": named, "bought": bool(done)})
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                if named and self.left() >= 3:
                    out.append((cid, "Not run yet (no step used). A sold-out line is no reason to end without a purchase: a careful "
                                "review thinks these listings fit: " + "; ".join(f"item {k}: {', '.join(v)}" for k, v in named.items())
                                + ". Put the best fitting one per item in the cart at the requested quantity and check out."))
                    continue
            if SOLD_BUY and self.is_refuse(name, args) and not self.sold_bought and self.gone:
                # k17m3: an ending without a purchase after a sell-out is taken over once (see the module notes)
                bought = self.sold_buy()
                if bought:
                    out.append((cid, "Not run (Kyo acted instead): a line sold out, and an ending without a purchase after a sell-out "
                                     "has not passed, so Kyo put the best remaining fit in the cart and checked out. [episode finished]"))
                    continue
                if self.sold_bought and self.cart_ids() and self.left() >= 2 and not self.finished:
                    out.append((cid, "Not run yet (no step used). After the sell-out Kyo set the cart to the best remaining fit ("
                                + self.cart_text() + "), and the checkout did not go through. Apply what the store said and check "
                                "out; an ending without a purchase after a sell-out has not passed."))
                    continue
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
                # too late for the review: a decline here usually throws away a built cart because time ran out
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
            if HEALTH and self.blind and self.is_refuse(name, args) and not self.blind_held and self.left() > 3:
                # k17m10 (HEALTH): Kyo cannot check the listings for the loop model in a blind store: one re-read first
                self.blind_held = True
                self.trace.append({"kind": "blind_decline_held", "step": self.step})
                out.append((cid, "Not run yet (no step used). Kyo cannot read this store's listings itself, so it cannot "
                            "check them for you: before ending without a purchase, go through the listings already shown in "
                            "the results above; if one meets every requirement, put it in the cart and check out. End without "
                            "a purchase only if none does; then send the ending call again."))
                continue
            if self.is_refuse(name, args) and not (HEALTH and self.blind) and not self.sought and self.left() > 8:
                # k2: before an ending without a purchase, each item with no fitting lead is searched for in code and
                # the review is run again on what that opened
                if self.seek_missing():
                    self.audited = False
                    review = self.audit()
                    if review:
                        out.append((cid, "Not run yet (no step used). Kyo searched again for the items that had no fitting "
                                    "listing. " + review))
                        continue
            if self.is_refuse(name, args) and not self.leads and not self.pushed and self.left() > self.max_steps // 3:
                self.pushed = True
                out.append((cid, f"Not run yet (no step used). {self.left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_refuse(name, args):
                done = self.complete_cart()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.finished else "")))
                    continue
                args = self.quiet(name, args)
            if ending and self.held_msg:
                # k8: the held message goes first (a step learned meanwhile keeps it; else the next question)
                late = self.late_question()
                if late and "New firm requirements" in late:
                    out.append((cid, "Not run yet (no step used). " + late + "\nCart now: " + self.cart_text()))
                    continue
                if late:
                    trimmed = "\n[Kyo] " + late + trimmed
            if ending and self.left() >= 2:
                go, said = self.gate_tell("order" if self.is_buy(name, args) else "end")
                if not go:
                    out.append((cid, "Not run yet. " + said + "\nCart now: " + self.cart_text()))
                    continue
                if said:
                    trimmed = "\n[Kyo] " + said + trimmed
            if SWAP_ONCE and name == self.roles.get("cart"):
                # k17j: a line already changed once is kept unless something shows it wrong (see the module notes)
                held = self.swap_hold(name, args)
                if held:
                    out.append((cid, held + "\nCart now: " + self.cart_text()))
                    continue
            if name == self.roles.get("cart") and self.removal_rule():
                lowered = self.lowered_by(name, args)
                held = self.ask_before_removal(lowered) if lowered else ""
                if held:
                    out.append((cid, held + "\nCart now: " + self.cart_text()))
                    continue
            reply = self.act(name, args)
            if reply is None:
                out.append((cid, "The call failed (network or server). You may try again."))
                continue
            observation = reply.get("observation")
            if name == self.roles.get("search") and self.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if SEARCH_KIND and name == self.roles.get("search") and isinstance(observation, dict) and not observation.get("error"):
                # k17m (C2, after Secretarybird v1): the store's word rule from two searches, and a short note
                said = self.search_note(name, args, observation)
                if said:
                    trimmed += "\n[Kyo] " + said
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _dump(observation)
                if not any(str(i) in shown for i in args[many]):
                    # k17m (A5, love): a size limit only when every id is a listing the store has shown (an unknown id
                    # fails a call too)
                    if not COMPARE_STEP or all(str(i) in self.rows for i in args[many]):
                        self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    self.join_failed = True
                    trimmed += f"\n[Kyo] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call."
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
                    self.join_failed = False
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))
            self.absorb(observation)
            if HEALTH:
                self.health(name, observation)
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
                        # k17 (E2): only a tool that reads like a listing view (k16b's guard; see the module notes)
                        other = next((t for t in self.schemas if t != name and self.single_target(t) and self.facts_from.get(t) is not False
                                      and t not in (self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("ask"), self.roles.get("search"))
                                      and (not END_SURE or t == self.roles.get("offer") or self.reader_like(t))), None)
                        if other:
                            self.roles["open"] = other
                    if not self.facts_from[name] and name != self.roles.get("open") and not self.roles.get("offer"):
                        self.roles["offer"] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get("cart"):
                if CART_SYNC:
                    self.book_cart(name, args, observation)
                else:
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
                    if REARM:
                        self.rearm()
            elif REARM and self.is_refuse(name, args) and not reply.get("finished"):
                self.rearm()
            text = self.show(observation) + paged + trimmed
            if name not in plain and not ending:
                text += self.cheapest_here([observation] + self.paged_obs)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += "\n" + proof
            if name == self.roles.get("cart") and any(_count(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.rows and _in_rows(observation, v) for v in args.values()):
                    text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
            self.note_sold = False
            self.change_now = []
            self.price_moved_now = False                     # k17m (C5): only this call's note counts
            lines = self.loose_lines(observation)
            if lines:
                self.note_gone = []
                meaning = self.take_note(lines)
                if self.note_sold and not self.note_gone and not ending:
                    # k16: see the module notes
                    self.doubt_at = self.step
                    self.trace.append({"kind": "doubt", "at": self.step, "from": name})
                if meaning:
                    text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
            if self.held_msg and name == self.roles.get("cart") and not self.finished:
                # k8: the store's first cart answer is read (its notes too): the held message is released now
                late = self.late_question()
                if late:
                    text += "\n" + late
            if self.is_refuse(name, args) and not reply.get("finished") and not self.finished and self.change_now:
                text += "\n" + self.rethink()
            if self.is_buy(name, args) and not reply.get("finished") and not self.finished and (self.change_now or (PRICE_AGAIN and self.price_moved_now)):
                again = self.recheckout(name, args)
                if again:
                    text += "\n" + again
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
        if HEALTH and not holding and not self.rows and self.own_cart:
            # k17m10 (HEALTH): no listing read, so the cart view cannot be matched: Kyo's record of its own cart calls (it
            # used to end without buying with lines in the cart)
            holding = True
            self.trace.append({"kind": "finish_own_cart", "lines": dict(self.own_cart)})
        if buy and (holding or (self.cart_touched and not fresh and self.left() >= 2)):
            faults: list[str] = []
            args = _shape(dict(self.brief.get("buy_args") or {}), self.schemas.get(buy), "args", faults)
            if faults and CODE_END and (self.model_down() or self.brief_failed):
                # k17m (C1): required arguments filled in code
                faults = []
                args = _shape(self.plain_args(buy, dict(self.brief.get("buy_args") or {}), "Please place the order for the cart as it stands.", True),
                              self.schemas.get(buy), "args", faults)
            if not faults:
                # a reply that adds a requirement does not stop a forced finish: no steps are left to act on it
                if self.left() >= 2:
                    self.gate_tell("order")
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
            if faults and CODE_END and (self.model_down() or self.brief_failed):
                # k17m (C1): required arguments filled in code (the fix below needs a model)
                faults = []
                args = _shape(self.quiet(refuse, self.plain_args(refuse, dict(self.brief.get("refuse_args") or {}),
                                                                 "Nothing in the store meets every requirement, so I am not ordering anything.", False)),
                              self.schemas.get(refuse), "args", faults)
            if faults and END_SURE:
                # k17 (E2): arguments the schema check turns down are fixed once (see the module notes)
                fixed = self.refix(refuse, args, "; ".join(faults))
                if fixed is not None:
                    args, faults = fixed, []
            if not faults:
                if self.left() >= 2:
                    self.gate("end")
                if self.finished or self.left() <= 0:
                    return ""
                reply = self.act(refuse, args)
                self.trace.append({"kind": "forced_refuse", "reply": _clip(reply, 200)})
                seen = (reply or {}).get("observation")
                why = self.last_fail if reply is None else (self.show(seen) if isinstance(seen, dict) and seen.get("error") else "")
                if END_SURE and why and not self.finished and self.left() > 0:
                    # k17 (E2): an ending the store turned down as malformed is fixed once and sent again
                    fixed = self.refix(refuse, args, why)
                    if fixed is not None:
                        reply = self.act(refuse, fixed)
                        self.trace.append({"kind": "forced_refuse", "again": True, "reply": _clip(reply, 200)})
                if reply is not None and not self.finished:
                    self.change_now = []
                    self.observe(refuse, reply.get("observation"))
                    # k2: a refusal that changes the task is reviewed (and acted on) here, as on the agent's own ending
                    review = self.rethink() if self.change_now and not self.finished and self.left() > 0 else ""
                    if self.finished:
                        return ""
                    return "ending without a purchase was refused: " + self.show(reply.get("observation")) + ("\n" + review if review else "")
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
                self.chat[index]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        opening = self.view.get("opening")
        if opening is not None and TEXT_ROWS:
            try:
                # k17m9 (TEXT_ROWS): the store's first view is read like any result
                opening = _text_rows(opening)
            except Exception as exc:  # noqa: BLE001
                self.lever_fail("text_rows", exc, "run")
        if opening is not None:
            self.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += "\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n" + self.status()
        self.chat = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]
        self.work(LOOP_CALLS, self.spend_hard)
        if CODE_END and not self.finished and self.model_down():
            # k17m (C1): the models are gone: Kyo fills the cart and checks out in code
            self.code_buy()
        if not self.finished:
            self.rescue()
        if SOLD_BUY and not self.finished and not self.cart_ids() and self.gone:
            # k17m3: the forced finish after a sell-out would end without a purchase: Kyo buys the best remaining fit
            self.sold_buy()
        if not self.finished:
            turned = self.finish(again=True)
            # the store turned the ending down (a change from the shopper, a sold-out line): a short second round deals with it
            if turned and not self.finished and not self.broke and not self.no_credit and self.left() > 1:
                self.chat.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status()})
                self.work(self.loop_calls + 10, self.spend_hard + 0.01)
            if not self.finished:
                self.finish()
        for again in range(2 if END_SURE else 0):
            # k17 (E2): a run that has still not ended tries its finish again
            if self.finished or self.broke or self.no_credit or self.left() <= 0 or self.elapsed() > CALL_SECONDS + 12:
                break
            self.trace.append({"kind": "end_retry", "n": again + 1})
            self.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "calls": self.calls})
        return self.trace

    def work(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        # k17m5 (COSTLY): the interview runs inside this loop, so a line lowered at its question plan lowers the cap as much
        start_line = self.spend_hard
        while (not self.finished and not self.broke and self.left() > 0 and self.loop_calls < loop_cap
               and self.cost < spend_cap - (start_line - self.spend_hard)
               and self.elapsed() < LOOP_SECONDS):
            if HEALTH and self.blind and not self.blind_told and self.chat and self.chat[-1].get("role") in ("tool", "user"):
                # k17m10 (HEALTH): told once, between turns
                self.blind_told = True
                self.chat.append({"role": "user", "content": BLIND_NOTE + "\n" + self.status()})
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
                continue
            if CODE_END and self.unheard and not self.model_down():
                # k17m (C1): answers no model could read are read again once a model answers
                told = self.reread_unheard()
                if told:
                    self.chat.append({"role": "user", "content": "[Kyo] The shopper's earlier answers were read again. New firm requirements: "
                                      + "; ".join(told) + ". Apply them to every pick.\n" + self.status()})
            self.trim()
            self.loop_calls += 1
            if EMPTY_SEEK and not (HEALTH and self.blind) and not self.sought and self.left() <= EMPTY_SEEK and not self.cart_ids() and self.buys == 0:
                # k17m4 (EMPTY_SEEK): steps are running out with nothing in the cart (see the module notes)
                if ROTATE and self.held_msg and not self.cart_touched and self.left() >= 9:
                    # k17m11 (ROTATE): no cart call yet with the steps running out, so the store's first cart answer that
                    # releases the held message may never come (practice 11-06: 2 of 15 runs ended so, with the message
                    # unused): it asks now, and a value its answer gives joins the searches below
                    late = self.late_question()
                    if late:
                        self.chat.append({"role": "user", "content": "[Kyo] " + late + "\n" + self.status()})
                found = self.empty_seek()
                if found:
                    self.chat.append({"role": "user", "content": found + "\n" + self.status()})
            # k17m (C4): the spare model leads after stuck turns (k17m2: through a guarded helper)
            models = self.loop_models()
            message = self.think(self.chat, models, 3000, self.offered())
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
                self.stuck += 1
                self.chat.append({"role": "assistant", "content": content or "(no call)"})
                self.chat.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get("cart"), self.roles.get("buy"), self.roles.get("refuse")} - {None}
            if not self.asked and self.roles.get("ask") and self.msgs_free() > 0 and any(c["function"]["name"] in committing for c in calls) and self.left() > self.ask_room():
                self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.chat.append({"role": "tool", "tool_call_id": call["id"], "content": "Not run (no step used): the shopper is asked first."})
                summary = self.interview()
                self.chat.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status()})
                continue
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.trace.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            step_before = self.step
            replies = self.run_calls(calls)
            # k17m (C4): a turn whose calls used no store step (all held, refused or malformed) counts as stuck
            self.stuck = 0 if self.step > step_before else self.stuck + 1
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": cid, "content": text})


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
