from __future__ import annotations

import json
import math
import re
import time
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient



# ------------------------------------------------------------ p00_tune.py
# p00: every fixed number of the errand (01a 3, 01b 3.13) and the model lineups (01a 3.10).
# Times in seconds, money in US dollars of model spend, steps = store tool calls.


class Tune:
    """Constants grouped by area; read as Tune.<Group>.<NAME>."""

    class Clock:
        DEADLINE = 265          # whole errand; time left = DEADLINE - elapsed
        CLOSING = 70            # below this optional model work stops
        LAST_CALL = 28          # below this the actor loop ends and code forces a finish
        MODEL_FLOOR = 18        # no model call is started below this (LAST_CALL - 10)
        CLOCK_STEP = 30         # status line rounds time left down to this
        GAP = 0.15              # minimum gap between two outgoing calls
        SLOW_GAP = 1.0          # gap used during a slow period
        SLOW_SPAN = 30          # a 429/503 reply starts a slow period this long
        DOWN_AFTER = 45         # service counts as down: a miss since the last success, success older than this
        TIMEOUT_MIN = 8         # model timeout = clamp(left - TIMEOUT_PAD, TIMEOUT_MIN, TIMEOUT_MAX)
        TIMEOUT_MAX = 75
        TIMEOUT_DOWN = 30       # timeout ceiling while the service is down
        TIMEOUT_PAD = 13        # LAST_CALL - 15
        STORE_TIMEOUT_MIN = 3
        STORE_TIMEOUT_MAX = 95
        SLOW_SEND = 20
        FINISH_CHECK_MIN = 38   # finish check skipped at or below this
        REREAD_MIN = 130        # request reader retries / re-read need this much left
        ASK_STOP = 110          # asking, sweeps and the product plan stop below this
        PAD_ITEM = 20           # CLOSING + pad: item check, change review
        PAD_RANK = 30           # rank check, audit, several search/sweep routines
        PAD_ASK = 40            # asking, product-focus plan, sweeps, widened search
        PAD_REREAD = 60         # request re-read, reader retry cycles, reference and some search starts

    class Tries:
        PER_MODEL = 2           # sends per model in a lineup (1 while the service is down)
        STORE_BASE = 3
        STORE_EXTRA = 5         # a store call goes out at most STORE_BASE + STORE_EXTRA = 8 times
        BACKOFF_WAITS = 4
        BACKOFF_CAP = 8.0
        BACKOFF_BASE = 1.0
        BACKOFF_BASE_429 = 2.0
        STEP_BUMPS = 2
        READER_ATTEMPTS = 12    # request reader: one model per attempt, up to 3 cycles
        READER_CYCLE = 4

    class Spend:
        SOFT = 0.03             # from here the strong model is skipped when a cheaper one follows it
        CAP = 0.045             # strong model skipped always; budget stage on the status line
        HARD = 0.10             # every further model call refused
        CHARS_PER_TOKEN = 2.3   # estimate = chars / 2.3 * IN_RATE + min(cap, OUT_EST_MAX) * OUT_RATE
        IN_RATE = 0.000002
        OUT_RATE = 0.00001
        OUT_EST_MAX = 1500
        PER_PRODUCT = 0.3       # scale = (1 + 0.3 * min(2, products - 1)) * clamp(40 / steps, 0.5, 1.0)
        EXTRA_PRODUCTS = 2
        STEP_REF = 40
        SCALE_MIN = 0.5

    class Loop:
        TURNS = 80              # actor turns at most
        IDLE = 3                # consecutive replies without a tool call
        STUCK = 4               # turns without store progress before the lineup is turned
        FAILS = 2
        GROUP = 16

    class Finish:
        STEPS = 2               # added to the order reserve with a message tool or a finish-time instruction
        CHECK_STEPS = 2         # owed steps taken from the finish check
        CODE_TRIES = 4          # forced-finish sends
        ORDER_BY = 7            # reserve floor: first order while at least this many steps remain
        ORDER_DUE = 10          # base reserve = max(7, min(10, A // 4 + 1))
        HELD_MAX = 16           # held = max(base, min(16, A // 3))
        TROUBLE_BASE = 8        # after trouble: held = max(held, 8 + products, 9)
        TROUBLE_MIN = 9

    class Ask:
        DEFAULT_MSGS = 3        # messages assumed allowed when the request states none
        READS = 2               # listings read before planned questions (multi)
        SOLO_EXTRA = 1          # one more read for a single product
        EARLIEST = 5            # earliest store step for a question without enough reads
        GRACE = 3               # steps 5..8 keep waiting while more than GRACE_ROOM steps remain
        GRACE_ROOM = 20
        STOP_ROOM = 10          # asking stops at steps left <= max(10, order reserve)

    class Text:
        VIEW = 20000            # default cut when a structure is rendered for a model
        SEARCH_CLIP = 600       # each string of a search result
        RECORD_CLIP = 2500      # each string of any other tool result
        NEST = 8
        CUT_MARK = "...(cut)"

    class Evidence:
        SYNONYM_MIN = 4         # filled fields needed before an equivalent value may count

    class Search:
        OPEN_PAGES = 3          # opening search: 1 + max(3 + 2 when cheapest, deep pages) rounds
        CHEAP_PAGES = 2
        DEEP_SOLO = 3
        DEEP_FEW = 1
        FEW_MAX = 3
        READ_ROUNDS = 8
        SWEEP_READS = 12
        SWEEP_PAGES = 3
        REVEAL_PAGES = 2
        REVEAL_OPENS = 4
        REVEAL_MAX = 2
        REACH_ROWS = 24
        CAP_ROWS = 12
        CAP_CALLS = 6
        FEW_ROWS = 16
        FEW_KEEP = 6
        PAIR_OPENS = 3
        PAIR_ANCHORS = 3
        PAIR_ALTS = 2
        LIST_CAP = 5            # ids per batch-details call: min(item limit, 5)
        WALK_PAGES = 6
        WALK_TOTAL = 8
        RUNGS = 4
        RUNG_PAGES = 3
        RUNG_CALLS = 6
        RUNG_OPENS = 4
        VALUE_PAGES = 4
        VALUE_TOTAL = 8
        VALUE_PER = 5           # value-search cap = max(VALUE_TOTAL, VALUE_PER * products)
        LABEL_TRIES = 4
        LABEL_OPENS = 3
        CEIL_CALLS = 4
        CEIL_OPENS = 3
        FIT_TRIES = 3
        FIT_OPENS = 4
        WIDE_TRIES = 3
        WIDE_OPENS = 2
        WIDE_KEEP = 6

    class Refs:
        HOPS = 8                # chain stops when the next hop would reach this count
        SEARCHES = 12           # reference searches per errand
        STEP_SHARE = 0.4        # share of the step cap, never below STEP_FLOOR
        STEP_FLOOR = 8
        DIFFER_FLOOR = 6        # floor when every open reference is a "differ"
        DIFFER_BLIND = 9        # "differ" chase dropped after this many steps with no chain under way
        HOP_CHECKS = 12

    class Models:
        SMALL = "anthropic/claude-haiku-5.5"
        STRONG = "anthropic/claude-sonnet-5.5"
        OTHER = "openai/gpt-6-luna"
        PREMIUM = frozenset({STRONG})   # models the spend guard skips

    class Aim:
        CART = "ADD TO THE CART"
        ORDER = "ORDER"

    class Marks:
        NOT_RUN = "Not run (no step used)"
        NOT_RUN_YET = "Not run yet (no step used). "
        NOTE = "[note]"
        SWEEP_DONE = "[shelf sweep done]"
        MEANING = "Meaning: "
        REFUSED = "REFUSED"
        STATED = " (stated)"
        REVEALED = " (revealed by the shopper)"
        OVER = "[episode over]"


_p00_H, _p00_S, _p00_L = Tune.Models.SMALL, Tune.Models.STRONG, Tune.Models.OTHER

# role -> (models tried in order, reply token cap); tuples so no caller can grow them across errands
LINEUPS = {
    "actor": ((_p00_H, _p00_L, _p00_H), 2500),
    "actor_turned": ((_p00_L, _p00_H, _p00_H), 2500),
    "reader": ((_p00_H, _p00_S, _p00_L, _p00_H), 8000),
    "rereader": ((_p00_H, _p00_S, _p00_L), 8000),
    "listener": ((_p00_H, _p00_S), 2500),
    "checker": ((_p00_H, _p00_S), 3000),
    "ranker": ((_p00_H, _p00_S), 6000),
    "auditor": ((_p00_H, _p00_S), 6000),
    "reviewer": ((_p00_H, _p00_S), 6000),
    "planner": ((_p00_H, _p00_S), 2500),
    "focus": ((_p00_H,), 1500),
    "hop": ((_p00_H,), 3000),
    "finisher": ((_p00_H, _p00_S), 2500),
    "meaning": ((_p00_H, _p00_S), 600),
}



# ------------------------------------------------------------ p01_voice.py
# p01: every model prompt (01a 4, 05 4.1-4.2, 08 4), written fresh. Placeholders are <<NAME>> tokens filled with
# str.replace. Link.consult / Intake put the "ROLE: <tag>" line in front; the actor prompt has none.

PROMPTS = {}

# ---- 01a 4.1: system message of the tool-using shopper
PROMPTS["actor"] = """You shop for one person inside a simulated online store. The store's tools are the only way you can act.

HOW A RUN IS JUDGED. A run scores either 1 or 0. The 1 comes only from a flawless ending, which needs all of this at once: the purchase honours each written condition of the shopper, each hidden need disclosed on asking and each change announced mid-errand; the quantity is right; the total respects the budget; under a demand for the cheapest or the best by some measure, it really is that extreme item; and every procedure the request lays down was carried out. When nothing in the store can satisfy all of this, the correct ending is a decline, done exactly the way the request prescribes.

EVERY REPLY makes at least one tool call. Keep text to a minimum: the conditions in force, the verified candidates with price and stock, your next move.

A. WHEN DOES A LISTING QUALIFY? (the score depends on this above all)
A1. A property is proven by its own field: the wanted value is held against the store's wording in that field.
- Recognise a field by its meaning. Another separator, an extra prefix, a longer or shorter form or a well-known alias still names it (a strap stored under "band"; the finish stored under "surface treatment").
- Deciding fields: (1) the one named exactly like the property; (2) the property name with one word in front of it (length -> "blade length"). A value in either one suffices.
- Names that differ only by separators (toe_cap, "toe cap") are a single field, met when either spelling holds the value.
- Lacking a deciding field, let an alias field settle it. Only if not a single field names the property may the value come from the title or the options list. Never treat a description as evidence.
A2. Comparing words: letter case, separators, singular or plural and spelling variants do not matter.
- Containment passes: a narrower value that includes the wanted one (linen <- "linen blend"; teal <- "deep teal"). In a list, one matching element is enough.
- An equivalent (synonym, abbreviation, related word, shade: aluminium ~ "alu", quilted ~ "quilt", ivory ~ "cream") is accepted just as long as every listing of the category leaves the exact wanted value out of that field. The moment one listing writes it (others write "matte"), the equivalent ("flat") fails.
- A value joined with "+" ("wool + nylon") fails for one of its parts when that field elsewhere holds the other part on its own.
- FIELD EVIDENCE lines are computed by code from this store's records; they decide which of these cases applies.
- These always fail: a different value of that property (manual vs electric; pine vs birch); a word that only looks similar; a catch-all claiming everything ("suits any decor"), except when the wanted value itself appears in its list; a value that merely tends to come with, or hints at, the wanted one ("padded" is not "waterproof").
- Exactly two stand-ins are allowed: a shade the trade commonly sells under the wanted colour name, and a narrower value containing the wanted one.
A3. A field about another property (handle material vs material; strap width vs width) never replaces the property's own field when the category's listings have one.
A4. Conditions are only what the shopper stated or revealed. Never knock a listing out over traits nobody asked for (use, gender, age group, size, quality, occasion, condition) that you guessed from the story or a description. Treat the story as context; when it seems to matter, ask about that property rather than assume. Your own questions create no conditions; only the answers do.
A5. Figures: convert units first, then compare amounts (1.5 l = 1500 ml). A limit includes its edges; a range must lie entirely inside it. If the title has to serve as evidence because the property has no field, an exact wanted value counts as met once the title carries those words and figures in unchanged sequence, extra text after them allowed (wanted "60 x 90 cm": "60 x 90 cm, set of 2" passes, "60 x 95 cm" fails).
A6. Number pairs in a title ("80 x 120 cm") do not tell which dimension each figure is. They are never a reason to reject a listing or to leave it unopened; read its detail fields.
A7. Exclusions and closed lists bind ("no plastic"; "grey or white, nothing else").
A8. The category alone settles the kind. A listing in the requested category counts as that kind even when its title or description names another object (a container, case, refill, set or accessory). Never skip, reject or rank it lower for how it looks; judge only conditions, price and stock. Properties need not repeat the category name. Where the store labels categories, take the label that suits the request's wording as a whole ("pans or skillets" -> the joint label "Pans & Skillets", not the narrower neighbour "Skillets").
A9. When the shopper wants it on hand, a listing qualifies only while its stock shows it available. Price x quantity must not exceed the budget (hitting it exactly is allowed). Read every price in the budget's own unit, except where a record names a different one (45 against a 3000-cent budget means 45 cents); rescale neither side.
A10. Declines are rarely right. If no listing matches word for word but one meets every condition through these rules (a narrower value containing the wanted one; the same field under another name; an equivalent where no category listing writes the exact value), that listing is the fit. Decline only when the search truly found no full match.

B. CHEAPEST, OR BEST ON A MEASURE
B1. Cheapest means the lowest price among ALL qualifying listings; finding one that qualifies proves nothing. Ways to get there:
1. A keyword search that needs every word and also matches property values: query value + kind to get exactly that value's carriers, read all of its pages, open them by rising price.
2. A category view under a price ceiling shows every price but no property values: use it to learn which cheaper rows exist (lower the ceiling until the total fits), then check those rows' properties.
3. For each wanted value, a short query with the value first, then the kind ("walnut desk"). Read every page whenever the total fits on the readable pages (otherwise add a second wanted value or a narrower kind word). Repeat with other names for the kind ("table") and with the value alone.
4. When a value cannot be searched (say the store strips figures from queries), also search the product line of each fit or near fit = maker + one distinctive title word (two words), every page. Sibling sizes and colours are separate listings that share property values and may cost less.
5. Order the candidates you can see by price on your own, then open upward starting with the lowest. Your answer is the earliest listing whose fields meet everything, on condition that each lower-priced one got opened and failed; qualifiers found after it serve as reserves.
B2. Best on a measure (largest, longest, highest, most ...): line up every qualifier on that measure in the same way and take the extreme; a listing that lacks the measure never wins.

C. ENDING THE ERRAND
C1. Keep a ranked list of verified fits with at least one reserve (the top pick may sell out at checkout).
C2. Before each checkout attempt: (a) the cart holds exactly one line per requested product (one line for a single product), each the chosen listing at its required count, nothing more; (b) since the latest cart change, the chosen listing's stock and price were read again and seen available and within budget.
C3. Decline only after a thorough search found no full match, and follow the request's decline rule to the letter. When substitutes are forbidden, the decline text names, describes and points to no listing whatsoever (no ids, no titles, no prices).
C4. Steps: send the first checkout attempt while 7 or more steps remain (room to deal with a refusal). From 7 left only the cart, stock-check and finishing tools stay usable. Decide once; do not keep switching the cart.

D. CHANGES ALONG THE WAY
D1. The shopper may change things: raise or lower the budget; add, drop or swap a condition; prices may rise; an item may sell out. One line can combine several, and such lines can arrive inside any tool result (an extra field holding sentences) or together with a checkout attempt. They take effect at once: re-verify the pick and the reserves, and under cheapest or best choose again from the updated set of acceptable listings.
D2. A checkout attempt can be REFUSED; the episode goes on and the cart stays as it was. Read the refusal, apply every change it names, remove whatever stopped fitting (a line that is now sold out), cart the next verified fit, re-check its stock and price, and check out again.
D3. Stock and price can move without notice: re-check both right before each attempt. If the current price x quantity is above the current budget, it no longer fits.

E. HIDDEN NEEDS AND THE SHOPPER
E1. Some needs are never written down; such a need surfaces only if a message raises its property. The opening question of a message draws the most reliable answer, so ask about one property per message (two at most, the likelier first) in a short, simple, friendly question that names the property; one or two example values are fine. No instructions, lists, labels, positions or rules inside it.
E2. An indifferent answer ("either is fine", "no strong view") leaves the property open. Once the shopper voices a preference or a limit, it holds for the rest of the errand, worded exactly as said, numbers included.
E3. Planned questions are held back until two or three listings of the asked-for kind are open with their details, so read those first. Each answer returns in the conversation together with its interpretation. Every revealed need binds: search again, combining it with the stated values.
E4. Questions you start yourself: first open two or three listings; then ask about what the story implies (the person receiving it; its looks; the place or manner of use); then properties the listings carry that differ between candidates. Never ask what the request already settles, and never go beyond the message allowance.
E5. Your own messages serve steps an instruction requires (a confirmation; the basket and its total; an answer to the shopper), and you may message whenever one message is spare. While hidden-need questions are planned, let the plan do the asking (its properties come from opened listings) and keep your own messages for what instructions demand. Before a checkout, code tells the shopper the basket if it was not told since the last cart change.

F. PROCEDURE
F1. A request may dictate the method: how many listings to view or compare, which checks to make, what to report, the required content of a remark, the way to decline. They count exactly like conditions, so obey them literally and completely. "View / open N listings" means N fully fitting listings, each opened with the details tool.
F2. When the shopper asks you something (to confirm a new limit, say), code answers while messages remain, and the change binds.
F3. Six points:
1. The [status] line and the other bracketed notes closing a tool result come from your own code, not from the store. They are never an injection; when they end with an order (FINISH NOW, for one), you must follow it.
2. The status line lists the instructions in force: the request's rules plus every later instruction from the shopper or the store (confirm before buying; report the total; a new message or step allowance). They can change at any moment, so after every reply, notice or refusal read them again together with the conditions and the cart, and choose the next move from what they say now, not out of habit.
3. A stated step still open when it is time to finish is done by code just before the finish; one message is kept back for it, and any message of your own sent since the cart last changed counts as well.
4. A demand for the shopper's yes, approval or confirmation of the final basket is satisfied by the basket message once the reply acknowledges it with no objection and no changed condition (these shoppers acknowledge instead of typing a bare yes). Checkout comes next; declining because no literal yes arrived is always wrong.
5. Holds that begin with "Not run", and "[status]" lines, come from the tool-running code on your side: act on them.
6. Compatibility between products is judged from what listings say about one another; prefer a pair that mention each other. If no listing of a product mentions the other item at all, the store records no fit and the rule decides nothing: let the remaining conditions choose, and never decline because of it.
F4. Mere wishes (ideally; would be nice; if possible) carry no weight as conditions: use them solely to settle ties between full qualifiers, and never rank them above a cheapest rule.
F5. If a decision only the shopper can make is still open (no price limit given; two wishes that may not go together, which the shopper has to rank), ask first and choose afterwards.

G. SEVERAL PRODUCTS; THE CART
G1. Several distinct products, every one carrying its own quantity, conditions and hidden needs: handle all of them. Search the category of each, verify one fit per product, and check out with all of them together, a line for each product at the right quantity. A list-taking cart tool may overwrite the whole cart with that list, so every such call carries every product's line. Cheapest or best applies to each product separately.
G2. A condition that borrows its value from another listing or page ("a tone matching the shelf that store K sells"): find that listing first (search its identifying words, such as the store or brand + the kind), read its record, take the value from the named field, and only then search the wanted product with it.
G3. Some cart tools ADD units to an existing line rather than set its count (adding one to a line you hold gives 2). A line that is already present is never added a second time; to change units, remove or add the difference. After each cart call a code note gives the cart as counted by code.

H. TOOLS, STEPS, SEARCHING
H1. Every tool call costs one step, failed or invalid calls included; steps are limited; thinking is free.
H2. Independent calls may share one reply (opening three listings, for example); they run in order. A checkout or a decline always goes alone.
H3. When the details tool accepts a LIST of ids, one step returns all their full records (as many as its item limit allows): use it every time two or more need opening.
H4. Tool names, parameter names and result layouts change from store to store: study the description and the JSON schema of every tool, and use exactly the names and allowed values they give. Use the first results to work out which tool gives price, which gives stock and which gives detailed properties.
H5. Find out early which category labels the store puts on the requested kind; among neighbouring labels choose the one nearest to the whole request's wording; then do your searching and buying within that label.
H6. Search behaviour varies, so infer it from the first results: rows sorted by best match or by a fixed id order; keywords in all-words mode (an added word lowers the total) or any-word mode (an added word raises it, and listings that match more of the words come higher); there may be only a handful of rows or pages visible; a filter that works in one mode only (keyword mode may ignore category or price filters; filter mode may ignore words).
H7. Reported total versus visible rows: a larger total means the fit or the cheapest may be on a later page or beyond the visible range. In all-words mode narrow the search (more exact words, or category + price filters with a lower ceiling) until the reported total is small enough to be seen in full on the visible pages.
H8. Queries: 1-3 words, the most distinctive one FIRST. Some stores read only the first word or two, so a long query may in effect search just its start; compare a longer query's total and rows with a shorter one that starts the same way (the notes report what came out).
H9. All-words mode: value + kind, value first, returns exactly the carriers. Any-word mode: common words (the kind, units) hit most of the shelf and bury the rest, so search only the value's own telling word or figure ("herringbone"; "35mm") to get just the carriers, plus the spellings listings use ("35 mm", "3.5 cm"); read every page when the total fits; add the kind only to pull the closest matches forward.
H10. Keyword queries can also hit category names and property values; the notes after each search interpret the totals seen so far in this store."""

# ---- 01a 4.2: request reader (tag read-request)
PROMPTS["reader"] = """Answer with a single JSON object written compactly on one line (no indentation) and nothing else. It has twenty keys, in this order: tool_roles, msg_limit, step_limit, spend_cap, money_unit, stock_only, pick_rule, rule_measure, procedures, duties, decline_unnamed, wishes, ask_before, story_hints, pointer_needs, pointer_targets, item_kind, units, musts, parts. Together they turn a shopper's request and the store's tool list into one plan. The tool list (TOOLS) and the request (REQUEST) come after these instructions.

1. Tool roles
- tool_roles: an object that maps each of eight roles to the exact name of one listed tool, or to null when no tool suits it. The roles: "search", "details", "view", "message" (writing to the shopper), "cart" (adding items to the cart or removing them from it), "cart_view", "order" (buying, checking out), "decline" (closing the errand without a purchase).

2. Limits and ranking
- msg_limit: the whole number of messages the shopper allows; null when not stated.
- step_limit: the whole number of moves or steps the request allows; null when not stated.
- spend_cap: the highest total price, as the very figure the request writes, in the request's own unit and never converted ("3500 cents" -> 3500); null when there is no cap.
- money_unit: the request's pricing unit, or null.
- stock_only: true when only goods in stock or on hand are wanted.
- pick_rule: exactly one of "cheapest", "best", "none". "cheapest" only for an unmistakable demand for the lowest-priced fitting item or the lowest order total; "best" only for an unmistakable demand for whatever ranks highest on some measure; otherwise "none". Remarks such as "funds are tight" or "I watch every penny" do not ask for the cheapest; they only explain the cap.
- rule_measure: the measure when pick_rule is "best"; otherwise "".

3. Procedure
- procedures: the request's procedural rules aimed at the assistant, each copied word for word (limits on how many listings get viewed; what must be checked or reported; how to decline; prohibitions; items to include). What the shopper plans to do on their own side is story and stays out.
- duties: the same rules again as objects, one each, [] when there are none; this also covers a rule that only governs how the errand is finished or what is said at the end. Each object: {"text": the request's wording, 20 words at most; "by_finish": true when it has to be done by the finish at the latest, that is before the order or the decline goes out (check back with the shopper first; report the basket or the total; a closing remark; a last check), else false; "via_message": true when it can only be done by messaging the shopper, else false; "before_change": true when the shopper must be consulted before any cart line is changed, swapped or removed, else false}.
- decline_unnamed: true when the decline rule bans any substitute or alternative from being offered, named or suggested; false when the rule asks for something to be named; null when the request has no decline rule.

4. Story, wishes and open points
- story_hints: strings "property (product): short detail" (for a single product "property: detail"), one per story detail that bears on a property without stating its value: the use (what for, with what), the place, the user, what it has to fit, match or withstand. Use the property name that listings are most likely to use. A detail that could bear on two properties: the likelier one first, a slash, then the other ("colour / pattern (cushion): must go with a plaid armchair"). For matching or suiting a room or wall, an outfit or the things it goes with, and for the look of the item, put colour first and pattern or style second. Leave out details that only justify a stated condition (why that size, why that material). Never a hint, only background: the weather; travel; someone's job or timetable; a place where nobody wears or uses the item; a purpose every listing of the kind serves (holding plants, for a planter). [] when none.
- wishes: what is liked but not strictly needed (ideally; preferably; nice to have; if possible). Never copy these into musts.
- ask_before: points only the shopper can settle, to be asked before choosing: the budget when none is set; which of two wishes weighs more when the request admits both may be impossible together (both wishes then go to wishes, not to musts); anything the request wants cleared with the shopper before a listing is chosen. A check that needs the chosen basket (its approval, its total, the order itself) belongs to finishing: put it in duties with by_finish true, never here. [] when none.

5. Values borrowed from another listing
- pointer_needs: conditions whose value is known only through a pointer to ANOTHER listing or page ("a finish like the stool store K sells"; "within 2 of M's figure"; "a colour unlike the one on P's page"). Three forms, where <name> is how the request names that listing:
  1. same value: "<property> (<product kind>): as shown on <name>";
  2. must differ: "<property> (<product kind>): different from the one shown on <name>";
  3. counterpart: "<property> (<product kind>): counterpart of what is shown on <name>". A counterpart means the opposite number of that value, the store's paired value for it, or any value required to suit, go with or be compatible with that listing's value rather than equal it.
  Add every entry to musts as well, in exactly that form. [] when none.
- pointer_targets: one object per pointed-at listing, in the same order as pointer_needs: {"property": the property concerned; "product": our product concerned; "seller": that listing's store or brand, copied exactly from the request; "kind": the product type it is; "details": strings in the request's wording for any further values given for it ("matte", "16gb", "net weight 250 g"); "relation": "same", or "pairs" (the value the store pairs with theirs: a counterpart, an opposite number, or a value that must suit or go with it - neither identical nor just any other), or "differ" (must NOT equal theirs: unlike, other than, set apart from), or "within", "at most", "at least"; "tolerance": the number that goes with "within", else null; "unit": that number's unit, else ""}. [] when none.

6. Products and conditions
- item_kind: the wanted kind in the request's wording; several kinds joined with "and".
- units: the whole number of units of one listing to buy; 1 by default. Something sold as one piece (a cutlery set; a pair of socks) is 1. Change it only for an explicit request for several units ("I need three of them"); how many people share or use it, or how many old ones there were, is story and keeps 1.
- musts: a list of strings, one per firm condition, "property: condition", carrying the value, limit, exclusion or closed list just as the shopper put it; with several products, all of their conditions, each written "property (product): condition".
  1. What counts: a property of the product together with a value the request demands for it ("the handle must be wooden"; "with a glass lid"; "at least 4 burners"; "the look has to be rustic"). The value must be words or figures that the request itself gives as the item's value. How many things one sold pack, box or set holds is the pack size, written as the figure followed by the noun that comes after it in the request ("a tin of 36 crayons" -> "pack size: 36 crayons", never "crayons per tin: 36").
  2. What does not count: price, stock, quantity, budgets, mere wishes; repeating the kind or category (words that are part of the category name - a season, an occasion, an audience - are already covered by it). Never from the story: the person it is meant for; an event at which it gets used or worn; where, when or how it gets used; how many share it; the fact that it takes the place of an old one ("a new toaster" only signals a replacement and sets no condition). An occasion, use or audience becomes a condition only if the request states it as a property of the item itself ("the style must be nautical"; "a bag built for climbing"). A quality, reason or wish drawn from the story (light and quiet; fits in a glovebox; works with my other kit) is a hint and never a condition. A saying or a comparison offering neither a property nor a value of its own ("solid as a rock") just colours the tone: treat it as neither condition nor hint.
- parts: one object per DIFFERENT product to buy, in the request's order: {"kind": its kind as worded; "units": units of one listing of it, counted as for units; "musts": its firm conditions, written as for musts; "with": the 1-based position of the other entry with which, per the request, this product has to work, fit, suit or be compatible, else null. A condition that only applies depending on how the other product turns out (written with "if") ties nothing: null}. A single product gives one entry.

Last check: every sentence of the request has to land somewhere - a must, the pick rule, a wish, a procedure, a limit, a hint, or plain story. When a sentence could be a rule, put it in procedures."""

# ---- 01a 4.3: reply listener (tag hear-shopper)
PROMPTS["listener"] = """Answer with one JSON object and nothing else. It holds twelve keys, in any order: pointer_refs, new_limits, duties, answer_back, musts_now, pace_only, gone_off_sale, new_cap, dropped_musts, dont_care, new_musts, gist. With them you decode the shopper's newest words for the shopping assistant. The newest words (SHOPPER'S NEWEST WORDS) may also be a store notice written as sentences. Part A explains how to read this shopper; part B defines each key.

PART A - READING THIS SHOPPER
A1. Report every change the words make; one line can carry several.
A2. When homely, figurative or indirect wording points to a value, treat it as a firm wish for that value, read literally and charitably. Exclude a value only when plainly told to avoid it.
A3. Emphasis is never part of the value. Firmness words ("only ever", "nothing other than", "sticking with") show how strongly the shopper feels and nothing else: record the bare value ("fabric: hemp"), never "hemp only". Write an exclusion only when mixes or other values are explicitly ruled out as well.
A4. A number with a comparison word (under, over, below, at least, no more than) is a limit; keep the unit. Any budget mentioned becomes new_cap and replaces the old one in either direction, even when it is put as good news.
A5. When the shopper ranks two wishes, the preferred one becomes a need. Plain wishes (would be nice; ideally) stay wishes.
A6. A need about a different property, however close (handle length next to total length; strap width next to width), is an extra condition that leaves the others in place. An earlier condition is dropped or replaced only if the words explicitly say so about that very property. A replacement puts the old one in dropped_musts and the new one in new_musts.
A7. Indifference ("I don't mind", "no preference", "anything works") puts the asked-about property in dont_care; when two were asked, only the FIRST one. It frees that property alone and leaves earlier needs untouched. If the assistant's message asked nothing (it only informed), the reply has no effect.
A8. This shopper also shows indifference by saying they could not follow the question ("that went over my head"; "too complicated for me"), as long as the question mentioned the product that owns the property: put that property in dont_care as well. With several products and a question that left the product unnamed, the property stays open and dont_care stays empty.
A9. With several products, write every need as "property (product): condition", taking the product from the words ("the lid of the stock pot" -> "lid (stock pot): ...").
A10. Conditions come from the shopper's own words alone; neither the assistant's question nor the example values inside it ever create one.
A11. A need stated by pointing at another listing takes the pointing words themselves as its value ("finish: unlike K's stool page"). Values the words attach to that other listing describe that listing, not ours: they are not needs, not drops and not freed properties.

PART B - THE KEYS
1. pointer_refs: one object per need whose value the words supply solely by referring to some other listing or page, describing that referred-to listing: {"property": the property; "seller": its store or brand, copied exactly from the words; "kind": its product type; "details": short strings in the words' own wording for any further values given for that listing; "relation": "same", "pairs" (the value the store pairs with theirs: a counterpart or one that must suit it), "differ" (must not equal theirs), "within", "at most" or "at least"; "tolerance": the number for "within", else null; "unit": that number's unit, else ""}. [] when none.
2. new_limits: null unless the words set a fresh limit on questions, messages, moves, steps or calls; then {"msgs_total": the total of messages allowed under the new limit, "msgs_left": how many may still be sent from now on, "steps_total": the total of steps allowed under the new limit, "steps_left": how many steps remain from now on}, each a number or null.
3. duties: procedural instructions the words give from now on (confirm before you buy; report the basket or its total to me; carry out or avoid some action; the manner of finishing or the closing words), one object each: {"text": in the words' own wording; "by_finish": true when it must be done before the order or decline goes out, else false; "via_message": true when it needs a message sent to the shopper, else false; "before_change": true when the shopper must be consulted before a cart line is changed, swapped or removed, else false}. [] when none. Never count needs, budgets or dropped conditions as duties.
4. answer_back: "" unless the shopper asks the assistant a question or wants something confirmed; then a short plain reply to send back, stating the change in substance (for instance confirming the new limit).
5. musts_now: every firm condition in force now, as a list of "property: condition". Start from REQUIREMENTS SO FAR; keep every entry that was neither dropped nor replaced; for a replaced one keep only the newest value; add each new need from the shopper's words. Additions come only from the shopper's words, never from the request's story unless the requirements so far already hold it. Leave out price, stock, quantity and category.
6. pace_only: true when the words merely ask the assistant to send less at a time (one question per message; smaller portions; the first part now and the rest later) without naming any value. Such words answer nothing, add nothing to new_musts, dont_care or dropped_musts, and none of their figures of speech is a value. Otherwise false. NOT pace_only: words that only say the shopper could not follow, or that ask for the question again in shorter or simpler words.
7. gone_off_sale: true when the shopper reports that an item can no longer be bought (sold out; gone; out of stock; a button that stopped working); else false.
8. new_cap: null, or the new upper limit on the order total as a number in the request's budget unit (take the figure as it is when it already uses that unit: "6400 cents" against a cents budget -> 6400).
9. dropped_musts: earlier conditions given up.
10. dont_care: properties the shopper declared unimportant.
11. new_musts: firm conditions that are new in these words, "property: condition" each, keeping value, limit, exclusion or closed list precisely as said. A comparison phrase (falls short of; nothing less than; at least) followed by a word value instead of a figure makes that word itself the value, spelled as the shopper spells it (a store field may literally hold a word such as "none": "nothing less than none" -> "property: none").
12. gist: one short sentence giving the shopper's intent."""

# ---- 01a 4.4: shared reading rules, embedded below into item, rank, audit and review
PROMPTS["rules"] = """READING RULES
(a) Kind: a listing inside the requested category is that kind, even when its title names another object (a container, case, refill, set or accessory). Judge it on the conditions, price and stock, nothing else.
(b) Scope: judge only the listed conditions. A use, gender, season, age group, size, quality, occasion or condition that nobody named never fails a listing.
(c) Money: prices are in the budget's own unit unless a record states another currency or unit (45 against a 3000-cent budget is 45 cents, well inside); rescale neither side.
(d) Figures: convert units first, then compare amounts. The edges of a limit are inside it; a range passes only when all of it passes. If the title has to serve as evidence because the property has no field, an exact wanted value passes once its words and figures occur in the title in unchanged sequence, extra text after them allowed (wanted "60 x 90 cm": "60 x 90 cm, set of 2" passes, "60 x 95 cm" fails).
(e) Wrong property: a field about something else (handle material vs material; strap width vs width) never stands in for a property that has a dedicated field in the category's listings.
(f) Where to look: each condition is settled by the field holding its property, compared word by word with the store's spelling there. The spelling of the field name does not matter (separators, prefixes, longer or shorter forms, common aliases). Deciding, when present: the field with the exact name, or the name with one word in front (length -> "blade length"); a value in either suffices. Names that differ only by separators (toe_cap, "toe cap") are one field, met by either spelling. Otherwise let the alias field settle it. Values shown only in the options list or the title count in just two situations: (i) not one field names the property, or (ii) a FIELD EVIDENCE line lets the title make up for a field missing from this listing. Descriptions never count.
(g) Matching words: ignore case, separators, singular or plural and spelling variants; one element of a list is enough; a longer, narrower value containing the wanted one matches (teal <- "deep teal"; linen <- "linen blend"). Equivalents (synonyms, abbreviations, related words, shades: aluminium ~ "alu") match only as long as every listing of the category leaves the exact wanted value out of that field; once one writes it, they stop matching (others write "matte", so "flat" fails). A value joined with "+" ("wool + nylon") fails one part when that field elsewhere holds the other part alone. FIELD EVIDENCE lines, where given, decide which case applies (code built them from this store's records). Also failing: another value of the property (manual vs electric); a word that only looks alike; a catch-all claiming every value unless its list includes the wanted one; a value that merely tends to come with or hints at the wanted one ("padded" is not "waterproof"). Only two stand-ins exist: a shade the trade sells under the wanted colour name, and a narrower value containing the wanted one."""

# ---- 01a 4.5: item check (tag check-item)
PROMPTS["item"] = """Answer with one JSON object and nothing around it: {"listing": "<the candidate's id>", "passes": true or false, "faults": ["<one short text per condition not met or not shown>"]}.
This is the final check before the assistant commits to the candidate. The errand pays 1 only when every condition holds. Judge only from the records shown here.
Give "passes": false in just two cases:
1. Under the rules below, the candidate's fields plainly break a listed condition (for example a FIELD EVIDENCE line says this listing does NOT meet one).
2. No source states a listed condition: no field holds it, and neither the options nor the title show it in the cases where they may count (the store has no field for it at all, or a FIELD EVIDENCE line allows the title to take a field's place).
Missing stock or price records are never, by themselves, a reason for false.

<<RULES>>
(h) The category satisfies any words that are part of its own name.
(i) Exclusions and closed lists stay binding.
(j) The candidate must be the requested kind. Where listings carry category labels, its label must fit the request's wording taken as a whole; a narrower label or a neighbouring one does not.
(k) Price x quantity must be within the budget, equality allowed. Records that show no stock fail.
(l) A wish marked as optional (ideally; would be nice) is not a condition.
(m) When records disagree (price, stock, ...), the newest one wins."""

# ---- 08 4: second-opinion section, passed as the item check's extra text under SECOND LOOK
PROMPTS["second"] = """SECOND OPINION NEEDED.
The review run before declining judged this listing to meet every requirement. Its reason, if it gave one: <<AUDIT_REASON>>
The earlier check of this listing said: <<EARLIER>>
Settle it from the records alone: a field meets a requirement when it expresses the requested value in other words, or as a shade, qualifier or alternative spelling of that very value; it fails when it expresses some other value."""

# ---- 01a 4.6: rank check (tag check-rank)
PROMPTS["rank"] = """Answer with a bare JSON object with two keys: {"beats": ["<id>", ...], "why": "<one sentence>"}, where beats holds up to 4 ids, strongest first.
The request wants the <<PRIORITY>>. The assistant has already chosen a listing to order (PICK ID). You get the conditions, the pick and the records of every listing met so far (LISTING CARDS).
Task: list the listings that can meet every condition while beating the pick on that priority (a lower price under cheapest; a stronger value of the measure under best).
Judging: price x quantity within the budget, and no record may show it out of stock. Leave out listings whose records show a failed condition, zero stock or a price over the budget.
Title figures on a ROW card ("80 x 120") do not say which dimension each figure is: never leave a ROW out because of them.

<<RULES>>"""

# ---- 01a 4.7: question planner (tag plan-questions for Asker, plan-product for Mender)
PROMPTS["planner"] = """Answer with a bare JSON object: {"asks": [{"key": "<property>", "next_key": "<another property, or empty>", "wording": "<the question>"}, ...]}, covering at most <<COUNT>> distinct properties, the strongest candidate first.

Why it matters: these questions go to a shopper whose unstated needs are each one value of one product property. Such a need is disclosed only when a message aims at exactly that property; the first question of a message gets the most dependable answer; asking about anything else earns only a polite "I don't mind".

1. NEVER ASK ABOUT
1.1 properties the request already settles (its conditions, price, stock, quantity); properties asked before; near-duplicates of each other (two style keys; two spellings of one key);
1.2 a property whose earlier question got a "did not understand" reply: no need sits there, so it is used up - never ask it again in other words; move on to another property or product;
1.3 brand, maker, product or model names, price, stock, delivery;
1.4 option or variant pickers, numbered twins of another key, catch-all keys (options, variants, description, title, name);
1.5 any property missing from the key list below. Exception: a story-hint property with no matching key may be asked under its plain name - unless the hint carries "[no key]" (enough listings of that product have been opened and none had such a field); then ask the key whose values differ in the way the hint's detail suggests, or skip it.

2. HOW A QUESTION IS WORDED
2.1 A step-1 question is one short, simple, friendly sentence of at most 15 words, asking which <property> the shopper wants for the <item kind>, optionally followed by up to two example values. Examples alone never draw out a need: the property itself must be named. Nothing may precede it - not a progress report, not instructions, not lists.
2.2 Name the property with the very key some listing of that product uses, spelled as written there.
2.3 ONE property of ONE product per question (the only exception is step 0's either-or between two wishes). A question must not cover two properties or two products: a reply reveals one need at most, and a question about two things may be ignored altogether.
2.4 next_key: the runner-up property of that SAME product (a genuinely different one, not key renamed), or empty; the assistant may add it to its last message.

3. SEVERAL PRODUCTS
Unstated needs may sit on any product. Share the questions out: no product gets a second one until each product has had one (skip products with no property left). Draw each product's properties from its own category's key list, format them "property (product)", and name the product inside the question ("Which blade material would you like for the knife?").

4. ORDER OF THE QUESTIONS
Step 0. First, plainly, any must-learn point (MUST LEARN FROM THE SHOPPER) not asked yet. For the budget, ask the most they would spend. For a choice between two wishes, name each value together with its property, along the lines of: if one listing cannot offer both, would they rather have <value> <property> or <value> <property>?
Step 1. Then properties that really occur in the listings' details, chosen this way:
a) Story first. Story hints, when given, lead; each one pairs a story detail with the property it concerns. As a rule, the unstated need lies in the property around which the story's specific details turn without ever giving its value.
b) Turn cues into properties. A cue shows which part or side of the item - and so which property - holds the need: something the user touches or presses, inspects up close or in photos, hears, measures or fits it against, reads on its tag, or the way it is made. Typical pairings (cue => property):
1. an activity, place or time of use or wear (a dance class; a workshop; early mornings) => style or type, meaning the property whose values name such uses;
2. what does the work inside it => the core component;
3. fastening or adjusting over and over => the closure or mechanism;
4. a present for someone else, or for someone barely known => the intended wearer or user (gender, age group);
5. how it feels against hand, skin or foot => the main material or fabric;
6. a body part or a space it must fit => that part's shape or size;
7. appearance and pictures => pattern, colour or the part on show;
8. tags, labels, small print => material, fabric or ingredients.
Never cues (background only): the weather; a job; a timetable; a place where nobody wears or uses the item.
c) The first question aims at the property those story details point to, using the listings' own name for it. The intended wearer or user is only one candidate: raise it when another person figures in the story, or when there is no other hint.
d) With no story cue left, first ask the property a buyer of this kind settles first of all: the property whose values divide the kind into sub-kinds no buyer could swap for one another in use (corded or cordless; the size it needs for a body or a space; the form it comes in; its base material; how it attaches or connects). Use the listings' own key. A key whose values differ only because every listing fills it with free text never qualifies. After that, descriptive beats logistic, in this order: 1 pattern, 2 material, 3 shape, 4 type or style variant, 5 closure, 6 length, 7 function - all ahead of 8 warranty, 9 condition, 10 pack size, 11 variant or size lists, 12 years.
e) The key list's order (widest coverage, most varied values) only breaks ties among candidates level under c) and d). An unstated need almost always points to a value carried by only a few listings of the category, so a key whose handful of read listings all show one single value is still strong (the needed value is probably one they lack). Among the listings read, coverage outweighs variety. A key flagged as figures with a unit (size, width, height, length, weight) deserves its own slot whenever there is one. Prefer keys present in most of that product's opened listings. With several products, prefer a product left with many fitting listings, since the need has to narrow that set."""

# ---- 01a 4.8: single-question add-on (Asker appends it to the planner)
PROMPTS["one_question"] = """THIS CALL: the shopper will hear only one question and answers it before anything else is chosen. <<LEFT>> question(s) are still available in all, and the plan should use all of them: the top property first, optionally one or two spares after it. Right after a need has come out, aim the next question at a new property and at a story detail not used before."""

# ---- 01a 4.7 variant / 05 4.2: focus block (Mender appends it to the planner)
PROMPTS["focus"] = """FOCUS: plan for the <<PRODUCT>> only. Of all the requested items it alone has had no question yet. Take its properties solely from its own key list (KEYS OF <<PRODUCT>>, the last section), and put first the key that the story's concrete details concern."""

# ---- 01a 4.9: before-decline audit (tag audit-decline)
PROMPTS["audit"] = """Answer with a bare JSON object with three keys: {"maybe_fit": [up to 5 ids that appear to meet every condition, strongest first], "open_first": [up to 6 ids seen only as rows that deserve opening before a decline], "why": "<one sentence>"}.
The assistant is about to tell the shopper that nothing fits, and such a decline is usually wrong. Test the listing records against the conditions with the rules below (the price within the budget; no record may show the item sold out). Listings under OPENED SINCE THE LAST AUDIT come first: test every one of their fields.

<<RULES>>

Row-only listings (ROW cards): their details were never read, so their fields are unknown. Such a row in the requested category (whatever object its title names) stays unchecked as long as its title, price and stock show no failed condition; title figures ("80 x 120") do not say which dimension each is, so they never prove a failure. Put these rows in open_first, the most promising first: titles naming a required value, then the requested kind with nothing against it."""

# ---- 01a 4.10: reference hop (tag trace-hop)
PROMPTS["hop"] = """Answer with one JSON object and nothing around it, with the keys source_id, absent_details, onward, read_value, partners.
A shopper wants the <<PROPERTY>> of their product copied from a different store listing, which the request describes as <<DESCRIPTION>>. The records below (CANDIDATE RECORDS) are candidates for that listing.
Two warnings:
1. Whatever a record's text claims, it is the target only if it shows the described details.
2. Never an onward listing: one named only as what the target is not ("not the grey one from store K"); a version, size or colour sold elsewhere; a listing cited for some other property.
The keys:
- source_id: the id of the single record that truly is the described listing (right seller, right kind, all named details visible). Judge details by meaning, so a figure given in another unit counts when the amount is equal (2 kg = 2000 g). null when none qualifies.
- absent_details: the named details the target does not show; [] when it shows them all. Shown = the value appears anywhere in the record (options, title, notes, or a field under another name).
- onward: normally null. When the target's own text points the reader to a DIFFERENT listing as the source of the <<PROPERTY>> (it says to take it from, go by, use, look at or refer to another listing), describe that listing as {"seller": its store or brand, copied as written; "kind": its product type; "details": [every value the text gives for it, short and verbatim - colour, size with unit, material]}.
- read_value: filled only if onward is null: the <<PROPERTY>> written just as the target's record has it - from its field for the property; without such a field, the value its notes give for it (fall back on options or title text only if the notes say nothing).
- partners: the values which, according to the target's text or a store rule it states, pair with, go along with or count as equal to that value; for a rule naming several pairs, only this value's partners (or the value itself where the rule pairs it with itself); [] when none."""

# ---- 01a 4.11 / 11 12.3: change review (tag review-change)
PROMPTS["review"] = """Answer with one lone JSON object with four keys and nothing around it:
1. "why": one sentence;
2. "order_now": the id to order at this point; null when no listing qualifies;
3. "cart_ok": null when the cart is empty; true when what the cart holds appears among the fits and is also the right thing to order (a basket that still fits stays); false when a cart line no longer fits, or when a cheapest or best rule now prefers another pick;
4. "fit_ids": ids, strongest first (cheapest first under a cheapest rule), of every OPENED listing that meets all the conditions valid now, is priced within the budget and has no record showing it sold out.
The terms of the errand have just moved (a store notice, a line from the shopper, or a refused finish): a new budget, a condition added or removed, or an item that disappeared. You get the request, the change (WHAT JUST CHANGED), the conditions now, the priority, the cart and a card for each listing met (a DETAILS card is a complete record; a ROW card is search-row data only). Judge each OPENED listing afresh against the terms valid NOW, using these rules:

<<RULES>>"""

# ---- 01a 4.12 / 11 15.4: finish check (tag check-finish)
PROMPTS["finish"] = """Answer with one JSON object and nothing around it, with two keys:
1. "done": false as long as some stated step is still owed; true once none is left;
2. "owed": the steps still owed, listed in running order, at most 2, each {"purpose": a few words saying which stated step it is, "tool": the exact name of the tool that performs it, "args": the arguments shaped exactly as that tool's schema requires}. For a message to the shopper, write out the whole text as one or two short plain sentences carrying the facts the step needs (for the basket: every cart line with id, count and price, followed by the total). Omit any step that no tool in [9] can perform.
This is the last look before the assistant ends the errand by ordering or declining. Steps can come from the request, from later shopper lines or from store notices; some of them become due before the finish or at it (checking back with the shopper first; reporting the basket or its total; a parting remark; one more look at something). Invent nothing: a step counts only if the request, the shopper or the store truly stated it, and only if some tool in [9] TOOLS FOR AN OPEN STEP can carry it out.
Method: go through [1] INSTRUCTIONS IN FORCE, the newest shopper and store words in [3] RECENT WORDS FROM SHOPPER OR STORE, and [7] ACTIVITY SINCE THE LATEST CART EDIT, then decide whether THIS finish ([6] THE FINISH ABOUT TO BE SENT) still owes a stated step. Basket steps done before the latest cart edit do not count as done.
Basket note: when this finish is an order and no message has reached the shopper since the cart's last edit, code sends the basket with its total by itself. Such a step belongs in owed only if a stated step needs more or something different (answering a question; waiting for a yes)."""

# ---- 05 4.1: meaning check for wording-only mismatches (tag match-wording)
PROMPTS["meaning"] = """Answer with exactly one JSON object and nothing else, one entry per numbered line: {"verdicts": [{"line": <line number>, "same_value": true or false}, ...]}. When in doubt, answer false.
An order would be turned down only because some listings word a wanted value differently from the shopper. For each numbered line under LINES, decide whether the listing's own field wording expresses the wanted value.
true: it does - the same meaning in other words, another spelling or phrasing, or a narrower, more exact form of that value.
false: the wording is a different value of the same property."""

# ---- 01a 4.13: fixed decline sentence (names no listing); 06 4.1 / 07 3.9: fixed order phrase
PROMPTS["decline_line"] = "I am sorry, but this store has nothing that meets every one of your requirements, so I will not buy anything."
PROMPTS["order_line"] = "Place the order for the current cart."

PROMPTS.update({k: PROMPTS[k].replace("<<RULES>>", PROMPTS["rules"]) for k in ("item", "rank", "audit", "review")})



# ------------------------------------------------------------ p02_kit.py
class kit:
    """Pure text, JSON, record and matching helpers; no store or model calls (01b, 01a 5, 07 3.12/3.24, 06 6.5, 08 3.19)."""
    LITERALS = set()
    TWINS = {}
    _corpus = {"base": None, "docs": 0, "df": {}}
    # True: the link-word test reads the errand corpus (01a 7.1); False: the grammar pattern (same behaviour as built)
    LINK_BY_CORPUS = False

    # ---- grammar and structure patterns (lowercased text unless noted)
    # function words by class: articles/determiners, pronouns, conjunctions and negation, prepositions, copulas
    _GLUE = re.compile(r"(?:a|an|the|this|that|all|any|it|its|you|your|our|and|or|not|no"
                       r"|in|on|to|by|at|as|of|for|with|from|is|are|be)")
    _STOPV = re.compile(r"(?:a|an|and|the|of)")
    _VOWEL = re.compile(r"[aeiouy]")
    _MARKS = re.compile(r"\s*\((?:\s*stated\s*|revealed[^)]*)\)", re.I)
    _EMPH = re.compile(r"\b(?:only|just|exactly|strictly)\b", re.I)
    _FLAG = re.compile(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*")
    _FLAG_S = re.compile(r"\W*(?:yes|no|true|false|required|included|needed)\W*")
    _BOUND_P = (r"at least|no more than|at most|no less than|more than|less than|up to|or more|or less|or larger"
                r"|or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|than")
    _BOUND = re.compile(r"\b(?:" + _BOUND_P + r")\b", re.I)
    _NEGCMP = re.compile(r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b")
    _NEGER = re.compile(r"\b(?:no|not)\s+[a-z]*er(?=\s+than\b)")
    _NEGER_ANY = re.compile(r"\b(?:no|not)\s+[a-z]*er\b")
    _LINKS = re.compile(r"\b(?:" + _BOUND_P + r"|exactly|only|either|neither|nor|plus)\b", re.I)
    _EXCL = re.compile(r"\bno\b(?!-)|\b(?:not|without|nothing|never|neither|nor|except|excluding|anything\s+but"
                       r"|other\s+than|rather\s+than|instead\s+of)\b|\bavoid\w*")
    _CMP = re.compile(r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+less\s+than|nothing\s+other\s+than|nothing\s+short\s+of|nothing\s+but"
                      r"|no\s+less\s+than|no\s+worse\s+than|at\s+least|less\s+than|more\s+than|better\s+than|worse\s+than|exactly|only"
                      r"|just)\b", re.I)
    _CLAUSE = re.compile(r"[,;/&()]|\bor\b|\band\b")
    _FIG = re.compile(r"\d+(?:\.\d+)?")
    _FIGRUN = re.compile(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*")
    # reference families (01b 3.13)
    _REF = re.compile(r"\b(?:as\s+shown|shown\s+on|same\s+as|page|sold\s+by|redirect\w*|refer(?:s|red|ring)?"
                      r"|listing\s+points|points\s+you)\b")
    _PTR = re.compile(r"\b(?:within|at\s+most|at\s+least|no(?:t\s+any)?\s+(?:more|less)\s+than"
                      r"|same\s+as|as\s+shown|the\s+value|the\s+one)\b")
    _DIFF = re.compile(r"\b(?:apart\s+from|distinct\s+from|different\s+(?:from|to|than)|differ(?:s|ent|ing)?\s+from"
                       r"|other\s+than|unlike|not\s+the\s+same(?:\s+as)|anything\s+but)\b")
    _PAIR = re.compile(r"\b(?:counterparts?|opposite\s+numbers?|partner\s+value|paired\s+value|compatible\s+with)\b")
    _ATMOST = re.compile(r"\b(?:no\s+more\s+than|at\s+most|up\s+to|or\s+less|or\s+lower|or\s+smaller)\b")
    _ATLEAST = re.compile(r"\b(?:at\s+least|no(?:t\s+any)?\s+less\s+than|or\s+more|or\s+higher|or\s+larger)\b")
    _WITHIN = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z]+|%)?")
    _POINTING = re.compile(r"as\s+shown|same\s+as|the\s+page|listing|sold\s+by|redirect|refer")
    _SELLER = re.compile(r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by")
    # record structure (field-name shapes)
    _JUNK_WORD = re.compile(r"\b(?:brand|shop|seller|store|vendor|manufacturer|categor(?:y|ies)|department|title"
                            r"|headline|heading|listing|description|overview|price|cost|amount|availability|sku|url)\b")
    _JUNK_WHOLE = re.compile(r"(?:name|notes?|stock|in\s+stock|on\s+hand|images?|links?|make|maker|shelf|ids?)")
    _PRICE_IN = re.compile(r"amount|fee|charge|pay|rate|value")
    _PRICE_OUT = re.compile(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|width"
                            r"|height|length|found|match")
    _STOCK_IN = re.compile(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store")
    _TITLE_IN = re.compile(r"headline|name|label|caption")
    _TITLE_OUT = re.compile(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id")
    _SHELF = re.compile(r"categor|shelf|department|aisle")
    _ONHAND = re.compile(r"on.?hand")
    _GONE = re.compile(r"\b(?:out|sold|none|unavailable|not|no|false)\b")
    _LATE = re.compile(r"back.?order|pre.?order|discontinu")
    _ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}")
    _CODE = re.compile(r"[A-Za-z0-9_\-]+")
    _FAULT = re.compile(r"error|fault|refus|reject|denied|invalid|fail")
    _DECLINE = re.compile(r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdr[ae]w|no.?(?:purchas|order|buy)"
                          r"|nothing|without")
    _ORDER = re.compile(r"commit|buy|order|purchas|check.?out|pay|confirm|complet|place|accept|proceed")
    _PUT = re.compile(r"(?:add|put|place|insert|includ|hold|keep|append|increas)")
    _DROP = re.compile(r"(?:remov|delet|clear|drop|discard|eras|decreas|reduc|subtract)")
    _VTOK = re.compile(r"[a-z0-9]*\d\.\d+|[a-z0-9]+")
    _RTOK = re.compile(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*")
    _FUSE = re.compile(r"(\d)(\s+|\s*-\s*)([a-z]{1,8})\b")
    _CODED = re.compile(r"(?<![a-z\d])[a-z]+\d+$")
    _RFIG = re.compile(r"(\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b")
    _CAP = re.compile(r"(?<![A-Za-z'])([A-Z][a-z]{3,})(?:'s)?(?![A-Za-z])")
    _VERB = re.compile(r"\s+(?:sold|made|offered|listed)\s+by\s*$", re.I)
    _ART = re.compile(r"\b(?:the|for|a|an)\s+", re.I)
    _KIND = re.compile(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|\s*[,;]|$)", re.I)
    _DASH = re.compile(r"\s[\u2014\u2013]\s|\s-\s|\s(?:shows|show|says|lists)\b", re.I)
    _DSPLIT = re.compile(r"[,;]|\b(?:that\s+(?:is|are|has|have|comes\s+as|comes|come)|and|with|which|in|at)\b", re.I)
    _DTOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.%:/&'\-]*")

    # ------------------------------------------------------------------ state (01b 2)
    @staticmethod
    def reset():
        kit.LITERALS.clear()
        kit.TWINS.clear()
        kit._corpus = {"base": None, "docs": 0, "df": {}}

    @staticmethod
    def add_literal(value):
        v = kit.flat_value(value)
        if v:
            kit.LITERALS.add(v)

    @staticmethod
    def add_twin(a, b):
        a, b = kit.low(a), kit.low(b)
        if a and b and a != b:
            kit.TWINS.setdefault(a, set()).add(b)
            kit.TWINS.setdefault(b, set()).add(a)

    @staticmethod
    def feed_words(text):
        ws = re.findall(r"[a-z]+", kit.low(text))
        c = kit._corpus
        if c["base"] is None:
            base = {}
            for w in ws:
                base[w] = base.get(w, 0) + 1
            c["base"] = base
        c["docs"] += 1
        for w in set(ws):
            c["df"][w] = c["df"].get(w, 0) + 1

    @staticmethod
    def _corpus_link(w):
        # 01a 7.1: short particles, or short words frequent in the request and the store's own tool texts
        if not w.isalpha():
            return False
        if len(w) <= 2:
            return bool(kit._VOWEL.search(w))
        return len(w) <= 4 and (kit._corpus["base"] or {}).get(w, 0) >= 3

    # ------------------------------------------------------------------ text basics (01b 0, 3.1-3.4)
    @staticmethod
    def low(v):
        return str(v).lower() if v else ""

    @staticmethod
    def low_plain(v):
        return "" if v is None else str(v).lower()

    @staticmethod
    def is_link(word):
        w = kit.low(word)
        if not w:
            return False
        if kit.LINK_BY_CORPUS:
            return kit._corpus_link(w)
        return bool(kit._GLUE.fullmatch(w))

    @staticmethod
    def content_words(text):
        return [w for w in re.findall(r"[a-z]{3,}", kit.low_plain(text)) if not kit.is_link(w)]

    @staticmethod
    def cut(text, limit, mark=Tune.Text.CUT_MARK):
        t = "" if text is None else str(text)
        return t if len(t) <= limit else t[:limit] + mark

    @staticmethod
    def render(value, limit=Tune.Text.VIEW):
        try:
            t = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            t = str(value)
        return kit.cut(t, limit, Tune.Text.CUT_MARK)

    @staticmethod
    def clip_deep(value, limit, depth=0):
        if depth > Tune.Text.NEST:
            return None
        if isinstance(value, str):
            return kit.cut(value, limit, "...")
        if isinstance(value, dict):
            return {k: kit.clip_deep(v, limit, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [kit.clip_deep(v, limit, depth + 1) for v in value]
        return value

    @staticmethod
    def json_from(text):
        if not isinstance(text, str):
            return None
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[A-Za-z]*", "", t).strip()
            if t.endswith("```"):
                t = t[:-3].strip()
        try:
            o = json.loads(t)
            if isinstance(o, dict):
                return o
        except Exception:
            pass
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            try:
                o = json.loads(t[i:j + 1])
                if isinstance(o, dict):
                    return o
            except Exception:
                pass
        return None

    @staticmethod
    def args_from(raw):
        if isinstance(raw, dict):
            return raw
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return {}
        return kit.json_from(raw) if isinstance(raw, str) else None

    @staticmethod
    def is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    @staticmethod
    def is_plain(v):
        return isinstance(v, (str, int, float)) and not isinstance(v, bool)

    @staticmethod
    def lengthy_texts(value, start=0):
        out = []

        def walk(v, d):
            if isinstance(v, str):
                if len(v.split()) > 3:
                    out.append(v)
            elif isinstance(v, list) and d < 4:
                for x in v[:10]:
                    walk(x, d + 1)
        walk(value, start)
        return out

    @staticmethod
    def rec_strings(rec):
        got = []
        for v in (rec.values() if isinstance(rec, dict) else ()):
            if isinstance(v, dict):
                got.extend(v.values())
            elif isinstance(v, str):
                got.append(v)
        return [x.strip() for x in got if isinstance(x, str) and x.strip()]

    @staticmethod
    def visit(value, fn, top=6, start=0):
        def walk(v, d):
            if d > top:
                return
            if isinstance(v, dict):
                fn(v)
                for x in list(v.values()):
                    walk(x, d + 1)
            elif isinstance(v, list):
                for x in v:
                    walk(x, d + 1)
        walk(value, start)

    # ------------------------------------------------------------------ small word tools (07 3.24, 08 3.19)
    @staticmethod
    def words(text, pattern=None):
        return re.findall(pattern or r"[a-z0-9]+", kit.low_plain(text))

    @staticmethod
    def stem(word):
        w = str(word)
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w

    @staticmethod
    def near(a, b, mode=""):
        a, b = str(a), str(b)
        if mode in ("equal", "lead") and (re.search(r"\d", a) or re.search(r"\d", b)):
            if a.lstrip("0") == b.lstrip("0"):
                return True
            return mode == "lead" and a.isdigit() and b.startswith(a) and not b[len(a):len(a) + 1].isdigit()
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        return len(s) >= 3 and l.startswith(s)

    @staticmethod
    def uncamel(text):
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", "" if text is None else str(text))

    @staticmethod
    def head_split(line):
        line = "" if line is None else str(line)
        front, _, value = line.partition(":")
        tags = re.findall(r"\(([^()]*)\)", front)
        return front, front.split("(")[0].strip(), tags[-1].strip() if tags else "", value

    @staticmethod
    def tags_of(name):
        return [x.strip() for x in re.findall(r"\(([^()]*)\)", "" if name is None else str(name))]

    @staticmethod
    def strip_marks(text):
        return kit._MARKS.sub("", "" if text is None else str(text))

    @staticmethod
    def flat_value(text):
        return " ".join(re.findall(r"[a-z0-9]+", kit.strip_marks(kit.low_plain(text))))

    @staticmethod
    def name_head(name):
        return ("" if name is None else str(name)).split("(")[0].rstrip()

    # ------------------------------------------------------------------ property names (01b 3.5)
    @staticmethod
    def _runs(name, pat):
        return re.findall(pat, kit.uncamel(kit.name_head(name)).lower())

    @staticmethod
    def is_junk_key(name):
        j = " ".join(kit._runs(name, r"[a-z0-9]+"))
        return bool(kit._JUNK_WORD.search(j) or kit._JUNK_WHOLE.fullmatch(j))

    @staticmethod
    def is_numbered_twin(name):
        r = kit._runs(name, r"[a-z0-9]+")
        return len(r) >= 2 and len(r[-1]) == 1 and r[-1].isdigit() and r[-2].isalpha()

    @staticmethod
    def property_terms(name):
        return [w[:-1] if len(w) >= 4 and w.endswith("s") and not w.endswith("ss") else w
                for w in kit._runs(name, r"[a-z]+")]

    @staticmethod
    def property_slug(name):
        return "".join(kit.property_terms(name))

    @staticmethod
    def within_edits(a, b, k):
        a, b = str(a), str(b)
        if abs(len(a) - len(b)) > k:
            return False
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            if min(cur) > k:
                return False
            prev = cur
        return prev[-1] <= k

    @staticmethod
    def property_match(a, b):
        ka, kb = kit.property_slug(a), kit.property_slug(b)
        if not ka or not kb:
            return False
        if ka == kb:
            return True
        s, l = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
        if len(s) < 4:
            return False
        if s[:3] == l[:3] and kit.within_edits(s, l, 2 if len(s) > 5 else 1):
            return True
        return len(s) > 4 and l.startswith(s)

    @staticmethod
    def near_spelling(short, long):
        s, l = str(short), str(long)
        d = len(l) - len(s)
        if d == 0:
            df = [i for i in range(len(s)) if s[i] != l[i]]
            return (len(df) == 2 and df[1] == df[0] + 1 and s[df[0]] == l[df[1]] and s[df[1]] == l[df[0]])
        if d not in (1, 2):
            return False

        def go(w, n):
            if n == 0:
                return w == s
            for i, c in enumerate(w):
                if c in "aeiouy" or (i > 0 and w[i - 1] == c) or (i + 1 < len(w) and w[i + 1] == c):
                    if go(w[:i] + w[i + 1:], n - 1):
                        return True
            return False
        return go(l, d)

    @staticmethod
    def exact_property(a, b):
        wa, wb = kit.property_terms(a), kit.property_terms(b)
        ka, kb = "".join(wa), "".join(wb)
        if not ka or not kb:
            return False
        if ka == kb:
            return True
        s, l = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
        if len(s) > 3 and s[:3] == l[:3] and kit.near_spelling(s, l):
            return True
        lw, sw = (wa, wb) if len(wa) > len(wb) else (wb, wa)
        if len(lw) == len(sw) + 1 and lw[:-1] == sw:
            return len(lw[-1]) < 4 and sum(len(w) for w in sw) > 4
        return False

    @staticmethod
    def vowel_twins(a, b):
        a, b = kit.low(a), kit.low(b)
        if len(a) != len(b) or len(a) < 4 or not (a.isalpha() and b.isalpha()) or a[:2] != b[:2]:
            return False
        df = [i for i in range(len(a)) if a[i] != b[i]]
        return len(df) == 1 and a[df[0]] in "aeiouy" and b[df[0]] in "aeiouy"

    @staticmethod
    def spelling_twins(a, b):
        a, b = kit.low(a), kit.low(b)
        if a == b:
            return False
        if kit.vowel_twins(a, b):
            return True
        s, l = (a, b) if len(a) <= len(b) else (b, a)
        return len(s) > 3 and a[:3] == b[:3] and a.isalpha() and b.isalpha() and kit.near_spelling(s, l)

    # ------------------------------------------------------------------ value words and matching (01b 3.6-3.9)
    @staticmethod
    def value_words(text):
        return [w for w in kit._VTOK.findall(kit.low_plain(text)) if not kit._STOPV.fullmatch(w)]

    @staticmethod
    def near_word(w, f):
        w, f = str(w), str(f)
        if not (w.isalpha() and f.isalpha()):
            return False
        return (kit.kind_match(w, f) or f in kit.TWINS.get(w, ())
                or (len(w) >= 6 and len(f) >= 6 and w[:2] == f[:2] and kit.within_edits(w, f, 1)))

    @staticmethod
    def in_pool(w, pool):
        return any(p == w or kit.near_word(w, p) for p in pool)

    @staticmethod
    def alternatives(value):
        t = kit._EMPH.sub(" ", kit.strip_marks(kit.low_plain(value)).replace('"', " "))
        out = []
        for piece in re.split(r"\bor\b", t):
            ws = kit.value_words(piece)
            if ws:
                out.append(ws)
        return out

    @staticmethod
    def alts_match(alts, pool):
        pool = [str(p) for p in pool or []]
        if not pool:
            return False
        dig = any(re.search(r"\d", p) for p in pool)
        for alt in alts or []:
            if all(kit.in_pool(w, pool) for w in alt):  # an empty alternative passes, as built
                return True
            g = "".join(alt)
            if len(alt) >= 2 or dig or re.search(r"\d", g):
                for i in range(len(pool)):
                    pos, j = 0, i
                    while j < len(pool) and pool[j] and g.startswith(pool[j], pos):
                        pos += len(pool[j])
                        j += 1
                        if pos == len(g):
                            return True
        return False

    @staticmethod
    def writes(field_text, value):
        return kit.alts_match(kit.alternatives(value), kit.value_words(field_text))

    @staticmethod
    def _lone(fig, text):
        return re.search(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", text)

    @staticmethod
    def touches(value, text):
        tw, raw = kit.value_words(text), kit.low_plain(text)
        for alt in kit.alternatives(value):
            for w in alt:
                if kit.in_pool(w, tw):
                    return True
                if re.fullmatch(r"\d+(?:\.\d+)?", w) and kit._lone(w, raw):
                    return True
                if len(w) >= 3 and any(len(x) >= 3 and x[:3] == w[:3] for x in tw):
                    return True
        return False

    @staticmethod
    def list_parts(text):
        t = ("" if text is None else str(text)).strip().lower()
        parts = [p.strip() for p in re.split(r"[,;|]", t)]
        parts = [p for p in parts if p]
        return parts or ([t] if t else [])

    @staticmethod
    def plus_sides(part):
        sides = [x.strip() for x in str(part or "").split("+") if x.strip()]
        return sides if len(sides) >= 2 else []

    @staticmethod
    def number_spelled(fig, low_request):
        fig, t = str(fig), str(low_request or "")
        for m in re.finditer(r"(?<![\d.])" + re.escape(fig) + r"(?!\d|\.\d)", t):
            if "." in fig:
                return True
            if len(re.match(r"\s*([a-z]*)", t[m.end():]).group(1)) < 4:
                return True
        return False

    @staticmethod
    def is_positive(value, prop=""):
        t = kit.low_plain(value).strip()
        if kit._FLAG.fullmatch(t):
            return True
        if kit.LITERALS and kit.flat_value(value) in kit.LITERALS:
            return True
        t = kit._NEGCMP.sub(" ", t)
        t = kit._NEGER.sub(" ", t)
        if re.search(r"\d", t):
            t = kit._NEGER_ANY.sub(" ", t)  # by ending only: "no rubber" also goes when a digit is present
        if any(len(kit.content_words(c)) > 3 for c in kit._CLAUSE.split(kit.strip_links(t))):
            return False
        m = kit._EXCL.search(t)
        if not m or kit.content_words(t[:m.start()]):
            return True
        tail = kit.content_words(t[m.end():])
        own = set(kit.property_terms(prop)) | set(re.findall(r"[a-z]+", kit.low(prop)))
        if not tail or not own:
            return False
        return all(w in own or w.rstrip("s") in own for w in tail)

    @staticmethod
    def _in_order(run, end):
        figs = kit._FIG.findall(run)
        return r"(?<![\d.])" + r"\D{0,4}".join(re.escape(f) for f in figs) + end

    @staticmethod
    def seen_in(value, samples):
        s = kit.low(samples)
        if not s.strip():
            return False
        v = kit.strip_links(kit.low_plain(value))
        for w in kit.content_words(v):
            if re.search(r"\b" + re.escape(w[:-1] if len(w) >= 5 else w), s):
                return True
        for run in kit._FIGRUN.findall(v):
            if re.search(kit._in_order(run, r"(?![\d.])"), s):
                return True
        return False

    # ------------------------------------------------------------------ phrase families (01b 5)
    @staticmethod
    def has_bound(text):
        t = kit.low(text)
        return bool(kit._BOUND.search(t) or kit._NEGCMP.search(t))

    @staticmethod
    def _squash(t):
        return re.sub(r"\s+", " ", t).strip()

    @staticmethod
    def strip_links(text):
        return kit._squash(kit._LINKS.sub(" ", "" if text is None else str(text)))

    @staticmethod
    def has_exclusion(text):
        return bool(kit._EXCL.search(kit.low(text)))

    @staticmethod
    def strip_comparison(text):
        return kit._squash(kit._CMP.sub(" ", "" if text is None else str(text)))

    @staticmethod
    def is_flag_value(text, story=False):
        return bool((kit._FLAG_S if story else kit._FLAG).fullmatch(kit.low_plain(text).strip()))

    @staticmethod
    def strip_emphasis(text):
        return kit._squash(kit._EMPH.sub(" ", "" if text is None else str(text)))

    # ------------------------------------------------------------------ kinds and product names (01b 3.10)
    @staticmethod
    def kind_terms(text):
        return {w.rstrip("s") for w in kit.content_words(text)} - {""}

    @staticmethod
    def kind_bits(text):
        t = kit.low_plain(text).replace("\u2019", "'").replace("\u2018", "'")
        return kit.content_words(re.sub(r"'s\b", "", t))

    @staticmethod
    def word_variants(word):
        w = str(word)
        f = {w, w + "s", w + "es"}
        if w.endswith("y"):
            f.add(w[:-1] + "ies")
        if w.endswith("fe"):
            f.add(w[:-2] + "ves")
        if w.endswith("f"):
            f.add(w[:-1] + "ves")
        if w.endswith("ies"):
            f.add(w[:-3] + "y")
        if w.endswith("ves"):
            f |= {w[:-3] + "f", w[:-3] + "fe"}
        if w.endswith("es"):
            f.add(w[:-2])
        if w.endswith("s"):
            f.add(w[:-1])
        return {x for x in f if len(x) > 2}

    @staticmethod
    def kind_match(a, b):
        a, b = str(a), str(b)
        return a == b or a in kit.word_variants(b) or b in kit.word_variants(a)

    @staticmethod
    def kind_present(word, pool):
        return any(kit.kind_match(word, p) for p in pool or ())

    @staticmethod
    def kinds_inside(words, pool):
        pool = list(pool or ())
        return all(kit.kind_present(w, pool) for w in words or ())

    @staticmethod
    def kind_overlap(own, other, shared):
        other, shared = list(other or ()), list(shared or ())
        return sum((0.5 if kit.kind_present(w, shared) else 1.0) for w in own or () if kit.kind_present(w, other))

    @staticmethod
    def name_regex(name):
        parts = []
        for t in kit.kind_bits(name):
            fs = sorted(kit.word_variants(t), key=lambda x: (-len(x), x)) or [t]
            parts.append("(?:" + "|".join(re.escape(x) for x in fs) + r")(?:'s\b|'|\b)")
        return r"[\s'\-]+".join(parts)

    # ------------------------------------------------------------------ planner hints and story guesses (01b 3.11-3.12)
    @staticmethod
    def split_hint(hint):
        try:
            h = str(hint)
            if ":" not in h:
                return hint
            front, detail = h.split(":", 1)
            if "/" not in front:
                return hint
            m = re.search(r"\(([^()]*)\)\s*$", front)
            if m:
                front = front[:m.start()]
            names = [n.strip() for n in front.split("/") if n.strip()]
            if len(names) < 2:
                return hint
            label = names[0] + (" (" + m.group(1).strip() + ")" if m else "")
            return label + ":" + detail + " (or " + ", ".join(names[1:]) + ")"
        except Exception:
            return hint

    @staticmethod
    def head_value(detail, request, product):
        try:
            d = kit.low(detail)
            m = re.match(r"\s*([a-z](?:[a-z\-]*[a-z])?)", d)
            if not m:
                return ""
            w, rest = m.group(1), d[m.end():]
            nx = re.match(r"\s+([a-z]+)\b", rest)
            if not (re.match(r"\s*(?:[,;.()]|$)", rest) or (nx and kit.is_link(nx.group(1)))):
                return ""
            pw = kit.content_words(product)
            if not pw or len(re.sub(r"[^a-z]", "", w)) < 3 or kit.is_link(w):
                return ""
            if w.rstrip("s") in {p.rstrip("s") for p in pw}:
                return ""
            head = pw[0].rstrip("s")
            toks = re.findall(r"[a-z](?:[a-z'\-]*[a-z])?", kit.low(request))
            for i in range(len(toks) - 1):
                if toks[i] == w and toks[i + 1].split("'")[0].rstrip("s") == head:
                    return w
            return ""
        except Exception:
            return ""

    @staticmethod
    def _covered(word, told, roots):
        root = (word[:-2] if word.endswith("es") and len(word) > 4
                else word[:-1] if word.endswith("s") and len(word) > 3 else word)
        for t in told:
            if t == word or (t.startswith(root) and len(t) - len(root) <= 3):
                return True
            if root.startswith(t) and len(t) >= max(4, len(root) - 2):
                return True
            if len(word) >= 4 and (t.startswith(word) or (len(t) >= 4 and t[0] == word[0]
                                                         and kit.within_edits(t, word, 1))):
                return True
        return any(len(r) >= 3 and root.startswith(r) and len(root) - len(r) <= 3 for r in roots)

    @staticmethod
    def tale_hint(line, request, ref_lines):
        line = "" if line is None else str(line)
        name, _, val = line.partition(":")
        want = val.strip().lower()
        refs = [str(r) for r in ref_lines or ()]
        if not want or line in refs or line.strip() in [r.strip() for r in refs] or kit._POINTING.search(want):
            return []
        if kit._FLAG_S.fullmatch(want):
            want = kit.name_head(name).lower()
        want = kit.strip_links(want)
        told = kit.low(request)
        figs = kit._FIG.findall(want)
        if figs and all(kit.number_spelled(f, told) for f in figs):
            return []
        tw = re.findall(r"[a-z]+", told)
        roots = [w[:-1] if len(w) >= 4 and w.endswith("s") else w for w in tw]
        cws = kit.content_words(want)
        unseen = [w for w in cws if not kit._covered(w, tw, roots)]
        if len(cws) < 3 and any(len(w) > 3 and w not in unseen for w in cws):
            return []
        for run in kit._FIGRUN.findall(want):
            if len(kit._FIG.findall(run)) >= 2 and not re.search(kit._in_order(run, r"(?!\d)"), told):
                unseen.append(run)
        return unseen

    # ------------------------------------------------------------------ references (01b 3.13-3.16)
    @staticmethod
    def is_pointer(text):
        return bool(kit._REF.search(kit.low(text)))

    @staticmethod
    def has_pointer_phrase(text):
        return bool(kit._PTR.search(kit.low(text)))

    @staticmethod
    def is_differ(text):
        return bool(kit._DIFF.search(kit.low(text)))

    @staticmethod
    def is_pair(text):
        return bool(kit._PAIR.search(kit.low(text)))

    @staticmethod
    def is_seller_field(name):
        return bool(kit._SELLER.search(kit.low(name)))

    @staticmethod
    def _plural_cut(tok, guard=True):
        if len(tok) < (4 if guard else 3) or not tok.endswith("s") or tok.endswith("ss"):
            return tok
        if guard and tok[-2].isdigit():
            return tok
        return tok[:-2] if re.search(r"(?:ches|shes|xes|zes)$", tok) else tok[:-1]

    @staticmethod
    def ref_tokens(text):
        t = kit.low(text)

        def fuse(m):
            if kit.is_link(m.group(3)) or kit._CODED.search(t[max(0, m.start(1) - 15):m.start(1) + 1]):
                return m.group(0)
            return m.group(1) + m.group(3)
        t = kit._FUSE.sub(fuse, t)
        out = []
        for tok in kit._RTOK.findall(t):
            if kit.is_link(tok) or (len(tok) == 1 and not tok.isdigit()):
                continue
            out.append(kit._plural_cut(tok))
        return out

    @staticmethod
    def ref_figures(text):
        out = set()
        try:
            for m in kit._RFIG.finditer(kit.low(text).replace(",", "")):
                if kit.is_link(m.group(2)):
                    continue
                unit = kit._plural_cut(m.group(2), False)
                for n in m.group(1).split("/"):
                    out.add((n, unit))
        except Exception:
            pass
        return out

    @staticmethod
    def name_key(text):
        return re.sub(r"[^a-z0-9]", "", re.sub(r"['\u2019]s\b", "", kit.low(text)))

    @staticmethod
    def relation_of(value):
        t = kit.low(value)
        if kit._PAIR.search(t):
            return "pairs", None, ""
        if kit._DIFF.search(t):
            return "differ", None, ""
        m = kit._WITHIN.search(t)
        if m:
            return "within", float(m.group(1)), m.group(2) or ""
        if kit._ATMOST.search(t):
            return "at most", None, ""
        if kit._ATLEAST.search(t):
            return "at least", None, ""
        return "same", None, ""

    @staticmethod
    def parse_listing(desc, seller_ok=None):
        try:
            t = str(desc or "")
            m = re.search(r"\(\s*or\b", t, re.I)
            if m:
                t = t[:m.start()]
            lead = len(t) - len(t.lstrip())
            sel = None
            for c in kit._CAP.finditer(t):
                if c.start() == lead or (seller_ok is not None and not seller_ok(c.group(1))):
                    continue
                sel = c
                break
            if sel is None:
                return None
            before, after = t[:sel.start()], t[sel.end():]
            kind, rest = "", None
            vb = kit._VERB.search(before)
            if vb:
                arts = list(kit._ART.finditer(before[:vb.start()]))
                if arts:
                    kind = before[arts[-1].end():vb.start()].strip()
                    if kind:
                        rest = after
            if rest is None:
                km = kit._KIND.match(after)
                if km:
                    kind, rest = km.group(1).strip(), after[km.end():]
                else:
                    kind, rest = "", after
            dm = kit._DASH.search(rest)
            if dm:
                rest = rest[:dm.start()]
            dets = []
            for piece in kit._DSPLIT.split(rest):
                ws = [w for w in kit._DTOK.findall(piece or "") if not kit.is_link(w.lower())]
                if ws:
                    dets.append(" ".join(ws))
            return sel.group(1), kind[:80], dets[:6]
        except Exception:
            return None

    @staticmethod
    def names_prop(text, prop, exact=False):
        want = re.findall(r"[a-z0-9]+", kit.uncamel(kit.name_head(prop)).lower())
        seen = re.findall(r"[a-z0-9]+", kit.low(text))
        if not want or len(" ".join(want)) < 3:
            return False

        def eq(a, b):
            if a == b:
                return True
            if exact:
                return False
            return (len(b) >= 3 and a in (b + "s", b + "es")) or (len(a) >= 3 and b in (a + "s", a + "es"))
        n = len(want)
        return any(all(eq(seen[i + j], want[j]) for j in range(n)) for i in range(len(seen) - n + 1))

    @staticmethod
    def after_prop(source, prop):
        try:
            ws = [w for w in re.findall(r"[a-z0-9]+", kit.name_head(prop).lower()) if not kit.is_link(w)]
            if not ws:
                return ""
            src = "" if source is None else str(source)
            m = re.search(r"\b" + r"\W+".join(re.escape(w) for w in ws) + r"\b", src, re.I)
            if not m:
                return ""
            tail = src[m.end():][:500]
            c = re.search(r"\(\s*or|(?<=[.!?])\s", tail, re.I)
            return tail[:c.start()] if c else tail
        except Exception:
            return ""

    # ------------------------------------------------------------------ the record lens (01b 3.17)
    @staticmethod
    def lens(obs):
        try:
            copy = json.loads(json.dumps(obs))
        except Exception:
            return obs
        kit.visit(copy, kit._fill)
        return copy

    @staticmethod
    def _fill(rec):
        if not any(isinstance(v, str) and "id" in str(k).lower() and kit.looks_like_id(v) for k, v in rec.items()):
            return
        keys = {}
        for k, v in rec.items():
            keys[str(k).lower()] = v
        if not any(("price" in k or "cost" in k) and kit.is_num(v) for k, v in keys.items()):
            c = [v for k, v in keys.items() if kit.is_num(v) and kit._PRICE_IN.search(k) and not kit._PRICE_OUT.search(k)]
            if len(c) == 1:
                rec["price"] = c[0]
        if not any("stock" in k or "avail" in k for k in keys):
            c = [v for k, v in keys.items() if isinstance(v, bool) and kit._STOCK_IN.search(k)]
            if len(c) == 1:
                rec["in_stock"] = c[0]
        if not any("title" in k or "heading" in k for k in keys):
            c = [v for k, v in keys.items()
                 if isinstance(v, str) and kit._TITLE_IN.search(k) and not kit._TITLE_OUT.search(k)]
            if len(c) == 1:
                rec["title"] = c[0]

    # ------------------------------------------------------------------ finish words, ids, labels, cart enums (01a 5)
    @staticmethod
    def is_decline_word(text):
        return bool(kit._DECLINE.search(kit.low(text)))

    @staticmethod
    def is_order_word(text):
        return bool(kit._ORDER.search(kit.low(text)))

    @staticmethod
    def looks_like_id(text):
        return isinstance(text, str) and bool(kit._ID.fullmatch(text))

    @staticmethod
    def is_code_token(text):
        return kit.is_plain(text) and bool(kit._CODE.fullmatch(str(text)))

    @staticmethod
    def split_joint(label):
        parts = [p.strip() for p in re.split(r"\s+and\s+|\s*[&/]\s*|\s+\+\s+", str(label or ""), flags=re.I)]
        return (parts[0], parts[1]) if len(parts) == 2 and all(parts) else None

    @staticmethod
    def is_fault_key(name):
        return bool(kit._FAULT.search(kit.low(name)))

    @staticmethod
    def cart_choices(options, learned=None):
        opts = list(options or [])
        if not opts:
            return None, None
        wl = [re.findall(r"[a-z]+", kit.uncamel(str(o)).lower()) for o in opts]

        def has(i, rx):
            return any(rx.match(w) for w in wl[i])
        if learned is not None and learned in opts:
            put = learned
        else:
            put = next((opts[i] for i in range(len(opts)) if has(i, kit._PUT) and not has(i, kit._DROP)), opts[0])
        drop = next((opts[i] for i in range(len(opts)) if opts[i] != put and has(i, kit._DROP)), None)
        if drop is None and len(opts) == 2:
            drop = opts[1] if opts[0] == put else opts[0]
        return put, drop

    # ------------------------------------------------------------------ record readers (07 3.12)
    @staticmethod
    def shelf_field(rec):
        for k, v in (rec.items() if isinstance(rec, dict) else ()):
            n = str(k).lower()
            if kit._SHELF.search(n) and "id" not in n and isinstance(v, str) and 1 <= len(v.strip()) <= 60:
                return v.strip().lower()
        return ""

    @staticmethod
    def field_stock(name, value):
        n = str(name).lower()
        if value is None:
            return None
        stocky, onhand = ("stock" in n or "avail" in n), bool(kit._ONHAND.search(n))
        if not stocky and not onhand:
            return None
        inverted = stocky and ("out" in n or "sold" in n)
        nil = value is False or (kit.is_num(value) and value <= 0)
        vt = str(value).lower()
        if inverted and value is True:
            return False
        if stocky and not inverted and (nil or kit._GONE.search(vt)):
            return False
        if onhand and nil:
            return False
        if kit._LATE.search(vt):
            return False
        return True

    @staticmethod
    def stock_of(rec):
        says = [kit.field_stock(k, v) for k, v in (rec.items() if isinstance(rec, dict) else ())]
        says = [x for x in says if x is not None]
        return (False not in says) if says else None

    @staticmethod
    def out_of_stock(rec):
        return isinstance(rec, dict) and any(kit.field_stock(k, v) is False for k, v in rec.items())

    @staticmethod
    def identity(rec):
        if isinstance(rec, dict):
            ok = [(str(k).lower(), v) for k, v in rec.items() if isinstance(v, (str, int)) and not isinstance(v, bool)]
            for k, v in ok:
                if k == "id":
                    return str(v)
            for k, v in ok:
                if k.endswith("id"):  # "paid", "valid" also qualify, as built
                    return str(v)
        try:
            return json.dumps(rec, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return str(rec)

    @staticmethod
    def row_id(rec):
        for k, v in (rec.items() if isinstance(rec, dict) else ()):
            if isinstance(v, str) and "id" in str(k).lower() and kit.looks_like_id(v):
                return v
        return ""

    @staticmethod
    def row_title(rec):
        if not isinstance(rec, dict):
            return ""
        v = rec.get("title")
        if isinstance(v, str) and 1 <= len(v.strip()) <= 300:
            return v.strip()
        for k, v in rec.items():
            n = str(k).lower()
            if (isinstance(v, str) and 1 <= len(v.strip()) <= 300 and re.search(r"title|heading|headline|name", n)
                    and "id" not in n and not kit._SHELF.search(n) and not kit.is_junk_key(str(k))):
                return v.strip()
        return ""

    # ------------------------------------------------------------------ evidence tests (06 6.5)
    @staticmethod
    def seen(value, text):
        t = kit.low(text)
        if not t.strip():
            return False
        v = kit._LINKS.sub(" ", kit.low_plain(value))
        for alt in re.split(r"\bor\b|[/,;]", v):
            ws = [w for w in re.findall(r"[a-z]{2,}", alt) if not kit.is_link(w)]
            fs = kit._FIG.findall(alt)
            if not ws and not fs:
                continue
            if (all(re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), t) for w in ws)
                    and all(kit._lone(f, t) for f in fs)):
                return True
        return False

    @staticmethod
    def _same_fields(row, prop):
        return [str(v) for k, v in ((row or {}).get("fields") or {}).items() if kit.property_match(k, prop)]

    @staticmethod
    def shows(row, prop, value, hits=None):
        same = kit._same_fields(row, prop)
        if same:
            return any(kit.seen(value, v) for v in same)
        if kit.seen(value, (row or {}).get("title") or ""):
            return True
        if (row or {}).get("fields") or not hits:  # the query-word test is for unread rows only
            return False
        kw = kit.kind_terms(value)
        return bool(kw) and kw <= kit.kind_terms(" ".join(str(h) for h in hits))

    @staticmethod
    def clash(row, prop, value):
        same = kit._same_fields(row, prop)
        return bool(same) and not any(kit.seen_in(value, v) for v in same)

    @staticmethod
    def value_rank(row, prop, value):
        alts = {" ".join(a) for a in kit.alternatives(value)}
        best, pk = None, kit.property_slug(prop)
        for k, v in (((row or {}).get("fields")) or {}).items():
            v = str(v)
            if not (kit.property_match(k, prop) and kit.seen(value, v)):
                continue
            a = 0 if kit.property_slug(k) == pk else 1
            if " ".join(kit.value_words(v)) in alts:
                b = 0
            elif any(" ".join(kit.value_words(p)) in alts for p in re.split(r"[,;|]", v)):
                b = 1
            else:
                b = 3
            best = (a, b) if best is None else min(best, (a, b))
        if best is not None:
            return best
        return (2, 2) if kit.seen(value, (row or {}).get("title") or "") else (2, 3)

    @staticmethod
    def apart(row, rules):
        fields = (row or {}).get("fields") or {}
        for prop, vals in rules or ():
            fs = [str(v) for k, v in fields.items() if kit.property_match(k, prop) and str(v).strip()]
            vals = [x for x in (vals or ()) if x]
            if not any(all(not kit.writes(f, x) for x in vals) for f in fs):
                return False
        return True



# ------------------------------------------------------------ p03_link.py
# p03_link: the wire to the ORO proxy (02 S1, S10.1, S10.13, S13) and the schema checker (01b 3.18, 02 S2).


class _p03_Out:
    """A proxy result made by our own code (a send that raised)."""
    def __init__(self, data, error): self.data, self.error = data, error


class Pace:
    """Send pacing shared by model and store calls (02 S1.5)."""
    def __init__(self):
        self.last = 0.0
        self.calm_until = 0.0
        self.hold_until = 0.0

    def delay(self, model=False):
        now = monotonic()
        gap = Tune.Clock.SLOW_GAP if now < self.calm_until else Tune.Clock.GAP
        d = self.last + gap - now
        if model: d = max(d, self.hold_until - now)
        return d

    def mark(self): self.last = monotonic()

    def see(self, res):
        if getattr(res, "data", None) is None and Link.err_of(res)[0] in (429, 503):
            self.calm_until = monotonic() + Tune.Clock.SLOW_SPAN


class ModelDesk:
    """One model query over a fallback chain (02 S1.8-S1.11)."""
    def __init__(self, link): self.link = link

    @staticmethod
    def shape(messages, model, flat):
        if flat:
            out = []
            for m in messages:
                m = dict(m)
                c = m.get("content")
                if isinstance(c, list):
                    m["content"] = "\n\n".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
                out.append(m)
            return out
        if isinstance(model, str) and model.startswith("anthropic/") and len(messages) > 2:
            out = [dict(m) for m in messages]
            for j in (0, len(out) - 1):
                c = out[j].get("content")
                if isinstance(c, str) and c:
                    out[j]["content"] = [{"type": "text", "text": c, "cache_control": {"type": "ephemeral"}}]
            return out
        return messages

    def _single(self, chain): return len({m for m in chain if m not in self.link.dropped}) <= 1

    def _dear(self, chain, i, messages, cap):
        L = self.link
        sc = L.scale or 1.0
        if L.spend >= Tune.Spend.CAP * sc: return True
        # the chain (lineup plus spares) is the list walked; "next entry" is read on it
        if i + 1 >= len(chain) or chain[i + 1] in Tune.Models.PREMIUM: return False
        est = (len(json.dumps(messages, default=str)) / Tune.Spend.CHARS_PER_TOKEN * Tune.Spend.IN_RATE
               + min(cap, Tune.Spend.OUT_EST_MAX) * Tune.Spend.OUT_RATE)
        return L.spend >= Tune.Spend.SOFT * sc or L.spend + est > Tune.Spend.CAP * sc

    @staticmethod
    def allowed(body):
        m = re.search(r"allowed_models\"?\s*:\s*\[([^\]]*)\]", body, re.I)
        return re.findall(r"\"([^\s\"]{3,100})\"", m.group(1)) if m else []

    def _quit(self, empty, notes):
        self.link.fail_kind = "empty" if empty else ("refused" if notes and all(n == "hard" for n in notes) else "")
        return None

    def query(self, messages, models, cap, tools=None):
        L = self.link
        L.fail_kind = ""
        L.tried = set()
        if L.mute or L.off: return None
        lineup = [m for m in (models or []) if isinstance(m, str)]
        chain = lineup + [m for m in L.spares if m not in lineup]
        single = self._single(chain)
        skip, notes, empty, seed = set(), [], False, None
        i = -1
        while True:
            i += 1
            if i >= len(chain): return self._quit(empty, notes)
            if not L.off and L.spend >= Tune.Spend.HARD * (L.scale or 1.0):
                L.off = True
                L.note("spend_stop", spend=round(L.spend, 5))
            if L.off: return self._quit(empty, notes)
            m = chain[i]
            if (m in L.dropped or m in skip or (tools and m in L.no_tools)
                    or (m in Tune.Models.PREMIUM and self._dear(chain, i, messages, cap))):
                continue
            tries = 1 if L.service_down() else Tune.Tries.PER_MODEL
            flat = False
            while tries > 0:
                tries -= 1
                left = L.time_left()
                if left < Tune.Clock.MODEL_FLOOR: return self._quit(empty, notes)
                top = Tune.Clock.TIMEOUT_DOWN if L.service_down() else Tune.Clock.TIMEOUT_MAX
                body = {"model": m, "messages": self.shape(messages, m, flat), "temperature": 0, "max_tokens": cap}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                L.tried.add(m)
                cl = L.model_client
                try:
                    cl.timeout = int(min(max(left - Tune.Clock.TIMEOUT_PAD, Tune.Clock.TIMEOUT_MIN), top))
                except Exception:
                    pass
                t0 = monotonic()
                res = L.post(cl, "/inference/chat/completions", body, True)
                took = monotonic() - t0
                data = getattr(res, "data", None)
                ch = data.get("choices") if isinstance(data, dict) else None
                if isinstance(ch, list) and ch:
                    L.last_good = monotonic()
                    L.misses = 0
                    use = data.get("usage")
                    cost = use.get("cost") if isinstance(use, dict) else None
                    if kit.is_num(cost): L.spend += cost
                    msg = ch[0].get("message") if isinstance(ch[0], dict) else None
                    if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")): return msg
                    empty = True
                    L.note("model_fail", model=m, error="empty")
                    break
                L.misses += 1
                status, text = Link.err_of(res)
                L.note("model_fail", model=m, error=(("%s: %s" % (status, text)) if (status or text) else "")[:300])
                low = text.lower()
                if status == 402 or (status == 403 and any(w in low for w in ("key limit", "insufficient", "credit"))):
                    L.off = True
                    return self._quit(empty, notes)
                if status in (401, 403, 404):
                    notes.append("hard")
                    skip.add(m)
                    if status == 404 and tools and "tool" in low: L.no_tools.add(m)
                    elif not (status == 403 and ("moderat" in low or "flagged" in low)): L.dropped.add(m)
                    if status == 403 and not L.spares:
                        got = [x for x in self.allowed(text) if x not in L.dropped and x not in lineup][:2]
                        if got:
                            L.spares = got
                            chain += [x for x in got if x not in chain]
                            single = self._single(chain)
                    break
                if status in (400, 413, 422):
                    notes.append("hard")
                    if flat:
                        skip.add(m)
                        break
                    flat = True
                    continue
                if status is not None and 400 <= status < 500 and status not in (408, 425, 429):
                    notes.append("hard")
                    skip.add(m)
                    break
                notes.append("soft")
                ends = single or len(L.tried) >= 2
                if took > Tune.Clock.SLOW_SEND:
                    if L.service_down() and ends: return self._quit(empty, notes)
                    break
                if seed is None: seed = json.dumps(messages[-1] if messages else "", default=str)[-600:]
                wait = L.pause(seed + m, min(L.misses - 1, 6), status)
                if L.time_left() - wait < Tune.Clock.MODEL_FLOOR: return self._quit(empty, notes)
                L.pace.hold_until = monotonic() + wait
                if L.service_down(monotonic() + wait) and ends: return self._quit(empty, notes)


class StoreDesk:
    """One store action with retries and step conflicts (02 S1.12)."""
    def __init__(self, link): self.link = link

    def call(self, name, args):
        L = self.link
        action = {"name": name, "args": args if isinstance(args, dict) else {}}
        step = L.steps + 1
        sends = Tune.Tries.STORE_BASE + Tune.Tries.STORE_EXTRA
        pauses = bumps = 0
        while sends > 0:
            sends -= 1
            left = L.time_left()
            if left < 2: return None
            body = dict(L.binding)
            body["step"] = step
            body["action"] = action
            cl = L.store_client
            try:
                cl.timeout = int(min(max(left - 1, Tune.Clock.STORE_TIMEOUT_MIN), Tune.Clock.STORE_TIMEOUT_MAX))
            except Exception:
                pass
            res = L.post(cl, "/environment/call", body)
            data = getattr(res, "data", None)
            if isinstance(data, dict):
                st = data.get("step")
                L.steps = st if isinstance(st, int) and not isinstance(st, bool) else L.steps + 1
                if data.get("finished"): L.done = True
                return data
            status, text = Link.err_of(res)
            L.note("store_fail", status=status, body=text[:300])
            low = text.lower()
            if status == 402:
                L.broke = True
                return None
            if status == 409:
                if "finished" in low:
                    L.done = True
                    return None
                hit = re.search(r"expected\s*:?\s*(\d+)", low)
                if hit:
                    step = int(hit.group(1))
                    L.steps = step - 1
                    continue
                if "reused" in low and bumps < Tune.Tries.STEP_BUMPS:
                    bumps += 1
                    step += 1
                    continue
                return None
            if status in (403, 422) or pauses >= Tune.Tries.BACKOFF_WAITS or (status in (400, 401, 404) and pauses >= 1):
                return None
            wait = L.pause(json.dumps(body, default=str)[-600:], pauses, status)
            pauses += 1
            if L.time_left() - wait < 2: return None
            sleep(wait)
        return None


class Link:
    """Everything that talks to the outside: models, store, clock, spend, run log (02 S1)."""
    def __init__(self, s):
        self.s = s
        self.binding = dict(s.binding) if isinstance(getattr(s, "binding", None), dict) else {}
        self.start = self.last_good = monotonic()
        self.spend = 0.0
        self.steps = 0
        self.misses = 0
        self.done = self.broke = self.mute = self.off = False
        self.log = []
        self.dropped, self.no_tools, self.tried = set(), set(), set()
        self.spares = []
        self.fail_kind = ""
        self.scale = 1.0
        self.pace = Pace()
        self.models = ModelDesk(self)
        self.store = StoreDesk(self)
        try:
            self.model_client = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
            self.store_client = ProxyClient(timeout=Tune.Clock.STORE_TIMEOUT_MAX, max_retries=1, retry_delay=1)
        except Exception as e:
            self.model_client = self.store_client = None
            self.fault("client_error", e)

    # -- clock, spend, availability (S1.4, S1.13)
    def time_left(self): return Tune.Clock.DEADLINE - (monotonic() - self.start)

    def budget_gone(self): return self.spend >= Tune.Spend.CAP * (self.scale or 1.0)

    def service_down(self, at=None):
        if self.off: return True
        if not self.misses: return False
        return (monotonic() if at is None else at) - self.last_good > Tune.Clock.DOWN_AFTER

    def usable(self, models, tools=False):
        if self.off or self.mute: return False
        for m in list(models or []) + list(self.spares):
            if m not in self.dropped and not (tools and m in self.no_tools): return True
        return False

    # -- the send wrapper (S1.6) and backoff (S1.7)
    @staticmethod
    def err_of(res):
        e = getattr(res, "error", None)
        if not isinstance(e, dict): return None, ""
        st = e.get("status")
        st = st if isinstance(st, int) and not isinstance(st, bool) else None
        b = e.get("body")
        b = b if isinstance(b, str) else ("" if b is None else json.dumps(b, default=str))
        return st, b

    def post(self, client, path, body, model=False):
        wait = min(self.pace.delay(model), self.time_left() - 1)
        if wait > 0: sleep(wait)
        self.pace.mark()
        try:
            res = client.post_verbose(path, body)
        except Exception as e:
            res = _p03_Out(None, {"kind": "network", "status": None, "body": ("%s: %s" % (type(e).__name__, e))[:300]})
        self.pace.see(res)
        return res

    def pause(self, seed, k, status):
        seed = str(seed)
        w = 7 * len(seed) + sum(ord(c) for c in seed[-64:]) + 31 * k + 13 * self.steps
        base = Tune.Tries.BACKOFF_BASE_429 if status == 429 else Tune.Tries.BACKOFF_BASE
        return min(Tune.Tries.BACKOFF_CAP, base * 2 ** k) * (0.75 + (w % 101) / 100 / 2)

    # -- public calls
    def chat(self, messages, models, cap, tools=None):
        try:
            return self.models.query(messages, models, cap, tools)
        except Exception as e:
            self.fault("model_error", e)
            return None

    def consult(self, tag, text, lineup):
        try:
            models, cap = LINEUPS[lineup]
            msg = self.chat([{"role": "user", "content": "ROLE: " + tag + "\n" + text}], models, cap)
            if not isinstance(msg, dict): return None
            c = msg.get("content")
            if isinstance(c, list): c = "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
            return kit.json_from(c if isinstance(c, str) else "")
        except Exception as e:
            self.fault("consult_error", e)
            return None

    def store_call(self, name, args):
        try:
            return self.store.call(name, args)
        except Exception as e:
            self.fault("store_error", e)
            return None

    def note(self, kind, **fields): self.log.append(dict({"kind": kind}, **fields))

    def fault(self, kind, err): self.log.append({"kind": kind, "error": ("%s: %s" % (type(err).__name__, err))[:300]})


class ArgFixer:
    """Check and lightly repair tool arguments against a JSON schema (01b 3.18, 02 S2)."""
    _BOUNDS = (("minimum", "below minimum", lambda v, n: v < n), ("maximum", "above maximum", lambda v, n: v > n),
               ("exclusiveMinimum", "must be above", lambda v, n: v <= n),
               ("exclusiveMaximum", "must be below", lambda v, n: v >= n))

    @staticmethod
    def _fits(v, k):
        if k == "string": return isinstance(v, str)
        if k == "integer": return isinstance(v, int) and not isinstance(v, bool)
        if k == "number": return kit.is_num(v)
        if k == "boolean": return isinstance(v, bool)
        if k == "object": return isinstance(v, dict)
        if k == "array": return isinstance(v, list)
        if k == "null": return v is None
        return True

    @staticmethod
    def _convert(v, k):
        if k == "integer":
            if isinstance(v, float) and v.is_integer(): return True, int(v)
            if isinstance(v, str):
                m = re.fullmatch(r"\s*(-?\d+)(?:\.0*)?\s*", v)
                if m: return True, int(m.group(1))
        elif k == "number":
            if isinstance(v, str) and re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", v):
                t = v.strip()
                if "." not in t: return True, int(t)
                f = float(t)
                return True, (int(f) if f.is_integer() else f)
        elif k == "string":
            if kit.is_num(v): return True, str(v)
        elif k == "boolean":
            if isinstance(v, str) and v.lower() in ("true", "false"): return True, v.lower() == "true"
        return False, None

    @staticmethod
    def _whole(n): return isinstance(n, int) and not isinstance(n, bool)

    @staticmethod
    def fix(value, schema, where="args"):
        if not isinstance(schema, dict): return value, []
        t = schema.get("type")
        kinds = [t] if isinstance(t, str) else ([str(x) for x in t] if isinstance(t, list) else [])
        if kinds:
            if any(ArgFixer._fits(value, k) for k in kinds):
                if isinstance(value, float) and "integer" in kinds and value.is_integer(): value = int(value)
            else:
                for k in kinds:
                    ok, nv = ArgFixer._convert(value, k)
                    if ok:
                        value = nv
                        break
            if not any(ArgFixer._fits(value, k) for k in kinds):
                return value, ["%s: expected %s" % (where, "/".join(kinds))]
        faults = []
        opts = schema.get("enum")
        if isinstance(opts, list) and value not in opts:
            hit = None
            if isinstance(value, str):
                lv = value.strip().lower()
                hit = next((o for o in opts if isinstance(o, str) and o.strip().lower() == lv), None)
            if hit is not None: value = hit
            else: faults.append("%s: must be one of %s" % (where, opts))
        if "const" in schema and value != schema["const"]:
            faults.append("%s: must equal %r" % (where, schema["const"]))
        n = schema.get("minLength")
        if isinstance(value, str) and ArgFixer._whole(n) and len(value) < n:
            faults.append("%s: shorter than %s characters" % (where, n))
        n = schema.get("maxLength")
        if isinstance(value, str) and ArgFixer._whole(n) and len(value) > n: value = value[:n]
        for key, word, test in ArgFixer._BOUNDS:
            n = schema.get(key)
            if kit.is_num(value) and kit.is_num(n) and test(value, n): faults.append("%s: %s %s" % (where, word, n))
        if isinstance(value, list):
            n = schema.get("minItems")
            if ArgFixer._whole(n) and len(value) < n: faults.append("%s: needs at least %s items" % (where, n))
            n = schema.get("maxItems")
            if ArgFixer._whole(n) and len(value) > n: faults.append("%s: allows at most %s items" % (where, n))
            item = schema.get("items")
            if isinstance(item, dict):
                fixed = []
                for i, x in enumerate(value):
                    x, f = ArgFixer.fix(x, item, "%s[%d]" % (where, i))
                    fixed.append(x)
                    faults += f
                value = fixed
        if "type" in schema and isinstance(value, dict):
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            need = schema.get("required") if isinstance(schema.get("required"), list) else []
            for name in need:
                if name not in value: faults.append("%s.%s: required" % (where, name))
            if schema.get("additionalProperties") is False:
                for name in value:
                    if name not in props: faults.append("%s.%s: not an allowed argument" % (where, name))
            out = {}
            for name, x in value.items():
                if name in props:
                    x, f = ArgFixer.fix(x, props[name], "%s.%s" % (where, name))
                    faults += f
                out[name] = x
            value = out
        return value, faults



# ------------------------------------------------------------ p04_memory.py
# p04_memory: what the store returned (02 S3-S4, 06 6.4), what the shopper wants (02 S5, 06 6.1), the cart (02 S6).

_p04_SHELF = re.compile(r"categor|shelf|department|aisle", re.I)
_p04_PRICE = re.compile(r"price|cost", re.I)
_p04_DIG_PRICE = re.compile(r"price|cost|amount")
_p04_DIG_NOT = re.compile(r"count|qty|quant|total|units|line|due|payable|sum")
_p04_TITLE = re.compile(r"title|name")
_p04_MONEY = re.compile(r"\b(?:price|stock)\b|budget|quantity|\bcents?\b")
_p04_PRICE_TALK = re.compile(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b")
_p04_COUNTISH = re.compile(r"qty|quant|count|units|number|total|held|ordered")
_p04_MONEYISH = re.compile(r"price|cost|due|sum|fee|charge")


def _p04_num(x):
    if isinstance(x, float) and x.is_integer(): return str(int(x))
    return str(x)


def _p04_texts(v): return [x.strip() for x in v if isinstance(x, str) and x.strip()] if isinstance(v, list) else []


def _p04_yes(v): return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _p04_pos_int(v): return isinstance(v, int) and not isinstance(v, bool) and v > 0


class _p04_Tally:
    """Property tally of one menu (02 S4)."""
    def __init__(self, scoped):
        self.scoped = scoped
        self.records = self.full = 0
        self.hits, self.rows, self.vals, self.samples = {}, {}, {}, {}

    def feed(self, rec):
        short = [(str(k), v.strip()) for k, v in rec.items()
                 if isinstance(v, str) and 0 < len(v.strip()) <= 70 and not kit.is_junk_key(str(k))]
        few = len(short) < 5
        if few and not (self.scoped and len(short) >= 2 and not any(isinstance(v, dict) for v in rec.values())):
            return
        self.records += 1
        row = few and any(isinstance(v, str) and "id" in str(k).lower() for k, v in rec.items())
        if not row: self.full += 1
        for k, v in short:
            if ("-" in v or "_" in v) and re.search(r"\d", v) and kit.is_code_token(v): continue
            name = k.replace("_", " ").replace(".", " ").strip().lower()
            val = v.lower()
            self.hits[name] = self.hits.get(name, 0) + 1
            if row: self.rows[name] = self.rows.get(name, 0) + 1
            c = self.vals.setdefault(name, {})
            c[val] = c.get(val, 0) + 1
            sm = self.samples.setdefault(name, [])
            if val not in sm and len(sm) < 5: sm.append(val)

    def text(self):
        names = list(self.hits)
        if self.full: names = [n for n in names if self.rows.get(n, 0) < self.hits[n]]
        def split(n):
            h = self.hits[n]
            return min(1, h / self.records) * (1 - max(self.vals[n].values()) / h) if self.records and h else 0
        names.sort(key=lambda n: (-self.hits[n], -split(n), n))
        out = []
        for n in names[:28]:
            sm, h = self.samples[n], self.hits[n]
            line = "- %s (in %d listings)" % (n, h)
            if len(sm) >= 2 and h >= 2: line += " [varies: %d distinct values]" % len(sm)
            if sm and all(re.search(r"\d", x) for x in sm) and any(re.search(r"\d\s*[a-z'\"]", x) for x in sm):
                line += " [figures with a unit]"
            out.append(line + ": " + " | ".join(sm))
        return "\n".join(out)


class Archive:
    """Every store observation (lens copies) and the views built from them (02 S3-S4, 06 6.4)."""
    def __init__(self, s):
        self.s = s
        self.obs = []
        self.known = set()
        self.read = set()
        self.shelf_by_id = {}
        self._memo, self._memo_n = {}, None
        self._prices, self._priced = {}, 0
        self._ci = {"n": 0, "cards": {}}
        self._dig = (-1, {})

    @staticmethod
    def _idish(k): return "id" in str(k).lower()

    @staticmethod
    def _rid(d):
        for k, v in d.items():
            if not isinstance(v, str): continue
            kl = str(k).lower()
            if kl in ("id", "_id") and v.strip(): return v
            if kl.endswith("id") and kit.looks_like_id(v): return v
        return ""

    @staticmethod
    def _shelf_text(d):
        for k, v in d.items():
            kl = str(k).lower()
            if isinstance(v, str) and _p04_SHELF.search(kl) and "id" not in kl and 0 < len(v.strip()) <= 60:
                return v.strip()
        return ""

    def _read_id(self, d):
        return next((v for k, v in d.items() if isinstance(v, str) and self._idish(k) and v in self.read), None)

    def _cached(self, key, make):
        n = (len(self.obs), len(self.read), len(self.shelf_by_id))
        if self._memo_n != n: self._memo, self._memo_n = {}, n
        if key not in self._memo: self._memo[key] = make()
        return self._memo[key]

    def add(self, obs):
        self.obs.append(obs)
        texts = []
        def take(d):
            for k, v in d.items():
                if isinstance(v, str):
                    if self._idish(k) and kit.looks_like_id(v): self.known.add(v)
                    t = v.strip()
                    if 0 < len(t) <= 80: texts.append(t)
            rid = self._rid(d)
            if not rid: return
            if any(isinstance(v, dict) and sum(1 for x in v.values() if isinstance(x, str) or kit.is_num(x)) >= 2
                   for v in d.values()):
                self.read.add(rid)
            if rid not in self.shelf_by_id:
                lab = self._shelf_text(d)
                if lab: self.shelf_by_id[rid] = lab
        kit.visit(obs, take)
        if texts: kit.feed_words(" ".join(texts))

    def price(self, lid):
        if self._priced < len(self.obs):
            for o in self.obs[self._priced:]: kit.visit(o, self._price_scan)
            self._priced = len(self.obs)
        return self._prices.get(lid)

    def _price_scan(self, d):
        ids = [v for k, v in d.items() if isinstance(v, str) and self._idish(k)]
        if not ids: return
        p = next((v for k, v in d.items() if kit.is_num(v) and _p04_PRICE.search(str(k))), None)
        if p is not None:
            for i in ids: self._prices[i] = p

    def about(self, lid, hold=4):
        if not lid: return []
        got = []
        def walk(v, depth):
            if len(got) >= 8 or depth > 6: return
            if isinstance(v, dict):
                if any(isinstance(x, str) and x == lid for x in v.values()):
                    t = kit.render(kit.clip_deep(v, 400), 2500)
                    if t not in got: got.append(t)
                    return
                for x in v.values(): walk(x, depth + 1)
            elif isinstance(v, list):
                for x in v: walk(x, depth + 1)
        for o in self.obs:
            if len(got) >= 8: break
            walk(o, 0)
        return got[-hold:] if hold > 0 else []

    def _cards(self):
        ci = self._ci
        if ci["n"] < len(self.obs):
            for o in self.obs[ci["n"]:]: kit.visit(o, self._card_take)
            ci["n"] = len(self.obs)
        return ci["cards"]

    def _card_take(self, d):
        owner = next((v for k, v in d.items() if isinstance(v, str) and self._idish(k) and v in self.known
                      and kit.looks_like_id(v)), None)
        if owner is None: return
        sheet = any(isinstance(v, dict) and len(v) >= 2 for v in d.values())
        if sheet or len(d) >= 4: self._ci["cards"].setdefault(owner, {}).update(d)
        if sheet: self.read.add(owner)

    @staticmethod
    def _card_price(c): return next((v for k, v in c.items() if kit.is_num(v) and _p04_PRICE.search(str(k))), None)

    def cards(self, budget=30000, anchor="", skip=(), lead=()):
        cards = self._cards()
        ap = None
        if anchor:
            ap = self._card_price(cards[anchor]) if anchor in cards else None
            if ap is None: ap = self.price(anchor)
        skip, lead = set(skip or ()), set(lead or ())
        rows = []
        for pos, (own, c) in enumerate(cards.items()):
            if own != anchor:
                if own in skip: continue
                p = self._card_price(c)
                if ap is not None and p is not None and not p < ap: continue
            line = ("DETAILS " if own in self.read else "ROW ") + kit.render(kit.clip_deep(c, 160))
            gone = any(kit.field_stock(str(k), v) is False for k, v in c.items() if not isinstance(v, (dict, list)))
            rows.append((own, line, gone, pos))
        over = sum(len(r[1]) for r in rows) > budget
        rows.sort(key=lambda r: (r[0] not in lead, r[0] not in self.read, bool(over and r[2]), r[3]))
        left, out = budget, []
        for r in rows:
            if len(r[1]) <= left:
                out.append(r[1])
                left -= len(r[1])
        return out

    def record_shelf(self, rec):
        if not isinstance(rec, dict): return ""
        lab = self._shelf_text(rec)
        if lab: return lab
        for k, v in rec.items():
            if isinstance(v, str) and self._idish(k) and v in self.shelf_by_id: return self.shelf_by_id[v]
        return ""

    def rich_count(self, shelf=None):
        want = shelf.strip().lower() if isinstance(shelf, str) and shelf.strip() else None
        return self._cached(("rich", want), lambda: self._rich(want))

    def _rich(self, want):
        found = set()
        def take(d):
            rid = self._read_id(d)
            if rid is None or rid in found: return
            if want is not None and self.record_shelf(d).lower() != want: return
            subs = [v for v in d.values() if isinstance(v, dict)]
            if subs: rich = max(len(v) for v in subs) >= 3
            else: rich = sum(1 for v in d.values() if isinstance(v, str) and 0 < len(v.strip()) <= 70) >= 5
            if rich: found.add(rid)
        for o in self.obs: kit.visit(o, take)
        return len(found)

    def shelves_opened(self): return list(self._cached(("shelves",), self._shelves))

    def _shelves(self):
        cnt = {}
        def take(d):
            if self._read_id(d) is None or not any(isinstance(v, dict) for v in d.values()): return
            lab = self.record_shelf(d)
            if lab: cnt[lab] = cnt.get(lab, 0) + 1
        for o in self.obs: kit.visit(o, take)
        return sorted(cnt, key=lambda x: -cnt[x])

    def menu(self, shelf=None):
        want = shelf.strip().lower() if isinstance(shelf, str) and shelf.strip() else None
        return self._cached(("menu", want), lambda: self._menu(want))

    def _menu(self, want):
        tally = _p04_Tally(want is not None)
        def walk(v, depth, inside):
            if depth > 6: return
            if isinstance(v, dict):
                if not inside:
                    inside = want is None or self.record_shelf(v).lower() == want or any(
                        isinstance(x, str) and _p04_SHELF.search(str(k)) and x.strip().lower() == want
                        for k, x in v.items())
                if inside: tally.feed(v)
                for x in v.values(): walk(x, depth + 1, inside)
            elif isinstance(v, list):
                for x in v: walk(x, depth + 1, inside)
        for o in self.obs: walk(o, 0, False)
        return tally.text(), tally.records

    def labels_listing(self): return self._cached(("labels",), self._labels)

    def _labels(self):
        per = {}
        def take(d):
            if len(d) < 4: return
            for k, v in d.items():
                if not isinstance(v, str): continue
                t = v.strip()
                if not 0 < len(t) <= 40 or (kit.is_code_token(t) and re.search(r"\d", t)): continue
                c = per.setdefault(str(k), {})
                c[t] = c.get(t, 0) + 1
        for o in self.obs: kit.visit(o, take)
        out = []
        for k, c in per.items():
            if sum(c.values()) >= 4 and 2 <= len(c) <= 8 and max(c.values()) >= 2:
                vals = sorted(c, key=lambda x: -c[x])
                out.append("- %s: " % k + " | ".join("%s (x%d)" % (x, c[x]) for x in vals))
                if len(out) >= 12: break
        return "\n".join(out)

    def digest(self):
        if self._dig[0] != len(self.obs):
            rows = {}
            for o in self.obs: kit.visit(o, lambda d: self._dig_take(d, rows))
            self._dig = (len(self.obs), rows)
        return self._dig[1]

    def _dig_take(self, rec, rows):
        mark = None
        for k, v in rec.items():
            if str(k).lower() == "id":
                mark = v
                break
        else:
            for k, v in rec.items():
                if str(k).lower().endswith("id") and (isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool))):
                    mark = v
                    break
        if isinstance(mark, bool) or not isinstance(mark, (str, int)) or str(mark) not in self.known: return
        lid = str(mark)
        row = rows.get(lid)
        if row is None:
            row = rows[lid] = {"id": lid, "shelf": "", "title": "", "price": None, "stock": None, "words": set(),
                               "fields": {}}
        sh = kit.shelf_field(rec)
        if sh: row["shelf"] = sh
        t = next((v for k, v in rec.items() if isinstance(v, str) and _p04_TITLE.search(str(k).lower())
                  and "id" not in str(k).lower()), None)
        if t and t.strip(): row["title"] = t.strip()
        p = next((v for k, v in rec.items() if kit.is_num(v) and _p04_DIG_PRICE.search(str(k).lower())
                  and not _p04_DIG_NOT.search(str(k).lower())), None)
        if p is not None: row["price"] = p
        st = kit.stock_of(rec)
        if st is not None: row["stock"] = st
        for v in rec.values():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (str, int, float)) and not isinstance(v2, bool):
                        row["fields"][str(k2)] = str(v2)
                        row["words"] |= kit.kind_terms(str(v2))
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool): row["words"] |= kit.kind_terms(str(v))


class Ledger:
    """What the shopper wants and how it changes as the shopper talks (02 S5, 06 6.1)."""
    def __init__(self, s):
        self.s = s
        self.request = s.request
        self.reading = {}
        self.written, self.revealed, self.freed, self.dropped = [], [], [], []
        self.rebuilt = None
        self.budget = None
        self.version = 0
        self.heard = set()
        self.exchanges = []
        self.reply_due = ""
        self.widened_at, self.widened_step = -1, -1000000
        self.sweepable = False
        self.resolved = []
        self.pointer_news = []
        self.duties = []
        self.duties_version = 0
        self.limits_news = None
        self.newest = []
        self.parked = []
        self.meta_question = None
        self.pinned = 0
        self.cart_told = False

    # -- small text helpers
    @staticmethod
    def _head(n): return str(n).split(":", 1)[0].strip()

    @staticmethod
    def _val(n):
        n = str(n)
        return n.split(":", 1)[1].strip() if ":" in n else ""

    @staticmethod
    def _norm(t): return " ".join(kit.strip_marks(str(t)).lower().split())

    def _gone(self, need): return self._norm(need) in {self._norm(d) for d in self.dropped}

    def _same_slot(self, a, b):
        ha, hb = self._head(a), self._head(b)
        return kit.property_match(kit.name_head(ha), kit.name_head(hb)) and kit.tags_of(ha) == kit.tags_of(hb)

    def _r(self, key): return self.reading.get(key) if isinstance(self.reading, dict) else None

    # -- reading accessors (S5.3)
    def products(self):
        rows = self._r("parts")
        out = []
        for r in rows if isinstance(rows, list) else []:
            if not isinstance(r, dict) or not isinstance(r.get("kind"), str) or not r["kind"].strip(): continue
            q = r.get("units")
            w = r.get("with")
            out.append({"name": r["kind"].strip(), "qty": q if _p04_pos_int(q) else 1,
                        "musts": [x for x in r.get("musts") if isinstance(x, str) and ":" in x] if isinstance(r.get("musts"), list) else [],
                        "anchor": w - 1 if _p04_pos_int(w) else None})
        return out if len(out) > 1 else []

    def slots(self):
        n = len(self.products())
        return list(range(n)) if n else [None]

    def quantity(self):
        if self.products(): return 1
        q = self._r("units")
        return q if _p04_pos_int(q) else 1

    def counts(self):
        ps = self.products()
        return [p["qty"] for p in ps] if ps else [self.quantity()]

    def priority(self):
        v = str(self._r("pick_rule") or "").strip().lower()
        return v if v in ("cheapest", "best") else "none"

    def _rs(self, key):
        v = self._r(key)
        return v.strip() if isinstance(v, str) else ""

    def product_name(self): return self._rs("item_kind")

    def money_unit(self): return self._rs("money_unit")

    def pick_measure(self): return self._rs("rule_measure")

    def stock_only(self): return self._r("stock_only") is True

    def by_code(self): return bool(self._r("by_code"))

    # -- duties (S5.4)
    def add_duties(self, items, source):
        if not isinstance(items, list): return []
        new = []
        for it in items[:12]:
            if isinstance(it, str): it = {"text": it}
            if not isinstance(it, dict): continue
            text = " ".join(str(it.get("text") or "").split())[:400]
            if not text or text.lower() in {d["text"].lower() for d in self.duties}: continue
            flag = lambda k: it.get(k) if isinstance(it.get(k), bool) else None
            self.duties.append({"text": text, "source": source, "by_finish": flag("by_finish"),
                                "via_message": flag("via_message"), "before_change": flag("before_change")})
            new.append(text[:120])
        if new: self.duties_version += 1
        return new

    def finish_duties(self): return [d for d in self.duties if d.get("by_finish") is not False]

    # -- renderings (S5.5)
    def products_text(self):
        ps = self.products()
        if not ps: return ""
        rows = " | ".join("%d) %s x%d" % (i, p["name"], p["qty"]) for i, p in enumerate(ps, 1))
        return ("ITEMS IN THIS ORDER (%d): %s. Cart rule: one cart line per item at its own count, no extra lines; "
                "a listing is judged only against the needs of the item whose category it sits in; a cheapest or "
                "best wish applies within each item, not across the order" % (len(ps), rows))

    def req_text(self):
        parts = []
        if self.rebuilt is not None: parts += list(self.rebuilt)
        else:
            parts += [w + Tune.Marks.STATED for w in self.written if not self._gone(w)]
            parts += [r + Tune.Marks.REVEALED for r in self.revealed if not self._gone(r)]
        if not parts:
            parts.append("no property requirement is left to match on, so any in-stock listing of the asked category "
                         "is acceptable" if self.dropped else "see the request")
        if self.budget is not None:
            parts.append(("budget, whole order, upper limit: %s %s" % (_p04_num(self.budget), self.money_unit())).rstrip())
        parts.append(self.products_text() if self.products() else "quantity wanted: %d" % self.quantity())
        if self.dropped: parts.append("dropped by the shopper: " + ", ".join(self.dropped))
        if self.freed: parts.append("no preference on: " + ", ".join(self.freed))
        return "; ".join(parts)

    def drops_text(self):
        if not self.dropped: return ""
        return ("LATER DROPS: the shopper withdrew these after writing the request, and this overrides it: "
                + "; ".join(self.dropped) + ". A withdrawn need is void: no listing is rejected, ranked or favoured "
                "for it.")

    def notes_text(self, steps=None, msgs=None):
        r = self.reading
        if not isinstance(r, dict) or not r: return ""
        steps = self.s.step_cap if steps is None else steps
        msgs = self.s.msg_cap if msgs is None else msgs
        b = []
        if self.product_name(): b.append("product: " + self.product_name())
        ps = self.products()
        if ps:
            for i, p in enumerate(ps, 1):
                b.append("item %d/%d: %s | count %d | gets its own cart row" % (i, len(ps), p["name"], p["qty"]))
            b.append("cart: nothing beyond these rows; a cart tool taking a list of lines gets the full set each call")
        else: b.append("quantity: %d" % self.quantity())
        for m in _p04_texts(r.get("musts")): b.append("firm need: " + m)
        for m in _p04_texts(r.get("pointer_needs")):
            b.append("need set by another listing: " + m + " (find it, read the field named, search with its value)")
        cap = r.get("spend_cap")
        if kit.is_num(cap) or (isinstance(cap, str) and cap.strip()):
            b.append(("spend ceiling: %s %s" % (_p04_num(cap), self.money_unit())).rstrip())
        if self.stock_only(): b.append("availability: currently stocked goods")
        pr = self.priority()
        if pr == "none":
            b.append("choice rule: none given; any listing meeting every need within budget is right, and hunting "
                     "for a lower price only burns steps unless the request asks for it")
        else:
            d = self.pick_measure()
            b.append("pick: the %s%s fit" % (pr, " (%s)" % d if d else ""))
        for m in _p04_texts(r.get("wishes")): b.append("soft wish (tie-break only): " + m)
        for m in _p04_texts(r.get("ask_before")): b.append("ask first: " + m)
        for m in _p04_texts(r.get("procedures")): b.append("rule: " + m)
        if msgs is not None: b.append("message allowance: %s" % msgs)
        b.append("step allowance: %s" % steps)
        return ("[request notes] Draft parse of the request; where it disagrees with the request text, the text "
                "wins.\n" + "\n".join("- " + x for x in b))

    def duties_text(self):
        if not self.duties: return ""
        rows = []
        for i, d in enumerate(self.duties[-10:], 1):
            tags = [t for t, k in (("before the finish", "by_finish"), ("by a message", "via_message")) if d.get(k) is True]
            row = "%d) %s [%sfrom the %s]" % (i, d["text"], ", ".join(tags) + "; " if tags else "", d["source"])
            if self.cart_told and d.get("by_finish") is True and d.get("via_message") is True:
                row += (" (carried out: the shopper's reply noted the told cart with no objection or change, which "
                        "is the approval asked for; ordering is the next move)")
            rows.append(row)
        return "\n[instructions in force] " + " ".join(rows)

    # -- needs left (S5.14) and code values (06 6.1)
    def in_force(self):
        pool = self.rebuilt if self.rebuilt is not None else self.written + self.revealed
        return [x for x in pool if not self._gone(x) and not _p04_MONEY.search(str(x).lower())]

    def values_in_code(self, slot):
        out = []
        multi = len(self.products()) > 1
        for req in self.in_force():
            if ":" not in req: continue
            head = self._head(req)
            if multi and slot is not None:
                try:
                    own = self.s.mender.item_for(head)
                except Exception:
                    own = None
                if own is not None and own != slot: continue
            val = kit.strip_marks(self._val(req)).strip()
            if not val or kit.is_pointer(val): continue
            if kit.is_positive(val, head) or (len(val.split()) <= 5 and not kit.has_exclusion(val.lower())):
                out.append((kit.name_head(head).strip(), val))
        return out

    # -- hearing (S5.6-S5.13)
    def hear(self, lines, question=None):
        try:
            return self._hear(lines, question)
        except Exception as e:
            self.s.link.fault("hear_error", e)
            return ""

    def _hear(self, lines, question):
        s, L = self.s, self.s.link
        fresh = []
        for x in lines or []:
            if isinstance(x, str) and x.strip() and x not in self.heard and x not in fresh: fresh.append(x)
        if not fresh: return ""
        self.heard.update(fresh)
        if L.time_left() < Tune.Clock.CLOSING: return ""
        q = question.strip() if isinstance(question, str) and question.strip() else None
        # stage 2: the listener
        req = self.req_text()
        entry = ("asked: %s -> shopper: " % q if q else "shopper: ") + " / ".join(fresh)
        history = self.exchanges[-12:]
        self.exchanges.append(entry)
        parts = [PROMPTS["listener"], "ORIGINAL REQUEST (BACKGROUND):\n" + str(self.request), "REQUIREMENTS SO FAR:\n" + req]
        if history: parts.append("EARLIER EXCHANGES, OLDEST FIRST:\n" + "\n".join(history))
        if q: parts.append("THE ASSISTANT'S MESSAGE:\n" + q)
        parts.append("SHOPPER'S NEWEST WORDS:\n" + "\n".join(fresh))
        ans = L.consult("hear-shopper", "\n\n".join(parts), "listener")
        if not isinstance(ans, dict) or not ans:
            if L.mute or L.misses:
                self.heard.difference_update(fresh)
                if self.exchanges and self.exchanges[-1] == entry: self.exchanges.pop()
                self.parked.append((fresh, q, monotonic()))
            return ""
        needs = _p04_texts(ans.get("new_musts"))
        # stage 3: meta-only replies to our question
        if _p04_yes(ans.get("pace_only")) and q and not self._spelled(needs, fresh):
            needs = []
            ans.update({"new_musts": [], "dont_care": [], "dropped_musts": [], "pointer_refs": [], "musts_now": None,
                        "new_cap": None, "gone_off_sale": False, "answer_back": ""})
            self.meta_question = q
            L.note("meta_only", question=q[:300])
        # stage 4: the shopper's own property wording
        needs = self._swap(needs, ans, self._rename(needs, fresh), "need_renamed")
        ans["new_musts"] = needs
        # stage 5: lines with pointer phrases cut out
        masked = fresh
        try:
            got = s.tracer.fix_reply(ans, fresh, q)
            if got: masked = list(got)
        except Exception as e:
            L.fault("pointer_hook_error", e)
        needs = _p04_texts(ans.get("new_musts"))
        # stage 6: store spelling of word values
        try:
            needs = self._swap(needs, ans, self._literal(needs), "literal_swap")
        except Exception as e:
            L.fault("literal_value_error", e)
        # stage 7: record
        said = (req + " " + " ".join(self.exchanges) + " " + (q or "")).lower()
        self.newest = list(needs)
        rem = []
        nb = ans.get("musts_now")
        if isinstance(nb, list):
            self.rebuilt = [x.strip() for x in nb if isinstance(x, str) and x.strip() and self._backed(x, said)]
        for n in needs:
            if n not in self.revealed and self._backed(n, said):
                self.revealed.append(n)
                rem.append("need added: " + n)
        marked = set(_p04_texts(ans.get("pointer_musts")))
        refs = ans.get("pointer_refs") if isinstance(ans.get("pointer_refs"), list) else []
        for n in needs:
            if n in self.revealed and (kit.is_pointer(self._val(n)) or n in marked):
                self.pointer_news.append({"need": n, "refs": refs, "lines": " / ".join(fresh)})
        # stage 8: no-preference properties
        pool = self.rebuilt if self.rebuilt is not None else self.written + self.revealed
        ruled = {self._head(x).lower() for x in pool if not self._gone(x)}
        have = {self._norm(f) for f in self.freed}
        freed_now = []
        for f in _p04_texts(ans.get("dont_care")):
            if self._norm(f) in have or f.lower() in ruled: continue
            if self._pins(f):
                self.pinned += 1
                continue
            self.freed.append(f)
            have.add(self._norm(f))
            freed_now.append(f)
        # stage 9: drops, budget, off-sale, instructions, allowance
        widened, dropped_now = False, []
        cap = ans.get("new_cap")
        capnum = kit.is_num(cap)
        for d in _p04_texts(ans.get("dropped_musts")):
            if self._gone(d): continue
            if capnum and _p04_PRICE_TALK.search(d.lower()): continue
            if self._drop_ok(d, masked):
                self.dropped.append(d)
                dropped_now.append(d)
                rem.append("need dropped: " + d)
                if self.priority() != "none": widened = True
            else: L.note("drop_kept", need=d[:300])
        if capnum and cap > 0 and cap != self.budget:
            self.budget = cap
            rem.append("budget is now " + _p04_num(cap))
        if _p04_yes(ans.get("gone_off_sale")):
            rem.append("a product has gone off sale: remove its cart line and never order that product")
        try:
            added = self.add_duties(ans.get("duties"), "shopper" if q else "store")
            if added: rem.append("instruction added: " + "; ".join(added))
        except Exception as e:
            L.fault("instruction_error", e)
        lim = ans.get("new_limits")
        if isinstance(lim, dict):
            news = {k: v for k, v in lim.items() if kit.is_num(v)}
            if news:
                self.limits_news = news
                rem.append("limit changed: " + json.dumps(news, separators=(",", ":")))
        if rem:
            self.version += 1
            if widened: self.widened_at, self.widened_step = self.version, L.steps
        # stage 10: restore what the rebuilt list forgot; stage 11: widening notes
        if self.rebuilt is not None: self._restore(masked)
        if widened: rem.append(self._widen_note())
        # stage 12: close
        L.note("heard", question=bool(q), needs=len(needs), freed=len(freed_now), dropped=len(dropped_now),
               budget=capnum)
        back = ans.get("answer_back")
        self.reply_due = (back.strip() if isinstance(back, str) and not q and any("?" in x for x in fresh)
                          and not _p04_texts(ans.get("dont_care")) else "")
        gist = ans.get("gist")
        note = Tune.Marks.MEANING + (gist.strip() if isinstance(gist, str) else "")
        return note + (" (" + "; ".join(rem) + ")" if rem else "")

    @staticmethod
    def _backed(text, said):
        ws = re.findall(r"[a-z]{3,}", text.lower())
        return not ws or any(w in said for w in ws)

    def _swap(self, needs, ans, pairs, kind):
        if not pairs: return needs
        table = dict(pairs)
        if isinstance(ans.get("musts_now"), list):
            ans["musts_now"] = [table.get(x.strip(), x) if isinstance(x, str) else x for x in ans["musts_now"]]
        self.s.link.note(kind, needs=["%s -> %s" % p for p in pairs][:12])
        return [table.get(n, n) for n in needs]

    def _spelled(self, needs, fresh):
        low = " ".join(fresh).lower()
        for n in needs:
            if ":" not in n: continue
            pw = [w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z]+", kit.name_head(self._head(n)).lower())
                  if len(w) >= 3]
            vw = [w for w in re.findall(r"[a-z0-9]+", self._val(n).lower()) if not kit.is_link(w)]
            if pw and vw and all(w in low for w in pw + vw): return True
        return False

    def _rename(self, needs, fresh):
        try:
            toks = [re.findall(r"[a-z]+", x.lower()) for x in fresh]
            pairs = []
            for n in needs:
                if ":" not in n: continue
                head, val = n.split(":", 1)
                pw = re.findall(r"[a-z]+", kit.name_head(head).lower())
                k = len(pw)
                if not pw or any(len(w) < 3 for w in pw) or any(t[i:i + k] == pw for t in toks for i in range(len(t))):
                    continue
                run = next((t[i:i + k] for t in toks for i in range(len(t) - k + 1)
                            if t[i:i + k] != pw and all(a.startswith(b) for a, b in zip(t[i:i + k], pw))), None)
                if run:
                    tag = head[head.index("("):].strip() if "(" in head else ""
                    pairs.append((n, " ".join(run) + (" " + tag if tag else "") + ":" + val))
            return pairs
        except Exception:
            return []

    def _literal(self, needs):
        pairs = []
        for n in needs:
            if ":" not in n: continue
            head, val = n.split(":", 1)
            head, val = head.strip(), val.strip()
            if not val or re.search(r"\d", val) or kit.is_pointer(val): continue
            raw = re.findall(r"[a-z0-9]+", val.lower())
            bare = re.findall(r"[a-z0-9]+", kit.strip_emphasis(kit.strip_comparison(val.lower())).lower())
            if not bare or (bare == raw and kit.is_positive(val, head)): continue
            prop = kit.name_head(head)
            hit = None
            for f, vals in (self.s.judge.values_of_field(head) or {}).items():
                if kit.exact_property(f, prop):
                    hit = next((str(v).strip().lower() for v in vals or [] if re.findall(r"[a-z0-9]+", str(v).lower()) == bare), None)
                    if hit: break
            if hit:
                kit.add_literal(" ".join(bare))
                pairs.append((n, head + ": " + hit))
        return pairs

    @staticmethod
    def _phrases(lines):
        out = []
        for x in lines:
            w = re.findall(r"[a-z]+", str(x).lower())
            for k in (1, 2, 3):
                for i in range(len(w) - k + 1):
                    p = " ".join(w[i:i + k])
                    if len(p) >= 3 and not (k == 1 and kit.is_link(p)) and p not in out: out.append(p)
        return out

    def _drop_ok(self, entry, masked):
        try:
            prop = self._head(entry)
            tags, head = kit.tags_of(prop), kit.name_head(prop)
            def same(o):
                op = self._head(o)
                ot = kit.tags_of(op)
                return kit.property_match(head, kit.name_head(op)) and (not ot or ot == tags)
            held = [w for w in self.written if same(w)]
            if not held:
                held = [r for r in self.revealed if same(r) and r not in self.written and r not in self.newest]
                if not held or any(same(n) and self._norm(n) != self._norm(entry) for n in self.newest): return True
            ph = self._phrases(masked)
            for h in held:
                hh = kit.name_head(self._head(h))
                if any(kit.property_match(hh, p) for p in ph): return True
            others = self.written + self.revealed
            for h in held:
                hp = self._head(h)
                hh = kit.name_head(hp)
                val = kit.strip_marks(self._val(h)).strip()
                if not val or kit.is_pointer(val) or not kit.value_words(val): continue
                fields = self.s.judge.values_of_field(hp) or {}
                rivals = [kit.name_head(self._head(o)) for o in others
                          if not kit.property_match(kit.name_head(self._head(o)), hh)]
                for p in ph:
                    if any(kit.exact_property(p, r) for r in rivals): continue
                    for f, vals in fields.items():
                        if kit.exact_property(p, f) and kit.seen_in(val, " | ".join(str(v) for v in vals or [])):
                            self.s.link.note("drop_by_field", need=entry[:200], phrase=p)
                            return True
            return False
        except Exception:
            return True

    def _pins(self, asked):
        try:
            at = kit.tags_of(asked)
            ahead = kit.name_head(asked)
            akey = kit.property_slug(ahead)
            refs = _p04_texts(self._r("pointer_needs")) + [x for x in self.resolved if isinstance(x, str)]
            for need in self.written:
                if ":" not in need or self._gone(need): continue
                prop, val = self._head(need), kit.strip_marks(self._val(need)).strip()
                nt = kit.tags_of(prop)
                if at and nt and not (kit.kind_terms(at[-1]) & kit.kind_terms(nt[-1])): continue
                if not kit.is_positive(val, prop) or kit.tale_hint(kit.strip_marks(need).strip(), self.request, refs):
                    continue
                nkey = kit.property_slug(kit.name_head(prop))
                if kit.property_match(ahead, kit.name_head(prop)) or (len(nkey) >= 4 and akey.endswith(nkey)): return True
            return False
        except Exception:
            return False

    def _restore(self, masked):
        rb = self.rebuilt
        for r in self.revealed:
            h = self._head(r).lower()
            if not any(h in x.lower() for x in rb) and not any(h in d.lower() for d in self.dropped): rb.append(r)
        for w in self.written:
            try:
                if self._gone(w) or self._norm(w) in {self._norm(x) for x in rb} or self._drop_ok(w, masked): continue
                if any(self._same_slot(d, w) for d in self.dropped): continue
                if not any(self._same_slot(x, w) for x in rb): rb.append(w)
            except Exception:
                continue

    def _widen_note(self):
        if self.in_force():
            return ("WIDER NOW: that need is gone; a listing refused only for it is back in play and unseen listings "
                    "may beat the pick on price. Next: search this kind of item again without it, one short query per "
                    "value still needed (value first, then the kind word), reading every page; the cheapest qualifying "
                    "listing wins")
        shelf = ("SHELF OPEN: no property need is left, so every in-stock listing filed in the asked category fits "
                 "(accessories, parts and sets on that shelf too) and the lowest price on the whole shelf wins; rows "
                 "come in relevance or id order, never assume price order")
        hand = ("if no %s list appears, do it by hand: first pages of short searches on the category label, on each "
                "lone kind word of it and on each item name recurring in titles, plus the category listing's pages"
                % Tune.Marks.SWEEP_DONE)
        if self.priority() == "cheapest" and self.sweepable:
            return (shelf + ". Code fetches more of the shelf itself (category listing pages, searches on the label's "
                    "words and on object-kind words in titles); " + hand + ". Finish: cart the shelf's cheapest "
                    "in-stock row within budget as the only line, re-check its stock and price, then order")
        return shelf + ". " + hand[0].upper() + hand[1:] + ". Finish: order the cheapest in-stock item"


class Basket:
    """Our own count of the cart (02 S6)."""
    def __init__(self, s, tool=None, schema=None):
        self.s = s
        self.tool = tool
        props = schema.get("properties") if isinstance(schema, dict) else None
        self.props = props if isinstance(props, dict) else {}
        self.put_form = None
        self.adds = None
        self.last_view = None
        self.view_step = -1
        self.changed_step = 0
        self.used = False
        self.units = {}
        self.probe = {}
        self._spec = False

    @staticmethod
    def _types(p):
        t = p.get("type") if isinstance(p, dict) else None
        return [t] if isinstance(t, str) else ([x for x in t if isinstance(x, str)] if isinstance(t, list) else [])

    def _enum(self, k):
        p = self.props.get(k)
        return isinstance(p, dict) and "enum" in p

    # -- reading a call (S6.3)
    def form(self, args):
        return json.dumps({k: v for k, v in (args or {}).items() if self._enum(k)}, separators=(",", ":"),
                          ensure_ascii=False, default=str)

    def list_spec(self):
        if self._spec is False:
            self._spec = None
            for k, p in self.props.items():
                if not isinstance(p, dict) or not ("array" in self._types(p) or "items" in p): continue
                it = p.get("items")
                ip = it.get("properties") if isinstance(it, dict) else None
                if not isinstance(ip, dict): continue
                texts = [f for f, q in ip.items() if isinstance(q, dict) and "string" in self._types(q) and "enum" not in q]
                nums = [f for f, q in ip.items() if isinstance(q, dict) and {"integer", "number"} & set(self._types(q))]
                if texts and nums:
                    self._spec = (k, next((f for f in texts if "id" in f.lower()), texts[0]), nums[0])
                    break
        return self._spec

    def _listed(self, args):
        sp = self.list_spec()
        return sp if sp and isinstance((args or {}).get(sp[0]), list) else None

    def _count(self, args):
        for k, p in self.props.items():
            if isinstance(p, dict) and "enum" not in p and {"integer", "number"} & set(self._types(p)):
                v = args.get(k, p.get("default"))
                if kit.is_num(v): return v
        return None

    def count_key(self):
        # the argument a single-line call carries its units in (S6.3 count argument); None for a list-style tool
        if self.list_spec(): return None
        for k, p in self.props.items():
            if isinstance(p, dict) and "enum" not in p and {"integer", "number"} & set(self._types(p)): return k
        return None

    def lines_of(self, args):
        args = args if isinstance(args, dict) else {}
        sp = self._listed(args)
        if sp:
            out = []
            for e in args[sp[0]]:
                if isinstance(e, dict) and isinstance(e.get(sp[1]), str) and e[sp[1]].strip():
                    a = e.get(sp[2])
                    out.append((e[sp[1]].strip(), a if kit.is_num(a) else None))
            return out
        tgt = next((v.strip() for k, v in args.items()
                    if isinstance(v, str) and not self._enum(k) and v.strip() and not re.search(r"\s", v.strip())), None)
        return [(tgt, self._count(args))] if tgt else []

    @staticmethod
    def _first(lines): return next((x for x in lines if x[1] is None or x[1] > 0), lines[0] if lines else None)

    def target(self, args):
        t = self._first(self.lines_of(args))
        return t[0] if t else None

    def count_of(self, args):
        args = args if isinstance(args, dict) else {}
        if self._listed(args):
            t = self._first(self.lines_of(args))
            return t[1] if t else None
        return self._count(args)

    def is_cart_call(self, args): return self.put_form is None or self.form(args) == self.put_form

    # -- booking (S6.4-S6.7)
    def book(self, args, reply):
        self.used = True
        if isinstance(reply, dict) and reply.get("error"): return ""
        args = args if isinstance(args, dict) else {}
        put = self.is_cart_call(args)
        if self.put_form is None: self.put_form = self.form(args)
        if self._listed(args):
            self.adds = False
            units = {}
            for lid, a in self.lines_of(args):
                n = 1 if a is None else a
                if n > 0: units[lid] = units.get(lid, 0) + n
            self.units = units
            shown = self.shown_units(reply)
            if shown is not None: self._learn(shown)
            return self.summary()
        lid = self.target(args)
        if lid is None: return ""
        c = self.count_of(args)
        amt = 1 if c is None else c
        held = self.units.get(lid, 0)
        if not put: n = max(0, held - amt) if amt > 0 else 0
        elif lid in self.units and self.adds is not False:
            n = held + amt
            self.probe = {lid: amt}
        else: n = amt if amt > 0 else 0
        if n > 0: self.units[lid] = n
        else: self.units.pop(lid, None)
        shown = self.shown_units(reply)
        if shown is not None:
            if set(shown) <= {lid}: self._settle(lid, shown.get(lid, 0))
            else: self._learn(shown)
        elif (not put or amt == 0) and self.looks_empty(lid, reply):
            self.units.pop(lid, None)
            self.probe.pop(lid, None)
        return self.summary()

    def _judge(self, lid, shown):
        pv = self.probe.get(lid)
        if pv is None: return
        if shown == pv and self.units.get(lid, 0) != shown: self.adds = False
        elif shown > pv: self.adds = True

    def _settle(self, lid, shown):
        if lid in self.probe:
            self._judge(lid, shown)
            self.probe.pop(lid, None)
        if shown > 0: self.units[lid] = shown
        else: self.units.pop(lid, None)

    def _learn(self, shown):
        for lid in list(self.probe):
            if lid in shown: self._judge(lid, shown[lid])
        self.probe = {}
        self.units = {k: v for k, v in shown.items() if v > 0}

    def shown_units(self, reply):
        known = set(self.units) | set(self.probe)
        got = {}
        def take(d):
            cands = [v for v in d.values() if isinstance(v, str) and v and not re.search(r"\s", v)
                     and (v in known or kit.looks_like_id(v))]
            if not cands: return
            lab = next((v for v in cands if v in known), cands[0])
            nums = [(str(k), v) for k, v in d.items()
                    if isinstance(v, int) and not isinstance(v, bool) and 0 < v < 1000 and k != "price"]
            if not nums: return
            if len(nums) == 1: amt = nums[0][1]
            else:
                cnt = [(k, v) for k, v in nums if _p04_COUNTISH.search(k.lower()) and not _p04_MONEYISH.search(k.lower())]
                if len(cnt) == 1: amt = cnt[0][1]
                else:
                    pool = cnt or nums
                    cur = (self.units.get(lab), self.probe.get(lab))
                    eq = [v for k, v in pool if v in cur]
                    amt = eq[0] if len(eq) == 1 else min(v for k, v in pool)
            got[lab] = got.get(lab, 0) + amt
        kit.visit(reply, take)
        if not got or (known and not known & set(got)): return None
        return got

    def looks_empty(self, lid, reply):
        if lid not in self.units or not isinstance(reply, dict): return False
        lists = [v for v in reply.values() if isinstance(v, list)]
        nums = [v for v in reply.values() if kit.is_num(v)]
        if not lists or not nums or any(lists) or any(n != 0 for n in nums): return False
        text = json.dumps(reply, ensure_ascii=False, default=str)
        return not any(o in text for o in self.units if o != lid)

    def view(self, reply, step):
        self.last_view = reply
        self.view_step = step
        shown = self.shown_units(reply)
        if shown is not None: self._learn(shown)
        elif isinstance(reply, (dict, list)) and not self.ids_shown(reply): self.units = {}

    # -- summary and faults (S6.8)
    def held(self, lid): return self.units.get(lid, 0)

    def lines(self): return list(self.units)

    def ids_shown(self, text):
        t = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False, default=str)
        return any(lid in t for lid in self.units)

    def _listing(self): return ", ".join("%s x%s" % (k, _p04_num(v)) for k, v in self.units.items())

    def summary(self):
        if not self.units: return "[cart, counted by code] no lines."
        return "[cart, counted by code] " + self._listing() + "."

    def faults(self, qty, counts):
        if not self.units: return []
        lines = list(self.units.items())
        counts = list(counts or [])
        if len(counts) > 1:
            if len(lines) != len(counts):
                return ["%d products were requested but %d lines sit in the cart (%s): trim it to one entry per "
                        "requested product" % (len(counts), len(lines), self._listing())]
            have, want = sorted(v for _, v in lines), sorted(counts)
            if have != want:
                return ["line counts [%s], wanted [%s]: correct them" % (", ".join(_p04_num(x) for x in have),
                                                                      ", ".join(_p04_num(x) for x in want))]
            return []
        if len(lines) > 1:
            return ["%d separate lines sit in the basket (%s): clear all but your pick" % (len(lines), self._listing())]
        lid, n = lines[0]
        if n != qty:
            d = qty - n
            return ["quantity off on %s: %s held, %s needed - %s %s via the cart tool"
                    % (lid, _p04_num(n), _p04_num(qty), "add" if d > 0 else "remove", _p04_num(abs(d)))]
        return []

    def total(self): return float(sum((self.s.archive.price(k) or 0) * v for k, v in self.units.items()))

    def priced(self): return all(self.s.archive.price(k) is not None for k in self.units)



# ------------------------------------------------------------ p05_intake.py
# p05_intake.py - Intake: reading the request and applying the plan, status and clock lines, the story-hint and
# property-menu toolkit, reserves, phases and the late gate (spec 03 all; 06 5 code reading; 05 3.21; 11 3.2).


class Intake:
    """The errand's intake: request reading, the plan, the status line and the services built on them."""
    _NUMW = ("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty").split()
    _PLAIN = "Which <<FIELD>> would you like for the <<PRODUCT>>?"
    _MONEY = re.compile("(?:([$\u20ac\u00a3])\\s*)?(\\d{1,3}(?:,\\d{3})+|\\d+)(\\.\\d+)?")
    _OR_GROUP = re.compile(r"\(\s*or\b([^()]*)\)", re.I)
    _COPULA = re.compile(r"(?:\b(?:where|whose)\s+(?:(?:the|its|their)\s+)?|(?:^|[.!?,;:]\s+|\b(?:and|with|but)\s+)"
                         r"(?:the|its|their)\s+)([a-z][a-z'&/ -]{1,32}?)\s+(?:is|are|(?:has|have|needs?)\s+to\s+be|"
                         r"(?:must|should|can)\s+be)\s+(?:(?:either|just)\s+)?")
    _LATE = 3      # late reads or searches allowed in checkout mode (03 3.16, 3.18)
    _BACKUPS = 2   # backup-row reads per errand (03 3.18, 3.19)

    def __init__(self, s):
        self.s = s
        self._first_cap = None
        self._blocks = None
        self._menus = {}
        self._alt_pick = {}
        self._used_slots = set()
        self._first_done = False
        self._grants = 0
        self._late_ids = set()
        self._backups = 0
        self._guessed = None

    # ------------------------------------------------------------------ small helpers
    def _name(self, spec):
        f = spec.get("function") if isinstance(spec, dict) else None
        return str(f.get("name") or "") if isinstance(f, dict) else ""

    def _bare(self, name):
        return kit.name_head(str(name or "")).strip()

    def _nw(self, name):
        return [w for w in kit.property_terms(self._bare(name)) if len(w) >= 3 and not kit.is_link(w)]

    def _norm(self, v):
        return " ".join(re.findall(r"[a-z0-9]+", kit.low_plain(v)))

    def _prod(self, text):
        s = self.s
        if not s.multi():
            return None
        try:
            return s.mender.item_for(text)
        except Exception as e:
            s.link.fault("product_of_error", e)
            return None

    def _pname(self, slot):
        s = self.s
        try:
            if s.multi() and slot is not None:
                rows = s.ledger.products()
                return str(rows[slot].get("name") or "") if 0 <= slot < len(rows) else ""
            return str((s.ledger.reading or {}).get("item_kind") or "")
        except Exception as e:
            s.link.fault("product_name_error", e)
            return ""

    def _homes(self, slot):
        s = self.s
        if slot is None and s.multi():
            return set()
        hl = s.judge.home_labels() or []
        i = 0 if slot is None else slot
        return {kit.low(x) for x in hl[i]} if 0 <= i < len(hl) else set()

    def _num(self, tok, limits=False):
        """Digits, or a spelled number (one..ten; for limits also twelve, fifteen, twenty)."""
        if tok.isdigit():
            return int(tok)
        if tok in self._NUMW:
            n = self._NUMW.index(tok) + 1
            if n <= 10 or (limits and n in (12, 15, 20)):
                return n
        return None

    def _lid(self, x):
        if isinstance(x, dict):
            return str(x.get("id") or "")
        return str(x) if x is not None else ""

    def _text_of(self, msg):
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, list):
            return "".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
        return str(c or "")

    def _sig(self, args):
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _fmt(self, n):
        return str(int(n)) if kit.is_num(n) and n == int(n) else str(n)

    def _mkey(self, line):
        name, _, val = str(line).partition(":")
        return kit.name_head(name).strip().lower(), val.strip().lower()

    # ------------------------------------------------------------------ 3.1 counters and reserves
    def question_limit(self):
        s = self.s
        return s.msg_cap if s.msg_cap is not None else Tune.Ask.DEFAULT_MSGS

    def message_reserve(self):
        # v1.4: one message is kept back only for a stated step that is done by a message at the finish; otherwise
        # every message can go to the hidden-need questions (r8: the kept one went out as an unasked-for basket note)
        s = self.s
        if not s.role("message"): return 0
        try:
            fin = s.ledger.finish_duties()
            if any(d.get("via_message") is not None or d.get("by_finish") is not None for d in s.ledger.duties):
                return 1 if any(d.get("via_message") is True for d in fin) else 0
            # rules with no structured reading (code reading, bare procedures): any of them may need that message
            return 1 if fin else 0
        except Exception as e:
            s.link.fault("message_reserve_error", e)
            return 1

    def question_room(self):
        return max(0, self.question_limit() - self.message_reserve())

    def finish_reserve(self):
        s = self.s
        try:
            held = bool(s.ledger.finish_duties())
        except Exception as e:
            s.link.fault("finish_reserve_error", e)
            held = False
        return Tune.Finish.STEPS if (s.role("message") or held) else 0

    def room(self, n, pad=None):
        s = self.s
        if s.steps_left() <= n:
            return False
        return pad is None or s.link.time_left() > Tune.Clock.CLOSING + pad

    # ------------------------------------------------------------------ 3.2 allowance news
    def apply_limits(self, news):
        s = self.s
        try:
            if not isinstance(news, dict):
                return
            used, mc, sc = s.link.steps, s.msg_cap, s.step_cap
            tot, left = news.get("msgs_total"), news.get("msgs_left")
            if kit.is_num(tot):
                mc = max(s.msgs_sent, int(tot))
            elif kit.is_num(left):
                mc = s.msgs_sent + max(0, int(left))
            tot, left = news.get("steps_total"), news.get("steps_left")
            if kit.is_num(tot) and tot > used:
                sc = min(sc, int(tot))
            elif kit.is_num(left) and left >= 0:
                sc = min(sc, used + int(left))
            if mc != s.msg_cap or sc != s.step_cap:
                s.msg_cap, s.step_cap = mc, sc
                s.asking = bool(s.asking and s.msgs_sent < self.question_room())
                s.link.note("allowance", news=news, msg_cap=mc, step_cap=sc, asking=s.asking)
        except Exception as e:
            s.link.fault("allowance_error", e)

    # ------------------------------------------------------------------ 3.3 finish kind
    def finish_kind(self, name, args):
        s = self.s
        try:
            o, d = s.role("order"), s.role("decline")
            if not name or name not in (o, d):
                return None
            if o != d:
                return "order" if name == o else "decline"
            args = args if isinstance(args, dict) else {}
            vals = [v.lower() for v in args.values() if isinstance(v, str) and len(v) <= 40]
            for v in vals:
                if kit.is_decline_word(v):
                    return "decline"
                if kit.is_order_word(v):
                    return "order"
            props = s.schema(name).get("properties")
            for p in (props.values() if isinstance(props, dict) else ()):
                opts = p.get("enum") if isinstance(p, dict) else None
                if not isinstance(opts, list) or len(opts) != 2:
                    continue
                low = [kit.low_plain(x) for x in opts]
                for i in (0, 1):
                    if low[i] in vals:
                        if kit.is_order_word(low[1 - i]):
                            return "decline"
                        if kit.is_decline_word(low[1 - i]):
                            return "order"
            return "order"
        except Exception as e:
            s.link.fault("finish_kind_error", e)
            return None

    def finishing_tools(self):
        s = self.s
        return {t for t in (s.role("order"), s.role("decline")) if t}

    # ------------------------------------------------------------------ 3.5-3.8 clock, status line, nudges
    def time_line(self):
        c = Tune.Clock
        usable = max(0.0, self.s.link.time_left() - c.LAST_CALL)
        if 2 * usable > c.DEADLINE - c.LAST_CALL:
            return ""
        n = int(usable // c.CLOCK_STEP) * c.CLOCK_STEP
        return "about %d s left" % n if n > 0 else "under %d s left" % c.CLOCK_STEP

    def _safe(self, fn):
        try:
            return fn() or ""
        except Exception as e:
            self.s.link.fault("status_piece_error", e)
            return ""

    def _line(self, piece):
        piece = str(piece or "")
        return "" if not piece else (piece if piece.startswith("\n") else "\n" + piece)

    def state_banner(self):
        s = self.s
        try:
            try:
                s.ledger.cart_told = bool(s.warden.told_current())
            except Exception as e:
                s.link.fault("status_told_error", e)
            bits = ["[status] steps used %d of %d, %d left" % (s.link.steps, s.step_cap, s.steps_left()),
                    "messages sent %d" % s.msgs_sent + (" of %d" % s.msg_cap if s.msg_cap is not None else "")]
            clock = self.time_line()
            if clock:
                bits.append(clock)
            out = "; ".join(bits) + ".\n[requirements now] " + self._safe(s.ledger.req_text)
            for fn in (s.tracer.status_text, s.pricer.item_steps, s.ledger.duties_text):
                out += self._line(self._safe(fn))
            try:
                out += self._line(s.warden.limit_hint()) + self._line(s.warden.fit_hint())
            except Exception as e:
                s.link.fault("status_hint_error", e)
            if s.role("message") and s.msgs_sent < self.question_limit() and self.message_reserve():
                out += (" (The question plan keeps one message back for the finishing step the request asks for: "
                        "you may send it yourself, or code sends it at the finish.)")
            offered = self.offered_tools()
            if len(offered) < len(s.tools):
                out += ("\n[tools usable now] " + ", ".join(self._name(t) for t in offered) + ". A call to any other "
                        "tool is turned away and uses no step." + (" Code also plans and sends the hidden-need "
                                                                    "questions." if s.asking else ""))
            if s.basket.used:
                held = ["%s x%s" % (lid, self._fmt(n)) for lid, n in s.basket.units.items() if n]
                out += "\n[cart, counted by code] " + (", ".join(held) if held else "the cart has no lines")
            nudge = self._nudge()
            return out + ("\n" + nudge if nudge else "")
        except Exception as e:
            s.link.fault("status_error", e)
            return "[status] steps used %d of %d, %d left." % (s.link.steps, s.step_cap, s.steps_left())

    def _stage(self):
        s = self.s
        left = s.steps_left()
        if left <= 2:
            return "last", 0
        if left <= 5 or s.link.time_left() < Tune.Clock.CLOSING:
            return "finish", 0
        if s.link.budget_gone():
            return "budget", 0
        if not s.orders_tried and s.role("order") and s.role("cart"):
            due = self.due_margin()
            gap = left - due
            if 0 < gap <= Tune.Finish.ORDER_BY:
                return "low", due
            if gap <= 0:
                return "closed", due
        return "", 0

    def _nudge(self):
        stage, due = self._stage()
        if not stage:
            return ""
        multi = self.s.multi()
        pick = ("the verified fit if there is one; failing that, the lowest-priced listing that is in stock, inside "
                "the budget and whose record breaks no requirement")
        warn = "Decline only if the records prove that no listing can fit."
        if stage == "last":
            return ("LAST STEPS: if the cart already holds a candidate for " + ("every product" if multi else "the item")
                    + ", order now; if not, cart the best candidate (" + pick + ") and order. " + warn)
        if stage == "finish":
            return ("FINISH NOW: make the cart " + ("exactly one line per product" if multi else "a single line")
                    + ", taking " + pick + " and removing any sold-out line; check its stock once more, then order. "
                    + warn)
        if stage == "budget":
            return ("MODEL BUDGET USED: exploring is over. In the next few steps cart the best candidate "
                    + ("for each product at its own quantity" if multi else "for the item") + " - a verified fit if "
                    "there is one, otherwise the closest fit the records support - recheck its stock and price, and "
                    "order. Declining is right only when the records show that " + ("some product" if multi else
                                                                                    "the store") +
                    " has nothing able to meet the requirements.")
        if stage == "low":
            return ("[steps short] Few steps remain: cart the best verified fit and order within the next few steps. "
                    "Searches and detail reads stop once %d steps are left, and repairing a refused order costs about "
                    "4 steps." % due)
        return ("[checkout mode] Searching is over. The only reads still allowed are the cart's line, listings an item "
                "check pointed to and, while some product still lacks a verified fit, at most %d more reads or "
                "searches altogether. Cart the best verified fit, recheck its stock and price, and order; repairing a "
                "refused order still costs about 4 steps." % self._LATE)

    # ------------------------------------------------------------------ 3.9-3.10 reading the request
    def reader_blocks(self):
        s = self.s
        if self._blocks is None:
            rows = []
            for spec in s.tools:
                f = spec.get("function") or {}
                name = str(f.get("name") or "")
                rows.append("- %s: %s | arguments: %s" % (name, f.get("description") or "", kit.render(s.schema(name))))
            head = "ROLE: read-request\n" + PROMPTS["reader"] + "\n\nTOOLS:\n" + "\n".join(rows)
            shown = "REQUEST:\n" + s.request
            if s.opening is not None:
                shown += "\n\nOPENING OBSERVATION:\n" + kit.render(s.opening)
            self._blocks = [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
                            {"type": "text", "text": shown}]
        return self._blocks

    def parse_errand(self):
        s = self.s
        last = {}
        try:
            blocks = self.reader_blocks()
            ladder, cap = LINEUPS["reader"]
            ladder = list(ladder)
            t = 0
            while t < Tune.Tries.READER_ATTEMPTS:
                if t >= Tune.Tries.READER_CYCLE and not (
                        s.link.misses > 0 and s.link.fail_kind != "refused" and not s.link.service_down()
                        and s.link.usable(ladder) and s.link.time_left() >= Tune.Clock.REREAD_MIN):
                    break
                model = ladder[t % len(ladder)]
                msg = s.link.chat([{"role": "user", "content": blocks}], [model], cap)
                got = kit.json_from(self._text_of(msg)) if msg else None
                last = got if isinstance(got, dict) else {}
                if isinstance(last.get("tool_roles"), dict):
                    self.apply_reading(last)
                    return
                s.link.note("reading_retry", model=model, attempt=t)
                t += 1
        except Exception as e:
            s.link.fault("reading_error", e)
        self._fallback(last)

    def _fallback(self, last):
        s = self.s
        try:
            plan = self._code_plan()
            for k, v in last.items():
                if k != "tool_roles":
                    plan[k] = v
        except Exception as e:
            s.link.fault("code_reading_error", e)
            plan = {}
        s.read_fallback_at = time.monotonic()
        s.link.note("reading_fallback", roles=plan.get("tool_roles"), product=plan.get("item_kind"),
                    cap=plan.get("spend_cap"), items=len(plan.get("parts") or []))
        self.apply_reading(plan)

    # ------------------------------------------------------------------ 3.11 applying a reading
    def apply_reading(self, plan, again=False):
        s = self.s
        try:
            if not isinstance(plan, dict):
                plan = {}
            # 1 roles
            if not again:
                self._first_cap = s.step_cap
            old_cart = s.roles.get("cart")
            names = {self._name(t) for t in s.tools}
            raw = plan.get("tool_roles")
            roles = {str(r): t for r, t in raw.items() if isinstance(t, str) and t in names} if isinstance(raw, dict) else {}
            keep = bool(again and s.basket.used)
            if keep and old_cart:
                roles["cart"] = old_cart
            s.roles = roles
            if not keep:
                tool = roles.get("cart")
                s.basket = Basket(s, tool, s.schema(tool) if tool else None)
            self._fix_search()
            try:
                s.ledger.sweepable = bool(s.role("search")) and s.seeker.paging() is not None
            except Exception as e:
                s.link.fault("sweepable_error", e)
            # 2 counts
            mb = plan.get("msg_limit")
            if kit.is_num(mb) and mb >= 0:
                s.msg_cap = int(mb)
            s.ledger.reading = plan
            # 3 written counts
            if not plan.get("by_code"):
                try:
                    self._written_counts(plan)
                except Exception as e:
                    s.link.fault("written_counts_error", e)
            # 4 told instructions
            try:
                duties = plan.get("duties")
                s.ledger.add_duties(duties if isinstance(duties, list) else [], "request")
                rules = plan.get("procedures")
                s.ledger.add_duties([r for r in (rules if isinstance(rules, list) else []) if isinstance(r, str)], "request")
            except Exception as e:
                s.link.fault("duties_error", e)
            # 5 spend scale for products
            try:
                n = len(s.ledger.products())
                scale = 1 + Tune.Spend.PER_PRODUCT * max(0, min(Tune.Spend.EXTRA_PRODUCTS, n - 1))
            except Exception:
                scale = 1.0
            s.link.scale = scale
            # 6 split written requirements
            musts = plan.get("musts")
            items = [x.strip() for x in musts if isinstance(x, str) and x.strip()] if isinstance(musts, list) else []
            firm = [x for x in items if self._firm(x)]
            stray = [x for x in items if not self._firm(x)]
            s.ledger.written = list(firm)
            if stray:
                wishes = plan.get("wishes")
                plan["wishes"] = (list(wishes) if isinstance(wishes, list) else []) + stray
                plan["musts"] = list(firm)
            # 7 story values
            try:
                self._story(plan)
            except Exception as e:
                s.link.fault("story_values_error", e)
            # 8 references
            try:
                s.tracer.open_from_request()
            except Exception as e:
                s.link.fault("references_error", e)
            # 9 ask-first points
            self._trim_ask(plan)
            # 10 price
            if not again or s.ledger.version == 0:
                s.ledger.budget = self._amount(plan.get("spend_cap"))
            if plan and s.ledger.budget is None and s.role("message"):
                ask = plan.get("ask_before")
                plan["ask_before"] = [
                    "the most the shopper wants to spend on the whole order, because the request sets no price limit "
                    "(skip this point when the list already holds it)"] + (list(ask) if isinstance(ask, list) else [])
            # 11 steps
            if again and self._first_cap is not None:
                s.step_cap = self._first_cap
            sb = plan.get("step_limit")
            if kit.is_num(sb) and 0 < sb < s.step_cap:
                s.step_cap = int(sb)
            s.link.scale = scale * max(Tune.Spend.SCALE_MIN, min(1.0, Tune.Spend.STEP_REF / max(1, s.step_cap)))
            if again:
                s.ledger.version += 1
        except Exception as e:
            s.link.fault("apply_reading_error", e)

    def _firm(self, req):
        name, sep, val = req.partition(":")
        return bool(sep and name.strip() and val.strip())

    def _amount(self, v):
        if kit.is_num(v):
            return v
        if isinstance(v, str):
            m = re.search(r"\d[\d,]*(?:\.\d+)?", v)
            if m:
                f = float(m.group().replace(",", ""))
                return int(f) if f == int(f) else f
        return None

    def _measure_before(self, before):
        """Not a unit count: the figure follows a word of its own noun phrase, a bound word or a times sign."""
        b = before.rstrip()
        if b.endswith("\u00d7"):
            return True
        m = re.search(r"([a-z]+)$", b)
        if not m:
            return False
        w = m.group(1)
        if w == "x":
            return True
        if w in ("and", "or"):
            return False
        return kit.is_link(w) or kit.has_bound(w)

    def _written_counts(self, plan):
        low = kit.low(self.s.request)
        parts = plan.get("parts")
        rows = [r for r in parts if isinstance(r, dict)] if isinstance(parts, list) and len(parts) >= 2 else [plan]
        raised = []
        for r in rows:
            prod = str((r.get("item_kind") if r is plan else r.get("kind")) or "")
            ws = re.findall(r"[a-z0-9]+", prod.lower())
            if not ws:
                continue
            head = ws[0][:-1] if len(ws[0]) > 3 and ws[0].endswith("s") else ws[0]
            figs = set()
            pat = r"(?<![a-z0-9_.$-])(\d+|[a-z]+)\s+" + re.escape(head) + r"(?:s|es)?\b"
            for m in re.finditer(pat, low):
                n = self._num(m.group(1))
                if n is not None and not self._measure_before(low[:m.start()]):
                    figs.add(n)
            if len(figs) != 1:
                continue
            n = figs.pop()
            q = r.get("units")
            if 2 <= n <= 99 and not (kit.is_num(q) and q > 1):
                r["units"] = n
                raised.append("%s x%d" % (prod[:40], n))
        if raised:
            self.s.link.note("written_counts", raised=raised)

    def _story(self, plan):
        s = self.s
        raw = plan.get("story_hints")
        if isinstance(raw, dict):
            flat = []
            for k, v in raw.items():
                for x in (v if isinstance(v, list) else [v]):
                    flat.append(x if isinstance(x, str) and ":" in x else "%s: %s" % (k, x))
            raw = flat
        hints = [str(x).strip() for x in (raw if isinstance(raw, list) else [])
                 if (isinstance(x, str) and x.strip()) or kit.is_num(x)]
        hints = [kit.split_hint(h) for h in hints]
        reader = list(hints)
        refs = [x for x in (plan.get("pointer_needs") or []) if isinstance(x, str)]
        written = list(s.ledger.written)
        guesses = [w for w in written if kit.tale_hint(w, s.request, refs)]
        demote = []
        if guesses and len(written) >= 2 and 2 * len(guesses) <= len(written):
            gone = {self._mkey(g) for g in guesses}

            def drop(lst):
                return [m for m in lst if not (isinstance(m, str) and self._mkey(m) in gone)]
            s.ledger.written = drop(written)
            if isinstance(plan.get("musts"), list):
                plan["musts"] = drop(plan["musts"])
            for row in (plan.get("parts") if isinstance(plan.get("parts"), list) else []):
                if isinstance(row, dict) and isinstance(row.get("musts"), list):
                    row["musts"] = drop(row["musts"])
            s.link.note("story_guesses", moved=guesses)
            for g in guesses:
                name, _, val = g.partition(":")
                demote.append("%s: the story suggests %s (not stated as a value; ask)" % (name.strip(), val.strip()))
        promoted = set()
        try:
            forms, seen = [], list(s.ledger.written)
            for h in reader:
                got = self._stated_form(h, plan, seen)
                if got:
                    forms.append((h, got[0], got[1]))
                    seen.append(got[0])
            parts = [r for r in (plan.get("parts") if isinstance(plan.get("parts"), list) else []) if isinstance(r, dict)]
            for h, form, owner in forms:
                s.ledger.written.append(form)
                if not isinstance(plan.get("musts"), list):
                    plan["musts"] = []
                plan["musts"].append(form)
                kinds = kit.kind_terms(owner)
                for row in parts:
                    if isinstance(row.get("musts"), list) and (
                            len(parts) == 1 or kit.kind_terms(str(row.get("kind") or "")) & kinds):
                        row["musts"].append(form)
                        break
                s.link.note("hint_promoted", hint=h, requirement=form)
                promoted.add(h)
        except Exception as e:
            s.link.fault("hint_promote_error", e)
        plan["story_hints"] = ([h for h in hints if h not in promoted] + demote)[:12]

    def _stated_form(self, hint, plan, written):
        subject, sep, detail = hint.partition(":")
        prop = self._bare(subject)
        if not sep or not prop or kit.is_junk_key(prop):
            return None
        tags = kit.tags_of(subject)
        parts = [r for r in (plan.get("parts") if isinstance(plan.get("parts"), list) else []) if isinstance(r, dict)]
        owners = [tags[-1]] if tags else ([str(r.get("kind")) for r in parts if r.get("kind")]
                                          or [str(plan.get("item_kind") or "")])
        word = owner = ""
        for o in owners:
            w = kit.head_value(detail, self.s.request, o)
            if w:
                word, owner = w, o
                break
        if not word:
            return None
        kinds = kit.kind_terms(owner)
        for r in written:
            rn, _, rv = str(r).partition(":")
            rt = kit.tags_of(rn)
            if rt and kinds and not (kit.kind_terms(rt[-1]) & kinds):
                continue
            if kit.property_match(self._bare(rn), prop) or re.search(r"\b%s\b" % re.escape(word.lower()), rv.lower()):
                return None
        return "%s: %s" % (subject.strip(), word), owner

    def _trim_ask(self, plan):
        duties = plan.get("duties")
        marks = []
        for d in (duties if isinstance(duties, list) else []):
            if isinstance(d, dict) and d.get("by_finish") is True and d.get("before_change") is not True:
                m = set(re.findall(r"[a-z]{4,}", kit.low(d.get("text"))))
                if m:
                    marks.append(m)
        ask = plan.get("ask_before")
        if not marks or not isinstance(ask, list):
            return
        keep, moved = [], []
        for a in ask:
            ws = set(re.findall(r"[a-z]{4,}", kit.low(a)))
            shared = max(len(ws & m) for m in marks)
            if ws and (shared >= 2 or 2 * shared >= len(ws)):
                moved.append(str(a)[:120])
            else:
                keep.append(a)
        if moved:
            plan["ask_before"] = keep
            self.s.link.note("ask_first_moved", moved=moved)

    # ------------------------------------------------------------------ 06 5 code reading
    def rule_reading(self):
        try:
            return self._code_plan()
        except Exception as e:
            self.s.link.fault("code_reading_error", e)
            return {}

    def _code_plan(self):
        s = self.s
        low = kit.low(s.request)
        plan = {"by_code": True, "procedures": [], "wishes": [], "ask_before": [], "story_hints": [],
                "pointer_needs": [], "rule_measure": "", "units": 1}
        got = self.budget_from_text(s.request)
        plan["spend_cap"], plan["money_unit"] = got if got else (None, None)
        plan["stock_only"] = bool(re.search(r"\bstock\b|\bavailable\s+now\b", low))
        cheap = re.search(r"\bcheapest\b|\b(?:lowest|least)\b[^.]{0,20}?\b(?:price|priced|pricing|cost|expensive)\b|"
                          r"\bsmallest\s+(?:total|price)\b", low)
        plan["pick_rule"] = "cheapest" if cheap else "none"
        words = self._tool_words("message")
        n = self._limit(low, words | {"message"}, words, 0)
        if n is not None:
            plan["msg_limit"] = n
        n = self._limit(low, self._step_words(), set(), Tune.Finish.ORDER_BY + 3)
        if n is not None:
            plan["step_limit"] = n
        plan["tool_roles"] = self.guess_roles()
        found = self._category_products(low)
        if len(found) >= 2:
            plan["parts"] = [{"kind": k, "units": q, "musts": r, "with": None} for k, q, r in found]
            plan["item_kind"] = " and ".join(k for k, q, r in found)
            plan["musts"] = [x for k, q, r in found for x in r]
            return plan
        if found:
            plan["item_kind"], plan["units"] = found[0][0], found[0][1]
        else:
            plan["item_kind"] = self.best_label(kit.kind_terms(s.request))
            plan["units"] = self._unit_count(low, plan["item_kind"])
        plan["musts"] = ["%s: %s" % pv for pv in self._phrases(low)]
        return plan

    def _tool_of(self, role):
        return self.s.role(role) or self.guess_roles().get(role)

    def _tool_words(self, role):
        """Words the store's own tool for a role uses (its name and description), for the code reading."""
        tool = self._tool_of(role)
        spec = self.s.spec(tool) if tool else None
        if not spec:
            return set()
        f = spec.get("function") or {}
        text = kit.uncamel(str(f.get("name") or "")).replace("_", " ") + " " + str(f.get("description") or "")
        return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and not kit.is_link(w)}

    def _step_words(self):
        """The noun of the step allowance, read from the policy view's own numeric keys (e.g. a max-steps key)."""
        out = set()
        for k, v in (self.s.view or {}).items():
            if kit.is_num(v):
                out |= {w for w in re.findall(r"[a-z]+", kit.uncamel(str(k)).replace("_", " ").lower())
                        if len(w) >= 4 and not kit.is_link(w) and not kit.has_bound(w)}
        return out

    def _noun(self, word, pool):
        return any(kit.kind_match(word, p) for p in pool)

    def _limit(self, low, nouns, cues, floor):
        if not nouns:
            return None
        for m in re.finditer(r"(?<![a-z0-9_.$-])(\d+|[a-z]+)\b", low):
            n = self._num(m.group(1), True)
            if n is None:
                continue
            rest = low[m.end():]
            w = re.match(r"\s+([a-z][\w'-]*)(?:\s+([a-z][\w'-]*))?(?:\s+([a-z][\w'-]*))?", rest)
            if not w:
                continue
            hit = next((i for i, x in enumerate(w.groups()) if x and self._noun(x, nouns)), None)
            if hit is None:
                continue
            before = re.findall(r"[a-z']+", low[max(0, m.start() - 60):m.start()])[-4:]
            after = re.findall(r"[a-z']+", rest[w.end(hit + 1):][:40])[:4]
            if (before and kit.has_bound(" ".join(before))) or (after and kit.has_bound(" ".join(after))) or any(
                    self._noun(x, cues) for x in before):
                return n if n >= floor else None
        return None

    def budget_from_text(self, text):
        """Money figure of a budget phrase -> (amount, unit) or None (06 5.2, cue sets from kit and the tools)."""
        s = self.s
        try:
            low = kit.low(text)
            nouns = self._tool_words("message") | {"message"} | self._step_words() | self._label_words()
            order = {w for w in re.findall(r"[a-z]+", kit.uncamel(str(self._tool_of("order") or "")).replace(
                "_", " ").lower()) if len(w) >= 4 and not kit.is_link(w)}
            best = None
            for sent in re.split(r"(?<=[.!?;])\s+", low):
                bonus = 1 if any(re.search(r"\b%s\b" % re.escape(w), sent) for w in order) else 0
                for m in self._MONEY.finditer(sent):
                    sign = m.group(1)
                    before, after = sent[:m.start()], sent[m.end():]
                    if re.match("\\s*(?:x(?![a-z])|\u00d7|%|[:/-]\\s*\\d)", after):
                        continue
                    nxt = re.match(r"\s*([a-z][a-z-]*)(?:\s+([a-z][a-z-]*))?(?:\s+([a-z][a-z-]*))?", after)
                    ws = [x for x in (nxt.groups() if nxt else ()) if x]
                    if any(x == "of" or self._noun(x, nouns) for x in ws):
                        continue
                    if not sign and re.search(r"(?:[\d.,:/-]|(?<![a-z])x)\s*$", before):
                        continue
                    score = bonus + (3 if sign else 0)
                    prev = re.findall(r"[a-z]+", before[-60:])[-2:]
                    if prev and kit.has_bound(" ".join(prev)):
                        score += 2
                    if ws and kit.has_bound(" ".join(ws)):
                        score += 2
                    if score < 2:
                        continue
                    amt = float(m.group(2).replace(",", "") + (m.group(3) or ""))
                    amt = int(amt) if amt == int(amt) else amt
                    u = re.match(r"\s*([a-z.]+)(?:\s+([a-z.]+))?", after)
                    unit = [x.replace(".", "") for x in (u.groups() if u else ()) if x]
                    unit = " ".join(x for x in unit if len(x) >= 2 and not kit.is_link(x)) or sign or None
                    if best is None or (score, amt) > (best[0], best[1]):
                        best = (score, amt, unit)
            return (best[1], best[2]) if best else None
        except Exception as e:
            s.link.fault("budget_text_error", e)
            return None

    def _category_list(self):
        tool = self._tool_of("search")
        props = self.s.schema(tool).get("properties") if tool else None
        best = []
        for p in (props.values() if isinstance(props, dict) else ()):
            opts = p.get("enum") if isinstance(p, dict) else None
            if isinstance(opts, list) and len(opts) >= 4 and len(opts) > len(best):
                best = [str(x) for x in opts]
        return best

    def _label_words(self):
        out = set()
        for lab in self._category_list():
            out |= kit.kind_terms(lab)
        return out

    def _toks(self, low):
        out = []
        for m in re.finditer(r"[a-z]+", low):
            w = m.group()
            if len(w) >= 3 and not kit.is_link(w):
                out.append((w.rstrip("s"), m.start(), m.end()))
        return out

    def _count_before(self, text):
        for m in reversed(list(re.finditer(r"(?<![a-z0-9_.$-])(\d+|[a-z]+)", text))):
            n = self._num(m.group(1))
            if n is not None and re.fullmatch("\\s*(?:x|\u00d7)?\\s+(?:(?:of|[a-z]{3,})\\s+){0,4}", text[m.end():]):
                return n
        return None

    def _category_products(self, low):
        """06 5.7: products named in the request from the search tool's category list, each with a count."""
        cats = self._category_list()
        if not cats:
            return []
        toks = self._toks(low)
        seq = [t for t, a, b in toks]
        found = []
        for lab in cats:
            lt = [t for t, a, b in self._toks(kit.low(lab))]
            k = len(lt)
            if not k:
                continue
            for i in range(len(toks) - k + 1):
                if seq[i:i + k] != lt:
                    continue
                a, b = toks[i][1], toks[i + k - 1][2]
                if len(low[a:b].split()) > 2 * k:
                    continue
                n = self._count_before(low[max(0, a - 56):a])
                if n is None or not 1 <= n <= 99:
                    continue
                found.append((a, b, kit.low(lab), n))
                break
        found.sort()
        out = []
        for j, (a, b, name, n) in enumerate(found):
            if j + 1 < len(found):
                end = found[j + 1][0]
            else:
                stop = re.search(r"[.!?](?:\s|$)", low[b:])
                end = b + stop.start() if stop else len(low)
            out.append((name, n, ["%s (%s): %s" % (p, name, v) for p, v in self._phrases(low[a:end])]))
        return out

    def _unit_count(self, low, label):
        """06 5.8 written count: '<number> unit(s)', or a figure written right before the label's name."""
        pats = [r"units?\b"]
        pat = kit.name_regex(label) if label else ""
        if pat:
            pats.append(pat)
        for p in pats:
            for m in re.finditer(r"(?<![a-z0-9_.$-])(\d+|[a-z]+)\s+" + p, low):
                n = self._num(m.group(1))
                if n is not None:
                    return n if 1 <= n <= 99 else 1
        return 1

    def _phrases(self, low):
        """06 5.9 requirement phrases by the copula pattern; kit's link, junk-key and positive-value tests filter."""
        out = []
        for m in self._COPULA.finditer(low):
            prop = m.group(1).strip()
            rest = low[m.end():]
            run = re.match(r"[^,.;:!?()]{1,60}", rest)
            if not run:
                continue
            ws, cut = run.group(0).split(), False
            keep = []
            for i, w in enumerate(ws):
                if w == "but" or (w == "and" and i + 1 < len(ws) and (kit.is_link(ws[i + 1]) or ws[i + 1][:1].isdigit())):
                    cut = True
                    break
                keep.append(w)
            ended = len(run.group(0)) == len(rest) or rest[len(run.group(0))] in ",.;:!?()"
            val = " ".join(keep).strip()
            if not val or not (cut or ended) or len(prop.split()) > 4 or len(keep) > 4:
                continue
            if any(kit.is_link(x) for x in re.findall(r"[a-z]+", prop)) or kit.is_junk_key(prop) or kit.is_link(keep[0]):
                continue
            if kit.is_pointer(val) or kit.is_pointer(rest[len(run.group(0)):][:60]):
                continue
            if kit.is_positive(val, prop):
                out.append((prop, val))
        return out

    # ------------------------------------------------------------------ 06 5.5-5.6 tool roles
    def _is_free(self, p):
        if not isinstance(p, dict) or p.get("enum"):
            return False
        t = p.get("type")
        return t == "string" or (isinstance(t, list) and "string" in t)

    def _idish(self, arg):
        ws = kit.words(kit.uncamel(str(arg)))
        return any(w in ("id", "ids", "ref", "code", "sku") for w in ws) or str(arg).lower().endswith("id")

    def _cards(self):
        out = []
        for spec in self.s.tools:
            f = spec.get("function") or {}
            name = str(f.get("name") or "")
            sch = self.s.schema(name)
            props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
            req = [r for r in sch.get("required") if r in props] if isinstance(sch.get("required"), list) else []
            head = kit.uncamel(name).replace("_", " ").lower()
            text = head + " | " + str(f.get("description") or "").lower()
            ps = [p if isinstance(p, dict) else {} for p in props.values()]
            paired = False
            for r in req:
                opts = props[r].get("enum") if isinstance(props[r], dict) else None
                if isinstance(opts, list):
                    lows = [kit.low_plain(o) for o in opts]
                    if any(kit.is_order_word(o) for o in lows) and any(kit.is_decline_word(o) for o in lows):
                        paired = True
            out.append({"name": name, "head": head, "text": text, "props": props, "req": req, "paired": paired,
                        "free": [k for k, p in props.items() if self._is_free(p)],
                        "array": any(p.get("type") == "array" for p in ps), "choice": any(p.get("enum") for p in ps),
                        "lines": any(p.get("type") == "array" and isinstance(p.get("items"), dict)
                                     and p["items"].get("type") == "object" for p in ps),
                        "count": any(isinstance(props[r], dict) and props[r].get("type") in ("integer", "number")
                                     for r in req)})
        return out

    def _cue(self, card, role):
        return bool(re.search(r"\b" + role[:6], card["text"]))

    def _fits(self, role, c, cued):
        one = len(c["req"]) == 1 and c["req"][0] in c["free"]
        if role == "search":
            if self._cue(c, "message") or not (c["free"] or c["choice"]):
                return False
            return cued or ((any(not self._idish(a) for a in c["free"]) or c["choice"])
                            and (len(c["props"]) >= 2 or c["choice"]))
        if role == "message":
            return one and (cued or not self._idish(c["req"][0]))
        return one and (cued or self._idish(c["req"][0]))

    def guess_roles(self):
        if self._guessed is None:
            try:
                self._guessed = self._guess()
            except Exception as e:
                self.s.link.fault("guess_roles_error", e)
                self._guessed = {}
        return dict(self._guessed)

    def _guess(self):
        cards = self._cards()
        roles, taken = {}, set()

        def give(role, c):
            roles[role] = c["name"]
            taken.add(c["name"])
        for c in cards:
            if c["paired"]:
                roles["order"] = roles["decline"] = c["name"]
                taken.add(c["name"])
                break
        for field, needs_final in (("head", False), ("text", True)):
            for c in cards:
                if c["name"] in taken or (needs_final and (c["req"] or not re.search(r"\bfinal\b", c["text"]))):
                    continue
                for role, test in (("decline", kit.is_decline_word), ("order", kit.is_order_word)):
                    if role not in roles and test(c[field]):
                        give(role, c)
                        break
        for c in cards:
            if c["name"] in taken or not self._cue(c, "cart"):
                continue
            role = "cart" if (c["req"] or c["array"]) else "cart_view"
            if role not in roles:
                give(role, c)
        if "cart" not in roles:
            c = next((c for c in cards if c["name"] not in taken and (c["lines"] or c["count"])), None)
            if c:
                give("cart", c)
        if "cart_view" not in roles and "cart" in roles:
            cw = set(re.findall(r"[a-z]{4,}", next(c["text"] for c in cards if c["name"] == roles["cart"])))
            cw = {w for w in cw if not kit.is_link(w)}
            c = next((c for c in cards if c["name"] not in taken and not c["req"] and not c["array"]
                      and set(re.findall(r"[a-z]{4,}", c["text"])) & cw), None)
            if c:
                give("cart_view", c)
        for cued in (True, False):
            for role in ("search", "message", "details", "view"):
                if role in roles:
                    continue
                c = next((c for c in cards if c["name"] not in taken and (not cued or self._cue(c, role))
                          and self._fits(role, c, cued)), None)
                if c:
                    give(role, c)
        return roles

    def _searchy(self, c):
        return (self._cue(c, "search") and self._fits("search", c, True)) or self._fits("search", c, False)

    def _capped_browser(self, c):
        free = c["free"]
        cat = [a for a in free if re.search(r"categ|shelf|depart|aisle", a, re.I)]
        cur = [a for a in free if re.search(r"cursor|token|next|page", a, re.I)]
        cap = [a for a in c["props"] if re.search(r"price|cost|ceil|max|budget", a, re.I)]
        return bool(cat and cur and cap and set(c["req"]) <= set(cat + cur + cap))

    def _fix_search(self):
        s = self.s
        try:
            cur = s.roles.get("search")
            if not cur:
                return
            cards = self._cards()
            mine = next((c for c in cards if c["name"] == cur), None)
            if mine and mine["free"] and not self._capped_browser(mine):
                return
            holders = set(s.roles.values())
            for c in cards:
                if c["name"] not in holders and c["free"] and self._searchy(c):
                    s.roles["search"] = c["name"]
                    s.link.note("search_role", old=cur, new=c["name"])
                    return
        except Exception as e:
            s.link.fault("search_role_error", e)

    def best_label(self, words):
        try:
            want = set(words) if isinstance(words, (set, frozenset, list, tuple)) else kit.kind_terms(str(words or ""))
            best, top = "", (0, 0.0)
            for lab in self._category_list():
                own = kit.kind_terms(lab)
                shared = sum(1 for w in own if kit.kind_present(w, want))
                if shared and (shared, shared / len(own)) > top:
                    best, top = lab, (shared, shared / len(own))
            return best
        except Exception as e:
            self.s.link.fault("best_label_error", e)
            return ""

    # ------------------------------------------------------------------ 3.12 story-hint and menu toolkit
    def _hints(self):
        h = (self.s.ledger.reading or {}).get("story_hints")
        return [x for x in h if isinstance(x, str)] if isinstance(h, list) else []

    def _hint_rows(self, slot, loose=False):
        out = []
        multi = self.s.multi()
        for h in self._hints():
            front, _, tail = h.partition(":")
            if multi:
                own = self._prod(front)
                if slot is None:
                    if not loose and own is not None:
                        continue
                elif own is not None and own != slot:
                    continue
            out.append((h, front.strip(), tail))
        return out

    def _rows(self, slot):
        return [r for r in self.product_menu(slot).split("\n") if r.startswith("- ")]

    def _key(self, row):
        return row.lstrip("- ").split(" (in ")[0].strip()

    def _samples(self, row):
        # rows with a bracketed note carry no "):" and so give no samples (kept as built, 03 7.6)
        return [x.strip() for x in row.split("):", 1)[1].split("|")] if "):" in row else []

    def product_menu(self, slot):
        s = self.s
        try:
            n = len(s.archive.obs)
            got = self._menus.get(slot)
            if got and got[0] == n:
                return got[1]
            text = ""
            if s.multi():
                homes = self._homes(slot)
                for lab in s.archive.shelves_opened():
                    if kit.low(lab) in homes:
                        text = s.archive.menu(lab)[0]
                        break
            else:
                own = s.judge.home_shelf()
                text = s.archive.menu(own)[0] if own else s.archive.menu()[0]
            self._menus[slot] = (n, text or "")
            return text or ""
        except Exception as e:
            s.link.fault("product_menu_error", e)
            return ""

    def menu_row(self, slot, field):
        try:
            want = self._bare(field)
            if want:
                for row in self._rows(slot):
                    if kit.exact_property(self._key(row), want):
                        return row
        except Exception as e:
            self.s.link.fault("menu_row_error", e)
        return ""

    def row_values(self, slot, field):
        return {self._norm(x) for x in self._samples(self.menu_row(slot, field))} - {""}

    def field_alike(self, a, b, slot=None):
        try:
            if kit.exact_property(self._bare(a), self._bare(b)):
                return True
            return bool(kit.property_match(self._bare(a), self._bare(b))
                        and self.row_values(slot, a) & self.row_values(slot, b))
        except Exception as e:
            self.s.link.fault("same_field_error", e)
            return False

    def hinted(self, slot, field):
        try:
            want = self._bare(field)
            single = not self.s.multi()
            for h in self._hints():
                front = h.split(":", 1)[0]
                if kit.exact_property(self._bare(front), want):
                    if slot is None or single or self._prod(front) in (slot, None):
                        return True
        except Exception as e:
            self.s.link.fault("hinted_error", e)
        return False

    def detailed_reads(self, slot):
        s = self.s
        try:
            if s.multi():
                return sum(s.archive.rich_count(h) for h in self._homes(slot))
            own = s.judge.home_shelf()
            return s.archive.rich_count(own) if own else s.archive.rich_count()
        except Exception as e:
            s.link.fault("rich_reads_error", e)
            return 0

    def hint_has_no_key(self, slot, field, flagged=False):
        try:
            want = self._bare(field)
            if not want or (not flagged and not self.hinted(slot, field)):
                return False
            if self.menu_row(slot, want) or self.detailed_reads(slot) < 2:
                return False
            return not any(kit.property_match(self._key(r), want) for r in self._rows(slot))
        except Exception as e:
            self.s.link.fault("no_key_error", e)
            return False

    def hint_tag(self, hint):
        try:
            front = str(hint).split(":", 1)[0]
            return hint + " [no key]" if self.hint_has_no_key(self._prod(front), front, True) else hint
        except Exception as e:
            self.s.link.fault("hint_mark_error", e)
            return hint

    def tail_alts(self, tail):
        out = []
        for g in self._OR_GROUP.findall(str(tail or "")):
            out += [p.strip() for p in re.split(r",|\bor\b|/", g, flags=re.I) if p.strip()]
        return out

    def names_nest(self, a, b):
        x, y = set(self._nw(a)), set(self._nw(b))
        return bool(x and y and (x <= y or y <= x))

    def _log_parts(self, full):
        s = self.s
        multi = s.multi()
        out = []
        for e in s.question_log:
            if not isinstance(e, str):
                continue
            label = e.split(":", 1)[0]
            label = label[2:] if label.startswith("- ") else label
            if full:
                parts = [p.strip() for p in re.split(r";|\band\b", re.sub(r"\s*\[[^\]]*\]", "", label)) if p.strip()]
            elif "[not answered]" in label:
                continue
            else:
                parts = label.split(";")
            asked = re.search(r'asked "(.*?)" -> ', e)
            for p in parts:
                own = self._prod(p) if multi else None
                if own is None and multi and asked:
                    own = self._prod(asked.group(1))
                out.append((p, own))
        return out

    def inside_question(self, slot, subject):
        try:
            single = not self.s.multi()
            return any((slot is None or single or own is None or own == slot) and self.names_nest(p, subject)
                       for p, own in self._log_parts(False))
        except Exception as e:
            self.s.link.fault("asked_inside_error", e)
            return False

    def requested_by(self, slot, field):
        try:
            want = self._bare(field)
            single = not self.s.multi()
            return any((slot is None or single or own == slot) and kit.exact_property(self._bare(p), want)
                       for p, own in self._log_parts(True))
        except Exception as e:
            self.s.link.fault("asked_for_error", e)
            return False

    def pending_hints(self, slot=None):
        s = self.s
        fresh, reopened = [], []
        try:
            multi = s.multi()
            for h in self._hints():
                front, _, tail = h.partition(":")
                field = self._bare(front)
                if not field:
                    continue
                own = self._prod(front) if multi else None
                if slot is not None and own is not None and slot != own:
                    continue
                if s.mender.open_property(slot, field, row=self.menu_row(slot, field), by_samples=True) and \
                        not self.inside_question(slot, field):
                    fresh.append(h)
                else:
                    alt = self.quiet_alt(slot, h, field)
                    if alt:
                        reopened.append(alt)
        except Exception as e:
            s.link.fault("open_hints_error", e)
        return reopened + fresh

    def quiet_alt(self, slot, hint, subject):
        s = self.s
        try:
            front, _, tail = str(hint).partition(":")
            if hint not in self._alt_pick:
                multi = s.multi()
                quiet = any((self.field_alike(f, subject, slot) or self.names_nest(f, subject))
                            and (not multi or self._prod(f) in (slot, None))
                            for f in (str(x) for x in (s.ledger.freed or [])))
                if not quiet:
                    return ""
                for alt in self.tail_alts(tail):
                    if kit.is_junk_key(alt) or kit.exact_property(self._bare(alt), self._bare(subject)):
                        continue
                    row = self.menu_row(slot, alt)
                    if row:
                        self._alt_pick[hint] = self._key(row)
                        break
            key = self._alt_pick.get(hint, "")
            if not key or key.lower() in s.spent_props or not self.open_for_planning(slot, key):
                return ""
            tag = re.search(r"\(([^()]*)\)\s*$", front)
            label = key + (" (%s)" % tag.group(1).strip() if tag else "")
            rest = re.sub(r"\s*\(\s*or\b[^()]*\)", "", tail, flags=re.I).strip()
            return "%s: %s (the shopper had no preference on %s, the first property of this hint)" % (label, rest, subject)
        except Exception as e:
            s.link.fault("quiet_alt_error", e)
            return ""

    def hint_bound(self, slot, field):
        try:
            want = self._bare(field)
            nw = self._nw(field)
            for h, front, tail in self._hint_rows(slot, loose=True):
                if kit.exact_property(self._bare(front), want) or any(
                        kit.exact_property(self._bare(a), want) for a in self.tail_alts(tail)):
                    return True
                low = kit.low(h)
                if nw and all(re.search(r"\b" + re.escape(w), low) for w in nw):
                    return True
        except Exception as e:
            self.s.link.fault("hint_mapped_error", e)
        return False

    def name_twins(self, a, b):
        if kit.exact_property(self._bare(a), self._bare(b)):
            return False
        x, y = self._nw(a), self._nw(b)
        if not x or not y:
            return False

        def eq(p, q):
            return p == q or (len(p) >= 4 and len(q) >= 4 and (p.startswith(q) or q.startswith(p)))
        return any(eq(p, q) for p in (x[:-1] or x) for q in y) or any(eq(p, q) for p in x for q in (y[:-1] or y))

    def pair_for(self, slot, a, b):
        try:
            return not kit.exact_property(self._bare(a), self._bare(b)) and (
                self.name_twins(a, b) or self.field_alike(a, b, slot))
        except Exception as e:
            self.s.link.fault("twin_of_error", e)
            return False

    def pair_slug(self, slot, field):
        s = self.s
        try:
            story = kit.low(s.request) + " " + " ".join(kit.low(h) for h, f, t in self._hint_rows(slot))
            base = set(kit.property_terms(self._bare(field)))
            kinds = kit.kind_terms(self._pname(slot))
            for row in self._rows(slot):
                key = self._key(row)
                if not key or kit.is_junk_key(key) or key.lower() in s.spent_props:
                    continue
                if not self.pair_for(slot, field, key) or not self.open_for_planning(slot, key) or \
                        self.names_nest(key, field):
                    continue
                extra = [w for w in self._nw(key) if w not in base and not kit.kind_present(w, kinds)]
                if any(re.search(r"\b" + re.escape(w), story) for w in extra) or self.hint_bound(slot, key):
                    return key
        except Exception as e:
            s.link.fault("twin_key_error", e)
        return ""

    def values_question(self, slot, field, name):
        try:
            row = self.menu_row(slot, field)
            if "):" not in row:
                return ""
            taken = set()
            for r in self._rows(slot):
                if r != row and "):" in r:
                    taken |= {self._norm(x) for x in self._samples(r)}
            picks, seen = [], set()
            for v in self._samples(row):
                if len(picks) >= 3:
                    break
                n = self._norm(v)
                if v and len(v) <= 30 and re.search(r"[A-Za-z]", v) and not re.search(r"[,;/]", v) and \
                        n not in taken and n not in seen:
                    picks.append(v)
                    seen.add(n)
            if len(picks) < 2:
                return ""
            return "For the %s, would you like %s or %s?" % (name, ", ".join(picks[:-1]), picks[-1])
        except Exception as e:
            self.s.link.fault("values_question_error", e)
            return ""

    def _slot_of(self, label):
        """(slot, slot key) of a label for the once-per-product services; None when it cannot be used."""
        multi = self.s.multi()
        slot = self._prod(label) if multi else None
        key = -1 if slot is None else slot
        if (multi and slot is None) or key in self._used_slots:
            return None
        return slot, key

    def distinct_reask(self, label):
        try:
            got = self._slot_of(label)
            if got is None:
                return None
            slot, key = got
            pname = self._pname(slot) or "item"
            target = self._bare(label)
            q = self.values_question(slot, target, pname)
            if not q:
                twin = self.pair_slug(slot, target)
                if not twin:
                    return None
                target = twin
                q = self.values_question(slot, twin, pname) or self._PLAIN.replace("<<FIELD>>", twin).replace(
                    "<<PRODUCT>>", pname)
            self._used_slots.add(key)
            return ("%s (%s)" % (target, pname) if self.s.multi() else target), q
        except Exception as e:
            self.s.link.fault("distinct_reask_error", e)
            return None

    def _alt_key(self, slot, field):
        s = self.s
        want = self._bare(field)
        for h, front, tail in self._hint_rows(slot):
            head = self._bare(front)
            if not (self.field_alike(head, want, slot) or self.names_nest(head, want)):
                continue
            for alt in self.tail_alts(tail):
                ba = self._bare(alt)
                if kit.is_junk_key(alt) or kit.exact_property(ba, head) or kit.exact_property(ba, want) or \
                        self.names_nest(alt, want):
                    continue
                row = self.menu_row(slot, alt)
                key = self._key(row) if row else ""
                if not key or kit.exact_property(key, want) or key.lower() in s.spent_props:
                    continue
                if self.requested_by(slot, key) or not self.open_for_planning(slot, key):
                    continue
                return key
        return ""

    def alt_after_unanswered(self, label):
        try:
            got = self._slot_of(label)
            if got is None:
                return ""
            slot, key = got
            alt = self._alt_key(slot, label)
            if not alt:
                return ""
            self._used_slots.add(key)
            return "%s (%s)" % (alt, self._pname(slot)) if self.s.multi() else alt
        except Exception as e:
            self.s.link.fault("alt_unanswered_error", e)
            return ""

    def pauses_text(self, text, plain):
        try:
            frame = set(re.findall(r"[a-z]+", re.sub(r"<<\w+>>", " ", self._PLAIN).lower()))
            sent = {kit.stem(w) for w in re.findall(r"[a-z0-9]+", kit.low(text))}
            need = {kit.stem(w) for w in re.findall(r"[a-z0-9]+", kit.low(plain)) if w not in frame and not kit.is_link(w)}
            return bool(need) and need <= sent
        except Exception as e:
            self.s.link.fault("holds_plain_error", e)
            return False

    def open_for_planning(self, slot, prop):
        s = self.s
        try:
            h = self.hinted(slot, prop)
            return bool(s.mender.open_property(slot, prop, row=self.menu_row(slot, prop) if h else None, by_samples=h))
        except Exception as e:
            s.link.fault("open_planning_error", e)
            try:
                return bool(s.mender.open_property(slot, prop))
            except Exception as e2:
                s.link.fault("open_planning_error", e2)
                return True

    def alternate_fresh(self, slot):
        try:
            for h, front, tail in self._hint_rows(slot):
                head = self._bare(front)
                if head and self.requested_by(slot, head) and self._alt_key(slot, head):
                    return True
        except Exception as e:
            self.s.link.fault("alt_untried_error", e)
        return False

    def trim_menu(self, menu, reads):
        keep = []
        for r in str(menu or "").split("\n"):
            m = re.match(r"- (.+?) \(in (\d+) listing", r)
            if m and (kit.is_numbered_twin(m.group(1)) or (reads >= 4 and int(m.group(2)) <= 1)):
                continue
            keep.append(r)
        out = "\n".join(keep)
        return out if out.strip() else menu

    # ------------------------------------------------------------------ 3.13 settles
    def settles(self, req, samples=""):
        s = self.s
        try:
            prop, _, val = kit.strip_marks(str(req)).partition(":")
            val = val.strip()
            low = kit.low(val)
            if kit.is_pointer(low):  # one reference detector for page pointers too (03 7.5)
                return True
            if not kit.is_positive(val, prop):
                return False
            told = s.request + " " + " ".join(str(x) for x in s.ledger.exchanges)
            refs = [x for x in ((s.ledger.reading or {}).get("pointer_needs") or []) if isinstance(x, str)]
            refs += [str(x) for x in s.ledger.resolved]
            if kit.tale_hint("%s: %s" % (prop.strip(), val), told, refs):
                return False
            if samples and not re.search(r"\d", val):
                ws = [w for w in re.findall(r"[a-z]{3,}", low) if not kit.is_link(w)]
                sl = kit.low(samples)
                pool = [w for w in re.findall(r"[a-z]+", sl) if len(w) >= 4]

                def echo(w):
                    if re.search(r"\b" + re.escape(w[:-1] if len(w) > 4 else w), sl):
                        return True
                    return len(w) >= 4 and any(kit.within_edits(w, x, 1) for x in pool)
                if ws and not any(echo(w) for w in ws):
                    return False
            return True
        except Exception as e:
            s.link.fault("settles_error", e)
            return False

    # ------------------------------------------------------------------ 3.14 first reads
    def opening_reads(self):
        s = self.s
        try:
            if self._first_done or s.mender.batch_spec() is not None or not s.role("details"):
                return False
            self._first_done = True
            budget = s.steps_left() - self.due_margin() - 2 * max(0, self.question_room() - s.msgs_sent) - 8
            counts = s.ledger.counts() or [1]
            claimed, plans = set(), []
            for i, slot in enumerate(s.ledger.slots() or [None]):
                homes = self._homes(slot)
                reads = self.detailed_reads(slot)
                picks = []
                if homes and reads < 2 and budget > 0:
                    qty = counts[i] if i < len(counts) else 1
                    got = s.mender.row_menu(homes, 2 - reads, skip=set(claimed), units=qty) or []
                    picks = [x for x in (self._lid(y) for y in got) if x][:2 - reads]
                    claimed.update(picks)
                plans.append((reads, i, picks))
            plans.sort(key=lambda p: (p[0], p[1]))
            queue = [p[2][0] for p in plans if p[2]] + [p[2][1] for p in plans if len(p[2]) > 1]
            queue = queue[:budget] if budget > 0 else []
            if not queue:
                return False
            s.judge.queue_open(queue, "menu", recheck=True,
                               say="Opening up to two listings of each product first, to learn which fields their "
                                   "records carry before the first message to the shopper.")
            s.link.note("opening_reads", ids=queue)
            return True
        except Exception as e:
            s.link.fault("first_reads_error", e)
            return False

    # ------------------------------------------------------------------ 3.15-3.17 margin, phases, verification
    def due_margin(self):
        s = self.s
        f = Tune.Finish
        c = s.step_cap
        floor = max(f.ORDER_BY, min(f.ORDER_DUE, c // 4 + 1))
        held = max(floor, min(f.HELD_MAX, c // 3))
        try:
            if s.trouble():
                try:
                    held = max(held, f.TROUBLE_BASE + max(1, len(s.ledger.products())))
                except Exception:
                    held = max(held, f.TROUBLE_MIN)
        except Exception as e:
            s.link.fault("due_margin_error", e)
        return max(floor, min(held, c // 2)) + self.finish_reserve()

    def can_narrow(self):
        s = self.s
        return bool(s.role("order") and s.role("cart"))

    def _gates(self):
        s = self.s
        left, margin = s.steps_left(), self.due_margin()
        on = (left <= margin and not s.orders_tried) or (left <= Tune.Finish.ORDER_BY and s.refusals >= 1)
        return left, margin, on

    def phase(self):
        left, margin, on = self._gates()
        return "last" if left <= 1 else "closing" if left <= 3 else "reserve" if on else "open"

    def reserve_left(self):
        left, margin, on = self._gates()
        return self.can_narrow() and left > 3 and on

    def _read_gate(self, left, hit, margin):
        return (self.no_verified_fit() and left > 5) or (hit and left >= min(9, margin))

    def _search_gate(self, left, paged):
        return (self.widened_recently() or paged or (self.no_verified_fit() and left > 5)) and left > 4

    def _unlocks(self, left, margin):
        s = self.s
        det, view = s.role("details"), s.role("view")
        out = {s.role("cart_view"), view or det, s.judge.supply_tool()}
        if det and s.judge.swap_row() and not s.swap_read_done:
            out.add(det)
        if s.late_reads < self._LATE:
            if self._read_gate(left, bool(self.named_unopened()), margin):
                out.add(det)
            if self._search_gate(left, bool(s.named_pages)):
                out.add(s.role("search"))
        if left >= Tune.Finish.ORDER_BY and self.fallback_rows():
            out.add(det)
        out.discard(None)
        return out

    def offered_tools(self):
        s = self.s
        try:
            full = list(s.tools)
            o, c, d = s.role("order"), s.role("cart"), s.role("decline")
            ph = self.phase()
            if ph == "open" or not o or not c:
                return full
            left, margin, on = self._gates()
            names = {o, d} if ph == "last" else {o, d, c}
            if ph == "reserve":
                names |= self._unlocks(left, margin)
            m = s.role("message")
            if m and left >= 2 and (s.msg_cap is None or s.msgs_sent < s.msg_cap):
                names.add(m)
            return [t for t in full if self._name(t) in names] or full
        except Exception as e:
            s.link.fault("offered_tools_error", e)
            return list(s.tools)

    def offered_names(self):
        return {self._name(t) for t in self.offered_tools()}

    def no_verified_fit(self):
        s = self.s
        try:
            ver = "|" + str(s.ledger.version)
            live = [t[:-len(ver)] for t in s.passed if t.endswith(ver)]
            live = [lid for lid in live if lid not in s.refused_lines and lid not in s.dead]
            if not live:
                return True
            if not s.multi():
                return False
            shelves = {kit.low(s.judge.label_of(lid)) for lid in live} - {""}
            return bool(shelves) and any(h and not ({kit.low(x) for x in h} & shelves) for h in s.judge.home_labels())
        except Exception as e:
            s.link.fault("verified_fit_error", e)
            return False

    def named_unopened(self):
        s = self.s
        return [lid for lid in s.named_leads if lid in s.archive.known and lid not in s.archive.read
                and lid not in s.refused_lines]

    def widened_recently(self):
        s = self.s
        led = s.ledger
        return led.widened_at == led.version and s.link.steps - led.widened_step <= self.due_margin()

    # ------------------------------------------------------------------ 3.18-3.20 late gate, backups, field counts
    def batch_start(self):
        self._grants = 0
        self._late_ids = set()

    def _lookups(self):
        out = set()
        for t in (self.s.judge.query_tools() or []):
            out.add(t if isinstance(t, str) else self._name(t))
        return out

    def late_gate(self, name, args):
        s = self.s
        try:
            det, srch = s.role("details"), s.role("search")
            if name and name == det:
                kind = "read"
            elif name and name == srch:
                kind = "search"
            elif det and name and name in self._lookups():
                kind = "read"
            else:
                return ""
            if not self.reserve_left():
                return ""
            args = args if isinstance(args, dict) else {}
            targets = [v for v in args.values() if isinstance(v, str)]
            if kind == "read":
                ever = {t.rsplit("|", 1)[0] for t in s.passed}
                if any(t in s.basket.units or t in ever for t in targets):
                    return ""
            queued = self._grants
            spare = s.steps_left() - queued
            grant = False
            if s.late_reads + queued < self._LATE:
                if kind == "read":
                    named = set(self.named_unopened())
                    grant = self._read_gate(spare, any(t in named for t in targets), self.due_margin())
                else:
                    grant = self._search_gate(spare, self._sig(args) in s.named_pages)
            room = kind == "read" and spare - 1 >= 6 and self._backups < self._BACKUPS
            fresh = []
            if not grant and room:
                fresh = [b for b in self.fallback_rows() if b not in self._late_ids][:self._BACKUPS - self._backups]
                if any(t in fresh for t in targets):
                    self._backups += 1
                    grant = True
            if not grant and kind == "read" and spare > 2:
                swap = s.judge.swap_row()
                if swap and swap in targets:
                    s.swap_read_done = True
                    grant = True
            if grant:
                self._grants = queued + 1
                self._late_ids.update(targets)
                return ""
            text = (Tune.Marks.NOT_RUN + ": the errand is in checkout mode. Only three sorts of read remain: a line "
                    "in the cart; a listing an item check flagged; and, as long as no candidate has passed "
                    "verification, %d reads in total. Cart the best verified fit, check its availability and cost "
                    "one more time, and order." % self._LATE)
            if room and fresh:
                text += " These unopened cheaper stand-ins may also be read: %s." % ", ".join(fresh)
            return text
        except Exception as e:
            s.link.fault("late_gate_error", e)
            return ""

    def fallback_rows(self):
        s = self.s
        try:
            if s.ledger.priority() != "cheapest" or self._backups >= self._BACKUPS:
                return []
            if not (s.event_seen or s.dead or self.named_unopened()):
                return []
            cands = []
            for slot in s.ledger.slots():
                pool = s.mender.pool(slot) or []
                fits = [e["price"] for e in pool if e.get("verified") and e.get("fit") and kit.is_num(e.get("price"))]
                if not fits:
                    continue
                low = min(fits)
                for e in pool:
                    p = e.get("price")
                    if e.get("read") or e.get("gone") or e.get("over") or e.get("pointer") or e.get("ruled") or \
                            e.get("clash") or not kit.is_num(p) or p >= low:
                        continue
                    cands.append((p, str(e.get("id"))))
            cands.sort()
            return [lid for p, lid in cands[:6]]
        except Exception as e:
            s.link.fault("backup_rows_error", e)
            return []

    def field_counts_block(self):
        try:
            menu = self.s.archive.menu()[0]
            if not menu:
                return ""
            return ("FIELD NAME COUNTS:\n(The rows come from the opened listings. Spellings that differ only in "
                    "separators count as one name. Each count tells in how many opened listings the name occurs; the "
                    "most frequent name is the main field of its property.)\n" + menu)
        except Exception as e:
            self.s.link.fault("field_counts_error", e)
            return ""



# ------------------------------------------------------------ p06_judge.py
# p06_judge.py - Judge (registry 3.2, spec part04): item check, field evidence, rank check, audit, open runner,
# lookup tools, tool learning, stock, shelves, joint-label merge, sold-out notes.


def _p06_guard(empty=None):
    """Public-method guard (INTERFACES 6.5): log the fault and return the neutral value."""
    def wrap(fn):
        def run(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as e:
                self.s.link.fault(fn.__name__ + "_error", e)
                return empty() if callable(empty) else empty
        run.__name__ = fn.__name__
        return run
    return wrap


def _p06_view():
    return {"lines": [], "rules": [], "fails": []}


class Judge:
    # marker of the evidence line saying the category label alone names the wanted value (11 3.5 reads it)
    CAT_MARK = "the category label itself names this value"
    _APPROX = r"\b(?:about|around|approx\w*|roughly)\b"  # 04 5.4 approximation words (flag 3: kept minimal, here only)
    _JOIN = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"  # 04 5.4 label joiners (flag 5)
    _OPTIONAL = ("menu", "quota", "walk")

    def __init__(self, s):
        self.s = s
        self._fv = {}
        self._cen = (None, None)
        self._ix = {"n": 0, "label": {}, "stock": {}, "labels": {}, "mapped": set()}
        self._home = None
        self._rank_leads = []
        self._job = None
        self._puts, self._put_tool = [], None
        self._last_reads = []
        self._stats, self._tried, self._calls = {}, {}, {}
        self._row_keys, self._row_text = set(), {}
        self._shist = {}
        self._moved = False
        self._guessed = None
        self._joined = {}
        self._swap_rows, self._swap_page = [], None
        self._seq = 0

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _ids(v):
        """Entries of a model list that are text or numbers, as strings."""
        out = []
        for x in v if isinstance(v, list) else []:
            if isinstance(x, dict):
                x = x.get("id")
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif kit.is_num(x):
                out.append(str(x))
        return out

    @staticmethod
    def _uniq(seq):
        out = []
        for x in seq:
            if x not in out:
                out.append(x)
        return out

    @staticmethod
    def _txt(v):
        return "" if v is None or isinstance(v, bool) else str(v).strip()

    @staticmethod
    def _money(p):
        return ("%.2f" % p).rstrip("0").rstrip(".") if kit.is_num(p) else "?"

    @staticmethod
    def _sig(args):
        return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _cart(self):
        return [l for l, u in self.s.basket.units.items() if not (kit.is_num(u) and u <= 0)]

    def _version_marks(self):
        v = str(self.s.ledger.version)
        return sorted(x for x, _, w in (m.rpartition("|") for m in self.s.passed) if w == v and x)

    def _over(self, lid, qty=None):
        bud = self.s.ledger.budget
        p = self.s.archive.price(lid)
        q = qty if qty is not None else self.s.ledger.quantity()
        return kit.is_num(bud) and kit.is_num(p) and p * (q or 1) > bud + 1e-9

    def _shelf_ok(self, lid):
        s = self.s
        try:
            want = {kit.low(x) for x in (s.seeker.wanted_shelves() or ())}
        except Exception:
            want = set()
        sh = kit.low(s.hit_shelf.get(lid) or "")
        return not (want and sh and sh not in want)

    def _hits_ranked(self):
        s = self.s
        return sorted(self._uniq([str(h) for h in s.all_hits]), key=lambda h: -(s.hit_weight.get(h) or 0))

    def _fails(self, lid):
        return bool(self.evidence_view(lid)["fails"])

    def _request_secs(self):
        s = self.s
        return ["REQUEST (the shopper's words; later changes follow):\n" + s.request, s.ledger.drops_text() or ""]

    def _ref_tail(self, lid=None):
        try:
            b = self.s.tracer.checker_block(lid)
        except Exception as e:
            self.s.link.fault("checker_block_error", e)
            b = ""
        return ("\n" + b) if b else ""

    @staticmethod
    def _join(secs):
        return "\n\n".join(x for x in secs if x)

    # ------------------------------------------------------------------ store index (incremental over archive.obs)
    def _idx(self):
        ix, obs = self._ix, self.s.archive.obs
        while ix["n"] < len(obs):
            o = obs[ix["n"]]
            ix["n"] += 1
            kit.visit(o, lambda d: self._feed(ix, d))
        return ix

    @staticmethod
    def _feed(ix, d):
        ids = [v for k, v in d.items() if isinstance(v, str) and "id" in str(k).lower() and kit.looks_like_id(v)]
        if not ids:
            return
        lab = kit.shelf_field(d)
        if lab:
            for i in ids:
                ix["label"][i] = lab
            ix["labels"].setdefault(lab, None)
        rid = kit.row_id(d)
        if rid:
            st = kit.stock_of(d)
            if st is not None:
                ix["stock"][rid] = st
            if Judge._has_map(d):
                ix["mapped"].add(rid)

    @staticmethod
    def _has_map(d):
        return any(isinstance(v, dict) and sum(1 for x in v.values() if kit.is_plain(x)) >= 2 for v in d.values())

    # ================================================================== 3.1 item check
    @_p06_guard("")
    def item_check(self, lid, aim, extra=""):
        s = self.s
        if s.link.time_left() < Tune.Clock.CLOSING + Tune.Clock.PAD_ITEM:
            return ""
        lid = str(lid)
        base = s.pass_key(lid)
        if base in s.near_ok:
            return ""
        key = base
        rd = s.ledger.reading or {}
        if [r for r in (rd.get("pointer_needs") or []) if r] or any(r.get("status") != "resolved" for r in s.refs):
            key += "|%d" % len(s.archive.read)
        if extra:
            key += "|second"
        trips = self._trips(lid) or []
        if trips:
            key += "|own%d" % len(trips)
        if key in s.rulings:
            return s.rulings[key]
        try:
            chain = s.tracer.pointer_note(lid) or ""
        except Exception as e:
            s.link.fault("pointer_note_error", e)
            chain = ""
        if chain:
            return self._rule_out(lid, key, chain, [chain], "reference chain")
        recs = list(s.archive.about(lid, 4) or [])
        if not recs:
            return ""
        fv = self.evidence_view(lid)
        if fv["fails"]:
            top = fv["fails"][:3]
            return self._rule_out(lid, key, "; ".join(top), top, "field evidence")
        ans = s.link.consult("check-item", self._item_text(lid, aim, extra, recs, trips, fv), "checker")
        s.link.note("verify", id=lid, result=ans)
        if isinstance(ans, dict) and ans.get("passes") is False:
            f = ans.get("faults")
            probs = [str(x) for x in (f if isinstance(f, list) else [f] if f else [])[:5]]
            verdict = ("Item check on %s: rejected - %s. Keep it only if you are certain the item check is wrong; "
                       "otherwise take a listing proven on every requirement." % (lid, "; ".join(probs) or "no reason given"))
            s.rulings[key] = verdict
            return verdict
        s.rulings[key] = ""
        if isinstance(ans, dict):
            s.passed.add(base)
            p = s.archive.price(lid)
            if kit.is_num(p):
                s.pass_price[lid] = p
        return ""

    def _rule_out(self, lid, key, reason, probs, tag):
        s = self.s
        verdict = ("Item check on %s: rejected - %s. The code ruled it out from the records, so the item check is "
                   "final here: another listing is needed." % (lid, reason))
        s.rulings[key] = verdict
        s.link.note("verify", id=lid, result={"passes": False, "faults": list(probs), "evidence": tag})
        return verdict

    def _item_text(self, lid, aim, extra, recs, trips, fv):
        s = self.s
        secs = [PROMPTS["item"]] + self._request_secs() + ["REQUIREMENTS IN FORCE:\n" + s.ledger.req_text()]
        if s.multi():
            held = s.basket.held(lid) or 0
            secs.append("ITEM SCOPE:\nThe request covers several products. This listing's category tells which "
                        "requested product it would serve: judge it only against that product's own requirements and "
                        "quantity, ignoring what the other products need."
                        + (" The cart already holds %s unit(s) of this listing." % self._money(held) if held else ""))
        ref = self._ref_tail(lid).strip()
        if ref:
            secs.append("REFERENCES:\n" + ref)
        pt = self._pointed(lid)
        if pt:
            secs.append("LISTINGS THE REQUEST MAY POINT AT:\n" + pt)
        if extra:
            secs.append("SECOND LOOK:\n" + extra)
        if trips:
            rows = ['- %s | requirement: %s | quote: "%s"' % (t[1], t[0], str(t[2])[:240]) for t in trips if len(t) >= 3]
            secs.append("REDIRECT NOTES OF THIS LISTING:\n" + "\n".join(rows) + "\nHow to read them: when a quote hands "
                        "this property over to some other listing, the value counts as the one that other listing shows, so "
                        "this item's own field proves nothing and only the other listing's record can meet the "
                        "requirement. Ignore quotes that name another listing only as a variant sold elsewhere, or that "
                        "say this item is not the one referred to.")
        secs.append("CANDIDATE ID: %s\nNEXT STEP: %s\nRECORDS OF THE CANDIDATE, OLDEST FIRST:\n%s" % (lid, aim, "\n".join(recs)))
        lv = s.basket.last_view
        if lv:
            secs.append("CART NOW:\n" + kit.render(lv, 3000))
        secs.append(s.intake.field_counts_block() or "")
        if fv["lines"]:
            secs.append("FIELD EVIDENCE:\nCompiled by code from this store's records: for each required value, how the "
                        "listings of this category spell it in the field, then this item's own entry.\n" + "\n".join(fv["lines"]))
        ll = s.archive.labels_listing()
        if ll:
            secs.append("CATEGORY LABELS:\nThe item must sit under the label that matches the request's whole "
                        "phrase; a neighbouring label is not enough.\n" + ll)
        return self._join(secs)

    def _pointed(self, lid):
        """3.2.1: read listings whose newest record holds a cue word of an unmatched request reference."""
        s = self.s
        refs = [str(r) for r in ((s.ledger.reading or {}).get("pointer_needs") or []) if r]
        try:
            refs = [r for r in refs if not any((t.get("status") == "resolved" or t.get("chain")) and s.tracer.resolves_to(r, t)
                                               for t in s.refs)]
        except Exception:
            pass
        cues = {w for r in refs for w in re.findall(r"[a-z]{4,}", r.split(":", 1)[-1].lower())}
        if not cues:
            return ""
        out = []
        for x in sorted(s.archive.read - {lid}):
            rec = s.archive.about(x, 1)
            if rec and any(c in str(rec[-1]).lower() for c in cues):
                out.append(str(rec[-1]))
                if len(out) == 3:
                    break
        if not out:
            return ""
        return ("When the request takes a value from one of these listings, read it from that listing's field for the "
                "named property.\n" + "\n".join(out))

    # ================================================================== 3.3-3.9 field evidence
    @_p06_guard(_p06_view)
    def evidence_view(self, lid):
        s = self.s
        lid = str(lid)
        lif = list(s.ledger.in_force() or [])
        key = (lid, len(s.archive.obs), s.ledger.version, "\n".join(lif))
        got = self._fv.get(key)
        if got is None:
            try:
                got = self._facts(lid, lif)
            except Exception as e:
                s.link.fault("field_view_error", e)
                got = ([], [], [])
            if len(self._fv) > 600:
                self._fv.clear()
            self._fv[key] = got
        view = {"lines": list(got[0]), "rules": list(got[1]), "fails": list(got[2])}
        acc = s.near_words.get(s.pass_key(lid))
        if acc:
            try:
                view = self._mark_near(view, acc)
            except Exception as e:
                s.link.fault("near_mark_error", e)
                view = {"lines": list(got[0]), "rules": list(got[1]), "fails": list(got[2])}
        return view

    @staticmethod
    def _mark_near(view, acc):
        acc = [tuple(a) for a in acc if isinstance(a, (list, tuple)) and len(a) >= 3]
        def ok(name):
            return any(kit.property_match(name, a[0]) or kit.exact_property(name, a[0]) for a in acc)
        lines = []
        for ln in view["lines"]:
            m = re.match(r"- (.+?) = '", ln)
            if m:
                for a in acc:
                    if kit.property_match(m.group(1), a[0]) or kit.exact_property(m.group(1), a[0]):
                        ln += (" [meaning check: this listing's wording '%s' counts as the wanted '%s'; requirement met]"
                               % (str(a[2])[:80], str(a[1])[:60]))
                        break
            lines.append(ln)
        fails = [f for f in view["fails"] if not ok(f.split(":", 1)[0])]
        return {"lines": lines, "rules": view["rules"], "fails": fails}

    def _census(self, label):
        key = (label, len(self.s.archive.obs))
        if self._cen[0] == key:
            return self._cen[1]
        dig = self.s.archive.digest()
        peers = [r for r in dig.values() if r.get("fields") and (not label or r.get("shelf") == label)]
        cnt, ent = {}, {}
        for r in peers:
            for k, v in r["fields"].items():
                t = self._txt(v)
                if not t or kit.is_junk_key(k):
                    continue
                cnt[k] = cnt.get(k, 0) + 1
                ps = list(kit.list_parts(t))
                if len(ps) > 1:
                    ps.append(t.lower())
                ent.setdefault(k, []).extend((r.get("id"), p, list(kit.value_words(p))) for p in ps)
        order = sorted(cnt, key=lambda k: -cnt[k])
        val = (peers, cnt, ent, order)
        self._cen = (key, val)
        return val

    def _facts(self, lid, lif):
        s = self.s
        lines, rules, fails = [], [], []
        dig = s.archive.digest()
        row = dig.get(lid)
        if not row or not row.get("fields"):
            return lines, rules, fails
        multi = s.multi()
        label = row.get("shelf") or ""
        if multi and not label:
            return lines, rules, fails
        slot = None
        if multi:
            for i, hs in enumerate(self.home_labels()):
                if label in hs:
                    slot = i
                    break
        peers, cnt, ent, order = self._census(label)
        if len(peers) < 3:
            return lines, rules, fails
        cx = {"lid": lid, "row": row, "label": label, "slot": slot, "multi": multi, "peers": peers, "cnt": cnt,
              "ent": ent, "order": order, "dig": dig, "trips": self._trips(lid)}
        for line in lif[:12]:
            self._weigh(str(line), cx, lines, rules, fails)
        return lines, rules, fails

    def _weigh(self, line, cx, lines, rules, fails):
        """3.4: parse one requirement line and route it."""
        s = self.s
        lab, colon, val = line.partition(":")
        if not colon:
            return
        val = " ".join(kit.strip_marks(val).split())
        prop = kit.name_head(lab).strip()
        if not prop or not val or kit.is_pointer(val):
            return
        lp = s.mender.item_for(lab) if cx["multi"] else None
        foreign = cx["multi"] and lp is not None and lp != cx["slot"]
        if re.search(r"\d", val) and kit.has_bound(val):
            if not foreign:
                self._limit(prop, val, cx, lines, fails)
            return
        if re.fullmatch(r"\d+(?:\.\d+)?\s*[^\W\d_]+", val.lower()):
            return
        if foreign:
            return
        if kit.is_positive(val, lab):
            pol = "pos"
        elif kit.has_exclusion(val):
            pol = "exc"
        else:
            return
        alts = kit.alternatives(val)
        if not alts:
            return
        if pol == "pos" and cx["label"] and kit.writes(cx["label"], val):
            t = ("- %s = '%s' | %s ('%s'), so any listing filed there meets it even when neither a field nor the title "
                 "restates it -> MEETS" % (prop, val[:70], self.CAT_MARK, cx["label"]))
            rules.append(t)
            lines.append(t)
            return
        pick = self._choose(prop, val, alts, pol, lp, cx)
        if not pick:
            return
        chosen, titled, pol = pick
        self._judge_line(prop, val, alts, pol, chosen, titled, cx, lines, rules, fails)

    def _limit(self, prop, val, cx, lines, fails):
        """3.5 numeric limits: the code fails a listing lacking a field the category carries."""
        stems = {w for w in kit.property_terms(prop) if len(w) >= 3 and not kit.is_link(w)}
        home = [k for k in cx["order"] if cx["cnt"][k] >= 2 and kit.exact_property(k, prop)]
        if not stems or not home:
            return
        for k, v in cx["row"]["fields"].items():
            if self._txt(v) and (kit.exact_property(k, prop) or set(kit.property_terms(k)) & stems):
                return
        n = sum(1 for r in cx["peers"] if any(self._txt(r["fields"].get(k)) for k in home))
        lines.append("- %s = '%s' | field in this store: %s | category listings read that fill it: %d of %d | this "
                     "listing: no such field -> the limit FAILS (a figure taken from some other field cannot replace it)"
                     % (prop, val[:70], " / ".join("'%s'" % k for k in home[:3]), n, len(cx["peers"])))
        fails.append("%s: no %s field here although the category's listings carry one -> '%s' unproven" % (prop, home[0], val))

    def _choose(self, prop, val, alts, pol, lp, cx):
        """3.6: fields that speak for the property -> (names, titled, polarity) or None."""
        cnt, ent, order = cx["cnt"], cx["ent"], cx["order"]
        named = [k for k in order if kit.exact_property(k, prop)]
        if named:
            if pol == "exc":
                fv = kit.flat_value(val)
                if fv and any(kit.flat_value(p) == fv for k in named for _o, p, _w in ent.get(k, ())):
                    pol = "pos"
                    kit.add_literal(val)
            return named, True, pol
        if pol == "exc" or (cx["multi"] and lp is None):
            return None
        holders = [k for k in order if cnt[k] >= 2 and any(w and kit.alts_match(alts, w) for _o, _p, w in ent.get(k, ()))]
        if not holders:
            return None
        return [k for k in order if kit.exact_property(k, holders[0])] or [holders[0]], False, pol

    @staticmethod
    def _drop_neg(val):
        """The value minus its exclusion words (judged by the shared exclusion test, word by word and in pairs)."""
        w, out, i = val.lower().split(), [], 0
        while i < len(w):
            if i + 1 < len(w) and not kit.has_exclusion(w[i]) and kit.has_exclusion(w[i] + " " + w[i + 1]):
                i += 2
                continue
            if kit.has_exclusion(w[i]):
                i += 1
                continue
            out.append(w[i])
            i += 1
        return " ".join(out)

    def _judge_line(self, prop, val, alts, pol, chosen, titled, cx, lines, rules, fails):
        """3.7 category facts and rule line, 3.8 the listing's own verdict line (and the code's own fails)."""
        lid, row, peers, ent = cx["lid"], cx["row"], cx["peers"], cx["ent"]
        others = [(o, p, w) for k in chosen for (o, p, w) in ent.get(k, ()) if o != lid]
        writers = {o for o, p, w in others if w and kit.alts_match(alts, w)}
        alone = {kit.flat_value(p) for o, p, w in others}
        filled = sum(1 for r in peers if any(self._txt(r["fields"].get(k)) for k in chosen))
        samples = []
        for r in peers:
            for k in chosen:
                v = self._txt(r["fields"].get(k))[:40]
                if v and v not in samples and len(samples) < 6:
                    samples.append(v)
        naming = []
        if not writers and kit.content_words(val):
            kw = set(kit.kind_terms(val))
            for r in cx["dig"].values():
                if r.get("id") != lid and (not cx["label"] or r.get("shelf") == cx["label"]) and kw and kw <= set(r.get("words") or ()):
                    naming.append(str(r.get("id")))
                    if len(naming) == 3:
                        break
        smp = "; ".join("'%s'" % x for x in samples) or "none"
        lead = "- %s = '%s' | field in this store: %s" % (prop, val[:70], " / ".join("'%s'" % k for k in chosen[:3]))
        if not titled:
            lead += " [picked by its values; no field is called '%s']" % prop
        lead += " | category listings read that fill it: %d of %d | " % (filled, len(peers))
        rest = self._drop_neg(val) if pol == "exc" else ""
        ralts = kit.alternatives(rest) if rest else []
        if ralts:
            n = len({o for o, p, w in others if w and kit.alts_match(ralts, w)})
            lead += "other listings holding '%s' there: %d (holding it means failing)" % (rest, n)
        else:
            lead += "other listings writing the value there verbatim: %d" % len(writers)
        if pol == "exc":
            tail = ""
        elif writers:
            tail = ("only the exact words count here (inside a longer value or as one list item); synonyms and related "
                    "words do not")
            if not titled:
                tail += ("; as no field is named after the property, a title holding the value also counts, unless a "
                         "field of that listing sharing a word with the name contradicts it")
        elif naming:
            tail = ("values seen: %s; category titles/texts use the value (%s), so it is the store's own term: a synonym "
                    "or related value counts only after the record of one of those listings shows the field worded "
                    "differently" % (smp, ", ".join(naming)))
        elif filled < Tune.Evidence.SYNONYM_MIN:
            tail = ("values seen: %s; the value appears nowhere, but so few read listings fill the field that it is "
                    "unknown whether the store ever writes it literally, so an equivalent stays unproven for now" % smp)
        else:
            tail = "values seen: %s; no listing has the value, so an equivalent may be accepted" % smp
        rules.append(lead + (" | " + tail if tail else ""))
        # ---- 3.8 the listing's own verdict
        f = row["fields"]
        own = [(k, self._txt(f.get(k))) for k in chosen if self._txt(f.get(k))]
        own_txt = "; ".join("%s: '%s'" % (k, v[:90]) for k, v in own)
        pw = list(kit.property_terms(prop))
        fuller = []
        if pw:
            for k, v in f.items():
                t, kw2 = self._txt(v), list(kit.property_terms(k))
                if k not in chosen and t and not kit.is_junk_key(k) and len(kw2) == len(pw) + 1 and kw2[1:] == pw:
                    fuller.append((k, t))
        hit = next(((k, v) for k, v in fuller if kit.writes(v, val)), None)
        near = any(kit.writes(v, val) or kit.touches(val, v) for k, v in fuller)
        note = ("'%s' ('%s') carries the value: one leading word more, same property -> MEETS" % (hit[0], hit[1][:60])
                if hit else "")
        if self._redirected(cx["trips"], prop):
            lines.append(lead + " | this listing's own notes suggest another listing may define this property, so its "
                                "own field is not final")
            return
        if pol == "exc":
            lines.append(lead + " | this listing: %s | holding the excluded value fails" % (own_txt or "field absent"))
            return
        stems = {w for w in pw if len(w) >= 3 and not kit.is_link(w)}
        rivals = [k for k, v in f.items() if self._txt(v) and not kit.is_junk_key(k)
                  and set(kit.property_terms(k)) & stems and not kit.writes(self._txt(v), val)]
        by_title = not rivals and kit.writes(self._txt(row.get("title")), val)
        if not own:
            if hit:
                text = "this listing: field absent | " + note
            elif by_title and titled:
                text = ("this listing: field absent; no field of its own bears a word of the property's name, and its "
                        "title has the value -> MEETS (the title replaces the missing field)")
            elif by_title:
                text = "this listing: field absent; no field is named after the property, and its title has the value -> MEETS"
            else:
                cand = []
                for k, v in f.items():
                    t = self._txt(v)
                    if k in chosen or not t or kit.is_junk_key(k):
                        continue
                    m = kit.writes(t, val)
                    if m or set(kit.property_terms(k)) & stems:
                        cand.append((0 if m else 1, "%s: '%s'" % (k, t[:60])))
                cand = [c[1] for c in sorted(cand, key=lambda c: c[0])[:4]]
                text = ("this listing: field absent" + ("; its own fields: " + "; ".join(cand) if cand else "")
                        + "; only another name for this very property can count, never a field about some other property")
            lines.append(lead + " | " + text)
            return
        by_title = by_title and not titled
        clean = coupled = None
        for k, v in own:
            ps = list(kit.list_parts(v))
            if len(ps) > 1:
                ps.append(v.lower())
            for p in ps:
                if not kit.alts_match(alts, kit.value_words(p)):
                    continue
                cpl = None
                sides = kit.plus_sides(p) or []
                if sides:
                    good = [x for x in sides if kit.writes(x, val)]
                    bad = [x for x in sides if not kit.writes(x, val) and self._held_alone(x, alone, others)]
                    if good and bad:
                        cpl = (p, bad[0])
                if cpl:
                    coupled = coupled or cpl
                elif clean is None:
                    clean = p
        plain = titled and not near
        if clean is not None:
            text = "contains '%s' -> MEETS" % val
        elif hit:
            text = note
        elif by_title:
            text = "its title has the value and no field is named after the property -> MEETS"
        elif coupled:
            text = ("'%s' joins the value to another value that other listings carry alone in this field -> FAILS"
                    % coupled[0])
            if plain:
                fails.append("%s: %s is '%s', joining %s with %s (which the store also writes alone there) -> not %s"
                             % (prop, chosen[0], coupled[0], val, coupled[1], val))
        elif writers:
            text = ("the value is absent although other category listings write it in this field -> FAILS (once the "
                    "store writes the value literally, synonyms and related words stop counting)")
            if plain and not any(kit.touches(val, v) for k, v in own):
                fails.append("%s: %s is '%s'; elsewhere the store writes '%s' literally in that field"
                             % (prop, own[0][0], own[0][1][:60], val))
        elif naming:
            text = ("the value is absent and no read listing carries it in this field so far (%s), yet category "
                    "titles/texts use it (%s): the store has a house term, so this other value is unproven"
                    % (smp, ", ".join(naming)))
        elif filled < Tune.Evidence.SYNONYM_MIN:
            text = ("absent from this field on all read listings (%s), and only %d fill it: too few to tell whether the "
                    "store writes it literally, so an equivalent stays unproven for now" % (smp, filled))
        else:
            text = "absent from this field on all read listings (%s): an equivalent may count" % smp
        lines.append(lead + " | this listing: " + own_txt + " | " + text)

    @staticmethod
    def _held_alone(side, alone, others):
        if kit.flat_value(side) in alone:
            return True
        sw = set(kit.value_words(side))
        return any(w and set(w) <= sw for _o, _p, w in others)

    def _trips(self, lid):
        """The reference area's redirect triples of a listing; None when the finder failed."""
        try:
            return [t for t in (self.s.tracer.self_pointers(lid) or []) if isinstance(t, (list, tuple)) and len(t) >= 3]
        except Exception as e:
            self.s.link.fault("own_redirects_error", e)
            return None

    @staticmethod
    def _redirected(trips, prop):
        """3.11: some own redirect triple of the listing names the property; errors count as redirected."""
        if trips is None:
            return True
        try:
            return any(kit.property_match(str(t[0]).split(":", 1)[0], prop) or kit.property_match(str(t[1]), prop) for t in trips)
        except Exception:
            return True

    # ================================================================== 3.10 field hook
    @_p06_guard(dict)
    def values_of_field(self, label):
        s = self.s
        hl = self.home_labels() or []
        if s.multi():
            p = s.mender.item_for(label)
            scope = hl[p] if p is not None and 0 <= p < len(hl) else None
        else:
            own = self.home_shelf()
            scope = {own} if own else (hl[0] if hl else set())
        out = {}
        for r in s.archive.digest().values():
            if not r.get("fields") or (scope is not None and r.get("shelf") not in scope):
                continue
            for k, v in r["fields"].items():
                t = self._txt(v)
                if t and not kit.is_junk_key(k):
                    out.setdefault(k, []).append(t)
        return out

    # ================================================================== 3.25 constants handed on
    @_p06_guard(False)
    def is_loose(self, value):
        t = kit.low(kit.strip_marks(str(value or "")))
        if not t.strip():
            return False
        if list(kit.words(str(kit.strip_links(t)))) != list(kit.words(t)) or kit.has_bound(t):
            return True
        return bool(re.search(r"\d\s*-\s*\d|\d\s+to\s+\d|[+<>]", t) or re.search(self._APPROX, t))

    def close_advice(self):
        return ("Order this cart as it stands; decline only if a record you opened shows a cart line failing today's "
                "requirements.")

    # ================================================================== 3.12 rank check
    @_p06_guard("")
    def order_audit(self, pick):
        s = self.s
        L = s.ledger
        pick = str(pick)
        pr = L.priority()
        if pr not in ("cheapest", "best"):
            return ""
        mark = "%s|%s|%s" % (pick, L.version, s.refusals)
        if mark in s.rank_marks or pick in (s.warden.best_of_pair() or {}):
            return ""
        if s.link.time_left() < Tune.Clock.CLOSING + Tune.Clock.PAD_RANK or s.steps_left() < 4:
            return ""
        s.rank_marks.add(mark)
        cheap = pr == "cheapest"
        meas = L.pick_measure()
        word = "cheapest (lowest price)" if cheap else ("best (%s)" % meas if meas else "best")
        fv = self.evidence_view(pick)
        cards = s.archive.cards(20000, pick if cheap else "", s.tracer.foreign_ids() or (), (pick,))
        secs = [PROMPTS["rank"].replace("<<PRIORITY>>", word)] + self._request_secs()
        secs.append("REQUIREMENTS IN FORCE:\n" + L.req_text() + self._ref_tail(pick))
        if s.multi():
            secs.append("ITEM SCOPE:\nA rival has to be the same product (category) as the pick; a listing of another "
                        "requested product never counts as better.")
        secs.append("PICK ID: " + pick)
        secs.append("LISTING CARDS:\nDETAILS card = read from a full record. ROW card = only a search-row sighting; its "
                    "fields are unknown. A ROW in the requested category may fit: name it when it would beat the pick and "
                    "nothing in its title, price or stock rules it out, whatever object its title names."
                    + (" Under cheapest, the cards cover only listings priced below the pick." if cheap else "")
                    + "\n" + "\n".join(cards))
        secs.append(s.intake.field_counts_block() or "")
        if fv["rules"]:
            secs.append("FIELD EVIDENCE:\nA code-made, per-value summary of how the store spells each wanted value in "
                        "the pick's category.\n" + "\n".join(fv["rules"]))
        ans = s.link.consult("check-rank", self._join(secs), "ranker")
        s.link.note("rank", pick=pick, result=ans)
        a = ans if isinstance(ans, dict) else {}
        got = []
        for x in self._ids(a.get("beats")):
            if x == pick or x in got or x in s.dead or x in s.refused_lines or self._fails(x):
                continue
            got.append(x)
            if len(got) == 4:
                break
        self._rank_leads = got
        self.add_leads(got)
        if not got:
            return ""
        why = self._txt(a.get("why"))
        return ("Rank check: %s may also fit and beat %s on the priority%s. Check each one's details, stock and price "
                "before ordering; if all of them fail, proceed with the current pick."
                % (", ".join(got), pick, " (%s)" % why if why else ""))

    # ================================================================== 3.13 rank verify, cheaper verified fit
    @_p06_guard("")
    def order_confirm(self, pick):
        s = self.s
        L = s.ledger
        if L.priority() != "cheapest":
            return ""
        done = 0
        for x in list(self._rank_leads):
            if done >= 2 or s.link.time_left() < Tune.Clock.CLOSING + Tune.Clock.PAD_RANK:
                break
            if (x not in s.archive.read or s.passed_now(x) or x in s.refused_lines or x in s.dead
                    or self.supply_state(x) is False or self._fails(x)):
                continue
            p = s.archive.price(x)
            if not kit.is_num(p):
                continue
            slot = s.walker.product_slot(x)
            cn = L.counts() or []
            q = cn[slot] if slot is not None and 0 <= slot < len(cn) else L.quantity()
            if kit.is_num(L.budget) and p * (q or 1) > L.budget + 1e-9:
                continue
            done += 1
            self.item_check(x, Tune.Aim.ORDER)
        return self.lower_price_fit(pick)

    @_p06_guard("")
    def lower_price_fit(self, lid):
        s = self.s
        lid = str(lid)
        if s.ledger.priority() != "cheapest" or lid in (s.warden.best_of_pair() or {}):
            return ""
        pp = s.archive.price(lid)
        if not kit.is_num(pp):
            return ""
        multi = s.multi()
        lab = self.label_of(lid) if multi else ""
        best = None
        for x in self._version_marks():
            if x == lid or x in s.refused_lines or x in s.dead:
                continue
            if multi and (not lab or self.label_of(x) != lab):
                continue
            p = s.archive.price(x)
            if not kit.is_num(p) or p >= pp - 1e-9 or (best and p >= best[0]):
                continue
            if self._fails(x):
                continue
            best = (p, x)
        return best[1] if best else ""

    # ================================================================== 3.14 pre-decline audit
    @_p06_guard("")
    def audit(self, early=False):
        s = self.s
        L = s.ledger
        nread = len(s.archive.read)
        again = s.audit_mark is not None and nread > s.audit_mark and s.audits < 2
        if not ((again or not s.audit_done) and s.link.time_left() >= Tune.Clock.CLOSING + Tune.Clock.PAD_RANK):
            return ""
        s.audit_done = True
        if not early:
            s.audits += 1
        new = sorted(set(s.archive.read) - set(s.audit_seen)) if s.audit_seen is not None else []
        cards = s.archive.cards(16000, "", s.tracer.foreign_ids() or (), tuple(new))
        s.audit_mark = nread
        if not cards:
            return ""
        s.audit_seen = set(s.archive.read)
        secs = [PROMPTS["audit"]] + self._request_secs()
        secs.append("REQUIREMENTS IN FORCE:\n" + L.req_text() + self._ref_tail(None))
        secs.append("LISTING CARDS:\nDETAILS card = read from a full record. ROW card = seen only as a search row; its "
                    "fields are unknown.\n" + "\n".join(cards))
        if new:
            secs.append("OPENED SINCE THE LAST AUDIT (examine first):\n" + ", ".join(new) + "\nExamine these before "
                        "all others and test every field of theirs by the value rules.")
        secs.append(s.intake.field_counts_block() or "")
        ans = s.link.consult("audit-decline", self._join(secs), "auditor")
        s.link.note("audit", early=bool(early), result=ans)
        a = ans if isinstance(ans, dict) else {}
        fits = [x for x in self._ids(a.get("maybe_fit"))[:5] if not self._fails(x)]
        s.audit_fits = fits
        known, read = s.archive.known, s.archive.read
        model = [x for x in self._ids(a.get("open_first")) if x in known and x not in read]
        hits = [h for h in self._hits_ranked() if h not in read and self._shelf_ok(h) and not self._over(h)]
        squeeze = []
        if s.multi():
            try:
                squeeze = [x for x in self._ids(list(s.mender.squeeze_rows() or [])) if x not in read]
            except Exception:
                squeeze = []
        unread = self._uniq(squeeze + hits[:4] + [x for x in model[:6] if x not in hits])[:6]
        pages, combo = [], []
        if s.steps_left() > 5:
            pages = [p for p in (s.seeker.best_pages(2) or []) if isinstance(p, dict)][:2]
            if pages:
                unread = unread[:4]
            try:
                combo = [c for c in (s.walker.combined_searches() or []) if isinstance(c, dict)]
            except Exception as e:
                s.link.fault("combined_searches_error", e)
                combo = []
        for q in pages + combo:
            s.named_pages.add(self._sig(q))
        self.add_leads(fits + unread)
        if not (fits or unread or pages or combo):
            return ""
        tool = s.role("search") or "search"
        out = []
        if fits:
            why = self._txt(a.get("why"))
            out.append("Second opinion: %s look like they may satisfy every requirement%s."
                       % (", ".join(fits), " (%s)" % why if why else ""))
        if pages:
            out.append("Value searches report more matches than have been read, so a fit may sit on a later page: %s. "
                       "Each page costs one step and shows many rows: read those pages first, then open every affordable "
                       "row of the requested kind." % "; ".join("%s %s" % (tool, kit.render(p)) for p in pages))
        if combo:
            out.append("Combined searches not run yet (every word is required, so only listings carrying all of them "
                       "come back): %s. Run them with the search tool, then open every affordable row of the requested "
                       "kind." % "; ".join("%s %s" % (tool, kit.render(c)) for c in combo))
        if unread:
            out.append("Rows only sighted in searches and never opened (fields unchecked, requested kind): %s. Begin "
                       "with their details; several fit in one reply." % ", ".join(unread))
        out.append("Decide every lead from its own details, stock and price and buy whichever survives; a decline is "
                   "correct only when every lead really fails.")
        return " ".join(out)

    # ================================================================== 3.15 leads
    @_p06_guard(list)
    def unread_leads(self):
        s = self.s
        out = []
        for x in self._hits_ranked() + [str(i) for i in s.named_leads]:
            if (x in out or x in s.archive.read or x in s.refused_lines or x not in s.archive.known or self._over(x)
                    or not self._shelf_ok(x)):
                continue
            out.append(x)
        return out

    @_p06_guard(None)
    def add_leads(self, ids):
        for i in ids or []:
            i = str(i)
            if i and i not in self.s.named_leads:
                self.s.named_leads.append(i)

    # ================================================================== 3.16 lookup calls
    @staticmethod
    def _is_text(v):
        t = v.get("type") if isinstance(v, dict) else None
        return (t == "string" or (isinstance(t, list) and "string" in t)) and not v.get("enum")

    def _id_arg(self, tool, strict=True):
        sch = self.s.schema(tool) or {}
        props = sch.get("properties") or {}
        req = [r for r in (sch.get("required") or []) if r in props]
        free = [k for k, v in props.items() if self._is_text(v)]
        arg = next((k for k in free if k in req), None) or (free[0] if len(free) == 1 else None)
        if arg and strict and [r for r in req if r != arg]:
            return None
        return arg

    def _tool_names(self):
        out = []
        for t in self.s.tools or []:
            fn = t.get("function") if isinstance(t, dict) else None
            n = fn.get("name") if isinstance(fn, dict) else None
            if n and n not in out:
                out.append(n)
        return out

    @_p06_guard(list)
    def query_tools(self):
        s = self.s
        busy = {s.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")}
        return [n for n in self._tool_names() if n not in busy and self._id_arg(n)]

    def _reuse(self, tool, lid):
        known = self.s.archive.known
        for a in reversed(self._calls.get(tool, [])):
            if any(isinstance(v, str) and v in known for v in a.values()):
                return {k: (lid if isinstance(v, str) and v in known else v) for k, v in a.items()}
        return None

    @_p06_guard(None)
    def query_call(self, lid, tool):
        lid = str(lid)
        if tool and tool == self.s.role("details"):
            c = self.detail_request(lid)
            return c[1] if c else None
        if not tool:
            return None
        a = self._reuse(tool, lid)
        if a is None:
            arg = self._id_arg(tool)
            a = {arg: lid} if arg else None
        return a

    @_p06_guard(None)
    def detail_request(self, lid):
        s = self.s
        d = s.role("details")
        if not d or d == s.role("search"):
            return None
        a = self._reuse(d, str(lid))
        if a is None:
            arg = self._id_arg(d, strict=False)
            if not arg:
                return None
            a = {arg: str(lid)}
        return d, a

    @_p06_guard(list)
    def question_ids(self, name, args):
        out = []
        def walk(v):
            if isinstance(v, str):
                if kit.looks_like_id(v) and v not in out:
                    out.append(v)
            elif isinstance(v, list):
                for x in v:
                    walk(x)
        for v in (args.values() if isinstance(args, dict) else []):
            walk(v)
        return out

    # ================================================================== 3.17 scripted reads and the open runner
    @_p06_guard(list)
    def scripted_read(self, ids, say, batch=None, tool=None):
        s = self.s
        ids = self._uniq([str(i) for i in ids or []])
        calls = []
        if batch:
            bt, arg, size = batch[0], batch[1], max(1, int(batch[2] or 1))
            for k in range(0, len(ids), size):
                calls.append((bt, {arg: ids[k:k + size]}))
        else:
            for i in ids:
                if tool:
                    a = self.query_call(i, tool)
                    c = (tool, a) if a else None
                else:
                    c = self.detail_request(i)
                if c:
                    calls.append(c)
        if not calls:
            return []
        self._seq += 1
        st = s.link.steps
        props = [s.engine.call(n, a, "read%d_%d_%d" % (self._seq, k, st)) for k, (n, a) in enumerate(calls)]
        n0 = len(s.archive.obs)
        s.covering += 1
        try:
            reps = s.engine.dispatch(say, props)
        finally:
            s.covering -= 1
            self._last_reads = list(s.archive.obs[n0:])
        return list(reps or [])

    @_p06_guard(None)
    def queue_open(self, ids, purpose, recheck=False, floor=None, tool=None, say=""):
        self._job = {"ids": [str(i) for i in ids or []], "purpose": purpose or "", "recheck": bool(recheck),
                     "floor": floor, "tool": tool, "say": say or ""}

    @_p06_guard(None)
    def queue_put_reads(self, ids, tool):
        q = list(self._puts or [])
        self._puts = (q + [str(i) for i in ids or [] if str(i) not in q])[:4]
        self._put_tool = tool

    @_p06_guard(False)
    def open_pending(self):
        return bool((self._job and self._job["ids"]) or self._puts)

    def _batch(self):
        try:
            b = self.s.mender.batch_spec()
            return b if b and len(b) >= 3 and b[0] else None
        except Exception:
            return None

    def _batching(self, spec, ids, purpose, tool):
        return bool(spec and len(ids) >= 2 and (purpose != "cart" or self.s.batch_proven)
                    and (tool is None or self.tool_stats(spec[0])["stock"] > 0))

    def _richer(self, ids):
        """Batch proof: the result is richer than search rows (nested maps, unseen keys, or no rows seen yet)."""
        try:
            want, keys, deep = set(ids), set(), [False]
            def look(d):
                if any(isinstance(v, str) and v in want for v in d.values()):
                    keys.update(d.keys())
                    if any(isinstance(v, dict) and len(v) >= 2 for v in d.values()):
                        deep[0] = True
            for o in self._last_reads:
                kit.visit(o, look)
            return deep[0] or not self._row_keys or bool(keys - self._row_keys)
        except Exception:
            return True

    @_p06_guard(False)
    def open_queued(self):
        s = self.s
        job, self._job = self._job, None
        over = s.link.done or s.link.broke
        put = False
        if not job or not job["ids"]:
            if not self._puts or over:
                return False
            job = {"ids": self._puts, "purpose": "cart", "recheck": True, "floor": None, "tool": self._put_tool,
                   "say": "Re-reading the newly carted item for stock: listings can sell out moments after carting."}
            self._puts, self._put_tool, put = [], None, True
        if over:
            return False
        ids, pur, rc, floor, tool = self._uniq(job["ids"]), job["purpose"], job["recheck"], job["floor"], job["tool"]
        say = job["say"] or ("Re-reading the cart lines before ordering." if rc else
                             "Looking at the flagged listings before any decline.")
        spec = self._batch()
        if pur in self._OPTIONAL and s.intake.can_narrow() and not s.link.mute:
            try:
                res = s.intake.due_margin() if not s.orders_tried else Tune.Finish.ORDER_BY
                room = s.steps_left() - res - 1
                bat = self._batching(spec, ids, pur, tool)
                if room < (1 if bat else len(ids)) and (self._cart() or not s.intake.no_verified_fit()):
                    ids = [] if bat else ids[:max(0, room)]
                    s.link.note("open_trim", purpose=pur, room=room, kept=list(ids))
                    if not ids:
                        return False
            except Exception as e:
                s.link.fault("open_trim_error", e)
        if pur == "cart":
            stale, cart = set(self.dated_lines()), set(self._cart())
            ids = [i for i in ids if i in stale or (not put and i not in cart)]
            if not ids:
                return False
        use = self._batching(spec, ids, pur, tool)
        reps = self.scripted_read(ids, say, spec if use else None, tool)
        if use and reps and not s.batch_proven:
            try:
                if s.mender.batch_records_back(reps, ids) and self._richer(ids):
                    s.batch_proven = True
                elif not s.mender.batch_ids_echoed(reps, ids) and len(ids) > 2:
                    s.mender.shrink_batch(ids)
                else:
                    s.mender.drop_batch()
                    res = floor if floor is not None else (s.intake.due_margin() if pur in ("menu", "quota") else 3)
                    n = min({"menu": 0, "quota": 2, "walk": 2}.get(pur, 3), len(ids), s.steps_left() - res - 1)
                    if n > 0 and not (s.link.done or s.link.broke) and (tool or s.role("details")):
                        s.engine.send_replies(reps, False)
                        reps = []
                        reps = self.scripted_read(ids[:n], say, None, tool)
                        if not reps:
                            return True
            except Exception as e:
                s.link.fault("batch_proof_error", e)
        if not reps:
            return False
        if not rc and s.link.time_left() > Tune.Clock.CLOSING + Tune.Clock.PAD_RANK:
            note = self.audit()
            if note:
                cid, text = reps[-1]
                reps[-1] = (cid, str(text) + "\n[check after opening] " + note)
        s.engine.send_replies(reps)
        return True

    # ================================================================== 3.18 stock state
    @_p06_guard(None)
    def supply_state(self, lid):
        return self._idx()["stock"].get(str(lid))

    # ================================================================== 3.19 shelves
    @_p06_guard("")
    def label_of(self, lid):
        lid = str(lid)
        return self._idx()["label"].get(lid) or kit.low(self.s.archive.shelf_by_id.get(lid) or "").strip()

    @_p06_guard(None)
    def chosen_line(self):
        c = self._cart()
        return c[-1] if c else None

    @_p06_guard(list)
    def home_labels(self):
        s = self.s
        n = len(s.archive.obs)
        if self._home and self._home[0] == n:
            return self._home[1]
        P = s.ledger.products() or []
        try:
            labs = list(self._idx()["labels"])
            names = [p.get("name") or "" for p in P] if P else [s.ledger.product_name() or ""]
            toks = [list(kit.kind_bits(x)) for x in names]
            ltok = {l: list(kit.kind_bits(l)) for l in labs}
            out = []
            for i, t in enumerate(toks):
                shared = [w for j, o in enumerate(toks) if j != i for w in o]
                sc = {l: kit.kind_overlap(t, ltok[l], shared) for l in labs}
                top = max(sc.values(), default=0)
                lead = [l for l in labs if top > 0 and sc[l] == top]
                if len(toks) > 1:
                    lead = [l for l in lead if kit.kinds_inside(ltok[l], t) and kit.kinds_inside(t, ltok[l])] or lead
                out.append(set(lead))
        except Exception as e:
            s.link.fault("home_labels_error", e)
            out = [set() for _ in (P or [0])]
        self._home = (n, out)
        return out

    @_p06_guard("")
    def home_shelf(self):
        s = self.s
        if s.multi():
            return ""
        hl = self.home_labels()
        home = hl[0] if hl else set()
        reads = [kit.low(x).strip() for x in (s.archive.shelves_opened() or []) if x]
        if not any(r not in home for r in reads):
            return ""
        for r in reads:
            if r in home:
                return r
        return sorted(home)[0] if home else ""

    # ================================================================== 3.20 joint-label merge (one merge per call)
    def _halves(self, label):
        hs = [h.strip() for h in re.split(self._JOIN, label)]
        return hs if len(hs) == 2 and all(hs) else None

    @_p06_guard("")
    def fuse_linked(self, obs):
        def grab(d):
            if not any(isinstance(v, str) and "id" in str(k).lower() and kit.looks_like_id(v) for k, v in d.items()):
                return
            lab = kit.shelf_field(d)
            if lab and self._halves(lab):
                raw = next((v.strip() for k, v in d.items() if isinstance(v, str) and "id" not in str(k).lower()
                            and kit.low(v).strip() == lab), lab)
                self._joined.setdefault(lab, raw)
        kit.visit(obs, grab)
        if not self._merge_ok():
            return ""
        hit = self._find_pair()
        return self._merge(*hit) if hit else ""

    def _merge_ok(self):
        s = self.s
        P = s.ledger.products() or []
        return bool(len(P) >= 2 and self._joined and not s.cart_edits and not self._cart() and not s.ledger.revealed
                    and not s.ledger.freed and not self._asked_about(P))

    def _asked_about(self, P):
        for ln in self.s.question_log:
            m = re.match(r'\s*-\s*(.*?):\s*asked\s+"(.*)"\s*->', str(ln))
            if not m:
                continue
            about, q = m.group(1), m.group(2).lower()
            for p in P:
                toks = list(kit.kind_bits(p.get("name") or ""))
                if any(any(kit.kind_present(t, toks) for t in kit.kind_bits(tag)) for tag in kit.tags_of(about)):
                    return True
                pat = kit.name_regex(p.get("name") or "")
                if pat and re.search(r"\b(?:" + pat + ")", q):
                    return True
        return False

    def _req_low(self):
        t = self.s.request.lower().replace("’", "'").replace("‘", "'")
        return " ".join(t.split())

    def _find_pair(self):
        s = self.s
        P = s.ledger.products() or []
        req = self._req_low()
        for low_lab in sorted(self._joined):
            h = self._halves(low_lab)
            if not h:
                continue
            H1, H2 = list(kit.kind_bits(h[0])), list(kit.kind_bits(h[1]))
            if not H1 or not H2:
                continue
            for i in range(len(P)):
                for j in range(i + 1, len(P)):
                    A, B = list(kit.kind_bits(P[i].get("name") or "")), list(kit.kind_bits(P[j].get("name") or ""))
                    if not self._covers(A, B, H1, H2) or P[i].get("qty") != P[j].get("qty"):
                        continue
                    if self._clash(P[i].get("musts") or [], P[j].get("musts") or []):
                        continue
                    ph = self._phrase(req, h, P[i].get("name") or "", P[j].get("name") or "", A, B)
                    if ph:
                        return low_lab, i, j, ph[0], ph[1]
        return None

    @staticmethod
    def _covers(A, B, H1, H2):
        if not A or not B or (kit.kinds_inside(A, B) and kit.kinds_inside(B, A)):
            return False
        if not kit.kinds_inside(A + B, H1 + H2):
            return False
        uA = [t for t in A if not kit.kind_present(t, B)]
        uB = [t for t in B if not kit.kind_present(t, A)]
        for X, Y in ((H1, H2), (H2, H1)):
            if (all(kit.kind_present(t, X) and not kit.kind_present(t, Y) for t in uA)
                    and all(kit.kind_present(t, Y) and not kit.kind_present(t, X) for t in uB)
                    and any(kit.kind_present(t, X) for t in A) and any(kit.kind_present(t, Y) for t in B)):
                return True
        return False

    @staticmethod
    def _clash(ma, mb):
        for x in ma:
            px, cx, vx = str(x).partition(":")
            for y in mb:
                py, cy, vy = str(y).partition(":")
                if cx and cy and kit.property_match(kit.name_head(px), kit.name_head(py)) and kit.flat_value(vx) != kit.flat_value(vy):
                    return True
        return False

    @staticmethod
    def _gap_ok(gap):
        return (len(gap) <= 40 and not re.search(r"[.!?;:\d]", gap) and len(re.findall(r"[a-z']+", gap)) <= 3
                and bool(re.search(r"\band\b|&|\bor\b|,|/|\+", gap)) and not re.search(r"\bx\b", gap))

    def _phrase(self, req, h, na, nb, A, B):
        """3.20 'written as one phrase' -> (matched text, label wording) or None."""
        p1, p2 = kit.name_regex(h[0]), kit.name_regex(h[1])
        if p1 and p2:
            m = re.search(r"\b(?:%s)\s*(?:and|&|or|/|\+)\s*(?:%s)" % (p1, p2), req)
            if m:
                return m.group(0).strip(), True
        if not A or not B or kit.kind_match(A[-1], B[-1]):
            return None
        pa, pb = kit.name_regex(na), kit.name_regex(nb)
        if not pa or not pb:
            return None
        for x, y in ((pa, pb), (pb, pa)):
            for ma in re.finditer(r"\b(?:%s)" % x, req):
                for mb in re.finditer(r"\b(?:%s)" % y, req):
                    if mb.start() < ma.end() or not self._gap_ok(req[ma.end():mb.start()]):
                        continue
                    rest = req[:ma.start()] + " | " + req[mb.end():]
                    if re.search(r"\b(?:%s)" % pa, rest) and re.search(r"\b(?:%s)" % pb, rest):
                        return None
                    return req[ma.start():mb.end()].strip(), False
        return None

    def _merge(self, low_lab, i, j, phrase, by_label):
        """3.20.1: products i and j become one product named by the phrase."""
        s = self.s
        L = s.ledger
        rd = L.reading if isinstance(L.reading, dict) else None
        P = L.products() or []
        parts = rd.get("parts") if rd else None
        rows = [r for r in parts if isinstance(r, dict) and self._txt(r.get("kind"))] if isinstance(parts, list) else []
        if (len(rows) != len(P) or not (0 <= i < j < len(P))
                or any(self._txt(r.get("kind")).lower() != str(p.get("name") or "").strip().lower() for r, p in zip(rows, P))):
            return ""
        left = len(P) - 1
        A, B = list(kit.kind_bits(P[i]["name"])), list(kit.kind_bits(P[j]["name"]))
        ltoks = list(kit.kind_bits(low_lab))
        rest_t = [list(kit.kind_bits(P[k]["name"])) for k in range(len(P)) if k not in (i, j)]
        def ours(tag):
            t = list(kit.kind_bits(tag))
            return bool(t) and kit.kinds_inside(t, ltoks) and any(kit.kind_present(x, A + B) for x in t) and not any(
                kit.kinds_inside(t, o) and kit.kinds_inside(o, t) for o in rest_t)
        def retag(text):
            if not isinstance(text, str):
                return text
            head, colon, tail = text.partition(":")
            head = re.sub(r"\s*\(([^()]*)\)", lambda m: ((" (%s)" % phrase) if left >= 2 else "") if ours(m.group(1))
                          else m.group(0), head)
            return head + colon + tail
        def fix(lst, dedupe=True):
            if not isinstance(lst, list):
                return
            new = [retag(x) for x in lst]
            lst[:] = self._uniq(new) if dedupe else new
        def untag(t):
            h, c, tl = str(t).partition(":")
            return re.sub(r"\s*\([^()]*\)", "", h).strip() + c + tl
        mu, low_seen = [], set()
        for x in list(rows[i].get("musts") or []) + list(rows[j].get("musts") or []):
            u = untag(x)
            if u.lower() not in low_seen:
                low_seen.add(u.lower())
                mu.append(u)
        merged = dict(rows[i], kind=phrase, units=P[i].get("qty"), musts=mu)
        new_rows = []
        for k, r in enumerate(rows):
            if k == j:
                continue
            r = merged if k == i else dict(r)
            w = r.get("with")
            if isinstance(w, int) and not isinstance(w, bool):
                w = i + 1 if w == j + 1 else (w - 1 if w > j + 1 else w)
                r["with"] = None if w == len(new_rows) + 1 else w
            new_rows.append(r)
        for lst in (L.written, L.revealed, L.dropped, L.freed, L.rebuilt):
            fix(lst)
        fix(L.resolved, False)
        for k in ("musts", "story_hints", "pointer_needs", "wishes"):
            fix(rd.get(k))
        for t in rd.get("pointer_targets") or []:
            if isinstance(t, dict) and t.get("product") and ours(str(t["product"])):
                t["product"] = phrase
        for r in s.refs:
            for k in ("text", "head", "settled"):
                if isinstance(r.get(k), str) and r[k]:
                    r[k] = retag(r[k])
            if isinstance(r.get("tag"), str) and r["tag"] and ours(r["tag"]):
                r["tag"] = phrase if left >= 2 else ""
        fix(s.ask_queue)
        fix(s.open_pairs)
        if left >= 2:
            rd["parts"] = new_rows
            rd["item_kind"] = " and ".join(self._txt(r.get("kind")) for r in new_rows)
        else:
            rd["parts"] = []
            rd["item_kind"] = phrase
            rd["units"] = P[i].get("qty") or 1
        L.version += 1
        self._home = None
        self._fv.clear()
        # the field-menu, quota-read and walk memos of Intake/Mender/Walker are private to them (keyed per version/obs)
        n = left if left >= 2 else 0
        s.link.scale = ((1 + Tune.Spend.PER_PRODUCT * min(max(n - 1, 0), Tune.Spend.EXTRA_PRODUCTS))
                        * min(max(Tune.Spend.STEP_REF / float(max(1, s.step_cap)), Tune.Spend.SCALE_MIN), 1.0))
        raw = self._joined.get(low_lab, low_lab)
        s.link.note("merge", label=raw, phrase=phrase, products=[P[i]["name"], P[j]["name"]], label_wording=bool(by_label))
        return ("The store files '%s' as one category, so '%s' is a single product here, not two: the order holds a "
                "single line for it%s, not one line for each kind the category title names. Requirements now: %s"
                % (raw, phrase, " (exactly one line, as it is now the only product)" if left < 2 else "", L.req_text()))

    # ================================================================== 3.21 learning; dead and stale lines
    def _stat(self, tool):
        st = self._stats.get(tool)
        if st is None:
            st = {"calls": 0, "ids": 0, "stock": 0, "props": 0, "notes": 0, "price": 0, "keys": set(),
                  "text_keys": set(), "note_keys": set(), "read_ids": set()}
            if tool:
                self._stats[tool] = st
        return st

    @_p06_guard(list)
    def learn(self, name, args, obs):
        s = self.s
        st = self._stat(name)
        st["calls"] += 1
        args = args if isinstance(args, dict) else {}
        asked = set(self.question_ids(name, args))
        if asked:
            self._tried[name] = self._tried.get(name, 0) + 1
        hist = self._calls.setdefault(name, [])
        hist.append(dict(args))
        del hist[:-12]
        if isinstance(obs, dict) and obs.get("error"):
            return []
        cartish = name is not None and name in (s.role("cart"), s.role("cart_view"))
        epoch, new, seen = s.cart_seq_mark, [], set()
        cart = set(self._cart())
        def look(d):
            lid = kit.row_id(d)
            if not lid:
                return
            src = "lookup" if (lid in asked or cartish) else "row"
            sv = kit.stock_of(d)
            if sv is not None:
                s.stock_seen = True
                h = self._shist.setdefault(lid, {})
                oth = h.get("row" if src == "lookup" else "lookup")
                if oth and oth[0] == epoch and oth[1] != sv:
                    s.rows_lag = True
                if src == "row" and "row" in h and h["row"][1] != sv:
                    s.rows_live = True
                h[src] = (epoch, sv)
                if sv:
                    if lid in s.dead and not (src == "row" and s.dead[lid] == "lookup"):
                        del s.dead[lid]
                else:
                    if lid in cart and lid not in s.dead and lid not in new:
                        new.append(lid)
                    s.dead[lid] = src
            if lid in asked:
                seen.add("ids")
                st["read_ids"].add(lid)
                st["keys"].update(d.keys())
                st["text_keys"].update(k for k, v in d.items() if isinstance(v, str))
                if sv is not None:
                    seen.add("stock")
                if self._has_map(d):
                    seen.add("props")
                if any(kit.is_num(v) and re.search(r"price|cost", str(k).lower()) for k, v in d.items()):
                    seen.add("price")
                rt = self._row_text.get(lid)
                for k, v in d.items():
                    if isinstance(v, str) and len(v.split()) >= 8 and (
                            (rt is None and k not in self._row_keys) or (rt is not None and rt.get(k) != v)):
                        st["note_keys"].add(k)
                        seen.add("notes")
            elif not cartish:
                self._row_keys.update(d.keys())
                self._row_text.setdefault(lid, {}).update((k, v) for k, v in d.items() if isinstance(v, str))
        kit.visit(obs, look)
        for f in seen:
            st[f] += 1
        return new

    @_p06_guard(list)
    def dated_lines(self):
        s = self.s
        return [l for l in self._cart() if s.line_seq.get(l, -1) < s.cart_seq_mark]

    @_p06_guard(dict)
    def tool_stats(self, tool):
        return self._stat(tool)

    @_p06_guard(set)
    def row_keys(self):
        return self._row_keys

    @_p06_guard(set)
    def row_fields(self, lid):
        return set(self._row_text.get(str(lid)) or ())

    # ================================================================== 3.22 stock tool, notes tool, role fix
    def _guess(self):
        if self._guessed is None:
            try:
                self._guessed = dict(self.s.intake.guess_roles() or {})
            except Exception:
                self._guessed = {}
        return self._guessed

    @_p06_guard(None)
    def supply_tool(self):
        s = self.s
        lt = self.query_tools()
        if not lt:
            return None
        view, det = s.role("view"), s.role("details")
        top = max(self._stat(t)["stock"] for t in lt)
        if top > 0:
            c = [t for t in lt if self._stat(t)["stock"] == top]
            return next((p for p in (view, det) if p in c), None) or sorted(c)[-1]
        g = self._guess()
        for t in (view, g.get("view"), det):
            if t and t in lt and (self._tried.get(t, 0) < 2 or self._stat(t)["stock"] > 0):
                return t
        bad = {g.get(r) for r in ("cart", "cart_view", "order", "decline", "message")}
        return next((t for t in lt if t not in bad and not self._tried.get(t)), None)

    @_p06_guard(None)
    def memo_tool(self):
        det = self.s.role("details")
        dk = self._stat(det)["keys"] if det else set()
        c = [t for t in self.query_tools() if t != det and (self._stat(t)["note_keys"] - dk)]
        if not c:
            return None
        top = max(self._stat(t)["notes"] for t in c)
        return sorted(t for t in c if self._stat(t)["notes"] == top)[-1]

    @_p06_guard(None)
    def fix_details_role(self):
        s = self.s
        if self._moved:
            return
        d = s.role("details")
        if not d:
            return
        sd = self._stat(d)
        if sd["ids"] < 2 or sd["props"] > 0:
            return
        c = [t for t in self.query_tools() if t != d and self._stat(t)["props"] > 0]
        if not c:
            return
        top = max(self._stat(t)["props"] for t in c)
        best = sorted(t for t in c if self._stat(t)["props"] == top)[-1]
        self._moved = True
        s.roles["details"] = best
        if s.roles.get("view") == best:
            s.roles["view"] = d
        keep = self._idx()["mapped"]
        before = len(s.archive.read)
        s.archive.read.intersection_update(keep)
        s.link.note("details_role", old=d, new=best, read_before=before, read_after=len(s.archive.read))

    # ================================================================== 3.23 sold-out note and swap
    @_p06_guard("")
    def soldout_note(self, lines):
        out = []
        for line in lines or []:
            self.s.event_seen = True
            t = self._soldout_one(str(line))
            if t:
                out.append(t)
        return " ".join(out)

    def _soldout_one(self, line):
        s = self.s
        L = s.ledger
        self._swap_page, self._swap_rows = None, []
        lab, multi, bud = self.label_of(line), s.multi(), L.budget
        u = s.basket.held(line)
        units = max(1, u if kit.is_num(u) else 1)
        rest = 0.0
        for l in self._cart():
            p = s.archive.price(l)
            if l != line and kit.is_num(p):
                q = s.basket.units.get(l)
                rest += p * (q if kit.is_num(q) else 1)
        cart = set(self._cart())
        def blocked(x):
            return x == line or x in s.dead or x in s.refused_lines or x in cart
        def affordable(p):
            return not kit.is_num(bud) or p * units + rest <= bud + 1e-9
        cheap = L.priority() == "cheapest"
        best = None
        for x in self._version_marks():
            if blocked(x):
                continue
            if lab:
                xl = self.label_of(x)
                if (multi and xl != lab) or (not multi and xl not in (lab, "")):
                    continue
            p = s.archive.price(x)
            if kit.is_num(p) and not affordable(p):
                continue
            k = (0, p, x) if kit.is_num(p) else (1, 0, x)
            if (best is None or k < best[0]) and not self._fails(x):
                best = (k, x)
        rows, page = [], None
        if not best and not (multi and not lab):
            slot = s.walker.product_slot(line) if multi else None
            vals = [(p, v) for p, v in (L.values_in_code(slot) or []) if not self.is_loose(v)]
            ver = str(L.version)
            failed = {k.split("|")[0] for k, v in s.rulings.items() if v and k.split("|")[1:2] == [ver]}
            ptr = set(s.tracer.chain_ids() or ())
            cand = []
            for r in s.archive.digest().values():
                x = str(r.get("id"))
                if blocked(x) or r.get("stock") is not True or (lab and r.get("shelf") != lab) or x in failed or x in ptr:
                    continue
                p = r.get("price") if kit.is_num(r.get("price")) else s.archive.price(x)
                if not kit.is_num(p) or not affordable(p) or s.tracer.excluded(r.get("fields") or {}, slot):
                    continue
                rd = x in s.archive.read
                if rd and (self._fails(x) or any(kit.clash(r, a, b) for a, b in vals)
                           or not all(kit.shows(r, a, b, s.code_hits.get(x)) for a, b in vals)):
                    continue
                cand.append((0 if rd else 1, p, x, rd))
            cand.sort()
            if cheap:
                un, rr = [c for c in cand if not c[3]], [c for c in cand if c[3]]
                if un and rr and un[0][1] < min(c[1] for c in rr):
                    cand.remove(un[0])
                    cand.insert(0, un[0])
            rows = cand[:3]
            self._swap_rows = [c[2] for c in rows if not c[3]]
            if not any(c[3] for c in cand):
                pick = None
                for key, e in (s.seeker.book() or {}).items():
                    car = s.walker.carries(key, slot) or []
                    if not car:
                        continue
                    spot = s.seeker.following_page(key)
                    if spot is None:
                        continue
                    k = (-len(car), len((e or {}).get("spots") or ()))
                    if pick is None or k < pick[0]:
                        pick = (k, key, spot)
                if pick:
                    page = s.seeker.page_call(pick[1], pick[2])
                    self._swap_page = page
        text = ("Cart line %s can no longer be bought: its latest record shows no availability, and a checkout still "
                "carrying it would be rejected." % line)
        if best:
            text += (" Plan: remove %s from the cart; put %s in its place (it passed the item check under today's "
                     "requirements%s); re-read it once more to confirm stock and price; then check out."
                     % (line, best[1], ", and it is the lowest-priced approved option" if cheap else ""))
        elif rows:
            text += (" Plan: remove %s; no approved substitute exists yet, so try these cheap available rows of its own "
                     "category, which the budget still allows, in this order: %s. Rows whose opened record fails are "
                     "already filtered out; an unread one must be opened before it is added. Add the first that passes, "
                     "re-read it, then check out." % (line, ", ".join(
                         "%s at %s%s" % (x, self._money(p), " (opened; shows every stated value)" if rd else "")
                         for _o, p, x, rd in rows)))
        else:
            text += " Plan: remove %s and search for another listing that satisfies the request." % line
        if page and not best:
            text += (" A further unread page exists for the search leading with a stated value: %s %s."
                     % (s.role("search") or "search", kit.render(page)))
        return text

    @_p06_guard(None)
    def swap_row(self):
        s = self.s
        cart = set(self._cart())
        return next((x for x in self._swap_rows if x not in s.archive.read and x not in s.dead and x not in cart), None)

    # ================================================================== 3.24 refusal re-read
    @_p06_guard("")
    def reread_refused(self, obs):
        s = self.s
        try:
            lead, sold = s.mender.soldout_deduction(obs)
        except Exception as e:
            s.link.fault("soldout_deduction_error", e)
            return s.mender.reread_queue() or ""
        tail = sold or s.mender.reread_queue() or ""
        return " ".join(x.strip() for x in (lead or "", tail) if x and x.strip())



# ------------------------------------------------------------ p07_mender.py
def _p07_safe(neutral=None):
    """Guard a public Mender method: log the fault and return the neutral value (a callable makes a fresh one)."""
    def wrap(fn):
        def run(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as e:
                self.s.link.fault(fn.__name__ + "_error", e)
                return neutral() if callable(neutral) else neutral
        run.__name__ = fn.__name__
        return run
    return wrap


class Mender:
    """05: refusal follow-ups, decline repair, candidate pools, near basket, cart freshness, list reader, read picks,
    message calls, product attribution, per-product question quota, open-property test, plan notes."""
    SAY_REREAD = ("The order was refused, so the availability of every cart line is looked up again: one of them may "
                  "have sold out.")
    SAY_FIRST = ("Before any question, a few listings of each wanted item are opened: the field names they carry decide "
                 "what is worth asking.")
    SAY_TOPUP = "A few more listings are opened, because the records read so far for some items showed very few fields."
    SAY_FIT = ("No opened listing writes the stated values yet, so rows found by the searches that carried those values "
               "are opened next; their field names guide the questions.")
    SAY_QUOTA = ("This item has no opened listing yet; opening a few of its listings gives its question a real field "
                 "name to use.")
    PAIR_HEAD = ("Kept open from a two-part question (only its first part was sent, or only its first part got an "
                 "answer), so each is next in line:")
    DRAFT_HEAD = ("Drafts the assistant wrote after opening listing records, never sent; a draft about a still-open "
                  "property that matches a listed key goes first, worded with that key's exact name:")
    HINT_HEAD = ("Guesses, not requirements: each line names a property, then the story detail behind it, likeliest "
                 "first.")
    KEY_HEAD = "Each row: field name, how many of this item's listings carry it so far, example values."

    def __init__(self, s):
        self.s = s
        self._marks = set()          # repair gate marks tried
        self._near_done = False
        self._near_basket = None     # (version, refusals, [lids])
        self._pools = {}
        self._squeeze = None
        self._first_put = set()
        self._put_q = []
        self._batch = None
        self._batch_known = False
        self._pre_round = 0
        self._pre_ids = []
        self._fit_done = False
        self._quota_done = set()
        self._plan_used = 0
        self._pc = None              # product_of cache

    # ------------------------------------------------------------------ small helpers
    def _quiet(self):
        return bool(self.s.link.mute or self.s.covering > 0)

    def _key(self, lid):
        return "%s|%s" % (lid, self.s.ledger.version)

    def _stale(self, lid):
        return self.s.line_seq.get(lid, -1) < self.s.cart_seq_mark

    def _cap(self):
        b = self.s.ledger.budget
        return b if kit.is_num(b) else None

    @staticmethod
    def _bare(text):
        return str(text or "").split("(", 1)[0].strip()

    @staticmethod
    def _norm(text):
        return " ".join(kit.low(kit.strip_marks(str(text or ""))).split())

    @staticmethod
    def _twords(text):
        return set(re.findall(r"[a-z0-9]{3,}", kit.low(text)))

    def _units(self, slot):
        if slot is None:
            q = self.s.ledger.quantity()
            return q if kit.is_num(q) and q > 0 else 1
        p = self.s.ledger.products()
        q = p[slot].get("qty") if 0 <= slot < len(p) else 1
        return q if kit.is_num(q) and q > 0 else 1

    def _group(self, i):
        hl = self.s.judge.home_labels() or []
        return {kit.low(x) for x in hl[i]} if i is not None and 0 <= i < len(hl) else set()

    def _price(self, lid):
        p = (self.s.archive.digest().get(lid) or {}).get("price")
        if kit.is_num(p):
            return p
        p = self.s.archive.price(lid)
        return p if kit.is_num(p) else None

    def _ver_ids(self, marks, need_value=False):
        v = str(self.s.ledger.version)
        if need_value:
            return {k.split("|")[0] for k, r in marks.items() if r and k.split("|")[1:2] == [v]}
        return {k.split("|")[0] for k in marks if k.split("|")[1:2] == [v]}

    def _belongs(self, line, slot):
        if not self.s.multi():
            return True
        p = self.item_for(str(line).split(":", 1)[0])
        return p is None or p == slot

    def _rows(self, slot):
        """Requirement rows of a product: (property, value, revealed)."""
        led = self.s.ledger
        stated = {self._norm(x) for x in (led.written or [])}
        out = []
        for line in led.in_force() or []:
            line = str(line)
            if ":" not in line or not self._belongs(line, slot):
                continue
            head, val = line.split(":", 1)
            prop = " ".join(re.sub(r"\([^)]*\)", " ", head).split())
            v = kit.strip_marks(val).strip()
            if prop and v:
                out.append((prop, v, self._norm(line) not in stated))
        return out

    def _word_value(self, v):
        t = kit.low(v)
        return (self.plain_value(v) and bool(re.search(r"[a-z]{3}", t)) and not re.search(r"\d", t)
                and not kit.has_exclusion(t) and not kit.is_pointer(t))

    def _lead_values(self, slot):
        out = []
        for line in self.s.ledger.in_force() or []:
            line = str(line)
            if ":" not in line or not self._belongs(line, slot):
                continue
            head, v = line.split(":", 1)
            v = kit.strip_marks(v).strip()
            if v and kit.is_positive(v, head.strip()) and not kit.is_pointer(v):  # flag 2: shared reference test
                out.append(v)
        return out

    def _field_texts(self, fields, prop):
        out = []
        for n, t in (fields or {}).items():
            t = str(t).strip()
            if t and kit.property_match(n, prop) and t not in out:
                out.append(t)
        return out

    # ------------------------------------------------------------------ 3.2 product of a text
    def item_for(self, text):
        try:
            names = tuple(str(p.get("name") or "") for p in self.s.ledger.products() or [])
            if len(names) < 2:
                return None
            if not self._pc or self._pc[0] != names:
                toks = [kit.kind_bits(n) for n in names]
                rivals = [{w for j, t in enumerate(toks) if j != i for w in t} for i in range(len(toks))]
                self._pc = (names, toks, rivals, {})
            _, toks, rivals, memo = self._pc
            t = str(text or "")
            if t in memo:
                return memo[t]
            m = re.search(r"\(([^)]*)\)", t)
            res = None
            for piece in ([m.group(1)] if m else []) + [t]:
                pt = kit.kind_bits(piece)
                if not pt:
                    continue
                sc = [kit.kind_overlap(tk, pt, rv) for tk, rv in zip(toks, rivals)]
                top = max(sc)
                if top <= 0:
                    continue
                lead = [i for i, v in enumerate(sc) if v == top]
                if len(lead) > 1:
                    lead = [i for i in lead if kit.kinds_inside(toks[i], pt)]
                if len(lead) == 1:
                    res = lead[0]
                    break
            if len(memo) > 5000:
                memo.clear()
            memo[t] = res
            return res
        except Exception as e:
            self.s.link.fault("product_of_error", e)
            return None

    def _entry_slot(self, e):
        p = self.item_for(str(e.get("key") or ""))
        return p if p is not None else self.item_for(str(e.get("wording") or ""))

    # ------------------------------------------------------------------ 3.3 values
    def plain_value(self, value):
        try:
            return not self.s.judge.is_loose(value)  # 05 3.3 plain = not the shared loose-value test (04 5.4)
        except Exception as e:
            self.s.link.fault("plain_value_error", e)
            return False

    # ------------------------------------------------------------------ 3.4 candidate pool
    def _home(self, slot):
        s = self.s
        if s.multi():
            return self._group(slot)
        cnt = {}
        dig = s.archive.digest()
        for lid in s.archive.read:
            lab = (dig.get(lid) or {}).get("shelf") or ""
            if lab:
                cnt[lab] = cnt.get(lab, 0) + 1
        if cnt:
            top = max(cnt.values())
            return {k for k, v in cnt.items() if v == top}
        return self._group(0)

    def _clashes(self, fields, judged, joined):
        out = []
        for p, v in judged:
            txt = self._field_texts(fields, p)
            if txt and not any(kit.seen_in(v, t) for t in txt) and not any(kit.seen(v, t) for t in txt):
                out.append(p)
        for p, parts in joined:
            txt = self._field_texts(fields, p)
            if txt and not all(any(kit.seen(x, t) for t in txt) for x in parts) and p not in out:
                out.append(p)
        return out

    @_p07_safe(list)
    def pool(self, slot):
        s = self.s
        led = s.ledger
        inf = tuple(str(x) for x in led.in_force() or [])
        key = (slot, led.version, len(s.archive.obs), len(s.archive.read), len(s.passed),
               sum(1 for r in s.rulings.values() if r), inf, led.budget, len(s.refs), frozenset(s.dead),
               frozenset(s.refused_lines))
        if key in self._pools:
            return self._pools[key]
        if len(self._pools) > 64:
            self._pools.clear()
        out = []
        home = self._home(slot)
        if home:
            multi = s.multi()
            scope = slot if multi else None
            units = self._units(scope)
            cv = list(led.values_in_code(scope) or [])
            joined = []
            for p, v in cv:
                parts = [x.strip() for x in str(v).split("+")]
                if "+" in str(v) and len(parts) >= 2 and all(re.search(r"[A-Za-z]{2}", x) for x in parts):
                    joined.append((p, parts))
            judged = [(p, v) for p, v in cv if self.plain_value(v)]
            needs = [(p, v) for p, v, r in self._rows(scope) if r]
            failed = self._ver_ids(s.rulings, True)
            live = self._ver_ids(s.passed)
            cap = self._cap()
            chain = set(s.tracer.chain_ids() or ())
            for lid, row in s.archive.digest().items():
                if row.get("shelf") not in home:
                    continue
                fields = row.get("fields") or {}
                read = lid in s.archive.read
                price = row.get("price") if kit.is_num(row.get("price")) else self._price(lid)
                gone = lid in s.dead or row.get("stock") is False or lid in s.refused_lines
                over = cap is not None and price is not None and price * units > cap + 1e-9
                pointer = lid in chain or bool(s.tracer.excluded(fields, scope))
                clash = self._clashes(fields, judged, joined)
                misses = [p for p, v in judged if not kit.shows(row, p, v)]
                ruled = lid in failed or (read and bool((s.judge.evidence_view(lid) or {}).get("fails")))
                verified = lid in live and not gone
                fit = (verified or (read and not misses and not clash)) and not (ruled or pointer or gone or over)
                out.append({"id": lid, "row": row, "read": read, "price": price, "gone": gone, "over": over,
                            "pointer": pointer, "clash": clash, "misses": misses, "ruled": ruled,
                            "verified": verified, "out": bool(gone or over or pointer or ruled or clash),
                            "revealed_ok": all(kit.shows(row, p, v) for p, v in needs), "fit": fit})
        self._pools[key] = out
        return out

    @_p07_safe(None)
    def cheapest_fit(self, slot):
        best = [(e["price"], e["id"]) for e in self.pool(slot) if e["fit"] and e["price"] is not None]
        return min(best) if best else None

    @_p07_safe(list)
    def unfit_products(self):
        out = []
        if self.s.multi():
            for slot in self.s.ledger.slots():
                flags = [e["out"] for e in self.pool(slot) if e["read"]]
                if flags and all(flags):
                    out.append(slot)
        return out

    # ------------------------------------------------------------------ 3.5
    @_p07_safe(list)
    def settled_differs(self, slot=None):
        s = self.s
        inf = [str(x) for x in s.ledger.in_force() or []]
        out = []
        for r in s.refs or []:
            if r.get("relation") != "differ" or r.get("status") != "resolved" or not str(r.get("value") or "").strip():
                continue
            if not s.tracer.live(r) or not any(s.tracer.resolves_to(q, r) for q in inf):
                continue
            if slot is not None and s.multi() and r.get("tag"):
                p = self.item_for("x (%s)" % r.get("tag"))
                if p is not None and p != slot:
                    continue
            out.append(r)
        return out

    @_p07_safe(False)
    def rules_cover_all(self):
        s = self.s
        inf = [str(x) for x in s.ledger.in_force() or []]
        n = sum(1 for sc in s.ledger.slots() for p, v in (s.ledger.values_in_code(sc) or []) if self.plain_value(v))
        refs = self.settled_differs()
        n += sum(1 for q in inf if any(s.tracer.resolves_to(q, r) for r in refs))
        return n >= len(inf)

    # ------------------------------------------------------------------ 3.6 refusal follow-ups
    @_p07_safe(lambda: ("", ""))
    def soldout_deduction(self, obs):
        s = self.s
        lines = [l for l, u in s.basket.units.items() if kit.is_num(u) and u > 0]
        if obs is None or len(lines) != 1 or self._quiet() or s.last_finish_kind != "order":
            return ("", "")
        notes = s.warden.buyer_lines(s.role("order"), obs, True) or []
        if not notes:
            return ("", "")
        led = s.ledger

        def shot():
            return (len(led.dropped or []), len(led.revealed or []), str(led.budget), tuple(led.in_force() or []))
        v0, f0 = led.version, shot()
        note = led.hear(list(notes), None) or ""
        lead = (" " + note) if note else ""
        if not note or led.version == v0 or shot() != f0:
            return (lead, "")
        lid = lines[0]
        s.dead[lid] = "lookup"
        s.link.note("soldout_deduced", id=lid)
        text = ("[stock] %s is now treated as sold out without a fresh lookup: it was the cart's only line and the "
                "store's refusal reports that an item is no longer available. " % lid)
        return (lead, text + (s.judge.soldout_note([lid]) or ""))

    @_p07_safe("")
    def reread_queue(self):
        s = self.s
        lines = list(s.basket.units)[:4]
        tool = s.judge.supply_tool()
        if (not lines or not tool or self._quiet() or s.judge.open_pending() or s.steps_left() < 3 + len(lines)
                or s.link.time_left() < 70):
            return ""
        s.judge.queue_open(lines, "cart", True, tool=tool, say=self.SAY_REREAD)
        return "[stock re-read queued] every cart line will be looked up again with %s." % tool

    # ------------------------------------------------------------------ 3.7 decline repair
    def _place(self, picks):
        """3.7.5: cart the picks by code, re-read stale ones; True when the cart now holds them, all in stock."""
        s = self.s
        was = s.link.mute
        s.link.mute = True
        try:
            bad = s.seeker.rule_cart(list(picks))
            ok = bad is not None and not bad and all((s.basket.held(l) or 0) > 0 for l, _ in picks)
            if ok:
                stale = [l for l, _ in picks if self._stale(l)]
                if stale and s.steps_left() - len(stale) >= 2:
                    looked, out = s.seeker.recheck_stock(stale)
                    if out:
                        ok = False
        finally:
            s.link.mute = was
        return bool(ok and s.basket.units and not any(l in s.dead for l in s.basket.units))

    def _fill(self):
        s = self.s
        lines = list(s.basket.units)
        dead = set(s.dead)
        backup = set(s.refused_lines)
        keep = {l for l in lines if l not in dead and not self._stale(l)}
        placed, picks = False, []
        try:
            s.refused_lines.difference_update(keep)
            own = self.rules_cover_all()
            several = s.multi()
            avoid, seen = set(dead), {}
            if several:
                for sl in s.ledger.slots():
                    for e in self.pool(sl):
                        seen[e["id"]] = e
                        if e["clash"]:
                            avoid.add(e["id"])

            def holds(lid):
                return self._key(lid) in s.passed or (own and lid in s.archive.read and bool(s.code_fit.get(lid)))

            def bare(lid):
                e = seen.get(lid)
                return e is None or not e["revealed_ok"] or bool(e["misses"])
            checks = rounds = 0
            while True:
                picks = list(s.closer.picks(avoid=set(avoid)) or [])
                if not several or not picks or checks >= 3 or rounds >= 30 or s.link.time_left() <= 100:
                    break
                wait = [l for l, _ in picks if not holds(l)]
                if not wait:
                    break
                drop = [l for l in wait if bare(l)]
                if drop:
                    avoid.update(drop)
                    avoid.update(l for l, e in seen.items() if not e["read"] and self._key(l) not in s.passed and bare(l))
                    rounds += 1
                else:
                    checks += 1
                    s.judge.item_check(wait[0], Tune.Aim.ORDER)
            if not picks or not all(holds(l) for l, _ in picks):
                s.link.note("repair", result="not done", picks=picks)
            elif s.steps_left() < 2 + len(picks):
                s.link.note("repair", result="not done, steps", picks=picks)
            else:
                placed = self._place(picks)
                s.link.note("repair", result="placed" if placed else "not placed", picks=picks,
                            cart=s.basket.summary())
        except Exception as e:
            s.link.fault("repair_fill_error", e)
            placed = False
        finally:
            if not placed:
                s.refused_lines.clear()
                s.refused_lines.update(backup)
        return placed

    @_p07_safe("")
    def repair(self):
        s = self.s
        if (not s.role("cart") or not s.role("order") or s.steps_left() < 3 or s.link.time_left() <= 70
                or self._quiet() or s.link.done):
            return ""
        mark = s.refusals if (s.refusals >= 1 or s.steps_left() <= s.intake.due_margin()) else "early"
        if mark in self._marks:
            return ""
        self._marks.add(mark)
        multi = s.multi()
        if multi:
            try:
                unfit = self.unfit_products()
            except Exception as e:
                s.link.fault("repair_proof_error", e)
                unfit = []
            if unfit:
                s.link.note("repair", result="unfit product", products=unfit)
                return ""
        if not self._fill():
            return ""
        what = "a fitting candidate for every product" if multi else "a fitting candidate"
        return (Tune.Marks.NOT_RUN_YET + "The decline is held: the records contain %s, either approved by the item "
                "check or opened with every requested value written in its own record, and a decline is only right "
                "when nothing fits. The code has already put %s in the cart and re-read the stock. %s %s"
                % (what, "them" if multi else "it", s.basket.summary(), s.judge.close_advice())).strip()

    # ------------------------------------------------------------------ 3.8 near basket
    def _near_ok_ruled(self, lid, lines):
        props = [p for p, _, _ in lines]
        for f in (self.s.judge.evidence_view(lid) or {}).get("fails") or []:
            fp = str(f).split(":", 1)[0].strip()
            if not any(kit.property_match(fp, p) for p in props):
                return False
        terms = set()
        for p, v, _ in lines:
            for w in list(kit.property_terms(p)) + re.findall(r"[a-z0-9]+", kit.low(v)):
                if len(w) > 2 and not kit.is_link(w):
                    terms.add(w)
        for prob in self.last_problems(lid) or []:
            low = kit.low(prob)
            if not any(re.search(r"\b" + re.escape(t), low) for t in terms):
                return False
        return True

    def _near_cands(self, slot, pool):
        rows = self._rows(slot)
        out = []
        for e in pool:
            if not e["read"] or e["price"] is None or e["gone"] or e["over"] or e["pointer"]:
                continue
            row, lines, ok = e["row"], [], True
            for p, v, rev in rows:
                if kit.shows(row, p, v):
                    continue
                txt = self._field_texts(row.get("fields"), p)
                if rev or not txt or not self._word_value(v) or any(re.search(r"\d", t) for t in txt):
                    ok = False
                    break
                lines.append((p, v, "; ".join(txt)[:120]))
            if not ok or not lines or (e["ruled"] and not self._near_ok_ruled(e["id"], lines)):
                continue
            out.append({"e": e, "lines": lines})
        out.sort(key=lambda c: (len(c["lines"]), c["e"]["price"], c["e"]["id"]))
        return out

    @_p07_safe("")
    def closest_check(self):
        s = self.s
        lk = s.link
        if (self._near_done or not s.role("cart") or not s.role("order") or s.steps_left() < 4 or lk.time_left() <= 100
                or self._quiet() or lk.done or lk.budget_gone() or lk.off):
            return ""
        multi = s.multi()
        slots = list(s.ledger.slots())
        keep = 3 if multi else 6
        cap = self._cap()
        chosen, near = {}, {}
        for sl in slots:
            pool = self.pool(sl)
            cf = self.cheapest_fit(sl)
            if cf:
                chosen[sl] = (cf[1], cf[0], self._units(sl))
                continue
            news = s.refusals >= 1 or any(self._belongs(x, sl) for x in list(s.ledger.revealed or []) +
                                          list(s.ledger.dropped or []))
            if not news:
                return ""
            c = self._near_cands(sl, pool)[:keep]
            if not c:
                return ""
            near[sl] = c
        if not near:
            return ""
        floor = sum(p * u for _, p, u in chosen.values())
        floor += sum(min(c["e"]["price"] for c in near[sl]) * self._units(sl) for sl in near)
        if cap is not None and floor > cap + 1e-9:
            return ""
        flat = []
        for sl in slots:
            for ci, c in enumerate(near.get(sl, [])):
                for p, v, w in c["lines"]:
                    flat.append((sl, ci, c["e"]["id"], p, v, w))
        flat = flat[:12]
        self._near_done = True  # set before the call: a failed call counts too
        body = "\n".join("%d. %s: %s = '%s' | listing field: %s" % (i + 1, lid, p, v[:60], w)
                         for i, (_, _, lid, p, v, w) in enumerate(flat))
        text = PROMPTS["meaning"] + "\n\nREQUEST (CONTEXT):\n" + s.request[:1500] + "\n\nLINES:\n" + body
        ans = lk.consult("match-wording", text, "meaning")
        got = ans.get("verdicts") if isinstance(ans, dict) else None
        agreed = {x.get("line") for x in (got if isinstance(got, list) else [])
                  if isinstance(x, dict) and x.get("same_value") is True and kit.is_num(x.get("line"))}
        lk.note("closest_check", lines=[list(x[2:]) for x in flat], agreed=sorted(agreed))
        pos = {}
        for i, x in enumerate(flat):
            pos.setdefault((x[0], x[1]), []).append(i + 1)
        confirmed = []
        for sl in slots:
            if sl not in near:
                continue
            pick = None
            for ci, c in enumerate(near[sl]):
                nums = pos.get((sl, ci), [])
                if len(nums) == len(c["lines"]) and all(n in agreed for n in nums):
                    pick = c
                    break
            if pick is None:
                return ""
            chosen[sl] = (pick["e"]["id"], pick["e"]["price"], self._units(sl))
            confirmed.append(pick)
        total = sum(p * u for _, p, u in chosen.values())
        if cap is not None and total > cap + 1e-9:
            return ""
        picks = [(chosen[sl][0], chosen[sl][2]) for sl in slots if sl in chosen]
        if s.steps_left() < len(picks) + 2:
            return ""
        for c in confirmed:  # commit (not undone when placing fails)
            lid = c["e"]["id"]
            key = self._key(lid)
            for k in list(s.rulings):
                if k == key or k.startswith(key + "|"):
                    s.rulings[k] = ""
            s.passed.add(key)
            s.near_ok.add(key)
            pr = s.archive.price(lid)
            s.pass_price[lid] = pr if kit.is_num(pr) else c["e"]["price"]
        placed = self._place(picks)
        lk.note("near_basket", picks=picks, placed=placed, cart=s.basket.summary())
        if not placed:
            return ""
        self._near_basket = (s.ledger.version, s.refusals, [l for l, _ in picks])
        said = []
        for c in confirmed:
            lid = c["e"]["id"]
            s.near_words[self._key(lid)] = [(p, v, w) for p, v, w in c["lines"]]
            said += ["%s: its %s field reads '%s', taken as '%s'" % (lid, p, w, v[:60]) for p, v, w in c["lines"]]
        return (Tune.Marks.NOT_RUN_YET + "The decline is held. A meaning check found these listings' own wording "
                "expresses the requested value: %s. The chosen listings meet every hidden need the shopper disclosed, "
                "are in stock and fit the budget; the code put them in the cart and re-read their stock. %s %s"
                % ("; ".join(said), s.basket.summary(), s.judge.close_advice())).strip()

    @_p07_safe("")
    def closest_pause(self):
        s = self.s
        nb = self._near_basket
        if not nb or nb[0] != s.ledger.version or nb[1] != s.refusals or not s.role("order"):
            return ""
        for lid in nb[2]:
            if (s.basket.held(lid) or 0) <= 0 or lid in s.dead or lid in s.refused_lines:
                return ""
        if s.steps_left() < 1 or s.link.done or s.link.broke or s.link.time_left() < 2:
            return ""
        self._near_basket = None
        s.link.note("closest_pause", lines=list(nb[2]))
        was = s.link.mute
        try:
            s.closer.force_finish()
        finally:
            s.link.mute = was
        return (Tune.Marks.NOT_RUN_YET + "Instead of declining, the code placed the order itself: an earlier meaning "
                "check judged the wording of the listed cart lines to express the requested values, and neither the "
                "requirements nor the cart changed since.")

    # ------------------------------------------------------------------ 3.9 budget squeeze
    def _lead_scores(self, slot):
        s = self.s
        sc = {lid: 1 for lid in s.all_hits or []}
        vals = []
        for p, v, _ in self._rows(slot):
            t = {w for w in re.findall(r"[a-z0-9]{2,}", kit.low(v)) if not kit.is_link(w)}
            if t:
                vals.append(t)
        n = s.lead_words or 0
        for ent in (s.seeker.book() or {}).values():
            q = []
            for _, txt in s.seeker.open_text(ent.get("args") or {}) or []:
                q += re.findall(r"[a-z0-9]{2,}", kit.low(txt))
            qs = set(q[:n] if n > 0 else q)
            score = sum(1 for t in vals if t & qs)
            if score > 0:
                for lid in ent.get("row_shelf") or {}:
                    sc[lid] = max(sc.get(lid, 0), score)
        return sc

    @_p07_safe(list)
    def squeeze_rows(self):
        s = self.s
        cap = self._cap()
        if not s.multi() or cap is None:
            return []
        slots = list(s.ledger.slots())
        fits = {sl: self.cheapest_fit(sl) for sl in slots}
        if any(f is None for f in fits.values()):
            return []
        floor = {sl: fits[sl][0] * self._units(sl) for sl in slots}
        total = sum(floor.values())
        if total <= cap + 1e-9:
            return []
        queues = []
        for sl in slots:
            u = self._units(sl)
            room = (cap - (total - floor[sl])) / max(1, u)
            lead = self._lead_scores(sl)
            ok = [e for e in self.pool(sl) if not (e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"])
                  and e["price"] is not None and room > 0 and e["price"] <= room + 1e-9]
            ok.sort(key=lambda e: (-lead.get(e["id"], 0), e["price"], e["id"]))
            queues.append([e["id"] for e in ok])
        rows = self.deal(queues, 3)[:3]
        if rows:
            self._squeeze = total
        return rows

    def squeeze_total(self):
        return self._squeeze

    # ------------------------------------------------------------------ 3.10
    @_p07_safe(None)
    def last_problems(self, lid):
        for e in reversed(self.s.link.log):
            if isinstance(e, dict) and e.get("kind") == "verify" and e.get("id") == lid:
                r = e.get("result")
                if isinstance(r, dict) and r.get("passes") is False:
                    f = r.get("faults")
                    f = f if isinstance(f, list) else ([f] if f else [])
                    return [str(x) for x in f if str(x).strip()]
                return None
        return None

    # ------------------------------------------------------------------ 3.11 cart bookkeeping
    @_p07_safe(None)
    def basket_shift(self, before):
        s = self.s
        before = before or {}
        s.cart_version += 1
        s.since_cart = 0
        now = s.basket.units
        if any(l not in before for l in now):
            s.cart_seq_mark = s.absorb
        for l, u in now.items():
            b = before.get(l)
            if b is not None and kit.is_num(u) and kit.is_num(b) and u > b + 1e-9:
                s.line_seq.pop(l, None)

    @_p07_safe(None)
    def stamp_current(self, obs):
        s = self.s
        if not s.stock_seen:
            return
        held = s.basket.units

        def stamp(d):
            rid = kit.row_id(d)
            if rid and rid in held and kit.stock_of(d) is not None:
                s.line_seq[rid] = s.absorb
        kit.visit(obs, stamp)

    @_p07_safe(None)
    def post_cart(self, args, obs):
        s = self.s
        if s.basket.shown_units(obs) is not None:
            s.cart_shown_version = s.cart_version
        new = []
        for lid, n in s.basket.lines_of(args or {}) or []:
            if (n is None or n > 0) and (s.basket.held(lid) or 0) > 0 and lid not in self._first_put and lid not in new:
                new.append(lid)
        if not new or not s.basket.is_cart_call(args or {}) or s.closer.has_fault(obs):
            return
        self._first_put.update(new)
        new = [l for l in new if self._stale(l)]
        tool = s.judge.supply_tool()
        if not new or not tool or self._quiet() or s.steps_left() < 4 or s.link.time_left() < 70:
            return
        # Judge keeps the live queue (emptied when the opener reads it) and adds these to it (05 3.11.2 step 5)
        s.judge.queue_put_reads(list(new), tool)
        s.link.note("put_reread", ids=list(new), tool=tool)

    @_p07_safe(set)
    def dead_with_recheck(self, ids):
        s = self.s
        ids = list(ids or [])
        gone = {i for i in ids if i in s.dead}
        unread = [i for i in ids if self._stale(i) and i not in gone]
        if not unread or not s.judge.supply_tool() or s.steps_left() - len(unread) < 2:
            return gone
        looked, out = s.seeker.recheck_stock(unread)
        return gone | set(out or ()) | {i for i in ids if i in s.dead}

    # ------------------------------------------------------------------ 3.12 list reader
    def batch_spec(self):
        if self._batch_known:
            return self._batch
        self._batch_known = True
        self._batch = None
        s = self.s
        try:
            taken = {s.role(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
            try:  # flag 1: tools the role guesser takes for writes/finishes stay out (no word stems)
                g = s.intake.guess_roles() or {}
                taken |= {g.get(r) for r in ("message", "cart", "cart_view", "order", "decline")} - {None}
            except Exception as e:
                s.link.fault("batch_roles_error", e)
            for spec in s.tools:
                name = (spec.get("function") or {}).get("name")
                if not name or name in taken:
                    continue
                sch = s.schema(name)
                props = sch.get("properties") or {}
                if not isinstance(props, dict):
                    continue
                pv = [p for p in props.values() if isinstance(p, dict)]
                # flag 1: no enum ("action") and no numeric ("quantity") parameter
                if any("enum" in p or p.get("type") in ("integer", "number") for p in pv):
                    continue
                arr = [k for k, p in props.items() if isinstance(p, dict) and p.get("type") == "array"
                       and isinstance(p.get("items"), dict) and p["items"].get("type") == "string"
                       and "enum" not in p["items"]]
                if len(arr) != 1 or any(r != arr[0] for r in sch.get("required") or []):
                    continue
                mx = props[arr[0]].get("maxItems")
                if kit.is_num(mx) and mx >= 2:
                    self._batch = (name, arr[0], int(min(mx, 5)))
                    break
        except Exception as e:
            s.link.fault("batch_spec_error", e)
            self._batch = None
        return self._batch

    def _reply_values(self, replies):
        out = []
        for r in replies or []:
            r = self._pair_text(r)
            if isinstance(r, str):
                t = r.split("\n" + Tune.Marks.NOTE)[0].strip()
                if t.endswith(Tune.Marks.OVER):
                    t = t[:-len(Tune.Marks.OVER)].strip()
                try:
                    out.append(json.loads(t))
                except Exception:
                    d = kit.json_from(t)
                    if d is not None:
                        out.append(d)
            elif isinstance(r, (dict, list)):
                out.append(r)
        return out

    @staticmethod
    def _pair_text(r):
        return r[1] if isinstance(r, (tuple, list)) and len(r) == 2 and isinstance(r[1], str) else r

    @staticmethod
    def _recordish(d):
        subs = [v for v in d.values() if isinstance(v, dict)]
        if subs:
            return any(sum(1 for x in v.values() if kit.is_plain(x)) >= 2 for v in subs)
        return sum(1 for v in d.values() if isinstance(v, str) and 1 <= len(v) <= 70) >= 5

    @_p07_safe(False)
    def batch_records_back(self, replies, asked):
        """Records came back (half the asked ids inside record-like dicts) and the reply is richer than rows."""
        ids = {str(x) for x in asked or [] if str(x)}
        if not ids:
            return False
        got, keys, deep = set(), set(), [False]
        data = self._reply_values(replies)

        def look(d):
            hit = {v for v in d.values() if isinstance(v, str) and v in ids}
            if not hit:
                return
            keys.update(str(k) for k in d)
            if any(isinstance(v, dict) and len(v) >= 2 for v in d.values()):
                deep[0] = True
            if self._recordish(d):
                got.update(hit)
        for x in data:
            kit.visit(x, look)
        if not got or len(got) * 2 < len(ids):
            return False
        try:
            rk = set(self.s.judge.row_keys() or ())
            return bool(deep[0] or not rk or (keys - rk))
        except Exception as e:
            self.s.link.fault("batch_richer_error", e)
            return True

    @_p07_safe(False)
    def batch_ids_echoed(self, replies, asked):
        ids = [str(x) for x in asked or [] if str(x)]
        if not ids:
            return False
        parts = []
        for r in replies or []:
            r = self._pair_text(r)
            parts.append(r if isinstance(r, str) else json.dumps(r, ensure_ascii=False, default=str))
        txt = "\n".join(parts)
        return any(i in txt for i in ids)

    @_p07_safe(False)
    def shrink_batch(self, asked):
        n = asked if isinstance(asked, int) else len(list(asked or []))
        spec = self.batch_spec()
        if not spec or n <= 2:
            return False
        size = max(2, min(spec[2], n - 1))
        self._batch = (spec[0], spec[1], size)
        self.s.link.note("batch_shrink", tool=spec[0], size=size)
        return True

    def drop_batch(self):
        self._batch_known = True
        self._batch = None

    # ------------------------------------------------------------------ 3.13 row picks
    def row_title(self, row):
        try:
            if not isinstance(row, dict):
                return ""
            t = row.get("title")
            if isinstance(t, str) and t.strip():
                return t
            for k, v in row.items():
                if "title" in kit.low(k) and isinstance(v, str) and v.strip():
                    return v
            # flag 3: else the longest short free text that is not the row's id or category (no heading-name list)
            rid, lab, best = kit.row_id(row), kit.shelf_field(row), ""
            for v in row.values():
                if not isinstance(v, str):
                    continue
                v2 = v.strip()
                if (v2 and len(v2) <= 160 and v2 != rid and kit.low(v2) != lab and not kit.looks_like_id(v2)
                        and re.search(r"[A-Za-z]{2}", v2) and len(v2) > len(best)):
                    best = v2
            return best
        except Exception as e:
            self.s.link.fault("row_title_error", e)
            return ""

    def _fit_read(self, shv, vals):
        """3.13.4: some opened listing of these shelves loosely shows every lead value in its property maps."""
        s = self.s
        for lid, row in s.archive.digest().items():
            f = row.get("fields") or {}
            if lid in s.archive.read and row.get("shelf") in shv and f:
                joined = " | ".join(str(x) for x in f.values())
                if all(kit.seen_in(v, joined) for v in vals):
                    return True
        return False

    def _lead_row(self, shv, m, pool):
        s = self.s
        try:
            slot = None
            if s.multi():
                slot = next((i for i in range(len(s.judge.home_labels() or [])) if self._group(i) == shv), None)
                if slot is None:
                    return None
            if m <= 0:
                return None
            vals = self._lead_values(slot)
            if not vals or self._fit_read(shv, vals):
                return None
            inpool = set(pool)
            walks = sorted((s.seeker.book() or {}).items(), key=lambda kv: -s.walker.tour_rating(kv[0], slot))
            for _, ent in walks:
                for lid in ent.get("row_shelf") or {}:
                    if lid in inpool:
                        return lid
        except Exception as e:
            s.link.fault("lead_row_error", e)
        return None

    @_p07_safe(list)
    def row_menu(self, shelves, m, skip=(), units=1, lead_only=False):
        s = self.s
        a = s.archive
        shv = {kit.low(x) for x in shelves or ()}
        skip = set(skip or ())
        if not shv:
            return []
        seen_t, gone, first = [], set(), {}

        def look(d):
            rid = kit.row_id(d)
            if not rid or rid not in a.known:
                return
            lab = kit.shelf_field(d)
            if not lab or lab not in shv:
                return
            if rid in a.read:
                w = self._twords(self.row_title(d))
                if w:
                    seen_t.append(w)
                return
            if kit.out_of_stock(d):
                gone.add(rid)
            if not first.get(rid):
                first[rid] = self.row_title(d)
        for o in a.obs:
            kit.visit(o, look)
        hits = sorted([h for h in s.all_hits or [] if h in first], key=lambda h: -(s.hit_weight.get(h, 0) or 0))
        hs = set(hits)
        cap = self._cap()
        pool = []
        for lid in hits + [i for i in first if i not in hs]:
            if lid in gone or lid in skip or lid in s.refused_lines:
                continue
            pr = self._price(lid) if cap is not None else None
            if pr is not None and pr * (units or 1) > cap + 1e-9:
                continue
            pool.append(lid)
        place = {lid: i for i, lid in enumerate(pool)}
        chosen = []
        lead = self._lead_row(shv, m, pool)
        if lead:
            chosen.append(lead)
            seen_t.append(self._twords(first.get(lead, "")))
        elif lead_only:
            return []

        def rank(lid):
            tw = self._twords(first.get(lid, ""))
            like = max([len(tw & t) / len(tw | t) for t in seen_t if tw | t] or [0])
            tier = 0 if lid in hs else (1 if (s.row_weight.get(lid, 0) or 0) > 0 else 2)
            return (tier, like, place[lid])
        while len(chosen) < m:
            win = [l for l in pool if l not in chosen][:16]
            if not win:
                break
            best = min(win, key=rank)
            chosen.append(best)
            seen_t.append(self._twords(first.get(best, "")))
        return chosen

    @staticmethod
    def deal(queues, limit):
        qs = [list(q) for q in queues or []]
        out, i = [], 0
        while len(out) < limit and any(i < len(q) for q in qs):
            for q in qs:
                if i < len(q) and len(out) < limit:
                    out.append(q[i])
            i += 1
        return out

    # ------------------------------------------------------------------ 3.14 reads before questions
    def _products_info(self):
        s = self.s
        counts = list(s.ledger.counts() or [1])
        out = []
        for i, sl in enumerate(s.ledger.slots()):
            g = self._group(i)
            u = counts[i] if i < len(counts) else counts[-1]
            rich = sum(s.archive.rich_count(x) for x in g) if g else 4
            out.append((sl, g, u if kit.is_num(u) and u > 0 else 1, rich))
        return out

    @_p07_safe(False)
    def pre_reads(self):
        s = self.s
        spec = self.batch_spec()
        if not spec or self._pre_round >= 2:
            return False
        first = self._pre_round == 0
        if len(s.question_log) > 1 or s.steps_left() <= (20 if first else 25) or s.link.time_left() < 130:
            return False
        size = spec[2]
        info = self._products_info()
        if first:
            bare, gaps = [], []
            for sl, g, u, r in info:
                if r == 0:
                    bare += self.row_menu(g, 2 if s.multi() else 4, skip=set(bare), units=u)
                elif r < 4:
                    gaps.append((g, u, 4 - r))
            ids = list(bare)
            for g, u, n in gaps:
                ids += self.row_menu(g, n, skip=set(ids), units=u)
            ids = ids[:size * (2 if len(bare) > size and s.steps_left() > 25 else 1)]
            if not ids:
                return False
            self._pre_round, self._pre_ids, say = 1, list(ids), self.SAY_FIRST
        else:
            self._pre_round = 2
            held, lists = set(self._pre_ids), []
            for sl, g, u, r in info:
                if r < 3:
                    got = self.row_menu(g, 4 - r, skip=set(held), units=u)
                    held.update(got)
                    lists.append(got)
            ids = self.deal(lists, size)
            if not ids:
                return False
            say = self.SAY_TOPUP
        s.judge.queue_open(ids, "menu", True, say=say)
        s.link.note("pre_reads", round=self._pre_round, ids=ids)
        return True

    @_p07_safe(False)
    def match_reads(self):
        s = self.s
        if (self._fit_done or s.question_log or s.msgs_sent or s.steps_left() <= 20
                or s.link.time_left() < 130):
            return False
        ids = []
        for sl, g, u, r in self._products_info():
            if not g:
                continue
            vals = self._lead_values(sl)
            if vals and not self._fit_read(g, vals):
                ids += self.row_menu(g, 2, skip=set(ids), units=u, lead_only=True)
        if not ids:
            return False
        self._fit_done = True
        s.judge.queue_open(ids, "menu", True, say=self.SAY_FIT)
        s.link.note("match_reads", ids=ids)
        return True

    # ------------------------------------------------------------------ 3.15 lacking rows
    @_p07_safe(list)
    def short_rows(self):
        s = self.s
        if not s.multi():
            return []
        live = (self._ver_ids(s.passed) - set(s.refused_lines) - set(s.dead)) | set(s.audit_fits or [])
        labs = {kit.low(s.judge.label_of(l)) for l in live} - {""}
        n = len(s.judge.home_labels() or [])
        groups = [self._group(i) for i in range(n)]
        lack = [i for i, g in enumerate(groups) if g and not (g & labs)]
        if not lack:
            return []
        shelf, trail, gone = {}, [], set()

        def look(d):
            rid = kit.row_id(d)
            if not rid or rid not in s.archive.known:
                return
            if kit.out_of_stock(d):
                gone.add(rid)
            lab = kit.shelf_field(d)
            if lab:
                trail.append(rid)
                shelf.setdefault(rid, lab)
        for o in reversed(s.archive.obs):
            kit.visit(o, look)
        hits = sorted([h for h in s.all_hits or [] if h in shelf], key=lambda h: -(s.hit_weight.get(h, 0) or 0))
        lists, done, cap = {i: [] for i in lack}, set(), self._cap()
        for lid in hits + trail:
            if lid in done:
                continue
            done.add(lid)
            if lid in s.archive.read or lid in s.refused_lines or lid in gone:
                continue
            owner = next((i for i in lack if shelf.get(lid) in groups[i]), None)
            if owner is None:
                continue
            pr = self._price(lid) if cap is not None else None
            if pr is not None and pr * self._units(owner) > cap + 1e-9:
                continue
            lists[owner].append(lid)
        spec = self.batch_spec()
        return self.deal([lists[i] for i in lack], spec[2] if spec else 3)

    # ------------------------------------------------------------------ 3.16 message call
    @_p07_safe(None)
    def sole_text_arg(self, tool):
        sch = self.s.schema(tool)
        props = sch.get("properties") or {}
        req = [r for r in sch.get("required") or [] if isinstance(r, str)]

        def plain(p):
            t = p.get("type") if isinstance(p, dict) else None
            return isinstance(p, dict) and "enum" not in p and (t == "string" or (isinstance(t, list) and "string" in t))
        texts = [k for k, p in props.items() if plain(p)]
        pick = next((r for r in req if r in texts), None)
        if pick is None and len(texts) == 1:
            pick = texts[0]
        if pick is None or any(r != pick for r in req):
            return None
        return pick

    @_p07_safe(None)
    def chat_call(self, text):
        tool = self.s.role("message")
        arg = self.sole_text_arg(tool) if tool else None
        if not arg:
            return None
        args, faults = ArgFixer.fix({arg: str(text)}, self.s.schema(tool))
        return None if faults or not isinstance(args, dict) else args

    # ------------------------------------------------------------------ 3.17 per-product quota
    def quota_order(self, plan, fresh=None):
        try:
            return self._quota(list(plan or []), fresh)
        except Exception as e:
            self.s.link.fault("quota_order_error", e)
            return plan

    def _quota(self, plan, fresh):
        s = self.s
        if not s.multi():
            return plan
        prods = s.ledger.products()
        n = len(prods)
        msg, ans, rev = [0] * n, [0] * n, [0] * n
        for line in s.question_log:
            line = str(line)
            subj = line.split(":", 1)[0]
            o = self.item_for(subj)
            if o is None:
                i = line.find('asked "')
                j = line.find('" -> ', i + 7) if i >= 0 else -1
                o = self.item_for(line[i + 7:j]) if j > i >= 0 else None
            if o is None or not 0 <= o < n:
                continue
            msg[o] += 1
            if "[not answered]" not in subj:
                ans[o] += 1
        first = self._entry_slot(plan[0]) if plan else None
        if not any(ans) and first is None:
            return plan
        hinted = set()
        try:
            keyed, keyless = set(), [0] * n
            for h in s.intake.pending_hints(None) or []:
                head = str(h).split(":", 1)[0]
                o = self.item_for(head)
                if o is None or not 0 <= o < n:
                    continue
                if s.intake.hint_has_no_key(o, self._bare(head), True):
                    keyless[o] += 1
                else:
                    keyed.add(o)
            hinted = {p for p in range(n) if p in keyed or keyless[p] > ans[p]}
        except Exception as e:
            s.link.fault("quota_hints_error", e)
            hinted = set()
        for line in s.ledger.revealed or []:
            o = self.item_for(str(line).split(":", 1)[0])
            if o is not None and 0 <= o < n:
                rev[o] += 1
        silent = {p for p in range(n) if ans[p] == 0}
        alt = set()
        for p in range(n):
            if msg[p] > ans[p]:
                try:
                    if s.intake.alternate_fresh(p):
                        alt.add(p)
                except Exception as e:
                    s.link.fault("quota_alt_error", e)
        keys = {}
        for p in range(n):
            waits = 1 if rev[p] > 0 and any(q != p for q in silent) else 0
            keys[p] = (waits, ans[p] + rev[p], 0 if p in hinted else 1, 0 if p in alt else msg[p] - ans[p], rev[p])
        low = min(keys.values())
        lowest = {p for p in range(n) if keys[p] == low}
        try:  # 3.17.3 lead jump
            if (first is not None and 0 <= first < n and first not in lowest and keys[first][:3] == low[:3]
                    and msg[first] <= min(msg) + 1 and any(self._entry_slot(e) in lowest for e in plan[1:])
                    and s.intake.question_room() - s.msgs_sent - 1 >= sum(1 for q in range(n)
                                                                         if q != first and msg[q] == 0)):
                s.link.note("quota_jump", product=first)
                return plan
        except Exception as e:
            s.link.fault("quota_jump_error", e)

        def ok(lab, p):
            return bool(lab) and (fresh is None or fresh(lab)) and not s.intake.requested_by(p, lab)
        for k in sorted(set(keys.values())):
            due = [p for p in range(n) if keys[p] == k]
            for i, e in enumerate(plan):
                if self._entry_slot(e) in due:
                    return [e] + plan[:i] + plan[i + 1:]
            for e in plan:
                sec = str(e.get("next_key") or "")
                p = self.item_for(sec) if sec else None
                if p in due:
                    m = re.search(r"\(([^)]*)\)", sec)
                    pname = m.group(1).strip() if m else str(prods[p].get("name") or "")
                    return [{"key": sec, "next_key": "", "wording": s.asker.plain_question(sec, pname)}] + plan
            for e in s.spare_asks or []:
                if isinstance(e, dict):
                    p = self._entry_slot(e)
                    if p in due and ok(str(e.get("key") or ""), p):
                        return [e] + plan
            for p in sorted(due, key=lambda q: (0 if q in hinted else 1, q)):
                pk = self._quota_pick(p, ans[p] == 0, fresh)
                if pk and ok(pk["key"], p):
                    return [pk] + plan
                if s.quota_wait:
                    return plan
        return plan

    def _quota_menu(self, p):
        """Trimmed key menu of the product's most-read home shelf; None when no shelf is known."""
        s = self.s
        group = self._group(p)
        shelf = next((x for x in s.archive.shelves_opened() or [] if kit.low(x) in group), None)
        if shelf is None and p in self._quota_done:
            shelf = next((x for x in sorted(group) if (s.archive.menu(x) or ("",))[0]), None)
        if shelf is None:
            return None
        menu = (s.archive.menu(shelf) or ("",))[0] or ""
        return s.intake.trim_menu(menu, s.intake.detailed_reads(p)) if menu else ""

    def _quota_pick(self, p, fallback, fresh):
        s = self.s
        name = str(s.ledger.products()[p].get("name") or "")
        menu = self._quota_menu(p)
        if menu is None:
            group = self._group(p)
            if not group:
                return None
            spec = self.batch_spec()
            ids = self.row_menu(group, min(4, spec[2]) if spec else 2, units=self._units(p))
            if not ids or (not s.role("details") and not spec) or p in self._quota_done or s.steps_left() <= 12:
                return None
            self._quota_done.add(p)
            s.judge.queue_open(ids, "quota", True, say=self.SAY_QUOTA)
            s.quota_wait = True
            s.link.note("quota_reads", product=p, ids=ids)
            return None

        def ok(lab, field):
            return (fresh is None or fresh(lab)) and not s.intake.requested_by(p, field)

        def entry(field):
            lab = "%s (%s)" % (field, name)
            return {"key": lab, "next_key": "", "wording": s.asker.plain_question(lab, name)}
        got = self._plan_with(p, menu)
        if got and ok(got["key"], got["key"]):
            return got
        for h in s.intake.pending_hints(p) or []:
            f = self._bare(str(h).split(":", 1)[0])
            if f and not kit.is_junk_key(f) and not s.intake.hint_has_no_key(p, f, True) and ok("%s (%s)" % (f, name), f):
                return entry(f)
        if fallback:
            rows = [r for r in menu.splitlines() if r.strip().startswith("-")]
            for by_samples in (False, True):
                for r in rows:
                    f = r.strip().lstrip("-").strip().split(" (in ")[0].strip()
                    if f and self.open_property(p, f, r, by_samples) and ok("%s (%s)" % (f, name), f):
                        return entry(f)
        return None

    # ------------------------------------------------------------------ 3.18 product plan
    @_p07_safe(None)
    def item_scheme(self, slot):
        if slot is None or not self.s.multi():
            return None
        return self._plan_with(slot, self._quota_menu(slot))

    def _plan_with(self, p, menu):
        s = self.s
        try:
            if not menu or self._plan_used >= 1 or s.link.time_left() < 110:
                return None
            self._plan_used += 1
            name = str(s.ledger.products()[p].get("name") or "")
            big = name.upper()
            count = min(3, max(1, s.intake.question_room() - s.msgs_sent))
            parts = [PROMPTS["planner"].replace("<<COUNT>>", str(count)) + "\n\n" + PROMPTS["focus"].replace(
                "<<PRODUCT>>", big), "REQUEST, AS WRITTEN:\n" + s.request,
                "REQUIREMENTS IN FORCE:\n" + s.ledger.req_text()]
            if s.question_log:
                parts.append("QUESTIONS SO FAR, WITH REPLIES:\n" + "\n".join(str(x) for x in s.question_log))
            hints = list(s.intake.pending_hints(p) or [])[:6]
            if hints:
                parts.append("STORY GUESSES FOR %s:\n%s\n" % (big, self.HINT_HEAD) +
                             "\n".join("- " + s.intake.hint_tag(h) for h in hints))
            notes = self.scheme_notes(p)
            if notes:
                parts.append("PLAN NOTES:\n" + notes)
            parts.append("KEYS OF %s:\n%s\n%s" % (big, self.KEY_HEAD, menu))
            ans = s.link.consult("plan-product", "\n\n".join(parts), "focus")
            s.link.note("item_scheme", product=p, reply=kit.cut(kit.render(ans), 2000))
            asks = ans.get("asks") if isinstance(ans, dict) else None
            inter = s.intake

            def plain(f):
                return s.asker.plain_question("%s (%s)" % (f, name), name)
            for e in asks if isinstance(asks, list) else []:
                if not isinstance(e, dict):
                    continue
                q = str(e.get("wording") or "").strip()
                lead = self._bare(e.get("key"))
                spare = self._bare(e.get("next_key"))
                sl = spare.lower()
                if "none" in sl or sl == "-" or "n/a" in sl:  # "nonetheless" also counts as empty (kept)
                    spare = ""
                if lead and inter.hint_has_no_key(p, lead):
                    lead = ""
                if spare and inter.hint_has_no_key(p, spare):
                    spare = ""
                if lead and inter.open_for_planning(p, lead):
                    topic = bool(kit.kind_terms(q) & kit.kind_terms(name))
                    paired = (topic and bool(spare) and not kit.property_match(spare, lead)
                              and inter.open_for_planning(p, spare))
                    return {"key": "%s (%s)" % (lead, name), "next_key": "%s (%s)" % (spare, name) if paired else "",
                            "wording": q if topic else plain(lead)}
                if spare and inter.open_for_planning(p, spare):
                    return {"key": "%s (%s)" % (spare, name), "next_key": "", "wording": plain(spare)}
            return None
        except Exception as e:
            s.link.fault("product_plan_error", e)
            return None

    # ------------------------------------------------------------------ 3.19 open-property test
    def open_property(self, slot, field, row=None, by_samples=False):
        s = self.s
        try:
            name = self._bare(field)
            if not name or kit.is_junk_key(name):
                return False
            multi = s.multi()

            def ours(text, tagged=False):
                if slot is None or not multi:
                    return True
                o = self.item_for(text)
                return (not tagged) if o is None else o == slot
            for line in s.question_log:
                subj = re.sub(r"^\s*-\s*", "", str(line).split(":", 1)[0]).strip()
                if "[not answered]" in subj:
                    continue
                for part in subj.split(";"):
                    pb = self._bare(re.sub(r"\[[^\]]*\]", " ", part))
                    if pb and ours(part) and s.intake.field_alike(pb, name, slot):
                        return False
            r = str(row or "")
            i = r.find("):")
            samples = kit.low(r[i + 2:]) if row and i >= 0 else ""  # rows with notes have no "):" (kept)
            for line in s.ledger.in_force() or []:
                line = str(line)
                if ":" not in line:
                    continue
                head, val = line.split(":", 1)
                if not ours(head):
                    continue
                prop, v = self._bare(head), kit.strip_marks(val).strip()
                if not prop:
                    continue
                if kit.exact_property(prop, name):
                    if s.intake.settles(head.strip() + ": " + v, samples if by_samples else ""):
                        return False
                elif kit.property_match(prop, name):
                    pool = samples or " | ".join(sorted(str(x) for x in s.intake.row_values(slot, name) or ()))
                    if kit.is_positive(v, head) and kit.seen_in(v, pool):
                        return False
                elif samples and not re.search(r"\d", v) and kit.is_positive(v, head):
                    ws = kit.content_words(kit.low(v))
                    if ws and all(re.search(r"\b" + re.escape(w), samples) for w in ws):
                        return False
            for fr in s.ledger.freed or []:
                fr = str(fr)
                if ours(fr, True) and s.intake.field_alike(self._bare(fr), name, slot):
                    return False
            return True
        except Exception as e:
            s.link.fault("open_property_error", e)
            return True

    # ------------------------------------------------------------------ 3.20 plan notes
    @_p07_safe("")
    def scheme_notes(self, slot=None):
        s = self.s
        asked = set()
        for line in s.question_log:
            subj = re.sub(r"^\s*-\s*", "", str(line).split(":", 1)[0])
            asked.update(kit.low(x).strip() for x in subj.split(";"))
        narrow = slot is not None and s.multi()

        def mine(t):
            return not narrow or self.item_for(t) in (slot, None)
        pairs = []
        for x in s.open_pairs or []:
            x = str(x)
            if mine(x) and kit.low(x).strip() not in asked:
                own = slot if slot is not None else self.item_for(x)
                if s.intake.open_for_planning(own, x):
                    pairs.append(x)
        drafts = [str(d) for d in s.drafts or [] if mine(str(d))]
        out = []
        if pairs:
            out.append(self.PAIR_HEAD + "\n" + "\n".join("- " + x for x in pairs[-3:]))
        if drafts:
            out.append(self.DRAFT_HEAD + "\n" + "\n".join("- " + x for x in drafts[-3:]))
        return "\n\n".join(out)



# ------------------------------------------------------------ p08_asker.py
class Asker:
    """Hidden-need asking engine (06 1.1, 2.1, 3 except 3.20, 7, 8, 10)."""
    COLD_PAD = 12
    PACE_PAD = 14
    PACE_STEPS = 4
    SURVEY_PAD = 8
    ROOMY = 12
    REPLY_STEPS = 4
    FALLBACK_READS = 3
    THIN_READS = 2
    WIDE_GAP = 2
    HINTS_MAX = 8
    CATS_MAX = 4
    REPLY_CLIP = 400
    ROW = re.compile(r"\s*- (.+?) \(in (\d+) listings?\)(?: \[varies: (\d+)\s+distinct\s+values?\])?")
    COUNT_COLON = re.compile(r"\(in \d+ listings?\):")
    MARKER = re.compile(r"(:|,|such\s+as|for\s+example|e\.g\.)(.*)", re.I | re.S)

    def __init__(self, s):
        self.s = s
        self._reasked = set()
        self._wait = None
        self._last = None
        self._surveyed = False
        self._no_pairs = False  # as built: written, never read (06 10.2)

    # ---------------------------------------------------------------- small helpers (3.21, 8)
    @staticmethod
    def _sing(w):
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w

    @staticmethod
    def _norm(t):
        return " ".join(re.findall(r"[a-z0-9]+", kit.low(t)))

    @staticmethod
    def _bare(label):
        return kit.name_head(str(label or "")).strip()

    @staticmethod
    def _head(label):
        return str(label or "").split(";", 1)[0].strip()

    @staticmethod
    def _second(v):
        t = str(v or "").strip()
        lo = t.lower()
        return "" if (not t or "none" in lo or lo == "-" or "n/a" in lo) else t

    def _mentions(self, found, name):
        return any(w in found for w in re.findall(r"[a-z]{4,}", kit.low(self._bare(name))))

    def _freed(self):
        L = self.s.ledger
        return len(L.freed or []) + int(L.pinned or 0)

    def _must(self):
        v = (self.s.ledger.reading or {}).get("ask_before") or []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    def _slot(self, label):
        return self.s.mender.item_for(str(label)) if self.s.multi() else None

    def _asked(self):
        out = set()
        for line in self.s.question_log:
            subj = str(line).split(":", 1)[0].lstrip("- ").strip()
            for p in subj.split(";"):
                p = p.strip().lower()
                if p:
                    out.add(p)
        return out

    def _messaged(self):
        s, cnt = self.s, {}
        for line in s.question_log:
            line = str(line)
            slot = s.mender.item_for(line.split(":", 1)[0].lstrip("- ").strip())
            if slot is None:
                m = re.search(r'"(.*?)" -> ', line, re.S)
                slot = s.mender.item_for(m.group(1)) if m else None
            if slot is not None:
                cnt[slot] = cnt.get(slot, 0) + 1
        return cnt

    def _open(self, prop):
        prop = str(prop or "").strip()
        if prop and prop.lower() not in (x.lower() for x in self.s.open_pairs):
            self.s.open_pairs.append(prop)

    def _rows(self, text):
        out = []
        for line in str(text or "").split("\n"):
            m = self.ROW.match(line)
            if m:
                out.append((m.group(1).strip(), int(m.group(2)), int(m.group(3) or 1), line.strip()))
        return out

    # ---------------------------------------------------------------- public services
    def reads_wanted(self):
        return Tune.Ask.READS + (0 if self.s.multi() else Tune.Ask.SOLO_EXTRA)

    def product_tag(self, label):
        s = self.s
        try:
            tags = kit.tags_of(str(label or ""))
            if tags and str(tags[0]).strip():
                return str(tags[0]).strip()
            if s.multi():
                bare = self._bare(label)
                for i, p in enumerate(s.ledger.products()):
                    if bare and s.intake.menu_row(i, bare):
                        return str(p.get("name") or "").strip() or "item"
            return str(s.ledger.product_name() or "").strip() or "item"
        except Exception as e:
            s.link.fault("product_tag_error", e)
            return "item"

    def plain_question(self, label, product=""):
        return "Which %s would you like for the %s?" % (self._bare(label), product or self.product_tag(label))

    def fresh(self, label):
        s = self.s
        try:
            lab = str(label or "").strip()
            if not lab or lab.lower() in s.spent_props or lab.lower() in self._asked():
                return False
            bare = self._bare(lab)
            if not bare or kit.is_junk_key(bare) or kit.is_numbered_twin(bare):
                return False
            slot = self._slot(lab)
            if s.intake.hint_has_no_key(slot, lab, False):
                return False
            return bool(s.intake.open_for_planning(slot, lab))
        except Exception as e:
            s.link.fault("fresh_error", e)
            return False

    def answer_shopper(self, phrase):
        s = self.s
        try:
            tool, room, left, t = s.role("message"), s.intake.question_room(), s.steps_left(), s.link.time_left()
            if not tool or not str(phrase or "").strip():
                return False
            if s.msgs_sent >= room or left <= self.REPLY_STEPS or t < Tune.Clock.CLOSING:
                return False
            if s.asking and s.msgs_sent >= room - 1 and left > Tune.Ask.STOP_ROOM and t >= Tune.Clock.ASK_STOP:
                return False  # the last question slot is kept for a hidden need
            args = s.mender.chat_call(str(phrase))
            if args is None:
                return False
            call = s.engine.call(tool, args, cid="reply_%d" % s.link.steps)
            pairs = s.engine.dispatch("I will answer the shopper's question first.", [call])
            s.engine.send_replies(pairs)
            return True
        except Exception as e:
            s.link.fault("answer_error", e)
            return False

    def ask(self, urgent=False):
        try:
            return self._pass(bool(urgent))
        except Exception as e:
            self.s.link.fault("ask_error", e)
            return False

    # ---------------------------------------------------------------- the pass (3.2)
    def _pass(self, urgent):
        s, A = self.s, Tune.Ask
        room = s.intake.question_room()
        tool = s.role("message")
        if not tool or s.msgs_sent >= room:  # gate (3.3)
            s.asking = False
            return False
        multi = s.multi()
        scope, sets = self._scope(multi)
        log = s.question_log
        must = self._must()
        steps, left = s.link.steps, s.steps_left()
        pressing = bool(must) and not log and (sets >= 1 or steps >= A.EARLIEST)
        want = self.reads_wanted()
        if not (sets >= want or pressing or urgent):  # timing (3.4)
            if steps < A.EARLIEST or (steps < A.EARLIEST + A.GRACE and left > A.GRACE_ROOM):
                return False
        margin = s.intake.due_margin()
        if left <= max(A.STOP_ROOM, margin) or s.link.time_left() < Tune.Clock.ASK_STOP:
            s.asking = False
            return False
        R = room - s.msgs_sent
        nprod = len(s.ledger.products())
        if 2 <= nprod <= 3 and not urgent:
            cnt = self._messaged()
            if any(cnt.values()) and left > margin + 2 * R + self.COLD_PAD:
                if any(not cnt.get(i) and s.intake.detailed_reads(i) == 0 for i in range(nprod)):
                    return False
        if self._wait is not None and sets <= self._wait and not urgent:
            return False
        if not multi and self._last and sets <= self._last[0] and not urgent and not pressing \
                and left > margin + 2 * R + self.PACE_PAD and steps - self._last[1] < self.PACE_STEPS:
            return False
        if s.mender.pre_reads() or s.mender.match_reads():  # reads before asking (3.5)
            return True
        if not log and s.msgs_sent == 0 and not pressing and not urgent:
            if self._survey():
                return True
            if s.intake.opening_reads():
                return True
        a = s.archive
        menu = a.menu()[0] if scope is None else "\n".join(t for t in (a.menu(h)[0] for h in scope) if t)
        disp = s.intake.trim_menu(menu, sets) if menu else ""
        if multi:
            menu, disp = self._cat_menus(menu, disp)
        queue = [l for l in s.ask_queue if self.fresh(l)]  # source (3.9)
        s.ask_queue[:] = queue
        pending = queue[0] if queue else ""
        if pending:
            text = s.prepared.pop(pending.lower(), "") or self.plain_question(pending)
            plan = [{"key": pending, "next_key": "", "wording": text}]
        else:
            ans = s.link.consult("plan-questions", self._brief(R, disp, multi, must), "planner")
            if ans is None and s.link.misses > 0:
                self._wait = sets
                return False
            plan = self._vet(ans)
        if not pending and not multi:  # scripting (3.11)
            if not plan:
                if sets >= self.FALLBACK_READS and all(s.intake.detailed_reads(i) >= self.THIN_READS for i in s.ledger.slots()):
                    key = next((r[0] for r in self._home_keys(False) if self.fresh(r[0])), "")
                    if key:
                        s.link.note("menu_fallback", key=key)
                        plan = [{"key": key, "next_key": "", "wording": self.plain_question(key)}]
            elif log and len(s.quiet_props) >= len(log):
                plan = self._widest(plan)
        elif not pending:
            ordered = s.mender.quota_order(plan, self.fresh)
            keys = {str(e.get("key") or "").lower() for e in plan}
            s.spare_asks[:] = list(plan) + [e for e in s.spare_asks if str(e.get("key") or "").lower() not in keys]
            if s.quota_wait:
                s.quota_wait = False
                return True
            plan = ordered if isinstance(ordered, list) else plan
        call = None
        for e in plan[:2]:  # shape and send (3.12, 3.13)
            if not isinstance(e, dict):
                continue
            key = str(e.get("key") or "").strip()
            if not key:
                continue
            text = self._shape(e, key, pending, menu, multi, must)
            split, sec = "", self._second(e.get("next_key"))
            if sec and sec.lower() != key.lower():
                text, split = self.plain_question(key), sec  # a pair is never sent in one message
            args = s.mender.chat_call(text)
            if args is not None:
                call = s.engine.call(tool, args, cid="ask_%d_%d" % (s.msgs_sent + 1, s.link.steps))
                break
        if call is None:
            return self._empty(sets, want, urgent)
        return self._send(call, key, text, split, pending, sets, room, multi)

    # ---------------------------------------------------------------- gate scope (3.3), menus (3.7, 3.11)
    def _scope(self, multi):
        s, a = self.s, self.s.archive
        if not multi:
            try:
                own = s.judge.home_shelf()
                if own:
                    return [own], a.rich_count(own)
                homes = s.judge.home_labels()
                home = sorted({kit.low(h) for h in (homes[0] if homes else ())})
                if home and any(kit.low(x) not in home for x in a.shelves_opened()):
                    return home, sum(a.rich_count(h) for h in home)
            except Exception as e:
                s.link.fault("ask_scope_error", e)
        return None, a.rich_count()

    def _cat_menus(self, menu, disp):
        s, a = self.s, self.s.archive
        opened, seen = [], set()
        for x in a.shelves_opened():
            if kit.low(x) not in seen:
                seen.add(kit.low(x))
                opened.append(x)
        homes = [{kit.low(h) for h in hs} for hs in s.judge.home_labels()]
        allhome = set().union(*homes) if homes else set()
        own = [x for x in opened if kit.low(x) in allhome]
        cats = (own or opened) if homes and all(homes) else own + [x for x in opened if x not in own]
        full, show = [], []
        for c in cats[:self.CATS_MAX]:
            t = a.menu(c)[0]
            if t:
                full.append("KEYS ON %s:\n%s" % (c, t))
                show.append("KEYS ON %s:\n%s" % (c, s.intake.trim_menu(t, a.rich_count(c))))
        return ("\n\n".join(full), "\n\n".join(show)) if full else (menu, disp)

    def _home_keys(self, all_keys):
        s = self.s
        homes = s.judge.home_labels()
        rows = []
        for h in sorted(homes[0]) if homes else []:
            for name, n, m, _ in self._rows(s.archive.menu(h)[0]):
                free = n >= 3 and m >= n  # every listing showed its own value
                if all_keys or not free:
                    rows.append((name, n, free))
        rows.sort(key=lambda r: -r[1])
        return rows

    # ---------------------------------------------------------------- brief (3.6) and vetting (3.10)
    def _brief(self, R, disp, multi, must):
        s, L = self.s, self.s.ledger
        top = PROMPTS["planner"].replace("<<COUNT>>", str(R)) + "\n\n" + PROMPTS["one_question"].replace("<<LEFT>>", str(R))
        parts = [top, "REQUEST, AS WRITTEN:\n" + str(s.request), "REQUIREMENTS IN FORCE:\n" + L.req_text()]
        if s.question_log:
            parts.append("QUESTIONS SO FAR, WITH REPLIES:\n" + "\n".join(str(x) for x in s.question_log))
        if must:
            parts.append("MUST LEARN FROM THE SHOPPER:\n" + "; ".join(must))
        notes = s.mender.scheme_notes()
        if notes:
            parts.append("PLAN NOTES:\n" + notes)
        hints = (s.intake.pending_hints(None) or [])[:self.HINTS_MAX]
        if hints:
            parts.append("STORY HINTS (not requirements):\nEach line names a property, then the story detail that points"
                         " at it. These are guesses, not requirements; usually one of these properties holds the unsaid"
                         " need.\n" + "\n".join("- " + s.intake.hint_tag(h) for h in hints))
        if multi:
            parts.append("ITEMS:\n" + " | ".join("%d) %s x%s" % (i + 1, p.get("name"), p.get("qty"))
                                                 for i, p in enumerate(L.products())))
        parts.append("KEYS IN THE LISTINGS:\n" + (disp or "(none seen yet)"))
        return "\n\n".join(p for p in parts if p)

    def _keyless(self, label):
        return bool(self.s.intake.hint_has_no_key(self._slot(label), label, False))

    def _vet(self, ans):
        asks = ans.get("asks") if isinstance(ans, dict) else None
        out = []
        for e in asks if isinstance(asks, list) else []:
            if not isinstance(e, dict) or not isinstance(e.get("wording"), str) or not e["wording"].strip():
                continue
            key = str(e.get("key") or "").strip()
            given = self._second(e.get("next_key"))
            if not self.fresh(key):
                if given and self.fresh(given):
                    out.append({"key": given, "next_key": "", "wording": self.plain_question(given)})
                continue
            ent = {"key": key, "next_key": given if given and self.fresh(given) else "", "wording": e["wording"].strip()}
            if given and self._keyless(given):
                ent["wording"] = self.plain_question(key)
            out.append(ent)
        return out

    def _widest(self, plan):
        s = self.s
        first = str(plan[0].get("key") or "")
        fb = self._bare(first)
        for h in (s.ledger.reading or {}).get("story_hints") or []:
            head = re.sub(r"\([^)]*\)", "", str(h).split(":", 1)[0])
            if any(p.strip() and kit.property_match(fb, p.strip()) for p in head.split("/")):
                return plan
        keys = self._home_keys(True)
        n0 = next((n for k, n, _ in keys if kit.exact_property(fb, k)), 0)
        wide = next(((k, n) for k, n, free in keys if not free and self.fresh(k)), None)
        if wide and not kit.property_match(wide[0], fb) and wide[1] >= n0 + self.WIDE_GAP:
            s.link.note("widest_first", key=wide[0], over=first)
            return [{"key": wide[0], "next_key": "", "wording": self.plain_question(wide[0])}] + list(plan)
        return plan

    # ---------------------------------------------------------------- wording (3.12)
    def _shape(self, e, key, pending, menu, multi, must):
        s = self.s
        q = str(e.get("wording") or "").strip()
        plain = self.plain_question(key)
        if not q:
            return plain
        if multi:
            tags = kit.tags_of(key)
            if tags:
                ws = re.findall(r"[a-z]{3,}", kit.low(tags[0]))
                lq = kit.low(q)
                if ws and not any((w[:-1] if w.endswith("s") else w) in lq for w in ws):
                    return plain
        if must and not s.question_log:
            return q  # step 0: a budget or either-or question goes out as worded
        m = self.MARKER.search(q)
        if m and len([p for p in re.split(r",|\bor\b", m.group(2)) if p.strip()]) >= 3:
            bare = self._bare(key)
            if any(kit.exact_property(bare, r[0]) for r in self._rows(menu)):
                return plain  # as built: also hits a queued three-value question (06 10.2)
        if pending:
            return q
        return plain if self._names_other(q, key, e.get("next_key"), menu) else q

    def _names_other(self, q, key, second, menu):
        s = self.s
        try:
            mine = self._bare(key)
            ws = re.findall(r"[a-z0-9]+", kit.low(q))
            heard = set(ws) | {self._sing(w) for w in ws}
            own = set(kit.property_terms(mine)) | set(re.findall(r"[a-z0-9]+", kit.low(self.plain_question(key))))  # frame words
            names = [p.get("name") or "" for p in s.ledger.products()] or [s.ledger.product_name() or ""]
            for n in names:
                for w in re.findall(r"[a-z0-9]+", kit.low(n)):
                    own.update((w, self._sing(w)))
            def spelled(name):
                core = [w for w in kit.property_terms(name) if len(w) > 2 and not kit.is_link(w)]
                return bool(core) and all(w in heard for w in core) and not all(w in own for w in core)
            sec = self._second(second)
            if sec and not kit.exact_property(self._bare(sec), mine) and spelled(self._bare(sec)):
                return True
            qn = " " + " ".join(ws) + " "
            for name, _, _, line in self._rows(menu):
                if line.endswith(":") or kit.exact_property(name, mine) or kit.is_junk_key(name):
                    continue
                if spelled(name):
                    return True
                cm = self.COUNT_COLON.search(line)
                if not cm:
                    continue  # as built: rows with bracketed notes show no samples (06 10.2)
                for v in line[cm.end():].split("|"):
                    vw = re.findall(r"[a-z0-9]+", kit.low(v))
                    if sum(len(w) for w in vw) >= 3 and not all(w in own or kit.is_link(w) for w in vw) \
                            and (" " + " ".join(vw) + " ") in qn:
                        return True
            return False
        except Exception as e:
            s.link.fault("ask_wording_error", e)
            return False

    # ---------------------------------------------------------------- empty plan (3.14)
    def _empty(self, sets, want, urgent):
        s = self.s
        roomy = s.steps_left() > self.ROOMY
        thin = any(s.intake.detailed_reads(i) < self.THIN_READS for i in s.ledger.slots())
        if roomy and not urgent and (sets < want or thin):
            self._wait = sets
            return False
        s.plan_empty += 1
        if roomy and s.plan_empty == 1:
            self._wait = sets
            return False
        s.asking = False
        return False

    # ---------------------------------------------------------------- reads before asking: own survey (3.5)
    def _survey(self):
        s = self.s
        if self._surveyed or s.multi():
            return False
        try:
            tool = s.role("search")
            if not tool or not s.refs or not s.archive.known:
                return False
            homes = s.judge.home_labels()
            home = {kit.low(h) for h in (homes[0] if homes else ())}
            if home and any(kit.low(r.get("shelf") or "") in home for r in s.archive.digest().values()):
                return False
            self._surveyed = True
            name = str(s.ledger.product_name() or "").strip()
            if not name or s.seeker.already_ran(name) or s.steps_left() <= s.intake.due_margin() + self.SURVEY_PAD:
                return False
            tpl = s.seeker.template()
            if tpl:
                args = dict(tpl[0])
                args[tpl[1]] = name
            else:
                args = s.closer.bare_args(tool, name, None)
            args, faults = ArgFixer.fix(args, s.schema(tool))
            if faults or not isinstance(args, dict):
                return False
            if not any(kit.low(v).strip() == kit.low(name) for _, v in s.seeker.open_text(args)):
                return False
            s.link.note("own_survey", query=name)
            before = s.link.steps
            s.walker.search_call(args, "Searching for the requested product itself first, to learn which fields its"
                                       " own listings carry before any question.")
            return s.link.steps > before
        except Exception as e:
            s.link.fault("own_survey_error", e)
            return False

    # ---------------------------------------------------------------- book, outcome, log (3.15-3.18)
    def _send(self, call, key, text, split, pending, sets, room, multi):
        s, L = self.s, self.s.ledger
        self._last = (sets, s.link.steps)
        rev0, free0 = len(L.revealed or []), self._freed()
        s.asking_label = key
        try:
            pairs = s.engine.dispatch("I will ask the shopper about %s." % key, [call]) or []
        finally:
            s.asking_label = ""
        s.engine.send_replies(pairs)
        if pending and self._head(key) == pending and pending in s.ask_queue:
            s.ask_queue.remove(pending)
        marker = L.meta_question
        L.meta_question = None
        new = [str(x) for x in (L.revealed or [])[rev0:]]
        found = " ".join(new).lower()
        pieces = [p.strip() for p in key.split(";") if p.strip()] or [key]
        label, notes = key, ""
        if marker and self._norm(text) in self._norm(marker):
            label, notes = self._one_at_a_time(pieces, text, split, multi)
        elif ";" in key:
            first, second = key.split(";", 1)[0].strip(), key.split(";", 1)[1].strip()
            if new:
                label = first
                if not self._mentions(found, second):
                    s.ask_queue.append(second)
                    notes = " [%s: not answered, still open]" % second
            elif self._freed() > free0:
                label = first
                drop = (second.lower(), self._bare(second).lower())
                L.freed[:] = [f for f in L.freed if str(f).strip().lower() not in drop]
                self._open(second)
                notes = " [%s: not answered, still open]" % second
            else:
                self._no_pairs = True
                for p in pieces:
                    self._open(p)
                label = " and ".join(pieces) + " [not answered]"
        elif split and not self._mentions(found, split):
            self._open(split)
        try:
            s.warden.disclosure_memo(new)
        except Exception as e:
            s.link.fault("reveal_note_error", e)
        if not marker and not new:
            s.quiet_props.append(self._head(key))
        if pairs:
            body = str(pairs[0][1]).split("[status]", 1)[0].strip()[:self.REPLY_CLIP]
            s.question_log.append('- %s: asked "%s" -> %s%s' % (label, text, body, notes))
        self._drop_drafts(label)
        s.asking = s.msgs_sent < room
        return True

    def _one_at_a_time(self, pieces, text, split, multi):
        s = self.s
        self._no_pairs = True
        lead = pieces[0]
        lk, alone = lead.lower(), len(pieces) == 1
        alt = (s.intake.alt_after_unanswered(lead) or "") if alone else ""
        plain = self.plain_question(lead)
        if alone and multi and not alt:
            self._reasked.add(lk)
            s.spent_props.add(lk)
            what = "not answered, spent; another product is asked next"
        elif alt:
            self._reasked.add(lk)
            s.spent_props.add(lk)
            s.ask_queue.insert(0, alt)
            what = "not answered, its story hint's alternative %s asked next" % alt
        elif lk in self._reasked or text == plain or (alone and s.intake.pauses_text(text, plain)):
            what = self._reask(lead)
        else:
            self._reasked.add(lk)
            s.ask_queue.insert(0, lead)
            what = "not answered, asked again alone next"
        for p in pieces[1:] + ([split] if split else []):
            self._open(p)
        return " and ".join(pieces) + " [not answered] [one at a time]", " [%s: %s]" % (lead, what)

    def _reask(self, lead):
        s = self.s
        lk = lead.lower()
        got = None if lk in self._reasked else s.intake.distinct_reask(lead)
        if not got:
            s.spent_props.add(lk)
            return "not answered, not asked again"
        lab, q = str(got[0]), str(got[1])
        self._reasked.update((lk, lab.lower()))
        s.prepared[lab.lower()] = q
        s.ask_queue.insert(0, lab)
        if self._bare(lab).lower() == self._bare(lead).lower():
            return "not answered, asked again by its values next"
        s.spent_props.add(lk)
        return "not answered, asked again as %s next" % lab

    def _drop_drafts(self, label):
        s = self.s
        if not s.drafts:
            return  # dormant: nothing records drafts (06 3.18)
        names = [self._bare(p) for p in re.split(r";|\band\b", label.replace("[not answered]", "")) if p.strip()]
        sets = [{self._sing(w) for w in re.findall(r"[a-z0-9]+", kit.low(n))} for n in names]
        keep = []
        for d in s.drafts:
            dw = {self._sing(w) for w in re.findall(r"[a-z0-9]+", kit.low(d))}
            if not any(ws and ws <= dw for ws in sets):
                keep.append(d)
        s.drafts[:] = keep



# ------------------------------------------------------------ p09_seeker.py
def _p09_guard(kind, neutral=None):
    """Wrap a public Seeker method: log a caught error and return the neutral value (6.5)."""
    def wrap(fn):
        def run(self, *a, **k):
            try:
                return fn(self, *a, **k)
            except Exception as e:
                self.s.link.fault(kind, e)
                return neutral() if callable(neutral) else neutral
        run.__name__, run.__doc__ = fn.__name__, fn.__doc__
        return run
    return wrap


class Seeker:
    """Code search and paging, page book, word rules, refusal notes, catch-up, shelf sweep, wide step, twins (07)."""
    EDGE = "()[]{}\"'.,;:!?/|"
    TRAP_EDGE = "\"'()[]{},.;!?|"
    PAGEISH = re.compile(r"page|offset|first|start|skip|from|cursor")
    NOTPAGE = re.compile(r"size|per[ _]?page|limit|price|cost")
    CUE = re.compile(r"next|onward|cursor|bookmark|token|continu|more")
    TOTAL = re.compile(r"count|total|matched|found|hits")
    CURSOR = re.compile(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start")
    SHELF = re.compile(r"categor|shelf|department|aisle")
    BOUND = re.compile(r"ceil|max|price|budget|upper|most")
    NOTBOUND = re.compile(r"page|first|offset|skip|count|size|num")
    SELLER = re.compile(r"shop|seller|store|vendor|brand|make|manufactur")
    BRAND = re.compile(r"brand|make|manufactur|vendor")
    PRICE = re.compile(r"price|cost")
    NOTSELL = re.compile(r"note|text|desc|comment|detail|info|message|section")
    TITLEISH = re.compile(r"title|heading|headline|name|label|caption")
    SPENT = "This tool's allowance is used up; make no more calls to it."
    DECLINE_REFUSED = ("The store did not accept the decline, so the errand goes on: by the store's own account some listing "
                       "does fit. Next: act on whatever the refusal states; go back to the closest candidates and check each "
                       "one on its own record (every property, stock and price) against the requirements as they stand now; "
                       "buy the one that passes; decline again only after every candidate has failed.")
    HOLD_A = ("The decline is on hold because the store passed on a message from the shopper just before closing; being on "
              "hold says nothing about any listing. Next: act on that message. Re-check only the listings it touches: those "
              "rejected on a condition the message removes or loosens, or on a price that a higher budget now allows. "
              "Listings rejected for any other reason stay rejected.")
    HOLD_IDS = " Item-check rejections that still stand: %s."
    HOLD_B = " Buy a listing that now passes every condition in force; if none does, send the decline once more."
    ORDER_REFUSED = ("The store refused the order and nothing was sold; keep working on this same cart. Next: make every "
                     "change the refusal names; repair the cart by taking out any line that fails now (a sold-out line, "
                     "for example); check stock and price of any item newly added; then send the order again.")
    ECHO = ("This repeats the message the refused finish already delivered (it was handled there); it has no bearing on "
            "this call.")
    NOTICE = "[shopper message: a change] This reply carries news that must be applied right away: "
    FRESH = ("[fresh plan] The model service is back: the shopper's request was read again, and this plan replaces the "
             "earlier one.")
    LATE = "[read late] These lines came from the shopper earlier, while the reply reader was down. What they say: "
    SWEEP_SAY = ("No property requirement is left, so only price decides: the cheapest item of this category that can be "
                 "bought now wins. Reading another slice of the category.")
    SWEEP_TAIL = (" Plan: the first row is the choice unless something speaks against it; make it the only basket line, "
                  "read its stock and price again, then send the order.")
    WIDE_SAY = ("A requirement was dropped, so listings the earlier searches missed may qualify now: searching '<<Q>>' "
                "without the dropped words.")
    WIDE_OPEN = ("Reading the cheaper rows that the searches without the dropped requirement turned up. Among listings that "
                 "meet every current condition the lowest price wins, so one of these may take over the cart line before "
                 "the basket message.")
    AFTER_DROP = ("[after the drop] Cart line(s): %s. Just opened, each cheaper than the cart line: %s. Judge each one on "
                  "its own record against the current conditions only, ignoring the dropped one. The cheapest one passing "
                  "every current condition replaces the cart line, and this comes before any basket message, so the basket "
                  "reported to the shopper is the one that gets ordered.")
    TRAP_NAME = ("This store demands every query word, and '%s' is part of a property's NAME, not of a wanted value: a "
                 "listing comes back only if it writes that name out somewhere (a field or the title), so listings that "
                 "show just the value are missed.")
    TRAP_DROP = ("'%s' comes only from a requirement the shopper gave up, yet the query still forces every listing to "
                 "contain it, which hides items that are acceptable now.")
    TRAP_END = " Run '%s' as an extra search and read it through all of its pages."

    def __init__(self, s):
        self.s = s
        self.searched = False      # the code has run its own search (Closer's "first go" test)
        self.pending = []          # code next-page calls, replaced after each round
        self._scanned = set()
        self._reread = False
        self._echo = None
        self._match_obs, self._lead_obs, self._bounds = [], [], {}
        self._book = {}
        self._twin_proof = {}
        self._pw = None
        self._ws, self._ws_busy = None, False
        self._sw_tried, self._sw_reads, self._sw_turn, self._sw_last = set(), 0, 0, None
        self._w_mark, self._w_queue, self._w_col = None, [], None

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _sig(x):
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def _props(self, tool=None):
        tool = self.s.role("search") if tool is None else tool
        p = self.s.schema(tool).get("properties") if tool else None
        return p if isinstance(p, dict) else {}

    @staticmethod
    def _typed(p, *kinds):
        t = p.get("type") if isinstance(p, dict) else None
        return any(k in (t if isinstance(t, list) else [t]) for k in kinds)

    def _pageish(self, name):
        n = str(name).lower()
        return bool(self.PAGEISH.search(n)) and not self.NOTPAGE.search(n)

    def _enum(self, props, k):
        return isinstance(props.get(k), dict) and bool(props[k].get("enum"))

    @staticmethod
    def _runs(text):
        return [w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 1]

    @staticmethod
    def _value_of(line):
        line = str(line or "")
        return line.split(":", 1)[1].strip() if ":" in line else line.strip()

    def _two(self, text):
        return tuple(kit.stem(w) for w in kit.words(text) if len(w) >= 2)[:2]

    @staticmethod
    def _shared(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _price_of(self, d):
        for k, v in d.items():
            if kit.is_num(v) and self.PRICE.search(str(k).lower()):
                return v
        return None

    def _plain_words(self, text):
        out = []
        for t in str(text).lower().split():
            t = t.strip(self.EDGE)
            if not re.search(r"[a-z0-9]", t) or t in out:
                continue
            if not re.fullmatch(r"[a-z0-9]{2,}", t):
                return []
            out.append(t)
        return out

    @staticmethod
    def _ascending(ids):
        def nat(x):
            return [int(p) if i % 2 else p for i, p in enumerate(re.split(r"(\d+)", x))]
        pairs = list(zip(ids, ids[1:]))
        if all(a <= b for a, b in pairs):
            return True
        try:
            return all(nat(a) <= nat(b) for a, b in pairs)
        except Exception:
            return False

    def _long_texts(self, v, d=0, out=None):
        out = [] if out is None else out
        if isinstance(v, str):
            if len(v.split()) >= 4:
                out.append(v)
        elif d < 6 and isinstance(v, dict):
            for x in v.values():
                self._long_texts(x, d + 1, out)
        elif d < 6 and isinstance(v, list):
            for x in v[:20]:
                self._long_texts(x, d + 1, out)
        return out

    @staticmethod
    def _norm(t):
        return re.sub(r"\s+", " ", str(t).lower()).strip()

    @staticmethod
    def _text_of(msg):
        c = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c if isinstance(b, dict) and isinstance(b.get("text"), str))
        return c if isinstance(c, str) else ""

    def _code_replies(self, n):
        # as built (07 3.1.3, flag F1): only the newest store step is read back, so a burst of several calls (each its
        # own step) yields just the last call's reply; callers pair it with the first call.
        try:
            return list(self.s.closer.newest_replies(max(1, n)) or [])
        except Exception as e:
            self.s.link.fault("code_replies_error", e)
            return []

    def _burst(self, word, calls):
        s = self.s
        props = [s.engine.call(t, a, "%s_%d_%d" % (word, i + 1, s.link.steps)) for i, (t, a) in enumerate(calls)]
        s.engine.send_booked(props)
        return self._code_replies(len(calls))

    # ------------------------------------------------------------ 3.2 code search
    @_p09_guard("code_search_error", False)
    def rule_query(self, n, more=False):
        """The code's own search rounds; True when it sent."""
        s = self.s
        tool = s.role("search")
        if not tool:
            return False
        n = max(1, int(n or 1))
        res, cheap = 2 + 2 * n, s.ledger.priority() == "cheapest"
        deep = Tune.Search.DEEP_SOLO if n == 1 else Tune.Search.DEEP_FEW if n <= Tune.Search.FEW_MAX else 0
        if more:
            calls, lead, rounds = self.pending[:max(0, s.steps_left() - res - 2 * n)], list(self.pending), 1
        else:
            per = [list(s.closer.search_calls(sl) or []) for sl in s.ledger.slots()]
            seen, order = set(), []
            for d in range(max([len(p) for p in per] + [0])):
                for p in per:
                    if d < len(p) and isinstance(p[d], dict) and self._sig(p[d]) not in seen:
                        seen.add(self._sig(p[d]))
                        order.append(p[d])
            room = s.steps_left() - res
            calls = order[:room] if room > 0 else []
            lead = [p[0] for p in per if p]
            rounds = 1 + max(3 + (2 if cheap else 0), deep)
        if not calls:
            return False
        self.searched = True
        props = self._props(tool)
        free = [k for k, p in props.items() if isinstance(p, dict) and self._typed(p, "string") and not p.get("enum")]
        rnd = 0
        while calls:
            pairs = list(zip(calls, self._burst("search", [(tool, c) for c in calls])))
            if s.word_mode != "any":
                for c, r in pairs:
                    txt = " ".join(c[k] for k in free if isinstance(c.get(k), str)).strip()
                    if txt:
                        self._tag(r, kit.kind_terms(txt))
            self.pending = [x for x in (self.next_page_call(c, r) for c, r in pairs if c in lead) if x]
            rnd += 1
            if rnd >= rounds:
                break
            plan = s.closer.picks()
            if plan and not cheap and all(s.code_fit.get(l) for l, _ in plan) and rnd > deep:
                break
            calls = self.pending[:max(0, s.steps_left() - res - 3 * n)]
            lead = list(self.pending)
        return True

    def _tag(self, reply, words):
        s = self.s
        if not words:
            return
        def see(d):
            i = kit.identity(d)
            if i in s.archive.known:
                s.code_hits.setdefault(i, set()).update(words)
        kit.visit(reply, see)

    # ------------------------------------------------------------ 3.3 next-page call
    @_p09_guard("next_call_error")
    def next_page_call(self, args, obs):
        """Follow-up call for the next page of a reply, or None."""
        s = self.s
        pg = self.paging()
        if not pg or not isinstance(args, dict):
            return None
        page = []
        def see(d):
            if not page and any(isinstance(v, list) and v and all(isinstance(x, dict) for x in v) for v in d.values()):
                page.append(d)
        kit.visit(obs, see, top=3)
        if not page:
            return None
        sizes, cues, total = [], [], None
        for k, v in page[0].items():
            nm = str(k).lower()
            if isinstance(v, list):
                sizes.append(len(v))
            if self.CUE.search(nm):
                cues.append(v)
            if total is None and kit.is_num(v) and self.TOTAL.search(nm):
                total = v
        def blank(v):
            return v is None or v is False or (isinstance(v, str) and not v.strip()) or (kit.is_num(v) and v == 0)
        if cues and all(blank(c) for c in cues):
            return None
        nxt, name = dict(args), pg["name"]
        if pg["mode"] == "token":
            tok = next((c for c in cues if isinstance(c, str) and c.strip()), None)
            if tok is None:
                return None
            nxt[name] = tok
        else:
            cur = args.get(name)
            cur = cur if kit.is_num(cur) else pg["start"]
            size = max(sizes) if sizes else 0
            ahead = cur + 1 if pg["mode"] == "page" else cur + size
            if pg["top"] is not None and ahead > pg["top"]:
                return None
            if not cues:
                cov = (cur - pg["start"] + 1) * size if pg["mode"] == "page" else cur + size
                if total is None or cov >= total:
                    return None
            nxt[name] = int(ahead) if float(ahead).is_integer() else ahead
        val, faults = ArgFixer.fix(nxt, s.schema(s.role("search")))
        return None if faults or not isinstance(val, dict) else val

    # ------------------------------------------------------------ 3.4 code read
    def rule_open(self, plan, barred, preferred):
        """Open the plan's unread records and re-pick; returns the latest plan."""
        s = self.s
        plan = list(plan or [])
        try:
            for _ in range(Tune.Search.READ_ROUNDS):
                unread = [l for l, _ in plan if l not in s.archive.read]
                bs, det = s.mender.batch_spec(), s.role("details")
                joint = bool(bs) and len(unread) >= 2
                res = 3 + 2 * len(plan) + (1 if joint else len(unread))
                if not unread or not (bs or det) or res > s.steps_left() or s.link.time_left() <= 8:
                    break
                if joint:
                    calls = [(bs[0], {bs[1]: unread[:max(1, int(bs[2] or len(unread)))]})]
                else:
                    calls = [c for c in (s.judge.detail_request(l) for l in unread) if c]
                if not calls:
                    break
                self._burst("read", calls)
                new = list(s.closer.picks(barred or (), preferred or ()) or [])
                if new == plan:
                    break
                plan = new
        except Exception as e:
            s.link.fault("code_read_error", e)
        return plan

    # ------------------------------------------------------------ 3.5 code put
    def rule_cart(self, plan):
        """Place a plan in the cart: None (could not), [] (done or nothing to do), or sorted failed ids."""
        s = self.s
        try:
            B, cart = s.basket, s.role("cart")
            if not cart:
                return None
            plan = [(l, u) for l, u in plan or []]
            wanted = [l for l, _ in plan]
            whole = B.list_spec() is not None
            others = [l for l in B.units if l not in wanted]
            duties = [d for d in (s.ledger.duties or []) if isinstance(d, dict)]
            if others and any(d.get("before_change") is not False for d in duties):
                asked = s.warden.swap_consent(others, wanted)
                if not asked and any(d.get("before_change") is True for d in duties):
                    return None
            sets = plan if whole else [(l, u) for l, u in plan if abs((B.held(l) or 0) - (u or 0)) > 1e-9]
            drops = [] if whole else s.closer.drop_calls(others)
            if drops is None:
                return None
            puts = s.closer.put_calls(sets) if sets else []
            if sets and not puts:
                return None
            calls = list(drops) + list(puts)
            if not calls:
                return []
            if s.steps_left() < len(calls) + 1:
                return None
            reps = self._burst("cart", [(cart, a) for a in calls])
            bad, view = [], None
            for a, r in zip(calls, reps):
                shown = B.shown_units(r)
                if shown:
                    view = shown
                if s.closer.has_fault(r):
                    txt = kit.render(r)
                    got = ([l for l in wanted if l in txt] or [l for l, _ in B.lines_of(a) if l in wanted]
                           or (list(wanted) if whole else []))
                    bad += [l for l in got if l not in bad]
            if len(reps) < len(calls):
                return None  # as built (F1): any placement of 2+ calls ends here although it was sent and booked
            vt = s.role("cart_view")
            if view is None and not bad and vt and s.steps_left() >= 3:
                va, f = ArgFixer.fix({}, s.schema(vt))
                rr = self._burst("look", [(vt, va if not f and isinstance(va, dict) else s.closer.bare_args(vt, "", None))])
                if rr:
                    shown = B.shown_units(rr[0])
                    if shown:
                        view = shown
                    else:
                        bad += [l for l in wanted if l not in B.units and l not in bad]
            if view is not None:
                bad += [l for l in wanted if l not in view and l not in bad]
            for l in bad:
                B.units.pop(l, None)
            return sorted(bad)
        except Exception as e:
            s.link.fault("code_put_error", e)
            return None

    # ------------------------------------------------------------ 3.6 stock recheck
    def recheck_stock(self, ids):
        """One lookup per held id; returns (looked, ids whose merged stock is out)."""
        s = self.s
        try:
            ids = list(dict.fromkeys(ids or []))
            tool = s.judge.supply_tool() or s.role("view") or s.role("details")
            if not ids or not tool or len(ids) + 2 > s.steps_left() or s.link.time_left() < 8:
                return (False, set())
            calls = [(tool, a) for a in (s.judge.query_call(l, tool) for l in ids) if a]
            if not calls:
                return (False, set())
            self._burst("stock", calls)
            last = {}
            def see(d):
                st = kit.stock_of(d)
                if st is not None:
                    last[kit.identity(d)] = st
            for o in s.archive.obs:
                kit.visit(o, see)
            return (True, {l for l in ids if last.get(l) is False})
        except Exception as e:
            s.link.fault("recheck_error", e)
            return (False, set())

    # ------------------------------------------------------------ 3.7 late budget news
    @_p09_guard("late_budget_error")
    def late_budget(self, lines):
        """Scan fresh lines and the parked backlog (each line once) for a budget figure."""
        s = self.s
        pool = [l for l in (lines or []) if isinstance(l, str)]
        for b in s.ledger.parked or []:
            got = b[0] if isinstance(b, (list, tuple)) and b else []
            pool += [l for l in (got if isinstance(got, list) else [got]) if isinstance(l, str)]
        for line in pool:
            if line in self._scanned:
                continue
            self._scanned.add(line)
            r = s.intake.budget_from_text(line)
            if r and kit.is_num(r[0]) and r[0] > 0:
                s.ledger.budget = r[0]
                s.link.note("code_news", budget=r[0])

    # ------------------------------------------------------------ 3.10 catch-up
    def make_up(self):
        """Late re-read of a code-made reading and late hearing of parked lines; tells the actor."""
        s = self.s
        try:
            if s.link.mute or s.link.time_left() < Tune.Clock.CLOSING:
                return
            notes, at = [], s.read_fallback_at
            if (not self._reread and at is not None and s.ledger.by_code() and s.link.last_good > at
                    and s.link.time_left() > Tune.Clock.REREAD_MIN):
                self._reread = True
                models, cap = LINEUPS["rereader"]
                msg = s.link.chat([{"role": "user", "content": s.intake.reader_blocks()}], models, cap)
                plan = kit.json_from(self._text_of(msg)) if msg else None
                ok = isinstance(plan, dict) and isinstance(plan.get("tool_roles"), dict)
                s.link.note("re_read", usable=ok)
                if ok:
                    s.intake.apply_reading(plan, True)
                    over = s.ledger.notes_text()
                    notes.append(self.FRESH + ("\n" + over if over else ""))
            parked = [b for b in (s.ledger.parked or []) if isinstance(b, (list, tuple)) and b]
            if parked:
                times = [b[2] for b in parked if len(b) > 2 and kit.is_num(b[2])]
                if s.link.last_good > (max(times) if times else 0) and s.link.time_left() >= Tune.Clock.CLOSING:
                    s.ledger.parked = []
                    for b in parked[-3:]:
                        lines = b[0] if isinstance(b[0], list) else [b[0]]
                        note = s.ledger.hear(list(lines), b[1] if len(b) > 1 else None)
                        if note:
                            notes.append(self.LATE + note)
            if notes:
                s.engine.append_note("\n".join(notes), status=True)
        except Exception as e:
            s.link.fault("catch_up_error", e)

    # ------------------------------------------------------------ 3.11 refusal test and notes
    def is_refused(self, name, obs, finished):
        """Does this reply refuse a finish (or carry a refusal)?"""
        try:
            if isinstance(obs, dict) and "refusal" in obs:
                return True
            if not name or name != self.s.role("order") or finished:
                return False
            err = obs.get("error") if isinstance(obs, dict) else None
            return err not in ("invalid_args", "unknown_tool")
        except Exception as e:
            self.s.link.fault("refusal_test_error", e)
            return False

    def refusal_note(self, name, obs, refused):
        """Note appended to a finish or notice reply ("" when none)."""
        s = self.s
        try:
            out, o = [], obs if isinstance(obs, dict) else {}
            if o.get("error") == "budget_exceeded":
                if (name and name == s.role("message")) or s.msg_cap is not None:
                    s.msg_cap = s.msgs_sent if s.msg_cap is None else min(s.msg_cap, s.msgs_sent)
                out.append(self.SPENT)
            if refused:
                if name and name == s.role("decline") and s.last_finish_kind == "decline":
                    s.audit_done = s.decline_vetted = False
                    s.event_seen = True
                    out.append(self._hold_note() if self._long_texts(obs) else self.DECLINE_REFUSED)
                else:
                    out.append(self._order_note(obs))
            elif o and len(o) >= 2 and not (name and name == s.role("message")):
                out.append(self._notice(name, obs))
            return " ".join(x for x in out if x)
        except Exception as e:
            s.link.fault("refusal_note_error", e)
            return ""

    def _hold_note(self):
        s = self.s
        v, bad = str(s.ledger.version), set()
        for k, r in s.rulings.items():
            p = str(k).split("|")
            if r and len(p) >= 2 and p[1] == v and not s.passed_now(p[0]):
                bad.add(p[0])
        ids = sorted(bad)[:6]
        return self.HOLD_A + (self.HOLD_IDS % ", ".join(ids) if ids else "") + self.HOLD_B

    def _order_note(self, obs):
        s = self.s
        s.refusals += 1
        s.order_vetted = False
        s.event_seen = True
        hint = ""
        try:
            hint = s.judge.reread_refused(obs) or ""
        except Exception as e:
            s.link.fault("refusal_reread_error", e)
        s.refused_lines |= set(s.basket.units)
        self._echo = [self._norm(t) for t in self._long_texts(obs)]
        return self.ORDER_REFUSED + (" " + hint if hint else "")

    def _notice(self, name, obs):
        s = self.s
        lines = [str(l) for l in (s.warden.buyer_lines(name, obs, False) or [])]
        if not lines:
            return ""
        echo, self._echo = self._echo, None
        if echo and all(any(self._norm(l) in e for e in echo) for l in lines):
            return self.ECHO
        s.event_seen = True
        return self.NOTICE + " | ".join(lines[:4])

    # ------------------------------------------------------------ 3.13 wanted shelves
    def wanted_shelves(self):
        """The errand's wanted shelves (memoised per archive size, reads, needs version, references)."""
        s = self.s
        if self._ws_busy:
            return set()
        try:
            key = (len(s.archive.obs), len(s.archive.read), s.ledger.version, len(s.refs), len(s.ledger.products() or []))
            if self._ws is None or self._ws[0] != key:
                self._ws_busy = True
                try:
                    self._ws = (key, self._wanted())
                finally:
                    self._ws_busy = False
            return set(self._ws[1])
        except Exception as e:
            s.link.fault("shelves_error", e)
            return set()

    def _wanted(self):
        s = self.s
        read = s.archive.read
        try:
            foreign = set(s.tracer.foreign_ids() or ())
        except Exception:
            foreign = set()
        tally, done = {}, set()
        def see(d):
            lid = next((v for k, v in d.items() if isinstance(v, str) and "id" in str(k).lower() and v in read), None)
            if not lid or lid in done or lid in foreign:
                return
            sh = kit.shelf_field(d)
            if sh:
                done.add(lid)
                tally[sh] = tally.get(sh, 0) + 1
        for o in s.archive.obs:
            kit.visit(o, see)
        top = max(tally.values()) if tally else 0
        common = {k for k, v in tally.items() if v == top}
        if not s.multi() or not tally:
            return common
        try:
            homes = [set(h or ()) for h in (s.judge.home_labels() or [])]
            chosen = set()
            for h in homes:
                inn = [l for l in h if l in tally]
                if inn:
                    m = max(tally[l] for l in inn)
                    chosen |= {l for l in inn if tally[l] == m}
                else:
                    chosen |= h
            if any(not h for h in homes):
                want = len(s.ledger.products() or [])
                for l in sorted(tally, key=lambda x: -tally[x]):
                    if len(chosen) >= want:
                        break
                    chosen.add(l)
            return chosen or common
        except Exception as e:
            s.link.fault("shelves_error", e)
            return common

    # ------------------------------------------------------------ 3.14 arguments and paging
    @_p09_guard("paging_error")
    def paging(self):
        """{"name", "mode", "start", "top"} of the search tool's paging field, or None."""
        props = self._props()
        for k, p in props.items():
            if isinstance(p, dict) and self._typed(p, "integer", "number") and self._pageish(k):
                n = k.lower()
                mode = "page" if "page" in n and not re.search(r"offset|skip|first|start|from", n) else "offset"
                st = next((p.get(x) for x in ("minimum", "default") if kit.is_num(p.get(x))), None)
                st = (1 if mode == "page" else 0) if st is None else st
                return {"name": k, "mode": mode, "start": st, "top": p.get("maximum") if kit.is_num(p.get("maximum")) else None}
        cur = self.cursor_param(props)
        return {"name": cur, "mode": "token", "start": "", "top": None} if cur else None

    @_p09_guard("cursor_error")
    def cursor_param(self, props=None):
        """First string, non-enum parameter named like a cursor."""
        props = self._props() if props is None else props
        for k, p in (props or {}).items():
            if isinstance(p, dict) and self._typed(p, "string") and not p.get("enum") and self.CURSOR.search(str(k).lower()):
                return k
        return None

    @_p09_guard("token_error")
    def page_cursor(self, obs):
        """First top-level non-blank string field of a reply named like a continuation token."""
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, str) and v.strip() and self.CUE.search(str(k).lower()):
                    return v.strip()
        return None

    @_p09_guard("parts_error", lambda: {"texts": {}, "modes": (), "first": True, "paging": []})
    def search_parts(self, args):
        """texts (dict field -> plain words), modes (sorted (name, value) tuple), first (bool), paging (names)."""
        props = self._props()
        cur = self.cursor_param(props)
        texts, modes, paging, first = {}, [], [], True
        for k, v in (args or {}).items():
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            if isinstance(v, str):
                if k == cur:
                    paging.append(k)
                    first = first and not v.strip()
                elif p.get("enum"):
                    modes.append((k, v))
                else:
                    w = self._plain_words(v)
                    if w:
                        texts[k] = w
            elif kit.is_num(v) and self._pageish(k):
                paging.append(k)
                fl = next((p.get(x) for x in ("minimum", "default") if kit.is_num(p.get(x))), None)
                fl = (1 if "page" in k.lower() else 0) if fl is None else fl
                first = first and v <= fl
        return {"texts": texts, "modes": tuple(sorted(modes)), "first": first, "paging": paging}

    @_p09_guard("free_text_error", list)
    def open_text(self, args):
        """(name, text) of every string argument except the cursor and enum parameters."""
        props = self._props()
        cur = self.cursor_param(props)
        return [(k, v) for k, v in (args or {}).items() if isinstance(v, str) and k != cur and not self._enum(props, k)]

    # ------------------------------------------------------------ 3.15 word matching
    @_p09_guard("words_error")
    def learn_words(self, args, obs, rows, total):
        """Learn the match mode ("all"/"any") and the lead length N from a first-page search."""
        s = self.s
        sp = self.search_parts(args)
        if not sp["first"] or not sp["texts"]:
            return
        rows = [r for r in rows or [] if isinstance(r, dict)]
        tot = len(rows)
        if isinstance(obs, dict):  # flag F7: any top-level integer may stand for the total
            tot = max([tot] + [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)])
        elif kit.is_num(total):
            tot = max(tot, total)
        ids = tuple(kit.identity(r) for r in rows)
        varied, modes = len(set(ids)) >= 3, sp["modes"]
        frames = {f: self._frame(args, f, modes, sp["paging"]) for f in sp["texts"]}
        for f, w in sp["texts"].items():
            narrower, n = False, s.lead_words or None
            for ofr, ow, ot in self._match_obs:
                if ofr != frames[f] or ot == tot:
                    continue
                if len(w) != len(ow) and (w[:len(ow)] == ow or ow[:len(w)] == w):
                    lt, st = (tot, ot) if len(w) > len(ow) else (ot, tot)
                    s.word_mode = "any" if lt > st else "all"
                    continue
                a, b = {kit.stem(x) for x in w[:n]}, {kit.stem(x) for x in ow[:n]}
                if (a > b and tot < ot) or (b > a and ot < tot):
                    narrower = True
            if narrower and not s.word_mode:
                s.word_mode = "all"
            self._match_obs.append((frames[f], w, tot))
        for f, w in sp["texts"].items():
            b = self._bounds.setdefault((f, modes), [-1, set()])
            for ofr, ow, ot, oids in self._lead_obs:
                if ofr != frames[f] or ow == w:
                    continue
                sh = self._shared(w, ow)
                if (ot, oids) != (tot, ids):
                    b[0] = max(b[0], sh)
                elif sh > 0 and varied and sh == min(len(w), len(ow)):
                    b[1].add(sh)
            self._lead_obs.append((frames[f], w, tot, ids))
        ready = sorted((b[0] + 1, k) for k, b in self._bounds.items() if b[0] + 1 in b[1])
        s.lead_words, s.lead_frame = (ready[0][0], ready[0][1]) if ready else (0, None)

    def _frame(self, args, f, modes, paging):
        rest = tuple(sorted((k, v if isinstance(v, str) else self._sig(v)) for k, v in args.items() if k != f and k not in paging))
        return (f, modes, rest)

    # ------------------------------------------------------------ 3.16 page book
    @_p09_guard("book_error", "")
    def log_page(self, args, rows, total, obs):
        """Record a search page in the book; returns its key ("" when not booked)."""
        pg = self.paging()
        rows = [r for r in rows or [] if isinstance(r, dict)]
        args = args if isinstance(args, dict) else {}
        if not pg or not rows:
            return ""
        name, mode = pg["name"], pg["mode"]
        v = args.get(name, pg["start"])
        if mode == "token":
            spot = v.strip() if isinstance(v, str) else ""
        elif kit.is_num(v):
            spot = v
        else:
            return ""
        base = {k: x for k, x in args.items() if k != name}
        key = self._sig(base)
        e = self._book.get(key)
        if e is None:
            e = self._book[key] = {"args": base, "text": " ".join(t for _, t in self.open_text(base)).lower().strip(),
                                   "spots": set(), "size": 0, "total": 0, "shelves": set(), "row_shelf": {},
                                   "first_ids": None, "tokens": {}, "id_order": None, "rows": None, "full": 0}
        e["spots"].add(spot)
        if mode == "token":
            e["tokens"][spot] = self.page_cursor(obs)
        n = len(rows)
        tot = total if kit.is_num(total) else n
        e["size"], e["total"] = max(e["size"], n), max(e["total"], tot)
        ids = [kit.identity(r) for r in rows]
        for r, i in zip(rows, ids):
            sh = kit.shelf_field(r)
            if sh:
                e["shelves"].add(sh)
            e["row_shelf"].setdefault(kit.row_id(r) or i, sh)
        if spot == pg["start"]:
            e["first_ids"] = tuple(ids)
            if tot <= n:
                e["rows"] = list(rows)
        if tot > n:
            e["full"] = max(e["full"], n)
        if n >= min(5, max(4, e["full"] or 5)) and e["id_order"] is not False:
            e["id_order"] = self._ascending(ids)
        return key

    def book(self):
        """The search book: key -> entry (registry 1.4.3)."""
        return self._book

    @_p09_guard("book_error", "")
    def book_key(self, args):
        pg = self.paging()
        return self._sig({k: v for k, v in (args or {}).items() if not pg or k != pg["name"]})

    @_p09_guard("page_error")
    def following_page(self, key):
        """Next unread page spot of a book entry, or None."""
        pg, e = self.paging(), self._book.get(key)
        if not pg or not e:
            return None
        if pg["mode"] == "token":
            return next((t for t in e["tokens"].values() if t and t not in e["spots"]), None)
        floor = min([pg["start"]] + [x for x in e["spots"] if kit.is_num(x)])
        size = max(1, e["size"])
        stride = 1 if pg["mode"] == "page" else size
        for n in range(1, 400):
            spot = floor + n * stride
            if n * size >= e["total"] or (pg["top"] is not None and spot > pg["top"]):
                break
            if spot not in e["spots"]:
                return spot
        return None

    def page_call(self, key, spot):
        """The entry's call at a page spot."""
        e = self._book.get(key)
        a = dict(e["args"]) if e else {}
        try:
            pg = self.paging()
            if pg:
                a[pg["name"]] = spot
        except Exception as ex:
            self.s.link.fault("page_error", ex)
        return a

    @_p09_guard("template_error")
    def template(self):
        """(args without the text argument, text argument name) of the newest single-text walk, or None."""
        for e in reversed(list(self._book.values())):
            a = e["args"]
            if self.s.pricer.is_cap_listing(a):
                continue
            ft = self.open_text(a)
            if len(ft) == 1 and ft[0][1].strip():
                return ({k: v for k, v in a.items() if k != ft[0][0]}, ft[0][0])
        return None

    @_p09_guard("ran_error", False)
    def already_ran(self, text):
        """An equivalent query (same sorted stems of its first N or 3 words) already ran."""
        w = self.s.lead_words or 3
        key = sorted(kit.stem(x) for x in kit.words(text)[:w])
        return any(sorted(kit.stem(x) for x in kit.words(e["text"])[:w]) == key for e in self._book.values())

    # ------------------------------------------------------------ 3.17 shelf rows and queries
    @_p09_guard("buyable_error", dict)
    def buyable_rows(self):
        """lid -> (price, title, brand) of wanted-shelf rows in stock, not refused, within budget."""
        s = self.s
        want = self.wanted_shelves()
        if not want:
            return {}
        rows = {}
        def see(d):
            rid = kit.row_id(d)
            if not rid or rid in s.refused_lines:
                return
            sh = kit.shelf_field(d)
            if not sh or sh not in want:
                return
            if kit.out_of_stock(d):
                rows.pop(rid, None)
                return
            p = self._price_of(d)
            if p is not None:
                brand = " ".join(v.strip() for k, v in d.items() if isinstance(v, str) and v.strip() and self.BRAND.search(str(k).lower()))
                rows[rid] = (p, kit.row_title(d), brand)
        for o in s.archive.obs:
            kit.visit(o, see)
        budget, qty = s.ledger.budget, 1 if s.multi() else s.ledger.quantity()
        out = {}
        for rid, (p, t, b) in rows.items():
            q = s.archive.price(rid)
            q = q if kit.is_num(q) else p
            if kit.is_num(budget) and q * qty > budget + 1e-9:
                continue
            out[rid] = (q, t, b)
        return out

    @_p09_guard("labels_error", list)
    def label_searches(self):
        """Search texts made from the wanted shelf labels."""
        out = []
        for lab in sorted(self.wanted_shelves()):
            runs, cur = [], []
            for w in str(lab).split():
                w = w.strip(self.EDGE).lower()
                if w and not kit.is_link(w) and re.fullmatch(r"[a-z0-9][a-z0-9'-]*", w):
                    cur.append(w)
                elif cur:
                    runs.append(cur)
                    cur = []
            if cur:
                runs.append(cur)
            flat, tails = [w for r in runs for w in r], [r[-1] for r in runs]
            emit = [kit.stem(t) for t in tails] + ([" ".join(flat[:2])] if len(flat) >= 2 else []) + tails
            for q in emit + [w for w in flat if w not in tails]:
                if q not in out:
                    out.append(q)
        return out

    @_p09_guard("kinds_error", list)
    def kind_searches(self, width):
        """Recurring title words (width 1) or word pairs (width 2) of buyable rows, cheapest first."""
        width = max(1, int(width or 1))
        stems = {kit.stem(w) for l in self.wanted_shelves() for w in kit.words(l)}
        titles = {}
        for p, t, b in self.buyable_rows().values():
            t = str(t or "").lower().strip()
            p = p if kit.is_num(p) else float("inf")
            if t in titles:
                titles[t][0] = min(titles[t][0], p)
            elif t:
                titles[t] = [p, b]
        tally = {}
        for t, (p, b) in titles.items():
            bw, ws, seen = set(re.findall(r"[a-z]+", str(b or "").lower())), re.findall(r"[a-z]+", t), set()
            for i in range(len(ws) - width + 1):
                win = ws[i:i + width]
                if not all(len(w) >= 3 and not kit.is_link(w) and w not in bw for w in win):
                    continue
                g = tuple(kit.stem(w) for w in win)
                if all(x in stems for x in g) or g in seen:
                    continue
                seen.add(g)
                e = tally.setdefault(g, [0, p, " ".join(win)])
                e[0], e[1] = e[0] + 1, min(e[1], p)
        return [x[2] for x in sorted((v[1], -v[0], v[2]) for v in tally.values() if v[0] >= 2)]

    # ------------------------------------------------------------ 3.18 shelf sweep (move 4)
    def _sweep_wanted(self):
        s = self.s
        return (s.ledger.priority() == "cheapest" and bool(s.ledger.dropped) and not s.ledger.in_force()
                and bool(s.role("search")) and not s.link.done and not s.link.broke)

    def _sweep_room(self):
        s = self.s
        res = 4 if s.basket.units else 3
        if not s.orders_tried:
            res = max(res + 3, 7)
        return (self._sw_reads < Tune.Search.SWEEP_READS and s.steps_left() > res
                and s.link.time_left() >= Tune.Clock.CLOSING + 40)

    def _shelf_tool(self):
        """(tool, cursor, shelf param, shelf value, bound param) of a role-less listing tool, or None."""
        s = self.s
        if s.pricer.cap_profile():
            return None
        label = s.pricer.bare_label(self.wanted_shelves())
        if not label:
            return None
        roled = {v for v in (s.roles or {}).values() if isinstance(v, str)}
        for t in s.tools:
            name = t["function"].get("name")
            if not isinstance(name, str) or name in roled:
                continue
            props = self._props(name)
            cur = self.cursor_param(props)
            if not cur:
                continue
            sp = sv = bp = None
            for k, p in props.items():
                if k == cur or not isinstance(p, dict):
                    continue
                n = str(k).lower()
                if sp is None and self.SHELF.search(n):
                    if p.get("enum"):
                        v = next((c for c in p["enum"] if isinstance(c, str) and c.lower() == label.lower()), None)
                        if v is not None:
                            sp, sv = k, v
                    elif self._typed(p, "string"):
                        sp, sv = k, label
                elif (bp is None and not p.get("enum") and self._typed(p, "integer", "number")
                      and (self.BOUND.search(n) or n.endswith("cap")) and not self.NOTBOUND.search(n)):
                    bp = k
            if sp and bp:
                return (name, cur, sp, sv, bp)
        return None

    def _ceiling(self, tried):
        s = self.s
        st = self._shelf_tool()
        if st:
            rows = self.buyable_rows()
            if s.multi() or not rows:
                return None
            name, cur, sp, sv, bp = st
            if self._sw_last is not None and self._sw_last[1] is None:
                return "stop"
            bound = s.pricer.below(min(r[0] for r in rows.values()))
            if bound is None or bound <= 0:
                return "stop"
            call = {sp: sv, bp: bound}
            if self._sig(call) not in tried:
                return (call, True)
            tok = self._sw_last[1] if self._sw_last else None
            if not tok:
                return "stop"
            nxt = dict(self._sw_last[0] if self._sw_last else call)
            nxt[cur] = tok
            return "stop" if self._sig(nxt) in tried else (nxt, True)
        call, ended = s.pricer.cap_scan(tried)
        if call:
            return (call, True)
        return "stop" if ended else None

    def _sweep_template(self, want):
        s = self.s
        cands = [e for e in self._book.values() if e["text"]]
        e = (next((x for x in reversed(cands) if x["id_order"] is False), None)
             or next((x for x in reversed(cands) if x["id_order"] is None), None))
        if not e:
            return None
        props = self._props()
        free = [k for k, v in e["args"].items() if isinstance(v, str) and not self._enum(props, k) and v.strip().lower() not in want]
        if not free:
            return None
        lf = s.lead_frame[0] if s.lead_frame else None
        f = lf if lf in free else max(free, key=lambda k: len(e["args"][k].split()))
        return ({k: v for k, v in e["args"].items() if k not in free}, f)

    def _sweep_next(self, turn, tried):
        """(call, is listing) for the next sweep read, "stop", or None."""
        pg, want, book = self.paging(), self.wanted_shelves(), self._book
        if not pg or not want or not book:
            return None
        lst, best = None, None
        for k, e in book.items():
            if e["id_order"] is True and e["shelves"] and e["shelves"] <= want and self.following_page(k) is not None:
                if best is None or e["total"] > best:
                    lst, best = k, e["total"]
        tpl = self._sweep_template(want)
        done = {self._two(e["text"]) for e in book.values() if pg["start"] in e["spots"] and e["id_order"] is not True}
        labels = self.label_searches()
        heads = {kit.stem(q) for q in labels if " " not in q}
        r = self._ceiling(tried)
        if r:
            return r
        if turn % 2 == 1:
            for k, e in book.items():
                ft = self.open_text(e["args"])
                if len(e["spots"]) == 1 and ft and kit.stem(ft[0][1].lower().strip()) in heads and e["shelves"] & want:
                    sp = self.following_page(k)
                    c = self.page_call(k, sp) if sp is not None else None
                    if c and self._sig(c) not in tried:
                        return (c, False)
        if lst:
            c = self.page_call(lst, self.following_page(lst))
            if self._sig(c) not in tried:
                return (c, True)
        if tpl:
            fams = [self.kind_searches(2)[:6], self.kind_searches(1)[:6], labels]
            i = turn % 3
            for fam in fams[i:] + fams[:i]:
                for q in fam:
                    sg = self._two(q)
                    if not sg or sg in done:
                        continue
                    c = dict(tpl[0])
                    c[tpl[1]] = q
                    if self._sig(c) not in tried:
                        return (c, False)
        return None

    @_p09_guard("sweep_error", False)
    def scan_turn(self):
        """Move 4: read another slice of the category when only price is left to decide."""
        s = self.s
        if not self._sweep_wanted() or not self._sweep_room():
            return False
        r = self._sweep_next(self._sw_turn, self._sw_tried)
        if not r or r == "stop":
            return False
        call, listing = r
        self._sw_reads += 1
        if not listing:
            self._sw_turn += 1
        self._sw_tried.add(self._sig(call))
        st = self._shelf_tool()
        tool = st[0] if st and st[4] in call else s.role("search")
        mark = len(s.archive.obs)
        s.covering += 1
        try:
            pairs = list(s.engine.dispatch(self.SWEEP_SAY, [s.engine.call(tool, call, "sweep_%d_%d" % (self._sw_reads, s.link.steps))]) or [])
        finally:
            s.covering -= 1
        if listing:
            tok = None
            for o in s.archive.obs[mark:]:
                for x in [o] + (list(o.values()) if isinstance(o, dict) else []):
                    tok = self.page_cursor(x) or tok
            self._sw_last = (call, tok)
        nxt = self._sweep_next(self._sw_turn, set(self._sw_tried)) if self._sweep_room() else None
        if (not nxt or nxt == "stop") and pairs:
            cid, txt = pairs[-1]
            pairs[-1] = (cid, str(txt) + "\n" + self._sweep_summary())
        s.engine.send_replies(pairs, status=True)
        return True

    def _sweep_summary(self):
        rows = sorted((p, lid, str(t or "")) for lid, (p, t, b) in self.buyable_rows().items())[:6]
        cells = "; ".join("%s '%s' %s" % (lid, t[:80], p) for p, lid, t in rows) or "none found"
        return (Tune.Marks.SWEEP_DONE + " Lowest prices found in the requested category, counting only rows that can be "
                "bought now and fit the budget: " + cells + "." + self.SWEEP_TAIL)

    # ------------------------------------------------------------ 3.19 wide step (move 3)
    @_p09_guard("wide_error", False)
    def broad_turn(self):
        """Move 3: after a dropped requirement, search without the dropped words and open cheaper rows."""
        s = self.s
        mark = s.ledger.widened_at
        if not kit.is_num(mark) or mark < 0 or s.multi() or s.orders_tried or s.link.done or s.link.broke:
            return False
        if mark != self._w_mark:
            self._w_mark = mark
            q = self._wide_texts() if s.ledger.priority() == "cheapest" and s.role("search") else []
            self._w_queue, self._w_col = q, ([] if q else None)
            s.link.note("wide_plan", queries=list(q))
        q = self._w_queue
        if q and s.steps_left() > Tune.Search.WIDE_KEEP + len(q) + 1 and s.link.time_left() > Tune.Clock.CLOSING + 40:
            return self._wide_search(q.pop(0))
        if q or self._w_col is None:
            self._w_queue = []
            return False
        col, self._w_col = self._w_col, None
        return self._wide_open(col)

    def _wide_texts(self):
        vals = []
        for opts in (self.s.pricer.value_words(None) or [])[:2]:
            if opts and opts[0]:
                vals += [str(w) for w in opts[0]]
        lead = " ".join(vals).strip()
        if not lead:
            return []
        lw = set(lead.split())
        out = []
        for t in [lead + " " + k for k in [k for k in self._wide_kinds() if k not in lw][:2]] + [lead]:
            if t not in out and not self.already_ran(t):
                out.append(t)
        return out[:Tune.Search.WIDE_TRIES]

    def _wide_kinds(self):
        heads = {kit.stem(w) for l in self.wanted_shelves() for w in re.findall(r"[a-z]{3,}", str(l).lower()) if not kit.is_link(w)}
        tally = {}
        for p, t, b in self.buyable_rows().values():
            bw = set(re.findall(r"[a-z]+", str(b or "").lower()))
            for w in dict.fromkeys(re.findall(r"[a-z]{3,}", str(t or "").lower())):
                if kit.is_link(w) or w in bw:  # "skip words" read as the row's brand words
                    continue
                r = kit.stem(w)
                if any(h in r for h in heads) or (len(r) >= 4 and any(r in h for h in heads)):
                    tally.setdefault(r, [0, w])[0] += 1
        return [x[3] for x in sorted((r in heads, -v[0], r, v[1]) for r, v in tally.items() if v[0] >= 2 or r in heads)]

    def _wide_search(self, text):
        s = self.s
        tpl = self.template()
        if not tpl:
            self._w_queue = []
            return False
        args = dict(tpl[0])
        args[tpl[1]] = text
        val, faults = ArgFixer.fix(args, s.schema(s.role("search")))
        if faults or not isinstance(val, dict):
            return False
        homes = s.judge.home_labels() or []
        home = set(homes[0]) if homes and homes[0] else self.wanted_shelves()
        mark, steps = len(s.archive.obs), s.link.steps
        s.walker.search_call(val, self.WIDE_SAY.replace("<<Q>>", text))
        got = []
        def see(d):
            rid = kit.row_id(d) if len(d) >= 4 else ""
            if rid and (not home or kit.shelf_field(d) in home):
                got.append(rid)
        for o in s.archive.obs[mark:]:
            kit.visit(o, see, top=3)
        if self._w_col is not None:
            self._w_col += got
        s.link.note("wide_search", query=text, rows=len(got))
        return s.link.steps > steps

    def _wide_open(self, col):
        s = self.s
        held, below = list(s.basket.units), None
        if held:
            ps = [p for p in (s.archive.price(l) for l in held) if kit.is_num(p)]
            if not ps:
                return False
            below = min(ps)
        budget, qty, ranked = s.ledger.budget, s.ledger.quantity(), []
        for lid in dict.fromkeys(col):
            p = s.archive.price(lid)
            if lid in s.archive.read or lid in s.refused_lines or s.judge.supply_state(lid) is False or not kit.is_num(p):
                continue
            if (kit.is_num(budget) and p * qty > budget + 1e-9) or (below is not None and not p < below):
                continue
            ranked.append((p, lid))
        ranked.sort(key=lambda x: x[0])
        room = max(0, min(Tune.Search.WIDE_OPENS, s.steps_left() - Tune.Search.WIDE_KEEP))
        queue = [l for _, l in ranked if l not in s.basket.units][:room]
        s.link.note("wide_open", below=below, ranked=[l for _, l in ranked[:6]], queue=queue)
        if not queue:
            return False
        s.judge.add_leads(queue)
        s.rank_marks.clear()
        s.order_vetted = False
        s.judge.queue_open(queue, "wide", recheck=True, say=self.WIDE_OPEN)
        sent = bool(s.judge.open_queued())
        if sent and held:
            opened = ", ".join("%s (%s)" % (l, s.archive.price(l)) for l in queue)
            s.engine.append_note(self.AFTER_DROP % (", ".join(map(str, held)), opened), status=True)
        return sent

    # ------------------------------------------------------------ 3.20 spelling twins
    @_p09_guard("twin_error")
    def study_pairs(self, args, rows):
        """Learn store spelling twins from rows that lack an asked word but carry a near spelling."""
        if self.s.word_mode == "any":
            return
        asked = list(dict.fromkeys(w for _, t in self.open_text(args or {}) for w in re.findall(r"[a-z]{4,}", t.lower())))
        if not asked:
            return
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            rw = set(re.findall(r"[a-z]+", " ".join(kit.rec_strings(r)).lower()))
            ident = kit.identity(r)
            for a in asked:
                if a in rw or any(kit.kind_match(a, x) for x in rw):
                    continue
                for x in rw:
                    if kit.spelling_twins(a, x):
                        proof = self._twin_proof.setdefault((a, x), set())
                        proof.add(ident)
                        if len(proof) >= 2:
                            kit.add_twin(a, x)

    @_p09_guard("twin_error")
    def pair_search(self, slot):
        """Search args with a value word swapped for its store twin, or None."""
        s = self.s
        tpl = self.template()
        if not tpl or not kit.TWINS:
            return None
        vw = set()
        for line in s.ledger.in_force():
            if s.multi() and s.mender.item_for(str(line).split(":", 1)[0]) not in (slot, None):
                continue
            vw |= set(re.findall(r"[a-z]{3,}", kit.strip_marks(self._value_of(line)).lower()))
        homes = s.judge.home_labels() or []
        i = slot if isinstance(slot, int) else 0
        home = set(homes[i]) if i < len(homes) and homes[i] else set()
        if slot is None and not home:
            home = self.wanted_shelves()
        best = None
        for e in self._book.values():
            if home and not (e["shelves"] & home):
                continue
            ws = kit.words(e["text"])
            for j, w in enumerate(ws):
                if w not in vw:
                    continue
                for t in sorted(kit.TWINS.get(w, ())):
                    c = " ".join(ws[:j] + [t] + ws[j + 1:])
                    if (best is None or len(c) < len(best)) and not self.already_ran(c):
                        best = c
        if not best:
            return None
        a = dict(tpl[0])
        a[tpl[1]] = best
        return a

    # ------------------------------------------------------------ 3.21 value score, pages, hits
    @_p09_guard("score_error", 0)
    def value_rating(self, text):
        """Live requirement values whose value words some (leading) query word matches."""
        s = self.s
        qw = self._runs(text)
        if s.lead_words:
            qw = qw[:s.lead_words]
        n = 0
        for line in s.ledger.in_force():
            v = kit.strip_marks(self._value_of(line))
            if kit.is_pointer(v):
                continue
            vw = [w for w in re.findall(r"[a-z0-9]+", v.lower()) if len(w) >= 3 or re.fullmatch(r"\d[a-z]|[a-z]\d", w)]
            if any(q == w or kit.near(q, w) for w in vw for q in qw):
                n += 1
        return n

    @_p09_guard("pages_error", list)
    def best_pages(self, n=2):
        """Best unread next pages of value-scored walks."""
        if not self.paging():
            return []
        want, out = self.wanted_shelves(), []
        for i, (k, e) in enumerate(self._book.items()):
            sp = self.following_page(k)
            if sp is None or (want and e["shelves"] and not (e["shelves"] & want)):
                continue
            sc = self.value_rating(e["text"])
            if sc > 0:
                out.append((-sc, e["total"], i, self.page_call(k, sp)))
        out.sort(key=lambda x: x[:3])
        return [x[3] for x in out[:n]]

    @_p09_guard("hits_error", list)
    def row_matches(self, rows, args):
        """Unopened in-budget, in-stock, on-shelf rows; also collects reference pointer rows."""
        s = self.s
        budget, qty, want = s.ledger.budget, s.ledger.quantity(), self.wanted_shelves()
        asked = set()
        if s.ledger.reading.get("pointer_needs"):
            asked = set(kit.words(" ".join(t for _, t in self.open_text(args or {}))))
        pw = self.cue_terms() if asked else set()
        hits = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            lid = kit.identity(r)
            if lid not in s.archive.known or lid in s.archive.read:
                continue
            if asked and lid not in s.ref_pointers:
                for k, v in r.items():
                    nm = str(k).lower()
                    if isinstance(v, str) and v.strip() and self.SELLER.search(nm) and "id" not in nm:
                        ws = kit.words(v)
                        if ws and all(w in asked and w in pw for w in ws):
                            s.ref_pointers.append(lid)
                            break
            p = self._price_of(r)
            if (p is not None and kit.is_num(budget) and p * qty > budget + 1e-9) or kit.out_of_stock(r):
                continue
            sh = kit.shelf_field(r)
            if want and sh and sh not in want:
                continue
            hits.append(lid)
            s.hit_shelf[lid] = sh
        return hits

    def cue_terms(self):
        """Words of the references plus the descriptive words of opened listings (minus their own seller words)."""
        s = self.s
        n = len(s.archive.obs)
        if self._pw and self._pw[0] == n:
            return set(self._pw[1])
        out = set()
        try:
            refs = s.ledger.reading.get("pointer_needs") or []
            for r in refs if isinstance(refs, list) else [refs]:
                out |= set(kit.words(r if isinstance(r, str) else kit.render(r)))
            read = s.archive.read
            def see(d):
                if kit.identity(d) not in read:
                    return
                desc, sell = set(), set()
                for k, v in d.items():
                    if not isinstance(v, str):
                        continue
                    nm = str(k).lower()
                    if self.SELLER.search(nm) and "id" not in nm and len(v) <= 40 and not self.NOTSELL.search(nm):
                        sell |= set(kit.words(v))
                    elif not self.TITLEISH.search(nm):
                        desc |= set(kit.words(v))
                out.update(desc - sell)
            for o in s.archive.obs:
                kit.visit(o, see)
        except Exception as e:
            s.link.fault("pointed_error", e)
        self._pw = (n, out)
        return set(out)

    # ------------------------------------------------------------ 3.22 value probe
    def _reqs(self):
        out = []
        for line in self.s.ledger.in_force():
            line = str(line)
            if ":" in line:
                nm, v = line.split(":", 1)
                out.append((nm.strip(), kit.strip_marks(v).strip().lower()))
        return out

    @_p09_guard("probe_error")
    def value_test(self, texts):
        """(property name, suggested query) when the query leads with a property's NAME, else None."""
        s = self.s
        if s.word_mode == "any":
            return None
        reach = s.lead_words or 2
        lead = self._runs(texts if isinstance(texts, str) else " ".join(str(t) for t in texts or []))[:reach]
        prod = kit.words(s.ledger.product_name() or "")[:3]
        reqs = [(nm, re.split(r"[,;]", v)[0].strip()) for nm, v in self._reqs()]
        def touch(pool):
            return any(kit.near(a, b, "lead") for a in lead for b in pool)
        known = set(prod)
        for _, v in reqs:
            known |= set(self._runs(v))
        if not lead or touch(known):
            return None
        hit = next(((nm, v) for nm, v in reqs if v and touch(self._runs(nm))), None)
        if not hit:
            return None
        m = re.search(r"\d", hit[1])
        core = hit[1][m.start():] if m else hit[1]
        probe = (core.strip(" .;") + " " + " ".join(prod)).strip()
        pl = self._runs(probe)[:reach]
        for e in self._book.values():
            wl = self._runs(e["text"])[:reach]
            if len(wl) == len(pl) and all(kit.near(a, b, "lead") for a, b in zip(wl, pl)):
                return None
        return (hit[0], probe)

    # ------------------------------------------------------------ 3.23 word trap
    def term_snare(self, texts):
        """(note, suggested query) when a query word hides acceptable listings, else None."""
        s = self.s
        try:
            if s.word_mode == "any":
                return None
            txt = texts if isinstance(texts, str) else " ".join(str(t) for t in texts or [])
            toks = [t for t in (x.strip(self.TRAP_EDGE).lower() for x in txt.split()) if re.search(r"[a-z0-9]", t)]
            width = s.lead_words or (max(3, len(toks)) if s.word_mode == "all" else 3)
            toks = toks[:width]
            reqs = [(nm.split(" (")[0].strip().lower(), nm, v) for nm, v in self._reqs()]
            prods = [str(p.get("name") or "") for p in (s.ledger.products() or []) if isinstance(p, dict)]
            prods = prods or [str(s.ledger.product_name() or "")]
            def link_free(pat, items):
                return {w for t in items for w in re.findall(pat, str(t).lower()) if not kit.is_link(w)}
            vals = {w for _, _, v in reqs for w in self._runs(v)}
            kinds = link_free(r"[a-z0-9]{3,}", prods + sorted(self.wanted_shelves()))
            names = link_free(r"[a-z]{3,}", [r[0] for r in reqs])
            drop = link_free(r"[a-z]{3,}", [kit.strip_marks(self._value_of(d)) for d in (s.ledger.dropped or [])])
            def near(ps, pool):
                return any(kit.near(p, w, "equal") for p in ps for w in pool)
            kept, why = [], None
            for t in toks:
                ps = self._runs(t)
                if near(ps, vals | kinds):
                    kept.append(t)
                    continue
                v = "name" if near(ps, names) else "dropped" if near(ps, drop) else None
                if v:
                    why = why or (t, v)
                else:
                    kept.append(t)
            kp = [p for t in kept for p in self._runs(t)]
            held = [r for r in reqs if near(kp, self._runs(r[2]))]
            if not why or not held:
                return None
            if not near(kp, kinds):
                nm = prods[0] if prods else ""
                if s.multi():
                    i, pr = s.mender.item_for(held[0][1]), s.ledger.products() or []
                    if isinstance(i, int) and 0 <= i < len(pr):
                        nm = str(pr[i].get("name") or "")
                w = [x for x in re.findall(r"[a-z]{3,}", nm.lower()) if not kit.is_link(x)]
                if w:
                    kept.append(w[-1])
            q = " ".join(kept[:width])
            if not q or self.already_ran(q):
                return None
            note = (self.TRAP_NAME if why[1] == "name" else self.TRAP_DROP) % why[0]
            return (note + self.TRAP_END % q, q)
        except Exception as e:
            s.link.fault("trap_error", e)
            return None



# ------------------------------------------------------------ p10_walker.py
_p10_ALTS = re.compile(r"\s*(?:[,;/]|\b(?:or|either|nor)\b)\s*")
_p10_FIRST = re.compile(r"[,;/]|\b(?:or|and)\b")
_p10_AN = re.compile(r"[^\W_]+")
_p10_AN2 = re.compile(r"[^\W_]{2,}")
_p10_AN3 = re.compile(r"[^\W_]{3,}")
_p10_AL = re.compile(r"[^\W\d_]+")
_p10_AL2 = re.compile(r"[^\W\d_]{2,}")
_p10_AL3 = re.compile(r"[^\W\d_]{3,}")
_p10_CODE = re.compile(r"\b(?:\d[^\W\d_]|[^\W\d_]\d)\b")
_p10_BARE = re.compile(r"\s*(\d+)\s+([^\W\d_]+)")
_p10_AMT = re.compile(r"\s*\d+(?:[.,]\d+)?\s+[^\W\d_]")
_p10_TITLE = re.compile(r"[^\W\d_][^\W\d_'\-]{3,}")
_p10_STOCK = re.compile(r"stock|avail")
_p10_PRICE = re.compile(r"price|cost")
_p10_MAKER = re.compile(r"brand|make|manufactur")


class Walker:
    """Arbitration, fit hold, unrefuted set, store order, cheapest walks and decline reads (spec 08)."""
    HURRY = Tune.Clock.CLOSING + Tune.Clock.CLOCK_STEP
    QUEUE_CLOCK = Tune.Clock.CLOSING + 60
    KEEP = 6
    DR_PAGES = 3
    DIGEST_ROWS = 12
    TITLE_CUT = 70
    DEEP = 12
    HEAD = "Rows on this page (id price 'title' [category]): "

    def __init__(self, s):
        self.s = s
        self.walked = set()
        self._fit_used = False
        self._dr_done = False
        self._job = None

    # ------------------------------------------------------------------ small helpers
    def _fail(self, name, e):
        self.s.link.fault(name + "_error", e)

    @staticmethod
    def _lows(labels):
        return {kit.low(x).strip() for x in (labels or ()) if x and kit.low(x).strip()}

    @staticmethod
    def _num(v):
        if not kit.is_num(v):
            return "?"
        v = float(v)
        return str(int(v)) if v.is_integer() else ("%.2f" % v).rstrip("0").rstrip(".")

    def _price(self, lid):
        p = self.s.archive.price(lid)
        return p if kit.is_num(p) else None

    def _cap(self):
        b = self.s.ledger.budget
        return b if kit.is_num(b) else None

    def _units(self, slot):
        ps = self.s.ledger.products() or []
        if slot is not None and 0 <= slot < len(ps):
            q = ps[slot].get("qty")
            return q if kit.is_num(q) and q > 0 else 1
        q = self.s.ledger.quantity()
        return q if kit.is_num(q) and q > 0 else 1

    def _nprod(self):
        return len(self.s.ledger.products() or []) if self.s.multi() else 0

    def _home(self, k):
        hs = self.s.judge.home_labels() or []
        return self._lows(hs[k]) if k is not None and 0 <= k < len(hs) else set()

    def _label(self, lid):
        return kit.low(self.s.judge.label_of(lid) or "").strip()

    def _cleared(self):
        suf = "|" + str(self.s.ledger.version)
        return {m[:-len(suf)] for m in self.s.passed if m.endswith(suf)}

    def _struck(self):
        v = str(self.s.ledger.version)
        out = set()
        for memo, text in self.s.rulings.items():
            parts = str(memo).split("|")
            if text and len(parts) >= 2 and parts[1] == v:
                out.add(parts[0])
        return out

    def _book(self):
        return self.s.seeker.book() or {}

    def _query(self, args):
        return " ".join(str(t) for _, t in (self.s.seeker.open_text(args) or [])).strip()

    def _short(self, margin, broke=False):
        s = self.s
        return (s.steps_left() <= margin or s.link.done or (broke and s.link.broke)
                or s.link.time_left() < self.HURRY)

    def _other(self, head, slot):
        """Requirement head tied to a product other than slot (multi-product errands only)."""
        if slot is None or not self.s.multi():
            return False
        t = self.s.mender.item_for(head)
        return t is not None and t != slot

    def _attach(self, text, status):
        c = self.s.chat
        if c and c[-1].get("role") == "tool":
            self.s.engine.append_note(text, status=status)

    def _marks(self, test):
        """Stock marks: identity -> sold out, newest record passing the filter wins."""
        marks = {}
        def fn(d):
            if test(d):
                marks[kit.identity(d)] = bool(kit.out_of_stock(d))
        for o in self.s.archive.obs:
            kit.visit(o, fn, top=self.DEEP)
        return marks

    def _batch(self):
        bs = self.s.mender.batch_spec()
        if not bs:
            return None, 0
        try:
            n = int(bs[2] or 1)
        except (TypeError, ValueError):
            n = 1
        return bs, max(1, min(n, Tune.Search.LIST_CAP))

    @staticmethod
    def _deal(pools, most):
        out, idx = [], [0] * len(pools)
        while len(out) < most:
            moved = False
            for i, p in enumerate(pools):
                while idx[i] < len(p) and p[idx[i]] in out:
                    idx[i] += 1
                if idx[i] < len(p):
                    out.append(p[idx[i]])
                    idx[i] += 1
                    moved = True
                    if len(out) >= most:
                        break
            if not moved:
                break
        return out

    def _trim(self, queue, floor, batch):
        q = list(queue)
        while q and (1 if batch and len(q) >= 2 else len(q)) >= self.s.steps_left() - floor:
            q.pop()
        return q

    # ------------------------------------------------------------------ 3.2 combined value searches
    def _odd(self, v):
        # flag 2: shared exclusion and reference tests replace the source's own phrase list
        return bool(re.search(r"\d", v)) or kit.has_exclusion(v) or kit.is_pointer(v) or "anything" in v

    def _groups(self, slot):
        out = []
        for line in self.s.ledger.in_force() or []:
            head, _, val = str(line).partition(":")
            if self._other(head, slot):
                continue
            v = kit.low(kit.strip_marks(val)).strip()
            if not v or self._odd(v):
                continue
            ws = [w for w in _p10_AL2.findall(_p10_FIRST.split(v)[0]) if not kit.is_link(w)]
            if 1 <= len(ws) <= 2:
                out.append(ws)
        return out

    def combined_searches(self):
        s = self.s
        try:
            tp = s.seeker.template()
            if not tp:
                return []
            base, field = tp
            room = s.lead_words if s.lead_words and s.lead_words > 0 else 3
            multi = s.multi()
            if multi:
                labs = {self._label(c) for c in self._cleared() - set(s.refused_lines)} - {""}
                slots = [k for k in range(self._nprod()) if self._home(k) and not (self._home(k) & labs)]
                names = [p.get("name") or "" for p in s.ledger.products()]
            else:
                slots = [None] if s.intake.no_verified_fit() else []
                names = []
            out = []
            for k in slots:
                groups = self._groups(k)
                if len(groups) < 2:
                    continue
                q = []
                for g in groups:
                    if len(q) + len(g) <= room:
                        q += g
                if len(q) < room:
                    name = names[k] if multi and k < len(names) else (s.ledger.product_name() or "")
                    ws = [w for w in _p10_AL3.findall(kit.low(name)) if not kit.is_link(w)]
                    if ws and ws[-1] not in q:
                        q.append(ws[-1])
                text = " ".join(q)
                if len(q) > 1 and not s.seeker.already_ran(text):
                    a = dict(base)
                    a[field] = text
                    out.append(a)
                if len(out) >= 2:
                    break
            return out[:2]
        except Exception as e:
            self._fail("combined_searches", e)
            return []

    # ------------------------------------------------------------------ 3.3 arbitration
    def _second(self, rejection):
        why = ""
        for ent in reversed(self.s.link.log):
            if isinstance(ent, dict) and ent.get("kind") == "audit" and isinstance(ent.get("result"), dict):
                r = ent["result"]
                why = str(r.get("why") or r.get("reason") or "").strip()[:400]
                break
        t = PROMPTS.get("second") or ""
        for a, b in (("(", ")"), ("[", "]")):
            tok = a + "<<AUDIT_REASON>>" + b
            if tok in t:
                t = t.replace(tok, (a + why + b) if why else "")
        t = t.replace("<<AUDIT_REASON>>", (" (" + why + ")") if why else "")
        return t.replace("<<EARLIER>>", str(rejection or "")[:500])

    def arbitrate(self, lid, rejection):
        s = self.s
        try:
            verdict = s.judge.item_check(lid, Tune.Aim.ORDER, self._second(rejection))
            if verdict:
                return verdict
            base = "%s|%s" % (lid, s.ledger.version)
            for k in list(s.rulings):
                if k == base or str(k).startswith(base + "|"):
                    s.rulings[k] = ""
            return ""
        except Exception as e:
            self._fail("arbitrate", e)
            return rejection

    # ------------------------------------------------------------------ 3.4 fit hold
    def match_pause(self):
        s = self.s
        try:
            if self._fit_used or s.basket.units or s.steps_left() < 5 or s.link.time_left() < self.HURRY:
                return ""
            todo = [f for f in (s.audit_fits or []) if f in s.archive.known and f not in s.refused_lines][:2]
            if not todo:
                return ""
            self._fit_used = True
            multi, good = s.multi(), []
            for lid in todo:
                if lid not in s.archive.read:
                    continue
                v = s.judge.item_check(lid, Tune.Aim.ORDER)
                if v and s.link.time_left() > self.HURRY:
                    v = self.arbitrate(lid, v)
                if not v and s.passed_now(lid):
                    good.append(lid)
                    if not multi:
                        break
            if not good:
                return ""
            if multi:
                labs = {self._label(c) for c in self._cleared() - set(s.refused_lines)}
                for k in range(self._nprod()):
                    h = self._home(k)
                    if h and not (h & labs):
                        return ""
            one = len(good) == 1
            what = "one passed line for each product" if multi else ("it" if one else "one of them")
            return ("The decline is on hold. The pre-decline audit named %s as %s, and a full-record item check "
                    "has since cleared %s on every rule now applying. Reconsider: if %s, put %s into the cart, "
                    "re-read %s stock and price, then send the order. A decline is only for once a detail shows %s "
                    "failing a rule." % (" and ".join(good), "a possible fit" if one else "possible fits",
                                         "it" if one else "them both",
                                         "it truly qualifies" if one else "they truly qualify", what,
                                         "its" if one else "their", "it" if one else "them"))
        except Exception as e:
            self._fail("match_pause", e)
            return ""

    # ------------------------------------------------------------------ 3.5 unrefuted cheapest set
    def unrefuted(self):
        s = self.s
        try:
            multi = s.multi()
            sets = [self._home(k) for k in range(self._nprod())] if multi else [self._lows(s.seeker.wanted_shelves())]
            if not sets or not all(sets):
                return []
            read = set(s.archive.read)
            fits = set(s.audit_fits or [])
            fresh = s.audit_mark is not None and len(read) <= s.audit_mark
            reviewed = read if s.audit_done and (s.audit_seen is None or fresh) else set(s.audit_seen or ())
            barred = (self._struck() | (reviewed - fits) | set(s.refused_lines)
                      | set(s.tracer.chain_ids() or ()))
            known = s.archive.known
            gone = self._marks(lambda d: kit.identity(d) in known
                               and any(_p10_STOCK.search(kit.low(k)) for k in d))
            cleared, cap = self._cleared(), self._cap()
            chosen, outlay = [], 0.0
            for i, slot in enumerate(range(len(sets)) if multi else [None]):
                units, best = self._units(slot), None
                for lid in sorted(read):
                    if lid in barred or gone.get(lid) or lid in chosen or self._label(lid) not in sets[i]:
                        continue
                    p = self._price(lid)
                    if p is None or (cap is not None and p * units > cap + 1e-9):
                        continue
                    r = (lid not in cleared, p, lid)
                    if best is None or r < best:
                        best = r
                if best is None:
                    return []
                chosen.append(best[2])
                outlay += best[1] * units
            if cap is not None and outlay > cap + 1e-9:
                return []
            return chosen
        except Exception as e:
            self._fail("unrefuted", e)
            return []

    # ------------------------------------------------------------------ 3.6 store result order
    def _verdicts(self):
        return [e.get("id_order") for e in self._book().values() if e.get("id_order") is not None]

    def id_sorted(self):
        try:
            v = self._verdicts()
            return not any(x is False for x in v) and sum(1 for x in v if x is True) >= 2
        except Exception as e:
            self._fail("id_sorted", e)
            return False

    def ranked(self):
        try:
            v = self._verdicts()
            return not any(x is True for x in v) and sum(1 for x in v if x is False) >= 2
        except Exception as e:
            self._fail("ranked", e)
            return False

    def tour_shop(self):
        try:
            return len(self._verdicts()) >= 2
        except Exception as e:
            self._fail("tour_shop", e)
            return False

    def walkable(self, key):
        try:
            e = self._book().get(key)
            if not e:
                return False
            v = e.get("id_order")
            if v is True:
                return True
            if v is False:
                if self.ranked():
                    return True
                if self.s.seeker.open_text(e.get("args") or {}):
                    homes = set()
                    for h in self.s.judge.home_labels() or []:
                        homes |= self._lows(h)
                    return bool(self._lows(e.get("shelves")) & homes)
            return False
        except Exception as e:
            self._fail("walkable", e)
            return False

    # ------------------------------------------------------------------ 3.7 figure tests, carry test
    def bare_amount(self, text):
        try:
            t = kit.low(text or "")
            if kit.has_bound(t):
                return None
            m = _p10_BARE.match(t)
            return (m.group(1), m.group(2)) if m else None
        except Exception as e:
            self._fail("bare_amount", e)
            return None

    def is_amount(self, text):
        try:
            t = kit.low(text or "")
            return bool(re.search(r"\d", t)) and (kit.has_bound(t) or bool(_p10_AMT.match(t)))
        except Exception as e:
            self._fail("is_amount", e)
            return False

    def _carry(self, args, slot, alts=False):
        joined = " ".join(kit.low(t) for _, t in (self.s.seeker.open_text(args) or []))
        words, toks = _p10_AN2.findall(joined), _p10_AN.findall(joined)
        n = self.s.lead_words or 0
        if n > 0:
            words, toks = words[:n], toks[:n]
        out = set()
        if not words and not toks:
            return out
        for i, line in enumerate(self.s.ledger.in_force() or []):
            head, _, val = str(line).partition(":")
            if self._other(head, slot):
                continue
            pieces = [p.strip() for p in _p10_ALTS.split(kit.low(kit.strip_marks(val)).strip()) if p and p.strip()]
            for j, alt in enumerate(pieces):
                ba = self.bare_amount(alt)
                if ba:
                    hit = ba[0] in toks and any(t == ba[1] or kit.near(t, ba[1]) for t in toks)
                elif self.is_amount(alt):
                    hit = False
                elif any(c in words for c in _p10_CODE.findall(alt)):
                    hit = True
                else:
                    hit = any(kit.near(w, p) for w in words for p in _p10_AN3.findall(alt))
                if hit:
                    out.add((i, j) if alts else i)
        return out

    def _args_of(self, key):
        if isinstance(key, dict):
            return key
        e = self._book().get(key)
        return (e.get("args") or {}) if e else None

    def carries(self, key, slot):
        try:
            args = self._args_of(key)
            return sorted(self._carry(args, slot)) if args is not None else []
        except Exception as e:
            self._fail("carries", e)
            return []

    def tour_rating(self, key, slot):
        try:
            args = self._args_of(key)
            return len(self._carry(args, slot)) if args is not None else 0
        except Exception as e:
            self._fail("tour_rating", e)
            return 0

    def _lead(self, args):
        toks = _p10_AN.findall(" ".join(kit.low(t) for _, t in (self.s.seeker.open_text(args) or [])))
        n = self.s.lead_words or 0
        return {kit.stem(t) for t in (toks[:n] if n > 0 else toks)}

    def _value_stems(self, slot):
        out = set()
        for line in self.s.ledger.in_force() or []:
            head, _, val = str(line).partition(":")
            if not self._other(head, slot):
                out |= {kit.stem(t) for t in _p10_AN2.findall(kit.low(val))}
        return out

    def bounds_only(self, slot):
        s = self.s
        try:
            if slot is None or not s.multi():
                return False
            own = []
            for line in s.ledger.in_force() or []:
                head, _, val = str(line).partition(":")
                if s.mender.item_for(head) == slot:
                    own.append(kit.low(kit.strip_marks(val)).strip())
            return bool(own) and all(kit.has_bound(v) and self.is_amount(v) for v in own)
        except Exception as e:
            self._fail("bounds_only", e)
            return False

    @staticmethod
    def _onshelf(e, shelf):
        rows = (e or {}).get("row_shelf") or {}
        return (bool(shelf) and len(rows) >= max(1, (e or {}).get("size") or 0)
                and all(kit.low(l or "").strip() in shelf for l in rows.values()))

    def on_shelf(self, key, slot):
        try:
            return self._onshelf(self._book().get(key), self._lows(self.tour_shelf(slot)))
        except Exception as e:
            self._fail("on_shelf", e)
            return False

    # ------------------------------------------------------------------ 3.8 walk options
    def _mode(self, args):
        tool = self.s.role("search")
        props = (self.s.schema(tool).get("properties") or {}) if tool else {}
        out = []
        for k, v in (args or {}).items():
            spec = props.get(k) if isinstance(props, dict) else None
            en = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(v, str) and isinstance(en, list) and len(en) <= 4:
                out.append((k, v))
        return tuple(sorted(out))

    def tour_choices(self, slot, deep=False):
        try:
            return self._options(slot, deep)
        except Exception as e:
            self._fail("tour_choices", e)
            return []

    def _options(self, slot, deep):
        s, sk = self.s, self.s.seeker
        shelf = self._lows(self.tour_shelf(slot))
        if not shelf:
            return []
        book = self._book()
        wide = {kit.stem(w) for lab in shelf for w in _p10_AN2.findall(lab)} | self._value_stems(slot)
        done = []
        for key, e in book.items():
            args = e.get("args") or {}
            if not self.walkable(key) or e.get("first_ids") is None or sk.following_page(key) is not None:
                continue
            lead = self._lead(args)
            if lead and len(e.get("row_shelf") or {}) >= (e.get("total") or 0):
                done.append((lead, self._carry(args, slot, True), self._mode(args)))
        bare = self.bounds_only(slot)
        deep = deep or bare
        pg = sk.paging() or {}
        opts, seen = [], {}
        for key, e in book.items():
            args = e.get("args") or {}
            if not self.walkable(key) or e.get("first_ids") is None or sk.following_page(key) is None:
                continue
            onshelf = self._onshelf(e, shelf)
            if not self._carry(args, slot) and not (bare and onshelf):
                continue
            if not any(kit.low(l or "").strip() in shelf for l in (e.get("row_shelf") or {}).values()):
                continue
            lead, pairs, mode = self._lead(args), self._carry(args, slot, True), self._mode(args)
            if any(f[2] == mode and (f[0] <= lead or (not bare and f[0] <= wide and pairs <= f[1])) for f in done):
                continue
            raw = e.get("size") or 0
            size, total = max(1, raw), e.get("total") or 0
            left = math.ceil(total / size)
            if pg.get("mode") in ("page", "offset") and kit.is_num(pg.get("start")) and kit.is_num(pg.get("top")):
                step = 1 if pg.get("mode") == "page" else size
                left = min(left, int((pg["top"] - pg["start"]) // step) + 1)
            left -= len(e.get("spots") or ())
            if not (0 < left <= 2 * Tune.Search.WALK_PAGES or (left > 0 and (1 <= raw <= 4 or (deep and onshelf)))):
                continue
            fid = e.get("first_ids")
            sig = (total, tuple(fid) if isinstance(fid, (list, tuple)) else fid)
            if sig in seen:
                j = seen[sig]
                if left < opts[j][0]:
                    opts[j] = (left, key, e, lead)
                continue
            seen[sig] = len(opts)
            opts.append((left, key, e, lead))
        if s.word_mode == "all" or (s.word_mode != "any" and (s.lead_words or 0) > 0):
            keep = []
            for o in opts:
                drop = False
                for p in opts:
                    if p is o:
                        continue
                    if p[3] < o[3] and not (o[3] - p[3]) <= wide and p[0] <= Tune.Search.WALK_PAGES:
                        drop = True
                    elif o[3] < p[3] and (p[3] - o[3]) <= wide:
                        drop = True
                    if drop:
                        break
                if not drop:
                    keep.append(o)
            opts = keep
        opts.sort(key=lambda o: (o[0], -len(self._carry(o[2].get("args") or {}, slot)), o[1]))
        return [(o[0], o[1], o[2]) for o in opts]

    # ------------------------------------------------------------------ 3.9-3.10 pick, slot, home
    def pick_of(self, slot):
        s = self.s
        try:
            def mine(l):
                return slot is None or self.product_slot(l) == slot
            pool = [l for l in s.basket.units if l not in s.dead and mine(l)]
            if not pool:
                pool = [l for l in self._cleared() if l not in s.refused_lines and l not in s.dead and mine(l)]
            best = None
            for l in pool:
                p = self._price(l)
                if p is not None and (best is None or (p, l) < best):
                    best = (p, l)
            return (best[1], best[0]) if best else None
        except Exception as e:
            self._fail("pick_of", e)
            return None

    def product_slot(self, lid):
        try:
            if not self.s.multi():
                return None
            lab = self._label(lid)
            if not lab:
                return None
            for k, h in enumerate(self.s.judge.home_labels() or []):
                if lab in self._lows(h):
                    return k
            return None
        except Exception as e:
            self._fail("product_slot", e)
            return None

    def tour_shelf(self, slot):
        try:
            hs = self.s.judge.home_labels() or []
            if slot is None:
                h = set(hs[0]) if hs else set()
                return h or set(self.s.seeker.wanted_shelves() or ())
            return set(hs[slot]) if 0 <= slot < len(hs) else set()
        except Exception as e:
            self._fail("tour_shelf", e)
            return set()

    # ------------------------------------------------------------------ 3.11 walk plan
    def tour_plan(self, budget=8, only=None, rule="cheapest"):
        s = self.s
        try:
            if (budget <= 0 or s.ledger.priority() != rule or not s.role("search") or not s.seeker.paging()
                    or not self.tour_shop()):
                return []
            single = not s.multi()
            wants, extra = [], []
            for k in ([None] if single else range(self._nprod())):
                if k in self.walked or (only is not None and k not in only):
                    continue
                opts = self.tour_choices(k, deep=single)
                if not opts:
                    continue
                wants.append((opts[0][0], -1 if k is None else k, k, opts[0][1]))
                if single:
                    extra = opts[1:]
            wants.sort(key=lambda w: (w[0], w[1]))
            plan, left = [], budget
            for w in wants:
                take = min(w[0], Tune.Search.WALK_PAGES, left)
                if take <= 0:
                    break
                plan.append((w[2], w[3], take))
                left -= take
            if single and plan:
                for o in extra:
                    take = min(o[0], Tune.Search.WALK_PAGES, left)
                    if take <= 0:
                        break
                    plan.append((None, o[1], take))
                    left -= take
            return plan
        except Exception as e:
            self._fail("tour_plan", e)
            return []

    # ------------------------------------------------------------------ 3.12 line search
    @staticmethod
    def _with_text(r, tp):
        if isinstance(r, dict) and r:
            return r
        if isinstance(r, str) and r.strip():
            a = dict(tp[0])
            a[tp[1]] = r.strip()
            return a
        return None

    def _line_search(self, slot):
        s = self.s
        pk, tp = self.pick_of(slot), s.seeker.template()
        if not pk or not tp:
            return None
        lid, found = pk[0], []
        def fn(d):
            if kit.identity(d) == lid and kit.row_title(d):
                found.append(d)
        for o in s.archive.obs:
            kit.visit(o, fn, top=self.DEEP)
        if not found:
            return None
        rec, maker = found[-1], ""
        for k, v in rec.items():
            lk = kit.low(k)
            if _p10_MAKER.search(lk) and "id" not in lk and isinstance(v, str) and v.strip():
                maker = v
                break
        mw = _p10_AN.findall(kit.low(maker))
        if not mw:
            return self._with_text(s.pricer.heading_line(slot, rec, tp), tp)
        brand, stops = mw[0], set(mw)
        names = [p.get("name") or "" for p in s.ledger.products()] if s.multi() else [s.ledger.product_name() or ""]
        for t in list(self.tour_shelf(slot)) + list(s.ledger.in_force() or []) + names:
            stops |= set(_p10_AL.findall(kit.low(t)))
        words = _p10_TITLE.findall(kit.low(kit.row_title(rec)))
        start = words.index(brand) + 1 if brand in words else 0
        fam = ""
        for w in words[start:]:
            t = w.strip("'-")
            if "'" not in w and t and t not in stops and not kit.is_link(t):
                fam = t
                break
        if not fam:
            return None
        q = brand + " " + fam
        return None if s.seeker.already_ran(q) else self._with_text(q, tp)

    # ------------------------------------------------------------------ 3.13 rows worth opening
    def _open_rows(self, slot, keys, most, use_bar=True):
        s, book = self.s, self._book()
        shelf = self._lows(self.tour_shelf(slot))
        scores = {k: self.tour_rating(k, slot) for k in book if self.walkable(k)}
        top = max(scores.values()) if scores else 0
        bare = top == 0 and self.bounds_only(slot)
        shown = []
        for k, e in book.items():
            if k in keys or (k in scores and ((top and scores[k] == top) or (bare and self._onshelf(e, shelf)))):
                for lid, lab in (e.get("row_shelf") or {}).items():
                    if kit.low(lab or "").strip() in shelf and lid not in shown:
                        shown.append(lid)
        pk = self.pick_of(slot)
        bar = pk[1] if pk else None
        if bar is None and use_bar:
            ps = [p for p in (self._price(l) for l in shown if l in s.archive.read) if p is not None]
            bar = min(ps) if ps else None
        units, cap, rows = self._units(slot), self._cap(), set(shown)
        gone = self._marks(lambda d: any("id" in kit.low(k) and isinstance(v, str) and v in rows
                                         for k, v in d.items()))
        out = []
        for lid in shown:
            p = self._price(lid)
            if lid in s.archive.read or lid in s.refused_lines or gone.get(lid) or p is None:
                continue
            if (cap is not None and p * units > cap + 1e-9) or (bar is not None and not p < bar - 1e-9):
                continue
            out.append((p, lid))
        out.sort()
        return [l for _, l in out[:most]]

    # ------------------------------------------------------------------ 3.14 page digest, code-run search call
    def page_hash(self, obs):
        try:
            known, cells = self.s.archive.known, []
            def fn(d):
                if len(cells) >= self.DIGEST_ROWS or len(d) <= 3:
                    return
                lid = None
                for k, v in d.items():
                    if ("id" in kit.low(k) and (isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)))
                            and str(v) in known):
                        lid = str(v)
                        break
                if lid is None:
                    return
                price = "?"
                for k, v in d.items():
                    if _p10_PRICE.search(kit.low(k)) and kit.is_num(v):
                        price = self._num(v)
                        break
                gone = " SOLD OUT" if kit.out_of_stock(d) else ""
                cells.append("%s %s%s '%s' [%s]" % (lid, price, gone, str(kit.row_title(d) or "")[:self.TITLE_CUT],
                                                    kit.shelf_field(d) or ""))
            kit.visit(obs, fn, top=3)
            return self.HEAD + ("; ".join(cells) if cells else "none")
        except Exception as e:
            self._fail("page_hash", e)
            return ""

    def search_call(self, args, say, tool=None):
        s, cid = self.s, None
        try:
            tool = tool or s.role("search")
            if not tool or not isinstance(args, dict):
                return False
            cid = "walk_%d_%d" % (len(s.chat), s.link.steps)
            call = s.engine.call(tool, args, cid)
            cid = call.get("id") or cid
            before = len(s.archive.obs)
            s.covering += 1
            try:
                pairs = s.engine.dispatch(say, [call]) or []
            finally:
                s.covering -= 1
            text = next((t for c, t in pairs if c == cid), None)
            if text is None and pairs:
                text = pairs[0][1]
            if text is None:
                s.engine.send_replies([(cid, "Not run.")], status=False)
                return False
            grew = len(s.archive.obs) > before
            if grew:
                body, mark = self.page_hash(s.archive.obs[-1]), "\n" + Tune.Marks.NOTE
                i = str(text).find(mark)
                if i >= 0:
                    note = str(text)[i + len(mark):].strip()
                    # flag 6: our own note texts are sniffed for shopper / meaning / refusal news
                    if "shopper" in note or Tune.Marks.MEANING.strip() in note or Tune.Marks.REFUSED in note:
                        body += mark + " " + note
                text = body
            s.engine.send_replies([(cid, str(text))], status=False)
            return grew
        except Exception as e:
            self._fail("search_call", e)
            try:
                m = s.chat[-1] if s.chat else None
                if cid and m and m.get("role") == "assistant" and any(
                        (c or {}).get("id") == cid for c in (m.get("tool_calls") or [])):
                    s.engine.send_replies([(cid, "Not run.")], status=False)
            except Exception as e2:
                self._fail("search_call_close", e2)
            return False

    # ------------------------------------------------------------------ 3.15 run the walk
    @staticmethod
    def _per_slot(x):
        if not x:
            return {}
        if isinstance(x, dict):
            if all(k is None or (isinstance(k, int) and not isinstance(k, bool)) for k in x):
                items = list(x.items())
            else:
                items = [(None, x)]
        elif all(isinstance(t, (list, tuple)) and len(t) == 2 and isinstance(t[1], (dict, list)) for t in x):
            items = [tuple(t) for t in x]
        else:
            items = [(None, list(x))]
        out = {}
        for k, v in items:
            v = [v] if isinstance(v, dict) else [a for a in (v or []) if isinstance(a, dict)]
            if v:
                out.setdefault(k, []).extend(v)
        return out

    def page_stage(self, plan, floor):
        try:
            return self._pages(list(plan or []), floor)
        except Exception as e:
            self._fail("page_stage", e)
            return []

    def _pages(self, plan, floor):
        sk, out = self.s.seeker, []
        how = "ranks results by match" if self.ranked() else "sorts results by id"
        for ln in plan:
            if self._short(floor + 1, True):
                break
            slot, key, want = ln[0], ln[1], ln[2]
            self.walked.add(slot)
            e = self._book().get(key)
            if not e:
                continue
            q = e.get("text") or self._query(e.get("args") or {})
            say = ("The lowest price is wanted, but this store %s instead of by price, so the code keeps paging "
                   "through '%s'." % (how, q))
            n = 0
            while n < want and not self._short(floor + 1, True):
                spot = sk.following_page(key)
                if spot is None or not self.search_call(sk.page_call(key, spot), say):
                    break
                n += 1
            if n:
                e = self._book().get(key) or e
                out.append("'%s' paged %d further time%s; rows now seen %d of %d"
                           % (q, n, "" if n == 1 else "s", len(e.get("row_shelf") or {}), e.get("total") or 0))
        return out

    def do_tour(self, plan, floor, ceilings=(), repair=False, twins=None, rungs=None):
        try:
            return self._walk(list(plan or []), floor, list(ceilings or []), repair, self._per_slot(twins),
                              self._per_slot(rungs))
        except Exception as e:
            self._fail("do_tour", e)
            return False

    def _walk(self, plan, floor, ceilings, repair, twins, rungs):
        s, sk = self.s, self.s.seeker
        if not (plan or ceilings or twins or rungs) or not sk.paging() or not s.role("search"):
            return False
        reports, ran = [], False
        rung_keys, twin_keys, fam_keys, ceil_rows = {}, {}, {}, {}
        calls, m2 = 0, floor + 2
        for slot, lst in rungs.items():  # 1. figure rungs
            if calls >= Tune.Search.RUNG_CALLS or self._short(m2):
                break
            for args in lst:
                if calls >= Tune.Search.RUNG_CALLS or self._short(m2):
                    break
                q = self._query(args)
                calls += 1
                if not self.search_call(args, "Only a figure limits this product, so the code searches the whole "
                                              "figures just past the limit ('%s')." % q):
                    continue
                ran, key = True, sk.book_key(args)
                rung_keys.setdefault(slot, []).append(key)
                while calls < Tune.Search.RUNG_CALLS:
                    e = self._book().get(key)
                    if not e or len(e.get("spots") or ()) >= Tune.Search.RUNG_PAGES or self._short(m2):
                        break
                    spot = sk.following_page(key)
                    if spot is None:
                        break
                    calls += 1
                    if not self.search_call(sk.page_call(key, spot), "Reading that figure search to its end."):
                        break
                e = self._book().get(key) or {}
                reports.append("the figure search '%s' (%d rows seen)" % (q, len(e.get("row_shelf") or {})))
        for slot, lst in twins.items():  # 2. spelling twins
            for args in lst:
                if self._short(m2):
                    break
                q = self._query(args)
                if self.search_call(args, "This store spells a required value two ways, so the code also "
                                          "searches '%s'." % q):
                    ran, twin_keys[slot] = True, sk.book_key(args)
                    reports.append("the other spelling '%s'" % q)
        rep = self._pages(plan, floor)  # 3. pages
        if rep:
            ran = True
            reports += rep
        fslots = []
        for ln in plan:
            if ln[0] not in fslots:
                fslots.append(ln[0])
        for slot in fslots:  # 4. product family
            if self._short(m2):
                break
            args = self._line_search(slot)
            if not args:
                continue
            q = self._query(args)
            if not self.search_call(args, "The code searches the pick's own product family (maker plus family "
                                          "name) '%s': sibling sizes or colours are listings of their own and may "
                                          "cost less." % q):
                continue
            ran, key = True, sk.book_key(args)
            fam_keys[slot] = key
            for _ in range(2):
                e = self._book().get(key)
                if not e or math.ceil((e.get("total") or 0) / max(1, e.get("size") or 0)) > 3:
                    break
                spot = sk.following_page(key)
                if (spot is None or s.steps_left() <= floor + 3 or s.link.done
                        or s.link.time_left() <= self.HURRY):
                    break
                if not self.search_call(sk.page_call(key, spot), "Going on to the line search's next page."):
                    break
            reports.append("the pick's line search '%s'" % q)
        cm = (floor if repair else max(floor, s.intake.due_margin() + 2)) + 1
        cs = []
        for slot in ceilings:
            if slot not in cs:
                cs.append(slot)
        for slot in cs:  # 5. price-cap ceiling
            if self._short(cm):
                break
            res = s.pricer.cap_tour(slot, floor, repair)
            if res:
                ran = True
                if res[0]:
                    reports.append(str(res[0]))
                ceil_rows[slot] = list(res[1] or [])
        if not ran:
            return False
        bs, size = self._batch()  # 6. queue the rows to open
        most = size if bs else (3 if any(ceil_rows.values()) else 2)
        order = fslots + [t for t in twins if t not in fslots]
        pools = [self._open_rows(k, [x for x in (fam_keys.get(k), twin_keys.get(k)) if x], most) for k in order]
        for k in rungs:
            if k not in order:
                pools.append(self._open_rows(k, rung_keys.get(k, []), min(most, Tune.Search.RUNG_OPENS), False))
        pools += [ceil_rows.get(k) or [] for k in cs]
        queue = self._trim(self._deal(pools, most), floor, bs)
        homes = set()  # 7. close
        for k in set(fslots) | set(cs) | set(twins) | set(rungs):
            homes |= self._lows(self.tour_shelf(k))
        s.rank_marks.difference_update({mk for mk in s.rank_marks if self._label(str(mk).split("|")[0]) in homes})
        s.order_vetted = False
        text = "[cheapest walk] " + "; ".join(reports) + "."
        if queue:
            text += (" Opened now, cheapest first: " + ", ".join("%s (%s)" % (l, self._num(self._price(l)))
                                                                 for l in queue)
                     + " - the unread rows of these searches that were in stock, affordable and priced under each "
                       "pick.")
        else:
            text += " None of the rows these searches list is unread, in stock, affordable and priced under the pick."
        text += (" Settle each product on its least expensive listing whose own fields meet every rule that applies;"
                 " ranking is checked again at each cart move and at the order.")
        opened = False
        if queue:
            s.judge.queue_open(queue, "walk", recheck=True, floor=floor,
                               say="Since the lowest price is wanted, the code now opens the cheapest unread rows "
                                   "the walk turned up.")
            opened = bool(s.judge.open_queued())
        self._attach(text, not opened)
        s.link.note("walk", plan=[[ln[0], ln[2]] for ln in plan], ceilings=cs, opened=list(queue))
        return True

    # ------------------------------------------------------------------ 3.16 what the decline relies on
    def refusal_matches(self):
        s = self.s
        try:
            struck = self._struck()
            fits = self._cleared() | {f for f in (s.audit_fits or []) if f not in struck}
            return fits - set(s.refused_lines)
        except Exception as e:
            self._fail("refusal_matches", e)
            return set()

    def lacking(self):
        s = self.s
        try:
            labs = {self._label(f) for f in self.refusal_matches()} - {""}
            if s.multi():
                return [k for k in range(self._nprod()) if self._home(k) and not (self._home(k) & labs)]
            return [] if self._lows(s.seeker.wanted_shelves()) & labs else [None]
        except Exception as e:
            self._fail("lacking", e)
            return []

    def _decline_walks(self):
        lines = self.s.ledger.in_force() or []
        plain = {i for i, l in enumerate(lines) if not kit.is_pointer(str(l))}  # flag 2: shared reference test
        out = []
        for slot in self.lacking():
            best = None
            for left, key, e in self.tour_choices(slot):
                got = self._carry(e.get("args") or {}, slot)
                n = len(got & plain)
                if not n or key in out:
                    continue
                r = (-n, -len(got), left, key)
                if best is None or r < best:
                    best = r
            if best:
                out.append(best[3])
            if len(out) >= 2:
                break
        return out[:2]

    # ------------------------------------------------------------------ 3.17 row pools, over-budget rows
    def _census(self):
        known, labels, gone = self.s.archive.known, {}, {}
        def fn(d):
            lid = kit.identity(d)
            if lid not in known:
                return
            lab = kit.shelf_field(d)
            if lab:
                labels[lid] = lab
                if any(_p10_STOCK.search(kit.low(k)) for k in d):
                    gone[lid] = bool(kit.out_of_stock(d))
        for o in self.s.archive.obs:
            kit.visit(o, fn, top=self.DEEP)
        return labels, gone

    def _pools(self, wanted, only=None, ceilings=None):
        s = self.s
        labels, gone = self._census()
        cap, res = self._cap(), []
        for i, (shelf, units) in enumerate(wanted):
            sh = self._lows(shelf)
            ce = ceilings[i] if ceilings and i < len(ceilings) else None
            rows = []
            for lid, lab in labels.items():
                if (kit.low(lab).strip() not in sh or lid in s.archive.read or lid in s.refused_lines
                        or gone.get(lid) or (only is not None and lid not in only)):
                    continue
                p = self._price(lid)
                if p is None or (cap is not None and p * units > cap + 1e-9) or (ce is not None and not p < ce - 1e-9):
                    continue
                rows.append((p, lid))
            rows.sort()
            res.append([l for _, l in rows])
        return res

    def row_groups(self, wanted=None, only=None, ceilings=None):
        """slot -> unread in-stock affordable ids, cheapest first (default: every product on its walk home)."""
        try:
            if wanted is None:
                slots = self.s.ledger.slots() or [None]
                pools = self._pools([(self.tour_shelf(k), self._units(k)) for k in slots], only, ceilings)
                return dict(zip(slots, pools))
            return dict(enumerate(self._pools(list(wanted), only, ceilings)))
        except Exception as e:
            self._fail("row_groups", e)
            return {}

    def pricey_rows(self):
        s = self.s
        try:
            cap = self._cap()
            if not s.multi() or s.ledger.priority() == "cheapest" or cap is None:
                return []
            fits, lows, n = self.refusal_matches(), [], self._nprod()
            for k in range(n):
                h = self._home(k)
                if not h:
                    return []
                ps = [p for p in (self._price(f) for f in fits if self._label(f) in h) if p is not None]
                if not ps:
                    return []
                lows.append(min(ps))
            weights = [lows[k] * self._units(k) for k in range(n)]
            if sum(weights) <= cap + 1e-9:
                return []
            pools = self._pools([(self._home(k), self._units(k)) for k in range(n)], None, lows)
            pools = [pools[k] for k in sorted(range(n), key=lambda k: -weights[k])]
            return pools if any(pools) else []
        except Exception as e:
            self._fail("pricey_rows", e)
            return []

    # ------------------------------------------------------------------ 3.18 decline reads
    def queue_decline_reads(self):
        s = self.s
        try:
            if self._dr_done or s.link.time_left() < self.QUEUE_CLOCK:
                return ""
            bs, _ = self._batch()
            if (not s.role("details") and not bs) or s.basket.units:
                return ""
            spend, left = self.KEEP + (1 if bs else 2), s.steps_left()
            room = min(self.DR_PAGES, left - spend)
            keys = (self._decline_walks() if room > 0 and s.seeker.paging() and s.role("search") and self.id_sorted()
                    else [])
            cheaper = self.pricey_rows() if left > spend else []
            if not keys and not cheaper:
                return ""
            self._dr_done = True
            self._job = (keys, room, self.KEEP)
            parts = ["The decline is held while the code reads a little more."]
            if keys:
                book = self._book()
                qs = ", ".join("'%s'" % ((book.get(k) or {}).get("text") or "") for k in keys)
                parts.append("[pages] The searches %s hold a required value and still have unread result pages. This "
                             "shop sorts rows by id, never by match or by price, so a good row is as likely to sit "
                             "past the pages seen; the code now fetches up to %d more page%s and opens the "
                             "lowest-priced affordable in-stock rows on them." % (qs, room, "" if room == 1 else "s"))
            if cheaper:
                parts.append("[cheaper rows] The cheapest fits of the products add up to more than the budget, so "
                             "unopened in-stock rows priced under each product's pick get opened too.")
            parts.append("Then weigh each record on sense against the active requirement list and buy %s; otherwise "
                         "decline." % ("one passing listing per product, in the requested quantities"
                                       if s.multi() else "the item if one passes"))
            return " ".join(parts)
        except Exception as e:
            self._fail("queue_decline_reads", e)
            return ""

    def do_refusal_reads(self):
        s = self.s
        try:
            job, self._job = self._job, None
            if not job:
                return False
            keys, room, keep = job
            sk = s.seeker
            bs, size = self._batch()
            cost, read, walked, ok = (1 if bs else 2), 0, [], True
            if sk.paging():
                for key in keys:
                    e = self._book().get(key)
                    if not e:
                        continue
                    walked.append(key)
                    say = ("Paging on through '%s' first: this shop sorts by id, so a match may come later."
                           % (e.get("text") or ""))
                    while (ok and read < room and s.steps_left() > keep + cost and not s.link.done
                           and not s.link.broke and s.link.time_left() > self.HURRY):
                        spot = sk.following_page(key)
                        if spot is None:
                            break
                        ok = self.search_call(sk.page_call(key, spot), say)
                        if ok:
                            read += 1
            pools = []
            if walked:
                seen = set()
                for key in walked:
                    seen |= set((self._book().get(key) or {}).get("row_shelf") or {})
                pools += self._pools([(self.tour_shelf(k), self._units(k)) for k in self.lacking()], seen)
            pools += self.pricey_rows()
            queue = self._deal(pools, size if bs else 2)
            while queue and (1 if bs and len(queue) >= 2 else len(queue)) >= s.steps_left() - keep:
                queue.pop()
            s.link.note("decline_reads", pages=read, ids=list(queue))
            if queue and not s.link.done:
                s.judge.queue_open(queue, "decline", floor=keep,
                                   say="Ahead of a decline, the code opens the cheapest affordable rows never read, "
                                       "found on the new pages and among the picks.")
                s.judge.open_queued()
            else:
                self._attach("[decline reads] Nothing worth opening came of it: no in-stock row within the cap.", True)
            return read > 0 or bool(queue)
        except Exception as e:
            self._fail("do_refusal_reads", e)
            return False



# ------------------------------------------------------------ p11_tracer.py
class Tracer:
    """Reference resolver (spec 09): requirements whose value is read off another listing in the store."""
    _NEG = re.compile(r"(?i)\b(?:not|no|never|nor|without)\b(?:(?:\s+|-)[A-Za-z0-9.:/&'-]+){1,3}")
    _WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
    _CUT4 = re.compile(r"(?<=[.!?;])\s+|\n+")
    _CUT3 = re.compile(r"(?<=[.!?])\s+|\n+")
    _LINKY = re.compile(r"[A-Za-z][\w+.-]*://\S*|\S*/\S*")

    def __init__(self, s):
        self.s = s
        self._tally = 0           # store steps spent by resolver reads and searches
        self._checks = 0          # hop-check calls made
        self._read_ids = set()    # ids fetched for references or shown in a hop check
        self._tells = []          # tells waiting for the next status note
        self._stamps = set()      # "lid|version" redirect remarks already made
        self._wait_key = None
        self._key = None
        self._memo = {}

    # ------------------------------------------------------------------ small helpers
    def _err(self, e):
        try:
            self.s.link.fault("tracer_error", e)
        except Exception:
            pass

    def _note(self, kind, **kw):
        try:
            self.s.link.note(kind, **kw)
        except Exception:
            pass

    @staticmethod
    def _tok(text):
        return set(kit.ref_tokens(str(text or "")))

    @staticmethod
    def _uniq(items):
        out = []
        for x in items:
            if x not in out:
                out.append(x)
        return out

    @staticmethod
    def _cut(line):
        front, prop, tag, value = kit.head_split(str(line or ""))
        return str(front or "").strip(), str(prop or "").strip(), str(tag or "").strip(), str(value or "")

    def _label(self, line):
        return self._cut(line)[1]

    @staticmethod
    def _desc(t):
        d = "the " + (str(t.get("kind") or "").strip() or "listing") + " sold by " + str(t.get("seller") or "")
        if t.get("details"):
            d += " (" + "; ".join(t["details"]) + ")"
        return d

    def _tkey(self, t):
        toks = set()
        for d in t.get("details") or []:
            toks |= self._tok(d)
        return kit.name_key(str(t.get("seller") or "")), tuple(sorted(toks))

    @staticmethod
    def _says(text, name):
        n = str(name or "").strip().lower()
        return bool(n) and bool(re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?:'s)?(?![a-z0-9])", str(text or "").lower()))

    @staticmethod
    def _negator(w):
        return bool(re.fullmatch(r"not|no|never|nor|neither|\w*n't", str(w).lower()))

    def _has_neg(self, text):
        return any(self._negator(w) for w in self._WORD.findall(str(text or "")))

    @staticmethod
    def _prose(v):
        return isinstance(v, str) and (len(v.split()) >= 6 or bool(re.search(r"[A-Za-z)][.!?;](?:\s|$)", v)))

    def _display(self, v, title, shelf):
        # flag 7.2: display fields by value shape (the title, the shelf label, a link) instead of name cues
        t = str(v).strip()
        return bool(t) and (t == title.strip() or t.lower() == shelf or bool(self._LINKY.fullmatch(t)))

    @staticmethod
    def _first_span(text, pred):
        """(start, end) of the earliest phrase the predicate finds in text, on chunk bounds; None when absent."""
        spans = [m.span() for m in re.finditer(r"\S+", text)]
        for k in range(len(spans)):
            end = spans[k][1]
            if pred(text[:end]):
                for i in range(k, -1, -1):
                    if pred(text[spans[i][0]:end]):
                        return spans[i][0], end
                return 0, end
        return None

    def _named(self, prop, passage):
        """Passage test (09 5.3): every property word of 3+ letters is named (whole word or shared 4-letter start)."""
        pw = [w.lower() for w in kit.property_terms(str(prop or "")) if len(w) >= 3]
        if not pw:
            return True
        tw = re.findall(r"[a-z]+", str(passage or "").lower())
        st, p4 = set(tw), {w[:4] for w in tw if len(w) >= 4}
        return all(w in st or (len(w) >= 4 and w[:4] in p4) for w in pw)

    def _stats(self, tool):
        try:
            return self.s.judge.tool_stats(tool) or {} if tool else {}
        except Exception:
            return {}

    def _opens(self):
        return [r for r in self.s.refs if r.get("status") == "open"]

    def _nprod(self):
        return len(self.s.ledger.products()) or 1

    def _slot_of(self, r):
        return self.s.mender.item_for(r["tag"]) if r.get("tag") else None

    # ------------------------------------------------------------------ cached record views
    def _m(self):
        s = self.s
        known = frozenset(k for r in s.refs for k in (kit.name_key(str(r["target"].get("seller") or "")),
                                                       kit.name_key(str(r["first_target"].get("seller") or ""))) if k)
        key = (len(s.archive.obs), len(s.archive.known), len(s.archive.read), known)
        if key != self._key:
            self._key, self._memo = key, {}
        return self._memo

    @staticmethod
    def _lid(d):
        v = d.get("id")
        if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip():
            return str(v).strip()
        for k, v in d.items():
            if str(k).lower().endswith("id") and isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip():
                return str(v).strip()
        return ""

    def _index(self):
        m = self._m()
        if "ix" not in m:
            known, ix, last = self.s.archive.known, {}, {}
            for n, ob in enumerate(self.s.archive.obs):
                def see(d, n=n):
                    lid = self._lid(d)
                    if not lid or lid not in known:
                        return
                    last[lid] = n
                    if len(d) > 3 or any(isinstance(v, dict) and len(v) > 1 for v in d.values()):
                        ix.setdefault(lid, {}).update(d)
                kit.visit(ob, see, 6)
            m["ix"] = (ix, last)
        return m["ix"]

    def _rec(self, lid):
        return self._index()[0].get(lid) or {}

    def _title(self, rec):
        try:
            return str(kit.row_title(rec) or "")
        except Exception:
            return ""

    def _shelf(self, lid):
        rec = self._rec(lid)
        try:
            return str(self.s.archive.record_shelf(rec) or "").strip().lower()
        except Exception:
            return kit.shelf_field(rec)

    def _row_names(self, lid):
        m = self._m()
        if "rk" not in m:
            try:
                m["rk"] = set(self.s.judge.row_keys() or ())
            except Exception:
                m["rk"] = set()
        c = m.setdefault("rf", {})
        if lid not in c:
            try:
                rf = set(self.s.judge.row_fields(lid) or ())
            except Exception:
                rf = set()
            c[lid] = rf or m["rk"]
        return c[lid]

    def _texts(self, lid):
        """(note text, about text) of a record (09 3.5.3); flag 7.2: a long text the search rows also show is
        description-like ("about"), one they never show is a note."""
        c = self._m().setdefault("tx", {})
        if lid not in c:
            rec = self._rec(lid)
            title, shelf, rows = self._title(rec), kit.shelf_field(rec), self._row_names(lid)
            note, about = [], []
            for k, v in rec.items():
                if not isinstance(v, str) or len(v.split()) < 3 or str(k).lower().endswith("id") or self._display(v, title, shelf):
                    continue
                (about if k in rows else note).append(v.strip())
            c[lid] = ("\n".join(note), "\n".join(about))
        return c[lid]

    def _notes(self, lid, fb=False):
        note, about = self._texts(lid)
        return note or (about if fb else "")

    def _seller_keys(self, lid):
        rec, out = self._rec(lid), []
        learned = self._learned()
        for k, v in rec.items():
            if not isinstance(v, str) or "id" in str(k).lower():
                continue
            t = v.strip()
            if not t or len(t) > 60 or not (kit.is_seller_field(str(k)) or k in learned):
                continue
            if len(t.split()) > 6 or re.search(r"[A-Za-z][.!?;]\s", t):   # note-like by shape (flag 7.2)
                continue
            out.append((k, t))
        return out

    def _sellers(self, lid):
        c = self._m().setdefault("sel", {})
        if lid not in c:
            c[lid] = {kit.name_key(t) for _, t in self._seller_keys(lid)} - {""}
        return c[lid]

    def _seller_map(self):
        m = self._m()
        if "sm" not in m:
            sm = {}
            for lid in self._index()[0]:
                for sq in self._sellers(lid):
                    sm.setdefault(sq, set()).add(lid)
            m["sm"] = sm
        return m["sm"]

    def _learned(self):
        m = self._m()
        if "lf" in m:
            return m["lf"]
        m["lf"] = set()
        try:
            known, ix, heard, out = set(self._key[3]), self._index()[0], set(), set()
            for lid, rec in ix.items():
                notes = self._notes(lid)
                if not notes:
                    continue
                nest = [v for v in rec.values() if isinstance(v, dict)]
                pv = {kit.name_key(x) for d in (nest or [rec]) for x in d.values() if isinstance(x, str)}
                for sent in self._CUT4.split(notes):
                    run = []
                    for w in self._WORD.findall(sent)[1:] + [""]:
                        if w and w[0].isupper() and any(ch.islower() for ch in w):
                            run.append(w)
                            continue
                        for i in range(len(run)):
                            for j in range(i + 1, min(i + 3, len(run)) + 1):
                                nm = kit.name_key(" ".join(run[i:j]))
                                if len(nm) > 3 and nm not in pv:
                                    heard.add(nm)
                        run = []
            for lid, rec in ix.items():
                title, shelf = self._title(rec), kit.shelf_field(rec)
                for k, v in rec.items():
                    if not isinstance(v, str) or "id" in str(k).lower():
                        continue
                    t = v.strip()
                    if not 1 <= len(t) <= 60 or self._display(t, title, shelf) or self._prose(t):
                        continue
                    sq = kit.name_key(t)
                    if sq in known or (t[:1].isupper() and len(t.split()) < 4 and sq in heard):
                        out.add(k)
            m["lf"] = out
        except Exception:
            m["lf"] = set()
        return m["lf"]

    def _maps(self, lid):
        """Property maps of a record: its nested dicts, else one flat map of plain fields (09 3.5.4)."""
        c = self._m().setdefault("maps", {})
        if lid not in c:
            rec = self._rec(lid)
            nest = [(str(k), v) for k, v in rec.items() if isinstance(v, dict)]
            if not nest:
                title, sk = self._title(rec).strip(), {k for k, _ in self._seller_keys(lid)}
                flat = {}
                for k, v in rec.items():
                    kl = str(k).lower()
                    if not kit.is_plain(v) or kl.endswith("id") or k in sk or re.search(r"price|cost", kl):
                        continue
                    if isinstance(v, str) and ((title and v.strip() == title) or (len(v.split()) >= 3 and self._prose(v))):
                        continue
                    flat[k] = v
                nest = [("fields", flat)] if flat else []
            c[lid] = nest
        return c[lid]

    def _map_values(self, lid):
        return [str(v) for _, mp in self._maps(lid) for v in mp.values() if kit.is_plain(v)]

    def _words(self, lid):
        """(field-name tokens, shown tokens) of a record (09 3.5.4)."""
        c = self._m().setdefault("wd", {})
        if lid not in c:
            rec = self._rec(lid)
            names = [str(k) for k in rec] + [str(k) for _, mp in self._maps(lid) for k in mp]
            nt = set()
            for k in names:
                nt |= self._tok(kit.uncamel(k).replace("_", " "))
            st = self._tok(" ".join([self._title(rec), self._NEG.sub(" ", self._notes(lid))] + self._map_values(lid)))
            for t in list(st):
                g = re.fullmatch(r"[\d.,:/]+([a-z]{3,})", t)
                if g:
                    st.add(g.group(1))
            c[lid] = (nt, st)
        return c[lid]

    def _facts(self, lid):
        c = self._m().setdefault("fa", {})
        if lid not in c:
            rec = self._rec(lid)
            c[lid] = self._words(lid)[0] | self._tok(" ".join([self._title(rec), kit.shelf_field(rec)] + self._map_values(lid)))
        return c[lid]

    def _store_fields(self):
        m = self._m()
        if "sf" not in m:
            out = set()

            def see(d):
                for v in d.values():
                    if isinstance(v, dict) and sum(1 for x in v.values() if kit.is_plain(x)) >= 2:
                        for k in v:
                            out.add(" ".join(re.findall(r"[a-z0-9]+", kit.uncamel(str(k)).lower())))
            for ob in self.s.archive.obs:
                kit.visit(ob, see)
            out.discard("")
            m["sf"] = out
        return m["sf"]

    @staticmethod
    def _num(x):
        try:
            return float(str(x).replace(",", ""))
        except Exception:
            return None

    def _shows(self, lid, detail):
        """Does the record show a target detail (09 3.5.5)."""
        nt, st = self._words(lid)
        vw = [t for t in kit.ref_tokens(str(detail)) if len(t) >= 3 and not re.search(r"\d", t)]
        want = [t for t in vw if t not in nt]
        if not want and len(vw) >= 2:
            best = None
            for _, mp in self._maps(lid):
                for k in mp:
                    kt = [t for t in kit.ref_tokens(kit.uncamel(str(k)).replace("_", " ")) if len(t) >= 3 and not re.search(r"\d", t)]
                    if kt and vw[:len(kt)] == kt and (best is None or len(kt) > len(best)):
                        best = kt
            want = vw[len(best):] if best else []
        if want and not any(t in st for t in want):
            return False
        figs = kit.ref_figures(str(detail))
        if figs:
            have = kit.ref_figures(" ".join([self._title(self._rec(lid))] + self._map_values(lid)))
            for num, unit in figs:
                same = [self._num(n) for n, u in have if u == unit]
                if same and self._num(num) not in same:
                    return False
        return True

    def _fully(self, lid, dets):
        nt, st = self._words(lid)
        allw = nt | st
        return sum(1 for d in dets if self._tok(d) and self._tok(d) <= allw)

    def _miss_words(self, lid, detail):
        nt, st = self._words(lid)
        want = [t for t in kit.ref_tokens(str(detail)) if len(t) >= 3 and not re.search(r"\d", t) and t not in nt]
        return bool(want) and not any(t in st for t in want)

    def _field(self, lid, prop):
        vals = [(str(k), str(v).strip()) for _, mp in self._maps(lid) for k, v in mp.items() if kit.is_plain(v) and str(v).strip()]
        for k, v in vals:
            if kit.exact_property(k, prop):
                return v
        for k, v in vals:
            if kit.property_match(k, prop):
                return v
        return ""

    def _mention(self, word, lid, strict):
        """Is a capitalised word a mention of another name (09 3.14)."""
        sq = kit.name_key(word)
        if not sq:
            return False
        if any(o != lid for o in self._seller_map().get(sq, ())):
            return True
        if not word[:1].isupper() or len(sq) < 4 or sq in self._sellers(lid):
            return False
        facts = self._facts(lid)
        if strict and any(t in facts for t in kit.ref_tokens(word)):
            return False
        return sq not in facts

    # ------------------------------------------------------------------ opening references
    def _vocab(self):
        led, words = self.s.ledger, set()
        names = [p.get("name", "") for p in led.products()] or [led.product_name()]
        for n in names:
            words |= kit.kind_terms(str(n or ""))
        tool = self.s.role("search")
        if tool:
            for p in (self.s.schema(tool).get("properties") or {}).values():
                if isinstance(p, dict):
                    it = p.get("items") if isinstance(p.get("items"), dict) else {}
                    for e in list(p.get("enum") or []) + list(it.get("enum") or []):
                        words |= kit.kind_terms(str(e))
        return words

    def _seller_ok(self, name, text):
        n = str(name or "").strip()
        if len(re.findall(r"[A-Za-z]", n)) < 3 or not self._says(text, n):
            return False
        kw = kit.kind_terms(n)
        return bool(kw) and not kw <= self._vocab()

    def _shop_in(self, value, src):
        # flag 7.1: a listed reference whose value names a seller valid in the request counts as pointing
        return bool(kit.parse_listing(value, lambda n: self._seller_ok(n, src)))

    def open_from_request(self):
        try:
            led = self.s.ledger
            rd = led.reading if isinstance(led.reading, dict) else {}
            hints = [h for h in (rd.get("pointer_targets") or []) if isinstance(h, dict)]
            listed = [x for x in (rd.get("pointer_needs") or []) if isinstance(x, str)]
            src = str(self.s.request or "")
            for req in [x for x in (led.written or rd.get("musts") or []) if isinstance(x, str) and ":" in x]:
                val = req.split(":", 1)[1]
                if kit.is_pointer(val) or (req in listed and (kit.is_differ(val) or self._shop_in(val, src))):
                    self._open(req, hints, src, "request")
        except Exception as e:
            self._err(e)

    def take_news(self):
        try:
            led = self.s.ledger
            for r in self.s.refs:
                if r["status"] == "resolved" and r.get("settled"):
                    if any(isinstance(x, str) and x != r["settled"] and self.resolves_to(x, r) for x in led.in_force()):
                        self._swap(r)
            news = list(led.pointer_news or [])
            if isinstance(led.pointer_news, list):
                del led.pointer_news[:]
            else:
                led.pointer_news = []
            rd = led.reading
            for item in news:
                if not isinstance(item, dict):
                    continue
                need = str(item.get("need") or "")
                hints = [h for h in (item.get("refs") or []) if isinstance(h, dict)]
                ln = item.get("lines")
                src = " ".join(str(x) for x in ln) if isinstance(ln, list) else str(ln or "")
                if self._open(need, hints, src, "reply") and isinstance(rd, dict):
                    lst = rd.get("pointer_needs")
                    if not isinstance(lst, list):
                        lst = rd["pointer_needs"] = []
                    if need not in lst:
                        lst.append(need)
        except Exception as e:
            self._err(e)

    def _open(self, req, hints, src, origin):
        front, prop, tag, value = self._cut(req)
        if not prop or kit.is_junk_key(prop):
            return None
        tk = kit.kind_terms(tag)
        for r in self.s.refs:
            if r["text"] == req or (kit.property_match(r["prop"], prop) and kit.kind_terms(r["tag"]) == tk):
                return None
        multi, hint = self.s.multi(), None
        for h in hints:
            if not kit.property_match(str(h.get("property") or ""), prop):
                continue
            hp = str(h.get("product") or "").strip()
            if not tag or not hp or not multi or (kit.kind_terms(hp) & tk):
                hint = h
                break
        tgt = self._target(prop, value, hint, src)
        if not tgt:
            self._note("ref_skipped", text=req[:300], why="no listing named")
            return None
        rel, tol, unit = self._relation(value, hint, tgt, src, prop)
        r = {"text": req, "head": front, "prop": prop, "tag": tag, "relation": rel, "tolerance": tol, "unit": unit,
             "status": "open", "value": "", "partners": [], "settled": "", "chain": [], "origin": origin, "tried": set(),
             "target": tgt, "first_target": dict(tgt, details=list(tgt["details"])), "seen_keys": {self._tkey(tgt)},
             "ruled_out": set(), "extra_pages": 0, "hops": 0, "fails": 0, "unindexed": False, "_hits": {},
             "_loose": set(), "_noted": set(), "_sheeted": set(), "_due": None, "_late": None, "_cause": "", "_rule": ""}
        self.s.refs.append(r)
        self._note("ref_opened", text=req[:300], relation=rel, target=self._desc(tgt), origin=origin)
        return r

    def _target(self, prop, value, hint, src):
        hint = hint or {}
        hs, hk = str(hint.get("seller") or "").strip(), str(hint.get("kind") or "").strip()
        raw = hint.get("details") if isinstance(hint.get("details"), list) else []
        hd = [str(x).strip() for x in raw if kit.is_plain(x) and str(x).strip()]
        ok = lambda n: self._seller_ok(n, src)
        hit = None
        for txt in (kit.after_prop(src, prop), value):
            if txt:
                hit = kit.parse_listing(txt, ok)
                if hit:
                    break
        if hs and ok(hs):
            seller, kind, det = hs, hk, hd
            if not det and hit and kit.name_key(hit[0]) == kit.name_key(hs):
                det = list(hit[2] or [])
        elif hit:
            seller, kind, det = hit[0], hk or str(hit[1] or ""), hd or list(hit[2] or [])
        else:
            return None
        st = self._tok(src)
        det = self._uniq([str(d) for d in det if self._tok(d) & st])
        if not kind and hit:
            kind = str(hit[1] or "")
        return {"seller": seller, "kind": kind[:80], "details": det[:6]}

    def _relation(self, value, hint, tgt, src, prop):
        rel, amt, unit = kit.relation_of(value)
        told = kit.low((hint or {}).get("relation"))
        span = str(kit.after_prop(src, prop) or "")
        sel = str(tgt.get("seller") or "").lower()
        i = span.lower().find(sel) if sel else -1
        lead = span[:i] if i > 0 else span
        fs = self._first_span(lead, kit.is_pointer)
        if fs:
            lead = lead[:fs[1]]
        if rel == "pairs" or told == "pairs" or kit.is_pair(lead):
            return "pairs", None, ""
        if rel == "same" and hint and told == "within" and kit.is_num(hint.get("tolerance")):
            rel, amt, unit = "within", hint.get("tolerance"), str(hint.get("unit") or "")
        if rel != "same":
            return rel, amt, str(unit or "")
        # flag 7.1: the told relation is the reader's structured field; "differ" or a differ phrase marks it
        if "differ" in told or kit.is_differ(told) or (i > 0 and kit.is_differ(span[:i])):
            return "differ", None, ""
        return "same", None, ""

    # ------------------------------------------------------------------ which requirement, liveness
    def resolves_to(self, req, ref):
        try:
            req = str(req or "")
            bare = kit.strip_marks(req).strip()
            if req and (req in (ref.get("text"), ref.get("settled") or None) or bare in (ref.get("text"), ref.get("settled") or None)):
                return True
            front, prop, tag, value = self._cut(bare)
            if not (kit.is_pointer(value) or (ref.get("relation") == "differ" and kit.is_differ(value))):
                return False
            if not kit.property_match(prop, ref.get("prop") or ""):
                return False
            tags = kit.tags_of(front)
            if not tags or not ref.get("tag"):
                return True
            return bool(kit.kind_terms(tags[-1]) & kit.kind_terms(ref["tag"]))
        except Exception as e:
            self._err(e)
            return False

    def live(self, ref):
        try:
            led, p = self.s.ledger, ref.get("prop") or ""
            for x in led.in_force():
                if isinstance(x, str) and (self.resolves_to(x, ref) or kit.property_match(self._label(x), p)):
                    return True
            for x in led.dropped or []:
                if isinstance(x, str) and kit.property_match(self._label(x), p):
                    return False
            return True
        except Exception as e:
            self._err(e)
            return True

    def untracked(self):
        try:
            rd = self.s.ledger.reading if isinstance(self.s.ledger.reading, dict) else {}
            return [x for x in (rd.get("pointer_needs") or []) if isinstance(x, str) and x.strip() and not any(
                (r["status"] == "resolved" or r["chain"]) and self.resolves_to(x, r) for r in self.s.refs)]
        except Exception as e:
            self._err(e)
            return [x for x in ((self.s.ledger.reading or {}).get("pointer_needs") or []) if isinstance(x, str)]

    # ------------------------------------------------------------------ budgets
    def _solo(self):
        return not self.s.multi() and bool(self._opens()) and not self.s.trouble()

    def _chained(self):
        return any(r["chain"] for r in self._opens())

    def _floor(self):
        cap, due = self.s.step_cap, self.s.intake.due_margin()
        if self._solo():
            return 7
        if self._chained():
            return max(12, due + max(0, self._nprod() - 1))
        return max(12, min(10, cap // 4 + 1) + 5, due + 7)

    def has_room(self):
        try:
            return self._room()
        except Exception as e:
            self._err(e)
            return False

    def _room(self):
        s, L = self.s, self.s.link
        if L.done or L.broke or L.budget_gone():
            return False
        op = self._opens()
        differ = bool(op) and all(r["relation"] == "differ" for r in op)
        chained, solo = self._chained(), self._solo()
        if differ and not chained and self._tally >= Tune.Refs.DIFFER_BLIND:
            return False
        cap = s.step_cap
        if chained or solo:
            limit = cap
        else:
            limit = max(Tune.Refs.DIFFER_FLOOR if differ else Tune.Refs.STEP_FLOOR, int(Tune.Refs.STEP_SHARE * cap))
        margin = 10 if (chained or solo) else 60
        if self._tally >= limit or L.time_left() <= Tune.Clock.CLOSING + margin:
            return False
        if self._waiting():
            return False
        return s.steps_left() > self._floor() or self._hop_room()

    def _waiting(self):
        try:
            s = self.s
            if not s.multi():
                return False
            op = self._opens()
            if not op or not all(r["relation"] == "differ" and r["chain"] for r in op):
                return False
            if s.steps_left() <= 4 + self._floor():
                return False
            owners = [self._slot_of(r) for r in op]
            lacking = [k for k in s.ledger.slots() if not any(e.get("fit") for e in s.mender.pool(k))]
            foreign = [k for k in lacking if k not in owners]
            wait = len(foreign) > (1 if None in owners else 0)
            key = tuple(foreign) if wait else None
            if wait and key != self._wait_key:
                self._note("ref_wait", products=list(foreign))
            self._wait_key = key
            return wait
        except Exception as e:
            self._err(e)
            return False

    def _verified_others(self, r):
        try:
            s = self.s
            if not s.multi():
                return True
            ok = [any(e.get("verified") and e.get("fit") for e in s.mender.pool(j)) for j in s.ledger.slots()]
            k = self._slot_of(r)
            if k is not None:
                return all(v for j, v in enumerate(ok) if j != k)
            return sum(1 for v in ok if not v) <= 1
        except Exception:
            return False

    def _hop_room(self):
        if self.s.steps_left() - self.s.intake.due_margin() <= 2:
            return False
        for r in self._opens():
            t = r["target"]
            if r["hops"] < 1 or not (t.get("seller") and t.get("kind") and t.get("details")):
                continue
            if r.get("_late") is not None:
                if self._tally - r["_late"] < 3:
                    return True
                continue
            if not r["tried"] and self._verified_others(r):
                r["_late"] = self._tally
                self._note("ref_late_room", head=r["head"], tally=self._tally)
                return True
        return False

    def _can_check(self):
        L = self.s.link
        pad = 20 if self._chained() else 40
        return (self._checks < Tune.Refs.HOP_CHECKS and L.time_left() > Tune.Clock.CLOSING + pad and not L.budget_gone()
                and L.spend < Tune.Spend.CAP * L.scale)

    # ------------------------------------------------------------------ candidates and search planning
    def _cands(self, r):
        ix, last = self._index()
        t = r["target"]
        sq, kw = kit.name_key(str(t.get("seller") or "")), kit.kind_terms(str(t.get("kind") or ""))
        dts = [self._tok(d) for d in t.get("details") or []]
        rows, hits = [], {}
        for lid, rec in ix.items():
            if sq not in self._sellers(lid):
                continue
            tt = self._tok(self._title(rec) + " " + self._texts(lid)[1])
            h = sum(1 for d in dts if d and d <= tt)
            b = 2 if kw & kit.kind_terms(self._shelf(lid)) else 0
            hits[lid] = h
            rows.append((-(b + h), -last.get(lid, -1), lid))
        r["_hits"] = hits
        return [x[2] for x in sorted(rows)]

    def _template(self):
        m = self._m()
        if "tpl" not in m:
            try:
                m["tpl"] = self._make_template()
            except Exception as e:
                self._err(e)
                m["tpl"] = None
        return m["tpl"]

    def _make_template(self):
        s = self.s
        tool = s.role("search")
        if not tool:
            return None
        sch, sk = s.schema(tool), s.seeker
        req, props = set(sch.get("required") or []), sch.get("properties") or {}
        pname, cur = (sk.paging() or {}).get("name"), sk.cursor_param()
        labels = {self._shelf(x) for x in self._index()[0]} | {str(v).strip().lower() for v in s.archive.shelf_by_id.values()}
        labels.discard("")

        def fixed(args, slot):
            return {k: v for k, v in args.items() if k not in (slot, pname, cur)
                    and (k in req or not (isinstance(v, str) and v.strip().lower() in labels))}

        def gkey(fx, slot):
            return json.dumps([fx, slot], sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        groups = {}
        for e in sk.book().values():
            args = e.get("args") or {}
            ft = sk.open_text(args)
            if len(ft) != 1:
                continue
            slot = ft[0][0]
            fx = fixed(args, slot)
            g = groups.setdefault(gkey(fx, slot), {"fx": fx, "slot": slot, "calls": [], "at": len(groups)})
            stems = frozenset(kit.stem(w) for w in re.findall(r"[a-z0-9]+", str(ft[0][1]).lower()))
            g["calls"].append((stems, tuple(e.get("first_ids") or ())))

        def blind(g):
            cs = g["calls"]
            return any(len(a[1]) >= 2 and a[1] == b[1] and not (a[0] & b[0]) for i, a in enumerate(cs) for b in cs[i + 1:])
        good = [g for g in groups.values() if not blind(g)]
        if good:
            g = min(good, key=lambda g: (-len({c[0] for c in g["calls"]}), g["at"]))
            return dict(g["fx"]), g["slot"]
        blinds = {k for k, g in groups.items() if blind(g)}
        for ent in reversed(s.link.log):
            if not isinstance(ent, dict) or ent.get("kind") != "env":
                continue
            for a in reversed(ent.get("actions") or []):
                if not isinstance(a, dict) or a.get("name") != tool:
                    continue
                args = a.get("args", a.get("arguments"))
                args = args if isinstance(args, dict) else kit.args_from(args)
                if not isinstance(args, dict):
                    continue
                args = {k: v for k, v in args.items() if k != pname}
                ft = sk.open_text(args)
                if len(ft) != 1:
                    continue
                fx = fixed(args, ft[0][0])
                if gkey(fx, ft[0][0]) not in blinds:
                    return fx, ft[0][0]

        def is_text(p):
            t = p.get("type")
            return (t == "string" or (isinstance(t, list) and "string" in t)) and not p.get("enum")
        free = [k for k, p in props.items() if isinstance(p, dict) and is_text(p) and k not in (pname, cur)]
        if len(free) == 1 and not (req - {free[0]}):
            return {}, free[0]
        return None

    def _query_words(self, r):
        c = self._m().setdefault("qw", {})
        key = (id(r["target"]), r["hops"])
        if key in c:
            return c[key]
        led, t = self.s.ledger, r["target"]
        known = set()
        for lid in self._index()[0]:
            for _, mp in self._maps(lid):
                for k in mp:
                    known |= set(re.findall(r"[a-z]+", kit.uncamel(str(k)).lower()))
        for x in list(led.written or []) + list(led.revealed or []):
            known |= set(re.findall(r"[a-z]+", self._label(x).lower()))
        seller = str(t.get("seller") or "")
        own = self._tok(seller) | set(re.findall(r"[a-z0-9]+", seller.lower()))
        joined, figs, words = [], [], []
        dets = [str(d) for d in t.get("details") or []]
        for d in dets:
            after_num = False
            for chunk in d.split():
                bare = chunk.strip("()[]{}\"'.,;:!?").lower()
                for tk in re.findall(r"[a-z0-9]+(?:[.,][a-z0-9]+)*", bare):
                    if kit.is_link(tk) or tk in own or (after_num and not re.search(r"\d", tk)):
                        continue
                    if re.search(r"\d", tk) and re.search(r"[a-z]", tk):
                        joined.append(tk)
                    elif re.fullmatch(r"[\d.,]+", tk):
                        if len(tk) >= 2:
                            figs.append(tk)
                    elif tk.isalpha() and len(tk) > 2 and tk == bare and tk not in known and kit.stem(tk) not in known:
                        words.append(tk)
                after_num = bool(re.fullmatch(r"\d+(?:[.,]\d+)?", bare))
        words = sorted(self._uniq(words), key=len, reverse=True)
        dtoks = [[x for x in re.findall(r"[a-z0-9]+", d.lower()) if not kit.is_link(x)] for d in dets]
        every = bool(dtoks) and all(len(x) >= 2 for x in dtoks) and r["hops"] == 0
        fields, front = self._store_fields(), []
        for x in dtoks:
            if len(x) >= 2 and x[-1] in words and (every or any(" ".join(x[:k]) in fields for k in range(1, len(x)))):
                front.append(x[-1])
        out = self._uniq(joined + figs + self._uniq(front + words))
        c[key] = out
        return out

    def _drafts(self, r):
        t = r["target"]
        room = max(2, self.s.lead_words or 3)
        shop = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(t.get("seller") or ""))))
        nouns = [w for w in re.findall(r"[A-Za-z]+", str(t.get("kind") or "")) if len(w) >= 3 and not kit.is_link(w.lower())]
        qw = self._query_words(r)
        if r["unindexed"]:
            drafts = [" ".join((qw[:1] + nouns[-1:]))]
            drafts = [" ".join(drafts[0].split()[:room])]
        elif qw and len(shop.split()) < room:
            drafts = [shop + " " + qw[0], shop]
        else:
            drafts = [shop]
        return [d for d in drafts if d.strip() and d.lower() not in r["tried"]]

    def _more_page(self, r):
        sk = self.s.seeker
        if not sk.paging() or r["extra_pages"] >= 3:
            return None
        sq = kit.name_key(str(r["target"].get("seller") or ""))
        book = sk.book()
        for key in reversed(list(book)):
            if sq and sq in kit.name_key(str(book[key].get("text") or "")):
                spot = sk.following_page(key)
                if spot is not None:
                    return sk.page_call(key, spot)
        return None

    def _more(self, r, cands):
        rd = self.s.archive.read
        return any(c not in rd and c not in r["ruled_out"] for c in cands) or bool(self._drafts(r)) or self._more_page(r) is not None

    # ------------------------------------------------------------------ tools for notes and sheets
    def _notes_tool(self):
        j, s = self.s.judge, self.s
        t = j.memo_tool()
        if t:
            return t
        det = s.role("details")
        ds = self._stats(det)
        if det and ds.get("note_keys"):
            return None
        look = [x for x in j.query_tools() if x != det]
        rk, dk = set(j.row_keys() or ()), set(ds.get("keys") or ())
        pick = [x for x in look if set(self._stats(x).get("text_keys") or ()) - dk - rk]
        if pick:
            return max(pick, key=lambda x: (self._stats(x).get("calls") or 0, x))
        for x in (s.role("view"), j.supply_tool()):
            st = self._stats(x)
            if x and x in look and not (st.get("ids") or st.get("read_ids")):
                return x
        return None

    def _sheet_tool(self, lid):
        try:
            unused = [t for t in self.s.judge.query_tools() if lid not in (self._stats(t).get("read_ids") or ())]
            pm = [t for t in unused if self._stats(t).get("props")]
            if pm:
                return max(pm, key=lambda t: (self._stats(t).get("props") or 0, t))
            for t in (self.s.role("view"), self.s.role("details")):
                if t and t in unused:
                    return t
        except Exception as e:
            self._err(e)
        return None

    # ------------------------------------------------------------------ store fetches
    def _fetch(self, ids, say, tool=None, batch=None):
        s = self.s
        st0, n0 = s.link.steps, len(s.archive.obs)
        pairs = s.judge.scripted_read(ids, say, batch=batch, tool=tool)
        return pairs or [], st0, n0

    def _book_fetch(self, pairs, st0, ids, kind, **kw):
        s = self.s
        if pairs:
            s.engine.send_replies(pairs)
        spent = max(0, s.link.steps - st0)
        self._tally += spent
        self._read_ids.update(ids)
        self._note(kind, ids=list(ids), steps=spent, **kw)
        return spent > 0

    def _richer(self, obs, asked):
        try:
            rk, keys, deep, aset = set(self.s.judge.row_keys() or ()), set(), [False], set(asked)

            def see(d):
                if any(isinstance(v, str) and v in aset for v in d.values()):
                    keys.update(d.keys())
                    if any(isinstance(v, dict) and len(v) >= 2 for v in d.values()):
                        deep[0] = True
            for ob in obs:
                kit.visit(ob, see)
            return deep[0] or not rk or bool(keys - rk)
        except Exception:
            return True

    def _read(self, ids):
        s, m = self.s, self.s.mender
        heads = ", ".join(r["head"] for r in self._opens())
        say = "Opening listings that may be the page each pointer requirement takes its value from, as the shopper described it: " + heads + "."
        spec = m.batch_spec() if len(ids) >= 2 else None
        pairs, st0, n0 = self._fetch(ids, say, batch=spec)
        failed = False
        if spec and pairs and not s.batch_proven:
            if m.batch_records_back(pairs, ids) and self._richer(s.archive.obs[n0:], ids):
                s.batch_proven = True
            else:
                failed = True
                if m.batch_ids_echoed(pairs, ids) or len(ids) <= 2:
                    m.drop_batch()
                    self._note("ref_batch_off", ids=list(ids))
                else:
                    m.shrink_batch(ids)
        out = self._book_fetch(pairs, st0, ids, "ref_read")
        if not failed:
            for lid in ids:
                if lid not in s.archive.read:
                    for r in self._opens():
                        r["ruled_out"].add(lid)
        return out

    def _read_notes(self, r, ids):
        tool = self._notes_tool()
        if not tool:
            return False
        ids = ids[:2]
        r["_noted"].update(ids)
        say = ("Reading the remarks of " + ", ".join(ids) + " with " + tool + ": this page may be where '" + r["head"]
               + "' takes its value, and its remarks can pass the pointer on to yet another page.")
        pairs, st0, _ = self._fetch(ids, say, tool=tool)
        return self._book_fetch(pairs, st0, ids, "ref_notes", head=r["head"], tool=tool)

    def _sheet(self, r, lid):
        tool = self._sheet_tool(lid)
        if not tool or not self._room():
            return False
        say = ("Reading the property sheet of " + lid + " with " + tool + ": '" + r["head"] + "' takes its value from this page,"
               " and the record seen so far shows no such field.")
        pairs, st0, _ = self._fetch([lid], say, tool=tool)
        return self._book_fetch(pairs, st0, [lid], "ref_sheet", head=r["head"], tool=tool)

    def _rows_in(self, ob):
        known, rows = self.s.archive.known, []

        def see(d):
            lid = self._lid(d)
            if lid and lid in known and len(d) >= 4:
                rows.append(lid)
        kit.visit(ob, see, 3)
        return rows

    def _search(self, r, args, query):
        s = self.s
        tool = s.role("search")
        args, faults = ArgFixer.fix(dict(args), s.schema(tool))
        if faults:
            self._note("ref_search", head=r["head"], args=args, faults=list(faults)[:3], steps=0)
            return False
        say = "Searching the store for the very page '" + r["head"] + "' is pegged to [" + self._desc(r["target"]) + "]."
        st0, n0 = s.link.steps, len(s.archive.obs)
        s.walker.search_call(args, say)
        spent = max(0, s.link.steps - st0)
        self._tally += spent
        self._note("ref_search", head=r["head"], args=args, steps=spent)
        sq = kit.name_key(str(r["target"].get("seller") or ""))
        if query is not None and len(s.archive.obs) > n0 and kit.name_key(query) == sq:
            rows = self._rows_in(s.archive.obs[-1])
            if rows and not any(sq in self._sellers(x) for x in rows):
                r["unindexed"] = True
                self._note("ref_unindexed", head=r["head"], seller=r["target"].get("seller"))
        return spent > 0

    # ------------------------------------------------------------------ the per-turn move
    def step(self):
        s = self.s
        if s.link.done or s.link.broke or not s.refs:
            return False
        try:
            return self._turn()
        except Exception as e:
            self._err(e)
            for r in self._opens():
                try:
                    self._stop(r, "the chase logic failed")
                except Exception:
                    r["status"] = "stuck"
            return False

    def _turn(self):
        s = self.s
        for r in s.refs:
            if r["status"] == "open" and not self.live(r):
                r["status"] = "dropped"
                self._note("ref_dropped", head=r["head"])
        live = self._opens()
        if not live:
            return False
        for r in live:
            due = r.get("_due")
            if due:
                r["_due"] = None
                if self._sheet(r, due):
                    return True
        st0, acted = s.link.steps, False
        for r in sorted(live, key=lambda x: x["hops"]):
            if r["status"] == "open" and self._judge(r):
                acted = True
        if s.link.steps != st0:
            return True
        live = self._opens()
        if any(r.get("_due") for r in live):
            return True
        if not live or not self._room():
            return acted
        spec = s.mender.batch_spec()
        size = spec[2] if spec else 2
        pools = []
        for r in live:
            pool = [c for c in self._cands(r) if c not in s.archive.read and c not in r["ruled_out"]]
            if not r["tried"] and self._drafts(r) and len(pool) > size and not r["_hits"].get(pool[0]):
                pool = []
            pools.append((r, pool))
        tpl = self._template()
        idle = [r for r, p in pools if not p and self._drafts(r)] if tpl else []
        queue = []
        for i in range(max(len(p) for _, p in pools)):
            for _, p in pools:
                if i < len(p) and p[i] not in queue and len(queue) < size:
                    queue.append(p[i])
        if queue and not (len(queue) < size and idle):
            return self._read(queue) or acted
        if not tpl:
            return acted
        order = idle or sorted(live, key=lambda x: (x["hops"], len(x["tried"])))
        for r in order:
            drafts = self._drafts(r)
            if drafts:
                q = drafts[0]
                r["tried"].add(q.lower())
                if s.seeker.already_ran(q):
                    return True
                args = dict(tpl[0])
                args[tpl[1]] = q
                return self._search(r, args, q)
        for r in order:
            pg = self._more_page(r)
            if pg:
                r["extra_pages"] += 1
                return self._search(r, pg, None)
        stopped = False
        for r in self._opens():
            cands = self._cands(r)
            if not any(c not in r["ruled_out"] for c in cands) and not self._drafts(r) and self._more_page(r) is None:
                self._stop(r, "no store query showed the page it names after every try")
                stopped = True
        return stopped or acted

    def _judge(self, r):
        cands = self._cands(r)
        rd = self.s.archive.read
        seen = [c for c in cands if c in rd]
        fresh = [c for c in seen if c not in r["ruled_out"]]
        if not fresh:
            return False
        dets = list(r["target"].get("details") or [])
        if len(seen) >= 4:
            for d in dets:
                if d not in r["_loose"] and not any(self._shows(c, d) for c in seen):
                    r["_loose"].add(d)
        need = [d for d in dets if d not in r["_loose"]]
        full = {c: self._fully(c, dets) for c in fresh}
        fits = {c: all(self._shows(c, d) for d in need) for c in fresh}
        rank = sorted(fresh, key=lambda c: (not fits[c], -full[c]))
        if not fits[rank[0]] and (self._room() or self._waiting()) and self._more(r, cands):
            return False
        picks = [c for c in rank if fits[c]][:4] or rank[:4]
        exact = [c for c in picks if fits[c] and dets and full[c] == len(dets)]
        sample = exact if len(exact) == 1 else (picks[:2] if any(fits[c] for c in picks) else picks[:1])
        tool = self._notes_tool()
        done = set(self._stats(tool).get("read_ids") or ()) if tool else set()
        missing = [c for c in sample if c not in done and c not in r["_noted"]][:2] if tool else []
        if missing:
            room = self._room()
            if room and self._read_notes(r, missing):
                return True
            if self._waiting():
                return False
            if not room and r["relation"] == "differ":
                r["ruled_out"].update(picks)
                self._stop(r, "too few steps remained to fetch the remarks of the page it names")
                return True
        if len(exact) == 1 and self._plain(r, exact[0]):
            r["ruled_out"].update(picks)
            self._read_ids.update(picks)
            self._resolve(r, exact[0], self._field(exact[0], r["prop"]), [])
            return True
        if not self._can_check():
            r["ruled_out"].update(picks)
            self._stop(r, "the hop checks ran out")
            return True
        tgt = r["target"]
        self._hop(r, picks)
        if r["status"] == "open" and r["target"] is tgt:
            for c in fresh:
                if not fits[c] and any(self._miss_words(c, d) for d in need):
                    r["ruled_out"].add(c)
        return True

    def _plain(self, r, lid):
        """Code-only settlement test (09 3.14)."""
        val = self._field(lid, r["prop"])
        if not val:
            return False
        notes = self._notes(lid)
        if self._named(r["prop"], notes) or self._tok(val) & self._tok(notes):
            return False
        for sent in self._CUT4.split(notes):
            if any(self._mention(w, lid, False) for w in self._WORD.findall(sent)[1:]):
                return False
        return True

    # ------------------------------------------------------------------ the hop check
    def _card(self, lid):
        rec = self._rec(lid)
        facts = {}
        for name, mp in self._maps(lid):
            items = [(str(k), v) for k, v in mp.items() if kit.is_plain(v)][:40]
            facts[name] = {k: kit.cut(str(v), 90) for k, v in items}
        card = {"id": lid, "sold by": " / ".join(sorted({t for _, t in self._seller_keys(lid)})), "category": self._shelf(lid),
                "title": self._title(rec)[:160], "facts": facts, "notes": kit.cut(self._notes(lid, True), 900)}
        return kit.render(card, 5000)

    def _ask(self, r, batch, reason):
        self._checks += 1
        desc = self._desc(r["target"])
        parts = [PROMPTS["hop"].replace("<<PROPERTY>>", r["prop"]).replace("<<DESCRIPTION>>", desc),
                 "PROPERTY WANTED: " + r["prop"], "LISTING DESCRIBED: " + desc,
                 "CANDIDATE RECORDS:\n" + "\n".join(self._card(x) for x in batch)]
        if reason:
            parts.append("PREVIOUS ANSWER REJECTED: " + reason + "\nAnswer again, reading only the records above.")
        self._read_ids.update(batch)
        ans = self.s.link.consult("trace-hop", "\n\n".join(parts), "hop")
        ans = ans if isinstance(ans, dict) else None
        self._note("ref_hop_asked", head=r["head"], ids=list(batch), retry=bool(reason),
                   answer=kit.render(ans, 2000) if ans is not None else None)
        return ans

    def _hop(self, r, batch):
        ans, retried = self._ask(r, batch, ""), False
        while True:
            if ans is None:
                r["fails"] += 1
                if r["fails"] >= 2:
                    r["ruled_out"].update(batch)
                    self._stop(r, "the hop check never answered")
                return
            nm = ans.get("source_id")
            named = str(nm).strip() if kit.is_plain(nm) else ""
            miss = ans.get("absent_details")
            miss = [str(x).strip() for x in miss if str(x).strip()] if isinstance(miss, list) else []
            cand = named if named in batch and not miss else ""
            go = ans.get("onward")
            has_go = isinstance(go, dict) and bool(str(go.get("seller") or "").strip())
            if not cand and has_go:
                dets = list(r["target"].get("details") or [])
                need = [d for d in dets if d not in r["_loose"]]
                ok = [x for x in batch if all(self._tok(d) <= self._tok(self._notes(x, True)) for d in dets)
                      and not self._onward(r, x, go)[0] and not any(self._miss_words(x, d) for d in need)]
                if len(ok) == 1:
                    cand = ok[0]
                    r["_loose"].update(d for d in dets if not self._shows(cand, d))
                    self._note("ref_rescued", head=r["head"], id=cand)
            if not cand:
                r["ruled_out"].update(batch)
                self._read_ids.update(batch)
                if named in batch:
                    self._note("ref_partial", head=r["head"], id=named, missing=miss[:6])
                return
            v = self._validate(r, cand, batch, ans)
            if v[1]:
                if (v[0] == "answer" and not has_go and not self._field(cand, r["prop"]) and cand not in r["_sheeted"]
                        and self._sheet_tool(cand)):
                    r["_sheeted"].add(cand)
                    r["_due"] = cand
                    r["ruled_out"].update(x for x in batch if x != cand)
                    self._read_ids.update(batch)
                    self._note("ref_sheet_due", head=r["head"], id=cand, why=v[1][:300])
                    return
                if not retried and self._can_check():
                    retried = True
                    ans = self._ask(r, batch, v[1])
                    continue
            self._settle(r, cand, batch, v)
            return

    def _onward(self, r, lid, go):
        """Check a go_to against the candidate's notes: (problem text, onward target or None)."""
        seller, notes = str(go.get("seller") or "").strip(), self._notes(lid, True)
        if not self._says(notes, seller):
            return "that seller (" + seller + ") is not written in the notes of " + lid, None
        sents, rel = self._CUT3.split(notes), []
        for i, x in enumerate(sents):
            if self._says(x, seller):
                if i and self._has_neg(sents[i - 1]) and sents[i - 1] not in rel:
                    rel.append(sents[i - 1])
                rel.append(x)
        if not any(self._named(r["prop"], x) for x in rel):
            return "the line of " + lid + " naming " + seller + " is about another property", None
        nt, dets = self._tok(notes), []
        raw = go.get("details")
        for d in raw if isinstance(raw, list) else []:
            if not kit.is_plain(d):
                continue
            d, dt = str(d).strip(), kit.ref_tokens(str(d))
            if d and dt and 2 * sum(1 for x in dt if x in nt) >= len(dt) and d not in dets:
                dets.append(d)
        nxt = {"seller": seller, "kind": str(go.get("kind") or "").strip()[:80], "details": dets[:6]}
        if kit.name_key(seller) in self._sellers(lid) and self._tkey(nxt) == self._tkey(r["target"]):
            return "the go_to describes " + lid + " itself", None
        return "", nxt

    def _validate(self, r, cand, batch, ans):
        """(problem kind, problem text, onward target, value, partners) for a named candidate (09 3.15.3 C)."""
        prop, dets = r["prop"], list(r["target"].get("details") or [])
        for d in dets:
            if d not in r["_loose"] and not self._shows(cand, d):
                return "target", cand + " does not show '" + d + "', so it is not the described listing", None, "", []
        score = {x: self._fully(x, dets) for x in batch}
        best = max(batch, key=lambda x: score[x])
        if score[best] > score.get(cand, 0):
            return "target", best + " holds more of the named details than " + cand, None, "", []
        go = ans.get("onward")
        if isinstance(go, dict) and str(go.get("seller") or "").strip():
            prob, nxt = self._onward(r, cand, go)
            return ("answer", prob, None, "", []) if prob else (None, "", nxt, "", [])
        notes, fv = self._notes(cand, True), self._field(cand, prop)
        rv = ans.get("read_value")
        val = str(rv).strip() if kit.is_plain(rv) else ""
        if not fv:
            said = [x for x in self._CUT4.split(notes) if self._named(prop, x)]
            if said and (not val or not self._tok(val) <= self._tok(" ".join(said))):
                return ("answer", cand + " has no field for " + prop + " but its notes state it, so the value must be the one"
                        " the notes give, with its partners there", None, "", [])
            if not self._named(prop, self._title(self._rec(cand)) + "\n" + notes):
                sh = self._shelf(cand)
                peers = [x for x in self._index()[0] if x != cand and x in self.s.archive.read and self._shelf(x) == sh]
                if peers and not any(self._field(x, prop) for x in peers):
                    return "target", "nothing on that shelf speaks of " + prop + ", so " + cand + " is not the listing", None, "", []
        nt, st = self._words(cand)
        if not val or not self._tok(val) <= (nt | st):
            return "answer", "the value '" + val + "' is not written in the record of " + cand, None, "", []
        ntok, parts = self._tok(notes), []
        gw = ans.get("partners")
        for p in gw if isinstance(gw, list) else []:
            if kit.is_plain(p) and str(p).strip():
                p = str(p).strip()
                pt = self._tok(p)
                if p not in parts and pt and pt <= ntok:
                    parts.append(p)
        return None, "", None, val, parts[:3]

    def _settle(self, r, cand, batch, v):
        kind, prob, nxt, val, parts = v
        self._read_ids.update(batch)
        if not prob:
            r["ruled_out"].update(batch)
            if cand in r["chain"]:
                self._stop(r, "its notes lead back to a listing the chain already went through")
            elif nxt:
                self._follow(r, cand, nxt)
            else:
                self._resolve(r, cand, val, parts)
        elif kind == "target":
            self._note("ref_invalid", head=r["head"], id=cand, kind=kind, why=prob[:300])
            r["ruled_out"].add(cand)
        else:
            self._note("ref_invalid", head=r["head"], id=cand, kind=kind, why=prob[:300])
            r["ruled_out"].update(batch)
            r["chain"].append(cand)
            self._stop(r, prob)

    # ------------------------------------------------------------------ follow, resolve, swap, stop, tell
    def _follow(self, r, cand, nxt):
        r["chain"].append(cand)
        key = self._tkey(nxt)
        if key in r["seen_keys"]:
            self._stop(r, "the notes loop back to a listing the chain already described")
            return
        if r["hops"] + 1 >= Tune.Refs.HOPS:
            self._stop(r, "more than " + str(Tune.Refs.HOPS) + " listings in a row each point elsewhere")
            return
        r["seen_keys"].add(key)
        r["hops"] += 1
        r["target"] = nxt
        r["ruled_out"], r["tried"], r["extra_pages"], r["fails"] = set(), set(), 0, 0
        r["unindexed"], r["_loose"], r["_late"] = False, set(), None
        self._note("ref_hop", head=r["head"], via=cand, target=self._desc(nxt), hops=r["hops"])
        self._tell(cand + " is the listing meant for '" + r["head"] + "', but its own notes pass the value on to "
                   + self._desc(nxt) + "; the code follows them.")

    def _rule(self, r, value, partners):
        rel, low = r["relation"], value.lower()
        others = [p for p in partners if p.lower() != low]
        if rel == "within" and kit.is_num(r.get("tolerance")):
            g = re.search(r"([-+]?\d+(?:\.\d+)?)\s*([A-Za-z]+|%)?", value.replace(",", ""))
            if g:
                mid, tol = float(g.group(1)), float(r["tolerance"])
                unit = g.group(2) or str(r.get("unit") or "")
                span = "%g to %g" % (round(mid - tol, 6), round(mid + tol, 6)) + (" " + unit if unit else "")
                return span + " (within %g of %s)" % (tol, value)
            rel = "same"
        if rel in ("at most", "at least"):
            return rel + " " + value
        if rel == "differ":
            return " and ".join("not " + x for x in [value] + others)
        if rel == "pairs":
            return " or ".join(self._uniq(partners))
        return " or ".join([value] + others)

    def _resolve(self, r, cand, value, partners):
        value = str(value or "").strip()
        if not value:
            self._stop(r, cand + " gives no value for it")
            return
        if r["relation"] != "pairs":
            vt = self._tok(value)
            partners = [p for p in partners if self._tok(p) != vt]
        elif not partners:
            self._stop(r, cand + " shows " + value + ", but no store rule saying which value pairs with it was found,"
                       " so the counterpart is unknown")
            return
        r["chain"].append(cand)
        r["status"], r["value"], r["partners"] = "resolved", value, list(partners)
        r["_rule"] = self._rule(r, value, list(partners))
        r["settled"] = r["head"] + ": " + r["_rule"]
        self._swap(r)
        led = self.s.ledger
        led.resolved.append(r["settled"])
        led.version += 1
        try:
            self.s.warden.disclosure_settle(r)
        except Exception as e:
            self._err(e)
        self._note("ref_resolved", head=r["head"], value=value, partners=list(partners), chain=list(r["chain"]), settled=r["settled"])
        rel, ps = r["relation"], " / ".join(partners)
        t = "SETTLED: " + r["head"] + " is fixed by a linked listing. " + cand + " shows " + r["prop"] + " '" + value + "'"
        if len(r["chain"]) >= 2:
            t += " (reached from " + r["chain"][0] + " through the listings' notes)"
        t += ". "
        if rel == "pairs":
            t += ("The store pairs '" + value + "' with " + ps + "; the shopper wants that counterpart, so a listing showing "
                  + ps + " meets it and '" + value + "' itself does not" + (" unless it is among them" if value in partners else "") + ". ")
        elif rel == "differ":
            if partners:
                t += "Its notes pair " + ps + " with it. "
            t += "THE SHOPPER'S " + r["prop"].upper() + " MUST DIFFER FROM IT: ANY OTHER VALUE MEETS IT, THAT ONE DOES NOT. "
        elif partners:
            t += "Its notes say '" + value + "' goes together with " + ps + ", so each of them meets it. "
        self._tell(t + "The requirement now reads: " + r["settled"])

    def _covers(self, req, r):
        if self.resolves_to(req, r):
            return True
        front, prop, tag, value = self._cut(req)
        tags = kit.tags_of(front)
        if r["tag"] and tags and not any(kit.kind_terms(x) & kit.kind_terms(r["tag"]) for x in tags):
            return False
        sq = kit.name_key(str(r["first_target"].get("seller") or ""))
        return bool(sq) and sq in kit.name_key(value) and kit.property_match(prop, r["prop"])

    def _swap(self, r):
        led, st, hit = self.s.ledger, r["settled"], False
        for name in ("written", "revealed"):
            lst = getattr(led, name, None)
            if isinstance(lst, list):
                new = []
                for x in lst:
                    if isinstance(x, str) and self._covers(x, r):
                        x, hit = st, True
                    if x not in new:
                        new.append(x)
                lst[:] = new
        if isinstance(led.rebuilt, list):
            new = self._uniq([st if isinstance(x, str) and self._covers(x, r) else x for x in led.rebuilt])
            if st not in new:
                new.append(st)
            led.rebuilt[:] = new
        if not hit and st not in led.revealed:
            led.revealed.append(st)
        rd = led.reading
        if isinstance(rd, dict):
            rd["musts"] = list(led.written)
            rows = rd.get("parts") if isinstance(rd.get("parts"), list) else []
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("musts"), list):
                    continue
                if len(rows) > 1 and r["tag"] and not (kit.kind_terms(str(row.get("kind") or "")) & kit.kind_terms(r["tag"])):
                    continue
                row["musts"] = [self._cut(x)[0] + ": " + r["_rule"] if isinstance(x, str) and self._covers(x, r) else x
                                for x in row["musts"]]

    def _stop(self, r, cause):
        if r.get("status") != "open":
            return
        r["status"], r["_cause"] = "stuck", cause
        self._note("ref_stuck", head=r["head"], cause=cause[:300], chain=list(r["chain"]))
        self._tell("UNRESOLVED: " + r["head"] + ": " + cause + ". Its value is unknown. If the notes of a listing already seen"
                   " show where the chain ends, take that value; otherwise this requirement must never be a reason to"
                   " decline or to turn a listing down.")

    def _tell(self, text):
        chat = self.s.chat
        last = chat[-1] if chat else {}
        if len(chat) > 2 and last.get("role") in ("tool", "user") and isinstance(last.get("content"), str):
            self.s.engine.append_note("\n[reference] " + text)
        else:
            self._tells.append(text)

    # ------------------------------------------------------------------ text for the actor and checkers
    def status_text(self):
        try:
            ents = []
            for r in self.s.refs:
                st = r.get("status")
                if st == "resolved":
                    e = r["settled"] + " (read on " + (r["chain"][-1] if r["chain"] else "?")
                    e += ("; must differ from it)" if r["relation"] == "differ" else ")")
                    fd = r["first_target"].get("details") or []
                    if fd:
                        e += " [" + "; ".join(fd) + ": these describe the pointed listing, not conditions on the item to buy]"
                elif st == "open":
                    e = r["head"] + ": the code is following it to " + self._desc(r["target"])
                elif st == "stuck":
                    e = r["head"] + ": unresolved, value unknown; never a reason to decline or to reject a listing"
                else:
                    continue
                ents.append(e)
            out = ("\n[references] " + "; ".join(ents)) if ents else ""
            for t in self._tells:
                out += "\n[reference] " + t
            self._tells = []
            return out
        except Exception as e:
            self._err(e)
            return ""

    def checker_block(self, lid=None):
        try:
            refs = [r for r in self.s.refs if r.get("status") in ("resolved", "open", "stuck")]
            if not refs:
                return ""
            out = ["Each value below comes from another listing; the code traced it through the notes in the store's records."]
            ends, gap = [], False
            for r in refs:
                ch = r["chain"]
                last = ch[-1] if ch else "?"
                via = (" (traced from " + ch[0] + " along the notes)") if len(ch) > 1 else ""
                ps = " / ".join(r["partners"])
                if r["status"] == "resolved" and r["relation"] == "differ":
                    out.append("- EXCLUSION " + r["head"] + ": the shopper's " + r["prop"] + " must NOT be '" + r["value"] + "', the value "
                               + last + via + " shows" + ("; its notes also link " + ps + " to it" if ps else "")
                               + ". Any other value passes and this one fails. Every listing on the chain (" + ", ".join(ch)
                               + ") carries this value.")
                elif r["status"] == "resolved" and r["relation"] == "pairs":
                    out.append("- COUNTERPART " + r["head"] + ": a listing showing any of " + ps + " meets it; '" + r["value"]
                               + "' itself does not, unless it is a partner. " + last + via + " shows '" + r["value"]
                               + "', the store pairs it with " + ps + ", and the shopper's " + r["prop"] + " is that partner value.")
                elif r["status"] == "resolved":
                    out.append("- SETTLED " + r["settled"] + ": taken from " + last + via + "."
                               + (" Its notes count " + ps + " as equivalent to the value, so each of them passes." if ps else ""))
                else:
                    gap = True
                    out.append("- OPEN " + r["head"] + ": " + ("the code got as far as " + " -> ".join(ch) if ch
                                                               else "the code never located the listing the pointer names") + ".")
                    ends.extend(x for x in ch[-2:] if x != lid)
            if gap:
                out.append("For an OPEN entry: if a record shown further down is that listing, or the last listing its chain"
                           " reached, and it gives its own value without pointing on, that value is what the requirement asks for"
                           " (for a counterpart, the partner value the store gives for it). Otherwise the value is unknown, and"
                           " this requirement alone can never make an item fail.")
            recs = []
            for x in self._uniq(list(reversed(ends)))[:2]:
                got = self.s.archive.about(x, 1)
                if got:
                    recs.append(got[-1])
            if recs:
                out.append("RECORDS WHERE AN OPEN CHAIN STOPPED:")
                out.extend(recs)
            return "\n".join(out)
        except Exception as e:
            self._err(e)
            return ""

    # ------------------------------------------------------------------ services
    def _self_paired(self, r):
        vt = self._tok(r.get("value"))
        return any(self._tok(p) == vt for p in r.get("partners") or [])

    def _barring(self, states, resolved_ok=True):
        out = []
        for r in self.s.refs:
            if r.get("status") not in states or not self.live(r):
                continue
            if r["relation"] == "differ" or (resolved_ok and r["status"] == "resolved" and r["relation"] == "pairs"
                                             and not self._self_paired(r)):
                out.append(r)
        return out

    def _belongs(self, r, lid):
        if not self.s.multi() or not r.get("tag"):
            return True
        k = self.s.walker.product_slot(lid)
        return k is None or k == self.s.mender.item_for(r["tag"])

    def barred(self):
        try:
            return {x for r in self._barring(("open", "stuck", "resolved")) for x in r["chain"] if self._belongs(r, x)}
        except Exception as e:
            self._err(e)
            return set()

    def chain_ids(self):
        try:
            return {x for r in self.s.refs for x in r.get("chain") or []}
        except Exception as e:
            self._err(e)
            return set()

    def pointer_note(self, lid):
        try:
            bars = self._barring(("open", "stuck", "resolved"))
            for r in bars:
                if lid in r["chain"] and self._belongs(r, lid):
                    val = (" (" + r["value"] + ")") if r["status"] == "resolved" and r["value"] else ""
                    return (lid + " is on the pointer chain for '" + r["head"] + "' (the source page, or a page its remarks route"
                            " through), so its " + r["prop"] + " carries exactly the value to avoid" + val + "; it can never qualify.")
            for r in bars:
                if r["status"] != "resolved" or r["relation"] != "pairs" or not self._belongs(r, lid):
                    continue
                vt = self._tok(r["value"])
                for _, mp in self._maps(lid):
                    for k, v in mp.items():
                        if kit.is_plain(v) and kit.property_match(str(k), r["prop"]) and vt and any(
                                self._tok(p) == vt for p in re.split(r"[,;/&]", str(v))):
                            return (lid + " lists '" + r["value"] + "' in its " + str(k) + ", but the wanted " + r["prop"]
                                    + " has to go with that value, not be it; it can never qualify.")
            return ""
        except Exception as e:
            self._err(e)
            return ""

    def excluded(self, fields, slot):
        try:
            items = fields.items() if isinstance(fields, dict) else list(fields or [])
            multi = self.s.multi()
            for r in self.s.refs:
                if r.get("status") != "resolved" or r["relation"] != "differ" or not str(r["value"]).strip() or not self.live(r):
                    continue
                if multi and r["tag"]:
                    k = self.s.mender.item_for(r["tag"])
                    if k is not None and k != slot:
                        continue
                bad = {kit.flat_value(x) for x in [r["value"]] + list(r["partners"])} - {""}
                for name, val in items:
                    if kit.property_match(str(name), r["prop"]) and kit.flat_value(str(val)) in bad:
                        return True
            return False
        except Exception as e:
            self._err(e)
            return False

    def _sends_on(self, sent, prop, lid):
        masked = re.sub(r"\b[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda g: " " * len(g.group(0)), sent)
        words = list(self._WORD.finditer(sent))
        anchor = next((i for i, w in enumerate(words) if kit.names_prop(masked[:w.end()], prop)), None)
        if anchor is None:
            return False
        for i in range(max(anchor + 1, 1), len(words)):
            if self._mention(words[i].group(0), lid, True) and not any(self._negator(w.group(0)) for w in words[max(0, i - 5):i]):
                return True
        return False

    def self_pointers(self, lid):
        try:
            if not self._rec(lid):
                return []
            notes = self._notes(lid)
            if not notes:
                return []
            slot = self.s.walker.product_slot(lid) if self.s.multi() else None
            sents, out = [x for x in self._CUT4.split(notes) if x.strip()], []
            for req in self.s.ledger.in_force():
                if not isinstance(req, str):
                    continue
                front, prop, tag, value = self._cut(req)
                if not prop:
                    continue
                if slot is not None:
                    k = self.s.mender.item_for(front)
                    if k is not None and k != slot:
                        continue
                for sent in sents:
                    if self._sends_on(sent, prop, lid):
                        out.append((req, prop, sent.strip()))
                        break
            return out
        except Exception as e:
            self._err(e)
            return []

    def redirect_remark(self, name, args):
        try:
            s = self.s
            skip = {s.role(x) for x in ("search", "cart", "cart_view", "message")} | set(s.intake.finishing_tools())
            if name in skip:
                return ""
            args = args if isinstance(args, dict) else (kit.args_from(args) or {})
            out = []
            for lid in sorted({str(x) for x in s.judge.question_ids(name, args)}):
                carted = bool(s.basket.held(lid))
                if not (carted or s.passed_now(lid)):
                    continue
                stamp = lid + "|" + str(s.ledger.version)
                if stamp in self._stamps:
                    continue
                reds = self.self_pointers(lid)
                if not reds:
                    continue
                self._stamps.add(stamp)
                s.passed.discard(s.pass_key(lid))
                self._note("ref_own_redirect", id=lid, props=[p for _, p, _ in reds])
                quotes = "; ".join(p + ' ("' + sent[:160] + '")' for _, p, sent in reds)
                t = (lid + ": its remarks may shift " + quotes + " onto another page. If so, its own field is no evidence for "
                     + " / ".join(q for q, _, _ in reds) + ", because grading reads that other page; the requirement is met only"
                     " where that page's record shows the value. A new item check is needed before any order.")
                if carted:
                    t += " It is in the cart: if the shift is real, remove that line and cart another confirmed pick."
                out.append(t)
            return " ".join(out)
        except Exception as e:
            self._err(e)
            return ""

    def foreign_ids(self):
        try:
            if not self._read_ids:
                return set()
            homes = self.s.judge.home_labels()
            if not homes or any(not h for h in homes):
                return set()
            allh = set().union(*[set(h) for h in homes])
            ix = self._index()[0]
            return {x for x in self._read_ids if x in ix and not self.s.basket.held(x) and self._shelf(x) and self._shelf(x) not in allh}
        except Exception as e:
            self._err(e)
            return set()

    def seller_fields(self):
        try:
            return set(self._learned())
        except Exception as e:
            self._err(e)
            return set()

    def record_index(self):
        try:
            ix, last = self._index()
            return ix, last
        except Exception as e:
            self._err(e)
            return {}, {}

    # ------------------------------------------------------------------ the listener reply-line fixer
    def fix_reply(self, answer, lines, question):
        if isinstance(lines, str):
            lines = [lines]
        out = []
        for x in list(lines or []):
            try:
                out.append(self._fix_line(answer, str(x), question) if isinstance(answer, dict) else str(x))
            except Exception as e:
                self._err(e)
                out.append(str(x))
        return out

    def _phrase(self, line):
        """(phrase, start, end, (seller, kind, details)) of the pointer phrase in a reply line, or None (09 3.19.1)."""
        if not kit.is_pointer(line):
            return None
        ok = lambda n: self._seller_ok(n, line)
        hit = kit.parse_listing(line, ok)
        if not hit or not ok(hit[0]):
            return None
        seller = str(hit[0])
        pat = r"(?<![A-Za-z'])" + re.escape(seller) + r"(?:'s)?(?![A-Za-z])"
        g = re.search(pat, line) or re.search(pat, line, re.I)
        if not g:
            return None
        a, b = g.span()
        cut = max([x.end() for x in re.finditer(r"[,;!?]|\.(?=\s)", line[:a])] or [0])
        clause = line[cut:a].lower()
        found = [self._first_span(clause, f) for f in (kit.is_differ, kit.is_pointer, kit.has_pointer_phrase, kit.is_pair)]
        starts = [x[0] for x in found if x]
        if not starts:
            return None
        p0 = cut + min(starts)
        e = re.search(r"[.!?;](?=\s|$)", line[b:])
        p1 = b + e.start() if e else len(line)
        o = re.search(r"\(\s*or\b[^)]*\)", line[b:p1])
        if o:
            p1 = b + o.end()
        while p0 < p1 and line[p0] in " ,;":
            p0 += 1
        while p1 > p0 and line[p1 - 1] in " ,;":
            p1 -= 1
        if p1 <= p0:
            return None
        return line[p0:p1], p0, p1, (seller, str(hit[1] or ""), [str(d) for d in hit[2] or []])

    @staticmethod
    def _place(text, sources):
        for src in sources:
            named = [n for n in src if n and kit.names_prop(text, n)]
            if named:
                return max(named, key=lambda n: (kit.names_prop(text, n, True), len(n.split()), len(n)))
        return ""

    def _fix_line(self, ans, line, question):
        found = self._phrase(line)
        if not found:
            return line
        phrase, p0, p1, (seller, kind, dets) = found
        s, led = self.s, self.s.ledger
        outside = line[:p0].rstrip() + " " + line[p1:].lstrip()
        refs_in = [e for e in ans.get("pointer_refs") or [] if isinstance(e, dict)] if isinstance(ans.get("pointer_refs"), list) else []
        needs = [n for n in ans.get("new_musts") or [] if isinstance(n, str)] if isinstance(ans.get("new_musts"), list) else []
        cneeds = [n for n in needs if ":" in n]
        src1 = [str(e.get("property") or "").strip() for e in refs_in]
        src2 = [self._label(n) for n in cneeds if kit.is_pointer(n.split(":", 1)[1]) or kit.is_differ(n.split(":", 1)[1])]
        src3 = [self._label(n) for n in needs]
        asked = [x.split("(")[0].strip() for x in str(s.asking_label or "").split(";") if x.strip()]
        src5 = sorted(self._store_fields()) + [self._label(x) for x in led.written or [] if isinstance(x, str)]
        prodw = set()
        for p in led.products() or [{"name": led.product_name()}]:
            prodw |= kit.kind_terms(str(p.get("name") or ""))
        qw = re.findall(r"[a-z]+", str(question or "").lower())
        good = lambda w: not kit.is_link(w) and len(w) >= 3 and not (kit.kind_terms(w) & prodw)
        src6 = [" ".join(qw[i:i + n]) for n in (3, 2, 1) for i in range(len(qw) - n + 1) if all(good(w) for w in qw[i:i + n])]
        sources = [src1, src2, src3, asked, src5, src6]
        prop = self._place(outside, sources)
        if not prop:
            low, pos = phrase.lower(), 0
            at = low.rfind(seller.lower())
            if at >= 0:
                pos = at + len(seller)
            for d in dets:
                for w in d.lower().split():
                    i = low.find(w, pos)
                    if i >= 0:
                        pos = i + len(w)
            prop = self._place(phrase[pos:], sources)
        if not prop:
            prop = next((n for n in src1 + src2 + (asked if len(asked) == 1 else []) if n and not kit.names_prop(phrase, n)), "")
        if not prop:
            self._note("ref_line_unplaced", line=line[:300])
            return outside
        tag, multi = "", s.multi()
        if multi:
            k = s.mender.item_for(outside)
            if k is None:
                tags = [t for n in cneeds if kit.property_match(self._label(n), prop) for t in kit.tags_of(self._cut(n)[0])[-1:]]
                if tags:
                    k = s.mender.item_for(tags[-1])
            if k is None and question:
                k = s.mender.item_for(str(question))
            prods = led.products()
            if k is not None and 0 <= k < len(prods):
                tag = str(prods[k].get("name") or "")
        tw = kit.kind_terms(tag)

        def product_match(n):
            if not (multi and tag):
                return True
            tg = kit.tags_of(self._cut(n)[0])
            return not tg or bool(kit.kind_terms(tg[-1]) & tw)
        sq = kit.name_key(seller)
        rel = kit.relation_of(phrase)
        ptok, otok = self._tok(phrase), self._tok(outside)

        def inside_only(lab):
            return kit.names_prop(phrase, lab) and not kit.names_prop(outside, lab)

        def describes(n):
            vt = self._tok(n.split(":", 1)[1] if ":" in n else "")
            return inside_only(self._label(n)) or (bool(vt) and vt <= ptok and not (vt & otok))

        def points_here(n):
            return ":" in n and bool(sq) and sq in kit.name_key(n.split(":", 1)[1])
        own = [n for n in cneeds if kit.property_match(self._label(n), prop) and product_match(n)]
        reuse = next((n for n in own if points_here(n) and kit.relation_of(n.split(":", 1)[1])[0] == rel[0]), "")
        head = self._cut(own[0])[0] if own else (prop + " (" + tag + ")" if tag else prop)
        if multi and tag and not kit.tags_of(head):
            head += " (" + tag + ")"
        new = reuse or head + ": " + phrase
        foreign = [n for n in cneeds if n not in own and (describes(n) or (product_match(n) and points_here(n)
                                                                           and not kit.names_prop(line, self._label(n))))]
        kept, spot = [], None
        for n in cneeds:
            if own and n == own[0] and spot is None:
                spot = len(kept)
            if n not in own and n not in foreign:
                kept.append(n)
        kept.insert(len(kept) if spot is None else spot, new)
        ans["new_musts"] = kept
        cur = ans.get("musts_now")
        if isinstance(cur, list):
            keep = []
            for n in cur:
                if not isinstance(n, str):
                    continue
                if n in (led.written or []) or not (n in foreign or describes(n) or (kit.property_match(self._label(n), prop) and product_match(n))):
                    keep.append(n)
            if new not in keep:
                keep.append(new)
            ans["musts_now"] = keep
        if isinstance(ans.get("dropped_musts"), list):
            ans["dropped_musts"] = [n for n in ans["dropped_musts"] if not (isinstance(n, str) and (inside_only(self._label(n)) or describes(n)))]
        if isinstance(ans.get("dont_care"), list):
            ans["dont_care"] = [n for n in ans["dont_care"] if not (isinstance(n, str) and (inside_only(self._label(n))
                                                                                            or kit.property_match(self._label(n), prop)))]
        ent = next((e for e in refs_in if kit.property_match(str(e.get("property") or ""), prop)), None)
        if ent is None:
            ent = {"property": prop}
            refs_in.append(ent)
        es = str(ent.get("seller") or "").strip()
        if not es or not self._seller_ok(es, line) or kit.name_key(es) != sq:
            ent["seller"], ent["kind"], ent["details"] = seller, kind, list(dets)
        elif not ent.get("details"):
            ent["details"] = list(dets)
        if not str(ent.get("kind") or "").strip():
            ent["kind"] = kind or "listing"
        if rel[0] != "same" or not ent.get("relation"):
            ent["relation"], ent["tolerance"], ent["unit"] = rel[0], rel[1], str(rel[2] or "")
        ans["pointer_refs"] = refs_in
        marks = ans.get("pointer_musts")
        if not isinstance(marks, list):
            marks = ans["pointer_musts"] = []
        if new not in marks:
            marks.append(new)
        self._note("ref_line_fixed", property=prop, tag=tag, need=new[:300], outside=outside[:300])
        return outside



# ------------------------------------------------------------ p12_pricer.py
# p12_pricer.py - Pricer (spec part 10): the search note, price-cap learning, ceiling walk and sweep, cap seeding,
# value-led searches and value pages, the value guard, the walk hold and the walk step. No model calls here.

_p12_AN = re.compile(r"[a-z0-9]+")
_p12_AN2 = re.compile(r"[a-z0-9]{2,}")
_p12_AN3 = re.compile(r"[a-z0-9]{3,}")
_p12_AL = re.compile(r"[a-z]+")
_p12_CAT = re.compile(r"categor|shelf|department|aisle")
_p12_NU = re.compile(r"\s*(\d+)\s+([a-z]+)")
_p12_ALTS = re.compile(r"[,;/]|\b(?:or|either|nor)\b")
_p12_CLOSED = re.compile(r"\bor\b|\beither\b|/")
_p12_CLOSED_SPLIT = re.compile(r"[,;/]|\bor\b|\beither\b")
_p12_AND_SPLIT = re.compile(r"[,;&+]|\band\b")
_p12_CODE = re.compile(r"\b(?:\d[a-z]|[a-z]\d)\b")
_p12_DEC = re.compile(r"\d[.,]\d+\s*[a-z]")
_p12_AMT = re.compile(r"\s*\d+(?:[.,]\d+)?\s+[a-z]")
_p12_TOK = re.compile(r"\d+(?:[.,]\d+)+|[a-z0-9]+")
_p12_FLAG = re.compile(r"next|more|onward|continu|cursor")
_p12_NOLIM = re.compile(r"page|size|count|result|min|low|from|start|offset")
_p12_NOCAP = re.compile(r"page|size|count|result|min|low|from|start")
_p12_PRICE = re.compile(r"price|cost")
_p12_CAPN = re.compile(r"ceil|max|price|budget|upper|most")
_p12_CAPX = re.compile(r"page|first|offset|skip|count|size|num")
_p12_PAREN = re.compile(r"\([^)]*\)")
_p12_POSS = re.compile(r"'s\b")
# figure rungs: comparator grammar classes of spec 10 3.14.5 (flagged there; kept as short structural patterns)
_p12_NOMORE = re.compile(r"\bno\W+(more|less)\W+than\b")
_p12_DOWN = re.compile(r"\b(?:at\W+most|up\W+to|or\W+(?:less|lower|smaller)|maximum|less\W+than|under|below)\b")
_p12_UPW = re.compile(r"\b(?:at\W+least|or\W+(?:more|higher|larger)|minimum|more\W+than|over|above)\b")
_p12_STRICT = re.compile(r"\b(?:more\W+than|less\W+than|over|above|under|below)\b")
_p12_FIG = re.compile(r"(?<![a-z0-9_.])(\d+(?:\.\d+)?)(?=\s*[a-z\"'\u2019\u201d])")
_p12_NUMS = re.compile(r"\d+(?:\.\d+)?")


class Pricer:
    """Price ceilings, value-led search and the cheapest walks (spec 10)."""

    def __init__(self, s):
        self.s = s
        self._sent, self._sent_args = {}, {}
        self._prof, self._broken, self._depth_cap = None, set(), None
        self._walked, self._aims, self._held = set(), {}, {}
        self._cap_keys, self._seed_keys, self._seeded = set(), set(), set()
        self._tries, self._settled, self._composed, self._indexing = {}, {}, {}, None
        self._done_keys, self._vkeys, self._vpages, self._vopens, self._waiting = set(), {}, {}, {}, {}
        self._queued, self._rungs, self._told = None, None, set()
        self._short_hint, self._probe_hints, self._trap_hints = False, 0, 0
        self._sigs, self._retro, self._cache = {}, False, {}

    # ------------------------------------------------------------------ plumbing
    def _fail(self, kind, e):
        try:
            self.s.link.fault(kind, e)
        except Exception:
            pass

    def _memo(self, tag, key, fn):
        s = self.s
        stamp = (len(s.archive.obs), s.ledger.version, s.lead_words, s.word_mode)
        if self._cache.get("_stamp") != stamp: self._cache = {"_stamp": stamp}
        k = (tag, key)
        if k not in self._cache: self._cache[k] = fn()
        return self._cache[k]

    @staticmethod
    def _canon(x):
        return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)

    @staticmethod
    def _money(p):
        if not kit.is_num(p): return "?"
        return str(int(p)) if float(p).is_integer() else "%.2f" % p

    def _priced(self, ids):
        return ", ".join("%s (%s)" % (i, self._money(self.s.archive.price(i))) for i in ids)

    def _tool(self, role="search"):
        return self.s.role(role)

    def _schema(self):
        t = self._tool()
        return self.s.schema(t) if t else {"type": "object", "properties": {}}

    def _props(self):
        p = self._schema().get("properties")
        return p if isinstance(p, dict) else {}

    def _enum(self, k, props=None):
        e = ((self._props() if props is None else props).get(k) or {}).get("enum")
        return e if isinstance(e, list) else None

    def _pg(self):
        return self.s.seeker.paging()

    def _pnames(self, args):
        sk, pg, out = self.s.seeker, self._pg(), set()
        out |= set((sk.search_parts(args) or {}).get("paging") or ())
        if pg: out.add(pg.get("name"))
        cur = sk.cursor_param()
        if cur: out.add(cur)
        return out

    def _modes(self, args, props=None):
        props = self._props() if props is None else props
        return sorted((k, v) for k, v in args.items() if isinstance(v, str) and self._enum(k, props))

    def _same_modes(self, fm, args):
        try:
            got = sorted(tuple(x) for x in (fm.items() if isinstance(fm, dict) else (fm or ())))
            return got == [tuple(x) for x in self._modes(args)]
        except Exception:
            return False

    def _cheapest(self):
        return self.s.ledger.priority() == "cheapest"

    def _slots(self):
        return list(self.s.ledger.slots() or [None])

    def _short(self):
        return bool(self.s.link.done) or self.s.link.time_left() < 130

    def _can_list(self, floor):
        l = self.s.link
        return not l.done and not l.broke and l.time_left() >= 100 and self.s.steps_left() - 1 > floor

    def _batch(self):
        return self.s.mender.batch_spec()

    def _batch_max(self):
        b = self._batch()
        return b[2] if b else None

    def _open_cost(self):
        return 1 if self._batch() else 2

    def _asks(self):
        return bool(self.s.asking) and bool(self._tool("message"))

    def _room(self):
        s = self.s
        q = max(s.intake.question_room() - s.msgs_sent, 0) if self._asks() else 0
        return q, s.steps_left() - s.intake.due_margin() - 2 * q

    def _home(self, slot):
        return {kit.low(x).strip() for x in (self.s.walker.tour_shelf(slot) or ()) if x}

    def _pick(self, slot):
        pk = self.s.walker.pick_of(slot)
        return pk if pk and kit.is_num(pk[1]) else None

    def _first(self, e):
        pg = self._pg()
        return e.get("first_ids") is not None or bool(pg and pg.get("start") in (e.get("spots") or ()))

    @staticmethod
    def _seen(e):
        return len(e.get("row_shelf") or {})

    @staticmethod
    def _on(e, home):
        return [i for i, sh in ((e or {}).get("row_shelf") or {}).items() if kit.low(sh).strip() in home]

    def _attach(self, text, status=False):
        ch = self.s.chat
        if ch and isinstance(ch[-1], dict) and ch[-1].get("role") == "tool" and (text or status):
            self.s.engine.append_note(text, status=status)

    def _qty(self, slot):
        s = self.s
        prods = s.ledger.products() or []
        q = prods[slot].get("qty") if isinstance(slot, int) and 0 <= slot < len(prods) else s.ledger.quantity()
        return q if kit.is_num(q) and q > 0 else 1

    @staticmethod
    def _row_price(r):
        for k, v in r.items():
            if kit.is_num(v) and _p12_PRICE.search(str(k).lower()): return v
        return None

    @staticmethod
    def _table(obs):
        def tab(v):
            return isinstance(v, list) and all(isinstance(x, dict) for x in v)
        ts = [v for v in obs.values() if tab(v)]
        if ts: return obs, next((t for t in ts if t), ts[0])
        for v in obs.values():
            if isinstance(v, dict):
                ts = [x for x in v.values() if tab(x)]
                if ts: return v, next((t for t in ts if t), [])
        return obs, None

    # ------------------------------------------------------------------ 3.1 the search note
    def query_memo(self, args, obs):
        out = []
        try:
            self._note(args if isinstance(args, dict) else {}, obs, out)
        except Exception as e:
            self._fail("search_note_error", e)
        return " ".join(x for x in out if x)

    def _note(self, args, obs, out):
        s, sk = self.s, self.s.seeker
        if not isinstance(obs, dict): return
        rec, rows = self._table(obs)
        if rows is None: return
        shown = len(rows)
        totals = [v for v in rec.values() if isinstance(v, int) and not isinstance(v, bool) and v > shown]
        count = max(totals) if totals else shown
        sk.log_page(args, rows, count, obs)
        for kind, fn in (("twin_learn_error", lambda: sk.study_pairs(args, rows)),
                         ("cap_learn_error", lambda: self._learn_cap(args, rows, count, rec))):
            try:
                fn()
            except Exception as e:
                self._fail(kind, e)
        texts = [str(t) for _, t in (sk.open_text(args) or []) if str(t).strip()]
        raw = " ".join(texts)
        weight = (sk.value_rating(raw) or 0) if rows and raw else 0
        if weight:
            for r in rows:
                lid = kit.row_id(r)
                if lid: s.row_weight[lid] = max(s.row_weight.get(lid, 0), weight)
        sk.learn_words(args, obs, rows, count)
        n = s.lead_words or 0
        if n > 0:
            fr = s.lead_frame
            fld = str(fr[0]) if fr else ""
            ws = str(args.get(fld) or "").split() if fr and self._same_modes(fr[1], args) else []
            t = ("[query length] This store reads only the first %d word%s of a text query%s; any later word is "
                 "ignored." % (n, "" if n == 1 else "s", " in '%s'" % fld if fld else ""))
            if len(ws) > n: t += " This one counts as '%s'." % " ".join(ws[:n])
            out.append(t + " Put the %d most telling words first (the required value, then the item kind) and give "
                           "every other value a short query of its own." % n)
        if (sk.search_parts(args) or {}).get("first") and texts:
            self._query_remarks(args, rows, totals, count, texts, raw, weight, out)
        if not rows: return
        pn = self._pnames(args)
        if totals:
            top = max(totals)
            t = ("[partial] This reply is partial: %d matches, %d row%s shown, and only some pages can be read, so "
                 "fitting offers may still be hidden. First read the further pages of this same query as far as they "
                 "go; if the total is more than those pages can show, " % (top, shown, "" if shown == 1 else "s"))
            if s.word_mode == "any":
                t += ("bear in mind that one shared word is enough for a match here and offers sharing more words rank "
                      "higher, so adding words makes the result bigger: a query of a single required value's "
                      "distinctive word or figure returns only offers that have it (often few enough to read fully "
                      "and compare prices), queries of kind plus values bring the closest offers to the top, and the "
                      "category mode or a price cap that works covers the rest.")
            elif s.word_mode == "all":
                t += "narrow it: each exact word added shrinks the result, and so does a category with a lower price cap."
            else:
                t += ("shrink it: if all words must match, add exact words; if one shared word is enough (a longer "
                      "query then gives more offers), put the required values in so matching offers rank first; a "
                      "price cap that works helps as well.")
            pg = self._pg()
            nc = sk.next_page_call(args, obs) if pg else None
            if isinstance(nc, dict) and pg.get("name") in nc:
                t += " To read on, repeat the same call with %s." % json.dumps({pg["name"]: nc[pg["name"]]},
                                                                             ensure_ascii=False)
            out.append(t)
        caps = [v for k, v in args.items() if kit.is_num(v) and v > 1 and k not in pn
                and not _p12_NOCAP.search(str(k).lower())]
        if caps:
            top = max(caps)
            over = sum(1 for r in rows if any(kit.is_num(v) and "price" in str(k).lower() and v > top + 1e-9
                                              for k, v in r.items()))
            if over:
                out.append("[cap ignored] %d row%s cost more than %s although the call asked for that price cap: this "
                           "search mode seems to skip the price filter, and maybe the category filter as well. Use a "
                           "mode in which the filters hold, or check each row on its own."
                           % (over, "" if over == 1 else "s", self._money(top)))

    def _take_hits(self, hits, weight):
        s = self.s
        for h in hits:
            s.hit_weight[h] = max(s.hit_weight.get(h, 0), weight)
            if h not in s.all_hits: s.all_hits.append(h)

    def _query_remarks(self, args, rows, totals, count, texts, raw, weight, out):
        s, sk = self.s, self.s.seeker
        roomy = s.steps_left() - 3 > s.intake.due_margin()
        toks = _p12_AN.findall(raw.lower())
        if toks and not rows and not self._short_hint and all(len(t) == 1 for t in toks):
            self._short_hint = True
            out.append("[short tokens] Nothing came back: this store ignores tokens of a single letter or digit, so a "
                       "figure written that way adds nothing. Write the value out as a word, or pair its unit with the "
                       "item kind, and read that result to its end.")
        probe = sk.value_test(texts)
        if probe:
            if self._probe_hints < 2 and roomy:
                self._probe_hints += 1
                if self._probe_hints == 1:
                    t = ("[property name] This query starts with the name of the property '%s' and leaves out the "
                         "wanted value. The store matches field values, titles and category text; a field's name "
                         "helps only when some value or title also holds it, so a listing that shows the value only "
                         "in that field cannot come back." % probe[0])
                else: t = "[property name] Again the name of the property '%s' leads the query." % probe[0]
                out.append(t + " Try '%s' (the value before the item kind; figures the store cannot read drop out, "
                               "their unit words stay) and read it to its end." % probe[1])
        elif self._trap_hints < 2 and roomy:
            trap = sk.term_snare(texts)
            if trap and trap[0]:
                self._trap_hints += 1
                out.append(str(trap[0]))
        pn, rawn = self._pnames(args), {k for k, _ in (sk.open_text(args) or [])}
        ids = tuple(kit.row_id(r) for r in rows)
        sig = (tuple(self._modes(args)), tuple(sorted((k, str(v)) for k, v in args.items()
                                                     if k not in pn and k not in rawn)), count, ids)
        q = raw.strip().lower()
        prev = self._sigs.setdefault(sig, q)
        if prev != q and len({i for i in ids if i}) >= 3:
            out.append("[same result] This reply equals the one for '%s' (same total, same rows): the store treats "
                       "both wordings as one search. Read its next page, or start the query with different words."
                       % prev)
        full = bool(rows) and not totals
        known = s.word_mode == "all" or (s.lead_words or 0) > 0
        told, hits = full and known, []
        if told or (full and s.word_mode != "any"):
            before = list(s.ref_pointers)
            hits = [h for h in (sk.row_matches(rows, args) or []) if h]
            if not told:
                s.ref_pointers[:] = before
            self._take_hits(hits, weight)
            if told and hits and toks and len(rows) <= 12:
                out.append("[full result] Open these in-budget entries before any further query: %s. Every query "
                           "word must match here and the whole result is on this page, so each listed entry carries "
                           "those words somewhere in its record (often not in its title)." % ", ".join(hits[:6]))
            if told:
                ptr = []
                for p in list(s.ref_pointers):
                    pid = p if isinstance(p, str) else kit.row_id(p) if isinstance(p, dict) else ""
                    if pid and pid not in hits and pid not in ptr: ptr.append(pid)
                s.ref_pointers[:] = []
                if ptr:
                    out.append("[reference listings] %s belong to the shop or brand this query names. Open them "
                               "whatever their price, stock or category, read the value they show and follow their "
                               "notes, but do not buy them: the request defines a value through another offer."
                               % ", ".join(ptr[:6]))
        if known and not self._retro:
            self._retro = True
            key, got = sk.book_key(args), []
            for k, e in list((sk.book() or {}).items()):
                if k == key or e.get("rows") is None: continue
                w = sk.value_rating(e.get("text") or "") or 0
                if w:
                    new = [h for h in (sk.row_matches(e["rows"], e.get("args") or {}) or [])
                           if h and h not in got and h not in hits]
                    self._take_hits(new, w)
                    got += new
            if got:
                out.append("[earlier results] Open these unread in-budget entries before any further query: %s. "
                           "Every word must match in this store, so the earlier fully shown results can only hold "
                           "entries that carry all of their query's words (often not in the title)." % ", ".join(got[:6]))

    # ------------------------------------------------------------------ 3.2 price-cap learning
    def _learn_cap(self, args, rows, count, rec):
        sk, pg = self.s.seeker, self._pg()
        pname = pg.get("name") if pg else None
        key = sk.book_key(args)
        self._sent[key] = int(count)
        self._sent_args[key] = {k: v for k, v in args.items() if k != pname}
        if pg and pg.get("mode") == "page" and rows and isinstance(rec, dict):
            flags = [v for k, v in rec.items() if _p12_FLAG.search(str(k).lower()) and not isinstance(v, (list, dict))]
            per = max(len(rows), self.page_rows())
            ended = bool(flags) and all(v is None or v is False or (isinstance(v, str) and not v.strip()) for v in flags)
            pos, start = args.get(pname), pg.get("start")
            if kit.is_num(pos) and kit.is_num(start) and ended and len(rows) >= per:
                depth = int((pos - start + 1) * per)
                if count > depth: self._depth_cap = min(self._depth_cap or depth, depth)
        pn = self._pnames(args)
        cands = [k for k, v in args.items() if kit.is_num(v) and k not in pn and v > 1
                 and not _p12_NOLIM.search(str(k).lower())]
        if len(cands) != 1 or not rows: return
        lim, props = cands[0], self._props()
        ceiling = args[lim]
        shelves = {kit.shelf_field(r) or "" for r in rows}
        sa = [k for k, v in args.items() if k not in pn and isinstance(v, str) and v.strip()
              and v.strip().lower() in shelves]
        shelf = sa[0] if len(sa) == 1 else None
        modes = self._modes(args, props)
        fixed = [(k, v) for k, v in modes if k != shelf]
        sig = self._canon([fixed, lim])
        prices = [self._row_price(r) for r in rows]
        if any(p is not None and p > ceiling + 1e-9 for p in prices):
            self._broken.add(sig)
            return
        if sig in self._broken or not shelf or len(shelves) > 1 or "" in shelves or any(p is None for p in prices):
            return
        mnames, tmpl = {k for k, _ in modes}, {}
        for k, v in args.items():
            if k in pn or k in (lim, shelf): continue
            if isinstance(v, str) and v.strip() and k not in mnames:
                lv = v.strip().lower()
                if all(any(isinstance(fv, str) and fv.strip().lower() == lv and not _p12_CAT.search(str(fk).lower())
                           for fk, fv in r.items()) for r in rows):
                    continue  # a filter every row shares (a maker field): later cap listings stay unnarrowed
            tmpl[k] = v
        free = sum(len(v.split()) for k, v in tmpl.items() if isinstance(v, str) and k not in mnames)
        old = self._prof
        if old and old["sig"] not in self._broken and old["free"] < free: return
        self._prof = {"sig": sig, "template": tmpl, "modes": fixed, "shelf": shelf, "limit": lim, "free": free,
                      "int": (props.get(lim) or {}).get("type") == "integer"}

    def cap_profile(self):
        p = self._prof
        return p if p and p["sig"] not in self._broken else None

    def _break(self):
        if self._prof: self._broken.add(self._prof["sig"])

    def is_cap_listing(self, args):
        try:
            p = self.cap_profile()
            if not p or not isinstance(args, dict): return False
            if p["modes"]: return all(k in args and args[k] == v for k, v in p["modes"])
            return p["limit"] in args
        except Exception as e:
            self._fail("cap_listing_error", e)
            return False

    def sent_searches(self):
        return dict(self._sent)

    # ------------------------------------------------------------------ 3.3 page size, visible depth
    def page_rows(self):
        try:
            return max([int(e.get("size") or 0) for e in (self.s.seeker.book() or {}).values()] or [0])
        except Exception as e:
            self._fail("page_size_error", e)
            return 0

    def seen_depth(self):
        try:
            pg, ps = self._pg(), self.page_rows()
            if pg and ps > 0 and pg.get("mode") != "token" and kit.is_num(pg.get("top")) and kit.is_num(pg.get("start")):
                span = pg["top"] - pg["start"]
                return int((span + 1) * ps if pg.get("mode") == "page" else span + ps)
            return self._depth_cap
        except Exception as e:
            self._fail("depth_error", e)
            return self._depth_cap

    # ------------------------------------------------------------------ 3.4 shared price and row helpers
    def bare_label(self, shelves):
        try:
            want = frozenset(kit.low(x).strip() for x in (shelves or ()) if x)
            return self._memo("label", want, lambda: self._label_scan(want)) if want else ""
        except Exception as e:
            self._fail("raw_label_error", e)
            return ""

    def _label_scan(self, want):
        known, tally = self.s.archive.known, {}

        def fn(d):
            if not any("id" in str(k).lower() and isinstance(v, str) and v in known for k, v in d.items()): return
            for k, v in d.items():
                kl = str(k).lower()
                if isinstance(v, str) and v.strip() and len(v) <= 60 and "id" not in kl and _p12_CAT.search(kl):
                    if v.strip().lower() in want:
                        tally[v.strip()] = tally.get(v.strip(), 0) + 1
                    return
        for o in self.s.archive.obs: kit.visit(o, fn, top=6)
        return min(tally, key=lambda t: (-tally[t], t)) if tally else ""

    def below(self, price):
        if not kit.is_num(price): return None
        if float(price).is_integer(): r = int(price) - 1
        else:
            p = self.cap_profile()
            r = int(price) if p and p["int"] else round(price - 0.01, 2)
        return r if r > 0 else None

    def mid(self, lo, hi):
        if not (kit.is_num(lo) and kit.is_num(hi)): return hi if kit.is_num(hi) else lo
        p = self.cap_profile()
        if (p and p["int"]) or (float(lo).is_integer() and float(hi).is_integer()): return (int(lo) + int(hi)) // 2
        return round((lo + hi) / 2, 2)

    def _cap_call(self, label, limit):
        p = self.cap_profile()
        c = dict(p["template"])
        c[p["shelf"]] = label
        c[p["limit"]] = int(limit) if p["int"] or float(limit).is_integer() else limit
        return c

    def aisle_rows(self, shelves):
        try:
            sh, known = {kit.low(x).strip() for x in (shelves or ()) if x}, self.s.archive.known
            return [lid for lid, r in (self.s.archive.digest() or {}).items()
                    if lid in known and kit.low(r.get("shelf")).strip() in sh]
        except Exception as e:
            self._fail("home_rows_error", e)
            return []

    def _clash(self, shelves, bound):
        sh = {kit.low(x).strip() for x in (shelves or ()) if x}
        for lid, r in (self.s.archive.digest() or {}).items():
            p = r.get("price")
            if (lid in self.s.archive.known and kit.low(r.get("shelf")).strip() in sh and kit.is_num(p)
                    and p <= bound + 1e-9 and r.get("stock") is True):
                return True
        return False

    def low_unopened(self, ids, slot, below=None):
        try:
            s = self.s
            units, bud, out, seen = self._qty(slot), s.ledger.budget, [], set()
            for lid in ids or ():
                if lid in seen: continue
                seen.add(lid)
                if lid in s.archive.read or lid in s.refused_lines or s.judge.supply_state(lid) is False: continue
                p = s.archive.price(lid)
                if not kit.is_num(p) or (kit.is_num(bud) and p * units > bud + 1e-9): continue
                if below is not None and p >= below - 1e-9: continue
                out.append((p, lid))
            return sorted(out)
        except Exception as e:
            self._fail("cheap_unread_error", e)
            return []

    def trim(self, queue, floor):
        try:
            q, batch = list(queue or []), bool(self._batch())
            while q:
                cost = 1 if batch and len(q) >= 2 else len(q)
                if cost < self.s.steps_left() - floor: break
                q.pop()
            return q
        except Exception as e:
            self._fail("trim_error", e)
            return []

    def launch(self, queue, shelves, floor, say):
        s = self.s
        try:
            sh = {kit.low(x).strip() for x in (shelves or ()) if x}
            for m in list(s.rank_marks):
                if kit.low(s.judge.label_of(str(m).split("|")[0])).strip() in sh: s.rank_marks.discard(m)
            s.order_vetted = False
            s.judge.queue_open(list(queue), "walk", recheck=True, floor=floor, say=say)
            return bool(s.judge.open_queued())
        except Exception as e:
            self._fail("launch_error", e)
            return False

    def every_term(self):
        s = self.s
        return s.word_mode == "all" or (s.word_mode != "any" and (s.lead_words or 0) > 0)

    def value_signature(self, args):
        props = self._props()

        def small(k):
            en = self._enum(k, props)
            return en is not None and len(en) <= 4
        return self._canon(sorted((k, v) for k, v in (args or {}).items() if isinstance(v, str) and small(k)))

    def _stems(self, text):
        ws = _p12_AN.findall(kit.low(text))
        n = self.s.lead_words or 0
        return frozenset(kit.stem(w) for w in (ws[:n] if n > 0 else ws))

    # ------------------------------------------------------------------ requirement lines of a slot
    def _own(self, slot, exact=False):
        s = self.s
        lines = tuple(str(x) for x in (s.ledger.in_force() or []))

        def build():
            multi, out = s.multi(), []
            for n, ln in enumerate(lines):
                front, colon, val = ln.partition(":")
                if not colon: front, val = "", ln
                if multi and slot is not None:
                    p = s.mender.item_for(front) if front.strip() else None
                    if p != slot and (exact or p is not None): continue
                out.append((n, ln, kit.low(kit.name_head(front)).strip(), kit.low(kit.strip_marks(val)).strip()))
            return out
        return self._memo("own", (slot, exact, lines), build)

    @staticmethod
    def _alts(v):
        return [p.strip() for p in _p12_ALTS.split(v or "") if p and p.strip()]

    @staticmethod
    def _is_amount(t):
        return bool(re.search(r"\d", t)) and (kit.has_bound(t) or bool(_p12_AMT.match(t)))

    def _toks(self, args):
        j = " ".join(kit.low(t) for _, t in (self.s.seeker.open_text(args) or []))
        words, toks, n = _p12_AN2.findall(j), _p12_AN.findall(j), self.s.lead_words or 0
        return (words[:n], toks[:n]) if n > 0 else (words, toks)

    def _carry(self, args, slot):
        """(requirement index, alternative index) pairs the query's opening words carry (10 3.12 / 08 3.7)."""
        lines = tuple(str(x) for x in (self.s.ledger.in_force() or []))
        return self._memo("carry", (self._canon(args or {}), slot, lines), lambda: self._carry_scan(args or {}, slot))

    def _carry_scan(self, args, slot):
        words, toks = self._toks(args)
        out = set()
        if not toks: return out
        for n, _, _, v in self._own(slot):
            for a, alt in enumerate(self._alts(v)):
                m = None if kit.has_bound(alt) else _p12_NU.match(alt)
                if m: hit = m.group(1) in toks and any(t == m.group(2) or kit.near(t, m.group(2)) for t in toks)
                elif self._is_amount(alt): hit = False
                else:
                    hit = (any(c in words for c in _p12_CODE.findall(alt))
                           or any(kit.near(w, p) for w in words for p in _p12_AN3.findall(alt)))
                if hit: out.add((n, a))
        return out

    def _own_stems(self, slot, home):
        st = set()
        for _, _, _, v in self._own(slot):
            st |= {kit.stem(w) for w in _p12_AN2.findall(v)}
        for h in home:
            st |= {kit.stem(w) for w in _p12_AN2.findall(h)}
        return st

    # ------------------------------------------------------------------ 3.9.1-3.9.3 value words, label words, forms
    def value_words(self, slot):
        try:
            lines = tuple(str(x) for x in (self.s.ledger.in_force() or []))
            return [list(o) for o in self._memo("vw", (slot, lines), lambda: self._value_words(slot))]
        except Exception as e:
            self._fail("value_words_error", e)
            return []

    def _value_words(self, slot):
        out = []
        for prop, val in self.s.ledger.values_in_code(slot) or []:
            v = _p12_PAREN.sub(" ", kit.low(val).replace("\u2019", "'").replace("\u2018", "'"))
            if not v.strip() or kit.is_pointer(v): continue
            head = set(_p12_AN.findall(kit.low(prop)))
            if _p12_CLOSED.search(v): pieces = [p.strip() for p in _p12_CLOSED_SPLIT.split(v) if p and p.strip()][:3]
            else: pieces = [p.strip() for p in _p12_AND_SPLIT.split(v) if p and p.strip()][:1]
            opts = []
            for p in pieces:
                w = self._piece_words(p, head)
                if w and w not in opts: opts.append(w)
            if opts: out.append(opts)
        return out

    def _piece_words(self, p, head):
        if kit.has_exclusion(p): return []
        if re.search(r"\d", p) and (kit.has_bound(p) or _p12_DEC.search(p)) and not _p12_NU.match(p): return []
        t = kit.strip_links(_p12_POSS.sub("", p))
        lead = _p12_NU.match(t)
        num, unit = (lead.group(1), lead.group(2)) if lead else (None, None)
        out = []
        for x in _p12_TOK.findall(t):
            if "." in x or "," in x or kit.is_link(x): continue
            if (len(x) >= 2 or x == num) and (x not in head or x == unit): out.append(x)
        return out

    @staticmethod
    def _label_tokens(label):
        t = _p12_POSS.sub("", kit.low(label).replace("\u2019", "'"))
        return list(dict.fromkeys(w for w in _p12_AN.findall(t) if len(w) >= 2 and not kit.is_link(w)))

    def label_words(self, slot):
        try:
            s, home = self.s, self._home(slot)
            lw = self._label_tokens(self.bare_label(home))
            if self._indexing is not False: return lw
            dig = s.archive.digest() or {}
            titles = [{kit.stem(w) for w in _p12_AN.findall(kit.low((dig.get(i) or {}).get("title")))}
                      for i in self.aisle_rows(home)]
            keep = [w for w in lw if titles and 2 * sum(1 for t in titles if kit.stem(w) in t) >= len(titles)]
            if keep: return keep
            prods = s.ledger.products() or []
            name = (prods[slot].get("name") if isinstance(slot, int) and 0 <= slot < len(prods)
                    else s.ledger.product_name()) or ""
            ws = [w for w in _p12_AL.findall(kit.low(name)) if len(w) > 2 and not kit.is_link(w)]
            return ws[-1:]
        except Exception as e:
            self._fail("label_words_error", e)
            return []

    def _shape(self, slot):
        s, sch = self.s, self._schema()
        props, req = sch.get("properties") or {}, set(sch.get("required") or [])
        for c in s.closer.search_calls(slot) or []:
            if not isinstance(c, dict): continue
            ft = [(k, t) for k, t in (s.seeker.open_text(c) or []) if str(t).strip()]
            if len(ft) != 1 or self.is_cap_listing(c): continue
            fld = ft[0][0]
            tmpl = {k: v for k, v in c.items() if k == fld or self._enum(k, props) or k in req}
            modes = {k: v for k, v in tmpl.items() if self._enum(k, props)}
            ml = (props.get(fld) or {}).get("maxLength")
            return tmpl, fld, (ml if kit.is_num(ml) and ml > 0 else None), modes
        return None

    def _compose(self, chosen, lw, maxlen):
        out, taken = [], set()
        for w in [w for o in chosen for w in o] + list(lw):
            st = kit.stem(w)
            if st not in taken:
                taken.add(st)
                out.append(w)
        n = self.s.lead_words or 0
        t = " ".join(out[:n] if n > 0 else out)
        if maxlen:
            while len(t) > maxlen and " " in t: t = t.rsplit(" ", 1)[0]
            t = t[:int(maxlen)]
        return t

    def _tiers(self, slot, maxlen):
        vw = self.value_words(slot)
        if not vw: return [], []
        lw, ch = self.label_words(slot), self._cheapest()

        def use(opts):
            return opts if ch else opts[:1]

        def dist(xs):
            return list(dict.fromkeys(x for x in xs if x))
        single = dist(self._compose([o], lw, maxlen) for o in use(vw[0]))
        if len(vw) == 1: return single, []
        wide = next((i for i, r in enumerate(vw) if len(use(r)) > 1), 0)
        heads = [r[0] for r in vw]
        prim = dist(self._compose(heads[:wide] + [o] + heads[wide + 1:], lw, maxlen) for o in use(vw[wide]))
        return prim, single

    def value_forms(self, slot):
        try:
            sh = self._shape(slot)
            a, b = self._tiers(slot, sh[2] if sh else None)
            return list(dict.fromkeys(a + b))
        except Exception as e:
            self._fail("value_forms_error", e)
            return []

    def _form_count(self, form, modes, fld):
        st, props, hit = self._stems(form), self._props(), None
        for k, a in self._sent_args.items():
            if fld in a and {x: y for x, y in a.items() if self._enum(x, props)} == modes \
                    and self._stems(str(a[fld])) == st:
                hit = self._sent.get(k)
        return hit

    def _next_form(self, tiers, modes, fld):
        """A form to send, "" when the value search is done, None for "fallback"."""
        had = False
        for tier in tiers:
            if not tier: continue
            had = True
            counts = [self._form_count(f, modes, fld) for f in tier]
            for f, c in zip(tier, counts):
                if c is None: return f
            if any(c > 0 for c in counts): return ""
        return None if had else ""

    # ------------------------------------------------------------------ 3.9.4-3.9.5 the value-led search run
    def _value_search(self):
        s = self.s
        if self._room()[1] <= 4: return False
        slots = self._slots()
        for slot in slots:
            if self._tries.get(slot, 0) >= Tune.Search.LABEL_TRIES: continue
            sig = self._canon(self.value_words(slot))
            if sig in self._settled.get(slot, []): continue
            home = self._home(slot)
            label = self.bare_label(home)
            if not label: continue
            if self._value_try(slot, home, label, len(slots)): return True
            self._settled.setdefault(slot, []).append(sig)
        return False

    def _value_try(self, slot, home, label, nslots):
        s, sk = self.s, self.s.seeker
        sh = self._shape(slot)
        if not sh: return False
        tmpl, fld, maxlen, modes = sh
        comp = self._composed.setdefault(slot, set())
        tiers = self._tiers(slot, maxlen)
        comp.update(self._stems(f) for f in tiers[0] + tiers[1])
        nxt = self._next_form(tiers, modes, fld)
        if nxt is None:
            nxt = ""
            if self._indexing is None and any(self._carry(e.get("args") or {}, slot) and self._on(e, home)
                                              for e in (sk.book() or {}).values()):
                self._indexing = False
                tiers = self._tiers(slot, maxlen)
                comp.update(self._stems(f) for f in tiers[0] + tiers[1])
                nxt = self._next_form(tiers, modes, fld) or ""
        if not nxt: return False
        call = dict(tmpl)
        call[fld] = nxt
        fixed, faults = ArgFixer.fix(call, self._schema())
        if faults: return False
        call = fixed if isinstance(fixed, dict) else call
        self._tries[slot] = self._tries.get(slot, 0) + 1
        s.walker.search_call(call, "Value search '%s': the required value comes first, then the category name as this "
                                   "store spells it. The store matches listing titles, category text and other listing "
                                   "fields, so the result should show this category's items that carry that value."
                             % nxt)
        total = self._form_count(nxt, modes, fld)
        self._detect_indexing(label)
        key = sk.book_key(call)
        e = sk.book().get(key)
        onh = self._on(e, home) if e else []
        allon = bool(e) and total is not None and self._seen(e) >= total and bool(onh)
        floor, queue, ch = s.intake.due_margin() + 2, [], self._cheapest()
        if allon:
            if ch:
                if self._pg() and e.get("id_order") is not False:
                    self._done_keys.add(key)
                    self._add_vkey(slot, key)
                    if nslots > 1: self._waiting[slot] = True
                    else: queue = self._value_queue(slot, onh, floor, 3)
            elif nxt in (tiers[0] or tiers[1]): queue = self._value_queue(slot, onh, floor, 3)
        if total is None: note = ""
        elif total == 0: note = "[value search] '%s': zero matches found." % nxt
        else:
            note = "[value search] '%s': %d match%s%s" % (nxt, total, "" if total == 1 else "es",
                                                         " - all of them on the page." if allon else ".")
        if queue:
            note += (" Opened next, cheapest first: the still-unread, in-stock, in-budget listings of the requested "
                     "category%s: %s." % (" below the pick" if ch and self._pick(slot) else "", self._priced(queue)))
        self._attach(note, not queue)
        s.link.note("value_search", slot=slot, query=nxt, total=total, all_on_page=allon, opened=list(queue))
        if queue:
            self.launch(queue, home, floor, "The cheapest unread hits are read now: every item of this category "
                                            "that has the required values is inside this result.")
        return True

    def _detect_indexing(self, label):
        s = self.s
        if self._indexing is not None or not s.archive.obs: return
        want = {kit.stem(w) for w in self._label_tokens(label)}
        lab, known, hits = kit.low(label).strip(), s.archive.known, [0]

        def fn(d):
            if kit.row_id(d) in known and kit.shelf_field(d) == lab:
                st = set()
                for k, v in d.items():
                    if isinstance(v, str) and not _p12_CAT.search(str(k).lower()):
                        st |= {kit.stem(w) for w in _p12_AN.findall(v.lower())}
                if want - st: hits[0] += 1
        kit.visit(s.archive.obs[-1], fn, top=3)
        if hits[0] > 1: self._indexing = True

    def _add_vkey(self, slot, key):
        ks = self._vkeys.setdefault(slot, [])
        if key in ks: ks.remove(key)
        ks.append(key)

    def value_keys(self, slot):
        return list(self._vkeys.get(slot, []))

    def _value_queue(self, slot, ids, floor, most):
        ch = self._cheapest()
        pk = self._pick(slot) if ch else None
        ranked = [lid for _, lid in self.low_unopened(ids, slot, pk[1] if pk else None)]
        quota = 3 - self._vopens.get(slot, 0) if ch else most
        bm = self._batch_max()
        q = self.trim(ranked[:max(0, min(bm if bm else most, quota))], floor)
        if ch: self._vopens[slot] = self._vopens.get(slot, 0) + len(q)
        return q

    # ------------------------------------------------------------------ 3.10 value pages (code move)
    def value_sheets(self):
        try:
            return self._value_pages()
        except Exception as e:
            self._fail("value_pages_error", e)
            return False

    def _value_pages(self):
        s, sk = self.s, self.s.seeker
        if any(isinstance(r, dict) and r.get("status") == "open" for r in s.refs) and s.tracer.has_room():
            return False
        if (s.orders_tried or s.refusals or s.link.done or s.link.broke or s.link.time_left() < 130
                or not self._tool()):
            return False
        if self._value_search(): return True
        if not self._cheapest() or s.word_mode == "any" or not self._pg(): return False
        depth, rpp = self.seen_depth(), max(1, self.page_rows())
        vp = Tune.Search.VALUE_PAGES
        per_cap = vp if depth is None else max(vp, math.ceil(depth / rpp) - 1)
        slots = self._slots()
        several = len(slots) > 1
        total_cap = max(Tune.Search.VALUE_TOTAL, Tune.Search.VALUE_PER * len(slots))
        opens = 1 if self._batch() else (2 if several else 3)
        floor, known = 2 + s.intake.due_margin(), self.every_term()
        q, room = self._room()
        homes = {sl: self._home(sl) for sl in slots}
        untouched = sum(1 for sl in slots if homes[sl] and not self._vpages.get(sl))
        share = max(4, room // max(untouched, 1))
        left, book = s.steps_left(), sk.book() or {}
        fully = []
        if known:
            for k, e in book.items():
                if (e.get("id_order") is True and self._first(e) and sk.following_page(k) is None
                        and not self.is_cap_listing(e.get("args") or {}) and self._seen(e) >= (e.get("total") or 0)):
                    fully.append((k, self._stems(e.get("text") or ""), self.value_signature(e.get("args") or {}), e))
        ctx = dict(depth=depth, known=known, opens=opens, floor=floor, left=left, q=q, room=room, several=several,
                   per_cap=per_cap, total_cap=total_cap, share=share, fully=fully, pending=False)
        opts = []
        for order, sl in enumerate(slots):
            home = homes[sl]
            if not home: continue
            cands = [c for c in (self._weigh(sl, k, e, home, ctx) for k, e in book.items()) if c]
            if known and len(cands) > 1:
                own = self._own_stems(sl, home)
                keep = []
                for a in cands:
                    drop = any(b is not a and ((b[3] < a[3] and not (a[3] - b[3]) <= own)
                                               or (a[3] < b[3] and (b[3] - a[3]) <= own)) for b in cands)
                    if not drop: keep.append(a)
                cands = keep
            opts += [(self._vpages.get(sl, 0), c[0], c[1], order, c[2], sl) for c in cands]
        if opts:
            o = min(opts, key=lambda x: x[:5])
            self._done_keys.add(o[4])
            return self._read_value(o[5], o[4], floor, o[1], several)
        if self._waiting and not ctx["pending"] and not (q > 0 and room > opens * len(self._waiting) + 4):
            return self._flush(floor)
        return False

    def _weigh(self, sl, key, e, home, c):
        sk = self.s.seeker
        if (key in self._done_keys or e.get("id_order") is not True or not self._first(e)
                or sk.following_page(key) is None):
            return None
        args = e.get("args") or {}
        if self.is_cap_listing(args) or not self._value_led(e, sl) or not self._on(e, home): return None
        words = self._stems(e.get("text") or "")
        composed = words in self._composed.get(sl, set())
        tot, depth, known = e.get("total") or 0, c["depth"], c["known"]
        if depth is None:
            if not known: return None
        elif tot > depth and not (known and composed): return None
        pairs = self._carry(args, sl)
        if known:
            msig = self.value_signature(args)
            own = None
            for ok, ow, osig, oe in c["fully"]:
                if ok == key or osig != msig: continue
                if ow <= words: return None
                if pairs:
                    own = self._own_stems(sl, home) if own is None else own
                    if ow <= own and pairs <= self._carry(oe.get("args") or {}, sl): return None
        reach = tot if depth is None else min(tot, depth)
        pages = math.ceil(reach / max(1, e.get("size") or 0)) - len(e.get("spots") or ())
        spent = self._vpages.get(sl, 0)
        allowed = min(c["per_cap"] - spent, c["total_cap"] - sum(self._vpages.values()))
        if c["several"]:
            used = self._vopens.get(sl, 0)
            allowed = min(allowed, c["share"] - spent - used - (0 if used else c["opens"]))
        take = pages if pages <= allowed else (allowed if composed else 0)
        if take <= 0 or take + c["opens"] + c["floor"] >= c["left"]: return None
        if (take > 2 or c["several"]) and c["q"] > 0 and c["room"] - take - c["opens"] > 4:
            c["pending"] = True
            return None
        return take, -len({n for n, _ in pairs}), key, words

    # ------------------------------------------------------------------ 3.11 reading one value search, flush
    def _page_run(self, key, room, floor, say, bump=None):
        sk, tries, got = self.s.seeker, 0, 0
        while tries < room:
            spot = sk.following_page(key)
            if spot is None or not self._can_list(floor): break
            tries += 1
            if bump: bump()
            if not self.s.walker.search_call(sk.page_call(key, spot), say): break
            got += 1
            if sk.following_page(key) == spot: break
        return tries, got

    def _read_value(self, slot, key, floor, most, defer):
        s, sk = self.s, self.s.seeker
        e = (sk.book() or {}).get(key)
        if not self._pg() or not e: return False
        words, strict = e.get("text") or "", self.every_term()
        say = ("Reading on through the pages of '%s': this store orders results by id, not by price%s, so every "
               "matched offer gets seen and the lowest price can be found."
               % (words, "; every word of a text query must match" if strict else ""))

        def bump():
            self._vpages[slot] = self._vpages.get(slot, 0) + 1
        tries, got = self._page_run(key, most, floor, say, bump)
        if not got: return tries > 0
        self._add_vkey(slot, key)
        e = (sk.book() or {}).get(key) or e
        shown, total, nxt = self._seen(e), e.get("total") or 0, sk.following_page(key)
        complete = nxt is None and total <= shown
        if complete: tail = " - complete%s." % (" for those words" if strict else "")
        elif nxt is None: tail = " - the remaining pages cannot be reached, and a cheaper offer may be among them."
        else: tail = "."
        head, queue = "[value search read] '%s' %d/%d%s" % (words, shown, total, tail), []
        home = self._home(slot)
        if defer:
            self._waiting[slot] = True
            body = (" Opening waits until the value searches of every product are read; then each product's "
                    "cheapest unread, in-stock, in-budget entries are opened.")
        else:
            queue = self._value_queue(slot, self._on(e, home), floor, 3)
            if queue:
                body = (" Opening now, cheapest first (requested category%s): %s."
                        % (", all below the pick" if self._pick(slot) else "", self._priced(queue)))
            else:
                body = (" Nothing is left to open: every entry of the requested category here is read, sold out or "
                        "over budget.")
        text = head + body + (" Take the lowest-priced offer whose own record meets every requirement in force; a "
                              "dearer match needs every cheaper candidate opened and rejected first.")
        if queue: self.launch(queue, home, floor, "Opening the cheapest unread entries of '%s'." % words)
        self._attach(text, not queue)
        s.link.note("value_read", slot=slot, query=words, pages=got, complete=complete, opened=list(queue),
                    defer=bool(defer))
        return True

    def _flush(self, floor):
        s, sk = self.s, self.s.seeker
        due = list(self._waiting)
        self._waiting.clear()
        if not due: return False
        each = 2 if s.steps_left() - floor - 1 >= 2 * len(due) else 1
        pools, homes = [], set()
        for sl in due:
            home = self._home(sl)
            homes |= home
            ids = []
            for k in self._vkeys.get(sl, []):
                e = (sk.book() or {}).get(k)
                if e and self._value_led(e, sl): ids += [i for i in self._on(e, home) if i not in ids]
            pk = self._pick(sl)
            r = [lid for _, lid in self.low_unopened(ids, sl, pk[1] if pk else None)]
            pools.append((sl, r[:max(0, min(each, 3 - self._vopens.get(sl, 0)))]))
        bm = self._batch_max()
        limit, dealt, owner, cur = (bm if bm else each * len(due)), [], {}, [0] * len(pools)
        moved = True
        while moved and len(dealt) < limit:
            moved = False
            for j, (sl, p) in enumerate(pools):
                while cur[j] < len(p) and p[cur[j]] in owner: cur[j] += 1
                if cur[j] < len(p) and len(dealt) < limit:
                    owner[p[cur[j]]] = sl
                    dealt.append(p[cur[j]])
                    cur[j] += 1
                    moved = True
        q = self.trim(dealt, floor)
        if not q: return False
        for lid in q: self._vopens[owner[lid]] = self._vopens.get(owner[lid], 0) + 1
        self.launch(q, homes, floor, "The lowest price is wanted, so every product now gets the cheapest unread hits "
                                     "of its value query opened.")
        self._attach("[value searches read] For each product, below its pick, the unread in-stock in-budget hits of "
                     "its value query are read now, cheapest first: %s. For each product choose the cheapest listing "
                     "whose own fields meet every requirement in force; take a dearer fit only after every cheaper "
                     "candidate was opened and rejected." % self._priced(q), False)
        s.link.note("value_flush", slots=due, opened=list(q))
        return True

    # ------------------------------------------------------------------ 3.12 value-led, valued, complete, carries
    def _value_led(self, e, slot):
        args = e.get("args") or {}
        if self._carry(args, slot): return True
        if self.value_words(slot): return False
        toks = self._toks(args)[1]
        for _, _, _, v in self._own(slot):
            for alt in self._alts(v):
                if kit.has_bound(alt):
                    m = _p12_NU.match(kit.strip_links(alt))
                    if m and m.group(1) in toks and any(t == m.group(2) or kit.near(t, m.group(2)) for t in toks):
                        return True
        return False

    def value_hit_ids(self, slot):
        try:
            if not self.every_term(): return set()
            home, out = self._home(slot), set()
            for e in (self.s.seeker.book() or {}).values():
                args = e.get("args") or {}
                if e.get("id_order") is True and not self.is_cap_listing(args) and self._carry(args, slot):
                    out.update(self._on(e, home))
            return out
        except Exception as e:
            self._fail("valued_ids_error", e)
            return set()

    def value_complete(self, slot):
        try:
            if not self.every_term(): return False
            sk = self.s.seeker
            single = {n for n, _, _, v in self._own(slot) if len(self._alts(v)) == 1}
            if not single: return False
            home = self._home(slot)
            for k, e in (sk.book() or {}).items():
                if (e.get("id_order") is True and self._first(e) and sk.following_page(k) is None
                        and self._seen(e) >= (e.get("total") or 0) and self._on(e, home)
                        and {n for n, _ in self._carry(e.get("args") or {}, slot)} & single):
                    return True
            return False
        except Exception as e:
            self._fail("value_complete_error", e)
            return False

    def _body(self, lid):
        def build():
            parts = []

            def grab(v, d):
                if isinstance(v, str): parts.append(v.lower())
                elif d < 4 and isinstance(v, dict):
                    for x in v.values(): grab(x, d + 1)
                elif d < 4 and isinstance(v, list):
                    for x in v: grab(x, d + 1)

            def fn(d):
                if kit.row_id(d) == lid: grab(d, 0)
            for o in self.s.archive.obs: kit.visit(o, fn)
            return " ".join(parts)
        return self._memo("body", lid, build)

    def row_bears(self, lid, slot):
        try:
            amounts, wsets = [], []
            for _, _, _, v in self._own(slot):
                if kit.is_pointer(v): continue
                for alt in self._alts(v):
                    m = _p12_NU.match(alt)
                    if m: amounts.append((m.group(1), kit.stem(m.group(2))))
                    elif not self._is_amount(alt):
                        ws = {kit.stem(w) for w in _p12_AN3.findall(alt)}
                        if ws: wsets.append(ws)
            if not amounts and not wsets: return False
            body = self._body(lid)
            for num, unit in amounts:
                if re.search(r"(?<![0-9.,])" + re.escape(num) + r"(?:\s*|-)" + re.escape(unit), body): return True
            bw = {kit.stem(w) for w in _p12_AN3.findall(body)}
            return any(ws <= bw for ws in wsets)
        except Exception as e:
            self._fail("row_carries_error", e)
            return False

    # ------------------------------------------------------------------ 3.5 readiness, aim, bound need
    def _bound_need(self, slot):
        if not self.s.multi() or slot is None: return False
        return any(re.search(r"\d", v) and kit.has_bound(v) for _, _, _, v in self._own(slot, exact=True))

    def _ceiling_ready(self, slot):
        if self.s.multi() and slot is None: return False
        return (self._cheapest() and slot not in self._walked and bool(self.cap_profile()) and bool(self._pg())
                and bool(self.bare_label(self._home(slot))))

    def _aim(self, slot):
        s = self.s
        pk = self._pick(slot)
        marks = ([pk[1]] if pk else []) + ([self._held[slot]] if slot in self._held else [])
        if marks: return min(marks)
        if not self._bound_need(slot): return None
        home = self._home(slot)
        c = [p for p, _ in self.low_unopened(self.aisle_rows(home), slot)]
        for lid in s.archive.read:
            if (kit.low(s.judge.label_of(lid)).strip() in home and lid not in s.dead and lid not in s.refused_lines
                    and s.judge.supply_state(lid) is not False):
                p = s.archive.price(lid)
                if kit.is_num(p): c.append(p)
        return min(c) if c else None

    def _hold_aim(self, slot, ids):
        s = self.s
        ps = [s.archive.price(i) for i in ids if s.walker.product_slot(i) == slot]
        ps = [p for p in ps if kit.is_num(p)]
        if ps: self._held[slot] = min(ps + ([self._held[slot]] if slot in self._held else []))

    # ------------------------------------------------------------------ 3.6 the ceiling walk
    def cap_tour(self, slot, floor, repair=False):
        try:
            return self._ceiling(slot, floor, repair)
        except Exception as e:
            self._fail("ceiling_walk_error", e)
            return None

    def _ceiling(self, slot, floor, repair):
        s, sk = self.s, self.s.seeker
        aim, home = self._aim(slot), self._home(slot)
        label = self.bare_label(home)
        if aim is None or not self.cap_profile() or not self._pg() or not label: return None
        self._walked.add(slot)
        pk = self._pick(slot)
        self._aims[slot] = (aim, pk[0] if pk else "")
        limit = self.below(aim)
        if limit is None: return None
        if repair: floor += 2
        elif not s.orders_tried: floor = max(s.intake.due_margin() + 2, floor)
        what = "the pick" if pk or slot in self._held else "the cheapest row seen"
        say = ("Listing the category '%s' through the store's price cap, below %s at %s: the cheapest fit is wanted."
               % (label, what, self._money(aim)))
        most = Tune.Search.CEIL_CALLS
        calls, wide, complete, ran, bound, last, keys = 0, None, False, False, limit, limit, []
        state, key, e, total = "list", None, None, 0
        while True:
            if state == "list":
                if calls >= most or not self._can_list(floor) or not self.cap_profile(): break
                call = self._cap_call(label, bound)
                key = sk.book_key(call)
                self._cap_keys.add(key)
                calls += 1
                e = sk.book().get(key)
                if not (e and self._first(e) and key in self._sent) and not s.walker.search_call(call, say): break
                ran, last, state = True, bound, "check"
                keys.append(key)
            elif state == "check":
                e = sk.book().get(key)
                if e and any(kit.is_num(s.archive.price(i)) and s.archive.price(i) > bound + 1e-9
                             for i in e.get("row_shelf") or {}):
                    self._break()
                if not self.cap_profile(): break
                total = self._sent.get(key)
                if total is None: break
                if not total or not e:
                    state = "empty"
                    continue
                pages = math.ceil(total / max(1, e.get("size") or 0))
                depth = self.seen_depth()
                shallow = depth is None or depth >= total
                state = "tail" if shallow and most - calls >= pages - len(e.get("spots") or ()) else "lower"
            elif state == "empty":
                if self._clash(home, bound):
                    self._break()
                    break
                if wide is None:
                    complete = True
                    break
                m = self.mid(bound, wide)
                if not bound < m < wide: break
                bound, state = m, "list"
            elif state == "tail":
                tries, _ = self._page_run(key, most - calls, floor,
                                          "Paging on through the price-cap listing of '%s'." % label)
                calls += tries
                complete = sk.following_page(key) is None
                break
            else:
                wide = bound
                ps = [p for p in (s.archive.price(i) for i in e.get("row_shelf") or {}) if kit.is_num(p)]
                fp = min(ps) if ps else 0
                readable = (e.get("size") or 0) * max(1, most - calls)
                spread = fp + 1.8 * (bound - fp) * readable / max(total, 1)
                nxt = min(self.mid(fp, bound), self.mid(fp, spread))
                if nxt < bound: bound, state = nxt, "list"
                else: break
        if not ran: return None
        seen = []
        for k in keys:
            for i in self._on(sk.book().get(k) or {}, home):
                if i not in seen: seen.append(i)
        cands, ranked = self._rank_found(seen, slot, aim)
        top = ranked[:Tune.Search.CEIL_OPENS]
        if not complete:
            rep = ("the capped listing of '%s' under %s: %d cheaper row%s seen, %d of them openable"
                   % (label, self._money(aim), len(seen), "" if len(seen) == 1 else "s", len(cands)))
        elif seen:
            rep = ("the capped listing of '%s' is complete up to %s: %d row%s found, %d openable"
                   % (label, self._money(last), len(seen), "" if len(seen) == 1 else "s", len(cands)))
        else:
            rep = "the capped listing of '%s' under %s shows no listing cheaper than %s" % (label, self._money(aim), what)
        s.link.note("cap_tour", slot=slot, label=label, aim=aim, calls=calls, complete=complete, rows=len(seen),
                    top=list(top))
        return rep, top

    def _rank_found(self, pool, slot, below):
        cand = [lid for _, lid in self.low_unopened(pool, slot, below)]
        valued = self.value_hit_ids(slot)
        a = [i for i in cand if i in valued]
        b = [i for i in cand if i not in valued and self.row_bears(i, slot)]
        c = [] if self.value_complete(slot) else [i for i in cand if i not in valued and i not in b]
        return cand, a + b + c

    # ------------------------------------------------------------------ 3.7 ceiling sweep
    def cap_scan(self, tried):
        try:
            s, sk = self.s, self.s.seeker
            if not self.cap_profile() or not self._pg() or s.multi(): return None, False
            rows = sk.buyable_rows() or {}
            shelves = {kit.low(x).strip() for x in (sk.wanted_shelves() or ()) if x}
            label = self.bare_label(shelves)
            if not rows or not label: return None, False
            ps = [v[0] for v in rows.values() if v and kit.is_num(v[0])]
            cap = self.below(min(ps)) if ps else None
            if cap is None: return None, True
            tried, call = set(tried or ()), self._cap_call(label, cap)
            key = sk.book_key(call)
            if self._canon(call) not in tried:
                self._cap_keys.add(key)
                return call, False
            e = sk.book().get(key)
            if not e: return None, self._sent.get(key) == 0 and not self._clash(shelves, cap)
            spot = sk.following_page(key)
            if spot is None: return None, (e.get("total") or 0) <= self._seen(e)
            nc = sk.page_call(key, spot)
            return (nc if self._canon(nc) not in tried else None), False
        except Exception as e:
            self._fail("ceiling_sweep_error", e)
            return None, False

    # ------------------------------------------------------------------ 3.8 seeding the price cap
    def _seed(self):
        s = self.s
        try:
            if self.cap_profile() or self._prof is not None or not self._cheapest(): return False
            multi, pg = s.multi(), self._pg()
            pname = pg.get("name") if pg else None
            for sl in self._slots():
                if sl in self._seeded or (multi and sl is None): continue
                pk, label = self._pick(sl), self.bare_label(self._home(sl))
                if not pk or not label: continue
                self._seeded.add(sl)
                cap = self.below(pk[1])
                call = self._seed_call(label, cap) if cap is not None else None
                if not call: return False
                props, ll = self._props(), label.strip().lower()
                modes = sorted((k, v) for k, v in call.items() if isinstance(v, str) and self._enum(k, props)
                               and v.strip().lower() != ll)
                num = next((k for k, v in call.items() if kit.is_num(v) and k != pname), "")
                key = s.seeker.book_key(call)
                if self._canon([modes, num]) in self._broken or key in self._sent: continue
                self._seed_keys.add(key)
                s.walker.search_call(call, "Testing the store's price cap: the category '%s' below the pick's %s; the "
                                           "cheapest fit is wanted." % (label, self._money(pk[1])))
                s.link.note("cap_seed", slot=sl, label=label, cap=cap, learned=bool(self.cap_profile()))
                if not self.cap_profile(): self._attach("", True)
                return True
            return False
        except Exception as e:
            self._fail("cap_seed_error", e)
            return False

    def _seed_call(self, label, cap):
        s, tool = self.s, self._tool()
        sch = self._schema()
        props, pg = sch.get("properties") or {}, self._pg()
        if not props or not pg: return None
        pname, cur = pg.get("name"), s.seeker.cursor_param()
        desc = kit.low(((s.spec(tool) or {}).get("function") or {}).get("description") or "")
        clauses = re.split(r"[.;]", desc)
        shelf = sval = capk = None
        for k, p in props.items():
            if k in (cur, pname) or not _p12_CAT.search(str(k).lower()) or not isinstance(p, dict): continue
            en = p.get("enum")
            if not en and p.get("type", "string") == "string":
                shelf, sval = k, label
                break
            if en:
                m = next((c for c in en if str(c).strip().lower() == label.strip().lower()), None)
                if m is not None:
                    shelf, sval = k, m
                    break
        for k, p in props.items():
            kl = str(k).lower()
            if k == pname or not isinstance(p, dict) or p.get("enum") or p.get("type") not in ("integer", "number"):
                continue
            if (_p12_CAPN.search(kl) or kl.endswith("cap")) and not _p12_CAPX.search(kl):
                capk = k
                break
        if not shelf or not capk: return None
        call = {shelf: sval, capk: int(cap) if props[capk].get("type") == "integer" or float(cap).is_integer() else cap}
        sw = re.compile(r"\b" + re.escape(str(shelf).lower()) + r"\b")
        for k, p in props.items():
            en = p.get("enum") if isinstance(p, dict) else None
            if k in call or not isinstance(en, list) or not 2 <= len(en) <= 4: continue
            pick = next((c for c in en if any(sw.search(cl) and re.search(r"(?<!\w)" + re.escape(str(c).lower())
                                                                          + r"(?!\w)", cl) for cl in clauses)), None)
            if pick is None: return None
            call[k] = pick
        if any(r not in call for r in sch.get("required") or []): return None
        fixed, faults = ArgFixer.fix(call, sch)
        if faults: return None
        return fixed if isinstance(fixed, dict) else call

    # ------------------------------------------------------------------ 3.13 value guard
    def value_gate(self, name, args):
        try:
            return self._guard(name, args)
        except Exception as e:
            self._fail("value_guard_error", e)
            return ""

    def _guard(self, name, args):
        s = self.s
        if (not name or name != self._tool("details") or not self._cheapest() or not self.every_term()
                or s.intake.reserve_left() or not isinstance(args, dict)):
            return ""
        ids = [v for v in args.values() if isinstance(v, str) and v in s.archive.known]
        if len(ids) != 1: return ""
        lid = ids[0]
        if lid in s.archive.read or lid in s.basket.units or lid in s.named_leads: return ""
        if s.refs or (s.ledger.reading or {}).get("pointer_needs"): return ""
        slot = s.walker.product_slot(lid) if s.multi() else None
        if s.multi() and slot is None: return ""
        mark = (slot, s.ledger.version)
        if mark in self._told: return ""
        valued = self.value_hit_ids(slot)
        if not valued or lid in valued: return ""
        pk, own = self._pick(slot), s.archive.price(lid)
        lims = [p for p in ((pk[1] if pk else None), own) if kit.is_num(p)]
        cheaper = [i for _, i in self.low_unopened(sorted(valued), slot, min(lims) if lims else None)][:6]
        if not cheaper: return ""
        self._told.add(mark)
        return ("%s: this details call did not run. First open these cheaper unread entries (cheaper than this one%s), "
                "cheapest first: %s; then send the call again if it is still wanted. Why: the lowest price is wanted "
                "and this store's text queries need every word, so the value-led results hold those words in their "
                "records, and %s is not among them." % (Tune.Marks.NOT_RUN, " and than the pick" if pk else "",
                                                       self._priced(cheaper), lid))

    # ------------------------------------------------------------------ 3.15 walk hold
    def tour_pause(self, args=None):
        try:
            return self._walk_hold(args)
        except Exception as e:
            self._fail("walk_hold_error", e)
            return ""

    def _walk_hold(self, args):
        s, sk = self.s, self.s.seeker
        if not self._cheapest() or self._queued is not None or s.link.time_left() < 130: return ""
        left = s.steps_left()
        if s.refusals and left <= 15: return ""
        if args is None: ids = list(s.basket.units)
        else: ids = [l for l, n in (s.basket.lines_of(args) or []) if n is None or (kit.is_num(n) and n > 0)]
        slots = []
        for lid in ids:
            sl = s.walker.product_slot(lid)
            if sl not in slots: slots.append(sl)
        slots.sort(key=lambda x: -1 if x is None else x)
        floor = 4 if s.orders_tried else 8
        half = (left - 1 - floor) // 2
        room, ceil, twins = half - self._open_cost(), [], {}
        for sl in slots:
            if room >= 2 and self._ceiling_ready(sl):
                ceil.append(sl)
                room -= 4
                self._hold_aim(sl, ids)
        for sl in slots:
            if room > 0:
                t = sk.pair_search(sl)
                if t:
                    twins[sl] = t
                    room -= 1
        plan = s.walker.tour_plan(min(room, Tune.Search.WALK_TOTAL), only=set(slots)) or []
        if not plan and not ceil and not twins: return ""
        stop = left - 1 - half
        self._queued = (plan, stop if stop > floor else floor, ceil, twins)
        jobs = []
        if twins:
            jobs.append("search the store's other spelling of a required value (%s)" % ", ".join(
                "'%s'" % " ".join(str(t) for _, t in (sk.open_text(a) or [])) for a in twins.values()))
        if plan:
            jobs.append("read the unread pages of %s" % ", ".join(
                "'%s'" % ((sk.book().get(p[1]) or {}).get("text") or "") for p in plan))
        if ceil: jobs.append("list the category through a price cap below the pick's price")
        return ("%s: this call has not run. The code now does the work itself: %s; after that it opens the cheapest "
                "unread entries priced under the pick. Then send this call again, or a call for a cheaper match. Why: "
                "the lowest price is wanted and this store returns results %s, never by price, so pages not yet read "
                "could hold a cheaper match." % (Tune.Marks.NOT_RUN, "; ".join(jobs),
                                                "by match rank" if s.walker.ranked() else "by id"))

    # ------------------------------------------------------------------ 3.14 the walk step (code move)
    def tour_turn(self):
        try:
            return self._walk_step()
        except Exception as e:
            self._fail("walk_step_error", e)
            return False

    def _walk_step(self):
        s = self.s
        n = len(s.link.log)
        if s.walker.do_refusal_reads(): return True
        if len(s.link.log) > n:
            return False  # a decline-reads job existed (it logs) but found nothing: that is the step's result
        if self._queued is not None:
            plan, floor, ceil, twins = self._queued
            self._queued = None
            if plan or ceil or twins: return bool(s.walker.do_tour(plan, floor, ceilings=ceil, twins=twins))
        for branch in (self._rearm, self._best_walk, self._rung_walk, self._cheap_walk):
            r = branch()
            if r is not None: return bool(r)
        return False

    def _rearm(self):
        s = self.s
        if not self._cheapest() or self._short() or not self._tool(): return None
        for slot, (aim, aid) in list(self._aims.items()):
            if not aid or (aid not in s.dead and aid not in s.refused_lines): continue
            if not self.cap_profile() or not self._pg(): continue
            pk = self._pick(slot)
            if not pk or pk[1] <= aim + 1e-9 or s.steps_left() < 7 + 3 + self._open_cost(): continue
            del self._aims[slot]
            self._held.pop(slot, None)
            s.link.note("ceiling_rearm", slot=slot, dead=aid, aim=pk[1])
            return s.walker.do_tour([], 7, ceilings=[slot], repair=True)
        return None

    def _best_walk(self):
        s, w = self.s, self.s.walker
        if s.ledger.priority() != "best" or s.multi(): return None
        if s.orders_tried or self._short(): return False
        due = s.intake.due_margin()
        spare, reserve = s.steps_left() - 4, due + 2
        plan = w.tour_plan(Tune.Search.WALK_TOTAL, rule="best") or []
        need = sum(p[2] for p in plan)
        if not plan or (spare - need > reserve and self._asks()): return False
        plan = w.tour_plan(min(spare - (reserve - 1), Tune.Search.WALK_TOTAL), rule="best") or []
        if not plan or not self._pg(): return False
        before = s.link.steps
        reports = w.page_stage(plan, due) or []
        if s.link.steps == before: return False
        shown = set()
        for p in plan:
            shown |= set(((s.seeker.book() or {}).get(p[1]) or {}).get("row_shelf") or {})
        measure = kit.low(s.ledger.pick_measure() or "")
        marks = {x for x in _p12_AL.findall(measure) if len(x) >= 2 and not kit.is_link(x)}
        fresh = [(l, v) for l, v in (s.seeker.buyable_rows() or {}).items()
                 if l in shown and l not in s.archive.read and l not in s.basket.units]

        def order(x):
            t = set(_p12_AL.findall(kit.low(x[1][1] if len(x[1]) > 1 else "")))
            p = x[1][0] if kit.is_num(x[1][0]) else float("inf")
            return (not (marks & t), p, x[0])
        fresh.sort(key=order)
        rows = ["%s '%s' %s" % (l, str(v[1] if len(v) > 1 else "")[:70], self._money(v[0])) for l, v in fresh[:12]]
        text = "[best walk] " + "; ".join(reports) + "."
        if rows:
            text += (" Never-opened, in-stock, in-budget rows of the asked category, ranked on %s: %s. Before deciding, "
                     "read the property lists (details tool) of those whose titles suggest they could beat the current "
                     "pick on that measure." % (measure or "the requested measure", "; ".join(rows)))
        else: text += " These searches hold no unread row of the asked category any more."
        self._attach(text, True)
        s.link.note("best_walk", plan=[[p[0], p[2]] for p in plan], rows=len(rows))
        return True

    def _figure_rungs(self, slot):
        for _, _, _, val in self._own(slot):
            figs = _p12_FIG.findall(val)
            if len(figs) != 1 or len(_p12_NUMS.findall(val)) != 1 or re.search(r"\bbetween\b", val): continue
            t, m = val, _p12_NOMORE.search(val)
            if m:
                down = m.group(1) == "more"
                t = t[:m.start()] + " " + t[m.end():]
            else: down = bool(_p12_DOWN.search(t))
            if not (down or m or _p12_UPW.search(t)): continue
            f = float(figs[0])
            if not 3 <= f < 100: return []
            whole = int(f)
            exact = f.is_integer() and not _p12_STRICT.search(t)
            if down:
                top = whole if exact else (whole - 1 if f.is_integer() else whole)
                return [str(top - i) for i in range(Tune.Search.RUNGS) if top - i > 0]
            first = whole if exact else whole + 1
            return [str(first + i) for i in range(Tune.Search.RUNGS)]
        return []

    def _rung_walk(self):
        s, sk = self.s, self.s.seeker
        if not self._cheapest() or s.multi() or self._rungs is not None: return None
        book = sk.book() or {}
        if (s.orders_tried or self._short() or not self._pg()
                or any(e.get("id_order") is False for e in book.values())):
            return None
        if self.value_words(None) and s.walker.tour_choices(None, deep=True): return None
        rungs, home, best, ps = self._figure_rungs(None), self._home(None), None, max(1, self.page_rows())
        for k, e in book.items():
            args = e.get("args") or {}
            ft = [(f, str(t)) for f, t in (sk.open_text(args) or []) if str(t).strip()]
            rs = e.get("row_shelf") or {}
            if (len(ft) != 1 or re.search(r"\d", ft[0][1]) or e.get("id_order") is not True
                    or self.is_cap_listing(args) or not rs or len(rs) < ps
                    or any(kit.low(x).strip() not in home for x in rs.values())):
                continue
            if best is None or (e.get("total") or 0) > (best[1].get("total") or 0): best = (k, e, ft[0])
        if not rungs or not best or sk.following_page(best[0]) is None: return None
        room = s.steps_left() - self._open_cost() - (s.intake.due_margin() + 2)
        if room < 2 or (self._asks() and not s.basket.units and room > 12): return None
        fld, text = best[2][0], best[2][1].strip()
        searches = []
        for r in rungs:
            q = "%s %s" % (text, r)
            if not sk.already_ran(q):
                a = dict(best[1].get("args") or {})
                a[fld] = q
                searches.append(a)
        self._rungs = [a[fld] for a in searches]
        s.link.note("rungs", queries=list(self._rungs))
        if not searches: return None
        return s.walker.do_tour([], s.intake.due_margin(), rungs={None: searches})

    def _cheap_walk(self):
        s, w = self.s, self.s.walker
        if s.orders_tried or not self._cheapest() or self._short(): return False
        only = None
        if self.unfound():
            only = {p[0] for p in (w.tour_plan(Tune.Search.WALK_TOTAL) or []) if w.bounds_only(p[0])}
            if not only: return False
        slots, due, oc = self._slots(), s.intake.due_margin(), self._open_cost()
        reserve, left = due + 2, s.steps_left()
        if only is None and left - oc - 4 > reserve:
            ready = [sl for sl in slots if (self._pick(sl) or self._bound_need(sl)) and self._ceiling_ready(sl)
                     and self._aim(sl) is not None]
            if ready:
                sl = min(ready, key=lambda x: (1 if self._pick(x) else 0, self._vpages.get(x, 0), slots.index(x)))
                return w.do_tour([], reserve, ceilings=[sl])
            if self._seed(): return True
        plan = w.tour_plan(Tune.Search.WALK_TOTAL, only=only) or []
        if not plan: return False
        need = sum(p[2] + (1 if self._pick(p[0]) else 0) for p in plan)
        early = 2 if any(w.bounds_only(p[0]) for p in plan) else 0
        if left - oc - need > reserve + early: return False
        plan = w.tour_plan(min(left - oc - (reserve - 1), Tune.Search.WALK_TOTAL), only=only) or []
        if not plan: return False
        s.early_walk = True
        return w.do_tour(plan, due)

    # ------------------------------------------------------------------ 3.16, 3.19, 3.20
    def unfound(self):
        s = self.s
        try:
            if not s.multi(): return False
            reads = list(s.archive.read)
            for sl in self._slots():
                home = self._home(sl)
                if not home or not any(kit.low(s.judge.label_of(l)).strip() in home for l in reads): return True
            tail = "|%s" % s.ledger.version
            if any(str(m).endswith(tail) for m in s.passed): return bool(s.intake.no_verified_fit())
            return False
        except Exception as e:
            self._fail("unfound_error", e)
            return False

    def heading_line(self, slot, record, template):
        """Family query text from the pick's title (INTERFACES: returns the text; the caller fills the template)."""
        s = self.s
        try:
            avoid = set()
            for lab in self._home(slot):
                avoid |= set(_p12_AL.findall(lab))
            for ln in s.ledger.in_force() or []:
                avoid |= set(_p12_AN.findall(kit.low(ln)))
            names = [p.get("name") or "" for p in (s.ledger.products() or [])] or [s.ledger.product_name() or ""]
            for nm in names:
                avoid |= set(_p12_AL.findall(kit.low(nm)))
            stems = {kit.stem(x) for x in avoid}

            def bad(x):
                return kit.is_link(x) or x in avoid or kit.stem(x) in stems
            ws = _p12_AN.findall(kit.low(kit.row_title(record) if isinstance(record, dict) else ""))
            for i, x in enumerate(ws):
                if re.fullmatch(r"[a-z]{3,}", x) and not bad(x):
                    t = x + (" " + ws[i + 1] if i + 1 < len(ws) and not bad(ws[i + 1]) else "")
                    return "" if s.seeker.already_ran(t) else t
            return ""
        except Exception as e:
            self._fail("title_line_error", e)
            return ""

    def item_steps(self):
        s = self.s
        try:
            if not s.multi() or not self._cheapest(): return ""
            prods, parts, nxt = s.ledger.products() or [], [], ""
            for sl in self._slots():
                name = (prods[sl].get("name") if isinstance(sl, int) and sl < len(prods) else "") or "item %s" % sl
                pk = self._pick(sl)
                n = self._vpages.get(sl, 0)
                parts.append("%s: %d value page%s read, pick %s" % (name, n, "" if n == 1 else "s", "yes" if pk else "no"))
                if not pk and not nxt: nxt = name
            return "[product steps] " + "; ".join(parts) + ("; next: %s" % nxt if nxt else "") + "."
        except Exception as e:
            self._fail("product_steps_error", e)
            return ""



# ------------------------------------------------------------ p13_warden.py
# Warden (spec 11, sections 1-14 and 17): screens, finish holds, product fits, reach moves, basket telling.


class Warden:
    _IDENT = re.compile(r"\b[A-Z][A-Za-z]{1,12}[ -]?\d{3,6}\b")
    _CODE = re.compile(r"(?<![A-Za-z0-9_-])[A-Z]{1,4}\d{1,3}(?![A-Za-z0-9_-])")
    _TAG = re.compile(r"(?<![A-Za-z0-9_-])([A-Z]{1,4}\d{1,3})-\d{3,6}(?![A-Za-z0-9_-])")
    _FIG = re.compile(r"(\d+(?:\.\d+)?)\s?([a-z'\"]{1,4})\b")
    _LOOSE = re.compile(r"\d\s*(?:-|to)\s*\d|[+<>]|\b(?:about|around|approx\w*|roughly)\b", re.I)
    _SPLIT = re.compile(r"\s*(?:[,;/]|\bor\b|\beither\b|\bnor\b)\s*")
    _TOKEN_ARG = re.compile(r"token|cursor|next|continu", re.I)
    _CAP_ARG = re.compile(r"ceil|max|price|budget|upper|most|cap$", re.I)
    _CAP_NOT = re.compile(r"page|first|offset|skip|count|size|num", re.I)

    def __init__(self, s):
        self.s = s
        self._ix, self._pair, self._stale, self._drop_n = None, (None, {}), None, None
        self._ptr, self._cheap, self._sold, self._dep, self._misfit, self._fit_tried = {}, {}, {}, {}, {}, {}
        self._views, self._fit_pairs, self._checked, self._warned = set(), set(), set(), set()
        self._fitq, self._rev_q = [], []
        self._told_n = self._early = self._whole = self._reviews = self._rev_sent = self._solo_rows = 0
        self._told_decl = self._kept = self._squeezed = self._lacked = self._leaded = self._best = False
        self._unknown = self._deep_done = self._cap_done = self._few_done = self._swept = False
        self._rev_deep = self._rev_job = self._cap_job = self._few = None

    # small helpers
    def _room(self, n, pad=None):
        return self.s.intake.room(n, pad)

    @staticmethod
    def _nr(text):
        return Tune.Marks.NOT_RUN + ": " + text

    @staticmethod
    def _pre(text):
        if not text: return ""
        return text if text.startswith("Not run") else Tune.Marks.NOT_RUN_YET + text

    @staticmethod
    def _num(x):
        return ("%.6f" % x).rstrip("0").rstrip(".") if kit.is_num(x) else ""

    def _cheapest(self):
        return self.s.ledger.priority() == "cheapest"

    def _in_budget(self, total=None):
        b = self.s.ledger.budget
        t = self.s.basket.total() if total is None else total
        return not kit.is_num(b) or t <= b + 1e-9

    def _reads(self, ids):
        bs, n = self.s.mender.batch_spec(), len(ids)
        return -(-n // max(1, bs[2])) if bs and n >= 2 else n

    def _pname(self, slot):
        pr = self.s.ledger.products()
        if slot is not None and 0 <= slot < len(pr): return str(pr[slot].get("name") or "product %d" % (slot + 1))
        return str(self.s.ledger.product_name() or "item")

    def _qty(self, slot):
        pr = self.s.ledger.products()
        q = pr[slot].get("qty") if slot is not None and 0 <= slot < len(pr) else self.s.ledger.quantity()
        return q if kit.is_num(q) and q > 0 else 1

    def _open_ref(self):
        return any(r.get("status") == "open" for r in self.s.refs)

    def _tool(self, role, generic):
        t = self.s.role(role)
        return ("the " + t + " tool") if t else generic

    @staticmethod
    def _dicts(o, top=12, d=0):
        if d > top: return
        if isinstance(o, dict):
            yield o
            vals = o.values()
        elif isinstance(o, list): vals = o
        else: return
        for v in vals:
            if isinstance(v, (dict, list)): yield from Warden._dicts(v, top, d + 1)

    @staticmethod
    def _strings(o, d=0):
        if isinstance(o, str): yield o
        elif d < 12 and isinstance(o, (dict, list)):
            for v in (o.values() if isinstance(o, dict) else o): yield from Warden._strings(v, d + 1)

    def _index(self):
        """One pass per archive size: listing pages (9.1), record recency (5.5), titles (5.6), shelf names (5.11a),
        tag-head stems of all archive text (9.5)."""
        s = self.s
        n = len(s.archive.obs)
        if self._ix and self._ix[0] == n: return self._ix[1]
        known = s.archive.known
        pages, last, titles, shelves, stems = {}, {}, {}, {}, set()
        for i, ob in enumerate(s.archive.obs):
            for d in self._dicts(ob):
                held = [v.strip() for k, v in d.items() if isinstance(v, str) and "id" in str(k).lower() and v.strip()]
                if held:
                    bits = []
                    for v in d.values():
                        if isinstance(v, str): bits.append(v.strip())
                        elif isinstance(v, dict): bits.extend(x.strip() for x in v.values() if isinstance(x, str))
                    for lid in held:
                        pg = pages.setdefault(lid, {})
                        for b in bits:
                            if b: pg[b] = None
                        last[lid] = i
                    lab = kit.shelf_field(d) if any(h in known for h in held) else ""
                    if lab and lab not in shelves:
                        shelves[lab] = next((v.strip() for v in d.values() if isinstance(v, str) and v.strip().lower() == lab), lab)
                rid = kit.row_id(d)
                t = kit.row_title(d).strip().lower() if rid and rid in known else ""
                if t: titles.setdefault(t, set()).add(rid)
            for st in self._strings(ob):
                if "-" in st: stems.update(self._stem(m.group(1)) for m in self._TAG.finditer(st))
        ix = {"pages": {k: " ".join(v) for k, v in pages.items()}, "last": last, "titles": titles, "shelves": shelves,
              "stems": stems}
        self._ix = (n, ix)
        return ix

    # fit shapes (9.1-9.5, 17.2)
    def _page(self, lid):
        return self._index()["pages"].get(lid, "") if lid else ""

    @staticmethod
    def _flat(text):
        return re.sub(r"[^a-z0-9]", "", str(text).lower())

    @staticmethod
    def _stem(code):
        return re.sub(r"\d+$", "", code)

    def _idents(self, page):
        out = []
        for m in self._IDENT.findall(page or ""):
            if m not in out: out.append(m)
        return out[:8]

    def _codes(self, page):
        return set(self._CODE.findall(page or ""))

    def _tags(self, text):
        return {m.group(1) for m in self._TAG.finditer(text or "")}

    def _series(self, page):
        st = self._index()["stems"]
        return sorted(c for c in self._codes(page) if self._stem(c) in st)

    def _verdict(self, ap, dp):
        ac, at, dc, dt = self._codes(ap), self._tags(ap), self._codes(dp), self._tags(dp)
        if (ac & dt) or (dc & at): return "match"
        if {self._stem(t) for t in dt} & {self._stem(c) for c in ac}: return "clash"
        return "unknown"

    def _named(self, names, page):
        fp = self._flat(page)
        return any(self._flat(n) in fp for n in names)

    def _slot_lines(self, first=()):
        m = {}
        for l in list(first) + list(self.s.basket.units):
            sl = self.s.walker.product_slot(l)
            if sl is not None and sl not in m: m[sl] = l
        return m

    def _pairs(self, m=None):
        """(dep, anc, dep line, anchor line, anchor page, dep page) for every link with both lines present."""
        m = self._slot_lines() if m is None else m
        return [(d, a, m[d], m[a], self._page(m[a]), self._page(m[d])) for d, _, a in self.links() if d in m and a in m]

    # 4. shopper words and store notices
    def buyer_lines(self, name, obs, refused):
        s, out = self.s, []
        try:
            if name and name == s.role("message"):
                if isinstance(obs, str): out = [obs]
                elif isinstance(obs, dict) and "error" not in obs:
                    def grab(d):
                        for v in [v for v in d.values() if isinstance(v, (str, list))][:10]: out.extend(kit.lengthy_texts(v))
                    kit.visit(obs, grab, top=4)
            elif isinstance(obs, dict):
                def take(d):
                    if d is not obs and len(d) > 6: return
                    small = len(d) <= 3
                    rec = any("id" in str(k).lower() and isinstance(v, str) and kit.looks_like_id(v) for k, v in d.items())
                    for v in d.values():
                        if isinstance(v, list):
                            if v and all(isinstance(x, str) for x in v): out.extend(x for x in v[:10] if len(x.split()) >= 4)
                        elif isinstance(v, str) and len(v.split()) >= 4 and (refused or (small and len(v) <= 600 and not rec)):
                            out.append(v)
                kit.visit(obs, take, top=4)
        except Exception as e:
            s.link.fault("shopper_lines_error", e)
        res = []
        for x in out:
            if isinstance(x, str) and x not in res: res.append(x)
        return res

    # 5. the proposal screen (unguarded as built, 12 3.8: an error goes to the crash handler)
    def screen(self, call, menu, scripted, size, queued_msgs):
        s, T, NR = self.s, None, self._nr
        fn = call.get("function") or {}
        name = str(fn.get("name") or "")
        mt = ", ".join(sorted(menu or ()))
        if s.spec(name) is None: return (NR("there is no tool called '%s'. Choose one of: %s." % (name, mt)), name, {})
        if not scripted and name not in (menu or ()): return (NR("too few steps remain for '%s'; the only tools usable now are: %s." % (name, mt)), name, {})
        args = kit.args_from(fn.get("arguments"))
        if not isinstance(args, dict): return (NR('the arguments could not be read; send them as one well-formed JSON object.'), name, {})
        fixed, faults = ArgFixer.fix(args, s.schema(name))
        if isinstance(fixed, dict): args = fixed
        if faults: return (NR("; ".join(str(f) for f in faults[:6])), name, args)
        if not scripted:
            args = self._cut_list(name, args)
            h, args = self._handles(name, args)
            if h: return (h, name, args)
        fin = bool(s.intake.finish_kind(name, args)) or name in s.intake.finishing_tools()
        if fin and size > 1: return (NR('an order or a decline must be the only call of its reply; send it by itself once the other results are back.'), name, args)
        msg, cart, cv = s.role("message"), s.role("cart"), s.role("cart_view")
        if name == msg:
            spent = queued_msgs + s.msgs_sent
            if s.msg_cap is not None and spent >= s.msg_cap: return (NR('no messages remain, so the shopper receives nothing more.'), name, args)
            if not scripted and not s.basket.units and spent >= s.intake.question_room(): return (NR('the message allowance is on hold: the last send is kept for the basket report that falls due once an item is in the cart, and for any closing step the instructions demand. Put a pick in the cart first; messaging opens again after that.'), name, args)
        if (not scripted and cv and name == cv and cv != cart and not s.batch_has_cart and s.basket.units
                and s.cart_shown_version == s.cart_version and s.since_cart == 0 and s.cart_version not in self._views):
            self._views.add(s.cart_version)
            st = s.judge.supply_tool()
            return (NR("a cart view adds nothing now, because the latest cart reply already lists every line. %s To learn one line's present stock, fetch that listing's record with %s." % (s.basket.summary(), ("the " + st + " tool") if st else "a tool that shows stock")), name, args)
        sig = name + kit.render(args)
        if not (fin or name == msg or name == cart):
            if sig in (s.lookup_window or ()): return (NR('the same call was made a moment ago and nothing has changed since; its result is above. Change the arguments or take another step.'), name, args)
            h = self._lookups(name, args)
            if h: return (h, name, args)
        if not scripted:
            h = s.intake.late_gate(name, args) or s.pricer.value_gate(name, args)
            if h: return (h, name, args)
        h = self.cart_screen(args, sig) if cart and name == cart else ""
        return (h or None, name, args)

    def _cut_list(self, name, args):
        s = self.s
        bs = s.mender.batch_spec()
        if not bs or name != bs[0]: return args
        arg, cap = bs[1], bs[2]
        ids = args.get(arg)
        if not isinstance(ids, list) or len(ids) <= cap: return args
        last = self._index()["last"]
        order = sorted(range(len(ids)), key=lambda i: (-last.get(str(ids[i]).strip(), -1), i))
        out = dict(args)
        out[arg] = [ids[i] for i in sorted(order[:cap])]
        s.link.note("list_cut", asked=len(ids), cap=cap)
        return out

    def _handles(self, name, args):
        s = self.s
        pg = s.seeker.paging() if name == s.role("search") else None
        ta = pg.get("name") if pg and pg.get("mode") == "token" else None
        val = args.get(ta) if ta else None
        if isinstance(val, str) and val.strip():
            book, seen = s.seeker.book(), set()
            for e in book.values():
                seen.update(str(x) for x in (e.get("spots") or ()))
                seen.update(str(x) for x in (e.get("tokens") or {}).values() if x)
            if val not in seen and val.strip() not in seen:
                rest = {k: v for k, v in args.items() if k != ta}
                key = s.seeker.book_key(rest)
                if key not in book:
                    s.link.note("token_dropped", value=val[:40])
                    args = rest
                else:
                    nxt = s.seeker.following_page(key)
                    tail = ("its next page comes with %s=%s" % (ta, nxt)) if nxt not in (None, "") else "it has no unread page left"
                    return (self._nr("'%s' is not a page token this store handed out; tokens are copied from a result's next-page field. This search already ran, and %s." % (val[:40], tail)), args)
        bs = s.mender.batch_spec()
        if name not in {s.role("cart"), s.role("details"), s.role("view"), bs[0] if bs else None} - {None}: return ("", args)
        strays = []
        for x in self._strings(args):
            if x.strip() not in s.archive.known and len(x.strip()) > 8 and x not in strays: strays.append(x)
        if not strays: return ("", args)
        tmap, swaps, clashes = self._index()["titles"], {}, []
        for x in strays:
            t = x.strip().lower()
            ids = set(tmap.get(t, ()))
            if not ids and len(t) >= 20:
                for k, v in tmap.items():
                    if k.startswith(t): ids |= v
            if len(ids) == 1: swaps[x] = next(iter(ids))
            elif len(ids) > 1: clashes.append((x, sorted(ids)))
        if clashes:
            return (self._nr('this tool takes listing ids, not titles. %s. Resend the call with the id you mean.' % "; ".join("'%s' is the title of %s" % (x.strip()[:60], ", ".join(i))
                                                          for x, i in clashes)), args)
        if swaps:
            def sub(o):
                if isinstance(o, str): return swaps.get(o, o)
                if isinstance(o, dict): return {k: sub(v) for k, v in o.items()}
                return [sub(v) for v in o] if isinstance(o, list) else o
            args = sub(args)
            s.link.note("title_to_id", ids=sorted(set(swaps.values())))
        return ("", args)

    def _lookups(self, name, args):
        s = self.s
        tops = [(k, v.strip()) for k, v in args.items() if isinstance(v, str) and v.strip()]
        if name == s.role("search"):
            for k, v in tops:
                if v in s.archive.known:
                    return self._nr("the search compares its text with the words of listings, so the listing id '%s' finds nothing; open that listing with %s instead." % (v, self._tool("details", "the tool that shows a listing's details")))
        names = self.shelf_titles()
        flat = {x.rstrip("s") for x in names}
        for k, v in (tops if names else ()):
            if kit.shelf_field({k: v}) and v.lower().rstrip("s") not in flat:
                return self._nr("'%s' expects a category name written exactly as listings write it, and free wording there finds nothing. Categories met so far: %s. Descriptive words belong in the search text." % (k, ", ".join(sorted(names.values())[:8])))
        return ""

    def shelf_titles(self):
        try:
            return dict(self._index()["shelves"])
        except Exception as e:
            self.s.link.fault("shelf_names_error", e)
            return {}

    def category_note(self, lid):
        # Judge writes a fixed phrase in the "category itself names the value" line; no shared flag, so read the wording
        try:
            got = []
            for l in (self.s.judge.evidence_view(lid) or {}).get("lines") or []:
                low = l.lower() if isinstance(l, str) else ""
                if "categor" in low and ("itself" in low or "alone" in low) and "field in this store" not in low:
                    got.append(re.sub(r"^\s*-\s*", "", l).strip())
            return (" Met by its category alone: %s." % "; ".join(got[:3])) if got else ""
        except Exception as e:
            self.s.link.fault("category_note_error", e)
            return ""

    # 6. the cart screen
    def cart_screen(self, args, sig):
        s, T, NR, b = self.s, None, self._nr, self.s.basket
        tgt, put, lines, cheap = b.target(args), b.is_cart_call(args), list(b.units), self._cheapest()
        if (s.intake.question_limit() - s.msgs_sent <= 0 and self.told_current() and not any(l in s.dead for l in lines)
                and all(s.passed_now(l) for l in lines) and self._in_budget() and self._told_n < 3):
            self._told_n += 1
            s.link.note("stated_hold", lines=lines)
            return NR('the shopper has already been given exactly this cart and no message remains to report a different one, so altering it now would put through an order the shopper was never shown. Nothing has happened since that requires a change (no notice, no refusal, no sold-out line, no total over the budget), so order the cart unchanged.')
        note = s.tracer.pointer_note(tgt) if put and tgt else ""
        if note and self._ptr.get(tgt, 0) < 2:
            self._ptr[tgt] = self._ptr.get(tgt, 0) + 1
            return NR(note + " Pick a different listing.")
        if tgt and not put and s.passed_now(tgt):
            p = s.archive.price(tgt)
            lower = cheap and (any(x != tgt and kit.is_num(s.archive.price(x)) and kit.is_num(p) and s.archive.price(x) < p
                                   for x in (s.batch_puts or ())) or bool(s.judge.lower_price_fit(tgt)))
            ap = s.pass_price.get(tgt)
            same = (ap is None and p is None) or (kit.is_num(ap) and kit.is_num(p) and abs(ap - p) <= 1e-9)
            rp = s.refusal_at_put.get(tgt)
            stamp = sig + "#" + str(s.put_tally.get(tgt, 0))
            if not lower and same and (rp is None or rp == s.refusals) and stamp not in self._warned:
                self._warned.add(stamp)
                return NR("the removal is held back. The independent check found %s meeting every condition now in force (properties read under the value rules: a store's dedicated value field counts only by what it says, and the kind of item is judged from its category label alone, whatever the title or text say)%s, and nothing has changed since. Send this same call again only when a change or a refusal rules the line out, a field plainly violates a condition, or a cheaper listing takes its place." % (tgt, "; under the cheapest priority, no verified fit costs less" if cheap else ""))
        if not (put and tgt): return ""
        h = self._zero_hold(args, sig)
        if h: return h
        held = b.held(tgt)
        if held and held > 0 and b.adds is not False and sig not in self._warned:
            self._warned.add(sig)
            if b.adds is None:
                # not yet learned whether a put on a held line adds units or sets the count: say both
                return NR("the cart already holds %s x%s. This cart tool may add the call's units to that line or set the line to them; the cart reply will show which. If the line already has its wanted units, an order needs no cart change; otherwise send the same call again." % (tgt, self._num(held)))
            return NR('the cart already holds %s x%s. With this tool a put on a held line raises its count, and an order needs no cart change; to adjust the units, add or remove only the difference. Send the same call again if the extra units really are intended.' % (tgt, self._num(held)))
        h = "" if note else s.pricer.tour_pause(args)
        if h: return h
        cl = b.lines_of(args)
        live = [l for l, a in cl if a is None or (kit.is_num(a) and a > 0)]
        for l in [l for l, a in cl[1:] if a is None or (kit.is_num(a) and a > 0)] + [tgt]:
            h = self._add_check(l)
            if h: return h
        h = self._clash_hold(live)
        if h: return h
        for l in (live if self._room(3) else ()):
            rn = s.judge.order_audit(l)
            if rn:
                rv = s.judge.order_confirm(l)
                return NR(rn) + ((' A lower-priced verified alternative exists: %s passes the current conditions and is priced below %s; %s' % (rv, l, "cart %s in its place." % rv) + self.category_note(rv)) if rv else "")
        for l in (live if cheap else ()):
            rv = s.judge.lower_price_fit(l)
            if rv and rv not in b.units and ("cart", l, rv) not in self._cheap:
                self._cheap[("cart", l, rv)] = 1
                return NR('a lower-priced verified alternative exists: %s passes the conditions now in force and is priced below %s. The lowest price is the priority and this check overrides your own reading, so cart %s instead; send the same call again only when some record proves %s fails a current condition.' % (rv, l, rv, rv) + self.category_note(rv))
        if s.multi():
            want = s.ledger.counts()
            bad = []
            for l, a in cl:
                w = self._want_of(l)
                if kit.is_num(a) and a > 0 and ((w is not None and abs(a - w) > 1e-9) or (w is None and int(a) not in want)):
                    bad.append((l, a, w))
            if not bad or sig in self._warned: return ""
            self._warned.add(sig)
            return NR("each product needs its own wanted number of units (%s), but the call has %s.%s Resend the identical call if this is intended." % (", ".join(str(x) for x in want), ", ".join("%s x%s" % (l, self._num(a)) for l, a, _ in bad), self._units_fix([(l, w) for l, _, w in bad], args)))
        asked = b.count_of(args)
        if asked is None: return ""
        goal = s.ledger.quantity() if b.adds is False else s.ledger.quantity() - (b.held(tgt) or 0)
        if abs(asked - goal) <= 1e-9 or sig in self._warned: return ""
        self._warned.add(sig)
        return NR('%s unit(s) of one listing are needed and the call asks for %s.%s Resend the identical call if this is intended.' % (self._num(goal), self._num(asked), self._units_fix([(tgt, goal)] if b.adds is False or not b.held(tgt) else [], args)))

    # cart units: the count argument names the listing's units, not a position in the cart
    def _want_of(self, lid):
        s = self.s
        if not s.multi(): return s.ledger.quantity()
        k, c = s.walker.product_slot(lid), s.ledger.counts()
        return c[k] if isinstance(k, int) and 0 <= k < len(c) else None

    def _units_fix(self, pairs, args):
        k = self.s.basket.count_key()
        fix = ["%s=%s for %s" % (k, self._num(w), l) for l, w in pairs if w is not None]
        if not k or not fix: return ""
        return " '%s' is the number of units for the listing the call names, not a cart position: send %s." % (k, ", ".join(fix))

    def _zero_hold(self, args, sig):
        b = self.s.basket
        k, cl = b.count_key(), b.lines_of(args)
        if not k or len(cl) != 1 or not kit.is_num(cl[0][1]) or cl[0][1] != 0 or b.held(cl[0][0]) or sig in self._warned:
            return ""
        self._warned.add(sig)
        lid, w = cl[0][0], self._want_of(cl[0][0])
        self.s.link.note("zero_units_hold", listing=lid, key=k, want=w)
        go = ("To put it in the cart, send %s=%s with it." % (k, self._num(w))) if w else ("To put it in the cart, send its wanted units in '%s'." % k)
        return self._nr("the cart holds no line for %s, and '%s' is the number of units the call gives that listing (not a position in the cart), so %s=0 changes nothing. %s" % (lid, k, k, go))

    def units_hint(self, args):
        # after a cart call, once the tool is known to SET a line's units: lines whose units differ from their want
        try:
            b = self.s.basket
            k = b.count_key()
            if not k or b.adds is not False or b.lines_of(args) and len(b.lines_of(args)) != 1: return ""
            off = [(l, self._want_of(l)) for l, n in b.units.items()]
            off = [(l, w) for l, w in off if w is not None and abs(b.held(l) - w) > 1e-9]
            if not off: return ""
            return "[cart units] '%s' sets the units of the listing a call names (0 removes its line): send %s." % (
                k, ", ".join("%s=%s for %s" % (k, self._num(w), l) for l, w in off))
        except Exception as e:
            self.s.link.fault("units_hint_error", e)
            return ""

    def _add_check(self, lid):
        s = self.s
        k = (lid, s.ledger.version)
        if not lid or not self._room(3) or k in self._checked: return ""
        self._checked.add(k)
        r = s.judge.item_check(lid, Tune.Aim.CART)
        return self._nr(r + " Send the same call again only if the fit is certain.") if r else ""

    def _clash_hold(self, live):
        s = self.s
        if not self.links() or not self._room(5, 0): return ""
        found = []
        for dep, anc, dl, al, ap, dp in self._pairs(self._slot_lines(live)):
            if (al, dl) in self._fit_pairs or len(self._fit_pairs) >= 5 or self._verdict(ap, dp) != "clash": continue
            self._fit_pairs.add((al, dl))
            dst = {self._stem(t) for t in self._tags(dp)}
            codes = sorted(c for c in self._codes(ap) if self._stem(c) in dst)
            s.link.note("code_fit_hold", anchor=al, dependent=dl, codes=codes)
            found.append((dep, anc, dl, al, codes))
        if not found: return ""
        for dep, anc, dl, al, codes in found:
            if codes and dep not in {q[0] for q in self._fitq}: self._fitq.append((dep, codes[0]))
        parts = "; ".join('%s (%s) belongs with %s (%s), whose page gives %s, yet the tag of %s is from the same series with a different number' % (dl, self._pname(dep), al, self._pname(anc), "/".join(codes) or "its code", dl)
                          for dep, anc, dl, al, codes in found)
        return self._nr("this store expresses fits as codes. %s, so %s. The code is now searching the dependent product's listings with the code as the query and will open the hits that carry it. For each, cart a hit whose tag begins with that code and which meets the remaining conditions, or move to an anchor whose stated code shows up in some tag of each; send the same call again only when no such listing exists." % (parts, "they do not fit" if len(found) > 1 else "it does not fit"))

    # 7-8. the finish screen (unguarded as built)
    def finish_screen(self, cid, name, args):
        kind = self.s.intake.finish_kind(name, args)
        if kind == "order": return (self._order_chain() or None, args)
        if kind == "decline":
            h = self._decline_chain()
            return (h, args) if h else (None, self.s.closer.clean_refusal(args, name))
        return (None, args)

    def _order_chain(self):
        s, T, P = self.s, None, self._pre
        lines = list(s.basket.units)
        h = self._sold_hold(lines)
        if h: return h
        if self._room(3) and lines:
            l, rv = "", ""
            for l in [lines[-1]] + lines[:-1]:
                rv = s.judge.lower_price_fit(l)
                if rv: break
            if rv and self._cheap.get((l, rv), 0) < 2:
                self._cheap[(l, rv)] = self._cheap.get((l, rv), 0) + 1
                return P('a lower-priced verified alternative exists: %s passes every condition and is priced below %s. The lowest price is the priority and this check overrides your own reading: take %s out, cart %s, refresh its stock and price, then order.' % (rv, l, l, rv) + self.category_note(rv))
        h = s.pricer.tour_pause() if lines else ""
        if h: return P(h)
        st = s.judge.supply_tool()
        if st and self._room(len(lines) + 2, 0):
            unread = [l for l in lines if s.line_seq.get(l, -1) < s.cart_seq_mark]
            stamp = (tuple(sorted(lines)), s.cart_seq_mark)
            if unread and stamp != self._stale:
                self._stale = stamp
                s.judge.queue_open(unread[:4], "cart", True, tool=st, say="Re-reading the stock and price of the cart lines before the order.")
                return P('stock and prices can move without any notice, and %s went unread since the cart last changed. The code now reads their stock through the %s tool; then, with every line available and the total within the budget, order, else replace the failing line.' % (", ".join(unread), st))
        h = self._sweep()
        if h: return h
        if self._room(5, 0) and self.fits_written():
            faults = self.fit_faults()
            stamp = (tuple(sorted(lines)), tuple(faults))
            n = self._dep.get(stamp, 0)
            if faults and n < 2 and (stamp in self._dep or len(self._dep) < 4):
                self._dep[stamp] = n + 1
                return P('compatibility gap: %s. Search for the dependent item using the identifiers one at a time as the query, open what comes back and cart a hit whose page names one of them; or else pick another anchor whose identifier some dependent page cites. Order once the pair matches.' % "; ".join(faults))
        h = self._vet(lines)
        if h: return h
        faults, key = self.misfits(), tuple(sorted(lines))
        if faults and s.steps_left() >= 1 and self._misfit.get(key, 0) < 2:
            self._misfit[key] = self._misfit.get(key, 0) + 1
            s.link.note("misfit_hold", lines=lines, faults=faults[:6])
            return P("the records exclude this cart (%s), and a basket like this cannot be accepted. Replace the failing line with an already opened listing that passes (for a fit, a listing whose page cites the anchor's identifiers), then order; if none of the opened listings passes, or time is running out, decline." % "; ".join(faults)[:900])
        return s.closer.closing_moves("order")

    def _sold_hold(self, lines):
        s = self.s
        gone = sorted(l for l in lines if l in s.dead)
        if not gone or not self._room(2): return ""
        stamp = (tuple(gone), s.cart_seq_mark)
        n = self._sold.get(stamp, 0)
        if n < 2:
            self._sold[stamp] = n + 1
            return self._pre(s.judge.soldout_note(gone) or '%s can no longer be bought. Drop it, add an approved replacement if one exists, re-read stock, then check out.' % ", ".join(gone))
        if n > 2 or not s.role("cart") or s.steps_left() < 6 or s.link.time_left() <= 70: return ""
        self._sold[stamp] = 3
        bar = sorted(set(gone) | set(s.dead))
        was, ok = s.link.mute, False
        try:
            ok = bool(s.closer.code_shop(bar, [l for l in lines if l not in bar]))
        except Exception as e:
            s.link.fault("soldout_swap_error", e)
        finally:
            s.link.mute = was
        s.link.note("soldout_swap", gone=gone, ok=ok, cart=sorted(s.basket.units))
        if not ok or not s.basket.units or any(l in s.dead for l in s.basket.units): return ""
        return self._pre("%s had no stock but were carted. The code swapped in a candidate, left the other lines untouched and read each line's stock. %s Compare the cart with the request and order if it matches." % (", ".join(gone), s.basket.summary()))

    def _vet(self, lines):
        s, P = self.s, self._pre
        if s.order_vetted or not self._room(2, 0): return ""
        s.order_vetted = True
        faults = s.basket.faults(s.ledger.quantity(), s.ledger.counts())
        if faults and self._room(3): return P('the cart does not match the requested quantities: %s. Correct this, and order afterwards.' % "; ".join(faults))
        multi = s.multi()
        chk = lines if multi else lines[-1:]
        for l in chk:
            r = s.judge.item_check(l, Tune.Aim.ORDER)
            if r: return P(r)
        for l in chk:
            rn = s.judge.order_audit(l)
            if rn:
                rv = s.judge.order_confirm(l)
                if rv: rn += ' A lower-priced verified alternative exists: %s passes the current conditions and is priced below %s; %s' % (rv, l, "drop %s, cart %s, refresh its stock and price, then order." % (l, rv)) + self.category_note(rv)
                return P(rn)
        return P("pre-order checklist; confirm it by repeating the order call. [cart] %s, and no other lines; [fresh read] after the latest cart change, stock and price were looked up again with a tool that reports them: available, and unit price times quantity within the current budget; [fit] the item's own properties meet every stated condition, disclosed need and announced change; [rank] when the best or the cheapest was asked for, nothing reachable beats it; [rules] every procedural step of the request is done. Verify anything unverified now, else repeat the order call." % ("a single line for each requested product at its own quantity" if multi else
                                        "just the chosen item at the asked quantity"))

    def _sweep(self):
        s = self.s
        st = s.role("search")
        if not (self._cheapest() and not s.orders_tried and not self._swept and st and s.basket.units
                and s.steps_left() >= 12 and s.link.time_left() >= 110): return ""
        self._swept = True
        try:
            sch = s.schema(st)
            free = [k for k, p in (sch.get("properties") or {}).items() if isinstance(p, dict) and p.get("type") == "string" and not p.get("enum")]
            if len(free) != 1 or any(r != free[0] for r in (sch.get("required") or [])): return ""
            qs, calls, turned, multi = [], [], 0, s.multi()
            past = {e.get("text", "") for e in s.seeker.book().values()}
            for q in ([] if multi else s.seeker.label_searches()):
                q = str(q).strip()
                if q and " " not in q and q.lower() not in past and q not in qs and len(qs) < 2: qs.append(q)
            for q in qs:
                a, bad = ArgFixer.fix({free[0]: q}, sch)
                if not bad: calls.append(s.engine.call(st, a))
            if calls: s.engine.send_booked(calls)
            asked = set(s.seeker.wanted_shelves() or ())
            for key, e in (list(s.seeker.book().items()) if s.seeker.paging() and asked else []):
                if turned >= 3 or not (set(e.get("shelves") or ()) & asked) or (e.get("total") or 0) > 4 * max(1, e.get("size") or 0):
                    continue
                while turned < 3 and s.steps_left() >= 10 and s.link.time_left() > 110:
                    sp = s.seeker.following_page(key)
                    if sp is None: break
                    turned += 1
                    s.walker.search_call(s.seeker.page_call(key, sp), "Reading the rest of a short result list before "
                                         "ordering, since a cheaper fit may appear further on.")
            ceil = {}
            for l in s.basket.units:
                k = s.judge.label_of(l) if multi else ""
                if multi and not k: return ""
                p = s.archive.price(l)
                if kit.is_num(p): ceil[k] = min(p, ceil.get(k, p))
            if not ceil: return ""
            cands = []
            for lid, row in s.seeker.buyable_rows().items():
                p, c = row[0], ceil.get(s.judge.label_of(lid) if multi else "")
                if lid not in s.basket.units and lid not in s.archive.read and kit.is_num(p) and c is not None and p < c - 1e-9:
                    cands.append((p, lid, row[1] or ""))
            cands = sorted(cands)[:4]
            s.link.note("cheap_sweep", queries=qs, pages=turned, rows=[c[1] for c in cands])
            if not cands: return ""
            s.judge.add_leads([c[1] for c in cands])
            what = ("the queries " + ", ".join("'%s'" % q for q in qs)) if qs else "further pages of short result sets"
            rows = "; ".join("%s '%s' at %s" % (lid, str(t)[:70], self._num(p)) for p, lid, t in cands)
            return self._pre('the lowest price matters, so extra code searches went out (%s). These never-opened, available rows of %s undercut the current cart line: %s. Open them all in one reply; if one satisfies every condition, put it in the cart in place of the current line, re-read stock and price and order; otherwise order the cart as it is.' % (what, "their own category" if multi else "the category", rows))
        except Exception as e:
            s.link.fault("cheap_sweep_error", e)
            return ""

    def _decline_chain(self):
        s, T, P = self.s, None, self._pre
        lines, multi, read, ref = list(s.basket.units), s.multi(), s.archive.read, s.refused_lines
        if (not self._told_decl and self.told_current() and not any(l in s.dead for l in lines)
                and all(s.passed_now(l) for l in lines) and not self.fit_contradiction() and self._in_budget()):
            self._told_decl = True
            s.link.note("decline_told", lines=lines)
            return P('this very basket (%s) was reported to the shopper, whose answer raised no objection, condition or change, and nothing has moved since. An answer that acknowledges the reported basket without objecting counts as the approval the request wants, because this shopper does not reply with a plain yes. Place the order now; decline only when a record proves a cart line fails a stated condition.' % ", ".join(lines))
        dropped = list(s.ledger.dropped or [])
        if not lines and not multi and dropped and self._room(3) and len(dropped) != self._drop_n:
            rows = sorted((r[0], l) for l, r in s.seeker.buyable_rows().items() if l in read and l not in ref and kit.is_num(r[0]))[:3]
            if rows:
                self._drop_n = len(dropped)
                s.judge.add_leads([l for _, l in rows])
                s.link.note("decline_dropped", ids=[l for _, l in rows])
                return P('conditions were dropped since the search began (%s), so opened listings may fit now. The three cheapest affordable opened rows of the category with stock left: %s. Test them against the conditions still standing, cart the cheapest passing one, report the basket when a message remains, then order; decline only when none passes.' % (", ".join(str(x) for x in dropped)[:200], ", ".join("%s at %s" % (l, self._num(p)) for p, l in rows)))
        left = s.steps_left()
        if not lines and not multi and left >= max(6, s.step_cap // 4) and self._early < 2:
            score = {}
            for e in s.seeker.book().values():
                v = s.seeker.value_rating(e.get("text", ""))
                for lid in (e.get("row_shelf") or {}): score[lid] = max(score.get(lid, 0), v)
            rows = s.seeker.buyable_rows()
            cands = sorted((l for l in rows if l not in read and l not in ref),
                           key=lambda l: (-score.get(l, 0), rows[l][0] if kit.is_num(rows[l][0]) else 1e18, l))[:3]
            self._early += 1
            s.link.note("decline_early", ids=cands)
            return P("the cart is empty with %d steps still available, so declining now is premature. First turn more pages of the search that gave the nearest rows, query every stated value together with the product word, browse the product's category within the budget, and open the most promising unread candidates%s. Cart the cheapest that satisfies all stated conditions; decline only if nothing does." % (left, ("; unopened category rows (rows from value searches first, then by price): " +
                                            ", ".join(cands)) if cands else ""))
        h = "" if self.misfits() else s.mender.closest_pause()
        if h: return P(h)
        if multi and lines and self._room(1) and self._whole < 2:
            slots = [s.walker.product_slot(l) for l in lines]
            if (all(sl is not None and s.passed_now(l) and l not in ref and l not in s.dead for l, sl in zip(lines, slots))
                    and set(slots) >= set(s.ledger.slots()) and self._in_budget() and not self.fit_contradiction()):
                self._whole += 1
                s.link.note("decline_whole", lines=lines)
                return P("each requested product is already covered by a cart line the independent check approved under today's conditions, all are available and together within the budget (%s). Declining would throw away a full basket. No dependent page names a model different from its anchor's, and a fit the pages simply do not confirm is no evidence of a mismatch. Order the cart; decline only when a record proves a line fails a stated condition." % ", ".join(lines))
        ver = [l for l in lines if s.passed_now(l) and l not in ref and l not in s.dead]
        if ver and not self._kept and self._room(1) and not self.misfits():
            self._kept = True
            return P('a verified cart line is present (%s): the check passed it on every condition in force, and a decline would discard it. Re-read its stock and price, then order%s; decline only when a later change plainly rules it out.' % (", ".join(ver), " (covering every product)" if multi else ""))
        bs = s.mender.batch_spec()
        reader = bool(s.role("details") or bs)
        rows = s.mender.squeeze_rows() if not self._squeezed and reader else []
        if rows and self._room(self._reads(rows) + 4, 40):
            self._squeezed = True
            s.judge.queue_open(rows, "decline", say="Reading cheaper unopened rows first: combined, they could bring the whole basket inside the budget.")
            tot = s.mender.squeeze_total()
            return P("every product has fitting listings, yet their cheapest fits add up to %s, above the budget. Some unopened rows lie within each product's share (the budget less the cheapest fits of the other products) and the code is opening them now (%s). Judge each against its own product's conditions with the value rules (a dedicated store value field counts by its words only); if some combination covers every product and stays affordable, cart one line per product and order; decline only if no combination works." % (self._num(tot) if kit.is_num(tot) else "a total beyond the budget", ", ".join(rows)))
        rows = s.mender.short_rows() if not self._lacked and reader else []
        if rows and self._room(self._reads(rows) + 3, 40):
            self._lacked = True
            s.judge.queue_open(rows, "decline", say="Opening never-read listings for products that still lack a verified fit, before any decline.")
            return P('some products still lack a verified fit while affordable listings of their categories were never opened; the code is opening them now, before any decline (%s). Judge each with the value rules; when every product is covered by a fitting listing, cart one line per product and order; decline only for a product that has no fit at all.' % ", ".join(rows))
        q = list(s.judge.unread_leads() or [])[:(bs[2] if bs else 3)] if not self._leaded and reader else []
        if q and self._room(self._reads(q) + 3, 40):
            self._leaded = True
            s.judge.queue_open(q, "decline", say="Fetching the records of listings the checks named but nobody opened, before a decline.")
            return P('these leads were named by the checks but never opened: %s. The code is fetching their records now, before a decline. Value rules: the field named after the property decides (a value the store writes there counts by its words only); when that field is absent, another field, the options menu or the title may give the value. Buy whichever one fits; decline only if all the leads fail.' % ", ".join(q))
        h = s.walker.queue_decline_reads()
        if h: return P(h)
        if (s.audit_mark is None or len(read) > s.audit_mark) and s.audits < 2 and s.decline_vetted and self._room(3, 30):
            h = s.judge.audit()
            if h: return P(h)
        if not s.decline_vetted and self._room(3, 30):
            s.decline_vetted = True
            return P(s.judge.audit() or 'decline checklist. [search] was the search tried from several sides (other words, filters, later pages)? [read] were the property fields of every candidate opened? [fail] does every candidate fail on a stated condition, on a need the shopper disclosed, on the budget or on availability? Field names count whatever their spelling. If a fitting item turns up, buy it; if not, send the decline again.')
        h = s.walker.match_pause()
        if h: return P(h)
        if not self._best and self._room(4):
            pool = list(s.walker.unrefuted() or [])
            for _ in range(3):
                todo = [l for l in pool if not s.passed_now(l)]
                if not todo or s.link.time_left() < 100: break
                s.judge.item_check(todo[0], Tune.Aim.ORDER)
                pool = list(s.walker.unrefuted() or [])
            if pool and self._room(len(pool) + 3, 20):
                self._best = True
                return P('an option remains. A decline is justified only if no listing could meet the conditions, but the records still hold %s that %s available, affordable and ruled out by no check (%s). Unless some record proves one of them fails a current condition, cart %s, re-read stock and price, and order; decline only when each option clearly fails.' % (("options for every product", "are", ", ".join(pool), "a line for every product at its quantity")
                                        if multi else ("one option", "is", ", ".join(pool), "the option")))
        heads = []
        for r in (s.refs if not self._unknown and self._room(3) else ()):
            if r.get("relation") == "differ" and r.get("status") in ("open", "stuck") and s.tracer.live(r):
                hd = str(r.get("head") or r.get("text") or "")
                if hd not in heads: heads.append(hd)
        if heads:
            self._unknown = True
            return P('an unresolved reference: %s depend(s) on a listing the code failed to pin down, so that value is unknown. An unknown value rules nothing out and is no reason to decline. Cart the listing that satisfies all the other conditions (available, affordable%s), re-read its stock and price and order; decline only when no listing satisfies those other conditions.' % ("; ".join(heads), "; one per product" if multi else ""))
        h = s.mender.repair() or s.mender.closest_check()
        if h: return P(h)
        return s.closer.closing_moves("decline")

    # 9. fits between products
    def links(self):
        try:
            pr, out = self.s.ledger.products(), []
            for i, r in enumerate(pr):
                a = r.get("anchor")
                if isinstance(a, int) and not isinstance(a, bool) and 0 <= a < len(pr) and a != i: out.append((i, r, a))
            return out
        except Exception as e:
            self.s.link.fault("links_error", e)
            return []

    def fit_faults(self):
        try:
            out = []
            for dep, anc, dl, al, ap, dp in self._pairs():
                v, names = self._verdict(ap, dp), self._idents(ap)
                marks = self._series(ap) + names
                if v == "match" or not marks or (v == "unknown" and self._named(names, dp)): continue
                out.append("%s (%s) belongs with %s (%s), but its page names none of the anchor's identifiers (%s)" %
                           (dl, self._pname(dep), al, self._pname(anc), ", ".join(marks)))
            return out
        except Exception as e:
            self.s.link.fault("fit_faults_error", e)
            return []

    def fits_written(self):
        s = self.s
        try:
            if not s.multi(): return False
            dg = s.archive.digest()
            for dep, anc, dl, al, ap, dp in self._pairs():
                names, coded, shelf = self._idents(ap), bool(self._series(ap)), s.judge.label_of(dl)
                if (not names and not coded) or not shelf: continue
                for lid, r in dg.items():
                    if lid == dl or r.get("shelf") != shelf: continue
                    op = self._page(lid)
                    if (coded and self._verdict(ap, op) != "unknown") or self._named(names, op): return True
            return False
        except Exception as e:
            s.link.fault("fits_written_error", e)
            return False

    def fit_contradiction(self):
        try:
            for dep, anc, dl, al, ap, dp in self._pairs():
                v = self._verdict(ap, dp)
                if v == "clash": return True
                an = self._idents(ap)
                if v == "unknown" and an and self._idents(dp) and not self._named(an, dp): return True
            return False
        except Exception as e:
            self.s.link.fault("fit_contradiction_error", e)
            return False

    def misfits(self):
        s, out = self.s, []
        try:
            for l in list(s.basket.units):
                f = (s.judge.evidence_view(l) or {}).get("fails") or []
                if f: out.append("%s: %s" % (l, f[0]))
            if self.fit_contradiction():
                out.extend(self.fit_faults() or ["a dependent line's page names models different from those of its anchor"])
        except Exception as e:
            s.link.fault("misfit_error", e)
        return out

    def _shown(self, row, prop, value):
        for f, t in (row.get("fields") or {}).items():
            if kit.property_match(f, prop) and kit.seen(value, str(t)): return True
        return kit.seen(value, row.get("title") or "")

    def _fits(self, slot):
        s, out = self.s, []
        dg = s.archive.digest()
        vals = [(p, v) for p, v in s.ledger.values_in_code(slot) if not kit.has_bound(v) and not self._LOOSE.search(v)]
        for lid in list(s.archive.read):
            p, row = s.archive.price(lid), dg.get(lid)
            if (s.walker.product_slot(lid) != slot or not kit.is_num(p) or s.judge.supply_state(lid) is False or lid in s.dead
                    or lid in s.refused_lines or not row or (s.judge.evidence_view(lid) or {}).get("fails")): continue
            if all(self._shown(row, pr, v) for pr, v in vals): out.append((p, lid))
        return [l for _, l in sorted(out)]

    def best_of_pair(self):
        s = self.s
        try:
            lk = self.links()
            if not self._cheapest() or not lk or not s.archive.read or self._open_ref(): return {}
            stamp = (len(s.archive.obs), s.ledger.version, tuple(sorted(s.refused_lines)), len(s.dead))
            if self._pair[0] == stamp: return dict(self._pair[1])
            anc = lk[0][2]
            tied = [d for d, _, a in lk if a == anc]
            fits, dg, best = {sl: self._fits(sl) for sl in [anc] + tied}, s.archive.digest(), None
            for af in fits[anc]:
                codes = set(self._series(self._page(af)))
                if not codes: continue
                pick, tot = {af: self._qty(anc)}, s.archive.price(af) * self._qty(anc)
                for d in tied:
                    mate = next((l for l in fits[d] if self._tags((dg.get(l) or {}).get("title") or "") & codes), None)
                    if not mate:
                        pick = None
                        break
                    pick[mate] = self._qty(d)
                    tot += s.archive.price(mate) * self._qty(d)
                if pick and (best is None or tot < best[0] - 1e-9): best = (tot, pick)
            self._pair = (stamp, best[1] if best else {})
            return dict(self._pair[1])
        except Exception as e:
            s.link.fault("pair_best_error", e)
            return {}

    def fit_hint(self):
        s, T = self.s, None
        try:
            lk, parts = self.links(), []
            if not lk: return ""
            for dep, row, anc in lk:
                al = next((l for l in s.basket.units if s.walker.product_slot(l) == anc), None)
                pg = self._page(al)
                marks = (self._series(pg) + self._idents(pg)) if al else []
                vw = s.pricer.value_words(dep)
                first = " ".join(vw[0][0]) if vw and vw[0] and vw[0][0] else ""
                tail = (" [search '%s %s' so that only hits also holding that value come back: a short list worth reading to the end]" % (marks[0] if marks else "<identifier>", first)) if first else ""
                dn, an = self._pname(dep), self._pname(anc)
                if marks:
                    parts.append('the %s goes with %s (%s): search its listings once for each of %s, open the hits and keep one whose page names any of them; if no page names any, this store does not write the fit and it decides nothing%s' % (dn, al, an, ", ".join(marks), tail))
                    continue
                seen = []
                for lid, r in s.archive.digest().items():
                    c = self._series(self._page(lid)) if r.get("shelf") and s.walker.product_slot(lid) == anc else []
                    if c: seen.append("%s %s" % (lid, "/".join(c)))
                parts.append('this store expresses the fit as a code: an %s page gives a series code (seen so far: %s); a %s fits only if its tag begins with exactly that code, and a tag of the same series with a different number is a misfit; cart a pair that shares one code, searching the %s listings with the code as the query%s' % (an, ", ".join(seen[-4:]), dn, dn, tail) if seen else 'choose the %s line first, then search the %s listings by model names or numbers and keep one whose page cites such a name' % (an, dn))
            pb = self.best_of_pair()
            if pb:
                tot = sum((s.archive.price(l) or 0) * n for l, n in pb.items())
                parts.append('lowest total so far: %s = %s; linked products are costed as a pair, so a cheaper line whose code is shared only by a more expensive partner does not win' % (", ".join("%s at %s" % (l, self._num(s.archive.price(l))) for l in pb), self._num(tot)))
            return "\n[works with] " + "; ".join(parts)
        except Exception as e:
            s.link.fault("fit_hint_error", e)
            return ""

    # 9.12 the fit search (move 2)
    def _fit_forms(self, slot, code):
        s = self.s
        keys = s.pricer.value_keys(slot)
        e = s.seeker.book().get(keys[-1]) if keys else None
        a = (e or {}).get("args") or {}
        ft = s.seeker.open_text(a) if e else []
        if len(ft) != 1: return None, []
        fld, sch = ft[0][0], s.schema(s.role("search"))
        props, req = sch.get("properties") or {}, sch.get("required") or []
        tpl = {k: v for k, v in a.items() if k == fld or k in req or (isinstance(props.get(k), dict) and props[k].get("enum"))}
        tail = list(s.pricer.label_words(slot) or []) or [w for w in kit.words(self._pname(slot)) if len(w) > 2 and not kit.is_link(w)]
        vw = s.pricer.value_words(slot)
        mx = props[fld].get("maxLength") if isinstance(props.get(fld), dict) else None
        forms = []
        for alt in (list(vw[0][:2]) if vw and vw[0] else []) + [[]]:
            ws = []
            for w in [code] + list(alt) + tail:
                if w and w not in ws: ws.append(w)
            ws = ws[:s.lead_words] if s.lead_words else ws
            t = " ".join(ws)
            if isinstance(mx, int) and mx > 0:
                while len(t) > mx and len(ws) > 1:
                    ws.pop()
                    t = " ".join(ws)
                t = t[:mx]
            if t and t not in forms: forms.append(t)
        return (tpl, fld), forms

    def _tagged(self, new, slot, code):
        s, out = self.s, []
        home = set(s.walker.tour_shelf(slot) or ())
        for ob in new:
            for d in self._dicts(ob, 3):
                rid = kit.row_id(d) if len(d) > 3 else ""
                if not rid or rid not in s.archive.known or rid in out: continue
                if (not home or (kit.shelf_field(d) or s.judge.label_of(rid)) in home) and code in self._tags(kit.row_title(d)):
                    out.append(rid)
        return out

    def match_query(self):
        s = self.s
        try:
            if not self._fitq: return False
            floor = s.intake.due_margin() + 2
            if not self._room(floor + 2, 15):
                self._fitq = []
                return False
            sch = s.schema(s.role("search"))
            while self._fitq:
                slot, code = self._fitq[0]
                tp, forms = self._fit_forms(slot, code)
                tried = self._fit_tried.setdefault(slot, [])
                todo = [f for f in forms if f not in tried]
                while tp and todo and len(tried) < 3:
                    q = todo.pop(0)
                    tried.append(q)
                    a = dict(tp[0])
                    a[tp[1]] = q
                    a, bad = ArgFixer.fix(a, sch)
                    if bad: continue
                    n0, st0, pn = len(s.archive.obs), s.link.steps, self._pname(slot)
                    s.walker.search_call(a, "Searching '%s': the series code comes first so that listings tagged with it come back." % q)
                    new = s.archive.obs[n0:]
                    if not new:
                        if s.link.steps != st0: return True
                        continue
                    hits = self._tagged(new, slot, code)
                    if not hits and len(tried) < 3 and todo:
                        s.engine.append_note("[code fit] '%s': no %s row tagged %s." % (q, pn, code), status=True)
                        return True
                    self._fitq.pop(0)
                    bs = s.mender.batch_spec()
                    opens = [l for _, l in s.pricer.low_unopened(hits, slot)][:min(bs[2] if bs else 4, 4)]
                    opens = list(s.pricer.trim(opens, floor) or [])
                    note = "[code fit] '%s': %s rows tagged %s: %s." % (q, pn, code, ", ".join(hits) or "none")
                    if opens: note += " Opening next: " + ", ".join("%s (%s)" % (l, self._num(s.archive.price(l))) for l in opens) + "."
                    s.engine.append_note(note, status=not opens)
                    s.link.note("match_query", slot=slot, code=code, query=q, hits=hits, opens=opens)
                    if opens:
                        s.pricer.launch(opens, set(s.walker.tour_shelf(slot) or ()), floor,
                                        "These %s hits carry the code %s that the anchor states; reading them." % (pn, code))
                    return True
                self._fitq.pop(0)
            return False
        except Exception as e:
            s.link.fault("fit_search_error", e)
            return False

    # 10. reach moves
    def disclosure_memo(self, needs):
        s = self.s
        try:
            for line in ([] if s.multi() else needs or []):
                line = str(line)
                v = kit.low(kit.strip_marks(line.split(":", 1)[1] if ":" in line else line)).strip()
                first = next((p.strip() for p in self._SPLIT.split(v) if p.strip()), "") if v and not kit.has_bound(v) else ""
                if (first and not kit.is_pointer(first) and len(first.split()) <= 4
                        and self._rev_sent + len(self._rev_q) < 2 and first not in self._rev_q): self._rev_q.append(first)
        except Exception as e:
            s.link.fault("reveal_note_error", e)

    def disclosure_settle(self, ref):
        s = self.s
        try:
            rel = ref.get("relation")
            if s.multi() or rel not in ("pairs", "same"): return
            v = ((ref.get("partners") or [""])[0] if rel == "pairs" else ref.get("value")) or ""
            v = kit.low(str(v)).strip()
            w, prop = v.split(), str(ref.get("prop") or "")
            if len(w) >= 2 and prop and (w[-1] == kit.low(prop).strip() or kit.property_slug(w[-1]) == kit.property_slug(prop)):
                rest = " ".join(w[:-1])
                if any(kit.property_match(f, prop) and rest in kit.low(str(t)) for r in s.archive.digest().values()
                       for f, t in (r.get("fields") or {}).items()): v = rest
            if not v or kit.is_pointer(v) or kit.has_bound(v) or len(v.split()) > 4: return
            self._rev_q = [" ".join([v] + self._rev_q)]
            self._rev_deep = self._rev_q[0]
        except Exception as e:
            s.link.fault("reveal_settle_error", e)

    def _open_rows(self, keys, whole):
        s, ids = self.s, []
        home, book = set(s.walker.tour_shelf(None) or ()), s.seeker.book()
        for k in keys:
            for lid, sh in ((book.get(k) or {}).get("row_shelf") or {}).items():
                if (not home or sh in home) and lid not in ids: ids.append(lid)
        pk = s.walker.pick_of(None)
        rows = [l for _, l in s.pricer.low_unopened(ids, None, below=pk[1] if pk else None)]
        if whole: rows = rows[:8]
        else:
            dg, vals = s.archive.digest(), [v for _, v in s.ledger.values_in_code(None)]
            hit = {l for l in rows if vals and all(kit.seen(v, (dg.get(l) or {}).get("title") or "") for v in vals)}
            rows = ([l for l in rows if l in hit] + [l for l in rows if l not in hit])[:4]
        # nothing to open with (no single-listing details tool and no list reader): the move did not act (sim parity)
        if not rows or not (s.role("details") or s.mender.batch_spec()): return False
        s.judge.queue_open(rows, "walk", False, say="Opening the cheapest rows of that search.")
        return True

    def _solo_gate(self, extra):
        s = self.s
        return not (s.multi() or s.basket.units or s.orders_tried or not s.seeker.paging()
                    or not self._room(s.intake.due_margin() + extra, 50))

    def disclosure_push(self):
        s = self.s
        try:
            if not self._rev_q and not self._rev_job: return False
            if not self._solo_gate(4):
                self._rev_q, self._rev_job = [], None
                return False
            j = self._rev_job
            while j is None and self._rev_q:
                v = self._rev_q.pop(0)
                tp, pw = s.seeker.template(), kit.low(s.ledger.product_name() or "").strip()
                if not tp or not pw: continue
                a = dict(tp[0])
                a[tp[1]] = "%s %s" % (v, pw)
                a, bad = ArgFixer.fix(a, s.schema(s.role("search")))
                if bad: continue
                self._rev_sent += 1
                j = self._rev_job = {"args": a, "pages": 0, "deep": v == self._rev_deep, "ran": False, "value": v}
            if j is None: return False
            if not j["ran"]:
                j["ran"] = True
                return bool(s.walker.search_call(j["args"], "Searching the revealed value '%s' together with the product words." % j["value"]))
            key = s.seeker.book_key(j["args"])
            sp = s.seeker.following_page(key) if j["pages"] < (24 if j["deep"] else 8) // max(4, s.pricer.page_rows() or 0) else None
            if sp is not None:
                j["pages"] += 1
                s.walker.search_call(s.seeker.page_call(key, sp), "Next page.")
                return True
            self._rev_job = None
            self._deep_done = self._deep_done or j["deep"]
            return self._open_rows([key], j["deep"])
        except Exception as e:
            s.link.fault("reveal_reach_error", e)
            return False

    def single_push(self):
        s = self.s
        try:
            if self._deep_done or (s.role("message") and s.msgs_sent == 0) or not self._solo_gate(3) or self._cap_settled():
                return False
            book, best = s.seeker.book(), None
            if self._solo_rows < 24:
                home, chase = set(s.walker.tour_shelf(None) or ()), set()
                for r in s.refs: chase |= {kit.low(t) for t in (r.get("tried") or ())}
                for k, e in book.items():
                    n = sum(1 for sh in (e.get("row_shelf") or {}).values() if not home or sh in home)
                    sp = s.seeker.following_page(k) if n > 0 and e.get("text", "") not in chase else None
                    rk = (s.seeker.value_rating(e.get("text", "")), n) if sp is not None else None
                    if rk and (best is None or rk > best[0]): best = (rk, k, sp)
            if best is None:
                self._deep_done = True
                return self._open_rows(list(book), False) if self._solo_rows else False
            self._solo_rows += max(1, s.pricer.page_rows() or 0)
            s.walker.search_call(s.seeker.page_call(best[1], best[2]), "Next page of the closest search.")
            return True
        except Exception as e:
            s.link.fault("solo_reach_error", e)
            return False

    def _cap_tool(self):
        s = self.s
        for t in s.tools:
            name = (t.get("function") or {}).get("name")
            sch = s.schema(name) if name else {}
            props, cat, tok, cap = sch.get("properties") or {}, None, None, None
            for k, p in props.items():
                ty = p.get("type") if isinstance(p, dict) and not p.get("enum") else None
                if ty == "string" and cat is None and kit.shelf_field({k: "x"}): cat = k
                elif ty == "string" and tok is None and self._TOKEN_ARG.search(k): tok = k
                elif ty in ("integer", "number") and cap is None and self._CAP_ARG.search(k) and not self._CAP_NOT.search(k): cap = k
            if cat and tok and cap and set(sch.get("required") or []) <= {cat, cap}:
                return name, cat, tok, cap, props[cap].get("type") == "integer"
        return None

    @staticmethod
    def _below(x, integer):
        if not kit.is_num(x): return None
        r = int(x) - 1 if float(x).is_integer() else (math.floor(x) if integer else round(x - 0.01, 2))
        return r if r > 0 else None

    def _table(self, ob):
        hold = [ob] + ([v for v in ob.values() if isinstance(v, dict)] if isinstance(ob, dict) else [])
        for empty_ok in (False, True):
            for h in hold:
                if isinstance(h, list):
                    if all(isinstance(x, dict) for x in h) and (h or empty_ok): return (len(h), len(h), None)
                    continue
                for v in h.values():
                    if isinstance(v, list) and all(isinstance(x, dict) for x in v) and (v or empty_ok):
                        big = [x for x in h.values() if isinstance(x, int) and not isinstance(x, bool) and x > len(v)]
                        return (len(v), max(big) if big else len(v), self.s.seeker.page_cursor(h))
        return None

    def _row_prices(self):
        return [r[0] for r in self.s.seeker.buyable_rows().values() if kit.is_num(r[0])]

    def cap_push(self):
        s = self.s
        try:
            if self._cap_done or not self._cheapest() or (s.role("message") and s.msgs_sent == 0): return False
            ct, home, prices = self._cap_tool(), set(s.walker.tour_shelf(None) or ()), self._row_prices()
            label = s.pricer.bare_label(home) if home else ""
            if not ct or not label or not s.seeker.buyable_rows() or not self._solo_gate(3):
                self._cap_done = True
                return False
            tool, cat, tok, cap, integer = ct
            j = self._cap_job
            if j is None:
                j = self._cap_job = {"calls": 0, "bound": self._below(min(prices), integer) if prices else None,
                                     "token": None, "lifted": False, "complete": False}
            if j["bound"] is None or j["calls"] >= 6 or j["complete"]: return self._cap_open()
            a = {cat: label, cap: j["bound"]}
            if j["token"]: a[tok] = j["token"]
            n0, low0 = len(s.archive.obs), (min(prices) if prices else None)
            if not s.walker.search_call(a, "Listing the category below the lowest price seen: results arrive in id order "
                                           "and the cheapest is wanted.", tool=tool): return self._cap_open()
            j["calls"] += 1
            tab = next((t for t in (self._table(ob) for ob in s.archive.obs[n0:]) if t), None)
            shown, total, nxt = tab or (0, 0, None)
            if shown == 0 and not j["lifted"]:
                j["lifted"], j["bound"] = True, (2 * low0 if kit.is_num(low0) else None)
            elif shown == 0 or total <= 12:
                j["token"], j["complete"] = nxt, not nxt
            else:
                now, hi = self._row_prices(), j["bound"]
                lo = min(now) if now else hi
                nb = (int(lo) + int(hi)) // 2 if integer or (float(lo).is_integer() and float(hi).is_integer()) else round((lo + hi) / 2, 2)
                j["complete"], j["bound"] = not (nb < hi), nb
            return True
        except Exception as e:
            s.link.fault("cap_reach_error", e)
            return False

    def _cap_open(self):
        s = self.s
        self._cap_done = True
        bs = s.mender.batch_spec()
        ids = [l for _, l in s.pricer.low_unopened(list(s.seeker.buyable_rows()), None)][:(2 * bs[2] if bs else 4)]
        if not ids: return False
        s.judge.queue_open(ids, "walk", False, say="Opening the category's cheapest rows.")
        return True

    def _cap_settled(self):
        s, j = self.s, self._cap_job
        if not j or not j["complete"] or not kit.is_num(j["bound"]): return False
        return any(lid in s.archive.read and kit.is_num(r[0]) and r[0] <= j["bound"] + 1e-9
                   and not (s.judge.evidence_view(lid) or {}).get("fails") for lid, r in s.seeker.buyable_rows().items())

    def _few_room(self):
        return self.s.steps_left() > self.s.intake.due_margin() + 6 and self.s.link.time_left() > 28

    def _few_launch(self, q, shelves, say):
        s = self.s
        floor = s.intake.due_margin() + 6
        while q and self._reads(q) >= s.steps_left() - floor: q = q[:-1]
        if q: s.pricer.launch(q, shelves, floor, say)
        return bool(q)

    def _fit_codes(self, slot):
        out = []
        for f in self._fits(slot):
            out.extend(c for c in self._series(self._page(f)) if c not in out)
        return out

    def sparse_push(self):
        s = self.s
        try:
            if (s.role("message") and s.msgs_sent == 0) or (self._open_ref() and s.tracer.has_room()) or self._few_done: return False
            prods = s.ledger.products()
            if (not self._cheapest() or len(prods) not in (2, 3) or s.basket.units or s.orders_tried
                    or not s.seeker.paging() or not self._few_room()):
                self._few_done = True
                return False
            deps = {d for d, _, _ in self.links()}
            j = self._few = self._few or {"todo": [i for i in range(len(prods)) if i not in deps and s.ledger.values_in_code(i)],
                                          "args": {}, "paged": {}, "sent": set(), "done": [], "hunts": None, "tagged": False}
            sch = s.schema(s.role("search"))
            while j["todo"]:
                sl = j["todo"][0]
                if sl not in j["args"]:
                    a, tp, fm = None, s.seeker.template(), s.pricer.value_forms(sl)
                    if tp and fm:
                        a = dict(tp[0])
                        a[tp[1]] = fm[0]
                        a, bad = ArgFixer.fix(a, sch)
                        a = None if bad else a
                    j["args"][sl] = a
                a = j["args"][sl]
                # "never ran" means no search with these arguments was booked yet, by any move (11 10.4)
                if a is not None and sl not in j["sent"] and s.seeker.book_key(a) not in (s.seeker.book() or {}):
                    j["sent"].add(sl)
                    s.walker.search_call(a, "Searching the required value together with the category words, because the "
                                            "cheapest fit may sit far down the list.")
                    return True
                if a is not None and j["paged"].get(sl, 0) < 16 and self._few_room():
                    key = s.seeker.book_key(a)
                    sp = s.seeker.following_page(key)
                    if sp is not None:
                        j["paged"][sl] = j["paged"].get(sl, 0) + max(1, s.pricer.page_rows() or 0)
                        s.walker.search_call(s.seeker.page_call(key, sp), "Next page.")
                        return True
                j["done"].append(j["todo"].pop(0))
            if j["done"]:
                q, sh = [], set()
                for sl in j["done"]:
                    home = set(s.walker.tour_shelf(sl) or ())
                    sh |= home
                    q.extend(l for _, l in s.pricer.low_unopened(s.pricer.aisle_rows(home), sl)[:5] if l not in q)
                j["done"] = []
                if self._few_launch(q, sh, "Reading the cheapest rows found by the value searches."): return True
            if j["hunts"] is None:
                j["hunts"] = []
                for dep, row, anc in ([] if self._open_ref() else self.links()):
                    vw = s.pricer.value_words(dep)
                    for c in self._fit_codes(anc)[:3]:
                        for alt in ((list(vw[0][:2]) if vw and vw[0] else []) or [[]]):
                            q = " ".join([c] + list(alt))
                            if q not in j["hunts"]: j["hunts"].append(q)
            tp = s.seeker.template()
            while tp and j["hunts"] and self._few_room():
                q = j["hunts"].pop(0)
                a = dict(tp[0])
                a[tp[1]] = q
                a, bad = ArgFixer.fix(a, sch)
                if bad or s.seeker.already_ran(q): continue
                s.walker.search_call(a, "Searching '%s' as keywords, with the anchor's series code leading." % q)
                return True
            if not j["tagged"]:
                j["tagged"], q, sh, dg = True, [], set(), s.archive.digest()
                for dep, row, anc in self.links():
                    home = set(s.walker.tour_shelf(dep) or ())
                    sh |= home
                    ids = s.pricer.aisle_rows(home)
                    for c in sorted(set(self._fit_codes(anc))):
                        tg = [l for l in ids if c in self._tags((dg.get(l) or {}).get("title") or "")]
                        q.extend(l for _, l in s.pricer.low_unopened(tg, dep)[:3] if l not in q)
                if self._few_launch(q, sh, "Reading the cheapest rows tagged with each anchor code."): return True
            self._few_done = True
            return False
        except Exception as e:
            s.link.fault("few_reach_error", e)
            return False

    # 11.2 bound hint
    def limit_hint(self):
        s = self.s
        try:
            if not self._cheapest() or s.link.time_left() < 70: return ""
            figs = []
            for line in s.ledger.in_force():
                v = kit.low(str(line).split(":", 1)[1] if ":" in str(line) else str(line))
                bnd = kit.has_bound(v)
                for m in self._FIG.finditer(v):
                    f = m.group(1) + m.group(2)  # as built: a short linking word counts as a unit ("3 or more" -> "3or")
                    if not (bnd and "." in m.group(1)) and f not in figs: figs.append(f)
            return ('\n[cheapest pick, limit given as a measure] the cheapest fitting listing usually writes that exact figure in its own text: search each of %s next to the product word, written both joined to its unit and with a space, read two result pages for each, then open the cheapest affordable rows, all before settling the pick.' % ", ".join(figs[:3])) if figs else ""
        except Exception as e:
            s.link.fault("bound_hint_error", e)
            return ""

    # 12. change review
    def shift_review(self):
        s, T = self.s, None
        try:
            if (self._reviews >= 3 or not s.archive.read or s.link.mute or s.link.time_left() < 90
                    or not s.link.usable(LINEUPS["reviewer"][0]) or s.steps_left() < 3): return ""
            self._reviews += 1
            cards = s.archive.cards(budget=14000, skip=s.tracer.foreign_ids())
            if not cards: return ""
            L = s.ledger
            pr, pm = L.priority(), L.pick_measure()
            pr = pr if pr == "cheapest" else (("best (%s)" % pm if pm else "best") if pr == "best" else "none")
            sec = lambda h, b: (h + "\n" + str(b)) if b and str(b).strip() else ""
            parts = [PROMPTS["review"], sec("REQUEST (the shopper's words; later changes follow):", s.request), L.drops_text(),
                     sec("WHAT JUST CHANGED:", " / ".join(str(x) for x in (L.exchanges or [])[-2:])),
                     sec("REQUIREMENTS NOW:", L.req_text()), "PRIORITY: " + pr, sec("CART:", s.basket.summary()),
                     sec("LISTING CARDS:", "Legend: a DETAILS card comes from the listing's whole record as read; a ROW card holds search-row data only." + "\n" + "\n".join(cards)), s.intake.field_counts_block()]
            ans = s.link.consult("review-change", "\n\n".join(p for p in parts if p), "reviewer")
            s.link.note("shift_review", result=ans if isinstance(ans, dict) else None)
            if not isinstance(ans, dict): return ""
            known, fits, raw = s.archive.known, [], ans.get("fit_ids")
            for x in (raw if isinstance(raw, list) else []):
                x = str(x).strip() if isinstance(x, (str, int, float)) and not isinstance(x, bool) else ""
                if x in known and x not in fits: fits.append(x)
            fits = fits[:6]
            pk, ok = ans.get("order_now"), ans.get("cart_ok")
            pk = str(pk).strip() if pk is not None and not isinstance(pk, bool) and str(pk).strip() in known else None
            if isinstance(ok, str): ok = {"true": True, "false": False}.get(ok.strip().lower())
            if fits: s.judge.add_leads(fits)
            out = ["[after the change]", ("Fits under today's terms, strongest first: %s." % ", ".join(fits)) if fits
                   else "No opened record is yet known to meet today's conditions."]
            if pk: out.append("pick now: %s." % pk)
            if s.basket.units and ok is True: out.append('Leave the cart alone: its line remains a valid pick, and the basket the shopper heard about is the one to send.')
            elif s.basket.units and ok is False: out.append('Swap the cart line: under the present terms it is no longer the right pick (if the instructions demand consent before a replacement, ask the shopper first).')
            why = str(ans.get("why") or "").strip()
            if why: out.append(why[:300])
            return " ".join(out + ['Choose again among the present fits (the cheapest or best of them when the priority says so), correct the cart, re-read stock and price, and finish; while an opened record still fits, do not decline.'])
        except Exception as e:
            s.link.fault("change_review_error", e)
            return ""

    # 13. code-asked swap
    def swap_consent(self, old, new):
        s = self.s
        try:
            mt = s.role("message")
            if (not mt or not s.mender.sole_text_arg(mt) or s.steps_left() < 4
                    or s.intake.question_limit() - s.msgs_sent < s.intake.message_reserve() + 1): return False
            old, new = list(old or [])[:3], list(new or [])[:3]
            args = s.mender.chat_call('Before changing your basket, I would like to remove %s and add %s, because the current pick has stopped fitting. Is that fine with you?' % (", ".join(old) or "nothing", ", ".join(new) or "nothing"))
            if not args: return False
            pairs = s.engine.send_booked([s.engine.call(mt, args)])
            yes = bool(pairs) and not str(pairs[0][1]).startswith("Not run")
            s.link.note("code_swap_ask", old=old, new=new, sent=yes)
            return yes
        except Exception as e:
            s.link.fault("code_swap_ask_error", e)
            return False

    # 14. telling the basket
    def cart_mark(self):
        s, L = self.s, self.s.ledger
        return (s.cart_edits, tuple(sorted(s.basket.units.items())), tuple(L.written or ()), tuple(L.revealed or ()),
                tuple(L.dropped or ()), L.budget, L.duties_version, s.refusals)

    def told_current(self):
        try:
            m = self.s.basket_told_mark
            return bool(m is not None and self.s.basket.units and m == self.cart_mark())
        except Exception as e:
            self.s.link.fault("told_current_error", e)
            return False

    def states_basket(self, text):
        s = self.s
        try:
            units = s.basket.units
            if not units or not text: return False
            # as built: text comparison, so "12.50" in the message does not name a price of 12.5
            nums = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"(?<=\d),(?=\d)", "", str(text))))
            pr = {l: s.archive.price(l) for l in units}
            if all(l in text or (kit.is_num(p) and self._num(p) in nums) for l, p in pr.items()): return True
            return all(kit.is_num(p) for p in pr.values()) and self._num(sum(p * units[l] for l, p in pr.items())) in nums
        except Exception as e:
            s.link.fault("states_basket_error", e)
            return False

    def line_vendor(self, lid):
        s, found = self.s, ""
        try:
            learned = set(s.tracer.seller_fields() or ())
            for ob in s.archive.obs:
                for d in self._dicts(ob):
                    if not any(isinstance(v, str) and v.strip() == lid and "id" in str(k).lower() for k, v in d.items()): continue
                    for k, v in d.items():
                        if (isinstance(v, str) and 1 <= len(v.strip()) <= 60 and "id" not in str(k).lower()
                                and (kit.is_seller_field(k) or k in learned)): found = v.strip()
            return found
        except Exception as e:
            s.link.fault("line_seller_error", e)
            return ""

    def cart_summary(self):
        s = self.s
        try:
            units = s.basket.units
            if not units: return "The cart holds nothing, so no order will follow."
            dg, parts, total, priced = s.archive.digest(), [], 0.0, True
            for lid, n in units.items():
                row = dg.get(lid) or {}
                t = "%s x %s" % (self._num(n) if kit.is_num(n) else n, lid)
                if row.get("title"): t += " (%s)" % str(row["title"])[:60]
                sl = self.line_vendor(lid)
                if sl: t += " from %s" % sl
                p = s.archive.price(lid)
                p = p if kit.is_num(p) else row.get("price")
                if kit.is_num(p):
                    t += " at %s each" % self._num(p)
                    total += p * (n if kit.is_num(n) else 1)
                else: priced = False
                parts.append(t)
            msg = "Your basket is about to be finished: " + "; ".join(parts) + "."
            cur = s.ledger.money_unit() if priced else ""
            if priced: msg += " Total: %s%s." % (self._num(total), (" " + cur) if cur else "")
            return msg + " I am placing the order now."
        except Exception as e:
            s.link.fault("basket_text_error", e)
            return ""



# ------------------------------------------------------------ p14_closer.py
class Closer:
    """Finish steps (11 15), code shopper, forced finish (07 3.8-3.9), shopping helpers (06 4, 6)."""
    ROUNDS = 4
    QUERY_RE = re.compile(r"^q$|query|text|needle|term|word|phrase|string|search")
    PAGING_RE = re.compile(r"page|first|offset|skip|count|size|num")
    CEIL_RE = re.compile(r"ceil|max|price|budget|upper|most|cap$")
    ID_RE = re.compile(r"id|item|product|listing|sku|target|which|number")
    TOKEN_RE = re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")
    MISFIT_RE = re.compile(r"(?:line )?(\S+?):? ")
    SPLIT_RE = re.compile(r"\bor\b|/|,")

    def __init__(self, s):
        self.s = s
        self._marks = set()
        self._searched = False

    def _guard(self, tag, fn, neutral, *a):
        try:
            return fn(*a)
        except Exception as e:
            self.s.link.fault(tag + "_error", e)
            return neutral

    def _blocked(self, steps, secs):
        s = self.s
        return bool(s.link.done or s.link.broke or s.steps_left() < steps or s.link.time_left() < secs)

    @staticmethod
    def _types(p):
        t = p.get("type") if isinstance(p, dict) else None
        return t if isinstance(t, list) else [t]

    def _is_text(self, p):
        return isinstance(p, dict) and "string" in self._types(p) and not p.get("enum")

    def _is_num(self, p):
        return bool({"integer", "number"} & set(map(str, self._types(p))))

    @staticmethod
    def _whole(x):
        return int(x) if isinstance(x, float) and x.is_integer() else x

    @staticmethod
    def _cut(text, p):
        mx = p.get("maxLength") if isinstance(p, dict) else None
        return text[:mx] if type(mx) is int and mx >= 0 else text

    def _shape(self, tool):
        sc = self.s.schema(tool) if tool else {}
        props = sc.get("properties") if isinstance(sc.get("properties"), dict) else {}
        req = sc.get("required") if isinstance(sc.get("required"), list) else []
        return props, [r for r in req if r in props]

    def _desc(self, tool):
        f = (self.s.spec(tool) or {}).get("function")
        return str(f.get("description") or "") if isinstance(f, dict) else ""

    def closing_moves(self, kind):
        return self._guard("closing_moves", self._finish_steps, "", kind)

    def _finish_steps(self, kind):
        s = self.s
        if s.link.done or s.link.broke or s.steps_left() < 2:
            return ""
        units = s.basket.units
        mark = (kind, tuple(sorted((str(k), str(v)) for k, v in units.items())), s.cart_edits,
                s.ledger.duties_version, s.refusals, s.ledger.version)
        if mark in self._marks:
            return ""
        self._marks.add(mark)
        quiet = not s.ledger.duties and not s.ledger.exchanges and not s.event_seen
        msg = s.role("message")
        left = max(0, s.intake.question_limit() - s.msgs_sent) if msg else 0
        since = [(e[0], e[1]) for e in s.msg_log if e[2] == s.cart_edits] if (s.cart_edits >= 1 or not units) else []
        shut = {s.role("order"), s.role("decline"), s.role("cart")}
        names = [(t.get("function") or {}).get("name") for t in s.tools if isinstance(t, dict)]
        offered = [n for n in names if n and n not in shut and not (n == msg and not left)]
        steps, ans = [], None
        if not quiet and s.link.usable(LINEUPS["finisher"][0]) and s.link.time_left() > Tune.Clock.FINISH_CHECK_MIN:
            ans = s.link.consult("check-finish", self._brief(kind, offered, since, left), "finisher")
            owed = ans.get("owed") if isinstance(ans, dict) else None
            for it in (owed if isinstance(owed, list) else [])[:Tune.Finish.CHECK_STEPS]:
                tool = it.get("tool") if isinstance(it, dict) else None
                if isinstance(tool, str) and tool in offered and isinstance(it.get("args"), dict):
                    fixed, faults = ArgFixer.fix(it["args"], s.schema(tool))
                    if not faults:
                        steps.append([tool, fixed, str(it.get("purpose") or "")[:120]])
        arg = s.mender.sole_text_arg(msg) if msg else None
        if kind == "order" and units and arg and not since and left >= 1:
            told = s.warden.cart_summary()
            mine = [st for st in steps if st[0] == msg]
            if not mine:
                a = s.mender.chat_call(told)
                if a:
                    steps.insert(0, [msg, a, "informs the shopper of the cart contents before ordering"])
            else:
                for st in mine:
                    if not s.warden.states_basket(" ".join(v for v in st[1].values() if isinstance(v, str))):
                        k = arg if isinstance(st[1].get(arg), str) else next(
                            (k for k, v in st[1].items() if isinstance(v, str)), arg)
                        st[1][k] = (str(st[1].get(k) or "").rstrip() + " " + told).strip()
                steps = mine + [st for st in steps if st[0] != msg]
        s.link.note("finish_check", finish=kind, quiet=quiet, answer=ans, steps=[list(st) for st in steps])
        steps = steps[:max(0, s.steps_left() - 1)]
        if not steps:
            return ""
        before = s.engine.ledger_mark()
        calls = [s.engine.call(t, a, cid="fstep_%d_%d" % (i, s.link.steps)) for i, (t, a, _r) in enumerate(steps)]
        was = s.scripted
        try:
            got = dict(s.engine.run_batch(calls, True) or [])
        finally:
            s.scripted = was
        moved = before != s.engine.ledger_mark()
        bits = ["%s %s -> %s" % (t, kit.render(a)[:400], str(got.get(c["id"], "")).split("\n[status]")[0][:700])
                for c, (t, a, _r) in zip(calls, steps)]
        note = "[code steps done first: %s] " % "; ".join(st[2] for st in steps if st[2]) + " | ".join(bits)
        s.link.note("closing_moves", finish=kind, moved=moved, note=note[:2000])
        if any(st[0] == msg for st in steps) and (moved or s.ledger.reply_due or s.link.done):
            return (Tune.Marks.NOT_RUN_YET + note + " That reply changed something: check it against the requirements"
                    " and every standing instruction, do what it asks, and only then try the finish again.")
        s.finish_note = note
        return ""

    def _brief(self, kind, offered, since, left):
        s = self.s
        duty = "\n".join("- %s (from the %s)" % (d.get("text", ""), d.get("source") or "request")
                         for d in s.ledger.duties if isinstance(d, dict)) or "(none stated)"
        act = "\n".join("- [%s] sent: %s" % x for x in since) or "(no message and no other step followed the latest cart edit)"
        parts = [("[1] INSTRUCTIONS IN FORCE:", duty), ("[2] THE SHOPPER'S ORIGINAL WORDS:", s.request),
                 ("[3] RECENT WORDS FROM SHOPPER OR STORE:", "\n".join(map(str, s.ledger.exchanges[-8:])) or "(none)"),
                 ("[4] REQUIREMENTS NOW:", s.ledger.req_text()),
                 ("[5] THE CART NOW:", "\n".join(x for x in (s.basket.summary(), s.warden.cart_summary()) if x)),
                 ("[6] THE FINISH ABOUT TO BE SENT:",
                  "order (buy what the cart holds)" if kind == "order" else "decline (end without buying)"),
                 ("[7] ACTIVITY SINCE THE LATEST CART EDIT:", act),
                 ("[8] MOVES LEFT:", "%d message(s) and %d step(s) left; one of those steps is needed for the finish"
                  " itself." % (left, s.steps_left())),
                 ("[9] TOOLS FOR AN OPEN STEP:", "\n".join("- %s: %s | arguments: %s" % (
                     n, self._desc(n), kit.render(s.schema(n))) for n in offered))]
        return "\n\n".join([PROMPTS["finish"]] + [h + "\n" + c for h, c in parts if c])

    def code_shop(self, barred, preferred):
        return self._guard("code_finish", self._shop, False, set(barred or ()), list(preferred or ()))

    def _shop(self, bar, pref):
        s = self.s
        if not s.role("cart") or self._blocked(2, 4):
            return False
        s.link.mute = True  # left on; callers that need models save and restore it
        n = max(1, len(s.ledger.products()))
        cheap = s.ledger.priority() == "cheapest"
        for _ in range(self.ROUNDS):
            plan = self.picks(bar, pref)
            if s.role("search"):
                unfit = not plan or not all(s.code_fit.get(l) for l, _u in plan)
                if not self._searched and s.steps_left() >= 4 and unfit:
                    self._searched = bool(s.seeker.rule_query(n)) or self._searched
                    plan = self.picks(bar, pref)
                elif s.steps_left() >= 6 + 2 * n and (unfit or cheap):
                    s.seeker.rule_query(n, more=True)  # sends nothing when no next pages are pending
                    plan = self.picks(bar, pref)
            if plan:
                plan = s.seeker.rule_open(plan, bar, pref) or []
            if not plan:
                s.link.note("code_finish", picks=[], result="nothing to order")
                return False
            got = s.seeker.rule_cart(plan)
            if got is None:
                s.link.note("code_finish", picks=[list(x) for x in plan], result="no cart call")
                return False
            bad = sorted(got)
            if not bad and s.steps_left() >= len(plan) + 3:
                bad = sorted(s.seeker.recheck_stock([l for l, _u in plan])[1] or ())
            if not bad:
                s.link.note("code_finish", picks=[list(x) for x in plan], cart=dict(s.basket.units))
                return bool(s.basket.units)
            bar |= set(bad)
            s.link.note("code_swap", lines=bad)
        return False

    def force_finish(self):
        # errors propagate to the caller's crash handler, as built (12 3.3.11)
        s = self.s
        if self._blocked(1, 2):
            return
        otool, dtool = s.role("order"), s.role("decline")
        st = {"bar": set(), "pref": None}
        move, kind, last, turn = None, None, None, 0
        while turn < Tune.Finish.CODE_TRIES:
            if self._blocked(1, 2):
                return
            if move is None:
                move = self._pick(turn == 0, st, otool, dtool)
                if move is None:
                    return
            name, args = move[0], move[1]
            kind = s.intake.finish_kind(name, args) or move[2]
            self.closing_moves(kind)  # guarded inside: errors are logged as finish_steps_error
            if s.link.done or s.link.broke or s.steps_left() <= 0:
                return
            last = s.link.store_call(name, args)
            s.link.note("forced_finish", action={"name": name, "args": args}, reply=last)
            if (not isinstance(last, dict) or last.get("finished") or s.link.done
                    or turn == Tune.Finish.CODE_TRIES - 1 or s.steps_left() <= 1):
                break
            s.link.mute = True
            held = dict(s.basket.units)
            many = last.get("observations")
            obs = many[0] if isinstance(many, list) and many else last.get("observation")
            if obs is not None:
                s.engine.book(name, args, obs, last)
                look = kit.lens(obs)
                refused = s.seeker.is_refused(name, look, bool(last.get("finished")))
                s.seeker.late_budget(s.warden.buyer_lines(name, look, refused))
            prev, move = move, None
            if kind == "order" and held:
                move = self._repair(held, prev, st)
            turn += 1
        if (s.link.done or s.link.broke or not isinstance(last, dict) or last.get("finished") or not dtool
                or kind == "decline" or s.steps_left() <= 0):
            return
        retry = self._guard("again", lambda: bool(otool) and kind == "order" and self._again_ok(st["bar"]), False)
        name, args, _k = self._move(otool, "order") if retry else self._move(dtool, "decline")
        resp = s.link.store_call(name, args)
        s.link.note("forced_finish", action={"name": name, "args": args}, reply=resp)

    def _move(self, tool, kind):
        return (tool, self.bare_args(tool, PROMPTS["order_line" if kind == "order" else "decline_line"], kind), kind)

    def _pick(self, first, st, otool, dtool):
        s = self.s
        lines, gone, mis = list(s.basket.units), set(), []
        if first and otool and lines:
            s.link.mute = True
            gone = self._guard("cart_gone", lambda: set(s.mender.dead_with_recheck(lines) or ()), set())
            if gone:
                st["bar"] |= gone
                st["pref"] = [l for l in lines if l not in gone]
                s.link.note("code_dead", lines=sorted(gone))
        if first:
            mis = list(s.warden.misfits() or [])
            for entry in mis:
                m = self.MISFIT_RE.match(str(entry))
                if m and m.group(1) in s.basket.units:
                    st["bar"].add(m.group(1))
        b = s.basket
        # a missing cart view counts as one that shows no readable lines
        if (first and otool and not gone and not mis and b.used and (b.units or not (
                b.last_view is not None and b.shown_units(b.last_view))) and (not s.multi() or all(
                i in {s.walker.product_slot(l) for l in b.units} for i in range(len(s.ledger.products()))))):
            return self._move(otool, "order")
        if otool and self.code_shop(st["bar"], st["pref"] or ()):
            return self._move(otool, "order")
        if otool and not first and self._again_ok(st["bar"]):
            return self._move(otool, "order")
        return self._move(dtool, "decline") if dtool else None

    def _again_ok(self, avoid):
        s = self.s
        bad = set(avoid or ()) | set(s.dead)
        return bool(s.basket.units) and not any(l in bad for l in s.basket.units) and bool(self.picks(bad))

    def _repair(self, held, prev, st):
        s = self.s
        looked, out = s.seeker.recheck_stock(list(held))
        if not looked:
            return None
        out = set(out or ())
        for l in held:
            if l not in out:
                s.refused_lines.discard(l)
        cost = sum((s.archive.price(l) or 0) * (q or 0) for l, q in held.items())
        over = kit.is_num(s.ledger.budget) and cost > s.ledger.budget + 1e-9
        if not out and not over:
            return prev
        st["bar"] = out
        st["pref"] = None if over else [l for l in held if l not in out]
        return None

    def bare_args(self, tool, phrase, kind):
        return self._guard("bare_args", self._plain, {}, tool, phrase, kind)

    def _plain(self, tool, phrase, kind):
        props, req = self._shape(tool)
        out = {}
        for name in req:
            p = props[name] if isinstance(props[name], dict) else {}
            enum, ts = p.get("enum"), self._types(p)
            if isinstance(enum, list) and enum:
                pick = None
                if kind in ("order", "decline"):
                    hit = kit.is_order_word if kind == "order" else kit.is_decline_word
                    pick = next((o for o in enum if hit(kit.low_plain(o))), None)
                    if pick is None and kind == "decline":
                        pick = next((o for o in enum if not kit.is_order_word(kit.low_plain(o))), None)
                out[name] = enum[0] if pick is None else pick
            elif "string" in ts:
                out[name] = self._cut(phrase, p)
            elif "boolean" in ts:
                out[name] = True
            elif "integer" in ts or "number" in ts:
                v = max(1, p["minimum"] if kit.is_num(p.get("minimum")) else 1)
                out[name] = int(v) if "integer" in ts and float(v).is_integer() else v
        if len(props) == 1:
            k, p = next(iter(props.items()))
            if k not in out and self._is_text(p):
                out[k] = self._cut(phrase, p)
        return out

    def clean_refusal(self, args, tool):
        return self._guard("clean_refusal", self._scrub, args, args, tool)

    def _scrub(self, args, tool):
        s = self.s
        if not isinstance(args, dict):
            return args
        rule = (s.ledger.reading if isinstance(s.ledger.reading, dict) else {}).get("decline_unnamed")
        if rule is False:
            return args
        known = s.archive.known
        named = any(isinstance(v, str) and any(t in known for t in self.TOKEN_RE.split(v)) for v in args.values())
        if rule is not True and not named:
            return args
        props, out = self._shape(tool)[0], dict(args)
        for k, v in args.items():
            p = props.get(k) if isinstance(props.get(k), dict) else {}
            out[k] = self._cut(PROMPTS["decline_line"], p) if isinstance(v, str) and not p.get("enum") else v
        return out

    def search_form(self):
        blank = dict(dict.fromkeys(("tool", "query", "category", "mode", "ceiling")), props={}, required=[],
                     clauses=[], free=[], modes=[], enums={})
        return self._guard("search_form", self._form, blank, blank)

    def _form(self, blank):
        s = self.s
        tool = s.role("search")
        if not tool:
            return blank
        props, req = self._shape(tool)
        clauses = [c.strip() for c in re.split(r"[.;]", kit.low(self._desc(tool))) if c.strip()]
        tok = s.seeker.cursor_param()
        free = [n for n, p in props.items() if self._is_text(p) and n != tok]
        query = next((n for n in free if self.QUERY_RE.search(n.lower())), free[0] if free else None)
        enums = {n: p["enum"] for n, p in props.items() if isinstance(p, dict) and isinstance(p.get("enum"), list)
                 and p["enum"]}
        category, most = None, 3
        for n, e in enums.items():
            if len(e) > most:
                category, most = n, len(e)
        modes = [n for n, e in enums.items() if n != category and len(e) <= 4]
        ceiling = None
        for n, p in props.items():
            low = n.lower()
            if (isinstance(p, dict) and not p.get("enum") and self._is_num(p) and not self.PAGING_RE.search(low)
                    and (self.CEIL_RE.search(low) or any(low in c and re.search("price|cost", c) for c in clauses))):
                ceiling = n
                break
        return {"tool": tool, "props": props, "required": req, "clauses": clauses, "free": free, "query": query,
                "category": category, "modes": modes, "mode": next((n for n in modes if n in req),
                                                                    modes[0] if modes else None),
                "ceiling": ceiling, "enums": enums}

    def _mode_value(self, form, key, is_cat):
        opts = form["enums"].get(form["mode"]) or []
        named = [c for c in form["clauses"] if re.search(r"\b%s\b" % re.escape(key.lower()), c)]
        for o in opts:
            if any(re.search(r"\b%s\b" % re.escape(kit.low_plain(o)), c) for c in named):
                return o
        return (opts[-1] if is_cat else opts[0]) if opts else None

    @staticmethod
    def _once(words):
        return list(dict.fromkeys(w for w in words if w))

    @staticmethod
    def _fit_len(text, mx):
        ws = text.split()
        while len(ws) > 1 and len(" ".join(ws)) > mx:
            ws.pop()
        return " ".join(ws)[:mx]

    def search_calls(self, slot):
        return self._guard("search_calls", self._calls, [], slot)

    def _calls(self, slot):
        s = self.s
        f = self.search_form()
        if not f["tool"]:
            return []
        props, schema = f["props"], s.schema(f["tool"])
        if s.multi() and slot is not None:
            row = s.ledger.products()[slot]
            name, qty = str(row.get("name") or ""), row.get("qty") or 1
        else:
            name, qty = str(s.ledger.product_name() or ""), s.ledger.quantity() or 1
        label = s.intake.best_label(kit.kind_terms(name) or kit.kind_terms(s.request)) or ""
        pw = [w for w in kit.words(name) if len(w) > 2 and not kit.is_link(w)]
        kind = pw[-1] if pw else ""
        solid = [v for _p, v in s.ledger.values_in_code(slot) if kit.words(v) and not self.SPLIT_RE.search(kit.low(v))
                 and kit.words(kit.strip_links(kit.low(v))) == kit.words(v)]
        t1 = t2 = ""
        if solid:
            vw = kit.words(solid[0])
            t1 = " ".join(self._once(vw + [kind]))
            if len(vw) > 1:
                t2 = " ".join(self._once([max(vw, key=len), kind]))
        else:
            t1 = " ".join(self._once([pw[0], pw[-1]]) if pw else sorted(kit.kind_terms(label))[:2])
        q, c = f["query"], f["ceiling"]
        mx = (props.get(q) or {}).get("maxLength") if q else None
        if type(mx) is int and mx >= 0:
            t1, t2 = self._fit_len(t1, mx), self._fit_len(t2, mx)
        ceil, b = None, s.ledger.budget
        if c and kit.is_num(b) and b > 0:
            ceil = int(b // qty) if "integer" in self._types(props.get(c)) else round(b / qty, 2)
        specs = [(q, t1, False)] if q and t1 else []
        if label and f["category"]:
            specs.append((f["category"], label, True))
        if q and t2:
            specs.append((q, t2, False))
        out = []
        for key, val, is_cat in specs:
            args = {key: val}
            if f["mode"]:
                args[f["mode"]] = self._mode_value(f, key, is_cat)
            if ceil is not None:
                args[c] = ceil
            for r in f["required"]:
                p = props[r] if isinstance(props[r], dict) else {}
                if r in args:
                    continue
                if isinstance(p.get("enum"), list) and p["enum"]:
                    args[r] = p["enum"][0]
                elif self._is_num(p):
                    args[r] = next((p[k] for k in ("default", "minimum") if kit.is_num(p.get(k))), 1)
                elif "string" in self._types(p):
                    args[r] = t1 or label
            fixed, faults = ArgFixer.fix(args, schema)
            if faults and ceil is not None:
                args.pop(c, None)
                fixed, faults = ArgFixer.fix(args, schema)
            if not faults and fixed not in out:
                out.append(fixed)
        return out

    def picks(self, avoid=(), keep=()):
        return self._guard("picks", self._picks, [], set(avoid or ()), set(keep or ()))

    def _picks(self, avoid, keep):
        s = self.s
        rows, vs, read = s.archive.digest(), str(s.ledger.version), s.archive.read
        failed = {k.split("|")[0] for k, v in s.rulings.items() if v and k.split("|")[1:2] == [vs]}
        passed = {k.rsplit("|", 1)[0] for k in s.passed if k.rsplit("|", 1)[-1] == vs}
        audited = set()
        entry = next((e for e in reversed(s.link.log) if isinstance(e, dict) and e.get("kind") == "audit"), None)
        if entry and isinstance(entry.get("result"), dict):
            seen = s.audit_seen
            if s.audit_done and (seen is None or s.audit_mark is None or len(read) <= s.audit_mark):
                seen = set(read)
            fits = entry["result"].get("maybe_fit")
            fits = {str(x) for x in fits if isinstance(x, (str, int))} if isinstance(fits, list) else set()
            audited = set(seen or ()) - fits - passed
        cap = s.ledger.budget if kit.is_num(s.ledger.budget) else None
        ctx = {"rows": rows, "read": read, "failed": failed, "passed": passed, "audited": audited,
               "own_all": bool(audited) and s.mender.rules_cover_all(), "cap": cap,
               "asked": kit.kind_terms(s.request), "hits": s.code_hits, "cheap": s.ledger.priority() == "cheapest",
               "avoid": avoid | set(s.dead) | set(s.tracer.chain_ids()), "keep": keep | set(s.warden.best_of_pair() or {})}
        multi, prods = s.multi(), s.ledger.products()
        homes = self._homes(multi, rows, read)
        pools = []
        for i, slot in enumerate(s.ledger.slots()):
            if multi:
                name, units = str(prods[slot].get("name") or ""), prods[slot].get("qty") or 1
            else:
                name, units = str(s.ledger.product_name() or ""), s.ledger.quantity() or 1
            pools.append((self._pool(slot, name, units, homes[i] if i < len(homes) else set(), ctx), units))
        if not pools or any(not p for p, _u in pools):
            return []
        got = self._choose(pools, read)
        if got is None or (cap is not None and got[1] > cap + 1e-9):
            inf = math.inf
            again = [(sorted(p, key=lambda e: (not e["fits"], cap is not None and e["price"] is None,
                                               inf if e["price"] is None else e["price"], e["id"])), u) for p, u in pools]
            got = self._choose(again, read)
            if got is None or (cap is not None and got[1] > cap + 1e-9):
                return []
        s.code_fit = {e["id"]: e["fits"] for e, _u in got[0]}
        return [(e["id"], self._whole(u)) for e, u in got[0]]

    def _homes(self, multi, rows, read):
        labels = [{kit.low(x) for x in (h or ())} for h in (self.s.judge.home_labels() or [])]
        if multi:
            return labels
        tally = {}
        for lid, r in rows.items():
            if lid in read and r.get("shelf"):
                tally[r["shelf"]] = tally.get(r["shelf"], 0) + 1
        if tally:
            return [{max(tally, key=tally.get)}]  # first-seen shelf wins a tie
        return [labels[0]] if labels and labels[0] else [set()]

    def _pool(self, slot, name, units, home, ctx):
        s = self.s
        vals = s.ledger.values_in_code(slot)
        judged = [(p, v) for p, v in vals if not s.judge.is_loose(v)]
        vwords = set().union(*(kit.kind_terms(v) for _p, v in vals))
        pwords = kit.kind_terms(name)
        rules = [(r.get("prop"), [r.get("value")] + list(r.get("partners") or []))
                 for r in (s.mender.settled_differs(slot) or []) if r.get("prop")]
        cap, inf, pool = ctx["cap"], math.inf, []
        for lid, row in ctx["rows"].items():
            if lid in ctx["failed"] or lid in s.refused_lines or lid in ctx["avoid"] or row.get("stock") is False:
                continue
            words = row.get("words") or set()
            if (row.get("shelf") not in home) if home else not (words & pwords):
                continue
            if any(kit.clash(row, p, v) for p, v in judged) or s.tracer.excluded(row.get("fields") or {}, slot):
                continue
            price = row.get("price") if kit.is_num(row.get("price")) else s.archive.price(lid)
            price = price if kit.is_num(price) else None
            if price is not None and cap is not None and units * price > cap + 1e-9:
                continue
            hits = ctx["hits"].get(lid)
            fits = all(kit.shows(row, p, v, hits) for p, v in judged) and kit.apart(row, rules)
            isread = lid in ctx["read"]
            ranks = [tuple(kit.value_rank(row, p, v)) for p, v in judged] if isread else []
            if lid in ctx["audited"] and not (ctx["own_all"] and fits and (judged or rules)
                                              and all(x <= 1 for r in ranks for x in r)):
                continue
            score = (3 * len(vwords & words) + 2 * len(pwords & words) + len(ctx["asked"] & words)
                     + 4 * len(vwords & set(hits or ())))
            rk = tuple(sum(r[j] for r in ranks) for j in range(len(ranks[0]))) if ranks else (9, 9)
            pk = inf if price is None else price
            key = (lid not in ctx["keep"], lid not in ctx["passed"], not fits, cap is not None and price is None,
                   pk if ctx["cheap"] else 0, not isread, row.get("stock") is not True, rk, -score, pk, lid)
            pool.append({"id": lid, "fits": fits, "price": price, "key": key})
        return sorted(pool, key=lambda e: e["key"])

    def _choose(self, pools, read):
        chosen, total = [], 0.0
        for pool, units in pools:
            pick = next((e for e in pool if all(e["id"] != c["id"] for c, _u in chosen)
                         and not (e["id"] in read and self._fails(e["id"]))), None)
            if pick is None:
                return None
            chosen.append((pick, units))
            total += (pick["price"] or 0) * units
        return chosen, total

    def _fails(self, lid):
        return self._guard("evidence_view", lambda: bool((self.s.judge.evidence_view(lid) or {}).get("fails")), False)

    def cart_form(self):
        blank = {"tool": None, "schema": {}, "props": {}, "puts": {}, "drops": {}, "id": None, "units": [],
                 "lines": None}
        return self._guard("cart_form", self._cart_form, blank)

    def _cart_form(self):
        s = self.s
        tool = s.role("cart")
        props, req = self._shape(tool)
        learned = self._guard("put_form", json.loads, {}, s.basket.put_form) if s.basket.put_form else {}
        learned = learned if isinstance(learned, dict) else {}
        puts, drops = {}, {}
        for r in req:
            enum = (props[r] or {}).get("enum") if isinstance(props[r], dict) else None
            if isinstance(enum, list) and enum:
                old = learned.get(r)
                puts[r], drops[r] = kit.cart_choices(enum, old if old in enum else None)
        free = [n for n, p in props.items() if self._is_text(p)]
        return {"tool": tool, "schema": s.schema(tool) if tool else {}, "props": props, "puts": puts, "drops": drops,
                "id": next((n for n in free if self.ID_RE.search(n.lower())), free[0] if free else None),
                "units": [n for n, p in props.items() if self._is_num(p)], "lines": s.basket.list_spec()}

    @staticmethod
    def _valid(calls, schema):
        got = [ArgFixer.fix(a, schema) for a in calls]
        return None if any(f for _v, f in got) else [v for v, _f in got]

    def put_calls(self, picks):
        return self._guard("put_calls", self._puts, None, picks)

    def _puts(self, picks):
        f = self.cart_form()
        if not f["tool"]:
            return None
        picks = [(l, 1 if u is None else self._whole(u)) for l, u in (picks or [])]
        if not picks:
            return []
        if f["lines"]:
            arg, idf, amf = f["lines"]
            calls = [dict(f["puts"], **{arg: [dict({idf: l}, **({amf: u} if amf else {})) for l, u in picks]})]
        elif not f["id"]:
            return None
        else:
            calls = [dict(f["puts"], **{f["id"]: l}, **({f["units"][0]: u} if f["units"] else {})) for l, u in picks]
        return self._valid(calls, f["schema"])

    def drop_calls(self, lines):
        return self._guard("drop_calls", self._drops, None, lines)

    def _drops(self, lines):
        s = self.s
        lines = list(lines or [])
        if not lines:
            return []
        f = self.cart_form()
        if f["lines"]:
            return []
        if not f["tool"] or not f["id"]:
            return None
        us = f["units"]
        if f["puts"]:
            if None in f["drops"].values():
                return None
            calls = [dict(f["drops"], **{f["id"]: l}, **({us[0]: self._whole(max(1, s.basket.held(l) or 0))}
                                                         if us else {})) for l in lines]
        else:
            lows = [(n, (f["props"].get(n) or {}).get("minimum")) for n in us]
            ua = next((n for n, lo in lows if not kit.is_num(lo) or lo <= 0), None)
            if not ua:
                return None
            calls = [{f["id"]: l, ua: 0} for l in lines]
        return self._valid(calls, f["schema"])

    def newest_replies(self, n=1):
        return self._guard("newest_replies", self._newest, [], n)

    def _newest(self, n):
        # as built (07 flag F1): only the newest store step is read, so a burst yields its last reply only
        e = next((e for e in reversed(self.s.link.log) if isinstance(e, dict) and e.get("kind") == "env"), None)
        r = e.get("reply") if e else None
        if not isinstance(r, dict):
            return []
        many = r.get("observations")
        if isinstance(many, list) and many:
            return many[:n]
        return [r["observation"]][:n] if "observation" in r else []

    def has_fault(self, obs):
        return self._guard("has_fault", self._fault_in, False, obs, 0)

    def _fault_in(self, v, depth):
        if isinstance(v, list):
            return any(self._fault_in(x, depth) for x in v)
        if not isinstance(v, dict) or depth > 1:
            return False
        if any(kit.is_fault_key(str(k)) and x and not (isinstance(x, str) and not x.strip()) for k, x in v.items()):
            return True
        return any(self._fault_in(x, depth + 1) for x in v.values() if isinstance(x, (dict, list)))



# ------------------------------------------------------------ p15_engine.py
class Session:
    """Errand state shared by every component, plus one instance of each component (registry 1)."""
    def __init__(self, problem):
        env = problem["environment"]
        self.binding = env.get("binding") or {}
        self.view = env.get("policy_view") or {}
        view = self.view
        self.request = str(view.get("query") or "")
        self.tools = [t for t in (view.get("tools") or []) if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.shapes = {}
        texts = []
        for t in self.tools:
            fn = t["function"]
            name = fn.get("name")
            if isinstance(name, str):
                par = fn.get("parameters")
                self.shapes[name] = par if isinstance(par, dict) else {"type": "object", "properties": {}}
            if isinstance(fn.get("description"), str):
                texts.append(fn["description"])
        self.opening = view.get("opening")
        try:
            cap = int(view.get("max_steps") or 0)
        except (TypeError, ValueError):
            cap = 0
        self.step_cap = cap or 30
        kit.reset()
        kit.feed_words(self.request + " " + " ".join(texts))
        # errand setup owned by Intake
        self.roles, self.msg_cap, self.read_fallback_at = {}, None, None
        # run counters and marks
        self.msgs_sent, self.msg_log, self.orders_tried, self.last_finish_kind = 0, [], 0, None
        self.refusals, self.cart_edits, self.absorb, self.since_cart = 0, 0, 0, 0
        self.cart_version, self.cart_shown_version, self.cart_seq_mark = 0, None, 0
        self.line_seq, self.put_tally, self.refusal_at_put = {}, {}, {}
        self.late_reads, self.lookup_window, self.basket_told_mark = 0, [], None
        self.scripted, self.batch_puts, self.batch_has_cart = False, set(), False
        self.finish_note, self.stuck, self.chat, self.covering = "", 0, [], 0
        # checks and verdicts
        self.passed, self.pass_price, self.rulings, self.rank_marks = set(), {}, {}, set()
        self.named_leads, self.named_pages = [], set()
        self.audit_done, self.audits, self.audit_fits, self.audit_seen, self.audit_mark = False, 0, [], None, None
        self.order_vetted, self.decline_vetted = False, False
        self.near_ok, self.near_words, self.code_fit, self.code_hits = set(), {}, {}, {}
        # stock, cart lines, events
        self.dead, self.refused_lines, self.event_seen, self.stock_seen = {}, set(), False, False
        self.rows_lag, self.rows_live, self.swap_read_done, self.batch_proven = False, False, False, False
        # search knowledge
        self.word_mode, self.lead_words, self.lead_frame = "", 0, None
        self.row_weight, self.hit_weight, self.all_hits, self.hit_shelf = {}, {}, [], {}
        self.ref_pointers, self.early_walk = [], False
        # asking
        self.asking, self.question_log, self.spent_props, self.ask_queue, self.prepared = True, [], set(), [], {}
        self.open_pairs, self.quiet_props, self.plan_empty, self.spare_asks = [], [], 0, []
        self.quota_wait, self.drafts, self.asking_label = False, [], ""
        # references
        self.refs = []
        # components, in registry order
        self.link = Link(self)
        self.archive = Archive(self)
        self.ledger = Ledger(self)
        self.basket = Basket(self)
        self.intake = Intake(self)
        self.judge = Judge(self)
        self.mender = Mender(self)
        self.asker = Asker(self)
        self.seeker = Seeker(self)
        self.walker = Walker(self)
        self.tracer = Tracer(self)
        self.pricer = Pricer(self)
        self.warden = Warden(self)
        self.closer = Closer(self)
        self.engine = Engine(self)
        self.link.done = bool(view.get("finished"))

    def role(self, name):
        tool = self.roles.get(name) if isinstance(self.roles, dict) else None
        return tool if isinstance(tool, str) and tool in self.shapes else None

    def schema(self, tool):
        sch = self.shapes.get(tool) if isinstance(tool, str) else None
        return sch if isinstance(sch, dict) else {"type": "object", "properties": {}}

    def spec(self, tool):
        for t in self.tools:
            if t["function"].get("name") == tool:
                return t
        return None

    def steps_left(self):
        return self.step_cap - self.link.steps

    def multi(self):
        return len(self.ledger.products() or []) >= 2

    def trouble(self):
        return bool(self.event_seen) or any(lid in self.dead for lid in self.basket.units)

    def pass_key(self, lid):
        return f"{lid}|{self.ledger.version}"

    def passed_now(self, lid):
        return self.pass_key(lid) in self.passed


class Engine:
    """Main loop, the one door to the store (batches), reply booking, and chat writing (12 all; 06 3.20)."""
    STALE = Tune.Marks.NOT_RUN + ": an earlier cart change in this same reply was held, so this cart call is out of date. Read that reason, then send the cart changes again."
    KEEP_LAST = Tune.Marks.NOT_RUN + ": only one step remains after the calls that ran, and it is kept for the order or the decline, sent alone."
    OVER = Tune.Marks.NOT_RUN + ": sending is over (step budget gone or episode ended)."
    NO_ANSWER = "No answer came back for this call (network or server fault); it can be sent again."
    CLOSED = Tune.Marks.NOT_RUN + ": the store closed while earlier calls of this step were answered and never reached this one."
    IDLE_NUDGE = "That reply called no tool. A turn without a tool call does nothing; pick one of the tools you were given and carry on."
    IDLE_TEXT = "(no tool was called)"
    KEEP = ("Keep {lid} in the cart unless something later rules it out: an independent check has already passed this listing "
            "against every condition now in force, judged by the value-matching rules (when the store gives a property its own "
            "field, only the words in that field can satisfy it).")
    NO_FIT = "No checked fit yet. "
    EARLY_MIN = 7

    def __init__(self, s):
        self.s = s
        self._turns = self._idle = self._fails = self._missed = self._seq = self._made = 0
        self._good = monotonic()
        self._early = False
        self._told = 0

    # ------------------------------------------------------------ small helpers
    @staticmethod
    def _fn(p):
        f = p.get("function") if isinstance(p, dict) else None
        return f if isinstance(f, dict) else {}

    def _pname(self, p):
        return self._fn(p).get("name")

    def _pargs(self, p):
        a = kit.args_from(self._fn(p).get("arguments"))
        return a if isinstance(a, dict) else {}

    @staticmethod
    def _plain(v):
        try:
            return json.loads(json.dumps(v, ensure_ascii=False, default=lambda o: sorted(o, key=str) if isinstance(o, (set, frozenset)) else str(o)))
        except Exception:
            return str(v)[:2000]

    @staticmethod
    def _said(args):
        return " ".join(v for v in args.values() if isinstance(v, str))

    @staticmethod
    def _named(args):
        out = set()
        for v in args.values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and kit.looks_like_id(x):
                    out.add(x)
        return out

    @staticmethod
    def _is_error(obs):
        return isinstance(obs, dict) and bool(obs.get("error"))

    def _fault(self, kind, err, **extra):
        self.s.link.note(kind, error=f"{type(err).__name__}: {err}"[:300], **extra)

    # ------------------------------------------------------------ proposals
    def call(self, name, args, cid=None):
        if cid is None:
            self._made += 1
            cid = f"code_{self._made}"
        text = args if isinstance(args, str) else json.dumps(args if args is not None else {}, ensure_ascii=False)
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": text}}

    def _normalise(self, raw):
        out = []
        for c in raw if isinstance(raw, list) else []:
            if not isinstance(c, dict):
                continue
            self._seq += 1
            fn = c.get("function") if isinstance(c.get("function"), dict) else {}
            a = fn.get("arguments")
            if not isinstance(a, str):
                a = json.dumps(a if a is not None else {}, ensure_ascii=False)
            out.append({"id": c.get("id") or f"call_{self._seq}", "type": "function", "function": {"name": fn.get("name"), "arguments": a}})
        return out

    # ------------------------------------------------------------ chat (6.6, 06 3.20)
    def send_replies(self, pairs, status=True):
        s = self.s
        pairs = list(pairs or [])
        for i, (cid, text) in enumerate(pairs):
            body = text if isinstance(text, str) else str(text)
            if status and i == len(pairs) - 1:
                body = body + "\n" + s.intake.state_banner()
            s.chat.append({"role": "tool", "tool_call_id": cid, "content": body})

    def dispatch(self, say, proposals):
        if not proposals:
            return []
        self.s.chat.append({"role": "assistant", "content": say or None, "tool_calls": list(proposals)})
        return self.run_batch(list(proposals), True)

    def script(self, say, proposals, status=True):
        pairs = self.dispatch(say, proposals)
        self.send_replies(pairs, status)
        return pairs

    def append_note(self, text, status=False):
        s = self.s
        text = text or ""
        if status:
            text = (text + "\n" if text else "") + s.intake.state_banner()
        if not text:
            return
        last = s.chat[-1] if s.chat else None
        if isinstance(last, dict) and last.get("role") in ("tool", "user") and isinstance(last.get("content"), str):
            last["content"] = last["content"] + "\n" + text
        else:
            s.chat.append({"role": "user", "content": text})

    def ledger_mark(self):
        s = self.s
        return (s.ledger.version, s.ledger.duties_version, s.msg_cap, s.step_cap, s.refusals, len(s.basket.units))

    # ------------------------------------------------------------ main loop (12 3.3)
    def run(self):
        s = self.s
        if s.link.done:
            return s.link.log
        s.intake.parse_errand()
        s.link.note("plan", reading=self._plain(s.ledger.reading), roles=self._plain(s.roles))
        parts = ["SHOPPER REQUEST:\n" + s.request]
        if s.opening is not None:
            parts.append("OPENING OBSERVATION:\n" + kit.render(s.opening))
            s.archive.add(s.opening)
        notes = s.ledger.notes_text()
        if notes:
            parts.append(notes)
        parts.append(s.intake.state_banner())
        s.chat = [{"role": "system", "content": PROMPTS["actor"]}, {"role": "user", "content": "\n\n".join(parts)}]
        self._good = monotonic()
        while self._turn():
            pass
        if not s.link.done and not s.link.broke:
            s.closer.force_finish()
        s.link.note("end", finished=bool(s.link.done), steps=s.link.steps, cost=round(s.link.spend, 5),
                    seconds=round(monotonic() - s.link.start, 1))
        return s.link.log

    def _turn(self):
        """One pass: guard, code step, actor. Returns False to close."""
        s = self.s
        if self._turns >= Tune.Loop.TURNS:
            return False
        self._turns += 1
        if s.link.done or s.link.broke or s.steps_left() <= 0 or s.link.time_left() < Tune.Clock.LAST_CALL:
            return False
        s.seeker.make_up()
        phrase = s.ledger.reply_due
        if phrase:
            s.ledger.reply_due = ""
            if s.asker.answer_shopper(phrase):
                return True
        if self._code_moves():
            return True
        models, cap = LINEUPS["actor_turned"] if s.stuck >= Tune.Loop.STUCK else LINEUPS["actor"]
        msg = s.link.chat(s.chat, models, cap, tools=s.tools)
        if not isinstance(msg, dict):
            self._missed += 1
            if self._gone(models):
                return False
            if s.link.fail_kind not in ("empty", "refused"):
                return True
            self._fails += 1
            return self._fails < Tune.Loop.FAILS
        self._good = monotonic()
        self._missed = 0
        calls = self._normalise(msg.get("tool_calls"))
        content = msg.get("content")
        # none counts as empty text (the idle placeholder and "no content" both need it)
        text = content if isinstance(content, str) else ("" if content is None else json.dumps(content, ensure_ascii=False))
        if s.asking and calls:
            binding = {s.role("cart"), s.role("order"), s.role("decline")} - {None}
            if any(self._pname(p) in binding for p in calls) and s.asker.ask(urgent=True):
                return True  # the actor's reply is dropped without a trace; it re-decides with the answer in view
        if not calls:
            self._idle += 1
            s.chat.append({"role": "assistant", "content": text or self.IDLE_TEXT})
            s.chat.append({"role": "user", "content": self.IDLE_NUDGE + "\n" + s.intake.state_banner()})
            return self._idle < Tune.Loop.IDLE
        self._idle = 0
        s.link.note("model", text=text[:2000], calls=[dict(self._fn(p)) for p in calls])
        s.chat.append({"role": "assistant", "content": text or None, "tool_calls": calls})
        before = s.link.steps
        pairs = self.run_batch(calls, False)
        self.send_replies(pairs, status=True)
        s.stuck = 0 if s.link.steps != before else s.stuck + 1
        return True

    def _code_moves(self):
        s = self.s
        moves = (("open_queued", None, lambda: s.judge.open_queued()),
                 ("match_query", None, lambda: s.warden.match_query()),
                 ("broad_turn", None, lambda: s.seeker.broad_turn()),
                 ("scan_turn", lambda: not s.asking or s.plan_empty > 0, lambda: s.seeker.scan_turn()),
                 ("value_sheets", None, lambda: s.pricer.value_sheets()),
                 ("ask", lambda: s.asking, lambda: s.asker.ask(urgent=False)),
                 ("tracer_step", None, lambda: s.tracer.step()),
                 ("disclosure_push", None, lambda: s.warden.disclosure_push()),
                 ("cap_push", None, lambda: s.warden.cap_push()),
                 ("single_push", None, lambda: s.warden.single_push()),
                 ("sparse_push", None, lambda: s.warden.sparse_push()),
                 ("tour_turn", None, lambda: s.pricer.tour_turn()))
        for name, guard, move in moves:
            if guard is not None and not guard():
                continue
            try:
                if move():
                    return True
            except Exception as e:  # hardening over 12 3.8 (moves unguarded there)
                self._fault("move_error", e, move=name)
        return False

    def _gone(self, models):
        s = self.s
        if not s.link.usable(models, tools=True):
            return True
        if monotonic() - self._good <= Tune.Clock.DOWN_AFTER:
            return False
        return self._missed > 1 or (s.link.service_down() and len(s.link.tried) > 1)

    # ------------------------------------------------------------ batch path (12 3.5)
    def run_batch(self, proposals, scripted):
        s = self.s
        proposals = list(proposals or [])
        s.scripted = scripted
        s.intake.batch_start()
        menu = s.intake.offered_names()
        cart, msg_tool = s.role("cart"), s.role("message")
        s.batch_has_cart = bool(cart) and any(self._pname(p) == cart for p in proposals)
        puts = set()
        for p in proposals:
            if cart and self._pname(p) == cart:
                a = self._pargs(p)
                if s.basket.is_cart_call(a):
                    t = s.basket.target(a)
                    if t:
                        puts.add(t)
        s.batch_puts = puts
        replies, queue, held_cart, queued_msgs = [], [], False, 0
        for p in proposals:
            cid = p.get("id")
            if held_cart and cart and self._pname(p) == cart:
                replies.append((cid, self.STALE))
                continue
            hold, name, args = s.warden.screen(p, menu, scripted, len(proposals), queued_msgs)
            if name is None:
                name = self._pname(p)
            if hold:
                if cart and name == cart and not held_cart:
                    a = args if isinstance(args, dict) else self._pargs(p)
                    if a and not s.basket.is_cart_call(a):
                        held_cart = True
                replies.append((cid, hold))
                continue
            if msg_tool and name == msg_tool:
                queued_msgs += 1
            queue.append([cid, name, args if isinstance(args, dict) else {}])
        fin = s.intake.finishing_tools()
        if queue and queue[0][1] in fin:
            cid, name, args = queue[0]
            hold, sent_args = s.warden.finish_screen(cid, name, args)
            if hold:
                replies.append((cid, hold))  # no trim, send or label: a finish note stays for the next batch
                return self._in_order(replies, proposals)
            if isinstance(sent_args, dict):
                queue[0][2] = sent_args
        elif fin and queue:
            allowed = max(0, s.steps_left() - 1)
            for q in queue[allowed:]:
                replies.append((q[0], self.KEEP_LAST))
            queue = queue[:allowed]
        group = Tune.Loop.GROUP
        for g in range(0, len(queue), group):
            chunk = queue[g:g + group]
            if s.link.done or s.link.broke or s.steps_left() <= 0:
                for q in chunk:
                    replies.append((q[0], self.OVER))
                continue
            # as built (12 9.1): calls past the steps-left cut get no reply at all
            for q in chunk[:max(0, s.steps_left())]:
                replies.append((q[0], self._send_one(q[1], q[2])))
        note, s.finish_note = s.finish_note, ""
        if note and queue and queue[0][1] in fin:
            for i, (cid, text) in enumerate(replies):
                if cid == queue[0][0]:
                    replies[i] = (cid, note + "\n" + text)
                    break
        return self._in_order(replies, proposals)

    @staticmethod
    def _in_order(replies, proposals):
        pos = {}
        for i, p in enumerate(proposals):
            pos.setdefault(p.get("id"), i)
        return sorted(replies, key=lambda r: pos.get(r[0], -1))

    def _send_one(self, name, args):
        s = self.s
        step = s.link.steps + 1
        resp = s.link.store_call(name, args)
        s.link.note("env", step=step, actions=[{"name": name, "args": dict(args)}], reply=resp)
        if not isinstance(resp, dict):
            return self.NO_ANSWER
        obs = resp.get("observations")
        if not isinstance(obs, list):
            obs = [resp.get("observation")]
        if not obs:
            return self.CLOSED
        return self.book(name, args, obs[0], resp, last=len(obs) == 1)

    def send_booked(self, proposals):
        s = self.s
        out = []
        for p in proposals or []:
            cid, name, args = p.get("id"), self._pname(p), self._pargs(p)
            if s.link.done or s.link.broke or s.steps_left() <= 0:
                out.append((cid, self.OVER))
            else:
                out.append((cid, self._send_one(name, args)))
        return out

    # ------------------------------------------------------------ booking (12 3.7)
    def book(self, name, args, raw, resp, last=True):
        s = self.s
        args = args if isinstance(args, dict) else {}
        resp = resp if isinstance(resp, dict) else {}
        obs = kit.lens(raw)
        rem = []
        msg_tool, cart, view_tool = s.role("message"), s.role("cart"), s.role("cart_view")
        search, details = s.role("search"), s.role("details")
        is_msg, is_cart = bool(msg_tool) and name == msg_tool, bool(cart) and name == cart
        # stage 1: counts and learning
        said = ""
        if is_msg:
            s.msgs_sent += 1
            said = self._said(args)
            s.msg_log.append((s.link.steps, said[:600], s.cart_edits))
            if s.warden.states_basket(said):
                s.basket_told_mark = s.warden.cart_mark()
            if not s.scripted:
                back = raw if isinstance(raw, str) else kit.render(raw)
                s.question_log.append('- (the assistant\'s own message): "%s" -> %s' % (said[:300], back[:300]))
        if is_cart:
            s.cart_edits += 1
        kind = s.intake.finish_kind(name, args)
        s.last_finish_kind = kind
        if kind == "order":
            s.orders_tried += 1
        if is_cart or is_msg or kind:
            s.lookup_window = []
        else:
            s.lookup_window = s.lookup_window[-1:] + [str(name) + kit.render(args)]
        s.archive.add(obs)
        s.absorb += 1
        if not is_cart and s.covering <= 0:
            s.since_cart += 1
        try:
            dead_new = list(s.judge.learn(name, args, obs) or [])
        except Exception as e:
            self._fault("learn_error", e)
            dead_new = []
        try:
            s.judge.fix_details_role()
        except Exception as e:
            self._fault("details_role_error", e)
        # stage 2: cart-line freshness
        if s.basket.units:
            self._freshen(name, args, obs, cart, view_tool)
        # stage 3: reads, late reads, stock and own-page remarks
        if details and name == details and not self._is_error(obs):
            for v in args.values():
                if isinstance(v, str) and v in s.archive.known:
                    s.archive.read.add(v)
        if self._late(name, args, details, search):
            s.late_reads += 1
        if dead_new:
            t = s.judge.soldout_note(dead_new)
            if t:
                rem.append("[stock] " + t)
        try:
            t = s.tracer.redirect_remark(name, args)
            if t:
                rem.append("[notes] " + t)
        except Exception as e:
            self._fault("redirect_error", e)
        # stage 4: cart booking
        if is_cart:
            s.order_vetted = False
            s.basket.changed_step = s.link.steps
            before = dict(s.basket.units)
            rem.append(s.basket.book(args, obs) or "")
            rem.append(s.warden.units_hint(args))
            try:
                s.mender.basket_shift(before)
                s.mender.stamp_current(obs)
                s.mender.post_cart(args, obs)
            except Exception as e:
                self._fault("cart_follow_error", e)
            tgt = s.basket.target(args)
            if tgt and s.basket.is_cart_call(args):
                s.refusal_at_put[tgt] = s.refusals
                s.put_tally[tgt] = s.put_tally.get(tgt, 0) + 1
                n = s.basket.count_of(args)
                if s.passed_now(tgt) and (n is None or n > 0) and (s.basket.held(tgt) or 0) > 0 and tgt not in s.dead:
                    rem.append(self.KEEP.replace("{lid}", str(tgt)))
        # stage 5: cart view and refusals
        finished = bool(resp.get("finished")) if last else False
        if view_tool and name == view_tool:
            s.basket.view(obs, s.link.steps)
            rem.append(s.basket.summary() or "")
        refused = s.seeker.is_refused(name, obs, finished)
        if refused and kind == "order":
            s.cart_seq_mark = s.absorb
        rem.append(s.seeker.refusal_note(name, obs, refused) or "")
        # stage 6: search notes
        if search and name == search:
            rem.append(s.pricer.query_memo(args, obs) or "")
            s.named_pages.discard(json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            try:
                for _ in range(2):
                    t = s.judge.fuse_linked(obs)
                    if not t:
                        break
                    rem.append(t)
            except Exception as e:
                self._fault("merge_error", e)
        # stage 7: early decline audit (once)
        if last and not self._early and not is_cart and not is_msg and not kind:
            left = s.steps_left()
            ver = str(s.ledger.version)
            if (self.EARLY_MIN < left <= s.step_cap // 3 + 2 and not s.basket.units
                    and not any(str(m).rsplit("|", 1)[-1] == ver for m in s.passed)):
                self._early = True
                t = s.judge.audit(early=True)
                if t:
                    rem.append(self.NO_FIT + t)
        # stage 8: shopper lines
        self._hear(name, obs, refused, said if is_msg else None, rem)
        clip = Tune.Text.SEARCH_CLIP if search and name == search else Tune.Text.RECORD_CLIP
        shown = kit.render(kit.clip_deep(raw, clip), Tune.Text.VIEW)
        if last and resp.get("finished"):
            shown += " " + Tune.Marks.OVER + " nothing more can be sent."
        joined = " ".join(r for r in rem if r)
        return shown + "\n" + Tune.Marks.NOTE + " " + joined if joined else shown

    def _freshen(self, name, args, obs, cart, view_tool):
        s = self.s
        units = s.basket.units
        known = s.stock_seen
        named = self._named(args)
        lookup_only = not (s.rows_live and not s.rows_lag) and name != cart and name != view_tool
        fresh = set()

        def look(d):
            hits = [(k, v) for k, v in d.items() if isinstance(v, str) and v in units]
            if not hits or not any("id" in kit.low(k) for k, _ in hits):
                return
            ids = {v for _, v in hits}
            if known:
                if kit.stock_of(d) is None or (lookup_only and not ids & named):
                    return
            elif not any(isinstance(v, dict) for v in d.values()):
                return
            fresh.update(ids)
        kit.visit(obs, look, top=6)
        for lid in fresh:
            s.line_seq[lid] = s.absorb

    def _late(self, name, args, details, search):
        s = self.s
        if not details or s.covering > 0:
            return False
        if name == search:
            kind = "search"
        elif name == details:
            kind = "details"
        elif name in {(self._fn(t).get("name") or t.get("name")) if isinstance(t, dict) else t for t in (s.judge.query_tools() or [])}:
            kind = "lookup"
        else:
            return False
        if not s.steps_left() < s.intake.due_margin():
            return False
        vals = [v for v in args.values() if isinstance(v, str)]
        if kind == "details":
            return not any(v in s.basket.units for v in vals)
        if kind == "lookup":
            return not any(v in s.basket.units or s.passed_now(v) for v in vals)
        return True

    def _hear(self, name, obs, refused, question, rem):
        s = self.s
        lines = s.warden.buyer_lines(name, obs, refused)
        if not lines:
            return
        ver, drops, cap = s.ledger.version, len(s.ledger.dropped or []), s.ledger.budget
        rem.append(s.ledger.hear(lines, question) or "")
        s.tracer.take_news()
        news = s.ledger.limits_news
        if news:
            try:
                s.intake.apply_limits(news)
            except Exception as e:
                self._fault("limits_error", e)
            s.ledger.limits_news = None
        if s.ledger.duties_version != self._told:
            self._told = s.ledger.duties_version
            s.order_vetted = False
        if not s.ledger.version > ver:
            return
        if question and len(s.ledger.dropped or []) <= drops and s.ledger.budget == cap:
            return
        try:
            rem.append(s.warden.shift_review() or "")
        except Exception as e:
            self._fault("review_error", e)


def agent_main(problem):
    s = Session(problem)
    try:
        return s.engine.run()
    except Exception as e:
        s.link.note("crash", error=f"{type(e).__name__}: {e}"[:500])
        if not s.link.done and not s.link.broke:
            try:
                s.closer.force_finish()
            except Exception as e2:
                s.link.note("crash", error=f"{type(e2).__name__}: {e2}"[:500])
        return s.link.log
