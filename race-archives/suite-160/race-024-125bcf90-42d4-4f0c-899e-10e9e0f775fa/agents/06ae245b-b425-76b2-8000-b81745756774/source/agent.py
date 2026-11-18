"""11-hello: one shopping errand per call. A tool-calling chat model shops; code frames every move it makes."""
from __future__ import annotations

import json
import re
from time import monotonic, sleep
from typing import Any

from src.agent.proxy_client import ProxyClient


# ------------------------------------------------------------------------------------------------------------ settings
class Knobs:
    """Every number the errand runs by, grouped by what it limits."""

    # time
    BUDGET_S = 265.0            # the whole errand ends before this many seconds
    WIND_DOWN_S = 70.0          # with less left, the errand heads for its finish (no new shopper readings)
    FINAL_S = 28.0              # with less left, code sends the finish itself
    CLOCK_ROUND = 30            # time left, once shown on the status line, in steps of this many seconds
    # text shown to the shopping model
    VIEW_MAX = 16000            # longest observation text
    ROW_FIELD_MAX = 240         # longest field text on a search page
    RECORD_FIELD_MAX = 700      # longest field text in other observations
    # money spent on models (scaled by the product count)
    SPEND_SOFT = 0.03           # after this, a dear model gives way to the cheaper one after it
    SPEND_HARD = 0.045          # after this, the errand heads for its finish
    # the loop
    ROUNDS = 80                 # turns of the main loop at most
    SILENT_TURNS = 3            # replies without a tool call before code finishes
    STALLED_TURNS = 4           # replies in a row that reached no store step before the backup model leads
    # traffic
    GAP_S = 0.15                # least time between two requests, model and store alike
    GAP_SLOW_S = 1.0            # the gap while slowed after a 429 / 503
    SLOW_HOLD_S = 30.0          # how long the slow gap lasts
    OUTAGE_S = 45.0             # models count as down after this long without an answer (with failures since)
    TRIES_PER_MODEL = 2         # attempts per model in one call
    STORE_TRIES = 5             # attempts of one store call after load or transport failures
    # finishing and the steps kept for it
    CODE_FINISHES = 4           # finishing calls code sends at most (a refused order repaired and sent again)
    LATE_STEPS = 7              # after a refusal, from this many steps left only cart, view and finishing tools
    ORDER_KEEP = 10             # steps kept for the first order and one repair, at most
    # code-driven search and reading
    MORE_PAGES = 3              # further result pages code reads of each product's best search
    READ_ROUNDS = 8             # rounds of record reads of the code's picks
    SHELF_SLICES = 8            # shelf slices code reads when a change leaves the whole shelf acceptable
    ID_WALK_EACH = 6            # cheapest rule, id-ordered store: further pages per product
    ID_WALK_ALL = 8             # ... in all
    VALUE_PAGES_ONCE = 4        # an id-ordered all-words search led by a required value: further pages at once
    VALUE_PAGES_ALL = 8         # ... in all
    VALUE_PAGES_EACH = 5        # several products: at least this many value pages per product in all
    LABEL_SEARCHES = 4          # composed value + category-label searches per product
    LABEL_ROWS = 3              # rows opened from a one-page all-values composed search (no cheapest rule)
    CAP_CALLS = 4               # price-filter listing calls of one product's ceiling walk
    CAP_ROWS = 3                # rows opened in that walk (a pick and two backups)
    # questions to the shopper
    ASK_ASSUMED = 3             # questions allowed when the request names no message budget
    ASK_AFTER_READS = 2         # listing records read before the first planned question
    ASK_WAIT_STEPS = 3          # extra steps the first question may wait for those reads (more than 20 steps left)
    ASK_AFTER_STEPS = 5         # or the first question goes once this many steps are used
    # value matching
    SYNONYM_READS = 4           # read listings of a category carrying a field before a synonym may stand for a value
    # references to other listings
    REF_STEP_SHARE = 0.4        # code's reference reads and searches: at most this share of the steps (at least 8)
    REF_MODEL_CALLS = 12        # reference-hop model calls per errand
    REF_HOPS = 8                # listings followed along one reference's chain
    REF_DIFFER_SHARE = 0.3      # only "differ" references open: reference steps stop at this share
    REF_DIFFER_BLIND = 5        # ... and at this many while none of their chains was found


LINEUP = dict(
    actor=["anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "anthropic/claude-haiku-5.5"],
    reader=["anthropic/claude-sonnet-5.5", "openai/gpt-6-luna"],
    checker=["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"],
    auditor=["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"],
    listener=["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"],
    planner=["anthropic/claude-haiku-5.5", "anthropic/claude-sonnet-5.5"],
)


# --------------------------------------------------------------------------------------------------------- the texts
READING = """How to read a listing against a requirement:
1. The listing's own field for the property decides, read by the value's words as this store writes them there. A field
   whose name means the same property counts however it is spelt (separators, a prefix, a shorter or longer name, a
   common alternative name such as "fit" for a pants fit type, "neckline" for a collar type, "color family" for a
   colour). A field carrying the property's own name decides, and so does that name with one more word in front of it
   ("pants length" for length, "clothing material" for material) - if either writes the value, the requirement is met.
   Names that differ only by separators ("size class", "size_class") are one field; either one writing the value is
   enough. Otherwise a field with an alternative name decides. Only when no field names the property may the options
   list or the title show the value. The description never counts as evidence.
2. Values match by words; case, separators, singular or plural and spelling variants do not matter. A more specific value
   that contains the required one meets it ("80% cotton" meets cotton, "forest green" meets green, "rtx 3060 12gb" meets
   rtx 3060). In a list one element is enough. A synonym, abbreviation, related word or shade meets the value ONLY when
   no listing of the category writes the value itself in that field (tech wood / technical wood, win11 / windows 11,
   embroidered / embroidery, solid / plain, ivory or cream / off-white): when other listings write "matte" as their
   finish, a "satin" finish does not meet matte. A value joined by a plus to another ("wool + nylon") does not meet one of
   its parts when the field also carries that other part alone ("nylon"). FIELD EVIDENCE lines, when given, are worked
   out from the store's records and say which case holds. A different value of the property never meets it (quartz is
   not mechanical, rosewood is not maple), nor does a word that only looks alike. A value claiming every value of the
   property ("suits all occasions", "all seasons", "universal") names none of them and meets no required value - a list
   that also names the required value still meets it. A field naming another value does not meet the required one just
   because the two often go together (pullover is not zip-up, slip-on is not lace-up); only a shade usually sold as the
   required colour, or a more specific value containing it, counts.
3. A field of another property never counts (lining material is not material, length is not style, strap length is not
   strap type) and never stands in for a property whose own field the category's listings carry.
4. Numbers compare as amounts after unit conversion (4.5 cm is 45 mm); limits include the edge; a range counts only when
   all of it meets the limit. When no field names the property and the title is the evidence, an exact required value is
   met when the title shows its words and figures in the same order, even with more after them (9' x 12' 4" shows 9' x
   12'; 9' x 13' does not).
5. A listing's price is in the budget's unit unless its record names another currency or unit (a price of 45 against a
   budget of 1000 US cents is 45 cents, well inside it). Never rescale the price or the budget.
6. Only the listed requirements count: never fail a listing for an age group, gender, size, occasion, condition, season,
   quality or use the requirements do not name.
7. The requested kind is met by the category label alone: a listing in the requested category is of that kind even when
   its title names another object (a container, case, refill, set or accessory). Judge it only on the requirements,
   the price and the stock."""

ACTOR_RULES = """You shop for one shopper in a simulated store, and the tools given are your only way to act. The errand scores 1 or 0, and it scores 1 only when the finish is right in every way: the ordered item meets every requirement the shopper wrote, every hidden need the shopper reveals when asked and every change announced along the way; the quantity and the budget are right; the cheapest (or best) fitting item is chosen when the request asks for that; and every rule of procedure in the request is kept. When nothing in the store meets everything, the right finish is a decline, done exactly as the request says.

THE STORE
- Every tool call costs one step - failed and invalid calls too. Steps are limited; thinking costs nothing.
- One reply may carry several tool calls when none needs another's result (opening several listings, say); they run in order. An order or a decline always goes alone.
- A details tool that takes a LIST of ids shows all their full records for ONE step, up to its item limit - use it whenever you open two or more listings.
- Tool names, argument names and result shapes differ from store to store: read each tool's description and schema and use exactly the names and allowed values given. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Learn early which category labels the store puts on listings of the requested kind; where neighbouring labels exist, the right one matches the request's wording as a whole. Search and buy inside it.
- Learn the search from its first results: rows may come best match first or in a fixed id order; a keyword search may need every word (an extra word shrinks the total) or take any word (an extra word grows the total, and rows matching more words rank higher); only a few rows or pages may be visible; a filter may work in one mode only (keyword mode may ignore the category or price filter, filter mode the words).
- Compare the reported total with the rows you can see: when it is larger, the fitting or cheapest listing may sit on a later page or past the visible ones. When every word is required, narrow with more exact words, or with category and price filters and a lower price limit, until the total fits the visible pages.
- Keep queries short - one to three words, the most distinctive first; some stores read only the first word or two. Compare a longer query's total and rows with a shorter one that starts the same way; the code's notes say what they show.
- When every word is required, a required value followed by the item kind, value first ("blue chair"), lists exactly the listings that carry it.
- When any word is enough, common words (the kind, a unit) match most of the shelf and bury the rest: search a required value's own distinctive word or figure alone ("herringbone", "30m", "0.4"), and in the other spellings the listings use ("30 m", "3 atm"); read every page of such a search when its total fits them; join the kind and values only to rank the closest first.
- A keyword search may also match category names and property values. Code notes after each search say what the totals so far show about this store's search.

THE SHOPPER AND HIDDEN NEEDS
- The shopper holds needs nobody wrote down and reveals one only when a message asks about that property. The first thing a message asks is answered most reliably, so a message asks about one property - two at most, the likelier first - as a short, plain, friendly question naming it; one or two example values are fine. No instructions, lists, labels, positions or rules in it.
- "Don't mind", "no preference" or "anything is fine" leaves the property free; a stated preference or limit is a hard requirement from then on, exactly as worded, numbers included.
- When questions are planned for you, they go out once you have opened the details of two or three listings of the requested kind - so open those first.
- Answers arrive with their meaning already explained: apply each revealed need as a hard requirement and search again with it and the stated values together.
- If you ask on your own: first open two or three listings' details, then ask about what the story hints at (who it is for, how it must look, where or how it is used), then about properties the listings carry that differ among candidates. Never ask what the request already settles, and never go past the message budget.

RULES OF PROCEDURE
- The request may set rules for the work - how many listings to look at or compare, what to check, what to tell the shopper, what a remark must say, how to decline. They are graded like requirements: keep them literally and fully. "Look at" or "open" N listings means N fully fitting listings, each opened with the details tool.
- When the shopper asks you something (to confirm a change, say), the code answers while messages remain; the change itself binds.
- Wishes ("ideally", "would be nice", "if possible") are not requirements: they only break ties among items that already meet everything, and never beat a cheapest rule.
- When the request leaves something only the shopper can settle (no price limit, two wishes that may clash where the shopper must say which matters more), ask before choosing.

CHANGES ALONG THE WAY
- The shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price rise, an item selling out. One line can carry several changes, and such lines can come inside any tool result (an extra field holding sentences) or with an order attempt. Each change binds from that moment: check the pick and the backups again, and when the cheapest or best is asked for, choose again over the new acceptable set.
- An order attempt can be REFUSED; the episode goes on and the cart is unchanged. Read the refusal, apply every change it states, take out of the cart whatever no longer fits (a sold-out item, say), put in the next verified fit, check its stock and price again and order again.
- Items can also sell out or change price without notice, so check stock and price right before every order attempt. An item whose current price times the quantity is over the current budget no longer fits.

SEVERAL PRODUCTS
- A request may ask for several different products, each with its own quantity, requirements and hidden needs. Work on every one: search each one's category, verify a fit for each, and order them together - one line per product at its quantity.
- A cart tool that takes a list of lines may replace the whole cart with that list, so every product's line goes into each such call.
- A cheapest or best rule applies within each product.
- A requirement may give its value by pointing at another listing or page: find that listing first by searching the words that name it (its shop or brand with its kind, say), read its record and take the value from the named field, then search the requested product with that value.

THE CART
- The cart tool may ADD units to a line it already holds instead of setting them: putting 1 more of a held line makes 2. Never put a held line in again; to change its units, take the difference out or in. A code note after each cart call says what the code counts in the cart.

JUDGING A LISTING
""" + READING.replace("such as \"fit\" for a pants fit type,", "such as \"fit\" for a pants fit type, \"dial size\" for a watch dial size,").split("\n", 1)[1] + """
- Only what the shopper stated or revealed counts. Never rule a listing out for something not asked for that you guess from the story or the description (age group, gender, size, occasion, condition, quality, use) - the story is context; if it seems to matter, ask the shopper about that property instead. Your own questions are not requirements; only the shopper's answers are.
- Figures in a title ("55 x 94 Inch", "117 x 228 cm") do not say which dimension each one is: never rule a listing out, or skip opening it, for figures in its title - read the fields in its details.
- Keep exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- No property has to repeat the category name. When the store labels categories, the item sits in the category whose label matches the request's wording as a whole: a request for "A or B" matches the category "A & B", not the narrower neighbour "B".
- The item must be in stock when the request wants things on hand; its price times the quantity must be within the budget (equal is fine).
- A decline is rarely right. When no listing meets the strict wording but one meets each requirement under these rules (a more specific value, another name of the same field, a synonym or shade where no listing of the category writes the value itself), that listing is the fit. Decline only when the search really found nothing meeting every requirement under these rules.

CHEAPEST OR BEST
- Cheapest: the order must be the lowest-priced item among ALL items meeting every requirement - one fitting item is not enough.
  * When the keyword search needs every word and matches property values too, the value-and-kind search lists the candidates carrying the value: read all its pages and open them by price.
  * A category listing with a price limit shows the whole category's prices but not its values: use it to see which cheaper rows exist, lowering the limit until the total fits, then check those rows' properties.
  * Search each required value with the item kind as a short query, value first ("blue chair", "30m watch"); read every page when the total fits the readable pages, and when it does not, add a second required value or use a narrower kind word. Do the same with the kind's other names ("stool", "bench") and with the value alone.
  * When a value cannot be searched (a figure the store drops, say), also search the product line of each fit or near-fit found - its maker plus one distinctive title word, two words, every page: a line's other sizes and colours are separate listings that share its values and may cost less.
  * Sort the visible candidates by price yourself and open details from the cheapest up: the first whose fields meet everything, with every cheaper one opened and turned down, is the answer. Keep the next fitting ones as backups.
- Best by a measure (most, largest, longest, highest...): compare ALL fitting items on that measure the same way and take the extreme; an item that states no value for the measure cannot win.

FINISHING
- Keep a ranked list of verified fits with at least one backup - the first pick may sell out at checkout.
- Before every order attempt: the cart holds exactly one line per product asked for (one line for a single product), each the chosen item at its required quantity, and nothing else; and after the last cart change the chosen item's stock and price were read again and it is in stock and within the budget.
- Decline only after a thorough search finds nothing meeting every requirement, and keep the request's decline rule exactly. If the request says not to offer substitutes, the decline remark names, describes or points to no listing - no ids, titles or prices.
- Budget the steps: the first order attempt goes out while at least seven steps remain, so a refusal can still be handled; with 7 steps left only the cart, stock checks and finishing tools remain; decide once and do not swap the cart back and forth.
- Every reply carries at least one tool call. Keep the text short: the requirements now, the verified fits with price and stock, and the next move."""

READER_TEXT = """Read a shopper's request and a store's tool list. Answer with ONE compact JSON object (no indentation) and nothing else, with exactly these keys:
- "product": the kind of product asked for, in the request's words (several products joined with "and").
- "items": one entry per DIFFERENT product to buy, in the request's order, each {"product": its kind in the request's words, "quantity": as below, "requirements": as below}; one entry when one product is asked for.
- "quantity": integer units of one listing to order - 1 unless the request plainly asks for several units ("I need three of them", "order 2 packs"). Something sold as one item (a pair of shoes, a set) is 1; how many people will use or share it, or how many old ones there were, is story and leaves it at 1.
- "requirements": a list of strings, one per firm requirement, each "property: constraint" carrying the shopper's exact value, limit, exclusion or closed list; with several products each is "property (product): constraint". Leave out price, stock, quantity, budgets and mere wishes. Do not restate the product kind or category: words belonging to the category's own name (a season, occasion or audience in it) are covered by the category. A requirement is a property of the product with the value the request says it must have ("the pattern has to be herringbone", "with a mechanical movement", "at least 4 drawers", "the occasion has to be party"). Story is never a requirement - whom it is for, an event it will be worn or used at, where, when or how it is used, how many share it, replacing an old one ("a new watch" asks for a replacement, not a condition). An occasion, use or audience is a requirement only when the request makes it the item's own property ("the occasion has to be business", "a blazer made for business"). A requirement's value must be words or figures the request itself writes as the item's value; a quality, reason or wish given in the story (natural and breathable, slim enough for a case, works with what I already have, covers the whole drop) is a hint, never a requirement.
- "price_cap": the highest total price as the number the request writes, in its own unit, never converted ("3050 US cents" gives 3050); null when none.
- "currency": the request's price unit, or null.
- "in_stock_only": true when the request wants only items on hand or in stock.
- "priority": "cheapest" only when the request plainly asks for the cheapest or lowest-priced fitting item; "best" only when it plainly asks for the best by some measure; else "none". Saying money is tight or every penny counts is NOT a cheapest request - it only explains the cap.
- "priority_detail": for "best", the measure; else "".
- "message_budget": how many messages the shopper allows, as an integer, or null.
- "step_budget": how many moves or steps the request allows, as an integer, or null.
- "rules": every procedural rule the request sets for the assistant, word for word (how many listings to look at, what to check or tell, how to decline, what not to do, what to include); what the shopper will do themselves is story, not a rule.
- "soft_wants": what the shopper would like without strictly needing it ("ideally", "preferably", "would be nice", "if possible") - kept out of requirements.
- "ask_first": what only the shopper can tell and should be asked before choosing - the budget when no limit is given; which of two wishes matters more when the request says both may not be possible (both wishes then go in soft_wants, not requirements); anything the request says to check with the shopper. Empty when none.
- "hints": story details that point at a product property without stating its value - what it is used for or with, where, by whom, what it must fit, match or survive. Each is "property (product): the detail in a few words" ("property: detail" for one product), the property named as listings would likely name it; when a detail may concern more than one property, the likeliest first and the next after a slash ("colour / style (chair): must suit a room with one painted wall"). What the item must match, suit or sit well with (a room, a wall, an outfit, things used with it), and how it looks, point first to colour, then to pattern or style. Leave out a detail that only explains a stated requirement. Weather, a job, a schedule, travel, or a place where the item is not worn or used is background, never a hint; so is a purpose every listing of the kind serves (storage for a dresser, warmth for a coat). Empty when none.
Every sentence of the request lands somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or plain story; when unsure whether a sentence is a rule, put it in rules.
- "references": requirements whose value is given only by pointing at ANOTHER listing or page, each "property (product): as shown on <how the request names that listing>", or "property (product): different from the one shown on <...>" when the value must differ; each also goes into requirements in that form. Empty when none.
- "reference_targets": for each reference, the listing it points at: {"property", "product", "seller": the shop or brand given for it, exactly as written, "kind": its product kind, "details": each other value the request names for it, short and in the request's words (a colour, "4gb", a net weight with its figure), "relation": "same" | "differ" (must NOT be that listing's value: set apart from, different from, other than) | "within" | "at most" | "at least", "tolerance": the number for within else null, "unit": its unit else ""}. Empty when none.
- "decline_no_names": true when the request's decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks to name something; null when it gives no decline rule.
- "roles": an object mapping each role to the exact tool name from the list, or null when no tool fits: "search", "details", "view", "message" (sends the shopper a message), "cart" (puts things in or takes them out of the cart), "cart_view", "order" (buys or checks out), "decline" (ends without buying)."""

LISTENER_TEXT = """Help a shopping assistant understand what the shopper just said, and keep the list of requirements current. Answer with ONE JSON object and nothing else, holding:
- "meaning": one short sentence on what the shopper means.
- "needs": new firm requirements, each "property: constraint" with the exact value, limit, exclusion or closed list. After a comparison phrase ("falls short of", "nothing less than", "at least"), a value made of words rather than a figure IS the value, written as the shopper writes it: "falls short of not available" gives "property: not available", because "not available" and "none" can be a field's own values.
- "no_preference": properties the shopper said they do not care about.
- "dropped": earlier requirements the shopper no longer needs.
- "new_budget": the new highest total price as a number in the unit of the request's budget (the figure as the line writes it when it uses that unit - 13075 for "13075 cents" with a cents budget), or null.
- "unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
- "meta_only": true when the line only asks the assistant to ask less at once (one question at a time, fewer things at once, only the first now) and states no value - such a line answers nothing yet: nothing goes in needs, no_preference or dropped, and its figurative words name no value. Else false. A line that only says the shopper did not follow or understand, or asks for the question again in simpler words, is NOT meta_only.
- "requirements_now": the full current list of firm requirements, each "property: constraint": start from REQUIREMENTS SO FAR, keep each unless dropped or replaced (for a replaced value keep only the newest), and add each new need from the shopper's lines. Never add a requirement from the request's story that the list did not hold - only the shopper's lines add requirements. No price, stock, quantity or category.
- "reply": when the shopper asks the assistant something or asks for a confirmation, a short plain answer to send back that names the change in substance (confirming the new value, say); else "".
- "references": for each need whose value is given only by pointing at another listing, {"property", "seller": as written, "kind", "details": each other value the line names for that listing, short and in the line's words, "relation": "same" | "differ" | "within" | "at most" | "at least", "tolerance": for within else null, "unit": else ""}; [] when none.
Reading guide:
1. One line can carry several changes - report each.
2. A need on another property, even a related one, replaces nothing (a seat height beside an overall height, a strap width beside a width); an earlier requirement is dropped or replaced only when the line says so about that same property.
3. A replaced requirement shows up twice: the old one under dropped, the new one under needs.
4. When the shopper says which of two wishes matters more, the preferred one becomes a need. A stated budget goes into new_budget. A mere wish is not a need.
5. Read the shopper literally and kindly: folksy, figurative or roundabout words that name a value mean they firmly want that value; a value is excluded only when they clearly say to avoid it.
6. Emphasis words are not part of the value (only, nothing but, nothing short of, strictly, stays, sticking with, locked to, holding out for): write just the value words ("material: jute"), never "<value> only". Add an exclusion only when the shopper also plainly rules out mixes or other values.
7. A number with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit. A new budget replaces the old one, higher or lower, even when the shopper calls it good news.
8. When the shopper only says they don't mind, have no preference or anything works, the property asked about goes into no_preference; when the message asked about two properties, only the FIRST goes in.
9. A reply that only says the shopper didn't follow or understand ("that was rather dense", "you've lost me", "could you put it another way") is this shopper's way of saying no preference when the question named the product the property belongs to: the asked property goes into no_preference. When several products are asked for and the question did not say which, the property stays open and nothing goes into no_preference.
10. With several products every need is "property (product): constraint", the product taken from the line (a line about the toe style of the women's boots gives "toe style (women's boots): ...").
11. The assistant's own question is never a requirement; values it named as examples are not needs.
12. A "don't mind" frees only the property the assistant's message asked about; it never drops or changes a need stated earlier, and when the message asked nothing (it only told something), such an answer changes nothing.
13. A need given by pointing at another listing keeps the line's own pointing words as its value ("finish: unlike the one shown on Y's kettle page"); the values the line names for that other listing describe that listing, not the product - they are not needs, drops or freed properties."""

CHECK_TEXT = """You are the last check before a shopping assistant commits to an item; the errand scores 1 only when the item meets every requirement below. Judge it ONLY from the records shown, by these rules:
""" + READING + """
Also: words belonging to the category's own name are met by the category; keep exclusions and closed lists; the item must be the requested kind, and when listings carry a category label it must be the label matching the request's wording as a whole, not a narrower or neighbouring one; its price times the quantity must be within the budget (equal is fine); if the records show it out of stock it fails; a wish marked as not required is not a requirement; when records disagree (price, stock), the newest counts.
Answer with ONE JSON object and nothing else: {"item": the item id, "fits": true or false, "problems": [short strings, each naming a requirement not met or not shown]}. "fits" is false only when a listed requirement is clearly failed by the item's fields under the rules, or when neither a field nor (where no field for it exists) the options list or the title states it; a FIELD EVIDENCE line saying the item does NOT meet a requirement counts as a clear failure. Never fail an item because a stock or price record is missing."""

RANK_TEXT = """A shopping assistant is about to order an item for a request that asks for the PRIORITY. Below are the requirements, the chosen item and the listing records seen (DETAILS: the full record was read; ROW: seen only in search rows, its fields unknown). Name the listings that may meet every requirement AND beat the chosen item on that priority - a lower price for the cheapest, a better value of the measure for the best. Leave out listings the records show failing a requirement, out of stock or over the budget. Figures in a ROW's title (such as 55 x 94) do not say which dimension each one is, so they never rule a ROW out.
Reading rules: price times quantity within the budget; not shown out of stock;
""" + READING + """
Answer with ONE JSON object and nothing else: {"better": [listing ids, best first, at most 4], "reason": "one sentence"}."""

PLAN_TEXT = """Plan the questions a shopping assistant sends to a shopper. The shopper holds hidden needs: each is a value of one property of the product, revealed only when a message asks about that exact property; the first thing a message asks is answered most reliably, and questions about anything else get a polite "don't mind". Choose up to QUESTION_COUNT different properties, most promising first:
0. First, anything in MUST BE LEARNED FROM THE SHOPPER not yet asked, asked plainly - for the budget, the most they want to spend; for a choice between two wishes, both values with their properties: "if one listing can't have both, would you rather have <value> <property> or <value> <property>?".
1. Then properties the listings' details actually carry. Read the story first - STORY HINTS, when given, come first: the hidden need is usually the property the story's concrete details concern without naming its value. What the user will touch or press, see up close or in photos, hear, measure or fit against, read on the label, or how it was made points at that part or aspect and its property; tags, labels and fine print point to material, fabric or ingredients; photos and looks to pattern, colour or the visible part; a body part or a space it must fit to that part's shape or size; buying for someone else, or someone barely known, to whom it is made for (gender, age group); how it must feel to the hand, skin or foot first to material or fabric; where, when or for what activity it is worn or used (a gym class, the office, outdoors) to style or type - the property whose values name such uses; something done up or adjusted again and again to the closure or mechanism; what does the work inside it to its core component. Weather, a job, a schedule or a place where the item is not worn or used is background. Ask first about the property the story points at, by the listings' name for it. "Whom it is made for" is a candidate like any other: ask it when the story points at someone else, or when nothing else is hinted. After that, prefer properties of what the product is like (pattern, material, shape, type or style variant, closure, length, function) over logistics (warranty, condition, pack size, variant or size lists, years).
Never pick a property the request already settles (its requirements, price, stock, quantity); never repeat a property already asked; never spend two questions on near-duplicates (two kinds of style, two spellings of one property); never ask about brand, maker, product or model name, price, stock or delivery. Ask only about a property that appears as a key in the listings' details, by the listings' name for it - a story-hint property matching no key may be asked by its plain name, unless the hint is marked [no key] (enough of that product's listings were read without such a field): then ask about the key whose values differ the way the hint's detail describes, or leave it. Each step-1 question is one short, plain, friendly sentence of at most 15 words asking which <property> the shopper would like for the <item kind>, the property named by the listings' own name for it (example values alone never reveal a need), optionally followed by one or two example values; no lead-in, status, instructions or lists. Never use a catch-all key (options, variants, description, title, name). Each question asks about ONE property of ONE product (the step-0 choice between two wishes excepted): a reply reveals at most one need, and a question naming two subjects may get no answer at all. "second" is the next most promising property of the SAME product (not another name for the first), or "" - the assistant may add it to its last message.
Several products: hidden needs may sit on any of them; spread the questions so every product gets one before any gets a second (a product with no property left is set aside); take each product's properties from its own category's key list; write each property as "property (product)" and name the product in the question ("Which toe style would you like for the boots?").
Answer with ONE JSON object and nothing else: {"questions": [{"property": "...", "second": "...", "question": "..."}]}."""

ONE_AT_A_TIME = """
This time only ONE question goes out, and its answer is read before the next is chosen. REMAINING questions are left in all - use every one. Put the single most promising property first; one or two spares may follow. After a revealed need, the next question goes to a different property and a different story detail."""

AUDIT_TEXT = """A shopping assistant is about to tell the shopper that nothing in the store fits - that is rarely the right finish. Check the listing records below against the requirements by these reading rules (the price must be within the budget and the item must not be shown out of stock):
""" + READING + """
Some listings are known only from search rows (ROW), so their property fields are unknown: such a row counts as not yet checked when it is in the requested category (whatever object its title names) and its title, price and stock do not show it failing a requirement - figures in a title (such as 55 x 94) never show a failure. List such rows under "unopened", likeliest first: a row whose title names a required value first, else one of the requested kind with nothing against it. Judge the listings OPENED SINCE THE LAST CHECK first, field by field.
Answer with ONE JSON object and nothing else: {"possible_fits": [ids that seem to meet every requirement, best first, at most 5], "unopened": [row-only ids to open before declining, at most 6], "reason": "one sentence"}."""

HOP_TEXT = """A shopper's request takes the PROPERTY of the product it asks for from another listing in the store, described as: DESCRIPTION. Below are the records of listings that may be that listing. Answer with ONE JSON object and nothing else, holding:
- "target": the id of the record that IS the described listing - sold by that seller, of that kind, and showing every named detail (details are read by meaning: a figure given in another unit counts when it is the same amount); null when none of the records is it.
- "missing": the described details the target does not show ([] when it shows all).
- "go_to": when the target's own text sends the reader to a DIFFERENT listing for the PROPERTY (take it from, use, refer to, look at or go by another listing): {"seller": as written, "kind", "details": each value the text names for that listing, short and as written - a colour, a size with its unit, a material}; else null. A listing named only as what the target is NOT ("not the green one from X") is not a go_to; nor is a version, size or colour sold elsewhere, nor a listing named for another property.
- "value": when there is no go_to, the target's own value of the PROPERTY as its record writes it.
- "goes_with": values the target's text says go together with, pair with or count the same as that value ([] when none).
A record that does not show the described details is not the target, whatever its text says."""


def listener_prompt(request: str, so_far: str, exchanges: list, question: str | None, lines: list) -> str:
    parts = [LISTENER_TEXT, "THE REQUEST (for context):\n" + request, "REQUIREMENTS SO FAR:\n" + so_far]
    if exchanges:
        parts.append("EARLIER EXCHANGES:\n" + "\n".join(exchanges))
    if question:
        parts.append("THE ASSISTANT'S QUESTION:\n" + question)
    parts.append("THE SHOPPER'S NEW LINES:\n" + "\n".join(lines))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------------------------- word sets
class WordPattern:
    """A class of words written as a regex alternation; `in` tests one whole word, and it mixes with plain sets."""

    def __init__(self, alternatives: str):
        self._whole = re.compile("(?:%s)" % alternatives)
        self.pattern = alternatives

    def __contains__(self, word: Any) -> bool:
        return isinstance(word, str) and self._whole.fullmatch(word) is not None

    def __iter__(self):
        return iter(self.pattern.split("|"))

    def __and__(self, other: Any) -> set:
        return {w for w in other if w in self}

    __rand__ = __and__

    def __or__(self, other: Any) -> set:
        return set(self) | set(other)

    __ror__ = __or__


PART_JOINS = r"\s+and\s+|\s*&\s*|\s*/\s*|\s+\+\s+"          # how a store joins two kinds in one category name
ADD_SENSE = WordPattern("add|put|hold|set|insert|place|keep|include|in|plus|increase|append")
REMOVE_SENSE = WordPattern("remove|drop|release|delete|take|out|clear|discard|minus|decrease|subtract|off|unhold")
FAULT_MARK = r"error|fault|refus|reject|denied|invalid|fail"
FILLER = WordPattern("and|or|the|for|with|of|a|an|in|on|to|by|from|at|as|is|are|be|this|that|it|its|your|our|you|all|any|not|no")
APOLOGY = ("I'm sorry - nothing in this store meets every one of your requirements, so I am not placing an order.")
CODE_SHAPE = re.compile(r"[A-Za-z0-9_\-]+")
DECLINE_SENSE = re.compile(r"refus|declin|cancel|abandon|reject|give.?up|abstain|withdraw|no.?(?:purchase|order|buy)|nothing|without")
SKIP_SMALL = {"and", "the", "a", "an", "of"}
ORDER_SENSE = re.compile(r"commit|buy|order|purchas|check.?out|pay|confirm|complete|place|accept|proceed")
ID_SHAPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}")
LIMIT_NEGATION = re.compile(r"\b(?:no|not)\s+(?:more|less|fewer|over|under|above|below|beyond|exceeding)\b")
LIMIT_WORDS = re.compile(r"\b(?:at least|at most|no more than|no less than|more than|less than|up to|or more|or less|"
                         r"or larger|or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|"
                         r"below|than)\b")
COMPARE_LEAD = re.compile(r"\b(?:falls?\s+short\s+of|short\s+of|nothing\s+less\s+than|nothing\s+other\s+than|"
                          r"nothing\s+short\s+of|nothing\s+but|no\s+less\s+than|no\s+worse\s+than|at\s+least|"
                          r"less\s+than|more\s+than|better\s+than|worse\s+than|exactly|only|just)\b")
SAID_VALUES: set = set()           # word values the shopper gave after a comparison and the store writes (per errand)
EXCLUSION = re.compile(r"\bno\b(?!-)|\b(?:not|without|nothing|never|neither|nor|except|excluding)\b|\bavoid\w*|"
                       r"\b(?:anything but|other than|rather than|instead of)\b")
LINK_PHRASES = (r"at least|at most|no more than|no less than|more than|less than|up to|or more|or less|or larger|"
                r"or smaller|or higher|or lower|minimum|maximum|between|within|under|over|above|below|exactly|only|"
                r"either|neither|nor|than|plus")
POINTING = re.compile(r"\b(?:as shown|shown on|same as|page|sold by|redirect\w*|refer(?:s|red|ring)?|listing points|"
                      r"points you)\b")
DIFFERING = re.compile(r"\b(?:set apart from|apart from|distinct from|different (?:from|to|than)|"
                       r"(?:differ|differs|different|differing)\s+from|other than|unlike|not the same as|anything but)\b")
POINT_OPEN = re.compile(r"\b(?:within|at most|at least|no more than|no less than|same as|as shown|the value|the one)\b")
SELLER_MARK = re.compile(r"shop|seller|store|vendor|brand|make|manufactur|merchant|retailer|supplier|sold.?by")
_YES_NO = re.compile(r"\W*(?:yes|no|true|false|none|required|included|needed)\W*")
_YES_NO_STORY = re.compile(r"\W*(?:yes|no|true|false|required|included|needed)\W*")

# compiled once: the patterns the helpers below share
_C_TIGHT = (",", ":")
_C_FENCE = re.compile(r"^```[A-Za-z]*")
_C_PAREN = re.compile(r"\s*\(")
_C_CAMEL = re.compile(r"([a-z])([A-Z])")
_C_ALNUM = re.compile(r"[a-z0-9]+")
_C_LETTERS = re.compile(r"[a-z]+")
_C_WORD3 = re.compile(r"[a-z]{3,}")
_C_DIGIT = re.compile(r"\d")
_C_FIGURE = re.compile(r"\d+(?:\.\d+)?")
_C_FIG_RUN = re.compile(r"\d+(?:\.\d+)?(?:\s*[:x/]\s*\d+(?:\.\d+)?)*")
_C_VALUE_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_C_MARKS = re.compile(r"\(stated\)|\(revealed[^)]*\)")
_C_EMPHASIS = re.compile(r"\b(?:only|just|exactly|strictly)\b")
_C_OR = re.compile(r"\bor\b")
_C_LIST_SEP = re.compile(r"[,;|]")
_C_PLUS_SEP = re.compile(r"\s*\+\s*")
_C_UNIT_AFTER = re.compile(r"\s*([a-z]*)")
_C_LINKS = re.compile(r"\b(?:%s)\b" % LINK_PHRASES)
_C_NOT_ER_THAN = re.compile(r"\b(?:no|not)\s+[a-z]+er(?=\s+than\b)")
_C_NOT_ER = re.compile(r"\b(?:no|not)\s+[a-z]+er\b")
_C_CLAUSE_SEP = re.compile(r"[,;/&()]|\bor\b|\band\b")
_C_OPEN_PAREN = re.compile(r"\(")
_C_POSSESSIVE = re.compile(r"'s\b")
_C_NON_ALNUM = re.compile(r"[^a-z0-9]")
_C_TAG_END = re.compile(r"\(([^()]*)\)\s*$")
_C_LEAD = re.compile(r"\s*([a-z](?:[a-z\-]*[a-z])?)(?=\s*[,;.()]|\s*$|\s+(?:%s)\b)" % "|".join(sorted(FILLER)))
_C_REQ_TOKEN = re.compile(r"[a-z][a-z'\-]*")
_C_FIG_UNIT = re.compile(r"(\d)(?:\s+|\s*-\s*)([a-z]{1,8})\b")
_C_REF_TOKEN = re.compile(r"\d+(?:[.,:/]\d+)*[a-z]*|[a-z][a-z0-9]*")
_C_SIBILANT_ES = re.compile(r"(?:ch|sh|x|z)es$")
_C_FIG_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*-?\s*([a-z]{1,8})\b")
_C_WITHIN = re.compile(r"\bwithin\s+(\d+(?:\.\d+)?)\s*([a-z%]*)")
_C_AT_MOST = re.compile(r"\b(?:at most|no more than|up to|or less|or lower|or smaller)\b")
_C_AT_LEAST = re.compile(r"\b(?:at least|no less than|or more|or higher|or larger)\b")
_C_OR_ALT = re.compile(r"\(\s*or\b")
_C_CAP_NAME = re.compile(r"(?<![A-Za-z'])([A-Z][a-z]{3,})('s)?(?![A-Za-z])")
_C_ARTICLE = re.compile(r"\b(?:the|for|a|an)(?=\s)", re.I)
_C_SOLD_BY_TAIL = re.compile(r"\s+(.*?)\s+(?:sold|made|offered|listed)\s+by\s*$", re.I | re.S)
_C_KIND_LEAD = re.compile(r"\s*([^,;(]*?)(?=\s+(?:with|that|which|in|at|having|from|of|for|sold)\b|[,;]|$)", re.I)
_C_DETAIL_SEP = re.compile(r"[,;]|\b(?:and|with|which|in|at)\b|\bthat\s+(?:is|are|has|have|comes?)\b(?:\s+as\b)?", re.I)
_C_DETAIL_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.%:/&'\-]*")
_C_SENTENCE_CUT = re.compile(r"\(\s*or\b|(?<=[.!?])\s")
_C_PRICE_LIKE = re.compile(r"amount|fee|charge|pay|rate|value")
_C_NOT_PRICE = re.compile(r"total|count|qty|quant|unit|due|sum|rating|star|score|page|number|num|size|weight|"
                          r"width|height|length|found|match")
_C_STOCK_LIKE = re.compile(r"hand|orderable|buyable|sellable|ready|ship|instock|purchas|in_store")
_C_TITLE_LIKE = re.compile(r"headline|name|label|caption")
_C_NOT_TITLE = re.compile(r"shop|seller|store|vendor|brand|make|categor|shelf|department|id")
_C_POINTER_BITS = ("as shown", "same as", "the page", "listing", "sold by", "redirect", "refer")


# ------------------------------------------------------------------------------------------------ text and JSON plumbing
def as_text(value: Any, ceiling: int = Knobs.VIEW_MAX) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=_C_TIGHT)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > ceiling:
        return text[:ceiling] + "...(cut)"
    return text


def clip(value: Any, n: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    deeper = depth + 1
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out[k] = clip(v, n, deeper)
        return out
    if isinstance(value, list):
        return list(clip(v, n, deeper) for v in value)
    if isinstance(value, str) and len(value) > n:
        return value[:n] + "..."
    return value


def _c_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return None


def first_json(text: Any) -> dict | None:
    if not isinstance(text, str):
        return None
    if (body := text.strip()).startswith("```"):
        body = _C_FENCE.sub("", body).strip()
        if body[-3:] == "```":
            body = body[:-3].strip()
    candidates = [body]
    open_at = body.find("{")
    if open_at >= 0:
        close_at = body.rfind("}")
        candidates.append(body[open_at:close_at + 1] if close_at > open_at else "")
    return next((parsed for parsed in map(_c_loads, candidates) if isinstance(parsed, dict)), None)


def numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def sentences_in(value: Any, out: list, depth: int = 0) -> None:
    """Sentence-like strings (4+ words) in a string or a list of strings."""
    if depth <= 4:
        if isinstance(value, list):
            for v in value[:10]:
                sentences_in(v, out, depth + 1)
        elif isinstance(value, str) and 4 <= len(value.split()):
            out.append(value)


def each_dict(value: Any, visit, ceiling: int = 6, depth: int = 0) -> None:
    if depth > ceiling:
        return
    if isinstance(value, dict):
        visit(value)
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return
    for child in children:
        each_dict(child, visit, ceiling, depth + 1)


# ------------------------------------------------------------------------------------------------ property names
_NOT_PROP_WORDS = WordPattern("brand|shop|seller|store|vendor|manufacturer|category|categories|department|title|headline|heading|listing|description|overview|price|cost|amount|availability|sku|url")
_NOT_PROP_WHOLE = WordPattern("name|note|notes|stock|in stock|on hand|image|images|link|links|make|maker|shelf|id|ids")


def _name_core(name: Any) -> str:
    head = _C_PAREN.split(str(name), 1)[0]
    return _C_CAMEL.sub(r"\1 \2", head).lower()


def not_a_property(name: Any) -> bool:
    words = _C_ALNUM.findall(_name_core(name))
    if any(w in _NOT_PROP_WORDS for w in words):
        return True
    return " ".join(words) in _NOT_PROP_WHOLE


def _c_singular(w: str) -> str:
    if len(w) <= 3 or w[-1:] != "s" or w[-2:] == "ss":
        return w
    return w[:-1]


def name_words(name: Any) -> list:
    return list(map(_c_singular, _C_LETTERS.findall(_name_core(name))))


def name_key(name: Any) -> str:
    return "".join(name_words(name))


def within_edits(a: str, b: str, most: int) -> bool:
    if not -most <= len(a) - len(b) <= most:
        return False
    width = len(b) + 1
    prev = [*range(width)]
    i = 0
    for ca in a:
        i += 1
        cur = [i] * width
        for j in range(1, width):
            swap = prev[j - 1] if ca == b[j - 1] else prev[j - 1] + 1
            cur[j] = min(swap, prev[j] + 1, cur[j - 1] + 1)
        if most < min(cur):
            return False
        prev = cur
    return most >= prev[-1]


def loosely_same(a: Any, b: Any) -> bool:
    lo, hi = sorted((name_key(a), name_key(b)), key=len)
    if not lo or lo == hi:
        return bool(lo)
    if len(lo) >= 4 and lo[:3] == hi[:3] and within_edits(lo, hi, 2 if len(lo) > 5 else 1):
        return True
    return len(lo) >= 5 and hi.startswith(lo)


def _soft_at(word: str, i: int) -> bool:
    c = word[i]
    if c in "aeiouy":
        return True
    return c in word[max(i - 1, 0):i] + word[i + 1:i + 2]


def _c_soft_drops(word: str) -> list:
    """Each spelling of the word with one soft letter (a vowel or a doubled letter) left out."""
    return [word[:i] + word[i + 1:] for i in range(len(word)) if _soft_at(word, i)]


def spelling_variant(lo: str, hi: str) -> bool:
    gap = len(hi) - len(lo)
    if gap == 0:
        diff = [i for i, (x, y) in enumerate(zip(lo, hi)) if x != y]
        if len(diff) != 2:
            return False
        first, second = diff
        return second == first + 1 and lo[first] == hi[second] and lo[second] == hi[first]
    if gap == 1:
        return lo in _c_soft_drops(hi)
    if gap == 2:
        return any(lo in _c_soft_drops(once) for once in _c_soft_drops(hi))
    return False


def strictly_same(a: Any, b: Any) -> bool:
    ka = name_key(a)
    kb = name_key(b)
    if not (ka and kb):
        return False
    if ka == kb:
        return True
    if len(kb) < len(ka):
        ka, kb = kb, ka
    if len(ka) >= 4 and ka[:3] == kb[:3] and spelling_variant(ka, kb):
        return True
    short, long_ = name_words(a), name_words(b)
    if len(long_) < len(short):
        short, long_ = long_, short
    if len(long_) != len(short) + 1 or len(long_[-1]) > 3:
        return False
    return long_[:-1] == short and len("".join(short)) >= 5


# ------------------------------------------------------------------------------------------------ value words
def _content_words(text: str) -> list:
    """Words of 3+ letters, link words left out."""
    return [w for w in _C_WORD3.findall(text) if w not in FILLER]


def value_words(text: Any) -> list:
    words = []
    if text:
        for t in _C_VALUE_TOKEN.findall(str(text).lower()):
            if t not in SKIP_SMALL:
                words.append(t)
    return words


def _c_near_word(word: str, p: str) -> bool:
    if p == word:
        return True
    if not (word.isalpha() and p.isalpha()):
        return False
    if same_kind(word, p):
        return True
    return min(len(word), len(p)) >= 6 and word[:2] == p[:2] and within_edits(word, p, 1)


def word_present(word: str, pool: list) -> bool:
    return any(_c_near_word(word, p) for p in pool)


def value_options(value: Any) -> list:
    low = _C_MARKS.sub(" ", str(value or "").lower())
    low = _C_EMPHASIS.sub(" ", low.replace('"', " "))
    options = []
    for part in _C_OR.split(low):
        words = value_words(part)
        if words:
            options.append(words)
    return options


def _c_spelled_by_run(pool, joined):
    """Some run of neighbouring pool words, written together, spells `joined`."""
    start = 0
    while start < len(pool):
        run = ""
        for p in pool[start:]:
            run += p
            if len(run) >= len(joined):
                break
        if run == joined:
            return True
        start += 1
    return False


def _c_option_written(opt, pool):
    if all(word_present(w, pool) for w in opt):
        return True
    joined = "".join(opt)
    if len(opt) < 2 and not _C_DIGIT.search(joined) and not any(_C_DIGIT.search(p) for p in pool):
        return False
    return _c_spelled_by_run(pool, joined)


def pool_has(options: list, pool: list) -> bool:
    return bool(pool) and any(_c_option_written(opt, pool) for opt in options)


def field_writes(value: Any, text: Any) -> bool:
    return pool_has(value_options(value), value_words(text))


def _c_figure_hits(fig: str, text: str):
    return re.finditer(r"(?<![\d.])%s(?!\d|\.\d)" % re.escape(fig), text)


def _c_shares_start(w: str, pool: list) -> bool:
    head = w[:3]
    return any(len(p) >= 3 and (p.startswith(head) or w.startswith(p[:3])) for p in pool)


def field_touches(value: Any, text: Any) -> bool:
    pool = value_words(text)
    low = str(text or "").lower()
    for w in (w for opt in value_options(value) for w in opt):
        if word_present(w, pool) or (len(w) >= 3 and _c_shares_start(w, pool)):
            return True
        if _C_FIGURE.fullmatch(w) and next(_c_figure_hits(w, low), None) is not None:
            return True
    return False


def list_parts(text: Any) -> list:
    low = str(text or "").strip().lower()
    parts = []
    for piece in _C_LIST_SEP.split(low):
        piece = piece.strip()
        if piece:
            parts.append(piece)
    if parts or not low:
        return parts
    return [low]


def plus_sides(element: str) -> list:
    sides = [s for s in map(str.strip, _C_PLUS_SEP.split(str(element))) if s]
    if len(sides) < 2:
        return []
    return sides


# ------------------------------------------------------------------------------------------------ value classes
def figure_in(fig: str, low_request: str) -> bool:
    hits = _c_figure_hits(fig, low_request)
    decimal = "." in fig
    for m in hits:
        if decimal:
            return True
        unit = _C_UNIT_AFTER.match(low_request, m.end()).group(1)
        if len(unit) <= 3:
            return True
    return False


def _c_bounds_off(low):
    """The value with its bound wording blanked: 'no more than', 'no thicker than', and with a figure 'no thicker'."""
    low = _C_NOT_ER_THAN.sub(" ", LIMIT_NEGATION.sub(" ", low))
    return _C_NOT_ER.sub(" ", low) if _C_DIGIT.search(low) else low


def names_the_item(value: Any, prop: Any = "") -> bool:
    if _YES_NO.fullmatch(low := str(value).lower().strip()):
        return True
    if SAID_VALUES and " ".join(_C_ALNUM.findall(_C_MARKS.sub(" ", low))) in SAID_VALUES:
        return True
    low = _c_bounds_off(low)
    if max(map(len, map(_content_words, _C_CLAUSE_SEP.split(_C_LINKS.sub(" ", low))))) >= 4:
        return False
    if (hit := EXCLUSION.search(low)) is None or _content_words(low[:hit.start()]):
        return True
    if not (after := _content_words(low[hit.end():])):
        return False
    own = set(name_words(prop))
    own.update(_C_LETTERS.findall(str(prop).lower()))
    return len(own) > 0 and not any(w not in own and w.rstrip("s") not in own for w in after)


def _c_run_pattern(run: str, tail: str) -> str:
    figs = _C_FIGURE.findall(run)
    return r"(?<![\d.])" + r"\D{0,4}".join(map(re.escape, figs)) + tail


def seen_in_samples(value: Any, samples: Any) -> bool:
    if not (low_s := str(samples or "").lower()).strip():
        return False
    body = _C_LINKS.sub(" ", str(value).lower())
    for w in _content_words(body):
        stem = w if len(w) <= 4 else w[:-1]
        if re.search(r"\b" + re.escape(stem), low_s):
            return True
    patterns = [_c_run_pattern(run, r"(?![\d.])") for run in _C_FIG_RUN.findall(body)]
    return any(re.search(p, low_s) for p in patterns)


def _c_stem(w: str) -> str:
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def _c_writes_word(s: str, w: str, stem: str) -> bool:
    """The request's word s writes w: itself, an inflection, a compound start or a 1-edit spelling."""
    if s == w or (s.startswith(stem) and len(s) - len(stem) <= 3):
        return True
    if stem.startswith(s) and len(s) >= max(4, len(stem) - 2):
        return True
    if len(w) < 4:
        return False
    return s.startswith(w) or (len(s) >= 4 and w[0] == s[0] and within_edits(w, s, 1))


def _c_written(w: str, rwords: list, rstems: list) -> bool:
    stem = _c_stem(w)
    if any(_c_writes_word(s, w, stem) for s in rwords):
        return True
    return any(len(s) >= max(3, len(stem) - 3) and stem.startswith(s) for s in rstems)


def unwritten_words(req: str, request: str, refs: list) -> list:
    """The value words (and figure runs) of a requirement the request text never writes; empty = it writes them."""
    req = str(req)
    if not (low := req.partition(":")[2].strip().lower()) or req in refs:
        return []
    if any(bit in low for bit in _C_POINTER_BITS):
        return []
    if _YES_NO_STORY.fullmatch(low):
        low = _C_OPEN_PAREN.split(req.partition(":")[0], 1)[0].strip().lower()
    rq = str(request).lower()
    body = _C_LINKS.sub(" ", low)
    if (figs := _C_FIGURE.findall(body)) and all(figure_in(f, rq) for f in figs):
        return []
    rwords = _C_LETTERS.findall(rq)
    rstems = [(w[:-1] if w.endswith("s") else w) if len(w) > 3 else w for w in rwords]
    content = _content_words(body)
    missing = [w for w in content if not _c_written(w, rwords, rstems)]
    if len(content) < 3 and any(w not in missing for w in content if len(w) > 3):
        return []
    for run in _C_FIG_RUN.findall(body):
        if len(_C_FIGURE.findall(run)) < 2:
            continue
        if re.search(_c_run_pattern(run, r"(?!\d)"), rq) is None:
            missing.append(run.strip())
    return missing


# ------------------------------------------------------------------------------------------------ kinds of product
def kind_set(text: Any) -> set:
    return {w.rstrip("s") for w in _content_words(str(text).lower())}


def kind_list(text: Any) -> list:
    return _content_words(_C_POSSESSIVE.sub("", str(text).lower().replace("’", "'")))


def _c_singulars(w: str) -> list:
    """The singular spellings a kind word ending in s can stand for."""
    found = [w[:-1]]
    if w[-2:] == "es":
        found.append(w[:-2])
        if w[-3:] == "ies":
            found.append(w[:-3] + "y")
        elif w[-3:] == "ves":
            found += [w[:-3] + "f", w[:-3] + "fe"]
    return found


def plurals(w: str) -> set:
    w = str(w)
    forms = set((w, w + "s", w + "es"))
    last = w[-1:]
    if last == "y":
        extra = [w[:-1] + "ies"]
    elif last == "f":
        extra = [w[:-1] + "ves"]
    elif w[-2:] == "fe":
        extra = [w[:-2] + "ves"]
    elif last == "s":
        extra = _c_singulars(w)
    else:
        extra = []
    forms.update(extra)
    return set(filter(lambda f: len(f) > 2, forms))


def same_kind(a: str, b: str) -> bool:
    if a == b:
        return True
    return str(a) in plurals(b) or str(b) in plurals(a)


def kind_found(word: str, pool: Any) -> bool:
    return next((True for p in pool if same_kind(word, p)), False)


def kinds_found(words: Any, pool: Any) -> bool:
    for w in words:
        if not kind_found(w, pool):
            return False
    return True


def kind_points(own: Any, words: Any, shared: Any) -> float:
    total = 0
    for w in own:
        if kind_found(w, words):
            total += 0.5 if kind_found(w, shared) else 1.0
    return total


def _c_forms_alternation(tok: str) -> str:
    longest_first = sorted(sorted(plurals(tok)), key=len, reverse=True)
    return "|".join(map(re.escape, longest_first))


def kind_regex(name: Any) -> str:
    return r"[\s'\-]+".join("(?:%s)(?:'s\\b|'|\\b)" % _c_forms_alternation(tok) for tok in kind_list(name))


# ------------------------------------------------------------------------------------------------ story hints
def single_prop_hint(hint: str) -> str:
    try:
        head, sep, detail = hint.partition(":")
        if not sep or "/" not in head:
            return hint
        tag = ""
        m = _C_TAG_END.search(head)
        if m:
            tag, head = " (%s)" % m.group(1).strip(), head[:m.start()]
        names = [n.strip() for n in head.split("/") if n.strip()]
        if len(names) < 2:
            return hint
        return "%s%s:%s (or %s)" % (names[0], tag, detail, ", ".join(names[1:]))
    except Exception:  # noqa: BLE001
        return hint


def leading_value(detail: Any, request: Any, product: Any) -> str:
    try:
        lead = _C_LEAD.match(str(detail).lower())
        word = "" if lead is None else lead.group(1)
        if len(word) < 3 or word in FILLER:
            return ""
        stems = [k.rstrip("s") for k in _content_words(str(product).lower())]
        if not stems or word.rstrip("s") in stems:
            return ""
        toks = _C_REQ_TOKEN.findall(str(request).lower())
        for here, nxt in zip(toks, toks[1:]):
            if nxt.partition("'")[0].rstrip("s") == stems[0] and here == word:
                return word
        return ""
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------ references
def points_elsewhere(value: Any) -> bool:
    return POINTING.search(str(value or "").lower()) is not None


def _c_glue_unit(m) -> str:
    if m.group(2) in FILLER:
        return m.group(0)
    return m.group(1) + m.group(2)


def _c_cut_plural(t: str) -> str:
    return t[:-2] if _C_SIBILANT_ES.search(t) else t[:-1]


def _c_ref_token(t: str):
    """One comparable token, or None for a link word or a lone letter."""
    if t in FILLER or (len(t) == 1 and not t.isdigit()):
        return None
    if len(t) <= 3 or not t.endswith("s") or t.endswith("ss") or t[-2].isdigit():
        return t
    return _c_cut_plural(t)


def ref_tokens(text: Any) -> list:
    low = _C_FIG_UNIT.sub(_c_glue_unit, str(text or "").lower())
    kept = (_c_ref_token(t) for t in _C_REF_TOKEN.findall(low))
    return [t for t in kept if t is not None]


def ref_figures(text: Any) -> set:
    found = set()
    try:
        low = str(text or "").lower().replace(",", "")
        for m in _C_FIG_WITH_UNIT.finditer(low):
            figures, unit = m.groups()
            if len(unit) >= 3 and unit[-1] == "s" and unit[-2:] != "ss":
                unit = _c_cut_plural(unit)
            if unit not in FILLER:
                found.update((float(f), unit) for f in _C_FIGURE.findall(figures))
    except Exception:  # noqa: BLE001
        pass
    return found


def seller_key(text: Any) -> str:
    bare = _C_POSSESSIVE.sub("", str(text or "").lower())
    return _C_NON_ALNUM.sub("", bare)


def ref_relation(value: Any) -> tuple:
    low = str(value or "").lower()
    if DIFFERING.search(low):
        return ("differ", None, "")
    m = _C_WITHIN.search(low)
    if m is not None:
        figure, unit = m.groups()
        return ("within", float(figure), unit)
    relation = "same"
    if _C_AT_MOST.search(low):
        relation = "at most"
    elif _C_AT_LEAST.search(low):
        relation = "at least"
    return (relation, None, "")


def _c_seller_match(t: str, accept):
    for m in _C_CAP_NAME.finditer(t):
        if m.start() > 0 and (accept is None or accept(m.group(1))):
            return m
    return None


def _c_sold_by_kind(before: str):
    """The product words between the last article and a closing 'sold by' (or made/offered/listed by), else None."""
    for art in reversed(list(_C_ARTICLE.finditer(before))):
        tail = _C_SOLD_BY_TAIL.match(before[art.end():])
        if tail:
            return tail.group(1)
    return None


def _c_detail_words(piece) -> list:
    return [w for w in _C_DETAIL_WORD.findall(piece or "") if w.lower() not in FILLER]


def ref_description(text: Any, accept=None) -> dict | None:
    try:
        t = _C_OR_ALT.split(str(text or ""), 1)[0]
        name = _c_seller_match(t, accept)
        if name is None:
            return None
        before = t[:name.start()]
        after = t[name.end():]
        kind = _c_sold_by_kind(before)
        if kind is None:
            kind = ""
            lead = _C_KIND_LEAD.match(after)
            if lead:
                kind = lead.group(1)
                after = after[lead.end():]
        details = [" ".join(ws) for ws in map(_c_detail_words, _C_DETAIL_SEP.split(after)) if ws]
        return dict(seller=name.group(1), kind=kind.strip()[:80], details=details[:6])
    except Exception:  # noqa: BLE001
        return None


def _c_plural_pair(tok: str, word: str) -> bool:
    """One is the other with a plural s or es (the singular 3+ letters)."""
    if len(word) >= 3 and tok in (word + "s", word + "es"):
        return True
    return len(tok) >= 3 and word in (tok + "s", tok + "es")


def mentions_prop(prop: Any, text: Any, exact: bool = False) -> bool:
    words = _C_ALNUM.findall(_name_core(prop or ""))
    toks = _C_ALNUM.findall(str(text or "").lower())
    if len(" ".join(words)) < 3:
        return False
    start = 0
    while start + len(words) <= len(toks):
        if all(t == w or (not exact and _c_plural_pair(t, w)) for t, w in zip(toks[start:], words)):
            return True
        start += 1
    return False


def after_prop(prop: Any, source: Any) -> str:
    try:
        head = str(prop).split("(")[0].lower()
        words = [w for w in _C_ALNUM.findall(head) if w not in FILLER]
        if not words:
            return ""
        src = str(source)
        pattern = "".join((r"\b", r"\W+".join(map(re.escape, words)), r"\b"))
        m = re.search(pattern, src.lower())
        if m is None:
            return ""
        end = m.end()
        return _C_SENTENCE_CUT.split(src[end:end + 500], 1)[0]
    except Exception:  # noqa: BLE001
        return ""


# ------------------------------------------------------------------------------------------------ observation lens
def _c_set_sole(d, names, field, fits):
    """Copy the one field whose (lower-case name, value) fits into `field`; nothing when none or several fit."""
    keys = [k for n, k in names.items() if fits(n, d[k])]
    if len(keys) == 1:
        d[field] = d[keys[0]]


def _c_id_field(item) -> bool:
    k, v = item
    return isinstance(v, str) and "id" in str(k).lower() and ID_SHAPE.fullmatch(v) is not None


def _c_plain_fields(d):
    if not any(map(_c_id_field, d.items())):
        return
    names = dict((str(k).lower(), k) for k in d)
    lowered = list(names)
    if not any(numeric(d[k]) and ("price" in n or "cost" in n) for n, k in names.items()):
        _c_set_sole(d, names, "price", lambda n, v: numeric(v) and _C_PRICE_LIKE.search(n) and not _C_NOT_PRICE.search(n))
    if not any("stock" in n or "avail" in n for n in lowered):
        _c_set_sole(d, names, "in_stock", lambda n, v: isinstance(v, bool) and _C_STOCK_LIKE.search(n))
    if not any("title" in n or "heading" in n for n in lowered):
        _c_set_sole(d, names, "title", lambda n, v: isinstance(v, str) and _C_TITLE_LIKE.search(n) and not _C_NOT_TITLE.search(n))


def with_plain_fields(observation: Any) -> Any:
    # a copy whose listing-like records also carry plain price / in_stock / title fields, taken from the one field
    # of a renamed kind that fits, when they lack them
    try:
        text = json.dumps(observation)
    except (TypeError, ValueError):
        return observation
    copy = json.loads(text)
    each_dict(copy, _c_plain_fields)
    return copy


# ------------------------------------------------------------------------------------------------------------- the link
_MODEL_PATH = "/inference/chat/completions"
_STORE_PATH = "/environment/call"
_LINK_KEY_GONE = ("key limit", "insufficient", "credit")
_LINK_ALLOWED = re.compile(r'allowed_models"?\s*:\s*\[([^\]]*)')
_LINK_QUOTED = re.compile(r'"([^"\s]{3,100})"')
_LINK_EXPECTED = re.compile(r"expected (\d+)")


def _exchange(client: Any, path: str, body: dict) -> tuple[Any, dict | None]:
    """One POST: (data, None) on a JSON reply, else (None, the client's error object: kind, status, body)."""
    verbose = getattr(client, "post_verbose", None)
    if verbose is not None:
        got = verbose(path, body)
        if got.data is not None:
            return got.data, None
        err = got.error
        return None, (err if isinstance(err, dict) or err is None else {"body": str(err)})
    data = client.post(path, json_data=body)
    if data is not None:
        return data, None
    return None, {"kind": "network", "status": None, "body": "no response"}


def _json_text(value: Any) -> str:
    """JSON text of a value as the pacing and cost guesses measure it (non-ASCII escaped, odd objects as text)."""
    return json.dumps(value, default=str)


def _link_fault(err) -> tuple:
    """(status, body text) of a failed request's error object."""
    fault = err or {}
    return fault.get("status"), str(fault.get("body") or "")


def _link_clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _link_client(timeout: int):
    # one try a request: the retries are ours
    return ProxyClient(timeout=timeout, max_retries=1, retry_delay=1)


class _LinkCall:
    """One model call in progress: its ladder (it grows when a refusal lists the allowed models), how its attempts
    failed, the models refused within it, the pause seed, and whether it has to end."""

    def __init__(self, link: "Link", models: list) -> None:
        self.given = models
        self.ladder = list(models) + [m for m in link.spares if m not in models]
        self.kinds: set = set()
        self.refused: set = set()
        self.seed = ""
        self.ended = False
        self.alone = False
        self.recount(link)

    def recount(self, link: "Link") -> None:
        self.alone = len({m for m in self.ladder if m not in link.banned}) <= 1


class Link:
    """All traffic of one errand: a shared pacing gate, model calls down a ladder with retries, store calls with step
    numbers, spend kept, models given up when refused, and a hard deadline."""

    def __init__(self, binding: dict) -> None:
        self.t0 = monotonic()
        self.bind = dict(binding or {})
        self.brain = _link_client(70)
        self.shop = _link_client(95)
        self.spend, self.scale = 0.0, 1.0
        self.tick, self.misses = 0, 0
        self.done = self.broke = self.no_model = self.mute = False
        self.last_send = self.slow_until = self.model_resume = 0.0
        self.last_answer = self.t0
        self.call_fail = ""
        self.trace: list = []
        self.spares: list = []
        self.banned: set = set()
        self.tools_banned: set = set()
        self.call_models: set = set()

    # ---- small reads
    def left(self) -> float:
        return Knobs.BUDGET_S - (monotonic() - self.t0)

    def spent_out(self) -> bool:
        return self.spend >= Knobs.SPEND_HARD * self.scale

    def down(self, at: float | None = None) -> bool:
        if at is None:
            at = monotonic()
        if self.no_model:
            return True
        return self.misses > 0 and at - self.last_answer > Knobs.OUTAGE_S

    def usable(self, models: list, tools: bool = False) -> bool:
        if self.no_model or self.mute:
            return False
        blocked = self.banned | self.tools_banned if tools else self.banned
        return any(m not in blocked for m in [*models, *self.spares])

    # ---- pacing
    def _send(self, client: Any, path: str, body: dict) -> tuple[Any, dict | None]:
        now = monotonic()
        wait = self.last_send + (Knobs.GAP_SLOW_S if now < self.slow_until else Knobs.GAP_S) - now
        if "/inference/" in path and self.model_resume - now > wait:
            wait = self.model_resume - now
        cap = self.left() - 1
        if cap < wait:
            wait = cap
        if wait > 0:
            sleep(wait)
        self.last_send = monotonic()
        data, err = _exchange(client, path, body)
        if data is None and _link_fault(err)[0] in (429, 503):
            self.slow_until = monotonic() + Knobs.SLOW_HOLD_S
        return (data, err)

    def _pause(self, seed: str, n: int, status: Any) -> float:
        """A growing pause, spread by a mix of the seed, the attempt and the step - never by the clock - so errands
        refused together do not come back together."""
        first = 2.0 if status == 429 else 1.0
        base = min(8.0, first * 2 ** n)
        mix = sum(map(ord, seed[-64:]))
        mix += 7 * len(seed) + 31 * n + 13 * self.tick
        mix %= 101
        spread = 0.75 + 0.5 * (mix / 100.0)
        return base * spread

    # ---- message shapes
    @staticmethod
    def flat(messages: list) -> list:
        def one(m):
            parts = m.get("content")
            if not isinstance(parts, list):
                return m
            texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict)]
            return dict(m, content="\n\n".join(texts))
        return list(map(one, messages))

    @staticmethod
    def marked(messages: list, model: str) -> list:
        if not model.startswith("anthropic/") or len(messages) <= 2:
            return messages
        out = list(map(dict, messages))
        for i in (0, -1):
            text = out[i].get("content")
            if text and isinstance(text, str):
                out[i]["content"] = [dict(type="text", text=text, cache_control=dict(type="ephemeral"))]
        return out

    # ---- a model call
    def ask_model(self, messages: list, models: list, cap: int, tools: list | None = None) -> dict | None:
        """The first message with content or tool calls from the ladder, or None (call_fail says how it failed)."""
        self.call_models = set()
        self.call_fail = ""
        if self.no_model or self.mute:
            return None
        call = _LinkCall(self, models)
        at = 0
        while at < len(call.ladder) and not self.no_model:
            model = call.ladder[at]
            given_next = models[at + 1] if at + 1 < len(models) else None
            at += 1
            if self._passes_over(model, given_next, call, messages, cap, tools):
                continue
            message = self._try_model(model, call, messages, cap, tools)
            if message is not None:
                return message
            if call.ended:
                break
        if "empty" in call.kinds:
            self.call_fail = "empty"
        else:
            self.call_fail = "refused" if call.kinds == {"hard"} else ""
        return None

    def _passes_over(self, model: str, given_next, call: _LinkCall, messages: list, cap: int, tools) -> bool:
        """The ladder skips this model: refused or banned, or a sonnet the spend rules leave out."""
        if model in self.banned or model in call.refused or (tools and model in self.tools_banned):
            return True
        if "sonnet" not in model:
            return False
        chars = sum(map(len, map(_json_text, messages)))
        guess = chars / 2.3 * 0.000002 + min(cap, 1500) * 0.00001
        if self.spent_out():
            return True
        if given_next is None or "sonnet" in given_next:
            return False
        return self.spend >= Knobs.SPEND_SOFT * self.scale or self.spend + guess > Knobs.SPEND_HARD * self.scale

    def _gone(self, call: _LinkCall, at: float | None = None) -> bool:
        return self.down(at) and (call.alone or len(self.call_models) >= 2)

    def _try_model(self, model: str, call: _LinkCall, messages: list, cap: int, tools) -> dict | None:
        """Attempts on one model: its message, or None (call.ended set when the whole call has to stop)."""
        plain_body = False
        tries_left = 1 if self.down() else Knobs.TRIES_PER_MODEL
        while tries_left > 0:
            tries_left -= 1
            if self.left() < Knobs.FINAL_S - 10:
                call.ended = True
                return None
            ceiling = 30 if self.down() else 75
            self.brain.timeout = int(_link_clamp(self.left() - 13, 8, ceiling))
            shaped = self.flat(messages) if plain_body else self.marked(messages, model)
            body: dict = dict(model=model, temperature=0, max_tokens=cap, messages=shaped)
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            self.call_models.add(model)
            started = monotonic()
            data, err = self._send(self.brain, _MODEL_PATH, body)
            choices = data.get("choices") if isinstance(data, dict) else None
            if choices:
                return self._reply(data, choices, model, call)
            self.misses += 1
            status, text = _link_fault(err)
            note = str(err)[:300] if err else "empty"
            self.trace.append(dict(kind="inference_failure", model=model, error=note))
            verdict = self._verdict(status, text, model, call, tools, plain_body)
            if verdict == "plain":
                plain_body = True
                continue
            if verdict == "end":
                call.ended = True
                return None
            if verdict == "next":
                return None
            if monotonic() - started > 20:
                call.ended = self._gone(call)
                return None
            if not call.seed:
                call.seed = _json_text(messages[-1:])[-600:]
            wait = self._pause(call.seed + model, min(self.misses - 1, 6), status)
            if Knobs.FINAL_S - 10 > self.left() - wait:
                call.ended = True
                return None
            self.model_resume = monotonic() + wait
            if self._gone(call, self.model_resume):
                call.ended = True
                return None
        return None

    def _reply(self, data: dict, choices: Any, model: str, call: _LinkCall) -> dict | None:
        """An answer came: the model is up again, its cost counts; its message, or None when it is empty."""
        self.last_answer = monotonic()
        self.misses = 0
        usage = data.get("usage")
        cost = usage.get("cost") if isinstance(usage, dict) else None
        if numeric(cost):
            self.spend += float(cost)
        first = choices[0] or {}
        message = first.get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message
        call.kinds.add("empty")
        self.trace.append(dict(kind="inference_failure", model=model, error="empty"))
        return None

    def _verdict(self, status: Any, text: str, model: str, call: _LinkCall, tools, plain_body: bool) -> str:
        """What a failed attempt means: "end" (no model may be called now), "next" (this model is done for the call),
        "plain" (send the plain body at once) or "soft" (load or network: pause and try again)."""
        low_text = text.lower()
        if status == 402 or (status == 403 and any(w in low_text for w in _LINK_KEY_GONE)):
            self.no_model = True
            return "end"
        if status in (401, 403, 404):
            call.kinds.add("hard")
            call.refused.add(model)
            if status == 404 and tools and "tool" in low_text:
                self.tools_banned.add(model)
            elif status != 403 or ("moderat" not in low_text and "flagged" not in low_text):
                self.banned.add(model)
            if status == 403 and not self.spares:
                self._take_spares(text, call)
            return "next"
        if status in (400, 413, 422):
            call.kinds.add("hard")
            if not plain_body:
                return "plain"
            call.refused.add(model)
            return "next"
        if isinstance(status, int) and 400 <= status < 500 and status not in (408, 425, 429):
            call.kinds.add("hard")
            call.refused.add(model)
            return "next"
        call.kinds.add("soft")
        return "soft"

    def _take_spares(self, text: str, call: _LinkCall) -> None:
        """A refusal that lists the allowed models: the first two not given up become the errand's fallback ladder."""
        found = _LINK_ALLOWED.search(text)
        if found is None:
            return
        listed = _LINK_QUOTED.findall(found.group(1))
        self.spares = [n for n in listed if n not in self.banned and n not in call.given][:2]
        call.ladder += [n for n in self.spares if n not in call.ladder]
        call.recount(self)

    def ask_json(self, prompt: str, cap: int, models: list) -> dict | None:
        message = self.ask_model([dict(role="user", content=prompt)], models, cap)
        if not message:
            return None
        return first_json(message.get("content"))

    # ---- a store call
    def _stepped(self, data: dict, count: int) -> dict:
        step = data.get("step")
        if not isinstance(step, int):
            step = self.tick + count
        self.tick = step
        if data.get("finished"):
            self.done = True
        return data

    def act(self, actions: list) -> dict | None:
        body = dict(self.bind, step=self.tick + 1)
        key, value = ("action", actions[0]) if len(actions) == 1 else ("actions", actions)
        body[key] = value
        tries = bumps = 0
        rounds = 3 + Knobs.STORE_TRIES
        while rounds > 0:
            rounds -= 1
            if 2 > self.left():
                break
            self.shop.timeout = int(_link_clamp(self.left() - 1, 3, 95))
            data, err = self._send(self.shop, _STORE_PATH, body)
            if isinstance(data, dict):
                return self._stepped(data, len(actions))
            status, text = _link_fault(err)
            self.trace.append(dict(kind="env_failure", status=status, body=text[:300]))
            if status == 402:
                self.broke = True
                break
            if status == 409:
                if "finished" in text:
                    self.done = True
                    break
                if (expected := _LINK_EXPECTED.search(text)) is not None:
                    wanted = int(expected.group(1))
                    self.tick = wanted - 1
                    body["step"] = wanted
                    continue
                if bumps >= 2 or "reused" not in text:
                    break
                body["step"] += 1
                bumps += 1
                continue
            if status in (403, 422) or Knobs.STORE_TRIES - 1 <= tries:
                break
            if tries and status in (400, 401, 404):
                break
            seed = _json_text(body)[-600:]
            wait = self._pause(seed, tries, status)
            tries += 1
            if 2 > self.left() - wait:
                break
            sleep(wait)
        return None


# --------------------------------------------------------------------------------------------------- argument repair
_TYPE_TESTS = dict(
    null=lambda v: v is None,
    boolean=lambda v: isinstance(v, bool),
    string=lambda v: isinstance(v, str),
    array=lambda v: isinstance(v, list),
    object=lambda v: isinstance(v, dict),
    number=numeric,
    integer=lambda v: isinstance(v, int) and not isinstance(v, bool),
)
_WHOLE = re.compile(r"\s*-?\d+(?:\.0+)?\s*")
_DECIMAL = re.compile(r"\s*-?\d+(?:\.\d+)?\s*")
_E_KEEP = object()        # no plain rewrite of a value into a kind
_E_BOUNDS = (("minimum", lambda v, n: v < n, "below minimum"),
             ("maximum", lambda v, n: v > n, "above maximum"),
             ("exclusiveMinimum", lambda v, n: v <= n, "must be above"),
             ("exclusiveMaximum", lambda v, n: v >= n, "must be below"))


class ArgFixer:
    """A tool argument object against the tool's schema: coerced where the meaning is plain (numbers written as text,
    enum case, over-long text cut), and every remaining breach named. Any breach keeps the call from the store."""

    @staticmethod
    def _types(schema: dict) -> list:
        if isinstance(t := schema.get("type"), list):
            return list(t)
        return [t] if t and isinstance(t, str) else []

    @staticmethod
    def _fits(value: Any, kinds: list) -> bool:
        for k in kinds:
            test = _TYPE_TESTS.get(k)
            if test is None or test(value):
                return True
        return False

    @staticmethod
    def _as_kind(value: Any, k: Any) -> Any:
        """The value rewritten as kind k where its meaning is plain, else _E_KEEP."""
        if k == "boolean":
            low = value.lower() if isinstance(value, str) else None
            if low in ("true", "false"):
                return low == "true"
        elif k == "string":
            if numeric(value):
                return str(value)
        elif k == "number":
            if isinstance(value, str) and _DECIMAL.fullmatch(value):
                n = float(value.strip())
                return n if not n.is_integer() else int(n)
        elif k == "integer":
            if isinstance(value, float):
                if value.is_integer():
                    return int(value)
            elif isinstance(value, str) and _WHOLE.fullmatch(value):
                return int(float(value.strip()))
        return _E_KEEP

    @classmethod
    def _coerce(cls, value: Any, kinds: list) -> Any:
        if not kinds:
            return value
        if cls._fits(value, kinds):
            whole_float = isinstance(value, float) and value.is_integer()
            return int(value) if whole_float and "integer" in kinds else value
        for k in kinds:
            if (new := cls._as_kind(value, k)) is not _E_KEEP:
                return new
        return value

    @staticmethod
    def _limit(schema: dict, key: str, kinds: Any = (int, float)) -> Any:
        return n if isinstance(n := schema.get(key), kinds) else None

    @classmethod
    def fix(cls, value: Any, schema: Any, path: str = "args") -> tuple[Any, list]:
        if isinstance(schema, dict):
            return cls._fix_with(value, schema, path)
        return value, []

    @classmethod
    def _fix_with(cls, value, schema, path):
        kinds = cls._types(schema)
        value = cls._coerce(value, kinds)
        if kinds and not cls._fits(value, kinds):
            return value, ["%s: expected %s" % (path, "/".join(kinds))]
        faults: list = []
        if "enum" in schema:
            value = cls._enum_rule(value, schema.get("enum") or [], path, faults)
        if "const" in schema and value != schema.get("const"):
            faults.append("%s: must equal %r" % (path, schema.get("const")))
        # one value has one of these types, so at most one group of rules applies
        if isinstance(value, str):
            value = cls._text_rules(value, schema, path, faults)
        elif isinstance(value, list):
            value = cls._list_rules(value, schema, path, faults)
        elif isinstance(value, dict):
            if "type" in schema:
                value = cls._members(value, schema, path, faults)
        elif numeric(value):
            for key, broken, words in _E_BOUNDS:
                if numeric(n := schema.get(key)) and broken(value, n):
                    faults.append("%s: %s %s" % (path, words, n))
        return value, faults

    @staticmethod
    def _enum_rule(value, options, path, faults):
        """A value outside the enum: the member it names in another case or with surrounding space, else a fault."""
        if value in options:
            return value
        hit = None
        if isinstance(value, str):
            wanted = value.strip().lower()
            hit = next(filter(lambda o: isinstance(o, str) and o.lower() == wanted, options), None)
        if hit is not None:
            return hit
        faults.append("%s: must be one of %s" % (path, options))
        return value

    @classmethod
    def _text_rules(cls, value: str, schema: dict, path: str, faults: list) -> str:
        shortest = cls._limit(schema, "minLength")
        longest = cls._limit(schema, "maxLength", int)
        if shortest is not None and len(value) < shortest:
            faults.append("%s: shorter than %s characters" % (path, shortest))
        if longest is not None and len(value) > longest:
            return value[:longest]
        return value

    @classmethod
    def _list_rules(cls, value: list, schema: dict, path: str, faults: list) -> list:
        fewest = cls._limit(schema, "minItems")
        most = cls._limit(schema, "maxItems")
        if fewest is not None and len(value) < fewest:
            faults.append("%s: needs at least %s items" % (path, fewest))
        if most is not None and len(value) > most:
            faults.append("%s: allows at most %s items" % (path, most))
        each = schema.get("items")
        if not isinstance(each, dict):
            return value
        done = [cls.fix(element, each, "%s[%d]" % (path, i)) for i, element in enumerate(value)]
        faults.extend(f for _, more in done for f in more)
        return [new for new, _ in done]

    @classmethod
    def _members(cls, value: dict, schema: dict, path: str, faults: list) -> dict:
        if not isinstance(members := schema.get("properties"), dict):
            members = {}
        missing = [name for name in schema.get("required") or [] if name not in value]
        faults.extend("%s.%s: required" % (path, name) for name in missing)
        if schema.get("additionalProperties") is False:
            faults.extend("%s.%s: not an allowed argument" % (path, name) for name in value if name not in members)
        out = {}
        for name in value:
            v = value[name]
            spec = members.get(name)
            if isinstance(spec, dict):
                v, more = cls.fix(v, spec, "%s.%s" % (path, name))
                faults.extend(more)
            out[name] = v
        return out


# --------------------------------------------------------------------------------------------------------- the archive
_SHELF_KEY = re.compile(r"categor|shelf|department|aisle")
_F_STOCKISH = re.compile(r"stock|avail|on.?hand")
_F_STOCK = re.compile(r"stock|avail")
_F_OUTISH = re.compile(r"out|sold")
_F_ON_HAND = re.compile(r"on.?hand")
_F_NONE_WORD = re.compile(r"\b(?:out|sold|none|unavailable|not|no|false)\b")
_F_LATER = re.compile(r"back.?order|pre.?order|discontinu")
_F_DIGIT = re.compile(r"\d")
_F_DASH = re.compile(r"[-_]")
_F_SEPARATORS = re.compile(r"[_.]+")


def _id_ish(key: Any) -> bool:
    return "id" in str(key).lower()


def _own_id(rec: dict) -> str:
    """The record's own listing id: an id / _id text field, or an ...id field holding an id-shaped text."""
    for k, v in rec.items():
        if not isinstance(v, str):
            continue
        lk = str(k).lower()
        if lk in ("id", "_id") or (lk.endswith("id") and ID_SHAPE.fullmatch(v)):
            return v
    return ""


def _own_label(rec: dict) -> str:
    for k, v in rec.items():
        if isinstance(v, str) and 0 < len(label := v.strip()) <= 60:
            lk = str(k).lower()
            if "id" not in lk and _SHELF_KEY.search(lk):
                return label
    return ""


def _has_map(rec: dict, least: int = 2) -> bool:
    for v in rec.values():
        if isinstance(v, dict) and len([x for x in v.values() if isinstance(x, (str, int, float))]) >= least:
            return True
    return False


def _f_price_field(fields: dict) -> Any:
    """The first numeric value under a price- or cost-like key, else None."""
    for k, v in fields.items():
        name = str(k).lower()
        if numeric(v) and ("price" in name or "cost" in name):
            return v
    return None


def _f_code_like(v: str, dashed: bool = True) -> bool:
    """A code-like value (figures and code characters only; with dashed, a dash or underscore too)."""
    if not (_F_DIGIT.search(v) and CODE_SHAPE.fullmatch(v)):
        return False
    return not dashed or _F_DASH.search(v) is not None


def _f_bump(tally: dict, key: Any) -> None:
    tally[key] = tally.get(key, 0) + 1


def _f_rich(rec: dict) -> bool:
    """A record with a property map of 3+ fields, or (with no map at all) 5+ short text fields."""
    maps = [v for v in rec.values() if isinstance(v, dict)]
    if maps:
        return max(map(len, maps)) >= 3
    return len([v for v in rec.values() if isinstance(v, str) and 0 < len(v) <= 70]) >= 5


def stock_word(key: Any, value: Any) -> str:
    """One field's stock reading: "no" (cannot buy now), "yes", or "" (not a stock / on-hand field, or no value)."""
    lk = str(key).lower()
    if value is None or _F_STOCKISH.search(lk) is None:
        return ""
    text = value.lower() if isinstance(value, str) else ""
    none_left = value is False or (numeric(value) and value <= 0)
    if _F_STOCK.search(lk):
        if not _F_OUTISH.search(lk):
            if none_left or (text and _F_NONE_WORD.search(text)):
                return "no"
        elif value is True:
            return "no"
    if none_left and _F_ON_HAND.search(lk):
        return "no"
    return "no" if text and _F_LATER.search(text) else "yes"


class _MenuTally:
    """The property names prop_menu counts: how often each shows, how often only on listing rows, sample values."""

    def __init__(self, shelf: str) -> None:
        self.shelf = shelf
        self.seen = 0
        self.mapped = 0
        self.tally: dict = {}
        self.rows: dict = {}
        self.samples: dict = {}

    def take(self, rec: dict) -> None:
        brief = [(k, v) for k, v in rec.items() if isinstance(v, str) and 0 < len(v) <= 70 and not not_a_property(k)]
        row_like = False
        if len(brief) < 5:
            # a short record counts only on a shelf, flat and with 2+ fields; one carrying an id is a listing row
            if not self.shelf or len(brief) < 2 or any(isinstance(v, dict) for v in rec.values()):
                return
            row_like = any(isinstance(v, str) and _id_ish(k) for k, v in rec.items())
        self.seen += 1
        if not row_like:
            self.mapped += 1
        for k, v in brief:
            if _f_code_like(v):
                continue
            name = _F_SEPARATORS.sub(" ", str(k)).strip().lower()
            _f_bump(self.tally, name)
            if row_like:
                _f_bump(self.rows, name)
            shown = self.samples.setdefault(name, [])
            low = v.lower()
            if len(shown) < 5 and low not in shown:
                shown.append(low)

    def menu(self) -> str:
        counts = self.tally
        if self.mapped > 0:
            # once records carry their properties, a field only the listing rows carry is no property
            counts = {n: c for n, c in counts.items() if self.rows.get(n, 0) < c}
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:28]
        return "\n".join(["- %s (in %d listings): %s" % (n, c, " | ".join(self.samples.get(n, []))) for n, c in ranked])


class Archive:
    """Every observation the store returned, and the views built on them."""

    def __init__(self) -> None:
        self.items: list = []
        self.ids: set = set()
        self.read: set = set()
        self.label_of: dict = {}

    def add(self, observation: Any) -> None:
        self.items.append(observation)
        each_dict(observation, self._note)

    def _note(self, rec: dict) -> None:
        self.ids.update(v for k, v in rec.items() if isinstance(v, str) and _id_ish(k) and ID_SHAPE.fullmatch(v))
        own = _own_id(rec)
        if not own:
            return
        if _has_map(rec):
            self.read.add(own)
        label = _own_label(rec)
        if label:
            self.label_of.setdefault(own, label)

    def _records(self):
        out: list = []
        for obs in self.items:
            each_dict(obs, out.append)
        return out

    def price(self, ref: str) -> float | None:
        found = None
        for rec in self._records():
            if not any(_id_ish(k) and v == ref for k, v in rec.items()):
                continue
            seen = _f_price_field(rec)
            if seen is not None:
                found = float(seen)
        return found

    def about(self, ref: str, hold: int = 4) -> list:
        texts: list = []
        stack = [(obs, 0) for obs in reversed(self.items)]
        while stack and len(texts) < 8:
            value, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == ref for v in value.values()):
                    shown = as_text(clip(value, 400))[:2500]
                    if shown not in texts:
                        texts.append(shown)
                    continue
                inner = [v for v in value.values() if isinstance(v, (dict, list))]
            elif isinstance(value, list):
                inner = value
            else:
                continue
            stack.extend((v, depth + 1) for v in reversed(inner))
        return texts[-hold:]

    def label_for(self, rec: dict) -> str:
        return _own_label(rec) or next((self.label_of[v] for k, v in rec.items()
                                         if isinstance(v, str) and v in self.label_of and _id_ish(k)), "")

    def _card_ref(self, rec: dict) -> str | None:
        for k, v in rec.items():
            if isinstance(v, str) and v in self.ids and _id_ish(k) and ID_SHAPE.fullmatch(v):
                return v
        return None

    def cards(self, cap: int = 30000, cheaper_than: str = "", skip: set | None = None, first: set | None = None) -> list:
        merged: dict = {}
        for rec in self._records():
            if (ref := self._card_ref(rec)) is None:
                continue
            two_map = any(isinstance(v, dict) and len(v) >= 2 for v in rec.values())
            if len(rec) < 4 and not two_map:
                continue
            merged.setdefault(ref, {}).update(rec)
            if two_map:
                self.read.add(ref)
        order = list(merged)
        refs = [r for r in order if r == cheaper_than or r not in skip] if skip else list(order)
        top = _f_price_field(merged[cheaper_than]) if cheaper_than and cheaper_than in merged else None
        if top is not None:
            refs = [r for r in refs if r == cheaper_than or (p := _f_price_field(merged[r])) is None or p < top]
        front = first or set()
        texts = {}
        for r in refs:
            head = "DETAILS " if r in self.read else "ROW "
            texts[r] = head + as_text(clip(merged[r], 160))
        crowded = sum(map(len, texts.values())) > cap

        def placing(r: str) -> tuple:
            # when the cards do not all fit, a read listing its record shows sold out goes after the other read ones
            sold_out = crowded and any(stock_word(k, v) == "no" for k, v in merged[r].items())
            return (r not in front, r not in self.read, sold_out, order.index(r))
        room = cap
        kept = []
        for r in sorted(refs, key=placing):
            text = texts[r]
            if len(text) <= room:
                kept.append(text)
                room -= len(text)
        return kept

    def _read_ref(self, rec: dict) -> str:
        for k, v in rec.items():
            if isinstance(v, str) and v in self.read and _id_ish(k):
                return v
        return ""

    def rich_reads(self, shelf: str = "") -> int:
        wanted = shelf.strip().lower() if shelf else ""
        rich: set = set()
        for rec in self._records():
            ref = self._read_ref(rec)
            if ref and (not shelf or self.label_for(rec).strip().lower() == wanted) and _f_rich(rec):
                rich.add(ref)
        return len(rich)

    def shelves_read(self) -> list:
        labels = [self.label_for(rec) for rec in self._records()
                  if any(isinstance(v, dict) for v in rec.values()) and self._read_ref(rec)]
        tally: dict = {}
        for label in filter(None, labels):
            _f_bump(tally, label)
        return [label for label, _ in sorted(tally.items(), key=lambda kv: kv[1], reverse=True)]

    def _on_shelf(self, rec: dict, shelf: str) -> bool:
        want = shelf.lower()
        if any(isinstance(v, str) and v.strip().lower() == want and _SHELF_KEY.search(str(k).lower()) for k, v in rec.items()):
            return True
        return self.label_for(rec).strip().lower() == shelf.strip().lower()

    def prop_menu(self, shelf: str = "") -> tuple:
        tally = _MenuTally(shelf)
        stack = [(obs, False, 0) for obs in reversed(self.items)]
        while stack:
            value, inside, depth = stack.pop()
            if depth > 6:
                continue
            if isinstance(value, dict):
                inside = inside or not shelf or self._on_shelf(value, shelf)
                if inside:
                    tally.take(value)
                inner = [v for v in value.values() if isinstance(v, (dict, list))]
            elif isinstance(value, list):
                inner = value
            else:
                continue
            stack.extend((v, inside, depth + 1) for v in reversed(inner))
        return tally.menu(), tally.seen

    def label_lines(self) -> str:
        fields: dict = {}
        pairs = (kv for rec in self._records() if len(rec) >= 4 for kv in rec.items())
        for k, v in pairs:
            if isinstance(v, str) and 0 < len(v) <= 40 and not _f_code_like(v, False):
                _f_bump(fields.setdefault(str(k), {}), v)
        lines = []
        for k, vals in fields.items():
            if len(lines) == 12:
                break
            counts = list(vals.values())
            if not (2 <= len(vals) <= 8 and sum(counts) >= 4 and max(counts) >= 2):
                continue
            shown = " | ".join("%s (x%d)" % pair for pair in sorted(vals.items(), key=lambda kv: kv[1], reverse=True))
            lines.append("- %s: %s" % (k, shown))
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- the requirement book
_F_TAG = re.compile(r"\(([^)]*)\)")
_F_LETTERS = re.compile(r"[a-z]+")
_F_ALNUM = re.compile(r"[a-z0-9]+")
_F_WORD3 = re.compile(r"[a-z]{3,}")
_F_MARKS = re.compile(r"\(stated\)|\(revealed[^)]*\)")
_F_NOT_PROP_REQ = re.compile(r"\bprice\b|budget|quantity|\bstock\b|\bcents?\b")
_F_MONEY_WORDS = re.compile(r"\b(?:prices?|budget|costs?|spend\w*|cents?|dollars?)\b")


def _head(need: str) -> str:
    return need.partition(":")[0]


def _tags(name: str) -> list:
    return _F_TAG.findall(name)


def _f_plain(text: Any) -> str:
    """Lower-case letter and figure runs of a text, one space apart."""
    return " ".join(_F_ALNUM.findall(str(text).lower()))


def _f_units_wanted(q: Any) -> int:
    """A quantity as the reading gives it: a whole number of 1 or more, else 1."""
    if numeric(q) and isinstance(q, int) and q >= 1:
        return q
    return 1


def _f_item_row(it: Any) -> dict | None:
    """One product line of a several-product reading; None when it names no product."""
    if not isinstance(it, dict):
        return None
    product = str(it.get("product") or "").strip()
    if not product:
        return None
    reqs = it.get("requirements")
    wants = [str(r) for r in reqs if r and ":" in str(r)] if isinstance(reqs, list) else []
    return dict(product=product, quantity=_f_units_wanted(it.get("quantity")), requirements=wants)


def _f_grounded(text: str, said: str) -> bool:
    """A requirement stands on what was read or said: none of its 3+ letter words, or one of them, appears there."""
    words = _F_WORD3.findall(text.lower())
    return not words or any(w in said for w in words)


def _f_swap_needs(answer: dict, swaps: dict) -> None:
    """Renamed needs, in the answer's needs and in its requirements_now list."""
    def one(n: Any) -> Any:
        return swaps.get(n, n) if isinstance(n, str) else n
    answer["needs"] = list(map(one, answer.get("needs") or []))
    now = answer.get("requirements_now")
    if isinstance(now, list):
        answer["requirements_now"] = list(map(one, now))


def _f_meta_blank() -> dict:
    """What a line that only asks for one question at a time says: nothing."""
    blank = {k: [] for k in ("needs", "no_preference", "dropped", "references")}
    blank.update(requirements_now=None, new_budget=None, unavailable=False, reply="")
    return blank


def _f_line_name(need: str, words: list, joined: str) -> str | None:
    """The need under the name the shopper's words use, when its own name is no phrase of them but each of its words
    starts the matching word of one run of them; else None."""
    head, value = need.split(":", 1)
    parts = _F_LETTERS.findall(head.split("(")[0].lower())
    if not parts or min(map(len, parts)) < 3:
        return None
    if re.search("".join((r"\b", r"\s+".join(parts), r"\b")), joined):
        return None
    tag = head[head.index("("):].strip() if "(" in head else ""
    width = len(parts)
    start = 0
    while start + width <= len(words):
        window = words[start:start + width]
        start += 1
        if window != parts and all(w.startswith(p) for w, p in zip(window, parts)):
            label = " ".join(window) + (" " + tag if tag else "")
            return label + ":" + value
    return None


def _f_phrases(lines: list) -> set:
    """The 1-3 word phrases of the lines (3+ characters, not a lone link word)."""
    words = _F_LETTERS.findall(" ".join(lines).lower())
    found = set()
    for n in (1, 2, 3):
        start = 0
        while start + n <= len(words):
            p = " ".join(words[start:start + n])
            start += 1
            if len(p) >= 3 and p not in FILLER:
                found.add(p)
    return found


def _f_fits(w: str, name: str, tag: list) -> bool:
    """w is a requirement of the property `name` (a product tag on both sides must agree)."""
    head = _head(w)
    if not loosely_same(head, name):
        return False
    their = _tags(head)
    return not tag or not their or their == tag


def _f_need_words(need: Any) -> tuple:
    """(property words of 3+ letters, value words without link words) of a need."""
    head, _, value = str(need).partition(":")
    return [w for w in name_words(head) if len(w) > 2], [w for w in _F_ALNUM.findall(value.lower()) if w not in FILLER]


def _f_matching(pool: list, name: str, tag: list) -> list:
    return [w for w in pool if _f_fits(w, name, tag)]


def _f_same_name(a: str, b: str) -> bool:
    return loosely_same(a, b) and _tags(a) == _tags(b)


def _f_after_compare(need: str) -> tuple | None:
    """(property part, value without its comparison words) of a need whose value may be a store value given after a
    comparison ('falls short of not available'); None for any other need."""
    head, value = need.split(":", 1)
    if _F_DIGIT.search(value) or points_elsewhere(value):
        return None
    low = value.lower()
    bare = _f_plain(COMPARE_LEAD.sub(" ", low))
    if not bare:
        return None
    if bare == _f_plain(low) and names_the_item(value, head):
        return None
    return head, bare


class Wishes:
    """What the shopper wants now: the reading of the request, stated and revealed needs, freed and dropped
    properties, the budget; fresh shopper lines are read by the listener model and applied with care."""

    def __init__(self, link: Link, request: str) -> None:
        self.link, self.request = link, request
        self.reading: dict = {}
        self.written, self.uncovered, self.indifferent, self.let_go = [], [], [], []
        self.resolved, self.ref_news, self.fresh_needs, self.pending, self.exchanges = [], [], [], [], []
        self.caught: set = set()
        self.now_reqs: list | None = None
        self.spend_cap: float | None = None
        self.field_hook = self.ref_hook = self.meta_last = None
        self.version = self.pinned_heard = 0
        self.widened_at, self.widened_tick = -1, -10 ** 6
        self.pending_at = 0.0
        self.reply_due = ""
        self.sweepable = False

    # ---- products and quantities
    def items(self) -> list:
        rows = [row for row in map(_f_item_row, self.reading.get("items") or []) if row is not None]
        return rows if len(rows) > 1 else []

    def quantity(self) -> int:
        return 1 if self.items() else _f_units_wanted(self.reading.get("quantity"))

    def counts(self) -> list:
        items = self.items()
        if not items:
            return [self.quantity()]
        return [it["quantity"] for it in items]

    def items_text(self) -> str:
        listed = ["%d) %s x%d" % (n, it["product"], it["quantity"]) for n, it in enumerate(self.items(), 1)]
        if not listed:
            return ""
        return ("THIS REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS: the order holds exactly one line per product at its "
                "quantity and nothing else; judge each listing only against the requirements of the product it is (its "
                "category decides); a cheapest or best rule applies within each product. " + " | ".join(listed))

    def priority(self) -> str:
        p = self.reading.get("priority")
        if p == "cheapest" or p == "best":
            return p
        return "none"

    def props_left(self) -> list:
        pool = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        return [r for r in pool if r not in self.let_go and not _F_NOT_PROP_REQ.search(str(r).lower())]

    def text_body(self) -> str:
        if self.now_reqs is None:
            parts = ["%s (stated)" % r for r in self.written if r not in self.let_go]
            parts += ["%s (revealed by the shopper)" % r for r in self.uncovered if r not in self.let_go]
        else:
            parts = list(self.now_reqs)
        if parts:
            head = "; ".join(parts)
        else:
            head = ("no property requirement is left - every in-stock listing in the requested category fits"
                    if self.let_go else "see the request")
        pieces = [head]
        if self.spend_cap is not None:
            pieces.append(("; total price at most %s %s" % (self.spend_cap, self.reading.get("currency") or "")).rstrip())
        pieces.append("; " + self.items_text() if self.items() else "; quantity %d" % self.quantity())
        if self.let_go:
            pieces.append("; no longer required: " + "; ".join(self.let_go))
        if self.indifferent:
            pieces.append("; shopper does not care about: " + ", ".join(self.indifferent))
        return "".join(pieces)

    def changes(self) -> str:
        if self.let_go:
            return ("CHANGES SINCE THE REQUEST (they override its text): the shopper dropped these requirements - " +
                    "; ".join(self.let_go) + ". A dropped requirement no longer counts: never fail, rank or prefer a listing "
                    "on it.")
        return ""

    def overview(self, steps: int, messages: int | None) -> str:
        r = self.reading
        if not r:
            return ""
        lines = ["A FIRST READING OF THE REQUEST (check it against the request itself):"]

        def each(key: str, template: str) -> None:
            lines.extend(template % q for q in r.get(key) or [])
        if r.get("product"):
            lines.append("- product: %s" % r["product"])
        items = self.items()
        for n, it in enumerate(items, 1):
            lines.append("- product %d: %s, quantity %d (one cart line)" % (n, it["product"], it["quantity"]))
        if items:
            lines.append("- the order holds exactly these lines and nothing else; send every line in each cart call when "
                         "the cart tool takes a list of lines")
        else:
            lines.append("- quantity: %d" % self.quantity())
        each("requirements", "- stated requirement: %s")
        each("references", "- value given by pointing at another listing: %s (read that listing first, then search with "
                           "its value)")
        if r.get("price_cap") is not None:
            lines.append("- total price at most %s %s" % (r.get("price_cap"), r.get("currency") or ""))
        if r.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if self.priority() == "none":
            lines.append("- no cheapest or best rule was found: any item meeting every requirement within the budget is "
                         "right - spend no steps hunting the lowest price unless the request itself says so")
        else:
            detail = r.get("priority_detail")
            lines.append("- choose the %s%s among all fitting items" % (self.priority(), (" (%s)" % detail) if detail else ""))
        each("soft_wants", "- nice to have (not required, only a tie-breaker): %s")
        each("ask_first", "- learn from the shopper first: %s")
        each("rules", "- rule: %s")
        if messages is not None:
            lines.append("- messages allowed: %s" % messages)
        lines.append("- steps allowed: %s" % steps)
        return "\n".join(lines)

    # ---- small rules used by the listener
    @staticmethod
    def own_names(needs: Any, lines: list) -> dict:
        words = _F_LETTERS.findall(" ".join(lines).lower())
        joined = " ".join(words)
        out = {}
        for need in needs if isinstance(needs, list) else []:
            if isinstance(need, str) and ":" in need:
                renamed = _f_line_name(need, words, joined)
                if renamed is not None:
                    out[need] = renamed
        return out

    def drop_named(self, old: str, lines: list) -> bool:
        try:
            name = _head(old)
            tag = _tags(name)
            stated = _f_matching(self.written, name, tag)
            if not stated:
                fresh = list(filter(lambda n: isinstance(n, str) and ":" in n, self.fresh_needs))
                stated = [w for w in _f_matching(self.uncovered, name, tag) if w not in self.written and w not in fresh]
                # a revealed value the reply replaces with a new one of the same property goes too
                if not stated or any(_f_fits(f, name, tag) and f != old for f in fresh):
                    return True
            phrases = _f_phrases(lines)
            if any(loosely_same(_head(s), p) for s in stated for p in phrases):
                return True
            if self.field_hook is None:
                return False
            for s in stated:
                if self._field_drop(s, name, phrases):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return True

    def _field_drop(self, s: str, name: str, phrases: set) -> bool:
        """A phrase names a field of the product's category (no other requirement's property) whose sample values
        carry the requirement's value."""
        if points_elsewhere(value := _F_MARKS.sub(" ", s.partition(":")[2])) or not value_options(value):
            return False
        fields = self.field_hook(_head(s)) or {}
        pool = [w for w in self.written + self.uncovered if isinstance(w, str) and ":" in w]
        others = [h for h in map(_head, pool) if not loosely_same(h, name)]
        free = [p for p in phrases if not any(strictly_same(p, o) for o in others)]
        for p in free:
            for f, vals in fields.items():
                if not strictly_same(f, p):
                    continue
                if seen_in_samples(value, " | ".join(map(str, vals))):
                    self.link.trace.append(dict(kind="drop_by_field", need=s, field=f))
                    return True
        return False

    def said_values(self, answer: dict) -> None:
        if self.field_hook is None:
            return
        mapped = {}
        for need in filter(lambda n: isinstance(n, str) and ":" in n, answer.get("needs") or []):
            if (found := _f_after_compare(need)) is None:
                continue
            head, bare = found
            prop = head.split("(")[0].strip()
            for f, vals in (self.field_hook(head) or {}).items():
                if strictly_same(f, prop) and (hit := next((v for v in vals if _f_plain(v) == bare), None)) is not None:
                    SAID_VALUES.add(bare)
                    mapped[need] = "%s: %s" % (head.strip(), str(hit).strip().lower())
                    break
        if mapped:
            _f_swap_needs(answer, mapped)
            self.link.trace.append(dict(kind="literal_value", needs=[*mapped.values()][:4]))

    def pinned(self, prop: str, question: Any = None) -> bool:
        try:
            given = self.reading.get("references")
            refs = [str(r) for r in given] if isinstance(given, list) else []
            refs.extend(self.resolved)
            tag = _tags(prop)
            return any(self._pins(w, prop, tag, refs) for w in self.written)
        except Exception:  # noqa: BLE001
            return False

    def _pins(self, w: str, prop: str, tag: list, refs: list) -> bool:
        """The stated requirement w keeps the property: in force, of the same product kind, a value the request writes,
        and the same property by name (or the asked name ends in its name)."""
        if w in self.let_go or ":" not in w:
            return False
        head, value = w.split(":", 1)
        rtag = _tags(head)
        if tag and rtag and not kind_set(tag[-1]) & kind_set(rtag[-1]):
            return False
        if not names_the_item(value, head) or unwritten_words(w, self.request, refs):
            return False
        if loosely_same(head, prop):
            return True
        own = name_key(head)
        return len(own) >= 4 and name_key(prop).endswith(own)

    @staticmethod
    def names_and_value(needs: Any, lines: list) -> bool:
        said = " ".join(map(str, lines)).lower()
        if not isinstance(needs, list):
            return False
        for names, values in map(_f_need_words, needs):
            if names and values and not any(w not in said for w in [*names, *values]):
                return True
        return False

    # ---- the listener
    def _listen(self, so_far: str, fresh: list, question: Any) -> dict | None:
        """The listener's reading of fresh lines; None when it gave none (lines lost to load or the network, or met
        while the code finishes alone, wait to be read again)."""
        prompt = listener_prompt(self.request, so_far, self.exchanges[-12:], question, fresh)
        asked = "asked: %s -> " % question if question else ""
        self.exchanges.append(asked + "shopper: " + " / ".join(fresh))
        answer = self.link.ask_json(prompt, 2500, LINEUP["listener"])
        if answer:
            return answer
        if self.link.mute or self.link.misses > 0:
            self.caught.difference_update(fresh)
            self.exchanges.pop()
            self.pending = [*self.pending, (list(fresh), question)]
            self.pending_at = monotonic()
        return None

    def _screened(self, answer: dict, fresh: list, question: Any) -> dict:
        """The answer with a one-question-at-a-time line emptied and needs renamed to the shopper's own words."""
        try:
            meta = answer.get("meta_only") is True and question and not self.names_and_value(answer.get("needs"), fresh)
        except Exception:  # noqa: BLE001
            meta = False
        if meta:
            answer = {**answer, **_f_meta_blank()}
            self.meta_last = question
            self.link.trace.append(dict(kind="meta_only", question=question))
        try:
            renames = self.own_names(answer.get("needs"), fresh)
        except Exception:  # noqa: BLE001
            renames = {}
        if renames:
            _f_swap_needs(answer, renames)
            self.link.trace.append(dict(kind="need_renamed", renames=renames))
        return answer

    def _masked(self, answer: dict, fresh: list, question: Any) -> list:
        if self.ref_hook is None:
            return fresh
        try:
            return self.ref_hook(answer, fresh, question) or fresh
        except Exception as error:  # noqa: BLE001
            self._trouble("ref_line_error", error)
            return fresh

    def _take_needs(self, answer: dict, said: str, notes: list) -> None:
        if isinstance(now := answer.get("requirements_now"), list) and all(isinstance(x, str) for x in now):
            self.now_reqs = [x for x in now if x.strip() and _f_grounded(x, said)]
        for need in answer.get("needs") or []:
            if not isinstance(need, str) or not need or need in self.uncovered:
                continue
            if _f_grounded(need, said):
                self.uncovered.append(need)
                notes.append("new requirement: %s" % need)

    def _take_pointed(self, answer: dict, fresh: list) -> None:
        """Needs the shopper gave by pointing at another listing go to the reference resolver."""
        try:
            given = answer.get("references")
            infos = list(filter(lambda r: isinstance(r, dict), given)) if isinstance(given, list) else []
            pointed = answer.get("ref_needs") or []
            line = " ".join(fresh)
            for need in filter(lambda n: isinstance(n, str) and n in self.uncovered, answer.get("needs") or []):
                if points_elsewhere(need.partition(":")[2]) or need in pointed:
                    self.ref_news = [*self.ref_news, dict(need=need, infos=infos, line=line)]
        except Exception:  # noqa: BLE001
            pass

    def _take_free(self, answer: dict, question: Any) -> None:
        pool = self.written + self.uncovered if self.now_reqs is None else self.now_reqs
        held = {_head(r).strip().lower() for r in pool if isinstance(r, str) and ":" in r and r not in self.let_go}
        for p in answer.get("no_preference") or []:
            if not isinstance(p, str) or not p or p in self.indifferent or p.strip().lower() in held:
                continue
            if self.pinned(p, question):
                self.pinned_heard += 1
            else:
                self.indifferent.append(p)

    def _take_drops(self, answer: dict, masked: list, notes: list, budget: Any) -> bool:
        widened = False
        for d in answer.get("dropped") or []:
            if not isinstance(d, str) or not d or d in self.let_go:
                continue
            if numeric(budget) and _F_MONEY_WORDS.search(d.lower()):
                continue
            if not self.drop_named(d, masked):
                self.link.trace.append(dict(kind="drop_kept", need=d))
                continue
            self.let_go.append(d)
            notes.append("dropped: %s" % d)
            if self.priority() != "none":
                widened = True
        return widened

    def _restore(self, masked: list) -> None:
        """A rebuilt requirement list keeps every revealed need and every stated requirement not dropped."""
        for need in self.uncovered:
            if (prop := _head(need).strip().lower()) and not any(prop in str(r).lower() for r in [*self.now_reqs, *self.let_go]):
                self.now_reqs.append(need)
        for w in self.written:
            try:
                lost = self._lost_statement(w, masked)
            except Exception:  # noqa: BLE001
                continue
            if lost:
                self.now_reqs.append(w)

    def _lost_statement(self, w: str, masked: list) -> bool:
        """A stated requirement the rebuilt list lost: not dropped, and no requirement of its property left."""
        if w in self.let_go or w in self.now_reqs or self.drop_named(w, masked):
            return False
        head = _head(w)
        if any(_f_same_name(_head(g), head) for g in self.let_go):
            return False
        return not any(isinstance(r, str) and _f_same_name(_head(r), head) for r in self.now_reqs)

    def _widen_note(self) -> str:
        if self.props_left():
            return ("the drop WIDENS what is acceptable: listings that failed only on the dropped requirement "
                    "now fit and unsearched listings may beat your pick - search the kind again without it, as "
                    "short queries of each remaining required value with the kind word, value first, reading "
                    "every page, and choose the cheapest over everything now acceptable")
        if self.priority() == "cheapest" and self.sweepable:
            return ("no property requirement is left: every in-stock listing of the requested category now "
                    "fits, whatever object it is (an accessory, a part or a set counts) and the cheapest on the "
                    "whole shelf wins; rows come by relevance or id, not price, so code reads further shelf "
                    "slices next - if no [shelf sweep done] list follows, cover the shelf yourself with short "
                    "queries (the category label, each kind word of it alone, each recurring item name from the "
                    "titles, the first page of each, plus the category listing's pages), then make the "
                    "lowest-priced in-stock row within budget the only cart line, check its stock and price "
                    "again, and order it")
        return ("no property requirement is left: every in-stock listing of the requested category now "
                "fits; cover the shelf with short queries and the category pages, then order the lowest "
                "in-stock price")

    def _trouble(self, kind: str, error: Exception) -> None:
        self.link.trace.append(dict(kind=kind, error=str(error)[:200]))

    def hear(self, lines: list, question: str | None = None) -> str:
        if not (fresh := [l for l in lines if l not in self.caught]):
            return ""
        self.caught.update(fresh)
        if self.link.left() < Knobs.WIND_DOWN_S:
            return ""
        so_far = self.text_body()
        if not (answer := self._listen(so_far, fresh, question)):
            return ""
        answer = self._screened(answer, fresh, question)
        masked = self._masked(answer, fresh, question)
        said = " ".join([so_far, " ".join(self.exchanges), question or ""]).lower()
        try:
            self.said_values(answer)
        except Exception as error:  # noqa: BLE001
            self._trouble("literal_value_error", error)
        self.fresh_needs = list(filter(lambda n: isinstance(n, str), answer.get("needs") or []))
        notes: list = []
        self._take_needs(answer, said, notes)
        self._take_pointed(answer, fresh)
        budget = answer.get("new_budget")
        self._take_free(answer, question)
        widened = self._take_drops(answer, masked, notes, budget)
        if numeric(budget) and 0 < budget != self.spend_cap:
            self.spend_cap = budget
            notes.append("new budget: %s" % budget)
        if answer.get("unavailable"):
            notes.append("an item became unavailable - take it out of the cart and do not order it")
        if notes:
            self.version += 1
            if widened:
                self.widened_at, self.widened_tick = self.version, self.link.tick
        if self.now_reqs is not None:
            self._restore(masked)
        if widened:
            notes.append(self._widen_note())
        sizes = {k: len(answer.get(k) or []) for k in ("needs", "no_preference", "dropped")}
        self.link.trace.append(dict(kind="heard", question=bool(question), **sizes, budget=numeric(budget)))
        self.reply_due = ""
        if question is None and not answer.get("no_preference") and any("?" in l for l in fresh):
            self.reply_due = str(answer.get("reply") or "").strip()
        meaning = "Meaning: %s" % (answer.get("meaning") or "")
        if notes:
            meaning += " (%s)" % "; ".join(notes)
        return meaning


# --------------------------------------------------------------------------------------------------------- the basket
_COUNTISH = re.compile(r"qty|quant|count|units|number|total|held|ordered")
_MONEYISH = re.compile(r"price|cost|due|sum|fee|charge")


def _f_line_fields(fields: dict) -> tuple | None:
    """(id field, count field) of a cart line object's fields: an id-named text field when one is, else the first
    text field; the first integer or number field."""
    texts, counts = [], []
    for f in fields:
        if isinstance(s := fields[f], dict):
            kind = s.get("type")
            if kind == "string" and "enum" not in s:
                texts.append(f)
            elif kind in ("integer", "number"):
                counts.append(f)
    ids = list(filter(lambda f: "id" in f.lower(), texts)) or texts
    if not ids or not counts:
        return None
    return ids[0], counts[0]


def _f_first_put(pairs: list) -> tuple | None:
    """The first (line, units) pair that puts units in (units unknown or above 0), else the first pair."""
    for pair in pairs:
        if pair[1] is None or pair[1] > 0:
            return pair
    return pairs[0] if pairs else None


def _f_row_line(row: Any, id_key: str, count_key: str) -> tuple | None:
    """(listing, units or None) of one line object of an array cart argument; None when it names no listing."""
    if not isinstance(row, dict) or not isinstance(ref := row.get(id_key), str) or not ref.strip():
        return None
    n = row.get(count_key)
    return ref.strip(), (float(n) if numeric(n) else None)


def _f_line_ref(v: Any, known: set) -> bool:
    return isinstance(v, str) and " " not in v and (v in known or ID_SHAPE.fullmatch(v) is not None)


def _f_count_named(key: Any) -> bool:
    low = str(key).lower()
    return bool(_COUNTISH.search(low)) and not _MONEYISH.search(low)


def _f_units_of(rec: dict) -> int | None:
    """The one units figure of a cart-like row (beside a price, the count-named field), else None."""
    pairs = [(k, v) for k, v in rec.items() if numeric(v) and isinstance(v, int) and 0 < v < 1000 and str(k) != "price"]
    if len(pairs) > 1:
        named = [pair for pair in pairs if _f_count_named(pair[0])]
        if len(named) == 1:
            pairs = named
    if len(pairs) != 1:
        return None
    return pairs[0][1]


class Basket:
    """The cart as code counts it: units per line, which call form puts units in, whether a put on a held line adds
    or sets, and what is wrong with the cart for the order."""

    def __init__(self, tool: str = "", schema: dict | None = None) -> None:
        self.tool = tool
        props = (schema or {}).get("properties")
        self.specs = props if isinstance(props, dict) else {}
        self.units: dict = {}
        self.pending: dict = {}
        self.put_form: str | None = None
        self.adds: bool | None = None
        self.last_view: Any = None
        self.used = False
        self.changed_at, self.viewed_at = 0, -1

    def form(self, args: dict) -> str:
        chosen = {}
        for k, v in args.items():
            spec = self.specs.get(k)
            if isinstance(spec, dict) and "enum" in spec:
                chosen[k] = v
        return as_text(chosen)

    def line_spec(self) -> tuple | None:
        for name, spec in self.specs.items():
            if not isinstance(spec, dict) or spec.get("type") != "array" or not isinstance(items := spec.get("items"), dict):
                continue
            fields = items.get("properties")
            if isinstance(fields, dict) and (pair := _f_line_fields(fields)) is not None:
                return (name,) + pair
        return None

    def _line_rows(self, args: dict) -> tuple | None:
        """(line spec, line objects) of a call whose array argument lists the lines; None for a flat call."""
        spec = self.line_spec()
        if not spec:
            return None
        rows = args.get(spec[0])
        return (spec, rows) if isinstance(rows, list) else None

    def lines(self, args: dict) -> list:
        if (listed := self._line_rows(args)) is None:
            target = self.aim_flat(args)
            if not target:
                return []
            return [(target, self.count_arg(args))]
        (_, id_key, count_key), rows = listed
        found = (_f_row_line(row, id_key, count_key) for row in rows)
        return [line for line in found if line is not None]

    def aim(self, args: dict) -> str | None:
        pick = _f_first_put(self.lines(args))
        return pick[0] if pick else None

    def aim_flat(self, args: dict) -> str | None:
        for k, v in args.items():
            if not isinstance(v, str) or not (word := v.strip()) or " " in word:
                continue
            spec = self.specs.get(k)
            if not isinstance(spec, dict) or "enum" not in spec:
                return v
        return None

    def count_arg(self, args: dict) -> float | None:
        if self._line_rows(args) is not None:
            pick = _f_first_put(self.lines(args))
            return pick[1] if pick else None
        for name, s in self.specs.items():
            if not isinstance(s, dict) or "enum" in s or s.get("type") not in ("integer", "number"):
                continue
            if numeric(v := args.get(name, s.get("default"))):
                return float(v)
        return None

    def is_put(self, args: dict) -> bool:
        if self.put_form is None:
            return True
        return self.form(args) == self.put_form

    def held(self, ref: str) -> float:
        return self.units.get(ref, 0.0)

    def _view_after(self, observation: Any) -> bool:
        """A cart view in the answer settles the count; True when there was one."""
        rows = self.view_units(observation)
        if not rows:
            return False
        self.learn(rows)
        return True

    def booked(self, args: dict, observation: Any) -> str:
        self.used = True
        if isinstance(observation, dict) and observation.get("error"):
            return ""
        self.put_form = self.form(args) if self.put_form is None else self.put_form
        if self._line_rows(args) is not None:
            # a list of lines states the whole basket: a line it leaves out is gone
            self.adds, self.units = False, {}
            for ref, n in self.lines(args):
                more = 1.0 if n is None else n
                if n is None or n > 0:
                    self.units[ref] = self.held(ref) + more
            self._view_after(observation)
            return self.summary()
        if (target := self.aim(args)) is None:
            return ""
        if (count := self.count_arg(args)) is None:
            count = 1.0
        prior, putting = self.held(target), self.is_put(args)
        if putting and prior > 0 and self.adds is not False:
            units = prior + count
            self.pending = {target: count}
        elif count > 0:
            units = count if putting else max(0.0, prior - count)
        else:
            units = 0
        if units <= 0:
            self.units.pop(target, None)
        else:
            self.units[target] = units
        if self._view_after(observation):
            return self.summary()
        if (count <= 0 or not putting) and self.shows_empty(observation, target):
            for book in (self.units, self.pending):
                book.pop(target, None)
        return self.summary()

    def shows_empty(self, observation: Any, ref: str) -> bool:
        if not isinstance(observation, dict) or ref not in self.units:
            return False
        lists, nums = [], []
        for v in observation.values():
            if isinstance(v, list):
                lists.append(v)
            elif numeric(v):
                nums.append(v)
        if not lists or not nums or any(lists) or any(nums):
            return False
        text = as_text(observation)
        return all(other not in text for other in self.units if other != ref)

    def view_units(self, observation: Any) -> dict | None:
        known = set(self.units) | set(self.pending)
        rows: dict = {}

        def visit(rec: dict) -> None:
            if (units := _f_units_of(rec)) is None:
                return
            refs = [v for v in rec.values() if _f_line_ref(v, known)]
            if not refs:
                return
            ref = ([c for c in refs if c in known] or refs)[0]
            rows[ref] = rows.get(ref, 0) + units
        each_dict(observation, visit)
        if rows and (not known or not known.isdisjoint(rows)):
            return rows
        return None

    def learn(self, rows: dict) -> None:
        for line, if_set in self.pending.items():
            if line not in rows:
                continue
            seen = rows[line]
            if abs(seen - if_set) < 1e-9 and abs(self.held(line) - if_set) > 1e-9:
                self.adds = False
            elif seen > if_set:
                self.adds = True
        self.pending = {}
        self.units = dict((r, n) for r, n in rows.items() if n > 0)

    def viewed(self, observation: Any, tick: int) -> None:
        self.last_view, self.viewed_at = observation, tick
        if self._view_after(observation):
            return
        if isinstance(observation, (dict, list)) and not self.ids_in(observation):
            self.units = {}

    def ids_in(self, observation: Any) -> bool:
        text = as_text(observation)
        for r in self.units:
            if r in text:
                return True
        return False

    @staticmethod
    def _n(x: float) -> str:
        whole = float(x).is_integer()
        return str(int(x) if whole else x)

    def _held_text(self) -> str:
        return ", ".join("%s x%s" % (r, self._n(n)) for r, n in self.units.items())

    def summary(self) -> str:
        return "[cart as counted] %s." % (self._held_text() if self.units else "empty")

    def faults(self, wanted: int, counts: list | None = None) -> list:
        if not self.units:
            return []
        lines = self._held_text()
        if counts and len(counts) > 1:
            if len(self.units) != len(counts):
                return ["the cart holds %s but the request asks for %d products, one line each - every product needs its "
                        "line and nothing else" % (lines, len(counts))]
            held = sorted(int(n) for n in self.units.values())
            if held != sorted(counts):
                return ["the cart lines hold %s units but the products need %s (one line per product at its quantity)"
                        % (held, sorted(counts))]
            return []
        if len(self.units) > 1:
            return ["the cart holds %s - take out every line but the chosen one" % lines]
        ref, n = next(iter(self.units.items()))
        if n == wanted:
            return []
        verb = "take out" if n > wanted else "put in"
        return ["the cart holds %s x%s but %d units are wanted - %s %s with the cart tool" % (
            ref, self._n(n), wanted, verb, self._n(abs(n - wanted)))]


# ------------------------------------------------------------------------------------------------------------ the frame
G_READ_CAP = 3000           # answer tokens of one request reading
G_READ_PASSES = 3           # passes over the reader ladder at most
G_HINTS_KEPT = 12           # story hints kept after the reading
G_CHECK_CAP = 3000          # answer tokens of an item check
G_RANK_CAP = 2500           # answer tokens of a rank check
G_AUDIT_CAP = 6000          # answer tokens of a decline audit
G_RANK_CARDS = 20000        # characters of listing cards shown to the rank check
G_AUDIT_CARDS = 16000       # characters of listing cards shown to the decline audit
G_VIEW_CUT = 3000           # characters of the newest cart view shown to the item check
G_SHORT_VALUE = 40          # a combined finish tool reads its mode from string arguments this short
G_NEVER = 10 ** 9           # read count of "no audit yet" (any count is below it)
G_POINTER_BITS = ("as shown", "same as", "the page", "sold by", "redirect", "refer")
G_FRAME_WORDS = {"which", "would", "you", "like", "for", "the"}
G_QUIET_NOTE = "drew no preference)"


def _g_flat(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _g_clip(text: Any, n: int) -> str:
    plain = " ".join(str(text).split())
    return plain if len(plain) <= n else plain[:n - 3] + "..."


def _g_unmarked(value: Any) -> str:
    return " ".join(re.sub(r"\(stated\)|\(revealed[^)]*\)", " ", str(value or "")).split())


def _g_points(value: Any) -> bool:
    low = str(value or "").lower()
    return points_elsewhere(value) or any(bit in low for bit in G_POINTER_BITS)


def _g_menu_key(line: Any) -> str:
    return str(line or "").strip("- ").split(" (in ", 1)[0].strip()


def _g_tags(text: Any) -> list:
    return re.findall(r"\(([^)]*)\)", str(text or ""))


class FrameMixin:
    """The errand's own state and frame: status line, request reading, the question-menu helpers, the step reserve,
    the item / rank / decline checks with their field evidence, scripted detail reads and joined categories."""

    # ================================================================================================ construction
    def _frame_setup(self, payload: dict) -> None:
        env = payload["environment"]
        view = env["policy_view"]
        self.wire = Link(env["binding"])
        self.pview = view
        self.request = str(view.get("query") or "")
        self.toolset = [t for t in (view.get("tools") or []) if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.shapes = {}
        for tool in self.toolset:
            fn = tool["function"]
            self.shapes[fn["name"]] = fn.get("parameters") or {"type": "object", "properties": {}}
        self.step_cap = int(view.get("max_steps") or 30)
        self.wire.done = bool(view.get("finished"))
        self.archive = Archive()
        self.needs = Wishes(self.wire, self.request)
        self.needs.ref_hook = self.ref_fix
        self.needs.field_hook = self.store_fields
        self.role_map = {}
        self.trolley = Basket()
        self.msgs_sent = 0
        self.msg_cap = None
        self.orders_tried = 0
        self.refusal_n = 0
        self.order_vetted = False
        self.rank_checked = set()
        self.decline_vetted = False
        self.audit_done = False
        self.questions_log = []
        self.asking = True
        self.rulings = {}
        self.checked_adds = set()
        self.passed = set()
        self.refused_lines = set()
        self.refusals_seen = {}
        self.warned = set()
        self.word_seen = []
        self.word_kind = ""
        self.lead_seen = []
        self.lead_bounds = {}
        self.lead_words = 0
        self.lead_ctx = None
        self.last_calls = []
        self.chat = []
        self.stuck = 0
        self.refs = []
        SAID_VALUES.clear()
        # created on demand by the methods of this part
        self.read_content = None
        self.read_at = None                 # None: the plan did not come from code
        self.steps_given = self.step_cap
        self.menu_memo = {}
        self.hint_alts = {}
        self.twin_asks = set()
        self.first_reads_done = False
        self.open_queue = None
        self.open_recheck = False
        self.open_say = None
        self.open_purpose = None
        self.open_floor = None
        self.open_tool = None
        self.late_q = 0
        self.late_ids = set()
        self.backup_n = 0
        self.swap_done = False
        self.pass_price = {}
        self.ch_rows_memo = None
        self.ch_view_memo = {}
        self.rank_better = []
        self.named = set()
        self.named_list = []
        self.named_pages = set()
        self.audits = 0
        self.audit_reads = G_NEVER
        self.audit_ids = None               # None: no audit ran yet
        self.audit_fits = []
        self.batch_proven = False
        self.last_reads = []
        self.covering = False
        self.home_memo = None
        self.joined_seen = {}

    # ================================================================================================ small reads
    @property
    def log(self) -> list:
        return self.wire.trace

    def moves_left(self) -> int:
        return self.step_cap - self.wire.tick

    def role(self, name: str) -> str | None:
        return self.role_map.get(name)

    def ask_limit(self) -> int:
        return self.msg_cap if self.msg_cap is not None else Knobs.ASK_ASSUMED

    def call_kind(self, tool: Any, args: Any) -> str | None:
        """"order", "decline", or None for any other call."""
        buy, quit_ = self.role("order"), self.role("decline")
        if not tool or tool not in (buy, quit_):
            return None
        if buy != quit_:
            return "order" if tool == buy else "decline"
        for value in (args.values() if isinstance(args, dict) else []):
            if isinstance(value, str) and len(value) <= G_SHORT_VALUE:
                low = value.lower()
                if DECLINE_SENSE.search(low):
                    return "decline"
                if ORDER_SENSE.search(low):
                    return "order"
        return "order"

    def finishers(self) -> set:
        return {t for t in (self.role("order"), self.role("decline")) if t}

    def clock_text(self) -> str:
        left = max(0.0, self.wire.left() - Knobs.FINAL_S)
        if left > (Knobs.BUDGET_S - Knobs.FINAL_S) / 2:
            return ""
        whole = int(left // Knobs.CLOCK_ROUND) * Knobs.CLOCK_ROUND
        return "roughly %d s left" % whole if whole > 0 else "less than %d s left" % Knobs.CLOCK_ROUND

    def _fr_several(self) -> bool:
        return bool(self.needs.items())

    def _fr_product_name(self, k: Any) -> str:
        rows = self.needs.items()
        if rows and isinstance(k, int) and 0 <= k < len(rows):
            return rows[k]["product"]
        return str(self.needs.reading.get("product") or "")

    @staticmethod
    def _fr_named_row(row: Any) -> bool:
        """A reading item that is a dict with a product name."""
        if not isinstance(row, dict):
            return False
        return bool(str(row.get("product") or "").strip())

    def _fr_hints(self) -> list:
        return [str(h) for h in self.needs.reading.get("hints") or []]

    @staticmethod
    def _fr_tool_names(tools: list) -> list:
        return [t["function"].get("name") for t in tools]

    @staticmethod
    def _fr_halves(text: Any) -> tuple:
        """(head, value) of a 'head: value' text; the value is "" when there is no colon."""
        pieces = str(text).partition(":")
        return pieces[0], pieces[2]

    @staticmethod
    def _fr_bare(name: Any) -> str:
        """A label without its bracketed tail."""
        return str(name).partition("(")[0].strip()

    @staticmethod
    def _fr_core_words(name: Any) -> list:
        """The name's words of three letters or more that are not filler, in order."""
        out = []
        for w in name_words(name):
            if len(w) >= 3 and w not in FILLER:
                out.append(w)
        return out

    # ================================================================================================ the status line
    def status_line(self) -> str:
        left = self.moves_left()
        head = ["%d of %d steps used, %d left" % (self.wire.tick, self.step_cap, left)]
        sent = "%s message(s) sent" % self.msgs_sent
        if self.msg_cap is not None:
            sent += " of %s allowed" % self.msg_cap
        head.append(sent)
        clock = self.clock_text()
        if clock:
            head.append(clock)
        out = ["[status] " + "; ".join(head),
               "[requirements now] " + self.needs.text_body() + self.ref_line() + self.product_steps()]
        usable = self._fr_tool_names(self.offered())
        if len(usable) < len(self.toolset):
            line = "[tools usable now] %s - a call to any other tool is refused and costs no step" % ", ".join(
                str(n) for n in usable)
            if self.asking:
                line += "; the code plans and sends the questions to the shopper"
            out.append(line)
        if self.trolley.used:
            out.append(self.trolley.summary())
        urgent = self._fr_urgency(left)
        if urgent:
            out.append(urgent)
        return "\n".join(out)

    def _fr_urgency(self, left: int) -> str:
        several = self._fr_several()
        if left <= 2:
            return ("[LAST STEPS] Order the cart now if it holds a candidate for %s. If it does not, put the best "
                    "candidate in - a verified fit, else the cheapest in-stock listing within the budget whose record shows "
                    "no requirement failing - and order. Decline only when the records show nothing can fit."
                    % ("every product asked for" if several else "the item"))
        if left <= 5 or self.wire.left() < Knobs.WIND_DOWN_S:
            return ("[FINISH NOW] Make the cart hold %s, chosen the same way (a verified fit, else the cheapest in-stock "
                    "listing within the budget whose record shows no requirement failing), with any sold-out line taken "
                    "out; read the stock again; order. Decline only when the records show nothing can fit."
                    % ("one line per product" if several else "one line"))
        if self.wire.spent_out():
            return ("[MODEL BUDGET USED] Stop exploring. Within a few steps put %s in the cart - a verified fit where one "
                    "exists, else the closest fit the records show - read stock and price again, and order. A decline is "
                    "right only when the records show %s has no listing that can meet its requirements."
                    % ("the best candidate of every product (one line each at its quantity)" if several else "the best "
                       "candidate", "some product (or the store)" if several else "the store"))
        if self.orders_tried == 0 and self.narrowing():
            due = self.order_due()
            if due < left <= due + 7:
                return ("[STEPS RUNNING LOW] Within a few steps put the best verified fit in the cart and order. Searches "
                        "and detail reads stop once %d steps are left; a refused order needs about 4 steps to repair."
                        % due)
            if left <= due:
                return ("[SEARCHING HAS ENDED] Details are read now only for the cart's line, for listings a check named, "
                        "or - while some product has no verified fit - for up to 3 more reads or searches in all. Put the "
                        "best verified fit in the cart, read its stock and price again, and order; a refused order still "
                        "needs about 4 steps to repair.")
        return ""

    # ================================================================================================ reading the request
    def read_request(self) -> None:
        listing = list(map(self._fr_tool_line, self.toolset))
        fixed = READER_TEXT + "\n\nTHE STORE'S TOOLS:\n" + "\n".join(listing)
        opening = self.pview.get("opening")
        varying = "THE REQUEST:\n" + self.request
        if opening is not None:
            varying += "\n\nWHAT THE STORE SHOWED FIRST:\n" + as_text(opening)
        self.read_content = [dict(type="text", text=fixed, cache_control=dict(type="ephemeral")),
                             dict(type="text", text=varying)]
        wire = self.wire
        ladder = list(LINEUP["reader"]) + LINEUP["listener"][:1]
        reading = partial = None
        for round_no, model in ((r, m) for r in range(G_READ_PASSES) for m in ladder):
            if round_no > 0 and not self._fr_retry_ok(ladder):
                break
            reply = wire.ask_model([dict(role="user", content=self.read_content)], [model], G_READ_CAP)
            got = first_json(reply.get("content")) if reply else None
            partial = got if isinstance(got, dict) else None
            if isinstance((partial or {}).get("roles"), dict):
                reading = partial
                break
            self.log.append(dict(kind="setup_retry", model=model))
        if reading is None:
            reading = self._fr_logged("code_reading_error", self._fr_code_reading, partial, fallback={})
            self.read_at = monotonic()
            items = reading.get("items")
            self.log.append(dict(kind="code_reading", roles=reading.get("roles"), product=reading.get("product"),
                                 price_cap=reading.get("price_cap"), items=len(items) if isinstance(items, list) else 0))
        self.use_reading(reading)

    @staticmethod
    def _fr_tool_line(tool: dict) -> str:
        fn = tool["function"]
        about = fn.get("description") or ""
        return "- %s: %s | arguments: %s" % (fn.get("name"), about, as_text(fn.get("parameters") or {}))

    def _fr_retry_ok(self, ladder: list) -> bool:
        """Another pass over the reader ladder is worth it: calls failed, none refused, and time remains."""
        wire = self.wire
        if wire.misses <= 0 or wire.call_fail == "refused" or wire.down():
            return False
        return bool(wire.usable(ladder)) and wire.left() >= Knobs.WIND_DOWN_S + 60

    def _fr_code_reading(self, partial: Any) -> dict:
        """The code's own reading of the request, with what a model's partial answer adds (its roles left out)."""
        reading = self.code_reading()
        for key, value in (partial or {}).items():
            if key != "roles":
                reading[key] = value
        return reading

    def _fr_scale(self) -> None:
        try:
            self.wire.scale = 1.0 + 0.3 * min(2, max(0, len(self.needs.items()) - 1))
        except Exception:  # noqa: BLE001
            self.wire.scale = 1.0

    def use_reading(self, reading: Any, again: bool = False) -> None:
        reading = reading if isinstance(reading, dict) else {}
        needs = self.needs
        old_cart = self.role("cart")
        roles = reading.get("roles")
        if not again:
            self.steps_given = self.step_cap
        self.role_map = {}
        if isinstance(roles, dict):
            for r in roles:
                tool = roles[r]
                if isinstance(tool, str) and tool in self.shapes:
                    self.role_map[r] = tool
        keep_cart = again and self.trolley.used
        if not keep_cart:
            cart = self.role("cart") or ""
            self.trolley = Basket(cart, self.shapes.get(cart) or {})
        elif old_cart:
            self.role_map["cart"] = old_cart
        search_ok = bool(self.role("search"))
        needs.sweepable = search_ok and self.page_arg() is not None
        needs.reading = reading
        self._fr_scale()
        budget = reading.get("message_budget")
        cap_ok = numeric(budget) and budget >= 0
        if cap_ok:
            self.msg_cap = int(budget)
        named, loose = [], []
        for req in map(str, filter(None, reading.get("requirements") or [])):
            head, value = self._fr_halves(req)
            if head.strip() and value.strip():
                named.append(req)
            else:
                loose.append(req)
        needs.written = list(named)
        if loose:
            soft = list(map(str, filter(None, reading.get("soft_wants") or [])))
            reading["soft_wants"] = soft + loose
            reading["requirements"] = list(named)
        self._fr_logged("story_values_error", self.story_values, reading)
        self._fr_logged("ref_register_error", self.ref_register)
        if not again or needs.version <= 0:
            needs.spend_cap = reading.get("price_cap")
        if reading and needs.spend_cap is None and self.role("message"):
            asks = list(map(str, filter(None, reading.get("ask_first") or [])))
            reading["ask_first"] = ["the most the shopper is willing to spend in all (ask it only once if the same "
                                    "is listed again below)"] + asks
        limit = reading.get("step_budget")
        cap = self.steps_given if again else self.step_cap
        if numeric(limit) and 0 < limit < cap:
            cap = int(limit)
        self.step_cap = cap
        if again:
            needs.version += 1

    def _fr_logged(self, kind: str, call: Any, *args: Any, fallback: Any = None) -> Any:
        """call(*args); on an error log it under `kind` and give `fallback`."""
        try:
            return call(*args)
        except Exception as error:  # noqa: BLE001
            self.log.append({"kind": kind, "error": str(error)[:300]})
        return fallback

    def story_values(self, reading: dict) -> None:
        hints = list(map(single_prop_hint, self._fr_hint_texts(reading.get("hints"))))
        own = list(hints)
        listed = reading.get("references")
        refs = list(map(str, listed)) if isinstance(listed, list) else []
        needs = self.needs
        written = needs.written
        moved = []
        for w in written:
            if unwritten_words(w, self.request, refs):
                moved.append(w)
        if moved and len(written) >= 2 and 2 * len(moved) <= len(written):
            kept = [w for w in written if w not in moved]
            needs.written = kept
            reading["requirements"] = list(kept)

            def base(req: Any) -> tuple:
                head, value = self._fr_halves(req)
                return self._fr_bare(head).lower(), value.strip().lower()
            gone = set(map(base, moved))
            for row in reading.get("items") or []:
                reqs = row.get("requirements") if isinstance(row, dict) else None
                if isinstance(reqs, list):
                    row["requirements"] = [r for r in reqs if base(r) not in gone]
            for m in moved:
                head, value = self._fr_halves(m)
                hints.append("%s: the story suggests %s (not stated as a value; ask)" % (head.strip(), value.strip()))
            self.log.append(dict(kind="story_values", moved=moved))
        turned = self._fr_logged("written_hints_error", self.written_hints, reading, own, fallback=[])
        hints = [h for h in hints if h not in turned]
        reading["hints"] = hints[:G_HINTS_KEPT]

    @staticmethod
    def _fr_hint_texts(raw: Any) -> list:
        """The stripped texts of the reading's hints; a dict of property -> value(s) gives 'property: value' texts."""
        if isinstance(raw, dict):
            pieces = []
            for prop in raw:
                val = raw[prop]
                for one in val if isinstance(val, list) else [val]:
                    if one:
                        text = str(one)
                        pieces.append("%s: %s" % (prop, text) if ":" not in text else text)
            raw = pieces
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(h).strip() for h in raw if isinstance(h, (str, int, float)) and str(h).strip()]

    def written_hints(self, reading: dict, hints: list) -> list:
        moved: list = []
        needs = self.needs
        rows = [r for r in (reading.get("items") or []) if isinstance(r, dict)]
        single = len(rows) == 1
        for hint in hints:
            text = str(hint)
            if ":" not in text:
                continue
            head, detail = self._fr_halves(text)
            prop = self._fr_bare(head)
            if not prop or not_a_property(prop):
                continue
            # the product named in the hint's own tag, else every product of the reading
            names = _g_tags(head)[-1:] or [str(r.get("product") or "") for r in rows or [reading]]
            for used in names:
                word = leading_value(detail, self.request, used)
                if word:
                    break
            else:
                continue
            mine = kind_set(used)
            if self._fr_hint_clash(prop, word, mine):
                continue
            new = "%s: %s" % (head.strip(), word)
            needs.written.append(new)
            reading["requirements"] = list(needs.written)
            for row in rows:
                reqs = row.get("requirements")
                if isinstance(reqs, list) and (single or mine & kind_set(row.get("product") or "")):
                    reqs.append(new)
                    break
            moved.append(hint)
            self.log.append(dict(kind="written_hint", hint=hint, requirement=new))
        return moved

    def _fr_hint_clash(self, prop: str, word: str, mine: set) -> bool:
        """A written requirement of the same product already names the property or the value."""
        pattern = r"\b%s\b" % re.escape(word)
        for req in self.needs.written:
            rhead, rvalue = self._fr_halves(req)
            rtags = _g_tags(rhead)
            if rtags and mine and not mine & kind_set(rtags[-1]):
                continue
            if loosely_same(rhead, prop) or re.search(pattern, rvalue.lower()):
                return True
        return False

    # ================================================================================================ question-menu helpers
    def _fr_menu(self, k: Any) -> str:
        count = len(self.archive.items)
        memo = self.menu_memo.get(k)
        if memo and memo[0] == count:
            return memo[1]
        menu = ""
        if self._fr_several():
            homes = self.home_shelves()
            home = homes[k] if isinstance(k, int) and 0 <= k < len(homes) else set()
            for label in self.archive.shelves_read():
                if label.strip().lower() in home:
                    menu = self.archive.prop_menu(label)[0]
                    break
        else:
            own = self.own_shelf()
            menu = self.archive.prop_menu(own)[0] if own else self.archive.prop_menu()[0]
        self.menu_memo[k] = (count, menu)
        return menu

    def _fr_menu_lines(self, k: Any) -> list:
        return [line for line in self._fr_menu(k).splitlines() if line.strip()]

    def open_hints(self, k: Any = None) -> list:
        alts, keep = [], []
        several = self._fr_several()
        for hint in self._fr_hints():
            prop = hint.split(":", 1)[0].strip()
            if not prop:
                continue
            j = self.product_of(prop) if several else None
            if k is not None and j is not None and j != k:
                continue
            if self.open_prop(j, prop, self.prop_line(j, prop), by_samples=True) and not self.asked_inside(j, prop):
                keep.append(hint)
                continue
            alt = self.hint_alt(j, hint, prop)
            if alt:
                alts.append(alt)
        return alts + keep

    def prop_line(self, k: Any, prop: Any) -> str:
        try:
            for line in self._fr_menu_lines(k):
                key = _g_menu_key(line)
                if key and strictly_same(key, prop):
                    return line
            return ""
        except Exception:  # noqa: BLE001
            return ""

    def field_values(self, k: Any, prop: Any) -> set:
        line = self.prop_line(k, prop)
        if "):" not in line:
            return set()
        return {v for v in (_g_flat(part) for part in line.split("):", 1)[1].split("|")) if v}

    def same_field(self, k: Any, a: Any, b: Any) -> bool:
        try:
            if strictly_same(a, b):
                return True
            return loosely_same(a, b) and bool(self.field_values(k, a) & self.field_values(k, b))
        except Exception:  # noqa: BLE001
            return False

    def hinted_prop(self, k: Any, prop: Any) -> bool:
        name = str(prop).split("(")[0].strip()
        several = self._fr_several()
        for hint in self._fr_hints():
            head = hint.split(":", 1)[0]
            if not strictly_same(head.split("(")[0], name):
                continue
            if k is None or not several or self.product_of(head) in (k, None):
                return True
        return False

    def product_reads(self, k: Any) -> int:
        try:
            if self._fr_several():
                homes = self.home_shelves()
                home = homes[k] if isinstance(k, int) and 0 <= k < len(homes) else set()
                return sum(self.archive.rich_reads(label) for label in home)
            own = self.own_shelf()
            return self.archive.rich_reads(own) if own else self.archive.rich_reads()
        except Exception:  # noqa: BLE001
            return 0

    def plain_hint(self, k: Any, prop: Any, hinted: bool = False) -> bool:
        try:
            name = str(prop or "").split("(")[0].strip()
            if not name or not (hinted or self.hinted_prop(k, prop)):
                return False
            if self.prop_line(k, prop) or self.product_reads(k) < 2:
                return False
            return not any(loosely_same(_g_menu_key(line), name) for line in self._fr_menu_lines(k) if _g_menu_key(line))
        except Exception:  # noqa: BLE001
            return False

    def hint_mark(self, hint: str) -> str:
        try:
            head = str(hint).split(":", 1)[0].strip()
            k = self.product_of(head) if self._fr_several() else None
            return "%s [no key]" % hint if self.plain_hint(k, head, hinted=True) else str(hint)
        except Exception:  # noqa: BLE001
            return str(hint)

    @staticmethod
    def hint_alts_of(detail: Any) -> list:
        out = []
        for group in re.findall(r"\(or ([^()]+)\)", str(detail)):
            for piece in re.split(r",|/|\bor\b", group):
                if piece.strip():
                    out.append(piece.strip())
        return out

    @staticmethod
    def name_inside(a: Any, b: Any) -> bool:
        wa = {w for w in name_words(a) if len(w) >= 3 and w not in FILLER}
        wb = {w for w in name_words(b) if len(w) >= 3 and w not in FILLER}
        return bool(wa) and bool(wb) and (wa <= wb or wb <= wa)

    @staticmethod
    def _fr_log_parts(line: Any) -> tuple:
        """(label, asked text) of a questions-log line; the asked text is "" when it cannot be found."""
        text = str(line or "")
        label = text.split(":", 1)[0].strip("- ")
        found = re.search(r'asked "(.*?)" -> ', text)
        return label, (found.group(1) if found else "")

    def asked_inside(self, k: Any, head: Any) -> bool:
        try:
            several = self._fr_several()
            for line in self.questions_log:
                label, asked = self._fr_log_parts(line)
                if "[not answered]" in label:
                    continue
                for part in label.split(";"):
                    j = self._fr_part_product(part, asked, several)
                    if (k is None or j is None or not several or j == k) and self.name_inside(part, head):
                        return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def _fr_part_product(self, part: str, asked: str, several: bool) -> int | None:
        """The product a questions-log label part is about, else the one its asked text names (several products)."""
        if not several:
            return None
        j = self.product_of(part)
        if j is None and asked:
            j = self.product_of(asked)
        return j

    def _fr_fits_product(self, text: Any, k: Any, several: bool) -> bool:
        """With several products: the text names product k or none of them."""
        return not several or self.product_of(text) in (k, None)

    def _fr_near_field(self, k: Any, a: Any, b: Any) -> bool:
        return self.same_field(k, a, b) or self.name_inside(a, b)

    def hint_alt(self, k: Any, hint: str, head: str) -> str:
        try:
            several = self._fr_several()
            detail = self._fr_halves(hint)[1]
            if hint not in self.hint_alts:
                quiet = [q for q in self.quiet_asks
                         if self._fr_near_field(k, q, head) and self._fr_fits_product(q, k, several)]
                if not quiet:
                    return ""
                for alt in self.hint_alts_of(detail):
                    line = "" if not_a_property(alt) or strictly_same(alt, head) else self.prop_line(k, alt)
                    if line:
                        self.hint_alts[hint] = _g_menu_key(line)
                        break
            key = self.hint_alts.get(hint)
            if not (key and key.lower() not in self.spent_asks and self.plan_open(k, key)):
                return ""
            tag = re.search(r"\(([^()]*)\)\s*$", self._fr_halves(hint)[0])
            tag_text = " (%s)" % tag.group(1).strip() if tag else ""
            bare = re.sub(r"\s*\(or [^()]+\)", "", detail)
            return "%s%s: %s (asked first as %s, which %s" % (key, tag_text, bare.strip(), self._fr_bare(head),
                                                             G_QUIET_NOTE)
        except Exception:  # noqa: BLE001
            return ""

    def hint_mapped(self, k: Any, prop: Any) -> bool:
        try:
            name = self._fr_bare(prop or "")
            quals = self._fr_core_words(name)
            scoped = self._fr_several() and k is not None
            for hint in self._fr_hints():
                head, detail = self._fr_halves(hint)
                if scoped and self.product_of(head) not in (k, None):
                    continue
                names = [self._fr_bare(head)] + self.hint_alts_of(detail)
                if any(strictly_same(h, name) for h in names if h):
                    return True
                if quals and self._fr_starts_all(quals, hint.lower()):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _fr_starts_all(words: list, text: str) -> bool:
        """Every word starts some word of the text."""
        for w in words:
            if not re.search(r"\b" + re.escape(w), text):
                return False
        return True

    @staticmethod
    def name_twin(a: Any, b: Any) -> bool:
        if strictly_same(a, b):
            return False

        def heads(words: list) -> list:
            return words if len(words) < 2 else words[:-1]

        def near(x: str, y: str) -> bool:
            if x == y:
                return True
            return len(x) >= 4 and len(y) >= 4 and (x.startswith(y) or y.startswith(x))
        words_a = FrameMixin._fr_core_words(a)
        words_b = FrameMixin._fr_core_words(b)
        if any(near(h, w) for h in heads(words_a) for w in words_b):
            return True
        return any(near(w, h) for w in words_a for h in heads(words_b))

    def twin_of(self, k: Any, a: Any, b: Any) -> bool:
        try:
            return not strictly_same(a, b) and (self.name_twin(a, b) or self.same_field(k, a, b))
        except Exception:  # noqa: BLE001
            return False

    def twin_key(self, k: Any, prop: Any) -> str:
        try:
            several = self._fr_several()
            menu = self._fr_menu_lines(k)
            own = kind_list(self._fr_product_name(k))
            mine_hints = [h.lower() for h in self._fr_hints() if self._fr_fits_product(self._fr_halves(h)[0], k, several)]
            told = " ".join([self.request.lower()] + mine_hints)
            prop_words = set(name_words(prop))
            for key in map(_g_menu_key, menu):
                if not self._fr_fresh_key(key):
                    continue
                if self.twin_of(k, key, prop) and self.plan_open(k, key) and not self.name_inside(key, prop):
                    rest = [w for w in self._fr_core_words(key) if w not in prop_words and not kind_found(w, own)]
                    if any(re.search(r"\b" + re.escape(w), told) for w in rest) or self.hint_mapped(k, key):
                        return key
            return ""
        except Exception:  # noqa: BLE001
            return ""

    def _fr_fresh_key(self, key: Any) -> bool:
        """A menu field that names a property and has not been asked about."""
        return bool(key) and not not_a_property(key) and key.lower() not in self.spent_asks

    def values_form(self, k: Any, prop: Any, product: str) -> str:
        try:
            line = self.prop_line(k, prop)
            if "):" not in line:
                return ""
            elsewhere: set = set()
            for other in self._fr_menu_lines(k):
                if other.strip() != line.strip() and "):" in other:
                    elsewhere |= {_g_flat(v) for v in other.split("):", 1)[1].split("|")}
            kept, seen = [], set()
            for v in line.split("):", 1)[1].split("|"):
                v = v.strip()
                flat = _g_flat(v)
                if not v or len(v) > 30 or not re.search(r"[a-z]", v.lower()) or re.search(r"[,;/]", v):
                    continue
                if flat in elsewhere or flat in seen:
                    continue
                kept.append(v)
                seen.add(flat)
                if len(kept) == 3:
                    break
            if len(kept) < 2:
                return ""
            return "Would you like %s or %s for the %s?" % (", ".join(kept[:-1]), kept[-1], product)
        except Exception:  # noqa: BLE001
            return ""

    def _fr_slot(self, prop: Any) -> tuple:
        """(several, k, budget slot) of a label; slot None = no product could be told."""
        several = self._fr_several()
        k = self.product_of(prop) if several else None
        if several and k is None:
            return several, k, None
        return several, k, (k if several else -1)

    def distinct_ask(self, prop: Any) -> tuple | None:
        try:
            several, k, slot = self._fr_slot(prop)
            if slot is None or slot in self.twin_asks:
                return None
            product = str(self.needs.reading.get("product") or "item") if not several else self._fr_product_name(k)
            target = self._fr_bare(prop)
            text = self.values_form(k, target, product)
            if not text:
                twin = self.twin_key(k, prop)
                if not twin:
                    return None
                target = twin
                text = self.values_form(k, twin, product) or "Which %s would you like for the %s?" % (twin, product)
            self.twin_asks.add(slot)
            label = target + (" (%s)" % product if several else "")
            return label, text
        except Exception:  # noqa: BLE001
            return None

    def asked_for(self, k: Any, prop: Any) -> bool:
        try:
            name = self._fr_bare(prop or "")
            if not name:
                return False
            several = self._fr_several()
            for line in self.questions_log:
                label, asked = self._fr_log_parts(line)
                pieces = re.split(r";|\band\b", re.sub(r"\s*\[[^\]]*\]", "", label))
                for part in filter(None, map(str.strip, pieces)):
                    j = self._fr_part_product(part, asked, several)
                    if (k is None or not several or j == k) and strictly_same(self._fr_bare(part), name):
                        return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def alt_key(self, k: Any, prop: Any) -> str:
        try:
            name = self._fr_bare(prop)
            if not name:
                return ""
            for _, head, detail in self._fr_product_hints(k):
                if self._fr_near_field(k, name, head):
                    key = self._fr_free_alt(k, name, head, detail)
                    if key:
                        return key
            return ""
        except Exception:  # noqa: BLE001
            return ""

    def _fr_product_hints(self, k: Any) -> Any:
        """(head, bare head, detail) of each story hint with a head that may belong to product k, lazily."""
        several = self._fr_several()
        for hint in self._fr_hints():
            whole, detail = self._fr_halves(hint)
            head = self._fr_bare(whole)
            if head and self._fr_fits_product(whole, k, several):
                yield whole, head, detail

    def _fr_free_alt(self, k: Any, name: str, head: str, detail: str) -> str:
        """The menu field of a hint's '(or ...)' alternative that is still free to ask, else ""."""
        for alt in self.hint_alts_of(detail):
            if not_a_property(alt) or any(strictly_same(alt, x) for x in (head, name)) or self.name_inside(alt, name):
                continue
            line = self.prop_line(k, alt)
            if not line:
                continue
            key = _g_menu_key(line)
            if key and not strictly_same(key, name) and key.lower() not in self.spent_asks \
                    and not self.asked_for(k, key) and self.plan_open(k, key):
                return key
        return ""

    def collide_alt(self, prop: Any) -> str:
        try:
            several, k, slot = self._fr_slot(prop)
            if slot is None or slot in self.twin_asks:
                return ""
            key = self.alt_key(k, prop)
            if not key:
                return ""
            self.twin_asks.add(slot)
            return "%s (%s)" % (key, self._fr_product_name(k)) if several else key
        except Exception:  # noqa: BLE001
            return ""

    def holds_plain(self, text: Any, plain_q: Any) -> bool:
        def stem(w: str) -> str:
            return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        have = {stem(w) for w in re.findall(r"[a-z0-9]+", str(text or "").lower())}
        need = {stem(w) for w in re.findall(r"[a-z0-9]+", str(plain_q or "").lower())
                if w not in G_FRAME_WORDS and w not in FILLER}
        return bool(need) and need <= have

    def alt_untried(self, k: Any) -> bool:
        try:
            several = self._fr_several()
            for hint in self._fr_hints():
                whole = hint.split(":", 1)[0]
                head = whole.split("(")[0].strip()
                if not head:
                    continue
                if several and self.product_of(whole) not in (k, None):
                    continue
                if self.asked_for(k, head) and self.alt_key(k, head):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def quiet_reask(self, prop: Any) -> str:
        try:
            several, k, slot = self._fr_slot(prop)
            if slot is None or slot in self.twin_asks or not self.hint_mapped(k, prop):
                return ""
            if any(" " + G_QUIET_NOTE in h for h in self.open_hints(k)):
                return ""
            if several:
                messaged = set()
                for line in self.questions_log:
                    label, asked = self._fr_log_parts(line)
                    for text in (label, asked):
                        j = self.product_of(text) if text else None
                        if j is not None:
                            messaged.add(j)
                if any(i not in messaged for i in range(len(self.needs.items())) if i != k):
                    return ""
            twin = self.twin_key(k, prop)
            if not twin:
                return ""
            self.twin_asks.add(slot)
            return "%s (%s)" % (twin, self._fr_product_name(k)) if several else twin
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def menu_trim(menu: str, reads: int) -> str:
        if reads < 4 or not menu:
            return menu
        kept = [line for line in menu.splitlines() if not re.search(r"\(in [01] listings?\)", line)]
        return "\n".join(kept) if kept else menu

    def first_reads(self) -> bool:
        return self._fr_logged("first_reads_error", self._fr_first_plan, fallback=False)

    def _fr_first_plan(self) -> bool:
        """Plan the first detail reads: up to 2 listings of each product's category before any question."""
        if self.first_reads_done or self.batch_tool() or not self.role("details"):
            return False
        self.first_reads_done = True
        spare_asks = max(0, self.ask_limit() - self.msgs_sent)
        room = self.moves_left() - self.order_due() - 2 * spare_asks - 8
        several = self._fr_several()
        counts = self.needs.counts()
        homes = self.home_shelves()
        taken: list = []
        found = []
        for idx in range(len(self.needs.items()) or 1):
            home = homes[idx] if idx < len(homes) else set()
            have = self.product_reads(idx if several else None)
            if not home or have >= 2 or room <= 0:
                continue
            want = counts[idx] if idx < len(counts) else counts[-1]
            picks = self.menu_rows(home, 2 - have, taken, want)
            taken.extend(picks)
            if picks:
                found.append((have, idx, picks))
        found = sorted(found, key=lambda t: t[0])
        firsts = [p[0] for _, _, p in found if p]
        seconds = [p[1] for _, _, p in found if len(p) > 1]
        queue = (firsts + seconds)[:max(0, room)]
        if not queue:
            return False
        self.open_purpose = "menu"
        self.open_queue = queue
        self.open_say = "Before asking anything I open a few listings of each product to learn which fields they carry."
        self.open_recheck = True
        self.log.append(dict(kind="first_reads", ids=queue))
        return True

    def plan_open(self, k: Any, prop: Any) -> bool:
        try:
            hinted = self.hinted_prop(k, prop)
            return self.open_prop(k, prop, self.prop_line(k, prop) if hinted else "", by_samples=hinted)
        except Exception:  # noqa: BLE001
            return self.open_prop(k, prop)

    def settles(self, req: Any, samples: str = "") -> bool:
        text = str(req)
        head, value = self._fr_halves(text)
        if _g_points(value):
            return True
        if not names_the_item(value, head):
            return False
        needs = self.needs
        listed = needs.reading.get("references")
        refs = list(map(str, listed)) if isinstance(listed, list) else []
        refs += list(needs.resolved or [])
        told = "%s %s" % (self.request, " ".join(map(str, needs.exchanges)))
        if unwritten_words(text, told, refs):
            return False
        if not samples or re.search(r"\d", value):
            return True
        return self._fr_in_samples(value, samples)

    @staticmethod
    def _fr_in_samples(value: str, samples: str) -> bool:
        """Some word of the value (a stem, or one letter off) is carried by the samples; True when it has no word."""
        words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in FILLER]
        if not words:
            return True
        pool = set(re.findall(r"[a-z]+", samples))
        for w in words:
            stem = w[:-1] if len(w) > 4 else w
            if re.search(r"\b" + re.escape(stem), samples):
                return True
            if len(w) >= 4 and any(len(p) >= 4 and within_edits(p, w, 1) for p in pool):
                return True
        return False

    # ================================================================================================ tools and the reserve
    def order_due(self) -> int:
        cap = self.step_cap
        base = max(Knobs.LATE_STEPS, min(Knobs.ORDER_KEEP, cap // 4 + 1))
        due = max(base, min(16, cap // 3))
        if self.event_seen or (self.dead and set(self.trolley.units) & self.dead):
            try:
                due = max(due, 8 + max(1, len(self.needs.items())))
            except Exception:  # noqa: BLE001
                due = max(due, 9)
        return max(base, min(due, cap // 2))

    def narrowing(self) -> bool:
        return bool(self.role("order")) and bool(self.role("cart"))

    def _fr_reserve_on(self, spare: int) -> bool:
        return (spare <= self.order_due() and self.orders_tried == 0) or \
            (spare <= Knobs.LATE_STEPS and self.refusal_n > 0)

    def offered(self) -> list:
        spare = self.moves_left()
        buy, quit_, cart, details, search = map(self.role, ("order", "decline", "cart", "details", "search"))
        allow = self._fr_allowed(spare, buy, quit_, cart, details, search)
        tools = list(self.toolset)
        if allow is not None and buy and cart:
            tools = [t for t in tools if t["function"].get("name") in allow] or tools
        message = self.role("message")
        if self.asking and message:
            tools = [t for t in tools if t["function"].get("name") != message]
        return tools

    def _fr_allowed(self, spare: int, buy: Any, quit_: Any, cart: Any, details: Any, search: Any) -> set | None:
        """The tools still usable with `spare` steps left; None: no limit."""
        if spare <= 1:
            return {buy, quit_}
        if spare <= 3:
            return {buy, quit_, cart}
        if not self._fr_reserve_on(spare):
            return None
        allow = {buy, quit_, cart, self.role("cart_view"), self.role("view") or details, self.stock_tool()}
        if spare > 3 and self.swap_open():
            allow.add(details)
        if self.late_n < 3:
            if self._fr_late_read_ok(spare, self.named_open):
                allow.add(details)
            if self._fr_late_search_ok(spare, lambda: self.named_pages):
                allow.add(search)
        if spare >= 7 and self.backup_rows():
            allow.add(details)
        return allow

    def _fr_late_read_ok(self, spare: int, opened: Any) -> bool:
        """A late detail read is allowed: nothing verified yet, or a listing a check named (opened()) is unread."""
        if self.unverified() and spare > 5:
            return True
        return bool(opened()) and spare >= min(Knobs.LATE_STEPS + 2, self.order_due())

    def _fr_late_search_ok(self, spare: int, paged: Any) -> bool:
        """A late search is allowed: the requirements were just widened, a check named the page, or nothing is verified."""
        if spare <= 4:
            return False
        return bool(self.widened_late() or paged() or (self.unverified() and spare > 5))

    def in_reserve(self) -> bool:
        spare = self.moves_left()
        return self.narrowing() and spare > 3 and self._fr_reserve_on(spare)

    def unverified(self) -> bool:
        try:
            ver = str(self.needs.version)
            live = set()
            for key in self.passed:
                ident, _, at = key.rpartition("|")
                if at == ver and ident not in self.refused_lines and ident not in self.dead:
                    live.add(ident)
            if not live:
                return True
            if not self._fr_several():
                return False
            got = {self.shelf_known(i) for i in live} - {""}
            if not got:
                return False
            return any(home and not (home & got) for home in self.home_shelves())
        except Exception:  # noqa: BLE001
            return False

    def named_open(self) -> set:
        return {i for i in self.named if i in self.archive.ids and i not in self.archive.read
                and i not in self.refused_lines}

    def widened_late(self) -> bool:
        return self.needs.widened_at == self.needs.version and \
            self.wire.tick - self.needs.widened_tick <= self.order_due()

    def begin_batch(self) -> None:
        self.late_q = 0
        self.late_ids = set()

    def late_hold(self, tool: str, args: Any) -> str:
        details, search = map(self.role, ("details", "search"))
        kind = tool
        if tool not in (details, search) and details and tool in self.lookup_tools():
            kind = details
        if not kind or kind not in (details, search) or not self.in_reserve():
            return ""
        if not isinstance(args, dict):
            args = {}
        values = list(args.values())
        named = {v for v in values if isinstance(v, str)}
        reads = kind == details
        if reads and (named & set(self.trolley.units) or any(key.rpartition("|")[0] in named for key in self.passed)):
            return ""
        spare = self.moves_left() - self.late_q
        ok = False
        if self.late_n + self.late_q < 3:
            if reads:
                ok = self._fr_late_read_ok(spare, lambda: named & self.named_open())
            else:
                ok = self._fr_late_search_ok(spare, lambda: self._fr_page_key(args) in self.named_pages)
        backups: list = []
        backup_room = spare - 1 >= 6 and self.backup_n < 2
        if not ok and reads and backup_room:
            unread = [b for b in self.backup_rows() if b not in self.late_ids]
            backups = unread[:2 - self.backup_n]
            if any(b in named for b in backups):
                ok = True
                self.backup_n += 1
        if not ok and reads:
            swap = self.swap_open()
            if swap and swap in named and spare > 2:
                ok = True
                self.swap_done = True
        if ok:
            self.late_q += 1
            self.late_ids |= named
            return ""
        text = ("Not run (no step used): the steps left are kept for the order. Details are read now only for the cart's "
                "line, for a listing a check named, or - while nothing is verified - for up to 3 listings in all. Put the "
                "best verified fit in the cart, read its stock and price again, and order.")
        if reads and backup_room and backups:
            text += " These cheaper unread rows may still be read to back up the pick: %s." % ", ".join(backups)
        return text

    def backup_rows(self) -> list:
        if self.needs.priority() != "cheapest" or self.backup_n >= 2:
            return []
        if not (self.event_seen or self.dead or self.named_open()):
            return []
        found = self._fr_logged("backup_rows_error", self._fr_cheaper_unread, fallback=None)
        if found is None:
            return []
        return [pair[1] for pair in sorted(found)][:6]

    def _fr_cheaper_unread(self) -> list:
        """(price, id) of the open rows cheaper than the cheapest verified fit of their product."""
        found = []
        for k in self.cg_products():
            pool = self.ed_pool(k)
            prices = [e.get("price") for e in pool if e.get("verified") and e.get("fit")]
            tops = [p for p in prices if p is not None]
            if not tops:
                continue
            low = min(tops)
            for e in pool:
                if any(e.get(flag) for flag in ("read", "gone", "over", "pointer", "ruled", "clash")):
                    continue
                price = e.get("price")
                if price is not None and price < low:
                    found.append((price, e.get("id")))
        return found

    # ================================================================================================ checks
    def field_names(self) -> str:
        menu = self.archive.prop_menu()[0]
        if not menu:
            return ""
        return "FIELD NAMES ON THIS STORE'S LISTINGS (with how many read listings carry each):\n" + menu

    def _fr_request_block(self) -> str:
        text = "THE REQUEST:\n" + self.request
        changes = self.needs.changes()
        return text + ("\n" + changes if changes else "")

    def _fr_refused(self, memo: str, thing: str, aim_text: str, by: str, problems: list, ruling: str) -> str:
        """Store a ruling the code reached alone and log it as a failed item check."""
        self.rulings[memo] = ruling
        self.log.append(dict(kind="verify", item=thing, aim=aim_text, by=by, result=dict(fits=False, problems=problems)))
        return ruling

    @staticmethod
    def _fr_add(parts: list, text: Any, head: str = "") -> None:
        if text:
            parts.append(head + text)

    def _fr_pointed(self, loose: list, thing: str) -> list:
        """Up to 3 newest records of other read listings that carry a word of the untracked references."""
        words: set = set()
        for r in loose:
            text = str(r)
            value = text.partition(":")[2] if ":" in text else text
            words.update(re.findall(r"[a-z]{4,}", value.lower()))
        found: list = []
        if not words:
            return found
        for ident in sorted(self.archive.read):
            if len(found) == 3:
                break
            if ident == thing:
                continue
            newest = self.archive.about(ident, 1)
            last = newest[-1] if newest else ""
            if not last:
                continue
            low = last.lower()
            if any(w in low for w in words):
                found.append(last)
        return found

    def check_item(self, thing: str, aim_text: str, second: str = "") -> str:
        if self.wire.left() < Knobs.WIND_DOWN_S + 20:
            return ""
        key = "%s|%s" % (thing, self.needs.version)
        if key in self.near_ok:
            return ""
        listed = self.needs.reading.get("references")
        refs = list(map(str, filter(None, listed))) if isinstance(listed, list) else []
        own = self.own_redirects(thing)
        tail = []
        if refs or not all(r.get("status") == "resolved" for r in self.refs):
            tail.append(str(len(self.archive.read)))
        if second:
            tail.append("second")
        if own:
            tail.append("own%d" % len(own))
        memo = "|".join([key] + tail)
        known = self.rulings.get(memo)
        if known is not None:
            return known
        point = self.ref_pointer_note(thing)
        if point:
            return self._fr_refused(memo, thing, aim_text, "reference chain", ["reference chain"],
                                    "The item check says %s does NOT meet every requirement: %s Choose another listing."
                                    % (thing, point))
        records = self.archive.about(thing, 4)
        if not records:
            return ""
        evidence = self.field_view(thing)
        fails = evidence["fails"][:3]
        if fails:
            return self._fr_refused(memo, thing, aim_text, "field evidence", fails,
                                    "The item check says %s does NOT meet every requirement: %s. Choose another listing."
                                    % (thing, "; ".join(fails)))
        parts = [CHECK_TEXT, self._fr_request_block(), "REQUIREMENTS NOW: " + self.needs.text_body()]
        if self._fr_several():
            note = ("SEVERAL PRODUCTS: this item is one of the products asked for (its category decides which); judge it "
                    "only against that product's requirements and quantity.")
            held = self.trolley.held(thing)
            if held > 0:
                note += " The cart holds %s unit(s) of it now." % Basket._n(held)
            parts.append(note)
        self._fr_add(parts, self.ref_note(thing))
        loose = self.ref_untracked(refs, self.refs)
        if loose:
            self._fr_add(parts, "\n".join(self._fr_pointed(loose, thing)),
                         "RECORDS THE REQUEST MAY POINT AT (other read listings naming its words):\n")
        self._fr_add(parts, second)
        for req, prop, sentence in own:
            parts.append("NOTE ON '%s': this listing's notes say \"%s\", which may send its %s to another listing. If it "
                         "does, the grader takes the %s from that listing, so this item's own field does not count and the "
                         "requirement is met only when that other listing's record shows it; a listing named only as "
                         "another version sold elsewhere, or as what this item is not, changes nothing."
                         % (req, sentence, prop, prop))
        parts.append("THE ITEM %s (about to %s) - its records, oldest first:\n%s" % (thing, aim_text, "\n".join(records)))
        if self.trolley.last_view is not None:
            parts.append("THE NEWEST CART VIEW:\n" + as_text(self.trolley.last_view)[:G_VIEW_CUT])
        self._fr_add(parts, self.field_names())
        if evidence["lines"]:
            parts.append("FIELD EVIDENCE (worked out by code from the store's records):\n" + "\n".join(evidence["lines"]))
        self._fr_add(parts, self.archive.label_lines(),
                     "RECURRING LABELS (values and how often they occur) - the item's category label must be the one "
                     "matching the request's wording as a whole:\n")
        prompt = "\n\n".join(parts)
        answer = self.wire.ask_json(prompt, G_CHECK_CAP, LINEUP["checker"])
        self.log.append(dict(kind="verify", item=thing, aim=aim_text, result=answer))
        if answer and answer.get("fits") is False:
            faults = list(map(str, answer.get("problems") or []))[:5]
            ruling = ("The item check says %s does NOT meet every requirement: %s. Unless you are certain the check is "
                      "wrong, choose another fully verified fit." % (thing, "; ".join(faults) or "a requirement is not shown"))
            self.rulings[memo] = ruling
            return ruling
        self.rulings[memo] = ""
        if not answer:
            return ""
        self.passed.add(key)
        self.pass_price[thing] = self.archive.price(thing)
        return ""

    def ch_rows(self) -> dict:
        count = len(self.archive.items)
        if self.ch_rows_memo is not None and self.ch_rows_memo[0] == count:
            return self.ch_rows_memo[1]
        rows = self.code_rows()
        self.ch_rows_memo = (count, rows)
        return rows

    def store_fields(self, label: Any = "") -> dict:
        try:
            if self._fr_several():
                k = self.product_of(label)
                homes = self.home_shelves()
                within = homes[k] if isinstance(k, int) and 0 <= k < len(homes) else set()
            else:
                own = self.own_shelf()
                if own:
                    within = {own}
                else:
                    homes = self.home_shelves()
                    within = homes[0] if homes and homes[0] else set()
            out: dict = {}
            for row in self.ch_rows().values():
                fields = row.get("fields") or {}
                if not fields or (within and row.get("shelf") not in within):
                    continue
                for name, value in fields.items():
                    if not str(value).strip() or not_a_property(name):
                        continue
                    out.setdefault(str(name), []).append(str(value))
            return out
        except Exception:  # noqa: BLE001
            return {}

    def ch_redirected(self, thing: str, prop: Any) -> bool:
        try:
            for got in self.own_redirects(thing) or []:
                pieces = got if isinstance(got, (list, tuple)) else [got]
                if any(loosely_same(str(x).partition(":")[0], prop) for x in pieces[:2]):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return True

    def field_view(self, thing: str) -> dict:
        needs = self.needs
        key = (thing, len(self.archive.items), needs.version, "|".join(needs.props_left()))
        if key not in self.ch_view_memo:
            self.ch_view_memo[key] = self._fr_logged("field_facts_error", self.field_facts, thing,
                                                     fallback=dict(lines=[], rules=[], fails=[]))
        view = self.ch_view_memo[key]
        said = self.near_said.get("%s|%s" % (thing, needs.version))
        if not said:
            return view
        try:
            return self._fr_confirmed_view(view, [entry[0] for entry in said])
        except Exception:  # noqa: BLE001
            return view

    @staticmethod
    def _fr_confirmed_view(view: dict, props: list) -> dict:
        """The view with the properties a meaning check confirmed marked as met and dropped from its failures."""
        def confirmed(name: Any) -> bool:
            return any(strictly_same(name, p) or loosely_same(name, p) for p in props)
        note = (" A meaning check confirmed this listing's wording as the value (the same value in other "
                "words): it meets it.")
        lines = []
        for line in view["lines"]:
            m = re.match(r"- (.+?) = '", line)
            lines.append(line + note if m and confirmed(m.group(1)) else line)
        fails = [f for f in view["fails"] if not confirmed(f.partition(":")[0])]
        return dict(lines=lines, rules=list(view["rules"]), fails=fails)

    def field_facts(self, thing: str) -> dict:
        lines: list = []
        rows = self.ch_rows()
        rules: list = []
        fails: list = []
        result = {"lines": lines, "rules": rules, "fails": fails}
        me = rows.get(thing) or {}
        shelf = me.get("shelf") or ""
        mine_fields = me.get("fields")
        if not mine_fields:
            return result
        several = self._fr_several()
        if several and not shelf:
            return result
        k = None
        if several:
            homes = self.home_shelves()
            k = next((idx for idx in range(len(homes)) if shelf in homes[idx]), None)
        peers = []
        for rid in rows:
            row = rows[rid]
            if row.get("fields") and (not shelf or row.get("shelf") == shelf):
                peers.append((rid, row))
        if len(peers) < 3:
            return result
        carry: dict = {}
        table: dict = {}
        for rid, row in peers:
            fields = row["fields"]
            for name in fields:
                text = str(fields[name]).strip()
                if not text or not_a_property(name):
                    continue
                carry[name] = 1 + carry.get(name, 0)
                elements = list_parts(text)
                if len(elements) >= 2:
                    elements = elements + [text.lower()]
                for el in elements:
                    entry = (rid, el, value_words(el))
                    table.setdefault(name, []).append(entry)
        filled = self._fr_filled
        wanted = self.needs.props_left()[:12]
        for req in wanted:
            head, raw = self._fr_halves(req)
            value = _g_unmarked(raw)
            head = head.strip()
            prop = self._fr_bare(head)
            if not (value and prop) or _g_points(value):
                continue
            prop_kin = self._fr_core_words(prop)
            low = value.lower()
            if re.search(r"\d", value) and LIMIT_WORDS.search(low):
                named_b = self._fr_often(carry, lambda n: strictly_same(n, prop))
                if prop_kin and named_b and (not several or self.product_of(head) in (None, k)):
                    kin_set = set(prop_kin)
                    has = any(filled(mine_fields, n) and (strictly_same(n, prop) or kin_set & set(name_words(n)))
                              for n in mine_fields)
                    if not has:
                        on_b = self._fr_carriers(peers, named_b)
                        lines.append("- %s = '%s': this store's field for it is %s, on %d of %d read listings of this "
                                     "category; this listing has no such field, so the limit is not shown met: it does NOT "
                                     "meet it (a figure in another field does not stand in)."
                                     % (prop, value, ", ".join("'%s'" % n for n in named_b[:3]), on_b, len(peers)))
                        fails.append("%s: it has no %s field, which the category's listings carry, so '%s' is not shown "
                                     "met" % (prop, named_b[0], value))
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?\s+[a-z]+", low.strip()) or (several and self.product_of(head) not in (None, k)):
                continue
            positive = names_the_item(value, head)
            alts = value_options(value) if positive or EXCLUSION.search(low) else []
            if not alts:
                continue
            named = self._fr_ranked(carry, lambda n: strictly_same(n, prop))
            by_name = bool(named)
            if by_name and not positive:
                flat = _g_flat(value)
                if any(" ".join(e[2]) == flat for e in self._fr_entries(table, named)):
                    SAID_VALUES.add(flat)
                    positive = True
            if not named:
                if not positive or (several and self.product_of(head) is None):
                    continue
                found = self._fr_often(carry, lambda n: any(pool_has(alts, e[2]) for e in table.get(n, [])))
                if not found:
                    continue
                top = max(found, key=carry.get)
                named = self._fr_ranked(carry, lambda n: strictly_same(n, top))
            peer_entries = [e for e in self._fr_entries(table, named) if e[0] != thing]
            writers = sorted({e[0] for e in peer_entries if pool_has(alts, e[2])})
            alone = {" ".join(e[2]) for e in peer_entries}
            on = self._fr_carriers(peers, named)
            samples: list = []
            for _, row in peers:
                for n in named:
                    short = _g_clip(filled(row["fields"], n), 40)
                    if short and len(samples) < 6 and short not in samples:
                        samples.append(short)
            keys_text = ", ".join("'%s'" % n for n in named[:3])
            tail = "%d other listing(s) write '%s' itself in it." % (len(writers), value)
            if not positive:
                bare = " ".join(EXCLUSION.sub(" ", low).split())
                bare_alts = value_options(bare)
                if bare_alts:
                    held = {e[0] for e in peer_entries if pool_has(bare_alts, e[2])}
                    tail = ("%d other listing(s) write '%s' there, the value it excludes (a listing that writes it does not "
                            "meet it)." % (len(held), bare))
            base = "- %s = '%s': this store's field for it is %s" % (prop, value, keys_text)
            if not by_name:
                base += " (no field is named for the property; this one's values write the value)"
            base += ", on %d of %d read listings of this category; %s" % (on, len(peers), tail)
            seen_text = " | ".join(samples) or "none"
            naming = []
            value_kinds = kind_set(value)
            if value_kinds and not writers:
                for rid in rows:
                    other = rows[rid]
                    if rid != thing and (not shelf or other.get("shelf") == shelf) \
                            and value_kinds <= set(other.get("words") or ()):
                        naming.append(rid)
                        if len(naming) == 3:
                            break
            if not positive:
                rules.append(base)
            elif writers:
                rule = (base + " Only a value writing those words meets it (a longer value containing them, or one element "
                        "of a list, counts; a synonym or related word does not).")
                if not by_name:
                    rule += (" With no field named for the property, a listing whose title writes the value meets it too, "
                             "unless its own field sharing the property's name says otherwise.")
                rules.append(rule)
            elif naming:
                rules.append(base + " Values seen: %s. Listings %s name the value in their title or text, so the store has "
                             "its own word for it: a synonym meets it only once such a listing's record shows this field "
                             "writes it otherwise." % (seen_text, ", ".join(naming)))
            elif on < Knobs.SYNONYM_READS:
                rules.append(base + " Values seen: %s. None writes it, but too few read listings carry this field to tell "
                             "whether the store writes the value itself: a same-meaning value is not shown to meet it yet."
                             % seen_text)
            else:
                rules.append(base + " Values seen: %s. None writes it, so a same-meaning value may meet it." % seen_text)
            pw = name_words(prop)
            mine = [(n, filled(mine_fields, n)) for n in named]
            mine = [pair for pair in mine if pair[1]]
            shown = self._fr_pairs_text(mine)
            qualified = []
            for n in (mine_fields if pw else ()):
                got = filled(mine_fields, n)
                if n in named or not got or not_a_property(n):
                    continue
                words = name_words(n)
                if words[1:] == pw and len(words) == len(pw) + 1:
                    qualified.append((n, got))
            qual_writes = [pair[0] for pair in qualified if field_writes(value, pair[1])]
            qual_near = True if qual_writes else any(field_touches(value, pair[1]) for pair in qualified)
            qual_note = (" Its field '%s' writes it (the same property's field under a fuller name), so it meets it."
                         % qual_writes[0]) if qual_writes else ""
            if self.ch_redirected(thing, prop):
                lines.append(base + " This listing's own notes may send this property to another listing, so its own field "
                             "may not decide.")
                continue
            if not positive:
                lines.append(base + " This listing: %s; it must not carry the excluded value." % (shown or "no such field"))
                continue
            title_says = False
            if not by_name:
                kin_set = set(prop_kin)
                kin = any(filled(mine_fields, n) and not not_a_property(n) and kin_set & set(self._fr_core_words(n))
                          and not field_writes(value, filled(mine_fields, n)) for n in mine_fields)
                title = me.get("title") or ""
                title_says = not kin and field_writes(value, title)
            if not mine:
                if qual_writes:
                    lines.append(base + " This listing has no such field." + qual_note)
                elif title_says:
                    lines.append(base + " This listing has no such field and no field is named for the property: its title "
                                 "writes the value, so it meets it.")
                else:
                    writing, naming_only = [], []
                    for n in mine_fields:
                        got = filled(mine_fields, n)
                        if n in named or not got or not_a_property(n):
                            continue
                        writes = field_writes(value, got)
                        if writes or any(len(w) >= 3 and w in pw for w in name_words(n)):
                            (writing if writes else naming_only).append((n, got))
                    related = self._fr_pairs_text((writing + naming_only)[:4])
                    lines.append(base + " This listing has no such field" + (" (fields that may name the property: %s)"
                                                                             % related if related else "")
                                 + ". Another name of the SAME property counts; a field of another property does not "
                                 "stand in.")
                continue
            parts, wholes = [], []
            for _, v in mine:
                elements = list_parts(v)
                parts += elements
                if len(elements) >= 2:
                    wholes.append(v.lower())
            parts += wholes

            def knocked(part: str) -> list | None:
                sides = plus_sides(part)
                hits = [pool_has(alts, value_words(s)) for s in sides]
                if not any(hits):
                    return None
                return [sides[i] for i in range(len(sides)) if not hits[i] and " ".join(value_words(sides[i])) in alone]
            clean, joined = [], []
            for p in parts:
                if not pool_has(alts, value_words(p)):
                    continue
                pushed = knocked(p)
                if not pushed:
                    clean.append(p)
                else:
                    joined.append((p, pushed))
            if clean:
                lines.append(base + " This listing: %s - it writes the value: it meets it." % shown)
            elif qual_writes:
                lines.append(base + " This listing: %s." % shown + qual_note)
            elif title_says:
                lines.append(base + " This listing: %s; with no field named for the property, its title writes the value: "
                             "it meets it." % shown)
            elif joined:
                part, others = joined[0]
                with_text = " and ".join(others)
                lines.append(base + " This listing: %s - '%s' joins the value with %s, which other listings carry alone in "
                             "this field: it does NOT meet it." % (shown, part, with_text))
                if by_name and not qual_near:
                    fails.append("%s: its %s is '%s', which joins %s with %s (a value this store also writes alone there), "
                                 "so it is not %s" % (prop, mine[0][0], part, value, with_text, value))
            elif writers:
                lines.append(base + " This listing: %s - it does not write the value, which other listings of this category "
                             "write in this field: it does NOT meet it (a synonym does not count where the store writes the "
                             "value itself)." % shown)
                if by_name and not (qual_near or any(field_touches(value, pair[1]) for pair in mine)):
                    fails.append("%s: its %s is '%s', while this store writes '%s' itself in that field on other listings"
                                 % (prop, mine[0][0], _g_clip(mine[0][1], 70), value))
            elif naming:
                lines.append(base + " This listing: %s - it does not write the value; no read listing writes it in this "
                             "field yet (values seen: %s), but listings %s name it in their title or text: the store has its "
                             "own word for it, so this listing's other value is not shown to meet it."
                             % (shown, seen_text, ", ".join(naming)))
            elif on < Knobs.SYNONYM_READS:
                lines.append(base + " This listing: %s - no read listing writes the value in this field (values seen: %s), "
                             "but only %d read listing(s) carry it: too few to tell, so a same-meaning value is not shown "
                             "to meet it yet." % (shown, seen_text, on))
            else:
                lines.append(base + " This listing: %s - no read listing writes the value in this field (values seen: %s), "
                             "so a same-meaning value may meet it." % (shown, seen_text))
        return result

    @staticmethod
    def _fr_pairs_text(pairs: list) -> str:
        bits = ["%s: '%s'" % (n, _g_clip(v, 70)) for n, v in pairs]
        return ", ".join(bits)

    def _fr_carriers(self, peers: list, names: list) -> int:
        """How many (id, row) peers fill at least one of the named fields."""
        count = 0
        for pair in peers:
            fields = pair[1]["fields"]
            if any(self._fr_filled(fields, n) for n in names):
                count += 1
        return count

    @staticmethod
    def _fr_filled(fields: dict, name: Any) -> str:
        got = fields.get(name, "")
        return str(got).strip()

    @staticmethod
    def _fr_often(carry: dict, test: Any) -> list:
        """Fields carried by two listings or more that pass `test`, in first-seen order."""
        return [n for n, times in carry.items() if times >= 2 and test(n)]

    @staticmethod
    def _fr_ranked(carry: dict, test: Any) -> list:
        """Fields that pass `test`, most carried first (first-seen order among equals)."""
        return sorted(filter(test, carry), key=carry.get, reverse=True)

    @staticmethod
    def _fr_entries(table: dict, names: list) -> list:
        out: list = []
        for n in names:
            out.extend(table.get(n, []))
        return out

    @staticmethod
    def _fr_ids(answer: Any, field: str) -> list:
        """The ids a check's answer lists under `field`, as text and in its order (text and whole numbers only)."""
        return [str(v) for v in (answer or {}).get(field) or [] if isinstance(v, (str, int))]

    def rank_check(self, thing: str) -> str:
        needs = self.needs
        priority = needs.priority()
        key = "%s|%s|%s" % (thing, needs.version, self.refusal_n)
        if priority == "none" or key in self.rank_checked:
            return ""
        if self.wire.left() < Knobs.WIND_DOWN_S + 30 or self.moves_left() < 4:
            return ""
        self.rank_checked.add(key)
        cheapest = priority == "cheapest"
        detail = needs.reading.get("priority_detail")
        what = "cheapest (lowest price)" if cheapest else ("best (%s)" % detail if detail else "best")
        parts = [RANK_TEXT.replace("PRIORITY", what), self._fr_request_block(), "REQUIREMENTS NOW: " + needs.text_body()]
        self._fr_add(parts, self.ref_note())
        cards = self.archive.cards(G_RANK_CARDS, cheaper_than=thing if cheapest else "", skip=self.ref_foreign(),
                                   first={thing})
        if self._fr_several():
            parts.append("SEVERAL PRODUCTS: only listings of the SAME category as the chosen item compete with it.")
        parts.append("THE CHOSEN ITEM: %s" % thing)
        parts.append("LISTINGS SEEN (DETAILS: the full record was read; ROW: seen only in search rows, its fields unknown - "
                     "a ROW of the requested category that beats the item and is not ruled out by its title, price or stock "
                     "may fit and should be named; figures in a row's title never rule it out):\n" + "\n".join(cards))
        self._fr_add(parts, self.field_names())
        self._fr_add(parts, "\n".join(self.field_view(thing)["rules"]),
                     "FIELD EVIDENCE (worked out by code from the store's records):\n")
        prompt = "\n\n".join(parts)
        answer = self.wire.ask_json(prompt, G_RANK_CAP, LINEUP["checker"])
        self.log.append(dict(kind="rank", item=thing, result=answer))
        if not isinstance(answer, dict):
            answer = {}
        better = [b for b in self._fr_ids(answer, "better") if b != thing and self._fr_live(b)][:4]
        better = self._fr_no_fails(better)
        self.rank_better = better
        self.named |= set(better)
        if not better:
            return ""
        return ("The rank check thinks these listings may also fit and beat %s on %s: %s%s. Check them (details, stock, "
                "price) before ordering; if each fails, order the current pick."
                % (thing, what, ", ".join(better), self._fr_reason(answer)))

    def _fr_live(self, ident: Any) -> bool:
        """Not refused at the order and not seen sold out."""
        return ident not in self.refused_lines and ident not in self.dead

    def _fr_no_fails(self, idents: list) -> list:
        """The listings whose field evidence shows no plain failure, in order."""
        return [i for i in idents if not self.field_view(i)["fails"]]

    @staticmethod
    def _fr_reason(answer: dict) -> str:
        said = str(answer.get("reason") or "").strip()
        return " (%s)" % said if said else ""

    @staticmethod
    def _fr_unique(items: Any) -> list:
        out: list = []
        for x in items:
            if x not in out:
                out.append(x)
        return out

    @staticmethod
    def _fr_page_key(args: Any) -> str:
        return json.dumps(args, sort_keys=True, default=str)

    def review_decline(self, early: bool = False) -> str:
        archive = self.archive
        reread = self.audits < 2 and len(archive.read) > self.audit_reads
        blocked = self.audit_done and not reread
        if blocked or self.wire.left() < Knobs.WIND_DOWN_S + 30:
            return ""
        self.audit_done = True
        if not early:
            self.audits += 1
        before = self.audit_ids
        first = None if before is None else set(archive.read) - before
        cards = archive.cards(G_AUDIT_CARDS, skip=self.ref_foreign(), first=first)
        self.audit_reads = len(archive.read)
        if not cards:
            return ""
        fresh = [] if before is None else sorted(set(archive.read) - before)
        self.audit_ids = set(archive.read)
        parts = [AUDIT_TEXT, self._fr_request_block(), "REQUIREMENTS NOW: " + self.needs.text_body()]
        self._fr_add(parts, self.ref_note())
        parts.append("LISTINGS SEEN (DETAILS: full record read; ROW: seen only in search rows):\n" + "\n".join(cards))
        if fresh:
            parts.append("OPENED SINCE THE LAST CHECK (judge these first, field by field): " + ", ".join(fresh))
        self._fr_add(parts, self.field_names())
        prompt = "\n\n".join(parts)
        answer = self.wire.ask_json(prompt, G_AUDIT_CAP, LINEUP["auditor"])
        self.log.append(dict(kind="audit", early=early, result=answer))
        if not isinstance(answer, dict):
            answer = {}
        conforms = self._fr_no_fails(self._fr_ids(answer, "possible_fits")[:5])
        self.audit_fits = conforms
        unopened = [u for u in self._fr_ids(answer, "unopened") if u not in archive.read and u in archive.ids][:6]
        hits = [h for h in self._fr_hits_ranked() if h not in archive.read and self._fr_in_scope(h)]
        try:
            slack = [s for s in self.slack_rows() if s not in archive.read]
        except Exception:  # noqa: BLE001
            slack = []
        extra = [u for u in unopened if u not in hits]
        merged = self._fr_unique(slack + hits[:4] + extra)[:6]
        pages = self.open_pages(2) if self.moves_left() > 5 else []
        if pages:
            merged = merged[:4]
            self.named_pages.update(map(self._fr_page_key, pages))
        combos = []
        if self.moves_left() > 5:
            combos = self._fr_logged("combined_queries_error", self.combined_queries, fallback=[])
            self.named_pages.update(map(self._fr_page_key, combos))
        self.named |= set(conforms) | set(merged)
        known = self.named_list
        known.extend(i for i in self._fr_unique(conforms + merged) if i not in known)
        if not (conforms or merged or pages or combos):
            return ""
        search = self.role("search") or "the search tool"
        notes = []
        if conforms:
            notes.append("The decline audit thinks these listings may meet every requirement: %s%s."
                         % (", ".join(conforms), self._fr_reason(answer)))
        if pages:
            notes.append("Read these result pages first with %s (one step shows a whole page; open the rows of the "
                         "requested kind within the budget): %s." % (search, " ; ".join(json.dumps(p) for p in pages)))
        if combos:
            notes.append("These combined searches were never run - where every word is required each lists only the "
                         "listings carrying all of them, and %s stays usable for them: %s."
                         % (search, " ; ".join(json.dumps(c) for c in combos)))
        if merged:
            notes.append("These listings were seen only in search rows and never opened - open their details first, "
                         "several in one reply: %s." % ", ".join(merged))
        notes.append("Verify them (details, stock, price) and buy the right one; decline only if each truly fails.")
        return " ".join(notes)

    def _fr_hits_ranked(self) -> list:
        pool = sorted(self.all_hits) if isinstance(self.all_hits, (set, frozenset)) else list(self.all_hits)
        return sorted(pool, key=lambda i: -(self.hit_score.get(i) or 0))

    def _fr_in_scope(self, ident: str, label: str | None = None) -> bool:
        """Not on another category than the main read ones (when both are known) and not over the budget."""
        labels = self.shelf_labels()
        lab = self.hit_shelf.get(ident) if label is None else label
        if labels and lab and lab not in labels:
            return False
        price, cap = self.archive.price(ident), self.needs.spend_cap
        return not (numeric(price) and numeric(cap) and price * self.needs.quantity() > cap)

    def named_unread(self) -> list:
        out = []
        for ident in self._fr_hits_ranked() + list(self.named_list):
            if ident in out or ident in self.archive.read or ident in self.refused_lines or ident not in self.archive.ids:
                continue
            if not self._fr_in_scope(ident):
                continue
            out.append(ident)
        return out

    # ================================================================================================ scripted detail reads
    def details_call(self, ident: str) -> dict | None:
        tool = self.role("details")
        if not tool:
            return None
        known = self.archive.ids
        for args in self._fr_past_args(tool):
            if any(isinstance(v, str) and v in known for v in args.values()):
                return {k: ident if isinstance(v, str) and v in known else v for k, v in args.items()}
        key = self._fr_text_arg(tool)
        return {key: ident} if key else None

    def _fr_past_args(self, tool: str) -> Any:
        """The dict arguments of the earlier calls of `tool`, newest first (lazily)."""
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "env":
                for action in entry.get("actions") or []:
                    if isinstance(action, dict) and action.get("name") == tool and isinstance(action.get("args"), dict):
                        yield action["args"]

    def _fr_text_arg(self, tool: str) -> Any:
        """The tool's text argument: the first required one, else the only one."""
        schema = self.shapes.get(tool) or {}
        props = schema.get("properties")
        if not isinstance(props, dict):
            props = {}
        need = schema.get("required") or []
        texts = [n for n in props if isinstance(props[n], dict) and props[n].get("type") == "string"
                 and not props[n].get("enum")]
        for n in texts:
            if n in need:
                return n
        return texts[0] if len(texts) == 1 else None

    def open_reads(self, queue: list, say: str, batch: Any, reader: Any = None) -> list:
        calls = self._fr_batch_calls(queue, batch) if batch else self._fr_single_calls(queue, reader)
        tick = self.wire.tick
        proposals = []
        for n, (name, args) in enumerate(calls):
            call = dict(name=name, arguments=json.dumps(args))
            proposals.append(dict(id="open_%d_%d" % (n, tick), type="function", function=call))
        if not proposals:
            return []
        archive = self.archive
        start = len(archive.items)
        self.chat.append(dict(role="assistant", content=say, tool_calls=proposals))
        responses = self._fr_covered(proposals)
        self.last_reads = archive.items[start:]
        return responses

    @staticmethod
    def _fr_batch_calls(queue: list, batch: Any) -> list:
        tool, arg, most = batch
        return [(tool, {arg: queue[at:at + most]}) for at in range(0, len(queue), most)]

    def _fr_single_calls(self, queue: list, reader: Any) -> list:
        tool = reader or self.role("details")
        out: list = []
        if not tool:
            return out
        for ident in queue:
            args = self.lookup_call(ident, tool)
            if args:
                out.append((tool, args))
        return out

    def _fr_covered(self, proposals: list) -> list:
        """Run scripted calls with the cover flag up."""
        self.covering = True
        try:
            return self.dispatch(proposals, scripted=True)
        finally:
            self.covering = False

    def open_named(self) -> bool:
        wire = self.wire
        stopped = wire.done or wire.broke
        queue, self.open_queue = self.open_queue, None
        put_read = not queue and bool(self.put_queue) and not stopped
        put_reader = None
        if put_read:
            queue, self.put_queue = self.put_queue, []
            self.open_recheck = True
            self.open_purpose = "cart"
            put_reader, self.put_tool = self.put_tool, None
            self.open_say = "I read the stock of what I just put in the cart - a listing can sell out right after it goes in."
        if stopped or not queue:
            return False
        ids: list = []
        for q in map(str, queue):
            if q not in ids:
                ids.append(q)
        recheck, purpose, floor, tool, say = self._fr_take_open()
        reader = put_reader if put_read else tool
        say = say or ("I re-read the cart lines before ordering." if recheck else
                      "I open the listings the checks named before any decline.")
        if purpose in ("menu", "quota", "walk") and self.narrowing() and not wire.mute:
            ids = self._fr_logged("reserve_skip_error", self._fr_room_cut, ids, purpose, fallback=ids)
            if not ids:
                return False
        if purpose == "cart":
            stale = set(self.stale_lines())
            units = self.trolley.units
            ids = [i for i in ids if i in stale or not (put_read or i in units)]
            if not ids:
                return False
        batch = self._fr_batch_for(ids, purpose, reader)
        responses = self.open_reads(ids, say, batch, reader)
        if batch and responses and not self.batch_proven:
            if not self.batch_ok(self.last_reads, ids):
                self.batch_spec = None
                self.log.append(dict(kind="batch_dropped", tool=batch[0]))
                if purpose == "menu":
                    most = 0
                elif purpose in ("quota", "walk"):
                    most = 2
                else:
                    most = 3
                if numeric(floor):
                    keep = int(floor)
                else:
                    keep = 3 if purpose not in ("menu", "quota") else self.order_due()
                spare = self.moves_left() - keep - 1
                n = min(most, len(ids), spare)
                if n > 0 and not wire.done and (reader or self.role("details")):
                    self.post_replies(responses)
                    first = ids[:n]
                    responses = self.open_reads(first, say, None, reader)
                    if not responses:
                        return True
            else:
                self.batch_proven = True
        if not responses:
            return False
        if not recheck and wire.left() > Knobs.WIND_DOWN_S + 30:
            note = self.review_decline()
            if note:
                last = responses[-1]
                responses[-1] = (last[0], last[1] + "\n[check after opening] " + note)
        self.post_replies(responses)
        return True

    def _fr_take_open(self) -> tuple:
        """(recheck, purpose, floor, tool, say) of the planned read, reset for the next one."""
        taken = (self.open_recheck, self.open_purpose, self.open_floor, self.open_tool, self.open_say)
        self.open_recheck = False
        self.open_purpose = self.open_floor = self.open_tool = self.open_say = None
        return taken

    def _fr_room_cut(self, ids: list, purpose: Any) -> list:
        """The ids a menu / quota / walk read may still open while the order's steps are kept."""
        keep = Knobs.LATE_STEPS if self.orders_tried != 0 else self.order_due()
        room = self.moves_left() - keep - 1
        listed = bool(self.batch_tool()) and len(ids) >= 2
        if room >= (1 if listed else len(ids)):
            return ids
        if not self.trolley.units and self.unverified():
            return ids
        kept = [] if listed else ids[:max(0, room)]
        self.log.append(dict(kind="reserve_skip", purpose=purpose, kept=kept))
        return kept

    def _fr_batch_for(self, ids: list, purpose: Any, reader: Any) -> Any:
        """The batch read spec for these ids, or a false value when they are read one by one."""
        if len(ids) < 2:
            return None
        spec = self.batch_tool()
        if not spec:
            return spec
        caps = self.tool_caps.get(spec[0]) or {}
        if purpose == "cart" and not self.batch_proven:
            return None
        if reader and not caps.get("stock"):
            return None
        return spec

    def last_stock(self, ident: str) -> bool | None:
        state = None
        for observation in self.archive.items:
            found: list = []

            def visit(d: dict) -> None:
                if self.row_mark(d) == ident:
                    s = self.stock_state(d)
                    if s is not None:
                        found.append(s)
            each_dict(observation, visit)
            if found:
                state = found[-1]
        return state

    # ================================================================================================ cheapest-rule helpers
    def rank_verify(self, thing: str) -> str:
        return self._fr_logged("rank_verify_error", self._fr_rank_verify, thing, fallback="")

    def _fr_rank_verify(self, thing: str) -> str:
        if self.needs.priority() != "cheapest":
            return ""
        checked = 0
        for ident in list(self.rank_better):
            if checked >= 2 or self.wire.left() < Knobs.WIND_DOWN_S + 30:
                break
            if self._fr_worth_check(ident):
                checked += 1
                self.check_item(ident, "ORDER")
        return self.cheaper_fit(thing)

    def _fr_worth_check(self, ident: str) -> bool:
        """A listing the rank check named is read, not passed yet, live, shows no plain failure and fits the budget."""
        needs = self.needs
        if ident not in self.archive.read or "%s|%s" % (ident, needs.version) in self.passed:
            return False
        if not self._fr_live(ident) or self.last_stock(ident) is False:
            return False
        if self.field_view(ident)["fails"]:
            return False
        k = self.product_index(ident) if self._fr_several() else None
        counts = needs.counts()
        qty = needs.quantity() if k is None or k >= len(counts) else counts[k]
        cost = self.archive.price(ident)
        cap = needs.spend_cap
        if not numeric(cost):
            return False
        return not (numeric(cap) and numeric(qty) and cost * qty > cap)

    def cheaper_fit(self, thing: Any) -> str:
        if not thing or self.needs.priority() != "cheapest":
            return ""
        archive = self.archive
        low = archive.price(thing)
        if not numeric(low):
            return ""
        several = self._fr_several()
        kind = self.shelf_known(thing)
        if several and not kind:
            return ""
        best = ""
        for ident in self._fr_passed_now():
            if ident == thing or not self._fr_live(ident):
                continue
            if several and self.shelf_known(ident) != kind:
                continue
            price = archive.price(ident)
            if numeric(price) and not price >= low and not self.field_view(ident)["fails"]:
                best, low = ident, price
        return best

    def _fr_passed_now(self) -> list:
        """The listings whose item check passed under the current requirements, in key order."""
        ver = str(self.needs.version)
        out = []
        for key in sorted(self.passed):
            ident, _, at = key.rpartition("|")
            if at == ver:
                out.append(ident)
        return out

    def shelf_known(self, thing: Any) -> str:
        label = ""
        for observation in self.archive.items:
            found: list = []

            def visit(d: dict) -> None:
                if any(isinstance(v, str) and v == thing and "id" in str(k).lower() for k, v in d.items()):
                    lab = self.shelf_of(d)
                    if lab:
                        found.append(lab)
            each_dict(observation, visit)
            if found:
                label = found[-1]
        if label:
            return label
        return str(self.archive.label_of.get(thing) or "").strip().lower()

    def chosen(self) -> str | None:
        lines = list(self.trolley.units)
        return lines[-1] if lines else None

    # ================================================================================================ products and categories
    def home_shelves(self) -> list:
        count = len(self.archive.items)
        memo = self.home_memo
        if memo is None or memo[0] != count:
            try:
                result = self._fr_homes()
            except Exception:  # noqa: BLE001
                result = [set() for _ in (self.needs.items() or [0])]
            memo = self.home_memo = (count, result)
        return list(map(set, memo[1]))

    def _fr_homes(self) -> list:
        """Per product, the category labels of the seen rows whose words match its name best."""
        rows = self.needs.items() or [{"product": str(self.needs.reading.get("product") or "")}]
        seen = self._fr_row_labels()
        names = [set(kind_list(r["product"])) for r in rows]
        result = []
        for k, own in enumerate(names):
            shared = set().union(*[other for j, other in enumerate(names) if j != k])
            scores = {label: kind_points(own, set(kind_list(label)), shared) for label in sorted(seen)}
            top = max(scores.values(), default=0.0)
            result.append({label for label, score in scores.items() if score == top} if top > 0 else set())
        return result

    def _fr_row_labels(self) -> set:
        """The category labels of every seen row that carries a known listing id."""
        seen: set = set()
        ids = self.archive.ids

        def visit(d: dict) -> None:
            if any(isinstance(v, str) and v in ids and "id" in str(k).lower() for k, v in d.items()):
                label = self.shelf_of(d)
                if label:
                    seen.add(label)
        for observation in self.archive.items:
            each_dict(observation, visit)
        return seen

    def own_shelf(self) -> str:
        try:
            if self._fr_several():
                return ""
            home = self.home_shelves()[0]
            if not home:
                return ""
            read = [label.strip().lower() for label in self.archive.shelves_read()]
            if all(label in home for label in read):
                return ""
            for label in read:
                if label in home:
                    return label
            return sorted(home)[0]
        except Exception:  # noqa: BLE001
            return ""

    def joined_phrase(self, a: str, b: str, label: str) -> tuple | None:
        text = self.request.lower().replace("’", "'")
        low = re.sub(r"\s+", " ", text)
        sides = self._fr_sides(str(label).lower())
        if len(sides) == 2:
            pats = list(map(kind_regex, sides))
            if all(pats):
                joiner = r"\s*(?:\band\b|&|\bor\b|/|\+)\s*"
                m = re.search(r"\b" + joiner.join(pats), low)
                if m is not None:
                    return m.group().strip(), True
        ta = kind_list(a)
        tb = kind_list(b)
        if not (ta and tb) or same_kind(ta[-1], tb[-1]):
            return None
        for x, y in [(a, b), (b, a)]:
            pats = [kind_regex(x), kind_regex(y)]
            if not all(pats):
                continue
            # the nearest second name after each first one (at most 40 characters on, no sentence mark between)
            pair = re.compile(r"\b(?:" + pats[0] + r")(?P<between>[^.;:!?]{0,40}?)(?:" + pats[1] + r")")
            for m in pair.finditer(low):
                if self._fr_loose_pair(m, low, a, b):
                    return m.group().strip(), False
        return None

    @staticmethod
    def _fr_loose_pair(m: Any, low: str, a: str, b: str) -> bool:
        """A match of the two names joined by a short 'and' / 'or' / sign, and not both named elsewhere too."""
        middle = m.group("between")
        if len(re.findall(r"[a-z0-9]+", middle)) > 3:
            return False
        if re.search(r"\d|\bx\b", middle) or not re.search(r"\b(?:and|or)\b|[&,/+]", middle):
            return False
        rest = "%s | %s" % (low[:m.start()], low[m.end():])
        return not all(re.search(r"\b" + kind_regex(z), rest) for z in (a, b))

    @staticmethod
    def _fr_sides(low_label: str) -> list:
        """The non-blank parts of a lower-case category label split where it joins two kinds."""
        return [s for s in re.split(PART_JOINS, low_label) if s.strip()]

    def product_question(self, line: Any, rows: list) -> bool:
        text = str(line)
        found = re.match(r'-\s*(.*?): asked "(.*?)" -> ', text, re.S)
        label, asked = (found.group(1), found.group(2)) if found else (text, text)
        kinds = [kind_list(r.get("product")) for r in rows if isinstance(r, dict)]
        for tag in re.findall(r"\(([^()]*)\)", label):
            for tok in kind_list(tag):
                if any(kind_found(tok, own) for own in kinds):
                    return True
        low = asked.lower().replace("’", "'")
        for r in rows:
            pattern = kind_regex(r.get("product")) if isinstance(r, dict) else ""
            if pattern and re.search(r"\b" + pattern, low):
                return True
        return False

    @staticmethod
    def joined_clash(a: dict, b: dict) -> bool:
        def pairs(row: dict) -> list:
            out = []
            for req in row.get("requirements") or []:
                head, _, value = str(req).partition(":")
                out.append((head.split("(")[0].strip(), _g_flat(value)))
            return out
        return any(loosely_same(pa, pb) and va != vb for pa, va in pairs(a) for pb, vb in pairs(b))

    def merge_joined(self, observation: Any) -> str:
        each_dict(observation, self._fr_note_joined)
        remarks = []
        for _ in range(2):
            pick = self._fr_merge_pick()
            if pick is None:
                break
            remark = self.merge_items(*pick)
            if remark:
                remarks.append(remark)
        return " ".join(remarks)

    def _fr_note_joined(self, d: dict) -> None:
        """Remember a two-kind category label ('Tables & Chairs') seen on a listing row."""
        if not any(isinstance(v, str) and "id" in str(k).lower() and ID_SHAPE.fullmatch(v) for k, v in d.items()):
            return
        for key in d:
            v = d[key]
            lk = str(key).lower()
            if not re.search(r"categor|shelf|department|aisle", lk) or "id" in lk or not isinstance(v, str):
                continue
            lab = v.strip()
            if not 1 <= len(lab) <= 60:
                continue
            if len(self._fr_sides(lab.lower())) == 2:
                self.joined_seen.setdefault(lab.lower(), lab)
            return

    def _fr_merge_pick(self) -> tuple | None:
        rows = self.needs.items()
        if len(rows) < 2 or not self.joined_seen or self.trolley.used or self.trolley.units:
            return None
        needs = self.needs
        if needs.uncovered or needs.indifferent:
            return None
        if any(self.product_question(line, rows) for line in self.questions_log):
            return None
        return self._fr_joined_pair(rows)

    def _fr_joined_pair(self, rows: list) -> tuple | None:
        kinds = [kind_list(r["product"]) for r in rows]
        count = len(rows)
        for low_label in sorted(self.joined_seen):
            sides = [kind_list(s) for s in self._fr_sides(low_label)]
            if len(sides) != 2 or not all(sides):
                continue
            whole = sides[0] + sides[1]
            label = self.joined_seen[low_label]
            for i, j in [(i, j) for i in range(count) for j in range(i + 1, count)]:
                ti, tj = kinds[i], kinds[j]
                if not (ti and tj) or (kinds_found(ti, tj) and kinds_found(tj, ti)):
                    continue
                if rows[i]["quantity"] != rows[j]["quantity"] or not kinds_found(ti + tj, whole):
                    continue
                if not self._fr_pair_fits(sides, ti, tj) or self.joined_clash(rows[i], rows[j]):
                    continue
                found = self.joined_phrase(rows[i]["product"], rows[j]["product"], label)
                if found:
                    return (i, j, label, *found)
        return None

    @staticmethod
    def _fr_pair_fits(sides: list, ti: list, tj: list) -> bool:
        """The label's two sides name the two products, one side each (in either order)."""
        only_i = [w for w in ti if not kind_found(w, tj)]
        only_j = [w for w in tj if not kind_found(w, ti)]

        def fits(x: list, y: list) -> bool:
            for w in only_i:
                if not kind_found(w, x) or kind_found(w, y):
                    return False
            for w in only_j:
                if not kind_found(w, y) or kind_found(w, x):
                    return False
            return any(kind_found(w, x) for w in ti) and any(kind_found(w, y) for w in tj)
        return fits(sides[0], sides[1]) or fits(sides[1], sides[0])

    def merge_items(self, i: int, j: int, label: str, phrase: str, in_order: bool = True) -> str:
        needs = self.needs
        rows = needs.items()
        reading = needs.reading
        count = len(rows)
        raw = list(filter(self._fr_named_row, reading.get("items") or []))
        if not (0 <= i < j < count) or len(raw) != count:
            return ""
        kinds = [kind_list(r["product"]) for r in rows]
        pair_kinds = kinds[i] + kinds[j]
        others = [kinds[n] for n in range(count) if n != i and n != j]
        whole = kind_list(label)
        left = count - 1
        new_tag = "" if left < 2 else phrase

        def owned(tag: Any) -> bool:
            toks = kind_list(tag)
            if not (toks and kinds_found(toks, whole)):
                return False
            if all(not kind_found(t, pair_kinds) for t in toks):
                return False
            for o in others:
                if kinds_found(toks, o) and kinds_found(o, toks):
                    return False
            return True

        def swap(m: Any) -> str:
            if not owned(m.group(1)):
                return m.group(0)
            return " (%s)" % new_tag if new_tag else ""

        def retag(text: Any) -> Any:
            if isinstance(text, str):
                cut = text.partition(":")
                text = re.sub(r"\s*\(([^()]*)\)", swap, cut[0]) + cut[1] + cut[2]
            return text

        def retag_all(items: list) -> list:
            out: list = []
            for y in map(retag, items):
                if y not in out:
                    out.append(y)
            return out
        seen: set = set()
        merged_reqs: list = []
        for req in rows[i]["requirements"] + rows[j]["requirements"]:
            head, value = self._fr_halves(req)
            new = re.sub(r"\s*\([^()]*\)", "", head)
            new = new.strip() + ":" + value
            low = new.lower()
            if low in seen:
                continue
            seen.add(low)
            merged_reqs.append(new)
        joined_row = dict(product=phrase, quantity=rows[i]["quantity"], requirements=merged_reqs)
        new_items = raw[:j] + raw[j + 1:]
        new_items[i] = joined_row
        needs.written = retag_all(needs.written)
        needs.uncovered = retag_all(needs.uncovered)
        needs.let_go = retag_all(needs.let_go)
        needs.indifferent = retag_all(needs.indifferent)
        if needs.now_reqs is not None:
            needs.now_reqs = retag_all(needs.now_reqs)
        needs.resolved = list(map(retag, needs.resolved or []))
        for key in ("requirements", "hints", "references", "soft_wants"):
            got = reading.get(key)
            if isinstance(got, list):
                reading[key] = retag_all(got)
        targets = reading.get("reference_targets")
        for target in (targets if isinstance(targets, list) else []):
            if not isinstance(target, dict):
                continue
            if owned(str(target.get("product") or "")):
                target["product"] = phrase
        for record in self.refs:
            for field in ("req", "head", "now"):
                got = record.get(field)
                if isinstance(got, str):
                    record[field] = retag(got)
            tag = record.get("tag")
            if tag and owned(tag):
                record["tag"] = new_tag
        self.pending_seconds = list(map(retag, self.pending_seconds))
        self.open_pair = list(map(retag, self.open_pair))
        if left < 2:
            reading.pop("items", None)
            reading.update(product=phrase, quantity=rows[i]["quantity"])
        else:
            names = [str(r.get("product") or "") for r in new_items]
            reading.update(items=new_items, product=" and ".join(names))
        needs.version += 1
        self.menu_memo = {}
        self.home_memo = None
        self.walked = set()
        self.quota_reads = {}
        entry = dict(kind="merge_joined", label=label, products=[r["product"] for r in (rows[i], rows[j])], phrase=phrase,
                     label_words=in_order, left=left)
        self._fr_scale()
        self.log.append(entry)
        return ("The store sells '%s' as ONE category, and the request asks for one product of it (%s): the order holds %s "
                "for it, not one line per kind its name joins. Requirements now: %s"
                % (label, phrase, "one line" if left >= 2 else "ONE line", self.needs.text_body()))
# ---------------------------------------------------------------------------------------------- cart work and questions
H_MESSAGE_NOUNS = r"messages?|questions?|notes?|repl(?:y|ies)|texts?"         # code_limit nouns of a message budget
H_STEP_NOUNS = r"steps?|moves?|actions?|(?:tool )?calls?|turns?"                # ... and of a step budget
H_STATUS_CUT = "\n[status]"                     # where the status line starts in a reply text
H_MARKS = re.compile(r"\(stated\)|\(revealed[^)]*\)")
H_LINKED = re.compile(r"\b(?:" + LINK_PHRASES + r")\b")
H_RANGE = re.compile(r"\d\s*(?:-|to)\s*\d")
H_ROUGH = re.compile(r"\b(?:about|around|approx\w*|roughly)\b")
H_POINT_WORDS = ("as shown", "same as", "the page", "sold by", "redirect", "refer")
H_BUSY_ROLES = ("search", "message", "cart", "cart_view", "order", "decline")
H_LIST_CHANGERS = re.compile(r"\b(?:remov|delet|cancel|clear|discard|releas|refund|purchas|checkout|wish|basket|cart|bag)")
H_STOCK_TALK = re.compile(r"\b(?:in[- ]stock|on hand|on the shelf|sold[- ]out|out of stock|available now)\b"
                          r"|\bback[- ]?order")              # backorder, back-ordered ... (no end boundary)
H_CHEAPEST_TALK = re.compile(r"\b(?:cheapest|least expensive|lowest[- ](?:price|priced|cost)|smallest (?:total|price))\b"
                             r"|\blowest\b[^.]{0,20}\bprice\b")
# the spending-limit reader
H_FIGURE = re.compile(r"(?:([$\u20ac\u00a3])\s*)?(\d[\d,]*(?:\.\d+)?)")
H_WHOLE_ORDER = re.compile(r"\b(?:order|total|whole|lot|sum|bill|altogether|overall|everything)\b")
H_COUNTED = re.compile(r"\s*(?:x\b|\u00d7|%|[:/-]\s*\d|(?:[a-z-]+\s+){0,2}?(?:messages?|questions?|notes?|repl\w*|"
                       r"texts?|steps?|moves?|actions?|(?:tool )?calls?|turns?|units?|items?|pieces?|packs?|pairs?|"
                       r"lots?|sets?|of|times?)\b)")
H_RUN_BEFORE = re.compile(r"(?:\d|[.,:/\-])\s*$|\bx\s*$")
# words may carry an apostrophe (don't, I'd) and count as one word
H_SPEND_CUE = re.compile(r"\b(?:spend\w*|budget\w*|afford\w*|pay\w*|price|priced|prices|cost|costs|cash|money)\b"
                         r"(?:\W+[\w']+){0,4}\W*$")
H_LIMIT_CUE = re.compile(r"\b(?:under|below|at most|no more than|not more than|up to|within|max\w*|less than|"
                         r"cheaper than|not (?:more|over|above)|limit\w*|reach\w*|go to|to|at|cap\w*|past|exceed\w*|"
                         r"beyond|go over)\W+(?:[\w']+\W+)?$")
H_CEILING_AFTER = re.compile(r"\s*(?:[a-z.()]+\s*){0,3}?(?:or (?:lower|less|under|below|cheaper)|"
                             r"at (?:the (?:very )?)?most|max\w*|tops\b|"
                             r",?\s*(?:and )?not (?:one|a) \w+ (?:over|more|above))")      # "and not a cent more" too
H_UNIT_WORDS = re.compile(r"\s*([a-z][a-z.]*)(?:\s+([a-z][a-z.]*))?")
H_UNIT_SKIP = re.compile(r"or|and|not|tops|max\w*|only|just|is|was")       # fullmatch: no unit word
H_ASKED = re.compile(r'asked "(.*?)" -> ')        # the question text of a questions_log line
H_UNSET = object()                              # batch_spec before the list-read tool was looked up


class CartworkMixin:
    """Tool lookups and call arguments, record identity, stock truth and cart bookkeeping, sold-out and refusal repair,
    the code's candidate pool per product and the holds before a decline, listings read for the question menu, the
    planned questions to the shopper, and the request reading code makes on its own."""

    def _cartwork_state(self) -> None:
        # what each tool's answers carry
        self.tool_caps: dict = {}
        self.tool_keys: dict = {}
        self.note_keys: dict = {}
        self.text_keys: dict = {}
        self.read_with: dict = {}
        self.row_keys: set = set()
        self.row_text: dict = {}
        self.stock_by: dict = {}
        self.stock_seen = False
        self.rows_live = False
        self.rows_lag = False
        self.role_guess: dict | None = None
        # stock and cart changes
        self.dead: set = set()
        self.dead_by: dict = {}
        self.cart_seq = 0
        self.line_seq: dict = {}
        self.cart_ver = 0
        self.since_cart = 0
        self.cart_shown: int | None = None
        self.first_puts: set = set()
        self.put_queue: list = []
        self.put_tool: str | None = None
        # the sold-out swap
        self.swap_unread: list = []
        self.swap_page: dict | None = None
        # holds and repairs before a decline
        self.repair_tried: set = set()
        self.near_done = False
        self.near_cart: dict | None = None
        self.near_ok: set = set()
        self.near_said: dict = {}
        self.slack_total: float | None = None
        self.ed_memo: dict = {}
        # questions to the shopper
        self.pending_seconds: list = []
        self.pending_text: dict = {}
        self.open_pair: list = []
        self.wanted_asks: list = []
        self.spent_asks: set = set()
        self.reasked: set = set()
        self.quiet_asks: list = []
        self.no_pairs = False
        self.plan_wait: int | None = None
        self.empty_plans = 0
        self.asked_at_sets: int | None = None
        self.asked_at_tick: int | None = None
        self.asking_label = ""
        self.product_plans = 0
        self.quota_reads: dict = {}
        self.quota_wait = False
        self.pre_read_done = False
        self.pre_rounds = 0
        self.pre_picked: list = []
        # the list-read tool: H_UNSET until looked up; None when the store has none or list reads were dropped
        self.batch_spec: Any = H_UNSET
        # a sold-out cart line, a refusal or a shopper change was seen (dead_note sets it; it enlarges the reserve)
        self.event_seen = False

    # ================================================================================== small private helpers
    def _h_ver(self) -> str:
        return str(self.needs.version)

    def _h_failed(self) -> set:
        """Ids a ruling failed under the current version."""
        ver = self._h_ver()
        out = set()
        for key, ruling in self.rulings.items():
            if not ruling:
                continue
            ident, bar, rest = str(key).partition("|")
            if bar and rest.partition("|")[0] == ver:
                out.add(ident)
        return out

    def _h_passed_now(self) -> list:
        """Ids the item check passed under the current version, in the passed set's own order."""
        ver = self._h_ver()
        split = [str(key).rpartition("|") for key in self.passed]
        return [head for head, _, tail in split if tail == ver]

    def _h_live(self) -> set:
        """Ids the item check passed under the current version."""
        return {ident for ident in self._h_passed_now() if ident}

    def _h_count(self, tool: Any, name: str) -> int:
        return (self.tool_caps.get(tool) or {}).get(name, 0)

    def _h_cost(self, ident: str) -> float:
        p = self.archive.price(ident)
        return float(p) if numeric(p) else float("inf")

    @staticmethod
    def _h_tag(label: Any) -> str:
        m = re.search(r"\(([^)]*)\)", str(label or ""))
        return m.group(1).strip() if m else ""

    def _h_product_word(self, label: Any, fallback: str | None = None) -> str:
        return (self._h_tag(label) or (fallback or "").strip()
                or str(self.needs.reading.get("product") or "").strip() or "item")

    def _h_plain(self, label: Any, fallback: str | None = None) -> str:
        """The plain one-property question for a label."""
        name = str(label or "").split("(")[0].strip()
        return "Which %s would you like for the %s?" % (name, self._h_product_word(label, fallback))

    @staticmethod
    def _h_second(raw: Any) -> str:
        s = str(raw or "").strip()
        low = s.lower()
        return "" if ("none" in low or s == "-" or "n/a" in low) else s

    @staticmethod
    def _h_head(row: Any) -> str:
        return str(row).split(":", 1)[0].strip(" -")

    @staticmethod
    def _h_asked_text(row: Any) -> str:
        """The question text of a questions_log line (on one line of text: the quote and its ' -> ' close it)."""
        m = H_ASKED.search(str(row))
        return m.group(1) if m else ""

    def _h_asked_parts(self) -> set:
        out = set()
        for row in self.questions_log:
            for part in self._h_head(row).split(";"):
                out.add(part.strip().lower())
        return out

    @staticmethod
    def _h_menu_key(line: str) -> str:
        return line.strip().lstrip("-").strip().split(" (in ")[0].strip()

    def _h_product_name(self, k: Any) -> str:
        items = self.needs.items()
        if isinstance(k, int) and 0 <= k < len(items):
            return items[k]["product"]
        return str(self.needs.reading.get("product") or "").strip() or "item"

    @staticmethod
    def _h_take_turns(pools: list, most: int) -> list:
        """One id from each pool per round, in pool order, until `most` are taken or the pools run dry (an id two pools
        hold is taken from each)."""
        taken: list = []
        depth = 0
        while len(taken) < most and any(depth < len(p) for p in pools):
            for p in pools:
                if depth < len(p) and len(taken) < most:
                    taken.append(p[depth])
            depth += 1
        return taken

    def _h_cart_exactly(self, picks: list) -> bool:
        """Code makes the cart hold exactly the picks (muted), re-reads the stale ones, and tells whether the cart now
        holds them all with no line sold out."""
        wanted = [p for p, _ in picks]
        wire = self.wire
        muted, wire.mute = wire.mute, True
        try:
            bad = self.code_put(picks)
            done = bad is not None and not bad and set(wanted).issubset(self.trolley.units)
            if done:
                stale_now = [line for line in self.stale_lines() if line in wanted]
                if stale_now and self.moves_left() >= len(stale_now) + 2:
                    done = not self.code_recheck(stale_now)[1]
        finally:
            wire.mute = muted
        units = self.trolley.units
        return done and bool(units) and all(line not in self.dead for line in units)

    # ================================================================================== 1. tool lookup and arguments
    @staticmethod
    def _h_schema_parts(schema: dict) -> tuple:
        """(properties, required list) of an argument schema; empty ones when absent or malformed."""
        props = schema.get("properties")
        need = schema.get("required")
        return (props if isinstance(props, dict) else {}), (need if isinstance(need, list) else [])

    def _h_schema(self, tool: Any) -> tuple:
        return self._h_schema_parts(self.shapes.get(tool) or {})

    @staticmethod
    def _h_sole(texts: list, need: list) -> str | None:
        """The text argument a call can carry on its own: the required one when it is a text (or the only text when
        nothing is required), and only when nothing else is required."""
        if not need:
            return texts[0] if len(texts) == 1 and texts[0] else None
        first = need[0]
        if need.count(first) != len(need) or first not in texts:
            return None
        return first or None

    @staticmethod
    def _h_clip(spec: dict, text_body: str) -> str:
        most = spec.get("maxLength")
        if isinstance(most, int) and not isinstance(most, bool):
            return text_body[:most]
        return text_body

    def lookup_key(self, tool: Any) -> str | None:
        props, listed = self._h_schema(tool)
        need = [r for r in listed if r in props]
        texts = []
        for n, spec in props.items():
            if isinstance(spec, dict) and spec.get("type") == "string" and not spec.get("enum"):
                texts.append(n)
        return self._h_sole(texts, need)

    def lookup_tools(self) -> list:
        busy = {self.role(r) for r in H_BUSY_ROLES}
        return [t for t in self.shapes if t not in busy and self.lookup_key(t)]

    def lookup_call(self, ident: str, tool: Any) -> dict | None:
        if not tool:
            return None
        if tool == self.role("details"):
            return self.details_call(ident)
        known = self.archive.ids

        def is_id(v: Any) -> bool:
            return isinstance(v, str) and v in known
        for entry in reversed(self.log):
            if not isinstance(entry, dict) or entry.get("kind") != "env":
                continue
            for action in entry.get("actions") or []:
                if isinstance(action, dict) and action.get("name") == tool:
                    args = action.get("args")
                    if isinstance(args, dict) and any(map(is_id, args.values())):
                        return {k: ident if is_id(v) else v for k, v in args.items()}
        key = self.lookup_key(tool)
        return {key: ident} if key else None

    def text_arg(self, tool: Any) -> str | None:
        props, need = self._h_schema(tool)
        texts = []
        for n, spec in props.items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            kind = spec.get("type")
            if kind == "string" or (isinstance(kind, list) and "string" in kind):
                texts.append(n)
        return self._h_sole(texts, need)

    @staticmethod
    def _h_call_dict(cid: str, tool: Any, args: Any) -> dict:
        return dict(id=cid, type="function", function=dict(name=tool, arguments=json.dumps(args)))

    def message_call(self, text_body: str, cid: str) -> dict | None:
        tool = self.role("message")
        if not tool:
            return None
        aim = self.text_arg(tool)
        if not aim:
            return None
        fixed, faults = ArgFixer.fix({aim: text_body}, self.shapes.get(tool) or {})
        return None if faults else self._h_call_dict(cid, tool, fixed)

    @staticmethod
    def _h_default_for(spec: dict, cue: Any, text_body: str) -> tuple:
        """(has a value, the value) a required argument gets in a call code makes on its own."""
        options = spec.get("enum")
        if isinstance(options, list) and options:
            if cue is not None:
                for o in options:
                    if isinstance(o, str) and cue.search(o.lower()):
                        return True, o
            return True, options[0]
        kind = spec.get("type")
        if kind == "string" or (isinstance(kind, list) and "string" in kind):
            return True, CartworkMixin._h_clip(spec, text_body)
        if kind == "boolean":
            return True, True
        if kind in ("integer", "number"):
            low = spec.get("minimum", 1)
            return True, (max(1, low) if numeric(low) else 1)
        return False, None

    def plain_args(self, call_name: Any, text_body: str, kind: str = "") -> dict:
        props, need = self._h_schema(call_name)
        cue = None
        if kind == "decline":
            cue = DECLINE_SENSE
        elif kind == "order":
            cue = ORDER_SENSE
        args: dict = {}
        for name in need:            # in the schema's required order
            spec = props.get(name)
            if isinstance(spec, dict):
                found, value = self._h_default_for(spec, cue, text_body)
                if found:
                    args[name] = value
        if len(props) == 1:
            [(name, spec)] = props.items()
            if name not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                args[name] = self._h_clip(spec, text_body)
        return args

    def scrub_decline(self, call_args: dict, call_name: Any) -> dict:
        flag = self.needs.reading.get("decline_no_names")
        if flag is False:
            return call_args
        known = self.archive.ids
        named = any(isinstance(v, str) and any(t in known for t in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", v) if t)
                    for v in call_args.values())
        if not flag and not named:
            return call_args
        schema = self.shapes.get(call_name) or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        out = dict(call_args)
        for name, value in call_args.items():
            if not isinstance(value, str):
                continue
            spec = props.get(name) if isinstance(props.get(name), dict) else {}
            if "enum" not in spec:
                out[name] = self._h_clip(spec, APOLOGY)
        return out

    def _h_tool_words(self) -> dict:
        """Tool name -> its spaced-out name and description (the last entry of a name wins)."""
        told: dict = {}
        for entry in self.toolset or []:
            fn = entry.get("function") if isinstance(entry, dict) else None
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                name = fn["name"]
                spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")
                told[name] = spaced + " " + str(fn.get("description") or "")
        return told

    @staticmethod
    def _h_id_list(props: dict) -> str | None:
        """The one argument that is a list of free strings, else None."""
        arrays = []
        for n, spec in props.items():
            if not isinstance(spec, dict) or spec.get("type") != "array":
                continue
            inner = spec.get("items")
            if isinstance(inner, dict) and inner.get("type") == "string" and not inner.get("enum"):
                arrays.append(n)
        return arrays[0] if len(arrays) == 1 else None

    def batch_tool(self) -> tuple | None:
        if self.batch_spec is not H_UNSET:
            return self.batch_spec
        self.batch_spec = None
        try:
            taken = {self.role(r) for r in H_BUSY_ROLES}
            told = self._h_tool_words()
            taken.update(name for name in self.shapes if H_LIST_CHANGERS.search(told.get(name, name).lower()))
            for tool, schema in self.shapes.items():
                if tool in taken or not isinstance(schema, dict):
                    continue
                props, need = self._h_schema_parts(schema)
                arg = self._h_id_list(props)
                if arg is None or need.count(arg) != len(need):
                    continue
                top = props[arg].get("maxItems")
                if numeric(top) and top >= 2:
                    self.batch_spec = (tool, arg, int(min(top, 10)))
                    break
        except Exception:  # noqa: BLE001
            self.batch_spec = None
        return self.batch_spec

    @staticmethod
    def batch_ok(found: Any, ids: Any) -> bool:
        try:
            wanted = set(ids)
            got: set = set()

            def visit(rec: dict) -> None:
                values = list(rec.values())
                if not any(CartworkMixin._h_mapish(v) for v in values):
                    if any(isinstance(v, dict) for v in values):
                        return
                    if len([v for v in values if isinstance(v, str) and 1 <= len(v) <= 70]) < 5:
                        return
                got.update(v for v in values if isinstance(v, str) and v in wanted)
            for answer in found:            # each answer walked from its own top
                each_dict(answer, visit)
            return len(got) > 0 and len(wanted) <= 2 * len(got)
        except Exception:  # noqa: BLE001
            return False

    # ================================================================================== 2. record identity
    @staticmethod
    def rec_id(d: dict) -> str | None:
        pairs = [(str(k).lower(), v) for k, v in d.items() if isinstance(v, str) and ID_SHAPE.fullmatch(v)]
        for name, v in pairs:
            if name == "id":
                return v
        for name, v in pairs:
            if name.endswith("id"):
                return v
        return None

    @staticmethod
    def asked_ids(call_args: Any) -> set:
        if not isinstance(call_args, dict):
            return set()
        flat = []
        for v in call_args.values():
            flat.extend(v if isinstance(v, list) else [v])
        return {x for x in flat if isinstance(x, str) and ID_SHAPE.fullmatch(x)}

    @staticmethod
    def title_of(row: dict) -> str:
        title = row.get("title")
        if isinstance(title, str):
            return title
        return next((v for k, v in row.items() if isinstance(v, str) and re.search(r"title|heading|headline", str(k).lower())),
                    "")

    # ================================================================================== 3. what tool answers carry
    @staticmethod
    def _h_mapish(v: Any) -> bool:
        """A nested map holding two or more plain values (a property block)."""
        if not isinstance(v, dict):
            return False
        return len([x for x in v.values() if isinstance(x, (str, int, float, bool))]) >= 2

    def _h_book_stock(self, ident: str, state: bool, source: str, epoch: int, newly: list) -> None:
        """One stock reading of a record, by its source (a lookup of the id, or a list row)."""
        self.stock_seen = True
        book = self.stock_by.setdefault(ident, {})
        other = book.get("lookup" if source == "row" else "row")
        if other is not None:
            seen_at, seen_state = other
            if seen_at == epoch and seen_state != state:
                self.rows_lag = True
        if source == "row":
            earlier = book.get("row")
            if earlier is not None and earlier[1] != state:
                self.rows_live = True
        book[source] = (epoch, state)
        if state is not False:
            if source == "lookup" or self.dead_by.get(ident) != "lookup":
                self.dead.discard(ident)
            return
        fresh_loss = ident not in self.dead and ident in self.trolley.units
        if fresh_loss and ident not in newly:
            newly.append(ident)
        self.dead.add(ident)
        self.dead_by[ident] = source

    @staticmethod
    def _h_bucket(book: dict, call_name: Any) -> set:
        return book.setdefault(call_name, set())

    def _h_book_read(self, call_name: Any, ident: str, rec: dict, got: set) -> None:
        """A record of an id the call asked for: what this tool's answers carry."""
        got.add("id")
        self._h_bucket(self.read_with, call_name).add(ident)
        self._h_bucket(self.text_keys, call_name).update(k for k, v in rec.items() if v is None or isinstance(v, str))
        if any(self._h_mapish(v) for v in rec.values()):
            got.add("props")
        if any(numeric(v) and re.search(r"price|cost", str(k).lower()) for k, v in rec.items()):
            got.add("price")
        self._h_bucket(self.tool_keys, call_name).update(rec)
        prior = self.row_text.get(ident)
        long_texts = [(k, v) for k, v in rec.items() if isinstance(v, str) and len(v.split()) >= 8]
        for k, v in long_texts:
            fresh_text = (k not in self.row_keys) if prior is None else (prior.get(k) != v)
            if fresh_text:
                self._h_bucket(self.note_keys, call_name).add(k)
                got.add("notes")

    def learn_caps(self, call_name: Any, call_args: Any, obs: Any) -> list:
        asked = self.asked_ids(call_args)
        on_cart = call_name in set(filter(None, (self.role("cart"), self.role("cart_view"))))
        epoch = self.cart_seq
        caps = self.tool_caps.setdefault(call_name, dict.fromkeys(("calls", "id", "stock", "props", "notes", "price"), 0))
        caps["calls"] += 1
        got: set = set()
        newly: list = []

        def visit(rec: dict) -> None:
            ident = self.rec_id(rec)
            if ident is None:
                return
            state = self.stock_state(rec)
            mine = ident in asked
            if state is not None:
                self._h_book_stock(ident, state, "lookup" if mine or on_cart else "row", epoch, newly)
            if mine:
                if state is not None:
                    got.add("stock")
                self._h_book_read(call_name, ident, rec, got)
            elif not on_cart:
                self.row_keys.update(rec)
                self.row_text.setdefault(ident, {}).update((k, v) for k, v in rec.items() if isinstance(v, str))
        refused = isinstance(obs, dict) and obs.get("error")
        if not refused:
            each_dict(obs, visit)
        for name in ("id", "stock", "props", "notes", "price"):
            if name in got:
                caps[name] += 1
        return newly

    def stock_tool(self) -> str | None:
        cands = self.lookup_tools()
        view, details = self.role("view"), self.role("details")
        count = self._h_count
        stocked = [t for t in cands if count(t, "stock") > 0]
        if stocked:
            def standing(t: Any) -> tuple:
                return count(t, "stock"), t == view, t == details, t
            return max(stocked, key=standing)
        if self.role_guess is None:
            try:
                self.role_guess = self.code_roles() or {}
            except Exception:  # noqa: BLE001
                self.role_guess = {}
        guess = self.role_guess
        for t in (view, guess.get("view"), details):
            if not t or t not in cands:
                continue
            if count(t, "id") < 2 or count(t, "stock") != 0:
                return t
        busy = set()
        for r in ("cart", "cart_view", "order", "decline", "message"):
            busy.add(guess.get(r))
        return next((t for t in cands if t not in busy and count(t, "id") == 0), None)

    def notes_tool(self) -> str | None:
        details = self.role("details")
        plain_keys = self.tool_keys.get(details) or set()
        found = [t for t in self.lookup_tools() if t != details and (self.note_keys.get(t) or set()) - plain_keys]
        if not found:
            return None
        return max(found, key=lambda t: (self._h_count(t, "notes"), t))

    def dead_lines(self) -> set:
        return set(self.dead)

    def stale_lines(self) -> list:
        return [line for line in self.trolley.units if self.line_seq.get(line, -1) < self.cart_seq]

    # ================================================================================== 4. cart change bookkeeping
    def cart_moved(self, before: dict) -> None:
        units = self.trolley.units
        self.since_cart = 0
        self.cart_ver += 1
        if not set(units).issubset(before):
            self.cart_seq = self.absorb_n
        grown = [line for line, n in units.items() if line in before and n > before[line] + 1e-9]
        for line in grown:          # held lines whose units grew
            self.line_seq.pop(line, None)

    def mark_fresh(self, obs: Any) -> None:
        if not self.stock_seen:
            return

        def visit(rec: dict) -> None:
            ident = self.rec_id(rec)
            if ident is not None and ident in self.trolley.units and self.stock_state(rec) is not None:
                self.line_seq[ident] = self.absorb_n
        each_dict(obs, visit)

    def after_put(self, call_args: dict, obs: Any) -> None:
        cart = self.trolley
        if cart.view_units(obs) is not None:
            self.cart_shown = self.cart_ver
        new_lines = []
        for ref, n in cart.lines(call_args):
            if n is None or n > 0:
                if ref in cart.units and ref not in self.first_puts:
                    new_lines.append(ref)
        if not new_lines or not cart.is_put(call_args) or self.code_fault(obs):
            return
        self.first_puts.update(new_lines)
        stale = set(self.stale_lines())
        due = [ref for ref in new_lines if ref in stale]
        tool = self.stock_tool()
        if not due or not tool:
            return
        wire = self.wire
        if wire.mute or self.covering or self.moves_left() < 4 or wire.left() < Knobs.WIND_DOWN_S:
            return
        queue = list(self.put_queue or [])
        queue += [ref for ref in due if ref not in queue]
        self.put_queue = queue[:4]
        self.put_tool = tool
        self._h_note("after_put_read", ids=list(self.put_queue), tool=tool)

    def cart_gone(self, lines: Any) -> set:
        lines = list(lines)
        gone = {line for line in lines if line in self.dead}
        stale = [line for line in self.stale_lines() if line in lines and line not in gone]
        if stale and self.stock_tool() and self.moves_left() >= len(stale) + 2:
            _, sold = self.code_recheck(stale)
            gone |= set(sold) | {line for line in lines if line in self.dead}
        return gone

    # ================================================================================== 5. sold-out handling
    def _h_cart_rest(self, line: str) -> float:
        """What the cart's other lines cost (lines of unknown price left out)."""
        rest = 0.0
        for other, units in self.trolley.units.items():
            if other == line:
                continue
            c = self._h_cost(other)
            if c != float("inf"):
                rest += c * units
        return rest

    def _h_read_ok(self, ident: str, row: dict, judged: list) -> bool:
        """A read listing's record: no ruled fault, no clash with a plain stated value, and every such value shown."""
        if self.field_view(ident).get("fails"):
            return False
        fields = row.get("fields") or {}
        if any(self.code_clash(fields, p, v) for p, v in judged):
            return False
        return all(self._h_shows(row, p, v) for p, v in judged)

    @staticmethod
    def _h_money(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else str(x)

    def dead_swap(self, line: str) -> tuple:
        dead = self.dead_lines()
        kind = self.shelf_known(line) or ""
        needs = self.needs
        several = bool(needs.items())
        cap = needs.spend_cap
        qty = max(1, self.trolley.held(line))
        in_cart = self.trolley.units
        rest = self._h_cart_rest(line)

        def taken(ident: str) -> bool:
            return ident == line or ident in dead or ident in self.refused_lines or ident in in_cart

        def too_dear(cost: float) -> bool:
            return numeric(cap) and cost * qty + rest > cap
        fits = []
        for ident in self._h_passed_now():
            if taken(ident) or self.field_view(ident).get("fails"):
                continue
            shelf = self.shelf_known(ident) or ""
            if kind and shelf != kind and (several or shelf != ""):
                continue
            cost = self._h_cost(ident)
            if cost == float("inf") or not too_dear(cost):
                fits.append((cost, ident))
        if fits:
            return min(fits)[1], ""
        if several and not kind:
            return "", ""
        k = next((i for i, home in enumerate(self.home_shelves()) if kind in home), None) if several else None
        judged = [(p, v) for p, v in self.code_values(k) if self.ed_plain(v)]
        failed = self._h_failed()
        pointers = self.ref_pointers()
        kept = []
        for ident, row in self.code_rows().items():
            if taken(ident) or row.get("stock") is not True or (kind and row.get("shelf") != kind):
                continue
            price = row.get("price")
            if not numeric(price):
                price = self._h_cost(ident)
            if price == float("inf") or too_dear(price):
                continue
            if ident in failed or ident in pointers or self.ref_excluded(row.get("fields") or {}, k):
                continue
            flag = 1
            if ident in self.archive.read:
                if not self._h_read_ok(ident, row, judged):
                    continue
                flag = 0
            kept.append((flag, price, ident))
        kept.sort()
        if needs.priority() == "cheapest":
            low = min((r for r in kept if r[0] == 1), key=lambda r: r[1:], default=None)
            if low is not None and all(r[1] > low[1] for r in kept if r[0] == 0):
                kept.remove(low)
                kept.insert(0, low)
        top = kept[:3]
        self.swap_unread = [ident for flag, _, ident in top if flag == 1]
        page = None
        if all(r[0] != 0 for r in kept):
            try:
                page = self.dead_page(k)
            except Exception:  # noqa: BLE001
                page = None
        self.swap_page = page
        shown = []
        for flag, price, ident in top:
            note = "; its record shows every stated value" if flag == 0 else ""
            shown.append("%s (%s%s)" % (ident, self._h_money(price), note))
        return "", ", ".join(shown)

    def dead_page(self, k: Any) -> dict | None:
        spec = self.page_arg()
        if not spec:
            return None
        best = None
        best_rank = None
        for walk in self.page_book.values():
            carried = self.walk_carries(walk.get("args") or {}, k)
            if not carried:
                continue
            nxt = self.next_page(walk)
            if nxt is None:
                continue
            rank = (len(carried), -len(walk.get("read") or ()))
            if best_rank is None or rank > best_rank:
                best, best_rank = (walk, nxt), rank
        if best is None:
            return None
        args = dict(best[0].get("args") or {})
        args[spec[0]] = best[1]
        return args

    def swap_open(self) -> str:
        if self.swap_done:
            return ""
        for ident in self.swap_unread:
            if ident not in self.archive.read and ident not in self.dead and ident not in self.trolley.units:
                return ident
        return ""

    def dead_note(self, lines: Any) -> str:
        self.event_seen = True
        cheapest = self.needs.priority() == "cheapest"
        parts = []
        for line in lines:
            self.swap_page = None
            self.swap_unread = []
            pick, rows = self.dead_swap(line)
            text = ("%s is OUT OF STOCK now (its newest record shows it), so an order holding it fails - take it out of "
                    "the cart." % line)
            if pick:
                text += (" Put %s in instead: the item check found it meets every requirement now in force%s. Read its "
                         "stock and price again, then order." % (pick, ", and it is the cheapest such listing" if cheapest
                                                                  else ""))
            elif rows:
                text += (" No other verified fit of its product is known, so check these cheapest in-stock rows of its "
                         "category within the budget: %s (read rows that lack or fail a stated requirement were left "
                         "out; an unread one needs its details read first). Put the first that fits in, read its stock "
                         "and price again, then order." % rows)
            else:
                text += " Find another listing that fits."
            if self.swap_page and not pick:
                text += (" More rows of its kind are on the next unread page of the search led by the stated value: "
                         + as_text(self.swap_page))
            parts.append(text)
        return " ".join(parts)

    # ================================================================================== 6. refused orders
    def refusal_reread(self, obs: Any = None) -> str:
        try:
            heard, gone = self.refusal_gone(obs)
        except Exception as error:  # noqa: BLE001
            self.log.append({"kind": "refusal_gone_error", "error": str(error)[:200]})
            heard, gone = "", ""
        if gone:
            return heard + " " + gone
        queued = self.refusal_queue()
        return heard + (" " + queued if queued else "")

    def refusal_gone(self, obs: Any) -> tuple:
        held = [line for line, n in self.trolley.units.items() if n > 0]
        if obs is None or self.last_kind != "order" or len(held) != 1:
            return "", ""
        if self.wire.mute or self.covering:
            return "", ""
        texts = self.shopper_lines(self.role("order"), obs, True)
        if not texts:
            return "", ""
        needs = self.needs

        def snapshot() -> tuple:
            return str(needs.spend_cap), len(needs.let_go), len(needs.uncovered), " | ".join(needs.props_left())
        ver, before = needs.version, snapshot()
        note = needs.hear(texts)
        heard = " " + note if note else ""
        if not note or needs.version == ver or snapshot() != before:
            return heard, ""
        line = held[0]
        self.dead.add(line)
        self.dead_by[line] = "lookup"
        self._h_note("refusal_dead", line=line)
        lead = ("The refusal says an item became unavailable and the cart held one line only, so the code takes %s as "
                "sold out without reading it again. " % line)
        return heard, lead + self.dead_note([line])

    def refusal_queue(self) -> str:
        tool = self.stock_tool()
        lines = list(self.trolley.units)[:4]
        if (not lines or not tool or self.wire.mute or self.covering or self.open_queue
                or self.moves_left() < len(lines) + 3 or self.wire.left() < Knobs.WIND_DOWN_S):
            return ""
        self.open_queue = lines
        self.open_recheck = True
        self.open_purpose = "cart"
        self.open_tool = tool
        self.open_say = "The order was refused, so I'm reading the stock of the cart's lines again."
        return "The code now reads the stock of the cart lines for you with %s." % tool

    # ================================================================================== 7. the code's own evidence
    @staticmethod
    def ed_plain(value: Any) -> bool:
        low = str(value or "").lower()
        return not (H_LINKED.search(low) or H_RANGE.search(low) or any(c in low for c in "+<>") or H_ROUGH.search(low))

    @staticmethod
    def ed_word(value: Any) -> bool:
        low = str(value or "").lower()
        return (bool(re.search(r"[a-z]{3,}", low)) and not re.search(r"\d", low) and CartworkMixin.ed_plain(low)
                and not EXCLUSION.search(low) and not points_elsewhere(low))

    def ed_reqs(self, k: Any) -> list:
        needs = self.needs
        several = bool(needs.items())
        written = needs.written
        out = []
        for line in needs.props_left():
            head, _, value = str(line).partition(":")
            if several and self.product_of(head) not in (None, k):
                continue
            name = head.split("(")[0].strip()
            out.append((name, H_MARKS.sub("", value).strip(), line not in written))
        return out

    def code_judges_all(self) -> bool:
        n = len(self.needs.items())
        plain = 0
        for k in (range(n) if n else [None]):
            plain += sum(1 for _, v in self.code_values(k) if self.ed_plain(v))
        reqs = self.needs.props_left()
        resolved = self.differ_done(None)
        judged = len([r for r in reqs if any(self.ref_match(rec, r) for rec in resolved)])
        return plain + judged >= len(reqs)

    def differ_done(self, k: Any) -> list:
        needs = self.needs
        several = bool(needs.items())
        reqs = needs.props_left()

        def settled(rec: Any) -> bool:
            if (rec.get("relation"), rec.get("status")) != ("differ", "resolved"):
                return False
            return bool(str(rec.get("value") or "").strip()) and self.ref_live(rec)

        def tagged_elsewhere(rec: Any) -> bool:
            tag = rec.get("tag")
            if not (several and k is not None and tag):
                return False
            return self.product_of("x (%s)" % tag) not in (None, k)
        return [rec for rec in self.refs
                if settled(rec) and any(self.ref_match(rec, r) for r in reqs) and not tagged_elsewhere(rec)]

    @staticmethod
    def _h_at(seq: Any, k: Any, default: Any) -> Any:
        return seq[k] if isinstance(k, int) and 0 <= k < len(seq) else default

    @staticmethod
    def _h_plus_parts(v: str) -> list:
        """The parts of a value joined by '+' (two or more, each with a word), else []."""
        if "+" not in v:
            return []
        bits = []
        for b in v.split("+"):
            b = b.strip()
            if not b:
                continue
            if not re.search(r"[a-z]{2,}", b.lower()):
                return []
            bits.append(b)
        return bits if len(bits) > 1 else []

    def _h_shows(self, row: dict, prop: Any, value: Any) -> bool:
        """The listing's own row (read) writes the value for the property."""
        return self.code_shows(row, prop, value, set(), True)

    def _h_row_price(self, ident: str, row: dict) -> float | None:
        price = row.get("price")
        if price is None:
            price = self.archive.price(ident)
        return float(price) if numeric(price) else None

    def _h_clashes(self, fields: dict, judged: list, joins: list) -> list:
        """Properties whose own fields name another value than the plain requested one (or miss a part of a joined
        one)."""
        out = []
        for p, v in judged:
            if not self.code_clash(fields, p, v):
                continue
            if not any(self.code_seen(v, fv) for fk, fv in fields.items() if loosely_same(fk, p)):
                out.append(p)
        for p, bits in joins:
            named = [fv for fk, fv in fields.items() if loosely_same(fk, p)]
            # every part must show in some field of the property (not necessarily the same one)
            if named and any(not any(self.code_seen(b, fv) for fv in named) for b in bits):
                out.append(p)
        return out

    def ed_pool(self, k: Any) -> list:
        needs, shelf_book = self.needs, self.archive
        reqs = needs.props_left()
        ruled_n = len([r for r in self.rulings.values() if r])
        memo_key = (k, needs.version, str(needs.spend_cap), " | ".join(reqs), len(shelf_book.items), len(shelf_book.read),
                    len(self.passed), ruled_n, len(self.refs), tuple(sorted(self.dead)), tuple(sorted(self.refused_lines)))
        memo = self.ed_memo
        hit = memo.get(memo_key)
        if hit is not None:
            return hit
        if 64 < len(memo):
            memo.clear()
        several = bool(needs.items())
        if several:
            home = self._h_at(self.home_shelves(), k, set())
        else:
            home = self.shelf_labels() or self._h_at(self.home_shelves(), 0, set())
        entries: list = []
        if not home:
            memo[memo_key] = entries
            return entries
        qty = self._h_at(needs.counts(), k if several else 0, 1)
        pointers = self.ref_pointers()
        cap = needs.spend_cap if numeric(needs.spend_cap) else None
        shown = [(p, v) for p, v, revealed in self.ed_reqs(k) if revealed]
        values = self.code_values(k if several else None)
        joins = [(p, bits) for p, v in values if (bits := self._h_plus_parts(v))]
        judged = [(p, v) for p, v in values if self.ed_plain(v)]
        live = self._h_live()
        failed = self._h_failed()
        rows = [(i, r) for i, r in self.code_rows().items() if r.get("shelf") in home]
        for ident, row in rows:
            fields = row.get("fields") or {}
            read = ident in shelf_book.read
            price = self._h_row_price(ident, row)
            gone = row.get("stock") is False or ident in self.dead or ident in self.refused_lines
            over = False if cap is None or price is None else qty * price > cap + 1e-9
            pointer = True if ident in pointers else self.ref_excluded(fields, k)
            clash = self._h_clashes(fields, judged, joins)
            misses = [(p, v) for p, v in judged if not self._h_shows(row, p, v)]
            if ident in failed:
                ruled = True
            else:
                ruled = read and bool(self.field_view(ident).get("fails"))
            verified = not gone and ident in live
            barred = ruled or pointer or gone or over
            fit = not barred and (verified or (read and not (misses or clash)))
            entries.append(dict(id=ident, price=price, read=read, verified=verified, fit=fit, gone=gone, over=over,
                                pointer=pointer, ruled=ruled, clash=clash, misses=misses, row=row,
                                out=barred or bool(clash),
                                revealed_ok=all(self._h_shows(row, p, v) for p, v in shown)))
        memo[memo_key] = entries
        return entries

    @staticmethod
    def _h_all_out(pool: list) -> bool:
        outs = {bool(e["out"]) for e in pool if e["read"]}
        return outs == {True}

    def ed_proof(self) -> list:
        n = len(self.needs.items())
        return [k for k in range(n) if self._h_all_out(self.ed_pool(k))]

    def verify_problems(self, ident: str) -> list | None:
        trace = self.log
        at = len(trace)
        while at:
            at -= 1
            entry = trace[at]
            if not isinstance(entry, dict) or entry.get("item") != ident or entry.get("kind") != "verify":
                continue
            result = entry.get("result")
            if not isinstance(result, dict) or result.get("fits") is not False:
                return None
            texts = map(str, result.get("problems") or [])
            return [t for t in texts if t.strip()]
        return None

    def ed_news(self, k: Any) -> bool:
        if self.refusal_n > 0:
            return True
        needs = self.needs
        news = list(needs.uncovered) + list(needs.let_go)
        if not news:
            return False
        if not needs.items():
            return True
        return any(self.product_of(str(line).split(":", 1)[0]) in (None, k) for line in news)

    # ================================================================================== 8. holds before a decline
    def _h_note(self, kind: str, **fields: Any) -> None:
        self.log.append(dict(kind=kind, **fields))

    def _h_proven(self) -> bool:
        """Several products: the records already prove some product has no fitting listing (the decline stands)."""
        try:
            proof = self.ed_proof()
        except Exception as error:  # noqa: BLE001
            self._h_oops("decline_proof_error", error)
            return False
        if not proof:
            return False
        self._h_note("decline_proof", products=proof)
        return True

    def _h_settle_picks(self, picks: list, avoid: set, seen: dict, vetted: Any, ver: str) -> list:
        """Several products: steer the code's picks off listings whose rows lack a revealed value, and have the item
        check judge the first unjudged pick (at most 3 checks)."""
        checks = rounds = 0
        wire = self.wire
        while picks and checks < 3 and rounds < 30:
            if wire.left() <= Knobs.WIND_DOWN_S + 30:
                break
            unchecked = [p for p in picks if not vetted(p[0])]
            if not unchecked:
                break
            bare = []
            for ident, _ in unchecked:
                e = seen.get(ident)
                if e is None or not e["revealed_ok"] or e["misses"]:
                    bare.append(ident)
            if not bare:
                checks += 1
                self.check_item(unchecked[0][0], "ORDER")
            else:
                avoid.update(bare)
                for i, e in seen.items():
                    if e["read"] or "%s|%s" % (i, ver) in self.passed:
                        continue
                    if e["misses"] or not e["revealed_ok"]:
                        avoid.add(i)
                rounds += 1
            picks = self.code_picks(avoid)
        return picks

    def decline_repair(self) -> str:
        items = self.needs.items()
        several = bool(items)
        left, due, refusals = self.moves_left(), self.order_due(), self.refusal_n
        early = several and refusals == 0 and left > due
        if refusals <= 0 and left > due and not early:
            return ""
        wire = self.wire
        if left < 3 or wire.mute or wire.done or self.covering or wire.left() <= Knobs.WIND_DOWN_S:
            return ""
        if not (self.role("cart") and self.role("order")):
            return ""
        mark = ("early", refusals) if early else refusals
        if mark in self.repair_tried:
            return ""
        self.repair_tried.add(mark)
        if several and self._h_proven():
            return ""
        dead = self.dead_lines()
        skip = dead | set(self.stale_lines())
        alive = [line for line in self.trolley.units if line not in skip]
        saved = set(self.refused_lines)
        ver = self._h_ver()
        own = self.code_judges_all()

        def vetted(p: str) -> bool:
            if "%s|%s" % (p, ver) in self.passed:
                return True
            return own and p in self.archive.read and bool(self.code_fit.get(p))

        def fitting(picks: list) -> bool:
            for p, _ in picks:
                if not vetted(p):
                    return False
            return bool(picks)
        done = False
        try:
            self.refused_lines.difference_update(alive)
            avoid = set(dead)
            seen: dict = {}
            if several:
                for k in range(len(items)):
                    seen.update((e["id"], e) for e in self.ed_pool(k))
                avoid.update(i for i, e in seen.items() if e["clash"])
            picks = self.code_picks(avoid)
            if several:
                picks = self._h_settle_picks(picks, avoid, seen, vetted, ver)
            why = "" if fitting(picks) else "unfit"
            if not why and self.moves_left() < len(picks) + 2:
                why = "steps"
            if why:
                self._h_note("decline_repair", why=why, picks=picks)
            else:
                done = self._h_cart_exactly(picks)
                self._h_note("decline_repair", picks=picks, cart=dict(self.trolley.units), done=done)
        except Exception as error:  # noqa: BLE001
            done = False
            self._h_oops("decline_repair_error", error)
        if done:
            whom = "for every product asked for" if several else "for the item"
            return ("Not run yet (no step used): a decline is right only when nothing can fit, and the records hold a "
                    "fitting candidate %s - passed by the item check, or read with every stated value in its own record "
                    "- so the code put %s in the cart and read the stock: %s Order this cart now; decline only if a "
                    "record you read shows one of these lines failing a requirement now in force."
                    % (whom, "them" if several else "it", self.trolley.summary()))
        self.refused_lines.clear()          # the same set object, as it was
        self.refused_lines.update(saved)
        return ""

    def _h_value_lead(self, k: Any) -> dict:
        """Row id -> how many stated values of product k the search that listed it carried (a found row: at least 1)."""
        value_sets = []
        for _, v, _ in self.ed_reqs(k):
            value_sets.append({w for w in re.findall(r"[a-z0-9]{2,}", v.lower()) if w not in FILLER})
        led = dict.fromkeys(self.all_hits, 1)
        cut = self.lead_words
        for walk in self.page_book.values():
            text = " ".join(self.free_text(walk.get("args") or {})).lower()
            words = re.findall(r"[a-z0-9]{2,}", text)
            said = set(words[:cut] if cut else words)
            score = len([vs for vs in value_sets if vs & said])
            if not score:
                continue
            for i in walk.get("row_shelf") or {}:
                led[i] = max(led.get(i, 0), score)
        return led

    def slack_rows(self) -> list:
        try:
            needs = self.needs
            items = needs.items()
            cap = needs.spend_cap
            if len(items) < 2 or not numeric(cap):
                return []
            counts = needs.counts()
            pools = [self.ed_pool(k) for k in range(len(items))]
            floors = []
            for pool, count in zip(pools, counts):
                low = min((e["price"] for e in pool if e["fit"] and e["price"] is not None), default=None)
                if low is None:
                    return []
                floors.append(low * count)
            total = sum(floors)
            if total <= cap + 1e-9:
                return []
            ranked = []
            for k, pool in enumerate(pools):
                slack = (cap - (total - floors[k])) / max(1, counts[k])
                led = self._h_value_lead(k)
                fresh = []
                if slack > 0:
                    for e in pool:
                        if e["read"] or e["gone"] or e["pointer"] or e["ruled"] or e["clash"]:
                            continue
                        if e["price"] is not None and e["price"] <= slack + 1e-9:
                            fresh.append(e)

                def order(e: dict) -> tuple:
                    return -led.get(e["id"], 0), e["price"], e["id"]
                ranked.append([e["id"] for e in sorted(fresh, key=order)])
            taken = self._h_take_turns(ranked, 3)
            if taken:
                self.slack_total = total
            return taken
        except Exception as error:  # noqa: BLE001
            self._h_oops("slack_error", error)
            return []

    def _h_shelves_held(self) -> set:
        """Categories of the listings that passed the item check (and of the audit's fits)."""
        live = self._h_live() - set(self.refused_lines) - self.dead
        got = {self.shelf_known(i) for i in [*live, *(self.audit_fits or [])]}
        got.discard("")
        return got

    def lacking_rows(self) -> list:
        try:
            n = len(self.needs.items())
            if not n:
                return []
            got = self._h_shelves_held()
            homes = self.home_shelves()
            lacking = [k for k in range(min(n, len(homes))) if homes[k] and homes[k].isdisjoint(got)]
            if not lacking:
                return []
            batch = self.batch_tool()
            most = batch[2] if batch else 3
            shelf_book = self.archive
            where: dict = {}        # id -> its category on its newest record
            sold: set = set()

            def visit(rec: dict) -> None:
                ident = self.row_mark(rec)
                if not isinstance(ident, str) or ident not in shelf_book.ids:
                    return
                label = self.shelf_of(rec)
                if not label:
                    return
                where.setdefault(ident, label)
                if self.sold_out(rec):      # only a record carrying a category counts
                    sold.add(ident)
            for obs in reversed(shelf_book.items):
                each_dict(obs, visit)
            hits = sorted(self.all_hits, key=lambda i: self.hit_score.get(i, 0), reverse=True)
            cands = list(dict.fromkeys([h for h in hits if h in where] + list(where)))
            cap = self.needs.spend_cap
            counts = self.needs.counts()
            pools: dict = {k: [] for k in lacking}
            for ident in cands:
                if ident in shelf_book.read or ident in self.refused_lines or ident in sold:
                    continue
                label = where.get(ident)
                k = next((j for j in lacking if label in homes[j]), None)
                if k is None:
                    continue
                price = shelf_book.price(ident)
                if numeric(cap) and numeric(price) and price * counts[k] > cap:
                    continue
                pools[k].append(ident)
            return self._h_take_turns([pools[k] for k in lacking], most)
        except Exception as error:  # noqa: BLE001
            self._h_oops("lacking_error", error)
            return []

    def _h_oops(self, kind: str, error: Exception) -> None:
        self.log.append({"kind": kind, "error": str(error)[:200]})

    def _h_held_near(self) -> list:
        """The lines of the basket a meaning check confirmed, while nothing has changed since; else []."""
        info = self.near_cart
        if not info or not self.role("order"):
            return []
        if (info.get("version"), info.get("refusals")) != (self.needs.version, self.refusal_n):
            return []
        lines = list(info.get("ids") or [])
        units = self.trolley.units
        for line in lines:
            if line not in units or line in self.dead or line in self.refused_lines:
                return []
        return lines

    def near_hold(self) -> str:
        try:
            lines = self._h_held_near()
            if not lines:
                return ""
            wire = self.wire
            if wire.done or wire.broke or self.moves_left() < 1 or wire.left() < 2:
                return ""
            self.near_cart = None
            self.log.append({"kind": "near_order", "ids": lines})
            muted = wire.mute
            try:
                self.finish_now()
            finally:
                wire.mute = muted
            joined = ", ".join(lines)
            return ("Not run (no step used): a meaning check confirmed that the cart's lines (%s) write the requested "
                    "values in other words, and nothing has changed since, so the code ordered the cart instead of "
                    "declining." % joined)
        except Exception as error:  # noqa: BLE001
            self._h_oops("near_hold_error", error)
            return ""

    def near_check(self) -> str:
        wire = self.wire
        blocked = self.near_done or self.covering or wire.mute or wire.done or wire.no_model
        if blocked or self.moves_left() < 4 or not (self.role("cart") and self.role("order")):
            return ""
        if wire.left() <= Knobs.WIND_DOWN_S + 30 or wire.spent_out():
            return ""
        try:
            return self.near_basket()
        except Exception as error:  # noqa: BLE001
            self._h_oops("near_error", error)
            return ""

    def _h_reworded(self, e: dict, reqs: list) -> list:
        """(property, value, what the listing's own field says) for each requested value a read listing writes only
        in other words; [] when the listing is not such a near fit."""
        row = e["row"]
        fields = row.get("fields") or {}
        out = []
        for prop, value, revealed in reqs:
            if self._h_shows(row, prop, value):
                continue
            if revealed:
                return []
            says = []
            for fk, fv in fields.items():
                text = str(fv).strip()
                if text and loosely_same(fk, prop):
                    says.append(text)
            if not says or not self.ed_word(value) or any(re.search(r"\d", t) for t in says):
                return []
            out.append((prop, value, "; ".join(dict.fromkeys(says))[:120]))
        return out

    def _h_near_cleared(self, e: dict, lines: list) -> bool:
        """A ruled-out listing stays a near fit only when every fault the check named is one of the reworded values."""
        if not e["ruled"]:
            return True
        ident = e["id"]
        fails = [str(f).split(":", 1)[0].strip() for f in self.field_view(ident).get("fails") or []]
        probs = self.verify_problems(ident) or []
        for f in fails:
            if not any(loosely_same(f, prop) for prop, _, _ in lines):
                return False
        if not probs:
            return True
        marks: set = set()
        for prop, value, _ in lines:
            marks.update(w for w in name_words(prop) if len(w) >= 3 and w not in FILLER)
            marks.update(w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 3 and w not in FILLER)
        for pr in probs:
            low = str(pr).lower()
            if not any(re.search(r"\b" + re.escape(m), low) for m in marks):
                return False
        return True

    @staticmethod
    def _h_said_same(answer: Any) -> set:
        got = answer.get("lines") if isinstance(answer, dict) else None
        yes: set = set()
        if not isinstance(got, list):
            return yes
        for ent in got:
            if not isinstance(ent, dict) or ent.get("same") is not True:
                continue
            n = ent.get("n")
            if numeric(n):
                yes.add(int(n) - 1)
        return yes

    def near_basket(self) -> str:
        needs = self.needs
        items = needs.items()
        several = bool(items)
        total = len(items) or 1
        counts = needs.counts()
        units = (list(counts) + [1] * total)[:total]
        cap = needs.spend_cap if numeric(needs.spend_cap) else None
        chosen: dict = {}
        pools: dict = {}
        for n in range(total):
            k = n if several else None
            pool = self.ed_pool(k)
            priced = sorted((e["price"], e["id"]) for e in pool if e["fit"] and e["price"] is not None)
            if priced:
                low, ident = priced[0]
                chosen[n] = (ident, low, units[n])
                continue
            if not self.ed_news(k):
                return ""
            reqs = self.ed_reqs(k)
            cands = []
            for e in pool:
                if e["gone"] or e["over"] or e["pointer"] or not e["read"] or e["price"] is None:
                    continue
                lines = self._h_reworded(e, reqs)
                if lines and self._h_near_cleared(e, lines):
                    cands.append((len(lines), e["price"], e["id"], lines))
            if not cands:
                return ""
            cands.sort(key=lambda c: c[:3])
            pools[n] = cands[:3 if several else 6]
        if not pools:
            return ""
        if cap is not None:
            fixed = sum(price * count for _, price, count in chosen.values())
            lows = sum(min(c[1] for c in pools[n]) * units[n] for n in pools)
            if fixed + lows > cap + 1e-9:
                return ""
        flat = [(n, c[2]) + line for n in sorted(pools) for c in pools[n] for line in c[3]][:12]
        numbered = []
        for i, (_, ident, prop, value, says) in enumerate(flat, 1):
            numbered.append("%d. %s: %s = '%s' | the listing's own field reads: %s" % (i, ident, prop, value[:60], says))
        intro = ("A shopping assistant is about to tell the shopper that nothing fits, because the listings below do "
                 "not write a requested value in the request's own words. The shopper's request, for context:\n")
        rules = ("\n\nEach numbered line gives a listing, a requested property with the requested value, and what that "
                 "listing's own field says. For each line decide whether the field names the requested value in other "
                 "words (a synonym, another spelling or wording, or a more exact form of the same thing) or names a "
                 "different value. When unsure, answer false.\n\nLINES:\n")
        reply_form = '\n\nAnswer with ONE JSON object and nothing else: {"lines": [{"n": line number, "same": true or false}]}.'
        self.near_done = True
        yes = self._h_said_same(self.wire.ask_json("".join([intro, self.request[:1500], rules, "\n".join(numbered),
                                                            reply_form]), 600, LINEUP["checker"]))
        self.log.append({"kind": "near_check", "lines": len(flat), "same": sorted(yes)})
        for n in sorted(pools):
            for c in pools[n]:
                rows = [i for i, f in enumerate(flat) if f[:2] == (n, c[2])]
                if len(rows) == len(c[3]) and yes.issuperset(rows):
                    chosen[n] = (c[2], c[1], units[n])
                    break
            else:
                return ""
        if cap is not None and sum(price * count for _, price, count in chosen.values()) > cap + 1e-9:
            return ""
        picks = []
        for n in range(total):
            ident, _, count = chosen[n]
            picks.append((ident, count))
        if self.moves_left() < len(picks) + 2:
            return ""
        ver = self._h_ver()
        for n in pools:
            ident = chosen[n][0]
            key = "%s|%s" % (ident, ver)
            older = [rk for rk in self.rulings if rk == key or str(rk).startswith(key + "|")]
            for rk in older:
                self.rulings[rk] = ""
            self.passed.add(key)
            self.near_ok.add(key)
            self.pass_price[ident] = self.archive.price(ident)
        done = self._h_cart_exactly(picks)
        self.log.append({"kind": "near_basket", "picks": picks, "cart": dict(self.trolley.units), "done": done})
        if not done:
            return ""
        self.near_cart = {"version": needs.version, "refusals": self.refusal_n, "ids": [ident for ident, _ in picks]}
        said = []
        for row in flat:
            n, ident = row[:2]
            if ident != chosen[n][0]:
                continue
            self.near_said.setdefault("%s|%s" % (ident, ver), []).append(row[2:])
            _, _, prop, value, says = row
            said.append("%s writes %s as '%s' for '%s'" % (ident, prop, says, value[:60]))
        return ("Not run yet (no step used): a meaning check found that %s - the same value in other words - and each "
                "such listing meets every revealed need, is in stock and within the budget, so the code put the basket "
                "in the cart and read its stock: %s Order this cart now; decline only if a record you read shows one of "
                "these lines failing a requirement now in force." % ("; ".join(said), self.trolley.summary()))

    # ================================================================================== 9. listings for the menu
    @staticmethod
    def _h_title_words(title: Any) -> set:
        return set(re.findall(r"[a-z0-9]{3,}", str(title or "").lower()))

    def _h_led_by_value(self, home: set, most: int, facts: dict, cands: list) -> str | None:
        """When no read listing of the shelf shows every plain stated value: the first candidate row of the search
        that carried the most stated values."""
        try:
            several = bool(self.needs.items())
            k = None
            if several:
                k = next((i for i, h in enumerate(self.home_shelves()) if set(h) == home), None)
            vals = []
            for line in self.needs.props_left():
                head, _, value = str(line).partition(":")
                if several and self.product_of(head) not in (None, k):
                    continue
                v = H_MARKS.sub("", value).strip()
                low = v.lower()
                if v and names_the_item(v) and all(s not in low for s in H_POINT_WORDS):
                    vals.append(v)
            if not vals or most <= 0 or (several and k is None):
                return None
            for f in facts.values():
                pool = " | ".join(f)
                if all(seen_in_samples(v, pool) for v in vals):
                    return None
            led = [(len(c), w) for w in self.page_book.values() if (c := self.walk_carries(w.get("args") or {}, k))]
            led.sort(key=lambda pair: pair[0], reverse=True)
            for _, walk in led:
                first = next((i for i in (walk.get("row_shelf") or {}) if i in cands), None)
                if first is not None:
                    return first
            return None
        except Exception:  # noqa: BLE001
            return None

    def menu_rows(self, home: Any, most: int, taken: Any, qty: Any = 1) -> list:
        home = set(home or ())
        shelf_book = self.archive
        known: list = []        # title words of the listings read (and of those picked)
        facts: dict = {}        # read listing -> the values of its property maps
        titles: dict = {}       # unread listing -> its first title, in first-seen order
        sold: set = set()

        def visit(rec: dict) -> None:
            ident = self.row_mark(rec)
            if not isinstance(ident, str) or ident not in shelf_book.ids or self.shelf_of(rec) not in home:
                return
            if ident not in shelf_book.read:
                if self.sold_out(rec):
                    sold.add(ident)
                if ident not in titles:
                    titles[ident] = self.title_of(rec)
                return
            known.append(self._h_title_words(self.title_of(rec)))
            bag = facts.setdefault(ident, [])
            for v in rec.values():
                if isinstance(v, dict):
                    bag.extend(str(x) for x in v.values() if isinstance(x, (str, int, float, bool)))
        for obs in shelf_book.items:
            each_dict(obs, visit)
        hits = sorted(self.all_hits, key=lambda i: self.hit_score.get(i, 0), reverse=True)
        hit_set = set(hits)
        blocked = sold | set(self.refused_lines) | set(taken or ())
        cap = self.needs.spend_cap
        limit = cap if numeric(cap) else None
        cands: list = []
        for ident in [h for h in hits if h in titles] + [i for i in titles if i not in hit_set]:
            if ident in blocked or ident in cands:
                continue
            price = shelf_book.price(ident)
            if limit is not None and numeric(price) and price * qty > limit:
                continue
            cands.append(ident)
        words = {i: self._h_title_words(titles.get(i)) for i in cands}
        lead = self.row_score
        picks: list = []
        first = self._h_led_by_value(home, most, facts, cands)
        if first is not None:
            picks.append(first)
            known.append(words[first])

        def rank(ident: str) -> tuple:
            if ident in hit_set:
                tier = 0
            elif lead.get(ident, 0) > 0:
                tier = 1
            else:
                tier = 2
            mine = words[ident]
            overlap = 0.0
            for seen in known:
                union = len(mine | seen)
                if union:
                    overlap = max(overlap, len(mine & seen) / union)
            return tier, overlap, cands.index(ident)
        while len(picks) < most:
            pool = [i for i in cands if i not in picks]
            if not pool:
                break
            best = min(pool[:16], key=rank)
            picks.append(best)
            known.append(words[best])
        return picks

    def _h_have(self, home: Any) -> int:
        if not home:
            return 4
        return sum(self.archive.rich_reads(label) for label in home)

    def _h_open_menu(self, picks: list, say: str) -> None:
        self.open_queue = picks
        self.open_recheck = True
        self.open_purpose = "menu"
        self.open_say = say

    def _h_pre_more(self, homes: list, qty_of: Any, width: int) -> bool:
        """The second pre-read round: more listings for products whose first reads showed few fields."""
        wants = []
        blocked = list(self.pre_picked)
        for k, home in enumerate(homes):
            have = self._h_have(home)
            if have >= 3:
                continue
            rows = self.menu_rows(home, 4 - have, blocked, qty_of(k))
            blocked.extend(rows)
            wants.append(rows)
        picks = self._h_take_turns(wants, width)
        if not picks:
            return False
        self._h_open_menu(picks, "Some products' first listings showed few fields, so I'm opening a few more before "
                                 "asking anything.")
        return True

    def pre_read(self) -> bool:
        try:
            batch = self.batch_tool()
            if not batch:
                return False
            again = self.pre_rounds == 1
            if self.pre_read_done and not again:
                return False
            if len(self.questions_log) > 1 or self.wire.left() < Knobs.WIND_DOWN_S + 60:
                return False
            if self.moves_left() <= (25 if again else 20):
                return False
            homes = self.home_shelves()
            qtys = self.needs.counts()
            several = bool(self.needs.items())
            width = batch[2]

            def qty_of(k: int) -> Any:
                return qtys[k] if k < len(qtys) else qtys[-1]
            if again:
                self.pre_rounds = 2
                return self._h_pre_more(homes, qty_of, width)
            empty: list = []
            thin = []
            for k, home in enumerate(homes):
                have = self._h_have(home)
                if have == 0:
                    empty.extend(self.menu_rows(home, 2 if several else 4, empty, qty_of(k)))
                elif have < 4:
                    thin.append((k, 4 - have))
            picks = list(empty)
            for k, n in thin:
                picks.extend(self.menu_rows(homes[k], n, picks, qty_of(k)))
            room = width * 2 if len(empty) > width and self.moves_left() > 25 else width
            picks = picks[:room]
            if not picks:
                return False
            self.pre_read_done = True
            self.pre_rounds = 1
            self.pre_picked = list(picks)
            self._h_open_menu(picks, "Before asking anything, I'm opening a few listings of each product to learn their "
                                     "fields.")
            return True
        except Exception as error:  # noqa: BLE001
            self._h_oops("pre_read_error", error)
            return False

    # ================================================================================== 10. questions to the shopper
    def product_of(self, text: Any) -> int | None:
        items = self.needs.items()
        if not items:
            return None
        names = [set(kind_list(it["product"])) for it in items]
        others = [set().union(*(o for j, o in enumerate(names) if j != k)) for k in range(len(names))]
        text = str(text or "")
        m = re.search(r"\(([^)]*)\)", text)
        for piece in ([m.group(1), text] if m else [text]):
            words = kind_list(piece)
            scores = [kind_points(own, words, shared) for own, shared in zip(names, others)]
            top = max(scores)
            if top > 0 and scores.count(top) == 1:
                return scores.index(top)
        return None

    def plan_product(self, qitem: dict) -> int | None:
        k = self.product_of(str(qitem.get("property") or ""))
        return k if k is not None else self.product_of(str(qitem.get("question") or ""))

    @staticmethod
    def _h_bare(label: Any) -> str:
        return str(label or "").split("(")[0].strip()

    @staticmethod
    def _h_samples_say(value: str, head: str, samples: str) -> bool:
        """Every word of a worded value shows in a key's example values."""
        if not samples or re.search(r"\d", value):
            return False
        words = [w for w in re.findall(r"[a-z]{3,}", value.lower()) if w not in FILLER]
        if not words or not names_the_item(value, head):
            return False
        return all(re.search(r"\b" + re.escape(w), samples) for w in words)

    def open_prop(self, k: Any, prop: Any, line: str = "", by_samples: bool = False) -> bool:
        try:
            name = self._h_bare(prop)
            if not name or not_a_property(name):
                return False
            scoped = k is not None and bool(self.needs.items())

            def mine(text: Any, tagged_only: bool = False) -> bool:
                if not scoped:
                    return True
                j = self.product_of(text)
                if j is None:
                    return not tagged_only
                return j == k
            for row in self.questions_log:
                head = self._h_head(row)
                if "[not answered]" in head:
                    continue
                if any(self.same_field(k, part, name) for part in head.split(";") if mine(part)):
                    return False
            samples = (line or "").partition("):")[2].lower()
            for req in self.needs.props_left():
                head, _, value = str(req).partition(":")
                if not mine(head):
                    continue
                if strictly_same(head, name):
                    settled = self.settles(req, samples if by_samples else "")
                elif loosely_same(head, name):
                    pool = samples or " | ".join(self.field_values(k, name))
                    settled = names_the_item(value, head) and seen_in_samples(value, pool)
                else:
                    settled = self._h_samples_say(value, head, samples)
                if settled:
                    return False
            return not any(self.same_field(k, p, name) for p in self.needs.indifferent if mine(p, True))
        except Exception:  # noqa: BLE001
            return True

    @staticmethod
    def _h_bullets(rows: list) -> str:
        return "\n".join("- %s" % x for x in rows)

    def plan_notes(self, k: Any) -> str:
        try:
            several = bool(self.needs.items())
            narrow = several and k is not None
            asked = self._h_asked_parts()
            opens = []
            for prop in self.open_pair:
                j = self.product_of(prop) if several else None
                if narrow and j not in (None, k):
                    continue
                if str(prop).lower() not in asked and self.plan_open(j, prop):
                    opens.append(prop)
            wanted = list(self.wanted_asks)
            if narrow:
                wanted = [w for w in wanted if self.product_of(w) in (k, None)]
            parts = []
            if opens:
                parts += ["\n\nOPEN FROM THE LAST PAIR - second properties of planned pairs that the reply left "
                          "unanswered (only the first was answered, or only the first was sent); each is the plan's next "
                          "pick:\n", self._h_bullets(opens[-3:])]
            if wanted:
                parts += ["\n\nQUESTIONS THE ASSISTANT WANTED TO ASK (it has read the listing records) - a still-open "
                          "property such a question names that is a key below comes first, asked by the key's own name:\n",
                          self._h_bullets(wanted[-3:])]
            return "".join(parts)
        except Exception:  # noqa: BLE001
            return ""

    def want_ask(self, raw_args: Any) -> None:
        try:
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str) and raw_args.strip():
                args = first_json(raw_args) or {}
            else:
                args = {}
            text = " ".join(v.strip() for v in args.values() if isinstance(v, str) and v.strip())[:300]
            if text:
                kept = [w for w in self.wanted_asks if w.lower() != text.lower()] + [text]
                self.wanted_asks = kept[-3:]
        except Exception:  # noqa: BLE001
            pass

    def drop_wanted(self, label: Any) -> None:
        try:
            def cut_s(w: str) -> str:
                return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            props = [p.split("(")[0] for p in re.split(r";|\band\b", str(label).replace("[not answered]", ""))]
            names = [name_words(p) for p in props]
            kept = []
            for wanted in self.wanted_asks:
                stems = {cut_s(w) for w in re.findall(r"[a-z]+", wanted.lower())}       # letters only
                if any(words and all(w in stems or cut_s(w) in stems for w in words) for words in names):
                    continue
                kept.append(wanted)
            self.wanted_asks = kept
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _h_unplural(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w

    @staticmethod
    def _h_name_set(name: Any) -> set:
        return {w for w in name_words(name) if len(w) >= 3 and w not in FILLER}

    @staticmethod
    def _h_tokens(text: Any) -> list:
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    def _h_product_names(self) -> list:
        items = self.needs.items()
        if items:
            return [it["product"] for it in items]
        return [str(self.needs.reading.get("product") or "")]

    def _h_usable_keys(self, menu: Any, prop: Any) -> Any:
        """(key, line) of each menu line whose key is a property other than prop."""
        for line in str(menu or "").splitlines():
            key = self._h_menu_key(line)
            if not key or key.endswith(":") or strictly_same(key, prop) or not_a_property(key):
                continue
            yield key, line

    @staticmethod
    def _h_sample_said(sample: str, own: set, flat: str) -> bool:
        """An example value (of 3+ letters, not only the question's own words) the question text writes out."""
        st = CartworkMixin._h_tokens(sample)
        if not st or len("".join(st)) < 3:
            return False
        if all(t in own or t in FILLER for t in st):
            return False
        return " %s " % " ".join(st) in flat

    def names_other(self, question: Any, prop: Any, menu: Any, second: Any = "") -> bool:
        try:
            toks = self._h_tokens(question)
            flat = " %s " % " ".join(toks)
            said = set(toks) | {self._h_unplural(t) for t in toks}
            own = set(name_words(prop))
            own.update(("which", "would", "you", "like"))
            for product in self._h_product_names():
                for w in self._h_tokens(product):
                    own.add(w)
                    if len(w) > 3 and w.endswith("s"):
                        own.add(w[:-1])

            def named(name: Any) -> bool:
                words = self._h_name_set(name)
                return bool(words) and words <= said and not words <= own
            other = str(second or "").strip()
            if other and not strictly_same(other, prop) and named(other):
                return True
            for key, line in self._h_usable_keys(menu, prop):
                if named(key):
                    return True
                _, cut, tail = line.partition("):")
                if cut and any(self._h_sample_said(sample, own, flat) for sample in tail.split("|")):
                    return True
            return False
        except Exception:  # noqa: BLE001
            return False

    def pair_fits(self, first: Any, second: Any) -> bool:
        try:
            k = None
            if self.needs.items():
                k = self.product_of(first)
                if k is None or k != self.product_of(second):
                    return False
            if strictly_same(first, second) or loosely_same(first, second) or self.same_field(k, first, second):
                return False
            if not self._h_name_set(first).isdisjoint(self._h_name_set(second)):
                return False
            if self.field_values(k, first) & self.field_values(k, second):
                return False
            cached = self.menu_memo.get(k)
            menu = ""
            if isinstance(cached, tuple) and len(cached) > 1 and isinstance(cached[1], str):
                menu = cached[1]
            for key in map(self._h_key_of, menu.splitlines()):
                if not key or strictly_same(key, first) or strictly_same(key, second):
                    continue
                if self.twin_of(k, key, second) and self.plan_open(k, key):
                    return False
            return True
        except Exception:  # noqa: BLE001
            return False

    def queue_second(self, prop: Any) -> None:
        if prop and str(prop).lower() not in [str(p).lower() for p in self.pending_seconds]:
            self.pending_seconds.append(prop)

    def _h_log_tally(self, n: int) -> tuple:
        """Per product: (questions sent, questions answered), from the questions log."""
        sent, answered = [0] * n, [0] * n
        for row in self.questions_log:
            head = str(row).partition(":")[0]
            k = self.product_of(head)
            if k is None:
                quoted = self._h_asked_text(row)
                if quoted:
                    k = self.product_of(quoted)
            if k is None:
                continue
            sent[k] += 1
            if "[not answered]" not in head:
                answered[k] += 1
        return sent, answered

    def _h_hinted(self, n: int, answered: list) -> set:
        """Products a story hint points at: a hint naming one of its keys, or more key-less hints than answers."""
        try:
            keyless = [0] * n
            named: set = set()
            for hint in self.open_hints():
                head = str(hint).partition(":")[0].strip()
                j = self.product_of(head)
                if j is None:
                    continue
                if not self.plain_hint(j, head, hinted=True):
                    named.add(j)
                else:
                    keyless[j] += 1
            named.update(j for j in range(n) if answered[j] < keyless[j])
            return named
        except Exception:  # noqa: BLE001
            return set()

    def _h_try_pick(self, k: int, fallback: bool, ok: Any) -> dict | None:
        try:
            pick = self.quota_pick(k, fallback=fallback, ok=ok)
        except Exception:  # noqa: BLE001
            return None
        if not pick:
            return None
        if ok is not None:
            label = pick["property"]
            if not ok(label) or self.asked_for(k, label):
                return None
        return pick

    def _h_ask_plain(self, label: Any) -> dict:
        return dict(property=label, question=self._h_plain(label))

    def quota_first(self, scripted: list, ok: Any = None) -> list:
        items = self.needs.items()
        n = len(items)
        if n == 0:
            return scripted
        sent, answered = self._h_log_tally(n)
        if scripted and max(answered) == 0 and self.plan_product(scripted[0]) is None:
            return scripted
        owners = [self.product_of(str(line).partition(":")[0]) for line in self.needs.uncovered]
        found = [owners.count(j) for j in range(n)]
        hinted = self._h_hinted(n, answered)
        alt_left = set()
        for k in range(n):
            if answered[k] < sent[k] and self.alt_untried(k):
                alt_left.add(k)
        bare = [j for j in range(n) if answered[j] == 0]

        def level(k: int) -> tuple:
            first = int(bool(found[k]) and any(j != k for j in bare))
            behind = 0 if k in alt_left else sent[k] - answered[k]
            return first, int(k not in hinted), answered[k] + found[k], behind, found[k]
        levels = [level(k) for k in range(n)]
        for lv in sorted(set(levels)):
            due = [k for k in range(n) if levels[k] == lv]
            at = next((i for i, q in enumerate(scripted) if self.plan_product(q) in due), None)
            if at is not None:
                rest = list(scripted)
                front = rest.pop(at)
                return [front] + rest
            for q in scripted:
                second = self._h_text_of(q, "second").strip()
                if not second:
                    continue
                j = self.product_of(second)
                if j in due:
                    ask = {"property": second, "question": self._h_plain(second, items[j]["product"])}
                    return [ask, *scripted]
            ordered = [k for k in due if k in hinted] + [k for k in due if k not in hinted]
            for k in ordered:
                pick = self._h_try_pick(k, answered[k] == 0, ok)
                if pick:
                    return [pick, *scripted]
                if self.quota_wait:
                    return scripted
        return scripted

    @staticmethod
    def _h_key_of(line: str) -> str:
        return line.strip("- ").split(" (in ")[0].strip()

    def _h_quota_open(self, k: int, home: set, product: str) -> None:
        """No listing of the product was read yet: queue a few of its rows to open before asking about it."""
        batch = self.batch_tool()
        counts = self.needs.counts()
        qty = counts[k] if 0 <= k < len(counts) else 1
        unread = []
        if home:
            unread = self.menu_rows(home, min(4, batch[2]) if batch else 2, [], qty)
        if not unread or not (self.role("details") or batch):
            return
        if self.quota_reads.get(k) or self.moves_left() <= 12:
            return
        self.quota_reads[k] = True
        self.open_purpose = "quota"
        self.open_queue = unread
        self.open_recheck = True
        self.open_say = "Before asking about the %s, I'm opening a few of its listings to learn their fields." % product
        self.quota_wait = True

    def quota_pick(self, k: int, fallback: bool = True, ok: Any = None) -> dict | None:
        product = self._h_product_name(k)
        home = self._h_at(self.home_shelves(), k, set())
        shelf_book = self.archive
        shelf = next((label for label in shelf_book.shelves_read() if label.strip().lower() in home), None)
        if shelf is None and home and self.quota_reads.get(k):
            shelf = next((label for label in sorted(home) if shelf_book.prop_menu(label)[0]), None)
        if shelf is None:
            self._h_quota_open(k, home, product)
            return None
        menu = self.menu_trim(shelf_book.prop_menu(shelf)[0], shelf_book.rich_reads(shelf))

        def good(lab: str) -> bool:
            if ok is None:
                return True
            return ok(lab) and not self.asked_for(k, lab)
        pick = self.product_plan(k, menu)
        if pick and good(pick["property"]):
            return pick
        for hint in self._h_hints(k, None):
            prop = str(hint).partition(":")[0].split("(")[0].strip()
            if not prop or not_a_property(prop) or self.plain_hint(k, prop, hinted=True):
                continue
            lab = "%s (%s)" % (prop, product)
            if good(lab):
                return self._h_ask_plain(lab)
        if not fallback:
            return None
        lines = menu.splitlines()
        for by_samples in (False, True):
            for line in lines:
                prop = self._h_key_of(line)
                if not prop or not self.open_prop(k, prop, line, by_samples):
                    continue
                lab = "%s (%s)" % (prop, product)
                if good(lab):
                    return self._h_ask_plain(lab)
        return None

    def product_plan(self, k: Any, menu: str) -> dict | None:
        try:
            return self.product_plan_call(k, menu)
        except Exception:  # noqa: BLE001
            return None

    def _h_prompt_head(self, count: int, extra: str = "") -> list:
        """The planner prompt's opening pieces: its rules, the request, what is required, and what was asked."""
        pieces = [PLAN_TEXT.replace("QUESTION_COUNT", str(count)), extra, "\n\nTHE SHOPPER'S REQUEST:\n", self.request,
                  "\n\nWHAT IS REQUIRED SO FAR:\n", self.needs.text_body()]
        if self.questions_log:
            pieces += ["\n\nASKED SO FAR, WITH WHAT THE SHOPPER SAID:\n", "\n".join(self.questions_log)]
        return pieces

    def _h_plan_answer(self, k: Any, name: str, answer: Any) -> dict | None:
        raw = answer.get("questions") if isinstance(answer, dict) else None
        if not isinstance(raw, list):
            return None
        words = kind_set(name)

        def usable(label: str) -> str:
            return label if label and not self.plain_hint(k, label) else ""
        for q in raw:
            if not isinstance(q, dict):
                continue
            first = self._h_bare(q.get("property"))
            second = self._h_second(self._h_bare(q.get("second")))
            question = self._h_text_of(q, "question").strip()
            first = usable(first)
            second = usable(second)
            if first and self.plan_open(k, first):
                label = "%s (%s)" % (first, name)
                named = bool(question) and not kind_set(question).isdisjoint(words)
                paired = ""
                if named and second and not strictly_same(second, first) and self.plan_open(k, second):
                    paired = "%s (%s)" % (second, name)
                return dict(property=label, second=paired, question=question if named else self._h_plain(label))
            if second and self.plan_open(k, second):
                return self._h_ask_plain("%s (%s)" % (second, name))
        return None

    def product_plan_call(self, k: Any, menu: str) -> dict | None:
        if not menu or self.product_plans >= 1 or self.wire.left() < Knobs.WIND_DOWN_S + 40:
            return None
        self.product_plans += 1
        remaining = max(1, self.ask_limit() - self.msgs_sent)
        name = self._h_product_name(k)
        pieces = self._h_prompt_head(min(3, remaining))
        pieces.append("\n\nTHIS TIME EVERY QUESTION IS ABOUT THE %s: every other product has had a question already and "
                      "this one has had none, so choose only this product's properties, from its own list below - the "
                      "one the story points at first." % name.upper())
        hints = self._h_hints(k, 6)
        if hints:
            marked = ["- " + self.hint_mark(h) for h in hints]
            pieces += ["\n\nSTORY HINTS (details of the shopper's story with the property each seems to concern, "
                       "likeliest first - they are not requirements):\n", "\n".join(marked)]
        pieces += [self.plan_notes(k), "\n\nPROPERTY KEYS OF THIS PRODUCT'S LISTINGS (key, how many listings carry it, "
                   "example values):\n", menu]
        answer = self.wire.ask_json("".join(pieces), 1500, LINEUP["planner"][:1])
        self._h_note("product_plan", product=k, answer=answer)
        return self._h_plan_answer(k, name, answer)

    def _h_hints(self, k: Any, most: Any) -> list:
        try:
            return self.open_hints(k)[:most]
        except Exception:  # noqa: BLE001
            return []

    @staticmethod
    def _h_flat(t: Any) -> str:
        words = re.findall(r"[a-z0-9]+", str(t or "").lower())
        return " ".join(words)

    @staticmethod
    def _h_stem(label: str) -> str:
        return label.split("(")[0].strip().lower()

    def _h_add_open(self, p: Any) -> None:
        if not p:
            return
        if str(p).lower() in {str(x).lower() for x in self.open_pair}:
            return
        self.open_pair.append(p)

    def _h_to_front(self, p: str) -> None:
        low = p.lower()
        rest = [x for x in self.pending_seconds if str(x).lower() != low]
        self.pending_seconds = [p] + rest

    def _h_plan_prompt(self, remaining: int, ask_first: Any, items: list, prop_list: str, sets: int) -> tuple:
        """The planner's prompt (built even when a queued re-ask goes out instead), and the property list a drafted
        question is checked against."""
        shown = self.menu_trim(prop_list, sets)
        products_line = ""
        if items:
            numbered = []
            for i, it in enumerate(items, 1):
                numbered.append("%d) %s x%d" % (i, it["product"], it["quantity"]))
            products_line = "THE PRODUCTS: " + " | ".join(numbered)
            shelf_book = self.archive
            read = shelf_book.shelves_read()
            homes = self.home_shelves()
            union = set().union(*homes)
            mine = [label for label in read if label.strip().lower() in union]
            if not (homes and all(homes)):
                labels = mine + [label for label in read if label not in mine]
            elif mine:
                labels = mine
            else:
                labels = read
            blocks, trimmed = [], []
            for label in labels[:4]:
                menu = shelf_book.prop_menu(label)[0]
                if not menu:
                    continue
                head = "CATEGORY %s:\n" % label
                blocks.append(head + menu)
                trimmed.append(head + self.menu_trim(menu, shelf_book.rich_reads(label)))
            if blocks:
                prop_list, shown = "\n".join(blocks), "\n".join(trimmed)
        pieces = self._h_prompt_head(remaining, ONE_AT_A_TIME.replace("REMAINING", str(remaining)))
        if ask_first:
            pieces += ["\n\nMUST BE LEARNED FROM THE SHOPPER: ", "; ".join(str(a) for a in ask_first)]
        pieces.append(self.plan_notes(None))
        hints = self._h_hints(None, 8)
        if hints:
            marked = ["- " + self.hint_mark(h) for h in hints]
            pieces += ["\n\nSTORY HINTS (details of the shopper's story and the property each seems to concern - they "
                       "are not requirements, but the hidden need usually sits on one of them):\n", "\n".join(marked)]
        if products_line:
            pieces += ["\n\n", products_line]
        if not shown.strip():
            pieces.append("\n\nPROPERTY KEYS ON THE LISTINGS READ SO FAR: none seen yet.")
        else:
            pieces += ["\n\nPROPERTY KEYS ON THE LISTINGS READ SO FAR (key, how many listings carry it, example "
                       "values):\n", shown]
        return "".join(pieces), prop_list

    @staticmethod
    def _h_text_of(q: dict, key: str) -> str:
        return str(q.get(key) or "")

    def _h_draft(self, q: dict, several: bool, plain_only: bool, prop_list: str, cid: str, remaining: int) -> tuple | None:
        """(message call, (label, question text), second property left for later) for one planned question, or None
        when no message call can carry it."""
        prop = self._h_text_of(q, "property")
        text = self._h_text_of(q, "question").strip()
        if several:
            low_text = text.lower()
            tags = re.findall(r"[a-z]{3,}", self._h_tag(prop).lower())
            if tags and all(w.rstrip("s") not in low_text for w in tags):
                text = self._h_plain(prop)
        if not plain_only and self.names_other(text, prop, prop_list, self._h_text_of(q, "second").strip()):
            text = self._h_plain(prop)
        call = self.message_call(text, cid)
        if call is None:
            return None
        second = self._h_second(q.get("second"))
        if not second or second.lower() == prop.lower():
            return call, (prop, text), ""
        if remaining == 1 and not self.no_pairs and self.pair_fits(prop, second) and \
                self.hint_mapped(self.product_of(prop) if several else None, prop):
            text = "Which %s would you like for the %s, and which %s?" % (
                self._h_bare(prop), self._h_product_word(prop), self._h_bare(second))
            drafted, spare = ("%s; %s" % (prop, second), text), ""
        else:
            text = self._h_plain(prop)
            drafted, spare = (prop, text), second
        call = self.message_call(text, cid)
        return None if call is None else (call, drafted, spare)

    def _h_unanswered(self, asked_label: str, asked_text: str, split_second: str) -> tuple:
        """The reply asked for one question at a time: what is asked next, as (note, log label)."""
        self.no_pairs = True
        parts = []
        for piece in asked_label.split(";"):
            piece = piece.strip()
            if piece:
                parts.append(piece)
        parts = parts or [asked_label]
        first = parts[0]
        key = first.lower()
        single = len(parts) < 2
        plain_q = self._h_plain(first)
        retry = key not in self.reasked
        alt = self.collide_alt(first) if single else ""
        if alt:
            self.reasked.add(key)
            self.spent_asks.add(key)
            self._h_to_front(alt)
            note = " [%s: not answered; the story hint's alternative %s goes next]" % (first, alt)
        elif (retry and self._h_flat(asked_text) != self._h_flat(plain_q)
              and not (single and self.holds_plain(asked_text, plain_q))):
            self.reasked.add(key)
            self._h_to_front(first)
            note = " [%s: not answered; asked again on its own next]" % first
        else:
            redo = self.distinct_ask(first) if retry else None
            if not redo:
                self.spent_asks.add(key)
                note = " [%s: not answered; not asked again]" % first
            else:
                redo_label, redo_text = redo
                redo_key = redo_label.lower()
                self.reasked.update((key, redo_key))
                self.pending_text[redo_key] = redo_text
                self._h_to_front(redo_label)
                if self._h_stem(redo_label) == self._h_stem(first):
                    note = " [%s: not answered; asked again by its values next]" % first
                else:
                    self.spent_asks.add(key)
                    note = " [%s: not answered; asked again as %s next]" % (first, redo_label)
        leftover = parts[1:]
        if split_second:
            leftover.append(split_second)
        for p in leftover:
            self._h_add_open(p)
        return note, " and ".join(parts) + " [not answered] [one at a time]"

    def pose_planned(self, urge: bool = False) -> bool:
        sent, ceiling = self.msgs_sent, self.ask_limit()
        if sent >= ceiling or not self.role("message"):
            self.asking = False
            return False
        wire, needs, shelf_book = self.wire, self.needs, self.archive
        shelf = self.own_shelf()
        prop_list, sets = shelf_book.prop_menu(shelf)[0], shelf_book.rich_reads(shelf)
        ask_first = needs.reading.get("ask_first")
        pressing = not self.questions_log and bool(ask_first)
        if not (urge or pressing) and sets < Knobs.ASK_AFTER_READS:
            tick = wire.tick
            if tick < Knobs.ASK_AFTER_STEPS:
                return False
            if tick < Knobs.ASK_AFTER_STEPS + Knobs.ASK_WAIT_STEPS and self.moves_left() > 20:
                return False
        late = self.moves_left() <= max(10, self.order_due())
        if late or wire.left() < Knobs.WIND_DOWN_S + 40:
            self.asking = False
            return False
        if not urge and self.plan_wait is not None and self.plan_wait >= sets:
            return False
        items = needs.items()
        several = bool(items)
        if not (several or urge or pressing) and self.asked_at_sets is not None and self.asked_at_sets >= sets:
            room = self.moves_left() - self.order_due() - 2 * max(0, ceiling - sent)
            if room > 14 and wire.tick - (self.asked_at_tick or 0) < 4:
                return False
        if self.pre_read():
            return True
        if not (urge or pressing or sent or self.questions_log) and self.first_reads():
            return True
        remaining = ceiling - sent
        prompt, prop_list = self._h_plan_prompt(remaining, ask_first, items, prop_list, sets)
        asked_props = self._h_asked_parts()

        def fresh(prop: Any) -> bool:
            name = str(prop or "").strip()
            low = name.lower()
            if not name or low in asked_props or low in self.spent_asks or not_a_property(name):
                return False
            k = self.product_of(name) if several else None
            if self.plain_hint(k, name):
                return False
            return self.plan_open(k, name)
        still = list(filter(fresh, self.pending_seconds))
        self.pending_seconds = still
        pending = still[0] if still else ""
        if not pending:
            answer = wire.ask_json(prompt, 2500, LINEUP["planner"])
            if answer is None and wire.misses > 0:
                self.plan_wait = sets
                return False
            listed = answer.get("questions") if isinstance(answer, dict) else None
        else:
            text = self.pending_text.pop(str(pending).lower(), None) or self._h_plain(pending)
            listed = [{"property": pending, "question": text}]
        offered = []
        for q in listed if isinstance(listed, list) else []:
            asked = q.get("question") if isinstance(q, dict) else None
            if str(asked or "").strip():
                offered.append(q)
        kept = []
        for q in offered:
            second = self._h_second(q.get("second"))
            prop = self._h_text_of(q, "property").strip()
            if not fresh(prop):
                if second and fresh(second):
                    kept.append({"property": second, "question": self._h_plain(second)})
                continue
            item = dict(q)
            item["second"] = second if fresh(second) else ""
            if second:
                owner = self.product_of(second) if several else None
                if self.plain_hint(owner, second):
                    item["question"] = self._h_plain(prop)
            kept.append(item)
        lined_up = kept
        if several and not pending:
            lined_up = self.quota_first(kept, fresh)
            if self.quota_wait:
                self.quota_wait = False
                return True
        cid = "ask_%d_%d" % (sent + 1, wire.tick)
        drafted = None
        for q in lined_up[:2]:
            drafted = self._h_draft(q, several, bool(pending) or pressing, prop_list, cid, remaining)
            if drafted is not None:
                break
        if drafted is None:
            roomy = self.moves_left() > 12
            if roomy and not urge and sets < Knobs.ASK_AFTER_READS:
                self.plan_wait = sets
                return False
            self.empty_plans += 1
            if roomy and self.empty_plans < 2:
                self.plan_wait = sets
            else:
                self.asking = False
            return False
        proposal, (asked_label, asked_text), split_second = drafted
        self.asked_at_sets, self.asked_at_tick = sets, wire.tick
        self.chat.append({"role": "assistant", "content": "I'll ask the shopper about %s." % asked_label,
                          "tool_calls": [proposal]})
        found_before = len(needs.uncovered)
        freed_before = len(needs.indifferent) + needs.pinned_heard
        self.asking_label = asked_label
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.asking_label = ""
        self.post_replies(responses)
        if pending and asked_label.split(";")[0].strip().lower() == pending.lower():
            self.pending_seconds = [x for x in self.pending_seconds if x.lower() != pending.lower()]
        heard_meta, needs.meta_last = needs.meta_last, None
        uncovered = needs.uncovered
        revealed = len(uncovered) > found_before
        freed_now = len(needs.indifferent) + needs.pinned_heard
        found_text = " ".join(str(x) for x in uncovered[found_before:]).lower()

        def name_absent(prop: str) -> bool:
            stems = re.findall(r"[a-z]{4,}", prop.split("(")[0].lower())
            return all(w not in found_text for w in stems)
        flat_text = self._h_flat(asked_text)
        notes, prop_label = "", asked_label
        if heard_meta and flat_text and flat_text in self._h_flat(heard_meta):
            notes, prop_label = self._h_unanswered(asked_label, asked_text, split_second)
        elif ";" in asked_label:
            head, _, second = asked_label.partition(";")
            second = second.strip()
            if revealed:
                if name_absent(second):
                    self.queue_second(second)
                    notes = " [%s: not answered; still open]" % second
                prop_label = head.strip()
            elif freed_now > freed_before:
                prop_label = head.strip()
                drop = {second.lower(), second.split("(")[0].strip().lower()}
                needs.indifferent = [p for p in needs.indifferent if str(p).strip().lower() not in drop]
                self._h_add_open(second)
                notes = " [%s: not answered; still open]" % second
            elif freed_now == freed_before:
                self.no_pairs = True
                for p in asked_label.split(";"):
                    self._h_add_open(p.strip())
                prop_label = asked_label.replace(";", " and") + " [not answered]"
            elif split_second and name_absent(split_second):
                self._h_add_open(split_second)
        elif split_second and name_absent(split_second):
            self._h_add_open(split_second)
        if not revealed and heard_meta is None:
            self.quiet_asks.append(asked_label.partition(";")[0].strip())
            twin = "" if ";" in asked_label else self.quiet_reask(asked_label)
            if twin:
                self._h_to_front(twin)
                notes += " [%s: its story twin, asked next]" % twin
        if responses:
            said = str(responses[0][1]).partition(H_STATUS_CUT)[0]
            self.questions_log.append('- %s: asked "%s" -> %s%s' % (prop_label, asked_text, said[:400], notes))
        self.drop_wanted(asked_label)
        self.asking = self.msgs_sent < ceiling
        return True

    def reply_back(self, text_body: str) -> bool:
        limit = self.ask_limit()
        sent = self.msgs_sent
        if (not self.role("message") or sent >= limit or self.moves_left() <= 4 or self.wire.left() < Knobs.WIND_DOWN_S
                or (self.asking and sent >= limit - 1 and self.moves_left() > 10
                    and self.wire.left() >= Knobs.WIND_DOWN_S + 40)):
            return False
        proposal = self.message_call(text_body, "reply_%d" % self.wire.tick)
        if proposal is None:
            return False
        self.chat.append({"role": "assistant", "content": "I'll answer the shopper's question.", "tool_calls": [proposal]})
        self.post_replies(self.dispatch([proposal], scripted=True))
        return True

    def post_replies(self, responses: list) -> None:
        last = len(responses) - 1
        for i, (cid, text) in enumerate(responses):
            body = "%s\n%s" % (text, self.status_line()) if i == last else text
            self.chat.append({"role": "tool", "tool_call_id": cid, "content": body})

    # ================================================================================== 11. code-only request reading
    def code_reading(self) -> dict:
        request = self.request
        low = request.lower()
        reading: dict = {"code_reading": True, "requirements": [], "items": [], "quantity": 1}
        for key in ("rules", "soft_wants", "ask_first", "hints", "references"):
            reading[key] = []
        reading["priority_detail"] = ""
        cap, unit = self.code_budget(request)
        reading.update(price_cap=cap, currency=unit, in_stock_only=bool(H_STOCK_TALK.search(low)),
                       priority="cheapest" if H_CHEAPEST_TALK.search(low) else "none")
        messages, moves = (self.code_limit(low, nouns) for nouns in (H_MESSAGE_NOUNS, H_STEP_NOUNS))
        if messages is not None:
            reading["message_budget"] = messages
        if moves is not None and Knobs.LATE_STEPS + 3 <= moves:
            reading["step_budget"] = moves
        reading["roles"] = roles = self.code_roles()
        search = roles.get("search")
        found = self.code_items(search)
        if len(found) > 1:
            listed, reqs = [], []
            for it in found:
                listed.append({"product": it["product"], "quantity": it["quantity"],
                               "requirements": list(it["requirements"])})
                reqs.extend(it["requirements"])
            reading["items"] = listed
            reading["product"] = " and ".join(it["product"] for it in found)
            reading["requirements"] = reqs
            return reading
        if not found:
            reading["product"] = self.code_label(search, kind_set(request))
            reading["quantity"] = self.code_count(low)
        else:
            reading["product"], reading["quantity"] = found[0]["label"], found[0]["quantity"]
        reading["requirements"] = ["%s: %s" % (p, v) for p, v in self.code_needs(low)]
        return reading

    @staticmethod
    def code_budget(text_body: Any) -> tuple:
        best = (0, 0.0)
        found: tuple = (None, None)
        for sentence in re.split(r"(?<=[.!?;])\s+", str(text_body or "").lower()):
            whole = 1 if H_WHOLE_ORDER.search(sentence) else 0
            for m in H_FIGURE.finditer(sentence):
                sign, digits = m.group(1), m.group(2)
                after = sentence[m.end():]
                if H_COUNTED.match(after):
                    continue
                start = m.start()
                if not sign and H_RUN_BEFORE.search(sentence[:start]):
                    continue
                before = sentence[max(0, start - 60):start]
                score = whole
                if sign or H_SPEND_CUE.search(before):
                    score += 3
                if H_LIMIT_CUE.search(before):
                    score += 2
                if H_CEILING_AFTER.match(after):
                    score += 2
                figure = float(digits.replace(",", ""))
                if score < 2 or (score, figure) <= best:
                    continue
                best = (score, figure)
                words = []
                um = H_UNIT_WORDS.match(after)
                for w in (um.groups() if um else ()):
                    w = (w or "").strip(".")          # dots at the ends only ("u.s.d." -> "u.s.d")
                    if w and w not in FILLER and not H_UNIT_SKIP.fullmatch(w):
                        words.append(w)
                amount = int(figure) if figure.is_integer() else figure
                found = (amount, " ".join(words) or sign or None)
        return found
# ------------------------------------------------------------------------------------------------------- self-driving
I_COUNT_WORDS = dict(zip("one two three four five six seven eight nine ten twelve fifteen twenty".split(),
                         (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)))
I_SMALL_COUNTS = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"
I_LIMIT_BEFORE = re.compile(r"\b(?:at most|no more than|not more than|up to|only|max\w*|use|using|take|send|allow\w*|"
                            r"receive|get|total of|limit\w*|inside|within|under|fewer than|less than|you've|you have|"
                            r"have|keep\w*|stick to|go past|past|exceed\w*|beyond)\W+(?:[\w']+\W+){0,3}$")
I_LIMIT_AFTER = re.compile(r"\W*(?:[\w']+\W+){0,3}?(?:in total|total|no more|at most|max\w*|tops|or fewer|or less|"
                           r"allowed|limit)\b")
I_UNIT_COUNT = re.compile(r"\b(%s)\s+(?:units?|of them|of those|of these|copies)\b" % I_SMALL_COUNTS)
I_NEED_SHAPE = re.compile(
    r"(?:\b(?:where|whose)\s+(?:(?:the|its|their)\s+)?"
    r"|(?:(?:\band|\bwith|\bbut)\s+|[,;:]\s*|^|[.!?]\s+)(?:the|its|their)\s+)"
    r"(?P<prop>[a-z][a-z'&/ \-]{1,32}?)\s+"
    r"(?:is|are|has to be|have to be|must be|should be|needs to be|need to be|can be)\s+(?:(?:either|just)\s+)?"
    r"(?P<value>[^,.;:!?()]{1,60}?)"
    r"(?=\s*(?:[,.;:!?()]|$)|\s+(?:(?:so|since|as|because|for|to|which|that|nothing|otherwise|but|when|while|if|then|"
    r"though|please)|and (?:the|its|their|a|an|one|two|three|\d))\b)")
I_NEED_NOT_PROP = WordPattern("i|you|we|it|he|she|they|my|your|our|there|this|that|which|what|who|me|us|them|one|ones|price|budget|cost|total|order|stock")
I_NEED_NOT_VALUE = re.compile(r"(?:a|an|the|my|your|our|his|her|their|me|us|for|to|in|on|at|of|see|go|get|be|have|only|"
                              r"too|very|so|not|no|what|how|it|all|mostly|always|never|just|really|still|also|from|"
                              r"about|like|one|ones)\b")
I_ITEM_COUNT = re.compile(r"\b(%s)\s*(?:x\s+|×\s*)?(?:(?:of|[a-z'&\-]{3,})\s+){0,4}$" % I_SMALL_COUNTS)


def _i_rx(text: str) -> Any:
    return re.compile(text)


I_PERSON = _i_rx(r"\b(?:shopper|customer|buyer|client|user)s?\b")
I_SPEAK = _i_rx(r"\b(?:send|repl|tell|messag|chat|writ|talk)|\bnotes?\b")
I_SEEK = _i_rx(r"\b(?:search|find|look|locat|hunt|brows|quer)")
I_DETAIL = _i_rx(r"\b(?:details?|records?|facts?|full)\b|\b(?:spec|propert|attribut)")
I_VIEW = _i_rx(r"\b(?:offer|price|stock)\b|\bavailab")
I_PAGE_NAME = _i_rx(r"page|offset|first|start|skip|from|cursor")
I_NOT_PAGE_NAME = _i_rx(r"size|per_?page|limit|price|cost")
I_MARK_KEYS = _i_rx(r"next|onward|cursor|bookmark|token|continu|more")
I_ORDER_TEXT = "Please place the order for the items in my cart."
I_STRIP = "'\"()[]{},.;:!?|/"
I_INF = float("inf")
# word runs and small shapes, compiled once
I_ALNUM, I_ALNUM2, I_ALNUM3 = _i_rx(r"[a-z0-9]+"), _i_rx(r"[a-z0-9]{2,}"), _i_rx(r"[a-z0-9]{3,}")
I_ALPHA, I_ALPHA2, I_ALPHA3 = _i_rx(r"[a-z]+"), _i_rx(r"[a-z]{2,}"), _i_rx(r"[a-z]{3,}")
I_LINKED = _i_rx(r"\b(?:%s)\b" % LINK_PHRASES)
I_DIGIT, I_ANY_ALNUM = _i_rx(r"\d"), _i_rx(r"[a-z0-9]")
I_CAMEL = _i_rx(r"([a-z])([A-Z])")
I_OR_WORD = _i_rx(r"\bor\b")
I_MARKS = _i_rx(r"\(stated\)|\(revealed[^)]*\)")
I_BOUND_SIGNS = (I_LINKED, _i_rx(r"\d\s*(?:-|to)\s*\d"), _i_rx(r"[+<>]"),
                 _i_rx(r"\b(?:about|around|approx\w*|roughly)\b"))
I_DIGIT_RUNS = _i_rx(r"(\d+)")
I_PAGE_TAIL = _i_rx(r"\b(?:page|listing|shown|sold by)\b")
I_SENTENCE_END = _i_rx(r"[.!?]\s")
I_FINAL = _i_rx(r"\bfinal\b")
I_CART_WORD = _i_rx(r"\b(?:cart|basket|bag|trolley)\b")
I_CHATTY = _i_rx(r"\b(?:messag|chat)")
I_PRICE_KEY = _i_rx(r"price|cost|amount")
I_NOT_PRICE_KEY = _i_rx(r"count|qty|quant|total|units|line|due|payable|sum")
I_COST_KEY = _i_rx(r"price|cost")
I_ALT_SPLIT = _i_rx(r"\bor\b|[/,;]")
I_FIGURE = _i_rx(r"\d+(?:\.\d+)?")
I_OR_SLASH = _i_rx(r"\bor\b|/")
I_DESC_SPLIT = _i_rx(r"[.;]")
I_QUERY_NAME = _i_rx(r"query|text|keyword|needle|term|word|phrase|string|search")
I_CEIL_NAME = _i_rx(r"ceil|max|price|budget|upper|most")
I_PRICE_WORD = _i_rx(r"\b(?:pric|cost)")
I_NOT_CEIL = _i_rx(r"page|first|offset|skip|count|size|num")
I_TOTAL_KEY = _i_rx(r"count|total|matched|found|hits")
I_FAULT = _i_rx(FAULT_MARK)
I_ID_ARG = _i_rx(r"id|item|product|listing|sku|target|which|number")
I_SHELF_KEY = _i_rx(r"categor|shelf|department|aisle")
I_STOCK_KEY = _i_rx(r"stock|avail|on.?hand")
I_NO_STOCK = _i_rx(r"\b(?:out|sold|none|unavailable|not|no|false)\b")
I_ON_HAND = _i_rx(r"on.?hand")
I_GONE_WORD = _i_rx(r"back.?order|pre.?order|discontinu")
I_TITLE_KEY = _i_rx(r"title|heading|headline|name")
I_NOT_TITLE_KEY = _i_rx(r"categor|shelf|brand|make|manufactur|vendor|shop|seller|store|id")
I_MAKER_KEY = _i_rx(r"brand|make|manufactur|vendor")
I_TOKEN_NAME = _i_rx(r"bookmark|cursor|token|resume|continu|after|next|page|offset|start")
I_OFFSET_NAME = _i_rx(r"offset|skip|first|start|from")
I_LABEL_WORD = _i_rx(r"[a-z0-9][a-z0-9'\-]*")
I_SELLER_KEY = _i_rx(r"shop|seller|store|vendor|brand|make|manufactur")
I_NOTE_KEY = _i_rx(r"note|text|desc|comment|detail|info|message|section")
I_HEAD_KEY = _i_rx(r"title|heading|headline|name|label|caption")
I_COMMA_SEMI = _i_rx(r"[,;]")
I_NOT_COMBINE = _i_rx(r"\d|as shown|same as|page|redirect|refer|\bnot\b|\bno\b|without|except|"
                      r"anything|other than")
I_COMBINE_SPLIT = _i_rx(r"[,;/]|\bor\b|\band\b")
I_STOCK_WORD = _i_rx(r"stock|avail")
I_FIG_UNIT = _i_rx(r"\s*(\d+)\s+([a-z]+)")
I_LEAD_AMOUNT = _i_rx(r"\s*\d+(?:[.,]\d+)?\s+[a-z]")
I_ALT_CUT = _i_rx(r"[,;/]|\b(?:or|either|nor)\b")


def _i_count(word: str) -> int:
    return int(word) if word.isdigit() else I_COUNT_WORDS.get(word, 0)


def _i_marks_off(text: Any) -> str:
    bare = str(text or "")
    return I_MARKS.sub("", bare)


def _i_is_bound(value: Any) -> bool:
    low = str(value or "").lower()
    return any(sign.search(low) for sign in I_BOUND_SIGNS)


def _i_unique(seq: Any) -> list:
    kept: list = []
    for item in seq:
        if item in kept:
            continue
        kept.append(item)
    return kept


def _i_near(a: str, b: str) -> bool:
    if min(len(a), len(b)) < 3:
        return False
    return a.startswith(b) or b.startswith(a)


def _i_near_figure(a: str, b: str, lead_digits: bool) -> bool:
    if not (I_DIGIT.search(a) or I_DIGIT.search(b)):
        return _i_near(a, b)
    if a.lstrip("0") == b.lstrip("0"):
        return True
    size = len(a)
    if not lead_digits or not a.isdigit() or len(b) <= size:
        return False
    return b.startswith(a) and not b[size].isdigit()


def _i_natural(text: str) -> list:
    pieces = I_DIGIT_RUNS.split(text)
    return [int(bit) if bit.isdigit() else bit for bit in pieces]


def _i_long_texts(value: Any, out: list, depth: int = 0) -> None:
    if depth <= 6:
        if isinstance(value, str):
            if len(value.split()) >= 4:
                out.append(value)
            return
        if isinstance(value, dict):
            inner = list(value.values())
        elif isinstance(value, list):
            inner = value[:20]
        else:
            return
        deeper = depth + 1
        for child in inner:
            _i_long_texts(child, out, deeper)


def _i_flat_text(text: str) -> str:
    words = str(text).lower().split()
    return " ".join(words)


def _i_key(args: Any) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _i_plain_str(spec: Any) -> bool:
    """A free-text property: a string with no enum."""
    if not isinstance(spec, dict):
        return False
    return spec.get("type") == "string" and not spec.get("enum")


def _i_count_spec(spec: Any) -> bool:
    """A free number property: integer or number, no enum."""
    if not isinstance(spec, dict):
        return False
    return spec.get("type") in ("integer", "number") and not spec.get("enum")


def _i_low(value):
    return str(value).lower()


def _i_terse(plain):
    low = plain.lower()
    if len(I_ALNUM.findall(low)) > 5:
        return False
    return not EXCLUSION.search(low)


def _i_blank_row():
    return dict(shelf="", title="", price=None, stock=None, words=set(), fields={})


def _i_fold_row(row, d, shelf):
    """Fold one record dict into the row kept for its id."""
    if shelf:
        row["shelf"] = shelf
    title = next((v for key, v in d.items() if isinstance(v, str) and _i_title_key(_i_low(key))), None)
    if title:
        row["title"] = title
    for key, v in d.items():
        lk = _i_low(key)
        if numeric(v) and I_PRICE_KEY.search(lk) and not I_NOT_PRICE_KEY.search(lk):
            row["price"] = float(v)
            break
    for key, v in d.items():
        state = SelfDriveMixin.key_stock(key, v)
        if state is not None:
            row["stock"] = state
    scalars = []
    for v in d.values():
        if isinstance(v, str) or numeric(v):
            scalars.append(str(v))
        elif isinstance(v, dict):
            for fk, fv in v.items():
                if not isinstance(fv, (str, int, float)):
                    continue
                scalars.append(str(fv))      # (a nested yes/no counts as a word too)
                if isinstance(fv, bool):
                    continue
                row["fields"][str(fk)] = str(fv)
    row["words"] |= kind_set(" ".join(scalars))


def _i_title_key(lk):
    return ("title" in lk or "name" in lk) and "id" not in lk


def _i_alt_seen(alt, hay):
    words = [w for w in I_ALPHA2.findall(alt) if w not in FILLER]
    figures = I_FIGURE.findall(alt)
    if not words and not figures:
        return False
    for w in words:
        root = w[:-1] if len(w) > 4 else w
        if re.search(r"\b" + re.escape(root), hay) is None:
            return False
    for fig in figures:
        if re.search(r"(?<![\d.])" + re.escape(fig) + r"(?![\d.])", hay) is None:
            return False
    return True


def _i_named_texts(fields, prop):
    return [text for key, text in fields.items() if loosely_same(key, prop)]


def _i_spoken(text):
    return " ".join(value_words(text))


def _i_apart_ok(fields, apart):
    for aprop, barred in apart:
        held = False
        for fk, fv in fields.items():
            if not loosely_same(fk, aprop) or not str(fv).strip():
                continue
            if any(field_writes(x, fv) for x in barred):
                continue
            held = True
            break
        if not held:
            return False
    return True


def _i_settle(pools, which, fails):
    picks, fit_of, total = [], {}, 0.0
    for units, pool in pools:
        ordered = sorted(pool, key=lambda e: e[which])
        chosen = next((e for e in ordered if e[2] not in fit_of and not fails(e[2])), None)
        if chosen is None:
            return None
        _, _, ident, price, fit = chosen
        picks.append((ident, units))
        fit_of[ident] = fit
        total += (0 if price == I_INF else price) * units
    return picks, fit_of, total


def _i_phrase(words):
    return " ".join(_i_unique(words))


def _i_short_enum(spec):
    options = spec.get("enum") if isinstance(spec, dict) else None
    return isinstance(options, list) and 0 < len(options) <= 4


def _i_text_key(strings):
    for name in strings:
        lowered = name.lower()
        if lowered == "q" or I_QUERY_NAME.search(lowered):
            return name
    return strings[0] if strings else None


def _i_ceiling_key(props, parts):
    for name, spec in props.items():
        if not _i_count_spec(spec):
            continue
        ln = name.lower()
        looks = I_CEIL_NAME.search(ln) or ln.endswith("cap") or any(ln in p and I_PRICE_WORD.search(p) for p in parts)
        if looks and not I_NOT_CEIL.search(ln):
            return name
    return None


def _i_trimmed(text, most):
    while " " in text and len(text) > most:
        text = text[:text.rindex(" ")]
    return text[:int(most)]


def _i_default_for(spec, text):
    """(known, value) a required argument the code leaves open takes."""
    if not isinstance(spec, dict):
        return False, None
    if spec.get("enum"):
        return True, spec["enum"][0]
    kind = spec.get("type")
    if kind in ("integer", "number"):
        return True, spec.get("default", spec.get("minimum", 1))
    if kind == "string":
        return True, text
    return False, None


def _i_fix_call(call, schema, bound):
    fixed, faults = ArgFixer.fix(call, schema)
    if faults and bound:
        loose = {a: b for a, b in call.items() if a not in bound}
        fixed, faults = ArgFixer.fix(loose, schema)
    return fixed, faults


def _i_by_rank(lists):
    merged: list = []
    rank = 0
    while True:
        tier = [x[rank] for x in lists if len(x) > rank]
        if not tier:
            return merged
        merged.extend([c for c in tier if c not in merged])
        rank += 1


def _i_first_table(obs):
    """The first dict (3 levels down at most) holding a non-empty list of dicts."""
    found = []

    def visit(d):
        if found:
            return
        for v in d.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                found.append(d)
                return
    each_dict(obs, visit, 3)
    return found[0] if found else None


def _i_numeric_next(cur, by, low, high, rows, marked, total):
    if not numeric(cur):
        cur = low
    nxt = cur + (1 if by else rows)
    if high is not None and nxt > high:
        return None
    if not marked:
        shown = (cur - low + 1) * rows if by else nxt
        if total is None or shown >= total:
            return None
    if isinstance(nxt, float) and nxt.is_integer():
        nxt = int(nxt)
    return nxt


def _i_enum_words(v):
    spaced = I_CAMEL.sub(r"\1 \2", str(v))
    return set(I_ALPHA.findall(spaced.lower()))


def _i_put_take(name, options, known):
    """The add and the remove value of one required enum of the cart tool."""
    put_like, take_like = [], []
    for v in options:
        said = _i_enum_words(v)
        if said & REMOVE_SENSE:
            take_like.append(v)
        elif said & ADD_SENSE:
            put_like.append(v)
    if name in known and known[name] in options:
        put = known[name]
    elif put_like:
        put = put_like[0]
    else:
        put = options[0]
    take = next((v for v in take_like if v != put), None)
    if take is None and len(options) == 2:
        take = options[1] if options[0] == put else options[0]
    return put, take


def _i_line_call(puts, id_arg, ident, units, n):
    call = {**puts, id_arg: ident}
    if units:
        call[units[0]] = n
    return call


def _i_all_fixed(drafts, schema):
    """Every draft through the schema check, or None when one of them has a fault."""
    out = []
    for call in drafts:
        fixed, faults = ArgFixer.fix(call, schema)
        if faults:
            return None
        out.append(fixed)
    return out


def _i_id_like(v):
    if isinstance(v, str):
        return True
    return isinstance(v, int) and not isinstance(v, bool)


def _i_listed_id(d, pool):
    """The first string value under an id-like key that the pool holds."""
    for k, v in d.items():
        if "id" in _i_low(k) and isinstance(v, str) and v in pool:
            return v
    return None


def _i_tops(tally, names):
    high = max(tally[name] for name in names)
    return {name for name in names if tally[name] == high}


def _i_first_cost(d):
    for k, v in d.items():
        if numeric(v) and I_COST_KEY.search(_i_low(k)):
            return v
    return None


def _i_shelf_title(d):
    title = d.get("title")
    if isinstance(title, str) and 1 <= len(title) <= 300:
        return title
    for k, v in d.items():
        if not isinstance(v, str) or not 1 <= len(v) <= 300:
            continue
        lk = _i_low(k)
        if I_TITLE_KEY.search(lk) and not I_NOT_TITLE_KEY.search(lk):
            return v
    return ""


def _i_page_spec(name, spec):
    ln = name.lower()
    if not isinstance(spec, dict) or spec.get("type") not in ("integer", "number"):
        return False
    return bool(I_PAGE_NAME.search(ln)) and not I_NOT_PAGE_NAME.search(ln)


def _i_floor(spec, paged):
    first = spec.get("minimum", spec.get("default"))      # (a minimum key of no number hides the default)
    if numeric(first):
        return first
    return 1 if paged else 0


def _i_query_words(text):
    words: list = []
    for piece in text.split():
        bit = piece.lower().strip(I_STRIP)
        if I_ANY_ALNUM.search(bit) and bit not in words:
            words.append(bit)
    return words


def _i_extends(longer, shorter):
    return len(longer) > len(shorter) and longer[:len(shorter)] == shorter


def _i_word_verdict(words, total, old_words, old_total, cut):
    """"any"/"all" when one query extends the other, "subset" when one word set holds the other, else ""."""
    if _i_extends(words, old_words):
        return "any" if total > old_total else "all"
    if _i_extends(old_words, words):
        return "any" if total < old_total else "all"
    new_set, old_set = set(words[:cut]), set(old_words[:cut])
    if (new_set > old_set and total < old_total) or (old_set > new_set and old_total < total):
        return "subset"
    return ""


def _i_common_lead(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _i_in_order(marks):
    if marks == sorted(marks):
        return True
    keys = [_i_natural(m) for m in marks]
    return keys == sorted(keys)


def _i_label_runs(label):
    runs: list = [[]]
    for piece in str(label).split():
        word = piece.lower().strip(".,;:!?'\"()[]")
        if word in FILLER or not I_LABEL_WORD.fullmatch(word):
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append(word)
    return [run for run in runs if run]


def _i_grams(title, maker, pairs):
    maker_words = set(I_ALPHA.findall(_i_low(maker)))
    words = I_ALPHA.findall(title)
    ok = [len(w) >= 3 and w not in FILLER and w not in maker_words for w in words]
    # (a pair is two neighbours in the title itself: a skipped word between them breaks it)
    if not pairs:
        return [(w,) for w, good in zip(words, ok) if good]
    return [(a, b) for a, b, ga, gb in zip(words, words[1:], ok, ok[1:]) if ga and gb]


def _i_makers(row):
    out = []
    for k, v in row.items():
        if not isinstance(v, str) or not v.strip():
            continue
        lk = _i_low(k)
        if I_SELLER_KEY.search(lk) and "id" not in lk:
            out.append(set(I_ALNUM.findall(v.lower())))
    return out


def _i_record_words(d):
    """Words of a record's free text, less its seller and brand words."""
    own, text = set(), set()
    for key, v in d.items():
        if not isinstance(v, str):
            continue
        lk = _i_low(key)
        tokens = set(I_ALNUM.findall(v.lower()))
        seller = I_SELLER_KEY.search(lk) and "id" not in lk and len(v) <= 40 and not I_NOTE_KEY.search(lk)
        if seller:
            own |= tokens
        elif not I_HEAD_KEY.search(lk):
            text |= tokens
    return text - own


def _i_touches(left, right, lead_digits):
    for a in left:
        for b in right:
            if _i_near_figure(a, b, lead_digits):
                return True
    return False


def _i_raw_tokens(raw):
    toks = []
    for piece in " ".join(raw or []).split():
        bit = piece.strip("'\"()[]{},.;!?|").lower()
        if I_ANY_ALNUM.search(bit):
            toks.append(bit)
    return toks


def _i_content(texts, rx):
    return {w for t in texts for w in rx.findall(t) if w not in FILLER}


def _i_trap_reason(ps, known, names, dropped):
    if not ps or _i_touches(ps, known, False):
        return ""
    if _i_touches(ps, names, False):
        return "name"
    if _i_touches(ps, dropped, False):
        return "dropped"
    return ""


def _i_pack(vals, width):
    query: list = []
    for words in vals:
        if len(query) + len(words) <= width:
            query += words
    return query


def _i_alt_carried(alt, words, toks):
    amount = SelfDriveMixin.fig_amount(alt)
    if amount is not None:
        figure, unit = amount
        return figure in toks and any(t == unit or _i_near(t, unit) for t in toks)
    if SelfDriveMixin.amount_part(alt):
        return False
    runs = I_ALNUM3.findall(alt)
    return any(_i_near(w, r) for w in words for r in runs)


class SelfDriveMixin:
    """What the code does on its own: a plan from the request's words, picks from the records, its own searches,
    cart calls and finish; refusal and message notes; the store's search habits and paging; the shelf sweep; search
    advice; and the checks that hold a decline back."""

    def _selfdrive_state(self) -> None:
        self.code_fit: dict = {}
        self.code_hits: dict = {}
        self.code_more: list = []
        self.code_searched = False
        self.news_read: set = set()
        self.reread = False
        self.refusal_texts: set = set()
        self.page_book: dict = {}
        self.swept = 0
        self.sweep_turn = 0
        self.sweep_tried: set = set()
        self.ref_hits: list = []
        self.hit_shelf: dict = {}
        self.pointed_memo = None
        self.fit_held = False

    # ---- small shared reads
    def _i_search_props(self) -> tuple:
        tool = self.role("search")
        schema, props = self._i_tool_props(tool)
        return tool, schema, props

    def _i_tool_props(self, tool: Any) -> tuple:
        found = self.shapes.get(tool) if tool else None
        if not isinstance(found, dict):
            return {}, {}
        inner = found.get("properties")
        return found, (inner if isinstance(inner, dict) else {})

    def _i_tool_desc(self, tool: Any) -> str:
        for entry in self.toolset:
            fn = entry.get("function") if isinstance(entry, dict) else None
            if not isinstance(fn, dict) or fn.get("name") != tool:
                continue
            return str(fn.get("description") or "")
        return ""

    def _i_shelf_enum(self, tool: Any) -> tuple:
        """The longest enum list (4+ values) among a tool's properties, first on a tie: (key, values)."""
        best_key, best = None, []
        for name, spec in self._i_tool_props(tool)[1].items():
            values = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(values, list) and len(values) >= 4 and len(values) > len(best):
                best_key, best = name, values
        return best_key, best

    def _i_version_sets(self) -> tuple:
        ver = str(self.needs.version)
        cut = [(str(key).split("|"), ruling) for key, ruling in self.rulings.items()]
        failed = {bits[0] for bits, ruling in cut if ruling and len(bits) > 1 and bits[1] == ver}
        ends = [str(key).rpartition("|") for key in self.passed]
        live = {trio[0] for trio in ends if trio[2] == ver}
        return failed, live

    def _i_audit_entry(self) -> dict | None:
        for entry in reversed(self.log):
            if isinstance(entry, dict) and entry.get("kind") == "audit":
                return entry
        return None

    def _i_reviewed(self) -> set:
        reviewed = set(self.audit_ids or ())
        # (an audit count still at its "never" mark compares as no count at all)
        counted = numeric(self.audit_reads) and self.audit_reads < G_NEVER
        if self.audit_done and (self.audit_ids is None or (counted and len(self.archive.read) <= self.audit_reads)):
            reviewed = set(self.archive.read)
        return reviewed

    # ================================================================== A. a plan from the request's own words
    @staticmethod
    def code_limit(low: str, nouns: Any) -> int | None:
        """nouns: a regex alternation (or a list of alternatives) naming the counted things."""
        alternatives = nouns if isinstance(nouns, str) else "|".join(str(n) for n in nouns)
        spelled = "|".join(I_COUNT_WORDS)
        shape = re.compile(r"\b(\d+|%s)\s+(?:[a-z\-]+\s+){0,2}?(?:%s)\b" % (spelled, alternatives))
        for m in shape.finditer(low):
            start, end = m.start(), m.end()
            if I_LIMIT_BEFORE.search(low[max(0, start - 60):start]) or I_LIMIT_AFTER.match(low[end:end + 40]):
                return _i_count(m.group(1))
        return None

    @staticmethod
    def code_count(low: str) -> int:
        m = I_UNIT_COUNT.search(low)
        n = _i_count(m.group(1)) if m else 0
        return n if 1 <= n <= 99 else 1

    @staticmethod
    def _i_need_kept(prop: str, value: str, tail: str) -> bool:
        if points_elsewhere(value):
            return False
        if points_elsewhere(tail) and I_PAGE_TAIL.search(tail):
            return False
        if not value:
            return False
        if len(value.split()) > 4 or len(prop.split()) > 4:
            return False
        prop_words = set(I_ALPHA.findall(prop.lower()))
        if prop_words & I_NEED_NOT_PROP:
            return False
        return not I_NEED_NOT_VALUE.match(value.lower())

    @staticmethod
    def code_needs(seg: str) -> list:
        found = []
        for m in I_NEED_SHAPE.finditer(seg):
            prop = m.group("prop").strip()
            value = m.group("value").strip()
            stop = m.end()
            if SelfDriveMixin._i_need_kept(prop, value, seg[stop:stop + 60]):
                found.append((prop, value))
        return found

    @staticmethod
    def _i_label_spot(toks: list, own: list, low: str) -> tuple | None:
        """Where a label's words stand together in the request, with the count before them: (start, end, count)."""
        n = len(own)
        for i in range(len(toks) - n + 1):
            window = toks[i:i + n]
            if [t[0] for t in window] != own:
                continue
            start, end = window[0][1], window[-1][2]
            if len(I_ALNUM.findall(low[start:end])) > 2 * n:
                continue
            hit = I_ITEM_COUNT.search(low[max(0, start - 56):start])
            if hit:
                return start, end, _i_count(hit.group(1))
        return None

    def code_items(self, tool: Any) -> list:
        labels = self._i_shelf_enum(tool)[1]
        if not labels:
            return []
        low = self.request.lower()
        toks = []
        for m in I_ALPHA3.finditer(low):
            if m.group(0) not in FILLER:
                toks.append((m.group(0).rstrip("s"), m.start(), m.end()))
        marks, named = [], set()
        for label in labels:
            own = [w.rstrip("s") for w in I_ALPHA3.findall(str(label).lower()) if w not in FILLER]
            if not own or str(label) in named:
                continue
            spot = self._i_label_spot(toks, own, low)
            if spot is not None and 1 <= spot[2] <= 99:
                marks.append(spot + (label,))
                named.add(str(label))
        marks = sorted(marks, key=lambda mark: mark[0])
        ahead = [mark[0] for mark in marks[1:]] + [None]
        out = []
        for (start, end, count, label), nxt in zip(marks, ahead):
            if nxt is None:
                stretch = low[end:]
                stop = I_SENTENCE_END.search(stretch)
                if stop:
                    stretch = stretch[:stop.start() + 1]
            else:
                stretch = low[end:nxt]
            name = str(label).lower()
            reqs = ["%s (%s): %s" % (p, name, v) for p, v in self.code_needs(stretch)]
            out.append({"product": name, "label": label, "quantity": count, "requirements": reqs})
        return out

    def _i_tool_card(self, entry: Any) -> dict | None:
        fn = entry.get("function") if isinstance(entry, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            return None
        schema, props = self._i_tool_props(name)
        head = I_CAMEL.sub(r"\1 \2", name).replace("_", " ").lower()
        listed = schema.get("required")
        about = str(fn.get("description") or "").lower()
        return {"name": name, "head": head, "text": head + " | " + about, "props": props,
                "req": [r for r in listed if r in props] if isinstance(listed, list) else [],
                "free": [key for key, spec in props.items() if _i_plain_str(spec)]}

    @staticmethod
    def _i_two_way(spec: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        values = [str(v).lower() for v in (spec.get("enum") or [])]
        return any(ORDER_SENSE.search(v) for v in values) and any(DECLINE_SENSE.search(v) for v in values)

    @staticmethod
    def _i_fits_role(role: str, tool: dict) -> bool:
        text = tool["text"]
        if role == "search":
            if not I_SEEK.search(text) or (I_PERSON.search(text) and I_SPEAK.search(text)):
                return False
            if tool["free"]:
                return True
            return any(isinstance(s, dict) and s.get("enum") for s in tool["props"].values())
        need = tool["req"]
        if len(need) != 1 or need[0] not in tool["free"]:
            return False
        if role == "message":
            return bool(I_SPEAK.search(text) and (I_PERSON.search(text) or I_CHATTY.search(text)))
        return bool((I_DETAIL if role == "details" else I_VIEW).search(text))

    def code_roles(self) -> dict:
        tools = [card for card in map(self._i_tool_card, self.toolset) if card is not None]
        roles: dict = {}
        taken: set = set()

        def give(role: str, tool: dict) -> None:
            roles[role] = tool["name"]
            taken.add(tool["name"])

        def by_words(tool: dict, text: str) -> None:
            if "decline" not in roles and DECLINE_SENSE.search(text):
                give("decline", tool)
            elif "order" not in roles and ORDER_SENSE.search(text):
                give("order", tool)
        for tool in tools:
            if any(self._i_two_way(tool["props"][r]) for r in tool["req"]):
                for role in ("order", "decline"):
                    if role not in roles:
                        give(role, tool)
        for tool in tools:
            if tool["name"] not in taken:
                by_words(tool, tool["head"])
        for tool in tools:
            if tool["name"] not in taken and not tool["req"] and I_FINAL.search(tool["text"]):
                by_words(tool, tool["text"])
        for tool in tools:
            if tool["name"] in taken or not I_CART_WORD.search(tool["text"]):
                continue
            listy = any(isinstance(s, dict) and s.get("type") == "array" for s in tool["props"].values())
            if not (tool["req"] or listy):
                if "cart_view" not in roles:
                    give("cart_view", tool)
            elif "cart" not in roles:
                give("cart", tool)
        for role in ("search", "message", "details", "view"):
            chosen = next((tool for tool in tools if tool["name"] not in taken and self._i_fits_role(role, tool)), None)
            if chosen is not None:
                give(role, chosen)
        return roles

    def code_label(self, tool: Any, want: Any) -> str:
        want = set(want or ())
        best, best_rank = "", (0, 0.0)
        for label in self._i_shelf_enum(tool)[1]:
            words = kind_set(label)
            hit = len(words & want)
            if hit == 0:
                continue
            rank = (hit, hit / max(1, len(words)))
            if rank > best_rank:
                best, best_rank = str(label), rank
        return best

    # ================================================================== B. evidence and picks from the records
    def code_values(self, k: Any) -> list:
        needs = self.needs
        several = bool(needs.items())
        out = []
        for req in needs.props_left():
            head, _, value = str(req).partition(":")
            if several and self.product_of(head) not in (k, None):
                continue
            plain = _i_marks_off(value).strip()
            short = _i_terse(plain)
            if (names_the_item(value, head) or short) and plain:
                if points_elsewhere(plain):
                    continue
                out.append((head.partition("(")[0].strip(), plain))
        return out

    def code_rows(self) -> dict:
        known = self.archive.ids
        rows: dict = {}

        def visit(d):
            mark = self.row_mark(d)
            if mark in known:
                _i_fold_row(rows.setdefault(mark, _i_blank_row()), d, self.shelf_of(d))
        for obs in self.archive.items:
            each_dict(obs, visit)
        return rows

    @staticmethod
    def code_seen(value: Any, text: Any) -> bool:
        hay = str(text or "").lower()
        if not hay.strip():
            return False
        body = I_LINKED.sub(" ", _i_low(value))
        for alt in I_ALT_SPLIT.split(body):
            if _i_alt_seen(alt, hay):
                return True
        return False

    @staticmethod
    def code_shows(row: dict, prop: Any, value: Any, hit_words: Any, read: bool) -> bool:
        seen = SelfDriveMixin.code_seen
        named = _i_named_texts(row["fields"], prop)
        if named:
            return any(seen(value, text) for text in named)
        in_title = seen(value, row["title"])
        if in_title or read:
            return in_title
        words = kind_set(value)
        return bool(words) and words <= set(hit_words or ())

    @staticmethod
    def code_clash(fields: dict, prop: Any, value: Any) -> bool:
        named = _i_named_texts(fields, prop)
        if not named:
            return False
        return not any(seen_in_samples(value, text) for text in named)

    def _i_value_rank(self, row, prop, value):
        options = set(map(_i_spoken, I_OR_SLASH.split(_i_low(value))))
        options.discard("")
        ranks = [(2, 3)]
        for key, text in row["fields"].items():
            if not loosely_same(key, prop) or not self.code_seen(value, text):
                continue
            if _i_spoken(text) in options:
                match = 0
            else:
                match = 1 if any(_i_spoken(p) in options for p in list_parts(text)) else 3
            exact = name_key(key) == name_key(prop)
            ranks.append((0 if exact else 1, match))
        best = min(ranks)
        if best == (2, 3) and self.code_seen(value, row["title"]):
            best = (2, 2)
        return best

    def _i_lone_home(self):
        only = self.shelf_labels()
        if only:
            return only
        homes = self.home_shelves()
        return homes[0] if homes else set()

    def _i_audited(self, live):
        entry = self._i_audit_entry()
        if entry is None or not isinstance(entry.get("result"), dict):
            return set()
        return self._i_reviewed() - set(self.audit_fits or ()) - live

    def _i_apart_specs(self, k):
        apart = []
        for rec in self.differ_done(k) or []:
            raw = [rec.get("value")] + list(rec.get("goes_with") or [])
            prop = str(rec.get("prop") or "")
            apart.append((prop, [str(t) for t in raw if str(t or "").strip()]))
        return apart

    def _i_row_open(self, ident, row, spot, gate):
        k, home, _, want, _, judged, _ = spot
        if row["stock"] is False or ident in self.refused_lines:
            return False
        if any(ident in group for group in gate):
            return False
        placed = row["shelf"] in home if home else bool(want & row["words"])
        if not placed:
            return False
        fields = row["fields"]
        if any(self.code_clash(fields, p, v) for p, v in judged):
            return False
        return not self.ref_excluded(fields, k)

    def _i_pick_entry(self, ident, row, spot, ctx):
        if not self._i_row_open(ident, row, spot, ctx["gate"]):
            return None
        _, _, qty, want, vals, judged, apart = spot
        cap = ctx["cap"]
        listed = row["price"]
        cost = listed if numeric(listed) else self.archive.price(ident)
        priced = numeric(cost)
        if cap is not None and priced and cost * qty > cap:
            return None
        hit_words = self.code_hits.get(ident) or set()
        was_read = ident in ctx["read"]
        fits = all(self.code_shows(row, p, v, hit_words, was_read) for p, v in judged) and \
            _i_apart_ok(row["fields"], apart)
        shows = []
        if was_read:
            shows = [self._i_value_rank(row, p, v) for p, v in judged]
        if ident in ctx["audited"]:
            trusted = ctx["own_all"] and fits and (judged or apart) and all(a <= 1 and b <= 1 for a, b in shows)
            if not trusted:
                return None
        words = row["words"]
        shared_vals = len(vals & words)
        score = 3 * shared_vals + 2 * len(want & words) + len(ctx["asked"] & words) + 4 * len(vals & hit_words)
        price = float(cost) if priced else I_INF
        quality = (sum(a for a, _ in shows), sum(b for _, b in shows)) if shows else (9, 9)
        unknown = cap is not None and not priced
        kept_first = 0 if ident in ctx["keep"] else 1
        live_first = 0 if ident in ctx["live"] else 1
        read_first = 0 if was_read else 1
        cheap_key = price if ctx["cheapest"] else 0
        full = (kept_first, live_first, not fits, unknown, cheap_key, read_first, row["stock"] is not True,
                (quality, -score), price, ident)
        return full, (not fits, unknown, price, ident), ident, price, fits

    def code_picks(self, avoid: Any = None, keep: Any = None) -> list:
        needs = self.needs
        items = needs.items()
        several = bool(items)
        counts = needs.counts()
        homes = self.home_shelves() if several else [self._i_lone_home()]
        rows = self.code_rows()
        failed, live = self._i_version_sets()
        audited = self._i_audited(live)
        own_all = bool(audited) and self.code_judges_all()
        limit = needs.spend_cap
        cap = float(limit) if numeric(limit) else None
        asked = kind_set(self.request)
        cheapest = needs.priority() == "cheapest"
        avoid = set(avoid or ()) | set(self.dead_lines()) | set(self.ref_pointers())
        keep = set(keep or ())
        read = self.archive.read
        ctx = dict(gate=(failed, avoid), cap=cap, read=read, audited=audited, own_all=own_all, asked=asked,
                   keep=keep, live=live, cheapest=cheapest)
        pools = []
        for at, home in enumerate(homes):
            k = at if several else None
            qty = counts[at] if at < len(counts) else 1
            want = kind_set(items[at]["product"] if several else (needs.reading.get("product") or ""))
            pairs = self.code_values(k)
            vals = set().union(*(kind_set(plain) for _, plain in pairs))
            judged = [pair for pair in pairs if not _i_is_bound(pair[1])]
            spot = (k, home, qty, want, vals, judged, self._i_apart_specs(k))
            entries = (self._i_pick_entry(ident, row, spot, ctx) for ident, row in rows.items())
            pool = [entry for entry in entries if entry is not None]
            if not pool:
                return []
            pools.append((qty, pool))
        verdicts: dict = {}

        def fails(ident):
            if ident not in read:
                return False
            if ident not in verdicts:
                view = self.field_view(ident) or {}
                verdicts[ident] = bool(view.get("fails"))
            return verdicts[ident]
        for which in (0, 1):
            settled = _i_settle(pools, which, fails)
            if settled is None:
                return []
            picks, fit_of, total = settled
            if cap is None or total <= cap:
                self.code_fit = fit_of
                return picks
        return []

    # ================================================================== C. the code's own searches
    def _i_query_texts(self, k, product, label):
        names = [w for w in I_ALNUM.findall(product.lower()) if len(w) > 2 and w not in FILLER]
        kind = names[-1] if names else ""
        lead = None
        for _, v in self.code_values(k):
            lowered = v.lower()
            if I_OR_WORD.search(lowered) or "/" in v or "," in v or I_LINKED.search(lowered):
                continue
            lead = v
            break
        words = [] if lead is None else [w for w in I_ALNUM.findall(lead.lower()) if w not in FILLER]
        if words:
            texts = [_i_phrase(words + [kind])]
            if len(words) >= 2:
                texts.append(_i_phrase([max(words, key=len), kind]))
            return texts
        if names:
            return [_i_phrase([names[0], names[-1]])]
        return [" ".join(sorted(kind_set(label))[:2])]

    def code_queries(self, k: Any) -> list:
        tool, schema, props = self._i_search_props()
        needed = [r for r in (schema.get("required") or []) if r in props]
        needs = self.needs
        items = needs.items()
        valid = bool(items) and isinstance(k, int) and 0 <= k < len(items)
        product = str(items[k]["product"] if valid else (needs.reading.get("product") or ""))
        label = self.code_label(tool, kind_set(product) or kind_set(self.request))
        parts = I_DESC_SPLIT.split(self._i_tool_desc(tool).lower())
        tk = self.token_key()
        text_key = _i_text_key([n for n, spec in props.items() if _i_plain_str(spec) and n != tk])
        shelf_key = self._i_shelf_enum(tool)[0]
        modes = [n for n, spec in props.items() if n != shelf_key and _i_short_enum(spec)]
        mode_key = ([n for n in modes if n in needed] or modes or [None])[0]

        def mode_for(arg, last):
            options = props[mode_key]["enum"]
            arg_rx = r"\b%s\b" % re.escape(_i_low(arg))
            for option in options:
                option_rx = r"\b%s\b" % re.escape(_i_low(option))
                if any(re.search(option_rx, part) and re.search(arg_rx, part) for part in parts):
                    return option
            return options[-1] if last else options[0]
        cap_key = _i_ceiling_key(props, parts)
        qty = (needs.counts()[k] if valid else needs.quantity()) or 1
        bound = {}
        limit = needs.spend_cap
        if cap_key and numeric(limit) and limit > 0:
            whole = props[cap_key].get("type") == "integer"
            bound[cap_key] = int(limit // qty) if whole else round(limit / qty, 2)
        cap_len = props[text_key].get("maxLength") if text_key else None
        queries = self._i_query_texts(k, product, label)
        if numeric(cap_len):
            queries = [_i_trimmed(q, cap_len) for q in queries]
        first_q, second_q = (queries + ["", ""])[:2]
        plan = []
        if text_key and first_q:
            plan.append((text_key, first_q, False))
        if label and shelf_key:
            plan.append((shelf_key, label, True))
        if text_key and len(queries) > 1:
            plan.append((text_key, second_q, False))
        out: list = []
        for key, value, last in plan:
            call = {key: value}
            if mode_key:
                call[mode_key] = mode_for(key, last)
            call.update(bound)
            for n in needed:
                if n in call:
                    continue
                known, filler = _i_default_for(props[n], first_q or label)
                if known:
                    call[n] = filler
            fixed, faults = _i_fix_call(call, schema, bound)
            if faults or fixed in out:
                continue
            out.append(fixed)
        return out

    def code_replies(self, n: int) -> list:
        env = next((e for e in reversed(self.log) if isinstance(e, dict) and e.get("kind") == "env"), None)
        reply = None if env is None else env.get("reply")
        if not isinstance(reply, dict):
            return []
        many = reply.get("observations")
        if not isinstance(many, list):
            many = [reply.get("observation")]
        return many[:n]

    @staticmethod
    def code_fault(obs: Any) -> bool:
        flagged: list = []

        def visit(d):
            flagged.extend(key for key, v in d.items()
                           if I_FAULT.search(_i_low(key)) and v not in (None, False, "", [], {}))
        each_dict(obs, visit, 2)
        return len(flagged) > 0

    def _i_slab(self, form, tool, calls):
        stamp = self.wire.tick
        return [(form % (n, stamp), tool, call) for n, call in enumerate(calls)]

    def _i_tag_rows(self, reply, words):
        def visit(d):
            mark = self.row_mark(d)
            if mark not in self.archive.ids:
                return
            if mark not in self.code_hits:
                self.code_hits[mark] = set()
            self.code_hits[mark].update(words)
        each_dict(reply, visit)

    def _i_note_hits(self, calls, replies, text_keys):
        for call, reply in zip(calls, replies):
            said = " ".join([str(call.get(key) or "") for key in text_keys])
            if said.strip():
                self._i_tag_rows(reply, kind_set(said))

    def code_search(self, products: int, more: bool = False) -> None:
        tool = self.role("search")
        if not tool:
            return
        text_keys = [name for name, spec in self._i_search_props()[2].items() if _i_plain_str(spec)]
        reserve = 2 + 2 * products
        left = self.moves_left
        if more:
            lead = list(self.code_more)
            calls = lead[:max(0, left() - reserve - 2 * products)]
            rounds = 1
        else:
            lists = []
            for k in range(products):
                lists.append(self.code_queries(k if self.needs.items() else None))
            # (a call is left out only when an earlier rank holds it: equal calls of one rank all go)
            calls = _i_by_rank(lists)[:max(0, left() - reserve)]
            lead = [x[0] for x in lists if x]
            extra = 2 if self.needs.priority() == "cheapest" else 0
            rounds = 1 + Knobs.MORE_PAGES + extra
        if not calls:
            return
        self.code_searched = True
        last = rounds - 1
        for rnd in range(rounds):
            self.send_chunk(self._i_slab("code_search_%d_%d", tool, calls))
            replies = self.code_replies(len(calls))
            if self.word_kind != "any":
                self._i_note_hits(calls, replies, text_keys)
            followers = (self.code_next(c, r) for c, r in zip(calls, replies) if c in lead)
            following = [nxt for nxt in followers if nxt]
            self.code_more = following
            if rnd == last:
                break
            picks = self.code_picks()
            if self.needs.priority() != "cheapest":
                if picks and all(self.code_fit.get(i) for i, _ in picks):
                    break
            lead = following
            calls = following[:max(0, left() - reserve - 3 * products)]
            if not calls:
                break

    def code_next(self, call_args: dict, obs: Any) -> dict | None:
        paging = self.page_arg()
        if paging is None:
            return None
        key, by, low, high = paging
        page = _i_first_table(obs)
        if page is None:
            return None
        rows = max(map(len, [v for v in page.values() if isinstance(v, list)]))
        marks, total = [], None
        for name, v in page.items():
            lowered = _i_low(name)
            if I_MARK_KEYS.search(lowered):
                marks.append(v)
            if total is None and numeric(v) and I_TOTAL_KEY.search(lowered):
                total = v
        if marks and all(v in (None, False, "", 0) for v in marks):
            return None
        if by == "token":
            nxt = next((v for v in marks if isinstance(v, str) and v.strip()), None)
        else:
            nxt = _i_numeric_next(call_args.get(key, low), by, low, high, rows, bool(marks), total)
        if nxt is None:
            return None
        fixed, faults = ArgFixer.fix({**call_args, key: nxt}, self._i_search_props()[1])
        if faults:
            return None
        return fixed

    # ================================================================== D. the cart tool, read from its schema
    def cart_form(self) -> tuple:
        schema, props = self._i_tool_props(self.role("cart"))
        saved = self.trolley.put_form
        known = (first_json(saved) if saved else None) or {}
        puts, takes = {}, {}
        for name in [r for r in (schema.get("required") or []) if r in props]:
            spec = props[name]
            options = spec.get("enum") if isinstance(spec, dict) else None
            if isinstance(options, list) and options:
                puts[name], takes[name] = _i_put_take(name, options, known)
        free = [n for n, spec in props.items() if _i_plain_str(spec)]
        id_arg = next((n for n in free if I_ID_ARG.search(n.lower())), None)
        if id_arg is None and free:
            id_arg = free[0]
        units = [n for n, spec in props.items() if _i_count_spec(spec)]
        return schema, puts, takes, id_arg, units

    def cart_args(self, picks: list) -> list:
        schema, puts, _, id_arg, units = self.cart_form()
        spec = self.trolley.line_spec()
        if spec:
            lines = [{spec[1]: ident, spec[2]: n} for ident, n in picks]
            drafts = [{**puts, spec[0]: lines}]
        elif id_arg:
            drafts = [_i_line_call(puts, id_arg, ident, units, n) for ident, n in picks]
        else:
            return []
        fixed = _i_all_fixed(drafts, schema)
        return [] if fixed is None else fixed

    def cart_drops(self, lines: list) -> list | None:
        book = self.trolley
        if not lines or book.line_spec():
            return []
        schema, puts, takes, id_arg, units = self.cart_form()
        if not id_arg:
            return None
        props = self._i_tool_props(self.role("cart"))[1]
        if puts:
            if any(v is None for v in takes.values()):
                return None
            drafts = []
            for line in lines:
                call = {**takes, id_arg: line}
                if units:
                    call[units[0]] = max(1, int(book.held(line)) or 1)
                drafts.append(call)
        else:
            floor = props[units[0]].get("minimum") if units else None
            if not units or (numeric(floor) and floor > 0):
                return None
            drafts = [{id_arg: line, units[0]: 0} for line in lines]
        return _i_all_fixed(drafts, schema)

    def _i_read_slab(self, unread, batch, details):
        stamp = self.wire.tick
        if batch:
            tool, arg, most = batch
            return [("code_read_0_%d" % stamp, tool, {arg: unread[:most]})]
        slab = []
        for n, ident in enumerate(unread):
            args = self.details_call(ident)
            if args:
                slab.append(("code_read_%d_%d" % (n, stamp), details, args))
        return slab

    def code_read(self, picks: list, avoid: Any, keep: Any) -> list:
        for _ in range(Knobs.READ_ROUNDS):
            unread = [ident for ident, _ in picks if ident not in self.archive.read]
            batch, details = self.batch_tool(), self.role("details")
            grouped = bool(batch) and len(unread) >= 2
            reads = 1 if grouped else len(unread)
            if not unread or not (batch or details):
                break
            if self.moves_left() < reads + 2 * len(picks) + 3 or self.wire.left() <= 8:
                break
            slab = self._i_read_slab(unread, batch if grouped else None, details)
            if not slab:
                break
            self.send_chunk(slab)
            fresh = self.code_picks(avoid, keep)
            if fresh == picks:
                break
            picks = fresh
        return picks

    def _i_blamed(self, reply, call, wanted, spec):
        text = as_text(reply)
        named = {i for i in wanted if i in text}
        if named:
            return named
        named = {r for r, _ in self.trolley.lines(call) if r in wanted}
        if named or not spec:
            return named
        return set(wanted)

    def code_put(self, picks: list) -> list | None:
        book = self.trolley
        tool = self.role("cart")
        wanted = dict(picks)
        spec = book.line_spec()
        extra = [line for line in book.units if line not in wanted]
        todo = list(picks) if spec else [(i, n) for i, n in picks if abs(book.held(i) - n) > 1e-9]
        drops = self.cart_drops(extra)
        puts = self.cart_args(todo) if todo else []
        if drops is None or (todo and not puts):
            return None
        calls = drops + puts
        if not calls:
            return []
        if self.moves_left() < len(calls) + 1:
            return None
        self.send_chunk(self._i_slab("code_cart_%d_%d", tool, calls))
        replies = self.code_replies(len(calls))

        def missing(pool):
            return {i for i in wanted if i not in pool}
        shown, bad = None, set()
        for call, reply in zip(calls, replies):
            listed = book.view_units(reply)
            if listed:
                shown = listed
            if self.code_fault(reply):
                bad |= self._i_blamed(reply, call, wanted, spec)
        if len(replies) < len(calls):
            return None
        view = self.role("cart_view")
        if shown is None and not bad and view and self.moves_left() >= 3:
            self.send_chunk([("code_cart_view_%d" % self.wire.tick, view, {})])
            got = self.code_replies(1)
            if got:
                shown = book.view_units(got[0])
                if shown is None:
                    bad |= missing(book.units)
        if shown:
            bad |= missing(shown)
        for ident in bad:
            book.units.pop(ident, None)
        return sorted(bad)

    def code_recheck(self, lines: Any) -> tuple:
        lines = list(lines or [])
        tool = self.stock_tool() or self.role("view") or self.role("details")
        if lines and tool and self.moves_left() >= len(lines) + 2 and self.wire.left() >= 8:
            stamp = self.wire.tick
            asks = [(n, self.lookup_call(line, tool)) for n, line in enumerate(lines)]
            slab = [("code_check_%d_%d" % (n, stamp), tool, args) for n, args in asks if args]
            if slab:
                self.send_chunk(slab)
                rows = self.code_rows()
                sold = {line for line in lines if (rows.get(line) or {}).get("stock") is False}
                return True, sold
        return False, set()

    # ================================================================== E. the code's own finish
    def code_news(self, lines: Any) -> None:
        heard = list(lines or [])
        for entry in list(self.needs.pending):
            if isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], list):
                heard += entry[0]
        fresh = [line for line in heard if isinstance(line, str)]
        for line in fresh:
            if line in self.news_read:
                continue
            self.news_read.add(line)
            got = self.code_budget(line)
            figure = got[0] if isinstance(got, (tuple, list)) else got
            if not numeric(figure) or figure <= 0:
                continue
            self.needs.spend_cap = figure
            self.log.append(dict(kind="code_news", budget=figure, line=line[:200]))

    def _i_cart_round(self, avoid, keep, products, search):
        """One pick/search/read/cart pass: True or False ends the finish; a list names lines to swap out."""
        picks = self.code_picks(avoid, keep)
        unfit = not picks or any(not self.code_fit.get(i) for i, _ in picks)
        if search and unfit and not self.code_searched and self.moves_left() >= 4:
            self.code_search(products)
            picks = self.code_picks(avoid, keep)
        elif search and self.code_more and self.moves_left() >= 6 + 2 * products and \
                (unfit or self.needs.priority() == "cheapest"):
            self.code_search(products, True)
            picks = self.code_picks(avoid, keep)
        picks = self.code_read(picks, avoid, keep)
        if not picks:
            self.log.append(dict(kind="code_finish", result="nothing to order"))
            return False
        bad = self.code_put(picks)
        if bad is None:
            self.log.append(dict(kind="code_finish", result="no cart call"))
            return False
        if not bad and self.moves_left() >= len(picks) + 3:
            bad = sorted(self.code_recheck([i for i, _ in picks])[1])
        if bad:
            return bad
        self.log.append(dict(kind="code_finish", picks=picks, cart=dict(self.trolley.units)))
        return bool(self.trolley.units)

    def code_cart(self, avoid: Any = None, keep: Any = None) -> bool:
        wire = self.wire
        if wire.done or wire.broke or not self.role("cart"):
            return False
        if self.moves_left() < 2 or wire.left() < 4:
            return False
        wire.mute = True
        try:
            products = max(1, len(self.needs.items()))
            avoid = set(avoid or ())
            search = self.role("search")
            for _ in range(4):
                outcome = self._i_cart_round(avoid, keep, products, search)
                if isinstance(outcome, bool):
                    return outcome
                avoid |= set(outcome)
                self.log.append(dict(kind="code_swap", bad=list(outcome)))
            return False
        except Exception as error:  # noqa: BLE001
            said = "%s: %s" % (type(error).__name__, error)
            self.log.append(dict(kind="code_finish_error", error=said[:500]))
            return False

    def _i_cart_dead(self, avoid):
        """Before the first finish: drop cart lines the store no longer sells (avoid grows); (gone, keep or None)."""
        self.wire.mute = True
        keep = None
        try:
            gone = set(self.cart_gone(list(self.trolley.units)) or ())
            if gone:
                avoid.update(gone)
                keep = set(self.trolley.units) - gone
                self.log.append(dict(kind="code_dead", lines=sorted(gone)))
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="cart_gone_error", error=str(error)[:300]))
            gone = set()
        return gone, keep

    def _i_finish_choice(self, turn, gone, avoid, keep, has_order, has_decline):
        """True: order, False: decline, None: neither is possible."""
        book = self.trolley
        if has_order:
            if turn == 0 and not gone and book.used and (book.units or book.view_units(book.last_view) is None):
                return True
            if self.code_cart(avoid, keep):
                return True
            if turn > 0 and self.again_ok(avoid):
                return True
        return False if has_decline else None

    def _i_finish_act(self, tool, args):
        resp = self.wire.act([dict(name=tool, args=args)])
        kind = self.call_kind(tool, args)
        self.log.append(dict(kind="forced_finish", tool=tool, move=kind, args=args, reply=resp))
        return resp, kind

    def _i_recount(self, lines):
        """After an order call: re-read the lines' stock; (sold-out lines, over budget) or None if nothing was read."""
        was_read, sold = self.code_recheck(lines)
        if not was_read:
            return None
        sold = set(sold)
        self.refused_lines.difference_update(set(lines) - sold)
        rows = self.code_rows()
        total = 0.0
        for line in lines:
            unit = (rows.get(line) or {}).get("price")
            per = unit if numeric(unit) else 0
            total += per * self.trolley.held(line)
        budget = self.needs.spend_cap
        return sold, numeric(budget) and total > budget

    def finish_now(self) -> None:
        wire = self.wire

        def closed():
            if wire.done or wire.broke:
                return True
            return self.moves_left() <= 0 or wire.left() < 2
        if closed():
            return
        order_tool, decline_tool = self.role("order"), self.role("decline")
        order_move = decline_move = None
        if order_tool:
            order_move = (order_tool, self.plain_args(order_tool, I_ORDER_TEXT, "order"))
        if decline_tool:
            decline_move = (decline_tool, self.plain_args(decline_tool, APOLOGY, "decline"))
        avoid: set = set()
        keep = move = kind = resp = None
        turns = Knobs.CODE_FINISHES
        for turn in range(turns):
            if closed():
                return
            if move is None:
                gone: set = set()
                if turn == 0 and order_tool and self.trolley.units:
                    gone, keep = self._i_cart_dead(avoid)
                choice = self._i_finish_choice(turn, gone, avoid, keep, order_tool, decline_tool)
                if choice is None:
                    return
                move = order_move if choice else decline_move
            tool, args = move
            resp, kind = self._i_finish_act(tool, args)
            if not resp or resp.get("finished") or wire.done:
                break
            if turn == turns - 1 or self.moves_left() <= 1:
                break
            wire.mute = True
            lines = list(self.trolley.units)
            obs = resp.get("observation")
            self.absorb(tool, args, obs, resp, True)
            self.code_news(self.shopper_lines(tool, obs, True))
            move = None
            checked = self._i_recount(lines) if kind == "order" and lines else None
            if checked is None:
                continue
            sold, over = checked
            if not sold and not over:
                move = order_move
            elif over:
                avoid, keep = sold, None
            else:
                avoid, keep = sold, set(lines) - sold
        if wire.done or not resp or resp.get("finished"):
            return
        if not decline_tool or kind == "decline" or self.moves_left() <= 0:
            return
        try:
            again = bool(order_tool) and kind == "order" and self.again_ok(avoid)
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="again_error", error=str(error)[:300]))
            again = False
        tool, args = order_move if again else decline_move
        self._i_finish_act(tool, args)

    def again_ok(self, avoid: Any) -> bool:
        gone = set(avoid or ()) | set(self.dead_lines())
        lines = list(self.trolley.units)
        if not lines or any(line in gone for line in lines):
            return False
        return bool(self.code_picks(gone))

    # ================================================================== F. after a model outage
    def _i_reread_note(self, wire):
        if not self.needs.reading.get("code_reading") or self.reread or self.read_at is None:
            return None
        if wire.last_answer <= self.read_at or wire.left() <= Knobs.WIND_DOWN_S + 60:
            return None
        self.reread = True
        prompt = [dict(role="user", content=self.read_content)]
        answer = wire.ask_model(prompt, LINEUP["reader"], 3000)
        reading = first_json(answer.get("content")) if answer else None
        has_roles = isinstance(reading, dict) and isinstance(reading.get("roles"), dict)
        self.log.append(dict(kind="re_read", roles=has_roles))
        if not has_roles:
            return None
        self.use_reading(reading, again=True)
        return ("Plan updated: the models answer again, and this reading of the request replaces the "
                "first one.\n" + self.needs.overview(self.step_cap, self.msg_cap))

    def _i_pending_notes(self, wire):
        if not self.needs.pending or wire.last_answer <= (self.needs.pending_at or 0):
            return []
        if wire.left() < Knobs.WIND_DOWN_S:
            return []
        held = self.needs.pending[-3:]
        self.needs.pending = []
        notes = []
        for lines, question in held:
            heard = self.needs.hear(lines, question)
            if heard:
                notes.append("The shopper's earlier lines, read only now: " + heard)
        return notes

    def _i_attach(self, text):
        tail = self.chat[-1] if self.chat else None
        joinable = isinstance(tail, dict) and tail.get("role") in ("user", "tool") and \
            isinstance(tail.get("content"), str)
        if not joinable:
            self.chat.append(dict(role="user", content=text))
            return
        tail["content"] += "\n" + text

    def catch_up(self) -> None:
        try:
            wire = self.wire
            if wire.mute or wire.left() < Knobs.WIND_DOWN_S:
                return
            notes = []
            plan_note = self._i_reread_note(wire)
            if plan_note is not None:
                notes.append(plan_note)
            notes += self._i_pending_notes(wire)
            if notes:
                body = "\n".join(notes)
                self._i_attach(body + "\n" + self.status_line())
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="catch_up_error", error=str(error)[:300]))

    # ================================================================== G. notes on refusals and shopper messages
    def refused(self, call_name: str, obs: Any, resp: Any) -> bool:
        is_map = isinstance(obs, dict)
        if is_map and "refusal" in obs:
            return True
        if call_name != self.role("order") or (resp or {}).get("finished"):
            return False
        return not (is_map and obs.get("error") in ("invalid_args", "unknown_tool"))

    def _i_budget_note(self, call_name, obs):
        if not isinstance(obs, dict) or obs.get("error") != "budget_exceeded":
            return None
        if call_name == self.role("message") or self.msg_cap is not None:
            sent = self.msgs_sent
            self.msg_cap = sent if self.msg_cap is None else min(self.msg_cap, sent)
        return "That budget is spent: do not call %s again." % call_name

    def _i_decline_refused_note(self):
        self.audit_done = False
        self.decline_vetted = False
        self.event_seen = True
        return ("The store turned the decline down, so the errand is not over: it holds that something "
                "here can still be bought. Read the refusal and apply what it says, look again at the closest "
                "candidates' own properties, stock and price, and order the one that fits; decline again "
                "only if each of them truly fails.")

    def _i_order_refused_note(self, obs):
        self.refusal_n += 1
        self.order_vetted = False
        self.event_seen = True
        try:
            reread = self.refusal_reread(obs)
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="refusal_reread_error", error=str(error)[:300]))
            reread = ""
        self.refused_lines.update(self.trolley.units)
        found: list = []
        _i_long_texts(obs, found)
        self.refusal_texts = set(map(_i_flat_text, found))
        return ("The order did not go through and the cart is as it was. Apply every change the refusal "
                "states, put the cart right (take out lines that no longer fit, such as sold-out ones), check "
                "the stock and price of the new pick again, then order again." + (reread or ""))

    def _i_message_note(self, obs):
        said: list = []
        for v in obs.values():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                sentences_in(v, said)
        if not said:
            return None
        told, self.refusal_texts = self.refusal_texts, set()
        if told and all(any(_i_flat_text(line) in t for t in told) for line in said):
            return ("This result repeats the message of the refused finish, which is already handled; it "
                    "says nothing about this result itself.")
        self.event_seen = True
        return "This result carries a message from the shopper (a change) - apply it: " + " | ".join(said[:4])

    def notes_for(self, call_name: str, obs: Any, refused: bool) -> str:
        remarks = []
        spent = self._i_budget_note(call_name, obs)
        if spent is not None:
            remarks.append(spent)
        if refused:
            declined = self.last_kind == "decline" and call_name == self.role("decline")
            remarks.append(self._i_decline_refused_note() if declined else self._i_order_refused_note(obs))
        elif isinstance(obs, dict) and len(obs) >= 2 and call_name != self.role("message"):
            heard = self._i_message_note(obs)
            if heard is not None:
                remarks.append(heard)
        return " ".join(remarks)

    # ================================================================== H. record readers
    @staticmethod
    def row_mark(row: dict) -> str:
        keyed = [(_i_low(key), v) for key, v in row.items() if _i_id_like(v)]
        hit = next((v for lk, v in keyed if lk == "id"), None)
        if hit is None:
            hit = next((v for lk, v in keyed if lk.endswith("id")), None)
        if hit is not None:
            return str(hit)
        try:
            return _i_key(row)
        except (TypeError, ValueError):
            return str(row)

    @staticmethod
    def shelf_of(row: dict) -> str:
        for key, v in row.items():
            if not isinstance(v, str):
                continue
            bare = v.strip()
            lk = _i_low(key)
            if 1 <= len(bare) <= 60 and "id" not in lk and I_SHELF_KEY.search(lk):
                return bare.lower()
        return ""

    @staticmethod
    def key_stock(key: Any, val: Any) -> bool | None:
        lk = _i_low(key)
        if val is None or not I_STOCK_KEY.search(lk):
            return None
        empty = val is False or (numeric(val) and val <= 0)
        gone = False
        if "stock" in lk or "avail" in lk:
            if "out" in lk or "sold" in lk:
                gone = val is True
            else:
                gone = empty or (isinstance(val, str) and bool(I_NO_STOCK.search(val.lower())))
        if empty and I_ON_HAND.search(lk):
            gone = True
        if isinstance(val, str) and I_GONE_WORD.search(val.lower()):
            gone = True
        return not gone

    @staticmethod
    def stock_state(row: dict) -> bool | None:
        states = [SelfDriveMixin.key_stock(k, v) for k, v in row.items()]
        known = [state for state in states if state is not None]
        if not known:
            return None
        return all(known)

    @staticmethod
    def sold_out(row: dict) -> bool:
        for k, v in row.items():
            if SelfDriveMixin.key_stock(k, v) is False:
                return True
        return False

    def shelf_labels(self) -> set:
        skip = set(self.ref_foreign() or ())
        read = self.archive.read
        counted: set = set()
        tally: dict = {}

        def visit(d):
            ident = _i_listed_id(d, read)
            if ident is None or ident in counted or ident in skip:
                return
            label = self.shelf_of(d)
            if not label:
                return
            counted.add(ident)
            tally[label] = tally.get(label, 0) + 1
        for obs in self.archive.items:
            each_dict(obs, visit)
        if not tally:
            return set()
        best = _i_tops(tally, tally)
        if not self.needs.items():
            return best
        try:
            homes = self.home_shelves()
            picked: set = set()
            for home in homes:
                labels = {_i_low(h) for h in home}
                seen = [label for label in labels if label in tally]
                picked |= _i_tops(tally, seen) if seen else labels
            if not all(homes):
                ranked = [label for label, _ in sorted(tally.items(), key=lambda kv: -kv[1])]
                for label in ranked:
                    if len(picked) >= len(homes):
                        break
                    picked.add(label)
            return picked or best
        except Exception:  # noqa: BLE001
            return best

    def shelf_rows(self) -> dict:
        shelves = self.shelf_labels()
        if not shelves:
            return {}
        known = self.archive.ids
        found: dict = {}

        def visit(d):
            ident = _i_listed_id(d, known)
            if ident is None or ident in self.refused_lines:
                return
            label = self.shelf_of(d)
            if not label or label not in shelves:
                return
            if self.sold_out(d):
                found.pop(ident, None)
                return
            cost = _i_first_cost(d)
            title = _i_shelf_title(d)
            maker = " ".join(v for k, v in d.items() if isinstance(v, str) and I_MAKER_KEY.search(_i_low(k)))
            if cost is not None:
                found[ident] = (float(cost), title, maker)
        for obs in self.archive.items:
            each_dict(obs, visit)
        budget = self.needs.spend_cap
        qty = self.needs.quantity()
        out = {}
        for ident, held in found.items():
            newest = self.archive.price(ident)
            price = held[0] if newest is None else newest
            if numeric(budget) and price * qty > budget:
                continue
            out[ident] = (float(price),) + held[1:]
        return out

    def product_index(self, thing: Any) -> int | None:
        if not self.needs.items():
            return None
        label = self.shelf_known(thing)
        if not label:
            return None
        wanted = label.lower()
        for k, home in enumerate(self.home_shelves()):
            if wanted in {_i_low(h) for h in home}:
                return k
        return None

    def walk_home(self, k: Any) -> set:
        homes = self.home_shelves()
        at = 0 if k is None else k
        home = set(homes[at]) if 0 <= at < len(homes) else set()
        if home:
            return home
        if k is not None:
            return set()
        return set(self.shelf_labels())

    # ================================================================== I. the store's search habits and paging
    @staticmethod
    def stem(word: str) -> str:
        return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word

    def free_text(self, call_args: Any) -> list:
        if not isinstance(call_args, dict):
            return []
        props = self._i_search_props()[2]
        tk = self.token_key()
        texts = []
        for k, v in call_args.items():
            if k == tk or not isinstance(v, str):
                continue
            spec = props.get(k)
            if isinstance(spec, dict) and spec.get("enum"):
                continue
            texts.append(v)
        return texts

    def token_key(self) -> str | None:
        props = self._i_search_props()[2]
        return next((name for name, spec in props.items() if _i_plain_str(spec) and I_TOKEN_NAME.search(name.lower())),
                    None)

    @staticmethod
    def next_token(obs: Any) -> str | None:
        if isinstance(obs, dict):
            return next((v for key, v in obs.items() if isinstance(v, str) and v.strip() and
                         I_MARK_KEYS.search(_i_low(key))), None)
        return None

    def page_arg(self) -> tuple | None:
        for name, spec in self._i_search_props()[2].items():
            if not _i_page_spec(name, spec):
                continue
            ln = name.lower()
            by_page = "page" in ln and not I_OFFSET_NAME.search(ln)
            top = spec.get("maximum")
            return name, by_page, _i_floor(spec, by_page), (top if numeric(top) else None)
        tk = self.token_key()
        if not tk:
            return None
        return tk, "token", "", None

    def search_keys(self, call_args: Any) -> tuple:
        props = self._i_search_props()[2]
        tk = self.token_key()
        texts, mode, paging = [], [], set()
        first = True
        for key, v in (call_args.items() if isinstance(call_args, dict) else []):
            spec = props.get(key)
            if not isinstance(spec, dict):
                spec = {}
            if isinstance(v, str):
                if key == tk:
                    paging.add(key)
                    first = first and not v.strip()
                elif spec.get("enum"):
                    mode.append((key, v))
                else:
                    words = _i_query_words(v)
                    if words and all(I_ALNUM2.fullmatch(w) for w in words):
                        texts.append((key, tuple(words)))
                continue
            lk = _i_low(key)
            if numeric(v) and I_PAGE_NAME.search(lk) and not I_NOT_PAGE_NAME.search(lk):
                paging.add(key)
                if v > _i_floor(spec, "page" in lk):
                    first = False
        return texts, tuple(sorted(mode)), first, paging

    @staticmethod
    def _i_others(call_args: dict, key: str, paging: Any) -> tuple:
        rest = [(k, str(v)) for k, v in call_args.items() if k != key and k not in paging]
        return tuple(sorted(rest))

    @staticmethod
    def _i_total(obs: Any, rows: Any) -> int:
        counts = []
        if isinstance(obs, dict):
            counts = [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)]
        counts.append(len(rows or []))
        return max(counts)

    def word_rule(self, call_args: dict, obs: Any, rows: Any) -> str:
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.word_kind
        total = self._i_total(obs, rows)
        for key, words in texts:
            ctx = (key, mode)
            others = self._i_others(call_args, key, paging)
            alike = [entry for entry in self.word_seen if entry[0] == ctx and entry[1] == others and entry[3] != total]
            subset = False
            for _, _, old_words, old_total in alike:
                verdict = _i_word_verdict(words, total, old_words, old_total, self.lead_words or None)
                if verdict == "subset":
                    subset = True
                elif verdict:
                    self.word_kind = verdict
            if subset and not self.word_kind:
                self.word_kind = "all"
            self.word_seen.append((ctx, others, words, total))
        return self.word_kind

    def lead_rule(self, call_args: dict, obs: Any, rows: Any) -> int:
        texts, mode, first, paging = self.search_keys(call_args)
        if not first:
            return self.lead_words
        total = self._i_total(obs, rows)
        marks = tuple(self.row_mark(r) for r in (rows or []) if isinstance(r, dict))
        varied = len(set(marks)) >= 3
        for key, words in texts:
            ctx = (key, mode)
            others = self._i_others(call_args, key, paging)
            bounds = self.lead_bounds.setdefault(ctx, [-1, set()])
            alike = [entry for entry in self.lead_seen if entry[0] == ctx and entry[1] == others and entry[2] != words]
            for _, _, old_words, old_total, old_marks in alike:
                shorter = min(len(words), len(old_words))
                common = _i_common_lead(words, old_words)
                if old_total != total or old_marks != marks:
                    bounds[0] = max(bounds[0], common)
                elif common > 0 and varied and common == shorter:
                    bounds[1].add(common)
            self.lead_seen.append((ctx, others, words, total, marks))
        claims = []
        for ctx, (low, highs) in self.lead_bounds.items():
            above = sorted(h for h in highs if h > low)
            if above and above[0] == low + 1:
                claims.append((above[0], ctx))
        self.lead_words, self.lead_ctx = min(claims) if claims else (0, None)
        return self.lead_words

    def book_page(self, call_args: dict, rows: Any, total: Any, obs: Any = None) -> None:
        paging = self.page_arg()
        if paging is None or not rows:
            return
        key, by, first, _ = paging
        at = call_args.get(key, first)
        tokened = by == "token"
        if tokened:
            at = at.strip() if isinstance(at, str) else ""
        elif not numeric(at):
            return
        bare = {k: v for k, v in call_args.items() if k != key}
        slot = _i_key(bare)
        if slot not in self.page_book:
            self.page_book[slot] = dict(args=bare, read=set(), size=0, total=0, shelves=set(), row_shelf={})
        walk = self.page_book[slot]
        walk["read"].add(at)
        if tokened:
            tokens = walk.setdefault("tokens", {})
            tokens[at] = self.next_token(obs)
        if len(rows) > walk["size"]:
            walk["size"] = len(rows)
        seen_total = total if numeric(total) else 0
        if seen_total > walk["total"]:
            walk["total"] = seen_total
        dicts = [r for r in rows if isinstance(r, dict)]
        marks = []
        for row in dicts:
            label = self.shelf_of(row)
            if label:
                walk["shelves"].add(label)
            mark = self.row_mark(row)
            walk["row_shelf"].setdefault(mark, label)
            marks.append(mark)
        if at == first:
            walk["first"] = tuple(marks)
            if numeric(total) and total <= len(rows):
                walk["every"] = list(dicts)
        if len(marks) >= 5:
            ordered = _i_in_order(marks)
            walk["id_order"] = walk.get("id_order") is not False and ordered

    def next_page(self, walk: dict) -> Any:
        paging = self.page_arg()
        if paging is None:
            return None
        _, by, first, high = paging
        seen = walk["read"]
        if by == "token":
            tokens = walk.get("tokens") or {}
            return next((tok for tok in tokens.values() if isinstance(tok, str) and tok not in seen), None)
        low = min([first] + [v for v in seen if numeric(v)])
        size = max(1, walk["size"])
        step = 1 if by else size
        i = 1
        while i < 400:
            value = low + i * step
            if high is not None and value > high:
                return None
            if i * size >= walk["total"]:
                return None
            if value not in seen:
                return value
            i += 1
        return None

    def id_store(self) -> bool:
        flags = [w["id_order"] for w in self.page_book.values() if "id_order" in w]
        if any(f is False for f in flags):
            return False
        return sum(1 for f in flags if f is True) >= 2

    # ================================================================== J. the shelf sweep
    def label_queries(self) -> list:
        out: list = []
        for label in sorted(self.shelf_labels()):
            runs = _i_label_runs(label)
            words = [w for run in runs for w in run]
            heads = [run[-1] for run in runs]
            if len(words) >= 2:
                out.append(" ".join(words[:2]))
            out += heads
            out += [w for w in words if w not in heads]
        return _i_unique(out)

    def kind_queries(self, pairs: bool) -> list:
        stem = self.stem
        label_stems = {stem(w) for label in self.shelf_labels() for w in I_ALNUM.findall(_i_low(label))}
        titles: dict = {}
        for price, title, maker in self.shelf_rows().values():
            t = _i_low(title).strip()
            if not t:
                continue
            if t in titles:
                price = min(titles[t][0], price)
            titles[t] = (price, maker)
        tally: dict = {}
        for t, (price, maker) in titles.items():
            here: set = set()
            for gram in _i_grams(t, maker, pairs):
                stems = [stem(w) for w in gram]
                if all(x in label_stems for x in stems):
                    continue
                base = " ".join(stems)
                if base in here:
                    continue
                here.add(base)
                if base not in tally:
                    tally[base] = [0, I_INF, " ".join(gram)]
                entry = tally[base]
                entry[0] += 1
                entry[1] = min(entry[1], price)
        ranked = sorted((b for b, e in tally.items() if e[0] >= 2), key=lambda b: (tally[b][1], -tally[b][0], b))
        return [tally[b][2] for b in ranked]

    def _i_sweep_listing(self, walks, shelves):
        listing, listing_next = None, None
        for w in walks:
            if w.get("id_order") is not True:
                continue
            if not w["shelves"] or not w["shelves"] <= shelves:
                continue
            nxt = self.next_page(w)
            if nxt is None:
                continue
            if listing is None or w["total"] > listing["total"]:
                listing, listing_next = w, nxt
        return listing, listing_next

    def _i_text_walks(self, walks):
        for flag in (False, None):
            ranked = [w for w in walks if w.get("id_order") is flag and self.free_text(w["args"])]
            if ranked:
                return ranked
        return []

    def _i_sweep_template(self, ranked, shelves, page_key):
        if not ranked:
            return None, None
        args = ranked[-1]["args"]
        props = self._i_search_props()[2]

        def fixed(k, v):
            if not isinstance(v, str):
                return True
            spec = props.get(k)
            if isinstance(spec, dict) and bool(spec.get("enum")):
                return True
            return v.lower().strip() in shelves or k == page_key
        loose = [k for k, v in args.items() if not fixed(k, v)]
        if not loose:
            return None, None
        lead = self.lead_ctx
        if lead and lead[0] in loose:
            text_key = lead[0]
        else:
            text_key = max(loose, key=lambda k: len(str(args[k]).split()))
        return {k: v for k, v in args.items() if fixed(k, v)}, text_key

    def _i_signature(self, text):
        return tuple(self.stem(w) for w in I_ALNUM2.findall(text.lower())[:2])

    def sweep_next(self, turn: int) -> tuple:
        paging = self.page_arg()
        shelves = {_i_low(s) for s in self.shelf_labels()}
        if paging is None or not shelves or not self.page_book:
            return None, False
        page_key = paging[0]
        tried = self.sweep_tried
        priced, done = self.ceiling_sweep(tried)
        if priced is not None:
            return priced, True
        if done:
            return None, False
        walks = list(self.page_book.values())
        listing, listing_next = self._i_sweep_listing(walks, shelves)
        template, text_key = self._i_sweep_template(self._i_text_walks(walks), shelves, page_key)
        done_sigs = set()
        for w in walks:
            if "first" not in w or w.get("id_order") is True:
                continue
            sig = self._i_signature(" ".join(self.free_text(w["args"])))
            if sig:
                done_sigs.add(sig)

        def unseen(call):
            return None if _i_key(call) in tried else call

        def as_call(text):
            if template is None:
                return None
            sig = self._i_signature(text)
            if not sig or sig in done_sigs:
                return None
            return unseen({**template, text_key: text})

        def from_source(n):
            if n == 0:
                return None if listing is None else unseen({**listing["args"], page_key: listing_next})
            texts = self.label_queries() if n == 3 else self.kind_queries(n == 1)[:6]
            return next((c for c in map(as_call, texts) if c is not None), None)
        order = [0] + [1 + (turn + s) % 3 for s in range(3)]
        for n in order:
            call = from_source(n)
            if call is not None:
                return call, n == 0
        return None, False

    def _i_sweep_room(self, keep):
        if self.swept >= Knobs.SHELF_SLICES or self.moves_left() <= keep:
            return False
        return self.wire.left() >= Knobs.WIND_DOWN_S + 40

    def _i_sweep_summary(self, responses):
        rows = sorted(self.shelf_rows().items(), key=lambda kv: (kv[1][0], kv[0]))[:6]
        if not rows:
            return
        listed = "; ".join("%s '%s' %g" % (ident, str(title)[:80], price) for ident, (price, title, _) in rows)
        cid, text = responses[-1]
        responses[-1] = (cid, text + "\n[shelf sweep done] The cheapest in-stock rows seen on the shelf: " +
                         listed + ". The lowest-priced one is the pick unless a change rules it out: make it "
                         "the cart's only line, check its stock and price again, and order it.")

    def sweep_step(self) -> bool:
        needs = self.needs
        tool = self.role("search")
        if needs.priority() != "cheapest" or not needs.let_go or needs.props_left():
            return False
        if not tool or self.wire.done:
            return False
        keep = 3 + (1 if self.trolley.units else 0)
        if not self.orders_tried:
            keep = max(keep + 3, Knobs.LATE_STEPS)
        if not self._i_sweep_room(keep):
            return False
        call, listing = self.sweep_next(self.sweep_turn)
        if call is None:
            return False
        self.swept += 1
        if not listing:
            self.sweep_turn += 1
        self.sweep_tried.add(_i_key(call))
        name = "sweep_%d_%d" % (self.swept, self.wire.tick)
        proposal = dict(id=name, type="function", function=dict(name=tool, arguments=json.dumps(call)))
        self.chat.append(dict(role="assistant", tool_calls=[proposal],
                              content="No property requirement is left, so the cheapest in-stock listing anywhere on "
                                      "the shelf wins; I read one more slice of the shelf."))
        self.covering = True
        try:
            responses = self.dispatch([proposal], scripted=True)
        finally:
            self.covering = False
        more = self._i_sweep_room(keep) and self.sweep_next(self.sweep_turn)[0] is not None
        if not more and responses:
            self._i_sweep_summary(responses)
        self.post_replies(responses)
        return True

    # ================================================================== K. search advice
    def value_score(self, call_args: Any) -> int:
        query = I_ALNUM2.findall(" ".join(self.free_text(call_args)).lower())
        if self.lead_words:
            query = query[:self.lead_words]
        score = 0
        for _, value in self._i_force_reqs():
            words = I_ALNUM3.findall(value.lower())
            if any(_i_near(a, b) for a in words for b in query):
                score += 1
        return score

    def row_hits(self, rows: Any, words: Any = None) -> list:
        needs = self.needs
        asked: set = set()
        if needs.reading.get("references"):
            joined = " ".join(str(w) for w in (words or []))
            asked = set(I_ALNUM.findall(joined.lower()))
        known, read = self.archive.ids, self.archive.read
        budget = needs.spend_cap
        qty = needs.quantity()
        labels = None
        out: list = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            ident = self.row_mark(row)
            makers = _i_makers(row)
            fresh = ident in known and ident not in read
            if asked and fresh and ident not in self.ref_hits:
                if any(m and m <= asked and m <= self.pointed_words() for m in makers):
                    self.ref_hits.append(ident)
            if not fresh:
                continue
            price = _i_first_cost(row)
            if (numeric(budget) and price is not None and price * qty > budget) or self.sold_out(row):
                continue
            label = self.shelf_of(row)
            if labels is None:
                labels = self.shelf_labels()
            if labels and label and label not in labels:
                continue
            out.append(ident)
            self.hit_shelf[ident] = label
        return out

    def pointed_words(self) -> set:
        size = len(self.archive.items)
        memo = self.pointed_memo
        if memo is not None and memo[0] == size:
            return memo[1]
        words: set = set()
        try:
            for ref in self.needs.reading.get("references") or []:
                words |= set(I_ALNUM.findall(_i_low(ref)))
            read = self.archive.read

            def visit(d):
                if self.row_mark(d) in read:
                    words.update(_i_record_words(d))
            for obs in self.archive.items:
                each_dict(obs, visit)
        except Exception:  # noqa: BLE001
            pass
        self.pointed_memo = (size, words)
        return words

    def open_pages(self, most: int = 2) -> list:
        paging = self.page_arg()
        if paging is None:
            return []
        labels = {_i_low(s) for s in self.shelf_labels()}
        found = []
        for w in self.page_book.values():
            nxt = self.next_page(w)
            if nxt is None:
                continue
            if w["shelves"] and labels and not (w["shelves"] & labels):
                continue
            score = self.value_score(w["args"])
            if score > 0:
                found.append((-score, w["total"], len(found), {**w["args"], paging[0]: nxt}))
        found.sort(key=lambda f: f[:3])
        return [f[3] for f in found[:most]]

    def _i_force_reqs(self) -> list:
        """(head, value) of requirements in force whose value does not point at another listing."""
        out = []
        for req in self.needs.props_left():
            if ":" not in req:
                continue
            head, value = req.split(":", 1)
            if not points_elsewhere(value):
                out.append((head, value))
        return out

    def _i_ran_words(self, walk, width):
        ran = " ".join(self.free_text(walk["args"]))
        return I_ALNUM2.findall(ran.lower())[:width]

    def _i_probe_of(self, name, value, product, width):
        digit = I_DIGIT.search(value)
        core = value[digit.start():] if digit else value
        probe = (core.strip(" .;") + " " + " ".join(product)).strip()
        head = I_ALNUM2.findall(probe.lower())[:width]
        for w in self.page_book.values():
            ran = self._i_ran_words(w, width)
            if len(ran) == len(head) and all(_i_near_figure(a, b, True) for a, b in zip(head, ran)):
                return "", ""
        return name, probe

    def value_probe(self, raw: Any) -> tuple:
        if self.word_kind == "any":
            return "", ""
        width = self.lead_words or 2
        lead = I_ALNUM2.findall(" ".join(raw or []).lower())[:width]
        product = I_ALNUM.findall(str(self.needs.reading.get("product") or "").lower())[:3]
        reqs = []
        for head, value in self._i_force_reqs():
            first = I_COMMA_SEMI.split(_i_marks_off(value), maxsplit=1)[0]
            reqs.append((head.strip(), first.lower().strip()))
        keep = [w for _, v in reqs for w in I_ALNUM2.findall(v)] + product
        if not lead or _i_touches(lead, keep, True):
            return "", ""
        for name, value in reqs:
            if value and _i_touches(lead, I_ALNUM2.findall(name.lower()), True):
                return self._i_probe_of(name, value, product, width)
        return "", ""

    def ran_query(self, text: str) -> bool:
        width = self.lead_words or 3

        def lead(t):
            picked = I_ALNUM.findall(_i_low(t))[:width]
            return sorted(self.stem(w) for w in picked)
        want = lead(text)
        for w in self.page_book.values():
            if lead(" ".join(self.free_text(w["args"]))) == want:
                return True
        return False

    def search_template(self) -> tuple | None:
        for w in list(self.page_book.values())[::-1]:
            args = w["args"]
            if self.cg_listing(args):
                continue
            texts = self.free_text(args)
            if len(texts) != 1:
                continue
            # (any argument whose value equals the free text counts: a shelf choice equal to it makes two)
            keys = [k for k, v in args.items() if isinstance(v, str) and v in texts and v.strip()]
            if len(keys) == 1:
                lone = keys[0]
                return {k: v for k, v in args.items() if k != lone}, lone
        return None

    def _i_trap_product(self, head, items):
        if not items:
            return str(self.needs.reading.get("product") or "")
        at = self.product_of(head)
        if at is not None and at < len(items):
            return items[at]["product"]
        return items[0]["product"]

    def word_trap(self, raw: Any) -> tuple:
        try:
            kind = self.word_kind
            if kind == "any":
                return "", ""
            toks = _i_raw_tokens(raw)
            width = self.lead_words or (max(3, len(toks)) if kind == "all" else 3)
            toks = toks[:width]
            reqs = [(head.partition("(")[0].strip().lower(), head, _i_marks_off(value).lower())
                    for head, value in self._i_force_reqs()]
            vals = {w for _, _, v in reqs for w in I_ALNUM2.findall(v)}
            items = self.needs.items()
            products = [it["product"] for it in items] or [str(self.needs.reading.get("product") or "")]
            shelf_names = [str(s) for s in self.shelf_labels()]
            kinds = _i_content([t.lower() for t in products + shelf_names], I_ALNUM3)
            names = _i_content([n for n, _, _ in reqs], I_ALPHA3)
            dropped = _i_content([str(d).partition(":")[2].lower() for d in self.needs.let_go], I_ALPHA3)
            known = vals | kinds
            marks = []
            for i, t in enumerate(toks):
                why = _i_trap_reason(I_ALNUM2.findall(t), known, names, dropped)
                if why:
                    marks.append((i, why, t))
            if not marks:
                return "", ""
            _, reason, word = marks[0]
            flagged = {i for i, _, _ in marks}
            rest = [t for i, t in enumerate(toks) if i not in flagged]
            rest_parts = [p for t in rest for p in I_ALNUM2.findall(t)]
            held = [(head, v) for _, head, v in reqs if _i_touches(rest_parts, I_ALNUM2.findall(v), False)]
            if not held:
                return "", ""
            if not _i_touches(rest_parts, kinds, False):
                pname = self._i_trap_product(held[0][0], items)
                kind_words = [w for w in I_ALPHA3.findall(_i_low(pname)) if w not in FILLER]
                if kind_words:
                    rest.append(kind_words[-1])
            query = " ".join(rest[:width])
            if not query or self.ran_query(query):
                return "", ""
            if reason == "name":
                note = ("The query's word '%s' is a word of a property's NAME, not of a value. This store needs every "
                        "word, so it keeps only listings that repeat the name and misses those whose field holds just "
                        "the value." % word)
            else:
                note = ("The query's word '%s' belongs only to a requirement the shopper dropped, and it still narrows "
                        "the search." % word)
            return note + " Also search '%s' and read every page of it." % query, query
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="trap_error", error=str(error)[:300]))
            return "", ""

    def _i_live_shelves(self, lowered):
        _, live = self._i_version_sets()
        live -= set(self.refused_lines)
        if lowered:
            return {self.shelf_known(i).lower() for i in live} - {""}
        return {self.shelf_known(i) for i in live} - {""}

    def _i_short_values(self, k, several):
        vals = []
        for req in self.needs.props_left():
            if ":" not in req:
                continue
            head, _, value = req.partition(":")
            if several and self.product_of(head) not in (k, None):
                continue
            value = _i_marks_off(value).lower().strip()
            if not value or I_NOT_COMBINE.search(value):
                continue
            lead = I_COMBINE_SPLIT.split(value)[0]
            words = [w for w in I_ALPHA2.findall(lead) if w not in FILLER]
            if 1 <= len(words) <= 2:
                vals.append(words)
        return vals

    def combined_queries(self) -> list:
        template = self.search_template()
        if template is None:
            return []
        base, text_key = template
        width = self.lead_words or 3
        got = self._i_live_shelves(False)
        items = self.needs.items()
        several = bool(items)
        homes = self.home_shelves() if several else []
        out = []
        for k in (range(len(items)) if several else [None]):
            if not several:
                if not self.unverified():
                    continue
            elif k >= len(homes) or not homes[k] or {_i_low(h) for h in homes[k]} & got:
                continue
            vals = self._i_short_values(k, several)
            if len(vals) < 2:
                continue
            query = _i_pack(vals, width)
            pname = items[k]["product"] if several else str(self.needs.reading.get("product") or "")
            kinds = [w for w in I_ALPHA3.findall(_i_low(pname)) if w not in FILLER]
            if kinds and len(query) < width and kinds[-1] not in query:
                query.append(kinds[-1])
            text = " ".join(query)
            if len(query) >= 2 and not self.ran_query(text):
                out.append({**base, text_key: text})
        return out[:2]

    # ================================================================== L. decline guards
    def arbitrate(self, thing: str, ruling: str) -> str:
        try:
            answered = next((e for e in reversed(self.log)      # (the newest audit that answered)
                             if isinstance(e, dict) and e.get("kind") == "audit" and isinstance(e.get("result"), dict)),
                            None)
            reason = "" if answered is None else str(answered["result"].get("reason") or "")
            found = (" (%s)" % reason[:400]) if reason else ""
            note = ("SECOND OPINION - two readings of this listing disagree. A decline audit found it meets every "
                    "requirement%s. An earlier check found: %s. Decide from the records alone: a field that states "
                    "the required value in other words, or a shade, qualifier or spelling of the same named value, "
                    "meets it; a field that states a different value fails it."
                    % (found, str(ruling)[:500]))
            again = self.check_item(thing, "ORDER", second=note)
            if again:
                return again
            mark = "%s|%s" % (thing, self.needs.version)
            prefix = mark + "|"
            stale = [key for key in self.rulings if key == mark or str(key).startswith(prefix)]
            for key in stale:
                self.rulings[key] = ""
            return ""
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="arbitrate_error", error=str(error)[:300]))
            return ruling

    def _i_stock_seen(self):
        sold: dict = {}
        known = self.archive.ids

        def visit(d):
            mark = self.row_mark(d)
            if mark not in known:
                return
            if any(I_STOCK_WORD.search(_i_low(k)) for k in d):
                sold[mark] = self.sold_out(d)
        for obs in self.archive.items:
            each_dict(obs, visit)
        return sold

    def unrefuted(self) -> list:
        try:
            items = self.needs.items()
            several = bool(items)
            homes = self.home_shelves() if several else [self.shelf_labels()]
            if not homes or not all(homes):
                return []
            failed, live = self._i_version_sets()
            audited = self._i_reviewed() - set(self.audit_fits or ())
            sold = self._i_stock_seen()
            pointers = set(self.ref_pointers())
            budget = self.needs.spend_cap
            picks: list = []
            total = 0.0
            for at, home in enumerate(homes):
                qty = items[at]["quantity"] if several else self.needs.quantity()
                blocked = (failed, audited, self.refused_lines, picks, pointers)
                pool = []
                for ident in sorted(self.archive.read):
                    if sold.get(ident) or any(ident in group for group in blocked):
                        continue
                    if self.shelf_known(ident) not in home:      # (an unknown shelf is no home either)
                        continue
                    cost = self.archive.price(ident)
                    if not numeric(cost):
                        continue
                    if numeric(budget) and cost * qty > budget:
                        continue
                    pool.append((ident not in live, cost, ident))
                if not pool:
                    return []
                _, cost, ident = min(pool)
                picks.append(ident)
                total += cost * qty
            if numeric(budget) and total > budget:
                return []
            return picks
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="unrefuted_error", error=str(error)[:300]))
            return []

    def _i_verified_fits(self, fits, several):
        good = []
        for fit in fits[:2]:
            if fit not in self.archive.read:
                continue
            ruling = self.check_item(fit, "ORDER")
            if ruling and self.wire.left() > Knobs.WIND_DOWN_S + 30:
                ruling = self.arbitrate(fit, ruling)
            if ruling or "%s|%s" % (fit, self.needs.version) not in self.passed:
                continue
            good.append(fit)
            if not several:
                break
        return good

    def _i_uncovered_home(self):
        got = self._i_live_shelves(True)
        for home in self.home_shelves():
            labels = {_i_low(h) for h in home}
            if labels and labels.isdisjoint(got):
                return True
        return False

    def fit_hold(self) -> str:
        try:
            if self.fit_held or self.trolley.units:
                return ""
            if self.moves_left() <= 4 or self.wire.left() < Knobs.WIND_DOWN_S + 30:
                return ""
            ids, refused = self.archive.ids, self.refused_lines
            fits = [f for f in (self.audit_fits or []) if f in ids and f not in refused]
            if not fits:
                return ""
            self.fit_held = True
            several = bool(self.needs.items())
            good = self._i_verified_fits(fits, several)
            if not good or (several and self._i_uncovered_home()):
                return ""
            put = "a verified line of every product" if several else good[0]
            many = len(good) > 1
            return ("The decline audit listed %s as possible fit%s, and the independent check, reading the full "
                    "record, found %s meeting every requirement in force. Weigh that before declining: if it fits, put "
                    "%s in the cart, read its stock and price again, and order; decline only if it fails a requirement "
                    "in force." % (", ".join(good), "s" if many else "", "them" if many else "it", put))
        except Exception as error:  # noqa: BLE001
            self.log.append(dict(kind="fit_hold_error", error=str(error)[:300]))
            return ""

    # ================================================================== M. amounts in the cheapest walk
    @staticmethod
    def fig_amount(text: Any) -> tuple | None:
        low = str(text or "").lower()
        if LIMIT_WORDS.search(low):
            return None
        m = I_FIG_UNIT.match(low)
        return m.groups() if m else None

    @staticmethod
    def amount_part(text: Any) -> bool:
        low = str(text or "").lower()
        if I_DIGIT.search(low) is None:
            return False
        return LIMIT_WORDS.search(low) is not None or I_LEAD_AMOUNT.match(low) is not None

    def walk_carries(self, args: Any, k: Any, alts: bool = False) -> set:
        several = bool(self.needs.items()) and k is not None
        text = " ".join(self.free_text(args)).lower()
        words, toks = I_ALNUM2.findall(text), I_ALNUM.findall(text)
        cut = self.lead_words
        if cut:
            words, toks = words[:cut], toks[:cut]
        out: set = set()
        for i, req in enumerate(self.needs.props_left()):
            head, _, value = str(req).partition(":")
            if several and self.product_of(head) not in (k, None):
                continue
            pieces = I_ALT_CUT.split(_i_marks_off(value).lower())
            for j, alt in enumerate(pieces):
                if _i_alt_carried(alt.strip(), words, toks):
                    out.add((i, j) if alts else i)
        return out
# ------------------------------------------------------------------------------------------ walks and references
J_REF_MIN_STEPS = 8             # the reference resolver may always use at least this many steps
J_REF_DIFFER_MIN_STEPS = 6      # ... at least this many when only "differ" references are open
J_REF_PAGES = 3                 # further result pages read for one reference target
J_HOP_PICK = 4                  # candidate records shown to one hop call
J_HOP_TOKENS = 3000             # answer cap of a hop call
J_HOP_SPEND_SHARE = 0.8         # hop calls stop at this share of the spend cap
J_POINTED = ("as shown", "same as", "the page", "sold by", "redirect", "refer")
J_NOTEY = re.compile(r"note|text|desc|comment|detail|info|message|section")
J_SHELFY = re.compile(r"categor|shelf|department|aisle")
J_CLAUSE_END = re.compile(r"(?<=[.!?;])\s+")
J_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
J_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
J_NOT_SELLER = re.compile(r"title|heading|name|label|caption|categor|shelf|department|aisle|note|text|desc|comment|"
                          r"detail|info|message|section|url|image|summary|overview")


def _j_bits(text: Any, pattern: str = r"[a-z0-9]+", /) -> list:
    return re.findall(pattern, str(text).lower())


def _j_tags(text: Any, /) -> list:
    return re.findall(r"\(([^)]*)\)", str(text))


def _j_listed(holder: dict, key: str, /) -> list:
    got = holder.get(key)
    return got if isinstance(got, list) else []


def _j_mapped(holder: dict, key: str, /) -> dict:
    got = holder.get(key)
    return got if isinstance(got, dict) else {}


def _j_text(holder: dict, key: str, /) -> str:
    return str(holder.get(key) or "").strip()


def _j_sig(value: Any, /) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _j_pointing(value: str, /) -> bool:
    return points_elsewhere(value) or DIFFERING.search(value.lower()) is not None


def _j_named(name: str, /) -> Any:
    return re.compile(r"(?<![a-z0-9])%s(?:'s)?(?![a-z0-9])" % re.escape(name.lower()))


def _j_shown(details: Any, pool: set, /) -> int:
    n = 0
    for d in details:
        toks = set(ref_tokens(d))
        n += 1 if toks and toks <= pool else 0
    return n


class RefTrail(dict):
    """One requirement whose value another listing gives, and how far the code got in following it. Its fields read
    and write both as keys and as attributes."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class WalksMixin:
    """Code-driven reads under a cheapest rule (further pages of id-ordered searches, the pick's product line, ceiling
    walks, the cheapest unread rows), the reads before a decline, requirements given by pointing at another listing
    (found, followed through notes, resolved into a value), and what a search result teaches about the price filter."""

    def _walks_state(self) -> None:
        self.walked: set = set()
        self.decline_read_done = False
        self.decline_queue = None
        self.ref_steps = 0
        self.ref_calls = 0
        self.ref_ids: set = set()
        self.ref_memo = None
        self.ref_seen_at: dict = {}
        self.seller_memo = None
        self.store_keys_memo = None
        self.ref_wait_said = None
        self.ref_told: list = []
        self.own_told: set = set()
        self.ceil_seen: dict = {}
        self.page_cap = None
        self.ceil_bad: set = set()
        self.ceil_mode = None

    # ------------------------------------------------------------------------------------------- small shared pieces
    @staticmethod
    def _wk_scalar(value: Any, /) -> bool:
        return isinstance(value, str) or numeric(value)

    @staticmethod
    def _wk_tag(head: Any, /) -> str:
        found = re.findall(r"\(([^)]*)\)", str(head or ""))
        return found[-1].strip() if found else ""

    @staticmethod
    def _wk_name(req: Any, /) -> str:
        return str(req or "").split(":", 1)[0].split("(", 1)[0].strip()

    @staticmethod
    def _wk_value(req: Any, /) -> str:
        text = str(req or "")
        return text.split(":", 1)[1] if ":" in text else ""

    @staticmethod
    def _wk_num(x: Any, /) -> str:
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x)

    @staticmethod
    def _wk_camel(name: Any, /) -> str:
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name or ""))

    @staticmethod
    def _wk_fill(pools: list, most: int) -> list:
        """Round-robin over the pools until `most` ids are queued (ids already queued are skipped)."""
        rest = [[*p] for p in pools]
        queue: list = []
        while len(queue) < most and any(rest):
            for pool in rest:
                del pool[:next((n for n, x in enumerate(pool) if x not in queue), len(pool))]
                if pool and len(queue) < most:
                    queue.append(pool.pop(0))
        return queue

    def _wk_trim(self, queue: list, floor: int, batch: Any, /) -> list:
        queue = list(queue)
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        return queue

    def _wk_live_ids(self) -> set:
        ver = str(self.needs.version)
        out = set()
        for key in self.passed:
            ident, _, v = str(key).rpartition("|")
            if ident and v == ver:
                out.add(ident)
        return out

    def _wk_units(self, k: Any, /) -> Any:
        items = self.needs.items()
        if isinstance(k, int) and 0 <= k < len(items):
            return items[k]["quantity"]
        return self.needs.quantity()

    def _wk_products(self) -> list:
        items = self.needs.items()
        if items:
            return [it["product"] for it in items]
        return [str(self.needs.reading.get("product") or "")]

    def _wk_query(self, args: Any, /) -> str:
        return " ".join(str(t) for t in self.free_text(args if isinstance(args, dict) else {}))

    def _wk_book_key(self, args: dict, pkey: Any, /) -> str:
        """The page-book key of the walk these arguments started (the paging argument left out)."""
        return json.dumps({a: v for a, v in args.items() if a != pkey}, sort_keys=True, default=str)

    def _wk_live(self) -> bool:
        return not self.wire.done and not self.wire.broke

    def _wk_rows_near(self, observation: Any, /) -> list:
        """Listing rows within 3 levels of an observation: 4+ keys and a known id."""
        rows: list = []

        def visit(d: dict) -> None:
            mark = self.row_mark(d)
            if len(d) >= 4 and isinstance(mark, str) and mark in self.archive.ids:
                rows.append(d)
        each_dict(observation, visit, 3)
        return rows

    def _wk_field_tokens(self, row: dict, /) -> set:
        """Field-name tokens of a record plus the tokens of its title, category and property-map values."""
        names, _ = self.ref_parts(row)
        bits = [self.title_of(row), self.shelf_of(row)]
        for v in row.values():
            if isinstance(v, dict):
                bits += [str(x) for x in v.values() if isinstance(x, (str, int, float))]
        return set(names) | set(ref_tokens(" | ".join(bits)))

    def _wk_log(self, kind: str, /, **fields: Any) -> None:
        entry = {"kind": kind}
        entry.update(fields)
        self.log.append(entry)

    def _wk_guard(self, kind: str, fallback: Any, body: Any, /, *args: Any) -> Any:
        """body(*args), or the fallback when it raises (logged under `kind` when one is given)."""
        try:
            return body(*args)
        except Exception as error:  # noqa: BLE001
            if kind:
                self._wk_log(kind, error=str(error)[:200])
            return fallback

    def _wk_give_up(self, record: Any, pick: Any, why: str, /) -> bool:
        record.judged.update(pick)
        self.ref_stop(record, why)
        return True

    @staticmethod
    def _wk_head(req: Any, /) -> str:
        return str(req).split(":", 1)[0]

    # ------------------------------------------------------------------------------------------- the cheapest walk
    def walk_score(self, args: dict, k: Any) -> int:
        return len(self.walk_carries(args, k))

    def walk_options(self, k: Any) -> list:
        if not (home := set(self.walk_home(k))):
            return []
        cut = self.lead_words or 0

        def lead(args: dict) -> set:
            toks = _j_bits(self._wk_query(args))
            return {self.stem(t) for t in (toks[:cut] if cut else toks)}
        several = bool(self.needs.items()) and k is not None
        texts = [*map(str, home)] + [self._wk_value(q) for q in self.needs.props_left()
                                     if not (several and self._wk_foreign_req(self._wk_head(q), k))]
        free = {self.stem(w) for t in texts for w in _j_bits(t) if len(w) >= 2}
        finished, cands = [], []
        for key, walk in self.page_book.items():
            if walk.get("id_order") is not True or not walk.get("first"):
                continue
            words = lead(args := walk.get("args") or {})
            shelf_rows = walk.get("row_shelf") or {}
            if self.next_page(walk) is not None:
                if not home.isdisjoint(shelf_rows.values()) and self.walk_carries(args, k):
                    cands.append((key, walk, words, args))
            elif words and len(shelf_rows) >= (walk.get("total") or 0):
                finished.append((words, self.walk_carries(args, k, alts=True), self.vl_mode(args)))
        kept = []
        for key, walk, words, args in cands:
            mode = self.vl_mode(args)
            pairs = None
            covered = False
            for done_words, done_pairs, done_mode in finished:
                if done_mode != mode:
                    continue
                if done_words <= words:
                    covered = True
                    break
                if done_words <= free:
                    if pairs is None:
                        pairs = self.walk_carries(args, k, alts=True)
                    if pairs <= done_pairs:
                        covered = True
                        break
            if covered:
                continue
            size = max(1, int(walk.get("size") or 0))
            left = -(-int(walk.get("total") or 0) // size) - len(walk.get("read") or ())
            if 0 < left <= 2 * Knobs.ID_WALK_EACH:
                kept.append((left, key, walk, words))
        twins: dict = {}
        for item in kept:
            mark = (item[2].get("total"), tuple(item[2].get("first") or ()))
            if mark not in twins or item[0] < twins[mark][0]:
                twins[mark] = item
        opts = [*twins.values()]
        if self.all_words() and len(opts) > 1:
            gone = set()
            for o in opts:
                b = o[3]
                for p in opts:
                    if p is o:
                        continue
                    a = p[3]
                    shorter_holds = a < b and not ((b - a) <= free) and p[0] <= Knobs.ID_WALK_EACH
                    longer_loses_nothing = b < a and (a - b) <= free
                    if shorter_holds or longer_loses_nothing:
                        gone.add(o[1])
                        break
            opts = [o for o in opts if o[1] not in gone]
        return [o[:3] for o in sorted(opts, key=lambda o: (o[0], -self.walk_score(o[2].get("args") or {}, k), o[1]))]

    def walk_plan(self, budget: int, only: Any = None) -> list:
        if not (budget > 0 and self.needs.priority() == "cheapest" and self.role("search") and self.page_arg()
                and self.id_store()):
            return []
        wanted = []
        for k in range(len(self.needs.items())) or [None]:
            if k in self.walked or (only is not None and k not in only):
                continue
            if opts := self.walk_options(k):
                wanted.append((opts[0][0], -1 if k is None else k, k, opts[0][1]))
        wanted.sort(key=lambda w: w[:2])
        plan = []
        for pages, _, k, key in wanted:
            if (take := min(pages, Knobs.ID_WALK_EACH, budget)) <= 0:
                break
            plan.append((k, key, take))
            budget -= take
        return plan

    def walk_pick(self, k: Any) -> Any:
        dead = self.dead_lines()

        def mine(i: Any) -> bool:
            return i not in dead and (k is None or self.product_index(i) == k)
        lines = [i for i in self.trolley.units if mine(i)] \
            or [i for i in self._wk_live_ids() if i not in self.refused_lines and mine(i)]
        priced = [(p, i) for i in lines if numeric(p := self.archive.price(i))]
        if not priced:
            return None
        low = min(priced)
        return low[1], float(low[0])

    def line_query(self, k: Any) -> Any:
        if (pick := self.walk_pick(k)) is None or (template := self.search_template()) is None:
            return None
        hits: list = []
        for obs in self.archive.items:
            each_dict(obs, lambda d: hits.append(d) if self.row_mark(d) == pick[0] and self.title_of(d) else None)
        if not hits:
            return None
        rec = hits[-1]
        maker = ""
        for key, val in rec.items():
            if isinstance(val, str) and val.strip() and "id" not in (low := str(key).lower()) \
                    and re.search(r"brand|make|manufactur", low):
                maker = val
                break
        if not (maker_words := _j_bits(maker)):
            return self.title_line(k, rec, template)
        avoid = set(maker_words) | FILLER
        for text in [*self.walk_home(k), *self.needs.props_left(), *self._wk_products()]:
            avoid.update(_j_bits(text, r"[a-z]+"))
        words = _j_bits(self.title_of(rec), r"[a-z][a-z'\-]{3,}")
        if maker_words[0] in words:
            del words[:words.index(maker_words[0]) + 1]
        lead = next((w for w in words if "'" not in w and w.strip("'-") not in avoid), None)
        if lead is None:
            return None
        text = "%s %s" % (maker_words[0], lead)
        return None if self.ran_query(text) else {**template[0], template[1]: text}

    def walk_rows(self, k: Any, keys: Any, most: int) -> list:
        home = set(self.walk_home(k))
        carried = {key: len(self.walk_carries(walk.get("args") or {}, k)) for key, walk in self.page_book.items()
                   if walk.get("id_order") is True}
        top = max(carried.values(), default=0)
        rows: set = set()
        for key, walk in self.page_book.items():
            if 0 < carried.get(key, 0) == top or key in keys:
                rows.update(i for i, lab in (walk.get("row_shelf") or {}).items() if lab in home)
        if (pick := self.walk_pick(k)) is not None:
            limit = pick[1]
        else:
            read_prices = sorted((p, r) for r in rows if r in self.archive.read
                                 for p in [self.archive.price(r)] if numeric(p))
            limit = read_prices[0][0] if read_prices else None
        qty = self._wk_units(k)
        cap = self.needs.spend_cap
        sold: dict = {}

        def visit(d: dict) -> None:
            if any("id" in str(key).lower() and isinstance(v, str) and v in rows for key, v in d.items()):
                sold[self.row_mark(d)] = self.sold_out(d)
        for obs in self.archive.items:
            each_dict(obs, visit)
        out = []
        for ident in rows:
            if self._wk_skip_row(ident, sold):
                continue
            price = self.archive.price(ident)
            if not self._wk_over(price, qty, cap) and (limit is None or not price >= limit):
                out.append((price, ident))
        return [ident for _, ident in sorted(out)[:most]]

    def _wk_skip_row(self, ident: Any, sold: dict, /) -> Any:
        return ident in self.archive.read or ident in self.refused_lines or sold.get(ident)

    @staticmethod
    def _wk_over(price: Any, units: Any, cap: Any, /) -> Any:
        return not numeric(price) or (numeric(cap) and price * units > cap)

    def page_digest(self, observation: Any) -> str:
        rows: list = []

        def visit(d: dict) -> None:
            if len(d) > 3 and any(isinstance(v, str) and v in self.archive.ids and "id" in str(key).lower()
                                  for key, v in d.items()):
                rows.append(d)
        each_dict(observation, visit, 3)
        bits = []
        for row in rows[:12]:
            price = next((v for key, v in row.items() if numeric(v) and re.search(r"price|cost", str(key).lower())), None)
            bits.append("%s %s%s '%s' [%s]" % (self.row_mark(row), "?" if price is None else price,
                                               " SOLD OUT" if self.sold_out(row) else "", self.title_of(row)[:70],
                                               self.shelf_of(row)))
        return "rows on this page (id, price, title, category): %s" % ("; ".join(bits) or "none")

    def walk_call(self, args: dict, say: str) -> bool:
        cid = "walk_%d_%d" % (len(self.chat), self.wire.tick)
        call = dict(id=cid, type="function", function=dict(name=self.role("search"), arguments=json.dumps(args)))
        self.chat.append(dict(role="assistant", content=say, tool_calls=[call]))
        before = len(self.archive.items)
        self.covering = True
        try:
            replies = self.dispatch([call], scripted=True)
        finally:
            self.covering = False
        if not replies:
            self._wk_tool_reply(cid, "Not run.")
            return False
        reply_id, raw = replies[0]
        if grew := len(self.archive.items) > before:
            raw = self.page_digest(self.archive.items[-1]) + self._wk_kept_note(raw)
        self._wk_tool_reply(reply_id, raw)
        return grew

    def _wk_tool_reply(self, call_id: Any, content: str, /) -> None:
        self.chat.append(dict(role="tool", tool_call_id=call_id, content=content))

    @staticmethod
    def _wk_kept_note(raw: str, /) -> str:
        _, marker, note = raw.partition("\n[note] ")
        if marker and note and re.search(r"shopper|Meaning:|REFUSED", note):
            return "\n[note] " + note
        return ""

    def _wk_tool_note(self, text: str, status: bool, /) -> None:
        last = self.chat[-1] if self.chat else None
        if isinstance(last, dict) and last.get("role") == "tool":
            last["content"] = str(last.get("content") or "") + text + ("\n" + self.status_line() if status else "")

    @staticmethod
    def _wk_next_args(walk: dict, pkey: Any, nxt: Any, /) -> dict:
        return {**(walk.get("args") or {}), pkey: nxt}

    def _wk_queue_opens(self, queue: list, say: str, purpose: str, floor: int, recheck: bool, /) -> None:
        self.open_queue = queue
        if recheck:
            self.open_recheck = True
        self.open_say = say
        self.open_purpose = purpose
        self.open_floor = floor
        self.open_named()

    def run_walk(self, plan: list, floor: int, ceil: Any = None, repair: bool = False) -> bool:
        if not (plan or ceil) or not (found := self.page_arg()) or not self.role("search"):
            return False
        late = Knobs.WIND_DOWN_S + 30
        reports: list = []
        pages = 0
        for k, key, n in plan:
            self.walked.add(k)
            if (walk := self.page_book.get(key)) is not None:
                pages += self._wk_id_pages(key, walk, n, found[0], floor, late, reports)
        plan_ks = [k for k, _, _ in plan]
        line_keys = self._wk_line_walks(plan_ks, found[0], floor, late, reports)
        ceil_rows: dict = {}
        for k in dict.fromkeys(ceil or []):
            stop_at = floor if repair else max(floor, self.order_due() + 2)
            if self.wire.done or self.moves_left() <= stop_at + 1 or self.wire.left() < late:
                break
            if got := self.ceiling_walk(k, floor, repair):
                reports.append("%s" % (got[0],))
                ceil_rows[k] = [*(got[1] or [])]
        if not (pages or line_keys or ceil_rows):
            return False
        # (a ceiling walk that ran widens the opens even when it listed no row)
        most = (Knobs.CAP_ROWS if ceil_rows else 2) if not (batch := self.batch_tool()) else batch[2]
        pools = [self.walk_rows(k, [line_keys.get(k, "")], most) for k in plan_ks]
        queue = self._wk_trim(self._wk_fill(pools + [*ceil_rows.values()], most), floor, batch)
        homes = set().union(*map(self.walk_home, plan_ks + [*ceil_rows]))
        self.rank_checked = set(filter(lambda m: self.shelf_known(str(m).partition("|")[0]) not in homes, self.rank_checked))
        self.order_vetted = False
        if queue:
            tail = "the cheapest unread rows below the pick are opened now: " + ", ".join(
                "%s (%s)" % (i, self._wk_num(self.archive.price(i))) for i in queue)
        else:
            tail = "no unread row in stock and within the budget, priced below the pick, was shown"
        summary = ("[cheapest walk] " + "; ".join(reports) + ". " + tail + ". For each product, choose the lowest-priced "
                   "listing whose own fields meet every requirement in force; the rank check runs again at the cart "
                   "call and at the order.")
        if queue:
            self._wk_queue_opens(queue, "The cheapest fitting item is asked for, so I open the cheapest unread rows the "
                                        "walk showed.", "walk", floor, True)
        self._wk_tool_note("\n" + summary, not queue)
        self._wk_log("walk", plan=[[*p[::2]] for p in plan], ceiling=[*ceil_rows], opened=queue)
        return True

    def _wk_id_pages(self, key: str, walk: dict, n: int, pkey: Any, floor: int, late: float, reports: list, /) -> int:
        query = self._wk_query(walk.get("args") or {})
        count = 0
        while count < n:
            walk = self.page_book.get(key) or walk
            nxt = self.next_page(walk)
            if nxt is None or self.moves_left() <= floor + 1 or not self._wk_live() or self.wire.left() < late:
                break
            say = ("The cheapest fitting item is asked for, and this store lists search results in a fixed id order "
                   "rather than by price, so I read the next page of '%s'." % query)
            if not self.walk_call(self._wk_next_args(walk, pkey, nxt), say):
                break
            count += 1
        walk = self.page_book.get(key) or walk
        reports.append("'%s': %d more page(s) read, %d of its %s matches seen"
                       % (query, count, len(walk.get("row_shelf") or {}), walk.get("total") or 0))
        return count

    def _wk_line_walks(self, plan_ks: list, pkey: Any, floor: int, late: float, reports: list, /) -> dict:
        line_keys: dict = {}
        for k in plan_ks:
            if self.moves_left() <= floor + 2 or self.wire.done or self.wire.left() < late:
                break
            if not (args := self.line_query(k)):
                continue
            text = self._wk_query(args)
            say = ("A product line's other sizes and colours are separate listings that may cost less, so I search the "
                   "pick's line '%s'." % text)
            if not self.walk_call(args, say):
                continue
            lkey = line_keys[k] = self._wk_book_key(args, pkey)
            extra = 0
            while extra < 2 and (walk := self.page_book.get(lkey)) is not None:
                size = max(1, int(walk.get("size") or 0))
                nxt = self.next_page(walk)
                if (walk.get("total") or 0) > 3 * size or nxt is None or self.moves_left() <= floor + 3 \
                        or self.wire.done or self.wire.left() <= late:
                    break
                if not self.walk_call(self._wk_next_args(walk, pkey, nxt),
                                      "I read the next page of the pick's line '%s'." % text):
                    break
                extra += 1
            reports.append("the pick's line '%s'" % text + ((" (%d more page(s))" % extra) if extra else ""))
        return line_keys

    # ------------------------------------------------------------------------------------------- reads before a decline
    def decline_fits(self) -> set:
        ver = str(self.needs.version)
        failed = set()
        for key, ruling in self.rulings.items():
            parts = str(key).split("|")
            if ruling and len(parts) > 1 and parts[1] == ver:
                failed.add(parts[0])
        audit = set(self.audit_fits or [])
        return (self._wk_live_ids() | (audit - failed)) - set(self.refused_lines)

    def decline_lacking(self) -> list:
        got = {self.shelf_known(i) for i in self.decline_fits()}
        got.discard("")
        if self.needs.items():
            return [k for k, home in enumerate(self.home_shelves()) if home and got.isdisjoint(home)]
        return [None] if got.isdisjoint(self.shelf_labels()) else []

    def decline_walks(self) -> list:
        reqs = self.needs.props_left()

        def stated(args: dict, k: Any) -> int:
            return sum(1 for i in self.walk_carries(args, k) if isinstance(i, int) and 0 <= i < len(reqs)
                       and not any(s in str(reqs[i]).lower() for s in J_POINTED))
        chosen: list = []
        for k in self.decline_lacking():
            opts = []
            for pages, key, walk in self.walk_options(k):
                args = walk.get("args") or {}
                if (n := stated(args, k)) > 0:
                    opts.append((-n, -self.walk_score(args, k), pages, key))
            if fresh := [o[3] for o in sorted(opts) if o[3] not in chosen]:
                chosen.append(fresh[0])
        return chosen[:2]

    def row_pools(self, homes: list, ids: Any, below: Any = None) -> list:
        shelf: dict = {}
        sold: dict = {}

        def visit(d: dict) -> None:
            if not isinstance(mark := self.row_mark(d), str) or mark not in self.archive.ids \
                    or not (lab := self.shelf_of(d)):
                return
            shelf[mark] = lab
            if any(w in str(key).lower() for key in d for w in ("stock", "avail")):
                sold[mark] = self.sold_out(d)
        self._wk_visit_all(visit)
        cap = self.needs.spend_cap
        out = []
        for n, (cats, units) in enumerate(homes):
            cats = {*cats}
            pool = sorted((p, i) for i, lab in shelf.items()
                          if lab in cats and not self._wk_skip_row(i, sold) and (ids is None or i in ids)
                          and not self._wk_over(p := self.archive.price(i), units, cap)
                          and (below is None or p < below[n]))
            out.append([i for _, i in pool])
        return out

    def over_budget_rows(self) -> list:
        return self._wk_guard("over_budget_error", [], self._wk_over_budget)

    def _wk_over_budget(self) -> list:
        """When the cheapest known fits of all products together cost more than the budget: per product (largest share
        first) the cheaper unread rows of its shelves."""
        if not (items := self.needs.items()) or self.needs.priority() == "cheapest" \
                or not numeric(cap := self.needs.spend_cap):
            return []
        homes = self.home_shelves()
        fits = self.decline_fits()
        total = 0.0
        got = []
        for k, home in enumerate(homes):
            priced = [p for i in fits if numeric(p := self.archive.price(i)) and self.shelf_known(i) in home]
            if not home or not priced:
                return []
            low = min(priced)
            units = items[k]["quantity"]
            share = low * units
            total += share
            got.append(((home, units), low, share))
        if total <= cap:
            return []
        pools = self.row_pools([g[0] for g in got], None, [g[1] for g in got])
        ranked = [p for p, _ in sorted(zip(pools, [g[2] for g in got]), key=lambda ps: -ps[1])]
        return ranked if any(ranked) else []

    def decline_reads(self) -> str:
        return self._wk_guard("decline_reads_error", "", self._wk_decline_plan)

    def _wk_decline_plan(self) -> str:
        if self.decline_read_done or self.wire.left() < Knobs.WIND_DOWN_S + 60:
            return ""
        if not ((batch := self.batch_tool()) or self.role("details")) or self.trolley.units:
            return ""
        spare = self.moves_left() - (1 if batch else 2) - 6
        room = min(3, spare)
        keys = self.decline_walks() if room > 0 and self.page_arg() and self.role("search") and self.id_store() else []
        cheaper = self.over_budget_rows() if spare > 0 else []
        if not (keys or cheaper):
            return ""
        self.decline_read_done, self.decline_queue = True, (keys, room, 6)
        said = []
        if keys:
            quoted = ", ".join("'%s'" % self._wk_query((self.page_book.get(key) or {}).get("args")) for key in keys)
            said.append(" searches carrying required values (%s) matched more listings than were read, and this "
                        "store lists results in a fixed id order, so fits sit on unread pages as often as on read "
                        "ones; the code reads up to %d of those pages and opens the cheapest rows in stock and within "
                        "the budget" % (quoted, room))
        if cheaper:
            said.append(" the listings found for all the products together cost more than the budget, so the "
                        "cheapest unread in-stock rows below each product's pick are opened as well")
        return ("Before a decline,%s. Then judge those records against the requirements in force by meaning, buy what "
                "fits (one line per product at its quantity, or the one item), and decline only if each of them "
                "fails." % ";".join(said))

    def run_decline_reads(self) -> bool:
        (keys, budget, floor), self.decline_queue = self.decline_queue, None
        opens = 1 if (batch := self.batch_tool()) else 2
        found = self.page_arg()
        read = 0
        walked_keys: list = []
        for key in keys if found else []:
            if (walk := self.page_book.get(key)) is None:
                continue
            walked_keys.append(key)
            query = self._wk_query(walk.get("args") or {})
            while read < budget and self.moves_left() > floor + opens and self._wk_live() \
                    and self.wire.left() > Knobs.WIND_DOWN_S + 30:
                walk = self.page_book.get(key) or walk
                if (nxt := self.next_page(walk)) is None or not self.walk_call(
                        self._wk_next_args(walk, found[0], nxt),
                        "Before declining, I read the next page of '%s': fitting listings may sit on unread pages." % query):
                    break
                read += 1
        seen: set = set()
        for key in walked_keys:
            seen.update((self.page_book.get(key) or {}).get("row_shelf") or {})
        homes = [(self.walk_home(k), self._wk_units(k)) for k in self.decline_lacking()]
        pools = (self.row_pools(homes, seen) if walked_keys else []) + self.over_budget_rows()
        queue = self._wk_trim(self._wk_fill(pools, batch[2] if batch else 2), floor, batch)
        self._wk_log("decline_reads", pages=read, opened=queue)
        if queue and not self.wire.done:
            self._wk_queue_opens(queue, "Before declining, I open the cheapest unread rows within the budget.", "decline",
                                 floor, False)
        else:
            self._wk_tool_note("\n[before declining] no unread row in stock and within the budget was shown there.", True)
        return read > 0 or bool(queue)

    # ------------------------------------------------------------------------------------------- references: opening
    def ref_register(self) -> None:
        self._wk_guard("ref_register_error", None, self._wk_register_written)

    def _wk_register_written(self) -> None:
        reading = self.needs.reading
        infos = _j_listed(reading, "reference_targets")
        listed = [str(r) for r in _j_listed(reading, "references")]
        for req in list(self.needs.written):
            value = self._wk_value(req)
            if points_elsewhere(value) or (req in listed and DIFFERING.search(value.lower())):
                self.ref_add(req, infos, self.request, "request")

    def ref_untracked(self, refs: list, tracked: list) -> list:
        def settled(r: Any) -> bool:
            return any(self.ref_match(t, r) and (t.status == "resolved" or t.chain) for t in tracked)
        return self._wk_guard("", refs, lambda: [r for r in refs if not settled(r)])

    def ref_take(self) -> None:
        self._wk_guard("ref_take_error", None, self._wk_take_news)

    def _wk_take_news(self) -> None:
        for rec in (r for r in self.refs if r.status == "resolved" and r.now):
            if any(q != rec.now and self.ref_match(rec, q) for q in self.needs.props_left()):
                self.ref_swap(rec, rec.now)
        news, self.needs.ref_news = list(self.needs.ref_news or []), []
        for item in news:
            if not isinstance(item, dict):
                continue
            if (rec := self.ref_add(item.get("need"), list(item.get("infos") or []), item.get("line") or "", "reply")) is None:
                continue
            listed = self.needs.reading.get("references")
            if not isinstance(listed, list):
                listed = self.needs.reading["references"] = []
            if rec.req not in listed:
                listed.append(rec.req)

    def ref_add(self, req: Any, infos: Any, source: Any, origin: str) -> Any:
        req = str(req or "")
        head = req.split(":", 1)[0].strip()
        value = self._wk_value(req)
        prop = self._wk_name(req)
        tag = self._wk_tag(head)
        if not prop or not_a_property(prop):
            return None
        if any(r.req == req or (loosely_same(r.prop, prop) and kind_set(r.tag) == kind_set(tag)) for r in self.refs):
            return None
        info = self._wk_ref_info(infos, prop, tag)
        if (target := self.ref_target(info, value, source, prop)) is None:
            self._wk_log("ref_skip", req=req)
            return None
        relation, tol, unit = self._wk_ref_relation(value, info or {}, after_prop(prop, source).lower(), target)
        first = dict(target, details=list(target["details"]))
        rec = RefTrail(req=req, head=head, prop=prop, tag=tag, relation=relation, tol=tol, unit=unit, target=target,
                       first=first, chain=[], seen={self.ref_key(target)}, judged=set())
        rec.update(queries=[], pages=0, hops=0, tries=0, fails=0, status="open", value="", goes_with=[], now="",
                   origin=origin, unindexed=False, why="", shows={}, loose=[], noted=set(), late_from=None)
        self.refs.append(rec)
        self._wk_log("ref_open", req=req, relation=relation, target=target, origin=origin)
        return rec

    def _wk_ref_info(self, infos: Any, prop: str, tag: str, /) -> Any:
        several = bool(self.needs.items())
        for entry in (infos if isinstance(infos, list) else []):
            if isinstance(entry, dict) and loosely_same(str(entry.get("property") or ""), prop):
                owner = entry.get("product")
                if not (tag and owner and several) or kind_set(owner) & kind_set(tag):
                    return entry
        return None

    @staticmethod
    def _wk_ref_relation(value: str, facts: dict, span: str, target: dict, /) -> tuple:
        """The relation a pointing value states; a plain "same" turns "within" (the listener's tolerance) or "differ"
        (the listener's word, or a differ word before the seller's name in the request)."""
        relation, tol, unit = ref_relation(value)
        told = str(facts.get("relation") or "").lower()
        seller = str(target["seller"] or "").lower()
        if relation != "same":
            return relation, tol, unit
        if told == "within" and numeric(facts.get("tolerance")):
            return "within", float(facts["tolerance"]), str(facts.get("unit") or "")
        cut = span.find(seller) if seller else -1
        if any(w in told for w in ("differ", "not the same", "exclu", "apart", "other than", "unlike")) \
                or (cut > 0 and DIFFERING.search(span[:cut])):
            return "differ", None, ""
        return relation, tol, unit

    def ref_target(self, info: Any, value: Any, source: Any, prop: Any) -> Any:
        info = info if isinstance(info, dict) else {}
        own_seller = _j_text(info, "seller")
        own_kind = _j_text(info, "kind")
        own_details = [str(d).strip() for d in _j_listed(info, "details") if self._wk_scalar(d) and str(d).strip()]

        def ok(name: Any) -> bool:
            return self.ref_seller_ok(name, source)
        parsed = next((got for text in (after_prop(prop, source), value)
                       if (got := ref_description(text, ok)) and ok(got["seller"])), None)
        if ok(own_seller):
            seller, details = own_seller, own_details
            if not details and parsed is not None and seller_key(parsed["seller"]) == seller_key(seller):
                details = list(parsed["details"])
        elif parsed is None:
            return None
        else:
            seller, details = parsed["seller"], own_details or list(parsed["details"])
        source_toks = set(ref_tokens(source))
        kept = [d for d in dict.fromkeys(details) if not source_toks.isdisjoint(ref_tokens(d))]
        kind = own_kind or (parsed["kind"] if parsed is not None else own_kind)
        return dict(seller=seller, kind=str(kind or "")[:80], details=kept[:6])

    def ref_seller_ok(self, seller: Any, source: Any) -> bool:
        name = str(seller or "").strip()
        if len(re.findall(r"[A-Za-z]", name)) < 3 or not _j_named(name).search(str(source or "").lower()):
            return False
        kinds = set().union(*(kind_set(p) for p in self._wk_products()), *(kind_set(c) for c in self._wk_enum_choices()))
        own = kind_set(name)
        return bool(own) and not own <= kinds

    def _wk_enum_choices(self) -> list:
        schema = self._wk_search_schema()
        props = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(props, dict):
            return []
        return [c for spec in props.values() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)
                for c in spec["enum"]]

    # ------------------------------------------------------------------------------------------- identity and wording
    @staticmethod
    def ref_key(target: dict) -> tuple:
        toks: set = set()
        for d in target.get("details") or []:
            toks.update(ref_tokens(d))
        return seller_key(target.get("seller")), tuple(sorted(toks))

    @staticmethod
    def ref_desc_of(target: dict) -> str:
        text = "the %s sold by %s" % (_j_text(target, "kind") or "listing", target.get("seller"))
        if details := target.get("details") or []:
            text += " (%s)" % "; ".join(str(d) for d in details)
        return text

    def ref_desc(self, record: Any) -> str:
        return self.ref_desc_of(record.target)

    # ------------------------------------------------------------------------------------------- reading the records
    def ref_index(self) -> dict:
        size = len(self.archive.items)
        if (memo := self.ref_memo) is not None and memo[0] == size:
            return memo[1]
        rows: dict = {}
        seen_at: dict = {}
        ids = self.archive.ids
        at = [0]

        def visit(d: dict) -> None:
            if not isinstance(mark := self.row_mark(d), str) or mark not in ids:
                return
            if len(d) >= 4 or any(isinstance(v, dict) and len(v) >= 2 for v in d.values()):
                rows.setdefault(mark, {}).update(d)
                seen_at[mark] = at[0]
        for at[0], obs in enumerate(self.archive.items):
            each_dict(obs, visit)
        self.ref_memo = (size, rows)
        self.ref_seen_at = seen_at
        return rows

    def ref_sellers(self, row: dict) -> set:
        out = {seller_key(v) for v in self._wk_seller_values(row)}
        out.discard("")
        return out

    def _wk_seller_values(self, row: dict, /) -> list:
        learned = self.seller_keys()
        out = []
        for key, v in row.items():
            if not isinstance(v, str) or not v.strip() or len(v) > 60:
                continue
            low_key = str(key).lower()
            if "id" not in low_key and not J_NOTEY.search(low_key) and (SELLER_MARK.search(low_key) or key in learned):
                out.append(v)
        return out

    def seller_keys(self) -> set:
        return self._wk_guard("", set(), self._wk_seller_fields)

    def _wk_seller_fields(self) -> set:
        """Field names that hold a seller: ones holding a reference's seller, or a short capitalised name the notes
        write that no property map holds (memoised by archive size and the sellers followed)."""
        names = {seller_key(r.target.get("seller")) for r in self.refs} | {seller_key(r.first.get("seller")) for r in self.refs}
        names.discard("")
        mark = (len(self.archive.items), tuple(sorted(names)))
        if self.seller_memo is not None and self.seller_memo[0] == mark:
            return self.seller_memo[1]
        rows = self.ref_index()
        vocab = {seller_key(x) for row in rows.values() for v in row.values() if isinstance(v, dict)
                 for x in v.values() if isinstance(x, str)}
        noted = {n for n in self._wk_name_runs(rows) if len(n) >= 4 and n not in vocab}
        keys = set()
        for row in rows.values():
            keys.update(key for key, v in row.items() if self._wk_seller_field(key, v, names, noted))
        self.seller_memo = (mark, keys)
        return keys

    def _wk_name_runs(self, rows: dict, /) -> set:
        noted: set = set()
        for row in rows.values():
            for sentence in J_CLAUSE_END.split(self.ref_texts(row, desc=False)):
                run: list = []
                for w in J_WORD.findall(sentence)[1:] + [None]:
                    if w is not None and w[0].isupper() and any(c.islower() for c in w):
                        run.append(w)
                        continue
                    noted.update(seller_key(" ".join(run[i:j])) for i in range(len(run))
                                 for j in range(i + 1, min(i + 3, len(run)) + 1))
                    run = []
        return noted

    @staticmethod
    def _wk_seller_field(key: Any, v: Any, names: set, noted: set, /) -> bool:
        if not isinstance(v, str) or not 1 <= len(v) <= 60:
            return False
        low_key = str(key).lower()
        if "id" in low_key or J_NOT_SELLER.search(low_key):
            return False
        reduced = seller_key(v)
        plain = v.strip()
        return reduced in names or (plain[:1].isupper() and len(plain.split()) <= 3 and reduced in noted)

    @staticmethod
    def ref_texts(row: dict, desc: bool = True) -> str:
        bins: dict = {True: [], False: []}
        for key, v in row.items():
            if isinstance(v, str) and len(v.split()) > 2 and not (low_key := str(key).lower()).endswith("id") \
                    and not re.search(r"title|heading|headline|name|label|caption|categor|shelf|department|url|image", low_key):
                bins[bool(re.search(r"desc|overview|summary|about", low_key))].append(v)
        return "\n".join(bins[False] or (bins[True] if desc else []))

    def ref_parts(self, row: dict) -> tuple:
        notes = re.sub(r"\b(?:not|no|never|without|nor)\b(?:\W+[A-Za-z0-9.:/&'\-]+){1,3}", " ",
                       self.ref_texts(row, desc=False), flags=re.I)
        maps = [v for v in row.values() if isinstance(v, dict)] or [self._wk_flat_props(row)]
        pairs = [(key, v) for m in maps for key, v in m.items()]
        keys = " ".join(self._wk_camel(key).replace("_", " ") for key, _ in pairs)
        values = [self.title_of(row), notes] + [str(v) for _, v in pairs if self._wk_scalar(v)]
        return set(ref_tokens(keys)), set(ref_tokens(" | ".join(values)))

    @staticmethod
    def _wk_flat_props(row: dict, /) -> dict:
        return {key: v for key, v in row.items() if (isinstance(v, str) or numeric(v)) and not str(key).lower().endswith("id")
                and not re.search(r"title|shop|seller|brand|price|amount|desc|overview|note", str(key).lower())}

    def ref_shows(self, row: dict, detail: Any, parts: Any = None) -> bool:
        names, values = parts if parts is not None else self.ref_parts(row)
        want = {t for t in ref_tokens(detail) if len(t) >= 3 and not re.search(r"\d", t)} - names
        if want and want.isdisjoint(values):
            return False
        if not (figs := ref_figures(detail)):
            return True
        mine = set(ref_figures(self.title_of(row))).union(
            *(ref_figures(str(x)) for v in row.values() if isinstance(v, dict) for x in v.values() if self._wk_scalar(x)))
        for fig, unit in figs:
            same_unit = [f for f, u in mine if u == unit]
            if same_unit and not any(abs(f - fig) < 1e-9 for f in same_unit):
                return False
        return True

    @staticmethod
    def ref_field(row: dict, prop: Any) -> str:
        maps = [v for v in row.values() if isinstance(v, dict)]
        for test in (strictly_same, loosely_same):
            for x in (x for m in maps for key, x in m.items() if test(key, prop)):
                if (isinstance(x, str) or numeric(x)) and (text := str(x).strip()):
                    return text
        return ""

    def store_keys(self) -> set:
        size = len(self.archive.items)
        if (memo := self.store_keys_memo) is not None and memo[0] == size:
            return memo[1]
        found: set = set()

        def visit(d: dict) -> None:
            for v in d.values():
                # (true/false values count toward the two fields here)
                if isinstance(v, dict) and len([x for x in v.values() if isinstance(x, (str, int, float))]) > 1:
                    found.update(w for w in (" ".join(_j_bits(self._wk_camel(key))) for key in v) if w)
        self._wk_visit_all(visit)
        self.store_keys_memo = (size, found)
        return found

    def _wk_visit_all(self, visit: Any, /) -> None:
        for obs in self.archive.items:
            each_dict(obs, visit)

    @staticmethod
    def ref_names_prop(prop: Any, text: Any) -> bool:
        if not (words := [w for w in name_words(prop) if len(w) >= 3]):
            return True
        return any(tok == w or (len(tok) >= 4 and len(w) >= 4 and tok[:4] == w[:4])
                   for tok in _j_bits(text or "", r"[a-z]+") for w in words)

    def ref_card(self, ident: str) -> str:
        row = self.ref_index().get(ident, {})
        sellers = set(self._wk_seller_values(row))
        facts: dict = {}
        for m in (v for v in row.values() if isinstance(v, dict)):
            # (the first 40 fields of each map are looked at; the plain ones are shown as text)
            facts.update((str(key), str(x)[:90]) for key, x in list(m.items())[:40] if self._wk_scalar(x))
        card = {"id": ident, "sold by": " / ".join(sorted(sellers)), "category": self.shelf_of(row),
                "title": self.title_of(row)[:160], "facts": facts, "notes": self.ref_texts(row)[:900]}
        return as_text(card, 5000)

    # ------------------------------------------------------------------------------------------- reference bookkeeping
    def ref_ids_add(self, ids: Any) -> None:
        self.ref_ids.update(ids)

    def ref_foreign(self) -> set:
        return self._wk_guard("", set(), self._wk_foreign_ids)

    def _wk_foreign_ids(self) -> set:
        if not self.ref_ids or not (homes := self.home_shelves()) or not all(homes):
            return set()
        allowed = set().union(*homes)
        rows = self.ref_index()
        return {i for i in self.ref_ids if i not in self.trolley.units
                and (lab := self.shelf_of(rows.get(i, {}))) and lab not in allowed}

    # ------------------------------------------------------------------------------------------- candidates and searches
    def ref_cands(self, record: Any) -> list:
        target = record.target
        name = seller_key(target.get("seller"))
        kinds = kind_set(target.get("kind"))
        details = target.get("details") or []
        rows = self.ref_index()
        shows: dict = {}
        for ident, row in rows.items():
            if name in self.ref_sellers(row):
                blurbs = " ".join(v for key, v in row.items()
                                  if isinstance(v, str) and re.search(r"desc|overview|summary", str(key).lower()))
                shows[ident] = _j_shown(details, set(ref_tokens(self.title_of(row) + " " + blurbs)))
        record.shows = shows

        def rank(i: str) -> tuple:
            score = (2 if kinds & kind_set(self.shelf_of(rows[i])) else 0) + shows[i]
            return -score, -self.ref_seen_at.get(i, 0), i
        return sorted(shows, key=rank)

    def ref_modes(self, needed: Any, labels: Any) -> tuple:
        groups: dict = {}
        for at, walk in enumerate(self.page_book.values()):
            texts = self.free_text(args := _j_mapped(walk, "args"))
            # (exactly one free-text value, blank ones counted, and exactly one argument holding a non-blank one of them)
            if len(texts) != 1 or len(holders := self._wk_text_holders(args, texts)) != 1:
                continue
            other = self._wk_unlabelled(args, needed, labels, (holders[0],))
            stems = frozenset(self.stem(w) for w in _j_bits(texts[0]))
            groups.setdefault(_j_sig(other), dict(tpl=(other, holders[0]), at=at, runs=[]))["runs"].append(
                (stems, tuple(walk.get("first") or ())))
        blind, ranked = set(), []
        for gkey, group in groups.items():
            runs = group["runs"]
            # (a mode is blind to its text when two runs of unrelated words listed the same first rows)
            if any(len(a[1]) >= 2 and a[1] == b[1] and a[0].isdisjoint(b[0]) for n, a in enumerate(runs) for b in runs[n + 1:]):
                blind.add(gkey)
                continue
            ranked.append((-len(set(r[0] for r in runs)), group["at"], group["tpl"]))
        ranked.sort(key=lambda r: r[:2])
        return [*(r[2] for r in ranked)], blind

    @staticmethod
    def _wk_text_holders(args: dict, texts: list, /) -> list:
        return [a for a, v in args.items() if isinstance(v, str) and v.strip() and v in texts]

    @staticmethod
    def _wk_unlabelled(args: dict, needed: Any, labels: Any, drop: tuple, /) -> dict:
        return {a: v for a, v in args.items()
                if a not in drop and (a in needed or not (isinstance(v, str) and v.strip().lower() in labels))}

    def ref_template(self) -> Any:
        schema = self.shapes.get((tool := self.role("search")) or "") or {}
        needed = set(_j_listed(schema, "required"))
        labels = set(map(self.shelf_of, self.ref_index().values())) - {""}
        pkey = found[0] if (found := self.page_arg()) else None
        tpls, blind = self.ref_modes(needed, labels)
        if (tpl := tpls[0] if tpls else self._wk_logged_template(tool, pkey, needed, labels, blind)) is not None:
            return self._wk_unlabelled(tpl[0], needed, labels, ()), tpl[1]
        token = self.token_key()
        texts = [p for p, s in _j_mapped(schema, "properties").items()
                 if p != token and isinstance(s, dict) and not s.get("enum") and s.get("type") == "string"]
        return ({}, texts[0]) if len(texts) == 1 and needed <= {texts[0]} else None

    def _wk_logged_template(self, tool: Any, pkey: Any, needed: set, labels: set, blind: set, /) -> Any:
        for entry in reversed(self.log):
            if not isinstance(entry, dict) or entry.get("kind") != "env":
                continue
            for action in reversed(entry.get("actions") or []):
                if not isinstance(action, dict) or action.get("name") != tool or not isinstance(action.get("args"), dict):
                    continue
                args = action["args"]
                if len(holders := self._wk_text_holders(args, self.free_text(args))) != 1:
                    continue
                other = {a: v for a, v in args.items() if a != holders[0] and a != pkey}
                if _j_sig(self._wk_unlabelled(other, needed, labels, ())) not in blind:
                    return other, holders[0]
        return None

    def ref_words(self, record: Any) -> list:
        keys = self._wk_name_words()
        target = record.target
        seller = set(_j_bits(target.get("seller") or ""))
        details = [str(d) for d in target.get("details") or []]
        bins: tuple = ([], [], [])
        for d in details:
            prev = ""
            for tok in d.split():
                self._wk_bin_word(tok, bool(re.fullmatch(r"\d+(?:[.,]\d+)?", prev)), seller, keys, bins)
                prev = tok
        joined, figures, words = bins
        words.sort(key=len, reverse=True)
        tokens = [[t for t in _j_bits(d) if t not in FILLER] for d in details]
        pairs = record.hops == 0 and bool(details) and all(len(t) >= 2 for t in tokens)
        known = self.store_keys()
        lasts: list = []
        for t in tokens:
            if len(t) < 2 or t[-1] not in words or t[-1] in lasts:
                continue
            if pairs or any(" ".join(t[:i]) in known for i in range(1, len(t))):
                lasts.append(t[-1])
        words = lasts + [w for w in words if w not in lasts]
        return list(dict.fromkeys(joined + figures + words))

    def _wk_name_words(self) -> set:
        keys: set = set()
        for row in self.ref_index().values():
            for m in (v for v in row.values() if isinstance(v, dict)):
                for key in m:
                    keys.update(_j_bits(self._wk_camel(key), r"[a-z]+"))
        for req in [*self.needs.written, *self.needs.uncovered]:
            keys.update(_j_bits(self._wk_name(req), r"[a-z]+"))
        return keys

    @staticmethod
    def _wk_bin_word(tok: str, after_figure: bool, seller: set, keys: set, bins: tuple, /) -> None:
        low = tok.lower()
        whole = low.strip(".,;:!?()[]{}\"'")
        for w in re.findall(r"[a-z0-9]+", low):
            digit = re.search(r"\d", w) is not None
            if w in FILLER or w in seller or (after_figure and not digit):
                continue
            if digit:
                if re.search(r"[a-z]", w):
                    bins[0].append(w)
                elif len(w) >= 2:
                    bins[1].append(w)
            elif len(w) >= 3 and w == whole and w not in keys and w.rstrip("s") not in keys:
                bins[2].append(w)

    def ref_plans(self, record: Any) -> list:
        width = max(2, self.lead_words or 3)
        target = record.target
        seller = " ".join(re.findall(r"[A-Za-z0-9]+", re.sub(r"'s\b", "", str(target.get("seller") or ""))))
        words = self.ref_words(record)
        kind = [w for w in _j_bits(target.get("kind") or "", r"[a-z]{3,}") if w not in FILLER][-1:]
        if record.unindexed:
            plans = [" ".join(" ".join(combo).split()[:width])] if (combo := words[:1] + kind) else []
        else:
            plans = [seller + " " + words[0]] if words and len(seller.split()) < width else []
            plans.append(seller)
        return [p for p in plans if p.strip() and p.lower() not in record.queries]

    def ref_page(self, record: Any) -> Any:
        found = self.page_arg()
        if not found or record.pages >= J_REF_PAGES or not (name := seller_key(record.target.get("seller"))):
            return None
        for walk in reversed(list(self.page_book.values())):
            if name in seller_key(self._wk_query(walk.get("args") or {})) and (nxt := self.next_page(walk)) is not None:
                return self._wk_next_args(walk, found[0], nxt)
        return None

    # ------------------------------------------------------------------------------------------- budgets and gating
    def ref_floor(self) -> int:
        base = max(Knobs.LATE_STEPS, min(Knobs.ORDER_KEEP, self.step_cap // 4 + 1))
        return max(base + 10, self.order_due() + 7)

    def ref_room(self) -> bool:
        open_recs = [r for r in self.refs if r.status == "open"]
        all_differ = bool(open_recs) and all(r.relation == "differ" for r in open_recs)
        if all_differ and not any(r.chain for r in open_recs) and self.ref_steps >= Knobs.REF_DIFFER_BLIND:
            return False
        if all_differ:
            cap = max(J_REF_DIFFER_MIN_STEPS, int(Knobs.REF_DIFFER_SHARE * self.step_cap))
        else:
            cap = max(J_REF_MIN_STEPS, int(Knobs.REF_STEP_SHARE * self.step_cap))
        if self.ref_steps >= cap or not self.wire.left() > Knobs.WIND_DOWN_S + 60 or self.wire.spent_out() \
                or not self._wk_live() or self.ref_waits(floor := self.ref_floor()):
            return False
        return self.moves_left() > floor or self.ref_hop_room()

    def ref_product(self, record: Any) -> Any:
        if self.needs.items() and record.tag:
            return self.product_of("x (%s)" % record.tag)
        return None

    def ref_waits(self, floor: Any = None) -> bool:
        items = self.needs.items()
        open_recs = [r for r in self.refs if r.status == "open"]
        if not items or not open_recs or any(r.relation != "differ" or not r.chain for r in open_recs):
            return False
        if self.moves_left() <= (floor if floor is not None else self.ref_floor()) + 4:
            return False
        got = self._wk_guard("ref_wait_error", None, self._wk_wait_parts, open_recs, len(items))
        if got is None:
            return False
        owners, lacking = got
        rest = tuple(k for k in lacking if k not in {o for o in owners if o is not None})
        waits = len(rest) >= 2 if None in owners else bool(rest)
        if waits and rest != self.ref_wait_said:
            self.ref_wait_said = rest
            self._wk_log("ref_wait", products=list(rest))
        return waits

    def _wk_wait_parts(self, open_recs: list, count: int, /) -> tuple:
        owners = [self.ref_product(r) for r in open_recs]
        return owners, [k for k in range(count) if not any(e["fit"] for e in self.ed_pool(k))]

    def ref_hop_room(self) -> bool:
        if self.moves_left() <= self.order_due() + 2:
            return False
        for r in self.refs:
            target = r.target
            if r.status != "open" or r.hops < 1 or not (_j_text(target, "seller") and _j_text(target, "kind")
                                                        and target.get("details")):
                continue
            if r.late_from is not None:
                if self.ref_steps - r.late_from < 3:
                    return True
            elif not r.queries and self.ed_others_verified(r):
                r.late_from = self.ref_steps
                self._wk_log("ref_late", req=r.req, steps=self.ref_steps)
                return True
        return False

    def ed_others_verified(self, record: Any) -> bool:
        if not (items := self.needs.items()):
            return True
        bare = self._wk_guard("", None, lambda: [k for k in range(len(items))
                                                 if not any(e["verified"] and e["fit"] for e in self.ed_pool(k))])
        if bare is None:
            return False
        own = self.ref_product(record)
        return len(bare) <= 1 if own is None else all(k == own for k in bare)

    def ref_calls_ok(self) -> bool:
        w = self.wire
        return (self.ref_calls < Knobs.REF_MODEL_CALLS and w.left() > Knobs.WIND_DOWN_S + 40 and not w.spent_out()
                and w.spend < J_HOP_SPEND_SHARE * Knobs.SPEND_HARD * w.scale)

    # ------------------------------------------------------------------------------------------- matching and liveness
    def ref_match(self, record: Any, req: Any) -> bool:
        if (text := str(req)) == record.req or text == record.now:
            return True
        value = self._wk_value(text)
        if not (points_elsewhere(value) or (record.relation == "differ" and DIFFERING.search(value.lower()))):
            return False
        if not loosely_same(self._wk_name(text), record.prop):
            return False
        tags = _j_tags(self._wk_head(text))
        return not (tags and record.tag) or not kind_set(tags[-1]).isdisjoint(kind_set(record.tag))

    def ref_live(self, record: Any) -> bool:
        for q in self.needs.props_left():
            if self.ref_match(record, q) or loosely_same(self._wk_name(q), record.prop):
                return True
        return not any(loosely_same(self._wk_name(d), record.prop) for d in self.needs.let_go)

    # ------------------------------------------------------------------------------------------- the resolver turn
    def ref_step(self) -> bool:
        if not self.refs or not self._wk_live():
            return False
        try:
            return self.ref_turn()
        except Exception as error:  # noqa: BLE001
            self._wk_log("ref_error", error="%s: %s" % (type(error).__name__, str(error)[:200]))
        for r in (r for r in self.refs if r.status == "open"):
            r.status, r.why = "stuck", "the code's resolver failed"
        return False

    def ref_turn(self) -> bool:
        for r in self.refs:
            if r.status != "open" or self.ref_live(r):
                continue
            r.status = "dropped"
            self._wk_log("ref_dropped", req=r.req)
        if not (open_recs := [r for r in self.refs if r.status == "open"]):
            return False
        tick = self.wire.tick
        acted = any([self.ref_judge(r) for r in sorted(open_recs, key=lambda x: x.hops)])
        if self.wire.tick != tick:
            return True
        open_recs = [r for r in open_recs if r.status == "open"]
        if open_recs and self.ref_room():
            return self._wk_ref_fetch(open_recs, acted)
        return acted

    def _wk_ref_fetch(self, open_recs: list, acted: bool, /) -> bool:
        most = batch[2] if (batch := self.batch_tool()) else 2
        pools = []
        for r in open_recs:
            pool = [i for i in self.ref_cands(r) if not (i in self.archive.read or i in r.judged)]
            ready = r.queries or len(pool) <= most or (pool and r.shows.get(pool[0], 0) >= 1)
            pools.append(pool if ready or not self.ref_plans(r) else [])
        tpl = self.ref_template()
        queue = self._wk_fill(pools, most)
        needy = [r for r, pool in zip(open_recs, pools) if not pool and self.ref_plans(r)] \
            if tpl and 0 < len(queue) < most else []
        if queue and not needy:
            return self.ref_read(queue, open_recs) or acted
        if tpl is None:
            return acted
        for r in needy or sorted(open_recs, key=lambda x: (x.hops, len(x.queries))):
            if plans := self.ref_plans(r):
                r.queries.append(plans[0].lower())
                done = self.ran_query(plans[0]) or self.ref_search(r, {**tpl[0], tpl[1]: plans[0]})
            elif (more := self.ref_page(r)) is not None:
                r.pages += 1
                done = self.ref_search(r, more)
            else:
                continue
            return done or acted
        for r in open_recs:
            if set(self.ref_cands(r)) <= r.judged and not self.ref_plans(r) and self.ref_page(r) is None:
                acted = self._wk_give_up(r, (), "the listing it points at was not found by the store's search")
        return acted

    def ref_read(self, queue: list, open_recs: list) -> bool:
        batch = self.batch_tool()
        if len(queue) < 2:
            batch = None
        say = ("The request takes %s from another listing, so I open the listings that may be it."
               % ", ".join(dict.fromkeys(r.head for r in open_recs)))
        tick = self.wire.tick
        replies = self.open_reads(queue, say, batch)
        dropped = batch is not None and replies and not self.batch_proven and self._wk_batch_failed(batch, queue)
        self.post_replies(replies)
        spent = self._wk_charge(tick)
        self.ref_ids_add(queue)
        self._wk_log("ref_read", ids=list(queue), steps=spent)
        if not dropped:
            unread = [i for i in queue if i not in self.archive.read]
            for r in open_recs:
                r.judged.update(unread)
        return spent != 0

    def _wk_batch_failed(self, batch: tuple, queue: list, /) -> bool:
        if self.batch_ok(self.last_reads, queue):
            self.batch_proven = True
            return False
        self.batch_spec = None
        self._wk_log("batch_dropped", tool=batch[0])
        return True

    def _wk_charge(self, tick: int, /) -> int:
        spent = self.wire.tick - tick
        self.ref_steps += spent
        return spent

    def ref_search(self, record: Any, args: dict) -> bool:
        say = ("The request takes %s from another listing (%s), so I search for that listing."
               % (record.head, self.ref_desc(record)))
        tick = self.wire.tick
        grew = self.walk_call(args, say)
        spent = self._wk_charge(tick)
        rows = self._wk_rows_near(self.archive.items[-1]) if grew and self.archive.items else []
        name = seller_key(record.target.get("seller"))
        if rows and seller_key(self._wk_query(args)) == name and all(name not in self.ref_sellers(r) for r in rows):
            record.unindexed = True
        self._wk_log("ref_search", req=record.req, args=args, rows=len(rows), unindexed=record.unindexed)
        return spent != 0

    def ref_judge(self, record: Any) -> bool:
        listed = self.ref_cands(record)
        seller_read = [i for i in listed if i in self.archive.read]
        if not (cands := [i for i in seller_read if i not in record.judged]):
            return False
        rows = self.ref_index()
        parts = {i: self.ref_parts(rows.get(i, {})) for i in seller_read}
        details = [*(record.target.get("details") or [])]
        if len(seller_read) > 3:
            for d in details:
                if d in record.loose or any(self.ref_shows(rows.get(i, {}), d, parts[i]) for i in seller_read):
                    continue
                record.loose.append(d)
        tight = [d for d in details if d not in record.loose]
        ok_map = {i: all(self.ref_shows(rows.get(i, {}), d, parts[i]) for d in tight) for i in cands}
        full_map = {i: _j_shown(details, set().union(*parts[i])) for i in cands}
        ranked = [i for *_, i in sorted((not ok_map[i], -full_map[i], n, i) for n, i in enumerate(cands))]
        if not ok_map[ranked[0]] and (self.ref_room() or self.ref_waits()) \
                and (any(i not in self.archive.read and i not in record.judged for i in listed) or self.ref_plans(record)
                     or self.ref_page(record) is not None):
            return False
        fine = [i for i in ranked if ok_map[i]]
        pick = fine[:J_HOP_PICK] or ranked[:J_HOP_PICK]
        exact = [i for i in pick if full_map[i] == len(details)] if details else []
        lone = exact[0] if len(exact) == 1 else None
        if missing := self.ref_notes_missing(record, [lone] if lone is not None else fine[:2] or pick[:1]):
            if self.ref_room() and self.ref_notes_read(record, missing):
                return True
            if self.ref_waits():
                return False
            if not self.ref_room() and record.relation == "differ":
                return self._wk_give_up(record, pick, "the notes of the listing it points at could not be read in the "
                                                      "steps left")
        if lone is not None and self.ref_plain(record, lone):
            self._wk_close(record, pick)
            self.ref_resolve(record, lone, self.ref_field(rows.get(lone, {}), record.prop), [])
            return True
        if not self.ref_calls_ok():
            return self._wk_give_up(record, pick, "no hop checks are left")
        before, answer = record.target, self.ref_ask(record, pick, "")
        self.ref_apply(record, pick, answer)
        if record.status == "open" and record.target is before:
            record.judged.update(i for i in cands if not ok_map[i])
        return True

    def _wk_close(self, record: Any, pick: Any, /) -> None:
        record.judged.update(pick)
        self.ref_ids_add(pick)

    def ref_plain(self, record: Any, ident: str) -> bool:
        rows = self.ref_index()
        row = rows.get(ident, {})
        if not (val := self.ref_field(row, record.prop)):
            return False
        notes = self.ref_texts(row, desc=False)
        if self.ref_names_prop(record.prop, notes) or not set(ref_tokens(notes)).isdisjoint(ref_tokens(val)):
            return False
        own, others, fields = self._wk_seller_split(rows, ident, row)
        for w in (w for sentence in J_CLAUSE_END.split(notes) for w in J_WORD.findall(sentence)[1:]):
            if not (reduced := seller_key(w)):
                continue
            if reduced in others or (w[0].isupper() and len(reduced) >= 4 and reduced not in own and reduced not in fields):
                return False
        return True

    def ref_notes_tool(self) -> Any:
        if tool := self.notes_tool():
            return tool
        details = self.role("details")
        if details and self.note_keys.get(details):
            return None
        lookups = [t for t in self.lookup_tools() if t != details]
        known = set(self.row_keys) | (set(self.tool_keys.get(details, ())) if details else set())
        texty = [t for t in lookups if not (set(self.tool_keys.get(t, ())) - known).isdisjoint(self.text_keys.get(t, ()))]
        if texty:
            return max(texty, key=lambda t: (self._wk_cap(t, "calls"), t))
        return next((t for t in (self.role("view"), self.stock_tool()) if t and t in lookups and not self._wk_cap(t, "id")),
                    None)

    def _wk_cap(self, tool: Any, key: str, /) -> Any:
        return (self.tool_caps.get(tool) or {}).get(key, 0)

    def ref_notes_missing(self, record: Any, ids: list) -> list:
        tool = self.ref_notes_tool()
        if not tool or not ids:
            return []
        done = self.read_with.get(tool, ())
        return [i for i in dict.fromkeys(ids) if i not in done and i not in record.noted][:2]

    def ref_notes_read(self, record: Any, ids: list) -> bool:
        if not (tool := self.ref_notes_tool()) or not ids:
            return False
        record.noted.update(ids)
        say = ("I read the notes of the listing that may be the one the request points at, since a note may send the "
               "reader on to another listing.")
        tick = self.wire.tick
        self.post_replies(self.open_reads(list(ids), say, None, tool))
        spent = self._wk_charge(tick)
        self.ref_ids_add(ids)
        self._wk_log("ref_notes", req=record.req, ids=list(ids), tool=tool)
        return spent != 0

    # ------------------------------------------------------------------------------------------- the hop call
    def ref_ask(self, record: Any, pick: list, reason: str) -> Any:
        if not self.ref_calls_ok():
            return None
        self.ref_calls += 1
        prompt = "%s\n\nCANDIDATE LISTING RECORDS:\n%s" % (
            HOP_TEXT.replace("PROPERTY", record.prop).replace("DESCRIPTION", self.ref_desc(record)),
            "\n".join(self.ref_card(i) for i in pick))
        if reason:
            prompt += ("\n\nYour previous answer could not be used: %s. Answer again from the records above." % reason)
        answer = self.wire.ask_json(prompt, J_HOP_TOKENS, LINEUP["checker"][:1])
        self._wk_log("ref_hop", req=record.req, pick=list(pick), retry=bool(reason), answer=answer)
        return answer

    def ref_apply(self, record: Any, pick: list, answer: Any, again: bool = True) -> None:
        if not isinstance(answer, dict):
            record.fails += 1
            if record.fails > 1:
                self._wk_give_up(record, pick, "the hop check gave no answer")
            return
        raw = answer.get("target")
        target = "" if isinstance(raw, bool) or not isinstance(raw, (str, int)) else str(raw).strip()
        missing = [str(m) for m in _j_listed(answer, "missing") if str(m).strip()]
        if missing or target not in pick:
            self._wk_close(record, pick)
            if target in pick:
                self._wk_log("ref_partial", req=record.req, target=target, missing=missing[:6])
            return
        where, problem, *found = self.ref_validate(record, target, answer, pick)
        if not problem:
            self._wk_close(record, pick)
            if target in record.chain:
                self.ref_stop(record, "its notes lead back to a listing already followed")
            elif found[0]:
                self.ref_go(record, target, found[0])
            else:
                self.ref_resolve(record, target, *found[1:])
        elif again and self.ref_calls_ok():
            self.ref_apply(record, pick, self.ref_ask(record, pick, problem), False)
        else:
            self.ref_ids_add(pick)
            self._wk_log("ref_invalid", req=record.req, target=target, problem=problem)
            if where == "target":
                record.judged.add(target)
            else:
                record.chain.append(target)
                self._wk_give_up(record, pick, problem)

    def ref_validate(self, record: Any, target: str, answer: dict, pick: list) -> tuple:
        rows = self.ref_index()
        mine = self.ref_parts(row := rows.get(target, {}))
        details = [*(record.target.get("details") or [])]
        for d in details:
            if d not in record.loose and not self.ref_shows(row, d, mine):
                return self._wk_reject("target", "%s does not show '%s', so it is not the listing described" % (target, d))

        def shown(ident: str) -> int:
            names, values = mine if ident == target else self.ref_parts(rows.get(ident, {}))
            return _j_shown(details, names | values)
        if shown(best := max(pick or [target], key=shown)) > shown(target):
            return self._wk_reject("target", "%s shows more of the named details than %s" % (best, target))
        notes = self.ref_texts(row)
        note_toks = set(ref_tokens(notes))
        if isinstance(go := answer.get("go_to"), dict) and _j_text(go, "seller"):
            return self._wk_check_go(record, target, row, notes, note_toks, go)
        value = _j_text(answer, "value")
        value_toks = set(ref_tokens(value))
        if not value or not value_toks or not value_toks <= (mine[0] | mine[1]):
            return self._wk_reject("answer", "the value '%s' is not written in %s's record" % (value, target))
        pairs: list = []
        for g in _j_listed(answer, "goes_with"):
            if not self._wk_scalar(g) or not (text := str(g).strip()):
                continue
            toks = set(ref_tokens(str(g)))
            if toks and toks <= note_toks and toks != value_toks and text not in pairs:
                pairs.append(text)
        return "", "", None, value, pairs[:3]

    @staticmethod
    def _wk_reject(where: str, why: str, /) -> tuple:
        return where, why, None, "", []

    def _wk_check_go(self, record: Any, target: str, row: dict, notes: str, note_toks: set, go: dict, /) -> tuple:
        """A go_to answer holds when its seller is written in the target's notes, in a sentence about the property,
        and names another listing; its details keep the ones half written there."""
        seller = _j_text(go, "seller")
        named = _j_named(seller)
        if not named.search(notes.lower()):
            return self._wk_reject("answer", "the seller '%s' is not written in %s's notes" % (seller, target))
        if not any(named.search(s.lower()) and self.ref_names_prop(record.prop, s) for s in J_SENTENCE_END.split(notes)):
            return self._wk_reject("answer", "the note of %s naming '%s' is about another property, not %s"
                                   % (target, seller, record.prop))
        kept: list = []
        for d in _j_listed(go, "details"):
            if not self._wk_scalar(d) or not str(d).strip():
                continue
            toks = set(ref_tokens(str(d)))
            if toks and 2 * len(toks & note_toks) >= len(toks):
                kept.append(str(d).strip())
        nxt = dict(seller=seller, kind=_j_text(go, "kind")[:80], details=kept[:6])
        if seller_key(seller) in self.ref_sellers(row) and self.ref_key(nxt) == self.ref_key(record.target):
            return self._wk_reject("answer", "the go_to names %s itself, not another listing" % target)
        return "", "", nxt, "", []

    def ref_go(self, record: Any, target: str, go: dict) -> None:
        record.chain.append(target)
        key = self.ref_key(go)
        if key in record.seen:
            self.ref_stop(record, "its notes lead back to a listing already followed")
        elif record.hops + 1 >= Knobs.REF_HOPS:
            self.ref_stop(record, "more than %d listings in a row send the reader elsewhere" % Knobs.REF_HOPS)
        else:
            record.seen.add(key)
            record.hops += 1
            record.update(target=go, judged=set(), queries=[], pages=0, tries=0, fails=0, unindexed=False, loose=[])
            self.log.append({"kind": "ref_go", "req": record.req, "from": target, "to": go, "hops": record.hops})
            self.ref_tell("%s is the listing named for %s, and its notes send the reader on to %s; the code follows them."
                          % (target, record.head, self.ref_desc(record)))

    def ref_value(self, record: Any) -> str:
        value = str(record.value or "")
        pairs = [p for p in record.goes_with if str(p).lower() != value.lower()]
        if record.relation == "within" and numeric(record.tol):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*([a-z%]*)", value.lower().replace(",", ""))
            if m:
                base = float(m.group(1))
                unit = m.group(2) or str(record.unit or "")
                low = format(round(base - record.tol, 6), "g")
                high = format(round(base + record.tol, 6), "g")
                span = "%s to %s" % (low, high) + ((" " + unit) if unit else "")
                return "%s (within %s of %s)" % (span, format(record.tol, "g"), value)
        if record.relation in ("at most", "at least"):
            return "%s %s" % (record.relation, value)
        if record.relation == "differ":
            return " and ".join(["not " + value] + ["not " + str(p) for p in pairs])
        return " or ".join([value] + [str(p) for p in pairs])

    def ref_resolve(self, record: Any, target: str, value: Any, pairs: list) -> None:
        if not str(value or "").strip():
            self.ref_stop(record, "%s shows no value for it" % target)
            return
        record.chain.append(target)
        record.update(status="resolved", value=str(value).strip(), goes_with=list(pairs))
        record.now = new_req = "%s: %s" % (record.head, self.ref_value(record))
        self.ref_swap(record, new_req)
        self.needs.resolved.append(new_req)
        self.needs.version += 1
        self._wk_log("ref_resolved", req=record.req, now=new_req, chain=list(record.chain))
        via = (" (reached from %s through the listings' notes)" % record.chain[0]) if len(record.chain) > 1 else ""
        told = "RESOLVED: %s is given by another listing, and %s shows %s%s." % (record.head, target, record.value, via)
        if record.relation == "differ":
            if record.goes_with:
                told += " Its notes pair that value with %s." % ", ".join(record.goes_with)
            told += " The shopper's value MUST DIFFER from it: any other value meets the requirement, that value does not."
        elif record.goes_with:
            told += (" Its notes pair that value with %s, so each of those meets it too." % ", ".join(record.goes_with))
        self.ref_tell(told + " The requirement is now: %s" % new_req)

    def ref_swap(self, record: Any, new_req: str) -> None:
        seller = seller_key(record.first.get("seller"))
        new_value = self._wk_value(new_req).strip() if ":" in new_req else new_req

        def mine(req: Any) -> bool:
            if self.ref_match(record, req):
                return True
            head = self._wk_head(req)
            if (tags := _j_tags(head)) and record.tag and kind_set(tags[-1]).isdisjoint(kind_set(record.tag)):
                return False
            return bool(seller) and seller in seller_key(self._wk_value(req)) \
                and loosely_same(head.split("(", 1)[0].strip(), record.prop)

        def swapped(reqs: list) -> tuple:
            """The list with this reference's requirements replaced (duplicates dropped), and whether any changed."""
            out: list = []
            changed = False
            for req in reqs:
                if mine(req):
                    changed = changed or req != new_req
                    req = new_req
                if req not in out:
                    out.append(req)
            return out, changed
        self.needs.written, hit_written = swapped(list(self.needs.written))
        self.needs.uncovered, hit_uncovered = swapped(list(self.needs.uncovered))
        if self.needs.now_reqs is not None:
            now = swapped(list(self.needs.now_reqs))[0]
            self.needs.now_reqs = now if new_req in now else now + [new_req]
        if not (hit_written or hit_uncovered):
            self.needs.uncovered.append(new_req)
        reading = self.needs.reading
        if isinstance(reading.get("requirements"), list):
            reading["requirements"] = list(self.needs.written)
        items = [it for it in _j_listed(reading, "items") if isinstance(it, dict) and isinstance(it.get("requirements"), list)]
        for item in items:
            if len(items) > 1 and record.tag and kind_set(record.tag).isdisjoint(kind_set(item.get("product"))):
                continue
            item["requirements"] = [("%s: %s" % (self._wk_head(r).strip(), new_value)) if mine(r) else r
                                    for r in item["requirements"]]

    def ref_stop(self, record: Any, why: str) -> None:
        if record.status != "open":
            return
        record.status = "stuck"
        record.why = why
        self.log.append({"kind": "ref_stuck", "req": record.req, "why": why})
        self.ref_tell("NOT RESOLVED: %s (%s). Its value stays unknown: if records you read show where the chain ends, "
                      "use that value; otherwise never decline or reject a listing over it - it is never a veto."
                      % (record.head, why))

    def ref_tell(self, text: str) -> None:
        if len(self.chat) > 2 and isinstance(last := self.chat[-1], dict) and last.get("role") in ("tool", "user") \
                and isinstance(last.get("content"), str):
            last["content"] += "\n[reference] " + text
        else:
            self.ref_told.append(text)

    # ------------------------------------------------------------------------------------------- what other checks see
    def ref_line(self) -> str:
        try:
            parts = []
            for r in self.refs:
                if r.status == "resolved":
                    parts.append("%s (read on %s%s)" % (r.now, r.chain[-1] if r.chain else "?",
                                                        ", must differ from it" if r.relation == "differ" else ""))
                elif r.status == "open":
                    parts.append("%s: the code is following it (now: %s)" % (r.head, self.ref_desc(r)))
                elif r.status == "stuck":
                    parts.append("%s: not resolved, value unknown (never decline or reject a listing over it)" % r.head)
            out = ("\n[references] " + "; ".join(parts)) if parts else ""
            told, self.ref_told = self.ref_told, []
            return out + "".join("\n[reference] " + t for t in told)
        except Exception:  # noqa: BLE001
            return ""

    def ref_note(self, thing: str = "") -> str:
        try:
            lines = []
            reached: list = []
            for r in self.refs:
                if r.status == "resolved":
                    last = r.chain[-1] if r.chain else "?"
                    via = (" (reached from %s through the listings' notes)" % r.chain[0]) if len(r.chain) > 1 else ""
                    pairing = ("; its notes pair it with %s" % ", ".join(r.goes_with)) if r.goes_with else ""
                    if r.relation == "differ":
                        lines.append("- %s: the shopper's %s MUST DIFFER from the value shown on %s (%s)%s%s: any other "
                                     "value meets it, that value does not; the listings of its chain (%s) carry it."
                                     % (r.now, r.prop, last, r.value, via, pairing, ", ".join(r.chain)))
                    else:
                        lines.append("- %s: the value shown on %s%s%s." % (r.now, last, via, pairing))
                elif r.status in ("open", "stuck"):
                    chain = " -> ".join(r.chain) if r.chain else "the listing it points at was not found"
                    lines.append("- %s: NOT RESOLVED by the code (its chain so far: %s). If a record below is that "
                                 "listing, or the last listing its chain reached, and writes its own value without "
                                 "sending the reader elsewhere, that value is the requirement; otherwise the value is "
                                 "unknown, so never fail an item on it alone." % (r.head, chain))
                    reached += [c for c in r.chain[-2:] if c != thing]
            if not lines:
                return ""
            text = ("REQUIREMENTS GIVEN BY POINTING AT ANOTHER LISTING (the code followed that listing's notes):\n"
                    + "\n".join(lines))
            shown: list = []
            done: set = set()
            for ident in reversed(reached):
                if ident in done:
                    continue
                done.add(ident)
                recs = self.archive.about(ident, 1)
                if recs:
                    shown.append(recs[0])
                if len(shown) >= 2:
                    break
            if shown:
                text += "\nRecords of the last listings those chains reached:\n" + "\n".join(shown)
            return text
        except Exception:  # noqa: BLE001
            return ""

    def ref_pointers(self) -> set:
        out = set()
        for r in self.refs:
            if r.relation != "differ" or r.status not in ("open", "stuck", "resolved") or not self.ref_live(r):
                continue
            for ident in r.chain:
                if self.ref_own_product(r, ident):
                    out.add(ident)
        return out

    def ref_own_product(self, record: Any, ident: str) -> bool:
        if not self.needs.items():
            return True
        label = self.shelf_known(ident)
        k = self.product_of("x (%s)" % label) if label else None
        if k is None or not record.tag:
            return True
        return k == self.ref_product(record)

    def ref_pointer_note(self, thing: str) -> str:
        for r in self.refs:
            if r.relation != "differ" or thing not in r.chain or not self.ref_live(r) or not self.ref_own_product(r, thing):
                continue
            shown = (" (%s)" % r.value) if r.status == "resolved" and r.value else ""
            return ("%s is the listing the shopper's reference for %s points at (or one its notes lead through), so its "
                    "%s is the very value%s the shopper's %s must differ from - it cannot meet that requirement."
                    % (thing, r.head, r.prop, shown, r.prop))
        return ""

    def ref_excluded(self, fields: Any, k: Any) -> bool:
        several = bool(self.needs.items())
        pairs = list(fields.items()) if isinstance(fields, dict) else []
        for r in self.refs:
            if r.relation != "differ" or r.status != "resolved" or not self.ref_live(r):
                continue
            if several and (owner := self.ref_product(r)) is not None and owner != k:
                continue
            bad = {" ".join(ref_tokens(v)) for v in [r.value, *r.goes_with] if str(v or "").strip()}
            if any(loosely_same(name, r.prop) and " ".join(ref_tokens(str(val))) in bad for name, val in pairs):
                return True
        return False

    def own_redirects(self, thing: str) -> list:
        return self._wk_guard("", [], self._wk_redirects, thing)

    def _wk_redirects(self, thing: str, /) -> list:
        rows = self.ref_index()
        if not (row := rows.get(thing)) or not (notes := self.ref_texts(row, desc=False)).strip():
            return []
        own, elsewhere, fields = self._wk_seller_split(rows, thing, row)
        label = self.shelf_of(row)
        mine_k = self.product_of("x (%s)" % label) if self.needs.items() and label else None
        out = []
        for req in self.needs.props_left():
            head = self._wk_head(req)
            if mine_k is not None and self._wk_foreign_req(head, mine_k):
                continue
            prop = head.split("(", 1)[0].strip()
            if prop and (sentence := self._wk_redirect_sentence(notes, prop, own, elsewhere, fields)) is not None:
                out.append((req, prop, sentence.strip()))
        return out

    @staticmethod
    def _wk_redirect_sentence(notes: str, prop: str, own: set, elsewhere: set, fields: set, /) -> Any:
        """The first sentence of the notes naming another seller (not negated) after it names the property."""
        for sentence in J_CLAUSE_END.split(notes):
            words = [(m.start(), m.end(), m.group()) for m in J_WORD.finditer(sentence)]
            blanked = re.sub(r"[A-Za-z0-9]+(?=\s+by\s+[A-Z])", lambda m: " " * len(m.group()), sentence)
            first_prop = next((start for start, end, _ in words if mentions_prop(prop, blanked[:end])), None)
            if first_prop is None:
                continue
            for idx, (start, _, w) in enumerate(words):
                if idx == 0 or start <= first_prop or not (reduced := seller_key(w)):
                    continue
                other = reduced in elsewhere or (w[0].isupper() and len(reduced) >= 4 and reduced not in own
                                                 and fields.isdisjoint(ref_tokens(w)) and reduced not in fields)
                if other and not any(words[j][2].lower() in ("not", "nor", "never", "no", "neither")
                                     for j in range(max(0, idx - 5), idx)):
                    return sentence
        return None

    def _wk_seller_split(self, rows: dict, ident: Any, row: dict, /) -> tuple:
        own = self.ref_sellers(row)
        others: set = set()
        for other_id, other_row in rows.items():
            if other_id != ident:
                others |= self.ref_sellers(other_row)
        return own, others - own, self._wk_field_tokens(row)

    def _wk_foreign_req(self, head: Any, k: Any, /) -> bool:
        owner = self.product_of(head)
        return owner is not None and owner != k

    def notes_remark(self, call_name: str, call_args: Any) -> str:
        if call_name in {self.role(r) for r in ("search", "cart", "cart_view", "message", "order", "decline")} - {None, ""}:
            return ""
        ver = str(self.needs.version)
        remarks = []
        for ident in sorted(self.asked_ids(call_args)):
            key = "%s|%s" % (ident, ver)
            in_cart = ident in self.trolley.units
            if key in self.own_told or not (in_cart or key in self.passed) or not (own := self.own_redirects(ident)):
                continue
            remarks.append(self._wk_redirect_text(ident, key, own, in_cart))
        return " ".join(remarks)

    def _wk_redirect_text(self, ident: str, key: str, own: list, in_cart: bool, /) -> str:
        self.own_told.add(key)
        self.passed.discard(key)
        self._wk_log("own_redirect", id=ident, props=[p for _, p, _ in own])
        said = ", ".join('%s ("%s")' % (p, s[:160]) for _, p, s in own)
        text = ("%s: its own notes may send %s to another listing. If they do, its own field does not count for "
                "that requirement (the grader takes the other listing's value), so it meets it only when that "
                "listing's record shows the value; the item check runs again before the order." % (ident, said))
        if in_cart:
            text += " It is in the cart: if they do, take it out and put another verified fit in."
        return text

    # ------------------------------------------------------------------------------------------- the shopper's pointing
    def ref_phrase(self, line: Any) -> Any:
        text = str(line or "")
        if not POINTING.search(text.lower()):
            return None
        parsed = ref_description(text, lambda name: self.ref_seller_ok(name, text))
        if not parsed or not self.ref_seller_ok(parsed["seller"], text):
            return None
        if (m := re.search(r"(?<![A-Za-z'])%s(?:'s)?(?![A-Za-z])" % re.escape(parsed["seller"]), text)) is None:
            return None
        if (start := self._wk_phrase_start(text[:m.start()])) is None:
            return None
        end = m.end() + self._wk_phrase_stop(text[m.end():])
        phrase = text[start:end].strip(" ,;")
        return (phrase, start, end, parsed) if phrase else None

    @staticmethod
    def _wk_phrase_start(head: str, /) -> Any:
        cuts = [c.start() for c in re.finditer(r"[,;!?]|\.(?=\s|$)", head)]
        cut = cuts[-1] if cuts else -1
        lead = head[cut + 1:].lower()
        marks = [x.start() for pat in (DIFFERING, POINTING, POINT_OPEN) for x in pat.finditer(lead)]
        return cut + 1 + min(marks) if marks else None

    @staticmethod
    def _wk_phrase_stop(tail: str, /) -> int:
        stop = re.search(r"[.!?;](?=\s|$)", tail)
        aside = re.search(r"\(\s*or\b[^)]*\)", tail)
        if aside and (stop is None or aside.start() < stop.start()):
            return aside.end()
        return stop.start() if stop else len(tail)

    def ref_line_words(self, question: Any, outside: Any) -> list:
        kinds = set().union(*(_j_bits(p, r"[a-z]+") for p in self._wk_products()))
        toks = _j_bits(question or "")
        ok = [len(w) > 2 and not ({w, w.rstrip("s")} & kinds) and w not in FILLER for w in toks]
        spans = [(i, i + n) for n in (3, 2, 1) for i in range(len(toks) - n + 1)]
        return [phrase for a, b in spans if all(ok[a:b]) and mentions_prop(phrase := " ".join(toks[a:b]), outside)]

    def ref_fix(self, answer: dict, lines: list, question: Any = None) -> list:
        items = self.needs.items()
        masked = []
        for raw in lines:
            line = str(raw)
            got = self.ref_phrase(line)
            if got is None:
                masked.append(line)
                continue
            outside = line[:got[1]] + " " + line[got[2]:]
            masked.append(outside)
            self._wk_fix_one(answer, line, outside, got, items, question)
        return masked

    def _wk_fix_one(self, answer: dict, line: str, outside: str, got: tuple, items: list, question: Any, /) -> None:
        phrase, start, end, listing = got
        inside = line[start:end]
        needs_l = [*filter(lambda n: isinstance(n, str) and ":" in n, _j_listed(answer, "needs"))]
        refs_l = [r for r in _j_listed(answer, "references") if isinstance(r, dict)]
        in_toks, out_toks = set(ref_tokens(inside)), set(ref_tokens(outside))

        def inside_only(p: Any) -> bool:
            return mentions_prop(p, inside) and not mentions_prop(p, outside)

        def detail(req: Any) -> bool:
            if inside_only(self._wk_name(req)):
                return True
            toks = set(ref_tokens(self._wk_value(req)))
            return bool(toks) and toks.isdisjoint(out_toks) and toks <= in_toks

        if not (prop := self._wk_fix_prop(needs_l, refs_l, outside, inside_only, question)):
            self._wk_log("ref_line_unplaced", line=line[:200])
            return
        tag = self._wk_fix_tag(items, needs_l, prop, outside, question)

        def ours(req: Any) -> bool:
            if not loosely_same(self._wk_name(req), prop):
                return False
            rtags = _j_tags(self._wk_head(req)) if tag else []
            return not rtags or not kind_set(rtags[-1]).isdisjoint(kind_set(tag))
        mine = [*filter(ours, needs_l)]
        relation, tol, unit = ref_relation(phrase)
        seller_red = seller_key(listing.get("seller"))
        need = self._wk_fix_need(mine, phrase, relation, seller_red, prop, tag)
        facts, rest = [], []
        for n in needs_l:
            if n not in mine:
                (facts if detail(n) else rest).append(n)
        if mine:
            rest.insert(needs_l.index(mine[0]), need)
        else:
            rest.append(need)
        answer["needs"] = rest
        if isinstance(now := answer.get("requirements_now"), list):
            stated = self.needs.written
            kept_now = [r for r in now if isinstance(r, str) and (r in stated or not (detail(r) or ours(r)))]
            answer["requirements_now"] = kept_now if need in kept_now else kept_now + [need]
        self._wk_strip(answer, "dropped", lambda d: inside_only(self._wk_name(d)) or detail(d))
        self._wk_strip(answer, "no_preference",
                       lambda p: inside_only(self._wk_name(p)) or loosely_same(self._wk_name(p), prop))
        self._wk_fix_entry(refs_l, prop, listing, line, seller_red, (relation, tol, unit))
        answer["references"] = refs_l
        answer["ref_needs"] = [*(answer.get("ref_needs") or []), need]
        self._wk_log("ref_line", need=need, property=prop, relation=relation)

    def _wk_fix_prop(self, needs_l: list, refs_l: list, outside: str, inside_only: Any, question: Any, /) -> str:
        """The property a pointing line names outside its phrase, by tiers (the most specific name of the first tier
        that has one); else the first reference, pointing need or lone asked property not named only inside."""
        asked = [*filter(None, map(self._wk_name, str(self.asking_label or "").split(";")))]
        listed = [str(r.get("property") or "") for r in refs_l]
        names = [*map(self._wk_name, needs_l)]
        pointed = [nm for nm, n in zip(names, needs_l) if _j_pointing(self._wk_value(n))]
        written = sorted(self.store_keys()) + [*map(self._wk_name, self.needs.written)]
        for tier in (listed, pointed, names, asked, written, self.ref_line_words(question or "", outside)):
            if named := [p for p in tier if p and mentions_prop(p, outside)]:
                return max(named, key=lambda p: (mentions_prop(p, outside, exact=True), len(_j_bits(p)), len(p)))
        for p in listed + pointed + (asked if len(asked) == 1 else []):
            if p and not inside_only(p):
                return p
        return ""

    def _wk_fix_tag(self, items: list, needs_l: list, prop: str, outside: str, question: Any, /) -> Any:
        if not items:
            return ""
        k = self.product_of(outside)
        if k is None:
            last = None
            for n in needs_l:
                if loosely_same(self._wk_name(n), prop):
                    found = _j_tags(self._wk_head(n))
                    if found:
                        last = found[-1]
            if last is not None:
                k = self.product_of("x (%s)" % last)
        if k is None and question:
            k = self.product_of(str(question))
        if k is None or not 0 <= k < len(items):
            return ""
        return items[k]["product"]

    def _wk_fix_need(self, mine: list, phrase: str, relation: str, seller_red: str, prop: str, tag: Any, /) -> str:
        for n in mine:
            value = self._wk_value(n)
            if _j_pointing(value) and ref_relation(value)[0] == relation and seller_red in seller_key(value):
                return n
        head = mine[0].split(":", 1)[0].strip() if mine else prop
        if tag and not (mine and _j_tags(head)):
            head = "%s (%s)" % (head, tag)
        return "%s: %s" % (head, phrase)

    @staticmethod
    def _wk_strip(answer: dict, key: str, drop: Any, /) -> None:
        if isinstance(answer.get(key), list):
            answer[key] = [x for x in answer[key] if not (isinstance(x, str) and drop(x))]

    def _wk_fix_entry(self, refs_l: list, prop: str, listing: dict, line: str, seller_red: str, rel: tuple, /) -> None:
        for entry in refs_l:
            if loosely_same(str(entry.get("property") or ""), prop):
                break
        else:
            entry = {"property": prop}
            refs_l.append(entry)
        current = str(entry.get("seller") or "")
        stale = not self.ref_seller_ok(current, line) or seller_key(current) != seller_red
        if stale:
            entry["seller"] = listing.get("seller")
            entry["kind"] = listing.get("kind") or ""
        if stale or not entry.get("details"):
            entry["details"] = list(listing.get("details") or [])
        if "kind" not in entry:
            entry["kind"] = listing.get("kind") or ""
        if rel[0] != "same" or not entry.get("relation"):
            entry["relation"], entry["tolerance"], entry["unit"] = rel

    # ------------------------------------------------------------------------------------------- price-filter learning
    def ceil_learn(self, call_args: Any, rows: Any, total: Any, observation: Any) -> None:
        args = call_args if isinstance(call_args, dict) else {}
        pkey = found[0] if (found := self.page_arg()) else None
        base = dict(kv for kv in args.items() if kv[0] != pkey)
        self.ceil_seen[_j_sig(base)] = int(total or 0)
        drows = [r for r in rows or [] if isinstance(r, dict)]
        if drows and found and found[1] and found[1] != "token" and isinstance(observation, dict):
            self._wk_depth_seen(args.get(pkey, found[2]), found[2], drows, total, observation)
        mode, paging = self.search_keys(args)[1::2]
        if (bound := self._wk_bound_arg(args, pkey, set(paging or ()))) is None or not drows:
            return
        labels = set(map(self.shelf_of, drows))
        shelf = self._wk_shelf_arg(args, pkey, labels)
        fixed = dict(pair for pair in (mode or ()) if pair[0] != shelf)
        sig = json.dumps([sorted(fixed.items()), bound], default=str)
        prices = [*map(self.row_cost, drows)]
        limit = args[bound] + 1e-9
        if any(p is not None and p > limit for p in prices):
            self.ceil_bad.add(sig)
            return
        if not shelf or len(labels) != 1 or "" in labels or sig in self.ceil_bad or any(p is None for p in prices):
            return
        template = self._wk_ceil_template(base, (bound, shelf), fixed, drows)
        words = len(" ".join(v for a, v in template.items() if isinstance(v, str) and a not in fixed).split())
        if (held := self.ceil_mode) is not None and held["sig"] not in self.ceil_bad and held["words"] < words:
            return
        spec = _j_mapped(_j_mapped(self._wk_search_schema(), "properties"), bound)
        self.ceil_mode = dict(sig=sig, tmpl=template, mode=fixed, shelf=shelf, bound=bound, words=words,
                              int=spec.get("type") == "integer")

    def _wk_search_schema(self) -> Any:
        return self.shapes.get(self.role("search") or "") or {}

    def _wk_depth_seen(self, at: Any, first: Any, drows: list, total: Any, observation: dict, /) -> None:
        nexts = []
        for key, v in observation.items():
            if isinstance(v, (list, dict)) or not re.search(r"next|more|onward|continu|cursor", str(key).lower()):
                continue
            nexts.append(v)
        size = max(len(drows), self.page_size())
        if not (numeric(at) and numeric(first) and nexts and len(drows) >= size):
            return
        # (no next page: every such field empty - None, false or "", a zero still counts as an answer)
        if any(v is not None and v is not False and v != "" for v in nexts):
            return
        reach = int((at - first + 1) * size)
        if numeric(total) and total > reach:
            self.page_cap = min(self.page_cap or reach, reach)

    @staticmethod
    def _wk_bound_arg(args: dict, pkey: Any, paging: set, /) -> Any:
        bounds = []
        for a, v in args.items():
            if not numeric(v) or a in paging or a == pkey or not v > 1:
                continue
            if re.search(r"page|size|count|result|min|low|from|start|offset", str(a).lower()) is None:
                bounds.append(a)
        return bounds[0] if len(bounds) == 1 else None

    @staticmethod
    def _wk_shelf_arg(args: dict, pkey: Any, labels: set, /) -> Any:
        hits = [a for a, v in args.items() if isinstance(v, str) and a != pkey and v.strip().lower() in labels and v.strip()]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _wk_ceil_template(base: dict, dropped: tuple, mode_keys: Any, drows: list, /) -> dict:
        template = {}
        for a, v in base.items():
            if a in dropped:
                continue
            if isinstance(v, str) and v.strip() and a not in mode_keys:
                want = v.strip().lower()
                for r in drows:
                    if not any(isinstance(x, str) and x.strip().lower() == want and not J_SHELFY.search(str(f).lower())
                               for f, x in r.items()):
                        break
                else:
                    continue
            template[a] = v
        return template


# ------------------------------------------------------------------------------------------------------- the loop (k)
K_NOT_RUN = "Not run (no step used): "
K_NOT_YET = "Not run yet (no step used): "
K_CART_STALE = K_NOT_RUN + ("an earlier cart change in this reply was not run; read why, then send the cart changes "
                            "again.")
K_ACTOR_TOKENS = 2500           # answer cap of one actor turn
K_SLAB = 16                     # store actions sent in one request at most
K_ROLES = ("cart", "cart_view", "message", "details", "search")
K_MARKS = re.compile(r"\(stated\)|\(revealed[^)]*\)")
K_CLOSED = r"[,;/]|\b(?:or|either|nor)\b"
K_SHELF_NAME = re.compile(r"categor|shelf|department|aisle")
K_ALNUM = re.compile(r"[a-z0-9]+")
K_BOUND_NAME = re.compile(r"ceil|max|price|cap$|budget|upper|most")
K_BOUND_NOT = re.compile(r"page|first|offset|skip|count|size|num")
K_PARENS = re.compile(r"\([^)]*\)")
K_OR_LIST = re.compile(r"\bor\b|\beither\b|/")
K_OR_SPLIT = re.compile(r"[,;/]|\bor\b|\beither\b")
K_AND_SPLIT = re.compile(r"[,;&+]|\band\b")
K_POSSESSIVE = re.compile(r"'s\b")
K_LINKS = re.compile(r"\b(?:%s)\b" % LINK_PHRASES)
K_VALUE_TOKEN = re.compile(r"\d+(?:[.,]\d+)+|[a-z0-9]+")
K_DIGIT = re.compile(r"\d")
K_LONG_WORD = re.compile(r"[a-z0-9]{3,}")
K_NOT_LIMIT = re.compile(r"page|size|count|result|min|low|from|start")
K_NUDGE = ("Every reply has to call one of the tools - search, open a listing, change the cart, order or decline. "
           "Go on with a tool call.")
K_LOWEST = ("For each product, choose the lowest-priced listing whose own fields meet every requirement in force; a fit "
            "that costs more needs every cheaper candidate opened and rejected first.")
K_BEFORE_ORDER = ("before the order goes out, confirm: 1) the cart holds exactly one line (one per product, each at its "
                  "quantity) and nothing else; 2) since the last cart change the stock and price were read again with a "
                  "tool that shows them - the item is in stock and price x quantity is within the budget; 3) its own "
                  "properties meet every stated requirement, every revealed need and every announced change; 4) if the "
                  "cheapest or the best was asked for, nothing better fits; 5) every rule of procedure in the request "
                  "was kept. Check whatever is still unchecked; otherwise call the order tool again.")
K_BEFORE_DECLINE = ("be sure before declining: several kinds of search were tried (words, filters, further pages); "
                    "every candidate's own properties were checked; each one fails a requirement, a revealed need, the "
                    "budget or the stock - and any spelling of a field's key counts. Buy what fits; otherwise call the "
                    "decline again.")


def _k_dicts(value, ceiling=6) -> list:
    found: list = []
    each_dict(value, found.append, ceiling)
    return found


def _k_err(error) -> str:
    return "%s: %s" % (type(error).__name__, error)


def _k_ceil(a, b) -> int:
    return int(-(-a // b))


def _k_money(x) -> str:
    if numeric(x) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _k_json(args) -> str:
    return json.dumps(args, sort_keys=True, default=str)


def _k_sig(name, args) -> str:
    return str(name) + as_text(args)


def _k_fn(proposal) -> dict:
    fn = proposal.get("function")
    return fn if isinstance(fn, dict) else {}


def _k_texts(args) -> list:
    return [v for v in args.values() if isinstance(v, str)]


def _k_top(book, key, value) -> None:
    book[key] = max(book.get(key, 0), value)


def _k_merge(ids, more) -> None:
    for i in more:
        if i not in ids:
            ids.append(i)


def _k_dict_of(holder, key) -> dict:
    value = holder.get(key)
    return value if isinstance(value, dict) else {}


def _k_list_of(holder, key) -> list:
    value = holder.get(key)
    return value if isinstance(value, list) else []


def _k_enum_of(spec) -> Any:
    return isinstance(spec, dict) and spec.get("enum")


def _k_first(pair) -> Any:
    return pair[0]


def _k_unique(forms) -> list:
    return [f for f in dict.fromkeys(forms) if f]


def _k_bump(book, key) -> Any:
    book[key] = book.get(key, 0) + 1
    return book[key]


class LoopMixin:
    """The episode loop, the code's cheapest-coverage engines (value searches, value pages, price-filter walks), and
    the path every proposed call takes: screening, sending, and booking what came back."""

    def _loop_state(self) -> None:
        # price-filter (ceiling) walks
        self.ceil_walked: set = set()
        self.ceil_aims: dict = {}
        self.ceil_aimed: dict = {}
        self.ceil_keys: set = set()
        self.ceil_seed_keys: set = set()
        self.ceil_seeded: set = set()
        # composed value + label searches and value pages
        self.label_tries: dict = {}
        self.label_done: dict = {}
        self.label_sets: dict = {}
        self.label_indexed: bool | None = None
        self.valued: set = set()
        self.valued_keys: dict = {}
        self.valued_by: dict = {}
        self.valued_opens: dict = {}
        self.valued_wait: dict = {}
        # guards and queued walks
        self.vguard_told: set = set()
        self.walk_queue: tuple | None = None
        self.walk_early = False
        # search-result notes
        self.short_told = False
        self.name_told = 0
        self.trap_told = 0
        self.page_sigs: dict = {}
        self.row_score: dict = {}
        self.hit_score: dict = {}
        self.all_hits: list = []
        self.hits_back = False
        # screens and booking
        self.dead_told: dict = {}
        self.cheaper_told: dict = {}
        self.pointer_told: dict = {}
        self.view_told: set = set()
        self.recheck_mark: tuple | None = None
        self.batch_puts: set = set()
        self.batch_cart = False
        self.line_read: dict = {}
        self.absorb_n = 0
        self.late_n = 0
        self.puts: dict = {}
        self.last_kind = ""
        self.early_audit = False
        # one-time holds of a decline
        self.decline_kept = False
        self.slack_done = False
        self.decline_fill = False
        self.decline_opens = False
        self.best_held = False
        self.ref_decline_held = False

    # ================================================================================================ the loop
    def _k_engines(self) -> tuple:
        # The code's own moves of one turn, highest priority first: (name, ready test or None, act). The first act
        # that returns True takes the turn.
        return (
            ("catch_up", None, self._k_catch_up),
            ("reply", lambda: bool(self.needs.reply_due), self._k_reply),
            ("opens", None, self.open_named),
            ("sweep", lambda: (not self.asking) or bool(self.empty_plans), self.sweep_step),
            ("value_pages", None, self.value_pages),
            ("question", lambda: bool(self.asking), self.pose_planned),
            ("references", None, self.ref_step),
            ("walk", None, self.walk_step),
        )

    def _k_catch_up(self) -> bool:
        self.catch_up()
        return False

    def _k_reply(self) -> bool:
        text = self.needs.reply_due
        self.needs.reply_due = ""
        return bool(self.reply_back(text))

    def execute(self) -> list:
        if not self.wire.done:
            self._k_open_chat()
            self._k_turns()
            wire = self.wire
            if not (wire.done or wire.broke):
                self.finish_now()
            self.log.append(dict(kind="end", finished=wire.done, steps=wire.tick, cost=round(wire.spend, 5),
                                 seconds=round(monotonic() - wire.t0, 1)))
        return self.log

    def _k_open_chat(self) -> None:
        self.read_request()
        self.log.append(dict(kind="plan", plan=self.needs.reading, roles=self.role_map))
        parts = ["THE SHOPPER'S REQUEST:\n" + str(self.request)]
        opening = self.pview.get("opening") if isinstance(self.pview, dict) else None
        if opening is not None:
            first_view = as_text(opening)
            self.archive.add(opening)
            parts.append("WHAT THE STORE SHOWED FIRST:\n" + first_view)
        overview = self.needs.overview(self.step_cap, self.msg_cap)
        parts += [overview] if overview else []
        parts.append(self.status_line())
        self.chat[:] = [dict(role="system", content=ACTOR_RULES), dict(role="user", content="\n\n".join(parts))]

    def _k_spent(self) -> bool:
        return self.wire.done or self.wire.broke or self.moves_left() <= 0

    def _k_over(self) -> bool:
        return self._k_spent() or self.wire.left() < Knobs.FINAL_S

    def _k_ladder(self) -> list:
        actors = list(LINEUP["actor"])
        if actors and self.stuck >= Knobs.STALLED_TURNS:
            actors.append(actors.pop(0))
        return actors

    def _k_gave_up(self, ladder, turn_ok, missed) -> Any:
        # No actor reply came: the loop ends when no model can be called, or after an outage of two failed turns (or
        # of one turn that tried two models).
        if not self.wire.usable(ladder, tools=True):
            return True
        if monotonic() - turn_ok > Knobs.OUTAGE_S:
            return missed >= 2 or (self.wire.down() and len(self.wire.call_models) >= 2)
        return False

    def _k_binds(self, proposals) -> bool:
        binding = set(filter(None, map(self.role, ("cart", "order", "decline"))))
        return any(p["function"]["name"] in binding for p in proposals)

    @staticmethod
    def _k_said(message) -> str:
        said = message.get("content") or ""
        return said if isinstance(said, str) else as_text(said)

    def _k_turns(self) -> None:
        idle_n = fails = seq_no = 0
        turn_ok = monotonic()
        missed = 0
        for _ in range(Knobs.ROUNDS):
            if self._k_over():
                break
            if any((ready is None or ready()) and act() for _, ready, act in self._k_engines()):
                continue
            ladder = self._k_ladder()
            message = self.wire.ask_model(self.chat, ladder, K_ACTOR_TOKENS, self.toolset)
            if message is None:
                missed += 1
                if self._k_gave_up(ladder, turn_ok, missed):
                    break
                fails += self.wire.call_fail in ("empty", "refused")
                if fails < 2:
                    continue
                break
            missed, turn_ok = 0, monotonic()
            proposals, seq_no = self.tool_calls(message, seq_no)
            said = self._k_said(message)
            if self.asking and proposals and self._k_binds(proposals) and self.pose_planned(urge=True):
                continue
            if proposals:
                idle_n = 0
                self._k_act_on(said, proposals)
                continue
            idle_n += 1
            self.chat.append(dict(role="assistant", content=said or "(no tool called)"))
            self.chat.append(dict(role="user", content=K_NUDGE + "\n" + self.status_line()))
            if idle_n >= Knobs.SILENT_TURNS:
                break

    def _k_act_on(self, said, proposals) -> None:
        self.log.append(dict(kind="model", content=said[:2000], calls=[p["function"] for p in proposals]))
        self.chat.append(dict(role="assistant", content=said or None, tool_calls=proposals))
        tick = self.wire.tick
        replies = self.dispatch(proposals)
        self.post_replies(replies)
        self.stuck = 0 if self.wire.tick != tick else self.stuck + 1

    def tool_calls(self, msg: dict, seq_no: int) -> tuple:
        calls = [c for c in msg.get("tool_calls") or [] if isinstance(c, dict)]
        proposals = [self._k_proposal(c, seq_no + n) for n, c in enumerate(calls, 1)]
        return proposals, seq_no + len(calls)

    @staticmethod
    def _k_proposal(call, n) -> dict:
        fn = call.get("function")
        fn = fn if isinstance(fn, dict) else {}
        given = fn.get("arguments")
        text = given if isinstance(given, str) else json.dumps(given or {})
        return dict(id=call.get("id") or "call_%d" % n, type="function", function=dict(name=fn.get("name"), arguments=text))

    # ================================================================================== small shared helpers
    def _k_bare(self, args) -> dict:
        found = self.page_arg()
        return {a: v for a, v in args.items() if not (found and a == found[0])}

    def _k_walk(self, args) -> tuple:
        # (key, walk) of this search's page-book entry (paging left out); the walk is None when not booked.
        key = _k_json(self._k_bare(args))
        return key, self.page_book.get(key)

    def _k_seen_total(self, args) -> Any:
        return self.ceil_seen.get(_k_json(self._k_bare(args)))

    @staticmethod
    def _k_rows_on(walk, home) -> list:
        return [i for i, lab in (walk.get("row_shelf") or {}).items() if lab in home]

    @staticmethod
    def _k_on_home(walk, home) -> bool:
        return any(lab in home for lab in (walk.get("row_shelf") or {}).values())

    @staticmethod
    def _k_whole(walk) -> bool:
        return len(walk.get("row_shelf") or {}) >= (walk.get("total") or 0)

    def _k_valued_key(self, k, key) -> None:
        self.valued_keys.setdefault(str(k), []).append(key)

    def _k_mine(self, head, k) -> bool:
        if not self.needs.items() or k is None:
            return True
        owner = self.product_of(head)
        return owner is None or owner == k

    def _k_to_last_tool(self, text, status=False) -> bool:
        if not self.chat or self.chat[-1].get("role") != "tool":
            return False
        extra = [t for t in (text, self.status_line() if status else "") if t]
        if not extra:
            return False
        body = self.chat[-1].get("content")
        body = body if isinstance(body, str) else ("" if body is None else str(body))
        self.chat[-1]["content"] = body + "\n" + "\n".join(extra)
        return True

    def _k_trim(self, queue, batch, floor) -> list:
        while queue and (1 if batch and len(queue) >= 2 else len(queue)) >= self.moves_left() - floor:
            queue.pop()
        return queue

    @staticmethod
    def _k_deal(pools, most) -> tuple:
        # Round-robin over (owner, ids) pools: (queue, id -> owner).
        queue: list = []
        owner: dict = {}
        while len(queue) < most and any(ids for _, ids in pools):
            for who, ids in pools:
                while ids and ids[0] in owner:
                    ids.pop(0)
                if ids and len(queue) < most:
                    taken = ids.pop(0)
                    queue.append(taken)
                    owner[taken] = who
        return queue, owner

    def _k_priced(self, ids) -> str:
        return ", ".join("%s (%s)" % (i, _k_money(self.archive.price(i))) for i in ids)

    def _k_ever_passed(self, ident) -> bool:
        return any(p.rsplit("|", 1)[0] == ident for p in self.passed)

    def _k_passed(self, ident) -> bool:
        return "%s|%s" % (ident, self.needs.version) in self.passed

    def _k_fresh_pass(self) -> bool:
        version = str(self.needs.version)
        return any(p.rsplit("|", 1)[-1] == version for p in self.passed)

    def _k_fail(self, kind, error, most=300) -> None:
        self.log.append(dict(kind=kind, error=_k_err(error)[:most]))

    @staticmethod
    def _k_quiet(act, fallback) -> Any:
        try:
            return act()
        except Exception:  # noqa: BLE001
            return fallback

    def _k_try(self, kind, act, fallback=None, most=300) -> Any:
        try:
            return act()
        except Exception as error:  # noqa: BLE001
            self._k_fail(kind, error, most)
            return fallback

    # ====================================================================== CG primitives: price-filter listing
    @staticmethod
    def row_cost(row: Any) -> float | None:
        if not isinstance(row, dict):
            return None
        for key, value in row.items():
            low = str(key).lower()
            if numeric(value) and ("price" in low or "cost" in low):
                return float(value)
        return None

    def ceil_spec(self) -> dict | None:
        spec = self.ceil_mode
        if spec and spec.get("sig") not in self.ceil_bad:
            return spec
        return None

    def cg_listing(self, args: Any) -> bool:
        spec = self.ceil_spec()
        if spec is None or not isinstance(args, dict):
            return False
        fixed = spec.get("mode") or ()
        pairs = list(fixed.items()) if isinstance(fixed, dict) else list(fixed)
        if pairs:
            return all(key in args and args[key] == value for key, value in pairs)
        return spec.get("bound") in args

    def page_size(self) -> int:
        sizes = [w.get("size") or 0 for w in self.page_book.values() if isinstance(w, dict)]
        return max(sizes) if sizes else 0

    def visible_depth(self) -> Any:
        found = self.page_arg()
        size = self.page_size()
        if found and size > 0 and found[1] != "token" and numeric(found[3]):
            first = found[2] if numeric(found[2]) else 0
            if found[1]:
                return int((found[3] - first + 1) * size)
            return int(found[3] - first + size)
        return self.page_cap

    def raw_label(self, labels: Any) -> str:
        if not labels:
            return ""
        ids = self.archive.ids
        tally: dict = {}
        for obs in self.archive.items:
            for d in _k_dicts(obs):
                if not any(isinstance(v, str) and v in ids and "id" in str(k).lower() for k, v in d.items()):
                    continue
                for key, v in d.items():
                    low = str(key).lower()
                    if K_SHELF_NAME.search(low) and "id" not in low and isinstance(v, str) and 1 <= len(v.strip()) <= 60:
                        spelled = v.strip()
                        if spelled.lower() in labels:
                            tally[spelled] = tally.get(spelled, 0) + 1
                        break
        if not tally:
            return ""
        return min(tally, key=lambda s: (-tally[s], s))

    def below(self, price: Any) -> Any:
        if not numeric(price):
            return None
        spec = self.ceil_spec()
        whole = float(price).is_integer()
        if (spec and spec.get("int")) or whole:
            limit = int(price) - 1 if whole else int(price)
        else:
            limit = round(price - 0.01, 2)
        return limit if limit > 0 else None

    def ceil_mid(self, a: Any, b: Any) -> Any:
        spec = self.ceil_spec()
        if (spec and spec.get("int")) or (float(a).is_integer() and float(b).is_integer()):
            return (int(a) + int(b)) // 2
        return round((a + b) / 2, 2)

    @staticmethod
    def ceil_args(spec: dict, label: str, bound: Any) -> dict:
        args = dict(spec.get("tmpl") or {})
        args[spec["shelf"]] = label
        args[spec["bound"]] = int(bound) if (spec.get("int") or float(bound).is_integer()) else bound
        return args

    def cg_products(self) -> list:
        n = len(self.needs.items())
        return list(range(n)) if n else [None]

    # ====================================================================== CG candidate tests and pools
    def ceiling_ready(self, k: Any) -> bool:
        if self.needs.items() and k is None:
            return False
        return (self.needs.priority() == "cheapest" and k not in self.ceil_walked and self.ceil_spec() is not None
                and self.page_arg() is not None and bool(self.raw_label(self.walk_home(k))))

    def ceil_aim(self, k: Any) -> Any:
        pick = self.walk_pick(k)
        held = self.ceil_aims.get(k)
        costs = ([pick[1]] if pick else []) + ([float(held)] if numeric(held) else [])
        if not costs and self.bound_need(k):
            seen = self._k_cheapest_seen(k)
            costs = [min(seen)] if seen else []
        return min(costs) if costs else None

    def _k_cheapest_seen(self, k) -> list:
        home = self.walk_home(k)
        seen = [cost for cost, _ in self.cheap_unread(self.home_rows(home), k)]
        gone = self.dead_lines()
        for ident in list(self.archive.read):
            if ident in gone or ident in self.refused_lines:
                continue
            if self.shelf_known(ident) in home:
                cost = self.archive.price(ident)
                if numeric(cost) and self.last_stock(ident) is not False:
                    seen.append(float(cost))
        return seen

    def bound_need(self, k: Any) -> bool:
        if k is None or not self.needs.items():
            return False
        return any(self._k_bounded(req, k) for req in self.needs.props_left())

    def _k_bounded(self, req, k) -> bool:
        head, _, value = str(req).partition(":")
        low = value.lower()
        return self.product_of(head) == k and bool(K_DIGIT.search(low)) and bool(LIMIT_WORDS.search(low))

    def home_rows(self, home: Any) -> list:
        out: list = []
        taken: set = set()
        for obs in self.archive.items:
            for d in _k_dicts(obs):
                mark = self.row_mark(d)
                if isinstance(mark, str) and mark in self.archive.ids and mark not in taken and self.shelf_of(d) in home:
                    taken.add(mark)
                    out.append(mark)
        return out

    def ceil_clash(self, home: Any, bound: Any) -> bool:
        ids = self.archive.ids
        for obs in self.archive.items:
            for d in _k_dicts(obs):
                if self.shelf_of(d) not in home:
                    continue
                if not any(isinstance(v, str) and v in ids and "id" in str(k).lower() for k, v in d.items()):
                    continue
                cost = self.row_cost(d)
                if cost is not None and cost <= bound and self.stock_state(d) is True:
                    return True
        return False

    def cheap_unread(self, ids: Any, k: Any, below: Any = None) -> list:
        if not ids:
            return []
        wanted = list(dict.fromkeys(ids))
        pool = {i for i in wanted if isinstance(i, str)}
        items = self.needs.items()
        qty = items[k]["quantity"] if isinstance(k, int) and 0 <= k < len(items) else self.needs.quantity()
        cap = self.needs.spend_cap
        sold: dict = {}
        for d in self._k_records():
            if any(isinstance(v, str) and v in pool and "id" in str(key).lower() for key, v in d.items()):
                state = self.stock_state(d)
                if state is not None:
                    sold[self.row_mark(d)] = not state
        out = []
        for ident in wanted:
            price = self._k_affordable(ident, sold, qty, cap, below)
            if price is not None:
                out.append((float(price), ident))
        return sorted(out, key=lambda pair: (pair[0], str(pair[1])))

    def _k_records(self) -> Any:
        for obs in self.archive.items:
            yield from _k_dicts(obs)

    def _k_affordable(self, ident, sold, qty, cap, below) -> Any:
        if ident in self.archive.read or ident in self.refused_lines or sold.get(ident):
            return None
        price = self.archive.price(ident)
        if not numeric(price):
            return None
        if numeric(cap) and numeric(qty) and price * qty > cap:
            return None
        return None if below is not None and price >= below else price

    def all_words(self) -> bool:
        return self.word_kind == "all" or (self.lead_words > 0 and self.word_kind != "any")

    def valued_ids(self, k: Any) -> set:
        if not self.all_words():
            return set()
        home = self.walk_home(k)
        out: set = set()
        for walk in self.page_book.values():
            args = walk.get("args") or {}
            if walk.get("id_order") is True and not self.cg_listing(args) and self.walk_carries(args, k):
                out.update(self._k_rows_on(walk, home))
        return out

    def valued_complete(self, k: Any) -> bool:
        return self._k_quiet(lambda: self._k_valued_complete(k), False)

    def _k_single_value(self, req, k) -> bool:
        head, _, value = str(req).partition(":")
        if not self._k_mine(head, k):
            return False
        parts = [p for p in re.split(K_CLOSED, K_MARKS.sub("", value).lower()) if p.strip()]
        return len(parts) == 1

    def _k_valued_complete(self, k) -> bool:
        if not self.all_words():
            return False
        lone = {n for n, req in enumerate(self.needs.props_left()) if self._k_single_value(req, k)}
        home = self.walk_home(k)
        for walk in self.page_book.values():
            args = walk.get("args") or {}
            if self._k_read_out(walk, args) and self._k_on_home(walk, home) and set(self.walk_carries(args, k)) & lone:
                return True
        return False

    def _k_read_out(self, walk, args) -> bool:
        # an id-ordered search (not the price filter) read to its last row
        if walk.get("id_order") is not True or not walk.get("first") or self.next_page(walk) is not None:
            return False
        return not self.cg_listing(args) and self._k_whole(walk)

    def row_carries(self, ident: Any, k: Any) -> bool:
        wants, figs = self._k_wanted_words(k)
        if not (wants or figs):
            return False
        text = " ".join(self._k_row_texts(ident))
        for fig, unit in figs:
            if re.search(r"(?<![\d.,])" + re.escape(str(fig)) + r"\s*-?\s*" + re.escape(self.stem(str(unit))), text):
                return True
        have = {self.stem(w) for w in K_LONG_WORD.findall(text)}
        return any(stems <= have for stems in wants)

    def _k_wanted_words(self, k) -> tuple:
        wants: list = []
        figs: list = []
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if not self._k_mine(head, k):
                continue
            low = K_MARKS.sub("", value).lower()
            if points_elsewhere(low):
                continue
            for part in re.split(K_CLOSED, low):
                amount = self.fig_amount(part)
                if amount:
                    figs.append(amount)
                elif not self.amount_part(part):
                    stems = {self.stem(w) for w in K_LONG_WORD.findall(part)}
                    if stems:
                        wants.append(stems)
        return wants, figs

    def _k_row_texts(self, ident) -> list:
        texts: list = []
        for d in self._k_records():
            if any("id" in str(key).lower() and v == ident for key, v in d.items()):
                texts += [v.lower() for v in d.values() if isinstance(v, str)]
        return texts

    # ====================================================================== EB: composed value + label searches
    def vl_room(self) -> tuple:
        left = max(0, self.ask_limit() - self.msgs_sent) if (self.asking and self.role("message")) else 0
        return left, self.moves_left() - self.order_due() - 2 * left

    def vl_values(self, k: Any) -> list:
        out: list = []
        for prop, value in self.code_values(k):
            alts = self._k_value_alts(prop, value)
            if alts and alts not in out:
                out.append(alts)
        return out

    def _k_value_alts(self, prop, value) -> list:
        low = K_PARENS.sub(" ", str(value).lower().replace("’", "'")).strip()
        if not low or points_elsewhere(low):
            return []
        if K_OR_LIST.search(low):
            pieces = [p for p in K_OR_SPLIT.split(low) if p.strip()][:3]
        else:
            pieces = K_AND_SPLIT.split(low)[:1]
        names = set(K_ALNUM.findall(str(prop).lower()))
        alts: list = []
        for piece in pieces:
            words = self._k_piece_words(piece, names)
            if words and words not in alts:
                alts.append(words)
        return alts

    def _k_piece_words(self, piece, names) -> list:
        amount = self.fig_amount(piece)
        if EXCLUSION.search(piece) or (self.amount_part(piece) and not amount):
            return []
        body = K_LINKS.sub(" ", K_POSSESSIVE.sub("", piece))
        fig, unit = (str(amount[0]), str(amount[1])) if amount else (None, None)
        return [tok for tok in K_VALUE_TOKEN.findall(body) if self._k_token_kept(tok, fig, unit, names)]

    @staticmethod
    def _k_token_kept(tok, fig, unit, names) -> bool:
        if len(tok) < 2 and tok != fig:
            return False
        if "." in tok or "," in tok or tok in FILLER:
            return False
        return tok not in names or tok == unit

    def vl_words(self, k: Any, label: str) -> list:
        words = self._k_label_words(label)
        if self.label_indexed is not False:
            return words
        home = self.walk_home(k)
        titles = [{self.stem(w) for w in K_ALNUM.findall(str(self.title_of(d)).lower())}
                  for d in self._k_records() if self.row_mark(d) in self.archive.ids and self.shelf_of(d) in home]
        kept = [w for w in words if 2 * sum(1 for t in titles if self.stem(w) in t) >= len(titles)] if titles else []
        return kept or self._k_kind_word(k)

    @staticmethod
    def _k_label_words(label) -> list:
        low = K_POSSESSIVE.sub("", str(label).lower().replace("’", "'"))
        return list(dict.fromkeys(t for t in K_ALNUM.findall(low) if len(t) >= 2 and t not in FILLER))

    def _k_kind_word(self, k) -> list:
        items = self.needs.items()
        if isinstance(k, int) and 0 <= k < len(items):
            name = items[k]["product"]
        else:
            name = str(self.needs.reading.get("product") or "")
        kinds = [w for w in K_ALNUM.findall(str(name).lower()) if len(w) > 2 and w not in FILLER]
        return kinds[-1:]

    def vl_queries(self, k: Any, label: str, cap_len: Any) -> tuple:
        vals = self.vl_values(k)
        if not vals:
            return [], []
        tail = [self.vl_words(k, label)]
        width = None if self.needs.priority() == "cheapest" else 1
        lone = _k_unique([self._k_compose([a] + tail, cap_len) for a in vals[0][:width]])
        if len(vals) < 2:
            return lone, []
        j = next((i for i, v in enumerate(vals) if len(v[:width]) > 1), 0)
        mixed = [self._k_compose([a if i == j else v[0] for i, v in enumerate(vals)] + tail, cap_len)
                 for a in vals[j][:width]]
        return _k_unique(mixed), lone

    def _k_compose(self, parts, cap_len) -> str:
        # each stem once, in order; cut to the lead words and the text field's length
        used: set = set()
        kept: list = []
        for w in (w for words in parts for w in words):
            s = self.stem(w)
            if s not in used:
                used.add(s)
                kept.append(w)
        if self.lead_words:
            kept = kept[:self.lead_words]
        return self._k_fit_len(" ".join(kept), cap_len)

    @staticmethod
    def _k_fit_len(text, cap_len) -> str:
        if not numeric(cap_len):
            return text
        while cap_len < len(text) and " " in text:
            text = text.rsplit(" ", 1)[0]
        return text[:int(cap_len)]

    def vl_next(self, first: list, second: list, tkey: str, mode: Any) -> Any:
        for forms in (first, second):
            if not forms:
                continue
            totals = [self.vl_seen(f, tkey, mode) for f in forms]
            for form, total in zip(forms, totals):
                if total is None:
                    return form
            if any(total > 0 for total in totals):
                return ""
        return None if (first or second) else ""

    def vl_mode(self, args: dict) -> tuple:
        schema = self.shapes.get(self.role("search")) or {}
        props = _k_dict_of(schema, "properties")
        out = []
        for key, v in args.items():
            spec = props.get(key)
            if isinstance(v, str) and isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) <= 4:
                out.append((key, v))
        return tuple(sorted(out))

    def vl_set(self, text: Any) -> frozenset:
        toks = re.findall(r"[a-z0-9]+", str(text).lower())
        if self.lead_words:
            toks = toks[:self.lead_words]
        return frozenset(self.stem(t) for t in toks)

    def vl_seen(self, text: str, tkey: str, mode: Any) -> Any:
        want = self.vl_set(text)
        pairs = list(mode.items()) if isinstance(mode, dict) else list(mode)
        for key, total in self.ceil_seen.items():
            try:
                args = json.loads(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(args, dict) or not isinstance(args.get(tkey), str):
                continue
            if any(args.get(mk) != mv for mk, mv in pairs):
                continue
            if self.vl_set(args[tkey]) == want:
                return int(total)
        return None

    def vl_indexed(self, label: str) -> None:
        if self.label_indexed is not None or not self.archive.items:
            return
        words = {self.stem(t) for t in self._k_label_words(label)}
        target = str(label).lower().strip()
        hits = 0
        for d in _k_dicts(self.archive.items[-1], 3):
            if self.row_mark(d) not in self.archive.ids or self.shelf_of(d) != target:
                continue
            other: set = set()
            for key, v in d.items():
                if isinstance(v, str) and not K_SHELF_NAME.search(str(key).lower()):
                    other.update(self.stem(w) for w in re.findall(r"[a-z0-9]+", v.lower()))
            if any(w not in other for w in words):
                hits += 1
        if hits >= 2:
            self.label_indexed = True

    def vl_queue(self, k: Any, ids: Any, floor: Any, most: int) -> list:
        batch = self.batch_tool()
        first = batch[2] if batch else most
        if self.needs.priority() != "cheapest":
            return self._k_trim(self._k_cheap_ids(ids, k, None)[:max(0, min(first, most))], batch, floor)
        pick = self.walk_pick(k)
        queue = self._k_cheap_ids(ids, k, pick)[:max(0, min(first, Knobs.CAP_ROWS - self.valued_opens.get(k, 0)))]
        self._k_trim(queue, batch, floor)
        self.valued_opens[k] = self.valued_opens.get(k, 0) + len(queue)
        return queue

    def _k_cheap_ids(self, ids, k, pick) -> list:
        return [i for _, i in self.cheap_unread(ids, k, below=pick[1] if pick else None)]

    def vl_launch(self, queue: list, homes: Any, floor: Any, say: str) -> None:
        for mark in list(self.rank_checked):
            if self.shelf_known(str(mark).split("|", 1)[0]) in homes:
                self.rank_checked.discard(mark)
        self.order_vetted = False
        self.open_queue = list(queue)
        self.open_recheck = True
        self.open_say = say
        self.open_purpose = "walk"
        self.open_floor = floor
        self.open_named()

    def vl_search(self) -> bool:
        if self.vl_room()[1] <= 4:
            return False
        schema = self.shapes.get(self.role("search")) or {}
        props = _k_dict_of(schema, "properties")
        state = (self.page_arg(), self.needs.priority() == "cheapest")
        products = self.cg_products()
        for k in products:
            plan = self._k_label_plan(k, schema, props)
            if plan is not None:
                self._k_label_send(k, plan, state, len(products) > 1)
                return True
        return False

    def _k_text_template(self, k) -> tuple | None:
        # The first code query of the product whose free text sits in exactly one argument: (call, that key).
        for call in self.code_queries(k):
            free = self.free_text(call)
            keys = [a for a, v in call.items() if isinstance(v, str) and v.strip() and v in free]
            if keys[1:] or not keys or self.cg_listing(call):
                continue
            return call, keys[0]
        return None

    def _k_forms(self, k, label, cap_len, sets) -> tuple:
        first, second = self.vl_queries(k, label, cap_len)
        sets.update(map(self.vl_set, first + second))
        return first, second

    def _k_label_plan(self, k, schema, props) -> tuple | None:
        # The value + label search to send for product k, or None (the reasons it is settled are recorded).
        if self.label_tries.get(k, 0) >= Knobs.LABEL_SEARCHES:
            return None
        sig = json.dumps(self.vl_values(k))
        settled = self.label_done.setdefault(str(k), set())
        if sig in settled:
            return None
        home = self.walk_home(k)
        label = self.raw_label(home)
        if not label:
            return None
        picked = self._k_text_template(k)
        if picked is None:
            settled.add(sig)
            return None
        template, tkey = picked
        mode = tuple(sorted((a, v) for a, v in template.items() if _k_enum_of(props.get(a))))
        keep = {tkey, *(a for a, _ in mode), *_k_list_of(schema, "required")}
        template = {a: template[a] for a in template if a in keep}
        cap_len = self._k_max_len(props.get(tkey))
        sets = self.label_sets.setdefault(str(k), set())
        forms = self._k_forms(k, label, cap_len, sets)
        todo = self.vl_next(*forms, tkey, mode)
        if todo is None:
            todo = ""
            if self.label_indexed is None and self._k_carried(k, home):
                self.label_indexed = False
                forms = self._k_forms(k, label, cap_len, sets)
                todo = self.vl_next(*forms, tkey, mode) or ""
        if todo:
            args, faults = ArgFixer.fix(dict(template, **{tkey: todo}), schema)
            if not faults and isinstance(args, dict):
                return args, todo, (tkey, mode), forms, home, label
        settled.add(sig)
        return None

    @staticmethod
    def _k_max_len(spec) -> Any:
        width = spec.get("maxLength") if isinstance(spec, dict) else None
        return width if numeric(width) else None

    def _k_carried(self, k, home) -> bool:
        return any(self.walk_carries(w.get("args") or {}, k) and self._k_on_home(w, home) for w in self.page_book.values())

    def _k_label_send(self, k, plan, state, several) -> None:
        args, todo, (tkey, mode), (first, second), home, label = plan
        found, cheapest = state
        _k_bump(self.label_tries, k)
        plural = "s" if len(self.vl_values(k)) >= 2 else ""
        self.walk_call(args, "A keyword search matches values, titles and categories alike, so I search the "
                             "required value%s together with the category's name as the store writes it: '%s'."
                       % (plural, todo))
        total = self.vl_seen(todo, tkey, mode)
        self.vl_indexed(label)
        key, walk = self._k_walk(args)
        every = bool(walk) and self._k_whole(walk) and self._k_on_home(walk, home)
        floor = self.order_due() + 2
        note = "" if total is None else self._k_value_head(todo, total, every)
        most = None
        if cheapest and every and found is not None and walk.get("id_order") is not False:
            self.valued.add(key)
            self._k_valued_key(k, key)
            if several:
                self.valued_wait[k] = True
            else:
                most = Knobs.CAP_ROWS
        elif not cheapest and every and todo in (first or second):
            most = Knobs.LABEL_ROWS
        queue = [] if most is None else self.vl_queue(k, self._k_rows_on(walk, home), floor, most)
        if queue:
            below_pick = " below the pick" if (cheapest and self.walk_pick(k)) else ""
            note += (" The unread in-stock rows within the budget on the requested category%s, cheapest first, "
                     "are opened next: %s." % (below_pick, self._k_priced(queue)))
        if note:
            self._k_to_last_tool(note, status=not queue)
        self.log.append(dict(kind="label_search", product=k, query=todo, total=total, every=every, opened=queue))
        if queue:
            self.vl_launch(queue, home, floor, "Every listing of this category that carries the required values "
                                               "is among that search's matches, so I open its cheapest unread rows.")

    @staticmethod
    def _k_value_head(todo, total, every) -> str:
        head = "[value search] '%s': " % todo
        if total == 0:
            return head + "no listing matched."
        return head + "%d listing%s matched%s." % (total, "" if total == 1 else "s",
                                                   " - every match is shown here" if every else "")

    def vl_flush(self, floor: Any) -> bool:
        ks = [k for k, waiting in self.valued_wait.items() if waiting]
        self.valued_wait.clear()
        if not ks:
            return False
        per = 2 if self.moves_left() - floor - 1 >= 2 * len(ks) else 1
        batch = self.batch_tool()
        pools = []
        for k in ks:
            home = self.walk_home(k)
            ids: list = []
            for key in self.valued_keys.get(str(k), []):
                walk = self.page_book.get(key)
                if walk is None or not self.walk_carries(walk.get("args") or {}, k):
                    continue
                ids.extend(i for i in self._k_rows_on(walk, home) if i not in ids)
            pick = self.walk_pick(k)
            ranked = [i for _, i in self.cheap_unread(ids, k, below=pick[1] if pick else None)]
            pools.append((k, ranked[:max(0, min(per, Knobs.CAP_ROWS - self.valued_opens.get(k, 0)))]))
        most = batch[2] if batch else per * len(ks)
        queue, owner = self._k_deal(pools, most)
        self._k_trim(queue, batch, floor)
        if not queue:
            return False
        for ident in queue:
            self.valued_opens[owner[ident]] = self.valued_opens.get(owner[ident], 0) + 1
        homes: set = set()
        for k in ks:
            homes |= set(self.walk_home(k))
        self.vl_launch(queue, homes, floor, "The shopper wants the cheapest, so I open the cheapest unread rows that "
                                            "each product's value search showed.")
        self._k_to_last_tool("[value searches read] opened: %s. %s" % (self._k_priced(queue), K_LOWEST))
        self.log.append({"kind": "value_flush", "products": ks, "opened": queue})
        return True

    def product_steps(self) -> str:
        try:
            items = self.needs.items()
            if len(items) < 2 or self.needs.priority() != "cheapest":
                return ""
            parts = []
            waiting = ""
            for k, item in enumerate(items):
                pick = self.walk_pick(k)
                n = self.valued_by.get(k, 0)
                parts.append("%s: %d value page%s read, pick %s" % (item["product"], n, "" if n == 1 else "s",
                                                                    "yes" if pick else "no"))
                if not pick and not waiting:
                    waiting = str(item["product"])
            if waiting:
                parts.append("next: %s" % waiting)
            return "\n[product steps] " + "; ".join(parts) + "."
        except Exception:  # noqa: BLE001
            return ""

    # ====================================================================== CG2: value-led searches to their end
    def value_pages(self) -> bool:
        return self._k_try("value_pages_error", self._k_value_pages, False)

    def _k_pages_blocked(self) -> bool:
        if any(r.get("status") == "open" for r in self.refs) and self.ref_room():
            return True
        wire = self.wire
        return bool(self.orders_tried or self.refusal_n or wire.done or wire.broke) or (
            wire.left() < Knobs.WIND_DOWN_S + 60 or not self.role("search"))

    def _k_value_pages(self) -> bool:
        if self._k_pages_blocked():
            return False
        if self.vl_search():
            return True
        if self.page_arg() is None or self.word_kind == "any" or self.needs.priority() != "cheapest":
            return False
        products = self.cg_products()
        several = len(products) > 1
        depth = self.visible_depth()
        per = Knobs.VALUE_PAGES_ONCE
        if depth:
            per = max(per, _k_ceil(depth, max(1, self.page_size())) - 1)
        lim = dict(depth=depth, per=per, several=several, known=self.all_words(), floor=self.order_due() + 2,
                   cap=max(Knobs.VALUE_PAGES_ALL, Knobs.VALUE_PAGES_EACH * len(products)),
                   opens=1 if self.batch_tool() else (2 if several else Knobs.CAP_ROWS))
        left, room = self.vl_room()
        homes = {k: self.walk_home(k) for k in products}
        fresh = len([k for k in products if homes[k] and not self.valued_by.get(k, 0)])
        lim.update(left=left, room=room, share=max(4, room // max(1, fresh)))
        options = []
        pending = False
        for n, k in enumerate(products):
            if not homes[k]:
                continue
            found, waits = self._k_page_cands(k, homes[k], lim)
            pending = pending or waits
            options.extend(((used, take, neg, n, key), k) for used, take, neg, key in found)
        if options:
            rank, k = min(options, key=_k_first)
            self.valued.add(rank[4])
            return self.read_value_search(k, rank[4], lim["floor"], rank[1], defer=several)
        waiting = len([k for k, flag in self.valued_wait.items() if flag])
        if not waiting or pending:
            return False
        if left > 0 and room > lim["opens"] * waiting + 4:
            return False
        return self.vl_flush(lim["floor"])

    def _k_query_set(self, args) -> frozenset:
        return self.vl_set(" ".join(self.free_text(args)))

    def _k_stem_set(self, text) -> set:
        return {self.stem(w) for w in K_ALNUM.findall(text.lower()) if len(w) >= 2}

    def _k_plain_words(self, k, home) -> set:
        # Stems of product k's required values and of its category labels.
        plain: set = set()
        for req in self.needs.props_left():
            head, _, value = str(req).partition(":")
            if self._k_mine(head, k):
                plain |= self._k_stem_set(value)
        for lab in home:
            plain |= self._k_stem_set(str(lab))
        return plain

    def _k_whole_reads(self, k) -> list:
        # (query stems, value pairs, mode) of every id-ordered search already read to its end.
        out = []
        for walk in self.page_book.values():
            args = walk.get("args") or {}
            if walk.get("id_order") is not True or not walk.get("first") or self.next_page(walk) is not None:
                continue
            if self.cg_listing(args) or not self._k_whole(walk):
                continue
            out.append((self._k_query_set(args), set(self.walk_carries(args, k, alts=True)), self.vl_mode(args)))
        return out

    def _k_unread_walk(self, walk, args, k, home) -> bool:
        if walk.get("id_order") is not True or not walk.get("first") or self.next_page(walk) is None:
            return False
        if self.cg_listing(args) or not self.walk_carries(args, k):
            return False
        return self._k_on_home(walk, home)

    @staticmethod
    def _k_too_deep(total, depth, known, composed) -> bool:
        if depth is None:
            return not known
        return depth < total and (not known or not composed)

    @staticmethod
    def _k_covered(whole, words, plain, pairs, how) -> bool:
        for w, p, m in whole:
            if m == how and (w <= words or (w <= plain and pairs <= p)):
                return True
        return False

    @staticmethod
    def _k_unnested(cands, plain) -> list:
        # Drops a candidate whose query words nest with another's beyond the plain value and label words.
        def nested(c: tuple) -> bool:
            for o in cands:
                if o is c:
                    continue
                mine, other = c[3], o[3]
                if other < mine and not (mine - other) <= plain:
                    return True
                if mine < other and (other - mine) <= plain:
                    return True
            return False
        return [c for c in cands if not nested(c)]

    def _k_page_cands(self, k, home, lim) -> tuple:
        # Unfinished value-led searches of product k worth reading on: ([(used, take, -score, key)], deferred).
        mine = self.label_sets.get(str(k), set())
        plain = self._k_plain_words(k, home)
        whole = self._k_whole_reads(k)
        used_k = self.valued_by.get(k, 0)
        opened_k = self.valued_opens.get(k, 0)
        depth, known, opens = lim["depth"], lim["known"], lim["opens"]
        cands = []
        pending = False
        for key, walk in self.page_book.items():
            if key in self.valued:
                continue
            args = walk.get("args") or {}
            if not self._k_unread_walk(walk, args, k, home):
                continue
            words = self._k_query_set(args)
            composed = words in mine
            total = walk.get("total") or 0
            if self._k_too_deep(total, depth, known, composed):
                continue
            if known and self._k_covered(whole, words, plain, set(self.walk_carries(args, k, alts=True)),
                                         self.vl_mode(args)):
                continue
            reach = total if depth is None else min(total, depth)
            pages = _k_ceil(reach, max(1, walk.get("size") or 0)) - len(walk.get("read") or ())
            allowed = min(lim["per"] - used_k, lim["cap"] - sum(self.valued_by.values()))
            if lim["several"]:
                allowed = min(allowed, lim["share"] - used_k - opened_k - (0 if opened_k else opens))
            take = pages if pages <= allowed else (allowed if composed else 0)
            if take <= 0 or self.moves_left() - take - opens <= lim["floor"]:
                continue
            if (take > 2 or lim["several"]) and lim["left"] > 0 and lim["room"] > take + opens + 4:
                pending = True
                continue
            cands.append((take, -self.walk_score(args, k), key, words))
        if known:
            cands = self._k_unnested(cands, plain)
        return [(used_k, take, neg, key) for take, neg, key, _ in cands], pending

    def read_value_search(self, k: Any, key: str, floor: Any, most: int = Knobs.VALUE_PAGES_ONCE,
                          defer: bool = False) -> bool:
        found = self.page_arg()
        walk = self.page_book.get(key)
        if found is None or walk is None:
            return False
        args = walk.get("args") or {}
        query = " ".join(self.free_text(args))
        known = self.all_words()
        say = ("The shopper wants the cheapest and this store lists results in a fixed id order, not by price%s, so I "
               "read the next page of '%s' to see every listing %s." % (
                   " (its keyword search needs every word)" if known else "", query,
                   "carrying these words" if known else "it matched"))
        got = 0
        tried = False
        while (self.next_page(walk) is not None and got < most and self.moves_left() > floor + 1
               and not (self.wire.done or self.wire.broke) and self.wire.left() >= Knobs.WIND_DOWN_S + 30):
            tried = True
            self.valued_by[k] = self.valued_by.get(k, 0) + 1
            nxt = self.next_page(walk)
            if not self.walk_call(dict(args, **{found[0]: nxt}), say):
                break
            got += 1
            walk = self.page_book.get(key, walk)
            if self.next_page(walk) == nxt:
                break
        if got == 0:
            return tried
        self.valued_keys.setdefault(str(k), []).append(key)
        seen = len(walk.get("row_shelf") or {})
        total = walk.get("total") or 0
        complete = self.next_page(walk) is None and seen >= total
        head = "[value search read] '%s': %d of %d matches seen" % (query, seen, total)
        if complete:
            head += " - every listing %s is among them" % ("carrying these words" if known else "it matched")
        elif self.next_page(walk) is None:
            head += " - the rest cannot be reached, so a cheaper fit may lie past the reachable rows"
        queue: list = []
        if defer:
            self.valued_wait[k] = True
            summary = (head + ". Its cheapest unread in-stock rows within the budget are opened once every product's "
                              "value search is read. " + K_LOWEST)
        else:
            home = self.walk_home(k)
            queue = self.vl_queue(k, self._k_rows_on(walk, home), floor, Knobs.CAP_ROWS)
            if queue:
                below_pick = " below the pick" if self.walk_pick(k) else ""
                summary = (head + ". Its unread in-stock rows within the budget%s, cheapest first, are opened next: "
                                  "%s. " % (below_pick, self._k_priced(queue)) + K_LOWEST)
                self.vl_launch(queue, home, floor, "That search is read to its end, so I open its cheapest unread "
                                                   "rows.")
            else:
                summary = (head + ". No unread in-stock row within the budget on the category is left among them. "
                           + K_LOWEST)
        self._k_to_last_tool(summary, status=not queue)
        self.log.append({"kind": "value_pages", "product": k, "query": query, "pages": got, "complete": complete,
                         "opened": queue, "defer": defer})
        return True

    # ============================================================= CG3: ceiling walk, sweep source, seed, re-arm, scheduler
    def ceiling_walk(self, k: Any, floor: Any, repair: bool = False) -> tuple | None:
        spec, found, home = self.ceil_spec(), self.page_arg(), self.walk_home(k)
        label, aim = self.raw_label(home), self.ceil_aim(k)
        if spec is None or found is None or aim is None or not label:
            return None
        self.ceil_walked.add(k)
        pick_now = self.walk_pick(k)
        self.ceil_aimed[k] = (float(aim), pick_now[0] if pick_now else "")
        bound = self.below(aim)
        if bound is None:
            return None
        if repair:
            floor += 2
        elif not self.orders_tried:
            floor = max(floor, self.order_due() + 2)
        what = "the pick" if (pick_now or numeric(self.ceil_aims.get(k))) else "the cheapest row seen"
        say = ("The shopper wants the cheapest, so I list '%s' priced below %s (%s) with the store's price filter."
               % (label, what, _k_money(aim)))
        walked, calls, complete, last = self._k_ceil_loop(spec, (found, label, home), bound, floor, say)
        if not walked:
            return None
        ids: list = []
        for args in walked:
            walk = self._k_walk(args)[1]
            if walk is not None:
                _k_merge(ids, self._k_rows_on(walk, home))
        cands = [i for _, i in self.cheap_unread(ids, k, below=aim)]
        valued = self.valued_ids(k)
        rest = [i for i in cands if i not in valued]
        texty = [i for i in rest if self.row_carries(i, k)]
        ranked = [i for i in cands if i in valued] + texty
        if not self.valued_complete(k):
            ranked += [i for i in rest if i not in texty]
        if not complete:
            report = ("below %s, %d of the cheaper listings on '%s' were seen, %d of them unread, in stock and within "
                      "budget" % (_k_money(aim), len(ids), label, len(cands)))
        elif ids:
            report = ("the price filter on '%s' saw every listing priced up to %s: %d rows, %d of them unread, in "
                      "stock and within budget" % (label, _k_money(last), len(ids), len(cands)))
        else:
            report = ("the price filter on '%s' below %s shows no listing that costs less than %s"
                      % (label, _k_money(aim), what))
        self.log.append(dict(kind="ceiling_walk", product=k, label=label, aim=aim, calls=calls, complete=complete,
                             rows=len(ids), ranked=ranked[:Knobs.CAP_ROWS]))
        return report, ranked[:Knobs.CAP_ROWS]

    def _k_ceil_live(self, floor) -> bool:
        wire = self.wire
        return floor + 1 < self.moves_left() and not (wire.done or wire.broke) and wire.left() >= Knobs.WIND_DOWN_S + 30

    def _k_ceil_fetch(self, args, say) -> tuple:
        # (sent or already listed, the page-book walk after it)
        walk = self._k_walk(args)[1]
        if walk is not None and walk.get("first") and self._k_seen_total(args) is not None:
            return True, walk
        if not self.walk_call(args, say):
            return False, None
        return True, self._k_walk(args)[1]

    def _k_filter_leaks(self, walk, bound) -> bool:
        price = self.archive.price
        return any(numeric(price(i)) and bound < price(i) for i in (walk.get("row_shelf") or {}))

    def _k_ceil_loop(self, spec, where, bound, floor, say) -> tuple:
        found, label, home = where
        depth = self.visible_depth()
        walked: list = []
        calls = 0
        top = None
        complete = False
        last = bound
        while calls < Knobs.CAP_CALLS and self._k_ceil_live(floor):
            args = self.ceil_args(spec, label, bound)
            self.ceil_keys.add(_k_json(self._k_bare(args)))
            calls += 1
            sent, walk = self._k_ceil_fetch(args, say)
            if not sent:
                break
            last = bound
            walked.append(args)
            if walk is not None and self._k_filter_leaks(walk, bound):
                self.ceil_bad.add(spec.get("sig"))
            total = None if self.ceil_spec() is None else self._k_seen_total(args)
            if total is None:
                break
            if total == 0 or walk is None:
                if self.ceil_clash(home, bound):
                    self.ceil_bad.add(spec.get("sig"))
                    break
                complete = top is None
                nb = None if complete else self.ceil_mid(bound, top)
                if nb is None or not bound < nb < top:
                    break
                bound = nb
                continue
            size = max(1, walk.get("size") or 0)
            unread = _k_ceil(total, size) - len(walk.get("read") or ())
            if (depth is None or total <= depth) and unread <= Knobs.CAP_CALLS - calls:
                calls, walk = self._k_ceil_pages(args, walk, found[0], calls, floor)
                complete = walk is not None and self.next_page(walk) is None
                break
            top = bound
            nb = self._k_ceil_lower(walk, bound, size * max(1, Knobs.CAP_CALLS - calls), total)
            if not nb < bound:
                break
            bound = nb
        return walked, calls, complete, last

    def _k_ceil_pages(self, args, walk, page_key, calls, floor) -> tuple:
        while self.next_page(walk) is not None and calls < Knobs.CAP_CALLS and self._k_ceil_live(floor):
            calls += 1
            nxt = self.next_page(walk)
            if not self.walk_call(dict(args, **{page_key: nxt}), "I read the next page of that price-filter listing."):
                break
            walk = self._k_walk(args)[1]
            if walk is None or self.next_page(walk) == nxt:
                break
        return calls, walk

    def _k_ceil_lower(self, walk, bound, fill, total) -> Any:
        costs = [p for p in map(self.archive.price, walk.get("row_shelf") or {}) if numeric(p)]
        low = min(costs) if costs else 0.0
        return min(self.ceil_mid(low, bound),
                   self.ceil_mid(low, low + (bound - low) * 1.8 * fill / max(total, 1)))

    def ceiling_sweep(self, tried: Any) -> tuple:
        spec, found = self.ceil_spec(), self.page_arg()
        if spec is None or found is None or self.needs.items():
            return None, False
        rows, shelves = self.shelf_rows(), self.shelf_labels()
        label = self.raw_label(shelves)
        if not (rows and label):
            return None, False
        bound = self.below(min(r[0] for r in rows.values()))
        if bound is None:
            return None, True
        args = self.ceil_args(spec, label, bound)
        if _k_json(args) not in tried:
            self.ceil_keys.add(_k_json(self._k_bare(args)))
            return args, False
        walk = self._k_walk(args)[1]
        if walk is None:
            return None, self._k_seen_total(args) == 0 and not self.ceil_clash(shelves, bound)
        nxt = self.next_page(walk)
        if nxt is None:
            return None, self._k_whole(walk)
        more = dict(args, **{found[0]: nxt})
        return (None, False) if _k_json(more) in tried else (more, False)

    def rearm_walk(self) -> Any:
        wire = self.wire
        if self.needs.priority() != "cheapest" or wire.done or wire.left() < Knobs.WIND_DOWN_S + 60:
            return None
        if not self.role("search"):
            return None
        gone = set(self.dead_lines()) | set(self.refused_lines)
        opens = 1 if self.batch_tool() else 2
        for k, (aim, ident) in list(self.ceil_aimed.items()):
            pick = self._k_rearm_pick(k, aim, ident, gone, opens)
            if pick:
                self.ceil_aimed.pop(k, None)
                self.ceil_aims.pop(k, None)
                self.log.append(dict(kind="ceil_rearm", product=k, dead=ident, aim=pick[1]))
                return self.run_walk([], Knobs.LATE_STEPS, [k], repair=True)
        return None

    def _k_rearm_pick(self, k, aim, ident, gone, opens) -> Any:
        # the walk's pick is gone and the product's best is dearer than the aim: the walk runs again
        if not ident or ident not in gone or self.ceil_spec() is None or self.page_arg() is None:
            return None
        pick = self.walk_pick(k)
        if not pick or pick[1] <= aim + 1e-9 or self.moves_left() < Knobs.LATE_STEPS + 3 + opens:
            return None
        return pick

    def walk_step(self) -> bool:
        return self._k_try("walk_error", self._k_walk_step, False)

    def _k_ceil_due(self, k) -> bool:
        return bool((self.walk_pick(k) or self.bound_need(k)) and self.ceiling_ready(k) and self.ceil_aim(k) is not None)

    def _k_ceil_rank(self, k) -> tuple:
        return bool(self.walk_pick(k)), self.valued_by.get(k, 0), -1 if k is None else k

    def _k_walk_step(self) -> bool:
        if self.decline_queue:
            return self.run_decline_reads()
        queued = self.walk_queue
        if queued:
            self.walk_queue = None
            plan, floor, ceil = queued
            return self.run_walk(plan, floor, ceil)
        again = self.rearm_walk()
        if again is not None:
            return again
        wire = self.wire
        if self.orders_tried or self.needs.priority() != "cheapest" or wire.left() < Knobs.WIND_DOWN_S + 60 or wire.done:
            return False
        if self.unfound_product():
            return False
        opens = 1 if self.batch_tool() else 2
        spare = self.moves_left()
        due = self.order_due()
        roomy = due + 2 + Knobs.CAP_CALLS + opens < spare
        ceil = [k for k in self.cg_products() if self._k_ceil_due(k)]
        if ceil:
            if roomy:
                return self.run_walk([], due + 2, sorted(ceil, key=self._k_ceil_rank)[:1])
        elif roomy and self.ceil_seed():
            return True
        plan = self.walk_plan(Knobs.ID_WALK_ALL)
        if not plan:
            return False
        pages = sum(entry[2] for entry in plan)
        pages += sum(1 for entry in plan if self.walk_pick(entry[0]))
        if due + 2 + pages + opens < spare:
            return False
        plan = self.walk_plan(min(Knobs.ID_WALK_ALL, spare - due - 1 - opens))
        if plan:
            self.walk_early = True
            return self.run_walk(plan, due)
        return False

    def seed_args(self, label: str, bound: Any) -> dict | None:
        tool = self.role("search")
        schema = self.shapes.get(tool) or {}
        props = _k_dict_of(schema, "properties")
        found = self.page_arg()
        if not (props and found is not None and numeric(bound)):
            return None
        parts = re.split(r"[.;]", self._k_tool_text(tool).lower())
        shelf_key, shelf_value = self._k_shelf_arg(props, label, (self.token_key(), found[0]))
        bound_key = self._k_bound_key(props, found[0])
        if not (shelf_key and bound_key):
            return None
        whole = props[bound_key].get("type") == "integer" or float(bound).is_integer()
        args = {shelf_key: shelf_value, bound_key: int(bound) if whole else bound}
        if not self._k_add_modes(args, props, shelf_key, parts):
            return None
        if any(name not in args for name in _k_list_of(schema, "required")):
            return None
        fixed, faults = ArgFixer.fix(args, schema)
        return None if faults or not isinstance(fixed, dict) else fixed

    def _k_tool_text(self, tool) -> str:
        for entry in self.toolset:
            fn = entry.get("function") or {}
            if fn.get("name") == tool:
                return str(fn.get("description") or "")
        return ""

    @staticmethod
    def _k_shelf_arg(props, label, skip) -> tuple:
        for key, spec in props.items():
            if not isinstance(spec, dict) or key in skip or not K_SHELF_NAME.search(str(key).lower()):
                continue
            options = spec.get("enum")
            if isinstance(options, list):
                hit = next((v for v in options if isinstance(v, str) and v.strip().lower() == label.strip().lower()),
                           None)
                if hit is not None:
                    return key, hit
            elif spec.get("type") == "string":
                return key, label
        return None, None

    @staticmethod
    def _k_bound_key(props, paging) -> Any:
        for key, spec in props.items():
            if not isinstance(spec, dict) or key == paging or spec.get("enum"):
                continue
            low = str(key).lower()
            if spec.get("type") in ("integer", "number") and K_BOUND_NAME.search(low) and not K_BOUND_NOT.search(low):
                return key
        return None

    @staticmethod
    def _k_add_modes(args, props, shelf_key, parts) -> bool:
        # every short enum the call leaves open takes the value the tool's description pairs with the category
        shelf_word = r"\b%s\b" % re.escape(str(shelf_key).lower())

        def paired(v) -> bool:
            own = r"\b%s\b" % re.escape(str(v).lower())
            return any(re.search(own, part) and re.search(shelf_word, part) for part in parts)

        for key, spec in props.items():
            if key in args or not isinstance(spec, dict):
                continue
            options = spec.get("enum")
            if isinstance(options, list) and 2 <= len(options) <= 4:
                choice = next(filter(paired, options), None)
                if choice is None:
                    return False
                args[key] = choice
        return True

    def ceil_seed(self) -> bool:
        return self._k_try("ceil_seed_error", self._k_ceil_seed, False)

    def _k_ceil_seed(self) -> bool:
        if self.ceil_spec() is not None or self.ceil_mode is not None or self.needs.priority() != "cheapest":
            return False
        several = bool(self.needs.items())
        for k in self.cg_products():
            if k in self.ceil_seeded or (several and k is None):
                continue
            pick = self.walk_pick(k)
            label = self.raw_label(self.walk_home(k))
            if not (pick and label):
                continue
            self.ceil_seeded.add(k)
            args = self.seed_args(label, self.below(pick[1]))
            if args is None:
                return False
            if self._k_seed_known(args, label):
                continue
            self.ceil_seed_keys.add(_k_json(self._k_bare(args)))
            self.walk_call(args, "The shopper wants the cheapest, so I list '%s' priced below the pick (%s) to see "
                                 "whether the store's price filter holds." % (label, _k_money(pick[1])))
            learned = self.ceil_spec() is not None
            self.log.append(dict(kind="ceil_seed", product=k, args=args, learned=learned))
            if not learned:
                self._k_to_last_tool("", status=True)
            return True
        return False

    def _k_seed_known(self, args, label) -> bool:
        # the filter form was retired already, or this exact listing was seen
        found = self.page_arg()
        paging = found[0] if found else None
        mode = dict(self.search_keys(args)[1])
        kept = sorted((m, v) for m, v in mode.items() if not (isinstance(args.get(m), str) and args[m] == label))
        number = next((key for key, v in args.items() if numeric(v) and key != paging), "")
        if json.dumps([kept, number], default=str) in self.ceil_bad:
            return True
        return self._k_seen_total(args) is not None

    def unfound_product(self) -> bool:
        try:
            if not self.needs.items():
                return False
            read = {str(lab).strip().lower() for lab in self.archive.shelves_read()}
            for home in self.home_shelves():
                if not home or not ({str(lab).strip().lower() for lab in home} & read):
                    return True
            version = str(self.needs.version)
            return any(p.rsplit("|", 1)[-1] == version for p in self.passed) and bool(self.unverified())
        except Exception:  # noqa: BLE001
            return False

    def walk_hold(self, call_args: Any) -> str:
        return self._k_try("walk_hold_error", lambda: self._k_walk_hold(call_args), "")

    def _k_walk_hold(self, call_args) -> str:
        if self.needs.priority() != "cheapest" or self.walk_queue or self.wire.left() < Knobs.WIND_DOWN_S + 60:
            return ""
        if self.refusal_n and self.moves_left() <= 15:
            return ""
        if call_args is None:
            ids = list(self.trolley.units)
        else:
            ids = [ref for ref, n in self.trolley.lines(call_args) if n is None or 0 < n]
        only = {self.product_index(i) for i in ids if i}
        floor = 4 if self.orders_tried else Knobs.LATE_STEPS + 1
        half = (self.moves_left() - floor - 1) // 2
        room = half - (1 if self.batch_tool() else 2)
        ceil = []
        for k in sorted(only, key=lambda x: -1 if x is None else x):
            if room < 2 or not self.ceiling_ready(k):
                continue
            ceil.append(k)
            room -= Knobs.CAP_CALLS
            self._k_aim_lower(k, ids)
        plan = self.walk_plan(min(Knobs.ID_WALK_ALL, room), only)
        if not (plan or ceil):
            return ""
        self.walk_queue = (plan, max(floor, self.moves_left() - half - 1), ceil)
        return (K_NOT_YET + "the shopper wants the cheapest, and this store lists results in a fixed id order, not "
                "by price, so a cheaper fit may sit on pages not read yet. The code will first %s, then open the "
                "cheapest unread rows below the pick. After that, send this call again - or the same call with the "
                "cheaper fit." % " and ".join(self._k_walk_steps(plan, ceil)))

    def _k_aim_lower(self, k, ids) -> None:
        mine = [i for i in ids if i and self.product_index(i) == k]
        prices = [float(p) for p in map(self.archive.price, mine) if numeric(p)]
        if not prices:
            return
        old = self.ceil_aims.get(k)
        if numeric(old):
            prices.append(float(old))
        self.ceil_aims[k] = min(prices)

    def _k_walk_steps(self, plan, ceil) -> list:
        steps = []
        if plan:
            walks = [w for w in map(self.page_book.get, [entry[1] for entry in plan]) if w is not None]
            queries = ["'%s'" % " ".join(self.free_text(w.get("args") or {})) for w in walks]
            steps.append("read the remaining pages of %s" % ", ".join(queries or ["the planned searches"]))
        if ceil:
            steps.append("list the category's listings below the pick with the store's price filter")
        return steps

    # ====================================================================== CG4 / CG5 guards
    def value_guard(self, call_name: Any, call_args: dict) -> str:
        return self._k_try("value_guard_error", lambda: self._k_value_guard(call_name, call_args), "")

    def _k_guard_skips(self, ident) -> bool:
        if ident in self.archive.read or ident in self.trolley.units:
            return True
        return bool(ident in self.named_open() or self.refs or self.needs.reading.get("references"))

    def _k_value_guard(self, call_name, call_args) -> str:
        if call_name != self.role("details") or self.in_reserve():
            return ""
        if self.needs.priority() != "cheapest" or not self.all_words():
            return ""
        named = [v for v in _k_texts(call_args) if v in self.archive.ids]
        ident = named[0] if len(named) == 1 else None
        if ident is None or self._k_guard_skips(ident):
            return ""
        k = self.product_index(ident)
        mark = (k, self.needs.version)
        if mark in self.vguard_told or (k is None and self.needs.items()):
            return ""
        valued = self.valued_ids(k)
        if ident in valued or not valued:
            return ""
        pick = self.walk_pick(k)
        own = self.archive.price(ident)
        limits = [pick[1]] if pick else []
        if numeric(own):
            limits.append(own)
        rows = self.cheap_unread(valued, k, below=min(limits))[:6] if limits else []
        if not rows:
            return ""
        self.vguard_told.add(mark)
        if not pick:
            than = " than it"
        elif numeric(own):
            than = " than it and the pick"
        else:
            than = " than the pick"
        listed = ", ".join("%s (%s)" % (i, _k_money(p)) for p, i in rows)
        return (K_NOT_RUN + "the shopper wants the cheapest and this store's keyword search needs every word, so "
                "the rows that value-led searches showed carry those words. %s is not among them, while these rows "
                "cost less%s and are still unread: %s. Open them first, cheapest first; send this call again if "
                "it is still wanted." % (ident, than, listed))

    def title_line(self, k: Any, rec: dict, template: Any) -> dict | None:
        items = self.needs.items()
        avoid = set(FILLER)
        for lab in self.walk_home(k):
            avoid.update(re.findall(r"[a-z]+", str(lab).lower()))
        for req in self.needs.props_left():
            avoid.update(re.findall(r"[a-z0-9]+", str(req).lower()))
        names = [it.get("product") for it in items] if items else [self.needs.reading.get("product")]
        for name in names:
            avoid.update(re.findall(r"[a-z]+", str(name or "").lower()))
        stems = {self.stem(w) for w in avoid}
        toks = re.findall(r"[a-z0-9]+", str(self.title_of(rec)).lower())
        for i, tok in enumerate(toks):
            if not re.fullmatch(r"[a-z]{3,}", tok) or tok in avoid or self.stem(tok) in stems:
                continue
            words = [tok]
            if i + 1 < len(toks) and toks[i + 1] not in avoid and self.stem(toks[i + 1]) not in stems:
                words.append(toks[i + 1])
            text = " ".join(words)
            if self.ran_query(text):
                return None
            other, text_key = template
            return dict(other, **{text_key: text})
        return None

    # ====================================================================== notes on observations
    @staticmethod
    def _k_row_list(piece) -> Any:
        # The first non-empty list of records in the piece, else its first (empty) one, else None.
        lists = []
        for v in piece.values():
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                if v:
                    return v
                lists.append(v)
        return lists[0] if lists else None

    def _k_result_page(self, obs) -> tuple | None:
        rows = self._k_row_list(obs)
        if rows is not None:
            return obs, rows
        for v in obs.values():
            if isinstance(v, dict) and self._k_row_list(v) is not None:
                return v, self._k_row_list(v) or []
        return None

    def search_note(self, call_args: dict, obs: Any) -> str:
        page = self._k_result_page(obs) if isinstance(obs, dict) else None
        if page is None:
            return ""
        obs, rows = page
        totals = [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool) and len(rows) < v]
        total = max(totals) if totals else len(rows)
        self.book_page(call_args, rows, total, obs)
        self._k_try("ceil_learn_error", lambda: self.ceil_learn(call_args, rows, total, obs))
        if rows:
            score = self.value_score(call_args)
            if 0 < score:
                for mark in map(self.row_mark, rows):
                    _k_top(self.row_score, mark, score)
        kind = self.word_rule(call_args, obs, rows)
        lead = self.lead_rule(call_args, obs, rows)
        texts, mode, first, paging_keys = self.search_keys(call_args)
        mode = tuple(map(tuple, mode))
        notes: list = [self._k_lead_note(lead, texts, mode)] if lead > 0 else []
        mode_keys = {p[0] for p in mode}
        raw = [v for key, v in call_args.items() if isinstance(v, str) and key not in mode_keys and key not in paging_keys]
        if first and raw:
            notes += self._k_name_notes(raw, rows)
            notes += self._k_repeat_note(call_args, rows, raw, mode, (mode_keys, paging_keys), total)
            found, hits = self._k_hit_notes(call_args, rows, raw, kind, texts, totals)
            notes += found
            notes += self._k_back_note(call_args, kind, paging_keys, hits)
        if rows:
            if totals:
                notes.append(self._k_total_note(kind, max(totals), len(rows)))
            notes += self._k_limit_note(call_args, rows, paging_keys)
        return " ".join(notes)

    def _k_lead_note(self, lead, texts, mode) -> str:
        context = self.lead_ctx
        where = " (%s)" % context[0] if context else ""
        clause = ""
        if context and tuple(map(tuple, context[1])) == mode:
            words = next((list(w) for key, w in texts if key == context[0]), [])
            if lead < len(words):
                clause = " (here only '%s' counted)" % " ".join(words[:lead])
        return ("The store's keyword search%s reads only the first %d words%s: put the %d most distinctive "
                "words first - a required value and the item kind, value first - and search other values "
                "separately." % (where, lead, clause, lead))

    def _k_name_notes(self, raw, rows) -> list:
        out = []
        pieces = re.findall(r"[a-z0-9]+", " ".join(raw).lower())
        if pieces and max(map(len, pieces)) == 1 and not (rows or self.short_told):
            self.short_told = True
            out.append("This store's search skips one-character words, so figures such as '0.4' or '5 x 7' "
                       "search nothing: search a word value, or the unit word with the item kind, and read every "
                       "page.")
        prop, probe = self.value_probe(raw)
        if probe:
            if self.name_told < 2 and self.order_due() + 3 < self.moves_left():
                self.name_told += 1
                if self.name_told == 1:
                    out.append("This query leads with words of a property's NAME (%s) and none of a required value; "
                               "keyword search matches values, titles and categories, so it misses listings whose "
                               "field holds just the value. Search value first, then the kind - '%s' (a figure the "
                               "store cannot read drops out, leaving its unit word) - and read every page."
                               % (prop, probe))
                else:
                    out.append("Again led by a property's name: search '%s' instead (value first, then the kind) and "
                               "read every page." % probe)
        elif self.trap_told < 2 and self.order_due() + 3 < self.moves_left():
            trap = self.word_trap(raw)[0]
            if trap:
                self.trap_told += 1
                out.append(trap)
        return out

    def _k_repeat_note(self, call_args, rows, raw, mode, skip, total) -> list:
        mode_keys, paging_keys = skip
        marks = tuple(self.row_mark(r) for r in rows if isinstance(r, dict))
        others = tuple(sorted((key, str(v)) for key, v in call_args.items()
                              if key not in paging_keys and not (isinstance(v, str) and key not in mode_keys)))
        query = " ".join(raw)
        varied = len(set(marks)) >= 3
        before = self.page_sigs.setdefault((mode, others, total, marks), query)
        if varied and before != query:
            return ["This matched the same total and the same rows as the earlier search '%s', so re-spelling "
                    "it shows nothing new: read the next page or change the leading words." % before]
        return []

    def _k_hit_notes(self, call_args, rows, raw, kind, texts, totals) -> tuple:
        told = (kind == "all" or bool(self.lead_words)) and not totals and bool(rows)
        if not (told or (kind != "any" and not totals and rows)):
            return [], []
        held = None if told else list(self.ref_hits)
        self.ref_hits = []
        hits = self.row_hits(rows, raw)
        score = self.value_score(call_args)
        for h in hits:
            _k_top(self.hit_score, h, score)
            if h not in self.all_hits:
                self.all_hits.append(h)
        if held is not None:
            self.ref_hits = held
        if not told:
            return [], hits
        out = []
        if hits and texts and len(rows) <= 12:
            out.append("Every word is required here and every match is shown, so each row carries these "
                       "words somewhere in its record - often in fields its title hides. Open the in-budget "
                       "ones before searching on: %s." % ", ".join(hits[:6]))
        pointers = [i for i in self.ref_hits if i not in hits][:6]
        self.ref_hits = []
        if pointers:
            out.append("The request gives a value by pointing at another listing, and these rows are sold "
                       "by the shop or brand this search names: open them whatever their price, stock or "
                       "category (such a listing is read, not bought) and follow its notes: %s."
                       % ", ".join(pointers))
        return out, hits

    def _k_back_note(self, call_args, kind, paging_keys, hits) -> list:
        if self.hits_back or not (kind == "all" or self.lead_words):
            return []
        self.hits_back = True
        here = _k_json({key: v for key, v in call_args.items() if key not in paging_keys})
        back: list = []
        for walk_key, walk in self.page_book.items():
            if walk_key == here or not walk.get("every"):
                continue
            score = self.value_score(walk.get("args") or {})
            if score <= 0:
                continue
            fresh = [i for i in self.row_hits(walk["every"]) if i not in back and i not in hits]
            for i in fresh:
                _k_top(self.hit_score, i, score)
            back += fresh
        self.all_hits.extend(i for i in dict.fromkeys(back) if i not in self.all_hits)
        if not back:
            return []
        return ["The store needs every word, so earlier searches that showed every match listed only "
                "rows carrying all their words. Open the in-budget unread ones before searching on: %s."
                % ", ".join(back[:6])]

    @staticmethod
    def _k_total_note(kind, most, shown) -> str:
        text = ("%d listings matched but this page shows %d. Only a few pages can be read and the listings that "
                "matter may be unseen, so read further pages of this same search first (as many as can be read). "
                "If the total is more than those pages show: " % (most, shown))
        if kind == "any":
            return text + ("here any word matches, so adding words widens the search - a required value's distinctive "
                           "word or figure alone lists only the listings carrying it (often few enough to read every page "
                           "and compare prices), and the item kind with values ranks the closest first; use the category "
                           "mode (or a price filter if one holds) for the rest.")
        if kind == "all":
            return text + ("here every word is required, so more exact words narrow the search, as do the category and "
                           "a price filter with a lower limit.")
        return text + ("narrow it for the rest - with exact words if every word is required; if any word matches (a "
                       "longer query matching more shows that), put required values in the query so fits rank first. "
                       "A price filter helps when one holds.")

    @staticmethod
    def _k_limit_note(call_args, rows, paging_keys) -> list:
        limits = [v for key, v in call_args.items() if numeric(v) and 1 < v and key not in paging_keys
                  and not K_NOT_LIMIT.search(str(key).lower())]
        if not limits:
            return []
        cap = max(limits)
        over = sum(1 for r in rows if isinstance(r, dict)
                   and any("price" in str(kk).lower() and numeric(vv) and cap < vv for kk, vv in r.items()))
        if not over:
            return []
        return ["%d rows here are priced above the limit %s: this mode ignored the filter (and may ignore "
                "others, such as the category) - use the mode whose filters hold, or check every row."
                % (over, _k_money(cap))]

    def shopper_lines(self, call_name: Any, obs: Any, refused: bool) -> list:
        message = self.role("message")
        if message and call_name == message:
            return self._k_reply_lines(obs)
        lines: list = []
        for v in (obs.values() if isinstance(obs, dict) else ()):
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                sentences_in(v, lines)
            elif refused and isinstance(v, str) and len(v.split()) >= 4:
                lines.append(v)
        return lines

    @staticmethod
    def _k_reply_lines(obs) -> list:
        if isinstance(obs, str):
            return [obs]
        lines: list = []
        if isinstance(obs, dict) and "error" not in obs:
            sentences_in([v for v in obs.values() if isinstance(v, (str, list))], lines)
        return lines

    # ====================================================================== screening one batch of proposals
    def screen(self, proposal: dict, permitted: Any, scripted: bool, batch: int, queued_messages: int) -> tuple:
        fn = proposal.get("function") if isinstance(proposal.get("function"), dict) else {}
        name = fn.get("name")
        given = fn.get("arguments")
        if name not in self.shapes:
            return (K_NOT_RUN + "there is no tool named '%s'. Tools you can use: %s."
                    % (name, ", ".join(sorted(str(t) for t in permitted))), name, {})
        message = self.role("message")
        if not scripted and name not in permitted:
            if message and name == message and self.asking:
                self.want_ask(given)
                return (K_NOT_RUN + "questions to the shopper are planned and sent by the code, once a few listings' "
                                    "details have been opened.", name, {})
            return (K_NOT_RUN + "too few steps are left for this call. Tools usable now: %s."
                    % ", ".join(sorted(str(t) for t in permitted)), name, {})
        if isinstance(given, dict):
            args = given
        elif isinstance(given, str):
            args = first_json(given) if given.strip() else {}
        elif given is None:
            args = {}
        else:
            args = None
        if args is None:
            return K_NOT_RUN + "the arguments were not a valid JSON object.", name, {}
        args, faults = ArgFixer.fix(args, self.shapes.get(name) or {})
        if faults:
            return K_NOT_RUN + "the arguments do not fit the tool: " + "; ".join(faults[:6]) + ".", name, args
        closing = self.finishers()
        if name in closing and batch > 1:
            return (K_NOT_RUN + "an order or a decline must be sent alone, after the other results are read - send it "
                                "again by itself.", name, args)
        if message and name == message and self.msg_cap is not None and self.msgs_sent + queued_messages >= self.msg_cap:
            return K_NOT_RUN + "the message budget is used up.", name, args
        view, cart = self.role("cart_view"), self.role("cart")
        since = 1 if self.since_cart is None else self.since_cart
        if (not scripted and view and name == view and name != cart and not self.batch_cart and self.trolley.units
                and self.cart_shown == self.cart_ver and since == 0 and self.cart_shown not in self.view_told):
            self.view_told.add(self.cart_shown)
            reader = self.stock_tool()
            return (K_NOT_RUN + "the last cart result already shows the basket (%s), so a cart view shows nothing new. "
                    "To check stock again, read the listing's record with %s." % (
                        self.trolley.summary(), reader or "the tool that shows a listing's stock"), name, args)
        sig = _k_sig(name, args)
        if name not in closing and name != message and name != cart and sig in self.last_calls:
            return (K_NOT_RUN + "this exact call was just made and nothing has changed since - change the arguments.",
                    name, args)
        if not scripted:
            held = self.late_hold(name, args) or self.value_guard(name, args)
            if held:
                return held, name, args
        if cart and name == cart:
            held = self.screen_cart(args, sig)
            if held:
                return held, name, args
        return None, name, args

    def _k_cart_check(self, line) -> str:
        key = "%s|%s" % (line, self.needs.version)
        if self.moves_left() > 3 and key not in self.checked_adds:
            self.checked_adds.add(key)
            ruling = self.check_item(line, "ADD TO THE CART")
            if ruling:
                return K_NOT_RUN + ruling + " If you are certain it fits, send the call again."
        return ""

    def screen_cart(self, call_args: dict, sig: str) -> str:
        thing = self.trolley.aim(call_args)
        putting = self.trolley.is_put(call_args)
        point = self.ref_pointer_note(thing) if (thing and putting) else ""
        if point and self.pointer_told.get(thing, 0) < 2:
            _k_bump(self.pointer_told, thing)
            return K_NOT_RUN + point + " Choose another listing."
        cheapest = self.needs.priority() == "cheapest"
        kept = self._k_kept_line(thing, putting, sig, cheapest)
        if kept:
            return kept
        if putting and thing:
            return self._k_put_screen(call_args, sig, thing, point, cheapest)
        return ""

    def _k_kept_line(self, thing, putting, sig, cheapest) -> str:
        # a removal of a line the independent check passed, with nothing changed since
        mark = "%s#%s" % (sig, self.puts.get(thing, 0))
        mine = self.archive.price(thing) if thing else None
        price = self.archive.price
        cheaper_put = numeric(mine) and any(
            other != thing and numeric(price(other)) and price(other) < mine for other in self.batch_puts)
        swap = cheapest and (cheaper_put or bool(self.cheaper_fit(thing)))
        moved = thing in self.pass_price and self.pass_price[thing] != mine
        if not thing or putting or swap or moved or not self._k_passed(thing):
            return ""
        if self.refusal_n != self.refusals_seen.get(thing, self.refusal_n) or mark in self.warned:
            return ""
        self.warned.add(mark)
        text = (K_NOT_RUN + "an independent check found %s meets every requirement in force under the value rules "
                "(only the category label decides the item kind)." % thing)
        if cheapest:
            text += " It is also the cheapest verified fit."
        return text + (" Nothing has changed since. If a change or a refusal rules it out, a field clearly fails, or "
                       "you are replacing it with a cheaper listing, send the same call again.")

    def _k_put_screen(self, call_args, sig, thing, point, cheapest) -> str:
        held = self.trolley.held(thing)
        if 0 < held and self.trolley.adds is not False and sig not in self.warned:
            self.warned.add(sig)
            return (K_NOT_RUN + "the cart already holds %s units of %s and this tool adds units, so the call would raise "
                    "them. To order, leave the cart as it is; to change the units, take the difference out or put it "
                    "in. Send the call again if adding is meant." % (_k_money(held), thing))
        hold = "" if point else self.walk_hold(call_args)
        if hold:
            return hold
        lines = self.trolley.lines(call_args)
        later = [line for line, n in lines[1:] if n is None or 0 < n]
        ruling = next(filter(None, map(self._k_cart_check, later + [thing])), "")
        if ruling:
            return ruling
        live = [line for line, n in lines if n is None or 0 < n]
        better = self._k_rank_hold(live) if 3 < self.moves_left() else ""
        if better:
            return better
        cheaper = self._k_cart_cheaper(live) if cheapest else ""
        if cheaper:
            return cheaper
        need_n = self.needs.quantity()
        count = self.trolley.count_arg(call_args)
        aim = need_n if self.trolley.adds is False else need_n - held
        items = self.needs.items()
        if items:
            return self._k_line_counts(items, lines, sig)
        if count is None or sig in self.warned or not 1e-9 < abs(count - aim):
            return ""
        self.warned.add(sig)
        return (K_NOT_RUN + "the request needs %d of one listing, but this call asks for %s. Send it again if you are "
                "sure." % (need_n, _k_money(count)))

    def _k_rank_hold(self, live) -> str:
        for line in live:
            better = self.rank_check(line)
            if not better:
                continue
            cheaper = self.rank_verify(line)
            text = better.replace("before ordering", "before putting it in the cart").replace(
                "order the current pick", "send the same call again")
            if cheaper:
                text += (" The item check found %s meets every requirement and costs less, so put it in "
                         "instead." % cheaper)
            return K_NOT_RUN + text
        return ""

    def _k_cart_cheaper(self, live) -> str:
        for line in live:
            cheaper = self.cheaper_fit(line)
            told = ("cart", line, cheaper)
            if not cheaper or cheaper in self.trolley.units or told in self.cheaper_told:
                continue
            self.cheaper_told[told] = 1
            return (K_NOT_RUN + "the check found %s meets every requirement and costs less than %s; its reading "
                    "stands, so put %s in instead - unless a record shows it failing a requirement, then send "
                    "the same call again." % (cheaper, line, cheaper))
        return ""

    def _k_line_counts(self, items, lines, sig) -> str:
        wanted = self.needs.counts()
        odd = any(numeric(n) and 0 < n and int(n) not in wanted for _, n in lines)
        if not odd or sig in self.warned:
            return ""
        self.warned.add(sig)
        return (K_NOT_RUN + "the products need %s (one line each) but this call asks for %s. Send it again if "
                "you are sure." % ("; ".join("%s x%d" % (it["product"], it["quantity"]) for it in items),
                                   ", ".join("%s x%s" % (line, _k_money(n)) for line, n in lines)))

    def screen_finish(self, cid: Any, call_name: Any, call_args: dict) -> tuple:
        kind = self.call_kind(call_name, call_args)
        if kind == "order":
            return self._k_order_hold(call_args)
        if kind == "decline":
            return self._k_decline_hold(call_name, call_args)
        return None, call_args

    def _k_order_hold(self, call_args: dict) -> tuple:
        for step in (self._k_dead_hold, self._k_cheaper_hold, self._k_walk_back, self._k_stale_hold, self._k_vet_order):
            text = step()
            if text:
                return text, call_args
        return None, call_args

    def _k_dead_hold(self) -> str:
        if self.moves_left() <= 2:
            return ""
        gone = self.dead_lines()
        dead = sorted(line for line in self.trolley.units if line in gone)
        if not dead:
            return ""
        mark = (tuple(dead), self.cart_seq)
        told = self.dead_told.get(mark, 0)
        if told < 2:
            self.dead_told[mark] = told + 1
            return K_NOT_YET + self.dead_note(dead)
        if told == 2 and self.role("cart") and 6 <= self.moves_left() and Knobs.WIND_DOWN_S < self.wire.left():
            return self._k_dead_swap(dead, gone, mark)
        return ""

    def _k_dead_swap(self, dead, gone, mark) -> str:
        self.dead_told[mark] = 3
        muted = self.wire.mute
        swapped = self._k_try("dead_swap_error", lambda: self.code_cart(
            avoid=set(dead) | set(gone), keep={line for line in self.trolley.units if line not in dead}), False)
        done = bool(swapped)
        self.wire.mute = muted
        self.log.append(dict(kind="dead_swap", lines=dead, done=done, cart=dict(self.trolley.units)))
        now_dead = self.dead_lines()
        if not (done and self.trolley.units) or any(line in now_dead for line in self.trolley.units):
            return ""
        return K_NOT_YET + ("the cart still held %s, which is sold out, so the code took it out and put a "
                            "candidate in, keeping the other lines, and read the lines' stock: %s Check the cart against "
                            "the request, then order." % (", ".join(dead), self.trolley.summary()))

    def _k_cheaper_hold(self) -> str:
        if self.moves_left() <= 3:
            return ""
        thing = self.chosen()
        cheaper = self.cheaper_fit(thing) if thing else ""
        if not cheaper:
            for line in self.trolley.units:
                found = self.cheaper_fit(line) if line != thing else ""
                if found:
                    thing, cheaper = line, found
                    break
        pair = (thing, cheaper)
        if not cheaper or self.cheaper_told.get(pair, 0) >= 2:
            return ""
        _k_bump(self.cheaper_told, pair)
        return K_NOT_YET + ("the check found %s meets every requirement and costs less than %s; its reading "
                            "stands, so take %s out, put %s in, read its stock and price again, and order."
                            % (cheaper, thing, thing, cheaper))

    def _k_walk_back(self) -> str:
        return self.walk_hold(None) if self.trolley.units else ""

    def _k_stale_hold(self) -> str:
        reader = self.stock_tool()
        if reader and len(self.trolley.units) + 2 < self.moves_left() and Knobs.WIND_DOWN_S < self.wire.left():
            stale = list(self.stale_lines())
            mark = (tuple(sorted(self.trolley.units)), self.cart_seq)
            if stale and self.recheck_mark != mark:
                self.recheck_mark = mark
                self.open_queue = stale[:4]
                self.open_recheck = True
                self.open_purpose = "cart"
                self.open_tool = reader
                one = len(stale) == 1
                return K_NOT_YET + ("stock and prices can change without notice, and %s %s not read since the last "
                                    "cart change. The code reads %s now with %s: order if each line is in stock and the "
                                    "total is within the budget, else replace the line." % (
                                        ", ".join(stale[:4]), "was" if one else "were",
                                        "its stock" if one else "their stock", reader))
        return ""

    def _k_vet_order(self) -> str:
        if self.order_vetted or self.moves_left() <= 2 or not Knobs.WIND_DOWN_S < self.wire.left():
            return ""
        self.order_vetted = True
        faults = self.trolley.faults(self.needs.quantity(), self.needs.counts())
        if faults and 3 < self.moves_left():
            return K_NOT_YET + "the cart as counted is not ready: " + "; ".join(faults) + ". Fix it, then order."
        chosen = self.chosen()
        if self.needs.items():
            lines = list(self.trolley.units)
        else:
            lines = [chosen] if chosen else []
        ruling = next(filter(None, (self.check_item(line, "ORDER") for line in lines)), "")
        if ruling:
            return K_NOT_YET + ruling
        for line in lines:
            better = self.rank_check(line)
            if not better:
                continue
            cheaper = self.rank_verify(line)
            if cheaper:
                better += (" The check found %s meets every requirement and costs less: take %s out, put %s "
                           "in, read its stock and price again, and order." % (cheaper, line, cheaper))
            return K_NOT_YET + better
        return K_NOT_YET + K_BEFORE_ORDER

    def _k_decline_hold(self, call_name: Any, call_args: dict) -> tuple:
        steps = (self.near_hold, self._k_keep_passed, self._k_decline_opens, self._k_decline_review,
                 self._k_decline_best, self._k_decline_refs, self.decline_repair, self.near_check)
        for step in steps:
            note = step()
            if note:
                return note, call_args
        return None, self.scrub_decline(call_args, call_name)

    def _k_keep_passed(self) -> str:
        gone = self.dead_lines()
        kept = [line for line in self.trolley.units
                if self._k_passed(line) and line not in self.refused_lines and line not in gone]
        if not kept or self.moves_left() <= 1 or self.decline_kept:
            return ""
        self.decline_kept = True
        return K_NOT_YET + ("the independent check passed %s: it meets every requirement in force. A decline would "
                            "throw that away - read its stock and price again and order%s. Decline only if a change "
                            "clearly rules it out." % (", ".join(kept),
                                                       " (with every product's line)" if self.needs.items() else ""))

    def _k_decline_opens(self) -> str:
        batch = self.batch_tool()
        reader_ok = bool(self.role("details") or batch)

        def reads(ids) -> int:
            return _k_ceil(len(ids), batch[2]) if (batch and len(ids) >= 2) else len(ids)

        slack = self.slack_rows() if (reader_ok and not self.slack_done) else []
        if slack and reads(slack) + 4 < self.moves_left() and Knobs.WIND_DOWN_S + 40 < self.wire.left():
            self.slack_done = True
            self.open_queue = list(slack)
            self.open_say = ("Every product has a fit, but together they cost more than the budget, so I open cheaper "
                             "rows that may let every product fit it.")
            cost = self.slack_total
            return K_NOT_YET + (
                "every product has a fitting listing, but the cheapest fits together cost %s - over the budget of "
                "%s. Rows within what the budget leaves each product were never opened, so they are read now: %s. "
                "Judge each against its own product's requirements under the value rules; when fits within the "
                "budget cover every product, put every line in and order. Decline only if no such set exists."
                % (_k_money(cost) if numeric(cost) else "more than the budget", _k_money(self.needs.spend_cap),
                   ", ".join(slack)))
        fill = self.lacking_rows() if (reader_ok and not self.decline_fill) else []
        if fill and reads(fill) + 3 < self.moves_left() and Knobs.WIND_DOWN_S + 40 < self.wire.left():
            self.decline_fill = True
            self.open_queue = list(fill)
            self.open_say = "Some products have no verified fit yet, so I open unread listings of their categories."
            return K_NOT_YET + ("some products have no verified fit, and in-budget listings of their categories "
                                "were never opened, so they are read now: %s. Judge each under the value rules; when "
                                "every product has a fit, put every line in and order. Decline only if a product truly "
                                "has none." % ", ".join(fill))
        queue = self.named_unread()[:batch[2] if batch else 3]
        if not queue or self.decline_opens or not reader_ok:
            return ""
        if reads(queue) + 3 < self.moves_left() and Knobs.WIND_DOWN_S + 40 < self.wire.left():
            self.decline_opens = True
            self.open_queue = list(queue)
            return K_NOT_YET + ("the listings the checks named were never opened, so their records are read now: %s. "
                                "Judge each under the value rules: a field with the property's own name decides, and "
                                "where the store writes the value there only those words meet it; without such a "
                                "field, another field, the options list or the title may show it. Buy one that fits, "
                                "and decline only if each one fails." % ", ".join(queue))
        return ""

    def _k_decline_review(self) -> str:
        note = self.decline_reads()
        if note:
            return K_NOT_YET + note
        last_reads = self.audit_reads if numeric(self.audit_reads) else 10 ** 9
        reread = len(self.archive.read) > last_reads and (self.audits or 0) < 2
        if self.decline_vetted:
            if reread and 3 < self.moves_left() and Knobs.WIND_DOWN_S + 30 < self.wire.left():
                note = self.review_decline()
                if note:
                    return K_NOT_YET + note
            return ""
        if 3 < self.moves_left() and Knobs.WIND_DOWN_S + 30 < self.wire.left():
            self.decline_vetted = True
            return K_NOT_YET + (self.review_decline() or K_BEFORE_DECLINE)
        return ""

    def _k_decline_best(self) -> str:
        note = self.fit_hold()
        if note:
            return K_NOT_YET + note
        best = self.unrefuted() if (not self.best_held and 4 < self.moves_left()) else []
        for _ in range(3):
            unchecked = [b for b in best if not self._k_passed(b)]
            if not unchecked or self.wire.left() < Knobs.WIND_DOWN_S + 30:
                break
            self.check_item(unchecked[0], "ORDER")
            best = self.unrefuted()
        if not (best and len(best) + 3 < self.moves_left() and Knobs.WIND_DOWN_S + 20 < self.wire.left()):
            return ""
        self.best_held = True
        several = bool(self.needs.items())
        return K_NOT_YET + ("a decline is right only when no listing can meet the requirements, and the records "
                            "hold %s that is in stock, within the budget and not ruled out: %s. Unless a record shows "
                            "one failing a requirement in force, put %s in, read its stock and price again, and order. "
                            "Decline only if each one clearly fails." % (
                                "a candidate for every product" if several else "a candidate", ", ".join(best),
                                "one line of each product, at its quantity," if several else "it"))

    def _k_decline_refs(self) -> str:
        soft = [r for r in self.refs if r.get("relation") == "differ" and r.get("status") in ("open", "stuck")
                and self.ref_live(r)]
        if not soft or self.moves_left() <= 3 or self.ref_decline_held:
            return ""
        self.ref_decline_held = True
        heads = ", ".join(dict.fromkeys(str(r["head"]) for r in soft))
        return K_NOT_YET + ("%s is given by another listing the code could not resolve, so its value is unknown: it "
                            "never rules a listing out and never justifies a decline. Put the listing that meets every "
                            "other requirement (in stock, within the budget) in%s, read its stock and price again, and "
                            "order. Decline only if none meets the others."
                            % (heads, " (one line per product)" if self.needs.items() else ""))

    # ====================================================================== dispatch and absorption
    def dispatch(self, proposals: list, scripted: bool = False) -> list:
        self.begin_batch()
        offered = [t for t in self.offered() if isinstance(t, dict)]
        permitted = {(t.get("function") or {}).get("name") for t in offered}
        cart, message = self.role("cart"), self.role("message")
        self._k_batch_puts(proposals, cart)
        replies, to_send = self._k_screen_all(proposals, permitted, scripted, cart, message)
        closing = self.finishers()
        head = to_send[0] if to_send else None
        if head is not None and head[1] in closing:
            held = self._k_hold_first(to_send)
            if held is not None:
                return replies + [held]
        elif head is not None and closing:
            keep = max(0, self.moves_left() - 1)
            replies.extend((pid, K_NOT_RUN + "only the last step is left, and it must be the order or the decline.")
                           for pid, _, _ in to_send[keep:])
            del to_send[keep:]
        for start in range(0, len(to_send), K_SLAB):
            replies.extend(self._k_slab(to_send[start:start + K_SLAB]))
        rank = {p.get("id"): i for i, p in enumerate(proposals)}
        replies.sort(key=lambda r: rank[r[0]] if r[0] in rank else 0)
        return replies

    def _k_hold_first(self, to_send) -> tuple | None:
        # Screens the order or decline that leads the batch: (id, hold) when it is held.
        pid, name, args = to_send[0]
        hold, fixed = self.screen_finish(pid, name, args)
        to_send[0] = (pid, name, fixed)
        return (pid, hold) if hold else None

    def _k_batch_puts(self, proposals, cart) -> None:
        # The listings this reply's cart calls put in, before any of them is screened.
        self.batch_puts = set()
        self.batch_cart = False
        for fn in map(_k_fn, proposals):
            if fn.get("name") == cart:
                self.batch_cart = True
                target = self._k_put_target(fn.get("arguments"))
                if target:
                    self.batch_puts.add(target)

    def _k_put_target(self, given) -> Any:
        if isinstance(given, dict):
            args = given
        elif isinstance(given, str) and given.strip():
            args = first_json(given)
        else:
            return None
        if isinstance(args, dict) and self.trolley.is_put(args):
            return self.trolley.aim(args)
        return None

    def _k_screen_all(self, proposals, permitted, scripted, cart, message) -> tuple:
        replies: list = []
        to_send: list = []
        queued = 0
        cart_held = False
        batch = len(proposals)
        for p in proposals:
            pid = p.get("id")
            if cart_held and _k_fn(p).get("name") == cart:
                replies.append((pid, K_CART_STALE))
                continue
            refusal, name, args = self.screen(p, permitted, scripted, batch, queued)
            if refusal is not None:
                replies.append((pid, refusal))
                if name == cart and isinstance(args, dict) and args and not self.trolley.is_put(args):
                    cart_held = True
                continue
            to_send.append((pid, name, args))
            if message and name == message:
                queued += 1
        return replies, to_send

    def _k_slab(self, slab) -> list:
        if self._k_spent():
            return [(pid, "Not run: the episode is over or no steps are left.") for pid, _, _ in slab]
        room = self.moves_left()
        late = [(pid, "Not run: no step was left for this call.") for pid, _, _ in slab[room:]]
        return late + self.send_chunk(slab[:room])

    def send_chunk(self, slab: list) -> list:
        wire = self.wire
        pids = [entry[0] for entry in slab]
        actions = [dict(name=name, args=args) for _, name, args in slab]
        step = wire.tick + 1
        resp = wire.act(actions)
        self.log.append(dict(kind="env", step=step, actions=actions, reply=resp))
        if resp is None:
            return [(pid, "The call failed (network or server); try again.") for pid in pids]
        many = resp.get("observations")
        seen = many if isinstance(many, list) else [resp.get("observation")]
        last = len(seen) - 1
        out = [(pid, self.absorb(name, args, obs, resp, i == last))
               for i, ((pid, name, args), obs) in enumerate(zip(slab, seen))]
        return out + [(pid, "The episode finished before this call ran.") for pid in pids[len(seen):]]

    def absorb(self, call_name: Any, call_args: Any, obs: Any, resp: Any, last: bool) -> str:
        shown = obs
        obs = with_plain_fields(obs)
        name = call_name
        args = {} if not isinstance(call_args, dict) else call_args
        cart, view, message, details, search = map(self.role, K_ROLES)
        closing = self.finishers()
        is_cart = bool(cart) and name == cart
        talk = bool(message) and name == message
        self._k_count_call(name, args, talk, is_cart or talk or name in closing)
        self.archive.add(obs)
        self.absorb_n += 1
        if not (is_cart or self.covering):
            self.since_cart = (self.since_cart or 0) + 1
        gone_now = self._k_try("caps_error", lambda: self.learn_caps(name, args, obs), [])
        if self.trolley.units:
            self._k_lines_seen(name, args, obs, (cart, view))
        failed = isinstance(obs, dict) and obs.get("error")
        if details and name == details and not failed:
            self.archive.read.update(v for v in _k_texts(args) if v in self.archive.ids)
        if self._k_late_read(name, args, details, search):
            self.late_n += 1
        notes = ["[stock] " + self.dead_note(gone_now)] if gone_now else []
        extra = self._k_try("notes_remark_error", lambda: self.notes_remark(name, args))
        if extra:
            notes += ["[notes] " + extra]
        if is_cart:
            self._k_cart_booked(args, obs, notes)
        if view and name == view:
            self.trolley.viewed(obs, self.wire.tick)
            notes += [self.trolley.summary()]
        refused = self.refused(name, obs, resp if last else dict(finished=False))
        if self.last_kind == "order" and refused:
            self.cart_seq = self.absorb_n
        notes += [self.notes_for(name, obs, refused)]
        searched = bool(search) and name == search
        if searched:
            notes += [self.search_note(args, obs)]
            self.named_pages.discard(_k_json(args))
            notes += [self._k_try("merge_joined_error", lambda: self.merge_joined(obs))]
        if last and not (self.early_audit or is_cart or talk) and name not in closing and self._k_audit_due():
            notes += self._k_early_text()
        caught = self.shopper_lines(name, obs, refused)
        if caught:
            question = " ".join(_k_texts(args)) if talk else None
            notes += [self.needs.hear(caught, question)]
            self.ref_take()
        body = as_text(clip(shown, Knobs.ROW_FIELD_MAX if searched else Knobs.RECORD_FIELD_MAX))
        if last and isinstance(resp, dict) and resp.get("finished"):
            body += "\n[episode finished]"
        said = " ".join(filter(None, notes))
        return body + "\n[note] " + said if said else body

    def _k_early_text(self) -> list:
        self.early_audit = True
        text = self.review_decline(early=True)
        return ["Nothing verified fits yet. " + text] if text else []

    def _k_count_call(self, name, args, talk, reset) -> None:
        kind = self.call_kind(name, args)
        if talk:
            self.msgs_sent += 1
        self.last_kind = kind
        if kind == "order":
            self.orders_tried += 1
        self.last_calls = [] if reset else [*self.last_calls, _k_sig(name, args)][-2:]

    def _k_lines_seen(self, name, args, obs, quiet) -> None:
        # Marks when each cart line's own record was last seen in this answer.
        held = self.trolley.units
        lookups_only = not (bool(self.rows_live) and not self.rows_lag) and name not in quiet
        rule = bool(self.stock_seen)
        asked = self._k_asked(args)
        fresh: set = set()
        for d in _k_dicts(obs):
            hits = {v for k, v in d.items() if "id" in str(k).lower() and isinstance(v, str) and v in held}
            if hits and self._k_line_row(d, hits, rule, lookups_only, asked):
                fresh.update(v for v in _k_texts(d) if v in held)
        tick, seq = self.wire.tick, self.absorb_n
        for line in fresh:
            self.line_read[line] = tick
            self.line_seq[line] = seq

    @staticmethod
    def _k_asked(args) -> set:
        found = set()
        for v in args.values():
            found.update(item for item in (v if isinstance(v, list) else [v])
                         if isinstance(item, str) and ID_SHAPE.fullmatch(item))
        return found

    def _k_line_row(self, d, hits, stock_rule, lookups_only, asked) -> bool:
        if not stock_rule:
            return any(isinstance(v, dict) for v in d.values())
        return self.stock_state(d) is not None and (not lookups_only or bool(hits & asked))

    def _k_late_read(self, name, args, details, search) -> bool:
        pair = (details, search)
        looked = bool(details) and name not in pair and name in self.lookup_tools()
        if not (name in pair or looked) or self.covering:
            return False
        if self.order_due() <= self.moves_left():
            return False
        named_ids = _k_texts(args)
        held = self.trolley.units
        if name == details and any(v in held for v in named_ids):
            return False
        if not looked:
            return True
        return not any(v in held or self._k_ever_passed(v) for v in named_ids)

    def _k_cart_booked(self, args, obs, notes) -> None:
        before = dict(self.trolley.units)
        self.trolley.changed_at = self.wire.tick
        self.order_vetted = False
        notes += [self.trolley.booked(args, obs)]
        self._k_try("after_put_error", lambda: (self.cart_moved(before), self.mark_fresh(obs), self.after_put(args, obs)))
        target = self.trolley.aim(args)
        if not (self.trolley.is_put(args) and target):
            return
        self.refusals_seen[target] = self.refusal_n
        _k_bump(self.puts, target)
        units = self.trolley.count_arg(args)
        if (self._k_passed(target) and (units is None or 0 < units) and 0 < self.trolley.held(target)
                and target not in self.dead_lines()):
            notes += ["An independent check found %s meets every requirement in force under the value rules: keep "
                         "it unless a new change rules it out." % target]

    def _k_audit_due(self) -> bool:
        spare = self.moves_left()
        if self.trolley.units or not Knobs.LATE_STEPS < spare <= self.step_cap // 3 + 2:
            return False
        return not self._k_fresh_pass()


class Episode(FrameMixin, CartworkMixin, SelfDriveMixin, WalksMixin, LoopMixin):
    """One shopping errand: every part's state and methods on one object."""

    def __init__(self, payload: dict) -> None:
        self._frame_setup(payload)
        self._cartwork_state()
        self._selfdrive_state()
        self._walks_state()
        self._loop_state()


def agent_main(problem: dict) -> list:
    episode = Episode(problem)
    try:
        episode.execute()
    except Exception as error:  # noqa: BLE001
        episode._k_fail("crash", error, 500)
        if not (episode.wire.done or episode.wire.broke):
            episode._k_try("crash", episode.finish_now, None, 500)
    return episode.log


__all__ = ["agent_main"]
