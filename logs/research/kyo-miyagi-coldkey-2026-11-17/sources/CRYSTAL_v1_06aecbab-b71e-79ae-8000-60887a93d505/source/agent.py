from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"
PLAN_MODEL = "anthropic/claude-sonnet-5.5"

TASK_SECONDS = 265.0
COST_SOFT = 0.035           # dollars spent in one episode after which the checks run on the cheaper model
COST_HARD = 0.055          # dollars spent in one episode after which only the finishing tools remain (race 12: both
                            # runs reached the $5 key cap, one with 26 tasks left; 2 of 24 Arena tasks past 0.065 passed)
PAGE_CHARS = 700            # strings on a details page are cut to this in the model's context
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
MAX_TURNS = 80
HOP_LIMIT = 8               # pages followed along one chain of page references
NO_THINKING = {"enabled": False}      # judge calls answer with JSON only; thinking tokens were cutting the answers off
LIGHT_THINKING = {"effort": "low"}
PAGE_OPENS = 5              # candidate pages opened per batch (the side-by-side tool's cap) when the search cannot pin the listing
HOP_BATCHES = 3             # batches of candidate pages opened per hop
ROW_CHARS = 240             # strings in a search result are cut to this (facts live on the details pages)
PAGE_SWEEP = 48             # a search with up to this many matches is read through to its last page before a decline
SWEEP_PAGES = 5             # pages the cheapest sweep reads on its own, for the searches that found the carted picks
KEEP_TURNS = 12             # tool results older than this many messages are shortened in the model's context
TRIM_BLOCK = 24             # ... in blocks of this many messages, so the cached prefix changes rarely
TRIM_CHARS = 900
WINDOW_CHARS = 160000       # ... and only once the conversation is this long: a shortened result rewrites the cached
                            # prefix after it, and on the main model a write costs 12x a cached read

SYSTEM_PROMPT = """Your job: act for a single shopper in a simulated shop, working only through the tools you have been handed. Grading is pass/fail. A pass needs a finish with nothing wrong in it: what you order satisfies each requirement written in the request, each hidden need the shopper discloses when you ask, and each change they announce during the episode; the unit count and the spending limit hold; when the request wants the cheapest (or the best), that is what you picked; and you obeyed every procedural rule it sets. When no listing can satisfy all of it, finish the way the request prescribes for that case: if it asks you to buy the nearest match, order whichever listing falls short on the fewest points; if it asks you to offer or name one, purchase nothing and name that listing in your no-purchase finish; in any other case, decline.

JUDGING A LISTING (this is what the score turns on)
- Honour exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The listing's category settles the requested product type; never insist that a property restate the category's name. The item does have to be the requested type: where the shop assigns categories, it must belong to the one whose label fits the request's wording as a whole (a request for "A or B" fits a category named "A & B", not a narrower neighbour named "B"). It must be in stock if the request wants things on hand, and price times quantity must stay within budget (equality is fine).
- A requirement counts as satisfied only when the listing's own property for it gives the value. Key spellings vary (spaces, dots, underscores, prefixes), and one property may show up under several keys; any key naming that property is valid. Values match on meaning: one entry of a comma-separated list suffices; singular equals plural; near-synonyms count; letter case is irrelevant. A word that just shares a beginning does not count, nor does a wider or different class (girls is not women, plain unisex is not female, for example).
- A property whose value lists several options, the required one among them (a comma list), simply matches; that is not conflicting evidence. Evidence conflicts only when the listing's own keys for the property give different, incompatible values. Only in that case, and only if no cheapest or best rule settles it, favour a candidate with clean evidence.
- Fall back on the title or description as evidence only when the listing has no property key at all for that requirement. If the listing's own property says otherwise, it fails even though the title or description gives the value.
- A value filed under some other, related property is no evidence (a length field cannot satisfy a style requirement, a lining material cannot satisfy the material).
- Numbers: convert units and compare numerically; limits include their boundary value; a range qualifies only when all of it satisfies the limit.
- If a listing comes in options or variants (sizes, colours, versions) carrying their own ids, prices or stock, treat each option as a separate candidate: the option's own value, price and stock count decide, and the cart line has to name the chosen option in the way the cart tool's schema requires.

CHEAPEST AND BEST
- For a best request on some measure (most, largest, longest, highest and so forth), compare ALL fitting items on that measure the same way and take the extreme. An item whose listing gives no value for the measure cannot win.
- For a cheapest request, the order has to be the lowest-priced item out of ALL items meeting every requirement. One fitting item seen is not enough. Apply price filters when a tool has them, push the price limit down until all remaining rows are visible, order the visible candidates by price yourself, and verify them from the lowest price up; the first that fits completely, with every cheaper one examined and rejected, is your answer. Hold the next fitting ones as backups.

CLOSING THE TASK
- Budget your steps: make your first order attempt while seven or more steps remain, so a refusal can still be dealt with. Once 7 steps remain, only the cart, page, stock-check and finishing tools are left. Settle on a choice once: do not flip the cart back and forth, as every cart change uses a step.
- Maintain a ranked list of verified fits with at least one backup, since your top pick can sell out at checkout.
- Ahead of every order attempt: the cart contains exactly the lines the request calls for (normally one: the chosen item at the required quantity; one line per item when several different items are requested) and nothing more (remove any other line); and since your last cart change you have re-read the chosen item's stock and price and it is in stock and within budget.
- Declining is hardly ever correct: in almost every task the shop carries an item for each requested product, and a decline scores 0. When no listing gives a requirement exactly under its own key, the intended item usually gives the value under a related key (a "bore: 12 mm" for a 12 mm hole size), in its title, options or description, or in close wording (matte silver for brushed silver): order that nearest fit instead of declining. Decline only after an exhaustive search, and only if every candidate's own records plainly contradict a stated requirement, a disclosed need, the budget or stock (another value under that same key, a price above budget, out of stock). Uncertainty, an absent key, a missing nice-to-have or a requirement you merely assumed is no grounds for declining; order the best fit. With several products, one that is hard to match never justifies abandoning the whole order: put its nearest fit in the cart alongside the others. Follow the request's declining rule to the letter. If the request forbids offering substitutes, the decline remark must not name, describe or point at any listing: no ids, no titles, no prices.
- Every reply needs at least one tool call. Keep reply text brief: your current requirements, verified fits with price and stock, and what you will do next.

WHEN THINGS CHANGE
- During the episode the shopper may change things: a new budget (up or down), a requirement removed, swapped or added, a price rise, an item going out of stock. A single shopper line may hold several changes. Shopper lines may turn up inside any tool result (an extra field containing sentences) or come back with an order attempt. Every change binds from that point: re-test your pick and backups against it, and if the request wants the cheapest or best, choose afresh among the new acceptable set.
- Besides announced changes, items may go out of stock or change price without notice. Hence stock and price get re-checked immediately before each order attempt; an item whose present price times quantity exceeds the present budget no longer fits.
- An order attempt may be REFUSED. The reason then sits in some field (refusal, complaints, objections or the like), even when the inner result looks like success (accepted: true, placed: true, executed: true). A successful order ends the episode, so as long as the episode goes on, nothing has been bought, whatever the result claims. The episode carries on with the cart as it was. Read the refusal, apply each change it gives, take out of the cart whatever is no longer acceptable (an item now sold out, for example), put in the next verified fit, re-check stock and price, and order again.

ASKING THE SHOPPER
- Requirements come only from the request's text plus what the shopper discloses or changes. Do not invent any from the backstory (an age group, a gender, a size, the intended user): those are not requirements, and a listing is not wrong for ignoring them. If one could matter, it might be a hidden need, so ask.
- Expect needs the shopper never wrote down (typically one or two, occasionally zero). Each comes out only when a message asks about that property, and ordering before all of them are known scores 0 even if the item happens to suit. So use every question message you are allowed, and send them all before ordering. The shopper replies only about the first-named property they have in mind and disregards the rest, so each message covers properties of ONE product, likeliest first, worded "Which <property>, <property> or <property> would you like the <product> to have?" (three at most; just one if the shopper has asked for one thing at a time). Spell each property the way listings spell it, call the product by the plain noun the request uses, and give no sample values. Do not instruct the shopper, do not bring up labels, positions or rules, do not ask about two products in one message, and do not ask open questions ("what should I know?"): a question gets an answer only when it names a property.
- With several distinct products requested, each question must say which product it concerns; leaving that open muddles the shopper. Distribute questions across the products, because hidden needs tend to sit on different ones: give each product at least one question whenever the message allowance permits.
- Shopper messages carry questions only (or your reply to a question the shopper put to you). Do not message them to report findings, purchases or a decline; the finishing tool does that.
- If you have to write questions yourself: view two or three listings' details first, then ask about properties the backstory hints at (who it is for, its look, where or how it gets used), then about properties the listings carry that vary across candidates. Never ask about anything the request already decides, and never go past the message allowance.
- When the shopper says a message made no sense to them, or wants it explained, simplified or reworded, neither explain nor resend it: they most likely do not care about that property. Move on to another property (still naming the product).
- An answer like "no preference", "don't mind" or "anything works" leaves that property open. An answer stating a preference or a limit becomes a firm requirement from then on, in its exact wording (numbers included).
- If a planner is present, it picks and sends your questions automatically once you have viewed the details of two or three listings of the requested type (of every type, when several products are requested), so do that among your first moves. The answers then show up in the conversation along with their meaning; treat each disclosed need as firm and search again with it joined to the stated values.
- The request may impose finishing rules: something to do with the shopper or the shop before or at the order, besides locating and buying the items (getting the shopper to approve the final cart, for instance). The messages those rules need are held back for them, and the rules are checked and performed for you when you call the order tool on the final cart, with the order following immediately. So do not use those messages for questions and do not decline because such a step is still pending: assemble the final cart and call the order tool. Altering the cart after such a step can cancel it.
- A need (or a written requirement) may come as a POINTER at another listing's page: "the same as the value given by the <kind> page that <shop> sells, with <facts>", or "anything but" that value, or its counterpart (the value that page says goes with, pairs with or answers to its own). The value is then taken from that page, which may in turn pass the property to a further listing ("for <property>, refer to the <shop> <kind> with <facts>"), hop by hop, until some page gives the value directly. All pages on that chain must be opened with the details tool (or the side-by-side tool) before ordering, since grading checks they were read. When a pointer is heard or written, a resolver walks it for you and reports the value; if it reports none, walk it yourself: search the shop's name, open the listing of that kind with those facts, read its notes, follow the next pointer, and continue.

PROCEDURAL RULES IN THE REQUEST
- Things the shopper only hopes for ("ideally", "would be nice", "if possible") are not requirements: let them break ties among items that already satisfy everything, and never let them override a cheapest rule.
- If the request leaves open something only the shopper can settle (no price limit given, say, or two wishes that may conflict where the shopper must rank them), ask about it before you choose.
- A request can dictate how you work: how many listings to inspect or compare before choosing, what to verify, what to tell the shopper, what a remark should contain, how to decline, how many tool calls or messages are allowed. Grading treats these like requirements, so carry out each literally and in full. A rule to inspect or open N listings means N fully fitting listings, each opened through the details tool.
- FIT RULES: wording such as "make sure the <accessory> fits the <anchor> you end up buying" (or anything saying one product has to work with whichever other one gets bought) is settled from the pages, never from how titles read. The anchor's page lists its identifiers (a model number, alternative names it sells under, a part series or code family that fits it), and an accessory that fits names one of those on its own page (in a works-with or fits list, as its own part code or series, now and then in its title). Pick the anchor first and open its page; next search the accessory using that identifier plus the accessory's type word, and buy only an accessory whose page names it. If an accessory's page refers you to another part's page for more, open that page before ordering. These links get checked for you at the order.
- CONDITIONAL REQUIREMENTS: a requirement on product <B> that applies only when product <A> has some value (if the <A> has it, the <B> must satisfy the requirement) binds the <B> only when the <A> you purchase shows that value on its own page. Select the <A> on its own requirements (and any cheapest or best rule), read its page, and impose the <B> requirement only when the condition is true; the order checks it for you.
- Should the shopper ask you something (confirming a change, say), it gets answered for you while messages remain; the change itself is binding regardless.

HOW THE SHOP WORKS (reference)
- Each shop names its tools, arguments and result fields its own way. Study the tool descriptions and JSON schemas, and pass precisely the argument names and permitted values they specify. Use the first answers to learn which tool reveals which facts (price, stock, detailed properties).
- Suppose one tool gives back a listing's page while another gives back only a slice of it (its properties alone, say). Then only the page tool counts as opening that listing, and grading looks for it: each listing you purchase and each page along a reference chain has to be opened through the page tool (or the side-by-side tool, for a group), even when you read its properties through the other tool.
- Early on, learn which category labels this shop gives the requested product type (listings show them). If similar labels sit side by side, pick the one whose label fits the request's whole wording, and do your searching and buying inside it.
- Searches may be restricted: a ceiling on visible rows or pages, ordering by id instead of relevance or price, all words mandatory, filters that function in a single mode only. Set the reported total against the rows actually visible to you and tighten (more precise words, category or price filters, further pages) until the whole relevant set has passed before you. Some shops read only the first one or two words of a query. In the status line, the [search tool] part records what this shop's search has revealed about itself so far (words read, whether all words must match, row ordering and depth): shape every query to fit it, and plan your remaining searches around it.
- Calls that do not depend on each other's output may share one reply (say, viewing several listings' detail pages together); they execute in sequence. An order or a decline always travels alone.
- Where a tool exists that returns full details for many listings in a single call, prefer it to one-at-a-time viewing, since it spends fewer steps.
- Each tool call, failed or erroring ones too, consumes a step, and steps run out. Thinking costs nothing; steps do."""

SETUP_PROMPT = """Role: intake parser for a shopping assistant. Below come the shopper's request and the store's tool list; turn them into one structured record.

Rules:
1. File every sentence of the request somewhere: requirement, priority, soft want, rule, budget, hint, or plain story. If you cannot tell whether a sentence is a rule, file it under "rules".
2. A firm requirement is a "property: constraint" string carrying exactly the shopper's value, bound, exclusion or fixed set of allowed values. With several different products, lead each string with that product's short name: "<product> <property>: constraint".
3. Never put price, stock, quantity, budgets or wishes in "requirements", and never repeat the product kind or category there; words already inside the category name (a season, an occasion, an audience) are covered by it. One exception: a cost limit applying to just ONE of several products, separate from the overall budget, is a requirement, written "<product> unit price: at most N".
4. A phrase that pins the product down by something its category name lacks is a requirement on that property: "a mug of the kind that suits a picnic" -> "occasion: picnic"; "a backpack in a vintage style" -> "style: vintage"; "shoes right for a gala" -> "occasion: gala".
5. Wishes (ideally, preferably, would be nice, if possible) go in "soft_wants", never in "requirements".
6. A name built from two joined words ("knives and forks", "nuts & bolts") is ONE product when the request treats it as one kind.
7. Something sold as a single unit, like a pair of gloves or a set, counts as quantity 1.
8. Saying money is short or every penny matters only explains the price cap; it is not a request for the cheapest.
9. Remarks about what the shopper will do on their own are story, not rules.
10. Facts that pin down another shop's page ("the jug Store Y sells, with a steel handle and a red cap") describe THAT page, not the product wanted: they belong in the reference's "facts" only, never in "requirements".
11. A requirement that depends on another requested product's value goes in "conditions", never in "requirements".
12. A sentence saying the nearest listing is wanted when nothing fits is not a finish step and needs no message of its own.

Output: exactly ONE compact JSON object (no indentation), nothing before or after it, with these keys:
"product": the requested product kind in the request's own words; name each one when several different items are asked for.
"quantity": integer count of units of one listing to buy; 1 unless the request asks for more.
"items": for several DIFFERENT products, one {"product": short name in the request's words, "quantity": integer units of it} per product; [] when only one kind is wanted.
"requirements": list of firm requirement strings (rules 2-4).
"soft_wants": what the shopper would like but can do without.
"conditions": one object per sentence making a requirement on one requested product hinge on what another turns out to be (if <A> shows a given value for some property, then <B> owes a requirement): {"if_product": A's short name, "if_property": the property of A, "if_values": [A's values that the sentence lists], "if_not": true for the "B needs it UNLESS A is one of those" form ("..., except when the A's <property> is ...", "unless", "but not if"), false for the "only IF A is one of those" form, "then_product": B's short name, "then_requirement": "<property>: <constraint>" for B, "text": the sentence quoted}. [] when none.
"fit_rules": one {"accessory": short product name, "anchor": short product name, "text": the sentence quoted} per sentence like "make sure the <accessory> fits the <anchor> that gets bought", or any line demanding that one requested product go with, or be compatible with, whichever other requested product ends up chosen. [] when none.
"references": one object per requirement or need phrased by pointing at some other listing's page (equal to, or anything except, what a named shop's page of some kind shows, that page being identified by a few facts; matching the listing a named shop sells; "of a shop's two lamps, take the lighter one's figure and avoid it"): {"shop": the shop or seller named, "kind": the product kind on that page, "facts": {property: value} identifying that page, "product": short name of the requested product it limits, "property": the property limited, "relation": "same" if our value has to equal the page's, "different" if it must differ ("anything but", "not that one"), "counterpart" if it must be what the page pairs with its own value (its counterpart, partner or opposite number, as that page names it), otherwise a few words stating the rule, "text": the sentence quoted}. [] when none.
"price_cap": highest total price allowed, as a number; null if none is given.
"currency": unit of money used in the request, or null.
"in_stock_only": true if only items on hand / in stock are acceptable.
"priority": "cheapest" only on an explicit ask for the lowest-priced or cheapest fit; "best" only on an explicit ask for the top item by a named measure; else "none".
"priority_detail": the measure behind "best"; "" otherwise.
"message_budget": integer cap on messages you may send the shopper, or null if unstated.
"step_budget": integer cap on moves or steps, or null if unstated.
"rules": all procedural instructions to the assistant, copied word for word (how many listings to view, what to check or report, how to decline, what to avoid, what to include).
"process_checks": for each such rule that demands a store tool be used in a fixed manner prior to ordering (looking over, opening, comparing or inspecting something some number of times, or once the cart is settled), one {"rule": quoted rule, "tool": the exact tool name that performs it, "minimum": integer calls of it needed before the order (1 if unspecified), "after_cart_final": true if those calls have to follow the last cart change}. Skip budgets, stock, cheapest/best choice, limits on messages or steps, shopper questions, declining, and pointers to other listings (covered elsewhere). [] when empty.
"finish_steps": one object per job the request sets around checkout besides finding and buying the items (whether with the shopper or a tool, before or at the order; excluding questions about what the shopper wants, and excluding budgets or caps): {"rule": quote of the sentence, "action": one sentence on what to do, "timing": the moment it is due (e.g. "once the cart is settled, just before ordering"), "messages": integer count of shopper messages it takes, 0 if none}. [] when empty.
"decline_no_names": true if the request's decline rule bans offering, naming or suggesting alternatives or substitutes; false if it wants something named; null if it sets no such rule.
"closest_if_none": true if the request says the nearest listing should be put forward or bought when nothing matches; else false.
"closest_finish": what that nearest-listing sentence wants: "order" to buy the closest listing; "name" to report no clean match and offer, suggest or name the closest one with no purchase (so the finish is the non-buying one, its message naming that listing); null if the request lacks such a sentence.
"decline_choice": if ONE tool covers both endings (purchase, and quitting with none, picked by an argument value), the exact allowed value of that argument meaning NO purchase, as listed by the tool; null otherwise.
"ask_first": what only the shopper can tell you and should be asked before choosing, e.g. the budget if no price limit appears, which of two wishes wins when the request warns both may not be achievable (list both wishes in soft_wants then, not requirements), or anything the request says to confirm with the shopper. [] when none.
"hint_topics": short phrases for what the story suggests the shopper quietly cares about, each phrased as the kind of property it points to, most telling first (the one the story lingers on or stresses), e.g. "who it is for: age group or gender", "where it goes: setting or occasion", "how it should look: colour, pattern or finish", "what it runs: operating system". If the request invites questions about a topic (who it is for, which room), convert that into the properties it implies, never an open question. [] when the story hints at nothing.
"roles": map each role to the exact tool name from the list, or null where no tool suits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "details" returns ONE listing complete (its whole page); "view" is a different tool showing just a slice of one listing (only its properties, or only stock and price), null if absent. "message" talks to the shopper. "cart" puts items in or takes them out of the cart. "order" purchases / checks out. "decline" closes the task with no purchase.
"""

INTERPRET_PROMPT = """Role: you decode the shopper's latest line for a shopping assistant.

Rules:
1. Read literally but generously. Folksy, figurative or roundabout wording that names a value means the shopper firmly wants it; count a value as excluded only when they plainly say to avoid it.
2. A number plus a comparison (at most, at least, under, over, no more than) is a limit; keep its unit. The comparison right beside the number decides, even in back-to-front sentences: "pass on anything that goes beyond a 40 cm ceiling in depth" = depth at most 40 cm; "stay clear of any that fall below a 2 kg floor" = at least 2 kg; a negative verb paired with "at most" still means at most.
3. Report every change one line carries. A replaced requirement is both dropped (old value) and a new need (new value).
4. When the shopper says which of two wishes matters more, the favoured one is a need. A mere wish ("ideally", "would be nice") is not.
5. Any budget the shopper states is new_budget, and it replaces the old one whether higher or lower, even when announced as good news.
6. "Don't mind", "no preference" or "anything works" puts the property that was asked about in no_preference. It covers only that property: it never drops a stated requirement, a price limit or a rule, and if the assistant's message asked about no property it changes nothing.
7. With several different products requested, start every string in needs and requirements_now with the short product name it belongs to: "<product> <property>: constraint" (shoppers often say "<property> of the <product kind>").
8. requirements_now uses only values the request states as requirements or the shopper states in a reply; never infer one from the story or your own reading of the request.

Output: a single JSON object, no other text. Keys:
"meaning": a brief sentence giving the shopper's intent.
"needs": firm requirements given just now, each "property: constraint" holding the shopper's exact value, bound, exclusion or fixed set of allowed values.
"no_preference": properties the shopper called unimportant to them.
"dropped": earlier requirements the shopper has let go.
"new_budget": the revised top total price as a number, or null.
"new_quantity": if the shopper alters the unit count, {"product": which product ("" when only one kind is requested), "quantity": the revised integer}; else null.
"unavailable": true if the shopper reports the item is no longer buyable (sold out, gone, out of stock, button disabled); else false.
"requirements_now": the whole current set of firm requirements after all the shopper has said, combining the requirements so far (given below) with the shopper's explicit statements, each "property: constraint". Carry each earlier one forward unless dropped or replaced; where a value was replaced, only the latest survives; include every need just revealed. Exclude price, stock, quantity and the product category.
"unclear": true if the shopper didn't get the assistant's message or wants it explained, simplified, reworded or repeated; else false.
"one_at_a_time": true if the shopper objects that the message held too many questions and wants just one per message; else false.
"reference": if a need or change points to some other listing's page (equal to, or anything except, what a named shop's page of some kind shows, that page being identified by a few facts; matching what a named shop's listing offers), {"shop": the shop or seller named, "kind": the product kind on that page, "facts": {property: value} identifying that page, "product": the product it limits ("" when one kind is requested), "property": the property, "relation": "same", "different", or "counterpart" if it must be what the page pairs with its own value (its counterpart, partner or opposite number), or else a few words stating the rule}; else null. It still counts as a need: until the page is read, list it in needs as "<property>: <relation> as the page of the <shop> <kind> (<facts>)".
"reply": if the shopper puts a question to the assistant or wants something confirmed, a short plain answer to send that states the change in substance (e.g. confirming the revised value); else "". Must be "" when unclear or one_at_a_time is true, and when the line merely responds to the assistant's message (a value, a "don't mind", an acknowledgement): a plea to clarify the assistant's own question needs no answer, and each message sent spends one of the few the shopper will reply to."""

VERIFY_PROMPT = """Role: last gate before a shopping assistant orders. A task earns 1 only when the ordered item satisfies every requirement listed below; the records shown are your sole evidence.

Rules:
1. Scope: test the listed requirements and nothing more; never make one up from the story (an age group, a gender, a size, the intended user). A wish ("ideally", "would be nice") marked optional is not one of them.
2. Several products requested: hold the item only to its own product's requirements; any requirement headed by another product's name is not its concern, and the quantity used is its own kind's.
3. Kind: it has to be the sort of product asked for. If listings carry a category label, the right label is the one fitting the request's phrasing taken whole, never a tighter or adjacent category.
4. Evidence: a requirement holds only if the item's own property for it gives the value. Key spellings vary (spaces, dots, underscores, prefixes) and one property may appear under several keys; any key naming that property is valid.
5. Matching: go by meaning. One entry in a comma-separated list is enough; singular vs plural, close synonyms and letter case make no difference. Lookalike words and wider or different classes are no match.
6. Title and description are fallbacks, used only for a requirement with no property key on the item. When the item's own property shows another value, that requirement fails regardless of the title.
7. Some other, related property carrying the value is no evidence.
8. Numbers: convert units first; a limit includes its boundary value; a range qualifies only if it lies wholly within the limit.
9. Exclusions and closed lists are binding.
10. Unit price multiplied by quantity may not exceed the budget (equal passes). Shown as out of stock = fail.
11. Records that conflict (on price, stock, etc.): trust the latest.
12. fits=false needs a requirement the records clearly fail or that no record covers at all; an absent stock or price record never fails the item.

Output: a single JSON object, no other text: {"item": id of the item, "fits": true | false, "problems": [one brief string per requirement that fails or has no record]}."""

# a line of a several-product cart, checked alone (race 10 replay: the checker failed 52% of the right lines, judging them against

# another product's requirements, the fit and if-then rules or the cart's total; once the line's own product is named, 25%)
LINE_PROMPT = """

Task: this cart line covers only "{product}", one of several requested products, at quantity {quantity}. Grade it for that product alone.
Check:
- Apply a requirement here only when it names "{product}" or names no product at all; one that names a different requested product belongs to another line, however this listing looks.
- Skip every rule linking two requested products (a fits-with / works-with rule, or a value demanded only when another product carries some value); the whole cart is graded on those elsewhere.
- Skip the cart's grand total: the budget test here is this line's price multiplied by its quantity.
- Quantity means how many units of this listing sit in the cart; a pack size or count among its facts is a listing property, not the quantity, unless some requirement concerns it.
Answer with: the same verdict format as above, for this line only."""

PLAN_PROMPT = """Task: plan what an assistant asks a shopper. Zero, one or two secrets may be held; a secret is a fixed value for ONE attribute of ONE wanted item, and it comes out only if a message targets exactly that attribute. Off-target asks just get "don't mind" (several items: possibly a repeat request, equivalent). A two-topic ask is answered with "one thing at a time" and gets nothing. Buying before every secret is out scores 0, hence aim each ask at a secret. Output QUESTION_COUNT separate attributes, likeliest first (fewer if sensible ones run out).
Check (higher rank wins):
0. Items listed below as MUST BE LEARNED FROM THE SHOPPER and still unasked: these lead, each one brief plain sentence. Empty list = no such step; in that case you must not raise cost, budget or spending ceilings unprompted (the request handles them), must not invent asks, and must keep each ask to ONE attribute present in the listings.
1. Narrative cues. Imagine the buyer: who wears or operates it (gender, age group, size), the when and where (occasion, setting, season), appearance (colour, pattern, finish, style), compatibility (operating system, connector, voltage). Lines that aren't explicit demands are there deliberately and hint at an unstated concern; the listing attribute that answers such a hint is your best ask.
2. Dividers: listing attributes, from the data below, shared by most of that kind yet with few distinct values (style, type, height, closure, fit, format, capacity class). Bad: constant across all, or unique per listing (identifier, title, model number, size chart). Any narrative cue, even weak, outranks dividers; past cues, rank by carrying count (numbers below). Statistically, after dropping settled attributes, a majority of secrets sit in the kind's four commonest attributes and 80% in its eight commonest. Colour, material, pattern go early only when that kind's listings commonly list them; asking colour of a part, a component or a food wastes a turn. Specs and rare attributes trail. Asking about a kind's "style" or "type" matches its own style/type key (e.g. "kettle style") even when spelled with the kind's name or a seller prefix.
3. Logistics trail or are dropped: warranty, shipping, delivery, seller, brand, maker, shop, price, stock, quantity. Not logistics, and fair game when values vary: pack type, pack size, condition, count, capacity.
- Exclude anything the request fixes (its explicit demands, price, stock, quantity, item kind).
- Exclude repeats per item; an unanswered ask still uses that attribute up, so advance (exception: that ask omitted the item's name).
- Exclude near-twins (variant spellings, two style flavours), vague buckets (features, options, details, other, care) and attributes no listing has.
- Got "one thing at a time"? It was too broad: pick a tighter member of that family, with the listings' label for it.
- Multiple items: one item per ask, every item covered at least once, favouring whichever has fewer asks so far. Secrets typically sit one per item, so an item whose secret is out gets no more asks while another's is still hidden. Source each item's attributes from its own kind's listings (the menu groups by listing label if present).
- Phrasing: "Which <property> would you like the <product> to have?". Attribute spelled as listings spell it (spaces replace underscores/dots, drop seller prefix, keep all words of the key, never one word alone: "memory type", not "memory"). Item = bare noun from the request ("the rug", "the curtains"), stripped of adjectives, materials, colours or values; items sharing words each get their full kind name as worded in the request, never a value to tell them apart. No example values, no padding, no second ask. Step 0 asks: one brief plain sentence.
Answer with: ONE JSON object only, {"analysis": "3-4 sentences, before anything else: the buyer and purpose the narrative implies, which listing attributes those cues point to, and which further attributes separate the options", "questions": [{"product": "...", "property": "...", "why": "story hint / splits candidates / learn first", "question": "..."}, ...]}."""

# how many properties ride in one message to the shopper. Race 14 ORO logs (every agent): a need placed FIRST in a message
# naming 2-5 properties was disclosed 90-100% of the time (alone: 95%), one placed further back 70-80% with no decoy ahead
# of it; hence the strongest property leads and two more follow. In races 12-13 a message touching two of the shopper's
# subjects drew "one at a time" (the message wasted): that reply flips the episode to a single property per message, and
# the properties of the wasted message are then asked one by one
LIST_SIZE = 3
LIST_NOTE = """

Task: batching is on. Each outgoing message packs at most GROUP_SIZE attributes, one item only, auto-worded as "Which <property>, <property> or <property> would you like the <product> to have?"; replies address whichever listed attribute the shopper holds that comes earliest.
Check:
- Deliver PROPERTY_COUNT entries in "questions" (not QUESTION_COUNT), a single attribute apiece, shaped as specified below.
- Rank everything likeliest first; positions 1..QUESTION_COUNT open each message and are answered best, the remainder tags along after.
- Request demands single-topic asks? Add "one_per_message": true and stop at QUESTION_COUNT.
Answer with: the JSON format below."""

SWEEP_LAST = False
SWEEP_NOTE = """

Task: your list's closing entry becomes a catch-all ask.
Check:
- Template: "Do you have a requirement for the <product>'s <property>, <property> or <property>?".
- Fill it with 2-3 additional likely attributes of ONE item, untouched by any earlier ask, spelled per listings, item as bare noun.
Answer with: its "property" field = those attributes joined by ", "."""

FINISH_PROMPT = """Task: final process review before the assistant orders. On top of picking the right items, the request imposes rules on how the purchase is wrapped up. Given below: those rules, the request, the store's tools, and the assistant's step-by-step log so far (searches, opened pages, cart edits, shopper messages with their replies, order attempts). From the log, judge whether each wrap-up rule is satisfied for the order now pending.
Check:
- Cart-related rules are satisfied only by actions logged after the most recent cart edit. A step the shopper did not follow (they asked for it again, or in plainer words) is not done.
- Allowed next calls exclude the order tool, the decline tool, searches and cart edits.
- Messages reach a simple reader: keep each to a single short plain sentence about the order or the cart overall, worded as a question when it requests something. Mention no product, listing or property and list no items unless the rule itself requires naming them; the reader takes any message naming a product as a question about that product.
- Numbers the shopper must be told (total, price, count): take them character for character from the latest store result after the most recent cart edit that shows them, otherwise from CART NOW below (its sum is the cart total at the newest prices seen, so no extra step is needed to read it again). No unit changes, no rounding, no recomputing (the request's currency word may come after the number).
- Fewer messages left than rules that need one: fold several rules into one message, one short clause each, in a single sentence. Rules the store's tools can handle (a stock check, a cart check) are met with that tool, not a message.
Answer with: text holding ONE JSON object only (it describes the next call; make no tool call yourself):
{"analysis": "first: rule by rule, whether the log shows it done and at which step",
 "next": null if all rules are met (or none can be met any more: no messages or steps remain for it); else the single tool call meeting the earliest unmet rule, as {"tool": exact tool name, "arguments": {...} precisely per that tool's schema},
 "then_order": false only if that call's result might alter WHAT gets bought (it reveals something the choice hinges on); otherwise true, and the order then comes straight after the call provided its result changes nothing and was understood}."""

AUDIT_PROMPT = """Task: a "no match" decline is pending. Declining scores 0 and stores almost always stock something per wanted item, so verify the records below against the demands first.
Check:
- Demands = the listed ones only; nothing inferred from the narrative. Multiple items: each listing is tested solely on its own kind's demands, and every item needs a pick.
- Pass = the listing's own key for that demand shows the value (any key spelling, a single member of a comma series, plural or singular, near-synonyms OK). If no listing has that key, whichever shows it via a related key, the title, options, description or similar phrasing is the nearest candidate.
- Numbers: convert units, then compare; bounds are inclusive. Cost must sit within budget; shown-as-unavailable fails.
- Discard only on clear conflict in a listing's own records: another value under that exact key, over budget, or no stock.
- No full pass anywhere: nearest = fewest failed demands (unavailable and over-budget are one failure each); equal ones resolve to the lower price.
- Budget-only case: full-pass listings were seen (multiple items: one each) but cost above budget (multiple items: the cheapest such cart too), and no in-budget listing passes all. Here budget alone blocks, and the decline is correct.
- Output fields: "possible_fits" = up to 5 ids, strongest first: listings that fit, or failing that the closest ones; with several products, lead with each product's top pick. "over_budget_fits" = ids that satisfy everything but price, only in the budget-only exception, otherwise []. "budget_only" = true in that exception, false otherwise. "all_met" = true if each pick in possible_fits passes every listed requirement plus budget and stock (a clean fit; with several products, one such pick per product), false if they are merely closest. "reason" = per pick, one sentence citing the records behind each requirement.
Answer with: ONE JSON object only, {"possible_fits": [...], "over_budget_fits": [...], "budget_only": true or false, "all_met": true or false, "reason": "..."}."""

POINTER_PROMPT = """Task: a demand takes its value by reference to a separate listing's page. Parse the sentence appended after this text.
Check:
- "property": the attribute whose value comes off that page.
- "relation": "same" = must match the page's value; "different" = must not match; "counterpart" = must be its partner, counterpart or opposite number; anything else = a short rule phrase.
- "shop": proper name of the seller or shop owning that page. "kind": item kind shown there, in the sentence's own words.
- "facts": {property: value}, one per identifying detail supplied.
- "search": 2-4 words for keyword lookup of that page, shop name leading, then its most distinctive fact value.
Answer with: ONE JSON object only, {"property": ..., "relation": ..., "shop": ..., "kind": ..., "facts": {property: value}, "search": ...}."""

HOP_PROMPT = """Task: follow a reference trail across store pages. Supplied below: each opened page (id, shop, kind, facts, notes) and the target reference, namely a shop, an item kind and identifying facts.
Check:
- Target = the page matching ALL given facts through its own facts or notes: shop, kind, every fact by meaning. Shop matches if the page's seller is it, or if the page's note presents itself as that shop's listing ("<shop>'s <kind> ...").
- Same-shop pages with a clashing fact or note ("not the hooded one", "unlike ...", another value) are decoys, never the target. Nothing matches everything: "page": null, so further pages open.
- Trail ends where notes give the page's own value. A forwarding page never supplies the value; titles never do.
- "page" must be a listed id; "next" only repeats that page's notes: OTHER shop named there (never its own), the note's kind, the note's facts for that other listing.
- Target absent from the pages: answer {"page": null, "value": null, "next": null}.
- Fields: "analysis" first (page by page: shop, kind, agreement with the reference; then what the agreeing page's note says on the property). "page" = id of the page that truly IS the target, else null. "value" = short string when that page's facts or notes state the property FOR ITSELF; null when notes forward it ("for <property>, refer to / look at / go by / take it from the <shop> <kind> with <facts>, not this item").
- "next": forwarded case only, {"shop": ..., "kind": ..., "facts": {...}, "search": 2-4 lookup words, shop first}; else null.
- "required" (needs "value", else null): per THE REQUEST'S SENTENCE below, what the wanted item must carry or shun: the page value itself when the sentence wants it or anything but it; for counterpart, partner or opposite number, the value the page couples with its own (multi-pair pages, "a goes together with b, and c goes together with d": partner of the page's own value).
- "avoid": true when the sentence wants anything but that value (wanted item must NOT carry it); false otherwise.
- "rule" (needs "value", else null): short constraint THE REQUEST'S SENTENCE imposes, derived from the page: the value; its coupled value ("compatible with", "goes together with", "plays nicely with" etc. want that partner if the page names one); "anything but <value>" (only on exclusion; "nothing but it", "none but it", "only it" and dialect variants want that exact value); or a computed bound or margin ("at least 3 pieces", "between 18.2 and 493.8 gb").
Answer with: ONE JSON object only:
{"analysis": "...", "page": ..., "value": ..., "next": ..., "required": ..., "avoid": true or false, "rule": ...}."""

DEPEND_PROMPT = """Task: the cart below is about to be ordered. Some requested products are linked by the request, either by a fit rule (product A must fit or work with whatever B is bought) or by a conditional requirement (should A turn out to have a certain value, B must satisfy an extra requirement). For the cart exactly as it stands, decide each linked rule using nothing but the pages.
Check:
- Conditional rules: in force only if the depended-on product, in its cart version, displays one of the named values on its own page. When in force, B's line has to satisfy it the way any requirement is satisfied (B's own property gives the value); when the condition is not met, ignore the requirement.
- Fit rules are proven by written identifiers. On the anchor's page find its own: model number, other sale names, a fitting part series or code family, a parts list. A listing that fits repeats one of those on its page, as an entry in a fits / works-with list, as its own part code or series (title often), or by quoting the anchor's model.
- Infer from the anchor's page plus the other pages which kind of link this particular store uses. No anchor identifier on a listing's page means no fit, no matter what its title or compatibility claims suggest.
- When a line's page refers you to a different part for fit details, open that part's page first: it may state the line's fit itself ("part <the line's own code> also fits <model>").
- Output fields, per rule: "rule" = its number as listed; "anchor" = id of the anchor line; "dependent" = id of the dependent line (null if the cart has none); "applies" = false solely for a conditional requirement whose condition the anchor's page fails, true in every other case; "holds" = true if the cart satisfies the rule; "link" = [identifiers a fitting listing has to name, or the value the dependent has to carry]; "fitting" = [ids from the OTHER LISTINGS satisfying the rule and apparently their product's remaining requirements, cheapest first]; "search" = "2-4 keyword-search words that would turn up listings of the dependent's kind satisfying the rule (an identifier or the needed value plus the kind's word)".
- "see_also" = [codes of further parts the cart lines' pages refer the reader to, when none of the shown pages carries that code]. "analysis" comes first: per rule, the anchor line with the identifiers or value its page gives, then the dependent line and whether its page repeats one of them or satisfies the requirement.
Answer with: ONE JSON object only:
{"analysis": "...", "rules": [{"rule": ..., "anchor": ..., "dependent": ..., "applies": ..., "holds": ..., "link": [...], "fitting": [...], "search": "..."}], "see_also": [...]}"""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
# models that reject a call made with reasoning disabled. OpenRouter's Sonnet route replies 400 "Reasoning is mandatory",
# so at ORO each of them fell through to the next model after one failed call per task; starting with reasoning on keeps
# those checks on one model across routes (a route that accepted the call ran them on Sonnet instead)
_REASONING_ON = {JUDGE_MODEL, PLAN_MODEL}
_RETRY_WORDS = re.compile(r"in.?flight|rate.?limit|too many|retry|overloaded|capacity|temporar", re.I)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_G1_JOINERS = frozenset(("and", "or", "the", "of", "for", "a", "an"))


def _g1_plain_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _g1_has_digit(text: str) -> bool:
    return any(map(str.isdigit, text))


def _g1_idish(key: Any, value: Any) -> bool:
    """A value that reads as a listing id under a key named like one."""
    if not isinstance(value, str) or not _IDENT_KEYS.search(str(key)):
        return False
    return bool(_ID_HINT.match(value)) and _g1_has_digit(value)


def _g1_record_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(x, dict) for x in value)


def _g1_children(value: Any) -> list[Any]:
    """The nested values of a dict or list, last first (popping them off a stack visits them in their own order)."""
    if isinstance(value, dict):
        return list(value.values())[::-1]
    return value[::-1] if isinstance(value, list) else []


def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    root: list[Any] = [None]
    todo = [(value, depth, root, 0)]
    while todo:
        item, level, holder, slot = todo.pop()
        if level > 8:
            holder[slot] = None
        elif isinstance(item, str):
            holder[slot] = item[:limit] + "..." if len(item) > limit else item
        elif isinstance(item, dict):
            holder[slot] = shell = dict.fromkeys(item)
            todo.extend((v, level + 1, shell, k) for k, v in item.items())
        elif isinstance(item, list):
            holder[slot] = shell = [None] * len(item)
            todo.extend((v, level + 1, shell, i) for i, v in enumerate(item))
        else:
            holder[slot] = item
    return root[0]


def _line_of(args: dict[str, Any], ident: str) -> str:
    # the cart line, as text, naming this listing when the cart call carries its lines in a list
    hits = (row for group in args.values() if isinstance(group, list)
            for row in group if isinstance(row, dict) and ident in row.values())
    return next(map(_compact, hits), "")


def _key_words(key: str) -> tuple[str, ...]:
    return tuple(filter(None, re.split(r"[^a-z0-9]+", str(key).lower())))


def _g1_join(group: dict[str, set[str]], name: str, other: str | None, together: set[frozenset[str]]) -> bool:
    """Merge the groups of two keys unless they are one group already or a listing gave them different values."""
    if not other or other == name:
        return False
    mine, theirs = group[name], group[other]
    if mine is theirs or any(frozenset((x, y)) in together for x in mine for y in theirs):
        return False
    merged = mine | theirs
    group.update(dict.fromkeys(merged, merged))
    return True


def _g1_spelling(members: set[str], words: dict[str, tuple[str, ...]], counts: dict[str, int]) -> str:
    # spellings carried by at least half the listings of the most carried one compete: free of one- or two-letter
    # code words first, then the longest, then the most carried
    ranked = sorted(members)
    carried = [counts.get(m, 0) for m in ranked]
    top = max(carried)
    pool = [m for m, c in zip(ranked, carried) if c * 2 >= top] or ranked
    return max(pool, key=lambda m: (all(len(x) > 2 for x in words[m]), len(words[m]), counts.get(m, 0)))


def _fold_keys(names: list[str], together: set[frozenset[str]] | None = None, counts: dict[str, int] | None = None) -> dict[str, str]:
    """Canonical spelling per property key. A key equal to another key plus one extra word at either end joins it when
    both occur, unless one listing gives the two different values (pairs in `together`); each joined group goes by
    its fullest shared spelling, since shoppers took a need named in full far more often than one named in part."""
    together = together or set()
    counts = counts or {}
    words = {n: _key_words(n) for n in names}
    by_words = {w: n for n, w in reversed(list(words.items()))}
    group = {n: {n} for n in names}
    # longest first, so a three-word key claims its two-word stem before that stem can pair with a one-word key
    for name in sorted((n for n in words if len(words[n]) >= 2), key=lambda k: (-len(words[k]), k)):
        w = words[name]
        any(_g1_join(group, name, by_words.get(stem), together) for stem in (w[1:], w[:-1]))
    return {n: _g1_spelling(group[n], words, counts) for n in names}


def _g1_units(row: dict[Any, Any]) -> int | None:
    # units only under a quantity-like key; a key that may be money or an id leaves the units to the cart calls
    return next((v for k, v in row.items()
                 if _g1_plain_int(v) and 0 <= v <= 500 and _QTY_KEYS.search(str(k)) and not _MONEYISH.search(str(k))), None)


def _cart_lines(observation: Any) -> list[tuple[str, int]] | None:
    """(listing id, units) per cart line when a result lists its lines as records; None when it shows no such list."""
    todo = [(observation, 0)]
    while todo:
        item, level = todo.pop()
        if level > 5:
            continue
        if not _g1_record_list(item):
            todo.extend((child, level + 1) for child in _g1_children(item))
            continue
        lines = [(ident, _g1_units(row)) for row in item
                 for ident in [next((v for k, v in row.items() if _g1_idish(k, v)), None)] if ident]
        if lines:
            return lines
    if isinstance(observation, dict) and any(isinstance(v, list) and not v for v in observation.values()):
        return []       # the cart shows a list with no lines: it is empty
    return None


def _g1_page_step(key: Any, value: Any, ctx: dict[str, Any]) -> tuple[Any] | None:
    """What one field of a search result says about the next page: (arguments,) or (None,) when it settles it."""
    if _g1_token(value):
        if ctx["strings"] and re.search(r"follow|next|token|cursor|after|marker|more|page|onward|continu|resume", str(key), re.I):
            return ({ctx["strings"][0]: value},)
        return None
    if not (_g1_plain_int(value) and value > 0 and ctx["ints"]):
        return None
    named = re.search(r"follow|next|after|offset|start|more|page|token|cursor|marker", str(key), re.I)
    landed = ctx["arg"] and ctx["shown"] and value == ctx["offset"] + ctx["shown"] and not _COUNT_KEYS.search(str(key))
    if not (named or landed):
        return None
    slot = ctx["arg"] or ctx["ints"][0]
    cap = (ctx["props"].get(slot) or {}).get("maximum")
    over = isinstance(cap, (int, float)) and not isinstance(cap, bool) and value > cap
    return (None,) if over else ({slot: value},)


def _g1_token(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and len(value) <= 24 and " " not in value


def _next_page(observation: Any, schema: dict[str, Any], text_key: str, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
    # a following page is reached through a handle (token) or a position (offset) handed back by the result, sent
    # under whichever tool argument accepts it; an integer counts as the position when its key is named so, or when
    # it is exactly the requested position plus the rows on show. Nothing pointing onward gives None
    if not isinstance(observation, dict):
        return None
    given = args or {}
    props = schema.get("properties") or {}
    typed = [(k, v.get("type"), v) for k, v in props.items() if k != text_key and isinstance(v, dict)]
    strings = [k for k, t, v in typed if t == "string" and "enum" not in v]
    ints = [k for k, t, _ in typed if t == "integer"]
    fallback = ints[0] if len(ints) == 1 else None
    arg = next((k for k in ints if isinstance(given.get(k), int)), fallback)
    asked = given.get(arg, 0) if arg else 0
    ctx = {"props": props, "strings": strings, "ints": ints, "arg": arg,
           "offset": asked if _g1_plain_int(asked) else 0, "shown": len(_rows_with_ids(observation))}
    verdict = next(filter(None, (_g1_page_step(k, v, ctx) for k, v in observation.items())), None)
    if verdict is not None:
        return verdict[0]
    sent = [str(v) for v in given.values()]
    loose = [v for k, v in observation.items() if _g1_token(v) and not _ID_HINT_KEYS.search(str(k)) and v not in sent]
    # a single bare token beside the rows, under any key, is the next page's handle
    if len(loose) != 1 or not ctx["shown"] or not strings:
        return None
    return {strings[0]: loose[0]}


def _match_count(observation: Any) -> int | None:
    # the match total a search result reports (an integer under a total- or count-like key)
    if not isinstance(observation, dict):
        return None
    return next((v for k, v in observation.items()
                 if _g1_plain_int(v) and v >= 0 and _COUNT_KEYS.search(str(k)) and not _NOT_COUNT.search(str(k))), None)


def _question_property(text: str) -> str:
    # the property a shopper question asks about ("Which <property> would you like the <product> to have?")
    found = re.search(r"\b(?:which|what)\s+(.+?)\s+(?:would|do|does|should|will|is|are)\b", text, re.I)
    if found:
        core = found[1]
    else:
        core = re.sub(r"\b(which|what|would|you|like|the|to|have|do|want|need)\b", " ", text, flags=re.I)
    return " ".join(re.sub(r"\b(the|a|an|your|you)\b", " ", core.lower()).split())


def _question_product(text: str) -> str:
    # the product named in "... would you like the <product> to have?", else ""
    found = re.search(r"\bthe\s+(.+?)\s+to\s+(?:have|be|come)\b", text, re.I)
    return " ".join(found[1].lower().split()) if found else ""


def _resolver_text(text: str) -> str:
    # a page-chain resolver result as the conversation keeps it: its pages live on in the listing index and the
    # resolver's own report follows, so a long body is shortened (any note it carries is kept)
    head, _, note = text.partition("\n[note]")
    if len(head) <= 900:
        return text
    tail = "\n[note]" + note if note else ""
    return f"{head[:900]} ...(rest seen by the resolver; what it found comes next){tail}"


def _cond_test(cond: dict[str, Any]) -> str:
    # the deciding side of a conditional requirement: "condition new", or "condition other than new" when it holds
    # unless the other product carries those values
    negate = "other than " if cond.get("if_not") else ""
    listed = " or ".join(cond.get("if_values") or [])
    return f"{cond.get('if_property')} {negate}{listed}"


def _same_product(a: str, b: str) -> bool:
    # two product names given with questions mean one product (a blank name matches only a blank one)
    x, y = (s.strip().lower() for s in (a, b))
    if not (x and y):
        return x == y
    return x in y or y in x or x.rstrip("s") == y.rstrip("s")


def _text_key(properties: dict[str, Any], keys: list[str]) -> str | None:
    """Which string key takes free text: query-like names beat filter or page-handle names, earlier keys win ties
    ("needle" over "resume_bookmark", "q_string" over "group_label")."""
    return max(keys, key=_key_score, default=None)


def _key_score(key: str) -> int:
    return 2 * bool(_FREE_TEXT_KEYS.search(key)) - bool(_FILTER_KEYS.search(key))


def _mode_value(values: list[Any], text: bool) -> Any:
    """The search-mode value for a worded call (text=True) or a filter-only one: one named for it ("keyword",
    "department"), else one not named for the other way ("typed" next to "fields")."""
    names = [v for v in values if isinstance(v, str)]
    texty = [v for v in names if _TEXT_MODE.search(v)]
    filters = [v for v in names if v not in texty and _FILTER_MODE.search(v)]
    want, avoid = (texty, filters) if text else (filters, texty)
    return want[0] if want else next((v for v in names if v not in avoid), None)


def _clip(text: str, limit: Any) -> str:
    # cut at a word boundary: half a word matches nothing
    if not _g1_plain_int(limit) or len(text) <= limit:
        return text
    cut = text[:limit]
    split_word = text[limit:limit + 1] not in ("", " ")
    return (cut.rsplit(" ", 1)[0] if split_word and " " in cut.strip() else cut).rstrip()


def _kind_words(kind: str) -> set[str]:
    # singular words of a product kind, joining words dropped ("fish & seafood" equals "fish and seafood")
    return set(w.rstrip("s") for w in re.findall(r"[a-z0-9]+", kind.lower()) if w not in _G1_JOINERS)


def _prop_parts(prop: str) -> list[str]:
    # the properties a planned question names: one, or two joined by "or" or a comma
    parts = list(filter(None, map(str.strip, re.split(r"\s+or\s+|,", prop))))
    if parts or not prop:
        return parts
    return [prop]


def _same_property(a: str, b: str) -> bool:
    x, y = (s.strip().lower() for s in (a, b))
    if not (x and y):
        return False
    if x == y:
        return True
    return min(len(x), len(y)) > 3 and (x in y or y in x)


def _g1_word_agrees(x: str, y: str) -> bool:
    # a word of up to three letters may shorten a longer one ("max"); longer words must match ("pack" vs "packaging")
    short, full = x.rstrip("s"), y.rstrip("s")
    if len(full) < len(short):
        short, full = full, short
    return short == full or (len(short) <= 3 and full.startswith(short))


def _near_same(a: str, b: str) -> bool:
    # one property spelt two ways: one inside the other, or word for word up to abbreviation
    if _same_property(a, b):
        return True
    wa, wb = (re.findall(r"[a-z0-9]+", s.lower()) for s in (a, b))
    return len(wa) >= 2 and len(wa) == len(wb) and all(map(_g1_word_agrees, wa, wb))


def _rows_with_ids(observation: Any) -> list[tuple[str, str]]:
    # (listing id, row text) for every record in a search result that carries an id, at most 60
    out: list[tuple[str, str]] = []
    todo = [(observation, 0)]
    while todo and len(out) < 60:
        item, level = todo.pop()
        if level > 5:
            continue
        ident = next((v for k, v in item.items() if _g1_idish(k, v)), None) if isinstance(item, dict) else None
        if ident:
            out.append((ident, _compact(_brief(item, 200))))
        else:
            todo.extend((child, level + 1) for child in _g1_children(item))
    return out


def _has_text(value: Any, depth: int = 0) -> bool:
    # whether a result holds any word, beyond ids, numbers and flags
    todo = [(value, depth)]
    while todo:
        item, level = todo.pop()
        if level > 4:
            continue
        if not isinstance(item, str):
            todo.extend((child, level + 1) for child in _g1_children(item))
        elif any(map(str.isalpha, item)) and not (_ID_HINT.match(item) and _g1_has_digit(item)):
            return True
    return False


def _pending_step(observation: dict[str, Any]) -> bool:
    # an order result waiting on a further step (a token to confirm with) that has no shopper sentence in it
    said: list[str] = []
    _sentences([v for v in observation.values() if isinstance(v, (list, str))], said)
    nested = [k for v in observation.values() if isinstance(v, dict) for k in v]
    flagged = any(_PENDING_KEYS.search(str(k)) for k in [*observation, *nested])
    return flagged and all(len(s.split()) < 8 for s in said)


def _g1_line_ident(line: dict[Any, Any]) -> str | None:
    # one id per line: the first id under an id-like key (normally the listing), else the first id-shaped value
    named = next((v for k, v in line.items() if _g1_idish(k, v)), None)
    if named:
        return named
    return next((v for v in line.values()
                 if isinstance(v, str) and _ID_HINT.match(v) and _g1_has_digit(v) and not v.isdigit()), None)


def _listed_ids(args: dict[str, Any]) -> list[str] | None:
    # listing ids of a cart call that carries its lines in a list; None when it has no such list
    if any(_g1_idish(k, v) for k, v in args.items()):
        return None             # a call naming its listing outright: a list next to it holds choices, not lines
    found = [ident for group in args.values() if _g1_record_list(group)
             for ident in map(_g1_line_ident, group) if ident]
    return found or None


# pattern tables used by the helpers above and by the agent
_PROPERTY_KEYS = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
_QTY_KEYS = re.compile(r"qty|quantit|units?\b|units_|count|number|pieces|how.?many", re.I)
_MONEYISH = re.compile(r"price|cost|amount|sum|money|value|stock|discount|avail|left", re.I)
_COUNT_KEYS = re.compile(r"total|hit|count|match|found|result|num", re.I)
_ID_HINT_KEYS = re.compile(r"id$|^id|sort|order|status|state|mode|kind|type|query|currency|unit|lang|categor|shelf|dept", re.I)
_NOT_COUNT = re.compile(r"price|cost|page|offset|per|size|step|quant|stock|unit", re.I)
_TEXT_MODE = re.compile(r"text|keyword|word|phrase|query|search", re.I)
_FREE_TEXT_KEYS = re.compile(r"(^|[^a-z])q([^a-z]|$)|query|keyword|search|text|term|phrase|word|string|input|prompt|message|payload|content|body|look|find|seek|match|entry|descri|wanted|needle", re.I)
_FILTER_KEYS = re.compile(r"group|categor|label|maker|brand|seller|shop|vendor|department|dept|shelf|aisle|section|range|collection|famil|producer|manufactur|kind|class|tag|sort|order|cursor|token|page|bookmark|resume|marker|continu|next|after|offset|id$|code", re.I)
_FILTER_MODE = re.compile(r"field|filter|criteri|department|categor|shelf|browse|narrow|attribute|facet|struct|aisle|dept|exact|list", re.I)
_GONE = re.compile(r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone|"
                   r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered|"
                   r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up|"
                   r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one", re.I)
_PRICE_WORDS = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)
_LISTING_META = re.compile(r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
                           r"blurb|tagline|caption|teaser|slogan)\b|"
                           r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                           r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                           r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
_USED_UP = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                      r"reached the limit|maximum (number|reached)|already used", re.I)
_IDENT_KEYS = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
_OPTION_KEYS = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
_KIND_KEYS = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                        r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
_ASKS = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|abstain|withdraw|give.?up|stand.?down|walk.?away|no.?(buy|purchase|order|sale)|reject|cancel|quit|skip|^pass$|^(none|nothing|no)$", re.I)
_BUY_WORDS = re.compile(r"order|buy|purchase|checkout|check.?out|commit|confirm|place|accept|proceed|complete|pay", re.I)
_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")

def _compact(value: Any) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered if len(rendered) <= OBS_CHARS else rendered[:OBS_CHARS] + "...(cut)"


def _g2_unfence(text: str) -> str:
    body = text.strip()
    if not body.startswith("```"):
        return body
    body = re.sub(r"^```[A-Za-z]*", "", body).strip()
    return body[:-3].strip() if body.endswith("```") else body


def _g2_whole_candidates(text: str):
    """The whole reply first, then the span from the first brace to the last one."""
    yield text
    left, right = text.find("{"), text.rfind("}")
    if left >= 0 and right > left:
        yield text[left : right + 1]


def _g2_top_objects(text: str):
    """Every object that decodes at a brace standing outside any other brace pair, in reading order."""
    decoder = json.JSONDecoder()
    cursor = text.find("{")
    while 0 <= cursor < len(text):
        try:
            value, stop = decoder.raw_decode(text, cursor)
        except ValueError:
            cursor = text.find("{", cursor + 1)
            continue
        if isinstance(value, dict) and text.count("{", 0, cursor) == text.count("}", 0, cursor):
            yield value
        cursor = text.find("{", stop)


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = _g2_unfence(text)
    for candidate in _g2_whole_candidates(body):
        try:
            decoded = json.loads(candidate)
        except ValueError:
            continue
        return decoded if isinstance(decoded, dict) else None
    latest = None
    for latest in _g2_top_objects(body):
        pass
    return latest


def _g2_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_G2_TYPE_TESTS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": _g2_is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def _type_ok(value: Any, kind: str) -> bool:
    test = _G2_TYPE_TESTS.get(kind) if isinstance(kind, str) else None
    return True if test is None else test(value)


_G2_MISS = object()


def _g2_to_integer(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
        return int(float(value))
    return _G2_MISS


def _g2_to_number(value: Any) -> Any:
    if not (isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value)):
        return _G2_MISS
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _g2_to_string(value: Any) -> Any:
    return str(value) if _g2_is_number(value) else _G2_MISS


def _g2_to_boolean(value: Any) -> Any:
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return _G2_MISS


_G2_CONVERTERS = {"integer": _g2_to_integer, "number": _g2_to_number, "string": _g2_to_string, "boolean": _g2_to_boolean}


def _g2_kinds(schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    return [kinds] if isinstance(kinds, str) else kinds


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = _g2_kinds(schema)
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        whole_float = "integer" in kinds and isinstance(value, float) and value.is_integer()
        return int(value) if whole_float else value
    for kind in kinds:
        convert = _G2_CONVERTERS.get(kind) if isinstance(kind, str) else None
        if convert is None:
            continue
        try:
            result = convert(value)
        except (TypeError, ValueError):
            continue
        if result is not _G2_MISS:
            return result
    return value


def _g2_enum_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if "enum" not in schema or value in schema["enum"]:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        value = next((o for o in schema["enum"] if isinstance(o, str) and o.lower() == wanted), value)
    if value not in schema["enum"]:
        problems.append(f"{path}: must be one of {schema['enum']}")
    return value


def _g2_const_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    return value


def _g2_text_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(value, str):
        return value
    if "minLength" in schema and len(value) < schema["minLength"]:
        problems.append(f"{path}: shorter than {schema['minLength']} characters")
    return _clip(value, schema["maxLength"]) if "maxLength" in schema and len(value) > schema["maxLength"] else value


# (bound key, bound must be a number, value breaks the bound, message words)
_G2_BOUNDS = (
    ("minimum", False, lambda v, b: v < b, "below minimum"),
    ("maximum", False, lambda v, b: v > b, "above maximum"),
    ("exclusiveMinimum", True, lambda v, b: v <= b, "must be above"),
    ("exclusiveMaximum", True, lambda v, b: v >= b, "must be below"),
)


def _g2_number_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not _g2_is_number(value):
        return value
    for key, numeric_only, breaks, words in _G2_BOUNDS:
        if key not in schema or (numeric_only and not isinstance(schema[key], (int, float))):
            continue
        if breaks(value, schema[key]):
            problems.append(f"{path}: {words} {schema[key]}")
    return value


def _g2_array_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(value, list):
        return value
    if "minItems" in schema and len(value) < schema["minItems"]:
        problems.append(f"{path}: needs at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        problems.append(f"{path}: allows at most {schema['maxItems']} items")
    member = schema.get("items")
    if not isinstance(member, dict):
        return value
    checked = [_check(item, member, f"{path}[{n}]") for n, item in enumerate(value)]
    for _, sub in checked:
        problems.extend(sub)
    return [item for item, _ in checked]


def _g2_object_step(value: Any, schema: dict[str, Any], path: str, problems: list[str]) -> Any:
    if not isinstance(value, dict):
        return value
    allowed = schema.get("properties") or {}
    problems.extend(f"{path}.{name}: required" for name in schema.get("required") or [] if name not in value)
    if schema.get("additionalProperties") is False:
        problems.extend(f"{path}.{name}: not an allowed argument" for name in list(value) if name not in allowed)
    rebuilt = {}
    for name, item in value.items():
        if name in allowed:
            item, sub = _check(item, allowed[name], f"{path}.{name}")
            problems.extend(sub)
        rebuilt[name] = item
    return rebuilt


_G2_CHECK_STEPS = (_g2_enum_step, _g2_const_step, _g2_text_step, _g2_number_step, _g2_array_step, _g2_object_step)


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    value = _coerce(value, schema)
    kinds = _g2_kinds(schema)
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    problems: list[str] = []
    for step in _G2_CHECK_STEPS:
        value = step(value, schema, path, problems)
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    pending = [(value, depth)]
    while pending:
        node, level = pending.pop()
        if level > 4:
            continue
        if isinstance(node, list):
            pending.extend((child, level + 1) for child in reversed(node[:10]))
        elif isinstance(node, str) and len(node.split()) >= 4:
            out.append(node)


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    pending = [(value, depth)]
    while pending:
        node, level = pending.pop()
        if level > 6:
            continue
        if isinstance(node, list):
            pending.extend((child, level + 1) for child in node)
            continue
        if not isinstance(node, dict):
            continue
        for key, item in node.items():
            if isinstance(item, str) and "id" in key.lower() and _ID_HINT.match(item):
                out.add(item)
            else:
                pending.append((item, level + 1))



_SELLER_KEYS = re.compile(r"seller|dealer|merchant|vendor|shop|store|maker|brand", re.I)
_MAKER_KEYS = re.compile(r"maker|brand", re.I)
_MONEY_KEYS = re.compile(r"price|amount|cost", re.I)
# a search argument that caps the price of the rows it returns (a number), and the argument names that are no price cap
_CAP_ARG = re.compile(r"price|cost|cap|ceiling|budget|spend|max", re.I)
_NOT_CAP_ARG = re.compile(r"row|page|offset|start|count|size|per|qty|quant|unit|item|result|num|min", re.I)


def _g2_label(raw: Any) -> str:
    return re.sub(r"[_.]+", " ", str(raw)).strip().lower()


def _g2_property_values(block: dict[Any, Any]) -> dict[str, str]:
    """Short text or numeric entries of a nested property block, under normalised labels."""
    picked: dict[str, str] = {}
    for raw_key, entry in block.items():
        if isinstance(entry, str):
            if 0 < len(entry) <= 70:
                picked[_g2_label(raw_key)] = entry
        elif _g2_is_number(entry):
            picked[_g2_label(raw_key)] = str(entry)
    return picked


class Listing:
    """Everything the store has said about one listing, folded together across the results that mention it."""

    __slots__ = ("ident", "category", "seller", "shop_named", "price", "stock", "facts", "texts", "title", "opened", "seen_at", "group_key")

    def __init__(self, ident: str):
        self.ident = ident
        self.category = ""
        self.seller = ""
        self.title = ""
        self.group_key = ""
        self.shop_named = False     # True once the seller name came from a shop-type key rather than a maker/brand key
        self.price: float | None = None
        self.stock: bool | None = None
        self.opened = False
        self.seen_at = 0
        self.facts: dict[str, str] = {}
        self.texts: list[str] = []

    def _g2_set_category(self, name: str, text: str, plain: dict[str, str]) -> None:
        self.category = text.strip()

    def _g2_set_seller(self, name: str, text: str, plain: dict[str, str]) -> None:
        if not _MAKER_KEYS.search(name):
            self.seller, self.shop_named = text.strip(), True
        elif not self.shop_named:
            self.seller = text.strip()

    def _g2_add_prose(self, name: str, text: str, plain: dict[str, str]) -> None:
        if text not in self.texts:
            self.texts.append(text[:700])
        if not self.title and len(text) <= 160:
            self.title = text

    def _g2_add_plain(self, name: str, text: str, plain: dict[str, str]) -> None:
        plain[_g2_label(name)] = text

    def _g2_set_long_title(self, name: str, text: str, plain: dict[str, str]) -> None:
        self.title = text[:160]

    def _g2_text_rules(self):
        # first matching (test, action) wins, as in an if/elif ladder
        return (
            (lambda name, text: _KIND_KEYS.search(name) and len(text) <= 60, self._g2_set_category),
            (lambda name, text: _SELLER_KEYS.search(name) and len(text) <= 40, self._g2_set_seller),
            (lambda name, text: len(text.split()) >= 6, self._g2_add_prose),
            (lambda name, text: len(text) <= 70 and not (_IDENT_KEYS.search(name) and _ID_HINT.match(text) and any(c.isdigit() for c in text)), self._g2_add_plain),
            (lambda name, text: len(text) > 70 and not self.title, self._g2_set_long_title),
        )

    def _g2_take_scalar(self, name: str, value: Any, plain: dict[str, str], rules) -> None:
        if isinstance(value, bool):
            if "out" not in name.lower():
                self.stock = value
            return
        if _g2_is_number(value) and _MONEY_KEYS.search(name) and "total" not in name.lower():
            self.price = float(value)
            return
        if not isinstance(value, str) or not value.strip():
            return
        action = next((act for test, act in rules if test(name, value)), None)
        if action is not None:
            action(name, value, plain)

    def _g2_settle_facts(self, nested: dict[str, str], plain: dict[str, str]) -> None:
        if nested:
            self.facts.update(nested)
            self.opened = True
            return
        if len(plain) >= 5 and not self.facts:
            self.facts.update({k: v for k, v in plain.items() if not _LISTING_META.search(k) and not _KIND_KEYS.search(k)})

    def take(self, record: dict[str, Any], step: int, opened: bool) -> None:
        self.seen_at = step
        self.opened = self.opened or opened
        nested: dict[str, str] = {}
        plain: dict[str, str] = {}
        rules = self._g2_text_rules()
        for key, value in record.items():
            name = str(key)
            if isinstance(value, dict) and _PROPERTY_KEYS.search(name):
                nested.update(_g2_property_values(value))
            else:
                self._g2_take_scalar(name, value, plain, rules)
        self._g2_settle_facts(nested, plain)
        self.group_key = self.category.lower()

    def _g2_head_fields(self) -> list[str]:
        fields = (
            (self.category, lambda: "kind: " + self.category),
            (self.seller, lambda: "seller: " + self.seller),
            (self.title, lambda: "title: " + self.title[:160]),
            (self.price is not None, lambda: f"price: {self.price:g}"),
            (self.stock is not None, lambda: "in stock: " + str(self.stock).lower()),
            (self.facts, lambda: "facts: " + _compact(self.facts)),
        )
        return [render() for present, render in fields if present]

    def _g2_extra_texts(self) -> list[str]:
        lead = self.title[:60]
        spare = [t for t in self.texts if t != self.title and not (self.title and t.startswith(lead))]
        # the shortest one is most often the seller's own note (it may hold a page reference), so it keeps more room
        return ["text: " + t[: 250 if rank else 450] for rank, t in enumerate(sorted(spare, key=len)[:3])]

    def page(self, limit: int = 2500) -> str:
        pieces = [self.ident, *self._g2_head_fields(), *self._g2_extra_texts()]
        return " | ".join(pieces)[:limit]


def _g2_record_ident(record: dict[Any, Any]) -> str | None:
    for key, value in record.items():
        if isinstance(value, str) and _IDENT_KEYS.search(str(key)) and _ID_HINT.match(value) and any(c.isdigit() for c in value):
            return value
    return None


class Store:
    """Index of all listings the store has returned, updated from every tool result as it comes in."""

    def __init__(self):
        self.listings: dict[str, Listing] = {}
        self.order: list[str] = []

    def _g2_listing_for(self, ident: str) -> Listing:
        known = self.listings.get(ident)
        if known is not None:
            return known
        fresh = self.listings[ident] = Listing(ident)
        self.order.append(ident)
        return fresh

    def absorb(self, observation: Any, step: int, opened: bool = False) -> list[str]:
        found: list[str] = []
        pending = [(observation, 0)]
        while pending:
            node, level = pending.pop()
            if level > 6:
                continue
            if isinstance(node, list):
                children = node
            elif isinstance(node, dict):
                ident = _g2_record_ident(node)
                if ident:
                    self._g2_listing_for(ident).take(node, step, opened)
                    found.append(ident)
                # a listing's options carry ids and facts of their own and get shown alongside it
                children = [v for v in node.values() if isinstance(v, (dict, list))]
            else:
                continue
            pending.extend((child, level + 1) for child in reversed(children))
        return found

    def get(self, ident: str) -> Listing | None:
        return self.listings.get(ident)

    def pages(self, idents: list[str], limit: int = 2500) -> list[str]:
        known = self.listings
        return [known[i].page(limit) for i in filter(known.__contains__, idents)]

    def opened(self) -> list[Listing]:
        ordered = (self.listings[i] for i in self.order)
        return [listing for listing in ordered if listing.facts]

    def groups(self) -> dict[str, list[Listing]]:
        grouped: dict[str, list[Listing]] = {}
        for listing in self.opened():
            grouped.setdefault(listing.group_key, []).append(listing)
        return grouped

    def by_seller(self, shop: str) -> list[Listing]:
        needle = shop.lower().strip()
        if not needle:
            return []

        def mentions(listing: Listing) -> bool:
            return needle in listing.seller.lower() or any(needle in t.lower() for t in listing.texts)

        return [listing for listing in (self.listings[i] for i in self.order) if mentions(listing)]

    def cheaper(self, ident: str, budget: float | None, quantity: int, opened: set[str], skip: list[str]) -> list[tuple[str, float]]:
        held = self.listings.get(ident)
        if held is None or held.price is None:
            return []

        def eligible(other: Listing) -> bool:
            if other.ident == ident or other.ident in opened or other.ident in skip:
                return False
            if other.price is None or other.price >= held.price or other.stock is False:
                return False
            if budget is not None and other.price * quantity > budget:
                return False
            return not (held.category and other.category and other.category.lower() != held.category.lower())

        ranked = sorted((other.price, other.ident) for other in self.listings.values() if eligible(other))
        return [(name, cost) for cost, name in ranked[:5]]

    def label_counts(self) -> str:
        tally: dict[str, int] = {}
        for listing in self.listings.values():
            if listing.category:
                tally[listing.category] = tally.get(listing.category, 0) + 1
        if sum(tally.values()) < 4 or not 2 <= len(tally) <= 12:
            return ""
        top = sorted(tally.items(), key=lambda kv: -kv[1])[:8]
        return "- category: " + " | ".join(f"{label} (x{n})" for label, n in top)

class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        policy = self.policy = env["policy_view"]
        self.tools = list(filter(self._g3_usable_tool, policy.get("tools") or []))
        self.schemas = dict(map(self._g3_schema_pair, self.tools))
        self.max_steps = int(policy.get("max_steps") or 30)
        # every other piece of episode state starts from this table (fresh containers per episode)
        for name, value in self._g3_initial_state(policy).items():
            setattr(self, name, value)

    @staticmethod
    def _g3_usable_tool(tool: Any) -> bool:
        return isinstance(tool, dict) and isinstance(tool.get("function"), dict)

    @staticmethod
    def _g3_schema_pair(tool: dict[str, Any]) -> tuple[str, Any]:
        spec = tool["function"]
        return spec["name"], spec.get("parameters") or {"type": "object", "properties": {}}

    @staticmethod
    def _g3_initial_state(policy: dict[str, Any]) -> dict[str, Any]:
        # notes on the less obvious fields:
        #   refused_at / searched_at: step of the latest refused order / latest search
        #   searches_in_a_row: searches since some listing page was opened
        #   view_called_at, view_tries: last cart-view call, cart states viewed by code before ordering
        #   process_checks / tool_calls / process_blocks: pre-order tool rules, (step, tool) of clean calls, held-back pairs
        #   view_facts: the view tool once returned a property list; key_votes: model-only search strings that hit rows
        #   refused_ids: cart contents at a refused order; pointers / resolved: page references pending / found
        #   cart_qty: units last requested per listing; compare_cap: ids per side-by-side call (shrinks on empty answers)
        #   commit_next: late decline -> closest fit ordered next turn; named_checked: near-miss-free decline bounced once
        #   capped_sweeps: id|price lines the price-capped search covered; forcing: last-chance order bypasses gates
        #   pointers_stalled: chain follower out of time/tools/budget; finish_*: setup's finishing rules and their bookkeeping
        #   events: step log for the finish check; conditions / cond_state / dep_*: cross-product rules and their checks
        #   changed_at: step of the latest heard change; see_read: follow-up parts already looked up
        #   single_only: one property per message; lead_words / lead_low / lead_same: how many query words the search reads
        #   price_votes / order_votes: first-page ordering evidence; page_rows / page_depth: page size and paging limit
        #   sweep_stamp: requirements+budget at the last cheapest sweep; finish_tries: finishing calls sent
        #   audit_clean: last decline audit named clean fits; asked_props: (product, property) per question
        #   paged: listings with a full page opened; word_rule: "all" / "any" from word-count probes
        #   blocked_turns: consecutive model turns with no call run; page_next: cart lines to open before the next turn
        return {
            "step": 0, "finished": bool(policy.get("finished")), "dead": False, "messages_used": 0, "message_budget": None,
            "order_attempts": 0, "refusals": 0, "order_checked": False, "decline_checks": 0, "opened": set(),
            "cheaper_checks": 0, "early_bounced": False, "refused_at": -1, "searched_at": -1, "searches_in_a_row": 0,
            "research_checks": 0, "cart_used": False, "seen_ids": set(), "stated": [], "revealed": [], "free": [],
            "dropped": [], "budget_now": None, "heard": set(), "cart_ids": [], "records": [], "last_cart_view": None,
            "cart_changed_at": 0, "cart_view_at": -1, "view_called_at": -1, "view_tries": set(), "process_checks": [],
            "tool_calls": [], "process_blocks": set(), "cart_state": {}, "add_signature": None, "asked": [],
            "questions_sent": [], "planner_on": True, "sets_at_last_ask": -1, "last_unclear": False, "plan_misses": 0,
            "capped": set(), "view_facts": False, "facts_gathered": False, "cart_lines": {}, "plan_retry_at": 0,
            "audited": False, "decline_ok": False, "commits": 0, "key_votes": {}, "commit_at": -1, "refused_ids": set(),
            "qty_warned": set(), "pending_reply": "", "current": None, "verdicts": {}, "ledger_version": 0,
            "cart_checked": set(), "said": [], "recent": [], "roles": {}, "plan": {}, "cost": 0.0, "trace": [],
            "messages": [], "pointers": [], "resolved": [], "chain_pages": [], "cart_seen": None, "cart_seen_at": -1,
            "cart_qty": {}, "compare_cap": PAGE_OPENS, "compare_left": [], "commit_next": False, "named_checked": False,
            "capped_sweeps": set(), "forcing": False, "pointers_stalled": False, "finish_steps": [], "finish_reserved": 0,
            "finish_messages": 0, "finish_next": None, "finish_clear": set(), "finish_rounds": {}, "events": [],
            "line_blocks": set(), "hints": [], "fit_rules": [], "conditions": [], "cond_state": {}, "dep_checks": 0,
            "dep_next": [], "value_swept": False, "changed_at": -100, "see_read": set(), "reask": False,
            "single_only": False, "lead_words": 0, "lead_low": 0, "lead_same": set(), "price_votes": [0, 0],
            "page_rows": 0, "page_depth": 0, "probed": False, "probe_args": None, "store": Store(), "searches_done": {},
            "search_log": {}, "page_checks": 0, "sweep_checks": 0, "sweep_stamp": "", "finish_tries": 0,
            "audit_clean": False, "asked_props": [], "order_votes": [0, 0], "paged": set(), "word_rule": "",
            "blocked_turns": 0, "page_next": [],
        }

    def time_left(self) -> float:
        elapsed = time.monotonic() - self.started
        return TASK_SECONDS - elapsed

    def steps_left(self) -> int:
        used = self.step
        return self.max_steps - used

    def order_due(self) -> int:
        # a quarter of the step allowance kept for the first order and one repair, clamped to 6..16
        # (very late first orders rarely pass; very early ones leave too few backups to fall back on)
        quarter = self.max_steps // 4
        if quarter < 6:
            return 6
        return 16 if quarter > 16 else quarter

    def _g3_late_window(self) -> bool:
        # within the reserved closing steps and no change heard in the last three steps
        if self.steps_left() > self.order_due():
            return False
        return self.step - self.changed_at > 3

    def in_reserve(self) -> bool:
        # reserved closing steps with a pick already in the cart and no order attempted yet (else: the empty-cart variant)
        if self.order_attempts == 0 and self.cart_ids and self._g3_late_window():
            return True
        return self.empty_reserve()

    def empty_reserve(self) -> bool:
        # reserved closing steps, empty cart, no finish attempted, but some listing pages opened: decide now
        if self.finish_tries != 0 or self.cart_ids or not self._g3_late_window():
            return False
        return bool(self.store.opened())

    @staticmethod
    def _g3_request_body(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int,
                         reasoning: dict[str, Any] | None) -> dict[str, Any]:
        sent = messages
        if len(messages) > 2 and model.startswith("anthropic/"):
            sent = list(map(dict, messages))
            # mark the first and last plain-text turns as cacheable
            for slot in {0: None, len(sent) - 1: None}:
                text = sent[slot].get("content")
                if text and isinstance(text, str):
                    sent[slot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
        extras: dict[str, Any] = {}
        if reasoning is not None:
            extras["reasoning"] = reasoning
        if tools:
            extras.update(tools=tools, tool_choice="auto")
        body.update(extras)
        return body

    def _g3_attempt(self, body: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        result = _llm.post_verbose("/inference/chat/completions", body)
        data = result.data
        if not (isinstance(data, dict) and data.get("choices")):
            return None, result
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
        reply = data["choices"][0].get("message") or {}
        usable = reply.get("content") or reply.get("tool_calls")
        return (reply if usable else None), result

    def _g3_backoff_due(self, model: str, result: Any, waits: int) -> bool:
        raw = result.error
        error = raw if isinstance(raw, dict) else {}
        said = str(error.get("body") or raw or "")
        self.trace.append({"kind": "inference_failure", "model": model, "error": said[:300] if said else "empty"})
        status = error.get("status")
        if status == 400 and re.search(r"reasoning is mandatory|cannot be disabled", said, re.I):
            _REASONING_ON.add(model)
            return False
        # in-flight budget, rate limits and gateway hiccups clear up after a short pause: same model again
        transient = status in (402, 429, 500, 502, 503, 504) or _RETRY_WORDS.search(said)
        return bool(transient) and waits < 3 and self.time_left() > WRAP_UP_SECONDS

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        lineup = list(models)
        if reasoning is not None and reasoning.get("enabled") is False:
            lineup = [m for m in models if m not in _REASONING_ON] or lineup
        waits = 0
        for model in lineup:
            while True:
                left = self.time_left()
                if left < FORCE_SECONDS - 10:
                    return None
                _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
                reply, result = self._g3_attempt(self._g3_request_body(model, messages, tools, max_tokens, reasoning))
                if reply is not None:
                    return reply
                if not self._g3_backoff_due(model, result, waits):
                    break
                waits += 1
                time.sleep(1.5 * waits)
        return None

    def _g3_env_payload(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        key, value = ("action", actions[0]) if len(actions) == 1 else ("actions", actions)
        return {**self.binding, "step": self.step + 1, key: value}

    def _g3_env_read(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        reported = data.get("step")
        self.step = reported if isinstance(reported, int) else self.step + count
        if data.get("finished"):
            self.finished = True
        return data

    def _g3_env_dead(self, body: str, payload: dict[str, Any]) -> bool:
        self.dead = True
        return False

    def _g3_env_conflict(self, body: str, payload: dict[str, Any]) -> bool:
        if "finished" in body:
            self.finished = True
            return False
        expected = re.search(r"expected (\d+)", body)
        if not expected:
            return False
        self.step = int(expected.group(1)) - 1
        payload["step"] = self.step + 1
        return True

    def _g3_env_keep_going(self, result: Any, payload: dict[str, Any]) -> bool:
        error = result.error or {}
        status = error.get("status")
        body = str(error.get("body") or "")
        self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
        handlers = ((402, self._g3_env_dead), (409, self._g3_env_conflict))
        handler = next((fn for code, fn in handlers if status == code), None)
        if handler is not None:
            return handler(body, payload)
        return status not in (403, 422, 500, 503, 504)

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload = self._g3_env_payload(actions)
        attempts = 3
        while attempts:
            attempts -= 1
            left = self.time_left()
            if left < 2:
                break
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                return self._g3_env_read(result.data, len(actions))
            if not self._g3_env_keep_going(result, payload):
                break
        return None

    def setup(self) -> None:
        plan = self._g4_read_request()
        context = {"query": str(self.policy.get("query") or ""), "names": [], "kept": []}
        for key, normalise in self._g4_setup_steps():
            normalise(plan, plan.get(key) if key else None, context)

    @staticmethod
    def _g4_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _g4_setup_steps(self) -> tuple:
        # each plan field in turn, in the order the later fields depend on the earlier ones
        return (
            ("closest_finish", self._g4_fix_closest),
            ("roles", self._g4_fix_roles),
            ("message_budget", self._g4_fix_message_budget),
            ("items", self._g4_merge_pair),
            ("requirements", self._g4_pin_requirements),
            ("product", self._g4_drop_category),
            ("ask_first", self._g4_split_asks),
            ("hint_topics", self._g4_add_hints),
            ("references", self._g4_add_pointers),
            ("fit_rules", self._g4_fit_rules),
            ("conditions", self._g4_conditions),
            ("ask_first", self._g4_price_question),
            (None, self._g4_cart_units),
            (None, self._g4_compare_cap),
            ("step_budget", self._g4_step_budget),
            ("finish_steps", self._g4_finish_steps),
            ("process_checks", self._g4_process_checks),
        )

    def _g4_read_request(self) -> dict:
        catalogue = "\n".join("- {}: {} | arguments: {}".format(fn["name"], fn.get("description", ""), _compact(fn.get("parameters") or {}))
                              for fn in (tool["function"] for tool in self.tools))
        head = "\n\nTOOLS:\n".join((SETUP_PROMPT, catalogue))
        pieces = ["REQUEST:\n" + str(self.policy.get("query") or "")]
        opening = self.policy.get("opening")
        if opening is not None:
            pieces.append("OPENING OBSERVATION:\n" + _compact(opening))
        content = [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": "\n\n".join(pieces)}]
        plan: dict = {}
        # a second try (no thinking, another fallback) only when the first reading brought no roles and time allows
        for attempt, (fallback, tokens, thinking) in enumerate(((BACKUP_MODEL, 3500, LIGHT_THINKING), (MAIN_MODEL, 4000, NO_THINKING))):
            if attempt and (isinstance(plan.get("roles"), dict) or self.time_left() <= 200):
                break
            reply = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, fallback], tokens, thinking)
            plan = _parse_json_object((reply or {}).get("content") or "") or plan
        return plan

    def _g4_fix_closest(self, plan: dict, value: Any, context: dict) -> None:
        # "name the closest" finish: buy nothing when nothing fits, and name the near misses in the no-purchase finish
        if str(value or "").lower() == "name":
            plan.update(closest_if_none=False, closest_named=True, decline_no_names=False)

    def _g4_fix_roles(self, plan: dict, value: Any, context: dict) -> None:
        self.plan = plan
        read = value if isinstance(value, dict) else {}
        self.roles = dict(pair for pair in read.items() if isinstance(pair[1], str) and pair[1] in self.schemas)
        if self.roles:
            return
        # no usable reading: fall back on the tool names so the role-based guards stay on
        self.roles = self.guess_roles()
        self.trace.append({"kind": "guessed_roles", "roles": self.roles})

    def _g4_fix_message_budget(self, plan: dict, value: Any, context: dict) -> None:
        if self._g4_number(value) and value >= 0:
            self.message_budget = int(value)

    def _g4_merge_pair(self, plan: dict, value: Any, context: dict) -> None:
        query = context["query"]
        listed = [entry for entry in value if isinstance(entry, dict)] if isinstance(value, list) else []
        names = [str(entry.get("product") or "").strip().lower() for entry in listed]
        units = [entry.get("quantity") for entry in listed]
        context["names"] = list(filter(None, names))
        if len(names) != 2 or not all(names) or len({str(u) for u in units}) != 1:
            return
        first, second = names
        if not re.search(re.escape(first) + r"\s+(and|&)\s+" + re.escape(second), query, re.I):
            return
        if re.search(r"\d+\s*x\s*" + re.escape(second), query, re.I):
            return
        # "<noun> and <noun>" (pots and pans) names one product
        lead = units[0]
        if isinstance(lead, int) and not isinstance(lead, bool) and lead >= 1:
            plan["quantity"] = lead
        plan["items"] = []
        plan["product"] = f"{first} and {second}"
        prefix = re.compile(r"^(" + re.escape(first) + "|" + re.escape(second) + r")\s+", re.I)
        plan["requirements"] = [prefix.sub("", str(r)) for r in plan.get("requirements") or []]
        self.trace.append({"kind": "merged_items", "names": names})

    def _g4_pinning_ref(self, plan: dict, prop: str, value: str, query: str, names: list) -> Any:
        lowered = query.lower()
        own = next((n for n in names if prop.startswith(n)), "")

        def pins(ref: Any) -> bool:
            if not isinstance(ref, dict) or not ref.get("shop") or not value or not prop.split():
                return False
            theirs = str(ref.get("product") or "").strip().lower()
            if own and theirs and own != theirs:
                return False        # it belongs to another requested product
            where = lowered.find(str(ref.get("shop")).lower())
            if where < 0:
                return False
            sentence = re.split(r"[.;!?](\s|$)", query[where:where + 300].lower())[0]
            return value in sentence and prop.split()[-1] in sentence

        return next(filter(pins, plan.get("references") or []), None)

    def _g4_pin_requirements(self, plan: dict, value: Any, context: dict) -> None:
        kept = []
        for requirement in [str(r) for r in value or [] if r]:
            prop, _, wanted = requirement.partition(":")
            prop, wanted = prop.strip().lower(), wanted.strip().lower()
            ref = self._g4_pinning_ref(plan, prop, wanted, context["query"], context["names"])
            if ref is None:
                kept.append(requirement)
                continue
            # the requirement only identifies the referenced page, so it becomes one of that page's facts
            facts = ref.get("facts") if isinstance(ref.get("facts"), dict) else {}
            owner = ref.get("product")
            facts.setdefault(prop.split(" ", 1)[-1] if owner and prop.startswith(str(owner).lower()) else prop, wanted)
            ref["facts"] = facts
            self.trace.append({"kind": "pinning_fact", "requirement": requirement})
        context["kept"] = kept

    @staticmethod
    def _g4_restates(requirement: str, vocabulary: set, query: str) -> bool:
        prop, _, value = requirement.partition(":")
        tokens = re.findall(r"[a-z0-9]+", value.lower())
        if len(tokens) != 1 or tokens[0] not in vocabulary or re.search(r"\d", value):
            return False
        return not any(re.search(r"\b" + re.escape(w), query, re.I) for w in re.findall(r"[a-z0-9]+", prop.lower()) if len(w) > 2)

    def _g4_drop_category(self, plan: dict, value: Any, context: dict) -> None:
        kept = context["kept"]
        kind = str(value or "").lower()
        if kind and not plan.get("items") and not re.search(r"&|\band\b|\bor\b", kind):
            vocabulary = set(re.findall(r"[a-z0-9]+", kind))
            marks = [self._g4_restates(r, vocabulary, context["query"]) for r in kept]
            if any(marks):
                self.trace.append({"kind": "category_words", "dropped": [r for r, m in zip(kept, marks) if m]})
                kept = [r for r, m in zip(kept, marks) if not m]
        plan["requirements"] = kept
        self.stated = [str(r) for r in kept if r]
        self.budget_now = plan.get("price_cap")

    def _g4_split_asks(self, plan: dict, value: Any, context: dict) -> None:
        # only price/preference questions stay open ones; a story hint becomes a property question
        texts = [str(entry) for entry in value or []]
        open_kind = re.compile(r"budget|price|spend|cost|afford|rather|matters more|prefer|which of", re.I)
        self.hints.extend(t for t in texts if not open_kind.search(t))
        plan["ask_first"] = [t for t in texts if open_kind.search(t)]

    def _g4_add_hints(self, plan: dict, value: Any, context: dict) -> None:
        self.hints.extend(str(h) for h in value or [] if isinstance(h, str) and h.strip())

    def _g4_add_pointers(self, plan: dict, value: Any, context: dict) -> None:
        usable = (r for r in value or [] if isinstance(r, dict) and r.get("property") and (r.get("shop") or r.get("kind")))
        for ref in usable:
            ref["source"] = "stated"
            self.pointers.append(ref)

    def _g4_fit_rules(self, plan: dict, value: Any, context: dict) -> None:
        self.fit_rules = list(filter(lambda r: isinstance(r, dict) and r.get("accessory") and r.get("anchor"), value or []))

    def _g4_conditions(self, plan: dict, value: Any, context: dict) -> None:
        for cond in value or []:
            if not isinstance(cond, dict) or not all(cond.get(k) for k in ("if_product", "then_product", "then_requirement")):
                continue
            raw = cond.get("if_values")
            cond["if_values"] = list(map(str, raw)) if isinstance(raw, list) else [str(raw or "")]
            cond["if_not"] = cond.get("if_not") is True
            self.conditions.append(cond)
            # the same requirement read as a plain one would bind whatever the other product turns out to be
            needle = str(cond["then_requirement"]).lower().partition(":")[2].strip()
            subject = str(cond["then_product"])
            self.stated = [r for r in self.stated
                           if not (needle and needle in r.lower() and _same_product(r.lower()[:len(subject)], subject.lower()))]
            plan["requirements"] = list(self.stated)

    def _g4_price_question(self, plan: dict, value: Any, context: dict) -> None:
        if not plan or self.budget_now is not None or not self.roles.get("message"):
            return
        plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + [str(a) for a in value or [] if a]

    def _g4_cart_units(self, plan: dict, value: Any, context: dict) -> None:
        # an integer argument of the cart tool counts units, whatever it is called (race 9: "lineNumber")
        cart = self.schemas.get(self.roles.get("cart") or "") or {}
        bare = [spec for spec in (cart.get("properties") or {}).values()
                if isinstance(spec, dict) and spec.get("type") == "integer" and not spec.get("description")]
        for spec in bare:
            removal = " (0 takes it out)" if spec.get("minimum") == 0 else ""
            spec["description"] = ("A number of units of this listing, never a line's position" + removal
                                   + ". The reply shows each line's units after the call: some stores add the units sent to what the line holds, so check them against the quantity asked.")

    def _g4_compare_cap(self, plan: dict, value: Any, context: dict) -> None:
        side = self.compare_tool()
        if not side:
            return
        # the declared limit is not trusted (stores answered fewer than declared and refused more with an error)
        spec = ((self.schemas.get(side[0]) or {}).get("properties") or {}).get(side[1]) or {}
        declared = spec.get("maxItems")
        if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 1:
            self.compare_cap = min(self.compare_cap, max(min(declared, 2), declared // 2))

    def _g4_step_budget(self, plan: dict, value: Any, context: dict) -> None:
        if self._g4_number(value) and 0 < value < self.max_steps:
            self.max_steps = int(value)

    def _g4_finish_steps(self, plan: dict, value: Any, context: dict) -> None:
        settled_by_order = plan.get("closest_if_none") or plan.get("closest_named")
        fallback_rule = re.compile(r"\b(fits?|closest|nearest|falls? short|substitute)\b", re.I)
        for step in value or []:
            if not isinstance(step, dict):
                continue
            if settled_by_order and fallback_rule.search(str(step.get("rule"))):
                continue        # the order itself meets the "nothing fits" rule
            if not str(step.get("action") or step.get("rule") or "").strip():
                continue
            self.finish_steps.append(step)
            count = step.get("messages")
            if self._g4_number(count) and count > 0:
                self.finish_reserved += int(count)

    def _g4_process_checks(self, plan: dict, value: Any, context: dict) -> None:
        # rules a count of tool calls settles: checked in code when the order is called
        finishing = {self.roles.get(r) for r in ("order", "decline", "message")} - {None}
        for check in value or []:
            tool = str(check.get("tool") or "") if isinstance(check, dict) else ""
            if tool in finishing or tool not in self.schemas:
                continue
            least = check.get("minimum")
            floor = int(least) if self._g4_number(least) else 1
            self.process_checks.append({"rule": str(check.get("rule") or "")[:300], "tool": tool, "minimum": max(1, min(floor, 6)),
                                        "after_cart_final": check.get("after_cart_final") is True})

    def guess_roles(self) -> dict[str, str]:
        cart_words = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}

        def tokens(name: str) -> set[str]:
            parts = re.split(r"[^A-Za-z0-9]+", re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name))
            return set(map(str.lower, filter(None, parts)))

        def single_text(name: str) -> bool:
            fields = ((self.schemas.get(name) or {}).get("properties") or {}).values()
            return sum(1 for f in fields if isinstance(f, dict) and f.get("type") == "string" and "enum" not in f) == 1

        def declines(w: set, n: str) -> bool:
            if w & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"}:
                return True
            return any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w

        def views_cart(w: set, n: str) -> bool:
            return bool(w & (cart_words | {"items", "order"})) and bool(w & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})

        def fills_cart(w: set, n: str) -> bool:
            if w & cart_words:
                return True
            return bool(w & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(w & (cart_words | {"items", "order", "all"}))

        def messages(w: set, n: str) -> bool:
            talk = {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk", "say", "tell", "reply", "user"}
            return bool(w & talk) and single_text(n)

        keyword_rules = {
            "order": {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"},
            "details": {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec", "specs", "describe", "examine"},
            "search": {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"},
        }
        ordered = [("decline", declines), ("cart_view", views_cart), ("cart", fills_cart),
                   ("order", lambda w, n: bool(w & keyword_rules["order"])), ("message", messages),
                   ("details", lambda w, n: bool(w & keyword_rules["details"])), ("search", lambda w, n: bool(w & keyword_rules["search"]))]
        names = [tool["function"]["name"] for tool in self.tools]
        guessed: dict[str, str] = {}
        for role, matches in ordered:
            free = [n for n in names if n not in guessed.values()]
            hit = next((n for n in free if matches(tokens(n), n)), None)
            if hit is not None:
                guessed[role] = hit
        return guessed

    @staticmethod
    def _g4_item_pair(entry: Any) -> Any:
        if isinstance(entry, str) and entry.strip():
            entry = {"product": entry, "quantity": 1}
        if not isinstance(entry, dict):
            return None
        name = str(entry.get("product") or "").strip()
        if not name:
            return None
        count = entry.get("quantity")
        if isinstance(count, str) and count.strip().isdigit():
            count = int(count.strip())
        whole = isinstance(count, (int, float)) and not isinstance(count, bool) and count >= 1
        return name, (int(count) if whole else 1)

    def items(self) -> list[tuple[str, int]]:
        # the separate products of a several-product request; empty when one product is asked for
        raw = self.plan.get("items")
        pairs = [pair for pair in map(self._g4_item_pair, raw if isinstance(raw, list) else []) if pair]
        return pairs if len(pairs) > 1 else []

    def _g4_brief_goal(self, plan: dict) -> list[str]:
        lines = [f"- product: {plan.get('product')}"] if plan.get("product") else []
        wanted = self.items()
        lines += ([f"- buy {name} x {quantity} (its own cart line)" for name, quantity in wanted]
                  or [f"- quantity: {plan.get('quantity') or 1}"])
        lines += [f"- requirement as stated: {r}" for r in plan.get("requirements") or []]
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap for the whole order: {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- in-stock / on-hand listings only")
        priority = plan.get("priority")
        if priority in ("cheapest", "best"):
            note = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- among the listings that fit, take the {priority}{note}")
        else:
            lines.append("- the request sets no cheapest/best rule: every listing that meets all requirements inside the budget is a right pick, so spend no steps chasing a lower price (unless the request text says to)")
        return lines

    def _g4_brief_asks(self, plan: dict) -> list[str]:
        return ([f"- optional, only breaks ties (NOT required): {w}" for w in plan.get("soft_wants") or []]
                + [f"- ask the shopper before choosing: {a}" for a in plan.get("ask_first") or []]
                + [f"- rule: {r}" for r in plan.get("rules") or []])

    def _g4_brief_process(self, plan: dict) -> list[str]:
        lines = [] if self.message_budget is None else [f"- shopper messages permitted: {self.message_budget}"]
        lines += [f"- closing rule (the order call triggers it and it is handled automatically): {s.get('action') or s.get('rule')} ({s.get('timing') or 'at the finish'})"
                  for s in self.finish_steps]
        for check in self.process_checks:
            when = " after the last cart change)" if check["after_cart_final"] else ")"
            lines.append(f"- process rule (verified when you call the order tool): {check['rule']} ({check['tool']} at least {check['minimum']} time(s){when}")
        if self.finish_reserved:
            lines.append(f"- messages held back for the finishing rules: {self.finish_reserved}; questions may use the remainder")
        lines.append(f"- step limit: {self.max_steps}")
        return lines

    def _g4_brief_fallbacks(self, plan: dict) -> list[str]:
        lines = []
        if plan.get("closest_if_none"):
            lines.append("- if no listing meets every requirement: order the one missing the fewest requirements (sold out and over budget count as one miss each; cheapest among ties); do not decline")
        if plan.get("closest_named"):
            lines.append("- if no listing meets every requirement (decide only after a thorough search): buy nothing and end with the no-purchase option; its message says no listing fits cleanly and gives the id of the nearest miss - fewest unmet points, with sold out or over budget counting as one each, cheapest among equally near ones (one per product when several are asked for) - together with what each one lacks, as the request's rule for that case asks")
        if self.hints:
            lines.append("- hidden needs the story hints at: " + "; ".join(self.hints) + " (questions about the matching listing attributes come first)")
        return lines

    def _g4_brief_links(self, plan: dict) -> list[str]:
        lines = [f"- requirement given as a pointer to another page: {ref.get('product') or ''} {ref.get('property')}: {ref.get('relation')} as the page of the {ref.get('shop')} {ref.get('kind')} ({_compact(ref.get('facts') or {})}); that chain of pages gets resolved automatically and the resulting value is added as a requirement"
                 for ref in self.pointers]
        lines += [f"- fit rule: the {r['accessory']} must be compatible with the chosen {r['anchor']}: settle the {r['anchor']} before it; compatibility shows as a matching model number, series or part code on the {r['accessory']} listing; the order call confirms it"
                  for r in self.fit_rules]
        lines += [f"- conditional requirement: the {c['then_product']} needs {c['then_requirement']} only when the {c['if_product']} you buy has {_cond_test(c)}, and not otherwise (pick the {c['if_product']} first and read its page; verified for you at the order)"
                  for c in self.conditions]
        return lines

    def brief(self) -> str:
        if not self.plan:
            return ""
        sections = (self._g4_brief_goal, self._g4_brief_asks, self._g4_brief_process, self._g4_brief_fallbacks, self._g4_brief_links)
        out = ["Parsed summary of what the shopper wants (double-check with the original wording):"]
        for build in sections:
            out += build(self.plan)
        return "\n".join(out)

    def _g4_status_counts(self, left: int) -> str:
        counts = [f"[status] {self.step} of {self.max_steps} steps used, {left} remaining"]
        if self.message_budget is None:
            counts.append(f"{self.messages_used} messages sent")
        else:
            held = self.kept_messages()
            counts.append(f"{self.messages_used} of {self.message_budget} messages sent" + (f" ({held} held for the finishing rules)" if held else ""))
        counts.append(f"roughly {int(max(0, self.time_left() - FORCE_SECONDS))} s remaining")
        return "; ".join(counts) + "."

    def _g4_status_warnings(self, left: int) -> str:
        notes = ""
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            # a refused order is easily taken for a placed one; nothing is bought until the episode ends
            notes += (f" NOTHING HAS BEEN BOUGHT YET: the order call {self.step - self.refused_at} steps back was rejected, so the episode is still running."
                      " Retry the order tool at once if the cart already holds suitable lines under the present budget; if it does not, repair the cart and then retry"
                      " (a decline is only for a store with no suitable listing left).")
        if self.cost > COST_HARD and left > 2:
            notes += " SPENDING LIMIT REACHED: no further lookups; load the strongest confirmed match into the cart (a line for each requested product) and place the order immediately, or decline if no match exists."
        return notes

    def _g4_status_pace(self, left: int) -> str:
        due = self.order_due
        if left <= 2:
            return " LAST STEPS: place the order immediately if a confirmed match sits in the cart; failing that, cart the nearest match you have opened and order it (a decline is almost always worth nothing)."
        if left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            return " FINISH NOW: trim the cart to the confirmed line(s) required, confirm availability once more and place the order; lacking a confirmed listing, buy the nearest opened candidate rather than declining."
        if self.finish_steps and self.order_attempts == 0 and left <= 9 and self.finish_pending():
            return " Budget of steps is getting tight and the closing rules of the request consume one or two of them right before ordering: settle the cart soon and invoke the order tool with no fewer than 3 steps remaining (the closing rules run automatically at that point)."
        if self.empty_reserve():
            return f" DECIDE NOW: searching is closed, because the remaining {due()} steps are reserved for the first finishing call and its aftermath (it often brings an updated request). Cart the strongest confirmed match (a line for each requested product) and invoke the order tool; finish with no purchase, per the request, only if none of the opened listings qualifies."
        if self.in_reserve():
            return f" ORDER NOW: searching is closed, because the remaining {due()} steps are reserved for the first order attempt plus one fix (first orders placed late seldom succeed). Verify the cart lines if necessary and invoke the order tool."
        if self.order_attempts == 0 and left <= due() + 4:
            return f" Budget of steps is getting tight: soon cart the strongest confirmed match and place the order; searching is cut off when a choice is carted and {due()} steps remain, and fixing a rejected order costs roughly 4 steps."
        return ""

    def status(self) -> str:
        left = self.steps_left()
        head = self._g4_status_counts(left) + self.search_profile()
        head += "\n[requirements now] " + self.ledger()
        return head + self._g4_status_warnings(left) + self._g4_status_pace(left)

    def _g4_ledger_needs(self) -> list[str]:
        if self.current is None:
            return ([f"{r} (stated)" for r in self.stated if r not in self.dropped]
                    + [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped])
        needs = list(self.current)
        joined = " ".join(needs).lower()
        for need in (n for n in self.revealed if n not in self.dropped):
            head = need.split(":")[0].strip().lower()
            word = next((w for w in head.split() if len(w) >= 4), head)
            if word and word not in joined:
                needs.append(f"{need} (revealed by the shopper)")
        return needs

    def _g4_ledger_condition(self, number: int, cond: dict, held: set) -> str:
        state, anchor = self.cond_state.get(number, (None, ""))
        rule = f"{cond['then_product']} {cond['then_requirement']}"
        if state is None or anchor not in held:
            return f"; applies only if the {cond['if_product']} bought has {_cond_test(cond)}: {rule}"
        if state:
            return f"; in force: {rule} (the {cond['if_product']} in the cart, {anchor}, has {_cond_test(cond)})"
        return f"; NOT in force: {rule} (the {cond['if_product']} in the cart, {anchor}, lacks {_cond_test(cond)})"

    def ledger(self) -> str:
        needs = self._g4_ledger_needs()
        parts = ["; ".join(needs) if needs else "see the request"]
        if self.budget_now is not None:
            parts.append(f"; whole order at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip())
        wanted = self.items()
        parts.append("; products and quantities: " + ", ".join(f"{name} x{quantity}" for name, quantity in wanted) if wanted
                     else f"; quantity {self.plan.get('quantity') or 1}")
        extras = ((self.dropped, "; dropped from the requirements: ", "; "), (self.free, "; shopper is indifferent to: ", ", "),
                  (self.resolved, "; taken from the referenced pages: ", "; "))
        parts += [label + glue.join(values) for values, label, glue in extras if values]
        held = set(self.held_lines()) if self.conditions else set()
        parts += [self._g4_ledger_condition(number, cond, held) for number, cond in enumerate(self.conditions, len(self.fit_rules) + 1)]
        if self.pointers:
            parts.append("; still to read from a referenced page: " + "; ".join(f"{r.get('property')} ({r.get('relation')} as the {r.get('shop')} {r.get('kind')} page)" for r in self.pointers))
        return "".join(parts)

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        chain = models or [JUDGE_MODEL, MAIN_MODEL]
        # past the soft spend line a check still happens, only on the lighter model, so one task cannot drain the key
        downgrade = chain[0] == JUDGE_MODEL and self.cost > COST_SOFT
        reply = self.infer([{"role": "user", "content": prompt}], None,
                           [READ_MODEL, JUDGE_MODEL] if downgrade else chain, max_tokens, NO_THINKING)
        return _parse_json_object((reply or {}).get("content") or "")

    def _g5_reader_prompt(self, fresh: list[str], question: str | None) -> str:
        parts = [INTERPRET_PROMPT,
                 "\n\nWHAT THE SHOPPER ORIGINALLY ASKED FOR (background only):\n" + str(self.policy.get("query") or ""),
                 "\n\nTHE REQUIREMENT LEDGER AS IT STANDS:\n" + self.ledger()]
        if self.said:
            parts.append("\n\nPRIOR BACK-AND-FORTH WITH THE SHOPPER, EARLIEST FIRST:\n" + "\n".join(self.said[-12:]))
        if question:
            parts.append("\n\nOUR LAST MESSAGE TO THE SHOPPER:\n" + question)
        parts.append("\n\nTHE SHOPPER'S NEW WORDS:\n" + "\n".join(fresh))
        return "".join(parts)

    def _g5_read_reply(self, prompt: str) -> dict[str, Any] | None:
        first = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not first:
            return None
        spoken = " ".join(map(str, first.get("needs") or []))
        # a bare value is read right first time; limits, exclusions, changes and page pointers get a second, stronger
        # read, since a reversed wording flips a limit and a misread need costs the whole task
        hard = re.search(r"\d|\b(not|no|never|except|but|bar|other than|without|any|only|least|most|under|over|below|above|between|same|different|page|listing)\b", spoken, re.I) is not None
        risky = (first.get("needs") and hard) or any(first.get(k) for k in ("new_budget", "dropped", "new_quantity", "reference"))
        if not risky or self.time_left() <= WRAP_UP_SECONDS + 30:
            return first
        second = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL], 2200, LIGHT_THINKING)
        return _parse_json_object((second or {}).get("content") or "") or first

    def _g5_take_current(self, now: Any, ctx: dict[str, Any]) -> None:
        if not isinstance(now, list) or any(not isinstance(r, str) for r in now):
            return
        def restated(text: str) -> bool:
            return self.restates_kind(text) and text not in self.revealed
        self.current = list(filter(lambda text: text.strip() and not restated(text), now))

    def _g5_collect(self, values: Any, bucket: list[str], tag: str, ctx: dict[str, Any]) -> None:
        for value in values or []:
            if not isinstance(value, str) or not value or value in bucket:
                continue
            bucket.append(value)
            if tag:
                ctx["notes"].append(tag + value)

    def _g5_take_budget(self, budget: Any, ctx: dict[str, Any]) -> None:
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            return
        if budget <= 0 or budget == self.budget_now:
            return
        self.budget_now = budget
        ctx["notes"].append(f"budget is now {budget}")

    def _g5_take_quantity(self, change: Any, ctx: dict[str, Any]) -> None:
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if not isinstance(change, dict):
            return
        wanted = change.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return
        product = str(change.get("product") or "").strip().lower()
        if not self.items():
            # a single kind of product: the plan carries the count
            self.plan["quantity"] = wanted
            ctx["notes"].append(f"quantity is now {wanted}")
            return
        entries = self.plan.get("items")
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict) or not product:
                continue
            named = str(entry.get("product") or "").lower()
            if product in named or named in product:
                entry["quantity"] = wanted
                ctx["notes"].append(f"quantity is now {entry.get('product')} x{wanted}")
                return

    def _g5_take_unavailable(self, flag: Any, ctx: dict[str, Any]) -> None:
        if flag:
            ctx["notes"].append("one item is no longer available: take it out of the cart and leave it unordered")

    def _g5_take_shift(self, result: dict[str, Any], ctx: dict[str, Any]) -> None:
        spoken = " ".join(str(l) for l in ctx["lines"])
        hinted = ctx["question"] is None and (_PRICE_WORDS.search(spoken) or _GONE.search(spoken))
        if not (result.get("dropped") or result.get("new_budget") or result.get("unavailable") or hinted):
            return
        self.changed_at = self.step
        goal = self.plan.get("priority")
        if goal in ("cheapest", "best"):
            ctx["notes"].append("which listings qualify has shifted, so pick afresh across everything seen: the " + str(goal)
                                + " one that satisfies the requirements as they are now (a listing turned down only because of a "
                                "requirement since dropped now qualifies; a sold-out one does not; a changed price is read again), "
                                "running one more search if steps remain")

    def _g5_take_pointer(self, pointer: Any, ctx: dict[str, Any]) -> None:
        if not isinstance(pointer, dict) or not pointer.get("property"):
            return
        if not (pointer.get("shop") or pointer.get("kind")):
            return
        pointer["source"] = "told"
        pointer.setdefault("text", " ".join(str(l) for l in ctx["lines"])[:500])
        clash = next((p for p in self.pointers if p.get("property") == pointer.get("property") and p.get("shop") == pointer.get("shop")), None)
        if clash is None:
            self.pointers.append(pointer)
            ctx["notes"].append("this requirement refers to a different listing's page; the resolver follows those pages next and puts the value into the requirements")

    def _g5_take_split(self, flag: Any, ctx: dict[str, Any]) -> None:
        question = ctx["question"]
        if flag is not True or question is None:
            return
        # two subjects were heard: a question that named two properties is put again one property at a time; one that
        # named a single property was simply too broad (race 9: "packaging" covering "pack type" and "packaging type")
        self.reask = len(_prop_parts(_question_property(str(question)))) > 1
        if self.reask:
            self.single_only = True
        ctx["notes"].append("that question covered two things the shopper cares about: next, ask about a single narrower property from that family, using the name the listings use")

    def _g5_names_product(self, question: str) -> bool:
        products = self.items()
        if not products:
            return True
        asked = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
        filler = ("and", "the", "for", "set", "with", "pair", "pack", "of")
        own = {w.rstrip("s") for name, _ in products for w in name.lower().split() if len(w) >= 3 and w not in filler}
        return bool(own & asked)

    def _g5_take_reply(self, result: dict[str, Any], ctx: dict[str, Any]) -> None:
        question = ctx["question"]
        unclear = question is not None and result.get("unclear") is True
        self.last_unclear = unclear
        if unclear:
            if self._g5_names_product(question):
                # repeating our own question seldom helps: the property asked about is most likely not a need
                ctx["notes"].append("the shopper was lost by that message: skip clarifying or re-sending it; treat that property as likely open, and if any messages remain, move to some other property while naming the product")
            else:
                # with several products, a question that names none of them is ambiguous; the property may still count
                ctx["notes"].append("the shopper was lost by that message because it left the product unnamed: if any messages remain, put the same property to them again with the product spelled out")
            return
        reply = str(result.get("reply") or "").strip()
        # only a line that asks something is answered; a stated need is not confirmed back
        if reply and self.roles.get("message") and _ASKS.search(" ".join(ctx["fresh"])):
            self.pending_reply = reply

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = list(filter(lambda line: line not in self.heard, lines))
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = self._g5_reader_prompt(fresh, question)
        self.said.append(("" if not question else f"asked: {question} -> ") + "shopper: " + " / ".join(fresh))
        result = self._g5_read_reply(prompt)
        if result is None:
            return ""
        ctx: dict[str, Any] = {"lines": lines, "question": question, "fresh": fresh, "notes": []}
        steps = (
            ("requirements_now", self._g5_take_current),
            ("needs", lambda v, c: self._g5_collect(v, self.revealed, "requirement added: ", c)),
            ("no_preference", lambda v, c: self._g5_collect(v, self.free, "", c)),
            ("dropped", lambda v, c: self._g5_collect(v, self.dropped, "no longer required: ", c)),
            ("new_budget", self._g5_take_budget),
            ("new_quantity", self._g5_take_quantity),
            ("unavailable", self._g5_take_unavailable),
            (None, self._g5_take_shift),
            ("reference", self._g5_take_pointer),
            ("one_at_a_time", self._g5_take_split),
        )
        for key, handler in steps:
            handler(result if key is None else result.get(key), ctx)
        notes = ctx["notes"]
        self.ledger_version += 1 if notes else 0
        self._g5_take_reply(result, ctx)
        tail = " (" + "; ".join(notes) + ")" if notes else ""
        return "Meaning: " + str(result.get("meaning") or "") + tail

    def item_records(self, item: str) -> list[str]:
        own = self.store.get(item)
        pages = [] if own is None else [own.page(3000)]
        # an option listed inside another listing shares its facts: include the holding listing too
        pages.extend(other.page(2000) for other in self.store.listings.values()
                     if other.ident != item and item in list(other.facts.values()))
        if pages:
            return pages[-3:]
        recent = (_compact(_brief(record, 260)) for record in self.records[-12:])
        return [text[:3000] for text in recent if item in text][-3:]

    def _g5_cart_units(self) -> list[tuple[str, int]]:
        fresh_view = self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at
        if not fresh_view:
            return [(ident, self.cart_qty.get(ident, 1)) for ident in self.cart_ids]
        return [(ident, units) for ident, units in self.cart_seen if units > 0]

    def cart_total(self) -> float | None:
        """Latest seen price times units, summed over the cart's lines (None when some line's price is unknown)."""
        units = self._g5_cart_units()
        if not units:
            return None
        found = [self.store.get(ident) for ident, _ in units]
        if any(listing is None or listing.price is None for listing in found):
            return None
        return sum((listing.price * count for listing, (_, count) in zip(found, units)), 0.0)

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        # every line is checked against its own product; the cart-wide total is checked here
        limit = self.budget_now
        if not isinstance(limit, (int, float)) or isinstance(limit, bool):
            limit = None
        total = self.cart_total()
        if None not in (limit, total) and total > limit + 1e-9:
            return (f"Summed over its lines (latest seen prices times units) the cart comes to {total:g}, which exceeds the budget of {limit:g}. "
                    "Trade the most expensive line for a lower-priced listing that still fits that product, or reduce a line's count to the quantity requested for it, and then place the order.")
        # several distinct products mean one cart line each: every one gets a check
        pool = self.cart_ids[-3:] if self.items() else self.cart_ids[-1:]
        checks = (self.verify_item(ident, "ORDER", self.cart_lines.get(ident, "")) for ident in pool)
        return next((verdict for verdict in checks if verdict), "")

    def _g5_verify_prompt(self, item: str, purpose: str, line: str, records: list[str]) -> str:
        parts = [VERIFY_PROMPT,
                 "\n\nWHAT THE SHOPPER ASKED FOR:\n" + str(self.policy.get("query") or ""),
                 "\n\nREQUIREMENTS IN FORCE (original, revealed since, and changed):\n" + self.ledger(),
                 f"\n\nTHE ITEM TO {purpose}: {item}\nWHAT THE STORE HAS SHOWN FOR THIS ITEM, EARLIEST FIRST:\n" + "\n".join(records)]
        if line:
            parts.append("\n\nHOW IT SITS IN THE CART (should this mention a variant or option, assess that variant itself: its values, its price, its stock count): " + line)
        if self.last_cart_view is not None:
            parts.append("\n\nMOST RECENT CART VIEW:\n" + _compact(self.last_cart_view)[:2000])
        labels = self.row_labels()
        if labels:
            parts.append("\n\nLABELS REPEATED ACROSS THIS STORE'S LISTINGS, WITH COUNTS (the item's category label has to match the request's wording taken as a whole, not a nearby label):\n" + labels)
        listing = self.store.get(item)
        kind = (listing.category or "") if listing is not None else ""
        matches = [(name, quantity) for name, quantity in self.items() if kind and _same_product(kind, name)]
        # a category that holds two requested products leaves the line's product undecided: no note then
        if len(matches) == 1:
            parts.append(LINE_PROMPT.format(product=matches[0][0], quantity=matches[0][1]))
        return "".join(parts)

    def _g5_second_opinion(self, prompt: str, item: str) -> dict[str, Any] | None:
        # a quick "does not fit" is re-judged by the heavy model reasoning on its own (it refuses no-thinking calls):
        # race 10 replay, right lines failed 20% -> 2%, wrong lines caught 9/17 -> 7/17, on about a quarter of checks;
        # an alarm makes the agent swap the line either way, so this runs past COST_SOFT too (checks come late)
        if self.cost >= COST_HARD or self.time_left() <= WRAP_UP_SECONDS + 30:
            return None
        reply = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL], 4000, None)
        check = _parse_json_object((reply or {}).get("content") or "")
        self.trace.append({"kind": "verify_confirm", "item": item, "result": check})
        return check

    def verify_item(self, item: str, purpose: str, line: str = "") -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = f"{item}|{line}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = self._g5_verify_prompt(item, purpose, line, records)
        reply = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL, READ_MODEL], 1500, NO_THINKING)
        result = _parse_json_object((reply or {}).get("content") or "")
        self.trace.append({"kind": "verify", "item": item, "result": result})
        verdict = ""
        if result and result.get("fits") is False:
            check = self._g5_second_opinion(prompt, item)
            fits = (check or {}).get("fits")
            if not (check and fits is True):
                basis = check if check and fits is False and check.get("problems") else result
                reasons = "; ".join(map(str, list(basis.get("problems") or [])[:5]))
                verdict = (f"A separate review finds that {item} fails at least one requirement: " + reasons
                           + ". Pick a different fit you have fully checked, unless you are sure the review is mistaken.")
        self.verdicts[key] = verdict
        return verdict

    def _g5_shorten(self, message: dict[str, Any]) -> dict[str, Any]:
        body = message.get("content")
        if message.get("role") != "tool" or not isinstance(body, str) or len(body) <= TRIM_CHARS + 200:
            return message
        return {**message, "content": body[:TRIM_CHARS] + " ...(earlier tool output cut short; the facts, prices and stock of its listings stay available to the checks)"}

    def window(self) -> list[dict[str, Any]]:
        """What the model sees: in a long conversation, tool output before the last KEEP_TURNS messages is cut short
        (the listing index still holds every fact for the checks), in whole blocks so the cached prefix survives."""
        size = 0
        for message in self.messages:
            body = message.get("content")
            size += len(body) if isinstance(body, str) else 0
        raw = len(self.messages) - KEEP_TURNS
        cut = raw - raw % TRIM_BLOCK
        if size < WINDOW_CHARS or cut <= 2:
            return self.messages
        head = [self._g5_shorten(message) for message in self.messages[:cut]]
        return head + self.messages[cut:]

    def _g5_uncapped(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [t for t in tools if t["function"]["name"] not in self.capped] or tools

    def offered_tools(self) -> list[dict[str, Any]]:
        """One fixed tool list for the main model on every turn (tools lead the cached prefix, so changing the list
        re-bills the whole conversation at the write price); calls outside tools_for_turn are refused with a reason,
        costing no step. Only the final steps and the spending cap switch to the narrowed list."""
        narrow = self.steps_left() <= 3 or self.cost > COST_HARD + 0.02
        return self.tools_for_turn() if narrow else self._g5_uncapped(self.tools)

    def _g5_turn_names(self, left: int) -> set[str] | None:
        role = self.roles.get
        finish = (role("order"), role("decline"))
        with_cart = finish + (role("cart"),)
        with_reads = with_cart + (role("cart_view"), role("view"), role("details"))
        rules = (
            (lambda: left <= 1, lambda: finish),
            (lambda: left <= 3 or self.cost > COST_HARD + 0.02, lambda: with_cart),
            (lambda: self.cost > COST_HARD, lambda: with_reads),
            (lambda: left <= 7 and (self.order_attempts == 0 or self.refusals > 0), lambda: with_reads),
            # steps held back for the first order: reads of the pick, the cart and the finish, no further searching
            (lambda: self.in_reserve(), lambda: with_reads + ((self.compare_tool() or ("",))[0],)),
        )
        picked = next((build for test, build in rules if test()), None)
        return None if picked is None else set(picked())

    def tools_for_turn(self) -> list[dict[str, Any]]:
        names = self._g5_turn_names(self.steps_left())
        tools = self.tools
        if names is not None and self.roles.get("order") and self.roles.get("cart"):
            # a checkout-finishing tool no role claims (a confirm step after ordering) stays on offer
            assigned = set(self.roles.values())
            every = [t["function"]["name"] for t in self.tools]
            names.update(n for n in every if n not in assigned and _FINISH_STEP.search(n))
            narrowed = [t for t, n in zip(self.tools, every) if n in names]
            tools = narrowed or tools
        message_tool = self.roles.get("message")
        if self.planner_on and message_tool:
            tools = [t for t in tools if t["function"]["name"] != message_tool]
        return self._g5_uncapped(tools) if self.capped else tools

    @staticmethod
    def _g6_take(bucket: list[str], values: list[str]) -> None:
        # a property keeps at most five example values, each once, in the order first seen
        for value in values:
            if len(bucket) >= 5:
                break
            if value not in bucket:
                bucket.append(value)

    @staticmethod
    def _g6_kept_fact(name: str, value: str) -> bool:
        # codes such as "ab-123" name the listing; meta and kind keys are no property to ask about
        code_like = bool(re.search(r"\d", value)) and re.fullmatch(r"[A-Za-z0-9_\-]+", value) is not None and ("-" in value or "_" in value)
        if code_like:
            return False
        return not (_LISTING_META.search(name) or _KIND_KEYS.search(name))

    def _g6_tally(self, groups: dict[str, list[Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, set[frozenset[str]]]]:
        """Per label, per property key: how many listings carry it, its values, up to five examples and the carriers;
        plus, per label, the key pairs one listing gives different values."""
        stats: dict[str, dict[str, dict[str, Any]]] = {}
        apart: dict[str, set[frozenset[str]]] = {}
        for group, listings in groups.items():
            for listing in listings:
                kept = [(name, value) for name, value in listing.facts.items() if self._g6_kept_fact(name, value)]
                for name, value in kept:
                    rec = stats.setdefault(group, {}).setdefault(name, {"n": 0, "vals": set(), "seen": [], "ids": set()})
                    rec["ids"].add(listing.ident)
                    rec["n"] += 1
                    rec["vals"].add(value.lower())
                    self._g6_take(rec["seen"], [value.lower()])
                plain = [(name, value.strip().lower()) for name, value in kept]
                pairs = apart.setdefault(group, set())
                pairs.update(frozenset((a, b)) for at, (a, va) in enumerate(plain) for b, vb in plain[at + 1:] if va != vb)
        return stats, apart

    def _g6_fold(self, recs: dict[str, dict[str, Any]], together: set[frozenset[str]] | None) -> None:
        # spellings of one property (a seller's prefix, a qualifier) join the most carried one, unless one listing
        # gives them different values
        canon = _fold_keys(list(recs), together, {name: rec["n"] for name, rec in recs.items()})
        for name, target in canon.items():
            if target == name or target not in recs:
                continue
            gone = recs.pop(name)
            into = recs[target]
            into["ids"] |= gone["ids"]
            into["n"] = len(into["ids"])
            into["vals"] |= gone["vals"]
            self._g6_take(into["seen"], gone["seen"])

    @staticmethod
    def _g6_names_listing(rec: dict[str, Any]) -> bool:
        # a value that differs on every listing, or runs long, names the listing rather than a property of it
        seen = rec["seen"]
        mean = sum(map(len, seen)) / len(seen) if seen else 0
        if mean > 45:
            return True
        return rec["n"] >= 3 and len(rec["vals"]) == rec["n"] and mean > 12

    def _g6_menu_lines(self, recs: dict[str, dict[str, Any]], limit: int) -> list[str]:
        usable = [(name, rec) for name, rec in recs.items() if not self._g6_names_listing(rec)]
        widest = max((rec["n"] for _, rec in usable), default=0)
        common = {name for name, _ in sorted(usable, key=lambda item: -item[1]["n"])[:5]}
        # keys whose values differ between listings lead: only they can change the choice
        ordered = sorted(usable, key=lambda item: (len(item[1]["vals"]) < 2, -item[1]["n"]))[:limit]
        # the line shape "- name (in N listings, D different values" is read back by the question planner
        lines = []
        for name, rec in ordered:
            note = "; most listings have it" if name in common and rec["n"] >= max(2, widest / 2) else ""
            lines.append(f"- {name} (in {rec['n']} listings, {len(rec['vals'])} different values{note}): " + " | ".join(rec["seen"]))
        return lines

    def property_menu(self) -> tuple[str, int]:
        # pages reached through a reference chain are often another kind of product: not this shopper's properties
        groups = {g: [item for item in v if item.ident not in self.chain_pages] for g, v in self.store.groups().items()}
        fact_sets = sum(map(len, groups.values()))
        stats, apart = self._g6_tally(groups)
        for group, recs in stats.items():
            self._g6_fold(recs, apart.get(group))
        sizes = {g: len(v) for g, v in groups.items()}
        labelled = sorted((g for g in stats if g and sizes.get(g, 0) >= 2), key=lambda g: -sizes.get(g, 0))
        if len(labelled) >= 2 and self.items():
            out: list[str] = []
            for group in labelled[:4]:
                out.append(f"Group \"{group}\" ({sizes[group]} listings opened):")
                out += self._g6_menu_lines(stats[group], 14)
            if stats.get(""):
                out.append("Listings carrying no label:")
                out += self._g6_menu_lines(stats[""], 10)
            return "\n".join(out), fact_sets
        blank = stats.get("", {})
        pool: dict[str, dict[str, Any]] = {}
        for recs in stats.values():
            for name, rec in recs.items():
                if name not in pool:
                    seed = blank.get(name)
                    pool[name] = {"n": 0, "vals": set(seed["vals"]) if seed else set(), "seen": list(seed["seen"]) if seed else []}
                slot = pool[name]
                slot["n"] += rec["n"]
                slot["vals"] |= rec["vals"]
                self._g6_take(slot["seen"], rec["seen"])
        return "\n".join(self._g6_menu_lines(pool, 28)), fact_sets

    def row_labels(self) -> str:
        return self.store.label_counts()

    @staticmethod
    def _g6_free_strings(properties: dict[str, Any]) -> list[str]:
        """Argument names that take free text: string-typed, without a closed list of values."""
        def free(spec: Any) -> bool:
            if not isinstance(spec, dict):
                return False
            kind = spec.get("type")
            return (kind == "string" or "string" in (kind or [])) and "enum" not in spec
        return [key for key, spec in properties.items() if free(spec)]

    def text_args(self, role: str, text: str) -> dict[str, Any] | None:
        """Call arguments for the role's tool when it accepts a single text field (searching or messaging), else None."""
        schema = self.schemas.get(self.roles.get(role) or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = self._g6_free_strings(properties)
        target = next(filter(strings.__contains__, required), None) or self.query_key(properties, strings)
        if not target:
            return None
        out: dict[str, Any] = {target: _clip(text, properties[target].get("maxLength"))}
        # one tool, several ways of working: each other required mode takes its free-text value
        modes = {key: _mode_value((properties.get(key) or {}).get("enum") or [], True) for key in required if key != target}
        if None in modes.values():
            return None
        out.update(modes)
        return out

    def query_key(self, properties: dict[str, Any], strings: list[str]) -> str | None:
        """Pick the text field by name; on a tie or with no query-ish name, fall back to the field the model
        most often put words in on searches that returned rows."""
        key = _text_key(properties, strings)
        if len(strings) < 2:
            return key
        best, runner = sorted(map(_key_score, strings), reverse=True)[:2]
        if best > 0 and best != runner:
            return key
        voted = [k for k in strings if self.key_votes.get(k)]
        return max(voted, key=lambda k: self.key_votes[k]) if voted else key

    def _g6_has_words(self, args: dict[str, Any], query: str | None, properties: dict[str, Any]) -> bool:
        # words belong in the free-text argument; a filter (a category, a maker) is no words (race 10: a filter search
        # sent in text mode came back empty)
        if query:
            value = args.get(query)
            return isinstance(value, str) and bool(value.strip())
        return any(isinstance(v, str) and v.strip() and "enum" not in (properties.get(k) or {}) for k, v in args.items())

    def search_mode(self, args: dict[str, Any]) -> dict[str, Any]:
        """For a search tool that requires a mode: a worded call gets the text mode, a filter-only call the opposite one."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        properties = schema.get("properties") or {}
        words = self._g6_has_words(args, self.query_key(properties, self._g6_free_strings(properties)), properties)
        changes: dict[str, Any] = {}
        for key in schema.get("required") or []:
            values = [v for v in (properties.get(key) or {}).get("enum") or [] if isinstance(v, str)]
            if len(values) < 2:
                continue
            want, other = _mode_value(values, words), _mode_value(values, not words)
            if want is None or want == other:
                continue
            # only a mode of the other way is switched; a third mode the model chose stays
            current = args.get(key)
            if current == other or current not in values:
                changes[key] = want
        return dict(args, **changes) if changes else args

    def shelf_args(self, kind: str) -> dict[str, Any]:
        """Fill an optional enum filter of the search tool with the value matching this product kind, if any."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        words = _kind_words(kind)
        fields = (schema.get("properties") or {}).items()
        if not words:
            return {}
        for key, spec in fields:
            values = spec.get("enum") if isinstance(spec, dict) else None
            if not values or key in (schema.get("required") or []):
                continue
            shelves = [(value, _kind_words(str(value))) for value in values]
            match = next((v for v, w in shelves if w == words), None) or next(
                (v for v, w in shelves if w >= words or words >= w), None)
            if match is not None:
                return {key: match}
        return {}

    def rank_target_rows(self, ids: list[str], target: dict[str, Any]) -> list[str]:
        """Candidate rows for a referenced page, best guess first: same-kind rows sharing more of the reference's
        fact words rank higher; rows plainly from a different seller are dropped, since opening one wastes a step."""
        shop = str(target.get("shop") or "").strip().lower()
        kind = _kind_words(str(target.get("kind") or ""))
        facts = target.get("facts") if isinstance(target.get("facts"), dict) else {}
        tokens = {w for v in facts.values() for w in re.findall(r"[a-z0-9.]+", str(v).lower()) if len(w) >= 2 or w.isdigit()}
        scored: list[tuple[tuple[int, int], str]] = []
        for ident in ids:
            listing = self.store.get(ident)
            if listing is None:
                scored.append(((1, 0), ident))
                continue
            if shop and listing.shop_named and shop not in listing.seller.lower():
                continue
            shelf = _kind_words(listing.category) if kind and listing.category else None
            astray = shelf is not None and not (shelf >= kind or kind >= shelf)
            words = set(re.findall(r"[a-z0-9.]+", " ".join([listing.title, *listing.texts]).lower()))
            scored.append(((int(astray), -len(tokens & words)), ident))
        return [ident for _, ident in sorted(scored, key=lambda pair: pair[0])]

    @staticmethod
    def _g6_id_array(spec: Any) -> bool:
        if not isinstance(spec, dict) or spec.get("type") != "array":
            return False
        return (spec.get("items") or {}).get("type", "string") == "string"

    def compare_tool(self) -> tuple[str, str] | None:
        """(tool name, id-array argument) of a non-core tool able to open many listings in one call, else None."""
        busy = {self.roles.get(role) for role in ("cart", "order", "decline", "message", "search")}
        for tool in self.tools:
            name = tool["function"]["name"]
            if name in busy:
                continue
            key = next((k for k, spec in ((self.schemas.get(name) or {}).get("properties") or {}).items() if self._g6_id_array(spec)), None)
            if key is not None:
                return name, key
        return None

    def chain_floor(self) -> int:
        """Steps to hold back from a reference chain: unsent questions plus what carting and ordering need (6 at least)."""
        due = self.order_due()
        unsent = self.question_limit() - self.messages_used
        return max(6, due + (unsent if unsent > 0 else 0))

    def resolve_pointers(self) -> None:
        """Take the page references one at a time: decide where each one points, follow it page after page (every page
        on the way gets opened, since the grader looks for those opens) and bind the value the last page gives."""
        search = self.roles.get("search")
        details = self.roles.get("details") or self.roles.get("view")
        if not (search and details):
            return
        while self.pointers:
            if self._g7_resolver_stopped():
                return
            ref = self.pointers.pop(0)
            chain = {"ref": ref, "search": search, "details": details, "opened": [], "result": {},
                     "target": self._g7_decide_target(ref),
                     "label": f"{ref.get('product') + ' ' if ref.get('product') else ''}{ref.get('property')}"}
            found_value, own_value = self._g7_walk_chain(chain)
            line = self._g7_binding_line(chain, found_value, own_value)
            if not line:
                self._g7_report_dead_end(chain)
                continue
            self._g7_apply_line(chain, line)

    def _g7_resolver_stopped(self) -> bool:
        if self.steps_left() <= self.chain_floor() + 2:
            return True
        if self.time_left() < WRAP_UP_SECONDS + 40:
            return True
        return bool(self.dead or self.finished)

    def _g7_decide_target(self, ref: dict[str, Any]) -> dict[str, Any]:
        """Where the reference points; the model reads the sentence when no search words came with it."""
        target = {"shop": ref.get("shop"), "kind": ref.get("kind"), "facts": ref.get("facts") or {}, "search": ref.get("search")}
        if target.get("search") or not ref.get("text"):
            return target
        parsed = self.judge(POINTER_PROMPT + "\n\nSENTENCE TO READ:\n" + str(ref.get("text")), 500, [READ_MODEL, JUDGE_MODEL]) or {}
        target.update({key: parsed[key] for key in ("shop", "kind", "facts", "search") if parsed.get(key)})
        ref.update({key: parsed[key] for key in ("property", "relation") if parsed.get(key)})
        return target

    def _g7_walk_chain(self, chain: dict[str, Any]) -> tuple[str | None, str]:
        """Hop along the chain until a page gives the value itself; (bound value, the page's own value)."""
        for _hop in range(HOP_LIMIT):
            if self.steps_left() <= self.chain_floor() or self.time_left() <= WRAP_UP_SECONDS + 30 or self.cost >= COST_HARD:
                break
            if not self._g7_run_hop(chain):
                break
            result = chain["result"]
            if result.get("value"):
                return self._g7_take_value(chain["ref"], result)
            onward = result.get("next")
            if not isinstance(onward, dict) or not (onward.get("shop") or onward.get("kind")):
                break
            chain["target"] = {"shop": onward.get("shop"), "kind": onward.get("kind"), "facts": onward.get("facts") or {},
                               "search": onward.get("search")}
        return None, ""

    def _g7_take_value(self, ref: dict[str, Any], result: dict[str, Any]) -> tuple[str, str]:
        own = str(result["value"])
        bound = own
        required = str(result.get("required") or "").strip()
        if required and str(ref.get("relation") or "same") in ("same", "different", "counterpart"):
            # which value binds is the sentence's call: the page's own one, or the one the page pairs with it
            if required.lower() != own.lower():
                bound = f"{required} (the counterpart of {own}, as the page pairs them)"
            ref["relation"] = "different" if result.get("avoid") is True else "same"
        elif str(ref.get("relation")) == "counterpart":
            ref["relation"] = f"the partner value that page pairs with its own {own} (the one it says goes with it)"
        return bound, own

    def _g7_hop_queries(self, target: dict[str, Any]) -> tuple[str, str, dict[str, Any], list[str]]:
        """The searches tried for one hop: shop plus the pinning facts, the shop alone, shop plus a fact, shop plus kind;
        then the reference's own search words."""
        shop = str(target.get("shop") or "").strip()
        facts = target.get("facts") if isinstance(target.get("facts"), dict) else {}
        kind = str(target.get("kind") or "").strip()
        queries: list[str] = []
        if shop:
            values = list(facts.values())
            if values:
                queries.append(" ".join([shop] + [str(v) for v in values])[:80])
            queries.append(shop)
            queries.extend(f"{shop} {value}" for value in values[:2])
            if kind:
                queries.append(f"{shop} {kind.split()[0]}")
        words = str(target.get("search") or "").strip()
        tokens = words.split()
        if words and len(tokens) <= 3:
            queries.append(words)
        if words and not queries:
            queries.append(" ".join(tokens[:3]))
        return shop, kind, facts, queries

    def _g7_run_hop(self, chain: dict[str, Any]) -> bool:
        """One hop: try the searches in turn until the model names one of the opened pages as the referenced one."""
        shop, kind, facts, queries = self._g7_hop_queries(chain["target"])
        chain["result"] = {}
        shelf = self.shelf_args(kind)
        for query in queries[:4]:
            if self.steps_left() <= self.chain_floor() or self.cost >= COST_HARD:
                return False
            args = self.text_args("search", query)
            if not args:
                return False
            walk = {"query": query, "shop": shop, "kind": kind, "facts": facts, "text_key": next(iter(args)),
                    "page_args": None, "left_over": [], "drop_shelf": False}
            args.update(shelf)
            walk["args"] = args
            settled = self._g7_scan_pages(chain, walk)
            if walk["drop_shelf"]:
                shelf = {}      # that filter may be the wrong one: later searches go without it
            if settled:
                return True
        return False

    def _g7_scan_pages(self, chain: dict[str, Any], walk: dict[str, Any]) -> bool:
        rounds = max(HOP_BATCHES + 1, 12 // max(1, self.compare_cap))
        for page in range(rounds):
            compare = self.compare_tool()
            per_batch = max(1, min(PAGE_OPENS, self.compare_cap)) if compare else 3
            ids = self._g7_next_ids(chain, walk, page, per_batch)
            if ids is None:
                return False
            if not ids:             # nothing new on this result page, but there is a next one
                continue
            opened = self._g7_open_ids(chain, walk, ids, compare)
            if opened is None:
                continue
            ids, replies = opened
            if self._g7_read_hop(chain, walk, ids, replies):
                return True
            exhausted = not walk["page_args"] and not walk["left_over"]
            if exhausted or self.steps_left() <= self.chain_floor() or (page >= 2 and not compare):
                return False
        return False

    def _g7_next_ids(self, chain: dict[str, Any], walk: dict[str, Any], page: int, per_batch: int) -> list[str] | None:
        """The listings to open next: rows of the last result page still unopened, else a fresh search page.
        None ends the query; an empty list skips to the next result page."""
        pending = walk["left_over"]
        if pending:
            walk["left_over"] = pending[per_batch:]
            return pending[:per_batch]
        call_args = dict(walk["args"])
        if walk["page_args"]:
            call_args.update(walk["page_args"])
        elif page:
            return None
        search = chain["search"]
        call = {"id": f"ref_search_{self.step}_{page}", "type": "function", "function": {"name": search, "arguments": json.dumps(call_args)}}
        self.messages.append({"role": "assistant", "content": f"Searching for the page that the {chain['label']} requirement refers to.", "tool_calls": [call]})
        self._g7_feed_replies(self.run_calls([call], planned=True))
        result = self.records[-1] if self.records else None
        rows = _rows_with_ids(result)
        if not rows:
            walk["drop_shelf"] = walk["query"] == walk["shop"]
            return None
        walk["page_args"] = _next_page(result, self.schemas[search], walk["text_key"], call_args)
        unopened = [ident for ident, _ in rows if ident not in chain["opened"]]
        fresh = self.rank_target_rows(unopened, chain["target"])
        if not fresh:
            return [] if walk["page_args"] else None
        walk["left_over"] = fresh[per_batch:]
        return fresh[:per_batch]

    def _g7_open_ids(self, chain: dict[str, Any], walk: dict[str, Any], ids: list[str], compare: Any) -> tuple[list[str], list[tuple[str, str]]] | None:
        """Open the candidate pages (one side-by-side call when the store has one); pages read before cost no step.
        None when a shrinking side-by-side tool showed none of them (they wait in the left-over queue)."""
        unread = [ident for ident in ids if ident not in self.paged]
        batched = bool(compare) and len(unread) > 1
        if batched:
            name, key = compare
            calls = [{"id": f"ref_open_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps({key: unread})}}]
        else:
            ids = ids[:3]
            details = chain["details"]
            calls = [{"id": f"ref_open_{self.step}_{index}", "type": "function",
                      "function": {"name": details, "arguments": json.dumps(self.text_args("details", ident) or self.neutral_args(details, ident))}}
                     for index, ident in enumerate(i for i in unread if i in ids)]
        if not calls:
            return ids, []
        self.messages.append({"role": "assistant", "content": "Opening the listings that could be the referenced page.", "tool_calls": calls})
        cap_before = self.compare_cap
        replies = self.run_calls(calls, planned=True)
        self._g7_feed_replies(replies)
        if batched and self.compare_cap < cap_before:
            # the side-by-side tool took fewer than asked: the unseen ones go again, fewer per call
            walk["left_over"] = [i for i in ids if i not in self.paged] + walk["left_over"]
            ids = [i for i in ids if i in self.paged]
            if not ids:
                return None
        return ids, replies

    def _g7_read_hop(self, chain: dict[str, Any], walk: dict[str, Any], ids: list[str], replies: list[tuple[str, str]]) -> bool:
        """The model reads the opened pages; True when it names one of them as the referenced page."""
        pages = self.store.pages(ids, 3000) or [text.split("\n[status]")[0][:3500] for _, text in replies]
        chain["opened"] += ids
        self.chain_pages += ids
        ref = chain["ref"]
        parts = [HOP_PROMPT,
                 f"PROPERTY TO TRACE: {ref.get('property')}\nREFERENCE BEING FOLLOWED: shop {walk['shop']}, kind {walk['kind']}, facts {_compact(walk['facts'])}"]
        if ref.get("text"):
            parts[-1] += f"\nTHE REQUEST'S SENTENCE: {ref.get('text')}"
        parts.append("PAGES THAT WERE OPENED:\n" + "\n".join(pages))
        reply = self.infer([{"role": "user", "content": "\n\n".join(parts)}], None, [JUDGE_MODEL, READ_MODEL], 1600, LIGHT_THINKING)
        result = _parse_json_object((reply or {}).get("content") or "") or {}
        chain["result"] = result
        self.trace.append({"kind": "hop", "property": ref.get("property"), "target": chain["target"], "result": result})
        return result.get("page") in ids

    def _g7_binding_line(self, chain: dict[str, Any], found_value: str | None, own_value: str) -> str:
        """The requirement line the value read gives ('' when the chain gave none)."""
        if not found_value:
            return ""
        ref, result, label = chain["ref"], chain["result"], chain["label"]
        where = f"the page chain of the {ref.get('shop')} {ref.get('kind')}"
        relation = str(ref.get("relation") or "same")
        rule = str(result.get("rule") or "").strip()
        excluding = ("anything but", "not ")
        if relation == "same" and rule.lower().startswith(excluding):
            # the first full reading of the request asked for the same value: an exclusion here is a misreading
            rule = own_value or found_value
            result["avoid"] = False
        if rule and rule.lower() not in ("null", "none"):
            line = f"{label}: {rule} (read from {where}, whose last page gives {own_value or found_value})"
            if result.get("avoid") is True or rule.lower().startswith(excluding):
                line += "; a listing that states no value for it does not count as different"
            return line
        by_relation = {
            "same": f"{label}: {found_value} (read from {where})",
            "different": f"{label}: anything but {found_value} (read along {where}; a listing that states no value for it does not count as different)",
        }
        return by_relation.get(relation, f"{label}: {relation}; {where} gives {found_value}")

    def _g7_apply_line(self, chain: dict[str, Any], line: str) -> None:
        """The value read replaces the requirement as worded (a pointer to that shop's page)."""
        ref = chain["ref"]
        self.resolved.append(line)
        shop_name = str(ref.get("shop") or "\n").lower()
        prop = str(ref.get("property") or "").lower()

        def unpointed(lines: list[str]) -> list[str]:
            kept = []
            for entry in lines:
                low = entry.lower()
                if shop_name in low or (prop in low and "page" in low):
                    continue
                kept.append(entry)
            return kept

        self.stated = unpointed(self.stated)
        self.revealed = unpointed(self.revealed)
        (self.stated if ref.get("source") == "stated" else self.revealed).append(line)
        if self.current is not None:
            self.current = unpointed(self.current) + [line]
        self.ledger_version += 1
        self.messages.append({"role": "user", "content": f"[resolver] {line}. Pages opened along the way: {', '.join(chain['opened'])}. "
                              "This value now decides the pick; run a fresh search that uses it.\n" + self.status()})

    def _g7_report_dead_end(self, chain: dict[str, Any]) -> None:
        target = chain["target"]
        read = ", ".join(chain["opened"]) or "none"
        self.messages.append({"role": "user", "content": (
            f"[resolver] Following the page chain for {chain['label']} ended at this hop: shop {target.get('shop')}, kind {target.get('kind')}, "
            f"facts {_compact(target.get('facts') or {})} (pages read so far, already counted: {read}; skip reopening them). "
            "Three more calls at most go to it: one search for the shop's name plus a fact value, one open of the matching listing, "
            "then its notes give the value or the next pointer. Should that come up empty, buy the best verified fit on the other "
            "requirements; this property is not a question for the shopper.\n") + self.status()})

    def _g7_feed_replies(self, replies: list[tuple[str, str]], with_status: bool = False) -> None:
        """Tool results back into the conversation: chain pages trimmed for the resolver, finish calls with the status."""
        for call_id, text in replies:
            body = text + "\n" + self.status() if with_status else _resolver_text(text)
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": body})

    @staticmethod
    def _g7_plain_string(spec: Any) -> bool:
        if not isinstance(spec, dict) or "enum" in spec:
            return False
        kind = spec.get("type")
        return kind == "string" or "string" in (kind or [])

    def message_args(self, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(self.roles.get("message") or "") or {}
        required = list(schema.get("required") or [])
        strings = [key for key, spec in (schema.get("properties") or {}).items() if self._g7_plain_string(spec)]
        first = next((key for key in required if key in strings), None)
        target = first or (strings[0] if len(strings) == 1 else None)
        if not target or set(required) - {target}:
            return None
        return {target: text}

    def message_limit(self) -> int:
        return 3 if self.message_budget is None else self.message_budget

    def kept_messages(self) -> int:
        """How many messages stay held back for the request's finishing rules."""
        reserved_left = self.finish_reserved - self.finish_messages
        free = self.message_limit() - self.messages_used
        return max(0, min(reserved_left, free))

    def question_limit(self) -> int:
        held = max(0, self.finish_reserved - self.finish_messages)
        return max(0, self.message_limit() - held)

    def finish_state(self) -> str:
        return "|".join(map(str, (self.cart_changed_at, self.ledger_version, self.messages_used, self.refusals, self.budget_now)))

    def finish_pending(self) -> bool:
        if not self.finish_steps:
            return False
        state = self.finish_state()
        return state not in self.finish_clear and self.finish_rounds.get(state, 0) < 2

    def note_event(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Add one line to the log that the finish check reads."""
        role = next((key for key, tool in reversed(self.roles.items()) if tool == name), "tool")
        verb = {"search": "searched", "details": "opened", "view": "opened"}.get(role)
        if verb:
            text = f"{verb} {_compact(args)[:120]}"
        else:
            cap = 1200 if role == "cart_view" else 500      # a cart view's total can come after its lines
            text = f"{role} {name} {_compact(args)[:300]} -> {_compact(_brief(observation, 200))[:cap]}"
        self.events.append(f"step {self.step}: {text}")

    def cart_now(self) -> str:
        """The cart as last known (the cart view's lines if it was read after the latest change), each listing at its
        latest price, plus the total of price times quantity when every price is known."""
        viewed = self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at
        lines = [(i, n) for i, n in self.cart_seen if n > 0] if viewed else [(i, self.cart_qty.get(i, 1)) for i in self.cart_ids]
        if not lines:
            return "empty (going by the log)"
        parts: list[str] = []
        amounts: list[float | None] = []
        for ident, units in lines:
            listing = self.store.get(ident)
            price = None if listing is None else listing.price
            parts.append(f"{ident} x{units}" + (" (price not seen)" if price is None else f" at {price:g} each"))
            amounts.append(None if price is None else price * units)
        text = "; ".join(parts)
        if all(a is not None for a in amounts):
            text += f"; sum of the latest prices times quantities: {sum(amounts, 0.0):g}"
        return text

    def _g7_finish_prompt(self) -> str:
        tool_lines = (f"- {fn['name']}: {fn.get('description', '')} | arguments: {_compact(fn.get('parameters') or {})}"
                      for fn in (tool["function"] for tool in self.tools))
        sections = [
            FINISH_PROMPT,
            "REQUEST FROM THE SHOPPER:\n" + str(self.policy.get("query") or ""),
            "FINISHING RULES (taken from the request):\n" + "\n".join(f"- {_compact(step)}" for step in self.finish_steps),
            "STORE TOOLS:\n" + "\n".join(tool_lines),
            f"NOW: at step {self.step} of {self.max_steps} ({self.steps_left()} remain, one kept for the order); "
            f"{self.messages_used} of {self.message_limit()} messages sent; latest cart change at step {self.cart_changed_at}; "
            f"ordering is done with {self.roles.get('order')}.",
            "CART NOW: " + self.cart_now(),
            "LOG OF STEPS:\n" + "\n".join(self.events[-40:]),
        ]
        return "\n\n".join(sections)

    def _g7_finish_ask(self, content: str, models: list[str]) -> dict[str, Any] | None:
        reply = self.infer([{"role": "user", "content": content}], None, models, 1500, NO_THINKING)
        return _parse_json_object((reply or {}).get("content") or "")

    def finish_decision(self) -> dict[str, Any] | None:
        # step 1: the stronger model first (one or two calls an episode; a wrong verdict here costs the whole order)
        prompt = self._g7_finish_prompt()
        result = self._g7_finish_ask(prompt, [JUDGE_MODEL, READ_MODEL])
        # step 2: no usable answer would wave the order through unchecked, so ask once more for plain text
        if result is None and self.time_left() > 35:
            result = self._g7_finish_ask(prompt + "\n\nReply now with only the JSON object, written as plain text.", [READ_MODEL, JUDGE_MODEL])
        # step 3: record the verdict
        self.trace.append({"kind": "finish_check", "result": result})
        return result

    def _g7_finish_window(self, final: bool) -> bool:
        return self.steps_left() >= 2 and self.time_left() >= (20 if final else 30)

    def _g7_finish_call(self, decision: dict[str, Any]) -> tuple[str, dict[str, Any], bool] | None:
        """The call the check asks for, when it is one this step may make: (tool, checked arguments, is a message)."""
        wanted = decision.get("next")
        if not isinstance(wanted, dict):
            return None
        name = str(wanted.get("tool") or "")
        raw = wanted.get("arguments") if isinstance(wanted.get("arguments"), dict) else {}
        blocked = {self.roles.get(r) for r in ("order", "decline", "cart", "search")} - {None}
        if name not in self.schemas or name in blocked:
            return None
        args, problems = _check(raw, self.schemas[name], "args")
        is_message = name == self.roles.get("message")
        if problems or (is_message and self.messages_used >= self.message_limit()):
            return None
        return name, args, is_message

    def _g7_finish_round(self, final: bool) -> str:
        """One check-and-act round: 'clear' (nothing to do), 'return' (stop here, the result needs a look first),
        'stop' (order now) or 'again' (check once more)."""
        state = self.finish_state()
        self.finish_rounds[state] = self.finish_rounds.get(state, 0) + 1
        decision = self.finish_decision() or {}
        planned = self._g7_finish_call(decision)
        if planned is None:
            self.finish_clear.add(state)
            return "clear"
        name, args, is_message = planned
        before = (self.ledger_version, self.budget_now, self.messages_used)
        call = {"id": f"finish_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "Before the order, I will carry out a finishing rule from the request.", "tool_calls": [call]})
        self._g7_feed_replies(self.run_calls([call], planned=True), with_status=True)
        if is_message and self.messages_used > before[2]:
            self.finish_messages += 1
        last = self.records[-1] if self.records else None
        if self.finished or (isinstance(last, dict) and bool(last.get("error"))) or self.steps_left() < 1:
            return "return"
        # a reply to a message goes to interpret(), which works out what it changed; another tool's result is looked
        # at first when the check said it may change what to buy
        if not final and self._g7_result_needs_look(decision, is_message, before):
            return "return"
        if is_message or not self._g7_finish_window(final):
            return "stop"
        return "again"

    def _g7_result_needs_look(self, decision: dict[str, Any], is_message: bool, before: tuple[Any, Any, Any]) -> bool:
        if decision.get("then_order") is False and not is_message:
            return True
        if (self.ledger_version, self.budget_now) != before[:2]:
            return True
        return bool(self.pending_reply or self.last_unclear)

    def run_finish(self, order_args: dict[str, Any] | None, final: bool = False) -> bool:
        """Ordered steps: guard, up to three check-and-act rounds, then the order. Returns False when no finishing call
        was made (the caller then orders as before); with `final` the order is placed regardless of what a round changed,
        and a preparatory call (a cart read for a total) is re-checked rather than followed by the order."""
        order = self.roles.get("order")
        if not order or self.finished or self.dead or not self._g7_finish_window(final):
            return False
        acted = False
        for _round in range(3):
            outcome = self._g7_finish_round(final)
            if outcome == "clear":
                break
            acted = True
            if outcome == "return":
                return True
            if outcome == "stop":
                break
        if not acted:
            return False
        # step: the rules are met, so the order goes out
        self.finish_clear.add(self.finish_state())
        args = dict(order_args) if isinstance(order_args, dict) else self.neutral_args(order, "Order the cart.")
        call = {"id": f"finish_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "All finishing rules are satisfied, so I am ordering the cart now.", "tool_calls": [call]})
        self._g7_feed_replies(self.run_calls([call], planned=True), with_status=True)
        return True

    def facts_tool(self) -> str | None:
        """The view tool, when it is the one that lists a listing's properties and the page tool does not."""
        view = self.roles.get("view")
        if not view or view == self.roles.get("details") or view in self.capped:
            return None
        if self.view_facts:
            return view
        opened_any = False
        for ident in self.paged:
            if ident not in self.store.listings:
                continue
            opened_any = True
            if self.store.listings[ident].facts:
                return None
        return view if opened_any else None

    @staticmethod
    def _g8_search_words(entry: dict[str, Any]) -> set[str]:
        text = " ".join(str(v) for v in entry["args"].values() if isinstance(v, str)).lower()
        return set(re.findall(r"[a-z]{3,}", text))

    def _g8_fact_pool(self, words: set[str], several: bool, entries: list, picks: list[str], known: set, budget) -> tuple[list[str], int]:
        """One product's unopened, buyable rows in search order, and how many of its rows are open already."""
        pool: list[str] = []
        have = 0
        for entry in entries:
            if several and not words & self._g8_search_words(entry):
                continue
            hits = entry["ids"]
            for ident in [i for i in self.store.order if i in hits]:
                listing = self.store.listings.get(ident)
                if listing is None or ident in picks or ident in pool or ident in self.chain_pages:
                    continue
                if ident in known:
                    have += 1
                elif listing.stock is not False and not (budget is not None and listing.price is not None and listing.price > budget):
                    pool.append(ident)
        return pool, have

    def gather_facts(self) -> None:
        """Opens the property lists of the first fitting rows of every requested kind before questions are planned."""
        tool = self.facts_tool()
        if self.facts_gathered or not tool:
            return
        items = self.items()
        several = bool(items)
        products = [str(self.plan.get("product") or "")] if not several else [name for name, _ in items]
        want = min(16, 4 * len(items)) if several else 6
        room = self.steps_left() - max(0, self.question_limit() - self.messages_used) - 14
        known = {l.ident for l in self.store.opened()}
        budget = self.budget_now
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            budget = None
        entries = sorted(self.search_log.values(), key=lambda e: e["step"])
        searched = [self._g8_search_words(e) for e in entries]
        per = max(2, want // max(1, len(products)))
        cheapest_first = self.plan.get("priority") == "cheapest"
        picks: list[str] = []
        found = 0
        for product in products:
            words = set(re.findall(r"[a-z]{3,}", product.lower()))
            if entries and (not several or any(words & s for s in searched)):
                found += 1
            pool, have = self._g8_fact_pool(words, several, entries, picks, known, budget)
            if cheapest_first:
                # cheapest buyable rows first: one reading serves the questions and the cheapest pick
                pool.sort(key=lambda i: float("inf") if self.store.listings[i].price is None else self.store.listings[i].price)
            picks.extend(pool[:max(0, per - have)])
        picks = picks[:max(0, min(want - len(known), room))]
        if len(picks) < 2:
            return
        if several and found < min(len(items), 3) and self.step < 14:
            return          # most of the products still wait for their first search
        self.facts_gathered = True
        calls = [{"id": f"facts_{self.step}_{n}", "type": "function",
                  "function": {"name": tool, "arguments": json.dumps(self.text_args("view", ident) or self.neutral_args(tool, ident))}}
                 for n, ident in enumerate(picks)]
        self.messages.append({"role": "assistant", "content": "First I open the property lists of a few fitting listings, to learn which properties products of this kind come with.", "tool_calls": calls})
        for cid, text in self.run_calls(calls, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": text})

    def _g8_ready(self, force: bool):
        """Whether a planning round may run now: (tool, limit, menu, fact_sets, urgent), or None to wait."""
        tool, limit = self.roles.get("message"), self.question_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return None
        if not (self.asked or self.finished) and self.time_left() > WRAP_UP_SECONDS + 60:
            self.gather_facts()
        menu, fact_sets = self.property_menu()
        urgent = not force and bool(self.plan.get("ask_first")) and not self.asked
        if not force:
            waits = (
                lambda: fact_sets < 4 and self.step < 8 and not urgent,
                # several products: each kind is opened before its questions are planned
                lambda: bool(self.items()) and not urgent and self.step < 14
                and sum(1 for g in self.store.groups().values() if len(g) >= 2) < min(len(self.items()), 3),
                # a later round waits for more listings (or an unclear answer), so it knows the earlier answers
                lambda: bool(self.asked) and not urgent and not self.last_unclear and fact_sets < self.sets_at_last_ask + 2,
                lambda: fact_sets < self.plan_retry_at,
            )
            if any(wait() for wait in waits):
                return None
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return None
        return tool, limit, menu, fact_sets, urgent

    def _g8_plan_prompt(self, count: int, group: int, limit: int, urgent: bool, menu: str) -> str:
        head = [PLAN_PROMPT.replace("QUESTION_COUNT", str(count)), "THE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")]
        if self.hints:
            head.append("WHAT A FIRST READING OF THE REQUEST SUGGESTS: " + "; ".join(self.hints))
        text = "\n\n".join(head)
        if group > 1:
            text = text.replace(f"Choose {count} different properties", f"Choose {count * group} different properties", 1)
            text += (LIST_NOTE.replace("GROUP_SIZE", str(group)).replace("PROPERTY_COUNT", str(count * group))
                     .replace("QUESTION_COUNT", str(count)))
        if SWEEP_LAST and not urgent and limit >= 2 and self.messages_used + count == limit:
            text += SWEEP_NOTE      # the last message lists a few more properties at once
        sections = ["REQUIREMENTS SO FAR:\n" + self.ledger()]
        items = self.items()
        if items:
            sections.append("SEVERAL PRODUCTS ARE REQUESTED (say which one every question concerns): " + "; ".join(name for name, _ in items))
        if self.asked:
            sections.append("ALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked))
        settled = [str(r.get("property")) for r in self.pointers if r.get("property")]
        settled += [line.split(":")[0] for line in self.resolved]
        if settled:
            sections.append("SETTLED BY A PAGE REFERENCE (never ask about these): " + "; ".join(settled))
        if self.plan.get("ask_first"):
            sections.append("MUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or []))
        sections.append("PROPERTIES FOUND SO FAR IN LISTING DETAILS (name, how many listings, sample values):\n" + (menu or "(none seen yet)"))
        return text + "".join("\n\n" + s for s in sections)

    def _g8_noun(self, planned: list[dict[str, Any]]) -> str:
        for q in planned:
            m = re.search(r"\blike the (.+?) to have\b", str(q.get("question") or ""), re.I)
            if m:
                return m.group(1)
        return str(self.plan.get("product") or "item").strip().lower()

    def _g8_is_settled(self, q: dict[str, Any], settled_props: list[str]) -> bool:
        """A question whose every property the request or an earlier answer already settles."""
        product = str(q.get("product") or "").strip().lower()
        source = str(q.get("property") or "") or _question_property(str(q.get("question") or ""))
        parts = _prop_parts(source.strip().lower())
        if not parts:
            return False
        known = [(lp[len(product):].strip() if product and lp.startswith(product) else lp, lp) for lp in settled_props]
        return all(part and any(part in pair for pair in known) for part in parts)

    def _g8_keep(self, q: Any, price_ok: bool, settled_props: list[str]) -> bool:
        """Filters one planned question (and cuts a two-property question to its first property)."""
        if not isinstance(q, dict) or not str(q.get("question") or "").strip():
            return False
        if not re.search(r"\b(which|what|how)\b", str(q.get("question")), re.I):
            return False        # an open question gets no answer
        parts = _prop_parts(str(q.get("property") or ""))
        noun = re.search(r"\blike the (.+?) to have\b", str(q.get("question") or ""), re.I)
        if noun and len(parts) >= 2:
            # one property per question: pairs reveal less and come back "one at a time"
            q["property"], q["question"] = parts[0], f"Which {parts[0]} would you like the {noun.group(1)} to have?"
        if " or " in str(q.get("property") or ""):
            return False
        if not price_ok and re.search(r"\b(spend|budget|price|cost|afford|how much|dollars|cents|money)\b",
                                      str(q.get("question")) + " " + str(q.get("property")), re.I):
            return False
        named = _prop_parts(str(q.get("property") or "") or _question_property(str(q.get("question") or "")))
        if all(self.asked_before(str(q.get("product") or ""), part) for part in named):
            return False
        return not self._g8_is_settled(q, settled_props)

    def _g8_hinted_first(self, planned: list[dict[str, Any]], menu: str, settled_props: list[str]) -> list[dict[str, Any]]:
        hinted = self.hinted_properties(menu)
        picked = [str(q.get("property") or "").lower() for q in planned]
        missing = [h for h in hinted if not any(_same_property(h.lower(), p) for p in picked)]
        if not missing:
            return planned
        noun = self._g8_noun(planned)
        extra = []
        for h in missing:
            q = {"product": "", "property": h, "why": "story hint", "question": f"Which {h} would you like the {noun} to have?"}
            if not self._g8_is_settled(q, settled_props) and not self.asked_before("", str(q["property"])):
                extra.append(q)
        lead: list[dict[str, Any]] = []
        rest: list[dict[str, Any]] = []
        for q in planned:
            target = lead if any(_same_property(str(q.get("property") or "").lower(), h.lower()) for h in hinted) else rest
            target.append(q)
        self.trace.append({"kind": "hinted_first", "added": missing})
        return lead + extra + rest

    def _g8_top_up(self, planned: list[dict[str, Any]], menu: str, slots: int, settled_props: list[str]) -> None:
        """Fills free places with the menu's most carried properties the planner left out."""
        noun = self._g8_noun(planned)
        product_words = {w for w in re.findall(r"[a-z]+", noun.lower()) if len(w) > 2}
        taken = [str(q.get("property") or "").lower() for q in planned]
        rows = [(int(n), int(d), name.strip()) for name, n, d in re.findall(r"^- (.+?) \(in (\d+) listings?, (\d+) different", menu, re.M)]
        for n, d, name in sorted(rows, key=lambda r: -r[0]):
            if n < 2 or len(planned) >= slots:
                break
            low = name.lower()
            if d < 2 or not set(re.findall(r"[a-z]+", low)) - product_words:
                continue
            if any(_same_property(low, t) for t in taken) or self.asked_before("", low):
                continue
            if any(_same_property(low, f.lower()) for f in self.free) or self._g8_is_settled({"product": "", "property": low}, settled_props):
                continue
            planned.append({"product": "", "property": name, "why": "most carried", "question": f"Which {name} would you like the {noun} to have?"})
            taken.append(low)
        self.trace.append({"kind": "topped_up", "planned": [str(q.get("property")) for q in planned]})

    def _g8_by_product_load(self, planned: list[dict[str, Any]], items: list) -> list[dict[str, Any]]:
        """Several products: products not yet told about and less asked about go first."""
        tally: dict[str, int] = {}
        for product in (p for entry in self.asked for p, _ in items if entry.lower().startswith("- " + p.lower())):
            tally[product] = tally.get(product, 0) + 1
        def load(q: dict[str, Any]) -> tuple[bool, int]:
            name = str(q.get("product") or "").lower()
            told = any(_kind_words(name) & _kind_words(line.split(":")[0]) for line in self.revealed)
            return told, next((n for product, n in tally.items() if product.lower() in name or name in product.lower()), 0)
        return sorted(planned, key=load)

    def _g8_select(self, result: Any, urgent: bool, menu: str, fact_sets: int, count: int, group: int) -> list[dict[str, Any]]:
        price_ok = any(re.search(r"spend|budget|price|cost|afford", str(a), re.I) for a in self.plan.get("ask_first") or [])
        settled_props = [line.split(":")[0].strip().lower() for line in self.stated + self.revealed + self.resolved if ":" in line]
        planned = [q for q in (result or {}).get("questions") or [] if self._g8_keep(q, price_ok, settled_props)]
        items = self.items()
        if not items and not urgent:
            planned = self._g8_hinted_first(planned, menu, settled_props)
        if urgent and fact_sets < 4:
            planned = planned[:1]
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.questions_sent]
        if group > 1 and not items and len(planned) < count * group:
            self._g8_top_up(planned, menu, count * group, settled_props)
        return self._g8_by_product_load(planned, items) if items else planned

    def _g8_missed(self, fact_sets: int) -> None:
        # a round that sent nothing is retried once, after another listing is open
        self.plan_misses += 1
        self.plan_retry_at = fact_sets + 1
        if self.plan_misses >= 2:
            self.planner_on = False

    def _g8_ask_call(self, tool: str, text: str) -> dict[str, Any] | None:
        args = self.message_args(text)
        if not args:
            return None
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return None
        return {"id": f"ask_{self.messages_used + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def _g8_carry(self, product: str, props: list[str], messages: list, group: int, revealed_before: int) -> None:
        """Records what was answered; properties named after the told one move into later messages of that product."""
        told = [line.split(":")[0].strip().lower() for line in self.revealed[revealed_before:]]
        hit = next((k for k, p in enumerate(props) if any(_same_property(p.lower(), t) for t in told)), None)
        cut = len(props) if hit is None else hit + 1
        self.asked_props.extend((product.strip().lower(), part.strip().lower()) for part in props[:cut])
        for left in props[cut:]:
            spot = next((k for k, m in enumerate(messages) if m[0] == product and len(m[1]) < group), None)
            if spot is None:
                # no free place: it replaces the last property of the next message about that product
                spot = next((k for k, m in enumerate(messages) if m[0] == product and len(m[1]) > 1), None)
                if spot is None:
                    continue
                messages[spot][1][-1] = left
            else:
                messages[spot][1].append(left)
            owner, names, old = messages[spot]
            messages[spot] = (owner, names, self.question_text(owner, names, old))

    def _g8_send(self, tool: str, limit: int, messages: list, group: int, items: list) -> int:
        sent = 0
        while messages and self.messages_used < limit and not self.finished and self.steps_left() > 2:
            product, props, text = messages.pop(0)
            call = self._g8_ask_call(tool, text)
            if call is None:
                continue
            label = (product + " " if product and items else "") + ", ".join(props)
            self.messages.append({"role": "assistant", "content": f"Next question for the shopper: {label}.", "tool_calls": [call]})
            revealed_before, used_before = len(self.revealed), self.messages_used
            for call_id, reply_text in self.run_calls([call], planned=True):
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + "\n" + self.status()})
                self.asked.append(f"- {label}: asked \"{text}\" -> {reply_text.split(chr(10) + '[status]')[0][:400]}")
            if self.messages_used == used_before:
                break           # not sent: the model carries on from here
            sent += 1
            if self.reask:
                # "one at a time": every listed property is asked singly from now on (the lost one not counted as asked)
                self.reask = False
                self.single_only = True
                singles = [(product, p) for p in props]
                singles += [(m[0], p) for m in messages for p in m[1] if p not in props]
                messages = [(owner, [p], self.question_text(owner, [p])) for owner, p in singles]
                continue
            self._g8_carry(product, props, messages, group, revealed_before)
            if items and messages and len(self.revealed) > revealed_before:
                # several products: the product just told about waits for the others
                told_words = _kind_words(product.lower())
                messages.sort(key=lambda m: bool(told_words & _kind_words(m[0].lower())))
        return sent

    def ask_planned(self, force: bool = False) -> bool:
        """One planning round: gate -> plan prompt -> planner model -> filter -> group -> send -> record."""
        ready = self._g8_ready(force)
        if ready is None:
            return False
        tool, limit, menu, fact_sets, urgent = ready
        several = len(self.items())
        remaining = max(1, limit - self.messages_used)
        # one product: every remaining question at once; several: a few per round, knowing the earlier answers
        count = min(remaining, max(2, several)) if several else remaining
        group = LIST_SIZE if not (self.single_only or urgent or LIST_SIZE < 2) else 1
        prompt = self._g8_plan_prompt(count, group, limit, urgent, menu)
        items = self.items()
        result = self.judge(prompt, 2000, [PLAN_MODEL, READ_MODEL])
        planned = self._g8_select(result, urgent, menu, fact_sets, count, group)
        if group > 1 and (result or {}).get("one_per_message") is True:
            group, self.single_only = 1, True      # the request wants one thing at a time
        messages = self.group_questions(planned[:count * group], count, group)
        if not messages:
            self._g8_missed(fact_sets)
            return False
        self.last_unclear = False
        if not self._g8_send(tool, limit, messages, group, items):
            self._g8_missed(fact_sets)
            return False
        self.reask = False
        self.sets_at_last_ask = fact_sets
        self.planner_on = self.messages_used < limit
        return True

    def question_text(self, product: str, props: list[str], fallback: str = "") -> str:
        """The message naming one product's properties, best first (the planner's wording for a single property)."""
        if fallback and len(props) == 1:
            return fallback
        noun = (product or str(self.plan.get("product") or "item")).strip().lower()
        named = props[-1] if len(props) == 1 else " or ".join([", ".join(props[:-1]), props[-1]])
        return f"Which {named} would you like the {noun} to have?"

    def _g8_entries(self, planned: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
        """(product, property, question, noun) per planned question, one per property (and product)."""
        several = bool(self.items())
        entries: list[tuple[str, str, str, str]] = []
        for q in planned:
            prop = str(q.get("property") or "").strip()
            text = str(q.get("question") or "").strip()
            if not (prop or text):
                continue
            m = re.search(r"\blike the (.+?) to have\b", text, re.I)
            noun = m.group(1) if m else ""
            product = (str(q.get("product") or "").strip() if several else "") or noun
            first = (_prop_parts(prop) or [_question_property(text)])[0]
            duplicate = any(_near_same(first, e[1]) and (not several or _same_product(product, e[0])) for e in entries)
            if not duplicate:
                entries.append((product, first, text, noun if m else product))
        return entries

    def _g8_one_product(self, entries: list, count: int, group: int) -> list[tuple[str, list[str], str]]:
        # message i carries ranks i, i+count, i+2*count, ...
        noun = next((e[3] for e in entries if e[3]), "")
        out = []
        for i, (product, prop, text, _) in enumerate(entries[:count]):
            names = [prop] + [e[1] for e in entries[count + i::count]][: group - 1]
            names = [p for k, p in enumerate(names) if all(not _same_property(p.lower(), o.lower()) for o in names[:k])]
            out.append((product or noun, names, self.question_text(noun or product, names, text)))
        return out

    def _g8_per_product(self, entries: list, count: int, group: int) -> list[tuple[str, list[str], str]]:
        # a message per product per round, in the planner's order of products
        order: list[str] = []
        bucket: dict[str, list] = {}
        for entry in entries:
            key = next((k for k in order if _same_product(k, entry[0])), None)
            if key is None:
                order.append(entry[0])
                bucket[entry[0]] = [entry]
            else:
                bucket[key].append(entry)
        out = []
        for turn in range(5):
            lo = turn * group
            if len(out) >= count or not any(len(v) > lo for v in bucket.values()):
                break
            for key in order:
                chunk = bucket[key][lo:lo + group]
                if not chunk or len(out) >= count:
                    continue
                names: list[str] = []
                for e in chunk:
                    if all(not _same_property(e[1].lower(), o.lower()) for o in names):
                        names.append(e[1])
                out.append((key, names, self.question_text(chunk[0][3] or key, names, chunk[0][2])))
        return out

    def group_questions(self, planned: list[dict[str, Any]], count: int, group: int) -> list[tuple[str, list[str], str]]:
        """The ranked properties as at most `count` messages of up to `group` properties of one product each."""
        entries = self._g8_entries(planned)
        if group <= 1:
            return [(e[0], [e[1]], e[2]) for e in entries[:count]]
        if self.items():
            return self._g8_per_product(entries, count, group)
        return self._g8_one_product(entries, count, group)

    @staticmethod
    def _g8_words(text: str) -> set[str]:
        found = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}
        return found | {w[:-3] + "or" for w in found if len(w) > 5 and w.endswith("our")}

    def hinted_properties(self, menu: str) -> list[str]:
        """Menu properties sharing a word with the story hints, earlier hints first; own name and settled ones excluded."""
        if not self.hints:
            return []
        # after a colon ("how it must look: colour, pattern") come the property words
        topics = [self._g8_words(h.split(":", 1)[-1]) for h in self.hints]
        hint_words = set().union(*topics)
        base_words = self._g8_words(str(self.plan.get("product") or ""))
        product_words = base_words | {w.rstrip("s") for w in base_words} | {w + "s" for w in base_words}
        settled = [line.split(":")[0].strip().lower() for line in self.stated + self.revealed + self.resolved]
        settled += [f.lower() for f in self.free] + [str(r.get("property") or "").lower() for r in self.pointers]
        settled = [s for s in settled if s]
        ranked = []
        for raw in re.findall(r"^- (.+?) \(in \d+ listing", menu, re.M):
            name = raw.strip()
            own = self._g8_words(name) - product_words
            if not own & hint_words:
                continue
            if any(_same_property(name.lower(), s) for s in settled) or self.asked_before("", name):
                continue
            ranked.append((next((i for i, t in enumerate(topics) if own & t), len(topics)), name))
        return [name for _, name in sorted(ranked, key=lambda r: r[0])]

    def restates_kind(self, requirement: str) -> bool:
        """True when the value is just one word taken from the product name (the category says it already)."""
        product_name = str(self.plan.get("product") or "").lower()
        if not product_name or self.items() or re.search(r"&|\band\b|\bor\b", product_name):
            return False
        prop, _, value = requirement.partition(":")
        value_words = re.findall(r"[a-z0-9]+", value.lower())
        if len(value_words) != 1 or value_words[0] not in set(re.findall(r"[a-z0-9]+", product_name)) or re.search(r"\d", value):
            return False
        query_text = str(self.policy.get("query") or "")
        return not any(re.search(r"\b" + re.escape(w), query_text, re.I) for w in re.findall(r"[a-z0-9]+", prop.lower()) if len(w) > 2)

    def asked_before(self, product: str, prop: str) -> bool:
        product, prop = product.strip().lower(), prop.strip().lower()
        several = self.asked_props and self.items()
        # several products: another product's property, or an earlier question with no product, is a new question
        return any(_same_property(prop, pr) for p, pr in self.asked_props
                   if not (several and (not _same_product(product, p) or (p == "" and product))))

    def _g9_text_key(self, properties: dict[str, Any], keys: Any) -> Any:
        """The free-text query argument among `keys` (a plain string field without a fixed choice list)."""
        free = [k for k in keys if isinstance(properties.get(k), dict) and properties[k].get("type") == "string" and "enum" not in properties[k]]
        return _text_key(properties, free)

    def _g9_word_rule(self, entry: dict[str, Any], query: Any, text_key: str) -> str:
        """Compares this search with earlier ones whose words are a strict subset of its words; settles word_rule."""
        mine = set(str(query or "").lower().split())
        for other in self.search_log.values():
            if other is entry or other.get("count") is None:
                continue
            theirs = set(str(other["args"].get(text_key) or "").lower().split())
            if not theirs or not theirs < mine:
                continue
            if other["count"] < entry["count"]:
                self.word_rule = "any"
                return (" Here a longer query returned more matches: the store treats a query as ANY of its words, so every"
                        " added word broadens the results; use as few and as precise words as you can.")
            if other["count"] > entry["count"]:
                self.word_rule = "all"      # a longer query that found fewer: each word is required
        return ""

    def track_search(self, args: dict[str, Any], observation: Any) -> str:
        """Book-keeping for one search reply (matches seen, next page, word rule); returns a note while pages stay unread."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        properties = schema.get("properties") or {}
        paging = {k for k, v in properties.items()
                  if isinstance(v, dict) and re.search(r"follow|next|token|cursor|after|marker|more|page|offset|start", k, re.I)}
        base = {k: v for k, v in args.items() if k not in paging}
        entry = self.search_log.setdefault(_compact(base), {"args": base, "count": None, "ids": set(), "next": None, "step": self.step})
        entry["step"] = self.step
        reported = _match_count(observation)
        if reported is not None:
            entry["count"] = reported
        rows = [ident for ident, _ in _rows_with_ids(observation)]
        entry["ids"].update(rows)
        entry["size"] = max(entry.get("size") or 0, len(rows))
        if len(rows) >= 4:
            self.order_votes[int(rows != sorted(rows))] += 1
        text_key = self._g9_text_key(properties, base) or ""
        entry["next"] = _next_page(observation, schema, text_key, args)
        seen = len(entry["ids"])
        widened = self.learn_lead_words(entry, base, text_key, rows, args, paging) or ""
        if text_key and entry["count"] is not None:
            widened = self._g9_word_rule(entry, base.get(text_key), text_key) or widened
        total = entry["count"]
        if total is None or total <= seen or not entry["next"]:
            return widened.strip()
        onward = json.dumps(dict(base, **entry["next"]))
        if self.row_order() == "ranked":
            return (f"The search reports {total} matches, of which {seen} are shown. Its rows are ranked (best match first), and fitting rows, "
                    f"cheaper ones included, often carry on past this page. Keep reading with {onward} as long as its rows are still the requested kind.{widened}")
        if total <= PAGE_SWEEP:
            return (f"The search reports {total} matches, {seen} shown so far. Rows come by listing id rather than by relevance, so the unseen ones "
                    f"may fit equally well. Before choosing or declining, read the other {total - seen} with {onward} and the pages that follow.{widened}")
        how = {"all": "a listing's title or description must hold every word, and listings seldom repeat a property's name, so add the words of one "
                      "required value the way a listing would put them (the kind's word plus the value, never the property's name; an empty result "
                      "means some word appears in no listing)"}.get(self.word_rule, "put in a word from a requirement or a revealed need, pick a category, or set a price limit")
        return (f"The search reports {total} matches and shows {seen}. What is shown is just the lowest listing ids, not the closest fits: tighten "
                f"the query ({how}) until few enough matches remain to read them all, or fetch the following page with {json.dumps(entry['next'])}.{widened}")

    def _g9_page_shape(self, entry: dict[str, Any], rows: list[str]) -> None:
        """Rows per page, and how deep the paging argument's own maximum lets a search be read."""
        self.page_rows = max(self.page_rows, len(rows))
        props = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        tops = [top for top in ((props.get(key) or {}).get("maximum") for key in entry.get("next") or {})
                if isinstance(top, (int, float)) and not isinstance(top, bool) and top > 0]
        if tops:
            self.page_depth = int(tops[-1] // max(1, self.page_rows)) + 1

    def _g9_compare_leads(self, entry: dict[str, Any], base: dict[str, Any], text_key: str, query: list[str], rows: list[str]) -> None:
        """Updates lead_low / lead_same from every earlier first-page search with the same non-text arguments."""
        others = {k: v for k, v in base.items() if k != text_key}
        peers = (o for o in self.search_log.values()
                 if o is not entry and "first" in o and {k: v for k, v in o["args"].items() if k != text_key} == others)
        for other in peers:
            words = str(other["args"].get(text_key) or "").lower().split()
            if words == query:
                continue
            shorter = min(len(words), len(query))
            shared = next((i for i, (a, b) in enumerate(zip(words, query)) if a != b), shorter)
            if other.get("count") != entry.get("count") or other["first"] != entry["first"]:
                self.lead_low = max(self.lead_low, shared)
            elif shared >= 2 and shared == shorter and len(set(rows)) >= 8:
                # a short page can agree by chance (a ranked store repeats its few best rows for a longer query)
                self.lead_same.add(shared)

    def learn_lead_words(self, entry: dict[str, Any], base: dict[str, Any], text_key: str, rows: list[str], args: dict[str, Any], paging: set[str]) -> str:
        """Learns how many leading words of a query this store's keyword search reads (race 14: two), from first-page
        searches with equal other arguments: differing results mean a word past the shared prefix counted; an extension
        with the same count and rows (8+ rows) means none did. Notes the finding once, then reminds on longer queries."""
        query = str(base.get(text_key) or "").lower().split() if text_key else []
        if not query or any(args.get(k) for k in paging):
            return ""
        entry["first"] = list(rows)
        if entry.get("count") is not None and rows:
            self._g9_page_shape(entry, rows)
        prices = [self.store.listings[i].price for i in rows if i in self.store.listings and self.store.listings[i].price is not None]
        if len(prices) >= 4:
            self.price_votes[int(prices != sorted(prices))] += 1
        known = self.lead_words
        self._g9_compare_leads(entry, base, text_key, query, rows)
        self.lead_words = min((k for k in self.lead_same if k > self.lead_low), default=0)
        n = self.lead_words
        if not n:
            return ""
        if not known:
            return (f" Two searches sharing their first {n} words gave exactly the same results, so this store reads just {n} words of a"
                    f" query and discards the remainder. Open each query with its {n} strongest words (for instance the wanted value beside"
                    f" the kind's name, or the shop's name beside the kind); every further value gets a brief query of its own.")
        if len(query) > n:
            return (f" This store counts only a query's first {n} words (\"{' '.join(query[n:])}\" went unread):"
                    f" lead with the {n} most telling words.")
        return ""

    def search_profile(self) -> str:
        """One status line of search-tool facts gathered from earlier replies; empty until something is known."""
        reading = (f"reads only the first {self.lead_words} words of a query (later words are ignored)" if self.lead_words
                   else f"reads {self.lead_low + 1} or more words of a query" if self.lead_low >= 2 else "")
        rule = {"all": "all words must match (adding words cuts the matches)",
                "any": "any single word matches (adding words adds matches; the closest ones need not come first)"}.get(self.word_rule, "")
        order = self.row_order()
        if self.price_votes[0] >= 2 and self.price_votes[0] > self.price_votes[1]:
            sorting = "rows are sorted cheapest first"
        else:
            sorting = {"id": "row order is by listing id: the top rows are not the closest or the cheapest",
                       "ranked": "rows are sorted best match first"}.get(order, "")
        paging = ""
        if self.page_rows:
            paging = f"{self.page_rows} rows per page" + (f", and roughly {self.page_depth} pages of one search can be read" if self.page_depth else "")
        facts = [fact for fact in (reading, rule, sorting, paging) if fact]
        if not facts:
            return ""
        how = [tip for tip in (f"no more than {self.lead_words} words, most telling first" if self.lead_words else "",
                               {"all": "one wanted value plus the kind's name",
                                "any": "the rarest word of one wanted value alone"}.get(self.word_rule, "")) if tip]
        return ("\n[search tool] what this store's replies have shown: " + "; ".join(facts) + "."
                + (" Shape every query as " + ", ".join(how) + "." if how else ""))

    def _g9_probe_blocked(self, words: list[str], text_key: Any, args: dict[str, Any]) -> bool:
        """True when the two-word probe should not be spent now."""
        if (self.probed or self.lead_words or self.lead_low >= 2 or len(words) < 3 or self.steps_left() <= 20
                or self.time_left() < WRAP_UP_SECONDS + 60 or self.finished or self.dead):
            return True
        if self.page_rows < 8:
            return True         # too few rows a page to tell a dropped word from a ranked coincidence
        return any(len(w) < 2 for w in words[:2]) or bool(str(args.get(text_key)).count('"'))

    def probe_search(self, args: dict[str, Any]) -> None:
        """A single extra step spent near the start: a 3+-word keyword query is resent trimmed to two words, which reveals
        word count and all/any matching for later queries."""
        search = self.roles.get("search")
        properties = (self.schemas.get(search or "") or {}).get("properties") or {}
        text_key = self._g9_text_key(properties, args)
        words = str(args.get(text_key) or "").split() if text_key else []
        if self._g9_probe_blocked(words, text_key, args):
            return
        self.probed = True
        probe = dict(args, **{text_key: " ".join(words[:2])})
        call = {"id": f"probe_{self.step}", "type": "function", "function": {"name": search, "arguments": json.dumps(probe)}}
        self.messages.append({"role": "assistant", "content": "Same search again, cut down to its first two words: this shows how the store reads a query.", "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            note = self.track_search(probe, self.records[-1] if self.records else None)
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + (" " + note if note else "") + "\n" + self.status()})

    def _g9_price_list(self, rows: list[Any]) -> str:
        """'id price' for each priced row the store's index knows, out-of-stock ones marked."""
        listed = (self.store.get(i) for i, _ in rows)
        return "; ".join(f"{l.ident} {l.price:g}" + (" (out of stock)" if l.stock is False else "")
                         for l in listed if l is not None and l.price is not None)

    def _g9_pages(self, tool: str, tag: str, say: str, args: dict[str, Any] | None, limit: int, open_for: Any, handle: Any) -> Any:
        """Page requests of one search: one store call per page while `open_for()` holds; `handle(args, replies, result)`
        posts the page and returns (next page's arguments or None, rows). Yields whether each page had rows."""
        for _ in range(limit):
            if not args or not open_for():
                return
            call = {"id": f"{tag}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
            self.messages.append({"role": "assistant", "content": say, "tool_calls": [call]})
            replies = self.run_calls([call], planned=True)
            args, rows = handle(args, replies, self.records[-1] if self.records else None)
            yield bool(rows)

    def _g9_post(self, replies: list[Any], rows: list[Any], summary: str, cut: int) -> None:
        """One tool line per reply: the short price summary when rows came back, else the raw reply cut short."""
        self.messages.extend({"role": "tool", "tool_call_id": call_id, "content": summary if rows else reply[:cut]} for call_id, reply in replies)

    def _g9_holds_pick(self, entry: dict[str, Any], picks: list[str]) -> bool:
        """A search with a next page that found one of the picks and was not read to its end."""
        ids = entry.get("ids") or set()
        count = entry.get("count")
        return bool(entry.get("next")) and any(i in ids for i in picks) and not (count is not None and len(ids) >= count)

    def sweep_cheapest(self, floor: int = 9) -> str:
        """Cheapest-item tasks: page through the not-yet-read results of searches containing the candidate (cart line or an
        opened listing) so no lower price stays hidden. Gives back the reference listing id, or '' if no page was fetched."""
        search = self.roles.get("search")
        picks = list(self.cart_ids) or [l.ident for l in self.store.listings.values() if (l.facts or l.ident in self.paged) and l.price is not None]
        if not search or not picks:
            return ""
        # a price-capped search goes first; the pick's own searches are paged only when it showed nothing
        read = self.cap_sweep(picks, floor)
        swept: set[str] = set()
        newest = sorted(self.search_log.values(), key=lambda e: -e["step"])
        for entry in (e for e in newest if not read and self._g9_holds_pick(e, picks)):
            ids = entry.get("ids") or set()
            count = entry.get("count")
            swept |= ids
            if isinstance(count, int) and count - len(ids) > SWEEP_PAGES * max(1, entry.get("size") or 1):
                continue            # deeper than the sweep reaches: value_sweep narrows it instead

            def follow(args: dict[str, Any], replies: list[Any], result: Any, entry: dict[str, Any] = entry) -> tuple[Any, Any]:
                self.track_search(args, result)
                rows = _rows_with_ids(result)
                self._g9_post(replies, rows, f"{len(rows)} further rows (id price): {self._g9_price_list(rows)}", 600)
                return (dict(entry["args"], **entry["next"]) if rows and entry.get("next") else None), rows

            pages = list(self._g9_pages(search, "sweep", "Reading this search's next page so no cheaper fit slips by.",
                                        dict(entry["args"], **entry["next"]), SWEEP_PAGES,
                                        lambda: self.steps_left() > floor and self.time_left() > WRAP_UP_SECONDS + 20, follow))
            read = bool(pages)
        read = read or self.value_sweep(picks, floor)
        if not read:
            return ""
        if self.cart_ids:
            return self.cart_ids[-1]
        # no cart yet: the cheapest in-stock listing opened from those searches
        pool = [l for l in map(self.store.get, (i for i in picks if not swept or i in swept))
                if l is not None and l.price is not None and l.stock is not False]
        best = min(pool, key=lambda l: l.price, default=None)
        return best.ident if best is not None else ""

    def change_stamp(self) -> str:
        return "|".join(str(part) for part in (self.ledger_version, self.budget_now, len(self.dropped)))

    def _g9_cap_shape(self, schema: dict[str, Any]) -> tuple[str, str | None] | None:
        """(ceiling argument, kind argument or None) when the schema takes a numeric price ceiling and nothing else required."""
        properties = schema.get("properties") or {}
        specs = [(k, v) for k, v in properties.items() if isinstance(v, dict)]
        cap = next((k for k, v in specs if v.get("type") in ("number", "integer") and _CAP_ARG.search(k) and not _NOT_CAP_ARG.search(k)), None)
        if not cap:
            return None
        kind = next((k for k, v in specs if v.get("type") == "string" and "enum" not in v and _KIND_KEYS.search(k)), None)
        if any(k not in (cap, kind) for k in schema.get("required") or []):
            return None
        return cap, kind

    def price_cap_tool(self) -> tuple[str, str, str | None] | None:
        """A non-role tool whose schema takes a price ceiling and needs nothing else: (tool, ceiling arg, kind arg or None)."""
        taken = {self.roles.get(r) for r in ("message", "cart", "cart_view", "order", "decline", "details", "view")} - {None}
        shapes = ((name, self._g9_cap_shape(schema)) for name, schema in self.schemas.items() if name not in taken)
        return next(((name,) + shape for name, shape in shapes if shape), None)

    def _g9_ceiling(self, price: float, whole: bool) -> float:
        """Just under `price` (a unit for whole prices, a cent otherwise), and within budget per unit before any items."""
        ceiling = price - (1 if whole or float(price).is_integer() else 0.01)
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        wanted = self.plan.get("quantity")
        quantity = wanted if isinstance(wanted, int) and wanted > 0 else 1
        if budget is None or self.items():
            return ceiling
        return min(ceiling, budget / quantity)

    def cap_sweep(self, picks: list[str], floor: int) -> bool:
        """Cheapest-item tasks with a max-price search tool: per cart line (up to 4), fetch at most two pages of same-category
        listings below its price. Ranked results bury cheap fits; a price ceiling surfaces them (seen in race 13)."""
        found = self.price_cap_tool()
        if not found:
            return False
        tool, cap_key, kind_key = found
        schema = self.schemas.get(tool) or {}
        whole = (((schema.get("properties") or {}).get(cap_key) or {}).get("type") == "integer")
        lines = [i for i in (self.cart_ids or picks) if self.store.get(i) is not None and self.store.get(i).price is not None][:4]
        read = False
        for ident in lines:
            listing = self.store.get(ident)
            stamp = f"{ident}|{listing.price}"
            if stamp in self.capped_sweeps or self.steps_left() <= floor + 1 or self.time_left() < WRAP_UP_SECONDS + 20:
                continue
            self.capped_sweeps.add(stamp)
            ceiling = self._g9_ceiling(listing.price, whole)
            if ceiling <= 0:
                continue
            args: dict[str, Any] = {cap_key: int(ceiling) if whole else ceiling}
            if kind_key and listing.category:
                args[kind_key] = listing.category

            def follow(sent: dict[str, Any], replies: list[Any], result: Any, listing: Any = listing) -> tuple[Any, Any]:
                rows = _rows_with_ids(result)
                self._g9_post(replies, rows, f"{_match_count(result) or len(rows)} matches under {listing.price:g}; rows (id price): {self._g9_price_list(rows)}", 400)
                count = _match_count(result)
                nxt = _next_page(result, schema, "", sent)
                done = not rows or not nxt or (isinstance(count, int) and count <= len(rows))
                return (None if done else dict(sent, **nxt)), rows

            say = (f"Looking for cheaper {listing.category or 'same kind of'} listings: every one priced below {listing.price:g}, "
                   f"which is what {ident} costs.")
            hits = list(self._g9_pages(tool, "capped", say, args, 2, lambda: self.steps_left() > floor, follow))
            read = read or any(hits)
        return read

    def _g9_value_terms(self, base: str) -> Any:
        """Words of each live requirement's value as a listing would write them (no exclusions, prices or page asks)."""
        taken = base.lower().split()
        for requirement in (r for r in self.stated + self.revealed if r not in self.dropped):
            name, _, value = requirement.partition(":")
            if re.search(r"\b(not|no|never|anything but|except|other than|avoid|without|exclud)", value, re.I) or "price" in name.lower():
                continue            # exclusions and price ceilings are words no fitting listing carries
            value = re.sub(r"\b(at (most|least)|no (more|less) than|or (less|more|fewer|higher|lower)|up to|under|over|only|exactly|any|either)\b", " ", value, flags=re.I)
            words = [w for w in re.split(r"[^A-Za-z0-9]+", value) if len(w) >= 2 and w.lower() not in taken]
            if words and len(words) <= 3 and "page" not in requirement.lower():
                yield " ".join(words)

    def value_sweep(self, picks: list[str], floor: int) -> bool:
        """Stores needing every word, where the candidate's search is too big to page: try at most two tighter queries, each
        adding one requirement's value words, and page each while its count stays readable."""
        search = self.roles.get("search")
        if self.word_rule == "any" or not search or self.value_swept:
            return False
        self.value_swept = True         # once per episode: every narrowed search spends a step
        schema = self.schemas.get(search) or {}
        properties = schema.get("properties") or {}
        homes = [e for e in self.search_log.values()
                 if e.get("next") and any(i in (e.get("ids") or ()) for i in picks)
                 and isinstance(e.get("count"), int) and e["count"] > len(e.get("ids") or ())]
        if not homes:
            return False
        home = max(homes, key=lambda e: e["count"])
        text_key = self._g9_text_key(properties, home["args"]) or ""
        base = str(home["args"].get(text_key) or "").strip()
        if not text_key or not base:
            return False

        def follow(sent: dict[str, Any], replies: list[Any], result: Any) -> tuple[Any, Any]:
            self.track_search(sent, result)
            rows = _rows_with_ids(result)
            self._g9_post(replies, rows, f"{_match_count(result)} matches; rows (id price): {self._g9_price_list(rows)}", 400)
            count = _match_count(result)
            nxt = _next_page(result, schema, text_key, sent)
            done = not rows or not nxt or (isinstance(count, int) and count > 4 * max(1, len(rows)))
            return (None if done else dict(sent, **nxt)), rows

        read = False
        for value in list(self._g9_value_terms(base))[:2]:
            if self.steps_left() <= floor + 1 or self.time_left() < WRAP_UP_SECONDS + 20:
                break
            hits = list(self._g9_pages(search, "value", f"Searching with one required value added ({value}) so no cheaper fit slips by.",
                                       dict(home["args"], **{text_key: f"{base} {value}"}), 3, lambda: self.steps_left() > floor, follow))
            read = read or any(hits)
        return read

    def row_order(self) -> str:
        """Vote result on row ordering: "id", "ranked", or "" when there is no clear winner of at least two votes."""
        in_order, other = self.order_votes
        leader = "id" if in_order > other else "ranked" if other > in_order else ""
        return leader if max(in_order, other) >= 2 else ""

    def _g9_next_hint(self, entry: dict[str, Any], seen: str) -> str:
        return f"{json.dumps(entry['args'])} ({seen}; next page: {json.dumps(dict(entry['args'], **entry['next']))})"

    def ranked_unread(self) -> list[str]:
        """Ranked stores: up to two searches that found a carted listing and were read under three pages deep."""
        if self.row_order() != "ranked":
            return []
        newest = sorted(self.search_log.values(), key=lambda e: -e["step"])
        shallow = [(e, e.get("ids") or set()) for e in newest if e.get("next")]
        return [self._g9_next_hint(e, f"{len(ids)} rows seen") for e, ids in shallow
                if len(ids) < 30 and any(i in ids for i in self.cart_ids)][:2]

    def unread_searches(self) -> list[str]:
        """Up to three searches (newest first) with a small match set that was not read to its end."""
        newest = sorted(self.search_log.values(), key=lambda e: -e["step"])
        sized = [(e, e.get("count"), len(e.get("ids") or ())) for e in newest]
        return [self._g9_next_hint(e, f"{count} matches, {seen} seen") for e, count, seen in sized
                if count is not None and e.get("next") and 8 < count <= PAGE_SWEEP and seen < count][:3]

    def order_note(self) -> str:
        return {"ranked": "Rows are ranked best match first, though fitting rows go on past the first page"}.get(
            self.row_order(), "Rows are listed in listing-id order")

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.question_limit()
        blocked = not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS
        args = None if blocked else self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "Replying to the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        last = len(replies) - 1
        for index, (call_id, reply_text) in enumerate(replies):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text + ("\n" + self.status() if index == last else "")})
        return True

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        ident = next((v for v in args.values() if isinstance(v, str) and v in self.store.listings), None)
        if ident and dict(self.cart_seen or []).get(ident, 0) > 0:
            return ""   # a held line: the store may add to it or set it, so the cart reply is checked instead
        counted = [args.get(k, s.get("default", 1)) for k, s in properties.items()
                   if isinstance(s, dict) and s.get("type") == "integer" and "enum" not in s]
        if not counted:
            return ""
        items = self.items()
        if items:
            value = counted[0]
            if isinstance(value, int) and value not in {q for _, q in items}:
                return f"this call asks for {value}, yet the request calls for these quantities: {', '.join(f'{n} x{q}' for n, q in items)}."
            return ""
        value = next((v for v in counted if isinstance(v, int) and v != wanted), None)
        if value is None:
            return ""
        return f"this call asks for {value}, yet the request needs {wanted} of one listing" + (", which takes it out of the cart." if value == 0 else ".")

    def line_gate(self) -> str:
        """Order gate on the cart's lines: each line's page opened, one line for a single product, one line per
        product (at its units) for several."""
        items = self.items()
        key = f"{self.cart_changed_at}|{self.cart_seen_at}"
        held_ids = self.held_lines()
        details, side = self.roles.get("details"), self.compare_tool()
        own_view = self.roles.get("view")
        if own_view == details:
            own_view = None
        listings = self.store.listings
        unopened = [ident for ident in held_ids if ident in listings and not listings[ident].facts]
        unpaged = self.unpaged_lines()
        note = self._g10_page_note(key, unopened, unpaged, own_view, details, side)
        if note is not None:
            return note
        if not items:
            return self._g10_single_note(key, held_ids)
        if key in self.line_blocks:
            return ""
        return self._g10_count_note(key, items)

    def _g10_page_note(self, key: str, unopened: list[str], unpaged: list[str], own_view: Any, details: Any, side: Any) -> str | None:
        """Lines whose properties or page were never shown hold the order once per cart state."""
        mark = "open|" + key
        if not (unopened or unpaged) or mark in self.line_blocks or self.steps_left() <= 3:
            return None
        self.line_blocks.add(mark)
        if not unopened:
            # only the pages are missing: code opens them, the model needs no turn for it
            self.page_next = unpaged
            return (f"Nobody opened the listing page of {', '.join(unpaged[:4])} yet; search rows and property lists are not the page itself. "
                    "It gets opened for you right now. Confirm every line still satisfies all current requirements and retry the order tool.")
        gaps = [f"{', '.join(unopened[:3])} never had their own properties displayed (use {own_view or details or 'the details tool'} on them)"]
        if unpaged:
            tool = f"{details}"
            if side:
                tool += f" (or {side[0]}, which takes several at once)"
            gaps.append(f"no one opened the page of {', '.join(unpaged[:4])} with {tool}; a search row or property list is not the listing's page")
        return ("The order has to wait: " + "; ".join(gaps)
                + ". Fix that first (one reply may hold several calls), confirm every line still satisfies all current requirements, and order afterwards.")

    def _g10_single_note(self, key: str, held_ids: list[str]) -> str:
        mark = "single|" + key
        if len(held_ids) < 2 or mark in self.line_blocks or self.steps_left() <= 3:
            return ""
        self.line_blocks.add(mark)
        return (f"Only one product is wanted, yet {len(held_ids)} lines sit in the cart ({', '.join(held_ids[:4])}). "
                "Remove every line except the listing you picked, leave that one at the quantity required, and then order.")

    def _g10_count_note(self, key: str, items: list[tuple[str, int]]) -> str:
        asked = ", ".join(f"{n} x{q}" for n, q in items)
        if self.cart_seen is None or self.cart_seen_at < self.cart_changed_at:
            cart_view = self.roles.get("cart_view")
            if not cart_view or self.cart_view_at >= self.cart_changed_at or self.steps_left() <= 3:
                return ""
            self.line_blocks.add(key)
            self.cart_view_at = self.cart_changed_at
            return (f"{len(items)} separate products are wanted. Look at the cart before ordering and confirm it has precisely one line "
                    f"for each of them, at these quantities ({asked}), with no other lines; then order.")
        wanted = sorted(q for _, q in items)
        held = sorted(n for _, n in self.cart_seen if n > 0)
        if held == wanted:
            return ""
        self.line_blocks.add(key)
        return (f"Cart and request disagree: the cart has {len(held)} line(s) with unit counts {held}, while {len(items)} lines with unit counts "
                f"{wanted} are needed ({asked}). Put in whatever product is missing (one verified match each, at its own quantity), drop any "
                "surplus line, and order afterwards. When a product really is not in the store, decline the way the request describes.")

    def tie_rules(self) -> list[tuple[str, dict[str, Any]]]:
        """Rules linking the requested products, numbered from 1: fit rules first, conditions after."""
        tagged: list[tuple[str, dict[str, Any]]] = [("fit", r) for r in self.fit_rules]
        tagged.extend(("if", c) for c in self.conditions)
        return tagged

    def dependency_gate(self) -> str:
        """One judged check of a multi-product cart against its fit rules and conditional requirements; a broken rule
        holds the order and queues a search for listings that keep it."""
        rules = self.tie_rules()
        if not rules or not self.items() or self.dep_checks >= 3 or self.cost >= COST_HARD:
            return ""
        held = self.held_lines()
        key = self._g10_dep_key(held)
        if len(held) < 2 or key in self.line_blocks or self.steps_left() <= 6 or self.time_left() < 45:
            return ""
        self.line_blocks.add(key)
        result = self.judge(self._g10_dep_prompt(rules, held), 1800) or {}
        self.dep_checks += 1
        self.trace.append({"kind": "depend", "result": result})
        failing = self._g10_dep_verdict(rules, held, result)
        onward = self._g10_dep_onward(result)
        if onward:
            # pages named by a cart line are read before the order, then the cart is judged again
            self.see_read.update(code.lower() for code in onward)
            self.dep_next = [(code, [code]) for code in onward]
            self.line_blocks.discard(key)
            return ("The cart lines' pages refer onward to: " + ", ".join(onward)
                    + ". They are fetched before your next turn; afterwards place the order again, unless what they say rules a line out, in which case replace it.")
        if not failing:
            return ""
        notes, jobs = [], []
        for kind, rule, entry in failing[:3]:
            note, job = self._g10_dep_failure(kind, rule, entry, held)
            notes.append(note)
            if job is not None:
                jobs.append(job)
        self.dep_next = jobs[:2]
        pick = "the cheapest listing" if self.plan.get("priority") == "cheapest" else "a listing"
        searching = "Next, a search for further listings that satisfy it runs for you and opens its most likely matches. " if self.dep_next else ""
        return ("Checking the rules that link the products showed: " + "; ".join(notes) + ". " + searching
                + f"Swap that line for {pick} satisfying the rule plus all other requirements of its product (with its page opened), "
                "leave the remaining lines as they are, and order.")

    def _g10_dep_key(self, held: list[str]) -> str:
        return "dep|" + ",".join(sorted(held)) + f"|{self.ledger_version}"

    def _g10_dep_prompt(self, rules: list[tuple[str, dict[str, Any]]], held: list[str]) -> str:
        numbered = []
        for number, (kind, rule) in enumerate(rules, 1):
            if kind == "fit":
                line = f"{number}. fit rule: whatever {rule['accessory']} is bought has to fit the {rule['anchor']} bought"
            else:
                line = (f"{number}. conditional requirement: when the {rule['if_product']} has {_cond_test(rule)}, "
                        f"the {rule['then_product']} needs {rule['then_requirement']}")
            if rule.get("text"):
                line += f' ("{rule["text"]}")'
            numbered.append(line)
        products = [str(r.get("accessory") if k == "fit" else r.get("then_product")) for k, r in rules]

        def related(listing: Any) -> bool:
            if listing.ident in held or not (listing.facts or listing.ident in self.paged):
                return False
            return not listing.category or any(_same_product(listing.category, p) for p in products)
        others = sorted(filter(related, self.store.listings.values()), key=lambda l: (l.price is None, l.price or 0))
        sections = [DEPEND_PROMPT,
                    "THE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or ""),
                    "THE RULES:\n" + "\n".join(numbered),
                    "CURRENT REQUIREMENTS:\n" + self.ledger(),
                    "THE CART'S LINES (their pages):\n" + "\n".join(self.store.pages(held, 2500))]
        if others:
            sections.append("OTHER LISTINGS OPENED:\n" + "\n".join(l.page(650) for l in others[:14]))
        return "\n\n".join(sections)

    @staticmethod
    def _g10_rule_number(entry: Any, count: int) -> int | None:
        if not isinstance(entry, dict):
            return None
        try:
            number = int(str(entry.get("rule")).strip().rstrip("."))
        except ValueError:
            return None
        return number if 1 <= number <= count else None

    def _g10_dep_verdict(self, rules: list[tuple[str, dict[str, Any]]], held: list[str], result: Any) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        """Broken rules from the judge's answer; a condition's applying state per anchor is kept, and a change of it
        bumps the ledger (the cart stays judged under the new version)."""
        broken = []
        moved = False
        for entry in result.get("rules") or []:
            number = self._g10_rule_number(entry, len(rules))
            if number is None:
                continue
            kind, rule = rules[number - 1]
            applies = entry.get("applies") is not False
            if kind == "if" and str(entry.get("anchor") or "") in held:
                state = (applies, str(entry.get("anchor")))
                if self.cond_state.get(number) != state:
                    self.cond_state[number] = state
                    moved = True
            if applies and entry.get("holds") is False:
                broken.append((kind, rule, entry))
        if moved:
            self.ledger_version += 1
            self.line_blocks.add(self._g10_dep_key(held))
        return broken

    def _g10_dep_onward(self, result: Any) -> list[str]:
        onward: list[str] = []
        for raw in result.get("see_also") or []:
            code = str(raw).strip()
            if code and code.lower() not in self.see_read:
                onward.append(code)
        return onward[:3]

    def _g10_dep_failure(self, kind: str, rule: dict[str, Any], entry: dict[str, Any], held: list[str]) -> tuple[str, tuple[str, list[str]] | None]:
        dep = entry.get("dependent")
        name = rule.get("accessory") if kind == "fit" else rule.get("then_product")
        link = [str(x) for x in entry.get("link") or [] if str(x).strip()][:4]
        note = f"the {name} in line {dep}" if dep else f"there is no {name} line"
        if kind == "fit":
            note += f" fails to fit the {rule['anchor']} in line {entry.get('anchor')}"
        else:
            note += (f" falls short of {rule['then_requirement']}, required here since the {rule['if_product']} in line {entry.get('anchor')} "
                     f"has {_cond_test(rule)}")
        if link:
            note += f" (a match would name or carry: {', '.join(link)})"
        listings = self.store.listings
        fitting = [str(i) for i in entry.get("fitting") or [] if str(i) in listings and str(i) not in held][:4]
        if fitting:
            priced = (i if listings[i].price is None else f"{i} ({listings[i].price:g})" for i in fitting)
            note += "; already-opened listings that satisfy it: " + ", ".join(priced)
        words = str(entry.get("search") or "").strip()
        return note, ((words, link) if words or link else None)

    def dependency_search(self, jobs: list[tuple[str, list[str]]]) -> None:
        """One round per job of a broken tie rule: query the store, then open fresh result rows, those naming a link
        code first and cheaper in-stock ones after, so the model picks from their pages."""
        search = self.roles.get("search")
        opened: list[str] = []
        onward_only = all(link == [words] for words, link in jobs)      # pages a cart line points the reader to
        rounds = iter(jobs)
        while (job := next(rounds, None)) is not None:
            if not search or self.steps_left() <= 7 or self.time_left() < 40 or self.cost >= COST_HARD:
                break
            words, link = job
            rows = self._g10_tie_rows(search, words, link)
            batch = self._g10_tie_batch(rows, link, opened, onward_only)
            if batch and self.steps_left() > 7:
                self.open_pages(batch)
                opened.extend(batch)
        if not opened:
            return
        if onward_only:
            note = ("[rules] Opened above: the pages of " + ", ".join(opened)
                    + ", which the cart lines' pages point to. Swap a line out if they show it does not fit; if not, order.\n")
        else:
            note = ("[rules] Opened above: the pages of " + ", ".join(opened)
                    + ". Whichever of them keeps the rule and meets its product's other requirements goes into the cart in place of the rule-breaking line"
                    + (" (choose the cheapest that qualifies)" if self.plan.get("priority") == "cheapest" else "") + "; the rest of the cart stays, then order.\n")
        self.messages.append({"role": "user", "content": note + self.status()})

    def _g10_tie_rows(self, search: str, words: str, link: list[str]) -> list[str]:
        """Search the rule's words, then its first linking identifier, until one query returns rows."""
        queries = ([words] if words else []) + link[:1]
        rows: list[str] = []
        for query in queries:
            args = self.text_args("search", query)
            if not args or self.steps_left() <= 7:
                break
            call = {"id": f"tie_search_{self.step}", "type": "function", "function": {"name": search, "arguments": json.dumps(args)}}
            self.messages.append({"role": "assistant", "content": f"Searching for listings that satisfy the rule ({query}).", "tool_calls": [call]})
            replies = self.run_calls([call], planned=True)
            result = self.records[-1] if self.records else None
            self.track_search(args, result)
            self.messages.extend({"role": "tool", "tool_call_id": call_id, "content": _resolver_text(text)} for call_id, text in replies)
            rows = [ident for ident, _ in _rows_with_ids(result)]
            if rows:
                break
        return rows

    def _g10_tie_rank(self, ident: str, tokens: list[str]) -> tuple[bool, float]:
        listing = self.store.get(ident)
        if listing is None:
            return (True, 1e12)
        flat = re.sub(r"[^a-z0-9]+", " ", " ".join([listing.title, *listing.texts, *listing.facts.values()]).lower())
        vocab, joined = set(flat.split()), flat.replace(" ", "")
        # a short code stands as its own word ("ZE2" in "[ZE2-98960]"); a long one may be split up ("DRZ 8040")
        named = any(t in vocab or (len(t) >= 5 and t in joined) for t in tokens)
        return (not named, 1e12 if listing.price is None else listing.price)

    def _g10_tie_batch(self, rows: list[str], link: list[str], opened: list[str], onward_only: bool) -> list[str]:
        squashed = (re.sub(r"[^a-z0-9]", "", piece.lower()) for piece in link)
        tokens = [t for t in squashed if len(t) >= 2]

        def unseen(ident: str) -> bool:
            if ident in self.paged or ident in opened:
                return False
            listing = self.store.get(ident)
            return listing is None or listing.stock is not False
        ranked = sorted(((self._g10_tie_rank(i, tokens), i) for i in rows if unseen(i)), key=lambda pair: pair[0])
        if onward_only:
            ranked = [pair for pair in ranked if not pair[0][0]]
        width = max(1, self.compare_cap) if self.compare_tool() else 2
        return [ident for _, ident in ranked][:width]

    def over_units(self) -> str:
        """Lines whose unit count exceeds the largest quantity requested (some cart tools add units to a line)."""
        items = self.items()
        wanted = max((q for _, q in items), default=0) if items else self.plan.get("quantity")
        if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
            return ""
        over = ", ".join(f"{n} units of {i}" for i, n in (self.cart_seen or []) if n > wanted)
        if not over:
            return ""
        scope = " for any product" if items else ""
        return (f"Right now the cart has {over}, above the {wanted} requested{scope} (a cart call can add its units on top of what the line "
                "already had). Use the cart tool to set every line to exactly its product's quantity (whatever its schema allows: removing "
                "the surplus units or setting the line's count).")

    def held_lines(self) -> list[str]:
        seen = self.cart_seen
        if seen is None or self.cart_seen_at < self.cart_changed_at:
            return list(self.cart_ids)
        return [ident for ident, units in seen if units > 0]

    def unpaged_lines(self) -> list[str]:
        """Held lines whose page neither the details tool nor the comparison tool has shown."""
        if not self.roles.get("details"):
            return []
        paged = self.paged
        return [ident for ident in self.held_lines() if ident not in paged]

    def open_pages(self, ids: list[str]) -> None:
        """Open the pages of these listings, several per call when a side-by-side tool exists, else singly."""
        details, side = self.roles.get("details"), self.compare_tool()
        queue = list(dict.fromkeys(ids))
        for _ in range(len(ids) + 2):
            if not queue:
                break
            size = max(1, self.compare_cap) if side and len(queue) > 1 else 1
            batch, queue = queue[:size], queue[size:]
            if self.steps_left() < 3 or self.finished or self.dead:
                return
            calls = [self._g10_page_call(batch, details, side)]
            self.messages.append({"role": "assistant", "content": "Opening the listing pages for the cart lines.", "tool_calls": calls})
            cap_before = self.compare_cap
            self.messages.extend({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()}
                                 for call_id, text in self.run_calls(calls, planned=True))
            if len(batch) > 1 and self.compare_cap < cap_before:
                # fewer answered than sent: the left-out ones are queued again, in smaller batches
                queue = [ident for ident in batch if ident not in self.paged] + queue

    def _g10_page_call(self, batch: list[str], details: Any, side: Any) -> dict[str, Any]:
        if side and len(batch) > 1:
            name, args = side[0], {side[1]: batch}
        else:
            name = details
            args = self.text_args("details", batch[0]) or self.neutral_args(details or "", batch[0])
        return {"id": f"page_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}

    def unchecked_cheaper(self) -> list[tuple[str, Any]]:
        """Cheapest request: up to five unopened listings of the cart item's kind priced below it (in stock as far as
        shown, inside the budget), cheapest first; several products: two per line, six in all."""
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        asked = self.plan.get("quantity")
        quantity = asked if isinstance(asked, int) and asked > 0 else 1
        cap = self.budget_now
        budget = None if isinstance(cap, bool) or not isinstance(cap, (int, float)) else cap
        if not self.items():
            return self.store.cheaper(self.cart_ids[-1], budget, quantity, self.opened, self.cart_ids)
        # several products, cheapest in total: every line's kind on its own (the budget is for the whole order)
        found: list[tuple[str, Any]] = []
        taken: set[str] = set()
        for ident in self.cart_ids:
            for other, price in self.store.cheaper(ident, None, 1, self.opened, self.cart_ids)[:2]:
                if other not in taken:
                    taken.add(other)
                    found.append((other, price))
        return found[:6]

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this finish call buys nothing; a store may share one finish tool and tell the two apart by an
        argument's enum value (e.g. "refuse" vs "commit")."""
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            spec = properties.get(key) or {}
            if isinstance(value, str) and "enum" in spec and value in self.no_purchase_values(spec):
                return True
        return False

    def no_purchase_values(self, spec: dict[str, Any]) -> set[str]:
        """Enum values of a shared finish tool's argument that buy nothing, by the first rule that yields any: the
        value the request reading named; values worded as a refusal; of two values with one worded as buying, the other."""
        values = [v for v in spec.get("enum") or [] if isinstance(v, str)]
        named = self.plan.get("decline_choice")
        rules = (
            lambda: {named} if isinstance(named, str) and named in values else set(),
            lambda: {v for v in values if _DECLINE_WORDS.search(v)},
            lambda: self._g10_other_of_pair(values),
        )
        for rule in rules:
            picked = rule()
            if picked:
                return picked
        return set()

    @staticmethod
    def _g10_other_of_pair(values: list[str]) -> set[str]:
        if len(values) != 2:
            return set()
        buying = [v for v in values if _BUY_WORDS.search(v)]
        return {v for v in values if v != buying[0]} if len(buying) == 1 else set()

    def audit_decline(self) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.audited = True
        records = self._g10_audit_records()
        if not records:
            return ""
        prompt = "\n\n".join([AUDIT_PROMPT,
                              "THE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or ""),
                              "CURRENT REQUIREMENTS:\n" + self.ledger(),
                              "LISTING RECORDS SEEN (newest first):\n" + "\n".join(records)])
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        return self._g10_audit_verdict(result or {})

    def _g10_audit_records(self) -> list[str]:
        room = 16000
        kept: list[str] = []
        for record in reversed(self.records):
            text = _compact(_brief(record, 160))
            if len(text) <= room:
                kept.append(text)
                room -= len(text)
        return kept

    def _g10_audit_verdict(self, verdict: dict[str, Any]) -> str:
        def ids(field: str) -> list[str]:
            return [str(x) for x in verdict.get(field) or [] if isinstance(x, (str, int))]
        fits = ids("possible_fits")[:5]
        self.audit_clean = bool(fits) and verdict.get("all_met") is True
        dear = [x for x in ids("over_budget_fits") if x in self.store.listings]
        if verdict.get("budget_only") is True and dear and self.budget_now is not None:
            # each fitting listing (or the cart) costs more than the budget: the decline stands
            self.decline_ok = True
            return ""
        if not fits:
            return ""
        reason = f" ({verdict.get('reason')})" if verdict.get("reason") else ""
        text = ("A second, independent review says these listings match or come nearest: " + ", ".join(fits) + reason
                + ". Declining earns 0 in almost every task: look them over (details, stock, price), place the strongest one (one per "
                "requested product) in the cart and order. Decline only when the records of each one plainly break a requirement, the budget or the stock.")
        if self.order_attempts:
            text += (" No purchase has gone through so far: a successful order would have ended the episode, and it is still going, "
                     "so a decline at this point leaves the shopper empty-handed.")
        return text

    def neutral_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            filled, value = self._g10_neutral_value(properties.get(key) or {}, text, declining)
            if filled:
                args[key] = value
        if len(properties) == 1:
            (key, spec), = properties.items()
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def _g10_neutral_value(self, spec: dict[str, Any], text: str, declining: bool) -> tuple[bool, Any]:
        """A plain value for one required argument; an enum of a shared finish tool takes the mode matching the intent."""
        if "enum" in spec and spec["enum"]:
            stop = self.no_purchase_values(spec)
            matching = [v for v in spec["enum"] if isinstance(v, str) and (v in stop) == declining]
            return True, (matching or spec["enum"])[0]
        kind = spec.get("type")
        if kind == "string" or "string" in (kind or []):
            return True, text[: spec.get("maxLength", len(text))]
        if kind == "boolean":
            return True, True
        if kind in ("integer", "number"):
            return True, max(1, spec.get("minimum", 1))
        return False, None

    def _g10_names_seen(self, value: Any) -> bool:
        """Whether a string argument mentions a listing id seen in this episode."""
        if not isinstance(value, str):
            return False
        return any(token and token in self.seen_ids for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value))

    def near_miss_named(self, args: dict[str, Any]) -> bool:
        """True when some finish argument mentions an id from this episode's listings."""
        return any(self._g10_names_seen(value) for value in args.values())

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        rule = self.plan.get("decline_no_names")
        if rule is False:
            return args
        if not rule and not any(self._g10_names_seen(value) for value in args.values()):
            return args
        neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        cleaned: dict[str, Any] = {}
        for key, value in args.items():
            spec = properties.get(key) or {}
            plain = isinstance(value, str) and "enum" not in spec
            cleaned[key] = neutral[: spec.get("maxLength", len(neutral))] if plain else value
        return cleaned

    _G11_ADDING = r"add|plus|put|set|insert|increase|more|include|place|keep|hold"
    _G11_TAKING = r"remov|delet|drop|discard|clear|minus|decreas|subtract|less|unhold|release|out"
    _G11_DECLINE_TEXT = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."

    def _g11_enum_choice(self, key: str, spec: dict[str, Any], required: list[Any]) -> list[Any]:
        """The value for a cart argument with fixed choices: a wording that adds, else one that neither declines nor
        takes away, else the first; an optional argument with no adding wording is left out (empty list)."""
        options = spec["enum"]
        texts = [v for v in options if isinstance(v, str)]
        taking = re.compile(self._G11_TAKING, re.I)
        adding = [v for v in texts if re.search(self._G11_ADDING, v, re.I) and not taking.search(v)]
        if not adding and key not in required:
            return []
        blocked = self.no_purchase_values(spec)
        harmless = [v for v in texts if not (_DECLINE_WORDS.search(v) or v in blocked or taking.search(v))]
        for group in (adding, harmless, texts, options):
            if group:
                return [group[0]]
        return []

    def _g11_line_value(self, spec: dict[str, Any], ident: str, units: int) -> list[Any]:
        kind = spec.get("type")
        if kind == "integer":
            return [units]
        if kind == "string":
            return [ident]
        if kind != "array":
            return []
        item = spec.get("items") or {}
        if item.get("type") != "object":
            return [[ident]]
        fields = item.get("properties") or {}
        return [[{k: units if (v or {}).get("type") == "integer" else ident for k, v in fields.items() if isinstance(v, dict)}]]

    def cart_args(self, ident: str, units: int) -> dict[str, Any] | None:
        """Cart-tool arguments adding `units` of listing `ident`, or None when the schema check fails."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        required = schema.get("required") or []
        built: dict[str, Any] = {}
        for key, spec in (schema.get("properties") or {}).items():
            if not isinstance(spec, dict):
                continue
            chosen = (self._g11_enum_choice(key, spec, required) if "enum" in spec and spec["enum"]
                      else self._g11_line_value(spec, ident, units))
            built.update((key, value) for value in chosen)
        checked, problems = _check(built, schema, "args")
        if problems:
            return None
        return checked

    def _g11_over(self) -> bool:
        return bool(self.finished or self.dead or self.steps_left() <= 0)

    def _g11_audit_text(self, listings: list[Listing], keep: int, chars: int) -> str:
        records = self.store.pages([l.ident for l in listings][-keep:], chars)
        sections = [AUDIT_PROMPT,
                    "WHAT THE SHOPPER ASKED FOR:\n" + str(self.policy.get("query") or ""),
                    "REQUIREMENTS AS THEY NOW STAND:\n" + self.ledger(),
                    "LISTING RECORDS READ SO FAR:\n" + "\n".join(records)]
        return "\n\n".join(sections)

    def _g11_send(self, call_id: str, tool: str, args: dict[str, Any], said: str, wrap: Any = None) -> None:
        """One planned tool call announced by an assistant turn, its results appended as tool messages."""
        call = {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": said, "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": wrap(text) if wrap else text})

    def _g11_order_tail(self, order: str) -> bool:
        """After the cart lines went in: the request's finishing rules when pending, else one plain order call."""
        if self.finish_pending() and self.steps_left() >= 2 and self.run_finish(None, final=True):
            return True
        self._g11_send(f"commit_order_{self.step}", order, self.neutral_args(order, "Order the cart."), "I will order it.")
        return True

    def _g11_last_pool(self) -> tuple[list[Listing], int, bool]:
        """Opened, priced listings never refused at checkout, trimmed by stock and budget (all of them when the
        request settles for the closest and nothing passes)."""
        priced = [l for l in self.store.opened() if l.price is not None and l.ident not in self.refused_ids]
        budget = self.budget_now
        if not isinstance(budget, (int, float)) or isinstance(budget, bool):
            budget = None
        wanted = self.plan.get("quantity")
        quantity = wanted if isinstance(wanted, int) and wanted > 0 else 1
        closest = bool(self.plan.get("closest_if_none"))
        fitting = [l for l in priced if l.stock is not False and (budget is None or l.price * quantity <= budget)]
        return (fitting or (priced if closest else [])), quantity, closest

    def last_commit(self) -> bool:
        """Single-product fallback near the end with an empty cart: buy the nearest opened match."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        if self.items():
            return self.commit_items()
        if (self.cart_ids or self.finished or self.dead or not cart or not order or not self.may_commit()
                or self.steps_left() < 2 or self.time_left() < 12):
            return False
        self.committed = True
        self.commits += 1
        self.commit_at = self.step
        pool, quantity, closest = self._g11_last_pool()
        if not pool:
            return False
        result = self.judge(self._g11_audit_text(pool, 14, 900), 900) or {}
        known = {l.ident for l in pool}
        named = [str(x) for x in result.get("possible_fits") or []]
        fits = [x for x in named if x in known]
        pick = fits[0] if fits else None
        if not pick and closest:
            pick = sorted(pool, key=lambda l: l.price)[0].ident
        self.trace.append({"kind": "last_commit", "pick": pick, "result": result})
        args = self.cart_args(pick, quantity) if pick else None
        if not args:
            return False
        self.forcing = True
        try:
            return self._commit_calls(cart, order, args)
        finally:
            self.forcing = False

    def basket_tool(self) -> bool:
        """True when the cart tool expects the full basket as an array of line objects (each call overwrites the cart)."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        for spec in (schema.get("properties") or {}).values():
            if isinstance(spec, dict) and spec.get("type") == "array" and (spec.get("items") or {}).get("type") == "object":
                return True
        return False

    def may_commit(self) -> bool:
        """A fallback buy happens once, plus one retry if checkout turned that order down."""
        return (not getattr(self, "committed", False)
                or (self.commits < 2 and self.refused_at > self.commit_at and not self.items()))

    def products_unheld(self) -> list[str]:
        """Names of wanted products that no cart line covers yet (single product: its name while the cart is empty)."""
        items = self.items()
        if items:
            held = [self.store.get(i) for i in self.cart_ids]
            kinds = [l.category for l in held if l is not None and l.category]
            return [name for name, _ in items if all(not _same_product(kind, name) for kind in kinds)]
        if self.cart_ids:
            return []
        return [str(self.plan.get("product") or "item")]

    def _g11_seen(self) -> list[Listing]:
        return [l for i, l in self.store.listings.items()
                if (l.facts or i in self.paged) and l.price is not None and l.stock is not False]

    def _g11_items_prompt(self, seen: list[Listing], items: list[tuple[str, int]]) -> str:
        wanted = "; ".join(f"{n} x{q}" for n, q in items)
        return (self._g11_audit_text(seen, 28, 650)
                + "\n\nTHIS REQUEST COVERS MORE THAN ONE PRODUCT: " + wanted
                + '. Add a "picks" list as well: [{"product": the product name exactly as written above, "id": the record'
                  ' that fits that product best, or failing that the nearest one}], one entry for each product; summed'
                  ' over the picks, price times quantity has to stay within the budget.')

    @staticmethod
    def _g11_answer_picks(result: dict[str, Any]) -> list[dict[str, Any]]:
        return [p for p in result.get("picks") or [] if isinstance(p, dict)]

    def _g11_choose(self, result: dict[str, Any], seen: list[Listing], items: list[tuple[str, int]],
                    need: list[str]) -> list[tuple[str, int]]:
        known = {l.ident for l in seen}
        answer = self._g11_answer_picks(result)
        chosen: list[tuple[str, int]] = []
        for name, quantity in filter(lambda pair: pair[0] in need, items):
            match = next((str(p.get("id")) for p in answer
                          if str(p.get("id")) in known and _same_product(str(p.get("product") or ""), name)), None)
            if match and all(match != ident for ident, _ in chosen):
                chosen.append((match, quantity))
        return chosen

    def _g11_missing(self, result: dict[str, Any], picks: list[tuple[str, int]], need: list[str]) -> list[str]:
        held = {i for i, _ in picks}
        covered = lambda name: any(_same_product(str(p.get("product") or ""), name) and str(p.get("id")) in held
                                   for p in self._g11_answer_picks(result))
        return [name for name in need if not covered(name)]

    def _g11_search_for(self, name: str, side: bool) -> bool:
        """One search for a product nothing was read for, and its first rows opened; False when no search can run."""
        args = self.text_args("search", name)
        tool = self.roles.get("search")
        if not args or not tool:
            return False
        self._g11_send(f"commit_search_{self.step}", self.roles["search"], args, f"I will look for the {name}.", wrap=_resolver_text)
        found = [i for i, _ in _rows_with_ids(self.records[-1] if self.records else None)]
        rows = found[: max(1, self.compare_cap)]
        if rows:
            self.open_pages(rows if side else rows[:1])
        return True

    def _g11_plan_picks(self, items: list[tuple[str, int]], need: list[str]) -> list[tuple[str, int]] | None:
        """Plan step: the judge picks one listing per product; a product left without one gets a search, then a second pick."""
        seen = self._g11_seen()
        if not seen:
            return None
        rounds = [("commit_items", None), ("commit_items_again", self._g11_search_missing)]
        picks: list[tuple[str, int]] = []
        for kind, before in rounds:
            if before is not None:
                missing = self._g11_missing(result, picks, need)
                if not (missing and len(missing) <= 2 and self.steps_left() >= 2 * len(missing) + len(items) + 3
                        and self.time_left() > 50):
                    break
                before(missing)
                seen = self._g11_seen()
            result = self.judge(self._g11_items_prompt(seen, items), 1400) or {}
            picks = self._g11_choose(result, seen, items, need)
            self.trace.append({"kind": kind, "picks": picks, "result": result})
        return picks

    def _g11_search_missing(self, missing: list[str]) -> None:
        side = self.compare_tool()
        for name in missing:
            if not self._g11_search_for(name, side):
                return

    def _g11_cart_calls(self, cart: str, picks: list[tuple[str, int]]) -> list[dict[str, Any]] | None:
        """Calls step: one whole-basket call carrying every pick, or one cart call per pick."""
        first_id, first_units = picks[0]
        whole = self.cart_args(first_id, first_units) or {}
        array_key = next((k for k, v in whole.items() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
        if not (whole and array_key):
            each = [self.cart_args(ident, units) for ident, units in picks]
            return each if all(each) else None
        template = whole[array_key][0]
        lines = [{k: units if isinstance(v, int) and not isinstance(v, bool) else ident if v == first_id else v
                  for k, v in template.items()} for ident, units in picks]
        merged, problems = _check(dict(whole, **{array_key: lines}), self.schemas[cart], "args")
        return None if problems else [merged]

    def _g11_apply(self, cart: str, order: str, picks: list[tuple[str, int]], calls: list[dict[str, Any]]) -> bool:
        """Apply step: unread picks opened, the cart calls made, then the order."""
        unpaged = [i for i, _ in picks if i not in self.paged]
        if unpaged and self.roles.get("details") and self.steps_left() >= len(calls) + 4:
            self.open_pages(unpaged)
        self.forcing = True
        try:
            for index, args in enumerate(calls):
                self._g11_send(f"commit_{self.step}_{index}", cart, args, "I will put the closest fit of each product in the cart.")
                if self.finished or self.steps_left() < 2:
                    return self.finished
            return self._g11_order_tail(order) if self.cart_used else False
        finally:
            self.forcing = False

    def commit_items(self) -> bool:
        """Multi-product fallback (model about to decline, or time running out): plan one pick per product still
        missing from the cart (every product for a whole-basket tool), build the cart calls, apply them and order,
        since declining almost always scores 0."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        items = self.items()
        if not items or self.order_attempts or self.finished or self.dead or not cart or not order:
            return False
        # lines already in a partly filled cart stay; a whole-basket tool re-sends every product
        partial = self.cart_ids and not self.basket_tool()
        need = self.products_unheld() if partial else [name for name, _ in items]
        if getattr(self, "committed", False) or not need or self.steps_left() < len(need) + 2 or self.time_left() < 25:
            return False
        self.committed = True
        picks = self._g11_plan_picks(items, need)
        if picks is None or len(picks) < len(need) or self.steps_left() < len(picks) + 2:
            return False
        calls = self._g11_cart_calls(cart, picks)
        return False if calls is None else self._g11_apply(cart, order, picks, calls)

    def _commit_calls(self, cart: str, order: str, args: dict[str, Any]) -> bool:
        self._g11_send(f"commit_{self.step}", cart, args, "I will put the closest fit in the cart.")
        if self.finished or self.steps_left() < 1 or not self.cart_used:
            return self.finished
        return self._g11_order_tail(order)

    def _g11_gate_note(self, index: int, check: dict[str, Any], view: Any) -> str:
        tool, least, late = check["tool"], check["minimum"], check["after_cart_final"]
        if tool == view and least <= 1:
            return ""       # view_before_order already looks at the cart before each order
        floor = self.cart_changed_at if late else None
        done = len([1 for step, used in self.tool_calls if used == tool and (floor is None or step > floor)])
        key = f"{index}|{self.cart_changed_at}"
        if done >= least or key in self.process_blocks:
            return ""
        self.process_blocks.add(key)
        since = " since the cart last changed" if late else ""
        return (f"Holding the order: the request's rule \"{check['rule']}\" wants {tool} used {least} time(s){since}, and it has been used {done}. "
                "Run the missing uses first (batching them into one reply is allowed), look over the basket against each need again, and only after that order.")

    def process_gate(self) -> str:
        """Order gate for process rules parsed at setup: each rule's tool must have been used the stated number of
        times (counting only uses after the latest cart change when the rule says so). The first unmet rule blocks the
        order once per cart state and returns a note; the model makes the calls, code just counts them."""
        view = self.roles.get("cart_view")
        notes = (self._g11_gate_note(index, check, view) for index, check in enumerate(self.process_checks))
        return next((note for note in notes if note), "")

    @staticmethod
    def _g11_fixed_value(spec: Any) -> tuple[bool, Any]:
        spec = spec if isinstance(spec, dict) else {}
        if spec.get("enum"):
            return True, spec["enum"][0]
        return (True, True) if spec.get("type") == "boolean" else (False, None)

    def _g11_view_absorb(self, view: str, args: dict[str, Any], observation: Any) -> str:
        self.records.append(observation)
        self.tool_calls.append((self.step, view))
        self.store.absorb(observation, self.step)
        self.last_cart_view = observation
        shown = _cart_lines(observation)
        if shown is not None:
            self.cart_seen = [(i, self.cart_qty.get(i, 1) if n is None else n) for i, n in shown]
            self.cart_seen_at = self.step
        self.note_event(view, args, observation)
        return f"[{view} was called first, one step, to look at the cart before the order] " + _compact(_brief(observation, PAGE_CHARS))[:1500]

    def view_before_order(self) -> str:
        """Pre-order cart look: if the cart changed since the store's cart-view tool last ran, run it once from code
        (one step, no model turn) so the cart is seen exactly as the order will see it. Gives back the view text to put
        ahead of the purchase result, or "" when no call was made."""
        view = self.roles.get("cart_view")
        others = {self.roles.get(r) for r in ("cart", "order", "decline", "details", "view", "search", "message")} - {None}
        skip_if = (lambda: not view, lambda: view in others, lambda: view not in self.schemas, lambda: not self.cart_ids,
                   lambda: self.finished, lambda: self.dead, lambda: self.view_called_at >= self.cart_changed_at,
                   lambda: self.cart_changed_at in self.view_tries, lambda: self.steps_left() < 2, lambda: self.time_left() < 8)
        if any(test() for test in skip_if):
            return ""
        schema = self.schemas.get(view) or {}
        properties = schema.get("properties") or {}
        filled = {key: self._g11_fixed_value(properties.get(key)) for key in schema.get("required") or []}
        if not all(ok for ok, _ in filled.values()):
            return ""       # an argument code cannot fill: left to the model
        args = {key: value for key, (_, value) in filled.items()}
        self.view_tries.add(self.cart_changed_at)
        action = {"name": view, "args": args}
        step_no = self.step + 1
        reply = self.env_call([action])
        self.trace.append({"kind": "env", "step": step_no, "actions": [action], "reply": reply, "by": "view_before_order"})
        if not isinstance(reply, dict):
            return ""
        many = reply.get("observations")
        observations = many if isinstance(many, list) else [reply.get("observation")]
        observation = observations[0] if observations else None
        self.view_called_at = self.step
        failed = observation is None or (isinstance(observation, dict) and observation.get("error"))
        return "" if failed else self._g11_view_absorb(view, args, observation)

    def _g11_decline_action(self, decline: str) -> dict[str, Any]:
        return {"name": decline, "args": self.neutral_args(decline, self._G11_DECLINE_TEXT, declining=True)}

    def _g11_ff_call(self, action: dict[str, Any]) -> Any:
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        return reply

    def _g11_ff_commit(self) -> bool:
        """Attempt 1: an empty cart gets the closest opened fit (the cart as it is now decides)."""
        return bool(not self.cart_ids and self.last_commit() and self._g11_over())

    def _g11_ff_rules(self) -> bool:
        """Attempt 2: the request's finishing rules, which order once they are met."""
        if self.roles.get("order") and self.cart_ids and self.finish_pending() and self.steps_left() >= 2:
            self.run_finish(None, final=True)
        return self._g11_over()

    @staticmethod
    def _g11_refusal_lines(reply: Any) -> list[str]:
        observation = (reply or {}).get("observation")
        refused = observation if isinstance(observation, dict) else {}
        parts = [v for v in refused.values() if isinstance(v, (list, str))]
        for v in refused.values():
            if isinstance(v, dict):
                parts.extend(x for x in v.values() if isinstance(x, (list, str)))
        lines: list[str] = []
        _sentences(parts, lines)
        return lines

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        if any(attempt() for attempt in (self._g11_ff_commit, self._g11_ff_rules)):
            return
        order, decline = self.roles.get("order"), self.roles.get("decline")
        if order and self.cart_ids:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
            self.view_before_order()
            if self._g11_over():
                return
        elif decline:
            action = self._g11_decline_action(decline)
        else:
            return
        reply = self._g11_ff_call(action)
        lines = self._g11_refusal_lines(reply)
        said = " ".join(lines)
        # follow-up attempts, in order: refused only for a changed price -> the same order once more; still open -> decline
        follow_ups = (
            (lambda r: (r and not r.get("finished") and lines and order and action["name"] == order and self.cart_ids
                        and _PRICE_WORDS.search(said) and not _GONE.search(said) and self.steps_left() > 0),
             lambda: action),
            (lambda r: (r and not r.get("finished") and isinstance(r.get("observation"), dict) and lines and decline
                        and self.steps_left() > 0),
             lambda: self._g11_decline_action(decline)),
        )
        for due, build in follow_ups:
            if due(reply):
                reply = self._g11_ff_call(build())

    def _g11_note_used_up(self, name: str, observation: dict[str, Any], declined: bool) -> list[str]:
        problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason")
                           if isinstance(observation.get(k), str))
        core = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
        if not (observation.get("error") and _USED_UP.search(problem)) or name in core:
            return []
        if name == self.roles.get("message"):
            used = self.messages_used
            self.message_budget = used if self.message_budget is None else min(self.message_budget, used)
        self.capped.add(name)
        return ["No uses of that tool remain, so it has been withdrawn. Stop calling it: take prices and stock from the "
                "search results or the cart view, and make the call with what you already know."]

    def _g11_note_declined(self, name: str, observation: dict[str, Any], declined: bool) -> list[str]:
        # the shopper answered the decline with a changed request (new budget, dropped need): weigh the decline again
        self.decline_ok = False
        self.decline_checks = min(self.decline_checks, 1)
        self.order_checked = False
        return ["YOUR DECLINE LEFT THE EPISODE RUNNING: the shopper replied with a change, so keep going. Take on every change "
                "they give (a different budget, a requirement removed or added), then decide afresh: when a listing already seen "
                "now satisfies every remaining requirement and the current budget, add it to the cart and order it; when none "
                "does, call the decline tool once more."]

    def _g11_note_refused(self, name: str, observation: dict[str, Any], declined: bool) -> list[str]:
        self.refusals += 1
        self.refused_at = self.step
        self.refused_ids.update(self.cart_ids)
        self.order_checked = False
        notes = ["CHECKOUT REFUSED THE ORDER; the episode goes on with the cart as it was. Take on every change the refusal "
                 "states, repair the cart (take out any line that no longer qualifies, for example an item now out of stock), "
                 "check the new pick's stock and price again, then place the order again."]
        refusal = observation.get("refusal")
        said = refusal if isinstance(refusal, str) else _compact(refusal)
        if _PRICE_WORDS.search(said) and not _GONE.search(said):
            notes.append("Nothing in this refusal says the item has gone: if it only reports a new price (or some other change "
                         "the item still satisfies), the item can still be bought. Check its current price; while price times "
                         "quantity stays inside the budget and every requirement still holds, order it again at once. A price "
                         "change is never a reason to decline.")
        return notes

    def _g11_note_change(self, name: str, observation: dict[str, Any], declined: bool) -> list[str]:
        lines: list[str] = []
        for value in observation.values():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                _sentences(value, lines)
        if not lines:
            return []
        return ["The shopper sent a change along with this result; act on it: " + " | ".join(lines[:4])]

    def _g11_note_dict(self, name: str, observation: dict[str, Any], declined: bool) -> list[str]:
        _collect_ids(observation, self.seen_ids)
        notes = self._g11_note_used_up(name, observation, declined)
        branches = (
            (lambda: "refusal" in observation and declined, self._g11_note_declined),
            (lambda: "refusal" in observation, self._g11_note_refused),
            (lambda: len(observation) >= 2 and name != self.roles.get("message"), self._g11_note_change),
        )
        handler = next((h for applies, h in branches if applies()), None)
        return notes + (handler(name, observation, declined) if handler else [])

    def _g11_note_list(self, name: str, observation: list[Any], declined: bool) -> list[str]:
        _collect_ids(observation, self.seen_ids)
        return []

    def note_observation(self, name: str, observation: Any, declined: bool = False) -> str:
        table = ((dict, self._g11_note_dict), (list, self._g11_note_list))
        handler = next((h for shape, h in table if isinstance(observation, shape)), None)
        return " ".join(handler(name, observation, declined)) if handler else ""

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        """Tool calls as a pipeline: each call is a small record that passes the vetting stages in order (the first
        stage that settles it gives its reply), the batch's finish call then meets the finish gates, the rest runs
        one environment request at a time, and each result is booked and turned into the model's reply text."""
        ctx = {
            "replies": [],
            "allowed": {t["function"]["name"] for t in self.tools_for_turn()},
            "finishing": {self.roles.get("order"), self.roles.get("decline")} - {None},
            "message": self.roles.get("message"),
            "cart": self.roles.get("cart"),
            "queued_messages": 0,
            "planned": planned,
            "count": len(calls),
        }
        stages = (self._g12_known_tool, self._g12_offered_now, self._g12_decode_args, self._g12_schema_ok,
                  self._g12_finish_alone, self._g12_message_asks, self._g12_message_fresh, self._g12_message_budget,
                  self._g12_message_kept, self._g12_message_topic, self._g12_search_repeat, self._g12_call_repeat,
                  self._g12_cart_verdict, self._g12_cart_ids, self._g12_cart_units, self._g12_compare_split)
        queue: list[tuple[str, str, dict[str, Any]]] = []
        for call in calls:
            function = call.get("function") or {}
            record = {"id": call["id"], "name": function.get("name"), "raw": function.get("arguments"), "args": None}
            verdict = next((said for said in (stage(record, ctx) for stage in stages) if said is not None), None)
            if verdict is not None:
                ctx["replies"].append((record["id"], verdict))
                continue
            ctx["queued_messages"] += record["name"] == ctx["message"]
            queue.append((record["id"], record["name"], record["args"]))
        if queue and queue[0][1] in ctx["finishing"]:
            if self._g12_finish_gates(queue, ctx):
                return ctx["replies"]
        elif queue and ctx["finishing"]:
            room = max(0, self.steps_left() - 1)
            if self.steps_left() - 1 < len(queue):
                ctx["replies"].extend((cid, "Skipped, no step spent: just one step remains and it is reserved for the order (or the decline).")
                                      for cid, _, _ in queue[room:])
                queue = queue[:room]
        ctx["view_notes"] = {}
        for entry in queue:
            self._g12_execute(entry, ctx)
        position = {c["id"]: i for i, c in enumerate(calls)}
        return sorted(ctx["replies"], key=lambda pair: position.get(pair[0], 0))

    # ---- vetting stages: each returns the refusal text that settles the call, or None to pass it on ----

    def _g12_text_of(self, args: dict[str, Any]) -> str:
        return " ".join(str(v) for v in args.values() if isinstance(v, str))

    def _g12_known_tool(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] in self.schemas:
            return None
        return f"Skipped, no step spent: there is no tool called '{record['name']}'. The tools you can call: {', '.join(sorted(ctx['allowed']))}."

    def _g12_offered_now(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if ctx["planned"] or record["name"] in ctx["allowed"]:
            return None
        if record["name"] == ctx["message"] and self.planner_on:
            why = "shopper questions are drafted and sent automatically once a few listings have been opened in detail"
        else:
            why = "the remaining steps are too few for it; right now the usable tools are " + ", ".join(sorted(ctx["allowed"]))
        return f"Skipped, no step spent: {why}."

    def _g12_decode_args(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        raw = record["raw"]
        if isinstance(raw, dict):
            record["args"] = raw
            return None
        decoded = _parse_json_object(raw or "{}") if (raw or "").strip() else {}
        if decoded is None:
            return "Skipped, no step spent: the arguments could not be read as one JSON object."
        record["args"] = decoded
        return None

    def _g12_schema_ok(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        record["args"], problems = _check(record["args"], self.schemas[record["name"]], "args")
        return ("Skipped, no step spent: the arguments break the tool's schema: " + "; ".join(problems[:6])) if problems else None

    def _g12_finish_alone(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] in ctx["finishing"] and ctx["count"] > 1:
            return "Skipped, no step spent: send the order or the decline as the only call of a reply, once the other calls' results are back."
        return None

    def _g12_message_asks(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if (record["name"] == ctx["message"] and not ctx["planned"] and "?" not in self._g12_text_of(record["args"])
                and not self.finish_reserved):
            return ("Skipped, no step spent: the shopper channel is for questions only, and this message contains none. Findings, picks "
                    "and plans are never reported to the shopper; the finishing call says all of that.")
        return None

    def _g12_message_fresh(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] == ctx["message"] and self._g12_text_of(record["args"]).strip().lower() in self.questions_sent:
            return "Skipped, no step spent: you already sent these exact words and the reply is further up. Ask something new, or move on."
        return None

    def _g12_message_budget(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if (record["name"] == ctx["message"] and self.message_budget is not None
                and self.messages_used + ctx["queued_messages"] >= self.message_budget):
            return "Skipped, no step spent: no messages remain in the budget. Stop writing to the shopper."
        return None

    def _g12_message_kept(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if (record["name"] == ctx["message"] and not ctx["planned"] and self.kept_messages()
                and self.messages_used + ctx["queued_messages"] >= self.question_limit()):
            return ("Skipped, no step spent: the remaining messages are held back for the request's closing rules, and those run "
                    "automatically when the order tool is called. Settle the final cart and call the order tool.")
        return None

    def _g12_message_topic(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] != ctx["message"] or ctx["planned"]:
            return None
        wording = self._g12_text_of(record["args"])
        prop = _question_property(wording)
        if prop and "?" in wording and self.asked_before(_question_product(wording), prop):
            return ("Skipped, no step spent: that property was already asked about (the reply is further up, or the shopper did not "
                    "take it up), so asking it again would only burn a message. Pick another listing property, or move on.")
        if self.budget_now is not None and re.search(r"\b(spend|budget|afford|how much|price limit)\b", wording, re.I):
            return ("Skipped, no step spent: the price limit is already given in the request, and money is not a topic for the "
                    "shopper. Ask about a listing property, or move on.")
        return None

    def _g12_search_repeat(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        name = record["name"]
        record["signature"] = name + _compact(record["args"])
        if name != self.roles.get("search"):
            return None
        record["args"] = self.search_mode(record["args"])
        if ctx["planned"]:
            return None
        stamp = f"{self.ledger_version}|{self.refusals}|{self.budget_now}"
        if self.searches_done.get(record["signature"]) == stamp:
            return ("Skipped, no step spent: an identical query was issued earlier with nothing different now, so its rows stand as "
                    "shown. Vary the wording, add a filter, move to a later page, or inspect the hits it returned.")
        self.searches_done[record["signature"]] = stamp
        return None

    def _g12_call_repeat(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        exempt = ctx["finishing"] | {ctx["message"], ctx["cart"]}
        if record["name"] not in exempt and record["signature"] in self.recent:
            return "Skipped, no step spent: this identical call was just made and nothing has changed; its result is further up. Alter the arguments or take another action."
        return None

    def _g12_cart_verdict(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        # several products: each line is checked once, at the order
        if record["name"] != ctx["cart"] or self.steps_left() <= 3 or self.items() or self.forcing:
            return None
        args = record["args"]
        spec = self.schemas[record["name"]].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(spec.get(k), dict) and "enum" in spec[k]})
        if self.add_signature is not None and mode != self.add_signature:
            return None
        target = next((v for k, v in args.items()
                       if isinstance(v, str) and "enum" not in (spec.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if target is None and _listed_ids(args):
            target = _listed_ids(args)[-1]
        mark = f"{target}|{self.ledger_version}"
        if not target or mark in self.cart_checked:
            return None
        self.cart_checked.add(mark)
        judged = self.verify_item(target, "ADD TO THE CART", _line_of(args, target))
        return f"Skipped, no step spent. {judged} Should you be sure it fits, repeat the identical call." if judged else None

    def _g12_cart_ids(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] != ctx["cart"] or not self.seen_ids:
            return None
        spec = self.schemas[record["name"]].get("properties") or {}
        titles = any(isinstance(v, str) and "enum" not in (spec.get(k) or {})
                     and (" " in v.strip() or (len(v) > 3 and not _ID_HINT.match(v))) and v not in self.seen_ids
                     for k, v in record["args"].items())
        return "Skipped, no step spent: give the cart tool the listing id shown in the results, not the listing's title. Resend with the id." if titles else None

    def _g12_cart_units(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        if record["name"] != ctx["cart"]:
            return None
        warning = self.quantity_warning(record["name"], record["args"])
        if not warning or record["signature"] in self.qty_warned:
            return None
        self.qty_warned.add(record["signature"])
        return f"Skipped, no step spent: {warning} Should you be sure, repeat the identical call."

    def _g12_compare_split(self, record: dict[str, Any], ctx: dict[str, Any]) -> str | None:
        side = self.compare_tool()
        if not side or record["name"] != side[0] or not isinstance(record["args"].get(side[1]), list):
            return None
        wanted = list(dict.fromkeys(str(i) for i in record["args"][side[1]]))
        if len(wanted) > self.compare_cap:
            record["args"] = {**record["args"], side[1]: wanted[: self.compare_cap]}
            self.compare_left = wanted[self.compare_cap:]
        return None

    # ---- finish gates: the batch's lone order/decline call; True means a gate answered and the batch ends ----

    def _g12_finish_gates(self, queue: list[tuple[str, str, dict[str, Any]]], ctx: dict[str, Any]) -> bool:
        call_id, name, args = queue[0]
        fin = {"id": call_id, "name": name, "args": args, "declining": self.is_decline(name, args),
               "closest": bool(self.plan.get("closest_if_none")), "named": bool(self.plan.get("closest_named")),
               "planned": ctx["planned"]}
        self._g12_decline_to_order(fin, queue)
        gates = (self._g12_gate_research, self._g12_gate_cheaper, self._g12_gate_pointers, self._g12_gate_lines,
                 self._g12_gate_cart_check, self._g12_gate_decline_pages, self._g12_gate_sweep, self._g12_gate_depth,
                 self._g12_gate_finish_rules, self._g12_gate_commit_empty, self._g12_gate_commit_items, self._g12_gate_decline)
        for gate in gates:
            said = gate(fin)
            if said is not None:
                ctx["replies"].append((call_id, said))
                return True
        if fin["declining"]:
            queue[0] = (call_id, fin["name"], self.clean_decline(fin["args"], fin["name"]))
        return False

    def _g12_ordering(self, fin: dict[str, Any]) -> bool:
        return not fin["declining"] and fin["name"] == self.roles.get("order")

    def _g12_decline_to_order(self, fin: dict[str, Any], queue: list[tuple[str, str, dict[str, Any]]]) -> None:
        # a line per product in the cart, and the request wants the closest ordered (or both audits named fits):
        # a decline is right in 1-2% of tasks at most, so the cart is ordered, through the usual checks
        if not (fin["declining"] and not fin["planned"] and self.roles.get("order") and self.cart_ids and not self.products_unheld()):
            return
        if not (fin["closest"] or self.decline_checks >= 2) or self.decline_ok or (fin["named"] and not self.audit_clean):
            return
        fin["name"] = self.roles["order"]
        fin["args"] = self.neutral_args(self.roles["order"], "Order the cart.")
        fin["declining"] = False
        queue[0] = (fin["id"], fin["name"], fin["args"])

    def _g12_gate_research(self, fin: dict[str, Any]) -> str | None:
        # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
        if not (self._g12_ordering(fin) and self.plan.get("priority") == "cheapest" and not self.forcing and self.cost < COST_HARD):
            return None
        if not (self.refused_at > self.searched_at and self.research_checks < 1 and self.steps_left() > 10
                and self.time_left() > WRAP_UP_SECONDS + 20):
            return None
        self.research_checks += 1
        return ("Held back for now (no step spent). Checkout bounced last time, and this shopper wants the lowest-priced match. A "
                "fallback chosen before stays valid only when no lower-priced match exists today, so query the catalogue once "
                "more first: requirement terms together with the category, a ceiling slightly over the bounced item's price if "
                "a ceiling filter exists, plus later pages. Inspect the lower-priced hits, keep the least expensive one meeting "
                "all conditions in the basket, then check out.")

    def _g12_gate_cheaper(self, fin: dict[str, Any]) -> str | None:
        if not (self._g12_ordering(fin) and self.cheaper_checks < 2 and self.steps_left() > 10
                and self.time_left() > WRAP_UP_SECONDS + 20 and not self.forcing and self.cost < COST_HARD):
            return None
        cheaper = self.unchecked_cheaper()
        if not cheaper:
            return None
        self.cheaper_checks += 1
        return ("Held back for now (no step spent). The request wants the cheapest item that fits, yet these lower-priced listings "
                "from your results were never opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper)
                + ". Inspect those (batch them in one reply); when one meets all conditions, the least expensive of them replaces "
                "your current basket line; otherwise just check out again.")

    def _g12_gate_pointers(self, fin: dict[str, Any]) -> str | None:
        if (self._g12_ordering(fin) and self.pointers and self.steps_left() > 8 and self.time_left() > WRAP_UP_SECONDS + 40
                and self.cost < COST_HARD and not self.pointers_stalled and not self.forcing):
            return ("Held back for now (no step spent). One requirement refers to another listing's page that nobody has read "
                    "yet; the resolver reads it next, and then the order goes ahead.")
        return None

    def _g12_gate_lines(self, fin: dict[str, Any]) -> str | None:
        if not self._g12_ordering(fin):
            return None
        if self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS and not self.forcing:
            gate = self.line_gate() or self.dependency_gate() or self.process_gate()
            return ("Held back for now (no step spent). " + gate) if gate else None
        if fin["planned"] or self.forcing or self.steps_left() <= 3 or self.time_left() <= 30:
            return None
        unpaged = self.unpaged_lines()
        mark = "late|" + str(self.cart_changed_at)
        if not unpaged or mark in self.line_blocks:
            return None
        self.line_blocks.add(mark)
        self.page_next = unpaged
        return (f"Held back for now (no step spent). Item {', '.join(unpaged[:4])} lacks a full detail view so far (attribute "
                "summaries and result rows don't qualify); that view is fetched automatically next. Retry checkout afterwards.")

    def _g12_gate_cart_check(self, fin: dict[str, Any]) -> str | None:
        if not (self._g12_ordering(fin) and not self.order_checked and self.steps_left() > 2
                and self.time_left() > WRAP_UP_SECONDS and not self.forcing):
            return None
        self.order_checked = True
        verdict = self.verify_cart()
        if verdict:
            return "Held back for now (no step spent). " + verdict
        return ("Held back for now (no step spent). Confirm these before checkout: (1) basket = exactly what the task needs "
                "(typically a single line, your pick at its required count), no extras; (2) after the latest basket edit you "
                "re-fetched the pick's availability and price via a tool displaying them, it is available, and its price "
                "multiplied by count stays under today's limit; (3) the pick's own attributes satisfy each written condition, "
                "each preference the shopper disclosed, each announced update; (4) if lowest price or top quality was wanted, "
                "no superior match turned up anywhere you looked; (5) all procedural instructions in the task are done. "
                "Verify anything unconfirmed now; once everything holds, retry checkout.")

    def _g12_gate_decline_pages(self, fin: dict[str, Any]) -> str | None:
        if not (fin["declining"] and not fin["closest"] and self.page_checks < 2 and self.steps_left() > 8
                and self.time_left() > WRAP_UP_SECONDS + 30 and self.cost < COST_HARD):
            return None
        unread = self.unread_searches()
        if not unread:
            return None
        self.page_checks += 1
        return ("Held back for now (no step spent). Prior to giving up, exhaust these queries, whose hit lists were cut short: "
                + "; ".join(unread) + ". " + self.order_note() + ", meaning an unseen hit is as likely a match as any seen one. "
                "Walk their remaining pages, inspect (individually or in a comparison) every candidate of the wanted type, and "
                "purchase the strongest match; give up only if each candidate plainly breaks a condition, a disclosed "
                "preference, the spending limit or availability.")

    def _g12_gate_sweep(self, fin: dict[str, Any]) -> str | None:
        # once, and once more after a change of the requirements or the budget (a dropped requirement or a raised
        # budget widens the fitting set, so a cheaper fit may now exist)
        if not (self._g12_ordering(fin) and self.plan.get("priority") == "cheapest" and not self.forcing):
            return None
        again = self.sweep_checks < 1 or (self.sweep_checks < 2 and self.sweep_stamp != self.change_stamp() and self.steps_left() > 6)
        if not (again and self.cost < COST_HARD and self.time_left() > WRAP_UP_SECONDS + 30 and self.sweep_cheapest() != ""):
            return None
        self.sweep_checks += 1
        self.sweep_stamp = self.change_stamp()
        self.cheaper_checks += 1
        cheaper = self.unchecked_cheaper()
        if not cheaper:
            return None
        return ("Held back for now (no step spent). Every page of the queries that produced your pick has been read; still, these "
                "same-type items undercut it, are available and affordable, and have no detail view yet: "
                + ", ".join(f"{i} ({p:g})" for i, p in cheaper) + ". Inspect those (batch them in one call); when one meets "
                "all conditions, the least expensive of them replaces your current basket line; otherwise just check out again.")

    def _g12_gate_depth(self, fin: dict[str, Any]) -> str | None:
        if not (self._g12_ordering(fin) and self.plan.get("priority") in ("cheapest", "best") and self.sweep_checks < 1
                and self.steps_left() > 8 and self.time_left() > WRAP_UP_SECONDS + 30 and self.cost < COST_HARD):
            return None
        unread = self.unread_searches() or self.ranked_unread()
        if not unread:
            return None
        self.sweep_checks += 1
        return ("Held back for now (no step spent). Target per the task: the " + str(self.plan.get("priority")) + " match; "
                "these queries stopped too early though: " + "; ".join(unread) + ". " + self.order_note()
                + ", hence a superior or lower-priced match might be hiding further down. Continue paging, inspect promising "
                "candidates, and only then check out the correct one.")

    def _g12_gate_finish_rules(self, fin: dict[str, Any]) -> str | None:
        if not (self._g12_ordering(fin) and not fin["planned"] and not self.forcing and self.finish_pending()
                and self.steps_left() >= 2 and self.time_left() > 22):
            return None
        self.finish_next = dict(fin["args"])
        return ("Held back for now (no step spent). The request's closing rules are being checked, and whatever is left of them is "
                "done for you; if nothing changes, the order goes out right after. Leave the cart as it is from now on.")

    def _g12_blocked_by_audits(self, fin: dict[str, Any]) -> bool:
        return fin["named"] and self.decline_checks >= 2 and not self.audit_clean

    def _g12_gate_commit_empty(self, fin: dict[str, Any]) -> str | None:
        # (late in steps, too late in time for the decline audits, or declining again after both audits named
        # possible fits: a decline is right in 1-2% of tasks, the closest fit far more often)
        if not fin["declining"] or fin["planned"]:
            return None
        late = fin["closest"] or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS + 30 or self.decline_checks >= 2
        if not (late and not self.cart_ids and not self.items() and not self.decline_ok and not self._g12_blocked_by_audits(fin)):
            return None
        if not (self.may_commit() and self.roles.get("cart") and self.steps_left() >= 2):
            return None
        self.commit_next = True
        return "Skipped, no step spent: the nearest fit you opened goes into the cart, and the order follows."

    def _g12_gate_commit_items(self, fin: dict[str, Any]) -> str | None:
        if not (fin["declining"] and not fin["planned"] and self.items() and not self.order_attempts and not self.decline_ok):
            return None
        if self._g12_blocked_by_audits(fin):
            return None
        if not (fin["closest"] or self.decline_checks >= 2 or self.time_left() < WRAP_UP_SECONDS + 30):
            return None
        if getattr(self, "committed", False) or not self.roles.get("cart") or self.steps_left() < 3 or self.time_left() <= 30:
            return None
        self.commit_next = True
        return "Skipped, no step spent: for each product the nearest fit you opened goes into the cart, and the order follows."

    def _g12_gate_decline(self, fin: dict[str, Any]) -> str | None:
        # A decline is checked twice: it is rarely the right finish, so each attempt first gets an
        # independent look for listings that meet the requirements.
        if not fin["declining"]:
            return None
        if self.decline_checks < 2 and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS + 30 and not self.decline_ok:
            self.decline_checks += 1
            audit = self.audit_decline()
            if audit:
                return "Held back for now (no step spent). " + audit
            if self.decline_checks == 1 and not self.decline_ok:
                return ("Held back for now (no step spent). Giving up needs certainty: queries were varied (alternate terms, "
                        "filters, later pages), each candidate's attributes were inspected, and each breaks a written condition, a "
                        "disclosed preference, the spending limit or availability. Close synonyms or alternate spellings are "
                        "acceptable matches, while assumptions drawn from the backstory are no condition at all. Purchase any "
                        "match you have; only with none at all, invoke the no-purchase tool once more.")
        if fin["named"] and not self.named_checked and self.steps_left() > 1 and not self.near_miss_named(fin["args"]):
            # the request wants the near misses named when nothing fits: a pass that names none is sent back once
            self.named_checked = True
            return ("Held back for now (no step spent). This task defines its no-perfect-match outcome: state that, and identify "
                    "the runner-up item, i.e. whichever breaks the fewest conditions (unavailability or exceeding the limit each "
                    "count once), ties going to the lower price, within whatever price tolerance the task allows (one runner-up "
                    "per wanted product). Compose the closing text that way: no item satisfies all conditions, followed by each "
                    "runner-up's id and its shortfall, and invoke the no-purchase tool again. Should an item you found satisfy "
                    "all conditions after all, buy it.")
        return None

    # ---- execution: one action per environment request, in order (the interface asks for it) ----

    def _g12_execute(self, entry: tuple[str, str, dict[str, Any]], ctx: dict[str, Any]) -> None:
        call_id, name, args = entry
        over = "Skipped: the episode has ended or no steps remain."
        if self.finished or self.dead or self.steps_left() <= 0:
            ctx["replies"].append((call_id, over))
            return
        if name == self.roles.get("order") and not self.is_decline(name, args):
            seen_cart = self.view_before_order()
            if seen_cart:
                ctx["view_notes"][call_id] = seen_cart
            if self.finished or self.dead or self.steps_left() <= 0:
                ctx["replies"].append((call_id, over))
                return
        actions = [{"name": name, "args": args}]
        before = self.step
        reply = self.env_call(actions)
        self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
        if reply is None:
            ctx["replies"].append((call_id, "The call did not go through (network or server trouble). Try it again."))
            return
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        if not observations:
            ctx["replies"].append((call_id, "Skipped: the episode ended before this call ran."))
            return
        observation = observations[0]
        last = len(observations) == 1
        self._g12_book_counts(name, args, observation, ctx)
        self._g12_book_store(name, args, observation)
        if name == ctx["cart"]:
            self._g12_book_cart(name, args)
        self._g12_book_reach(name, args, observation, ctx)
        self._g12_book_cart_seen(name, observation, ctx["cart"])
        self.note_event(name, args, observation)
        notes = [self.note_observation(name, observation, self.is_decline(name, args))]
        if name == ctx["cart"] and self.cart_seen_at == self.step:
            notes.append(self.over_units())
        notes.extend(self._g12_compare_notes(name, args, observation))
        notes.extend(self._g12_search_notes(name, args, observation, ctx["planned"]))
        notes.append(self._g12_order_outcome(name, args, observation, reply))
        notes.append(self._g12_heard_meaning(name, args, observation, ctx["message"]))
        failed = isinstance(observation, dict) and bool(observation.get("error"))
        if (name == ctx["cart"] and not ctx["planned"] and not reply.get("finished") and not failed and last
                and len(self.held_lines()) >= len(self.items())):
            # the cart now holds the lines a tie rule joins: checked here, so the model can fix it before ordering
            notes.append(self.dependency_gate())
        text = _compact(_brief(observation, ROW_CHARS if name == self.roles.get("search") else PAGE_CHARS))
        if last and reply.get("finished"):
            text += " [the episode is finished]"
        if call_id in ctx["view_notes"]:
            text = ctx["view_notes"][call_id] + "\nThen the order call answered: " + text
        joined = " ".join(n for n in notes if n)
        if joined:
            text += "\n[note] " + joined
        ctx["replies"].append((call_id, text))

    # ---- observation bookkeeping ----

    def _g12_book_counts(self, name: str, args: dict[str, Any], observation: Any, ctx: dict[str, Any]) -> None:
        errored = isinstance(observation, dict) and observation.get("error")
        if name == ctx["message"]:
            self.messages_used += 1
            said = self._g12_text_of(args).strip().lower()
            if said and not errored:
                self.questions_sent.append(said)      # a message that failed may be sent again
        if name in ctx["finishing"]:
            self.finish_tries += 1
        if name == self.roles.get("order") and not self.is_decline(name, args):
            self.order_attempts += 1
        if name == ctx["cart"]:
            self.cart_used = True
            self.order_checked = False
        clears = name in ctx["finishing"] or name in (ctx["cart"], ctx["message"])
        self.recent = [] if clears else (self.recent + [name + _compact(args)])[-2:]
        self.records.append(observation)
        if not errored:
            self.tool_calls.append((self.step, name))

    def _g12_book_store(self, name: str, args: dict[str, Any], observation: Any) -> None:
        reads = name in (self.roles.get("details"), self.roles.get("view")) or name == (self.compare_tool() or ("",))[0]
        self.store.absorb(observation, self.step, opened=reads)
        if name == self.roles.get("view") and name != self.roles.get("details") and not self.view_facts:
            self.view_facts = any(isinstance(v, str) and v in self.store.listings and self.store.listings[v].facts for v in args.values())

    def _g12_book_cart(self, name: str, args: dict[str, Any]) -> None:
        spec = self.schemas[name].get("properties") or {}
        mode = _compact({k: v for k, v in args.items() if isinstance(spec.get(k), dict) and "enum" in spec[k]})
        if self.add_signature is None:
            self.add_signature = mode
        listed = _listed_ids(args)
        if listed is not None:
            if mode != self.add_signature:
                self.cart_ids = [i for i in self.cart_ids if i not in listed]     # the same tool used to take lines out
            else:
                self._g12_take_basket(args, listed)
        asked_units = next((v for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)
                            and isinstance(spec.get(k), dict) and spec[k].get("type") == "integer" and "enum" not in spec[k]), None)
        singles = [] if listed is not None else [
            value for key, value in args.items()
            if isinstance(value, str) and "enum" not in (spec.get(key) if isinstance(spec.get(key), dict) else {})
            and value.strip() and " " not in value.strip()]
        for value in singles:
            if asked_units is not None:
                self.cart_qty[value] = asked_units
            self.cart_state[value] = mode
            kept = [i for i in self.cart_ids if i != value]
            self.cart_ids = kept + [value] if mode == self.add_signature and asked_units != 0 else kept
        self.cart_changed_at = self.step

    def _g12_take_basket(self, args: dict[str, Any], listed: list[str]) -> None:
        # a cart tool that takes the whole basket as a list of lines
        self.cart_ids = listed
        self.cart_lines = {i: _line_of(args, i) for i in listed}
        rows = [r for v in args.values() if isinstance(v, list) for r in v if isinstance(r, dict)]
        for row in rows:
            ident = next((v for v in row.values() if v in listed), None)
            units = next((v for v in row.values() if isinstance(v, int) and not isinstance(v, bool)), None)
            if ident and units is not None:
                self.cart_qty[ident] = units

    def _g12_book_reach(self, name: str, args: dict[str, Any], observation: Any, ctx: dict[str, Any]) -> None:
        planned = ctx["planned"]
        if name == self.roles.get("search"):
            self.searched_at = self.step
            self.searches_in_a_row += 1
            if not planned and _rows_with_ids(observation):
                sprops = (self.schemas.get(name) or {}).get("properties") or {}
                worded = [k for k, v in args.items() if isinstance(v, str) and v.strip() and isinstance(sprops.get(k), dict)
                          and "enum" not in sprops[k] and not (_ID_HINT.match(v.strip()) and any(c.isdigit() for c in v))]
                if len(worded) == 1:
                    self.key_votes[worded[0]] = self.key_votes.get(worded[0], 0) + 1
        if name == ctx["message"] and not planned:
            own = self._g12_text_of(args)
            if "?" in own:
                self.asked_props.append((_question_product(own), _question_property(own)))
        strings = [v for v in args.values() if isinstance(v, str)]
        if name in (self.roles.get("details"), self.roles.get("view")):
            self.opened.update(strings)
        if name == self.roles.get("details") and not (isinstance(observation, dict) and observation.get("error")):
            self.paged.update(strings)
            self.searches_in_a_row = 0
        side = self.compare_tool()
        if side and name == side[0] and isinstance(args.get(side[1]), list):
            shown = {i for i, _ in _rows_with_ids(observation)}
            hits = [str(i) for i in args[side[1]] if str(i) in shown]
            self.paged.update(hits)
            self.searches_in_a_row = 0
            self.opened.update(hits)

    def _g12_book_cart_seen(self, name: str, observation: Any, cart_tool: str | None) -> None:
        if name == self.roles.get("cart_view"):
            self.last_cart_view = observation
            self.view_called_at = self.step
        if name not in (cart_tool, self.roles.get("cart_view")) or (isinstance(observation, dict) and observation.get("error")):
            return
        lines_seen = _cart_lines(observation)
        if lines_seen is None:
            return
        self.cart_seen = [(i, n if n is not None else self.cart_qty.get(i, 1)) for i, n in lines_seen]
        self.cart_seen_at = self.step
        if name == cart_tool:
            self.cart_ids = [i for i, n in self.cart_seen if n > 0]
            self.cart_lines = {}

    # ---- reply notes ----

    def _g12_compare_notes(self, name: str, args: dict[str, Any], observation: Any) -> list[str]:
        side = self.compare_tool()
        if not side or name != side[0] or not isinstance(args.get(side[1]), list):
            return []
        sent_ids = args[side[1]]
        empty = not observation or (isinstance(observation, dict) and all(not v for v in observation.values()))
        # a bare error ({"issue_text": "E13"}: no listing in it) for many ids is the tool's cap, not a bad id
        bare = isinstance(observation, dict) and len(observation) <= 3 and not any(isinstance(v, (dict, list)) for v in observation.values())
        shown_ids = {i for i, _ in _rows_with_ids(observation)}
        unanswered = [str(i) for i in dict.fromkeys(sent_ids) if str(i) not in shown_ids and (str(i) in self.store.listings or str(i) in self.seen_ids)]
        said: list[str] = []
        if len(sent_ids) > 1 and (empty or bare) and not shown_ids:
            self.compare_cap = max(1, min(self.compare_cap, len(sent_ids) // 2))
            said.append(f"Comparison returned zero items for that batch of {len(sent_ids)}; treat all of them as still unviewed, "
                        f"since its per-request limit is lower. Batch size from now: {self.compare_cap} max.")
        elif len(sent_ids) > 1 and shown_ids and unanswered:
            # fewer rows than ids, no error: the tool answers only so many per call (known listings were left out)
            self.compare_cap = max(1, min(self.compare_cap, len(shown_ids)))
            said.append(f"Only {len(shown_ids)} of the {len(set(sent_ids))} requested listings came back from the side-by-side tool; "
                        f"these remain UNOPENED: {', '.join(unanswered[:8])}. Keep it to {self.compare_cap} ids or fewer per call.")
        elif getattr(self, "compare_left", None):
            said.append(f"The side-by-side tool takes no more than {self.compare_cap} listings per call, so only the first "
                        f"{len(sent_ids)} went out; these remain UNOPENED: {', '.join(self.compare_left)}. Send them in a further call if needed.")
        self.compare_left = []
        return said

    def _g12_search_notes(self, name: str, args: dict[str, Any], observation: Any, planned: bool) -> list[str]:
        if name != self.roles.get("search") or planned:
            return []
        said = [self.track_search(args, observation)]
        if not self.probed and not self.probe_args:
            self.probe_args = dict(args)
        if self.searches_in_a_row >= 3:
            side = self.compare_tool()
            hint = f" ({side[0]} shows {self.compare_cap} pages for one step)" if side and self.compare_cap >= 2 else ""
            said.append("Several searches back to back without opening a listing: rows leave out a listing's properties, which only "
                        "its page shows. Open the most promising rows now" + hint + " before the next search.")
        return said

    def _g12_order_outcome(self, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any]) -> str | None:
        refused = isinstance(observation, dict) and "refusal" in observation
        if name != self.roles.get("order") or self.is_decline(name, args) or reply.get("finished") or refused:
            return None
        if isinstance(observation, dict) and _pending_step(observation):
            return ("Checkout is incomplete: a follow-up action is expected (a token was issued, or confirmation was requested). "
                    "Perform it immediately via the checkout-completing tool; no purchase exists before the session closes.")
        # the order did not end the episode: refused (in any wording), failed, or waiting for another step
        self.refusals += 1
        self.refused_at = self.step
        self.refused_ids.update(self.cart_ids)
        self.order_checked = False
        return ("SESSION STILL OPEN AFTER CHECKOUT, HENCE NO PURCHASE HAS HAPPENED, regardless of the wording above (real success "
                "closes the session). When a follow-up action is requested (e.g. confirming with a supplied token or code), "
                "perform it right away. When rejected or errored, study the cause, adopt each stated update, correct the basket, "
                "re-fetch availability and cost of your pick, then retry; a changed price alone never disqualifies an item while "
                "cost multiplied by count remains affordable.")

    def _g12_heard_meaning(self, name: str, args: dict[str, Any], observation: Any, message_tool: str | None) -> str | None:
        heard: list[str] = []
        question = None
        if name == message_tool:
            question = self._g12_text_of(args)
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                if isinstance(observation, dict):
                    _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                if not heard and _has_text(observation):
                    # a short or structured answer ("Navy.", {"value": "navy"}) is still the shopper's answer
                    heard = [_compact(observation)[:600]]
        elif isinstance(observation, dict):
            wrapper = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
            refusal = "refusal" in observation
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    _sentences(value, heard)
                    continue
                if not isinstance(value, str):
                    continue
                words = len(value.split())
                if (words >= 4 and refusal) or (wrapper and words >= 6 and (_FIRST_PERSON.search(value) or name == self.roles.get("order"))):
                    heard.append(value)     # a line wrapped around the real result, as a plain string
        return self.interpret(heard, question) if heard else None

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self._g13_open_episode()
        tally = {"idle": 0, "failures": 0, "counter": 0}
        turns = 0
        while turns < MAX_TURNS and self._g13_turn(tally):
            turns += 1
        self._g13_close_episode()
        return self.trace

    # ---- phases -------------------------------------------------------------------------------------------------

    def _g13_open_episode(self) -> None:
        """Setup, the plan trace, and the first user message (request, opening page, brief, status)."""
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        sections = ["SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")]
        if opening is not None:
            sections.append("OPENING OBSERVATION:\n" + _compact(opening))
        brief = self.brief()
        if brief:
            sections.append(brief)
        sections.append(self.status())
        if opening is not None:
            self.records.append(opening)
            self.store.absorb(opening, 0)
            _collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(sections)}]

    def _g13_close_episode(self) -> None:
        if not (self.finished or self.dead):
            self.force_finish()
        elapsed = time.monotonic() - self.started
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(elapsed, 1)})

    def _g13_halt(self) -> bool:
        self.force_finish()
        return False

    def _g13_turn(self, tally: dict[str, int]) -> bool:
        """One pass of the loop. True: go on to the next turn; False: leave the loop."""
        if self.finished or self.dead or self.steps_left() <= 0:
            return False
        if self.time_left() < FORCE_SECONDS:
            return self._g13_halt()
        queued = self._g13_queued_work()
        if queued is not None:
            return queued
        message = self.infer(self.window(), self.offered_tools(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
        if message is None:
            return self._g13_model_silent(tally)
        tally["failures"] = 0
        calls = self._g13_tool_calls(message, tally)
        content = message.get("content") or ""
        content = content if isinstance(content, str) else _compact(content)
        gated = self._g13_planner_gate(calls, content)
        if gated is not None:
            return gated
        if calls:
            self._g13_dispatch(calls, content, tally)
            return True
        return self._g13_idle_reply(tally)

    # ---- work queued before the model is asked (None: nothing done here, fall through) -----------------------------

    def _g13_queued_work(self) -> bool | None:
        stages = (self._g13_pending_reply, self._g13_cost_cap, self._g13_finish_queue, self._g13_list_queues,
                  self._g13_commit_queue, self._g13_pointer_pass, self._g13_cheapest_sweep, self._g13_probe_queue,
                  self._g13_planned_question, self._g13_last_chance)
        for stage in stages:
            verdict = stage()
            if verdict is not None:
                return verdict
        return None

    def _g13_pending_reply(self) -> bool | None:
        text = self.pending_reply
        if not text:
            return None
        self.pending_reply = ""
        return True if self.send_reply(text) else None

    def _g13_cost_cap(self) -> bool | None:
        # far beyond the episode's spending limit: finish at once (the closest fit when the cart is empty)
        return self._g13_halt() if self.cost > COST_HARD + 0.02 else None

    def _g13_finish_queue(self) -> bool | None:
        order_args = self.finish_next
        if order_args is None:
            return None
        self.finish_next = None
        if self.run_finish(order_args, final=self.time_left() < 45):
            return True
        # the finish had nothing to do (or could not be done): the order is let through next time
        self.finish_clear.add(self.finish_state())
        return None

    def _g13_list_queues(self) -> bool | None:
        if self.page_next:
            ids, self.page_next = self.page_next, []
            self.open_pages(ids)
            return True
        if not self.dep_next:
            return None
        jobs, self.dep_next = self.dep_next, []
        self.dependency_search(jobs)
        return True

    def _g13_commit_queue(self) -> bool | None:
        if not self.commit_next:
            return None
        self.commit_next = False
        return True if self.last_commit() else None

    def _g13_pointer_pass(self) -> bool | None:
        if not self.pointers or self.step < 1 or self.steps_left() <= 8:
            return None
        if self.cost >= COST_HARD or self.pointers_stalled:
            return None
        mark = (len(self.pointers), self.step)
        self.resolve_pointers()
        if mark == (len(self.pointers), self.step):
            self.pointers_stalled = True      # no progress was possible: do not come back to it
        return True

    def _g13_sweep_due(self) -> bool:
        if self.plan.get("priority") != "cheapest" or self.order_attempts or self.sweep_checks >= 1:
            return False
        if self.items() and not self.cart_ids:
            return False
        left = self.steps_left()
        if left <= self.order_due() + 2:
            return False
        ceiling = self.max_steps if self.cart_ids else self.order_due() + 3 + SWEEP_PAGES
        if left > ceiling or self.cost >= COST_HARD:
            return False
        return self.time_left() > WRAP_UP_SECONDS + 40 and not self.pointers

    def _g13_budget_number(self) -> Any:
        budget = self.budget_now
        numeric = isinstance(budget, (int, float)) and not isinstance(budget, bool)
        return budget if numeric else None

    def _g13_wanted_count(self) -> Any:
        quantity = self.plan.get("quantity")
        return quantity if isinstance(quantity, int) and quantity > 0 else 1

    def _g13_cheapest_sweep(self) -> bool | None:
        # just before the steps kept for the order (or once a pick sits in the cart), the pick's searches are read to the end
        if not self._g13_sweep_due():
            return None
        pick = self.sweep_cheapest(self.order_due() + 1)
        if not pick:
            return None
        self.sweep_checks += 1
        self.sweep_stamp = self.change_stamp()
        if self.cart_ids:
            cheaper = self.unchecked_cheaper()
        else:
            cheaper = self.store.cheaper(pick, self._g13_budget_number(), self._g13_wanted_count(), self.opened, self.cart_ids)
        head = "[sweep] Every search behind your current pick has now been paged through to its last row. "
        if cheaper:
            listed = ", ".join(f"{i} ({p:g})" for i, p in cheaper)
            body = (f"Never-opened listings of that kind, in stock, inside the budget and priced below your pick: {listed}. "
                    "Open them together in one call; if any satisfies every requirement, switch the cart to the cheapest one that does.")
        else:
            body = "No cheaper listing of that kind showed up."
        self.messages.append({"role": "user", "content": head + body + "\n" + self.status()})
        return True

    def _g13_probe_queue(self) -> bool | None:
        # how the store reads a query is learned once, straight after the model's first keyword search
        probe = self.probe_args
        if not probe:
            return None
        self.probe_args = None
        before = self.step
        self.probe_search(probe)
        return True if self.step != before else None

    def _g13_planned_question(self) -> bool | None:
        return True if self.planner_on and self.ask_planned() else None

    def _g13_cart_untouched(self) -> bool:
        return not self.cart_used and not self.order_attempts

    def _g13_last_chance(self) -> bool | None:
        # almost out of steps with an empty cart: the closest opened fit (one line per product) beats ordering nothing
        if not self.items():
            if self.steps_left() <= 3 and self._g13_cart_untouched() and self.last_commit():
                return True
        products = self.items()
        if not products or self.steps_left() > len(products) + 3 or not self._g13_cart_untouched():
            return None
        return True if self.last_commit() else None

    # ---- the model's turn ---------------------------------------------------------------------------------------

    def _g13_model_silent(self, tally: dict[str, int]) -> bool:
        tally["failures"] += 1
        if tally["failures"] >= 4 or self.time_left() < WRAP_UP_SECONDS + 20:
            return self._g13_halt()
        time.sleep(2.0 * tally["failures"])
        return True

    def _g13_tool_calls(self, message: dict[str, Any], tally: dict[str, int]) -> list[dict[str, Any]]:
        calls = []
        for raw in filter(lambda item: isinstance(item, dict), message.get("tool_calls") or []):
            tally["counter"] += 1
            spec = raw.get("function") or {}
            arguments = spec.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append({"id": raw.get("id") or f"call_{tally['counter']}", "type": "function",
                          "function": {"name": spec.get("name"), "arguments": arguments}})
        return calls

    def _g13_planner_gate(self, calls: list[dict[str, Any]], content: str) -> bool | None:
        """A cart/order/decline call while the question planner is on: bounce it once when too few listings were opened,
        else let a planned question go first."""
        committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
        if not self.planner_on or all(c["function"]["name"] not in committing for c in calls):
            return None
        ready = (not self.early_bounced and self.property_menu()[1] < 3 and self.steps_left() > 12
                 and self.time_left() > WRAP_UP_SECONDS + 60)
        if ready:
            self._g13_bounce(calls, content)
            return True
        return True if self.ask_planned(force=True) else None

    def _g13_bounce(self, calls: list[dict[str, Any]], content: str) -> None:
        self.early_bounced = True
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        note = ("Not run (no step used): it is too early to choose. First look inside a few more listings of the requested type, "
                "two or three in a single reply and ideally unlike the ones viewed so far, since what they reveal decides which questions the shopper is asked.")
        final = len(calls) - 1
        self.messages.extend({"role": "tool", "tool_call_id": c["id"], "content": note + ("\n" + self.status() if index == final else "")}
                             for index, c in enumerate(calls))

    def _g13_idle_reply(self, tally: dict[str, int]) -> bool:
        # a reply with no call tells nothing about the store (it may even claim the tools are gone), so it is not kept
        tally["idle"] += 1
        self.messages += [{"role": "assistant", "content": "(no tool call)"},
                          {"role": "user", "content": "Each reply needs at least one call to the tools provided, and every one of them is working. Go on.\n" + self.status()}]
        if tally["idle"] >= 5 or self.time_left() < WRAP_UP_SECONDS:
            return self._g13_halt()
        return True

    def _g13_dispatch(self, calls: list[dict[str, Any]], content: str, tally: dict[str, int]) -> None:
        tally["idle"] = 0
        self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
        self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
        step_before = self.step
        self._g13_post_replies(self.run_calls(calls), every=False)
        self._g13_due_order(self.step == step_before and not self.finished)

    def _g13_post_replies(self, replies: list[tuple[str, str]], every: bool) -> None:
        replies = list(replies)
        final = len(replies) - 1
        for index, (call_id, text) in enumerate(replies):
            if every or index == final:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def _g13_due_order(self, idle_turn: bool) -> None:
        """Turns where nothing ran: inside the steps kept for the order, the third one in a row sends the order call itself
        (through every check, exactly as if the model had sent it)."""
        self.blocked_turns = self.blocked_turns + 1 if idle_turn else 0
        if self.blocked_turns < 3 or not self.in_reserve() or not self.cart_ids:
            return
        order = self.roles.get("order")
        if not order or self.finish_next is not None:
            return
        self.blocked_turns = 0
        call = {"id": f"due_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": "We are now inside the steps set aside for ordering, so I am placing the order for the cart.", "tool_calls": [call]})
        self._g13_post_replies(self.run_calls([call]), every=True)


def _g13_crash_entry(error: BaseException) -> dict[str, Any]:
    return {"kind": "crash", "error": f"{type(error).__name__}: {error}"[:500]}


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append(_g13_crash_entry(exc))
    try:
        if not (episode.finished or episode.dead):
            episode.force_finish()
    except Exception as again:
        episode.trace.append(_g13_crash_entry(again))
    return episode.trace


__all__ = ["agent_main"]

