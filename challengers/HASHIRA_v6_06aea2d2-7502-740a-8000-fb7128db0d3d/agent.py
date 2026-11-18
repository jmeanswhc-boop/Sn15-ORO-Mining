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

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is what the request says for that: when it wants the closest listing put forward, order the one that misses the fewest requirements; otherwise decline.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If the store has a tool that shows the full details of several listings in one call, use it instead of opening them one by one: it saves steps.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- When one tool returns a listing's page and another returns only part of it (for example only its property list), only the page counts as having opened that listing, and the grader checks it: every listing you buy and every page on a reference chain must be opened with the page tool (or the side-by-side tool, several at once), even when the other tool is where its properties are read.
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down (usually one or two, sometimes none). They reveal one only when a message asks about that property, and an order placed before every hidden need has been told scores 0 even when the item happens to fit. So every question message allowed is worth sending, and all of them go out before anything is ordered. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, in the form "Which <property> would you like the <product> to have?" with the property's name as the listings name it, the product named by its plain noun from the request, and no example values. Never give instructions, never mention lists, labels, positions or rules, never ask two things at once, and never ask an open question ("what should I know?"): the shopper answers only a question that names a property.
- When the request asks for several different products, every question must say which product it is about; a question that leaves the product open confuses the shopper. Spread the questions over the products: the hidden needs usually sit on different products, so every product should get at least one question when messages allow.
- If the shopper says they did not follow a message, or asks you to explain, simplify or rephrase it, do not explain or repeat it: the property you asked about is most likely not one they care about. Ask about a different property next (naming the product).
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- The requirements are only what the request states and what the shopper reveals or changes. Never add one of your own from the story (an age group, a gender, a size, who will use it): such details are not requirements, and a listing does not fail because of them. If such a detail may matter, it may be a hidden need; ask about it.
- When a planner is available, your questions are chosen and sent for you as soon as you have opened the details of two or three listings of the requested kind (of each kind, when several different products are requested), so do that in your first moves. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- Messages to the shopper are only for questions (and for answering a question the shopper asked). Never message the shopper to report what you found, what you bought or that you are declining; the finishing tool carries that.
- The request may set rules for the finish: something to do with the shopper or the store before or at the order, beyond finding and buying the items (for example getting the shopper's approval of the final cart). The messages such rules need are kept for them, and the rules are checked and carried out for you when you call the order tool with the final cart; the order follows at once. So never spend those messages on questions and never decline because such a step is still to do: build the final cart and call the order tool. A cart change after such a step may void it.
- A need (or a stated requirement) may be worded as a POINTER to another listing's page: "the same as the value given by the <kind> page that <shop> sells, with <facts>", or "anything but" that value, or its counterpart (the value that page names as going with, pairing with or answering to its own). The value is then read from that page, and that page may itself send the property on to yet another listing ("for <property>, refer to the <shop> <kind> with <facts>"), one hop after another, until a page states the value for itself. Every page on that chain must be opened with the details tool (or the side-by-side tool) before the order: the grader checks that they were read. A resolver does this for you when a pointer is heard or stated, and reports the resulting value; if it reports no value, do the walk yourself: search the shop's name, open the listing of that kind with those facts, read its notes, follow the next pointer, and so on.
- If you must ask yourself: first open the details of two or three listings, then ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline, how many tool calls or messages you may use. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- FIT RULES: "make sure the <accessory> fits the <anchor> you end up buying" (or any wording that one product must work with whichever other one is bought) is decided on the pages, never from the titles' wording: the anchor's page names its own identifiers (a model number, other names it is sold under, a part series or code family that fits it), and a fitting accessory's page names one of them (in a works-with or fits list, as its own part code or series, sometimes in its title). Choose the anchor first and open its page; then search the accessory with that identifier and the accessory's kind word, and buy only an accessory whose page names it. When an accessory's page sends you to another part's page for more, show that page before the order. These links are checked for you at the order.
- CONDITIONAL REQUIREMENTS: a requirement on one product <B> that holds only if another product <A> has some value (when the <A> has it, the <B> must meet the requirement) binds the <B> only when the <A> you buy has that value on its own page. Choose the <A> by its own requirements (and the cheapest or best rule), read its page, and apply the <B> requirement only if the condition holds; it is checked for you at the order.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.
- An order attempt can be REFUSED. The result then carries the reason in some field (a refusal, complaints, objections or similar), even if the inner result looks successful (accepted: true, placed: true, executed: true). An order that goes through ends the episode: while the episode continues, nothing has been bought, whatever the result says. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- A requirement is met only when the listing's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key that names that property counts. Values match by meaning: one element of a comma-separated list is enough; singular and plural are the same; close synonyms count; case does not matter. A word that merely starts the same way does not count, and a broader or different class does not count (for example girls is not women, plain unisex is not female).
- Use the title or description as evidence only when the listing has no property key for that requirement at all. If the listing's own property says something different, the listing fails even when the title or description says the value.
- A value under a different, related property does not count (a length field does not satisfy a style requirement, a material of the lining does not satisfy the material).
- Numbers: compare as numbers after converting units; limits include the edge value; a range counts only when the whole range meets the limit.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else").
- The requested kind of product is met by the listing's category; never demand that some property repeat the category's name. The item must be the requested kind of product: when the store labels listings with categories, it must sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category called "A & B", not a narrower neighbour called "B"). It must be in stock when the request asks for things on hand, and price times quantity must be within the budget (equal is allowed).
- When a listing offers options or variants (sizes, colours, versions) with their own ids, prices or stock, each option is its own candidate: a requirement is met by the option's own value, its own price and stock count, and the cart line must name the chosen option the way the cart tool's schema asks.
- A property whose value lists several options including the required one (a comma list) simply matches; that is not mixed evidence. Evidence is mixed only when the listing's own keys for the property state different, incompatible values. Only then, and only when no cheapest or best rule decides, prefer a candidate whose evidence is clean.

CHEAPEST OR BEST
- When the request asks for the cheapest, the order must be the lowest-priced item among ALL items that meet every requirement. Seeing one fitting item is not enough. Use price filters when a tool offers them, lower the price limit until every remaining row is visible, sort the visible candidates by price yourself, and verify them in ascending price; the first one that fully fits, with every cheaper one checked and rejected, is the answer. Keep the next fitting ones as backups.
- When the request asks for the best by some measure (most, largest, longest, highest, and so on), compare ALL fitting items on that measure the same way and pick the extreme. An item whose listing states no value for that measure cannot win.

FINISHING
- Keep a ranked list of verified fits, with at least one backup, because your first pick may sell out when you check out.
- Before every order attempt: the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity; one line per item when the request asks for several different items) and nothing else (remove any other line); you have re-read the chosen item's stock and price after your last cart change, and it is in stock and within budget.
- Declining is almost never the right finish: in nearly every task the store holds an item for each requested product, and a decline scores 0. When no listing states a requirement exactly under its own key, the item meant is usually one that gives the value under a related key (a "bore: 12 mm" for a 12 mm hole size), in its title, options or description, or in a close wording (matte silver for brushed silver): order that closest fit rather than decline. Decline only after a thorough search, and only when, for every candidate, its own records clearly contradict a stated requirement, a revealed need, the budget or stock (a different value under that very key, a price over the budget, sold out). Doubt, a missing key, a missing nice-to-have or a requirement you only assumed is not a reason to decline; order the best fit instead. For several products, one product that is hard to match is never a reason to drop the whole order: put the closest fit for it in the cart with the others. Follow the request's rule for declining exactly. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Plan your steps: get your first order attempt in while at least seven steps remain, so a refusal can still be handled. When 7 steps remain, only cart, page, stock-check and finishing tools stay available. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words (when the request asks for several different items, name each of them).
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"items": when the request asks for several DIFFERENT products, one object per product: {"product": a short name for it in the request's words, "quantity": the units of that product (integer)}; an empty list when only one kind of product is requested. A kind whose name joins two words ("pots and pans", "rakes & hoes") is ONE product when the request names it as one kind.
"requirements": a list of strings, one per firm stated requirement on the product, each written as "property: constraint" with the shopper's exact value, limit, exclusion or closed list. When several different products are requested, start each one with the short product name it applies to, as "<product> <property>: constraint". Do not include price, stock, quantity, budgets or mere wishes here (but a price ceiling on ONE of several products, apart from the total budget, IS a requirement: "<product> unit price: at most N"), and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement.
"price_cap": the maximum total price as a number, or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"finish_steps": a list of objects, one per thing the request asks the assistant to do around the finish beyond finding the right items and buying them (with the shopper or with a tool, before or at the order; not questions about what the shopper wants, not budgets or caps): {"rule": the sentence quoted, "action": what must be done, in one sentence, "timing": when it must happen (for example "after the cart is final, right before the order"), "messages": how many messages to the shopper it needs (an integer, 0 when none)}. Empty when there is none.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements), or anything the request says to check with the shopper. Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"closest_if_none": true when the request says that, should nothing fit, the listing that comes closest is what it wants put forward or bought; otherwise false. Such a sentence is not a finish step and needs no message of its own.
"hint_topics": a list of short phrases naming what the story suggests the shopper secretly cares about, each as the kind of property it points to (for example "who will wear them: their gender or age group", "where it will be used: occasion or setting", "how it must look: colour, pattern or finish", "what runs on it: operating system"). The request may even invite questions about some topic (the person it is for, the room it goes in): turn that into the properties the topic points to, never into an open question. Empty when the story gives no hint.
"references": a list of objects, one per requirement or need that the request words as a pointer to another listing's page (the same as, or anything but, the value a named shop's page of some kind gives, pinned by a few facts; matching the listing a named shop sells; "of a shop's two lamps, use the lighter one's figure and avoid it"): {"product": the requested product it constrains (short name), "property": the property constrained, "relation": "same" when the value must equal the page's value, "different" when it must not ("anything but", "not the one"), "counterpart" when it must be the counterpart, partner or opposite number of the page's value (the value that page names as going with its own), otherwise the rule in a few words, "shop": the shop or seller named, "kind": the kind of product on that page, "facts": {property: value} given to pin that page, "text": the sentence quoted}. The facts that pin the page ("the kettle that Shop X sells, with a glass body and a blue lid") describe THAT page, not the requested product: they go in "facts" only, never in "requirements". Empty when there is none.
"fit_rules": a list of objects for every sentence like "make sure the <accessory> fits the <anchor> that gets bought" (or any sentence saying one requested product must suit or work with whatever other requested product is chosen): {"accessory": short product name, "anchor": short product name, "text": the sentence quoted}. Empty when there is none.
"conditions": a list of objects, one per sentence that makes a requirement on one requested product depend on what another requested product turns out to be (when the <A> has some value of one property, the <B> must meet a requirement): {"if_product": short name of A, "if_property": the property of A, "if_values": [the values of A named in the sentence], "if_not": true when B's requirement holds UNLESS A has one of those values (worded "..., except when the A's <property> is ...", "unless", "but not if"), false when it holds only IF A has one of them, "then_product": short name of B, "then_requirement": "<property>: <constraint>" for B, "text": the sentence quoted}. A conditional requirement is never put in "requirements". Empty when there is none.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "details" is the tool that returns ONE listing in full (its whole page); "view" is another tool that shows only part of one listing (only its property list, or only its stock and price), null when there is none. "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number, or null.
"new_quantity": when the shopper changes how many units they want, {"product": the product it is for (or "" when only one kind is requested), "quantity": the new number as an integer}; otherwise null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, built from the requirements so far (given below) and what the shopper has explicitly said, each as "property: constraint". Never add a requirement from the story or from your own reading of the request: only values the request states as requirements or the shopper states in a reply count. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category.
"unclear": true when the shopper says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false.
"one_at_a_time": true when the shopper says the message asked too many things and wants a single question per message; otherwise false.
"reference": when a need or change is worded as a pointer to another listing's page (the same as, or anything but, the value a named shop's page of some kind gives, pinned by a few facts; matching what a named shop's listing gives), an object {"product": the product it constrains ("" when one kind is requested), "property": the property, "relation": "same", "different", or "counterpart" when it must be the counterpart, partner or opposite number of the page's value (or the rule in a few words), "shop": the shop or seller named, "kind": the kind of product on that page, "facts": {property: value} naming that page}; otherwise null. Such a line is still a need: put it in "needs" as "<property>: <relation> as the page of the <shop> <kind> (<facts>)" until the page has been read.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "". Always "" when "unclear" or "one_at_a_time" is true, and when the line only answers the assistant's message (a value, a "don't mind", an acknowledgement): a request to explain the assistant's own question is not a question to answer, and every message sent uses up one of the few the shopper answers.
When the request asks for several different products, start every requirement in "needs" and "requirements_now" with the short name of the product it applies to, as "<product> <property>: constraint" (the shopper often says "<property> of the <product kind>").
One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit; the comparison that sits right next to the number is the one that counts, even when the sentence is worded backwards ("skip whatever goes past a ceiling of 30 inches in width" means width at most 30 inches; "keep away from any that drop under a floor of 16 inches" means at least 16 inches; a sentence that pairs a negative verb with "at most" still means at most). A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such a reply applies only to the property the assistant asked about, never drops a stated requirement, a price limit or a rule, and when the assistant's message did not ask about a property it changes nothing."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant places an order. The task scores 1 only if the ordered item meets every requirement below. Judge the item ONLY from the records shown, using these reading rules:
- A requirement is met only when the item's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by meaning: one element of a comma list is enough, singular/plural and close synonyms count, case does not matter. A merely similar-looking word or a broader/different class does not count.
- The title or description counts only when the item has no property key for that requirement. If its own property says something else, it fails even when the title says the value.
- A value under a different, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the whole range meets the limit.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement.
- Judge only the requirements listed below; never add one of your own from the story (an age group, a gender, a size, who will use it).
- When the request asks for several different products, judge the item only against the requirements of its own kind of product: a requirement that starts with another product's name does not apply to it, and the quantity is the one for its kind.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

# a line of a several-product cart, checked alone (race 10 replay: the checker failed 52% of the right lines, judging them against
# another product's requirements, the fit and if-then rules or the cart's total; named this way, 25%)
LINE_PROMPT = """

THIS CART LINE IS ONE PRODUCT OF SEVERAL: it is the pick for "{product}" (quantity {quantity}). Judge it as that product only:
- only the requirements that name this product (or name no product) apply to it; a requirement that names another requested product is not this line's, whatever the listing is;
- leave out the rules that tie two requested products together (one must fit or work with another product bought, or must have some value only if another product has one): they are checked separately against the whole cart;
- leave out the total of the whole cart: only this line's own price times its quantity must be within the budget;
- the quantity is the number of units of this listing in the cart: a pack size or a count in its facts is a property of the listing, not the quantity, unless a requirement is about it."""

PLAN_PROMPT = """You choose the questions a shopping assistant sends to a shopper. The shopper has one or two hidden needs (sometimes none): each is a firm value of ONE property of ONE requested product, revealed only when a message asks about that very property; a question about any other property gets a polite "don't mind" (or, with several products, a request to say it again, which means the same). A question names ONE property: a wide one that touches two things the shopper has in mind comes back asking for one thing at a time, and is lost. An order placed before every hidden need has been told scores 0, so the questions must hit the needs. Choose QUESTION_COUNT different properties (fewer only when fewer sensible ones exist), most promising first, by these rules:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet, asked plainly in one short sentence. When nothing is listed there, nothing must be learned first: never ask about money, a budget or a price limit on your own (the request settles it), and never invent a question; every question names ONE property that the listings carry.
1. then the properties the STORY points to. Read the request as a person's situation: who will use or wear the thing (their gender, age group, size), where or when it will be used (occasion, setting, season), how it must look (colour, pattern, finish, style), what it must work with (operating system, connector, voltage). A sentence that is not a stated requirement is there for a reason: it hints at a property the shopper cares about but did not state. The listing property that answers that hint is the best question.
2. then properties the listings below carry that would split the candidates: a property many listings of the kind carry, with a few different values among them (a style, a type, a height, a closure, a fit, a format, a capacity class). A property with one value in every listing seen, or with a different value in each (an identifier, a title, a model number, a size chart), is a poor question. A property the story points at, even faintly, comes before any other; after those, go by how many listings of that kind carry a property (the counts below): over many shoppers, more than half of all hidden needs are among the four properties most listings of the kind carry, once the settled ones are set aside, and eight in ten among the eight most carried. Colour, material or pattern come early only when many listings of that kind carry them; a colour no listing of the kind cares about (a part, a component, a food) is a wasted question. Technical details and rarely carried properties come last. A question about the kind's "style" or "type" counts for the kind's own style/type property (for example "kettle style") even when the listings spell it with the kind's name or a seller's prefix.
3. logistics come last or never: warranty, shipping, delivery, seller, brand, maker, shop, price, stock, quantity. Pack type, pack size, condition, count and capacity are ordinary properties: ask about them when the listings differ on them. Never pick a property the request already settles (its stated requirements, the price, stock, quantity, the kind of product), never repeat a property already asked for the same product (a question the shopper did not follow counts as asked: move on, unless that question did not name the product), never spend two questions on near-duplicate properties (two spellings or two kinds of style), avoid catch-all keys (features, options, details, other, care) that name no single property, and avoid properties no listing carries. An answer asking for one thing at a time means the property asked was too wide: go on with a narrower property of that family, by the listings' own name for it.
When the request asks for several different products, each question is about ONE of them, and the questions are spread so that every product gets at least one (prefer the product with fewer questions so far; the hidden needs usually sit on different products, one each, so a product whose need has been told gets no further question while another has none told); take a product's properties from the listings of its own kind (the menu is grouped by the listings' labels when they carry one).
Write each question as "Which <property> would you like the <product> to have?", with the property's name exactly as the listings name it (spaces instead of underscores or dots, without a seller's prefix) and the product named by its plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values; when two requested products share words, each is named by its own full kind name as the request gives it, never told apart by a value). No example values, no other words, no second question. For something that must be learned first (rule 0), ask it plainly in one short sentence.
Return ONE JSON object and nothing else: {"analysis": "three or four sentences, written first: what the story implies about the person and the use, which listing properties that points at, and which other properties split the candidates", "questions": [{"product": "...", "property": "...", "why": "story hint / splits candidates / learn first", "question": "..."}, ...]}."""

SWEEP_LAST = False
SWEEP_NOTE = """

THE LAST QUESTION IN YOUR LIST IS A SWEEP: write it as "Do you have a requirement for the <product>'s <property>, <property> or <property>?", naming two or three further promising properties of ONE product that no other question covers (names as the listings name them, product by its plain noun). Its "property" field lists them joined by ", "."""

FINISH_PROMPT = """You check a shopping assistant's procedure right before it places the order. Besides the right items, the request sets rules for the finish, listed below with the request, the store's tools and a log of what the assistant has done so far (in step order: searches, pages opened, cart changes, messages with the shopper's replies, order attempts). Decide from the log whether every finishing rule is met for the order about to be placed. A rule about the final cart is met only by an action taken after the cart's last change; a step the shopper did not follow (they asked for it again, or in plainer words) is not done. Answer in text with ONE JSON object and nothing else (it describes the next call; make no tool call yourself):
{"analysis": "written first: each rule, and whether and where the log shows it done",
 "next": null when every rule is met (or nothing more can be done: no messages or steps left for it), otherwise the ONE tool call that meets the first unmet rule, as {"tool": the exact tool name, "arguments": {...} exactly as that tool's schema asks},
 "then_order": false only when that call's result could change WHAT should be bought (it shows something the choice depends on); true otherwise, and then the order follows right after the call when its result changes nothing and was understood}.
Never choose the order or the decline tool, a search, or a cart change. A message to the shopper is read by a plain reader: write it as one short plain sentence about the order or the cart as a whole (a question when it asks for something). Unless the rule itself asks for something to be named, never name a product, a listing or a property in it, and never list items: the reader takes a message that names a product as a question about that product. A figure the shopper must be told (a total, a price, a count) is copied digit for digit from the store's latest result after the cart's last change that gives it, or else from CART NOW below (its sum is the cart's total at the latest prices seen, so no step is needed to read it again); never convert it to another unit, round it or work it out another way (the request's own currency word may follow it). When fewer messages are left than the unmet rules that need one, let one message meet several of them at once (one short clause per rule, in the same sentence); a rule the store's tools can meet (a check of stock or of the cart) is met by that tool, not by a message."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. That is almost always wrong: nearly every request has an item for each product in this store, and a decline scores 0. Check the listing records below against the requirements, using these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms all count); when no listing states it under its own key, a listing that gives the value under a related key, in its title, options or description, or in a close wording is the closest fit; numbers are compared after converting units, limits include the edge; the price must be within the budget and the item must not be shown out of stock; only the listed requirements count, never one assumed from the story; when several different products are requested, a listing is checked only against the requirements of its own kind of product, and every product needs its own pick. Leave a listing out only when its own records clearly contradict a listed requirement (a different value under that very key), the budget or stock. When none fits everything, the closest fit is the listing with the smallest number of unmet requirements (being sold out or over the budget each counts as one unmet); among equally close ones, the cheapest.
One case is different: when listings that meet every listed requirement were seen (for several products: one for every product) but they cost more than the budget allows (for several products: even the cheapest such cart is over it), and nothing within the budget meets every requirement, the budget alone stands in the way, and declining is right.
Return ONE JSON object and nothing else: {"possible_fits": [ids of fitting listings, or else the closest fits, best first, at most 5; for several products, the best one of each product first], "over_budget_fits": [ids of listings that meet every requirement except the price, when the budget alone stands in the way; else []], "budget_only": true when the budget alone stands in the way as described, else false, "reason": "one sentence per pick: which records show each requirement"}."""

POINTER_PROMPT = """A shopper's requirement points at another listing's page. Read the sentence and return ONE JSON object and nothing else: {"property": the property whose value is taken from that page, "relation": "same" when the required value must equal the page's value, "different" when it must differ from it, "counterpart" when it must be the counterpart, partner or opposite number of the page's value, otherwise the rule in a few words, "shop": the shop or seller the page belongs to (a proper name), "kind": the kind of product on that page, in the words used, "facts": {property: value} for each fact given to pin that page, "search": two to four words that would find that page in a keyword search (the shop's name first, then the most distinctive fact value)}."""

HOP_PROMPT = """You are following a chain of page references in a store. Below are pages that were opened (each with its id, shop, kind, facts and notes), and the reference being followed: a shop, a kind of product and facts that pin one listing. Return ONE JSON object and nothing else:
{"analysis": "written first: for each page, its shop and kind and whether its facts or notes match the reference; then what the matching page's note says about the property",
 "page": the id of the page that IS the referenced listing (same shop, same kind, and its facts or notes match the given facts), or null when none of the pages is it,
 "value": when that page's notes or facts state the property's value FOR ITSELF, that value as a short string; null when its notes send the property on to another listing instead ("for <property>, refer to / look at / go by / take it from the <shop> <kind> with <facts>, not this item"),
 "next": when the notes send the property on, {"shop": ..., "kind": ..., "facts": {...}, "search": two to four keyword-search words naming that next page (shop first)}; otherwise null,
 "required": when "value" is given, the value the requested product must have (or must avoid) by THE REQUEST'S SENTENCE below: the page's own value when the sentence asks for the same value or for anything but it; when the sentence asks for the counterpart, partner or opposite number of it, the value the page pairs with its own value (a page that pairs several values, "a goes together with b, and c goes together with d", gives the partner of its own value); otherwise null,
 "avoid": true when the sentence asks for anything but that value (the requested product must NOT have it), else false,
 "rule": when "value" is given, what THE REQUEST'S SENTENCE then demands of the requested product, as a short constraint worked out from the page: the value itself, the value the page pairs it with ("compatible with", "goes together with", "plays nicely with" and the like ask for that partner when the page names one), "anything but <value>" (only when the sentence excludes it: "nothing but it", "none but it", "only it" and their dialect forms ask for that very value), or a limit or margin computed from it ("at least 3 pieces", "between 18.2 and 493.8 gb"); otherwise null}.
A page whose notes say its own value is where the chain ends. Never take the value from a page that passes the property on, and never take it from the page's title. The referenced page is the one whose own facts or notes match EVERY fact the reference gives (shop, kind, and each fact by meaning; the shop matches when the page's seller is that shop or when the page's own note names the page as that shop's listing, "<shop>'s <kind> ..."): a page from the same shop whose facts or notes contradict one of them ("not the hooded one", "unlike ...", a different value) is a lookalike, never the page. When no page shown matches every fact, answer "page": null so that more pages are opened. "page" must be one of the ids shown below, and "next" may only repeat what that page's own notes say: the shop in "next" is the OTHER shop the note names (never the page's own shop), its kind is the kind the note names, and its facts are the ones the note lists for that other listing. When the referenced listing is not among the pages shown, answer {"page": null, "value": null, "next": null}."""

DEPEND_PROMPT = """A shopping assistant is about to order the cart below. The request ties some requested products together: a fit rule (one product must fit or work with whichever other one is bought) or a conditional requirement (should one product turn out to have some value, another product must meet a requirement). Judge each such rule for the cart as it stands, from the pages only:
- A fit shows in identifiers the pages write. The anchor's page names its own: a model number, other names it is sold under, a part series or code family that fits it, a list of parts. A fitting listing's page names one of them: in a works-with or fits list, as its own part code or series (often in its title), or by the anchor's model. Work out from the anchor's page and the other pages which link this store uses; a listing that names none of the anchor's identifiers does not fit, whatever its title or compatibility wording suggests. A line's page may send the reader on to another part for more about its fit; that other part's page can then state the line's fit ("part <the line's own code> also fits <model>"), so read it before judging.
- A conditional requirement applies only when the product it depends on, as it is in the cart, has one of the named values on its own page; then the other product's line must meet it like any requirement (its own property states the value). When the condition does not hold, the requirement is not in force.
Return ONE JSON object and nothing else:
{"analysis": "written first: for each rule, the anchor line and the identifiers or the value its page shows, then the dependent line and whether its page names one of them or meets the requirement",
 "rules": [{"rule": the rule's number as listed, "anchor": the anchor line's id, "dependent": the dependent line's id (null when no line of it is in the cart), "applies": false only for a conditional requirement whose condition the anchor line's page does not meet, otherwise true, "holds": true when the cart meets the rule, "link": [the identifiers a fitting listing must name, or the value the dependent must have], "fitting": [ids among the OTHER LISTINGS that meet the rule and seem to meet their product's other requirements, cheapest first], "search": "two to four words for a keyword search that finds listings of the dependent's kind meeting the rule (an identifier or the required value, plus the kind's word)"}],
 "see_also": [the codes of other parts that the cart lines' pages send the reader on to, when no page shown carries that code]}"""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_REASONING_ON = set()       # models that refuse a call with reasoning switched off
_RETRY_WORDS = re.compile(r"in.?flight|rate.?limit|too many|retry|overloaded|capacity|temporar", re.I)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

def _brief(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _brief(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_brief(v, limit, depth + 1) for v in value]
    return value


_PROPERTY_KEYS = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
def _line_of(args: dict[str, Any], ident: str) -> str:
    """The cart line (as text) that names this listing, when the cart call passes its lines as a list."""
    for value in args.values():
        if isinstance(value, list):
            for line in value:
                if isinstance(line, dict) and ident in line.values():
                    return _compact(line)
    return ""


def _key_words(key: str) -> tuple[str, ...]:
    return tuple(w for w in re.split(r"[^a-z0-9]+", str(key).lower()) if w)


def _fold_keys(names: list[str]) -> dict[str, str]:
    """Map each property key to a canonical spelling: a key that is another key plus one leading or trailing word
    (a seller's prefix, a qualifier) folds into the shorter key when both occur."""
    words = {n: _key_words(n) for n in names}
    canon = {n: n for n in names}
    by_words = {}
    for n, w in words.items():
        by_words.setdefault(w, n)
    for n, w in words.items():
        if len(w) < 2:
            continue
        for shorter in (w[1:], w[:-1]):
            other = by_words.get(shorter)
            if other and other != n:
                canon[n] = canon.get(other, other)
                break
    return canon


_QTY_KEYS = re.compile(r"qty|quantit|units?\b|units_|count|number|pieces|how.?many", re.I)
_MONEYISH = re.compile(r"price|cost|amount|sum|money|value|stock|discount|avail|left", re.I)


def _cart_lines(observation: Any) -> list[tuple[str, int]] | None:
    """The cart's lines (listing id, units) when a result shows them as a list of records; None when it does not."""
    found: list[list[tuple[str, int]]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 5:
            return
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            lines = []
            for row in value:
                ident = next((v for k, v in row.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), None)
                # units sit under a key named like a quantity ("number", "units_total", "qty"); a key that could be money
                # or an identifier is never read as units: such a line's units come from the cart calls instead (None)
                count = next((v for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 500
                              and _QTY_KEYS.search(str(k)) and not _MONEYISH.search(str(k))), None)
                if ident:
                    lines.append((ident, count))
            if lines:
                found.append(lines)
            return
        if isinstance(value, dict):
            for v in value.values():
                walk(v, depth + 1)
        elif isinstance(value, list):
            for v in value:
                walk(v, depth + 1)

    if isinstance(observation, dict) and not observation:
        return None
    walk(observation, 0)
    if not found and isinstance(observation, dict) and any(isinstance(v, list) and v == [] for v in observation.values()):
        return []       # an empty list of lines: the cart is empty
    return found[0] if found else None


def _next_page(observation: Any, schema: dict[str, Any], text_key: str, args: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Arguments that fetch the next page of a search result: a token or an offset the result hands back, placed
    in the search tool's argument that takes it (None when the result shows no further page). An integer the result
    hands back counts as the next offset when its key says so, or when it equals the offset asked for plus the rows
    shown, whatever its key is called."""
    if not isinstance(observation, dict):
        return None
    properties = schema.get("properties") or {}
    string_args = [k for k, v in properties.items() if k != text_key and isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
    int_args = [k for k, v in properties.items() if k != text_key and isinstance(v, dict) and v.get("type") == "integer"]
    offset_arg = next((k for k in int_args if isinstance((args or {}).get(k), int)), int_args[0] if len(int_args) == 1 else None)
    offset = (args or {}).get(offset_arg, 0) if offset_arg else 0
    offset = offset if isinstance(offset, int) and not isinstance(offset, bool) else 0
    shown = len(_rows_with_ids(observation))
    loose = []
    for key, value in observation.items():
        if isinstance(value, str) and value and len(value) <= 24 and " " not in value and re.search(r"follow|next|token|cursor|after|marker|more|page|onward|continu|resume", str(key), re.I) and string_args:
            return {string_args[0]: value}
        if (isinstance(value, str) and value and len(value) <= 24 and " " not in value and not _ID_HINT_KEYS.search(str(key))
                and value not in [str(v) for v in (args or {}).values()]):
            loose.append(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0 and int_args:
            named = re.search(r"follow|next|after|offset|start|more|page|token|cursor|marker", str(key), re.I)
            if named or (offset_arg and shown and value == offset + shown and not _COUNT_KEYS.search(str(key))):
                top = (properties.get(offset_arg or int_args[0]) or {}).get("maximum")
                if isinstance(top, (int, float)) and not isinstance(top, bool) and value > top:
                    return None
                return {(offset_arg or int_args[0]): value}
    if len(loose) == 1 and shown and string_args:
        # one bare token beside the rows, whatever its key is called, is the handle of the next page
        return {string_args[0]: loose[0]}
    return None


_COUNT_KEYS = re.compile(r"total|hit|count|match|found|result|num", re.I)
_ID_HINT_KEYS = re.compile(r"id$|^id|sort|order|status|state|mode|kind|type|query|currency|unit|lang|categor|shelf|dept", re.I)
_NOT_COUNT = re.compile(r"price|cost|page|offset|per|size|step|quant|stock|unit", re.I)


def _match_count(observation: Any) -> int | None:
    """The number of matches a search result reports (an integer field named like a total or a count)."""
    if not isinstance(observation, dict):
        return None
    for key, value in observation.items():
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0 and _COUNT_KEYS.search(str(key)) and not _NOT_COUNT.search(str(key)):
            return value
    return None


def _question_property(text: str) -> str:
    """The property a question to the shopper asks about ("Which <property> would you like the <product> to have?")."""
    m = re.search(r"\b(?:which|what)\s+(.+?)\s+(?:would|do|does|should|will|is|are)\b", text, re.I)
    core = m.group(1) if m else re.sub(r"\b(which|what|would|you|like|the|to|have|do|want|need)\b", " ", text, flags=re.I)
    core = re.sub(r"\b(the|a|an|your|you)\b", " ", core.lower())
    return " ".join(core.split())


def _question_product(text: str) -> str:
    """The product a question names ("... would you like the <product> to have?"), or ""."""
    m = re.search(r"\bthe\s+(.+?)\s+to\s+(?:have|be|come)\b", text, re.I)
    return " ".join(m.group(1).lower().split()) if m else ""


def _resolver_text(text: str) -> str:
    """A result the page-chain resolver read, as kept in the conversation: its pages stay in the index of listings and
    the resolver reports what they gave, so the long text is cut (a note it carries stays)."""
    head, _, note = text.partition("\n[note]")
    if len(head) <= 900:
        return text
    return head[:900] + " ...(read by the resolver; its finding is reported below)" + ("\n[note]" + note if note else "")


def _cond_test(cond: dict[str, Any]) -> str:
    """The side of a conditional requirement that decides it: "condition new", or "condition other than new" when the
    requirement holds unless the other product has those values."""
    values = " or ".join(cond.get("if_values") or [])
    return f"{cond.get('if_property')} " + (f"other than {values}" if cond.get("if_not") else values)


def _same_product(a: str, b: str) -> bool:
    """Two product names given with questions name the same product (an unnamed product only matches an unnamed one)."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return a == b
    return a == b or a in b or b in a or a.rstrip("s") == b.rstrip("s")


_TEXT_MODE = re.compile(r"text|keyword|word|phrase|query|search", re.I)
_FREE_TEXT_KEYS = re.compile(r"(^|[^a-z])q([^a-z]|$)|query|keyword|search|text|term|phrase|word|string|input|prompt|message|payload|content|body|look|find|seek|match|entry|descri|wanted|needle", re.I)
_FILTER_KEYS = re.compile(r"group|categor|label|maker|brand|seller|shop|vendor|department|dept|shelf|aisle|section|range|collection|famil|producer|manufactur|kind|class|tag|sort|order|cursor|token|page|bookmark|resume|marker|continu|next|after|offset|id$|code", re.I)
_FILTER_MODE = re.compile(r"field|filter|criteri|department|categor|shelf|browse|narrow|attribute|facet|struct|aisle|dept|exact|list", re.I)


def _text_key(properties: dict[str, Any], keys: list[str]) -> str | None:
    """The free-text argument among these string keys: one named like a query before one named like a filter or a page
    handle, else the first (race 2's "needle" beside "resume_bookmark", race 10's "q_string" beside "group_label")."""
    if not keys:
        return None
    return max(keys, key=lambda k: (_key_score(k), -keys.index(k)))


def _key_score(key: str) -> int:
    return (2 if _FREE_TEXT_KEYS.search(key) else 0) - (1 if _FILTER_KEYS.search(key) else 0)


def _mode_value(values: list[Any], text: bool) -> Any:
    """A search mode's value for a call with words (text=True) or with filters only: named so ("text", "keyword" /
    "criteria", "department"), else the value not named like the other way ("typed" beside "fields")."""
    names = [v for v in values if isinstance(v, str)]
    texty = [v for v in names if _TEXT_MODE.search(v)]
    filters = [v for v in names if _FILTER_MODE.search(v) and v not in texty]
    if text:
        return (texty or [v for v in names if v not in filters] or [None])[0]
    return (filters or [v for v in names if v not in texty] or [None])[0]


def _clip(text: str, limit: Any) -> str:
    """Text cut to a length limit at a word boundary (a word cut in half matches nothing)."""
    if not isinstance(limit, int) or isinstance(limit, bool) or len(text) <= limit:
        return text
    cut = text[:limit]
    if text[limit:limit + 1] not in ("", " ") and " " in cut.strip():
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip()


def _kind_words(kind: str) -> set[str]:
    """The words of a product kind, singular, without joining words ("canned fish & seafood" = "canned fish and seafood")."""
    return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", kind.lower()) if w not in ("and", "or", "the", "of", "for", "a", "an")}


def _prop_parts(prop: str) -> list[str]:
    """The properties a planned question names: one, or two joined by "or" (or a comma)."""
    return [p.strip() for p in re.split(r"\s+or\s+|,", prop) if p.strip()] or ([prop] if prop else [])


def _same_property(a: str, b: str) -> bool:
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    return a == b or (len(a) > 3 and len(b) > 3 and (a in b or b in a))


def _rows_with_ids(observation: Any) -> list[tuple[str, str]]:
    """(listing id, the row's text) for each record of a search result that carries an id."""
    out: list[tuple[str, str]] = []

    def walk(value: Any, depth: int) -> None:
        if depth > 5 or len(out) >= 60:
            return
        if isinstance(value, dict):
            ident = next((v for k, v in value.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), None)
            if ident:
                out.append((ident, _compact(_brief(value, 200))))
                return
            for v in value.values():
                walk(v, depth + 1)
        elif isinstance(value, list):
            for v in value:
                walk(v, depth + 1)

    walk(observation, 0)
    return out


def _has_text(value: Any, depth: int = 0) -> bool:
    """Whether a result holds any word (not only ids, numbers and flags)."""
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(c.isalpha() for c in value) and not (_ID_HINT.match(value) and any(c.isdigit() for c in value))
    if isinstance(value, dict):
        return any(_has_text(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(_has_text(v, depth + 1) for v in value)
    return False


def _pending_step(observation: dict[str, Any]) -> bool:
    """An order result that waits for another step (a token to confirm with) and carries no shopper sentence."""
    lines: list[str] = []
    _sentences([v for v in observation.values() if isinstance(v, (list, str))], lines)
    keys = list(observation)
    for value in observation.values():
        if isinstance(value, dict):
            keys += list(value)
    return any(_PENDING_KEYS.search(str(k)) for k in keys) and not any(len(line.split()) >= 8 for line in lines)


def _listed_ids(args: dict[str, Any]) -> list[str] | None:
    """Listing ids in a cart call that passes its lines as a list (None when the call has no such list)."""
    def id_like(key: Any, value: Any) -> bool:
        return isinstance(value, str) and bool(_ID_HINT.match(value)) and any(c.isdigit() for c in value) and bool(_IDENT_KEYS.search(str(key)))

    if any(id_like(k, v) for k, v in args.items()):
        return None             # the call names its listing directly; a list beside it holds choices, not lines
    lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    # one id per line (the first id-like value, normally the listing; an option id that follows it is shown with the listing)
    out = []
    for lines in lists:
        for line in lines:
            ident = next((v for k, v in line.items() if id_like(k, v)), None) or next(
                (v for v in line.values() if isinstance(v, str) and _ID_HINT.match(v) and any(c.isdigit() for c in v) and not v.isdigit()), None)
            if ident:
                out.append(ident)
    return out or None


# a refusal that says the item can no longer be bought (as against one that reports a changed price)
_GONE = re.compile(r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone|"
                   r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered|"
                   r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up|"
                   r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one", re.I)
# a line that is about the price (a changed price alone does not make an item unacceptable)
_PRICE_WORDS = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)


# listing metadata rather than properties of the product: never offered to the question planner. Words that are
# always metadata match anywhere in the name; words that also end real property names ("energy rating", "scent notes",
# "mounting options") only when they are the whole name
_LISTING_META = re.compile(r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
                           r"blurb|tagline|caption|teaser|slogan)\b|"
                           r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                           r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                           r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
# an error that says a tool's allowance is spent (not a bad argument or an over-budget cart)
_USED_UP = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                      r"reached the limit|maximum (number|reached)|already used", re.I)
_IDENT_KEYS = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
_OPTION_KEYS = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
_KIND_KEYS = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                        r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
# a key that marks a pending step of a checkout (a token to confirm with), not a refusal
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
# a shopper line that wants an answer, and a request rule that wants the shopper told something
_ASKS = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|abstain|withdraw|give.?up|stand.?down|walk.?away|no.?(buy|purchase|order|sale)|reject|cancel|quit|skip|^(none|nothing|no)$", re.I)
_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


def _compact(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > OBS_CHARS:
        text = text[:OBS_CHARS] + "...(cut)"
    return text


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            pass
    # several objects with words between them (a model that corrects itself): the last complete one wins
    decoder = json.JSONDecoder()
    found = None
    index = text.find("{")
    while 0 <= index < len(text):
        try:
            value, stop = decoder.raw_decode(text, index)
        except ValueError:
            index = text.find("{", index + 1)
            continue
        # only an object at the top level counts: an inner piece of a broken outer object is not an answer
        if isinstance(value, dict) and text.count("{", 0, index) == text.count("}", 0, index):
            found = value
        index = text.find("{", stop)
    return found


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "null":
        return value is None
    return True


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if not kinds:
        return value
    if any(_type_ok(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        try:
            if kind == "integer":
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
            if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                return int(number) if number.is_integer() else number
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except (TypeError, ValueError):
            continue
    return value


def _check(value: Any, schema: dict[str, Any], path: str) -> tuple[Any, list[str]]:
    if not isinstance(schema, dict):
        return value, []
    problems: list[str] = []
    value = _coerce(value, schema)
    kinds = schema.get("type")
    if isinstance(kinds, str):
        kinds = [kinds]
    if kinds and not any(_type_ok(value, k) for k in kinds):
        return value, [f"{path}: expected {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            for option in schema["enum"]:
                if isinstance(option, str) and option.lower() == value.strip().lower():
                    value = option
                    break
        if value not in schema["enum"]:
            problems.append(f"{path}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = _clip(value, schema["maxLength"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{path}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and isinstance(schema["exclusiveMinimum"], (int, float)) and value <= schema["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and isinstance(schema["exclusiveMaximum"], (int, float)) and value >= schema["exclusiveMaximum"]:
            problems.append(f"{path}: must be below {schema['exclusiveMaximum']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{path}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{path}: allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            fixed = []
            for index, item in enumerate(value):
                item, sub = _check(item, schema["items"], f"{path}[{index}]")
                fixed.append(item)
                problems += sub
            value = fixed
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{path}.{name}: required")
        if schema.get("additionalProperties") is False:
            for name in list(value):
                if name not in properties:
                    problems.append(f"{path}.{name}: not an allowed argument")
        fixed = {}
        for name, item in value.items():
            if name in properties:
                item, sub = _check(item, properties[name], f"{path}.{name}")
                problems += sub
            fixed[name] = item
        value = fixed
    return value, problems


def _sentences(value: Any, out: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            out.append(value)
    elif isinstance(value, list):
        for item in value[:10]:
            _sentences(item, out, depth + 1)


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "id" in key.lower() and _ID_HINT.match(item):
                out.add(item)
            else:
                _collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)



_SELLER_KEYS = re.compile(r"seller|dealer|merchant|vendor|shop|store|maker|brand", re.I)
_MAKER_KEYS = re.compile(r"maker|brand", re.I)
_MONEY_KEYS = re.compile(r"price|amount|cost", re.I)


class Listing:
    """What the store has shown about one listing, merged over every result that named it."""

    __slots__ = ("ident", "category", "seller", "shop_named", "price", "stock", "facts", "texts", "title", "opened", "seen_at", "group_key")

    def __init__(self, ident: str):
        self.ident = ident
        self.category = ""
        self.seller = ""
        self.shop_named = False     # the seller came from a shop's key, not from a maker's
        self.price: float | None = None
        self.stock: bool | None = None
        self.facts: dict[str, str] = {}
        self.texts: list[str] = []
        self.title = ""
        self.opened = False
        self.seen_at = 0
        self.group_key = ""

    def take(self, record: dict[str, Any], step: int, opened: bool) -> None:
        self.seen_at = step
        self.opened = self.opened or opened
        nested: dict[str, str] = {}
        plain: dict[str, str] = {}
        for key, value in record.items():
            name = str(key)
            if isinstance(value, dict) and _PROPERTY_KEYS.search(name):
                for k, v in value.items():
                    if isinstance(v, str) and 0 < len(v) <= 70:
                        nested[re.sub(r"[_.]+", " ", str(k)).strip().lower()] = v
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        nested[re.sub(r"[_.]+", " ", str(k)).strip().lower()] = str(v)
                continue
            if isinstance(value, bool):
                if "out" not in name.lower():
                    self.stock = value
                continue
            if isinstance(value, (int, float)) and _MONEY_KEYS.search(name) and "total" not in name.lower():
                self.price = float(value)
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            if _KIND_KEYS.search(name) and len(value) <= 60:
                self.category = value.strip()
            elif _SELLER_KEYS.search(name) and len(value) <= 40:
                if not _MAKER_KEYS.search(name):
                    self.seller, self.shop_named = value.strip(), True
                elif not self.shop_named:
                    self.seller = value.strip()
            elif len(value.split()) >= 6:
                if value not in self.texts:
                    self.texts.append(value[:700])
                if not self.title and len(value) <= 160:
                    self.title = value
            elif len(value) <= 70 and not (_IDENT_KEYS.search(name) and _ID_HINT.match(value) and any(c.isdigit() for c in value)):
                plain[re.sub(r"[_.]+", " ", name).strip().lower()] = value
            elif len(value) > 70 and not self.title:
                self.title = value[:160]
        if nested:
            self.facts.update(nested)
            self.opened = True
        elif plain and len(plain) >= 5 and not self.facts:
            self.facts.update({k: v for k, v in plain.items() if not _LISTING_META.search(k) and not _KIND_KEYS.search(k)})
        self.group_key = self.category.lower()

    def page(self, limit: int = 2500) -> str:
        parts = [self.ident]
        if self.category:
            parts.append("kind: " + self.category)
        if self.seller:
            parts.append("seller: " + self.seller)
        if self.title:
            parts.append("title: " + self.title[:160])
        if self.price is not None:
            parts.append(f"price: {self.price:g}")
        if self.stock is not None:
            parts.append("in stock: " + str(self.stock).lower())
        if self.facts:
            parts.append("facts: " + _compact(self.facts))
        others = [t for t in self.texts if t != self.title and not (self.title and t.startswith(self.title[:60]))]
        for index, text in enumerate(sorted(others, key=len)[:3]):
            # the shortest text is usually the seller's note (it can carry a page reference): it is kept whole
            parts.append("text: " + text[: 450 if index == 0 else 250])
        return " | ".join(parts)[:limit]


class Store:
    """An index of every listing the store has shown, filled from each tool result as it arrives."""

    def __init__(self):
        self.listings: dict[str, Listing] = {}
        self.order: list[str] = []

    def absorb(self, observation: Any, step: int, opened: bool = False) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                ident = next((v for k, v in value.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), None)
                if ident:
                    listing = self.listings.get(ident)
                    if listing is None:
                        listing = self.listings[ident] = Listing(ident)
                        self.order.append(ident)
                    listing.take(value, step, opened)
                    found.append(ident)
                    # options of a listing carry their own ids and facts; they are shown with the listing
                for k, v in value.items():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        walk(observation, 0)
        return found

    def get(self, ident: str) -> Listing | None:
        return self.listings.get(ident)

    def pages(self, idents: list[str], limit: int = 2500) -> list[str]:
        return [self.listings[i].page(limit) for i in idents if i in self.listings]

    def opened(self) -> list[Listing]:
        return [self.listings[i] for i in self.order if self.listings[i].facts]

    def groups(self) -> dict[str, list[Listing]]:
        out: dict[str, list[Listing]] = {}
        for listing in self.opened():
            out.setdefault(listing.group_key, []).append(listing)
        return out

    def by_seller(self, shop: str) -> list[Listing]:
        shop = shop.lower().strip()
        if not shop:
            return []
        return [self.listings[i] for i in self.order if shop in self.listings[i].seller.lower() or any(shop in t.lower() for t in self.listings[i].texts)]

    def cheaper(self, ident: str, budget: float | None, quantity: int, opened: set[str], skip: list[str]) -> list[tuple[str, float]]:
        held = self.listings.get(ident)
        if held is None or held.price is None:
            return []
        out = []
        for other in self.listings.values():
            if other.ident == ident or other.ident in opened or other.ident in skip or other.price is None or other.price >= held.price:
                continue
            if other.stock is False or (budget is not None and other.price * quantity > budget):
                continue
            if held.category and other.category and other.category.lower() != held.category.lower():
                continue
            out.append((other.price, other.ident))
        return [(i, p) for p, i in sorted(out)[:5]]

    def label_counts(self) -> str:
        counts: dict[str, dict[str, int]] = {}
        for listing in self.listings.values():
            for key, value in (("category", listing.category), ("seller", listing.seller)):
                if value:
                    bucket = counts.setdefault(key, {})
                    bucket[value] = bucket.get(value, 0) + 1
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 12 and key == "category":
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])[:8]))
        return "\n".join(lines)

class Episode:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.messages_used = 0
        self.message_budget: int | None = None
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checks = 0
        self.opened: set[str] = set()
        self.cheaper_checks = 0
        self.early_bounced = False
        self.refused_at = -1        # step of the last refused order
        self.searched_at = -1       # step of the last search
        self.searches_in_a_row = 0  # searches since a listing's page was last opened
        self.research_checks = 0
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.heard: set[str] = set()
        self.cart_ids: list[str] = []
        self.records: list[Any] = []
        self.last_cart_view: Any = None
        self.cart_changed_at = 0
        self.cart_view_at = -1
        self.cart_state: dict[str, str] = {}
        self.add_signature: str | None = None
        self.asked: list[str] = []
        self.questions_sent: list[str] = []
        self.planner_on = True
        self.sets_at_last_ask = -1
        self.last_unclear = False
        self.plan_misses = 0
        self.capped: set[str] = set()
        self.view_facts = False         # a reply of the view tool gave a listing its property list
        self.facts_gathered = False
        self.cart_lines: dict[str, str] = {}
        self.plan_retry_at = 0
        self.audited = False
        self.decline_ok = False
        self.commits = 0
        self.key_votes: dict[str, int] = {}     # the search's string argument the model filled alone, in calls that found rows
        self.commit_at = -1
        self.refused_ids: set[str] = set()      # listings in the cart when an order was refused
        self.qty_warned: set[str] = set()
        self.pending_reply = ""
        self.current: list[str] | None = None
        self.verdicts: dict[str, str] = {}
        self.ledger_version = 0
        self.cart_checked: set[str] = set()
        self.said: list[str] = []
        self.recent: list[str] = []
        self.roles: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.cost = 0.0
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.pointers: list[dict[str, Any]] = []      # page references still to be read
        self.resolved: list[str] = []                 # what the walks along page chains found
        self.chain_pages: list[str] = []
        self.cart_seen: list[tuple[str, int]] | None = None
        self.cart_seen_at = -1
        self.cart_qty: dict[str, int] = {}          # units last asked for each listing with the cart tool
        self.compare_cap = PAGE_OPENS                # ids the side-by-side tool shows per call (lowered when it answers empty)
        self.compare_left: list[str] = []
        self.commit_next = False        # the model declined in its last steps: the closest fit is ordered next turn
        self.forcing = False            # the last-chance order is running: its calls skip the checking gates
        self.pointers_stalled = False   # the page-chain follower cannot run (no time, tools or budget left)
        self.finish_steps: list[dict[str, Any]] = []      # the request's rules for the finish (setup reading)
        self.finish_reserved = 0                          # messages those rules need, kept from the questions
        self.finish_messages = 0                          # messages sent for them
        self.finish_next: dict[str, Any] | None = None    # the order call's arguments, run after the finishing steps
        self.finish_clear: set[str] = set()               # states the finish check passed
        self.finish_rounds: dict[str, int] = {}           # finish checks per state
        self.events: list[str] = []                       # what was done, step by step, for the finish check
        self.line_blocks: set[str] = set()
        self.hints: list[str] = []
        self.fit_rules: list[dict[str, str]] = []
        self.conditions: list[dict[str, Any]] = []          # requirements on one product that depend on another (setup reading)
        self.cond_state: dict[int, tuple[bool, str]] = {}   # condition number -> (in force, the anchor line it was judged on)
        self.dep_checks = 0                                 # checks of the rules that tie products together
        self.dep_next: list[tuple[str, list[str]]] = []     # searches for listings that meet such a rule, run before the next turn
        self.value_swept = False
        self.changed_at = -100                              # step at which the last change (drop, budget, stockout, price) was heard
        self.see_read: set[str] = set()                     # other parts a cart line's page sends the reader on to, looked up
        self.reask = False
        self.store = Store()
        self.searches_done: dict[str, str] = {}
        self.search_log: dict[str, dict[str, Any]] = {}     # per query: matches reported, rows seen, next page
        self.page_checks = 0
        self.sweep_checks = 0
        self.asked_props: list[tuple[str, str]] = []        # (product, property) of every question sent
        self.order_votes = [0, 0]                           # search pages seen in listing-id order / in another order
        self.paged: set[str] = set()                        # listings whose full page was opened (details or side-by-side tool)
        self.word_rule = ""                                 # "all" when more words found fewer matches, "any" when more
        self.blocked_turns = 0                              # model turns in a row in which no call ran
        self.page_next: list[str] = []                      # cart lines whose pages are opened before the next turn

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def order_due(self) -> int:
        """Steps kept for the first order attempt and one repair: a quarter of the steps (9 of 37, 10 of 43), at least 6, at
        most 16. A first order in the last few steps almost never passes (race 7: 1 of 46 at step 38+ of 43), but a third
        (race 8: 12 of 37) put the first pick in the cart before the cheaper fits and backups were seen, and the changes
        heard at that first cart add then had only those to fall back on."""
        return max(6, min(16, self.max_steps // 4))

    def in_reserve(self) -> bool:
        """Inside the steps kept for the first order, with a pick in the cart and no order tried yet. A change heard in the
        last 3 steps lifts it, so the choice can be made again over the new acceptable set."""
        return (self.order_attempts == 0 and bool(self.cart_ids) and self.steps_left() <= self.order_due()
                and self.step - self.changed_at > 3)

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, reasoning: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if reasoning is not None and reasoning.get("enabled") is False:
            models = [m for m in models if m not in _REASONING_ON] or list(models)
        queue = list(models)
        waits = 0
        while queue:
            model = queue.pop(0)
            left = self.time_left()
            if left < FORCE_SECONDS - 10:
                return None
            _llm.timeout = int(max(8, min(75, left - FORCE_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    if isinstance(sent[index].get("content"), str) and sent[index]["content"]:
                        sent[index]["content"] = [{"type": "text", "text": sent[index]["content"], "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if reasoning is not None:
                body["reasoning"] = reasoning
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    return message
            error = result.error if isinstance(result.error, dict) else {}
            said = str(error.get("body") or result.error or "")
            self.trace.append({"kind": "inference_failure", "model": model, "error": said[:300] if said else "empty"})
            if error.get("status") == 400 and re.search(r"reasoning is mandatory|cannot be disabled", said, re.I):
                _REASONING_ON.add(model)
            elif (error.get("status") in (402, 429, 503) or _RETRY_WORDS.search(said)) and waits < 3 and self.time_left() > WRAP_UP_SECONDS:
                # the key's in-flight budget or rate limit: wait for running requests to settle, then try the same model again
                waits += 1
                time.sleep(1.5 * waits)
                queue.insert(0, model)
        return None

    def env_call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = self.time_left()
            if left < 2:
                return None
            _env.timeout = int(max(3, min(95, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if isinstance(data.get("step"), int):
                    self.step = data["step"]
                else:
                    self.step += len(actions)
                if data.get("finished"):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.dead = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                match = re.search(r"expected (\d+)", body)
                if match:
                    self.step = int(match.group(1)) - 1
                    payload["step"] = self.step + 1
                    continue
                return None
            if status in (403, 422, 500, 503, 504):
                return None
        return None

    def setup(self) -> None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        static = SETUP_PROMPT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        variable = "REQUEST:\n" + str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            variable += "\n\nOPENING OBSERVATION:\n" + _compact(self.policy.get("opening"))
        content = [{"type": "text", "text": static, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": variable}]
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 3500, LIGHT_THINKING)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) and self.time_left() > 200:
            # without the reading every later step works blind: try once more
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 4000, NO_THINKING)
            plan = _parse_json_object((message or {}).get("content") or "") or plan
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        if not self.roles:
            # the reading failed: without roles every guard that depends on them is off, so guess from the tool names
            self.roles = self.guess_roles()
            self.trace.append({"kind": "guessed_roles", "roles": self.roles})
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        query_text = str(self.policy.get("query") or "")
        raw_items = plan.get("items") if isinstance(plan.get("items"), list) else []
        names = [str(i.get("product") or "").strip().lower() for i in raw_items if isinstance(i, dict)]
        units = [i.get("quantity") for i in raw_items if isinstance(i, dict)]
        if (len(names) == 2 and all(names) and len(set(map(str, units))) == 1
                and re.search(re.escape(names[0]) + r"\s+(and|&)\s+" + re.escape(names[1]), query_text, re.I)
                and not re.search(r"\d+\s*x\s*" + re.escape(names[1]), query_text, re.I)):
            # a kind named as "<noun> and <noun>" (pots and pans) is one product, not two
            if isinstance(units[0], int) and not isinstance(units[0], bool) and units[0] >= 1:
                plan["quantity"] = units[0]
            plan["items"] = []
            plan["product"] = f"{names[0]} and {names[1]}"
            plan["requirements"] = [re.sub(r"^(" + re.escape(names[0]) + "|" + re.escape(names[1]) + r")\s+", "", str(r), flags=re.I) for r in plan.get("requirements") or []]
            self.trace.append({"kind": "merged_items", "names": names})
        kept_requirements = []
        item_names = [n for n in names if n]
        for requirement in [str(r) for r in plan.get("requirements") or [] if r]:
            prop, _, value = requirement.partition(":")
            prop, value = prop.strip().lower(), value.strip().lower()
            pinning = None
            for ref in plan.get("references") or []:
                if not isinstance(ref, dict) or not ref.get("shop") or not value or not prop.split():
                    continue
                ref_product = str(ref.get("product") or "").strip().lower()
                own_product = next((n for n in item_names if prop.startswith(n)), "")
                if own_product and ref_product and own_product != ref_product:
                    continue        # a requirement of another requested product
                where = query_text.lower().find(str(ref.get("shop")).lower())
                if where < 0:
                    continue
                # the rest of the sentence that names the shop (up to its full stop or semicolon)
                tail = re.split(r"[.;!?](\s|$)", query_text[where:where + 300].lower())[0]
                if value in tail and prop.split()[-1] in tail:
                    pinning = ref
                    break
            if pinning is not None:
                # "the page of the lamp that Shop X lists, with a linen shade": the facts pin that page
                facts = pinning.get("facts") if isinstance(pinning.get("facts"), dict) else {}
                facts.setdefault(prop.split(" ", 1)[-1] if pinning.get("product") and prop.startswith(str(pinning.get("product")).lower()) else prop, value)
                pinning["facts"] = facts
                self.trace.append({"kind": "pinning_fact", "requirement": requirement})
                continue
            kept_requirements.append(requirement)
        product_name = str(plan.get("product") or "").lower()
        if product_name and not plan.get("items") and not re.search(r"&|\band\b|\bor\b", product_name):
            name_words = set(re.findall(r"[a-z0-9]+", product_name))
            def restates(requirement: str) -> bool:
                prop, _, value = requirement.partition(":")
                value_words = re.findall(r"[a-z0-9]+", value.lower())
                prop_words = re.findall(r"[a-z0-9]+", prop.lower())
                said = any(re.search(r"\b" + re.escape(w), query_text, re.I) for w in prop_words if len(w) > 2)
                return len(value_words) == 1 and value_words[0] in name_words and not re.search(r"\d", value) and not said
            dropped_kind = [r for r in kept_requirements if restates(r)]
            if dropped_kind:
                self.trace.append({"kind": "category_words", "dropped": dropped_kind})
                kept_requirements = [r for r in kept_requirements if not restates(r)]
        plan["requirements"] = kept_requirements
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        self.budget_now = plan.get("price_cap")
        # a story hint ("ask me about the person who will wear them") is a property question, never an open one
        kept = []
        for entry in plan.get("ask_first") or []:
            text = str(entry)
            if re.search(r"budget|price|spend|cost|afford|rather|matters more|prefer|which of", text, re.I):
                kept.append(text)
            else:
                self.hints.append(text)
        plan["ask_first"] = kept
        self.hints += [str(h) for h in plan.get("hint_topics") or [] if isinstance(h, str) and h.strip()]
        for ref in plan.get("references") or []:
            if isinstance(ref, dict) and ref.get("property") and (ref.get("shop") or ref.get("kind")):
                ref["source"] = "stated"
                self.pointers.append(ref)
        self.fit_rules = [r for r in plan.get("fit_rules") or [] if isinstance(r, dict) and r.get("accessory") and r.get("anchor")]
        for cond in plan.get("conditions") or []:
            if isinstance(cond, dict) and cond.get("if_product") and cond.get("then_product") and cond.get("then_requirement"):
                values = cond.get("if_values")
                cond["if_values"] = [str(v) for v in values] if isinstance(values, list) else [str(values or "")]
                cond["if_not"] = cond.get("if_not") is True
                self.conditions.append(cond)
                # a conditional requirement read as a plain one would bind whatever the other product turns out to be
                then = str(cond["then_requirement"]).lower()
                self.stated = [r for r in self.stated if not (then.partition(":")[2].strip() and then.partition(":")[2].strip() in r.lower()
                                                             and _same_product(r.lower()[:len(str(cond["then_product"]))], str(cond["then_product"]).lower()))]
                plan["requirements"] = list(self.stated)
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        for spec in ((self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}).values():
            # an integer of the cart tool is a line's units, whatever its name suggests (race 9: "lineNumber")
            if isinstance(spec, dict) and spec.get("type") == "integer" and not spec.get("description"):
                spec["description"] = ("A number of units of this listing, never a line's position" + (" (0 takes it out)" if spec.get("minimum") == 0 else "")
                                       + ". The reply shows each line's units after the call: some stores add the units sent to what the line holds, so check them against the quantity asked.")
        side = self.compare_tool()
        if side:
            # the declared limit is not trusted: race 8 declared 10 and answered 5, race 9 declared 4, answered 2 and
            # refused more with an error that costs a step
            spec = ((self.schemas.get(side[0]) or {}).get("properties") or {}).get(side[1]) or {}
            if isinstance(spec.get("maxItems"), int) and not isinstance(spec.get("maxItems"), bool) and spec["maxItems"] >= 1:
                self.compare_cap = min(self.compare_cap, max(min(spec["maxItems"], 2), spec["maxItems"] // 2))
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)
        for step in plan.get("finish_steps") or []:
            if isinstance(step, dict) and plan.get("closest_if_none") and re.search(r"\b(fits?|closest|nearest|falls? short|substitute)\b", str(step.get("rule")), re.I):
                continue        # what to do should nothing fit is met by the order itself, not by a message
            if isinstance(step, dict) and str(step.get("action") or step.get("rule") or "").strip():
                self.finish_steps.append(step)
                count = step.get("messages")
                if isinstance(count, (int, float)) and not isinstance(count, bool) and count > 0:
                    self.finish_reserved += int(count)

    def guess_roles(self) -> dict[str, str]:
        def words(name: str) -> set[str]:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
            return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}

        def one_text_argument(name: str) -> bool:
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            texts = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
            return len(texts) == 1

        cart_words = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}
        tests = [
            ("decline", lambda w, n: bool(w & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"})
             or any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w),
            ("cart_view", lambda w, n: bool(w & (cart_words | {"items", "order"})) and bool(w & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})),
            ("cart", lambda w, n: bool(w & cart_words) or bool(w & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(w & (cart_words | {"items", "order", "all"}))),
            ("order", lambda w, n: bool(w & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"})),
            ("message", lambda w, n: bool(w & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk", "say", "tell", "reply", "user"}) and one_text_argument(n)),
            ("details", lambda w, n: bool(w & {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info", "spec", "specs", "describe", "examine"})),
            ("search", lambda w, n: bool(w & {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"})),
        ]
        guessed: dict[str, str] = {}
        taken: set[str] = set()
        for role, test in tests:
            for tool in self.tools:
                name = tool["function"]["name"]
                if name not in taken and test(words(name), name):
                    guessed[role] = name
                    taken.add(name)
                    break
        return guessed

    def items(self) -> list[tuple[str, int]]:
        # the different products of a request that asks for several; empty for a one-product request
        out = []
        raw = self.plan.get("items")
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str) and item.strip():
                item = {"product": item, "quantity": 1}
            if isinstance(item, dict) and str(item.get("product") or "").strip():
                quantity = item.get("quantity")
                if isinstance(quantity, str) and quantity.strip().isdigit():
                    quantity = int(quantity.strip())
                quantity = int(quantity) if isinstance(quantity, (int, float)) and not isinstance(quantity, bool) and quantity >= 1 else 1
                out.append((str(item["product"]).strip(), quantity))
        return out if len(out) >= 2 else []

    def brief(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        items = self.items()
        if items:
            for name, quantity in items:
                lines.append(f"- item to buy: {name}, quantity {quantity} (one cart line per item)")
        else:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        for requirement in plan.get("requirements") or []:
            lines.append(f"- stated requirement: {requirement}")
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        if plan.get("priority") in ("cheapest", "best"):
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {plan.get('priority')}{detail} among all fitting items")
        else:
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        for want in plan.get("soft_wants") or []:
            lines.append(f"- nice to have, NOT required (only a tie-breaker): {want}")
        for item in plan.get("ask_first") or []:
            lines.append(f"- must be learned from the shopper first: {item}")
        for rule in plan.get("rules") or []:
            lines.append(f"- rule: {rule}")
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        for step in self.finish_steps:
            lines.append(f"- finishing rule (checked and carried out for you when you call the order tool): {step.get('action') or step.get('rule')} ({step.get('timing') or 'at the finish'})")
        if self.finish_reserved:
            lines.append(f"- messages kept for the finishing rules: {self.finish_reserved}; the questions use the rest")
        lines.append(f"- steps allowed: {self.max_steps}")
        if self.plan.get("closest_if_none"):
            lines.append("- when nothing fits every requirement: order the listing with the smallest number of unmet requirements (sold out or over the budget each counts as one; the cheapest of those); do not decline")
        if self.hints:
            lines.append("- the story points at hidden needs about: " + "; ".join(self.hints) + " (the planner asks about the matching listing properties first)")
        for ref in self.pointers:
            lines.append(f"- requirement worded as a pointer to another page: {ref.get('product') or ''} {ref.get('property')}: {ref.get('relation')} as the page of the {ref.get('shop')} {ref.get('kind')} ({_compact(ref.get('facts') or {})}); the resolver reads that chain of pages, then the value appears in the requirements")
        for rule in self.fit_rules:
            lines.append(f"- fit rule: the {rule['accessory']} must fit the {rule['anchor']} you buy: choose the {rule['anchor']} first; a fitting {rule['accessory']} names one of the identifiers on its page (a model number, a part series or code); checked for you at the order")
        for cond in self.conditions:
            lines.append(f"- conditional requirement: only if the {cond['if_product']} you buy has {_cond_test(cond)}, the {cond['then_product']} must have {cond['then_requirement']}; otherwise it does not apply (choose the {cond['if_product']} first, read its page; checked for you at the order)")
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            kept = self.kept_messages()
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}" + (f" ({kept} kept for the finishing rules)" if kept else ""))
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        text = "; ".join(parts) + "."
        text += "\n[requirements now] " + self.ledger()
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            # a refused order is often misread as a placed one; until the episode ends nothing has been bought
            text += (f" NOTHING HAS BEEN BOUGHT YET: the order attempt {self.step - self.refused_at} steps ago did not end the episode."
                     " If the cart holds fitting items within the current budget, call the order tool now; if not, fix the cart first, then order"
                     " (decline only if nothing in the store fits any more).")
        if self.cost > COST_HARD and left > 2:
            text += " SPENDING LIMIT REACHED: put the best verified fit in the cart (one line per requested product) and order now, or decline if nothing fits; no more searching or reading."
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a verified fit, otherwise put the closest opened fit in the cart and order (a decline almost always scores 0)."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS:
            text += " FINISH NOW: fix the cart to the verified line(s) the request needs, re-check stock, then order; when no opened listing is verified, order the closest opened fit rather than decline."
        elif self.finish_steps and self.order_attempts == 0 and left <= 9 and self.finish_pending():
            text += " Steps are running low, and the request's finishing rules take a step or two right before the order: within the next few steps put the final cart together and call the order tool while at least 3 steps remain (the rules are carried out for you then)."
        elif self.in_reserve():
            text += f" ORDER NOW: the last {self.order_due()} steps are kept for the first order and a repair (a late first order almost never passes); searching has ended. Re-check the cart's lines if needed, then call the order tool."
        elif self.order_attempts == 0 and left <= self.order_due() + 4:
            text += f" Steps are running low: within the next few steps put your best verified fit in the cart and order; searching ends once a pick is in the cart and {self.order_due()} steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
            joined = " ".join(items).lower()
            for need in self.revealed:
                if need in self.dropped:
                    continue
                head = need.split(":")[0].strip().lower()
                word = next((w for w in head.split() if len(w) >= 4), head)
                if word and word not in joined:
                    items.append(f"{need} (revealed by the shopper)")
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(items) if items else "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        items = self.items()
        if items:
            text += "; items and quantities: " + ", ".join(f"{name} x{quantity}" for name, quantity in items)
        else:
            text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        if self.resolved:
            text += "; read from the referenced pages: " + "; ".join(self.resolved)
        held = set(self.held_lines()) if self.conditions else set()
        for number, cond in enumerate(self.conditions, len(self.fit_rules) + 1):
            state, anchor = self.cond_state.get(number, (None, ""))
            rule = f"{cond['then_product']} {cond['then_requirement']}"
            if state is not None and anchor in held:
                text += (f"; in force: {rule} (the {cond['if_product']} in the cart, {anchor}, has {_cond_test(cond)})" if state
                         else f"; NOT in force: {rule} (the {cond['if_product']} in the cart, {anchor}, does not have {_cond_test(cond)})")
            else:
                text += f"; only if the {cond['if_product']} bought has {_cond_test(cond)}: {rule}"
        if self.pointers:
            text += "; still to be read from a referenced page: " + "; ".join(f"{r.get('property')} ({r.get('relation')} as the {r.get('shop')} {r.get('kind')} page)" for r in self.pointers)
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.cost > COST_SOFT and models[0] == JUDGE_MODEL:
            # an expensive episode keeps its checks but makes them on the cheaper model, so one task cannot eat the key
            models = [READ_MODEL, JUDGE_MODEL]
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens, NO_THINKING)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        if not result:
            return ""
        needs_text = " ".join(str(n) for n in result.get("needs") or [])
        # a plain value ("flavor: strawberry") is read reliably the first time; a limit, an exclusion, a change or a
        # page reference is read once more by the stronger model (backwards wordings flip limits)
        tricky = bool(re.search(r"\d|\b(not|no|never|except|but|bar|other than|without|any|only|least|most|under|over|below|above|between|same|different|page|listing)\b", needs_text, re.I))
        if ((result.get("needs") and tricky) or result.get("new_budget") or result.get("dropped") or result.get("new_quantity") or result.get("reference")) and self.time_left() > WRAP_UP_SECONDS + 30:
            # a line that changes the requirements is read once more by the stronger model: a misread need loses the task
            message = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL], 2200, LIGHT_THINKING)
            confirmed = _parse_json_object((message or {}).get("content") or "")
            if confirmed:
                result = confirmed
        notes = []
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            self.current = [r for r in now if r.strip() and not (self.restates_kind(r) and r not in self.revealed)]
        for need in result.get("needs") or []:
            if isinstance(need, str) and need and need not in self.revealed:
                self.revealed.append(need)
                notes.append(f"new requirement: {need}")
        for prop in result.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        for drop in result.get("dropped") or []:
            if isinstance(drop, str) and drop and drop not in self.dropped:
                self.dropped.append(drop)
                notes.append(f"dropped: {drop}")
        budget = result.get("new_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget_now:
            self.budget_now = budget
            notes.append(f"new budget: {budget}")
        change = result.get("new_quantity")
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if isinstance(change, dict) and isinstance(change.get("quantity"), int) and not isinstance(change.get("quantity"), bool) and change["quantity"] >= 1:
            wanted = change["quantity"]
            product = str(change.get("product") or "").strip().lower()
            if not self.items():
                # one kind of product: the plan's quantity is the one that counts
                self.plan["quantity"] = wanted
                notes.append(f"new quantity: {wanted}")
            else:
                items = self.plan.get("items") if isinstance(self.plan.get("items"), list) else []
                target = next((i for i in items if isinstance(i, dict) and product and (product in str(i.get("product") or "").lower() or str(i.get("product") or "").lower() in product)), None)
                if target is not None:
                    target["quantity"] = wanted
                    notes.append(f"new quantity: {target.get('product')} x{wanted}")
        if result.get("unavailable"):
            notes.append("an item became unavailable; remove it from the cart and do not order it")
        heard_text = " ".join(str(l) for l in lines)
        if (result.get("dropped") or result.get("new_budget") or result.get("unavailable")
                or (question is None and (_PRICE_WORDS.search(heard_text) or _GONE.search(heard_text)))):
            self.changed_at = self.step
            if self.plan.get("priority") in ("cheapest", "best"):
                notes.append("the set of fitting listings has changed, so choose again over everything seen so far: the "
                             + str(self.plan.get("priority")) + " listing that meets the requirements as they stand now (one rejected earlier only for a dropped "
                             "requirement fits now; a sold-out one is out; a changed price is re-read), searching once more if the steps allow")
        pointer = result.get("reference")
        if isinstance(pointer, dict) and pointer.get("property") and (pointer.get("shop") or pointer.get("kind")):
            pointer["source"] = "told"
            pointer.setdefault("text", " ".join(str(l) for l in lines)[:500])
            if not any(p.get("property") == pointer.get("property") and p.get("shop") == pointer.get("shop") for p in self.pointers):
                self.pointers.append(pointer)
                notes.append("this need points at another listing's page; the resolver reads that chain of pages next and adds the value to the requirements")
        if result.get("one_at_a_time") is True and question is not None:
            # the reader saw two subjects in the message: a question naming two properties is asked again, one at a time;
            # one naming a single property was too wide (race 9: "packaging" for "pack type" and "packaging type")
            self.reask = bool(_prop_parts(_question_property(str(question)))[1:])
            notes.append("the question touched two things the shopper has in mind: ask next about one narrower property of that family, by the name the listings give it")
        if notes:
            self.ledger_version += 1
        reply = str(result.get("reply") or "").strip()
        unclear = result.get("unclear") is True and question is not None
        self.last_unclear = unclear
        named = True
        if unclear and self.items():
            asked_words = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
            named = any(w.rstrip("s") in asked_words for product, _ in self.items() for w in product.lower().split()
                        if len(w) >= 3 and w not in ("and", "the", "for", "set", "with", "pair", "pack", "of"))
        if unclear and not named:
            # with several products, a question that does not say which one is ambiguous: the property may still matter
            notes.append("the shopper did not follow that question because it did not say which product it is about: if messages remain, ask about that property once more, naming the product")
        elif unclear:
            # explaining our own question again rarely helps; the asked property is most likely not a need
            notes.append("the shopper did not follow that question: do not explain or repeat it; the property asked about is most likely free, so ask about a different one (naming the product) if messages remain")
        elif reply and self.roles.get("message") and _ASKS.search(" ".join(fresh)):
            # only a line that asks something gets an answer; a stated need needs no confirmation message
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def item_records(self, item: str) -> list[str]:
        listing = self.store.get(item)
        found: list[str] = []
        if listing is not None:
            found.append(listing.page(3000))
        # an option inside a listing shares the listing's facts: show the holder as well
        for other in self.store.listings.values():
            if other.ident != item and any(item == v for v in other.facts.values()):
                found.append(other.page(2000))
        if not found:
            for record in self.records[-12:]:
                text = _compact(_brief(record, 260))
                if item in text:
                    found.append(text[:3000])
        return found[-3:]

    def cart_total(self) -> float | None:
        """The sum of the latest prices seen times the units of the cart's lines (None when a line's price was never seen)."""
        if self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at:
            lines = [(i, n) for i, n in self.cart_seen if n > 0]
        else:
            lines = [(i, self.cart_qty.get(i, 1)) for i in self.cart_ids]
        total = 0.0
        for ident, units in lines:
            listing = self.store.get(ident)
            if listing is None or listing.price is None:
                return None
            total += listing.price * units
        return total if lines else None

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        # the lines are checked one by one, each against its own product: the whole cart's total is checked here
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        total = self.cart_total()
        if budget is not None and total is not None and total > budget + 1e-9:
            return (f"The cart's lines add up to {total:g} (the latest prices seen times the units), over the budget of {budget:g}. "
                    "Replace the dearest line with a cheaper fit of the same product, or bring a line's units down to its product's quantity, then order.")
        # a request for several different products puts one line per product in the cart: check each line
        for item in (self.cart_ids[-3:] if self.items() else self.cart_ids[-1:]):
            verdict = self.verify_item(item, "ORDER", self.cart_lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def verify_item(self, item: str, purpose: str, line: str = "") -> str:
        if self.time_left() < WRAP_UP_SECONDS + 20:
            return ""
        key = f"{item}|{line}|{self.ledger_version}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger()
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if line:
            prompt += "\n\nTHE CART LINE FOR IT (when it names an option or variant, judge that option: its own values, price and stock count): " + line
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:2000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        listing = self.store.get(item)
        kind = listing.category if listing is not None and listing.category else ""
        picks = [(name, quantity) for name, quantity in self.items() if kind and _same_product(kind, name)]
        if len(picks) == 1:         # a category holding two requested products leaves the line's product open: no note
            prompt += LINE_PROMPT.format(product=picks[0][0], quantity=picks[0][1])
        message = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL, READ_MODEL], 1500, NO_THINKING)
        result = _parse_json_object((message or {}).get("content") or "")
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        # a "does not fit" from the quick check is confirmed by the heavy model with its own reasoning (it refuses a
        # no-thinking call): race 10 replay, right lines failed 20% -> 2%, wrong lines caught 9/17 -> 7/17, asked on a
        # quarter of the checks; the agent swaps a line after an alarm whether it is right or not (past COST_SOFT too: the
        # checks come late, when most episodes of several products have spent more than that)
        if self.cost < COST_HARD and self.time_left() > WRAP_UP_SECONDS + 30:
            second = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL], 4000, None)
            check = _parse_json_object((second or {}).get("content") or "")
            self.trace.append({"kind": "verify_confirm", "item": item, "result": check})
            if check and check.get("fits") is True:
                self.verdicts[key] = ""
                return ""
            if check and check.get("fits") is False and check.get("problems"):
                result = check
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def window(self) -> list[dict[str, Any]]:
        """The conversation for the model: once it is long, tool results older than the last KEEP_TURNS messages are
        shortened (the index of listings keeps every fact for the checks), trimmed in blocks so the cached prefix holds."""
        if sum(len(m["content"]) for m in self.messages if isinstance(m.get("content"), str)) < WINDOW_CHARS:
            return self.messages
        cut = len(self.messages) - KEEP_TURNS
        cut -= cut % TRIM_BLOCK
        if cut <= 2:
            return self.messages
        out = []
        for index, message in enumerate(self.messages):
            content = message.get("content")
            if index < cut and message.get("role") == "tool" and isinstance(content, str) and len(content) > TRIM_CHARS + 200:
                message = dict(message)
                message["content"] = content[:TRIM_CHARS] + " ...(older result shortened; its listings' facts, prices and stock remain in the checks)"
            out.append(message)
        return out

    def offered_tools(self) -> list[dict[str, Any]]:
        """The tool list sent to the main model: the same list on every turn (the tools come first in the cached prefix,
        so a changed list rewrites the whole conversation at the write price), with calls outside tools_for_turn refused
        with the reason at no step; only the last steps and the spending limit send the narrowed list."""
        if self.steps_left() <= 3 or self.cost > COST_HARD + 0.02:
            return self.tools_for_turn()
        kept = [t for t in self.tools if t["function"]["name"] not in self.capped]
        return kept or self.tools

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3 or self.cost > COST_HARD + 0.02:
            names = {order, decline, cart}
        elif self.cost > COST_HARD:
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view"), self.roles.get("details")}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view"), self.roles.get("details")}
        elif self.in_reserve():
            # the steps kept for the first order: the pick's own reads, the cart and the finish, no more searching
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view"), self.roles.get("details"), (self.compare_tool() or ("",))[0]}
        tools = self.tools
        if names is not None and order and cart:
            # a tool no role covers that finishes a checkout (a confirm step after the order tool) stays available
            assigned = set(self.roles.values())
            names |= {t["function"]["name"] for t in self.tools if t["function"]["name"] not in assigned
                      and _FINISH_STEP.search(t["function"]["name"])}
            narrowed = [t for t in self.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if self.planner_on and self.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != self.roles.get("message")]
        if self.capped:
            kept = [t for t in tools if t["function"]["name"] not in self.capped]
            tools = kept or tools
        return tools

    def property_menu(self) -> tuple[str, int]:
        # the pages of a reference chain are often of another kind: their properties are no question for this shopper
        groups = {g: [l for l in v if l.ident not in self.chain_pages] for g, v in self.store.groups().items()}
        fact_sets = sum(len(v) for v in groups.values())
        counts: dict[str, dict[str, int]] = {}
        distinct: dict[str, dict[str, set[str]]] = {}
        samples: dict[str, dict[str, list[str]]] = {}
        for group, listings in groups.items():
            for listing in listings:
                for name, value in listing.facts.items():
                    if re.search(r"\d", value) and re.fullmatch(r"[A-Za-z0-9_\-]+", value) and ("-" in value or "_" in value):
                        continue
                    if _LISTING_META.search(name) or _KIND_KEYS.search(name):
                        continue
                    counts.setdefault(group, {})[name] = counts.setdefault(group, {}).get(name, 0) + 1
                    distinct.setdefault(group, {}).setdefault(name, set()).add(value.lower())
                    bucket = samples.setdefault(group, {}).setdefault(name, [])
                    if value.lower() not in bucket and len(bucket) < 5:
                        bucket.append(value.lower())
        # the same property under a seller's prefix or a qualifier ("kid sneaker height", "material filter") is one property
        for group in list(counts):
            canon = _fold_keys(list(counts[group]))
            for name, target in canon.items():
                if target != name and target in counts[group]:
                    counts[group][target] += counts[group].pop(name)
                    distinct[group][target] |= distinct[group].pop(name, set())
                    bucket = samples[group].setdefault(target, [])
                    for v in samples[group].pop(name, []):
                        if v not in bucket and len(bucket) < 5:
                            bucket.append(v)

        def identifier_like(group: str, name: str) -> bool:
            # a value that differs on every listing, or runs long, names the listing rather than a property of it
            c, d, sm = counts[group][name], distinct[group].get(name, set()), samples[group].get(name, [])
            if c >= 3 and len(d) == c and sum(len(v) for v in sm) / max(1, len(sm)) > 12:
                return True
            return bool(sm) and sum(len(v) for v in sm) / len(sm) > 45

        def lines_for(group: str, limit: int) -> list[str]:
            c, d, sm = counts.get(group, {}), distinct.get(group, {}), samples.get(group, {})
            usable = [k for k in c if not identifier_like(group, k)]
            # properties whose values differ between the listings come first: only they can change the choice
            ranked = sorted(usable, key=lambda k: (len(d.get(k, ())) < 2, -c[k]))[:limit]
            top = sorted(usable, key=lambda k: -c[k])[:5]
            total = max((c[k] for k in usable), default=0)
            return [f"- {k} (in {c[k]} listings, {len(d.get(k, ()))} different values{'; carried by most' if k in top and c[k] >= max(2, total / 2) else ''}): " + " | ".join(sm.get(k, [])) for k in ranked]

        tables = {g: len(v) for g, v in groups.items()}
        named = [g for g in sorted(counts, key=lambda g: -tables.get(g, 0)) if g and tables.get(g, 0) >= 2]
        if len(named) >= 2 and self.items():
            out = []
            for group in named[:4]:
                out.append(f"Listings labelled \"{group}\" ({tables[group]} with details):")
                out.extend(lines_for(group, 14))
            if counts.get(""):
                out.append("Listings without a label:")
                out.extend(lines_for("", 10))
            return "\n".join(out), fact_sets
        merged: dict[str, int] = {}
        for group in list(counts):
            for name, n in counts[group].items():
                merged[name] = merged.get(name, 0) + n
                distinct.setdefault("", {}).setdefault(name, set()).update(distinct[group][name])
                bucket = samples.setdefault("", {}).setdefault(name, [])
                for v in samples[group][name]:
                    if v not in bucket and len(bucket) < 5:
                        bucket.append(v)
        counts[""] = merged
        return "\n".join(lines_for("", 28)), fact_sets

    def row_labels(self) -> str:
        return self.store.label_counts()

    def text_args(self, role: str, text: str) -> dict[str, Any] | None:
        """Arguments for a tool of this role that takes one free-text argument (a search or a message)."""
        name = self.roles.get(role)
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or self.query_key(properties, strings)
        if not target:
            return None
        limit = properties[target].get("maxLength")
        out: dict[str, Any] = {target: _clip(text, limit)}
        for key in required:
            if key == target:
                continue
            # one tool, several ways of working: a required mode is set to its free-text one
            mode = _mode_value((properties.get(key) or {}).get("enum") or [], True)
            if mode is None:
                return None
            out[key] = mode
        return out

    def query_key(self, properties: dict[str, Any], strings: list[str]) -> str | None:
        """The free-text argument by its name; when the names do not tell (no query-like name, or two alike), the one the
        model itself filled with words in searches that found rows."""
        key = _text_key(properties, strings)
        if len(strings) > 1:
            ranked = sorted(strings, key=_key_score, reverse=True)
            if _key_score(ranked[0]) <= 0 or _key_score(ranked[0]) == _key_score(ranked[1]):
                voted = [k for k in strings if self.key_votes.get(k)]
                if voted:
                    key = max(voted, key=lambda k: self.key_votes[k])
        return key

    def search_mode(self, args: dict[str, Any]) -> dict[str, Any]:
        """A search tool with a required mode: words go with its free-text mode, a call without words with the other one."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        properties = schema.get("properties") or {}
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        query = self.query_key(properties, strings)
        # words are the free-text argument's; a filter (a category, a maker) is not words (race 10: a filter search
        # sent in text mode came back empty)
        words = (isinstance(args.get(query), str) and bool(args[query].strip())) if query else any(
            isinstance(v, str) and v.strip() and "enum" not in (properties.get(k) or {}) for k, v in args.items())
        for key in schema.get("required") or []:
            values = [v for v in (properties.get(key) or {}).get("enum") or [] if isinstance(v, str)]
            if len(values) < 2:
                continue
            want, other = _mode_value(values, words), _mode_value(values, not words)
            # only a mode of the other way is changed (a third mode the model chose stays)
            if want is not None and want != other and (args.get(key) == other or args.get(key) not in values):
                args = dict(args, **{key: want})
        return args

    def shelf_args(self, kind: str) -> dict[str, Any]:
        """The search tool's optional closed-list filter set to the kind named, when one of its values is that kind."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        words = _kind_words(kind)
        for key, spec in (schema.get("properties") or {}).items():
            values = spec.get("enum") if isinstance(spec, dict) else None
            if key in (schema.get("required") or []) or not values or not words:
                continue
            match = next((v for v in values if _kind_words(str(v)) == words), None) or next(
                (v for v in values if _kind_words(str(v)) >= words or words >= _kind_words(str(v))), None)
            if match is not None:
                return {key: match}
        return {}

    def rank_target_rows(self, ids: list[str], target: dict[str, Any]) -> list[str]:
        """Rows that may be the page a reference names, likeliest first: the shop's own rows, of the kind, with the most
        of the reference's fact values in their text. A row that shows another shop is left out: each page costs a step."""
        shop = str(target.get("shop") or "").strip().lower()
        kind = _kind_words(str(target.get("kind") or ""))
        facts = target.get("facts") if isinstance(target.get("facts"), dict) else {}
        tokens = {w for v in facts.values() for w in re.findall(r"[a-z0-9.]+", str(v).lower()) if len(w) >= 2 or w.isdigit()}

        def rank(ident: str) -> tuple[int, int]:
            listing = self.store.get(ident)
            if listing is None:
                return (1, 0)
            flat = " ".join([listing.title, *listing.texts]).lower()
            other_kind = bool(kind and listing.category and not (_kind_words(listing.category) >= kind or kind >= _kind_words(listing.category)))
            return (int(other_kind), -len(tokens & set(re.findall(r"[a-z0-9.]+", flat))))

        def own(ident: str) -> bool:
            listing = self.store.get(ident)
            return not (listing is not None and shop and listing.shop_named and shop not in listing.seller.lower())
        return sorted([i for i in ids if own(i)], key=rank)

    def compare_tool(self) -> tuple[str, str] | None:
        """A tool that opens several listings at once: its name and the array argument that takes their ids."""
        for tool in self.tools:
            name = tool["function"]["name"]
            if name in (self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline"), self.roles.get("message"), self.roles.get("search")):
                continue
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, spec in properties.items():
                if isinstance(spec, dict) and spec.get("type") == "array" and (spec.get("items") or {}).get("type", "string") == "string":
                    return name, key
        return None

    def chain_floor(self) -> int:
        """The steps a chain of page references leaves for the questions still to send, the cart and the order."""
        return max(6, self.order_due() + max(0, self.question_limit() - self.messages_used))

    def resolve_pointers(self) -> None:
        """Walk the chain of pages a requirement points at, opening every page (the grader checks that), and add the
        value found to the requirements."""
        search = self.roles.get("search")
        details = self.roles.get("details") or self.roles.get("view")
        while self.pointers and search and details:
            if self.steps_left() <= self.chain_floor() + 2 or self.time_left() < WRAP_UP_SECONDS + 40 or self.dead or self.finished:
                return
            ref = self.pointers.pop(0)
            target = {"shop": ref.get("shop"), "kind": ref.get("kind"), "facts": ref.get("facts") or {}, "search": ref.get("search")}
            if not target.get("search") and ref.get("text"):
                parsed = self.judge(POINTER_PROMPT + "\n\nTHE SENTENCE:\n" + str(ref.get("text")), 500, [READ_MODEL, JUDGE_MODEL]) or {}
                for key in ("shop", "kind", "facts", "search"):
                    if parsed.get(key):
                        target[key] = parsed[key]
                if parsed.get("property"):
                    ref["property"] = parsed["property"]
                if parsed.get("relation"):
                    ref["relation"] = parsed["relation"]
            label = f"{ref.get('product') + ' ' if ref.get('product') else ''}{ref.get('property')}"
            found_value = None
            own_value = ""
            hops = 0
            opened_here: list[str] = []
            while hops < HOP_LIMIT and self.steps_left() > self.chain_floor() and self.time_left() > WRAP_UP_SECONDS + 30 and self.cost < COST_HARD:
                hops += 1
                shop = str(target.get("shop") or "").strip()
                facts = target.get("facts") if isinstance(target.get("facts"), dict) else {}
                kind = str(target.get("kind") or "").strip()
                queries: list[str] = []
                if shop:
                    # any-word and all-word stores alike: the shop with what pins the page, then the shop alone
                    if facts:
                        queries.append(" ".join([shop, *(str(v) for v in facts.values())])[:80])
                    queries.append(shop)
                    for value in list(facts.values())[:2]:
                        queries.append(f"{shop} {value}")
                    if kind:
                        queries.append(f"{shop} {kind.split()[0]}")
                words = str(target.get("search") or "").strip()
                if words and len(words.split()) <= 3:
                    queries.append(words)
                if not queries and words:
                    queries.append(" ".join(words.split()[:3]))
                hop_result: dict[str, Any] = {}
                settled = False
                shelf = self.shelf_args(kind)
                for query in queries[:4]:
                    if settled or self.steps_left() <= self.chain_floor() or self.cost >= COST_HARD:
                        break
                    args = self.text_args("search", query)
                    if not args:
                        break
                    text_key = next(iter(args))
                    args.update(shelf)
                    page_args: dict[str, Any] | None = None
                    left_over: list[str] = []
                    for page in range(max(HOP_BATCHES + 1, 12 // max(1, self.compare_cap))):
                        compare = self.compare_tool()
                        per_batch = max(1, min(PAGE_OPENS, self.compare_cap)) if compare else 3
                        if left_over:
                            # rows of the last result page that were not opened yet come before the next page
                            ids, left_over = left_over[:per_batch], left_over[per_batch:]
                        else:
                            call_args = dict(args)
                            if page_args:
                                call_args.update(page_args)
                            elif page:
                                break
                            call = {"id": f"ref_search_{self.step}_{page}", "type": "function", "function": {"name": search, "arguments": json.dumps(call_args)}}
                            self.messages.append({"role": "assistant", "content": f"I will look up the page that the {label} requirement points at.", "tool_calls": [call]})
                            for call_id, text in self.run_calls([call], planned=True):
                                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _resolver_text(text)})
                            result = self.records[-1] if self.records else None
                            rows = _rows_with_ids(result)
                            if not rows:
                                if query == shop:
                                    shelf = {}  # the filter may be the wrong one: the next search goes without it
                                break
                            page_args = _next_page(result, self.schemas[search], text_key, call_args)
                            fresh = self.rank_target_rows([i for i, _ in rows if i not in opened_here], target)
                            if not fresh:
                                if page_args:
                                    continue
                                break
                            ids, left_over = fresh[:per_batch], fresh[per_batch:]
                        # a page opened earlier is read again from memory: it costs no step
                        unread = [i for i in ids if i not in self.paged]
                        if compare and len(unread) > 1:
                            name, key = compare
                            calls = [{"id": f"ref_open_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps({key: unread})}}]
                        else:
                            calls = []
                            ids = ids[:3]
                            unread = [i for i in unread if i in ids]
                            for index, ident in enumerate(unread):
                                open_args = self.text_args("details", ident) or self.neutral_args(details, ident)
                                calls.append({"id": f"ref_open_{self.step}_{index}", "type": "function", "function": {"name": details, "arguments": json.dumps(open_args)}})
                        replies: list[tuple[str, str]] = []
                        if calls:
                            self.messages.append({"role": "assistant", "content": "I will open the pages that may be the one referred to.", "tool_calls": calls})
                            cap_before = self.compare_cap
                            replies = self.run_calls(calls, planned=True)
                            for call_id, text in replies:
                                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _resolver_text(text)})
                            if compare and len(unread) > 1 and self.compare_cap < cap_before:
                                # the tool showed nothing, or only some, for this many: the rest are opened again, fewer at a time
                                missing = [i for i in ids if i not in self.paged]
                                left_over = missing + left_over
                                ids = [i for i in ids if i in self.paged]
                                if not ids:
                                    continue
                        pages = self.store.pages(ids, 3000) or [text.split("\n[status]")[0][:3500] for _, text in replies]
                        opened_here += ids
                        self.chain_pages += ids
                        prompt = HOP_PROMPT + f"\n\nTHE PROPERTY BEING TRACED: {ref.get('property')}\nTHE REFERENCE: shop {shop}, kind {kind}, facts {_compact(facts)}"
                        if ref.get("text"):
                            prompt += f"\nTHE REQUEST'S SENTENCE: {ref.get('text')}"
                        prompt += "\n\nPAGES OPENED:\n" + "\n".join(pages)
                        hop_message = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL, READ_MODEL], 1600, LIGHT_THINKING)
                        hop_result = _parse_json_object((hop_message or {}).get("content") or "") or {}
                        self.trace.append({"kind": "hop", "property": ref.get("property"), "target": target, "result": hop_result})
                        if hop_result.get("page") in ids:
                            settled = True
                            break
                        if (not page_args and not left_over) or self.steps_left() <= self.chain_floor() or (page >= 2 and not compare):
                            break
                if not settled:
                    break
                if hop_result.get("value"):
                    found_value = own_value = str(hop_result["value"])
                    required = str(hop_result.get("required") or "").strip()
                    if required and str(ref.get("relation") or "same") in ("same", "different", "counterpart"):
                        # the sentence decides which value binds (the page's own, or the one it pairs with its own)
                        if required.lower() != found_value.lower():
                            found_value = f"{required} (the counterpart of {found_value}, as the page pairs them)"
                        ref["relation"] = "different" if hop_result.get("avoid") is True else "same"
                    elif str(ref.get("relation")) == "counterpart":
                        ref["relation"] = f"the counterpart of what that page states (the value it names as going with its own {found_value})"
                    break
                nxt = hop_result.get("next")
                if isinstance(nxt, dict) and (nxt.get("shop") or nxt.get("kind")):
                    target = {"shop": nxt.get("shop"), "kind": nxt.get("kind"), "facts": nxt.get("facts") or {}, "search": nxt.get("search")}
                    continue
                break
            relation = str(ref.get("relation") or "same")
            rule = str(hop_result.get("rule") or "").strip() if found_value else ""
            if rule.lower().startswith(("anything but", "not ")) and relation == "same":
                # the first reading of the whole request says the same value is wanted: an exclusion here is a misreading
                rule = own_value or found_value
                hop_result["avoid"] = False
            line = ""
            if found_value and rule and rule.lower() not in ("null", "none"):
                # the reader worked the requirement out from the request's own sentence and the last page
                line = f"{label}: {rule} (read from the page chain of the {ref.get('shop')} {ref.get('kind')}, whose last page gives {own_value or found_value})"
                if hop_result.get("avoid") is True or rule.lower().startswith(("anything but", "not ")):
                    line += "; a listing that states no value for it does not count as different"
            elif found_value:
                if relation == "same":
                    line = f"{label}: {found_value} (read from the page chain of the {ref.get('shop')} {ref.get('kind')})"
                elif relation == "different":
                    line = f"{label}: anything but {found_value} (read along the page chain of the {ref.get('shop')} {ref.get('kind')}; a listing that states no value for it does not count as different)"
                else:
                    line = f"{label}: {relation}; the page chain of the {ref.get('shop')} {ref.get('kind')} gives {found_value}"
            if line:
                self.resolved.append(line)
                # the requirement as worded (a pointer to that shop's page) gives way to the value read
                shop_name = str(ref.get("shop") or "\n").lower()
                def is_pointer(c: str) -> bool:
                    return shop_name in c.lower() or (str(ref.get("property") or "").lower() in c.lower() and "page" in c.lower())
                self.stated = [c for c in self.stated if not is_pointer(c)]
                self.revealed = [c for c in self.revealed if not is_pointer(c)]
                if ref.get("source") == "stated":
                    self.stated.append(line)
                else:
                    self.revealed.append(line)
                if self.current is not None:
                    self.current = [c for c in self.current if not is_pointer(c)] + [line]
                self.ledger_version += 1
                self.messages.append({"role": "user", "content": f"[resolver] {line}. Pages read on the way: {', '.join(opened_here)}. This now binds the choice; search again with it.\n" + self.status()})
            else:
                self.messages.append({"role": "user", "content": f"[resolver] The page chain for {label} stopped at this hop: shop {target.get('shop')}, kind {target.get('kind')}, facts {_compact(target.get('facts') or {})} (pages already read, which count: {', '.join(opened_here) or 'none'}; do not read them again). Spend at most three more calls on it: one search with the shop's name and a fact value, open the matching listing, read its notes for the value or the next pointer. If that fails, choose the best verified fit on the other requirements and order; do not ask the shopper about this property.\n" + self.status()})

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get("message")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        strings = [k for k, v in properties.items() if isinstance(v, dict) and (v.get("type") == "string" or "string" in (v.get("type") or [])) and "enum" not in v]
        target = next((k for k in required if k in strings), None) or (strings[0] if len(strings) == 1 else None)
        if not target:
            return None
        if any(k != target for k in required):
            return None
        return {target: text}

    def message_limit(self) -> int:
        return self.message_budget if self.message_budget is not None else 3

    def kept_messages(self) -> int:
        """Messages still kept for the request's finishing rules."""
        return max(0, min(self.finish_reserved - self.finish_messages, self.message_limit() - self.messages_used))

    def question_limit(self) -> int:
        return max(0, self.message_limit() - max(0, self.finish_reserved - self.finish_messages))

    def finish_state(self) -> str:
        return f"{self.cart_changed_at}|{self.ledger_version}|{self.messages_used}|{self.refusals}|{self.budget_now}"

    def finish_pending(self) -> bool:
        state = self.finish_state()
        return bool(self.finish_steps) and state not in self.finish_clear and self.finish_rounds.get(state, 0) < 2

    def note_event(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """One line of the log the finish check reads."""
        roles = {v: k for k, v in self.roles.items()}
        role = roles.get(name, "tool")
        if role == "search":
            text = f"searched {_compact(args)[:120]}"
        elif role in ("details", "view"):
            text = f"opened {_compact(args)[:120]}"
        else:
            limit = 1200 if role == "cart_view" else 500      # a cart view's total can sit after its lines
            text = f"{role} {name} {_compact(args)[:300]} -> {_compact(_brief(observation, 200))[:limit]}"
        self.events.append(f"step {self.step}: {text}")

    def cart_now(self) -> str:
        """The cart's lines as last known (the cart view's lines when it was read after the last change), with each
        listing's latest price and the sum of price times quantity."""
        if self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at:
            lines = [(i, n) for i, n in self.cart_seen if n > 0]
        else:
            lines = [(i, self.cart_qty.get(i, 1)) for i in self.cart_ids]
        if not lines:
            return "empty (as far as the log shows)"
        parts: list[str] = []
        total: float | None = 0.0
        for ident, units in lines:
            listing = self.store.get(ident)
            price = listing.price if listing is not None else None
            parts.append(f"{ident} x{units}" + (f" at {price:g} each" if price is not None else " (price not seen)"))
            total = total + price * units if price is not None and total is not None else None
        text = "; ".join(parts)
        if total is not None:
            text += f"; sum of the latest prices times quantities: {total:g}"
        return text

    def finish_decision(self) -> dict[str, Any] | None:
        tool_lines = []
        for tool in self.tools:
            function = tool["function"]
            tool_lines.append(f"- {function['name']}: {function.get('description', '')} | arguments: {_compact(function.get('parameters') or {})}")
        prompt = FINISH_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nFINISHING RULES (read from the request):\n" + "\n".join(f"- {_compact(step)}" for step in self.finish_steps)
        prompt += "\n\nTOOLS:\n" + "\n".join(tool_lines)
        prompt += f"\n\nNOW: step {self.step} of {self.max_steps} ({self.steps_left()} left, one of them for the order); messages sent {self.messages_used} of {self.message_limit()}; the cart last changed at step {self.cart_changed_at}; the order tool is {self.roles.get('order')}."
        prompt += "\n\nCART NOW: " + self.cart_now()
        prompt += "\n\nLOG:\n" + "\n".join(self.events[-40:])
        # always the stronger model: one or two calls an episode, and a wrong call here loses the whole order
        message = self.infer([{"role": "user", "content": prompt}], None, [JUDGE_MODEL, READ_MODEL], 1500, NO_THINKING)
        result = _parse_json_object((message or {}).get("content") or "")
        if result is None and self.time_left() > 35:
            # an empty or broken answer would let the order through unchecked: ask once more for the text answer
            message = self.infer([{"role": "user", "content": prompt + "\n\nAnswer now with the JSON object as plain text."}], None, [READ_MODEL, JUDGE_MODEL], 1500, NO_THINKING)
            result = _parse_json_object((message or {}).get("content") or "")
        self.trace.append({"kind": "finish_check", "result": result})
        return result

    def run_finish(self, order_args: dict[str, Any] | None, final: bool = False) -> bool:
        """Carry out the request's finishing rules the LLM finds unmet, then order when nothing changed (always, when
        `final`: the episode is ending); False when nothing was done (the order then goes through as it is). A call that
        only prepares a rule (reading the cart for a figure) is followed by a fresh check on its result, not the order."""
        order = self.roles.get("order")
        if not order or self.finished or self.dead or self.steps_left() < 2 or self.time_left() < (20 if final else 30):
            return False
        done = False
        for _ in range(3):
            state = self.finish_state()
            self.finish_rounds[state] = self.finish_rounds.get(state, 0) + 1
            decision = self.finish_decision() or {}
            wanted = decision.get("next") if isinstance(decision.get("next"), dict) else None
            name = str((wanted or {}).get("tool") or "")
            args = (wanted or {}).get("arguments") if isinstance((wanted or {}).get("arguments"), dict) else {}
            blocked = {self.roles.get(r) for r in ("order", "decline", "cart", "search")} - {None}
            if wanted is None or name not in self.schemas or name in blocked:
                self.finish_clear.add(state)
                break
            args, problems = _check(args, self.schemas[name], "args")
            message = name == self.roles.get("message")
            if problems or (message and self.messages_used >= self.message_limit()):
                self.finish_clear.add(state)
                break
            version, budget, used = self.ledger_version, self.budget_now, self.messages_used
            call = {"id": f"finish_{self.step}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            self.messages.append({"role": "assistant", "content": "I will carry out the request's finishing rule before ordering.", "tool_calls": [call]})
            for call_id, text in self.run_calls([call], planned=True):
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
            done = True
            if message and self.messages_used > used:
                self.finish_messages += 1
            failed = isinstance(self.records[-1] if self.records else None, dict) and bool(self.records[-1].get("error"))
            if self.finished or failed or self.steps_left() < 1:
                return True
            # a message's reply is read by interpret(): it decides from what the reply changed; another tool's result
            # is looked at first when the check said it may change what to buy
            if not final and ((decision.get("then_order") is False and not message) or self.ledger_version != version
                              or self.budget_now != budget or self.pending_reply or self.last_unclear):
                return True         # the result must be looked at first: the model (and the next check) take it from here
            if message or self.steps_left() < 2 or self.time_left() < (20 if final else 30):
                break
        if not done:
            return False
        self.finish_clear.add(self.finish_state())
        args = dict(order_args) if isinstance(order_args, dict) else self.neutral_args(order, "Order the cart.")
        call = {"id": f"finish_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "The finishing rules are met; I will order the cart.", "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
        return True

    def facts_tool(self) -> str | None:
        """The tool that shows a listing's own property list when the page tool does not (race 11: InspectDetails gave
        the page, TechnicalData the properties): the view role, once a reply of it carried properties, or when the pages
        opened so far carried none."""
        view, details = self.roles.get("view"), self.roles.get("details")
        if not view or view == details or view in self.capped:
            return None
        if self.view_facts:
            return view
        paged = [self.store.listings[i] for i in self.paged if i in self.store.listings]
        return view if paged and not any(l.facts for l in paged) else None

    def gather_facts(self) -> None:
        """Before the questions are planned: the property lists of the first fitting rows of each requested kind (six for
        one product, four a product for several, at most 16), so the planner sees which properties the kind's listings
        carry. Race 11: the agents ahead read about 6 lists before the first question (13 for several products), Hashira 1,
        and its questions went to properties few listings carry (57% outside a kind's six most carried, DNA 36%). Race 12,
        several products: Hashira (three a product) about 7 lists and 52% of the needs told, the agents ahead about 15 and
        59-74%."""
        tool = self.facts_tool()
        if not tool or self.facts_gathered:
            return
        items = self.items()
        products = [name for name, _ in items] or [str(self.plan.get("product") or "")]
        want = min(16, 4 * len(items)) if items else 6
        left = max(0, self.question_limit() - self.messages_used)
        room = self.steps_left() - left - 14
        known = {l.ident for l in self.store.opened()}
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        entries = sorted(self.search_log.values(), key=lambda e: e["step"])
        per = max(2, want // max(1, len(products)))
        picks: list[str] = []
        found = 0
        for product in products:
            words = set(re.findall(r"[a-z]{3,}", product.lower()))
            have = 0
            found += 1 if any(not items or (words & set(re.findall(r"[a-z]{3,}", " ".join(str(v) for v in e["args"].values() if isinstance(v, str)).lower()))) for e in entries) else 0
            for entry in entries:
                query = " ".join(str(v) for v in entry["args"].values() if isinstance(v, str)).lower()
                if items and not (words & set(re.findall(r"[a-z]{3,}", query))):
                    continue
                for ident in [i for i in self.store.order if i in entry["ids"]]:
                    listing = self.store.listings.get(ident)
                    if listing is None or ident in picks or ident in self.chain_pages:
                        continue
                    if ident in known:
                        have += 1
                        continue
                    if listing.stock is False or (budget is not None and listing.price is not None and listing.price > budget):
                        continue
                    if have >= per:
                        break
                    picks.append(ident)
                    have += 1
        picks = picks[: max(0, min(want - len(known), room))]
        if len(picks) < 2 or (items and found < min(len(items), 3) and self.step < 14):
            return          # several products: wait until most of them were searched for
        self.facts_gathered = True
        calls = []
        for index, ident in enumerate(picks):
            args = self.text_args("view", ident) or self.neutral_args(tool, ident)
            calls.append({"id": f"facts_{self.step}_{index}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
        self.messages.append({"role": "assistant", "content": "I will read the property lists of the first fitting listings, to see which properties this kind of product carries.", "tool_calls": calls})
        for call_id, text in self.run_calls(calls, planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.question_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        if not self.asked and not self.finished and self.time_left() > WRAP_UP_SECONDS + 60:
            self.gather_facts()
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 4 and self.step < 8 and not urgent and not force:
            return False
        if self.items() and not urgent and not force and self.step < 14:
            # several products: the questions are planned from each product's own listings, so each kind is opened first
            kinds_opened = sum(1 for group in self.store.groups().values() if len(group) >= 2)
            if kinds_opened < min(len(self.items()), 3):
                return False
        if force:
            urgent = False
        # a later planning round (questions left over, or one by one for several products) waits for more listings
        # (or a commit attempt), so it is chosen knowing the earlier answers and more of the store
        if self.asked and not force and not urgent and not self.last_unclear and fact_sets < self.sets_at_last_ask + 2:
            return False
        if not force and fact_sets < self.plan_retry_at:
            return False
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        # one product: plan every remaining question at once, so the questions do not eat into the steps needed for
        # the search. Several products: one by one, so each is chosen knowing which products were already asked
        # about and what more of the store shows
        remaining = max(1, limit - self.messages_used)
        count = min(remaining, max(2, len(self.items()))) if self.items() else remaining
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        if self.hints:
            prompt += "\n\nTHE STORY POINTS AT (from a first reading of the request): " + "; ".join(self.hints)
        if SWEEP_LAST and limit >= 2 and self.messages_used + count == limit and not urgent:
            # the last message names a few more properties at once: the shopper answers about one of them when one is a
            # need, and no later question is lost if the answer is "don't mind"
            prompt += SWEEP_NOTE
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        items = self.items()
        if items:
            prompt += "\n\nTHE REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS (name the one each question is about): " + "; ".join(name for name, _ in items)
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        settled = [str(r.get("property")) for r in self.pointers if r.get("property")] + [line.split(":")[0] for line in self.resolved]
        if settled:
            prompt += "\n\nSETTLED BY A PAGE REFERENCE (never ask about these): " + "; ".join(settled)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings, example values):\n" + (menu or "(none seen yet)")
        result = self.judge(prompt, 2000, [PLAN_MODEL, READ_MODEL])
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        # the shopper answers only a question that names one property: an open question is never sent
        planned = [q for q in planned if re.search(r"\b(which|what|how)\b", str(q.get("question")), re.I)]
        # one property a question: a pair is cut to its first property (race 9, all agents: a question about one property
        # revealed a need 21% of the time, a pair 17%, and a pair came back "one at a time" 26% against 8%)
        for q in planned:
            parts = _prop_parts(str(q.get("property") or ""))
            noun = re.search(r"\blike the (.+?) to have\b", str(q.get("question") or ""), re.I)
            if len(parts) >= 2 and noun:
                q["property"], q["question"] = parts[0], f"Which {parts[0]} would you like the {noun.group(1)} to have?"
        planned = [q for q in planned if " or " not in str(q.get("property") or "")]
        # the price limit is asked only when the request states none; a property already asked (for that product) is not asked again
        if not any(re.search(r"spend|budget|price|cost|afford", str(a), re.I) for a in self.plan.get("ask_first") or []):
            planned = [q for q in planned if not re.search(r"\b(spend|budget|price|cost|afford|how much|dollars|cents|money)\b", str(q.get("question")) + " " + str(q.get("property")), re.I)]
        planned = [q for q in planned if not all(self.asked_before(str(q.get("product") or ""), part)
                                                 for part in _prop_parts(str(q.get("property") or "") or _question_property(str(q.get("question") or ""))))]
        # a property the request (or an earlier answer) already settles is never asked
        settled_props = [line.split(":")[0].strip().lower() for line in self.stated + self.revealed + self.resolved if ":" in line]
        def settled_for(q: dict[str, Any]) -> bool:
            product = str(q.get("product") or "").strip().lower()
            def settled_part(prop: str) -> bool:
                for line_prop in settled_props:
                    bare = line_prop[len(product):].strip() if product and line_prop.startswith(product) else line_prop
                    if prop and (prop == bare or prop == line_prop):
                        return True
                return False
            parts = _prop_parts((str(q.get("property") or "") or _question_property(str(q.get("question") or ""))).strip().lower())
            return bool(parts) and all(settled_part(part) for part in parts)
        planned = [q for q in planned if not settled_for(q)]
        if not self.items() and not urgent:
            # every story-hinted property of the menu goes before the planner's other picks
            hinted = self.hinted_properties(menu)
            def is_hinted(q: dict[str, Any]) -> bool:
                return any(_same_property(str(q.get("property") or "").lower(), h.lower()) for h in hinted)
            picked = [str(q.get("property") or "").lower() for q in planned]
            missing = [h for h in hinted if not any(_same_property(h.lower(), p) for p in picked)]
            if missing:
                noun = next((m.group(1) for q in planned for m in [re.search(r"\blike the (.+?) to have\b", str(q.get("question") or ""), re.I)] if m), None)
                noun = noun or str(self.plan.get("product") or "item").strip().lower()
                extra = [{"product": "", "property": h, "why": "story hint", "question": f"Which {h} would you like the {noun} to have?"} for h in missing]
                planned = [q for q in planned if is_hinted(q)] + extra + [q for q in planned if not is_hinted(q)]
                self.trace.append({"kind": "hinted_first", "added": missing})
        if urgent and fact_sets < 4:
            planned = planned[:1]
        # never send a question that has been sent before in the same words
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.questions_sent]
        if self.items():
            asked_per_product: dict[str, int] = {}
            for entry in self.asked:
                for product, _ in self.items():
                    if entry.lower().startswith("- " + product.lower()):
                        asked_per_product[product] = asked_per_product.get(product, 0) + 1
            def product_load(q: dict[str, Any]) -> tuple[bool, int]:
                name = str(q.get("product") or "").lower()
                told = any(_kind_words(name) & _kind_words(line.split(":")[0]) for line in self.revealed)
                return told, next((n for product, n in asked_per_product.items() if product.lower() in name or name in product.lower()), 0)
            planned = sorted(planned, key=product_load)
        planned = planned[:count]
        calls = []
        labels = []
        for index, item in enumerate(planned):
            args = self.message_args(str(item.get("question")).strip())
            if not args:
                continue
            args, problems = _check(args, self.schemas[tool], "args")
            if problems:
                continue
            calls.append({"id": f"ask_{self.messages_used + index + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}})
            product = str(item.get("product") or "").strip()
            labels.append(((product + " " if product and items else "") + str(item.get("property")), str(item.get("question")).strip()))
        if not calls:
            # a planning call that gave nothing usable is retried once, after another listing has been opened
            self.plan_misses += 1
            self.plan_retry_at = fact_sets + 1
            if self.plan_misses >= 2:
                self.planner_on = False
            return False
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + ", then ".join(l[0] for l in labels) + ".", "tool_calls": calls})
        self.last_unclear = False      # used up by this round; the new answers set it again
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if index < len(labels):
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {text.split(chr(10) + '[status]')[0][:400]}")
                for part in _prop_parts(str(planned[index].get("property") or "").strip().lower() or _question_property(labels[index][1])):
                    self.asked_props.append((str(planned[index].get("product") or "").strip().lower(), part))
        if self.reask and self.asked:
            self.asked.pop()
            last = planned[min(len(calls), len(planned)) - 1] if planned else {}
            for _ in _prop_parts(str(last.get("property") or "").strip().lower()) or [None]:
                if self.asked_props:
                    self.asked_props.pop()
        self.reask = False
        self.sets_at_last_ask = fact_sets
        self.planner_on = self.messages_used < limit
        return True

    def hinted_properties(self, menu: str) -> list[str]:
        """Menu properties that share a word with the story hints (a hinted property is a need about three times as
        often as another one), in menu order; the product's own name and settled properties do not count."""
        if not self.hints:
            return []
        def words(text: str) -> set[str]:
            out = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}
            return out | {w[:-3] + "or" for w in out if w.endswith("our") and len(w) > 5}
        # "how it must look: colour, pattern or finish": the words after the colon name the properties
        hint_words = words(" ".join(h.split(":", 1)[1] if ":" in h else h for h in self.hints))
        product_words = words(str(self.plan.get("product") or ""))
        product_words |= {w.rstrip("s") for w in product_words} | {w + "s" for w in product_words}
        settled = [line.split(":")[0].strip().lower() for line in self.stated + self.revealed + self.resolved] + [f.lower() for f in self.free]
        settled += [str(r.get("property") or "").lower() for r in self.pointers]
        out = []
        for name in re.findall(r"^- (.+?) \(in \d+ listing", menu, re.M):
            name = name.strip()
            own = words(name) - product_words
            if not own or not (own & hint_words):
                continue
            if any(_same_property(name.lower(), s) for s in settled if s) or self.asked_before("", name):
                continue
            out.append(name)
        return out

    def restates_kind(self, requirement: str) -> bool:
        """Whether a requirement's value is only a word of the requested product's own name ("frame: picture"
        for picture frames): the category already covers it."""
        product_name = str(self.plan.get("product") or "").lower()
        if not product_name or self.items() or re.search(r"&|\band\b|\bor\b", product_name):
            return False
        prop, _, value = requirement.partition(":")
        value_words = re.findall(r"[a-z0-9]+", value.lower())
        query_text = str(self.policy.get("query") or "")
        said = any(re.search(r"\b" + re.escape(w), query_text, re.I) for w in re.findall(r"[a-z0-9]+", prop.lower()) if len(w) > 2)
        return (len(value_words) == 1 and value_words[0] in set(re.findall(r"[a-z0-9]+", product_name))
                and not re.search(r"\d", value) and not said)

    def asked_before(self, product: str, prop: str) -> bool:
        product, prop = product.strip().lower(), prop.strip().lower()
        for p, pr in self.asked_props:
            if self.items() and not _same_product(product, p):
                continue        # several products: the same property of another product is a new question
            if p == "" and product and self.items():
                continue        # the earlier question named no product: asking again with the product is allowed
            if _same_property(prop, pr):
                return True
        return False

    def track_search(self, args: dict[str, Any], observation: Any) -> str:
        """Remember how much of a search's matches have been seen; a note when pages remain unread."""
        name = self.roles.get("search") or ""
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        paging = {k for k, v in properties.items() if isinstance(v, dict) and re.search(r"follow|next|token|cursor|after|marker|more|page|offset|start", k, re.I)}
        base = {k: v for k, v in args.items() if k not in paging}
        key = _compact(base)
        entry = self.search_log.setdefault(key, {"args": base, "count": None, "ids": set(), "next": None, "step": self.step})
        entry["step"] = self.step
        count = _match_count(observation)
        if count is not None:
            entry["count"] = count
        rows = [ident for ident, _ in _rows_with_ids(observation)]
        for ident in rows:
            entry["ids"].add(ident)
        entry["size"] = max(entry.get("size") or 0, len(rows))
        if len(rows) >= 4:
            self.order_votes[0 if rows == sorted(rows) else 1] += 1
        text_key = _text_key(properties, [k for k in base if isinstance(properties.get(k), dict) and properties[k].get("type") == "string" and "enum" not in properties[k]]) or ""
        entry["next"] = _next_page(observation, schema, text_key, args)
        seen = len(entry["ids"])
        widened = ""
        words = set(str(base.get(text_key) or "").lower().split())
        if text_key and entry["count"] is not None:
            for other in self.search_log.values():
                other_words = set(str(other["args"].get(text_key) or "").lower().split())
                if other is not entry and other_words and other_words < words and other.get("count") is not None and other["count"] < entry["count"]:
                    widened = " More words found more matches here, so this search matches ANY of its words: extra words widen it; use the fewest, most exact words."
                    self.word_rule = "any"
                    break
                if other is not entry and other_words and other_words < words and other.get("count") is not None and other["count"] > entry["count"]:
                    self.word_rule = "all"      # more words, fewer matches: every word must be found
        if entry["count"] is not None and entry["count"] > seen and entry["next"]:
            if self.row_order() == "ranked":
                return (f"This search reports {entry['count']} matches and shows {seen}; its rows are ranked (best matches first), and fitting rows, cheaper ones too, often "
                        f"continue on the next pages. Read on with {json.dumps(dict(base, **entry['next']))} while its rows still look like the requested kind.{widened}")
            if entry["count"] <= PAGE_SWEEP:
                return (f"This search reports {entry['count']} matches and {seen} have been shown; results come in listing-id order, not by relevance, so the "
                        f"unseen ones may fit just as well. Read the remaining {entry['count'] - seen} with {json.dumps(dict(base, **entry['next']))} (and the pages after it) before choosing or declining.{widened}")
            how = ("every word must appear in a listing's title or description, which rarely repeat a property's name, so add one required value's own words "
                   "as a listing would write them (the kind's word plus the value, never the property's name; a search that finds nothing has a word no listing writes)"
                   if self.word_rule == "all" else "add words from the requirements or the revealed needs, a category or a price filter")
            return (f"This search reports {entry['count']} matches and shows {seen}; the rows are a sample in listing-id order, not the best fits. Narrow it "
                    f"({how}) until the count is small enough to read through, or page on with {json.dumps(entry['next'])}.{widened}")
        return widened.strip()

    def sweep_cheapest(self, floor: int = 9) -> str:
        """For the cheapest: read the unread pages of the searches whose rows hold the pick (the cart's line, else the
        listings opened so far), so every price of the set is known (a store that lists rows by id or by relevance hides
        cheaper fits on later pages). The rows go to the store's index; the conversation gets one short line per page.
        Returns the pick the cheaper rows are measured against ('' when nothing was read)."""
        search = self.roles.get("search")
        picks = list(self.cart_ids) or [l.ident for l in self.store.listings.values() if (l.facts or l.ident in self.paged) and l.price is not None]
        if not search or not picks:
            return ""
        read = False
        swept: set[str] = set()
        for entry in sorted(self.search_log.values(), key=lambda e: -e["step"]):
            ids = entry.get("ids") or set()
            count = entry.get("count")
            if not entry.get("next") or not any(i in ids for i in picks) or (count is not None and len(ids) >= count):
                continue
            swept |= ids
            if isinstance(count, int) and count - len(ids) > SWEEP_PAGES * max(1, entry.get("size") or 1):
                continue            # more than the sweep can read: the narrowed searches (value_sweep) take it
            pages = 0
            while entry.get("next") and pages < SWEEP_PAGES and self.steps_left() > floor and self.time_left() > WRAP_UP_SECONDS + 20:
                args = dict(entry["args"], **entry["next"])
                call = {"id": f"sweep_{self.step}", "type": "function", "function": {"name": search, "arguments": json.dumps(args)}}
                self.messages.append({"role": "assistant", "content": "I will read the next page of this search, so that no cheaper fit is missed.", "tool_calls": [call]})
                replies = self.run_calls([call], planned=True)
                result = self.records[-1] if self.records else None
                self.track_search(args, result)
                rows = _rows_with_ids(result)
                listed = [self.store.get(i) for i, _ in rows]
                text = "; ".join(f"{l.ident} {l.price:g}" + (" (out of stock)" if l.stock is False else "") for l in listed if l is not None and l.price is not None)
                for call_id, reply in replies:
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": (f"{len(rows)} more rows (id price): {text}" if rows else reply[:600])})
                pages += 1
                read = True
                if not rows:
                    break
        read = self.value_sweep(picks, floor) or read
        if not read:
            return ""
        if self.cart_ids:
            return self.cart_ids[-1]
        # no cart yet: measured against the cheapest listing opened from those searches
        mine = [self.store.get(i) for i in picks if i in swept]
        mine = [l for l in mine if l is not None and l.price is not None and l.stock is not False]
        return min(mine, key=lambda l: l.price).ident if mine else ""

    def value_sweep(self, picks: list[str], floor: int) -> bool:
        """An all-words store whose search for the pick holds more rows than the sweep can read: up to two narrowed
        searches, the search's own words plus one required value's words (as a listing's text would carry them), each
        read to its end when it is small enough. The cheapest fit then often shows among their rows."""
        search = self.roles.get("search")
        if self.word_rule == "any" or not search or self.value_swept:
            return False
        self.value_swept = True         # once per episode: each narrowed search costs a step
        schema = self.schemas.get(search) or {}
        properties = schema.get("properties") or {}
        homes = [e for e in self.search_log.values() if e.get("next") and any(i in (e.get("ids") or ()) for i in picks)
                 and isinstance(e.get("count"), int) and e["count"] > len(e.get("ids") or ()) ]
        if not homes:
            return False
        home = max(homes, key=lambda e: e["count"])
        text_key = _text_key(properties, [k for k in home["args"] if isinstance(properties.get(k), dict) and properties[k].get("type") == "string" and "enum" not in properties[k]]) or ""
        base = str(home["args"].get(text_key) or "").strip()
        if not text_key or not base:
            return False
        values = []
        for requirement in [r for r in self.stated + self.revealed if r not in self.dropped]:
            value = requirement.partition(":")[2]
            if re.search(r"\b(not|no|never|anything but|except|other than|avoid|without|exclud)", value, re.I) or "price" in requirement.partition(":")[0].lower():
                continue            # an exclusion (or a price ceiling) is no word a fitting listing writes
            value = re.sub(r"\b(at (most|least)|no (more|less) than|or (less|more|fewer|higher|lower)|up to|under|over|only|exactly|any|either)\b", " ", value, flags=re.I)
            words = [w for w in re.split(r"[^A-Za-z0-9]+", value) if len(w) >= 2 and w.lower() not in base.lower().split()]
            if words and len(words) <= 3 and "page" not in requirement.lower():
                values.append(" ".join(words))
        read = False
        for value in values[:2]:
            if self.steps_left() <= floor + 1 or self.time_left() < WRAP_UP_SECONDS + 20:
                break
            args = dict(home["args"], **{text_key: f"{base} {value}"})
            pages = 0
            while args and pages < 3 and self.steps_left() > floor:
                call = {"id": f"value_{self.step}", "type": "function", "function": {"name": search, "arguments": json.dumps(args)}}
                self.messages.append({"role": "assistant", "content": f"I will search with a required value ({value}) so that no cheaper fit is missed.", "tool_calls": [call]})
                replies = self.run_calls([call], planned=True)
                result = self.records[-1] if self.records else None
                self.track_search(args, result)
                rows = _rows_with_ids(result)
                listed = [self.store.get(i) for i, _ in rows]
                text = "; ".join(f"{l.ident} {l.price:g}" + (" (out of stock)" if l.stock is False else "") for l in listed if l is not None and l.price is not None)
                for call_id, reply in replies:
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": (f"{_match_count(result)} matches; rows (id price): {text}" if rows else reply[:400])})
                read = read or bool(rows)
                count = _match_count(result)
                nxt = _next_page(result, schema, text_key, args)
                if not rows or not nxt or (isinstance(count, int) and count > 4 * max(1, len(rows))):
                    break           # nothing, the end, or still too many to read through
                args = dict(args, **nxt)
                pages += 1
        return read

    def row_order(self) -> str:
        """How this store lists search rows, as seen so far: "id" (listing-id order), "ranked" or "" (not known)."""
        in_order, other = self.order_votes
        if in_order >= 2 and in_order > other:
            return "id"
        if other >= 2 and other > in_order:
            return "ranked"
        return ""

    def ranked_unread(self) -> list[str]:
        """Ranked stores: the searches that found a carted listing, read less than three pages deep (at most two)."""
        if self.row_order() != "ranked":
            return []
        out = []
        for entry in sorted(self.search_log.values(), key=lambda e: -e["step"]):
            ids = entry.get("ids") or set()
            if entry.get("next") and len(ids) < 30 and any(i in ids for i in self.cart_ids):
                out.append(f"{json.dumps(entry['args'])} ({len(ids)} rows seen; next page: {json.dumps(dict(entry['args'], **entry['next']))})")
        return out[:2]

    def unread_searches(self) -> list[str]:
        """Searches with a small set of matches that was not read to its end (newest first, at most three)."""
        out = []
        for entry in sorted(self.search_log.values(), key=lambda e: -e["step"]):
            count, seen = entry.get("count"), len(entry.get("ids") or ())
            if count is None or not entry.get("next") or seen >= count or count > PAGE_SWEEP or count <= 8:
                continue
            out.append(f"{json.dumps(entry['args'])} ({count} matches, {seen} seen; next page: {json.dumps(dict(entry['args'], **entry['next']))})")
        return out[:3]

    def order_note(self) -> str:
        if self.row_order() == "ranked":
            return "Results are ranked, best matches first, yet fitting rows continue on later pages"
        return "Results come in listing-id order"

    def send_reply(self, text: str) -> bool:
        tool = self.roles.get("message")
        limit = self.question_limit()
        if not tool or self.messages_used >= limit or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS:
            return False
        args = self.message_args(text)
        if not args:
            return False
        args, problems = _check(args, self.schemas[tool], "args")
        if problems:
            return False
        call = {"id": f"reply_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, reply_text) in enumerate(replies):
            if index == len(replies) - 1:
                reply_text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": reply_text})
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
            return ""   # a line already held: the store may add to it or set it; the cart reply is checked instead
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if self.items():
                    allowed = {q for _, q in self.items()}
                    if isinstance(value, int) and value not in allowed:
                        return f"the request needs these quantities: {', '.join(f'{n} x{q}' for n, q in self.items())}, but this call asks for {value}."
                    return ""
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}" + (", which takes it out of the cart." if value == 0 else ".")
        return ""

    def line_gate(self) -> str:
        """Before an order: every cart line was opened with the details tool; one product means one line; several
        different products mean one line per product at its quantity."""
        items = self.items()
        key = f"{self.cart_changed_at}|{self.cart_seen_at}"
        held_ids = [i for i, n in self.cart_seen if n > 0] if self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at else list(self.cart_ids)
        details, side = self.roles.get("details"), self.compare_tool()
        view = self.roles.get("view") if self.roles.get("view") != details else None
        unopened = [i for i in held_ids if i in self.store.listings and not self.store.listings[i].facts]
        # a property list or a search row is not the listing's page, and whether each bought line's page was opened
        # can be graded
        unpaged = self.unpaged_lines()
        if unpaged and not unopened and ("open|" + key) not in self.line_blocks and self.steps_left() > 3:
            # only the pages are missing: opened in code, no turn of the model needed
            self.line_blocks.add("open|" + key)
            self.page_next = unpaged
            return f"The page of {', '.join(unpaged[:4])} was never opened (a property list or a search row does not count as opening a listing); it is opened for you now. Check that each line still meets every requirement in force, then call the order tool again."
        if (unopened or unpaged) and ("open|" + key) not in self.line_blocks and self.steps_left() > 3:
            self.line_blocks.add("open|" + key)
            parts = []
            if unopened:
                parts.append(f"the own properties of {', '.join(unopened[:3])} were never shown (open them with {view or details or 'the details tool'})")
            if unpaged:
                how = f"{details}" + (f" (or {side[0]}, several in one call)" if side else "")
                parts.append(f"the page of {', '.join(unpaged[:4])} was never opened with {how}: a property list or a search row does not count as opening a listing")
            return "Before ordering: " + "; ".join(parts) + ". Do that (several calls in one reply), check that each line still meets every requirement in force, then order."
        if not items:
            if len(held_ids) >= 2 and ("single|" + key) not in self.line_blocks and self.steps_left() > 3:
                self.line_blocks.add("single|" + key)
                return f"The request asks for one product, but the cart holds {len(held_ids)} lines ({', '.join(held_ids[:4])}). Keep only the chosen listing at the required quantity: take the other line(s) out of the cart, then order."
            return ""
        if key in self.line_blocks:
            return ""
        view = self.roles.get("cart_view")
        if self.cart_seen is None or self.cart_seen_at < self.cart_changed_at:
            if view and self.cart_view_at < self.cart_changed_at and self.steps_left() > 3:
                self.line_blocks.add(key)
                self.cart_view_at = self.cart_changed_at
                return f"The request asks for {len(items)} different products. View the cart first, check that it holds exactly one line per product at the quantities asked ({', '.join(f'{n} x{q}' for n, q in items)}) and nothing else, then order."
            return ""
        wanted = sorted(q for _, q in items)
        held = sorted(n for _, n in self.cart_seen if n > 0)
        if len(held) == len(wanted) and held == wanted:
            return ""
        self.line_blocks.add(key)
        return (f"The cart does not match the request: it holds {len(held)} line(s) with units {held}, but the request needs {len(items)} lines with units {wanted} "
                f"({', '.join(f'{n} x{q}' for n, q in items)}). Add the missing product(s) (one verified fit per product, at its quantity), remove any extra line, then order. If a product truly cannot be found, decline as the request says.")

    def tie_rules(self) -> list[tuple[str, dict[str, Any]]]:
        """The rules that tie requested products together, numbered from 1 in this order: fit rules, then conditions."""
        return [("fit", r) for r in self.fit_rules] + [("if", c) for c in self.conditions]

    def dependency_gate(self) -> str:
        """Before an order of several products under a fit rule or a conditional requirement: one check of the cart's
        lines against those rules, read from the pages. A broken rule holds the order; the fitting listings seen are
        named, and a search for more is run before the next turn."""
        rules = self.tie_rules()
        if not rules or not self.items() or self.dep_checks >= 3 or self.cost >= COST_HARD:
            return ""
        held = self.held_lines()
        key = "dep|" + ",".join(sorted(held)) + f"|{self.ledger_version}"
        if len(held) < 2 or key in self.line_blocks or self.steps_left() <= 6 or self.time_left() < 45:
            return ""
        self.line_blocks.add(key)
        listed = []
        for number, (kind, rule) in enumerate(rules, 1):
            if kind == "fit":
                text = f"{number}. fit rule: the {rule['accessory']} must fit the {rule['anchor']} that is bought"
            else:
                text = (f"{number}. conditional requirement: if the {rule['if_product']} has {_cond_test(rule)}, "
                        f"the {rule['then_product']} must have {rule['then_requirement']}")
            listed.append(text + (f' ("{rule["text"]}")' if rule.get("text") else ""))
        names = [str(r.get("accessory") if k == "fit" else r.get("then_product")) for k, r in rules]
        others = [l for l in self.store.listings.values() if l.ident not in held and (l.facts or l.ident in self.paged)
                  and (not l.category or any(_same_product(l.category, n) for n in names))]
        others.sort(key=lambda l: (l.price is None, l.price or 0))
        prompt = DEPEND_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nTHE RULES:\n" + "\n".join(listed)
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nTHE CART'S LINES (their pages):\n" + "\n".join(self.store.pages(held, 2500))
        if others:
            prompt += "\n\nOTHER LISTINGS OPENED:\n" + "\n".join(l.page(650) for l in others[:14])
        result = self.judge(prompt, 1800) or {}
        self.dep_checks += 1
        self.trace.append({"kind": "depend", "result": result})
        failing = []
        changed = False
        for entry in result.get("rules") or []:
            if not isinstance(entry, dict):
                continue
            try:
                number = int(str(entry.get("rule")).strip().rstrip("."))
            except ValueError:
                continue
            if not 1 <= number <= len(rules):
                continue
            kind, rule = rules[number - 1]
            if kind == "if" and str(entry.get("anchor") or "") in held:
                state = (entry.get("applies") is not False, str(entry.get("anchor")))
                if self.cond_state.get(number) != state:
                    self.cond_state[number] = state
                    changed = True
            if entry.get("applies") is not False and entry.get("holds") is False:
                failing.append((kind, rule, entry))
        if changed:
            self.ledger_version += 1
            self.line_blocks.add("dep|" + ",".join(sorted(held)) + f"|{self.ledger_version}")    # this cart is judged
        see = [str(x).strip() for x in result.get("see_also") or [] if str(x).strip() and str(x).strip().lower() not in self.see_read][:3]
        if see:
            # a cart line's page sends the reader on: those pages are read before the order, and the cart is judged again
            self.see_read.update(x.lower() for x in see)
            self.dep_next = [(code, [code]) for code in see]
            self.line_blocks.discard(key)
            return "The pages of the cart's lines send the reader on to: " + ", ".join(see) + ". Those pages are looked up for you next; order again after that (or change the line if they show it does not fit)."
        if not failing:
            return ""
        parts = []
        jobs: list[tuple[str, list[str]]] = []
        for kind, rule, entry in failing[:3]:
            dep = entry.get("dependent")
            name = rule.get("accessory") if kind == "fit" else rule.get("then_product")
            link = [str(x) for x in entry.get("link") or [] if str(x).strip()][:4]
            text = f"the {name} line {dep}" if dep else f"no {name} line"
            if kind == "fit":
                text += f" does not fit the {rule['anchor']} line {entry.get('anchor')}"
            else:
                text += (f" does not meet {rule['then_requirement']}, which is in force because the {rule['if_product']} line {entry.get('anchor')} "
                         f"has {_cond_test(rule)}")
            if link:
                text += f" (a fitting one names or has: {', '.join(link)})"
            fitting = [str(i) for i in entry.get("fitting") or [] if str(i) in self.store.listings and str(i) not in held][:4]
            if fitting:
                text += "; opened listings that meet it: " + ", ".join(
                    f"{i} ({self.store.listings[i].price:g})" if self.store.listings[i].price is not None else i for i in fitting)
            parts.append(text)
            words = str(entry.get("search") or "").strip()
            if words or link:
                jobs.append((words, link))
        self.dep_next = jobs[:2]
        pick = "the cheapest listing" if self.plan.get("priority") == "cheapest" else "a listing"
        return ("A check of the rules that tie the products together found: " + "; ".join(parts) + ". "
                + ("A search for more listings that meet it is run for you next, and its likeliest matches are opened. " if self.dep_next else "")
                + f"Replace that line with {pick} that meets the rule and every other requirement of its product (its page opened), keep the other lines, then order.")

    def dependency_search(self, jobs: list[tuple[str, list[str]]]) -> None:
        """The searches a broken tie rule asks for: each one run, and its unopened rows that name a linking identifier
        (then the cheapest in stock) opened, so the model can choose among their pages."""
        search = self.roles.get("search")
        opened: list[str] = []
        see_also = all(link == [words] for words, link in jobs)      # pages a cart line sends the reader on to
        for words, link in jobs:
            if not search or self.steps_left() <= 7 or self.time_left() < 40 or self.cost >= COST_HARD:
                break
            queries = [words] if words else []
            if link:
                queries.append(link[0])
            rows: list[str] = []
            for query in queries[:2]:
                args = self.text_args("search", query)
                if not args or self.steps_left() <= 7:
                    break
                call = {"id": f"tie_search_{self.step}", "type": "function", "function": {"name": search, "arguments": json.dumps(args)}}
                self.messages.append({"role": "assistant", "content": f"I will search for listings that meet the rule ({query}).", "tool_calls": [call]})
                replies = self.run_calls([call], planned=True)
                result = self.records[-1] if self.records else None
                self.track_search(args, result)
                for call_id, text in replies:
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _resolver_text(text)})
                rows = [i for i, _ in _rows_with_ids(result)]
                if rows:
                    break
            tokens = [re.sub(r"[^a-z0-9]", "", t.lower()) for t in link if len(re.sub(r"[^a-z0-9]", "", t.lower())) >= 2]

            def rank(ident: str) -> tuple[bool, float]:
                listing = self.store.get(ident)
                flat = re.sub(r"[^a-z0-9]+", " ", " ".join([listing.title, *listing.texts, *listing.facts.values()]).lower()) if listing else ""
                words, squashed = set(flat.split()), flat.replace(" ", "")
                # a short code must stand as a word of its own ("ZE2" in "[ZE2-98960]"); a long one may be written apart ("DRZ 8040")
                named = any(t in words or (len(t) >= 5 and t in squashed) for t in tokens)
                return (not named, listing.price if listing and listing.price is not None else 1e12)
            fresh = [i for i in rows if i not in self.paged and i not in opened and (self.store.get(i) is None or self.store.get(i).stock is not False)]
            fresh.sort(key=rank)
            if see_also:
                fresh = [i for i in fresh if not rank(i)[0]]
            batch = fresh[: max(1, self.compare_cap) if self.compare_tool() else 2]
            if batch and self.steps_left() > 7:
                self.open_pages(batch)
                opened += batch
        if opened and see_also:
            self.messages.append({"role": "user", "content": "[rules] The pages of " + ", ".join(opened) + ", which the cart's lines send the reader on to, were opened above. If they show that a line does not fit, replace it; otherwise order.\n" + self.status()})
        elif opened:
            self.messages.append({"role": "user", "content": "[rules] The pages of " + ", ".join(opened) + " were opened above. Put the one that meets the rule and every other requirement of its product in the cart in place of the line that broke it"
                                  + (" (the cheapest such one)" if self.plan.get("priority") == "cheapest" else "") + ", keep the other lines, then order.\n" + self.status()})

    def over_units(self) -> str:
        """A cart line holding more units than the request asks of any product (a store whose cart tool adds units)."""
        items = self.items()
        wanted = max((q for _, q in items), default=0) if items else self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        over = [(i, n) for i, n in (self.cart_seen or []) if n > wanted]
        if not over:
            return ""
        return ("The cart now holds " + ", ".join(f"{n} units of {i}" for i, n in over) + f", more than the {wanted} the request asks for"
                + (" of any product" if items else "") + " (a cart call may add the units it sends to what the line already holds). Bring each line to exactly"
                + " its product's quantity with the cart tool (as its schema says: taking the extra units out, or setting the line's units).")

    def held_lines(self) -> list[str]:
        if self.cart_seen is not None and self.cart_seen_at >= self.cart_changed_at:
            return [i for i, n in self.cart_seen if n > 0]
        return list(self.cart_ids)

    def unpaged_lines(self) -> list[str]:
        """Cart lines whose page was never opened with the details or the side-by-side tool."""
        return [i for i in self.held_lines() if i not in self.paged] if self.roles.get("details") else []

    def open_pages(self, ids: list[str]) -> None:
        """Open these listings' pages (side by side when the store can, else one by one)."""
        details, side = self.roles.get("details"), self.compare_tool()
        queue = list(dict.fromkeys(ids))
        tries = 0
        while queue and tries < len(ids) + 2:
            tries += 1
            size = max(1, self.compare_cap) if side and len(queue) > 1 else 1
            batch, queue = queue[:size], queue[size:]
            if self.steps_left() < 3 or self.finished or self.dead:
                return
            if side and len(batch) > 1:
                calls = [{"id": f"page_{self.step}", "type": "function", "function": {"name": side[0], "arguments": json.dumps({side[1]: batch})}}]
            else:
                args = self.text_args("details", batch[0]) or self.neutral_args(details or "", batch[0])
                calls = [{"id": f"page_{self.step}", "type": "function", "function": {"name": details, "arguments": json.dumps(args)}}]
            self.messages.append({"role": "assistant", "content": "I will open the pages of the cart's lines.", "tool_calls": calls})
            cap_before = self.compare_cap
            for call_id, text in self.run_calls(calls, planned=True):
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
            if len(batch) > 1 and self.compare_cap < cap_before:
                # the tool answered fewer than it was sent: the ones it left out go again, in smaller batches
                queue = [i for i in batch if i not in self.paged] + queue

    def unchecked_cheaper(self) -> list[tuple[str, Any]]:
        """For a cheapest request: listings priced below the item in the cart, of its kind, not shown out of stock or over
        the budget, whose details were never opened (at most five, cheapest first)."""
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        if not self.items():
            return self.store.cheaper(self.cart_ids[-1], budget, quantity, self.opened, self.cart_ids)
        # several products, cheapest total: each line's own kind is checked (the budget covers the whole order)
        found: list[tuple[str, Any]] = []
        for ident in self.cart_ids:
            for other, price in self.store.cheaper(ident, None, 1, self.opened, self.cart_ids)[:2]:
                if all(other != f for f, _ in found):
                    found.append((other, price))
        return found[:6]

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this finish call ends without buying. A store may use one tool for both finishes, told apart by an
        argument value (for example a mode of "refuse" against "commit")."""
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        return any(isinstance(v, str) and "enum" in (properties.get(k) or {}) and _DECLINE_WORDS.search(v) for k, v in args.items())

    def audit_decline(self) -> str:
        if self.time_left() < WRAP_UP_SECONDS + 30:
            return ""
        self.audited = True
        blocks: list[str] = []
        total = 0
        for record in reversed(self.records):
            text = _compact(_brief(record, 160))
            if total + len(text) > 16000:
                continue
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(blocks)
        result = self.judge(prompt, 1200)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        dear = [str(x) for x in (result or {}).get("over_budget_fits") or [] if isinstance(x, (str, int)) and str(x) in self.store.listings]
        if (result or {}).get("budget_only") is True and dear and self.budget_now is not None:
            # every fitting listing (or cart) costs more than the budget: the decline goes ahead
            self.decline_ok = True
            return ""
        if not fits:
            return ""
        text = "An independent check finds these listings fit, or come closest: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". A decline scores 0 in nearly every task, so check them (details, stock, price), put the best one (one per requested product) in the cart and order; decline only if each one's own records clearly contradict a requirement, the budget or stock."
        if self.order_attempts:
            text += " Nothing has been bought yet: an order that goes through ends the episode, and this one is still running, so declining now would leave the shopper with nothing."
        return text

    def neutral_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = properties.get(key) or {}
            if "enum" in spec and spec["enum"]:
                # a finish tool shared by both finishes: pick the mode that matches the intent
                fitting = [v for v in spec["enum"] if isinstance(v, str) and bool(_DECLINE_WORDS.search(v)) == declining]
                args[key] = (fitting or spec["enum"])[0]
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        for key, spec in properties.items():
            if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec and len(properties) == 1:
                args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        if self.plan.get("decline_no_names") is False:
            return args
        dirty = False
        for key, value in args.items():
            if isinstance(value, str):
                for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value):
                    if token and token in self.seen_ids:
                        dirty = True
        if self.plan.get("decline_no_names") or dirty:
            neutral = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."
            cleaned = {}
            properties = (self.schemas.get(name) or {}).get("properties") or {}
            for key, value in args.items():
                spec = properties.get(key) or {}
                if isinstance(value, str) and "enum" not in spec:
                    cleaned[key] = neutral[: spec.get("maxLength", len(neutral))]
                else:
                    cleaned[key] = value
            return cleaned
        return args

    def cart_args(self, ident: str, units: int) -> dict[str, Any] | None:
        """Arguments for the cart tool that put `units` of listing `ident` in the cart."""
        name = self.roles.get("cart")
        schema = self.schemas.get(name or "") or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key, spec in properties.items():
            if not isinstance(spec, dict):
                continue
            if "enum" in spec and spec["enum"]:
                values = [v for v in spec["enum"] if isinstance(v, str)]
                adding = [v for v in values if re.search(r"add|plus|put|set|insert|increase|more|include|place|keep|hold", v, re.I)
                          and not re.search(r"remov|delet|drop|discard|clear|minus|decreas|subtract|less|unhold|release|out", v, re.I)]
                safe = [v for v in values if not _DECLINE_WORDS.search(v)
                        and not re.search(r"remov|delet|drop|discard|clear|minus|decreas|subtract|less|unhold|release|out", v, re.I)]
                if key in (schema.get("required") or []) or adding:
                    args[key] = (adding or safe or values or spec["enum"])[0]
            elif spec.get("type") == "integer":
                args[key] = units
            elif spec.get("type") == "string" and key not in args:
                args[key] = ident
            elif spec.get("type") == "array":
                item = spec.get("items") or {}
                if item.get("type") == "object":
                    fields = item.get("properties") or {}
                    row = {k: (units if (v or {}).get("type") == "integer" else ident) for k, v in fields.items() if isinstance(v, dict)}
                    args[key] = [row]
                else:
                    args[key] = [ident]
        args, problems = _check(args, schema, "args")
        return None if problems else args

    def last_commit(self) -> bool:
        """One product, nothing in the cart, the episode about to end: order the closest opened fit."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        if self.items():
            return self.commit_items()
        if self.cart_ids or self.finished or self.dead or not cart or not order:
            return False
        if not self.may_commit() or self.steps_left() < 2 or self.time_left() < 12:
            return False
        self.committed = True
        self.commits += 1
        self.commit_at = self.step
        # a listing whose order was refused (sold out, repriced, over a new budget) is not picked again
        priced = [l for l in self.store.opened() if l.price is not None and l.ident not in self.refused_ids]
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        opened = [l for l in priced if l.stock is not False and (budget is None or l.price * quantity <= budget)]
        closest = bool(self.plan.get("closest_if_none"))
        if closest and not opened:
            opened = priced     # stock and budget are requirements like the others: the closest may miss one of them
        if not opened:
            return False
        prompt = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
        prompt += "\n\nLISTING RECORDS SEEN:\n" + "\n".join(self.store.pages([l.ident for l in opened][-14:], 900))
        result = self.judge(prompt, 900) or {}
        ids = {l.ident for l in opened}
        pick = next((str(x) for x in result.get("possible_fits") or [] if str(x) in ids), None)
        if closest and not pick:
            pick = min(opened, key=lambda l: l.price).ident
        self.trace.append({"kind": "last_commit", "pick": pick, "result": result})
        if not pick:
            return False
        args = self.cart_args(pick, quantity)
        if not args:
            return False
        self.forcing = True
        try:
            return self._commit_calls(cart, order, args)
        finally:
            self.forcing = False

    def basket_tool(self) -> bool:
        """Whether the cart tool takes the whole basket as a list of lines (a call then replaces what the cart held)."""
        properties = (self.schemas.get(self.roles.get("cart") or "") or {}).get("properties") or {}
        return any(isinstance(v, dict) and v.get("type") == "array" and (v.get("items") or {}).get("type") == "object" for v in properties.values())

    def may_commit(self) -> bool:
        """The closest opened fit is committed once, and once more when the order it went into was refused at checkout."""
        if not getattr(self, "committed", False):
            return True
        return self.commits < 2 and self.refused_at > self.commit_at and not self.items()

    def products_unheld(self) -> list[str]:
        """The requested products without a cart line (one product: itself, when the cart is empty)."""
        items = self.items()
        if not items:
            return [] if self.cart_ids else [str(self.plan.get("product") or "item")]
        return [name for name, _ in items if not any(self.store.get(i) is not None and self.store.get(i).category
                                                     and _same_product(self.store.get(i).category, name) for i in self.cart_ids)]

    def commit_items(self) -> bool:
        """Several products, the model declining or the episode about to end: put the closest opened fit of each product
        the cart does not hold yet in the cart (a whole-basket tool: of every product) and order (a decline scores 0 in
        nearly every task)."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        items = self.items()
        if not items or self.order_attempts or self.finished or self.dead or not cart or not order:
            return False
        # a partly filled cart keeps its lines: only the products without one get a pick (a whole-basket tool takes all)
        need = self.products_unheld() if self.cart_ids and not self.basket_tool() else [name for name, _ in items]
        if getattr(self, "committed", False) or not need or self.steps_left() < len(need) + 2 or self.time_left() < 25:
            return False
        self.committed = True
        seen = [l for i, l in self.store.listings.items() if (l.facts or i in self.paged) and l.price is not None and l.stock is not False]
        if not seen:
            return False
        def audit_prompt(seen: list[Listing]) -> str:
            text = AUDIT_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
            text += "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
            text += "\n\nLISTING RECORDS SEEN:\n" + "\n".join(self.store.pages([l.ident for l in seen][-28:], 650))
            return text + ("\n\nSEVERAL PRODUCTS ARE REQUESTED: " + "; ".join(f"{n} x{q}" for n, q in items)
                           + '. Also return "picks": [{"product": the product name as listed here, "id": the best fit of that product among the records, or else its closest fit}], one per product; the picks\' prices times their quantities must add up to no more than the budget.')

        prompt = audit_prompt(seen)
        def choose(result: dict[str, Any], ids: set[str]) -> list[tuple[str, int]]:
            out: list[tuple[str, int]] = []
            for name, quantity in items:
                if name not in need:
                    continue        # the cart already holds a line of it
                pick = next((str(p.get("id")) for p in result.get("picks") or [] if isinstance(p, dict) and str(p.get("id")) in ids
                             and _same_product(str(p.get("product") or ""), name)), None)
                if pick and pick not in [i for i, _ in out]:
                    out.append((pick, quantity))
            return out

        result = self.judge(prompt, 1400) or {}
        picks = choose(result, {l.ident for l in seen})
        self.trace.append({"kind": "commit_items", "picks": picks, "result": result})
        held = [i for i, _ in picks]
        missing = [name for name in need if not any(_same_product(str(p.get("product") or ""), name) and str(p.get("id")) in held
                                                      for p in result.get("picks") or [] if isinstance(p, dict))]
        if missing and len(missing) <= 2 and self.steps_left() >= 2 * len(missing) + len(items) + 3 and self.time_left() > 50:
            # a product nothing was read for: one search for it and its first rows opened, then the picks again
            side = self.compare_tool()
            for name in missing:
                args = self.text_args("search", name)
                if not args or not self.roles.get("search"):
                    break
                call = {"id": f"commit_search_{self.step}", "type": "function", "function": {"name": self.roles["search"], "arguments": json.dumps(args)}}
                self.messages.append({"role": "assistant", "content": f"I will look for the {name}.", "tool_calls": [call]})
                for call_id, text in self.run_calls([call], planned=True):
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": _resolver_text(text)})
                rows = [i for i, _ in _rows_with_ids(self.records[-1] if self.records else None)][: max(1, self.compare_cap)]
                if rows:
                    self.open_pages(rows if side else rows[:1])
            seen = [l for i, l in self.store.listings.items() if (l.facts or i in self.paged) and l.price is not None and l.stock is not False]
            result = self.judge(audit_prompt(seen), 1400) or {}
            picks = choose(result, {l.ident for l in seen})
            self.trace.append({"kind": "commit_items_again", "picks": picks, "result": result})
        if len(picks) < len(need) or self.steps_left() < len(picks) + 2:
            return False
        calls = []
        whole = self.cart_args(picks[0][0], picks[0][1])
        array_key = next((k for k, v in (whole or {}).items() if isinstance(v, list) and v and isinstance(v[0], dict)), None)
        if whole and array_key:
            # a cart tool that takes the whole basket as a list of lines: one call carries every pick
            rows = []
            for ident, units in picks:
                row = dict(whole[array_key][0])
                for k, v in row.items():
                    row[k] = units if isinstance(v, int) and not isinstance(v, bool) else ident if v == picks[0][0] else v
                rows.append(row)
            args, problems = _check(dict(whole, **{array_key: rows}), self.schemas[cart], "args")
            if problems:
                return False
            calls.append(args)
        else:
            for ident, units in picks:
                args = self.cart_args(ident, units)
                if not args:
                    return False
                calls.append(args)
        unpaged = [i for i, _ in picks if i not in self.paged]
        if unpaged and self.roles.get("details") and self.steps_left() >= len(calls) + 4:
            self.open_pages(unpaged)
        self.forcing = True
        try:
            for index, args in enumerate(calls):
                call = {"id": f"commit_{self.step}_{index}", "type": "function", "function": {"name": cart, "arguments": json.dumps(args)}}
                self.messages.append({"role": "assistant", "content": "I will put the closest fit of each product in the cart.", "tool_calls": [call]})
                for call_id, text in self.run_calls([call], planned=True):
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
                if self.finished or self.steps_left() < 2:
                    return self.finished
            if not self.cart_used:
                return False
            if self.finish_pending() and self.steps_left() >= 2 and self.run_finish(None, final=True):
                return True
            args = self.neutral_args(order, "Order the cart.")
            call = {"id": f"commit_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(args)}}
            self.messages.append({"role": "assistant", "content": "I will order it.", "tool_calls": [call]})
            for call_id, text in self.run_calls([call], planned=True):
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            return True
        finally:
            self.forcing = False

    def _commit_calls(self, cart: str, order: str, args: dict[str, Any]) -> bool:
        call = {"id": f"commit_{self.step}", "type": "function", "function": {"name": cart, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will put the closest fit in the cart.", "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if self.finished or self.steps_left() < 1 or not self.cart_used:
            return self.finished
        if self.finish_pending() and self.steps_left() >= 2 and self.run_finish(None, final=True):
            return True
        args = self.neutral_args(order, "Order the cart.")
        call = {"id": f"commit_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": "I will order it.", "tool_calls": [call]})
        for call_id, text in self.run_calls([call], planned=True):
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return True

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        # the cart as it is now decides (a line put in and taken out again leaves nothing to order)
        if not self.cart_ids and self.last_commit():
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if order and self.cart_ids and self.finish_pending() and self.steps_left() >= 2:
            self.run_finish(None, final=True)       # the request's finishing rules first; it orders when they are met
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        if order and self.cart_ids:
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
        elif decline:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True)}
        else:
            return
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        refused = (reply or {}).get("observation") if isinstance((reply or {}).get("observation"), dict) else {}
        lines: list[str] = []
        parts = [v for v in refused.values() if isinstance(v, (list, str))]
        parts += [x for v in refused.values() if isinstance(v, dict) for x in v.values() if isinstance(x, (list, str))]
        _sentences(parts, lines)
        if (reply and not reply.get("finished") and lines and order and action["name"] == order and self.cart_ids
                and _PRICE_WORDS.search(" ".join(lines)) and not _GONE.search(" ".join(lines)) and self.steps_left() > 0):
            # refused only for a changed price: the cart still holds a good item, so it is ordered once more
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True)}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def note_observation(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason") if isinstance(observation.get(k), str))
            core = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
            if observation.get("error") and _USED_UP.search(problem) and name not in core:
                if name == self.roles.get("message"):
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                self.capped.add(name)
                notes.append("That tool's allowance is used up; it is no longer offered. Do not try it again: read prices and stock from search results or the cart view instead, and decide with what you have.")
            if "refusal" in observation and name == self.roles.get("decline") and name != self.roles.get("order"):
                # a decline answered with a change of the request: the episode goes on, and the change (a new budget, a
                # dropped requirement) can make a seen listing fit, or leave nothing, so the decline is weighed again
                self.decline_ok = False
                self.decline_checks = min(self.decline_checks, 1)
                self.order_checked = False
                notes.append("THE DECLINE DID NOT END THE EPISODE: the shopper sent a change with it, and the episode continues. Apply every change it states (a new budget, a requirement dropped or added). Then decide again under the changed request: if a listing you have seen now meets every remaining requirement and the current budget, put it in the cart and order it; if none does, call the decline tool again.")
            elif "refusal" in observation:
                self.refusals += 1
                self.refused_at = self.step
                self.refused_ids.update(self.cart_ids)
                self.order_checked = False
                notes.append("THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again.")
                refusal = observation.get("refusal")
                said = refusal if isinstance(refusal, str) else _compact(refusal)
                if _PRICE_WORDS.search(said) and not _GONE.search(said):
                    notes.append("This refusal does not say the item is gone: when it only reports a changed price (or another change the item still meets), the item is still available. Re-read its current price; if price times quantity is still within the budget and it still meets every requirement, order it again right away. Never decline because a price changed.")
            elif len(observation) >= 2 and name != self.roles.get("message"):
                lines: list[str] = []
                for key, value in observation.items():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _sentences(value, lines)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
        elif isinstance(observation, list):
            _collect_ids(observation, self.seen_ids)
        return " ".join(notes)

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {t["function"]["name"] for t in self.tools_for_turn()}
        runnable: list[tuple[str, str, dict[str, Any]]] = []
        finishing = {self.roles.get("order"), self.roles.get("decline")} - {None}
        message_tool = self.roles.get("message")
        cart_tool = self.roles.get("cart")
        planned_messages = 0
        for call in calls:
            call_id = call["id"]
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            if name not in self.schemas:
                replies.append((call_id, f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}."))
                continue
            if not planned and name not in allowed:
                if name == message_tool and self.planner_on:
                    reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
                else:
                    reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
                replies.append((call_id, f"Not run (no step used): {reason}."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _parse_json_object(raw or "{}") if (raw or "").strip() else {}
                if args is None:
                    replies.append((call_id, "Not run (no step used): the arguments were not a valid JSON object."))
                    continue
            args, problems = _check(args, self.schemas[name], "args")
            if problems:
                replies.append((call_id, "Not run (no step used): invalid arguments: " + "; ".join(problems[:6])))
                continue
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if (name == message_tool and not planned and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str))
                    and not self.finish_reserved):
                replies.append((call_id, "Not run (no step used): messages to the shopper are only for questions, and this one asks nothing. Do not report findings, choices or plans to the shopper; the finishing tool carries that."))
                continue
            if name == message_tool and " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower() in self.questions_sent:
                replies.append((call_id, "Not run (no step used): this exact message was already sent, and its answer is above. Ask about something else, or carry on."))
                continue
            if name == message_tool and self.message_budget is not None and self.messages_used + planned_messages >= self.message_budget:
                replies.append((call_id, "Not run (no step used): the message budget is used up. Do not message the shopper again."))
                continue
            if name == message_tool and not planned and self.kept_messages() and self.messages_used + planned_messages >= self.question_limit():
                replies.append((call_id, "Not run (no step used): the messages left are kept for the request's finishing rules, which are carried out for you when you call the order tool. Build the final cart and call the order tool."))
                continue
            if name == message_tool and not planned:
                sent_text = " ".join(str(v) for v in args.values() if isinstance(v, str))
                prop = _question_property(sent_text)
                if prop and "?" in sent_text and self.asked_before(_question_product(sent_text), prop):
                    replies.append((call_id, "Not run (no step used): a question about this property was already sent (its answer is above, or the shopper did not follow it), and asking again only spends a message. Ask about a different property of the listings, or carry on."))
                    continue
                if self.budget_now is not None and re.search(r"\b(spend|budget|afford|how much|price limit)\b", sent_text, re.I):
                    replies.append((call_id, "Not run (no step used): the request already states the price limit; the shopper is not asked about money. Ask about a listing property instead, or carry on."))
                    continue
            signature = name + _compact(args)
            if name == self.roles.get("search"):
                args = self.search_mode(args)
            if name == self.roles.get("search") and not planned:
                stamp = f"{self.ledger_version}|{self.refusals}|{self.budget_now}"
                if self.searches_done.get(signature) == stamp:
                    replies.append((call_id, "Not run (no step used): this exact search was already made and nothing has changed since; its rows are above. Use other words, a filter, the next page, or open the listings it showed."))
                    continue
                self.searches_done[signature] = stamp
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if name == cart_tool and self.steps_left() > 3 and not self.items() and not self.forcing:      # several products: each line is checked once, at the order
                properties = self.schemas[name].get("properties") or {}
                signature_now = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                if self.add_signature is None or signature_now == self.add_signature:
                    target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
                    if target is None and _listed_ids(args):
                        target = _listed_ids(args)[-1]
                    if target and f"{target}|{self.ledger_version}" not in self.cart_checked:
                        self.cart_checked.add(f"{target}|{self.ledger_version}")
                        verdict = self.verify_item(target, "ADD TO THE CART", _line_of(args, target))
                        if verdict:
                            replies.append((call_id, "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again."))
                            continue
            if name == cart_tool and self.seen_ids:
                stray = [v for k, v in args.items() if isinstance(v, str) and "enum" not in ((self.schemas[name].get("properties") or {}).get(k) or {})
                         and (" " in v.strip() or (len(v) > 3 and not _ID_HINT.match(v))) and v not in self.seen_ids]
                if stray:
                    replies.append((call_id, "Not run (no step used): the cart tool takes the listing's id (as shown in results), not its title. Send the call again with the id."))
                    continue
            if name == cart_tool:
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            compare = self.compare_tool()
            if compare and name == compare[0] and isinstance(args.get(compare[1]), list):
                wanted = list(dict.fromkeys(str(i) for i in args[compare[1]]))
                if len(wanted) > self.compare_cap:
                    args = dict(args)
                    args[compare[1]] = wanted[: self.compare_cap]
                    self.compare_left = wanted[self.compare_cap:]
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            declining = self.is_decline(name, args)
            closest = bool(self.plan.get("closest_if_none"))
            if (declining and not planned and self.roles.get("order") and self.cart_ids and not self.products_unheld()
                    and (closest or self.decline_checks >= 2) and not self.decline_ok):
                # a line per product in the cart, and the request wants the closest ordered (or both audits named fits):
                # a decline is right in 1-2% of tasks at most, so the cart is ordered, through the usual checks
                name, args = self.roles["order"], self.neutral_args(self.roles["order"], "Order the cart.")
                runnable[0] = (call_id, name, args)
                declining = False
            # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest" and not self.forcing and self.cost < COST_HARD
                    and self.refused_at > self.searched_at and self.research_checks < 1 and self.steps_left() > 10
                    and self.time_left() > WRAP_UP_SECONDS + 20):
                self.research_checks += 1
                replies.append((call_id, "Not run yet (no step used). The last order was refused, and the request asks for the cheapest fitting item. A backup chosen earlier is right only if nothing cheaper fits now, so search again before ordering: the stated requirement words and the category, with a price limit just above the refused item's price if the search offers one, and the next pages. Open the details of cheaper candidates, put the cheapest one that meets every requirement in the cart, then order."))
                return replies
            if not declining and name == self.roles.get("order") and self.cheaper_checks < 2 and self.steps_left() > 10 and self.time_left() > WRAP_UP_SECONDS + 20 and not self.forcing and self.cost < COST_HARD:
                cheaper = self.unchecked_cheaper()
                if cheaper:
                    self.cheaper_checks += 1
                    replies.append((call_id, "Not run yet (no step used). The request asks for the cheapest fitting item, but these cheaper listings that you have seen in results were never opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Open their details (several in one reply), and if one meets every requirement, put the cheapest such one in the cart instead; if none does, order again."))
                    return replies
            if (not declining and name == self.roles.get("order") and self.pointers and self.steps_left() > 8 and self.time_left() > WRAP_UP_SECONDS + 40
                    and self.cost < COST_HARD and not self.pointers_stalled and not self.forcing):
                replies.append((call_id, "Not run yet (no step used). A requirement still points at another listing's page that has not been read; the resolver reads it next, then order."))
                return replies
            if not declining and name == self.roles.get("order") and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS and not self.forcing:
                gate = self.line_gate() or self.dependency_gate()
                if gate:
                    replies.append((call_id, "Not run yet (no step used). " + gate))
                    return replies
            elif not declining and name == self.roles.get("order") and not planned and not self.forcing and self.steps_left() > 3 and self.time_left() > 30:
                unpaged = self.unpaged_lines()
                if unpaged and ("late|" + str(self.cart_changed_at)) not in self.line_blocks:
                    self.line_blocks.add("late|" + str(self.cart_changed_at))
                    self.page_next = unpaged
                    replies.append((call_id, f"Not run yet (no step used). The page of {', '.join(unpaged[:4])} was never opened (a property list or a search row does not count as opening a listing); it is opened for you now. Then call the order tool again."))
                    return replies
            if not declining and name == self.roles.get("order") and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS and not self.forcing:
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                replies.append((call_id, "Not run yet (no step used). Pre-order check: (1) the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity) and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."))
                return replies
            if declining and not closest and self.page_checks < 2 and self.steps_left() > 8 and self.time_left() > WRAP_UP_SECONDS + 30 and self.cost < COST_HARD:
                unread = self.unread_searches()
                if unread:
                    self.page_checks += 1
                    replies.append((call_id, "Not run yet (no step used). Before declining, read the rest of these searches, whose matches were not all shown: " + "; ".join(unread)
                                    + ". " + self.order_note() + ", so an unseen match may fit as well as a seen one. Page through them, open (or compare side by side) every row of the requested kind that may fit, and buy the best fit; decline only if each one truly fails a requirement, a revealed need, the budget or stock."))
                    return replies
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest" and self.sweep_checks < 1 and not self.forcing
                    and self.cost < COST_HARD and self.time_left() > WRAP_UP_SECONDS + 30 and self.sweep_cheapest() != ""):
                self.sweep_checks += 1
                self.cheaper_checks += 1
                cheaper = self.unchecked_cheaper()
                if cheaper:
                    replies.append((call_id, "Not run yet (no step used). The searches that found your pick were read to their end, and these listings of the same kind are cheaper than your pick, in stock and within budget, but were never opened: "
                                    + ", ".join(f"{i} ({p:g})" for i, p in cheaper) + ". Open them (several in one call), and if one meets every requirement, put the cheapest such one in the cart instead; if none does, order again."))
                    return replies
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") in ("cheapest", "best") and self.sweep_checks < 1
                    and self.steps_left() > 8 and self.time_left() > WRAP_UP_SECONDS + 30 and self.cost < COST_HARD):
                unread = self.unread_searches() or self.ranked_unread()
                if unread:
                    self.sweep_checks += 1
                    replies.append((call_id, "Not run yet (no step used). The request asks for the " + str(self.plan.get("priority")) + " fitting item, but these searches were not read deep enough: " + "; ".join(unread)
                                    + ". " + self.order_note() + ", so a better or cheaper fit may sit on an unseen page. Read the next pages, open any row that may fit, then order the right item."))
                    return replies
            if (not declining and name == self.roles.get("order") and not planned and not self.forcing and self.finish_pending()
                    and self.steps_left() >= 2 and self.time_left() > 22):
                self.finish_next = dict(args)
                replies.append((call_id, "Not run yet (no step used). The request's finishing rules are checked now and anything still to do is carried out for you; the order follows at once when nothing changes. Do not change the cart from here on."))
                return replies
            if (declining and not planned and (closest or self.steps_left() <= 4 or self.time_left() < WRAP_UP_SECONDS + 30 or self.decline_checks >= 2)
                    and not self.cart_ids and not self.items() and not self.decline_ok
                    and self.may_commit() and self.roles.get("cart") and self.steps_left() >= 2):
                # (late in steps, too late in time for the decline audits, or declining again after both audits named
                # possible fits: a decline is right in 1-2% of tasks, the closest fit far more often)
                self.commit_next = True
                replies.append((call_id, "Not run (no step used): the closest opened fit is put in the cart and ordered next."))
                return replies
            if (declining and not planned and self.items() and not self.order_attempts and not self.decline_ok
                    and (closest or self.decline_checks >= 2 or self.time_left() < WRAP_UP_SECONDS + 30)
                    and not getattr(self, "committed", False) and self.roles.get("cart") and self.steps_left() >= 3
                    and self.time_left() > 30):
                self.commit_next = True
                replies.append((call_id, "Not run (no step used): the closest opened fit of each product is put in the cart and ordered next."))
                return replies
            if declining:
                # A decline is checked twice: it is rarely the right finish, so each attempt first gets an
                # independent look for listings that meet the requirements.
                if self.decline_checks < 2 and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS + 30 and not self.decline_ok:
                    self.decline_checks += 1
                    audit = self.audit_decline()
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    if self.decline_checks == 1 and not self.decline_ok:
                        replies.append((call_id, "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count, and that a requirement you only assumed from the story does not count. If something does fit, buy it instead; if nothing fits, call the decline tool again."))
                        return replies
                args = self.clean_decline(args, name)
                runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
                runnable = runnable[: max(0, keep)]
        for start in range(0, len(runnable)):
            # one action per environment request, in order (the interface asks for it)
            chunk = runnable[start : start + 1]
            if self.finished or self.dead or self.steps_left() <= 0:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "Not run: the episode is over or no steps are left."))
                continue
            chunk = chunk[: self.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            before = self.step
            reply = self.env_call(actions)
            self.trace.append({"kind": "env", "step": before + 1, "actions": actions, "reply": reply})
            if reply is None:
                for call_id, _, _ in chunk:
                    replies.append((call_id, "The call failed (network or server). Try again."))
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (call_id, name, args) in enumerate(chunk):
                if index >= len(observations):
                    replies.append((call_id, "Not run: the episode finished before this call."))
                    continue
                observation = observations[index]
                if name == message_tool:
                    self.messages_used += 1
                    sent_text = " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower()
                    if sent_text and not (isinstance(observation, dict) and observation.get("error")):
                        self.questions_sent.append(sent_text)      # a message that failed may be sent again
                if name == self.roles.get("order"):
                    self.order_attempts += 1
                if name == cart_tool:
                    self.cart_used = True
                    self.order_checked = False
                if name in (cart_tool, message_tool) or name in finishing:
                    self.recent = []
                else:
                    self.recent = (self.recent + [name + _compact(args)])[-2:]
                self.records.append(observation)
                self.store.absorb(observation, self.step, opened=name in (self.roles.get("details"), self.roles.get("view")) or name == (self.compare_tool() or ("",))[0])
                if name == self.roles.get("view") and name != self.roles.get("details") and not self.view_facts:
                    self.view_facts = any(isinstance(v, str) and v in self.store.listings and self.store.listings[v].facts for v in args.values())
                if name == cart_tool:
                    properties = self.schemas[name].get("properties") or {}
                    signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
                    if self.add_signature is None:
                        self.add_signature = signature
                    listed = _listed_ids(args)
                    if listed is not None and signature != self.add_signature:
                        self.cart_ids = [i for i in self.cart_ids if i not in listed]     # the same tool used to take lines out
                    elif listed is not None:
                        self.cart_ids = listed     # a cart tool that takes the whole basket as a list of lines
                        self.cart_lines = {i: _line_of(args, i) for i in listed}
                        for rows in (v for v in args.values() if isinstance(v, list)):
                            for row in (r for r in rows if isinstance(r, dict)):
                                ident = next((v for v in row.values() if v in listed), None)
                                units = next((v for v in row.values() if isinstance(v, int) and not isinstance(v, bool)), None)
                                if ident and units is not None:
                                    self.cart_qty[ident] = units
                    asked_units = next((v for k, v in args.items() if isinstance(v, int) and not isinstance(v, bool)
                                        and isinstance(properties.get(k), dict) and properties[k].get("type") == "integer" and "enum" not in properties[k]), None)
                    for key, value in (args.items() if listed is None else ()):
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip() and asked_units is not None:
                            self.cart_qty[value] = asked_units
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature and asked_units != 0:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
                if name == self.roles.get("search"):
                    self.searched_at = self.step
                    self.searches_in_a_row += 1
                    if not planned and _rows_with_ids(observation):
                        sprops = (self.schemas.get(name) or {}).get("properties") or {}
                        worded = [k for k, v in args.items() if isinstance(v, str) and v.strip() and isinstance(sprops.get(k), dict)
                                  and "enum" not in sprops[k] and not (_ID_HINT.match(v.strip()) and any(c.isdigit() for c in v))]
                        if len(worded) == 1:
                            self.key_votes[worded[0]] = self.key_votes.get(worded[0], 0) + 1
                if name == message_tool and not planned:
                    own = " ".join(str(v) for v in args.values() if isinstance(v, str))
                    if "?" in own:
                        self.asked_props.append((_question_product(own), _question_property(own)))
                if name in (self.roles.get("details"), self.roles.get("view")):
                    self.opened.update(v for v in args.values() if isinstance(v, str))
                failed_call = isinstance(observation, dict) and bool(observation.get("error"))
                if name == self.roles.get("details") and not failed_call:
                    self.paged.update(v for v in args.values() if isinstance(v, str))
                    self.searches_in_a_row = 0
                side = self.compare_tool()
                if side and name == side[0] and isinstance(args.get(side[1]), list):
                    shown = {i for i, _ in _rows_with_ids(observation)}
                    self.paged.update(str(i) for i in args[side[1]] if str(i) in shown)
                    self.searches_in_a_row = 0
                    self.opened.update(str(i) for i in args[side[1]] if str(i) in shown)
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                if name in (cart_tool, self.roles.get("cart_view")) and not (isinstance(observation, dict) and observation.get("error")):
                    lines_seen = _cart_lines(observation)
                    if lines_seen is not None:
                        lines_seen = [(i, n if n is not None else self.cart_qty.get(i, 1)) for i, n in lines_seen]
                        self.cart_seen = lines_seen
                        self.cart_seen_at = self.step
                        if name == cart_tool:
                            self.cart_ids = [i for i, n in lines_seen if n > 0]
                            self.cart_lines = {}
                self.note_event(name, args, observation)
                note = self.note_observation(name, observation)
                if name == cart_tool and self.cart_seen_at == self.step:
                    over = self.over_units()
                    if over:
                        note = (note + " " if note else "") + over
                compare = self.compare_tool()
                if compare and name == compare[0] and isinstance(args.get(compare[1]), list):
                    sent_ids = args[compare[1]]
                    empty = not observation or (isinstance(observation, dict) and all(not v for v in observation.values()))
                    # a bare error ({"issue_text": "E13"}: no listing in it) for many ids is the tool's cap, not a bad id
                    bare = isinstance(observation, dict) and len(observation) <= 3 and not any(isinstance(v, (dict, list)) for v in observation.values())
                    shown_ids = {i for i, _ in _rows_with_ids(observation)}
                    unanswered = [str(i) for i in dict.fromkeys(sent_ids) if str(i) not in shown_ids and (str(i) in self.store.listings or str(i) in self.seen_ids)]
                    if len(sent_ids) > 1 and (empty or bare) and not shown_ids:
                        self.compare_cap = max(1, min(self.compare_cap, len(sent_ids) // 2))
                        note = (note + " " if note else "") + (f"The side-by-side tool showed no listing for {len(sent_ids)} ids, so none of them counts as opened: it takes fewer per call. Send at most {self.compare_cap} ids per call.")
                    elif len(sent_ids) > 1 and shown_ids and unanswered:
                        # fewer rows than ids, no error: the tool answers only so many per call (known listings were left out)
                        self.compare_cap = max(1, min(self.compare_cap, len(shown_ids)))
                        note = (note + " " if note else "") + (f"The side-by-side tool showed only {len(shown_ids)} of the {len(set(sent_ids))} listings asked for; these were NOT opened: {', '.join(unanswered[:8])}. Send at most {self.compare_cap} ids per call.")
                    elif getattr(self, "compare_left", None):
                        note = (note + " " if note else "") + (f"The side-by-side tool shows at most {self.compare_cap} listings per call, so only the first {len(sent_ids)} were sent; these were NOT opened: {', '.join(self.compare_left)}. Send them in another call if you need them.")
                    self.compare_left = []
                if name == self.roles.get("search") and not planned:
                    page_note = self.track_search(args, observation)
                    if page_note:
                        note = (note + " " if note else "") + page_note
                    if self.searches_in_a_row >= 3:
                        side = self.compare_tool()
                        note = (note + " " if note else "") + ("Several searches in a row and no listing opened: rows do not show a listing's properties, its page does. Open the likeliest rows now"
                                                               + (f" ({side[0]} shows {self.compare_cap} pages for one step)" if side and self.compare_cap >= 2 else "") + " before searching again.")
                pending = (isinstance(observation, dict) and _pending_step(observation)) and not (isinstance(observation, dict) and "refusal" in observation)
                if (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and pending):
                    note = (note + " " if note else "") + ("The order is waiting for a further step (it gives a token or asks for a confirmation): take that step now with the tool that finishes the checkout. Nothing is bought until the episode ends.")
                elif (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished")
                        and not (isinstance(observation, dict) and "refusal" in observation)):
                    # the order did not end the episode: refused (in any wording), failed, or waiting for another step
                    self.refusals += 1
                    self.refused_at = self.step
                    self.refused_ids.update(self.cart_ids)
                    self.order_checked = False
                    note = (note + " " if note else "") + ("THE ORDER CALL DID NOT END THE EPISODE, SO NOTHING HAS BEEN BOUGHT YET, whatever its result says (an order that goes through ends the episode). If its result asks for a further step (for example a confirmation, with a token or code it gives), take that step now. If it was refused or failed, read why, apply every change it states, fix the cart, re-check stock and price of the pick, and order again; a price change alone does not make an item unacceptable while price times quantity stays within the budget.")
                heard: list[str] = []
                if name == message_tool:
                    if isinstance(observation, str):
                        heard = [observation]
                    elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                        if isinstance(observation, dict):
                            _sentences(list(v for v in observation.values() if isinstance(v, (str, list))), heard)
                        if not heard and _has_text(observation):
                            # a short or structured answer ("Navy.", {"value": "navy"}) is still the shopper's answer
                            heard = [_compact(observation)[:600]]
                    question = " ".join(str(v) for v in args.values() if isinstance(v, str))
                else:
                    question = None
                    if isinstance(observation, dict):
                        wrapper = len(observation) <= 3 and any(isinstance(v, (dict, list)) for v in observation.values())
                        for value in observation.values():
                            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                                _sentences(value, heard)
                            elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                                heard.append(value)
                            elif (isinstance(value, str) and wrapper and len(value.split()) >= 6
                                  and (_FIRST_PERSON.search(value) or name == self.roles.get("order"))):
                                heard.append(value)     # a line wrapped around the real result, as a plain string
                if heard:
                    meaning = self.interpret(heard, question)
                    if meaning:
                        note = (note + " " if note else "") + meaning
                if (name == cart_tool and not planned and not reply.get("finished") and not failed_call and index == len(observations) - 1
                        and len(self.held_lines()) >= len(self.items())):
                    # the cart now holds the lines a tie rule joins: checked here, so the model can fix it before ordering
                    tie = self.dependency_gate()
                    if tie:
                        note = (note + " " if note else "") + tie
                text = _compact(_brief(observation, ROW_CHARS if name == self.roles.get("search") else PAGE_CHARS))
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        order = {c["id"]: i for i, c in enumerate(calls)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        return replies

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.setup()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        user = "SHOPPER REQUEST:\n" + str(self.policy.get("query") or "")
        if opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + _compact(opening)
        brief = self.brief()
        if brief:
            user += "\n\n" + brief
        user += "\n\n" + self.status()
        if opening is not None:
            self.records.append(opening)
            self.store.absorb(opening, 0)
            _collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.cost > COST_HARD + 0.02:
                # well past the episode's spending limit: finish now (the closest fit when the cart is empty)
                self.force_finish()
                break
            if self.finish_next is not None:
                order_args, self.finish_next = self.finish_next, None
                if self.run_finish(order_args, final=self.time_left() < 45):
                    continue
                self.finish_clear.add(self.finish_state())      # nothing to do (or it cannot be done): the order goes through
            if self.page_next:
                ids, self.page_next = self.page_next, []
                self.open_pages(ids)
                continue
            if self.dep_next:
                jobs, self.dep_next = self.dep_next, []
                self.dependency_search(jobs)
                continue
            if self.commit_next:
                self.commit_next = False
                if self.last_commit():
                    continue
            if self.pointers and self.step >= 1 and self.steps_left() > 8 and self.cost < COST_HARD and not self.pointers_stalled:
                before = (len(self.pointers), self.step)
                self.resolve_pointers()
                if (len(self.pointers), self.step) == before:
                    self.pointers_stalled = True      # nothing could be done: never loop on it
                continue
            if (self.plan.get("priority") == "cheapest" and not self.order_attempts and self.sweep_checks < 1 and (not self.items() or self.cart_ids)
                    and self.order_due() + 2 < self.steps_left() <= (self.max_steps if self.cart_ids else self.order_due() + 3 + SWEEP_PAGES)
                    and self.cost < COST_HARD and self.time_left() > WRAP_UP_SECONDS + 40 and not self.pointers):
                # right before the steps kept for the order (or once a pick is in the cart): the pick's searches read to the end
                pick = self.sweep_cheapest(self.order_due() + 1)
                if pick:
                    self.sweep_checks += 1
                    cheaper = self.unchecked_cheaper() if self.cart_ids else self.store.cheaper(
                        pick, self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None,
                        self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1,
                        self.opened, self.cart_ids)
                    self.messages.append({"role": "user", "content": ("[sweep] The searches that found your pick were read to their end. "
                        + ("These listings of the same kind are cheaper than your pick, in stock and within budget, and were never opened: "
                           + ", ".join(f"{i} ({p:g})" for i, p in cheaper) + ". Open them (several in one call), and if one meets every requirement, put the cheapest such one in the cart instead."
                           if cheaper else "Nothing cheaper of the same kind turned up.") + "\n" + self.status())})
                    continue
            if self.planner_on and self.ask_planned():
                continue
            if self.steps_left() <= 3 and not self.cart_used and not self.order_attempts and not self.items():
                # three steps left and nothing in the cart: the closest opened fit is ordered rather than nothing
                if self.last_commit():
                    continue
            if self.items() and self.steps_left() <= len(self.items()) + 3 and not self.cart_used and not self.order_attempts:
                if self.last_commit():      # the same for several products, one line each
                    continue
            message = self.infer(self.window(), self.offered_tools(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 4 or self.time_left() < WRAP_UP_SECONDS + 20:
                    self.force_finish()
                    break
                time.sleep(2.0 * failures)
                continue
            failures = 0
            calls = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                counter += 1
                function = call.get("function") or {}
                calls.append({"id": call.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": function.get("arguments") if isinstance(function.get("arguments"), str) else json.dumps(function.get("arguments") or {})}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _compact(content)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if self.planner_on and any(c["function"]["name"] in committing for c in calls):
                # questions are planned from the listings opened so far; with too few of them the plan is a guess
                if (not self.early_bounced and self.property_menu()[1] < 3 and self.steps_left() > 12
                        and self.time_left() > WRAP_UP_SECONDS + 60):
                    self.early_bounced = True
                    self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                    for index, c in enumerate(calls):
                        note = ("Not run (no step used): before choosing, open the details of two or three more listings of the requested kind "
                                "(several in one reply, preferring ones that differ from those you have seen); the questions to the shopper are planned from what they show.")
                        if index == len(calls) - 1:
                            note += "\n" + self.status()
                        self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": note})
                    continue
                if self.ask_planned(force=True):
                    continue
            if not calls:
                idle += 1
                # a reply without a call says nothing about the store (it may even claim the tools are gone): it is not kept
                self.messages.append({"role": "assistant", "content": "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools; they all work. Continue.\n" + self.status()})
                if idle >= 5 or self.time_left() < WRAP_UP_SECONDS:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            step_before = self.step
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            # turns in which nothing ran: in the steps kept for the order, the third such turn sends the order call itself
            # (through every check, as if the model had sent it)
            self.blocked_turns = self.blocked_turns + 1 if self.step == step_before and not self.finished else 0
            if self.blocked_turns >= 3 and self.in_reserve() and self.roles.get("order") and self.finish_next is None:
                self.blocked_turns = 0
                order = self.roles["order"]
                call = {"id": f"due_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
                self.messages.append({"role": "assistant", "content": "The steps kept for the order have come: I will order the cart.", "tool_calls": [call]})
                for call_id, text in self.run_calls([call]):
                    self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text + "\n" + self.status()})
        if not self.finished and not self.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.step, "cost": round(self.cost, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception as exc:
        episode.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not episode.finished and not episode.dead:
                episode.force_finish()
        except Exception as again:
            episode.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return episode.trace


__all__ = ["agent_main"]
