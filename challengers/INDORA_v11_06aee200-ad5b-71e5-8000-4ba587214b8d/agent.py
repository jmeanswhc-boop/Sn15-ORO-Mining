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
PLAN_MODEL = "anthropic/claude-sonnet-5.5"   # the first round of questions is planned by the stronger model

TASK_SECONDS = 280.0
COST_SOFT = 0.035            # dollars spent in one episode after which the checks run on the cheaper model
COST_HARD = 0.05            # beyond this the optional checks are skipped, so one episode cannot drain the run's key
COST_WRAP = 0.045           # dollars spent in one episode after which the agent is told to finish
COST_STOP = 0.055            # dollars spent in one episode after which only the finishing tools remain, then the finish is forced
CONTEXT_WRAP = 45000        # prompt tokens of the main conversation after which the agent is told to finish
CONTEXT_STOP = 55000        # and after which only the finishing tools remain (no passing episode has needed this much)
LOW_EFFORT = "low"          # reasoning effort for the side calls (setup, planning, reading, checks)
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 12000
MAX_TURNS = 80

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is the one the request names for that case: a decline the way it says, or an order of the nearest candidate when the request wants the nearest one offered instead.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If the store has a tool that shows the full details of several listings in one call, use it instead of opening them one by one: it saves steps.
- Whenever an argument names a listing (to put it in the cart, open it or compare it), give the listing's id exactly as the results show it, never its title or a shortened name, whatever the argument is called.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties). A store may split a listing over two tools, one showing only its property list and one its full page (notes, description, price, stock): check requirements on the property list (read it with that tool for every candidate before choosing; a full page or a side-by-side view does not replace it), and open in full every listing you put in the cart or reach through a reference (several at once when a tool allows it).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (a category or price filter, another short wording, other pages) until you have seen the whole relevant set. Keep every keyword query short: the product's plain name and one or two telling words (a required value), the most telling first. A long query matches few listings or has its later words ignored, so when a search shows nothing useful, try another short wording rather than a longer one.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down: firm values of the product's own properties, of the same kind as the requirement the request states (the properties the store's listings carry for that kind of product: who or what it is for, its style or type, its material or finish, its colour or pattern, its key specification, a size, length or capacity limit, its condition). They reveal one only when a message asks about that property. The score is 0 unless every hidden need was asked about: an item that happens to meet a need nobody asked about still fails. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, in the form "Which <property> would you like the <product> to have?" with the property's name as the listings name it, the product named by its plain noun from the request, and no example values. Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once. An open question ("what should I know about it?") is wasted: it names no property.
- The story around the request usually hints at which property matters (who will use it, where it goes, how it will be used, what it must match or fit, what went wrong with an earlier one). Those properties are asked about first.
- A need, stated or revealed, may be given by reference to another listing: the same value as that listing shows, or anything other than it, where the listing is identified as a <product kind> from <seller> carrying a few named property values. To resolve it: search with that seller's name as a keyword (with the product kind), open the listings of that seller that carry the named values, and read the named property on it. The requirement is then that value (the very same) or anything except that value (anything bar). Until it is resolved the status shows it as unresolved; resolve it before choosing, while steps allow. The listing you reach may say in its notes that this property is to be read from yet another listing (of another seller, maybe another kind of product): then that one is the source; the status names it, so search for that seller and open it the same way, as many times as the notes send you on. Every listing on that path must be opened in full (with the tool that shows its whole page, notes included; a property list alone is not enough), and the notes on that page say whether the reading goes on.
- When the request asks for several different products, every question must say which product it is about; a question that leaves the product open confuses the shopper. Spread the questions over the products.
- If the shopper says they did not follow a message, or asks you to explain, simplify or rephrase it, do not explain or repeat it: when the request asks for several products this means the property is not one they care about for that product; ask about a different property next (naming the product). A reply asking for the questions singly means your message did not read as one plain question about one property: that property is still open.
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- The requirements are only what the request states and what the shopper reveals or changes. Never add one of your own from the story (an age group, a gender, a size, who will use it): such details are not requirements, and a listing does not fail because of them. If such a detail may matter, it may be a hidden need; ask about it.
- When a planner is available, your questions are chosen and sent for you as soon as you have opened the details of four or five listings of the requested kind (of each kind, when several different products are requested; one call of a multi-listing details tool when the store has one), so do that in your first moves. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- Messages to the shopper are only for questions (and for answering a question the shopper asked). A reply saying the shopper does not mind about the property asked (any value suits them, or they leave the pick to you) is not a question to answer, even when it ends by asking what you or other people would pick: that property is free, so choose silently and use the next message for the next question. Never message the shopper to report what you found, what you bought or that you are declining; the finishing tool carries that.
- If you must ask yourself: first open the details of two or three listings, then ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened in full.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Besides finding and buying the item, a request may demand actions of its own (for example showing the shopper something, getting their OK, telling them something, opening a page), and a store page may say what else is needed for an item (for example another page to read). The actions due right before the order (after the basket is final) are worked out and carried out for you when you call the order tool, with messages and steps kept for them (the status shows them); a cart change after them means they are done again. Do the other actions yourself when they are due.
- Something the shopper would merely like ("ideally", "would be nice", "if possible") is not a requirement: use it only to choose between items that already meet everything, and never over a cheapest rule.
- If the request leaves something you can only learn from the shopper (for example no price limit, or two wishes that may not both be possible and the shopper must say which matters more), find it out by asking before you choose.

CHANGES
- While you work, the shopper may change things: a new budget (higher or lower), a requirement dropped, replaced or added, a price going up, an item sold out. One shopper line can carry several changes. Shopper lines can appear inside any tool result (an extra field holding sentences) or attached to an order attempt. Each change is binding from that moment: re-check your pick and your backups against it, and when the request asks for the cheapest or best, choose again over the new acceptable set.
- An order attempt can be REFUSED. The result then carries the reason in some field (a refusal, complaints, objections or similar), even if the inner result looks successful (accepted: true, placed: true, executed: true). An order that goes through ends the episode: while the episode continues, nothing has been bought, whatever the result says. The episode continues and the cart is unchanged. Read the refusal, apply every change it states, remove anything that is no longer acceptable from the cart (for example a sold-out item), add the next verified fit, re-check stock and price, and order again.
- Items can also sell out or change price silently. That is why stock and price must be re-checked right before every order attempt; an item whose current price times quantity is over the current budget no longer fits.

HOW TO JUDGE A LISTING (this decides the score)
- A requirement is met only when the listing's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key that names that property counts. Values match by meaning: one element of a comma-separated list is enough; singular and plural are the same; close synonyms count; case does not matter. A word that merely starts the same way does not count, and a broader or different class does not count (for example girls is not women, plain unisex is not female). A value written with a qualifier in front of the asked word (a kind or finish of it, such as pu leather or faux leather for leather, faux suede for suede, modern classic for modern) states that value and meets it; an exclusion of the word excludes it too.
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
- Declining is almost never the right finish: in nearly every task the store holds an item that meets every requirement, and a wrong decline scores the same 0 as a wrong order, so when in doubt order the best verified fit. Decline only after a thorough search (several wordings, the category, price filters, further pages), and only when, for every candidate, you can name a stated requirement, a revealed need, the budget or stock that its own records clearly fail. Doubt, a missing nice-to-have, a requirement you only assumed, a reply the shopper did not follow, or a reference you could not resolve is never a reason to decline. Follow the request's rule for declining exactly. When the request wants the nearest candidate offered in case nothing fits, the task ends with an order: put the nearest candidate for each requested product in the cart and order it, naming in the order's remark the requirements it misses. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- With several different products, put each product's verified fit in the cart as soon as you have it (together with the lines already there), then go on to the next product: a cart that holds the finished products can still be ordered when time runs short, while an empty one can only end in a decline.
- Plan your steps: get your first order attempt in while at least seven steps remain, so a refusal can still be handled. When 7 steps remain, only cart, stock-check and finishing tools stay available. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words (when the request asks for several different items, name each of them).
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"items": when the request asks for several DIFFERENT products, one object per product: {"product": a short name for it in the request's words, "quantity": the units of that product (integer)}; an empty list when only one kind of product is requested. A name joined with "and" or "&" that the request uses as one kind of product (the way stores name a department, such as "<A> and <B>" or "<A> & <B>" with one quantity and one set of requirements) is ONE product, not two.
"requirements": a list of strings, one per firm stated requirement on the product, each written as "property: constraint" with the shopper's exact value, limit, exclusion or closed list. When several different products are requested, start each one with the short product name it applies to, as "<product> <property>: constraint". Do not include stock, quantity, the total budget or mere wishes here (a price limit the request sets for one product's item, such as "I won't pay over N for a <product>", IS a requirement on that product: "<product> price: at most N", per unit, in the request's unit), and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement. A figure of speech, a simile, or a description of how the shopper feels about the product is story, not a requirement: a requirement names a property of the product and the value it must have.
"price_cap": the maximum total price as a number in the unit the request writes it in (a cap written in cents stays in cents: 75¢ is 75, never 0.75), or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"process": every action the request demands besides finding the item, putting it in the cart and ordering it (for example showing, telling or asking the shopper something, getting their OK, opening or comparing pages), in the order it must happen, one object each: {"do": the action in plain words, "when": "before_order" when it must happen once the basket is final and before the order, "before_cart" when it must happen before anything goes into the cart, "before_choice" when it must happen before choosing, else "any", "messages": how many messages to the shopper it takes (0 when none), "steps": how many tool calls it takes}. Limits on messages or steps are not actions, and neither is a question the shopper puts in the request asking for advice or an opinion (on what would suit them or whether it will fit): the right item answers it. Empty when there is none.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, or which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements). An open invitation to put questions to the shopper, or to check with them, is NOT an item here: it only says that unwritten needs exist. Nor is a story detail (who uses it, where it goes, what it must suit) or a question the request itself puts: those are hints, and hidden needs are learned by asking about the product's properties. Empty when none.
"hints": the story sentences that point at an unstated property of the product (who will use it, how or where it will be used or kept, what it must match, fit or be told apart from, how long it must last, what went wrong with an earlier one, what the shopper says they cannot judge themselves), one object each: {"product": the product it is about, "clue": that sentence, shortened, "properties": two or three plain names of the product's own classifying properties it may point at, most likely first}. A classifying property is one a listing states as a short value of its own: who it is for (gender, age group), its type or style, material, finish, colour, pattern, closure, height, condition, operating system, processor, memory, storage, capacity, length, flavour, supply. Never name a size offered per listing, a brand, a price, delivery or shipping, or an abstract quality (durability, comfort, quality); a fit or cut of a garment (how its legs or sleeves are cut) and a packaging type of a food or drink are properties. A sentence about who will use it points at gender (then age group) for things worn or sized for a person, otherwise at the product's type or key specification; about telling two items apart at colour or flavour; about how long it lasts or when it is restocked at a supply or capacity property; about the space where it goes at the dimension along that space (width for a stretch of wall, depth for how far it reaches in, height for what it must stand under or over), or at form factor or length for how it travels; about how it looks from a distance or in photos at pattern, colour or style; about how it sits on the body at sleeves, length or fit; about how much it holds, serves or weighs at volume, weight, count or capacity; about how far away a screen is watched at resolution; about what the shopper cannot judge at the key technical property. When several different products are requested, a sentence that does not name its product may be about any of them: give one object for each product it could fit, with that product's own properties. Leave out anything already stated as a requirement. Empty when the request is all business.
"pointers": requirements the request states by reference to another listing (the same value as that listing shows, a value that goes with it or is its counterpart, anything other than it, or a value no lower or no higher than it; the listing identified as a <product kind> from <seller> carrying <property> <value>, ...; the values that identify that other listing are not requirements on the product and never go into "requirements" on their own): one object each: {"product": the product it is about, "property": the property whose value is to be read off that listing, "relation": "same" (the very same value, a match), "paired" (a value that goes with, pairs with, is compatible with or is the counterpart of that listing's value: stores state on some listings which values go together), "different", "at_least" (no lower, no less, at least as much), "at_most" (no higher, no more, at most as much) or "other" (a relation none of these names, such as one size up, the next one or the opposite), "rule": for "other", the request's own words for the relation, else "", "seller": the seller's name as written, "category": the product kind of that listing as written, "attributes": {property: value, ...} as written, "pick": when several such listings exist, the rule for choosing the one meant, else ""}. Also put it in "requirements" as "<property>: the same as (or: paired with, anything except, at least, at most) the <property> on <seller>'s <category> listing (<attributes>)". Empty when none.
"links": rules that tie one requested product to another one the shopper is buying, one object each: a fit rule (one product must fit, suit, match or work with the other product bought) or a condition ("should <product A>'s <property> turn out to be <value>, I need <product B>'s <property> to be <value>"): {"product": the product the rule constrains, "on": the product it depends on, "kind": "fit" or "if", "if": for a condition, the other product's property and the value(s) it depends on, as written, else "", "then": for a condition, what it then demands of the product as "property: value", for a fit rule the rule's own words}. These never go into "requirements": what they demand depends on the item chosen for the other product. Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
"nothing_fits": what the request says to do when nothing meets every requirement: "closest" when it wants the nearest candidate offered or chosen instead (the one missing the fewest requirements), "decline" when it asks only to decline or buy nothing, "" when it does not say.
"decline_no_names": true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks you to name something; null when the request gives no rule.
"roles": an object naming the tool for each role, using the exact tool names from the list, or null when no tool fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" is the tool that sends a message to the shopper. "cart" is the tool that adds to or removes from the cart. "order" is the tool that buys or checks out. "decline" is the tool that ends the task without buying.
"""

INTERPRET_PROMPT = """You help a shopping assistant understand what a shopper just said. Return ONE JSON object and nothing else, with these keys:
"meaning": one short sentence saying what the shopper means.
"needs": a list of new firm requirements on the product, each written as "property: constraint" with the exact value, limit, exclusion or closed list the shopper gave.
"no_preference": a list of properties the shopper said they do not care about.
"dropped": a list of earlier requirements the shopper no longer needs.
"new_budget": the new maximum total price as a number in the unit the shopper writes it in (cents stay cents: 75¢ is 75), or null.
"new_quantity": when the shopper changes how many units they want, {"product": the product it is for (or "" when only one kind is requested), "quantity": the new number as an integer}; otherwise null.
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false; a general wish for items in stock is not this.
"repriced": true when the shopper says the price of an item (the assistant's pick or another) has changed, with or without the new figure, else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, built from the requirements so far (given below) and what the shopper has explicitly said, each as "property: constraint". Never add a requirement from the story or from your own reading of the request: only values the request states as requirements or the shopper states in a reply count. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category.
"unclear": true when the shopper says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false. When the request asks for several products, such a reply usually means the property asked about is not one the shopper cares about for that product.
"singly": true when the shopper asks to be given the questions singly, one per message, rather than saying they did not follow; else false.
"pointer": when the shopper gives a value by reference to another listing (the same value as that listing shows, a value that goes with it or is its counterpart, anything other than it, or a value no lower or no higher than it; the listing identified as a <product kind> from <seller> carrying <property> <value>...), an object {"product": the product it is about or "", "property": the property whose value is to be read off that listing, named as the shopper names it in this reply (not as the assistant's question named it), "relation": "same", "paired", "different", "at_least", "at_most" or "other" (a relation none of these names), "rule": for "other", the shopper's own words for the relation, else "", "seller": the seller's name as written, "category": the product kind of that listing as written, "attributes": {property: value, ...} as written}; otherwise null. Such a need goes into "needs" and "requirements_now" as ONE entry, "<property>: the same as (or: anything except) the <property> on <seller>'s <category> listing (<attributes>)". The values that identify that other listing are not requirements on the product the shopper is buying: never list them as separate needs.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "". Always "" when "unclear" is true: a request to explain the assistant's own question is not a question to answer. Always "" when the shopper only says they have no preference on the property asked about, even when the line ends by asking what the assistant or most people would pick, or whether it must be settled now: the assistant simply picks, and no message is needed. Always "" when the shopper only asks for the questions one at a time: the next question is the answer.
When the request asks for several different products, start every requirement in "needs" and "requirements_now" with the short name of the product it applies to, as "<product> <property>: constraint" (the shopper often says "<property> of the <product kind>").
One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. Comparison words (at least, short of, below, no less than, or better) are limits only for numbers: with a value that is a word or a name (an interface, a type, a material, a pack type, a connector), the shopper wants exactly that value, so write it as that value alone, never "that or better". A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such a reply applies only to the property the assistant asked about, never drops a stated requirement, a price limit or a rule, and when the assistant's message did not ask about a property it changes nothing."""

VERIFY_PROMPT = """You are the final checker before a shopping assistant places an order. The task scores 1 only if the ordered item meets every requirement below. Judge the item ONLY from the records shown, using these reading rules:
- A requirement is met only when the item's own property for it states the value. Property keys vary in spelling (spaces, dots, underscores, prefixes) and the same property may appear under several keys; any key naming that property counts. Values match by meaning: one element of a comma list is enough, singular/plural and close synonyms count, case does not matter. A merely similar-looking word or a broader/different class does not count. A value written with a qualifier in front of the asked word (a kind or finish of it, such as pu leather or faux leather for leather, faux suede for suede, modern classic for modern) states that value and meets it; an exclusion of the word excludes it too.
- The title or description counts only when the item has no property key for that requirement. If its own property says something else, it fails even when the title says the value.
- A value under a different, related property does not count.
- Numbers: compare after converting units; limits include the edge; a range counts only when the whole range meets the limit.
- Exclusions and closed lists must be respected.
- The item must be the requested kind of product; when listings carry a category label, it must be the category whose label matches the request's wording as a whole, not a narrower or neighbouring one.
- When the item's own property for a requirement states a different value, the requirement fails, even if the title suggests otherwise.
- Price times quantity must be within the budget (equal is fine). A listing's price is a plain number in the same money unit as the budget: compare them as they stand, never convert either. A price requirement listed for the item's kind of product is checked against the item's own price per unit. If the records show it out of stock, it fails.
- A mere wish ("ideally", "would be nice") listed as not required is not a requirement.
- Judge only the requirements listed below; never add one of your own from the story (an age group, a gender, a size, who will use it).
- When the request asks for several different products, judge the item only against the requirements of its own kind of product: a requirement that starts with another product's name does not apply to it, and the quantity is the one for its kind.
- A requirement marked UNRESOLVED (a value that was to be read off another seller's listing and never was) cannot be judged: never fail an item for it.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: firm values of the product's own properties, of the same kind as the requirement the request already states (the properties the store's listings carry for that kind of product), revealed only when a message asks about that exact property. The task scores 0 unless every hidden need is asked about, so the choice of properties decides the score. Each message is answered only about the first thing it asks, and questions about anything else get a polite "don't mind". Choose QUESTION_COUNT different properties (fewer only when fewer useful ones exist), most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then the properties the STORY HINTS below point at. A sentence of the story that tells what the product must fit, hold, match, last for or look like, or how, where or by whom it will be used, without giving the value, is there to point at the one property that settles it: ask that property, by the menu's own name for it, before any other (the hint's guessed names are only a start: read the sentence itself). Map the sentence to what it is about: the space it must go into points at the dimension along that space (how wide, how deep, how tall), how it looks from a distance or in photos at its pattern, colour or style, how it sits on the body at its cut (sleeves, length, fit), how much it must hold, serve or weigh at its volume, weight, count or capacity, how far away a screen is watched at its resolution or size, what it plugs into or runs at at its plug, power or voltage. A sentence that only tells the occasion, the mood or who is buying is not a hint. With several hints, the most plainly pointed property first.
2. then the properties of this kind of product from the menu below, in its order: it is drawn from the opened listings that meet the stated requirements, the most widely carried properties first (a hidden need is most often a property nearly every listing of that kind states), and among those carried about equally the ones whose values differ between listings. Keep that order. Among properties carried about equally, prefer one that a shopper can state as one firm value (a colour, a dimension such as depth, height or width, a type or form, a material or finish, a rating such as voltage or wattage, a packaging or pack type, a compatible model, who it is for); a warranty, a certification or a long free-text field comes last. A property with one value in every listing seen may still be a need.
A story detail is never asked back as it stands: always ask the property it points at. An item bought for another person to wear points at who it is for (gender, age group) when the menu carries that.
A size offered per listing (shoe, clothing or ring sizes), an identifier (part or model number, SKU) or a listing's options list is a detail of the listing, never a hidden need: never ask about them.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never a listing's metadata (brand, maker, seller, title, options list, identifiers), never repeat a property already asked for the same product (a question the shopper did not follow counts as asked, unless that question did not name the product; a reply asking for the questions singly does not count: ask that property again), never spend two questions on near-duplicate properties (two spellings of one property, a property and its "filter" twin), and avoid properties no listing carries: when the menu lists properties for a product, every question about it names one of them by the menu's name (a hinted property no menu key matches only when no menu property fits the hint). Use the plain property name as the listings name it (without prefixes, suffixes or underscores).
When the request asks for several different products, each question is about ONE of them and names it, and the questions are spread over the products: give every product at least one question (its most promising property, story hints first), then spend the rest on the products with the most plausible open properties, preferring products that have had fewer questions so far; take each product's properties from the listings of its own kind (the menu below is grouped by the listings' labels when they carry one).
Write each question as "Which <property> would you like the <product> to have?", with the property's name exactly as the listings name it (spaces instead of underscores or dots) and the product named by its plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values). No example values, no other words, no second question. For something that must be learned first (rule 0), ask it plainly in one short sentence.
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

LATER_ROUND_NOTE = """

THIS IS A LATER ROUND: the answers so far are listed under ALREADY ASKED. A reply that says the shopper did not follow, does not mind or has no preference means that property is not a hidden need for that product: never ask it again. A product rarely has more than one hidden need: when several products are requested, put these questions on the products whose need has NOT been revealed yet (their next most promising properties), and ask about a product whose need is already known only when no other product is left; for a single product, ask its next most promising properties. A property whose question got a reply asking for the questions singly is still open: ask it again with the property's exact name from the listings."""

SWEEP_LAST = False
LIST_SIZE = 5           # properties named in one message (1: one property per message)
# chance that a need named at each place of a list is told (shopper model, races 13-15): what a reply without one says
# about each named property
LIST_REACH = (1.0, 0.7, 0.5, 0.3, 0.25, 0.2, 0.15, 0.1)
SWEEP_NOTE = """

THE LAST QUESTION IN YOUR LIST IS A SWEEP: write it as "Do you have a requirement for the <product>'s <property>, <property> or <property>?", naming two or three further promising properties of ONE product that no other question covers (names as the listings name them, product by its plain noun). Its "property" field lists them joined by ", "."""

REDIRECT_PROMPT = """A shopping assistant must read the PROPERTY_NAME of the listing below. Some listings say, usually in a note, that a property is to be read from a different listing instead (a note naming the property and pointing the reader to another seller's listing of some product, rather than to this one). Return ONE JSON object and nothing else: {"redirect": true when the listing says that its PROPERTY_NAME (or every property) is to be read from another listing, else false, "seller": that other listing's seller or brand name as written, or "", "category": that other listing's kind of product as written, or "", "description": the other words that pick out that listing (values such as colour, type, sizes, specifications) as written, or "", "value": when there is no such redirect, the value this listing itself states for the PROPERTY_NAME (its own property field first, else its notes or description), or null, "pairs": when the listing states a rule (of the store or the seller) on which PROPERTY_NAME values go together, every pair in it as [value, the value it goes with], else []}. A sentence that only says what this item is not, or that sends the reader elsewhere for a DIFFERENT property, is not a redirect for the PROPERTY_NAME."""

PROCESS_PROMPT = """A shopping assistant is about to place the order for the cart below. The task scores 0 unless every action the request demands has been done the way and at the time it demands, so check the record before the order goes in. Return ONE compact JSON object (no indentation) and nothing else: {"missing": [each demanded action not yet done for this very basket, in a few words], "actions": [the tool calls to make now, in order, at most 3, each {"tool": exact tool name, "args": {...}}], "ready": true when nothing is missing}.
How to judge:
- An action demanded right before the order (once the basket is final) counts only when it happened after the last cart change; one made earlier must be made again. An action demanded at another time counts when it was done at that time.
- The demands come from the request's own words, from the shopper's messages (including lines that arrive inside tool results), and from the pages of the items in the cart (a page may say what else is needed, for example another page to read: open the page it points to, searching for its code or name first when its id is not known).
- Judge only the demanded actions. Whether the items meet the requirements, and the search for them, were settled before this point: never look them up again here. THE CART below is the true basket, read from the cart calls, with each line's latest page and the total. Make a look-up only when a demanded action needs a page or a check that the record does not already show after the last cart change.
- A demanded message to the shopper is never put off: when one is missing, it is in this very answer (after any look-up the message depends on, within the same list), written from the cart and pages below. It is ONE plain message that does exactly what was asked, in the shopper's own terms, with one question at most. When it gives the basket or its total, list each line (quantity, name, price) and state the total exactly as shown under TOTAL, in the request's price unit; when the request wants their OK, ask for it. Never ask about preferences here, never repeat an action already done for this basket, and never use the cart, order or decline tools.
- Use no more messages or steps than are left; when nothing is missing or nothing more can be done, return "ready": true with no actions.
"""

TELL_PROMPT = """A shopping assistant is about to place the order for the cart below (or, at the CART STAGE, to put the lines named there into the cart), and the request demands the actions under DUE NOW right before the order, made by a message to the shopper. Write that ONE message. It does exactly what those actions ask, in the shopper's own terms, with one question at most (for example asking for their OK when the request wants it). When it gives the basket or its total, list each line (quantity, name, price) and state the total exactly as shown under TOTAL, in the request's price unit. Say nothing about preferences, searches or anything else. Return ONE compact JSON object and nothing else: {"message": "..."}."""

SEARCH_PLAN_PROMPT = """You plan the next store searches of a shopping assistant. Below are the shopper's request, the requirements in force, the store's search tool (its description and argument schema), what the assistant has learned about how this store's search behaves, and the searches made so far with what they returned. Write the next searches most likely to show listings that meet every requirement (for a request that wants the cheapest, the cheapest such listing), best first, two to four of them. Follow the learned behaviour exactly: when the store reads only the first words of a query, a query is those words and nothing more, so lead with the most telling ones (a required value's own word or figure and the product's plain noun, value first); when every word is required, a value with the noun lists the listings that carry it and more words narrow; when any word matches, a value's distinctive word alone finds the listings carrying it. When a category listing can show only part of a shelf, reach the rest with keyword searches (the product's plain names, other names for it, words from the titles of fitting listings seen, a maker or seller named in the request). Use only argument names and values the schema allows; never repeat a search already made with the same arguments (a further page of one is fine when its total shows more rows than were read). Claim nothing about the store's search beyond what is listed as learned, and add no rule of your own to the request: "advice" is about how to search and check, and names the cheapest only when the request asks for it. Return ONE compact JSON object and nothing else: {"searches": [{"product": the product it is for, "arguments": {the exact arguments}, "why": a few words}], "advice": "one short sentence on how to go on from these results"}."""

LINK_PROMPT = """A shopping assistant is buying several products, and the request ties one of them to another: the rule below sets what is required of the PRODUCT, depending on the item bought for the OTHER product. That item is now in the cart. From its records (its page, description, notes and properties) work out what the rule now demands of the PRODUCT. For a condition, check whether the other item's own records state the property value(s) the condition names (by meaning; a different value means it does not hold). For a fit rule, find what the other item's records say the PRODUCT must carry to fit it (a model, a series or part code, a size, a connector, a compatibility), and how listings of the PRODUCT show it (the sample listings below may show the pattern, for example a code in their titles). An item often goes by more than one name in its records (a model number plus another name or code it is also known by): a listing that names any of them fits, so write the requirement with every one of them, joined by " or ", and search with one of them. Several such rules may be listed (RULE 0, RULE 1, ...): answer each. Return ONE compact JSON object and nothing else: {"links": [one object per rule: {"rule": its number, "other_item": the id of the cart line that is that rule's OTHER product, or null, "holds": for a condition true or false (null when its records do not show the property), else null, "requirement": what the PRODUCT now needs, as "<product> <property>: <value>" with the value exactly as the records write it, or "" when nothing (a condition that does not hold), "search": a few words to search for fitting listings of the PRODUCT (the code or model plus the product kind), or "", "product_line_ok": true or false when the cart already holds a line of the PRODUCT and its records show whether it meets that requirement, else null, "none_fit": true when the sample listings of the PRODUCT include none that meets that requirement, else false}]}."""

CHANGE_PROMPT = """A shopping assistant has a pick in the cart, and the shopper has just announced a change (below). Check each listing below against the CURRENT requirements, judging it ONLY from its own records with these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms count, and so does the value with a qualifier in front (pu leather for leather, faux suede for suede); a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after converting units, limits include the edge; a requirement marked UNRESOLVED cannot fail anything; only the listed requirements count, never one assumed from the story, and a requirement the shopper no longer needs does not count; when several different products are requested, a listing is checked only against the requirements of its own kind of product. Stock, quantity and the total budget are checked separately: ignore them here (a listed price requirement for one kind of product does count: compare the listing's own price as it stands, it is in the same money unit, never convert it).
Return ONE compact JSON object and nothing else, with no reasoning: {"fits": [the ids of the listings that meet every current requirement], "cart_line": "the requirement THE PICK IN THE CART fails, or an empty string"}."""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Check the listing records below against the requirements, using these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms all count, and so does the value with a qualifier in front (pu leather for leather, faux suede for suede); a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after converting units, limits include the edge; the price must be within the budget (a listing's price is in the budget's own money unit, never convert it) and the item must not be shown out of stock; only the listed requirements count, never one assumed from the story, and a requirement marked UNRESOLVED cannot fail anything; when several different products are requested, a listing is checked only against the requirements of its own kind of product. A wrong decline scores 0 and nearly every request has a fitting listing, so list every listing that may fit; leave one out only when its own records clearly fail a listed requirement.
Return ONE JSON object and nothing else: {"possible_fits": [ids of listings that appear to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
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


def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    """Long text cut to its start; in a property list, one of the keys that spell the same property with the same value."""
    if depth > 8:
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        return [_trim(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        seen: set[tuple[str, str]] = set()
        for key, item in value.items():
            if isinstance(item, str) and len(value) >= 4:
                mark = (re.sub(r"[^a-z0-9]+", "", str(key).lower()), item.strip().lower())
                if mark in seen:
                    continue
                seen.add(mark)
            out[key] = _trim(item, limit, depth + 1)
        return out
    return value


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


def _cart_lines(args: dict[str, Any]) -> list[tuple[str, Any]] | None:
    """(listing id, quantity) per line of a cart call that passes its lines as a list (None when it has no such list)."""
    def id_like(key: Any, value: Any) -> bool:
        return isinstance(value, str) and bool(_ID_HINT.match(value)) and any(c.isdigit() for c in value) and bool(_IDENT_KEYS.search(str(key)))

    if any(id_like(k, v) for k, v in args.items()):
        return None             # the call names its listing directly; a list beside it holds choices, not lines
    lists = [v for v in args.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
    # one id per line (the first id-like value, normally the listing; an option id that follows it is shown with the listing)
    out = []
    for lines in lists:
        for line in lines:
            ident = next((v for k, v in line.items() if id_like(k, v)), None)
            if ident is None:
                # a key that does not read as an id ("whichItem"): the line's first id-looking value
                ident = next((v for v in line.values() if isinstance(v, str) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), None)
            qty = next((v for k, v in line.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and _QTY_KEYS.search(str(k))
                        and not re.search(r"price|cost", str(k), re.I)), None)
            if ident:
                out.append((ident, qty))
    return out or None


def _listed_ids(args: dict[str, Any]) -> list[str] | None:
    """Listing ids in a cart call that passes its lines as a list (None when the call has no such list)."""
    lines = _cart_lines(args)
    return [ident for ident, _ in lines] if lines else None


def _words_of(text: str) -> set[str]:
    """The content words of a property name (singular, colour = color)."""
    out = set()
    for w in re.findall(r"[a-z0-9]+", str(text).lower().replace("colour", "color")):
        if len(w) < 2 or w in ("the", "of", "and", "for", "a", "an", "its"):
            continue
        out.add(w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w)
    return out


_PROPERTY_BLOCK = re.compile(r"^(facts?|propert(y|ies)|attributes?|specs?|specifications?|traits?|features?|characteristics?)$", re.I)


def _listing_records(value: Any, depth: int = 0):
    """Every dict in a result that carries a listing id (a listing's own property block is not looked into)."""
    if depth > 5:
        return
    if isinstance(value, dict):
        if any(isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v) for k, v in value.items()):
            yield value
        for key, item in value.items():
            if isinstance(item, dict) and _PROPERTY_BLOCK.match(str(key).strip()):
                continue
            yield from _listing_records(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            yield from _listing_records(item, depth + 1)


def _ident_of(record: dict[str, Any]) -> str:
    return next((v for k, v in record.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), "?")


def _num(value: Any) -> str:
    return str(int(value)) if isinstance(value, (int, float)) and float(value).is_integer() else str(round(value, 2))


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
                           r"blurb|tagline|caption|teaser|slogan|name|overview|summary|description)\b|"
                           r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
                           r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
                           r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
# an error that says a tool's allowance is spent (not a bad argument or an over-budget cart)
_USED_UP = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                      r"reached the limit|maximum (number|reached)|already used", re.I)
_IDENT_KEYS = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
_QTY_KEYS = re.compile(r"qty|quant|piece|count|unit|amount|number|many|wanted", re.I)
_TEXT_KEYS = re.compile(r"note|know|blurb|descr|text|about|remark|comment|summary|overview|info", re.I)
_TITLE_KEYS = re.compile(r"title|headline|heading|name|caption|label", re.I)
_OPTION_KEYS = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
_KIND_KEYS = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                        r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
# a key that marks a pending step of a checkout (a token to confirm with), not a refusal
_FIRST_PERSON = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
# a shopper line that wants an answer, and a request rule that wants the shopper told something
_ASKS = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
# an answer that is more than a plain acknowledgement
_NOT_ACK = re.compile(r"\d|¢|\$|€|£|\?|\b(no|not|never|instead|rather|only|without|but)\b|n't|longer|scrap|drop|swap|remov|replac|chang|cheap|budget|price|cost|must|need|want", re.I)
_TELL_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order|sale)|reject|cancel|quit|skip|abstain|withdraw|stand.?down|forgo|walk.?away", re.I)
_NO_PURCHASE = re.compile(r"no purchase|without (a |any )?(buying|purchas|order)|(with )?nothing (is |gets )?(bought|purchased|ordered)|not (buy|purchas|order)|no order|empty.handed", re.I)
_FILLER = re.compile(r"^(the|an?|and|or|with|of|in|on|to|for|from|by|its?|this|that|which|has|have|is|are|not|one|s|items?|listings?|products?|made|piece)$")
_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")


_PRICE_KEY = re.compile(r"price|cost|amount", re.I)
_NOT_PRICE = re.compile(r"total|sum|ship|deliver", re.I)
_OLD_PRICE = re.compile(r"was|old|list|original|regular|compare|before", re.I)


def _price_keys(record: dict[str, Any]) -> list[str]:
    """A record's keys holding its price: the current price first; a "list" or "was" price only when it is the only one
    (race 9's listings carry their price as item_list_price)."""
    keys = [k for k, v in record.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
            and _PRICE_KEY.search(str(k)) and not _NOT_PRICE.search(str(k))]
    fresh = [k for k in keys if not _OLD_PRICE.search(str(k))]
    return fresh or keys


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
            value = value[: schema["maxLength"]]
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


_ROW_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_ .\-]{0,40}):\s?(.*)$", re.S)
_PAIR_FIELD = re.compile(r"^([A-Za-z_][A-Za-z0-9_ .\-/()]{0,60})=(.*)$", re.S)


def _literal(text: str) -> Any:
    value = text.strip()
    if value in ("True", "true"):
        return True
    if value in ("False", "false"):
        return False
    if value in ("None", "null"):
        return None
    if re.fullmatch(r"-?\d{1,15}", value):
        return int(value)
    if re.fullmatch(r"-?\d{1,15}\.\d+", value):
        return float(value)
    return value


def _pair_list(text: str) -> dict[str, Any] | None:
    """'name=value; name=value' as an object, when the whole text is such a list of two pairs or more."""
    out: dict[str, Any] = {}
    last = None
    for part in text.split("; "):
        match = _PAIR_FIELD.match(part.strip())
        if match and match.group(1).strip() not in out:
            last = match.group(1).strip()
            out[last] = match.group(2).strip()
        elif last is not None:
            out[last] += "; " + part
        else:
            return None
    return {k: _literal(v) for k, v in out.items()} if len(out) >= 2 else None


def _text_row(text: str) -> dict[str, Any] | None:
    """A listing written as one string of 'key: value' fields joined by ' | ', as an object (None when it is not one)."""
    if text.count(" | ") < 3 or len(text) > 40000:
        return None
    fields: dict[str, str] = {}
    last = None
    for part in text.split(" | "):
        match = _ROW_FIELD.match(part)
        if match and match.group(1).strip() not in fields:
            last = match.group(1).strip()
            fields[last] = match.group(2)
        elif last is not None:
            fields[last] += " | " + part
        else:
            return None
    if len(fields) < 4 or not any(_IDENT_KEYS.search(k) and _ID_HINT.match(v.strip()) and any(c.isdigit() for c in v)
                                  for k, v in fields.items()):
        return None
    row: dict[str, Any] = {}
    for key, value in fields.items():
        if _IDENT_KEYS.search(key) and _ID_HINT.match(value.strip()):
            row[key] = value.strip()
            continue
        pairs = _pair_list(value) if "=" in value and "; " in value else None
        row[key] = pairs if pairs is not None else _literal(value)
    return row


def _decode_rows(value: Any, depth: int = 0) -> Any:
    """Listings sent as text rows, read as objects; everything else as it came."""
    if depth > 8:
        return value
    if isinstance(value, str):
        row = _text_row(value) if " | " in value else None
        return row if row is not None else value
    if isinstance(value, list):
        return [_decode_rows(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _decode_rows(v, depth + 1) for k, v in value.items()}
    return value


def _collect_ids(value: Any, out: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and "id" in key.lower() and _ID_HINT.match(item):
                out.add(item)
            elif not (isinstance(item, dict) and _PROPERTY_BLOCK.match(str(key).strip())):
                _collect_ids(item, out, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _collect_ids(item, out, depth + 1)


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
        self.process: list[dict[str, Any]] = []
        self.process_at = -1            # step of the last action the pre-order process check made
        self.process_checks: dict[int, int] = {}
        self.cart_qty: dict[str, Any] = {}
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checks = 0
        self.cart_warned = False
        self.opened: set[str] = set()
        self.cheaper_checks = 0
        self.early_bounced = False
        self.refused_at = -1        # step of the last refused order
        self.searched_at = -1       # step of the last search
        self.research_checks = 0
        self.cart_used = False
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.gone: set[str] = set()                 # listings the shopper said can no longer be bought
        self.change_due: set[str] = set()           # kinds of change heard while the cart held a pick, not checked yet
        self.change_heard_at = -1
        self.change_checks = 0
        self.repick: str | None = None
        self.side_notes: list[str] = []             # what calls made for the model brought (shopper lines), shown next
        self.size_fails: dict[str, int] = {}
        self.search_kind: dict[str, str] = {}       # a code search's phrase -> the product kind it is for (a shelf filter)
        self.closest_asks = 0
        self.audit_fits: list[str] = []
        self.substitutes = 0
        self.substitute_due: list[dict[str, Any]] | None = None
        self.cart_shape_asks = 0
        self.chain_holds = 0
        self.cheapest_checks = 0
        self.cart_overrides: set[tuple[str, int]] = set()
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
        self.cart_lines: dict[str, str] = {}
        self.plan_retry_at = 0
        self.audited = False
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
        self.hints: list[dict[str, Any]] = []
        self.pointers: list[dict[str, Any]] = []
        self.reask: list[str] = []
        self.round_pending = False
        self.rounds = 0
        self.last_reading: dict[str, Any] = {}
        self.list_queue: dict[str, list[list[Any]]] | None = None     # product -> [[property, weight], ...]
        self.list_order: list[str] = []
        self.list_hits: dict[str, int] = {}
        self.list_sent: dict[str, int] = {}
        self.sweep_due = False
        self.swept: set[str] = set()
        self.plan_rest: list[str] = []
        self.redirects: dict[str, dict[str, Any]] = {}
        self.views: dict[str, dict[str, Any]] = {}      # every fact seen for a listing, merged over the tools that show it
        self.traits: dict[str, dict[str, bool]] = {}    # what each listing tool's results carry: page text, a property table
        self.paged: dict[str, int] = {}                 # listings opened in full (page text shown), with the step
        self.tabled: set[str] = set()                   # listings whose property list has been shown
        self.opened_with: dict[str, set[str]] = {}      # per tool: the listings it has shown
        self.open_queue: list[tuple[str, str]] = []     # (tool, listing) a reference reading waits for
        self.open_tried: set[str] = set()
        self.message_steps: list[int] = []              # steps of messages that were not planned questions
        self.many_ok: dict[str, int] = {}               # most listings a several-at-once tool has shown in one call
        self.many_cap: dict[str, int] = {}              # most it takes, learned from a call that failed with more
        self.chased: set[str] = set()                   # searches made for a reference's next seller
        self.menu_filled = False
        self.cut_notes: dict[str, str] = {}
        self.multi_failed: set[str] = set()             # tools taking a list of ids that showed no listing pages
        self.pair_rules: dict[str, dict[str, str]] = {}  # property -> value -> the value the store's rule pairs it with
        self.filter_stats: dict[tuple[str, str], list[int]] = {}
        self.filter_warned: set[tuple[str, str]] = set()
        self.search_fallback: dict[str, str] = {}       # a chase search with the description's words -> the plain one
        self.searches: dict[str, dict[str, Any]] = {}   # per search wording: total reported, rows seen, next page token
        self.phrase_words: dict[str, int] = {}          # per search argument: words the model put in it over its own searches
        self.pages_read = 0
        self.reach_noted: set[str] = set()     # searches whose pages were said to stop short of every match
        self.search_log: list[dict[str, Any]] = []          # first-page searches: context, words, total, row ids
        self.search_sigs: set[str] = set()                    # every search's arguments, as run
        self.search_rule: dict[str, Any] = {"low": -1, "high": set(), "words": "", "category": None}
        self.rules_told: set[str] = set()
        self.empty_streak = 0
        self.search_plans = 0
        self.search_plan_due = ""
        self.ask_plan_done = True               # (43g) no plan when the questions are over: the learned rules are in the status
        self.word_probed = False                # the search was probed once with a query's first two words
        self.wider_held = False                 # a cheapest order was held once for searches under other names
        self.links: list[dict[str, Any]] = []
        self.link_done: dict[int, str] = {}             # link -> the cart lines it was last worked out for
        self.derived: dict[int, str] = {}               # link -> the requirement it now puts on its product
        self.link_other: dict[int, str] = {}            # link -> the cart item it was worked out from
        self.compacted_at = 0
        self.links_due = False
        self.cart_stage_done = False
        self.cost = 0.0
        self.cost_est = 0.0
        self.context_tokens = 0
        self.trace: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []

    def time_left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def due_before_order(self) -> list[dict[str, Any]]:
        """The request's actions due once the basket is final, while none was made since the last cart change."""
        if self.process_at > self.cart_changed_at:
            return []
        return [p for p in self.process if p.get("when") == "before_order"]

    def kept_messages(self) -> int:
        # kept for every action due right before the order, also after it was made once: a cart change makes it due again
        if not self.roles.get("message"):
            return 0
        stages = ("before_order",) if self.cart_stage_done or self.cart_used else ("before_order", "before_cart")
        return min(2, sum(p.get("messages") or 0 for p in self.process if p.get("when") in stages))

    def kept_steps(self) -> int:
        return min(3, sum(max(p.get("steps") or 0, p.get("messages") or 0) for p in self.process if p.get("when") == "before_order"))

    def question_limit(self) -> int:
        """Messages that questions and replies may use: those the request's last actions need are kept."""
        if self.message_budget is None:
            return 3
        return max(self.messages_used, self.message_budget - self.kept_messages())

    def messages_left(self) -> int:
        return 99 if self.message_budget is None else self.message_budget - self.messages_used

    def listing_facts(self, ident: str) -> tuple[str, Any]:
        """The title and the price of a listing, read off the results seen so far."""
        title, price = "", None

        def walk(value: Any, depth: int) -> None:
            nonlocal title, price
            if depth > 6:
                return
            if isinstance(value, dict):
                if ident in value.values():
                    for k, v in value.items():
                        if isinstance(v, str) and not title and re.search(r"name|title|caption|heading|label", str(k), re.I) and len(v) >= 4 and v != ident:
                            title = v
                    keys = _price_keys(value)
                    if price is None and keys:
                        price = value[keys[0]]
                for v in value.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for record in reversed(self.records):
            walk(record, 0)
            if title and price is not None:
                break
        return title, price

    def cart_total(self) -> tuple[Any, Any]:
        """The basket's total as price x quantity summed over its lines (None when a price is unknown), and the total
        the cart view shows when the cart was viewed after the last cart change."""
        computed: Any = 0
        for ident in self.cart_ids:
            _, price = self.listing_facts(ident)
            qty = self.cart_qty.get(ident)
            qty = qty if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 0 else 1
            if computed is None or not isinstance(price, (int, float)) or isinstance(price, bool):
                computed = None
                continue
            computed += price * qty
        shown = None
        if self.cart_view_at > self.cart_changed_at:
            def walk(value: Any, depth: int) -> Any:
                if depth > 4 or not isinstance(value, dict):
                    return None
                for k, v in value.items():
                    if (isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"total|grand", str(k), re.I)
                            and not re.search(r"sub|line|item|count|qty|quant|piece|unit", str(k), re.I)):
                        return v
                for v in value.values():
                    found = walk(v, depth + 1)
                    if found is not None:
                        return found
                return None
            shown = walk(self.last_cart_view, 0)
        return (computed if self.cart_ids else None), shown

    def cart_text(self) -> str:
        lines = []
        for ident in self.cart_ids[:8]:
            title, price = self.listing_facts(ident)
            qty = self.cart_qty.get(ident)
            qty = qty if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 0 else 1
            page = self.views.get(ident) or next((r for r in reversed(self.records) if isinstance(r, dict) and ident in r.values()
                                                  and any(isinstance(v, (str, dict)) and len(_compact(v)) > 40 for v in r.values())), None)
            text = _compact(_brief({k: v for k, v in page.items() if k not in ("title",)}, 300))[:1200] if page else ""
            full = "opened in full" if ident in self.paged else "not opened in full"
            lines.append(f"- {ident}: {_num(qty)} x {title or '?'} at {_num(price) if price is not None else '?'} each ({full}); its page: {text}")
        if not lines:
            return "(empty)"
        computed, shown = self.cart_total()
        if computed is not None:
            lines.append(f"TOTAL (each line's price x quantity, summed): {_num(computed)} {self.plan.get('currency') or ''}".rstrip())
        if shown is not None and shown != computed:
            lines.append(f"The cart view read after the last cart change shows the total {_num(shown)}: that figure is the total.")
        return "\n".join(lines)

    def action_log(self) -> str:
        out = []
        busy = {self.roles.get(r) for r in ("message", "cart", "order", "decline")} - {None}
        for entry in self.trace:
            if entry.get("kind") not in ("env", "process", "auto_open"):
                continue
            reply = entry.get("reply") or {}
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, action in enumerate(entry.get("actions") or []):
                result = observations[index] if index < len(observations) else None
                name = action.get("name")
                out.append(f"{name} {_compact(action.get('args'))[:200]} -> {_compact(result)[:300 if name in busy else 120]}")
        return "\n".join(out[-40:])

    def learn_listings(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Merge what a result shows about each listing into one view per listing, and learn whether this tool shows a
        listing's full page (notes, description) or only its property table."""
        if not isinstance(observation, (dict, list)):
            return
        core = {self.roles.get(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        named = {v for v in args.values() if isinstance(v, str)} | {x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str)}
        for record in _listing_records(observation):
            ident = _ident_of(record)
            if ident == "?":
                continue
            view = self.views.setdefault(ident, {})
            for key, value in record.items():
                if value is None and key in view:
                    continue
                if isinstance(value, dict) and isinstance(view.get(key), dict):
                    view[key] = {**view[key], **value}
                else:
                    view[key] = value
            if not name or name in core or ident not in named:
                continue
            self.opened_with.setdefault(name, set()).add(ident)
            text = any(_TEXT_KEYS.search(str(k)) and not _TITLE_KEYS.search(str(k)) and (v is None or isinstance(v, str))
                       for k, v in record.items())
            table = any(isinstance(v, dict) and sum(isinstance(x, (str, int, float)) for x in v.values()) >= 3 for v in record.values())
            if table:
                self.tabled.add(ident)
            trait = self.traits.setdefault(name, {"text": False, "table": False})
            trait["text"] = trait["text"] or text
            trait["table"] = trait["table"] or table
            if text:
                self.paged[ident] = self.step

    def id_argument(self, tool: str | None) -> tuple[str, bool] | None:
        """The argument that names the listing(s) for a tool that needs nothing else, and whether it takes several."""
        schema = self.schemas.get(tool or "") or {}
        properties = schema.get("properties") or {}
        required = [k for k in schema.get("required") or [] if k in properties]
        many = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "array"
                and isinstance(v.get("items"), dict) and v["items"].get("type") == "string"]
        one = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
        one.sort(key=lambda k: 0 if _IDENT_KEYS.search(k) else 1)
        for key, several in [(k, True) for k in many] + [(k, False) for k in one]:
            if all(r == key for r in required):
                return key, several
        return None

    def multi_page_tool(self) -> str | None:
        """The tool that shows several listings' full pages in one call: known from its results; else, once the store's
        single-listing page tool is known, a tool not used yet whose one required argument is a list of listing ids (it is
        tried, and learned from its first result)."""
        known = next((n for n, t in self.traits.items() if t["text"] and (self.id_argument(n) or ("", False))[1]
                      and n not in self.multi_failed and self.many_cap.get(n, 2) > 1), None)
        if known:
            return known
        single = [n for n, t in self.traits.items() if t["text"] and self.id_argument(n)]
        single += [r for r in (self.roles.get("details"), self.roles.get("view")) if r and self.id_argument(r) and not self.id_argument(r)[1]]
        if not single:
            return None
        core = {self.roles.get(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        for tool in self.tools:
            name = tool["function"]["name"]
            argument = self.id_argument(name)
            if name in core or name in self.traits or name in self.multi_failed or not argument or not argument[1]:
                continue
            return name
        return None

    def page_tool(self) -> str | None:
        """The tool that opens a listing's full page (several at once first). Known from results; while none has shown it
        in a store that splits listings (a tool showing only property tables), the setup's view or details tool."""
        multi = self.multi_page_tool()
        if multi:
            return multi
        known = [n for n, t in self.traits.items() if t["text"] and self.id_argument(n)]
        if any(t["table"] and not t["text"] for t in self.traits.values()):
            known += [r for r in (self.roles.get("view"), self.roles.get("details")) if r and r not in self.traits and r not in known and self.id_argument(r)]
        if not known:
            return None
        many = [n for n in known if self.id_argument(n)[1] and self.many_cap.get(n, 2) > 1 and n not in self.multi_failed]
        return (many or known)[0]

    def table_tool(self) -> str | None:
        return next((n for n, t in self.traits.items() if t["table"] and not t["text"] and self.id_argument(n)), None)

    def missing_views(self, ident: str) -> list[str]:
        """The tools still to open for a listing a reference leads to: its full page (the notes) and its property list."""
        if self.steps_left() <= self.finish_reserve() + 1 or self.time_left() < WRAP_UP_SECONDS + 20:
            return []
        out = []
        page = self.page_tool()
        if page and ident not in self.paged and f"{page}|{ident}" not in self.open_tried:
            out.append(page)
        table = self.table_tool()
        if (table and table not in out and f"{table}|{ident}" not in self.open_tried
                and not any(isinstance(v, dict) and len(v) >= 3 for v in (self.views.get(ident) or {}).values())):
            out.append(table)
        return out

    def many_limit(self, tool: str) -> int:
        """How many listings a several-at-once tool is given per call: its schema's limit, else what it was seen to take."""
        argument = self.id_argument(tool)
        spec = (self.schemas[tool].get("properties") or {}).get(argument[0]) or {} if argument else {}
        cap = self.many_cap.get(tool) or max(5, self.many_ok.get(tool, 0))
        if isinstance(spec.get("maxItems"), int) and spec.get("maxItems") > 0:
            cap = min(cap, spec["maxItems"])
        return max(1, cap)

    def text_argument(self, tool: str | None) -> str | None:
        """The one text argument a tool needs (a search phrase), when it needs nothing else."""
        schema = self.schemas.get(tool or "") or {}
        properties = schema.get("properties") or {}
        required = [k for k in schema.get("required") or [] if k in properties]
        texts = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v]
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        return target if target and all(k == target for k in required) else None

    def search_text_key(self) -> str | None:
        """The search tool's text argument (the phrase), also when the tool needs a mode choice besides it."""
        schema = self.schemas.get(self.roles.get("search") or "") or {}
        properties = schema.get("properties") or {}
        texts = [k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "string" and "enum" not in v
                 and not re.search(r"next|page|cursor|token|after|onward|offset|continu|marker", k, re.I)]
        if not texts:
            return None
        used = [k for k in texts if self.phrase_words.get(k)]
        if used:
            return max(used, key=lambda k: self.phrase_words[k])     # where the model itself writes its search phrases
        required = schema.get("required") or []
        return (next((k for k in texts if re.search(r"phrase|query|text|keyword|term|search|words|^q$|^q[_-]|[_-]q$", k, re.I)), None)
                or next((k for k in texts if k in required), None) or texts[0])

    def fit_phrase(self, phrase: str) -> str:
        """A search phrase cut to the length the search tool's text field takes, at a word boundary."""
        phrase = str(phrase or "").strip()
        key = self.search_text_key()
        properties = (self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}
        limit = (properties.get(key) or {}).get("maxLength") if key and isinstance(properties.get(key), dict) else None
        if isinstance(limit, int) and limit > 0 and len(phrase) > limit:
            cut = phrase[:limit].strip()
            phrase = (cut.rsplit(" ", 1)[0] if " " in cut and not phrase[limit:limit + 1].isspace() else cut).strip()
        return phrase

    def search_args(self, phrase: str, kind: str = "") -> dict[str, Any] | None:
        """Arguments of a keyword search made by code: the phrase, a required choice set to its keyword-search value, and
        the store's shelf for the product kind when it offers one by that name."""
        search = self.roles.get("search")
        schema = self.schemas.get(search or "") or {}
        properties = schema.get("properties") or {}
        key = self.search_text_key()
        if not key:
            return None
        phrase = self.fit_phrase(phrase)
        args: dict[str, Any] = {key: phrase}
        for name in schema.get("required") or []:
            if name == key or name not in properties:
                continue
            options = [v for v in (properties[name] or {}).get("enum") or [] if isinstance(v, str)]
            if not options:
                return None
            args[name] = next((v for v in options if re.search(r"text|keyword|word|query|search|free|match", v, re.I)), options[0])
        if kind:
            def squash(text: Any) -> str:
                return re.sub(r"[^a-z0-9]+", "", str(text).lower()).rstrip("s")
            for name, spec in properties.items():
                if name in args or not isinstance(spec, dict) or not spec.get("enum") or not _KIND_KEYS.search(name):
                    continue
                match = next((v for v in spec["enum"] if isinstance(v, str) and squash(v) == squash(kind)), None)
                if match is not None:
                    args[name] = match
                    break
        checked, problems = _check(args, schema, "args")
        return None if problems else checked

    def open_actions(self, wanted: list[tuple[str, str]]) -> list[dict[str, Any]]:
        by_tool: dict[str, list[str]] = {}
        for tool, ident in wanted:
            if ident not in by_tool.setdefault(tool, []):
                by_tool[tool].append(ident)
        actions = []
        for tool, idents in by_tool.items():
            if tool == "?search":
                search = self.roles.get("search")
                for phrase in idents if search else []:
                    args = self.search_args(phrase, self.search_kind.get(phrase.lower(), ""))
                    if args is not None:
                        actions.append({"name": search, "args": args})
                continue
            argument = self.id_argument(tool)
            if argument is None:
                continue
            key, several = argument
            if several:
                cap = self.many_limit(tool)
                candidates = [{key: idents[i:i + cap]} for i in range(0, len(idents), cap)]
            else:
                candidates = [{key: ident} for ident in idents]
            for args in candidates:
                args, problems = _check(args, self.schemas[tool], "args")
                if not problems:
                    actions.append({"name": tool, "args": args})
        return actions

    def open_queued(self) -> str:
        """Follow references for the model: open the pages a reference reading waits for (full pages with their notes,
        then the property list when the value is needed), search for the next seller a note sends the reading to, and
        open that seller's listings, a few rounds while steps allow."""
        notes = []
        for _ in range(6):
            queue, self.open_queue = self.open_queue, []
            open_reference = any(p["value"] is None for p in self.pointers)
            if (not queue or self.finished or self.dead or self.time_left() < WRAP_UP_SECONDS
                    or self.steps_left() <= (self.finish_reserve() + 1 if open_reference else self.kept_steps() + 8)):
                self.open_queue = queue + self.open_queue
                break
            actions = self.open_actions(queue)[:2]
            if not actions:
                break
            done = set()
            sent = [a["args"] for a in actions]
            for tool, ident in queue:
                if tool == "?search":
                    built = self.search_args(ident, self.search_kind.get(str(ident).lower(), ""))
                    if built is None or built in sent:
                        done.add(f"?search|{ident}")    # made now, or cannot be made: never left waiting
            for action in actions:
                for value in action["args"].values():
                    for ident in value if isinstance(value, list) else [value]:
                        done.add(f"{'?search' if action['name'] == self.roles.get('search') else action['name']}|{ident}")
            self.open_tried.update(done)
            self.open_queue += [entry for entry in queue if f"{entry[0]}|{entry[1]}" not in done and entry not in self.open_queue]
            reply = self.env_call(actions)
            self.trace.append({"kind": "auto_open", "actions": actions, "reply": reply})
            if reply is None:
                break
            self.recent = []
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, action in enumerate(actions):
                observation = observations[index] if index < len(observations) else None
                self.records.append(observation)
                if isinstance(observation, (dict, list)):
                    _collect_ids(observation, self.seen_ids)
                self.learn_listings(action["name"], action["args"], observation)
                if self.learn_size(action["name"], action["args"], observation):
                    self.requeue_refused(action["name"], action["args"])
                if action["name"] == self.roles.get("search"):
                    self.searched_at = self.step
                    self.note_search(action["args"], observation)
                    try:
                        self.learn_search(action["name"], action["args"], observation, told=False)
                    except Exception as exc:
                        self.trace.append({"kind": "crash", "where": "learn_search", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    self.queue_seller_rows(observation)
                    key = self.search_text_key()
                    phrase = str(action["args"].get(key) or "").strip().lower() if key else ""
                    fallback = self.search_fallback.pop(phrase, None)
                    if fallback and not any(True for _ in _listing_records(observation)) and fallback.lower() not in self.chased:
                        self.chased.add(fallback.lower())
                        if self.search_kind.get(phrase):
                            self.search_kind[fallback.lower()] = self.search_kind[phrase]
                        self.open_queue.append(("?search", fallback))
                else:
                    self.opened.update(v for v in action["args"].values() if isinstance(v, str))
                    self.opened.update(x for v in action["args"].values() if isinstance(v, list) for x in v if isinstance(x, str))
                self.take_lines(observation)
                resolved = self.resolve_pointers(observation)
                notes.append(f"{action['name']} {_compact(action['args'])} was run for you (following a reference or a linked rule): "
                             + _compact(self.shown(action["name"], observation))[:3500] + (" " + resolved if resolved else ""))
        return " ".join(notes)

    def widen_seller(self, pointer: dict[str, Any], exclude: str) -> bool:
        """Queue the full pages of the pointer's seller's listings seen in results and not opened yet (several at once);
        False when there are none left."""
        page = self.page_tool()
        if not page or self.steps_left() <= self.finish_reserve() + 1:
            return False
        seller = pointer["seller"].lower()
        tried = pointer.setdefault("widened", [])
        found = []
        for ident, view in self.views.items():
            if (ident == exclude or ident in self.paged or ident in pointer["hops"] or ident in tried or f"{page}|{ident}" in self.open_tried
                    or not any(isinstance(v, str) and v.strip().lower() == seller for v in view.values())):
                continue
            found.append(ident)
        limit = self.many_limit(page) if (self.id_argument(page) or ("", False))[1] else 2
        for ident in found[:limit]:
            tried.append(ident)
            if (page, ident) not in self.open_queue:
                self.open_queue.append((page, ident))
        if found:
            self.trace.append({"kind": "widen_seller", "seller": pointer["seller"], "ids": found[:limit]})
        return bool(found)

    def queue_seller_rows(self, observation: Any) -> None:
        """Rows of a search result that belong to the seller an open reference names are opened in full next (several
        at once), those showing more of the reference's identifying words first."""
        page = self.page_tool()
        if not page or not isinstance(observation, (dict, list)):
            return
        for pointer in self.pointers:
            if pointer["value"] is not None:
                continue
            seller = pointer["seller"].lower()
            words = [w for w in re.split(r"[^a-z0-9]+", " ".join(list(pointer["attributes"].values()) + [pointer["describe"], pointer["category"]]).lower())
                     if len(w) >= 2 and not _FILLER.match(w)]
            rows = []
            for record in _listing_records(observation):
                ident = _ident_of(record)
                if (ident == "?" or ident in self.paged or ident in pointer["hops"] or f"{page}|{ident}" in self.open_tried
                        or not any(isinstance(v, str) and v.strip().lower() == seller for v in record.values())):
                    continue
                text = _compact(record).lower()
                rows.append((-sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", text)), ident))
            limit = self.many_limit(page) if (self.id_argument(page) or ("", False))[1] else 2
            for _, ident in sorted(rows)[:limit]:
                if (page, ident) not in self.open_queue:
                    self.open_queue.append((page, ident))

    def learn_size(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """How many listings a several-at-once tool takes. A call that shows only the first few of the listings asked sets
        the limit; one that shows none with a short status (the tool declares more than it answers) is tried smaller
        before the tool is given up as a page tool."""
        argument = self.id_argument(name)
        if not argument or not argument[1] or not isinstance(args.get(argument[0]), list):
            return ""
        asked = [str(x) for x in args[argument[0]]]
        count = len(asked)
        shown = []
        for record in _listing_records(observation):
            ident = _ident_of(record)
            if ident in asked and ident not in shown:
                shown.append(ident)
        if shown:
            self.many_ok[name] = max(self.many_ok.get(name, 0), len(shown))
            if len(shown) < len(set(asked)) and shown == asked[:len(shown)] and self.many_ok[name] == len(shown):
                # only the first ones answered: that is its size
                self.many_cap[name] = len(shown)
                rest = asked[len(shown):]
                return f"{name} answered only the first {len(shown)} of the {count} listings asked, so it takes at most {len(shown)} per call; {', '.join(rest)} were not opened."
            return ""
        text = _compact(observation) if isinstance(observation, (dict, list, str)) else ""
        if len(text) >= 300 or (isinstance(observation, dict) and any(str(k) in asked for k in observation)):
            # it answers in some other form (or per listing): not a page tool
            if name not in self.traits and not self.many_ok.get(name):
                self.multi_failed.add(name)
            return ""
        fails = self.size_fails[name] = self.size_fails.get(name, 0) + 1
        if self.many_ok.get(name):
            cap = min(self.many_ok[name], count - 1) if count > 1 else 1
        elif count <= 2 or fails >= 2:
            cap = 1
        else:
            cap = max(2, count // 2)      # half the size (race 9: 4 refused, 2 answered), then give up
        if cap <= 1 and not self.many_ok.get(name):
            if name not in self.traits:
                self.multi_failed.add(name)
            return f"{name} gave no listings for {count} at once; pages are opened one at a time instead."
        cap = max(1, cap)
        self.many_cap[name] = min(self.many_cap.get(name, cap), cap)
        return f"{name} gave no listings for {count} at once; it takes at most {self.many_cap[name]} per call, so send at most {self.many_cap[name]} listings in each call."

    def requeue_refused(self, name: str, args: dict[str, Any]) -> None:
        """A several-at-once call that showed none (or only some) of its listings: the rest are opened again by code, in
        calls of the size now known (or one at a time)."""
        argument = self.id_argument(name)
        if not argument or not isinstance(args.get(argument[0]), list):
            return
        page = self.page_tool()
        if not page:
            return
        for ident in [str(x) for x in args[argument[0]]]:
            if ident in self.paged:
                continue
            self.open_tried.discard(f"{page}|{ident}")
            if (page, ident) not in self.open_queue:
                self.open_queue.append((page, ident))

    def fill_ids(self, chosen: list[str], room: int) -> list[str]:
        """Listings from the latest results whose pages were never opened, to read along with the chosen ones: in result
        order (newest results first), of the chosen ones' category when known, not shown out of stock, within the budget."""
        if room <= 0:
            return []
        recent = self.records[-8:]
        rows = self.priced_rows(recent)
        kinds = {(rows.get(i) or {}).get("category") for i in chosen} - {None}
        several = bool(self.items())
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        out: list[str] = []
        for record in reversed(recent):
            for row in _listing_records(record):
                ident = _ident_of(row)
                if ident == "?" or ident in chosen or ident in out or ident in self.paged or ident in self.cart_ids:
                    continue
                known = rows.get(ident) or {}
                if known.get("in_stock") is False or (kinds and known.get("category") and known["category"] not in kinds):
                    continue
                if not several and budget is not None and isinstance(known.get("price"), (int, float)) and known["price"] * quantity > budget:
                    continue
                out.append(ident)
                if len(out) >= room and self.plan.get("priority") != "cheapest":
                    return out
        if self.plan.get("priority") == "cheapest" and len(out) > room:
            # read the rows most likely to be the cheapest fit: those showing the stated requirements, cheapest first
            def rank(ident: str) -> tuple[int, Any]:
                view = self.views.get(ident) or {}
                label = next((str(v).strip().lower() for k, v in view.items() if isinstance(v, str) and len(v) <= 70 and _KIND_KEYS.search(str(k))), "")
                return (-self.stated_score(view, self.stated_checks(label)), (rows.get(ident) or {}).get("price", 0))
            out = sorted(out, key=rank)[:room]
        return out[:room]

    def merge_pages(self, calls: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
        """Page calls are sent as calls of the tool that shows several listings' pages at once: single-listing page calls
        of one reply go together, and a call with room left is filled up with other unopened listings from the latest
        results, so one step reads several pages."""
        cut_calls = []
        held_replies: list[tuple[str, str]] = []
        untested: set[str] = set()
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") or {})
            argument = self.id_argument(name)
            size = self.many_cap.get(name) if name in self.many_cap else (1 if name in self.multi_failed else None)
            if (argument and argument[1] and size and isinstance(args, dict) and len(args) == 1 and isinstance(args.get(argument[0]), list)
                    and len(args[argument[0]]) > size):
                listed = [str(x) for x in args[argument[0]] if isinstance(x, (str, int))]
                keep, rest = listed[:size], listed[size:]
                call = {"id": call["id"], "type": "function", "function": {"name": name, "arguments": json.dumps({argument[0]: keep})}}
                page = self.page_tool() or name
                for ident in rest:
                    if (page, ident) not in self.open_queue and ident not in self.paged:
                        self.open_queue.append((page, ident))
                self.cut_notes[call["id"]] = f"{name} takes at most {size} listing(s) per call: {', '.join(keep)} sent; {', '.join(rest)} are opened next when steps allow."
            elif (argument and argument[1] and isinstance(args, dict) and isinstance(args.get(argument[0]), list) and len(args[argument[0]]) > 1
                    and not self.many_ok.get(name) and name not in self.many_cap and name not in self.multi_failed):
                if name in untested:
                    # the tool has not answered yet: its first call of this turn goes out, this one's listings follow
                    page = self.page_tool() or name
                    for ident in [str(x) for x in args[argument[0]] if isinstance(x, (str, int))]:
                        if (page, ident) not in self.open_queue and ident not in self.paged:
                            self.open_queue.append((page, ident))
                    held_replies.append((call["id"], f"Held back: {name} has not answered a call yet, so its first call of this turn goes out alone; these listings are opened right after it (results follow)."))
                    continue
                untested.add(name)
            cut_calls.append(call)
        calls = cut_calls
        multi = self.multi_page_tool()
        if not multi:
            return calls, held_replies
        key = self.id_argument(multi)[0]
        cap = self.many_limit(multi)
        pages = {n for n, t in self.traits.items() if t["text"]} | ({self.roles.get("details"), self.roles.get("view")} - {None} - set(self.traits))
        singles: list[tuple[dict[str, Any], str]] = []
        own: list[dict[str, Any]] = []
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments")
            args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") or {})
            if name == multi:
                if isinstance(args, dict) and isinstance(args.get(key), list) and len(args) == 1:
                    own.append(call)
                continue
            argument = self.id_argument(name)
            if name not in pages or not argument or argument[1]:
                continue
            ident = args.get(argument[0]) if isinstance(args, dict) and len(args) == 1 else None
            if isinstance(ident, str) and ident.strip():
                singles.append((call, ident.strip()))
        if not singles and not own:
            return calls, held_replies
        merged_ids = {id(call) for call, _ in singles}
        groups = [singles[i:i + cap] for i in range(0, len(singles), cap)]
        if len(groups) > 1 and not self.many_ok.get(multi):
            # the tool's real size is not known yet: one call now, the rest right after it at the size learned
            later = [pair for group in groups[1:] for pair in group]
            groups = groups[:1]
            for call, ident in later:
                if (multi, ident) not in self.open_queue:
                    self.open_queue.append((multi, ident))
            held = {id(call) for call, _ in later}
            singles = [pair for pair in singles if id(pair[0]) not in held]
            calls = [call for call in calls if id(call) not in held]
            extra_held = [(call["id"], f"Opened right after the first {multi} call (results follow), once its size is known.") for call, _ in later]
        else:
            extra_held = []
        heads = {id(group[0][0]): group for group in groups}
        out: list[dict[str, Any]] = []
        extra: list[tuple[str, str]] = []
        filled = False
        for call in calls:
            if call in own:
                raw = (call.get("function") or {}).get("arguments")
                args = raw if isinstance(raw, dict) else (_parse_json_object(raw or "{}") or {})
                listed = [str(x) for x in args.get(key) or [] if isinstance(x, (str, int))]
                if len(listed) > cap:
                    rest = listed[cap:]
                    listed = listed[:cap]
                    call = {"id": call["id"], "type": "function", "function": {"name": multi, "arguments": json.dumps({key: listed})}}
                    for ident in rest:
                        if (multi, ident) not in self.open_queue:
                            self.open_queue.append((multi, ident))
                    self.cut_notes[call["id"]] = f"{multi} takes at most {cap} listings per call: the first {cap} were sent; {', '.join(rest)} are opened next when steps allow."
                more = [] if filled else self.fill_ids(listed, cap - len(listed))
                if more:
                    filled = True
                    call = {"id": call["id"], "type": "function", "function": {"name": multi, "arguments": json.dumps({key: listed + more})}}
                    self.cut_notes[call["id"]] = f"Also opened, to save steps: {', '.join(more)} (other listings from the latest results whose pages you had not read)."
                out.append(call)
                continue
            if id(call) not in merged_ids:
                out.append(call)
                continue
            group = heads.get(id(call))
            if group is None:
                continue
            listed = [ident for _, ident in group]
            more = [] if filled else self.fill_ids(listed, cap - len(listed))
            filled = filled or bool(more)
            out.append({"id": call["id"], "type": "function", "function": {"name": multi, "arguments": json.dumps({key: listed + more})}})
            if more:
                self.cut_notes[call["id"]] = (f"Opened with {multi} (several pages in one step) together with {', '.join(more)}, other listings from the "
                                              "latest results whose pages you had not read.")
            for other, _ in group[1:]:
                extra.append((other["id"], f"Opened together with the other listings in one {multi} call (its result is given for call {call['id']}), to save steps."))
        self.trace.append({"kind": "merged_pages", "into": multi, "count": len(singles), "filled": filled})
        return out, extra + extra_held + held_replies

    @staticmethod
    def split_reply(observation: Any, wrapper_only: bool) -> tuple[list[str], list[str]]:
        """A result's own answer text, and the shopper lines that came with it (lists of sentences)."""
        replies: list[str] = []
        pending: list[str] = []
        if isinstance(observation, str):
            return [observation], []
        if not isinstance(observation, dict) or (wrapper_only and len(observation) > 3):
            return [], []
        for value in observation.values():
            inner = value.values() if isinstance(value, dict) else [value]
            for item in inner:
                if isinstance(item, str) and any(c.isalpha() for c in item) and not wrapper_only:
                    replies.append(item)
                elif isinstance(item, list) and item and all(isinstance(x, str) for x in item):
                    pending.extend(x for x in item if x.strip())
        return replies, pending

    def owed(self, stage: str = "order") -> list[dict[str, Any]]:
        """Actions due at this stage that take a message, while none went to the shopper for it: right before the order
        (after the last cart change), or before anything goes into the cart."""
        if not self.roles.get("message"):
            return []
        if stage == "cart":
            if self.message_steps:
                return []
            return [p for p in self.process if p.get("when") == "before_cart" and (p.get("messages") or 0) > 0]
        if any(s > self.cart_changed_at for s in self.message_steps):
            return []
        return [p for p in self.process if p.get("when") == "before_order" and (p.get("messages") or 0) > 0]

    def due_before_cart(self) -> list[dict[str, Any]]:
        if self.cart_stage_done or self.cart_used:
            return []
        return [p for p in self.process if p.get("when") == "before_cart" and ((p.get("messages") or 0) > 0 or (p.get("steps") or 0) > 0)]

    def page_cart(self, notes: list[str]) -> str:
        """The cart lines' full pages (stock, price, notes) are opened when not opened in full since the last cart change:
        one call of a several-at-once page tool, or single pages only for lines never opened in full."""
        tool = self.page_tool()
        argument = self.id_argument(tool)
        if not tool or argument is None or not self.cart_ids:
            return ""
        stale = [i for i in self.cart_ids if self.paged.get(i, -1) <= self.cart_changed_at]
        if not argument[1]:
            stale = [i for i in stale if i not in self.paged]
        stale = [i for i in stale if f"cart|{tool}|{i}|{self.cart_changed_at}" not in self.open_tried]
        table = self.table_guess()
        unlisted = ([i for i in self.cart_ids if i not in self.opened_with.get(table, set()) and f"cart|{table}|{i}" not in self.open_tried]
                    if table and table != tool and self.id_argument(table) else [])
        if not stale and not unlisted:
            return ""
        actions = self.open_actions([(tool, i) for i in stale])[:2] + self.open_actions([(table, i) for i in unlisted])[:5]
        if not actions or self.steps_left() < 2 + len(actions) + (1 if self.owed() else 0):
            return ""
        for ident in stale:
            self.open_tried.add(f"cart|{tool}|{ident}|{self.cart_changed_at}")
        for ident in unlisted:
            self.open_tried.add(f"cart|{table}|{ident}")
        return self.process_actions(actions, ["open every cart line in full after the last cart change, and its property list"], notes)

    def process_actions(self, actions: list[dict[str, Any]], missing: Any, notes: list[str]) -> str:
        """Run the pre-order actions; returns "over", "changed" (the shopper said something to apply), "looked",
        "messaged", or "" when the call failed."""
        reply = self.env_call(actions)
        self.trace.append({"kind": "process", "actions": actions, "reply": reply, "missing": missing})
        if reply is None:
            return ""
        self.recent = []
        observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
        changed = looked = False
        for index, action in enumerate(actions):
            observation = observations[index] if index < len(observations) else None
            self.records.append(observation)
            if isinstance(observation, (dict, list)):
                _collect_ids(observation, self.seen_ids)
            name, args = action["name"], action["args"]
            self.learn_listings(name, args, observation)
            text = " ".join(str(v) for v in args.values() if isinstance(v, str))
            failed = isinstance(observation, dict) and bool(observation.get("error"))
            if name == self.roles.get("message"):
                if failed:
                    notes.append(f"A message to the shopper failed: {_compact(observation)[:200]}.")
                    continue
                self.messages_used += 1
                self.message_steps.append(self.step)
                self.questions_sent.append(text.strip().lower())
                said, pending = self.split_reply(observation, False)
                if not said and not pending and _has_text(observation):
                    said = [_compact(observation)[:600]]
                answer = " ".join(said + pending)
                notes.append(f"Sent to the shopper: \"{text[:300]}\"; answer: \"{answer[:400]}\".")
                if pending or (said and _NOT_ACK.search(" ".join(said))):
                    meaning = self.interpret(said + pending, text)
                    if meaning:
                        notes.append(meaning)
                    changed = True
                continue
            looked = True
            if name == self.roles.get("cart_view"):
                self.last_cart_view = observation
                self.cart_view_at = self.step
            if name in (self.roles.get("details"), self.roles.get("view")) or name in self.traits:
                self.opened.update(v for v in args.values() if isinstance(v, str))
                self.opened.update(x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str))
            notes.append(f"Before the order, {name} {_compact(args)[:120]} was run: {_compact(_brief(observation, 200))[:600]}")
            _, pending = self.split_reply(observation, True)
            if pending:
                meaning = self.interpret(pending, None)
                notes.append("A shopper line came with that result: " + " | ".join(pending[:3]) + (" " + meaning if meaning else ""))
                changed = True
        if self.finished or self.dead:
            return "over"
        if changed:
            return "changed"
        return "looked" if looked else "messaged"

    def process_end(self, outcome: str, notes: list[str]) -> tuple[bool, str]:
        if outcome == "changed":
            self.process_at = self.step
            return False, " ".join(notes) + " Apply what the shopper said (a cart change means the actions due before the order are made again at the next order call); if nothing needs to change, call the order tool again now."
        return False, " ".join(notes)

    def write_owed(self, owed: list[dict[str, Any]], planned: str = "") -> str:
        prompt = (TELL_PROMPT + "\n\nREQUEST:\n" + str(self.policy.get("query") or "")
                  + (("\n\nCART STAGE: these lines are about to be put into the cart: " + planned) if planned else "")
                  + "\n\nDUE NOW:\n" + "\n".join("- " + str(p.get("do")) for p in owed)
                  + "\n\nRULES QUOTED FROM THE REQUEST:\n" + ("\n".join(str(r) for r in self.plan.get("rules") or []) or "(none)")
                  + "\n\nTHE CART (the final basket), WITH EACH ITEM'S PAGE:\n" + self.cart_text())
        result = self.judge(prompt, 800, [JUDGE_MODEL, READ_MODEL])
        self.trace.append({"kind": "process_tell", "result": result})
        return str((result or {}).get("message") or "").strip()

    def run_process(self, stage: str = "order", planned: str = "") -> tuple[bool, str]:
        """Before an order: the cart lines are opened in full, then the model reads the request, the conversation and the
        cart's pages, says which demanded actions are still missing for this basket and makes them. A demanded message
        that the check did not send is written and sent at the end. Returns whether the order can go ahead, and a note."""
        allowed = {self.roles.get(r) for r in ("message", "details", "view", "search", "cart_view")} | set(self.traits)
        allowed -= {self.roles.get(r) for r in ("cart", "order", "decline")}
        allowed.discard(None)
        if not self.process:
            allowed.discard(self.roles.get("message"))      # nothing demanded: pages may still be read, the shopper is not messaged
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or {})}"
                          for t in self.tools if t["function"]["name"] in allowed)
        notes: list[str] = []
        outcome = self.page_cart(notes) if stage == "order" else ""
        if outcome in ("over", "changed"):
            return self.process_end(outcome, notes)
        acted = bool(outcome)
        done: set[str] = set()
        if stage == "cart":
            self.cart_stage_done = True
        for _ in range(3):
            key = self.cart_changed_at if stage == "order" else -2
            if self.process_checks.get(key, 0) >= 3 or self.steps_left() < 2 or self.time_left() < WRAP_UP_SECONDS / 2:
                break
            self.process_checks[key] = self.process_checks.get(key, 0) + 1
            left = self.messages_left() if self.message_budget is not None else "no limit on"
            if not self.process:
                left = 0
            owed = self.owed(stage) if self.messages_left() > 0 else []
            prompt = (PROCESS_PROMPT + (("\n\nCART STAGE: the assistant is about to put these lines into the cart (that call has not run yet): " + planned
                                         + ". Judge only the actions demanded before anything goes into the cart; the ones due right before the order come later.") if stage == "cart" else "")
                      + "\n\nREQUEST:\n" + str(self.policy.get("query") or "")
                      + "\n\nACTIONS THE REQUEST DEMANDS (as read from it):\n" + (_compact(self.process) if self.process else "(none listed)")
                      + "\n\nRULES QUOTED FROM THE REQUEST:\n" + ("\n".join(str(r) for r in self.plan.get("rules") or []) or "(none)")
                      + "\n\nREQUIREMENTS NOW:\n" + self.ledger()
                      + "\n\nTOOLS YOU MAY USE NOW:\n" + tools
                      + "\n\nTHE CART (the final basket), WITH EACH ITEM'S PAGE:\n" + self.cart_text()
                      + f"\n\nEVERYTHING DONE SO FAR, oldest first (the last cart change was at step {self.cart_changed_at}; this is step {self.step}):\n" + self.action_log()
                      + (("\n\nNO MESSAGE HAS GONE TO THE SHOPPER SINCE THE LAST CART CHANGE, and these demanded actions take one: "
                          + "; ".join(str(p.get("do")) for p in owed) + ". Your actions must include that message now.") if owed else "")
                      + f"\n\nLEFT: {left} messages to the shopper, {self.steps_left() - 1} steps besides the order.")
            result = self.judge(prompt, 2500, [JUDGE_MODEL, READ_MODEL] if self.process else [READ_MODEL, JUDGE_MODEL])
            self.trace.append({"kind": "process_check", "result": result})
            if not isinstance(result, dict):
                continue        # an unreadable answer is asked again (the round cap still holds), never taken as "ready"
            planned = result.get("actions")
            if result.get("ready") is True or not isinstance(planned, list):
                break
            actions: list[dict[str, Any]] = []
            for raw in planned:
                tool = raw.get("tool") if isinstance(raw, dict) else None
                if not isinstance(tool, str) or tool not in allowed:
                    continue
                args, problems = _check(raw.get("args") if isinstance(raw.get("args"), dict) else {}, self.schemas[tool], "args")
                if problems:
                    continue
                if tool == self.roles.get("message") and self.messages_left() - sum(a["name"] == tool for a in actions) <= 0:
                    continue
                actions.append({"name": tool, "args": self.ids_for_titles(tool, args)})
            # a look-up already made for this basket is not made again: its result is in the record
            actions = [a for a in actions if a["name"] == self.roles.get("message") or a["name"] + _compact(a["args"]) not in done]
            actions = actions[: max(0, min(3, self.steps_left() - 1))]
            if not actions:
                break
            done.update(a["name"] + _compact(a["args"]) for a in actions)
            outcome = self.process_actions(actions, result.get("missing"), notes)
            if not outcome:
                break
            acted = True
            if outcome in ("over", "changed"):
                return self.process_end(outcome, notes)
            if outcome == "messaged":
                break
        # a demanded message is never dropped because the rounds went on look-ups or the check called the basket ready
        owed = self.owed(stage)
        if owed and self.process and self.messages_left() > 0 and self.steps_left() >= 2 and self.time_left() > 15:
            text = self.write_owed(owed, planned if stage == "cart" else "")
            args = self.message_args(text[:780]) if text else None
            if args:
                args, problems = _check(args, self.schemas[self.roles["message"]], "args")
                if not problems:
                    outcome = self.process_actions([{"name": self.roles["message"], "args": args}], [str(p.get("do")) for p in owed], notes)
                    acted = acted or bool(outcome)
                    if outcome in ("over", "changed"):
                        return self.process_end(outcome, notes)
        if acted and stage == "order":
            self.process_at = self.step
        return True, " ".join(notes)

    def infer(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int,
              effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
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
            if effort and model.startswith("anthropic/"):
                body["reasoning"] = {"effort": effort}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            result = _llm.post_verbose("/inference/chat/completions", body)
            for pause in (2.0, 5.0):
                if result.data is not None or not self.transient(result.error) or self.time_left() < FORCE_SECONDS + 20 + pause:
                    break
                # a busy or briefly over-committed key (402 "in flight", 429, 5xx) clears within seconds
                self.trace.append({"kind": "inference_retry", "model": model, "error": str(result.error)[:200]})
                time.sleep(pause)
                _llm.timeout = int(max(8, min(75, self.time_left() - FORCE_SECONDS + 15)))
                result = _llm.post_verbose("/inference/chat/completions", body)
            data = result.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.cost += float(usage["cost"])
                self.estimate_spend(model, usage, bool(tools))
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    return message
            self.trace.append({"kind": "inference_failure", "model": model, "error": str(result.error)[:300] if result.error else "empty"})
        return None

    def estimate_spend(self, model: str, usage: dict[str, Any], main: bool) -> None:
        """A spend estimate from the token counts (list prices per million tokens, cached input at a tenth), so the
        spending guards work even when a response carries no cost; the main conversation's size is kept too."""
        prompt = usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else 0
        completion = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        cached = details.get("cached_tokens") if isinstance(details.get("cached_tokens"), int) else None
        rate_in, rate_out = (2.0, 10.0) if "sonnet" in model or "opus" in model else (0.1, 0.5)
        if cached is None:
            fresh = prompt      # no cache reported: billed in full (ORO's race-15 bill matched the uncached figure)
        else:
            fresh = (prompt - cached) * (1.25 if main else 1.0) + cached * 0.1
        self.cost_est += (fresh * rate_in + completion * rate_out) / 1e6
        if main and prompt:
            self.context_tokens = prompt

    def spent(self) -> float:
        return max(self.cost, self.cost_est)

    @staticmethod
    def transient(error: Any) -> bool:
        """Whether a failed model call is worth repeating: a busy, rate-limited or over-committed key, or a server hiccup."""
        if not error:
            return False
        status = error.get("status") if isinstance(error, dict) else None
        if isinstance(status, int):
            return status in (402, 425, 429, 500, 502, 503, 520, 529)
        return bool(re.search(r"in.flight|rate.limit|overloaded", str(error).lower()))

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
                for key in ("observation", "observations"):
                    if key in data:
                        data[key] = _decode_rows(data[key])
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
        message = self.infer([{"role": "user", "content": content}], None, [READ_MODEL, BACKUP_MODEL], 4000, effort=LOW_EFFORT)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
        several = [i for i in plan.get("items") or [] if isinstance(i, dict) and str(i.get("product") or "").strip()]
        if (not isinstance(plan.get("roles"), dict) or len(several) < 2) and self.time_left() > 200:
            # a one-product request is read again by the stronger model: its rules, hints and requirements decide it
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL], 2500, effort=LOW_EFFORT)
            stronger = _parse_json_object((message or {}).get("content") or "")
            if isinstance(stronger, dict) and isinstance(stronger.get("roles"), dict):
                plan = stronger
                self.trace.append({"kind": "setup_reread"})
        if not isinstance(plan.get("roles"), dict) and self.time_left() > 200:
            # without the reading every later step works blind: try once more
            message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, MAIN_MODEL], 3000, effort=LOW_EFFORT)
            plan = _parse_json_object((message or {}).get("content") or "") or plan
        self.plan = plan
        roles = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        if not self.roles:
            # the reading failed: without roles every guard that depends on them is off, so guess from the tool names
            self.roles = self.guess_roles()
            self.trace.append({"kind": "guessed_roles", "roles": self.roles})
        self.fix_roles()
        budget = plan.get("message_budget")
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget >= 0:
            self.message_budget = int(budget)
        self.stated = [str(r) for r in plan.get("requirements") or [] if r]
        self.hints = [h for h in plan.get("hints") or [] if isinstance(h, dict) and str(h.get("clue") or "").strip()][:6]
        for raw in plan.get("pointers") or []:
            self.add_pointer(raw, "stated")
        if self.pointers:
            self.stated = [r for r in self.stated if not any(
                re.sub(r"[^a-z0-9]+", "", r.partition(":")[2].split("(")[0].lower()) == re.sub(r"[^a-z0-9]+", "", v.lower())
                and re.sub(r"[^a-z0-9]+", "", k.lower()) in re.sub(r"[^a-z0-9]+", "", r.partition(":")[0].lower())
                and "listing" not in r.lower() and "listed" not in r.lower()
                for p in self.pointers for k, v in p["attributes"].items())]
        self.budget_now = self.as_written(plan.get("price_cap"), str(self.policy.get("query") or ""))
        if plan and self.budget_now is None and self.roles.get("message"):
            ask = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = ["the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"] + ask
        steps = plan.get("step_budget")
        if isinstance(steps, (int, float)) and not isinstance(steps, bool) and 0 < steps < self.max_steps:
            self.max_steps = int(steps)
        self.links = [l for l in plan.get("links") or [] if isinstance(l, dict) and str(l.get("product") or "").strip() and str(l.get("on") or "").strip()][:4]
        self.process = [p for p in plan.get("process") or [] if isinstance(p, dict) and str(p.get("do") or "").strip()][:6]
        for item in self.process:
            for key in ("messages", "steps"):
                value = item.get(key)
                item[key] = int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 5 else 0

    @staticmethod
    def as_written(cap: Any, text: str) -> Any:
        """A cap converted to dollars when the text writes it in cents is put back in the text's own number."""
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0:
            return cap
        cents = {float(n.replace(",", "")) for n in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(?:¢|cents?\b|c\b)", text, re.I)}
        written = {float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
        scaled = round(cap * 100, 2)
        return scaled if scaled in cents and cap not in written else cap

    @staticmethod
    def reads_decline(name: str) -> bool:
        """A tool whose name says it ends the task without buying (stand down, cancel, abandon...)."""
        return bool(_DECLINE_WORDS.search(re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name or "")))

    def fix_roles(self) -> None:
        """The order role never goes to a tool whose name reads as ending without a purchase."""
        order = self.roles.get("order")
        if not order or not self.reads_decline(order) or self.roles.get("decline") == order:
            return
        properties = (self.schemas.get(order) or {}).get("properties") or {}
        for spec in properties.values():
            values = [str(v) for v in spec.get("enum") or []] if isinstance(spec, dict) else []
            if values and any(_DECLINE_WORDS.search(v) for v in values) and not all(_DECLINE_WORDS.search(v) for v in values):
                return      # one finishing tool that both buys and declines, by an argument
        guessed = self.guess_roles()
        other = guessed.get("order")
        self.trace.append({"kind": "fixed_roles", "order_was": order, "order_now": other})
        if other and other != order and not self.reads_decline(other):
            self.roles["order"] = other
        else:
            self.roles.pop("order", None)
        if not self.roles.get("decline") or self.roles.get("decline") == self.roles.get("order"):
            self.roles["decline"] = order

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
             or any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w or self.reads_decline(n)),
            ("cart_view", lambda w, n: bool(w & (cart_words | {"items", "order"})) and bool(w & {"view", "show", "list", "review", "status", "current", "my", "see", "contents", "summary"})),
            ("cart", lambda w, n: bool(w & cart_words) or bool(w & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(w & (cart_words | {"items", "order", "all"}))),
            ("order", lambda w, n: bool(w & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay", "execute", "finalize", "submit"})),
            ("message", lambda w, n: bool(w & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact", "write", "talk"}) and one_text_argument(n)),
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
        lines.append(f"- steps allowed: {self.max_steps}")
        for hint in self.hints:
            names = ", ".join(str(p) for p in hint.get("properties") or [] if p)
            lines.append(f"- story hint about the {hint.get('product') or plan.get('product')}: \"{hint.get('clue')}\" -> probably about: {names or 'an unstated property'} (asked about first)")
        for pointer in self.pointers:
            lines.append("- requirement given by reference to another listing: " + self.pointer_text(pointer))
        for link in self.links:
            rule = (f"if its {link.get('if')}, then {link.get('then')}" if str(link.get("kind")).lower().startswith("i") else str(link.get("then") or "it must fit"))
            lines.append(f"- linked rule: the {link['product']} depends on the {link['on']} you end up buying ({rule}). Choose the {link['on']} and put it in the cart first (when the order must have the smallest total, compare whole combinations: a dearer {link['on']} whose fitting {link['product']} is cheap can make the cheaper order);"
                         f" what it then demands of the {link['product']} is worked out for you from its page and shown under the requirements.")
        return "\n".join(lines)

    def status(self) -> str:
        left = self.steps_left()
        parts = [f"[status] steps used {self.step} of {self.max_steps}, {left} left"]
        if self.message_budget is not None:
            parts.append(f"messages sent {self.messages_used} of {self.message_budget}")
        else:
            parts.append(f"messages sent {self.messages_used}")
        parts.append(f"about {int(max(0, self.time_left() - FORCE_SECONDS))} s left")
        due = self.due_before_order()
        if due:
            parts.append("due before the order, made for you when you call the order tool: " + "; ".join(str(p.get("do"))[:120] for p in due)
                         + f" ({self.kept_messages()} message(s) and {self.kept_steps()} step(s) are kept for it)")
        elif self.process_at > self.cart_changed_at:
            parts.append("the actions due before the order were made for this basket; order with no cart change")
        text = "; ".join(parts) + "."
        text += "\n[requirements now] " + self.ledger()
        learned = self.search_rules() if left > 7 else []
        if learned:
            text += "\n[this store's search] " + "; ".join(learned) + "."
        number = self.cart_number(self.roles.get("cart") or "")
        if number and not self.cart_used:
            text += f"\n[cart tool] {number[0]} is the number of units of that listing on its line, not a line number" + ("; 0 takes a line out." if number[1].get("minimum") == 0 else ".")
        if self.cart_used:
            lines = [f"{i} x{self.cart_qty.get(i)}" if self.cart_qty.get(i) is not None else i for i in self.cart_ids]
            text += "\n[cart now] " + (", ".join(lines) if lines else "empty")
            without = self.products_without_line()
            if without:
                text += "\n[no cart line yet for] " + "; ".join(without)
        table, page = self.table_tool(), self.page_tool()
        if table and page and table != page and left > 7:
            text += (f"\n[listing tools] {table} shows only a listing's property list; {page} shows its full page (notes, description, price, stock)."
                     f" Check requirements on the property list; open in full every listing you put in the cart or reach through a reference.")
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            # a refused order is often misread as a placed one; until the episode ends nothing has been bought
            text += (f" NOTHING HAS BEEN BOUGHT YET: the order attempt {self.step - self.refused_at} steps ago did not end the episode."
                     " If the cart holds fitting items within the current budget, call the order tool now; if not, fix the cart first, then order"
                     " (decline only if nothing in the store fits any more).")
        if self.gone:
            text += "\n[sold out, never order] " + ", ".join(sorted(self.gone))
        open_pointers = [p for p in self.pointers if p["value"] is None]
        if open_pointers and left > 7:
            text += "\n[reference to resolve] " + " / ".join(self.pointer_text(p) for p in open_pointers)
        loose = [p for p in self.pointers if p["value"] is not None and not p["firm"]]
        if loose and left > 9:
            text += "\n[reference read off a partial match] " + " / ".join(
                f"{p['property']} was read off {p['listing']}, which does not show every value naming {p['seller']}'s listing"
                + (f" ({', '.join(f'{k} {v}' for k, v in p['attributes'].items())})" if p["attributes"] else "")
                + f": if steps allow, look for the {p['seller']} listing that shows them all (search the seller's name with the product kind and read further result pages) and read it there"
                for p in loose)
        if (self.spent() > COST_STOP or self.context_tokens > CONTEXT_STOP) and left > 2:
            text += " SPENDING LIMIT REACHED: this turn must end the episode. Order the cart if it holds a fit (fix a single line first if needed); decline only if nothing fits."
        if left <= 2:
            text += " LAST STEPS: order now if the cart holds a fit (a wrong decline scores the same 0 as a wrong order, and a fit is far more likely than none); decline only with an empty cart and nothing seen that could fit."
        elif left <= 5 or self.time_left() < WRAP_UP_SECONDS or self.spent() > COST_WRAP or self.context_tokens > CONTEXT_WRAP:
            text += (" FINISH NOW: fix the cart to the verified line(s) the request needs, re-check stock, then order; a product with no line yet gets its nearest"
                     " candidate. Decline only if nothing seen could fit.")
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self, linked: bool = True) -> str:
        if self.current is not None:
            items = list(self.current)
        else:
            items = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            items += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        items += [f"{r} (follows from the {self.links[i]['on']} in the cart)" for i, r in self.derived.items() if linked and i < len(self.links)]
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
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None, optional: bool = False) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.spent() > COST_SOFT and models[0] in (JUDGE_MODEL, PLAN_MODEL):
            # an expensive episode keeps its checks but makes them on the cheaper model, so one task cannot eat the key
            models = [READ_MODEL, JUDGE_MODEL]
        if optional and self.spent() > COST_HARD:
            return None
        if models[0] == READ_MODEL and len(models) > 1 and models[1] != READ_MODEL:
            # the cheap model first, then once more with room to finish its thinking, the stronger one last (and not
            # at all once the episode is expensive)
            for room in (max_tokens, max(3000, 2 * max_tokens)):
                message = self.infer([{"role": "user", "content": prompt}], None, [READ_MODEL], room, effort=LOW_EFFORT)
                parsed = _parse_json_object((message or {}).get("content") or "")
                if parsed is not None:
                    return parsed
                self.trace.append({"kind": "cheap_retry", "tokens": room, "empty": message is None})
                if self.time_left() < FORCE_SECONDS + 20:
                    return None
            if self.spent() > COST_SOFT:
                return None
            message = self.infer([{"role": "user", "content": prompt}], None, models[1:], max_tokens, effort=LOW_EFFORT)
            return _parse_json_object((message or {}).get("content") or "")
        message = self.infer([{"role": "user", "content": prompt}], None, models, max_tokens, effort=LOW_EFFORT)
        return _parse_json_object((message or {}).get("content") or "")

    def interpret(self, lines: list[str], question: str | None) -> str:
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if self.time_left() < WRAP_UP_SECONDS:
            return ""
        prompt = INTERPRET_PROMPT + "\n\nTHE SHOPPER'S REQUEST (for context):\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger(False)
        if self.said:
            prompt += "\n\nEARLIER EXCHANGES WITH THE SHOPPER (oldest first):\n" + "\n".join(self.said[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT ASKED:\n" + question
        prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        result = self.judge(prompt, 1500, [READ_MODEL, JUDGE_MODEL])
        self.last_reading = result if isinstance(result, dict) else {}
        if not result:
            return ""
        notes = []
        now = result.get("requirements_now")
        if isinstance(now, list) and all(isinstance(r, str) for r in now):
            self.current = [r for r in now if r.strip()]
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
        budget = self.as_written(result.get("new_budget"), " ".join(lines))
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
        if self.cart_ids:
            # a change announced while the cart holds a pick: the pick is checked again once this result is in
            repriced = result.get("repriced") is True or ("repriced" not in result and question is None and not result.get("unavailable")
                                                          and bool(_PRICE_WORDS.search(" ".join(fresh))))
            kinds = {kind for kind, hit in (("gone", bool(result.get("unavailable"))), ("price", repriced),
                                            ("budget", any(n.startswith("new budget") for n in notes)),
                                            ("dropped", any(n.startswith("dropped") for n in notes)),
                                            ("needs", any(n.startswith(("new requirement", "new quantity")) for n in notes))) if hit}
            if kinds:
                self.change_due |= kinds
                self.change_heard_at = self.step
                if "gone" in kinds and len(self.cart_ids) == 1:
                    # "your pick is sold out" with one line in the cart: that line
                    self.gone.add(self.cart_ids[0])
        pointer = self.add_pointer(result.get("pointer"), "revealed") if isinstance(result.get("pointer"), dict) else None
        if pointer:
            notes.append("this need is given by reference to another listing; see [reference to resolve] in the status and resolve it before choosing; the values that identify that listing are not requirements on the product")
        self.strip_pointer_attributes()
        if notes:
            self.ledger_version += 1
        reply = str(result.get("reply") or "").strip()
        # a shopper with several products on the list answers a property they do not care about with a confused or
        # "one question at a time" line: that is a non-answer, not a request to repeat. Only a single-product shopper
        # asking for the questions singly leaves the property open
        singly = result.get("singly") is True and question is not None and not self.items()
        unclear = question is not None and not singly and (result.get("unclear") is True or (result.get("singly") is True and bool(self.items())))
        if singly:
            self.reask.append(str(question).strip().lower())
            notes.append("the shopper wants the questions singly: that property is still open, ask about it again in one plain sentence")
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
        elif reply and self.roles.get("message") and _ASKS.search(" ".join(fresh)) and not (result.get("no_preference") and not notes) and result.get("singly") is not True:
            # only a line that asks something gets an answer; a stated need needs no confirmation message
            self.pending_reply = reply
        meaning = str(result.get("meaning") or "")
        return "Meaning: " + meaning + (" (" + "; ".join(notes) + ")" if notes else "")

    def item_records(self, item: str) -> list[str]:
        found: list[str] = []

        def walk(value: Any, depth: int, listing: Any = None, via_option: bool = False) -> None:
            if depth > 6 or len(found) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and v == item for v in value.values()):
                    # an option inside a listing is shown with the listing that holds it, whose facts it shares
                    holder = listing if listing is not None and via_option and listing is not value else value
                    text = _compact(_brief(holder, 260))[:3000]
                    if text not in found:
                        found.append(text)
                    return
                if any(isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)
                       for k, v in value.items()):
                    listing, via_option = value, False
                for k, v in value.items():
                    walk(v, depth + 1, listing, via_option or bool(_OPTION_KEYS.search(str(k))))
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1, listing, via_option)

        for record in self.records:
            walk(record, 0)
        view = self.views.get(item)
        if view and len(found) > 1:
            # facts split over several tools are shown together
            return found[-2:] + [_compact(_brief(view, 260))[:3000]]
        return found[-3:]

    def verify_cart(self) -> str:
        if not self.cart_ids:
            return ""
        # a request for several different products puts one line per product in the cart: check each line
        for item in (self.cart_ids[-3:] if self.items() else self.cart_ids[-1:]):
            verdict = self.verify_item(item, "ORDER", self.cart_lines.get(item, ""))
            if verdict:
                return verdict
        return ""

    def verify_item(self, item: str, purpose: str, line: str = "", linked: bool = True, strict: bool = False) -> str:
        if self.time_left() < (25 if strict else WRAP_UP_SECONDS + 20):
            return "unchecked" if strict else ""
        key = f"{item}|{line}|{self.ledger_version}|{linked}"
        if key in self.verdicts:
            return self.verdicts[key]
        records = self.item_records(item)
        if not records:
            return "unchecked" if strict else ""
        prompt = VERIFY_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nCURRENT REQUIREMENTS (stated, revealed and changed):\n" + self.ledger(linked)
        prompt += f"\n\nTHE ITEM TO {purpose}: {item}\nRECORDS OF THIS ITEM SEEN IN THE STORE (oldest first):\n" + "\n".join(records)
        if line:
            prompt += "\n\nTHE CART LINE FOR IT (when it names an option or variant, judge that option: its own values, price and stock count): " + line
        if self.last_cart_view is not None:
            prompt += "\n\nLATEST CART VIEW:\n" + _compact(self.last_cart_view)[:2000]
        labels = self.row_labels()
        if labels:
            prompt += "\n\nLABELS THAT RECUR ON THIS STORE'S LISTINGS (with counts; the item's category label must be the one that matches the request's wording as a whole, not a neighbouring one):\n" + labels
        result = self.judge(prompt, 1200, [READ_MODEL, JUDGE_MODEL], optional=True)
        self.trace.append({"kind": "verify", "item": item, "result": result})
        if strict and (not result or result.get("fits") is not True):
            return "unchecked" if not result else "does not fit"
        if not result or result.get("fits") is not False:
            self.verdicts[key] = ""
            return ""
        problems = [str(p) for p in result.get("problems") or []][:5]
        verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        self.verdicts[key] = verdict
        return verdict

    def add_pointer(self, raw: Any, origin: str) -> dict[str, Any] | None:
        """A requirement whose value must be read off another seller's listing."""
        if not isinstance(raw, dict):
            return None
        prop = str(raw.get("property") or "").strip()
        seller = str(raw.get("seller") or "").strip()
        if not prop or not seller or len(seller) > 60:
            return None
        if any(p["property"].lower() == prop.lower() and seller.lower() in p["sellers"] for p in self.pointers):
            return None
        attributes = raw.get("attributes") if isinstance(raw.get("attributes"), dict) else {}
        pointer = {"product": str(raw.get("product") or "").strip(), "property": prop,
                   "relation": self.relation_of(raw.get("relation")),
                   "seller": seller, "category": str(raw.get("category") or "").strip(),
                   "attributes": {str(k).strip(): str(v).strip() for k, v in attributes.items() if str(k).strip() and str(v).strip()},
                   "pick": str(raw.get("pick") or "").strip(), "rule": str(raw.get("rule") or "").strip()[:120], "origin": origin, "value": None, "listing": None,
                   "describe": "", "hops": [], "sellers": [seller.lower()], "score": None, "firm": False, "line": ""}
        self.pointers.append(pointer)
        self.trace.append({"kind": "pointer", "pointer": dict(pointer)})
        return pointer

    @staticmethod
    def relation_of(raw: Any) -> str:
        text = str(raw or "").strip().lower().replace(" ", "_")
        if text.startswith("other"):
            return "other"
        if re.search(r"pair|compat|counterpart|goes|partner|complement", text):
            return "paired"
        if re.search(r"least|lower|less|min|floor|exceed", text):
            return "at_least"
        if re.search(r"most|higher|more|max|ceil", text):
            return "at_most"
        return "same" if text.startswith(("s", "c", "m")) else "different"

    def strip_pointer_attributes(self) -> None:
        """A requirement line that only restates a value identifying a referenced listing is not a requirement."""
        if not self.pointers:
            return

        def squash(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text.lower())

        def identifying(line: str) -> bool:
            if ":" not in line or "listed" in line.lower() or "listing" in line.lower() or "read off" in line.lower():
                return False
            name, _, value = line.partition(":")
            name, value = squash(name), squash(value.split("(")[0])
            for pointer in self.pointers:
                for attr, wanted in pointer["attributes"].items():
                    if squash(attr) and squash(attr) in name and squash(wanted) == value:
                        return True
            return False

        self.revealed = [r for r in self.revealed if not identifying(r)]
        if self.current is not None:
            self.current = [r for r in self.current if not identifying(r)]

    def pointer_text(self, pointer: dict[str, Any]) -> str:
        head = (pointer["product"] + " " if pointer["product"] and self.items() else "") + pointer["property"]
        if pointer["value"] is not None and pointer["relation"] == "paired":
            partner = self.partner(pointer["property"], pointer["value"])
            if partner:
                return (f"{head}: {partner} (the store's rule pairs {pointer['value']}, the {pointer['property']} on {pointer['seller']}'s listing "
                        f"{pointer['listing']}, with {partner})")
            return (f"{head}: the value the store pairs with {pointer['value']} (the {pointer['property']} on {pointer['seller']}'s listing "
                    f"{pointer['listing']}); some listings' notes state the store's rule for {pointer['property']} (which values go together): "
                    f"read it there; until it is found, {pointer['value']} itself")
        if pointer["value"] is not None and pointer["relation"] == "other":
            return (f"{head}: {pointer.get('rule') or 'the relation the request states to'} that listing's value, where {pointer['seller']}'s listing "
                    f"{pointer['listing']} shows {pointer['property']} = {pointer['value']} (work out the value this demands)")
        if pointer["value"] is not None:
            want = {"same": pointer["value"], "at_least": "at least " + pointer["value"], "at_most": "at most " + pointer["value"]}.get(pointer["relation"], "anything except " + pointer["value"])
            return f"{head}: {want} (read off {pointer['seller']}'s listing {pointer['listing']})"
        attrs = ", ".join(f"{k} {v}" for k, v in pointer["attributes"].items())
        relation = {"same": "the same as", "paired": "the value the store's rule pairs with", "at_least": "at least", "at_most": "at most",
                    "other": (pointer.get("rule") or "related as the request says to") + " (the request's words for it):"}.get(pointer["relation"], "anything except")
        text = f"{head}: {relation} the {pointer['property']} on the {pointer['category'] or 'listing'} that {pointer['seller']} has listed"
        if attrs:
            text += f" with {attrs}"
        if pointer["describe"]:
            text += f" (described as: {pointer['describe']})"
        if pointer["pick"]:
            text += f" ({pointer['pick']})"
        if pointer["hops"]:
            text += f" (the notes on {pointer['hops'][-1]} send the reading there)"
        return text + (f" [UNRESOLVED: search for \"{pointer['seller']}\" with that product kind (read further result pages too), open that seller's listings that fit the description, "
                       f"read its {pointer['property']}; if its notes point to yet another listing, follow them]")

    def partner(self, prop: str, value: str) -> str | None:
        """The value the store's rule for this property pairs with the given one (read in either direction)."""
        def squash(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(text).lower())
        words = {w for w in re.split(r"[^a-z0-9]+", prop.lower()) if w}
        for name, rule in self.pair_rules.items():
            if not (words & {w for w in re.split(r"[^a-z0-9]+", name) if w}):
                continue
            for left, right in rule.items():
                if squash(left) == squash(value):
                    return right
            for left, right in rule.items():
                if squash(right) == squash(value):
                    return left
        return None

    def learn_pairs(self, prop: str, pairs: Any) -> None:
        """A rule stated on a listing of which values of a property go together; the paired references are rewritten."""
        if not isinstance(pairs, list):
            return
        rule = self.pair_rules.setdefault(prop.lower(), {})
        before = dict(rule)
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) and x.strip() for x in pair):
                rule.setdefault(pair[0].strip(), pair[1].strip())
        if rule != before:
            self.trace.append({"kind": "pair_rule", "property": prop, "rule": dict(rule)})
            for pointer in self.pointers:
                if pointer["relation"] == "paired" and pointer["value"] is not None:
                    self.ledger_version += 1
                    self.apply_pointer(pointer)

    def read_redirect(self, pointer: dict[str, Any], record: dict[str, Any], ident: str) -> dict[str, Any]:
        """Whether a matched listing sends the reading of the pointer's property on to another listing (its notes), and
        otherwise the value it states for it. One cheap reading per listing and property."""
        key = f"{ident}|{pointer['property'].lower()}|{len(_compact(record))}"
        if key in self.redirects:
            return self.redirects[key]
        if self.spent() > COST_HARD or self.time_left() < WRAP_UP_SECONDS + 20:
            return {}
        prompt = (REDIRECT_PROMPT.replace("PROPERTY_NAME", pointer["property"]) + "\n\nTHE LISTING (seller " + pointer["seller"] + "):\n"
                  + _compact(_brief(record, 700))[:3500])
        message = self.infer([{"role": "user", "content": prompt}], None, [READ_MODEL, BACKUP_MODEL], 450, effort=LOW_EFFORT)
        result = _parse_json_object((message or {}).get("content") or "") or {}
        self.redirects[key] = result
        self.learn_pairs(pointer["property"], result.get("pairs"))
        self.trace.append({"kind": "redirect_read", "listing": ident, "property": pointer["property"], "result": result})
        return result

    def resolve_pointers(self, observation: Any) -> str:
        """Read a pointer's value off the listing of the named seller that best carries the named values, following the
        listing's notes when they send the reading on to another listing."""
        if not isinstance(observation, (dict, list)) or not any(p["value"] is None or not p["firm"] for p in self.pointers):
            return ""

        def key_words(name: Any) -> set[str]:
            return {w for w in re.split(r"[^a-z0-9]+", str(name).lower()) if w}

        def same_property(name: Any, wanted: set[str]) -> bool:
            words = key_words(name)
            return bool(wanted) and bool(words) and (wanted <= words or words <= wanted)

        def squash(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", text.lower())

        def tables(record: Any) -> list[dict[str, str]]:
            out: list[dict[str, str]] = []

            def walk(value: Any, depth: int) -> None:
                if depth > 5:
                    return
                if isinstance(value, dict):
                    strings = {str(k): v for k, v in value.items() if isinstance(v, str) and v.strip()}
                    if strings:
                        out.append(strings)
                    for item in value.values():
                        walk(item, depth + 1)
                elif isinstance(value, list):
                    for item in value:
                        walk(item, depth + 1)
            walk(record, 0)
            return out

        def listings(value: Any, depth: int = 0):
            if depth > 5:
                return
            if isinstance(value, dict):
                if any(isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v) for k, v in value.items()):
                    yield value
                for item in value.values():
                    yield from listings(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    yield from listings(item, depth + 1)

        def detailed(record: dict[str, Any]) -> bool:
            # a full listing record (a property table or a note), not a bare search row
            return any((isinstance(v, dict) and sum(isinstance(x, str) for x in v.values()) >= 2)
                       or (isinstance(v, str) and v.strip() and re.search(r"note|remark|comment|know|info", str(k), re.I))
                       for k, v in record.items())

        def ident_of(record: dict[str, Any]) -> str:
            return next((v for k, v in record.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v) and any(c.isdigit() for c in v)), "?")

        def tokens(text: str, kind: set[str], seller: str) -> list[str]:
            out = []
            for raw in re.split(r"[\s,;/()]+", text.lower()):
                word = re.sub(r"('s|s')$", "", raw.strip(".:!?\"'"))
                if len(word) >= 2 and not _FILLER.match(word) and word not in kind and word != seller and word not in out:
                    out.append(word)
            return out

        def has(word: str, text: str) -> bool:
            return bool(re.search(r"\b" + re.escape(word).replace("\\-", "[\\s-]?") + r"(s|es)?\b", text))

        def score(pointer: dict[str, Any], record: dict[str, Any], flat: list[dict[str, str]]) -> tuple[int, int, int, int] | None:
            seller = pointer["seller"].lower()
            if not any(v.strip().lower() == seller for table in flat for v in table.values()):
                return None
            matched = missing = clashing = in_text = 0
            text = " ".join(v.lower() for table in flat for v in table.values())
            for name, wanted in pointer["attributes"].items():
                words, value = key_words(name), squash(wanted)
                keyed = [v for table in flat for k, v in table.items() if same_property(k, words)]
                if not keyed:
                    # no property for it (yet): the value written in the page's own text (a note often restates them)
                    wanted_words = tokens(wanted, set(), seller)
                    if wanted_words and all(has(w, text) for w in wanted_words):
                        in_text += 1
                    else:
                        missing += 1
                elif any(value and (value in squash(v) or squash(v) in value) for v in keyed):
                    matched += 1
                else:
                    # a value written another way (1000 gb against 1tb) looks like a clash: it lowers the rank, and only a
                    # page with more clashes than matches is ruled out
                    clashing += 1
            if pointer["attributes"] and (not (matched + in_text) or clashing > matched + in_text):
                return None
            # words that pick the listing out count first in its own property values, then anywhere in its text (a note
            # may name a value only to say the item lacks it)
            values = " ".join(str(x).lower() for v in record.values() if isinstance(v, dict) for x in v.values() if isinstance(x, str))
            kind = {w for w in key_words(pointer["category"]) if not _FILLER.match(w)}
            described = tokens(pointer["describe"], kind, seller)
            in_values = sum(1 for w in described if has(w, values))
            anywhere = sum(1 for w in described if has(w, text))
            if described and anywhere * 2 < len(described):
                return None
            tabled = any(isinstance(v, dict) and len(v) >= 3 for v in record.values())
            if described and tabled and in_values == 0 and not matched:
                return None
            same_kind = 1 if kind and all(re.search(r"\b" + re.escape(w[:5]), text) for w in kind) else 0
            full = missing == 0 and clashing == 0 and (in_values == len(described) or not tabled) and (in_text == 0 or not self.table_tool())
            return (1 if full else 0, matched + in_text + in_values + same_kind - clashing, anywhere, -missing - clashing)

        # a listing's facts may be split over several tools (seller in the search row, notes on its page, properties in
        # its table): each listing is read as the merged view of everything seen for it
        records = []
        for record in listings(observation):
            ident = ident_of(record)
            if all(ident_of(r) != ident for r in records):
                records.append(self.views.get(ident) or record)
        notes: list[str] = []
        for _ in range(4):
            progressed = False
            for pointer in self.pointers:
                if pointer["value"] is not None and pointer["firm"]:
                    continue
                best = None
                for record in records:
                    ident = ident_of(record)
                    if ident in pointer["hops"] or (pointer["value"] is not None and ident == pointer["listing"]) or not detailed(record):
                        continue
                    flat = tables(record)
                    rank = score(pointer, record, flat)
                    if rank is not None and (best is None or rank > best[0]):
                        best = (rank, record, flat, ident)
                if best is None:
                    seller = pointer["seller"].lower()
                    if pointer["value"] is None and any(detailed(r) and any(isinstance(v, str) and v.strip().lower() == seller for v in r.values()) for r in records):
                        # a page of that seller was read and it is not the listing meant: its other listings are opened
                        self.widen_seller(pointer, "")
                    continue
                rank, record, flat, ident = best
                if pointer["value"] is not None and pointer["score"] is not None and rank <= pointer["score"]:
                    continue
                if not rank[0] and self.widen_seller(pointer, ident):
                    # this page only partly fits what picks out the listing meant (a seller's other listings may carry notes
                    # that send the reader elsewhere): that seller's other listings are opened first
                    continue
                wants = self.missing_views(ident)
                page = self.page_tool()
                if page in wants:
                    # the reading waits for the listing's full page: its notes may send the reading on
                    if (page, ident) not in self.open_queue:
                        self.open_queue.append((page, ident))
                    continue
                reading = self.read_redirect(pointer, record, ident)
                onward = str(reading.get("seller") or "").strip()
                if (reading.get("redirect") is True and onward and len(onward) <= 60 and len(pointer["hops"]) < 10
                        and (onward.lower() != pointer["seller"].lower() or str(reading.get("description") or "").strip())):
                    previous = pointer["seller"]
                    pointer["hops"].append(ident)
                    pointer.update(seller=onward, category=str(reading.get("category") or "").strip(), describe=str(reading.get("description") or "").strip(),
                                   attributes={}, pick="", value=None, listing=None, score=None, firm=False)
                    if onward.lower() not in pointer["sellers"]:
                        pointer["sellers"].append(onward.lower())
                    self.ledger_version += 1
                    self.apply_pointer(pointer)
                    plain = f"{onward} {pointer['category']}".strip()
                    picked = [w for w in re.split(r"[^A-Za-z0-9.%-]+", pointer["describe"]) if len(w) >= 3 and not _FILLER.match(w.lower())][:4]
                    phrase = " ".join([onward] + picked + ([pointer["category"]] if pointer["category"] else [])).strip()[:100] if picked else plain
                    phrase, plain = self.fit_phrase(phrase), self.fit_phrase(plain)
                    if phrase != plain:
                        self.search_fallback[phrase.lower()] = plain
                    if self.roles.get("search") and phrase.lower() not in self.chased and self.page_tool():
                        # the next seller is searched for the model (with the words that pick out the listing meant); its
                        # listings are then opened in full
                        self.chased.add(phrase.lower())
                        if pointer["category"]:
                            self.search_kind[phrase.lower()] = pointer["category"]
                        self.open_queue.append(("?search", phrase))
                    notes.append(f"REFERENCE LEADS ON: {previous}'s listing {ident} says its {pointer['property']} is to be read from {onward}'s "
                                 f"{pointer['category'] or 'listing'}" + (f" ({pointer['describe']})" if pointer["describe"] else "")
                                 + f", not from {ident}. Search for \"{onward}\" with that product kind, open that listing in full "
                                 f"and read its {pointer['property']} there (its notes may send you on again).")
                    self.trace.append({"kind": "pointer_hop", "pointer": dict(pointer)})
                    progressed = True
                    continue
                wanted = key_words(pointer["property"])
                found = [(k, v.strip()) for table in flat for k, v in table.items() if same_property(k, wanted) and v.strip()]
                table = self.table_tool()
                if not found and table in wants:
                    # no redirect: the value is read off the listing's property list, opened next
                    if (table, ident) not in self.open_queue:
                        self.open_queue.append((table, ident))
                    continue
                if found:
                    key, value = sorted(found, key=lambda kv: len(key_words(kv[0])))[0]
                else:
                    key, value = pointer["property"], str(reading.get("value") or "").strip()
                if not value or value.lower() in ("null", "none"):
                    continue
                pointer["value"], pointer["listing"], pointer["score"], pointer["firm"] = value, ident, rank, bool(rank[0])
                self.ledger_version += 1
                self.apply_pointer(pointer)
                note = f"REFERENCE RESOLVED: {pointer['seller']}'s listing {ident} shows {key} = {value}, so the requirement is now \"{self.pointer_text(pointer)}\"."
                if not pointer["firm"]:
                    note += " That listing does not show every value that identifies the one meant: if another listing of that seller shows them all, the value is read there instead."
                if pointer["pick"]:
                    note += f" The request says which of that seller's listings is meant ({pointer['pick']}): if another of its listings fits that rule better, read the property there instead."
                notes.append(note)
                self.trace.append({"kind": "pointer_resolved", "pointer": dict(pointer)})
                progressed = True
            if not progressed:
                break
        return " ".join(notes)

    def apply_pointer(self, pointer: dict[str, Any]) -> None:
        """Replace the requirement line of a pointer (placeholder or earlier reading) in every requirement list."""
        text = self.pointer_text(pointer)
        prop = pointer["property"].lower()
        old = (pointer.get("line") or "")[:50].lower()

        def rewrite(items: list[str]) -> list[str]:
            out, done = [], False
            for item in items:
                low = item.lower()
                if not done and ((old and old in low) or (prop in low and (any(s in low for s in pointer["sellers"]) or "unresolved" in low))):
                    out.append(text)
                    done = True
                else:
                    out.append(item)
            if not done:
                out.append(text)
            return out

        if pointer["origin"] == "stated":
            self.stated = rewrite(self.stated)
        else:
            self.revealed = rewrite(self.revealed)
        if self.current is not None:
            self.current = rewrite(self.current)
        pointer["line"] = text

    def tools_for_turn(self) -> list[dict[str, Any]]:
        left = self.steps_left() - self.kept_steps()
        order, decline, cart = self.roles.get("order"), self.roles.get("decline"), self.roles.get("cart")
        names: set[str] | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3 or self.spent() > COST_STOP or self.context_tokens > CONTEXT_STOP:
            names = {order, decline, cart}
        elif (left <= 7 or self.spent() > COST_WRAP + 0.01) and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, self.roles.get("cart_view"), self.roles.get("view")}
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

    def line_kind(self, ident: str) -> str:
        view = self.views.get(ident) or {}
        return next((str(v).strip().lower() for k, v in view.items() if isinstance(v, str) and len(v) <= 70 and _KIND_KEYS.search(str(k))), "")

    def line_products(self) -> dict[str, list[str]]:
        """Each requested product with the cart lines of its kind: a line belongs to the product whose name is closest to
        the line's kind label (lines of unknown kind go under "")."""
        names = self.requested_names()
        out: dict[str, list[str]] = {n: [] for n in names}
        out[""] = []

        def closeness(name: str, label: str) -> float:
            a, b = self.word_stems(name), self.word_stems(label)
            return len(a & b) / max(1, len(a | b))
        for ident in self.cart_ids:
            label = self.line_kind(ident)
            best = max(names, key=lambda n: closeness(n, label)) if label else ""
            out[best if best and closeness(best, label) > 0 else ""].append(ident)
        return out

    def products_without_line(self) -> list[str]:
        if not self.items():
            return []
        grouped = self.line_products()
        if grouped[""] or len(self.cart_ids) >= len(self.items()):
            return []
        return [n for n, _ in self.items() if not grouped.get(n)]

    def cart_shape_problem(self) -> str:
        """More cart lines than products of a kind, or a requested product with no line: an order like that cannot pass."""
        grouped = self.line_products()
        surplus = [(n, lines) for n, lines in grouped.items() if n and len(lines) > 1]
        if surplus:
            told = "; ".join(f"{len(lines)} lines for the {n} ({', '.join(lines)}, the newest last)" for n, lines in surplus)
            return (f"the cart holds {told}, and the request wants one line per product (a new pick does not replace an earlier line by itself). "
                    "Take out each line you no longer want, then order.")
        without = self.products_without_line()
        if without:
            return (f"the cart has no line for: {'; '.join(without)}. Add the best verified fit (or the nearest candidate) for each with its quantity, then order.")
        return ""

    def finish_reserve(self) -> int:
        """Steps the end of an episode needs: the actions due before the order, a cart call per product without a line, the
        order and one spare."""
        return self.kept_steps() + max(0, len(self.requested_names()) - len(self.cart_ids)) + 2

    def requested_names(self) -> list[str]:
        return [n for n, _ in self.items()] or [str(self.plan.get("product") or "")]

    @staticmethod
    def word_stems(text: str) -> set[str]:
        return {re.sub(r"(es|s)$", "", w) for w in re.split(r"[^a-z0-9]+", str(text).lower()) if len(w) >= 3}

    def kind_named(self, label: str) -> int:
        """How many words a kind label shares with the best-matching requested product."""
        return max((len(self.word_stems(label) & self.word_stems(n)) for n in self.requested_names()), default=0)

    def stated_checks(self, label: str) -> list[list[set[str]]]:
        """The stated requirements on a kind of listing that can be checked by their words: one list of alternatives
        (each a set of word stems) per requirement. Limits, exclusions and references are not checked here."""
        names = [n for n, _ in self.items()]
        wanted = self.word_stems(label)
        checks = []
        for line in self.stated:
            if ":" not in line or line in self.dropped:
                continue
            head, _, value = line.partition(":")
            if names:
                owner = max(names, key=lambda n: len(self.word_stems(n) & self.word_stems(head)))
                if not (self.word_stems(owner) & self.word_stems(head)) or (wanted and not (self.word_stems(owner) & wanted)):
                    continue
            if re.search(r"\d|\b(most|least|under|over|than|below|above|within|between|except|anything|not|other|never|without|listing|same|paired)\b|[<>]", value, re.I):
                continue
            alternatives = [self.word_stems(part) for part in re.split(r",|/|\bor\b|\band\b", value)]
            alternatives = [a for a in alternatives if a]
            if alternatives:
                checks.append(alternatives)
        return checks

    def stated_score(self, view: dict[str, Any], checks: list[list[set[str]]]) -> int:
        """How many of those requirements a listing's title and property values show."""
        values: list[str] = []

        def walk(value: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, str) and (depth > 0 or _TITLE_KEYS.search(str(k))):
                        values.append(v)
                    elif isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)
        walk(view, 0)
        words = self.word_stems(" ".join(values))
        return sum(1 for alternatives in checks if any(a <= words for a in alternatives))

    def menu_listings(self) -> list[dict[str, Any]]:
        """The listings the question menu is drawn from: opened ones of the requested kinds, and of each kind those that show
        the stated requirements (when at least two do): a hidden need is a property of the listing meant, which meets them."""
        groups: dict[str, list[dict[str, Any]]] = {}
        for ident, view in self.views.items():
            if not any(isinstance(v, dict) and len(v) >= 3 for v in view.values()):
                continue
            label = next((str(v).strip().lower() for k, v in view.items() if isinstance(v, str) and len(v) <= 70 and _KIND_KEYS.search(str(k))), "")
            groups.setdefault(label, []).append(view)
        if not groups:
            return []
        if any(self.kind_named(g) for g in groups):
            groups = {g: v for g, v in groups.items() if self.kind_named(g)}
        chosen = []
        for label, views in groups.items():
            checks = self.stated_checks(label)
            fits = [v for v in views if self.stated_score(v, checks) == len(checks)] if checks else views
            chosen += fits if len(fits) >= 2 else views
        return chosen

    def property_menu(self) -> tuple[str, int]:
        counts: dict[str, dict[str, int]] = {}
        distinct: dict[str, dict[str, set[str]]] = {}
        samples: dict[str, dict[str, list[str]]] = {}
        tables: dict[str, int] = {}
        fact_sets = 0

        def short(value: Any) -> bool:
            return isinstance(value, str) and 0 < len(value) <= 70

        def walk(value: Any, depth: int, key: str = "", group: str = "") -> None:
            nonlocal fact_sets
            if depth > 6:
                return
            if isinstance(value, dict):
                # a listing's kind label (category, department, shelf...) groups its properties, so that a request for
                # several different products gets one menu per kind
                label = next((v.strip().lower() for k, v in value.items() if short(v) and _KIND_KEYS.search(str(k))), "")
                group = label or group
                strings = [(k, v) for k, v in value.items() if short(v)]
                # a property table: two short values under a key that names properties, or five anywhere when the
                # record holds no such table (then its own short values are the properties)
                nested = any(isinstance(v, dict) and _PROPERTY_KEYS.search(str(k)) for k, v in value.items())
                if (len(strings) >= 2 and (_PROPERTY_KEYS.search(key) or _OPTION_KEYS.search(key))) or (len(strings) >= 5 and not nested):
                    added = 0
                    for prop_key, item in strings:
                        if re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item) and ("-" in item or "_" in item):
                            continue
                        name = re.sub(r"[_.]+", " ", str(prop_key)).strip().lower()
                        if _LISTING_META.search(name) or _KIND_KEYS.search(name):
                            continue          # who sells it, what it is called, where it sits: not a property to ask about
                        counts.setdefault(group, {})[name] = counts.setdefault(group, {}).get(name, 0) + 1
                        distinct.setdefault(group, {}).setdefault(name, set()).add(item.lower())
                        bucket = samples.setdefault(group, {}).setdefault(name, [])
                        if item.lower() not in bucket and len(bucket) < 5:
                            bucket.append(item.lower())
                        added += 1
                    # a row of only ids, titles and seller names is not a set of facts, and the options of one listing
                    # are not several listings
                    if added >= 2 and not _OPTION_KEYS.search(key):
                        fact_sets += 1
                        tables[group] = tables.get(group, 0) + 1
                for name, item in value.items():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1, str(name), group)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1, key, group)

        chosen = self.menu_listings()
        if chosen:
            # each listing is read once, as the merged view of what was shown of it
            for record in chosen + [r for r in self.records if not any(True for _ in _listing_records(r))]:
                walk(record, 0)
        elif self.table_tool() and any(t["text"] and not t["table"] for t in self.traits.values()):
            # a store that shows a listing's page and its property list with different tools: each listing is read once,
            # as the merged view of both, so its property list carries the listing's kind label
            for record in list(self.views.values()) + [r for r in self.records if not any(True for _ in _listing_records(r))]:
                walk(record, 0)
        else:
            for record in self.records:
                walk(record, 0)

        def key_words(name: str) -> set[str]:
            return {w for w in re.split(r"[^a-z0-9]+", name.lower()) if w}

        def lines_for(group: str, limit: int) -> list[str]:
            c, d, s = dict(counts.get(group, {})), {k: set(v) for k, v in distinct.get(group, {}).items()}, {k: list(v) for k, v in samples.get(group, {}).items()}
            # a key whose words contain all the words of a shorter key is the same property under a longer name
            # (a "filter" twin, a prefixed copy): it is folded into the shorter name, which is the one to ask by
            also: dict[str, list[str]] = {}

            def same_property(shorter: str, longer: str) -> bool:
                # a prefixed copy ("fa pattern") or a key carrying the same values under a longer name; a longer key with
                # its own values ("warranty type" next to "type") is a different property
                extra = key_words(longer) - key_words(shorter)
                if all(len(w) <= 3 for w in extra):
                    return True
                values = d.get(longer, set())
                return bool(values) and len(values & d.get(shorter, set())) * 2 >= len(values)

            for longer in sorted(c, key=lambda k: (-len(key_words(k)), k)):
                lw = key_words(longer)
                shorter = next((k for k in sorted(c, key=lambda k: len(key_words(k))) if k != longer and k in c and key_words(k) and key_words(k) < lw
                                and same_property(k, longer)), None)
                if shorter is None:
                    continue
                c[shorter] = max(c[shorter], c[longer])
                d.setdefault(shorter, set()).update(d.get(longer, set()))
                for v in s.get(longer, []):
                    if v not in s.setdefault(shorter, []) and len(s[shorter]) < 5:
                        s[shorter].append(v)
                also.setdefault(shorter, []).append(longer)
                also[shorter] += also.pop(longer, [])
                del c[longer]
            # properties the request already states are settled: left out
            stated = {re.sub(r"[^a-z0-9]+", " ", r.split(":")[0].lower()).strip() for r in self.stated if ":" in r}
            for k in list(c):
                if any(k == s or (len(k) >= 4 and s.endswith(" " + k)) for s in stated):
                    del c[k]
            # the most widely carried properties first (a hidden need is a value of a property the kind of product
            # commonly has); among equals the ones whose values differ between listings
            ranked = sorted(c, key=lambda k: (len(d.get(k, ())) <= 1, -c[k], -len(d.get(k, ())), k))[:limit]
            return [f"- {k} (in {c[k]} listings, {len(d.get(k, ()))} different values" + (f"; also written as {', '.join(also[k][:2])}" if k in also else "") + "): " + " | ".join(s.get(k, [])) for k in ranked]

        groups = [g for g in sorted(counts, key=lambda g: -tables.get(g, 0)) if g and tables.get(g, 0) >= (1 if self.items() else 2)]
        if len(groups) >= 2 and self.items():
            out = []
            for group in groups[:4]:
                out.append(f"Listings labelled \"{group}\" ({tables[group]} with details):")
                out.extend(lines_for(group, 14))
            if counts.get(""):
                out.append("Listings without a label:")
                out.extend(lines_for("", 10))
            return "\n".join(out), fact_sets
        merged_counts: dict[str, int] = {}
        for group in counts:
            for name, n in counts[group].items():
                merged_counts[name] = merged_counts.get(name, 0) + n
                distinct.setdefault("", {}).setdefault(name, set()).update(distinct[group][name])
                bucket = samples.setdefault("", {}).setdefault(name, [])
                for v in samples[group][name]:
                    if v not in bucket and len(bucket) < 5:
                        bucket.append(v)
        counts[""] = merged_counts
        return "\n".join(lines_for("", 28)), fact_sets

    def row_labels(self) -> str:
        counts: dict[str, dict[str, int]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                if len(value) >= 4:
                    for key, item in value.items():
                        if isinstance(item, str) and 0 < len(item) <= 40 and not (re.search(r"\d", item) and re.fullmatch(r"[A-Za-z0-9_\-]+", item)):
                            bucket = counts.setdefault(str(key), {})
                            bucket[item] = bucket.get(item, 0) + 1
                for item in value.values():
                    if isinstance(item, (dict, list)):
                        walk(item, depth + 1)
            elif isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)

        for record in self.records:
            walk(record, 0)
        lines = []
        for key, bucket in counts.items():
            total = sum(bucket.values())
            if total >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                lines.append(f"- {key}: " + " | ".join(f"{v} (x{n})" for v, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(lines[:12])

    def kind_label(self, product: str) -> str:
        """The store's own category label for one requested product, lowercased: the label of opened listings that shares
        most words with the product's name (and fits it better than any other requested product), else the label most
        rows of the searches made for it carry; "" when unknown."""
        names = [n for n, _ in self.items()] or [product]
        labels: dict[str, int] = {}
        for ident in self.views:
            label = self.line_kind(ident)
            if label:
                labels[label] = labels.get(label, 0) + 1

        def stems(text: str) -> set[str]:
            out = set()
            for w in re.split(r"[^a-z0-9]+", str(text).lower()):
                if len(w) < 3 or w in ("and", "the", "for", "with"):
                    continue
                out.add(w[:-2] if w.endswith(("ches", "shes", "sses", "xes")) else w[:-1] if w.endswith("s") and not w.endswith("ss") else w)
            return out

        def closeness(name: str, label: str) -> float:
            a, b = stems(name), stems(label)
            return len(a & b) / max(1, len(a | b))
        mine = [l for l in labels if closeness(product, l) > 0 and closeness(product, l) >= max(closeness(n, l) for n in names)]
        if mine:
            return max(mine, key=lambda l: (closeness(product, l), labels[l]))
        wanted = stems(product)
        votes: dict[str, int] = {}
        for wording, entry in self.searches.items():
            if not wanted & stems(wording) or any(closeness(n, wording) > closeness(product, wording) for n in names if n != product):
                continue
            for ident in entry.get("seen") or []:
                label = self.line_kind(ident)
                if label and not any(closeness(n, label) > 0 for n in names if n != product):
                    votes[label] = votes.get(label, 0) + 1
        return max(votes, key=votes.get) if votes else ""

    def named_question(self, item: dict[str, Any], text: str) -> str:
        """A planned question on a several-product request, naming its product by the store's own label for it."""
        prop = re.sub(r"[_.]+", " ", str(item.get("property") or "")).strip()
        product = str(item.get("product") or "").strip()
        if not self.items() or not prop or not product or len(prop) > 40:
            return text
        label = self.kind_label(product)
        if not label or label in text.lower():
            return text
        return f"Which {prop} would you like the {label} to have?"

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

    def list_noun(self, product: str) -> str:
        """How a list question names its product: the store's own label for it on a several-product request, else the
        request's plain noun."""
        if self.items():
            return self.kind_label(product) or product
        return str(self.plan.get("product") or product or "item").strip()

    def list_text(self, product: str, labels: list[str]) -> str:
        noun = self.list_noun(product)
        if len(labels) == 1:
            return f"What {labels[0]} would you like for the {noun}?"
        return f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {noun}?"

    def fill_lists(self, planned: list[dict[str, Any]]) -> None:
        """The planned properties as one ranked queue per product (the planner's order is its ranking)."""
        self.list_queue, self.list_order = {}, []
        default = str(self.plan.get("product") or "")
        for rank, item in enumerate(planned):
            prop = re.sub(r"[_.]+", " ", str(item.get("property") or "")).strip()
            if not prop or len(prop) > 40 or "," in prop:
                continue
            product = str(item.get("product") or "").strip() if self.items() else default
            if self.items():
                names = [n for n, _ in self.items()]
                product = next((n for n in names if n.lower() == product.lower()), None) or next(
                    (n for n in names if product and (product.lower() in n.lower() or n.lower() in product.lower())), product or names[0])
            queue = self.list_queue.setdefault(product, [])
            if product not in self.list_order:
                self.list_order.append(product)
            if all(_words_of(prop) != _words_of(p) for p, _ in queue) and len(queue) < LIST_SIZE + 3:
                queue.append([prop, 0.85 ** len(queue)])

    def ask_listed(self) -> bool:
        """One list message from the queues: the next product's most likely properties; the answer re-ranks them."""
        tool = self.roles.get("message")
        limit = self.question_limit()
        if not tool or self.messages_used >= limit or self.list_queue is None:
            self.planner_on = self.planner_on and self.messages_used < limit
            return False
        if self.steps_left() <= 8 or self.time_left() < WRAP_UP_SECONDS + 30:
            self.planner_on = False
            return False
        live = [p for p in self.list_order if any(w > 0.02 for _, w in self.list_queue.get(p) or [])]
        if not live:
            # every planned property has been asked: a later plan may draw new ones from more listings
            self.list_queue = None
            self.round_pending = False
            return False
        product = min(live, key=lambda p: (self.list_hits.get(p, 0), self.list_sent.get(p, 0), self.list_order.index(p)))
        ranked = sorted((x for x in self.list_queue[product] if x[1] > 0.02), key=lambda x: -x[1])
        labels = [p for p, _ in ranked[:LIST_SIZE]]
        if self.list_text(product, labels).strip().lower() in self.questions_sent:
            # the same list again would be refused unsent: its first property makes way
            ranked[0][1] = 0.0
            return self.ask_listed()
        args = None
        while labels:
            text = self.list_text(product, labels)
            args = self.message_args(text)
            if args:
                args, problems = _check(args, self.schemas[tool], "args")
                if not problems and text in args.values():
                    break
            labels = labels[:-1]
            args = None
        if not args:
            self.list_queue = None
            return False
        call = {"id": f"ask_{self.messages_used + 1}_{self.step}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
        self.messages.append({"role": "assistant", "content": f"Next question, on the {self.list_noun(product)}: {', '.join(labels)}.", "tool_calls": [call]})
        revealed, free, reask = len(self.revealed), len(self.free), len(self.reask)
        self.last_reading = {}
        self.last_unclear = False
        used = self.messages_used
        replies = self.run_calls([call], planned=True)
        if self.messages_used == used:
            # not sent (budget, a refusal by the checks): no more lists in this episode
            for index, (call_id, text_back) in enumerate(replies):
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text_back + ("\n" + self.status() if index == len(replies) - 1 else "")})
            self.list_queue = None
            self.planner_on = False
            return True
        for index, (call_id, text_back) in enumerate(replies):
            if index == len(replies) - 1:
                text_back += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text_back})
        answer = replies[0][1].split(chr(10) + "[status]")[0][:400] if replies else ""
        reading = self.last_reading or {}
        new = self.revealed[revealed:]
        if len(labels) > 1:
            # a list's "don't mind" or "one at a time" is about one of its properties, not all of them
            del self.free[free:]
            self.reask = self.reask[:reask]
            self.last_unclear = False
        # the place of each told need in the list
        hits = sorted({i for need in new for i, label in enumerate(labels)
                       if _words_of(label) and _words_of(label) <= _words_of(need.split(":", 1)[0])})
        kind = ("told" if new else "singly" if reading.get("singly") is True else "unclear" if reading.get("unclear") is True
                else "free" if reading.get("no_preference") else "other")
        weights = {p: x for p, x in ((x[0], x) for x in self.list_queue[product])}
        for place, label in enumerate(labels):
            entry = weights.get(label)
            if entry is None:
                continue
            reach = LIST_REACH[min(place, len(LIST_REACH) - 1)]
            if kind == "told":
                if place in hits or (not hits and place == 0):
                    entry[1] = 0.0
                elif hits and place < hits[0]:
                    entry[1] *= 0.3
            elif kind == "singly":
                if place == 0:
                    entry[1] = 0.0
            elif kind == "free":
                entry[1] = 0.0 if place == 0 else entry[1] * 0.15
            elif len(labels) == 1:
                entry[1] = 0.0
            else:
                entry[1] = 0.0 if place == 0 and kind == "other" else entry[1] * (1 - 0.5 * reach)
        if new:
            self.list_hits[product] = self.list_hits.get(product, 0) + len(new)
        self.list_sent[product] = self.list_sent.get(product, 0) + 1
        self.asked.append(f"- {product} ({', '.join(labels)}): asked \"{self.list_text(product, labels)}\" -> {answer}")
        self.trace.append({"kind": "list_ask", "product": product, "labels": labels, "reply": kind, "told": new})
        self.sets_at_last_ask = self.property_menu()[1]
        self.planner_on = self.messages_used < limit
        self.round_pending = self.planner_on
        self.rounds += 1
        return True

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.question_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        if LIST_SIZE > 1 and self.list_queue is not None:
            return self.ask_listed()
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 5 and self.step < (5 if self.multi_page_tool() else 8) and not urgent and not force:
            return False
        if ((not self.asked or self.plan.get("ask_first")) and not urgent and not force and self.steps_left() > 10
                and self.time_left() > WRAP_UP_SECONDS + 40 and self.fill_menu()):
            return True         # the pages just read go into the menu of the next planning call
        # several products: the whole round is planned at once, so it waits until listings of at least two kinds
        # have been opened (the menu is then grouped by kind) unless the episode is already well under way
        if self.items() and not urgent and not force and not self.asked and menu.count("Listings labelled") < min(len(self.items()), 2) and self.step < 12:
            return False
        if force:
            urgent = False
        # a later planning round (questions left over, or one at a time for several products) waits for more listings
        # (or a commit attempt), so it is chosen knowing the earlier answers and more of the store
        if self.asked and not force and not urgent and not self.last_unclear and not self.round_pending and fact_sets < self.sets_at_last_ask + 2:
            return False
        if not force and fact_sets < self.plan_retry_at:
            return False
        if self.asked and not urgent and self.steps_left() < 14 + self.kept_steps() + 3 * sum(1 for p in self.pointers if p["value"] is None):
            # a later round of questions never takes the steps that open references and the actions due before the order need
            return False
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        if (not self.asked or self.plan.get("ask_first")) and not urgent and self.fill_menu():
            return True         # the questions are planned next, from the property lists just read
        # every remaining question is planned at once (for several products spread over them), so the questions do
        # not eat into the steps needed for the search and the stronger planner sees the whole menu once
        count = max(1, limit - self.messages_used)
        send = count
        if not self.asked and not urgent and count >= 3:
            # the whole list is planned by the stronger model, but only the first round is sent: the most promising
            # property of each product (three for a single product); the rest follow right after, re-aimed with the
            # answers in view (a product whose need was revealed rarely has another)
            send = min(count, max(2, len(self.items())) if self.items() else 3)
        listing = LIST_SIZE > 1 and not urgent
        wanted = min(20, (LIST_SIZE + 3) * max(1, len(self.items()))) if listing else count
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(wanted)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        if self.asked:
            prompt += LATER_ROUND_NOTE
            if self.plan_rest:
                prompt += ("\n\nTHE FIRST PLAN'S REMAINING QUESTIONS (in its order of promise, made from the whole menu by a careful planner): keep them "
                           "in this order unless an answer above makes one pointless (its product's need is already known, or it repeats a property "
                           "the shopper did not care about); replace only those:\n" + "\n".join(f"- {q}" for q in self.plan_rest))
        if SWEEP_LAST and limit >= 2 and self.messages_used + count == limit and not urgent:
            # the last message names a few more properties at once: the shopper answers about one of them when one is a
            # need, and no later question is lost if the answer is "don't mind"
            prompt += SWEEP_NOTE
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger(False)
        items = self.items()
        if items:
            prompt += "\n\nTHE REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS (name the one each question is about): " + "; ".join(name for name, _ in items)
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        if self.hints:
            prompt += "\n\nSTORY HINTS (each points at a property to ask before the menu order; map it to the menu's own name for the property the sentence is about):\n" + "\n".join(
                f"- about the {h.get('product') or self.plan.get('product')}: \"{h.get('clue')}\" -> probably: " + ", ".join(str(p) for p in h.get("properties") or [] if p)
                for h in self.hints)
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings carrying it, number of different values, example values; properties with varied values first, then the most widely carried):\n" + (menu or "(none seen yet)")
        # every round is planned by the cheaper model (a twentieth of the price; one stronger-model call cost a fifth of an
        # episode's spend), the stronger one only when its answer is unreadable
        models = [READ_MODEL, PLAN_MODEL]
        result = self.judge(prompt, 2000, models)
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        for item in planned:
            item["question"] = self.named_question(item, str(item.get("question")).strip())
        if listing and planned:
            self.fill_lists(planned)
            if self.list_queue:
                return self.ask_listed()
            self.list_queue = None
        if urgent and fact_sets < 4:
            planned = planned[:1]
        elif urgent and LIST_SIZE > 1:
            # with lists, the must-ask questions go alone first; the lists follow at once
            planned = planned[:max(1, len(self.plan.get("ask_first") or []))]
        # never send a question that has been sent before in the same words (unless the shopper asked for it again, singly)
        planned = [q for q in planned if str(q.get("question")).strip().lower() not in self.questions_sent
                   or str(q.get("question")).strip().lower() in self.reask]
        planned = planned[:count]
        if send < len(planned):
            self.plan_rest = [str(q.get("question")).strip() for q in planned[send:]]
            planned = planned[:send]
        elif self.asked:
            self.plan_rest = []
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
                answer = text.split(chr(10) + '[status]')[0][:400]
                if labels[index][1].lower() in self.reask:
                    self.questions_sent = [q for q in self.questions_sent if q != labels[index][1].lower()]
                    answer += " [no answer: the shopper wants the questions singly; this property is still open]"
                self.asked.append(f"- {labels[index][0]}: asked \"{labels[index][1]}\" -> {answer}")
        self.sets_at_last_ask = fact_sets
        self.planner_on = self.messages_used < limit
        # the rest of the questions follow at once, chosen with these answers in view
        self.round_pending = self.planner_on and self.rounds == 0
        self.rounds += 1
        return True

    def page_argument(self) -> tuple[str, str] | None:
        """The search tool's text argument and its page-token argument, when it has both."""
        search = self.roles.get("search")
        text = self.text_argument(search) if search else None
        properties = (self.schemas.get(search or "") or {}).get("properties") or {}
        page = next((k for k, v in properties.items() if k != text and isinstance(v, dict) and v.get("type") in ("string", "integer")
                     and re.search(r"next|page|cursor|token|after|onward|offset|continu|marker|begin|start|skip", k, re.I)), None)
        if text is None:
            text = self.search_text_key()
        if text is None:
            # a search that also takes a page token has two text arguments: the query is the required one
            required = [k for k in (self.schemas.get(search or "") or {}).get("required") or [] if k in properties and k != page]
            text = required[0] if len(required) == 1 else None
        return (text, page) if text and page else None

    def note_search(self, args: dict[str, Any], observation: Any) -> None:
        """Remember, per search wording, the total it reports, the rows seen over its pages and the next page's token."""
        keys = self.page_argument()
        if not keys or not isinstance(observation, dict):
            return
        text, page = keys
        wording = str(args.get(text) or "").strip().lower()
        if not wording:
            return
        total = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                      and re.search(r"found|total|count|hits|number|matches", str(k), re.I)), None)
        token = next((v for k, v in observation.items() if ((isinstance(v, str) and v.strip()) or (isinstance(v, int) and not isinstance(v, bool)))
                      and re.search(r"next|onward|cursor|token|after|continu|page|marker", str(k), re.I)), None)
        rows = [_ident_of(r) for r in _listing_records(observation)]
        entry = self.searches.setdefault(wording, {"total": total, "seen": [], "next": None, "size": len(rows), "args": dict(args)})
        entry["total"] = total if total is not None else entry["total"]
        entry["seen"] += [r for r in rows if r != "?" and r not in entry["seen"]]
        if token is None and rows:
            # a store paged by a plain number gives no token: the next page is worked out from the argument's schema
            spec = ((self.schemas.get(self.roles.get("search") or "") or {}).get("properties") or {}).get(page) or {}
            if spec.get("type") == "integer":
                first = spec.get("minimum") if isinstance(spec.get("minimum"), int) else (0 if re.search(r"offset|begin|start|skip|from", page, re.I) else 1)
                current = args.get(page) if isinstance(args.get(page), int) and not isinstance(args.get(page), bool) else first
                step = len(rows) if re.search(r"offset|begin|start|skip|from", page, re.I) else 1
                most = spec.get("maximum")
                if (isinstance(entry["total"], int) and len(entry["seen"]) < entry["total"]
                        and (not isinstance(most, int) or current + step <= most)):
                    token = current + step
        entry["next"] = token
        entry["size"] = max(entry["size"], len(rows))

    def reach_note(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """A search reports more matches than its page argument can ever reach (its schema caps the offset): the model is
        told how many can be shown at most, and that other searches reach the rest."""
        if len(self.reach_noted) >= 2 or not isinstance(observation, dict):
            return
        rows = len(list(_listing_records(observation)))
        total = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                      and re.search(r"found|total|count|hits|number|matches", str(k), re.I)), None)
        if not rows or not isinstance(total, int):
            return
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        page = next((k for k, v in properties.items() if isinstance(v, dict) and v.get("type") == "integer" and isinstance(v.get("maximum"), int)
                     and re.search(r"page|begin|start|offset|skip|row|from|at", k, re.I)), None)
        if not page:
            return
        most = properties[page]["maximum"]
        # an offset counts rows; a page number counts pages
        least = properties[page].get("minimum") if isinstance(properties[page].get("minimum"), int) else 0
        reach = most + rows if most >= rows else (most - least + 1) * rows
        signature = _compact({k: v for k, v in args.items() if k != page})
        if total <= reach or signature in self.reach_noted:
            return
        self.reach_noted.add(signature)
        cheapest = self.plan.get("priority") == "cheapest"
        self.side_notes.append(f"This search reports {total} matches, but its '{page}' argument stops at {most}, so at most the first {reach} of them can ever be shown, "
                               "in the store's own order (not by price). The others are reached only by other searches: keyword searches with the product's plain "
                               "names, other names for it and words from the titles of fitting listings" + (
                                   "; for the cheapest fit this matters, since it may be among the ones these pages never show." if cheapest else "."))
        self.trace.append({"kind": "reach_note", "tool": name, "total": total, "reach": reach})
        # (43g) the reach note itself names the way round; no plan for it

    def search_parts(self, name: str, args: dict[str, Any]) -> tuple[str | None, str | None, str]:
        """(text argument, page argument, signature of the other arguments) of a search call."""
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        text = self.search_text_key()
        page = next((k for k, v in properties.items() if k != text and isinstance(v, dict) and v.get("type") in ("integer", "string") and "enum" not in v
                     and re.search(r"next|page|cursor|token|after|onward|offset|continu|marker|begin|start|skip|from", k, re.I)), None)
        others = _compact({k: v for k, v in sorted(args.items()) if k not in (text, page)})
        return text, page, others

    @staticmethod
    def query_words(text: str) -> tuple[str, ...]:
        return tuple(w for w in re.split(r"[\s(),./:;|]+", str(text or "").lower()) if w)

    def learn_search(self, name: str, args: dict[str, Any], observation: Any, told: bool = True) -> None:
        """Learn how this store's search reads a query from the searches made: how many leading words it reads, whether
        it needs every word, whether its keyword mode keeps to the category argument. A newly learned rule is told once."""
        if name != self.roles.get("search") or not isinstance(observation, dict):
            return
        self.search_sigs.add(_compact(args))
        rows = list(_listing_records(observation))
        _total = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                       and re.search(r"found|total|count|hits|number|matches", str(k), re.I)), None)
        self.trace.append({"kind": "search_seen", "line": f"- {_compact(args)} -> {_total if _total is not None else '?'} matches, {len(rows)} rows shown"})
        self.empty_streak = 0 if rows else self.empty_streak + 1
        if self.empty_streak >= 2 and self.search_plans < 2:
            self.search_plan_due = self.search_plan_due or "two searches in a row found nothing"
        text_key, page_key, others = self.search_parts(name, args)
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        # the keyword mode and the category argument: rows of other categories show the argument is not applied
        category = next((k for k, v in properties.items() if isinstance(v, dict) and k != text_key
                         and re.search(r"categor|group|shelf|section|department|aisle|range", k, re.I)), None)
        if text_key and str(args.get(text_key) or "").strip() and category and isinstance(args.get(category), str) and len(rows) >= 4:
            wanted = args[category].strip().lower()
            labels = [next((str(v).strip().lower() for k, v in r.items() if isinstance(v, str) and re.search(r"categor|shelf|section|department|aisle|group", k, re.I)), "") for r in rows]
            labels = [l for l in labels if l]
            if len(labels) >= 4:
                off = sum(1 for l in labels if l != wanted and wanted not in l and l not in wanted)
                if off >= len(labels) / 2:
                    self.search_rule["category"] = False
                elif off == 0 and self.search_rule["category"] is None:
                    self.search_rule["category"] = True
        first_page = page_key is None or args.get(page_key) in (None, "", 0, 1, properties.get(page_key, {}).get("minimum"))
        words = self.query_words(args.get(text_key)) if text_key else ()
        if not words or not first_page:
            self.tell_rules(told)
            return
        total = next((v for k, v in observation.items() if isinstance(v, int) and not isinstance(v, bool)
                      and re.search(r"found|total|count|hits|number|matches", str(k), re.I)), len(rows))
        ids = tuple(_ident_of(r) for r in rows)
        rule = self.search_rule
        for seen in self.search_log:
            if seen["others"] != others or seen["words"] == words:
                continue
            k = 0
            while k < min(len(words), len(seen["words"])) and words[k] == seen["words"][k]:
                k += 1
            extends = k == min(len(words), len(seen["words"]))
            if total != seen["total"] or ids != seen["ids"]:
                rule["low"] = max(rule["low"], k)          # a word after the k-th was read
                if extends and k >= 1 and total != seen["total"] and min(rule["high"] or {99}) > k:
                    longer = total if len(words) > len(seen["words"]) else seen["total"]
                    shorter = seen["total"] if len(words) > len(seen["words"]) else total
                    rule["words"] = "all" if longer < shorter else "any"
            elif extends and k >= 1 and len(set(ids)) >= 3:
                rule["high"].add(k)                          # the words after the k-th were not read
        self.search_log.append({"others": others, "words": words, "total": total, "ids": ids})
        self.tell_rules(told)

    def probe_words(self, name: str, args: dict[str, Any], observation: Any) -> None:
        """Once: a keyword search of three or more words is run again with its first two, to learn whether the later
        words are read (one step, only with steps to spare and the rule still unknown)."""
        if self.word_probed or self.lead_words() or self.search_rule["low"] >= 2 or not isinstance(observation, dict):
            return
        text_key, page_key, _ = self.search_parts(name, args)
        words = str(args.get(text_key) or "").split() if text_key else []
        if len(self.query_words(" ".join(words))) < 3 or len(words) < 3 or not list(_listing_records(observation)):
            return
        if page_key and args.get(page_key) not in (None, "", 0, 1):
            return
        if self.steps_left() <= self.kept_steps() + 15 or self.time_left() < WRAP_UP_SECONDS + 40 or self.finished:
            return
        self.word_probed = True
        probe = {**args, text_key: " ".join(words[:2])}
        probe, problems = _check(probe, self.schemas[name], "args")
        if problems or _compact(probe) in self.search_sigs:
            return
        reply = self.env_call([{"name": name, "args": probe}])
        if reply is None:
            return
        seen = reply.get("observation") if not isinstance(reply.get("observations"), list) else (reply["observations"] or [None])[0]
        self.records.append(seen)
        if isinstance(seen, (dict, list)):
            _collect_ids(seen, self.seen_ids)
        self.learn_listings(name, probe, seen)
        self.note_search(probe, seen)
        self.trace.append({"kind": "word_probe", "args": probe})
        before = set(self.rules_told)
        self.learn_search(name, probe, seen)
        if not (set(self.rules_told) - before) and isinstance(seen, dict):
            rows = list(_listing_records(seen))
            self.side_notes.append(f"(A check made for you, one step: {name} {_compact(probe)} -> {len(rows)} rows" +
                                   (": " + ", ".join(f"{_ident_of(r)}" for r in rows[:12]) if rows else "") + ".)")

    def lead_words(self) -> int:
        """K when the store is shown to read no word after the K-th of a query (0 while unknown)."""
        rule = self.search_rule
        above = [k for k in rule["high"] if k > rule["low"]]
        return min(above) if above else 0

    def search_rules(self) -> list[str]:
        """What has been learned about this store's search, as short sentences."""
        out = []
        lead = self.lead_words()
        if lead:
            out.append(f"its keyword search reads only the first {lead} word{'s' if lead > 1 else ''} of a query: words after the {lead}{'st' if lead == 1 else 'nd' if lead == 2 else 'rd' if lead == 3 else 'th'} change nothing, so a query is its {lead} most telling words, value first")
        if self.search_rule["words"] == "all":
            out.append("it needs every word it reads: another word narrows the result, and a required value with the product's noun lists the listings that carry it")
        elif self.search_rule["words"] == "any":
            out.append("it takes any of the words (another word widens the result): a required value's distinctive word alone finds the listings carrying it")
        if self.search_rule["category"] is False:
            out.append("its keyword mode ignores the category argument: check each row's category yourself")
        return out

    def tell_rules(self, told: bool) -> None:
        for line in self.search_rules():
            key = line.split(":")[0]
            if key in self.rules_told:
                continue
            self.rules_told.add(key)
            self.trace.append({"kind": "search_rule", "rule": line})
            if told:
                self.side_notes.append("LEARNED ABOUT THIS STORE'S SEARCH (from your searches so far): " + line + ".")

    def plan_searches(self, reason: str) -> str:
        """The next searches, written by a model from the learned behaviour of the store's search and the searches made;
        code keeps only calls the schema accepts and that were not run already."""
        search = self.roles.get("search")
        if not search or self.search_plans >= 2 or self.steps_left() <= self.kept_steps() + 8 or self.time_left() < WRAP_UP_SECONDS + 60:
            return ""
        self.search_plans += 1
        tool = next((t["function"] for t in self.tools if t["function"]["name"] == search), {})
        rules = self.search_rules()
        reach = [n for n in self.trace if n.get("kind") == "reach_note"][-2:]
        history = []
        for entry in self.trace:
            if entry.get("kind") == "search_seen":
                history.append(entry["line"])
        prompt = SEARCH_PLAN_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
        prompt += "\n\nREQUIREMENTS NOW:\n" + self.ledger(False)
        items = self.items()
        if items:
            prompt += "\n\nPRODUCTS: " + "; ".join(f"{n} x{q}" for n, q in items)
        if self.cart_ids:
            prompt += "\n\nTHE CART NOW: " + ", ".join(self.cart_ids)
        prompt += f"\n\nWHY A PLAN NOW: {reason}."
        prompt += "\n\nTHE SEARCH TOOL: " + str(tool.get("name")) + ": " + str(tool.get("description") or "")[:400] + " | arguments: " + _compact(tool.get("parameters"))[:1500]
        prompt += "\n\nLEARNED ABOUT THIS STORE'S SEARCH:\n" + ("\n".join(f"- {r}" for r in rules) if rules else "- nothing proven yet (assume short queries, value first)")
        for n in reach:
            prompt += f"\n- a search with {n.get('total')} matches can show at most {n.get('reach')} rows over its pages, in the store's own order"
        prompt += "\n\nSEARCHES SO FAR (oldest first; arguments -> total matches, rows shown):\n" + ("\n".join(history[-16:]) if history else "(none)")
        result = self.judge(prompt, 1800, [READ_MODEL, JUDGE_MODEL], optional=True)
        self.trace.append({"kind": "search_plan", "reason": reason, "result": result})
        lead = self.lead_words()
        text_key = self.search_text_key()
        lines = []
        for entry in (result or {}).get("searches") or []:
            if not isinstance(entry, dict) or not isinstance(entry.get("arguments"), dict):
                continue
            call = dict(entry["arguments"])
            if lead and text_key and isinstance(call.get(text_key), str):
                words = [w for w in re.split(r"\s+", call[text_key].strip()) if w]
                call[text_key] = " ".join(words[:lead])
            checked, problems = _check(call, self.schemas[search], "args")
            if problems or _compact(checked) in self.search_sigs:
                continue
            lines.append(f"{len(lines) + 1}) {search} {_compact(checked)}" + (f" - {str(entry.get('why'))[:120]}" if entry.get("why") else ""))
            if len(lines) >= 4:
                break
        if not lines:
            return ""
        advice = str((result or {}).get("advice") or "").strip()[:240]
        return ("SEARCH PLAN (written from what this store's search is known to do" + (": " + "; ".join(rules) if rules else "") + "): "
                + " ".join(lines) + (f" Then: {advice}" if advice else "") + " Run these (several in one turn), then open the pages of the rows that may fit.")

    def page_through(self) -> str:
        """The remaining pages of the searches whose rows hold a cart line, when at most three pages remain; returns a
        note with the rows read (or "" when nothing was read)."""
        keys = self.page_argument()
        if not keys:
            return ""
        text, page = keys
        search = self.roles["search"]
        notes = []
        for wording, entry in list(self.searches.items()):
            if not entry["next"] or not entry["size"] or not any(i in entry["seen"] for i in self.cart_ids):
                continue
            remaining = (entry["total"] or 0) - len(entry["seen"])
            limit = ((self.schemas.get(search) or {}).get("properties") or {}).get(page) or {}
            most = limit.get("maximum")
            capped = isinstance(most, int) and (most <= 5 or (entry["size"] > 1 and most // entry["size"] + 1 <= 4))      # the store only shows a few pages: all can be read
            if remaining <= 0 or (not capped and remaining > (4 if self.plan.get("priority") == "cheapest" else 3) * entry["size"]):
                continue
            while entry["next"] and self.pages_read < 6 and self.steps_left() > self.kept_steps() + 6:
                args = {**{k: v for k, v in entry["args"].items() if k != page}, page: entry["next"]}
                args, problems = _check(args, self.schemas[search], "args")
                if problems:
                    break
                token = entry["next"]
                reply = self.env_call([{"name": search, "args": args}])
                self.pages_read += 1
                self.trace.append({"kind": "auto_open", "actions": [{"name": search, "args": args}], "reply": reply})
                if reply is None:
                    break
                observation = reply.get("observation") if not isinstance(reply.get("observations"), list) else (reply["observations"] or [None])[0]
                self.records.append(observation)
                if isinstance(observation, (dict, list)):
                    _collect_ids(observation, self.seen_ids)
                self.learn_listings(search, args, observation)
                self.note_search(args, observation)
                self.take_lines(observation)
                notes.append(f"{search} {_compact(args)} was run for you: " + _compact(_brief(observation, 160))[:2500])
                if entry["next"] == token:
                    break
        return " ".join(notes)

    def compact_history(self) -> None:
        """Once the conversation is long, tool results older than the last ten are cut to their start (the listings'
        facts stay in the notes, the requirements and the cart text; a page can be opened again). Done at most once per
        25k tokens of growth, so the prompt cache is broken rarely."""
        if self.context_tokens < 16000 or self.context_tokens - self.compacted_at < 8000:
            return
        self.compacted_at = self.context_tokens
        tools = [i for i, m in enumerate(self.messages) if m.get("role") == "tool"]
        shortened = 0
        for index in tools[:-3]:
            content = self.messages[index].get("content")
            if isinstance(content, str) and len(content) > 500 and "(older result shortened" not in content:
                ids = [i for i in dict.fromkeys(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-:.]{5,40}", content)) if i in self.seen_ids][:20]
                self.messages[index] = {**self.messages[index], "content": content[:300] + " ...(older result shortened"
                                        + (f"; listings in it: {', '.join(ids)}" if ids else "") + "; open a page again if a detail is needed)"}
                shortened += 1
        self.trace.append({"kind": "compacted", "context": self.context_tokens, "shortened": shortened})

    def derive_links(self) -> str:
        """For each rule tying a product to another whose item is in the cart: that line's page and property list are
        opened when missing, and the model works out what the rule now demands of the product (a condition that holds
        or not, what a fit needs); the requirement goes into the ledger and a search for it is queued."""
        if not self.links or not self.cart_ids or self.time_left() < WRAP_UP_SECONDS:
            return ""
        signature = ",".join(sorted(self.cart_ids))
        # a link is worked out again only when the item it was worked out from has left the cart
        todo = [i for i in range(len(self.links)) if self.link_done.get(i) != signature and self.link_other.get(i) not in self.cart_ids]
        if not todo:
            return ""
        for i in todo:
            self.link_done[i] = signature
        page, table = self.page_tool(), self.table_guess()
        wanted = []
        for ident in self.cart_ids[:6]:
            if page and ident not in self.paged:
                wanted.append((page, ident))
            if table and table != page and not any(isinstance(v, dict) and len(v) >= 3 for v in (self.views.get(ident) or {}).values()):
                wanted.append((table, ident))
        actions = self.open_actions(wanted)[:3] if self.steps_left() > self.kept_steps() + 6 else []
        if actions:
            reply = self.env_call(actions)
            self.trace.append({"kind": "auto_open", "actions": actions, "reply": reply})
            observations = (reply or {}).get("observations") if isinstance((reply or {}).get("observations"), list) else [(reply or {}).get("observation")]
            for action, observation in zip(actions, observations):
                self.records.append(observation)
                if isinstance(observation, (dict, list)):
                    _collect_ids(observation, self.seen_ids)
                self.learn_listings(action["name"], action["args"], observation)
                self.take_lines(observation)
        cart = "\n".join(f"- {ident}: " + _compact(_brief(self.views.get(ident) or {}, 400))[:2000] for ident in self.cart_ids[:6])
        sample = [v for i, v in reversed(list(self.views.items())) if i not in self.cart_ids][:14]
        titles = "\n".join("- " + _compact({k: (x if len(x) <= 160 else x[:150] + "...") for k, x in v.items() if isinstance(x, str)})[:420] for v in sample)
        notes = []
        rules = "\n".join(f"RULE {i}: PRODUCT = {self.links[i]['product']}; OTHER product = {self.links[i]['on']}; " + _compact(self.links[i]) for i in todo)
        prompt = (LINK_PROMPT + "\n\nREQUEST:\n" + str(self.policy.get("query") or "")
                  + "\n\nTHE RULES:\n" + rules
                  + "\n\nREQUIREMENTS NOW:\n" + self.ledger(False)
                  + "\n\nTHE CART, WITH WHAT IS KNOWN OF EACH LINE:\n" + cart
                  + "\n\nSAMPLE LISTINGS SEEN (not in the cart):\n" + (titles or "(none)"))
        result = self.judge(prompt, 400 + 350 * len(todo), [READ_MODEL, JUDGE_MODEL])
        self.trace.append({"kind": "link", "links": todo, "result": result})
        answers = result.get("links") if isinstance(result, dict) and isinstance(result.get("links"), list) else []
        for answer in answers:
            i = answer.get("rule") if isinstance(answer, dict) else None
            if not isinstance(i, int) or i not in todo or not answer.get("other_item"):
                continue
            link = self.links[i]
            self.link_other[i] = str(answer.get("other_item"))
            requirement = str(answer.get("requirement") or "").strip()
            before = self.derived.get(i)
            if requirement:
                self.derived[i] = requirement
            else:
                self.derived.pop(i, None)
            if self.derived.get(i) != before:
                self.ledger_version += 1
            if requirement:
                note = f"LINKED REQUIREMENT: the {link['on']} in the cart is {answer.get('other_item')}, so the {link['product']} (and only it) now needs: {requirement}."
                if answer.get("product_line_ok") is False:
                    note += f" The cart's {link['product']} line does not meet it: replace it with a fitting listing."
                if answer.get("none_fit") is True:
                    note += (f" None of the {link['product']} listings seen so far meets it. If the search finds none either, this {link['on']} is the wrong pick:"
                             f" put in another {link['on']} listing that meets its own requirements and has fitting {link['product']} listings.")
                phrase = self.fit_phrase(str(answer.get("search") or ""))
                if phrase and self.roles.get("search") and phrase.lower() not in self.chased:
                    self.chased.add(phrase.lower())
                    self.open_queue.append(("?search", phrase))
                    note += f" A search for \"{phrase}\" is made for you."
            else:
                note = f"LINKED RULE: with the {link['on']} in the cart ({answer.get('other_item')}), the rule puts nothing on the {link['product']}" + (" (its condition does not hold)." if answer.get("holds") is False else ".")
            notes.append(note)
        return " ".join(notes)

    def table_guess(self) -> str | None:
        """The tool that shows a listing's property list: known from results, or, in a store whose page tool showed no
        property table, the remaining single-listing tool whose name or description speaks of properties."""
        known = self.table_tool()
        if known or not any(t["text"] and not t["table"] for t in self.traits.values()):
            return known
        core = {self.roles.get(r) for r in ("search", "message", "cart", "cart_view", "order", "decline")} - {None}
        for tool in self.tools:
            name = tool["function"]["name"]
            argument = self.id_argument(name)
            if (name not in core and name not in self.traits and argument and not argument[1]
                    and _PROPERTY_KEYS.search(name + " " + str(tool["function"].get("description") or ""))):
                return name
        return None

    def fill_menu(self) -> bool:
        """In a store whose pages show no property table, the property lists of a few listings of each requested kind
        are opened before the questions are planned (listings opened in full first, spread over the kinds)."""
        if self.menu_filled:
            return False
        table = self.table_guess()
        split = bool(table)
        if not split:
            # a store whose pages carry the property list: the pages themselves (several in one call when a tool takes them)
            table = self.page_tool()
        argument = self.id_argument(table)
        if not table or not argument:
            return False        # no tool known yet that shows property lists: tried again once one is
        self.menu_filled = True
        if self.steps_left() < 18 or self.time_left() < WRAP_UP_SECONDS + 80:
            return False
        self.search_missing_kinds()
        if self.finished or self.dead:
            return True
        products = max(1, len(self.items()))
        per_kind = (3 if products == 1 else 2) if split or not argument[1] else (5 if products == 1 else 3)
        kinds: dict[str, list[str]] = {}
        tabled: dict[str, int] = {}
        for ident, view in self.views.items():
            label = next((str(v).strip().lower() for k, v in view.items() if isinstance(v, str) and _KIND_KEYS.search(str(k))), "")
            if any(isinstance(v, dict) and len(v) >= 3 for v in view.values()):
                tabled[label] = tabled.get(label, 0) + 1
            elif ident in self.paged or ident in self.opened:
                kinds.setdefault(label, []).insert(0, ident)
            else:
                kinds.setdefault(label, []).append(ident)
        # the kinds named like the requested products first, then those with the most listings the model opened
        order = sorted(kinds, key=lambda k: (-self.kind_named(k), -sum(1 for i in kinds[k] if i in self.paged or i in self.opened), -len(kinds[k])))[:products]
        rows = self.priced_rows([self.views])
        queues = []
        for label in order:
            # the likely fits first: rows showing the stated requirements, for a cheapest request the cheapest of them
            checks = self.stated_checks(label)
            cheapest = self.plan.get("priority") == "cheapest"
            ranked = sorted(kinds[label], key=lambda i: (-self.stated_score(self.views.get(i) or {}, checks),
                                                          (rows.get(i) or {}).get("price", 0) if cheapest else 0))
            queues.append(ranked[: max(0, per_kind - tabled.get(label, 0))])
        wanted = []
        while any(queues):
            for queue in queues:
                if queue:
                    wanted.append(queue.pop(0))
        if argument[1]:
            cap = self.many_limit(table)
            # a tool taking few at once gets more calls, while steps allow (the questions decide the score)
            rounds = 2 if cap >= 4 else (4 if self.steps_left() >= 30 else 3)
            wanted = wanted[: cap * min(rounds, max(1, self.steps_left() - 16))]
            batches = [{argument[0]: wanted[i:i + cap]} for i in range(0, len(wanted), cap)]
        else:
            wanted = wanted[: min(5 if split else 3, self.steps_left() - 14)]
            batches = [{argument[0]: ident} for ident in wanted]
        if not wanted:
            return False
        calls = []
        for index, batch in enumerate(batches):
            args, problems = _check(batch, self.schemas[table], "args")
            if not problems:
                calls.append({"id": f"props_{self.step}_{index}", "type": "function", "function": {"name": table, "arguments": json.dumps(args)}})
        if not calls:
            return False
        self.trace.append({"kind": "fill_menu", "tool": table, "ids": wanted})
        self.messages.append({"role": "assistant", "content": "Before planning the questions I read the details of a few listings of each requested kind.", "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        return True

    def search_missing_kinds(self) -> None:
        """A requested product with no listing of its kind seen yet is searched for (its stated words, on its shelf when the
        store has one), so that the pages read before the questions cover every product. At most two such searches."""
        search = self.roles.get("search")
        names = [n for n, _ in self.items()]
        if not search or len(names) < 2:
            return
        labels = {next((str(v).strip().lower() for k, v in view.items() if isinstance(v, str) and len(v) <= 70 and _KIND_KEYS.search(str(k))), "") for view in self.views.values()}
        missing = [n for n in names if not any(self.word_stems(n) & self.word_stems(g) for g in labels)][:2]
        calls = []
        for index, name in enumerate(missing):
            words = [line.partition(":")[2].strip() for line in self.stated if ":" in line and self.word_stems(name) & self.word_stems(line.partition(":")[0])]
            phrase = " ".join([w for w in words if w and not re.search(r"\d|listing", w)][:2] + [name]).strip()[:80]
            args = self.search_args(phrase, name)
            if args:
                calls.append({"id": f"kind_{self.step}_{index}", "type": "function", "function": {"name": search, "arguments": json.dumps(args)}})
        if not calls:
            return
        self.messages.append({"role": "assistant", "content": "I look for listings of the requested products not seen yet.", "tool_calls": calls})
        replies = self.run_calls(calls, planned=True)
        for call_id, text in replies:
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

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

    def cart_number(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """The cart tool's number argument (the units of a line, whatever it is called) when the tool has one."""
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                return key, spec
        return None

    def wanted_units(self, ident: str) -> tuple[int, str] | None:
        """The units the request wants of a listing: the plan's quantity, for several products that of the product whose
        name is closest to the listing's kind label; None when the kind is not known."""
        items = self.items()
        if not items:
            quantity = self.plan.get("quantity")
            return (quantity, str(self.plan.get("product") or "the product")) if isinstance(quantity, int) and quantity > 0 else None
        label = self.line_kind(ident)
        if not label:
            return None

        def closeness(name: str) -> float:
            a, b = self.word_stems(name), self.word_stems(label)
            return len(a & b) / max(1, len(a | b))
        best = max(items, key=lambda item: closeness(item[0]))
        if closeness(best[0]) <= 0 or not isinstance(best[1], int) or best[1] < 1:
            return None
        return best[1], best[0]

    def fix_cart_number(self, call_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A cart call adding a listing with a number other than the units wanted (the number's name can read like a line
        index: 0, 1, 2 ...) goes out with the units wanted, with a note; the same call sent again stands. A zero on a line held
        in the cart takes it out."""
        number = self.cart_number(name)
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if not number or (self.add_signature is not None and signature != self.add_signature):
            return args
        key, spec = number
        value = args.get(key)
        target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip() and " " not in v.strip()), None)
        if not isinstance(value, int) or isinstance(value, bool) or not target or (value <= 0 and target in self.cart_ids):
            return args
        wanted = self.wanted_units(target)
        if not wanted or wanted[0] == value or (target, value) in self.cart_overrides:
            return args
        if value > 0:
            self.cart_overrides.add((target, value))
        fixed = dict(args)
        fixed[key] = wanted[0]
        checked, problems = _check(fixed, self.schemas[name], "args")
        if problems:
            return args
        self.cut_notes[call_id] = (self.cut_notes.get(call_id, "") + f" {key} is the number of units on that line, not a line number: it was set to {wanted[0]}, the quantity the request wants of the {wanted[1]}."
                                   + (f" If the shopper really wants {value}, send the same call again." if value > 0 else f" {key} 0 takes out a line that is in the cart.")).strip()
        self.trace.append({"kind": "cart_number", "item": target, "sent": value, "set": wanted[0]})
        return checked

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        wanted = self.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        properties = self.schemas[name].get("properties") or {}
        signature = _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]})
        if self.add_signature is not None and signature != self.add_signature:
            return ""
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (properties.get(k) or {}) and v.strip()), None)
                if value == 0 and spec.get("minimum", 1) == 0 and target in self.cart_ids:
                    return ""       # this store's way of taking a line out
                if self.items():
                    allowed = {q for _, q in self.items()}
                    if isinstance(value, int) and value not in allowed:
                        return f"the request needs these quantities: {', '.join(f'{n} x{q}' for n, q in self.items())}, but this call asks for {value}."
                    return ""
                if isinstance(value, int) and value != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {value}."
        return ""

    @staticmethod
    def priced_rows(values: list[Any]) -> dict[str, dict[str, Any]]:
        """Listing rows that carry an id and a price, with the store's own keys read as price, stock and category."""
        rows: dict[str, dict[str, Any]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                ident = next((v for k, v in value.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v)
                              and any(c.isdigit() for c in v)), None)
                prices = [(k, value[k]) for k in _price_keys(value)]
                if ident and prices:
                    row = rows.setdefault(ident, {})
                    row["price"] = sorted(prices, key=lambda kv: (0 if "price" in str(kv[0]).lower() else 1))[0][1]
                    for k, v in value.items():
                        if isinstance(v, bool) and re.search(r"stock|can.?buy|available|buyable|orderable|purchas|^ready$|ready.?to.?ship|ships?.?now", str(k), re.I):
                            row["in_stock"] = v
                        elif isinstance(v, str) and v.strip() and _KIND_KEYS.search(str(k)):
                            row["category"] = v.strip()
                for v in value.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        for value in values:
            walk(value, 0)
        return rows

    def ids_for_titles(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A listing named by its title (or the start of it, as a length limit cuts it) where a cart or details tool
        wants an id gets the id of the one listing seen with that title."""
        if name not in {self.roles.get(r) for r in ("cart", "details", "view")} - {None}:
            return args
        titles: dict[str, set[str]] = {}

        def walk(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, dict):
                ident = next((v for k, v in value.items() if isinstance(v, str) and _IDENT_KEYS.search(str(k)) and _ID_HINT.match(v)
                              and any(c.isdigit() for c in v)), None)
                if ident:
                    for k, v in value.items():
                        if isinstance(v, str) and len(v.strip()) >= 8 and re.search(r"name|title|caption|heading|label", str(k), re.I):
                            titles.setdefault(" ".join(v.lower().split()), set()).add(ident)
                for v in value.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth + 1)
            elif isinstance(value, list):
                for v in value:
                    walk(v, depth + 1)

        def resolve(value: Any) -> Any:
            if not isinstance(value, str) or " " not in value.strip() or value in self.seen_ids:
                return value
            if not titles:
                for record in self.records:
                    walk(record, 0)
            wanted = " ".join(value.lower().split())
            if len(wanted) < 8:
                return value
            found = {i for title, ids in titles.items() if title == wanted or title.startswith(wanted) for i in ids}
            return found.pop() if len(found) == 1 else value

        fixed: dict[str, Any] = {}
        for key, value in args.items():
            fixed[key] = [resolve(v) for v in value] if isinstance(value, list) else resolve(value)
        if fixed != args:
            self.trace.append({"kind": "title_to_id", "tool": name, "before": args, "after": fixed})
        return fixed

    def unchecked_cheaper(self) -> list[tuple[str, Any]]:
        """For a cheapest request: result rows priced below an item in the cart, in that item's category and not shown out
        of stock or over the budget, whose details were never opened (at most five, cheapest first)."""
        if self.plan.get("priority") != "cheapest" or not self.cart_ids:
            return []
        rows = self.priced_rows(self.records)
        several = bool(self.items())
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        found: dict[str, Any] = {}
        for held_id in (self.cart_ids[-4:] if several else self.cart_ids[-1:]):
            held = rows.get(held_id)
            if not held:
                continue
            for ident, row in rows.items():
                if ident in self.opened or ident in self.cart_ids or row["price"] >= held["price"]:
                    continue
                if row.get("in_stock") is False or (not several and budget is not None and row["price"] * quantity > budget):
                    continue
                if (held.get("category") or row.get("category")) and row.get("category") != held.get("category"):
                    continue
                found[ident] = row["price"]
        return [(ident, price) for ident, price in sorted(found.items(), key=lambda kv: kv[1])[:5]]

    def sweep_tool(self) -> tuple[str, str, str, str | None, dict[str, Any]] | None:
        """A search tool that filters by category and by a price ceiling: (tool, category key, price key, page key, fixed
        arguments). A required mode switch is set to its filtering value; a tool with other required arguments is skipped."""
        busy = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view", "message", "details", "view")} - {None}
        for tool in self.tools:
            name = tool["function"]["name"]
            if name in busy or name in self.capped:
                continue
            schema = self.schemas.get(name) or {}
            props = schema.get("properties") or {}
            kind = next((k for k, v in props.items() if isinstance(v, dict) and (v.get("type") == "string" or v.get("enum"))
                         and re.search(r"categor|range|shelf|department|section|aisle|dept", k, re.I)), None)
            price = next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") in ("number", "integer")
                          and re.search(r"price|cost|ceiling|budget|cap\b", k, re.I) and not re.search(r"min|floor|offset|row|page|count", k, re.I)), None)
            page = next((k for k, v in props.items() if isinstance(v, dict) and k not in (kind, price)
                         and re.search(r"token|cursor|after|page|offset|start|marker|skip", k, re.I)), None)
            if not (kind and price):
                continue
            fixed: dict[str, Any] = {}
            usable = True
            for key in schema.get("required") or []:
                if key in (kind, price):
                    continue
                spec = props.get(key) if isinstance(props.get(key), dict) else {}
                choice = next((v for v in spec.get("enum") or [] if isinstance(v, str) and re.search(r"filter|facet|field|categor|browse|section|shelf|attribute", v, re.I)), None)
                if choice is None:
                    usable = False
                    break
                fixed[key] = choice
            if usable:
                return name, kind, price, page, fixed
        return None

    def sweep_args(self, category: str, price: float) -> dict[str, Any] | None:
        """The first page's arguments of a price sweep for a category, or None when the store's tool cannot express it."""
        tool = self.sweep_tool()
        if not tool:
            return None
        name, kind_key, price_key, _, fixed = tool
        props = (self.schemas.get(name) or {}).get("properties") or {}
        spec = props.get(kind_key) or {}
        label: Any = category
        if spec.get("enum"):
            label = next((v for v in spec["enum"] if isinstance(v, str) and v.strip().lower() == category.lower()), None)
            if label is None:
                return None
        ceiling: Any = price - 1 if (props.get(price_key) or {}).get("type") == "integer" else round(price - 0.01, 2)
        if ceiling < 0:
            return None
        args = dict(fixed)
        args.update({kind_key: label, price_key: ceiling})
        checked, problems = _check(dict(args), self.schemas[name], "args")
        return None if problems else checked

    def sweep_targets(self) -> list[tuple[str, str, float]]:
        """Cart items of a cheapest request not yet swept: (id, category label, unit price)."""
        if self.plan.get("priority") != "cheapest" or not self.cart_ids or not self.sweep_tool():
            return []
        rows = self.priced_rows(self.records)
        out = []
        for held_id in (self.cart_ids[-4:] if self.items() else self.cart_ids[-1:]):
            held = rows.get(held_id)
            if held and held.get("category") and held_id not in self.swept and self.sweep_args(held["category"], float(held["price"])) is not None:
                out.append((held_id, held["category"], float(held["price"])))
        return out

    def run_sweep(self) -> bool:
        """Search the cart item's own category below its price (every page, up to three) and name the cheaper listings
        that were never opened, so a cheapest request is not settled on the first fitting item seen."""
        targets = self.sweep_targets()
        tool = self.sweep_tool()
        self.sweep_due = False
        if not targets or not tool or self.steps_left() < 12 or self.time_left() < WRAP_UP_SECONDS + 40:
            return False
        name, kind_key, price_key, page_key, _ = tool
        held_id, category, price = targets[0]
        self.swept.add(held_id)
        args = self.sweep_args(category, price)
        if args is None:
            return False
        seen_rows: dict[str, dict[str, Any]] = {}
        last_text = ""
        last_id = ""
        for page in range(3):
            checked, problems = _check(dict(args), self.schemas[name], "args")
            if problems:
                break
            call = {"id": f"sweep_{self.step}_{page}", "type": "function", "function": {"name": name, "arguments": json.dumps(checked)}}
            self.messages.append({"role": "assistant", "content": f"Checking {category} below {price} for anything cheaper than {held_id}.", "tool_calls": [call]})
            replies = self.run_calls([call], planned=True)
            last_id, last_text = replies[-1] if replies else (call["id"], "")
            for call_id, text in replies[:-1]:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            observation = self.records[-1] if self.records else None
            if self.finished or self.dead or not isinstance(observation, (dict, list)):
                break
            seen_rows.update(self.priced_rows([observation]))
            token = None
            if isinstance(observation, dict) and page_key:
                token = next((v for k, v in observation.items() if isinstance(v, (str, int)) and not isinstance(v, bool) and v not in ("", None)
                              and re.search(r"follow|next|cursor|token|after|marker|more", str(k), re.I)), None)
            if token is None or self.steps_left() < 10:
                break
            args[page_key] = token
            if page < 2:
                self.messages.append({"role": "tool", "tool_call_id": last_id, "content": last_text})
                last_id, last_text = "", ""
        cheaper = sorted(((r["price"], i) for i, r in seen_rows.items()
                          if r["price"] < price and i not in self.opened and i not in self.cart_ids and r.get("in_stock") is not False
                          and (not r.get("category") or r.get("category") == category)), key=lambda x: x[0])[:6]
        if cheaper:
            note = (f"\nPRICE CHECK for the cheapest rule: {category} listings priced below {held_id} ({price}) that you never opened: "
                    + ", ".join(f"{i} ({p})" for p, i in cheaper)
                    + f". Open their details (several in one call). If one meets every requirement, put the cheapest such one in the cart instead of {held_id}; if none does, order again.")
        else:
            note = f"\nPRICE CHECK for the cheapest rule: no unopened {category} listing in stock below {held_id}'s price ({price}) turned up; if the cart is otherwise checked, order."
        self.trace.append({"kind": "sweep", "item": held_id, "category": category, "price": price, "cheaper": cheaper})
        if last_id:
            self.messages.append({"role": "tool", "tool_call_id": last_id, "content": last_text + note + "\n" + self.status()})
        return True

    def closest_wanted(self) -> bool:
        """The request asks, when nothing fits, for the closest listing to be put forward (so the task ends with an order)."""
        said = str(self.plan.get("nothing_fits") or "").strip().lower()
        if said == "closest":
            return True
        if said == "decline":
            return False
        query = str(self.policy.get("query") or "")
        return bool(re.search(r"\bif\b.{0,25}\b(fits?|suits?|works?|matches|meets)\b", query, re.I) and re.search(r"\b(nearest|closest|fewest|next best)\b", query, re.I))

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether this finish call ends without buying. A store may use one tool for both finishes, told apart by an
        argument value (for example a mode of "refuse" against "commit")."""
        if name != self.roles.get("decline"):
            return False
        if self.roles.get("order") != name:
            return True
        properties = (self.schemas.get(name) or {}).get("properties") or {}
        return any(isinstance(v, str) and "enum" in (properties.get(k) or {}) and self.no_purchase_value(name, k, v) for k, v in args.items())

    def no_purchase_value(self, name: str, key: str, value: str) -> bool:
        """Whether an enum value of a finish tool shared by both finishes ends the task without buying: by its own
        wording, or else by what the tool's description says that value does (trusted only when exactly one value of
        the enum reads that way)."""
        spec = ((self.schemas.get(name) or {}).get("properties") or {}).get(key) or {}
        values = [v for v in spec.get("enum") or [] if isinstance(v, str)]
        if _DECLINE_WORDS.search(value) or any(_DECLINE_WORDS.search(v) for v in values):
            return bool(_DECLINE_WORDS.search(value))
        description = next((str(t["function"].get("description") or "") for t in self.tools if t["function"]["name"] == name), "").lower()

        def reads_as_no_purchase(candidate: str) -> bool:
            # the value as the description quotes it, else as a whole word
            match = (re.search(r"[\"'`]" + re.escape(candidate.lower()) + r"[\"'`]", description)
                     or re.search(r"(?<![a-z0-9_])" + re.escape(candidate.lower()) + r"(?![a-z0-9_])", description))
            if not match:
                return False
            window = description[match.end(): match.end() + 90]
            for other in values:
                if other != candidate:
                    cut = re.search(r"(?<![a-z0-9_])" + re.escape(other.lower()) + r"(?![a-z0-9_])", window)
                    if cut:
                        window = window[:cut.start()]
            window = re.split(r"[.;]", window)[0]
            return bool(_NO_PURCHASE.search(window))

        marked = [v for v in values if reads_as_no_purchase(v)]
        return len(marked) == 1 and marked[0] == value

    def sold_out_lines(self) -> list[str]:
        """Cart lines the shopper said are sold out or whose latest page or row shows them out of stock."""
        rows = self.priced_rows([self.views])
        out = [i for i in self.cart_ids if i in self.gone or (rows.get(i) or {}).get("in_stock") is False]
        self.gone |= set(out)
        return out

    def take_lines(self, observation: Any) -> str:
        """Shopper lines that came with the result of a call made for the model are read like any other; what they mean
        is shown with the next result."""
        _, pending = self.split_reply(observation, True)
        if not pending:
            return ""
        meaning = self.interpret(pending, None)
        note = "A shopper line came with a result fetched for you: " + " | ".join(pending[:4]) + (" " + meaning if meaning else "")
        self.side_notes.append(note)
        return note

    def auto_run(self, action: dict[str, Any], notes: list[str]) -> Any:
        """One call made for the model (a cart view, cart lines' pages), with the usual bookkeeping."""
        checked, problems = _check(dict(action["args"]), self.schemas.get(action["name"]) or {}, "args")
        if problems:
            return None
        reply = self.env_call([{"name": action["name"], "args": checked}])
        self.trace.append({"kind": "auto_open", "actions": [action], "reply": reply})
        if reply is None:
            return None
        self.recent = []
        observation = reply.get("observation") if not isinstance(reply.get("observations"), list) else (reply["observations"] or [None])[0]
        self.records.append(observation)
        if isinstance(observation, (dict, list)):
            _collect_ids(observation, self.seen_ids)
        self.learn_listings(action["name"], checked, observation)
        if action["name"] == self.roles.get("cart_view"):
            self.last_cart_view = observation
            self.cart_view_at = self.step
        elif action["name"] in (self.roles.get("details"), self.roles.get("view")) or action["name"] in self.traits:
            self.opened.update(v for v in checked.values() if isinstance(v, str))
            self.opened.update(x for v in checked.values() if isinstance(v, list) for x in v if isinstance(x, str))
        notes.append(f"{action['name']} {_compact(checked)[:160]} was run for you: " + _compact(self.shown(action["name"], observation))[:2500])
        self.take_lines(observation)
        return observation

    def check_cheapest(self) -> str:
        """For a cheapest request, before the order: every cart line against the opened listings of its kind and the
        cheapest rows never opened; a note when a cheaper listing meets the requirements."""
        lines = list(self.cart_ids)
        notes = [self.check_change(forced=line) for line in lines[:3]]
        return " ".join(n for n in notes if n)

    def check_change(self, forced: str | None = None) -> str:
        """After the shopper announced a change while the cart holds a pick (several at once, in any wording): the line
        that went out of stock is known, the cart's new price is read, and the listings opened so far are judged against
        the requirements as they stand now, so the best one that meets everything is named (or a new search asked for)."""
        if forced:
            due = {"price"}
        else:
            due, self.change_due = self.change_due, set()
        if (not due or not self.cart_ids or self.finished or self.dead or self.steps_left() < 3 or (self.change_checks >= 4 and not forced)
                or self.time_left() < WRAP_UP_SECONDS + 20):
            return ""
        if not forced:
            self.change_checks += 1
        notes: list[str] = []
        several = bool(self.items())
        rows = self.priced_rows([self.views])
        if "gone" in due and not (set(self.cart_ids) & self.gone):
            # several lines held: their pages show which one went out of stock
            page = self.page_tool()
            if page and self.steps_left() > 4:
                for action in self.open_actions([(page, i) for i in self.cart_ids])[:1]:
                    self.auto_run(action, notes)
                rows = self.priced_rows([self.views])
            self.gone |= {i for i in self.cart_ids if (rows.get(i) or {}).get("in_stock") is False}
        if (not forced and "price" in due and self.roles.get("cart_view") and self.cart_view_at < self.change_heard_at and self.steps_left() > 4
                and not self.finished):
            # the new price is read where the cart shows it
            self.auto_run({"name": self.roles["cart_view"], "args": {}}, notes)
            rows = self.priced_rows([self.views])
        if self.finished or self.dead:
            return " ".join(notes)
        self.gone |= {i for i in self.cart_ids if (rows.get(i) or {}).get("in_stock") is False}
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        hops = {i for p in self.pointers for i in p["hops"] + ([p["listing"]] if p["listing"] else [])}
        lines = [forced] if forced else ([i for i in self.cart_ids if i in self.gone] if several else self.cart_ids[-1:])
        for line in lines:
            held = rows.get(line) or {}
            plan_qty = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
            qty = self.cart_qty.get(line) if isinstance(self.cart_qty.get(line), int) and self.cart_qty.get(line) > 0 else plan_qty
            kind = held.get("category")
            pool = []
            for ident, row in sorted(rows.items(), key=lambda kv: kv[1]["price"]):
                view = self.views.get(ident) or {}
                detailed = ident in self.paged or any(isinstance(v, dict) and len(v) >= 3 for v in view.values())
                if ident in self.gone or (ident != line and (row.get("in_stock") is False or not detailed or ident in hops)):
                    continue
                if ident != line and kind and row.get("category") and row["category"] != kind:
                    continue
                if ident != line and not several and budget is not None and row["price"] * qty > budget:
                    continue
                pool.append(ident)
            pool = ([line] if line in pool else []) + [i for i in pool if i != line][:9]
            price = held.get("price")
            over = not several and budget is not None and isinstance(price, (int, float)) and price * qty > budget
            # the opened listings are judged when the answer can change the pick
            wanted = (line in self.gone or over or bool(due & {"needs", "dropped"})
                      or (self.plan.get("priority") == "cheapest" and bool(due & {"price", "budget"})))
            page = self.page_tool()
            if self.plan.get("priority") == "cheapest" and wanted and page and self.steps_left() > self.finish_reserve() + (4 if forced else 3):
                # results seen but never opened that may be the cheapest fit now: opened in one call, so they are judged too
                bar = price if isinstance(price, (int, float)) and line not in self.gone and not over else None
                unread = []
                for ident, row in sorted(rows.items(), key=lambda kv: kv[1]["price"]):
                    view = self.views.get(ident) or {}
                    if (ident in pool or ident in self.gone or ident == line or row.get("in_stock") is False or ident in hops or ident in self.paged
                            or any(isinstance(v, dict) and len(v) >= 3 for v in view.values())):
                        continue
                    if kind and row.get("category") and row["category"] != kind:
                        continue
                    if (bar is not None and row["price"] >= bar) or (not several and budget is not None and row["price"] * qty > budget):
                        continue
                    unread.append(ident)
                limit = self.many_limit(page) if (self.id_argument(page) or ("", False))[1] else 1
                if unread:
                    for action in self.open_actions([(page, i) for i in unread[:limit]])[:1]:
                        self.auto_run(action, notes)
                    rows = self.priced_rows([self.views])
                    for ident in unread[:limit]:
                        view = self.views.get(ident) or {}
                        if ((ident in self.paged or any(isinstance(v, dict) and len(v) >= 3 for v in view.values()))
                                and (rows.get(ident) or {}).get("in_stock") is not False and ident in rows):
                            pool.append(ident)
            fits: list[str] = []
            fails: dict[str, Any] = {}
            judged = False
            if pool and wanted and (pool != [line] or bool(due & {"needs", "dropped"})):
                prompt = (CHANGE_PROMPT + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
                          + "\n\nTHE CHANGE THE SHOPPER ANNOUNCED:\n" + " / ".join(self.said[-2:])
                          + "\n\nCURRENT REQUIREMENTS:\n" + self.ledger()
                          + "\n\nTHE PICK IN THE CART: " + line
                          + "\n\nTHE LISTINGS (their records):\n" + "\n".join(f"- {i}: " + _compact(_brief(self.views.get(i) or {}, 160))[:900] for i in pool))
                result = self.judge(prompt, 900, [READ_MODEL, MAIN_MODEL], optional=True)
                self.trace.append({"kind": "change_check", "line": line, "due": sorted(due), "pool": pool, "result": result})
                if isinstance(result, dict) and isinstance(result.get("fits"), list):
                    judged = True
                    fits = [str(i) for i in result["fits"] if str(i) in pool]
                    fails = {line: result.get("cart_line")} if result.get("cart_line") else {}
            problems = []
            if line in self.gone:
                problems.append(f"{line} can no longer be bought (sold out)")
            if over:
                problems.append(f"{line} now costs {_num(price)} x {qty} = {_num(price * qty)}, over the budget of {_num(budget)}")
            if judged and line not in fits and line not in self.gone:
                problems.append(f"{line} no longer meets every requirement" + (f" ({str(fails.get(line))[:160]})" if fails.get(line) else ""))
            options = [i for i in fits if i != line]
            if self.plan.get("priority") == "cheapest":
                options.sort(key=lambda i: (rows.get(i) or {}).get("price", 0))
                if (not problems and options and isinstance(price, (int, float))
                        and (rows.get(options[0]) or {}).get("price", price) < price):
                    problems.append(f"{options[0]} ({_num(rows[options[0]]['price'])}) meets every requirement now and costs less than {line} ({_num(price)})")
            if forced and not (options and isinstance(price, (int, float)) and (rows.get(options[0]) or {}).get("price", price) < price):
                continue        # the check before an order only speaks up about a cheaper fit
            if not problems:
                if judged or "price" in due or "budget" in due:
                    notes.append(f"CHANGE CHECK: {line} still meets everything after the change" + (f" (price {_num(price)})" if isinstance(price, (int, float)) else "")
                                 + ("; nothing you opened beats it" if self.plan.get("priority") == "cheapest" else "") + ": keep it, re-check its stock and price, then order.")
                continue
            if options:
                best = options[0]
                self.repick = best
                notes.append(f"CHANGE CHECK: " + "; ".join(problems) + f". Of the listings you have opened, {best}"
                             + (f" ({_num(rows[best]['price'])})" if best in rows else "") + " meets every current requirement"
                             + (" and is the cheapest such one" if self.plan.get("priority") == "cheapest" else "")
                             + f": take {line} out of the cart, add {best} (quantity {qty}), then order. Other fits: " + (", ".join(options[1:4]) or "none") + ".")
            else:
                self.repick = None
                notes.append(f"CHANGE CHECK: " + "; ".join(problems) + ". None of the other listings you have opened meets every current requirement"
                             + (" within the budget" if budget is not None and not several else "") + ": take it out of the cart, search again (the requested kind with the "
                             "requirement words, the category, a price limit if the search takes one, further result pages), open the candidates, put the best fit in the cart, then order.")
        return " ".join(notes)

    def audit_decline(self, late: bool = False) -> str:
        if self.time_left() < (30 if late else WRAP_UP_SECONDS + 30):
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
        result = self.judge(prompt, 1200, [READ_MODEL, MAIN_MODEL], optional=True)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.audit_fits = [f for f in fits if f in self.seen_ids]
        if not fits:
            return ""
        text = "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."
        page = self.page_tool()
        unread = [f for f in fits if f not in self.paged]
        if page and unread and self.steps_left() > self.kept_steps() + 2:
            several = (self.id_argument(page) or ("", False))[1]
            wanted = [(page, f) for f in unread[: self.many_limit(page) if several else 1]]
            actions = self.open_actions(wanted)[:1]
            reply = self.env_call(actions) if actions else None
            self.trace.append({"kind": "auto_open", "actions": actions, "reply": reply})
            if reply is not None:
                observation = reply.get("observation") if not isinstance(reply.get("observations"), list) else (reply["observations"] or [None])[0]
                self.records.append(observation)
                if isinstance(observation, (dict, list)):
                    _collect_ids(observation, self.seen_ids)
                self.learn_listings(actions[0]["name"], actions[0]["args"], observation)
                self.take_lines(observation)
                self.opened.update(f for _, f in wanted)
                text += f" Their pages were opened for you ({actions[0]['name']} {_compact(actions[0]['args'])}): " + _compact(self.shown(actions[0]["name"], observation))[:3500]
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
                fitting = [v for v in spec["enum"] if isinstance(v, str) and self.no_purchase_value(name, key, v) == declining]
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

    def held_total(self) -> float:
        rows = self.priced_rows([self.views, self.records])
        total = 0.0
        for ident in self.cart_ids:
            price = (rows.get(ident) or {}).get("price")
            if isinstance(price, (int, float)) and not isinstance(price, bool):
                qty = self.cart_qty.get(ident)
                total += price * (qty if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 0 else 1)
        return total

    def decline_substitute(self, empty_ok: bool = False) -> list[dict[str, Any]] | None:
        """Cart calls that complete the cart instead of a decline: for each requested product without a line, the nearest
        opened candidate of its kind (a listing an independent check named first, then the last one judged a fit, then
        the cheapest in stock). An empty cart is filled only from the check's names. None when a product has no candidate,
        the total would go over the budget, or nothing is in the cart and no check named a listing (then the decline may
        well be right)."""
        cart, order = self.roles.get("cart"), self.roles.get("order")
        if not cart or not order or self.steps_left() < 2:
            return None
        held_over = bool(self.refusals and self.repick and self.repick not in self.gone)     # the change check named a fit after a refused order
        if not self.cart_ids and not self.audit_fits and not held_over and not empty_ok:
            return None
        rows = self.priced_rows([self.views, self.records])
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        items = self.items()
        if items:
            if self.line_products().get(""):
                return []           # lines of unknown kind: the cart is ordered as it stands
            missing = self.products_without_line()
            quantities = dict(items)
        else:
            missing = [] if self.cart_ids else [str(self.plan.get("product") or "")]
            quantity = self.plan.get("quantity")
            quantities = {missing[0]: quantity if isinstance(quantity, int) and quantity > 0 else 1} if missing else {}

        def closeness(ident: str, name: str) -> float:
            label = self.line_kind(ident) or str((rows.get(ident) or {}).get("category") or "")
            a, b = self.word_stems(name), self.word_stems(label)
            return len(a & b) / max(1, len(a | b)) if label else (1.0 if not items else 0.0)

        def usable(ident: str) -> bool:
            row = rows.get(ident) or {}
            price = row.get("price")
            return (ident not in self.gone and ident not in self.cart_ids and row.get("in_stock") is not False
                    and isinstance(price, (int, float)) and not isinstance(price, bool) and (ident in self.paged or ident in self.opened))

        adds: list[dict[str, Any]] = []
        total = self.held_total()
        chosen: set[str] = set()
        for name in missing:
            def fitting(ident: str) -> bool:
                return usable(ident) and ident not in chosen and closeness(ident, name) > 0 and (
                    not items or closeness(ident, name) >= max(closeness(ident, other) for other, _ in items))
            pool = [i for i in self.audit_fits if fitting(i)]
            if not pool and not self.cart_ids and held_over:
                pool = [self.repick] if usable(self.repick) and self.repick not in chosen else []    # only the checked fit, never a guess
            elif not pool and (self.cart_ids or empty_ok):
                if not items and self.repick and fitting(self.repick):
                    pool = [self.repick]
                else:
                    pool = sorted((i for i in set(self.paged) | set(self.opened) if fitting(i)), key=lambda i: rows[i]["price"])
            if not pool:
                return None
            ident = pool[0]
            units = quantities.get(name) or 1
            add = self.add_action(ident, units)
            if not add:
                return None
            chosen.add(ident)
            total += rows[ident]["price"] * units
            adds.append(add)
        if budget is not None and total > budget + 1e-9:
            return None
        shape = self.basket_shape()
        if adds and shape:
            lines = []
            for add in adds:
                for line in add["args"].get(shape[0]) or []:
                    if isinstance(line, dict) and line.get(shape[1]) in chosen and (line[shape[1]], line[shape[2]]) not in lines:
                        lines.append((line[shape[1]], line[shape[2]]))
            whole = self.basket_action(lines)
            return [whole] if whole else None
        return adds

    def run_substitute(self, adds: list[dict[str, Any]]) -> None:
        """Puts the substitute lines in the cart and orders, as calls made for the model (so the usual bookkeeping and the
        order's own checks apply)."""
        order = self.roles.get("order")
        if adds:
            calls = [{"id": f"fill_{self.step}_{i}", "type": "function", "function": {"name": a["name"], "arguments": json.dumps(a["args"])}}
                     for i, a in enumerate(adds)]
            self.messages.append({"role": "assistant", "content": "Completing the cart with the nearest candidate for each product without a line.", "tool_calls": calls})
            replies = self.run_calls(calls, planned=True)
            for call_id, text in replies:
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if self.finished or self.dead or self.steps_left() <= 0:
                return
        if not order or not self.cart_ids:
            return
        self.order_checked = True       # the alternative is a decline: the model's own pre-order review is skipped, the store checks stay
        call = {"id": f"fill_order_{self.step}", "type": "function", "function": {"name": order, "arguments": json.dumps(self.neutral_args(order, "Order the cart."))}}
        self.messages.append({"role": "assistant", "content": "Ordering the cart: a decline is rarely the right finish.", "tool_calls": [call]})
        replies = self.run_calls([call], planned=True)
        for index, (call_id, text) in enumerate(replies):
            if index == len(replies) - 1 and not self.finished:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if order and not self.closest_wanted() and self.steps_left() >= 2 and (self.cart_ids or self.audit_fits or len(self.items()) >= 2):
            try:
                adds = self.decline_substitute(len(self.items()) >= 2)
            except Exception as exc:
                self.trace.append({"kind": "crash", "where": "substitute", "error": f"{type(exc).__name__}: {exc}"[:300]})
                adds = None
            for add in (adds or [])[: max(0, self.steps_left() - 1)]:
                reply = self.env_call([add])
                self.trace.append({"kind": "forced_fill", "action": add, "reply": reply})
                self.cart_used = True
                if self.finished or self.dead:
                    return
        if order and self.cart_used:
            if self.due_before_order() and self.steps_left() >= 2 and self.time_left() > 12:
                try:
                    self.run_process()
                except Exception as exc:
                    self.trace.append({"kind": "crash", "where": "process", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if self.finished or self.dead or self.steps_left() <= 0:
                    return
            action = {"name": order, "args": self.neutral_args(order, "Order the cart.")}
        elif order and self.closest_wanted() and self.steps_left() >= 2 and self.closest_line():
            self.order_closest()
            return
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
        if (reply and not reply.get("finished") and lines and order and action["name"] == order and self.cart_used
                and _PRICE_WORDS.search(" ".join(lines)) and not _GONE.search(" ".join(lines)) and self.steps_left() > 0):
            # refused only for a changed price: the cart still holds a good item, so it is ordered once more
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})
        if reply and not reply.get("finished") and isinstance(reply.get("observation"), dict) and lines and decline and self.steps_left() > 0:
            action = {"name": decline, "args": self.neutral_args(decline, "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything.", declining=True)}
            reply = self.env_call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def order_closest(self) -> None:
        """Put the nearest opened candidate in the cart and order it (an empty cart near the end, when the request wants
        the nearest candidate rather than a decline)."""
        order = self.roles.get("order")
        ident = self.closest_line()
        add = self.add_action(ident) if ident else None
        if not order or not add:
            return
        reply = self.env_call([add])
        self.trace.append({"kind": "forced_closest", "action": add, "reply": reply})
        self.cart_used = True
        if self.finished or self.dead or self.steps_left() <= 0:
            return
        action = {"name": order, "args": self.neutral_args(order, "Nothing met every requirement; this is the nearest listing found.")}
        reply = self.env_call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": reply})

    def closest_line(self) -> str | None:
        """For a forced finish with an empty cart: the opened listing of the requested kind that was last judged a fit, else
        the cheapest opened one in stock within the budget."""
        if self.repick and self.repick not in self.gone:
            return self.repick
        rows = self.priced_rows([self.views])
        budget = self.budget_now if isinstance(self.budget_now, (int, float)) and not isinstance(self.budget_now, bool) else None
        words = {w.rstrip("s") for w in re.split(r"[^a-z0-9]+", str(self.plan.get("product") or "").lower()) if len(w) >= 3}
        best = None
        for ident, row in rows.items():
            view = self.views.get(ident) or {}
            if ident in self.gone or row.get("in_stock") is False or ident not in self.paged:
                continue
            kind = {w.rstrip("s") for w in re.split(r"[^a-z0-9]+", str(row.get("category") or "").lower()) if len(w) >= 3}
            if words and kind and not (words & kind):
                continue
            if budget is not None and row["price"] > budget:
                continue
            if best is None or row["price"] < best[0]:
                best = (row["price"], ident)
        return best[1] if best else None

    def basket_shape(self) -> tuple[str, str, str] | None:
        """(list argument, id field, quantity field) when the cart tool takes the whole basket as a list of line objects."""
        schema = self.schemas.get(self.roles.get("cart") or "") or {}
        for name, spec in (schema.get("properties") or {}).items():
            items = spec.get("items") if isinstance(spec, dict) and spec.get("type") == "array" else None
            fields = (items or {}).get("properties") if isinstance(items, dict) else None
            if not isinstance(fields, dict):
                continue
            id_key = next((k for k, v in fields.items() if isinstance(v, dict) and v.get("type") == "string"
                           and (_IDENT_KEYS.search(k) or re.search(r"item|sku|listing|product", k, re.I))), None)
            qty_key = next((k for k, v in fields.items() if isinstance(v, dict) and v.get("type") == "integer" and _QTY_KEYS.search(k)), None)
            if id_key and qty_key:
                return name, id_key, qty_key
        return None

    def basket_action(self, lines: list[tuple[str, int]]) -> dict[str, Any] | None:
        """One call that sets the whole basket: the lines already held (with their quantities) plus the given ones."""
        shape = self.basket_shape()
        tool = self.roles.get("cart")
        if not shape or not tool:
            return None
        name, id_key, qty_key = shape
        merged: dict[str, int] = {}
        for ident in self.cart_ids:
            qty = self.cart_qty.get(ident)
            merged[ident] = qty if isinstance(qty, int) and qty > 0 else 1
        for ident, units in lines:
            merged[ident] = units if isinstance(units, int) and units > 0 else 1
        schema = self.schemas.get(tool) or {}
        args: dict[str, Any] = {name: [{id_key: i, qty_key: q} for i, q in merged.items()]}
        for key, spec in (schema.get("properties") or {}).items():
            if key != name and isinstance(spec, dict) and spec.get("enum") and key in (schema.get("required") or []):
                args[key] = next((v for v in spec["enum"] if isinstance(v, str) and re.search(r"add|put|hold|insert|place|include|set", v, re.I)), spec["enum"][0])
        checked, problems = _check(args, schema, "args")
        return None if problems else {"name": tool, "args": checked}

    def add_action(self, ident: str, units: int | None = None) -> dict[str, Any] | None:
        """A cart call that adds one unit (the plan's quantity, or the given units) of a listing, built from the cart tool's schema."""
        tool = self.roles.get("cart")
        schema = self.schemas.get(tool or "") or {}
        properties = schema.get("properties") or {}
        quantity = self.plan.get("quantity") if isinstance(self.plan.get("quantity"), int) and self.plan.get("quantity") > 0 else 1
        if isinstance(units, int) and units > 0:
            quantity = units
        if self.basket_shape():
            return self.basket_action([(ident, quantity)])
        args: dict[str, Any] = {}
        for name, spec in properties.items():
            spec = spec if isinstance(spec, dict) else {}
            if spec.get("enum"):
                if name in (schema.get("required") or []):
                    args[name] = next((v for v in spec["enum"] if isinstance(v, str) and re.search(r"add|put|hold|insert|place|include", v, re.I)), spec["enum"][0])
            elif spec.get("type") == "integer" and _QTY_KEYS.search(name):
                args[name] = quantity
            elif spec.get("type") == "string" and _IDENT_KEYS.search(name) or (spec.get("type") == "string" and re.search(r"item|sku|listing|product", name, re.I)):
                args.setdefault(name, ident)
        checked, problems = _check(args, schema, "args")
        return None if problems or ident not in json.dumps(checked) else {"name": tool, "args": checked}

    def shown(self, name: str, observation: Any) -> Any:
        """A result as the model reads it: a list of search results keeps the start of each row's long text (a page shows
        it in full, and every row stays in the agent's own records); other results only lose very long text."""
        if not isinstance(observation, (dict, list)):
            return observation
        rows = list(_listing_records(observation))
        listing_rows = len(rows) >= 2 and not any(isinstance(v, dict) and len(v) >= 3 for r in rows for v in r.values())
        return _trim(observation, 140 if listing_rows else 700)

    def filter_watch(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """An optional text filter of a listing search that has returned nothing every time it was set (twice or more,
        never anything): the model is told to leave it out."""
        if not isinstance(observation, dict) or not any(isinstance(v, list) for v in observation.values()):
            return ""
        required = set((self.schemas.get(name) or {}).get("required") or [])
        keys = [k for k, v in args.items() if k not in required and isinstance(v, str) and v.strip()
                and not re.search(r"next|page|cursor|token|after|onward|offset|continu|marker", k, re.I)]
        if not keys:
            return ""
        empty = not any(True for _ in _listing_records(observation))
        # an empty result counts against a filter only when its value is one the store's listings really carry (a
        # seller or a category seen on them), so a mistyped value never condemns the filter
        carried = {str(v).strip().lower() for view in self.views.values() for v in view.values() if isinstance(v, str) and len(v) <= 60} if empty else set()
        notes = []
        for key in keys:
            if empty and str(args[key]).strip().lower() not in carried:
                continue
            stat = self.filter_stats.setdefault((name, key), [0, 0])
            stat[0 if empty else 1] += 1
            if empty and stat[0] >= 2 and not stat[1] and (name, key) not in self.filter_warned:
                self.filter_warned.add((name, key))
                notes.append(f"{name} has returned nothing every time {key} was set ({stat[0]} calls): in this store that filter finds nothing, so leave {key} out (search by words or by the other fields).")
        return " ".join(notes)

    def note_observation(self, name: str, observation: Any, declined: bool = False) -> str:
        notes = []
        if isinstance(observation, dict) and "refusal" not in observation:
            inner = next((v for v in observation.values() if isinstance(v, dict) and "refusal" in v), None)
            if inner is not None and name in (self.roles.get("order"), self.roles.get("decline")):
                observation = inner     # the finish result sits inside a wrapper that also carries the shopper's lines
        if isinstance(observation, dict):
            _collect_ids(observation, self.seen_ids)
            problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason") if isinstance(observation.get(k), str))
            core = {self.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
            if observation.get("error") and _USED_UP.search(problem) and name not in core:
                if name == self.roles.get("message"):
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                self.capped.add(name)
                notes.append("That tool's allowance is used up; it is no longer offered. Do not try it again: read prices and stock from search results or the cart view instead, and decide with what you have.")
            if "refusal" in observation and (declined or (name != self.roles.get("order") and self.reads_decline(name))):
                notes.append("THE DECLINE WAS NOT ACCEPTED and the episode continues: the shopper's change in this result came first. Apply it, check again whether something now fits (a raised budget or a dropped requirement often makes one fit), and order it if so; decline again only if nothing fits.")
            elif "refusal" in observation:
                self.refusals += 1
                self.refused_at = self.step
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
        replies = self.run_turn(calls, planned)
        if not replies or self.finished:
            return replies
        if (self.asked and not self.planner_on and not self.ask_plan_done and not self.finished and self.roles.get("search")):
            self.ask_plan_done = True
            self.search_plan_due = self.search_plan_due or "the questions to the shopper are over and the search for listings that meet every requirement starts"
        if self.search_plan_due and not self.finished and not self.dead:
            reason, self.search_plan_due = self.search_plan_due, ""
            try:
                planned_note = self.plan_searches(reason)
            except Exception as exc:      # a plan must never cost the episode
                self.trace.append({"kind": "crash", "where": "plan_searches", "error": f"{type(exc).__name__}: {exc}"[:300]})
                planned_note = ""
            if planned_note:
                self.side_notes.append(planned_note)
        extra = list(self.side_notes)
        self.side_notes = []
        if self.change_due and self.cart_ids:
            try:
                checked = self.check_change()
            except Exception as exc:      # the check must never cost the episode
                self.trace.append({"kind": "crash", "where": "change", "error": f"{type(exc).__name__}: {exc}"[:300]})
                checked = ""
            extra += self.side_notes + ([checked] if checked else [])
            self.side_notes = []
        if extra:
            call_id, text = replies[-1]
            replies[-1] = (call_id, text + "\n[note] " + " ".join(extra))
        return replies

    def run_turn(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        original = calls
        if not planned:
            calls, merged = self.merge_pages(calls)
            replies.extend(merged)
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
            args = self.ids_for_titles(name, args)
            argument = self.id_argument(name)
            # the most this tool takes at once: as learned, else five (a common limit) unless more were seen to work
            cap = self.many_cap.get(name) or (5 if self.many_ok.get(name, 0) <= 5 else None)
            if argument and argument[1] and cap and isinstance(args.get(argument[0]), list) and len(args[argument[0]]) > cap:
                # more listings than this tool takes at once: the first ones are opened, the rest named for the next call
                listed = args[argument[0]]
                fresh = [i for i in listed if i not in self.paged] + [i for i in listed if i in self.paged]
                left_over = [i for i in fresh[cap:] if i not in self.paged]
                args = {**args, argument[0]: fresh[:cap]}
                self.cut_notes[call_id] = (f"Only {cap} of these listings were opened ({name} takes at most {cap} at once)"
                                           + (f"; still to open: {', '.join(map(str, left_over))}." if left_over else "; the others were opened in full before."))
            if name in finishing and len(calls) > 1:
                replies.append((call_id, "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls."))
                continue
            if (name == message_tool and not planned and "?" not in " ".join(str(v) for v in args.values() if isinstance(v, str))
                    and not any(_TELL_RULE.search(str(rule)) for rule in self.plan.get("rules") or [])):
                replies.append((call_id, "Not run (no step used): messages to the shopper are only for questions, and this one asks nothing. Do not report findings, choices or plans to the shopper; the finishing tool carries that."))
                continue
            if name == message_tool and " ".join(str(v) for v in args.values() if isinstance(v, str)).strip().lower() in self.questions_sent:
                replies.append((call_id, "Not run (no step used): this exact message was already sent, and its answer is above. Ask about something else, or carry on."))
                continue
            if name == message_tool and self.message_budget is not None and self.messages_used + planned_messages >= self.message_budget:
                replies.append((call_id, "Not run (no step used): the message budget is used up. Do not message the shopper again."))
                continue
            if (name == message_tool and not planned and self.message_budget is not None and self.kept_messages()
                    and self.messages_used + planned_messages + self.kept_messages() >= self.message_budget):
                replies.append((call_id, "Not run (no step used): the remaining message(s) are kept for what the request demands right before the order (made for you when you call the order tool). Decide with what you have."))
                continue
            signature = name + _compact(args)
            if name not in finishing and name != message_tool and name != cart_tool and signature in self.recent:
                replies.append((call_id, "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else."))
                continue
            if name == cart_tool and self.due_before_cart() and self.steps_left() > 3 and self.time_left() > WRAP_UP_SECONDS:
                # actions the request demands before anything goes into the cart
                try:
                    go, note = self.run_process("cart", _compact(args)[:600])
                except Exception as exc:
                    self.trace.append({"kind": "crash", "where": "cart_process", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    go, note = True, ""
                if not go or self.finished or self.dead:
                    replies.append((call_id, "Not run yet. " + (note or "")))
                    continue
                if note:
                    self.cut_notes[call_id] = (self.cut_notes.get(call_id, "") + " " + note).strip()
            if name == cart_tool and self.steps_left() > 3 and not self.items():      # several products: each line is checked once, at the order
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
            if name == cart_tool and self.gone:
                properties = self.schemas[name].get("properties") or {}
                adding = self.add_signature is None or _compact({k: v for k, v in args.items() if isinstance(properties.get(k), dict) and "enum" in properties[k]}) == self.add_signature
                named = [v for v in args.values() if isinstance(v, str) and v in self.gone] + [x for x in (_listed_ids(args) or []) if x in self.gone]
                if adding and named:
                    replies.append((call_id, f"Not run (no step used): {', '.join(named)} can no longer be bought (the shopper said it is sold out). Choose another listing that meets every requirement."))
                    continue
            if name == cart_tool:
                args = self.fix_cart_number(call_id, name, args)
                warning = self.quantity_warning(name, args)
                if warning and signature not in self.qty_warned:
                    self.qty_warned.add(signature)
                    replies.append((call_id, "Not run (no step used): " + warning + " If you are sure, send the same call again."))
                    continue
            if name == message_tool:
                planned_messages += 1
            runnable.append((call_id, name, args))
        if runnable and runnable[0][1] in finishing:
            call_id, name, args = runnable[0]
            declining = self.is_decline(name, args)
            held_gone = self.sold_out_lines() if not declining and name == self.roles.get("order") else []
            if not declining and name == self.roles.get("order") and held_gone and self.steps_left() >= 3:
                replies.append((call_id, f"Not run (no step used): the cart still holds {', '.join(held_gone)}, which the shopper said can no longer be bought, so this order would fail. "
                                f"Take it out and add " + (f"{self.repick} (the opened listing that meets every current requirement)" if self.repick and self.repick not in self.gone else "the best verified fit")
                                + ", then order."))
                return replies
            if not declining and name == self.roles.get("order") and self.cart_shape_asks < 3 and self.steps_left() >= 3:
                shape = self.cart_shape_problem()
                if shape:
                    self.cart_shape_asks += 1
                    replies.append((call_id, "Not run (no step used): " + shape))
                    return replies
            if (not declining and name == self.roles.get("order") and self.open_queue and self.chain_holds < 3
                    and any(p["value"] is None for p in self.pointers) and self.steps_left() > self.finish_reserve() + 1
                    and self.time_left() > WRAP_UP_SECONDS / 2):
                # a reference is still open and its next page is queued: an order now misses it for sure
                self.chain_holds += 1
                opened = self.open_queued()
                if opened:
                    replies.append((call_id, "Not run yet (no step used by the order). A reference is still open, and the request is not met until its last page is read; the next page was read for you: " + opened))
                    return replies
            # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest"
                    and self.refused_at > self.searched_at and self.research_checks < 1 and self.steps_left() > 10
                    and self.time_left() > WRAP_UP_SECONDS + 20):
                self.research_checks += 1
                replies.append((call_id, "Not run yet (no step used). The last order was refused, and the request asks for the cheapest fitting item. A backup chosen earlier is right only if nothing cheaper fits now, so search again before ordering: the stated requirement words and the category, with a price limit just above the refused item's price if the search offers one, and the next pages. Open the details of cheaper candidates, put the cheapest one that meets every requirement in the cart, then order."))
                return replies
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest" and not self.wider_held and self.cart_ids
                    and self.steps_left() > self.kept_steps() + 12 + 2 * len(self.items()) and self.time_left() > WRAP_UP_SECONDS + 40):
                self.wider_held = True
                rows = self.priced_rows(self.records)
                lines = [f"{i} at {_num(rows[i]['price'])}" for i in self.cart_ids if i in rows]
                self.trace.append({"kind": "wider_hold", "cart": list(self.cart_ids)})
                try:
                    wider_plan = self.plan_searches("the request wants the cheapest fit and the order waits for searches under other names, for rows priced below the cart's lines")
                except Exception as exc:
                    self.trace.append({"kind": "crash", "where": "plan_searches", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    wider_plan = ""
                replies.append((call_id, f"Not run yet (no step used). The request wants the cheapest fit and {self.steps_left()} steps are left. Keyword searches match "
                                "listings' titles and a category listing may show only part of the shelf, so fitting listings under other names are easily missed. "
                                "Before ordering, run two or three more keyword searches for each product under other plain names (its one-word name, other names "
                                "for it, words from the titles of fitting listings), look only at in-stock rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else "")
                                + ", open the cheapest of those, swap in any that meets every requirement, then order." + (" " + wider_plan if wider_plan else "")))
                return replies
            if (not declining and name == self.roles.get("order") and len(self.swept) < 3 and self.steps_left() > 14
                    and self.time_left() > WRAP_UP_SECONDS + 50 and self.sweep_targets()):
                self.sweep_due = True
                replies.append((call_id, "Not run yet (no step used). The request asks for the cheapest fitting item: the store is searched for cheaper listings in the same category first (results follow)."))
                return replies
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") in ("cheapest", "best") and self.pages_read < 6
                    and self.steps_left() > self.kept_steps() + 9 and self.time_left() > WRAP_UP_SECONDS + 30):
                read = self.page_through()
                if read:
                    replies.append((call_id, "Not run yet (no step used). The request asks for the " + str(self.plan.get("priority")) + " fitting item, and the search that found a cart item had more result pages than were read: "
                                    + read + " Check the listings on those pages that may beat your pick (open their property lists), put the right one in the cart, then order."))
                    return replies
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest" and self.cheapest_checks < 2
                    and self.steps_left() > self.finish_reserve() + 2 and self.time_left() > WRAP_UP_SECONDS + 20):
                self.cheapest_checks += 1
                cheaper_note = self.check_cheapest()
                if self.finished or self.dead:
                    return replies
                if "CHANGE CHECK" in cheaper_note:
                    replies.append((call_id, "Not run yet (no step used by the order). " + cheaper_note))
                    return replies
            if not declining and name == self.roles.get("order") and self.cheaper_checks < 2 and self.steps_left() > 10 and self.time_left() > WRAP_UP_SECONDS + 20:
                cheaper = self.unchecked_cheaper()
                if cheaper:
                    self.cheaper_checks += 1
                    replies.append((call_id, "Not run yet (no step used). The request asks for the cheapest fitting item, but these cheaper listings that you have seen in results were never opened with the details tool: " + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Open their details (several in one reply), and if one meets every requirement, put the cheapest such one in the cart instead; if none does, order again."))
                    return replies
            if not declining and name == self.roles.get("order") and not self.order_checked and self.steps_left() > 2 and self.time_left() > WRAP_UP_SECONDS:
                self.order_checked = True
                verdict = self.verify_cart()
                if verdict:
                    replies.append((call_id, "Not run yet (no step used). " + verdict))
                    return replies
                replies.append((call_id, "Not run yet (no step used). Pre-order check: (1) the cart holds exactly the lines the request needs (usually one: the chosen item at the required quantity) and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."))
                return replies
            if not declining and name == self.roles.get("order") and self.links and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS:
                linked = self.derive_links()
                if linked and self.derived:
                    self.order_checked = False
                    replies.append((call_id, "Not run yet (no step used). " + linked + " Make sure the cart meets it, then order."))
                    return replies
            if not declining and name == self.roles.get("order") and self.steps_left() >= 2 and self.time_left() > WRAP_UP_SECONDS / 2:
                try:
                    go, note = self.run_process()
                except Exception as exc:      # the check must never cost the order
                    self.trace.append({"kind": "crash", "where": "process", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    go, note = True, ""
                if not go or self.finished or self.dead or self.steps_left() <= 0:
                    replies.append((call_id, "Not run yet. " + (note or "No steps are left.")))
                    return replies
                gone_now = self.sold_out_lines()
                if gone_now and self.steps_left() >= 3:
                    # the look at the cart lines' pages before the order showed one out of stock
                    replies.append((call_id, f"Not run (no step used): {', '.join(gone_now)} now shows out of stock, so this order would fail. "
                                    "Take it out and add the best verified fit that is in stock, then order." + (" " + note if note else "")))
                    return replies
            if declining and not self.closest_wanted() and self.substitutes < 2 and self.steps_left() >= 2 and self.time_left() > 12:
                pressed = (self.time_left() < WRAP_UP_SECONDS + 20 or self.spent() > COST_WRAP
                           or self.steps_left() - self.kept_steps() <= 7 or self.decline_checks >= 2)
                named = bool(self.audit_fits) and self.decline_checks >= 1
                held_over = bool(self.refusals and self.repick and self.repick not in self.gone and not self.items())
                # several products and an empty cart under pressure: the opened candidates are ordered rather than nothing
                empty_ok = pressed and not self.cart_ids and len(self.items()) >= 2
                if (self.cart_ids and (pressed or self.cart_warned or self.decline_checks >= 1)) or named or held_over or empty_ok:
                    adds = self.decline_substitute(empty_ok)
                    if adds is not None:
                        self.substitutes += 1
                        self.substitute_due = adds
                        self.trace.append({"kind": "substitute", "adds": adds, "cart": list(self.cart_ids), "pressed": pressed, "named": named})
                        replies.append((call_id, "Not run (no step used): a decline is rarely right and this cart (or a listing the check named) can still be ordered. "
                                        "The cart is completed with the nearest candidate for each product without a line and ordered for you."))
                        return replies
            if declining and self.closest_wanted() and (self.closest_asks >= 3 or self.steps_left() < 3 or self.time_left() <= 25):
                # no room left to look further: the cart is ordered as it stands, or the nearest opened candidate is
                # put in and ordered (a decline is never what the request asked for)
                order = self.roles.get("order")
                if self.cart_ids and order:
                    declining = False
                    name, args = order, self.neutral_args(order, "Nothing met every requirement; these are the nearest listings found.")
                    runnable[0] = (call_id, name, args)
                elif self.steps_left() >= 2 and self.closest_line() and order:
                    self.order_closest()
                    replies.append((call_id, "Not run: the nearest candidate was put in the cart and ordered instead." + (" [the episode is finished]" if self.finished else "")))
                    return replies
            if declining and self.closest_wanted() and self.closest_asks < 3 and self.steps_left() >= 3 and self.time_left() > 25:
                self.closest_asks += 1
                order = self.roles.get("order")
                audit = ""
                if self.decline_checks < 2 and self.steps_left() > 5 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.decline_checks += 1
                    audit = self.audit_decline()
                held = ", ".join(self.cart_ids[:6])
                text = ("Not run (no step used). For the case that nothing fits, the request wants the nearest candidate offered, "
                        "so this task ends with an order, not a decline. ")
                if self.cart_ids:
                    text += (f"The cart holds {held}: if each line is the nearest candidate for its product, order now with '{order}' "
                             "(its remark may name the requirements it misses); if a nearer one exists, swap it in first. ")
                else:
                    text += ("Put the nearest candidate for each requested product in the cart (the one missing the fewest requirements, stock and "
                             f"budget included; in stock and within the budget when possible), then order with '{order}', naming in the remark what it misses. ")
                replies.append((call_id, text + audit))
                return replies
            if (declining and self.cart_ids and self.decline_checks < 2 and self.steps_left() > 1 and self.roles.get("order") and not self.cart_warned
                    and self.time_left() > WRAP_UP_SECONDS):
                self.cart_warned = True
                replies.append((call_id, f"Not run yet (no step used). The cart holds {', '.join(self.cart_ids[:6])}, and this call ENDS THE TASK WITHOUT BUYING ANYTHING. "
                                f"To buy what is in the cart, check out with '{self.roles.get('order')}'. Decline only if nothing in the cart (and nothing else in the store) meets every requirement; then send this call again."))
                return replies
            if declining:
                # A decline is checked twice: it is rarely the right finish, so each attempt first gets an
                # independent look for listings that meet the requirements.
                # the task was orderable a moment ago, or no check has been made yet: still worth one, while time allows
                late = (self.refusals > 0 or self.decline_checks == 0) and self.time_left() > 40
                if self.decline_checks < 2 and self.steps_left() > 4 and (self.time_left() > WRAP_UP_SECONDS + 30 or late):
                    self.decline_checks += 1
                    audit = self.audit_decline(late)
                    if audit:
                        replies.append((call_id, "Not run yet (no step used). " + audit))
                        return replies
                    if self.decline_checks == 1:
                        replies.append((call_id, "Not run yet (no step used). Nearly every request in this store has a fitting listing, and a wrong decline scores the same 0 as a wrong order. Before declining, be sure: you searched in several ways (different words, the category, price filters, further pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Close synonyms and any key spelling count; a requirement you only assumed from the story, a reply the shopper did not follow, or an unresolved reference does not count. If something may fit, buy the best verified candidate instead; only if nothing fits, call the decline tool again."))
                        return replies
                args = self.clean_decline(args, name)
                runnable[0] = (call_id, name, args)
        if runnable and runnable[0][1] not in finishing and finishing:
            keep = self.steps_left() - 1 - self.kept_steps()
            if keep < len(runnable):
                for call_id, name, _ in runnable[max(0, keep):]:
                    replies.append((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline)."))
                runnable = runnable[: max(0, keep)]
        for start in range(0, len(runnable), 16):
            chunk = runnable[start : start + 16]
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
                    if not planned and re.search(r"total|basket|cart|order|\d", sent_text):
                        self.message_steps.append(self.step)       # a basket message the model sent on its own
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
                size_note = ""
                if name != message_tool:
                    self.learn_listings(name, args, observation)
                    size_note = self.learn_size(name, args, observation)
                    if size_note:
                        self.requeue_refused(name, args)
                    if name == self.roles.get("search"):
                        self.queue_seller_rows(observation)
                resolved = self.resolve_pointers(observation) if name != message_tool else ""
                if size_note:
                    resolved = (resolved + " " if resolved else "") + size_note
                if call_id in self.cut_notes:
                    resolved = (resolved + " " if resolved else "") + self.cut_notes.pop(call_id)
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
                        for ident, qty in _cart_lines(args) or []:
                            if qty is not None:
                                self.cart_qty[ident] = qty
                    for key, value in (args.items() if listed is None else ()):
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
                    self.links_due = bool(self.links)
                    qty = next((v for k, v in args.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                                and re.search(r"qty|quant|piece|count|unit|amount|number|many", str(k), re.I)), None)
                    if isinstance(observation, dict):
                        held = next((v for k, v in observation.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                                     and re.search(r"qty|quant|piece|count|units", str(k), re.I)), None)
                        qty = held if held is not None else qty
                    for ident in [v for v in args.values() if isinstance(v, str) and v in self.cart_ids]:
                        if qty is not None:
                            self.cart_qty[ident] = qty
                    if qty == 0:
                        self.cart_ids = [i for i in self.cart_ids if self.cart_qty.get(i) != 0]
                if name == self.roles.get("search"):
                    self.searched_at = self.step
                    if not planned:
                        for key, value in args.items():
                            if isinstance(value, str) and value.strip():
                                self.phrase_words[key] = self.phrase_words.get(key, 0) + len(value.split())
                    self.note_search(args, observation)
                    try:
                        self.reach_note(name, args, observation)
                        self.learn_search(name, args, observation)
                        self.probe_words(name, args, observation)
                    except Exception as exc:      # a note must never cost the episode
                        self.trace.append({"kind": "crash", "where": "reach_note", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if name in (self.roles.get("details"), self.roles.get("view")) or name in self.traits:
                    self.opened.update(v for v in args.values() if isinstance(v, str))
                    self.opened.update(x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str))
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                    self.cart_view_at = self.step
                note = self.note_observation(name, observation, self.is_decline(name, args))
                watched = self.filter_watch(name, args, observation)
                if watched:
                    note = (note + " " if note else "") + watched
                if resolved:
                    note = (note + " " if note else "") + resolved
                pending = (isinstance(observation, dict) and _pending_step(observation)) and not (isinstance(observation, dict) and "refusal" in observation)
                declined = self.is_decline(name, args)
                if (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and pending):
                    note = (note + " " if note else "") + ("The order is waiting for a further step (it gives a token or asks for a confirmation): take that step now with the tool that finishes the checkout. Nothing is bought until the episode ends.")
                elif (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and not declined
                        and not (isinstance(observation, dict) and "refusal" in observation)):
                    # the order did not end the episode: refused (in any wording), failed, or waiting for another step
                    self.refusals += 1
                    self.refused_at = self.step
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
                text = _compact(self.shown(name, observation))
                if index == len(observations) - 1 and reply.get("finished"):
                    text += " [the episode is finished]"
                if note:
                    text += "\n[note] " + note
                replies.append((call_id, text))
        order = {c["id"]: i for i, c in enumerate(original)}
        replies.sort(key=lambda r: order.get(r[0], 0))
        if self.links_due and replies and not self.finished:
            self.links_due = False
            linked = self.derive_links()
            if linked:
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[note] " + linked)
        if self.open_queue and replies:
            opened = self.open_queued()
            if opened:
                call_id, text = replies[-1]
                replies[-1] = (call_id, text + "\n[note] " + opened)
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
            _collect_ids(opening, self.seen_ids)
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        idle = 0
        failures = 0
        counter = 0
        over_spend = 0
        for _ in range(MAX_TURNS):
            if self.finished or self.dead or self.steps_left() <= 0:
                break
            if self.time_left() < FORCE_SECONDS:
                self.force_finish()
                break
            if self.spent() > COST_STOP or self.context_tokens > CONTEXT_STOP:
                # a runaway episode could drain the run's shared key: one turn with the finishing tools, then the finish
                over_spend += 1
                if over_spend > 2:
                    self.trace.append({"kind": "spend_stop", "cost": round(self.cost, 4), "estimate": round(self.cost_est, 4), "context": self.context_tokens})
                    self.force_finish()
                    break
            if self.substitute_due is not None:
                adds, self.substitute_due = self.substitute_due, None
                try:
                    self.run_substitute(adds)
                except Exception as exc:
                    self.trace.append({"kind": "crash", "where": "substitute", "error": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            if self.sweep_due and self.run_sweep():
                continue
            self.compact_history()
            message = self.infer(self.messages, self.tools_for_turn(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 4 or self.time_left() < FORCE_SECONDS + 30:
                    self.force_finish()
                    break
                time.sleep(3.0 * failures)
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
                self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
                self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
                if idle >= 3:
                    self.force_finish()
                    break
                continue
            idle = 0
            self.trace.append({"kind": "model", "content": content[:2000], "calls": [c["function"] for c in calls]})
            self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            replies = self.run_calls(calls)
            for index, (call_id, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.status()
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
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
