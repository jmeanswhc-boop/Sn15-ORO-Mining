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

TASK_SECONDS = 265.0
COST_SOFT = 0.05            # dollars spent in one episode after which the checks run on the cheaper model
COST_HARD = 0.085            # beyond this the optional checks are skipped, so one episode cannot drain the run's key
COST_WRAP = 0.075           # dollars spent in one episode after which the agent is told to finish
COST_STOP = 0.09            # dollars spent in one episode after which only the finishing tools remain, then the finish is forced
CONTEXT_WRAP = 80000        # prompt tokens of the main conversation after which the agent is told to finish
CONTEXT_STOP = 95000        # and after which only the finishing tools remain (no passing episode has needed this much)
LOW_EFFORT = "low"          # reasoning effort for the side calls (setup, planning, reading, checks)
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 12000
MAX_TURNS = 80

SYSTEM_PROMPT = """You are a meticulous shopping assistant. You serve one shopper inside a simulated store, and you can act only through the supplied tools. The task is scored 1 or 0. It scores 1 only if your finish is fully right: the ordered item meets every requirement the shopper stated, every hidden need the shopper reveals when asked, and every change the shopper announces while you work; the quantity and the budget are right; the cheapest (or best) choice is made when the request asks for it; and every rule the request states is followed. If nothing in the store can meet everything, the right finish is to decline, exactly the way the request says.

THE WORLD
- Every tool call uses one step, including calls that fail or return errors, and the number of steps is limited. Model thinking is free; steps are not.
- You may put several tool calls in one reply when none depends on another's result (for example, opening the details of several listings at once). They run in order. Never put an order or a decline together with other calls.
- If the store has a tool that shows the full details of several listings in one call, use it instead of opening them one by one: it saves steps.
- Whenever an argument names a listing (to put it in the cart, open it or compare it), give the listing's id exactly as the results show it, never its title or a shortened name, whatever the argument is called.
- Tool names, arguments and the shape of results differ from store to store. Read the tool descriptions and the JSON schemas carefully and use exactly the argument names and allowed values they define. Learn from the first results which tool shows which facts (price, stock, detailed properties).
- Find out early which category labels the store uses for the requested kind of product (they appear on listings). When neighbouring labels exist, the right one is the label that matches the request's wording as a whole; search and buy within it.
- Search tools can be limited: a capped number of visible rows or pages, results sorted by id rather than by relevance or price, every word required, filters that only work in one mode. Compare the reported total with the rows you can actually see, and narrow (more specific words, category or price filters, other pages) until you have seen the whole relevant set.

THE SHOPPER AND HIDDEN NEEDS
- The shopper has needs they did not write down: firm values of the product's own properties, of the same kind as the requirement the request states (the properties the store's listings carry for that kind of product: who or what it is for, its style or type, its material or finish, its colour or pattern, its key specification, a size, length or capacity limit, its condition). They reveal one only when a message asks about that property. The score is 0 unless every hidden need was asked about: an item that happens to meet a need nobody asked about still fails. Each message is answered only about the FIRST thing it asks, so ask about exactly ONE property per message, in the form "Which <property> would you like the <product> to have?" with the property's name as the listings name it, the product named by its plain noun from the request, and no example values. Never give instructions, never mention lists, labels, positions or rules, and never ask two things at once. An open question ("what should I know about it?") is wasted: it names no property.
- The story around the request usually hints at which property matters (who will use it, where it goes, how it will be used, what it must match or fit, what went wrong with an earlier one). Those properties are asked about first.
- A need, stated or revealed, may be given by reference to another listing: the same value as that listing shows, or anything other than it, where the listing is identified as a <product kind> from <seller> carrying a few named property values. To resolve it: search with that seller's name as a keyword (with the product kind), open the listings of that seller that carry the named values, and read the named property on it. The requirement is then that value (the very same) or anything except that value (anything bar). Until it is resolved the status shows it as unresolved; resolve it before choosing, while steps allow. The listing you reach may say in its notes that this property is to be read from yet another listing (of another seller, maybe another kind of product): then that one is the source; the status names it, so search for that seller and open it the same way, as many times as the notes send you on. Every listing on that path must be opened with the details tool.
- When the request asks for several different products, every question must say which product it is about; a question that leaves the product open confuses the shopper. Spread the questions over the products.
- If the shopper says they did not follow a message, or asks you to explain, simplify or rephrase it, do not explain or repeat it: when the request asks for several products this means the property is not one they care about for that product; ask about a different property next (naming the product). A reply asking for the questions singly means your message did not read as one plain question about one property: that property is still open.
- A reply saying they don't mind, have no preference, or anything is fine, means that property is free. A reply that states a preference or a limit is a hard requirement from now on, exactly as worded (numeric limits included).
- The requirements are only what the request states and what the shopper reveals or changes. Never add one of your own from the story (an age group, a gender, a size, who will use it): such details are not requirements, and a listing does not fail because of them. If such a detail may matter, it may be a hidden need; ask about it.
- When a planner is available, your questions are chosen and sent for you as soon as you have opened the details of four or five listings of the requested kind (of each kind, when several different products are requested; one call of a multi-listing details tool when the store has one), so do that in your first moves. The shopper's answers then appear in the conversation with their meaning; apply every revealed need as a hard requirement and search again with it combined with the stated values.
- Messages to the shopper are only for questions (and for answering a question the shopper asked). Never message the shopper to report what you found, what you bought or that you are declining; the finishing tool carries that.
- If you must ask yourself: first open the details of two or three listings, then ask about the properties the story hints at (who it is for, how it must look, where or how it will be used), then about properties the listings carry that differ among candidates. Never ask about something the request already settles, and never exceed the message budget.

RULES OF PROCEDURE
- The request may set rules about how to work: how many listings to look at or compare before choosing, what to check, what to tell the shopper, what to put in a remark, how to decline. These are graded like requirements: follow each one literally and completely. A rule to look at or open N listings means N listings that fully fit, each opened with the details tool.
- If the shopper asks you something (for example to confirm a change), it will be answered for you when messages remain; treat the change itself as binding.
- Besides finding and buying the item, a request may demand actions of its own (for example showing the shopper something, getting their OK, telling them something, opening a page), and a store page may say what else is needed for an item (for example another page to read). The actions due right before the order (after the basket is final) are worked out and carried out for you when you call the order tool, with messages and steps kept for them (the status shows them); a cart change after them means they are done again. Do the other actions yourself when they are due.
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
- Declining is almost never the right finish: in nearly every task the store holds an item that meets every requirement, and a wrong decline scores the same 0 as a wrong order, so when in doubt order the best verified fit. Decline only after a thorough search (several wordings, the category, price filters, further pages), and only when, for every candidate, you can name a stated requirement, a revealed need, the budget or stock that its own records clearly fail. Doubt, a missing nice-to-have, a requirement you only assumed, a reply the shopper did not follow, or a reference you could not resolve is never a reason to decline. Follow the request's rule for declining exactly. If the request says not to offer substitutes, the decline remark must not name, describe or point to any listing: no ids, no titles, no prices.
- Plan your steps: get your first order attempt in while at least seven steps remain, so a refusal can still be handled. When 7 steps remain, only cart, stock-check and finishing tools stay available. Decide once: do not swap the cart back and forth, since every cart change costs a step.
- Every reply must contain at least one tool call. Keep the reply text short: your current requirements, the verified fits with price and stock, and your next move."""

SETUP_PROMPT = """Read the shopper's request and the store's tool list below. Return ONE compact JSON object (no indentation) and nothing else, with these keys:

"product": the kind of product requested, in the request's words (when the request asks for several different items, name each of them).
"quantity": the number of units of one listing to order (an integer; 1 when the request does not ask for several). Something naturally sold as one item, such as a pair of shoes or a set, is quantity 1.
"items": when the request asks for several DIFFERENT products, one object per product: {"product": a short name for it in the request's words, "quantity": the units of that product (integer)}; an empty list when only one kind of product is requested. A name joined with "and" or "&" that the request uses as one kind of product (the way stores name a department, such as "<A> and <B>" or "<A> & <B>" with one quantity and one set of requirements) is ONE product, not two.
"requirements": a list of strings, one per firm stated requirement on the product, each written as "property: constraint" with the shopper's exact value, limit, exclusion or closed list. When several different products are requested, start each one with the short product name it applies to, as "<product> <property>: constraint". Do not include price, stock, quantity, budgets or mere wishes here, and do not restate the product kind or category as a requirement: words that are part of the category's name (a season, an occasion or an audience in the category name) are covered by the category and are not a separate property requirement. A figure of speech, a simile, or a description of how the shopper feels about the product is story, not a requirement: a requirement names a property of the product and the value it must have.
"price_cap": the maximum total price as a number in the unit the request writes it in (a cap written in cents stays in cents: 75¢ is 75, never 0.75), or null when none is stated.
"currency": the price unit the request uses, or null.
"in_stock_only": true when the request wants only items actually on hand or in stock.
"priority": "cheapest" only when the request explicitly asks for the cheapest or lowest-priced fitting option; "best" only when it explicitly asks for the best by some measure; otherwise "none". Remarks that money is tight or that every penny counts are not a request for the cheapest; they only explain the price cap.
"priority_detail": for "best", what "best" is measured by; otherwise "".
"message_budget": the number of messages the shopper allows you to send, as an integer, or null when not stated.
"step_budget": the number of moves or steps the request allows, as an integer, or null when not stated.
"process": every action the request demands besides finding the item, putting it in the cart and ordering it (for example showing, telling or asking the shopper something, getting their OK, opening or comparing pages), in the order it must happen, one object each: {"do": the action in plain words, "when": "before_order" when it must happen once the basket is final and before the order, "before_choice" when it must happen before choosing, else "any", "messages": how many messages to the shopper it takes (0 when none), "steps": how many tool calls it takes}. Limits on messages or steps are not actions. Empty when there is none.
"rules": a list of every procedural rule the request states for the assistant (how many listings to look at, what to check or tell, how to decline, what not to do, what to include), quoted word for word. Remarks about what the shopper will do themselves are story, not rules.
"soft_wants": a list of things the shopper would like but does not strictly require (phrased as ideally, preferably, would be nice, if possible). Keep these out of "requirements".
"ask_first": a list of things the agent can only learn from the shopper and should ask about before choosing: for example the budget when no price limit is stated, or which of two wishes matters more when the request says both may not be possible (then also put both wishes in soft_wants, not in requirements). An open invitation to put questions to the shopper, or to check with them, is NOT an item here: it only says that unwritten needs exist. Empty when none.
"hints": the story sentences that point at an unstated property of the product (who will use it, how or where it will be used or kept, what it must match, fit or be told apart from, how long it must last, what went wrong with an earlier one, what the shopper says they cannot judge themselves), one object each: {"product": the product it is about, "clue": that sentence, shortened, "properties": two or three plain names of the product's own classifying properties it may point at, most likely first}. A classifying property is one a listing states as a short value of its own: who it is for (gender, age group), its type or style, material, finish, colour, pattern, closure, height, condition, operating system, processor, memory, storage, capacity, length, flavour, supply. Never name a size offered per listing, a brand, a price, delivery, packaging, or an abstract quality (durability, comfort, quality, fit). A sentence about who will use it points at gender first; about telling two items apart at colour or flavour; about how long it lasts or when it is restocked at a supply or capacity property; about where it goes or how it travels at form factor, length or type; about what the shopper cannot judge at the key technical property. When several different products are requested, a sentence that does not name its product may be about any of them: give one object for each product it could fit, with that product's own properties. Leave out anything already stated as a requirement. Empty when the request is all business.
"pointers": requirements the request states by reference to another listing (the same value as that listing shows, or anything other than it; the listing identified as a <product kind> from <seller> carrying <property> <value>, ...; the values that identify that other listing are not requirements on the product and never go into "requirements" on their own): one object each: {"product": the product it is about, "property": the property whose value is to be read off that listing, "relation": "same" or "different", "seller": the seller's name as written, "category": the product kind of that listing as written, "attributes": {property: value, ...} as written, "pick": when several such listings exist, the rule for choosing the one meant, else ""}. Also put it in "requirements" as "<property>: the same as (or: anything except) the <property> on <seller>'s <category> listing (<attributes>)". Empty when none.
Every sentence of the request must land somewhere: a requirement, the priority, a soft want, a rule, a budget, a hint, or pure story. When unsure whether a sentence is a rule, put it in "rules".
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
"unavailable": true when the shopper says an item can no longer be bought (sold out, gone, out of stock, button disabled), else false.
"requirements_now": the complete list of firm requirements on the product as they stand after everything the shopper has said so far, built from the requirements so far (given below) and what the shopper has explicitly said, each as "property: constraint". Never add a requirement from the story or from your own reading of the request: only values the request states as requirements or the shopper states in a reply count. Keep every earlier requirement unless the shopper dropped or replaced it; for a replaced value keep only the newest one; add every newly revealed need. Do not include price, stock, quantity or the product category.
"unclear": true when the shopper says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false. When the request asks for several products, such a reply usually means the property asked about is not one the shopper cares about for that product.
"singly": true when the shopper asks to be given the questions singly, one per message, rather than saying they did not follow; else false.
"pointer": when the shopper gives a value by reference to another listing (the same value as that listing shows, or anything other than it; the listing identified as a <product kind> from <seller> carrying <property> <value>...), an object {"product": the product it is about or "", "property": the property whose value is to be read off that listing, named as the shopper names it in this reply (not as the assistant's question named it), "relation": "same" or "different", "seller": the seller's name as written, "category": the product kind of that listing as written, "attributes": {property: value, ...} as written}; otherwise null. Such a need goes into "needs" and "requirements_now" as ONE entry, "<property>: the same as (or: anything except) the <property> on <seller>'s <category> listing (<attributes>)". The values that identify that other listing are not requirements on the product the shopper is buying: never list them as separate needs.
"reply": when the shopper asks the assistant something or asks for a confirmation, a short, plain reply to send back that answers it and names the change in substance (for example confirming the new value); otherwise "". Always "" when "unclear" is true: a request to explain the assistant's own question is not a question to answer.
When the request asks for several different products, start every requirement in "needs" and "requirements_now" with the short name of the product it applies to, as "<product> <property>: constraint" (the shopper often says "<property> of the <product kind>").
One line can carry several changes: report every one. A requirement replaced by another is both dropped (the old one) and a new need (the new one). When the shopper says which of two wishes matters more, the preferred wish is a need. When the shopper states a budget, it is new_budget. A mere wish ("ideally", "would be nice") is not a need. Read the shopper literally and charitably. Folksy, figurative or roundabout phrasing that names a value means the shopper firmly wants that value; treat a value as excluded only when the shopper clearly says to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit, kept with its unit. Comparison words (at least, short of, below, no less than, or better) are limits only for numbers: with a value that is a word or a name (an interface, a type, a material, a pack type, a connector), the shopper wants exactly that value, so write it as that value alone, never "that or better". A new budget replaces the old one whether it is higher or lower, even when the shopper calls it good news. When the shopper only says they don't mind, have no preference, or anything works, put the property asked about in no_preference; such a reply applies only to the property the assistant asked about, never drops a stated requirement, a price limit or a rule, and when the assistant's message did not ask about a property it changes nothing."""

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
- A requirement marked UNRESOLVED (a value that was to be read off another seller's listing and never was) cannot be judged: never fail an item for it.
When records disagree (for example on price or stock), the newest record counts.
Return ONE JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, each naming a requirement that is not met or not shown]}. Use fits=false only for a requirement that the records clearly fail, or that no record addresses at all; do not fail an item for a missing record of stock or price."""

PLAN_PROMPT = """You plan the questions a shopping assistant sends to a shopper. The shopper has hidden needs: firm values of the product's own properties, of the same kind as the requirement the request already states (the properties the store's listings carry for that kind of product), revealed only when a message asks about that exact property. The task scores 0 unless every hidden need is asked about, so the choice of properties decides the score. Each message is answered only about the first thing it asks, and questions about anything else get a polite "don't mind". Choose QUESTION_COUNT different properties (fewer only when fewer useful ones exist), most promising first:
0. first, anything listed below as MUST BE LEARNED FROM THE SHOPPER that has not been asked yet; ask it plainly. For the budget, ask how much they want to spend at most. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. then the properties the STORY HINTS point at, each mapped to the name the listings use for it (a hint about who will use it points at a gender or age property; where it goes or how it will carried points at a form factor, size or length; what it must match points at a colour, finish or material; what went wrong before points at the specification or feature that failed);
2. then the classifying properties of this kind of product, the ones a shopper states in one line of a request just as the stated requirement does: who or what it is for (gender first; an age group only as a separate question when messages remain), its type or style, material or finish, colour or pattern, closure or height, condition, the key specification (processor, memory, storage, operating system, connector), a measured length or capacity of the product itself, its flavour or supply. Peripheral details (certifications, voltages, ports, adapters, weights, years, warranties, packaging, cushioning, decoration) come last. The menu below is drawn from the few listings opened so far and lists the properties in the order of how many carry them: keep that order among properties of the same kind, but a classifying property carried by only one or two opened listings is still a better question than a peripheral one carried by all. A property with one value in every listing seen may still be a need.
A size offered per listing (shoe, clothing or ring sizes, a pack size), a weight, a count, a model name or an identifier is an option or a detail of the listing, never a hidden need: never ask about them.
Never pick a property the request already settles (its stated requirements, the price, stock, quantity), never a listing's metadata (brand, maker, seller, title, options list, identifiers), never repeat a property already asked for the same product (a question the shopper did not follow counts as asked, unless that question did not name the product; a reply asking for the questions singly does not count: ask that property again), never spend two questions on near-duplicate properties (two spellings of one property, a property and its "filter" twin), and avoid properties no listing carries. Use the plain property name as the listings name it (without prefixes, suffixes or underscores).
When the request asks for several different products, each question is about ONE of them and names it, and the questions are spread over the products: give every product at least one question (its most promising property, story hints first), then spend the rest on the products with the most plausible open properties, preferring products that have had fewer questions so far; take each product's properties from the listings of its own kind (the menu below is grouped by the listings' labels when they carry one).
Write each question as "Which <property> would you like the <product> to have?", with the property's name exactly as the listings name it (spaces instead of underscores or dots) and the product named by its plain noun from the request ("the rug", "the curtains"; never with adjectives, materials, colours or other property values). No example values, no other words, no second question. For something that must be learned first (rule 0), ask it plainly in one short sentence.
Return ONE JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

LATER_ROUND_NOTE = """

THIS IS A LATER ROUND: the answers so far are listed under ALREADY ASKED. A reply that says the shopper did not follow, does not mind or has no preference means that property is not a hidden need for that product: never ask it again. A product rarely has more than one hidden need: when several products are requested, put these questions on the products whose need has NOT been revealed yet (their next most promising properties), and ask about a product whose need is already known only when no other product is left; for a single product, ask its next most promising properties. A property whose question got a reply asking for the questions singly is still open: ask it again with the property's exact name from the listings."""

SWEEP_LAST = False
SWEEP_NOTE = """

THE LAST QUESTION IN YOUR LIST IS A SWEEP: write it as "Do you have a requirement for the <product>'s <property>, <property> or <property>?", naming two or three further promising properties of ONE product that no other question covers (names as the listings name them, product by its plain noun). Its "property" field lists them joined by ", "."""

REDIRECT_PROMPT = """A shopping assistant must read the PROPERTY_NAME of the listing below. Some listings say, usually in a note, that a property is to be read from a different listing instead (a note naming the property and pointing the reader to another seller's listing of some product, rather than to this one). Return ONE JSON object and nothing else: {"redirect": true when the listing says that its PROPERTY_NAME (or every property) is to be read from another listing, else false, "seller": that other listing's seller or brand name as written, or "", "category": that other listing's kind of product as written, or "", "description": the other words that pick out that listing (values such as colour, type, sizes, specifications) as written, or "", "value": when there is no such redirect, the value this listing itself states for the PROPERTY_NAME (its own property field first, else its notes or description), or null}. A sentence that only says what this item is not, or that sends the reader elsewhere for a DIFFERENT property, is not a redirect for the PROPERTY_NAME."""

PROCESS_PROMPT = """A shopping assistant is about to place the order for the cart below. The task scores 0 unless every action the request demands has been done the way and at the time it demands, so check the record before the order goes in. Return ONE compact JSON object (no indentation) and nothing else: {"missing": [each demanded action not yet done for this very basket, in a few words], "actions": [the tool calls to make now, in order, at most 3, each {"tool": exact tool name, "args": {...}}], "ready": true when nothing is missing}.
How to judge:
- An action demanded right before the order (once the basket is final) counts only when it happened after the last cart change; one made earlier must be made again. An action demanded at another time counts when it was done at that time.
- The demands come from the request's own words, from the shopper's messages (including lines that arrive inside tool results), and from the pages of the items in the cart (a page may say what else is needed, for example another page to read: open the page it points to, searching for its code or name first when its id is not known).
- A message to the shopper is ONE plain message that does exactly what was asked in the shopper's own terms, with one question at most. When it puts the basket to the shopper, list each line (quantity, name, price) and the total, and ask for their OK to order it. Never ask about preferences here, never repeat an action already done for this basket, and never use the cart, order or decline tools.
- Use no more messages or steps than are left; when nothing is missing or nothing more can be done, return "ready": true with no actions.
"""

AUDIT_PROMPT = """A shopping assistant is about to tell the shopper that nothing in the store fits. Check the listing records below against the requirements, using these reading rules: a requirement is met when the listing's own property for it states the value (any key spelling, one element of a comma list, singular/plural and close synonyms all count; a broader or different class does not); the title or description counts only when the listing has no key for that property; numbers are compared after converting units, limits include the edge; the price must be within the budget and the item must not be shown out of stock; only the listed requirements count, never one assumed from the story, and a requirement marked UNRESOLVED cannot fail anything; when several different products are requested, a listing is checked only against the requirements of its own kind of product. A wrong decline scores 0 and nearly every request has a fitting listing, so list every listing that may fit; leave one out only when its own records clearly fail a listed requirement.
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
            ident = next((v for k, v in line.items() if id_like(k, v)), None)
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
                           r"blurb|tagline|caption|teaser|slogan|name|overview|summary|description)\b|"
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
# an answer that is more than a plain acknowledgement
_NOT_ACK = re.compile(r"\d|¢|\$|€|£|\?|\b(no|not|never|instead|rather|only|without|but)\b|n't|longer|scrap|drop|swap|remov|replac|chang|cheap|budget|price|cost|must|need|want", re.I)
_TELL_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
_FINISH_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
_PENDING_KEYS = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
_DECLINE_WORDS = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order|sale)|reject|cancel|quit|skip|abstain|withdraw|stand.?down|forgo|walk.?away", re.I)
_NO_PURCHASE = re.compile(r"no purchase|without (a |any )?(buying|purchas|order)|(with )?nothing (is |gets )?(bought|purchased|ordered)|not (buy|purchas|order)|no order|empty.handed", re.I)
_FILLER = re.compile(r"^(the|an?|and|or|with|of|in|on|to|for|from|by|its?|this|that|which|has|have|is|are|not|one|s|items?|listings?|products?|made|piece)$")
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
        self.sweep_due = False
        self.swept: set[str] = set()
        self.plan_rest: list[str] = []
        self.redirects: dict[str, dict[str, Any]] = {}
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
        return min(2, sum(p.get("messages") or 0 for p in self.process if p.get("when") == "before_order"))

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
                        if (isinstance(v, (int, float)) and not isinstance(v, bool) and price is None
                                and re.search(r"price|cost|amount", str(k), re.I) and not re.search(r"total|sum|was|old|list|ship|deliver", str(k), re.I)):
                            price = v
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

    def cart_text(self) -> str:
        lines = []
        for ident in self.cart_ids[:8]:
            title, price = self.listing_facts(ident)
            qty = self.cart_qty.get(ident)
            qty = qty if isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 0 else 1
            page = next((r for r in reversed(self.records) if isinstance(r, dict) and ident in r.values()
                         and any(isinstance(v, (str, dict)) and len(_compact(v)) > 40 for v in r.values())), None)
            text = _compact({k: v for k, v in page.items() if k not in ("title",)})[:900] if page else ""
            lines.append(f"- {ident}: {qty:g} x {title or '?'} at {price if price is not None else '?'} each; its page: {text}")
        return "\n".join(lines) or "(empty)"

    def action_log(self) -> str:
        out = []
        busy = {self.roles.get(r) for r in ("message", "cart", "order", "decline")} - {None}
        for entry in self.trace:
            if entry.get("kind") not in ("env", "process"):
                continue
            reply = entry.get("reply") or {}
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, action in enumerate(entry.get("actions") or []):
                result = observations[index] if index < len(observations) else None
                name = action.get("name")
                out.append(f"{name} {_compact(action.get('args'))[:200]} -> {_compact(result)[:300 if name in busy else 120]}")
        return "\n".join(out[-40:])

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

    def run_process(self) -> tuple[bool, str]:
        """Before an order: the model reads the request, the conversation and the cart's pages, says which demanded
        actions are still missing for this basket and makes them. Returns whether the order can go ahead, and a note."""
        allowed = {self.roles.get(r) for r in ("message", "details", "view", "search", "cart_view")} - {None}
        if not self.process:
            allowed.discard(self.roles.get("message"))      # nothing demanded: pages may still be read, the shopper is not messaged
        tools = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | arguments: {_compact(t['function'].get('parameters') or {})}"
                          for t in self.tools if t["function"]["name"] in allowed)
        notes: list[str] = []
        acted = False
        for _ in range(3):
            key = self.cart_changed_at
            if self.process_checks.get(key, 0) >= 3 or self.steps_left() < 2 or self.time_left() < WRAP_UP_SECONDS / 2:
                break
            self.process_checks[key] = self.process_checks.get(key, 0) + 1
            left = self.messages_left() if self.message_budget is not None else "no limit on"
            if not self.process:
                left = 0
            prompt = (PROCESS_PROMPT + "\n\nREQUEST:\n" + str(self.policy.get("query") or "")
                      + "\n\nACTIONS THE REQUEST DEMANDS (as read from it):\n" + (_compact(self.process) if self.process else "(none listed)")
                      + "\n\nRULES QUOTED FROM THE REQUEST:\n" + ("\n".join(str(r) for r in self.plan.get("rules") or []) or "(none)")
                      + "\n\nREQUIREMENTS NOW:\n" + self.ledger()
                      + "\n\nTOOLS YOU MAY USE NOW:\n" + tools
                      + "\n\nTHE CART (the final basket), WITH EACH ITEM'S PAGE:\n" + self.cart_text()
                      + f"\n\nEVERYTHING DONE SO FAR, oldest first (the last cart change was at step {self.cart_changed_at}; this is step {self.step}):\n" + self.action_log()
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
            actions = actions[: max(0, min(3, self.steps_left() - 1))]
            if not actions:
                break
            reply = self.env_call(actions)
            self.trace.append({"kind": "process", "actions": actions, "reply": reply, "missing": result.get("missing")})
            if reply is None:
                break
            acted = True
            self.recent = []
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            changed = looked = False
            for index, action in enumerate(actions):
                observation = observations[index] if index < len(observations) else None
                self.records.append(observation)
                if isinstance(observation, (dict, list)):
                    _collect_ids(observation, self.seen_ids)
                name, args = action["name"], action["args"]
                text = " ".join(str(v) for v in args.values() if isinstance(v, str))
                failed = isinstance(observation, dict) and bool(observation.get("error"))
                if name == self.roles.get("message"):
                    if failed:
                        notes.append(f"A message to the shopper failed: {_compact(observation)[:200]}.")
                        continue
                    self.messages_used += 1
                    self.questions_sent.append(text.strip().lower())
                    said, pending = self.split_reply(observation, False)
                    if not said and not pending and _has_text(observation):
                        said = [_compact(observation)[:600]]
                    answer = " ".join(said + pending)
                    notes.append(f"Sent to the shopper before the order: \"{text[:300]}\"; answer: \"{answer[:400]}\".")
                    if pending or (said and _NOT_ACK.search(" ".join(said))):
                        meaning = self.interpret(said + pending, text)
                        if meaning:
                            notes.append(meaning)
                        changed = True
                    continue
                looked = True
                if name in (self.roles.get("details"), self.roles.get("view")):
                    self.opened.update(v for v in args.values() if isinstance(v, str))
                notes.append(f"Before the order, {name} {_compact(args)[:120]} was run: {_compact(observation)[:400]}")
                _, pending = self.split_reply(observation, True)
                if pending:
                    meaning = self.interpret(pending, None)
                    notes.append("A shopper line came with that result: " + " | ".join(pending[:3]) + (" " + meaning if meaning else ""))
                    changed = True
            if self.finished or self.dead:
                return False, " ".join(notes)
            if changed:
                self.process_at = self.step
                return False, " ".join(notes) + " Apply what the shopper said (a cart change means the actions due before the order are made again at the next order call); if nothing needs to change, call the order tool again now."
            if not looked:
                break
        if acted:
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
            fresh = prompt * (0.6 if main else 1.0)     # a cached conversation is usually billed for about 0.6 of it
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
        message = self.infer([{"role": "user", "content": content}], None, [SETUP_MODEL, BACKUP_MODEL], 2500, effort=LOW_EFFORT)
        plan = _parse_json_object((message or {}).get("content") or "") or {}
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
        if self.refusals and not self.finished and self.step - self.refused_at >= 2:
            # a refused order is often misread as a placed one; until the episode ends nothing has been bought
            text += (f" NOTHING HAS BEEN BOUGHT YET: the order attempt {self.step - self.refused_at} steps ago did not end the episode."
                     " If the cart holds fitting items within the current budget, call the order tool now; if not, fix the cart first, then order"
                     " (decline only if nothing in the store fits any more).")
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
            text += " FINISH NOW: fix the cart to the verified line(s) the request needs, re-check stock, then order; decline only if nothing seen could fit."
        elif self.order_attempts == 0 and left <= 14:
            text += " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def ledger(self) -> str:
        if self.current is not None:
            items = list(self.current)
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
        return text

    def judge(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None, optional: bool = False) -> dict[str, Any] | None:
        models = models or [JUDGE_MODEL, MAIN_MODEL]
        if self.spent() > COST_SOFT and models[0] in (JUDGE_MODEL, PLAN_MODEL):
            # an expensive episode keeps its checks but makes them on the cheaper model, so one task cannot eat the key
            models = [READ_MODEL, JUDGE_MODEL]
        if optional and self.spent() > COST_HARD:
            return None
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
        elif reply and self.roles.get("message") and _ASKS.search(" ".join(fresh)):
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
        result = self.judge(prompt, 1200, optional=True)
        self.trace.append({"kind": "verify", "item": item, "result": result})
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
                   "relation": "same" if str(raw.get("relation") or "").strip().lower().startswith("s") else "different",
                   "seller": seller, "category": str(raw.get("category") or "").strip(),
                   "attributes": {str(k).strip(): str(v).strip() for k, v in attributes.items() if str(k).strip() and str(v).strip()},
                   "pick": str(raw.get("pick") or "").strip(), "origin": origin, "value": None, "listing": None,
                   "describe": "", "hops": [], "sellers": [seller.lower()], "score": None, "firm": False, "line": ""}
        self.pointers.append(pointer)
        self.trace.append({"kind": "pointer", "pointer": dict(pointer)})
        return pointer

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
        if pointer["value"] is not None:
            want = pointer["value"] if pointer["relation"] == "same" else "anything except " + pointer["value"]
            return f"{head}: {want} (read off {pointer['seller']}'s listing {pointer['listing']})"
        attrs = ", ".join(f"{k} {v}" for k, v in pointer["attributes"].items())
        relation = "the same as" if pointer["relation"] == "same" else "anything except"
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
        message = self.infer([{"role": "user", "content": prompt}], None, [READ_MODEL, BACKUP_MODEL], 300, effort=LOW_EFFORT)
        result = _parse_json_object((message or {}).get("content") or "") or {}
        self.redirects[key] = result
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
            matched = missing = clashing = 0
            for name, wanted in pointer["attributes"].items():
                words, value = key_words(name), squash(wanted)
                keyed = [v for table in flat for k, v in table.items() if same_property(k, words)]
                if not keyed:
                    missing += 1
                elif any(value and (value in squash(v) or squash(v) in value) for v in keyed):
                    matched += 1
                else:
                    # a value written another way (1000 gb against 1tb) looks like a clash: it lowers the rank, and only a
                    # page with more clashes than matches is ruled out
                    clashing += 1
            if pointer["attributes"] and (not matched or clashing > matched):
                return None
            # words that pick the listing out count first in its own property values, then anywhere in its text (a note
            # may name a value only to say the item lacks it)
            values = " ".join(str(x).lower() for v in record.values() if isinstance(v, dict) for x in v.values() if isinstance(x, str))
            text = " ".join(v.lower() for table in flat for v in table.values())
            kind = {w for w in key_words(pointer["category"]) if not _FILLER.match(w)}
            described = tokens(pointer["describe"], kind, seller)
            in_values = sum(1 for w in described if has(w, values))
            anywhere = sum(1 for w in described if has(w, text))
            if described and anywhere * 2 < len(described):
                return None
            same_kind = 1 if kind and all(re.search(r"\b" + re.escape(w[:5]), text) for w in kind) else 0
            full = missing == 0 and clashing == 0 and in_values == len(described)
            return (1 if full else 0, matched + in_values + same_kind - clashing, anywhere, -missing - clashing)

        records = list(listings(observation))
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
                    continue
                rank, record, flat, ident = best
                if pointer["value"] is not None and pointer["score"] is not None and rank <= pointer["score"]:
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
                    notes.append(f"REFERENCE LEADS ON: {previous}'s listing {ident} says its {pointer['property']} is to be read from {onward}'s "
                                 f"{pointer['category'] or 'listing'}" + (f" ({pointer['describe']})" if pointer["describe"] else "")
                                 + f", not from {ident}. Search for \"{onward}\" with that product kind, open that listing with the details tool "
                                 f"and read its {pointer['property']} there (its notes may send you on again).")
                    self.trace.append({"kind": "pointer_hop", "pointer": dict(pointer)})
                    progressed = True
                    continue
                wanted = key_words(pointer["property"])
                found = [(k, v.strip()) for table in flat for k, v in table.items() if same_property(k, wanted) and v.strip()]
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

        for record in self.records:
            walk(record, 0)

        def key_words(name: str) -> set[str]:
            return {w for w in re.split(r"[^a-z0-9]+", name.lower()) if w}

        def lines_for(group: str, limit: int) -> list[str]:
            c, d, s = dict(counts.get(group, {})), {k: set(v) for k, v in distinct.get(group, {}).items()}, {k: list(v) for k, v in samples.get(group, {}).items()}
            # a key whose words contain all the words of a shorter key is the same property under a longer name
            # (a "filter" twin, a prefixed copy): it is folded into the shorter name, which is the one to ask by
            also: dict[str, list[str]] = {}
            for longer in sorted(c, key=lambda k: (-len(key_words(k)), k)):
                lw = key_words(longer)
                shorter = next((k for k in sorted(c, key=lambda k: len(key_words(k))) if k != longer and k in c and key_words(k) and key_words(k) < lw), None)
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
            # the most widely carried properties first (a hidden need is a value of a property the kind of product
            # commonly has); among equals the ones whose values differ between listings
            ranked = sorted(c, key=lambda k: (-c[k], -len(d.get(k, ())), k))[:limit]
            return [f"- {k} (in {c[k]} listings, {len(d.get(k, ()))} different values" + (f"; also written as {', '.join(also[k][:2])}" if k in also else "") + "): " + " | ".join(s.get(k, [])) for k in ranked]

        groups = [g for g in sorted(counts, key=lambda g: -tables.get(g, 0)) if g and tables.get(g, 0) >= 2]
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

    def ask_planned(self, force: bool = False) -> bool:
        tool = self.roles.get("message")
        limit = self.question_limit()
        if not tool or self.messages_used >= limit:
            self.planner_on = False
            return False
        menu, fact_sets = self.property_menu()
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        if fact_sets < 5 and self.step < 8 and not urgent and not force:
            return False
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
        if self.steps_left() <= (6 if force else 10) or self.time_left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        # every remaining question is planned at once (for several products spread over them), so the questions do
        # not eat into the steps needed for the search and the stronger planner sees the whole menu once
        count = max(1, limit - self.messages_used)
        send = count
        if not self.asked and not urgent and count >= 3:
            # the whole list is planned by the stronger model, but only the first round is sent: the most promising
            # property of each product (three for a single product); the rest follow right after, re-aimed with the
            # answers in view (a product whose need was revealed rarely has another)
            send = min(count, max(2, len(self.items())) if self.items() else 3)
        prompt = PLAN_PROMPT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE SHOPPER'S REQUEST:\n" + str(self.policy.get("query") or "")
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
        prompt += "\n\nREQUIREMENTS SO FAR:\n" + self.ledger()
        items = self.items()
        if items:
            prompt += "\n\nTHE REQUEST ASKS FOR SEVERAL DIFFERENT PRODUCTS (name the one each question is about): " + "; ".join(name for name, _ in items)
        if self.asked:
            prompt += "\n\nALREADY ASKED (with what the shopper said):\n" + "\n".join(self.asked)
        if self.plan.get("ask_first"):
            prompt += "\n\nMUST BE LEARNED FROM THE SHOPPER: " + "; ".join(str(h) for h in self.plan.get("ask_first") or [])
        if self.hints:
            prompt += "\n\nSTORY HINTS (ask about these first, each mapped to the listings' name for that property):\n" + "\n".join(
                f"- about the {h.get('product') or self.plan.get('product')}: \"{h.get('clue')}\" -> probably: " + ", ".join(str(p) for p in h.get("properties") or [] if p)
                for h in self.hints)
        prompt += "\n\nPROPERTIES IN THE LISTINGS' DETAILS SEEN SO FAR (name, number of listings carrying it, number of different values, example values; most widely carried first):\n" + (menu or "(none seen yet)")
        # the first round is planned by the stronger model; later rounds mostly re-aim the remaining questions with the
        # answers in view, which the cheaper model does at a twentieth of the price
        models = [PLAN_MODEL, READ_MODEL] if self.messages_used == 0 and self.spent() < COST_SOFT else [READ_MODEL, JUDGE_MODEL]
        result = self.judge(prompt, 2000, models)
        planned = [q for q in (result or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]
        if urgent and fact_sets < 4:
            planned = planned[:1]
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
        for key, spec in properties.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
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
                prices = [(k, v) for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                          and re.search(r"price|cost|amount", str(k), re.I) and not re.search(r"total|sum|was|old|list|ship|deliver", str(k), re.I)]
                if ident and prices:
                    row = rows.setdefault(ident, {})
                    row["price"] = sorted(prices, key=lambda kv: (0 if "price" in str(kv[0]).lower() else 1))[0][1]
                    for k, v in value.items():
                        if isinstance(v, bool) and re.search(r"stock|can.?buy|available|buyable|orderable|purchas", str(k), re.I):
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
        result = self.judge(prompt, 1200, optional=True)
        self.trace.append({"kind": "audit", "result": result})
        fits = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        if not fits:
            return ""
        text = "An independent check thinks these listings may meet every requirement: " + ", ".join(fits) + (f" ({result.get('reason')})" if result.get("reason") else "") + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."
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

    def force_finish(self) -> None:
        if self.finished or self.dead or self.steps_left() <= 0 or self.time_left() < 2:
            return
        order = self.roles.get("order")
        decline = self.roles.get("decline")
        if order and self.cart_used:
            if self.due_before_order() and self.steps_left() >= 2 and self.time_left() > 12:
                try:
                    self.run_process()
                except Exception as exc:
                    self.trace.append({"kind": "crash", "where": "process", "error": f"{type(exc).__name__}: {exc}"[:300]})
                if self.finished or self.dead or self.steps_left() <= 0:
                    return
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
        if (reply and not reply.get("finished") and lines and order and action["name"] == order and self.cart_used
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
            if "refusal" in observation and name != self.roles.get("order") and self.reads_decline(name):
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
            args = self.ids_for_titles(name, args)
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
            if name == cart_tool:
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
            # After a refused order a backup picked earlier may no longer be the cheapest fit: search once more first.
            if (not declining and name == self.roles.get("order") and self.plan.get("priority") == "cheapest"
                    and self.refused_at > self.searched_at and self.research_checks < 1 and self.steps_left() > 10
                    and self.time_left() > WRAP_UP_SECONDS + 20):
                self.research_checks += 1
                replies.append((call_id, "Not run yet (no step used). The last order was refused, and the request asks for the cheapest fitting item. A backup chosen earlier is right only if nothing cheaper fits now, so search again before ordering: the stated requirement words and the category, with a price limit just above the refused item's price if the search offers one, and the next pages. Open the details of cheaper candidates, put the cheapest one that meets every requirement in the cart, then order."))
                return replies
            if (not declining and name == self.roles.get("order") and len(self.swept) < 3 and self.steps_left() > 14
                    and self.time_left() > WRAP_UP_SECONDS + 50 and self.sweep_targets()):
                self.sweep_due = True
                replies.append((call_id, "Not run yet (no step used). The request asks for the cheapest fitting item: the store is searched for cheaper listings in the same category first (results follow)."))
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
            if not declining and name == self.roles.get("order") and self.steps_left() >= 2 and self.time_left() > WRAP_UP_SECONDS / 2:
                try:
                    go, note = self.run_process()
                except Exception as exc:      # the check must never cost the order
                    self.trace.append({"kind": "crash", "where": "process", "error": f"{type(exc).__name__}: {exc}"[:300]})
                    go, note = True, ""
                if not go or self.finished or self.dead or self.steps_left() <= 0:
                    replies.append((call_id, "Not run yet. " + (note or "No steps are left.")))
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
                if self.decline_checks < 2 and self.steps_left() > 4 and self.time_left() > WRAP_UP_SECONDS + 30:
                    self.decline_checks += 1
                    audit = self.audit_decline()
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
                resolved = self.resolve_pointers(observation) if name != message_tool else ""
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
                    for key, value in (args.items() if listed is None else ()):
                        spec = properties.get(key) if isinstance(properties.get(key), dict) else {}
                        if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                            self.cart_state[value] = signature
                            if signature == self.add_signature:
                                self.cart_ids = [i for i in self.cart_ids if i != value] + [value]
                            else:
                                self.cart_ids = [i for i in self.cart_ids if i != value]
                    self.cart_changed_at = self.step
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
                if name in (self.roles.get("details"), self.roles.get("view")):
                    self.opened.update(v for v in args.values() if isinstance(v, str))
                    self.opened.update(x for v in args.values() if isinstance(v, list) for x in v if isinstance(x, str))
                if name == self.roles.get("cart_view"):
                    self.last_cart_view = observation
                note = self.note_observation(name, observation)
                if resolved:
                    note = (note + " " if note else "") + resolved
                pending = (isinstance(observation, dict) and _pending_step(observation)) and not (isinstance(observation, dict) and "refusal" in observation)
                if (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished") and pending):
                    note = (note + " " if note else "") + ("The order is waiting for a further step (it gives a token or asks for a confirmation): take that step now with the tool that finishes the checkout. Nothing is bought until the episode ends.")
                elif (name == self.roles.get("order") and name != self.roles.get("decline") and not reply.get("finished")
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
                text = _compact(observation)
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
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            if self.sweep_due and self.run_sweep():
                continue
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
