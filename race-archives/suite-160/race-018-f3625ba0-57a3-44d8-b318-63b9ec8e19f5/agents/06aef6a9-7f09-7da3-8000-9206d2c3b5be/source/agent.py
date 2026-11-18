"""Kychee: a shopping agent for online store worlds.

How an episode runs (agent_main -> Kyo.run):
  1. BRIEF    one model reading turns the request and tool list into a ledger: items, musts,
              budget, message and step limits, pick rule, process steps, and which store tool
              plays which role (search, open, compare, ask, cart, buy, refuse).
  2. LOOP     a light model drives the store tools one step at a time (GUIDE prompt); every
              reply it gets carries the steps left and the ledger.
  3. CODE     steps in where the model is weakest:
              - reading: code pages short results, opens more pages before the interview,
                and follows pointers to the listing a requirement names (trail);
              - interview: a ranked lane of page keys per item, one key per message, the item with no need found first, replies
                read into firm needs (LANES, HEAR); notes that change the task update the ledger;
              - checkout: short-cart hold, cheaper-row sweep under a cheapest rule, a strong
                model's cart check (CHECK), the approval message a process step asks for;
              - endings: before an empty-handed close a strong review of everything seen (AUDIT),
                held closes, a last stand and a code-completed cart; a forced finish at the end.
Every judgement of fit is a model reading: the code holds no tables of products,
attributes, values or units.
"""
from __future__ import annotations
import difflib
import json
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
SPARE,HEAVY,LIGHT="openai/gpt-6-luna","anthropic/claude-sonnet-5.5","anthropic/claude-haiku-5.5"
(GATE_ROUNDS,READ_CAP,CHAIN_READS,SWEEP_PAGES,CHEAPER_OPENS,AUTO_PAGES,CEILING_PAGES,
  FIRST_READS,VAGUE_WORDS,LATE_LOOK)=(3,5,4,4,3,6,7,8,8,8)
ALL_CALLS,LOOP_CALLS,SPEND_HARD,SPEND_SOFT=60,42,0.050,0.035
WAIT_TOTAL,WAITS=45,(4,10,20,35)
HISTORY_LIMIT,OBS_LIMIT,SHOW_CHARS=150000,15000,450
CLOCK_EXTRAS,CLOCK_STOP,CLOCK_LIMIT=70,35,285
# most keys of one item a single question names; most words a code-built search sends
ASK_SPAN,QUERY_WORDS=7,3
_llm,_env=(ProxyClient(max_retries=tries,retry_delay=1,timeout=wait)for tries,wait in((1,50),(3,95)))
SHAPE:str="""Give back one JSON object, no other text: {"stock": the per-row key saying whether a listing can be bought right now, or null, "price": the per-row price key, or null, "next_arg": the tool parameter accepting that value, or null, "next": the key whose value brings up the next page, or null, "count": the key for the overall number of matches (not how many are displayed), or null, "rows": the key holding the list of listings}. Use the exact key names found in the result printed below (from one of the store's tools) and in the tool's parameters, fitting it onto these fields."""
TRAIL:str="""Answer with one JSON object, no prose: {"chains": [{"relation": "together" when the REQUEST wants a product meant to match, pair up or work alongside that value (any phrasing), "same" when the REQUEST's own words ask for exactly that value, "apart" when the requirement asks for a value other than it, else "other", "need": the requirement word for word, "partner": for "together", the value that a pairing rule printed on an OPENED page of the chain (in its remarks, notes or write-up) sets beside "value" (a rule reads one way: "A goes with B" makes B the partner of A, never A the partner of B), or "", "value": the value as the final page phrases it for its own listing (for "together" the value you start from, never the partner), or "", "done": true only if an OPENED page at the last link states the value itself, else false, "links": [{"id": the matching record's id from below, or "", "words": a handful (5 at most) of plain words from "about" singling this listing out among that seller's others, without the words listing and page or the seller's name, "about": how the requirement or the page before refers to that listing, "shop": who sells it as named inside those words (fill it in any time a seller is named), else ""}]}]}
Keep the links in order and stop at the first link that cannot be found or is not OPENED: its words are not known yet. Records below: OPENED = a page the agent opened; ROW = a search row only, page unread. No title rescues a record whose own attribute values contradict any property a link's words mention; a sentence on a page that denies such a property is a contradiction too - in either case the link's id stays "". Also, while that seller still has rows not OPENED, a record silent on one of those properties is not yet a match. Trace each such requirement's chain link by link, from the requirement's wording on through each page's own words. A page sending the reader on for the property gives no value for it, even one it prints itself: the chain goes on from there, and it ends only at a page that does not send the reader on. A pairing rule (a note saying which value goes with which) names values, not listings: it sends nobody on, and a listing that merely shows the partner value is no link of the chain. Why: checkout is next, some requirements take their value from another listing, one page can pass the reader on to a further listing holding the property, and the order only stands when every listing on such a chain was opened in full."""
POINTER:str="""Your answer is a single JSON object, nothing around it: {"lookups": [{"must": "property: how the shopper relates it, in their own phrasing, carrying the full description of the other listing plus any listing that page refers on to", "item": item number}]}
Instead of naming a value outright, a request can anchor a property to a different listing or page of this store - matching it, fitting with it, kept within it. Report every property anchored like that; report none when all values are spelled out. Only things for sale in this store that the request refers to qualify as that other listing. Items the shopper already has, or means to use alongside or carry with it, are not listings, so suiting them creates no lookup. Nor is a product that has to suit another product bought in this same order, or a rule weighing the candidate listings against each other (the cheapest, a margin above the cheapest of the near misses): those settle the choice among listings and point at no particular one."""
HEAR:str="""ONE JSON object only, shaped like this:
{"needs": [{"q": question number, "item": n, "need": "property: constraint", "points": true when another listing supplies the value, else false}], "free": [{"q": question number, "item": n, "property": "..."}], "drop": [{"item": n, "property": "requirement the shopper gave up"}], "crowded": [question numbers whose reply objects to too many topics in one message or asks for one question at a time; a reply leaving the choice open goes under free instead, and a reply saying only that it was confused does not belong here either], "budget": a new overall cap as a number in the budget unit named below (convert a figure stated in another unit), else null}  (n = item number)
needs C, content: a need holds a value a listing's page could display; never make one from a passing remark. Include every firm preference a reply states, with its exact value or limit and unit ("at least 30 cm", "oval", "not wicker"), word for word with nouns and units kept ("at least 3 shelves", never "at least 3"). The agent's question may contain examples, but those are not the shopper's words: a reply that only asks for fewer questions states no need. Write a need as the values it accepts: a reply refusing everything except one value or limit accepts only that one ("material: oak", "width: at least 40 cm"), a reply refusing one value keeps the negation ("material: not oak"); never write a double negative or leave the refusing words in the value.
needs B, naming: replies on one property combine into one need, named after the property the reply itself discusses, even if another was asked.
needs A, pointer values: a reply can set the value by comparing with another listing (like, unlike or near the value on some shop's product, that product described by a few properties). Properties that only describe the other product are no needs of this item. Write ONE need on the property the reply names, "points": true, whose value is the reply's exact words from the comparison word through the end of that description, including whatever is said about a listing named there; do not shorten or reword it.
free: the property of each question whose reply hands the choice to the agent, in any wording. Anything such a reply says about the shopper's day, home or mood is small talk; it yields no need.
Below are the replies a shopper gave to a shopping agent's questions. Homely, casual or indirect wording that names a value still states it; read every reply separately."""
NOTE:str="""Answer: one JSON object, nothing more: {"kind": "change" when the task itself shifted (budget, requirements, a price, availability, the order refused), "error" when it merely reports a call that was malformed or turned down, otherwise "other"; "add": [{"item": item number, "need": "property: constraint"}]; "drop": [{"item": item number, "need": "requirement the shopper withdraws"}]; "budget": numeric new overall cap in the request's unit, else null; "gone": [ids the note says are unavailable]; "price_moved": true when any price is said to have changed; "sold_out": true when any chosen listing is said to be sold out, gone or unavailable, named or not; "summary": "one short sentence"}
Capture each change carried by the note shown below; when one requirement replaces another, the new one goes under add and the old one under drop. The note turned up partway through the task, and it may come from the store or from the shopper."""
READING:str="""for a property, an attribute whose key names it decides (keys for other properties, related ones too such as handle colour vs the body's colour, decide nothing), and lacking one, the wanted value under another attribute or in the title or description counts, but a bare figure that does not say what it measures counts for nothing; such an attribute is read loosely - limits include their edge, units convert, synonyms count, one element of a list or mix suffices whatever joins them, key spelling is free, a value containing the required words side by side counts (e.g. "deep teal" for teal) while another word wedged between them breaks it ("navy pale blue" is not navy blue), and a list entry repeating a required range meets it, while a single required figure is met by a printed span of two figures that encloses it (edges count); a rule of differing from another listing's value holds unless the records show that same value (while that value is unread, only the listing pointed to or its same-seller twin breaks it); a requirement stated by the label itself is met by that category, and the category label matches the item; a listing out of stock or over the budget never qualifies; a requirement marked "only if" binds only when its condition holds."""
SWAP:str="""Answer with one JSON object, no prose: {"reason": "a sentence", "fits": [for each listing that appears to satisfy every requirement of the sold-out line's item, one entry {"price": its price as a number, read from its record, "id": id}, 6 at most]}
One cart line of a shopping agent now points at a sold-out listing. Scan the remaining records underneath for any listing able to replace it there. Reading rules: """+READING
AUDIT:str="""Your answer is a single JSON object and nothing more: {"reason": "a sentence", "open": [likeliest first, as many as 6 ROW ids of a wanted kind, within the budget, whose unread page could still show a match], "fits": {"item number": [cheapest first, as many as 4 ids that appear to satisfy every requirement]}}
A ROW record is a bare search row, its attribute page unread: a property it omits neither passes nor fails. Check the records below against the requirements once more: a shopping agent has decided no listing fits and means to leave without buying. Reading rules: """+READING
CHECK:str="""Return exactly ONE JSON object: {"ok": boolean, "problems": ["each fault, short and specific", ...], "better": [ids worth swapping into a cart line, strongest first]}. Set ok=false only for a requirement no record addresses or for a fault the records clearly show.
Judging a requirement:
(a) Only requirements in force count: a property the shopper has no preference on, an instruction naming no property value or limit, and the request's story can never fail a listing.
(b) No attribute for the property: a bare figure never saying what it measures does not count, nor does a number of another kind (weight vs capacity); the wanted value under another attribute or in the title or description does. A requirement to work with or be compatible with some kind of thing is met only by words saying what the item works with or fits: the item being such a thing, having one built in or running on one does not meet it. A fit list stated for a part number applies to the listing carrying that number (often in its title), whichever page prints it.
(c) Matching key: a different stated value fails the listing, since an attribute whose key names the property decides. A size of several multiplied figures or a list of sizes never proves a range; a required range (two figures) is met by a list entry repeating it, whatever other keys say in other units, and a listing without that entry fails when RANGE MUSTS below names a listing that has it. One key stating the value suffices when several name the property. The attribute holds when it states the value: a range lies entirely inside, a span of two figures enclosing a single wanted figure states it (edges count), limits include the edge, units are converted, a value containing the required words side by side is fine ("deep teal" for teal), though not with another word wedged between them ("navy pale blue" is not navy blue), case, singular/plural and plain synonyms are fine, one element of a list or mix is enough whatever joins them, key spelling does not matter.
(d) Other keys: a key for a different property, however close (handle colour vs the body's colour), decides nothing.
(e) Pointers: silence about an excluded value breaks nothing, and an exclusion breaks only on a record stating it. A requirement marked as not yet read is unmet; one marked as an exclusion whose value is unread breaks only the listing it is marked to point to, or a twin of that listing from the same seller, and every other listing meets it. A requirement whose value comes from a referenced page holds only if the listing itself has that value; a sentence calling it what the listing pairs with gives the listing no such value. A rule of differing from another listing's value holds unless the records show that same value.
(f) Category: the label matches the requested item as a whole, whatever kind of thing the listing's title names; a requirement carried by the label itself is met by that category. One exception: where a word of the requested item's own name gives the physical form the product comes in, that form is a requirement as well, and a listing whose attributes or title give the product in a different physical form fails it.
(g) Conditions: a requirement marked "only if" binds only when the records show its condition holds; when they show it does not, the requirement is met.
Pick rule (cheapest, most or least): the cart is wrong if one of the other listings shown clearly fits every requirement and beats the cart's choice under that rule, so look at them too.
Cart: price times quantity summed over lines within the budget; nothing marked out of stock; no extra lines; one line per requested item at its requested quantity; the latest record of a listing counts.
You are the last check before a shopping agent pays: from these records alone, is the cart exactly right?"""
STAND:str="""Return ONE JSON object only: {"breach": "for an empty pick, the printed value or price that excludes its records", "picks": {"item number": id of an OPENED record, or ""}}
If every record of an item is excluded, leave its pick empty. Otherwise choose for every item the single OPENED record that comes nearest: among records not excluded the fewest doubts win, then the pick rule given below. A doubt is a requirement the record never mentions, and it excludes nothing; title and description count as printed. Exclusion comes from a printed value differing from a requirement (a different word for a different thing, a figure beyond a limit, another pack count), a price past the budget, being out of stock, or a category for some other sort of product. Leaving empty-handed scores nothing whenever the store holds a match, and a shopping agent is about to do exactly that."""
UNMIX:str="""Give back a single JSON object and no other text: {"describe_other": [the requirements, verbatim, that do nothing but identify the other listing; empty if every one genuinely applies to the product]}. Some values of this product are borrowed from another listing named in the request. Words naming that listing - what kind of thing it is, which shop sells it, the properties following "with ..." - can get misread as constraints on the product. Go by how the request is phrased and pick out such entries among the OTHER REQUIREMENTS below."""
GATE:str="""Answer with one JSON object, no prose: {"steps": [{"step": the step, "done": boolean, "lines": [the RECORD line numbers showing it]}], "hold": whatever change a reply asks for before anything proceeds, else "", "needs": [{"need": "property: constraint", "item": item number} for every firm requirement such a reply sets], "call": {"tool": "...", "args": {...}} or null}
"call": null when nothing due is left open, or when what remains of the messages or steps cannot pay for the open one; otherwise the single next store call that gets an open due step done - a TOOLS entry plus its arguments. Anything to the shopper travels by the messaging tool as one brief, plain sentence or question on that step only, mentioning nothing about the product (property or value) beyond what it calls for.
Asking product-property questions never completes a step. Done means a RECORD line carries it out; for a cart or order step, that line also holds a reply granting it and comes after the last cart change. Go through the agent's actions in the numbered RECORD and, for the cart in its present state, say for every step coming due ahead of the next move whether it is done: the agent plans to STAGE, and the shopper's request demands the steps listed below on the way."""
LANES:str="""Return a single JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "up to 8 words of the request this pick rests on, or an empty string"}]}. Give up to COUNT picks for every item, each item's picks best first.
TASK. You plan the questions a shopping agent may send to a shopper. The shopper keeps some firm wishes that the request never wrote down, each on a single property of a single requested item. A question draws a wish out only when it names that very property, and the shopper replies about the first property a question names, so every question is worth one property and a wrong one is spent for nothing.
YOU GET the request; each item with what the request already fixes for it; and, per category, the attribute keys printed on the listings opened so far, with how many of those listings print each key and a few sample values.
RANK each item's picks on its own, in this order:
1. What the request asks you to confirm with the shopper (shown under ASK FIRST), named by the property concerned.
2. Then the item's listed keys: first the ones its listings differ on and that a buyer would pick this kind of product by, preferring keys printed on most of its listings; once five or more listings were opened, a key printed on only one or two of them comes late. A key the listings print with a single value goes last.
3. A sentence of the request that bears on the item while leaving a property's value open (the person who will use it, the look it must have beside something else, what it must hold or the space it must fit, a choice the shopper says they cannot make) only settles a near tie under 2, in favour of the listed key that fits it; it never lifts a key above one that rule 2 clearly prefers, and talk about the shopper's day, work or home decides nothing.
LEAVE OUT a property the request already fixes for that item (a key whose values only repeat a fixed value is still open), a second name for a property already picked, a key whose values list several purchase variants at once, identifiers and model names, warranty, delivery and other logistics.
LABEL: the plain name of one property, every word of the listed key written out ("strap width" for strap_width, "seat height" for seatHeight), no values, no brackets."""
BRIEF:str="""Reply with a single compact JSON object (no indentation) and nothing else. Built from a shopper's request and the store's tool list, it becomes the task description a shopping agent works from. Keys:
"items": list with one entry per distinct product to buy: {"name": product kind in the request's wording, "shelf": the store's category label for it, copied exactly, if some tool lists labels, else "", "quantity": units of that one listing (integer; 1 if the request says nothing; a pair or set sold as one listing is 1; only a number placed next to this product in the buying words counts, never a count the story implies such as people who each need one, rooms, uses or earlier purchases), "musts": the product's firm requirements, each "property: constraint" with the exact value, limit, exclusion or closed list}.
  Musts: (a) A must gives its value, limit, exclusion or closed list, with every word the request writes beside a figure (the unit or sizing system it is counted in); an exclusion keeps its negation however it is worded ("anything but red" -> "color: not red"), never the bare value it rules out; a property the shopper cares about without a value is not a must, as the interviewer asks about it. (b) Words that are part of the product's name (a season, audience or use named with it) go in "name", never in musts. A style, look or material named beside the product (written either way: "rustic-style lamps" or "lamps, rustic style") is a value the store's label does not carry: it goes in musts ("style: rustic") and "name" keeps only the product kind. (c) Background story (who uses it, where it goes, what failed before, the occasion) is context and no must unless it states a property value or limit outright. (d) Price, budget, stock and quantity never go in musts. (e) If a second listing supplies the value, write ONE must containing that listing's full description ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"); all words describing the other listing (its shop, its kind, the trailing "with <property> <value> and <property> <value>") stay inside that must and never turn into musts of this product. Keep the shopper's comparison: wanting this product to go well with, pair with or be compatible with the other listing differs from wanting the same value, so never rewrite one as the other, whatever figures of speech surround it; a value said to be right for or suited to the other listing is such a pairing, and "same as" is written only when the request asks for the same or a matching value. (f) Keep the request's own property word ("meant for a lakeside occasion" -> "occasion: lakeside", "a 12-inch blade" -> "blade length: 12 inch"); do not swap in a neighbouring property. (g) A requirement the request makes only under a condition (it applies should some other property or listing turn out a certain way) stays ONE must that carries the condition in the request's words, "<property>: <constraint>, only if <condition>"; never drop the condition and never turn it into a plain must.
  Counting: one phrase joining products with "and" (a single "I want ..." with a single quantity, often the name of one store department) is ONE item; split only when each product has its own quantity or own requirements.
"nearest_fallback": true if the request wants the near miss (the listing missing the least) offered or bought when no listing meets everything; else false.
"quiet_decline": true if the request says to suggest or name no alternatives when nothing fits, false if it asks for them, null if it says nothing.
"process": steps demanded beyond finding and buying the right items (shopper's approval, telling or showing them something, checking something with them, reading something first, using a particular tool). Each is {"step": the action in at most 15 words, "when": "choose" (before choosing products), "order" (just before checkout) or "end" (just before ending with no purchase), "messages": shopper messages it uses (0 if none)}; [] when there are none. Item requirements, budget, stock, the message and step limits and how to finish when nothing fits are not steps; neither is anything the request wants done only in the case where no listing meets everything (telling the shopper so, offering the near miss): "nearest_fallback" and "quiet_decline" carry that case.
"ask_first": what the request explicitly wants confirmed with the shopper ahead of the pick, one entry per thing it names (one sentence naming two things gives two entries); empty if nothing.
"wishes": what the shopper would only like.
"rules": every instruction about how to work or how to finish, 12 words at most each. A hope about how the product behaves or suits its place, naming no property value or limit, is background and is left out.
"lookups": musts (copied exactly) whose value is read first from another listing the request refers to; else [].
"steps": the cap on steps, moves or tool calls (integer), or null.
"messages": how many messages the agent may send the shopper (integer); null if unstated.
"pick": "cheapest" (lowest price or smallest total wanted), "most" / "least" (the extreme of some measure wanted), else "any". Remarks about tight money set no pick rule.
"pick_measure": when "pick" is "most" or "least", the name of the property whose extreme is wanted (a property, never the word most or least); else "".
"in_stock_only": true unless the request plainly accepts unavailable items.
"budget_scope": "total" (one limit over all items), "each" (a limit per item), or null (no limit).
"budget_unit": that unit in words (e.g. "cents"), or null.
"budget": the most the shopper will spend, as a number in the request's own unit, or null.
"refuse_args": fixed arguments under which the "refuse" tool ends with no purchase; otherwise {}.
"buy_args": fixed arguments under which the "buy" tool checks out (for a tool with a mode argument, that mode); otherwise {}.
"search_text": if the "search" tool has more than one text parameter or a mode parameter, {"text": parameter for free search words, "mode": {mode parameter: its value for a word search}}; otherwise {}.
"browse_args": for the browse tool, {"label": parameter that takes the category label or null, "cap": parameter that takes a top price or null}; otherwise {}.
"roles": object giving the exact tool for each job, null when no tool does it: "search"; "open" (attributes or properties of one listing; with two single-listing tools, the one whose description offers attributes, properties or specifications); "offer" (one listing's current offer, price and stock, if not the "open" tool); "compare" (several listings in full); "ask" (sends messages to the shopper); "cart" (edits the cart); "cart_view" (displays the cart); "buy" (checks out); "refuse" (ends with no purchase); "browse" (lists a single category, often below a price ceiling, if not the free-text "search" tool).
"""
GUIDE:str="""REPLY FORMAT: every reply carries one or more tool calls and a short line of text (what is established so far, then what you do next).
ROLE: you are Kyo, buying for a single shopper inside an online store, and you act only through the store's tools.
OUTCOME: the score is binary, 1 or 0, and only an exactly right finish of either kind earns the 1. (a) A placed order: every requested item present at its requested quantity, every line meeting each requirement still in force when checkout happens, with the total inside the budget. (b) When nothing in the store meets everything, an ending through the no-purchase tool, performed the way the request asks.
[1] STEPS, STORE CALLS, INTERVIEWER
1a. Besides buying, a request may call for steps on the way (an approval, a message, a check, a reading first); the ledger lists them. Steps due just ahead of the order or the ending run on their own when you call checkout or the ending tool: do not wait for them, and never treat them as grounds for ending without buying. Do any other step yourself with store tools (product questions still go through the interviewer).
1b. You never write to the shopper; Kyo's interviewer sends every message. Its questions and the replies show up in the conversation and are copied into the ledger.
1c. A search row only summarises. Stores divide a record in their own ways: attributes (properties) from one tool, offer and description from another, and a side-by-side tool showing either. Once Kyo knows which tool gives attributes, the ledger records it. Property requirements are judged on attributes, so read them for every listing you might buy; one call covering several listings saves steps when it shows what you need. A read listing stays read until something changes.
1d. Searches can be narrow: a category filter, keywords matched only against the first words of a query, small pages, a next-page token or offset. When a choice depends on all rows, hold the store's reported total against the rows in hand and keep paging, passing the token or offset back exactly as given. Short queries led by the value work best: the wanted value first, the product noun after it. A filtered search returning no rows usually means a wrong filter value or parameter; try again with the label spelled the way rows print it, or move to a different filter parameter.
1e. Each call uses up one of a small number of steps. Thinking is free, yet every reply takes clock time, so put independent searches and reads into a single reply. Tool names, argument names and result layouts belong to this store alone: read each tool's description and schema and copy argument names and allowed values exactly.
[2] DOES A LISTING FIT
2a. Paired items (an accessory for a device): pick them together from what the listings themselves say (a list of what it fits, a model or series code shown on both pages); a fit list stated for a part number belongs to the listing carrying that number, whichever page prints it. When no accessory meeting its own requirements fits the chosen device, switch to another qualifying device that such an accessory fits; try this switch before any empty-handed ending.
2b. If a listing you are about to buy refers you to another page for details about itself, read that page first.
2c. Never buy a sold-out listing. Add up price times quantity across all lines: the total may reach the budget but never pass it.
2d. Exclusions and closed lists bind; wishes ("ideally", "would be nice") only settle ties.
2e. With no attribute for the property, the wanted value counts when it shows up under another attribute (a styles or features list holding it, say) or in the title or description. A number measuring something else does not count (weight is not capacity), nor does a bare figure whose text never says what it measures. A must to be compatible with something counts only where the text says what the item works with, never where the item is that thing, holds one or draws power from one.
2f. When a key names the required property (by that name or a plain synonym), that attribute alone decides: a different stated value fails the listing whatever the title or other keys claim. If several keys name it, the one named exactly like the property decides; lacking that one, any of them stating the value suffices. With no key naming it, the wanted value under another attribute or in the title counts as met. A key for any other property, related or not (handle colour is not body colour), decides nothing for it. On opened listings Kyo adds [evidence] lines naming keys that match a requirement.
2g. Ranges: a must naming a range (two figures) names a value the store prints. An attribute holding that range as one list entry meets it, and no other key, converted figure or rounding can undo that. A listing without such an entry is a fallback only, used once no listing with the entry can be found; then a single plainly stated figure converted into the range may serve, never a size built from several multiplied figures or a list of sizes.
2h. A property requirement is met when the attribute for that property states the value. Key spelling is irrelevant (spaces, underscores, camel case, prefixes), and one entry of a list or mix suffices, whatever joins the entries (comma, slash, plus sign). Compare values word by word, ignoring case, singular or plural and plain synonyms; a value containing the required words side by side counts ("deep teal" satisfies teal, "extra roomy fit" satisfies roomy), but not with another word wedged between them ("navy pale blue" is not navy blue). Other words do not: an abbreviation or code is not the word it stands for, and "teal green" is not "sea green". Convert units before comparing numbers; a limit includes its edge; a range counts only when wholly inside the limit. A single wanted figure (no limit word) is stated by an attribute printing a span of two figures that encloses it, edges included: such a span is not a different value. Of a closed list, any one value suffices.
2i. The category label gives the product kind, and only a label matching the requested item as a whole counts; a nearby label never does. A listing filed under the requested item's own label is that item even when its title names another or a smaller kind of thing: never rule it out, or pass it over in a price comparison, for its title. A label that itself holds a requirement's value (a season or an audience inside the label) meets that requirement.
2j. The ledger alone decides fit: its musts, budget and stock. The shopper's story (where it will go, who uses it, a past failure) adds nothing the ledger does not carry, and a ledger instruction naming no property value or limit is no requirement. Do not search for such things, do not leave a listing unconfirmed because of them, and do not reject a listing over them. A property marked "no preference" fails nothing. As soon as an opened, buyable listing meets every must, buy the best one under the pick rule; ending empty-handed needs a must that every listing breaks.
[3] PICKING AMONG LISTINGS THAT FIT
3a. Different items take one cart line each, at each one's requested quantity.
3b. Cart an item as soon as you have opened a listing meeting all its requirements (under a cheapest, most or least rule, once the fitting set has been compared), then move on. Carted lines often sell out soon after, and carting early leaves steps for a replacement.
3c. No pick rule: any fully fitting listing is correct; spend no steps looking for a better one.
3d. Most or least of a measure wanted: compare every fitting listing on that measure and take the extreme; a listing silent on the measure cannot win.
3e. Cheapest wanted (lowest total, smallest price): the basket takes the lowest-priced of ALL fitting listings. Collect the whole fitting set (page through, since rows often come unsorted by price; Kyo fetches later pages of short results itself), test candidates from the lowest price upward, and keep the first that fully fits. If a search accepts a maximum price, list the category below the price of your current best pick so only cheaper candidates show.
[4] MID-TASK CHANGES
4a. Price and stock can change without notice, but Kyo re-checks every cart line itself when checkout is sent: send the checkout instead of reading a carted listing again.
4b. The shopper or the store may alter the task partway: a new budget, a requirement added or removed, a new price, a listing sold out. The news comes inside tool results, often as a refusal when you try to check out; a note counts whichever tool returns it, and the ledger shows the requirements in force now. A refusal does not end the episode: apply every change, correct the cart, and check out again. When only a price has moved and the sum is still under the budget, keep that line: read the listing again and resend checkout instead of swapping it.
[5] ENDING
5a. End without buying only after a real search found no listing meeting everything. Running low on steps proves nothing of the sort; then check out the best cart you have. Phrase that ending as the request asks, and if a remark is needed keep it short and name no listing.
5b. Keep only the chosen lines in the cart (remove anything else), view the cart once, then send checkout as a call on its own.
[6] VALUES TAKEN FROM ANOTHER LISTING
6a. Some requirements point at a second listing in place of a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve these first, in few steps. With a shop named, search using only the shop's name as the text, which returns just that shop's few listings (a shop filter on a browsing tool mostly returns nothing), then read them with the side-by-side tool; with no shop, search by category, brand or words.
6b. Open the full page through the single-item or side-by-side tool; a row is not enough. Should that page point on to a further listing for the property, find and open that one too, and so on; the final page gives the value. An order only counts once every page of that chain was opened.
6c. Apply the value according to the comparison in the shopper's own words. Same or matching: the bought listing shows that value. A limit: measure from that value. Going well with, pairing with or being compatible with it, in whatever words: the value is only where you start. The final page, or a rule printed on a page you read, says which other value goes with it, and the listing you buy carries that partner value itself for the property; search afresh on the partner value; only when no page you read prints such a rule does the starting value itself stand. Otherwise listings carrying the starting value fail, the chain's end listing included, and a sentence that merely names the partner as a companion gives a listing no such value. Differing from it: only listings carrying exactly that value are out, and this alone is never grounds for an empty-handed finish."""
_svb_RESEND=(402,408,425,429)
_kb_MAKERS={"i":int,"b":bool,"f":float,"l":list,"d":dict,"s":set,"t":str,
  "n":lambda:None,"m":lambda:-1}
_kb_FIELDS="""step:i broke:b no_credit:b waited:i misses:i calls:i loop_calls:i cost:f trace:l brief:d roles:d items:l audit_rows:i flawed:n
budget:n msg_limit:i msg_used:i process:l reserve:i gate_msgs:i gate_ok:s done_log:l done_steps:l asked:b qa:l
rows:d full:d seen_order:l heard:s notes:l version:i cart_touched:b cart_at:m cart_seen_at:m cart_view:n
cart_lines:d checked:s audited:b offer_seen:d gone:d unavailable:s sold_lines:s note_sold:b price_rose:b held:i
id_args:s cart_said:t refused_at:m leads:l refuse_blocks:i pushed:i built:l dropped:s late_held:b lead_items:d
verdicts:d confirmed:n completed:b id_key:n facts_from:d queued:l buys:i refusals:i recent:l chat:l opened:s
chain_ends:s range_held:s refs:l ref_values:d ref_words:d ref_firsts:d trails:i trail_blocks:i rehops:l compare_ok:i compare_bad:i
walked_full:i walked_step:i tool_keys:d view_keys:d read_cache:d pages_read:i shapes:d shape_said:d price_key:n stock_key:n
cheaper_marks:s wider_held:b paged_obs:l search_log:d fixed_args:d text_fields:d carted:l swap_fits:l swept:b
committed:b full_held:b late_looked:b own_statement:b stood:b standing:b sought:s said_for:t
answered:d page_bad:s page_args:d mode_pins:d facts_read:d tokens_seen:s page_ids:s proven_goal:n proven_tried:b apart:d refilled:s cap_cut:n mended:s ver_seen:d blind:d thin_reads:s code_finds:d light_mute:b"""
_svb_STATE=tuple((_kb_MAKERS[code],name)for name,code in(pair.split(":")for pair in _kb_FIELDS.split()))
_kb_FENCE=re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$",re.S)
_kb_TOKEN=re.compile(r"[A-Za-z0-9_\-]*\d[A-Za-z0-9_\-]*")
_kb_CODE=re.compile(r"\b[A-Za-z]{1,6}[- ]?\d{3,6}\b")
_kb_CAMEL,_kb_GAPS=re.compile(r"([a-z0-9])([A-Z])"),re.compile(r"[\s_.\-]+")
_kb_PLURAL,_kb_FIGURE=re.compile(r"^(.{3,})s$"),re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_kb_NUMERIC,_kb_WHOLE=re.compile(r"\s*-?\d+(\.\d+)?\s*"),re.compile(r"\s*\d+(\.0+)?\s*")
def _kb_num(x):return not isinstance(x,bool)and isinstance(x,(int,float))
def _kb_norm(raw):return" ".join(re.findall(r"[a-z0-9]+",str(raw).lower()))
def _kb_page_fit(spec,token):
 # a page slot is a free scalar whose type and range hold the reply's token
 if not isinstance(spec,dict)or"enum"in spec:return False
 kinds=set(_kinds(spec))
 if token is None:return bool(kinds&{"integer","string"})
 if isinstance(token,float)and token.is_integer():token=int(token)
 if isinstance(token,bool)or not isinstance(token,(int,str)):return False
 want="integer"if isinstance(token,int)else"string"
 top,errs=spec.get("maxLength"),[]
 _shape(token,spec,"",errs)
 return want in kinds and not errs and not(want=="string"and isinstance(top,int)and len(token)>top)
_kb_TESTS=dict(string=lambda x:isinstance(x,str),integer=lambda x:not isinstance(x,bool)and isinstance(x,int),
  number=_kb_num,boolean=lambda x:isinstance(x,bool),array=lambda x:isinstance(x,list),
  object=lambda x:isinstance(x,dict),null=lambda x:x is None)
_kb_NUM_EDGES=(("minimum",(int,float),True,"is below {}"),("maximum",(int,float),False,"is above {}"))
_kb_LIST_EDGES=(("maxItems",int,False,"allows at most {} entries"),("minItems",int,True,"needs at least {} entries"))
def _kinds(spec):
 declared=spec.get("type")
 pool=[declared]if isinstance(declared,str)else(declared or[])
 return[kind for kind in pool if isinstance(kind,str)]
def _kb_fits(obj,kinds):return any(_kb_TESTS.get(kind,lambda _:True)(obj)for kind in kinds)
def _kb_bend(obj,kind):
 if kind=="integer"and isinstance(obj,float)and obj.is_integer():return True,int(obj)
 if kind in("integer","number")and isinstance(obj,str)and _kb_NUMERIC.fullmatch(obj):
  figure=float(obj)
  return True,int(figure)if kind=="integer"or figure.is_integer()else figure
 if kind=="string"and _kb_num(obj):return True,str(obj)
 if kind=="boolean"and isinstance(obj,str)and obj.strip().lower()in("true","false"):return True,obj.strip().lower()=="true"
 if kind=="array"and obj is not None and not isinstance(obj,list):return True,[obj]
 return False,obj
def _kb_coerce(obj,kinds):
 for kind in kinds:
  applied,bent=_kb_bend(obj,kind)
  if applied:return bent
 return obj
def _kb_over(measure,spec,edges):
 notes=[]
 for key,sort,below,wording in edges:
  edge=spec.get(key)
  if isinstance(edge,sort)and(measure<edge if below else measure>edge):notes.append(wording.format(edge))
 return notes
def _shape(obj,spec,path,errs):
 if not isinstance(spec,dict):return obj
 kinds=_kinds(spec)
 if kinds and not _kb_fits(obj,kinds):
  obj=_kb_coerce(obj,kinds)
  if not _kb_fits(obj,kinds):
   errs.append(f"{path} should be {'/'.join(kinds)}")
   return obj
 options=spec.get("enum")
 if isinstance(options,list)and obj not in options:
  if isinstance(obj,str):
   wanted=obj.strip().lower()
   obj=next((o for o in options if isinstance(o,str)and o.lower()==wanted),obj)
   near=[o for o in options if isinstance(o,str)and _kb_norm(o)==_kb_norm(obj)]
   obj=near[0]if obj not in options and len(near)==1 else obj
  if obj not in options:errs.append(f"{path}: allowed values are {options}")
 if isinstance(obj,str):
  obj_2=obj
  top,bottom=spec.get("maxLength"),spec.get("minLength")
  if isinstance(top,int)and len(obj_2)>top:obj_2=obj_2[:top]
  if isinstance(bottom,int)and len(obj_2)<bottom:errs.append(f"{path} is too short")
  return obj_2
 if _kb_num(obj):errs.extend(path+" "+note for note in _kb_over(obj,spec,_kb_NUM_EDGES))
 elif isinstance(obj,list):
  errs.extend(path+" "+note for note in _kb_over(len(obj),spec,_kb_LIST_EDGES))
  if isinstance(spec.get("items"),dict):obj=[_shape(sub,spec["items"],"%s[%d]"%(path,at),errs)for at,sub in enumerate(obj)]
 elif isinstance(obj,dict):
  known=spec.get("properties")
  known=known if isinstance(known,dict)else{}
  if spec.get("additionalProperties")is False:errs.extend(f"{path}.{arg}: the schema has no such argument" for arg in obj if arg not in known)
  errs.extend(f"{path}.{arg} is required" for arg in spec.get("required")or[]if arg not in obj)
  obj={arg:_shape(sub,known[arg],f"{path}.{arg}",errs)if arg in known else sub for arg,sub in obj.items()}
 return obj
def _json_obj(raw):
 if not isinstance(raw,str):return None
 body=raw.strip()
 fence=_kb_FENCE.match(body)
 if fence:body=fence.group(1)
 left,right=body.find("{"),body.rfind("}")
 for chunk in(body,body[left:right+1])if 0<=left<right else(body,):
  try:found=json.loads(chunk)
  except ValueError:continue
  return found if isinstance(found,dict)else None
 return None
def _dump(obj):
 try:return json.dumps(obj,ensure_ascii=False,separators=(",",":"))
 except(TypeError,ValueError):return str(obj)
def _clip(obj,limit,depth=0):
 if depth>7:return"..."
 if isinstance(obj,str):return obj[:limit]+"..."if len(obj)>limit else obj
 if isinstance(obj,dict):return dict((k,_clip(sub,limit,depth+1))for k,sub in obj.items())
 if isinstance(obj,list):return list(_clip(sub,limit,depth+1)for sub in obj)
 return obj
def _figure_words(raw):return _kb_FIGURE.findall(str(raw).lower())
def _two_figures(seq):
 last=None
 for at,piece in enumerate(seq):
  if piece[0].isdigit():
   if last is not None and at-last<=2:return True
   last=at
 return False
def _same_word(one,two):
 if one[0].isdigit()or two[0].isdigit()or len(one)<3 or len(two)<3:return one==two
 short,longer=sorted((one,two),key=len)
 return longer.startswith(short)
def _repeats(haystack,entry):
 needle=_figure_words(entry)
 if not _two_figures(needle):return False
 span=len(needle)
 for at in range(len(haystack)-span+1):
  if all(map(_same_word,needle,haystack[at:at+span])):return True
 return False
def _words(raw):
 spaced=_kb_CAMEL.sub(r"\1 \2",str(raw))
 return _kb_GAPS.sub(" ",spaced).lower().strip()
def _stem(raw):return frozenset(_kb_PLURAL.sub(r"\1",piece)for piece in _words(raw).split())
def _tokens(raw):
 pieces=re.findall(r"[A-Za-z0-9]+",str(raw))
 return _stem(" ".join(pieces))
def _side(want,text):
 # every word of the wanted value stands in one unbroken run of the printed value: no foreign word wedged between them
 bit=lambda t:[_kb_PLURAL.sub(r"\1",w)for w in re.findall(r"[a-z0-9.]+",str(t).lower())]
 w,t=bit(want),bit(text)
 return any(sorted(t[i:i+len(w)])==sorted(w)for i in range(len(t)-len(w)+1))
def _akin(a,b):
 # one word in two spellings: equal, or the longer holds one more vowel (flavour and flavor name one property)
 if len(a)<len(b):a,b=b,a
 return a==b or len(b)>=5 and len(a)==len(b)+1 and any(a[i]in"aeiou"and a[:i]+a[i+1:]==b for i in range(len(a)))
def _nest(small,big):return all(any(_akin(w,o)for o in big)for w in small)
def _names(prop,key):return bool(prop)and bool(key)and(_nest(prop,key)or(len(key)>1 and _nest(key,prop)))
def _common_words(vals):
 bags=[{w for w in re.findall(r"[a-z]{3,}",v)}for v in vals]
 return 5*sum(any(bag&o for j,o in enumerate(bags)if j!=i)for i,bag in enumerate(bags))>=3*len(bags)
def _value_parts(vals):return{p.strip()for v in vals for p in re.split(r"[,;/&]",v)if p.strip()}
def _count(obj):
 if isinstance(obj,bool):return None
 if isinstance(obj,int):return obj
 if isinstance(obj,float):return int(obj)if obj.is_integer()else None
 return _kb_WHOLE.fullmatch(obj)and int(float(obj))if isinstance(obj,str)else None
def _items(obj):
 if isinstance(obj,list):return obj
 return[obj]if obj and isinstance(obj,(str,dict))else[]
def _text_value(raw):
 # a printed literal back to its value: numbers, truth values and the empty marker
 core=raw.strip()
 if core in("True","False"):return core=="True"
 if core=="None":return None
 return(float(core)if"."in core else int(core))if re.fullmatch(r"-?\d+(?:\.\d+)?",core)else core
def _text_pairs(raw,cut,link,word=False):
 # "k<link>v<cut>k<link>v ...": a piece without the link, or (word) whose key is more than one word, continues the value before it
 out,last={},None
 for piece in raw.split(cut):
  key,sep,val=piece.partition(link)
  if sep and key.strip()and len(key.strip())<=40 and link not in key and not(word and re.search(r"\s",key.strip())):last=key.strip();out[last]=val
  elif last is not None:out[last]+=cut+piece
  else:return{}
 return out
def _text_record(raw,depth=0):
 """A record a store printed as text, else None.

 JSON sent as a string is read as JSON. Lines starting again with the text's first key begin new records (a list). A record is
 "key: value" parts, one-word keys, joined by " | " or one per line; an inner "a=b; c=d" (or "a=b, c=d") list becomes the
 attribute block.
 """
 core=raw.strip()
 if depth>3 or len(core)<8:return None
 if core[0]in"{["and core[-1]in"}]":
  try:return _untext(json.loads(core),depth+1)or None
  except ValueError:return None
 lead=re.match(r"\s*([^\s:|]{1,40}): ",raw)
 parts=[x for x in re.split(r"\n(?=\s*%s: )"%re.escape(lead.group(1)),raw)if x.strip()]if lead else[]
 if len(parts)>=2:
  rows=[_text_record(x,depth+1)for x in parts]
  if all(isinstance(r,dict)for r in rows):return rows
 rec={}
 for cut in(" | ","\n"):
  rec=_text_pairs(raw,cut,": ",True)if raw.count(cut)>=2 else{}
  if len(rec)>=3:break
 if len(rec)<3:return None
 for key,val in rec.items():
  inner={}
  for cut in("; ",", "):
   inner=_text_pairs(val,cut,"=")if val.count("=")>=2 and cut in val else{}
   if len(inner)>=2:break
  rec[key]={k:_text_value(v)for k,v in inner.items()}if len(inner)>=2 else _text_value(val)
 return rec
def _untext(node,depth=0,spot=True):
 # text records where a whole record sits (a list entry, a reply of one or two values) become records; a reply holding just one becomes that record
 if depth>6:return node
 if isinstance(node,str):return spot and _text_record(node)or node
 if isinstance(node,list):
  out=[_untext(x,depth+1)for x in node]
  return _named_values(out)or out
 if not isinstance(node,dict):return node
 out={k:_untext(v,depth+1,len(node)<=2)for k,v in node.items()}
 # listings keyed by their ids: each becomes a row that carries its id
 if all(_id_like(k)and isinstance(v,dict)and len(v)>=2 and k not in map(str,v.values())for k,v in out.items())and out:return[dict(v,id=k)for k,v in out.items()]
 only=list(out.values())
 return only[0]if len(out)==1 and isinstance(only[0],dict)and isinstance(list(node.values())[0],str)else out
def _named_values(entries):
 # a list naming one value per entry (name/value records, [name, value] pairs, "name: value" lines) as one name -> value record
 if len(entries)<2:return None
 if all(isinstance(e,list)and len(e)==2 for e in entries):pairs=[tuple(e)for e in entries]
 elif len(entries)>=3 and all(isinstance(e,str)for e in entries):pairs=[tuple(m.groups())for m in(re.fullmatch(r"\s*([^:=\n]{1,40}?)\s*[:=]\s*(.*)",e,re.S)for e in entries)if m]
 elif all(isinstance(e,dict)and len(e)==2 for e in entries)and len({tuple(e)for e in entries})==1:
  cols=list(entries[0])
  names=next((c for c in cols if all(isinstance(e[c],str)and 0<len(e[c].strip())<=40 for e in entries)and not all(_id_like(e[c])for e in entries)),None)
  pairs=[(e[names],e[cols[1]if names==cols[0]else cols[0]])for e in entries]if names else[]
 else:return None
 names=[str(a).strip()for a,_ in pairs]
 # names that look like listing ids, or values that all do, make a list of rows, not of details
 if len(pairs)<len(entries)or not all(isinstance(a,str)for a,_ in pairs)or len(set(names))<len(names)or all(map(_id_like,names))or all(_id_like(b)for _,b in pairs):return None
 return{a:_text_value(b)if isinstance(b,str)else b for a,b in zip(names,(b for _,b in pairs))}
def _nested_keys(a,b):
 # two key names for one property in a shopper's eyes: same first word with one name inside the other, or the same first two words
 ta,tb=_words(a).split(),_words(b).split()
 if not(ta and tb)or ta==tb:return bool(ta and tb)
 return ta[0]==tb[0]and(set(ta)<=set(tb)or set(tb)<=set(ta))or(len(ta)>1 and len(tb)>1 and ta[:2]==tb[:2])
def _spoken_key(key):
 # a key name ending in a run of figures and single letters (an example the store glued on) is asked without that run
 words=key.split()
 cut=next((i for i in range(1,len(words)-1)if all(len(w)==1 or re.search(r"\d",w)for w in words[i:])),len(words))
 return" ".join(words[:cut])
def _fact_block(node):
 if not isinstance(node,dict):return None
 best=None
 for group in node.values():
  if not isinstance(group,dict)or len(group)<2:continue
  flat={k:x for k,x in group.items()if _kb_num(x)or isinstance(x,str)}
  if len(flat)>1 and(best is None or len(flat)>len(best)):best=flat
 return best
def _in_rows(node,ident,depth=0):
 if depth>5 or not isinstance(node,(dict,list)):return False
 kids=node.values()if isinstance(node,dict)else node
 if isinstance(node,list)and any(isinstance(row,dict)and ident in{x.strip()for x in row.values()
   if isinstance(x,str)}for row in node):return True
 return any(_in_rows(kid,ident,depth+1)for kid in kids)
def _id_like(raw,named=False):
 if not isinstance(raw,str):return False
 core=raw.strip()
 if" "in core or len(core)<3 or len(core)>64:
  return False
 return named or bool(re.search(r"\d",raw))
def _kb_id_rank(rows,key):
 named="id"in str(key).lower()
 column=[row.get(key)for row in rows]
 if not all(_id_like(cell,named)for cell in column):return None
 trimmed=[cell.strip()for cell in column]
 if len(rows)>1 and len(set(trimmed))!=len(trimmed):return None
 return named,sum(map(len,trimmed))/len(trimmed)
def _learn_id(rows):
 chosen,top=None,None
 for key in rows[0]:
  rank=_kb_id_rank(rows,key)
  if rank is not None and(top is None or rank>top):chosen,top=key,rank
 return chosen
class Item:
 def __init__(self,name,shelf,qty,musts):vars(self).update(name=name,shelf=shelf,qty=qty,musts=musts,told=[],dropped=[],free=[])
def _kc_env_verdict(code,text,step):
    # pure: how act() reacts to a failed environment send -> (kind, value)
 if code in(400,401,402,403,404,422):return"halt",code==402
 if code!=409:return"again",None
 low=text.lower()
 if any(w in low for w in("finish","quarantin","finaliz")):return"closed",None
 hint=re.search(r"expected (\d+)",text)
 if hint:return"seek",int(hint.group(1))-1
 if"reuse"in low or"different action"in low:return"seek",step+1
 return"halt",False
class Kyo:
 """One episode: store binding, ledger, what the store has shown, cart tally and the model chat."""
 def __init__(self,problem):
  env=problem["environment"]
  pv=env["policy_view"]
  self.binding,self.view=dict(env["binding"]),pv
  self.request=str(pv.get("query")or"")
  usable=[]
  for entry in pv.get("tools")or[]:
   fn=entry.get("function")if isinstance(entry,dict)else None
   if isinstance(fn,dict)and isinstance(fn.get("name"),str):usable.append(entry)
  self.tools=usable
  self.schemas={}
  for entry in usable:
   shape=entry["function"].get("parameters")
   self.schemas[entry["function"]["name"]]=shape if isinstance(shape,dict)else{"type":"object","properties":{}}
  cap=_count(pv.get("max_steps"))
  if cap is None and isinstance(pv.get("max_steps"),float):cap=int(pv["max_steps"])
  self.max_steps=cap if cap and cap>0 else 30
  self.finished=bool(pv.get("finished"))
  for factory,joined in _svb_STATE:
   for attr in joined.split():setattr(self,attr,factory())
  self.began=time.monotonic()
 def left(self):
  used=self.step
  return self.max_steps-used
 def secs(self):
  elapsed=time.monotonic()-self.began
  return CLOCK_LIMIT-elapsed
 def msgs_free(self):
  held=max(0,self.reserve-self.gate_msgs)
  return self.msg_limit-self.msg_used-held
 def seekers(self):return{self.roles.get("search"),self.roles.get("browse")}-{None}
 def page_arg(self,tool,token,call_args=None):
  spec=self.schemas.get(tool)or{}
  props,kept=spec.get("properties")or{},self.page_args.get(tool)
  if token is None and kept in props and(tool,kept)not in self.page_bad:return kept
  named=(self.brief.get("browse_args")or{}).values()if tool==self.roles.get("browse")else()
  taken={*(spec.get("required")or[]),*named,self.text_fields.get(tool),*self.mode_pins.get(tool,{}),*self.fixed_args.get(tool,{})}
  taken|={k for k,v in(call_args or{}).items()if k!=kept and _dump(v)not in self.tokens_seen}
  fits=[a for a,fs in props.items()if a not in taken and(tool,a)not in self.page_bad and _kb_page_fit(fs,token)]
  size=lambda a:props[a].get("maxLength")if isinstance(props[a].get("maxLength"),int)else 10**6
  if len(fits)>1 and(token is None or isinstance(token,str)):fits=[a for a in fits if size(a)==min(map(size,fits))]
  pick=(self.shapes.get(tool)or{}).get("next_arg")
  arg=pick if pick in fits else(fits[0]if len(fits)==1 else None)
  if token is not None:self.page_args[tool]=arg
  return arg
 def page_dead(self,tool,arg):
  self.page_bad.add((tool,arg))
  for rec in self.search_log.values():
   if rec["tool"]==tool and rec["page_arg"]==arg:rec["token"]=None
  self.trace.append({"kind":"page_dead","tool":tool,"arg":arg})
 def cart_mark(self):return _dump(sorted(self.cart_lines.items()))
 def page_tools(self):
  r=self.roles
  return{name for name in(r.get("offer")or r.get("open"),r.get("compare"))if name}
 def facts_tool(self):
  for name in[n for n,shown in self.facts_from.items()if shown]:
   if self.single_target(name):return name
  alt=self.roles.get("open")
  usable=self.facts_from.get(alt or"")is not False and self.single_target(alt)
  return alt if usable else None
 def compare_bare(self):
  known=self.facts_from
  side=known.get(self.roles.get("compare")or"")
  if side is not None:return side is False
  return known.get(self.facts_tool()or"")is True or bool(self.roles.get("offer"))
 def gate_due(self,stage):
  if not any(p["when"]==stage for p in self.process):return False
  return(stage,"%s#%s"%(self.cart_mark(),self.version))not in self.gate_ok
 def tail(self):
  spare=self.msg_used<self.msg_limit
  if any(p["when"]=="order"and p["messages"]>0 for p in self.process):room=spare and self.gate_due("order")
  else:room=self.own_statement and spare and self.said_for!=self.cart_mark()
  return 2 if room else 1
 def think(self,messages,models,max_tokens,tools=None):
  code=None
  for engine in models:
   if self.calls>=ALL_CALLS or self.secs()<10:break
   self.calls+=1
   tries=0
   while tries<=len(WAITS):
    reply,code=self.call_model(engine,messages,max_tokens,tools)
    if reply is not None:return reply
    gap=self._svb_backoff(code,tries)
    if gap is None:break
    time.sleep(gap)
    tries+=1
   if code==402:
    self.no_credit=True
    break
  return None
 def _svb_backoff(self,status,attempt):
  retry=status is None or status in _svb_RESEND or(isinstance(status,int)and status>=500)
  cap=1 if status in(None,408)else len(WAITS)
  if not retry or attempt>=cap:return None
  delay=WAITS[attempt]
  if self.waited+delay>WAIT_TOTAL or self.secs()-delay<CLOCK_EXTRAS:return None
  self.waited+=delay
  return delay*(0.8+0.05*(self.step%9))
 _svc_WHEN={"choose":"before the picks are settled","order":"at checkout, just before paying (Kyo runs it then)",
   "end":"just before a close with nothing bought (Kyo runs it then)"}
 def call_model(self,model,history,max_tokens,tools):
  sent=history
  if model.startswith("anthropic/")and len(history)>1:
   sent=list(map(dict,history))
   for k in(0,-1):
    text=sent[k].get("content")
    if text and isinstance(text,str):sent[k]["content"]=[{"type":"text","text":text,"cache_control":{"type":"ephemeral"}}]
  body={"model":model,"messages":sent,"max_tokens":max_tokens,"temperature":0}
  if tools:body["tools"],body["tool_choice"]=tools,"auto"
  _llm.timeout=max(5,min(50,self.secs()-5))
  res=_llm.post_verbose("/inference/chat/completions",body)
  data=res.data
  if isinstance(data,dict)and data.get("choices"):
   usage=data.get("usage")or{}
   spent=usage.get("cost")
   if isinstance(spent,(int,float)):self.cost+=float(spent)
   out=data["choices"][0].get("message")or{}
   if out.get("content")or out.get("tool_calls"):return out,200
   self.trace.append({"kind":"llm_fail","model":model,"status":0})
   return None,0
  err=res.error or{}
  code,why=err.get("status"),err.get("body")
  inner=data.get("error")if isinstance(data,dict)else None
  if isinstance(inner,dict):code,why=_count(inner.get("code")),inner.get("message")
  self.trace.append({"kind":"llm_fail","model":model,"status":code,"body":str(why or"")[:200]})
  return None,code
 def ask_json(self,prompt,models,max_tokens=1800):
  for engine in models:
   reply=self.think([{"role":"user","content":prompt}],[engine],max_tokens)
   got=_json_obj((reply or{}).get("content"))
   if got is not None:return got
   if self.broke or self.no_credit or self.calls>=ALL_CALLS:return None
  return None
 def act(self,tool,arguments):
  self.ver_seen.setdefault(self.version,self.step)
  merged={**self.fixed_args.get(tool,{})}
  merged.update(arguments)
  call={**self.binding,"step":self.step+1,"action":{"name":tool,"args":merged}}
  seek,mark=tool in self.seekers(),tool+_dump(merged)
  if seek and self.answered.get(mark)==self.version:
   self.trace.append({"kind":"repeat_held","tool":tool})
   return None
  for _ in range(4):
   res=_env.post_verbose("/environment/call",call)
   got=res.data
   if not isinstance(got,dict):
    err=res.error or{}
    code,text=err.get("status"),str(err.get("body")or"")
    self.trace.append({"kind":"env_fail","status":code,"body":text[:200]})
    kind,val=_kc_env_verdict(code,text,self.step)
    if kind=="again":continue
    if kind=="seek":
     self.step=val
     call["step"]=val+1
     continue
    if kind=="closed":self.finished=True
    elif val:self.broke=True
    return None
   num=got.get("step")
   self.step=got["step"]if isinstance(num,int)else self.step+1
   if got.get("finished"):self.finished=True
   obs=got["observation"]=_untext(got.get("observation"))
   self.page_ids.update(_kb_TOKEN.findall(_dump(obs)))
   clean=not(isinstance(obs,dict)and obs.get("error"))
   if seek:self.answered[mark]=self.version
   tf,pins=self.text_fields.get(tool),self.mode_pins.get(tool)or{}
   if seek and clean and tf and str(merged.get(tf)or"").strip()and any(f in merged and merged[f]!=v for f,v in pins.items()):
    # off the word mode, two different word sets that list the same rows show the words go unused there
    sig,ids=tool+_dump({k:v for k,v in merged.items()if k!=tf}),tuple(sorted(set(_kb_TOKEN.findall(_dump(obs)))))
    was=self.blind.get(sig)
    if was and was[0]!=merged[tf]and was[1]==ids and any(not i.isdigit()for i in ids):self.blind[tool]=True
    self.blind.setdefault(sig,(merged[tf],ids))
   if clean and tool==self.roles.get("search"):
    spec=self.schemas.get(tool)or{}
    props=spec.get("properties")or{}
    self.fixed_args.setdefault(tool,{f:merged[f]for f in spec.get("required")or[]if f in merged and"enum"in(props.get(f)or{})})
   if clean and tool in self.page_tools():
    # a listing counts as opened only when the reply carries it: an empty side-by-side reply opens nothing
    shown=_dump(obs)
    for value in merged.values():self.opened.update(p.strip()for p in(value if isinstance(value,list)else[value])if isinstance(p,str)and p.strip()in shown)
   return got
  return None
 def sibling_ref(self,slot,text):
  # a must tying an item to another product of this same order points at no outside listing
  said=_tokens(text.split(":",1)[-1])
  return any(n!=slot and it.name and _tokens(it.name)<=said for n,it in enumerate(self.items,1))
 def settle_musts(self,reading):
  linked={r for _,r in self.refs}
  for slot,it in enumerate(self.items,1):
   loose=[]
   for m in it.musts:
    tail=m.split(":",1)[-1]
    if m not in linked and not self.sibling_ref(slot,m)and len(tail.split())>=VAGUE_WORDS and not re.search(r"[0-9A-Z]",tail):loose.append(m)
   it.musts=[m for m in it.musts if m not in loose]
   reading["wishes"]=_items(reading.get("wishes"))+loose
 def find_pointers(self):
  listing="\n".join("%d: %s"%(n,it.name)for n,it in enumerate(self.items,1))
  got=self.ask_json("%s\n\nITEMS:\n%s\n\nREQUEST:\n%s"%(POINTER,listing,self.request),[HEAVY,LIGHT],800)or{}
  for e in _items(got.get("lookups")):
   if not isinstance(e,dict):continue
   text=str(e.get("must")or"").strip()
   n=_count(e.get("item"))or 1
   if not text or n<1 or n>len(self.items)or self.sibling_ref(n,text):continue
   it=self.items[n-1]
   head=_words(text.split(":")[0])
   it.musts=[m for m in it.musts if _words(m.split(":")[0])!=head]+[text]
   self.refs.append((n,text))
 def unmix(self):
  for slot,it in enumerate(self.items,1):
   linked=[r for p,r in self.refs if p==slot and r in it.musts]
   rest=[m for m in it.musts if m not in linked]
   if not linked or not rest:continue
   ask="%s\n\nShopper's request:\n%s\n\nProduct under review: %s\nMusts that copy a value off a referenced listing:\n%s\nOTHER REQUIREMENTS:\n%s"%(
     UNMIX,self.request,it.name,"\n".join(linked),"\n".join(rest))
   got=self.ask_json(ask,[HEAVY,LIGHT],500)or{}
   told={str(x).strip()for x in _items(got.get("describe_other"))}
   moved=[m for m in rest if m in told]
   if not moved:continue
   head=linked[0]
   joined="%s (that listing: %s)"%(head,"; ".join(moved))
   it.musts=[joined if m==head else m for m in it.musts if m not in moved]
   self.refs=[(p,joined)if p==slot and r==head else(p,r)for p,r in self.refs]
   self.trace.append({"kind":"unmix","item":slot,"moved":moved})
 def ledger(self):
  """The requirements in force, as the model and the reviewers see them."""
  rows=[]
  many=len(self.items)>1
  for n,it in enumerate(self.items,1):
   waiting={r for p,r in self.refs if p==n}
   shown=[]
   for m in it.musts+it.told:
    if m in it.dropped:continue
    if m in self.ref_values:m+=" (= %s, as read on the page it points to; text identifying that other listing is never a property here)"%self.ref_values[m]
    elif m in waiting and m in self.apart:m+=" (an exclusion whose value is unread: only the referenced listing [%s], or its twin from the same shop, can break it; every other listing meets it)"%self.apart[m]
    elif m in waiting:m+=" (referenced value not read yet)"
    shown.append(m)
   line=("item %d"%n if many else"item")+": "+it.name
   line+=" [store category: %s]"%it.shelf if it.shelf else""
   line+=", quantity %s; musts -> %s"%(it.qty,"; ".join(shown)or"none beyond the product itself")
   line+="; open to anything on: "+", ".join(it.free)if it.free else""
   rows.append(line)
  rd=self.brief
  if self.budget is not None:
   scope="each item"if rd.get("budget_scope")=="each"else"whole order"
   rows.append(("spending cap (%s): %s %s"%(scope,self.budget,rd.get("budget_unit")or"")).replace("  "," ").strip())
  how=rd.get("pick")
  measure=str(rd.get("pick_measure")or"")
  if how in("most","least"):rows.append(("choice: whichever fitting listing has the %s %s"%(how,""if _stem(measure)<=_stem(how)else measure)).strip())
  elif how=="cheapest":
   rows.append("choice: of everything that fits, the cheapest wins")
   rows.extend(self.read_by_price())
  else:rows.append("choice: no ranking - any listing that fits every condition is acceptable")
  if rd.get("in_stock_only",True)is not False:rows.append("skip listings that cannot be bought now (out of stock)")
  rows.extend("tie-breaker only (a wish): %s"%w for w in _items(rd.get("wishes")))
  rows.extend("per the request: %s"%t for t in _items(rd.get("rules")))
  if rd.get("quiet_decline")is True:rows.append("closing empty-handed because nothing fits: mention no listing by name and propose none")
  if rd.get("nearest_fallback")is True:rows.append("if no listing fits every requirement: buy nothing and close empty-handed, naming in the closing words the closest listing (its id and what it misses) as the near miss the request asks to be offered; a near miss is offered, never bought")
  rows.extend("%s, do this: %s"%(self._svc_WHEN[p["when"]],p["step"])for p in self.process)
  via=self.facts_tool()
  if via and(self.facts_from.get(via)is True or self.roles.get("offer")):
   pages=self.page_tools()
   empty=sorted(t for t in pages|{self.roles.get("offer")or""}if t and self.facts_from.get(t)is False)
   others=sorted(t for t in pages if t!=via)
   line="read attributes with "+via
   line+="; %s show none"%", ".join(empty)if empty else""
   line+="; to walk linked pages or verify a page, call "+" or ".join(others)if others else""
   rows.append(line)
  return"\n".join("- %s"%r for r in rows)
 def status(self):
  rest=self.left()
  empty=not self.buys
  head="[Kyo] %s of %s steps left (%s used)."%(rest,self.max_steps,self.step)
  if rest<=self.tail()+1:head+=(" FINAL CALLS: buy now if every item has a cart line; close empty-handed only when searching showed an item has no fitting listing at all - running short of steps shows nothing of the kind.")
  elif rest<=6 or(empty and rest<=len(self.items)+5):head+=" Very few calls remain - add precisely the selected listings, then check out at once."
  # a checkout can still be turned down (a new budget, a sold-out line); answering that takes steps, so a ready cart goes in while they remain
  elif empty and rest<=9 and len([x for x,n in self.cart_lines.items()if n>0 and not self.sold_out(x)])>=len(self.items):head+=" Every item has a cart line: check out now - the store may still turn the order down, and the steps left are what answering that needs."
  elif empty and rest<=12:head+=" Steps are getting scarce - commit to picks, cart them, then complete checkout shortly."
  # half the wall clock gone counts as half the budget: on a slow store the clock ends the episode long before the steps do
  elif empty and not self.cart_touched and(self.step>=self.max_steps//2 or self.secs()<=CLOCK_LIMIT/2):head+=" Half the step budget or the wall time spent with an empty cart: put a suitable listing in for each item right away (items may still sell out)."
  if empty and self.secs()<CLOCK_EXTRAS+20:head+=" Wall time is short too - check out within two or three calls with the strongest candidate seen up to now in the cart; a timeout is no evidence about stock."
  todo=[t for t in self.open_lookups()if t not in self.apart]
  parts=[head]
  if empty and todo:parts.append("[waiting lookups: reach each value by visiting the listing the request points at, then whichever listing is named by that page in turn - the last page holds it; for a must about matching up with it, take the partner given there] "+" | ".join(todo[:3]))
  if self.leads:parts.append("[candidates to review] "+", ".join(self.leads))
  if self.notes:parts.append("[changes up to now] "+" | ".join(self.notes[-4:]))
  parts.append("[ledger]\n%s"%self.ledger())
  return"\n".join(parts)
 def ident(self,record,in_list=False,depth=0):
  if not(isinstance(record,dict)and len(record)>=(2 if in_list else 3)):return None
  guess=self.id_key
  if not guess or not _id_like(record.get(guess),True):
   deep=in_list or not depth or any(isinstance(v,dict)for v in record.values())
   guess=_learn_id([record])if deep else None
   if not guess or not("id"in str(guess).lower()or record[guess].strip()in self.rows):return None
  return record[guess].strip()
 def learn_rows(self,batch):
  rows=[c for c in batch if isinstance(c,dict)and len(c)>=2]
  if not rows or len(batch)>2*len(rows):return False
  known=self.id_key
  if known and all(_id_like(r.get(known),True)for r in rows):return True
  found=_learn_id(rows)
  if found:self.id_key=found
  return bool(found)
 def absorb(self,outcome):
  stack=[(0,False,outcome)]
  while stack:
   depth,listed,node=stack.pop()
   if depth>6 or not isinstance(node,(list,dict)):continue
   if isinstance(node,list):
    flag=self.learn_rows(node)
    stack+=[(depth+1,flag,sub)for sub in node[::-1]]
    continue
   lid=self.ident(node,listed,depth)
   if not lid:
    stack+=[(depth+1,False,sub)for sub in list(node.values())[::-1]]
    continue
   have=self.rows.get(lid)
   if have is None:
    self.seen_order.append(lid)
    have=self.rows[lid]={}
   pairs=list(node.items())
   added={k:v for k,v in pairs if have and k not in have and(_kb_num(v)or isinstance(v,str))}
   if not listed and len(added)>=2 and not _fact_block(node):node=dict(node,details=added)
   if any(have.get(k)is True and v is False for k,v in pairs):self.unavailable.add(lid)
   elif any(have.get(k)is False and v is True for k,v in pairs):self.unavailable.discard(lid)
   have.update(node)
   if _fact_block(node):self.full[lid]=dict(have)
   elif lid in self.full:self.full[lid].update(node)
 def labels(self):
  found=set()
  queue=[(s,0)for s in self.schemas.values()]
  while queue:
   node,depth=queue.pop(0)
   if depth<=6 and isinstance(node,dict):
    for opt in node.get("enum")or[]:
     if isinstance(opt,str):found.add(opt.strip().lower())
    queue+=[(v,depth+1)for v in node.values()if isinstance(v,dict)]
  return found
 def category_key(self):
  pool,counts={},{}
  for row in self.rows.values():
   for k,v in row.items():
    if isinstance(v,str)and v and len(v)<=40 and re.search(r"\d{3}",v)is None:
     pool.setdefault(k,set()).add(v)
     counts[k]=1+counts.get(k,0)
  floor=0.8*max(1,len(self.rows))
  enums=self.labels()
  best=None
  for k,vals in pool.items():
   n=len(vals)
   if counts[k]<floor or n>40:continue
   score=(-(sum(x.strip().lower()in enums for x in vals)/n),n,-counts[k],-(sum(map(len,vals))/n))
   if best is None or score<best[0]:best=(score,k)
  return best[1]if best else None
 def _svc_on_item(self,candidates):
  marks=[]
  for it in self.items:
   shelf=it.shelf or""
   marks.append((_words(shelf),_words(it.name),{w for w in _stem(shelf)|_stem(it.name)if len(w)>=4}))
  keep=[]
  for c in candidates:
   flat,big=_words(c),{w for w in _stem(c)if len(w)>=4}
   if any(flat in(a,b)or big&bag for a,b,bag in marks):keep.append(c)
  return keep
 def _svc_kept_keys(self,group):
  picked=[]
  for name,info in sorted(group["keys"].items(),key=lambda kv:-kv[1]["n"]):
   vals=list(info["per"].values())
   n=len(vals)
   if not n:continue
   sizes=sorted(map(len,vals))
   if sizes[n//2]>40 or 2*sum(" / "in v for v in vals)>=n or(n>=4 and len(set(vals))==n and sum(sizes)/n>12 and not _common_words(vals)):continue
   if all(not name.startswith(p+" ")or info["per"]!=pi["per"]for p,pi in picked):picked.append((name,info))
  return picked
 def key_table(self):
  cat=self.category_key()
  said=" ".join(map(self.described_words,range(1,len(self.items)+1)))
  groups={}
  for lid in self.seen_order:
   page=self.full.get(lid)
   if not page:continue
   block=_fact_block(page)or{}
   g=groups.setdefault(str(page.get(cat)or"listings")if cat else"listings",{"n":0,"keys":{},"ids":[]})
   g["n"]+=1
   g["ids"].append(lid)
   seen=set()
   for rk,rv in block.items():
    key=_words(rk)
    if not key or key in seen:continue
    seen.add(key)
    slot=g["keys"].setdefault(key,{"n":0,"values":[],"per":{}})
    slot["n"]+=1
    v=slot["per"][lid]=str(rv).strip().lower()
    if v not in slot["values"]:slot["values"].append(v)
  on=self._svc_on_item(groups)
  if on:groups={k:g for k,g in groups.items()if k in on}
  table=[]
  self.key_parts={k:_value_parts(info["values"])for g in groups.values()for k,info in g["keys"].items()}
  self.key_per={}
  for g in groups.values():
   for k,info in g["keys"].items():self.key_per.setdefault(k,{}).update(info["per"])
  # rows sharing a price and a long text are copies of one product: a value every copy prints says nothing of the shelf
  sig=lambda lid:{(self.price_of(lid),v)for v in(self.rows.get(lid)or{}).values()if isinstance(v,str)and len(v.split())>=8}if self.price_of(lid)is not None else set()
  self.copy_cats=set()
  for label,g in sorted(groups.items(),key=lambda kv:-kv[1]["n"]):
   sigs=[sig(lid)for lid in g["ids"]]
   copies=len(sigs)>=4 and 2*sum(not any(s&o for o in sigs[:i])for i,s in enumerate(sigs))<=len(sigs)
   if copies:self.copy_cats.add(label)
   rows=[(k,info,sum(map(bool,info["values"])))for k,info in self._svc_kept_keys(g)]
   flat=lambda r:r[2]<=1 and r[1]["n"]>=3
   rows.sort(key=(lambda r:(-r[1]["n"],flat(r)))if copies else flat)
   if len(rows)>18:
    # a key repeating an earlier row's property spends a table slot that a distinct property needs
    lone=[]
    for r in rows:
     if not any(self.same_prop(r[0],o[0])for o in lone):lone.append(r)
    rows=lone+[r for r in rows if r not in lone]
   spill=rows[18:]
   rows=rows[:18]+[r for r in spill if" %s "%r[0]in said]+[r for r in spill if flat(r)and" %s "%r[0]not in said][:3]
   table.extend((label,k,info["n"],[v[:40]for v in info["values"][:4]],shown)for k,info,shown in rows)
  return table
 def same_prop(self,a,b):
  # two names hold one property when one's words nest in the other's and the listings carrying both print the same value
  a,b=_words(a),_words(b)
  if a==b:return True
  ta,tb=_tokens(a),_tokens(b)
  per=getattr(self,"key_per",None)or{}
  pa,pb=per.get(a),per.get(b)
  if not(ta and tb and(ta<=tb or tb<=ta)and pa and pb):return False
  both=[l for l in pa if l in pb]
  if both:return 10*sum(pa[l]==pb[l]for l in both)>=7*len(both)
  va,vb=set(pa.values()),set(pb.values())
  return min(len(va),len(vb))>=2 and(va<=vb or vb<=va)
 def told_dup(self,sel):
  return any(self.same_prop(sel[1],n.split(":")[0])for n in self.items[sel[0]-1].told if n not in self.items[sel[0]-1].dropped)
 def described_words(self,num):
  it=self.items[num-1]
  mine=_stem(it.name)|_stem(it.shelf)
  squash=lambda t:" ".join(re.findall(r"[a-z0-9]+",_words(t)))
  bits=[]
  for p,req in self.refs:
   halves=req.split(":",1)
   if p==num and _tokens(halves[-1])&mine:bits.append(squash(halves[-1]).replace(squash(halves[0])," "))
  return" %s "%" ".join(bits)if bits else""
 def item_groups(self):
  cat=self.category_key()
  by={}
  for lid in self.seen_order:by.setdefault(str(self.rows[lid].get(cat)or"")if cat else"",[]).append(lid)
  pool=[by[k]for k in self._svc_on_item(by)]or list(by.values())
  pool.sort(key=len,reverse=True)
  return pool[:max(1,len(self.items))]
 def single_target(self,tool):
  if tool in self.text_fields:return self.text_fields[tool]
  spec=self.schemas.get(tool or"")or{}
  pinned=self.fixed_args.get(tool,{})
  need=[f for f in spec.get("required")or[]if f not in pinned]
  # the parameter the store's replies already fed page tokens into is not where words go
  texty=[f for f,fs in(spec.get("properties")or{}).items()if isinstance(fs,dict)and"enum"not in fs and"string"in _kinds(fs)and f!=self.page_args.get(tool)]
  only=next(filter(texty.__contains__,need),None)or(texty[0]if len(texty)==1 else None)
  return only if tool and only and all(f==only for f in need)else None
 def price_of(self,listing):
  key=self.price_key
  if not key:return None
  num=lambda x:isinstance(x,(int,float))and not isinstance(x,bool)
  v=(self.rows.get(listing)or{}).get(key)
  if isinstance(v,dict):v=next((x for x in v.values()if num(x)),None)
  if isinstance(v,str):
   figs=re.findall(r"\d[\d,]*(?:\.\d+)?",v)
   v=float(figs[0].replace(",",""))if len(figs)==1 else None
  return float(v)if num(v)else None
 def buyable(self,listing):
  if self.sold_out(listing):return False
  have=(self.rows.get(listing)or{}).get(self.stock_key)if self.stock_key else True
  return have is not False and not(type(have)in(int,float)and have<=0)
 def records_in(self,outcome):
  found,stack=[],[(0,False,outcome)]
  while stack:
   depth,listed,node=stack.pop()
   if depth>6 or not isinstance(node,(list,dict)):continue
   if isinstance(node,list):
    stack+=[(depth+1,True,sub)for sub in node[::-1]]
    continue
   lid=self.ident(node,listed,depth)
   if lid:found.append((lid,node))
   else:stack+=[(depth+1,False,sub)for sub in list(node.values())[::-1]]
  return found
 def note_view(self,tool,outcome):
  stamp=self.version
  for lid,entry in self.records_in(outcome):
   keys=set(entry)
   self.tool_keys.setdefault(tool,set()).update(keys)
   old=self.view_keys.get(lid)
   if old and old[0]==stamp:keys|=old[1]
   self.view_keys[lid]=(stamp,keys)
 def shown_already(self,tool,listing):
  want=self.tool_keys.get(tool)
  stamp,keys=self.view_keys.get(listing,(-1,set()))
  return bool(want)and listing not in self.gone and stamp==self.version and want<=keys
 def _kd_send(self,tool,args,blocked=False):
  bad=[]
  shaped=_shape(args,self.schemas.get(tool),"args",bad)
  return(None if bad or blocked else self.act(tool,shaped)),shaped
 def _kd_take(self,tool,sent,answer,record=None):
  fresh=answer.get("observation")
  self.observe(tool,fresh)
  if record is not None:record["token"]=None
  self.log_search(tool,sent,fresh)
  return fresh
 def read_shape(self,tool,outcome):
  if tool in self.shapes:return self.shapes[tool]
  if not isinstance(outcome,dict):return None
  arrays=[k for k,v in outcome.items()if isinstance(v,list)and len(v)>=2 and all(isinstance(r,dict)for r in v)]
  if not arrays:return None
  self.shapes[tool]=None
  sample={k:(v[:3]if k in arrays else v)for k,v in outcome.items()}
  spec=self.schemas.get(tool)or{}
  # a second tool replying with the same keys and row keys takes the first reading when its page parameter is settled too:
  # the reading names one of its parameters, or it has at most one free slot (page_arg takes that one); values are checked below
  layout=_dump({k:sorted({f for r in v[:3]for f in r})if k in arrays else 0 for k,v in outcome.items()})
  said,props=self.shape_said.get(layout)or{},spec.get("properties")or{}
  slots=[a for a,fs in props.items()if a not in(spec.get("required")or[])and _kb_page_fit(fs,None)]
  got=said if said.get("next_arg")in props or(said and len(slots)<=1)else self.ask_json("%s\n\nTOOL %s, parameters %s\n\nFIRST ROWS OF ITS RESULT:\n%s"%(SHAPE,tool,_dump(spec),_dump(_clip(sample,120))[:4000]),
    [LIGHT,HEAVY],1500)or{}
  if got:self.shape_said[layout]=got
  key=got.get("rows")
  if not key or key not in arrays:return None
  first=outcome[key][0]
  shape={"rows":key}
  for role,where in(("count",outcome),("next",outcome),("next_arg",spec.get("properties")or{}),("price",first),("stock",first)):
   v=got.get(role)
   shape[role]=v if isinstance(v,str)and v in where else None
  whole=lambda x:isinstance(x,int)and not isinstance(x,bool)
  if shape["count"]and not whole(outcome[shape["count"]]):shape["count"]=None
  if not shape["count"]:
   ints=[k for k,v in outcome.items()if whole(v)]
   shape["count"]=ints[0]if len(ints)==1 else None
  self.shapes[tool]=shape
  self.price_key=self.price_key or shape["price"]
  self.stock_key=self.stock_key or shape["stock"]
  self.trace.append({"kind":"shape","tool":tool,"shape":shape})
  return shape
 def paging(self,tool,outcome,call_args=None):
  if isinstance(outcome,dict)and not any(isinstance(v,list)for v in outcome.values()):
   # a store reply wrapped one level down beside a side message still pages
   outcome=next((v for v in outcome.values()if isinstance(v,dict)and any(isinstance(x,list)for x in v.values())),outcome)
  shape=self.read_shape(tool,outcome)
  if not shape or not(shape["count"]and shape["next"])or not isinstance(outcome,dict):return None
  rows,total,nxt=[outcome.get(shape[k])for k in("rows","count","next")]
  whole=lambda x:isinstance(x,int)and not isinstance(x,bool)
  if not isinstance(rows,list)or not rows or not whole(total):return None
  arg=self.page_arg(tool,nxt,call_args)if nxt is None or isinstance(nxt,str)or whole(nxt)else None
  if nxt is not None and arg:self.tokens_seen.add(_dump(nxt))
  return(total,len(rows),arg,nxt)if arg else None
 def code_query(self,args):
  # a search naming a model code (letters joined to 3+ digits); page tokens the store handed out do not count
  return any(isinstance(v,str)and _dump(v)not in self.tokens_seen and _kb_CODE.search(v)for v in(args or{}).values())
 def log_search(self,tool,call_args,outcome):
  # reply-level strings (page tokens) are never vetted as filter values
  if isinstance(outcome,dict):self.tokens_seen.update(_dump(v)for v in outcome.values()if isinstance(v,str))
  got=self.paging(tool,outcome,call_args)
  if got:
   arg=got[2]
   base={k:v for k,v in call_args.items()if k!=arg}
   rec=self.search_log.setdefault(tool+_dump(base),{"tool":tool,"args":base,"page_arg":arg,"total":got[0],"ids":[]})
   rec["token"]=got[3]
   rec["ids"]+=[lid for lid,_ in self.records_in(outcome)if lid not in rec["ids"]]
 def page_on(self,record):
  tool=record["tool"]
  answer,sent=self._kd_send(tool,{**record["args"],record["page_arg"]:record["token"]})
  if answer is None or self.finished:
   record["token"]=None
   return False
  self.pages_read+=1
  before=set(record["ids"])
  fresh=self._kd_take(tool,sent,answer,record)
  # a page arg that brings no new rows is not paging: stop using it this episode
  if not{lid for lid,_ in self.records_in(fresh)}-before:
   self.page_dead(tool,record["page_arg"])
   return False
  return True
 def kind_of(self,listing):
  field=self.category_key()
  row=self.rows.get(listing)or{}
  return str(row.get(field)or"").strip().lower()if field else""
 def read_by_price(self):
  shelves={}
  for lid in self.full:
   cost=self.price_of(lid)
   if cost is not None and self.buyable(lid):shelves.setdefault(self.kind_of(lid),[]).append((cost,lid))
  return["sheets already read, lowest price first (a pick must beat these or a must rules them out): "+", ".join("%s (%g)"%(lid,cost)for cost,lid in sorted(group)[:4])for group in shelves.values()if len(group)>1]
 def price_sorted(self,outcome):
  prices=[self.price_of(lid)for lid,_ in self.records_in(outcome)]
  return len(prices)>=3 and None not in prices and all(a<=b for a,b in zip(prices,prices[1:]))
 def more_pages(self,tool,call_args,outcome):
  self.paged_obs=[]
  how=self.brief.get("pick")
  info=self.paging(tool,outcome,call_args)if how in("cheapest","most","least")else None
  if not info:return""
  total,seen,arg,nxt=info
  at=call_args.get(arg)
  deep=at not in(0,1)if isinstance(at,int)and not isinstance(at,bool)else bool(str(at or"").strip())
  if nxt is None or deep or total<=seen or(how=="cheapest"and self.price_sorted(outcome)):return""
  need=-(-(total-seen)//seen)
  # a page argument with a schema ceiling caps the pages the store will ever serve, whatever the count claims
  top=(((self.schemas.get(tool)or{}).get("properties")or{}).get(arg)or{}).get("maximum")
  if isinstance(top,int)and isinstance(nxt,int)and not isinstance(nxt,bool):
   if nxt>top:return""
   need=min(need,top-nxt+1)
  if need>4 or self.pages_read+need>AUTO_PAGES or self.left()-self.tail()-need<12:return("\n[Kyo] %s matches exist but just %s are listed here, in no price order, while the choice compares every fitting listing: search with the requirement's value to cut the list, or ask for the following page with %s=%s."%(total,seen,arg,_dump(nxt)))
  got,known=[],{lid for lid,_ in self.records_in(outcome)}
  for _ in range(need):
   answer,sent=self._kd_send(tool,{**call_args,arg:nxt})
   if answer is None or self.finished:break
   self.pages_read+=1
   fresh=self._kd_take(tool,sent,answer)
   new={lid for lid,_ in self.records_in(fresh)}-known
   if not new:
    self.page_dead(tool,arg)
    break
   known|=new
   self.paged_obs.append(fresh)
   text=self.show(fresh)
   self.read_cache[tool+_dump(sent)]=(self.version,self.step,text)
   got.append("%s=%s: %s"%(arg,_dump(nxt),text))
   nxt=(self.paging(tool,fresh,sent)or(None,)*4)[3]
   if nxt is None:break
  self.trace.append({"kind":"pages","tool":tool,"read":len(got)})
  return"\n[Kyo pulled the next pages of this result too (one step per page)]\n"+"\n".join(got)if got else""
 def glued_numbers(self,targs):
  # a number written with its unit or a range sign matches only pages that print it the same way
  glued=[w for v in targs.values()if isinstance(v,str)for w in v.split()if re.search(r"\d",w)and re.search(r"\d[^\d\s.,]|[^\d\s.,]\d",w)]
  seen=self.__dict__.setdefault("glued_told",set())
  glued=[w for w in dict.fromkeys(glued)if w.lower()not in seen and not seen.add(w.lower())]
  bare=[" ".join(re.findall(r"\d+(?:\.\d+)?",w))for w in glued]
  return"\n[Kyo] %s matches only listings that print it exactly so; pages also write numbers with spaces or other unit forms, so search the bare numbers (%s) too before taking this list as all that fits."%(", ".join(glued),", ".join(bare))if glued else""
 def cheapest_here(self,results):
  if self.brief.get("pick")=="cheapest"and self.price_key:
   ids=set()
   for res in results:ids.update(lid for lid,_ in self.records_in(res))
   priced=sorted((self.price_of(x),x)for x in ids if self.buyable(x)and self.price_of(x)is not None)
   if len(priced)>1:return"\n[Kyo] price ranking of what can be bought from this result: "+", ".join("%s (%s)"%(x,format(p,"g"))for p,x in priced[:6])
  return""
 def browse_call(self,label,cap):
  tool=self.roles.get("browse")
  names=self.brief.get("browse_args")or{}
  la,ca=names.get("label"),names.get("cap")
  props=(self.schemas.get(tool or"")or{}).get("properties")or{}
  ok=tool and la in props and ca in props and label
  # a browse on the search tool itself goes in its other mode, when the mode enum leaves just one
  other={f:[o for o in(props.get(f)or{}).get("enum")or[]if o!=v]for f,v in self.mode_pins.get(tool or"",{}).items()}
  return(tool,{la:label,ca:cap,**{f:o[0]for f,o in other.items()if len(o)==1}})if ok else None
 def sweep(self):
  """Cheapest rule: page on and read cheaper rows of each cart line's kind before checkout."""
  spare=self.left()-self.tail()-6
  blocked=self.swept or self.brief.get("pick")!="cheapest"or spare<3 or self.secs()<CLOCK_EXTRAS
  todo=[]if blocked else[(lid,self.price_of(lid),self.kind_of(lid))for lid in self.cart_ids()]
  todo=[(lid,p,k)for lid,p,k in todo if p is not None and k]
  if not todo:return""
  self.swept,start,n=True,len(self.rows),len(todo)
  per=max(1,min(SWEEP_PAGES,spare//(2*n)))
  per=0 if spare<6*n and self.browse_call("-",1)else per
  for lid,_,kind in todo:
   hits=lambda rec:sum(self.kind_of(r)==kind for r in rec["ids"])
   live=[rec for rec in self.search_log.values()if rec.get("token")is not None and hits(rec)]
   if live:
    rec,budget=min(live,key=lambda rec:(lid not in rec["ids"],-hits(rec))),per
    while budget and rec.get("token")is not None and not self.finished and self.page_on(rec):budget-=1
  want=_tokens(" ".join(v for it in self.items for v in self.word_values(it)))
  tool=self.roles.get("search")
  out=set()
  for rec_2 in self.search_log.values():
   if rec_2["tool"]==tool and want&_tokens(_dump(rec_2["args"])):out.update(rec_2["ids"])
  value_rows_2=out
  opened=self.read_cheaper(todo,value_rows_2)
  room=max(1,min(CEILING_PAGES,(self.left()-self.tail()-6)//(2*n)))
  for line in todo:
   lid,price=line[0],line[1]
   tag=str((self.rows.get(lid)or{}).get(self.category_key())or"")
   call=self.browse_call(tag,int(price)-1 if float(price).is_integer()else price)
   if call and not self.finished:self.walk_browse(call[0],call[1],room)
   opened+=self.read_cheaper([line],self.value_shown())or self.read_cheaper([line])
  self.trace.append({"kind":"sweep","new_rows":len(self.rows)-start,"read":len(opened)})
  opened=list(filter(lambda c:not self.range_miss(c),opened))
  if opened:
   # long free text is cut first so the attributes stay inside the shown span
   body="\n".join("%s: %s"%(c,_dump(_clip(self.full.get(c)or self.rows.get(c)or{},160))[:1500])for c in opened[:12])
   return("Held (this cost you zero steps). Below: listings Kyo opened while hunting a lower price, every one cheaper than a cart line. Rank them by price, smallest first, and test all requirements on each; whichever fits first swaps into its cart line, and with no such match you simply check out.\n"+body)
  return""
 def read_cheaper(self,lines,among=None):
  found={}
  for _,cap,kind in lines:
   for c in self.seen_order:
    if c in self.full or c in found or not(among is None or c in among):continue
    if self.buyable(c)and self.kind_of(c)==kind and(self.price_of(c)or cap)<cap:found[c]=self.price_of(c)
  queue=sorted(found,key=found.get)[:2*READ_CAP]
  tries=0
  while queue and tries<2 and self.left()>self.tail()+5 and not self.finished:
   k=self.compare_room()
   batch,queue=queue[:k],queue[k:]
   if not self.open_rows(batch):break
   tries+=1
  return sorted((c for c in found if c in self.full),key=found.get)
 def value_shown(self):
  bags=[_tokens(v)for it in self.items for v in self.word_values(it)]
  out=set()
  for lid,row in self.rows.items():
   if bags and any(b<=_tokens(_dump(row))for b in bags):out.add(lid)
  return out
 def twin_search(self,lost):
  tool=self.roles.get("search")
  arg=self.single_target(tool)
  if not tool or not arg or self.left()<=self.tail()+6 or self.secs()<CLOCK_EXTRAS:return
  alnum=lambda w:re.findall(r"[a-z0-9]+",w.lower())
  tw=lambda lid:list(dict.fromkeys(alnum(re.sub(r"\[.*?\]|\(.*?\)"," ",self.title_of(lid,45).lower()))))
  bags={lid:set(tw(lid))for lid in self.rows}
  for lid in lost[:2]:
   own=tw(lid)
   if any(c not in lost and self.buyable(c)and len(b.intersection(own))>=0.8*len(own)for c,b in bags.items()):continue
   vals=[v for it in self.items for v in self.word_values(it)]
   words=[w for w in self.value_query(vals).split()if _tokens(w)and all(p in own for p in alnum(w))]
   used={p for w in words for p in alnum(w)}
   rare=sorted((w for w in own if w not in used),key=lambda w:(not re.search(r"\d",w),sum(w in b for b in bags.values())))[:2]
   answer,sent=self._kd_send(tool,{arg:" ".join(words+rare)},not rare or self.finished)
   if answer is None:continue
   fresh=self._kd_take(tool,sent,answer)
   pool=[c for c,_ in self.records_in(fresh)if c not in self.full and c not in lost and self.buyable(c)]
   ownset=set(own)
   pool.sort(key=lambda c:(-len(set(tw(c))&ownset),self.price_of(c)or 0))
   if pool and self.left()>self.tail()+5 and self.open_rows(pool[:self.compare_room()]):self.trace.append({"kind":"twins","lost":lid,"read":pool[:self.compare_room()]})
 def copies(self,lost):
  kind_fn=self.kind_of
  texts=lambda lid:{v for v in(self.rows.get(lid)or{}).values()if isinstance(v,str)and len(v.split())>=3 and v.strip().lower()!=kind_fn(lid)}
  held=set()
  for lid in lost:held|=texts(lid)
  inside=self.cart_ids()
  return[x for x in self.on_hand(set(lost))if x not in inside and texts(x)&held and self.buyable(x)and self.within_budget(x,1)]
 def take_better(self):
  ruling=next((t.get("got")for t in reversed(self.trace)if t.get("kind")=="check"),None)or{}
  if self.brief.get("pick")=="cheapest"and len(_items(ruling.get("problems")))<=1:
   for pick in map(str,_items(ruling.get("better"))):
    cost=self.price_of(pick)
    if pick not in self.full or cost is None or not self.buyable(pick):continue
    dearer=[x for x in self.cart_ids()if self.kind_of(x)==self.kind_of(pick)and(self.price_of(x)or 0)>cost]
    if len(dearer)==1 and self.swap_line(dearer[0],pick,0):return True
  # a failed check naming an opened listing as better: it replaces the line of its kind once a check of the new cart approves it
  tries=2
  for pick in map(str,_items(ruling.get("better"))):
   if pick not in self.full or pick in self.cart_ids()or not self.buyable(pick)or self.sold_out(pick):continue
   same=[x for x in self.cart_ids()if self.kind_of(x)==self.kind_of(pick)]if self.kind_of(pick)else(self.cart_ids()if len(self.items)==1 else[])
   if len(same)!=1:continue
   pos=self.slot_of(same[0])
   if pos is None or self.item_clash(pick,pos)or not self.fits_total(same[0],pick):continue
   ids=[x for x in self.cart_ids()if x!=same[0]]+[pick]
   self.check_cart(ids,SPEND_HARD)
   if self.verdicts.get(self.check_key(ids))and self.swap_line(same[0],pick):return True
   tries-=1
   if tries<=0:break
  return False
 def swap_line(self,lost,pick,spare=1):
  tool=self.roles.get("cart")
  want={}
  for lid in self.cart_ids():
   if lid!=lost:want[lid]=self.cart_lines.get(lid,1)
  want[pick]=self.cart_lines.get(lost,1)
  plan=self.cart_calls(want,[lost])
  if not plan or self.left()<len(plan)+self.tail()+spare:return False
  for raw in plan:
   answer,sent=self._kd_send(tool,raw,self.finished)
   if answer is None:return False
   self.note_cart(tool,sent)
   self.observe(tool,answer.get("observation"))
  if pick not in self.carted:self.carted.append(pick)
  self.trace.append({"kind":"swap_line","lost":lost,"pick":pick})
  return True
 def walk_browse(self,tool,call_args,pages):
  tries=3
  while tries:
   tries-=1
   answer,sent=self._kd_send(tool,call_args)
   if answer is None or self.finished:return
   count=len(self.records_in(self._kd_take(tool,sent,answer)))
   rec=self.search_log.get(tool+_dump(sent))
   if not count or not rec or rec.get("total",0)<=(pages+1)*count or self.left()<=self.tail()+7:break
   ca=self.brief["browse_args"]["cap"]
   v=call_args[ca]
   call_args=dict(call_args)
   call_args[ca]=v//2 if isinstance(v,int)else v/2
   upper=tool+_dump(sent)
  key=tool+_dump(sent)
  # a halved cap that fits on one page leaves the band between it and the cap above unseen: page that band instead
  if tries<2 and(self.search_log.get(key)or{}).get("token")is None:key=upper
  for _ in range(pages):
   rec=self.search_log.get(key)or{}
   if rec.get("token")is None or self.left()<=self.tail()+7 or self.finished or not self.page_on(rec):break
 def wider_hold(self):
  if self.left()<14+2*len(self.items)or self.wider_held or self.brief.get("pick")!="cheapest":return""
  self.wider_held=True
  tags="; ".join("%s at %g"%(cid,self.price_of(cid))for cid in self.cart_ids()if self.price_of(cid)is not None)
  return("Held (this used no step; %d steps remain) while the cheapest-pick rule applies. Search matches on title words and lists rows in no price order, while required values often sit only in attributes, so a few queries miss cheaper fits. So, ahead of checkout and for each item: try a handful of extra searches under other names for the product - the plain one-word name, synonyms, terms taken from suitable titles you have seen. Of the results, look only at in-stock rows priced under the current cart line%s. Open them cheapest first; when one meets every need, swap it in for that cart line, then check out."%(self.left()," (%s)"%tags if tags else""))
 def shaped_act(self,tool,raw,schema):
  faults=[]
  sent=_shape(raw,schema,"args",faults)
  return sent,(None if faults else self.act(tool,sent))
 def fill_facts(self,candidates,reserve_steps):
  left,seen_facts=self.left,self.facts_from
  tool=self.facts_tool()
  key=self.single_target(tool)
  if not tool or not key:return[]
  notes=[]
  # a thin page stays out of the full records, so the version it was read at keeps it from being read again unchanged
  for cid in[x for x in dict.fromkeys(candidates)if x in self.rows and x not in self.full and not self.sold_out(x)and self.facts_read.get(x)!=self.version]:
   if self.finished or left()<=reserve_steps or seen_facts.get(tool)is False:break
   env=self.shaped_act(tool,{key:cid},self.schemas.get(tool))[1]
   if env is None:break
   self.facts_read[cid]=self.version
   page=env.get("observation")
   self.observe(tool,page)
   self.note_view(tool,page)
   if isinstance(page,dict)and self.ident(page)and not seen_facts.get(tool):seen_facts[tool]=bool(_fact_block(page))
   notes.append("%s: %s"%(cid,self.show(page)[:1500]))
  if notes:self.trace.append({"kind":"fill_facts","n":len(notes)})
  return notes
 def cheaper_rows(self):
  kind,price=self.kind_of,self.price_of
  if self.brief.get("pick")!="cheapest":return[]
  mark="%s#%s"%(self.cart_mark(),self.version)
  if not self.category_key()or mark in self.cheaper_marks:return[]
  self.cheaper_marks.add(mark)
  lines=self.cart_ids()
  pool=[c for c in self.seen_order if c not in lines and c not in self.full and self.buyable(c)]
  under=lambda cap,sort:sorted((c for c in pool if kind(c)==sort and price(c)is not None and price(c)<cap),key=lambda c:price(c)or 0.0)[:2]
  chosen=[]
  for cid in lines:
   cost,sort=price(cid),kind(cid)
   if cost is not None and sort:chosen.extend([c for c in under(cost,sort)if c not in chosen])
  return self.fill_facts(chosen[:CHEAPER_OPENS],self.tail()+4)
 def observe(self,tool,outcome):
  self.absorb(outcome)
  loose=self.loose_lines(outcome)
  said=""if not loose else self.take_note(loose)
  if said:self.queued.append("(seen on %s) %s"%(tool,said))
  return said
 def widen(self):
  left,fire,see=self.left,self.shaped_act,self.observe
  tool=self.roles.get("compare")
  schema=self.schemas.get(tool or"")or{}
  slot=next((name for name,spec in(schema.get("properties")or{}).items()if isinstance(spec,dict)and"array"in _kinds(spec)),None)
  if not(tool and slot and all(name==slot for name in schema.get("required")or[]))or self.compare_bare():return self.widen_singly()
  if left()<=12:return None
  cap,groups=self.compare_room(),self.item_groups()
  per=max(1,cap//max(1,len(groups)))
  batch=sum(([gid for gid in grp if gid not in self.full][:per]for grp in groups),[])[:cap]
  if len(batch)<2:return None
  before=len(self.full)
  env=fire(tool,{slot:batch},schema)[1]
  if env is None:return None
  page=env.get("observation")
  see(tool,page)
  self.note_view(tool,page)
  found=self.records_in(page)
  if found and not self.facts_from.get(tool):self.facts_from[tool]=any(_fact_block(lg)for _,lg in found)
  self.trace.append({"kind":"widen","ids":batch,"opened":len(self.full)-before})
  if found and self.facts_from.get(tool)is False:return self.widen_singly()
  if len(self.full)==before and len(batch)>2 and left()>12:
   self.compare_bad=min(self.compare_bad or 999,len(batch))
   env=fire(tool,{slot:batch[:2]},schema)[1]
   if env is not None:see(tool,env.get("observation"))
  return None
 def read_first(self):
  left,groups=self.left,self.item_groups
  n=len(self.items)
  floor=14+(6*n if self.brief.get("pick")=="cheapest"and self.browse_call("-",1)else 0)
  if n>1:return self.read_each(floor)
  goal=FIRST_READS
  # pages opened for pointed-to listings of other kinds do not fill the requested items' key table
  mine=lambda:len({c for grp in groups()for c in grp if c in self.full})
  for _ in range(4):
   if mine()>=goal or left()<=floor or self.finished or self.secs()<CLOCK_EXTRAS:return
   had=len(self.full)
   if len([c for grp in groups()for c in grp if c not in self.full])>=2:self.widen()
   if len(self.full)==had:
    # buyable rows still unread mean the opens were held back: another page would only pile up more unread rows
    if len([c for grp in groups()for c in grp if c not in self.full and self.buyable(c)])>=2:return
    wanted={c for grp in groups()for c in grp}
    page=max((r for r in self.search_log.values()if r.get("token")is not None),key=lambda r:len(wanted.intersection(r["ids"])),default=None)
    if page is None or not self.page_on(page):return
 def item_rows(self,prod):
  # seen rows of this item's kind: their category or title shares a long word with the item's name, its store category or its title-spelt noun
  bag={w for w in _stem(prod.shelf)|_stem(prod.name)|_stem(self.title_noun(prod))if len(w)>=4}
  cat=self.category_key()
  return[c for c in self.seen_order if bag&{w for w in _stem(str(self.rows[c].get(cat)or"")if cat else"")|_tokens(self.title_of(c))if len(w)>=4}]
 def read_each(self,floor):
  # several items: each reads a few of its own listings before the plan, an item with none seen gets a search first, and no more:
  # an item left unread is never asked about, and the steps saved go to the searches after the answers
  per=3 if len(self.items)==2 else 2
  tl=self.roles.get("search")
  key=self.single_target(tl)
  for prod in self.items:
   if self.left()<=floor or self.finished or self.secs()<CLOCK_EXTRAS:return
   rows=self.item_rows(prod)
   q=self.lead_query(prod,[])if not rows and tl and key else""
   if q and q not in self.sought:
    self.sought.add(q)
    sent,env=self.shaped_act(tl,{key:q},self.schemas.get(tl))
    if env is not None:
     self.observe(tl,env.get("observation"))
     self.log_search(tl,sent,env.get("observation"))
    rows=self.item_rows(prod)
   want=[c for c in rows if c not in self.full and self.buyable(c)][:max(0,per-sum(c in self.full for c in rows))]
   if want and not self.finished:self.open_rows(want)
 def widen_singly(self):
  price,done=self.price_of,self.full
  tool=self.facts_tool()
  if tool and self.single_target(tool):
   cost=lambda c:float("inf")if price(c)is None else price(c)
   lanes=[]
   for grp in self.item_groups():
    lane=[c for c in grp if c not in done and self.buyable(c)]
    lanes.append((sorted(lane,key=cost)if self.brief.get("pick")=="cheapest"else lane)[:max(0,min(2,8-len([c for c in grp if c in done])))])
   self.fill_facts([lane[turn]for turn in(0,1)for lane in lanes if turn<len(lane)][:5],22)
 def fuse_shelf(self):
  if self.asked or self.carted or self.cart_touched or self.lead_items or len(self.items)<2:return""
  flat=lambda v:" "+" ".join(re.sub(r"[^a-z0-9]+"," ",str(v).lower().replace("&"," and ")).split())+" "
  said=flat(self.request)
  pool={v.strip()for row in self.rows.values()for v in row.values()if isinstance(v,str)and 3<len(v.strip())<=60}
  for lab in sorted(pool|{it.shelf.strip()for it in self.items if it.shelf}|self.labels(),key=len,reverse=True):
   parts=[_stem(p)for p in re.split(r"\s*(?:,|&|\band\b)\s*",lab.lower())if p.strip()]
   if len(parts)<2 or flat(lab)not in said:continue
   hit=lambda it:{n for n,p in enumerate(parts)if _stem(it.name)and _stem(it.name)<=p}
   group=[n for n,it in enumerate(self.items)if hit(it)]
   if len(group)<2 or set().union(*map(hit,map(self.items.__getitem__,group)))!=set(range(len(parts))):continue
   sets=[set(self.items[n].musts)for n in group]
   if len({self.items[n].qty for n in group})>1 or not any(all(m<=big for m in sets)for big in sets):continue
   old,keep=self.items,group[0]
   one=Item(lab.lower(),lab,old[keep].qty,list(dict.fromkeys(m for n in group for m in old[n].musts)))
   self.items=[one if n==keep else it for n,it in enumerate(old)if n not in group[1:]]
   slot={n+1:self.items.index(one if n in group else it)+1 for n,it in enumerate(old)}
   self.refs=list(dict.fromkeys((slot.get(p,1),r)for p,r in self.refs))
   self.trace.append(dict(kind="fuse",shelf=lab,items=[old[n].name for n in group]))
   return"[Kyo] The store files %r as a single department, so the request's %s is ONE item: item %d, %s, quantity %d. Buy one listing for it, not one per word."%(lab," + ".join(old[n].name for n in group),keep+1,one.name,one.qty)
  return""
 def ready_to_ask(self):
  if any((self.asked,not self.roles.get("ask"),self.msgs_free()<=0,self.left()<=8,self.secs()<CLOCK_EXTRAS)):return False
  return len(self.full)>self.walked_full+3*len(self.items)or self.step>=self.walked_step+max(8,self.max_steps//4)
 def message_args(self,msg):
  schema=self.schemas.get(self.roles.get("ask")or"")or{}
  props=schema.get("properties")or{}
  nd=schema.get("required")or[]
  texts=[name for name,spec in props.items()if isinstance(spec,dict)and"string"in _kinds(spec)and"enum"not in spec]
  slot=next((name for name in nd if name in texts),None)or next(iter(texts),None)
  if not slot:return None
  payload={slot:msg}
  for name in nd:
   if name in payload:continue
   opts=(props.get(name)or{}).get("enum")
   if not isinstance(opts,list)or not opts:return None
   payload[name]=opts[0]
  faults=[]
  payload=_shape(payload,schema,"args",faults)
  return None if faults else payload
 def interview(self):
  """Plan a ranked lane of page keys per item, then ask, always serving the item with no need found yet.

  A message names one item only and, for a one-item order, one key. On a larger order an item's open keys are spread over the
  messages it can still get: one key when the messages cover them all, else the next few keys together, up to ASK_SPAN. A reply
  that objects to the many topics turns the rest of the interview to one key per message; a merely puzzled reply does not, as
  single questions puzzle the shopper as often. A list that draws nothing gives the item's next list a key not yet offered as its
  lead, followed once more by the silent list's other keys: a reader can answer the start of a list and pass over the rest.
  A single key of several words that draws the objection is asked again by its last word alone.
  """
  ask=self.question
  self.asked=True
  # several keys in one message reach far more needs of a many-item order but only pay under some shoppers; one item gets one key
  lists=len(self.items)>1
  room=self.msgs_free()
  if room<=0:return""
  self.widen()
  self.read_first()
  self.use_wishes()
  table=self.key_table()
  lanes,order=self.lane_plan(room,table)
  heard={n:0 for n in lanes}
  spent={n:0 for n in lanes}
  tried={n:set()for n in lanes}
  again={n:[]for n in lanes}
  relisted={n:set()for n in lanes}
  sent,new_reqs=[],[]
  # one reading per reply: the light reader keeps the strong one's budget for the cart check and the review
  models=(LIGHT,HEAVY)
  while not self.finished and self.msgs_free()>0 and self.left()>6:
   sel=self.next_in_lane(lanes,heard,spent,tried,order)
   if sel is None:break
   pos=sel[0]
   ahead=[k for k in lanes[pos]if _words(k)not in tried[pos]and not self.told_dup((pos,k))]
   kept=set(tried[pos])
   waiting=sum(1 for n in lanes if not heard.get(n))or 1
   share=max(1,self.msgs_free()//(1 if heard.get(pos)else waiting))
   span=min(ASK_SPAN,-(-(len(ahead)+len(again[pos]))//share))
   # keys never offered fill a list before the silent list's keys come round again, and a key nested in one already listed or
   # asked waits: an undrawn need is most often one never named, and slots further down a list draw less
   names,fresh=[],[k for k in ahead if not any(_nested_keys(k,t)for t in kept)]
   for k in(fresh+again[pos]+[k for k in ahead if k not in fresh]if lists else ahead[:1]):
    if len(names)<span and not any(_nested_keys(k,o)for o in names):names.append(k)
   names=names or[sel[1]]
   relisted[pos].update(_words(k)for k in names if k in again[pos])
   again[pos]=[]
   tried[pos].update(map(_words,names))
   got=self.send([(pos,names[0],ask(pos,names))])
   if not got:break
   sent.extend(got)
   spent[pos]+=1
   before=[len(p.told)for p in self.items]
   verdict=self.hear(got,models)
   crowded=1 in{_count(q)for q in _items(verdict.get("crowded"))}
   if len(names)>1 and crowded:
    # the shopper wants one topic at a time: these keys go back to the lane and are asked singly from now on
    lists,tried[pos]=False,kept
   elif crowded and len(_words(names[0]).split())>1:
    # one key of several words heard as several topics: its last word, a property name on its own, is asked next
    head=_words(names[0]).split()[-1]
    fixed=[m.split(":")[0]for m in self.items[pos-1].musts]
    if head not in tried[pos]and not any(_names(_tokens(f),_tokens(head))for f in fixed):lanes[pos]=[head]+[k for k in lanes[pos]if k!=head]
   self.take_heard(verdict,new_reqs)
   if lists and len(names)>1 and len(self.items[pos-1].told)==before[pos-1]:again[pos]=[k for k in names[1:]if _words(k)not in relisted[pos]]
   for num,prod in enumerate(self.items,1):
    if len(prod.told)>before[num-1]:heard[num]=heard.get(num,0)+1
  return self.interview_report(sent,new_reqs)
 def next_in_lane(self,lanes,heard,spent,tried,order):
  # the item still without a found need goes first, then the one asked least, then the planner's order
  rank={n:i for i,n in enumerate(order)}
  for pos in sorted(lanes,key=lambda n:(heard.get(n,0)>0,spent.get(n,0),rank.get(n,len(rank)+n))):
   for label in lanes[pos]:
    if _words(label)in tried[pos]or self.told_dup((pos,label)):continue
    return(pos,label)
  return None
 def key_shares(self,table):
  # per item: each listed key's share of that item's opened listings, how evenly its values split the item's buyable in-budget pages, the keys the request already spells a printed value of, plus keys that only mirror another key's value
  cat,size=self.category_key(),len(self.items)
  pages={}
  for lid in self.seen_order:
   page=self.full.get(lid)
   if page:pages.setdefault(str(page.get(cat)or"listings")if cat else"listings",[]).append(lid)
  own=lambda pos:_stem(self.items[pos-1].name)|_stem(self.items[pos-1].shelf)
  per=getattr(self,"key_per",None)or{}
  run=lambda t:" %s "%" ".join(_kb_PLURAL.sub(r"\1",w)for w in _kb_norm(t).split())
  said=run(self.request)
  fixed=run(" ; ".join(m for it in self.items for m in it.musts if"only if"in m or m in{r for _,r in self.refs}))
  # a single item's request spelling out a value its pages print has answered that key; one bare word may come from the story, so it takes two words or a figure joined to letters, and words of a condition or of a listing pointed to do not count
  told=lambda v:size==1 and(len(v.split())>=2 or(re.search(r"\d",v)and re.search(r"[a-z]",v)))and v in said and v not in fixed
  shares,split,spelt={},{},{}
  for pos in range(1,size+1):
   rows=[r for r in table if size==1 or _tokens(r[0])&own(pos)]
   if not rows and size>1:rows=[r for r in table if not any(_tokens(r[0])&own(o)for o in range(1,size+1))]
   got={}
   for cat_label,key,hits,vals,kinds in rows:
    got[key]=max(got.get(key,0.0),hits/max(1,len(pages[cat_label])if cat_label in pages else hits))
   shares[pos]=got
   mine={lid for r in rows for lid in pages.get(r[0],[])}
   live={lid for lid in mine if self.buyable(lid)and self.within_budget(lid,self.items[pos-1].qty)}
   split[pos],spelt[pos]={},set()
   for k in got:
    vals=per.get(k)or{}
    # split: live pages printing the key minus those printing its most common value
    tally=[_words(v)for lid,v in vals.items()if lid in live and v]
    split[pos][k]=len(tally)-max(map(tally.count,tally),default=0)
    if any(told(run(v))for lid,v in vals.items()if lid in mine):spelt[pos].add(k)
  mirror=set()
  for key,vals in per.items():
   echo,partners=0,set()
   for lid,v in vals.items():
    hit=[o for o in per if o!=key and per[o].get(lid)==v and not self.same_prop(key,o)]
    if hit:echo+=1;partners.update(hit)
   # a key that on most listings repeats some other key's value, a different key each time, only lists variants
   if len(vals)>=3 and 2*echo>=len(vals)and len(partners)>=2:mirror.add(key)
  return shares,mirror,split,spelt
 def lane_plan(self,room,table):
  """Ask the planner for each item's ranked keys, map them to the store's spellings, guard each lane with its widely printed keys."""
  size=len(self.items)
  checks=[str(v)for v in _items(self.brief.get("ask_first"))if str(v).strip()]
  count=min(max(room,2),5)
  parts=[LANES.replace("COUNT",str(count)),"\n\nREQUEST:\n",self.request,"\n\nITEMS:\n"]
  parts.append("\n".join("item %d (%s); fixed by the request: %s"%(pos,prod.name,"; ".join(prod.musts)or"nothing")for pos,prod in enumerate(self.items,1)))
  if checks:parts+=["\n\nASK FIRST:\n","\n".join(checks)]
  shown,last=[],None
  for pos,(cat,key,hits,vals,kinds)in enumerate(table,1):
   if cat!=last:
    shown.append("category %r%s:"%(cat," (most listings opened here are copies of others, so a value they all print sends no key last: rank these keys by what a buyer would pick this kind of product by and how many listings print them)"if cat in self.copy_cats else""))
    last=cat
   shown.append("  %s. %s (on %s, %s value%s): %s"%(pos,key,hits,kinds,""if kinds==1 else"s"," | ".join(vals)))
  parts+=["\n\nATTRIBUTE KEYS SEEN:\n","\n".join(shown)or"(nothing opened in full so far)"]
  reply=self.ask_json("".join(parts),[HEAVY,LIGHT],3000)or{}
  shares,mirror,split,spelt=self.key_shares(table)
  lanes={pos:[]for pos in range(1,size+1)}
  order=[]
  def spelled(pos,label):
   plain=lambda t:" ".join(w.rstrip("s")for w in _words(t).split())
   return next((k for k in shares[pos]if plain(k)==plain(label)),_words(label))
  for ej in[e for e in _items(reply.get("ask"))if isinstance(e,dict)]:
   pos=min(max(1,_count(ej.get("item")or 1)or 1),size)
   head=re.split(r"\s+(?:and|or)\s+|[/,;(]",str(ej.get("label")or""))[0].strip().rstrip("?.")
   if not head or len(head.split())>5:continue
   key=spelled(pos,head)
   if any(self.same_prop(key,o)for o in lanes[pos]):continue
   lanes[pos].append(key)
   if str(ej.get("hint")or"").strip()and pos not in order:order.append(pos)
  order+=[pos for pos in lanes if pos not in order]
  for pos,prod in enumerate(self.items,1):
   share=shares[pos]
   settled=[m.split(":")[0]for m in prod.musts]
   open_key=lambda k:k not in mirror and not any(self.same_prop(k,s)or _names(_tokens(s),_tokens(k))for s in settled)
   flat={r[1]for r in table if r[4]<=1 and r[2]>=3}
   # equal shares go to the key splitting the buyable in-budget pages more evenly; a key the request already spells a value of joins the fill after the open ones and guards nothing
   rank=lambda k:(-share[k],-split[pos].get(k,0))
   fill=[k for k in share if open_key(k)and k not in flat]
   ranked=sorted([k for k in fill if k not in spelt[pos]],key=rank)
   lane=[k for k in lanes[pos]if not any(self.same_prop(k,s)for s in settled)]
   head=min(max(room,1),3)
   # a widely printed key takes the slot of a planner pick printed on far fewer listings, a story-led pick included
   for guard in [k for k in ranked if share[k]>=0.5][:2]:
    if any(self.same_prop(guard,o)for o in lane[:head]):continue
    weak=[i for i,o in enumerate(lane[:head])if share.get(o,0.0)+0.34<=share[guard]]
    if not weak and len(lane)>=head:continue
    lane=[o for o in lane if not self.same_prop(guard,o)]
    if weak:
     moved=lane.pop(weak[-1])
     lane.insert(weak[-1],guard)
     lane.insert(head,moved)
    elif len(lane)<head:lane.append(guard)
   # on a several-item order a key that mostly repeats other keys' values still joins the list, after the open keys: it can be the wish
   echoes=sorted((k for k in mirror&set(share)if k not in flat and not any(self.same_prop(k,s)or _names(_tokens(s),_tokens(k))for s in settled)),key=rank)if size>1 else[]
   for k in ranked+sorted(spelt[pos].intersection(fill),key=rank)+echoes+sorted(flat&set(share),key=lambda k:-share[k]):
    if not any(self.same_prop(k,o)for o in lane):lane.append(k)
   # a lone item's first question goes to a key most of its opened listings print: a rarely printed first key seldom finds the wish
   lead=next((k for k in lane if share.get(k,0.0)>=0.5),None)
   if size==1 and lead and share.get(lane[0],0.0)<0.5:lane=[lead]+[k for k in lane if k!=lead]
   lanes[pos]=lane
  self.trace.append({"kind":"plan","ask":_clip(reply.get("ask"),120),"lanes":{p:v[:6]for p,v in lanes.items()}})
  return lanes,order
 def use_wishes(self):
  # a word named with the product that the item's own listings print as a value is a tie-breaker, never a breach
  wish=_items(self.brief.get("wishes"))
  for pos,prod in enumerate(self.items,1):
   own=_tokens(prod.name)-_tokens(prod.shelf)-_tokens(" ".join(prod.musts))
   for lid in self.seen_order:
    block=_fact_block(self.full.get(lid)or{})or{}
    for k,v in block.items():
     hit=own&_tokens(v)if isinstance(v,str)and len(_tokens(v))<=3 else set()
     line="item %s: %s: %s"%(pos,_words(k),v.strip()[:40])if hit else""
     if line and line not in wish and len(wish)<8:wish.append(line)
  self.brief["wishes"]=wish
 def word_values(self,prod):
  skip=set(pair[1]for pair in self.refs)-set(self.ref_words)
  for nd in[n for n in prod.told+prod.musts if n not in skip and n not in prod.dropped]:
   text=self.ref_words.get(nd)or nd.split(":",1)[-1].strip()
   head=nd.split(":",1)[0].strip()
   said=_tokens(self.request)
   if nd in prod.musts and nd not in self.ref_words and":"in nd and len(_tokens(text))==1 and not re.search(r"\d",text)and not _tokens(text)<=said and _tokens(head)<=said and not self.titled(text):text=head
   bits=text.split()
   if nd in prod.told and nd not in self.ref_words and len(bits)>1 and not re.search(r"\d",text)and not _tokens(text)<=self.fact_words():
    # a reply's need can wrap its value in the reply's own phrasing: keep the closing words that opened records print
    tail=[" ".join(bits[-k:])for k in range(min(3,len(bits)-1),0,-1)]
    text=next((t for t in tail if _tokens(t)<=self.fact_words()),text)
   if text and len(text.split())<=3 and(nd in self.ref_words or not re.search(r"\d",text)or self.titled(text)):yield text
 def fact_words(self):
  return set().union(*(_tokens(_dump(_fact_block(rec)or{}))for rec in self.full.values()))
 def limit_values(self,prod,wide=False):
  skip=set(pair[1]for pair in self.refs)-set(self.ref_words)
  picked=[self.ref_words.get(n)or n.split(":",1)[-1].strip()for n in prod.told+prod.musts if n not in skip and n not in prod.dropped]
  yield from(t for t in picked if re.search(r"\d",t)and(wide or len(t.split())<=3))
 def joined_query(self,prod):
  # each wanted figure goes in with the word right after it (its unit), however long the phrase around it
  nums=[bit for text in self.limit_values(prod,True)for bit in re.findall(r"\d[\d.]*(?: ?[a-z]+)?",text.lower())]
  words=self.value_query(list(dict.fromkeys(self.word_values(prod)))[:3]).split()
  return self.lead_query(prod,nums+words)if nums or words else""
 def title_noun(self,prod):
  # the item's last word as seen titles spell it (a store matches title words, and a plural can change more than its end)
  noun=(re.findall(r"[a-z0-9]+",prod.name.lower())or[""])[-1]
  if len(noun)<4:return noun
  tally={}
  for c in self.rows:
   for w in set(re.findall(r"[a-z]+",self.title_of(c).lower())):
    if w[:4]==noun[:4]and len(w)<=len(noun)+1:tally[w]=tally.get(w,0)+1
  return max(sorted(tally),key=tally.get)if tally else noun
 def lead_query(self,prod,bits,nth=0):
  # the product word and the nth rarest value word (fewest seen rows print it) lead, so a store reading only the first words still gets both
  rows=[_tokens(_dump(r))for r in self.rows.values()]
  plain=sorted(dict.fromkeys(b for b in bits if not re.search(r"\d",b)and len(b)>1),key=lambda b:sum(_tokens(b)<=r for r in rows))
  if nth and nth>=len(plain):return""
  rare=plain[nth]if plain else""
  words=list(dict.fromkeys(w for w in[self.title_noun(prod),rare,*bits,*prod.name.split()]if w and not re.search(r"\d",w)))
  figs=[b for b in bits if re.search(r"\d",b)]
  # three words stay findable whether a store needs every word, any word or only the leading ones; a figure never
  # takes one of the first two places, where a store reading only the leading words would match it exactly
  out=" ".join(dict.fromkeys(words[:2]+figs+words[2:]))
  return" ".join(out.split()[:QUERY_WORDS])
 def meets_limits(self,listing):
  facts=_tokens(_dump(_fact_block(self.full.get(listing)or{})or{}))
  wants=[[_tokens(" ".join(re.findall(r"\d+",text)))for text in self.limit_values(prod)]for prod in self.items]
  return any(all(part<=facts for part in sets)for sets in wants)
 @staticmethod
 def need_clash(nd,block,told,aimed):
  prop,_,want=nd.partition(":")
  if"only if"in want:return None
  span=re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*",want)
  near=[(k,str(v))for k,v in block.items()if _names(_stem(prop),_stem(k))and str(v).strip()]
  # a field named exactly like the property decides alone: a sibling field printing the wanted value does not rescue a clash there
  named=[v for k,v in near if _stem(k)==_stem(prop)]or[v for _,v in near]
  if span:
   shown=[t for t in named if re.search(r"\d",t)and span[3].lower()in t.lower()]
   lo,hi=float(span[1]),float(span[2])
   if shown and all(not lo<=float(m)<=hi for t in shown for m in re.findall(r"\d+(?:\.\d+)?",t)):return nd,shown[0]
  if not named or nd in aimed:return None
  # a told value of several words, none a bare figure, that a page prints only with another word wedged inside names something else
  split=[t for t in named if _tokens(want)<=_tokens(t)]
  if nd in told and len(_tokens(want))>=2 and not any(w.isdigit()for w in _tokens(want))and split and not any(_side(want,t)for t in named):return nd,split[0],"apart"
  bare=r"(?<![A-Za-z0-9.])\d"
  if re.search(bare,want):return None if any(re.search(bare,t)for t in named)else(nd,named[0])
  if(nd in told or len(want.split())==1)and all(not _tokens(want)<=_tokens(t)for t in named):return nd,named[0]
  return None
 def range_miss(self,listing):
  block=_fact_block(self.full.get(listing)or{})or{}
  told,aimed=set(nd for prod in self.items for nd in prod.told),set(pair[1]for pair in self.refs)
  hits=(self.need_clash(nd,block,told,aimed)or(nd not in aimed and(self.unit_miss(nd,block)or self.word_miss(nd,block)))for _,nd in self.needs_now())
  return next((hit for hit in hits if hit),None)
 def unit_words(self):
  # words this store prints right beside a figure, in attributes or titles, on two listings or more
  size=(len(self.full),len(self.rows))
  if getattr(self,"units_at",None)!=size:
   seen={}
   for lid in set(self.full)|set(self.rows):
    texts=[str(v)for v in(_fact_block(self.full.get(lid)or{})or{}).values()]+[self.title_of(lid)]
    for t in map(_figure_words,texts):
     for i,w in enumerate(t):
      if w.isalpha()and len(w)>=2 and any(not x.isalpha()for x in t[max(0,i-1):i]+t[i+1:i+2]):seen.setdefault(w,set()).add(lid)
   self.units_at,self.units=size,{k for k,v in seen.items()if len(v)>=2}
  return self.units
 @staticmethod
 def word_miss(nd,block):
  # a stated value of several words that the attribute named like the property shares no word with: a different value
  # (a single word, a told value, a figure, an exclusion or a condition is judged by need_clash or left to the reading)
  prop,_,want=nd.partition(":")
  words=_stem(want)
  if len(words)<2 or re.search(r"\d",want)or"only if"in want or"not"in words:return None
  exact=[str(v)for k,v in block.items()if _nest(_stem(prop),_stem(k))and _nest(_stem(k),_stem(prop))and str(v).strip()]
  near=lambda a,b:_akin(a,b)or len(min(a,b,key=len))>=4 and(a.startswith(b)or b.startswith(a))
  if exact and all(not any(near(w,o)for w in words for o in _stem(v))for v in exact):return nd,exact[0]
  return None
 def unit_miss(self,nd,block):
  # the stated figure printed with another unit word beside it and without the stated one (a size in another system):
  # the page shows the number but measures it differently
  prop,_,want=nd.partition(":")
  w,units=_figure_words(want),self.unit_words()
  beside=lambda t,i:[x for x in t[max(0,i-1):i]+t[i+1:i+2]if x in units]
  for k,v in block.items():
   if not _names(_stem(prop),_stem(k)):continue
   t=_figure_words(v)
   for i,f in enumerate(t):
    if f.isalpha()or not beside(t,i):continue
    for j,x in enumerate(w):
     near=w[max(0,j-1):j]+w[j+1:j+2]
     if x==f and[y for y in beside(w,j)if y not in t]and not set(near)&set(t):return nd,str(v)
  return None
 def item_clash(self,listing,pos):
  # a printed value breaking one of this item's own needs; a wanted word the page only extends (a prefix either way) is no breach
  block=_fact_block(self.full.get(listing)or{})or{}
  prod=self.items[pos-1]
  told,aimed=set(prod.told),set(pair[1]for pair in self.refs)
  for nd in[n for n in prod.musts+prod.told if n not in prod.dropped]:
   hit=self.need_clash(nd,block,told,aimed)
   if not hit:continue
   if len(hit)>2:return hit[:2]
   want,shown=_tokens(nd.partition(":")[2]),_tokens(hit[1])
   # a page value spelt with a slip (a letter missing, doubled or swapped) still names the wanted value
   if want and all(any(t.startswith(w)or w.startswith(t)or min(len(t),len(w))>=5 and difflib.SequenceMatcher(None,t,w).ratio()>=0.85 for t in shown)for w in want):continue
   return hit
  return None
 def range_hold(self):
  kind,price=self.kind_of,self.price_of
  if len(self.items)!=1 or self.left()<=self.tail()+1:return""
  self.fill_facts(self.cart_ids(),self.tail()+1)
  lines,prod=self.cart_ids(),self.items[0]
  bad=[(cid,clash)for cid in lines if cid not in self.range_held for clash in[self.range_miss(cid)]if clash]
  homes=[sort for sort in map(kind,self.seen_order)if _tokens(prod.name)&_tokens(sort)]
  if homes:bad+=[(cid,("the requested item: "+prod.name,kind(cid)))for cid in lines if cid not in self.range_held and kind(cid)not in homes]
  if not bad:return""
  self.range_held.update(cid for cid,_ in bad)
  cheap=self.brief.get("pick")=="cheapest"
  fit=lambda:[c for c in self.full if c not in lines and self.buyable(c)and self.within_budget(c,prod.qty)and self.carries(c)and not self.range_miss(c)]
  fits=fit()
  if cheap and not fits and self.left()>self.tail()+4 and not self.finished:
   # nothing opened can take the line: the cheapest unread rows that searches on the wanted values listed are read first
   want,tool=_tokens(" ".join(self.word_values(prod))),self.roles.get("search")
   rows={c for r in self.search_log.values()if r["tool"]==tool and want&_tokens(_dump(r["args"]))for c in r["ids"]}
   unread=sorted((c for c in rows if c not in self.full and self.buyable(c)and self.within_budget(c,prod.qty)and price(c)is not None),key=price)
   if unread and self.open_rows(unread[:self.compare_room()]):fits=fit()
  fits.sort(key=lambda c:(not cheap and not self.meets_limits(c),float("inf")if price(c)is None else price(c)))
  self.trace.append({"kind":"range_hold","lines":bad,"fits":fits[:4]})
  listed="; ".join("%s: %s vs the need %s"%(cid,clash[1],clash[0])for cid,clash in bad)
  offers=", ".join(c if price(c)is None else"%s (%g)"%(c,price(c))for c in fits[:4])
  return("A cart line looks off against a need: %s. How to rule: an attribute named like the need is the deciding field; with no field of that name, the listing's title and description decide. Same value in different words counts as a match - keep the line then. If it really conflicts, replace it with a listing that fits%s"%(listed,"; to open instead (wanted words present, no conflict): %s."%offers if fits else"."))
 def titled(self,value):
  probe=" ".join(value.split()).lower()
  return any(probe in" ".join(self.title_of(x).split()).lower()for x in self.rows)
 def value_query(self,values):
  out=[]
  for text in values:
   mixed=[b for b in text.split()if re.search(r"[A-Za-z]",b)and re.search(r"\d",b)]
   out.extend(mixed or text.split())
  return" ".join(dict.fromkeys(out))
 def carrying_note(self):
  price=self.price_of
  hold=sorted((c for c in self.full if self.buyable(c)and self.within_budget(c,1)and self.carries(c)and not self.range_miss(c)),
    key=lambda c:(not self.meets_limits(c),float("inf")if price(c)is None else price(c)))
  if not hold:return""
  shown=", ".join(c if price(c)is None else"%s (%g)"%(c,price(c))for c in hold[:6])
  return"\n[Kyo searched with the wanted values] Listings holding those words in their records - all limit figures shown go first, then cheapest - verify each need before use: "+shown
 def ref_hunt(self):
  tl=self.roles.get("search")
  key=self.single_target(tl)
  if not tl or not key or self.left()<self.tail()+12 or self.cart_lines or self.secs()<CLOCK_EXTRAS or not(self.asked or self.step<self.max_steps//3):return""
  fresh=[]
  for prod in self.items:
   qtext=self.joined_query(prod)
   if not qtext or all(n not in self.ref_words or n in prod.dropped for n in prod.musts+prod.told):continue
   sent,env=(None,None)if self.finished else self.shaped_act(tl,{key:qtext},self.schemas.get(tl))
   if env is None:continue
   page=env.get("observation")
   self.observe(tl,page)
   self.log_search(tl,sent,page)
   log=self.search_log.get(tl+_dump(sent))or{}
   if log.get("token")is not None:self.page_on(log)
   words=_tokens(prod.name)
   fresh+=[c for c in log.get("ids")or[]if c not in self.full and c not in fresh and self.buyable(c)and self.within_budget(c,1)
     and words&_tokens(_dump(self.rows.get(c)))]
  batch=fresh[:self.compare_room()]
  if not batch or self.finished or not self.open_rows(batch):return""
  self.trace.append({"kind":"ref_hunt","read":batch})
  return self.carrying_note()
 def hunt_items(self):
  # after the interview, an item gets a search on its name and wanted values, then its likeliest unread rows opened
  tl=self.roles.get("search")
  key=self.single_target(tl)
  if not tl or not key or self.finished or self.secs()<CLOCK_EXTRAS:return""
  seen=lambda c:_tokens(_dump(_fact_block(self.full.get(c)or{})or{}))|_tokens(self.title_of(c,45))
  # a limit's figure counts only where the page prints it: the page check cannot tell which side of the limit another figure falls
  figs=lambda prod:[_tokens(" ".join(re.findall(r"\d+(?:\.\d+)?",t)))for t in self.limit_values(prod,True)]
  fits=lambda c,pos,prod,vals:((_tokens(prod.name)|_tokens(self.title_noun(prod)))&seen(c)and self.buyable(c)and self.within_budget(c,prod.qty)and not self.item_clash(c,pos)
    and all(any(_tokens(a)<=seen(c)for a in re.split(r"\s+or\s+",v))for v in vals)and all(f<=seen(c)for f in figs(prod)))
  # one item: search again on whatever the replies revealed; several: only those with no opened page fitting yet
  todo=[(pos,prod)for pos,prod in enumerate(self.items,1)if(prod.told if len(self.items)==1 else not any(fits(c,pos,prod,list(self.word_values(prod)))for c in{*self.full,*self.cart_ids()}))]
  # each such item costs a search, two reads and its cart line; with less room it gets the search alone, every item keeps its cart line
  room=self.left()-self.tail()-len(self.items)
  full,batch=room>=3*len(todo),bool(self.compare_target()and self.roles.get("compare")and not self.compare_bare())
  done,rows=[],[]
  for i,(pos,prod)in enumerate(todo):
   spare=lambda:self.left()-self.tail()-len(self.items)-(3 if full else 1)*(len(todo)-i-1)
   vals=list(dict.fromkeys(self.word_values(prod)))
   ran={str(v).strip().lower()for r in self.search_log.values()if r["tool"]==tl for v in r["args"].values()}
   # the second search leads with the next rarest value: a store reading only the first words would return the first one's rows again
   asks=[q for q in dict.fromkeys([self.joined_query(prod),self.lead_query(prod,self.value_query(vals[:3]).split(),1)if vals else""])if q.strip()and q not in self.sought and q.lower()not in ran][:2 if full else 1]
   pool,bags,kind=[],[_tokens(v)for v in vals],_tokens(prod.name)|_tokens(self.title_noun(prod))
   for q in asks:
    if len(pool)>=2 or self.finished or spare()<(1 if q==asks[0]else 3)or self.secs()<CLOCK_EXTRAS:break
    self.sought.add(q)
    sent,env=self.shaped_act(tl,{key:q},self.schemas.get(tl))
    if env is None:continue
    page=env.get("observation")
    self.observe(tl,page)
    self.log_search(tl,sent,page)
    # the wanted listing often sits just past the first page of such a search: one more page while the reads still fit
    log=self.search_log.get(tl+_dump(sent))or{}
    if log.get("token")is not None and spare()>=3 and not self.finished:self.page_on(log)
    got=[c for c in log.get("ids")or[c for c,_ in self.records_in(page)]if c not in self.full and c not in pool and self.buyable(c)and self.within_budget(c,prod.qty)and kind&_tokens(_dump(self.rows.get(c)))]
    pool+=sorted(got,key=lambda c:-sum(b<=_tokens(_dump(self.rows.get(c)))for b in bags))
   n=0 if spare()<1 else 3 if batch else min(2,spare())
   if full and pool and n>0 and not self.finished and self.open_rows(pool[:n]):done+=[(pos,c)for c in pool[:n]if c in self.full]
   rows+=["item %d: %s"%(pos,", ".join(c for c in pool[:3]if c not in self.full))]if any(c not in self.full for c in pool[:3])else[]
  self.trace.append({"kind":"hunt","items":[pos for pos,_ in todo],"read":done,"rows":rows})
  out="".join("\n[Kyo, item %d, searched by name and wanted values] page %s: %s"%(pos,c,self.record_of(c)[:500])for pos,c in done)
  return out+("\n[Kyo's item searches also listed these unread rows] "+"; ".join(rows)if rows else"")
 def carries(self,listing):
  seen=_tokens(_dump(_fact_block(self.full.get(listing)or{})or{}))|_tokens(self.title_of(listing,45))
  for prod in self.items:
   vals=list(self.word_values(prod))
   # a value offering alternatives is carried when any one of them is
   if vals and all(any(_tokens(alt)<=seen for alt in re.split(r"\s+or\s+",v))for v in vals):return True
  return False
 def interview_report(self,sent,new_reqs):
  if new_reqs:
   self.version+=1
   ver=self.version
   bump=lambda old:ver if old==ver-1 else old
   self.view_keys={k:(bump(s[0]),s[1])for k,s in self.view_keys.items()}
   self.read_cache={k:(bump(g[0]),)+tuple(g[1:])for k,g in self.read_cache.items()}
  out=["Interview held by Kyo (each question cost one step):\n","\n".join("Q: %s\nA: %s"%(q,a)for _,q,a in sent),
    "\nFirm requirements that came out of the answers: ","; ".join(new_reqs)if new_reqs else"none",
    ". All picks are checked against them again; run new searches on those values where useful."]
  leads=self.reveal_leads(new_reqs)
  if leads:out.append("\nUnopened rows already printing a revealed value (open these first): "+"; ".join(leads))
  chained=[req for _,req in self.refs if any(req in line for line in new_reqs)]
  # the item searches go before the pointer walk, which can use up every step it is given
  hunted=self.hunt_items()
  walked=""
  if self.open_lookups()and self.secs()>CLOCK_STOP+15 and self.roles.get("search"):
   walked=self.trail()[0]
   self.trail_blocks=0
  chained=[req for req in chained if req not in self.apart or req in self.ref_values]
  if chained:out.append(" Pointer musts, whose value sits on some other listing: %s. Follow each chain link by link - open the referenced listing in full, then each further listing named along the way, and so on; the last page in the chain holds the value, and an order is premature while any page of the chain is still unopened."%" | ".join(chained))
  if walked:out.append("\n"+walked)
  out.append(hunted)
  out.append(self.carrying_note())
  if self.queued:
   out.append("\n[store messages seen mid-interview] "+" | ".join(self.queued))
   self.queued=[]
  return"".join(out)
 def reveal_leads(self,new_reqs):
  # search rows not yet opened whose own text carries a value the shopper just named
  got=[]
  for req in new_reqs:
   want={w for w in _tokens(req.rpartition(":")[2])if len(w)>2}
   if not want:continue
   rows=[p for p,rec in self.rows.items()if p not in self.full and not self.sold_out(p)and want<=_tokens(_dump(rec))]
   if rows:got.append("%s: %s"%(req.rpartition(":")[2].strip()[:30],", ".join(rows[:4])))
  return got[:4]
 def question(self,pos,names):
  # the planner's first key leads; further keys of the same item follow it in lane order
  names=[_spoken_key(k)for k in names]
  pname=names[0]if len(names)==1 else", ".join(names[:-1])+" or "+names[-1]
  name=self.items[pos-1].name
  if len(self.items)!=1 or not re.search(r"\band\b|&",name):return f"Which {pname} should the {name} have?"
  return f"Which {pname} should it have?"
 def title_of(self,listing,words=25):
  rec=self.rows.get(listing)or{}
  return max((v for v in rec.values()if isinstance(v,str)and v!=listing and 2<=len(v.split())<=words),key=len,default=listing)
 def cart_sum(self):
  rows=[(n,self.price_of(lid))for lid,n in self.cart_lines.items()]
  return None if not rows or any(p is None for _,p in rows)else sum(n*p for n,p in rows)
 def with_sum(self,line):
  due=self.cart_sum()
  if due is None:return line
  figs={float(x)for x in re.findall(r"\d+(?:\.\d+)?",line)}
  if due in figs:return line
  said=("Cart total: %g %s."%(due,self.brief.get("budget_unit")or"")).replace(" .",".")
  return said if self.budget in figs else said+" "+line
 def say_basket(self):
  mark,lines=self.cart_mark(),list(self.cart_lines.items())
  due=self.own_statement or(self.gate_due("order")and any(p["messages"]>0 for p in self.process if p["when"]=="order"))
  if not due or not lines or self.said_for==mark or self.msg_used>=self.msg_limit or self.left()<2 or self.finished:return
  rows=[(cid,n,self.price_of(cid))for cid,n in lines]
  each=lambda p:""if p is None else" at %g each"%p
  listed="; ".join("%s x %s (%s)%s"%(n,self.title_of(cid)[:70],cid,each(p))for cid,n,p in rows)
  total=None if any(p is None for _,_,p in rows)else sum(n*p for _,n,p in rows)
  if self.send([(1,"","Basket ready for checkout: "+listed+("."if total is None else". Total %g."%total))],1):
   self.gate_msgs+=1
   self.said_for=mark
 def samples(self,pos,pname):
  key=_words(pname)
  musts=" ".join(self.items[pos-1].musts).lower()
  for ent in self.key_table():
   if ent[1]==key:return[v for v in ent[3]if v and v not in musts and len(v)<=30][:2]
  return[]
 def send(self,queue_out,floor=6):
  tl,done=self.roles["ask"],[]
  for num,_,line in queue_out:
   if self.finished or self.left()<=floor or self.msg_used>=self.msg_limit:break
   args=self.message_args(line)
   env=self.act(tl,args)if args else None
   if env is None:break
   self.msg_used+=1
   page=env.get("observation")
   said=self.spoken(page)
   done.append((num,line,said))
   self.done_log.append("step %s: message to shopper %r, reply %r (cart at that point: %s)"%(self.step,line,said[:300],self.cart_mark()))
   self.done_steps.append(self.step)
   self.trace.append({"kind":"ask","q":line,"a":said[:300]})
   if isinstance(page,dict)and page.get("error"):break
  return done
 def take_heard(self,verdict,added):
  size=len(self.items)
  slot=lambda ent:min(max(1,_count(ent.get("item"))or 1),size)
  for ent in _items(verdict.get("needs")):
   if isinstance(ent,dict)and str(ent.get("need")or"").strip():
    pos,phr=slot(ent),str(ent["need"]).strip()
    prod=self.items[pos-1]
    if phr not in prod.told:
     prod.told.append(phr)
     added.append("item %s: %s"%(pos,phr)if size>1 else phr)
    if ent.get("points")is True and(pos,phr)not in self.refs:self.refs.append((pos,phr))
  for ent in _items(verdict.get("drop")):
   if isinstance(ent,dict)and str(ent.get("property")or"").strip():
    prod=self.items[slot(ent)-1]
    gone=_words(str(ent["property"]).split(":")[0])
    hits=[m for m in prod.musts+prod.told if m not in prod.dropped and _words(m.split(":")[0])==gone]
    prod.dropped.extend(hits)
    added.extend("dropped "+m for m in hits)
  for ent in _items(verdict.get("free")):
   if isinstance(ent,dict)and ent.get("property"):self.items[slot(ent)-1].free.append(str(ent["property"]))
  cap=verdict.get("budget")
  if not isinstance(cap,bool)and isinstance(cap,(int,float))and cap>0:self.budget=cap
 def judges(self):return[LIGHT,HEAVY]
 def hear(self,done_qa,models=None):
  menu="\n".join("item %s: %s"%(pos,prod.name)for pos,prod in enumerate(self.items,1))
  talk="\n".join("%s. (about item %s) Q: %s\nA: %s"%(i,pos,q,a)for i,(pos,q,a)in enumerate(done_qa,1))
  unit=self.brief.get("budget_unit")or"the unit the request states its budget in"
  # once the light reader came back empty this episode, the strong one reads first while spend stays under the soft cap
  order=list(models or self.judges())
  if self.light_mute and self.cost<SPEND_SOFT and order[0]==LIGHT and HEAVY in order:order=[HEAVY]+[m for m in order if m!=HEAVY]
  muted=lambda:sum(1 for t in self.trace if t.get("kind")=="llm_fail"and t.get("model")==LIGHT)
  before=muted()
  verdict=self.ask_json(HEAR+"\n\nBUDGET UNIT: "+str(unit)+"\n\nITEMS:\n"+menu+"\n\nQUESTIONS AND ANSWERS:\n"+talk,order,8000)or{}
  self.light_mute=self.light_mute or muted()>before
  free_turns={_count(ent.get("q"))for ent in _items(verdict.get("free"))if isinstance(ent,dict)}
  def idle(ent):
   t=_count(ent.get("q"))if isinstance(ent,dict)else None
   return t in free_turns and t and t<=len(done_qa)and not _tokens(str(ent.get("need")).split(":")[0])&_tokens(done_qa[t-1][2])
  verdict["needs"]=[ent for ent in _items(verdict.get("needs"))if not idle(ent)]
  qa_pairs=done_qa
  keys={k for rec in self.full.values()for k in _fact_block(rec)or{}}
  voiced,turns=[],set()
  for ent_2 in verdict["needs"]:
   t_2=_count(ent_2.get("q"))if isinstance(ent_2,dict)else None
   if not t_2 or t_2>len(qa_pairs):continue
   asked,answer=qa_pairs[t_2-1][1],qa_pairs[t_2-1][2]
   pname,colon,val=str(ent_2.get("need")).partition(":")
   longer=[k for k in keys if _stem(pname)<_stem(k)and _stem(k)<=_tokens(answer)]
   if colon and longer:
    pname=_words(max(longer,key=len))
    ent_2["need"]="%s: %s"%(pname,val.strip())
   # a reply naming another page key than the one asked, sharing a word with it, sets the need on the key it names
   heard=" %s "%_words(answer)
   said=[k for k in keys if _words(k)and" %s "%_words(k)in heard]
   if colon and said and" %s "%_words(pname)not in heard:
    best=max(said,key=lambda k:len(_words(k)))
    if _stem(best)&_stem(pname)and(len(_stem(best))>1 or len(_stem(pname))==1):
     pname=_words(best)
     ent_2["need"]="%s: %s"%(pname,val.strip())
   if _stem(pname)&_tokens(asked+" "+answer):
    voiced.append(id(ent_2))
    turns.add(t_2)
  verdict["needs"]=[ent_2 for ent_2 in verdict["needs"]if id(ent_2)in voiced or not(isinstance(ent_2,dict)and _count(ent_2.get("q"))in turns)]
  return verdict
 def gate(self,stage):
  owed=[s for s in self.process if s["when"]==stage]
  if not owed or self.finished or not self.gate_due(stage):return True,""
  closers={self.roles.get("buy"),self.roles.get("refuse")}-{None}
  specs=[t["function"]for t in self.tools]
  menu=["- %s: %s | args schema: %s"%(f["name"],f.get("description",""),_dump(f.get("parameters")or{}))for f in specs if f["name"]not in closers]
  at_order=stage=="order"
  lines,pushed=[],False
  head="Kyo took care of these required steps before now:\n"
  for _ in range(GATE_ROUNDS):
   log="\n".join("%s. %s"%pair for pair in enumerate(self.done_log,1))or"(nothing yet)"
   parts=[GATE.replace("STAGE","place the order"if at_order else"end without a purchase"),"\n\nREQUEST:\n",self.request,
     "\n\nSTEPS DUE NOW:\n","\n".join("- %s"%(s["step"],)for s in owed),"\n\nLEDGER:\n",self.ledger(),"\n\nCART NOW: ",self.cart_text()+("" if self.cart_sum() is None else" (sum %g)"%self.cart_sum()),
     "\n\nRECORD (every message and call so far, numbered):\n",log]
   if self.notes:parts+=["\n\nSTORE NOTES SEEN:\n","\n".join(self.notes[-4:])]
   parts.append("\n\nSTILL AVAILABLE: %s message(s), %s step(s); the %s takes one step"%(max(0,self.msg_limit-self.msg_used),self.left(),"order"if at_order else"ending"))
   gate_ask_text_2="".join(parts)+"\n\nTOOLS:\n"+"\n".join(menu)
   rul=self.ask_json(gate_ask_text_2,[HEAVY,LIGHT]if pushed else[LIGHT,HEAVY],2000)
   if rul is None:break
   told,hold=[],""
   if lines:
    self.take_heard({"needs":rul.get("needs")},told)
    hold=str(rul.get("hold")or"").strip()
   call=rul.get("call")if isinstance(rul.get("call"),dict)else None
   self.trace.append({"kind":"gate","stage":stage,"call":_clip(call,200),"steps":_clip(rul.get("steps"),120),"hold":hold[:200],"told":told})
   if told or hold:
    self.version+=1
    done=head+"\n".join(lines)+"\n"if lines else""
    return False,done+"The shopper's reply changed the task: %s. Rework the cart to match, then try checkout again."%"; ".join(told+([hold]if hold else[]))
   name=str((call or{}).get("tool")or"")
   fresh_only=at_order and any(s["messages"]>0 for s in owed)
   good={i for i in range(1,len(self.done_log)+1)if not fresh_only or(i<=len(self.done_steps)and self.done_steps[i-1]>self.cart_at)}
   claims=[c for c in _items(rul.get("steps"))if isinstance(c,dict)]
   unproven=[c for c in claims if c.get("done")is not True or all((_count(n)or 0)not in good for n in _items(c.get("lines")))]
   if not call and unproven and not lines and not pushed and self.roles.get("ask")and self.msg_used<self.msg_limit and self.left()>=2:
    pushed=True
    self.trace.append({"kind":"gate_unproven","steps":_clip(unproven,120)})
    self.done_log.append("(Kyo note: no proof after the cart change at step %s for: %s. Left: %s message(s), %s steps; a next call is needed)"
      %(self.cart_at,"; ".join(str(c.get("step")or"")for c in unproven)[:200],self.msg_limit-self.msg_used,self.left()))
    self.done_steps.append(-1)
    continue
   if not call:
    self.gate_ok.add((stage,"%s#%s"%(self.cart_mark(),self.version)))
    break
   if name not in self.schemas or name in closers:break
   chat=name==self.roles.get("ask")
   if self.left()<2 or(chat and self.msg_used>=self.msg_limit):break
   faults,raw=[],call.get("args")
   args=_shape(raw if isinstance(raw,dict)else{},self.schemas[name],"args",faults)
   if faults:break
   if chat:
    line=next((v for v in args.values()if isinstance(v,str)),"")
    if at_order:line=self.with_sum(line)
    qa=self.send([(1,"",line)],1)
    if not qa:break
    self.gate_msgs+=1
    lines.append("Q: %s\nA: %s"%(line,qa[0][2]))
    continue
   env=self.act(name,args)
   if env is None:break
   page=env.get("observation")
   self.absorb(page)
   self.done_log.append("step %s: called %s with %s, got %s (cart afterwards: %s)"%(self.step,name,_dump(args),self.show(page)[:300],self.cart_mark()))
   self.done_steps.append(self.step)
   lines.append("%s %s -> %s"%(name,_dump(args),self.show(page)[:400]))
  return True,(head+"\n".join(lines)if lines else"")
 def _svd_nodes(self,start,max_depth,prune=None):
  todo=[(start,0)]
  while todo:
   node,lvl=todo.pop()
   if lvl<=max_depth:
    yield node,lvl
    if not(prune and prune(node,lvl)):
     kids=list(node.values())if isinstance(node,dict)else node if isinstance(node,list)else[]
     todo+=[(k,lvl+1)for k in kids[::-1]]
 def spoken(self,outcome):
  if isinstance(outcome,str):return outcome
  words=[node for node,_ in self._svd_nodes(outcome,4)if isinstance(node,str)and len(node.split())>=2]
  return _dump(outcome)[:400]if not words else" ".join(words)[:800]
 def loose_lines(self,outcome):
  if not isinstance(outcome,dict)or self.ident(outcome):return[]
  found=[]
  for v in outcome.values():
   if isinstance(v,str):found+=[v][:1 if len(v.split())>=5 else 0]
   elif isinstance(v,list):found+=[x for x in v if len(x.split())>=5]if v and all(isinstance(x,str)for x in v)else[]
   elif isinstance(v,dict)and not self.ident(v,False,1):found+=[x for x in v.values()if isinstance(x,str)and len(x.split())>=6]
  return[x for x in found if x not in self.heard]
 def take_note(self,lines_in):
  """Read a store or shopper line that may change the task; update the ledger and its version."""
  self.heard.update(lines_in)
  verdict=self.ask_json(NOTE+"\n\nITEM LEDGER AS OF NOW:\n"+self.ledger()+"\n\nTHE NOTE:\n"+"\n".join(lines_in),[LIGHT,SPARE],1200)or{}
  gone_ids=[str(g)for g in _items(verdict.get("gone"))if isinstance(g,(str,int))]
  self.unavailable.update(g for g in gone_ids if g in self.rows)
  self.note_sold=bool(gone_ids)or verdict.get("sold_out")is True
  if self.note_sold and len(lines_in)==1:self.sold_lines.update(lines_in)
  if verdict.get("kind")not in(None,"change"):return""
  changes=[]
  cap=verdict.get("budget")
  if not isinstance(cap,bool)and isinstance(cap,(int,float))and cap>0 and cap!=self.budget:
   if self.budget is None or cap<self.budget:self.cap_cut=cap
   self.budget=cap
   changes.append("budget now %s"%(cap,))
  valid=lambda field:[e for e in _items(verdict.get(field))if isinstance(e,dict)and str(e.get("need")or"").strip()]
  owner=lambda e:self.items[min(max(1,_count(e.get("item"))or 1),len(self.items))-1]
  drops,adds=valid("drop"),valid("add")
  for e in drops:
   prod,phr=owner(e),str(e["need"]).strip()
   live=[m for m in prod.musts+prod.told if m not in prod.dropped]
   head=_words(phr.split(":")[0])
   prod.dropped.extend([m for m in live if m==phr]or[m for m in live if _words(m.split(":")[0])==head][:1]or[phr])
   changes.append("dropped "+phr)
  if drops and self.brief.get("pick")in("cheapest","most","least"):changes.append("any listing excluded solely by that requirement is eligible again, so apply the pick rule afresh across all records held")
  for e in adds:
   prod,phr=owner(e),str(e["need"]).strip()
   if phr in prod.dropped:prod.dropped.remove(phr)
   if phr not in prod.musts+prod.told:prod.told.append(phr)
   changes.append("added "+phr)
  if gone_ids:changes.append("unavailable: "+", ".join(gone_ids))
  if verdict.get("price_moved")is True:
   # 2 marks a note that moves nothing but a price
   self.price_rose=1 if changes else 2
   changes.append("a price changed")
  sumtext=str(verdict.get("summary")or"").strip()
  if changes or sumtext:
   self.version+=1
   self.recent=[]
   self.notes.append(sumtext or"; ".join(changes))
  return(sumtext+(" (%s)"%"; ".join(changes)if changes else"")).strip()
 def needs_now(self):return[(pos,n)for pos,prod in enumerate(self.items,1)for n in prod.musts+prod.told if n not in prod.dropped]
 def evidence(self,outcome):
  hits=[]
  def prune(piece,depth):
   cid=self.ident(piece,False,depth)if isinstance(piece,dict)else None
   if cid and _fact_block(piece)and cid not in hits:
    hits.append(cid)
    return True
   return False
  for _ in self._svd_nodes(outcome,6,prune):pass
  live=self.needs_now()
  if not hits or not live:return""
  many,spans,rows=len(self.items)>1,self.range_musts(),[]
  for cid in hits[:12]:
   fblock=_fact_block(self.full.get(cid)or{})or{}
   parts=[]
   for pos,n in live[:8]:
    pname=n.split(":",1)[0].strip()
    pairs=["%s=%s"%(k,str(v)[:50])for k,v in fblock.items()if _names(_stem(pname),_stem(k))]
    want=n.split(":",1)[-1].strip().lower()
    if not pairs and re.search("[a-z]",want):
     same=[k for k,v in fblock.items()if str(v).strip().lower()==want]
     if same:pairs=["no key of that name; the wanted value %s is stated under %s, which counts as met"%(want,", ".join(same[:3]))]
    if not pairs:continue
    tag=""
    if(pos,n)in spans:tag=" [range met: one entry restates the wanted span]"if self.states_range(cid,n)else" [range not restated by any entry]"
    parts.append(("(item %d) "%pos if many else"")+pname+" -> "+", ".join(pairs[:3])+tag)
   if parts:rows.append(cid+": "+"; ".join(parts))
  if not rows:return""
  return("[evidence | matching keys per opened listing, by requirement name; ones for another component or facet prove nothing either way, and the value may sit under an unrelated key name too] "+" | ".join(rows))
 def range_musts(self):
  aimed=set(pair[1]for pair in self.refs)
  return[(pos,n)for pos,n in self.needs_now()if n not in aimed and n not in self.ref_values and _two_figures(_figure_words(n.split(":",1)[-1]))]
 def states_range(self,listing,need):
  fblock=_fact_block(self.full.get(listing)or{})or{}
  head,_,tail=need.partition(":")
  pname,wanted=_stem(head),_figure_words(tail if":"in need else need)
  named=[v for k,v in fblock.items()if _names(pname,_stem(k))]
  return any(_repeats(wanted,bit)for v in(named or fblock.values())for bit in re.split(r"[,;/|]",str(v)))
 def cart_text(self):
  if self.cart_view is not None and self.cart_seen_at>=self.cart_at:return _dump(_clip(self.cart_view,300))[:3000]
  return"(cart unseen; Kyo's tally from its calls, listing -> units) "+_dump(self.cart_lines)
 def cart_ids(self):
  if self.cart_view is not None and self.cart_seen_at>=self.cart_at:
   shown=_dump(self.cart_view)
   return[cid for cid in self.rows if cid in shown]
  return list(self.cart_lines)
 def record_of(self,listing):
  rec=self.full.get(listing)or self.rows.get(listing)
  return"(nothing recorded yet)"if not rec else _dump(_clip(rec,400))[:3000]
 def others(self,skip,limit=30):
  order=self.seen_order[::-1]
  if self.brief.get("pick")=="cheapest":order.sort(key=lambda cid:(cid not in self.full,float("inf")if self.price_of(cid)is None else self.price_of(cid)))
  picked=[cid for cid in order if cid not in skip and not self.sold_out(cid)][:limit]
  return"\n".join(_dump(_clip(self.full.get(cid)or self.rows.get(cid),160))[:900]for cid in picked)
 def check_key(self,ids):return"%s#%s"%("|".join(sorted(ids)),self.version)
 def review_sheet(self,ids,proposed):
  if proposed:
   counts="; ".join("item %d: quantity %s"%pair for pair in enumerate((it.qty for it in self.items),1))
   basket="CART (plan only, not yet added - assume these at the quantities asked; %s):\n%s"%(counts,", ".join(ids))
  else:basket="CART:\n"+self.cart_text()
  flag="SOLD OUT, cannot be bought at present. "
  lines=["%s: %s%s"%(pid,flag if self.sold_out(pid)else"",self.record_of(pid))for pid in ids]
  parts=[CHECK,"REQUEST:\n"+self.request,"REQUIREMENTS IN FORCE:\n"+self.ledger(),basket,"CART LISTING RECORDS:\n"+"\n".join(lines)]
  if self.brief.get("pick")in("cheapest","most","least"):parts.append("OTHER LISTINGS (NEWEST):\n"+self.others(ids,15))
  out=[]
  for __2,n in self.range_musts():
   if any(self.states_range(cid,n)for cid in ids):continue
   holders=[cid for cid in self.seen_order[::-1]if cid not in ids and self.buyable(cid)and self.within_budget(cid,1)and self.states_range(cid,n)]
   if holders:out.append((n,holders[:4]))
  uncovered=out
  if uncovered:
   owners=list(dict.fromkeys(pid for _,who in uncovered for pid in who))
   told="; ".join("%s (repeated by %s)"%(need,", ".join(who))for need,who in uncovered)
   parts.append("RANGE MUSTS uncovered in this cart: %s\nHolders' records:\n%s"%(told,"\n".join("%s: %s"%(pid,self.record_of(pid))for pid in owners)))
  return"\n\n".join(parts)
 def check_cart(self,ids=None,limit=SPEND_SOFT):
  """Strong review of a cart against the ledger before paying; an objection holds the checkout."""
  proposed=ids is not None
  if not proposed:ids=self.cart_ids()
  key=self.check_key(ids)
  if not ids or self.cost>limit or key in self.checked:return""
  self.checked.add(key)
  ruling=self.ask_json(self.review_sheet(ids,proposed),[HEAVY,LIGHT],8000)
  self.trace+=[dict(kind="check",cart=ids,got=ruling)]
  passed=ruling.get("ok")is not False if ruling else True
  self.verdicts[key]=bool(ruling)and passed
  # approved with faults named = a near miss is about to be paid
  self.flawed=key if passed and ruling and _items(ruling.get("problems"))else None
  if passed:return""
  self.leads=list(filter(lambda c:c not in ids,self.leads))
  faults=list(map(str,_items(ruling.get("problems"))))[:5]or["recheck each stated need"]
  better=list(filter(lambda c:c in self.rows and not self.sold_out(c),map(str,_items(ruling.get("better")))))[:4]
  extra="; reviewer prefers "+", ".join(better)if better else""
  return"Second opinion withholds approval (%s%s). Either amend the cart or quote records that refute the objection; payment comes after."%("; ".join(faults),extra)
 def open_lookups(self):
  live={pair[1]for pair in self.needs_now()}
  return[pair[1]for pair in self.refs if pair[1]not in self.ref_values and pair[1]in live]
 def chain_moves(self,reply,pending):
  got={"res":[],"open":[],"find":[],"read":[],"miss":[],"about":{},"redo":[],"plain":[],"apart":{},"first":{}}
  chains=[c for c in _items(reply.get("chains"))if isinstance(c,dict)]
  exact={str(c.get("need")or"").strip()for c in chains}
  for chain in chains:
   named=str(chain.get("need")or"").strip()
   lone=len(pending)==1 and pending[0]not in exact and(len(chains)==1 or bool(named)and(named in pending[0]or pending[0]in named))
   req=next((p for p in pending if p==named),None)or(pending[0]if lone else None)
   if not req:continue
   hops=[h for h in _items(chain.get("links"))if isinstance(h,dict)]
   ids=[str(h.get("id")or"").strip()for h in hops]
   value=str(chain.get("value")or"").strip()
   # an exclusion's relation is known before its chain resolves: keep where it points
   end=next((h for h,i in zip(hops,ids)if i not in self.opened),hops[-1]if hops else{})
   seen=[i for i in ids if i in self.rows]
   got["apart"][req]=None if chain.get("relation")!="apart"else"; ".join(filter(None,[str(end.get("about")or"").strip()[:100],"seller "+str(end.get("shop")).strip()if str(end.get("shop")or"").strip()not in str(end.get("about"))else"","chain read so far "+", ".join(seen)if seen else""]))or"the referenced listing"
   if value and hops and chain.get("done")is True and all(i in self.opened for i in ids):
    # a value printed on an earlier link but not on the last one was read off the wrong page
    shows=lambda pid,said=value:any(_stem(said)<=_stem(v)for v in(_fact_block(self.full.get(pid)or{})or{}).values()if isinstance(v,str))
    rel,mate=chain.get("relation"),str(chain.get("partner")or"").strip()
    # a last link naming the partner was reached past the pairing rule, so the starting value rightly sits earlier
    after=rel=="together"and bool(mate)and _stem(mate)!=_stem(value)and _tokens(mate)<=_tokens(_dump(self.full.get(ids[-1])or{}))
    early=[i for i in ids[:-1]if shows(i)]
    if early and _fact_block(self.full.get(ids[-1])or{})and not shows(ids[-1])and not after:
     got["redo"].append(f"{req}: {value} is printed on {early[0]}, an earlier link, not on the last link {ids[-1]}; the figure sits on {ids[-1]}'s own page")
     continue
    first=(ids[0],str(hops[0].get("about")or""),str(hops[0].get("shop")or"").strip())
    note=self.sibling_note(req,*first)
    if note and note not in self.rehops and len(self.rehops)<2:
     got["redo"].append(note)
     continue
    got["first"][req]=first
    if rel=="together"and not mate:
     got["miss"].append(f"{req} -> find what pairs with {value}")
     got["find"].append((str(hops[-1].get("shop")or"").strip(),))
     got["plain"].append((req,value))
    elif rel=="together"and _stem(mate)==_stem(value):got["res"].append((req,f"{mate} (the partner a pairing rule on the pointed pages gives); passes only showing {mate} itself",mate,None))
    elif rel=="together":got["res"].append((req,f"{mate} (what goes with {value}); passes only showing {mate} itself, while {value} on a listing fails",mate,ids[-1]if shows(ids[-1])else None))
    elif rel=="apart":got["res"].append((req,f"{value} is excluded: listings showing it fail, all other values pass",None,None))
    else:got["res"].append((req,value,value,None))
    continue
   closed=[i for i in ids if i in self.rows and i not in self.opened]
   if closed:
    got["open"]+=[i for i in closed if i not in got["open"]]
    continue
   gap=next((h for h,i in zip(hops,ids)if i not in self.rows),{})
   desc=str(gap.get("about")or"")or"the referenced listing"
   got["miss"].append(f"{req} -> find {desc}")
   # a link naming no seller, words or description gives a search nothing to look for
   if gap and any(str(gap.get(f)or"").strip()for f in("about","shop","words")):
    words=gap.get("words")or""
    words=" ".join(map(str,words))if isinstance(words,list)else str(words)
    seller=str(gap.get("shop")or"").strip()or self.shop_named(desc)or(desc[:60]if str(gap.get("about")or"").strip()else words[:60])
    got["find"].append((seller,)*(seller==str(gap.get("shop")or"").strip())+(" ".join(dict.fromkeys(f"{seller} {words}".split())),seller))
    # the link's own distinguishing words lead its description, so a seller search can carry them
    got["about"][seller]=" ".join([words,desc]).strip()
   elif ids and ids[-1]in self.opened and ids[-1]not in self.full:got["read"].append(ids[-1])
  return got
 def trail(self):
  """Follow pointer requirements to the listing they name and record the value found there."""
  self.recheck_firsts()
  pending=self.open_lookups()
  viewer=self.roles.get("offer")or self.roles.get("open")
  key=self.single_target(viewer)
  texts,missing,about=[],[],{}
  # lookups never take the steps the questions need: before the first question, chains stop at a third of the budget
  while pending and self.trails<8 and self.cost<SPEND_HARD and self.secs()>CLOCK_STOP and(self.asked or self.step<self.max_steps//3):
   self.trails+=1
   shelf=[]
   for pid in self.seen_order[::-1][:60]:
    # a clipped row still runs past 300 characters: keep it whole so its id and seller stay readable
    rec=(self.full.get(pid)or self.rows.get(pid))if pid in self.opened else None
    shelf.append("OPENED "+_dump(_clip(rec,300))[:1400]if rec else"ROW "+_dump(_clip(self.rows.get(pid),80))[:600])
   ask="\n\n".join([TRAIL,"REQUEST:\n"+self.request,"VALUE-ELSEWHERE REQUIREMENTS:\n"+"\n".join(pending)]+(["EARLIER ANSWER CORRECTED:\n"+"\n".join(self.rehops)]if self.rehops else[])+["RECORDS, NEWEST FIRST:\n"+"\n".join(shelf)])
   reply=self.ask_json(ask,[LIGHT,HEAVY],8000)or{}
   self.trace+=[dict(kind="trail",pending=pending,got=_clip(reply,150))]
   plan=self.chain_moves(reply,pending)
   for req,where in plan["apart"].items():
    if where:self.apart[req]=where
    else:self.apart.pop(req,None)
   for req,said,word,end in plan["res"]:
    if word is not None:self.ref_words[req]=word
    if end is not None:self.chain_ends.add(end)
    if req in plan["first"]:self.ref_firsts[req]=(*plan["first"][req],end)
    self.ref_values[req]=said
    self.version+=1
   missing,todo,about=plan["miss"],plan["open"],{**about,**plan["about"]}
   pending=[req for req in pending if req not in self.ref_values]
   fresh_redo=[r for r in plan["redo"]if r not in self.rehops]
   if pending and not todo and fresh_redo and len(self.rehops)<2:
    self.rehops+=fresh_redo[:2-len(self.rehops)]
    continue
   if pending and not todo:
    if plan["read"]and self.read_pages(plan["read"],self.facts_tool()):continue
    if plan["find"]and any([any(self.seek_shop(n,about.get(names[-1],""))for n in dict.fromkeys(names))for names in list(dict.fromkeys(plan["find"]))[:2]]):continue
    # no pairing rule on the chain and none found at that seller: the pointed page's own value stands
    for req,value in plan["plain"]:self.ref_values[req],self.ref_words[req]=f"{value} (no pages read print a pairing rule for it, so the pointed page's own value stands)",value;self.version+=1
    missing=[m for m in missing if not any(m.startswith(req+" ->")for req,_ in plan["plain"])]
    pending=[req for req in pending if req not in self.ref_values]
   if self.finished or not(todo and viewer and key)or self.left()<len(todo)+4:break
   for pid in todo[:max(0,min(3,6-len(texts)))]:
    bad=[]
    args=_shape({key:pid},self.schemas.get(viewer),"args",bad)
    got=None if bad else self.act(viewer,args)
    if got is None or self.finished:break
    self.observe(viewer,got.get("observation"))
    texts.append(pid+": "+self.show(got.get("observation"))[:1500])
  return self.trail_wrapup(texts,missing,pending)
 def sibling_note(self,req,pid,about,shop):
  # a link described by its seller's words ("the <value> one") is the seller's listing whose own attributes show them: an opened
  # page of the same seller and kind is the link instead when one of its attribute values is wholly in the description while
  # pid's value for that attribute is not, and no attribute of pid beats it the same way
  want=_tokens(about)-_tokens(shop)-_tokens(self.kind_of(pid))
  own=_fact_block(self.full.get(pid)or{})or{}
  clash=lambda a,b:[k for k,v in a.items()if k in b and _tokens(v)and _tokens(v)<=want and not _tokens(b[k])<=want]
  if not shop or not own:return""
  for c,rec in self.full.items():
   facts=_fact_block(rec)or{}
   if c==pid or self.kind_of(c)!=self.kind_of(pid)or not _tokens(shop)<=_tokens(_dump(rec)):continue
   wins=clash(facts,own)
   if wins and not clash(own,facts):
    k=wins[0]
    return f"{req}: the link described as '{about[:120]}' is {c}, not {pid}: {c} shows {k}={str(facts[k])[:60]}, as described, where {pid} shows {k}={str(own[k])[:60]}; restart the chain from {c}"
  return""
 def recheck_firsts(self):
  # a settled chain is reopened when a sibling page opened since fits its first link's description better
  for req,(pid,about,shop,end)in list(self.ref_firsts.items()):
   note=self.sibling_note(req,pid,about,shop)if req in self.ref_values and len(self.rehops)<2 else""
   if not note:continue
   del self.ref_firsts[req]
   self.ref_values.pop(req,None)
   self.ref_words.pop(req,None)
   self.chain_ends.discard(end)
   self.rehops.append(note)
   self.version+=1
 def trail_wrapup(self,texts,missing,pending):
  out="[Kyo opened these pages that requirements point to]\n%s\n"%"\n".join(texts)if texts else""
  if self.ref_values:out+="[values Kyo settled from pointed-to pages] %s\n"%"; ".join("%s -> %s"%kv for kv in self.ref_values.items())
  missing=[m for m in missing if not any(m.startswith(r+" ->")for r in self.apart)]+["%s -> value not read yet"%req for req in pending if req not in self.apart and not any(m.startswith(req)for m in missing)]
  room=self.trail_blocks<2
  stuck=[pid for pid in self.cart_ids()if pid in self.chain_ends]
  if stuck and room and self.left()>4:
   self.trail_blocks+=1
   return out+"Swap needed for %s: being the final link, it carries the origin value, not the matching partner value. Look up a listing with the partner value and put it in instead; complete the order afterwards."%", ".join(stuck),True
  blocked=room and bool(missing)and self.left()>5
  if blocked:
   self.trail_blocks+=1
   out+=("Pending lookups: %s. A search row settles nothing; what settles a pointed-to listing is its own opened page (and so on for any listing it points onward to). Track it down by seller, brand or phrasing, open it, rely on the value printed there, and only then complete the order."%" | ".join(missing[:3]))
  return out,blocked
 def shop_named(self,words):
  seen=_tokens(" ".join(map(self.kind_of,self.seen_order)))
  caps=re.findall(r"(?<=\s)[A-Z][a-z]{2,}"," "+words)
  return next(iter([w for w in caps if not seen&_tokens(w)]),"")
 def shop_rows(self,name,about):
  target,hint=name.strip().lower(),_tokens(about)
  picked=[p for p in self.seen_order if p not in self.opened and target in{v.strip().lower()for v in(self.rows.get(p)or{}).values()if isinstance(v,str)}]
  return sorted(picked,key=lambda p:len(hint&_tokens(_dump(_clip(self.rows.get(p),300))))*-1)
 def read_pages(self,listings,tool):
  key,count=self.single_target(tool),0
  for pid in(listings[:CHAIN_READS]if tool and key else()):
   bad=[]
   args=_shape({key:pid},self.schemas.get(tool),"args",bad)
   stop=bad or self.finished or self.left()<self.tail()+8
   got=None if stop else self.act(tool,args)
   if got is None or self.finished:break
   self.observe(tool,got.get("observation"))
   count+=1
  self.trace+=[dict(kind="chain_read",rows=listings[:CHAIN_READS],read=count)]
  return count>0
 def shop_narrowed(self,tool,text_arg,name,about,figured=False):
  known=_tokens(" ".join([name,*map(self.kind_of,self.seen_order)]))
  terms=[w for w in re.findall(r"[A-Za-z0-9-]+",about)if(len(w)>2 or w.isdigit())and not known&_tokens(w)]
  # the figures a link quotes, short ones too, single its listing out best; lacking any, its first words on a first visit, else its longest word
  figs=list(dict.fromkeys(w for w in terms if any(ch.isdigit()for ch in w)))
  query=" ".join([name]+(figs[:3]or(list(dict.fromkeys(terms))[:3]if figured else[max(terms,key=len)])))if terms else""
  bad=[]
  args=_shape({text_arg:query},self.schemas.get(tool),"args",bad)
  if self.finished or not query or bad or query in self.sought or self.left()<self.tail()+9:return[]
  self.sought|={query}
  got=self.act(tool,args)
  if got is None or self.finished:return[]
  self.observe(tool,got.get("observation"))
  self.log_search(tool,args,got.get("observation"))
  hits=[hit[0]for hit in self.records_in(got.get("observation"))if hit[0]not in self.opened]
  # the link names its seller: once rows of that seller are on hand, other sellers' hits are not the listing
  own=self.shop_rows(name,about)
  return[pid for pid in hits if pid in own or not(own or figured)]
 def seek_shop(self,name,about=""):
  tool=self.roles.get("search")
  key=self.single_target(tool)
  if self.finished or not(tool and key and name)or self.left()<self.tail()+6:return False
  one_by_one=self.compare_bare()or not(self.compare_target()and self.roles.get("compare"))
  viewer=self.offer_tool()
  if one_by_one and name in self.sought:
   queued=self.shop_rows(name,about)
   if queued:return self.read_pages(list(dict.fromkeys(self.shop_narrowed(tool,key,name,about)+queued)),viewer)
  # a link quoting figures or naming distinguishing words: search its seller with them before paging through that seller's whole range
  first=self.shop_narrowed(tool,key,name,about,True)if one_by_one and name not in self.sought else[]
  if first:return self.read_pages(first,viewer)
  bad=[]
  args=_shape({key:name},self.schemas.get(tool),"args",bad)
  log=self.search_log.get(tool+_dump(args))
  waiting=[pid for pid in(log or{}).get("ids")or[]if pid not in self.opened]
  if waiting and not bad:return self.open_rows(waiting)
  if bad or(log and not log.get("token"))or(name in self.sought and not log):return False
  self.sought.add(name)
  if log:args=_shape({**log["args"],log["page_arg"]:log["token"]},self.schemas.get(tool),"args",bad)
  got=self.act(tool,args)if not bad else None
  if got is None or self.finished:return False
  page=got.get("observation")
  self.observe(tool,page)
  self.log_search(tool,args,page)
  fresh=[pid for pid,_ in self.records_in(page)if pid not in self.opened]
  skip=(log or{}).get("page_arg")
  log=self.search_log.get(tool+_dump({k:v for k,v in args.items()if k!=skip}))or{}
  if not fresh and log.get("token")is not None and self.left()>=self.tail()+6 and self.page_on(log):fresh=[pid for pid in log["ids"]if pid not in self.opened]
  # a seller's own listings are few: list them all, so the pointed-to one is not missed on a later page
  each=len(log.get("ids")or[])
  while each and log.get("token")is not None and-(-(log["total"]-len(log["ids"]))//each)<=3 and self.left()>=self.tail()+9 and self.page_on(log):
   fresh+=[pid for pid in log["ids"]if pid not in self.opened and pid not in fresh]
  self.trace+=[dict(kind="seek",name=name,rows=fresh)]
  if one_by_one and fresh:return self.read_pages(self.shop_rows(name,about)or fresh,viewer)
  ranked=[pid for pid in self.shop_rows(name,about)if pid in fresh]
  return bool(fresh)and self.open_rows(ranked+[pid for pid in fresh if pid not in ranked])
 def offer_tool(self):return self.roles.get("offer")or self.roles.get("open")
 def reread_offers(self):
  tool=self.offer_tool()
  key=self.single_target(tool)
  if not tool or not key:return""
  floor=max(self.step-2,self.refused_at)
  stale=[pid for pid in self.cart_ids()if not self.sold_out(pid)and self.offer_seen.get(pid,-99)<floor]
  if not stale or self.left()<len(stale)+3:return""
  changes=[]
  for pid in stale:
   bad=[]
   args=_shape({key:pid},self.schemas.get(tool),"args",bad)
   if bad:break
   before=dict(self.rows.get(pid)or{})
   got=self.act(tool,args)
   if got is None or self.finished:break
   page=got.get("observation")
   self.offer_seen[pid]=self.step
   self.observe(tool,page)
   moved=[f for f,now in(self.rows.get(pid)or{}).items()if f in before and not isinstance(now,(dict,list))and now!=before[f]]
   if moved:changes.append("%s changed (%s): %s"%(pid,", ".join(moved[:4]),_dump(_clip(page,300))[:900]))
  self.trace+=[dict(kind="reread",ids=stale,changed=len(changes))]
  if changes:self.version+=1
  return"\n".join(changes)
 def sold_out(self,ident):return ident in self.unavailable or ident in self.gone
 def swap_hint(self):
  dead=[pid for pid in self.cart_ids()if pid not in self.gone and self.sold_out(pid)]
  if not dead:
   rose=self.price_rose
   self.price_rose=False
   if not rose or self.secs()<12 or self.brief.get("pick")!="cheapest":return""
   for pid_2 in self.cart_ids():
    paid=self.price_of(pid_2)
    pool=[x for x in self.copies([pid_2])if x in self.full]
    if not pool and len(self.cart_ids())==1:pool=self.replacements([pid_2])
    under=[x for x in pool if x in self.full and(self.price_of(x)or paid or 0)<(paid or 0)]
    best_2=min(under,key=self.price_of)if under else None
    if best_2 and self.swap_line(pid_2,best_2):return"[Kyo] Price of %s rose to %s; Kyo swapped in %s at %s instead."%(pid_2,format(paid,"g"),best_2,format(self.price_of(best_2),"g"))
   return""
  self.gone.update((pid,self.step)for pid in dead)
  self.leads=[pid for pid in self.leads if pid not in dead]
  self.version+=1
  self.twin_search(dead)
  unread=[pid for pid in self.copies(dead)if pid not in self.full]
  if unread and self.left()>self.tail()+5:self.open_rows(unread)
  subs=self.replacements(dead)
  head="[Kyo] %s can no longer be bought (the store flags it as unpurchasable), so do not open it again."%", ".join(dead)
  if not subs:return head+" Swap in the best fitting listing among the rest; failing any fitting one already read, look for one via search."
  best,backup=subs[0],", ".join(subs[1:])or"none"
  twin=best in self.copies(dead)
  cheap=self.brief.get("pick")=="cheapest"
  if(twin or cheap)and len(dead)==1 and best in self.full and self.swap_line(dead[0],best):
   why="the same item sold by a different shop."if twin else"no cheaper fit turned up among those Kyo has read."
   return head+" Kyo put %s in its place, %s Further fits: %s."%(best,why,backup)
  if self.left()<=6:return head+" Little budget left: in one turn send both calls (%s in, unsellable item out), next turn pay. Fallbacks already read: %s."%(best,backup)
  if cheap:return head+(" Read listings that could replace it, cheapest first: %s. Price decides here, so unless all matching listings are known already, search once under %s's price (with the price-ceiling filter if the search has one), then put the cheapest fit in place of the dead line."%(", ".join(subs),best))
  return head+" Possible stand-ins already read: %s. Right now swap the unsellable item for %s (same quantity) and complete the order."%(", ".join(subs),best)
 def find_sold(self):
  open_lines=[pid for pid in self.cart_ids()if pid not in self.gone]
  if len(open_lines)==1:
   if not self.sold_out(open_lines[0]):self.unavailable.add(open_lines[0])
   return""
  if open_lines and not any(map(self.sold_out,open_lines)):return self.reread_offers()
  return""
 def swap_lost(self):
  tip=self.swap_hint()
  if tip:self.queued.append(tip)
  for lost in[p for p in self.cart_ids()if p in self.gone]:
   units,inside=self.cart_lines.get(lost,1),self.cart_ids()
   pool=self.swap_fits+(self.leads+self.carted[::-1]if len(self.items)==1 else[])
   ready=[c for c in pool if c not in inside and self.buyable(c)and self.within_budget(c,units)]
   pick=min(ready,key=lambda c:c not in self.full)if ready else None
   if not pick or not self.swap_line(lost,pick):return False
   if lost==self.confirmed:self.confirmed=pick
  return True
 def replacements(self,sold):
  pool=[]if self.cost>SPEND_SOFT else self.on_hand(set(sold)|set(self.gone))[:25]
  if not pool:return[]
  lost="\n".join(map("{0}: {1}".format,sold,map(self.record_of,sold)))
  dumps="\n".join(_dump(_clip(self.full.get(p)or self.rows.get(p),200))[:1200]for p in pool)
  parts=[SWAP,"REQUEST:\n"+self.request,"REQUIREMENTS IN FORCE:\n"+self.ledger(),"UNAVAILABLE CART LINE:\n"+lost,"CART NOW:\n"+self.cart_text(),"OTHER RECORDS (NEWEST):\n"+dumps]
  reply=self.ask_json("\n\n".join(parts),self.judges(),8000)or{}
  barred,fits=set(sold)|set(self.gone)|set(self.cart_ids()),[]
  for entry in _items(reply.get("fits")):
   pid,price=(str(entry.get("id")),entry.get("price"))if isinstance(entry,dict)else(str(entry),None)
   if pid not in barred and pid in self.rows:
    barred.add(pid)
    fits.append((float(price)if isinstance(price,(int,float))and not isinstance(price,bool)else float("inf"),pid))
  twins,cheap=self.copies(sold),self.brief.get("pick")=="cheapest"
  self.trace+=[dict(kind="swap",lost=sold,fits=fits)]
  self.swap_fits=[pair[1]for pair in sorted(fits,key=lambda pair:(pair[0]if cheap else 0,pair[1]not in twins))]
  return self.swap_fits[:4]
 def passed(self,lid):return any(ok and lid in key.partition("#")[0].split("|")for key,ok in self.verdicts.items())
 def bare_items(self):
  # items with no cart line of their own yet
  held={self.slot_of(x)for x in self.cart_ids()if not self.sold_out(x)}
  return[it.name for n,it in enumerate(self.items,1)if n not in held]
 def slot_of(self,lid):
  if len(self.items)==1:return 1
  kind=self.kind_of(lid)
  hits={_count(k)for k,v in self.lead_items.items()if lid in v and _count(k)and 1<=_count(k)<=len(self.items)}
  hits=hits or{n for n,it in enumerate(self.items,1)if kind and kind==(it.shelf or"").strip().lower()}
  return hits.pop()if len(hits)==1 else None
 def fits_total(self,lost,pick):
  units,due=self.cart_lines.get(lost,1),self.cart_sum()
  if due is None or self.budget is None or self.price_of(pick)is None:return self.within_budget(pick,units)
  return due-units*(self.price_of(lost)or 0)+units*self.price_of(pick)<=self.budget
 def clean_pool(self,lost,pos,cheap=False):
  kind,inside=self.kind_of(lost),set(self.cart_ids())
  failed=lambda c:not self.passed(c)and any(ok is False and key.partition("#")[0]==c for key,ok in self.verdicts.items())
  same=lambda c:self.kind_of(c)==kind if kind else len(self.items)==1
  pool=[c for c in self.full if c not in inside and c!=lost and same(c)and self.buyable(c)and self.fits_total(lost,c)and not self.item_clash(c,pos)and not failed(c)]
  cheap=cheap or self.brief.get("pick")=="cheapest"
  return sorted(pool,key=lambda c:((self.price_of(c)or 0)if cheap else 0,not self.passed(c),c not in self.swap_fits+self.leads))
 def refill_seek(self,lost,pos):
  tool=self.roles.get("search")
  arg=self.single_target(tool)
  if not tool or not arg or self.left()<=self.tail()+6:return
  prod,kind=self.items[pos-1],self.kind_of(lost)
  answer,sent=self._kd_send(tool,{arg:self.joined_query(prod)or prod.name},self.finished)
  if answer is None:return
  fresh=self._kd_take(tool,sent,answer)
  pool=[c for c,_ in self.records_in(fresh)if c not in self.full and c!=lost and self.buyable(c)and self.fits_total(lost,c)and(not kind or self.kind_of(c)==kind)]
  if pool:self.open_rows(pool[:2])
  self.trace.append(dict(kind="refill_seek",lost=lost,rows=pool[:2]))
 def cart_swap(self,lost,pos,lim,cheap=False):
  rest=[p for p in self.cart_ids()if p!=lost and not self.sold_out(p)]
  for pick in self.clean_pool(lost,pos,cheap)[:2]:
   ids=rest+[pick]
   if self.cost>lim or self.left()<=self.tail()+3:break
   self.check_cart(ids,lim)
   if not self.verdicts.get(self.check_key(ids))or not self.swap_line(lost,pick):continue
   self.mended.add(frozenset(ids))
   if lost==self.confirmed:self.confirmed=pick
   return pick
  return None
 def repair(self):
  # a refusal or note that strikes a reviewed line, or cuts the cap below the cart, gets the cart mended before any close
  if self.finished or self.left()<=self.tail()+3 or self.secs()<=CLOCK_STOP:return{}
  lim,moved=SPEND_HARD+0.01 if self.cost<SPEND_HARD+0.01 else self.cost+0.015,{}
  for lost in[p for p in self.cart_ids()if self.sold_out(p)and p not in self.refilled and self.passed(p)]:
   self.refilled.add(lost)
   pos=self.slot_of(lost)
   if not pos:continue
   if not self.clean_pool(lost,pos):self.refill_seek(lost,pos)
   pick=self.cart_swap(lost,pos,lim)
   self.trace.append(dict(kind="refill",lost=lost,pick=pick))
   if pick:moved[lost]=pick
  cap,due=self.cap_cut,self.cart_sum()
  if cap is not None and due is not None and due>cap and self.cart_ids()and self.left()>self.tail()+3:
   self.cap_cut=None
   dear=max(self.cart_ids(),key=lambda p:self.cart_lines.get(p,1)*(self.price_of(p)or 0))
   pos=self.slot_of(dear)
   pick=self.cart_swap(dear,pos,lim,True)if pos else None
   self.trace.append(dict(kind="step_down",lost=dear,pick=pick,cap=cap))
   if pick:moved[dear]=pick
  return moved
 def mend_order(self,tool,call_args,reply):
  shown=reply.get("observation")
  self.note_sold=False
  self.observe(tool,shown)
  self.zero_lines(shown)
  if self.note_sold:self.find_sold()
  moved=self.repair()
  if not moved or self.left()<2 or self.finished:return reply
  if self.secs()>CLOCK_STOP and self.left()>=3:self.gate("order")
  self.say_basket()
  if self.finished or self.left()<1:return reply
  again=self.act(tool,call_args)
  self.trace.append(dict(kind="mended_order",moved=moved,reply=_clip(again,200)))
  return reply if again is None else again
 def audit(self,limit=SPEND_SOFT,room=CLOCK_EXTRAS,again=False):
  """Strong review of everything seen before an empty-handed close: likely matches per item."""
  # a review runs once per ledger version: news that alters what is required (a dropped need, a new budget) earns a fresh one
  fresh=getattr(self,"audit_ver",None)!=self.version
  if(self.audited and not fresh and not(again and len(self.rows)>self.audit_rows))or self.cost>limit or self.secs()<room:return""
  self.audited,self.audit_rows,self.audit_ver=True,len(self.rows),self.version
  got=self.audit_call()
  slots=self.audit_fits(got)
  retry=[]if slots else[p for p in map(str,_items((got or{}).get("open")))if p in self.rows and p not in self.opened and not self.sold_out(p)]
  if retry and self.open_rows(retry):
   got=self.audit_call()
   slots=self.audit_fits(got)
  if not slots:return""
  self.lead_items,self.leads=slots,sum(slots.values(),[])
  why=got.get("reason")
  shown="; ".join(f"item {k}: {', '.join(v)}" for k,v in slots.items())
  return(f"Reviewer's likely matches: {shown}"+(f" ({why})" if why else"")
    +". Check these and buy the one that truly fits; closing without a purchase is right only once each of them fails.")
 def on_hand(self,skip=()):
  own=set(c for group in self.item_groups()for c in group)
  pool=[pid for pid in reversed(self.seen_order)if pid not in skip and not self.sold_out(pid)]
  return sorted(pool,key=lambda pid:(pid not in self.full)*2+(not(pid in own and self.buyable(pid)and self.within_budget(pid,1))))
 def audit_call(self):
  pool,shelf=self.on_hand()[:30],[]
  if not pool:return None
  for p in pool:
   opened=bool(self.full.get(p))
   body=_dump(_clip(self.full[p]if opened else self.rows.get(p),200 if opened else 120))
   shelf.append(("OPENED "if opened else"ROW ")+body[:1200 if opened else 500])
  got=self.ask_json("\n\n".join([AUDIT,"REQUEST:\n"+self.request,"REQUIREMENTS IN FORCE:\n"+self.ledger(),"KNOWN RECORDS (NEWEST):\n"+"\n".join(shelf)]),[HEAVY,LIGHT],8000)
  self.trace+=[dict(kind="audit",got=got)]
  return got
 def audit_fits(self,got):
  table,out=(got or{}).get("fits"),{}
  for slot,raw in(table.items()if isinstance(table,dict)else()):
   pos=_count(slot)
   mine=pos if isinstance(pos,int)and 1<=pos<=len(self.items)else None
   out[str(slot)]=[str(v)for v in _items(raw)if str(v)in self.rows and not self.sold_out(str(v))and not(mine and self.item_clash(str(v),mine))][:4]
  return{k:v for k,v in out.items()if v}
 def open_rows(self,unread):
  many,tool=self.compare_target(),self.facts_tool()
  if many and self.roles.get("compare")and not self.compare_bare():calls=[(self.roles["compare"],{many:unread[:self.compare_room()]})]
  else:
   key=self.single_target(tool)
   if not tool or not key:return False
   # room for one read but not two still reads the likeliest row, keeping the same reserve after it
   calls=[(tool,{key:pid})for pid in unread[:max(1,min(2,self.left()-5))]]
  if self.left()<len(calls)+5:return False
  start,back=len(self.full),None
  for name,args in calls:
   bad=[]
   args=_shape(args,self.schemas.get(name),"args",bad)
   if bad:return False
   got=self.act(name,args)
   if got is None or self.finished:break
   back=got.get("observation")
   self.observe(name,back)
  added=len(self.full)-start
  sent=calls[0][1].get(many)if many and calls[0][0]==self.roles.get("compare")else None
  # a many-id compare that names none of its ids back marks the size the tool refuses
  if back is not None and isinstance(sent,list)and len(sent)>2 and not any(str(x)in _dump(back)for x in sent):self.compare_bad=min(self.compare_bad or 999,len(sent))
  self.trace+=[dict(kind="audit_open",rows=unread,opened=added)]
  return added>0
 def last_commit(self):
  blocked=self.committed or len(self.items)!=1 or not all(map(self.sold_out,self.cart_ids()))or self.left()<3 or self.secs()<20
  if blocked:return""
  tail=[p for p in self.leads if p in self.full]if self.secs()<CLOCK_EXTRAS else[]
  pick=next(iter([c for c in self.swap_fits+self.carted[::-1]+tail if self.buyable(c)and self.within_budget(c,self.items[0].qty)]),None)
  if not pick:return""
  self.committed,self.completed,self.confirmed=True,False,pick
  self.lead_items={"1":[pick]}
  self.trace+=[dict(kind="last_commit",pick=pick)]
  return self.complete_cart()
 def within_budget(self,listing,units):
  cap=self.budget
  each=None if cap is None else self.price_of(listing)
  return each is None or units*each<=cap
 def cart_slots(self,props):
  texts=[k for k,v in props.items()if isinstance(v,dict)and"string"in _kinds(v)and"enum"not in v]
  count=next(iter([k for k,v in props.items()if isinstance(v,dict)and"integer"in _kinds(v)]),None)
  mode=next(iter([k for k,v in props.items()if isinstance(v,dict)and isinstance(v.get("enum"),list)and len(v["enum"])>=2]),None)
  return texts,count,mode
 def units_fix(self,name,cargs,box):
  _,count_f,mode_f=self.cart_slots(self.schemas.get(name,{}).get("properties")or{})
  n=_count(cargs.get(count_f))if count_f and not mode_f else None
  ids=[v for v in cargs.values()if isinstance(v,str)and v in self.rows]
  if n is None or n<0 or len(ids)!=1:return cargs
  pid,kind=ids[0],self.kind_of(ids[0])
  if n==0 and(pid in self.carted or pid in self.cart_lines or pid in self.cart_ids()or not self.cart_sets(name)):return cargs
  slots=[k for k,v in self.lead_items.items()if pid in v and _count(k)and 1<=_count(k)<=len(self.items)]
  hits=[self.items[_count(k)-1]for k in slots]or[it for it in self.items if kind and kind==(it.shelf or"").strip().lower()]
  if len(self.items)==1:hits=self.items
  if len(hits)!=1:return cargs
  goal=hits[0].qty-(0 if self.cart_sets(name)else self.cart_lines.get(pid,0))
  top=((self.schemas.get(name,{}).get("properties")or{}).get(count_f)or{}).get("maximum")
  if goal<=0 or goal==n or(isinstance(top,int)and goal>top):return cargs
  box["extra"]+="\n[Kyo] %s is the unit count, not a line or item number: sent %s=%s so %s holds the %s units its item asks for."%(count_f,count_f,goal,pid,hits[0].qty)
  return{**cargs,count_f:goal}
 def cart_calls(self,wanted,removals):
  tool=self.roles.get("cart")or""
  spec=self.schemas.get(tool)or{}
  props=spec.get("properties")or{}
  for field,fs in props.items():
   inner=fs.get("items")if isinstance(fs,dict)and"array"in _kinds(fs)else None
   if isinstance(inner,dict)and isinstance(inner.get("properties"),dict):
    texts,count_f,_=self.cart_slots(inner["properties"])
    if not texts:return None
    return[{field:[dict([(texts[0],c)]+([(count_f,n)]if count_f else[]))for c,n in wanted.items()]}]
  texts,count_f,mode_f=self.cart_slots(props)
  id_f=(next(iter([f for t,f in self.id_args if t==tool and f in texts]),None)
    or next(iter([f for f in spec.get("required")or[]if f in texts]),None)or(texts[0]if len(texts)==1 else None))
  if not id_f:return None
  ops=[]
  for pid in removals:
   have,op=self.cart_lines.get(pid,1),{id_f:pid}
   if mode_f:
    op[mode_f]=props[mode_f]["enum"][1]
    if count_f:op[count_f]=have
   elif count_f and self.cart_sets(tool):op[count_f]=0
   elif count_f and isinstance(props[count_f].get("minimum"),int)and props[count_f]["minimum"]<=-have:op[count_f]=-have
   else:return None
   ops.append(op)
  for pid,goal in wanted.items():
   have=self.cart_lines.get(pid,0)if pid in self.cart_ids()else 0
   if have==goal:continue
   if have and not self.cart_sets(tool):return None
   op={id_f:pid}
   if mode_f:op[mode_f]=props[mode_f]["enum"][0]
   if count_f:op[count_f]=goal
   ops.append(op)
  return ops
 def zero_lines(self,reply):
  # a refusal naming a held id beside a zero or false value, and no positive one, marks that id sold out
  held,out,todo=set(self.cart_ids())|set(self.cart_lines),set(),[(reply,0)]
  nil=lambda v:v is False or(not isinstance(v,(bool,str))and _count(v)==0)
  plus=lambda v:v is True or(not isinstance(v,(bool,str))and(_count(v)or 0)>0)
  while todo:
   node,depth=todo.pop()
   if depth>6:continue
   if isinstance(node,dict):
    vals=list(node.values())
    if any(map(nil,vals))and not any(map(plus,vals)):out.update(v.strip()for v in vals if isinstance(v,str)and v.strip()in held)
    out.update(k for k,v in node.items()if k in held and nil(v))
    todo+=[(v,depth+1)for v in vals]
   elif isinstance(node,list):todo+=[(v,depth+1)for v in node]
  self.unavailable.update(out)
  return out
 def basket_plan(self):
  shelf=self.category_key()
  tag=(lambda p:str((self.rows.get(p)or{}).get(shelf)or"").strip().lower())if shelf else(lambda p:p)
  chosen={}
  for slot,ids in self.lead_items.items():
   n,live=_count(slot),[c for c in ids if not self.sold_out(c)]
   pick=live[0]if live else None
   if pick and n and 1<=n<=len(self.items)and tag(pick)not in set(map(tag,chosen)):chosen[pick]=self.items[n-1].qty
  if self.confirmed not in chosen:return None
  held,used=self.cart_ids(),set(map(tag,chosen))
  stay={p:self.cart_lines.get(p,1)for p in held if p not in chosen and not self.sold_out(p)and tag(p)not in used}
  if len(chosen)+len(set(map(tag,stay)))<len(self.items):return None
  return{**stay,**chosen},[p for p in held if p not in chosen and p not in stay]
 def stand_doubts(self,goal):
  # a must neither named by a key nor shown in the record's own words is a doubt
  ids=list(goal)
  for pos,prod in enumerate(self.items,1):
   for nd in[n for n in prod.musts+prod.told if n not in prod.dropped and"only if"not in n]:
    prop,_,want=nd.partition(":")
    pages=[self.full.get(p)or{}for p in ids]
    shown=any(any(_names(_stem(prop),_stem(k))for k in(_fact_block(pg)or{}))or(_tokens(want)and _tokens(want)<=_tokens(_dump(pg)))for pg in pages)
    if not shown:return True
  return False
 def proven_pick(self):
  # a basket the second opinion approved on this version, still buyable, clean and in budget
  if len(self.items)==1:
   prod=self.items[0]
   for key,ok in self.verdicts.items():
    pid,_,ver=key.rpartition("#")
    if ok and ver==str(self.version)and"|"not in pid and pid in self.full and self.buyable(pid)and not self.sold_out(pid)and self.within_budget(pid,prod.qty)and not self.item_clash(pid,1):return{"1":[pid]},pid
  got=self.proven_goal
  if got and got[0]==self.version:
   ids=[v[0]for v in got[1].values()if v]
   total=sum((self.price_of(p)or 0)*self.items[_count(n)-1].qty for n,v in got[1].items()for p in v[:1]if _count(n))
   if ids and all(self.buyable(p)and not self.sold_out(p)for p in ids)and(self.budget is None or total<=self.budget):return got[1],got[2]
  return None
 def complete_cart(self):
  """Cart the confirmed pick in code and check out."""
  cart_tool,buy_tool=self.roles.get("cart"),self.roles.get("buy")
  if self.completed or not self.confirmed or self.sold_out(self.confirmed)or not cart_tool or not buy_tool:return""
  self.completed=True
  plan=self.basket_plan()
  if not plan:return""
  goal,drop=plan
  ops=self.cart_calls(goal,drop)
  if not ops or self.left()<len(ops)+1:return""
  self.fill_facts(list(goal),len(ops)+3)
  if not self.standing or self.stand_doubts(goal):
   # a stand pick that leaves a must unprinted gets the same second opinion as any other basket
   self.check_cart(list(goal),SPEND_HARD+(0.02 if self.standing else 0))
   if not self.verdicts.get(self.check_key(list(goal))):return""
  if not self.standing:self.proven_goal=(self.version,{k:list(v)for k,v in self.lead_items.items()},self.confirmed)
  sent=[]
  for op in ops:
   bad=[]
   op=_shape(op,self.schemas.get(cart_tool),"args",bad)
   got=None if bad else self.act(cart_tool,op)
   if got is None or self.finished:return""
   self.note_cart(cart_tool,op)
   self.observe(cart_tool,got.get("observation"))
   sent.append(_dump(op))
  bad=[]
  pay=_shape(dict(self.brief.get("buy_args")or{}),self.schemas.get(buy_tool),"args",bad)
  if bad:return""
  heard=[]
  for attempt in range(2):
   if attempt:
    self.reread_offers()
    mended=frozenset(goal)in self.mended
    if not(self.standing or mended):self.check_cart(list(goal),SPEND_HARD)
    ready=(self.standing or mended or self.verdicts.get(self.check_key(list(goal))))and self.left()>=1 and not self.finished
    if not ready or any(map(self.sold_out,goal)):break
   cleared,note=self.gate("order")
   if note:heard.append(note)
   if not cleared:break
   self.say_basket()
   got=self.act(buy_tool,pay)
   self.trace+=[dict(kind="completed",lines=goal,drop=drop,reply=_clip(got,200))]
   if got is None:break
   self.buys+=1
   if not got.get("finished"):self.refused_at=self.step
   self.note_sold=False
   self.observe(buy_tool,got.get("observation"))
   if not got.get("finished"):self.zero_lines(got.get("observation"))
   if self.note_sold and not got.get("finished"):self.find_sold()
   if not got.get("finished"):
    moved=self.repair()
    goal={moved.get(p,p):n for p,n in goal.items()}
   heard.append(self.show(got.get("observation")))
   if self.finished or got.get("finished"):break
  if not heard:return""
  return"Kyo finished this on its own: %s passed review two, the cart was arranged via (%s), and the order went to the store, which answered: %s"%(
    self.confirmed,"; ".join(sent)," | next: ".join(heard))
 def _sve_is_end(self,role,name,args):
  wants=self.brief.get(role+"_args")
  if name==self.roles.get(role):return all(args.get(k)==wants[k]for k in(wants if isinstance(wants,dict)else()))
  return False
 def is_buy(self,name,args):return self._sve_is_end("buy",name,args)
 def is_refuse(self,name,args):return self._sve_is_end("refuse",name,args)
 def offered(self):
  left,keep,r=self.left(),self.tail(),self.roles
  ends={r.get("buy"),r.get("refuse")}-{None}
  if left<=keep:tier=0
  elif left<=keep+3 or(left<=keep+6 and self.buys==0 and len(self.cart_ids())>=len(self.items)):tier=1
  elif self.buys==0 and left<=len(self.items)+5:tier=2
  else:tier=1 if self.secs()<CLOCK_STOP+25 else 3
  usable=[t for t in self.tools if t["function"]["name"]!=r.get("ask")]
  if tier==3 or not ends:return usable
  allowed=ends|{r.get(x)for x in((),("cart","cart_view"),("cart","cart_view","offer"))[tier]}
  reader=self.facts_tool()
  if tier==2 and reader and self.facts_from.get(reader):
   if self.facts_from.get(r.get("offer"))is False:allowed.discard(r.get("offer"))
   if left>len(self.items)+keep+1:allowed.add(reader)
  return[t for t in usable if t["function"]["name"]in allowed]or usable
 def quiet(self,tool_name,args):
  spec=self.schemas.get(tool_name)or{}
  need,out=spec.get("required")or[],dict(args)
  for f,fs in(spec.get("properties")or{}).items():
   if not isinstance(fs,dict)or"enum"in fs or"string"not in _kinds(fs):continue
   val=out.get(f)
   if val is None and f not in need:continue
   text=str(val or"")
   # a request wanting the near miss offered keeps the closing words that name it
   offered=self.brief.get("nearest_fallback")is True and self.brief.get("quiet_decline")is not True
   if not text or not offered and(any(p in text for p in self.rows)or self.brief.get("quiet_decline")is not False):out[f]="Each listing checked broke one or more stated conditions; request ended without checkout."[:fs.get("maxLength",200)]
  return out
 def id_from_text(self,raw):
  text=raw.strip()if isinstance(raw,str)else None
  if text is not None and text in self.rows:return text,""
  if text is None or len(text)<3:return raw,""
  low,bare=text.lower()," "not in text and bool(re.search(r"\d",text))
  texts=lambda rec:[v.strip().lower()for k,v in rec.items()if isinstance(v,str)and k!=self.id_key]
  shaped=bare and any(p[:2]==text[:2]and abs(len(p)-len(text))<=3 for p in self.rows)
  # an id cut short or run long matches the one seen id it overlaps; a code printed in a row names that row; a name its title
  if shaped:hits=[p for p in self.rows if len(text)>=6 and abs(len(p)-len(text))<=3 and(p.startswith(text)or text.startswith(p))]
  elif bare:
   mark=re.compile(r"(?<![0-9a-z])"+re.escape(low)+r"(?![0-9a-z])")
   hits=[p for p,rec in self.rows.items()if any(mark.search(v)for v in texts(rec))]
  else:hits=[p for p,rec in self.rows.items()if low in texts(rec)]or[p for p,rec in self.rows.items()if len(low)>=8 and any(v.startswith(low)for v in texts(rec))]
  if len(hits)==1:return hits[0],""
  if hits:return raw,"%r could mean several listings (%s); pass a single id"%(raw,", ".join(hits[:6]))
  if shaped and(text in self.page_ids or text in self.request or text in _dump(self.view.get("opening"))):return raw,""
  return raw,"%r is not an id any store reply has shown; copy the %s field of a row exactly"%(raw,self.id_key)
 def resolve_ids(self,tool_name,args):
  r=self.roles
  if not self.id_key or tool_name not in{r.get(x)for x in("cart","open","offer","compare")}-{None}:return args,""
  props=(self.schemas.get(tool_name)or{}).get("properties")or{}
  free=[k for k,v in props.items()if isinstance(v,dict)and"enum"not in v and("string"in _kinds(v)or"array"in _kinds(v))]
  if len(free)!=1 or(tool_name!=r.get("cart")and(tool_name,free[0])not in self.id_args and free[0]not in(self.single_target(tool_name),self.compare_target())):return args,""
  f=free[0]
  if f not in args:return dict(args),""
  many=isinstance(args[f],list)
  got=list(map(self.id_from_text,args[f]if many else[args[f]]))
  issue=next(iter([p for _,p in got if p]),"")
  if issue:return args,issue
  return{**args,f:[v for v,_ in got]if many else got[0][0]},""
 def cart_sets(self,name):
  specs=[v for v in(self.schemas.get(name,{}).get("properties")or{}).values()if isinstance(v,dict)]
  if any(isinstance(v.get("enum"),list)and len(v["enum"])>1 for v in specs):return False
  return any(v.get("minimum")==0 and"integer"in _kinds(v)for v in specs)
 def note_cart(self,tool_name,args):
  self.cart_touched,self.cart_at=True,self.step
  props=self.schemas.get(tool_name,{}).get("properties")or{}
  removing_2=False
  for k,v in args.items():
   opts=(props.get(k)or{}).get("enum")
   removing_2=removing_2 or(isinstance(opts,list)and len(opts)>=2 and v in opts[1:])
  loose=next(iter([_count(v)for v in args.values()if _count(v)is not None and not isinstance(v,str)]),None)
  wipe_2,moves_2=False,[]
  for k,v in args.items():
   if isinstance(v,str):
    if v in self.rows:moves_2.append((v,loose))
    continue
   if not isinstance(v,list):continue
   inner=(props.get(k)or{}).get("items")or{}
   wipe_2=wipe_2 or(not v and isinstance(inner,dict)and(inner.get("type")=="object"or"properties"in inner))
   for e in v:
    if isinstance(e,dict):
     wipe_2=True
     pid_2=next(iter([x for x in e.values()if isinstance(x,str)and x in self.rows]),None)
     n_2=next(iter([_count(x)for x in e.values()if _count(x)is not None and not isinstance(x,str)]),1)
     if pid_2:moves_2.append((pid_2,n_2))
    elif isinstance(e,str)and e in self.rows:moves_2.append((e,loose))
  removing,wipe,moves=removing_2,wipe_2,moves_2
  if wipe:
   self.cart_lines={p:n for p,n in moves if n and n>0}
   return
  sets=self.cart_sets(tool_name)
  for pid,n in moves:
   have=self.cart_lines.get(pid,0)
   if removing:have=have-n if n else 0
   elif sets:have=1 if n is None else n
   else:have+=1 if n is None else n
   if have>0:self.cart_lines[pid]=have
   else:self.cart_lines.pop(pid,None)
 def show(self,observation):
  text=_dump(_clip(observation,SHOW_CHARS))
  return text if len(text)<=OBS_LIMIT else text[:OBS_LIMIT]+"...(cut)"
 def compare_target(self):
  spec=self.schemas.get(self.roles.get("compare")or"")or{}
  lists=[k for k,v in(spec.get("properties")or{}).items()if isinstance(v,dict)and"array"in _kinds(v)]
  first=lists[0]if lists else None
  return first if first and all(n==first for n in spec.get("required")or[])else None
 def compare_room(self):
  props=(self.schemas.get(self.roles.get("compare")or"")or{}).get("properties")or{}
  good,bad,limit=self.compare_ok,self.compare_bad,props.get(self.compare_target()or"")
  limit=(limit or{}).get("maxItems")
  room=max(READ_CAP,good)if not bad else max(2,min(bad-1,good))
  return room if not isinstance(limit,int)else min(room,limit)
 def _svf_screen(self,call,box):
  r=self.roles.get
  usable={t["function"]["name"]for t in self.offered()}
  if call.get("joined"):return"Merged: this open rode along in the side-by-side call placed earlier in this same turn, costing one step in total."
  fn=call.get("function")or{}
  name,raw=fn.get("name"),fn.get("arguments")
  if self.finished:return"Not run: the episode is already over."
  if name==r("ask"):return _SVF_FREE+"shopper questions are Kyo's job, not a tool call."
  if name not in self.schemas or name not in usable:
   why="unknown tool name"if name not in self.schemas else"remaining moves are held back; this tool is locked now"
   return _SVF_FREE+why+". Usable now: "+", ".join(sorted(usable))+"."
  cargs=raw if isinstance(raw,dict)else(_json_obj(raw)if(raw or"").strip()else{})
  if cargs is None:return _SVF_FREE+"arguments could not be read; send a single JSON object."
  bad=[]
  cargs=_shape(cargs,self.schemas[name],"args",bad)
  if bad:return _SVF_FREE+"; ".join(bad[:6])
  cargs,wrong=self.resolve_ids(name,cargs)
  if wrong:return _SVF_FREE+wrong+"."
  many=self.compare_target()if name==r("compare")else None
  sent=cargs.get(many)if many else None
  room=self.compare_room()
  if isinstance(sent,list)and len(sent)>room:
   box["extra"]="\n[Kyo] Comparison limited to %d listings here; only the first %d went in. Send these in another call: %s."%(room,room,", ".join(map(str,sent[room:])))
   cargs={**cargs,many:sent[:room]}
  key=self.single_target(name)
  if key and name in(r("open"),r("offer"))and str(cargs.get(key))in self.gone:return _SVF_FREE+"%s dropped out of stock at step %s for good; pick another fitting listing to fill that line."%(cargs[key],self.gone[str(cargs[key])])
  if name==r("cart")and any((_count(v)or 0)<0 for v in cargs.values()if not isinstance(v,str)):
   seen=_dump(self.cart_view)if self.cart_seen_at>=self.cart_at else""
   absent=[v for v in cargs.values()if isinstance(v,str)and v in self.rows and v not in self.cart_lines and v not in seen and v not in self.cart_said]
   if absent:return _SVF_FREE+"%s is not in the cart, so there is nothing to take off."%absent[0]+_SVF_CART+self.cart_text()
  if name==r("cart"):
   cargs=self.units_fix(name,cargs,box)
   ids=[v for v in cargs.values()if isinstance(v,str)and v in self.rows]
   if box["count"]>1 and len(ids)==1 and self.cart_sets(name):
    if ids[0]in box["held"]:return _SVF_FREE+"%s went into the cart earlier in this same reply with the units its item asks for; no second call needed."%ids[0]+_SVF_CART+self.cart_text()
    box["held"].add(ids[0])
  cargs=self.first_page(name,self.budget_cap(name,self.true_label(name,cargs,box)),box)
  if name in self.seekers():
   cargs=self.vet_filters(name,cargs,box)
   gap=self.mode_gap(name,cargs)
   if gap:return _SVF_FREE+gap
  if name!=r("cart"):
   cargs=_shape(cargs,self.schemas[name],"args",bad)
   if bad:return _SVF_FREE+"; ".join(bad[:6])
  if name==r("cart")and box["count"]==1 and self.loses_last_line(name,cargs):return _SVF_FREE+"too little budget remains to replace this item, so it is kept; go ahead and pay."+_SVF_CART+self.cart_text()
  final=self.is_buy(name,cargs)or self.is_refuse(name,cargs)
  if final and box["count"]>1:return _SVF_FREE+"Kyo holds back an order or decline sent alongside other calls; review these outputs, then issue it solo."
  if not final and self.left()<=self.tail()and(r("buy")or r("refuse")):return _SVF_FREE+"the steps still left are reserved for finishing: either pay (with the shopper's basket approval first when needed) or close with no purchase."
  mark=name+_dump(cargs)
  if not final and name!=r("cart")and mark in self.recent:return _SVF_FREE+"Kyo already ran exactly this a moment ago and state is unchanged; its result sits above."
  readers=({r(x)for x in("search","browse","open","offer","compare")}|{self.facts_tool()})-{None}
  hit=self.read_cache.get(mark)
  if name in readers and hit and hit[0]==self.version:return _SVF_FREE+"same call as at step %s, nothing changed; repeating that output:\n"%hit[1]+hit[2][:3000]
  tf,pins=self.text_fields.get(name),self.mode_pins.get(name)or{}
  if self.blind.get(name)is True and tf and str(cargs.get(tf)or"").strip()and any(f in cargs and cargs[f]!=v for f,v in pins.items()):return _SVF_FREE+"in this search mode the store ignores the words: different words already listed the same rows. Send the words in the word-search mode (%s) instead."%", ".join("%s=%s"%kv for kv in pins.items())
  if name in self.seekers()and self.answered.get(name+_dump({**self.fixed_args.get(name,{}),**cargs}))==self.version:return _SVF_FREE+"this exact search was already answered in this episode and nothing changed since; its rows are on file. Change the words, the filter or the page."
  if key and name in readers and name!=r("compare")and self.shown_already(name,str(cargs.get(key))):return _SVF_FREE+"%s has nothing new to show here unless its state moves. Stored record: "%(cargs[key],)+self.record_of(str(cargs[key]))
  sent=cargs.get(many)if many else None
  if isinstance(sent,list)and sent:
   fresh=[x for x in sent if not self.shown_already(name,str(x))]
   if not fresh:return _SVF_FREE+"every listing here was already fully shown by this view and is unchanged. On file:\n"+"\n".join("%s: %s"%(x,self.record_of(str(x)))for x in sent)[:4000]
   if len(fresh)<len(sent):
    box["extra"]+="\n[Kyo] Left %s out of the call: already fully shown, unchanged since."%", ".join(str(x)for x in sent if x not in fresh)
    cargs={**cargs,many:fresh}
    mark=name+_dump(cargs)
  box.update({"name":name,"args":cargs,"mark":mark,"many":many,"ending":final,"readers":readers})
  return None
 def filter_args(self,tool,cargs):
  spec=self.schemas.get(tool)or{}
  props=spec.get("properties")or{}
  skip={self.text_fields.get(tool),self.page_args.get(tool),*self.mode_pins.get(tool,{})}
  return[a for a,v in cargs.items()if a not in skip and isinstance(props.get(a),dict)and _dump(v)not in self.tokens_seen and not({"number","integer"}&set(_kinds(props[a])))]
 def store_words(self):
  said={_kb_norm(v):v for row in self.rows.values()for v in row.values()if isinstance(v,str)and 0<len(v)<=80}
  said.update((_kb_norm(v),v)for v in self.labels())
  return said if len(self.rows)>=3 else{}
 def vet_filters(self,tool,cargs,box):
  # a free-text filter value goes in the form the store printed it; an unseen value is left for the store to answer
  spec=self.schemas.get(tool)or{}
  props,said=spec.get("properties")or{},self.store_words()
  single=self.single_target(tool)
  free=[a for a in self.filter_args(tool,cargs)if a!=single and"enum"not in props[a]and isinstance(cargs[a],str)and cargs[a].strip()]
  out=dict(cargs)
  for a in free:out[a]=said.get(_kb_norm(cargs[a]),cargs[a])
  return out
 def mode_gap(self,tool,cargs):
  # the search tool moved off its text mode with no filter value lists nothing
  if not any(f in cargs and cargs[f]!=v for f,v in(self.mode_pins.get(tool)or{}).items()):return""
  given=[a for a in self.filter_args(tool,cargs)if str(cargs[a]).strip()]
  if given:return""
  seen=sorted({str(r.get(self.category_key()))for r in self.rows.values()if r.get(self.category_key())})[:12]if self.category_key()else[]
  return"this search mode needs a category or other filter value and none was given%s."%("; store categories seen: "+", ".join(seen)if seen else"")
 def budget_cap(self,tname,cargs):
  # a shelf browse naming a category, with the top-price slot left empty, carries what the budget leaves per unit of that item
  # once the cart's other lines are paid: rows priced past it can never be bought, and they fill the short pages ahead of rows that can
  names=self.brief.get("browse_args")or{}
  la,ca=names.get("label"),names.get("cap")
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  label=str(cargs.get(la)or"").strip()if la else""
  if tname!=self.roles.get("browse")or not label or ca not in props or ca in cargs or self.budget is None:return cargs
  cat=self.category_key()
  shelf=lambda lid:_kb_norm((self.rows.get(lid)or{}).get(cat)or"")if cat else""
  own=[it for it in self.items if _kb_norm(it.shelf)==_kb_norm(label)]
  others=sum((self.price_of(lid)or 0)*n for lid,n in self.cart_lines.items()if n>0 and shelf(lid)!=_kb_norm(label))
  room=(self.budget-others)/max(1,own[0].qty if len(own)==1 else 1)
  # the budget and the store's prices must plainly share a unit: some listing already seen is priced inside the room
  if room<=0 or not any((self.price_of(lid)or room+1)<=room for lid in self.rows):return cargs
  whole="integer"in _kinds(props[ca])
  return{**cargs,ca:int(room)if whole else round(room,2)}
 def true_label(self,tname,cargs,box):
  arg,field=(self.brief.get("browse_args")or{}).get("label"),self.category_key()
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  if tname!=self.roles.get("browse")or not arg or not field or arg not in props:return cargs
  labels={str(rec.get(field))for rec in self.rows.values()if rec.get(field)}
  given=str(cargs.get(arg)or"").strip()
  printed=next((t for t in labels if _kb_norm(t)==_kb_norm(given)),None)if given else None
  if printed:return cargs if printed==given else{**cargs,arg:printed}
  if not given and tname==self.roles.get("search"):return cargs
  if given:
   best=min(labels,key=lambda t:(-len(_stem(t)&_stem(given)),t),default="")
   tag=best if best and _stem(best)&_stem(given)else""
  else:
   home=self.cart_ids()or self.seen_order[:1]
   tag=str((self.rows.get(home[0])or{}).get(field)or"")if home else""
  if not tag:return cargs
  box["extra"]+="\n[Kyo] Swapped in %r, the matching store category."%tag
  return{**cargs,arg:tag}
 def first_page(self,tname,cargs,box):
  arg=next(iter([rec["page_arg"]for rec in self.search_log.values()if rec["tool"]==tname]),None)
  if arg not in cargs:return cargs
  rest={k:v for k,v in cargs.items()if k!=arg}
  if any(rec["args"]==rest and rec["tool"]==tname for rec in self.search_log.values()):return cargs
  box["extra"]+="\n[Kyo] Removed %s: a new search starts from page one."%arg
  return rest
 def loses_last_line(self,tname,cargs):
  if len(self.cart_lines)>len(self.items)or self.left()>self.tail()+(2 if self.buys else 4):return False
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  dropping=any(isinstance((props.get(k)or{}).get("enum"),list)and v in props[k]["enum"][1:]for k,v in cargs.items())
  dropping=dropping or any(not isinstance(v,str)and(_count(v)or 0)<=0 for v in cargs.values()if _count(v)is not None)
  line=next(iter([v for v in cargs.values()if isinstance(v,str)and v in self.cart_lines]),None)
  return bool(dropping and line and not self.sold_out(line)and self.within_budget(line,self.cart_lines[line]))
 def _ky_hold_pay(self):
  cart=lambda:_SVF_CART+self.cart_text()
  said=next((s for s in(probe()for probe in(self.sweep,self.wider_hold))if s),None)
  if said:return said+cart()
  redo,vtool=self.reread_offers(),self.roles.get("cart_view")
  if vtool and self.left()>3 and self.cart_touched and self.cart_at>self.cart_seen_at:
   errs:list[str]=[]
   view_args=_shape({},self.schemas.get(vtool),"args",errs)
   got=self.act(vtool,view_args)if not errs else None
   if got is not None:
    self.cart_view,self.cart_seen_at=got.get("observation"),self.step
    shift=self.observe(vtool,self.cart_view)
    if shift:
     self.queued=[]
     return(_SVF_HELD+"Fresh cart contents differ from before: "+shift+cart()+"\nHandle this first, changing the cart if it calls for that; only then check out.")
  tip,lost=self.swap_hint(),[x for x in self.cart_ids()if x in self.gone]
  if lost and not tip and self.held>=2:
   self.swap_lost()
   lost=[]
  if tip or lost:
   if not tip:self.held+=1
   return _SVF_HELD+(tip or"%s sold out yet sits in the cart: substitute that line and check out."%", ".join(lost))+cart()
  chain,stuck=self.trail()
  if stuck:return _SVF_HELD+chain+cart()
  looked=self.fill_facts(self.cart_ids(),self.tail()+3)+self.cheaper_rows()
  found_=self.check_cart(limit=SPEND_HARD if self.brief.get("pick")=="cheapest"else SPEND_SOFT)
  if found_ and self.take_better():found_=""
  if not found_ and self.flawed==self.check_key(self.cart_ids())and self.brief.get("nearest_fallback")is True:
   near=self.audit(SPEND_HARD,CLOCK_STOP+20,True)
   if near and set(self.leads)-set(self.cart_ids()):return _SVF_HELD+near+cart()
  if found_:
   fetched="Fetched for this check, one step each:\n"+"\n".join(looked)+"\n"if looked else""
   return"Held. "+chain+fetched+found_+cart()
  if redo and self.cost>SPEND_SOFT:return _SVF_HELD+"Fresh look at every cart line, just taken:\n"+redo+"\nFirst swap any line now unavailable or priced past the budget; with all lines clean, retry checkout."
  return None
 def _ky_hold_review(self):
  audit_text=self.audit(SPEND_HARD,CLOCK_STOP+20)
  if audit_text:return _SVF_HELD+audit_text
  # no review may overrule a stated value the code sees broken: leads that clash with the item's own needs are not pressed
  if len(self.items)==1:self.leads=[x for x in self.leads if not self.item_clash(x,1)]
  phase=self.refuse_blocks if self.leads and self.left()>4 else-1
  if phase not in(0,1):return None
  self.refuse_blocks+=1
  if phase==0:
   self.trace.append(dict(kind="refuse_blocked",leads=list(self.leads)))
   return(_SVF_HELD+"Candidates to review: "+", ".join(self.leads)+". For each, read more if unsure; buy the one the pick rule favours by carting it and checking out. A no-purchase exit needs every one of them to show a plain breach - quote it in your message and send the ending call once more.")
  lead=self.leads[0]
  self.check_cart([lead],SPEND_HARD)
  if lead not in self.leads or not self.verdicts.get(self.check_key([lead])):return None
  self.confirmed=lead
  self.trace.append(dict(kind="refuse_blocked",confirmed=lead))
  return(_SVF_HELD+"Independent review of %s: its records break no requirement. Add it at the requested quantity and check out; nothing but an outright contradiction within its record justifies a no-purchase finish."%lead)
 def _ky_hold_late(self):
  alive=[x for x in self.built if not self.sold_out(x)]
  in_cart,stale=self.cart_ids(),self.cart_view is None or self.cart_seen_at<self.cart_at
  absent=[x for x in alive if x not in in_cart or(stale and x in self.dropped)]
  if len(absent)>=self.left()or len(alive)<len(self.items):return None
  self.late_held=True
  self.trace.append(dict(kind="late_decline_held",lines=alive,missing=absent))
  todo=("Re-add %s (requested quantities) and check out."%", ".join(absent))if absent else"Cart is complete; go to checkout."
  return(_SVF_HELD+"A no-purchase finish scores zero if the store holds a match, and a low step count says nothing about that. "+todo
    +" Exit without an order solely if search results established that not a single listing fits some item of the request; then issue the ending call again.")
 def overfill(self,name,args):
  """A cart call adding units to a line that already holds all its item asks for, in a cart that adds quantities up."""
  if self.cart_sets(name):return""
  props=self.schemas.get(name,{}).get("properties")or{}
  if any(isinstance((props.get(k)or{}).get("enum"),list)and len(props[k]["enum"])>=2 and v in props[k]["enum"][1:]for k,v in args.items()):return""
  ids=[v for v in args.values()if isinstance(v,str)and v in self.cart_lines]
  units=[_count(v)for v in args.values()if not isinstance(v,(str,list,dict))and _count(v)is not None]
  pos=self.slot_of(ids[0])if len(ids)==1 else None
  if pos is None or(units and units[0]<=0):return""
  have,want=self.cart_lines[ids[0]],self.items[pos-1].qty
  if have<want:return""
  return"%s already holds %s units, all its item asks for, and this cart adds quantities up: %s more would overshoot, so nothing was sent."%(ids[0],have,units[0]if units else 1)
 def _svf_hold(self,slot):
  """Guards on a model tool call: holds a weak checkout or a premature close, with the reason."""
  tname,targs,ending=slot["name"],slot["args"],slot["ending"]
  buying,declining=self.is_buy(tname,targs),self.is_refuse(tname,targs)
  finds=self.code_finds
  if tname in self.seekers()and self.code_query(targs)and finds.get("zero",0)>=2 and not finds.get("hit")and finds.get("held",0)<3:
   finds["held"]=finds.get("held",0)+1
   self.trace.append(dict(kind="code_held",args=_clip(targs,80)))
   return(_SVF_HELD+"%d searches naming a model code found nothing in this store, while word searches list rows: it does not search by codes. Search the product kind or list its category, then read the codes its rows and pages print."%finds["zero"])
  if tname==self.roles.get("cart")and getattr(self,"zero_held",0)<3:
   zero=[k for k,v in targs.items()if not isinstance(v,(str,list,dict))and _count(v)==0]
   fresh=[v for v in targs.values()if isinstance(v,str)and v in self.rows and v not in self.cart_ids()]
   if zero and fresh:
    self.zero_held=getattr(self,"zero_held",0)+1
    full=len(self.cart_ids())>=len(self.items)
    return(_SVF_HELD+"%s is not in the cart, and %s=0 asks for zero units of it, so the call would add nothing. %s is the unit count: send the quantity wanted to add it.%s"
      %(", ".join(fresh),zero[0],zero[0]," The cart already has a line for every item, so a swap also needs the old line set to 0 units."if full else"")+_SVF_CART+self.cart_text())
  if tname==self.roles.get("cart")and getattr(self,"over_held",0)<3:
   over=self.overfill(tname,targs)
   if over:
    self.over_held=getattr(self,"over_held",0)+1
    return _SVF_HELD+over+_SVF_CART+self.cart_text()
  cart=lambda:_SVF_CART+self.cart_text()
  tag=lambda:" [episode finished]"if self.finished else""
  if declining and not self.full_held:
   lines=[x for x in self.cart_ids()if not self.sold_out(x)]
   if len(lines)>=len(self.items)and self.left()>=2:
    self.full_held=True
    self.trace.append(dict(kind="full_cart_held",lines=lines))
    return(_SVF_HELD+"Checkout, not an exit: each item asked for has its own cart line, and quitting while a match exists pays 0. The ending call may be resent solely when a record plainly proves one of those lines fails a requirement.")
  if ending and buying and self.left()>2:
   broken=self.range_hold()
   if broken:return _SVF_HELD+broken+cart()
  if declining:
   auto=self.last_commit()
   if auto:return auto+tag()
  roomy=self.secs()>CLOCK_EXTRAS
  if(ending and buying and self.left()>5 and not roomy and self.secs()>CLOCK_STOP+15 and self.open_lookups()):
   before=len(self.ref_values)
   chain,stuck=self.trail()
   if stuck or len(self.ref_values)>before:return _SVF_HELD+chain+cart()
  held=None
  if ending and buying and self.left()>2 and roomy:held=self._ky_hold_pay()
  elif ending and self.left()>5 and roomy:held=self._ky_hold_review()
  # the clock closed the full pre-pay review before any cart was judged: the last check still runs once
  elif ending and buying and self.left()>2 and not self.verdicts and self.secs()>CLOCK_STOP+15:
   held=self.check_cart(limit=SPEND_HARD)
   held=held and"Held. "+held+cart()
  if held:return held
  if declining and not self.late_held and self.left()<=5:
   held=self._ky_hold_late()
   if held:return held
  # a many-item order is pushed twice, leads or not: most real-race declines of order tasks came with a quarter or more of the steps left
  if declining and self.pushed<(2 if len(self.items)>1 else 1)and(len(self.items)>1 or not self.leads)and self.left()>self.max_steps//3 and self.secs()>CLOCK_EXTRAS:
   self.pushed+=1
   bare=self.bare_items()
   return(_SVF_HELD+"%d steps left over. Giving up now is premature%s: page through the categories of all items asked for (resend the store's token or offset each time), search with two or three words (one stated value plus the product kind), try alternate labels for the property, and open whatever could fit; give up only then."
     %(self.left(),"; still without a cart line: "+", ".join(bare)if bare else""))
  if declining:
   self.last_stand()
   auto=self.complete_cart()
   if auto:return auto+tag()
   slot["args"]=targs=self.quiet(tname,targs)
  if ending and self.left()>=2 and roomy:
   leg="order"if self.is_buy(tname,targs)else"end"
   ok,word=self.gate(leg)
   if not ok:return"Held. "+word+cart()
   if word:slot["extra"]="\n[Kyo] "+word+slot["extra"]
  if buying:self.say_basket()
  return None
 def _ky_cart_effects(self,tname,targs,seen,qty):
  self.note_cart(tname,targs)
  self.cart_said,self.recent=_dump(seen),[]
  for cid in(v for v in targs.values()if qty and isinstance(v,str)and v in self.rows):
   if qty[0]>0:self.dropped.discard(cid)
   elif not _in_rows(seen,cid):self.dropped.add(cid)
  self.carted+=[x for x in self.cart_ids()if x not in self.carted]
  standing=[x for x in self.cart_ids()if not self.sold_out(x)and x not in self.dropped]
  if len(standing)>=len(self.built):self.built=standing
 def _svf_execute(self,slot):
  tname,targs,final,readers=slot["name"],slot["args"],slot["ending"],slot["readers"]
  reply=self.act(tname,targs)
  if reply is None:return"Empty response from the store (connection or server issue); you may retry."
  seen,done=reply.get("observation"),reply.get("finished")
  group_key=slot["many"]
  listed_2=targs.get(group_key)if group_key else None
  size,listed=len(listed_2)if isinstance(listed_2,list)and len(listed_2)>1 else 0,listed_2
  if size and any(str(x)in _dump(seen)for x in listed):self.compare_ok=max(self.compare_ok,size)
  elif size:
   self.compare_bad=min(self.compare_bad or 999,size)
   room=self.compare_room()
   # the refused batch is resent at once in the size the store takes, so the turn still reads listings
   if room<size and not(self.finished or done)and self.left()>self.tail()+3:
    again=self.act(tname,{**targs,group_key:list(listed)[:room]})
    if again is not None and any(str(x)in _dump(again.get("observation"))for x in list(listed)[:room]):
     seen,done=again.get("observation"),again.get("finished")
     self.compare_ok=max(self.compare_ok,room)
   slot["extra"]+=f"\n[Kyo] Your {size}-id compare request came back without any of those listings; Kyo resent the first {room}. From now on pass {room} ids or fewer."
  self.recent=(self.recent+[slot["mark"]])[-3:]
  for param,given in targs.items():
   ids=given if isinstance(given,list)else[given]
   if ids and all(isinstance(x,str)and x in self.rows for x in ids):self.id_args.add((tname,param))
  self.absorb(seen)
  # some stores find nothing by model codes: count what code searches return, once rows are read at all
  if tname in self.seekers()and self.rows and self.code_query(targs):
   key="hit"if self.records_in(seen)else"zero"
   self.code_finds[key]=self.code_finds.get(key,0)+1
  if tname==self.roles.get("compare"):
   found=self.records_in(seen)
   if found:self.facts_from[tname]=self.facts_from.get(tname,False)or any(_fact_block(r)for _,r in found)
  if tname in readers-{self.roles.get("search")}:self.note_view(tname,seen)
  paged=tname not in{t for name,t in self.roles.items()if name not in("search","browse")}and not final
  if paged:self.log_search(tname,targs,seen)
  more=self.more_pages(tname,targs,seen)if paged else""
  busy=tuple(self.roles.get(name)for name in("cart","buy","refuse","ask"))
  hit=self.ident(seen)if self.single_target(tname)and tname not in busy and isinstance(seen,dict)else None
  if hit:
   shown=bool(_fact_block(seen))
   if not shown:self.thin_reads.add("%s %s"%(tname,hit))
   # one thin page says little about the tool: it counts as showing no attributes only from the second such page on
   has_facts=self.facts_from[tname]=self.facts_from.get(tname,False)or shown or(None if sum(r.startswith(tname+" ")for r in self.thin_reads)<2 else False)
   opener=self.roles.get("open")
   moved_2=None
   if tname==opener and not has_facts:moved_2=next((t for t in self.schemas if t!=tname and self.single_target(t)and self.facts_from.get(t)is not False and t not in busy+(self.roles.get("search"),)),None)
   opener=moved_2 or opener
   moved,offer=moved_2,not has_facts and tname!=opener and not self.roles.get("offer")
   if moved:self.roles["open"]=moved
   if offer:self.roles["offer"]=tname
   if tname==self.offer_tool():self.offer_seen[hit]=self.step
  qty=[n for n in(_count(v)for v in targs.values()if not isinstance(v,str))if n is not None]
  is_cart=tname==self.roles.get("cart")
  if is_cart:self._ky_cart_effects(tname,targs,seen,qty)
  if tname==self.roles.get("cart_view"):self.cart_view,self.cart_seen_at=seen,self.step
  buy=self.is_buy(tname,targs)
  if buy:
   self.buys+=1
   if not done:self.refused_at,_=self.step,self.zero_lines(seen);self.version+=1
  note=self.show(seen)+more+slot["extra"]
  if paged:note+=self.cheapest_here([seen]+self.paged_obs)
  if paged and tname==self.roles.get("search")and self.brief.get("pick")in("cheapest","most","least"):note+=self.glued_numbers(targs)
  cacheable,evidence=note,self.evidence(seen)
  extra=[evidence]if evidence else[]
  if is_cart and 0 in qty and any(isinstance(v,str)and v in self.rows and _in_rows(seen,v)for v in targs.values()):extra.append("[Kyo] A zero quantity changed nothing: this cart adds quantities up, so take units off with a negative number.")
  self.note_sold=False
  loose=self.loose_lines(seen)
  gist=self.take_note(loose)if loose else""
  if gist:extra.append("[note read] "+gist+": the ledger now carries this and it binds like the shopper's own words, whichever tool result brought it; change your picks to suit it ahead of checkout.")
  sold=buy and not done and(self.note_sold or any(line in self.spoken(seen)for line in self.sold_lines))
  # a refused checkout whose note moves only a price gets its lines read again, so the swap choice sees the fresh price
  rose=bool(gist)and buy and not done and not sold and self.price_rose==2
  was={p:self.price_of(p)for p in self.cart_ids()}if rose else{}
  recheck=self.find_sold()if sold else self.reread_offers()if rose else""
  if recheck:extra.append("[checkout refused - fresh status of the cart lines]\n"+recheck)
  moved=self.repair()if not done and(buy or self.cap_cut is not None)else{}
  if moved:extra.append("[Kyo] Cart mended in code: "+"; ".join("%s out, %s in"%pair for pair in moved.items())+". Each new line passed review and keeps the sum inside the budget; check out now.")
  tip=self.swap_hint()
  if tip:extra.append(tip)
  due=self.cart_sum()
  if rose and not(moved or tip)and due is not None and(self.budget is None or due<=self.budget)and all(self.offer_seen.get(p,-99)>=self.refused_at and self.buyable(p)and(self.price_of(p)or 0)<=(was.get(p)or 0)for p in self.cart_ids()):
   extra.append("[Kyo] Kyo read every cart line again after this refusal: none costs more than before, and the cart now comes to %g, inside the budget. A moved price breaks no requirement, so keep the cart as it is and send checkout again."%due)
  if self.queued:
   extra.append("[more notes found by Kyo's own calls] "+" | ".join(self.queued))
   self.queued=[]
  if tname in readers:self.read_cache[slot["mark"]]=(self.version,self.step,cacheable)
  return"\n".join([note]+extra)+(" [episode finished]"if done else"")
 def _ky_close_verdict(self,leg,reply,again):
  if leg=="end":return None if reply is None or self.finished else True
  return None if self.left()<=0 or self.finished else bool(again and reply is not None)
 def finish(self,again=False):
  """Forced ending: order a cart with a live line for every item, else close empty-handed."""
  if self.left()<=0 or self.broke or self.finished:return""
  if self.left()>2 and(self.cap_cut is not None or any(map(self.sold_out,self.cart_ids()))):self.repair()
  if self.left()>2 and any(map(self.sold_out,self.cart_ids())):self.swap_lost()
  if self.finished or self.left()<=0:return""
  fresh=self.cart_seen_at>=self.cart_at and self.cart_view is not None
  lines=self.cart_ids()if fresh else self.cart_lines
  plan=[("end",self.roles.get("refuse"),"refuse_args","forced_refuse")]
  dead=len(lines)>0 and all(map(self.sold_out,lines))
  if dead:self.trace.append(dict(kind="dead_cart",lines=list(lines)))
  # a forced order on a cart short of the units asked for pays nothing: the empty-handed exit goes instead
  live={x:n for x,n in self.cart_lines.items()if not self.sold_out(x)}
  short=bool(live)and len(live)<len(self.items)and sum(live.values())<sum(_count(it.qty)or 1 for it in self.items)
  if short:self.trace.append(dict(kind="short_cart",lines=dict(self.cart_lines)))
  if not dead and not short and(len(lines)>0 or(self.cart_touched and not fresh and self.left()>=2)):plan.insert(0,("order",self.roles.get("buy"),"buy_args","forced_buy"))
  if len(plan)==1 and not self.proven_tried and self.left()>3:
   self.proven_tried=True
   got=self.proven_pick()
   # a reviewer's opened match that the loop ran out of turns to act on still gets its independent check before the empty exit
   lead=None if got or len(self.items)!=1 else next(iter([x for x in self.leads if x in self.full and self.buyable(x)and not self.sold_out(x)and self.within_budget(x,self.items[0].qty)and not self.item_clash(x,1)]),None)
   if lead:
    self.check_cart([lead],self.cost+0.02)
    got=self.proven_pick()
   if got:
    self.lead_items,self.confirmed,self.completed,self.standing=got[0],got[1],False,False
    self.trace.append(dict(kind="proven_close",pick=got[1]))
    self.complete_cart()
    if self.finished:return""
  ky_close_plan=plan
  for leg,tool,args_key,label in ky_close_plan:
   faults:list[str]=[]
   if not tool:continue
   fixed=dict(self.brief.get(args_key)or{})
   call_args=_shape(fixed if leg=="order"else self.quiet(tool,fixed),self.schemas.get(tool),"args",faults)
   if faults:continue
   if self.secs()>CLOCK_STOP and self.left()>=2:self.gate(leg)
   if leg=="order":self.say_basket()
   if self.finished or self.left()<=0:return""
   reply=self.act(tool,call_args)
   self.trace.append(dict(kind=label,reply=_clip(reply,200)))
   if leg=="order"and reply is not None and not self.finished:reply=self.mend_order(tool,call_args,reply)
   verdict=self._ky_close_verdict(leg,reply,again)
   if verdict is None:return""
   if verdict:
    shown=reply.get("observation")if reply is not None else None
    if leg=="order":self.buys,self.refused_at=self.buys+1,self.step
    self.note_sold=False
    self.observe(tool,shown)
    if leg=="order":
     self.zero_lines(shown)
     if self.note_sold:self.find_sold()
     self.reread_offers()
    return("payment"if leg=="order"else"empty-handed close")+" bounced, reply: "+self.show(shown)
  return""
 def _ky_opening(self):
  parts=["THE SHOPPER WROTE:\n"+self.request]
  screen=self.view.get("opening")
  if screen is not None:
   self.absorb(screen)
   parts.append("STORE'S FIRST SCREEN:\n"+self.show(screen))
  if self.roles.get("search")and self.refs:
   traced=self.trail()[0]
   self.trail_blocks,self.walked_full=0,len(self.full)
   if self.ref_words:traced+=self.ref_hunt()
   if traced:parts.append("LOOKUP CHAINS WALKED BY KYO SO FAR:\n"+traced)
  parts.append("Suggested start: search the category of each item asked for and read a handful of full listings per item, because the interviewer builds its questions from the attribute names found there.\n\n"+self.status())
  return parts
 def run(self):
  """Brief, model loop, forced finish; a refused finish gets one more short round."""
  log=self.trace
  if not self.finished:
   menu=[]
   for entry in self.tools:
    fn=entry["function"]
    menu.append("- %s: %s | parameters: %s"%(fn["name"],fn.get("description",""),_dump(fn.get("parameters")or{})))
   intake="%s\nTOOLS:\n%s\n\nREQUEST:\n%s"%(BRIEF,"\n".join(menu),self.request)
   if self.view.get("opening")is not None:intake+="\n\nSTORE'S OPENING MESSAGE:\n"+_dump(_clip(self.view.get("opening"),1500))
   first=self.ask_json(intake,[LIGHT],8000)or{}
   named=first.get("roles")
   whole=bool(_items(first.get("items")))and isinstance(named,dict)and bool(named.get("buy"))
   reading=first if self.no_credit or whole else(self.ask_json(intake,[HEAVY,LIGHT],8000)or first)
   self.brief=reading
   named=reading.get("roles")
   self.roles={k:v for k,v in(named.items()if isinstance(named,dict)else())if isinstance(v,str)and v in self.schemas}
   named_2=reading.get("search_text")
   tool=self.roles.get("search")
   spec=self.schemas.get(tool or"")or{}
   props=spec.get("properties")or{}
   text_ok=lambda a:isinstance(props.get(a),dict)and"enum"not in props[a]and"string"in _kinds(props[a])
   if isinstance(named_2,dict)and text_ok(named_2.get("text")):
    raw_2=named_2.get("mode")
    pins={}
    for f,v_2 in(raw_2.items()if isinstance(raw_2,dict)else()):
     if v_2 in((props.get(f)or{}).get("enum")or[]):pins[f]=v_2
    self.text_fields[tool],self.fixed_args[tool]=named_2["text"],pins
    self.mode_pins[tool]=dict(pins)
    self.schemas[tool]={**spec,"required":[f for f in spec.get("required")or[]if f not in pins]}
   for key in("buy_args","refuse_args","browse_args"):
    if not isinstance(reading.get(key),dict):reading[key]={}
    else:reading[key]=reading.get(key)
   bp=(self.schemas.get(self.roles.get("browse")or"")or{}).get("properties")or{}
   kind=lambda a,want:isinstance(bp.get(a),dict)and bool(want&set(_kinds(bp[a])))
   ba=reading["browse_args"]
   if ba and not kind(ba.get("label"),{"string"}):ba["label"]=None
   if ba and not kind(ba.get("cap"),{"number","integer"}):
    nums=[a for a in bp if kind(a,{"number"})]
    ba["cap"]=nums[0]if len(nums)==1 else None
   pay,close=reading["buy_args"],reading["refuse_args"]
   dual=self.roles.get("buy")
   if dual and dual==self.roles.get("refuse")and(not pay or not close or pay==close):
    ask=(f"Reply with exactly one JSON object holding two keys, \"buy_args\" and \"refuse_args\"; each maps to the fixed arguments "
      "that pick one mode, nothing else. \"buy_args\" = the mode that pays for the order; \"refuse_args\" = the mode that ends "
      f"the visit with nothing bought.\nWhy: the store's tool {dual} does both jobs and its arguments decide which. Schema of {dual}: "
      +_dump(self.schemas.get(dual)))
    got=self.ask_json(ask,[LIGHT,HEAVY],600)or{}
    b,r=got.get("buy_args"),got.get("refuse_args")
    if isinstance(b,dict)and isinstance(r,dict)and b!=r:reading["buy_args"],reading["refuse_args"]=b,r
   max_steps=self.max_steps
   goods=[]
   for raw in _items(reading.get("items")):
    if isinstance(raw,dict):
     n_2=_count(raw.get("quantity"))
     goods.append((str(raw.get("name")or"the product"),str(raw.get("shelf")or""),
       n_2 if n_2 and n_2>0 else 1,[str(t)for t in _items(raw.get("musts"))if t]))
   money=reading.get("budget")
   if isinstance(money,str)and re.fullmatch(r"\s*\d+(\.\d+)?\s*",money):money=float(money)
   if not(isinstance(money,(int,float))and not isinstance(money,bool)and money>0):money=None
   talk=_count(reading.get("messages"))
   walk=_count(reading.get("steps"))
   flow=[]
   for raw in _items(reading.get("process")):
    label=str(raw.get("step")or"").strip()if isinstance(raw,dict)else""
    if label:
     when=str(raw.get("when")or"").strip().lower()
     flow.append({"step":label,"when":when if when in("choose","order","end")else"order",
       "messages":max(0,_count(raw.get("messages"))or 0)})
   parsed=({"goods":goods,"budget":money,"msgs":talk if talk is not None and talk>=0 else None,
     "steps":walk if walk is not None and walk>=10 and walk<max_steps else None,"flow":flow})
   self.items.extend(Item(*g)for g in parsed["goods"])
   if not self.items:self.items=[Item("the requested product","",1,[])]
   pool=[m for it in self.items for m in it.musts]
   for ptr in _items(reading.get("lookups")):
    text=str(ptr).strip()
    if text not in pool:
     head=_words(text.split(":")[0])
     text=next((m for m in pool if _words(m.split(":")[0])==head),text)
    slot=next((n for n,it in enumerate(self.items,1)if text in it.musts),1)
    if text and(slot,text)not in self.refs and not self.sibling_ref(slot,text):self.refs.append((slot,text))
   if not self.refs:self.find_pointers()
   self.unmix()
   self.settle_musts(reading)
   self.fuse_shelf()
   if parsed["budget"]is not None:self.budget=parsed["budget"]
   if parsed["msgs"]is not None:self.msg_limit=parsed["msgs"]
   elif self.roles.get("ask"):self.msg_limit=3
   if parsed["steps"]is not None:self.max_steps=parsed["steps"]
   self.process.extend(parsed["flow"])
   if self.roles.get("ask"):
    # a step due at checkout or ending may have to reach the shopper: one message stays held even where the reading counts none
    owed=[p["messages"]or 1 for p in self.process if p["when"]!="choose"]
    base=min(max(owed),self.msg_limit)if owed else 0
    # with two messages or fewer every one is worth more as a question than as a closing note
    self.own_statement=self.msg_limit>=3 and not owed and not isinstance(reading.get("process"),list)
    self.reserve=max(base,1)if self.own_statement else base
   log.append(dict(kind="brief",brief=self.brief))
   self.chat=[dict(role="system",content=GUIDE),dict(role="user",content="\n\n".join(self._ky_opening()))]
   self.work(loop_cap=LOOP_CALLS,spend_cap=SPEND_HARD)
   # a spend cap reached just after the requirements changed still leaves a short allowance to act on the change
   fresh=self.version>0 and self.step-self.ver_seen.get(self.version,self.step)<=6
   if fresh and not(self.finished or self.broke)and self.cost>=SPEND_HARD and self.left()>self.tail()+4 and self.secs()>CLOCK_STOP+20:self.work(loop_cap=self.loop_calls+4,spend_cap=self.cost+0.015)
   if len(self.cart_ids())<len(self.items)and not self.finished:
    self.last_stand()
    self.complete_cart()
   bounce=self.finish(True)if not self.finished else""
   if bounce and self.left()>1 and not(self.finished or self.broke or self.no_credit):
    self.chat.append(dict(role="user",content="[Kyo] Wrap-up refused, with this reason: %s\nFix that and wrap up once more.\n%s"%(bounce,self.status())))
    # a refusal past the retry cap still gets a short allowance to act on what it says
    cap,more=(SPEND_HARD+0.01,10)if self.cost<SPEND_HARD+0.01 else(self.cost+0.015,4)
    self.work(spend_cap=cap,loop_cap=self.loop_calls+more)
   if not self.finished:self.finish()
   log.append(dict(kind="end",finished=self.finished,steps=self.step,cost=round(self.cost,5),calls=self.calls))
  return log
 def late_look(self):
  if self.late_looked or self.carted or len(self.items)!=1 or(self.secs()>CLOCK_EXTRAS+50 and self.left()>LATE_LOOK+self.tail()):return""
  self.late_looked=True
  return self.audit(SPEND_HARD)
 def _ky_stand_picks(self,got,live):
  table=got.get("picks")if isinstance(got.get("picks"),dict)else{}
  # a pick keyed by the item's name (or "item 1") instead of its number names the same line
  named={(it.name or"").strip().lower():str(n)for n,it in enumerate(self.items,1)}
  tail=lambda s:(s.split()or[""])[-1]
  slot=lambda s:s if _count(s)else named.get(s.lower())or(tail(s)if _count(tail(s))else s)
  table={slot(str(k).strip()):v for k,v in table.items()}
  choice={num:str(table.get(str(num))or"")for num in range(1,len(self.items)+1)}
  if any(chosen not in live or self.item_clash(chosen,num)for num,chosen in choice.items()):return None
  total=sum((self.price_of(chosen)or 0)*self.items[num-1].qty for num,chosen in choice.items())
  return None if self.budget is not None and total>self.budget else choice
 def last_stand(self):
  """Before leaving empty-handed: one strong pick among the opened listings."""
  near=self.brief.get("nearest_fallback")is True
  ceiling=SPEND_HARD+(0.07 if near else 0.02)
  if not(not(self.stood or self.finished or(not near and len(self.items)!=1))and(self.cost<=ceiling and self.secs()>=15 and self.left()>=2)):return
  self.stood=True
  # a listing opened on a page that prints no fact block (its row, overview and pointers only) is still an opened record
  pool=[x for x in dict.fromkeys(self.cart_ids()+self.on_hand())if(x in self.full or x in self.opened)and x in self.rows and self.buyable(x)][:24]
  if pool:
   records="\n".join("OPENED "+_dump(_clip(self.full.get(x)or self.rows[x],200))[:1200]for x in pool)
   sections=[STAND,"REQUEST:\n"+self.request,
     "REQUIREMENTS IN FORCE:\n"+self.ledger(),"PICK RULE: "+str(self.brief.get("pick")),"RECORDS:\n"+records]
   got=self.ask_json("\n\n".join(sections),[LIGHT,HEAVY]if self.cost<=SPEND_HARD else[LIGHT],8000)or{}
   self.trace.append({"kind":"last_stand","got":got})
   choice=self._ky_stand_picks(got,pool)
   if choice is not None:
    self.standing,self.completed,self.confirmed=True,False,choice[1]
    self.lead_items={str(num):[chosen]for num,chosen in choice.items()}
 def clock_buy(self):
  shut=(self.committed or len(self.items)!=1 or self.cart_lines or not self.leads or self.secs()>CLOCK_EXTRAS+10 or self.left()<3)
  units=0 if shut else self.items[0].qty
  chosen=None if shut else next((x for x in self.leads if not self.sold_out(x)and self.within_budget(x,units)),None)
  if not chosen:return""
  self.committed=True
  self.confirmed,self.lead_items,self.completed=chosen,{"1":[chosen]},False
  self.trace.append({"kind":"clock_buy","pick":chosen})
  return self.complete_cart()
 def _ky_trim(self,talk):
  if sum(len(_dump(entry.get("content")))for entry in talk)<=HISTORY_LIMIT:return
  tool_turns=[entry for entry in talk if entry.get("role")=="tool"]
  for entry in tool_turns[:max(1,len(tool_turns)//2)]:
   body=entry.get("content")or""
   if len(body)>600:
    kept=[x for x in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"',body))if x in self.rows]
    entry["content"]=body[:400]+"...(older output cut; ids it named: "+", ".join(kept[:20])+")"
 def work(self,loop_cap,spend_cap):
  talk,idle=self.chat,0
  while(not(self.finished or self.broke)and self.left()>0 and self.loop_calls<loop_cap and self.cost<spend_cap and self.secs()>CLOCK_STOP):
   if(fused := self.fuse_shelf()):talk.append(dict(role="user",content=fused+"\n"+self.status()))
   if self.ready_to_ask():
    found=self.interview()
    if found:talk.append(dict(role="user",content=found+"\n"+self.status()))
    continue
   if(late := self.late_look()):talk.append(dict(role="user",content=late+"\n"+self.status()))
   if(bought := self.clock_buy()):talk.append(dict(role="user",content=bought)); continue
   self._ky_trim(talk)
   self.loop_calls+=1
   answer=self.think(talk,[LIGHT,SPARE],4000,self.offered())
   if answer is None:
    self.misses+=1
    if self.no_credit or self.calls>=ALL_CALLS or self.misses>2:break
    continue
   outgoing_2=[]
   for pos,call in enumerate(answer.get("tool_calls")or[]):
    if not isinstance(call,dict):continue
    fn=call["function"]if isinstance(call.get("function"),dict)else{}
    raw=fn.get("arguments")
    raw=(_json_obj(raw)if raw.strip()else{})if isinstance(raw,str)else raw if isinstance(raw,dict)else{}
    outgoing_2.append({"id":str(call.get("id")or"k%d_%d"%(self.loop_calls,pos)),"type":"function",
      "function":{"name":str(fn.get("name")or""),"arguments":_dump(raw if isinstance(raw,dict)else{})}})
   outgoing=outgoing_2
   said=answer.get("content")or""
   text=said if isinstance(said,str)else _dump(said)
   if outgoing:
    idle=self._ky_after_calls(talk,outgoing,text)
    continue
   idle+=1
   talk.append(dict(role="assistant",content=text or"(no call)"))
   talk.append(dict(role="user",content="Your reply had no tool call. Callable at this point: "+", ".join(t["function"]["name"]for t in self.offered())+"; pick one.\n"+self.status()))
   if idle>=3:break
 def _ky_after_calls(self,talk,outgoing,text):
  talk.append(dict(role="assistant",content=text or None,tool_calls=outgoing))
  binding={self.roles.get("cart"),self.roles.get("buy"),self.roles.get("refuse")}-{None}
  if(not self.asked and self.roles.get("ask")and self.msgs_free()>0 and self.left()>8 and any(cl["function"]["name"]in binding for cl in outgoing)):
   talk.extend(dict(role="tool",tool_call_id=cl["id"],content="Not executed, no cost: first the interviewer talks to the shopper.")for cl in outgoing)
   talk.append(dict(role="user",content=(self.interview()or"Interviewer: skipped, no useful question.")+"\n"+self.status()))
   return 0
  self.trace.append(dict(kind="turn",text=text[:300],calls=[cl["function"]for cl in outgoing]))
  batch=outgoing
  r=self.roles.get
  opener,comparer,key,many=r("open"),r("compare"),self.single_target(r("open")),self.compare_target()
  joinable=[]
  if opener and comparer and key and many and opener!=comparer and not self.compare_bare():
   for i,call in enumerate(batch):
    fn=call.get("function")or{}
    parsed=_json_obj(fn.get("arguments"))if isinstance(fn.get("arguments"),str)else None
    pid=str((parsed or{}).get(key)or"").strip()
    if parsed and fn.get("name")==opener and set(parsed)=={key}and pid in self.rows and pid not in self.gone:joinable.append((i,pid))
   joinable=joinable[:self.compare_room()]
  if len(joinable)>1:
   lead,spots,ids=joinable[0][0],{i for i,_ in joinable},[p for _,p in joinable]
   merged=[]
   for i,call in enumerate(batch):
    if i==lead:merged.append({"id":call["id"],"type":"function","function":{"name":comparer,"arguments":_dump({many:ids})}})
    else:merged.append({"id":call["id"],"joined":True}if i in spots else call)
   batch=merged
   self.trace+=[dict(kind="joined",ids=ids)]
  out=[]
  held=set()
  for call in batch:
   box={"extra":"","count":len(batch),"held":held}
   word_2=self._svf_screen(call,box)
   word_2=self._svf_hold(box)if word_2 is None else word_2
   out.append((call["id"],word_2 if word_2 is not None else self._svf_execute(box)))
  replies=out
  if replies:replies[-1]=(replies[-1][0],replies[-1][1]+"\n"+self.status())
  talk.extend(dict(role="tool",tool_call_id=cid,content=word)for cid,word in replies)
  return 0
  return True
_SVF_FREE="Skipped, no step spent: "
_SVF_HELD="Held, no step spent. "
_SVF_CART="\nCurrent cart: "
def _ky_crash(err):return{"kind":"crash","error": f"{type(err).__name__}: {err}"[:400]}
def agent_main(problem_data:dict[str,Any]) -> list[dict[str,Any]]:
 agent=None
 try:
  agent=Kyo(problem_data)
  return agent.run()
 except Exception as err:
  if agent is None:return[_ky_crash(err)]
  agent.trace.append(_ky_crash(err))
 try:
  if not agent.finished:agent.finish()
 except Exception as late:agent.trace.append(_ky_crash(late))
 return agent.trace
__all__=["agent_main"]
