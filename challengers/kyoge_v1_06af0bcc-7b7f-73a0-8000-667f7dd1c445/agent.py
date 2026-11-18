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
  FIRST_READS,VAGUE_WORDS,LATE_LOOK,HOP_READS)=(3,5,4,4,3,6,7,8,8,8,8)
ALL_CALLS,LOOP_CALLS,SPEND_HARD,SPEND_SOFT=60,42,0.050,0.035
WAIT_TOTAL,WAITS=45,(4,10,20,35)
HISTORY_LIMIT,OBS_LIMIT,SHOW_CHARS=150000,15000,450
CLOCK_EXTRAS,CLOCK_STOP,CLOCK_LIMIT=70,35,285
# most keys of one item a single question names; most words a code-built search sends
ASK_SPAN,QUERY_WORDS=7,3
_llm,_env=(ProxyClient(max_retries=tries,retry_delay=1,timeout=wait)for tries,wait in((1,50),(3,95)))
SHAPE:str="""Printed below is a result one of this store's tools returned, together with that tool's parameters. Map the store's own key names onto the fields of this answer and copy them character for character: "rows" = the key whose value is the list of listings; "price" = the key inside a row carrying its price (null if none); "stock" = the key inside a row telling if it can be purchased at this moment (null if none); "count" = the key with the total number of matches across all pages, not the number on this page (null if none); "next" = the key whose value fetches the following page (null if none); "next_arg" = the tool parameter that takes that value back (null if none). Output: a lone JSON object with exactly those six fields, no surrounding text."""
TRAIL:str="""Situation: a shopping agent will check out next. Certain requirements get their value from a listing other than the one bought, and a page may hand the reader on to yet another listing that holds the property, so the purchase is valid only once every listing along such a chain has been opened completely. Records further down are tagged OPENED (the agent opened that page) or ROW (only a search row, page never read).
Work through each such requirement one link at a time, starting from the requirement's wording and then following each page's own words.
- A page that refers the reader elsewhere for the property supplies no value for it, even a value it happens to print; the chain continues, and it stops only at a page that refers nowhere further.
- A pairing rule (a note telling which value belongs with which) speaks of values, never of listings: it refers nobody anywhere, and a listing that just displays the partner value is not a chain link.
- A record is never a match if its own attribute values conflict with a property named in the link's words, whatever its title says; a sentence on its page denying that property is a conflict as well. Either way that link's id remains "". And as long as that seller has rows that are not OPENED, a record saying nothing about such a property is not a match yet.
- List the links in order and halt at the first one that is not found or not OPENED, since its words cannot be known yet.
Output a lone JSON object, no commentary: {"chains": [{"need": the requirement copied verbatim, "relation": "same" if the REQUEST's own words want exactly that value, "apart" if the requirement wants some value other than it, "together" if the REQUEST wants a product made to match, pair with or function alongside that value (however it is phrased), otherwise "other", "value": the value worded as the last page words it for its own listing (for "together", the value you begin from, never the partner), or "", "partner": only for "together", the value that a pairing rule printed on an OPENED page of the chain (in remarks, notes or write-up) places next to "value" - rules are one-directional, "A goes with B" makes B the partner of A and not the reverse - or "", "done": true only when an OPENED page at the final link states the value itself, otherwise false, "links": [{"about": the way the requirement or the preceding page refers to that listing, "shop": the seller as named within those words (always fill it whenever a seller is named), otherwise "", "words": no more than 5 plain words taken from "about" that distinguish this listing from the seller's other ones, leaving out the words listing and page and the seller's name, "id": the id of the matching record below, or ""}]}]}"""
POINTER:str="""A request sometimes ties a property, rather than spelling its value, to some other listing or page in this store: equal to it, compatible with it, staying inside it. Find every property tied that way; if every value is stated outright, find none.
What can be that other listing: only goods sold in this store that the request mentions. Things the shopper owns already, or plans to use or carry together with the purchase, are not listings, so being suited to them makes no lookup. Neither does a product that must suit another product in this very order, nor a rule that ranks the candidates against one another (the cheapest one, a margin over the cheapest near miss): such rules decide between listings and refer to none in particular.
Output a lone JSON object and nothing else: {"lookups": [{"item": item number, "must": "property: the relation in the shopper's own phrasing, including the complete description of the other listing and of any listing that page refers on to"}]}"""
HEAR:str="""A shopping agent put questions to a shopper; their answers follow below. Treat each answer by itself. Folksy, offhand or roundabout phrasing that names a value still states that value.
How to fill each list:
* needs - what counts: only a value that a product page could show; a passing remark is never a need. Record every firm preference an answer gives, with its exact value or bound and its unit ("at least 30 cm", "oval", "not wicker"), keeping nouns and units verbatim ("at least 3 shelves", not "at least 3"). Examples inside the agent's question are not the shopper's words, and an answer that merely asks for fewer questions states nothing. Phrase a need as what it accepts: an answer turning down all but one value or bound accepts only that one ("material: oak", "width: at least 40 cm"); an answer turning down one value keeps the "not" ("material: not oak"). No double negatives, and no refusing words left inside the value.
* needs - one property, one need: answers about the same property merge into a single need, named for the property the answer actually talks about, even when a different one was asked.
* needs - borrowed values: an answer may fix the value by reference to another listing (like, unlike or close to the value on some shop's product, which it describes by a few properties). The properties that merely describe that other product are not needs of this item. Write ONE need for the property the answer names, with "points": true, whose value is the answer's exact wording from the comparison word to the end of that description, including anything said about a listing named there; neither trim nor paraphrase it.
* free - the property of each question whose answer leaves the decision to the agent, however it is worded. What such an answer says about the shopper's day, home or mood is chit-chat and gives no need.
* crowded - only answers objecting to too many topics in one message or asking for a single question at a time. An answer leaving the choice open belongs in free, and an answer only saying it was confused does not belong here.
Output a lone JSON object, n meaning the item number: {"needs": [{"item": n, "q": question number, "need": "property: constraint", "points": true if another listing supplies the value, else false}], "free": [{"item": n, "q": question number, "property": "..."}], "drop": [{"item": n, "property": "a requirement the shopper abandoned"}], "crowded": [question numbers], "budget": a new total cap as a number in the budget unit given below (convert any figure stated in another unit), otherwise null}"""
NOTE:str="""Midway through a shopping task the note below arrived, sent either by the store or by the shopper. Record every change it brings. When it swaps one requirement for another, put the new one under add and the old one under drop.
Output a lone JSON object, nothing else: {"summary": "one brief sentence", "kind": "change" if the task itself changed (budget, requirements, a price, availability, the order being refused), "error" if it only reports a call that was malformed or rejected, else "other"; "budget": the new total cap as a number in the request's unit, else null; "add": [{"item": item number, "need": "property: constraint"}]; "drop": [{"item": item number, "need": "the requirement the shopper takes back"}]; "gone": [ids the note calls unavailable]; "price_moved": true if it says any price changed; "sold_out": true if it says any chosen listing is sold out, gone or unavailable, whether named or not}"""
READING:str="""an attribute whose key names the property settles it, and keys for other properties settle nothing, even close relatives (the handle's colour is not the body's colour); with no such attribute, the wanted value under some other attribute or in the title or description counts, though a lone figure that never says what it measures counts for nothing. Read such attributes generously: bounds include their edge, units are converted, synonyms are accepted, any one member of a list or blend is enough whatever separates the members, key spelling is irrelevant, and a value holding the required words next to each other passes ("deep teal" for teal) while an extra word between them fails ("navy pale blue" is not navy blue); a list entry repeating a required range meets it, and a single required figure is met by a printed span of two figures containing it, edges included. A rule of differing from another listing's value is kept unless the records show that very value; while that value is still unread, only the listing referred to or its twin from the same seller breaks it. A requirement carried by the category label itself is met by that category, and the category label matches the item. Out-of-stock listings and listings above the budget never qualify. A requirement tagged "only if" applies only when its condition is true."""
SWAP:str="""A line in a shopping agent's cart now refers to a listing that has sold out. Go through the other records below and find listings that could take its place. Output a lone JSON object, no commentary: {"fits": [up to 6 entries {"id": id, "price": its price as a number taken from its record}, one for each listing that seems to meet every requirement of that line's item], "reason": "a sentence"}. How to read records: """+READING
AUDIT:str="""A shopping agent has concluded that nothing fits and is about to leave without buying. Go over the records below against the requirements one more time. A ROW record is just a search row whose attribute page was never read, so a property missing from it neither passes nor fails. Output a lone JSON object, nothing else: {"fits": {"item number": [up to 4 ids that seem to meet every requirement, cheapest first]}, "open": [up to 6 ROW ids of a wanted kind and within budget whose unread page might still reveal a match, most promising first], "reason": "a sentence"}. How to read records: """+READING
CHECK:str="""A shopping agent is about to pay, and you are the final check. Using nothing but these records, decide whether the cart is exactly right.
How to judge one requirement:
1. In force only. A property the shopper has no preference about, an instruction that names no property value or bound, and the background story of the request never make a listing fail.
2. Matching key decides. An attribute whose key names the property settles it, so a different stated value fails the listing. When several keys name the property, one of them stating the value is enough. The attribute is met when it states the value: a range sits wholly inside, a two-figure span containing a single wanted figure states that figure (edges count), bounds include their edge, units are converted, words standing next to each other inside a longer value pass ("deep teal" for teal) but not with an extra word between them ("navy pale blue" is not navy blue), case, singular/plural and plain synonyms pass, any one member of a list or blend is enough whatever separates them, and key spelling is irrelevant. A size made of several multiplied figures, or a list of sizes, never proves a range; a required range (two figures) is met by a list entry repeating it, whatever other keys say in other units, and a listing lacking that entry fails when RANGE MUSTS below names a listing that has it.
3. Unrelated keys. A key for a different property, however close (the handle's colour against the body's colour), settles nothing.
4. No key for the property. The wanted value under another attribute or in the title or description counts; a lone figure that never says what it measures does not, and neither does a number of another kind (weight against capacity). A requirement to work with or be compatible with some kind of thing is met only by words stating what the item works with or fits; being such a thing, containing one or running on one does not meet it. A fit list given for a part number belongs to the listing bearing that number (often in its title), whichever page prints it.
5. Pointers. Silence about an excluded value breaks nothing; an exclusion breaks only on a record that states it. A requirement marked not yet read is unmet; one marked as an exclusion whose value is still unread breaks only the listing it is marked as pointing to, or that listing's twin from the same seller, and every other listing meets it. A requirement taking its value from a referenced page holds only when the listing itself carries that value; a sentence naming the value as what the listing pairs with does not give it that value. A rule of differing from another listing's value holds unless the records show that very value.
6. Category. The label matches the requested item as a whole, whatever kind of thing the listing title names, and a requirement carried by the label itself is met by that category. Exception: when a word in the requested item's own name gives the physical form the product comes in, that form is a requirement too, and a listing whose attributes or title present the product in another physical form fails it.
7. Conditions. A requirement marked "only if" binds only when the records show its condition is true; when they show it is false, the requirement is met.
The cart as a whole: the sum of price times quantity over its lines stays within the budget; every line can still be bought; there are no extra lines; each requested item has one line at its requested quantity; the most recent record of a listing is the one that counts.
Pick rule (cheapest, most or least): look at the other listings shown as well; the cart is wrong when one of them plainly meets every requirement and beats the cart's choice under that rule.
Mark ok=false only for a requirement that no record addresses or for a fault the records plainly show.
Output exactly ONE JSON object: {"ok": boolean, "problems": ["one short, specific line per fault", ...], "better": [ids worth putting into a cart line instead, strongest first]}"""
STAND:str="""A shopping agent is about to leave holding nothing, and leaving empty-handed earns zero whenever the store does hold a match. For each item, pick the one OPENED record that comes closest. Exclusion: a printed value that differs from a requirement (another word meaning another thing, a figure past a bound, a different pack count), a price above the budget, out of stock, or a category for a different sort of product. A doubt is a requirement the record never mentions; it does not exclude, and title and description count as printed. Among records that are not excluded, fewest doubts first, then the pick rule given below. If all records of an item are excluded, its pick stays empty.
Output a lone JSON object: {"picks": {"item number": an OPENED record's id, or ""}, "breach": "for an empty pick, the printed value or price that rules its records out"}"""
UNMIX:str="""This product takes some of its values from a different listing the request names. The words that describe that listing - its kind, the shop selling it, the properties after "with ..." - are easy to mistake for constraints on this product. Using the request's wording, find which of the OTHER REQUIREMENTS below are like that. Output a lone JSON object, no commentary: {"describe_other": [those requirements, copied verbatim, that serve only to identify the other listing; empty when each one truly applies to this product]}"""
GATE:str="""On its way, the shopper's request requires the steps listed below, and the agent is about to STAGE. Go through the agent's actions in the numbered RECORD and, for the cart as it stands now, decide for every step that falls due before that next move whether it has been done.
- Done means some RECORD line performs it; for a step about the cart or the order, that line must also hold a reply granting it and must come after the last change to the cart.
- Questions about product properties never complete a step.
- "call": null if no due step is still open, or if the remaining messages or steps cannot cover the open one; otherwise the one next store call that gets an open due step done, a TOOLS entry with its arguments. Anything addressed to the shopper goes through the messaging tool as a single short, plain sentence or question about that step alone, saying nothing about the product (property or value) beyond what the step needs.
Output a lone JSON object, no commentary: {"steps": [{"step": the step, "lines": [the RECORD line numbers that show it], "done": boolean}], "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply sets], "hold": any change a reply asks for before anything else goes ahead, else "", "call": {"tool": "...", "args": {...}} or null}"""
LANES:str="""Your job: plan which questions a shopping agent may put to a shopper. The shopper has firm wishes the request never wrote down, each about one property of one requested item. A question uncovers a wish only if it names exactly that property, and the shopper answers about the first property a question names; so each question buys one property, and a wrong guess is wasted.
Input: the request; every item with what the request already settles for it; and per category, the attribute keys printed on the listings opened so far, with how many of those listings carry each key and some sample values.
Ordering, separately for each item:
(i) First, whatever the request wants checked with the shopper (listed under ASK FIRST), named by the property involved.
(ii) Next, the item's listed keys: those on which its listings vary and by which a buyer would choose this kind of product come first, favouring keys found on most of its listings; after five or more listings have been opened, a key found on just one or two of them moves down. A key whose listings all show the same value comes last.
(iii) A sentence in the request that concerns the item but leaves a property's value open (who will use it, how it must look next to something, what it must hold or the room it must fit, a choice the shopper says they cannot make) only breaks a near tie within (ii), favouring the listed key it suits; it never raises a key over one that (ii) clearly prefers, and talk of the shopper's day, job or home settles nothing.
Skip: a property the request already settles for that item (a key whose values just echo a settled value is still open), a second name for an already picked property, a key whose values list several purchase variants together, identifiers and model names, warranty, delivery and similar logistics.
Labels: the plain name of a single property, spelling out every word of the listed key ("seat height" for seatHeight, "strap width" for strap_width), with no values and no brackets.
Return as many as COUNT picks per item, best first within each item, as a lone JSON object with no other text: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 words of the request supporting this pick, or an empty string"}]}"""
BRIEF:str="""Turn the shopper's request and the store's tool list below into the task sheet a shopping agent will work from. Output one compact JSON object on a single line (no indentation), with nothing before or after it. Its fields:
"items": one entry for each distinct product to be bought, {"name": the kind of product in the request's words, "shelf": the store's category label for it copied verbatim when a tool lists labels, else "", "quantity": how many units of that single listing (an integer; 1 when the request is silent; a pair or set sold as one listing is 1; only a number written next to this product in the buying sentence counts, never a number the story suggests such as several people each needing one, rooms, uses or past purchases), "musts": the product's firm requirements, each written "property: constraint" with the precise value, bound, exclusion or closed list}.
  Rules for musts:
  - Value and wording. A must carries its value, bound, exclusion or closed list together with every word the request puts beside a figure (its unit or sizing system). An exclusion keeps its "not" however phrased ("anything but red" -> "color: not red"), never just the value it excludes. If the shopper cares about a property but gives no value, it is not a must: the interviewer will ask about it.
  - Name versus must. Words forming part of the product's name (a season, audience or use attached to it) belong in "name", not in musts. A style, look or material mentioned with the product, in either word order ("rustic-style lamps", "lamps, rustic style"), is a value the store label lacks: put it in musts ("style: rustic") and leave just the product kind in "name".
  - Story. Background (who will use it, where it goes, what broke before, the occasion) is context and gives no must unless it states a property value or bound outright.
  - Not musts: price, budget, stock and quantity.
  - Borrowed values. When another listing supplies the value, write ONE must holding that listing's complete description ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"); every word describing the other listing (its shop, its kind, the trailing "with <property> <value> and <property> <value>") stays inside that must and never becomes a must of this product. Preserve the shopper's relation: wanting this product to go with, pair with or be compatible with the other listing is not the same as wanting its value, so never turn one into the other, whatever figures of speech surround it; a value called right for or suited to the other listing is such a pairing, and "same as" is used only when the request asks for the same or a matching value.
  - Own words. Keep the request's property word ("meant for a lakeside occasion" -> "occasion: lakeside", "a 12-inch blade" -> "blade length: 12 inch"); never substitute a neighbouring property.
  - Conditions. A requirement the request makes only under a condition (it applies if some other property or listing turns out a given way) stays ONE must carrying the condition in the request's words, "<property>: <constraint>, only if <condition>"; never lose the condition and never make it unconditional.
  How many items: a single phrase joining products with "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; split only when each product has its own quantity or its own requirements.
"lookups": the musts (copied verbatim) whose value must first be read from another listing the request refers to; else [].
"wishes": things the shopper would merely like.
"ask_first": what the request explicitly wants confirmed with the shopper before the pick, one entry per thing named (a sentence naming two things makes two entries); empty when none.
"rules": every instruction on how to work or how to finish, each at most 12 words. A hope about how the product will behave or suit its place, with no property value or bound, is background and stays out.
"process": steps required besides finding and buying the right items (the shopper's approval, telling or showing them something, checking something with them, reading something first, using a certain tool). Each is {"step": the action in no more than 15 words, "when": "choose" (before products are chosen), "order" (right before checkout) or "end" (right before ending without a purchase), "messages": shopper messages it takes (0 if none)}; [] when there are none. Not steps: item requirements, budget, stock, the message and step limits, how to finish when nothing fits, and anything the request wants only for the case where no listing meets everything (telling the shopper, offering the near miss) - "nearest_fallback" and "quiet_decline" cover that case.
"nearest_fallback": true when the request wants the near miss (the listing missing least) offered or bought if no listing meets everything; else false.
"quiet_decline": true when the request says to suggest or name no alternatives if nothing fits, false when it asks for them, null when it is silent.
"pick": "cheapest" (lowest price or smallest total wanted), "most" / "least" (an extreme of some measure wanted), otherwise "any". Comments about money being tight set no pick rule.
"pick_measure": for "most" or "least", the property whose extreme is wanted (a property name, never the word most or least); otherwise "".
"budget": the most the shopper will spend, as a number in the request's own unit, or null.
"budget_unit": that unit in words (e.g. "cents"), or null.
"budget_scope": "total" (one cap across all items), "each" (a cap per item), or null (no cap).
"in_stock_only": true unless the request clearly accepts unavailable items.
"steps": the limit on steps, moves or tool calls (integer), or null.
"messages": how many messages the agent may send the shopper (integer); null when not stated.
"roles": an object naming the exact tool for each job, null where no tool does it: "search"; "open" (attributes or properties of one listing; if two tools handle a single listing, the one whose description offers attributes, properties or specifications); "offer" (the current offer of one listing, price and stock, when that is not the "open" tool); "compare" (several listings in full); "ask" (messages the shopper); "cart" (changes the cart); "cart_view" (shows the cart); "buy" (checks out); "refuse" (ends without a purchase); "browse" (lists one category, often under a price ceiling, when that is not the free-text "search" tool).
"buy_args": fixed arguments with which the "buy" tool checks out (for a tool taking a mode argument, that mode); otherwise {}.
"refuse_args": fixed arguments with which the "refuse" tool ends without a purchase; otherwise {}.
"search_text": when the "search" tool has more than one text parameter or a mode parameter, {"text": the parameter for free search words, "mode": {mode parameter: its value for a word search}}; otherwise {}.
"browse_args": for the browse tool, {"label": the parameter taking the category label or null, "cap": the parameter taking a top price or null}; otherwise {}.
"""
GUIDE:str="""You are Kyo. You shop for one shopper in an online store and can act only through that store's tools.
How you are scored: 1 or 0, nothing between, and only a finish that is exactly right in one of two ways earns the 1. Either (a) an order is placed holding every requested item at its requested quantity, each line meeting every requirement still in force at checkout, the total within the budget; or (b) the store has nothing that meets everything, and you end through the no-purchase tool in the way the request asks.
Each reply you send holds one or more tool calls plus one short line of text: what is settled so far, then your next move.

== A. Working with the store ==
A1. Every call costs one step out of a small allowance. Thinking costs nothing, but each reply costs clock time, so batch independent searches and reads into one reply. Tool names, argument names and result shapes are particular to this store: read every tool's description and schema, and copy argument names and permitted values exactly.
A2. A search row is only a summary. Stores split a record their own way: attributes (properties) from one tool, offer and description from another, and maybe a side-by-side tool showing either. When Kyo learns which tool returns attributes, the ledger notes it. Property requirements are judged on attributes, so read them for every listing you may buy; one call that covers several listings saves steps when it shows what you need. A listing once read stays read until something changes.
A3. Searches may be limited: a category filter, keywords matched only against a query's first words, short pages, a next-page token or offset. When a decision depends on every row, compare the total the store reports with the rows you hold and keep paging, handing back the token or offset exactly as received. Short queries that lead with the value work best: wanted value first, product noun after. A filtered search with no rows usually means a wrong filter value or parameter; retry with the label spelled as rows print it, or switch to another filter parameter.
A4. A request may require steps besides buying (an approval, a message, a check, reading something first); the ledger lists them. Steps that fall due just before the order or the ending run by themselves when you call checkout or the ending tool: don't wait for them, and never count them as a reason to leave without buying. Carry out any other step yourself with store tools (product questions still go through the interviewer).
A5. You never message the shopper; Kyo's interviewer sends every message. Its questions and the answers appear in the conversation and are copied into the ledger.

== B. Deciding whether a listing fits ==
B1. Only the ledger decides fit: its musts, budget and stock. The shopper's story (where it goes, who uses it, what failed before) adds nothing beyond the ledger, and a ledger instruction naming no property value or bound is not a requirement. Don't search for such things, don't hold a listing unconfirmed over them, and don't reject a listing because of them. A property marked "no preference" fails nothing. The moment an opened, buyable listing meets every must, buy the best one by the pick rule; leaving empty-handed needs a must that every listing breaks.
B2. When some key names the required property (by that name or a plain synonym), that attribute alone settles it: a different stated value fails the listing whatever the title or other keys say. When several keys name it, the one named exactly like the property settles it; failing that, any of them stating the value is enough. When no key names it, the wanted value under another attribute or in the title counts as met. A key for any other property, related or not (the handle's colour is not the body's colour), settles nothing for it. Kyo adds [evidence] lines to opened listings naming keys that match a requirement.
B3. When no attribute covers the property, the wanted value counts if it appears under another attribute (say a styles or features list that holds it) or in the title or description. A number measuring something else doesn't count (weight isn't capacity), and neither does a lone figure whose text never says what it measures. A must to be compatible with something counts only where the text says what the item works with, never where the item is that thing, contains one or is powered by one.
B4. A property requirement is met when the attribute for that property states the value. Key spelling doesn't matter (spaces, underscores, camel case, prefixes), and any one entry of a list or blend is enough, whatever joins the entries (comma, slash, plus sign). Compare values word by word, ignoring case, singular or plural and plain synonyms; a value holding the required words next to each other counts ("deep teal" meets teal, "extra roomy fit" meets roomy), but not with another word in between ("navy pale blue" is not navy blue). Other words don't count: an abbreviation or code is not the word it abbreviates, and "teal green" is not "sea green". Convert units before comparing figures; a bound includes its edge; a range counts only when it lies wholly within the bound. A single wanted figure (no bound word) is stated by an attribute printing a two-figure span that contains it, edges included: that span is not a different value. For a closed list, any one value is enough.
B5. Ranges: a must naming a range (two figures) names a value the store prints. An attribute holding that range as one list entry meets it, and no other key, converted figure or rounding can override that. A listing lacking such an entry is only a fallback, used once no listing with the entry can be found; then one plainly stated figure converted into the range may do, never a size built from several multiplied figures or a list of sizes.
B6. The category label gives the kind of product, and only a label matching the requested item as a whole counts; a neighbouring label never does. A listing filed under the requested item's own label is that item even when its title names another or a smaller kind of thing: never rule it out, or skip it in a price comparison, because of its title. A label that itself holds a requirement's value (a season or audience within the label) meets that requirement.
B7. Exclusions and closed lists bind; wishes ("ideally", "would be nice") only break ties.
B8. Never buy a listing no longer in stock. Sum price times quantity over all lines: the total may equal the budget but never exceed it.
B9. Before buying a listing that sends you to another page for details about itself, read that page.
B10. Paired items (an accessory for a device): choose them together from what the listings themselves say (a fit list, a model or series code shown on both pages); a fit list given for a part number belongs to the listing carrying that number, whichever page prints it. If no accessory that meets its own requirements fits the chosen device, move to another qualifying device that such an accessory fits; try that move before any empty-handed ending.

== C. Choosing among fitting listings ==
C1. Each distinct item gets its own cart line, at its requested quantity.
C2. Put an item in the cart as soon as you have opened a listing meeting all its requirements (under a cheapest, most or least rule, once the fitting set has been compared), then go on. Carted lines often sell out shortly after, and carting early keeps steps free for a replacement.
C3. With no pick rule, any fully fitting listing is right; don't spend steps hunting for a better one.
C4. When the most or least of a measure is wanted, compare every fitting listing on that measure and take the extreme; a listing that says nothing about the measure cannot win.
C5. When the cheapest is wanted (lowest total, smallest price), the basket takes the lowest-priced of ALL fitting listings. Gather the whole fitting set (page through, as rows often arrive unsorted by price; Kyo fetches further pages of short results by itself), test candidates from the lowest price upward and keep the first one that fully fits. If a search takes a maximum price, list the category below your current best pick's price so only cheaper candidates appear.

== D. Values that come from another listing ==
D1. Some requirements name a second listing instead of a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Settle these first, in few steps. If a shop is named, search with just the shop's name as text, which brings back only that shop's handful of listings (a shop filter on a browsing tool mostly returns nothing), then read them with the side-by-side tool; if no shop is named, search by category, brand or words.
D2. Open the full page with the single-item or side-by-side tool; a row is not enough. Should that page defer the property to a further listing, find and open that one too, and so on; the last page gives the value. An order only counts once every page in that chain has been opened.
D3. Apply the value as the shopper's own comparison says. Same or matching: the listing you buy shows that value. A bound: measure from that value. Going well with, pairing with or being compatible with it, however worded: the value is just your starting point. The last page, or a rule printed on a page you read, says which other value goes with it, and the listing you buy must itself carry that partner value for the property; search anew on the partner value; only when no page you read prints such a rule does the starting value itself stand. Otherwise listings with the starting value fail, the chain's end listing too, and a sentence that just names the partner as a companion gives a listing no such value. Differing from it: only listings with exactly that value are out, and this alone never justifies an empty-handed finish.

== E. Changes during the task ==
E1. Prices and stock can change unannounced, but Kyo re-checks every cart line itself when checkout is sent: send checkout rather than reading a carted listing again.
E2. The shopper or the store may change the task midway: a new budget, a requirement added or dropped, a new price, a listing sold out. The news arrives inside tool results, often as a refusal when you try to check out; a note counts whichever tool returns it, and the ledger shows the requirements now in force. A refusal does not end the episode: apply each change, fix the cart and check out again. If only a price moved and the sum is still under the budget, keep that line: read the listing again and resend checkout instead of replacing it.

== F. Finishing ==
F1. Hold only the chosen lines in the cart (remove the rest), view the cart once, then send checkout as a call by itself.
F2. End without buying only after a genuine search found no listing meeting everything. Being short of steps proves no such thing; then check out the best cart you have. Word that ending as the request asks, and if a remark is needed keep it brief and name no listing."""
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
answered:d page_bad:s page_args:d mode_pins:d facts_read:d tokens_seen:s page_ids:s proven_goal:n proven_tried:b apart:d refilled:s cap_cut:n mended:s ver_seen:d blind:d thin_reads:s code_finds:d light_mute:b sweeps:d trail_asks:d hop_reads:d cap_modes:d cap_said:s last_reads:b cap_keys:d start_learned:n valued:s line_at:d cap_stated:b"""
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
 if token is None:return not kinds.isdisjoint({"integer","string"})
 if isinstance(token,float)and token.is_integer():token=int(token)
 if isinstance(token,bool)or not isinstance(token,(int,str)):return False
 want="string"if isinstance(token,str)else"integer"
 errs=[]
 _shape(token,spec,"",errs)
 if want not in kinds or errs:return False
 top=spec.get("maxLength")
 return not(want=="string"and isinstance(top,int)and len(token)>top)
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
 if isinstance(obj,str):
  low=obj.strip().lower()
  if kind in("integer","number")and _kb_NUMERIC.fullmatch(obj):
   figure=float(obj)
   return True,(int(figure)if kind=="integer"or figure.is_integer()else figure)
  if kind=="boolean"and low in("true","false"):return True,low=="true"
 elif kind=="integer"and isinstance(obj,float)and obj.is_integer():return True,int(obj)
 elif kind=="string"and _kb_num(obj):return True,str(obj)
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
def _kb_pick_option(obj,options,path,errs):
 if isinstance(obj,str):
  wanted=obj.strip().lower()
  obj=next((o for o in options if isinstance(o,str)and o.lower()==wanted),obj)
  near=[o for o in options if isinstance(o,str)and _kb_norm(o)==_kb_norm(obj)]
  if len(near)==1 and obj not in options:obj=near[0]
 if obj not in options:errs.append(f"{path}: allowed values are {options}")
 return obj
def _shape(obj,spec,path,errs):
 if not isinstance(spec,dict):return obj
 kinds=_kinds(spec)
 if kinds and not _kb_fits(obj,kinds):
  obj=_kb_coerce(obj,kinds)
  if not _kb_fits(obj,kinds):
   errs.append(f"{path} should be {'/'.join(kinds)}")
   return obj
 options=spec.get("enum")
 if isinstance(options,list)and obj not in options:obj=_kb_pick_option(obj,options,path,errs)
 if isinstance(obj,str):
  top,bottom=spec.get("maxLength"),spec.get("minLength")
  cut=obj[:top]if isinstance(top,int)and len(obj)>top else obj
  if isinstance(bottom,int)and len(cut)<bottom:errs.append(f"{path} is too short")
  return cut
 if _kb_num(obj):
  errs.extend(path+" "+note for note in _kb_over(obj,spec,_kb_NUM_EDGES))
  return obj
 if isinstance(obj,list):
  errs.extend(path+" "+note for note in _kb_over(len(obj),spec,_kb_LIST_EDGES))
  inner=spec.get("items")
  return[_shape(sub,inner,"%s[%d]"%(path,at),errs)for at,sub in enumerate(obj)]if isinstance(inner,dict)else obj
 if not isinstance(obj,dict):return obj
 known=spec.get("properties")
 if not isinstance(known,dict):known={}
 if spec.get("additionalProperties")is False:errs.extend(f"{path}.{arg}: the schema has no such argument" for arg in obj if arg not in known)
 errs.extend(f"{path}.{arg} is required" for arg in spec.get("required")or[]if arg not in obj)
 return{arg:_shape(sub,known[arg],f"{path}.{arg}",errs)if arg in known else sub for arg,sub in obj.items()}
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
def _loose_owner(named,pending,taken):
 # a chain that restates its requirement in other words answers the one open lookup whose property it names and that shares
 # most of its words, at least half of the lookup's
 said=_tokens(named)
 share=lambda p:len(_tokens(p)&said)/max(1,len(_tokens(p)))
 ranked=sorted((p for p in pending if p not in taken and _tokens(p.partition(":")[0])&said),key=share,reverse=True)
 if not said or not ranked or share(ranked[0])<.5 or len(ranked)>1 and share(ranked[1])==share(ranked[0]):return None
 return ranked[0]
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
 if lead:
  parts=[x for x in re.split(r"\n(?=\s*%s: )"%re.escape(lead.group(1)),raw)if x.strip()]
  rows=[_text_record(x,depth+1)for x in parts]if len(parts)>=2 else[]
  if rows and all(isinstance(r,dict)for r in rows):return rows
 tries=(_text_pairs(raw,cut,": ",True)if raw.count(cut)>=2 else{}for cut in(" | ","\n"))
 rec=next((r for r in tries if len(r)>=3),None)
 if rec is None:return None
 for key,val in rec.items():
  lists=(_text_pairs(val,cut,"=")if val.count("=")>=2 and cut in val else{}for cut in("; ",", "))
  inner=next((p for p in lists if len(p)>=2),None)
  rec[key]=_text_value(val)if inner is None else{k:_text_value(v)for k,v in inner.items()}
 return rec
def _untext(node,depth=0,spot=True):
 # text records where a whole record sits (a list entry, a reply of one or two values) become records; a reply holding just one becomes that record
 if depth>6:return node
 if isinstance(node,str):return spot and _text_record(node)or node
 if isinstance(node,list):
  out=[_untext(x,depth+1)for x in node]
  return _named_values(out)or out
 if not isinstance(node,dict):return node
 small=len(node)<=2
 out={k:_untext(v,depth+1,small)for k,v in node.items()}
 # listings keyed by their ids: each becomes a row that carries its id
 keyed=lambda k,v:_id_like(k)and isinstance(v,dict)and len(v)>=2 and k not in map(str,v.values())
 if out and all(keyed(k,v)for k,v in out.items()):return[{**v,"id":k}for k,v in out.items()]
 if len(out)!=1:return out
 (only,)=out.values()
 return only if isinstance(only,dict)and isinstance(next(iter(node.values())),str)else out
def _named_values(entries):
 # a list naming one value per entry (name/value records, [name, value] pairs, "name: value" lines) as one name -> value record
 if len(entries)<2:return None
 if all(isinstance(e,list)and len(e)==2 for e in entries):pairs=list(map(tuple,entries))
 elif len(entries)>=3 and all(isinstance(e,str)for e in entries):
  lines=(re.fullmatch(r"\s*([^:=\n]{1,40}?)\s*[:=]\s*(.*)",e,re.S)for e in entries)
  pairs=[m.groups()for m in lines if m]
 elif all(isinstance(e,dict)and len(e)==2 for e in entries)and len({tuple(e)for e in entries})==1:
  first,second=list(entries[0])
  label=lambda c:all(isinstance(e[c],str)and 0<len(e[c].strip())<=40 for e in entries)and not all(_id_like(e[c])for e in entries)
  names=next((c for c in(first,second)if label(c)),None)
  other=second if names==first else first
  pairs=[(e[names],e[other])for e in entries]if names else[]
 else:return None
 names=[str(a).strip()for a,_ in pairs]
 values=[b for _,b in pairs]
 if len(pairs)<len(entries)or any(not isinstance(a,str)for a,_ in pairs)or len(set(names))<len(names):return None
 # names that look like listing ids, or values that all do, make a list of rows, not of details
 if all(map(_id_like,names))or all(map(_id_like,values)):return None
 return{a:_text_value(b)if isinstance(b,str)else b for a,b in zip(names,values)}
def _lettered_figure(word):
 return bool(re.search(r"\d",word)and re.search(r"[A-Za-z]",word))
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
  self.binding=dict(env["binding"])
  pv=self.view=env["policy_view"]
  self.request=str(pv.get("query")or"")
  named=lambda e:isinstance(e,dict)and isinstance(e.get("function"),dict)and isinstance(e["function"].get("name"),str)
  self.tools=[e for e in pv.get("tools")or[]if named(e)]
  params=lambda f:f.get("parameters")if isinstance(f.get("parameters"),dict)else{"type":"object","properties":{}}
  self.schemas={e["function"]["name"]:params(e["function"])for e in self.tools}
  raw=pv.get("max_steps")
  cap=_count(raw)
  if cap is None and isinstance(raw,float):cap=int(raw)
  self.max_steps=cap if cap and cap>0 else 30
  self.finished=bool(pv.get("finished"))
  for factory,name in _svb_STATE:setattr(self,name,factory())
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
  props=spec.get("properties")or{}
  kept=self.page_args.get(tool)
  if kept in props and(tool,kept)not in self.page_bad and(token is None or _kb_page_fit(props[kept],token)):return kept
  named=(self.brief.get("browse_args")or{}).values()if tool==self.roles.get("browse")else()
  taken={*(spec.get("required")or[]),*named,self.text_fields.get(tool),*self.mode_pins.get(tool,{}),*self.fixed_args.get(tool,{})}
  taken.update(k for k,v in(call_args or{}).items()if k!=kept and _dump(v)not in self.tokens_seen)
  fits=[a for a,fs in props.items()if a not in taken and(tool,a)not in self.page_bad and _kb_page_fit(fs,token)]
  if len(fits)>1 and(token is None or isinstance(token,str)):
   size=lambda a:props[a].get("maxLength")if isinstance(props[a].get("maxLength"),int)else 10**6
   least=min(map(size,fits))
   fits=[a for a in fits if size(a)==least]
  pick=(self.shapes.get(tool)or{}).get("next_arg")
  if pick in fits:arg=pick
  else:arg=fits[0]if len(fits)==1 else None
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
   sent=[dict(m)for m in history]
   for turn in(sent[0],sent[-1]):
    text=turn.get("content")
    if text and isinstance(text,str):turn["content"]=[{"type":"text","text":text,"cache_control":{"type":"ephemeral"}}]
  body={"model":model,"messages":sent,"max_tokens":max_tokens,"temperature":0}
  if tools:body.update(tools=tools,tool_choice="auto")
  _llm.timeout=max(5,min(50,self.secs()-5))
  res=_llm.post_verbose("/inference/chat/completions",body)
  data=res.data
  if not(isinstance(data,dict)and data.get("choices")):
   err=res.error or{}
   code,why=err.get("status"),err.get("body")
   inner=data.get("error")if isinstance(data,dict)else None
   if isinstance(inner,dict):code,why=_count(inner.get("code")),inner.get("message")
   self.trace.append({"kind":"llm_fail","model":model,"status":code,"body":str(why or"")[:200]})
   return None,code
  spent=(data.get("usage")or{}).get("cost")
  if isinstance(spent,(int,float)):self.cost+=float(spent)
  out=data["choices"][0].get("message")or{}
  if out.get("content")or out.get("tool_calls"):return out,200
  self.trace.append({"kind":"llm_fail","model":model,"status":0})
  return None,0
 def ask_json(self,prompt,models,max_tokens=1800):
  for engine in models:
   reply=self.think([{"role":"user","content":prompt}],[engine],max_tokens)
   got=_json_obj((reply or{}).get("content"))
   if got is not None:return got
   if self.broke or self.no_credit or self.calls>=ALL_CALLS:return None
  return None
 def post_env(self,call):
  """POST one store call, retrying as the error verdict says; the reply, or None once the call cannot go through."""
  for _ in range(4):
   res=_env.post_verbose("/environment/call",call)
   if isinstance(res.data,dict):return res.data
   err=res.error or{}
   code,text=err.get("status"),str(err.get("body")or"")
   self.trace.append({"kind":"env_fail","status":code,"body":text[:200]})
   kind,val=_kc_env_verdict(code,text,self.step)
   if kind=="seek":
    self.step=val
    call["step"]=val+1
   elif kind!="again":
    if kind=="closed":self.finished=True
    elif val:self.broke=True
    return None
  return None
 def after_clean(self,tool,merged,obs,seek):
  tf,pins=self.text_fields.get(tool),self.mode_pins.get(tool)or{}
  if seek and tf and str(merged.get(tf)or"").strip()and any(f in merged and merged[f]!=v for f,v in pins.items()):
   # off the word mode, two different word sets that list the same rows show the words go unused there
   sig=tool+_dump({k:v for k,v in merged.items()if k!=tf})
   ids=tuple(sorted(set(_kb_TOKEN.findall(_dump(obs)))))
   was=self.blind.get(sig)
   if was and was[0]!=merged[tf]and was[1]==ids and any(not i.isdigit()for i in ids):self.blind[tool]=True
   self.blind.setdefault(sig,(merged[tf],ids))
  if tool==self.roles.get("search"):
   spec=self.schemas.get(tool)or{}
   enum=lambda f:"enum"in((spec.get("properties")or{}).get(f)or{})
   self.fixed_args.setdefault(tool,{f:merged[f]for f in spec.get("required")or[]if f in merged and enum(f)})
  if tool in self.page_tools():
   # a listing counts as opened only when the reply carries it: an empty side-by-side reply opens nothing
   shown=_dump(obs)
   named=[p for v in merged.values()for p in(v if isinstance(v,list)else[v])if isinstance(p,str)]
   self.opened.update(p.strip()for p in named if p.strip()in shown)
 def act(self,tool,arguments):
  self.ver_seen.setdefault(self.version,self.step)
  merged={**self.fixed_args.get(tool,{}),**arguments}
  seek=tool in self.seekers()
  mark=tool+_dump(merged)
  if seek and self.answered.get(mark)==self.version:
   self.trace.append({"kind":"repeat_held","tool":tool})
   return None
  got=self.post_env({**self.binding,"step":self.step+1,"action":{"name":tool,"args":merged}})
  if got is None:return None
  self.step=got["step"]if isinstance(got.get("step"),int)else self.step+1
  if got.get("finished"):self.finished=True
  obs=got["observation"]=_untext(got.get("observation"))
  self.page_ids.update(_kb_TOKEN.findall(_dump(obs)))
  if seek:self.answered[mark]=self.version
  if not(isinstance(obs,dict)and obs.get("error")):self.after_clean(tool,merged,obs,seek)
  return got
 def sibling_ref(self,slot,text):
  # a must tying an item to another product of this same order points at no outside listing
  said=_tokens(text.split(":",1)[-1])
  return any(n!=slot and it.name and _tokens(it.name)<=said for n,it in enumerate(self.items,1))
 def settle_musts(self,reading):
  linked={r for _,r in self.refs}
  tail=lambda m:m.split(":",1)[-1]
  for slot,it in enumerate(self.items,1):
   vague=lambda m:m not in linked and not self.sibling_ref(slot,m)and len(tail(m).split())>=VAGUE_WORDS and not re.search(r"[0-9A-Z]",tail(m))
   loose=[m for m in it.musts if vague(m)]
   it.musts=[m for m in it.musts if m not in loose]
   reading["wishes"]=_items(reading.get("wishes"))+loose
 def find_pointers(self):
  menu="\n".join("%d: %s"%(n,it.name)for n,it in enumerate(self.items,1))
  got=self.ask_json("%s\n\nITEMS:\n%s\n\nREQUEST:\n%s"%(POINTER,menu,self.request),[HEAVY,LIGHT],800)or{}
  size=len(self.items)
  for e in(e for e in _items(got.get("lookups"))if isinstance(e,dict)):
   text=str(e.get("must")or"").strip()
   n=_count(e.get("item"))or 1
   if not text or not 1<=n<=size or self.sibling_ref(n,text):continue
   it=self.items[n-1]
   head=_words(text.split(":")[0])
   it.musts=[*(m for m in it.musts if _words(m.split(":")[0])!=head),text]
   self.refs.append((n,text))
 def unmix(self):
  for slot,it in enumerate(self.items,1):
   linked=[r for p,r in self.refs if p==slot and r in it.musts]
   rest=[m for m in it.musts if m not in linked]
   if not(linked and rest):continue
   ask="%s\n\nShopper's request:\n%s\n\nProduct under review: %s\nMusts that copy a value off a referenced listing:\n%s\nOTHER REQUIREMENTS:\n%s"%(
     UNMIX,self.request,it.name,"\n".join(linked),"\n".join(rest))
   got=self.ask_json(ask,[HEAVY,LIGHT],500)or{}
   told={str(x).strip()for x in _items(got.get("describe_other"))}
   moved=[m for m in rest if m in told]
   if not moved:continue
   head=linked[0]
   joined="%s (that listing: %s)"%(head,"; ".join(moved))
   it.musts=[m if m!=head else joined for m in it.musts if m not in moved]
   self.refs=[(p,joined if(p,r)==(slot,head)else r)for p,r in self.refs]
   self.trace.append({"kind":"unmix","item":slot,"moved":moved})
 def ledger(self):
  """The requirements in force, as the model and the reviewers see them."""
  many,rows=len(self.items)>1,[]
  for n,it in enumerate(self.items,1):
   waiting={r for p,r in self.refs if p==n}
   shown=[]
   for m in(m for m in it.musts+it.told if m not in it.dropped):
    if m in self.ref_values:shown.append(m+" (= %s, as read on the page it points to; text identifying that other listing is never a property here)"%self.ref_values[m])
    elif m not in waiting:shown.append(m)
    elif m in self.apart:shown.append(m+" (exclusion, value not read yet: it can fail only the listing it refers to [%s] or that listing's same-shop twin; all other listings pass it)"%self.apart[m])
    else:shown.append(m+" (referenced value not read yet)")
   bits=["item %d"%n if many else"item",": ",it.name]
   if it.shelf:bits.append(" [store category: %s]"%it.shelf)
   bits.append(", quantity %s; musts -> %s"%(it.qty,"; ".join(shown)or"none beyond the product itself"))
   if it.free:bits.append("; open to anything on: "+", ".join(it.free))
   rows.append("".join(bits))
  rd=self.brief
  if self.budget is not None:
   scope="each item"if rd.get("budget_scope")=="each"else"whole order"
   rows.append(("spending cap (%s): %s %s"%(scope,self.budget,rd.get("budget_unit")or"")).replace("  "," ").strip())
  how=rd.get("pick")
  if how=="cheapest":rows+=["choice: of everything that fits, the cheapest wins",*self.read_by_price()]
  elif how in("most","least"):
   measure=str(rd.get("pick_measure")or"")
   rows.append(("choice: whichever fitting listing has the %s %s"%(how,measure if not _stem(measure)<=_stem(how)else"")).strip())
  else:rows.append("choice: no ranking - any listing that fits every condition is acceptable")
  if rd.get("in_stock_only",True)is not False:rows.append("skip listings that cannot be bought now (out of stock)")
  rows.extend("tie-breaker only (a wish): %s"%w for w in _items(rd.get("wishes")))
  rows.extend("per the request: %s"%t for t in _items(rd.get("rules")))
  if rd.get("quiet_decline")is True:rows.append("closing empty-handed because nothing fits: mention no listing by name and propose none")
  if rd.get("nearest_fallback")is True:rows.append("when nothing meets every requirement: place no order and end without buying, and in the ending text offer the nearest listing (id plus what it lacks) as the near miss the request wants mentioned; the near miss is only offered, not purchased")
  rows.extend("%s, do this: %s"%(self._svc_WHEN[p["when"]],p["step"])for p in self.process)
  via=self.facts_tool()
  if via and(self.facts_from.get(via)is True or self.roles.get("offer")):
   pages=self.page_tools()
   empty=sorted(t for t in pages|{self.roles.get("offer")or""}if t and self.facts_from.get(t)is False)
   others=sorted(t for t in pages if t!=via)
   rows.append("".join(("read attributes with ",via,"; %s show none"%", ".join(empty)if empty else"",
    "; to walk linked pages or verify a page, call "+" or ".join(others)if others else"")))
  return"\n".join("- %s"%r for r in rows)
 def status(self):
  rest,empty=self.left(),not self.buys
  live=lambda:len([x for x,n in self.cart_lines.items()if n>0 and not self.sold_out(x)])
  tips=((lambda:rest<=self.tail()+1," LAST FEW CALLS: with a cart line for each item, check out now; end without buying only if your searches proved some item has no fitting listing anywhere - a shortage of steps proves no such thing."),
   (lambda:rest<=6 or(empty and rest<=len(self.items)+5)," Very few calls remain - add precisely the selected listings, then check out at once."),
   # a checkout can still be turned down (a new budget, a sold-out line); answering that takes steps, so a ready cart goes in while they remain
   (lambda:empty and rest<=9 and live()>=len(self.items)," Every item has its line in the cart: send checkout now - the store can still refuse the order, and handling that refusal needs the steps that remain."),
   (lambda:empty and rest<=12," Steps are getting scarce - commit to picks, cart them, then complete checkout shortly."),
   # half the wall clock gone counts as half the budget: on a slow store the clock ends the episode long before the steps do
   (lambda:empty and not self.cart_touched and(self.step>=self.max_steps//2 or self.secs()<=CLOCK_LIMIT/2)," The cart is still empty after half the steps or half the clock: cart a suitable listing for every item at once (stock may still run out)."))
  head="[Kyo] %s of %s steps left (%s used)."%(rest,self.max_steps,self.step)+next((tip for test,tip in tips if test()),"")
  if empty and self.secs()<CLOCK_EXTRAS+20:head+=" The clock is nearly out as well: within two or three calls, cart the best candidate found so far and check out; running out of time says nothing about stock."
  todo=[t for t in self.open_lookups()if t not in self.apart]
  parts=[head]
  if todo and empty:parts.append("[open lookups: get each value by opening the listing the request refers to, then every next link that page hands on - the final page has it; for a must about pairing with it, use the partner stated there] "+" | ".join(todo[:3]))
  if self.leads:parts.append("[candidates to review] "+", ".join(self.leads))
  if self.notes:parts.append("[changes up to now] "+" | ".join(self.notes[-4:]))
  return"\n".join(parts+["[ledger]\n%s"%self.ledger()])
 def ident(self,record,in_list=False,depth=0):
  if not isinstance(record,dict)or len(record)<(2 if in_list else 3):return None
  guess=self.id_key
  if guess and _id_like(record.get(guess),True):return record[guess].strip()
  if not(in_list or not depth or any(isinstance(v,dict)for v in record.values())):return None
  guess=_learn_id([record])
  if guess and("id"in str(guess).lower()or record[guess].strip()in self.rows):return record[guess].strip()
  return None
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
   if depth>6:continue
   if isinstance(node,list):
    flag=self.learn_rows(node)
    stack.extend((depth+1,flag,sub)for sub in reversed(node))
    continue
   if not isinstance(node,dict):continue
   lid=self.ident(node,listed,depth)
   if not lid:
    stack.extend((depth+1,False,sub)for sub in reversed(list(node.values())))
    continue
   have=self.rows.get(lid)
   if have is None:
    self.seen_order.append(lid)
    have=self.rows[lid]={}
   pairs=list(node.items())
   added={k:v for k,v in pairs if have and k not in have and(_kb_num(v)or isinstance(v,str))}
   if not listed and len(added)>=2 and not _fact_block(node):node={**node,"details":added}
   flips={(have.get(k),v)for k,v in pairs if isinstance(have.get(k),bool)and isinstance(v,bool)}
   if(True,False)in flips:self.unavailable.add(lid)
   elif(False,True)in flips:self.unavailable.discard(lid)
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
  short=lambda v:isinstance(v,str)and v and len(v)<=40 and re.search(r"\d{3}",v)is None
  for k,v in((k,v)for row in self.rows.values()for k,v in row.items()if short(v)):
   pool.setdefault(k,set()).add(v)
   counts[k]=counts.get(k,0)+1
  floor=0.8*max(1,len(self.rows))
  enums=self.labels()
  mean=lambda xs,n:sum(xs)/n
  scored=[((-mean((x.strip().lower()in enums for x in vals),len(vals)),len(vals),-counts[k],-mean(map(len,vals),len(vals))),k)
   for k,vals in pool.items()if counts[k]>=floor and len(vals)<=40]
  return min(scored,key=lambda sk:sk[0])[1]if scored else None
 def _svc_on_item(self,candidates):
  long=lambda t:{w for w in _stem(t)if len(w)>=4}
  marks=[(_words(it.shelf or""),_words(it.name),long(it.shelf or"")|long(it.name))for it in self.items]
  return[c for c in candidates if any(_words(c)in(a,b)or long(c)&bag for a,b,bag in marks)]
 def _svc_kept_keys(self,group):
  picked=[]
  for name,info in sorted(group["keys"].items(),key=lambda kv:-kv[1]["n"]):
   vals=list(info["per"].values())
   n=len(vals)
   if not n:continue
   sizes=sorted(map(len,vals))
   if sizes[n//2]>40 or 2*sum(" / "in v for v in vals)>=n:continue
   if n>=4 and len(set(vals))==n and sum(sizes)/n>12 and not _common_words(vals):continue
   if any(name.startswith(p+" ")and info["per"]==pi["per"]for p,pi in picked):continue
   picked.append((name,info))
  return picked
 def key_groups(self):
  """Opened pages per category: how many, which, and per attribute key its count, distinct values and value by listing."""
  cat,groups=self.category_key(),{}
  for lid in self.seen_order:
   page=self.full.get(lid)
   if not page:continue
   g=groups.setdefault(str(page.get(cat)or"listings")if cat else"listings",{"n":0,"keys":{},"ids":[]})
   g["n"]+=1
   g["ids"].append(lid)
   seen=set()
   for rk,rv in(_fact_block(page)or{}).items():
    key=_words(rk)
    if not key or key in seen:continue
    seen.add(key)
    slot=g["keys"].setdefault(key,{"n":0,"values":[],"per":{}})
    slot["n"]+=1
    slot["per"][lid]=v=str(rv).strip().lower()
    if v not in slot["values"]:slot["values"].append(v)
  return groups
 def group_rows(self,g,copies,said):
  flat=lambda r:r[2]<=1 and r[1]["n"]>=3
  rows=sorted(((k,info,sum(map(bool,info["values"])))for k,info in self._svc_kept_keys(g)),key=(lambda r:(-r[1]["n"],flat(r)))if copies else flat)
  if len(rows)>18:
   # a key repeating an earlier row's property spends a table slot that a distinct property needs
   lone=[]
   for r in rows:
    if all(not self.same_prop(r[0],o[0])for o in lone):lone.append(r)
   rows=[*lone,*(r for r in rows if r not in lone)]
  named=lambda r:" %s "%r[0]in said
  spill=rows[18:]
  return rows[:18]+[r for r in spill if named(r)]+[r for r in spill if flat(r)and not named(r)][:3]
 def key_table(self):
  said=" ".join(self.described_words(n)for n in range(1,len(self.items)+1))
  groups=self.key_groups()
  on=self._svc_on_item(groups)
  if on:groups={k:g for k,g in groups.items()if k in on}
  self.key_per={}
  for k,info in((k,info)for g in groups.values()for k,info in g["keys"].items()):self.key_per.setdefault(k,{}).update(info["per"])
  # rows sharing a price and a long text are copies of one product: a value every copy prints says nothing of the shelf
  long_texts=lambda lid:[v for v in(self.rows.get(lid)or{}).values()if isinstance(v,str)and len(v.split())>=8]
  sig=lambda lid:set()if self.price_of(lid)is None else{(self.price_of(lid),v)for v in long_texts(lid)}
  self.copy_cats,table=set(),[]
  for label,g in sorted(groups.items(),key=lambda kv:-kv[1]["n"]):
   sigs=list(map(sig,g["ids"]))
   firsts=sum(not any(s&o for o in sigs[:i])for i,s in enumerate(sigs))
   copies=len(sigs)>=4 and 2*firsts<=len(sigs)
   if copies:self.copy_cats.add(label)
   table+=[(label,k,info["n"],[v[:40]for v in info["values"][:4]],shown)for k,info,shown in self.group_rows(g,copies,said)]
  return table
 def same_prop(self,a,b):
  # two names hold one property when one's words nest in the other's and the listings carrying both print the same value
  a,b=_words(a),_words(b)
  if a==b:return True
  ta,tb=_tokens(a),_tokens(b)
  per=getattr(self,"key_per",None)or{}
  pa,pb=per.get(a),per.get(b)
  nested=ta and tb and(ta<=tb or tb<=ta)
  if not(nested and pa and pb):return False
  both=[l for l in pa if l in pb]
  if not both:
   va,vb=set(pa.values()),set(pb.values())
   return min(len(va),len(vb))>=2 and(va<=vb or vb<=va)
  return 10*sum(pa[l]==pb[l]for l in both)>=7*len(both)
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
  if not tool:return None
  spec=self.schemas.get(tool)or{}
  pinned=self.fixed_args.get(tool,{})
  need=[f for f in spec.get("required")or[]if f not in pinned]
  # the parameter the store's replies already fed page tokens into is not where words go
  paged=self.page_args.get(tool)
  texty=[f for f,fs in(spec.get("properties")or{}).items()if f!=paged and isinstance(fs,dict)and"enum"not in fs and"string"in _kinds(fs)]
  only=next((f for f in need if f in texty),None)
  if not only and len(texty)==1:only=texty[0]
  return only if only and all(f==only for f in need)else None
 def price_of(self,listing):
  if not self.price_key:return None
  v=(self.rows.get(listing)or{}).get(self.price_key)
  if isinstance(v,dict):v=next((x for x in v.values()if _kb_num(x)),None)
  if isinstance(v,str):
   figs=re.findall(r"\d[\d,]*(?:\.\d+)?",v)
   if len(figs)!=1:return None
   return float(figs[0].replace(",",""))
  return float(v)if _kb_num(v)else None
 def buyable(self,listing):
  if self.sold_out(listing):return False
  have=(self.rows.get(listing)or{}).get(self.stock_key)if self.stock_key else True
  return have is not False and not(type(have)in(int,float)and have<=0)
 def records_in(self,outcome):
  found,stack=[],[(0,False,outcome)]
  while stack:
   depth,listed,node=stack.pop()
   if depth>6:continue
   if isinstance(node,list):
    stack.extend((depth+1,True,sub)for sub in reversed(node))
   elif isinstance(node,dict):
    lid=self.ident(node,listed,depth)
    if not lid:stack.extend((depth+1,False,sub)for sub in reversed(list(node.values())))
    else:found.append((lid,node))
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
  spec=self.schemas.get(tool)or{}
  props=spec.get("properties")or{}
  # a second tool replying with the same keys and row keys takes the first reading when its page parameter is settled too:
  # the reading names one of its parameters, or it has at most one free slot (page_arg takes that one); values are checked below
  layout=_dump({k:0 if k not in arrays else sorted({f for r in v[:3]for f in r})for k,v in outcome.items()})
  said=self.shape_said.get(layout)or{}
  need=spec.get("required")or[]
  slots=[a for a,fs in props.items()if a not in need and _kb_page_fit(fs,None)]
  if said.get("next_arg")in props or said and len(slots)<=1:got=said
  else:
   sample={k:v[:3]if k in arrays else v for k,v in outcome.items()}
   got=self.ask_json("%s\n\nTOOL %s, parameters %s\n\nFIRST ROWS OF ITS RESULT:\n%s"%(SHAPE,tool,_dump(spec),_dump(_clip(sample,120))[:4000]),[LIGHT,HEAVY],1500)or{}
  if got:self.shape_said[layout]=got
  key=got.get("rows")
  if not key or key not in arrays:return None
  first=outcome[key][0]
  shape={"rows":key}
  for role,pool in(("count",outcome),("next",outcome),("next_arg",props),("price",first),("stock",first)):
   v=got.get(role)
   shape[role]=v if isinstance(v,str)and v in pool else None
  whole=lambda x:isinstance(x,int)and not isinstance(x,bool)
  count=shape["count"]
  if not(count and whole(outcome[count])):
   ints=[k for k,v in outcome.items()if whole(v)]
   count=ints[0]if len(ints)==1 else None
  shape["count"]=count
  self.shapes[tool]=shape
  if not self.price_key:self.price_key=shape["price"]
  if not self.stock_key:self.stock_key=shape["stock"]
  self.trace.append({"kind":"shape","tool":tool,"shape":shape})
  return shape
 def paging(self,tool,outcome,call_args=None):
  if isinstance(outcome,dict)and all(not isinstance(v,list)for v in outcome.values()):
   # a store reply wrapped one level down beside a side message still pages
   inner=[v for v in outcome.values()if isinstance(v,dict)and any(isinstance(x,list)for x in v.values())]
   if inner:outcome=inner[0]
  shape=self.read_shape(tool,outcome)
  if not(shape and shape["count"]and shape["next"]and isinstance(outcome,dict)):return None
  rows,total,nxt=(outcome.get(shape[k])for k in("rows","count","next"))
  whole=lambda x:isinstance(x,int)and not isinstance(x,bool)
  if not(isinstance(rows,list)and whole(total)):return None
  self.learn_start(tool,call_args,len(rows),total,nxt)
  if not rows:return None
  arg=self.page_arg(tool,nxt,call_args)if nxt is None or isinstance(nxt,str)or whole(nxt)else None
  # the store's own next value is a handed-out start whether or not its slot is known yet
  if nxt is not None:self.tokens_seen.add(_dump(nxt))
  return(total,len(rows),arg,nxt)if arg else None
 def learn_start(self,tool,call_args,shown,total,nxt):
  # an optional whole-number argument the store never handed out is the row its list starts at when the reply's own numbers
  # say so: the next value is that number plus the rows shown, the rows shown are all that is left past it, or nothing is left
  spec=self.schemas.get(tool)or{}
  props=spec.get("properties")or{}
  if self.page_args.get(tool)in props:return
  need={*(spec.get("required")or[]),(self.brief.get("browse_args")or{}).get("cap"),self.cap_keys.get(tool)}
  pick=(self.shapes.get(tool)or{}).get("next_arg")
  whole=lambda x:isinstance(x,int)and not isinstance(x,bool)
  for k,v in(call_args or{}).items():
   if k in need or pick in props and pick!=k or not whole(v)or v<=0 or _dump(v)in self.tokens_seen or(tool,k)in self.page_bad:continue
   if"integer"not in _kinds(props.get(k)or{}):continue
   if whole(nxt)and nxt==v+shown:kind="past"
   elif nxt is None and 0<shown==total-v:kind="rest"
   elif shown==0 and 0<total<=v:kind="none"
   else:continue
   self.page_args[tool]=k
   self.start_learned=(tool,k,kind,_dump(call_args))
   self.trace.append({"kind":"start_slot","tool":tool,"arg":k,"seen":kind})
   return
 def code_query(self,args):
  # a search naming a model code (letters joined to 3+ digits); page tokens the store handed out do not count
  return any(isinstance(v,str)and _dump(v)not in self.tokens_seen and _kb_CODE.search(v)for v in(args or{}).values())
 def log_search(self,tool,call_args,outcome):
  # reply-level strings (page tokens) are never vetted as filter values
  if isinstance(outcome,dict):self.tokens_seen.update(_dump(v)for v in outcome.values()if isinstance(v,str))
  got=self.paging(tool,outcome,call_args)
  if not got:return
  total,_,arg,token=got
  rest={k:v for k,v in call_args.items()if k!=arg}
  rec=self.search_log.setdefault(tool+_dump(rest),{"tool":tool,"args":rest,"page_arg":arg,"total":total,"ids":[]})
  rec["token"]=token
  known=rec["ids"]
  known.extend([lid for lid,_ in self.records_in(outcome)if lid not in known])
 def sweep_key(self,tool,cargs):
  # a list of a shelf with no words and no price cap, keyed without its page argument, values compared case-blind
  page=self.page_args.get(tool)or(self.shapes.get(tool)or{}).get("next_arg")
  tf=self.text_fields.get(tool)
  cap=(self.brief.get("browse_args")or{}).get("cap")
  if not tf or str(cargs.get(tf)or"").strip()or cap and cap in cargs:return None,page
  kept={k:str(v).strip().lower()for k,v in sorted(cargs.items())if k!=page}
  return tool+_dump(kept),page
 def note_sweep(self,tool,cargs,outcome):
  key,_=self.sweep_key(tool,cargs)
  if isinstance(outcome,dict)and all(not isinstance(v,list)for v in outcome.values()):
   inner=[v for v in outcome.values()if isinstance(v,dict)and any(isinstance(x,list)for x in v.values())]
   if inner:outcome=inner[0]
  counted=(self.read_shape(tool,outcome)or{}).get("count")
  total=outcome.get(counted)if counted and isinstance(outcome,dict)else None
  rows=len(self.records_in(outcome))
  if not(key and isinstance(total,int)and not isinstance(total,bool)and rows):return
  was=self.sweeps.get(key)or{"seen":0,"each":0}
  self.sweeps[key]={**was,"total":total,"each":max(was["each"],rows),"seen":was["seen"]+rows}
 def cap_note(self,tool,cargs,outcome):
  # a top price sent with a search whose reply still lists dearer rows is not applied under that search's settings: said once
  # per setting, with the settings seen to apply it
  cap=(self.brief.get("browse_args")or{}).get("cap")or self.cap_keys.get(tool)
  if not(cap and cap in cargs)and _kb_num(self.budget)and self.budget>0:
   # a number argument first sent at the very budget the request states is that search's top price
   cap=next((k for k,v in cargs.items()if _kb_num(v)and v==self.budget),cap)
   if cap in cargs:self.cap_keys[tool]=cap
  top=cargs.get(cap)if cap else None
  if not _kb_num(top)or top<=0:return""
  props=(self.schemas.get(tool)or{}).get("properties")or{}
  modes=sorted(k for k in props if"enum"in(props.get(k)or{}))
  how=tuple((k,str(cargs[k]))for k in modes if k in cargs)
  prices=[p for p in map(self.price_of,(pid for pid,_ in self.records_in(outcome)))if p is not None]
  if not prices:return""
  over=len([p for p in prices if p>top])
  kept=self.cap_modes.setdefault(tool,{})
  kept[how]=kept.get(how,True)and over==0
  if not over or(tool,how)in self.cap_said:return""
  self.cap_said.add((tool,how))
  fmt=lambda h:", ".join("%s=%s"%kv for kv in h)
  good=[fmt(h)for h,ok in kept.items()if ok and h and h!=how]
  if good:tip=" It held under %s."%" or ".join(good[:2])
  elif modes:tip=" Other settings of %s may apply it."%", ".join(modes)
  else:tip=""
  return"\n[Kyo] %d of the %d rows shown are priced above %s=%s: with %s this store ignores that ceiling, so the rows are not limited to ones under it.%s"%(over,len(prices),cap,_dump(top),fmt(how)or"these settings",tip)
 def shelf_sweep(self,tool,cargs):
  # paging on through such a list with more than four pages to go seldom reaches a fitting row: it comes in store order, so a
  # word search with the item's stated values gets there sooner (said once per list; a second ask goes through)
  key,page=self.sweep_key(tool,cargs)
  rec=self.sweeps.get(key)if key else None
  if not rec or rec.get("held")or cargs.get(page)in(None,0,1,"")or not rec["each"]:return""
  left=-(-(rec["total"]-rec["seen"])//rec["each"])
  if left<=4:return""
  rec["held"]=True
  return"%s rows in the store's own order, roughly %s pages still to come; searching words of the item's stated values finds the fitting rows faster than reading every page."%(rec["total"],left)
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
  if how not in("cheapest","most","least"):return""
  info=self.paging(tool,outcome,call_args)
  if not info:return""
  total,seen,arg,nxt=info
  at=call_args.get(arg)
  if isinstance(at,int)and not isinstance(at,bool):deep=at not in(0,1)
  else:deep=bool(str(at or"").strip())
  if nxt is None or deep or total<=seen:return""
  if how=="cheapest"and self.price_sorted(outcome):return""
  need=-(-(total-seen)//seen)
  # a page argument with a schema ceiling caps the pages the store will ever serve, whatever the count claims
  top=(((self.schemas.get(tool)or{}).get("properties")or{}).get(arg)or{}).get("maximum")
  if isinstance(top,int)and isinstance(nxt,int)and not isinstance(nxt,bool):
   if nxt>top:return""
   need=min(need,top-nxt+1)
  if need>4 or self.pages_read+need>AUTO_PAGES or self.left()-self.tail()-need<12:return("\n[Kyo] The store reports %s matches yet shows only %s, unsorted by price, and the pick must weigh every fitting listing: narrow it with a search on the requirement's value, or fetch the next page with %s=%s."%(total,seen,arg,_dump(nxt)))
  got=[]
  known={lid for lid,_ in self.records_in(outcome)}
  while len(got)<need:
   answer,sent=self._kd_send(tool,{**call_args,arg:nxt})
   if answer is None or self.finished:break
   self.pages_read+=1
   fresh=self._kd_take(tool,sent,answer)
   new={lid for lid,_ in self.records_in(fresh)}-known
   if not new:
    self.page_dead(tool,arg)
    break
   known.update(new)
   self.paged_obs.append(fresh)
   text=self.show(fresh)
   self.read_cache[tool+_dump(sent)]=(self.version,self.step,text)
   got.append("%s=%s: %s"%(arg,_dump(nxt),text))
   after=self.paging(tool,fresh,sent)
   nxt=after[3]if after else None
   if nxt is None:break
  self.trace.append({"kind":"pages","tool":tool,"read":len(got)})
  return"\n[Kyo pulled the next pages of this result too (one step per page)]\n"+"\n".join(got)if got else""
 def glued_numbers(self,targs):
  # a number written with its unit or a range sign matches only pages that print it the same way
  glued=(w for v in targs.values()if isinstance(v,str)for w in v.split()if re.search(r"\d",w)and re.search(r"\d[^\d\s.,]|[^\d\s.,]\d",w))
  seen=self.__dict__.setdefault("glued_told",set())
  fresh=[]
  for w in dict.fromkeys(glued):
   if w.lower()in seen:continue
   seen.add(w.lower())
   fresh.append(w)
  if not fresh:return""
  bare=[" ".join(re.findall(r"\d+(?:\.\d+)?",w))for w in fresh]
  return"\n[Kyo] %s finds only listings printing that exact form; other pages put spaces in numbers or use other unit spellings, so also search the plain numbers (%s) before treating this as every fit."%(", ".join(fresh),", ".join(bare))
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
  if not(tool and la in props and ca in props and label):return None
  # a browse on the search tool itself goes in its other mode, when the mode enum leaves just one
  extra={}
  for f,v in self.mode_pins.get(tool,{}).items():
   rest=[o for o in(props.get(f)or{}).get("enum")or[]if o!=v]
   if len(rest)==1:extra[f]=rest[0]
  return tool,{la:label,ca:cap,**extra}
 def sweep_pages(self,todo,per):
  # per cart line: page on the live search holding most rows of its kind, one listing the line itself first
  for lid,_,kind in todo:
   hits=lambda rec:sum(self.kind_of(r)==kind for r in rec["ids"])
   live=[rec for rec in self.search_log.values()if rec.get("token")is not None and hits(rec)]
   if not live:continue
   rec=min(live,key=lambda rec:(lid not in rec["ids"],-hits(rec)))
   for _ in range(per):
    if rec.get("token")is None or self.finished or not self.page_on(rec):break
 def sweep(self):
  """Cheapest rule: page on and read cheaper rows of each cart line's kind before checkout."""
  spare=self.left()-self.tail()-6
  if self.swept or self.brief.get("pick")!="cheapest"or spare<3 or self.secs()<CLOCK_EXTRAS:return""
  todo=[(lid,p,k)for lid,p,k in((lid,self.price_of(lid),self.kind_of(lid))for lid in self.cart_ids())if p is not None and k]
  if not todo:return""
  self.swept=True
  start,n=len(self.rows),len(todo)
  per=max(1,min(SWEEP_PAGES,spare//(2*n)))
  if spare<6*n and self.browse_call("-",1):per=0
  self.sweep_pages(todo,per)
  opened=self.read_cheaper(todo,self.searched_rows(self.items))
  room=max(1,min(CEILING_PAGES,(self.left()-self.tail()-6)//(2*n)))
  for line in todo:
   lid,price,_=line
   tag=str((self.rows.get(lid)or{}).get(self.category_key())or"")
   call=self.browse_call(tag,int(price)-1 if float(price).is_integer()else price)
   if call and not self.finished:self.walk_browse(call[0],call[1],room)
   opened+=self.read_cheaper([line],self.value_shown())or self.read_cheaper([line])
  self.trace.append({"kind":"sweep","new_rows":len(self.rows)-start,"read":len(opened)})
  kept=[c for c in opened if not self.range_miss(c)]
  if not kept:return""
  # long free text is cut first so the attributes stay inside the shown span
  body="\n".join("%s: %s"%(c,_dump(_clip(self.full.get(c)or self.rows.get(c)or{},160))[:1500])for c in kept[:12])
  return"Held, and no step was used. Kyo opened the listings below while looking for a lower price; each costs less than one of your cart lines. Go through them from the lowest price up, testing every requirement: the first that fits replaces its cart line; if none fits, just check out.\n"+body
 def read_cheaper(self,lines,among=None):
  found={}
  for _,cap,kind in lines:
   for c in self.seen_order:
    if c in self.full or c in found or among is not None and c not in among:continue
    if self.buyable(c)and self.kind_of(c)==kind and(self.price_of(c)or cap)<cap:found[c]=self.price_of(c)
  queue=sorted(found,key=found.get)[:2*READ_CAP]
  for _ in range(2):
   if not queue or self.left()<=self.tail()+5 or self.finished:break
   k=self.compare_room()
   batch,queue=queue[:k],queue[k:]
   if not self.open_rows(batch):break
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
  if not(tool and arg)or self.left()<=self.tail()+6 or self.secs()<CLOCK_EXTRAS:return
  alnum=lambda w:re.findall(r"[a-z0-9]+",w.lower())
  tw=lambda lid:list(dict.fromkeys(alnum(re.sub(r"\[.*?\]|\(.*?\)"," ",self.title_of(lid,45).lower()))))
  bags={lid:set(tw(lid))for lid in self.rows}
  spread=lambda w:(not re.search(r"\d",w),sum(w in b for b in bags.values()))
  for lid in lost[:2]:
   own=tw(lid)
   ownset=set(own)
   if any(c not in lost and self.buyable(c)and len(b.intersection(own))>=0.8*len(own)for c,b in bags.items()):continue
   query=self.value_query([v for it in self.items for v in self.word_values(it)])
   words=[w for w in query.split()if _tokens(w)and set(alnum(w))<=ownset]
   used={p for w in words for p in alnum(w)}
   rare=sorted([w for w in own if w not in used],key=spread)[:2]
   answer,sent=self._kd_send(tool,{arg:" ".join([*words,*rare])},not rare or self.finished)
   if answer is None:continue
   found=(c for c,_ in self.records_in(self._kd_take(tool,sent,answer))if c not in self.full and c not in lost and self.buyable(c))
   pool=sorted(found,key=lambda c:(-len(set(tw(c))&ownset),self.price_of(c)or 0))
   if pool and self.left()>self.tail()+5 and self.open_rows(pool[:self.compare_room()]):self.trace.append({"kind":"twins","lost":lid,"read":pool[:self.compare_room()]})
 def copies(self,lost):
  texts=lambda lid:{v for v in(self.rows.get(lid)or{}).values()if isinstance(v,str)and len(v.split())>=3 and v.strip().lower()!=self.kind_of(lid)}
  held=set().union(*map(texts,lost))
  inside=self.cart_ids()
  return[x for x in self.on_hand(set(lost))if texts(x)&held and x not in inside and self.buyable(x)and self.within_budget(x,1)]
 def proof(self,listing,pos):
  # stated musts a page prints among its attributes: one named for the property, or values holding every wanted word of
  # one alternative; a title or prose saying it does not count
  block=_fact_block(self.full.get(listing)or self.rows.get(listing)or{})or{}
  seen=_tokens(" ".join(map(str,block.values())))
  named=lambda prop:any(_names(_stem(prop),_stem(k))and str(v).strip()for k,v in block.items())
  shown=lambda want:any(_tokens(alt)and _tokens(alt)<=seen for alt in re.split(r"\s+or\s+",want))
  return{nd for prod in([self.items[pos-1]]if pos else self.items)for nd in prod.musts+prod.told if nd not in prod.dropped and(named(nd.partition(":")[0])or shown(nd.partition(":")[2]))}
 def contrary(self,lost,pick,need):
  # the incoming page fills an attribute like the one the dropped page states this must under, and not with the wanted words
  a,b=[_fact_block(self.full.get(x)or self.rows.get(x)or{})or{}for x in(lost,pick)]
  prop,_,want=need.partition(":")
  keys=[k for k,v in a.items()if _names(_stem(prop),_stem(k))or _tokens(want)and _tokens(want)<=_tokens(str(v))]
  return any(_names(_stem(k),_stem(j))and str(b[j]).strip()for k in keys for j in b)
 def fair_swap(self,lost,pick,pos,ruling=None):
  # a code swap keeps every must the dropped line prints that the new page speaks to; it then needs a price no higher under the
  # cheapest rule, more printed musts, or a breach the dropped line's own record shows (a code clash, or a quoted value only it prints)
  had,gets=self.proof(lost,pos),self.proof(pick,pos)
  # a must the incoming page leaves unprinted is no loss (real passing orders often leave a must unprinted); one it states otherwise is
  if any(self.contrary(lost,pick,nd)for nd in had-gets):return False
  was,now=self.price_of(lost),self.price_of(pick)
  if self.brief.get("pick")=="cheapest"and was is not None and now is not None and now<=was:return True
  if had<gets or pos and self.item_clash(lost,pos):return True
  own,other=_dump(self.full.get(lost)or self.rows.get(lost)or{}).lower(),_dump(self.full.get(pick)or self.rows.get(pick)or{}).lower()
  said=[str(p)for p in _items((ruling or{}).get("problems"))if lost in str(p)or not any(c in str(p)for c in self.rows)]
  quotes=[q.lower()for p in said for q in re.findall(r"(?<![A-Za-z0-9])[\"'`]([^\"'`]{2,60})[\"'`](?![A-Za-z0-9])",p)]
  return any(q in own and q not in other for q in quotes)
 def take_better(self):
  ruling=next((t.get("got")for t in reversed(self.trace)if t.get("kind")=="check"),None)or{}
  picks=[str(p)for p in _items(ruling.get("better"))]
  if self.brief.get("pick")=="cheapest"and len(_items(ruling.get("problems")))<=1:
   for pick in picks:
    cost=self.price_of(pick)
    if pick not in self.full or cost is None or not self.buyable(pick):continue
    kind=self.kind_of(pick)
    dearer=[x for x in self.cart_ids()if self.kind_of(x)==kind and(self.price_of(x)or 0)>cost]
    if len(dearer)!=1:continue
    old=dearer[0]
    pos=self.slot_of(old)
    if pos is None or self.item_clash(pick,pos)or not self.fair_swap(old,pick,pos,ruling):continue
    ids=[*(x for x in self.cart_ids()if x!=old),pick]
    self.check_cart(ids,SPEND_HARD)
    if self.verdicts.get(self.check_key(ids))and self.swap_line(old,pick,0):return True
  # a failed check naming an opened listing as better: it replaces the line of its kind once a check of the new cart approves it
  tries=2
  for pick in picks:
   if pick not in self.full or pick in self.cart_ids()or not self.buyable(pick)or self.sold_out(pick):continue
   kind=self.kind_of(pick)
   if kind:same=[x for x in self.cart_ids()if self.kind_of(x)==kind]
   else:same=self.cart_ids()if len(self.items)==1 else[]
   if len(same)!=1:continue
   old=same[0]
   pos=self.slot_of(old)
   if pos is None or self.item_clash(pick,pos)or not self.fits_total(old,pick)or not self.fair_swap(old,pick,pos,ruling):continue
   ids=[*(x for x in self.cart_ids()if x!=old),pick]
   self.check_cart(ids,SPEND_HARD)
   if self.verdicts.get(self.check_key(ids))and self.swap_line(old,pick):return True
   tries-=1
   if not tries:break
  return False
 def swap_line(self,lost,pick,spare=1):
  tool=self.roles.get("cart")
  want={lid:self.cart_lines.get(lid,1)for lid in self.cart_ids()if lid!=lost}
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
  runs=0
  for _ in range(3):
   runs+=1
   answer,sent=self._kd_send(tool,call_args)
   if answer is None or self.finished:return
   count=len(self.records_in(self._kd_take(tool,sent,answer)))
   rec=self.search_log.get(tool+_dump(sent))
   if not(count and rec)or rec.get("total",0)<=(pages+1)*count or self.left()<=self.tail()+7:break
   cap=self.brief["browse_args"]["cap"]
   v=call_args[cap]
   call_args={**call_args,cap:v//2 if isinstance(v,int)else v/2}
   upper=tool+_dump(sent)
  key=tool+_dump(sent)
  # a halved cap that fits on one page leaves the band between it and the cap above unseen: page that band instead
  if runs>=2 and(self.search_log.get(key)or{}).get("token")is None:key=upper
  for _ in range(pages):
   rec=self.search_log.get(key)or{}
   if rec.get("token")is None or self.left()<=self.tail()+7 or self.finished or not self.page_on(rec):break
 def wider_hold(self):
  if self.left()<14+2*len(self.items)or self.wider_held or self.brief.get("pick")!="cheapest":return""
  self.wider_held=True
  tags="; ".join("%s at %g"%(cid,self.price_of(cid))for cid in self.cart_ids()if self.price_of(cid)is not None)
  return("Held without spending a step (%d steps left), because the pick rule is cheapest. The search tool matches title words and returns rows unsorted by price, and required values are often printed only in attributes, so a few queries can overlook cheaper fits. Before checkout, for every item, run some more searches under other names for the product: its plain single-word name, synonyms, words from fitting titles already seen. From what comes back, consider only in-stock rows below the current cart line's price%s. Open those lowest price first; as soon as one meets every need, put it in place of that cart line, then check out."%(self.left()," (%s)"%tags if tags else""))
 def shaped_act(self,tool,raw,schema):
  faults=[]
  sent=_shape(raw,schema,"args",faults)
  return sent,(None if faults else self.act(tool,sent))
 def fill_facts(self,candidates,reserve_steps):
  tool=self.facts_tool()
  key=self.single_target(tool)
  if not(tool and key):return[]
  # a thin page stays out of the full records, so the version it was read at keeps it from being read again unchanged
  unread=[x for x in dict.fromkeys(candidates)if x in self.rows and x not in self.full and not self.sold_out(x)and self.facts_read.get(x)!=self.version]
  notes=[]
  for cid in unread:
   if self.finished or self.left()<=reserve_steps or self.facts_from.get(tool)is False:break
   _,env=self.shaped_act(tool,{key:cid},self.schemas.get(tool))
   if env is None:break
   self.facts_read[cid]=self.version
   page=env.get("observation")
   self.observe(tool,page)
   self.note_view(tool,page)
   if isinstance(page,dict)and self.ident(page)and not self.facts_from.get(tool):self.facts_from[tool]=bool(_fact_block(page))
   notes.append("%s: %s"%(cid,self.show(page)[:1500]))
  if notes:self.trace.append({"kind":"fill_facts","n":len(notes)})
  return notes
 def cheaper_rows(self):
  if self.brief.get("pick")!="cheapest":return[]
  mark="%s#%s"%(self.cart_mark(),self.version)
  if not self.category_key()or mark in self.cheaper_marks:return[]
  self.cheaper_marks.add(mark)
  price,lines=self.price_of,self.cart_ids()
  pool=[c for c in self.seen_order if c not in lines and c not in self.full and self.buyable(c)]
  chosen=[]
  for cid in lines:
   cost,sort=price(cid),self.kind_of(cid)
   if cost is None or not sort:continue
   under=sorted((c for c in pool if self.kind_of(c)==sort and price(c)is not None and price(c)<cost),key=lambda c:price(c)or 0.0)[:2]
   chosen+=[c for c in under if c not in chosen]
  return self.fill_facts(chosen[:CHEAPER_OPENS],self.tail()+4)
 def observe(self,tool,outcome):
  self.absorb(outcome)
  loose=self.loose_lines(outcome)
  said=""if not loose else self.take_note(loose)
  if said:self.queued.append("(seen on %s) %s"%(tool,said))
  return said
 def widen(self):
  tool,slot=self.roles.get("compare"),self.compare_target()
  if not(tool and slot)or self.compare_bare():return self.widen_singly()
  if self.left()<=12:return None
  schema=self.schemas.get(tool)or{}
  cap,groups=self.compare_room(),self.item_groups()
  per=max(1,cap//max(1,len(groups)))
  batch=[gid for grp in groups for gid in[g for g in grp if g not in self.full][:per]][:cap]
  if len(batch)<2:return None
  before=len(self.full)
  _,env=self.shaped_act(tool,{slot:batch},schema)
  if env is None:return None
  page=env.get("observation")
  self.observe(tool,page)
  self.note_view(tool,page)
  found=self.records_in(page)
  if found and not self.facts_from.get(tool):self.facts_from[tool]=any(_fact_block(lg)for _,lg in found)
  self.trace.append({"kind":"widen","ids":batch,"opened":len(self.full)-before})
  if found and self.facts_from.get(tool)is False:return self.widen_singly()
  if len(self.full)!=before or len(batch)<=2 or self.left()<=12:return None
  self.compare_bad=min(self.compare_bad or 999,len(batch))
  _,env=self.shaped_act(tool,{slot:batch[:2]},schema)
  if env is not None:self.observe(tool,env.get("observation"))
  return None
 def read_first(self):
  n=len(self.items)
  extra=6*n if self.brief.get("pick")=="cheapest"and self.browse_call("-",1)else 0
  floor=14+extra
  if n>1:return self.read_each(floor)
  # pages opened for pointed-to listings of other kinds do not fill the requested items' key table
  ids=lambda:[c for grp in self.item_groups()for c in grp]
  for _ in range(4):
   if len({c for c in ids()if c in self.full})>=FIRST_READS or self.left()<=floor or self.finished or self.secs()<CLOCK_EXTRAS:return
   had=len(self.full)
   if sum(c not in self.full for c in ids())>=2:self.widen()
   if len(self.full)!=had:continue
   # buyable rows still unread mean the opens were held back: another page would only pile up more unread rows
   if sum(c not in self.full and self.buyable(c)for c in ids())>=2:return
   wanted=set(ids())
   live=[r for r in self.search_log.values()if r.get("token")is not None]
   page=max(live,key=lambda r:len(wanted.intersection(r["ids"])),default=None)
   if page is None or not self.page_on(page):return
 def item_rows(self,prod):
  # seen rows of this item's kind: their category or title shares a long word with the item's name, its store category or its title-spelt noun
  bag={w for w in _stem(prod.shelf)|_stem(prod.name)|_stem(self.title_noun(prod))if len(w)>=4}
  cat=self.category_key()
  return[c for c in self.seen_order if bag&{w for w in _stem(str(self.rows[c].get(cat)or"")if cat else"")|_tokens(self.title_of(c))if len(w)>=4}]
 def read_each(self,floor):
  # several items: each reads a few of its own listings before the plan, an item with none seen gets a search first, and no more:
  # an item left unread is never asked about, and the steps saved go to the searches after the answers
  per=2 if len(self.items)!=2 else 3
  tl=self.roles.get("search")
  key=self.single_target(tl)
  for prod in self.items:
   if self.left()<=floor or self.finished or self.secs()<CLOCK_EXTRAS:return
   rows=self.item_rows(prod)
   # the first search for an item carries its stated values: in race 16 a first-page search of category words alone found a
   # paying listing 5% of the time, one with other words 28%
   q=""
   if not rows and tl and key:q=self.joined_query(prod)or self.lead_query(prod,[])
   if q and q not in self.sought:
    self.sought.add(q)
    sent,env=self.shaped_act(tl,{key:q},self.schemas.get(tl))
    if env is not None:
     page=env.get("observation")
     self.observe(tl,page)
     self.log_search(tl,sent,page)
    rows=self.item_rows(prod)
   want=[c for c in rows if c not in self.full and self.buyable(c)]
   want=want[:max(0,per-sum(c in self.full for c in rows))]
   if want and not self.finished:self.open_rows(want)
 def widen_singly(self):
  tool=self.facts_tool()
  if not(tool and self.single_target(tool)):return
  price=self.price_of
  cost=lambda c:float("inf")if price(c)is None else price(c)
  cheap=self.brief.get("pick")=="cheapest"
  lanes=[]
  for grp in self.item_groups():
   lane=[c for c in grp if c not in self.full and self.buyable(c)]
   if cheap:lane=sorted(lane,key=cost)
   lanes.append(lane[:max(0,min(2,8-sum(c in self.full for c in grp)))])
  picks=[lane[turn]for turn in(0,1)for lane in lanes if turn<len(lane)]
  self.fill_facts(picks[:5],22)
 def fuse_shelf(self):
  if self.asked or self.carted or self.cart_touched or self.lead_items or len(self.items)<2:return""
  flat=lambda v:" %s "%" ".join(re.sub(r"[^a-z0-9]+"," ",str(v).lower().replace("&"," and ")).split())
  said=flat(self.request)
  pool={v.strip()for row in self.rows.values()for v in row.values()if isinstance(v,str)and 3<len(v.strip())<=60}
  shelves={it.shelf.strip()for it in self.items if it.shelf}
  for lab in sorted(pool|shelves|self.labels(),key=len,reverse=True):
   if flat(lab)not in said:continue
   parts=[_stem(p)for p in re.split(r"\s*(?:,|&|\band\b)\s*",lab.lower())if p.strip()]
   if len(parts)<2:continue
   cover=lambda it:{n for n,p in enumerate(parts)if _stem(it.name)and _stem(it.name)<=p}
   group=[n for n,it in enumerate(self.items)if cover(it)]
   if len(group)<2:continue
   members=[self.items[n]for n in group]
   if set().union(*map(cover,members))!=set(range(len(parts)))or len({it.qty for it in members})>1:continue
   sets=[set(it.musts)for it in members]
   if all(any(not m<=big for m in sets)for big in sets):continue
   old,keep=self.items,group[0]
   one=Item(lab.lower(),lab,old[keep].qty,list(dict.fromkeys(m for it in members for m in it.musts)))
   self.items=[one if n==keep else it for n,it in enumerate(old)if n not in group[1:]]
   slot={n+1:self.items.index(one if n in group else it)+1 for n,it in enumerate(old)}
   self.refs=list(dict.fromkeys((slot.get(p,1),r)for p,r in self.refs))
   self.trace.append(dict(kind="fuse",shelf=lab,items=[it.name for it in members]))
   return"[Kyo] %r is one department in this store, so the request's %s counts as ONE item: item %d, %s, quantity %d. It takes a single listing, not a listing per word."%(lab," + ".join(it.name for it in members),keep+1,one.name,one.qty)
  return""
 def ready_to_ask(self):
  if any((self.asked,not self.roles.get("ask"),self.msgs_free()<=0,self.left()<=8,self.secs()<CLOCK_EXTRAS)):return False
  return len(self.full)>self.walked_full+3*len(self.items)or self.step>=self.walked_step+max(8,self.max_steps//4)
 def message_args(self,msg):
  schema=self.schemas.get(self.roles.get("ask")or"")or{}
  props=schema.get("properties")or{}
  need=schema.get("required")or[]
  texts=[name for name,spec in props.items()if isinstance(spec,dict)and"string"in _kinds(spec)and"enum"not in spec]
  slot=next((name for name in need if name in texts),None)or next(iter(texts),None)
  if not slot:return None
  payload={slot:msg}
  for name in need:
   if name in payload:continue
   opts=(props.get(name)or{}).get("enum")
   if not(isinstance(opts,list)and opts):return None
   payload[name]=opts[0]
  faults=[]
  shaped=_shape(payload,schema,"args",faults)
  return shaped if not faults else None
 def interview(self):
  """Plan a ranked lane of page keys per item, then ask, always serving the item with no need found yet.

  A one-item order opens with one key and then lists two. On a larger order an item's open keys are spread over the messages it
  can still get: one key when the messages cover them all, else the next few keys together, up to ASK_SPAN. When the request
  states the message cap and the items outnumber the free messages, even runs of the planner's order first share a message, two
  keys per item, so no item goes unasked; a lone item in its run gets its own list. A reply that objects to the many topics turns
  the rest of the interview to one key per message, items not yet asked first; a merely puzzled reply does not, as single
  questions puzzle the shopper as often. A list that draws nothing gives the item's next list a key not yet offered as its lead,
  followed once more by the silent list's other keys: a reader can answer the start of a list and pass over the rest. A single
  key of several words that draws the objection is asked again by its last word alone. A message over the tool's length limit
  sheds its last keys instead of being cut.
  """
  self.asked=True
  # several keys in one message reach more needs: a one-item order still opens with one key (strict shoppers take a single best)
  # and lists at most two after it (real race 17: a single then lists drew .76 of one-item needs, singles alone .71)
  one,lists=len(self.items)==1,True
  room=self.msgs_free()
  if room<=0:return""
  self.widen()
  self.read_first()
  self.use_wishes()
  lanes,order=self.lane_plan(room,self.key_table())
  heard,spent={n:0 for n in lanes},{n:0 for n in lanes}
  tried,relisted,again={n:set()for n in lanes},{n:set()for n in lanes},{n:[]for n in lanes}
  sent,new_reqs=[],[]
  # one reading per reply: the light reader keeps the strong one's budget for the cart check and the review
  models=(LIGHT,HEAVY)
  top=self.ask_room()
  # items outnumbering the free messages, on a stated cap: runs of the planner's order share a message (real race 17, three or
  # more items: grouped asks drew every need in .28 of episodes, one item per message .17); an unstated cap groups nothing
  ranked=[n for n in order if n in lanes]+[n for n in lanes if n not in order]
  free=self.msgs_free()
  if self.cap_stated and 0<free<len(ranked):
   cuts=[len(ranked)//free+(1 if i<len(ranked)%free else 0)for i in range(free)]
   runs=[ranked[sum(cuts[:i]):sum(cuts[:i+1])]for i in range(free)]
   for run in[r for r in runs if len(r)>1]:
    if self.finished or self.msgs_free()<=0 or self.left()<=6:break
    parts=[(n,[k for k in lanes[n]if _words(k)not in tried[n]and not self.told_dup((n,k))][:2])for n in run]
    line,parts=self.grouped([(n,ks)for n,ks in parts if ks],top)
    # a group that cannot fit the length limit even at one key per item is not sent: the items go one per message instead
    if len(parts)<2 or top and len(line)>top:break
    got=self.send([(" and ".join(str(n)for n,_ in parts),parts[0][1][0],line)])
    if not got:break
    sent+=got
    keep={n:set(tried[n])for n,_ in parts}
    for n,ks in parts:
     spent[n]+=1
     tried[n].update(map(_words,ks))
    before=[len(p.told)for p in self.items]
    verdict=self.hear(got,models)
    if 1 in{_count(q)for q in _items(verdict.get("crowded"))}:
     # refused as too many topics: the keys go back to their lanes and singles follow, items never asked first
     lists=False
     tried.update(keep)
    self.take_heard(verdict,new_reqs)
    for num in[n for n,p in enumerate(self.items,1)if len(p.told)>before[n-1]]:heard[num]=heard.get(num,0)+1
    if not lists:break
  while not self.finished and self.msgs_free()>0 and self.left()>6:
   sel=self.next_in_lane(lanes,heard,spent,tried,order)
   if sel is None:break
   pos,first=sel
   kept=set(tried[pos])
   names=self.pick_names(pos,first,lanes,heard,kept,again[pos],lists,(2 if spent[pos]else 1)if one else ASK_SPAN)
   while len(names)>1 and top and len(self.question(pos,names))>top:names.pop()
   relisted[pos].update(_words(k)for k in names if k in again[pos])
   again[pos]=[]
   tried[pos].update(map(_words,names))
   got=self.send([(pos,names[0],self.question(pos,names))])
   if not got:break
   sent+=got
   spent[pos]+=1
   before=[len(p.told)for p in self.items]
   verdict=self.hear(got,models)
   crowded=1 in{_count(q)for q in _items(verdict.get("crowded"))}
   several=len(names)>1
   if crowded and several:
    # the shopper wants one topic at a time: these keys go back to the lane and are asked singly from now on
    lists,tried[pos]=False,kept
   elif crowded and len(_words(names[0]).split())>1:
    # one key of several words heard as several topics: its last word, a property name on its own, is asked next
    last=_words(names[0]).split()[-1]
    fixed=[m.split(":")[0]for m in self.items[pos-1].musts]
    if last not in tried[pos]and all(not _names(_tokens(f),_tokens(last))for f in fixed):lanes[pos]=[last,*(k for k in lanes[pos]if k!=last)]
   self.take_heard(verdict,new_reqs)
   if lists and several and len(self.items[pos-1].told)==before[pos-1]:again[pos]=[k for k in names[1:]if _words(k)not in relisted[pos]]
   for num in[n for n,p in enumerate(self.items,1)if len(p.told)>before[n-1]]:heard[num]=heard.get(num,0)+1
  return self.interview_report(sent,new_reqs)
 def pick_names(self,pos,first,lanes,heard,kept,again,lists,most=ASK_SPAN):
  """The keys one message names, at most `most`: unoffered ones first, a key nested in one already listed or asked held back."""
  ahead=[k for k in lanes[pos]if _words(k)not in kept and not self.told_dup((pos,k))]
  idle=len([n for n in lanes if not heard.get(n)])or 1
  share=max(1,self.msgs_free()//(idle if not heard.get(pos)else 1))
  span=min(most,-(-(len(ahead)+len(again))//share))
  # keys never offered fill a list before the silent list's keys come round again, and a key nested in one already listed or
  # asked waits: an undrawn need is most often one never named, and slots further down a list draw less
  fresh=[k for k in ahead if not any(_nested_keys(k,t)for t in kept)]
  names=[]
  for k in(fresh+again+[k for k in ahead if k not in fresh]if lists else ahead[:1]):
   if len(names)>=span:break
   if all(not _nested_keys(k,o)for o in names):names.append(k)
  return names or[first]
 def next_in_lane(self,lanes,heard,spent,tried,order):
  # the item still without a found need goes first, then the one asked least, then the planner's order
  rank={n:i for i,n in enumerate(order)}
  for pos in sorted(lanes,key=lambda n:(heard.get(n,0)>0,spent.get(n,0),rank.get(n,len(rank)+n))):
   for label in lanes[pos]:
    if _words(label)in tried[pos]or self.told_dup((pos,label)):continue
    return(pos,label)
  return None
 def mirror_keys(self,per):
  out=set()
  for key,vals in per.items():
   hits=[[o for o in per if o!=key and per[o].get(lid)==v and not self.same_prop(key,o)]for lid,v in vals.items()]
   partners={o for h in hits for o in h}
   # a key that on most listings repeats some other key's value, a different key each time, only lists variants
   if len(vals)>=3 and 2*sum(map(bool,hits))>=len(vals)and len(partners)>=2:out.add(key)
  return out
 def key_shares(self,table):
  # per item: each listed key's share of that item's opened listings, how evenly its values split the item's buyable in-budget pages, the keys the request already spells a printed value of, plus keys that only mirror another key's value
  size=len(self.items)
  pages={label:g["ids"]for label,g in self.key_groups().items()}
  own=lambda pos:_stem(self.items[pos-1].name)|_stem(self.items[pos-1].shelf)
  per=getattr(self,"key_per",None)or{}
  run=lambda t:" %s "%" ".join(_kb_PLURAL.sub(r"\1",w)for w in _kb_norm(t).split())
  said=run(self.request)
  pointed={r for _,r in self.refs}
  fixed=run(" ; ".join(m for it in self.items for m in it.musts if"only if"in m or m in pointed))
  # a single item's request spelling out a value its pages print has answered that key; one bare word may come from the story, so it takes two words or a figure joined to letters, and words of a condition or of a listing pointed to do not count
  shaped=lambda v:len(v.split())>=2 or(re.search(r"\d",v)and re.search(r"[a-z]",v))
  told=lambda v:size==1 and shaped(v)and v in said and v not in fixed
  shares,split,spelt={},{},{}
  for pos in range(1,size+1):
   mark=own(pos)
   rows=[r for r in table if size==1 or _tokens(r[0])&mark]
   if size>1 and not rows:rows=[r for r in table if all(not _tokens(r[0])&own(o)for o in range(1,size+1))]
   got=shares[pos]={}
   for label,key,hits,_,_ in rows:
    whole=len(pages[label])if label in pages else hits
    got[key]=max(got.get(key,0.0),hits/max(1,whole))
   mine={lid for r in rows for lid in pages.get(r[0],[])}
   qty=self.items[pos-1].qty
   live={lid for lid in mine if self.buyable(lid)and self.within_budget(lid,qty)}
   split[pos]={}
   for k in got:
    # split: live pages printing the key minus those printing its most common value
    tally=[_words(v)for lid,v in(per.get(k)or{}).items()if lid in live and v]
    split[pos][k]=len(tally)-max((tally.count(t)for t in tally),default=0)
   spelt[pos]={k for k in got if any(told(run(v))for lid,v in(per.get(k)or{}).items()if lid in mine)}
  return shares,self.mirror_keys(per),split,spelt
 def lane_prompt(self,room,table):
  count=min(max(room,2),5)
  fixed=["item %d (%s); fixed by the request: %s"%(pos,prod.name,"; ".join(prod.musts)or"nothing")for pos,prod in enumerate(self.items,1)]
  parts=[LANES.replace("COUNT",str(count)),"\n\nREQUEST:\n",self.request,"\n\nITEMS:\n","\n".join(fixed)]
  checks=[str(v)for v in _items(self.brief.get("ask_first"))if str(v).strip()]
  if checks:parts.extend(("\n\nASK FIRST:\n","\n".join(checks)))
  shown,last=[],None
  for pos,(cat,key,hits,vals,kinds)in enumerate(table,1):
   if cat!=last:
    last=cat
    copied=cat in self.copy_cats
    shown.append("category %r%s:"%(cat," (many listings opened here duplicate one another, so a key is not pushed down just because they all share its value: order these keys by how a buyer chooses this kind of product and by how many listings show them)"if copied else""))
   shown.append("  %s. %s (on %s, %s value%s): %s"%(pos,key,hits,kinds,"s"if kinds!=1 else""," | ".join(vals)))
  parts.extend(("\n\nATTRIBUTE KEYS SEEN:\n","\n".join(shown)or"(nothing opened in full so far)"))
  return"".join(parts)
 def lane_picks(self,reply,shares):
  """The planner's picks per item in the store's spelling, and the items whose picks rest on the request's words first."""
  size=len(self.items)
  lanes,order={pos:[]for pos in range(1,size+1)},[]
  plain=lambda t:" ".join(w.rstrip("s")for w in _words(t).split())
  for ej in _items(reply.get("ask")):
   if not isinstance(ej,dict):continue
   pos=min(max(1,_count(ej.get("item")or 1)or 1),size)
   head=re.split(r"\s+(?:and|or)\s+|[/,;(]",str(ej.get("label")or""))[0].strip().rstrip("?.")
   if not head or len(head.split())>5:continue
   key=next((k for k in shares[pos]if plain(k)==plain(head)),_words(head))
   if any(self.same_prop(key,o)for o in lanes[pos]):continue
   lanes[pos].append(key)
   if pos not in order and str(ej.get("hint")or"").strip():order.append(pos)
  return lanes,order+[pos for pos in lanes if pos not in order]
 def lane_plan(self,room,table):
  """Ask the planner for each item's ranked keys, map them to the store's spellings, guard each lane with its widely printed keys."""
  size=len(self.items)
  reply=self.ask_json(self.lane_prompt(room,table),[HEAVY,LIGHT],3000)or{}
  shares,mirror,split,spelt=self.key_shares(table)
  lanes,order=self.lane_picks(reply,shares)
  flat={key for _,key,hits,_,kinds in table if kinds<=1 and hits>=3}
  top=min(max(room,1),3)
  for pos,prod in enumerate(self.items,1):
   share=shares[pos]
   settled=[m.split(":")[0]for m in prod.musts]
   free=lambda k:not any(self.same_prop(k,s)or _names(_tokens(s),_tokens(k))for s in settled)
   # equal shares go to the key splitting the buyable in-budget pages more evenly; a key the request already spells a value of joins the fill after the open ones and guards nothing
   rank=lambda k:(-share[k],-split[pos].get(k,0))
   fill=[k for k in share if k not in mirror and free(k)and k not in flat]
   ranked=sorted([k for k in fill if k not in spelt[pos]],key=rank)
   lane=[k for k in lanes[pos]if all(not self.same_prop(k,s)for s in settled)]
   # a widely printed key takes the slot of a planner pick printed on far fewer listings, a story-led pick included
   for guard in[k for k in ranked if share[k]>=0.5][:2]:
    near=lambda o:self.same_prop(guard,o)
    if any(map(near,lane[:top])):continue
    weak=[i for i,o in enumerate(lane[:top])if share.get(o,0.0)+0.34<=share[guard]]
    if weak:
     lane=[o for o in lane if not near(o)]
     at=weak[-1]
     moved,lane[at]=lane[at],guard
     lane.insert(top,moved)
    elif len(lane)<top:lane.append(guard)
   # on a several-item order a key that mostly repeats other keys' values still joins the list, after the open keys: it can be the wish
   echoes=sorted((k for k in mirror&set(share)if k not in flat and free(k)),key=rank)if size>1 else[]
   for k in[*ranked,*sorted(spelt[pos].intersection(fill),key=rank),*echoes,*sorted(flat&set(share),key=lambda k:-share[k])]:
    if all(not self.same_prop(k,o)for o in lane):lane.append(k)
   # a lone item's first question goes to a key most of its opened listings print: a rarely printed first key seldom finds the wish
   lead=next((k for k in lane if share.get(k,0.0)>=0.5),None)
   if size==1 and lead and share.get(lane[0],0.0)<0.5:lane=[lead,*(k for k in lane if k!=lead)]
   lanes[pos]=lane
  self.trace.append({"kind":"plan","ask":_clip(reply.get("ask"),120),"lanes":{p:v[:6]for p,v in lanes.items()}})
  return lanes,order
 def use_wishes(self):
  # a word named with the product that the item's own listings print as a value is a tie-breaker, never a breach
  wish=_items(self.brief.get("wishes"))
  for pos,prod in enumerate(self.items,1):
   own=_tokens(prod.name)-_tokens(prod.shelf)-_tokens(" ".join(prod.musts))
   for k,v in((k,v)for lid in self.seen_order for k,v in(_fact_block(self.full.get(lid)or{})or{}).items()):
    if not(isinstance(v,str)and len(_tokens(v))<=3 and own&_tokens(v)):continue
    line="item %s: %s: %s"%(pos,_words(k),v.strip()[:40])
    if line not in wish and len(wish)<8:wish.append(line)
  self.brief["wishes"]=wish
 def word_values(self,prod):
  skip=set(pair[1]for pair in self.refs)-set(self.ref_words)
  said=_tokens(self.request)
  for nd in[n for n in prod.told+prod.musts if n not in skip and n not in prod.dropped]:
   pointed=nd in self.ref_words
   text=self.ref_words.get(nd)or nd.split(":",1)[-1].strip()
   head=nd.split(":",1)[0].strip()
   lone=len(_tokens(text))==1 and not re.search(r"\d",text)
   if nd in prod.musts and not pointed and":"in nd and lone and not _tokens(text)<=said and _tokens(head)<=said and not self.titled(text):text=head
   bits=text.split()
   if nd in prod.told and not pointed and len(bits)>1 and not re.search(r"\d",text):
    seen=self.fact_words()
    if not _tokens(text)<=seen:
     # a reply's need can wrap its value in the reply's own phrasing: keep the closing words that opened records print
     ends=(" ".join(bits[-k:])for k in range(min(3,len(bits)-1),0,-1))
     text=next((t for t in ends if _tokens(t)<=seen),text)
   if text and len(text.split())<=3 and(pointed or not re.search(r"\d",text)or self.titled(text)):yield text
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
  for w in(w for c in self.rows for w in set(re.findall(r"[a-z]+",self.title_of(c).lower()))):
   if w[:4]==noun[:4]and len(w)<=len(noun)+1:tally[w]=tally.get(w,0)+1
  if not tally:return noun
  return max(sorted(tally),key=tally.get)
 def lead_query(self,prod,bits,nth=0):
  # the product word and the nth rarest value word (fewest seen rows print it) lead, so a store reading only the first words still gets both
  rows=[_tokens(_dump(r))for r in self.rows.values()]
  hits=lambda b:sum(_tokens(b)<=r for r in rows)
  plain=sorted(dict.fromkeys(b for b in bits if len(b)>1 and not re.search(r"\d",b)),key=hits)
  if nth and nth>=len(plain):return""
  rare=plain[nth]if plain else""
  pool=[self.title_noun(prod),rare,*bits,*prod.name.split()]
  words=list(dict.fromkeys(w for w in pool if w and not re.search(r"\d",w)))
  # a figure goes in only joined to letters (a unit or code such as 12v): in race 16 a search holding a bare number came back
  # empty 43% of the time and found a paying listing 6% (no figure: 8% and 32%)
  figs=[w for b in bits for w in b.split()if _lettered_figure(w)]
  # three words stay findable whether a store needs every word, any word or only the leading ones; a figure never
  # takes one of the first two places, where a store reading only the leading words would match it exactly
  head,rest=words[:2],words[2:]
  return" ".join(" ".join(dict.fromkeys(head+figs+rest)).split()[:QUERY_WORDS])
 def meets_limits(self,listing):
  facts=_tokens(_dump(_fact_block(self.full.get(listing)or{})or{}))
  wants=[[_tokens(" ".join(re.findall(r"\d+",text)))for text in self.limit_values(prod)]for prod in self.items]
  return any(all(part<=facts for part in sets)for sets in wants)
 @staticmethod
 def need_clash(nd,block,told,aimed):
  prop,_,want=nd.partition(":")
  if"only if"in want:return None
  near=[(k,str(v))for k,v in block.items()if _names(_stem(prop),_stem(k))and str(v).strip()]
  # a field named exactly like the property decides alone: a sibling field printing the wanted value does not rescue a clash there
  named=[v for k,v in near if _stem(k)==_stem(prop)]or[v for _,v in near]
  span=re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*",want)
  if span:
   lo,hi,unit=float(span[1]),float(span[2]),span[3].lower()
   shown=[t for t in named if re.search(r"\d",t)and unit in t.lower()]
   inside=lambda t:any(lo<=float(m)<=hi for m in re.findall(r"\d+(?:\.\d+)?",t))
   if shown and not any(map(inside,shown)):return nd,shown[0]
  if not named or nd in aimed:return None
  wt=_tokens(want)
  split=[t for t in named if wt<=_tokens(t)]
  if split:
   # a told value of several words, none a bare figure, that a page prints only with another word wedged inside names something else
   if len(wt)>=2 and all(not w.isdigit()for w in wt)and all(not _side(want,t)for t in named):return nd,split[0],"apart"
   # one value asked for is not met by a printed mix of several parts given in percent
   if"%"not in want and all(len(re.findall(r"\d+(?:\.\d+)?\s*%",t))>=2 for t in split):return nd,split[0]
  bare=r"(?<![A-Za-z0-9.])\d"
  if re.search(bare,want):return None if any(re.search(bare,t)for t in named)else(nd,named[0])
  return(nd,named[0])if not split and(nd in told or len(want.split())==1)else None
 def range_miss(self,listing):
  block=_fact_block(self.full.get(listing)or{})or{}
  told,aimed=set(nd for prod in self.items for nd in prod.told),set(pair[1]for pair in self.refs)
  hits=(self.need_clash(nd,block,told,aimed)or(nd not in aimed and(self.unit_miss(nd,block)or self.word_miss(nd,block)))for _,nd in self.needs_now())
  return next((hit for hit in hits if hit),None)
 def unit_words(self):
  # words this store prints right beside a figure, in attributes or titles, on two listings or more
  size=(len(self.full),len(self.rows))
  if getattr(self,"units_at",None)==size:return self.units
  seen={}
  for lid in set(self.full)|set(self.rows):
   block=_fact_block(self.full.get(lid)or{})or{}
   for t in map(_figure_words,[*map(str,block.values()),self.title_of(lid)]):
    for i,w in enumerate(t):
     beside=t[max(0,i-1):i]+t[i+1:i+2]
     if w.isalpha()and len(w)>=2 and not all(x.isalpha()for x in beside):seen.setdefault(w,set()).add(lid)
  self.units_at=size
  self.units={k for k,v in seen.items()if len(v)>=2}
  return self.units
 @staticmethod
 def word_miss(nd,block):
  # a stated value of several words that the attribute named like the property shares no word with: a different value
  # (a single word, a told value, a figure, an exclusion or a condition is judged by need_clash or left to the reading)
  prop,_,want=nd.partition(":")
  words=_stem(want)
  if len(words)<2 or"only if"in want or"not"in words or re.search(r"\d",want):return None
  same=lambda k:_nest(_stem(prop),_stem(k))and _nest(_stem(k),_stem(prop))
  exact=[str(v)for k,v in block.items()if same(k)and str(v).strip()]
  if not exact:return None
  near=lambda a,b:_akin(a,b)or len(min(a,b,key=len))>=4 and(a.startswith(b)or b.startswith(a))
  shares=lambda v:any(near(w,o)for w in words for o in _stem(v))
  return None if any(map(shares,exact))else(nd,exact[0])
 def unit_miss(self,nd,block):
  # the stated figure printed with another unit word beside it and without the stated one (a size in another system):
  # the page shows the number but measures it differently
  prop,_,want=nd.partition(":")
  w,units=_figure_words(want),self.unit_words()
  beside=lambda t,i:[x for x in t[max(0,i-1):i]+t[i+1:i+2]if x in units]
  for v in(v for k,v in block.items()if _names(_stem(prop),_stem(k))):
   t=_figure_words(v)
   shown=set(t)
   marked={f for i,f in enumerate(t)if not f.isalpha()and beside(t,i)}
   for j,x in enumerate(w):
    near=set(w[max(0,j-1):j]+w[j+1:j+2])
    if x in marked and any(y not in shown for y in beside(w,j))and not near&shown:return nd,str(v)
  return None
 def item_clash(self,listing,pos):
  # a printed value breaking one of this item's own needs; a wanted word the page only extends (a prefix either way) is no breach
  block=_fact_block(self.full.get(listing)or{})or{}
  prod=self.items[pos-1]
  told=set(prod.told)
  aimed={pair[1]for pair in self.refs}
  # a page value spelt with a slip (a letter missing, doubled or swapped) still names the wanted value
  slip=lambda t,w:t.startswith(w)or w.startswith(t)or min(len(t),len(w))>=5 and difflib.SequenceMatcher(None,t,w).ratio()>=0.85
  for nd in prod.musts+prod.told:
   if nd in prod.dropped:continue
   hit=self.need_clash(nd,block,told,aimed)
   if not hit:continue
   if len(hit)>2:return hit[:2]
   want,shown=_tokens(nd.partition(":")[2]),_tokens(hit[1])
   if not(want and all(any(slip(t,w)for t in shown)for w in want)):return hit
  return None
 def searched_rows(self,prods):
  """Ids listed by searches whose words name a wanted value of these items."""
  want,tool=_tokens(" ".join(v for it in prods for v in self.word_values(it))),self.roles.get("search")
  return{c for r in self.search_log.values()if r["tool"]==tool and want&_tokens(_dump(r["args"]))for c in r["ids"]}
 def range_hold(self):
  if len(self.items)!=1 or self.left()<=self.tail()+1:return""
  kind,price=self.kind_of,self.price_of
  self.fill_facts(self.cart_ids(),self.tail()+1)
  lines,prod=self.cart_ids(),self.items[0]
  fresh=[cid for cid in lines if cid not in self.range_held]
  bad=[(cid,clash)for cid,clash in((cid,self.range_miss(cid))for cid in fresh)if clash]
  homes=[sort for sort in map(kind,self.seen_order)if _tokens(prod.name)&_tokens(sort)]
  if homes:bad.extend((cid,("the requested item: "+prod.name,kind(cid)))for cid in fresh if kind(cid)not in homes)
  if not bad:return""
  self.range_held.update(cid for cid,_ in bad)
  cheap=self.brief.get("pick")=="cheapest"
  fit=lambda:[c for c in self.full if c not in lines and self.buyable(c)and self.within_budget(c,prod.qty)and self.carries(c)and not self.range_miss(c)]
  fits=fit()
  if not fits and cheap and self.left()>self.tail()+4 and not self.finished:
   # nothing opened can take the line: the cheapest unread rows that searches on the wanted values listed are read first
   ok=lambda c:c not in self.full and self.buyable(c)and self.within_budget(c,prod.qty)and price(c)is not None
   unread=sorted(filter(ok,self.searched_rows([prod])),key=price)
   if unread and self.open_rows(unread[:self.compare_room()]):fits=fit()
  inf=float("inf")
  fits.sort(key=lambda c:(not cheap and not self.meets_limits(c),inf if price(c)is None else price(c)))
  self.trace.append({"kind":"range_hold","lines":bad,"fits":fits[:4]})
  listed="; ".join("%s: %s vs the need %s"%(cid,need_got[1],need_got[0])for cid,need_got in bad)
  shown=[c if price(c)is None else"%s (%g)"%(c,price(c))for c in fits[:4]]
  tail="; candidates to open (they hold the wanted words and clash with nothing): %s."%", ".join(shown)if fits else"."
  return"A cart line may clash with a need: %s. Decide it this way: the attribute named like the need settles it; when there is no such attribute, title and description settle it. The same value phrased differently is a match, and the line stays. On a real clash, swap in a listing that fits%s"%(listed,tail)
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
  return"\n[Kyo ran searches on the wanted values] Listings whose records contain those words, those showing every limit figure first and then by price - check each need before relying on one: "+shown
 def ref_hunt(self):
  tl=self.roles.get("search")
  key=self.single_target(tl)
  if not tl or not key or self.left()<self.tail()+12 or self.cart_lines or self.secs()<CLOCK_EXTRAS:return""
  if not(self.asked or self.step<self.max_steps//3):return""
  fresh=[]
  for prod in self.items:
   qtext=self.joined_query(prod)
   if not qtext or not any(n in self.ref_words and n not in prod.dropped for n in prod.musts+prod.told)or self.finished:continue
   sent,env=self.shaped_act(tl,{key:qtext},self.schemas.get(tl))
   if env is None:continue
   page=env.get("observation")
   self.observe(tl,page)
   self.log_search(tl,sent,page)
   log=self.search_log.get(tl+_dump(sent))or{}
   if log.get("token")is not None:self.page_on(log)
   words=_tokens(prod.name)
   new=[c for c in log.get("ids")or[]if c not in self.full and c not in fresh and self.buyable(c)and self.within_budget(c,1)and words&_tokens(_dump(self.rows.get(c)))]
   fresh.extend(new)
  batch=fresh[:self.compare_room()]
  if not batch or self.finished or not self.open_rows(batch):return""
  self.trace.append({"kind":"ref_hunt","read":batch})
  return self.carrying_note()
 def hunt_fit(self,c,pos,prod,vals):
  seen=_tokens(_dump(_fact_block(self.full.get(c)or{})or{}))|_tokens(self.title_of(c,45))
  if not(_tokens(prod.name)|_tokens(self.title_noun(prod)))&seen:return False
  if not(self.buyable(c)and self.within_budget(c,prod.qty))or self.item_clash(c,pos):return False
  if not all(any(_tokens(a)<=seen for a in re.split(r"\s+or\s+",v))for v in vals):return False
  # a limit's figure counts only where the page prints it: the page check cannot tell which side of the limit another figure falls
  return all(_tokens(" ".join(re.findall(r"\d+(?:\.\d+)?",t)))<=seen for t in self.limit_values(prod,True))
 def hunt_one(self,tl,key,prod,spare,full):
  """Search one item on its name and wanted values; its unread, buyable rows come back best first."""
  vals=list(dict.fromkeys(self.word_values(prod)))
  ran={str(v).strip().lower()for r in self.search_log.values()if r["tool"]==tl for v in r["args"].values()}
  # the second search leads with the next rarest value: a store reading only the first words would return the first one's rows again
  first=self.joined_query(prod)
  second=self.lead_query(prod,self.value_query(vals[:3]).split(),1)if vals else""
  asks=[q for q in dict.fromkeys((first,second))if q.strip()and q not in self.sought and q.lower()not in ran][:2 if full else 1]
  bags=[_tokens(v)for v in vals]
  kind=_tokens(prod.name)|_tokens(self.title_noun(prod))
  text=lambda c:_tokens(_dump(self.rows.get(c)))
  pool=[]
  for n,q in enumerate(asks):
   if len(pool)>=2 or self.finished or spare()<(3 if n else 1)or self.secs()<CLOCK_EXTRAS:break
   self.sought.add(q)
   sent,env=self.shaped_act(tl,{key:q},self.schemas.get(tl))
   if env is None:continue
   page=env.get("observation")
   self.observe(tl,page)
   self.log_search(tl,sent,page)
   # the wanted listing often sits just past the first page of such a search: one more page while the reads still fit
   log=self.search_log.get(tl+_dump(sent))or{}
   if log.get("token")is not None and spare()>=3 and not self.finished:self.page_on(log)
   found=log.get("ids")or[c for c,_ in self.records_in(page)]
   got=[c for c in found if c not in self.full and c not in pool and self.buyable(c)and self.within_budget(c,prod.qty)and kind&text(c)]
   pool.extend(sorted(got,key=lambda c:-sum(b<=text(c)for b in bags)))
  return pool
 def hunt_items(self):
  # after the interview, an item gets a search on its name and wanted values, then its likeliest unread rows opened
  tl=self.roles.get("search")
  key=self.single_target(tl)
  if not tl or not key or self.finished or self.secs()<CLOCK_EXTRAS:return""
  # one item: search again on whatever the replies revealed; several: only those with no opened page fitting yet
  todo=[]
  for pos,prod in enumerate(self.items,1):
   if len(self.items)>1:
    vals=list(self.word_values(prod))
    want=not any(self.hunt_fit(c,pos,prod,vals)for c in{*self.full,*self.cart_ids()})
   else:want=prod.told
   if want:todo.append((pos,prod))
  # each such item costs a search, two reads and its cart line; with less room it gets the search alone, every item keeps its cart line
  full=self.left()-self.tail()-len(self.items)>=3*len(todo)
  batch=bool(self.compare_target()and self.roles.get("compare")and not self.compare_bare())
  done,rows=[],[]
  for i,(pos,prod)in enumerate(todo):
   later=(3 if full else 1)*(len(todo)-i-1)
   spare=lambda:self.left()-self.tail()-len(self.items)-later
   pool=self.hunt_one(tl,key,prod,spare,full)
   room=spare()
   n=0 if room<1 else 3 if batch else min(2,room)
   picked=pool[:n]
   if full and picked and not self.finished and self.open_rows(picked):done.extend((pos,c)for c in picked if c in self.full)
   unread=[c for c in pool[:3]if c not in self.full]
   if unread:rows.append("item %d: %s"%(pos,", ".join(unread)))
  self.trace.append({"kind":"hunt","items":[pos for pos,_ in todo],"read":done,"rows":rows})
  out="".join("\n[Kyo, item %d, searched by name and wanted values] page %s: %s"%(pos,c,self.page_note(c))for pos,c in done)
  return out+(rows and"\n[Kyo's item searches also listed these unread rows] "+"; ".join(rows)or"")
 def carries(self,listing):
  seen=_tokens(_dump(_fact_block(self.full.get(listing)or{})or{}))|_tokens(self.title_of(listing,45))
  # a value offering alternatives is carried when any one of them is
  alts=lambda v:any(_tokens(alt)<=seen for alt in re.split(r"\s+or\s+",v))
  return any(vals and all(map(alts,vals))for vals in(list(self.word_values(prod))for prod in self.items))
 def interview_report(self,sent,new_reqs):
  if new_reqs:
   self.version+=1
   ver=self.version
   bump=lambda old:ver if old==ver-1 else old
   self.view_keys={k:(bump(first),second)for k,(first,second,*_)in self.view_keys.items()}
   self.read_cache={k:(bump(g[0]),*g[1:])for k,g in self.read_cache.items()}
  talk="\n".join("Q: %s\nA: %s"%(q,a)for _,q,a in sent)
  out=["Interview held by Kyo (each question cost one step):\n",talk,"\nFirm requirements that came out of the answers: ",
   "; ".join(new_reqs)if new_reqs else"none",". All picks are checked against them again; run new searches on those values where useful."]
  leads=self.reveal_leads(new_reqs)
  if leads:out.append("\nUnopened rows already printing a revealed value (open these first): "+"; ".join(leads))
  chained=[req for _,req in self.refs if any(req in line for line in new_reqs)]
  # the item searches go before the pointer walk, which can use up every step it is given
  hunted=self.hunt_items()
  walked=""
  if self.open_lookups()and self.secs()>CLOCK_STOP+15 and self.roles.get("search"):
   walked=self.trail()[0]
   self.trail_blocks=0
  chained=[req for req in chained if req in self.ref_values or req not in self.apart]
  if chained:out.append(" Musts that take their value from another listing: %s. Walk each chain one link at a time: open the listing referred to in full, then every further listing named on the way; the value is on the chain's final page, and ordering is too early while any page of the chain stays unopened."%" | ".join(chained))
  if walked:out.append("\n"+walked)
  out+=[hunted,self.carrying_note()]
  if self.queued:
   out.append("\n[store messages seen mid-interview] "+" | ".join(self.queued))
   self.queued=[]
  return"".join(out)
 def reveal_leads(self,new_reqs):
  # search rows not yet opened whose own text carries a value the shopper just named
  got=[]
  for req in new_reqs:
   value=req.rpartition(":")[2]
   want={w for w in _tokens(value)if len(w)>2}
   if not want:continue
   rows=[p for p,rec in self.rows.items()if p not in self.full and not self.sold_out(p)and want<=_tokens(_dump(rec))]
   if rows:got.append("%s: %s"%(value.strip()[:30],", ".join(rows[:4])))
  return got[:4]
 def ask_room(self):
  # the message argument's length limit, when the ask tool states one
  props=(self.schemas.get(self.roles.get("ask")or"")or{}).get("properties")or{}
  args=self.message_args("x")or{}
  top=(props.get(next(iter(args),""))or{}).get("maxLength")
  return top if isinstance(top,int)and top>0 else None
 def grouped(self,parts,top=None):
  # one message over several items, each item's keys in a clause of its own; the clause holding most keys sheds its last key
  # until the text fits the tool's length limit
  parts=[(n,list(ks))for n,ks in parts]
  def say(ps):
   bits=[]
   for n,ks in ps:
    spoken=[_spoken_key(k)for k in ks]
    bits.append("which %s should the %s have"%(spoken[0]if len(spoken)==1 else", ".join(spoken[:-1])+" or "+spoken[-1],self.items[n-1].name))
   text=", ".join(bits[:-1])+", and "+bits[-1]if len(bits)>1 else"".join(bits)
   return text[:1].upper()+text[1:]+"?"
  line=say(parts)
  while top and len(line)>top and parts and max(len(ks)for _,ks in parts)>1:
   big=max(range(len(parts)),key=lambda i:len(parts[i][1]))
   parts[big][1].pop()
   line=say(parts)
  return line,parts
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
  mark=self.cart_mark()
  lines=list(self.cart_lines.items())
  due=self.own_statement or(self.gate_due("order")and any(p["messages"]>0 for p in self.process if p["when"]=="order"))
  if not(due and lines)or self.said_for==mark or self.msg_used>=self.msg_limit or self.left()<2 or self.finished:return
  priced=[(cid,n,self.price_of(cid))for cid,n in lines]
  bits=["%s x %s (%s)%s"%(n,self.title_of(cid)[:70],cid,""if p is None else" at %g each"%p)for cid,n,p in priced]
  close=". Total %g."%sum(n*p for _,n,p in priced)if all(p is not None for *_,p in priced)else"."
  if not self.send([(1,"","Basket ready for checkout: "+"; ".join(bits)+close)],1):return
  self.gate_msgs+=1
  self.said_for=mark
 def send(self,queue_out,floor=6):
  tl=self.roles["ask"]
  done=[]
  for num,_,line in queue_out:
   if self.finished or self.left()<=floor or self.msg_used>=self.msg_limit:break
   args=self.message_args(line)
   if not args:break
   env=self.act(tl,args)
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
  filled=lambda key,field:[e for e in _items(verdict.get(key))if isinstance(e,dict)and str(e.get(field)or"").strip()]
  for ent in filled("needs","need"):
   pos=slot(ent)
   phr=str(ent["need"]).strip()
   told=self.items[pos-1].told
   if phr not in told:
    told.append(phr)
    added.append("item %s: %s"%(pos,phr)if size>1 else phr)
   if ent.get("points")is True and(pos,phr)not in self.refs:self.refs.append((pos,phr))
  for ent in filled("drop","property"):
   prod=self.items[slot(ent)-1]
   gone=_words(str(ent["property"]).split(":")[0])
   hits=[m for m in prod.musts+prod.told if m not in prod.dropped and _words(m.split(":")[0])==gone]
   prod.dropped.extend(hits)
   added.extend("dropped "+m for m in hits)
  for ent in(e for e in _items(verdict.get("free"))if isinstance(e,dict)and e.get("property")):self.items[slot(ent)-1].free.append(str(ent["property"]))
  cap=verdict.get("budget")
  if _kb_num(cap)and cap>0:self.budget=cap
 def judges(self):return[LIGHT,HEAVY]
 def hear_key(self,need,answer,keys):
  """Re-aim one heard need at the page key its reply names; return the property it ends on."""
  pname,colon,val=str(need.get("need")).partition(":")
  if not colon:return pname
  aim=lambda p:need.__setitem__("need","%s: %s"%(p,val.strip()))or p
  longer=[k for k in keys if _stem(pname)<_stem(k)and _stem(k)<=_tokens(answer)]
  if longer:pname=aim(_words(max(longer,key=len)))
  # a reply naming another page key than the one asked, sharing a word with it, sets the need on the key it names
  heard=" %s "%_words(answer)
  said=[k for k in keys if _words(k)and" %s "%_words(k)in heard]
  if not said or" %s "%_words(pname)in heard:return pname
  best=max(said,key=lambda k:len(_words(k)))
  wide=len(_stem(best))>1 or len(_stem(pname))==1
  return aim(_words(best))if wide and _stem(best)&_stem(pname)else pname
 def hear(self,done_qa,models=None):
  unit=self.brief.get("budget_unit")or"the unit the request states its budget in"
  menu="\n".join("item %s: %s"%(pos,prod.name)for pos,prod in enumerate(self.items,1))
  talk="\n".join("%s. (about item %s) Q: %s\nA: %s"%(i,pos,q,a)for i,(pos,q,a)in enumerate(done_qa,1))
  order=list(models or self.judges())
  # once the light reader came back empty this episode, the strong one reads first while spend stays under the soft cap
  if self.light_mute and self.cost<SPEND_SOFT and order[0]==LIGHT and HEAVY in order:order=[HEAVY,*(m for m in order if m!=HEAVY)]
  light_fails=lambda:len([t for t in self.trace if t.get("kind")=="llm_fail"and t.get("model")==LIGHT])
  fails=light_fails()
  verdict=self.ask_json("".join((HEAR,"\n\nBUDGET UNIT: ",str(unit),"\n\nITEMS:\n",menu,"\n\nQUESTIONS AND ANSWERS:\n",talk)),order,8000)or{}
  if light_fails()>fails:self.light_mute=True
  turn=lambda ent:_count(ent.get("q"))if isinstance(ent,dict)else None
  free_turns={_count(ent.get("q"))for ent in _items(verdict.get("free"))if isinstance(ent,dict)}
  open_turn=lambda t:t in free_turns and t and t<=len(done_qa)
  idle=lambda ent:open_turn(turn(ent))and not _tokens(str(ent.get("need")).split(":")[0])&_tokens(done_qa[turn(ent)-1][2])
  needs=verdict["needs"]=[ent for ent in _items(verdict.get("needs"))if not idle(ent)]
  keys={k for rec in self.full.values()for k in _fact_block(rec)or{}}
  voiced,turns=set(),set()
  for ent in needs:
   t=turn(ent)
   if not t or t>len(done_qa):continue
   _,asked,answer=done_qa[t-1]
   if _stem(self.hear_key(ent,answer,keys))&_tokens(asked+" "+answer):
    voiced.add(id(ent))
    turns.add(t)
  verdict["needs"]=[ent for ent in needs if id(ent)in voiced or turn(ent)not in turns]
  return verdict
 def gate_text(self,owed,at_order,menu):
  log="\n".join("%s. %s"%pair for pair in enumerate(self.done_log,1))or"(nothing yet)"
  notes=["\n\nSTORE NOTES SEEN:\n","\n".join(self.notes[-4:])]if self.notes else[]
  room=(max(0,self.msg_limit-self.msg_used),self.left(),"order"if at_order else"ending")
  return"".join([GATE.replace("STAGE","place the order"if at_order else"end without a purchase"),"\n\nREQUEST:\n",self.request,
   "\n\nSTEPS DUE NOW:\n","\n".join("- %s"%(s["step"],)for s in owed),"\n\nLEDGER:\n",self.ledger(),"\n\nCART NOW: ",
   self.cart_text(),""if self.cart_sum()is None else" (sum %g)"%self.cart_sum(),"\n\nRECORD (every message and call so far, numbered):\n",log,
   *notes,"\n\nSTILL AVAILABLE: %s message(s), %s step(s); the %s takes one step"%room,"\n\nTOOLS:\n","\n".join(menu)])
 def gate_unproven(self,rul,owed,at_order):
  fresh_only=at_order and any(s["messages"]>0 for s in owed)
  ok_line=lambda i:not fresh_only or(i<=len(self.done_steps)and self.done_steps[i-1]>self.cart_at)
  good={i for i in range(1,len(self.done_log)+1)if ok_line(i)}
  proven=lambda c:c.get("done")is True and any((_count(n)or 0)in good for n in _items(c.get("lines")))
  return[c for c in _items(rul.get("steps"))if isinstance(c,dict)and not proven(c)]
 def gate_ask(self,args,at_order):
  line=next((v for v in args.values()if isinstance(v,str)),"")
  line=self.with_sum(line)if at_order else line
  qa=self.send([(1,"",line)],1)
  if not qa:return None
  self.gate_msgs+=1
  return"Q: %s\nA: %s"%(line,qa[0][2])
 def gate_run(self,name,args):
  env=self.act(name,args)
  if env is None:return None
  page=env.get("observation")
  self.absorb(page)
  self.done_log.append("step %s: called %s with %s, got %s (cart afterwards: %s)"%(self.step,name,_dump(args),self.show(page)[:300],self.cart_mark()))
  self.done_steps.append(self.step)
  return"%s %s -> %s"%(name,_dump(args),self.show(page)[:400])
 def gate(self,stage):
  owed=[s for s in self.process if s["when"]==stage]
  if not owed or self.finished or not self.gate_due(stage):return True,""
  closers={self.roles.get(job)for job in("buy","refuse")}-{None}
  menu=["- %s: %s | args schema: %s"%(f["name"],f.get("description",""),_dump(f.get("parameters")or{}))
   for f in(t["function"]for t in self.tools)if f["name"]not in closers]
  at_order=stage=="order"
  head="Kyo took care of these required steps before now:\n"
  lines,pushed=[],False
  for _ in range(GATE_ROUNDS):
   rul=self.ask_json(self.gate_text(owed,at_order,menu),[HEAVY,LIGHT]if pushed else[LIGHT,HEAVY],2000)
   if rul is None:break
   told=[]
   if lines:self.take_heard({"needs":rul.get("needs")},told)
   hold=str(rul.get("hold")or"").strip()if lines else""
   call=rul.get("call")
   if not isinstance(call,dict):call=None
   self.trace.append({"kind":"gate","stage":stage,"call":_clip(call,200),"steps":_clip(rul.get("steps"),120),"hold":hold[:200],"told":told})
   if told or hold:
    self.version+=1
    done=lines and head+"\n".join(lines)+"\n"or""
    return False,done+"The shopper's reply changed the task: %s. Rework the cart to match, then try checkout again."%"; ".join(told+[hold]*bool(hold))
   if not call:
    unproven=self.gate_unproven(rul,owed,at_order)
    can_push=unproven and not(lines or pushed)and self.roles.get("ask")and self.msg_used<self.msg_limit and self.left()>=2
    if not can_push:
     self.gate_ok.add((stage,"%s#%s"%(self.cart_mark(),self.version)))
     break
    pushed=True
    self.trace.append({"kind":"gate_unproven","steps":_clip(unproven,120)})
    self.done_log.append("(Kyo note: no proof after the cart change at step %s for: %s. Left: %s message(s), %s steps; a next call is needed)"
      %(self.cart_at,"; ".join(str(c.get("step")or"")for c in unproven)[:200],self.msg_limit-self.msg_used,self.left()))
    self.done_steps.append(-1)
    continue
   name=str(call.get("tool")or"")
   if name in closers or name not in self.schemas:break
   chat=name==self.roles.get("ask")
   if self.left()<2 or chat and self.msg_used>=self.msg_limit:break
   raw,faults=call.get("args"),[]
   args=_shape(raw if isinstance(raw,dict)else{},self.schemas[name],"args",faults)
   if faults:break
   note=self.gate_ask(args,at_order)if chat else self.gate_run(name,args)
   if note is None:break
   lines.append(note)
  return True,lines and head+"\n".join(lines)or""
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
  said=" ".join(node for node,_ in self._svd_nodes(outcome,4)if isinstance(node,str)and len(node.split())>=2)
  return said[:800]if said else _dump(outcome)[:400]
 def loose_lines(self,outcome):
  if not isinstance(outcome,dict)or self.ident(outcome):return[]
  wordy=lambda x,n:len(x.split())>=n;found=[]
  for v in outcome.values():
   if isinstance(v,str):
    if wordy(v,5):found.append(v)
   elif isinstance(v,list):
    if v and all(isinstance(x,str)for x in v):found+=[x for x in v if wordy(x,5)]
   elif isinstance(v,dict)and not self.ident(v,False,1):found+=[x for x in v.values()if isinstance(x,str)and wordy(x,6)]
  return[x for x in found if x not in self.heard]
 def take_note(self,lines_in):
  """Read a store or shopper line that may change the task; update the ledger and its version."""
  self.heard.update(lines_in);prompt=f"{NOTE}\n\nITEM LEDGER AS OF NOW:\n{self.ledger()}\n\nTHE NOTE:\n"+"\n".join(lines_in)
  verdict=self.ask_json(prompt,[LIGHT,SPARE],1200)or{};gone_ids=list(map(str,filter(lambda g:isinstance(g,(str,int)),_items(verdict.get("gone")))))
  self.unavailable.update(filter(self.rows.__contains__,gone_ids));self.note_sold=True if gone_ids else verdict.get("sold_out")is True
  self.sold_lines.update(lines_in if self.note_sold and len(lines_in)==1 else());kind=verdict.get("kind")
  if kind is not None and kind!="change":return""
  changes,cap=[],verdict.get("budget")
  if isinstance(cap,(int,float))and not isinstance(cap,bool)and cap>0 and cap!=self.budget:
   if self.budget is None or cap<self.budget:self.cap_cut=cap
   self.budget=cap;changes.append("budget now %s"%(cap,))
  stated=lambda e:isinstance(e,dict)and str(e.get("need")or"").strip();valid=lambda field:[*filter(stated,_items(verdict.get(field)))]
  last=len(self.items);owner=lambda e:self.items[min(max(1,_count(e.get("item"))or 1),last)-1]
  drops=valid("drop")
  for e in drops:
   phr=str(e["need"]).strip();prod=owner(e)
   changes.append("dropped "+phr);live=[m for m in(*prod.musts,*prod.told)if m not in prod.dropped]
   gone=[m for m in live if m==phr]
   if not gone:
    head=_words(phr.partition(":")[0]);gone=next(([m]for m in live if _words(m.partition(":")[0])==head),[phr])
   prod.dropped.extend(gone)
  # a dropped requirement only widens what fits: a cart that met every requirement still meets the rest, while an empty cart's
  # search may take what that requirement excluded
  if drops:
   if self.cart_ids():changes.append("the cart still meets every remaining requirement, so it can be sent again as it is")
   elif self.brief.get("pick")in("cheapest","most","least"):changes.append("any listing excluded solely by that requirement is eligible again, so apply the pick rule afresh across all records held")
  for e in valid("add"):
   phr=str(e["need"]).strip();prod=owner(e)
   changes.append("added "+phr)
   if prod.dropped.count(phr):prod.dropped.remove(phr)
   if all(phr not in group for group in(prod.musts,prod.told)):prod.told.append(phr)
  if gone_ids:changes.append("unavailable: "+", ".join(gone_ids))
  if verdict.get("price_moved")is True:
   # 2 marks a note that moves nothing but a price
   self.price_rose=2-bool(changes);changes.append("a price changed")
  summary=str(verdict.get("summary")or"").strip();told="; ".join(changes)
  if summary or changes:
   self.version+=1;self.recent=[]
   self.notes.append(summary or told)
  return f"{summary} ({told})".strip()if changes else summary
 def needs_now(self):
  out=[]
  for pos,prod in enumerate(self.items,1):out+=[(pos,n)for n in prod.musts+prod.told if n not in prod.dropped]
  return out
 def _evidence_parts(self,cid,live,spans,many):
  fblock=_fact_block(self.full.get(cid)or{})or{};parts=[]
  for pos,n in live:
   head,colon,rest=n.partition(":");pname,want=head.strip(),(rest if colon else head).strip().lower()
   pairs=[f"{k}={str(v)[:50]}"for k,v in fblock.items()if _names(_stem(pname),_stem(k))]
   if not pairs:
    same=[k for k,v in fblock.items()if str(v).strip().lower()==want]if re.search("[a-z]",want)else[]
    if not same:continue
    pairs=["no key of that name; the wanted value %s is stated under %s, which counts as met"%(want,", ".join(same[:3]))]
   tag=""if(pos,n)not in spans else" [range met: one entry restates the wanted span]"if self.states_range(cid,n)else" [range not restated by any entry]";lead="(item %d) "%pos if many else""
   parts.append(f"{lead}{pname} -> "+", ".join(pairs[:3])+tag)
  return parts
 def evidence(self,outcome):
  hits=[]
  def prune(piece,depth):
   if not isinstance(piece,dict):return False
   cid=self.ident(piece,False,depth)
   if not cid or not _fact_block(piece)or cid in hits:return False
   hits.append(cid)
   return True
  list(self._svd_nodes(outcome,6,prune));live=self.needs_now()
  if not(hits and live):return""
  many=len(self.items)>1;spans=self.range_musts()
  rows=[]
  for cid in hits[:12]:
   parts=self._evidence_parts(cid,live[:8],spans,many)
   if parts:rows.append(cid+": "+"; ".join(parts))
  return"[evidence | matching keys per opened listing, by requirement name; ones for another component or facet prove nothing either way, and the value may sit under an unrelated key name too] "+" | ".join(rows)if rows else""
 def range_musts(self):
  aimed={pair[1]for pair in self.refs}|set(self.ref_values);ranged=lambda n:_two_figures(_figure_words(n.partition(":")[2]if":"in n else n))
  return[(pos,n)for pos,n in self.needs_now()if n not in aimed and ranged(n)]
 def states_range(self,listing,need):
  fblock=_fact_block(self.full.get(listing)or{})or{};head,colon,tail=need.partition(":")
  wanted=_figure_words(tail if colon else need);pname=_stem(head)
  named=[v for k,v in fblock.items()if _names(pname,_stem(k))]
  for v in named or list(fblock.values()):
   if any(_repeats(wanted,bit)for bit in re.split(r"[,;/|]",str(v))):return True
  return False
 def cart_text(self):
  if self.cart_view is None or self.cart_seen_at<self.cart_at:return"(cart unseen; Kyo's tally from its calls, listing -> units) "+_dump(self.cart_lines)
  return _dump(_clip(self.cart_view,300))[:3000]
 def cart_ids(self):
  if self.cart_view is None or self.cart_seen_at<self.cart_at:return list(self.cart_lines)
  shown=_dump(self.cart_view)
  return list(filter(lambda cid:cid in shown,self.rows))
 def fixed_page(self,listing):
  # a stored page printing attributes and neither price nor stock reads the same on every visit
  rec=self.full.get(listing)
  if listing in self.gone or not _fact_block(rec):return False
  text=_dump(rec)
  return not any(f'"{k}":'in text for k in(self.price_key,self.stock_key)if k)
 def codes_of(self,listing):
  rec=self.full.get(listing)or self.rows.get(listing)or{};found=re.findall(r"\b[A-Za-z]{2,}-?\d+\b",_dump(rec))
  return{m.replace("-","").lower()for m in found}
 def record_of(self,listing):
  rec=self.full.get(listing)or self.rows.get(listing)
  return _dump(_clip(rec,400))[:3000]if rec else"(nothing recorded yet)"
 def page_note(self,listing,size=500):
  # a short page note leads with the attribute map, whole; the other fields share the room left, each clipped alike (a title keeps
  # its first 40 characters at least), so a long description cannot push the attributes off the note
  rec=self.full.get(listing)or{}
  facts=_fact_block(rec)
  lead={k:_clip(v,400)for k,v in rec.items()if facts and _fact_block({k:v})==facts}
  note=lambda cap:_dump({**lead,**{k:_clip(v,cap)for k,v in rec.items()if k not in lead}})
  return next((n for n in map(note,(400,160,80,60))if len(n)<=size),note(40))
 def others(self,skip,limit=30):
  price=lambda cid:float("inf")if self.price_of(cid)is None else self.price_of(cid);order=list(reversed(self.seen_order))
  if self.brief.get("pick")=="cheapest":order.sort(key=lambda cid:(cid not in self.full,price(cid)))
  picked=list(filter(lambda c:c not in skip and not self.sold_out(c),order))[:limit];recs=[self.full.get(c)or self.rows.get(c)for c in picked]
  return"\n".join(_dump(_clip(r,160))[:900]for r in recs)
 def check_key(self,ids):return"|".join(sorted(ids))+"#"+str(self.version)
 def review_sheet(self,ids,proposed):
  if not proposed:basket="CART:\n"+self.cart_text()
  else:
   counts="; ".join(f"item {pos}: quantity {it.qty}"for pos,it in enumerate(self.items,1));basket=f"CART (plan only, not yet added - assume these at the quantities asked; {counts}):\n"+", ".join(ids)
  sheet=[f"{pid}: "+("SOLD OUT, cannot be bought at present. "if self.sold_out(pid)else"")+self.record_of(pid)for pid in ids]
  parts=[CHECK]+[f"{head}:\n{body}"for head,body in(("REQUEST",self.request),("REQUIREMENTS IN FORCE",self.ledger()))]+[basket,"CART LISTING RECORDS:\n"+"\n".join(sheet)]
  if self.brief.get("pick")in("cheapest","most","least"):parts.append("OTHER LISTINGS (NEWEST):\n"+self.others(ids,15))
  # parts made to go together print one code (letters joined to figures) in common with what they go with: shown as evidence
  codes={pid:self.codes_of(pid)for pid in ids}
  if len(ids)>1 and any(codes.values()):
   tally={}
   for pid in ids:
    for c in codes[pid]:tally[c]=tally.get(c,0)+1
   shared=sorted(c for c,k in tally.items()if k>1);printed="; ".join(f"{pid}: "+(", ".join(sorted(codes[pid]))or"none")for pid in ids)
   parts.append("CODES THE CART RECORDS PRINT (letters joined to figures): "+printed+(". Printed by two or more lines: "+", ".join(shared)if shared else""))
  gaps=[]
  for _,n in self.range_musts():
   if any(self.states_range(cid,n)for cid in ids):continue
   holders=[]
   for cid in reversed(self.seen_order):
    if cid in ids or not self.buyable(cid)or not self.within_budget(cid,1):continue
    if self.states_range(cid,n):holders.append(cid)
   if holders:gaps.append((n,holders[:4]))
  if gaps:
   owners=[]
   for _,who in gaps:owners+=[p for p in who if p not in owners]
   told="; ".join(f"{need} (repeated by {', '.join(who)})"for need,who in gaps)
   parts.append("RANGE MUSTS uncovered in this cart: "+told+"\nHolders' records:\n"+"\n".join(f"{pid}: {self.record_of(pid)}"for pid in owners))
  return"\n\n".join(parts)
 def check_cart(self,ids=None,limit=SPEND_SOFT):
  """Strong review of a cart against the ledger before paying; an objection holds the checkout."""
  proposed=ids is not None;ids=ids if proposed else self.cart_ids()
  key=self.check_key(ids)
  if not ids or self.cost>limit or key in self.checked:return""
  self.checked.add(key);ruling=self.ask_json(self.review_sheet(ids,proposed),[HEAVY,LIGHT],8000)
  self.trace.append(dict(kind="check",cart=ids,got=ruling))
  if not ruling:
   self.verdicts[key],self.flawed=False,None
   return""
  passed=ruling.get("ok")is not False;problems=_items(ruling.get("problems"))
  self.verdicts[key]=passed
  # approved with faults named = a near miss is about to be paid
  self.flawed=key if passed and problems else None
  if passed:return""
  self.leads=[c for c in self.leads if c not in ids];faults=[str(p)for p in problems][:5]or["recheck each stated need"]
  better=[c for c in map(str,_items(ruling.get("better")))if c in self.rows and not self.sold_out(c)][:4];extra="; reviewer prefers "+", ".join(better)if better else""
  return f"Second opinion withholds approval ({'; '.join(faults)}{extra}). Either amend the cart or quote records that refute the objection; payment comes after."
 def open_lookups(self):
  live={n for _,n in self.needs_now()}
  return[n for n in(pair[1]for pair in self.refs)if n in live and n not in self.ref_values]
 def _chain_settled(self,got,req,chain,hops,ids,value):
  want=_stem(value);facts=lambda pid:_fact_block(self.full.get(pid)or{})
  # a value printed on an earlier link but not on the last one was read off the wrong page
  shows=lambda pid:any(want<=_stem(v)for v in(facts(pid)or{}).values()if isinstance(v,str));rel=chain.get("relation")
  last=ids[-1];mate=str(chain.get("partner")or"").strip()
  # a last link naming the partner was reached past the pairing rule, so the starting value rightly sits earlier
  past=False
  if rel=="together"and mate and _stem(mate)!=want:past=_tokens(mate)<=_tokens(_dump(self.full.get(last)or{}))
  early=list(filter(shows,ids[:-1]))
  if early and facts(last)and not shows(last)and not past:
   got["redo"].append(f"{req}: {value} is printed on {early[0]}, an earlier link, not on the last link {last}; the figure sits on {last}'s own page")
   return
  head=hops[0];about,shop=head.get("about")or"",head.get("shop")or""
  first=ids[0],str(about),str(shop).strip();note=self.sibling_note(req,*first)
  if note and len(self.rehops)<2 and note not in self.rehops:
   got["redo"].append(note)
   return
  got["first"][req]=first
  if rel=="apart":entry=req,f"{value} is excluded: listings showing it fail, all other values pass",None,None
  elif rel!="together":entry=req,value,value,None
  elif not mate:
   seller=str(hops[-1].get("shop")or"").strip();got["miss"].append(f"{req} -> find what pairs with {value}")
   got["find"].append((seller,));got["plain"].append((req,value))
   return
  elif _stem(mate)==want:entry=req,f"{mate} (the partner a pairing rule on the pointed pages gives); passes only showing {mate} itself",mate,None
  else:entry=req,f"{mate} (what goes with {value}); passes only showing {mate} itself, while {value} on a listing fails",mate,last if shows(last)else None
  got["res"].append(entry)
 def _chain_gap(self,got,req,hops,ids):
  gap={}
  for h,i in zip(hops,ids):
   if i not in self.rows:
    gap=h
    break
  about=str(gap.get("about")or"");desc=about or"the referenced listing"
  got["miss"].append(f"{req} -> find {desc}");filled=[str(gap.get(f)or"").strip()for f in("about","shop","words")]
  # a link naming no seller, words or description gives a search nothing to look for
  if not gap or not any(filled):
   tail=ids[-1]if ids else None
   if tail is not None and tail in self.opened and tail not in self.full:got["read"].append(tail)
   return
  raw=gap.get("words")or""
  if isinstance(raw,list):raw=" ".join(map(str,raw))
  words=str(raw);named=filled[1]or self.shop_named(desc)
  seller=named or(desc if filled[0]else words)[:60]
  # a named seller is sought with the link's own words first and by its name alone after; those words also rank its rows
  if named:got["find"].append((seller,))
  else:got["find"].append((" ".join(dict.fromkeys(seller.split()+words.split())),seller))
  # the link's own distinguishing words lead its description, so a seller search can carry them
  got["about"][seller]=f"{words} {desc}".strip()
 def chain_moves(self,reply,pending):
  got=dict(res=[],open=[],find=[],read=[],miss=[],about={},redo=[],plain=[],apart={},first={});text=lambda node,field:str(node.get(field)or"").strip()
  chains=list(filter(lambda c:isinstance(c,dict),_items(reply.get("chains"))));exact=set(map(lambda c:text(c,"need"),chains))
  owned=set();only=pending[0]if len(pending)==1 else None
  for chain in chains:
   named=text(chain,"need")
   if named and named in pending:req=named
   elif only and only not in exact and(len(chains)==1 or named!=""and(named in only or only in named)):req=only
   else:req=_loose_owner(named,pending,exact|owned)
   if not req:continue
   owned.add(req);hops=list(filter(lambda h:isinstance(h,dict),_items(chain.get("links"))))
   ids=[text(h,"id")for h in hops];value=text(chain,"value")
   if chain.get("relation")=="apart":
    # an exclusion's relation is known before its chain resolves: keep where it points
    end=([h for h,i in zip(hops,ids)if i not in self.opened]or hops[-1:]or[{}])[0];shop=str(end.get("shop")or"").strip()
    parts=[str(end.get("about")or"").strip()[:100]]
    if shop not in str(end.get("about")):parts.append("seller "+shop)
    seen=[i for i in ids if i in self.rows]
    if seen:parts.append("chain read so far "+", ".join(seen))
    got["apart"][req]="; ".join(p for p in parts if p)or"the referenced listing"
   else:got["apart"][req]=None
   if value and hops and chain.get("done")is True and all(map(self.opened.__contains__,ids)):
    self._chain_settled(got,req,chain,hops,ids,value)
    continue
   closed=[i for i in ids if i in self.rows and i not in self.opened]
   if closed:got["open"].extend([i for i in closed if i not in got["open"]])
   else:self._chain_gap(got,req,hops,ids)
  return got
 def _trail_take(self,plan):
  ends=plan["apart"]
  for req in[r for r,where in ends.items()if not where]:self.apart.pop(req,None)
  self.apart.update((r,where)for r,where in ends.items()if where)
  for req,said,word,end in plan["res"]:
   self.ref_values[req]=said
   if req in plan["first"]:self.ref_firsts[req]=(*plan["first"][req],end)
   if end is not None:self.chain_ends.add(end)
   if word is not None:self.ref_words[req]=word
  self.version+=len(plan["res"])
 def _fetch_page(self,tool,key,pid,stop):
  # one listing page through TOOL: None when the call cannot be shaped, STOP() holds it back, or the episode ends
  bad=[];args=_shape({key:pid},self.schemas.get(tool),"args",bad)
  if bad or stop():return None
  got=self.act(tool,args)
  if got is None or self.finished:return None
  self.observe(tool,got.get("observation"))
  return got
 def trail(self):
  """Follow pointer requirements to the listing they name and record the value found there."""
  self.recheck_firsts();pending=self.open_lookups()
  viewer=self.roles.get("offer")or self.roles.get("open");key=self.single_target(viewer)
  texts,missing,about=[],[],{};unsettled=lambda reqs:[r for r in reqs if r not in self.ref_values]
  while pending:
   if self.trails>=8 or self.cost>=SPEND_HARD or self.secs()<=CLOCK_STOP:break
   # lookups never take the steps the questions need: before the first question, chains stop at a third of the budget
   if not self.asked and self.step>=self.max_steps//3:break
   self.trails+=1;shelf=[]
   for pid in self.seen_order[::-1][:60]:
    # a clipped row still runs past 300 characters: keep it whole so its id and seller stay readable
    rec=None if pid not in self.opened else self.full.get(pid)or self.rows.get(pid)
    if rec:shelf.append("OPENED "+_dump(_clip(rec,300))[:1400])
    else:shelf.append("ROW "+_dump(_clip(self.rows.get(pid),80))[:600])
   sections=[("REQUEST",[self.request]),("VALUE-ELSEWHERE REQUIREMENTS",pending)]
   if self.rehops:sections.append(("EARLIER ANSWER CORRECTED",self.rehops))
   sections.append(("RECORDS, NEWEST FIRST",shelf));ask="\n\n".join([TRAIL]+[f"{head}:\n"+"\n".join(body)for head,body in sections])
   # the same lookups over the same records were walked already: nothing new to read, the links still missing stay as found
   if ask in self.trail_asks:
    missing=self.trail_asks[ask]
    break
   reply=self.ask_json(ask,[LIGHT,HEAVY],8000)or{};self.trace.append({"kind":"trail","pending":pending,"got":_clip(reply,150)})
   plan=self.chain_moves(reply,pending);self._trail_take(plan)
   todo=plan["open"];missing=self.trail_asks[ask]=plan["miss"]
   about.update(plan["about"]);pending=unsettled(pending)
   if pending and not todo:
    spare=2-len(self.rehops);fresh=[r for r in plan["redo"]if r not in self.rehops]
    if fresh and spare>0:
     self.rehops+=fresh[:spare]
     continue
    if plan["read"]and self.read_pages(plan["read"],self.facts_tool()):continue
    if plan["find"]:
     found=[]
     for names in list(dict.fromkeys(plan["find"]))[:2]:
      hint=about.get(names[-1],"");found.append(any(self.seek_shop(n,hint)for n in dict.fromkeys(names)))
     if any(found):continue
    # no pairing rule on the chain and none found at that seller: the pointed page's own value stands
    for req,value in plan["plain"]:
     self.ref_words[req]=value;self.ref_values[req]=f"{value} (no pages read print a pairing rule for it, so the pointed page's own value stands)"
     self.version+=1
    marks=tuple(req+" ->"for req,_ in plan["plain"]);missing=[m for m in missing if not m.startswith(marks)]
    pending=unsettled(pending)
   if not(todo and viewer and key)or self.finished:break
   if self.left()-4<len(todo):break
   for pid in todo[:max(0,min(3,6-len(texts)))]:
    got=self._fetch_page(viewer,key,pid,lambda:False)
    if got is None:break
    texts.append(f"{pid}: {self.show(got.get('observation'))[:1500]}")
  return self.trail_wrapup(texts,missing,pending)
 def sibling_note(self,req,pid,about,shop):
  # a link described by its seller's words ("the <value> one") is the seller's listing whose own attributes show them: an opened
  # page of the same seller and kind is the link instead when one of its attribute values is wholly in the description while
  # pid's value for that attribute is not, and no attribute of pid beats it the same way
  if not shop:return""
  own=_fact_block(self.full.get(pid)or{})or{}
  if not own:return""
  kind=self.kind_of(pid);seller=_tokens(shop)
  want=_tokens(about)-seller-_tokens(kind);beats=lambda a,b:[k for k,v in a.items()if k in b and _tokens(v)and _tokens(v)<=want and not _tokens(b[k])<=want]
  for c,rec in self.full.items():
   if c==pid or self.kind_of(c)!=kind or not seller<=_tokens(_dump(rec)):continue
   facts=_fact_block(rec)or{};wins=beats(facts,own)
   if not wins or beats(own,facts):continue
   k=wins[0];said,held=str(facts[k])[:60],str(own[k])[:60]
   return"%s: the link described as '%s' is %s, not %s: %s shows %s=%s, as described, where %s shows %s=%s; restart the chain from %s"%(req,about[:120],c,pid,c,k,said,pid,k,held,c)
  return""
 def recheck_firsts(self):
  # a settled chain is reopened when a sibling page opened since fits its first link's description better
  for req,(pid,about,shop,end)in list(self.ref_firsts.items()):
   if req not in self.ref_values or len(self.rehops)>=2:continue
   note=self.sibling_note(req,pid,about,shop)
   if not note:continue
   self.ref_firsts.pop(req)
   for book in(self.ref_values,self.ref_words):book.pop(req,None)
   self.chain_ends.discard(end);self.rehops.append(note)
   self.version+=1
 def trail_wrapup(self,texts,missing,pending):
  out=""
  if texts:out="[Kyo opened these pages that requirements point to]\n"+"\n".join(texts)+"\n"
  if self.ref_values:out+="[values Kyo settled from pointed-to pages] "+"; ".join(f"{r} -> {v}"for r,v in self.ref_values.items())+"\n"
  kept=[m for m in missing if not m.startswith(tuple(r+" ->"for r in self.apart))]
  unread=[f"{req} -> value not read yet"for req in pending if req not in self.apart and not any(m.startswith(req)for m in missing)];missing=kept+unread
  # paying before a pointed page is opened has no chance: past two holds, the hold stays while steps and seconds remain
  if self.trail_blocks<2:room=True
  else:room=self.trail_blocks<4 and self.left()>=self.tail()+6 and self.secs()>CLOCK_EXTRAS
  stuck=[pid for pid in self.cart_ids()if pid in self.chain_ends]
  if stuck and room and self.left()>4:
   self.trail_blocks+=1
   return out+"Swap needed for "+", ".join(stuck)+": being the final link, it carries the origin value, not the matching partner value. Look up a listing with the partner value and put it in instead; complete the order afterwards.",True
  if not room or not missing or self.left()<=5:return out,False
  self.trail_blocks+=1
  return out+"Pending lookups: "+" | ".join(missing[:3])+". A search row settles nothing; what settles a pointed-to listing is its own opened page (and so on for any listing it points onward to). Track it down by seller, brand or phrasing, open it, rely on the value printed there, and only then complete the order.",True
 def shop_named(self,words):
  seen=_tokens(" ".join(map(self.kind_of,self.seen_order)))
  for w in re.findall(r"(?<=\s)[A-Z][a-z]{2,}"," "+words):
   if not seen&_tokens(w):return w
  return""
 def shop_rows(self,name,about):
  target=name.strip().lower();sells=lambda p:any(isinstance(v,str)and v.strip().lower()==target for v in(self.rows.get(p)or{}).values())
  hint=_tokens(about);picked=[p for p in self.seen_order if p not in self.opened and sells(p)]
  picked.sort(key=lambda p:-len(hint&_tokens(_dump(_clip(self.rows.get(p),300)))))
  return picked
 def read_pages(self,listings,tool):
  key=self.single_target(tool);done=0
  stop=lambda:self.finished or self.left()<self.tail()+8
  for pid in(listings[:CHAIN_READS]if tool and key else()):
   if self._fetch_page(tool,key,pid,stop)is None:break
   done+=1
  self.trace.append({"kind":"chain_read","rows":listings[:CHAIN_READS],"read":done})
  return done>0
 def shop_narrowed(self,tool,text_arg,name,about,figured=False):
  kinds=_tokens(" ".join(map(self.kind_of,self.seen_order)));mine=_tokens(name)
  known=mine|kinds;printed=set()
  if len(self.rows)>=3:printed=set().union(*map(lambda r:_tokens(_dump(r)),self.rows.values()))
  terms=[]
  for w in re.findall(r"[A-Za-z0-9-]+",about):
   if len(w)<=2 and not _lettered_figure(w)or w.isdigit()or known&_tokens(w):continue
   if not printed or _tokens(w)<=printed:terms.append(w)
  # the link's own words single its listing out: a code it quotes, else the values it describes; when those list no row at
  # all, the seller with the kind of item the link names
  figs=list(dict.fromkeys(filter(_lettered_figure,terms)));noun=lambda w:len(w)>2 and _tokens(w)<=kinds and not _tokens(w)&mine
  nouns=list(dict.fromkeys(filter(noun,re.findall(r"[A-Za-z]+",about))));picks=figs[:3]or[*dict.fromkeys(terms)][:3]
  for words in(picks,nouns[:2]):
   query=" ".join([name,*words])if words else"";bad=[]
   args=_shape({text_arg:query},self.schemas.get(tool),"args",bad);blocked=self.finished or not query or bad or query in self.sought
   if blocked or self.left()-9<self.tail():continue
   self.sought.add(query);got=self.act(tool,args)
   if got is None or self.finished:return[]
   seen=got.get("observation");self.observe(tool,seen)
   self.log_search(tool,args,seen);found=self.records_in(seen)
   if not found:continue
   # the link names its seller: once rows of that seller are on hand, other sellers' hits are not the listing
   own=self.shop_rows(name,about);loose=not(own or figured)
   return[pid for pid,*_ in found if pid not in self.opened and(loose or pid in own)]
  return[]
 def seek_shop(self,name,about=""):
  tool=self.roles.get("search");key=self.single_target(tool)
  if self.finished or not(tool and key and name):return False
  if self.left()<self.tail()+6:return False
  one_by_one=True if self.compare_bare()else not(self.compare_target()and self.roles.get("compare"));viewer=self.offer_tool()
  if one_by_one and name in self.sought:
   queued=self.shop_rows(name,about)
   if queued:
    joined=self.shop_narrowed(tool,key,name,about)+queued
    return self.hop_read(name,list(dict.fromkeys(joined)),viewer)
  elif one_by_one:
   # a link quoting figures or naming distinguishing words: search its seller with them before paging through that seller's whole range
   first=self.shop_narrowed(tool,key,name,about,True)
   if first:return self.hop_read(name,first,viewer)
  spec,bad=self.schemas.get(tool),[];args=_shape({key:name},spec,"args",bad)
  if bad:return False
  log=self.search_log.get(tool+_dump(args));waiting=[pid for pid in(log.get("ids")if log else None)or[]if pid not in self.opened]
  if waiting:return self.open_rows(waiting)
  if not log:
   if name in self.sought:return False
  elif not log.get("token"):return False
  self.sought.add(name)
  if log:
   args=_shape({**log["args"],log["page_arg"]:log["token"]},spec,"args",bad)
   if bad:return False
  got=self.act(tool,args)
  if got is None or self.finished:return False
  page=got.get("observation");self.observe(tool,page)
  self.log_search(tool,args,page);unopened=lambda ids,also=():[pid for pid in ids if pid not in self.opened and pid not in also]
  fresh=unopened(pid for pid,_ in self.records_in(page));plain=dict(args)
  if log:plain.pop(log.get("page_arg"),None)
  log=self.search_log.get(tool+_dump(plain))or{}
  if not fresh and log.get("token")is not None and self.left()>=self.tail()+6 and self.page_on(log):fresh=unopened(log["ids"])
  # a seller's own listings are few: list them all, so the pointed-to one is not missed on a later page, unless a row on hand
  # already carries half the words the link describes
  each=len(log.get("ids")or[]);want=_tokens(about)-_tokens(name)
  near=lambda:bool(want)and any(len(want)<=2*len(want&_tokens(_dump(_clip(self.rows.get(pid),300))))for pid in fresh)
  def more():
   if not each or log.get("token")is None:return False
   if-(-(log["total"]-len(log["ids"]))//each)>3 or self.left()<self.tail()+9:return False
   return not near()and self.page_on(log)
  while more():fresh+=unopened(log["ids"],fresh)
  self.trace+=[{"kind":"seek","name":name,"rows":fresh}]
  if not fresh:return False
  if one_by_one:return self.hop_read(name,self.shop_rows(name,about)or fresh,viewer)
  ranked=[pid for pid in self.shop_rows(name,about)if pid in fresh]
  return self.open_rows(ranked+[pid for pid in fresh if pid not in ranked])
 def hop_read(self,name,rows,viewer):
  # a pointed seller's rows are opened best match first and only so many in all; a link those opens did not reach is sought
  # with other words
  done=self.hop_reads.get(name,0);rows=rows[:max(0,HOP_READS-done)]
  before=len(self.opened);got=self.read_pages(rows,viewer)if rows else False
  self.hop_reads[name]=done+(len(self.opened)-before)
  return got
 def offer_tool(self):return self.roles.get("offer")or self.roles.get("open")
 def reread_offers(self,unread=False):
  tool=self.offer_tool();key=self.single_target(tool)
  if not(tool and key):return""
  live=[pid for pid in self.cart_ids()if not self.sold_out(pid)]
  # unread: only lines whose own page was never read, kept to the steps and seconds a pay call still has after them
  if unread:
   stale=[pid for pid in live if pid not in self.offer_seen];short=self.left()<len(stale)+1 or self.secs()<5+3*len(stale)
  else:
   # otherwise also a line whose page was last read before its own cart step, however recent that read (stock can go right after
   # carting), while the reads leave the pay its steps
   floor=max(self.step-2,self.refused_at);stale=[pid for pid in live if self.offer_seen.get(pid,-99)<max(floor,self.line_at.get(pid,-99))]
   short=self.left()<len(stale)+1+self.tail()
  if short or not stale:return""
  changes=[]
  for pid in stale:
   errs=[];args=_shape({key:pid},self.schemas.get(tool),"args",errs)
   if errs:break
   before=dict(self.rows.get(pid)or{});got=self.act(tool,args)
   if self.finished or got is None:break
   self.offer_seen[pid]=self.step;page=got.get("observation")
   self.observe(tool,page);after=self.rows.get(pid)or{}
   moved=[f for f in after if f in before and not isinstance(after[f],(dict,list))and after[f]!=before[f]]
   if moved:changes.append("%s changed (%s): %s"%(pid,", ".join(moved[:4]),_dump(_clip(page,300))[:900]))
  self.trace+=[{"kind":"reread","ids":stale,"changed":len(changes)}]
  if changes:self.version+=1
  return"\n".join(changes)
 def sold_out(self,ident):return ident in self.unavailable or ident in self.gone
 def price_swap(self):
  rose,self.price_rose=self.price_rose,False
  if not rose or self.brief.get("pick")!="cheapest"or self.secs()<12:return""
  for line in self.cart_ids():
   paid=self.price_of(line);bar=paid or 0
   pool=[x for x in self.copies([line])if x in self.full]or(self.replacements([line])if len(self.cart_ids())==1 else[]);under=[x for x in pool if x in self.full and(self.price_of(x)or bar)<bar and self.fair_swap(line,x,self.slot_of(line))]
   if not under:continue
   best=min(under,key=self.price_of)
   if not best or not self.swap_line(line,best):continue
   return"[Kyo] Price of %s rose to %s; Kyo swapped in %s at %s instead."%(line,format(paid,"g"),best,format(self.price_of(best),"g"))
  return""
 def swap_hint(self):
  dead=[pid for pid in self.cart_ids()if pid not in self.gone and self.sold_out(pid)]
  if not dead:return self.price_swap()
  for pid in dead:self.gone[pid]=self.step
  self.leads=[pid for pid in self.leads if pid not in dead];self.version+=1
  self.twin_search(dead);unread=[pid for pid in self.copies(dead)if pid not in self.full]
  if self.left()>self.tail()+5 and unread:self.open_rows(unread)
  subs=self.replacements(dead);head="[Kyo] %s can no longer be bought (the store flags it as unpurchasable), so do not open it again."%", ".join(dead)
  if not subs:return head+" Swap in the best fitting listing among the rest; failing any fitting one already read, look for one via search."
  best,*rest=subs;backup=", ".join(rest)or"none"
  twin=best in self.copies(dead);cheap=self.brief.get("pick")=="cheapest"
  if len(dead)==1 and(twin or cheap)and best in self.full and self.swap_line(dead[0],best):
   return head+" Kyo put %s in its place, %s Further fits: %s."%(best,"the same item sold by a different shop."if twin else"no cheaper fit turned up among those Kyo has read.",backup)
  if self.left()<=6:return head+" Little budget left: in one turn send both calls (%s in, unsellable item out), next turn pay. Fallbacks already read: %s."%(best,backup)
  listed=", ".join(subs)
  if not cheap:return head+" Possible stand-ins already read: %s. Right now swap the unsellable item for %s (same quantity) and complete the order."%(listed,best)
  return head+" Read listings that could replace it, cheapest first: %s. Price decides here, so unless all matching listings are known already, search once under %s's price (with the price-ceiling filter if the search has one), then put the cheapest fit in place of the dead line."%(listed,best)
 def find_sold(self):
  open_lines=[pid for pid in self.cart_ids()if pid not in self.gone]
  if len(open_lines)==1:
   only=open_lines[0]
   if not self.sold_out(only):self.unavailable.add(only)
   return""
  if not open_lines or any(map(self.sold_out,open_lines)):return""
  return self.reread_offers()
 def swap_lost(self):
  tip=self.swap_hint()
  if tip:self.queued.append(tip)
  gone_lines=[p for p in self.cart_ids()if p in self.gone]
  for lost in gone_lines:
   inside=self.cart_ids();units=self.cart_lines.get(lost,1)
   pool=self.swap_fits+(self.leads+self.carted[::-1]if len(self.items)==1 else[]);ready=[c for c in pool if self.buyable(c)and c not in inside and self.within_budget(c,units)]
   if not ready:return False
   pick=next((c for c in ready if c in self.full),ready[0])
   if not pick or not self.swap_line(lost,pick):return False
   if self.confirmed==lost:self.confirmed=pick
  return True
 def replacements(self,sold):
  if self.cost>SPEND_SOFT:return[]
  pool=self.on_hand(set(sold)|set(self.gone))[:25]
  if not pool:return[]
  lost="\n".join("{0}: {1}".format(p,self.record_of(p))for p in sold);dumps="\n".join(_dump(_clip(self.full.get(p)or self.rows.get(p),200))[:1200]for p in pool)
  sections=(("REQUEST:\n",self.request),("REQUIREMENTS IN FORCE:\n",self.ledger()),("UNAVAILABLE CART LINE:\n",lost),("CART NOW:\n",self.cart_text()),("OTHER RECORDS (NEWEST):\n",dumps))
  reply=self.ask_json("\n\n".join([SWAP]+[h+b for h,b in sections]),self.judges(),8000)or{};barred=set(sold)|set(self.gone)|set(self.cart_ids())
  fits=[]
  for entry in _items(reply.get("fits")):
   if isinstance(entry,dict):pid,price=str(entry.get("id")),entry.get("price")
   else:pid,price=str(entry),None
   if pid in barred or pid not in self.rows:continue
   barred.add(pid);rank=float("inf")
   if isinstance(price,(int,float))and not isinstance(price,bool):rank=float(price)
   fits.append((rank,pid))
  twins=self.copies(sold);cheap=self.brief.get("pick")=="cheapest"
  self.trace+=[{"kind":"swap","lost":sold,"fits":fits}];self.swap_fits=[pid for _,pid in sorted(fits,key=lambda pair:(pair[0]if cheap else 0,pair[1]not in twins))]
  return self.swap_fits[:4]
 def passed(self,lid):return any(ok and lid in key.partition("#")[0].split("|")for key,ok in self.verdicts.items())
 def unread_fits(self,printed=False):
  # per item still without a cart line: its unread buyable rows, those printing more of the item's stated values first, ties in
  # list order; printed keeps only rows printing at least one of them
  held={self.slot_of(x)for x in self.cart_ids()if not self.sold_out(x)};out=[]
  waiting=[prod for n,prod in enumerate(self.items,1)if n not in held]
  for prod in waiting:
   vals=[t for t in(_tokens(w)for w in self.word_values(prod))if t];rows=[c for c in self.item_rows(prod)if self.buyable(c)and c not in self.full and c not in self.opened]
   score=lambda c,vals=vals:sum(map(_tokens(_dump(_clip(self.rows.get(c),300))).__ge__,vals))
   out.extend(sorted([c for c in rows if score(c)or not printed],key=lambda c:-score(c))[:2])
  return out
 def bare_items(self):
  # items with no cart line of their own yet
  held={self.slot_of(x)for x in self.cart_ids()if not self.sold_out(x)}
  return[it.name for n,it in enumerate(self.items,1)if n not in held]
 def slot_of(self,lid):
  count=len(self.items)
  if count==1:return 1
  kind=self.kind_of(lid);hits={_count(k)for k,v in self.lead_items.items()if lid in v and _count(k)and 1<=_count(k)<=count}
  if not hits and kind:hits={n for n,it in enumerate(self.items,1)if kind==(it.shelf or"").strip().lower()}
  if len(hits)!=1:return None
  only,=hits
  return only
 def fits_total(self,lost,pick):
  units=self.cart_lines.get(lost,1);due=self.cart_sum()
  each=self.price_of(pick)
  if due is None or self.budget is None or each is None:return self.within_budget(pick,units)
  return due-units*(self.price_of(lost)or 0)+units*each<=self.budget
 def clean_pool(self,lost,pos,cheap=False):
  kind,inside=self.kind_of(lost),set(self.cart_ids());pool=[]
  for c in self.full:
   if c in inside or c==lost:continue
   if not(self.kind_of(c)==kind if kind else len(self.items)==1):continue
   if not self.buyable(c)or not self.fits_total(lost,c)or self.item_clash(c,pos):continue
   if not self.passed(c)and any(ok is False and key.partition("#")[0]==c for key,ok in self.verdicts.items()):continue
   pool.append(c)
  if not cheap:cheap=self.brief.get("pick")=="cheapest"
  known=self.swap_fits+self.leads
  return sorted(pool,key=lambda c:((self.price_of(c)or 0)if cheap else 0,not self.passed(c),c not in known))
 def refill_seek(self,lost,pos):
  tool=self.roles.get("search");arg=self.single_target(tool)
  if not(tool and arg)or self.left()<=self.tail()+6:return
  prod=self.items[pos-1];kind=self.kind_of(lost)
  query=self.joined_query(prod)or prod.name;answer,sent=self._kd_send(tool,{arg:query},self.finished)
  if answer is None:return
  pool=[]
  for c,_ in self.records_in(self._kd_take(tool,sent,answer)):
   if c in self.full or c==lost:continue
   if self.buyable(c)and self.fits_total(lost,c)and(not kind or self.kind_of(c)==kind):pool.append(c)
  if pool:self.open_rows(pool[:2])
  self.trace.append({"kind":"refill_seek","lost":lost,"rows":pool[:2]})
 def cart_swap(self,lost,pos,lim,cheap=False):
  rest=[p for p in self.cart_ids()if p!=lost and not self.sold_out(p)];picks=self.clean_pool(lost,pos,cheap)[:2]
  for pick in picks:
   if self.left()<=self.tail()+3 or self.cost>lim:return None
   ids=rest+[pick];self.check_cart(ids,lim)
   if not self.verdicts.get(self.check_key(ids)):continue
   if not self.swap_line(lost,pick):continue
   self.mended.add(frozenset(ids))
   if self.confirmed==lost:self.confirmed=pick
   return pick
  return None
 def repair(self):
  # a refusal or note that strikes a reviewed line, or cuts the cap below the cart, gets the cart mended before any close
  if self.finished:return{}
  if self.left()<=self.tail()+3 or self.secs()<=CLOCK_STOP:return{}
  lim=SPEND_HARD+0.01
  if not self.cost<lim:lim=self.cost+0.015
  moved={};struck=[p for p in self.cart_ids()if self.sold_out(p)and p not in self.refilled and self.passed(p)]
  for lost in struck:
   self.refilled.add(lost);pos=self.slot_of(lost)
   if not pos:continue
   if not self.clean_pool(lost,pos):self.refill_seek(lost,pos)
   pick=self.cart_swap(lost,pos,lim)
   self.trace.append({"kind":"refill","lost":lost,"pick":pick})
   if pick:moved[lost]=pick
  cap=self.cap_cut
  due=self.cart_sum()
  if cap is None or due is None or not due>cap:return moved
  if not self.cart_ids()or self.left()<=self.tail()+3:return moved
  self.cap_cut=None
  dear=max(self.cart_ids(),key=lambda p:self.cart_lines.get(p,1)*(self.price_of(p)or 0))
  pos=self.slot_of(dear)
  pick=None
  if pos:pick=self.cart_swap(dear,pos,lim,True)
  self.trace.append({"kind":"step_down","lost":dear,"pick":pick,"cap":cap})
  if pick:moved[dear]=pick
  return moved
 def mend_order(self,tool,call_args,reply):
  shown=reply.get("observation")
  self.note_sold=False
  self.observe(tool,shown)
  self.zero_lines(shown)
  if self.note_sold:self.find_sold()
  moved=self.repair()
  if not moved or self.finished or self.left()<2:return reply
  if self.left()>=3 and self.secs()>CLOCK_STOP:self.gate("order")
  self.say_basket()
  if self.left()<1 or self.finished:return reply
  again=self.act(tool,call_args)
  self.trace.append({"kind":"mended_order","moved":moved,"reply":_clip(again,200)})
  return again if again is not None else reply
 def audit(self,limit=SPEND_SOFT,room=CLOCK_EXTRAS,again=False):
  """Strong review of everything seen before an empty-handed close: likely matches per item."""
  if self.cost>limit or self.secs()<room:return""
  # a review runs once per ledger version: news that alters what is required (a dropped need, a new budget) earns a fresh one
  if self.audited and getattr(self,"audit_ver",None)==self.version and not(again and len(self.rows)>self.audit_rows):return""
  self.audited,self.audit_ver,self.audit_rows=True,self.version,len(self.rows)
  got=self.audit_call()
  slots=self.audit_fits(got)
  if not slots:
   named=(str(x)for x in _items((got or{}).get("open")))
   retry=[p for p in named if p in self.rows and p not in self.opened and not self.sold_out(p)]
   if not(retry and self.open_rows(retry)):return""
   got=self.audit_call()
   slots=self.audit_fits(got)
   if not slots:return""
  self.lead_items,self.leads=slots,sum(slots.values(),[])
  why=got.get("reason")
  shown="; ".join(f"item {k}: {', '.join(v)}" for k,v in slots.items())
  note=f" ({why})" if why else""
  return f"Reviewer's likely matches: {shown}"+note+". Check these and buy the one that truly fits; an empty close is right only after every one of them fails."
 def on_hand(self,skip=()):
  own={c for group in self.item_groups()for c in group}
  pool=[pid for pid in self.seen_order[::-1]if pid not in skip and not self.sold_out(pid)]
  rank=lambda pid:2*(pid not in self.full)+(not(pid in own and self.buyable(pid)and self.within_budget(pid,1)))
  return sorted(pool,key=rank)
 def audit_call(self):
  pool=self.on_hand()[:30]
  if not pool:return None
  shelf=[]
  for p in pool:
   page=self.full.get(p)
   if page:shelf.append("OPENED "+_dump(_clip(page,200))[:1200])
   else:shelf.append("ROW "+_dump(_clip(self.rows.get(p),120))[:500])
  sections=[AUDIT,"REQUEST:\n"+self.request,"REQUIREMENTS IN FORCE:\n"+self.ledger(),"KNOWN RECORDS (NEWEST):\n"+"\n".join(shelf)]
  got=self.ask_json("\n\n".join(sections),[HEAVY,LIGHT],8000)
  self.trace.append({"kind":"audit","got":got})
  return got
 def audit_fits(self,got):
  table=(got or{}).get("fits")
  if not isinstance(table,dict):return{}
  out,count={},len(self.items)
  for slot,raw in table.items():
   pos=_count(slot)
   mine=pos if isinstance(pos,int)and 1<=pos<=count else None
   keep=[]
   for v in map(str,_items(raw)):
    if v not in self.rows or self.sold_out(v):continue
    if mine and self.item_clash(v,mine):continue
    keep.append(v)
   out[str(slot)]=keep[:4]
  return{k:v for k,v in out.items()if v}
 def open_rows(self,unread,keep=5):
  many,tool=self.compare_target(),self.facts_tool()
  compare=self.roles.get("compare")
  if many and compare and not self.compare_bare():calls=[(compare,{many:unread[:self.compare_room()]})]
  else:
   key=self.single_target(tool)
   if not(tool and key):return False
   # room for one read but not two still reads the likeliest row, keeping the same reserve after it
   width=max(1,min(2,self.left()-keep))
   calls=[(tool,{key:pid})for pid in unread[:width]]
  if self.left()<len(calls)+keep:return False
  start=len(self.full)
  back=None
  for name,args in calls:
   errs=[]
   args=_shape(args,self.schemas.get(name),"args",errs)
   if errs:return False
   got=self.act(name,args)
   if self.finished or got is None:break
   back=got.get("observation")
   self.observe(name,back)
  added=len(self.full)-start
  sent=None
  if many:
   lead_tool,lead_args=calls[0]
   if lead_tool==self.roles.get("compare"):sent=lead_args.get(many)
  # a many-id compare that names none of its ids back marks the size the tool refuses
  if back is not None and isinstance(sent,list)and len(sent)>2:
   text=_dump(back)
   if all(str(x)not in text for x in sent):self.compare_bad=min(self.compare_bad or 999,len(sent))
  self.trace+=[{"kind":"audit_open","rows":unread,"opened":added}]
  return added>0
 def last_commit(self):
  if self.committed or len(self.items)!=1:return""
  if not all(map(self.sold_out,self.cart_ids()))or self.left()<3 or self.secs()<20:return""
  pool=self.swap_fits+self.carted[::-1]
  if self.secs()<CLOCK_EXTRAS:pool+=[p for p in self.leads if p in self.full]
  qty=self.items[0].qty
  pick=next((c for c in pool if self.buyable(c)and self.within_budget(c,qty)),None)
  if not pick:return""
  self.confirmed,self.committed,self.completed=pick,True,False
  self.lead_items={"1":[pick]}
  self.trace+=[{"kind":"last_commit","pick":pick}]
  return self.complete_cart()
 def within_budget(self,listing,units):
  cap=self.budget
  each=None if cap is None else self.price_of(listing)
  return each is None or units*each<=cap
 def cart_slots(self,props):
  specs=[(k,v)for k,v in props.items()if isinstance(v,dict)]
  texts=[k for k,v in specs if"string"in _kinds(v)and"enum"not in v]
  count=next((k for k,v in specs if"integer"in _kinds(v)),None)
  mode=next((k for k,v in specs if isinstance(v.get("enum"),list)and len(v["enum"])>=2),None)
  return texts,count,mode
 def units_fix(self,name,cargs,box):
  props=self.schemas.get(name,{}).get("properties")or{}
  _,count_f,mode_f=self.cart_slots(props)
  if mode_f or not count_f:return cargs
  n=_count(cargs.get(count_f))
  if n is None or n<0:return cargs
  ids=[v for v in cargs.values()if isinstance(v,str)and v in self.rows]
  if len(ids)!=1:return cargs
  pid,=ids
  kind=self.kind_of(pid)
  sets=self.cart_sets(name)
  if n==0 and(pid in self.carted or pid in self.cart_lines or pid in self.cart_ids()or not sets):return cargs
  if len(self.items)==1:hits=self.items
  else:
   hits=[]
   for k,v in self.lead_items.items():
    pos=_count(k)
    if pid in v and pos and 1<=pos<=len(self.items):hits.append(self.items[pos-1])
   if not hits:hits=[it for it in self.items if kind and kind==(it.shelf or"").strip().lower()]
  if len(hits)!=1:return cargs
  item,=hits
  goal=item.qty-(0 if sets else self.cart_lines.get(pid,0))
  if goal<=0 or goal==n:return cargs
  top=(props.get(count_f)or{}).get("maximum")
  if isinstance(top,int)and top<goal:return cargs
  box["extra"]+="\n[Kyo] %s is the unit count, not a line or item number: sent %s=%s so %s holds the %s units its item asks for."%(count_f,count_f,goal,pid,item.qty)
  return dict(cargs,**{count_f:goal})
 def cart_calls(self,wanted,removals):
  tool=self.roles.get("cart")or""
  spec=self.schemas.get(tool)or{}
  props=spec.get("properties")or{}
  for field,fs in props.items():
   if not(isinstance(fs,dict)and"array"in _kinds(fs)):continue
   inner=fs.get("items")
   if not isinstance(inner,dict)or not isinstance(inner.get("properties"),dict):continue
   texts,count_f,_=self.cart_slots(inner["properties"])
   if not texts:return None
   return[{field:[{texts[0]:c,**({count_f:n}if count_f else{})}for c,n in wanted.items()]}]
  texts,count_f,mode_f=self.cart_slots(props)
  named=[f for t,f in self.id_args if t==tool and f in texts]
  needed=[f for f in spec.get("required")or[]if f in texts]
  firsts=[ids[0]for ids in(named,needed)if ids]
  if len(texts)==1:firsts.append(texts[0])
  id_f=next((f for f in firsts if f),None)
  if not id_f:return None
  ops=[]
  for pid in removals:
   have=self.cart_lines.get(pid,1)
   op={id_f:pid}
   if mode_f:
    op[mode_f]=props[mode_f]["enum"][1]
    if count_f:op[count_f]=have
   elif not count_f:return None
   elif self.cart_sets(tool):op[count_f]=0
   else:
    low=props[count_f].get("minimum")
    if not(isinstance(low,int)and low<=-have):return None
    op[count_f]=-have
   ops.append(op)
  for pid,goal in wanted.items():
   have=0
   if pid in self.cart_ids():have=self.cart_lines.get(pid,0)
   if have==goal:continue
   if have and not self.cart_sets(tool):return None
   op={id_f:pid}
   if mode_f:op[mode_f]=props[mode_f]["enum"][0]
   if count_f:op[count_f]=goal
   ops.append(op)
  return ops
 def zero_lines(self,reply):
  # a refusal naming a held id beside a zero or false value, and no positive one, marks that id sold out
  held=set(self.cart_ids())|set(self.cart_lines)
  nil=lambda v:v is False or(not isinstance(v,(bool,str))and _count(v)==0)
  plus=lambda v:v is True or(not isinstance(v,(bool,str))and(_count(v)or 0)>0)
  out,stack=set(),[(reply,0)]
  while stack:
   node,depth=stack.pop()
   if depth>6:continue
   if isinstance(node,list):
    stack.extend((v,depth+1)for v in node)
    continue
   if not isinstance(node,dict):continue
   vals=list(node.values())
   if not any(map(plus,vals))and any(map(nil,vals)):out.update(v.strip()for v in vals if isinstance(v,str)and v.strip()in held)
   out.update(k for k,v in node.items()if nil(v)and k in held)
   stack.extend((v,depth+1)for v in vals)
  self.unavailable.update(out)
  return out
 def basket_plan(self):
  shelf=self.category_key()
  if shelf:tag=lambda p:str((self.rows.get(p)or{}).get(shelf)or"").strip().lower()
  else:tag=lambda p:p
  chosen={}
  for slot,ids in self.lead_items.items():
   n=_count(slot)
   pick=next((c for c in ids if not self.sold_out(c)),None)
   if not(pick and n and 1<=n<=len(self.items)):continue
   if tag(pick)in{tag(c)for c in chosen}:continue
   chosen[pick]=self.items[n-1].qty
  if self.confirmed not in chosen:return None
  held=self.cart_ids()
  used={tag(c)for c in chosen}
  stay={}
  for p in held:
   if p in chosen or self.sold_out(p)or tag(p)in used:continue
   stay[p]=self.cart_lines.get(p,1)
  if len(chosen)+len({tag(p)for p in stay})<len(self.items):return None
  basket=dict(stay)
  basket.update(chosen)
  return basket,[p for p in held if p not in chosen and p not in stay]
 def stand_doubts(self,goal):
  # a must neither named by a key nor shown in the record's own words is a doubt
  ids=list(goal)
  for prod in self.items:
   for nd in prod.musts+prod.told:
    if nd in prod.dropped or"only if"in nd:continue
    prop,_,want=nd.partition(":")
    words=_tokens(want)
    pages=[self.full.get(p)or{}for p in ids]
    named=lambda pg:any(_names(_stem(prop),_stem(k))for k in(_fact_block(pg)or{}))
    if not any(named(pg)or(words and words<=_tokens(_dump(pg)))for pg in pages):return True
  return False
 def proven_pick(self):
  # a basket the second opinion approved on this version, still buyable, clean and in budget
  if len(self.items)==1:
   qty,now=self.items[0].qty,str(self.version)
   for key,ok in self.verdicts.items():
    if not ok:continue
    pid,_,ver=key.rpartition("#")
    if ver!=now or"|"in pid or pid not in self.full:continue
    if self.buyable(pid)and not self.sold_out(pid)and self.within_budget(pid,qty)and not self.item_clash(pid,1):return{"1":[pid]},pid
  got=self.proven_goal
  if not got or got[0]!=self.version:return None
  table=got[1]
  ids=[v[0]for v in table.values()if v]
  total=sum((self.price_of(p)or 0)*self.items[_count(n)-1].qty for n,v in table.items()for p in v[:1]if _count(n))
  if not ids:return None
  if not all(self.buyable(p)and not self.sold_out(p)for p in ids):return None
  if self.budget is not None and not total<=self.budget:return None
  return table,got[2]
 def checkout_ready(self,goal):
  self.reread_offers()
  mended=frozenset(goal)in self.mended
  if not(self.standing or mended):self.check_cart(list(goal),SPEND_HARD)
  if not(self.standing or mended or self.verdicts.get(self.check_key(list(goal)))):return False
  if self.left()<1 or self.finished or any(map(self.sold_out,goal)):return False
  due=self.cart_sum()
  return due is None or self.budget is None or not due>self.budget
 def complete_cart(self):
  """Cart the confirmed pick in code and check out."""
  cart_tool,buy_tool=self.roles.get("cart"),self.roles.get("buy")
  if self.completed or not self.confirmed:return""
  if self.sold_out(self.confirmed)or not(cart_tool and buy_tool):return""
  self.completed=True
  plan=self.basket_plan()
  if not plan:return""
  goal,drop=plan
  ops=self.cart_calls(goal,drop)
  if not ops:return""
  need=len(ops)+1
  if self.left()<need:return""
  self.fill_facts(list(goal),need)
  if self.standing:
   # a stand pick that leaves a must unprinted gets the same second opinion as any other basket
   if self.stand_doubts(goal):
    self.check_cart(list(goal),SPEND_HARD+0.02)
    if not self.verdicts.get(self.check_key(list(goal))):return""
  else:
   self.check_cart(list(goal),SPEND_HARD)
   if not self.verdicts.get(self.check_key(list(goal))):return""
   self.proven_goal=self.version,dict((k,list(v))for k,v in self.lead_items.items()),self.confirmed
  sent=[]
  for op in ops:
   errs=[]
   line=_shape(op,self.schemas.get(cart_tool),"args",errs)
   back=None if errs else self.act(cart_tool,line)
   if back is None or self.finished:return""
   self.note_cart(cart_tool,line)
   self.observe(cart_tool,back.get("observation"))
   sent.append(_dump(line))
  errs,heard=[],[]
  pay=_shape(dict(self.brief.get("buy_args")or{}),self.schemas.get(buy_tool),"args",errs)
  if errs:return""
  for again in(False,True):
   if again and not self.checkout_ready(goal):break
   cleared,note=self.gate("order")
   if note:heard.append(note)
   if not cleared:break
   self.say_basket()
   got=self.act(buy_tool,pay)
   self.trace+=[{"kind":"completed","lines":goal,"drop":drop,"reply":_clip(got,200)}]
   if got is None:break
   self.buys+=1
   done,said=got.get("finished"),got.get("observation")
   if not done:self.refused_at=self.step
   self.note_sold=False
   self.observe(buy_tool,said)
   if not done:
    self.zero_lines(said)
    if self.note_sold:self.find_sold()
    swaps=self.repair()
    goal=dict((swaps.get(p,p),n)for p,n in goal.items())
   heard.append(self.show(said))
   if done or self.finished:break
  if heard:return"Kyo finished this on its own: %s passed review two, the cart was arranged via (%s), and the order went to the store, which answered: %s"%(self.confirmed,"; ".join(sent)," | next: ".join(heard))
  return""
 def _sve_is_end(self,role,name,args):
  if name!=self.roles.get(role):return False
  wants=self.brief.get(role+"_args")
  if not isinstance(wants,dict):return True
  return all(args.get(k)==v for k,v in wants.items())
 def is_buy(self,name,args):return self._sve_is_end("buy",name,args)
 def is_refuse(self,name,args):return self._sve_is_end("refuse",name,args)
 def offered(self):
  r=self.roles
  left,keep=self.left(),self.tail()
  usable=[t for t in self.tools if r.get("ask")!=t["function"]["name"]]
  ends={r.get(x)for x in("buy","refuse")}-{None}
  if not ends:return usable
  idle=self.buys==0
  if left<=keep:tier=0
  elif left<=keep+3:tier=1
  elif idle and left<=keep+6 and len(self.cart_ids())>=len(self.items):tier=1
  elif idle and left<=len(self.items)+5:tier=2
  elif self.secs()<CLOCK_STOP+25:tier=1
  else:return usable
  allowed=set(ends)
  allowed.update(r.get(x)for x in((),("cart","cart_view"),("cart","cart_view","offer"))[tier])
  reader=self.facts_tool()
  if tier==2 and reader and self.facts_from.get(reader):
   offer=r.get("offer")
   if self.facts_from.get(offer)is False:allowed.discard(offer)
   if left>len(self.items)+keep+1:allowed.add(reader)
  picked=[t for t in usable if t["function"]["name"]in allowed]
  return picked or usable
 def quiet(self,tool_name,args):
  spec=self.schemas.get(tool_name)or{}
  need=spec.get("required")or[]
  out=dict(args)
  # a request wanting the near miss offered keeps the closing words that name it
  fallback=self.brief.get("nearest_fallback")is True
  offered=fallback and self.brief.get("quiet_decline")is not True
  for f,fs in(spec.get("properties")or{}).items():
   plain=isinstance(fs,dict)and"enum"not in fs and"string"in _kinds(fs)
   if not plain:continue
   val=out.get(f)
   if val is None and f not in need:continue
   if str(val or""):
    if offered:continue
    if self.brief.get("quiet_decline")is False and not any(p in str(val or"")for p in self.rows):continue
   out[f]="Every checked listing failed at least one stated condition, so Kyo closes empty-handed."[:fs.get("maxLength",200)]
  return out
 def id_from_text(self,raw):
  if not isinstance(raw,str):return raw,""
  text=raw.strip()
  if text in self.rows:return text,""
  if len(text)<3:return raw,""
  low=text.lower()
  bare=" "not in text and bool(re.search(r"\d",text))
  shaped=bare and any(abs(len(p)-len(text))<=3 and p[:2]==text[:2]for p in self.rows)
  # an id cut short or run long matches the one seen id it overlaps; a code printed in a row names that row; a name its title
  hits=[]
  if shaped:
   if len(text)>=6:hits=[p for p in self.rows if(p.startswith(text)or text.startswith(p))and abs(len(p)-len(text))<=3]
  else:
   table=[(p,[v.strip().lower()for k,v in rec.items()if isinstance(v,str)and k!=self.id_key])for p,rec in self.rows.items()]
   if bare:
    mark=re.compile(r"(?<![0-9a-z])"+re.escape(low)+r"(?![0-9a-z])")
    hits=[p for p,vals in table if any(map(mark.search,vals))]
   else:
    hits=[p for p,vals in table if low in vals]
    if not hits and len(low)>=8:hits=[p for p,vals in table if any(v.startswith(low)for v in vals)]
  if hits:
   if len(hits)>1:return raw,"%r could mean several listings (%s); pass a single id"%(raw,", ".join(hits[:6]))
   return hits[0],""
  seen=shaped and(text in self.page_ids or text in self.request or text in _dump(self.view.get("opening")))
  return(raw,"")if seen else(raw,"%r is not an id any store reply has shown; copy the %s field of a row exactly"%(raw,self.id_key))
 def resolve_ids(self,tool_name,args):
  r=self.roles
  if not self.id_key:return args,""
  if tool_name not in{r.get(x)for x in("cart","open","offer","compare")}-{None}:return args,""
  props=(self.schemas.get(tool_name)or{}).get("properties")or{}
  free=[k for k,v in props.items()if isinstance(v,dict)and"enum"not in v and("string"in _kinds(v)or"array"in _kinds(v))]
  if len(free)!=1:return args,""
  f,=free
  if tool_name!=r.get("cart")and(tool_name,f)not in self.id_args and f not in(self.single_target(tool_name),self.compare_target()):return args,""
  if f not in args:return dict(args),""
  sent=args[f]
  many=isinstance(sent,list)
  got=[self.id_from_text(x)for x in(sent if many else[sent])]
  issue=next((p for _,p in got if p),"")
  if issue:return args,issue
  return dict(args,**{f:[v for v,_ in got]if many else got[0][0]}),""
 def cart_sets(self,name):
  specs=[v for v in(self.schemas.get(name,{}).get("properties")or{}).values()if isinstance(v,dict)]
  modes=[v for v in specs if isinstance(v.get("enum"),list)and len(v["enum"])>1]
  return not modes and any(v.get("minimum")==0 and"integer"in _kinds(v)for v in specs)
 def note_cart(self,tool_name,args):
  self.cart_touched,self.cart_at=True,self.step
  was=dict(self.cart_lines)
  props=self.schemas.get(tool_name,{}).get("properties")or{}
  field=lambda k:props.get(k)or{}
  removing=any(isinstance(field(k).get("enum"),list)and len(field(k)["enum"])>=2 and v in field(k)["enum"][1:]for k,v in args.items())
  number=lambda x:None if isinstance(x,str)else _count(x)
  loose=next((number(v)for v in args.values()if number(v)is not None),None)
  wipe,moves=False,[]
  for k,v in args.items():
   if isinstance(v,str):
    if v in self.rows:moves.append((v,loose))
    continue
   if not isinstance(v,list):continue
   inner=field(k).get("items")or{}
   if not v and isinstance(inner,dict)and(inner.get("type")=="object"or"properties"in inner):wipe=True
   for e in v:
    if isinstance(e,str):
     if e in self.rows:moves.append((e,loose))
    elif isinstance(e,dict):
     wipe=True
     vals=list(e.values())
     pid=next((x for x in vals if isinstance(x,str)and x in self.rows),None)
     n=next((number(x)for x in vals if number(x)is not None),1)
     if pid:moves.append((pid,n))
  if wipe:self.cart_lines=dict((p,n)for p,n in moves if n and n>0)
  sets=self.cart_sets(tool_name)
  for pid,n in[]if wipe else moves:
   have=self.cart_lines.get(pid,0)
   step=1 if n is None else n
   if removing:have=have-n if n else 0
   elif sets:have=step
   else:have+=step
   if have>0:self.cart_lines[pid]=have
   else:self.cart_lines.pop(pid,None)
  # a line's own cart step: the call that put it in the cart or moved its quantity
  self.line_at.update((p,self.step)for p,n in self.cart_lines.items()if was.get(p)!=n)
 def show(self,observation):
  text=_dump(_clip(observation,SHOW_CHARS))
  return text if len(text)<=OBS_LIMIT else text[:OBS_LIMIT]+"...(cut)"
 def compare_target(self):
  spec=self.schemas.get(self.roles.get("compare")or"")or{}
  first=next((k for k,v in(spec.get("properties")or{}).items()if isinstance(v,dict)and"array"in _kinds(v)),None)
  if not first:return None
  return first if all(n==first for n in spec.get("required")or[])else None
 def compare_room(self):
  props=(self.schemas.get(self.roles.get("compare")or"")or{}).get("properties")or{}
  good,bad=self.compare_ok,self.compare_bad
  room=max(2,min(bad-1,good))if bad else max(READ_CAP,good)
  limit=(props.get(self.compare_target()or"")or{}).get("maxItems")
  return min(room,limit)if isinstance(limit,int)else room
 def _svf_screen(self,call,box):
  r=self.roles.get
  usable={spec["function"]["name"]for spec in self.offered()}
  if call.get("joined"):return"Merged: this open rode along in the side-by-side call placed earlier in this same turn, costing one step in total."
  if self.finished:return"Not run: the episode is already over."
  no=lambda why:_SVF_FREE+why
  cart_no=lambda why:no(why)+_SVF_CART+self.cart_text()
  fn=call.get("function")or{}
  name=fn.get("name")
  if name==r("ask"):return no("shopper questions are Kyo's job, not a tool call.")
  if name not in usable or name not in self.schemas:
   return no(("remaining moves are held back; this tool is locked now"if name in self.schemas else"unknown tool name")+". Usable now: "+", ".join(sorted(usable))+".")
  def fit(a):
   errs=[]
   a=_shape(a,self.schemas[name],"args",errs)
   return a,errs and no("; ".join(errs[:6]))
  raw=fn.get("arguments")
  if isinstance(raw,dict):cargs=raw
  elif not(raw or"").strip():cargs={}
  else:
   cargs=_json_obj(raw)
   if cargs is None:return no("arguments could not be read; send a single JSON object.")
  seeking=name in self.seekers()
  if seeking:
   sweep=self.shelf_sweep(name,cargs)
   if sweep:return no(sweep)
  cargs,err=fit(cargs)
  if err:return err
  cargs,wrong=self.resolve_ids(name,cargs)
  if wrong:return no(wrong+".")
  many=None
  if name==r("compare"):many=self.compare_target()
  room=self.compare_room()
  sent=many and cargs.get(many)
  if isinstance(sent,list)and room<len(sent):
   box["extra"]="\n[Kyo] Comparison limited to %d listings here; only the first %d went in. Send these in another call: %s."%(room,room,", ".join(str(x)for x in sent[room:]))
   cargs=dict(cargs,**{many:sent[:room]})
  key=self.single_target(name)
  if key and name in(r("open"),r("offer")):
   lid=str(cargs.get(key))
   if lid in self.gone:return no("%s dropped out of stock at step %s for good; pick another fitting listing to fill that line."%(cargs[key],self.gone[lid]))
  is_cart=name==r("cart")
  if is_cart:
   if any(not isinstance(v,str)and(_count(v)or 0)<0 for v in cargs.values()):
    seen=_dump(self.cart_view)if self.cart_seen_at>=self.cart_at else""
    off=next((v for v in cargs.values()if isinstance(v,str)and v in self.rows and v not in self.cart_lines and v not in seen and v not in self.cart_said),None)
    if off is not None:return cart_no("%s is not in the cart, so there is nothing to take off."%off)
   cargs=self.units_fix(name,cargs,box)
   ids=[v for v in cargs.values()if isinstance(v,str)and v in self.rows]
   if box["count"]>1 and len(ids)==1 and self.cart_sets(name):
    one,=ids
    if one in box["held"]:return cart_no("%s went into the cart earlier in this same reply with the units its item asks for; no second call needed."%one)
    box["held"].add(one)
  cargs=self.true_label(name,cargs,box)
  cargs=self.budget_cap(name,cargs)
  cargs=self.first_page(name,cargs,box)
  if seeking:
   cargs=self.valued_words(name,self.vet_filters(name,cargs),box)
   gap=self.mode_gap(name,cargs)
   if gap:return no(gap)
  if not is_cart:
   cargs,err=fit(cargs)
   if err:return err
  elif box["count"]==1 and self.loses_last_line(name,cargs):return cart_no("too little budget remains to replace this item, so it is kept; go ahead and pay.")
  final=self.is_buy(name,cargs)or self.is_refuse(name,cargs)
  if final:
   if box["count"]>1:return no("Kyo holds back an order or decline sent alongside other calls; review these outputs, then issue it solo.")
  elif self.left()<=self.tail()and(r("buy")or r("refuse")):return no("the steps still left are reserved for finishing: either pay (with the shopper's basket approval first when needed) or close with no purchase.")
  mark=name+_dump(cargs)
  if not(final or is_cart)and mark in self.recent:return no("Kyo already ran exactly this a moment ago and state is unchanged; its result sits above.")
  readers={r(x)for x in("search","browse","open","offer","compare")}|{self.facts_tool()}
  readers.discard(None)
  hit=self.read_cache.get(mark)
  if hit and hit[0]==self.version and name in readers:return no("same call as at step %s, nothing changed; repeating that output:\n"%hit[1]+hit[2][:3000])
  pins,tf=self.mode_pins.get(name)or{},self.text_fields.get(name)
  if tf and self.blind.get(name)is True and str(cargs.get(tf)or"").strip():
   if any(cargs[f]!=v for f,v in pins.items()if f in cargs):return no("in this search mode the store ignores the words: different words already listed the same rows. Send the words in the word-search mode (%s) instead."%", ".join("%s=%s"%kv for kv in pins.items()))
  if seeking:
   if self.answered.get(name+_dump(dict(self.fixed_args.get(name,{}),**cargs)))==self.version:return no("this exact search was already answered in this episode and nothing changed since; its rows are on file. Change the words, the filter or the page.")
  if key:
   lid=str(cargs.get(key))
   if name in readers and name!=r("compare")and self.shown_already(name,lid):return no("%s has nothing new to show here unless its state moves. Stored record: "%(cargs[key],)+self.record_of(lid))
   if name==self.facts_tool()and self.fixed_page(lid):return no("%s was read already and its attributes stay as they were. Stored record: "%(cargs[key],)+self.record_of(lid))
  sent=many and cargs.get(many)
  if isinstance(sent,list)and sent:
   fresh=[x for x in sent if not self.shown_already(name,str(x))]
   if not fresh:return no("every listing here was already fully shown by this view and is unchanged. On file:\n"+"\n".join("%s: %s"%(x,self.record_of(str(x)))for x in sent)[:4000])
   if len(fresh)<len(sent):
    box["extra"]+="\n[Kyo] Left %s out of the call: already fully shown, unchanged since."%", ".join(str(x)for x in sent if x not in fresh)
    cargs=dict(cargs,**{many:fresh})
    mark=name+_dump(cargs)
  box.update({"name":name,"args":cargs,"mark":mark,"many":many,"ending":final,"readers":readers})
  return None
 def filter_args(self,tool,cargs):
  props=(self.schemas.get(tool)or{}).get("properties")or{}
  skip={self.text_fields.get(tool),self.page_args.get(tool),*self.mode_pins.get(tool,{})}
  out=[]
  for a,v in cargs.items():
   spec=props.get(a)
   if a in skip or not isinstance(spec,dict):continue
   if _dump(v)in self.tokens_seen or{"number","integer"}&set(_kinds(spec)):continue
   out.append(a)
  return out
 def store_words(self):
  said={_kb_norm(v):v for row in self.rows.values()for v in row.values()if isinstance(v,str)and 0<len(v)<=80}
  said.update((_kb_norm(v),v)for v in self.labels())
  if len(self.rows)<3:return{}
  return said
 def valued_words(self,tname,cargs,box):
  # an item's first word search that names none of its stated values goes out as the item's noun and rarest value words (three
  # in all): in races 12-17 a search carrying a stated value listed a later-paid row two to three times as often as one without
  tf=self.text_fields.get(tname)
  said=cargs.get(tf)if tname==self.roles.get("search")and tf else None
  if not isinstance(said,str)or not said.strip()or self.code_query({tf:said}):return cargs
  have=_tokens(said)
  kin=lambda p:{w for w in _tokens(p.name)|_tokens(self.title_noun(p))|_tokens(p.shelf)if len(w)>=4}
  hits=[p for p in self.items if have&kin(p)]
  if len(hits)!=1 or hits[0].name in self.valued:return cargs
  vals=list(dict.fromkeys(self.word_values(hits[0])))
  if not vals or any(_tokens(v)&have for v in vals):return cargs
  q=self.lead_query(hits[0],self.value_query(vals[:3]).split())
  out={**cargs,tf:q}
  if not q or _tokens(q)<=have or self.answered.get(tname+_dump({**self.fixed_args.get(tname,{}),**out}))==self.version:return cargs
  self.valued.add(hits[0].name)
  box["valued"]=dict(cargs)
  box["extra"]+="\n[Kyo] Searched with %s=%r: the item's name now carries its stated values, because a bare name rarely brings up a row that fits."%(tf,q)
  return out
 def vet_filters(self,tool,cargs):
  # a free-text filter value goes in the form the store printed it; an unseen value is left for the store to answer
  props=(self.schemas.get(tool)or{}).get("properties")or{}
  said=self.store_words()
  single=self.single_target(tool)
  out=dict(cargs)
  for a in self.filter_args(tool,cargs):
   val=cargs[a]
   if a==single or"enum"in props[a]or not isinstance(val,str)or not val.strip():continue
   out[a]=said.get(_kb_norm(val),val)
  return out
 def mode_gap(self,tool,cargs):
  # the search tool moved off its text mode with no filter value lists nothing
  pins=self.mode_pins.get(tool)or{}
  if all(f not in cargs or cargs[f]==v for f,v in pins.items()):return""
  if any(str(cargs[a]).strip()for a in self.filter_args(tool,cargs)):return""
  cat=self.category_key()
  seen=sorted({str(r.get(cat))for r in self.rows.values()if r.get(cat)})[:12]if cat else[]
  extra="; store categories seen: "+", ".join(seen)if seen else""
  return"this search mode needs a category or other filter value and none was given%s."%extra
 def budget_cap(self,tname,cargs):
  # a shelf browse naming a category, with the top-price slot left empty, carries what the budget leaves per unit of that item
  # once the cart's other lines are paid: rows priced past it can never be bought, and they fill the short pages ahead of rows that can
  if tname!=self.roles.get("browse")or self.budget is None:return cargs
  names=self.brief.get("browse_args")or{}
  la,ca=names.get("label"),names.get("cap")
  if not la:return cargs
  label=str(cargs.get(la)or"").strip()
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  if not label or ca in cargs or ca not in props:return cargs
  cat=self.category_key()
  want=_kb_norm(label)
  shelf=lambda lid:_kb_norm((self.rows.get(lid)or{}).get(cat)or"")if cat else""
  own=[it for it in self.items if _kb_norm(it.shelf)==want]
  others=sum((self.price_of(lid)or 0)*n for lid,n in self.cart_lines.items()if n>0 and shelf(lid)!=want)
  per=own[0].qty if len(own)==1 else 1
  room=(self.budget-others)/max(1,per)
  if room<=0:return cargs
  # the budget and the store's prices must plainly share a unit: some listing already seen is priced inside the room
  if not any((self.price_of(lid)or room+1)<=room for lid in self.rows):return cargs
  return dict(cargs,**{ca:int(room)if"integer"in _kinds(props[ca])else round(room,2)})
 def true_label(self,tname,cargs,box):
  if tname!=self.roles.get("browse"):return cargs
  arg=(self.brief.get("browse_args")or{}).get("label")
  field=self.category_key()
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  if not(arg and field)or arg not in props:return cargs
  labels={str(rec.get(field))for rec in self.rows.values()if rec.get(field)}
  given=str(cargs.get(arg)or"").strip()
  if given:
   key=_kb_norm(given)
   printed=next((t for t in labels if _kb_norm(t)==key),None)
   if printed==given:return cargs
   if printed:tag=printed
   else:
    words=_stem(given)
    best=min(labels,key=lambda t:(-len(_stem(t)&words),t),default="")
    tag=best if best and _stem(best)&words else""
    if not tag:return cargs
    box["extra"]+="\n[Kyo] Swapped in %r, the matching store category."%tag
  elif tname==self.roles.get("search"):return cargs
  else:
   home=self.cart_ids()or self.seen_order[:1]
   if not home:return cargs
   tag=str((self.rows.get(home[0])or{}).get(field)or"")
   if not tag:return cargs
   box["extra"]+="\n[Kyo] Swapped in %r, the matching store category."%tag
  return dict(cargs,**{arg:tag})
 def first_page(self,tname,cargs,box):
  pages=[rec["page_arg"]for rec in self.search_log.values()if rec["tool"]==tname]
  arg=pages[0]if pages else self.page_args.get(tname)
  if arg not in cargs:return cargs
  rest=dict(cargs)
  given=rest.pop(arg)
  same=lambda a:{k:_kb_norm(v)if isinstance(v,str)else v for k,v in a.items()}
  rec=next((r for r in self.search_log.values()if r["tool"]==tname and same(r["args"])==same(rest)),None)
  if rec and(_dump(given)in self.tokens_seen or given in(0,1,"")):return cargs
  if rec and rec.get("token")is not None:
   # known words with a start the store never handed out go on from where the store says they continue
   box["extra"]+="\n[Kyo] Used %s=%s instead: that is where the store said these words continue."%(arg,_dump(rec["token"]))
   return{**cargs,arg:rec["token"]}
  box["extra"]+="\n[Kyo] Removed %s: a new search starts from page one."%arg
  return rest
 def loses_last_line(self,tname,cargs):
  if len(self.cart_lines)>len(self.items):return False
  if self.left()>self.tail()+(2 if self.buys else 4):return False
  line=next((v for v in cargs.values()if isinstance(v,str)and v in self.cart_lines),None)
  if not line or self.sold_out(line):return False
  props=(self.schemas.get(tname)or{}).get("properties")or{}
  if not any(isinstance((props.get(k)or{}).get("enum"),list)and v in props[k]["enum"][1:]for k,v in cargs.items()):
   if not any((_count(v)or 0)<=0 for v in cargs.values()if not isinstance(v,str)and _count(v)is not None):return False
  return bool(self.within_budget(line,self.cart_lines[line]))
 def _ky_hold_pay(self):
  cart=lambda:_SVF_CART+self.cart_text()
  for probe in(self.sweep,self.wider_hold):
   if(said:=probe()):return said+cart()
  redo=self.reread_offers()
  vtool=self.roles.get("cart_view")
  stale=self.cart_touched and self.cart_at>self.cart_seen_at
  if vtool and self.left()>3 and stale:
   errs=[]
   view_args=_shape({},self.schemas.get(vtool),"args",errs)
   if not errs and(got:=self.act(vtool,view_args))is not None:
    self.cart_view,self.cart_seen_at=got.get("observation"),self.step
    if(shift:=self.observe(vtool,self.cart_view)):
     self.queued=[]
     return _SVF_HELD+"Fresh cart contents differ from before: "+shift+cart()+"\nHandle this first, changing the cart if it calls for that; only then check out."
  tip=self.swap_hint()
  if tip:return _SVF_HELD+tip+cart()
  lost=[x for x in self.cart_ids()if x in self.gone]
  if lost and self.held>=2:self.swap_lost()
  elif lost:
   self.held+=1
   return _SVF_HELD+"%s sold out yet sits in the cart: substitute that line and check out."%", ".join(lost)+cart()
  chain,stuck=self.trail()
  if stuck:return _SVF_HELD+chain+cart()
  looked=self.fill_facts(self.cart_ids(),self.tail()+3)+self.cheaper_rows()
  verdict=self.check_cart(limit=SPEND_HARD if self.brief.get("pick")=="cheapest"else SPEND_SOFT)
  if verdict and not self.take_better():
   return"Held. "+chain+("Fetched for this check, one step each:\n"+"\n".join(looked)+"\n"if looked else"")+verdict+cart()
  if self.brief.get("nearest_fallback")is True and self.flawed==self.check_key(self.cart_ids()):
   near=self.audit(SPEND_HARD,CLOCK_STOP+20,True)
   if near and set(self.leads)-set(self.cart_ids()):return _SVF_HELD+near+cart()
  if redo and self.cost>SPEND_SOFT:return _SVF_HELD+"Fresh look at every cart line, just taken:\n"+redo+"\nFirst swap any line now unavailable or priced past the budget; with all lines clean, retry checkout."
  return None
 def _ky_hold_review(self):
  if(said:=self.audit(SPEND_HARD,CLOCK_STOP+20)):return _SVF_HELD+said
  # no review may overrule a stated value the code sees broken: leads that clash with the item's own needs are not pressed
  if len(self.items)==1:self.leads=[x for x in self.leads if not self.item_clash(x,1)]
  if not self.leads or self.left()<=4 or self.refuse_blocks not in(0,1):return None
  phase=self.refuse_blocks
  self.refuse_blocks+=1
  if phase==0:
   self.trace.append(dict(kind="refuse_blocked",leads=list(self.leads)))
   return _SVF_HELD+"Candidates to review: "+", ".join(self.leads)+". For each, read more if unsure; buy the one the pick rule favours by carting it and checking out. A no-purchase exit needs every one of them to show a plain breach - quote it in your message and send the ending call once more."
  lead=self.leads[0]
  self.check_cart([lead],SPEND_HARD)
  if lead in self.leads and self.verdicts.get(self.check_key([lead])):
   self.confirmed=lead
   self.trace.append(dict(kind="refuse_blocked",confirmed=lead))
   return _SVF_HELD+"Independent review of %s: its records break no requirement. Add it at the requested quantity and check out; nothing but an outright contradiction within its record justifies a no-purchase finish."%lead
  return None
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
  if any(isinstance(e:=(props.get(k)or{}).get("enum"),list)and len(e)>=2 and v in e[1:]for k,v in args.items()):return""
  ids=[v for v in args.values()if isinstance(v,str)and v in self.cart_lines]
  units=[n for n in map(_count,(v for v in args.values()if not isinstance(v,(str,list,dict))))if n is not None]
  if len(ids)!=1:return""
  pos=self.slot_of(ids[0])
  if pos is None or units and units[0]<=0:return""
  have=self.cart_lines[ids[0]]
  if have<self.items[pos-1].qty:return""
  return"%s already holds %s units, all its item asks for, and this cart adds quantities up: %s more would overshoot, so nothing was sent."%(ids[0],have,units[0]if units else 1)
 def _svf_hold(self,slot):
  """Ordered guards on one model call, tried in plan order: the first that holds returns its reason; None lets the executor run it."""
  tname,targs,ending=slot["name"],slot["args"],slot["ending"]
  buying,declining=self.is_buy(tname,targs),self.is_refuse(tname,targs)
  finds,cart=self.code_finds,lambda:_SVF_CART+self.cart_text()
  tag=lambda:" [episode finished]"if self.finished else""
  if tname in self.seekers()and self.code_query(targs)and finds.get("zero",0)>=2 and not finds.get("hit")and finds.get("held",0)<3:
   finds["held"]=1+finds.get("held",0)
   self.trace.append(dict(kind="code_held",args=_clip(targs,80)))
   return _SVF_HELD+"%d searches naming a model code found nothing in this store, while word searches list rows: it does not search by codes. Search the product kind or list its category, then read the codes its rows and pages print."%finds["zero"]
  carting=tname==self.roles.get("cart")
  if carting and getattr(self,"zero_held",0)<3:
   zero=[k for k,v in targs.items()if not isinstance(v,(str,list,dict))and _count(v)==0]
   fresh=[v for v in targs.values()if isinstance(v,str)and v in self.rows and v not in self.cart_ids()]
   if zero and fresh:
    self.zero_held=1+getattr(self,"zero_held",0)
    swap=" The cart already has a line for every item, so a swap also needs the old line set to 0 units."if len(self.cart_ids())>=len(self.items)else""
    return _SVF_HELD+"%s is not in the cart, and %s=0 asks for zero units of it, so the call would add nothing. %s is the unit count: send the quantity wanted to add it.%s"%(", ".join(fresh),zero[0],zero[0],swap)+cart()
  if carting and getattr(self,"over_held",0)<3 and(over:=self.overfill(tname,targs)):
   self.over_held=1+getattr(self,"over_held",0)
   return _SVF_HELD+over+cart()
  if declining and not self.full_held:
   lines=[x for x in self.cart_ids()if not self.sold_out(x)]
   if self.left()>=2 and len(lines)>=len(self.items):
    self.full_held=True
    self.trace.append(dict(kind="full_cart_held",lines=lines))
    return _SVF_HELD+"Checkout, not an exit: each item asked for has its own cart line, and quitting while a match exists pays 0. The ending call may be resent solely when a record plainly proves one of those lines fails a requirement."
  pay=ending and buying
  if pay and self.left()>2 and(broken:=self.range_hold()):return _SVF_HELD+broken+cart()
  if declining and(auto:=self.last_commit()):return auto+tag()
  roomy=self.secs()>CLOCK_EXTRAS
  if pay and self.left()>5 and not roomy and self.secs()>CLOCK_STOP+15 and self.open_lookups():
   before=len(self.ref_values)
   chain,stuck=self.trail()
   if stuck or before<len(self.ref_values):return _SVF_HELD+chain+cart()
  # a pay call reads each cart line's own page once first, also when the clock or the steps are too short for the full review
  if pay and(redo:=self.reread_offers(True)):
   return _SVF_HELD+"Each cart line's own page was just read once:\n"+redo+"\nReplace any line now unavailable or past the budget, then check out."+cart()
  held=None
  if ending:
   if buying and self.left()>2 and roomy:held=self._ky_hold_pay()
   elif self.left()>5 and roomy:held=self._ky_hold_review()
   # the clock closed the full pre-pay review before any cart was judged: the last check still runs once
   elif buying and self.left()>2 and not self.verdicts and self.secs()>CLOCK_STOP+15:
    held=self.check_cart(limit=SPEND_HARD)
    held=held and"Held. "+held+cart()
  if held:return held
  if declining and not self.late_held and self.left()<=5 and(held:=self._ky_hold_late()):return held
  # a many-item order is pushed twice, leads or not: most real-race declines of order tasks came with a quarter or more of the steps left
  many=len(self.items)>1
  if declining and self.pushed<1+many and(many or not self.leads)and self.left()>self.max_steps//3 and self.secs()>CLOCK_EXTRAS:
   self.pushed+=1
   bare=self.bare_items()
   return(_SVF_HELD+"%d steps left over. Giving up now is premature%s: page through the categories of all items asked for (resend the store's token or offset each time), search with two or three words (one stated value plus the product kind), try alternate labels for the property, and open whatever could fit; give up only then."
     %(self.left(),"; still without a cart line: "+", ".join(bare)if bare else""))
  # an item whose fitting listing was listed but never read is lost by closing: its likeliest unread rows are read once first
  # with less room only one item may lack its line, and the reads leave a cart and a pay call (or a second close)
  wide=self.left()>=self.tail()+6
  if declining and not self.last_reads and self.secs()>CLOCK_STOP+25 and(wide or self.left()>=self.tail()+3 and len(self.bare_items())==1):
   self.last_reads=True
   rows=self.unread_fits(printed=not wide)
   if rows and(self.open_rows(rows)if wide else self.open_rows(rows,self.tail()+2)):return _SVF_HELD+"Before an empty close Kyo opened unread listings of the items still without a cart line, those whose rows print most of the item's stated values (%s); their pages are in the records now. Cart one that meets every requirement, else close again."%", ".join(rows[:2])+cart()
  if declining:
   self.last_stand()
   if(auto:=self.complete_cart()):return auto+tag()
   slot["args"]=targs=self.quiet(tname,targs)
  if ending and self.left()>=2 and roomy:
   ok,word=self.gate("order"if self.is_buy(tname,targs)else"end")
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
 def _ky_resend(self,slot,tname,targs,seen,done):
  """Compare batch check: a reply naming none of the sent ids marks that size refused, and the head the store takes goes again."""
  sent=targs.get(slot["many"])if slot["many"]else None
  if not isinstance(sent,list)or len(sent)<2:return seen,done
  size=len(sent)
  if any(str(x)in _dump(seen)for x in sent):
   self.compare_ok=max(self.compare_ok,size)
   return seen,done
  self.compare_bad=min(self.compare_bad or 999,size)
  room=self.compare_room()
  # the refused batch is resent at once in the size the store takes, so the turn still reads listings
  if room<size and not(self.finished or done)and self.left()>self.tail()+3:
   head=list(sent)[:room]
   again=self.act(tname,{**targs,slot["many"]:head})
   if again is not None and any(str(x)in _dump(again.get("observation"))for x in head):
    self.compare_ok=max(self.compare_ok,room)
    seen,done=again.get("observation"),again.get("finished")
  slot["extra"]+=f"\n[Kyo] Your {size}-id compare request came back without any of those listings; Kyo resent the first {room}. From now on pass {room} ids or fewer."
  return seen,done
 def _ky_row_one(self,slot,tname,targs,seen,done,paged):
  """A start slot learned from this very reply is said once; a reply of the rest or of nothing goes again from the first row."""
  learned,self.start_learned=self.start_learned,None
  if not(paged and learned and learned[0]==tname and learned[3]==_dump(targs)):return seen,done,targs
  arg,kind=learned[1],learned[2]
  key=(self.shapes.get(tname)or{}).get("next")
  word="\n[Kyo] %s names the row a result list starts at; it is no count. %s=%s passed over every row before that one (the reply's %s holds where the next page starts). From here on, fresh words are sent from the first row."%(arg,arg,_dump(targs[arg]),key or"next value")
  base={k:v for k,v in targs.items()if k!=arg}
  again=self.act(tname,base)if kind!="past"and not(self.finished or done)and self.left()>self.tail()+3 else None
  if again is not None:
   seen,done,targs=again.get("observation"),again.get("finished"),base
   self.absorb(seen)
   self.log_search(tname,targs,seen)
   word+=" The same words went out again from the first row, and this is what came back."
  slot["extra"]+=word
  return seen,done,targs
 def _svf_execute(self,slot):
  tname,targs,final,readers=slot["name"],slot["args"],slot["ending"],slot["readers"]
  reply=self.act(tname,targs)
  if reply is None:return"Empty response from the store (connection or server issue); you may retry."
  seen,done=reply.get("observation"),reply.get("finished")
  back=slot.get("valued")
  if back and not self.records_in(seen)and not(self.finished or done)and self.left()>self.tail()+3:
   # the value words listed nothing: the model's own words go out once instead
   if(again:=self.act(tname,back))is not None:
    seen,done,targs=again.get("observation"),again.get("finished"),back
    slot["args"]=targs
    slot["extra"]+=" Nothing came back for them, so your own words went out unchanged; this is their reply."
  seen,done=self._ky_resend(slot,tname,targs,seen,done)
  self.recent=[*self.recent,slot["mark"]][-3:]
  self.id_args.update((tname,param)for param,given in targs.items()
    if(ids:=given if isinstance(given,list)else[given])and all(isinstance(x,str)and x in self.rows for x in ids))
  self.absorb(seen)
  if tname in self.seekers():
   self.note_sweep(tname,targs,seen)
   slot["extra"]+=self.cap_note(tname,targs,seen)
  # some stores find nothing by model codes: count what code searches return, once rows are read at all
  if tname in self.seekers()and self.rows and self.code_query(targs):
   bucket="hit"if self.records_in(seen)else"zero"
   self.code_finds[bucket]=1+self.code_finds.get(bucket,0)
  if tname==self.roles.get("compare")and(found:=self.records_in(seen)):
   self.facts_from[tname]=self.facts_from.get(tname,False)or any(_fact_block(rec)for _,rec in found)
  if tname in readers-{self.roles.get("search")}:self.note_view(tname,seen)
  paged=not final and tname not in{t for name,t in self.roles.items()if name not in("search","browse")}
  if paged:self.log_search(tname,targs,seen)
  seen,done,targs=self._ky_row_one(slot,tname,targs,seen,done,paged)
  more=self.more_pages(tname,targs,seen)if paged else""
  busy=tuple(map(self.roles.get,("cart","buy","refuse","ask")))
  hit=None
  if isinstance(seen,dict)and tname not in busy and self.single_target(tname):hit=self.ident(seen)
  if hit:
   shown=bool(_fact_block(seen))
   if not shown:self.thin_reads.add("%s %s"%(tname,hit))
   thin=sum(r.startswith(tname+" ")for r in self.thin_reads)
   # one thin page says little about the tool: it counts as showing no attributes only from the second such page on
   has_facts=self.facts_from.get(tname,False)or shown or(None if thin<2 else False)
   self.facts_from[tname]=has_facts
   opener,skip=self.roles.get("open"),busy+(self.roles.get("search"),)
   moved=(next((t for t in self.schemas if t!=tname and self.single_target(t)and self.facts_from.get(t)is not False and t not in skip),None)
     if tname==opener and not has_facts else None)
   lone=not has_facts and tname!=(moved or opener)and not self.roles.get("offer")
   if moved:self.roles["open"]=moved
   if lone:self.roles["offer"]=tname
   if tname==self.offer_tool():self.offer_seen[hit]=self.step
  qty=[n for v in targs.values()if not isinstance(v,str)and(n:=_count(v))is not None]
  is_cart=tname==self.roles.get("cart")
  if is_cart:self._ky_cart_effects(tname,targs,seen,qty)
  if tname==self.roles.get("cart_view"):self.cart_view,self.cart_seen_at=seen,self.step
  buy=self.is_buy(tname,targs)
  if buy:
   self.buys+=1
   if not done:self.refused_at,_=self.step,self.zero_lines(seen);self.version+=1
  refused=buy and not done
  note=self.show(seen)+more+slot["extra"]
  if paged:
   note+=self.cheapest_here([seen]+self.paged_obs)
   if tname==self.roles.get("search")and self.brief.get("pick")in("cheapest","most","least"):note+=self.glued_numbers(targs)
  cacheable=note
  extra=[e for e in(self.evidence(seen),)if e]
  if is_cart and 0 in qty and any(isinstance(v,str)and v in self.rows and _in_rows(seen,v)for v in targs.values()):extra.append("[Kyo] A zero quantity changed nothing: this cart adds quantities up, so take units off with a negative number.")
  self.note_sold=False
  gist=self.take_note(loose)if(loose:=self.loose_lines(seen))else""
  if gist:extra.append("[note read] "+gist+": the ledger now carries this and it binds like the shopper's own words, whichever tool result brought it; change your picks to suit it ahead of checkout.")
  sold=refused and(self.note_sold or any(line in self.spoken(seen)for line in self.sold_lines))
  # a refused checkout whose note moves only a price gets its lines read again, so the swap choice sees the fresh price; any other
  # note that refused a checkout gets the same re-read before the next one (race 16: a page re-read before checking out again
  # passed .66 of the time, a checkout sent straight back .42)
  rose=gist and refused and not sold and self.price_rose==2
  was={p:self.price_of(p)for p in self.cart_ids()}if rose else{}
  if sold:recheck=self.find_sold()
  else:recheck=self.reread_offers()if gist and refused else""
  if recheck:extra.append("[checkout refused - fresh status of the cart lines]\n"+recheck)
  moved=self.repair()if not done and(buy or self.cap_cut is not None)else{}
  if moved:extra.append("[Kyo] Cart mended in code: "+"; ".join("%s out, %s in"%pair for pair in moved.items())+". Each new line passed review and keeps the sum inside the budget; check out now.")
  if(tip:=self.swap_hint()):extra.append(tip)
  due=self.cart_sum()
  steady=lambda p:self.offer_seen.get(p,-99)>=self.refused_at and self.buyable(p)and(self.price_of(p)or 0)<=(was.get(p)or 0)
  if rose and not(moved or tip)and due is not None and(self.budget is None or due<=self.budget)and all(map(steady,self.cart_ids())):
   extra.append("[Kyo] Kyo read every cart line again after this refusal: none costs more than before, and the cart now comes to %g, inside the budget. A moved price breaks no requirement, so keep the cart as it is and send checkout again."%due)
  if self.queued:
   extra.append("[more notes found by Kyo's own calls] "+" | ".join(self.queued))
   self.queued=[]
  if tname in readers:self.read_cache[slot["mark"]]=(self.version,self.step,cacheable)
  body="\n".join([note,*extra])
  return body+" [episode finished]"if done else body
 def _ky_close_verdict(self,leg,reply,again):
  if leg=="end":return None if reply is None or self.finished else True
  return None if self.left()<=0 or self.finished else bool(again and reply is not None)
 def finish(self,again=False):
  """Forced ending as planned close legs: an order leg when a live cart covers the items, then the empty-handed leg."""
  if self.left()<=0 or self.broke or self.finished:return""
  if self.left()>2 and(self.cap_cut is not None or any(map(self.sold_out,self.cart_ids()))):self.repair()
  if self.left()>2 and any(map(self.sold_out,self.cart_ids())):self.swap_lost()
  if self.finished or self.left()<=0:return""
  fresh=self.cart_view is not None and self.cart_seen_at>=self.cart_at
  lines=self.cart_ids()if fresh else self.cart_lines
  dead=bool(lines)and all(map(self.sold_out,lines))
  # a forced order on a cart short of the units asked for pays nothing: the empty-handed exit goes instead
  live={x:n for x,n in self.cart_lines.items()if not self.sold_out(x)}
  short=bool(live)and len(live)<len(self.items)and sum(live.values())<sum(_count(it.qty)or 1 for it in self.items)
  self.trace+=[d for hit,d in((dead,dict(kind="dead_cart",lines=list(lines))),(short,dict(kind="short_cart",lines=dict(self.cart_lines))))if hit]
  can_pay=not(dead or short)and(lines or(self.cart_touched and not fresh and self.left()>=2))
  legs=([("order",self.roles.get("buy"),"buy_args","forced_buy")]if can_pay else[])+[("end",self.roles.get("refuse"),"refuse_args","forced_refuse")]
  if not can_pay and not self.proven_tried and self.left()>3:
   self.proven_tried=True
   got=self.proven_pick()
   # a reviewer's opened match that the loop ran out of turns to act on still gets its independent check before the empty exit
   if not got and len(self.items)==1:
    units=self.items[0].qty
    lead=([x for x in self.leads if x in self.full and self.buyable(x)and not self.sold_out(x)and self.within_budget(x,units)and not self.item_clash(x,1)]or[None])[0]
    if lead:
     self.check_cart([lead],self.cost+0.02)
     got=self.proven_pick()
   if got:
    self.lead_items,self.confirmed,self.completed,self.standing=got[0],got[1],False,False
    self.trace.append(dict(kind="proven_close",pick=got[1]))
    self.complete_cart()
    if self.finished:return""
  for leg,tool,args_key,label in legs:
   if not tool:continue
   faults,ordering,fixed=[],leg=="order",dict(self.brief.get(args_key)or{})
   if not ordering:fixed=self.quiet(tool,fixed)
   call_args=_shape(fixed,self.schemas.get(tool),"args",faults)
   if faults:continue
   if self.secs()>CLOCK_STOP and self.left()>=2:self.gate(leg)
   if ordering:self.say_basket()
   if self.finished or self.left()<=0:return""
   reply=self.act(tool,call_args)
   self.trace.append(dict(kind=label,reply=_clip(reply,200)))
   if ordering and reply is not None and not self.finished:reply=self.mend_order(tool,call_args,reply)
   verdict=self._ky_close_verdict(leg,reply,again)
   if verdict is None:return""
   if not verdict:continue
   shown=None if reply is None else reply.get("observation")
   self.note_sold=False
   if ordering:self.buys,self.refused_at=self.buys+1,self.step
   self.observe(tool,shown)
   if ordering:
    self.zero_lines(shown)
    if self.note_sold:self.find_sold()
    self.reread_offers()
   return{"order":"payment","end":"empty-handed close"}[leg]+" bounced, reply: "+self.show(shown)
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
  """Read the brief, plan the turns, close; a refused close earns one short extra round."""
  if self.finished:return self.trace
  menu="\n".join("- %s: %s | parameters: %s"%(f["name"],f.get("description",""),_dump(f.get("parameters")or{}))for f in(e["function"]for e in self.tools))
  intake="%s\nTOOLS:\n%s\n\nREQUEST:\n%s"%(BRIEF,menu,self.request)
  if self.view.get("opening")is not None:intake+="\n\nSTORE'S OPENING MESSAGE:\n"+_dump(_clip(self.view.get("opening"),1500))
  first=self.ask_json(intake,[LIGHT],8000)or{}
  roles=first.get("roles")
  if self.no_credit or(_items(first.get("items"))and isinstance(roles,dict)and roles.get("buy")):reading=first
  else:reading=self.ask_json(intake,[HEAVY,LIGHT],8000)or first
  self.brief=reading
  roles=reading.get("roles")
  self.roles={k:v for k,v in(roles.items()if isinstance(roles,dict)else())if isinstance(v,str)and v in self.schemas}
  # a second page tool the reading left without a role: the only unassigned tool whose one input names a single listing
  spare=[t for t in self.schemas if t not in self.roles.values()and self.single_target(t)]
  if len(spare)==1 and self.roles.get("open")and not self.roles.get("offer"):self.roles["offer"]=spare[0]
  tool=self.roles.get("search")
  spec=self.schemas.get(tool or"")or{}
  props=spec.get("properties")or{}
  wanted=reading.get("search_text")
  slot=props.get(wanted.get("text"))if isinstance(wanted,dict)else None
  if isinstance(slot,dict)and"enum"not in slot and"string"in _kinds(slot):
   modes=wanted.get("mode")
   pins={f:v for f,v in(modes.items()if isinstance(modes,dict)else())if v in((props.get(f)or{}).get("enum")or[])}
   self.text_fields[tool],self.fixed_args[tool]=wanted["text"],pins
   self.mode_pins[tool]=dict(pins)
   self.schemas[tool]={**spec,"required":[f for f in spec.get("required")or[]if f not in pins]}
  for key in("buy_args","refuse_args","browse_args"):
   if not isinstance(reading.get(key),dict):reading[key]={}
  bp=(self.schemas.get(self.roles.get("browse")or"")or{}).get("properties")or{}
  kind=lambda a,want:isinstance(bp.get(a),dict)and bool(want&set(_kinds(bp[a])))
  ba=reading["browse_args"]
  if ba:
   if not kind(ba.get("label"),{"string"}):ba["label"]=None
   if not kind(ba.get("cap"),{"number","integer"}):
    nums=[a for a in bp if kind(a,{"number"})]
    ba["cap"]=None if len(nums)!=1 else nums[0]
  dual=self.roles.get("buy")
  pay,close=reading["buy_args"],reading["refuse_args"]
  if dual and dual==self.roles.get("refuse")and not(pay and close and pay!=close):
   got=self.ask_json((f"Reply with exactly one JSON object holding two keys, \"buy_args\" and \"refuse_args\"; each maps to the fixed arguments "
      "that pick one mode, nothing else. \"buy_args\" = the mode that pays for the order; \"refuse_args\" = the mode that ends "
      f"the visit with nothing bought.\nWhy: the store's tool {dual} does both jobs and its arguments decide which. Schema of {dual}: "
      +_dump(self.schemas.get(dual))),[LIGHT,HEAVY],600)or{}
   modes=got.get("buy_args"),got.get("refuse_args")
   if all(isinstance(m,dict)for m in modes)and modes[0]!=modes[1]:reading["buy_args"],reading["refuse_args"]=modes
  for raw in filter(lambda r:isinstance(r,dict),_items(reading.get("items"))):
   n=_count(raw.get("quantity"))
   self.items.append(Item(str(raw.get("name")or"the product"),str(raw.get("shelf")or""),n if n and n>0 else 1,[str(t)for t in _items(raw.get("musts"))if t]))
  money=reading.get("budget")
  if isinstance(money,str)and re.fullmatch(r"\s*\d+(\.\d+)?\s*",money):money=float(money)
  money_ok=isinstance(money,(int,float))and not isinstance(money,bool)and money>0
  talk,walk=_count(reading.get("messages")),_count(reading.get("steps"))
  talk=None if talk is None or talk<0 else talk
  walk=walk if walk is not None and 10<=walk<self.max_steps else None
  flow=[{"step":label,"when":when if(when:=str(raw.get("when")or"").strip().lower())in("choose","order","end")else"order",
    "messages":max(0,_count(raw.get("messages"))or 0)}for raw in _items(reading.get("process"))if isinstance(raw,dict)and(label:=str(raw.get("step")or"").strip())]
  if not self.items:self.items=[Item("the requested product","",1,[])]
  pool=[m for it in self.items for m in it.musts]
  stem=lambda s:_words(s.split(":")[0])
  for ptr in _items(reading.get("lookups")):
   text=str(ptr).strip()
   if text not in pool:text=next((m for m in pool if stem(m)==stem(text)),text)
   at=next((n for n,it in enumerate(self.items,1)if text in it.musts),1)
   if not text or(at,text)in self.refs or self.sibling_ref(at,text):continue
   self.refs.append((at,text))
  if not self.refs:self.find_pointers()
  self.unmix()
  self.settle_musts(reading)
  self.fuse_shelf()
  if money_ok:self.budget=money
  if talk is not None:self.msg_limit,self.cap_stated=talk,True
  elif self.roles.get("ask"):self.msg_limit=3
  if walk is not None:self.max_steps=walk
  self.process.extend(flow)
  if self.roles.get("ask"):
   # a step due at checkout or ending may have to reach the shopper: one message stays held even where the reading counts none
   owed=[p["messages"]or 1 for p in self.process if p["when"]!="choose"]
   # with two messages or fewer every one is worth more as a question than as a closing note
   self.own_statement=not owed and self.msg_limit>=3 and not isinstance(reading.get("process"),list)
   self.reserve=1 if self.own_statement else(min(max(owed),self.msg_limit)if owed else 0)
  self.trace.append(dict(kind="brief",brief=self.brief))
  self.chat=[dict(role="system",content=GUIDE),dict(role="user",content="\n\n".join(self._ky_opening()))]
  self.work(LOOP_CALLS,SPEND_HARD)
  # a spend cap reached just after the requirements changed still leaves a short allowance to act on the change
  if(self.version>0 and self.step-self.ver_seen.get(self.version,self.step)<=6 and not(self.finished or self.broke)and self.cost>=SPEND_HARD
    and self.left()>self.tail()+4 and self.secs()>CLOCK_STOP+20):self.work(self.loop_calls+4,self.cost+0.015)
  if len(self.cart_ids())<len(self.items)and not self.finished:
   self.last_stand()
   self.complete_cart()
  bounce="" if self.finished else self.finish(True)
  if bounce and self.left()>1 and not(self.finished or self.broke or self.no_credit):
   self.chat.append(dict(role="user",content="[Kyo] Wrap-up refused, with this reason: %s\nFix that and wrap up once more.\n%s"%(bounce,self.status())))
   # a refusal past the retry cap still gets a short allowance to act on what it says
   low=self.cost<SPEND_HARD+0.01
   self.work(self.loop_calls+(10 if low else 4),SPEND_HARD+0.01 if low else self.cost+0.015)
  if not self.finished:self.finish()
  self.trace.append(dict(kind="end",finished=self.finished,steps=self.step,cost=round(self.cost,5),calls=self.calls))
  return self.trace
 def late_look(self):
  if self.late_looked or self.carted or len(self.items)!=1 or(self.secs()>CLOCK_EXTRAS+50 and self.left()>LATE_LOOK+self.tail()):return""
  self.late_looked=True
  return self.audit(SPEND_HARD)
 def _ky_stand_picks(self,got,live):
  raw=got.get("picks")
  # a pick keyed by the item's name (or "item 1") instead of its number names the same line
  named={(it.name or"").strip().lower():str(n)for n,it in enumerate(self.items,1)}
  picked={}
  for k,v in(raw.items()if isinstance(raw,dict)else()):
   key=str(k).strip()
   if not _count(key):
    end=(key.split()or[""])[-1]
    key=named.get(key.lower())or(end if _count(end)else key)
   picked[key]=v
  # a pick given as a code a record prints, not an id, names the one seen record printing it as a whole token
  def own(code):
   if code in self.rows or not _id_like(code):return code
   mark=re.compile(r"(?<![0-9a-z])"+re.escape(code.strip().lower())+r"(?![0-9a-z])")
   hits=[p for p in self.rows if mark.search(_dump([self.rows[p],self.full.get(p)]).lower())]
   return hits[0]if len(hits)==1 else code
  choice,total={},0
  for n,it in enumerate(self.items,1):
   c=choice[n]=own(str(picked.get(str(n))or""))
   if c not in live or self.item_clash(c,n):return None
   total+=(self.price_of(c)or 0)*it.qty
  return choice if self.budget is None or total<=self.budget else None
 def last_stand(self):
  """Before leaving empty-handed: one strong pick among the opened listings."""
  near=self.brief.get("nearest_fallback")is True
  if self.stood or self.finished or(not near and len(self.items)!=1):return
  if self.cost>SPEND_HARD+(0.07 if near else 0.02)or self.secs()<15 or self.left()<2:return
  self.stood=True
  # a listing opened on a page that prints no fact block (its row, overview and pointers only) is still an opened record
  pool=[x for x in dict.fromkeys(self.cart_ids()+self.on_hand())if x in self.rows and(x in self.full or x in self.opened)and self.buyable(x)][:24]
  if not pool:return
  records="\n".join("OPENED "+_dump(_clip(self.full.get(x)or self.rows[x],200))[:1200]for x in pool)
  got=self.ask_json("\n\n".join([STAND,"REQUEST:\n"+self.request,"REQUIREMENTS IN FORCE:\n"+self.ledger(),"PICK RULE: "+str(self.brief.get("pick")),"RECORDS:\n"+records]),
    [LIGHT]if self.cost>SPEND_HARD else[LIGHT,HEAVY],8000)or{}
  self.trace.append({"kind":"last_stand","got":got})
  choice=self._ky_stand_picks(got,pool)
  if choice is None:return
  self.standing,self.completed,self.confirmed=True,False,choice[1]
  self.lead_items={str(n):[c]for n,c in choice.items()}
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
  """Planner-executor turn loop: the plan lists step records (gate, act, status tail, stop rule) tried in order before the model
  thinks; the executor runs each and feeds its text back into the chat; a model turn then runs its calls through _ky_after_calls."""
  talk,idle=self.chat,0
  plan=((None,self.fuse_shelf,True,""),(self.ready_to_ask,self.interview,True,"always"),(None,self.late_look,True,""),(None,self.clock_buy,False,"hit"))
  while not(self.finished or self.broke)and self.left()>0 and self.loop_calls<loop_cap and self.cost<spend_cap and self.secs()>CLOCK_STOP:
   stop=False
   for gate,act,tail,rule in plan:
    if gate and not gate():continue
    said=act()
    if said:talk.append(dict(role="user",content=said+"\n"+self.status()if tail else said))
    if rule=="always"or(rule and said):
     stop=True
     break
   if stop:continue
   self._ky_trim(talk)
   self.loop_calls+=1
   answer=self.think(talk,[LIGHT,SPARE],4000,self.offered())
   if answer is None:
    self.misses+=1
    if self.no_credit or self.calls>=ALL_CALLS or self.misses>2:break
    continue
   outgoing=[]
   for pos,call in enumerate(answer.get("tool_calls")or[]):
    if not isinstance(call,dict):continue
    fn=call.get("function")
    fn=fn if isinstance(fn,dict)else{}
    raw=fn.get("arguments")
    if isinstance(raw,str):raw=_json_obj(raw)if raw.strip()else{}
    outgoing.append({"id":str(call.get("id")or"k%d_%d"%(self.loop_calls,pos)),"type":"function",
      "function":{"name":str(fn.get("name")or""),"arguments":_dump(raw if isinstance(raw,dict)else{})}})
   said=answer.get("content")or""
   text=_dump(said)if not isinstance(said,str)else said
   if outgoing:
    idle=self._ky_after_calls(talk,outgoing,text)
    continue
   idle+=1
   talk+=[dict(role="assistant",content=text or"(no call)"),dict(role="user",content="Your reply had no tool call. Callable at this point: "+", ".join(t["function"]["name"]for t in self.offered())+"; pick one.\n"+self.status())]
   if idle>=3:break
 def _ky_after_calls(self,talk,outgoing,text):
  talk.append(dict(role="assistant",content=text or None,tool_calls=outgoing))
  r=self.roles.get
  binding={r("cart"),r("buy"),r("refuse")}-{None}
  if not self.asked and r("ask")and self.msgs_free()>0 and self.left()>8 and any(c["function"]["name"]in binding for c in outgoing):
   talk+=[dict(role="tool",tool_call_id=c["id"],content="Not executed, no cost: first the interviewer talks to the shopper.")for c in outgoing]
   talk.append(dict(role="user",content=(self.interview()or"Interviewer: skipped, no useful question.")+"\n"+self.status()))
   return 0
  self.trace.append(dict(kind="turn",text=text[:300],calls=[c["function"]for c in outgoing]))
  opener,comparer,key,many=r("open"),r("compare"),self.single_target(r("open")),self.compare_target()
  batch,picks=outgoing,[]
  if opener and comparer and key and many and opener!=comparer and not self.compare_bare():
   picks=[(i,pid)for i,c in enumerate(batch)if(fn:=c.get("function")or{}).get("name")==opener and isinstance(fn.get("arguments"),str)
     and(parsed:=_json_obj(fn["arguments"]))and set(parsed)=={key}and(pid:=str(parsed.get(key)or"").strip())in self.rows and pid not in self.gone][:self.compare_room()]
  if len(picks)>1:
   lead,spots,ids=picks[0][0],dict(picks),[p for _,p in picks]
   batch=[{"id":c["id"],"type":"function","function":{"name":comparer,"arguments":_dump({many:ids})}}if i==lead else{"id":c["id"],"joined":True}if i in spots else c
     for i,c in enumerate(batch)]
   self.trace+=[dict(kind="joined",ids=ids)]
  held,words=set(),[]
  for call in batch:
   box={"extra":"","count":len(batch),"held":held}
   word=self._svf_screen(call,box)
   if word is None:word=self._svf_hold(box)
   words.append(self._svf_execute(box)if word is None else word)
  if words:words[-1]+="\n"+self.status()
  talk.extend(dict(role="tool",tool_call_id=c["id"],content=w)for c,w in zip(batch,words))
  return 0
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
