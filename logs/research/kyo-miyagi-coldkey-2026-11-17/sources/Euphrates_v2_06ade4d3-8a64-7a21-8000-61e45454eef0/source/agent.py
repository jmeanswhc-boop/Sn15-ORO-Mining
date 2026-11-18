from __future__ import annotations
import enum
import json
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient
DRIVER_MODEL = 'anthropic/claude-haiku-5.5'
READING_MODEL = DRIVER_MODEL
LARGE_MODEL = 'anthropic/claude-sonnet-5.5'
CHECKER_MODEL = LARGE_MODEL
SPARE_MODEL = 'openai/gpt-6-luna'
SECOND_VENDOR_MODEL = 'google/gemini-3.8-flash'
EPISODE_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
LAST_SECONDS = 28.0
MAX_TURNS = 80
SPEND_SOFT = 0.05
SPEND_FINISH = 0.065
SPEND_STOP = 0.085
TEXT_LIMIT = 16000
_ROOM_FACTOR = {SPARE_MODEL: 4, SECOND_VENDOR_MODEL: 2}
_BRIEF_FAULTS = frozenset([429, 500, 502, 503, 504])
_RETRY_PAUSES = (1.5, 4.0)
_STORE_FINAL_FAULTS = frozenset([403, 422, 500, 503, 504])
_COMPLETIONS_PATH = '/inference/chat/completions'
_INTAKE_TRIES = (([LARGE_MODEL, SECOND_VENDOR_MODEL, DRIVER_MODEL, SPARE_MODEL], 1500), ([LARGE_MODEL, SECOND_VENDOR_MODEL, SPARE_MODEL], 4500), ([LARGE_MODEL, SECOND_VENDOR_MODEL, SPARE_MODEL, DRIVER_MODEL], 4500))
_ROLE_MODELS = [LARGE_MODEL, DRIVER_MODEL, SPARE_MODEL]
_model_proxy = ProxyClient(retry_delay=1, max_retries=1, timeout=70)
_store_proxy = ProxyClient(retry_delay=1, max_retries=2, timeout=95)
DRIVER_GUIDE = 'THE JOB\nYou are a shopping agent serving a single shopper in a simulated store. Nothing happens except through the store\'s tools, so put at least one tool call in each reply. Any words you add should be few: which conditions currently apply, which listings you have confirmed (their price and stock), and your next move.\n\nHOW CALLS COUNT\n- A step is used by every call you send - including a call that fails, gets an error back, or has arguments its schema rejects. Reasoning is free of charge; steps are limited.\n- Independent calls (for instance reading several listings) may share a reply; they are carried out in the written order. Send an order or a decline in a reply that contains nothing else.\n- Some results begin with "Not run, no step used" or "Paused by your own pre-send check". Those come from checks inside your own agent; the store never received the call. Treat them as your own second thoughts (nobody else wrote them) and follow them, unless you can name a stated requirement or a revealed need that they overlook. If the request is for several different products, a fitting listing named in such a note applies to its own product only.\n- Stores differ in tool names, argument names and the layout of results. Go by each tool\'s description and JSON schema, and copy argument names and permitted values character for character. Your first few results tell you which tool gives which facts (price, stock, detailed properties).\n\nSEARCHING\n- Searches may be restricted in ways you have to discover: few visible rows or pages, ordering by id instead of relevance or price, all words required, filters active in only one mode. Hold the count a search reports against the rows you can actually see, and keep tightening (more specific words, a category or price filter, later pages) until nothing relevant is left unseen.\n- Find out soon which category label the store gives the wanted product type; it appears on listings. If several labels are close, take the one matching the wording of the request as a whole, and do both searching and buying within it.\n\nTHE MATCHING RULES (they decide the score)\n1. Proof comes from the listing\'s own property for the condition. The same property may be written in many forms (spaces, dots, underscores, prefixes) and may appear under several keys; any key naming it is acceptable. Judge values by meaning: ignore case, accept close synonyms and singular or plural, and accept a single member of a comma-separated list. When a condition is a value followed by an explanation in brackets, the value alone has to be met; the bracketed part just explains it. A separate qualifier placed in front of the wanted word (a shade or a finish of a colour) still counts as stating it. These do not count: a word that only begins with the same letters; a single compound word containing the wanted word but naming something narrower ("tealwood" is no proof of "wood"); a broader class or a different class ("girls" does not give "women", a plain "unisex" does not give "female").\n2. Title and description may serve as proof only when the listing has no key whatsoever for that condition; in that case the variant or option key also counts if it carries the complete wanted value, since it names the exact version sold ("options: teal" answers a wish for teal). A listing whose own property gives a different value fails, no matter what its title or description says. A word in a foreign language, or something you infer from other facts, is not a stated value.\n3. One property cannot stand in for another (a length tells you nothing about the style; the fabric of the lining tells you nothing about the main material).\n4. Numbers: first bring them to the same unit, then compare numerically. Hitting a limit exactly is within it. A range qualifies only if the whole range satisfies the limit. "covers N" requires the item\'s own declared range to reach N; a range ending below N does not qualify. If the request words such a range in exactly the form this store\'s listings use (a figure plus words), search for that phrase; a listing that shows the number only in another form (other units, separate grades) does not qualify.\n5. Respect closed lists ("only A or B") and exclusions ("nothing that contains X"). An exclusion is satisfied only when the listing\'s own key for the property shows some other value; a silent listing proves nothing, so choose a fit that states the property and use a silent one only when no stating fit can be reached.\n6. The product type is given by the category, so no property needs to repeat the category name. When listings are filed under category labels, the label decides the type: anything under the label that matches the full wording of the request is that product, even if its title or text sounds like a spare part, an accessory or something else entirely, and no other label will do (asked for "A or B", go to a category "A & B", not to its narrower sibling "B"). Only listings shown as available qualify, and price times quantity may not exceed the budget (equal is allowed).\n7. A comma list containing the required option is a match. However, a key named after the property itself (any spelling, singular or plural) that holds just one other value disqualifies the listing, even when another key lists the wanted value (several widths under one key but one different width under an identically named key: no match). A list under a longer key with a different name (such as available widths) does not count against the listing.\n\nPICKING THE WINNER\n- Words like "ideally", "would be nice" or "if possible" signal wishes rather than conditions; use them only to decide between listings that already satisfy everything, and never against a cheapest rule.\n- "Best" by a measure (most, largest, longest, highest...): line up all fitting listings on that measure in the same way and take the extreme; a listing without a value for the measure cannot be chosen.\n- "Cheapest": finding one fit is not enough; you must buy the lowest-priced listing among ALL that meet every condition. Make sure all candidate rows have been displayed (more pages or narrower searches whenever the reported total exceeds what you see), compare prices yourself and verify from the lowest price up. Also run a search on the required value by itself (only its words, inside the section if possible) and read it through: other titles word things differently, and longer searches overlook listings. When every cheaper listing has been checked and rejected, the first complete fit is your answer and the ones after it are your backups.\n\nTALKING TO THE SHOPPER\n- Requests never spell out all needs: a hidden need surfaces only if a message asks about its property. Ask about exactly ONE property per message, in a short, plain, friendly question using the words of the listings\' keys (give example values only to separate it from a similar neighbouring key). Do not ask about the listing as such (title, maker, category, description) or about a key that collects the variants for sale or several features at once. No instructions, no mention of lists, labels, positions or rules, and never two questions in one message.\n- When the request leaves a decision that only the shopper can make - usually two wishes that might not be possible together, and which one matters more - ask before choosing.\n- If you raise questions on your own, follow this order: open the details of two or three listings first; then ask about what the story points to (who it is for, how it looks, where or how it will be used); finally about listing properties whose values vary among the candidates. Leave out anything the request has already decided, and keep within the message allowance.\n- When a planner is active it picks and sends the questions as soon as you have opened two or three listings of the wanted type; so open them early, and before putting anything in the cart whenever messages are allowed: ordering with messages left unused overlooks needs that were never told. The replies then show up in the conversation along with their meaning; each need they reveal has to be met, so search again pairing it with the values from the request.\n- "Don\'t mind" or "anything is fine" leaves that property free. As soon as the shopper names a preference or a limit (numbers included), it applies for the rest of the task, exactly as they worded it.\n- When the shopper asks you something (to confirm a change, for example), the reply is sent for you while messages remain; either way, the change is binding.\n\nWHEN THINGS CHANGE\n- Prices and stock can move without warning; read both again immediately before each order attempt. A listing whose price times quantity goes above the current budget no longer fits.\n- The shopper may alter the task as you go: the budget (higher or lower), a condition (removed, swapped or added), a price rise, an item selling out - possibly several in one line. Such lines can turn up inside any tool result (as an extra field of sentences) or together with an order attempt. Each change counts from the moment it appears: test your choice and backups again, and with a cheapest or best rule choose afresh among what is now acceptable.\n- If an order result contains a "refusal" field, the order was REFUSED, even when the inner part looks like a success. Nothing in the cart changed and the task continues: read the refusal, carry out every change it lists, remove whatever is no longer acceptable (for example a sold-out item), add the next verified fit, recheck price and stock, and order once more.\n\nTHE REQUEST\'S OWN RULES\nA request can lay down how you must work: how many listings to view or compare first, what to verify, what to tell the shopper, what a remark has to say, how to decline. Treat every such rule exactly like a condition on the product and obey it literally and completely. To "look at" or "open" N listings means N listings that fit in full, each opened with the details tool.\n\nENDING THE EPISODE\n- The opening brief lists this store\'s finishing calls: which tool, with which exact argument values, orders and which ends without buying. Use them precisely in that form.\n- Place your first order attempt while at least seven steps are left, so there is room to react to a refusal. Close to the step limit, only the cart, stock-check and finishing tools are still offered. Make up your mind once; every swap in the cart costs a step.\n- Maintain a ranked list of verified fits with at least one spare, since the first choice can sell out at checkout.\n- Before each order attempt make sure that: the cart contains one line per requested product - a single line when one product is asked for - each holding the chosen listing at its required quantity (remove anything else); stock and prices were read again after the last cart change; everything is available and the total is within the budget.\n- Declining almost never pays. Decline only after a careful search has found nothing satisfying every condition, and always the way the request says to decline. A listing breaks a requirement only when its own data give a different value; a property that no listing of its section records at all is no reason to pass a listing over or to decline - judge those listings on what they do state. If the request forbids proposing substitutes, the remark must not name, describe or hint at any listing: no ids, no titles, no prices.\n\nSCORING\nThe highest score goes to a correct order, less any cost the request itself sets (such as a fee per question); a valid order that is not the best possible choice still keeps part of the score. A wrong order, a wrong decline, ordering when a decline was correct, a correct order that skipped a procedure the request prescribes, and an episode left unfinished all score zero. An order is correct when every rule of the request was followed, the cheapest (or best) was taken where requested, budget and quantity are right, and the listing meets all conditions of the request, all needs the shopper revealed and all changes announced during the task. If nothing in the store can satisfy all that, a decline made exactly as the request describes is the correct ending.'
INTAKE_PROMPT = 'Your answer is a single JSON object on one line - no line breaks or indentation inside it and no text around it.\n\nWhy: a shopping assistant needs the shopper\'s request broken into parts, plus a map of which store tool performs which job. Below you will find the tools first and then the request.\n\nABOUT THE PRODUCT\n- "product": the type of product requested, using the request\'s wording.\n- "requirements": list of strings, one for every firm condition on the product, each in the form "property: constraint" and keeping exactly what the shopper said: the value, limit, exclusion or closed set. Write exclusions as "property: != value" (use "!= a, b" for more than one). Numbers keep their relation: write "property: <= N" or "property: >= N", and "property: covers N" when the range the item itself declares (its offered sizes, powers or settings) has to include N. If the request uses a word of its own as the value and then explains it, that word counts as the value and the explanation follows it in brackets (e.g. "colour: dusk (a deep blue)"); the explanation must never stand alone. If the request itself says which property a value is for, take that property name ("a heavy build" turns into "build: heavy", not into a guessed related property). Exclude: price, stock, quantity, budgets, simple wishes, and the story of the purchase - its reason, timing or occasion (say an event, a date or a mood) - unless the request makes it a condition of the item itself; also the product type or its category - words belonging to the category\'s name (a season, occasion or target group, for instance) are covered by the category already.\n- "quantity": the number of whole units of a single listing; 1 if the request does not ask for more, and something normally sold as a single unit (a set, a pair of shoes) is still 1.\n- "quantity_quote": when the quantity is above 1, the words of the request that state it, copied exactly; null otherwise.\n- "lines": only if two or more different products are requested: one object per product, in the order of the request, each {"product": its type in the request\'s wording, "requirements": its own firm conditions written as above, "quantity": its number of whole units}. In that case "requirements" above still collects all products\' conditions and "quantity" holds the first product\'s. Use [] for a request for one product. A shop category that joins two kinds with "and", "&" or "or" ("flats and heels", "binoculars or telescopes") names one product, not two.\n\nPRICE AND AVAILABILITY\n- "price_cap": the maximum total price as a number; null if the request gives none.\n- "currency": the unit in which the request states prices, or null.\n- "in_stock_only": true if only items that are actually available (in stock, on hand) are acceptable.\n\nTHE SELECTION RULE\n- "priority": "cheapest" if the request explicitly wants the cheapest or lowest-priced suitable option; "best" if it explicitly wants the best according to some measure; otherwise "none". Saying the budget is tight or that every penny matters does not ask for the cheapest - it merely explains the cap.\n- "priority_detail": the measure behind "best"; "" in all other cases.\n\nALLOWANCES\n- "message_budget": the number of messages the assistant is allowed to send the shopper, as an integer; null if not mentioned.\n- "step_budget": the permitted number of moves or steps, as an integer; null if not mentioned.\n\nHOW THE ASSISTANT MUST WORK\n- "rules": each instruction about the assistant\'s way of working, copied unchanged from the request (how many listings to view, what to check or report, how to decline, what to avoid, what to include). When unsure whether a sentence is a rule, put it here. Things the shopper says they will do themselves are story, not rules.\n- "procedures": those rules that require a tool call of their own before the end, each as {"do": a brief description, "quote": the request\'s exact words, "role": the job in "roles" whose tool performs it ("message", "view", "details", "search" or "cart_view"), "count": the number of such calls (1 unless a number is stated), "after_pick": true if it concerns the chosen item and therefore comes after the item is in the cart (confirming the choice with the shopper, for instance), "before": "order", "decline" or "any"}; [] if there are none.\n- "decline_no_names": true if, when declining, the request forbids proposing or mentioning any substitute or alternative; false if it asks for something to be named; null if it gives no declining rule.\n- "ask_first": points that only the shopper can decide and that should be asked before choosing - which of two wishes is more important when the request says they may not both be achievable (both wishes then go to "soft_wants", not to "requirements"), or whatever the request says to confirm with the shopper; [] if none.\n- "soft_wants": what the shopper would like without strictly requiring it (ideally, preferably, would be nice, if possible); never put these in "requirements".\n- "question_cost": if the request puts a price on asking the shopper, {"amount": number, "per": "question" or "step", "share": the fraction of the task\'s score (0 to 1) that one question costs when the request relates its price to the score, otherwise null, "quote": exact words}; null otherwise.\n\nTHE STORE\'S TOOLS\n- "roles": an object with the keys "search", "details", "view", "message", "cart", "cart_view", "order" and "decline"; each maps to the exact name of the listed tool that performs that job, or null. "message" contacts the shopper; "cart" adds items to the cart or removes them; "order" checks out / buys; "decline" ends the task without a purchase. For "order" and "decline" give an object instead of a name: {"tool": exact tool name, "fixed": {argument: value}}, where "fixed" assigns every required argument that has a list of permitted values the exact value making the tool perform that job ({} if there is no such argument). If one tool serves both jobs, its two "fixed" objects must differ.\n\nPlace each sentence of the request in exactly one category: requirement, priority, soft want, rule, budget, hint, or pure story.\n'
NEWS_PROMPT = 'The shopper has just written something new. Determine its effect on the shopping assistant\'s task and answer with one JSON object and nothing else, containing:\n\n- "meaning": one short sentence giving the shopper\'s point.\n- "needs": firm product requirements that appear for the first time in this line, each as "property: constraint" and carrying exactly what the shopper gave - a value, a limit, a closed set, an exclusion written "property: != value", or a number relation written "property: <= N", ">= N" or "covers N". Use the property that the shopper\'s own words point to, even if the assistant\'s question was about some other property.\n- "dropped": earlier requirements that are withdrawn.\n- "no_preference": properties left to the assistant\'s choice.\n- "new_budget": the fresh limit on the total price, as a number, or null.\n- "unavailable": normally false; true only if the shopper says an item cannot be bought any more (gone, sold out, no stock left, its button disabled).\n- "requirements_now": every firm product requirement that holds now, taking into account all that has been said - rebuilt from the request, the shopper\'s earlier lines and this line. Keep each earlier requirement in its earlier wording unless it was dropped or replaced; for a replaced one keep only the newest version; add every need just revealed. Write them like "needs" (when the request explains a value it named, the named word stays and the explanation follows it in brackets). Never list price, stock, quantity, the product category or the story behind the purchase as requirements.\n- "reply": if the shopper asks the assistant a question or wants something confirmed, a brief plain answer to send back that states the substance of the change (confirming the new value, for example); otherwise "".\n- "answer_class": null unless the assistant\'s question is included below; then {"class": "reveal" (a wish about the asked property is stated), "not_picky" (indifferent about it), "split" (the shopper read the question as several things, or asks which thing is meant), "unclear" (the shopper did not understand the message and wants it put differently) or "other"; "key": the property the answer concerns; "value": the wanted value, for reveal only; "relation": "is", "at most", "at least" or "not", for reveal only; "quote": the shopper\'s exact words that carry it}.\n\nReading rules:\n* A single line may contain several changes; list all of them.\n* A replacement is reported twice: the new requirement in "needs" and the old one in "dropped".\n* Every budget the shopper mentions, larger or smaller, goes into "new_budget" and supersedes the old one, also when it is good news.\n* If the shopper says which of two wishes is more important, that wish becomes a need.\n* Words such as under, over, at most, at least or no more than together with a number form a limit; keep the unit.\n* A plain "anything works", "no preference" or "don\'t mind" places the asked property into "no_preference".\n* Wishes ("ideally", "would be nice" and similar) are not needs.\n* Read the words literally but generously: a value expressed in a homely, figurative or indirect way is still firmly wanted; count a value as excluded only if the shopper clearly asks to avoid it.'
VERIFY_PROMPT = 'You are the last checkpoint before a shopping assistant orders one candidate item. The order is only worth something if the item satisfies every requirement listed below; base your verdict solely on the records shown here.\n\nAnswer with a single JSON object and no other text: {"item": the item\'s id, "fits": true or false, "problems": [brief strings, one for each requirement that is unmet or unshown]}\n\nUse "fits": false only for a requirement that the records plainly violate or that none of the records addresses. Lacking a stock or price record is not a reason to reject the item.\n\nHow to match a requirement:\n- Search for the value under the item\'s own property for that requirement. Any key that names the property counts, whatever its spelling (spaces, dots, underscores, prefixes), and one property may be spread over several keys. Compare by meaning: ignore case, accept singular or plural and close synonyms, and accept a single element of a comma list; a separate qualifier placed before the wanted word (a shade, a finish) still matches, but a longer single word that merely includes it does not; a broader or different class, or a merely similar-looking word, is not a match. If a requirement is a value plus an explanation in brackets, the value is what must be met; the explanation just clarifies it.\n- If a key with the property\'s own name (any spelling, singular or plural) holds a single, different value, the item fails, even if some other key\'s list includes the wanted value.\n- A related but distinct property does not count.\n- Title and description, or the variant / option key, may decide only when the item has no key at all for that property: a complete wanted value stated there ("options: teal" for a teal colour) satisfies it. If the item\'s own property gives a different value, the requirement fails regardless of the title. A word in another language, or a conclusion reached from other facts, is not a statement of the value.\n- Numbers: convert units first; a value on the limit is within it; a range passes only when every part of it satisfies the limit. "covers N" means the listing\'s own range must extend to N, so a range that stops below N fails. When the requirements say the store words the value exactly as the request does, only a listing using that wording satisfies it.\n- Closed lists and exclusions must be honoured. An exclusion holds only if the listing\'s own key for that property shows another value; if the listing says nothing about it, report that.\n\nFurther checks:\n- Item type: it must be the type requested. When listings carry category labels, the label decides; it has to match the request\'s full wording, not a narrower or adjacent label, and anything under that label is the requested type whatever its title or description implies (even if the title sounds like a part, an add-on or a different kind of item).\n- Stock and money: an item shown as out of stock fails; price times quantity must not exceed the budget (equality passes).\n- If records conflict (about stock or price, for instance), go by the most recent one.\n- A wish that is marked as not required is not a requirement.'
ASK_PROMPT = 'Help a shopping assistant pick the next message to the shopper. Shoppers frequently hold back a need, and that need is always the desired value of a single product property; it only comes out when a message asks about precisely that property. Your job: choose one property and phrase one question about it.\n\nBelow you will find: the requirements that apply now; possibly a line "Only the shopper can settle"; the questions already sent together with the answers; the block of properties on the opened listings\' sheets (each line gives a name, how many listings carry it and some sample values; each line is a separate property even when two names resemble each other, and the names in "[also spelled: ...]" are just alternative spellings of that line); possibly a numbered list of candidate properties ("Candidate properties ..."); possibly a note that the last answer seemed to cover several things; and the request itself.\n\nExcluded from your choice:\n1. properties absent from all listings, and properties already asked about;\n2. properties whose values mostly repeat the requested item\'s own name (that name already decides them);\n3. properties the request has decided - its stated requirements, price, stock, quantity - except two cases that remain open: a property the request presents only as a choice between alternatives, and one stated with a general word that the listings divide into more specific values (asking for "wood" while listings show oak, pine or walnut). A single exact value that the listings use as it is stays decided;\n4. terms of the sale or the listing rather than characteristics of the item;\n5. a key collecting the purchasable variants, or one that names several features at once (mixed kinds of values, or long lists of items): hidden needs never look like that.\nNever ask for a confirmation, for a choice among listed options, for anything else the shopper might want, for the purpose, or for the budget.\n\nRanking the rest: the property most likely to conceal an unspoken need comes first. Hints come from the request (for whom, where and how the item is used, how it should look), the requirements, the earlier questions and answers, and the listing properties. An "Only the shopper can settle" point that has not been asked takes precedence over everything. If two wishes conflict on one property, ask about that property and mention both values; wishes about two different properties need two separate messages. Wide coverage wins over rarity (a property stated by most or all opened listings beats one carried by only a few), and variation wins over sameness (values that differ across listings beat a value marked "(same value on every listing)"). If the candidate list is present, pick from it - from its top three when the list requires that - steered by the hints in the request.\n\nThe question: a single brief, plain, open sentence to the shopper (about ten words, far shorter than a hundred characters) about the property, using the words the listings use for the key. Name only the product and the property: no details from the story (who it is for, where or when it is used), and no example values unless the next sentence calls for them. If that key word also occurs inside other, longer keys, add the item\'s noun plus one or two values found only under this key, so the neighbour cannot be confused with it; otherwise give no example values. Use one noun for the item, never two item names glued together; if several different products are requested, mention only the product the property belongs to, in the request\'s wording for it. Nothing to approve, no second question, no lists, no instructions.\n\nIf the note about an answer that covered several things is present, ask about the same property once more but more precisely - add the item\'s noun, or if it was already included, one or two values found only under that property - using different words from the previous question, and set "repeat" to true; otherwise "repeat" is false.\n\nAnswer format, a single JSON object without any other text:\n{"property": "...", "question": "...", "chance": 0.0, "repeat": false, "deviate": ""}\n"chance" = your estimate (0 to 1) that the property hides an unspoken need. "deviate" = "" if the choice is allowed by the candidate list; for any other choice, the reason (a hint in the request pointing to it). If no message is worth sending: {"property": null}.'
AUDIT_PROMPT = 'Before a shopping assistant tells the shopper that no listing is suitable, review the records one final time. After these instructions you will find the requirements that currently apply, the shopper\'s request and the listing records.\n\nHow to judge: a requirement is met only if the listing\'s own key for that property contains the value (key spelling is irrelevant). A comma list containing it, a plural for a singular, or a close synonym passes; a broader or different class fails, and a key with exactly the property\'s name holding one single different value disqualifies the listing. Rely on title or description only when no such key exists. When none of the records mentions a requirement\'s property anywhere, it cannot be proven in this shop: judge the listings on the other requirements. Convert units before comparing numbers; a value exactly at a limit is fine. The price has to fit the budget, and a listing marked out of stock is ruled out.\n\nAnswer with one JSON object only: {"possible_fits": [ids of up to 5 listings that appear to satisfy every requirement, best first], "reason": "one sentence"}.'
ROLES_PROMPT = 'This is the tool list of an online shop. For each job below, find the tool that does it and give its name exactly as listed (null when no tool does it; one tool may do two jobs):\nsearch - looks listings up\ndetails - the full set of one listing\'s properties\nview - a listing with its price and stock\nmessage - sends a note to the shopper\ncart - puts a listing in the basket or takes it out\ncart_view - shows the basket\norder - buys what is in the basket and ends the task\ndecline - ends the task without buying\nReply with one JSON object on one line and nothing else: {"roles": {"search": ..., "details": ..., "view": ..., "message": ..., "cart": ..., "cart_view": ..., "order": ..., "decline": ...}}'
FINISH_ROLES_PROMPT = 'Give one line of JSON, with no other text, in precisely this form:\n{"order": {"tool": "<the tool\'s name, as listed>", "fixed": {"<argument>": <its value>}}, "decline": {"tool": "<the tool\'s name, as listed>", "fixed": {"<argument>": <its value>}}}\n\nBackground: the tools listed below belong to an online shop and are operated by an assistant. Exactly two sorts of call close the task -\n- "order" purchases the current contents of the basket;\n- "decline" finishes without purchasing anything.\nFor each of the two, state which tool it is and which argument values set it off, going by the full schemas (descriptions plus allowed values).\n\nFilling in "fixed":\n- list every argument limited to options (a list of values, or true/false) that decide the call\'s behaviour, with exactly the value this job needs;\n- leave free-text arguments out; write {} when there are none of that kind;\n- if one tool covers both jobs, the two entries name identical arguments but give them different values.\nTools that only display the basket or only message the shopper count as neither.'
REFEREE_PROMPT = 'Settle a disagreement between two readings. Below this text you get a first and a second reading; both say which shop tool buys what is now in the basket - the "order" job - and which one closes the task with nothing bought - the "decline" job.\n\nProcedure:\n1. Check each claim against the tool list: the argument schemas, permitted values and descriptions.\n2. Determine the correct tool for each job; it may come from the first reading, from the second, or from neither.\n3. Under "fixed", put every argument whose allowed options switch what the call does, each with its exact value. If a single tool handles both jobs, the two entries name identical arguments but with different values.\n\nYour whole answer is this one line of JSON:\n{"order": {"tool": "<the tool\'s name, as listed>", "fixed": {"<argument>": <its value>}}, "decline": {"tool": "<the tool\'s name, as listed>", "fixed": {"<argument>": <its value>}}}'
NO_FIT_REMARK = 'Ending without a purchase: I could not verify any listing that meets every stated requirement.'
_CODE_FINISH_TEXT = {'order': 'Buying what is in the cart.', 'decline': NO_FIT_REMARK}
SKIPPED = 'Not run, no step used'
HELD = 'Paused by your own pre-send check (the store never saw this call), so no step was used.'
_NO_CALL_TEXT = 'Every reply has to call a tool, and this one called none. Carry on.\n'
_ANSWER_FOLLOW_UP = "\n[note from your checks] The shopper's answer was added to the requirements in their words. A good next move: search that value together with the product type, then check the current pick and every opened listing against it again."
NOTES = {'relax': 'Prices need comparing again: {change} is not what it was, and only the lowest-priced acceptable listing is right. Run the product search once more, leaving out what was dropped, and weigh the results before buying.{seen} If your present pick remains the cheapest match, resend the identical order call.', 'relax_seen': ' Rows seen earlier at a lower price that might now qualify: {ids}.', 'cheap_rows': 'Only the lowest-priced acceptable listing will do. These rows cost less, are within budget, belong to the same section, and appeared in results without ever being opened: {ids}. Open them in that order and buy the first that satisfies everything; if none does, resend the identical order call.', 'cheap_pages': 'Only the lowest-priced acceptable listing will do, and prices are visible on the rows, yet some searches still have matches you have not been shown: {pages}. Continue paging through those, open the in-budget rows of that section that cost less, then buy the cheapest full match (or resend the identical order call if your pick is still the lowest).', 'cheap_bare': 'Only the lowest-priced acceptable listing will do, but none of your searches used a required value by itself. Search for just {values}{scope} and page through all of the results, because extra words hide listings phrased another way. Afterwards open the in-budget rows priced below your pick and buy the cheapest full match (or resend the identical order call if your pick is still the lowest).', 'cheap_scope': ' - limited to the section called "{label}" when the tool can do so', 'release': 'Think again before removing {target}: its own data still satisfy the request, at {price}{within}. Leave it in the cart unless some message says it is unavailable or unwanted; to remove it anyway, send the same cart call once more.', 'release_budget': ' (spending limit {budget})', 'fit_unread': 'Nothing about {held} is confirmed yet, because its data sheet was never read. Read it, check each requirement against it, and only then buy; if you have just read it, resend the identical order call.', 'fit_money': 'Latest reading of the data: {held} {why}. A change happened mid-session, so inspect it again. If that is confirmed, remove it and buy the next listing that is still acceptable (comparing prices afresh when the cheapest is wanted). Resend the identical order call only once a fresh check shows it acceptable again.', 'fit_sold': 'now appears as sold out', 'fit_dear': 'has gone up to {price} each, above the {budget} allowed in total', 'fit_unmet': 'The data of {held} do not confirm {unmet} - its value differs, or it stays silent on that property where most listings of its section state it.{pointer} Buy a listing whose own data confirm every requirement; resend the identical order call only if {held} truly meets them.', 'fit_others': ' Opened listings that do confirm all requirements: {ids}.', 'fit_words': " The request's own wording for it: {phrases}. Put that wording together with the product into a search and open the listings mentioning it.", 'page_arg': 'Page value check: {value} was not offered - the most recent reply to this same search handed out {offered} for its next page, and any other value jumps over or repeats rows. To keep reading send {call}; to jump on purpose, repeat this call unchanged.', 'flood': 'Already {count} searches without opening a single listing: reading what is on screen usually beats searching more. Begin by opening these unopened rows from the wanted sections (sharpest search first): {ids}.{mixed} Afterwards search again only for what remains open; to search regardless, repeat this call unchanged.', 'flood_mixed': " In the worded searches most rows belonged to other sections, so that mode appears to disregard the section setting; joining a required value with the product's own noun keeps to the section, e.g. {example}.", 'page_fresh': 'You have never sent this search before, so the page value {value} comes from a different search and would make this one begin after its first rows. For its first page send {call}; to jump on purpose, repeat this call unchanged.', 'reach_early': 'Already {count} searches, and none of the opened listings satisfies every requirement yet. {advice} Before trying new wordings, open the rows there whose text clashes with no requirement; to search regardless, repeat this call unchanged.', 'early_pages': 'Searches with a required value in them still hold matches you have not read, and the store will page them out: {pages} - go through them to the last page.', 'early_words': "In the worded searches most rows belonged to other sections; pairing a required value with the product's own noun keeps nearer to it: {needles}.", 'best_probe': 'You want the best by {detail}, and listing texts rarely open with that figure. Figures for it so far (lowest and highest in each unit): {ends}. Search a required value combined with such a figure, and with figures past these ends in the same unit (e.g. {needles}); open the results, then buy the best full match, or resend the identical order call if your pick is still the best.', 'page_empty': 'Empty paging argument {arg}: that names no page at all. For the first page drop {arg} from the call: {call}.', 'page_none': 'Page value check: nobody issued {value} - the latest reply to this very search gave no value for another page, so an invented one returns nothing. Rows past the ones shown are reached by narrowing instead (put in a required value or the noun of the product).{words} If you still want to send it, repeat this call unchanged.', 'page_words': ' Wordings that stay inside the section: {needles}.', 'reach_rows': 'Not yet - open these before giving up: {ids}. Searches that named every requirement listed them, yet nobody has read their data sheets. If none proves acceptable, resend the identical decline call.', 'reach_pages': 'Not yet - first: {advice}. If nothing new qualifies, resend the identical decline call.', 'reach_read': 'some searches have rows left unread that paging could reach completely ({pages}); open each row whose text clashes with no requirement', 'reach_idle': 'with or without {ignored} the results came out the same, so those rows tell nothing about it - narrow another way (put those words first or use them on their own, switch the filter or the mode)', 'ask_none': 'there is no question mark, and messages are rationed. Contact the shopper only to raise one single property, in one single question.', 'ask_many': '{marks} question marks in it; a shopper answers one single point per message. Keep it to one property.', 'unfit': "Given this store's tool rules, the {tool} call you sent neither orders nor declines. {forms}", 'empty': 'Nothing is in the cart. Add your pick first - or decline.', 'items': 'Basket check: {wanted} products are requested, each needing its own cart line at its own quantity, but the cart has {lines} line(s). Missing a line: {missing}. Add a verified fit for each, then order.', 'step_open': 'Still to do, set by the request: {do} ("{quote}"). Carry it out first; finish afterwards.', 'step_ask': 'The request still requires: {do}. The code sends that message to the shopper itself; finish once the answer has been read.', 'count': 'Unit count wrong for {held}: the cart view shows {shown} where the request wants {wanted}. Fix it before buying - remove the line and add it back with the right count.', 'decline_again': 'Declining scores nothing while acceptable listings are available. Both their own data and the reviewer accept: {ids}. Put the strongest in the cart and buy it; decline only once you can quote, for each of them, a value of its own that contradicts a stated requirement; saying nothing about a property is no contradiction.', 'reserve': 'the final step is reserved for finishing: only an order or a decline may use it.', 'swap': 'Decline withheld: by its own data and by the reviewer, {pick} satisfies every requirement, and buying is worth more than giving up, so the code carted it (if it was not in the cart yet) and ordered it. Replies from the store: {said}', 'swap_cut': 'Decline withheld: by its own data and by the reviewer, {pick} satisfies every requirement, so the code carted it; deal with the reply before ordering: {said}', 'reserve_late': 'this close to the step limit the steps that remain are held for cart, stock check and finish.', 'held_over': 'Not run and no step spent, because {stop} Deal with that result first.', 'closed': 'Not run: no steps remain, or the episode is already over.', 'unreached': 'The store did not answer this call (connection or server trouble). Send it again.', 'why_unreached': 'an earlier store call in this batch got no answer.', 'why_fault': 'an earlier call in this batch already hit a fault.', 'why_refused': 'the finish attempt before it was refused.', 'why_rejected': 'a finish attempt before it was turned down.', 'why_news': 'the result before it carried news from the shopper.', 'fault_twice': "From now on blocked: calls with this shape, because {tool} failed with '{code}' two times. Rework the arguments to match the schema, or get there with a different tool.", 'spent': "This tool's allowance is used up - no more calls to it.", 'order_refused': 'ORDER REFUSED. Work continues and the cart is untouched. Next: carry out every change the refusal names, take out lines that are no longer acceptable (an item that sold out, say), read stock and price of the replacement again, and order once more.', 'decline_refused': 'DECLINE REFUSED. It is not over: take in every change the reply lists, keep looking, and finish afterwards.', 'recheck': 'Lowest price wanted: these in-budget listings from the same section, priced at most what your pick cost last, were shown but never opened: {priced}. Work upward from the cheapest, checking each against every requirement, before you accept anything more expensive.', 'rejected': "The finish was not accepted; the store's reply says why. Resolve that, then finish again.", 'news': 'News from the shopper came with this result - apply it: {heard}'}
ORDER_CHECKLIST = HELD + " Run through this checklist, fix anything that is not yet so, then repeat the order call. 1) Cart: one line for each requested product (a single line for one product), each with the selected listing at its required quantity. 2) Fit: the listing's own properties satisfy all stated requirements, all needs the shopper has revealed and all announced changes. 3) Price and stock: re-read since your most recent cart change using a tool that displays them; in stock, and price times quantity inside the current budget. 4) Choice: if the cheapest or best was requested, nothing you were able to find fits better. 5) Rules: all procedures required by the request have been completed."
DECLINE_CHECKLIST = HELD + ' Once sent, a decline is final. Repeat it only when you are sure of three things - the own data of every candidate has been read; your searches tried different words, filters and pages; and each candidate breaks the budget, the stock, or a stated requirement or revealed need through a value of its own (a listing that is silent about a property does not break it). Remember that matching keys can have other spellings and that near-synonyms count. If anything fits, buy it; if nothing does, call the decline tool again.'
_BRIEF_LISTS = (('soft_wants', '- wished for, NOT required (only breaks ties): '), ('ask_first', '- to learn from the shopper before choosing: '), ('rules', '- rule set by the request: '))
ID_PATTERN = re.compile('^[A-Za-z0-9][A-Za-z0-9_\\-:.]{3,63}$')
_SPLIT_WORDS = re.compile('[\\s.,;:!?()\\[\\]{}<>\\"\'`]+')
_LIST_SPLIT = re.compile('[,/;|]')
_RELATION_MARKS = ('<=', '>=', 'covers')
_JSON_TEXT = json.JSONEncoder(separators=(',', ':'), ensure_ascii=False)
_FENCE_OPEN = re.compile('^```[A-Za-z]*')

def _trim_strings(field_value, limit, nest_level=0):

    def shrink(branch, level):
        if 8 < level:
            return None
        elif isinstance(branch, str):
            if limit >= len(branch):
                return branch
            held_1 = branch[:limit] + '...'
            return held_1
        elif not isinstance(branch, dict):
            if isinstance(branch, list):
                held_3 = list((shrink(nested, level + 1) for nested in branch))
                return held_3
            return branch
        else:
            held_2 = dict(((key, shrink(nested, level + 1)) for key, nested in branch.items()))
            return held_2
    held_4 = shrink(field_value, nest_level)
    return held_4

def _json_line(field_value):
    try:
        text = _JSON_TEXT.encode(field_value)
    except (TypeError, ValueError):
        text = str(field_value)
    if TEXT_LIMIT < len(text):
        text = text[:TEXT_LIMIT] + '...(cut)'
    return text

def _try_object(text):
    try:
        decoded = json.loads(text)
    except ValueError:
        held_5 = (False, None)
        return held_5
    held_6 = (True, decoded if isinstance(decoded, dict) else None)
    return held_6

def _reply_object(text):
    if isinstance(text, str):
        body = text.strip()
        if body.startswith('```'):
            body = _FENCE_OPEN.sub('', body).strip()
            if body.endswith('```'):
                body = body[:-3].strip()
        decoded, found = _try_object(body)
        if not decoded:
            opened = body.find('{')
            closed = body.rfind('}')
            if not (-1 == opened or opened >= closed):
                held_7 = _try_object(body[opened:closed + 1])[1]
                return held_7
            else:
                return None
        else:
            return found
    else:
        return None

def _collect_sentences(field_value, out, nest_level=0):
    if 4 < nest_level:
        return
    elif isinstance(field_value, list):
        for element in field_value[:10]:
            _collect_sentences(element, out, nest_level + 1)
    elif isinstance(field_value, str):
        if len(field_value.split()) > 3:
            out.append(field_value)

def _collect_ids(field_value, out, nest_level=0):
    stack = [(nest_level, field_value)]
    while stack:
        level, branch = stack.pop()
        if not 6 < level:
            below = []
            if isinstance(branch, list):
                below = list(branch)
            elif isinstance(branch, dict):
                for key, nested in branch.items():
                    if not (not isinstance(nested, str) or 'id' not in key.lower() or (not ID_PATTERN.match(nested))):
                        out.add(nested)
                    else:
                        below.append(nested)
                        continue
            stack.extend(((level + 1, nested) for nested in below))
        else:
            continue

def _number_or_none(field_value):
    if not (True is field_value or False is field_value):
        if isinstance(field_value, (int, float)):
            return field_value
        return None
    else:
        return None

def _terms(text):
    held_8 = str(text or '').lower()
    found = re.findall('[a-z0-9]+', held_8)
    held_9 = set((tk for tk in found if tk.isdigit() or 2 < len(tk)))
    return held_9

def _flat(text):
    chunks = str(text or '').lower().split()
    held_10 = ' '.join(chunks)
    return held_10
_INFLECTED = {'e': ('ing', 'ed', 'er', 'ers', 'es'), 'y': ('ies', 'ied', 'ier')}

def _word_like(have, wish):
    a, b = (have, wish) if len(wish) >= len(have) else (wish, have)
    if not (b.startswith(a) and len(a) + 2 >= len(b)):
        held_11 = 3 < len(a) and any((b == a[:-1] + end for end in _INFLECTED.get(a[-1], ())))
        return held_11
    else:
        return True

def _covers(needed, have):
    for term in needed:
        if not any((_word_like(term, other) for other in have)):
            return False
    return True

def _key_terms(key):
    held_12 = _terms(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(key)))
    return held_12

def _key_is_property(key, need_terms):
    call_args = _key_terms(key)
    held_13 = bool(call_args) and len(need_terms) == len(call_args) and _covers(need_terms, call_args)
    return held_13

def _singular_terms(label_name):
    out = set()
    for term in _terms(label_name):
        held_14 = 3 < len(term)
        if held_14:
            held_14 = 's' == term[-1]
        if held_14:
            held_14 = 's' != term[-2]
        plural = held_14
        out.add(term[:-1] if plural else term)
    return out

def _tokens(text: Any) -> list[str]:
    held_15 = re.findall('[a-z0-9]+', str(text).lower())
    return held_15

def _same_name(a: str, b: str) -> bool:
    held_16 = _tokens(a) == _tokens(b)
    return held_16

def _walk_dicts(roots):

    def visit(branch, nest_level):
        if not 6 < nest_level:
            if isinstance(branch, dict):
                yield branch
                kids = list((fv for fv in branch.values() if isinstance(fv, (dict, list))))
            elif not isinstance(branch, list):
                return
            else:
                kids = branch
            for kid in kids:
                yield from visit(kid, nest_level + 1)
        else:
            return
    for root in roots:
        yield from visit(root, 0)

def _all_containers(roots):
    todo = list(roots)
    while todo:
        branch = todo.pop()
        if isinstance(branch, list):
            yield branch
            todo += branch
        elif isinstance(branch, dict):
            yield branch
            todo += list((fv for fv in branch.values() if isinstance(fv, (dict, list))))

def _is_string_list(field_value):
    if not (not isinstance(field_value, list) or not field_value):
        held_17 = all((isinstance(fv, str) for fv in field_value))
        return held_17
    else:
        return False

def _peel_wrappers(field_value, nest_level=0):
    if not (4 <= nest_level or not isinstance(field_value, dict) or 2 != len(field_value)):
        (_, v1), (_, v2) = field_value.items()
        first, second = (_is_string_list(v1), _is_string_list(v2))
        if not second == first:
            said, rest = (v1, v2) if first else (v2, v1)
            deeper, nested = _peel_wrappers(rest, nest_level + 1)
            held_20 = (list(said) + deeper, nested)
            return held_20
        else:
            held_19 = ([], field_value)
            return held_19
    else:
        held_18 = ([], field_value)
        return held_18

def _is_refusal_code(field_value):
    if not isinstance(field_value, dict) or not 1 <= len(field_value) <= 3:
        return False
    elif not any((isinstance(fv, (dict, list)) for fv in field_value.values())):
        for raw_obj in field_value.values():
            if isinstance(raw_obj, str):
                token = raw_obj.strip()
            else:
                token = ''
            if token:
                if ' ' not in token:
                    if token != token.lower() or '_' in token or re.search('\\d', token):
                        return True
        return False
    else:
        return False

def _is_code(text, need_separator):
    if not (not re.search('\\d', text) or not re.fullmatch('[A-Za-z0-9_\\-]+', text)):
        if need_separator:
            held_21 = '-' in text or '_' in text
            return held_21
        return True
    else:
        return False

def _is_listing_id(field_value, ids):
    if not (isinstance(field_value, str) and 8 <= len(field_value) and (field_value in ids)):
        return False
    else:
        held_22 = any(map(str.isdigit, field_value)) and any(map(str.isalpha, field_value))
        return held_22

def _schema_types(arg_shape):
    declared = arg_shape.get('type')
    if not isinstance(declared, str):
        return declared
    else:
        held_23 = [declared]
        return held_23
_NOT_CONVERTED = object()
_INT_LITERAL = re.compile('\\s*-?\\d+(\\.0+)?\\s*')
_NUM_LITERAL = re.compile('\\s*-?\\d+(\\.\\d+)?\\s*')
_PY_TYPES = {'string': str, 'boolean': bool, 'object': dict, 'array': list}
_SCHEMA_TYPE_NAMES = frozenset(['integer', 'number', 'null', *_PY_TYPES])

def _is_of_type(kind, field_value):
    if 'null' == kind:
        held_24 = None is field_value
        return held_24
    elif not kind in _PY_TYPES:
        real = _number_or_none(field_value) is not None
        if 'integer' == kind:
            held_26 = real and isinstance(field_value, int)
            return held_26
        return real
    else:
        held_25 = isinstance(field_value, _PY_TYPES[kind])
        return held_25

def _type_allows(field_value, type_names):
    for kind in type_names:
        if not isinstance(kind, str) or kind not in _SCHEMA_TYPE_NAMES or _is_of_type(kind, field_value):
            return True
    return False

def _as_int(field_value):
    if isinstance(field_value, float):
        if field_value.is_integer():
            held_27 = int(field_value)
            return held_27
        return _NOT_CONVERTED
    elif not (isinstance(field_value, str) and _INT_LITERAL.fullmatch(field_value)):
        return _NOT_CONVERTED
    else:
        held_28 = int(float(field_value))
        return held_28

def _as_float(field_value):
    if not (not isinstance(field_value, str) or not _NUM_LITERAL.fullmatch(field_value)):
        amount = float(field_value)
        if amount.is_integer():
            held_29 = int(amount)
            return held_29
        return amount
    else:
        return _NOT_CONVERTED

def _as_text(field_value):
    if _number_or_none(field_value) is None:
        return _NOT_CONVERTED
    held_30 = str(field_value)
    return held_30

def _as_bool(field_value):
    if isinstance(field_value, str):
        term = field_value.lower()
    else:
        term = ''
    if 'true' == term:
        return True
    elif not 'false' == term:
        return _NOT_CONVERTED
    else:
        return False
_TYPE_CASTS = {'integer': _as_int, 'number': _as_float, 'string': _as_text, 'boolean': _as_bool}

def _cast_to_schema(field_value, arg_shape):
    type_names = _schema_types(arg_shape)
    if not type_names:
        return field_value
    elif not _type_allows(field_value, type_names):
        for kind in type_names:
            if not (not isinstance(kind, str) or kind not in _TYPE_CASTS):
                try:
                    moved = _TYPE_CASTS[kind](field_value)
                except (TypeError, ValueError):
                    continue
                if _NOT_CONVERTED is not moved:
                    return moved
            else:
                continue
        return field_value
    else:
        held_31 = isinstance(field_value, float)
        if held_31:
            held_31 = field_value.is_integer()
        whole = held_31
        if whole and 'integer' in type_names:
            held_32 = int(field_value)
            return held_32
        return field_value
_LIMIT_RULES = {'number': (('minimum', False, lambda got, edge: edge > got, 'is below the minimum of {}'), ('maximum', False, lambda got, edge: edge < got, 'is above the maximum of {}'), ('exclusiveMinimum', True, lambda got, edge: edge >= got, 'must be more than {}'), ('exclusiveMaximum', True, lambda got, edge: edge <= got, 'has to stay strictly below {}')), 'text': (('minLength', False, lambda got, edge: edge > got, 'needs at least {} characters of text'),), 'list': (('minItems', False, lambda got, edge: edge > got, 'give at least {} items in the list'), ('maxItems', False, lambda got, edge: edge < got, 'give at most {} items in the list'))}

def _limit_problems(measure, arg_shape, group, trail):
    found = []
    for key, numeric_only, broken, pattern in _LIMIT_RULES[group]:
        if not key not in arg_shape:
            edge = arg_shape[key]
            if numeric_only and (not isinstance(edge, (int, float))):
                continue
            elif broken(measure, edge):
                found.append('%s: %s' % (trail, pattern.format(edge)))
        else:
            continue
    return found

def _check_choice(field_value, arg_shape, trail, issues):
    if 'enum' in arg_shape:
        choices = arg_shape.get('enum')
        if field_value not in choices:
            if isinstance(field_value, str):
                lowered = field_value.strip().lower()
                for option in choices:
                    if isinstance(option, str):
                        if option.lower() == lowered:
                            field_value = option
                            break
            if field_value not in choices:
                issues.append('%s: must be one of %s' % (trail, choices))
    if 'const' in arg_shape:
        only = arg_shape['const']
        if only != field_value:
            issues.append('%s: the single value this allows is %r' % (trail, only))
    return field_value

def _check_object(field_value, arg_shape, trail, issues):
    held_33 = arg_shape.get('properties')
    if not held_33:
        held_33 = {}
    member_list = held_33
    for label_name in arg_shape.get('required') or []:
        if label_name not in field_value:
            issues.append('%s.%s: required but not given' % (trail, label_name))
    if arg_shape.get('additionalProperties') is False:
        for label_name in list(field_value):
            if label_name not in member_list:
                issues.append('%s.%s: no such argument' % (trail, label_name))
    out = {}
    for label_name, nested in field_value.items():
        if label_name in member_list:
            nested, inner_issues = _fit_schema(nested, member_list[label_name], '%s.%s' % (trail, label_name))
            issues.extend(inner_issues)
        out[label_name] = nested
    return out

def _check_array(field_value, arg_shape, trail, issues):
    held_35 = len(field_value)
    held_34 = _limit_problems(held_35, arg_shape, 'list', trail)
    issues.extend(held_34)
    shape = arg_shape.get('items')
    if isinstance(shape, dict):
        out = []
        for at, element in enumerate(field_value):
            element, entry_issues = _fit_schema(element, shape, '%s[%d]' % (trail, at))
            out.append(element)
            issues.extend(entry_issues)
        return out
    else:
        return field_value

def _cut_at_word(text, cap):
    cut = text[:max(0, cap)]
    if len(cut) < len(text):
        if cut[-1:].isalnum():
            if text[len(cut)].isalnum():
                if cut.strip().count(' '):
                    cut = cut.rstrip().rsplit(None, 1)[0]
    held_36 = cut.rstrip() or text[:max(0, cap)]
    return held_36

def _fit_schema(field_value, arg_shape, trail):
    if isinstance(arg_shape, dict):
        field_value = _cast_to_schema(field_value, arg_shape)
        type_names = _schema_types(arg_shape)
        if not (type_names and (not _type_allows(field_value, type_names))):
            issues = []
            field_value = _check_choice(field_value, arg_shape, trail, issues)
            if isinstance(field_value, str):
                held_40 = len(field_value)
                held_39 = _limit_problems(held_40, arg_shape, 'text', trail)
                issues.extend(held_39)
                cap = arg_shape.get('maxLength')
                if 'maxLength' in arg_shape:
                    if cap < len(field_value):
                        field_value = _cut_at_word(field_value, cap)
            if _number_or_none(field_value) is not None:
                held_41 = _limit_problems(field_value, arg_shape, 'number', trail)
                issues.extend(held_41)
            if isinstance(field_value, list):
                field_value = _check_array(field_value, arg_shape, trail, issues)
            if isinstance(field_value, dict):
                field_value = _check_object(field_value, arg_shape, trail, issues)
            held_42 = (field_value, issues)
            return held_42
        else:
            held_38 = (field_value, [trail + ': has to be of type ' + '/'.join(type_names)])
            return held_38
    else:
        held_37 = (field_value, [])
        return held_37

def _is_selector(tool_desc):
    if not isinstance(tool_desc, dict):
        return False
    elif not (tool_desc.get('enum') or 'const' in tool_desc):
        held_43 = 'boolean' in (_schema_types(tool_desc) or [])
        return held_43
    else:
        return True

def _free_text_args(props):
    out = []
    for key, tool_desc in props.items():
        if isinstance(tool_desc, dict):
            if 'enum' not in tool_desc:
                if 'string' in (tool_desc.get('type') or []):
                    out.append(key)
    return out

def _fit_length(text, tool_desc):
    held_44 = len(text)
    cap = tool_desc.get('maxLength', held_44)
    held_45 = text[:cap]
    return held_45

def _filler_value(tool_desc, text):
    kind = tool_desc.get('type')
    if 'string' in (kind or []):
        held_46 = (True, _fit_length(text, tool_desc))
        return held_46
    elif tool_desc.get('enum'):
        held_47 = (True, tool_desc['enum'][0])
        return held_47
    elif not 'boolean' == kind:
        if kind in ('integer', 'number'):
            held_49 = (True, max(1, tool_desc.get('minimum', 1)))
            return held_49
        held_50 = (False, None)
        return held_50
    else:
        held_48 = (True, True)
        return held_48
FINISH_JOBS = ('order', 'decline')

def _tool_line(tool_desc):
    held_51 = '* %s -- %s -- arguments it takes: %s' % (tool_desc['name'], tool_desc.get('description', ''), _json_line(tool_desc.get('parameters') or {}))
    return held_51

def _reading_usable(plan):
    held_52 = bool(plan.get('product') or plan.get('requirements'))
    return held_52

def _is_tool_entry(element):
    if isinstance(element, dict):
        nested = element.get('function')
    else:
        nested = None
    held_53 = isinstance(nested, dict)
    return held_53

def _tool_params(tool_desc):
    held_54 = tool_desc.get('parameters') or dict(type='object', properties={})
    return held_54

def _name_terms(text):
    held_55 = _terms(re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', re.sub('[_.-]+', ' ', str(text or ''))))
    return held_55

def _arg_shape(arg_spec, depth=0):
    if not isinstance(arg_spec, dict) or 4 < depth:
        return ''
    elif not _is_selector(arg_spec):
        kinds = _schema_types(arg_spec)
        if isinstance(kinds, list):
            kinds = kinds
        else:
            kinds = []
        for type_name, shape in (('integer', 'count'), ('number', 'number'), ('string', 'text')):
            if type_name in kinds:
                return shape
        if 'array' in kinds:
            inner = arg_spec.get('items')
        else:
            inner = arg_spec if 'object' in kinds else None
        if isinstance(inner, dict):
            if arg_spec is not inner:
                member = _arg_shape(inner, depth + 1)
                if member in ('text', 'lines'):
                    if 'text' == member:
                        return 'ids'
                    return 'lines'
            members = inner.get('properties')
            if isinstance(members, dict):
                parts = list((_arg_shape(sub, depth + 1) for sub in members.values()))
            else:
                parts = []
            if 'lines' in parts or {'text', 'count'} <= set(parts):
                return 'lines'
            return ''
        else:
            return ''
    else:
        return 'choice'

def _tool_profiles(tools):
    shapes = {}
    for tool_desc in (tk['function'] for tk in tools if _is_tool_entry(tk)):
        params = _tool_params(tool_desc)
        if isinstance(params, dict):
            props = params.get('properties')
        else:
            props = None
        if isinstance(tool_desc.get('name'), str) and isinstance(props, dict):
            if isinstance(params.get('required'), list):
                must = params.get('required')
            else:
                must = []
            said = str(tool_desc.get('description') or '')
            pieces = (_name_terms(piece) for piece in re.split('[.!?]+', said))
            shapes[tool_desc['name']] = dict(kinds=dict(((arg, _arg_shape(sub)) for arg, sub in props.items())), props=props, must=set((arg for arg in must if isinstance(arg, str) and arg in props)), words=_name_terms(tool_desc['name']) | _name_terms(said), said=said, sentences=set((frozenset(piece) for piece in pieces if 1 < len(piece))))
        else:
            continue
    return shapes

def _text_capacity(arg_spec):
    if isinstance(arg_spec, dict):
        room = arg_spec.get('maxLength')
    else:
        room = None
    if isinstance(room, int) and (not isinstance(room, bool)):
        return room
    held_56 = float('inf')
    return held_56

def _id_capacity(shape):
    for arg, kind in shape['kinds'].items():
        spec = shape['props'][arg]
        if not 'text' == kind:
            while 'lines' == kind and isinstance(spec, dict):
                if isinstance(spec.get('items'), dict):
                    spec = spec.get('items')
                else:
                    spec = spec
                if isinstance(spec.get('properties'), dict):
                    inner = spec.get('properties')
                else:
                    inner = {}
                texts = list((sub for sub in inner.values() if _arg_shape(sub) == 'text'))
                if not texts:
                    spec = next((sub for sub in inner.values() if _arg_shape(sub) == 'lines'), None)
                else:
                    held_58 = _text_capacity(texts[0])
                    return held_58
        else:
            held_57 = _text_capacity(spec)
            return held_57
    return None

def _looks_like_cart(shape):
    kinds = shape['kinds']
    texts = list((arg for arg, kind in kinds.items() if 'text' == kind))
    if not 'lines' in kinds.values():
        held_59 = 1 == len(texts) and texts[0] in shape['must'] and ('count' in kinds.values())
        return held_59
    else:
        return True

def _choose_cart_tool(shapes, common):
    carts = list((name for name, shape in shapes.items() if _looks_like_cart(shape)))
    if not 2 > len(carts):
        lined = list((name for name in carts if 'lines' in shapes[name]['kinds'].values()))
        if not 1 == len(lined):
            bare = set().union(*(shape['words'] for shape in shapes.values() if not shape['kinds'])) - common
            held_62 = _sole_leader(carts, lambda name: len(shapes[name]['words'] & bare))
            return held_62
        else:
            held_61 = lined[0]
            return held_61
    else:
        if carts:
            held_60 = carts[0]
            return held_60
        return None

def _sole_leader(names, score):
    ranked = sorted(names, reverse=True, key=score)
    if not (not ranked or score(ranked[0]) <= 0 or (1 < len(ranked) and score(ranked[1]) == score(ranked[0]))):
        held_63 = ranked[0]
        return held_63
    else:
        return None

def _option_sides(said, values, marks):
    low = said.lower()
    spots = []
    for choice_value in values:
        hit = re.search('\\b%s\\b' % re.escape(str(choice_value).lower()), low)
        if hit:
            spots.append((hit.start(), hit.end(), choice_value))
        else:
            return None
    spots.sort()
    reach = {}
    for pos, (_, end, choice_value) in enumerate(spots):
        if len(spots) > pos + 1:
            stop = spots[pos + 1][0]
        else:
            stop = len(low)
        reach[choice_value] = len(_name_terms(low[end:stop]) & marks)
    first, second = values
    if not reach[second] == reach[first]:
        if reach[second] < reach[first]:
            held_64 = (first, second)
            return held_64
        held_65 = (second, first)
        return held_65
    else:
        return None

def _two_way_finish(shape, marks):
    picks = list((arg for arg in shape['must'] if 'choice' == shape['kinds'][arg]))
    rest = list((kind for arg, kind in shape['kinds'].items() if arg not in picks))
    if not (1 != len(picks) or 1 != len(shape['must']) or any(('text' != kind for kind in rest))):
        values = shape['props'][picks[0]].get('enum')
        if not (not isinstance(values, list) or 2 != len(values) or (not all((isinstance(fv, str) for fv in values)))):
            sides = _option_sides(shape['said'], values, marks)
            if None is sides:
                return None
            held_66 = ({picks[0]: sides[0]}, {picks[0]: sides[1]})
            return held_66
        else:
            return None
    else:
        return None

def _roles_from_shapes(tools):
    shapes = _tool_profiles(tools)
    counts = {}
    for shape in shapes.values():
        for tk in shape['words']:
            counts[tk] = counts.get(tk, 0) + 1
    common = set((tk for tk, seen in counts.items() if len(shapes) < 2 * seen))
    found = {}
    cart = _choose_cart_tool(shapes, common)
    if not None is cart:
        found['cart'] = [(cart, None)]
        marks = shapes[cart]['words'] - common
        overlap = lambda name: len(shapes[name]['words'] & marks)
        left = list((name for name in shapes if cart != name))
        closing = dict(((name, _two_way_finish(shapes[name], marks)) for name in left))
        closing = dict(((name, pins) for name, pins in closing.items() if pins))
        if 1 == len(closing):
            held_68 = closing.items()
            held_67 = iter(held_68)
            both, pins = next(held_67)
            found['order'], found['decline'] = ([(both, pins[0])], [(both, pins[1])])
            left.remove(both)
        wide = list((name for name in left if 1 < len(shapes[name]['kinds']) and set(shapes[name]['kinds'].values()) & {'text', 'choice'}))
        search = _sole_leader(wide, lambda name: len(shapes[name]['kinds']))
        if search:
            found['search'] = [(search, None)]
        held_69 = list((name for name in left if search != name))
        _one_arg_roles(shapes, cart, held_69, overlap, found)
        held_70 = list((name for name in left if search != name))
        _no_arg_roles(shapes, held_70, overlap, found)
        return found
    else:
        return found

def _one_arg_roles(shapes, cart, left, overlap, found):
    lone = list((name for name in left if 1 == len(shapes[name]['kinds'])))
    lists = list((name for name in lone if 'ids' in shapes[name]['kinds'].values()))
    texts = list((name for name in lone if 'text' in shapes[name]['kinds'].values() and shapes[name]['must']))
    room = dict(((name, _text_capacity(next(iter(shapes[name]['props'].values())))) for name in texts))
    id_room = _id_capacity(shapes[cart])
    if None is id_room:
        if 1 < len(texts):
            held_71 = room.values()
            id_room = min(held_71)
    wordy = list((name for name in texts if None is not id_room and 2 * id_room < room[name]))
    if 1 == len(wordy):
        found['message'] = [(wordy[0], None)]
    short = list((name for name in texts if name not in wordy))
    if 'choice' not in shapes[cart]['kinds'].values():
        if 1 < len(short):
            remover = _sole_leader(short, overlap)
            if remover:
                found['cart_remove'] = [(remover, None)]
                short.remove(remover)
    if 1 == len(short):
        found['details'] = [(short[0], None)]
    if 1 == len(lists):
        found['view'] = [(lists[0], None)]
    elif not lists:
        if 2 == len(short):
            found['details'] = found['view'] = list(((name, None) for name in short))

def _no_arg_roles(shapes, left, overlap, found):
    bare = list((name for name in left if not shapes[name]['must'] and 1 >= len(shapes[name]['kinds']) and (set(shapes[name]['kinds'].values()) <= {'text'})))
    if not 'order' in found:
        pairs = []
        for pos, one in enumerate(bare):
            for other in bare[pos + 1:]:
                joint = shapes[one]['sentences'] & shapes[other]['sentences']
                elsewhere = set().union(*(shapes[name]['sentences'] for name in shapes if name not in (one, other)))
                if joint - elsewhere:
                    pairs.append((one, other))
        if 1 == len(pairs):
            closers = list(pairs[0])
        elif not 3 == len(bare):
            closers = []
        else:
            top = _sole_leader(bare, overlap)
            if top:
                closers = list((name for name in bare if top != name))
            else:
                closers = []
        views = list((name for name in bare if name not in closers))
        if 2 == len(closers):
            if 1 == len(views):
                found['cart_view'] = [(views[0], None)]
        if 2 == len(closers):
            buyer = _sole_leader(closers, overlap)
        else:
            buyer = None
        if buyer:
            found['order'] = [(buyer, {})]
            found['decline'] = [(next((name for name in closers if buyer != name)), {})]
    else:
        if 1 == len(bare):
            found['cart_view'] = [(bare[0], None)]
        return

def _shape_suits_job(bag, shape):
    kinds, must = (shape['kinds'], shape['must'])
    needed_text = any(('text' == kinds[arg] for arg in must))
    if 'search' == bag:
        held_72 = bool(set(kinds.values()) & {'text', 'choice'})
        return held_72
    elif bag in ('details', 'cart_remove'):
        return needed_text
    elif not 'view' == bag:
        if 'message' == bag:
            held_74 = 'text' in kinds.values()
            return held_74
        if 'cart' == bag:
            held_75 = bool(set(kinds.values()) & {'text', 'lines'})
            return held_75
        if 'cart_view' == bag:
            held_76 = not must
            return held_76
        if bag in FINISH_JOBS:
            held_77 = all((kinds[arg] in ('choice', 'text') for arg in must))
            return held_77
        return True
    else:
        held_73 = needed_text or 'ids' in kinds.values()
        return held_73

def _without_cache_mark(part):
    if isinstance(part, dict):
        clean = dict(part)
        clean.pop('cache_control', None)
        return clean
    else:
        return part

def _messages_for_model(model, messages):
    if not model.startswith('anthropic/'):
        cleaned = []
        for msg_text in messages:
            parts = msg_text.get('content')
            if isinstance(parts, list):
                msg_text = dict(msg_text, content=list(map(_without_cache_mark, parts)))
            cleaned.append(msg_text)
        return cleaned
    elif not 3 > len(messages):
        held_78 = map(dict, messages)
        marked = list(held_78)
        for at in (0, len(marked) - 1):
            text = marked[at].get('content')
            if text:
                if isinstance(text, str):
                    marked[at]['content'] = [dict(type='text', text=text, cache_control=dict(type='ephemeral'))]
        return marked
    else:
        return messages

def _http_status(result_info):
    fault = result_info.error
    if isinstance(fault, dict):
        held_79 = fault.get('status')
        return held_79
    return None

def _latest_records(records, limit):
    kept, size = ([], 0)
    for bag in records[::-1]:
        held_80 = _trim_strings(bag, 160)
        piece = _json_line(held_80)
        if limit >= size + len(piece):
            kept.append(piece)
            size += len(piece)
    return kept

def _need_terms(text_line):
    lead_block, colon, tail = str(text_line).partition(':')
    if colon:
        held_81 = (_terms(lead_block), _terms(tail))
        return held_81
    return None

def _value_terms(fact_sheet):
    held_82 = _terms(' '.join((str(fv) for fv in fact_sheet.values())))
    return held_82

def _listing_records(records):
    out = []
    for bag in _walk_dicts(records):
        for key, cell in bag.items():
            if isinstance(cell, str):
                if 'id' in str(key).lower():
                    if ID_PATTERN.match(cell):
                        out.append((cell, bag))
                        break
    return out

def _is_real(field_value):
    held_83 = isinstance(field_value, (int, float)) and (not isinstance(field_value, bool))
    return held_83

def _field_slots(found):
    numeric, boolean, texts = ([], [], {})
    for ident, bag in found:
        for key, cell in bag.items():
            if ident == cell:
                continue
            elif isinstance(cell, bool):
                boolean.append(key)
            elif isinstance(cell, (int, float)):
                numeric.append(key)
            elif isinstance(cell, str):
                if 0 < len(cell) <= 40:
                    texts.setdefault(key, []).append(cell)
    repeated = list((kname for kname, vals in texts.items() if len(vals) >= 2 * len(set(vals))))
    label = max(repeated, key=lambda kname: len(texts[kname]), default=None)
    held_84 = [('price', _most_common(numeric), _is_real), ('stock', _most_common(boolean), lambda fv: isinstance(fv, bool)), ('label', label, lambda fv: isinstance(fv, str))]
    return held_84

def _most_common(keys):
    held_85 = max(dict.fromkeys(keys), default=None, key=keys.count)
    return held_85

def _facts_of(bag, ident):
    fact_sheet = {}
    for key, cell in bag.items():
        if isinstance(cell, str):
            if ident != cell:
                if 0 < len(cell) <= 120:
                    fact_sheet[key] = cell
    for child in bag.values():
        if isinstance(child, dict):
            fact_sheet.update(((kname, fv) for kname, fv in child.items() if isinstance(fv, (str, list))))
    return fact_sheet

def _result_row_ids(roots, ids):
    selected = set()
    for branch in _all_containers(roots):
        if not isinstance(branch, dict):
            dicts = list((mt for mt in branch if isinstance(mt, dict)))
            layouts = list((tuple(sorted(mt)) for mt in dicts))
            for member, layout in zip(dicts, layouts):
                if layouts.count(layout) >= 2:
                    if any((_is_listing_id(fv, ids) for fv in member.values())):
                        held_86 = id(member)
                        selected.add(held_86)
        else:
            continue
    return selected

def _listing_own_fields(node_list, ids):
    forms = {}
    for bag in node_list:
        owner = next((fv for fv in bag.values() if _is_listing_id(fv, ids)), None)
        if not None is owner:
            held_87 = sorted(bag)
            layout = tuple(held_87)
            for key, cell in bag.items():
                if isinstance(cell, str):
                    if cell not in ids:
                        forms.setdefault((owner, key, cell), set()).add(layout)
        else:
            continue
    held_88 = set((triple for triple, layouts in forms.items() if 2 <= len(layouts)))
    return held_88

def _constraint_parts(tail):
    plain = re.sub('\\([^)]*\\)', ' ', tail)
    if not _terms(plain):
        plain = tail
    keep, _, ban = plain.partition('!=')
    if not _terms(ban):
        held_89 = (plain, False, None)
        return held_89
    elif _terms(keep):
        held_91 = (keep, False, ban)
        return held_91
    else:
        held_90 = (ban, True, None)
        return held_90

def _property_members(fact_sheet, prop_terms):
    stated, singles = ([], [])
    for key, cell in fact_sheet.items():
        exact = _key_is_property(key, prop_terms)
        if exact or prop_terms <= _key_terms(key):
            stated += list((str(fv) for fv in (cell if isinstance(cell, list) else [cell])))
        if exact:
            if not isinstance(cell, list):
                chunks = list((p for p in _LIST_SPLIT.split(str(cell)) if p.strip()))
                if 1 == len(chunks):
                    if _terms(chunks[0]):
                        singles.append(chunks[0])
    member_list = list((p for fv in stated for p in [fv] + _LIST_SPLIT.split(fv) if _terms(p)))
    held_92 = (member_list, singles)
    return held_92
_BLURB_WORDS = 5

def _blurb_states(fact_sheet, condition):
    lead_block, colon, tail = str(condition).partition(':')
    target, negated, absent = _constraint_parts(tail)
    prop_terms, vw = (_terms(lead_block), _terms(target))
    if not (colon and prop_terms and vw) or negated or None is not absent or re.search('\\d', target):
        return False
    elif not _property_members(fact_sheet, prop_terms)[0]:
        for cell in fact_sheet.values():
            if isinstance(cell, str):
                said = _terms(cell)
            else:
                said = set()
            if _BLURB_WORDS <= len(said):
                if _covers(vw, said):
                    return True
        return False
    else:
        return False

def _whole_value_stated(fact_sheet, vw):
    for cell in fact_sheet.values():
        for fv in cell if isinstance(cell, list) else [cell]:
            text = str(fv)
            if any((_terms(p) == vw for p in [text] + _LIST_SPLIT.split(text))):
                return True
    return False

def _qualified_match(fact_sheet, condition):
    lead_block, _, tail = str(condition).partition(':')
    target, negated, absent = _constraint_parts(tail)
    prop_terms, vw = (_terms(lead_block), _terms(target))
    if not (prop_terms and vw) or negated or None is not absent or re.search('\\d', target):
        return False
    elif not _word_verdict(fact_sheet, condition) is not True:
        member_sets = list((_terms(mt) for mt in _property_members(fact_sheet, prop_terms)[0]))
        if not any((vw == ws for ws in member_sets)):
            held_93 = any((_covers(vw, ws) and (not _covers(ws, vw)) for ws in member_sets))
            return held_93
        else:
            return False
    else:
        return False

def _only_qualified_objections(fact_sheet, conditions, problems):
    if not (not problems or not conditions):
        for complaint in problems:
            said = _terms(complaint)
            named = list((text_line for text_line in conditions if _terms(str(text_line).partition(':')[0]) and _covers(_terms(str(text_line).partition(':')[0]), said)))
            if not named or not all((_qualified_match(fact_sheet, text_line) for text_line in named)):
                return False
        return True
    else:
        return False

def _word_verdict(facts, condition, phrasing=None):
    lead_block, colon, tail = str(condition).partition(':')
    target, negated, absent = _constraint_parts(tail)
    if not (None is not absent and _word_verdict(facts, '{}: != {}'.format(lead_block, absent)) is False):
        prop_terms, vw = (_terms(lead_block), _terms(target))
        if colon and prop_terms and vw:
            member_list, singles = _property_members(facts, prop_terms)
            numeric = re.search('\\d', target) is not None
            if member_list:
                if negated:
                    bans = list((_terms(p) for p in _LIST_SPLIT.split(target) if _terms(p)))
                    held_94 = not any((_covers(b, _terms(mt)) for mt in member_list for b in bans))
                    return held_94
                rel = _relation_of(target)
                if None is not rel:
                    if not phrasing:
                        held_96 = _figures_fit(rel, member_list)
                        return held_96
                    else:
                        held_95 = any((phrase in _flat(mt) for mt in member_list for phrase in phrasing))
                        return held_95
                if numeric:
                    if any((_covers(vw, _terms(mt)) for mt in member_list)):
                        return True
                    return None

                def states(part):
                    held_97 = _covers(vw, _terms(part))
                    return held_97

                def touches(part):
                    held_98 = states(part) or _covers(_terms(part), vw)
                    return held_98
                if singles:
                    if not any((touches(pager) for pager in singles)):
                        return False
                split = not all((touches(pager) for pager in singles))
                if any((states(mt) for mt in member_list)):
                    if split:
                        return None
                    return True
                if split or any((touches(mt) for mt in member_list)):
                    return None
                return False
            elif not (negated or numeric or _relation_of(target) is not None):
                if _whole_value_stated(facts, vw):
                    return True
                return None
            else:
                return None
        else:
            return None
    else:
        return False

def _relation_of(wish):
    text = wish.strip()
    sign = next((mt for mt in _RELATION_MARKS if text.lower().startswith(mt)), None)
    if not None is sign:
        number = re.search('\\d+(?:\\.\\d+)?', text[len(sign):])
        if not None is number:
            held_99 = (sign, float(number.group()), '+/-' in text or '±' in text)
            return held_99
        else:
            return None
    else:
        return None

def _figures_fit(rel, member_list):
    sign, target_value, both = rel
    seen = list((float(el) for mt in member_list for el in re.findall('-?\\d+(?:\\.\\d+)?', mt)))
    if both:
        seen = list((abs(el) for el in seen))
    if not seen:
        return None
    elif not 'covers' == sign:
        hits = sum((1 for el in seen if (target_value >= el if '<=' == sign else target_value <= el)))
        if not len(seen) == hits:
            if hits:
                return None
            return False
        else:
            return True
    else:
        signed = list((float(el) for mt in member_list for el in re.findall('-?\\d+(?:\\.\\d+)?', mt)))
        if not (both and any((0 > el for el in signed))):
            held_101 = min(seen) <= target_value <= max(seen)
            return held_101
        else:
            held_100 = -target_value >= min(signed) and target_value <= max(signed)
            return held_100

def _cart_target(ep, label_name, arguments):
    held_102 = ep.param_schemas[label_name].get('properties')
    if not held_102:
        held_102 = {}
    field_specs = held_102
    held_103 = _id_args(field_specs, arguments)
    own = list(held_103)
    if not own:
        lines = list((listing for listing, _ in _line_args(field_specs, arguments)))
        fresh = list((listing for listing in lines if listing not in ep.cart_ids))
        if fresh:
            held_105 = fresh[0]
            return held_105
        if lines:
            held_106 = lines[-1]
            return held_106
        return None
    else:
        held_104 = own[0]
        return held_104

def _cart_lines_arg(ep, tool_desc):
    if tool_desc.get('type') == 'array':
        entry = tool_desc.get('items')
    else:
        entry = None
    held = list(ep.cart_ids)
    if not isinstance(entry, dict) or not held:
        held_107 = (False, None)
        return held_107
    elif not ('string' in (entry.get('type') or []) and 'enum' not in entry):
        if isinstance(entry.get('properties'), dict):
            inner = entry.get('properties')
        else:
            inner = {}
        texts = _free_text_args(inner)
        counts = list((key for key, sub in inner.items() if isinstance(sub, dict) and sub.get('type') == 'integer'))
        if not (1 != len(texts) or 1 < len(counts)):
            lines = []
            for idx in held:
                line = {texts[0]: idx}
                if counts:
                    held_110 = int(ep.units_in_cart.get(idx) or 1)
                    line[counts[0]] = max(1, held_110)
                lines.append(line)
            held_111 = (True, lines)
            return held_111
        else:
            held_109 = (False, None)
            return held_109
    else:
        held_108 = (True, held)
        return held_108

def _id_args(props, dispatched):
    for key, field_value in dispatched.items():
        tool_desc = props.get(key)
        if isinstance(field_value, str):
            if not (isinstance(tool_desc, dict) and 'enum' in tool_desc):
                token = field_value.strip()
                if token:
                    if ' ' not in token:
                        yield field_value

def _line_args(props, dispatched):
    found = list(((field_value, None) for field_value in _id_args(props, dispatched)))

    def walk(tool_desc, branch, depth):
        if not 6 < depth:
            if isinstance(tool_desc, dict):
                tool_desc = tool_desc
            else:
                tool_desc = {}
            if isinstance(branch, dict):
                held_112 = tool_desc.get('properties')
                if not held_112:
                    held_112 = {}
                inner = held_112
                for key, cell in branch.items():
                    held_113 = inner.get(key)
                    walk(held_113, cell, depth + 1)
            elif isinstance(branch, list):
                if isinstance(tool_desc.get('items'), dict):
                    member = tool_desc.get('items')
                else:
                    member = {}
                for line in branch:
                    if isinstance(line, dict):
                        counts = list((cell for cell in line.values() if isinstance(cell, int) and (not isinstance(cell, bool))))
                        for listing in _id_args(member.get('properties') or {}, line):
                            found.append((listing, counts[0] if 1 == len(counts) else None))
                    walk(member, line, depth + 1)
        else:
            return
    for key, field_value in dispatched.items():
        if isinstance(field_value, (dict, list)):
            held_114 = props.get(key)
            walk(held_114, field_value, 1)
    return found

class Wishes:

    def __init__(self):
        for bucket in ('stated', 'revealed', 'dropped', 'free', 'history'):
            setattr(self, bucket, [])
        self.newest_lines = self.budget = None
        self.heard = set()

    def summary(self, currency, quantity):

        def alive(pool, tag):
            held_115 = list((element + tag for element in pool if element not in self.dropped))
            return held_115
        if not None is self.newest_lines:
            entry_list = list(self.newest_lines)
        else:
            entry_list = alive(self.stated, ' (in the request)') + alive(self.revealed, ' (told by the shopper)')
        chunks = ['; '.join(entry_list) or 'as in the request']
        if None is not self.budget:
            held_116 = ('total price no more than %s %s' % (self.budget, currency)).rstrip()
            chunks.append(held_116)
        chunks.append('quantity %s' % (quantity,))
        tails = ((self.dropped, 'no longer wanted: ', '; '), (self.free, 'the shopper does not mind about: ', ', '))
        chunks.extend((opener + glue.join(pool) for pool, opener, glue in tails if pool))
        held_117 = '; '.join(chunks)
        return held_117

    @staticmethod
    def _add_fresh(target, incoming):
        new_lines = []
        for element in incoming or []:
            if isinstance(element, str):
                if element:
                    if element not in target:
                        target.append(element)
                        new_lines.append(element)
        return new_lines

    def merge_reading(self, parsed_reading):
        current = parsed_reading.get('requirements_now')
        if isinstance(current, list):
            if not any((not isinstance(element, str) for element in current)):
                held_118 = filter(str.strip, current)
                self.newest_lines = list(held_118)
        held_119 = map('added requirement: '.__add__, self._add_fresh(self.revealed, parsed_reading.get('needs')))
        notes = list(held_119)
        self._add_fresh(self.free, parsed_reading.get('no_preference'))
        notes += map('dropped: '.__add__, self._add_fresh(self.dropped, parsed_reading.get('dropped')))
        held_120 = parsed_reading.get('new_budget')
        cap = _number_or_none(held_120)
        if None is not cap:
            if 0 < cap:
                if self.budget != cap:
                    self.budget = cap
                    notes.append('the budget was changed to %s' % (cap,))
        if parsed_reading.get('unavailable'):
            notes.append('a listing can no longer be bought: do not order it, and remove it from the cart')
        return notes
_INITIAL_VALUES = ((0, 'step messages_used order_attempts refusals needs_revision cart_changed_at cart_generation refused_at empty_plans'), (0.0, 'cost'), (-1, 'cart_seen_at'), (True, 'planner_on'), (False, 'halted order_checked decline_checked key_dry cart_used audit_done split_retried urgent_tried'), (None, 'message_budget quantity_field add_signature last_cart_view answer_reading split_pending forced_basis question_cost'), ('', 'asking sibling_next queued_reply rerank_reason unclear_next'))
_INITIAL_CONTAINERS = ((set, 'seen_ids cart_checked qty_warned held_arms relooked faulty proc_held'), (list, 'cart_ids records recent asked asked_props reask_later sibling_props proven_fits procedures messages trace'), (dict, 'cart_state units_in_cart sent_units property_menu verdicts empty_holds mentioned decline_holds faults roles fixed calls_by_role plan need_homes loose_ok'))

class ModelsAndStore:

    def _note_trace(self, kind, **field_specs):
        held_121 = dict(kind=kind, **field_specs)
        self.trace.append(held_121)

    def over_soft_spend(self):
        held_122 = SPEND_SOFT < self.cost
        return held_122

    def seconds_left(self):
        held_123 = EPISODE_SECONDS - (time.monotonic() - self.start_clock)
        return held_123

    def steps_left(self):
        held_124 = self.max_steps - self.step
        return held_124

    def ask_models(self, messages, tools, models, max_tokens):
        payment_refusals = 0
        for model in models:
            remaining = self.seconds_left()
            if not LAST_SECONDS - 10 > remaining:
                held_126 = max(8, remaining + 15 - LAST_SECONDS)
                held_125 = min(75, held_126)
                _model_proxy.timeout = int(held_125)
                result_info = self._call_model(model, messages, tools, max_tokens)
                reply_obj = self._first_message(result_info.data)
                if not reply_obj:
                    if _http_status(result_info) in (402, 403):
                        payment_refusals = payment_refusals + 1
                    if not result_info.error:
                        reason = 'empty'
                    else:
                        reason = str(result_info.error)[:300]
                    self._note_trace('inference_failure', error=reason, model=model)
                else:
                    reply_obj['model'] = model
                    return reply_obj
            else:
                return None
        if models:
            if len(models) == payment_refusals:
                self.key_dry = True
                self._note_trace('key_dry', models=list(models))
        return None

    def _call_model(self, model, messages, tools, max_tokens):
        body = {'model': model, 'messages': _messages_for_model(model, messages), 'temperature': 0, 'max_tokens': max_tokens * _ROOM_FACTOR.get(model, 1)}
        if tools:
            body.update(tool_choice='auto', tools=tools)
        result_info = _model_proxy.post_verbose('/inference/chat/completions', body)
        for wait in _RETRY_PAUSES:
            held_127 = None is result_info.data
            if held_127:
                held_127 = _http_status(result_info) in _BRIEF_FAULTS
            passing = held_127
            if not (not passing or self.seconds_left() < LAST_SECONDS + 30):
                time.sleep(wait)
                result_info = _model_proxy.post_verbose('/inference/chat/completions', body)
            else:
                break
        return result_info

    def _first_message(self, data):
        if not (not isinstance(data, dict) or not data.get('choices')):
            held_128 = data.get('usage')
            if not held_128:
                held_128 = {}
            usage = held_128
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            first = data['choices'][0]
            msg_text = dict(first.get('message') or {})
            msg_text['cut'] = first.get('finish_reason') == 'length'
            if not (msg_text.get('content') or msg_text.get('tool_calls')):
                return None
            else:
                return msg_text
        else:
            return None

    def store_call(self, action):
        wrapper = dict(self.binding, action=action, step=self.step + 1)
        for attempt in range(1, 5):
            remaining = self.seconds_left()
            if not 2 > remaining:
                held_130 = max(3, remaining - 1)
                held_129 = min(95, held_130)
                _store_proxy.timeout = int(held_129)
                result_info = _store_proxy.post_verbose('/environment/call', wrapper)
                if isinstance(result_info.data, dict):
                    held_131 = self._count_step(result_info.data)
                    return held_131
                elif not self._retry_store(result_info.error or {}, wrapper, attempt):
                    return None
            else:
                return None
        return None

    def _count_step(self, data):
        counted = data.get('step')
        if not isinstance(counted, int):
            self.step = self.step + 1
        else:
            self.step = counted
        if data.get('finished'):
            self.finished = True
        return data

    def _retry_store(self, error, wrapper, attempt):
        status_line = error.get('status')
        body = str(error.get('body') or '')
        self._note_trace('env_failure', status=status_line, body=body[:300])
        if 402 == status_line:
            self.halted = True
            return False
        elif 429 == status_line:
            spread = 1.0 + 0.15 * ((7 * self.step + 3 * attempt) % 5)
            wait = 0.6 * attempt * spread
            if not self.seconds_left() <= wait + LAST_SECONDS:
                time.sleep(wait)
                return True
            else:
                return False
        elif not 409 != status_line:
            if 'finished' in body:
                self.finished = True
                return False
            expected = re.search('expected (\\d+)', body)
            if not expected:
                return False
            self.step = int(expected.group(1)) - 1
            wrapper['step'] = self.step + 1
            return True
        else:
            held_132 = status_line not in _STORE_FINAL_FAULTS
            return held_132

    def ask_json(self, prompt, max_tokens=700, models=None):
        held_133 = models
        if not held_133:
            held_133 = [CHECKER_MODEL, SECOND_VENDOR_MODEL, DRIVER_MODEL]
        seat_list = held_133
        ask = lambda who, room: self.ask_models([{'role': 'user', 'content': prompt}], None, who, room)
        read = lambda raw_obj: None if None is raw_obj else _reply_object(raw_obj.get('content') or '')
        first = ask(seat_list, max_tokens)
        if not None is first:
            parsed = read(first)
            if not (None is not parsed or not first.get('cut')):
                roomy = max_tokens * 3
                self.trace.append({'kind': 'cut_reply', 'limit': max_tokens})
                second = ask(seat_list, roomy)
                parsed = read(second)
                if not (None is not parsed or None is second or (not second.get('cut'))):
                    who = second.get('model')
                    if who in seat_list:
                        rest = seat_list[seat_list.index(who) + 1:]
                    else:
                        rest = []
                    if not rest:
                        return None
                    self.trace.append({'kind': 'cut_reply', 'limit': roomy, 'next': rest[0]})
                    held_134 = read(ask(rest, roomy))
                    return held_134
                else:
                    return parsed
            else:
                return parsed
        else:
            return None

class TextFields:

    @staticmethod
    def _split_need(need_line):
        held_135 = str(need_line).partition(':')
        return held_135

    @staticmethod
    def _need_property_terms(need_line):
        held_136 = _terms(str(need_line).partition(':')[0])
        return held_136

    @staticmethod
    def _key_as_words(field_name):
        held_137 = re.sub('[_.]+', ' ', re.sub('(?<=[a-z0-9])(?=[A-Z])', ' ', str(field_name)))
        return held_137

    @staticmethod
    def _short_text_fields(record_blob, limit):
        held_138 = list(((label_key, cell) for label_key, cell in record_blob.items() if isinstance(cell, str) and 0 < len(cell) <= limit))
        return held_138

    @staticmethod
    def _int_fields(record_blob):
        held_139 = dict(((label_key, cell) for label_key, cell in record_blob.items() if isinstance(cell, int) and (not isinstance(cell, bool))))
        return held_139

    def _arg_specs(self, tool, strict=False):
        if strict:
            arg_shape = self.param_schemas[tool]
        else:
            arg_shape = self.param_schemas.get(tool) or {}
        held_140 = arg_shape.get('properties') or {}
        return held_140

class RequestIntake:

    def read_request(self):
        held_141 = map(_tool_line, (tool['function'] for tool in self.tools))
        tool_catalog = '\n'.join(held_141)
        query, opening = (self.policy.get('query') or '', self.policy.get('opening'))
        story = 'REQUEST FROM THE SHOPPER:\n' + str(query)
        if None is not opening:
            story = story + "\n\nTHE STORE'S FIRST VIEW:\n" + _json_line(opening)
        lead_block = dict(type='text', text=INTAKE_PROMPT + '\n\nTOOLS OF THIS STORE:\n' + tool_catalog, cache_control=dict(type='ephemeral'))
        turn = [dict(role='user', content=[lead_block, dict(text=story, type='text')])]
        first, roomy, last = _INTAKE_TRIES
        reply_obj, self.plan = self._intake_once(turn, first)
        if not self.plan:
            if reply_obj:
                if reply_obj.get('cut'):
                    reply_obj, self.plan = self._intake_once(turn, roomy)
        if not (self.key_dry or _reading_usable(self.plan)):
            self._note_trace('setup_retry', had=sorted(self.plan))
            again = self._intake_once(turn, last)[1]
            if _reading_usable(again):
                self.plan = again
        if not self._apply_reading(self.plan):
            self._roles_only(tool_catalog)

    def _intake_once(self, turn, seating):
        seat_list, tokens = seating
        reply_obj = self.ask_models(turn, None, seat_list, tokens)
        if reply_obj:
            text = reply_obj.get('content') or ''
        else:
            text = ''
        held_142 = (reply_obj, _reply_object(text) or {})
        return held_142

    def _roles_only(self, tool_catalog):
        held_143 = self.ask_json(ROLES_PROMPT + '\n\nTHE TOOL LIST:\n' + tool_catalog, 400, _ROLE_MODELS)
        if not held_143:
            held_143 = {}
        reply = held_143
        named = reply.get('roles')
        if isinstance(named, dict):
            self.plan['roles'] = named
            self._accept_roles(named)
        self._note_trace('roles_fallback', roles=self.roles)

    def _accept_roles(self, named):
        self.roles, self.fixed = ({}, {})
        for bag, element in named.items() if isinstance(named, dict) else ():
            if isinstance(element, dict):
                tool_desc = element
            else:
                tool_desc = {'tool': element, 'fixed': {}}
            tool = tool_desc.get('tool')
            if not (not isinstance(tool, str) or tool not in self.param_schemas):
                self.roles[bag] = tool
                if bag in FINISH_JOBS:
                    preset = tool_desc.get('fixed')
                    self.fixed[bag] = dict(preset) if isinstance(preset, dict) else {}
            else:
                continue
        named_ok = len(self.roles)
        self._cross_check_roles()
        self._settle_closing()
        return named_ok

    def _cross_check_roles(self):
        read = _roles_from_shapes(self.tools)
        shapes = _tool_profiles(self.tools)
        changed, differs = ({}, {})
        for bag, options in read.items():
            mine = self.roles.get(bag)
            names = list((tool for tool, _ in options))
            if mine in names:
                pins = options[names.index(mine)][1]
                if bag in FINISH_JOBS:
                    if pins:
                        if not self.fixed.get(bag):
                            self.fixed[bag] = dict(pins)
                            changed[bag] = {'tool': mine, 'fixed': pins}
                continue
            elif not (mine and (mine not in shapes or _shape_suits_job(bag, shapes[mine]))):
                taken = set((tool for job, tool in self.roles.items() if bag != job and job not in FINISH_JOBS))
                if 1 < len(options):
                    free = list((opt for opt in options if opt[0] not in taken))
                else:
                    free = options
                if not 1 != len(free):
                    self.roles[bag] = free[0][0]
                    if bag in FINISH_JOBS:
                        self.fixed[bag] = dict(free[0][1] or {})
                    changed[bag] = {'was': mine, 'now': free[0][0]}
                else:
                    continue
            else:
                differs[bag] = {'model': mine, 'list': names}
                continue
        self._note_trace('roles_code', read=dict(((bag, list((tool for tool, _ in opts))) for bag, opts in read.items())), changed=changed, differs=differs)

    def _apply_reading(self, parsed_reading):
        held_144 = parsed_reading.get('roles')
        named_ok = self._accept_roles(held_144)
        held_145 = parsed_reading.get('procedures')
        self.procedures = self._procedures_from(held_145)
        held_146 = parsed_reading.get('question_cost')
        self.question_cost = self._ask_price_if_quoted(held_146)
        held_147 = parsed_reading.get('message_budget')
        messages_allowed = _number_or_none(held_147)
        if None is not messages_allowed:
            if 0 <= messages_allowed:
                self.message_budget = int(messages_allowed)
        wants = self.wishes
        held_149 = filter(None, parsed_reading.get('requirements') or [])
        held_148 = map(str, held_149)
        wants.stated = list(held_148)
        wants.budget = parsed_reading.get('price_cap')
        held_150 = parsed_reading.get('step_budget')
        steps_allowed = _number_or_none(held_150)
        if None is not steps_allowed:
            if 0 < steps_allowed < self.max_steps:
                self.max_steps = int(steps_allowed)
        return named_ok

    def _quoted_in_request(self, quote):
        fold = lambda snip: ' '.join(str(snip or '').lower().split())
        needle = fold(quote)
        held_151 = bool(needle) and needle in fold(self.policy.get('query'))
        return held_151

    def _ask_price_if_quoted(self, element):
        if isinstance(element, dict):
            if _number_or_none(element.get('amount')) is not None:
                if self._quoted_in_request(element.get('quote')):
                    return element
        return None

    def _procedures_from(self, entry_list):
        steps = []
        for need_line in entry_list if isinstance(entry_list, list) else []:
            if isinstance(need_line, dict) and self._quoted_in_request(need_line.get('quote')):
                rname = need_line.get('role')
                if not (rname not in self.roles or rname in FINISH_JOBS or 'cart' == rname):
                    times, when = (_number_or_none(need_line.get('count')), need_line.get('before'))
                    held_152 = need_line.get('do')
                    if not held_152:
                        held_152 = need_line.get('quote')
                    snip = held_152
                    held_153 = dict(do=str(snip)[:200], quote=str(need_line.get('quote'))[:200], role=rname, count=1 if None is times else max(1, int(times)), after_pick=need_line.get('after_pick') is True, before=when if when in ('order', 'decline') else 'any')
                    steps.append(held_153)
                else:
                    continue
            else:
                continue
        return steps

    def _procedures_due(self, kind):
        owed = []
        for step_proc in self.procedures:
            late = step_proc['after_pick']
            if not (step_proc['before'] not in ('any', kind) or (late and 'order' != kind)):
                if late:
                    floor = self.cart_changed_at
                else:
                    floor = 0
                if len(list((pager for pager in self.calls_by_role.get(step_proc['role'], []) if floor < pager))) < step_proc['count']:
                    owed.append(step_proc)
            else:
                continue
        return owed

class FinishJobs:
    _CLOSING_FIX_HEAD = 'Correct the mapping of the finishing tools. Expected answer (JSON alone, nothing around it):\n{"order": {"tool": name, "fixed": {argument: value}}, "decline": {"tool": name, "fixed": {...}}}\nMeaning of the keys -\n  order: tool and fixed argument values that purchase the cart\'s contents\n  decline: tool and fixed argument values that end the task without a purchase\n  fixed: argument -> exact value, for each required argument restricted to listed options or to true/false; if one tool does both jobs, the two entries fix the same arguments at different values\n\nProblems found in the previous mapping:\n'

    def _pinned_args(self, bag):
        held_154 = self.fixed.get(bag) or {}
        return held_154

    def _closing_faults(self, bag):
        tool = self.roles.get(bag)
        if tool:
            arg_specs = self._arg_specs(tool, True)
            pinned = self._pinned_args(bag)
            held_156 = self.param_schemas[tool].get('required')
            if not held_156:
                held_156 = []
            must = held_156
            out = list(("{} requires the option argument '{}', which '{}' does not set it".format(tool, a, bag) for a in must if _is_selector(arg_specs.get(a)) and a not in pinned))
            for arg in list(pinned):
                tool_desc = arg_specs.get(arg)
                if isinstance(tool_desc, dict):
                    spelled, wrong = _fit_schema(pinned[arg], tool_desc, arg)
                    if wrong:
                        out.append("bad fixed value for '{}': {}.{} = {!r} (".format(bag, tool, arg, pinned[arg]) + '; '.join(wrong) + ')')
                    else:
                        pinned[arg] = spelled
                        continue
                else:
                    held_157 = "unknown argument '{}' pinned by '{}' ({} has no such argument)".format(arg, bag, tool)
                    out.append(held_157)
                    continue
            held_158 = out + self._closing_clashes(bag, tool, pinned)
            return held_158
        else:
            held_155 = ["the '{}' job is mapped to no tool".format(bag)]
            return held_155

    def _closing_clashes(self, bag, tool, pinned):
        notes = []
        rival = {'order': 'decline'}.get(bag, 'order')
        if self.roles.get(rival) == tool:
            theirs = self._pinned_args(rival)
            if pinned == theirs:
                held_159 = 'order and decline come out as the very same call of {}'.format(tool)
                notes.append(held_159)
            elif set(theirs) != set(pinned):
                held_161 = sorted(pinned)
                held_162 = sorted(theirs)
                held_160 = 'order and decline both use {} but fix different argument sets {} and {}; the argument that tells them apart must be fixed in both'.format(tool, held_161, held_162)
                notes.append(held_160)
        if not pinned:
            for rname in self.roles:
                if tool == self.roles[rname]:
                    if rname not in FINISH_JOBS:
                        held_163 = "{} doubles as the '{}' tool for '{}' with no fixed value to keep the two uses apart".format(tool, rname, bag)
                        notes.append(held_163)
        return notes

    def _closing_reading(self, prompt, seat_list):
        held_164 = self.ask_json(prompt, 500, seat_list)
        if not held_164:
            held_164 = {}
        reply = held_164
        selected = ((bag, reply.get(bag)) for bag in FINISH_JOBS)
        held_165 = dict(((bag, (raw_obj['tool'], dict(raw_obj['fixed']) if isinstance(raw_obj.get('fixed'), dict) else {})) for bag, raw_obj in selected if isinstance(raw_obj, dict) and raw_obj.get('tool') in self.param_schemas))
        return held_165

    def _selector_values(self, tool, pins):
        arg_specs = self._arg_specs(tool)
        kept = list((arg for arg in pins if _is_selector(arg_specs.get(arg))))
        held_166 = dict(((arg, _fit_schema(pins[arg], arg_specs[arg], arg)[0]) for arg in kept))
        return held_166

    def _take_closing(self, read_reply):
        for bag in read_reply:
            self.roles[bag], self.fixed[bag] = read_reply[bag]

    def _closing_now(self):
        held_167 = dict(((bag, (self.roles[bag], self._pinned_args(bag))) for bag in FINISH_JOBS if bag in self.roles))
        return held_167

    def _same_closing(self, one, two):
        if not two[0] != one[0]:
            held_168 = self._selector_values(*one) == self._selector_values(*two)
            return held_168
        else:
            return False

    def _closing_second_look(self, tool_catalog):
        own_row = self._closing_now()
        other = self._closing_reading(FINISH_ROLES_PROMPT + '\n\nThe tools:\n' + tool_catalog, [DRIVER_MODEL, SPARE_MODEL])
        if not other:
            return
        elif not any((not self._same_closing(own_row[j], other[j]) for j in own_row if j in other)):
            extra = dict(((j, other[j]) for j in other if j not in own_row))
            if extra:
                self.trace.append({'kind': 'finish_read', 'took': extra})
                self._take_closing(extra)
        else:
            self._closing_referee(own_row, other, tool_catalog)
            return

    def _closing_referee(self, own_row, other, tool_catalog):

        def as_text(read_reply):
            held_169 = _json_line(dict(((j, {'tool': pair[0], 'fixed': pair[1]}) for j, pair in read_reply.items())))
            return held_169
        settle = '\n\n'.join([REFEREE_PROMPT, 'The tools:\n' + tool_catalog, 'The first reading: ' + as_text(own_row) + '\nThe second reading: ' + as_text(other)])
        final = self._closing_reading(settle, [LARGE_MODEL, SECOND_VENDOR_MODEL, SPARE_MODEL])
        self.trace.append({'kind': 'finish_disagree', 'setup': own_row, 'second': other, 'final': final})
        self._take_closing(final)

    def _settle_closing(self):
        tool_catalog = '\n'.join((_tool_line(need_line['function']) for need_line in self.tools))
        self._closing_second_look(tool_catalog)
        rounds = 0
        while 2 > rounds:
            rounds = rounds + 1
            faults = sum((self._closing_faults(j) for j in FINISH_JOBS), [])
            if not (self.key_dry or not faults):
                self.trace.append({'kind': 'finish_repair', 'issues': faults})
                snip = self._CLOSING_FIX_HEAD + '\n'.join(('- ' + fobj for fobj in faults)) + "\n\nThe store's tools:\n" + tool_catalog
                held_170 = self._closing_reading(snip, [LARGE_MODEL, SECOND_VENDOR_MODEL, DRIVER_MODEL, SPARE_MODEL])
                self._take_closing(held_170)
            else:
                break
        for bag in list((j for j in FINISH_JOBS if j in self.roles and self._closing_faults(j))):
            del self.roles[bag]
            self.fixed.pop(bag, None)

    def closing_kind(self, label_name, arguments):
        held_171 = arguments
        if not held_171:
            held_171 = {}
        dispatched = held_171
        for bag in (j for j in FINISH_JOBS if self.roles.get(j) == label_name):
            pins = self._pinned_args(bag)
            if not any((dispatched.get(a) != pins[a] for a in pins)):
                return bag
        return None

    def _roles_summary(self):
        logged = dict(self.roles)
        for bag in FINISH_JOBS:
            if bag in logged:
                logged[bag] = {'tool': logged[bag], 'fixed': self._pinned_args(bag)}
        lacking = list((j for j in ('search', 'details', 'cart', 'order', 'decline') if j not in self.roles))
        if self.roles.get('message'):
            if self._message_args('?') is None:
                lacking.append('message: some required argument takes no plain text')
        logged['gaps'] = lacking
        return logged
    _FINISH_PURPOSE = {'order': 'to buy what is in the cart', 'decline': 'to end without buying'}

    def _finish_lines(self):
        out = []
        for bag in FINISH_JOBS:
            tool = self.roles.get(bag)
            if tool:
                held_172 = self._pinned_args(bag)
                choice = self._selector_values(tool, held_172)
                if choice:
                    suffix = ' with ' + _json_line(choice)
                else:
                    suffix = ''
                held_173 = '- {}: {}{}'.format(self._FINISH_PURPOSE[bag], tool, suffix)
                out.append(held_173)
        return out

class Briefing:

    def reading_brief(self):
        plan = self.plan
        if plan:
            rank, detail = (plan.get('priority'), plan.get('priority_detail'))
            text_lines = ['How the request was read - check every line against its own words:']
            if plan.get('product'):
                held_175 = plan.get('product')
                held_174 = '- product asked for: {}'.format(held_175)
                text_lines.append(held_174)
            held_176 = '- quantity: {}'.format(plan.get('quantity') or 1)
            text_lines.append(held_176)
            text_lines.extend(('- requirement from the request: {}'.format(needed) for needed in plan.get('requirements') or []))
            held_177 = self._basket_lines()
            text_lines.extend(held_177)
            if plan.get('price_cap') is not None:
                held_178 = '- the total price may be at most: {} {}'.format(plan.get('price_cap'), plan.get('currency') or '').rstrip()
                text_lines.append(held_178)
            if plan.get('in_stock_only'):
                text_lines.append('- only listings in stock (on hand) can be bought')
            if not rank in ('cheapest', 'best'):
                text_lines.append(self._NO_RANKING)
            else:
                text_lines.append('- ranking: take the {} one of the listings that meet every condition'.format(rank) + (' ({})'.format(detail) if detail else ''))
            for field_name, opener in _BRIEF_LISTS:
                text_lines.extend((opener + str(unit_ref) for unit_ref in plan.get(field_name) or []))
            if None is not self.message_budget:
                held_179 = '- messages allowed: {}'.format(self.message_budget)
                text_lines.append(held_179)
            held_180 = '- steps allowed: {}'.format(self.max_steps)
            text_lines.append(held_180)
            held_181 = '\n'.join(text_lines + self._finish_lines())
            return held_181
        else:
            return ''
    _NO_RANKING = '- ranking: none (neither cheapest nor best); unless the request says otherwise, any listing within the budget that meets every requirement is right, and comparing prices only wastes steps'

    def status_line(self):
        remaining = self.steps_left()
        if None is self.message_budget:
            cap = ''
        else:
            cap = ' of {}'.format(self.message_budget)
        held_182 = max(0, self.seconds_left() - LAST_SECONDS)
        clock = int(held_182)
        snip = '[status] step count: {} used of {}, {} remaining; messages sent: {}{}; seconds remaining: about {}.\n[requirements in force] '.format(self.step, self.max_steps, remaining, self.messages_used, cap, clock) + self.wants_text()
        tiers = ((lambda: 2 >= remaining, self._LAST_MOVES), (lambda: 5 >= remaining or self.seconds_left() < WRAP_UP_SECONDS or SPEND_FINISH < self.cost, self._WRAP_MANY if self.basket_items() else self._WRAP_ONE), (lambda: 0 == self.order_attempts and 14 >= remaining, self._STEPS_RUNNING_OUT))
        held_183 = snip + next((' ' + note for test, note in tiers if test()), '')
        return held_183
    _LAST_MOVES = 'FINAL STEPS - a verified fit is in the cart already: order it now. None in the cart: decline.'
    _WRAP_ONE = 'WRAP UP NOW: keep just one verified line in the cart; check its stock one last time; then place the order, or decline if no line qualifies.'
    _WRAP_MANY = 'WRAP UP NOW: the cart needs a verified line for every requested product, each with its own quantity; check their stock one last time; then place the order, or decline if no line qualifies.'
    _STEPS_RUNNING_OUT = 'Step check: putting a refused order right takes about 4 steps, and with 7 left the search and detail tools are taken away; put the strongest verified listing in the cart and buy it soon.'

    def wants_text(self):
        self._anchor_stated()
        many = self.basket_items()
        text_lines = [self.wishes.summary(self.plan.get('currency') or '', 'for each product as listed below' if many else self.units_wanted())]
        held_184 = self._basket_lines()
        text_lines.extend(held_184)
        for clause in self.need_lines():
            phrasing = self._store_phrases(clause)
            if phrasing:
                quoted = ' or '.join(('"{}"'.format(term_word) for term_word in phrasing))
                held_186 = self._split_need(clause)[0].strip()
                held_185 = "- {}: the phrase the request uses, {}, is this store's own wording of the value; only a listing that states it qualifies, and figures worked out from other wordings do not count".format(held_186, quoted)
                text_lines.append(held_185)
        for prop_name, wish in self.loose_ok.items():
            held_187 = '- {}: when asked which listed value with "{}" was meant, the shopper did not mind; any listing whose {} contains the words "{}" meets it, whatever else its value says'.format(prop_name, wish, prop_name, wish)
            text_lines.append(held_187)
        held_188 = '\n'.join(text_lines)
        return held_188

    @staticmethod
    def _values_under(row_map, need_terms):
        for listing_row in row_map:
            for field_name, datum in (listing_row.get('facts') or {}).items():
                if need_terms <= _terms(field_name) or _key_is_property(field_name, need_terms):
                    yield datum

    def _store_phrases(self, condition, row_map=None):
        pname, sep, wanted = self._split_need(condition)
        if sep:
            rel = _relation_of(wanted)
        else:
            rel = None
        need_terms = _terms(pname)
        if not (not need_terms or None is rel or 'covers' != rel[0]):
            target, request = (rel[1], _flat(self.policy.get('query')))
            hits = []
            if None is row_map:
                row_index = self.listing_table()
            else:
                row_index = row_map
            for datum in self._values_under(row_index.values(), need_terms):
                for snip in map(str, datum if isinstance(datum, list) else [datum]):
                    for piece in [snip, *re.split('[,;|]', snip)]:
                        flat = _flat(piece)
                        if not flat or flat in hits or flat not in request:
                            continue
                        elif target in list((float(cnt) for cnt in re.findall('\\d+(?:\\.\\d+)?', flat))):
                            hits.append(flat)
            return hits
        else:
            held_189 = []
            return held_189

class ShopperNews:

    def _news_prompt(self, new_lines, question):
        past = self.wishes.history
        sections = [('The request as it was first written', str(self.policy.get('query') or '')), ('The requirements so far', self.wants_text()), ('The conversation up to now, oldest first', '\n'.join(past[-12:]) if past else None), ("The assistant's question", question or None), ('What the shopper just said', '\n'.join(new_lines))]
        held_190 = '\n\n'.join([NEWS_PROMPT] + list(('{}:\n{}'.format(title, body) for title, body in sections if None is not body)))
        return held_190

    def hear_shopper(self, text_lines, question):
        need_state = self.wishes
        new_lines = list((ln for ln in text_lines if ln not in need_state.heard))
        if new_lines:
            need_state.heard.update(new_lines)
            if not self.seconds_left() < WRAP_UP_SECONDS:
                prompt = self._news_prompt(new_lines, question)
                need_state.history.append(('asked: {} -> '.format(question) if question else '') + 'shopper: ' + ' / '.join(new_lines))
                read_reply = self.ask_json(prompt, 1500, [READING_MODEL, CHECKER_MODEL])
                if read_reply:
                    if question:
                        held_191 = read_reply.get('answer_class')
                        held_192 = ' '.join(new_lines)
                        self.answer_reading = self._vetted_answer(held_191, held_192)
                        if not self.asking:
                            read_reply['no_preference'] = []
                    chain = (self._refile_needs, self._guard_stated if question else None, self._restore_stated, self._drop_unsaid)
                    for screen in chain:
                        if None is not screen:
                            screen(read_reply, new_lines)
                    known = list(need_state.revealed)
                    changes = self._merge_reading(read_reply)
                    held_193 = list((ent for ent in need_state.revealed if ent not in known))
                    self._file_needs_by_product(held_193, new_lines)
                    reply = str(read_reply.get('reply') or '').strip()
                    if reply:
                        if not question:
                            if self.roles.get('message'):
                                self.queued_reply = reply
                    if changes:
                        tail = ' (' + '; '.join(changes) + ')'
                    else:
                        tail = ''
                    held_194 = 'What it means: ' + str(read_reply.get('meaning') or '') + tail
                    return held_194
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def _file_needs_by_product(self, new_needs, text_lines):
        items = self.basket_items()
        if not (not items or not new_needs):
            products = list((el['product'] for el in items))
            if self.asking:
                owner = _owning_product(self, self.asking)
            else:
                owner = ''
            for ent in new_needs:
                held_195 = ' '.join(list(text_lines) + [ent])
                said = self._stems(held_195)
                named = list((seg for seg in products if self._stems(' '.join(_distinct_terms(seg, products))) & said))
                if 1 == len(named):
                    home = named[0]
                else:
                    home = owner
                if home in products:
                    self.need_homes[ent] = home
        else:
            return

    def _merge_reading(self, read_reply):
        need_state = self.wishes
        dropped_before, budget_before = (len(need_state.dropped), need_state.budget)
        changes = need_state.merge_reading(read_reply)
        if changes:
            self.needs_revision = self.needs_revision + 1
            moved = (dropped_before, budget_before) != (len(need_state.dropped), need_state.budget)
            if moved:
                if self.plan.get('priority') == 'cheapest':
                    self.rerank_reason = '; '.join(need_state.dropped[dropped_before:]) or 'the budget'
        return changes
    _ANSWER_KINDS = ('reveal', 'not_picky', 'split', 'unclear', 'other')

    def _vetted_answer(self, raw_obj, said):
        if not (not isinstance(raw_obj, dict) or raw_obj.get('class') not in self._ANSWER_KINDS):
            field_specs = dict(((label_name, str(raw_obj.get(label_name) or '')) for label_name in ('key', 'value', 'relation')))
            quote = str(raw_obj.get('quote') or '').strip().lower()
            held_197 = bool(quote)
            if held_197:
                held_197 = quote in said.lower()
            trusted = held_197
            if trusted:
                if 'reveal' == raw_obj['class']:
                    trusted = self._about_listing_property(field_specs['key'], field_specs['value'].strip().lower())
            if trusted:
                held_198 = {'class': raw_obj['class'], **field_specs}
                return held_198
            held_199 = {'class': 'other'}
            return held_199
        else:
            held_196 = {'class': 'other'}
            return held_196

    def _about_listing_property(self, key_text, datum):
        tag_names, values = self._sheet_vocabulary()
        field_name = _terms(key_text)
        if not (field_name and (field_name <= _terms(self.asking) or any((tag >= field_name for tag in tag_names)))):
            held_200 = bool(datum) and datum in values
            return held_200
        else:
            return True

    def _property_in_doubt(self, have, tag_names):
        asked = _terms(self.asking)
        if asked and (asked >= have or have >= asked):
            return True
        elif not have in list((fobj for fobj in (_terms(el) for el in self.wishes.free or []) if fobj)):
            given_reqs = list((self._need_property_terms(need_entry) for need_entry in self.wishes.stated or []))
            held_201 = have not in tag_names or have in given_reqs
            return held_201
        else:
            return True

    @staticmethod
    def _property_named_in(ln, pname, datum, tag_names):
        said = _terms(ln)
        if not (_terms(pname) <= said or not _covers(datum, said)):
            found_keys = list((term_word for term_word in tag_names if said >= term_word and (not term_word & datum)))
            if found_keys:
                widest = max(found_keys, key=len)
                held_202 = ' '.join((term_word for term_word in re.findall('[a-z0-9]+', ln.lower()) if term_word in widest))
                return held_202
            else:
                return None
        else:
            return None

    def _refiled_need(self, need_line, text_lines, tag_names):
        if isinstance(need_line, str) and ':' in need_line:
            pname, constraint = need_line.split(':', 1)
            datum = _terms(constraint) - _terms(pname)
            if not (not datum or not self._property_in_doubt(_terms(pname), tag_names)):
                homes = (self._property_named_in(ln, pname, datum, tag_names) for ln in text_lines)
                held_203 = next((home + ':' + constraint for home in homes if None is not home), None)
                return held_203
            else:
                return None
        else:
            return None

    def _refile_needs(self, read_reply, text_lines):
        tag_names = list((term_word for term_word in self._sheet_vocabulary()[0] if term_word))
        if not (not tag_names or not isinstance(read_reply, dict)):
            moved = {}
            for need_line in read_reply.get('needs') or []:
                target = self._refiled_need(need_line, text_lines, tag_names)
                if None is not target:
                    moved[need_line] = target
            if moved:
                swap = lambda ent: moved.get(ent, ent)
                read_reply['needs'] = list((swap(ent) for ent in read_reply.get('needs') or []))
                now = read_reply.get('requirements_now')
                if isinstance(now, list):
                    read_reply['requirements_now'] = list((swap(ent) if isinstance(ent, str) else ent for ent in now))
                self.trace.append({'kind': 'attributed', 'moved': moved})
            else:
                return
        else:
            return

    def _guard_stated(self, read_reply, _lines=None):
        if isinstance(read_reply, dict):
            given_reqs = list((str(ent) for ent in self.wishes.stated))
            arg_specs = list((self._need_property_terms(ent) for ent in given_reqs))

            def touches(snip):
                if not any((pager in snip or snip in pager for pager in given_reqs if pager)):
                    label_name = self._need_property_terms(snip)
                    held_204 = bool(label_name) and label_name in arg_specs
                    return held_204
                else:
                    return True
            guarded = list((str(dd) for dd in read_reply.get('dropped') or [] if touches(str(dd))))
            if guarded:
                read_reply['dropped'] = list((dd for dd in read_reply.get('dropped') or [] if str(dd) not in guarded))
                self.trace.append({'kind': 'stated_guarded', 'entries': guarded})
            free = read_reply.get('no_preference')
            if isinstance(free, list):
                overlaps = lambda term_set: any((prop_terms and (term_set >= prop_terms or prop_terms >= term_set) for prop_terms in arg_specs))
                read_reply['no_preference'] = list((fobj for fobj in free if not (_terms(fobj) and overlaps(_terms(fobj)))))
            self._guard_revealed(read_reply)
        else:
            return

    def _guard_revealed(self, read_reply):
        need_state = self.wishes
        fresh = list((self._need_property_terms(ent) for ent in read_reply.get('needs') or [] if isinstance(ent, str)))
        held = list((ent for ent in need_state.revealed if ent not in need_state.dropped and self._need_property_terms(ent) and (self._need_property_terms(ent) not in fresh)))
        if held:
            homes = list((self._need_property_terms(ent) for ent in held))
            near = lambda term_set: bool(term_set) and any((home >= term_set or term_set >= home for home in homes))
            lost = list((str(dd) for dd in read_reply.get('dropped') or [] if str(dd) in held or near(self._need_property_terms(dd))))
            if lost:
                read_reply['dropped'] = list((dd for dd in read_reply.get('dropped') or [] if str(dd) not in lost))
            free = read_reply.get('no_preference')
            if isinstance(free, list):
                read_reply['no_preference'] = list((fobj for fobj in free if not near(_terms(fobj))))
            relist = self._rebuilt_list(read_reply)
            if None is not relist:
                have = list((self._need_property_terms(ent) for ent in relist if isinstance(ent, str)))
                relist.extend((ent for ent in held if self._need_property_terms(ent) not in have))
            if lost:
                self.trace.append({'kind': 'revealed_guarded', 'entries': lost})
        else:
            return

    @staticmethod
    def _rebuilt_list(read_reply):
        if isinstance(read_reply, dict):
            relist = read_reply.get('requirements_now')
        else:
            relist = None
        if isinstance(relist, list):
            return relist
        return None

    @staticmethod
    def _line_terms(text_lines):
        spoken = set()
        for ln in text_lines or []:
            spoken |= _terms(ln)
        return spoken

    def _restore_stated(self, read_reply, text_lines):
        relist = self._rebuilt_list(read_reply)
        if not None is relist:
            spoken = self._line_terms(text_lines)
            gone = list((str(dd) for dd in read_reply.get('dropped') or [])) + list(self.wishes.dropped)
            lifted = lambda need_line: any((need_line in dd or dd in need_line for dd in gone if dd))
            for need_line in self.wishes.stated:
                pname = self._need_property_terms(need_line)
                if not (not pname or need_line in relist or lifted(need_line) or _covers(pname, spoken)):
                    relist[:] = list((need_entry for need_entry in relist if not isinstance(need_entry, str) or self._need_property_terms(need_entry) != pname)) + [need_line]
                    self.trace.append({'kind': 'kept_stated', 'entry': need_line})
                else:
                    continue
        else:
            return

    def _drop_unsaid(self, read_reply, text_lines):
        relist = self._rebuilt_list(read_reply)
        if not None is relist:
            need_state = self.wishes
            known = need_state.stated + need_state.revealed + list(need_state.newest_lines or []) + list(read_reply.get('needs') or [])
            arg_specs = list((prop_terms for prop_terms in map(self._need_property_terms, known) if prop_terms))
            spoken = self._line_terms(text_lines)

            def invented(need_line):
                if isinstance(need_line, str) and ':' in need_line:
                    pname = self._need_property_terms(need_line)
                    if not (not pname or any((pname & prop_terms for prop_terms in arg_specs))):
                        held_205 = not any((_word_like(term_word, pager) for term_word in _terms(need_line) for pager in spoken))
                        return held_205
                    else:
                        return False
                else:
                    return False
            made_up = list((need_line for need_line in relist if invented(need_line)))
            if made_up:
                relist[:] = list((need_entry for need_entry in relist if need_entry not in made_up))
                self.trace.append({'kind': 'unsaid_dropped', 'entries': made_up})
        else:
            return

class StoreWording:

    @staticmethod
    def _replace_need(need_state, old_matches, new):
        for bucket in (need_state.stated, need_state.revealed, need_state.newest_lines or []):
            bucket[:] = list((new if isinstance(need_entry, str) and old_matches(need_entry) else need_entry for need_entry in bucket))

    def _excluded_by_property(self):
        barred = {}
        for need_line in self.wishes.stated:
            pname, sep, wanted = self._split_need(need_line)
            if sep:
                if '!=' in wanted:
                    held_206 = _terms(pname)
                    slot_name = frozenset(held_206)
                    barred[slot_name] = barred.get(slot_name, set()) | _terms(wanted.partition('!=')[2])
        return barred

    def _anchor_stated(self):
        if not (not self.records or not isinstance(self.policy, dict)):
            opened = list((listing_row for listing_row in self.listing_table().values() if listing_row['opened']))
            if opened:
                self._rehome_stated(opened)
                held_207 = self.policy.get('query')
                request = _terms(held_207)
                barred = self._excluded_by_property()
                need_state = self.wishes
                for need_line in list(need_state.stated):
                    pname, sep, wanted = self._split_need(need_line)
                    key_words = _terms(pname)
                    held_208 = sep
                    if held_208:
                        held_208 = key_words
                    if held_208:
                        held_208 = '(' not in wanted
                    if held_208:
                        held_208 = _relation_of(wanted) is None
                    plain = held_208
                    if plain:
                        positive, _, excluded = wanted.partition('!=')
                        held_210 = frozenset(key_words)
                        held_211 = set()
                        held_209 = barred.get(held_210, held_211)
                        pick = self._store_value_for(opened, key_words, positive, excluded, request, held_209)
                        if not None is pick:
                            held_212 = pname.strip()
                            held_213 = wanted.strip()
                            anchored = '{}: {} ({})'.format(held_212, pick, held_213)
                            self._replace_need(need_state, lambda need_entry, old=need_line: need_entry.startswith(old), anchored)
                            self.needs_revision = self.needs_revision + 1
                            self.trace.append({'kind': 'anchored', 'from': need_line, 'to': anchored})
                        else:
                            continue
                    else:
                        continue
            else:
                return
        else:
            return

    def _store_value_for(self, opened, need_terms, positive, excluded, request, barred):
        member_list = set()
        for datum in self._values_under(opened, need_terms):
            member_list.update((prop_terms.strip().lower() for prop_terms in re.split('[,/;|]', str(datum)) if _terms(prop_terms)))
        want_words = _terms(positive)
        if not (not member_list or any((_terms(mt) <= want_words for mt in member_list))):
            off = _terms(excluded) | barred
            reworded = bool(excluded.strip() and want_words)
            picks = []
            for member in member_list:
                term_set = _terms(member)
                if term_set & off or not all((any((_word_like(term_word, q) for q in request)) for term_word in term_set)):
                    continue
                elif want_words and _covers(want_words, term_set):
                    continue
                elif reworded or any((_word_like(term_word, prop_terms) for term_word in term_set for prop_terms in want_words)):
                    picks.append(member)
            if 1 == len(picks):
                held_214 = picks[0]
                return held_214
            return None
        else:
            return None

    def _rehome_stated(self, opened):
        need_state = self.wishes
        sentences = list((_terms(pager) for pager in re.split('(?<=[.!?])\\s+', str(self.policy.get('query') or ''))))
        for need_line in list(need_state.stated):
            pname, sep, wanted = self._split_need(need_line)
            key_words = _terms(pname)
            datum = _terms(wanted.partition('!=')[0].partition('(')[0])
            if not (not (sep and key_words and datum) or _relation_of(wanted) is not None):
                homes = list((term_set for term_set in sentences if _covers(datum, term_set)))
                if not (not homes or any((_covers(key_words, term_set) for term_set in homes))):
                    found_keys = self._keys_holding_value(opened, key_words, datum, homes)
                    if found_keys:
                        shortest = min(found_keys, key=len)
                        if not all((other >= shortest for other in found_keys)):
                            continue
                        moved = '{}: {}'.format(found_keys[shortest], wanted.strip())
                        self._replace_need(need_state, lambda need_entry, old=need_line: old == need_entry, moved)
                        self.needs_revision = self.needs_revision + 1
                        self.trace.append({'kind': 'rekeyed', 'from': need_line, 'to': moved})
                    else:
                        continue
                else:
                    continue
            else:
                continue

    def _keys_holding_value(self, opened, need_terms, datum, homes):
        found_keys = {}
        for listing_row in opened:
            for field_name, snip in listing_row['facts'].items():
                key_words = _terms(field_name)
                if not key_words or key_words & need_terms:
                    continue
                elif not any((_covers(key_words, term_set) for term_set in homes)):
                    continue
                elif not datum not in list((_terms(part) for part in re.split('[,/;|]', str(snip)))):
                    plain = ' '.join(self._key_as_words(field_name).split()).lower()
                    slot_name = frozenset(key_words)
                    if slot_name not in found_keys or len(found_keys[slot_name]) > len(plain):
                        found_keys[slot_name] = plain
                else:
                    continue
        return found_keys

class SheetMenu:

    def _sheet_vocabulary(self):
        tag_names = []
        values = set()
        for record_blob in _walk_dicts(self.records):
            texts = self._short_text_fields(record_blob, 70)
            if not 5 > len(texts):
                values.update((tk.strip().lower() for _, tk in texts))
                for term_set in (_terms(self._key_as_words(label_key)) for label_key, _ in texts):
                    if term_set:
                        if term_set not in tag_names:
                            tag_names.append(term_set)
            else:
                continue
        held_215 = (tag_names, values)
        return held_215

    @staticmethod
    def _dicts_naming(record_blob, unit_ref, nest_level):
        if not 6 < nest_level:
            if isinstance(record_blob, dict):
                if not any((isinstance(cell, str) and unit_ref == cell for cell in record_blob.values())):
                    held_216 = record_blob.values()
                    kids = list(held_216)
                else:
                    yield record_blob
                    return
            elif not isinstance(record_blob, list):
                return
            else:
                kids = record_blob
            for kid in kids:
                yield from Episode._dicts_naming(kid, unit_ref, nest_level + 1)
        else:
            return

    def records_about(self, unit_ref):
        found = []
        for record in self.records:
            for record_blob in self._dicts_naming(record, unit_ref, 0):
                snip = _json_line(_trim_strings(record_blob, 400))[:2500]
                if snip not in found:
                    found.append(snip)
                if 8 <= len(found):
                    held_217 = found[-4:]
                    return held_217
        held_218 = found[-4:]
        return held_218

    def _detail_sheets(self):
        held_219 = _walk_dicts(self.records)
        node_list = list(held_219)
        held_220 = self.seen_ids
        if not held_220:
            held_220 = set()
        ids = held_220
        dual = _listing_own_fields(node_list, ids)
        row_ids = _result_row_ids(self.records, ids)
        owned = set()
        above = {}
        for record_blob in node_list:
            above.update(((id(child), record_blob) for child in record_blob.values() if isinstance(child, dict)))
            if any((_is_listing_id(el, ids) for el in record_blob.values())):
                owned.update((id(child) for child in record_blob.values() if isinstance(child, dict)))
        sheets = []
        self._sheet_texts = []
        for record_blob in node_list:
            if not id(record_blob) in row_ids:
                owner = next((cell for cell in record_blob.values() if _is_listing_id(cell, ids)), None)
                texts = list(((label_key, cell) for label_key, cell in self._short_text_fields(record_blob, 70) if (owner, label_key, cell) not in dual))
                if not len(texts) < (3 if id(record_blob) in owned else 5):
                    fact_sheet = {}
                    for field_name, snip in texts:
                        tag = self._key_as_words(field_name).strip().lower()
                        if tag not in fact_sheet:
                            if not _is_code(snip, True):
                                fact_sheet[tag] = snip.lower()
                    sheets.append(fact_sheet)
                    around = [record_blob, above.get(id(record_blob)) or {}]
                    held_221 = set((str(cell).strip().lower() for blob in around for cell in blob.values() if isinstance(cell, str)))
                    self._sheet_texts.append(held_221)
                else:
                    continue
            else:
                continue
        return sheets

    def _sheets_of_wanted(self, sheets):
        held_222 = list((el['product'] for el in self.basket_items()))
        if not held_222:
            held_222 = [(self.plan or {}).get('product') or '']
        products = held_222
        labels = set((rw['label'].strip().lower() for rw in self.listing_table().values() if isinstance(rw.get('label'), str)))
        homes = set()
        for product in products:
            stems = self._stems(product)
            hit = set((lb for lb in labels if stems & self._stems(lb)))
            if hit:
                homes |= hit
            else:
                return sheets
        words = getattr(self, '_sheet_texts', None)
        if not isinstance(words, list) or len(sheets) != len(words):
            words = list((set(page.values()) for page in sheets))
        named = list((said & labels for said in words))
        keep = list((page for page, said in zip(sheets, named) if not said or said & homes))
        if not len(sheets) == len(keep):
            return keep
        else:
            return sheets

    @staticmethod
    def _spelling_groups(sheets, freq):
        rank = lambda label_key: (-freq[label_key], label_key)
        home = {}
        roots = set()
        for tag in sorted(freq, key=rank):

            def same(main):
                both = list((page for page in sheets if tag in page and main in page))
                held_223 = bool(both) and bool(_singular_terms(tag) & _singular_terms(main)) and all((page[main] == page[tag] for page in both))
                return held_223
            match = next((main for main in sorted(roots, key=rank) if same(main)), None)
            if None is match:
                roots.add(tag)
                match = tag
            home[tag] = match
        return home

    def property_menu_text(self):
        held_224 = self._detail_sheets()
        sheets = self._sheets_of_wanted(held_224)
        seen = {}
        for fact_sheet in sheets:
            for tag, low in fact_sheet.items():
                seen.setdefault(tag, []).append(low)
        freq = dict(((tag, len(vals)) for tag, vals in seen.items()))
        distinct = dict(((tag, list(dict.fromkeys(vals))) for tag, vals in seen.items()))
        home = self._spelling_groups(sheets, freq)
        aliases = {}
        for tag in home:
            if tag != home[tag]:
                aliases.setdefault(home[tag], []).append(tag)
        held_225 = (self.plan or {}).get('product')
        product = _terms(held_225)

        def item_named(label_key: str) -> bool:
            hits = sum((1 for cell in seen[label_key] if product & _terms(cell)))
            held_226 = bool(product) and len(seen[label_key]) < 2 * hits
            return held_226
        uniform = set((label_key for label_key in freq if 3 <= freq[label_key] and 1 == len(distinct[label_key])))
        top = sorted((label_key for label_key in freq if label_key == home[label_key] and (not item_named(label_key))), key=lambda label_key: (-freq[label_key], label_key in uniform, label_key))[:28]
        self.property_menu = dict(((label_key, {'count': freq[label_key], 'values': sorted(distinct[label_key]), 'uniform': label_key in uniform, 'also': sorted(aliases.get(label_key, []))}) for label_key in top))
        text_lines = []
        for label_key in top:
            if label_key in uniform:
                same = ' (same value on every listing)'
            else:
                same = ''
            if aliases.get(label_key):
                also = ' [also spelled: ' + ', '.join(sorted(aliases[label_key])) + ']'
            else:
                also = ''
            text_lines.append('* {}: {} listings{}{}; values seen '.format(label_key, freq[label_key], same, also) + ' / '.join(distinct[label_key][:5]))
        held_227 = ('\n'.join(text_lines), len(sheets))
        return held_227

    def _recurring_labels(self):
        per_key = {}
        for record_blob in (cnt for cnt in _walk_dicts(self.records) if 4 <= len(cnt)):
            for field_name, snip in self._short_text_fields(record_blob, 40):
                if not _is_code(snip, False):
                    held_228 = str(field_name)
                    bucket = per_key.setdefault(held_228, {})
                    bucket[snip] = 1 + bucket.get(snip, 0)
                else:
                    continue

        def worth(counts):
            held_229 = 2 <= len(counts) <= 8 and max(counts.values()) >= 2 and (sum(counts.values()) >= 4)
            return held_229
        shown = list(('* {}: '.format(field_name) + ', '.join(('{} ({}x)'.format(cell, cnt) for cell, cnt in sorted(ch.items(), key=lambda pair: -pair[1]))) for field_name, ch in per_key.items() if worth(ch)))
        held_230 = '\n'.join(shown[:12])
        return held_230

class Reviews:

    def second_check(self, unit_ref, intent):
        if not (self.seconds_left() < WRAP_UP_SECONDS + 20 or self.over_soft_spend()):
            held_231 = self._state_tag()
            memo = '{}|{}'.format(unit_ref, held_231)
            cached = self.verdicts.get(memo)
            if None is cached:
                evidence = self.records_about(unit_ref)
                if evidence:
                    cached = self.verdicts[memo] = self._second_check_call(unit_ref, intent, evidence)
            held_232 = cached or ''
            return held_232
        else:
            return ''

    def _second_check_call(self, unit_ref, intent, evidence):
        in_force = self.wants_text()
        if None is self.last_cart_view:
            cart = None
        else:
            cart = _json_line(self.last_cart_view)[:3000]
        silent = self._silent_conditions()
        recurring = self._recurring_labels()
        blocks = [VERIFY_PROMPT, 'The request word for word:\n' + str(self.policy.get('query') or ''), 'The requirements now, from the request plus later answers and changes:\n' + in_force, 'The listing to judge ({}): {}'.format(intent, unit_ref) + self._product_scope(unit_ref), 'All store records that mention this listing, oldest first:\n' + '\n'.join(evidence)]
        if None is not cart:
            blocks.append('The newest view of the cart:\n' + cart)
        if silent:
            blocks.append('Conditions that no opened listing here mentions (no key and no text covers them, so no listing can prove them): ' + '; '.join(silent) + '. Do not fail the listing for these alone; judge it on the other requirements.')
        if recurring:
            blocks.append("Labels repeated across several listings, with counts. Judge this listing by the label equal to the request's whole wording, never a neighbouring one:\n" + recurring)
        held_233 = '\n\n'.join(blocks)
        result_info = self.ask_json(held_233, 1200)
        self.trace.append({'kind': 'verify', 'item': unit_ref, 'result': result_info})
        if result_info and result_info.get('fits') is False:
            gaps = list((str(prop_terms) for prop_terms in result_info.get('problems') or []))[:5]
            held_234 = self.listing_table().get(unit_ref)
            if not held_234:
                held_234 = {}
            own = held_234
            every = list((str(prop_terms) for prop_terms in result_info.get('problems') or []))
            if not (own.get('opened') and self._passes_by_words(own) is True and _only_qualified_objections(own['facts'], self.need_lines(), every)):
                held_235 = 'The second check of {} found problems: '.format(unit_ref) + '; '.join(gaps) + '. Pick another, fully verified listing instead, unless you are sure this check is mistaken.'
                return held_235
            else:
                self.trace.append({'kind': 'verify_kept', 'item': unit_ref})
                return ''
        else:
            return ''

    def _decline_audit(self):
        if not (self.audit_done or self.over_soft_spend() or self.seconds_left() < WRAP_UP_SECONDS + 30):
            self.audit_done = True
            excerpts = _latest_records(self.records, 24000)
            if excerpts:
                request = str(self.policy.get('query') or '')
                held_236 = '\n\n'.join([AUDIT_PROMPT, 'THE REQUIREMENTS THAT APPLY:\n' + self.wants_text(), 'THE REQUEST WORD FOR WORD:\n' + request, 'WHAT HAS BEEN SEEN (NEWEST FIRST):\n' + '\n'.join(excerpts)])
                verdict = self.ask_json(held_236, 1200)
                self.trace.append({'kind': 'audit', 'result': verdict})
                held_237 = (verdict or {}).get('possible_fits')
                if not held_237:
                    held_237 = []
                named = held_237
                picks = list((str(el) for el in named if isinstance(el, (str, int))))[:5]
                if picks:
                    because = verdict.get('reason')
                    held_238 = 'Wait before declining: looking over all records again, these listings may meet every condition' + (' (%s)' % because if because else '') + ': ' + ', '.join(picks) + '. Open each for its data sheet, stock and price, and buy one that qualifies; a decline is right only once all of them are ruled out.'
                    return held_238
                else:
                    return ''
            else:
                return ''
        else:
            return ''

    def _evidence_against_decline(self):
        row_index = self.listing_table()
        affordable = dict(((idx, rw) for idx, rw in row_index.items() if rw['opened'] and self._affordable(rw)))
        marks = {}
        for idx, rw in affordable.items():
            foreign = self._needs_of_other_products(idx)
            marks[idx] = list(((ch, fv) for ch, fv in self._verdicts_for(rw) if ch not in foreign))
        sure = list((idx for idx in affordable if all((True is fv for _, fv in marks[idx]))))
        good = self._confirmed_fits(sure)
        near_notes = self._relook_close_fits(row_index, marks, sure, good)
        held_239 = dict.fromkeys((ch for got in marks.values() for ch, _ in got))
        text_lines = list(held_239)
        unopened = self._unopened_worth_reading(row_index, affordable, text_lines)
        self.proven_fits = list(good)
        if good or near_notes or unopened:
            parts = []
            if good:
                parts.append('by what they state themselves, every requirement is met by ' + ', '.join(good) + ' - a decline earns nothing while any of them can be bought, and a valid purchase counts, so buy the strongest')
            if near_notes:
                parts.append('these opened listings break no requirement, but one point on each is still unconfirmed: ' + '; '.join(near_notes) + ' - settle it for each by opening its data sheet (title and variant too) and buy one where it holds')
            if unopened:
                parts.append('never opened but priced within the budget: ' + ', '.join(unopened) + ' - open these first' + self._view_tool_hint())
            if text_lines:
                if not good:
                    per_line = []
                    for ch in text_lines:
                        backers = list((idx for idx in affordable if (ch, True) in marks[idx]))[:3]
                        per_line.append("'{}': ".format(ch) + (', '.join(backers) or 'confirmed nowhere'))
                    parts.append('confirmed so far, requirement by requirement: ' + '; '.join(per_line))
            held_240 = 'Wait before declining: ' + '; '.join(parts) + '.'
            return held_240
        else:
            return ''

    def _confirmed_fits(self, sure):
        queue = list((idx for idx in self.cart_ids if idx in sure))
        queue += list((idx for idx in sure if idx not in queue))
        good = []
        for candidate in queue:
            if 3 <= len(good):
                break
            elif candidate not in good:
                if not self.second_check(candidate, 'meets everything, checked before a decline'):
                    good.append(candidate)
        return good

    def _relook_close_fits(self, row_index, marks, sure, good):

        def still_open(idx):
            held_241 = list((ch for ch, fv in marks[idx] if None is fv))
            return held_241

        def price_of(idx):
            known = row_index[idx]['price']
            if None is known:
                return 1e+18
            return known
        close_fits = list((idx for idx, got in marks.items() if idx not in sure and got and all((False is not fv for _, fv in got)) and (len(still_open(idx)) <= max(1, len(got) // 2))))
        close_fits.sort(key=lambda idx: (len(still_open(idx)), price_of(idx), idx))
        notes = []
        relooks_left = 2
        for candidate in close_fits:
            if not 3 <= len(good):
                if relooks_left:
                    if candidate not in self.relooked:
                        self.relooked.add(candidate)
                        relooks_left = relooks_left - 1
                        if not self.second_check(candidate, 'nearly a match; checked again before a decline'):
                            good.append(candidate)
                            continue
                pending = ' and '.join(("'%s'" % ch for ch in still_open(candidate)))
                held_242 = '{} (not yet confirmed: {})'.format(candidate, pending)
                notes.append(held_242)
            else:
                break
        return notes

    def _view_tool_hint(self):
        offered = set((tk['function']['name'] for tk in self.tools_offered()))
        details, view = (self.roles.get('details'), self.roles.get('view'))
        if not (not view or view not in offered or (not details) or (details in offered)):
            held_243 = ' (the details tool is no longer offered; {} is still offered and shows their data too)'.format(view)
            return held_243
        else:
            return ''

    def _unopened_worth_reading(self, row_index, affordable, text_lines):
        spare = max(0, self.steps_left() - self._reserve() - 1)
        held_244 = min(spare, 4)
        aimed = _rows_for_unfitted(self, held_244)
        if not aimed:
            sections = set((rw['label'] for rw in affordable.values() if rw['label']))
            wanted = set()
            for ch in text_lines:
                held_245 = _terms(str(ch).partition(':')[2].split('!=')[0])
                wanted.update(held_245)
            ranked = sorted(((-len(wanted & _value_terms(rw.get('facts') or {})), rw['price'], idx) for idx, rw in row_index.items() if not rw['opened'] and None is not rw['price'] and self._affordable(rw) and (not sections or rw['label'] in sections)))
            held_246 = list((element[2] for element in ranked[:min(spare, 4)]))
            return held_246
        else:
            return aimed

class ToolOffer:

    def _jobs_still_open(self):
        remaining = self.steps_left()
        if 1 >= remaining:
            held_247 = ('order', 'decline')
            return held_247
        elif 3 >= remaining:
            held_248 = ('order', 'decline', 'cart')
            return held_248
        elif not (7 < remaining or (self.order_attempts and (not self.refusals))):
            late = ('order', 'decline', 'cart', 'cart_view', 'view')
            if self.refusals:
                if _cheaper_after_refusal(self):
                    held_249 = late + ('details',)
                    return held_249
                return late
            if remaining >= self._reserve() + 2:
                if _rows_for_unfitted(self, 1):
                    held_250 = late + ('details',)
                    return held_250
            return late
        else:
            return None

    def tools_offered(self):
        jobs = self._jobs_still_open()
        tool_of = self.roles.get
        offered = self.tools
        if None is not jobs:
            if tool_of('order'):
                if tool_of('cart'):
                    held_251 = map(tool_of, jobs)
                    names = set(held_251)
                    held_252 = list((tk for tk in offered if tk['function']['name'] in names))
                    if not held_252:
                        held_252 = self.tools
                    offered = held_252
        if self.planner_on:
            muted = tool_of('message')
        else:
            muted = None
        if muted:
            held_253 = list((tk for tk in offered if muted != tk['function']['name']))
            return held_253
        return offered

    def _call_signature(self, label_name, arguments):
        held_254 = '|'.join((label_name, ','.join(sorted(arguments)), self._mode_signature(label_name, arguments)))
        return held_254

    def _error_in(self, label_name, seen):
        nothing = ('', False)
        if not isinstance(seen, dict) or not seen:
            return nothing
        elif not ('error' in seen and 'refusal' not in seen):
            held_true = self.__dict__.setdefault('_true_flags_seen', {}).setdefault(label_name, set())
            blobs = list((blob for blob in [seen] + list((cell for cell in seen.values() if isinstance(cell, dict))) if 3 >= len(blob) and (not any((isinstance(cell, list) or cell in self.seen_ids for cell in blob.values() if not isinstance(cell, dict))))))
            flipped = list((key for blob in blobs for key, cell in blob.items() if False is cell and key in held_true))
            if not (flipped and 'refusal' not in seen and (label_name != self.roles.get('message'))):
                held_true.update((key for blob in blobs for key, cell in blob.items() if True is cell))
                if 1 < len(seen) or label_name == self.roles.get('message'):
                    return nothing
                held_258 = seen.items()
                held_257 = iter(held_258)
                field_name, datum = next(held_257)
                held_259 = isinstance(datum, str)
                if held_259:
                    held_259 = datum.strip()
                if held_259:
                    held_259 = datum not in self.seen_ids
                lone_text = held_259
                if lone_text:
                    held_260 = ('{}: {}'.format(field_name, datum)[:200], False)
                    return held_260
                return nothing
            else:
                held_256 = (_json_line(dict(list(seen.items())[:2]))[:200], True)
                return held_256
        else:
            held_255 = (_json_line(dict(list(seen.items())[:2]))[:200], True)
            return held_255

    @staticmethod
    def _has_list(record_blob):
        if isinstance(record_blob, list):
            held_261 = bool(record_blob)
            return held_261
        elif not isinstance(record_blob, dict):
            return False
        else:
            held_262 = any((Episode._has_list(cell) for cell in record_blob.values()))
            return held_262

    def _empty_search_note(self, label_name, arguments, nested):
        ids = set()
        _collect_ids(nested, ids)
        if not (ids or self._has_list(nested)):
            arg_specs = self._arg_specs(label_name)
            for arg, given in arguments.items():
                if isinstance(given, str):
                    if given.strip():
                        if not _is_selector(arg_specs.get(arg)):
                            return ''
            choices = dict(((label_key, pager['enum']) for label_key, pager in arg_specs.items() if label_key not in arguments and isinstance(pager, dict) and pager.get('enum')))
            if choices:
                unset = list(('%s (e.g. %s)' % (label_key, ', '.join(map(str, vals[:4]))) for label_key, vals in choices.items()))
                held_263 = 'Nothing found. Options this call left out: ' + '; '.join(unset[:3]) + '. Some search modes find nothing until one of them is set, so an empty result here does not mean the store lacks the item.'
                return held_263
            else:
                return ''
        else:
            return ''

    def _units_sent(self, label_name, arguments):
        arg_specs = self._arg_specs(label_name, True)
        counter = next((label_key for label_key, pager in arg_specs.items() if isinstance(pager, dict) and pager.get('type') == 'integer' and ('enum' not in pager)), None)
        if not None is counter:
            if counter in arguments:
                units = arguments[counter]
            else:
                units = arg_specs[counter].get('default', 1)
            if isinstance(units, int) and (not isinstance(units, bool)):
                return units
            return 1
        else:
            return 1

    def _read_cart_units(self, seen):
        self.cart_seen_at = self.step
        text_lines = []
        for record_blob in _walk_dicts([seen]):
            held = next((cell for cell in record_blob.values() if isinstance(cell, str) and cell in self.sent_units), None)
            if None is not held:
                text_lines.append((held, self._int_fields(record_blob)))
        counts = {}
        for held, ints in text_lines:
            if None is self.quantity_field:
                matches = list((label_key for label_key in ints if self.sent_units[held] == ints[label_key]))
                if 1 == len(matches):
                    self.quantity_field = matches[0]
            if self.quantity_field in ints:
                counts[held] = ints[self.quantity_field]
        if None is not self.quantity_field:
            self.units_in_cart = counts

    def _mode_signature(self, label_name, arguments):
        arg_specs = self._arg_specs(label_name, True)
        chosen = {}
        for arg in arguments:
            tool_desc = arg_specs.get(arg)
            if isinstance(tool_desc, dict):
                if 'enum' in tool_desc:
                    chosen[arg] = arguments[arg]
        held_264 = _json_line(chosen)
        return held_264

    def _reserve(self):
        held_265 = _spare_steps(self.max_steps)
        return held_265

    def _room_for(self, cost):
        held_266 = self.steps_left() - cost >= self._reserve()
        return held_266

    def _state_tag(self):
        held_267 = '%s.%s' % (self.needs_revision, self.cart_generation)
        return held_267

    def _new_cart_state(self):
        self.cart_generation = self.cart_generation + 1

class Messages:

    def _message_cap(self):
        limit = self.message_budget
        if None is limit:
            return 3
        return limit

    def _message_args(self, text):
        held_268 = self.param_schemas.get(self.roles.get('message') or '')
        if not held_268:
            held_268 = {}
        tool_spec = held_268
        must = list(tool_spec.get('required') or [])
        writable = _free_text_args(tool_spec.get('properties') or {})
        slot_name = next((label_name for label_name in must if label_name in writable), None)
        if not slot_name:
            if 1 == len(writable):
                slot_name = writable[0]
        if not (slot_name and (not set(must) - {slot_name})):
            return None
        else:
            held_269 = {slot_name: text}
            return held_269

    def _message_call(self, text, call_id):
        payload = self._message_args(text)
        if payload:
            sender = self.roles.get('message')
            payload, problems = _fit_schema(payload, self.param_schemas[sender], 'args')
            if not problems:
                body = {'name': sender, 'arguments': json.dumps(payload)}
                held_270 = {'id': call_id, 'type': 'function', 'function': body}
                return held_270
            else:
                return None
        else:
            return None

    def _post_replies(self, replies):
        last = len(replies) - 1
        for cnt, (call_id, said) in enumerate(replies):
            if last == cnt:
                shown = said + '\n' + self.status_line()
            else:
                shown = said
            self.messages.append({'role': 'tool', 'tool_call_id': call_id, 'content': shown})
        held_271 = list((said for _, said in replies))
        return held_271

    def _ask_round(self, force=False):
        held_272 = _run_ask(self, {'force': force})['exit']
        return held_272

    def _send_reply(self, text):
        if self._reply_allowed():
            call = self._message_call(text, 'reply_{}'.format(self.step))
        else:
            call = None
        if not None is call:
            self.messages.append({'role': 'assistant', 'tool_calls': [call], 'content': "Answering the shopper's question."})
            dispatched = self.run_batch([call], planned=True)
            self._post_replies(dispatched)
            return True
        else:
            return False

    def _reply_allowed(self):
        if not self.roles.get('message'):
            return False
        elif not self.messages_used >= self._message_cap():
            spare = self.steps_left() - _spare_steps(self.max_steps)
            held_273 = 0 < spare and self.seconds_left() >= WRAP_UP_SECONDS
            return held_273
        else:
            return False

    def _units_mismatch(self, label_name, arguments):
        units = self.units_wanted()
        if not (None is not self.add_signature and self.add_signature != self._mode_signature(label_name, arguments)):
            if self.basket_items():
                named = list((fv for fv in arguments.values() if isinstance(fv, str) and fv in self.seen_ids))
                if 1 == len(named):
                    units = self._units_for(named[0])
                else:
                    units = None
                if None is units:
                    return ''
            counters = list(((kname, pager) for kname, pager in (self.param_schemas[label_name].get('properties') or {}).items() if isinstance(pager, dict) and pager.get('type') == 'integer' and ('enum' not in pager)))
            for field, tool_desc in counters:
                if field in arguments:
                    dispatched = arguments[field]
                else:
                    dispatched = tool_desc.get('default', 1)
                if isinstance(dispatched, int):
                    if units != dispatched:
                        held_274 = 'wrong number of units: {} in this call, but {} of one listing are what the request wants.'.format(dispatched, units)
                        return held_274
            return ''
        else:
            return ''

class BasketLines:

    def units_wanted(self):
        count = self.plan.get('quantity')
        if isinstance(count, bool) or not isinstance(count, int) or 1 > count:
            return 1
        elif not (1 == count or 'quantity_quote' not in self.plan):
            if self._count_in_request(self.plan.get('quantity_quote')):
                return count
            return 1
        else:
            return count

    def basket_items(self):
        if isinstance(self.plan, dict):
            plan = self.plan
        else:
            plan = {}
        raw = plan.get('lines')
        items = []
        for element in raw if isinstance(raw, list) else ():
            if not (not isinstance(element, dict) or not str(element.get('product') or '').strip()):
                held_275 = element.get('requirement')
                needs = element.get('requirements', held_275)
                if isinstance(needs, str):
                    needs = [needs]
                else:
                    needs = needs if isinstance(needs, list) else []
                count = element.get('quantity')
                held_276 = isinstance(count, int)
                if held_276:
                    held_276 = not isinstance(count, bool)
                if held_276:
                    held_276 = 1 <= count
                ok = held_276
                items.append({'product': str(element['product']).strip(), 'quantity': count if ok else 1, 'requirements': list((str(fv).strip() for fv in needs if isinstance(fv, (str, int, float)) and str(fv).strip()))})
            else:
                continue
        if 2 <= len(items):
            return items
        held_277 = []
        return held_277

    @staticmethod
    def _stems(text):
        held_278 = set((tk.rstrip('s')[:6] for tk in re.findall('[a-z0-9]+', str(text).lower()) if 2 < len(tk)))
        return held_278

    def _product_index(self, unit_ref):
        items = self.basket_items()
        if not (not items or None is unit_ref):
            have = set()
            for branch in _walk_dicts(self.records):
                if unit_ref in branch.values():
                    for fv in branch.values():
                        if isinstance(fv, str):
                            if unit_ref != fv:
                                have |= self._stems(fv)
            scores = []
            for element in items:
                words = self._stems(element['product'])
                values = set().union(*(list((self._stems(rw.partition(':')[2]) for rw in element['requirements'])) or [set()]))
                scores.append(2 * len(words & have) + len(values & have))
            top = max(scores)
            if 0 < top and scores.count(top) == 1:
                held_279 = scores.index(top)
                return held_279
            return None
        else:
            return None

    def _units_for(self, unit_ref):
        index = self._product_index(unit_ref)
        if None is index:
            return None
        held_280 = self.basket_items()[index]['quantity']
        return held_280

    def _needs_of_other_products(self, unit_ref):
        items, index = (self.basket_items(), self._product_index(unit_ref))
        if not None is index:
            own = set(items[index]['requirements'])
            foreign = set((rw for k, el in enumerate(items) if index != k for rw in el['requirements'] if rw not in own))
            mine = list((self._need_property_terms(rw) for rw in self._filed_needs(items[index]['product'])))
            away = list((rw for k, el in enumerate(items) if index != k for rw in self._filed_needs(el['product'])))
            keys = list((self._need_property_terms(rw) for rw in away if self._need_property_terms(rw) not in mine))
            held_282 = foreign | set(away) | set((rw for rw in self.need_lines() if self._need_property_terms(rw) in keys))
            return held_282
        else:
            held_281 = set()
            return held_281

    def _filed_needs(self, product):
        state = self.wishes
        held_283 = list((rw for rw, home in self.need_homes.items() if product == home and rw not in state.dropped))
        return held_283

    def _product_needs_text(self, element):
        homed = list((rw + ' (told by the shopper)' for rw in self._filed_needs(element['product']) if rw not in element['requirements']))
        held_284 = '; '.join(element['requirements'] + homed) or 'as in the request'
        return held_284

    def _basket_lines(self):
        many = self.basket_items()
        if many:
            out = ['- {} different products are wanted; the cart needs a line for each, at its own quantity:'.format(len(many))]
            for k, element in enumerate(many, 1):
                needs = self._product_needs_text(element)
                held_286 = '  product {}: {}, quantity {}, requirements: {}'.format(k, element['product'], element['quantity'], needs)
                out.append(held_286)
            return out
        else:
            held_285 = []
            return held_285

    def _product_scope(self, unit_ref):
        many = self.basket_items()
        if many:
            index = self._product_index(unit_ref)
            if not None is index:
                element = many[index]
                held_287 = ' (wanted product %d: %s, quantity %d; its own requirements alone count here: %s)' % (index + 1, element['product'], element['quantity'], self._product_needs_text(element))
                return held_287
            else:
                return ' (several products are wanted; judge a listing only by the requirements of its own product, never by those of the others)'
        else:
            return ''

    def _uncarted_products(self):
        items, held = (self.basket_items(), list(self.cart_ids))
        if None is not self.last_cart_view:
            if self.cart_changed_at <= self.cart_seen_at:
                held += list((idx for idx in _cart_view_ids(self, self.last_cart_view) if idx not in held))
        if not (not items or not held):
            covered = list((self._product_index(idx) for idx in held))
            missing = list((el for k, el in enumerate(items) if k not in covered))
            if not (len(items) <= len(held) and None in covered):
                held_290 = (missing, len(held))
                return held_290
            else:
                held_289 = ([], len(held))
                return held_289
        else:
            held_288 = ([], len(held))
            return held_288

    def _count_in_request(self, quote):
        if not isinstance(quote, str) or not quote.strip():
            return False
        elif quote.strip().lower() not in str(self.policy.get('query') or '').lower():
            return False
        elif not re.search('\\d', quote):
            held_291 = bool(_terms(self.plan.get('product')) & _terms(quote))
            return held_291
        else:
            return True

class ListingTable:

    def listing_table(self):
        found = _listing_records(self.records)
        slots = _field_slots(found)
        book = {}
        for ident, bag in found:
            element = book.get(ident)
            if None is element:
                element = book[ident] = dict(price=None, stock=None, label=None, facts={}, opened=False)
            for slot_name, key, accepts in slots:
                if accepts(bag.get(key)):
                    element[slot_name] = bag[key]
            fact_sheet = _facts_of(bag, ident)
            if len(list((fv for fv in fact_sheet.values() if isinstance(fv, str)))) >= 5:
                element['facts'].update(fact_sheet)
                element['opened'] = True
        return book

    def need_lines(self):
        self._anchor_stated()
        w = self.wishes
        if None is not w.newest_lines:
            pool = list(w.newest_lines)
        else:
            pool = list((rw for rw in w.stated + w.revealed if rw not in w.dropped))
        if None is not w.newest_lines:
            pool = list((text_line for text_line in pool if not self._just_limits(text_line)))
        held_292 = list((text_line for text_line in pool if all((half.strip() for half in str(text_line).partition(':')[::2]))))
        return held_292

    def _just_limits(self, text_line):
        w = self.wishes
        prop_name, _, constraint = str(text_line).partition(':')
        figures = set((float(fv) for fv in re.findall('\\d+(?:\\.\\d+)?', constraint)))
        limits = set((float(fv) for fv in [_number_or_none(w.budget), self.units_wanted()] + list((el['quantity'] for el in self.basket_items())) if isinstance(fv, (int, float))))
        if not (not figures or not limits >= figures or text_line in w.stated + w.revealed):
            own = _terms(prop_name)
            held_293 = not any((own and _terms(str(rw).partition(':')[0]) == own for rw in w.stated + w.revealed))
            return held_293
        else:
            return False

    def _affordable(self, listing_row):
        if not listing_row.get('stock') is False:
            cap, cost = (_number_or_none(self.wishes.budget), listing_row.get('price'))
            many = self.basket_items()
            if many:
                units = min((el['quantity'] for el in many))
            else:
                units = self.units_wanted()
            if None is cap or None is cost:
                return True
            held_294 = cap >= cost * units
            return held_294
        else:
            return False

    def _silent_conditions(self, row_map=None):
        if None is row_map:
            row_index = self.listing_table()
        else:
            row_index = row_map
        shown = list((rw['facts'] for rw in row_index.values() if rw['opened']))
        if not 3 > len(shown):
            named = list((_terms(kname) for fact_sheet in shown for kname in fact_sheet))
            voiced = list((_value_terms(fact_sheet) for fact_sheet in shown))
            quiet = []
            for text_line in self.need_lines():
                parsed = _need_terms(text_line)
                if not (None is parsed or not parsed[0]):
                    prop_terms, vw = parsed
                    if not (any((kname >= prop_terms for kname in named)) or (vw and any((tk >= vw for tk in voiced)))):
                        quiet.append(text_line)
                    else:
                        continue
                else:
                    continue
            return quiet
        else:
            held_295 = []
            return held_295

    def _verdicts_for(self, listing_row):
        held_296 = self._silent_conditions()
        quiet = set(held_296)
        held_297 = listing_row.get('facts')
        if not held_297:
            held_297 = {}
        fact_sheet = held_297
        pairs = []
        for text_line in self.need_lines():
            if text_line not in quiet:
                pairs.append((text_line, _word_verdict(fact_sheet, text_line, self._store_phrases(text_line))))
        return pairs

    def _passes_by_words(self, listing_row):
        held_298 = listing_row.get('facts')
        if not held_298:
            held_298 = {}
        fact_sheet = held_298
        outcomes = list((True if None is fv and _blurb_states(fact_sheet, text_line) else fv for text_line, fv in self._verdicts_for(listing_row)))
        for decisive in (False, None):
            if any((decisive is fv for fv in outcomes)):
                return decisive
        return True

    def _unmet_for(self, unit_ref, row_map):
        held_299 = row_map.get(unit_ref)
        if not held_299:
            held_299 = {}
        me = held_299
        held_300 = me.get('facts')
        if not held_300:
            held_300 = {}
        fact_sheet = held_300
        kind = me.get('label')
        section = list((rw for rw in row_map.values() if rw['opened'] and (None is kind or kind == rw['label'])))
        said = _value_terms(fact_sheet)
        missed = []
        foreign = self._needs_of_other_products(unit_ref)
        for text_line in self.need_lines():
            if not text_line in foreign:
                held_301 = self._store_phrases(text_line, row_map)
                result_info = _word_verdict(fact_sheet, text_line, held_301)
                if False is result_info or (None is result_info and self._quiet_where_peers_speak(me, text_line, said, section)):
                    missed.append(text_line)
            else:
                continue
        return missed

    def _quiet_where_peers_speak(self, me, text_line, said, section):
        prop_terms, vw = _need_terms(text_line) or (set(), set())
        held_302 = me.get('facts')
        if not held_302:
            held_302 = {}
        fact_sheet = held_302
        if not prop_terms or not vw or said >= vw:
            return False
        elif not any((prop_terms <= _terms(kname) or _key_is_property(kname, prop_terms) or (_terms(kname) and _terms(kname) < prop_terms) for kname in fact_sheet)):
            speakers = list((rw for rw in section if any((prop_terms <= _terms(kname) for kname in rw['facts']))))
            if not (2 <= len(speakers) and len(section) <= 2 * len(speakers)):
                for other in speakers:
                    if me is not other:
                        if self._affordable(other):
                            if _word_verdict(other['facts'], text_line) is True:
                                return True
                return False
            else:
                return True
        else:
            return False

    def _request_sentence(self, prop_name):
        prop_terms = _terms(prop_name)
        if prop_terms:
            for sentence in re.split('(?<=[.!?])\\s+', str(self.policy.get('query') or '')):
                if prop_terms <= _terms(sentence):
                    held_303 = sentence.strip()[:240]
                    return held_303
        return ''

class Finishing:

    def _closing_args(self, label_name, text, bag=None):
        held_304 = self.param_schemas.get(label_name)
        if not held_304:
            held_304 = {}
        arg_shape = held_304
        held_305 = arg_shape.get('properties')
        if not held_305:
            held_305 = {}
        field_specs = held_305
        if bag:
            made = dict(self.fixed.get(bag) or {})
        else:
            made = {}
        for key in arg_shape.get('required') or []:
            if not key in made:
                ok, field_value = _filler_value(field_specs.get(key) or {}, text)
                if not ok:
                    if 'order' == bag:
                        ok, field_value = _cart_lines_arg(self, field_specs.get(key) or {})
                if ok:
                    made[key] = field_value
            else:
                continue
        if 1 == len(field_specs):
            held_307 = field_specs.items()
            held_306 = iter(held_307)
            only, tool_desc = next(held_306)
            if only not in made:
                if isinstance(tool_desc, dict):
                    if tool_desc.get('type') == 'string':
                        if 'enum' not in tool_desc:
                            made[only] = _fit_length(text, tool_desc)
        return made

    def _scrubbed_decline(self, label_name, arguments):
        rule = self.plan.get('decline_no_names')
        if not (False is rule or not (rule or self._mentions_listing(arguments))):
            held_308 = (self.param_schemas.get(label_name) or {}).get('properties')
            if not held_308:
                held_308 = {}
            field_specs = held_308
            out = {}
            for key, field_value in arguments.items():
                held_309 = field_specs.get(key)
                if not held_309:
                    held_309 = {}
                tool_desc = held_309
                held_310 = isinstance(field_value, str)
                if held_310:
                    held_310 = 'enum' not in tool_desc
                free = held_310
                out[key] = _fit_length(NO_FIT_REMARK, tool_desc) if free else field_value
            return out
        else:
            return arguments

    def _mentions_listing(self, arguments):
        texts = list((fv for fv in arguments.values() if isinstance(fv, str)))
        held_311 = any((tk and tk in self.seen_ids for fv in texts for tk in _SPLIT_WORDS.split(fv)))
        return held_311

    def _send_finish(self, bag, text):
        tool = self.roles[bag]
        draft = self._closing_args(tool, text, bag)
        if 'decline' == bag:
            draft = self._scrubbed_decline(tool, draft)
        action = {'name': tool, 'args': draft}
        action['args'], issues = _fit_schema(draft, self.param_schemas[tool], 'args')
        log = {'kind': 'forced_finish', 'action': action}
        if not issues:
            log['reply'] = self.store_call(action)
            self.trace.append(log)
            held_312 = log['reply']
            return held_312
        else:
            log.update(issues=issues, reply=None)
            self.trace.append(log)
            return None

    def _listing_name(self, unit_ref):
        texts = list((fv for branch in _walk_dicts(self.records) if unit_ref in branch.values() for fv in branch.values() if isinstance(fv, str) and unit_ref != fv and (160 >= len(fv)) and fv.split()))
        top_choice = max(texts, key=lambda fv: len(fv.split()), default='')
        if top_choice:
            held_313 = '"%s"' % top_choice
            return held_313
        held_314 = 'item ' + unit_ref
        return held_314

    def finish_from_code(self):
        if not (self.finished or self.halted or self.steps_left() <= 0 or (self.seconds_left() < 2)):
            has = self.roles.get
            if has('order') and self.cart_used and (self.cart_ids or not has('cart')):
                bag = 'order'
            else:
                bag = 'decline'
            if has(bag):
                result_info = self._send_finish(bag, _CODE_FINISH_TEXT[bag])
                if 'order' != bag or not result_info or result_info.get('finished'):
                    return
                elif has('decline'):
                    if self.steps_left() > 0:
                        self._send_finish('decline', NO_FIT_REMARK)
            else:
                return
        else:
            return

    def run_batch(self, calls, planned=False):
        held_315 = _run_batch(self, {'calls': calls, 'planned': planned})['replies']
        return held_315

    def run(self):
        _run_episode(self, {})
        held_316 = self.trace
        return held_316

class Episode(ModelsAndStore, TextFields, RequestIntake, FinishJobs, Briefing, ShopperNews, StoreWording, SheetMenu, Reviews, ToolOffer, Messages, BasketLines, ListingTable, Finishing):

    def __init__(self, problem_data):
        self.start_clock = time.monotonic()
        env = problem_data['environment']
        self.binding = dict(env['binding'])
        self.policy = view = env['policy_view']
        self.tools = list((element for element in view.get('tools') or [] if _is_tool_entry(element)))
        specs = list((tool['function'] for tool in self.tools))
        self.param_schemas = dict(((tool_desc['name'], _tool_params(tool_desc)) for tool_desc in specs))
        self.max_steps = int(view.get('max_steps') or 30)
        held_317 = view.get('finished')
        self.finished = bool(held_317)
        for start, names in _INITIAL_VALUES:
            for label_name in names.split():
                setattr(self, label_name, start)
        for make, names in _INITIAL_CONTAINERS:
            for label_name in names.split():
                held_318 = make()
                setattr(self, label_name, held_318)
        self.wishes = Wishes()

class TurnCounters:
    __slots__ = ('idle', 'failures', 'counter')

    def __init__(self):
        self.idle = self.failures = self.counter = 0

class SearchPages:

    def __init__(self, query: Rec):
        self.query = query
        self.total = 0
        self.seen: set[int] = set()
        self.size = 0
        self.in_section = False
        self.lead_block = None
        self.holes: list[int] = []

    def take(self, start: int, shown: list[Rec], totals: list[int], label: str | None) -> None:
        self.total = max([self.total] + totals)
        held_319 = range(start, start + len(shown))
        self.seen.update(held_319)
        self.size = max(self.size, len(shown))
        if None is self.lead_block:
            if 0 == start:
                self.lead_block = tuple((json.dumps(rw, sort_keys=True) for rw in shown))
        if None is label or any((label in list((fv for fv in rw.values() if isinstance(fv, str))) for rw in shown)):
            self.in_section = True

    def term_set(self) -> set[str]:
        held_320 = _terms(' '.join((str(fv) for fv in self.query.values())))
        return held_320

    def key(self) -> str:
        held_321 = json.dumps(self.query, sort_keys=True)
        return held_321
Rec = dict[str, Any]

def _first_message_text(ep: Episode) -> str:
    blocks = ["The shopper's request:\n" + str(ep.policy.get('query') or '')]
    parsed_reading = ep.reading_brief()
    if parsed_reading:
        blocks.append(parsed_reading)
    first_view = ep.policy.get('opening')
    if None is not first_view:
        blocks.append('What the store showed first:\n' + _json_line(first_view))
    held_322 = ep.status_line()
    blocks.append(held_322)
    held_323 = '\n\n'.join(blocks)
    return held_323

def _flow_open(ep: Episode, bag: Rec) -> Rec:
    if ep.finished:
        bag['exit'] = 'already finished'
    else:
        ep.read_request()
        held_324 = dict(kind='plan', plan=ep.plan, roles=ep._roles_summary())
        ep.trace.append(held_324)
        seed = _first_message_text(ep)
        view = ep.policy.get('opening')
        if None is not view:
            ep.records.append(view)
            _collect_ids(view, ep.seen_ids)
        ep.messages = [dict(content=DRIVER_GUIDE, role='system'), dict(content=seed, role='user')]
    return bag

def _flow_turns(ep: Episode, bag: Rec) -> Rec:
    state = TurnCounters()
    remaining = MAX_TURNS
    while 0 < remaining:
        remaining = remaining - 1
        if _run_turn(ep, {'carry': state}).get('exit') == 'stop':
            break
    return bag

def _flow_settle(ep: Episode, bag: Rec) -> Rec:
    if not (ep.finished or ep.halted):
        ep.finish_from_code()
    return bag

def _flow_close(ep: Episode, bag: Rec) -> Rec:
    took = time.monotonic() - ep.start_clock
    held_325 = dict(kind='end', finished=ep.finished, steps=ep.step, cost=round(ep.cost, 5), seconds=round(took, 1))
    ep.trace.append(held_325)
    return bag

def _end_turn(ep: Episode, bag: Rec, give_up: bool) -> Rec:
    if give_up:
        ep.finish_from_code()
    bag['exit'] = 'stop' if give_up else 'next'
    return bag

def _flow_guard(ep: Episode, bag: Rec) -> Rec:
    if ep.finished or ep.halted or ep.steps_left() <= 0:
        bag['exit'] = 'stop'
        return bag
    elif not (ep.key_dry or ep.seconds_left() < LAST_SECONDS or SPEND_STOP < ep.cost):
        return bag
    else:
        held_326 = _end_turn(ep, bag, True)
        return held_326

def _flow_reply(ep: Episode, bag: Rec) -> Rec:
    queued = ep.queued_reply
    if queued:
        ep.queued_reply = ''
        if ep._send_reply(queued):
            bag['exit'] = 'next'
        return bag
    else:
        return bag

def _flow_planner(ep: Episode, bag: Rec) -> Rec:
    held_327 = ep.planner_on
    if held_327:
        held_327 = ep._ask_round()
    asked_now = held_327
    if asked_now:
        bag['exit'] = 'next'
    return bag

def _flow_infer(ep: Episode, bag: Rec) -> Rec:
    seat_list = [DRIVER_MODEL, SPARE_MODEL, DRIVER_MODEL]
    reply_obj = ep.ask_models(ep.messages, ep.tools_offered(), seat_list, 2500)
    if not None is reply_obj:
        bag['reply'] = reply_obj
        return bag
    else:
        state = bag['carry']
        state.failures = state.failures + 1
        held_328 = _end_turn(ep, bag, 2 <= state.failures or ep.key_dry)
        return held_328

def _tidy_call(raw_obj: Rec, number: int) -> Rec:
    held_329 = raw_obj.get('function')
    if not held_329:
        held_329 = {}
    func = held_329
    arguments = func.get('arguments')
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {})
    held_330 = {'id': raw_obj.get('id') or 'call_%d' % number, 'type': 'function', 'function': {'name': func.get('name'), 'arguments': arguments}}
    return held_330

def _flow_calls(ep: Episode, bag: Rec) -> Rec:
    state, reply_obj = (bag['carry'], bag['reply'])
    tidy = []
    for raw_obj in reply_obj.get('tool_calls') or []:
        if isinstance(raw_obj, dict):
            state.counter = state.counter + 1
            held_331 = _tidy_call(raw_obj, state.counter)
            tidy.append(held_331)
    held_332 = reply_obj.get('content')
    if not held_332:
        held_332 = ''
    text = held_332
    bag['calls'] = tidy
    bag['said'] = text if isinstance(text, str) else _json_line(text)
    return bag

def _commits(ep: Episode, proposed: list[Rec]) -> bool:
    jobs = list((ep.roles.get(bag) for bag in ('cart', 'order', 'decline')))
    names = set((ch['function']['name'] for ch in proposed))
    held_333 = any((None is not tool and tool in names for tool in jobs))
    return held_333

def _flow_commit(ep: Episode, bag: Rec) -> Rec:
    if ep.planner_on and _commits(ep, bag['calls']):
        basis = (ep.needs_revision, ep.property_menu_text()[1], len(ep.asked))
        if ep.forced_basis != basis:
            ep.forced_basis = basis
            before = ep.needs_revision
            dispatched = ep._ask_round(force=True)
            if dispatched:
                if ep.needs_revision != before:
                    bag['exit'] = 'next'
        return bag
    else:
        return bag

def _flow_idle(ep: Episode, bag: Rec) -> Rec:
    state = bag['carry']
    if not bag['calls']:
        state.idle = state.idle + 1
        ep.messages += [{'role': 'assistant', 'content': bag['said'] or '(no tool was called)'}, {'role': 'user', 'content': _NO_CALL_TEXT + ep.status_line()}]
        held_334 = _end_turn(ep, bag, 3 <= state.idle)
        return held_334
    else:
        state.idle = 0
        return bag

def _flow_act(ep: Episode, bag: Rec) -> Rec:
    proposed, text = (bag['calls'], bag['said'])
    held_335 = dict(kind='model', content=text[:2000], calls=list((ch['function'] for ch in proposed)))
    ep.trace.append(held_335)
    held_336 = dict(role='assistant', content=text or None, tool_calls=proposed)
    ep.messages.append(held_336)
    held_337 = ep.run_batch(proposed)
    ep._post_replies(held_337)
    return bag

def _spare_steps(max_steps: int) -> int:
    held_338 = max(5, round(0.11 * max_steps))
    return held_338
_REVEAL_RATE = 0.9
_FOLLOW_UP_STEPS = 2
_LEADING_WORD = re.compile('[^A-Za-z0-9]*([A-Za-z0-9]+)[^A-Za-z0-9]+(?=[A-Za-z0-9])')

def _contains_run(longer: list[str], shorter: list[str]) -> bool:
    width = len(shorter)
    held_339 = 2 <= width and any((longer[pos:pos + width] == shorter for pos in range(len(longer) - width + 1)))
    return held_339

def _already_asked(ep: Episode, prop_name: str) -> bool:
    own = _tokens(prop_name)
    for done in ep.asked_props:
        theirs = _tokens(done)
        if theirs == own or _contains_run(own, theirs) or _contains_run(theirs, own):
            return True
    return False

def _messages_kept(ep: Episode) -> int:
    held_340 = sum((step_proc['count'] for step_proc in ep._procedures_due('order') if 'message' == step_proc['role']))
    return held_340

def _want_terms(wish: str) -> set[str]:
    held_341 = _terms(re.sub('\\(.*?\\)', ' ', wish)) - _terms('!=')
    return held_341

def _settled(ep: Episode, prop_name: str, details_map: Rec) -> bool:
    spellings = [_terms(prop_name)] + list((_terms(alias) for alias in details_map.get('also') or []))
    for clause in ep.need_lines():
        lead_block, _, wish = str(clause).partition(':')
        head_words = _terms(lead_block)
        if not (head_words and any((pager and (head_words >= pager or pager >= head_words) for pager in spellings))):
            needed = _want_terms(wish)
            if needed:
                if any((needed <= _terms(field_value) for field_value in details_map.get('values') or [])):
                    return True
        else:
            return True
    return False

def _ask_order(ep: Episode) -> list[str]:
    details_map = ep.property_menu
    open_props = list((p for p in details_map if not _already_asked(ep, p) and (not _settled(ep, p, details_map[p])) and (not _variant_key(details_map[p]))))

    def order_key(p: str):
        element = details_map[p]
        held_342 = (_labelish(p, element), _not_an_item_trait(p), element.get('uniform', False), -element['count'], -_value_clusters(element), -len(element['values']), p)
        return held_342
    ranked = sorted(open_props, key=order_key)
    fresh = list((p for p in ranked if not _relative_asked(ep, p)))
    held_343 = fresh or ranked
    return held_343
_NOT_ITEM_TRAIT = re.compile('\\b(condition|options?|model|variants?|variety|grade|brand|age|ages|gender|occasion|kids?|application|recommended|care|assembly|warranty|features?|certifications?|included|suitable)\\b')

def _not_an_item_trait(prop_name: str) -> bool:
    held_344 = bool(_NOT_ITEM_TRAIT.search(re.sub('[^a-z0-9]+', ' ', prop_name.lower())))
    return held_344

def _relative_asked(ep: Episode, prop_name: str) -> bool:
    own = _tokens(prop_name)
    for done in ep.asked_props:
        theirs = _tokens(done)
        short, long_ = (own, theirs) if len(theirs) >= len(own) else (theirs, own)
        width = len(short)
        if width:
            if any((all((_name_word_like(short[k], long_[pos + k]) for k in range(width))) for pos in range(len(long_) - width + 1))):
                return True
    return False

def _labelish(prop_name: str, details_map: Rec) -> bool:
    held_345 = details_map.get('values')
    if not held_345:
        held_345 = []
    vals = held_345
    if not any(('/' in fv for fv in vals)):
        own = _singular_terms(prop_name)
        if own:
            self_named = sum((1 for fv in vals if own & _singular_terms(fv)))
        else:
            self_named = 0
        held_346 = len(vals) < self_named * 2
        return held_346
    else:
        return True

def _value_clusters(details_map: Rec) -> int:
    pools: list[set[str]] = []
    for field_value in details_map.get('values') or []:
        held_347 = _tokens(field_value)
        ws = set(held_347)
        linked = list((p for p in pools if p & ws))
        pools = list((p for p in pools if not p & ws)) + [ws.union(*linked)]
    held_348 = len(pools)
    return held_348

def _broad_value(wish: str, values: list[str]) -> bool:
    held_349 = re.sub('\\(.*?\\)', ' ', wish)
    phrase = _flat(held_349)
    needed = _terms(phrase) - _terms('!=')
    if not ('!=' in wish or not needed):
        flat_values = set((_flat(fv) for fv in values))
        members = set((_flat(mt) for fv in values for mt in re.split('[,;/|]', str(fv)) if mt.strip()))
        if not (phrase in flat_values or phrase in members):
            held_350 = len(list((fv for fv in flat_values if needed <= _terms(fv)))) >= 2
            return held_350
        else:
            return False
    else:
        return False

def _broad_wants(ep: Episode) -> dict[str, str]:
    found: dict[str, str] = {}
    menu = ep.property_menu
    for clause in ep.need_lines():
        lead_block, _, wish = str(clause).partition(':')
        head_words = _terms(lead_block)
        if head_words:
            close_fits = list((label_name for label_name in menu if head_words & _terms(label_name)))
            if not close_fits:
                continue
            elif _broad_value(wish, list((fv for label_name in close_fits for fv in menu[label_name].get('values') or []))):
                found[max(close_fits, key=lambda label_name: (menu[label_name]['count'], -len(label_name)))] = wish.strip()
        else:
            continue
    return found

def _variant_key(details_map: Rec) -> bool:
    held_351 = details_map.get('values')
    if not held_351:
        held_351 = []
    values = held_351
    held_352 = details_map.get('count')
    if not held_352:
        held_352 = 0
    seen_on = held_352
    if not (3 > seen_on or seen_on != len(values)):
        held_353 = 2 * len(list((fv for fv in values if '/' in fv))) > len(values)
        return held_353
    else:
        return False

def _time_to_ask(ep: Episode, sheets: int, forced: bool) -> bool:
    if sheets >= (1 if forced else 2):
        return True
    elif ep.plan.get('ask_first') and (not ep.asked) and (not ep.urgent_tried):
        return True
    elif not round(0.22 * ep.max_steps) > ep.step:
        held_354 = 2 <= sheets or not ep.roles.get('details')
        return held_354
    else:
        return False

def _ask_gate(ep: Episode, bag: Rec) -> Rec:
    tool = ep.roles.get('message')
    limit = ep._message_cap()
    used = ep.messages_used
    verdict = None
    if not (not tool or limit <= used):
        limit -= _messages_kept(ep)
        if limit <= used:
            verdict = False
    else:
        verdict = True
    if None is verdict:
        menu, sheets = ep.property_menu_text()
        if _time_to_ask(ep, sheets, bag['force']):
            spare = ep.steps_left() - _spare_steps(ep.max_steps)
            if limit - used + _FOLLOW_UP_STEPS > spare or ep.seconds_left() < WRAP_UP_SECONDS + 40:
                verdict = True
        else:
            verdict = False
    if not None is verdict:
        if verdict:
            ep.planner_on = False
        bag['exit'] = False
        return bag
    else:
        bag.update(sheets=sheets, menu=menu, cap=limit, tool=tool)
        return bag

def _neighbour_pick(ep: Episode, bag: Rec, sibling: str) -> Rec | None:
    hint = 'Ask about this property: "{}" (by its own name, with the product\'s noun). Why: the answer about "{}" brushed it off, and the two names overlap - that answer may have been meant for this one.'.format(sibling, ep.asked_props[-1])
    pick = _plan_question(ep, bag, hint)
    if None is pick or not _same_name(pick['property'], sibling):
        pick = {'property': sibling, 'question': _plain_question(ep, sibling), 'repeat': False, 'chance': None}
    if _worth_asking(ep, pick):
        return pick
    return None

def _planned_pick(ep: Episode, bag: Rec) -> Rec | None:
    split = ep.split_pending
    note = ''
    if split:
        note = 'The last answer seemed to cover several things. Property asked: "{}". Question sent: "{}".'.format(split[0], split[1])
    choice = _plan_question(ep, bag, note)
    held_355 = None is not choice
    if held_355:
        held_355 = not choice['repeat']
    new_lines = held_355
    if new_lines:
        if _quotes_sample(ep, choice):
            choice['question'] = _plain_question(ep, choice['property'])
    if split:
        if not _sharper_retry(choice, split):
            ep.asked_props.append(split[0])
            ep.split_pending = None
            if not new_lines:
                choice = _plan_question(ep, bag, '')
    if None is choice:
        if ep.reask_later:
            again = ep.reask_later.pop(0)
            held_356 = 'One more question is allowed about "{}", this time with the item\'s own noun.'.format(again)
            choice = _plan_question(ep, bag, held_356, again)
    if None is not choice and _worth_asking(ep, choice):
        return choice
    return None

def _reworded_pick(ep: Episode, prop_name: str) -> Rec | None:
    question = _plain_question(ep, prop_name)
    before = list((ent for ent in ep.asked if ent.startswith('[%s] ' % prop_name)))
    if not (before and _tokens(question) == _tokens(before[-1].split('"')[1] if '"' in before[-1] else '')):
        pick = {'property': prop_name, 'question': question, 'repeat': True, 'chance': None}
        if _worth_asking(ep, pick):
            return pick
        return None
    else:
        ep.asked_props.append(prop_name)
        return None

def _ask_draft(ep: Episode, bag: Rec) -> Rec:
    bag['picks'] = []
    sibling = ep.sibling_next
    ep.sibling_next = ''
    unclear = ep.unclear_next
    ep.unclear_next = ''
    if unclear:
        pick = _reworded_pick(ep, unclear)
    else:
        pick = None
    if None is pick:
        if ep.planner_on:
            if not (sibling and (not ep.split_pending) and (not _already_asked(ep, sibling)) and _products_turn(ep, sibling)):
                pick = _planned_pick(ep, bag)
            else:
                pick = _neighbour_pick(ep, bag, sibling)
                if None is not pick:
                    ep.sibling_props.append(pick['property'])
    if None is not pick:
        bag['picks'] = [pick]
    return bag

def _candidates_text(ep: Episode, order: list[str]) -> str:
    if not ep.asked_props:
        rule = 'Candidate properties follow, strongest first; the first question must be about one of the top three. A pick outside that needs a reason in "deviate".'
    else:
        rule = 'Candidate properties follow. A pick outside them needs a reason in "deviate".'
    text_lines = [rule]
    for cnt, prop_name in enumerate(order[:8], 1):
        held_357 = '[{}] {}'.format(cnt, prop_name)
        text_lines.append(held_357)
    held_358 = '\n'.join(text_lines)
    return held_358

def _planner_prompt(ep: Episode, bag: Rec, extra: str, steer: list[str] | None, wants_text: str) -> str:
    musts = ep.plan.get('ask_first')
    blocks = [ASK_PROMPT, 'The request, word for word:\n' + str(ep.policy.get('query') or ''), 'Requirements that apply now:\n' + wants_text]
    if musts:
        blocks.append('Only the shopper can settle: ' + '; '.join(map(str, musts)))
    if ep.asked:
        blocks.append('Questions already sent, with what came back:\n' + '\n'.join(ep.asked))
    blocks.append("Properties on the opened listings' sheets (name / listings carrying it / sample values):\n" + (bag['menu'] or '(nothing opened yet)'))
    if steer:
        held_359 = _candidates_text(ep, steer)
        blocks.append(held_359)
    if extra:
        blocks.append(extra)
    held_360 = '\n\n'.join(blocks)
    return held_360

def _first_question(reply: Rec) -> Rec:
    many = reply.get('questions')
    if not (isinstance(many, list) and many and isinstance(many[0], dict)):
        return reply
    else:
        held_361 = many[0]
        return held_361

def _text_field(raw_obj: Rec, key: str) -> str:
    held_362 = str(raw_obj.get(key) or '').strip()
    return held_362

def _held_to_list(ep: Episode, steer: list[str], raw_obj: Rec, prop_name: str, phrasing: str) -> tuple[str, str]:
    if ep.asked_props:
        window = steer[:8]
    else:
        window = steer[:3]
    if any((_same_name(prop_name, kname) for kname in window)):
        held_363 = (prop_name, phrasing)
        return held_363
    elif not (_text_field(raw_obj, 'deviate') and any((_same_name(prop_name, kname) for kname in steer))):
        top = steer[0]
        plain = _plain_question(ep, top)
        held_365 = dict(kind='ask_ranked', property=top, dropped=raw_obj.get('property'))
        ep.trace.append(held_365)
        held_366 = (top, plain)
        return held_366
    else:
        held_364 = (prop_name, phrasing)
        return held_364

def _plan_question(ep: Episode, bag: Rec, extra: str, again_for: str='') -> Rec | None:
    wants_text = ep.wants_text()
    held_367 = _ask_order(ep)
    ranked = _rotate_products(ep, held_367)
    if ranked and (not again_for) and (not ep.split_pending):
        steer = ranked
    else:
        steer = None
    prompt = _planner_prompt(ep, bag, extra, steer, wants_text)
    raw_obj = _first_question(ep.ask_json(prompt, 2000, [READING_MODEL, CHECKER_MODEL]) or {})
    prop_name, phrasing = (_text_field(raw_obj, 'property'), _text_field(raw_obj, 'question'))
    if steer:
        prop_name, phrasing = _held_to_list(ep, steer, raw_obj, prop_name, phrasing)
    if prop_name and phrasing:
        phrasing = _single_product_question(ep, prop_name, phrasing)
        if ep.split_pending:
            redo = ep.split_pending[0]
        else:
            redo = again_for
        held_368 = bool(redo)
        if held_368:
            held_368 = _same_name(prop_name, redo)
        repeat = held_368
        if again_for:
            if not (repeat and _terms(ep.plan.get('product')) & _terms(phrasing)):
                return None
        elif not repeat:
            if _already_asked(ep, prop_name):
                return None
        held_369 = raw_obj.get('chance')
        odds = _number_or_none(held_369)
        if None is not odds:
            held_370 = max(0.0, odds)
            odds = min(1.0, held_370)
        held_371 = dict(chance=odds, repeat=repeat, question=phrasing, property=prop_name)
        return held_371
    else:
        return None

def _name_word_like(a: str, b: str) -> bool:
    if not b == a:
        short, long_ = (a, b) if len(b) >= len(a) else (b, a)
        held_372 = 4 <= len(short) and long_.startswith(short)
        return held_372
    else:
        return True

def _neighbour_property(ep: Episode, prop_name: str) -> str:
    menu = list(ep.property_menu)
    if not 2 > len(menu):
        split = dict(((label_name, _tokens(label_name)) for label_name in menu))
        own_row = _tokens(prop_name)
        products = _product_names(ep)
        if 2 <= len(products):
            own_slot = _product_position(ep, prop_name, products)
        else:
            own_slot = None

        def too_common(term: str) -> bool:
            carriers = sum((1 for label_name in menu if any((_name_word_like(term, el) for el in split[label_name]))))
            held_373 = len(menu) <= carriers * 2
            return held_373
        top_choice, top_score = ('', 0)
        for label_name in menu:
            if _same_name(label_name, prop_name) or _already_asked(ep, label_name):
                continue
            elif _relative_asked(ep, label_name) or _settled(ep, label_name, ep.property_menu[label_name]):
                continue
            elif not (None is not own_slot and _product_position(ep, label_name, products) != own_slot):
                score = 0
                for pos, term in enumerate(split[label_name]):
                    if not (too_common(term) or not any((_name_word_like(term, el) for el in own_row))):
                        score += 2 if 0 == pos and _name_word_like(term, own_row[0]) else 1
                    else:
                        continue
                if not score:
                    continue
                if top_score < score or (top_score == score and ep.property_menu[label_name]['count'] > ep.property_menu.get(top_choice, {}).get('count', 0)):
                    top_choice, top_score = (label_name, score)
            else:
                continue
        return top_choice
    else:
        return ''

def _spoken_name(ep: Episode, prop_name: str) -> str:
    clean = ' '.join((tk for tk in prop_name.split() if 1 < len(tk) or tk.isdigit()))
    if not (clean and prop_name != clean):
        opener = _LEADING_WORD.match(prop_name)
        if not None is opener:
            shorter = prop_name[opener.end():].strip()
            if not (len(opener.group(1)) > 3 and len(_tokens(shorter)) < 2):
                every = list((cnt for kname, element in ep.property_menu.items() for cnt in [kname, *(element.get('also') or [])]))
                if any((_same_name(shorter, cnt) for cnt in every)):
                    return prop_name
                prefix = opener.group(1).lower()
                starts = set((tuple(_tokens(cnt)) for cnt in every if not _same_name(cnt, prop_name)))
                if len(list((tk for tk in starts if 2 <= len(tk) and prefix == tk[0]))) >= 2:
                    return shorter
                return prop_name
            else:
                return prop_name
        else:
            return prop_name
    else:
        held_374 = _spoken_name(ep, clean)
        return held_374

def _product_names(ep: Episode) -> list[str]:
    lined = list((item['product'] for item in getattr(ep, 'basket_items', list)()))
    if not lined:
        parts = list((seg.strip() for seg in re.split('[;,]', str(ep.plan.get('product') or '')) if seg.strip()))
        if 2 <= len(parts):
            return parts
        held_375 = []
        return held_375
    else:
        return lined

def _distinct_terms(segment: str, products: list[str]) -> set[str]:
    others = set().union(*list((_terms(seg) for seg in products if segment != seg)))
    held_376 = _terms(segment) - others
    return held_376

def _owning_product(ep: Episode, prop_name: str) -> str:
    products = _product_names(ep)
    if products:
        names = {prop_name.strip().lower(), *list((al.lower() for al in (ep.property_menu.get(prop_name) or {}).get('also') or []))}
        items = getattr(ep, 'basket_items', list)()
        if not items:
            pool: set[str] = set()
            for sheet in ep._detail_sheets():
                if names & set(sheet):
                    for cell in sheet.values():
                        pool |= _terms(cell)
            scored = list(((len(_distinct_terms(seg, products) & pool), -pos, seg) for pos, seg in enumerate(products)))
            top = max(scored)
            if top[0]:
                words = top[2].split()
                while 1 < len(words) and _terms(words[0]) and (not _terms(words[0]) & pool):
                    words.pop(0)
                held_378 = ' '.join(words)
                return held_378
            else:
                return ''
        else:
            votes: dict[int, int] = {}
            for record_blob in _walk_dicts(ep.records):
                listing = next((cell for cell in record_blob.values() if isinstance(cell, str) and cell in ep.seen_ids), None)
                if not None is listing:
                    keys = set((ep._key_as_words(kname).strip().lower() for part in _walk_dicts([record_blob]) for kname in part))
                    if names & keys:
                        index = ep._product_index(listing)
                    else:
                        index = None
                    if None is not index:
                        votes[index] = votes.get(index, 0) + 1
                else:
                    continue
            ranked = sorted(votes.items(), key=lambda kv: -kv[1])
            if not (ranked and (1 == len(ranked) or ranked[1][1] < ranked[0][1])):
                return ''
            else:
                held_377 = items[ranked[0][0]]['product']
                return held_377
    else:
        return ''

def _single_product_question(ep: Episode, prop_name: str, phrasing: str) -> str:
    products = _product_names(ep)
    if products:
        owner = _owning_product(ep, prop_name)
    else:
        owner = ''
    if owner:
        said = _terms(phrasing)
        named = list((seg for seg in products if _distinct_terms(seg, products) & said))
        if not (1 == len(named) and _terms(owner) & said):
            held_379 = _plain_question(ep, prop_name)
            return held_379
        else:
            return phrasing
    else:
        return phrasing

def _product_position(ep: Episode, prop_name: str, products: list[str]) -> int | None:
    said = _owning_product(ep, prop_name)
    if not said in products:
        owner = _terms(said)
        held_381 = next((pos for pos, seg in enumerate(products) if owner and owner <= _terms(seg)), None)
        return held_381
    else:
        held_380 = products.index(said)
        return held_380

def _least_asked_products(ep: Episode, products: list[str]) -> set[int] | None:
    if not (2 > len(products) or not ep.asked_props):
        asks = [0] * len(products)
        for done in ep.asked_props:
            slot = _product_position(ep, done, products)
            if None is not slot:
                asks[slot] += 1
        fewest = min(asks)
        least = set((pos for pos, n in enumerate(asks) if fewest == n))
        if len(products) == len(least):
            return None
        return least
    else:
        return None

def _products_turn(ep: Episode, prop_name: str) -> bool:
    products = _product_names(ep)
    least = _least_asked_products(ep, products)
    if None is not least:
        slot = _product_position(ep, prop_name, products)
    else:
        slot = None
    held_382 = None is slot or slot in least
    return held_382

def _rotate_products(ep: Episode, ranked: list[str]) -> list[str]:
    products = _product_names(ep)
    least = _least_asked_products(ep, products)
    if not None is least:
        kept = list((p for p in ranked if _product_position(ep, p, products) in least))
        held_383 = kept or ranked
        return held_383
    else:
        return ranked

def _names_its_product(ep: Episode, prop_name: str, question: str) -> bool:
    held_384 = _owning_product(ep, prop_name)
    if not held_384:
        held_384 = str(ep.plan.get('product') or '')
    noun = held_384
    words = list((tk for tk in _tokens(noun) if 2 < len(tk)))
    held_385 = ' '.join(words[:1] + words[-1:])
    edges = _singular_terms(held_385)
    held_386 = bool((set((tk for tk in _singular_terms(noun) if 3 < len(tk))) | edges) & _singular_terms(question))
    return held_386

def _plain_question(ep: Episode, prop_name: str) -> str:
    said = _spoken_name(ep, prop_name)
    owner = _owning_product(ep, prop_name)
    if not owner:
        noun = str(ep.plan.get('product') or '').strip()
        if not (not noun or len(noun.split()) > 2):
            held_389 = 'What {} do you want for the {}?'.format(said, noun)
            return held_389
        else:
            held_388 = 'What {} do you want?'.format(said)
            return held_388
    else:
        held_387 = 'What {} do you want for the {}?'.format(said, owner)
        return held_387

def _quotes_sample(ep: Episode, pick: Rec) -> bool:
    menu, prop_name = (ep.property_menu, pick['property'])
    held_390 = menu.get(prop_name)
    if not held_390:
        held_390 = next((menu[kname] for kname in menu if _same_name(kname, prop_name)), None)
    element = held_390
    own = _terms(prop_name)
    if not (not element or any((own < _terms(kname) for kname in menu))):
        spoken = _terms(pick['question']) - own - _terms(ep.plan.get('product'))
        held_391 = (getattr(ep, 'policy', None) or {}).get('query')
        wished = _terms(held_391)
        samples = list((_terms(field_value) for field_value in element['values']))
        held_392 = any((vw and spoken >= vw and (not wished >= vw) for vw in samples))
        return held_392
    else:
        return False

def _sharper_retry(pick: Rec | None, pending: tuple[str, str]) -> bool:
    if not (None is pick or not pick['repeat'] or (not _same_name(pick['property'], pending[0]))):
        held_393 = _tokens(pick['question']) != _tokens(pending[1])
        return held_393
    else:
        return False

def _worth_asking(ep: Episode, pick: Rec) -> bool:
    price = ep.question_cost
    if price:
        held_394 = price.get('share')
        share = _number_or_none(held_394)
        if not (None is share or not 0 <= share <= 1):
            if None is pick['chance']:
                odds = 0.5
            else:
                odds = pick['chance']
            allowed = share < odds * _REVEAL_RATE
        else:
            held_395 = not ep.asked_props
            if held_395:
                held_395 = not ep.split_pending
            allowed = held_395
        if not allowed:
            ep.planner_on = False
        return allowed
    else:
        return True

def _ask_build(ep: Episode, bag: Rec) -> Rec:
    ready = []
    for pick in bag['picks']:
        text = str(pick.get('question')).strip()
        msg = ep._message_call(text, 'ask_%d_%d' % (ep.messages_used + 1, ep.step))
        if None is not msg:
            ready.append((msg, (str(pick.get('property')), text, bool(pick.get('repeat')))))
    if not ready:
        ep.empty_plans = ep.empty_plans + 1
        ep.urgent_tried = True
        if bag.get('sheets', 0) >= 2 or 3 <= ep.empty_plans:
            ep.planner_on = False
        bag['exit'] = False
        return bag
    else:
        bag['calls'] = list((msg for msg, _ in ready))
        bag['topics'] = list((topic for _, topic in ready))
        return bag

def _log_answer(ep: Episode, subject: tuple[str, str, bool], text: str) -> None:
    prop_name, question, repeat = subject
    ep.asked.append('[%s] "%s" => %s' % (prop_name, question, text[:400]))
    heard = (ep.answer_reading or {}).get('class', 'other')
    _file_answer(ep, prop_name, question, repeat, heard)

def _mark_new_need(ep: Episode) -> None:
    if ep.messages:
        last = ep.messages[-1]
    else:
        last = {}
    if last.get('role') == 'tool':
        last['content'] += _ANSWER_FOLLOW_UP

def _ask_send(ep: Episode, bag: Rec) -> Rec:
    outgoing = bag['calls']
    subjects = bag['topics']
    version_before = ep.needs_revision
    note = 'Asking the shopper (code-planned question) about %s.' % '; '.join((tk[0] for tk in subjects))
    held_396 = dict(tool_calls=outgoing, content=note, role='assistant')
    ep.messages.append(held_396)
    ep.asking, ep.answer_reading = (subjects[0][0], None)
    held_397 = ep.run_batch(outgoing, planned=True)
    texts = ep._post_replies(held_397)
    for subject, text in zip(subjects, texts):
        _log_answer(ep, subject, text)
    ep.asking = ''
    if version_before != ep.needs_revision:
        _mark_new_need(ep)
    ep.planner_on = ep.messages_used < bag['cap']
    bag['exit'] = True
    return bag

def _file_answer(ep: Episode, prop_name: str, text: str, repeat: bool, heard: str) -> None:
    if repeat:
        ep.split_retried = True
        ep.split_pending = None
    elif 'split' == heard:
        ep.split_pending = (prop_name, text)
        return
    elif 'unclear' == heard:
        if not _names_its_product(ep, prop_name, text):
            ep.unclear_next = prop_name
            return
        elif prop_name not in ep.reask_later:
            ep.reask_later.append(prop_name)
    elif 'not_picky' == heard:
        wish = _broad_wants(ep).get(prop_name)
        if wish:
            if _want_terms(wish) <= _terms(text):
                ep.loose_ok[prop_name] = wish
        if prop_name in ep.sibling_props:
            ep.sibling_next = ''
        else:
            ep.sibling_next = _neighbour_property(ep, prop_name)
        spoken, label_name = (_terms(text), _terms(prop_name))
        held_398 = bool(spoken & _terms(ep.plan.get('product')))
        if not held_398:
            held_399 = 2 <= len(label_name)
            if held_399:
                held_399 = spoken >= label_name
            held_398 = held_399
        anchored = held_398
        if not anchored:
            if prop_name not in ep.reask_later:
                ep.reask_later.append(prop_name)
    if not _already_asked(ep, prop_name):
        ep.asked_props.append(prop_name)

def _cheaper_rows(ep, pick: str, row_map: dict[str, Rec], unopened_only: bool, ties: bool=False) -> list[str]:
    held_400 = row_map.get(pick)
    if not held_400:
        held_400 = {}
    ref = held_400
    top, label = (ref.get('price'), ref.get('label'))
    if not None is top:

        def under(price) -> bool:
            if ties:
                held_402 = top >= price
                return held_402
            held_403 = top > price
            return held_403
        hits = []
        for rid, listing_row in row_map.items():
            if pick == rid or None is listing_row['price'] or (not under(listing_row['price'])):
                continue
            elif not ep._affordable(listing_row) or (None is not label and label != listing_row['label']):
                continue
            elif not (ep._passes_by_words(listing_row) is False or (unopened_only and listing_row['opened'])):
                hits.append((listing_row['price'], rid))
            else:
                continue
        hits.sort()
        held_404 = list((rid for _, rid in hits))
        return held_404
    else:
        held_401 = []
        return held_401

def _cheaper_after_refusal(ep, row_map: dict[str, Rec] | None=None) -> list[str]:
    target_value = (getattr(ep, 'plan', None) or {}).get('priority')
    if ep.cart_ids:
        held = ep.cart_ids[-1]
    else:
        held = None
    if not (not ep.refusals or None is held or 'cheapest' != target_value or (ep.steps_left() < 4)):
        spare = ep.steps_left() - 3
        found = _cheaper_rows(ep, held, ep.listing_table() if None is row_map else row_map, True, True)
        held_406 = found[:max(1, min(4, spare))]
        return held_406
    else:
        held_405 = []
        return held_405

def _offset_arg(ep, tool: str) -> tuple[str | None, int | None]:
    held_407 = ep.param_schemas.get(tool)
    if not held_407:
        held_407 = {}
    spec_of = held_407
    must = set(spec_of.get('required') or [])
    for key, tool_desc in (spec_of.get('properties') or {}).items():
        held_408 = isinstance(tool_desc, dict)
        if held_408:
            held_408 = key not in must
        if held_408:
            held_408 = tool_desc.get('type') == 'integer'
        paging = held_408
        if paging:
            if 'maximum' in tool_desc:
                held_409 = (key, int(tool_desc['maximum']))
                return held_409
    held_410 = (None, None)
    return held_410

def _offset_pages(ep, tool: str, offset: str):
    for element in ep.trace:
        reply = element.get('reply')
        if not (element.get('kind') != 'env' or not isinstance(reply, dict)):
            first = (element.get('actions') or [{}])[0]
            if not first.get('name') != tool:
                query = dict(first.get('args') or {})
                start = query.pop(offset, 0)
                body = _peel_wrappers(reply.get('observation'))[1]
                if isinstance(body, dict):
                    if isinstance(start, int):
                        yield (query, start, body)
            else:
                continue
        else:
            continue

def _searches_by_query(ep, tool: str, offset: str, label: str | None) -> list[SearchPages]:
    by_query: dict[str, SearchPages] = {}
    for query, start, body in _offset_pages(ep, tool, offset):
        tables = list((fv for fv in body.values() if isinstance(fv, list) and fv and all((isinstance(rw, dict) for rw in fv))))
        shown = max(tables, key=len, default=[])
        totals = list((fv for fv in body.values() if isinstance(fv, int) and (not isinstance(fv, bool)) and (len(shown) <= fv)))
        if shown:
            if totals:
                sig = json.dumps(query, sort_keys=True)
                if sig not in by_query:
                    by_query[sig] = SearchPages(query)
                by_query[sig].take(start, shown, totals, label)
    held_411 = list(by_query.values())
    return held_411

def _search_terms(ep) -> tuple[set[str], set[str]]:
    try:
        conds = ep.need_lines()
    except AttributeError:
        conds = []
    wanted = set().union(*(_terms(str(ch).split(':', 1)[-1]) for ch in conds))
    vocab = wanted | _terms((getattr(ep, 'plan', None) or {}).get('product'))
    try:
        conds, row_index = (ep.need_lines(), ep.listing_table())
    except AttributeError:
        conds, row_index = ([], {})
    pairs = list(((_terms(p), _terms(w)) for p, _, w in (str(ch).partition(':') for ch in conds)))
    for listing_row in row_index.values():
        facts = (listing_row.get('facts') or {}).items()
        for needed, field_value in pairs:
            if field_value:
                for key, text in facts:
                    if needed <= _terms(key) or _key_is_property(key, needed):
                        if field_value & _terms(text):
                            vocab |= _terms(text)
    held_412 = (wanted, vocab)
    return held_412

def _unfinished_searches(ep, label: str | None, singular: bool=False) -> list[tuple[int, Rec, int, int]]:
    tool = ep.roles.get('search')
    if tool:
        offset, limit = _offset_arg(ep, tool)
        if not None is offset:
            searches = _searches_by_query(ep, tool, offset, label)
            wanted, vocab = _search_terms(ep)
            held_415 = (ep.param_schemas.get(tool) or {}).get('properties')
            if not held_415:
                held_415 = {}
            tool_desc = held_415

            def typed(pager: SearchPages) -> set[str]:
                held_416 = _terms(' '.join((str(fv) for kname, fv in pager.query.items() if isinstance(fv, str) and (not (tool_desc.get(kname) or {}).get('enum')))))
                return held_416
            pending = []
            for pager in searches:
                pager.holes = list((idx for idx in range(min(pager.total, limit + pager.size)) if idx not in pager.seen))
                if pager.holes:
                    if pager.in_section:
                        if pager.size:
                            pending.append(pager)
            pending.sort(key=lambda pager: (len(pager.term_set()), pager.key()))
            unique: list[SearchPages] = []
            for pager in pending:
                if not any((u.lead_block and pager.lead_block == u.lead_block and (pager.total == u.total) for u in unique)):
                    unique.append(pager)
            if not singular:
                aimed = list((pager for pager in unique if typed(pager) and typed(pager) <= vocab))
            else:
                held_417 = ' '.join(vocab)
                roots = _singular_terms(held_417)
                aimed = list((pager for pager in unique if typed(pager) and _singular_terms(' '.join(typed(pager))) <= roots))
            if wanted:
                if aimed:
                    unique = list((pager for pager in unique if pager in aimed or not typed(pager)))
            scored = []
            for pager in unique:
                step_on = dict(pager.query)
                step_on[offset] = pager.holes[0]
                remaining = (len(pager.holes) + pager.size - 1) // pager.size
                scored.append((-len(wanted & pager.term_set()), pager.total, json.dumps(step_on, sort_keys=True), (pager.total, step_on, len(pager.seen), remaining)))
            scored.sort(key=lambda tk: tk[:3])
            held_418 = list((tk[3] for tk in scored))
            return held_418
        else:
            held_414 = []
            return held_414
    else:
        held_413 = []
        return held_413

def _search_history(ep):
    tool = ep.roles.get('search')
    if tool:
        for element in ep.trace:
            if not element.get('kind') != 'env':
                dispatched = (element.get('actions') or [{}])[0]
                if dispatched.get('name') == tool:
                    yield (element, dispatched.get('args') or {})
            else:
                continue
    else:
        return

def _search_results(ep):
    for element, sent_args in _search_history(ep):
        reply = element.get('reply')
        if isinstance(reply, dict):
            body = _peel_wrappers(reply.get('observation'))[1]
            if isinstance(body, dict):
                tables = list((fv for fv in body.values() if isinstance(fv, list) and fv and all((isinstance(rw, dict) for rw in fv))))
                yield (sent_args, body, max(tables, key=len, default=[]))
            else:
                continue
        else:
            continue

def _plain_values(ep, conditions=None) -> list[str]:
    try:
        if None is conditions:
            conditions = ep.need_lines()
        else:
            conditions = conditions
    except AttributeError:
        held_419 = []
        return held_419
    selected = []
    for condition in conditions:
        constraint = str(condition).partition(':')[2]
        if _relation_of(constraint) is None:
            lead_block = re.split('[(,;]|!=', constraint, maxsplit=1)[0]
            held_420 = lead_block.strip().strip('"\'.')
            selected.append(held_420)
    held_421 = list(dict.fromkeys((fv for fv in selected if fv and 33 >= len(fv))))
    return held_421

def _searched_alone(ep, field_value: str) -> bool:
    own = _terms(field_value)
    for _, sent_args in _search_history(ep):
        typed_sets = (_terms(tk) for tk in sent_args.values() if isinstance(tk, str))
        if any((typed and own >= typed for typed in typed_sets)):
            return True
    return False

def _typed_terms(sent_args, paging, props) -> set[str]:
    term_set: set[str] = set()
    for key, text in sent_args.items():
        tool_desc = props.get(key)
        if paging != key:
            if isinstance(text, str):
                if not (isinstance(tool_desc, dict) and 'enum' in tool_desc):
                    term_set |= _terms(text)
    return term_set

def _row_sources(ep) -> dict[str, tuple[int, int]]:
    tool = ep.roles.get('search')
    if tool:
        paging = _offset_arg(ep, tool)[0]
    else:
        paging = None
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    wanted = list((w for w in map(_terms, _plain_values(ep)) if w))
    top_choice: dict[str, tuple[int, int]] = {}
    for sent_args, _, row_map in _search_results(ep):
        if paging:
            base = sent_args.get(paging, 0)
        else:
            base = 0
        if isinstance(base, bool) or not isinstance(base, int):
            base = 0
        typed = _typed_terms(sent_args, paging, props)
        held_422 = list((w for w in wanted if typed >= w))
        score = len(held_422)
        for position, listing_row in enumerate(row_map, base):
            ident = next((fv for kname, fv in listing_row.items() if isinstance(fv, str) and 'id' in str(kname).lower() and ID_PATTERN.match(fv)), None)
            if None is not ident:
                old_score, old_pos = top_choice.get(ident, (0, position))
                top_choice[ident] = (max(old_score, score), min(old_pos, position))
    return top_choice

def _ignored_part(key, own_row, theirs, here, there):
    if None is own_row or None is theirs:
        field_value, owner = (theirs, there) if None is own_row else (own_row, here)
        held_423 = (True, '{} {}'.format(key, field_value), '{} "{}"'.format(key, field_value), _terms(field_value), owner)
        return held_423
    elif isinstance(own_row, str) and isinstance(theirs, str):
        owner = here
        if _terms(own_row) < _terms(theirs):
            owner, own_row, theirs = (there, theirs, own_row)
        big, small = (_terms(own_row), _terms(theirs))
        if not big > small:
            held_425 = (True, key, '%s (both "%s" and "%s" brought back the very same rows)' % (key, own_row, theirs), big ^ small, owner)
            return held_425
        else:
            extra = big - small
            quoted = '"' + ' '.join((w for w in re.findall('[a-z0-9]+', own_row.lower()) if w in extra)) + '"'
            held_424 = (False, quoted, quoted, extra, owner)
            return held_424
    else:
        return None

def _ignored_parts(ep) -> list[str]:
    tool = ep.roles.get('search')
    if tool:
        paging = _offset_arg(ep, tool)[0]
    else:
        paging = None
    history: list[tuple[dict[str, Any], str]] = []
    groups: dict[bool, list[str]] = {True: [], False: []}
    tag_names: set[str] = set()
    for raw_args, body, row_map in _search_results(ep):
        if row_map:
            here = dict(raw_args)
            totals = list((fv for fv in body.values() if isinstance(fv, int) and (not isinstance(fv, bool)) and (len(row_map) <= fv)))
            stamp = json.dumps([max(totals, default=None), row_map], sort_keys=True)
            row_stems = list((_singular_terms(' '.join((str(fv) for fv in rw.values() if not isinstance(fv, (dict, list))))) for rw in row_map))
            for there, there_stamp in history:
                if not stamp != there_stamp:
                    differing = list((kname for kname in set(here) | set(there) if here.get(kname) != there.get(kname)))
                    if not (1 != len(differing) or paging == differing[0]):
                        key = differing[0]
                        held_426 = here.get(key)
                        held_427 = there.get(key)
                        found = _ignored_part(key, held_426, held_427, here, there)
                        if not None is found:
                            whole, tag, phrasing, term_set, owner = found
                            if not term_set or tag in tag_names or all((_singular_terms(' '.join(term_set)) <= pager for pager in row_stems)):
                                continue
                            tag_names.add(tag)
                            held_429 = json.dumps(owner, sort_keys=True)
                            held_428 = '{} in {}'.format(phrasing, held_429)
                            groups[whole].insert(0, held_428)
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            history.append((here, stamp))
        else:
            continue
    held_430 = groups[True] + groups[False]
    return held_430

def _paging_finishes(ep, view: tuple[int, dict[str, Any], int, int]) -> bool:
    tool = ep.roles.get('search')
    if tool:
        limit = _offset_arg(ep, tool)[1]
    else:
        limit = None
    if not None is limit:
        total, _, read, pages = view
        if pages:
            per_page = -((read - total) // pages)
        else:
            per_page = 0
        held_431 = limit + max(per_page, 1) >= total
        return held_431
    else:
        return False

def _searches_text(views) -> str:
    held_432 = '; '.join(('%s (%s of %s rows read)' % (json.dumps(view[1], sort_keys=True), view[2], view[0]) for view in views))
    return held_432

def _digit_shape(text) -> str:
    held_433 = re.sub('[0-9]+', '#', str(text))
    return held_433

def _page_candidates(ep) -> list[str]:
    held_435 = ep.roles.get('search')
    held_434 = ep.param_schemas.get(held_435)
    if not held_434:
        held_434 = {}
    spec_of = held_434
    must = set(spec_of.get('required') or [])
    held_436 = list((key for key, tool_desc in (spec_of.get('properties') or {}).items() if key not in must and isinstance(tool_desc, dict) and ('enum' not in tool_desc) and (tool_desc.get('type') in ('integer', 'string'))))
    return held_436

def _next_page_values(body, kind, start) -> list:
    found = list((fv for fv in (body or {}).values() if isinstance(fv, (int, str)) and (not isinstance(fv, bool))))
    if not int is kind:
        held_439 = list((fv for fv in found if isinstance(fv, str) and re.fullmatch('[A-Za-z0-9_.:=+/-]{1,64}', fv)))
        return held_439
    else:
        numbers = list((fv for fv in found if isinstance(fv, int)))
        if numbers:
            held_438 = list((fv for fv in numbers if (start or 0) < fv < max(numbers)))
            return held_438
        else:
            held_437 = []
            return held_437

def _result_body(element):
    reply = element.get('reply')
    if isinstance(reply, dict):
        body = _peel_wrappers(reply.get('observation'))[1]
    else:
        body = None
    if isinstance(body, dict):
        return body
    return None

def _previous_page(ep, key, sent_args):
    rest = dict(((kname, fv) for kname, fv in sent_args.items() if key != kname))
    found = None
    for element, past in _search_history(ep):
        body = _result_body(element)
        if None is not body:
            if dict(((kname, fv) for kname, fv in past.items() if key != kname)) == rest:
                found = (past.get(key), body)
    return found

def _cursor_arg(ep) -> str | None:
    keys = list((key for key in _page_candidates(ep) if ((ep.param_schemas.get(ep.roles.get('search')) or {}).get('properties') or {})[key].get('type') == 'string'))
    handed: set[str] = set()
    used, free = (set(), set())
    for element, sent_args in _search_history(ep):
        for key in keys:
            if key in sent_args:
                (used if isinstance(sent_args[key], str) and sent_args[key] in handed else free).add(key)
        handed |= set(_next_page_values(_result_body(element), str, 0))
    held_440 = list((key for key in keys if key in used and key not in free))
    if not held_440:
        held_440 = list((key for key in keys if key not in used | free))
    picks = held_440
    if 1 == len(picks) and handed:
        held_441 = picks[0]
        return held_441
    return None

def _cursor_searches(ep, label) -> list[tuple[int, dict[str, Any], int, int]]:
    tool = ep.roles.get('search')
    if tool and _offset_arg(ep, tool)[0] is None:
        key = _cursor_arg(ep)
    else:
        key = None
    if not None is key:
        state: dict[str, list] = {}
        for element, sent_args in _search_history(ep):
            body = _result_body(element)
            if not None is body:
                tables = list((fv for fv in body.values() if isinstance(fv, list) and fv and all((isinstance(rw, dict) for rw in fv))))
                shown = max(tables, key=len, default=[])
                query = dict(((kname, fv) for kname, fv in sent_args.items() if key != kname))
                sig = json.dumps(query, sort_keys=True)
                rec = state.setdefault(sig, [query, set(), 0, len(shown), False, None])
                rec[1].update((json.dumps(rw, sort_keys=True) for rw in shown))
                rec[2] = max([rec[2]] + list((fv for fv in body.values() if isinstance(fv, int) and (not isinstance(fv, bool)) and (len(shown) <= fv))))
                rec[4] = rec[4] or None is label or any((label in list((fv for fv in rw.values() if isinstance(fv, str))) for rw in shown))
                tokens = list((fv for fv in _next_page_values(body, str, 0) if fv != sent_args.get(key)))
                rec[5] = tokens[0] if 1 == len(tokens) else None
            else:
                continue
        views = []
        for query, rows, total, size, in_section, token in state.values():
            if None is not token:
                if in_section:
                    if size:
                        if total > len(rows):
                            step_on = dict(query)
                            step_on[key] = token
                            views.append((total, step_on, len(rows), -(-(total - len(rows)) // size)))
        held_443 = sorted(views, key=lambda view: (view[3], json.dumps(view[1], sort_keys=True)))
        return held_443
    else:
        held_442 = []
        return held_442

def _note(key: str, **field_specs) -> str:
    held_444 = NOTES[key].format(**field_specs)
    return held_444

def _hold(key: str, **field_specs) -> str:
    held_445 = HELD + ' ' + _note(key, **field_specs)
    return held_445

def _check_known_tool(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    if not unit_ref['name'] in ep.param_schemas:
        held_446 = sorted(batch['allowed'])
        tools = ', '.join(held_446)
        held_447 = "{}: there is no tool '{}'. Tools you can call now: {}.".format(SKIPPED, unit_ref['name'], tools)
        return held_447
    else:
        return None

def _check_offered(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    if batch['planned'] or unit_ref['name'] in batch['allowed']:
        return None
    elif not (ep.planner_on and batch['message'] == unit_ref['name']):
        held_449 = SKIPPED + ': this close to the step limit, the tools left are ' + ', '.join(sorted(batch['allowed'])) + '.'
        return held_449
    else:
        held_448 = '{}: skip messaging - the code puts the questions to the shopper and the answers come in on their own. Use the turn on listings instead: open details, cart a good fit, then order.'.format(SKIPPED)
        return held_448

def _check_json(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    raw_obj = unit_ref['raw']
    if isinstance(raw_obj, dict):
        parsed = raw_obj
    elif not (raw_obj or '').strip():
        parsed = {}
    else:
        parsed = _reply_object(raw_obj)
    unit_ref['args'] = parsed
    if not None is parsed:
        return None
    else:
        held_450 = SKIPPED + ': the arguments could not be read (one JSON object is expected).'
        return held_450

def _check_schema(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    unit_ref['args'], issues = _fit_schema(unit_ref['args'], ep.param_schemas[unit_ref['name']], 'args')
    if issues:
        held_451 = SKIPPED + ": the arguments do not fit the tool's schema: " + '; '.join(issues[:6])
        return held_451
    else:
        return None

def _check_finish_alone(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    if not (1 >= batch['count'] or unit_ref['name'] not in batch['finishing']):
        held_452 = SKIPPED + ': an order or a decline must travel by itself. Let the other results come back first, then put it in a separate reply.'
        return held_452
    else:
        return None

def _check_repeat_fault(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    label_name, arguments = (unit_ref['name'], unit_ref['args'])
    if not arguments or label_name in batch['finishing']:
        return None
    elif not (ep._call_signature(label_name, arguments) not in ep.faulty and label_name + _json_line(arguments) not in ep.faulty):
        held_453 = '{}: {} gave the same error twice for calls shaped like this; send arguments that follow its schema exactly, or use another tool.'.format(SKIPPED, label_name)
        return held_453
    else:
        return None

def _check_message_quota(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    limit = ep.message_budget
    if None is limit or batch['message'] != unit_ref['name']:
        return None
    elif not limit > ep.messages_used + batch['queued_messages']:
        held_454 = SKIPPED + ': the message allowance is used up - no more messages to the shopper.'
        return held_454
    else:
        return None

def _check_duplicate(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    label_name = unit_ref['name']
    unit_ref['sig'] = label_name + _json_line(unit_ref['args'])
    if not (label_name in (batch['message'], batch['cart']) or label_name in batch['finishing'] or unit_ref['sig'] not in ep.recent):
        held_455 = SKIPPED + ': the same call as just before. Nothing has changed and its result is above; change the arguments or do something else.'
        return held_455
    else:
        return None

def _check_late_reads(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    held_456 = unit_ref['name'] == ep.roles.get('details')
    if held_456:
        held_456 = ep.steps_left() <= 7
    if held_456:
        held_456 = ep.refusals
    late_retry = held_456
    if late_retry:
        allowed = _cheaper_after_refusal(ep)
        asked_for = list((fv for fv in unit_ref['args'].values() if isinstance(fv, str)))
        if not (not allowed or any((fv in allowed for fv in asked_for))):
            held_457 = SKIPPED + ': this near the end, a details read is allowed only for cheaper listings never opened, which are ' + ', '.join(allowed) + '. For the listing in the cart, the cart view shows price and stock.'
            return held_457
        else:
            return None
    else:
        return None

def _check_late_unfitted(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    tool = ep.roles.get('details')
    remaining = ep.steps_left()
    held_458 = ep.roles.get('order')
    if held_458:
        held_458 = ep.roles.get('cart')
    if held_458:
        held_458 = not ep.order_attempts
    narrowed = held_458
    if not (tool != unit_ref['name'] or 7 < remaining or ep.refusals or (not narrowed)):
        held_459 = list((1 for _, name, _ in batch.get('ready') or [] if tool == name))
        queued = len(held_459)
        allowed = _rows_for_unfitted(ep, 2)
        asked_for = list((fv for fv in unit_ref['args'].values() if isinstance(fv, str)))
        if allowed and remaining - queued >= ep._reserve() + 2 and any((fv in allowed for fv in asked_for)):
            return None
        elif allowed:
            held_461 = SKIPPED + ': with the end this close, a single further details read remains, reserved for a product still lacking a fit, from ' + ', '.join(allowed) + '.'
            return held_461
        else:
            held_460 = SKIPPED + ': ' + _note('reserve_late')
            return held_460
    else:
        return None

def _cart_add_target(ep, label_name: str, arguments: Rec) -> str | None:
    held_462 = _cart_target(ep, label_name, arguments)
    return held_462

def _check_cart_add(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    tool, call_args = (unit_ref['name'], unit_ref['args'])
    if not (batch['cart'] != tool or ep.steps_left() <= 3):
        sig = ep.add_signature
        if not (None is not sig and sig != ep._mode_signature(tool, call_args)):
            target = _cart_add_target(ep, tool, call_args)
            if target:
                tag = '%s|%s' % (target, ep._state_tag())
            else:
                tag = ''
            if not (not tag or tag in ep.cart_checked):
                ep.cart_checked.add(tag)
                doubt = ep.second_check(target, 'about to go into the cart')
                if doubt:
                    held_463 = SKIPPED + '. ' + doubt + ' If you are sure it fits, send it again unchanged.'
                    return held_463
                return None
            else:
                return None
        else:
            return None
    else:
        return None

def _check_units(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    if not batch['cart'] != unit_ref['name']:
        issue = ep._units_mismatch(unit_ref['name'], unit_ref['args'])
        if not (not issue or unit_ref['sig'] in ep.qty_warned):
            ep.qty_warned.add(unit_ref['sig'])
            held_464 = '{}: {} If you are sure, send it again unchanged.'.format(SKIPPED, issue)
            return held_464
        else:
            return None
    else:
        return None

def _check_stacking(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    tool, call_args = (unit_ref['name'], unit_ref['args'])
    if batch['cart'] != tool or None is ep.add_signature:
        return None
    elif not ep._mode_signature(tool, call_args) != ep.add_signature:
        in_cart = list((fv for fv in call_args.values() if isinstance(fv, str) and fv in ep.cart_ids))
        if in_cart:
            held, needed = (in_cart[0], ep.units_wanted())
            if ep.basket_items():
                held_465 = ep._units_for(held)
                if not held_465:
                    held_465 = max((el['quantity'] for el in ep.basket_items()))
                needed = held_465
            units = None
            if ep.cart_changed_at <= ep.cart_seen_at:
                units = ep.units_in_cart.get(held)
            if None is not units:
                if units + ep._units_sent(tool, call_args) <= needed:
                    return None
            tag = 'readd %s %s' % (held, ep.cart_changed_at)
            if tag in ep.held_arms:
                return None
            ep.held_arms.add(tag)
            if None is units:
                shown = ''
            else:
                shown = ' (%s according to the newest cart view)' % units
            held_466 = '%s One thing to watch: %s is in the cart already%s. Adding it again can take the units beyond %s, the number the request wants. Order it as it is, or remove it first if the number has to change. Only if more units are really meant, send it again unchanged.' % (HELD, held, shown, needed)
            return held_466
        else:
            return None
    else:
        return None

def _check_reread(ep: Episode, batch: Rec, unit_ref: Rec) -> str | None:
    if ep.closing_kind(unit_ref['name'], unit_ref['args']) != 'order':
        return None
    elif ep.refusals and ep.cart_ids and ep._room_for(1):
        since = max(ep.refused_at, ep.cart_changed_at)
        stale = list((idx for idx in (ep.cart_ids if ep.basket_items() else ep.cart_ids[-1:]) if ep.mentioned.get(idx, -1) <= since))
        tag = 'reread %s %s' % (ep.refused_at, ep.cart_changed_at)
        if not (not stale or tag in ep.held_arms):
            held = ', '.join(stale)
            ep.held_arms.add(tag)
            held_467 = '%s The order waits: %s (price, stock, units in the cart) has not been read again since the refusal or the last cart change. Read it with a tool that shows those - the cart view lists the units - and then order. If that read just happened, send it again unchanged.' % (HELD, held)
            return held_467
        else:
            return None
    else:
        return None

def _is_order(ep, call) -> bool:
    held_468 = ep.closing_kind(call['name'], call['args']) == 'order'
    return held_468

def _check_rerank(ep, batch, call):
    if not (not ep.rerank_reason or not _is_order(ep, call) or (not ep.cart_ids)):
        change = ep.rerank_reason
        ep.rerank_reason = ''
        if ep._room_for(3):
            lower = _cheaper_rows(ep, ep.cart_ids[-1], ep.listing_table(), False)[:4]
            if lower:
                seen = _note('relax_seen', ids=', '.join(lower))
            else:
                seen = ''
            held_469 = _hold('relax', seen=seen, change=change)
            return held_469
        else:
            return None
    else:
        return None

def _cheaper_lead(ep, row_map, held):
    held_470 = min(4, ep.steps_left() - ep._reserve() - 1)
    limit = max(1, held_470)
    sources = _row_sources(ep)
    ranked = sorted(_cheaper_rows(ep, held, row_map, True), key=lambda idx: (-sources.get(idx, (0, 0))[0], row_map[idx]['price']))
    if not ranked[:limit]:
        label = (row_map.get(held) or {}).get('label')
        open_views = list((fv for fv in _unfinished_searches(ep, label) if ep._room_for(fv[3] + 2)))
        complete = list((fv for fv in open_views if _paging_finishes(ep, fv)))
        complete += list((fv for fv in _cursor_searches(ep, label) if ep._room_for(fv[3] + 2)))
        bare = list((fv for fv in _plain_values(ep) if not _searched_alone(ep, fv)))
        held_472 = complete
        if not held_472:
            if bare:
                held_472 = []
            else:
                held_472 = open_views
        pending = held_472
        if pending:
            held_473 = ('pages', pending[:2])
            return held_473
        elif not (bare and ep._room_for(4)):
            return None
        else:
            held_474 = ('bare', (bare[:2], label))
            return held_474
    else:
        held_471 = ('rows', ranked[:limit])
        return held_471

def _check_cheapest(ep, batch, call):
    if not (ep.plan.get('priority') != 'cheapest' or not _is_order(ep, call) or (not ep.cart_ids)):
        row_map = ep.listing_table()
        held_475 = ep._state_tag()
        held_476 = len(row_map)
        arm_key = 'cheapest {} {}'.format(held_475, held_476)
        if not (arm_key in ep.held_arms or not ep._room_for(2)):
            opener = _cheaper_lead(ep, row_map, ep.cart_ids[-1])
            if not None is opener:
                ep.held_arms.add(arm_key)
                form, what = opener
                if 'rows' == form:
                    held_477 = _hold('cheap_rows', ids=', '.join(what))
                    return held_477
                if 'pages' == form:
                    held_478 = _hold('cheap_pages', pages=_searches_text(what))
                    return held_478
                values, label = what
                held_479 = _hold('cheap_bare', values=' / '.join(('"%s"' % fv for fv in values)), scope=_note('cheap_scope', label=label) if label else '')
                return held_479
            else:
                return None
        else:
            return None
    else:
        return None

def _measure_extremes(ep, detail) -> list[str]:
    terms = _terms(detail)
    found: dict[str, dict[str, float]] = {}
    for facts in list((rw['facts'] for rw in ep.listing_table().values() if rw['opened'])):
        for key, text in facts.items():
            if isinstance(text, str):
                if terms & _terms(key):
                    for num, unit in re.findall('(\\d+(?:\\.\\d+)?)\\s*([a-z]+)', text.lower()):
                        found.setdefault(unit, {})[num + unit] = float(num)
    if found:
        for _, _, shown in _search_results(ep):
            for rw in shown:
                for text in list((fv for fv in rw.values() if isinstance(fv, str))):
                    for num, unit in re.findall('(\\d+(?:\\.\\d+)?)\\s*([a-z]+)\\b', text.lower()):
                        if unit in found:
                            found[unit][num + unit] = float(num)
        ends = []
        for unit in sorted(found, key=lambda un: -len(found[un]))[:2]:
            ranked = sorted(found[unit], key=lambda tk: found[unit][tk])
            ends += list(dict.fromkeys([ranked[0], ranked[-1]]))
        return ends
    else:
        held_480 = []
        return held_480

def _check_best_probe(ep, batch, call):
    held_481 = ep.plan
    if not held_481:
        held_481 = {}
    plan = held_481
    if not (plan.get('priority') != 'best' or not _is_order(ep, call) or (not ep.cart_ids)):
        arm_key = 'best ' + ep._state_tag()
        if not (arm_key in ep.held_arms or not ep._room_for(4)):
            detail = str(plan.get('priority_detail') or '')
            ends, values = (_measure_extremes(ep, detail), list((fv for fv in _plain_values(ep) if _terms(fv))))
            key, cap = _query_arg(ep)
            if not (not ends or not values or None is key):
                units = set((re.sub('[0-9.]+', '', tk) for tk in ends))
                for _, args in _search_history(ep):
                    text = str(args.get(key) or '').lower()
                    figures = set((unit for _, unit in re.findall('(\\d+(?:\\.\\d+)?)\\s*([a-z]+)', text)))
                    if figures & units:
                        if any((_terms(fv) <= _terms(text) for fv in values)):
                            return None
                ep.held_arms.add(arm_key)
                needles = list((_cut_at_word(values[0] + ' ' + tk, cap or 64) for tk in ends[:2]))
                held_482 = _hold('best_probe', detail=detail or 'the measure', ends=', '.join(ends), needles=' / '.join((json.dumps(fv) for fv in needles)))
                return held_482
            else:
                return None
        else:
            return None
    else:
        return None

def _check_release(ep, batch, call):
    tool, dispatched = (call['name'], call['args'])
    if batch['cart'] != tool or None is ep.add_signature:
        return None
    elif not ep._mode_signature(tool, dispatched) == ep.add_signature:
        target = _cart_target(ep, tool, dispatched)
        held_483 = ep._state_tag()
        arm_key = 'keep {} {}'.format(target, held_483)
        if not (not ep.cart_generation or target not in ep.cart_ids or arm_key in ep.held_arms or (not ep._room_for(1))):
            facts = ep.listing_table().get(target)
            if not facts or not ep._affordable(facts) or ep._passes_by_words(facts) is not True:
                return None
            ep.held_arms.add(arm_key)
            budget = ep.wishes.budget
            if None is budget:
                within = ''
            else:
                within = _note('release_budget', budget=budget)
            held_484 = _hold('release', within=within, price=facts['price'], target=target)
            return held_484
        else:
            return None
    else:
        return None

def _fit_hint(ep, held, row_map, unmet) -> str:
    section = row_map[held].get('label')
    passing = []
    for ident, facts in row_map.items():
        if held == ident or not facts['opened'] or (not ep._affordable(facts)):
            continue
        elif None is section or section == facts['label']:
            if ep._passes_by_words(facts) is True:
                passing.append(ident)
    if not passing:
        phrases = list((ch for ch in dict.fromkeys((ep._request_sentence(str(u).partition(':')[0]) for u in unmet)) if ch))
        if phrases:
            held_486 = _note('fit_words', phrases=' | '.join(('"%s"' % ch for ch in phrases[:2])))
            return held_486
        else:
            return ''
    else:
        held_485 = _note('fit_others', ids=', '.join(passing[:3]))
        return held_485

def _check_fit(ep, batch, call):
    if not (not _is_order(ep, call) or not ep.cart_ids or (not ep._room_for(2))):
        row_map = ep.listing_table()
        for held in ep.cart_ids[:-1] if ep.basket_items() else []:
            note = _check_fit_line(ep, held, row_map)
            if note:
                return note
        held_487 = _check_fit_line(ep, ep.cart_ids[-1], row_map)
        return held_487
    else:
        return None

def _check_fit_line(ep, held, row_map):
    held_488 = ep._state_tag()
    arm_key = 'fit {} {}'.format(held, held_488)
    if not arm_key in ep.held_arms:
        held_489 = row_map.get(held)
        if not held_489:
            held_489 = {}
        facts = held_489
        if not facts.get('opened'):
            note = _hold('fit_unread', held=held)
        elif ep._affordable(facts):
            unmet = ep._unmet_for(held, row_map)
            if unmet:
                note = _hold('fit_unmet', held=held, unmet='; '.join(unmet), pointer=_fit_hint(ep, held, row_map, unmet))
            else:
                return None
        else:
            if not facts.get('stock') is False:
                why = _note('fit_dear', price=facts.get('price'), budget=ep.wishes.budget)
            else:
                why = _note('fit_sold')
            note = _hold('fit_money', why=why, held=held)
        ep.held_arms.add(arm_key)
        return note
    else:
        return None

def _unopened_hits(ep, needed, limit):
    if needed and 0 < limit:
        row_map = ep.listing_table()
    else:
        row_map = {}
    if row_map:
        picks = []
        for ident, (score, position) in _row_sources(ep).items():
            facts = row_map.get(ident)
            if needed > score or None is facts or facts['opened'] or (False is facts['stock']):
                continue
            elif ep._affordable(facts):
                if ep._passes_by_words(facts) is not False:
                    picks.append((-score, position, ident))
        held_491 = list((ident for _, _, ident in sorted(picks)))
        return held_491
    else:
        held_490 = []
        return held_490
_SEARCHES_BEFORE_READING = 6

def _wanted_sections(ep) -> set[str]:
    tool = ep.roles.get('search')
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    chosen = set()
    for _, sent_args in _search_history(ep):
        for key, fv in sent_args.items():
            if isinstance(fv, str):
                if isinstance(props.get(key), dict):
                    if props[key].get('enum'):
                        held_492 = fv.strip().lower()
                        chosen.add(held_492)
    labels = set((rw['label'].strip().lower() for rw in ep.listing_table().values() if isinstance(rw.get('label'), str)))
    exact = set((lb for lb in labels if lb in chosen))
    held_493 = list((el['product'] for el in ep.basket_items()))
    if not held_493:
        held_493 = [(ep.plan or {}).get('product') or '']
    products = held_493
    for product in products:
        stems = ep._stems(product)
        if stems:
            if not any((stems & ep._stems(lb) for lb in exact)):
                exact |= set((lb for lb in labels if stems & ep._stems(lb)))
    return exact

def _result_total(body, shown) -> int:
    counts = list((fv for fv in body.values() if isinstance(fv, int) and (not isinstance(fv, bool)) and (len(shown) <= fv)))
    held_494 = max(counts, default=len(shown))
    return held_494

def _row_id(listing_row):
    held_495 = next((fv for kname, fv in listing_row.items() if isinstance(fv, str) and 'id' in str(kname).lower() and ID_PATTERN.match(fv)), None)
    return held_495

def _sharpest_sources(ep) -> dict[str, tuple[int, int, int]]:
    tool = ep.roles.get('search')
    if tool:
        paging = _offset_arg(ep, tool)[0]
    else:
        paging = None
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    wanted = list((w for w in map(_terms, _plain_values(ep)) if w))
    best: dict[str, tuple[int, int, int]] = {}
    for sent_args, body, shown in _search_results(ep):
        typed = _typed_terms(sent_args, paging, props)
        rank = (-len(list((w for w in wanted if typed >= w))), _result_total(body, shown))
        for position, listing_row in enumerate(shown):
            ident = _row_id(listing_row)
            if None is not ident:
                held_496 = best.get(ident, rank + (position,))
                best[ident] = min(held_496, rank + (position,))
    return best

def _rows_to_open(ep, limit, only=None, typed_only=False) -> list[str]:
    row_map = ep.listing_table()
    sections = _wanted_sections(ep)
    ranked = []
    for ident, (minus, total, position) in _sharpest_sources(ep).items():
        facts = row_map.get(ident)
        if not (None is facts or facts['opened'] or (typed_only and (not minus))):
            label = str(facts.get('label') or '').strip().lower()
            if sections and label not in sections or not ep._affordable(facts):
                continue
            elif not ep._passes_by_words(facts) is False:
                if isinstance(facts.get('price'), (int, float)):
                    price = facts['price']
                else:
                    price = float('inf')
                ranked.append(((minus, total, price, position), label, ident))
            else:
                continue
        else:
            continue
    queues: dict[str, list[str]] = {}
    for _, label, ident in sorted(ranked):
        queues.setdefault(label, []).append(ident)
    picks: list[str] = []
    while limit > len(picks) and any(queues.values()):
        for label, waiting in queues.items():
            while waiting and limit > len(picks):
                ident = waiting.pop(0)
                if None is only or ep._product_index(ident) in only:
                    picks.append(ident)
                    break
    return picks

def _products_without_fit(ep) -> set:
    row_map = ep.listing_table()
    many = ep.basket_items()
    fitted = set()
    for ident, facts in row_map.items():
        if facts['opened']:
            if ep._affordable(facts):
                if not ep._unmet_for(ident, row_map):
                    fitted.add(ep._product_index(ident) if many else 0)
    held_497 = set(range(len(many) or 1)) - fitted
    return held_497

def _rows_for_unfitted(ep, limit) -> list[str]:
    still = _products_without_fit(ep)
    if still:
        held_499 = _rows_to_open(ep, limit, still if ep.basket_items() else None, typed_only=True)
        return held_499
    else:
        held_498 = []
        return held_498

def _reads_so_far(ep, batch) -> int:
    tools = {ep.roles.get('details'), ep.roles.get('view')} - {None}
    sent = list(((el.get('actions') or [{}])[0].get('name') for el in ep.trace if el.get('kind') == 'env'))
    sent += list((name for _, name, _ in batch.get('ready') or []))
    held_500 = len(list((name for name in sent if name in tools)))
    return held_500

def _is_next_page(ep, sent_args) -> bool:
    tool = ep.roles.get('search')
    keys = {_offset_arg(ep, tool)[0], _cursor_arg(ep)} - {None}
    handed: set = set()
    for element, _ in _search_history(ep):
        held_502 = _result_body(element)
        held_501 = _next_page_values(held_502, str, 0)
        handed.update(held_501)
    keys |= set((key for key, fv in sent_args.items() if isinstance(fv, str) and fv in handed))
    if keys & set(sent_args):
        rest = dict(((key, fv) for key, fv in sent_args.items() if key not in keys))
        held_503 = any((dict(((key, fv) for key, fv in past.items() if key not in keys)) == rest for _, past in _search_history(ep)))
        return held_503
    else:
        return False

def _section_noun(label: str) -> str:
    words = list((w for w in re.findall("[A-Za-z]+(?:'[A-Za-z]+)?", label) if "'" not in w and 2 < len(w)))
    if words:
        held_504 = max(reversed(words), key=len).lower()
        return held_504
    return ''

def _in_section_example(ep) -> str:
    tool = ep.roles.get('search')
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    page_keys = {_offset_arg(ep, tool)[0], _cursor_arg(ep)} - {None}
    row_map, sections = (ep.listing_table(), _wanted_sections(ep))
    off = total = 0
    example = ''
    for sent_args, _, shown in _search_results(ep):
        texts = list(((key, fv) for key, fv in sent_args.items() if isinstance(fv, str) and key not in page_keys and (not (props.get(key) or {}).get('enum')) and fv.strip()))
        picked = list((fv.strip().lower() for key, fv in sent_args.items() if isinstance(fv, str) and (props.get(key) or {}).get('enum') and (fv.strip().lower() in sections)))
        if not (not texts or not picked or (not shown)):
            labels = list((str((row_map.get(_row_id(rw)) or {}).get('label') or '').strip().lower() for rw in shown))
            held_505 = list((lb for lb in labels if lb and picked[0] != lb))
            strays = len(held_505)
            off, total = (off + strays, total + len(shown))
            noun = _section_noun(picked[0])
            key, words = texts[0]
            if len(shown) < strays * 2:
                if noun:
                    if noun not in _terms(words):
                        cap = (props.get(key) or {}).get('maxLength')
                        if isinstance(cap, int):
                            room = cap - len(noun) - 1
                        else:
                            room = len(words)
                        if 0 < room:
                            example = _cut_at_word(words.strip(), room) + ' ' + noun
                        else:
                            example = ''
        else:
            continue
    if total and total < off * 2:
        return example
    return ''

def _check_search_flood(ep, batch, call):
    tool = ep.roles.get('search')
    if not (not tool or tool != call['name'] or batch['planned'] or ('flood' in ep.held_arms)):
        sent = len(list(_search_history(ep))) + len(list((1 for _, name, _ in batch.get('ready') or [] if tool == name)))
        if _SEARCHES_BEFORE_READING > sent or _reads_so_far(ep, batch) or (not ep._room_for(3)):
            return None
        elif not _is_next_page(ep, call.get('args') or {}):
            picks = _rows_to_open(ep, 4)
            if not picks:
                return None
            ep.held_arms.add('flood')
            example = _in_section_example(ep)
            if example:
                mixed = _note('flood_mixed', example=json.dumps(example))
            else:
                mixed = ''
            held_506 = _hold('flood', count=sent, ids=', '.join(picks), mixed=mixed)
            return held_506
        else:
            return None
    else:
        return None

def _query_arg(ep) -> tuple[str | None, int | None]:
    tool = ep.roles.get('search')
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    paging = {_offset_arg(ep, tool)[0], _cursor_arg(ep)} - {None}
    keys = list((key for key, tk in props.items() if isinstance(tk, dict) and tk.get('type') == 'string' and (not tk.get('enum')) and (key not in paging)))
    if keys:
        spaced = dict(((key, len(list((1 for _, args in _search_history(ep) if ' ' in str(args.get(key) or '').strip())))) for key in keys))
        key = max(keys, key=lambda kname: (spaced[kname], props[kname].get('maxLength') or 0))
        cap = props[key].get('maxLength')
        held_508 = (key, cap if isinstance(cap, int) else None)
        return held_508
    else:
        held_507 = (None, None)
        return held_507

def _stray_sections(ep) -> list[str]:
    tool = ep.roles.get('search')
    if tool:
        props = (ep.param_schemas.get(tool) or {}).get('properties') or {}
    else:
        props = {}
    page_keys = {_offset_arg(ep, tool)[0], _cursor_arg(ep)} - {None}
    row_map, sections = (ep.listing_table(), _wanted_sections(ep))
    off = total = 0
    chosen: list[str] = []
    for sent_args, _, shown in _search_results(ep):
        worded = any((isinstance(fv, str) and fv.strip() and (key not in page_keys) and (not (props.get(key) or {}).get('enum')) for key, fv in sent_args.items()))
        picked = list((fv.strip().lower() for key, fv in sent_args.items() if isinstance(fv, str) and (props.get(key) or {}).get('enum') and (fv.strip().lower() in sections)))
        if not (not worded or not picked or (not shown)):
            labels = list((str((row_map.get(_row_id(rw)) or {}).get('label') or '').strip().lower() for rw in shown))
            off, total = (off + len(list((lb for lb in labels if lb and picked[0] != lb))), total + len(shown))
            chosen = [picked[0]] + list((lb for lb in chosen if picked[0] != lb))
        else:
            continue
    if total and total < off * 2:
        return chosen
    held_509 = []
    return held_509

def _section_queries(ep) -> list[str]:
    key, cap = _query_arg(ep)
    if None is not key:
        labels = _stray_sections(ep)
    else:
        labels = []
    many, still = (ep.basket_items(), _products_without_fit(ep))
    conditions = None
    if many:
        owner = dict(((lb, list((k for k, el in enumerate(many) if ep._stems(el['product']) & ep._stems(lb)))) for lb in labels))
        labels = list((lb for lb in labels if 1 == len(owner[lb]) and owner[lb][0] in still))
        if labels:
            element = many[owner[labels[0]][0]]
            conditions = element['requirements'] + ep._filed_needs(element['product'])
    if labels:
        noun = _section_noun(labels[0])
    else:
        noun = ''
    if noun:
        sent = list((_singular_terms(fv) for _, args in _search_history(ep) for kname, fv in args.items() if key == kname and isinstance(fv, str)))
        out: list[str] = []
        for field_value in _plain_values(ep, conditions):
            if not len(field_value.split()) > 3:
                for needle in (field_value + ' ' + noun, field_value):
                    if None is cap or cap >= len(needle):
                        if _singular_terms(needle) not in sent:
                            if needle not in out:
                                out.append(needle)
                                break
            else:
                continue
        held_511 = out[:2]
        return held_511
    else:
        held_510 = []
        return held_510

def _stays_in_section(ep, query, keys, sections) -> bool:
    row_map = ep.listing_table()
    labels = []
    for sent_args, _, shown in _search_results(ep):
        if dict(((kname, fv) for kname, fv in sent_args.items() if kname not in keys)) == query:
            labels += list((str((row_map.get(_row_id(rw)) or {}).get('label') or '').strip().lower() for rw in shown))
    labels = list((lb for lb in labels if lb))
    held_512 = not sections or not labels or 2 * len(list((lb for lb in labels if lb in sections))) >= len(labels)
    return held_512

def _value_searches(ep) -> list[tuple[int, dict[str, Any], int, int]]:
    tool = ep.roles.get('search')
    if tool:
        number_key = _offset_arg(ep, tool)[0]
        keys = {number_key, _cursor_arg(ep)} - {None}
        held_514 = (ep.param_schemas.get(tool) or {}).get('properties')
        if not held_514:
            held_514 = {}
        props = held_514
        wanted, sections = (_search_terms(ep)[0], _wanted_sections(ep))
        tokens = _cursor_searches(ep, None)
        _, limit = _offset_arg(ep, tool)
        picked = []
        for view in tokens + _unfinished_searches(ep, None):
            per_page = max(1, (view[0] - view[2]) // max(view[3], 1))
            if None is not limit and view not in tokens:
                reach = limit + per_page
            else:
                reach = _CURSOR_PAGE_LIMIT * per_page
            if not 2 * reach < view[0]:
                query = dict(((kname, fv) for kname, fv in view[1].items() if kname not in keys))
                if not wanted & _typed_terms(query, number_key, props) or not _stays_in_section(ep, query, keys, sections):
                    continue
                elif ep._room_for(min(view[3], 2) + 2):
                    picked.append((view not in tokens and (not _paging_finishes(ep, view)), view))
            else:
                continue
        held_515 = list((view for _, view in sorted(picked, key=lambda tk: tk[0])))[:2]
        return held_515
    else:
        held_513 = []
        return held_513
_CURSOR_PAGE_LIMIT = 5
_EARLY_ROOM = 8

def _check_reach_early(ep, batch, call):
    tool = ep.roles.get('search')
    if not (not tool or tool != call['name'] or batch['planned']):
        arm_key = 'reach early ' + ep._state_tag()
        if not (arm_key in ep.held_arms or not ep._room_for(_EARLY_ROOM)):
            sent = len(list(_search_history(ep))) + len(list((1 for _, name, _ in batch.get('ready') or [] if tool == name)))
            if not (_SEARCHES_BEFORE_READING > sent or _is_next_page(ep, call.get('args') or {}) or (not _products_without_fit(ep))):
                views, needles = (_value_searches(ep), _section_queries(ep))
                if not views:
                    if not needles:
                        return None
                ep.held_arms.add(arm_key)
                advice = []
                if views:
                    held_516 = _note('early_pages', pages=_searches_text(views))
                    advice.append(held_516)
                if needles:
                    held_517 = _note('early_words', needles=' / '.join((json.dumps(fv) for fv in needles)))
                    advice.append(held_517)
                held_518 = _hold('reach_early', count=sent, advice=' '.join(advice))
                return held_518
            else:
                return None
        else:
            return None
    else:
        return None

def _check_reach(ep, batch, call):
    kind = ep.closing_kind(call['name'], call['args'])
    if not 'decline' != kind:
        held_519 = ep._state_tag()
        arm_key = 'reach {} {}'.format(kind, held_519)
        if not (arm_key in ep.held_arms or not ep._room_for(3) or ep.seconds_left() < WRAP_UP_SECONDS + 30):
            held_520 = list((fv for fv in _plain_values(ep) if _terms(fv)))
            needed = len(held_520)
            limit = min(4, ep.steps_left() - ep._reserve() - 2)
            hits = _unopened_hits(ep, needed, limit)
            if not hits:
                readable = list((fv for fv in _unfinished_searches(ep, None, singular=True) if _paging_finishes(ep, fv) and ep._room_for(fv[3] + 2)))[:2]
                ignored = _ignored_parts(ep)[:2]
                if ep.roles.get('search') in batch['allowed']:
                    needles = _section_queries(ep)
                else:
                    needles = []
                if not readable:
                    if not needles:
                        return None
                ep.held_arms.add(arm_key)
                if readable:
                    advice = [_note('reach_read', pages=_searches_text(readable))]
                else:
                    advice = []
                if needles:
                    held_522 = _note('early_words', needles=' / '.join((json.dumps(fv) for fv in needles)))
                    advice.append(held_522)
                if ignored:
                    held_523 = _note('reach_idle', ignored='; '.join(ignored))
                    advice.append(held_523)
                held_524 = _hold('reach_pages', advice='; also '.join(advice))
                return held_524
            else:
                ep.held_arms.add(arm_key)
                held_521 = _hold('reach_rows', ids=', '.join(hits[:limit])) + ep._view_tool_hint()
                return held_521
        else:
            return None
    else:
        return None

def _check_paging(ep, batch, call):
    tool = ep.roles.get('search')
    held_525 = call.get('args')
    if not held_525:
        held_525 = {}
    sent_args = held_525
    if not (not tool or tool != call['name'] or (not isinstance(sent_args, dict)) or (not ep._room_for(2))):
        text_key = _cursor_arg(ep)
        number_key = _offset_arg(ep, tool)[0]
        for key in _page_candidates(ep):
            field_value = sent_args.get(key)
            if isinstance(field_value, int) and (not isinstance(field_value, bool)):
                kind = int
            else:
                kind = str
            if not isinstance(field_value, (int, str)) or isinstance(field_value, bool) or (int is kind and number_key != key):
                continue
            elif not (str is kind and text_key == key and (not field_value.strip())):
                prior = _previous_page(ep, key, sent_args)
                if not (None is prior and str is kind and (text_key == key) and field_value.strip()):
                    if None is prior or field_value == prior[0]:
                        continue
                    if any((tool == name and dict(((k, v) for k, v in (args or {}).items() if key != k)) == dict(((k, v) for k, v in sent_args.items() if key != k)) for _, name, args in batch.get('ready') or [])):
                        continue
                    if isinstance(prior[0], int) and (not isinstance(prior[0], bool)):
                        start = prior[0]
                    else:
                        start = 0
                    offered = _next_page_values(prior[1], kind, start)
                    if int is kind:
                        if start >= field_value:
                            continue
                    if str is kind:
                        if text_key != key:
                            offered = list((fv for fv in offered if _digit_shape(fv) == _digit_shape(field_value)))
                    if not offered:
                        if str is kind:
                            if _invented_page(ep, key, field_value):
                                memo = 'none ' + tool + _json_line(sent_args)
                                if not memo in ep.held_arms:
                                    ep.held_arms.add(memo)
                                    needles = _section_queries(ep)
                                    if needles:
                                        words = _note('page_words', needles=' / '.join((json.dumps(fv) for fv in needles)))
                                    else:
                                        words = ''
                                    held_528 = _hold('page_none', value=json.dumps(field_value), words=words)
                                    return held_528
                                else:
                                    return None
                    if 1 != len(offered) or field_value in offered:
                        continue
                    memo = 'page ' + tool + _json_line(sent_args)
                    if memo in ep.held_arms:
                        return None
                    ep.held_arms.add(memo)
                    step_on = dict(sent_args)
                    step_on[key] = offered[0]
                    held_529 = _hold('page_arg', value=json.dumps(field_value), offered=json.dumps(offered[0]), call=_json_line(step_on))
                    return held_529
                else:
                    memo = 'fresh ' + tool + _json_line(sent_args)
                    if not memo in ep.held_arms:
                        ep.held_arms.add(memo)
                        first = dict(((kname, fv) for kname, fv in sent_args.items() if key != kname))
                        held_527 = _hold('page_fresh', value=json.dumps(field_value), call=_json_line(first))
                        return held_527
                    else:
                        return None
            else:
                memo = 'empty ' + tool + _json_line(sent_args)
                if not memo in ep.held_arms:
                    ep.held_arms.add(memo)
                    first = dict(((kname, fv) for kname, fv in sent_args.items() if key != kname))
                    held_526 = _hold('page_empty', arg=key, call=_json_line(first))
                    return held_526
                else:
                    return None
        return None
    else:
        return None

def _invented_page(ep, key, field_value) -> bool:
    value = field_value.strip()
    if not (not value or ' ' in value or key == _query_arg(ep)[0]):
        handed: set = set()
        typed: set = _terms(ep.policy.get('query')) | _terms(' '.join(ep.need_lines()))
        for element, sent_args in _search_history(ep):
            held_531 = _result_body(element)
            held_530 = _next_page_values(held_531, str, 0)
            handed.update(held_530)
            typed |= _terms(' '.join((str(fv) for kname, fv in sent_args.items() if key != kname and isinstance(fv, str))))
        if not value in handed:
            held_532 = bool(re.search('[0-9]', value)) or value in ep.seen_ids or (not _terms(value) & typed)
            return held_532
        else:
            return False
    else:
        return False

def _check_question_form(ep, batch, call):
    if not (batch['planned'] or batch['message'] != call['name'] or ep._procedures_due('order')):
        marks = sum((fv.count('?') for fv in (call['args'] or {}).values() if isinstance(fv, str)))
        if not 1 == marks:
            held_533 = SKIPPED + ': ' + (_note('ask_many', marks=marks) if marks else _note('ask_none'))
            return held_533
        else:
            return None
    else:
        return None
CALL_CHECKS = (_check_known_tool, _check_offered, _check_json, _check_schema, _check_finish_alone, _check_repeat_fault, _check_message_quota, _check_duplicate, _check_paging, _check_search_flood, _check_reach_early, _check_late_reads, _check_late_unfitted, _check_question_form, _check_stacking, _check_cart_add, _check_units, _check_release, _check_reread, _check_rerank, _check_cheapest, _check_best_probe, _check_fit, _check_reach)
_NEXT = '_next_phase'

def _batch_facts(ep, calls, planned) -> dict[str, Any]:
    offered = set((tk['function']['name'] for tk in ep.tools_offered()))
    roles = ep.roles
    held_534 = {'planned': planned, 'allowed': offered, 'finishing': {roles.get('order'), roles.get('decline')} - {None}, 'message': roles.get('message'), 'cart': roles.get('cart'), 'count': len(calls), 'queued_messages': 0}
    return held_534

def _first_objection(ep, batch, probe):
    for gate in CALL_CHECKS:
        note = gate(ep, batch, probe)
        if None is not note:
            return note
    return None

def _batch_screen(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    batch = bag['ctx'] = _batch_facts(ep, bag['calls'], bag['planned'])
    bag['replies'], bag['ready'] = ([], [])
    batch['ready'] = bag['ready']
    for proposed in bag['calls']:
        held_535 = proposed.get('function')
        if not held_535:
            held_535 = {}
        func = held_535
        probe = {'id': proposed['id'], 'name': func.get('name'), 'raw': func.get('arguments')}
        note = _first_objection(ep, batch, probe)
        if not None is not note:
            if batch['message'] == probe['name']:
                batch['queued_messages'] += 1
            bag['ready'].append((probe['id'], probe['name'], probe['args']))
        else:
            bag['replies'].append((probe['id'], note))
            continue
    return bag

def _hold_finish(bag: dict[str, Any], call_id: str | None, note: str | None) -> dict[str, Any]:
    if None is not note:
        bag['replies'].append((call_id, note))
    bag['ready'] = []
    bag[_NEXT] = 'held'
    return bag

def _finish_pick(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    queue = bag['ready']
    if not (queue and queue[0][1] in bag['ctx']['finishing']):
        bag[_NEXT] = 'pass'
    else:
        ident, tool, dispatched = queue[0]
        bag['finish'] = {'id': ident, 'name': tool, 'args': dispatched, 'kind': ep.closing_kind(tool, dispatched)}
    return bag

def _finish_unfit(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    closing = bag['finish']
    if not None is not closing['kind']:
        tool = closing['name']
        shared = any((job_name not in FINISH_JOBS and tool == bound for job_name, bound in ep.roles.items()))
        memo = 'unfit ' + tool + _json_line(closing['args'])
        if not (shared or memo in ep.held_arms):
            ep.held_arms.add(memo)
            held_536 = ep._finish_lines()
            forms = ' '.join(held_536)
            held_537 = _hold_finish(bag, closing['id'], _hold('unfit', forms=forms, tool=tool))
            return held_537
        else:
            bag[_NEXT] = 'pass'
            return bag
    else:
        return bag

def _finish_empty(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    if 'order' != bag['finish']['kind'] or not ep.roles.get('cart') or ep.cart_ids:
        return bag
    elif not ep.steps_left() <= 1:
        stamp = ep.cart_changed_at
        ep.empty_holds[stamp] = tries = ep.empty_holds.get(stamp, 0) + 1
        if not 3 < tries:
            held_539 = _hold_finish(bag, bag['finish']['id'], _hold('empty'))
            return held_539
        else:
            ep.finish_from_code()
            held_538 = _hold_finish(bag, None, None)
            return held_538
    else:
        return bag

def _finish_products(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    if not ('order' != bag['finish']['kind'] or ep.steps_left() <= 2):
        missing, lines = ep._uncarted_products()
        memo = 'items {}'.format(ep.cart_changed_at)
        if not (not missing or memo in ep.held_arms):
            ep.held_arms.add(memo)
            named = '; '.join(('%s (units to buy: %d)' % (el['product'], el['quantity']) for el in missing))
            held_540 = _hold_finish(bag, bag['finish']['id'], _hold('items', wanted=len(ep.basket_items()), lines=lines, missing=named))
            return held_540
        else:
            return bag
    else:
        return bag

def _finish_procedures(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    closing = bag['finish']
    kind = closing['kind']
    spare = ep.steps_left() - 2
    for step_proc in ep._procedures_due(kind):
        memo = (step_proc['quote'], kind, ep.cart_changed_at)
        if not (0 >= spare or memo in ep.proc_held):
            ep.proc_held.add(memo)
            if 'message' != step_proc['role']:
                held_541 = _hold_finish(bag, closing['id'], _hold('step_open', quote=step_proc['quote'], do=step_proc['do']))
                return held_541
            elif not (ep.messages_used >= ep._message_cap() or not ep.cart_ids or ep.queued_reply or (ep.steps_left() <= 4)):
                item_name = ' and '.join((ep._listing_name(idx) for idx in (ep.cart_ids if ep.basket_items() else ep.cart_ids[-1:])))
                ep.queued_reply = 'One check before I order: my pick is {item_name}. Is it all right to go ahead?'.format(item_name=item_name)
                held_542 = _hold_finish(bag, closing['id'], _hold('step_ask', do=step_proc['do']))
                return held_542
            else:
                continue
        else:
            continue
    bag[_NEXT] = kind
    return bag

def _finish_units(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    if not ep.basket_items():
        lines = list(((idx, ep.units_wanted()) for idx in ep.cart_ids[-1:]))
    else:
        lines = list(((idx, ep._units_for(idx)) for idx in ep.cart_ids))
    if not (not lines or ep.cart_changed_at > ep.cart_seen_at):
        for held, wanted in lines:
            shown = ep.units_in_cart.get(held)
            memo = 'count {} {}'.format(held, ep.cart_changed_at)
            if not (None is shown or None is wanted or wanted == shown or (ep.steps_left() <= 2) or (memo in ep.held_arms)):
                ep.held_arms.add(memo)
                held_543 = _hold_finish(bag, bag['finish']['id'], _hold('count', wanted=wanted, shown=shown, held=held))
                return held_543
            else:
                continue
        return bag
    else:
        return bag

def _keep_repriced(ep: Episode) -> bool:
    if not ep.refusals or ep.basket_items() or 1 != len(ep.cart_ids):
        return False
    elif not ((ep.plan or {}).get('priority') in ('cheapest', 'best') or ep.refused_at < ep.cart_changed_at):
        held = ep.cart_ids[0]
        held_544 = ep.listing_table().get(held)
        if not held_544:
            held_544 = {}
        facts = held_544
        if not (ep.mentioned.get(held, -1) <= ep.refused_at or facts.get('stock') is not True):
            if not ep._affordable(facts) or ep._passes_by_words(facts) is not True:
                return False
            found = _listing_records(ep.records)
            _, key, accepts = _field_slots(found)[0]
            prices = set((bag[key] for ident, bag in found if held == ident and key and accepts(bag.get(key))))
            held_545 = 1 < len(prices)
            return held_545
        else:
            return False
    else:
        return False

def _finish_review(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    if not (ep.order_checked or ep.steps_left() <= 2 or ep.seconds_left() <= WRAP_UP_SECONDS):
        ep.order_checked = True
        warning = ''
        for held in ep.cart_ids if ep.basket_items() else ep.cart_ids[-1:]:
            warning = ep.second_check(held, 'about to be bought')
            if warning:
                break
        if warning:
            held_546 = _hold_finish(bag, bag['finish']['id'], '{} {}'.format(HELD, warning))
            return held_546
        elif not _keep_repriced(ep):
            held_547 = _hold_finish(bag, bag['finish']['id'], ORDER_CHECKLIST)
            return held_547
        else:
            return bag
    else:
        return bag

def _finish_decline_again(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    closing = bag['finish']
    closing['arm'] = arm_key = 'decline ' + ep._state_tag()
    count = ep.decline_holds.get(arm_key, 0)
    if arm_key not in ep.held_arms or not ep.proven_fits:
        return bag
    elif not (1 > count and ep._room_for(2)):
        done = _order_instead(ep, bag['ctx'])
        if None is done:
            return bag
        held_549 = _hold_finish(bag, closing['id'], done)
        return held_549
    else:
        ep.decline_holds[arm_key] = count + 1
        held_548 = _hold_finish(bag, closing['id'], _hold('decline_again', ids=', '.join(ep.proven_fits)))
        return held_548

def _request_sections(ep) -> set[str]:
    tool = ep.roles.get('search')
    held_550 = (ep.param_schemas.get(tool) or {}).get('properties')
    if not held_550:
        held_550 = {}
    shape = held_550
    sent = set()
    for entry in ep.trace:
        for action in entry.get('actions') or [] if entry.get('kind') == 'env' else []:
            if action.get('name') == tool:
                sent |= set((str(fv).strip().lower() for kname, fv in (action.get('args') or {}).items() if isinstance(fv, str) and (shape.get(kname) or {}).get('enum')))
    product = ep._stems(ep.plan.get('product') or '')
    labels = set((rw['label'] for rw in ep.listing_table().values() if rw['opened'] and isinstance(rw['label'], str)))
    held_551 = set((lb for lb in labels if lb.strip().lower() in sent or ep._stems(lb) & product))
    return held_551

def _last_add_call(ep):
    tool = ep.roles.get('cart')
    held_552 = (ep.param_schemas.get(tool) or {}).get('properties')
    if not held_552:
        held_552 = {}
    props = held_552
    proposed = list((_reply_object((tk.get('function') or {}).get('arguments') or '') for msg in ep.messages if msg.get('role') == 'assistant' for tk in msg.get('tool_calls') or [] if (tk.get('function') or {}).get('name') == tool))
    for args in reversed(list((fv for fv in proposed if isinstance(fv, dict)))):
        if not (None is not ep.add_signature and ep._mode_signature(tool, args) != ep.add_signature):
            lines = set((listing for listing, _ in _line_args(props, args) if listing in ep.seen_ids))
            if 1 == len(lines):
                held_553 = (args, lines.pop())
                return held_553
        else:
            continue
    return None

def _swap_listing(branch, old, new, units):
    if isinstance(branch, list):
        held_554 = list((_swap_listing(fv, old, new, units) for fv in branch))
        return held_554
    elif isinstance(branch, dict):
        counts = list((kname for kname, fv in branch.items() if isinstance(fv, int) and (not isinstance(fv, bool))))
        held_555 = old in branch.values()
        if held_555:
            held_555 = 1 == len(counts)
        near = held_555
        held_556 = dict(((kname, units if near and counts[0] == kname else _swap_listing(fv, old, new, units)) for kname, fv in branch.items()))
        return held_556
    else:
        if old == branch:
            return new
        return branch

def _basket_proven(ep) -> bool:
    many, held = (ep.basket_items(), list(ep.cart_ids))
    indexes = list((ep._product_index(idx) for idx in held))
    if not (len(many) != len(held) or None in indexes or len(set(indexes)) != len(many) or ep._uncarted_products()[0]):
        row_map = ep.listing_table()
        if not any((idx not in ep.proven_fits or idx not in row_map or None is row_map[idx]['price'] for idx in held)):
            units = list((many[k]['quantity'] for k in indexes))
            if not any((ep.units_in_cart.get(idx) not in (None, count) for idx, count in zip(held, units))):
                cap = _number_or_none(ep.wishes.budget)
                held_557 = None is cap or sum((row_map[idx]['price'] * count for idx, count in zip(held, units))) <= cap
                return held_557
            else:
                return False
        else:
            return False
    else:
        return False

def _order_instead(ep, batch):
    tool, order = (ep.roles.get('cart'), ep.roles.get('order'))
    if not (tool and order) or ep._procedures_due('order'):
        return None
    elif not ep.basket_items():
        row_map, sections = (ep.listing_table(), _request_sections(ep))
        fits = list((idx for idx in ep.proven_fits if idx in row_map and row_map[idx]['label'] in sections and ep._affordable(row_map[idx]) and (ep._passes_by_words(row_map[idx]) is True)))
        held = list((idx for idx in ep.cart_ids if idx in fits))
        if held and 1 == len(ep.cart_ids):
            pick, add = (held[0], None)
        elif not (fits and (not ep.cart_ids)):
            return None
        else:
            pick = min(fits, key=lambda idx: (None is row_map[idx]['price'], row_map[idx]['price'] or 0, idx))
            template = _last_add_call(ep)
            if not None is template:
                held_560 = _swap_listing(template[0], template[1], pick, ep.units_wanted())
                add, issues = _fit_schema(held_560, ep.param_schemas[tool], 'args')
                if issues or _single_listing(ep, tool, add) != pick:
                    return None
            else:
                return None
        held_561 = ep._closing_args(order, _CODE_FINISH_TEXT['order'], 'order')
        closing, issues = _fit_schema(held_561, ep.param_schemas[order], 'args')
        if not (issues or ep.steps_left() < (2 if add else 1)):
            ep.trace.append({'kind': 'decline_swapped', 'pick': pick, 'fits': fits})
            said = []
            if None is not add:
                text, stop = _send_one(ep, batch, tool, add)
                said.append(text)
                if stop or ep.finished or pick not in ep.cart_ids:
                    held_562 = _note('swap_cut', pick=pick, said=' | '.join(said))
                    return held_562
            said.append(_send_one(ep, batch, order, closing)[0])
            held_563 = _note('swap', pick=pick, said=' | '.join(said))
            return held_563
        else:
            return None
    elif _basket_proven(ep):
        held_558 = ep._closing_args(order, _CODE_FINISH_TEXT['order'], 'order')
        closing, issues = _fit_schema(held_558, ep.param_schemas[order], 'args')
        if not (issues or ep.steps_left() < 1):
            ep.trace.append({'kind': 'decline_swapped', 'pick': list(ep.cart_ids), 'fits': list(ep.proven_fits)})
            held_559 = _note('swap', pick=', '.join(ep.cart_ids), said=_send_one(ep, batch, order, closing)[0])
            return held_559
        else:
            return None
    else:
        return None

def _single_listing(ep, tool, args):
    held_564 = (ep.param_schemas.get(tool) or {}).get('properties')
    if not held_564:
        held_564 = {}
    props = held_564
    lines = set((listing for listing, _ in _line_args(props, args) if listing in ep.seen_ids))
    if 1 == len(lines):
        held_565 = lines.pop()
        return held_565
    return None

def _finish_decline_first(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    closing = bag['finish']
    arm_key = closing['arm']
    if not (arm_key in ep.held_arms or not ep._room_for(2) or ep.seconds_left() <= WRAP_UP_SECONDS + 30):
        ep.held_arms.add(arm_key)
        first, ep.decline_checked = (not ep.decline_checked, True)
        held_566 = ep._evidence_against_decline()
        if not held_566:
            if first:
                held_566 = ep._decline_audit()
            else:
                held_566 = ''
        doubt = held_566
        if doubt:
            held_567 = _hold_finish(bag, closing['id'], '{} {}'.format(HELD, doubt))
            return held_567
        elif first:
            held_568 = _hold_finish(bag, closing['id'], DECLINE_CHECKLIST)
            return held_568
        else:
            return bag
    else:
        return bag

def _finish_scrub(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    closing = bag['finish']
    bag['ready'][0] = (closing['id'], closing['name'], ep._scrubbed_decline(closing['name'], closing['args']))
    return bag

def _batch_reserve(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    queue, finishing = (bag['ready'], bag['ctx']['finishing'])
    if not (not queue or not finishing or queue[0][1] in finishing):
        spare = ep.steps_left() - 1
        if not len(queue) <= spare:
            cut = max(0, spare)
            note = SKIPPED + ': ' + _note('reserve')
            bag['replies'].extend(((ident, note) for ident, _, _ in queue[cut:]))
            bag['ready'] = queue[:cut]
            return bag
        else:
            return bag
    else:
        return bag

def _send_one(ep, batch, tool, dispatched):
    action = {'name': tool, 'args': dispatched}
    number = ep.step + 1
    reply_obj = ep.store_call(action)
    ep.trace.append({'kind': 'env', 'step': number, 'actions': [action], 'reply': reply_obj})
    if not None is reply_obj:
        read = _run_result(ep, {'name': tool, 'args': dispatched, 'obs': reply_obj.get('observation'), 'closing': bool(reply_obj.get('finished')), 'ctx': batch})
        held_570 = (read['text'], read.get('halt') or '')
        return held_570
    else:
        held_569 = (_note('unreached'), _note('why_unreached'))
        return held_569

def _batch_send(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    blocked = ''
    for ident, tool, dispatched in bag['ready']:
        if blocked:
            text = _note('held_over', stop=blocked)
        elif not (ep.finished or ep.halted or ep.steps_left() <= 0):
            text, blocked = _send_one(ep, bag['ctx'], tool, dispatched)
        else:
            text = _note('closed')
        bag['replies'].append((ident, text))
    return bag

def _batch_order(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    place = {}
    for cnt, proposed in enumerate(bag['calls']):
        place[proposed['id']] = cnt
    bag['replies'].sort(key=lambda pair: place.get(pair[0], 0))
    return bag

def _result_books(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    tool, dispatched, batch = (bag['name'], bag['args'], bag['ctx'])
    bag['kind'] = ep.closing_kind(tool, dispatched)
    if batch['message'] == tool:
        ep.messages_used = ep.messages_used + 1
    if 'order' == bag['kind']:
        ep.order_attempts = ep.order_attempts + 1
    for role, bound in ep.roles.items():
        if tool == bound:
            if role not in FINISH_JOBS:
                ep.calls_by_role.setdefault(role, []).append(ep.step)
    if batch['cart'] == tool:
        ep.cart_used = True
        ep.order_checked = False
    held_571 = tool in batch['finishing']
    if not held_571:
        held_571 = tool in (batch['cart'], batch['message'])
    resets = held_571
    if resets:
        ep.recent = []
    else:
        ep.recent = ep.recent[-1:] + [tool + _json_line(dispatched)]
    ep.records.append(bag['obs'])
    ids: set[str] = set()
    _collect_ids(bag['obs'], ids)
    held_572 = dict.fromkeys(ids, ep.step)
    ep.mentioned.update(held_572)
    return bag

def _result_fault(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    tool, dispatched = (bag['name'], bag['args'])
    code, certain = ep._error_in(tool, bag['obs'])
    bag['fault'] = code
    if code:
        if certain:
            shape = ep._call_signature(tool, dispatched)
        else:
            shape = tool + _json_line(dispatched)
        tally = ep.faults.get((shape, code), 0) + 1
        ep.faults[shape, code] = tally
        if certain:
            bag['halt'] = _note('why_fault')
        if 2 <= tally:
            if dispatched:
                if tool not in bag['ctx']['finishing']:
                    ep.faulty.add(shape)
                    bag['fault_note'] = _note('fault_twice', code=code, tool=tool)
        return bag
    else:
        return bag

def _cart_view_ids(ep: Episode, seen) -> list[str]:
    out: list[str] = []
    for record_blob in _walk_dicts([seen]):
        for cell in record_blob.values():
            if isinstance(cell, str):
                if cell in ep.seen_ids:
                    if cell not in out:
                        out.append(cell)
    return out

def _result_cart(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    tool, dispatched = (bag['name'], bag['args'])
    if bag['ctx']['cart'] == tool:
        mode = ep._mode_signature(tool, dispatched)
        if None is ep.add_signature:
            ep.add_signature = mode
        adding = ep.add_signature == mode
        held_573 = ep.param_schemas[tool].get('properties')
        if not held_573:
            held_573 = {}
        props = held_573
        lines = _line_args(props, dispatched)
        if adding:
            if mode == _json_line({}):
                if len(lines) > len(list(_id_args(props, dispatched))):
                    named = list((field_value for field_value, _ in lines))
                    for gone in list((idx for idx in ep.cart_ids if idx not in named)):
                        ep.cart_ids.remove(gone)
                        ep.units_in_cart.pop(gone, None)
        for field_value, line_units in lines:
            ep.cart_state[field_value] = mode
            rest = list((idx for idx in ep.cart_ids if field_value != idx))
            if not adding:
                ep.cart_ids = rest
                ep.units_in_cart.pop(field_value, None)
            else:
                ep.cart_ids = rest + [field_value]
                ep.sent_units[field_value] = line_units if None is not line_units else ep._units_sent(tool, dispatched)
        ep.cart_changed_at = ep.step
    if tool == ep.roles.get('cart_view'):
        ep.last_cart_view = bag['obs']
        shown = _cart_view_ids(ep, bag['obs'])
        if shown:
            ep.cart_ids = list((idx for idx in ep.cart_ids if idx in shown)) + list((idx for idx in shown if idx not in ep.cart_ids))
        ep._read_cart_units(bag['obs'])
    return bag

def _result_kind(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    raw_obj = bag['obs']
    bag['lines'], bag['inner'] = text_lines, body = _peel_wrappers(raw_obj)
    held_574 = bool(bag['kind'])
    if held_574:
        held_574 = not bag['closing']
    if held_574:
        held_574 = not ep.finished
    unsettled = held_574
    held_575 = isinstance(raw_obj, dict)
    if held_575:
        held_575 = 'refusal' in raw_obj
    marked = held_575
    bag['refused'] = marked or (unsettled and (not _is_refusal_code(body)))
    bag['notes'] = []
    if isinstance(raw_obj, (dict, list)):
        _collect_ids(raw_obj, ep.seen_ids)
    if isinstance(body, dict):
        if body.get('error') == 'budget_exceeded':
            tool = bag['name']
            if tool == ep.roles.get('message') or (None is not ep.message_budget and tool not in ep.roles.values()):
                cap = ep.message_budget
                if None is cap:
                    ep.message_budget = ep.messages_used
                else:
                    ep.message_budget = min(cap, ep.messages_used)
            held_576 = _note('spent')
            bag['notes'].append(held_576)
    if bag['refused']:
        bag[_NEXT] = 'refused'
    elif unsettled:
        bag[_NEXT] = 'rejected'
    elif text_lines:
        if bag['name'] != ep.roles.get('message'):
            bag[_NEXT] = 'news'
    return bag

def _result_refused(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    kind = bag['kind']
    bag['halt'] = _note('why_refused')
    ep.refusals = ep.refusals + 1
    ep.refused_at = ep.step
    ep._new_cart_state()
    ep.order_checked = False
    held_577 = _note('decline_refused' if 'decline' == kind else 'order_refused')
    bag['notes'].append(held_577)
    if not 'order' != kind:
        row_map = ep.listing_table()
        if row_map:
            lower = _cheaper_after_refusal(ep, row_map)
        else:
            lower = []
        if lower:
            priced = ', '.join(('%s (%s)' % (idx, row_map[idx]['price']) for idx in lower))
            held_578 = _note('recheck', priced=priced)
            bag['notes'].append(held_578)
        return bag
    else:
        return bag

def _result_rejected(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    if not bag.get('halt'):
        bag['halt'] = _note('why_rejected')
    held_579 = _note('rejected')
    bag['notes'].append(held_579)
    return bag

def _result_news(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    heard: list[str] = []
    _collect_sentences(bag['lines'], heard)
    if heard:
        ep._new_cart_state()
        if not bag.get('halt'):
            bag['halt'] = _note('why_news')
        held_580 = _note('news', heard=' | '.join(heard[:4]))
        bag['notes'].append(held_580)
        return bag
    else:
        return bag

def _result_tail(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    notes = bag['notes']
    if bag.get('fault_note'):
        notes.append(bag['fault_note'])
    if bag['name'] == ep.roles.get('search'):
        if not bag.get('fault'):
            empty = ep._empty_search_note(bag['name'], bag['args'], bag['inner'])
            if empty:
                notes.append(empty)
    if bag['closing']:
        bag['halt'] = 'the episode is over.'
    bag['note'] = ' '.join(notes)
    return bag

def _append_note(bag: dict[str, Any], extra: str) -> None:
    if extra:
        bag['note'] = ' '.join((p for p in (bag['note'], extra) if p))

def _result_listen(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    body = bag['inner']
    heard: list[str] = []
    _collect_sentences(bag['lines'], heard)
    question = None
    if bag['ctx']['message'] == bag['name']:
        if heard:
            held_581 = ep.hear_shopper(heard, None)
            _append_note(bag, held_581)
            heard = []
        if isinstance(body, str):
            heard.append(body)
        elif isinstance(body, dict):
            if 'error' not in body:
                if not _is_refusal_code(body):
                    held_582 = list((fv for fv in body.values() if isinstance(fv, (str, list))))
                    _collect_sentences(held_582, heard)
        question = ' '.join((str(fv) for fv in bag['args'].values() if isinstance(fv, str)))
    elif bag['refused']:
        if isinstance(body, dict):
            heard += list((fv for fv in body.values() if isinstance(fv, str) and len(fv.split()) >= 4))
    if heard:
        held_583 = ep.hear_shopper(heard, question)
        _append_note(bag, held_583)
    return bag

def _result_text(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    parts = [_json_line(bag['obs'])]
    if bag['closing']:
        parts.append(' (the episode is over now)')
    if bag['note']:
        parts.append('\n[note from your checks] ' + bag['note'])
    bag['text'] = ''.join(parts)
    return bag

class Phase(enum.Enum):
    OPEN = 'open'
    TURNS = 'turns'
    SETTLE = 'settle'
    CLOSE = 'close'
    GUARD = 'guard'
    ANSWER = 'answer'
    PLANNER = 'planner'
    INFER = 'infer'
    CALLS = 'calls'
    COMMIT = 'commit'
    IDLE = 'idle'
    ACT = 'act'
    ASK_GATE = 'ask_gate'
    ASK_DRAFT = 'ask_draft'
    ASK_BUILD = 'ask_build'
    ASK_SEND = 'ask_send'
    SCREEN = 'screen'
    FIN_PICK = 'fin_pick'
    FIN_UNFIT = 'fin_unfit'
    FIN_EMPTY = 'fin_empty'
    FIN_STEPS = 'fin_steps'
    FIN_COUNT = 'fin_count'
    FIN_ITEMS = 'fin_items'
    FIN_REVIEW = 'fin_review'
    FIN_DECLINE_AGAIN = 'fin_decline_again'
    FIN_DECLINE_FIRST = 'fin_decline_first'
    FIN_SCRUB = 'fin_scrub'
    RESERVE = 'reserve'
    DISPATCH = 'dispatch'
    COLLATE = 'collate'
    O_COUNT = 'o_count'
    O_FAULT = 'o_fault'
    O_CART = 'o_cart'
    O_READ = 'o_read'
    O_REFUSED = 'o_refused'
    O_REJECTED = 'o_rejected'
    O_NEWS = 'o_news'
    O_TAIL = 'o_tail'
    O_LISTEN = 'o_listen'
    O_RENDER = 'o_render'
    END = 'end'
P = Phase
PHASE_STEPS = {P.OPEN: _flow_open, P.TURNS: _flow_turns, P.SETTLE: _flow_settle, P.CLOSE: _flow_close, P.GUARD: _flow_guard, P.ANSWER: _flow_reply, P.PLANNER: _flow_planner, P.INFER: _flow_infer, P.CALLS: _flow_calls, P.COMMIT: _flow_commit, P.IDLE: _flow_idle, P.ACT: _flow_act, P.ASK_GATE: _ask_gate, P.ASK_DRAFT: _ask_draft, P.ASK_BUILD: _ask_build, P.ASK_SEND: _ask_send, P.SCREEN: _batch_screen, P.FIN_PICK: _finish_pick, P.FIN_UNFIT: _finish_unfit, P.FIN_EMPTY: _finish_empty, P.FIN_STEPS: _finish_procedures, P.FIN_COUNT: _finish_units, P.FIN_ITEMS: _finish_products, P.FIN_REVIEW: _finish_review, P.FIN_DECLINE_AGAIN: _finish_decline_again, P.FIN_DECLINE_FIRST: _finish_decline_first, P.FIN_SCRUB: _finish_scrub, P.RESERVE: _batch_reserve, P.DISPATCH: _batch_send, P.COLLATE: _batch_order, P.O_COUNT: _result_books, P.O_FAULT: _result_fault, P.O_CART: _result_cart, P.O_READ: _result_kind, P.O_REFUSED: _result_refused, P.O_REJECTED: _result_rejected, P.O_NEWS: _result_news, P.O_TAIL: _result_tail, P.O_LISTEN: _result_listen, P.O_RENDER: _result_text}

def _phase_table() -> dict[tuple[Phase, str], Phase]:
    S = P
    text_lines = ((S.OPEN, S.TURNS, S.SETTLE, S.CLOSE, S.END), (S.GUARD, S.ANSWER, S.PLANNER, S.INFER, S.CALLS, S.COMMIT, S.IDLE, S.ACT, S.END), (S.ASK_GATE, S.ASK_DRAFT, S.ASK_BUILD, S.ASK_SEND, S.END), (S.SCREEN, S.FIN_PICK, S.FIN_UNFIT, S.FIN_EMPTY, S.FIN_STEPS), (S.FIN_COUNT, S.FIN_ITEMS, S.FIN_REVIEW, S.RESERVE, S.DISPATCH, S.COLLATE, S.END), (S.FIN_DECLINE_AGAIN, S.FIN_DECLINE_FIRST, S.FIN_SCRUB, S.RESERVE), (S.O_COUNT, S.O_FAULT, S.O_CART, S.O_READ, S.O_TAIL, S.O_LISTEN, S.O_RENDER, S.END), (S.O_REFUSED, S.O_TAIL), (S.O_REJECTED, S.O_TAIL), (S.O_NEWS, S.O_TAIL))
    row_index: dict[tuple[Phase, str], Phase] = {}
    for text_line in text_lines:
        for here, there in zip(text_line, text_line[1:]):
            row_index[here, 'done'] = there
    for stage in PHASE_STEPS:
        row_index[stage, 'exit'] = S.END
    for check in (S.FIN_UNFIT, S.FIN_EMPTY, S.FIN_STEPS, S.FIN_COUNT, S.FIN_ITEMS, S.FIN_REVIEW, S.FIN_DECLINE_AGAIN, S.FIN_DECLINE_FIRST):
        row_index[check, 'held'] = S.COLLATE
    row_index.update({(S.FIN_PICK, 'pass'): S.RESERVE, (S.FIN_UNFIT, 'pass'): S.RESERVE, (S.FIN_STEPS, 'order'): S.FIN_COUNT, (S.FIN_STEPS, 'decline'): S.FIN_DECLINE_AGAIN, (S.O_READ, 'refused'): S.O_REFUSED, (S.O_READ, 'rejected'): S.O_REJECTED, (S.O_READ, 'news'): S.O_NEWS})
    return row_index
PHASE_NEXT = _phase_table()

def _run_phases(start: Phase, ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    stage = start
    while P.END is not stage:
        bag = PHASE_STEPS[stage](ep, bag)
        result_info = bag.pop(_NEXT, 'done')
        if 'exit' in bag:
            result_info = 'exit'
        stage = PHASE_NEXT[stage, result_info]
    return bag

def _run_episode(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    held_584 = _run_phases(P.OPEN, ep, bag)
    return held_584

def _run_turn(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    held_585 = _run_phases(P.GUARD, ep, bag)
    return held_585

def _run_ask(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    held_586 = _run_phases(P.ASK_GATE, ep, bag)
    return held_586

def _run_batch(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    held_587 = _run_phases(P.SCREEN, ep, bag)
    return held_587

def _run_result(ep: Episode, bag: dict[str, Any]) -> dict[str, Any]:
    held_588 = _run_phases(P.O_COUNT, ep, bag)
    return held_588

def _log_crash(ep: Episode, exc: BaseException) -> None:
    text = '%s: %s' % (type(exc).__name__, exc)
    ep.trace.append({'kind': 'crash', 'error': text[:500]})

def _rescue(ep: Episode, exc: BaseException) -> list[dict[str, Any]]:
    _log_crash(ep, exc)
    try:
        if not (ep.finished or ep.halted):
            ep.finish_from_code()
    except Exception as again:
        _log_crash(ep, again)
    held_589 = ep.trace
    return held_589

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    ep = Episode(problem_data)
    try:
        held_590 = ep.run()
        return held_590
    except Exception as exc:
        held_591 = _rescue(ep, exc)
        return held_591
__all__ = ['agent_main']
