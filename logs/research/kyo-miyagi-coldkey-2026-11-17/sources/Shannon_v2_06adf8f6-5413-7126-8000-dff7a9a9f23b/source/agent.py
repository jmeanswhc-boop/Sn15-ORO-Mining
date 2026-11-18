"""A store errand worked by a language model, with the code keeping the books around it.

One hub (Visit) holds the parts; each part keeps its own state:
  Pacer      counted seconds and the store's step count
  Link       requests to the models (with stand-ins) and to the store (step contract)
  Intake     the first reading of the request and the tools, tool roles, the products asked for, the brief
  Notebook   requirements in force and what each shopper line changes about them
  Shelf      everything the store returned: records of one listing, the property menu, recurring labels, cheaper rows
  Basket     the cart as the cart calls left it
  Checker    second opinions before a cart add, before an order, before a decline
  Inquiry    planned questions to the shopper and answers owed to them
  Door       the screens a call passes before it may spend a step, and the finish review
  Clerk      sending what passed and booking what came back
  Closer     the finishing calls the code builds itself
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.agent.proxy_client import ProxyClient

PILOT = "anthropic/claude-haiku-5.5"         # works the turns
STANDBY = "openai/gpt-6-luna"                # takes over when the pilot gives nothing
READER = "anthropic/claude-sonnet-5.5"       # the first reading of the request
ASSESSOR = "anthropic/claude-sonnet-5.5"     # second opinions
LIGHT = "anthropic/claude-haiku-5.5"         # shopper lines, question plans, checks once spending runs high

SECONDS_FOR_TASK = 265.0
SOFT_SPEND = 0.06            # dollars one episode may spend before its checks move to the lighter model
WIND_DOWN = 70.0
HARD_STOP = 28.0
MODEL_CHARGE = 4.5           # counted seconds per model request (measured episodes average about 4.3 s a call)
STORE_CHARGE = 1.0           # counted seconds per store request
REPLY_CAP = 16000
TURN_LIMIT = 80

# ============================================================================ texts for the models
def keyed(intro: str, pairs: list[tuple[str, str]], outro: str = "") -> str:
    """A prompt that lists JSON keys, one line each: the intro, then '"key": meaning', then the closing note."""
    return intro + "\n" + "\n".join(f'"{k}": {d}' for k, d in pairs) + ("\n" + outro if outro else "")

PILOT_TEXT = """You act as the purchasing agent for one customer of a simulated shop; nothing happens except through the tools provided. Marking is binary. A finish earns the point only when all of this holds: the item ordered satisfies each requirement written in the request, each hidden need the customer discloses once asked, and each change they make during the episode; the unit count and the spend are right; a request for the cheapest (or the best) gets exactly that item; and every rule the request lays down was kept. If no item in the shop can satisfy all of it, the point goes to a decline done in the way the request prescribes.

1. STEPS AND TOOLS
- Steps are the scarce resource: each tool call takes one, even one that fails or errors, while your reasoning takes none.
- Independent calls can be bundled into one reply (several detail lookups at once, for example) and run in the order given. Orders and declines are never bundled with anything.
- A tool returning full details for several listings in one call beats opening them singly, because it saves steps.
- Shops differ in tool names, argument names and result layouts. Read every description and JSON schema closely and use exactly the argument names and allowed values defined there. The first results tell you which tool shows price, which shows stock and which shows the detailed properties.
- Find out soon which category label the shop gives the requested kind of product (it appears on listings). Of two similar neighbouring labels, choose the one that matches the request's wording read as a whole, and search and buy inside it.
- Search tools may be crippled: few rows or pages on show, ordering by id instead of relevance or price, all words required, filters valid in one mode only. Compare each reported total with the rows you can see, and keep tightening (more exact words, category or price filters, later pages) until nothing relevant is left unseen.

2. HIDDEN NEEDS AND QUESTIONS
- Some of the customer's needs were never written down; one comes out only when a message asks about exactly that property, and only the FIRST thing a message asks gets an answer. So each message covers ONE property, worded "Which <property> would you like the <product> to have?", with the property called what the listings call it, the product called by its plain noun from the request, and no example values. No instructions, no mention of lists, labels, positions or rules, no double questions.
- For a request naming several different products, every question must say which product it is about (an open product confuses the customer), and the questions are shared out across the products.
- A customer who says they did not follow a message, or asks for an explanation, simpler words or a rewording, should get neither an explanation nor a repeat: the property in question is most likely one they do not care about. Move on to another property (naming the product).
- "Don't mind", "no preference" or "anything is fine" frees that property. A preference or limit stated in an answer becomes a hard requirement from then on, in the customer's exact words (number limits included).
- The requirements are exactly what the request states plus what the customer reveals or changes. Derive none from the story (an age group, a gender, a size, the person who will use it): those details are not requirements and cannot make a listing fail. If one of them might matter, treat it as a possible hidden need and ask.
- With a planner active, questions are chosen and sent for you once you have opened the details of two or three listings of the requested kind (of each kind, if several different products are wanted), so open those early. Answers then appear in the conversation with their meaning; every revealed need is a hard requirement, and you search again combining it with the stated values.
- Messages to the customer carry questions only (and answers to a question the customer asked). Never use them to report findings, purchases or a decline; the finishing tool says that.
- When asking unaided: open the details of two or three listings first, then ask about what the story points at (who the item is for, how it should look, where or how it will be used), then about properties that vary among the candidates' listings. Never ask about anything the request settles, and never exceed the message budget.

3. THE REQUEST'S OWN PROCEDURE
- Instructions in the request about method (how many listings to view or compare before choosing, what to check, what to tell the customer, what a remark should contain, how to decline) are marked like requirements; carry each out literally and completely. Viewing or opening N listings means N fully fitting listings, each opened with the details tool.
- A question from the customer (to confirm a change, say) gets answered for you while messages remain; the change itself binds you.
- Mere wishes ("ideally", "would be nice", "if possible") are not requirements: they only break ties between items that already meet everything, and never outrank a cheapest rule.
- Whatever only the customer can settle (no price limit given, or two wishes that may clash where the customer must rank them) should be asked before you choose.

4. CHANGES MID-EPISODE
- The customer may revise things as you go: budget up or down, a requirement removed, replaced or added, a price rise, an item gone. One line may bundle several changes, and such lines can arrive in any tool result (an extra field of sentences) or with an order attempt. Each takes effect at once: re-test your pick and your backups, and under a cheapest or best rule, choose again among what is acceptable now.
- An order attempt may be REFUSED, with the reason in some field (a refusal, complaints, objections and so on) even when the inner result reads like success (accepted: true, placed: true, executed: true). A successful order ends the episode, so while the episode is still running nothing has been bought, whatever the result says. The cart is untouched and the episode continues. Read the refusal, apply each change it states, remove anything no longer acceptable from the cart (a sold-out item, for instance), add the next verified fit, recheck stock and price, and order again.
- Stock and prices also move silently, which is why both are rechecked right before every order attempt; once current price times quantity exceeds the current budget, the item no longer fits.

5. DECIDING WHETHER A LISTING FITS (the mark depends on this)
- A requirement holds only if the listing's own property for it gives the value. Keys appear in many spellings (spaces, dots, underscores, prefixes) and one property may sit under several of them; any key naming it is valid. Values match on meaning: one item of a comma list suffices, singular and plural are equivalent, close synonyms count, capitalisation is irrelevant. Sharing a word start is no match, nor is a wider or different class (girls is not women, plain unisex is not female).
- Fall back on title or description only when the listing lacks any key for the requirement; in that case an option or variant value of the listing that names the wanted value counts as well (it names the version on sale). When the listing's own property states something different, it fails regardless of what the title or description says.
- A value under a different but related property is no evidence (a length field says nothing about style; the lining's material says nothing about the material).
- Compare numbers as numbers after converting units; limits include their edge; a range qualifies only if all of it satisfies the limit.
- Respect exclusions ("leave out anything with X") and closed lists ("A or B and nothing else"). A closed list is matched by meaning too: a listing value that says one of the allowed options in other words, or sits inside an allowed bound, meets it; only a value that means something outside the list fails.
- A requirement may give its value by reference to another listing (one from a named seller with stated details, which may in turn defer to a listing it names). Find that listing, open its details and read all of it: the property's value, and any note there on how its value carries over (a pairing or conversion it states, or a further listing it names, which you then follow). The value that results is the requirement from then on.
- The requested kind is established by the listing's category; never demand that a property repeat the category's name. Only the requested kind will do: in a shop that labels listings by category, what you buy has to sit in the category whose label matches the request's wording as a whole (a request for "A or B" matches a category "A & B", not the narrower neighbour "B"). Availability counts too: when the request wants items on hand, only a listing shown as available qualifies, and price times quantity must not exceed the budget (equality is fine); prices are plain numbers in the shop's own unit, compared with the budget as they stand unless a record states another unit.
- Options or variants within a listing (sizes, colours, versions) that carry their own ids, prices or stock are separate candidates: the option's own value meets a requirement, its own price and stock count, and the cart line must identify the chosen option the way the cart tool's schema asks.
- A value listing several options that include the wanted one (a comma list) is a plain match, not mixed evidence. A composition is different: parts given with shares (a blend such as one material at one percentage and another at the rest) describe one mixed thing, so it does not meet a requirement for just one of those parts. Evidence is mixed only when the listing's own keys for the property give incompatible values; only then, and only when no cheapest or best rule decides, prefer a candidate with clean evidence.

6. CHEAPEST AND BEST
- For the cheapest, the order must be the lowest-priced item among ALL items meeting every requirement; finding one fit proves nothing. Use price filters where offered, lower the price ceiling until all remaining rows are visible, sort visible candidates by price yourself and verify from the cheapest up; the first full fit, with each cheaper candidate checked and rejected, is the answer, and the next fits are backups.
- For the best on a measure (most, largest, longest, highest and so on), compare ALL fits on that measure the same way and take the extreme; a listing with no value for the measure cannot win.

7. ENDING THE EPISODE
- Hold a ranked list of verified fits with at least one backup, since the first pick may sell out at checkout.
- Before each order attempt: the cart contains exactly the lines the request needs (normally one, the chosen item at the required quantity; one per item when several different items are requested) and nothing else (remove any other line); after your last cart change you reread the chosen item's stock and price, and it is in stock and within budget.
- Declining is rarely correct: most tasks have an item meeting every requirement, and a wrong decline scores 0. Decline only after a thorough search, and only when for each candidate you can point to a stated requirement, a revealed need, the budget or stock that its own records clearly fail. Doubt, a missing nice-to-have or a merely assumed requirement is no ground to decline; order the best verified fit. Decline exactly as the request's rule says; if it forbids offering substitutes, the remark must not name, describe or point to any listing (no ids, titles or prices).
- Pace yourself: make the first order attempt while at least seven steps remain, leaving room to handle a refusal. With 7 steps left, only cart, stock-check and finishing tools remain available. Commit once rather than swapping the cart back and forth; every cart change costs a step.
- Every reply needs at least one tool call. Keep its text short: current requirements, verified fits with price and stock, and the next move."""

INTAKE_TEXT = keyed("Below are a customer's request and the shop's tool list. Reply with a single compact JSON object (no indentation) and nothing else. Its keys:\n", [
    ("product", "what kind of product is wanted, in the request's wording (list each one when several different items are wanted)."),
    ("quantity", "units of a single listing to order (integer; 1 unless several are asked for). Something normally sold as one unit, like a pair of shoes or a set, counts as 1."),
    ("items", "only when several DIFFERENT products are wanted: an object per product, {\"product\": its short name in the request's words, \"quantity\": its units as an integer}; otherwise an empty list."),
    ("requirements", "a list of strings, one per firm requirement stated on the product, each as \"property: constraint\" keeping the customer's exact value, limit, exclusion or closed list. With several different products, prefix each with the short name of its product: \"<product> <property>: constraint\". Leave out price, stock, quantity, budgets and wishes, and do not turn the product kind or category into a requirement: words belonging to the category's name (a season, an occasion or an audience inside it) are already covered by the category and are no separate property requirement."),
    ("price_cap", "the maximum total price as a number, exactly as the request writes it in the request's own unit, never converted (\"53¢\" gives 53, \"$12\" gives 12); null when none is given."),
    ("currency", "the price unit the request uses (as written, e.g. \"¢\"), or null."),
    ("in_stock_only", "true when only items actually in stock or on hand are wanted."),
    ("priority", "\"cheapest\" only when the request explicitly wants the cheapest or lowest-priced fitting option; \"best\" only when it explicitly wants the best by some measure; else \"none\". Saying money is tight or every penny counts does not ask for the cheapest; it only explains the cap."),
    ("priority_detail", "for \"best\", the measure it is judged by; otherwise \"\"."),
    ("message_budget", "how many messages the customer lets the assistant send, as an integer, or null when not given."),
    ("step_budget", "how many moves or steps the request allows, as an integer, or null when not given."),
    ("rules", "a list of every procedural rule the request sets for the assistant (how many listings to view, what to check or tell, how to decline, what to avoid, what to include), copied verbatim. What the customer says they will do personally is story, not a rule."),
    ("soft_wants", "a list of things the customer would like without requiring them (ideally, preferably, would be nice, if possible); they never go under \"requirements\"."),
    ("ask_first", "a list of what only the customer can tell and should be asked before choosing: e.g. the budget when no price limit is given, which of two wishes weighs more when the request admits both may not be possible (both wishes then go in soft_wants, not requirements), or anything the request says to confirm with the customer. Empty if none.\nEvery sentence of the request must go somewhere: requirement, priority, soft want, rule, budget, hint or plain story. If unsure whether a sentence is a rule, list it under \"rules\"."),
    ("finish_values", "an object {\"order\": {...}, \"decline\": {...}}: for the tool that buys and the tool that declines, the value to give each of its arguments that has a fixed list of allowed values (a mode, route or choice) so that the call buys (order) or ends with no purchase (decline); use only values from that argument's own list, judging by the tool's description and schema; {} for a tool with no such argument."),
    ("decline_no_names", "true when the request's rule for declining forbids offering, naming or suggesting substitutes or alternatives; false when it asks for something to be named; null when there is no such rule."),
    ("roles", "an object giving, for each role, the exact name of the listed tool that fills it, or null when none does: \"search\", \"details\", \"view\", \"message\", \"cart\", \"cart_view\", \"order\", \"decline\". \"message\" sends a message to the customer; \"cart\" puts items into the cart or takes them out; \"order\" buys or checks out; \"decline\" ends the task with no purchase."),
], "")

LISTEN_TEXT = keyed("A shopping assistant needs the customer's latest words interpreted. Reply with a single JSON object and nothing else. Its keys:", [
    ("meaning", "a short sentence giving the customer's point."),
    ("needs", "a list of new firm requirements on the product, each \"property: constraint\" with the exact value, limit, exclusion or closed list given."),
    ("no_preference", "a list of properties the customer says they are indifferent to."),
    ("dropped", "a list of earlier requirements the customer has given up."),
    ("new_budget", "the new maximum total price as a number in the same unit as the request's budget, never converted to another unit, or null."),
    ("new_quantity", "if the customer changes the number of units: {\"product\": the product concerned (\"\" when only one kind is wanted), \"quantity\": the new integer}; otherwise null."),
    ("unavailable", "true if the customer reports an item can no longer be bought (sold out, gone, out of stock, button disabled), else false."),
    ("requirements_now", "every firm requirement on the product as things stand after all the customer has said, assembled from the requirements so far (below) and what the customer has explicitly said, each as \"property: constraint\". Add nothing from the story or from your own reading of the request: only values the request states as requirements or the customer states in an answer count. Every earlier requirement stays unless dropped or replaced; of a replaced value only the newest remains; each newly revealed need is added. Leave out price, stock, quantity and the product category."),
    ("unclear", "true if the customer says they did not follow or understand the assistant's message, or asks for it to be explained, simplified, reworded or repeated; otherwise false."),
    ("reply", "if the customer asks the assistant something or wants a confirmation, a short plain answer to send back that responds and states the change in substance (confirming the new value, for instance); otherwise \"\". Always \"\" when \"unclear\" is true: asking for the assistant's own question to be explained is not a question to answer."),
], "With several different products requested, every requirement in \"needs\" and \"requirements_now\" starts with the short name of its product, as \"<product> <property>: constraint\" (customers often say \"<property> of the <product kind>\").\n"
   "Report every change a line contains; there may be several. A replaced requirement counts twice: the old one as dropped, the new one as a need. If the customer ranks two wishes, the preferred one is a need. Any budget the customer gives is new_budget. A wish (\"ideally\", \"would be nice\") is not a need. Read the customer literally but generously: chatty, figurative or indirect wording that names a value means the value is firmly wanted, and a value is excluded only when the customer clearly says to avoid it. A number paired with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit. Wording that insists on a value, even when it sounds negative (refusing to move from it, refusing anything that falls short of it), means that value is wanted: write it as the plain value, with a comparison only when the customer states one. Mark a value as excluded only when the customer asks to stay away from it. Use the property name the customer uses, never a similar-sounding one. A new budget overrides the old one in either direction, even when presented as good news. If the customer only says they don't mind, have no preference or anything goes, list the property asked about under no_preference; such an answer touches only that property, never removes a stated requirement, a price limit or a rule, and changes nothing when the assistant's message asked about no property.")

ASSESS_TEXT = """Your verdict decides whether a shopping assistant may place its order; the task scores 1 only if the ordered item satisfies every requirement listed below. Base the verdict solely on the records shown, applying these rules:
- A requirement holds only if the item's own property for it gives the value. Keys come in many spellings (spaces, dots, underscores, prefixes) and one property may appear under several keys; any key naming the property is valid. Values match on meaning: one entry of a comma list suffices, singular/plural and close synonyms count, capitalisation is irrelevant. A lookalike word or a wider/different class is no match.
- Use the title or description, or an option or variant value naming the wanted value (it describes the version on sale), only when the item has no key at all for the requirement. When its own property says otherwise, it fails even if the title has the value.
- A composition (parts given with shares, such as a blend of materials) is one mixed value and does not meet a requirement for only one of its parts; a list of alternative options does match any of its members.
- A value under a different, related property is no evidence.
- Numbers: compare after unit conversion; limits include their edge; a range qualifies only if all of it meets the limit.
- Exclusions and closed lists must hold, judged by meaning: a value that says an allowed option in other words, or lies inside an allowed bound, meets the list; only a value meaning something outside it fails.
- A requirement whose value is set by reference to another listing is met only when the item's value equals the value that results from that listing: its own value, carried over the way any note in that listing says (a pairing or conversion it states, or a further listing it names). Records of listings the requirements may refer to are shown below when seen.
- Only the requested kind of product passes; where listings carry category labels, its category has to be the one whose label matches the request's wording as a whole, not a narrower or adjacent one.
- If the item's own property for a requirement gives another value, the requirement fails, whatever the title suggests.
- Price times quantity may not exceed the budget (equality passes). Prices in the records are numbers in the shop's own unit: compare them with the budget as they stand, without converting, unless a record states another unit. Records showing it out of stock mean it fails.
- A wish ("ideally", "would be nice") marked as not required is not a requirement.
- Judge only the listed requirements; never add one from the story (an age group, a gender, a size, the person who will use it). A listed demand that no listing attribute could ever state (a delivery date, running costs, plans, who will use it) is background: never fail an item over it.
- With several different products requested, judge the item only against the requirements of its own product kind: a requirement starting with another product's name does not apply, and the quantity is the one for its kind.
Where records conflict (on price or stock, for example), the newest record wins.
Reply with a single JSON object and nothing else: {"item": the item's id, "fits": true or false, "problems": [short strings, one per requirement that is unmet or unshown]}. Set fits=false only for a requirement the records clearly fail or that no record covers at all; never fail an item because no record shows its stock or price."""

INQUIRY_TEXT = """Your task is to plan what a shopping assistant asks the customer. The customer keeps some needs hidden; each is the value of one product property and surfaces only when a message asks about exactly that property. Only the first thing in a message gets answered, and anything else earns a polite "don't mind". Settle on QUESTION_COUNT different properties (fewer only if fewer useful ones exist), best prospects first:
0. Before all else, anything listed below under TO BE LEARNED FROM THE CUSTOMER FIRST that has not yet been asked, put plainly. For the budget, ask the most they want to spend. For a choice between two wishes, name both values with their properties, for example "If one listing can't have both, would you rather have the <value> <property> or the <value> <property>?";
1. Next, properties the listings' details below actually carry, favouring (a) ones most listings carry with values that vary between them, of any type (material, look, build or working, intended user, technical specifications), since a property with one value across every listing seen is a weak question; and (b) a property the request's story points to (who will use it, how it must look, where or how it will be used). Steer clear of logistics (warranty, condition, pack size, years) and of values that merely list everything on offer (all sizes or variants in one string), and never ask about brand, maker, seller or shop. A size, colour or other option that a listing, or one of its options, holds as a single value of its own is an ordinary property.
Skip any property the request settles (its stated requirements, price, stock, quantity); never ask again about a property already asked for the same product (a question the customer did not follow counts as asked, so move on, unless that question failed to name the product); never use two questions on near-duplicates (two kinds of style, two spellings of one property); and avoid properties no listing carries. Name properties the way the listings name them.
With several different products requested, each question is about ONE of them and the questions are shared out: prefer a product with fewer questions so far, and take its properties from listings of that product's kind (the menu below is grouped by the listings' labels where they have one).
Phrase each question as "Which <property> would you like the <product> to have?", with the property name exactly as the listings give it (spaces in place of underscores or dots) and the product by its plain noun from the request ("the lamp", "the blanket"; never with adjectives, materials, colours or other property values). No example values, no extra words, no second question. Something that must be learned first (rule 0) is asked plainly in one short sentence.
Reply with a single JSON object and nothing else: {"questions": [{"product": "...", "property": "...", "question": "..."}, ...]}."""

SWEEP_TEXT = """A shopping assistant is about to tell the customer that the shop has nothing suitable. Test the listing records below against the requirements by these rules: a requirement holds when the listing's own property for it gives the value (any key spelling, one entry of a comma list, singular/plural and close synonyms all count; a wider or different class does not); title or description, or an option value naming the wanted value, count only where the listing has no key for that property; prices are compared with the budget as they stand; numbers are compared after unit conversion and limits include their edge; closed lists are matched by meaning, not wording; the price must fit the budget and the item must not be shown out of stock; only the listed requirements count, never one assumed from the story, and a demand no listing attribute could ever state (a delivery date, plans, who will use it) is background that fails nothing; with several different products requested, each listing is tested only against its own product kind's requirements. A wrong decline scores 0, so list every listing that might fit and leave one out only when its own records clearly fail a listed requirement.
Reply with a single JSON object and nothing else: {"possible_fits": [ids of listings that seem to meet every requirement, best first, at most 5], "reason": "one sentence"}."""

ENDER_TEXT = """One of a shop's tools ends the task. Its schema follows, with the job or jobs it does: "order" (buy what the cart holds) and/or "decline" (end without buying). For each job, give the value each of its arguments with a fixed list of allowed values must take for the call to do that job, using only values from that argument's list. Reply with a single JSON object and nothing else: {"order": {argument: value}, "decline": {argument: value}} (leave out a job the tool does not do)."""

NO_ORDER_REMARK = "Unfortunately nothing in this shop meets all of your requirements, so I am placing no order."
CHECKOUT_REMARK = "Please check out the cart."
BUDGET_QUESTION = ("the most the customer wants to spend, as the request gives no price limit (skip it if the same thing "
                   "comes again below)")

_models = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_store = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

# ---- patterns the code reads results and names with
PROPS_KEY = re.compile(r"fact|propert|trait|spec|attribut|feature|detail|characteristic", re.I)
# a refusal saying the item can no longer be had (as against one reporting a changed price)
SOLD_OUT = re.compile("|".join([
    r"sold out|out of stock|unavailable|no longer|zero left|none left|all gone|is gone|it's gone|now gone",
    r"gone from stock|dried up|ran dry|run dry|cleared out|cleared from|bought the last|can't be ordered|cannot be ordered",
    r"grayed out|greyed out|hit zero|no units|none remaining|no .{0,12}remaining|been sold|been taken|snapped up",
    r"in stock any ?more|not in stock|isn't in stock|no stock|stock.{0,20}\bzero\b|last one"]), re.I)
# a line about the price (a changed price alone leaves an item acceptable)
PRICE_TALK = re.compile(r"price|cost|charg|priced|bumped|gone up|went up|come down|dropped|cents|dearer|cheaper", re.I)
# names of listing metadata rather than of product properties, kept out of the question menu: some words count anywhere
# in a name, others (which also end real property names) only as the whole name
NOT_A_PROPERTY = re.compile(
    r"\b(brand|maker|manufacturer|seller|vendor|merchant|headline|heading|sku|url|title|made by|sold by|"
    r"blurb|tagline|caption|teaser|slogan)\b|"
    r"^(make|shop|shop name|store|store name|outlet|company|category|department|shelf|description|"
    r"(overview|description|summary)( text)?|notes?|image|images|rating|reviews?|review count|link|"
    r"options?|variants?|id|name|price|amount|cost|stock|on hand|in stock)$", re.I)
# an error saying a tool's allowance is spent (not a bad argument or an over-budget cart)
SPENT = re.compile(r"budget_exceeded|quota|allowance|used up|no more\b.{0,40}\b(may|can|allowed|left)|limit reached|"
                   r"reached the limit|maximum (number|reached)|already used", re.I)
ID_KEY = re.compile(r"id|ref|sku|code|number|key|listing|product", re.I)
OPTION_KEY = re.compile(r"option|variant|choice|config|version|edition|sku|model", re.I)
KIND_KEY = re.compile(r"categor|department|^dept$|shelf|aisle|section|^kind$|product.?(type|family|group|kind)|"
                      r"item.?(type|kind)|offer.?kind|^family$|^group$", re.I)
SPEAKS_AS_SELF = re.compile(r"\b(I|I'm|I'd|I've|I'll|my|me|mine)\b")
WANTS_ANSWER = re.compile(r"\?|\b(confirm|let me know|tell me|acknowledge|reply|answer|get back to me)\b", re.I)
INFORM_RULE = re.compile(r"\b(tell|inform|let (me|them|the shopper) know|update|confirm|report|mention|explain|message)\b", re.I)
CHECKOUT_STEP = re.compile(r"final|confirm|complete|checkout|pay|place|submit|commit|token", re.I)
AWAIT_KEY = re.compile(r"token|pending|await|confirm|next.?step|checkout.?id|session", re.I)
NO_PURCHASE = re.compile(r"refus|declin|abandon|give.?up|no.?(buy|purchase|order)|reject|cancel|quit|skip", re.I)
ID_FORM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
TEXT_BREAKS = re.compile(r"[\s.,;:!?()\[\]{}<>\"'`]+")
CODE_FORM = re.compile(r"[A-Za-z0-9_\-]+")


# ============================================================================ value helpers
def clip_strings(value: Any, limit: int, depth: int = 0) -> Any:
    """A copy whose strings are cut to `limit` characters ('...' added); nothing deeper than eight levels."""
    if depth > 8:
        return None
    if isinstance(value, dict):
        return {k: clip_strings(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip_strings(v, limit, depth + 1) for v in value]
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def as_json(value: Any) -> str:
    """One-line JSON (non-ASCII kept), cut at REPLY_CAP characters."""
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= REPLY_CAP else text[:REPLY_CAP] + "...(cut)"


def _whole(text: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except ValueError:
        return False, None


def read_object(text: Any) -> dict[str, Any] | None:
    """The JSON object a model wrote: plain, fenced, between the outer braces, or (when it corrected itself) the last
    complete top-level object in the text."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[A-Za-z]*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    ok, value = _whole(text)
    if ok:
        return value if isinstance(value, dict) else None
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        ok, value = _whole(text[first: last + 1])
        if ok:
            return value if isinstance(value, dict) else None
    decoder = json.JSONDecoder()
    found = None
    at = text.find("{")
    while 0 <= at < len(text):
        try:
            value, stop = decoder.raw_decode(text, at)
        except ValueError:
            at = text.find("{", at + 1)
            continue
        # (only a top-level object counts; an inner piece of a broken outer object is no answer)
        if isinstance(value, dict) and text.count("{", 0, at) == text.count("}", 0, at):
            found = value
        at = text.find("{", stop)
    return found


def id_like(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_FORM.match(value)) and any(c.isdigit() for c in value)


def cart_line_text(args: dict[str, Any], ident: str) -> str:
    """The cart line naming this listing, as text, when the cart call passes its lines in a list."""
    for value in args.values():
        if isinstance(value, list):
            for line in value:
                if isinstance(line, dict) and ident in line.values():
                    return as_json(line)
    return ""


def holds_words(value: Any, depth: int = 0) -> bool:
    """Whether a result holds any word (more than ids, numbers and flags)."""
    if depth > 4:
        return False
    if isinstance(value, str):
        return any(c.isalpha() for c in value) and not id_like(value)
    if isinstance(value, dict):
        return any(holds_words(v, depth + 1) for v in value.values())
    if isinstance(value, list):
        return any(holds_words(v, depth + 1) for v in value)
    return False


def gather_sentences(value: Any, into: list[str], depth: int = 0) -> None:
    """Strings of four words or more: the value itself, or inside lists (ten per level, four levels down)."""
    if depth > 4:
        return
    if isinstance(value, list):
        for part in value[:10]:
            gather_sentences(part, into, depth + 1)
    elif isinstance(value, str) and len(value.split()) >= 4:
        into.append(value)


def gather_ids(value: Any, into: set[str], depth: int = 0) -> None:
    """Every id-shaped string held under a key containing 'id'."""
    if depth > 6:
        return
    if isinstance(value, list):
        for part in value:
            gather_ids(part, into, depth + 1)
    elif isinstance(value, dict):
        for key, part in value.items():
            if isinstance(part, str) and "id" in key.lower() and ID_FORM.match(part):
                into.add(part)
            else:
                gather_ids(part, into, depth + 1)


def awaits_next_step(observation: dict[str, Any]) -> bool:
    """An order result waiting for one more step (a token to confirm with) that carries no customer sentence."""
    lines: list[str] = []
    gather_sentences([v for v in observation.values() if isinstance(v, (list, str))], lines)
    keys = list(observation)
    for value in observation.values():
        if isinstance(value, dict):
            keys += list(value)
    if not any(AWAIT_KEY.search(str(k)) for k in keys):
        return False
    return not any(len(line.split()) >= 8 for line in lines)


def line_ids(args: dict[str, Any]) -> list[str] | None:
    """Listing ids of a cart call passing its lines as a list (None when there is no such list). A call naming its
    listing directly has no lines: a list beside it holds choices."""
    def named(key: Any, value: Any) -> bool:
        return id_like(value) and bool(ID_KEY.search(str(key)))

    if any(named(k, v) for k, v in args.items()):
        return None
    found = []
    for value in args.values():
        if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            for line in value:
                # one id per line: the first id-like value (normally the listing; an option id after it rides along)
                ident = next((v for k, v in line.items() if named(k, v)), None)
                if ident:
                    found.append(ident)
    return found or None


def enum_part(args: dict[str, Any], fields: dict[str, Any]) -> str:
    """The choice-valued arguments of a call, as text: tells an adding cart call from a removing one."""
    return as_json({k: v for k, v in args.items() if isinstance(fields.get(k), dict) and "enum" in fields[k]})


def free_token(args: dict[str, Any], fields: dict[str, Any]) -> str | None:
    """The first free one-token string argument (a listing id, normally)."""
    for key, value in args.items():
        if isinstance(value, str) and "enum" not in (fields.get(key) or {}) and value.strip() and " " not in value.strip():
            return value
    return None


def joined_strings(args: dict[str, Any]) -> str:
    return " ".join(str(v) for v in args.values() if isinstance(v, str))


def is_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ============================================================================ schema checking
def _kind_holds(value: Any, kind: str) -> bool:
    tests = {"string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
             "number": is_count, "boolean": lambda v: isinstance(v, bool), "object": lambda v: isinstance(v, dict),
             "array": lambda v: isinstance(v, list), "null": lambda v: v is None}
    test = tests.get(kind)
    return True if test is None else test(value)


def _kinds(schema: dict[str, Any]) -> Any:
    kinds = schema.get("type")
    return [kinds] if isinstance(kinds, str) else kinds


def convert(value: Any, schema: dict[str, Any]) -> Any:
    """The value turned into a declared type where nothing is lost (2.0 -> 2, '3' -> 3, 'true' -> True)."""
    kinds = _kinds(schema)
    if not kinds:
        return value
    if any(_kind_holds(value, k) for k in kinds):
        return int(value) if "integer" in kinds and isinstance(value, float) and value.is_integer() else value
    for kind in kinds:
        try:
            if kind == "integer":
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                if isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
            if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                figure = float(value)
                return int(figure) if figure.is_integer() else figure
            if kind == "string" and is_count(value):
                return str(value)
            if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
        except (TypeError, ValueError):
            continue
    return value


def validate(value: Any, schema: Any, path: str) -> tuple[Any, list[str]]:
    """The value fitted to a JSON schema where that is safe, and every way it still breaks the schema."""
    if not isinstance(schema, dict):
        return value, []
    faults: list[str] = []
    value = convert(value, schema)
    kinds = _kinds(schema)
    if kinds and not any(_kind_holds(value, k) for k in kinds):
        return value, [f"{path} should be {'/'.join(kinds)}"]
    if "enum" in schema and value not in schema["enum"]:
        if isinstance(value, str):
            folded = value.strip().lower()
            value = next((o for o in schema["enum"] if isinstance(o, str) and o.lower() == folded), value)
        if value not in schema["enum"]:
            faults.append(f"{path} takes only {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        faults.append(f"{path} has to be {schema['const']!r}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            faults.append(f"{path} needs {schema['minLength']} characters or more")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
    if is_count(value):
        faults += _range_faults(value, schema, path)
    if isinstance(value, list):
        value = _list_fit(value, schema, path, faults)
    if isinstance(value, dict):
        value = _object_fit(value, schema, path, faults)
    return value, faults


def _range_faults(value: float, schema: dict[str, Any], path: str) -> list[str]:
    out = []
    if "minimum" in schema and value < schema["minimum"]:
        out.append(f"{path} is under the minimum {schema['minimum']}")
    if "maximum" in schema and value > schema["maximum"]:
        out.append(f"{path} is over the maximum {schema['maximum']}")
    low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
        out.append(f"{path} has to exceed {low}")
    if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
        out.append(f"{path} has to stay below {high}")
    return out


def _list_fit(value: list, schema: dict[str, Any], path: str, faults: list[str]) -> list:
    if "minItems" in schema and len(value) < schema["minItems"]:
        faults.append(f"{path} needs {schema['minItems']} entries or more")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        faults.append(f"{path} takes {schema['maxItems']} entries at most")
    if not isinstance(schema.get("items"), dict):
        return value
    fitted = []
    for n, part in enumerate(value):
        part, more = validate(part, schema["items"], f"{path}[{n}]")
        fitted.append(part)
        faults += more
    return fitted


def _object_fit(value: dict, schema: dict[str, Any], path: str, faults: list[str]) -> dict:
    fields = schema.get("properties") or {}
    faults += [f"{path}.{name} is missing" for name in schema.get("required") or [] if name not in value]
    if schema.get("additionalProperties") is False:
        faults += [f"{path}.{name} is not an accepted argument" for name in list(value) if name not in fields]
    fitted = {}
    for name, part in value.items():
        if name in fields:
            part, more = validate(part, fields[name], f"{path}.{name}")
            faults += more
        fitted[name] = part
    return fitted


# ============================================================================ counted time and steps
class Pacer:
    def __init__(self, cap: int) -> None:
        self.used = 0.0
        self.step = 0
        self.cap = cap

    def time_left(self) -> float:
        return SECONDS_FOR_TASK - self.used

    def steps_left(self) -> int:
        return self.cap - self.step


# ============================================================================ models and store
class Link:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.cost = 0.0

    @staticmethod
    def _marked(model: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """For a caching model with a longer history: the first and the last message marked cacheable."""
        if not (model.startswith("anthropic/") and len(messages) > 2):
            return messages
        marked = [dict(m) for m in messages]
        for n in (0, len(marked) - 1):
            content = marked[n].get("content")
            if isinstance(content, str) and content:
                marked[n]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return marked

    def talk(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, chain: list[str],
             max_tokens: int) -> dict[str, Any] | None:
        """The first model in the chain whose answer holds text or tool calls; None when none does in time."""
        pace = self.v.pace
        for model in chain:
            left = pace.time_left()
            if left < HARD_STOP - 10:
                return None
            _models.timeout = int(max(8, min(75, left - HARD_STOP + 15)))
            body: dict[str, Any] = {"model": model, "messages": self._marked(model, messages), "temperature": 0,
                                    "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            pace.used += MODEL_CHARGE
            outcome = _models.post_verbose("/inference/chat/completions", body)
            data = outcome.data
            if isinstance(data, dict) and data.get("choices"):
                price = (data.get("usage") or {}).get("cost")
                if isinstance(price, (int, float)):
                    self.cost += float(price)
                said = data["choices"][0].get("message") or {}
                if said.get("content") or said.get("tool_calls"):
                    return said
            self.v.trace.append({"kind": "inference_failure", "model": model,
                                 "error": str(outcome.error)[:300] if outcome.error else "empty"})
        return None

    def opinion(self, prompt: str, max_tokens: int = 700, chain: list[str] | None = None) -> dict[str, Any] | None:
        chain = chain or [ASSESSOR, PILOT]
        if self.cost > SOFT_SPEND and chain[0] == ASSESSOR:
            chain = [LIGHT, ASSESSOR]       # (a costly episode keeps its checks, on the lighter model)
        said = self.talk([{"role": "user", "content": prompt}], None, chain, max_tokens)
        return read_object((said or {}).get("content") or "")

    def act(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        """One store request, one action or several; the step count follows the store's."""
        v, pace = self.v, self.v.pace
        payload: dict[str, Any] = dict(v.binding)
        payload["step"] = pace.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            left = pace.time_left()
            if left < 2:
                return None
            _store.timeout = int(max(3, min(95, left - 1)))
            pace.used += STORE_CHARGE
            outcome = _store.post_verbose("/environment/call", payload)
            if isinstance(outcome.data, dict):
                data = outcome.data
                pace.step = data["step"] if isinstance(data.get("step"), int) else pace.step + len(actions)
                if data.get("finished"):
                    v.finished = True
                return data
            error = outcome.error or {}
            status, said = error.get("status"), str(error.get("body") or "")
            v.trace.append({"kind": "env_failure", "status": status, "body": said[:300]})
            if status == 402:
                v.dead = True
                return None
            if status == 409:
                if "finished" in said:
                    v.finished = True
                    return None
                expected = re.search(r"expected (\d+)", said)
                if expected is None:
                    return None
                pace.step = int(expected.group(1)) - 1
                payload["step"] = pace.step + 1
                continue
            if status in (403, 422, 500, 503, 504):
                return None
        return None


# ============================================================================ the first reading
class Intake:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit

    def read(self) -> None:
        v = self.v
        tool_lines = [f"- {fn['name']}: {fn.get('description', '')} | arguments: {as_json(fn.get('parameters') or {})}"
                      for fn in (t["function"] for t in v.tools)]
        fixed = INTAKE_TEXT + "\n\nTOOLS:\n" + "\n".join(tool_lines)
        varying = "REQUEST:\n" + v.request()
        if v.policy.get("opening") is not None:
            varying += "\n\nOPENING OBSERVATION:\n" + as_json(v.policy.get("opening"))
        content = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]
        said = v.link.talk([{"role": "user", "content": content}], None, [READER, STANDBY], 1500)
        plan = read_object((said or {}).get("content") or "") or {}
        if not isinstance(plan.get("roles"), dict) and v.pace.time_left() > 200:
            # (every later step leans on this reading: one more try)
            said = v.link.talk([{"role": "user", "content": content}], None, [READER, PILOT], 2000)
            plan = read_object((said or {}).get("content") or "") or plan
        v.plan = plan
        offered = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        v.roles = {role: name for role, name in offered.items() if isinstance(name, str) and name in v.schemas}
        if not v.roles:
            # (no reading: every guard keyed to a role would be off, so the roles are guessed from the tool names)
            v.roles = self.guess_roles()
            v.trace.append({"kind": "guessed_roles", "roles": v.roles})
        allowance = plan.get("message_budget")
        if is_count(allowance) and allowance >= 0:
            v.inquiry.budget = int(allowance)
        v.book.stated = [str(r) for r in plan.get("requirements") or [] if r]
        v.book.budget = plan.get("price_cap")
        if plan and v.book.budget is None and v.roles.get("message"):
            queued = [str(a) for a in plan.get("ask_first") or [] if a]
            plan["ask_first"] = [BUDGET_QUESTION] + queued
        moves = plan.get("step_budget")
        if is_count(moves) and 0 < moves < v.pace.cap:
            v.pace.cap = int(moves)

    def guess_roles(self) -> dict[str, str]:
        v = self.v

        def name_words(name: str) -> set[str]:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
            return {w.lower() for w in re.split(r"[^A-Za-z0-9]+", spaced) if w}

        def single_text_arg(name: str) -> bool:
            fields = (v.schemas.get(name) or {}).get("properties") or {}
            return len([k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]) == 1

        holders = {"cart", "basket", "bag", "trolley", "stash", "hold", "line", "lines", "keep", "selection"}

        def is_decline(w: set[str], n: str) -> bool:
            return (bool(w & {"decline", "refuse", "reject", "abandon", "dismiss", "quit", "giveup"})
                    or any(x.startswith(("declin", "refus")) for x in w) or {"give", "up"} <= w or "nothing" in w)

        def is_cart_view(w: set[str], n: str) -> bool:
            return bool(w & (holders | {"items", "order"})) and bool(w & {"view", "show", "list", "review", "status",
                                                                       "current", "my", "see", "contents", "summary"})

        def is_cart(w: set[str], n: str) -> bool:
            if w & holders:
                return True
            return bool(w & {"add", "put", "set", "remove", "rewrite", "update"}) and bool(w & (holders | {"items", "order", "all"}))

        def is_order(w: set[str], n: str) -> bool:
            return bool(w & {"checkout", "purchase", "order", "buy", "commit", "confirm", "place", "settle", "pay",
                             "execute", "finalize", "submit"})

        def is_message(w: set[str], n: str) -> bool:
            return bool(w & {"message", "shopper", "buyer", "customer", "client", "ask", "chat", "note", "contact",
                             "write", "talk"}) and single_text_arg(n)

        def is_details(w: set[str], n: str) -> bool:
            return bool(w & {"detail", "details", "inspect", "open", "read", "view", "fetch", "get", "datasheet", "info",
                             "spec", "specs", "describe", "examine"})

        def is_search(w: set[str], n: str) -> bool:
            return bool(w & {"search", "find", "browse", "locate", "hunt", "query", "lookup", "catalog", "seek", "list"})

        order_of_tests = [("decline", is_decline), ("cart_view", is_cart_view), ("cart", is_cart), ("order", is_order),
                          ("message", is_message), ("details", is_details), ("search", is_search)]
        guessed: dict[str, str] = {}
        taken: set[str] = set()
        for role, test in order_of_tests:
            pick = next((t["function"]["name"] for t in v.tools
                         if t["function"]["name"] not in taken and test(name_words(t["function"]["name"]), t["function"]["name"])), None)
            if pick is not None:
                guessed[role] = pick
                taken.add(pick)
        return guessed

    def products(self) -> list[tuple[str, int]]:
        """The different products of a request asking for several (name, units); empty for one product."""
        raw = self.v.plan.get("items")
        found = []
        for entry in raw if isinstance(raw, list) else []:
            if isinstance(entry, str) and entry.strip():
                entry = {"product": entry, "quantity": 1}
            if not (isinstance(entry, dict) and str(entry.get("product") or "").strip()):
                continue
            units = entry.get("quantity")
            if isinstance(units, str) and units.strip().isdigit():
                units = int(units.strip())
            units = int(units) if is_count(units) and units >= 1 else 1
            found.append((str(entry["product"]).strip(), units))
        return found if len(found) >= 2 else []

    def brief(self) -> str:
        p, v = self.v.plan, self.v
        if not p:
            return ""
        rows = ["How the request was first read (hold it against the request itself):"]
        if p.get("product"):
            rows.append(f"- product: {p.get('product')}")
        many = self.products()
        if many:
            rows += [f"- to buy: {name}, quantity {units} (its own cart line)" for name, units in many]
        else:
            rows.append(f"- quantity: {p.get('quantity') or 1}")
        rows += [f"- stated requirement: {r}" for r in p.get("requirements") or []]
        if p.get("price_cap") is not None:
            rows.append(f"- total price cap: {p.get('price_cap')} {p.get('currency') or ''}".rstrip())
        if p.get("in_stock_only"):
            rows.append("- in-stock / on-hand items only")
        if p.get("priority") in ("cheapest", "best"):
            measure = f" ({p.get('priority_detail')})" if p.get("priority_detail") else ""
            rows.append(f"- take the {p.get('priority')}{measure} of all fitting items")
        else:
            rows.append("- no cheapest-or-best rule found: any item honouring every requirement within the budget is "
                        "right, so spend no steps chasing the lowest price (unless the request itself says otherwise)")
        rows += [f"- nice to have, NOT required (tie-breaker only): {w}" for w in p.get("soft_wants") or []]
        rows += [f"- to learn from the customer first: {a}" for a in p.get("ask_first") or []]
        rows += [f"- rule: {r}" for r in p.get("rules") or []]
        if v.inquiry.budget is not None:
            rows.append(f"- messages to the customer allowed: {v.inquiry.budget}")
        rows.append(f"- steps allowed: {v.pace.cap}")
        hinted = [h for h in p.get("hint_properties") or [] if isinstance(h, str)]
        if hinted:
            rows.append("- properties the story may point to (ask these first, one per message): " + ", ".join(hinted))
        return "\n".join(rows)


# ============================================================================ requirements in force
class Notebook:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget: Any = None
        self.heard: set[str] = set()
        self.whole: list[str] | None = None
        self.version = 0
        self.exchanges: list[str] = []
        self.owed = ""
        self.last_unclear = False

    def summary(self) -> str:
        if self.whole is not None:
            entries = list(self.whole)
        else:
            entries = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            entries += [f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped]
        text = "; ".join(entries) if entries else "see the request"
        if self.budget is not None:
            text += f"; total price at most {self.budget} {self.v.plan.get('currency') or ''}".rstrip()
        many = self.v.intake.products()
        if many:
            text += "; items and quantities: " + ", ".join(f"{name} x{units}" for name, units in many)
        else:
            text += f"; quantity {self.v.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def hear(self, lines: list[str], question: str | None) -> str:
        v = self.v
        fresh = [line for line in lines if line not in self.heard]
        if not fresh:
            return ""
        self.heard.update(fresh)
        if v.pace.time_left() < WIND_DOWN:
            return ""
        prompt = LISTEN_TEXT + "\n\nTHE CUSTOMER'S REQUEST (background):\n" + v.request()
        prompt += "\n\nREQUIREMENTS UP TO NOW:\n" + self.summary()
        if self.exchanges:
            prompt += "\n\nEARLIER EXCHANGES WITH THE CUSTOMER (oldest first):\n" + "\n".join(self.exchanges[-12:])
        if question:
            prompt += "\n\nTHE ASSISTANT'S QUESTION:\n" + question
        prompt += "\n\nTHE SHOPPER HAS JUST SAID:\n" + "\n".join(fresh)
        self.exchanges.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        reading = v.link.opinion(prompt, 1500, [LIGHT, ASSESSOR])
        if reading and self._strays(reading, fresh, question):
            # (a need named after a property the customer never mentioned is likely misread: one read on the stronger model)
            second = v.link.opinion(prompt, 1500, [ASSESSOR, LIGHT])
            if second:
                reading = second
        return self._take(reading, fresh, question) if reading else ""

    @staticmethod
    def _strays(reading: dict[str, Any], fresh: list[str], question: str | None) -> bool:
        """A new need whose property words are not all in the customer's lines (or the question)."""
        said = set(re.findall(r"[a-z0-9]+", (" ".join(fresh) + " " + (question or "")).lower()))
        for need in reading.get("needs") or []:
            if not isinstance(need, str) or ":" not in need:
                continue
            words = [w for w in re.findall(r"[a-z0-9]+", need.split(":", 1)[0].lower()) if len(w) > 2]
            if words and not all(w in said or w.rstrip("s") in said for w in words):
                return True
        return False

    def _take(self, reading: dict[str, Any], fresh: list[str], question: str | None) -> str:
        v = self.v
        changes: list[str] = []
        rebuilt = reading.get("requirements_now")
        if isinstance(rebuilt, list) and all(isinstance(r, str) for r in rebuilt):
            self.whole = [r for r in rebuilt if r.strip()]
        for need in reading.get("needs") or []:
            if isinstance(need, str) and need and need not in self.revealed:
                self.revealed.append(need)
                changes.append(f"new requirement: {need}")
        for prop in reading.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in self.free:
                self.free.append(prop)
        for gone in reading.get("dropped") or []:
            if isinstance(gone, str) and gone and gone not in self.dropped:
                self.dropped.append(gone)
                changes.append(f"dropped: {gone}")
        cap = reading.get("new_budget")
        if is_count(cap) and cap > 0 and cap != self.budget:
            self.budget = cap
            changes.append(f"new budget: {cap}")
        changes += self._units_change(reading.get("new_quantity"))
        if reading.get("unavailable"):
            changes.append("an item is no longer for sale; remove it from the cart and do not order it")
        if changes:
            self.version += 1
        answer = str(reading.get("reply") or "").strip()
        unclear = reading.get("unclear") is True and question is not None
        self.last_unclear = unclear
        if unclear and not self._names_a_product(question):
            # (with several products a question not saying which one is ambiguous: the property may still matter)
            changes.append("the customer lost track because the question named no product: if a message is left, ask "
                           "about the same property again and say which product")
        elif unclear:
            # (explaining our own question rarely helps: the asked property is most likely not a need)
            changes.append("the customer lost track of that question: neither explain nor repeat it; that property is "
                           "probably of no concern to them, so use any remaining message on a different one (with the product named)")
        elif answer and v.roles.get("message") and WANTS_ANSWER.search(" ".join(fresh)):
            v.book.owed = answer        # (only a line asking something is owed an answer)
        tail = " (" + "; ".join(changes) + ")" if changes else ""
        return "Meaning: " + str(reading.get("meaning") or "") + tail

    def _names_a_product(self, question: str | None) -> bool:
        many = self.v.intake.products()
        if not many:
            return True
        asked = {w.rstrip("s") for w in str(question).lower().replace("?", " ").split()}
        filler = ("and", "the", "for", "set", "with", "pair", "pack", "of")
        return any(w.rstrip("s") in asked for product, _ in many for w in product.lower().split()
                   if len(w) >= 3 and w not in filler)

    def _units_change(self, change: Any) -> list[str]:
        v = self.v
        if isinstance(change, int) and not isinstance(change, bool):
            change = {"product": "", "quantity": change}
        if not isinstance(change, dict):
            return []
        wanted = change.get("quantity")
        if not (isinstance(wanted, int) and not isinstance(wanted, bool) and wanted >= 1):
            return []
        product = str(change.get("product") or "").strip().lower()
        if not v.intake.products():
            v.plan["quantity"] = wanted         # (one kind of product: the plan's quantity counts)
            return [f"new quantity: {wanted}"]
        entries = v.plan.get("items") if isinstance(v.plan.get("items"), list) else []
        target = next((e for e in entries if isinstance(e, dict) and product
                       and (product in str(e.get("product") or "").lower() or str(e.get("product") or "").lower() in product)), None)
        if target is None:
            return []
        target["quantity"] = wanted
        return [f"new quantity: {target.get('product')} x{wanted}"]


# ============================================================================ what the store returned
class Shelf:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.records: list[Any] = []
        self.ids: set[str] = set()
        self.opened: set[str] = set()
        self.read_at: dict[str, int] = {}      # listing id -> step of the latest details / view / cart-view result naming it

    def records_of(self, item: str) -> list[str]:
        """The latest three distinct records naming the item; an option inside a listing comes with its listing."""
        hits: list[str] = []

        def scan(value: Any, depth: int, holder: Any = None, inside_option: bool = False) -> None:
            if depth > 6 or len(hits) >= 8:
                return
            if isinstance(value, list):
                for part in value:
                    scan(part, depth + 1, holder, inside_option)
                return
            if not isinstance(value, dict):
                return
            if any(isinstance(x, str) and x == item for x in value.values()):
                shown = holder if holder is not None and inside_option and holder is not value else value
                text = as_json(clip_strings(shown, 260))[:3000]
                if text not in hits:
                    hits.append(text)
                return
            if any(isinstance(x, str) and ID_KEY.search(str(k)) and id_like(x) for k, x in value.items()):
                holder, inside_option = value, False
            for k, x in value.items():
                scan(x, depth + 1, holder, inside_option or bool(OPTION_KEY.search(str(k))))

        for record in self.records:
            scan(record, 0)
        return hits[-3:]

    def menu(self) -> tuple[str, int]:
        """The properties the listings' details carry (per kind label when several products are requested), and how many
        sets of facts were seen."""
        counts: dict[str, dict[str, int]] = {}
        values: dict[str, dict[str, set[str]]] = {}
        examples: dict[str, dict[str, list[str]]] = {}
        tables: dict[str, int] = {}
        sets = 0

        def short(x: Any) -> bool:
            return isinstance(x, str) and 0 < len(x) <= 70

        def visit(value: Any, depth: int, key: str = "", group: str = "") -> None:
            nonlocal sets
            if depth > 6:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1, key, group)
                return
            if not isinstance(value, dict):
                return
            label = next((x.strip().lower() for k, x in value.items() if short(x) and KIND_KEY.search(str(k))), "")
            group = label or group
            strings = [(k, x) for k, x in value.items() if short(x)]
            nested = any(isinstance(x, dict) and PROPS_KEY.search(str(k)) for k, x in value.items())
            table_here = len(strings) >= 2 and bool(PROPS_KEY.search(key) or OPTION_KEY.search(key))
            if table_here or (len(strings) >= 5 and not nested):
                taken = 0
                for prop_key, item in strings:
                    if re.search(r"\d", item) and CODE_FORM.fullmatch(item) and ("-" in item or "_" in item):
                        continue
                    name = re.sub(r"[_.]+", " ", str(prop_key)).strip().lower()
                    if NOT_A_PROPERTY.search(name) or KIND_KEY.search(name):
                        continue        # (who sells it, what it is called, where it sits)
                    per = counts.setdefault(group, {})
                    per[name] = per.get(name, 0) + 1
                    values.setdefault(group, {}).setdefault(name, set()).add(item.lower())
                    bucket = examples.setdefault(group, {}).setdefault(name, [])
                    if item.lower() not in bucket and len(bucket) < 5:
                        bucket.append(item.lower())
                    taken += 1
                # (a row of ids, titles and sellers is no set of facts, and one listing's options are not several listings)
                if taken >= 2 and not OPTION_KEY.search(key):
                    sets += 1
                    tables[group] = tables.get(group, 0) + 1
            for name, part in value.items():
                if isinstance(part, (dict, list)):
                    visit(part, depth + 1, str(name), group)

        for record in self.records:
            visit(record, 0)

        def lines_for(group: str, limit: int) -> list[str]:
            c, d, s = counts.get(group, {}), values.get(group, {}), examples.get(group, {})
            # (properties whose values differ between listings come first: only they can change the choice)
            ranked = sorted(c, key=lambda k: (len(d.get(k, ())) < 2, -c[k]))[:limit]
            return [f"- {k} (in {c[k]} listings, {len(d.get(k, ()))} different values): " + " | ".join(s.get(k, []))
                    for k in ranked]

        kinds = [g for g in sorted(counts, key=lambda g: -tables.get(g, 0)) if g and tables.get(g, 0) >= 2]
        if len(kinds) >= 2 and self.v.intake.products():
            out = []
            for group in kinds[:4]:
                out.append(f"Listings labelled \"{group}\" ({tables[group]} with details):")
                out += lines_for(group, 14)
            if counts.get(""):
                out.append("Listings without a label:")
                out += lines_for("", 10)
            return "\n".join(out), sets
        merged: dict[str, int] = {}
        for group in counts:
            for name, n in counts[group].items():
                merged[name] = merged.get(name, 0) + n
                values.setdefault("", {}).setdefault(name, set()).update(values[group][name])
                bucket = examples.setdefault("", {}).setdefault(name, [])
                for x in examples[group][name]:
                    if x not in bucket and len(bucket) < 5:
                        bucket.append(x)
        counts[""] = merged
        return "\n".join(lines_for("", 28)), sets

    def referenced(self, item: str) -> list[str]:
        """Records of other listings the request or requirements may point to: listings with at least two distinct
        short values of their own that appear word for word in that text (two at most, best matched first)."""
        v = self.v
        def plain(x: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", x.lower()))

        text = " " + plain(v.request() + " " + v.book.summary()) + " "
        scored: dict[str, tuple[int, str]] = {}

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)
                return
            if not isinstance(value, dict):
                return
            ident = next((x for k, x in value.items() if isinstance(x, str) and ID_KEY.search(str(k)) and id_like(x)), None)
            if ident and ident != item:
                flat: dict[str, Any] = dict(value)
                for x in value.values():
                    if isinstance(x, dict):
                        flat.update(x)
                hits = {plain(x) for x in flat.values() if isinstance(x, str) and len(x.strip()) <= 40 and not id_like(x)
                        and len(plain(x)) >= 3 and f" {plain(x)} " in text}
                if len(hits) >= 2 and len(hits) > scored.get(ident, (0, ""))[0]:
                    scored[ident] = (len(hits), as_json(clip_strings(value, 200))[:1500])
            for x in value.values():
                if isinstance(x, (dict, list)):
                    visit(x, depth + 1)

        for record in self.records:
            visit(record, 0)
        best = sorted(scored.items(), key=lambda kv: -kv[1][0])[:2]
        return [f"{ident}: {shown}" for ident, (_, shown) in best]

    def labels(self) -> str:
        """Short values that recur under one key across records (a category label, say), with counts."""
        tally: dict[str, dict[str, int]] = {}

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)
                return
            if not isinstance(value, dict):
                return
            if len(value) >= 4:
                for key, x in value.items():
                    if isinstance(x, str) and 0 < len(x) <= 40 and not (re.search(r"\d", x) and CODE_FORM.fullmatch(x)):
                        bucket = tally.setdefault(str(key), {})
                        bucket[x] = bucket.get(x, 0) + 1
            for x in value.values():
                if isinstance(x, (dict, list)):
                    visit(x, depth + 1)

        for record in self.records:
            visit(record, 0)
        rows = []
        for key, bucket in tally.items():
            if sum(bucket.values()) >= 4 and 2 <= len(bucket) <= 8 and max(bucket.values()) >= 2:
                rows.append(f"- {key}: " + " | ".join(f"{x} (x{n})" for x, n in sorted(bucket.items(), key=lambda kv: -kv[1])))
        return "\n".join(rows[:12])

    def cheaper_unopened(self) -> list[tuple[str, Any]]:
        """For a cheapest request: result rows priced below the cart's item, in its section, not shown out of stock or
        over the budget, never opened with details (five at most, cheapest first). Which key holds the price, the
        stock flag and the section label is read from the rows themselves: the number most rows carry, the yes/no
        value most rows carry, and the key named like a kind label."""
        v = self.v
        held_ids = v.basket.ids
        if v.plan.get("priority") != "cheapest" or not held_ids:
            return []
        found_rows: list[tuple[str, dict]] = []

        def visit(value: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(value, list):
                for x in value:
                    visit(x, depth + 1)
                return
            if not isinstance(value, dict):
                return
            ident = next((x for k, x in value.items() if isinstance(x, str) and "id" in k.lower() and ID_FORM.match(x)), None)
            if ident:
                found_rows.append((ident, value))
            for x in value.values():
                if isinstance(x, (dict, list)):
                    visit(x, depth + 1)

        for record in self.records:
            visit(record, 0)
        numbers: dict[str, int] = {}
        flags: dict[str, int] = {}
        for _, row in found_rows:
            for k, x in row.items():
                if isinstance(x, bool):
                    flags[k] = flags.get(k, 0) + 1
                elif is_count(x):
                    numbers[k] = numbers.get(k, 0) + 1
        price_key = max(numbers, key=numbers.get, default=None)
        stock_key = max(flags, key=flags.get, default=None)
        kind_key = next((k for _, row in found_rows for k in row if KIND_KEY.search(str(k))), None)
        if price_key is None:
            return []
        rows: dict[str, dict[str, Any]] = {}
        for ident, row in found_rows:
            if is_count(row.get(price_key)):
                rows[ident] = {**rows.get(ident, {}), **{k: x for k, x in row.items() if x is not None}}
        held = rows.get(held_ids[-1])
        if not held:
            return []
        q = v.plan.get("quantity")
        units = q if isinstance(q, int) and q > 0 else 1
        cap = v.book.budget if is_count(v.book.budget) else None
        found = []
        for ident, row in rows.items():
            price = row[price_key]
            if ident in self.opened or ident in held_ids or price >= held[price_key]:
                continue
            if (stock_key and row.get(stock_key) is False) or (cap is not None and price * units > cap):
                continue
            if kind_key and held.get(kind_key) and row.get(kind_key) and row.get(kind_key) != held.get(kind_key):
                continue
            found.append((price, ident))
        return [(ident, price) for price, ident in sorted(found)[:5]]


# ============================================================================ the cart
class Basket:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.lines: dict[str, str] = {}
        self.add_mode: str | None = None       # the choice settings of the first cart call stand for "add"
        self.changed_at = 0
        self.view: Any = None
        self.used = False
        self.checked: set[str] = set()
        self.qty_flagged: set[str] = set()
        self.template: tuple[dict[str, Any], str] | None = None    # the latest adding call and the listing it named

    def book(self, args: dict[str, Any], fields: dict[str, Any], step: int) -> None:
        mode = enum_part(args, fields)
        if self.add_mode is None:
            self.add_mode = mode
        listed = line_ids(args)
        if listed is not None:
            if mode != self.add_mode:
                self.ids = [i for i in self.ids if i not in listed]        # (the same tool taking lines out)
            else:
                self.ids = listed                                          # (a tool that takes the whole basket)
                self.lines = {i: cart_line_text(args, i) for i in listed}
        else:
            for key, value in args.items():
                spec = fields.get(key) if isinstance(fields.get(key), dict) else {}
                if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                    kept = [i for i in self.ids if i != value]
                    self.ids = kept + [value] if mode == self.add_mode else kept
                    if mode == self.add_mode:
                        self.template = (dict(args), value)
        self.changed_at = step


# ============================================================================ second opinions
class Checker:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.memo: dict[str, str] = {}
        self.audited = False
        self.suspects: list[str] = []

    def cart(self) -> str:
        v = self.v
        if not v.basket.ids:
            return ""
        # (several products: one line each, each checked)
        for item in (v.basket.ids[-3:] if v.intake.products() else v.basket.ids[-1:]):
            warning = self.item(item, "ORDER", v.basket.lines.get(item, ""))
            if warning:
                return warning
        return ""

    def item(self, item: str, purpose: str, line: str = "") -> str:
        v = self.v
        if v.pace.time_left() < WIND_DOWN + 20:
            return ""
        key = f"{item}|{line}|{v.book.version}"
        if key in self.memo:
            return self.memo[key]
        records = v.shelf.records_of(item)
        if not records:
            return ""
        prompt = ASSESS_TEXT + "\n\nTHE CUSTOMER'S REQUEST:\n" + v.request()
        prompt += "\n\nREQUIREMENTS IN FORCE (stated, revealed, changed):\n" + v.book.summary()
        prompt += (f"\n\nTHE LISTING UNDER CHECK, BEFORE WE {purpose}: {item}\nITS RECORDS IN THE SHOP (oldest first):\n"
                   + "\n".join(records))
        if line:
            prompt += ("\n\nITS CART LINE (when the line names an option or variant, judge that option: its own values, "
                       "price and stock count): " + line)
        if v.basket.view is not None:
            prompt += "\n\nNEWEST CART VIEW:\n" + as_json(v.basket.view)[:2000]
        pointed = v.shelf.referenced(item)
        if pointed:
            prompt += ("\n\nOTHER LISTINGS THE REQUEST OR REQUIREMENTS MAY REFER TO (records seen in the shop):\n"
                       + "\n".join(pointed))
        recurring = v.shelf.labels()
        if recurring:
            prompt += ("\n\nLABELS RECURRING ON THIS SHOP'S LISTINGS (with counts; the item's category label has to be "
                       "the one fitting the request's wording as a whole, not a neighbouring one):\n" + recurring)
        verdict = v.link.opinion(prompt, 1200)
        v.trace.append({"kind": "verify", "item": item, "result": verdict})
        if not verdict or verdict.get("fits") is not False:
            self.memo[key] = ""
            return ""
        problems = [str(p) for p in verdict.get("problems") or []][:5]
        warning = (f"A separate review says {item} falls short: " + "; ".join(problems)
                   + ". Switch to another fully verified fit unless you can show the review is mistaken.")
        self.memo[key] = warning
        return warning

    def decline(self) -> str:
        """Before a decline: listings a second model thinks may fit after all."""
        v = self.v
        if v.pace.time_left() < WIND_DOWN + 30:
            return ""
        self.audited = True
        picked: list[str] = []
        size = 0
        for record in reversed(v.shelf.records):
            text = as_json(clip_strings(record, 160))
            if size + len(text) > 16000:
                continue
            picked.append(text)
            size += len(text)
        if not picked:
            return ""
        prompt = SWEEP_TEXT + "\n\nTHE CUSTOMER'S REQUEST:\n" + v.request()
        prompt += "\n\nREQUIREMENTS IN FORCE:\n" + v.book.summary()
        prompt += "\n\nLISTING RECORDS SEEN (newest first):\n" + "\n".join(picked)
        verdict = v.link.opinion(prompt, 1200)
        v.trace.append({"kind": "audit", "result": verdict})
        maybe = [str(x) for x in (verdict or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.suspects = maybe
        if not maybe:
            return ""
        why = f" ({verdict.get('reason')})" if verdict.get("reason") else ""
        text = ("A separate review thinks these listings could satisfy everything: " + ", ".join(maybe) + why
                + ". Examine them (details, stock, price) and purchase the one that qualifies; decline only when every one fails.")
        if v.door.order_attempts:
            text += (" Remember that no purchase exists yet: a successful order would have closed the episode, which is "
                     "still open, so declining now leaves the customer empty-handed.")
        return text


# ============================================================================ questions to the shopper
class Inquiry:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.used = 0
        self.budget: int | None = None
        self.asked: list[str] = []
        self.sent_texts: list[str] = []
        self.planning = True
        self.sets_at_last = -1
        self.misses = 0
        self.retry_at = 0

    def limit(self) -> int:
        return self.budget if self.budget is not None else 3

    def args_for(self, text: str) -> dict[str, Any] | None:
        """The message tool's arguments carrying the text: its one free-text argument, nothing else required."""
        v = self.v
        schema = v.schemas.get(v.roles.get("message") or "") or {}
        fields = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        texts = [k for k, s in fields.items() if isinstance(s, dict) and (s.get("type") == "string" or "string" in (s.get("type") or []))
                 and "enum" not in s]
        slot = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not slot or any(k != slot for k in required):
            return None
        return {slot: text}

    def _call(self, call_id: str, text: str) -> dict[str, Any] | None:
        v = self.v
        tool = v.roles.get("message")
        args = self.args_for(text)
        if not args:
            return None
        args, faults = validate(args, v.schemas[tool], "args")
        if faults:
            return None
        return {"id": call_id, "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}

    def ask(self, force: bool = False) -> bool:
        """Plan questions and send them; True when any went out."""
        v = self.v
        tool, limit = v.roles.get("message"), self.limit()
        if not tool or self.used >= limit:
            self.planning = False
            return False
        menu, sets = v.shelf.menu()
        early = bool(v.plan.get("ask_first")) and not self.asked
        if sets < 4 and v.pace.step < 8 and not early and not force:
            return False
        if force:
            early = False
        # (a later round, with questions left or one at a time for several products, waits for more listings or a
        # commit attempt, so it knows the earlier answers and more of the shop)
        if self.asked and not force and not early and not v.book.last_unclear and sets < self.sets_at_last + 2:
            return False
        if not force and sets < self.retry_at:
            return False
        if v.pace.steps_left() <= (6 if force else 10) or v.pace.time_left() < WIND_DOWN + 40:
            self.planning = False
            return False
        many = v.intake.products()
        # (one product: every remaining question at once, sparing the search its steps; several: one at a time)
        count = 1 if many else max(1, limit - self.used)
        proposals = self._plan(count, menu, many)
        if early and sets < 4:
            proposals = proposals[:1]
        proposals = [q for q in proposals if str(q.get("question")).strip().lower() not in self.sent_texts][:count]
        calls, topics = [], []
        for n, entry in enumerate(proposals):
            wording = str(entry.get("question")).strip()
            call = self._call(f"ask_{self.used + n + 1}_{v.pace.step}", wording)
            if call is None:
                continue
            calls.append(call)
            product = str(entry.get("product") or "").strip()
            topics.append(((product + " " if product and many else "") + str(entry.get("property")), wording))
        if not calls:
            # (a plan that gave nothing usable gets one more try, after another listing is opened)
            self.misses += 1
            self.retry_at = sets + 1
            if self.misses >= 2:
                self.planning = False
            return False
        v.chat.append({"role": "assistant", "content": "Questions for the customer: " + "; then ".join(t[0] for t in topics) + ".",
                       "tool_calls": calls})
        v.book.last_unclear = False
        results = v.run_calls(calls, planned=True)
        v.post(results)
        for n, (_, text) in enumerate(results):
            if n < len(topics):
                self.asked.append(f"- {topics[n][0]}: asked \"{topics[n][1]}\" -> {text[:400]}")
        self.sets_at_last = sets
        self.planning = self.used < limit
        return True

    def _plan(self, count: int, menu: str, many: list[tuple[str, int]]) -> list[dict[str, Any]]:
        v = self.v
        prompt = INQUIRY_TEXT.replace("QUESTION_COUNT", str(count)) + "\n\nTHE CUSTOMER'S REQUEST:\n" + v.request()
        prompt += "\n\nREQUIREMENTS UP TO NOW:\n" + v.book.summary()
        if many:
            prompt += ("\n\nTHE REQUEST WANTS SEVERAL DIFFERENT PRODUCTS (say which one each question is about): "
                       + "; ".join(name for name, _ in many))
        if self.asked:
            prompt += "\n\nASKED ALREADY (with the customer's replies):\n" + "\n".join(self.asked)
        if v.plan.get("ask_first"):
            prompt += "\n\nTO BE LEARNED FROM THE CUSTOMER FIRST: " + "; ".join(str(h) for h in v.plan.get("ask_first") or [])
        prompt += ("\n\nPROPERTIES IN THE LISTINGS' DETAILS SO FAR (name, number of listings, example values):\n"
                   + (menu or "(none seen yet)"))
        answer = v.link.opinion(prompt, 2000, [LIGHT, ASSESSOR])
        return [q for q in (answer or {}).get("questions") or [] if isinstance(q, dict) and str(q.get("question") or "").strip()]

    def reply(self, text: str) -> bool:
        """Send the answer the customer is owed; True when it went out."""
        v = self.v
        if (not v.roles.get("message") or self.used >= self.limit() or v.pace.steps_left() <= 4
                or v.pace.time_left() < WIND_DOWN):
            return False
        call = self._call(f"reply_{v.pace.step}", text)
        if call is None:
            return False
        v.chat.append({"role": "assistant", "content": "Answering the customer's question.", "tool_calls": [call]})
        v.post(v.run_calls([call], planned=True))
        return True


def once_per(seen: set[str], tag: str) -> bool:
    """True the first time a tag is seen."""
    if tag in seen:
        return False
    seen.add(tag)
    return True


# ============================================================================ the screens a call passes
_SKIP = "Not run (no step used): "
_HOLD = "Not run yet (no step used). "
ORDER_CHECKLIST = (_HOLD + "A final review before buying. Are the cart's lines exactly what the request calls for (usually "
                   "a single line, the chosen item at the wanted quantity) with nothing extra? Has this item's stock and "
                   "price been looked up again, with a tool that shows them, after the most recent cart change, and is it "
                   "available with price times quantity inside today's budget? Do its own properties satisfy each stated "
                   "requirement, each need the customer disclosed and each announced change? If the cheapest or the best "
                   "was requested, is there truly nothing better that fits among everything found? Have all procedural "
                   "rules of the request been followed? Settle any open point now; if none remains, repeat the order call.")
DECLINE_CHECKLIST = (_HOLD + "Giving up is final, so confirm first: the searching tried several routes (different words, "
                     "filters, further pages), every candidate's own properties were read, and each candidate violates a "
                     "stated requirement, a disclosed need, the budget or stock. Close synonyms and every key spelling "
                     "count as matches, and a requirement merely inferred from the story is not one. Buy anything that "
                     "fits; if nothing does, send the decline again.")


class Door:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checks = 0
        self.cheaper_checks = 0
        self.research_checks = 0
        self.early_bounced = False
        self.refused_at = -1
        self.searched_at = -1
        self.recent: list[str] = []
        self.capped: set[str] = set()
        self.reread_tags: set[str] = set()

    def offered(self) -> list[dict[str, Any]]:
        """The tools on offer this turn."""
        v = self.v
        left = v.pace.steps_left()
        order, decline, cart = v.roles.get("order"), v.roles.get("decline"), v.roles.get("cart")
        names: set | None = None
        if left <= 1:
            names = {order, decline}
        elif left <= 3:
            names = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            names = {order, decline, cart, v.roles.get("cart_view"), v.roles.get("view")}
        tools = v.tools
        if names is not None and order and cart:
            # (a tool no role covers that finishes a checkout, a confirm step after the order tool, stays on offer)
            assigned = set(v.roles.values())
            names |= {t["function"]["name"] for t in v.tools
                      if t["function"]["name"] not in assigned and CHECKOUT_STEP.search(t["function"]["name"])}
            narrowed = [t for t in v.tools if t["function"]["name"] in names]
            if narrowed:
                tools = narrowed
        if v.inquiry.planning and v.roles.get("message"):
            tools = [t for t in tools if t["function"]["name"] != v.roles.get("message")]
        if self.capped:
            tools = [t for t in tools if t["function"]["name"] not in self.capped] or tools
        return tools

    def units_issue(self, name: str, args: dict[str, Any]) -> str:
        v = self.v
        wanted = v.plan.get("quantity")
        if not isinstance(wanted, int) or isinstance(wanted, bool) or wanted < 1:
            return ""
        fields = v.schemas[name].get("properties") or {}
        if v.basket.add_mode is not None and enum_part(args, fields) != v.basket.add_mode:
            return ""
        many = v.intake.products()
        for key, spec in fields.items():
            if not (isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec):
                continue
            asked = args.get(key, spec.get("default", 1))
            if many:
                if isinstance(asked, int) and asked not in {q for _, q in many}:
                    return (f"the request wants these quantities: {', '.join(f'{n} x{q}' for n, q in many)}, yet this "
                            f"call asks for {asked}.")
                return ""
            if isinstance(asked, int) and asked != wanted:
                return f"the request wants {wanted} of one listing, yet this call asks for {asked}."
        return ""

    def screen(self, calls: list[dict[str, Any]], planned: bool, out: list[tuple[str, str]]) -> list[tuple[str, str, dict]]:
        """Each call through the screens in order; the ones that pass, ready to send."""
        v = self.v
        allowed = {t["function"]["name"] for t in self.offered()}
        ends = v.ends()
        speaker, cart = v.roles.get("message"), v.roles.get("cart")
        queued_messages = 0
        passed: list[tuple[str, str, dict[str, Any]]] = []
        for call in calls:
            key = call["id"]
            fn = call.get("function") or {}
            name, raw = fn.get("name"), fn.get("arguments")
            reason, args = self._reason(name, raw, planned, allowed, ends, speaker, cart, len(calls), queued_messages)
            if reason is not None:
                out.append((key, reason))
                continue
            if name == speaker:
                queued_messages += 1
            passed.append((key, name, args))
        return passed

    def _reason(self, name: Any, raw: Any, planned: bool, allowed: set, ends: set, speaker: Any, cart: Any,
                count: int, queued: int) -> tuple[str | None, dict[str, Any]]:
        v = self.v
        if name not in v.schemas:
            return _SKIP + f"'{name}' is no tool of this shop. Use one of: {', '.join(sorted(allowed))}.", {}
        if not planned and name not in allowed:
            if name == speaker and v.inquiry.planning:
                why = "questions to the customer are planned and sent for you once you have opened a few listings' details"
            else:
                why = "too few steps are left for this; only these tools work now: " + ", ".join(sorted(allowed))
            return _SKIP + why + ".", {}
        if isinstance(raw, dict):
            args = raw
        elif (raw or "").strip():
            args = read_object(raw or "{}")
            if args is None:
                return _SKIP + "the arguments did not form a valid JSON object.", {}
        else:
            args = {}
        args, faults = validate(args, v.schemas[name], "args")
        if faults:
            return _SKIP + "the arguments break the schema: " + "; ".join(faults[:6]), args
        if name in ends and count > 1:
            return _SKIP + "an order or a decline goes out alone, once you have seen the results of your other calls.", args
        text = joined_strings(args)
        if (name == speaker and not planned and "?" not in text
                and not any(INFORM_RULE.search(str(rule)) for rule in v.plan.get("rules") or [])):
            return (_SKIP + "this message asks nothing, and messages to the customer exist for questions alone. Findings, "
                    "choices and plans are not reported to the customer; the finishing tool conveys them."), args
        if name == speaker and text.strip().lower() in v.inquiry.sent_texts:
            return _SKIP + "this very message went out before, and its answer is above. Ask about something else, or carry on.", args
        if name == speaker and v.inquiry.budget is not None and v.inquiry.used + queued >= v.inquiry.budget:
            return _SKIP + "the message budget is spent. Do not write to the customer again.", args
        fingerprint = name + as_json(args)
        if name not in ends and name != speaker and name != cart and fingerprint in self.recent:
            return (_SKIP + "this identical call was just made and nothing has moved since; the result is already above. "
                    "Alter the arguments or take another action."), args
        if name == cart:
            held = self._cart_review(name, args)
            if held:
                return held, args
            issue = self.units_issue(name, args)
            if issue and fingerprint not in v.basket.qty_flagged:
                v.basket.qty_flagged.add(fingerprint)
                return _SKIP + issue + " If you are sure, send the same call again.", args
        return None, args

    def _cart_review(self, name: str, args: dict[str, Any]) -> str:
        """A one-listing add is checked once per requirements version (several products: at the order instead)."""
        v = self.v
        if v.pace.steps_left() <= 3 or v.intake.products():
            return ""
        fields = v.schemas[name].get("properties") or {}
        if v.basket.add_mode is not None and enum_part(args, fields) != v.basket.add_mode:
            return ""
        target = free_token(args, fields)
        if target is None and line_ids(args):
            target = line_ids(args)[-1]
        if not target or f"{target}|{v.book.version}" in v.basket.checked:
            return ""
        v.basket.checked.add(f"{target}|{v.book.version}")
        warning = v.checker.item(target, "ADD TO THE CART", cart_line_text(args, target))
        return ("Not run (no step used). " + warning + " If you are sure it fits, send the same call again.") if warning else ""

    def review_finish(self, passed: list[tuple[str, str, dict]], out: list[tuple[str, str]]) -> bool:
        """The checks an order or a decline goes through when it is the first call left; True when it was held."""
        v = self.v
        if not passed or passed[0][1] not in v.ends():
            return False
        key, name, args = passed[0]
        declining = v.closer.is_decline(name, args)
        buying = not declining and name == v.roles.get("order")
        roomy = v.pace.steps_left() > 10 and v.pace.time_left() > WIND_DOWN + 20
        if (buying and v.plan.get("priority") == "cheapest" and self.refused_at > self.searched_at
                and self.research_checks < 1 and roomy):
            # (after a refused order a backup picked earlier may no longer be the cheapest fit: search once more)
            self.research_checks += 1
            out.append((key, _HOLD + "Your previous order was refused and the cheapest fitting item is wanted. A backup "
                        "chosen before the refusal only qualifies if nothing cheaper fits today, so run a fresh search "
                        "first: the required words plus the category, a price ceiling a little above the refused item's "
                        "price when the search allows one, and the following pages. Open the cheaper candidates' details, "
                        "move the cheapest full fit into the cart, then order."))
            return True
        if buying and self.cheaper_checks < 2 and roomy:
            cheaper = v.shelf.cheaper_unopened()
            if cheaper:
                self.cheaper_checks += 1
                out.append((key, _HOLD + "The cheapest fitting item is wanted, yet these lower-priced listings from your "
                            "search results never had their details opened: "
                            + ", ".join(f"{i} ({p})" for i, p in cheaper) + ". Open them (bundled in one reply); swap the "
                            "cheapest one that satisfies every requirement into the cart, or, if none does, order again."))
                return True
        stale = self._unread_since_cart_change() if buying and v.pace.steps_left() > 3 else []
        if stale and once_per(self.reread_tags, f"{v.basket.changed_at}|{self.refusals}"):
            out.append((key, _HOLD + "Stock and price can change at any moment, and these cart items have not been read "
                        "since the latest cart change: " + ", ".join(stale) + ". Look them up now with a tool that shows "
                        "stock and price (several in one reply), then order if they are still available and within budget."))
            return True
        if buying and not self.order_checked and v.pace.steps_left() > 2 and v.pace.time_left() > WIND_DOWN:
            self.order_checked = True
            warning = v.checker.cart()
            out.append((key, _HOLD + warning if warning else ORDER_CHECKLIST))
            return True
        if not declining:
            return False
        # (a decline is rarely right: each of the first two attempts first gets an independent look)
        if self.decline_checks < 2 and v.pace.steps_left() > 4 and v.pace.time_left() > WIND_DOWN + 30:
            self.decline_checks += 1
            found = v.checker.decline()
            if found:
                out.append((key, _HOLD + found))
                return True
            if self.decline_checks == 1:
                out.append((key, DECLINE_CHECKLIST))
                return True
        bought = v.closer.buy_instead()
        if bought:
            out.append((key, "Not run: the decline was replaced. " + bought))
            return True
        passed[0] = (key, name, v.closer.clean(args, name))
        return False

    def _unread_since_cart_change(self) -> list[str]:
        v = self.v
        held = v.basket.ids if v.intake.products() else v.basket.ids[-1:]
        return [x for x in held if v.shelf.read_at.get(x, -1) < v.basket.changed_at]

    def spare_last_step(self, passed: list[tuple[str, str, dict]], out: list[tuple[str, str]]) -> list[tuple[str, str, dict]]:
        """Calls that would leave no step for the finish are not run."""
        ends = self.v.ends()
        if not passed or passed[0][1] in ends or not ends:
            return passed
        keep = self.v.pace.steps_left() - 1
        if keep >= len(passed):
            return passed
        for key, _, _ in passed[max(0, keep):]:
            out.append((key, _SKIP + "a single step is left, and it belongs to the order (or the decline)."))
        return passed[: max(0, keep)]


# ============================================================================ sending and booking
ORDER_STILL_OPEN = ("STILL OPEN: the episode did not end with that order call, so no purchase exists yet, however its result "
                    "reads (a successful order closes the episode). Should the result call for a further step (confirming "
                    "with a token or code it supplied, for example), do it now. Should it have been refused or have failed, "
                    "find out why, put every stated change into effect, repair the cart, look up the pick's stock and price "
                    "afresh and try the order again; a new price on its own leaves an item acceptable as long as price times "
                    "quantity is inside the budget.")
ORDER_WAITS = ("Checkout is half done: the result supplies a token or wants a confirmation. Complete it now with the "
               "tool that closes the checkout; no purchase exists until the episode ends.")


class Clerk:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit

    def send(self, passed: list[tuple[str, str, dict]], out: list[tuple[str, str]]) -> None:
        v = self.v
        for start in range(0, len(passed), 16):
            chunk = passed[start: start + 16]
            if v.finished or v.dead or v.pace.steps_left() <= 0:
                out.extend((key, "Not run: the episode has ended or no steps are left.") for key, _, _ in chunk)
                continue
            chunk = chunk[: v.pace.steps_left()]
            actions = [{"name": name, "args": args} for _, name, args in chunk]
            first = v.pace.step + 1
            answer = v.link.act(actions)
            v.trace.append({"kind": "env", "step": first, "actions": actions, "reply": answer})
            if answer is None:
                out.extend((key, "The call did not get through (network or server). Try again.") for key, _, _ in chunk)
                continue
            observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
            for n, (key, name, args) in enumerate(chunk):
                if n >= len(observations):
                    out.append((key, "Not run: the episode ended before this call."))
                    continue
                body, note = self.book(name, args, observations[n], bool(answer.get("finished")))
                if n == len(observations) - 1 and answer.get("finished"):
                    body += " [the episode is finished]"
                out.append((key, body + ("\n[note] " + note if note else "")))

    def book(self, name: str, args: dict[str, Any], observation: Any, batch_finished: bool) -> tuple[str, str]:
        """Bookkeeping for one result; its text and the note for the pilot."""
        v = self.v
        door, speaker, cart = v.door, v.roles.get("message"), v.roles.get("cart")
        if name == speaker:
            v.inquiry.used += 1
            sent = joined_strings(args).strip().lower()
            if sent and not (isinstance(observation, dict) and observation.get("error")):
                v.inquiry.sent_texts.append(sent)          # (a message that failed may go out again)
        if name == v.roles.get("order"):
            door.order_attempts += 1
        if name == cart:
            v.basket.used = True
            door.order_checked = False
        if name in (cart, speaker) or name in v.ends():
            door.recent = []
        else:
            door.recent = (door.recent + [name + as_json(args)])[-2:]
        v.shelf.records.append(observation)
        if name == cart:
            v.basket.book(args, v.schemas[name].get("properties") or {}, v.pace.step)
        if name == v.roles.get("search"):
            door.searched_at = v.pace.step
        if name in (v.roles.get("details"), v.roles.get("view")):
            v.shelf.opened.update(x for x in args.values() if isinstance(x, str))
        if name == v.roles.get("cart_view"):
            v.basket.view = observation
        if name in (v.roles.get("details"), v.roles.get("view"), v.roles.get("cart_view")):
            named: set[str] = set()
            gather_ids(observation, named)
            v.shelf.read_at.update(dict.fromkeys(named, v.pace.step))
        note = self.remark(name, observation)
        note = self._order_followup(name, observation, batch_finished, note)
        lines, question = self.customer_lines(name, args, observation)
        if lines:
            meaning = v.book.hear(lines, question)
            if meaning:
                note = f"{note} {meaning}" if note else meaning
        return as_json(observation), note

    def _order_followup(self, name: str, observation: Any, batch_finished: bool, note: str) -> str:
        v, door = self.v, self.v.door
        if name != v.roles.get("order") or name == v.roles.get("decline") or batch_finished:
            return note
        refused = isinstance(observation, dict) and "refusal" in observation
        waiting = isinstance(observation, dict) and awaits_next_step(observation) and not refused
        if waiting:
            return (note + " " if note else "") + ORDER_WAITS
        if refused:
            return note
        # (the order did not end the episode: refused in other words, failed, or waiting for another step)
        door.refusals += 1
        door.refused_at = v.pace.step
        door.order_checked = False
        return (note + " " if note else "") + ORDER_STILL_OPEN

    def remark(self, name: str, observation: Any) -> str:
        """What a result means for the pilot: an allowance spent, a refused order, a customer line."""
        v = self.v
        if isinstance(observation, list):
            gather_ids(observation, v.shelf.ids)
            return ""
        if not isinstance(observation, dict):
            return ""
        gather_ids(observation, v.shelf.ids)
        notes = []
        problem = " ".join(str(observation.get(k) or "") for k in ("error", "detail", "message", "reason")
                           if isinstance(observation.get(k), str))
        core = {v.roles.get(r) for r in ("order", "decline", "cart", "cart_view")} - {None}
        if observation.get("error") and SPENT.search(problem) and name not in core:
            if name == v.roles.get("message"):
                iq = v.inquiry
                iq.budget = iq.used if iq.budget is None else min(iq.budget, iq.used)
            v.door.capped.add(name)
            notes.append("No allowance is left for this tool, so it has been withdrawn; stop calling it. Take prices and "
                         "stock from search rows or the cart view instead and decide on what you already know.")
        if "refusal" in observation:
            door = v.door
            door.refusals += 1
            door.refused_at = v.pace.step
            door.order_checked = False
            notes.append("REFUSED: no purchase took place, the episode continues and the cart is untouched. Put each "
                         "change the refusal mentions into effect, repair the cart (drop whatever line stopped qualifying, "
                         "a sold-out item for example), look up the replacement's stock and price afresh, then order again.")
            refusal = observation.get("refusal")
            said = refusal if isinstance(refusal, str) else as_json(refusal)
            if PRICE_TALK.search(said) and not SOLD_OUT.search(said):
                notes.append("Nothing in this refusal says the item has gone: if all it reports is a new price (or a change "
                             "the item still satisfies), the item can still be bought. Look up its present price; while "
                             "price times quantity stays inside the budget and every requirement still holds, order it "
                             "again straight away. A price change is never a reason to decline.")
        elif len(observation) >= 2 and name != v.roles.get("message"):
            lines: list[str] = []
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                    gather_sentences(value, lines)
            if lines:
                notes.append("The customer speaks in this result (something changed); act on it: " + " | ".join(lines[:4]))
        return " ".join(notes)

    def customer_lines(self, name: str, args: dict[str, Any], observation: Any) -> tuple[list[str], str | None]:
        """The customer's words in a result, and the question they answer (for a message)."""
        v = self.v
        heard: list[str] = []
        if name == v.roles.get("message"):
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, (dict, list)) and not (isinstance(observation, dict) and "error" in observation):
                if isinstance(observation, dict):
                    gather_sentences([x for x in observation.values() if isinstance(x, (str, list))], heard)
                if not heard and holds_words(observation):
                    heard = [as_json(observation)[:600]]     # (a short or structured answer is an answer too)
            return heard, joined_strings(args)
        if isinstance(observation, dict):
            wrapper = len(observation) <= 3 and any(isinstance(x, (dict, list)) for x in observation.values())
            for value in observation.values():
                if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                    gather_sentences(value, heard)
                elif isinstance(value, str) and len(value.split()) >= 4 and "refusal" in observation:
                    heard.append(value)
                elif (isinstance(value, str) and wrapper and len(value.split()) >= 6
                      and (SPEAKS_AS_SELF.search(value) or name == v.roles.get("order"))):
                    heard.append(value)     # (a line wrapped round the real result, as a plain string)
        return heard, None


# ============================================================================ finishing calls built by the code
class Closer:
    def __init__(self, visit: "Visit") -> None:
        self.v = visit
        self.probed: set[str] = set()
        self.bought_once = False

    def learn(self, name: str) -> None:
        """Once per tool, when one tool does both finishes and the first reading named no values for it: a model reads
        the tool's schema and names the values that buy and that decline; they are merged into the reading."""
        v = self.v
        if name in self.probed or v.roles.get("order") != name or v.roles.get("decline") != name:
            return
        self.probed.add(name)
        known = v.plan.get("finish_values") if isinstance(v.plan.get("finish_values"), dict) else {}
        if all(isinstance(known.get(job), dict) and known[job] for job in ("order", "decline")):
            return
        fn = next((t["function"] for t in v.tools if t["function"]["name"] == name), {})
        prompt = (ENDER_TEXT + "\n\nTHE TOOL:\n" + as_json({"name": name, "description": fn.get("description", ""),
                                                            "parameters": fn.get("parameters") or {}})
                  + "\n\nJOBS THIS TOOL DOES: order, decline")
        answer = v.link.opinion(prompt, 300, [ASSESSOR, PILOT, STANDBY])
        if not isinstance(answer, dict):
            return
        for job in ("order", "decline"):
            if isinstance(answer.get(job), dict) and not (isinstance(known.get(job), dict) and known[job]):
                known[job] = dict(answer[job])
        v.plan["finish_values"] = known

    def chosen_value(self, purpose: str, key: str, options: list) -> Any:
        """The allowed value the first reading named for this choice argument and purpose (order / decline), or None."""
        named = self.v.plan.get("finish_values")
        named = named.get(purpose) if isinstance(named, dict) else None
        value = named.get(key) if isinstance(named, dict) else None
        return value if value in options else None

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        """Whether a finishing call walks away. One tool may do both finishes, told apart by a choice value: the values
        the first reading named decide; the wording of the value is only a fallback."""
        v = self.v
        if name != v.roles.get("decline"):
            return False
        if v.roles.get("order") != name:
            return True
        self.learn(name)
        fields = (v.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            options = (fields.get(key) or {}).get("enum")
            if not isinstance(value, str) or not options:
                continue
            quits, buys = self.chosen_value("decline", key, options), self.chosen_value("order", key, options)
            if quits is not None and quits != buys:
                return value == quits
        return any(isinstance(x, str) and "enum" in (fields.get(k) or {}) and NO_PURCHASE.search(x) for k, x in args.items())

    def buy_instead(self) -> str:
        """One product requested and a decline about to go through: if the decline audit named listings and one of them
        passes the separate pre-order check on the records seen, the code puts it in the cart (shaped like the agent's
        own earlier adding call) and orders it. '' when that cannot be done; the decline then stands."""
        v = self.v
        cart, order = v.roles.get("cart"), v.roles.get("order")
        if (self.bought_once or not cart or not order or v.intake.products()
                or v.pace.steps_left() < 2 or v.pace.time_left() < WIND_DOWN + 30):
            return ""
        self.bought_once = True
        pick = next((x for x in v.checker.suspects if x in v.shelf.ids and v.shelf.records_of(x)
                     and not v.checker.item(x, "ORDER")), None)
        if pick is None:
            return ""
        fields = v.schemas[cart].get("properties") or {}
        q = v.plan.get("quantity")
        units = q if isinstance(q, int) and not isinstance(q, bool) and q >= 1 else 1
        template, named = v.basket.template or self._schema_add(cart, units)
        if not template:
            return ""
        add = {}
        for k, x in template.items():
            spec = fields.get(k) if isinstance(fields.get(k), dict) else {}
            if x == named:
                add[k] = pick
            elif isinstance(x, int) and not isinstance(x, bool) and spec.get("type") == "integer" and "enum" not in spec:
                add[k] = units
            else:
                add[k] = x
        add, faults = validate(add, v.schemas[cart], "args")
        if faults:
            return ""
        said: list[tuple[str, str]] = []
        if any(x != pick for x in v.basket.ids):
            return ""            # (other lines in the cart: the decline is left to the agent's judgement)
        if v.basket.ids != [pick]:
            v.clerk.send([("code_add", cart, add)], said)
            if v.finished or v.basket.ids != [pick]:
                return f"The code put {pick} in the cart; handle this result before ordering: " + " | ".join(t for _, t in said)
        closing, faults = validate(self.plain_args(order, CHECKOUT_REMARK), v.schemas[order], "args")
        if faults:
            return ""
        v.clerk.send([("code_order", order, closing)], said)
        v.trace.append({"kind": "decline_replaced", "pick": pick, "suspects": v.checker.suspects})
        return (f"{pick} was named by the independent review and passes the pre-order check, so the code ordered it "
                "instead of giving up. Shop replies: " + " | ".join(t for _, t in said))

    def _schema_add(self, cart: str, units: int) -> tuple[dict[str, Any], str]:
        """An adding call built from the cart tool's schema when the agent made none: its one free-text argument takes
        the listing id, a whole-number argument the units; any other required argument means it cannot be built."""
        schema = self.v.schemas.get(cart) or {}
        fields = schema.get("properties") or {}
        texts = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "string" and "enum" not in s]
        counts = [k for k, s in fields.items() if isinstance(s, dict) and s.get("type") == "integer" and "enum" not in s]
        if len(texts) != 1 or len(counts) > 1:
            return {}, ""
        built = {texts[0]: "\x00listing"}
        if counts:
            built[counts[0]] = units
        if any(k not in built for k in schema.get("required") or []):
            return {}, ""
        return built, "\x00listing"

    def plain_args(self, name: str, text: str, declining: bool = False) -> dict[str, Any]:
        self.learn(name)
        schema = self.v.schemas.get(name) or {}
        fields = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = fields.get(key) or {}
            if "enum" in spec and spec["enum"]:
                # (the mode matching the intent: the value the first reading named, else one whose wording fits)
                named = self.chosen_value("decline" if declining else "order", key, list(spec["enum"]))
                fitting = [x for x in spec["enum"] if isinstance(x, str) and bool(NO_PURCHASE.search(x)) == declining]
                args[key] = named if named is not None else (fitting or spec["enum"])[0]
            elif spec.get("type") == "string" or "string" in (spec.get("type") or []):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        if len(fields) == 1:
            for key, spec in fields.items():
                if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    args[key] = text[: spec.get("maxLength", len(text))]
        return args

    def clean(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        """A decline that may name nothing: every free text becomes the plain remark."""
        v = self.v
        rule = v.plan.get("decline_no_names")
        if rule is False:
            return args
        names_one = any(isinstance(x, str) and any(t and t in v.shelf.ids for t in TEXT_BREAKS.split(x)) for x in args.values())
        if not (rule or names_one):
            return args
        fields = (v.schemas.get(name) or {}).get("properties") or {}
        cleaned = {}
        for key, value in args.items():
            spec = fields.get(key) or {}
            if isinstance(value, str) and "enum" not in spec:
                cleaned[key] = NO_ORDER_REMARK[: spec.get("maxLength", len(NO_ORDER_REMARK))]
            else:
                cleaned[key] = value
        return cleaned

    def finish(self) -> None:
        """Time or patience ran out: order the cart if it was used, else decline; a price-only refusal is ordered once
        more, and an order still not through is followed by a decline."""
        v = self.v
        if v.finished or v.dead or v.pace.steps_left() <= 0 or v.pace.time_left() < 2:
            return
        order, decline = v.roles.get("order"), v.roles.get("decline")
        if order and v.basket.used:
            action = {"name": order, "args": self.plain_args(order, CHECKOUT_REMARK)}
        elif decline:
            action = {"name": decline, "args": self.plain_args(decline, NO_ORDER_REMARK, declining=True)}
        else:
            return
        answer = v.link.act([action])
        v.trace.append({"kind": "forced_finish", "action": action, "reply": answer})
        seen = (answer or {}).get("observation") if isinstance((answer or {}).get("observation"), dict) else {}
        parts = [x for x in seen.values() if isinstance(x, (list, str))]
        parts += [y for x in seen.values() if isinstance(x, dict) for y in x.values() if isinstance(y, (list, str))]
        lines: list[str] = []
        gather_sentences(parts, lines)
        joined = " ".join(lines)
        if (answer and not answer.get("finished") and lines and order and action["name"] == order and v.basket.used
                and PRICE_TALK.search(joined) and not SOLD_OUT.search(joined) and v.pace.steps_left() > 0):
            # (refused only over a price: the cart still holds a good item, so it is ordered once more)
            answer = v.link.act([action])
            v.trace.append({"kind": "forced_finish", "action": action, "reply": answer})
        if (answer and not answer.get("finished") and isinstance(answer.get("observation"), dict) and lines and decline
                and v.pace.steps_left() > 0):
            action = {"name": decline, "args": self.plain_args(decline, NO_ORDER_REMARK, declining=True)}
            answer = v.link.act([action])
            v.trace.append({"kind": "forced_finish", "action": action, "reply": answer})


# ============================================================================ the hub
EARLY_COMMIT = ("Not run (no step used): it is too soon to choose. Open the details of two or three further listings of the "
                "requested kind first (bundle them in one reply and prefer ones that differ from what you have seen); the "
                "customer's questions are planned from what those pages show.")


class Visit:
    def __init__(self, problem_data: dict[str, Any]) -> None:
        env = problem_data["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}}
                        for t in self.tools}
        self.finished = bool(self.policy.get("finished"))
        self.dead = False
        self.trace: list[dict[str, Any]] = []
        self.chat: list[dict[str, Any]] = []
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.pace = Pacer(int(self.policy.get("max_steps") or 30))
        self.link = Link(self)
        self.intake = Intake(self)
        self.book = Notebook(self)
        self.shelf = Shelf(self)
        self.basket = Basket()
        self.checker = Checker(self)
        self.inquiry = Inquiry(self)
        self.door = Door(self)
        self.clerk = Clerk(self)
        self.closer = Closer(self)

    def request(self) -> str:
        return str(self.policy.get("query") or "")

    def ends(self) -> set[str]:
        return {self.roles.get("order"), self.roles.get("decline")} - {None}

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        passed = self.door.screen(calls, planned, out)
        if self.door.review_finish(passed, out):
            return out
        passed = self.door.spare_last_step(passed, out)
        self.clerk.send(passed, out)
        position = {c["id"]: n for n, c in enumerate(calls)}
        out.sort(key=lambda pair: position.get(pair[0], 0))
        return out

    def status(self) -> str:
        pace, door, iq = self.pace, self.door, self.inquiry
        left = pace.steps_left()
        messages = f"messages sent {iq.used}" + ("" if iq.budget is None else f" of {iq.budget}")
        text = (f"[status] {pace.step} of {pace.cap} steps used, {left} left; {messages}; "
                f"roughly {int(max(0, pace.time_left() - HARD_STOP))} s left.")
        text += "\n[requirements now] " + self.book.summary()
        if door.refusals and not self.finished and pace.step - door.refused_at >= 2:
            # (a refused order is easily taken for a placed one; until the episode ends nothing is bought)
            text += (f" NO PURCHASE YET: the episode survived the order attempt made {pace.step - door.refused_at} steps "
                     "ago. With fitting items inside the current budget in the cart, call the order tool now; otherwise "
                     "repair the cart and then order (declining only when nothing in the shop fits any longer).")
        if left <= 2:
            text += " LAST STEPS: buy now if a verified fit sits in the cart; if not, decline."
        elif left <= 5 or pace.time_left() < WIND_DOWN:
            text += (" WRAP UP: reduce the cart to the verified line(s) the request calls for, recheck stock, and order "
                     "(decline when nothing fits).")
        elif door.order_attempts == 0 and left <= 14:
            text += (" Few steps remain: cart your strongest verified fit and order within the next few steps; search and"
                     " detail tools close at 7 steps left, and recovering from a refusal costs roughly 4 steps.")
        return text

    def post(self, results: list[tuple[str, str]]) -> None:
        last = len(results) - 1
        for n, (key, text) in enumerate(results):
            self.chat.append({"role": "tool", "tool_call_id": key, "content": text + ("\n" + self.status() if n == last else "")})

    def _bounce_early(self, said: str, calls: list[dict[str, Any]]) -> bool:
        """A commit before enough listings were opened is sent back once: the questions are planned from them."""
        if (self.door.early_bounced or self.shelf.menu()[1] >= 3 or self.pace.steps_left() <= 12
                or self.pace.time_left() <= WIND_DOWN + 60):
            return False
        self.door.early_bounced = True
        self.chat.append({"role": "assistant", "content": said or None, "tool_calls": calls})
        self.post([(c["id"], EARLY_COMMIT) for c in calls])
        return True

    def run(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.intake.read()
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        opening = self.policy.get("opening")
        first = "SHOPPER REQUEST:\n" + self.request()
        if opening is not None:
            first += "\n\nOPENING OBSERVATION:\n" + as_json(opening)
        brief = self.intake.brief()
        if brief:
            first += "\n\n" + brief
        first += "\n\n" + self.status()
        if opening is not None:
            self.shelf.records.append(opening)
            gather_ids(opening, self.shelf.ids)
        self.chat = [{"role": "system", "content": PILOT_TEXT}, {"role": "user", "content": first}]
        quiet = misses = numbered = 0
        iq = self.inquiry
        for _ in range(TURN_LIMIT):
            if self.finished or self.dead or self.pace.steps_left() <= 0:
                break
            if self.pace.time_left() < HARD_STOP:
                self.closer.finish()
                break
            if self.book.owed:
                owed, self.book.owed = self.book.owed, ""
                if iq.reply(owed):
                    continue
            if iq.planning and iq.ask():
                continue
            said_msg = self.link.talk(self.chat, self.door.offered(), [PILOT, STANDBY, PILOT], 2500)
            if said_msg is None:
                misses += 1
                if misses >= 2:
                    self.closer.finish()
                    break
                continue
            calls = []
            for raw in said_msg.get("tool_calls") or []:
                if not isinstance(raw, dict):
                    continue
                numbered += 1
                fn = raw.get("function") or {}
                arguments = fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {})
                calls.append({"id": raw.get("id") or f"call_{numbered}", "type": "function",
                              "function": {"name": fn.get("name"), "arguments": arguments}})
            said = said_msg.get("content") or ""
            if not isinstance(said, str):
                said = as_json(said)
            committing = {self.roles.get("cart"), self.roles.get("order"), self.roles.get("decline")} - {None}
            if iq.planning and any(c["function"]["name"] in committing for c in calls):
                if self._bounce_early(said, calls):
                    continue
                if iq.ask(force=True):
                    continue
            if not calls:
                quiet += 1
                self.chat.append({"role": "assistant", "content": said or "(no tool call)"})
                self.chat.append({"role": "user", "content": "A reply without a tool call does nothing; call one of the tools you were given and go on.\n"
                                  + self.status()})
                if quiet >= 3:
                    self.closer.finish()
                    break
                continue
            quiet = 0
            self.trace.append({"kind": "model", "content": said[:2000], "calls": [c["function"] for c in calls]})
            self.chat.append({"role": "assistant", "content": said or None, "tool_calls": calls})
            self.post(self.run_calls(calls))
        if not self.finished and not self.dead:
            self.closer.finish()
        self.trace.append({"kind": "end", "finished": self.finished, "steps": self.pace.step,
                           "cost": round(self.link.cost, 5), "seconds": round(self.pace.used, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    visit = Visit(problem_data)
    try:
        return visit.run()
    except Exception as exc:
        visit.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not visit.finished and not visit.dead:
                visit.closer.finish()
        except Exception as again:
            visit.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return visit.trace


__all__ = ["agent_main"]
